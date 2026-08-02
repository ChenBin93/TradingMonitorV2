"""多空力量识别 — 用户方法论核心模块

在主要周期及细节周期观察近期多空力量对比:
  1. 前面一些K线是多头强势还是空头强势
  2. 占据优势的一方是越来越强还是越来越弱
  3. 预测后续行情发展, 为机会点提前准备

评分: -100(空极强) ~ +100(多极强)
变化: strengthening(增强) / weakening(减弱) / stable(持平)
"""
import numpy as np
import pandas as pd

POWER_WINDOW = 30      # 力量统计窗口 (近30根)
SHIFT_WINDOW = 10      # 变化对比窗口 (后10 vs 前10)


def _atr(df: pd.DataFrame) -> float:
    """ATR (近14根均值) — 全向量化"""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    if n < 2:
        return 0.0
    tr = np.zeros(n)
    tr[1:] = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-14:]))


def score_from_components(bull_body, bear_body, bull_wick, bear_wick,
                          bull_vol, bear_vol, atr, h5, l5, h6, l6) -> float:
    """力量评分核心 — 纯函数 (analyze_power 与预计算调优共用)"""
    if atr <= 0:
        return 0.0
    net_body = (bull_body - bear_body) / atr
    s_body = float(np.clip(net_body * 12, -100, 100))
    net_wick = (bull_wick - bear_wick) / atr
    s_wick = float(np.clip(net_wick * 15, -100, 100))
    tot_v = bull_vol + bear_vol
    s_vol = (bull_vol / tot_v - 0.5) * 200 if tot_v > 0 else 0.0
    hh = h5 > h6
    hl = l5 > l6
    if hh and hl:
        s_struct = 100
    elif hh:
        s_struct = 40
    elif hl:
        s_struct = 20
    elif h5 < h6 and l5 < l6:
        s_struct = -100
    else:
        s_struct = -60
    score = 0.35 * s_body + 0.15 * s_wick + 0.25 * s_vol + 0.25 * s_struct
    return float(np.clip(score, -100, 100))


def _power_score(tail: pd.DataFrame, atr: float, power_window: int = POWER_WINDOW) -> float:
    """近窗口多空力量评分 (-100~+100) — 全向量化"""
    tail = tail.tail(power_window)
    o = tail["open"].values
    h = tail["high"].values
    l = tail["low"].values
    c = tail["close"].values
    v = tail["volume"].values if "volume" in tail.columns else np.ones(len(tail))
    n = len(tail)
    if n < 5 or atr <= 0:
        return 0.0

    body = c - o
    bull = body > 0
    bear = body < 0
    bull_body = float(np.sum(body[bull]))
    bear_body = float(-np.sum(body[bear]))
    upper_wick = np.maximum(0, h - c)
    lower_wick = np.maximum(0, o - l)
    bull_wick = float(np.sum(upper_wick[bull]))
    bear_wick = float(np.sum(lower_wick[bear]))
    bull_vol = float(np.sum(v[bull]))
    bear_vol = float(np.sum(v[bear]))
    h5 = h[-1] if n >= 6 else h[-1]
    l5 = l[-1] if n >= 6 else l[-1]
    h6 = h[-6] if n >= 6 else h[-1]
    l6 = l[-6] if n >= 6 else l[-1]
    return score_from_components(bull_body, bear_body, bull_wick, bear_wick,
                                 bull_vol, bear_vol, atr, h5, l5, h6, l6)


def analyze_power(df: pd.DataFrame, power_window: int = POWER_WINDOW,
                  shift_window: int = SHIFT_WINDOW) -> dict:
    """多空力量识别 — 输入 K 线 DataFrame
    返回: score(-100~+100) + shift(增强/减弱/持平) + 明细
    """
    if df is None or len(df) < power_window + shift_window * 2 + 5:
        return {"score": 0, "shift": "unknown", "bull": 0, "bear": 0, "reason": ""}

    atr = _atr(df)
    tail = df.tail(power_window).reset_index(drop=True)
    score = _power_score(tail, atr, power_window)

    # 变化: 后shift_window根 vs 前shift_window根
    if len(tail) >= shift_window * 2:
        prev_score = _power_score(tail.iloc[:-shift_window], atr, shift_window)
        cur_score = _power_score(tail.iloc[-shift_window:], atr, shift_window)
        delta = cur_score - prev_score
        if delta >= 20:
            shift = "strengthening"
        elif delta <= -20:
            shift = "weakening"
        else:
            shift = "stable"
    else:
        shift = "stable"
        delta = 0.0

    bull = max(score, 0)
    bear = max(-score, 0)

    # 描述
    if score >= 30:
        power = "多头强势"
    elif score <= -30:
        power = "空头强势"
    elif score >= 10:
        power = "略偏多"
    elif score <= -10:
        power = "略偏空"
    else:
        power = "多空均衡"
    shift_names = {"strengthening": "优势方增强", "weakening": "优势方减弱", "stable": "持平"}
    reason = f"{power}({score:+.0f}) · {shift_names.get(shift, '')}"
    if shift == "strengthening":
        reason += f"({delta:+.0f})"
    elif shift == "weakening":
        reason += f"({delta:+.0f})"

    return {
        "score": round(score),
        "shift": shift,
        "delta": round(delta, 1),
        "bull": round(bull),
        "bear": round(bear),
        "reason": reason,
    }


def label_of(pw: dict) -> str:
    """多空力量标签 (推送用)"""
    if not pw or pw.get("shift") == "unknown":
        return ""
    score = pw.get("score", 0)
    icons = {"strengthening": "↑", "weakening": "↓", "stable": "→"}
    icon = icons.get(pw.get("shift", "stable"), "")
    d = "🟢多" if score > 0 else "🔴空"
    return f"{d}强{abs(score):.0f}{icon}"
