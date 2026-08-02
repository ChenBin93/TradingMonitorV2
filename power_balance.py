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
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    if n < 2:
        return 0.0
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    return float(np.mean(tr[-14:]))


def _power_score(tail: pd.DataFrame, atr: float) -> float:
    """近窗口多空力量评分 (-100~+100)"""
    o = tail["open"].values
    h = tail["high"].values
    l = tail["low"].values
    c = tail["close"].values
    v = tail["volume"].values if "volume" in tail.columns else np.ones(len(tail))
    n = len(tail)
    if n < 5 or atr <= 0:
        return 0.0

    bull_body = bear_body = 0.0
    bull_vol = bear_vol = 0.0
    bull_wick = bear_wick = 0.0
    for i in range(n):
        body = c[i] - o[i]
        rng = h[i] - l[i]
        if body > 0:
            bull_body += body
            bull_vol += v[i]
            bull_wick += max(0, h[i] - c[i])      # 阳线上影 = 空头打压
        elif body < 0:
            bear_body += -body
            bear_vol += v[i]
            bear_wick += max(0, o[i] - l[i])      # 阴线下影 = 多头抵抗

    # 1. 实体力量 (35%) — 净实体 / ATR, 归一化
    net_body = (bull_body - bear_body) / atr
    s_body = np.clip(net_body * 12, -100, 100)

    # 2. 影线对抗 (15%) — 多头抵抗 vs 空头打压
    net_wick = (bull_wick - bear_wick) / atr
    s_wick = np.clip(net_wick * 15, -100, 100)

    # 3. 量能方向 (25%) — 阳线量占比
    tot_v = bull_vol + bear_vol
    if tot_v > 0:
        s_vol = (bull_vol / tot_v - 0.5) * 200
    else:
        s_vol = 0.0

    # 4. 结构推进 (25%) — 高低点推进方向
    # 近5根高点/低点 vs 前5根 (HH/HL 多, LH/LL 空)
    if n >= 12:
        hh = h[-1] > h[-6]       # 高点抬高
        hl = l[-1] > l[-6]       # 低点抬高
        if hh and hl:
            s_struct = 100
        elif hh:
            s_struct = 40
        elif hl:
            s_struct = 20
        elif not hh and not hl and h[-1] < h[-6] and l[-1] < l[-6]:
            s_struct = -100
        elif not hh and not hl:
            s_struct = -60
        else:
            s_struct = -20
    else:
        s_struct = 0.0

    score = 0.35 * s_body + 0.15 * s_wick + 0.25 * s_vol + 0.25 * s_struct
    return float(np.clip(score, -100, 100))


def analyze_power(df: pd.DataFrame) -> dict:
    """多空力量识别 — 输入 K 线 DataFrame
    返回: score(-100~+100) + shift(增强/减弱/持平) + 明细
    """
    if df is None or len(df) < 60:
        return {"score": 0, "shift": "unknown", "bull": 0, "bear": 0, "reason": ""}

    atr = _atr(df)
    tail = df.tail(POWER_WINDOW).reset_index(drop=True)
    score = _power_score(tail, atr)

    # 变化: 后10根 vs 前10根
    if len(tail) >= 20:
        prev_score = _power_score(tail.iloc[:-10], atr)
        cur_score = _power_score(tail.iloc[-10:], atr)
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
