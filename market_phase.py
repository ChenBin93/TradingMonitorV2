"""市场状态分类 — 用户方法论核心模块

分析框架 (用户方法论):
  1. 大周期(日线)定长期状态: 趋势/震荡 + 阶段(初期/加速/末期)
  2. 中周期(4H)、小周期(1H) 递进分析
  3. 每周期窗口 80-100 根 (更早的历史K线影响小, 除非重大高低点)

状态: trend_up / trend_down / range / transition
阶段: early(初期) / accelerate(加速) / late(末期) — 仅趋势状态有
"""
import numpy as np
import pandas as pd

WINDOW = 90  # 用户经验: 往前看不超过 80-100 根

# 阶段阈值
EARLY_DEV = 0.6       # |偏离| < 0.6 ATR 且刚启动 → 初期
LATE_DEV = 2.0        # |偏离| > 2.0 ATR → 末期(过冲)
ADX_TREND = 25        # ADX >= 25 趋势
ADX_RANGE = 20        # ADX < 20 震荡
LATE_ADX_FALL = 3.0   # ADX 较10根前回落超3 → 动能下降
WEAK_BODY_RATIO = 0.6 # 近3根实体/前10根实体 < 0.6 → 衰竭


def _atr_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ATR 序列 — 全向量化 (numpy + pandas ewm)"""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    if n > 1:
        tr[1:] = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values


def _adx_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """ADX 序列 — 全向量化"""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    up = np.zeros(n)
    dn = np.zeros(n)
    if n > 1:
        up[1:] = h[1:] - h[:-1]
        dn[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.zeros(n)
    if n > 1:
        tr[1:] = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    alpha = 1 / period
    tr_s = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    pdi_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values
    mdi_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values
    s = pdi_s + mdi_s
    dx = np.where(s > 0, 100 * np.abs(pdi_s - mdi_s) / np.where(s > 0, s, 1), 0.0)
    # adx[i] = dx[i] (i<period), 之后 adx[i] = (adx[i-1]*(period-1)+dx[i])/period
    # 等价于从 dx[period-1] 起的 ewm(alpha=1/period, adjust=False)
    adx = np.zeros(n)
    if n > period:
        adx[:period] = dx[:period]
        adx[period:] = pd.Series(dx[period - 1:]).ewm(alpha=alpha, adjust=False).mean().values[1:]
    else:
        adx = dx
    return adx


def _empty() -> dict:
    return {
        "state": "unknown", "stage": None, "trend_dir": None,
        "strength": 0, "adx": 0, "dev": 0.0, "slope": 0.0,
        "reason": "", "score": 0,
    }


def classify(atr_now: float, adx_now: float, adx_prev: float, slope: float,
             spread: float, dev: float, mom: float,
             body_recent: float, body_prior: float) -> dict:
    """状态/阶段判定核心 — 纯函数 (analyze_market_state 与预计算调优共用)"""
    if atr_now <= 1e-9:
        return _empty()

    trend_ok = adx_now >= ADX_TREND and abs(slope) >= 0.15
    range_ok = adx_now < ADX_RANGE and abs(slope) < 0.15 and abs(spread) < 0.5
    direction = "up" if mom > 0 else "down"
    state = "trend_up" if (trend_ok and mom > 0) else \
            "trend_down" if (trend_ok and mom < 0) else \
            "range" if range_ok else "transition"

    stage = None
    dev_abs = abs(dev)
    if state.startswith("trend"):
        adx_turn = adx_now - adx_prev
        body_ratio = body_recent / body_prior if body_prior > 0 else 1.0
        late = (dev_abs > LATE_DEV) or (adx_turn < -LATE_ADX_FALL and adx_now >= ADX_TREND) \
               or (body_ratio < WEAK_BODY_RATIO and dev_abs > 1.0)
        early = (dev_abs < EARLY_DEV and adx_now >= ADX_TREND) \
                or (adx_prev < ADX_TREND - 3 and adx_now >= ADX_TREND)
        if late:
            stage = "late"
        elif early:
            stage = "early"
        else:
            stage = "accelerate"

    strength = min(100, adx_now * 1.5 + abs(mom) * 8)
    return {
        "state": state, "stage": stage,
        "trend_dir": direction if state.startswith("trend") else None,
        "strength": round(strength),
        "adx": round(adx_now, 1),
        "dev": round(dev, 2),
        "slope": round(slope, 2),
        "spread": round(spread, 2),
        "mom": round(mom, 2),
        "adx_turn": round(adx_now - adx_prev, 1),
        "body_ratio": round(body_recent / body_prior, 2) if body_prior > 0 else 1.0,
        "score": round(strength * (1 if state.startswith("trend_up") else -1 if state.startswith("trend_down") else 0)),
    }


def analyze_market_state(df: pd.DataFrame, window: int = WINDOW) -> dict:
    """单周期市场状态分析 — 输入 K 线 DataFrame (≥80根)
    返回: state(趋势/震荡/转换) + stage(初期/加速/末期) + 指标明细
    """
    if df is None or len(df) < 70:
        return _empty()
    tail = df.tail(window).reset_index(drop=True)
    c = tail["close"].values
    o = tail["open"].values
    n = len(c)

    atr = _atr_series(tail)
    adx = _adx_series(tail)
    atr_now = atr[-1]
    if atr_now <= 1e-9:
        return _empty()

    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    ma60_now = ma60[-1] if not np.isnan(ma60[-1]) else ma20[-1]  # 小窗口时 MA60 退化为 MA20

    dev = (c[-1] - ma20[-1]) / atr_now
    slope = (ma20[-1] - ma20[-11]) / atr_now
    spread = (ma20[-1] - ma60_now) / atr_now
    mom = (c[-1] - c[-11]) / atr_now
    adx_now = adx[-1]
    adx_prev = adx[-11] if n > 11 else adx_now

    body = np.abs(c - o)
    body_recent = np.mean(body[-3:]) if n >= 3 else 0
    body_prior = np.mean(body[-13:-3]) if n >= 13 else body_recent

    ms = classify(atr_now, adx_now, adx_prev, slope, spread, dev, mom,
                  body_recent, body_prior)

    reason_parts = [f"ADX{ms['adx']:.0f}", f"偏离{ms['dev']:+.1f}ATR"]
    if ms["state"].startswith("trend"):
        stage_names = {"early": "初期", "accelerate": "加速中期", "late": "末期"}
        reason_parts.append(f"阶段:{stage_names.get(ms['stage'], '?')}")
        reason_parts.append(f"斜{ms['slope']:+.1f}")
    elif ms["state"] == "range":
        reason_parts.append("震荡")
    else:
        reason_parts.append("转换期")
    if ms["body_ratio"] < WEAK_BODY_RATIO:
        reason_parts.append(f"动能减{ms['body_ratio']:.1f}")
    ms["reason"] = " ".join(reason_parts)
    return ms


def label_of(ms: dict) -> str:
    """状态标签 (推送用)"""
    if not ms or ms.get("state") == "unknown":
        return "未知"
    state = ms["state"]
    if state == "range":
        return "📦震荡"
    if state == "transition":
        return "🔄转换"
    d = "📈" if state == "trend_up" else "📉"
    stage_names = {"early": "初期", "accelerate": "加速", "late": "末期"}
    st = stage_names.get(ms.get("stage"), "")
    return f"{d}趋势{st}" if st else f"{d}趋势"
