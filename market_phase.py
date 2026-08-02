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
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    out = np.zeros(n)
    e = 0.0
    alpha = 1 / period
    for i in range(n):
        e = tr[i] if i == 0 else alpha * tr[i] + (1 - alpha) * e
        out[i] = e
    return out


def _adx_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    alpha = 1 / period
    tr_s = np.zeros(n); pdi_s = np.zeros(n); mdi_s = np.zeros(n)
    for i in range(n):
        tr_s[i] = tr[i] if i == 0 else alpha*tr[i] + (1-alpha)*tr_s[i-1]
        pdi_s[i] = plus_dm[i] if i == 0 else alpha*plus_dm[i] + (1-alpha)*pdi_s[i-1]
        mdi_s[i] = minus_dm[i] if i == 0 else alpha*minus_dm[i] + (1-alpha)*mdi_s[i-1]
    dx = np.zeros(n)
    for i in range(n):
        s = pdi_s[i] + mdi_s[i]
        dx[i] = 100 * abs(pdi_s[i] - mdi_s[i]) / s if s > 0 else 0
    adx = np.zeros(n)
    for i in range(n):
        adx[i] = dx[i] if i < period else (adx[i-1]*(period-1) + dx[i]) / period
    return adx


def _empty() -> dict:
    return {
        "state": "unknown", "stage": None, "trend_dir": None,
        "strength": 0, "adx": 0, "dev": 0.0, "slope": 0.0,
        "reason": "", "score": 0,
    }


def analyze_market_state(df: pd.DataFrame) -> dict:
    """单周期市场状态分析 — 输入 K 线 DataFrame (≥80根)
    返回: state(趋势/震荡/转换) + stage(初期/加速/末期) + 指标明细
    """
    if df is None or len(df) < 70:
        return _empty()
    tail = df.tail(WINDOW).reset_index(drop=True)
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

    # 核心指标 (ATR 归一化)
    dev = (c[-1] - ma20[-1]) / atr_now                     # 价格偏离 MA20
    slope = (ma20[-1] - ma20[-11]) / atr_now               # MA20 10根斜率
    spread = (ma20[-1] - ma60[-1]) / atr_now               # MA20-MA60 发散度
    mom = (c[-1] - c[-11]) / atr_now                       # 10根动量
    adx_now = adx[-1]
    adx_prev = adx[-11] if n > 11 else adx_now
    adx_turn = adx_now - adx_prev                          # ADX 变化

    # 实体动能: 近3根 vs 前10根
    body = np.abs(c - o)
    body_recent = np.mean(body[-3:]) if n >= 3 else 0
    body_prior = np.mean(body[-13:-3]) if n >= 13 else body_recent
    body_ratio = body_recent / body_prior if body_prior > 0 else 1.0

    # ── 状态判定 ──
    trend_ok = adx_now >= ADX_TREND and abs(slope) >= 0.15
    range_ok = adx_now < ADX_RANGE and abs(slope) < 0.15 and abs(spread) < 0.5
    direction = "up" if mom > 0 else "down"
    state = "trend_up" if (trend_ok and mom > 0) else \
            "trend_down" if (trend_ok and mom < 0) else \
            "range" if range_ok else "transition"

    # ── 阶段判定 (仅趋势) ──
    stage = None
    dev_abs = abs(dev)
    if state.startswith("trend"):
        # 末期: 过冲 / ADX动能下降 / 实体衰竭
        late = (dev_abs > LATE_DEV) or (adx_turn < -LATE_ADX_FALL and adx_now >= ADX_TREND) \
               or (body_ratio < WEAK_BODY_RATIO and dev_abs > 1.0)
        # 初期: 刚启动 (偏离小 + ADX从低位上升)
        early = (dev_abs < EARLY_DEV and adx_now >= ADX_TREND) \
                or (adx_prev < ADX_TREND - 3 and adx_now >= ADX_TREND)
        if late:
            stage = "late"
        elif early:
            stage = "early"
        else:
            stage = "accelerate"

    # ── 趋势强度 0-100 ──
    strength = min(100, adx_now * 1.5 + abs(mom) * 8)

    reason_parts = [f"ADX{adx_now:.0f}", f"偏离{dev:+.1f}ATR"]
    if state.startswith("trend"):
        stage_names = {"early": "初期", "accelerate": "加速中期", "late": "末期"}
        reason_parts.append(f"阶段:{stage_names.get(stage, '?')}")
        reason_parts.append(f"斜{slope:+.1f}")
    elif state == "range":
        reason_parts.append("震荡")
    else:
        reason_parts.append("转换期")
    if body_ratio < WEAK_BODY_RATIO:
        reason_parts.append(f"动能减{body_ratio:.1f}")

    return {
        "state": state,
        "stage": stage,
        "trend_dir": direction if state.startswith("trend") else None,
        "strength": round(strength),
        "adx": round(adx_now, 1),
        "dev": round(dev, 2),
        "slope": round(slope, 2),
        "spread": round(spread, 2),
        "mom": round(mom, 2),
        "adx_turn": round(adx_turn, 1),
        "body_ratio": round(body_ratio, 2),
        "reason": " ".join(reason_parts),
        "score": round(strength * (1 if state.startswith("trend_up") else -1 if state.startswith("trend_down") else 0)),
    }


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
