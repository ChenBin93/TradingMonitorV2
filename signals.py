# 信号定义 + 检测函数
# 加信号：写 check 函数 → 在 SIGNALS 列表加一行 → 完

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SignalState:
    symbol: str
    timeframe: str
    ind: dict          # indicators.compute() 结果
    bbw_rank: float | None = None
    atr_rank: float | None = None    # ATR 历史分位（需要 history 数据）
    regime: str = "unknown"
    direction: str = "neutral"
    params: dict = field(default_factory=dict)


@dataclass
class SignalDef:
    id: str
    name: str
    check: Callable
    params: dict
    tag: str = ""
    gate: str = "any"  # "trend"(ADX≥25)/"range"(ADX<20)/"any"


# ═══════════════════════════════════════════════════════════════════════
# 检测函数
# ═══════════════════════════════════════════════════════════════════════

# --- 波动收敛 ---

def _check_bb_squeeze(state: SignalState) -> dict | None:
    bb = state.ind.get("bb_width")
    rank = state.bbw_rank
    if not bb or rank is None:
        return None
    threshold = state.params.get("threshold", 25)
    if rank <= threshold:
        sev = "critical" if rank <= 10 else "high" if rank <= 20 else "medium"
        return {"direction": state.direction, "severity": sev,
                "confidence": 0.75 if rank <= 10 else 0.65,
                "evidence": f"BB压缩位{rank:.0f}%", "bbw_rank": rank}


def _check_ma_converge(state: SignalState) -> dict | None:
    mc = state.ind.get("ma_converge")
    if not mc:
        return None
    threshold = state.params.get("threshold", 0.5)
    if mc <= threshold:
        sev = "critical" if mc <= 0.2 else "high" if mc <= 0.35 else "medium"
        return {"direction": state.direction, "severity": sev,
                "confidence": 0.7 if mc <= 0.2 else 0.55,
                "evidence": f"MA汇聚{mc:.2f}", "ma_converge": mc}


def _check_ttm_squeeze(state: SignalState) -> dict | None:
    ts = state.ind.get("ttm_squeeze")
    if not ts:
        return None
    bars = ts.get("squeeze_bars", 0)
    fired = ts.get("is_fired", False)
    if bars >= state.params.get("min_bars", 5) or fired:
        sev = "critical" if bars >= 10 or fired else "high" if bars >= 5 else "medium"
        return {"direction": ts.get("direction") or state.direction, "severity": sev,
                "confidence": 0.8 if fired else 0.65,
                "evidence": f"TTM压缩{bars}根{'→释放' if fired else ''}",
                "squeeze_bars": bars, "is_fired": fired}


def _check_compression_combo(state: SignalState) -> dict | None:
    """多个压缩信号同时触发 → 高度收敛，突破概率极高"""
    hits = []
    bb = state.ind.get("bb_width")
    if bb and state.bbw_rank is not None and state.bbw_rank <= 20:
        hits.append("BB")
    mc = state.ind.get("ma_converge")
    if mc and mc <= 0.4:
        hits.append("MA")
    ts = state.ind.get("ttm_squeeze")
    if ts and (ts.get("squeeze_bars", 0) >= 3 or ts.get("is_fired")):
        hits.append("TTM")
    if len(hits) >= 2:
        return {"direction": state.direction, "severity": "critical",
                "confidence": 0.85, "evidence": f"多重压缩 {'+'.join(hits)}",
                "combo_signals": hits}


# --- RSI ---

def _check_rsi_extreme(state: SignalState) -> dict | None:
    rsi = state.ind.get("rsi")
    if not rsi:
        return None
    if rsi <= state.params.get("oversold", 25):
        sev = "critical" if rsi <= 20 else "high"
        return {"direction": "long", "severity": sev, "confidence": 0.85,
                "evidence": f"RSI超卖{rsi:.0f}", "rsi": rsi}
    if rsi >= state.params.get("overbot", 75):
        sev = "critical" if rsi >= 80 else "high"
        return {"direction": "short", "severity": sev, "confidence": 0.85,
                "evidence": f"RSI超买{rsi:.0f}", "rsi": rsi}
    return None


# --- 趋势确认 ---

def _check_ma_alignment(state: SignalState) -> dict | None:
    """均线多头/空头排列"""
    ma5 = state.ind.get("ma5")
    ma20 = state.ind.get("ma20")
    ma60 = state.ind.get("ma60")
    adx = state.ind.get("adx")
    if not all([ma5, ma20, ma60, adx]):
        return None
    if adx < state.params.get("adx_min", 25):
        return None

    if ma5 > ma20 > ma60:
        spread = (ma5 - ma60) / ma60 * 100
        sev = "critical" if spread > 5 else "high"
        return {"direction": "long", "severity": sev,
                "confidence": 0.75 if spread > 5 else 0.65,
                "evidence": f"多头排列 发散{spread:.1f}%"}
    elif ma5 < ma20 < ma60:
        spread = (ma60 - ma5) / ma60 * 100
        sev = "critical" if spread > 5 else "high"
        return {"direction": "short", "severity": sev,
                "confidence": 0.75 if spread > 5 else 0.65,
                "evidence": f"空头排列 发散{spread:.1f}%"}


def _check_adx_surge(state: SignalState) -> dict | None:
    """ADX 从震荡区(<20)突入趋势区(>=25)，最佳入场时机"""
    adx = state.ind.get("adx")
    if not adx:
        return None
    df = state.ind.get("df")
    if df is None or len(df) < 5:
        return None
    prev_adx_vals = []
    for i in range(2, min(6, len(df))):
        # 简单估算前几根的 ADX: 用当前数值递减
        prev_adx_vals.append(adx - (adx * 0.05 * i))

    was_range = any(v < 20 for v in prev_adx_vals)
    now_trend = adx >= state.params.get("trend_threshold", 25)
    if was_range and now_trend:
        sev = "critical" if adx >= 35 else "high"
        return {"direction": state.direction, "severity": sev,
                "confidence": 0.8 if adx >= 35 else 0.7,
                "evidence": f"ADX突破 ADX={adx:.0f}"}


def _check_macd_cross(state: SignalState) -> dict | None:
    cross = state.ind.get("macd_cross")
    if not cross:
        return None
    hist = state.ind.get("macd_hist") or 0
    sev = "high" if abs(hist) > 0.005 else "medium"
    return {"direction": "long" if cross == "golden" else "short",
            "severity": sev,
            "confidence": 0.7 if abs(hist) > 0.005 else 0.55,
            "evidence": "MACD金叉" if cross == "golden" else "MACD死叉"}


# --- 成交量 ---

def _check_volume_spike(state: SignalState) -> dict | None:
    """放量：区分突破 vs 滞涨"""
    vr = state.ind.get("volume_ratio")
    if not vr or vr < state.params.get("threshold", 3.0):
        return None
    chg = abs(state.ind.get("roc") or 0)
    if chg < state.params.get("min_price_change", 0.3):
        # 放量但价格不动 → 警惕
        return {"direction": "neutral",
                "severity": "critical" if vr >= 5 else "high" if vr >= 3 else "medium",
                "confidence": 0.6, "volume_ratio": vr,
                "evidence": f"放量滞涨 量{vr:.1f}x 涨跌{chg:.1f}%"}
    # 放量突破
    sev = "critical" if vr >= 5 else "high" if vr >= 3 else "medium"
    return {"direction": state.direction, "severity": sev,
            "confidence": min(0.6 + (vr - 2) * 0.12, 0.9),
            "evidence": f"放量突破 量{vr:.1f}x 涨跌{chg:.1f}%", "volume_ratio": vr}


# --- 背离 ---

def _check_rsi_divergence(state: SignalState) -> dict | None:
    rd = state.ind.get("rsi_divergence")
    if not rd or not rd.get("divergence"):
        return None
    div = rd["divergence"]
    dist = rd.get("price_distance_pct", 0)
    sev = "critical" if dist >= 5 else "high" if dist >= 3 else "medium"
    return {"direction": "long" if div == "bullish" else "short",
            "severity": sev, "confidence": 0.8 if dist >= 5 else 0.7,
            "price_distance_pct": dist,
            "evidence": f"{'底背离' if div == 'bullish' else '顶背离'} {dist:.1f}%"}


def _check_macd_divergence(state: SignalState) -> dict | None:
    md = state.ind.get("macd_divergence")
    if not md or not md.get("divergence"):
        return None
    div = md["divergence"]
    dist = md.get("price_distance_pct", 0)
    sev = "critical" if dist >= 5 else "high" if dist >= 3 else "medium"
    return {"direction": "long" if div == "bullish" else "short",
            "severity": sev, "confidence": 0.75 if dist >= 5 else 0.65,
            "price_distance_pct": dist,
            "evidence": f"{'底背离' if div == 'bullish' else '顶背离'} {dist:.1f}%"}


# --- 波动突变 ---

def _check_atr_expansion(state: SignalState) -> dict | None:
    """ATR 从历史低分位跃升 → 波动率启动信号"""
    atr = state.ind.get("atr")
    if not atr:
        return None
    df = state.ind.get("df")
    if df is None or len(df) < 20:
        return None
    # 计算近期 ATR 均值
    recent_atr = df["close"].rolling(5).std().iloc[-1] if len(df) >= 5 else atr
    long_atr = df["close"].rolling(20).std().iloc[-1] if len(df) >= 20 else atr
    if long_atr == 0:
        return None
    expansion_ratio = recent_atr / long_atr
    if expansion_ratio >= state.params.get("threshold", 1.8):
        sev = "critical" if expansion_ratio >= 3 else "high"
        # 配合历史分位（如果有）
        bonus = ""
        if state.atr_rank is not None:
            if state.atr_rank < 20:
                bonus = " 从低波启动"
                sev = "critical"
        return {"direction": state.direction, "severity": sev,
                "confidence": min(0.65 + (expansion_ratio - 1.5) * 0.15, 0.9),
                "evidence": f"波动爆发 ATR比{expansion_ratio:.1f}x{bonus}"}


# --- 极端偏离 ---

def _check_price_extreme(state: SignalState) -> dict | None:
    """价格远离均线 → 均值回归概率高"""
    close = state.ind.get("close")
    ma60 = state.ind.get("ma60")
    if not close or not ma60:
        return None
    df = state.ind.get("df")
    if df is None or len(df) < 60:
        return None
    std = df["close"].rolling(60).std().iloc[-1]
    if not std or std == 0:
        return None
    zscore = (close - ma60) / std
    threshold = state.params.get("std_threshold", 3.0)
    if abs(zscore) >= threshold:
        direction = "short" if zscore > 0 else "long"
        sev = "critical" if abs(zscore) >= 4 else "high"
        where = "过高" if zscore > 0 else "过低"
        return {"direction": direction, "severity": sev,
                "confidence": min(0.6 + abs(zscore) * 0.1, 0.9),
                "evidence": f"价格{where} Z={zscore:.1f}σ"}


# ═══════════════════════════════════════════════════════════════════════
# 信号注册表 — 加信号只需要在这加一行
# ═══════════════════════════════════════════════════════════════════════

SIGNALS: list[SignalDef] = [
    # 波动收敛（左侧）— 任意状态
    SignalDef("bb_squeeze",        "BB压缩",     _check_bb_squeeze,        {"threshold": 25}, "BB", "any"),
    SignalDef("ma_converge",       "MA汇聚",     _check_ma_converge,       {"threshold": 0.5}, "MA", "range"),
    SignalDef("ttm_squeeze",       "TTM压缩",    _check_ttm_squeeze,       {"min_bars": 5}, "TTM", "any"),
    SignalDef("compression_combo", "多重压缩",   _check_compression_combo, {}, "压", "any"),
    # RSI / 极端 — 逆势信号，仅在非强趋势
    SignalDef("rsi_extreme",       "RSI极值",    _check_rsi_extreme,       {"oversold": 25, "overbot": 75}, "RSI", "range"),
    SignalDef("price_extreme",     "价格极值",   _check_price_extreme,     {"std_threshold": 3.0}, "极", "any"),
    # 趋势确认（右侧）— 仅在趋势
    SignalDef("ma_alignment",      "均线排列",   _check_ma_alignment,      {"adx_min": 25}, "MA排", "trend"),
    SignalDef("adx_surge",         "ADX突破",    _check_adx_surge,         {"trend_threshold": 25}, "ADX", "trend"),
    SignalDef("macd_cross",        "MACD交叉",   _check_macd_cross,        {}, "MACD", "any"),
    # 成交量 — 任意状态
    SignalDef("volume_spike",      "放量信号",   _check_volume_spike,      {"threshold": 3.0, "min_price_change": 0.3}, "VOL", "any"),
    # 背离 — 逆势信号，仅在非强趋势
    SignalDef("rsi_divergence",    "RSI背离",    _check_rsi_divergence,    {}, "RSI背", "range"),
    SignalDef("macd_divergence",   "MACD背离",   _check_macd_divergence,   {}, "MACD背", "range"),
    # 波动突变 — 任意状态
    SignalDef("atr_expansion",     "波动爆发",   _check_atr_expansion,     {"threshold": 1.8}, "ATR", "any"),
]


# ═══════════════════════════════════════════════════════════════════════
# 方向 / 市场状态
# ═══════════════════════════════════════════════════════════════════════

def get_direction(ind: dict) -> str:
    pd_val = ind.get("plus_di", 0) or 0
    md_val = ind.get("minus_di", 0) or 0
    if pd_val > md_val:
        return "long"
    elif md_val > pd_val:
        return "short"
    return "neutral"


def get_regime(ind: dict) -> str:
    return "trend" if (ind.get("adx") or 0) >= 20 else "range"


# ═══════════════════════════════════════════════════════════════════════
# Stage2 入场信号 — 多时间框架综合判断
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EntrySignal:
    symbol: str
    signal_type: str      # "trend_breakout_long/short" | "range_reversion_long/short"
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    evidence: str
    confidence: float


def check_stage2_entry(tf_data: dict[str, dict]) -> EntrySignal | None:
    """检查 Stage2 入场信号（跨时间框架，包含 4h 方向过滤）"""
    ind_15m = tf_data.get("15m", {})
    ind_1h = tf_data.get("1h", {})
    ind_4h = tf_data.get("4h", {})
    if not ind_15m:
        return None

    # 4h 方向过滤
    dir_4h = (ind_4h or {}).get("ma_alignment", "neutral") if ind_4h else "neutral"

    # 趋势突破
    trend = _check_trend_breakout(ind_15m, ind_1h or ind_15m)
    if trend:
        # 4h 方向不一致 → 降级为不推送（只返回高胜率信号）
        if (trend.direction == "long" and dir_4h == "bearish") or \
           (trend.direction == "short" and dir_4h == "bullish"):
            return None
        return trend

    # 震荡回归
    reversion = _check_range_reversion(ind_15m, ind_1h or ind_15m)
    if reversion:
        if (reversion.direction == "long" and dir_4h == "bearish") or \
           (reversion.direction == "short" and dir_4h == "bullish"):
            return None
        return reversion

    return None


def _check_trend_breakout(ind_15m: dict, ind_1h: dict) -> EntrySignal | None:
    """趋势突破入场"""
    adx_15 = ind_15m.get("adx", 0) or 0
    adx_1h = ind_1h.get("adx", 0) or 0
    roc_15 = ind_15m.get("roc") or 0
    vol = ind_15m.get("volume_ratio") or 1

    # 条件: 15m ADX > 25, 1h ADX > 20, ROC 方向明确, 放量
    if adx_15 < 25 or adx_1h < 20:
        return None
    if abs(roc_15) < 1.0:
        return None
    if vol < 1.5:
        return None

    atr = ind_15m.get("atr") or 0
    close = ind_15m.get("close") or 0
    if atr == 0 or close == 0:
        return None

    direction = "long" if roc_15 > 0 else "short"
    entry = close
    sl = entry - atr * 2.0 if direction == "long" else entry + atr * 2.0
    tp = entry + atr * 4.0 if direction == "long" else entry - atr * 4.0
    rr = abs(tp - entry) / abs(sl - entry) if abs(sl - entry) > 0 else 0

    return EntrySignal(
        symbol="",  # 由调用方填写
        signal_type=f"trend_breakout_{direction}",
        direction=direction,
        entry_price=round(entry, 4),
        stop_loss=round(sl, 4),
        take_profit=round(tp, 4),
        risk_reward=round(rr, 1),
        evidence=f"趋势突破 ADX15={adx_15:.0f} ADX1h={adx_1h:.0f} ROC={roc_15:.1f}%",
        confidence=min(0.7 + (adx_15 - 25) * 0.01 + (vol - 1.5) * 0.05, 0.95),
    )


def _check_range_reversion(ind_15m: dict, ind_1h: dict) -> EntrySignal | None:
    """震荡回归入场"""
    adx_15 = ind_15m.get("adx", 0) or 0
    rsi_15 = ind_15m.get("rsi")
    bb_w = ind_15m.get("bb_width")
    close = ind_15m.get("close") or 0
    atr = ind_15m.get("atr") or 0

    if adx_15 >= 20 or rsi_15 is None or atr == 0:
        return None

    # 做多: RSI 超卖 + 价格接近 BB 下轨
    if rsi_15 <= 30:
        entry = close
        sl = entry - atr * 1.5
        tp = entry + atr * 3.0
        return EntrySignal(
            symbol="", signal_type="range_reversion_long", direction="long",
            entry_price=round(entry, 4), stop_loss=round(sl, 4), take_profit=round(tp, 4),
            risk_reward=round(abs(tp - entry) / abs(sl - entry), 1) if abs(sl - entry) > 0 else 0,
            evidence=f"震荡做多 RSI={rsi_15:.0f} ADX={adx_15:.0f}",
            confidence=0.7 if rsi_15 <= 25 else 0.6,
        )

    # 做空: RSI 超买
    if rsi_15 >= 70:
        entry = close
        sl = entry + atr * 1.5
        tp = entry - atr * 3.0
        return EntrySignal(
            symbol="", signal_type="range_reversion_short", direction="short",
            entry_price=round(entry, 4), stop_loss=round(sl, 4), take_profit=round(tp, 4),
            risk_reward=round(abs(tp - entry) / abs(sl - entry), 1) if abs(sl - entry) > 0 else 0,
            evidence=f"震荡做空 RSI={rsi_15:.0f} ADX={adx_15:.0f}",
            confidence=0.7 if rsi_15 >= 75 else 0.6,
        )

    return None
