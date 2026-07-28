import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable

from support_resistance import find_swing_levels
from defense_state import get_defense_state


@dataclass
class SignalState:
    symbol: str
    timeframe: str
    ind: dict
    regime: str = "unknown"
    direction: str = "neutral"
    params: dict = field(default_factory=dict)
    rs_scores: dict[str, dict] = field(default_factory=dict)
    # rs_scores = {"5m": {"score": 50, "level": "strong", "zscore": 3.0}, "1h": {...}, "4h": {...}}


@dataclass
class SignalDef:
    id: str
    name: str
    check: Callable
    params: dict
    tag: str = ""
    gate: str = "any"


# ═══════════════════════════════════════════════════════════════════════
# S/R 结构信号
# ═══════════════════════════════════════════════════════════════════════

def _check_breakout(state: SignalState) -> dict | None:
    df = state.ind.get("df")
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    vr = state.ind.get("volume_ratio") or 1
    if df is None or len(df) < 20 or not close:
        return None

    levels = find_swing_levels(df, lookback=50)
    idx = len(df) - 1
    o = df["open"].iloc[idx]

    for lvl in levels:
        if lvl.touch_count < 2:
            continue
        direction = None
        body_penetration = 0.0
        if lvl.side == "resistance" and close > lvl.price:
            body_penetration = min(close, max(o, lvl.price)) - lvl.price
            direction = "long"
        elif lvl.side == "support" and close < lvl.price:
            body_penetration = lvl.price - max(close, min(o, lvl.price))
            direction = "short"
        if direction and body_penetration >= atr * 0.3 and vr >= 2.0:
            get_defense_state().record_break(
                state.symbol, lvl.price, lvl.side,
                "up" if direction == "long" else "down"
            )
            return {
                "direction": direction,
                "severity": "high",
                "confidence": 0.7 if vr >= 3 else 0.6,
                "evidence": f"突破{lvl.price:.5g}({lvl.side}) 量{vr:.1f}x",
                "break_price": lvl.price,
                "break_side": lvl.side,
            }
    return None


def _check_fakeout(state: SignalState) -> dict | None:
    df = state.ind.get("df")
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    if df is None or len(df) < 20 or not close:
        return None

    levels = find_swing_levels(df, lookback=50)
    idx = len(df) - 1
    h = df["high"].iloc[idx]
    l = df["low"].iloc[idx]
    o = df["open"].iloc[idx]

    for lvl in levels:
        if lvl.touch_count < 2:
            continue
        direction = None
        penetration = 0.0
        if lvl.side == "resistance" and h > lvl.price and close < lvl.price:
            penetration = h - lvl.price
            direction = "short"
        elif lvl.side == "support" and l < lvl.price and close > lvl.price:
            penetration = lvl.price - l
            direction = "long"
        if direction and penetration >= atr * 0.3:
            total_range = h - l
            if total_range > 0 and penetration >= total_range * 0.5:
                return {
                    "direction": direction,
                    "severity": "critical" if lvl.touch_count >= 3 else "high",
                    "confidence": 0.75 if lvl.touch_count >= 3 else 0.65,
                    "evidence": f"假突破{lvl.price:.5g}({lvl.touch_count}触)",
                }
    return None


def _check_retest(state: SignalState) -> dict | None:
    df = state.ind.get("df")
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    if df is None or len(df) < 20 or not close:
        return None

    ds = get_defense_state()
    breaks = ds.get_recent_breaks(state.symbol)
    if not breaks:
        return None

    idx = len(df) - 1
    h = df["high"].iloc[idx]
    l = df["low"].iloc[idx]
    o = df["open"].iloc[idx]

    for br in breaks:
        if br["side"] == "resistance" and br["dir"] == "up":
            dist = close - br["price"]
            if 0 <= dist <= atr * 0.8:
                body = abs(close - o)
                lower_wick = min(o, close) - l if l < min(o, close) else 0
                if lower_wick >= atr * 0.3 and close > o:
                    return {
                        "direction": "long",
                        "severity": "critical",
                        "confidence": 0.75,
                        "evidence": f"回踩确认{br['price']:.5g}(阻力→支撑)",
                    }
        elif br["side"] == "support" and br["dir"] == "down":
            dist = br["price"] - close
            if 0 <= dist <= atr * 0.8:
                body = abs(close - o)
                upper_wick = h - max(o, close) if h > max(o, close) else 0
                if upper_wick >= atr * 0.3 and close < o:
                    return {
                        "direction": "short",
                        "severity": "critical",
                        "confidence": 0.75,
                        "evidence": f"回踩确认{br['price']:.5g}(支撑→阻力)",
                    }
    return None


# ═══════════════════════════════════════════════════════════════════════
# 市场信号
# ═══════════════════════════════════════════════════════════════════════

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


def _check_volume_spike(state: SignalState) -> dict | None:
    vr = state.ind.get("volume_ratio")
    if not vr or vr < state.params.get("threshold", 3.0):
        return None
    chg = abs(state.ind.get("roc") or 0)
    if chg < state.params.get("min_price_change", 0.5):
        return None
    sev = "critical" if vr >= 5 else "high" if vr >= 3 else "medium"
    return {"direction": state.direction, "severity": sev,
            "confidence": min(0.6 + (vr - 2) * 0.12, 0.9),
            "evidence": f"放量突破 量{vr:.1f}x 涨跌{chg:.1f}%",
            "volume_ratio": vr}


# ═══════════════════════════════════════════════════════════════════════
# 相对强弱信号
# ═══════════════════════════════════════════════════════════════════════

def _check_rs_strength(state: SignalState) -> dict | None:
    rs5 = state.rs_scores.get("5m", {})
    score = rs5.get("score", 0)
    threshold = state.params.get("score_threshold", 30)
    if score < threshold:
        return None
    sev = "critical" if score >= 60 else "high"
    return {"direction": "long", "severity": sev,
            "confidence": min(0.65 + score * 0.004, 0.95),
            "evidence": f"RS强势 5m:{score:.0f}",
            "rs_5m_score": score,
            "rs_scores": state.rs_scores}


def _check_rs_weakness(state: SignalState) -> dict | None:
    rs5 = state.rs_scores.get("5m", {})
    score = rs5.get("score", 0)
    threshold = state.params.get("score_threshold", 30)
    if score > -threshold:
        return None
    sev = "critical" if score <= -60 else "high"
    return {"direction": "short", "severity": sev,
            "confidence": min(0.65 + abs(score) * 0.004, 0.95),
            "evidence": f"RS弱势 5m:{score:.0f}",
            "rs_5m_score": score,
            "rs_scores": state.rs_scores}


# ═══════════════════════════════════════════════════════════════════════
# 信号注册表
# ═══════════════════════════════════════════════════════════════════════

SIGNALS: list[SignalDef] = [
    SignalDef("breakout",    "防线突破",   _check_breakout,     {},                    "突破", "trend"),
    SignalDef("fakeout",     "假突破反转", _check_fakeout,      {},                    "假破", "any"),
    SignalDef("retest",      "回踩确认",   _check_retest,       {},                    "回踩", "any"),
    SignalDef("rsi_extreme", "RSI极值",    _check_rsi_extreme,  {"oversold": 25, "overbot": 75}, "RSI", "range"),
    SignalDef("volume_spike","放量异动",   _check_volume_spike, {"threshold": 3.0, "min_price_change": 0.5}, "VOL", "any"),
    SignalDef("rs_strength", "RS强势",     _check_rs_strength,  {"score_threshold": 30}, "RS", "any"),
    SignalDef("rs_weakness", "RS弱势",     _check_rs_weakness,  {"score_threshold": 30}, "RS", "any"),
]


# ═══════════════════════════════════════════════════════════════════════
# 辅助
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


def is_compressing(ind: dict) -> bool:
    bb_state = ind.get("bb_state", "")
    comp_bars = ind.get("compression_bars", 0)
    ts = ind.get("ttm_squeeze")
    squeeze_active = ts.get("squeeze_active", False) if ts else False
    return bb_state == "contracting" or comp_bars >= 3 or squeeze_active
