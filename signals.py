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
    ind_1h: dict = field(default_factory=dict)


@dataclass
class SignalDef:
    id: str
    name: str
    check: Callable
    params: dict
    tag: str = ""
    gate: str = "any"


def _sr_df(state: SignalState):
    """返回 S/R 计算用的 DataFrame: 1H 优先（大结构），否则用当前 TF"""
    df = (state.ind_1h or {}).get("df")
    return df if df is not None and len(df) > 0 else state.ind.get("df")


# ═══════════════════════════════════════════════════════════════════════
# S/R 结构信号 (用 1H S/R, 5m 入场)
# ═══════════════════════════════════════════════════════════════════════

def _check_breakout(state: SignalState) -> dict | None:
    df_sr = _sr_df(state)
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    vr = state.ind.get("volume_ratio") or 1
    if df_sr is None or len(df_sr) < 20 or not close:
        return None

    df_cur = state.ind.get("df")
    if df_cur is None or len(df_cur) == 0:
        return None
    idx = len(df_cur) - 1
    o = df_cur["open"].iloc[idx]

    levels = find_swing_levels(df_sr, lookback=50)
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
    df_sr = _sr_df(state)
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    if df_sr is None or len(df_sr) < 20 or not close:
        return None

    df_cur = state.ind.get("df")
    if df_cur is None or len(df_cur) == 0:
        return None
    idx = len(df_cur) - 1
    h = df_cur["high"].iloc[idx]
    l = df_cur["low"].iloc[idx]

    levels = find_swing_levels(df_sr, lookback=50)
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
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    if not close:
        return None

    df_cur = state.ind.get("df")
    if df_cur is None or len(df_cur) == 0:
        return None
    idx = len(df_cur) - 1
    h = df_cur["high"].iloc[idx]
    l = df_cur["low"].iloc[idx]
    o = df_cur["open"].iloc[idx]

    ds = get_defense_state()
    breaks = ds.get_recent_breaks(state.symbol)
    if not breaks:
        return None

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
    body_dir = state.ind.get("body_dir")
    if not rsi or not body_dir:
        return None
    if rsi <= state.params.get("oversold", 25):
        if body_dir != "bullish":
            return None
        sev = "critical" if rsi <= 20 else "high"
        return {"direction": "long", "severity": sev, "confidence": 0.85,
                "evidence": f"RSI超卖{rsi:.0f}", "rsi": rsi}
    if rsi >= state.params.get("overbot", 75):
        if body_dir != "bearish":
            return None
        sev = "critical" if rsi >= 80 else "high"
        return {"direction": "short", "severity": sev, "confidence": 0.85,
                "evidence": f"RSI超买{rsi:.0f}", "rsi": rsi}
    return None


def _check_volume_spike(state: SignalState) -> dict | None:
    vr = state.ind.get("volume_ratio")
    if not vr or vr < state.params.get("threshold", 3.0):
        return None
    roc = state.ind.get("roc") or 0
    chg = abs(roc)
    if chg < state.params.get("min_price_change", 0.5):
        return None
    direction = "long" if roc > 0 else "short"
    sev = "critical" if vr >= 5 else "high" if vr >= 3 else "medium"
    return {"direction": direction, "severity": sev,
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
# 趋势回调信号
# ═══════════════════════════════════════════════════════════════════════

def _check_trend_pullback(state: SignalState) -> dict | None:
    ind_1h = state.ind_1h
    if not ind_1h:
        return None

    ma_1h = ind_1h.get("ma_alignment", "neutral")
    adx_1h = ind_1h.get("adx", 0) or 0
    if adx_1h < 20:
        return None

    df_1h = ind_1h.get("df")
    close_5m = state.ind.get("close")
    atr_5m = state.ind.get("atr") or 1
    if df_1h is None or len(df_1h) < 20 or not close_5m:
        return None

    levels = find_swing_levels(df_1h, lookback=50)
    pinbar = state.ind.get("pinbar")
    vr_5m = state.ind.get("volume_ratio") or 1
    body_dir = state.ind.get("body_dir")
    body_pct = state.ind.get("body_pct", 0)
    rsi_5m = state.ind.get("rsi")

    for lvl in levels:
        if lvl.touch_count < 2:
            continue
        dist = abs(close_5m - lvl.price)

        # ── 做多: 1H多头 + 5m回踩到1H支撑 + 反转确认 ──
        if ma_1h == "bullish" and lvl.side == "support" and dist <= atr_5m * 1.2:
            reversal = False
            rev_type = ""
            if pinbar == "bullish":
                reversal = True
                rev_type = "Pinbar"
            elif rsi_5m is not None and rsi_5m < 40 and body_dir == "bullish" and body_pct > 0.3:
                reversal = True
                rev_type = f"RSI{rsi_5m:.0f}+阳线"
            if reversal:
                conf = 0.78 if pinbar == "bullish" else 0.70
                sev = "critical" if dist <= atr_5m * 0.5 else "high"
                return {
                    "direction": "long",
                    "severity": sev,
                    "confidence": conf,
                    "evidence": f"趋势回调多 1H支撑{lvl.price:.5g}({lvl.touch_count}触) {rev_type}",
                }

        # ── 做空: 1H空头 + 5m反弹到1H阻力 + 反转确认 ──
        elif ma_1h == "bearish" and lvl.side == "resistance" and dist <= atr_5m * 1.2:
            reversal = False
            rev_type = ""
            if pinbar == "bearish":
                reversal = True
                rev_type = "Pinbar"
            elif rsi_5m is not None and rsi_5m > 60 and body_dir == "bearish" and body_pct > 0.3:
                reversal = True
                rev_type = f"RSI{rsi_5m:.0f}+阴线"
            if reversal:
                conf = 0.78 if pinbar == "bearish" else 0.70
                sev = "critical" if dist <= atr_5m * 0.5 else "high"
                return {
                    "direction": "short",
                    "severity": sev,
                    "confidence": conf,
                    "evidence": f"趋势回调空 1H阻力{lvl.price:.5g}({lvl.touch_count}触) {rev_type}",
                }

    return None


# ═══════════════════════════════════════════════════════════════════════
# 信号注册表
# ═══════════════════════════════════════════════════════════════════════

SIGNALS: list[SignalDef] = [
    SignalDef("breakout",        "防线突破",   _check_breakout,        {},                    "突破", "trend"),
    SignalDef("fakeout",         "假突破反转", _check_fakeout,         {},                    "假破", "any"),
    SignalDef("retest",          "回踩确认",   _check_retest,          {},                    "回踩", "any"),
    SignalDef("rsi_extreme",     "RSI极值",    _check_rsi_extreme,     {"oversold": 25, "overbot": 75}, "RSI", "any"),
    SignalDef("volume_spike",    "放量异动",   _check_volume_spike,    {"threshold": 3.0, "min_price_change": 0.5}, "VOL", "any"),
    SignalDef("trend_pullback",  "趋势回调",   _check_trend_pullback,  {},                    "回调", "trend"),
]


# ═══════════════════════════════════════════════════════════════════════
# 预警系统 (30m推送, 比信号更早的预备提示)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class WarningDef:
    id: str
    name: str
    check: Callable


def _check_near_sr(state: SignalState) -> dict | None:
    ind_1h = state.ind_1h
    if not ind_1h:
        return None
    df_1h = ind_1h.get("df")
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    if df_1h is None or len(df_1h) < 20 or not close:
        return None

    levels = find_swing_levels(df_1h, lookback=50)
    for lvl in levels:
        if lvl.touch_count < 2:
            continue
        dist = abs(close - lvl.price)
        if dist <= atr * 2.0:
            direction = "多" if lvl.side == "support" else "空"
            return {
                "type": "near_sr",
                "evidence": f"接近{lvl.side}{lvl.price:.5g}({dist/atr:.1f}ATR)",
                "direction": direction,
                "sr_price": lvl.price,
                "sr_side": lvl.side,
            }
    return None


def _check_coiling(state: SignalState) -> dict | None:
    bb_state = state.ind.get("bb_state", "")
    comp_bars = state.ind.get("compression_bars", 0)
    adx = state.ind.get("adx", 0) or 0
    ts = state.ind.get("ttm_squeeze")
    squeeze = ts.get("squeeze_bars", 0) if ts else 0

    if comp_bars >= 4 or squeeze >= 5 or bb_state == "contracting":
        return {
            "type": "coiling",
            "evidence": f"蓄力中 {'压{comp_bars}根' if comp_bars>=4 else ''}{' ADX={adx:.0f}' if adx>0 else ''}",
            "comp_bars": comp_bars,
            "adx": adx,
        }
    return None


def _check_rsi_approaching(state: SignalState) -> dict | None:
    rsi = state.ind.get("rsi")
    if not rsi:
        return None
    if 30 <= rsi <= 35:
        return {"type": "rsi_approaching", "evidence": f"RSI趋近超卖{rsi:.0f}", "rsi": rsi, "side": "oversold"}
    if 65 <= rsi <= 70:
        return {"type": "rsi_approaching", "evidence": f"RSI趋近超买{rsi:.0f}", "rsi": rsi, "side": "overbot"}
    return None


def _check_rs_moving(state: SignalState) -> dict | None:
    rs5 = state.rs_scores.get("5m", {})
    score = rs5.get("score", 0)
    if 15 <= abs(score) < 30:
        side = "强" if score > 0 else "弱"
        return {"type": "rs_moving", "evidence": f"RS走{side}{score:+.0f}", "rs_score": score}
    return None


WARNINGS: list[WarningDef] = [
    WarningDef("near_sr",        "接近防线",   _check_near_sr),
    WarningDef("coiling",        "蓄力待发",   _check_coiling),
    WarningDef("rsi_approaching","RSI趋近",    _check_rsi_approaching),
    WarningDef("rs_moving",      "RS异动",     _check_rs_moving),
]


def check_warnings(state: SignalState) -> list[dict]:
    results = []
    for wdef in WARNINGS:
        try:
            r = wdef.check(state)
            if r:
                r["id"] = wdef.id
                r["name"] = wdef.name
                results.append(r)
        except Exception:
            pass
    return results


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
