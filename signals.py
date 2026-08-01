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
    ind_4h: dict = field(default_factory=dict)


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
    atr = state.ind.get("atr") or 1
    close = state.ind.get("close") or 1
    atr_pct = atr / close * 100
    min_atr = state.params.get("min_price_change", 0.3)
    if abs(roc) < atr_pct * min_atr:
        return None
    direction = "long" if roc > 0 else "short"
    sev = "critical" if vr >= 5 else "high" if vr >= 3 else "medium"
    return {"direction": direction, "severity": sev,
            "confidence": min(0.6 + (vr - 2) * 0.12, 0.9),
            "evidence": f"放量突破 量{vr:.1f}x 涨跌{roc:+.1f}%",
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

def _detect_double_bottom(df, atr, bullish: bool = True) -> bool:
    """检测最近 ~15 根 5m K 线内的双底/双顶形态"""
    if df is None or len(df) < 12:
        return False
    tail = df.tail(15).reset_index(drop=True)
    lows = tail["low"].values
    highs = tail["high"].values
    n = len(tail)
    if n < 5:
        return False
    tolerance = atr * 1.5
    mid_offset = atr * 0.5
    if bullish:
        min_idx1 = int(np.argmin(lows[:n-3]))
        min_val1 = lows[min_idx1]
        min_idx2 = int(np.argmin(lows[min_idx1+3:])) + min_idx1 + 3
        if min_idx2 >= n: min_idx2 = n - 1
        min_val2 = lows[min_idx2]
        if abs(min_val1 - min_val2) > tolerance:
            return False
        if min_val2 < min_val1:
            return False
        mid_range = highs[min(min_idx1, min_idx2):max(min_idx1, min_idx2)+1]
        if len(mid_range) == 0:
            return False
        mid_high = np.max(mid_range)
        if mid_high - max(min_val1, min_val2) < mid_offset:
            return False
        last_close = tail["close"].iloc[-1]
        last_open = tail["open"].iloc[-1]
        return last_close > last_open and last_close > tail["close"].iloc[-3]
    else:
        max_idx1 = int(np.argmax(highs[:n-3]))
        max_val1 = highs[max_idx1]
        max_idx2 = int(np.argmax(highs[max_idx1+3:])) + max_idx1 + 3
        if max_idx2 >= n: max_idx2 = n - 1
        max_val2 = highs[max_idx2]
        if abs(max_val1 - max_val2) > tolerance:
            return False
        if max_val2 > max_val1:
            return False
        mid_range = lows[min(max_idx1, max_idx2):max(max_idx1, max_idx2)+1]
        if len(mid_range) == 0:
            return False
        mid_low = np.min(mid_range)
        if min(max_val1, max_val2) - mid_low < mid_offset:
            return False
        last_close = tail["close"].iloc[-1]
        last_open = tail["open"].iloc[-1]
        return last_close < last_open and last_close < tail["close"].iloc[-3]


def _check_trend_pullback(state: SignalState) -> dict | None:
    ind_1h = state.ind_1h
    if not ind_1h:
        return None

    # 方向: 4H 宏观 bias 定 (回测: 4H bias 是唯一方向 edge)
    ind_4h = state.ind_4h
    ma_4h = ind_4h.get("ma_alignment", "neutral")
    adx_4h = ind_4h.get("adx", 0) or 0
    close_4h = ind_4h.get("close")
    ma20_4h = ind_4h.get("ma20")
    ma60_4h = ind_4h.get("ma60")
    above_ma20 = ma20_4h and close_4h and close_4h > ma20_4h
    above_ma60 = ma60_4h and close_4h and close_4h > ma60_4h
    if adx_4h >= 20 and ma_4h == "bullish":
        bias_dir = "long"
    elif adx_4h >= 20 and ma_4h == "bearish":
        bias_dir = "short"
    elif above_ma20 and above_ma60 and adx_4h >= 18:
        bias_dir = "long"
    elif not above_ma20 and not above_ma60 and adx_4h >= 18:
        bias_dir = "short"
    else:
        bias_dir = "neutral"
    if bias_dir == "neutral":
        return None

    df_1h = ind_1h.get("df")
    close_5m = state.ind.get("close")
    atr_5m = state.ind.get("atr") or 1
    df_5m = state.ind.get("df")
    if df_1h is None or len(df_1h) < 20 or not close_5m:
        return None

    pinbar = state.ind.get("pinbar")
    body_dir = state.ind.get("body_dir")
    body_pct = state.ind.get("body_pct", 0)
    rsi_5m = state.ind.get("rsi")
    ma20_1h = ind_1h.get("ma20")

    levels = find_swing_levels(df_1h, lookback=50)
    db_bull = _detect_double_bottom(df_5m, atr_5m, bullish=True)
    db_bear = _detect_double_bottom(df_5m, atr_5m, bullish=False)

    def _reversal_check(bullish: bool):
        if bullish:
            if pinbar == "bullish": return True, "Pinbar"
            if db_bull: return True, "双底"
            if rsi_5m is not None and rsi_5m < 40 and body_dir == "bullish" and body_pct > 0.3:
                return True, f"RSI{rsi_5m:.0f}+阳线"
        else:
            if pinbar == "bearish": return True, "Pinbar"
            if db_bear: return True, "双顶"
            if rsi_5m is not None and rsi_5m > 60 and body_dir == "bearish" and body_pct > 0.3:
                return True, f"RSI{rsi_5m:.0f}+阴线"
        return False, ""

    def _try_fire(bullish: bool, price_ref: float, evidence_ref: str, confidence: float):
        dist = abs(close_5m - price_ref) if price_ref else 0
        if dist > atr_5m * 1.2:
            return None
        ok, rev_type = _reversal_check(bullish)
        if not ok:
            return None
        sev = "critical" if dist <= atr_5m * 0.5 else "high"
        return {
            "direction": "long" if bullish else "short",
            "severity": sev,
            "confidence": confidence,
            "evidence": evidence_ref + f" {rev_type}",
        }

    # ── 1H 极点支撑/阻力 ──
    for lvl in levels:
        if lvl.touch_count < 2:
            continue
        if bias_dir == "long" and lvl.side == "support":
            r = _try_fire(True, lvl.price, f"趋势回调多 1H支撑{lvl.price:.5g}({lvl.touch_count}触)", 0.78 if pinbar == "bullish" else 0.70)
            if r: return r
        elif bias_dir == "short" and lvl.side == "resistance":
            r = _try_fire(False, lvl.price, f"趋势回调空 1H阻力{lvl.price:.5g}({lvl.touch_count}触)", 0.78 if pinbar == "bearish" else 0.70)
            if r: return r

    # ── 1H MA20 支撑/阻力 (无极点时替代) ──
    if ma20_1h and close_5m:
        if bias_dir == "long" and close_5m > ma20_1h:
            r = _try_fire(True, ma20_1h, f"趋势回调多 MA20{ma20_1h:.5g}", 0.65)
            if r: return r
        elif bias_dir == "short" and close_5m < ma20_1h:
            r = _try_fire(False, ma20_1h, f"趋势回调空 MA20{ma20_1h:.5g}", 0.65)
            if r: return r

    # ── 1H BB band 支撑/阻力 (无极点 & MA20不适用时替代) ──
    bbw_1h = ind_1h.get("bb_width")
    if ma20_1h and bbw_1h and close_5m and bbw_1h > 0:
        bb_lower = ma20_1h * (1 - bbw_1h / 200)
        bb_upper = ma20_1h * (1 + bbw_1h / 200)
        if bias_dir == "long" and close_5m > bb_lower:
            r = _try_fire(True, bb_lower, f"趋势回调多 BB下轨{bb_lower:.5g}", 0.60)
            if r: return r
        elif bias_dir == "short" and close_5m < bb_upper:
            r = _try_fire(False, bb_upper, f"趋势回调空 BB上轨{bb_upper:.5g}", 0.60)
            if r: return r

    return None


# ═══════════════════════════════════════════════════════════════════════
# 布林带反转信号 (3年1094天: 日线空/中性 + 1H BB上轨破轨做空 = 68%稳定)
# 高胜率小目标策略: TP=0.3-0.5×1H ATR, SL=0.5×1H ATR, W=24-48h
# ═══════════════════════════════════════════════════════════════════════

def _check_bb_reversal(state: SignalState) -> dict | None:
    ind_1h = state.ind_1h
    if not ind_1h:
        return None
    ma20 = ind_1h.get("ma20")
    bbw = ind_1h.get("bb_width")
    close = state.ind.get("close")
    atr_5m = state.ind.get("atr") or 1
    adx_1h = ind_1h.get("adx", 0) or 0
    if not ma20 or not bbw or not close or bbw <= 0:
        return None
    upper = ma20 * (1 + bbw / 200)
    if close <= upper:
        return None
    bias = state.params.get("bias", "neutral")
    # 3年验证环境: 日线空/中性 + 不做1H强趋势中的破轨
    if bias == "long":
        return None
    if adx_1h >= 30:
        return None
    # 破轨幅度 (ATR 单位) — 越远越极端越有效 (σ越大越强)
    overshoot = (close - upper) / max(atr_5m, 1e-9)
    sev = "critical" if overshoot >= 0.5 else "high"
    conf = 0.74 if overshoot >= 0.5 else 0.68
    return {
        "direction": "short",
        "severity": sev,
        "confidence": conf,
        "evidence": f"BB上轨破轨{overshoot:.1f}σ(回归)",
        "bb_overshoot": overshoot,
    }


# ═══════════════════════════════════════════════════════════════════════
# 信号酝酿 — 提前告知哪些信号接近触发
# ═══════════════════════════════════════════════════════════════════════

def _brew_breakout(state: SignalState) -> dict | None:
    df_sr = (state.ind_1h or {}).get("df") or state.ind.get("df")
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    vr = state.ind.get("volume_ratio") or 1
    adx = state.ind.get("adx", 0) or 0
    if df_sr is None or not close:
        return None
    levels = find_swing_levels(df_sr, lookback=50)
    near_sr = any(abs(close - l.price) <= atr * 2 and l.touch_count >= 2 for l in levels)
    conds = {"近S/R": near_sr, "VR≥1.5": vr >= 1.5, "ADX≥25": adx >= 25}
    met_list = [k for k, v in conds.items() if v]
    missing_list = [k for k, v in conds.items() if not v]
    if met_list:
        return {"signal_id": "breakout", "signal_name": "防线突破", "met": len(met_list), "total": 3, "missing": missing_list[:1], "detail": ""}
    return None


def _brew_rsi(state: SignalState) -> dict | None:
    rsi = state.ind.get("rsi")
    body_dir = state.ind.get("body_dir")
    if not rsi or not body_dir:
        return None
    rsi_near = (28 <= rsi <= 40 and body_dir == "bullish") or (60 <= rsi <= 72 and body_dir == "bearish")
    if not rsi_near:
        return None
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    df_sr = (state.ind_1h or {}).get("df")
    near_sr = False
    if df_sr is not None and close:
        levels = find_swing_levels(df_sr, lookback=50)
        near_sr = any(abs(close - l.price) <= atr * 2 and l.touch_count >= 2 for l in levels)
    tail = "→超卖" if rsi <= 40 else "→超买"
    return {"signal_id": "rsi_extreme", "signal_name": "RSI极值", "met": 1, "total": 2, "missing": [] if near_sr else ["近S/R"], "detail": f"RSI={rsi:.0f}{tail}"}


def _brew_volume(state: SignalState) -> dict | None:
    vr = state.ind.get("volume_ratio") or 1
    roc = state.ind.get("roc") or 0
    atr = state.ind.get("atr") or 1
    close = state.ind.get("close") or 1
    atr_pct = atr / close * 100
    chg = abs(roc)
    conds = {"VR≥2": vr >= 2.0, "ROC≥ATR": chg >= atr_pct * 0.3}
    met_list = [k for k, v in conds.items() if v]
    missing_list = [k for k, v in conds.items() if not v]
    if met_list:
        return {"signal_id": "volume_spike", "signal_name": "放量异动", "met": len(met_list), "total": 2, "missing": missing_list[:1], "detail": f"VR={vr:.1f}x ROC={roc:+.1f}%"}
    return None


def _brew_trend_pb(state: SignalState) -> dict | None:
    ind_1h = state.ind_1h
    if not ind_1h:
        return None
    ma_1h = ind_1h.get("ma_alignment", "neutral")
    adx_1h = ind_1h.get("adx", 0) or 0
    close = state.ind.get("close")
    atr = state.ind.get("atr") or 1
    if not close:
        return None
    trend_ok = (ma_1h in ("bullish", "bearish")) and adx_1h >= 18
    df_1h = ind_1h.get("df")
    near_sr = False
    sr_side = ""
    if df_1h is not None:
        levels = find_swing_levels(df_1h, lookback=50)
        for l in levels:
            if l.touch_count < 2: continue
            if ma_1h == "bullish" and l.side == "support" and abs(close - l.price) <= atr * 2:
                near_sr = True; sr_side = "支撑"; break
            if ma_1h == "bearish" and l.side == "resistance" and abs(close - l.price) <= atr * 2:
                near_sr = True; sr_side = "阻力"; break
    rsi = state.ind.get("rsi")
    body_dir = state.ind.get("body_dir")
    reversal = state.ind.get("pinbar") in ("bullish", "bearish")
    if not reversal and rsi and body_dir:
        if ma_1h == "bullish" and rsi < 45 and body_dir == "bullish":
            reversal = True
        elif ma_1h == "bearish" and rsi > 55 and body_dir == "bearish":
            reversal = True
    conds = {"1H趋势": trend_ok, "近防线": near_sr, "反转信号": reversal}
    met_list = [k for k, v in conds.items() if v]
    missing_list = [k for k, v in conds.items() if not v]
    if met_list:
        detail = f"1H{ma_1h}" if trend_ok else ""
        detail += f" {sr_side}" if near_sr else ""
        return {"signal_id": "trend_pullback", "signal_name": "趋势回调", "met": len(met_list), "total": 3, "missing": missing_list[:1], "detail": detail}
    return None


BREWERS = [_brew_breakout, _brew_trend_pb]


def check_brewing(state: SignalState) -> list[dict]:
    results = []
    for fn in BREWERS:
        try:
            r = fn(state)
            if r:
                results.append(r)
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════════════
# 信号注册表
# ═══════════════════════════════════════════════════════════════════════

SIGNALS: list[SignalDef] = [
    SignalDef("breakout",        "防线突破",   _check_breakout,        {},                    "突破", "trend"),
    SignalDef("fakeout",         "假突破反转", _check_fakeout,         {},                    "假破", "any"),
    SignalDef("retest",          "回踩确认",   _check_retest,          {},                    "回踩", "any"),
    SignalDef("trend_pullback",  "趋势回调",   _check_trend_pullback,  {},                    "回调", "trend"),
    SignalDef("bb_reversal",     "BB反转",     _check_bb_reversal,     {},                    "回归", "any"),
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


def _check_bb_explosion(state: SignalState) -> dict | None:
    """BB 收口后开始扩张 → 波动率释放, 大行情前兆"""
    bb_state = state.ind.get("bb_state", "")
    ts = state.ind.get("ttm_squeeze")
    fired = ts.get("is_fired", False) if ts else False
    squeeze_bars = ts.get("squeeze_bars", 0) if ts else 0
    if bb_state == "expanding" and (fired or squeeze_bars >= 8):
        evidence = f"BB爆破 {'TTM释放' if fired else f'压{squeeze_bars}根'}"
        return {"type": "bb_explosion", "evidence": evidence}
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
    WarningDef("bb_explosion",   "BB爆破",     _check_bb_explosion),
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
