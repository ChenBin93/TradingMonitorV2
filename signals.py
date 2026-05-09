# 信号定义 + 检测函数

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SignalState:
    symbol: str
    timeframe: str
    ind: dict          # indicators.compute() 结果
    bbw_rank: float | None = None
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


# ---- 检测函数 ----------------------------------------------------------------

def _check_bb_squeeze(state: SignalState) -> dict | None:
    bb = state.ind.get("bb_width")
    rank = state.bbw_rank
    if not bb or rank is None:
        return None
    threshold = state.params.get("threshold", 25)
    if rank <= threshold:
        return {
            "direction": state.direction,
            "severity": "critical" if rank <= 10 else "high",
            "confidence": 0.7,
            "evidence": f"压缩位{rank:.0f}%",
        }


def _check_rsi_extreme(state: SignalState) -> dict | None:
    rsi = state.ind.get("rsi")
    if not rsi:
        return None
    if rsi <= state.params.get("oversold", 30):
        return {"direction": "long", "severity": "critical" if rsi <= 25 else "high",
                "confidence": 0.8, "evidence": f"RSI={rsi:.0f}"}
    if rsi >= state.params.get("overbot", 70):
        return {"direction": "short", "severity": "critical" if rsi >= 80 else "high",
                "confidence": 0.8, "evidence": f"RSI={rsi:.0f}"}
    return None


def _check_ma_converge(state: SignalState) -> dict | None:
    mc = state.ind.get("ma_converge")
    if not mc:
        return None
    threshold = state.params.get("threshold", 0.5)
    if mc <= threshold:
        return {"direction": state.direction,
                "severity": "critical" if mc <= 0.3 else "high",
                "confidence": 0.6, "evidence": f"汇聚度{mc:.2f}"}


def _check_macd_cross(state: SignalState) -> dict | None:
    cross = state.ind.get("macd_cross")
    if not cross:
        return None
    hist = state.ind.get("macd_hist") or 0
    return {"direction": "long" if cross == "golden" else "short",
            "severity": "high", "confidence": 0.7 if abs(hist) > 0.001 else 0.55,
            "evidence": "金叉" if cross == "golden" else "死叉"}


def _check_volume_spike(state: SignalState) -> dict | None:
    vr = state.ind.get("volume_ratio")
    threshold = state.params.get("threshold", 3.0)
    if vr and vr >= threshold:
        return {"direction": state.direction,
                "severity": "critical" if vr >= 5 else "high",
                "confidence": min(0.6 + (vr - 2) * 0.1, 0.9),
                "evidence": f"量{vr:.1f}x"}


def _check_ttm_squeeze(state: SignalState) -> dict | None:
    ts = state.ind.get("ttm_squeeze")
    if not ts:
        return None
    bars = ts.get("squeeze_bars", 0)
    fired = ts.get("is_fired", False)
    if bars >= 3 or fired:
        return {"direction": ts.get("direction") or state.direction,
                "severity": "critical" if bars >= 8 or fired else "high",
                "confidence": 0.75 if fired else 0.65,
                "evidence": f"压缩{bars}根{'释放' if fired else ''}"}


def _check_rsi_divergence(state: SignalState) -> dict | None:
    rd = state.ind.get("rsi_divergence")
    if not rd or not rd.get("divergence"):
        return None
    div = rd["divergence"]
    dist = rd.get("price_distance_pct", 0)
    return {"direction": "long" if div == "bullish" else "short",
            "severity": "critical" if dist >= 5 else "high",
            "confidence": 0.8,
            "evidence": f"{'底背离' if div == 'bullish' else '顶背离'} {dist:.1f}%"}


def _check_macd_divergence(state: SignalState) -> dict | None:
    md = state.ind.get("macd_divergence")
    if not md or not md.get("divergence"):
        return None
    div = md["divergence"]
    dist = md.get("price_distance_pct", 0)
    return {"direction": "long" if div == "bullish" else "short",
            "severity": "critical" if dist >= 5 else "high",
            "confidence": 0.75,
            "evidence": f"{'底背离' if div == 'bullish' else '顶背离'} {dist:.1f}%"}


def _check_volume_breakout(state: SignalState) -> dict | None:
    vb = state.ind.get("volume_breakout")
    if not vb or not vb.get("confirmed"):
        return None
    vr = vb.get("vol_ratio", 0)
    chg = vb.get("price_change_pct", 0)
    return {"direction": state.direction,
            "severity": "critical" if vr >= 3 else "high",
            "confidence": min(0.7 + (vr - 1.5) * 0.15, 0.95),
            "evidence": f"量{vr:.1f}x 涨跌{chg:.1f}%"}


# ---- 信号注册表 --------------------------------------------------------------

SIGNALS: list[SignalDef] = [
    SignalDef("bb_squeeze",       "BB压缩",   _check_bb_squeeze,       {"threshold": 25}, "BB"),
    SignalDef("rsi_extreme",      "RSI极值",  _check_rsi_extreme,      {"oversold": 30, "overbot": 70}, "RSI"),
    SignalDef("ma_converge",      "MA汇聚",   _check_ma_converge,      {"threshold": 0.5}, "MA"),
    SignalDef("macd_cross",       "MACD交叉", _check_macd_cross,       {}, "MACD"),
    SignalDef("volume_spike",     "量能爆发", _check_volume_spike,     {"threshold": 3.0}, "VOL"),
    SignalDef("ttm_squeeze",      "TTM压缩",  _check_ttm_squeeze,      {"min_bars": 5}, "TTM"),
    SignalDef("rsi_divergence",   "RSI背离",  _check_rsi_divergence,   {}, "RSI背"),
    SignalDef("macd_divergence",  "MACD背离", _check_macd_divergence,  {}, "MACD背"),
    SignalDef("volume_breakout",  "量价突破", _check_volume_breakout,  {"threshold": 1.5}, "突破"),
]


# ---- 方向判断 ----------------------------------------------------------------

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
