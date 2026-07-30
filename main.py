import asyncio
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import yaml
from loguru import logger

from okx import OKXClient, KlineCache, Candle
from indicators import compute as compute_indicators
from signals import SIGNALS, SignalState, get_direction, get_regime, is_compressing, check_warnings, check_brewing
from notify import Feishu
from utils import setup_logging, start_health_server
from support_resistance import find_swing_levels, get_nearest_levels
from volume_profile import compute_volume_profile, get_nearest_nodes
from market_state import compute_market_state
from relative_strength import compute_rs

_US_STOCKS = {
    "AAPL/USDT:USDT", "MSFT/USDT:USDT", "GOOGL/USDT:USDT", "AMZN/USDT:USDT",
    "META/USDT:USDT", "NVDA/USDT:USDT", "TSLA/USDT:USDT", "AMD/USDT:USDT",
    "INTC/USDT:USDT", "MU/USDT:USDT", "ORCL/USDT:USDT", "PLTR/USDT:USDT",
    "MRVL/USDT:USDT", "TSM/USDT:USDT", "HOOD/USDT:USDT", "RKLB/USDT:USDT",
    "WDC/USDT:USDT", "QQQ/USDT:USDT", "SPY/USDT:USDT",
}

_METALS = {"XAU/USDT:USDT", "XAG/USDT:USDT", "XPT/USDT:USDT", "XPD/USDT:USDT", "XCU/USDT:USDT"}
_ETFS = {"EWY/USDT:USDT"}
_CAT_LABELS = {"stock": "美股", "metal": "贵金属", "etf": "ETF", "crypto": "Crypto"}


def _symbol_category(symbol: str) -> str:
    if symbol in _US_STOCKS:
        return "stock"
    if symbol in _METALS:
        return "metal"
    if symbol in _ETFS:
        return "etf"
    return "crypto"


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    try:
        with open("secrets.yaml") as f:
            secrets = yaml.safe_load(f)
    except FileNotFoundError:
        secrets = {}
    cfg["secrets"] = secrets
    return cfg


def load_secrets() -> dict:
    try:
        with open("secrets.yaml") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {"okx": {}, "feishu": {}}


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Alert:
    symbol: str
    timeframe: str
    signal_type: str
    signal_name: str
    regime: str
    direction: str
    severity: str
    confidence: float
    evidence: str
    details: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    checklist: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def tag(self) -> str:
        tags = {
            "breakout": "突破", "fakeout": "假破", "retest": "回踩",
            "rsi_extreme": "RSI", "volume_spike": "VOL",
            "trend_pullback": "回调",
        }
        return tags.get(self.signal_type, self.signal_type[:4])

    @property
    def name(self) -> str:
        names = {
            "breakout": "防线突破", "fakeout": "假突破反转", "retest": "回踩确认",
            "rsi_extreme": "RSI极值", "volume_spike": "放量异动",
            "trend_pullback": "趋势回调",
        }
        return names.get(self.signal_type, self.signal_type)

    @property
    def tf_role(self) -> str:
        return {"1h": "方向", "5m": "执行"}.get(self.timeframe, self.timeframe)


# =============================================================================
# 去重 + 置信度增强
# =============================================================================

class AlertFilter:
    def __init__(self, silence_minutes: int = 30, min_confidence: float = 0.65):
        self.silence_minutes = silence_minutes
        self.min_confidence = min_confidence
        self._last: dict[str, datetime] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(a: Alert) -> str:
        return f"{a.symbol}_{a.timeframe}_{a.signal_type}_{a.direction}"

    def _boost_confidence(self, a: Alert, ind: dict) -> float:
        conf = a.confidence
        d = a.details

        if a.signal_type == "breakout":
            vr = d.get("volume_ratio", 1)
            if vr >= 5: conf += 0.10
        elif a.signal_type == "fakeout":
            if d.get("touch_count", 0) >= 3: conf += 0.08
        elif a.signal_type == "rsi_extreme":
            rsi = ind.get("rsi", 50)
            if rsi <= 20 or rsi >= 80: conf += 0.08
        elif a.signal_type == "volume_spike":
            vr = d.get("volume_ratio", 1)
            if vr >= 5: conf += 0.10

        return min(conf, 1.0)

    def filter(self, alerts: list[Alert]) -> list[Alert]:
        result = []
        with self._lock:
            for a in alerts:
                if a.confidence < self.min_confidence:
                    continue
                key = self._key(a)
                last = self._last.get(key)
                if last and (datetime.now() - last).total_seconds() < self.silence_minutes * 60:
                    continue
                self._last[key] = datetime.now()
                result.append(a)
        deduped: dict[str, Alert] = {}
        for a in result:
            k = f"{a.symbol}_{a.signal_type}"
            if k not in deduped or a.confidence > deduped[k].confidence:
                deduped[k] = a
        return sorted(deduped.values(), key=lambda x: x.confidence, reverse=True)


# =============================================================================
# 排序
# =============================================================================

@dataclass
class SymbolRank:
    symbol: str
    timeframe: str
    direction: str
    regime: str
    score: float
    confidence: float
    signal_tags: list[str]
    reasons: list[str]


def rank_symbols(alerts: list[Alert], all_ind: dict[str, dict[str, dict]]) -> list[SymbolRank]:
    scores: dict[str, dict] = {}

    for a in alerts:
        s = scores.setdefault(a.symbol, {
            "direction": a.direction, "regime": a.regime, "timeframe": a.timeframe,
            "tags": [], "conf": 0, "momentum": 0.0, "compression": 0.0, "volume": 0.0, "reasons": [],
        })
        s["tags"].append(a.tag)
        s["conf"] = max(s["conf"], a.confidence)

        if a.signal_type == "volume_spike":
            vr = a.details.get("volume_ratio", 1)
            s["volume"] = max(s["volume"], min(vr / 3, 1))

        tf_data = all_ind.get(a.symbol, {})
        ind_1h = tf_data.get("1h", {})
        adx_v = ind_1h.get("adx", 0) or 0
        s["momentum"] = max(s["momentum"], min(adx_v / 40, 1))

    result = []
    for sym, s in scores.items():
        total = s["conf"] * 0.35 + s["momentum"] * 0.25 + s["compression"] * 0.20 + s["volume"] * 0.20
        reasons = []
        if s["momentum"] > 0.6: reasons.append("动量强")
        if s["volume"] > 0.6: reasons.append("量能足")
        if s["conf"] > 0.8: reasons.append("信号强")
        if len(s["tags"]) >= 2: reasons.append("多信号")
        if not reasons: reasons.append("综合评估")
        result.append(SymbolRank(
            symbol=sym, timeframe=s["timeframe"], direction=s["direction"],
            regime=s["regime"], score=total, confidence=s["conf"],
            signal_tags=list(dict.fromkeys(s["tags"])), reasons=reasons,
        ))
    return sorted(result, key=lambda x: x.score, reverse=True)


def _is_us_stock(symbol: str) -> bool:
    return _US_STOCKS is not None and symbol in _US_STOCKS


_SESSION_LABELS = [
    (0, 7, "🌙", "亚洲"),
    (7, 13, "🌅", "欧洲"),
    (13, 21, "🌇", "美盘"),
    (21, 24, "🌃", "低流动"),
]

def _current_session() -> str:
    now = datetime.utcnow()
    if now.weekday() >= 5:
        return "🌃周末"
    hour = now.hour + now.minute / 60.0
    for start, end, icon, label in _SESSION_LABELS:
        if start <= hour < end:
            return f"{icon}{label}"
    return "🌃低流动"


def _symbol_bias(tf_ind: dict) -> str:
    """4H MA20/MA60+ADX 稳定趋势判定, 1H MA排列仅做动量确认"""
    ind_4h = tf_ind.get("4h", {})
    ind_1h = tf_ind.get("1h", {})
    if not ind_4h or not ind_1h:
        return "neutral"

    close_4h = ind_4h.get("close")
    ma20_4h = ind_4h.get("ma20")
    ma60_4h = ind_4h.get("ma60")
    adx_4h = ind_4h.get("adx", 0) or 0
    adx_1h = ind_1h.get("adx", 0) or 0
    ma_1h = ind_1h.get("ma_alignment", "neutral")

    if close_4h is None:
        return "neutral"
    above_ma20 = ma20_4h and close_4h > ma20_4h
    above_ma60 = ma60_4h and close_4h > ma60_4h

    if above_ma20 and above_ma60 and adx_4h >= 20 and ma_1h == "bullish":
        return "long"
    if not above_ma20 and not above_ma60 and adx_4h >= 20 and ma_1h == "bearish":
        return "short"
    if above_ma20 and above_ma60 and ma_1h == "bullish" and adx_1h >= 20:
        return "long"
    if not above_ma20 and not above_ma60 and ma_1h == "bearish" and adx_1h >= 20:
        return "short"
    if above_ma60 and adx_4h >= 18:
        return "long"
    if not above_ma60 and adx_4h >= 18:
        return "short"
    return "neutral"


def _is_us_market_hours() -> bool:
    try:
        now = datetime.now()
        start = datetime(now.year, now.month, now.day, 13, 30)
        end = datetime(now.year, now.month, now.day, 21, 0)
        if now.weekday() >= 5:
            return False
        return start <= now <= end
    except Exception:
        return True


# =============================================================================
# 信号池 — 30min 持久化, 解决信号闪烁
# =============================================================================

# ═══════════════════════════════════════════════════════════════════════
# 信号池 — 入场窗口 + 持仓时限
# ═══════════════════════════════════════════════════════════════════════

ENTRY_WINDOWS: dict[str, int] = {
    "volume_spike": 15, "breakout": 30, "fakeout": 30,
    "retest": 60, "trend_pullback": 60, "rsi_extreme": 90,
}

POSITION_LIMITS: dict[str, int] = {
    "volume_spike": 60, "fakeout": 120, "breakout": 180,
    "retest": 180, "trend_pullback": 360, "rsi_extreme": 480,
}

DEFAULT_ENTRY = 30
DEFAULT_LIMIT = 180


class SignalPool:
    def __init__(self):
        self._pool: dict[str, Alert] = {}
        self._first_seen: dict[str, datetime] = {}
        self._fresh: set[str] = set()
        self._rs_prev: dict[str, float] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(a: Alert) -> str:
        return f"{a.symbol}|{a.signal_type}|{a.direction}"

    def update(self, alerts: list[Alert]):
        now = datetime.now()
        with self._lock:
            self._fresh = set()
            for a in alerts:
                k = self._key(a)
                if k not in self._first_seen:
                    self._first_seen[k] = now
                a.details["persist_since"] = now
                a.details["is_fresh"] = True
                self._pool[k] = a
                self._fresh.add(k)

            expired = []
            for k, a in self._pool.items():
                ttl_sec = ENTRY_WINDOWS.get(a.signal_type, DEFAULT_ENTRY) * 60
                age = (now - self._first_seen.get(k, a.details.get("persist_since", a.timestamp))).total_seconds()
                if age > ttl_sec:
                    expired.append(k)
            for k in expired:
                del self._pool[k]
                self._first_seen.pop(k, None)

    def get_active(self) -> list[Alert]:
        now = datetime.now()
        with self._lock:
            result = []
            for k, a in self._pool.items():
                a.details["is_fresh"] = k in self._fresh
                first = self._first_seen.get(k, a.timestamp)
                elapsed = (now - first).total_seconds() / 60.0
                entry_win = ENTRY_WINDOWS.get(a.signal_type, DEFAULT_ENTRY)
                pct = elapsed / entry_win * 100 if entry_win > 0 else 100
                a.details["elapsed_min"] = int(elapsed)
                a.details["entry_win_min"] = entry_win
                a.details["position_limit_min"] = POSITION_LIMITS.get(a.signal_type, DEFAULT_LIMIT)
                if pct <= 40:
                    a.details["entry_status"] = "🟢"
                elif pct <= 80:
                    a.details["entry_status"] = "🟡"
                else:
                    a.details["entry_status"] = "🔴"
                result.append(a)
            return sorted(result, key=lambda x: x.confidence, reverse=True)

    def get_rs_delta(self, symbol: str, current: float) -> float | None:
        with self._lock:
            prev = self._rs_prev.get(symbol, None)
            self._rs_prev[symbol] = current
            return round(current - prev, 1) if prev is not None else None

    def is_fresh(self, a: Alert) -> bool:
        return self._key(a) in self._fresh


# =============================================================================
# 飞书格式化
# =============================================================================

def fmt_short_alert(a: Alert) -> str:
    sym = a.symbol.replace("-USDT-SWAP", "/USDT").split(":")[0]
    dir_map = {"long": "多", "short": "空"}
    d = dir_map.get(a.direction, "")
    m = a.meta

    is_new = a.details.get("is_fresh", True)
    icon = "🔴" if is_new else "🟡"

    stars = "⭐⭐⭐" if a.confidence >= 0.80 else "⭐⭐" if a.confidence >= 0.70 else "⭐"

    elapsed = a.details.get("elapsed_min", 0)
    entry_win = a.details.get("entry_win_min", 30)
    pos_limit = a.details.get("position_limit_min", 180)
    entry_status = a.details.get("entry_status", "🟢") if elapsed > 0 else ""

    entry_timer = f" 🚪{entry_status}{elapsed}/{entry_win}m" if elapsed > 0 else ""
    pos_hint = f" ⏱{pos_limit}m" if pos_limit > 0 else ""

    rr = m.get("rr", "-")
    s = m.get("support", "-")
    r = m.get("resistance", "-")
    checks = " ".join([c for c in a.checklist if c.startswith("✓") or c.startswith("⚠")][:4])
    rs = m.get("rs", "")
    rs_delta = a.details.get("rs_delta")
    rs_d_str = f"↑{rs_delta:+.0f}" if rs_delta and rs_delta > 0 else f"↓{rs_delta:+.0f}" if rs_delta and rs_delta < 0 else ""
    rs_str = f" RS:{rs}{rs_d_str}" if rs else ""
    margin = m.get("margin", "")
    margin_str = f" 💰{margin}" if margin else ""
    persist = "" if is_new else " [持续中]"

    line = f"▸ {icon} {sym} {d} {a.name}{persist}  RR:{rr}  {stars}{entry_timer}{pos_hint}{margin_str}{rs_str}"
    line2 = f"    S:{s} R:{r}  |  {checks}"
    opt_entry = m.get("opt_entry", "")
    opt_rr = m.get("opt_rr", "")
    opt_line = f"\n   最优: {opt_entry}→RR:{opt_rr}" if opt_entry and opt_rr else ""
    vp_s = m.get("vp_support", "")
    vp_r = m.get("vp_resistance", "")
    vp_line = f"\n   量防线: S:{vp_s} R:{vp_r}" if vp_s or vp_r else ""
    tp_full = m.get("tp_full", "")
    tp_line = ""
    tp1 = m.get("tp1", "")
    tp2 = m.get("tp2", "")
    if tp1 and tp2:
        tp_line = f"\n   出场: TP1@{tp1} TP2@{tp2} Moon(30%)"
    elif tp_full:
        tp_line = f"\n   TP:{tp_full}"
    return f"{line}\n{line2}{opt_line}{vp_line}{tp_line}"


def format_warning_report(warnings: dict[str, list[dict]], scan_time: datetime) -> str | None:
    """格式化预警消息, 无预警返回 None"""
    if not warnings:
        return None

    time_str = scan_time.strftime("%H:%M")
    icon_map = {
        "near_sr": "🔵", "coiling": "⚡", "rsi_approaching": "🟠", "rs_moving": "🔵",
        "bb_explosion": "💥",
    }

    lines = [f"━━━ ⚡预警 {time_str} {_current_session()} ━━━"]
    count = 0
    for sym, items in warnings.items():
        sym_short = sym.replace("-USDT-SWAP", "/USDT").split(":")[0]
        parts = []
        for w in items:
            icon = icon_map.get(w.get("type", ""), "")
            parts.append(f"{icon}{w['evidence']}")
        if parts:
            lines.append(f"{sym_short}  {'  '.join(parts)}")
            count += 1
        if count >= 8:
            break

    return "\n".join(lines) if count > 0 else None


def format_consolidated_report(
    filtered: list[Alert],
    ranks: list[SymbolRank],
    total_alerts: int,
    symbol_count: int,
    scan_time: datetime,
    category_label: str = "",
) -> str:
    time_str = scan_time.strftime("%H:%M")
    cat_prefix = f"【{category_label}】" if category_label else ""

    quality = []
    for a in filtered:
        is_fresh = a.details.get("is_fresh", True)
        if is_fresh:
            rr_str = a.meta.get("rr", "0:1").split(":")[0]
            try:
                rr_val = float(rr_str)
            except ValueError:
                rr_val = 0
            if rr_val < 1.2:
                continue
            has_dir_fail = any(c == "✗方向" for c in a.checklist)
            if has_dir_fail:
                continue
            has_trend_fail = any(c == "✗趋势" for c in a.checklist)
            if has_trend_fail:
                continue
            has_pos_fail = any(c == "✗位置" for c in a.checklist)
            if has_pos_fail:
                continue
            has_stale = any(c == "⚠数据陈旧" for c in a.checklist)
            if has_stale:
                continue
        if _is_us_stock(a.symbol) and not _is_us_market_hours():
            continue

        quality.append(a)

    rank_order = {r.symbol: i for i, r in enumerate(ranks)}
    merged: dict[str, list[Alert]] = {}
    for a in quality:
        key = f"{a.symbol}|{a.signal_type}|{a.direction}"
        merged.setdefault(key, []).append(a)

    merged_list = []
    for key, alerts_in_group in merged.items():
        best = max(alerts_in_group, key=lambda x: x.confidence)
        best._merged_tfs = sorted(set(a.timeframe for a in alerts_in_group))
        merged_list.append(best)
    merged_list.sort(key=lambda a: (rank_order.get(a.symbol, 999), -a.confidence))

    longs = sum(1 for a in quality if a.direction == "long")
    shorts = sum(1 for a in quality if a.direction == "short")

    lines = [f"━━━ {cat_prefix}V2 扫描 {time_str} {_current_session()} ━━━",
             f"{symbol_count}币 · {total_alerts}信号 · 推送{len(merged_list)}条 · 多{longs}/空{shorts}"]

    if merged_list:
        lines.append("")
        shown = 0
        per_sym_count: dict[str, int] = {}
        for a in merged_list:
            sym_key = a.symbol
            if per_sym_count.get(sym_key, 0) >= 3:
                continue
            lines.append(fmt_short_alert(a))
            per_sym_count[sym_key] = per_sym_count.get(sym_key, 0) + 1
            shown += 1
            if shown >= 20:
                break

    if ranks:
        lines.append(f"\n━━━ TOP{min(10, len(ranks))} ━━━")
        for i, r in enumerate(ranks[:10], 1):
            sym = r.symbol.replace("-USDT-SWAP", "/USDT").split(":")[0]
            d = {"long": "多", "short": "空"}.get(r.direction, "")
            tags = "/".join(r.signal_tags[:3])
            reason = "/".join(r.reasons[:2])
            lines.append(f"  {i}. {sym} {d} {r.score:.0%}  {tags} · {reason}")

    return "\n".join(lines)


# =============================================================================
# 扫描逻辑
# =============================================================================

def get_symbols(okx: OKXClient, config: dict) -> list[str]:
    watchlist = config.get("watchlist", [])
    if watchlist:
        return watchlist
    return okx.get_top_symbols(config.get("top_n", 20))


def do_scan(
    symbols: list[str],
    cache: KlineCache,
    config: dict,
) -> tuple[list[Alert], dict[str, dict[str, dict]], dict]:
    timeframes = config["timeframes"]
    alerts: list[Alert] = []
    all_ind: dict[str, dict[str, dict]] = {}

    rs_cfg = config.get("rs", {})
    windows_cfg = rs_cfg.get("momentum_windows", {"5m": [5], "1h": [5], "4h": [5]})

    # ── 第一遍：收集所有 TF 的指标 + RS 数据 ──
    tf_close: dict[str, dict[str, float | None]] = {tf: {} for tf in timeframes}
    tf_prev_maps: dict[str, list[dict[str, float | None]]] = {tf: [] for tf in timeframes}
    tf_atr: dict[str, dict[str, float | None]] = {tf: {} for tf in timeframes}

    for sym in symbols:
        tf_ind: dict[str, dict] = {}
        for tf in timeframes:
            df = cache.get_df(sym, tf)
            if len(df) < 30:
                continue
            ind_params = config["indicators"].get(tf, config["indicators"]["5m"])
            ind = compute_indicators(df, ind_params)
            if ind:
                tf_ind[tf] = ind

        if not tf_ind:
            continue
        all_ind[sym] = tf_ind

        for tf in timeframes:
            ind = tf_ind.get(tf, {})
            if not ind:
                continue
            tf_close[tf][sym] = ind.get("close")
            tf_atr[tf][sym] = ind.get("atr_rs")
            df = ind.get("df")
            if df is None:
                continue
            lookbacks = windows_cfg.get(tf, [5])
            if len(tf_prev_maps[tf]) == 0:
                tf_prev_maps[tf] = [{} for _ in lookbacks]
            for i, w in enumerate(lookbacks):
                if len(df) > w:
                    tf_prev_maps[tf][i][sym] = df["close"].iloc[-w - 1]

    # ── 多周期 RS 计算 ──
    btc_symbol = "BTC/USDT:USDT"
    rs_by_tf: dict[str, dict[str, object]] = {}
    for tf in timeframes:
        lookbacks = windows_cfg.get(tf, [5])
        btc_close = tf_close[tf].get(btc_symbol)
        btc_atr = tf_atr[tf].get(btc_symbol)
        btc_prev_list = []
        for i in range(len(lookbacks)):
            if i < len(tf_prev_maps[tf]):
                btc_prev_list.append(tf_prev_maps[tf][i].get(btc_symbol))
            else:
                btc_prev_list.append(None)
        rs_by_tf[tf] = compute_rs(tf_close[tf], tf_prev_maps[tf], tf_atr[tf],
                                  btc_close, btc_prev_list, btc_atr, lookbacks)

    # ── 汇总每个 symbol 的多周期 RS ──
    rs_scores_all: dict[str, dict[str, dict]] = {}
    for sym in all_ind:
        rs_scores_all[sym] = {}
        for tf in timeframes:
            r = rs_by_tf[tf].get(sym)
            if r:
                rs_scores_all[sym][tf] = {"score": r.rs_score, "level": r.rs_level,
                                            "zscore": r.rs_zscore}

    # ── 第二遍：逐 TF 检查信号（1H/4H 仅做方向锚）──
    for sym, tf_ind in all_ind.items():
        for tf, ind in tf_ind.items():
            if tf in ("1h", "4h"):
                continue
            direction = get_direction(ind)
            regime = get_regime(ind)
            ind_1h = tf_ind.get("1h", {})
            state = SignalState(
                symbol=sym, timeframe=tf, ind=ind,
                regime=regime, direction=direction,
                rs_scores=rs_scores_all.get(sym, {}),
                ind_1h=ind_1h,
            )

            for sig_def in SIGNALS:
                adx_val = ind.get("adx", 0) or 0
                if sig_def.gate == "trend" and adx_val < 25:
                    continue
                if sig_def.gate == "range" and adx_val >= 20:
                    continue
                try:
                    state.params = sig_def.params
                    result = sig_def.check(state)
                    if result:
                        alert = Alert(
                            symbol=sym, timeframe=tf,
                            signal_type=sig_def.id, signal_name=sig_def.name,
                            regime=regime, direction=result.get("direction", direction),
                            severity=result.get("severity", "medium"),
                            confidence=result.get("confidence", 0.5),
                            evidence=result.get("evidence", ""),
                            details=result,
                        )
                        alerts.append(alert)
                except Exception as e:
                    logger.debug(f"Signal check error {sig_def.id} {sym}: {e}")

        # ── 富化 ──
        sym_alerts = [a for a in alerts if a.symbol == sym]
        rs_dict = rs_scores_all.get(sym, {})
        if rs_dict:
            for alert in sym_alerts:
                alert.details.setdefault("rs_scores", rs_dict)
        for alert in sym_alerts:
            _enrich_alert(alert, tf_ind, sym, sym_alerts)

    return alerts, all_ind, rs_scores_all


def _enrich_alert(alert: Alert, tf_ind: dict, sym: str, sym_alerts: list[Alert] | None = None):
    check = []

    # ── 信号共振 ──
    if sym_alerts:
        same_tf_dir = [a for a in sym_alerts
                       if a.timeframe == alert.timeframe
                       and a.direction == alert.direction
                       and a.signal_type != alert.signal_type]
        if len(same_tf_dir) >= 2:
            alert.confidence = min(alert.confidence * 1.15, 1.0)
            check.append(f"✓共振{len(same_tf_dir)+1}信")
        elif len(same_tf_dir) >= 1:
            alert.confidence = min(alert.confidence * 1.08, 1.0)

    # ── 1H 方向 ──
    ind_1h = tf_ind.get("1h", {})
    dir_1h = ind_1h.get("ma_alignment", "neutral") if ind_1h else "neutral"
    if ind_1h:
        alert.meta["1h_ma"] = "多头排列" if dir_1h == "bullish" else "空头排列" if dir_1h == "bearish" else "均线交叉"
        alert.meta["1h_adx"] = f"{ind_1h.get('adx', 0) or 0:.0f}"
        alert.meta["1h_bb"] = ind_1h.get("bb_state", "unknown")

    # ── 数据新鲜度 ──
    ind_5m = tf_ind.get("5m", {})
    df_5m = ind_5m.get("df")
    is_stale = False
    if df_5m is not None and len(df_5m) > 0:
        latest_ts = df_5m["timestamp"].iloc[-1]
        age_min = (datetime.now() - latest_ts).total_seconds() / 60
        if age_min > 15:
            is_stale = True
    if is_stale:
        alert.meta["stale_data"] = True
        check.append("⚠数据陈旧")

    # ── 1H 方向确认 ──
    sig_dir = alert.direction
    if dir_1h in ("bullish", "bearish"):
        if (dir_1h == "bullish" and sig_dir == "long") or (dir_1h == "bearish" and sig_dir == "short"):
            check.append("✓方向")
        else:
            check.append("✗方向")
    else:
        check.append("?方向")

    # ── 4H 大结构边界 (提前检测, 供趋势判断参考) ──
    ind_4h = tf_ind.get("4h", {})
    current_price = (ind_1h or tf_ind.get("5m", {})).get("close") if ind_1h else None
    atr_1h = (ind_1h or tf_ind.get("5m", {})).get("atr") or 1
    has_4h_boundary = False
    df_4h = ind_4h.get("df")
    if df_4h is not None and len(df_4h) >= 20 and current_price:
        levels_4h = find_swing_levels(df_4h, lookback=50)
        for lvl in levels_4h:
            if lvl.touch_count >= 2 and abs(current_price - lvl.price) <= atr_1h * 2.5:
                alert.meta["4h_boundary"] = f"4H{lvl.side}{lvl.price:.5g}"
                has_4h_boundary = True
                check.append("⚠4H边界")
                break

    # ── 宏观趋势: 4H+1H 方向一致性 ──
    bias = _symbol_bias(tf_ind)
    alert.details["macro_bias"] = bias
    if bias != "neutral":
        counter = (bias == "long" and sig_dir == "short") or (bias == "short" and sig_dir == "long")
        if counter:
            if alert.signal_type == "fakeout" or has_4h_boundary:
                check.append("✓反趋势" if not has_4h_boundary else "✓4H反转")
            else:
                check.append("✗趋势")
                alert.confidence = min(alert.confidence * 0.65, 0.60)
        else:
            check.append("✓趋势")

    # ── S/R + SL/TP/RR ──
    ind_base = ind_1h or tf_ind.get("5m", {})
    current_price = ind_base.get("close")
    atr_1h = ind_base.get("atr") or 1
    atr_5m = (tf_ind.get("5m") or {}).get("atr") or 1

    sr_info = {}
    df_1h = (ind_1h or {}).get("df")
    if df_1h is not None and current_price:
        levels = find_swing_levels(df_1h, lookback=50)
        support, resistance = get_nearest_levels(levels, current_price)

        if support:
            p = support.price
            sf = f"{p:.0f}" if p > 100 else f"{p:.1f}" if p > 1 else f"{p:.5f}"
            alert.meta["support"] = f"{sf}({support.strength},{support.touch_count}触)"
            sr_info["support"] = support
        if resistance:
            p = resistance.price
            sf = f"{p:.0f}" if p > 100 else f"{p:.1f}" if p > 1 else f"{p:.5f}"
            alert.meta["resistance"] = f"{sf}({resistance.strength},{resistance.touch_count}触)"
            sr_info["resistance"] = resistance

        pos_in_range = None
        if support and resistance and resistance.price > support.price:
            pos_in_range = (current_price - support.price) / (resistance.price - support.price)
        if alert.direction == "long" and support:
            if pos_in_range is not None and pos_in_range >= 0.7:
                check.append("✗位置")
            elif pos_in_range is not None and pos_in_range <= 0.3:
                check.append("✓近支撑")
            elif support.touch_count >= 2:
                check.append("✓有支撑")
            else:
                check.append("?无支撑")
        elif alert.direction == "short" and resistance:
            if pos_in_range is not None and pos_in_range <= 0.3:
                check.append("✗位置")
            elif pos_in_range is not None and pos_in_range >= 0.7:
                check.append("✓近阻力")
            elif resistance.touch_count >= 2:
                check.append("✓有阻力")
            else:
                check.append("?无阻力")
        else:
            check.append("?边界")

    # ── RSI 极值位置收紧: 必须靠近 1H S/R 才有效 ──
    if alert.signal_type == "rsi_extreme":
        dist_to_sr = None
        if alert.direction == "long" and "support" in sr_info:
            dist_to_sr = (current_price - sr_info["support"].price) / atr_1h if atr_1h > 0 else None
            if dist_to_sr is not None and dist_to_sr > 1.0:
                check.append("✗位置")
        elif alert.direction == "short" and "resistance" in sr_info:
            dist_to_sr = (sr_info["resistance"].price - current_price) / atr_1h if atr_1h > 0 else None
            if dist_to_sr is not None and dist_to_sr > 1.0:
                check.append("✗位置")
        else:
            check.append("✗位置")

    # ── SL/TP/RR ──
    entry_price = current_price or ind_base.get("close") or 0
    sl = tp = 0

    if alert.direction == "long":
        sl = sr_info["support"].price - atr_1h * 0.3 if "support" in sr_info else entry_price - atr_5m * 1.5
        tp = sr_info["resistance"].price if "resistance" in sr_info else entry_price + atr_1h * 2.5
        if tp <= sl or tp <= entry_price or (tp - entry_price) < (entry_price - sl) * 1.5:
            tp = entry_price + atr_1h * 2.5
    elif alert.direction == "short":
        sl = sr_info["resistance"].price + atr_1h * 0.3 if "resistance" in sr_info else entry_price + atr_5m * 1.5
        tp = sr_info["support"].price if "support" in sr_info else entry_price - atr_1h * 2.5
        if tp >= sl or tp >= entry_price or (entry_price - tp) < (sl - entry_price) * 1.5:
            tp = entry_price - atr_1h * 2.5
    else:
        sl = entry_price - atr_5m * 1.5
        tp = entry_price + atr_1h * 1.5

    sf = "{:.0f}" if entry_price > 100 else "{:.1f}" if entry_price > 1 else "{:.5f}"
    alert.meta["sl"] = sf.format(sl)
    alert.meta["tp"] = sf.format(tp)
    sl_dist = abs(entry_price - sl)
    tp_dist = abs(tp - entry_price)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0
    alert.meta["rr"] = f"{rr:.1f}:1"

    # ── 最优入场 ──
    opt_entry_price = None
    opt_rr_val = 0
    touches = 0
    if entry_price > 0 and atr_1h > 0:
        if alert.direction == "long" and "support" in sr_info:
            support_lvl = sr_info["support"]
            touches = support_lvl.touch_count
            opt_entry_price = support_lvl.price
            if opt_entry_price < entry_price:
                opt_sl = opt_entry_price - atr_1h * 0.3
                resistance_lvl = sr_info.get("resistance")
                opt_tp = resistance_lvl.price if resistance_lvl else opt_entry_price + atr_1h * 3.0
                if opt_tp > opt_entry_price and opt_sl < opt_entry_price:
                    opt_rr_val = (opt_tp - opt_entry_price) / (opt_entry_price - opt_sl)
        elif alert.direction == "short" and "resistance" in sr_info:
            resistance_lvl = sr_info["resistance"]
            touches = resistance_lvl.touch_count
            opt_entry_price = resistance_lvl.price
            if opt_entry_price > entry_price:
                opt_sl = opt_entry_price + atr_1h * 0.3
                support_lvl = sr_info.get("support")
                opt_tp = support_lvl.price if support_lvl else opt_entry_price - atr_1h * 3.0
                if opt_tp < opt_entry_price and opt_sl > opt_entry_price:
                    opt_rr_val = (opt_entry_price - opt_tp) / (opt_sl - opt_entry_price)

        if (opt_entry_price and abs(opt_entry_price - entry_price) <= atr_1h * 2
                and opt_rr_val > rr and opt_rr_val >= 2.0):
            sf_opt = "{:.0f}" if opt_entry_price > 100 else "{:.1f}" if opt_entry_price > 1 else "{:.5f}"
            alert.meta["opt_entry"] = sf_opt.format(opt_entry_price)
            alert.meta["opt_rr"] = f"{opt_rr_val:.1f}:1"
            if touches >= 3:
                check.append("🔵左侧挂单")
            elif touches >= 2:
                check.append("🟡右侧等K@防线")
            else:
                check.append("⚠防线弱·RR高")

    # ── 成交量分布防线 ──
    if df_1h is not None and current_price:
        try:
            vp = compute_volume_profile(df_1h)
            if vp:
                vp_nodes = get_nearest_nodes(vp, current_price)
                if vp_nodes["support"]:
                    p = vp_nodes["support"]["price"]
                    sp = f"{p:.0f}" if p > 100 else f"{p:.1f}" if p > 1 else f"{p:.5f}"
                    alert.meta["vp_support"] = f"{sp}(量节点)"
                if vp_nodes["resistance"]:
                    p = vp_nodes["resistance"]["price"]
                    sp = f"{p:.0f}" if p > 100 else f"{p:.1f}" if p > 1 else f"{p:.5f}"
                    alert.meta["vp_resistance"] = f"{sp}(量节点)"

                if touches == 2 and opt_entry_price:
                    upgraded = False
                    if alert.direction == "long" and "support" in sr_info:
                        if vp_nodes["support"]:
                            dist = abs(vp_nodes["support"]["price"] - sr_info["support"].price)
                            if dist <= atr_1h * 0.5:
                                if "🟡右侧等K@防线" in check:
                                    check.remove("🟡右侧等K@防线")
                                check.append("🔵左侧挂单(量升级)")
                                upgraded = True
                    elif alert.direction == "short" and "resistance" in sr_info:
                        if vp_nodes["resistance"]:
                            dist = abs(vp_nodes["resistance"]["price"] - sr_info["resistance"].price)
                            if dist <= atr_1h * 0.5:
                                if "🟡右侧等K@防线" in check:
                                    check.remove("🟡右侧等K@防线")
                                check.append("🔵左侧挂单(量升级)")
                                upgraded = True
                    if upgraded:
                        alert.details["vp_upgrade"] = True
        except Exception:
            pass

    # ── 1H 逆方向降级 ──
    if "✗方向" in check and dir_1h in ("bullish", "bearish"):
        has_strong_def = False
        if alert.direction == "long" and "support" in sr_info:
            sup_touches = sr_info["support"].touch_count
            vp_sup = alert.meta.get("vp_support")
            has_strong_def = sup_touches >= 3 or (sup_touches >= 2 and vp_sup)
        elif alert.direction == "short" and "resistance" in sr_info:
            res_touches = sr_info["resistance"].touch_count
            vp_res = alert.meta.get("vp_resistance")
            has_strong_def = res_touches >= 3 or (res_touches >= 2 and vp_res)
        if has_strong_def:
            check.remove("✗方向")
            check.append("⚠逆1H")

    # ── 仓位参考: 1×1H ATR 标准距离, 1%本金风险 ──
    try:
        risk_pct = 1
        leverage = 10
        risk_dist = atr_1h * 1.0
        if entry_price > 0 and risk_dist > 0:
            pos_pct = entry_price * risk_pct / (risk_dist * leverage)
            alert.meta["margin"] = f"参考{pos_pct:.0f}%({leverage}x 1ATR)" if pos_pct <= 100 else f"⚠波大参考{pos_pct:.0f}%"
    except Exception:
        pass

    if rr >= 1.5:
        check.append("✓盈亏比")
    elif rr >= 1.2:
        check.append("⚠盈亏比")
    else:
        check.append("✗盈亏比")

    # ── 压缩 ──
    comp_bars = ind_base.get("compression_bars", 0)
    if comp_bars >= 6:
        check.append(f"✓压缩{comp_bars}根")
    elif comp_bars >= 3:
        check.append(f"⚠压缩{comp_bars}根")

    # ── 压缩蓄力增强 ──
    if is_compressing(ind_base):
        alert.confidence = min(alert.confidence * 1.10, 1.0)
        alert.details["compression_boost"] = True
        check.append("⚡蓄力中")

    vr_5m = ind_base.get("volume_ratio") or 1
    if vr_5m >= 1.5:
        check.append(f"✓放量{vr_5m:.1f}x")

    # ── Pinbar 成型 ──
    pinbar_5m = ind_5m.get("pinbar")
    atr_5m = ind_5m.get("atr") or 1
    if pinbar_5m and current_price and atr_5m and vr_5m:
        near_sr = False
        if alert.direction == "long" and pinbar_5m == "bullish" and "support" in sr_info:
            dist = current_price - sr_info["support"].price
            near_sr = 0 <= dist <= atr_5m
        elif alert.direction == "short" and pinbar_5m == "bearish" and "resistance" in sr_info:
            dist = sr_info["resistance"].price - current_price
            near_sr = 0 <= dist <= atr_5m
        if near_sr and vr_5m >= 1.2:
            alert.severity = "critical"
            alert.details["setup"] = True
            check.append("✅成型·Pinbar")

    # ── 多周期 RS 独立评分 ──
    rs_dict = alert.details.get("rs_scores", {})
    if rs_dict:
        level_label = {"strong": "🟢", "mild_strong": "🔵", "neutral": "⚪", "mild_weak": "🟠", "weak": "🔴"}
        parts = []
        marks = []
        for tf_display, tf_key in [("5m", "5m"), ("1H", "1h"), ("4H", "4h")]:
            d = rs_dict.get(tf_key, {})
            if not d:
                continue
            score = d.get("score", 0)
            icon = level_label.get(d.get("level", ""), "")
            parts.append(f"{icon}{tf_display}{score:+.0f}")
            marks.append(score)

        alert.meta["rs"] = " ".join(parts) if parts else ""

        # 每周期独立增强: 方向与信号匹配的 TF 越多, 置信度越高
        agree_count = 0
        for s in marks:
            if (alert.direction == "long" and s > 0) or (alert.direction == "short" and s < 0):
                agree_count += 1
        if agree_count >= 3:
            alert.confidence = min(alert.confidence * 1.15, 1.0)
            check.append("✓RS三重")
        elif agree_count >= 2:
            alert.confidence = min(alert.confidence * 1.10, 1.0)
            check.append("✓RS双周期")
        elif agree_count >= 1:
            alert.confidence = min(alert.confidence * 1.05, 1.0)

    # ── 量能耗尽 ──
    body_pct_5m = ind_5m.get("body_pct", 1)
    vr_5m_val = ind_5m.get("volume_ratio") or 1
    if body_pct_5m < 0.3 and vr_5m_val >= 2.5:
        check.append("⚠量能耗尽")

    # ── ATR 移动止损 ──
    if entry_price > 0 and atr_1h > 0:
        sf = "{:.0f}" if entry_price > 100 else "{:.1f}" if entry_price > 1 else "{:.5f}"
        if alert.direction == "long":
            move_price = entry_price + atr_1h * 2
            if tp > entry_price and move_price < tp:
                alert.meta["trailing"] = f"移损@{sf.format(entry_price+atr_1h*0.5)}(+2ATR移保本)"
        elif alert.direction == "short":
            move_price = entry_price - atr_1h * 2
            if tp < entry_price and move_price > tp:
                alert.meta["trailing"] = f"移损@{sf.format(entry_price-atr_1h*0.5)}(+2ATR移保本)"

    # ── 分批止盈 ──
    if entry_price > 0 and atr_1h > 0 and tp != 0:
        sf = "{:.0f}" if entry_price > 100 else "{:.1f}" if entry_price > 1 else "{:.5f}"
        if alert.direction == "long" and tp > entry_price:
            tp1 = entry_price + atr_5m * 0.8
            if tp1 < tp:
                alert.meta["tp1"] = f"{sf.format(tp1)}(40%)"
                alert.meta["tp2"] = f"{sf.format(tp)}(30%)"
            else:
                alert.meta["tp_full"] = f"{sf.format(tp)}(70%)"
        elif alert.direction == "short" and tp < entry_price:
            tp1 = entry_price - atr_5m * 0.8
            if tp1 > tp:
                alert.meta["tp1"] = f"{sf.format(tp1)}(40%)"
                alert.meta["tp2"] = f"{sf.format(tp)}(30%)"
            else:
                alert.meta["tp_full"] = f"{sf.format(tp)}(70%)"

    # ── 拥挤度 ──
    funding = alert.details.get("btc_funding")
    if funding is not None:
        if funding > 0.05 and alert.direction == "long":
            check.append("⚠拥挤")
            alert.confidence = min(alert.confidence * 0.85, 1.0)
        elif funding < -0.03 and alert.direction == "short":
            check.append("⚠拥挤")
            alert.confidence = min(alert.confidence * 0.85, 1.0)
        elif funding < -0.03 and alert.direction == "long":
            check.append("✓反向")
            alert.confidence = min(alert.confidence * 1.10, 1.0)
        elif funding > 0.05 and alert.direction == "short":
            check.append("✓反向")
            alert.confidence = min(alert.confidence * 1.10, 1.0)

    # ── RS 分布 ──
    dispersion = alert.details.get("rs_dispersion", 0)
    if dispersion > 15:
        check.append("✓分化")
    elif 0 < dispersion < 8:
        alert.confidence = min(alert.confidence * 0.92, 1.0)

    # ── 波动率结构: 5m ATR(14) / 1H ATR(14) ──
    atr_5m_val = (tf_ind.get("5m") or {}).get("atr", 0) or 1
    atr_1h_val = (ind_1h or {}).get("atr", 0) or 1
    if atr_1h_val > 0 and atr_5m_val > 0:
        vol_ratio = atr_5m_val / atr_1h_val
        if vol_ratio > 0.6:
            check.append("⚡波动加速")

    # ── 震荡判定 ──
    adx_5m_val = (tf_ind.get("5m") or {}).get("adx", 0) or 0
    adx_1h_val = (ind_1h or {}).get("adx", 0) or 0
    if adx_5m_val < 18 and adx_1h_val < 18:
        check.append("📦震荡")

    # ── 4H边界 RSI反转 ──
    if has_4h_boundary and alert.signal_type == "rsi_extreme":
        bias_val = alert.details.get("macro_bias", "neutral")
        if bias_val != "neutral":
            counter = (bias_val == "long" and alert.direction == "short") or (bias_val == "short" and alert.direction == "long")
            if counter:
                check.append("✓4H反转")

    # ── BB 带区域 ──
    ma20_1h = (ind_1h or {}).get("ma20")
    bbw_1h = (ind_1h or {}).get("bb_width")
    if ma20_1h and bbw_1h and current_price and bbw_1h > 0:
        bb_upper = ma20_1h * (1 + bbw_1h / 200)
        bb_lower = ma20_1h * (1 - bbw_1h / 200)
        if current_price > bb_upper:
            check.append("📈BB上轨")
        elif current_price > ma20_1h:
            check.append("📈BB偏上")
        elif current_price < bb_lower:
            check.append("📉BB下轨")
        elif current_price < ma20_1h:
            check.append("📉BB偏下")

    alert.checklist = check


def apply_mtf_boost(alerts: list[Alert]):
    grouped: dict[str, list[Alert]] = {}
    for a in alerts:
        key = f"{a.symbol}_{a.signal_type}"
        grouped.setdefault(key, []).append(a)

    for key, group in grouped.items():
        tfs = set(a.timeframe for a in group)
        if len(tfs) >= 2:
            dirs = set(a.direction for a in group if a.direction != "neutral")
            if len(dirs) <= 1:
                boost = 1.10 if len(tfs) == 2 else 1.20
                for a in group:
                    a.confidence = min(a.confidence * boost, 1.0)
                    a.details["mtf_boost"] = True
                    a.details["mtf_timeframes"] = list(tfs)


# =============================================================================
# 异步主入口
# =============================================================================

async def async_main():
    config = load_config()
    secrets = load_secrets()
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("file", "logs/main.log"),
    )

    okx_cfg = secrets.get("okx", {})
    okx = OKXClient(
        api_key=okx_cfg.get("api_key", ""),
        api_secret=okx_cfg.get("api_secret", ""),
        passphrase=okx_cfg.get("passphrase", ""),
        testnet=okx_cfg.get("testnet", False),
    )
    cache = KlineCache(max_candles=500)

    feishu_cfg = secrets.get("feishu", secrets.get("feishu_signal", {}))
    feishu = Feishu(
        app_id=feishu_cfg.get("app_id", ""),
        app_secret=feishu_cfg.get("app_secret", ""),
        chat_id=feishu_cfg.get("chat_id", ""),
        webhook_url=feishu_cfg.get("webhook_url", ""),
    )

    pos_cfg = secrets.get("feishu_position", {})
    feishu_position = Feishu(
        app_id=pos_cfg.get("app_id", ""),
        app_secret=pos_cfg.get("app_secret", ""),
        chat_id=pos_cfg.get("chat_id", ""),
        webhook_url=pos_cfg.get("webhook_url", ""),
    )

    symbols = get_symbols(okx, config)
    timeframes = config["timeframes"]
    logger.info(f"Monitoring {len(symbols)} symbols x {len(timeframes)} timeframes")

    if config.get("health", {}).get("enabled", True):
        start_health_server(config["health"]["port"])

    # ── 并行预填 ──
    async def _parallel_prefill(okx_client, syms, tfs, kcache, limit=200):
        sem = asyncio.Semaphore(5)
        async def _fetch(s, t):
            async with sem:
                try:
                    bars = await asyncio.get_event_loop().run_in_executor(
                        None, okx_client.fetch_ohlcv, s, t, limit)
                    for bar in bars:
                        kcache.update(s, t, Candle(
                            timestamp=bar["timestamp"], open=bar["open"],
                            high=bar["high"], low=bar["low"],
                            close=bar["close"], volume=bar["volume"]))
                except Exception as e:
                    logger.warning(f"Prefill {s} {t}: {e}")
        await asyncio.gather(*[_fetch(s, t) for s in syms for t in tfs])

    await _parallel_prefill(okx, symbols, timeframes, cache)
    logger.info("Cache prefill done")

    def on_kline(sym: str, tf: str, candle: Candle):
        cache.update(sym, tf, candle)

    try:
        await okx.ws_connect(symbols, timeframes, on_kline)
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        return

    def history_thread():
        from history import download_and_compute
        download_and_compute(okx, symbols, timeframes, config.get("history", {}))

    threading.Thread(target=history_thread, daemon=True, name="history").start()

    pos_thread = None
    pos_cfg = config.get("position", {})
    if pos_cfg.get("enabled", True) and okx_cfg.get("api_key"):
        from position import PositionMonitor
        pm = PositionMonitor(config, feishu_position, secrets)
        pos_thread = threading.Thread(target=pm.run, daemon=True, name="position")
        pos_thread.start()

    alert_filter = AlertFilter(
        silence_minutes=config.get("alert", {}).get("dedup_minutes", 30),
        min_confidence=config.get("alert", {}).get("min_confidence", 0.65),
    )
    scan_count = 0
    warn_buf: dict[str, list[dict]] = {}
    brew_buf: dict[str, list[dict]] = {}
    first_scan = True
    signal_pool = SignalPool()

    def _wait_next_5m():
        nonlocal first_scan
        if first_scan:
            first_scan = False
            return 2.0
        now = datetime.utcnow()
        s = (now.minute % 5) * 60 + now.second + now.microsecond / 1e6
        return max(300 - s, 0.1)

    async def _refresh_cache_if_stale():
        sem = asyncio.Semaphore(5)
        refreshed = 0
        async def _fetch(s, t):
            nonlocal refreshed
            async with sem:
                try:
                    bars = await asyncio.get_event_loop().run_in_executor(
                        None, okx.fetch_ohlcv, s, t, 5)
                    for bar in bars:
                        cache.update(s, t, Candle(
                            timestamp=bar["timestamp"], open=bar["open"],
                            high=bar["high"], low=bar["low"],
                            close=bar["close"], volume=bar["volume"]))
                    if bars:
                        refreshed += 1
                except Exception:
                    pass
        await asyncio.gather(*[_fetch(s, t) for s in symbols for t in timeframes])
        if refreshed > 0:
            logger.debug(f"REST cache refresh: {refreshed} bars")

    while True:
        await asyncio.sleep(_wait_next_5m())
        scan_count += 1
        scan_start = datetime.now()
        try:
            await _refresh_cache_if_stale()
            alerts, all_ind, rs_scores_all = do_scan(symbols, cache, config)

            # ── 拥挤度: BTC 资金费率 ──
            btc_funding = None
            try:
                fr = okx.fetch_funding_rate("BTC/USDT:USDT")
                if fr:
                    btc_funding = fr["funding_rate"] * 100
            except Exception:
                pass

            # ── RS 分布: 标准差判断资金分化 ──
            rs_5m_vals = [rs_scores_all[s].get("5m", {}).get("score", 0) for s in rs_scores_all if s in rs_scores_all]
            rs_dispersion = round(float(np.std(rs_5m_vals)), 1) if len(rs_5m_vals) > 5 else 0.0

            # 注入到 alert.details
            for a in alerts:
                a.details["btc_funding"] = btc_funding
                a.details["rs_dispersion"] = rs_dispersion

            # ── RS 加速度 ──
            for a in alerts:
                rs5 = (a.details.get("rs_scores") or {}).get("5m", {}).get("score", 0)
                if rs5:
                    a.details["rs_delta"] = signal_pool.get_rs_delta(a.symbol, rs5)

            apply_mtf_boost(alerts)

            for a in alerts:
                ind = all_ind.get(a.symbol, {}).get(a.timeframe, {})
                a.confidence = alert_filter._boost_confidence(a, ind)

            filtered = alert_filter.filter(alerts)

            signal_pool.update(filtered)
            active = signal_pool.get_active()

            ranks = rank_symbols(alerts, all_ind) if active else []

            if active:
                ms = compute_market_state(all_ind)
                if ms["bias"] != "neutral":
                    bias_icon = "🔺" if ms["bias"] == "long" else "🔻"
                    bias_text = "做多" if ms["bias"] == "long" else "做空"
                    ms_line = f"━━━ 市场状态 {scan_start.strftime('%H:%M')} ━━━\n{ms['desc']}\n→ {bias_icon}倾向{bias_text}({ms['confidence']}%) {ms['reason']}"
                    if btc_funding is not None:
                        crowd = "🟢安全" if abs(btc_funding) < 0.03 else "🟠拥挤" if abs(btc_funding) < 0.07 else "🔴极端"
                        ms_line += f" · 费率:{btc_funding:+.3f}%{crowd}"
                    if rs_dispersion > 5:
                        ms_line += f" · RS分化:σ={rs_dispersion:.0f}"
                    feishu.send(ms_line)

                buckets: dict[str, list[Alert]] = {}
                for a in active:
                    cat = _symbol_category(a.symbol)
                    buckets.setdefault(cat, []).append(a)

                for cat, cat_alerts in buckets.items():
                    cat_label = _CAT_LABELS.get(cat, cat)
                    cat_ranks = [r for r in ranks if _symbol_category(r.symbol) == cat]
                    report = format_consolidated_report(
                        cat_alerts, cat_ranks, len(alerts), len(symbols), scan_start,
                        category_label=cat_label)
                    feishu.send(report)
                new_count = sum(1 for a in active if a.details.get("is_fresh"))
                persist_count = len(active) - new_count
                logger.info(f"Scan #{scan_count}: {len(active)} alerts ({new_count} new + {persist_count} persisted), {len(ranks)} ranked ({'/'.join(f'{k}:{len(v)}' for k,v in buckets.items())})")

                # ── 酝酿报告 ──
                if brew_buf:
                    brew_lines = [f"\n📋 酝酿中:"]
                    count = 0
                    for sym, items in brew_buf.items():
                        sym_short = sym.replace("-USDT-SWAP", "/USDT").split(":")[0]
                        for b in items:
                            missing_str = " 缺"+",".join(b.get("missing", [])) if b.get("missing") else ""
                            detail = " "+b.get("detail", "") if b.get("detail") else ""
                            brew_lines.append(f"{sym_short}  ⚠{b['signal_name']}({b['met']}/{b['total']}){missing_str}{detail}")
                            count += 1
                            if count >= 5: break
                        if count >= 5: break
                    feishu.send("\n".join(brew_lines))
                    brew_buf.clear()
            else:
                logger.info(f"Scan #{scan_count}: 0 push alerts (raw {len(alerts)} signals scanned)")

            # ── 预警收集 ──
            for sym, tf_ind in all_ind.items():
                ind_5m = tf_ind.get("5m", {})
                if not ind_5m:
                    continue
                state = SignalState(
                    symbol=sym, timeframe="5m", ind=ind_5m,
                    regime=get_regime(ind_5m), direction=get_direction(ind_5m),
                    rs_scores=rs_scores_all.get(sym, {}),
                    ind_1h=tf_ind.get("1h", {}),
                )
                ws = check_warnings(state)
                if ws:
                    warn_buf[sym] = ws
                bs = check_brewing(state)
                if bs:
                    brew_buf.setdefault(sym, []).extend(bs)

            # ── 30m 整点推送预警 (首次扫描立即推) ──
            now = datetime.utcnow()
            if now.minute % 30 == 0 or scan_count == 1:
                w_report = format_warning_report(warn_buf, now)
                if w_report:
                    feishu.send(w_report)
                    logger.info(f"Warnings pushed: {sum(len(v) for v in warn_buf.values())} items")
                warn_buf.clear()

        except Exception as e:
            logger.error(f"Scan #{scan_count} error: {e}")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(sig, frame):
        logger.info(f"Signal {sig} received, shutting down...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(async_main())
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
