import asyncio
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import yaml
from loguru import logger

from okx import OKXClient, KlineCache, Candle
from indicators import compute as compute_indicators
from signals import SIGNALS, SignalState, get_direction, get_regime, is_compressing, check_warnings, check_brewing
from notify import Feishu
from chart import make_chart
from utils import setup_logging, start_health_server, fmt_price
from support_resistance import find_swing_levels, find_recent_extremes, get_nearest_levels
from volume_profile import compute_volume_profile, get_nearest_nodes
from market_state import compute_market_state, scene_of, SCENE_WR, SCENE_BOOST
from relative_strength import compute_rs
import market_structure

BJ_TZ = timezone(timedelta(hours=8))


def _bj_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(BJ_TZ)


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
            "trend_pullback": "回调", "bb_reversal": "回归",
        }
        return tags.get(self.signal_type, self.signal_type[:4])

    @property
    def name(self) -> str:
        names = {
            "breakout": "防线突破", "fakeout": "假突破反转", "retest": "回踩确认",
            "trend_pullback": "趋势回调", "bb_reversal": "BB反转",
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
    (0, 5, "🌙", "美盘"),
    (5, 8, "🌃", "低流动"),
    (8, 15, "🌅", "亚洲"),
    (15, 21, "🌇", "欧洲"),
]

def _current_session() -> str:
    now = _bj_now()
    if now.weekday() >= 5:
        return "🌃周末"
    hour = now.hour + now.minute / 60.0
    if hour >= 21:
        return "🌙美盘"
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
        now = datetime.now(timezone.utc)
        start = datetime(now.year, now.month, now.day, 13, 30, tzinfo=timezone.utc)
        end = datetime(now.year, now.month, now.day, 21, 0, tzinfo=timezone.utc)
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
    "breakout": 30, "fakeout": 30,
    "retest": 60, "trend_pullback": 60,
}

POSITION_LIMITS: dict[str, int] = {
    "fakeout": 120, "breakout": 180,
    "retest": 180, "trend_pullback": 360,
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

    # ── 宏观趋势标记 (修复后真实 edge: bias一致 +1.9pp) ──
    bias = a.details.get("macro_bias", "")
    bias_icons = {"long": "📈4H多", "short": "📉4H空", "neutral": "➖无趋势"}
    bias_str = f" {bias_icons.get(bias, '')}" if bias else ""

    # ── 场景标记 (3年验证胜率) ──
    scene = m.get("scene", "")
    scene_str = f" {scene}" if scene else ""

    # ── 4大币 5M反转K执行状态 ──
    exec_str = f" [5M{m.get('exec', '')}]" if m.get("exec") else ""

    # ── 多空力量 (用户方法论: 机会前提) ──
    power_str = f" ⚔{m['power']}" if m.get("power") else ""

    line = f"▸ {icon} {sym} {d} {a.name}{persist}{bias_str}{scene_str}{exec_str}  RR:{rr}  {stars}{entry_timer}{pos_hint}{margin_str}{rs_str}{power_str}"
    line2 = f"    S:{s} R:{r}  |  {checks}"
    opt_entry = m.get("opt_entry", "")
    opt_rr = m.get("opt_rr", "")
    opt_dist = m.get("opt_dist", "")
    opt_line = f"\n   挂单: {opt_entry}→RR:{opt_rr}({opt_dist})" if opt_entry and opt_rr and opt_dist else ""
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

    # ── 场景统计 (3年验证: 顺日逆时最优/逆势低期望) ──
    from collections import Counter
    scene_cnt = Counter(a.details.get("scene", "") for a in quality)
    ep_cnt = scene_cnt.get("episode_long", 0) + scene_cnt.get("episode_short", 0)
    ct_cnt = scene_cnt.get("counter_long", 0) + scene_cnt.get("counter_short", 0)
    scene_line = ""
    if ep_cnt or ct_cnt:
        parts = []
        if ep_cnt:
            parts.append(f"顺日逆时{ep_cnt}")
        if scene_cnt.get("follow_long", 0) + scene_cnt.get("follow_short", 0):
            parts.append(f"顺势{scene_cnt.get('follow_long',0)+scene_cnt.get('follow_short',0)}")
        if ct_cnt:
            parts.append(f"逆势{ct_cnt}")
        scene_line = f"\n场景: " + " · ".join(parts)

    lines = [f"━━━ {cat_prefix}V2 扫描 {time_str} {_current_session()} ━━━",
             f"{symbol_count}币 · {total_alerts}信号 · 推送{len(merged_list)}条 · 多{longs}/空{shorts}{scene_line}"]

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


def _send_warning_chart(cache: KlineCache, sym: str,
                        item: dict | None = None, tf: str = "1h"):
    """为预警标的生成 K线图 → 返回 (sym, path); 失败返回 None (静默)

    图表内容: 顶部信息栏 (4H方向/日线一致性/统计状态/关键位距离)
             + 1h K线 + BB + MA60 + 1h/4h 关键位 + 预警标注 + 成交量
    """
    try:
        df = cache.get_df(sym, tf)
        if df is None or len(df) < 60:
            logger.warning(f"Chart {sym}: {tf} 数据不足")
            return
        from market_phase import _atr_series
        from key_levels import detect_levels, bollinger_bands
        from chart import fused_levels
        d = df.copy()
        if "timestamp" in d.columns:
            d = d.sort_values("timestamp").reset_index(drop=True)
        atr = _atr_series(d)
        price_now = float(d["close"].iloc[-1])
        atr_now = float(atr[-1]) or 1.0
        # 融合关键位: 成交量分布 HVN + swing 极点 (业界推荐, 资金聚集处)
        levels = fused_levels(d, price_now, atr_now)
        bb = bollinger_bands(d)
        ma60 = d["close"].rolling(60, min_periods=60).mean()

        # ── 15M 短线面板 (优先 cache 15m, 否则 5m 聚合) ──
        df15_raw = None
        levels_15m_chart = []
        try:
            df15_raw = cache.get_df(sym, "15m")
            if df15_raw is None or len(df15_raw) < 60:
                # 5m → 15m 聚合 (3根5m = 1根15m, 已收盘对齐)
                df5 = cache.get_df(sym, "5m")
                if df5 is not None and len(df5) >= 180:
                    d5 = df5.copy()
                    if "timestamp" in d5.columns:
                        d5 = d5.set_index("timestamp")
                    d5 = d5.sort_index()
                    df15_raw = d5.resample("15min").agg({
                        "open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum",
                    }).dropna(subset=["close"]).reset_index()
            if df15_raw is not None and len(df15_raw) >= 60:
                d15 = df15_raw.copy()
                if "timestamp" in d15.columns:
                    d15 = d15.sort_values("timestamp").reset_index(drop=True)
                atr15 = _atr_series(d15)
                atr15_now = float(atr15[-1]) or 1.0
                levels_15m_chart = fused_levels(d15, price_now, atr15_now,
                                                max_dist_pct=0.04)
        except Exception:
            pass

        # ── 4H 大周期 (辅助面板 + 关键位) ──
        df4_raw = None
        levels_4h_chart = []
        try:
            df4_raw = cache.get_df(sym, "4h")
            if df4_raw is not None and len(df4_raw) >= 60:
                d4 = df4_raw.copy()
                if "timestamp" in d4.columns:
                    d4 = d4.sort_values("timestamp").reset_index(drop=True)
                atr4 = _atr_series(d4)
                atr4_now = float(atr4[-1]) or 1.0
                levels_4h_chart = fused_levels(d4, price_now, atr4_now,
                                               max_dist_pct=0.06)
        except Exception:
            pass

        # ── 顶部信息栏 (来自 do_scan 的 item) ──
        info = {}
        dow = (item or {}).get("dow") or {}
        dd = (item or {}).get("dow_daily") or {}
        stats = (item or {}).get("stats") or {}
        dists = (item or {}).get("dists") or {}
        info["dow4h"] = dow.get("seg_dir", "")
        info["dow4h_age"] = dow.get("seg_age")
        info["dow_daily"] = dd.get("seg_dir", "")
        info["cons"] = (item or {}).get("cons", "")
        st = stats.get("4h") or stats.get("1h") or {}
        info["stat"] = st.get("label", "")
        d4h = dists.get("4h") or {}
        d1h = dists.get("1h") or {}
        info["dist4h"] = f"{d4h.get('sup_dist_atr', '')}/{d4h.get('res_dist_atr', '')}"
        info["dist1h"] = f"{d1h.get('sup_dist_atr', '')}/{d1h.get('res_dist_atr', '')}"

        # 预警标注: 全部 warns 按 tf 分配 (4H/1H/15M 面板各自标注)
        alerts = []
        warns = (item or {}).get("warns") or []
        for w in warns:
            w_tf = w.get("tf", tf)
            # 无 price 的预警 (布林/波动启动) 用对应面板最新收盘价
            px = w.get("price")
            if px is None:
                if w_tf == "4h" and df4_raw is not None:
                    px = float(df4_raw["close"].iloc[-1])
                elif w_tf == "15m" and df15_raw is not None:
                    px = float(df15_raw["close"].iloc[-1])
                else:
                    px = float(d["close"].iloc[-1])
            alerts.append({
                "price": float(px), "level": w.get("level", "L2"),
                "tf": w_tf, "text": f"{w.get('desc', '')[:18]}",
            })

        # 关键位标注: 1h 支撑/阻力 (带 band) + 15M/4H 辅助面板
        path = make_chart(d, sym, tf, levels=levels,
                          alerts=alerts, bb=bb, ma60=ma60, info=info,
                          df_15m=df15_raw, levels_15m_chart=levels_15m_chart,
                          df_4h=df4_raw, levels_4h_chart=levels_4h_chart)
        if path:
            return sym, path
    except Exception as e:
        logger.warning(f"Chart {sym} failed: {e}")
    return None


def do_scan(
    symbols: list[str],
    cache: KlineCache,
    config: dict,
) -> dict[str, dict]:
    """新版扫描 (2026-08-04 大改造): 三柱 (关键位/波动/趋势状态) → 预警

    无方向预测 (用户确认): 系统只描述"价格 vs 关键位 / 波动状态 /
    趋势段状态", 方向由人判断。
    """
    from market_phase import _atr_series
    from key_levels import (detect_levels, bollinger_bands, level_relation,
                            recent_touches, break_nearest)
    from volatility_state import vol_z, vol_state, squeeze, vol_start
    from early_warning import compose as compose_warns
    import market_structure as ms_mod

    warnings: dict[str, dict] = {}

    def _clean_df(df):
        """防御: 时间戳排序 + 去重 (KlineCache 曾有乱序/重复污染) — 返回 (df, dup_count, last_ts)"""
        if df is None or len(df) == 0:
            return df, 0, None
        d = df.copy()
        if "timestamp" in d.columns:
            d = d.sort_values("timestamp")
            dup = int(d["timestamp"].duplicated().sum())
            d = d.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        else:
            dup = 0
        return d, dup, (d["timestamp"].iloc[-1] if len(d) and "timestamp" in d.columns else None)

    for sym in symbols:
        try:
            df_1h = cache.get_df(sym, "1h")
            df_4h = cache.get_df(sym, "4h")
            tfs: dict[str, dict] = {}
            diag: dict[str, str] = {}
            for tf, df in (("1h", df_1h), ("4h", df_4h)):
                df, dup, last_ts = _clean_df(df)
                if df is None or len(df) < 60:
                    continue
                atr = _atr_series(df)
                price = float(df["close"].values[-1])
                atr_now = float(atr[-1]) or 1.0
                n = len(df)
                levels = detect_levels(df, atr)
                ma, up, lo_ = bollinger_bands(df)
                z = vol_z(atr)
                vstate = vol_state(z)
                bbw = np.abs(up - lo_) / np.maximum(np.abs(ma), 1e-9)
                rel = level_relation(price, levels, atr_now, n - 1)
                tfs[tf] = {
                    "price": price,
                    "atr": atr_now,
                    "rel": rel,
                    "touches": recent_touches(levels, df["high"].values,
                                              df["low"].values, n - 1),
                    "breaks": break_nearest(rel, df["close"].values, atr, n - 1),
                    "bb": (float(ma[-1]), float(up[-1]), float(lo_[-1])),
                    "vol_state": vstate,
                    "squeeze": bool(squeeze(bbw)),
                    "vol_start": bool(vol_start(atr)),
                }
                diag[tf] = (f"p{price:.5g} bb({float(ma[-1]):.5g},{float(up[-1]):.5g},"
                            f"{float(lo_[-1]):.5g}) len{len(df)} dup{dup} "
                            f"last{str(last_ts)[:19]}")
            if df_4h is not None and len(df_4h) >= 60:
                try:
                    daily = ms_mod.resample_daily(df_4h)
                    if len(daily) >= 30:
                        datr = _atr_series(daily)
                        dprice = float(daily["close"].values[-1])
                        datr_now = float(datr[-1]) or 1.0
                        dn = len(daily)
                        _dma, _dup, _dlo = bollinger_bands(daily)
                        tfs["日线"] = {
                            "price": dprice,
                            "atr": datr_now,
                            "rel": level_relation(dprice, detect_levels(daily, datr),
                                                   datr_now, dn - 1),
                            "touches": [],
                            "breaks": [],
                            "bb": (float(_dma[-1]), float(_dup[-1]), float(_dlo[-1])),
                            "vol_state": vol_state(vol_z(datr)),
                            "squeeze": False,
                            "vol_start": False,
                        }
                except Exception:
                    pass
            if not tfs:
                continue
            dow4h = ms_mod.compute_dow_info(df_4h)
            warns = compose_warns(sym, tfs)
            if warns:
                # 推荐标的多周期状态 (仅对预警标的计算 — 性能)
                dow_h1 = ms_mod.compute_dow_info(df_1h)
                daily_df = None
                if df_4h is not None and len(df_4h) >= 60:
                    try:
                        daily_df = ms_mod.resample_daily(df_4h)
                    except Exception:
                        daily_df = None
                dow_daily = (ms_mod.compute_dow_info(daily_df)
                             if daily_df is not None and len(daily_df) >= 30 else {})
                stats = {}
                for tf, dfx in (("日线", daily_df), ("4h", df_4h), ("1h", df_1h)):
                    st = ms_mod.stat_state(dfx)
                    if st:
                        stats[tf] = st
                # 每周期最近位带距离 (ATR 归一, 取最近一侧)
                dists = {}
                for tf, info in tfs.items():
                    rel = info.get("rel") or {}
                    dmin = None
                    for side in ("support", "resistance"):
                        lv = rel.get(side)
                        if lv and (dmin is None or lv["dist_atr"] < dmin):
                            dmin = lv["dist_atr"]
                    if dmin is not None:
                        dists[tf] = round(float(dmin), 2)
                # 一致性 (A6d): 日线段方向 vs 4H 段方向
                cons = ""
                dd = dow_daily.get("seg_dir")
                d4 = dow4h.get("seg_dir")
                if dd in ("up", "down") and d4 in ("up", "down"):
                    cons = "顺风" if dd == d4 else "逆风"
                elif dd in ("up", "down") or d4 in ("up", "down"):
                    cons = "单边"
                diag["日线"] = (f"daily_len={len(daily_df) if daily_df is not None else 0} "
                               f"dow={dow_daily.get('seg_dir', '?') if dow_daily else 'EMPTY'}")
                warnings[sym] = {"warns": warns, "dow": dow4h,
                                 "dow_daily": dow_daily, "dow_h1": dow_h1,
                                 "stats": stats, "dists": dists, "cons": cons,
                                 "diag": diag}
        except Exception as e:
            logger.debug(f"scan warn error {sym}: {e}")
    return warnings

def _level_quality(lvl, overlap4h: bool) -> str:
    """关键位质量分 (研究: age 200-400最优 + touch≥3 + 4H重叠 → 单调 53.8→62.1%)"""
    sc = 0
    if 200 <= lvl.age_bars < 400:
        sc += 2
    elif lvl.age_bars >= 100:
        sc += 1
    if lvl.touch_count >= 5:
        sc += 1
    elif lvl.touch_count >= 3:
        sc += 0.5
    if overlap4h:
        sc += 1.5
    return "高" if sc >= 3 else "中" if sc >= 1.5 else "低"


def _enrich_alert(alert: Alert, tf_ind: dict, sym: str,
                  sym_alerts: list[Alert] | None = None,
                  dow_info: dict | None = None):
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

    # ── 1H 方向确认 (回测: 1H 方向在 4H bias 下无增量, 仅作展示不过滤) ──
    sig_dir = alert.direction
    if dir_1h in ("bullish", "bearish"):
        if (dir_1h == "bullish" and sig_dir == "long") or (dir_1h == "bearish" and sig_dir == "short"):
            check.append("✓方向")
        else:
            check.append("?方向(1H逆)")
    else:
        check.append("?方向")

    # ── 4H 大结构边界 (提前检测, 供趋势判断参考) ──
    ind_4h = tf_ind.get("4h", {})
    current_price = (ind_1h or tf_ind.get("5m", {})).get("close") if ind_1h else None
    atr_4h = ind_4h.get("atr") or 1
    has_4h_boundary = False
    df_4h = ind_4h.get("df")
    if df_4h is not None and len(df_4h) >= 20 and current_price:
        levels_4h = find_swing_levels(df_4h, lookback=50)
        for lvl in levels_4h:
            if lvl.touch_count >= 2 and abs(current_price - lvl.price) <= atr_4h * 1.0:
                alert.meta["4h_boundary"] = f"4H{lvl.side}{lvl.price:.5g}"
                has_4h_boundary = True
                check.append("⚠4H边界")
                break

    # ── 宏观趋势: 4H bias (修复未来函数后: 真实 edge +1.9pp, 弱但方向正确) ──
    bias = _symbol_bias(tf_ind)
    alert.details["macro_bias"] = bias
    counter = (bias == "long" and sig_dir == "short") or (bias == "short" and sig_dir == "long")
    if bias != "neutral" and not counter:
        check.append("✓趋势")
    elif bias == "neutral":
        check.append("?无趋势")
    else:
        # bias 相反 — 修复后 49.2% 略负, 监控保留但标记 (不做硬过滤)
        check.append("⚠逆趋势")

    # ── 场景引擎 (3年1094天平衡数据验证: 顺大逆小是唯一稳定edge) ──
    # episode=日线顺势+4H逆向(最优) | follow=全顺势 | counter=逆日线(低期望)
    scene = scene_of(alert.direction, ind_4h, bias)
    alert.details["scene"] = scene
    if scene in SCENE_WR:
        label, wr = SCENE_WR[scene]
        alert.meta["scene"] = f"{label} {wr}"
        boost = SCENE_BOOST.get(scene, 1.0)
        alert.confidence = min(alert.confidence * boost, 1.0)
        if scene.startswith("episode"):
            check.append(f"✓{label} {wr}")
        elif scene.startswith("follow"):
            check.append(f"✓{label} {wr}")
        else:
            check.append(f"⚠{label} {wr}")

    # ── BB 反转环境补充 (3年: 日线空/中性 + 上轨破轨 = 68% 稳定, 中性也成立) ──
    if alert.signal_type == "bb_reversal":
        if bias == "neutral" and scene == "neutral":
            alert.meta["scene"] = "回归 68%"
            check.append("✓回归 68%")
        elif bias == "short":
            alert.confidence = min(alert.confidence * 1.05, 1.0)

    # ── 真实确认的毒药组合 (修复后回测 1:1 胜率, 监控标记+轻度降权) ──
    # fakeout/long 45.1% | 1H空头+做多 46.1%
    if alert.signal_type == "fakeout" and sig_dir == "long":
        check.append("⚠fakeout多")
        alert.confidence = min(alert.confidence * 0.85, 1.0)
    if dir_1h == "bearish" and sig_dir == "long":
        check.append("⚠逆1H抄底")
        alert.confidence = min(alert.confidence * 0.90, 1.0)

    # ── S/R + SL/TP/RR ──
    ind_base = ind_1h or tf_ind.get("5m", {})
    current_price = ind_base.get("close")
    atr_1h = ind_base.get("atr") or 1
    atr_5m = (tf_ind.get("5m") or {}).get("atr") or 1

    sr_info = {}
    df_1h = (ind_1h or {}).get("df")
    if df_1h is not None and current_price:
        # lookback=600: 质量分年龄维度需要 200-400 根历史 (研究结论)
        levels = find_swing_levels(df_1h, lookback=600)
        support, resistance = get_nearest_levels(levels, current_price)

        if support:
            p = support.price
            dist_a = abs(current_price - p) / max(atr_1h, 1e-9)
            dist_lbl = "近" if dist_a <= 0.5 else "远" if dist_a >= 1.5 else ""
            q = _level_quality(support, has_4h_boundary)
            alert.meta["support"] = f"{fmt_price(p, atr_1h)}({support.strength},{support.touch_count}触{dist_lbl}质{q})"
            sr_info["support"] = support
        if resistance:
            p = resistance.price
            dist_a = abs(current_price - p) / max(atr_1h, 1e-9)
            dist_lbl = "近" if dist_a <= 0.5 else "远" if dist_a >= 1.5 else ""
            q = _level_quality(resistance, has_4h_boundary)
            alert.meta["resistance"] = f"{fmt_price(p, atr_1h)}({resistance.strength},{resistance.touch_count}触{dist_lbl}质{q})"
            sr_info["resistance"] = resistance

        # ── 类型A: 最近极值贴位 (研究: 贴而未破 63.7-67.1% vs 刚跌破 31-40%) ──
        rec = find_recent_extremes(df_1h, lookback=600)
        touch_type = None
        touch_dist = None
        if alert.direction == "long" and rec["low"] is not None:
            d = (current_price - rec["low"][1]) / max(atr_1h, 1e-9)
            if -0.5 <= d <= 0.5:
                touch_dist = d
                touch_type = "A_贴位未破" if d >= 0 else "A_位已破"
        elif alert.direction == "short" and rec["high"] is not None:
            d = (rec["high"][1] - current_price) / max(atr_1h, 1e-9)
            if -0.5 <= d <= 0.5:
                touch_dist = d
                touch_type = "A_贴位未破" if d >= 0 else "A_位已破"
        if touch_type is not None and touch_dist is not None:
            alert.meta["touch_type"] = touch_type
            if touch_type == "A_贴位未破":
                check.append(f"✓贴位{touch_dist:.1f}ATR")
                if scene.startswith("episode"):
                    alert.confidence = min(alert.confidence * 1.05, 1.0)
                    base_scene = alert.meta.get("scene", "")
                    alert.meta["scene"] = f"{base_scene}·贴位65-67%"
            else:
                check.append(f"⚠位已破{touch_dist:.1f}ATR")
                alert.confidence = min(alert.confidence * 0.90, 1.0)

        pos_in_range = None
        if support and resistance and resistance.price > support.price:
            pos_in_range = (current_price - support.price) / (resistance.price - support.price)
        if alert.direction == "long" and support:
            if pos_in_range is not None and pos_in_range >= 0.7:
                check.append("?远S/R")
            elif pos_in_range is not None and pos_in_range <= 0.3:
                check.append("✓近支撑")
            elif support.touch_count >= 2:
                check.append("✓有支撑")
            else:
                check.append("?无支撑")
        elif alert.direction == "short" and resistance:
            if pos_in_range is not None and pos_in_range <= 0.3:
                check.append("?远S/R")
            elif pos_in_range is not None and pos_in_range >= 0.7:
                check.append("✓近阻力")
            elif resistance.touch_count >= 2:
                check.append("✓有阻力")
            else:
                check.append("?无阻力")
        else:
            check.append("?边界")

        # ── 第一批: 市场结构标签 (A6 道氏段 + B 关键位, 2026-08-04, 描述性) ──
        if dow_info:
            seg_dir = dow_info.get("seg_dir", "")
            age = dow_info.get("seg_age")
            if seg_dir in ("up", "down") and age is not None:
                surv = dow_info.get("seg_surv", 1.0)
                seg_lbl = "上升" if seg_dir == "up" else "下降"
                alert.meta["4h_seg"] = f"{seg_lbl}段{age}根(存活{surv:.0%})"
                check.append(f"4H段{age}根")
                if age > 25:
                    check.append("⚠4H段老")
                pos = dow_info.get("seg_pos")
                if pos is not None:
                    pos_lbl = "早" if pos < 0.33 else ("晚" if pos > 0.67 else "中")
                    alert.meta["seg_pos"] = pos_lbl
                    if pos_lbl == "晚":
                        check.append("?段晚期")
            cons = dow_info.get("daily_cons", "")
            if cons:
                alert.meta["daily_cons"] = cons
                check.append(f"{'✓' if cons == '顺风' else '?'}{cons}")
        # B1 区间内 (简化: 成对位带间距 ≤2.5×ATR) / B2 触碰释放
        if market_structure.check_range_bounds(levels, current_price, atr_1h):
            alert.meta["in_range"] = "区间"
            check.append("✓区间")
        touch = market_structure.check_recent_touch(levels, df_1h)
        if touch is not None:
            alert.meta["recent_touch"] = f"{touch.side}触"
            check.append("⚡触位")

    # ── SL/TP/RR (3年1094天验证参数: 按场景给可达目标, SL下限1×1H ATR防RR虚高) ──
    entry_price = current_price or ind_base.get("close") or 0
    sl = tp = 0
    atr_sl_buffer = 0.1
    scene = alert.details.get("scene", "neutral")
    if alert.signal_type == "bb_reversal":
        # BB反转 (3年68%稳定): 高胜率小目标 1:1 — TP=0.5×1H ATR, SL=0.5×1H ATR
        atr_tp_mult = 0.5
        sl_min = atr_1h * 0.5
    elif scene.startswith("episode"):
        # 顺日逆时: TP=2×1H ATR 吃完整波段 (3年57-61%), SL≥1×1H ATR 保RR真实
        atr_tp_mult = 2.0
        sl_min = atr_1h * 1.0
    elif scene.startswith("follow"):
        atr_tp_mult = 1.5
        sl_min = atr_1h * 1.0
    else:
        # 逆势/中性: 保守小目标 (低期望场景)
        atr_tp_mult = 1.0
        sl_min = atr_1h * 1.0

    if alert.direction == "long":
        sl = sr_info["support"].price - atr_1h * atr_sl_buffer if "support" in sr_info else entry_price - atr_5m * 1.5
        tp = entry_price + atr_1h * atr_tp_mult
        # SL 距离下限: 按场景 (bb_reversal 0.5 / 其余 1.0) — SL太近→RR虚高→实盘被噪声打掉
        if entry_price - sl < sl_min:
            sl = entry_price - sl_min
        if tp <= sl or tp <= entry_price:
            tp = entry_price + atr_1h * atr_tp_mult
    elif alert.direction == "short":
        sl = sr_info["resistance"].price + atr_1h * atr_sl_buffer if "resistance" in sr_info else entry_price + atr_5m * 1.5
        tp = entry_price - atr_1h * atr_tp_mult
        if sl - entry_price < sl_min:
            sl = entry_price + sl_min
        if tp >= sl or tp >= entry_price:
            tp = entry_price - atr_1h * atr_tp_mult
    else:
        sl = entry_price - atr_5m * 1.5
        tp = entry_price + atr_1h * 1.5

    alert.meta["sl"] = fmt_price(sl, atr_1h)
    alert.meta["tp"] = fmt_price(tp, atr_1h)
    sl_dist = abs(entry_price - sl)
    tp_dist = abs(tp - entry_price)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0
    # RR 上限标注: 超3 提示低胜率 (RR虚高不可达)
    if rr > 3.0:
        alert.meta["rr"] = f"3.0+:1(低胜率)"
    else:
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
                opt_sl = opt_entry_price - atr_1h * atr_sl_buffer
                opt_tp = opt_entry_price + atr_1h * atr_tp_mult
                if opt_tp > opt_entry_price and opt_sl < opt_entry_price:
                    opt_rr_val = (opt_tp - opt_entry_price) / (opt_entry_price - opt_sl)
        elif alert.direction == "short" and "resistance" in sr_info:
            resistance_lvl = sr_info["resistance"]
            touches = resistance_lvl.touch_count
            opt_entry_price = resistance_lvl.price
            if opt_entry_price > entry_price:
                opt_sl = opt_entry_price + atr_1h * atr_sl_buffer
                opt_tp = opt_entry_price - atr_1h * atr_tp_mult
                if opt_tp < opt_entry_price and opt_sl > opt_entry_price:
                    opt_rr_val = (opt_entry_price - opt_tp) / (opt_sl - opt_entry_price)

        opt_dist_atr = abs(opt_entry_price - entry_price) / max(atr_1h, 1e-9) if opt_entry_price else 99
        # 只建议近防线挂单 (≤0.5×1H ATR 等得到) — 远防线不显示挂单 (现价入场即可)
        if (opt_entry_price and opt_dist_atr <= 0.5
                and opt_rr_val > rr and opt_rr_val >= 1.2):
            alert.meta["opt_entry"] = fmt_price(opt_entry_price, atr_1h)
            alert.meta["opt_rr"] = f"{opt_rr_val:.1f}:1"
            alert.meta["opt_dist"] = f"{opt_dist_atr:.1f}ATR"
            if touches >= 3:
                check.append("🔵左侧挂单(近)")
            elif touches >= 2:
                check.append("🟡右侧等K@防线(近)")
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
                    alert.meta["vp_support"] = f"{fmt_price(p, atr_1h)}(量节点)"
                if vp_nodes["resistance"]:
                    p = vp_nodes["resistance"]["price"]
                    alert.meta["vp_resistance"] = f"{fmt_price(p, atr_1h)}(量节点)"

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

    # ── 仓位参考: 1%本金风险 / 实际止损距离 (3年: 与SL匹配, 小目标仓位自动放大) ──
    try:
        risk_pct = 1
        leverage = 10
        risk_dist = sl_dist if sl_dist > 0 else atr_1h * 1.0
        if entry_price > 0 and risk_dist > 0:
            pos_pct = entry_price * risk_pct / (risk_dist * leverage)
            alert.meta["margin"] = f"参考{pos_pct:.0f}%({leverage}x {risk_dist/atr_1h:.1f}ATR)" if pos_pct <= 100 else f"⚠波大参考{pos_pct:.0f}%"
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

    # ── 波动状态 × 插曲 (3年20标的验证: 高波动插曲 62.7% / 常态 56.3% / 低波动 53.4%) ──
    # atr_ratio = 当前1H ATR / 过去90根均值
    try:
        df_1h_v = (ind_1h or {}).get("df")
        scene_now = alert.details.get("scene", "neutral")
        if scene_now in ("episode_long", "episode_short") and df_1h_v is not None and len(df_1h_v) >= 95:
            from market_phase import _atr_series as _atr_s
            atr_v = _atr_s(df_1h_v.tail(95).reset_index(drop=True))
            if len(atr_v) > 90 and atr_v[-1] > 0:
                atr_ma_v = float(np.mean(atr_v[-91:-1]))
                atr_ratio = atr_v[-1] / atr_ma_v if atr_ma_v > 0 else 1.0
                if atr_ratio >= 1.3:
                    check.append(f"✓高波动插曲 {atr_ratio:.1f}x")
                    alert.confidence = min(alert.confidence * 1.05, 1.0)
                elif atr_ratio < 0.7:
                    check.append(f"⚠低波动插曲 {atr_ratio:.1f}x")
                    alert.confidence = min(alert.confidence * 0.95, 1.0)
    except Exception:
        pass

    # ── 深回调提示 (3年验证: 4H深度>2ATR 的插曲做多更优 55.2% vs 浅回调 54.2%) ──
    try:
        close_4h_v = ind_4h.get("close")
        ma20_4h_v = ind_4h.get("ma20")
        atr_4h_v = ind_4h.get("atr") or 1
        scene_now = alert.details.get("scene", "neutral")
        if scene_now in ("episode_long", "episode_short") and close_4h_v and ma20_4h_v and atr_4h_v > 0:
            dev_4h = abs(close_4h_v - ma20_4h_v) / atr_4h_v
            if dev_4h >= 1.5:
                check.append(f"✓深回调{dev_4h:.1f}ATR")
                alert.confidence = min(alert.confidence * 1.03, 1.0)
    except Exception:
        pass

    # ── 4大币 5M反转K执行确认 (3年5M验证: 仅BTC/ETH/SOL/BNB +2.4pp, 其余币无效) ──
    if sym in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"):
        body_dir_5m = ind_5m.get("body_dir")
        body_pct_5m = ind_5m.get("body_pct", 0)
        pinbar_5m = ind_5m.get("pinbar")
        want_bear = alert.direction == "short"
        rev_ok = False
        if want_bear:
            rev_ok = (pinbar_5m == "bearish") or (body_dir_5m == "bearish" and body_pct_5m >= 0.6)
        else:
            rev_ok = (pinbar_5m == "bullish") or (body_dir_5m == "bullish" and body_pct_5m >= 0.6)
        if rev_ok:
            check.append("✅5M反转K确认")
            alert.meta["exec"] = "5M已确认"
            alert.confidence = min(alert.confidence * 1.04, 1.0)
        else:
            check.append("⏳等5M反转K")
            alert.meta["exec"] = "等5M反转K"

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

        # ── RS 方向一致性 (416天回测: 全顺势组合下 RS与方向相反是危险信号) ──
        # bull×bull×bull 顺RS 52.6% vs 逆RS 46.5% | bear×bull×bull 顺RS 67.6% vs 逆RS 62.5%
        rs_4h_score = (rs_dict.get("4h") or {}).get("score", 0)
        rs_dir_ok = (alert.direction == "long" and rs_4h_score > 0) or \
                    (alert.direction == "short" and rs_4h_score < 0)
        bias_ok = alert.details.get("macro_bias") == alert.direction
        h1_ok = (dir_1h == "bullish" and alert.direction == "long") or \
                (dir_1h == "bearish" and alert.direction == "short")
        if bias_ok and h1_ok and not rs_dir_ok:
            check.append("⚠RS逆势")
            alert.confidence = min(alert.confidence * 0.90, 1.0)
        elif bias_ok and h1_ok and rs_dir_ok:
            check.append("✓RS同向")

        # ── RS 位置依赖 (3年验证: RS价值取决于所处场景) ──
        # 顺日逆时(回调)中 RS强 = 抗跌性好 → 做多加分 (57.1%)
        # 全顺势中   RS强 = 过冲高潮   → 做多减分 (52.8%)
        # 日线空 + RS<-60 (极端弱势) → 做空加分 (58.3%)
        scene = alert.details.get("scene", "neutral")
        rs_4h_val = (rs_dict.get("4h") or {}).get("score", 0)
        if scene == "episode_long" and rs_4h_val > 60:
            check.append("✓抗跌")
            alert.confidence = min(alert.confidence * 1.05, 1.0)
        elif scene == "follow_long" and rs_4h_val > 60:
            check.append("⚠过冲")
            alert.confidence = min(alert.confidence * 0.93, 1.0)
        elif scene == "follow_short" and rs_4h_val < -60:
            check.append("✓弱势延续")
            alert.confidence = min(alert.confidence * 1.04, 1.0)
        elif scene == "episode_short" and rs_4h_val < -60:
            check.append("✓极弱")
            alert.confidence = min(alert.confidence * 1.05, 1.0)

        # 每周期独立增强: 方向与信号匹配的 TF 越多, 置信度越高
        agree_count = 0
        for s in marks:
            if (alert.direction == "long" and s > 0) or (alert.direction == "short" and s < 0):
                agree_count += 1
        if agree_count >= 3:
            alert.confidence = min(alert.confidence * 1.05, 1.0)
            check.append("✓RS三重")
        elif agree_count >= 2:
            alert.confidence = min(alert.confidence * 1.03, 1.0)
            check.append("✓RS双周期")

    # ── 量能耗尽 ──
    body_pct_5m = ind_5m.get("body_pct", 1)
    vr_5m_val = ind_5m.get("volume_ratio") or 1
    if body_pct_5m < 0.3 and vr_5m_val >= 2.5:
        check.append("⚠量能耗尽")

    # ── ATR 移动止损 ──
    if entry_price > 0 and atr_1h > 0:
        if alert.direction == "long":
            move_price = entry_price + atr_1h * 2
            if tp > entry_price and move_price < tp:
                alert.meta["trailing"] = f"移损@{fmt_price(entry_price+atr_1h*0.5, atr_1h)}(+2ATR移保本)"
        elif alert.direction == "short":
            move_price = entry_price - atr_1h * 2
            if tp < entry_price and move_price > tp:
                alert.meta["trailing"] = f"移损@{fmt_price(entry_price-atr_1h*0.5, atr_1h)}(+2ATR移保本)"

    # ── 分批止盈 ──
    if entry_price > 0 and atr_1h > 0 and tp != 0:
        if alert.direction == "long" and tp > entry_price:
            tp1 = entry_price + atr_5m * 0.8
            if tp1 < tp:
                alert.meta["tp1"] = f"{fmt_price(tp1, atr_1h)}(40%)"
                alert.meta["tp2"] = f"{fmt_price(tp, atr_1h)}(30%)"
            else:
                alert.meta["tp_full"] = f"{fmt_price(tp, atr_1h)}(70%)"
        elif alert.direction == "short" and tp < entry_price:
            tp1 = entry_price - atr_5m * 0.8
            if tp1 > tp:
                alert.meta["tp1"] = f"{fmt_price(tp1, atr_1h)}(40%)"
                alert.meta["tp2"] = f"{fmt_price(tp, atr_1h)}(30%)"
            else:
                alert.meta["tp_full"] = f"{fmt_price(tp, atr_1h)}(70%)"

    # ── 多空力量识别 (3年验证: 插曲环境+力量同向极端 = +2.5pp 59.6%) ──
    # 用户方法论: 观察近期多空力量对比及优势方变化, 为机会点提前准备
    try:
        from power_balance import analyze_power as _pw
        df_1h_pw = (ind_1h or {}).get("df")
        if df_1h_pw is not None and len(df_1h_pw) >= 60:
            pw = _pw(df_1h_pw.tail(60).reset_index(drop=True))
            pw_score = pw.get("score", 0)
            alert.meta["power"] = pw.get("reason", "")
            scene_now = alert.details.get("scene", "neutral")
            # 插曲 + 力量同向极端 → 回调结束确认 (59.6% 3年)
            if scene_now == "episode_long" and pw_score > 40:
                check.append("✓力量转多")
                alert.confidence = min(alert.confidence * 1.05, 1.0)
            elif scene_now == "episode_short" and pw_score < -40:
                check.append("✓力量转空")
                alert.confidence = min(alert.confidence * 1.05, 1.0)
            # 力量反向极端: 优势方拐点提示 (展示辅助, 不加降权)
            elif scene_now == "episode_long" and pw_score < -60:
                check.append("⚠力量仍空")
            elif scene_now == "episode_short" and pw_score > 60:
                check.append("⚠力量仍多")
            # 趋势末期提示 (反转风险预警)
            try:
                from market_phase import analyze_market_state as _phase, HISTORY
                ms_1h_p = _phase(df_1h_pw.tail(HISTORY).reset_index(drop=True))
                if ms_1h_p.get("stage") == "late" and ms_1h_p.get("state", "").startswith("trend"):
                    check.append("⚠1H趋势末期")
                    alert.meta["phase"] = ms_1h_p.get("reason", "")
            except Exception:
                pass
    except Exception:
        pass

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


def _active_symbols(all_symbols: list[str]) -> list[str]:
    if _is_us_market_hours():
        return all_symbols
    return [s for s in all_symbols if not _is_us_stock(s)]


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
    active_symbols = _active_symbols(symbols)
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

    await _parallel_prefill(okx, symbols, timeframes, cache, limit=500)
    logger.info("Cache prefill done")

    def on_kline(sym: str, tf: str, candle: Candle):
        cache.update(sym, tf, candle)

    rebuild_flag = {"need": False}

    def _on_ws_reconnect():
        rebuild_flag["need"] = True
        logger.warning("WS reconnected: cache rebuild queued")

    okx.set_on_reconnect(_on_ws_reconnect)

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

    scan_count = 0
    first_scan = True

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
        await asyncio.gather(*[_fetch(s, t) for s in active_symbols for t in timeframes])
        if refreshed > 0:
            logger.debug(f"REST cache refresh: {refreshed} bars")

    while True:
        await asyncio.sleep(_wait_next_5m())
        scan_count += 1
        scan_start = _bj_now()
        try:
            active_symbols = _active_symbols(symbols)
            if len(active_symbols) < len(symbols):
                logger.debug(f"Off-market hours: {len(symbols) - len(active_symbols)} US stocks skipped")
            if rebuild_flag["need"]:
                # WS 重连后: REST 全量重建缓存 (清空再填充, 消除断线缺口/污染)
                sem = asyncio.Semaphore(8)
                async def _rebuild(s, t):
                    async with sem:
                        try:
                            bars = await asyncio.get_event_loop().run_in_executor(
                                None, okx.fetch_ohlcv, s, t, 500)
                            cache.reset(s, t)
                            for bar in bars:
                                cache.update(s, t, Candle(
                                    timestamp=bar["timestamp"], open=bar["open"],
                                    high=bar["high"], low=bar["low"],
                                    close=bar["close"], volume=bar["volume"]))
                        except Exception:
                            pass
                await asyncio.gather(*[_rebuild(s, t)
                                      for s in active_symbols for t in timeframes])
                rebuild_flag["need"] = False
                logger.info(f"Cache rebuilt after WS reconnect ({len(active_symbols)} symbols)")
            await _refresh_cache_if_stale()

            # ── 三柱扫描 → 预警 (无方向预测) ──
            warnings = do_scan(active_symbols, cache, config)

            if warnings:
                lines_out = [f"━━━ ⚡预警 {scan_start.strftime('%H:%M')} 北京时间 {_current_session()} ━━━"]

                def _seg_str(sd):
                    # 段位置早/中/晚不可标注: 进行中段被 close_seg(n-1) 截断,
                    # seg_pos 几乎恒为"晚" (误导) — 用段龄+存活率表达
                    d = sd.get("seg_dir")
                    age = sd.get("seg_age")
                    surv = sd.get("seg_surv")
                    if d in ("up", "down") and age is not None:
                        arrow = "↑" if d == "up" else "↓"
                        return f"{arrow}{age}根({surv:.0%})"
                    return "—"

                def _stat_line(item):
                    """推荐标的多周期状态行: 日线/4H/1H 段 + A2 统计 + 位距 + 一致性"""
                    stats = item.get("stats") or {}
                    parts = []
                    for tf, tfname, sd in (("日线", "日线", item.get("dow_daily") or {}),
                                           ("4h", "4H", item.get("dow") or {}),
                                           ("1h", "1H", item.get("dow_h1") or {})):
                        seg_s = _seg_str(sd)
                        st = stats.get(tf)
                        if st:
                            seg_s += f"·{st['label']}(dev{st['dev']:+.1f},adx{st['adx']:.0f})"
                        parts.append(f"{tfname}:{seg_s}")
                    dists = item.get("dists") or {}
                    dparts = []
                    for tf, tfname in (("日线", "D"), ("4h", "4H"), ("1h", "1H")):
                        v = dists.get(tf)
                        if v is not None:
                            dparts.append(f"{tfname}{v}")
                    line = " | ".join(parts)
                    if dparts:
                        line += f" | 距" + "/".join(dparts) + "ATR"
                    cons = item.get("cons")
                    if cons:
                        line += f" | {cons}"
                    return line

                ranked = []
                for sym, item in warnings.items():
                    sym_short = sym.replace("-USDT-SWAP", "/USDT").split(":")[0]
                    item["_short"] = sym_short
                    dow = item.get("dow") or {}
                    # 仓位参考 (用户: 10X 杠杆 / 1H ATR 止损 / 本金 1% 风险 → 本金开仓比例)
                    # pos_pct = 价格 × risk_pct / (1×ATR_1h × leverage) — 旧系统公式
                    pos_ref = ""
                    try:
                        from market_phase import _atr_series
                        df1 = cache.get_df(sym, "1h")
                        if df1 is not None and len(df1) >= 60:
                            atr1 = float(_atr_series(df1)[-1]) or 0.0
                            p1 = float(df1["close"].values[-1]) or 0.0
                            if atr1 > 0 and p1 > 0:
                                rp = config.get("account", {}).get("risk_pct", 1)
                                lv = config.get("account", {}).get("leverage", 10)
                                pos_pct = p1 * rp / (atr1 * lv)
                                pos_ref = (f" 仓位参考{pos_pct:.0f}%({lv}x 1.0ATR)"
                                           if pos_pct <= 100 else f"  ⚠波大参考{pos_pct:.0f}%")
                    except Exception:
                        pass
                    for w in item["warns"]:
                        ranked.append((w["level"], sym_short, w, dow, pos_ref))
                ranked.sort(key=lambda x: (x[0], x[1]))
                count = 0
                seen = set()
                for _, sym_short, w, dow, pos_ref in ranked:
                    if count >= 12:
                        break
                    seg = ""
                    d = dow.get("seg_dir")
                    age = dow.get("seg_age")
                    if d in ("up", "down") and age is not None:
                        seg = f"·段{'↑' if d == 'up' else '↓'}{age}根"
                    icon = {"L1": "🔵", "L2": "⚡", "L3": "💥"}.get(w["level"], "")
                    lines_out.append(f"{sym_short} {icon}{w['tf']} {w['desc']}{seg}{pos_ref}")
                    count += 1
                    # 状态行: 紧跟该标的的第一条预警 (带标的名, 便于对应)
                    if sym_short not in seen:
                        seen.add(sym_short)
                        item = next((it for it in warnings.values()
                                     if it.get("_short") == sym_short), None)
                        if item is not None:
                            lines_out.append(f"  {sym_short} 状态: {_stat_line(item)}")
                push_text = "\n".join(lines_out)

                # ── 富文本推送: 一条消息 = 汇总文本 + 内嵌预警图 (防刷屏) ──
                # 配置: config.yaml -> feishu_image: {enabled, max_per_scan, min_level}
                img_cfg = config.get("feishu_image", {})
                blocks: list[dict] = []
                chart_syms: list[str] = []
                try:
                    if img_cfg.get("enabled", False):
                        min_level = img_cfg.get("min_level", "L2")
                        max_charts = int(img_cfg.get("max_per_scan", 3))
                        # 按级别排序取最严重的标的 (L3 > L2 > L1)
                        for _, sym_short, w, _dow, _pr in sorted(
                                ranked, key=lambda x: x[0]):
                            if w["level"] < min_level or len(chart_syms) >= max_charts:
                                continue
                            full = next((s for s, it in warnings.items()
                                         if it.get("_short") == sym_short), None)
                            if full and full not in chart_syms:
                                chart_syms.append(full)
                        for full in chart_syms:
                            r = _send_warning_chart(cache, full, warnings.get(full, {}))
                            if r:
                                blocks.append({"image": r[1]})
                except Exception as e:
                    logger.warning(f"Chart push failed: {e}")

                if blocks:
                    # 图文混合: 先文本汇总, 图直接跟在后面
                    rich_blocks = [{"text": push_text}] + blocks
                    feishu.send_rich(f"⚡ 预警 {scan_start.strftime('%H:%M')} 北京时间 {_current_session()}",
                                     rich_blocks)
                else:
                    feishu.send(push_text)
                diag_lines = []
                for sym, item in warnings.items():
                    d = item.get("diag") or {}
                    if d:
                        sym_short = sym.replace("-USDT-SWAP", "/USDT").split(":")[0]
                        diag_lines.append(f"[diag] {sym_short}: "
                                          + " | ".join(f"{tf}:{v}" for tf, v in d.items()))
                logger.info(f"Scan #{scan_count}: {len(warnings)} symbols with warnings, {count} pushed\n"
                            + push_text + ("\n" + "\n".join(diag_lines) if diag_lines else ""))
            else:
                logger.info(f"Scan #{scan_count}: no warnings")

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
