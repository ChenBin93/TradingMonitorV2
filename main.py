# TradingMonitor V2 — 入口 + 主循环 + 飞书报告格式化

import asyncio
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import yaml
from loguru import logger

from okx import OKXClient, KlineCache, Candle
from indicators import compute as compute_indicators
from signals import SIGNALS, SignalState, get_direction, get_regime
from notify import Feishu
from utils import setup_logging, start_health_server
from support_resistance import find_swing_levels, get_nearest_levels
from volume_profile import compute_volume_profile, get_nearest_nodes

# 美股代币化合約列表（非交易时段过滤）
_US_STOCKS = {
    "AAPL/USDT:USDT", "MSFT/USDT:USDT", "GOOGL/USDT:USDT", "AMZN/USDT:USDT",
    "META/USDT:USDT", "NVDA/USDT:USDT", "TSLA/USDT:USDT", "AMD/USDT:USDT",
    "INTC/USDT:USDT", "MU/USDT:USDT", "ORCL/USDT:USDT", "PLTR/USDT:USDT",
    "MRVL/USDT:USDT", "TSM/USDT:USDT", "HOOD/USDT:USDT", "RKLB/USDT:USDT",
    "WDC/USDT:USDT", "QQQ/USDT:USDT", "SPY/USDT:USDT",
}

# 贵金属
_METALS = {"XAU/USDT:USDT", "XAG/USDT:USDT", "XPT/USDT:USDT", "XPD/USDT:USDT", "XCU/USDT:USDT"}

# ETF（非美股）
_ETFS = {"EWY/USDT:USDT"}

_CAT_LABELS = {"stock": "美股", "metal": "贵金属", "etf": "ETF", "crypto": "Crypto"}


def _symbol_category(symbol: str) -> str:
    """返回品种类别: stock | metal | etf | crypto"""
    if symbol in _US_STOCKS:
        return "stock"
    if symbol in _METALS:
        return "metal"
    if symbol in _ETFS:
        return "etf"
    return "crypto"


# =============================================================================
# 配置加载
# =============================================================================

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # 合并 secrets
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
    meta: dict = field(default_factory=dict)    # 增强信息(S/R/SL/TP/RR)
    checklist: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def tag(self) -> str:
        tags = {"bb_squeeze": "BB", "rsi_extreme": "RSI", "ma_converge": "MA",
                "macd_cross": "MACD", "volume_spike": "VOL", "ttm_squeeze": "TTM",
                "rsi_divergence": "RSI背", "macd_divergence": "MACD背",
                "ma_alignment": "MA排", "adx_surge": "ADX",
                "atr_expansion": "ATR", "compression_combo": "压",
                "price_extreme": "极"}
        return tags.get(self.signal_type, self.signal_type[:4])

    @property
    def name(self) -> str:
        names = {"bb_squeeze": "BB压缩", "rsi_extreme": "RSI极值", "ma_converge": "MA汇聚",
                 "macd_cross": "MACD交叉", "volume_spike": "放量", "ttm_squeeze": "TTM压缩",
                 "rsi_divergence": "RSI背离", "macd_divergence": "MACD背离",
                 "ma_alignment": "均线排列", "adx_surge": "ADX突破",
                 "atr_expansion": "波动爆发", "compression_combo": "多重压缩",
                 "price_extreme": "价格极值"}
        return names.get(self.signal_type, self.signal_type)

    @property
    def tf_role(self) -> str:
        """三层角色: 4h→方向 1h→结构 15m→执行"""
        return {"4h": "方向", "1h": "结构", "15m": "执行"}.get(self.timeframe, self.timeframe)


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

        if a.signal_type == "bb_squeeze":
            rank = d.get("bbw_rank", 50)
            if rank <= 10: conf += 0.15
            elif rank <= 20: conf += 0.10
            elif rank <= 30: conf += 0.05

        elif a.signal_type == "rsi_extreme":
            rsi = ind.get("rsi", 50)
            if rsi <= 20 or rsi >= 80: conf += 0.08

        elif a.signal_type == "volume_spike":
            vr = d.get("volume_ratio", 1)
            if vr >= 5: conf += 0.10

        elif a.signal_type == "ma_converge":
            mc = d.get("ma_converge", 1)
            if mc <= 0.2: conf += 0.08
            # 回测: 多头胜率55% vs 空头18% — 空头降权
            if a.direction == "short": conf -= 0.15

        elif a.signal_type == "ttm_squeeze":
            if d.get("is_fired"): conf += 0.15
            elif d.get("squeeze_bars", 0) >= 8: conf += 0.10

        elif a.signal_type == "compression_combo":
            hits = d.get("combo_signals", [])
            if len(hits) >= 3: conf += 0.10
            elif len(hits) >= 2: conf += 0.05

        elif a.signal_type == "ma_alignment":
            spread = abs(ind.get("close", 0) - (ind.get("ma60") or 0)) / (ind.get("ma60") or 1) * 100
            if spread > 8: conf += 0.08
            # 回测: 空头胜率45% vs 多头18% — 方向不对称
            if a.direction == "long": conf -= 0.10
            elif a.direction == "short": conf += 0.05

        elif a.signal_type == "ma_converge":
            mc = d.get("ma_converge", 1)
            if mc <= 0.2: conf += 0.08
            # 回测: 多头胜率55% vs 空头18% — 只做多
            if a.direction == "short": conf -= 0.15

        elif a.signal_type == "adx_surge":
            adx = ind.get("adx", 0)
            if adx >= 35: conf += 0.08

        elif a.signal_type == "atr_expansion":
            if d.get("direction") != "neutral":
                conf += 0.05

        elif a.signal_type == "rsi_divergence":
            if d.get("price_distance_pct", 0) >= 5: conf += 0.12

        elif a.signal_type == "macd_divergence":
            if d.get("price_distance_pct", 0) >= 5: conf += 0.10

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
        # 同 symbol/signal_type 只保留最高置信度（不同信号类型各自保留）
        deduped: dict[str, Alert] = {}
        for a in result:
            k = f"{a.symbol}_{a.signal_type}"
            if k not in deduped or a.confidence > deduped[k].confidence:
                deduped[k] = a
        return sorted(deduped.values(), key=lambda x: x.confidence, reverse=True)


# =============================================================================
# TOP5 排序
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
    """多维度符号排序：综合信号强度、动量、压缩、成交量"""
    scores: dict[str, dict] = {}

    for a in alerts:
        s = scores.setdefault(a.symbol, {
            "direction": a.direction, "regime": a.regime, "timeframe": a.timeframe,
            "tags": [], "conf": 0, "momentum": 0.0, "compression": 0.0, "volume": 0.0, "reasons": [],
        })
        s["tags"].append(a.tag)
        s["conf"] = max(s["conf"], a.confidence)

        d = a.details

        # 压缩维度：BB rank / TTM squeeze / compression combo
        if a.signal_type in ("bb_squeeze", "ttm_squeeze", "compression_combo", "ma_converge"):
            rank_val = d.get("bbw_rank", 50)
            s["compression"] = max(s["compression"], (100 - rank_val) / 100)
        if a.signal_type == "atr_expansion":
            s["compression"] = max(s["compression"], 0.7)

        # 成交量维度
        if a.signal_type in ("volume_spike",):
            vr = d.get("volume_ratio", 1)
            s["volume"] = max(s["volume"], min(vr / 3, 1))

        # 动量维度：用 4h + 1h ADX
        tf_data = all_ind.get(a.symbol, {})
        for tf in ("4h", "1h"):
            ind = tf_data.get(tf, {})
            adx_v = ind.get("adx", 0) or 0
            s["momentum"] = max(s["momentum"], min(adx_v / 40, 1))

    result = []
    for sym, s in scores.items():
        total = s["conf"] * 0.35 + s["momentum"] * 0.25 + s["compression"] * 0.20 + s["volume"] * 0.20
        reasons = []
        if s["momentum"] > 0.6: reasons.append("动量强")
        if s["compression"] > 0.6: reasons.append("蓄力中")
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
    """判断是否为美股代币化合約"""
    return _US_STOCKS is not None and symbol in _US_STOCKS


def _is_us_market_hours() -> bool:
    """判断当前是否在美股交易时段 (UTC 13:30-21:00)"""
    try:
        now = datetime.now()
        start = datetime(now.year, now.month, now.day, 13, 30)
        end = datetime(now.year, now.month, now.day, 21, 0)
        # 周一至周五
        if now.weekday() >= 5:
            return False
        return start <= now <= end
    except Exception:
        return True  # 出错时不过滤


# =============================================================================
# 飞书报告格式化
# =============================================================================

def fmt_short_alert(a: Alert) -> str:
    """单条预警的紧凑格式"""
    sym = a.symbol.replace("-USDT-SWAP", "/USDT").split(":")[0]
    dir_map = {"long": "多", "short": "空"}
    d = dir_map.get(a.direction, "")
    m = a.meta
    severity_icon = "🟢" if a.details.get("stage2_upgrade") else ("🔴" if a.severity == "critical" else "🟠")
    rr = m.get("rr", "-")
    s = m.get("support", "-")
    r = m.get("resistance", "-")
    checks = " ".join([c for c in a.checklist if c.startswith("✓") or c.startswith("⚠")][:4])
    margin = m.get("margin", "")
    role = "/".join(getattr(a, "_merged_tfs", [a.timeframe]))
    line = f"{severity_icon} {sym}[{role}] {d} {a.name} RR:{rr} {margin}"
    line2 = f"   S:{s} R:{r} | {checks}"
    opt_entry = m.get("opt_entry", "")
    opt_rr = m.get("opt_rr", "")
    opt_line = f"\n   最优: {opt_entry}→RR:{opt_rr}" if opt_entry and opt_rr else ""
    vp_s = m.get("vp_support", "")
    vp_r = m.get("vp_resistance", "")
    vp_line = f"\n   量防线: S:{vp_s} R:{vp_r}" if vp_s or vp_r else ""
    trail = m.get("trailing", "")
    trail_line = f"\n   {trail}" if trail else ""
    tp_full = m.get("tp_full", "")
    tp_line = f"\n   TP:{tp_full}" if tp_full else ""
    tp1 = m.get("tp1", "")
    tp2 = m.get("tp2", "")
    if tp1 and tp2:
        tp_line = f"\n   出场: TP1@{tp1} TP2@{tp2} Moon(30%)"
    return f"{line}\n{line2}{opt_line}{vp_line}{tp_line}{trail_line}"


def format_consolidated_report(
    filtered: list[Alert],
    ranks: list[SymbolRank],
    total_alerts: int,
    symbol_count: int,
    scan_time: datetime,
    category_label: str = "",
) -> str:
    """合并预警 + 排名为一条消息"""
    time_str = scan_time.strftime("%H:%M")
    cat_prefix = f"【{category_label}】" if category_label else ""

    # 低质量过滤
    quality = []
    for a in filtered:
        rr_str = a.meta.get("rr", "0:1").split(":")[0]
        try:
            rr_val = float(rr_str)
        except ValueError:
            rr_val = 0

        if rr_val < 1.5:
            continue

        mtf_boost = a.details.get("mtf_boost", False)
        is_mid = any(c.startswith("?") for c in a.checklist if "边界" in c or "支撑" in c or "阻力" in c)
        if is_mid and not mtf_boost:
            continue

        # 方向冲突：高 TF 方向不一致 → 丢弃
        has_dir_fail = any(c == "✗方向" for c in a.checklist)
        if has_dir_fail:
            continue

        # 位置-方向冲突：做多在阻力附近 / 做空在支撑附近 → 丢弃
        has_pos_fail = any(c == "✗位置" for c in a.checklist)
        if has_pos_fail:
            continue

        # 数据陈旧：WS 断连超 2 小时 → 丢弃
        has_stale = any(c == "⚠数据陈旧" for c in a.checklist)
        if has_stale:
            continue

        # 美股非交易时段过滤：代币化美股在休市期流动性极低，假信号多
        if _is_us_stock(a.symbol) and not _is_us_market_hours():
            continue

        quality.append(a)

    # 合并同一币种+同信号+同方向 → 一行展示所有TF
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

    lines = [f"━━━ {cat_prefix}V2 扫描 {time_str} ━━━",
             f"{symbol_count}币 | {total_alerts}信号 | 推送{len(merged_list)}条 | 多{longs}/空{shorts}"]

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
            lines.append(f"{i}. {sym} {d} {r.score:.0%} | {tags} | {reason}")

    return "\n".join(lines)

    return "\n".join(lines)


# =============================================================================
# 主扫描逻辑
# =============================================================================

def get_symbols(okx: OKXClient, config: dict) -> list[str]:
    """获取监控标列表：白名单优先，否则 top_n"""
    watchlist = config.get("watchlist", [])
    if watchlist:
        return watchlist
    return okx.get_top_symbols(config.get("top_n", 20))


def do_scan(
    symbols: list[str],
    cache: KlineCache,
    config: dict,
) -> tuple[list[Alert], dict[str, dict[str, dict]]]:
    """返回: (alerts, {sym: {tf: ind}})"""
    timeframes = config["timeframes"]
    alerts: list[Alert] = []
    all_ind: dict[str, dict[str, dict]] = {}

    for sym in symbols:
        tf_ind: dict[str, dict] = {}

        # ── 第一遍：收集所有 TF 的指标数据 ──
        for tf in timeframes:
            df = cache.get_df(sym, tf)
            if len(df) < 30:
                continue
            ind_params = config["indicators"].get(tf, config["indicators"]["15m"])
            ind = compute_indicators(df, ind_params)
            if ind:
                tf_ind[tf] = ind

        if not tf_ind:
            continue

        all_ind[sym] = tf_ind

        # ── 第二遍：逐 TF 检查信号（4H 仅做方向锚，不独立出信号）──
        for tf, ind in tf_ind.items():
            if tf == "4h":
                continue  # 4H 只提供方向确认，信号在 15m/1h 上检测
            direction = get_direction(ind)
            regime = get_regime(ind)
            state = SignalState(symbol=sym, timeframe=tf, ind=ind,
                                bbw_rank=ind.get("bb_width_short_pct"), regime=regime, direction=direction)

            for sig_def in SIGNALS:
                # ── 市场状态门控 ──
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

        # ── 第三遍：所有 TF 数据齐全，统一 enrich ──
        sym_alerts = [a for a in alerts if a.symbol == sym]
        for alert in sym_alerts:
            _enrich_alert(alert, tf_ind, sym, sym_alerts)

    return alerts, all_ind


def _enrich_alert(alert: Alert, tf_ind: dict, sym: str, sym_alerts: list[Alert] | None = None):
    """为 alert 附加 4h 方向 + 1h S/R + SL/TP/RR + 清单"""
    check = []

    # ── 信号共振加权 ──
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

    # ── 4h 方向信息 ──
    ind_4h = tf_ind.get("4h")
    dir_4h = "neutral"
    if ind_4h:
        dir_4h = ind_4h.get("ma_alignment", "neutral")
        alert.meta["4h_ma"] = "多头排列" if dir_4h == "bullish" else "空头排列" if dir_4h == "bearish" else "均线交叉"
        alert.meta["4h_adx"] = f"{ind_4h.get('adx', 0) or 0:.0f}"
        alert.meta["4h_adx_trend"] = "↑" if ind_4h.get("adx_trend") == "up" else "↓"
        alert.meta["4h_bb"] = ind_4h.get("bb_state", "unknown")

    # 1h 方向
    ind_1h = tf_ind.get("1h")
    dir_1h = "neutral"
    if ind_1h:
        dir_1h = ind_1h.get("ma_alignment", "neutral")

    # ── 数据新鲜度检查 ──
    ind_15m = tf_ind.get("15m", {})
    df_15m = ind_15m.get("df")
    df_1h = (ind_1h or {}).get("df")
    is_stale = False
    if df_15m is not None and len(df_15m) > 0:
        latest_ts = df_15m["timestamp"].iloc[-1]
        age_min = (datetime.now() - latest_ts).total_seconds() / 60
        if age_min > 30:  # 30分钟无15m更新 → WS或REST都断了
            is_stale = True
    elif df_1h is not None and len(df_1h) > 0:
        latest_ts = df_1h["timestamp"].iloc[-1]
        age_min = (datetime.now() - latest_ts).total_seconds() / 60
        if age_min > 75:  # 1H超过75分钟无新K
            is_stale = True
    if is_stale:
        alert.meta["stale_data"] = True
        check.append("⚠数据陈旧")

    # ── 方向确认：至少一个高 TF 同意信号方向，优先信 4h ──
    sig_dir = alert.direction
    if dir_4h in ("bullish", "bearish"):
        if (dir_4h == "bullish" and sig_dir == "long") or (dir_4h == "bearish" and sig_dir == "short"):
            check.append("✓方向")
        else:
            check.append("✗方向")
    elif dir_1h in ("bullish", "bearish"):
        if (dir_1h == "bullish" and sig_dir == "long") or (dir_1h == "bearish" and sig_dir == "short"):
            check.append("✓方向(1h)")
        else:
            check.append("✗方向")
    else:
        check.append("?方向")

    # ── 1h S/R + SL/TP/RR ──
    ind_base = ind_1h or tf_ind.get("15m", {})
    current_price = ind_base.get("close")
    atr = ind_base.get("atr") or 1

    sr_info = {}
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

        # 位置
        pos_in_range = None
        if support and resistance and resistance.price > support.price:
            pos_in_range = (current_price - support.price) / (resistance.price - support.price)
        if alert.direction == "long" and support:
            if pos_in_range is not None and pos_in_range >= 0.7:
                check.append("✗位置")  # 做多但价格在阻力附近
            elif pos_in_range is not None and pos_in_range <= 0.3:
                check.append("✓近支撑")
            elif support.touch_count >= 2:
                check.append("✓有支撑")
            else:
                check.append("?无支撑")
        elif alert.direction == "short" and resistance:
            if pos_in_range is not None and pos_in_range <= 0.3:
                check.append("✗位置")  # 做空但价格在支撑附近
            elif pos_in_range is not None and pos_in_range >= 0.7:
                check.append("✓近阻力")
            elif resistance.touch_count >= 2:
                check.append("✓有阻力")
            else:
                check.append("?无阻力")
        else:
            check.append("?边界")

    # ── SL/TP/RR 计算 ──
    entry_price = current_price or ind_base.get("close") or 0
    sl = tp = 0

    if alert.direction == "long":
        sl = sr_info["support"].price - atr * 0.3 if "support" in sr_info else entry_price - atr * 1.5
        tp = sr_info["resistance"].price if "resistance" in sr_info else entry_price + atr * 2.5
        # 确保 tp 足够远
        if tp <= sl or tp <= entry_price or (tp - entry_price) < (entry_price - sl) * 1.5:
            tp = entry_price + atr * 2.5
    elif alert.direction == "short":
        sl = sr_info["resistance"].price + atr * 0.3 if "resistance" in sr_info else entry_price + atr * 1.5
        tp = sr_info["support"].price if "support" in sr_info else entry_price - atr * 2.5
        if tp >= sl or tp >= entry_price or (entry_price - tp) < (sl - entry_price) * 1.5:
            tp = entry_price - atr * 2.5
    else:
        sl = entry_price - atr * 1.5
        tp = entry_price + atr * 1.5

    sf = "{:.0f}" if entry_price > 100 else "{:.1f}" if entry_price > 1 else "{:.5f}"
    alert.meta["sl"] = sf.format(sl)
    alert.meta["tp"] = sf.format(tp)
    sl_dist = abs(entry_price - sl)
    tp_dist = abs(tp - entry_price)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0
    alert.meta["rr"] = f"{rr:.1f}:1"

    # ── 最优入场（防线价）计算 ──
    # 如果在防线位置入场，RR 有多大？距当前价 ≤2ATR 才显示。
    opt_entry_price = None
    opt_rr_val = 0
    touches = 0
    if entry_price > 0 and atr > 0:
        if alert.direction == "long" and "support" in sr_info:
            support_lvl = sr_info["support"]
            touches = support_lvl.touch_count
            opt_entry_price = support_lvl.price
            if opt_entry_price < entry_price:
                opt_sl = opt_entry_price - atr * 0.3
                resistance_lvl = sr_info.get("resistance")
                opt_tp = resistance_lvl.price if resistance_lvl else opt_entry_price + atr * 3.0
                if opt_tp > opt_entry_price and opt_sl < opt_entry_price:
                    opt_sl_dist = opt_entry_price - opt_sl
                    opt_tp_dist = opt_tp - opt_entry_price
                    opt_rr_val = opt_tp_dist / opt_sl_dist if opt_sl_dist > 0 else 0
        elif alert.direction == "short" and "resistance" in sr_info:
            resistance_lvl = sr_info["resistance"]
            touches = resistance_lvl.touch_count
            opt_entry_price = resistance_lvl.price
            if opt_entry_price > entry_price:
                opt_sl = opt_entry_price + atr * 0.3
                support_lvl = sr_info.get("support")
                opt_tp = support_lvl.price if support_lvl else opt_entry_price - atr * 3.0
                if opt_tp < opt_entry_price and opt_sl > opt_entry_price:
                    opt_sl_dist = opt_sl - opt_entry_price
                    opt_tp_dist = opt_entry_price - opt_tp
                    opt_rr_val = opt_tp_dist / opt_sl_dist if opt_sl_dist > 0 else 0

        if (opt_entry_price and abs(opt_entry_price - entry_price) <= atr * 2
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

    # ── 成交量分布防线（量加权 S/R 与极点聚类互补）──
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

                # ── 动态防线升级: 2触极点 + 量节点重合 → 等效3触 ──
                if touches == 2 and opt_entry_price:
                    upgraded = False
                    if alert.direction == "long" and "support" in sr_info:
                        if vp_nodes["support"]:
                            dist = abs(vp_nodes["support"]["price"] - sr_info["support"].price)
                            if dist <= atr * 0.5:
                                # 2触极点 + 量节点在附近 → 升级为左侧挂单
                                if "🟡右侧等K@防线" in check:
                                    check.remove("🟡右侧等K@防线")
                                check.append("🔵左侧挂单(量升级)")
                                upgraded = True
                    elif alert.direction == "short" and "resistance" in sr_info:
                        if vp_nodes["resistance"]:
                            dist = abs(vp_nodes["resistance"]["price"] - sr_info["resistance"].price)
                            if dist <= atr * 0.5:
                                if "🟡右侧等K@防线" in check:
                                    check.remove("🟡右侧等K@防线")
                                check.append("🔵左侧挂单(量升级)")
                                upgraded = True
                    if upgraded:
                        alert.details["vp_upgrade"] = True
        except Exception:
            pass

    # ── 4H 方向分级: 逆4H但防线极强 → 降级标记而非丢弃 ──
    if "✗方向" in check and dir_4h in ("bullish", "bearish"):
        # 检查防线强度
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
            check.append("⚠逆4H")
            alert.details["counter_4h"] = True

    # ── 仓位计算: 保证金占比 = 入场价 × risk_pct / (SL距离 × leverage) × 100% ──
    try:
        with open("config.yaml") as f:
            import yaml
            acct = yaml.safe_load(f).get("account", {})
        risk_pct = acct.get("risk_pct", 15)
        leverage = acct.get("leverage", 10)
        if sl_dist > 0:
            margin_pct = entry_price * risk_pct / (sl_dist * leverage)
            if margin_pct <= 100:
                alert.meta["margin"] = f"保证金{margin_pct:.0f}%({leverage}x)"
            else:
                need_lev = int(margin_pct * leverage / 100) + 1
                alert.meta["margin"] = f"保证金{margin_pct:.0f}%({leverage}x)→需{need_lev}x"
    except Exception:
        pass

    if rr >= 2.0:
        check.append("✓盈亏比")
    elif rr >= 1.5:
        check.append("⚠盈亏比")
    else:
        check.append("✗盈亏比")

    # ── 压缩 & 量 ──
    comp_bars = ind_base.get("compression_bars", 0)
    if comp_bars >= 6:
        check.append(f"✓压缩{comp_bars}根")
    elif comp_bars >= 3:
        check.append(f"⚠压缩{comp_bars}根")

    vr = ind_base.get("volume_ratio") or 1
    if vr >= 1.5:
        check.append(f"✓放量{vr:.1f}x")

    # ── Stage2 升级: 近边界 + 趋势/回归组合满足 → 🟢 标注 ──
    ind_15m = tf_ind.get("15m", {})
    is_near_boundary = any(c.startswith("✓") for c in check if "支撑" in c or "阻力" in c)
    if is_near_boundary and ind_15m:
        adx_15 = ind_15m.get("adx", 0) or 0
        adx_1h = (tf_ind.get("1h") or {}).get("adx", 0) or 0
        roc_15 = ind_15m.get("roc") or 0
        rsi_15 = ind_15m.get("rsi")
        vr_15 = ind_15m.get("volume_ratio") or 1

        # 趋势突破升级
        if adx_15 >= 25 and adx_1h >= 20 and abs(roc_15) >= 1.0 and vr_15 >= 1.5:
            if (alert.direction == "long" and roc_15 > 0) or (alert.direction == "short" and roc_15 < 0):
                alert.severity = "critical"
                alert.details["stage2_upgrade"] = "trend"
                check.append("🟢趋势突破")

        # 震荡回归升级
        elif adx_15 < 20 and rsi_15 is not None:
            if (alert.direction == "long" and rsi_15 <= 30) or (alert.direction == "short" and rsi_15 >= 70):
                alert.severity = "critical"
                alert.details["stage2_upgrade"] = "range"
                check.append("🟢震荡回归")

    # ── 成型判定: Pinbar @ S/R + 放量 → 可入场信号 ──
    pinbar_15 = ind_15m.get("pinbar")
    if pinbar_15 and current_price and atr and vr:
        near_sr = False
        if alert.direction == "long" and pinbar_15 == "bullish" and "support" in sr_info:
            dist = current_price - sr_info["support"].price
            near_sr = 0 <= dist <= atr
        elif alert.direction == "short" and pinbar_15 == "bearish" and "resistance" in sr_info:
            dist = sr_info["resistance"].price - current_price
            near_sr = 0 <= dist <= atr

        if near_sr and vr >= 1.2:
            alert.severity = "critical"
            alert.details["setup"] = True
            check.append("✅成型·Pinbar")

    # ── 出场精密度 ──
    # 量能耗尽: 高量 + 小实体 → 无方向性跟随
    body_pct_15 = ind_15m.get("body_pct", 1)
    vr_15m = ind_15m.get("volume_ratio") or 1
    if body_pct_15 < 0.3 and vr_15m >= 2.5:
        check.append("⚠量能耗尽")

    # ATR 移动止损建议（入场后价格有利移动 2ATR 时移损至保本）
    if entry_price > 0 and atr > 0:
        sf = "{:.0f}" if entry_price > 100 else "{:.1f}" if entry_price > 1 else "{:.5f}"
        if alert.direction == "long":
            move_price = entry_price + atr * 2
            if tp > entry_price and move_price < tp:
                alert.meta["trailing"] = f"移损@{sf.format(entry_price+atr*0.5)}(+2ATR移保本)"
        elif alert.direction == "short":
            move_price = entry_price - atr * 2
            if tp < entry_price and move_price > tp:
                alert.meta["trailing"] = f"移损@{sf.format(entry_price-atr*0.5)}(+2ATR移保本)"

    # ── 分批止盈 ──
    if entry_price > 0 and atr > 0 and tp != 0:
        sf = "{:.0f}" if entry_price > 100 else "{:.1f}" if entry_price > 1 else "{:.5f}"
        if alert.direction == "long" and tp > entry_price:
            tp1 = entry_price + atr * 0.8
            if tp1 < tp:
                alert.meta["tp1"] = f"{sf.format(tp1)}(40%)"
                alert.meta["tp2"] = f"{sf.format(tp)}(30%)"
            else:
                alert.meta["tp_full"] = f"{sf.format(tp)}(70%)"
        elif alert.direction == "short" and tp < entry_price:
            tp1 = entry_price - atr * 0.8
            if tp1 > tp:
                alert.meta["tp1"] = f"{sf.format(tp1)}(40%)"
                alert.meta["tp2"] = f"{sf.format(tp)}(30%)"
            else:
                alert.meta["tp_full"] = f"{sf.format(tp)}(70%)"

    alert.checklist = check


def apply_mtf_boost(alerts: list[Alert]):
    """多时间框架确认：同一 signal_type 在多个 TF 出现且方向一致 → 提升置信度"""
    grouped: dict[str, list[Alert]] = {}
    for a in alerts:
        key = f"{a.symbol}_{a.signal_type}"
        grouped.setdefault(key, []).append(a)

    for key, group in grouped.items():
        tfs = set(a.timeframe for a in group)
        if len(tfs) >= 2:
            # 检查方向一致性
            dirs = set(a.direction for a in group if a.direction != "neutral")
            if len(dirs) <= 1:  # 所有信号同向或中性
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

    # 初始化
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

    # 持仓推送用独立群
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

    # 健康检查
    if config.get("health", {}).get("enabled", True):
        start_health_server(config["health"]["port"])

    # 预填 REST 历史数据
    for sym in symbols:
        for tf in timeframes:
            try:
                for bar in okx.fetch_ohlcv(sym, tf, limit=200):
                    cache.update(sym, tf, Candle(
                        timestamp=bar["timestamp"],
                        open=bar["open"], high=bar["high"],
                        low=bar["low"], close=bar["close"], volume=bar["volume"],
                    ))
            except Exception as e:
                logger.warning(f"Prefill {sym} {tf}: {e}")
    logger.info("Cache prefill done")

    # WebSocket 连接
    def on_kline(sym: str, tf: str, candle: Candle):
        cache.update(sym, tf, candle)

    try:
        await okx.ws_connect(symbols, timeframes, on_kline)
        await asyncio.sleep(10)  # 等初批数据
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        return

    # 后台线程：历史数据下载
    def history_thread():
        from history import download_and_compute
        download_and_compute(okx, symbols, timeframes, config.get("history", {}))

    threading.Thread(target=history_thread, daemon=True, name="history").start()

    # 持仓监控
    pos_thread = None
    pos_cfg = config.get("position", {})
    if pos_cfg.get("enabled", True) and okx_cfg.get("api_key"):
        from position import PositionMonitor
        pm = PositionMonitor(config, feishu_position, secrets)
        pos_thread = threading.Thread(target=pm.run, daemon=True, name="position")
        pos_thread.start()

    # 主扫描循环
    interval = config.get("scan_interval", 120)
    alert_filter = AlertFilter(
        silence_minutes=config.get("alert", {}).get("dedup_minutes", 30),
        min_confidence=config.get("alert", {}).get("min_confidence", 0.65),
    )
    scan_count = 0
    async def _refresh_cache_if_stale():
        refreshed = 0
        for sym in symbols:
            for tf in timeframes:
                try:
                    bars = okx.fetch_ohlcv(sym, tf, limit=5)
                    for bar in bars:
                        cache.update(sym, tf, Candle(
                            timestamp=bar["timestamp"], open=bar["open"],
                            high=bar["high"], low=bar["low"],
                            close=bar["close"], volume=bar["volume"]))
                    refreshed += 1
                except Exception:
                    pass
        if refreshed > 0:
            logger.debug(f"REST cache refresh: {refreshed} bars")

    while True:
        await asyncio.sleep(interval)
        scan_count += 1
        scan_start = datetime.now()
        try:
            await _refresh_cache_if_stale()
            alerts, all_ind = do_scan(symbols, cache, config)

            # 多时间框架确认
            apply_mtf_boost(alerts)

            # 置信度增强
            for a in alerts:
                ind = all_ind.get(a.symbol, {}).get(a.timeframe, {})
                a.confidence = alert_filter._boost_confidence(a, ind)

            # 去重过滤
            filtered = alert_filter.filter(alerts)

            # TOP5 排序
            ranks = rank_symbols(alerts, all_ind) if filtered else []

            # 合并推送（按品类分组）
            if filtered:
                # 按品类分桶
                buckets: dict[str, list[Alert]] = {}
                for a in filtered:
                    cat = _symbol_category(a.symbol)
                    buckets.setdefault(cat, []).append(a)

                for cat, cat_alerts in buckets.items():
                    cat_label = _CAT_LABELS.get(cat, cat)
                    cat_ranks = [r for r in ranks if _symbol_category(r.symbol) == cat]
                    report = format_consolidated_report(
                        cat_alerts, cat_ranks, len(alerts), len(symbols), scan_start,
                        category_label=cat_label)
                    feishu.send(report)
                logger.info(f"Scan #{scan_count}: {len(filtered)} alerts, {len(ranks)} ranked ({'/'.join(f'{k}:{len(v)}' for k,v in buckets.items())})")
            else:
                logger.info(f"Scan #{scan_count}: 0 push alerts (raw {len(alerts)} signals scanned)")

        except Exception as e:
            logger.error(f"Scan #{scan_count} error: {e}")


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    shutdown_flag = False

    def _shutdown(sig, frame):
        nonlocal shutdown_flag
        shutdown_flag = True
        logger.info(f"Signal {sig} received, shutting down...")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(async_main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
