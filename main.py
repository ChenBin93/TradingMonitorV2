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
from signals import SIGNALS, SignalState, get_direction, get_regime, check_stage2_entry, EntrySignal
from notify import Feishu
from utils import setup_logging, start_health_server
from support_resistance import find_swing_levels, get_nearest_levels


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
        # 同 symbol/timeframe 只保留最高置信度
        deduped: dict[str, Alert] = {}
        for a in result:
            k = f"{a.symbol}_{a.timeframe}"
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


# =============================================================================
# 飞书报告格式化
# =============================================================================

def fmt_short_alert(a: Alert) -> str:
    """单条预警的紧凑格式"""
    sym = a.symbol.replace("-USDT-SWAP", "/USDT").split(":")[0]
    dir_map = {"long": "多", "short": "空"}
    d = dir_map.get(a.direction, "")
    m = a.meta
    severity_icon = "🔴" if a.severity == "critical" else "🟠"
    rr = m.get("rr", "-")
    s = m.get("support", "-")
    r = m.get("resistance", "-")
    checks = " ".join([c for c in a.checklist if c.startswith("✓") or c.startswith("⚠")][:4])
    role = a.tf_role
    return f"{severity_icon} {sym}[{a.timeframe}·{role}] {d} {a.name} RR:{rr}\n   S:{s} R:{r} | {checks}"


def format_consolidated_report(
    filtered: list[Alert],
    ranks: list[SymbolRank],
    stage2: list[EntrySignal],
    total_alerts: int,
    symbol_count: int,
    scan_time: datetime,
) -> str:
    """合并预警 + 排名 + Stage2 为一条消息"""
    time_str = scan_time.strftime("%H:%M")

    # 低质量过滤：RR < 1.5 且非 critical 的不推
    # 中间位置 + 无MTF确认 → 降级（回测胜率仅 35%）
    quality = []
    for a in filtered:
        rr_str = a.meta.get("rr", "0:1").split(":")[0]
        try:
            rr_val = float(rr_str)
        except ValueError:
            rr_val = 0

        # RR 硬门槛
        if rr_val < 1.5 and a.severity != "critical":
            continue

        # 中间位置 + 无 MTF → 只保留 critical
        mtf_boost = a.details.get("mtf_boost", False)
        is_mid = any(c.startswith("?") for c in a.checklist if "边界" in c or "支撑" in c or "阻力" in c)
        if is_mid and not mtf_boost and a.severity != "critical":
            continue

        quality.append(a)

    # 按排名顺序对齐
    rank_order = {r.symbol: i for i, r in enumerate(ranks)}
    quality.sort(key=lambda a: (rank_order.get(a.symbol, 999), -a.confidence))

    # 多空统计
    longs = sum(1 for a in quality if a.direction == "long")
    shorts = sum(1 for a in quality if a.direction == "short")

    lines = [f"━━━ V2 扫描 {time_str} ━━━",
             f"{symbol_count}币 | {total_alerts}信号 | 推送{len(quality)}条 | 多{longs}/空{shorts}"]

    # ── Stage2 入场 (稀有，放最前面) ──
    if stage2:
        lines.append("")
        for e in stage2[:3]:
            sym = e.symbol.replace("-USDT-SWAP", "/USDT").split(":")[0]
            dir_text = "多" if e.direction == "long" else "空"
            sig_type = "趋势突破" if "trend" in e.signal_type else "震荡回归"
            lines.append(f"🟢 {sym} Stage2 {sig_type}{dir_text} 入场:{e.entry_price} SL:{e.stop_loss} TP:{e.take_profit} RR:{e.risk_reward}:1")

    # ── 预警列表 ──
    if quality:
        lines.append("")
        for i, a in enumerate(quality[:8]):
            lines.append(fmt_short_alert(a))

    # ── TOP5 排名 ──
    if ranks:
        lines.append(f"\n━━━ TOP{min(5, len(ranks))} ━━━")
        for i, r in enumerate(ranks[:5], 1):
            sym = r.symbol.replace("-USDT-SWAP", "/USDT").split(":")[0]
            d = {"long": "多", "short": "空"}.get(r.direction, "")
            tags = "/".join(r.signal_tags[:3])
            reason = "/".join(r.reasons[:2])
            lines.append(f"{i}. {sym} {d} {r.score:.0%} | {tags} | {reason}")

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
) -> tuple[list[Alert], dict[str, dict[str, dict]], list[EntrySignal]]:
    """返回: (alerts, {sym: {tf: ind}}, stage2_entries)"""
    timeframes = config["timeframes"]
    alerts: list[Alert] = []
    all_ind: dict[str, dict[str, dict]] = {}
    stage2_entries: list[EntrySignal] = []

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

        # ── 第二遍：逐 TF 检查信号 ──
        for tf, ind in tf_ind.items():
            direction = get_direction(ind)
            regime = get_regime(ind)
            state = SignalState(symbol=sym, timeframe=tf, ind=ind,
                                bbw_rank=ind.get("bb_width_short_pct"), regime=regime, direction=direction)

            for sig_def in SIGNALS:
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
            _enrich_alert(alert, tf_ind, sym)

        # Stage2 入场
        if "15m" in tf_ind:
            entry = check_stage2_entry(tf_ind)
            if entry:
                entry.symbol = sym
                stage2_entries.append(entry)

    return alerts, all_ind, stage2_entries


def _enrich_alert(alert: Alert, tf_ind: dict, sym: str):
    """为 alert 附加 4h 方向 + 1h S/R + SL/TP/RR + 清单"""
    check = []

    # ── 4h 方向信息 ──
    ind_4h = tf_ind.get("4h")
    if ind_4h:
        ma = ind_4h.get("ma_alignment", "neutral")
        alert.meta["4h_ma"] = "多头排列" if ma == "bullish" else "空头排列" if ma == "bearish" else "均线交叉"
        alert.meta["4h_adx"] = f"{ind_4h.get('adx', 0) or 0:.0f}"
        alert.meta["4h_adx_trend"] = "↑" if ind_4h.get("adx_trend") == "up" else "↓"
        alert.meta["4h_bb"] = ind_4h.get("bb_state", "unknown")

        if ma in ("bullish", "bearish"):
            if (ma == "bullish" and alert.direction == "long") or (ma == "bearish" and alert.direction == "short"):
                check.append("✓方向")
            else:
                check.append("✗方向")
        else:
            check.append("?方向")

    # ── 1h S/R + SL/TP/RR ──
    ind_1h = tf_ind.get("1h")
    ind_base = ind_1h or tf_ind.get("15m", {})
    df_1h = (ind_1h or {}).get("df")
    current_price = ind_base.get("close")
    atr = ind_base.get("atr") or 1

    sr_info = {}
    if df_1h is not None and current_price:
        levels = find_swing_levels(df_1h, lookback=50)
        support, resistance = get_nearest_levels(levels, current_price)

        if support:
            alert.meta["support"] = f"{support.price:.0f}({support.strength},{support.touch_count}触)"
            sr_info["support"] = support
        if resistance:
            alert.meta["resistance"] = f"{resistance.price:.0f}({resistance.strength},{resistance.touch_count}触)"
            sr_info["resistance"] = resistance

        # 位置
        pos_in_range = None
        if support and resistance and resistance.price > support.price:
            pos_in_range = (current_price - support.price) / (resistance.price - support.price)
        if alert.direction == "long" and support:
            if pos_in_range is not None and pos_in_range <= 0.3:
                check.append("✓近支撑")
            elif support.touch_count >= 2:
                check.append("✓有支撑")
            else:
                check.append("?无支撑")
        elif alert.direction == "short" and resistance:
            if pos_in_range is not None and pos_in_range >= 0.7:
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
        # 确保 tp > sl
        if tp <= sl or tp <= entry_price:
            tp = entry_price + atr * 2.5
    elif alert.direction == "short":
        sl = sr_info["resistance"].price + atr * 0.3 if "resistance" in sr_info else entry_price + atr * 1.5
        tp = sr_info["support"].price if "support" in sr_info else entry_price - atr * 2.5
        if tp >= sl or tp >= entry_price:
            tp = entry_price - atr * 2.5
    else:
        sl = entry_price - atr * 1.5
        tp = entry_price + atr * 1.5

    alert.meta["sl"] = f"{sl:.0f}"
    alert.meta["tp"] = f"{tp:.0f}"
    sl_dist = abs(entry_price - sl)
    tp_dist = abs(tp - entry_price)
    rr = tp_dist / sl_dist if sl_dist > 0 else 0
    alert.meta["rr"] = f"{rr:.1f}:1"

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


def format_stage2_report(entries: list[EntrySignal], scan_time: datetime) -> str:
    if not entries:
        return ""
    time_str = scan_time.strftime("%Y-%m-%d %H:%M")
    lines = [f"\n[Stage2入场] {time_str}"]
    for e in entries[:5]:
        sym = e.symbol.replace("-USDT-SWAP", "/USDT")
        dir_text = "做多" if e.direction == "long" else "做空"
        sig_type = "趋势突破" if "trend" in e.signal_type else "震荡回归"
        lines.append(f"  {sym} {sig_type}{dir_text} 入场:{e.entry_price} 止损:{e.stop_loss} 止盈:{e.take_profit} RR:{e.risk_reward}:1")
        lines.append(f"  {e.evidence}")
    return "\n".join(lines)


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

    while True:
        await asyncio.sleep(interval)
        scan_count += 1
        scan_start = datetime.now()
        try:
            alerts, all_ind, stage2 = do_scan(symbols, cache, config)

            # 多时间框架确认：同 signal 在多 TF 出现 → 置信度提升
            apply_mtf_boost(alerts)

            # 置信度增强
            for a in alerts:
                ind = all_ind.get(a.symbol, {}).get(a.timeframe, {})
                a.confidence = alert_filter._boost_confidence(a, ind)

            # 去重过滤
            filtered = alert_filter.filter(alerts)

            # TOP5 排序
            ranks = rank_symbols(alerts, all_ind) if filtered else []

            # 合并推送
            if filtered or stage2:
                report = format_consolidated_report(
                    filtered, ranks, stage2, len(alerts), len(symbols), scan_start)
                feishu.send(report)
                logger.info(f"Scan #{scan_count}: {len(filtered)} alerts, {len(ranks)} ranked, {len(stage2)} stage2")

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
