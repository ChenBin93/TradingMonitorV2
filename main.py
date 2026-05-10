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


def rank_symbols(alerts: list[Alert], ind_map: dict) -> list[SymbolRank]:
    scores: dict[str, dict] = {}
    for a in alerts:
        s = scores.setdefault(a.symbol, {
            "direction": a.direction, "regime": a.regime, "timeframe": a.timeframe,
            "tags": [], "conf": 0, "momentum": 0.0, "compression": 0.0, "volume": 0.0, "reasons": [],
        })
        s["tags"].append(a.tag)
        s["conf"] = max(s["conf"], a.confidence)

        d = a.details
        if a.signal_type in ("bb_squeeze", "ttm_squeeze"):
            rank_val = d.get("bbw_rank", 50)
            s["compression"] = max(s["compression"], (100 - rank_val) / 100)
        if a.signal_type in ("volume_spike", "volume_breakout"):
            vr = d.get("volume_ratio", d.get("vol_ratio", 1))
            s["volume"] = max(s["volume"], min(vr / 3, 1))

        ind = ind_map.get(a.symbol, {})
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

def format_alert_report(alerts: list[Alert], scan_time: datetime) -> str:
    time_str = scan_time.strftime("%Y-%m-%d %H:%M")
    lines = [f"[信号预警] {time_str}"]

    critical = [a for a in alerts if a.severity == "critical"]
    high = [a for a in alerts if a.severity == "high"]

    if critical:
        lines.append(f"\n【强烈信号】({len(critical)}个)")
        for a in critical:
            sym = a.symbol.replace("-USDT-SWAP", "/USDT")
            dir_text = "做多" if a.direction == "long" else "做空"
            lines.append(f"  {sym}[{a.timeframe}] {dir_text} {a.name} 置信:{a.confidence:.0%}")
            lines.append(f"  {a.evidence}")

    if high:
        lines.append(f"\n【准备信号】({len(high)}个)")
        for a in high[:8]:
            sym = a.symbol.replace("-USDT-SWAP", "/USDT")
            dir_text = "做多" if a.direction == "long" else "做空"
            lines.append(f"  {sym}[{a.timeframe}] {dir_text} {a.name} 置信:{a.confidence:.0%}")

    return "\n".join(lines)


def format_ranking_report(ranks: list[SymbolRank], scan_time: datetime) -> str:
    time_str = scan_time.strftime("%Y-%m-%d %H:%M")
    lines = [f"[TOP5推荐] {time_str}"]

    trending = [r for r in ranks if r.regime == "trend"][:5]
    consolidating = [r for r in ranks if r.regime == "range"][:5]

    if trending:
        lines.append("\n【趋势市场】TOP5:")
        for i, r in enumerate(trending, 1):
            sym = r.symbol.replace("-USDT-SWAP", "/USDT")
            dir_text = "做多" if r.direction == "long" else "做空"
            tags = "/".join(r.signal_tags[:3])
            reason = "/".join(r.reasons)
            lines.append(f"{i}. {sym}[{r.timeframe}] {dir_text} {r.score:.0%}")
            lines.append(f"   理由:{reason} 信号:{tags} 置信:{r.confidence:.0%}")

    if consolidating:
        lines.append("\n【震荡市场】TOP5:")
        for i, r in enumerate(consolidating, 1):
            sym = r.symbol.replace("-USDT-SWAP", "/USDT")
            dir_text = "做多" if r.direction == "long" else "做空"
            tags = "/".join(r.signal_tags[:3])
            reason = "/".join(r.reasons)
            lines.append(f"{i}. {sym}[{r.timeframe}] {dir_text} {r.score:.0%}")
            lines.append(f"   理由:{reason} 信号:{tags} 置信:{r.confidence:.0%}")

    return "\n".join(lines)


# =============================================================================
# 主扫描逻辑
# =============================================================================

def do_scan(
    okx: OKXClient,
    cache: KlineCache,
    config: dict,
) -> tuple[list[Alert], dict[str, dict[str, dict]], list[EntrySignal]]:
    """返回: (alerts, {sym: {tf: ind}}, stage2_entries)"""
    timeframes = config["timeframes"]
    alerts: list[Alert] = []
    ind_map: dict[str, dict] = {}
    all_ind: dict[str, dict[str, dict]] = {}  # {sym: {tf: ind}}
    stage2_entries: list[EntrySignal] = []

    for sym in okx.get_top_symbols(config["top_n"]):
        tf_ind = {}
        for tf in timeframes:
            df = cache.get_df(sym, tf)
            if len(df) < 30:
                continue

            ind_params = config["indicators"].get(tf, config["indicators"]["15m"])
            ind = compute_indicators(df, ind_params)
            if not ind:
                continue

            tf_ind[tf] = ind
            ind_map[sym] = ind  # 最后写入的 tf（一般是 4h）用于 TOP5 排序
            direction = get_direction(ind)
            regime = get_regime(ind)
            state = SignalState(symbol=sym, timeframe=tf, ind=ind,
                                bbw_rank=None, regime=regime, direction=direction)

            for sig_def in SIGNALS:
                try:
                    state.params = sig_def.params
                    result = sig_def.check(state)
                    if result:
                        alerts.append(Alert(
                            symbol=sym, timeframe=tf,
                            signal_type=sig_def.id, signal_name=sig_def.name,
                            regime=regime, direction=result.get("direction", direction),
                            severity=result.get("severity", "medium"),
                            confidence=result.get("confidence", 0.5),
                            evidence=result.get("evidence", ""),
                            details=result,
                        ))
                except Exception as e:
                    logger.debug(f"Signal check error {sig_def.id} {sym}: {e}")

        if tf_ind:
            all_ind[sym] = tf_ind

        # Stage2 入场
        if "15m" in tf_ind:
            entry = check_stage2_entry(tf_ind)
            if entry:
                entry.symbol = sym
                stage2_entries.append(entry)

    return alerts, all_ind, stage2_entries


def apply_mtf_boost(alerts: list[Alert]):
    """多时间框架确认：同一 symbol+signal_type 在多个 TF 出现 → 提升置信度"""
    grouped: dict[str, list[Alert]] = {}
    for a in alerts:
        key = f"{a.symbol}_{a.signal_type}"
        grouped.setdefault(key, []).append(a)

    for key, group in grouped.items():
        tfs = set(a.timeframe for a in group)
        if len(tfs) >= 2:
            boost = 1.10 if len(tfs) == 2 else 1.20  # 2TF +10%, 3TF +20%
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

    feishu_cfg = secrets.get("feishu", {})
    feishu = Feishu(
        app_id=feishu_cfg.get("app_id", ""),
        app_secret=feishu_cfg.get("app_secret", ""),
        chat_id=feishu_cfg.get("chat_id", ""),
    )

    symbols = okx.get_top_symbols(config["top_n"])
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

    await okx.ws_connect(symbols, timeframes, on_kline)
    await asyncio.sleep(10)  # 等初批数据

    # 后台线程：历史数据下载
    def history_thread():
        from history import download_and_compute
        download_and_compute(okx, symbols, timeframes, config.get("history", {}))

    threading.Thread(target=history_thread, daemon=True, name="history").start()

    # 持仓监控
    pos_thread = None
    if okx_cfg.get("api_key"):
        from position import PositionMonitor
        pm = PositionMonitor(config, feishu, secrets)
        pos_thread = threading.Thread(target=pm.run, daemon=True, name="position")
        pos_thread.start()

    # 主扫描循环
    interval = config.get("scan_interval", 120)
    alert_filter = AlertFilter(
        silence_minutes=config.get("alert", {}).get("dedup_minutes", 30),
        min_confidence=config.get("alert", {}).get("min_confidence", 0.65),
    )
    scan_count = 0
    last_ranking_time: datetime | None = None
    ranking_interval = 300  # TOP5 每 5 分钟推送一次

    while True:
        await asyncio.sleep(interval)
        scan_count += 1
        scan_start = datetime.now()
        try:
            alerts, all_ind, stage2 = do_scan(okx, cache, config)

            # 多时间框架确认：同 signal 在多 TF 出现 → 置信度提升
            apply_mtf_boost(alerts)

            # 置信度增强
            for a in alerts:
                ind = all_ind.get(a.symbol, {}).get(a.timeframe, {})
                a.confidence = alert_filter._boost_confidence(a, ind)

            # 去重过滤
            filtered = alert_filter.filter(alerts)

            # 信号预警报告
            if filtered or stage2:
                stage2_text = format_stage2_report(stage2, scan_start) if stage2 else ""
                if filtered:
                    report = format_alert_report(filtered, scan_start) + stage2_text
                    feishu.send(report)
                    logger.info(f"Scan #{scan_count}: {len(filtered)} alerts, {len(stage2)} stage2 pushed")
                elif stage2_text:
                    feishu.send(stage2_text)
                    logger.info(f"Scan #{scan_count}: {len(stage2)} stage2 entries")

            # TOP5 排序
            now = datetime.now()
            if filtered and (not last_ranking_time or (now - last_ranking_time).total_seconds() >= ranking_interval):
                # 用所有 TF 的 ind 数据做排序（取最新有数据的 tf）
                ranks = rank_symbols(alerts, {sym: next(iter(tfs.values())) for sym, tfs in all_ind.items() if tfs})
                if ranks:
                    ranking_report = format_ranking_report(ranks, scan_start)
                    feishu.send(ranking_report)
                    last_ranking_time = now
                    logger.info(f"Scan #{scan_count}: ranking pushed ({len(ranks)} symbols)")

            elapsed = (datetime.now() - scan_start).total_seconds()
            logger.debug(f"Scan #{scan_count}: {len(alerts)} alerts, {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Scan #{scan_count} error: {e}")


def main():
    signal.signal(signal.SIGINT, lambda s, f: asyncio.get_event_loop().stop())
    signal.signal(signal.SIGTERM, lambda s, f: asyncio.get_event_loop().stop())
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutdown")


if __name__ == "__main__":
    main()
