# 历史数据：下载 + 统计 + SQLite 分位数查询

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class SymbolStats:
    symbol: str = ""
    timeframe: str = ""
    volatility_percentile_short: float = 0.0
    volatility_percentile_medium: float = 0.0
    volatility_percentile_long: float = 0.0
    return_percentile_short: float = 0.0
    return_percentile_medium: float = 0.0
    return_percentile_long: float = 0.0
    volume_percentile_short: float = 0.0
    volume_percentile_medium: float = 0.0
    volume_percentile_long: float = 0.0
    bb_width_mean: float = 0.0
    bb_width_std: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "timeframe": self.timeframe,
            "volatility_short": self.volatility_percentile_short,
            "volatility_medium": self.volatility_percentile_medium,
            "volatility_long": self.volatility_percentile_long,
            "return_short": self.return_percentile_short,
            "return_medium": self.return_percentile_medium,
            "return_long": self.return_percentile_long,
            "volume_short": self.volume_percentile_short,
            "volume_medium": self.volume_percentile_medium,
            "volume_long": self.volume_percentile_long,
            "bb_width_mean": self.bb_width_mean, "bb_width_std": self.bb_width_std,
        }


# =============================================================================
# SQLite 操作
# =============================================================================

class HistoryDB:
    def __init__(self, db_path: str = "data/history.db"):
        self.db_path = db_path
        self._lock = threading.RLock()

    def init(self):
        os.makedirs(os.path.dirname(self.db_path) or "data", exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS history_candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
                    close REAL NOT NULL, volume REAL NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS symbol_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
                    lookback_days INTEGER NOT NULL, computed_at DATETIME NOT NULL,
                    volatility TEXT, bb_width TEXT, return_stat TEXT,
                    volume TEXT, drawdown TEXT, streak TEXT,
                    UNIQUE(symbol, timeframe, lookback_days, computed_at)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_hc_sym_tf_ts ON history_candles(symbol, timeframe, timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ss_sym_tf ON symbol_stats(symbol, timeframe, lookback_days, computed_at DESC)")
            conn.commit()
            conn.close()

    def has_candles(self, symbol: str, timeframe: str, days: int) -> bool:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            c.execute("""
                SELECT COUNT(*) FROM history_candles
                WHERE symbol=? AND timeframe=? AND timestamp>?
            """, (symbol, timeframe, cutoff))
            count = c.fetchone()[0]
            conn.close()
            return count > 100

    def save_candles(self, candles: list[dict], symbol: str, timeframe: str):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            for bar in candles:
                ts = bar["timestamp"]
                if isinstance(ts, datetime):
                    ts = ts.strftime("%Y-%m-%d %H:%M:%S")
                try:
                    c.execute("""
                        INSERT OR REPLACE INTO history_candles
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, timeframe, ts, bar["open"], bar["high"],
                          bar["low"], bar["close"], bar["volume"]))
                except Exception:
                    pass
            conn.commit()
            conn.close()

    def save_stats(self, symbol: str, timeframe: str, lookback_days: int, stats: dict):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("""
                INSERT OR REPLACE INTO symbol_stats
                (symbol, timeframe, lookback_days, computed_at, volatility, bb_width,
                 return_stat, volume, drawdown, streak)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe, lookback_days, now,
                json.dumps(stats.get("volatility", {})),
                json.dumps(stats.get("bb_width", {})),
                json.dumps(stats.get("return_stat", {})),
                json.dumps(stats.get("volume", {})),
                json.dumps(stats.get("drawdown", {})),
                json.dumps(stats.get("streak", {})),
            ))
            conn.commit()
            conn.close()

    def get_stats(self, symbol: str, timeframe: str, lookback_days: int = 90) -> dict | None:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT * FROM symbol_stats
                WHERE symbol=? AND timeframe=? AND lookback_days=?
                ORDER BY computed_at DESC LIMIT 1
            """, (symbol, timeframe, lookback_days))
            row = c.fetchone()
            conn.close()
            if not row:
                return None
            return {
                "symbol": symbol, "timeframe": timeframe,
                "lookback_days": lookback_days, "computed_at": row["computed_at"],
                "volatility": json.loads(row["volatility"]),
                "bb_width": json.loads(row["bb_width"]),
                "return_stat": json.loads(row["return_stat"]),
                "volume": json.loads(row["volume"]),
                "drawdown": json.loads(row["drawdown"]),
                "streak": json.loads(row["streak"]),
            }


# =============================================================================
# 分位数查询
# =============================================================================

class HistoryManager:
    def __init__(self, db: HistoryDB, default_lookback: int = 90):
        self.db = db
        self.default_lookback = default_lookback
        self._cache: dict[str, dict] = {}

    def get_stats(self, symbol: str, timeframe: str,
                  lookback_days: int | None = None) -> dict | None:
        if lookback_days is None:
            lookback_days = self.default_lookback
        key = f"{symbol}_{timeframe}_{lookback_days}"
        if key in self._cache:
            return self._cache[key]
        stats = self.db.get_stats(symbol, timeframe, lookback_days)
        if stats:
            self._cache[key] = stats
        return stats

    def get_bbw_rank(self, symbol: str, timeframe: str,
                     lookback: int = 90, bb_width: float = 0) -> float | None:
        stats = self.get_stats(symbol, timeframe, lookback)
        if not stats:
            return None
        bb = stats.get("bb_width", {})
        percs = bb.get("percentiles", {})
        if not percs or len(percs) < 2:
            return None
        p_vals = sorted(float(k) for k in percs.keys())
        v_vals = [percs[str(int(p))] for p in p_vals]
        return _interpolate(p_vals, v_vals, bb_width)

    def get_volatility_rank(self, symbol: str, timeframe: str,
                            lookback: int = 90, window: str = "long") -> float:
        stats = self.get_stats(symbol, timeframe, lookback)
        if not stats:
            return 50.0
        vol = stats.get("volatility", {})
        key = "percentiles_short" if window == "short" else "percentiles_medium" if window == "medium" else "percentiles"
        percs = vol.get(key, {})
        if not percs or len(percs) < 2:
            return 50.0
        p_vals = sorted(float(k) for k in percs.keys())
        v_vals = [percs[str(int(p))] for p in p_vals]
        return _interpolate(p_vals, v_vals, v_vals[len(v_vals) // 2])

    def get_percentile_rank(self, symbol: str, timeframe: str, lookback: int,
                            category: str, window: str, current: float) -> float:
        stats = self.get_stats(symbol, timeframe, lookback)
        if not stats:
            return 50.0
        cat = stats.get(category, {})
        if not cat:
            return 50.0
        key = "percentiles_short" if window == "short" else "percentiles_medium" if window == "medium" else "percentiles"
        percs = cat.get(key, {})
        if not percs or len(percs) < 2:
            return 50.0
        p_vals = sorted(float(k) for k in percs.keys())
        v_vals = [percs[str(int(p))] for p in p_vals]
        return _interpolate(p_vals, v_vals, current)


def _interpolate(p_values: list, v_values: list, current: float) -> float:
    if len(p_values) < 2:
        return 50.0
    if current <= v_values[0]:
        return max(0.0, p_values[0])
    if current >= v_values[-1]:
        return min(100.0, p_values[-1])
    for i in range(len(v_values) - 1):
        if v_values[i] <= current <= v_values[i + 1]:
            t = (current - v_values[i]) / (v_values[i + 1] - v_values[i]) if v_values[i + 1] > v_values[i] else 0.5
            return p_values[i] + t * (p_values[i + 1] - p_values[i])
    return 50.0


# =============================================================================
# 后台下载 + 统计
# =============================================================================

def download_and_compute(okx_client, symbols: list[str], timeframes: list[str], config: dict):
    """后台任务：下载历史 K 线 + 计算统计分位数"""
    db = HistoryDB(config.get("db_path", "data/history.db"))
    db.init()

    lookback_days = config.get("lookback_days", [90, 180, 365])
    batch_size = config.get("download_batch_size", 50)
    interval = config.get("download_interval_seconds", 60)

    logger.info(f"[History] Starting download: {len(symbols)} symbols, {lookback_days} days")

    for days in lookback_days:
        for sym in symbols:
            for tf in timeframes:
                if db.has_candles(sym, tf, days):
                    continue
                try:
                    candles = okx_client.fetch_ohlcv(sym, tf, limit=300)
                    if candles:
                        db.save_candles(candles, sym, tf)
                    time.sleep(interval / 1000)  # rate limit
                except Exception as e:
                    logger.debug(f"[History] Download {sym} {tf}: {e}")

    logger.info("[History] Download complete, computing stats...")
    _compute_stats(db, symbols, timeframes, lookback_days)
    logger.info("[History] Stats computation complete")


def _compute_stats(db: HistoryDB, symbols: list[str], timeframes: list[str], lookback_days: list[int]):
    for days in lookback_days:
        for sym in symbols:
            for tf in timeframes:
                try:
                    stats = _calc_symbol_stats(db, sym, tf, days)
                    if stats:
                        db.save_stats(sym, tf, days, stats)
                except Exception as e:
                    logger.debug(f"[History] Stats {sym} {tf} {days}d: {e}")


def _calc_symbol_stats(db: HistoryDB, symbol: str, timeframe: str, days: int) -> dict | None:
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        SELECT open, high, low, close, volume
        FROM history_candles
        WHERE symbol=? AND timeframe=? AND timestamp>?
        ORDER BY timestamp ASC
    """, (symbol, timeframe, cutoff))
    rows = c.fetchall()
    conn.close()

    if len(rows) < 30:
        return None

    closes = np.array([r["close"] for r in rows])
    volumes = np.array([r["volume"] for r in rows])

    returns = np.diff(closes) / closes[:-1]
    volatility = np.abs(returns)

    def _percentiles(arr: np.ndarray) -> dict:
        ps = [5, 10, 25, 50, 75, 90, 95]
        return {str(p): float(np.percentile(arr, p)) for p in ps}

    return {
        "volatility": {"percentiles": _percentiles(volatility)},
        "return_stat": {"percentiles": _percentiles(returns)},
        "volume": {"percentiles": _percentiles(volumes)},
        "bb_width": {"percentiles": {}, "mean": 0.0, "std": 0.0},
        "drawdown": {"percentiles": {}},
        "streak": {"up": {}, "down": {}},
    }
