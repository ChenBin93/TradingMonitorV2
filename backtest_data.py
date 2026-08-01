#!/usr/bin/env python3
"""下载回测用历史数据 (5m/1h/4h) — 独立 backtest.db, 不影响 live 数据

OKX history-candles 分页:
  after  = 比请求时间戳更早的数据 (翻向过去)  ← 必须用这个
  before = 比请求时间戳更新的数据 (翻向现在)
每次请求最多 100 根。
"""
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

DB_PATH = "data/backtest.db"

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
    "NEAR/USDT:USDT", "OP/USDT:USDT", "ARB/USDT:USDT", "ATOM/USDT:USDT",
    "SUI/USDT:USDT", "INJ/USDT:USDT", "TIA/USDT:USDT", "UNI/USDT:USDT",
]

TIMEFRAMES = ["5m", "1h", "4h"]
BAR_MAP = {"5m": "5m", "1h": "1H", "4h": "4H"}
DAYS = 60


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    """)
    conn.commit()
    return conn


def fetch_5m_history(symbol: str, timeframe: str, days: int) -> list[dict]:
    inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")
    bar = BAR_MAP[timeframe]
    url = "https://www.okx.com/api/v5/market/history-candles"

    cutoff = datetime.now() - timedelta(days=days)
    all_bars: dict[int, dict] = {}
    after_ts = None

    while True:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after_ts:
            params["after"] = str(after_ts)

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=15)
                data = resp.json()
                break
            except Exception:
                time.sleep(1)
        else:
            break

        if data.get("code") != "0" or not data.get("data"):
            break

        batch = data["data"]
        new_count = 0
        for b in batch:
            ts_ms = int(b[0])
            if ts_ms not in all_bars:
                ts = datetime.fromtimestamp(ts_ms / 1000)
                all_bars[ts_ms] = {
                    "timestamp": ts,
                    "open": float(b[1]), "high": float(b[2]),
                    "low": float(b[3]), "close": float(b[4]), "volume": float(b[5]),
                }
                new_count += 1

        if new_count == 0:
            break

        oldest_ts = datetime.fromtimestamp(int(batch[-1][0]) / 1000)
        after_ts = int(batch[-1][0]) - 1

        if oldest_ts <= cutoff:
            break
        time.sleep(0.12)

    bars = [b for b in all_bars.values() if b["timestamp"] >= cutoff]
    return sorted(bars, key=lambda x: x["timestamp"])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--days", type=int, default=DAYS)
    p.add_argument("--timeframes", nargs="*", default=None)
    p.add_argument("--force", action="store_true", help="跳过已有完整数据")
    args = p.parse_args()

    symbols = args.symbols or SYMBOLS
    timeframes = args.timeframes or TIMEFRAMES

    conn = init_db()
    for sym in symbols:
        for tf in timeframes:
            if not args.force:
                row = conn.execute(
                    "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM candles WHERE symbol=? AND timeframe=?",
                    (sym, tf)).fetchone()
                if row[0] > 1000 and row[2] and row[1]:
                    covered_days = (pd.Timestamp(row[2]) - pd.Timestamp(row[1])).days
                    if covered_days >= args.days:
                        print(f"{sym} {tf}: skip ({row[0]} bars, {covered_days}d)")
                        continue
            try:
                bars = fetch_5m_history(sym, tf, args.days)
                if not bars:
                    print(f"{sym} {tf}: no data", flush=True)
                    continue
                conn.executemany(
                    "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?)",
                    [(sym, tf, b["timestamp"], b["open"], b["high"], b["low"], b["close"], b["volume"]) for b in bars]
                )
                conn.commit()
                first = bars[0]["timestamp"].strftime("%m-%d %H:%M")
                last = bars[-1]["timestamp"].strftime("%m-%d %H:%M")
                print(f"{sym} {tf}: {len(bars)} bars ({first} → {last})", flush=True)
            except Exception as e:
                print(f"{sym} {tf}: ERROR {e}", flush=True)

    conn.close()
    print("Done")


if __name__ == "__main__":
    main()
