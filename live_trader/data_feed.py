#!/usr/bin/env python3
"""行情数据源 — paper 模式用

- ReplayFeed: 从 data/backtest.db 回放历史 bar (端到端测试/验证撮合逻辑)
- OkxFeed: ccxt 公开行情 (只取数不交易), 供 paper 模式实时模拟

用法 (回放):
  python3 live_trader/main.py --mode paper --feed replay --symbol "BTC/USDT:USDT" --speed 100
"""
import argparse
import logging
import sqlite3
import sys
import os
import time

import pandas as pd

log = logging.getLogger("trader.feed")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ReplayFeed:
    """从 backtest.db 回放历史 bar — 按真实时间节奏或加速

    warmup_bars: 回放前自动多取 N 根作指标预热 (不计入交易逻辑)。
    track_history: False 时不维护内部 history (外部自行累积, 性能优先)。
    """

    def __init__(self, symbol: str, tf: str, db_path: str = None,
                 speed: float = 1.0, start: str = "", warmup_bars: int = 500,
                 track_history: bool = True):
        db_path = db_path or os.path.join(ROOT, "data/backtest.db")
        self.symbol = symbol
        self.tf = tf
        self.speed = speed
        self.warmup_bars = warmup_bars
        conn = sqlite3.connect(db_path)
        q = ("SELECT timestamp, open, high, low, close, volume FROM candles "
             "WHERE symbol=? AND timeframe=? ORDER BY timestamp")
        params = [symbol, tf]
        if start:
            q += " AND timestamp >= ?"
            params.append(start)
        self.df = pd.read_sql_query(q, conn, params=params)
        conn.close()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)
        self.df = self.df.set_index("timestamp")
        # 预热: 从正片起点前多取 warmup_bars 根
        self.warmup_df = pd.DataFrame()
        if start and len(self.df) > warmup_bars:
            # 重新查询 start 前 warmup 根 (含 start 起点)
            conn = sqlite3.connect(db_path)
            wq = ("SELECT timestamp, open, high, low, close, volume FROM candles "
                  "WHERE symbol=? AND timeframe=? AND timestamp < ? "
                  "ORDER BY timestamp DESC LIMIT ?")
            wdf = pd.read_sql_query(wq, conn, params=[symbol, tf, start, warmup_bars])
            conn.close()
            if not wdf.empty:
                wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], utc=True)
                wdf = wdf.set_index("timestamp").sort_index()
                self.warmup_df = wdf
        self.pos = 0
        self.history: pd.DataFrame = self.warmup_df  # 预热历史
        log.info("ReplayFeed: %s %s %d bars + %d warmup (speed=%sx)", symbol, tf,
                 len(self.df), len(self.warmup_df), speed)

    def next_bar(self) -> dict | None:
        """取下一根 bar; 返回 None 表示回放结束"""
        if self.pos >= len(self.df):
            return None
        bar = self.df.iloc[self.pos].to_dict()
        self.pos += 1
        # 推进 history (只保留最近 120 根供指标: MA60+BB40 需要)
        self.history = pd.concat(
            [self.history, self.df.iloc[[self.pos - 1]]]).tail(120)
        return {**bar, "ts": self.df.index[self.pos - 1]}

    def set_history(self, df: pd.DataFrame):
        """用完整历史初始化指标预热 (回放前调用)"""
        self.history = df.tail(500)


class OkxFeed:
    """ccxt 公开行情 — 只取数不交易, 供 paper 实时模拟"""

    def __init__(self, symbol: str, tf: str):
        import ccxt
        self.ex = ccxt.okx({"enableRateLimit": True})
        self.symbol = symbol
        self.tf = tf
        self.history: pd.DataFrame = pd.DataFrame()
        log.info("OkxFeed: %s %s", symbol, tf)

    def _fetch_page(self, after_ts: int | None = None) -> pd.DataFrame:
        """拉一页 (最多300根); after_ts 时返回其之前的数据 (OKX after 语义)"""
        if after_ts is not None:
            ohlcv = self.ex.fetch_ohlcv(self.symbol, timeframe=self.tf,
                                        limit=300, params={"after": str(after_ts)})
        else:
            ohlcv = self.ex.fetch_ohlcv(self.symbol, timeframe=self.tf, limit=300)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    def warmup(self, n: int = 1600) -> None:
        """分页拉取历史填满窗口 (OKX 单页限 300 根, after 往前翻)"""
        pages = []
        last = None
        while True:
            df = self._fetch_page(last)
            if df.empty:
                break
            pages.append(df)
            last = int(df.index[0].value // 1e6)  # 最早一根 ts(ms), 继续往前
            total = sum(len(p) for p in pages)
            if total >= n:
                break
        if pages:
            self.history = pd.concat(pages).sort_index().tail(n)
        log.info("OkxFeed warmup: %d 根", len(self.history))

    def refresh(self) -> dict:
        self.history = self._fetch_page()
        return self.history.iloc[-1].to_dict()


def replay_main():
    """回放测试入口: python3 live_trader/data_feed.py --symbol ... --speed 100"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--tf", default="5m")
    ap.add_argument("--speed", type=float, default=100.0)
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--bars", type=int, default=500)
    args = ap.parse_args()
    feed = ReplayFeed(args.symbol, args.tf, speed=args.speed, start=args.start)
    interval = 5 * 60 / args.speed  # 5m bar 按 speed 加速
    n = 0
    while n < args.bars:
        bar = feed.next_bar()
        if bar is None:
            print("回放结束")
            break
        print(f"{bar['ts']} O={bar['open']:.1f} H={bar['high']:.1f} "
              f"L={bar['low']:.1f} C={bar['close']:.1f}")
        n += 1
        time.sleep(interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    replay_main()
