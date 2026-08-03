#!/usr/bin/env python3
"""backtest.db 数据加载 + 完整性检查 + 多周期对齐

数据语义:
- 时间戳 = bar 开盘时间 (OKX), 全部 UTC
- 研究只能用已收盘 bar: bar 的数据在 open + tf 时长后才可用

多周期对齐 (无未来函数边界):
- 低位 bar 时间 t 只能使用满足 open + tf 时长 <= t 的最近一根高位 bar
- 例: 4H bar 00:00-04:00, 在 1H bar 03:00 时尚未收盘 → 禁用
"""
import sqlite3

import numpy as np
import pandas as pd

from research.caliber import TF_HOURS

DB_PATH = "data/backtest.db"


def load_candles(db_path: str = DB_PATH,
                 timeframes=("1h", "4h")) -> dict[str, dict[str, pd.DataFrame]]:
    """{symbol: {tf: DataFrame(DatetimeIndex, ohlcv)}} — 时间升序, 已去重清洗"""
    conn = sqlite3.connect(db_path)
    out = {}
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol, timeframe FROM candles ORDER BY symbol").fetchall()
        for sym, tf in rows:
            if tf not in timeframes:
                continue
            df = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume FROM candles "
                "WHERE symbol=? AND timeframe=? ORDER BY timestamp",
                conn, params=(sym, tf))
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
            df = df.sort_index()
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[~df["close"].isna()]
            out.setdefault(sym, {})[tf] = df
    finally:
        conn.close()
    return out


def verify(df: pd.DataFrame, symbol: str = "", tf: str = "") -> list[str]:
    """数据完整性检查 — 返回问题列表 (空 = 干净)"""
    problems = []
    if df.index.has_duplicates:
        problems.append("duplicate timestamps")
    if not df.index.is_monotonic_increasing:
        problems.append("index not monotonic")
    o, h, l, c = (df[col].values for col in ("open", "high", "low", "close"))
    bad_ohlc = (h < l) | (h < np.maximum(o, c)) | (l > np.minimum(o, c))
    n_bad = int(np.sum(bad_ohlc))
    if n_bad:
        problems.append(f"OHLC 不自洽 {n_bad} 根 (high<low 或 high<max(o,c) 等)")
    n_na = int(df.isna().sum().sum())
    if n_na:
        problems.append(f"{n_na} 个 NaN")
    if len(df) < 2:
        problems.append("数据不足 2 根")
    for p in problems:
        print(f"[verify] {symbol} {tf}: {p}", flush=True)
    return problems


def align_higher(higher: pd.DataFrame, higher_tf: str,
                 lower_index: pd.DatetimeIndex, cols=None) -> pd.DataFrame:
    """把高位 bar 的已收盘特征对齐到每个低位 bar 时间戳 (无未来函数)

    返回 DataFrame (行数 = len(lower_index)), 每行是当时已收盘的最近一根高位 bar;
    若当时还没有任何已收盘的高位 bar → NaN。
    """
    if higher is None or higher.empty:
        return pd.DataFrame(np.nan, index=lower_index, columns=cols or ["close"])
    dur = pd.Timedelta(hours=TF_HOURS[higher_tf])
    closed = (higher.index + dur).values.astype("datetime64[ns]")
    t = lower_index.values.astype("datetime64[ns]")
    pos = np.searchsorted(closed, t, side="right") - 1
    cols = list(cols) if cols else list(higher.columns)
    out = {}
    for c in cols:
        v = higher[c].values
        res = np.full(len(t), np.nan)
        m = pos >= 0
        res[m] = v[pos[m]]
        out[c] = res
    return pd.DataFrame(out, index=lower_index)


def daily_resample(df_4h: pd.DataFrame) -> pd.DataFrame:
    """4H → 日线 (last bar 重采样, 已收盘) — 用于日线状态研究"""
    daily = df_4h.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"])
    return daily
