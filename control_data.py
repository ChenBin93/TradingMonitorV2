"""对照市场数据下载器：Kaufman 书中所用传统市场（股指/原油/黄金/外汇/利率）。

用途：学习计划的"学习级"考证需要传统市场对照组——验证书中结论在传统市场成立，
才能说明加密市场的证伪是加密的特殊性。数据落 data/control.db（gitignored）。

表结构：{ticker}_{freq}（freq ∈ 1d/1h），列 ts(UTC epoch 秒)/open/high/low/close/volume。
yfinance 限制：1h 历史约 730 天；期货 1h 约 2 年；自动调整关闭（auto_adjust=False，
用原始价，与 OKX 数据口径一致：不复权）。
"""
import sqlite3
import time

import pandas as pd
import yfinance as yf

BASKET = {
    "SPY": "equity index (S&P500 ETF)",
    "CL=F": "crude oil futures",
    "GC=F": "gold futures",
    "EURUSD=X": "fx",
    "^TNX": "10y treasury yield",
}
FREQS = {"1d": "max", "1h": "730d"}  # 日线取全历史（书口径需要长历史）；1h 受 yfinance 730d 限制


def _fetch(ticker, freq, period):
    df = yf.download(ticker, period=period, interval=freq,
                     auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return []
    df = df.reset_index()
    # yfinance 1.2.x 部分标的返回 MultiIndex 列 (Price, Ticker) —— 取首层并小写
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c).lower() for c in df.columns.get_level_values(0)]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    ts_col = "date" if "date" in df.columns else "datetime"
    ts = df[ts_col]
    if hasattr(ts, "dt"):
        df["ts"] = (ts.astype("int64") // 10**9).astype("int64")
    else:
        df["ts"] = ts.astype("int64")
    cols = ["ts", "open", "high", "low", "close", "volume"]
    return df[cols].values


def main():
    conn = sqlite3.connect("data/control.db")
    for ticker, kind in BASKET.items():
        for freq, period in FREQS.items():
            table = f"{ticker}_{freq}"
            rows = _fetch(ticker, freq, period)
            if rows is None or len(rows) == 0:
                print(f"{ticker:10s} {freq} FAIL (0 rows)")
                continue
            # 表名含特殊字符（CL=F、^TNX）——SQL 标识符用双引号包裹
            conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{table}" '
                "(ts INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, "
                "close REAL, volume REAL)")
            conn.execute(f'DELETE FROM "{table}"')
            conn.executemany(
                f'INSERT INTO "{table}" VALUES (?,?,?,?,?,?)', rows)
            conn.commit()
            print(f"{ticker:10s} {kind:26s} {freq} rows={len(rows)} "
                  f"({rows[0][0]}~{rows[-1][0]})")
            time.sleep(1)  # 温和节流
    conn.close()


if __name__ == "__main__":
    main()
