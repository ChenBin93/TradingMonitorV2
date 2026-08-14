"""OKX 永续 funding rate 历史下载器（M4 学习模块数据基础）。

funding 是加密永续的 carry（每 8h 结算，withRate 口径），映射书 CH13 的
carry/forward-bias 逻辑。数据落 data/funding.db（gitignored）。

表结构：funding(instId TEXT, ts INTEGER, funding_rate REAL, realized_rate REAL, PRIMARY KEY(instId, ts))
分页：OKX funding-rate-history 用 after 参数翻向过去（每次返回 100 条，最新在前）。
"""
import sqlite3
import time

import requests

API = "https://www.okx.com/api/v5/public/funding-rate-history"
SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LINK",
           "LTC", "UNI", "ATOM", "NEAR", "BCH", "OP", "ARB", "INJ", "SUI", "TIA"]
TARGET_START = 1691971200000  # 2023-08-14 00:00 UTC（与 backtest.db 对齐）


def fetch_all(inst_id, start_ts):
    rows, after = [], None
    for _ in range(60):  # 最多 60 页 × 100 = 6000 条
        params = {"instId": f"{inst_id}-USDT-SWAP", "limit": 100}
        if after is not None:
            params["after"] = after  # OKX 语义：after=<ts> 返回更早（更旧）的记录
        r = requests.get(API, params=params, timeout=20)
        data = r.json().get("data", [])
        if not data:
            break  # 空页=到达历史深度上限（该端点约 3 个月）
        for d in data:
            ts = int(d["fundingTime"])
            if ts >= start_ts:
                rows.append((f"{inst_id}-USDT-SWAP", ts,
                             float(d["fundingRate"]),
                             float(d.get("realizedRate", 0))))
        oldest = int(data[-1]["fundingTime"])
        if oldest < start_ts:
            break
        after = oldest - 1
        time.sleep(0.15)
    return rows


def main():
    conn = sqlite3.connect("data/funding.db")
    conn.execute("CREATE TABLE IF NOT EXISTS funding ("
                 "instId TEXT, ts INTEGER, funding_rate REAL, realized_rate REAL, "
                 "PRIMARY KEY(instId, ts))")
    total = 0
    for sym in SYMBOLS:
        conn.execute("DELETE FROM funding WHERE instId=?", (f"{sym}-USDT-SWAP",))
        rows = fetch_all(sym, TARGET_START)
        conn.executemany("INSERT OR IGNORE INTO funding VALUES (?,?,?,?)", rows)
        conn.commit()
        rng = (rows[-1][1], rows[0][1]) if rows else (None, None)
        print(f"{sym:6s} rows={len(rows):5d} ts={rng}")
        total += len(rows)
    conn.close()
    print("total", total)


if __name__ == "__main__":
    main()
