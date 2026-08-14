"""OKX 期权数据下载器 v2（c57 定向版）。

只下载近期 2σ 带触碰事件实际需要的跨式合约（ATM strike × 最近在列到期），
大幅收窄范围。数据落 data/options.db。

表结构：options_meta(inst_id PK, uly, opt_type, strike, exp_ts, list_ts)
        options_candles(inst_id, ts, open, high, low, close, PK(inst_id, ts))
"""
import sqlite3
import datetime
import time

import numpy as np
import requests

BASE = "https://www.okx.com/api/v5"
WINDOW_START_TS = 1782000000000  # 2026-06-24（近期触碰窗口）
LIVE_EXPIRIES = ["260815", "260821", "260828", "260904"]  # 周/月度在列到期（≥7 天规则内使用）
STRIKE_STEP = {"BTC": 500, "ETH": 25}


def _json(url, params):
    for _ in range(3):
        try:
            r = requests.get(url, params=params, timeout=20)
            return r.json()
        except Exception:
            time.sleep(1)
    return {"code": -1, "data": []}


def touches(uly):
    conn = sqlite3.connect("data/backtest.db")
    rows = conn.execute(
        "SELECT timestamp, close FROM candles WHERE symbol=? AND timeframe='1h' AND timestamp >= ? ORDER BY timestamp",
        (f"{uly}/USDT:USDT", datetime.datetime.fromtimestamp(WINDOW_START_TS / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
    ).fetchall()
    conn.close()
    ts = np.array([int(datetime.datetime.fromisoformat(r[0]).timestamp() * 1000) for r in rows])
    close = np.array([r[1] for r in rows], dtype=float)
    n = len(close)
    ma = np.full(n, np.nan); sd = np.full(n, np.nan)
    for i in range(19, n):
        w = close[i - 19:i + 1]
        ma[i] = w.mean(); sd[i] = w.std(ddof=0)
    out = []
    for i in range(20, n):
        if sd[i] <= 0 or not np.isfinite(sd[i]):
            continue
        if close[i] > ma[i] + 2 * sd[i] or close[i] < ma[i] - 2 * sd[i]:
            out.append((int(ts[i]), float(close[i])))
    return out


def expiry_for(touch_ts):
    """触碰时点选最近在列到期（≥7 天剩余）。"""
    for e in LIVE_EXPIRIES:
        exp_dt = datetime.datetime.strptime("20" + e, "%Y%m%d").replace(tzinfo=datetime.timezone.utc)
        exp_ts = int(exp_dt.timestamp() * 1000)
        if exp_ts - touch_ts >= 7 * 86400000:
            return e, exp_ts
    return LIVE_EXPIRIES[-1], int(datetime.datetime.strptime("20" + LIVE_EXPIRIES[-1], "%Y%m%d")
                                  .replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def build_needed():
    needed = set()
    for uly in ["BTC", "ETH"]:
        step = STRIKE_STEP[uly]
        for ts, px in touches(uly):
            strike = int(round(px / step) * step)
            exp_code, exp_ts = expiry_for(ts)
            for opt_type in ["C", "P"]:
                needed.add((f"{uly}-USD-{exp_code}-{strike}-{opt_type}", uly, opt_type, float(strike), exp_ts))
    return needed


def fetch_candles(inst_id):
    rows, after = [], None
    for _ in range(40):
        p = {"instId": inst_id, "bar": "1H", "limit": 100}
        if after is not None:
            p["after"] = after
        d = _json(f"{BASE}/market/history-candles", p).get("data", [])
        if not d:
            break
        for x in d:
            ts = int(x[0])
            if ts >= WINDOW_START_TS:
                rows.append((inst_id, ts, float(x[1]), float(x[2]), float(x[3]), float(x[4])))
        oldest = int(d[-1][0])
        if oldest < WINDOW_START_TS:
            break
        after = oldest - 1
        time.sleep(0.1)
    return rows


def main():
    needed = sorted(build_needed())
    print(f"needed contracts: {len(needed)}")
    conn = sqlite3.connect("data/options.db")
    conn.execute("DROP TABLE IF EXISTS options_meta")  # v1 为 5 列旧 schema，重建为 6 列
    conn.execute("CREATE TABLE IF NOT EXISTS options_meta ("
                 "inst_id TEXT PRIMARY KEY, uly TEXT, opt_type TEXT, strike REAL, exp_ts INTEGER, list_ts INTEGER)")
    conn.execute("CREATE TABLE IF NOT EXISTS options_candles ("
                 "inst_id TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, "
                 "PRIMARY KEY(inst_id, ts))")
    total = 0
    for inst_id, uly, opt_type, strike, exp_ts in needed:
        conn.execute("INSERT OR REPLACE INTO options_meta VALUES (?,?,?,?,?,0)",
                     (inst_id, uly, opt_type, strike, exp_ts))
        have = conn.execute("SELECT COUNT(*) FROM options_candles WHERE inst_id=?", (inst_id,)).fetchone()[0]
        if have >= 100:
            continue  # 已有足够覆盖
        rows = fetch_candles(inst_id)
        if rows:
            conn.executemany("INSERT OR IGNORE INTO options_candles VALUES (?,?,?,?,?,?)", rows)
        conn.commit()
        total += len(rows)
        if len(rows) > 0:
            print(f"{inst_id} rows={len(rows)} (had {have})")
    conn.close()
    print("total new rows:", total)


if __name__ == "__main__":
    main()
