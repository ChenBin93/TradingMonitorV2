#!/usr/bin/env python3
"""下载 OKX 历史 K 线（支持历史端点迭代）"""
import sys, time
from datetime import datetime
import requests

sys.path.insert(0, ".")
from history import HistoryDB


def fetch_okx_history(symbol: str, timeframe: str, target_bars: int = 1000) -> list[dict]:
    """迭代下载历史 K 线，使用 OKX history-candles 端点"""
    inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")
    bar = {"15m": "15m", "1h": "1H", "4h": "4H"}.get(timeframe, timeframe.upper())

    all_bars = []
    before_ts = None

    while len(all_bars) < target_bars:
        params = {"instId": inst_id, "bar": bar, "limit": "300"}
        if before_ts:
            params["before"] = str(before_ts)

        # 尝试历史端点
        url = "https://www.okx.com/api/v5/market/history-candles"
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("code") != "0" or not data.get("data"):
            # 回退到实时端点
            url = "https://www.okx.com/api/v5/market/candles"
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()

        if data.get("code") != "0" or not data.get("data"):
            break

        batch = data["data"]
        batch_bars = []
        for b in batch:
            ts = datetime.fromtimestamp(int(b[0]) / 1000)
            batch_bars.append({
                "timestamp": ts,
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "volume": float(b[5]),
            })

        if not all_bars:
            all_bars = batch_bars
        else:
            existing_ts = {b["timestamp"] for b in all_bars}
            new = [b for b in batch_bars if b["timestamp"] not in existing_ts]
            if not new:
                break
            all_bars = sorted(batch_bars + all_bars, key=lambda x: x["timestamp"])

        # 下一次请求更早的数据: before = 本次最老 K 线的时间戳
        oldest_ts = min(b["timestamp"] for b in batch_bars)
        before_ts = int(oldest_ts.timestamp()) - 1
        time.sleep(0.2)

        print(f"    {symbol} {timeframe}: {len(all_bars)} 根", end="\r")

    return sorted(all_bars, key=lambda x: x["timestamp"])[-target_bars:]


def boost_all(symbols: list[str], timeframes: list[str], target_bars: int = 600):
    db = HistoryDB("data/history.db")
    db.init()

    for sym in symbols:
        for tf in timeframes:
            try:
                bars = fetch_okx_history(sym, tf, target_bars)
                if bars:
                    db.save_candles(bars, sym, tf)
                    print(f"{sym} {tf}: {len(bars)} 根")
            except Exception as e:
                print(f"{sym} {tf}: ERROR {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=int, default=600)
    p.add_argument("--pool", type=int, default=10)
    p.add_argument("--symbols", nargs="*", default=None)
    args = p.parse_args()

    if args.symbols is None:
        from okx import OKXClient
        okx = OKXClient()
        args.symbols = okx.get_top_symbols(args.pool)

    print(f"下载: {len(args.symbols)} 币 × 3 TF, 目标 {args.bars} 根")
    boost_all(args.symbols, ["15m", "1h", "4h"], target_bars=args.bars)
    print("完成")
