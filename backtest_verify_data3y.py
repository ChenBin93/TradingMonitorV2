#!/usr/bin/env python3
"""3年数据验证: 每标的 bar 数/时间覆盖 + 日线多空分布"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all

data = load_all()
print(f"标的数: {len(data)}")

total = {}
for sym, tfs in data.items():
    for tf, df in tfs.items():
        total.setdefault(tf, 0)
        total[tf] += len(df)
print(f"总 bar 数: {total}")

print()
print(f"{'标的':<20} {'1H bars':>9} {'1H覆盖(天)':>10} {'4H bars':>8} {'4H覆盖(天)':>10}")
for sym in sorted(data):
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    s1 = f"{len(df1):>9}" if df1 is not None else "   --"
    d1 = f"{(df1.index[-1]-df1.index[0]).days:>9}d" if df1 is not None and len(df1) > 1 else "   --"
    s4 = f"{len(df4):>8}" if df4 is not None else "  --"
    d4 = f"{(df4.index[-1]-df4.index[0]).days:>9}d" if df4 is not None and len(df4) > 1 else "  --"
    print(f"{sym:<20} {s1} {d1} {s4} {d4}")

# 日线多空分布 (4H resample)
print()
print("=== 日线状态分布 (3年, 全部标的) ===")
days = {}
for sym, tfs in data.items():
    df4 = tfs.get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    for ts, row in daily.iterrows():
        if pd.isna(ma20[ts]):
            continue
        d = days.setdefault(ts.date(), [0, 0, 0])  # bull, bear, neut
        if row["close"] > ma20[ts]:
            d[0] += 1
        else:
            d[1] += 1
if days:
    arr = np.array(list(days.values()))
    print(f"bull天数(均值): {np.mean(arr[:,0]):.1f}  bear: {np.mean(arr[:,1]):.1f}")
    total_d = len(days)
    bull_pct = (arr[:,0] > arr[:,1]).sum() / total_d * 100
    bear_pct = (arr[:,1] > arr[:,0]).sum() / total_d * 100
    print(f"多空一致天数: bull主导 {bull_pct:.0f}% / bear主导 {bear_pct:.0f}% (共 {total_d} 天)")

# 按年分布
print()
print("=== 按年: BTC 日线 bull/bear 天数 ===")
btc = data.get("BTC/USDT:USDT", {}).get("4h")
if btc is not None and len(btc) > 100:
    daily = btc.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    by_year = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20[ts]):
            continue
        y = ts.year
        d = by_year.setdefault(y, [0, 0])
        if row["close"] > ma20[ts]:
            d[0] += 1
        else:
            d[1] += 1
    for y in sorted(by_year):
        b, s = by_year[y]
        print(f"{y}: bull {b}天 ({b/(b+s)*100:.0f}%) / bear {s}天 ({s/(b+s)*100:.0f}%)")
