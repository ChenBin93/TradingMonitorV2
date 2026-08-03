#!/usr/bin/env python3
"""关键位质量分验证 — 扩展窗口 (lookback 600) + 质量分单调性

1. 扩展窗口: 验证 400+ 根年龄效应, touch 拐点 (8-10 vs 11+)
2. 质量分 = 年龄分 + touch 分 + 跨周期重叠分 → 贴锚胜率单调性
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series
from support_resistance import find_swing_levels

data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20d = daily["close"].rolling(20).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]):
            continue
        d[ts.date()] = "bull" if row["close"] > ma20d[ts] else "bear"
    daily_states[sym] = d

LOOKBACK = 600  # 扩展窗口 (1H 600根 = 25天)

def swing_extrema(df, w=2):
    h = df["high"].values
    l = df["low"].values
    h_roll = pd.Series(h).rolling(2*w+1, center=True).max().values
    l_roll = pd.Series(l).rolling(2*w+1, center=True).min().values
    return (h >= h_roll), (l <= l_roll)

sw4 = {}
swd = {}
daily_data = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 300:
        continue
    sw4[sym] = swing_extrema(df4)
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(daily) >= 150:
        daily_data[sym] = daily
        swd[sym] = swing_extrema(daily)

def overlap_flags(sym, price, a1):
    flags = [False, False]  # 4h, daily
    df4 = data[sym].get("4h")
    if df4 is not None:
        h4 = df4["high"].values
        l4 = df4["low"].values
        mh4, ml4 = sw4[sym]
        idx4 = np.where(mh4 | ml4)[0]
        for j in idx4[-50:]:
            if abs(h4[j] - price) < 0.5 * a1 or abs(l4[j] - price) < 0.5 * a1:
                flags[0] = True
                break
    if sym in daily_data:
        daily = daily_data[sym]
        hd = daily["high"].values
        ld = daily["low"].values
        mhd, mld = swd[sym]
        idxd = np.where(mhd | mld)[0]
        for j in idxd[-20:]:
            if abs(hd[j] - price) < 0.5 * a1 or abs(ld[j] - price) < 0.5 * a1:
                flags[1] = True
                break
    return flags

samples = []  # (age, touch, ov4h, ovdaily, long_side, out)
for sym in syms:
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    if df1 is None or df4 is None or len(df1) < LOOKBACK + 400:
        continue
    c = df1["close"].values
    h = df1["high"].values
    l = df1["low"].values
    n = len(df1)
    atr = _atr_series(df1)
    idx1 = df1.index.values.astype("datetime64[ns]")
    idx4 = df4.index.values.astype("datetime64[ns]")
    c4 = df4["close"].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    ds = daily_states.get(sym, {})

    for i in range(LOOKBACK + 50, n - 25, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]):
            continue
        s4 = "bull" if c4[t4] > ma20_4[t4] else "bear"
        if s4 == sd:
            continue
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c[i]
        levels = find_swing_levels(df1.iloc[:i + 1].tail(LOOKBACK), LOOKBACK)
        hit_level = None
        for lv in levels:
            dist = entry - lv.price if lv.side == "support" else lv.price - entry
            if 0 <= dist <= 0.5 * a:
                hit_level = lv
                break
        if hit_level is None:
            continue
        age = i - (len(df1.iloc[:i + 1]) - LOOKBACK + hit_level.last_touch_idx) if hit_level.last_touch_idx >= 0 else 999
        long_side = sd == "bull"
        hit = 0
        for k in range(1, 25):
            if long_side:
                if h[i+k] >= entry + a:
                    hit = 1; break
                if l[i+k] <= entry - a:
                    hit = -1; break
            else:
                if l[i+k] <= entry - a:
                    hit = 1; break
                if h[i+k] >= entry + a:
                    hit = -1; break
        if hit == 0:
            continue
        ov = overlap_flags(sym, hit_level.price, a)
        samples.append((age, hit_level.touch_count, ov[0], ov[1], long_side, 1 if hit == 1 else 0))
    print(f"{sym}: {len(samples)}", flush=True)

print(f"\n贴锚样本 (lookback={LOOKBACK}): {len(samples)}")
base = np.mean([s[5] for s in samples]) * 100
print(f"基线胜率: {base:.1f}%")

def show(tag, sub):
    w = sum(1 for s in sub if s[5] == 1)
    l = sum(1 for s in sub if s[5] == 0)
    nn = w + l
    if nn < 100:
        print(f"{tag:<46} n={nn:>5} 样本不足")
    else:
        print(f"{tag:<46} n={nn:>6} 胜率={w/nn*100:>5.1f}% Δ={w/nn*100-base:>+5.1f}pp")

print("\n═══ 扩展窗口: 年龄分档 ═══")
for lo, hi, name in [(0, 50, "0-50根"), (50, 100, "50-100根"), (100, 200, "100-200根"),
                     (200, 400, "200-400根"), (400, 600, "400-600根"), (600, 99999, "600+根")]:
    show(f"  年龄 {name}", [s for s in samples if lo <= s[0] < hi])

print("\n═══ 扩展窗口: touch 分档 (拐点确认) ═══")
for lo, hi, name in [(2, 3, "touch=2"), (3, 5, "3-4"), (5, 8, "5-7"), (8, 11, "8-10"),
                     (11, 16, "11-15"), (16, 99999, "16+")]:
    show(f"  {name}", [s for s in samples if lo <= s[0] < hi])

print("\n═══ 关键位质量分 (年龄+touch+重叠) 单调性 ═══")
def quality_score(s):
    age, touch, ov4, ovd = s[0], s[1], s[2], s[3]
    sc = 0
    if age >= 200:
        sc += 2
    elif age >= 100:
        sc += 1
    if touch >= 5:
        sc += 2
    elif touch >= 3:
        sc += 1
    if ov4:
        sc += 1.5
    elif ovd:
        sc += 0.5
    return sc

qs = [quality_score(s) for s in samples]
print(f"{'质量分':>8} {'n':>7} {'胜率':>7} {'Δ':>6}")
prev = None
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5.5)]:
    sub = [s for s, q in zip(samples, qs) if lo <= q < hi]
    w = sum(1 for s in sub if s[5] == 1)
    l = sum(1 for s in sub if s[5] == 0)
    nn = w + l
    if nn < 100:
        print(f"  {lo}-{hi:<4} {nn:>7} 样本不足")
        continue
    wr = w / nn * 100
    print(f"  {lo}-{hi:<4} {nn:>7} {wr:>6.1f}% {wr-base:>+5.1f}pp")
