#!/usr/bin/env python3
"""多周期关键位重叠验证 — 1H贴锚位 × 是否也是4H/日线关键位

用户假设: 1H 200根关键位 ≈ 4H 50根关键位 (周期重叠) —
长年龄1H位有效, 可能是因为它们构成上一级周期的关键位

验证: 1H贴锚样本按"跨周期重叠"分组 → 贴锚胜率
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

# 4H/日线 swing 极值预计算 (每标的)
def swing_extrema(df, w=2):
    h = df["high"].values
    l = df["low"].values
    n = len(df)
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

def has_overlap(sym, price, a1, lookback_4h=50, lookback_d=20, tol=0.5):
    """价格附近 (tol×1H ATR) 在 4H/日线图上是否有 swing 极值 (跨周期重叠)"""
    res = {"4h": False, "daily": False}
    df4 = data[sym].get("4h")
    if df4 is not None:
        h4 = df4["high"].values
        l4 = df4["low"].values
        n4 = len(df4)
        mh4, ml4 = sw4[sym]
        idx4 = np.where(mh4 | ml4)[0]
        # 最近 lookback_4h 根内的极值
        for j in idx4[-lookback_4h:]:
            if abs(h4[j] - price) < tol * a1 or abs(l4[j] - price) < tol * a1:
                res["4h"] = True
                break
    if sym in daily_data:
        daily = daily_data[sym]
        hd = daily["high"].values
        ld = daily["low"].values
        mhd, mld = swd[sym]
        idxd = np.where(mhd | mld)[0]
        for j in idxd[-lookback_d:]:
            if abs(hd[j] - price) < tol * a1 or abs(ld[j] - price) < tol * a1:
                res["daily"] = True
                break
    return res

# 收集: 1H插曲贴锚样本 + 跨周期重叠标记
samples = []  # (age, touch, overlap_4h, overlap_daily, long_side, out)
for sym in syms:
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    if df1 is None or df4 is None or len(df1) < 600:
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

    for i in range(250, n - 25, 4):
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
        levels = find_swing_levels(df1.iloc[:i + 1].tail(300), 300)
        hit_level = None
        for lv in levels:
            dist = entry - lv.price if lv.side == "support" else lv.price - entry
            if 0 <= dist <= 0.5 * a:
                hit_level = lv
                break
        if hit_level is None:
            continue
        age = i - (len(df1.iloc[:i + 1]) - 300 + hit_level.last_touch_idx) if hit_level.last_touch_idx >= 0 else 999
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
        ov = has_overlap(sym, hit_level.price, a)
        samples.append((age, hit_level.touch_count, ov["4h"], ov["daily"], long_side, 1 if hit == 1 else 0))
    print(f"{sym}: {len(samples)}", flush=True)

print(f"\n贴锚样本: {len(samples)}")
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

print("\n═══ 跨周期重叠 × 贴锚胜率 ═══")
show("无跨周期重叠 (仅1H位)", [s for s in samples if not s[2] and not s[3]])
show("重叠4H位", [s for s in samples if s[2]])
show("重叠日线位", [s for s in samples if s[3]])
show("同时重叠4H+日线", [s for s in samples if s[2] and s[3]])

print("\n═══ 年龄 × 跨周期重叠 交互 ═══")
for a_lo, a_hi, an in [(0, 100, "年龄<100"), (100, 99999, "年龄>=100")]:
    for ov_tag, ov_cond in [("无重叠", lambda s: not s[2] and not s[3]), ("重叠4H", lambda s: s[2])]:
        show(f"{an} + {ov_tag}", [s for s in samples if a_lo <= s[0] < a_hi and ov_cond(s)])

print("\n═══ touch × 跨周期重叠 ═══")
for t_lo, t_hi, tn in [(2, 3, "touch=2"), (3, 99999, "touch>=3")]:
    for ov_tag, ov_cond in [("无重叠", lambda s: not s[2] and not s[3]), ("重叠4H", lambda s: s[2])]:
        show(f"{tn} + {ov_tag}", [s for s in samples if t_lo <= s[0] < t_hi and ov_cond(s)])
