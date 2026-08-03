#!/usr/bin/env python3
"""关键位质量 × 市场状态 — 质量效应在不同状态下的稳健性

市场状态维度: 波动(atr_ratio) × 插曲深度(4H偏离) × 1H阶段(末期?)
验证: 质量分的贴锚胜率效应是否跨状态稳健, 找最强组合
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series, _adx_series, analyze_market_state
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

LOOKBACK = 600

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
    flags = [False, False]
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

def quality_score(s):
    age, touch, ov4, ovd = s
    sc = 0
    if 200 <= age < 400:
        sc += 2
    elif age >= 100 or age >= 400:
        sc += 1
    if touch >= 5:
        sc += 1
    elif touch >= 3:
        sc += 0.5
    if ov4:
        sc += 1.5
    elif ovd:
        sc += 0.5
    return sc

# 收集: (质量属性, 波动ratio, 4H偏离, 1H阶段末期, long_side, out)
samples = []
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
    atr_ma = pd.Series(atr).rolling(90).mean().values
    adx1 = _adx_series(df1)
    idx1 = df1.index.values.astype("datetime64[ns]")
    idx4 = df4.index.values.astype("datetime64[ns]")
    c4 = df4["close"].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    atr4 = _atr_series(df4)
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
        # 市场状态特征
        vol_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
        dev4 = abs(c4[t4] - ma20_4[t4]) / atr4[t4] if atr4[t4] > 0 else 0
        ms1 = analyze_market_state(df1.iloc[max(0, i - 120):i + 1].reset_index(drop=True))
        is_late = ms1.get("stage") == "late"
        samples.append((age, hit_level.touch_count, ov[0], ov[1], vol_ratio, dev4, is_late, long_side, 1 if hit == 1 else 0))
    print(f"{sym}: {len(samples)}", flush=True)

print(f"\n贴锚样本: {len(samples)}")
base = np.mean([s[8] for s in samples]) * 100
print(f"基线胜率: {base:.1f}%")

def show(tag, sub):
    w = sum(1 for s in sub if s[8] == 1)
    l = sum(1 for s in sub if s[8] == 0)
    nn = w + l
    if nn < 100:
        print(f"{tag:<46} n={nn:>5} 样本不足")
    else:
        print(f"{tag:<46} n={nn:>6} 胜率={w/nn*100:>5.1f}% Δ={w/nn*100-base:>+5.1f}pp")

def qscore(s):
    return quality_score((s[0], s[1], s[2], s[3]))

print("\n═══ 1. 波动状态 × 质量分 ═══")
for v_lo, v_hi, vn in [(0, 0.7, "低波动"), (0.7, 1.3, "常态"), (1.3, 99, "高波动")]:
    sub = [s for s in samples if v_lo <= s[4] < v_hi]
    show(f"  {vn} 全部", sub)
    hi_q = [s for s in sub if qscore(s) >= 3]
    lo_q = [s for s in sub if qscore(s) < 1.5]
    show(f"  {vn} + 高质量(>=3)", hi_q)
    show(f"  {vn} + 低质量(<1.5)", lo_q)

print("\n═══ 2. 插曲深度 × 质量分 ═══")
for d_lo, d_hi, dn in [(0, 1.0, "浅回调(<1ATR)"), (1.0, 2.0, "中回调(1-2)"), (2.0, 99, "深回调(>2)")]:
    sub = [s for s in samples if d_lo <= s[5] < d_hi]
    show(f"  {dn} 全部", sub)
    hi_q = [s for s in sub if qscore(s) >= 3]
    show(f"  {dn} + 高质量(>=3)", hi_q)

print("\n═══ 3. 1H 阶段 (末期 vs 非末期) × 质量分 ═══")
for late_flag, ln in [(True, "1H末期"), (False, "1H非末期")]:
    sub = [s for s in samples if s[6] == late_flag]
    show(f"  {ln} 全部", sub)
    hi_q = [s for s in sub if qscore(s) >= 3]
    show(f"  {ln} + 高质量(>=3)", hi_q)

print("\n═══ 4. 最强组合扫描 (状态 × 质量) ═══")
best = []
for v_lo, v_hi, vn in [(0.7, 1.3, "常态"), (1.3, 99, "高波动")]:
    for d_lo, d_hi, dn in [(1.0, 2.0, "中回调"), (2.0, 99, "深回调")]:
        for late_flag, ln in [(False, "非末期")]:
            sub = [s for s in samples if v_lo <= s[4] < v_hi and d_lo <= s[5] < d_hi and s[6] == late_flag]
            hi_q = [s for s in sub if qscore(s) >= 3]
            w = sum(1 for s in hi_q if s[8] == 1)
            l = sum(1 for s in hi_q if s[8] == 0)
            nn = w + l
            if nn >= 100:
                best.append((w/nn*100, nn, f"{vn}+{dn}+{ln}+高质量"))
best.sort(key=lambda x: -x[0])
for wr, nn, tag in best[:8]:
    print(f"  {tag:<38} n={nn:>5} 胜率={wr:.1f}%")
