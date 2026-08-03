#!/usr/bin/env python3
"""结构记忆细化: 距离梯度 (接近未贴 vs 远离) + 年龄细分"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/workspace/project/TradingMonitor/v2')
from backtest_engine import load_all
from market_phase import _atr_series

data = load_all(timeframes=['1h', '4h'])
syms = list(data.keys())
daily_states = {}
for sym in syms:
    df4 = data[sym].get('4h')
    if df4 is None or len(df4) < 100: continue
    daily = df4.resample('1D').last()
    ma20d = daily['close'].rolling(20).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]): continue
        d[ts.date()] = 'bull' if row['close'] > ma20d[ts] else 'bear'
    daily_states[sym] = d

samples = []
for sym in syms:
    df1 = data[sym].get('1h')
    df4 = data[sym].get('4h')
    if df1 is None or df4 is None or len(df1) < 600: continue
    c = df1['close'].values; h = df1['high'].values; l = df1['low'].values
    n = len(df1)
    atr = _atr_series(df1)
    idx1 = df1.index.values.astype('datetime64[ns]')
    idx4 = df4.index.values.astype('datetime64[ns]')
    c4 = df4['close'].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    ds = daily_states.get(sym, {})
    h_roll = pd.Series(h).rolling(5, center=True).max().values
    l_roll = pd.Series(l).rolling(5, center=True).min().values
    is_high = h >= h_roll
    is_low = l <= l_roll
    idx_arr = np.arange(n)
    last_high = np.maximum.accumulate(np.where(is_high, idx_arr, -1))
    last_low = np.maximum.accumulate(np.where(is_low, idx_arr, -1))
    for i in range(250, n - 25, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ('bull', 'bear'): continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240,'m'), side='right')) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]): continue
        s4 = 'bull' if c4[t4] > ma20_4[t4] else 'bear'
        if s4 == sd: continue
        a = atr[i]
        if a <= 0 or np.isnan(a): continue
        entry = c[i]
        ih = last_high[i]; il = last_low[i]
        age_hi = i - ih if ih >= 0 else 99999
        age_lo = i - il if il >= 0 else 99999
        dist_hi = (h[ih] - entry) / a if ih >= 0 else 99.0
        dist_lo = (entry - l[il]) / a if il >= 0 else 99.0
        long_side = sd == 'bull'
        hit = 0
        for k in range(1, 25):
            if long_side:
                if h[i+k] >= entry + a: hit = 1; break
                if l[i+k] <= entry - a: hit = -1; break
            else:
                if l[i+k] <= entry - a: hit = 1; break
                if h[i+k] >= entry + a: hit = -1; break
        if hit == 0: continue
        samples.append((age_hi, dist_hi, age_lo, dist_lo, long_side, 1 if hit == 1 else 0))
print(f'样本: {len(samples)}')
base_long = np.mean([s[5] for s in samples if s[4]]) * 100
base_short = np.mean([s[5] for s in samples if not s[4]]) * 100
print(f'基线: 做多 {base_long:.1f}% / 做空 {base_short:.1f}%')

def show(tag, sub, base):
    w = sum(1 for s in sub if s[5] == 1); l = sum(1 for s in sub if s[5] == 0)
    nn = w + l
    if nn < 200:
        print(f'{tag:<40} n={nn:>6} 样本不足')
    else:
        print(f'{tag:<40} n={nn:>7} 胜率={w/nn*100:>5.1f}% Δ={w/nn*100-base:>+5.1f}pp')

print('\n═══ 做多 (插曲) × 距最近低点距离梯度 ═══')
for lo_, hi_, name in [(0, 0.5, '贴(<0.5)'), (0.5, 1.0, '0.5-1.0'), (1.0, 1.5, '1.0-1.5'),
                       (1.5, 3.0, '1.5-3.0'), (3.0, 6.0, '3.0-6.0'), (6.0, 99, '>6.0')]:
    show(f'做多 低点距 {name}ATR', [s for s in samples if s[4] and lo_ <= s[3] < hi_], base_long)

print('\n═══ 做空 (插曲) × 距最近高点距离梯度 ═══')
for lo_, hi_, name in [(0, 0.5, '贴(<0.5)'), (0.5, 1.0, '0.5-1.0'), (1.0, 1.5, '1.0-1.5'),
                       (1.5, 3.0, '1.5-3.0'), (3.0, 6.0, '3.0-6.0'), (6.0, 99, '>6.0')]:
    show(f'做空 高点距 {name}ATR', [s for s in samples if not s[4] and lo_ <= s[1] < hi_], base_short)

print('\n═══ 做多 距低点 1.0-1.5 × 低点年龄 ═══')
for lo_, hi_, name in [(5, 50, '5-50根'), (50, 100, '50-100根'), (100, 250, '100-250根'), (250, 99999, '250+根')]:
    show(f'做多 距低点1.0-1.5 年龄{name}', [s for s in samples if s[4] and 1.0 <= s[3] < 1.5 and lo_ <= s[2] < hi_], base_long)
