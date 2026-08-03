#!/usr/bin/env python3
"""三层特征对比: 快照(1根) vs 聚合序列(统计量) vs 全序列(窗口内每根K指标)
真实插曲环境, 逻辑回归 AUC 对比"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
sys.path.insert(0, '/root/workspace/project/TradingMonitor/v2')
from backtest_engine import load_all
from market_phase import _atr_series, _adx_series

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

W = 20
X_snap, X_agg, X_full, y = [], [], [], []
for sym in syms:
    df1 = data[sym].get('1h')
    df4 = data[sym].get('4h')
    if df1 is None or df4 is None or len(df1) < 400: continue
    c1 = df1['close'].values; hi = df1['high'].values; lo = df1['low'].values
    n1 = len(df1)
    atr = _atr_series(df1)
    adx = _adx_series(df1)
    ma20_1 = pd.Series(c1).rolling(20).mean().values
    idx1 = df1.index.values.astype('datetime64[ns]')
    idx4 = df4.index.values.astype('datetime64[ns]')
    c4 = df4['close'].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    ds = daily_states.get(sym, {})
    ret1 = np.diff(c1) / c1[:-1] * 100

    for i in range(200, n1 - 25, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ('bull', 'bear'): continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240,'m'), side='right')) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]): continue
        s4 = 'bull' if c4[t4] > ma20_4[t4] else 'bear'
        if s4 == sd: continue
        a = atr[i]
        if a <= 0 or np.isnan(a) or i < W + 2: continue
        dev = (c1[i] - ma20_1[i]) / a
        # 快照: 当前K
        snap = [dev, adx[i]/40, ret1[i-1]/(a/c1[i]*100)]
        # 聚合序列
        agg = [(ma20_1[i]-ma20_1[i-W])/a, adx[i]-adx[i-W],
               np.mean(ret1[i-W:i])/(a/c1[i]*100), np.mean(np.abs(ret1[i-W:i]))/(a/c1[i]*100)]
        # 全序列: 窗口每根收益+偏离
        full = []
        for k in range(i-W, i):
            full.append(ret1[k]/(a/c1[i]*100))
            full.append((c1[k]-ma20_1[k])/a if not np.isnan(ma20_1[k]) else 0)
        long_side = sd == 'bull'
        entry = c1[i]
        hit = 0
        for k in range(1, 25):
            if long_side:
                if hi[i+k] >= entry + a: hit = 1; break
                if lo[i+k] <= entry - a: hit = -1; break
            else:
                if lo[i+k] <= entry - a: hit = 1; break
                if hi[i+k] >= entry + a: hit = -1; break
        if hit == 0: continue
        X_snap.append(snap); X_agg.append(agg); X_full.append(full)
        y.append(1 if hit == 1 else 0)

X_snap = np.array(X_snap); X_agg = np.array(X_agg); X_full = np.array(X_full)
y = np.array(y)
print(f'插曲样本: {len(y)}, 基线胜率: {np.mean(y)*100:.1f}%')

for name, X in [('快照(1根, 3维)', X_snap), ('聚合序列(4维)', X_agg), ('全序列(40维)', X_full)]:
    clf = LogisticRegression(max_iter=1000, C=0.1)
    aucs = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
    print(f'  {name:<16} AUC: {aucs.mean():.4f} ± {aucs.std():.4f}')
