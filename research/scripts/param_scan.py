#!/usr/bin/env python3
"""指标参数扫描: MA周期 × 动量窗口 — 插曲环境判定质量 (AUC)
验证 MA20/动量10 是否在平台区/最优区"""
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

# 预收集: 每标的的插曲样本 (i, c, atr, ret1, 方向)
raw = []  # (sym, i, long_side, out)
per_sym = {}
for sym in syms:
    df1 = data[sym].get('1h')
    df4 = data[sym].get('4h')
    if df1 is None or df4 is None or len(df1) < 400: continue
    c1 = df1['close'].values; hi = df1['high'].values; lo = df1['low'].values
    n1 = len(df1)
    atr = _atr_series(df1)
    idx1 = df1.index.values.astype('datetime64[ns]')
    idx4 = df4.index.values.astype('datetime64[ns]')
    c4 = df4['close'].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    ds = daily_states.get(sym, {})
    ret1 = np.diff(c1) / c1[:-1] * 100
    rows = []
    for i in range(200, n1 - 25, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ('bull', 'bear'): continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240,'m'), side='right')) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]): continue
        s4 = 'bull' if c4[t4] > ma20_4[t4] else 'bear'
        if s4 == sd: continue
        a = atr[i]
        if a <= 0 or np.isnan(a): continue
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
        rows.append((i, long_side, 1 if hit == 1 else 0))
    per_sym[sym] = {'df': df1, 'atr': atr, 'ret': ret1, 'rows': rows}
    print(f'{sym}: {len(rows)}', flush=True)

# 参数扫描
MA_GRID = [10, 15, 20, 25, 30]
MOM_GRID = [5, 10, 15, 20]
print('\n═══ 指标参数扫描 (插曲环境, AUC) ═══')
print(f"{'MA':>4} {'动量':>5} {'AUC':>7} {'基线':>6}")
results = []
for ma_p in MA_GRID:
    for mom_p in MOM_GRID:
        X = []
        y = []
        for sym, pd_ in per_sym.items():
            df1 = pd_['df']
            c1 = df1['close'].values
            atr = pd_['atr']
            ret1 = pd_['ret']
            ma = pd.Series(c1).rolling(ma_p).mean().values
            for i, long_side, out in pd_['rows']:
                a = atr[i]
                if a <= 0: continue
                dev = (c1[i] - ma[i]) / a if not np.isnan(ma[i]) else 0
                mom = np.sum(ret1[i-mom_p:i]) / (a / c1[i] * 100) if i >= mom_p else 0
                X.append([dev * (1 if long_side else -1), mom * (1 if long_side else -1)])
                y.append(out)
        X = np.array(X)
        y = np.array(y)
        clf = LogisticRegression(max_iter=1000)
        aucs = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
        a_ = aucs.mean()
        results.append((a_, ma_p, mom_p))
        print(f"{ma_p:>4} {mom_p:>5} {a_:>7.4f} {np.mean(y)*100:>5.1f}%")

results.sort(key=lambda r: -r[0])
print(f"\n最优: MA={results[0][1]} 动量={results[0][2]} AUC={results[0][0]:.4f}")
print(f"当前配置: MA=20 动量=10 AUC={[r[0] for r in results if r[1]==20 and r[2]==10][0]:.4f}")
