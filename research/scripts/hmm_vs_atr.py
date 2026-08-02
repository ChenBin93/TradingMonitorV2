#!/usr/bin/env python3
"""HMM 波动状态 vs atr_ratio 分档 对照 (BTC 1H)

问题: HMM 是否提供 atr_ratio 之外的波动信息?
- HMM 状态按 σ 排序 = 低/中/高波动
- atr_ratio = ATR_1h / 过去90根均值, 分档 <0.7/0.7-1.3/>1.3
- 混淆矩阵 + 一致率
"""
import sys
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

SYM = "BTC/USDT:USDT"
N_INIT = 20

data = load_all(timeframes=["1h"])
df = data[SYM]["1h"]
ret = df["close"].pct_change().dropna().values * 100
obs = ret.reshape(-1, 1)

# 训练单维 HMM (按 σ 解释为波动状态)
best = None
best_ll = -np.inf
for seed in range(N_INIT):
    try:
        m = GaussianHMM(n_components=3, covariance_type="full", n_iter=200,
                        random_state=seed, tol=1e-4)
        m.fit(obs)
        ll = m.score(obs)
        if ll > best_ll:
            best_ll = ll
            best = m
    except Exception:
        continue

# 滤波状态
logprob, log_alpha = best.score_samples(obs)
alpha = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
probs = alpha / alpha.sum(axis=1, keepdims=True)
filt = probs.argmax(axis=1)

# HMM 状态按 σ 排序 → 低/中/高波动
sigmas = np.sqrt(best.covars_.reshape(3, 1, 1)[:, 0, 0])
sigma_order = np.argsort(sigmas)  # 0=最低波动
hmm_tier = np.array([np.where(sigma_order == s)[0][0] for s in filt])
hmm_names = ["HMM低", "HMM中", "HMM高"]

# atr_ratio 分档 (对齐索引: ret 从 idx1 开始, atr 从 idx0)
atr = _atr_series(df)
atr_ma = pd.Series(atr).rolling(90).mean().values
n = len(df)

def atr_tier(i):
    if atr_ma[i] <= 0 or atr[i] <= 0:
        return None
    r = atr[i] / atr_ma[i]
    if r < 0.7:
        return 0
    if r < 1.3:
        return 1
    return 2

# 混淆矩阵 (ret 索引 j → df 索引 j+1)
conf = np.zeros((3, 3), dtype=int)
for j in range(90, len(ret)):
    a = atr_tier(j + 1)
    if a is None:
        continue
    conf[hmm_tier[j], a] += 1

print("═══ HMM波动状态 (行) × atr_ratio分档 (列) 混淆矩阵 ═══")
print(f"{'':>8} {'atr低':>8} {'atr常态':>8} {'atr高':>8} {'合计':>8}")
for i in range(3):
    print(f"{hmm_names[i]:>8} {conf[i,0]:>8} {conf[i,1]:>8} {conf[i,2]:>8} {conf[i].sum():>8}")

total = conf.sum()
diag = np.trace(conf)
print(f"\n总样本: {total}, 完全一致率: {diag/total*100:.1f}%")

# 加权一致率 (按 atr 分档边缘归一)
print("\n按列(条件一致率): 给定atr分档, HMM判对的占比")
for j, name in enumerate(["atr低", "atr常态", "atr高"]):
    col = conf[:, j]
    s = col.sum()
    if s > 0:
        print(f"  {name}: {col[j]}/{s} = {col[j]/s*100:.1f}%")

# HMM 各状态的 atr_ratio 分布特征
print("\nHMM 各状态的 atr_ratio 统计:")
for i in range(3):
    idxs = np.where(hmm_tier == i)[0] + 1
    ratios = np.array([atr_tier(x) for x in idxs])
    valid = ratios[ratios != np.array(None)]
    r_vals = []
    for x in idxs:
        if atr_ma[x] > 0 and atr[x] > 0:
            r_vals.append(atr[x] / atr_ma[x])
    if r_vals:
        rv = np.array(r_vals)
        print(f"  {hmm_names[i]}: atr_ratio 均值 {rv.mean():.2f} 中位 {np.median(rv):.2f} "
              f"p25 {np.percentile(rv,25):.2f} p75 {np.percentile(rv,75):.2f}")

# HMM 状态是否含 atr_ratio 之外的信息: 预测价值对比
print("\n预测价值对比 (未来24根 1:1, TP1/SL1):")
print(f"{'':>10} {'n':>8} {'胜率':>8}")
c = df["close"].values
h1 = df["high"].values
l1 = df["low"].values

def eval_1v1(idxs, direction):
    wins = loses = 0
    for i in idxs:
        if i + 24 >= n or atr[i] <= 0:
            continue
        entry = c[i]
        a = atr[i]
        hit = 0
        for k in range(1, 25):
            if direction == 1:
                if h1[i+k] >= entry + a:
                    hit = 1; break
                if l1[i+k] <= entry - a:
                    hit = -1; break
            else:
                if l1[i+k] <= entry - a:
                    hit = 1; break
                if h1[i+k] >= entry + a:
                    hit = -1; break
        if hit == 1:
            wins += 1
        elif hit == -1:
            loses += 1
    return wins, loses

# 无条件沿日线方向? 简化: 双向1:1 (either direction counts as win if either hits first)
print("(双向 1:1: 先到 ±1ATR 判定, 无方向偏好)")
for i in range(3):
    idxs = np.where(hmm_tier == i)[0] + 1
    wins = loses = 0
    for i2 in idxs:
        if i2 + 24 >= n or atr[i2] <= 0:
            continue
        entry = c[i2]
        a = atr[i2]
        hit = 0
        for k in range(1, 25):
            if h1[i2+k] >= entry + a:
                hit = 1; break
            if l1[i2+k] <= entry - a:
                hit = -1; break
        if hit == 1:
            wins += 1
        elif hit == -1:
            loses += 1
    nn = wins + loses
    print(f"HMM{hmm_names[i]}: {nn:>8} {wins/nn*100 if nn else 0:>7.1f}%")
for tier, name in [(0, "atr低"), (1, "atr常态"), (2, "atr高")]:
    idxs = [i for i in range(150, n - 25) if atr_tier(i) == tier]
    wins = loses = 0
    for i2 in idxs:
        entry = c[i2]
        a = atr[i2]
        hit = 0
        for k in range(1, 25):
            if h1[i2+k] >= entry + a:
                hit = 1; break
            if l1[i2+k] <= entry - a:
                hit = -1; break
        if hit == 1:
            wins += 1
        elif hit == -1:
            loses += 1
    nn = wins + loses
    print(f"{name}: {nn:>8} {wins/nn*100 if nn else 0:>7.1f}%")
