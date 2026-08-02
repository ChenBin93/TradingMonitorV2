#!/usr/bin/env python3
"""二维 HMM 验证 — 观测 [收益, 滚动波动率] (BTC, 3周期)

与单维 HMM (只学波动状态) 对比: 二维能否同时学出方向+波动状态
- 观测: ret = pct_change; vol = rolling(20).std() (平滑波动率, 独立信息)
- 标准化训练 (hmmlearn 对尺度敏感), 反标准化解释
- 3 态, 20 次随机初始化取最优 log-likelihood
- 验证: 状态参数/停留/分布/预测价值 (滤波, 无未来函数)
"""
import sys
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

SYM = "BTC/USDT:USDT"
N_STATES = 3
N_INIT = 20
HORIZON = {"1h": 24, "4h": 12, "1d": 10}


def train_hmm_2d(obs, n_states=3, n_init=20):
    best = None
    best_ll = -np.inf
    for seed in range(n_init):
        try:
            model = GaussianHMM(
                n_components=n_states, covariance_type="full",
                n_iter=200, random_state=seed, tol=1e-4,
            )
            model.fit(obs)
            ll = model.score(obs)
            if ll > best_ll:
                best_ll = ll
                best = model
        except Exception:
            continue
    return best, best_ll


def forward_filter_probs(model, obs):
    logprob, log_alpha = model.score_samples(obs)
    alpha = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
    return alpha / alpha.sum(axis=1, keepdims=True)


def main():
    data = load_all(timeframes=["1h", "4h"])
    df1 = data[SYM]["1h"]
    df4 = data[SYM]["4h"]
    df1d = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    tfs = {"1h": df1, "4h": df4, "1d": df1d}

    for tf_name, df in tfs.items():
        print(f"\n{'='*62}\n═══ {tf_name} 二维HMM (n={len(df)}) ═══\n{'='*62}", flush=True)
        c = df["close"].values
        ret = np.diff(c) / c[:-1] * 100
        vol = pd.Series(ret).rolling(20).std().values
        # 丢弃前 20 根 (vol 未定义)
        ret = ret[19:]
        vol = vol[19:]
        obs_raw = np.column_stack([ret, vol])
        obs = (obs_raw - obs_raw.mean(axis=0)) / obs_raw.std(axis=0)

        model, ll = train_hmm_2d(obs, N_STATES, N_INIT)
        if model is None:
            print("训练失败")
            continue

        # 反标准化参数
        mu = model.means_ * obs_raw.std(axis=0) + obs_raw.mean(axis=0)
        cov = model.covars_ * np.outer(obs_raw.std(axis=0), obs_raw.std(axis=0))
        trans = model.transmat_

        # 状态排序: 先按 μ_ret 排序看方向, 再标注波动
        order = np.argsort(mu[:, 0])
        labels = ["态0", "态1", "态2"]
        print(f"\n[最优 log-likelihood: {ll:.1f}]  (观测: 收益% + 波动std%)")
        print("\n状态参数 (按 μ_收益 排序):")
        print(f"{'状态':<6} {'μ收益':>8} {'σ收益':>8} {'μ波动':>8} {'σ波动':>8} {'停留(根)':>9}")
        for rank in range(N_STATES):
            s = order[rank]
            stay = 1.0 / (1.0 - trans[s, s]) if trans[s, s] < 1 else np.inf
            print(f"{labels[rank]:<6} {mu[s,0]:>8.4f} {np.sqrt(cov[s,0,0]):>8.4f} "
                  f"{mu[s,1]:>8.4f} {np.sqrt(cov[s,1,1]):>8.4f} {stay:>9.1f}")

        print("\n转移矩阵 (行=当前, 列=下一):")
        for i in range(N_STATES):
            print("  " + "  ".join(f"{trans[order[i], order[j]]:>7.4f}" for j in range(N_STATES)))

        # 滤波状态
        probs = forward_filter_probs(model, obs)
        filt = probs.argmax(axis=1)
        filt_rank = np.array([np.where(order == s)[0][0] for s in filt])

        print("\n状态分布 (滤波):")
        for rank in range(N_STATES):
            cnt = (filt_rank == rank).sum()
            print(f"  {labels[rank]}: {cnt} 根 ({cnt/len(filt_rank)*100:.1f}%)")

        # 预测价值: 沿状态方向 (μ_ret>0 做多 / <0 做空), 震荡态跳过
        h = HORIZON[tf_name]
        atr = _atr_series(df)
        hi = df["high"].values
        lo = df["low"].values
        n = len(df)
        print(f"\n预测价值: 滤波状态 → 未来 {h} 根 1:1 (±1ATR, 沿状态方向)")
        print(f"{'状态':<6} {'n':>7} {'沿向胜率':>9}")
        for rank in range(N_STATES):
            s = order[rank]
            if abs(mu[s, 0]) < 0.005:
                continue  # μ_ret 接近 0 = 方向不明, 跳过
            idxs = np.where(filt_rank == rank)[0]
            wins = loses = 0
            direction = 1 if mu[s, 0] > 0 else -1
            for i in idxs:
                j = i + 19  # 滤波索引 → 原序列索引 (ret 前移19)
                if j + h >= n or atr[j] <= 0:
                    continue
                entry = c[j]
                a = atr[j]
                hit = 0
                for k in range(1, h + 1):
                    if direction == 1:
                        if hi[j+k] >= entry + a:
                            hit = 1; break
                        if lo[j+k] <= entry - a:
                            hit = -1; break
                    else:
                        if lo[j+k] <= entry - a:
                            hit = 1; break
                        if hi[j+k] >= entry + a:
                            hit = -1; break
                if hit == 1:
                    wins += 1
                elif hit == -1:
                    loses += 1
            nn = wins + loses
            print(f"{labels[rank]:<6} {nn:>7} {wins/nn*100 if nn else 0:>8.1f}%")


if __name__ == "__main__":
    main()
