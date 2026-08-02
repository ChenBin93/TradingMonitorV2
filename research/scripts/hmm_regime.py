#!/usr/bin/env python3
"""HMM 市场状态研究 — BTC 单标的, 3 周期 (日线/4H/1H), 3 态高斯 HMM

步骤:
  1. 每周期训练 3 态高斯 HMM (20 次随机初始化取最优 log-likelihood)
  2. 输出: 状态 μ/σ (经济含义) + 转移矩阵 + 平均停留时间
  3. 状态序列: Viterbi (事后) + 前向滤波 (无未来函数)
  4. 验证:
     a. 状态可解释性
     b. 停留时间 (对照 1H 状态活 6h 的已知结论)
     c. 预测价值: 滤波状态 → 未来 24 根 1:1 胜率 (1H/4H) / 10 根 (日线)
     d. 与 market_phase 规则化判定对照 (一致率)
"""
import sys
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import analyze_market_state, _atr_series

SYM = "BTC/USDT:USDT"
HORIZON = {"1h": 24, "4h": 12, "1d": 10}
N_INIT = 20
N_STATES = 3


def train_hmm(obs, n_states=3, n_init=20):
    """多次随机初始化训练, 返回最优模型 (按 log-likelihood)"""
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


def order_states(model, obs):
    """按状态均值 μ 排序: 0=低均值(趋势下), 1=中(震荡), 2=高均值(趋势上)"""
    means = model.means_.flatten()
    order = np.argsort(means)
    return order


def forward_filter_probs(model, obs):
    """前向滤波概率 (无未来函数): P(s_t | o_1..o_t)"""
    # hmmlearn score_samples 返回每个时刻各状态的 log 前向概率 (归一化)
    logprob, log_alpha = model.score_samples(obs)
    alpha = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
    probs = alpha / alpha.sum(axis=1, keepdims=True)
    return probs


def main():
    data = load_all(timeframes=["1h", "4h"])
    df1 = data[SYM]["1h"]
    df4 = data[SYM]["4h"]
    df1d = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    tfs = {
        "1h": df1,
        "4h": df4,
        "1d": df1d,
    }

    for tf_name, df in tfs.items():
        print(f"\n{'='*60}\n═══ {tf_name} (n={len(df)}) ═══\n{'='*60}", flush=True)
        ret = df["close"].pct_change().dropna().values * 100  # 百分比收益
        obs = ret.reshape(-1, 1)

        model, ll = train_hmm(obs, N_STATES, N_INIT)
        if model is None:
            print("训练失败")
            continue
        order = order_states(model, obs)
        means = model.means_.flatten()
        covs = np.sqrt(model.covars_.reshape(N_STATES, 1, 1)[:, 0, 0]).flatten() \
            if model.covariance_type == "full" else np.sqrt(model.covars_).flatten()
        trans = model.transmat_

        # 对齐状态
        labels = ["趋势下", "震荡", "趋势上"]
        print(f"\n[最优 log-likelihood: {ll:.1f}]")
        print("\n状态参数 (按均值排序):")
        print(f"{'状态':<6} {'μ(收益%/根)':>12} {'σ(波动%/根)':>12} {'平均停留(根)':>12}")
        for rank in range(N_STATES):
            s = order[rank]
            stay = 1.0 / (1.0 - trans[s, s]) if trans[s, s] < 1 else np.inf
            print(f"{labels[rank]:<6} {means[s]:>12.4f} {covs[s]:>12.4f} {stay:>12.1f}")

        print("\n转移矩阵 (行=当前, 列=下一):")
        print("      " + "  ".join(f"{labels[j]:>6}" for j in range(N_STATES)))
        for i in range(N_STATES):
            print(f"{labels[i]:>6}" + "  ".join(f"{trans[order[i], order[j]]:>8.4f}" for j in range(N_STATES)))

        # 状态序列 (滤波, 无未来函数)
        probs = forward_filter_probs(model, obs)
        filt_state = probs.argmax(axis=1)
        # 映射到排序后的标签
        filt_label = np.array([np.where(order == s)[0][0] for s in filt_state])

        # ── 状态分布 ──
        print("\n状态分布 (滤波):")
        for rank in range(N_STATES):
            cnt = (filt_label == rank).sum()
            print(f"  {labels[rank]}: {cnt} 根 ({cnt/len(filt_label)*100:.1f}%)")

        # ── 预测价值: 滤波状态 → 未来 N 根 1:1 胜率 ──
        h = HORIZON[tf_name]
        atr = _atr_series(df)
        c = df["close"].values
        hi = df["high"].values
        lo = df["low"].values
        n = len(df)
        print(f"\n预测价值: 滤波状态 → 未来 {h} 根 1:1 (±1ATR, 沿状态方向)")
        print(f"{'状态':<6} {'n':>7} {'沿向胜率':>9} {'反向往胜率':>10}")
        for rank in range(N_STATES):
            # 状态 rank: 0=趋势下(做空), 1=震荡(双向? 取绝对), 2=趋势上(做多)
            idxs = np.where(filt_label == rank)[0]
            wins = loses = 0
            for i in idxs:
                if i + h >= n or atr[i] <= 0:
                    continue
                entry = c[i]
                a = atr[i]
                if rank == 2:
                    direction = 1
                elif rank == 0:
                    direction = -1
                else:
                    direction = 0
                # 沿向做单 1:1
                hit = 0
                for k in range(1, h + 1):
                    if direction == 1:
                        if hi[i+k] >= entry + a:
                            hit = 1
                            break
                        if lo[i+k] <= entry - a:
                            hit = -1
                            break
                    elif direction == -1:
                        if lo[i+k] <= entry - a:
                            hit = 1
                            break
                        if hi[i+k] >= entry + a:
                            hit = -1
                            break
                    else:
                        # 震荡状态: 测双向 1:1 (任一方向先到)
                        hit = 0
                        break
                if hit == 1:
                    wins += 1
                elif hit == -1:
                    loses += 1
            nn = wins + loses
            wr = wins / nn * 100 if nn else 0
            print(f"{labels[rank]:<6} {nn:>7} {wr:>8.1f}%")

        # ── 与 market_phase 对照 (一致率) ──
        print("\n与 market_phase 规则化判定对照 (滤波状态 vs 判定状态):")
        agree = total = 0
        for i in range(150, len(df) - 1, 6):
            ms = analyze_market_state(df.iloc[max(0, i - 100):i + 1].reset_index(drop=True))
            st = ms.get("state", "")
            # 映射: 判定状态 → HMM 标签
            if st == "trend_up":
                pred_rank = 2
            elif st == "trend_down":
                pred_rank = 0
            else:
                pred_rank = 1  # range/transition → 震荡
            if filt_label[i] == pred_rank:
                agree += 1
            total += 1
        print(f"  一致率: {agree}/{total} = {agree/total*100:.1f}% (trend_up→趋势上, trend_down→趋势下, 其余→震荡)")


if __name__ == "__main__":
    main()
