#!/usr/bin/env python3
"""演示: 纯随机游走下跑同样的插曲策略, 胜率应该是多少?

如果真实市场的 edge (插曲 54-61%, 高波动插曲 62.7%) 来自市场结构,
那么纯随机游走模拟下这些策略应该 ≈50%
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/root/workspace/project/TradingMonitor/v2')

rng = np.random.default_rng(42)
N_SYM = 20
N_H = 26280  # 3年 1H

# 模拟 1: 纯几何随机游走 (收益 i.i.d. 正态)
def sim_rw():
    ret = rng.normal(0.0001, 0.002, (N_H, N_SYM))  # 微漂移
    return ret

# 模拟 2: 随机游走 + 波动聚集 (简化 GARCH: 波动自回归)
def sim_garch():
    ret = np.zeros((N_H, N_SYM))
    vol = np.full(N_SYM, 0.002)
    for i in range(N_H):
        ret[i] = rng.normal(0.0001, 1) * vol
        vol = 0.00002 + 0.94 * vol + 0.05 * np.abs(ret[i])  # 收敛的 GARCH
    return ret

def run_strategy(ret_mat, name):
    """日线顺势+4H逆插曲 → 1:1 (用模拟数据自身的 MA 状态)"""
    n = len(ret_mat)
    # 价格路径
    price = 100 * np.exp(np.cumsum(ret_mat, axis=0))
    hi = price * (1 + np.abs(rng.normal(0, 0.0008, price.shape)))
    lo = price * (1 - np.abs(rng.normal(0, 0.0008, price.shape)))
    wins = losses = 0
    for sym in range(N_SYM):
        c = price[:, sym]
        # 1H ATR (简化: 滚动 std*2)
        atr = pd.Series(c).pct_change().rolling(14).std().values * c
        ma20_1 = pd.Series(c).rolling(20).mean().values
        ma20_4 = pd.Series(c).rolling(80).mean().values  # 4H MA20 近似 (20*4)
        for i in range(150, n - 25, 6):
            a = atr[i]
            if a <= 0 or np.isnan(a):
                continue
            # 日线: 80根MA; 4H状态: 20根MA — 价格在80MA之上且20MA之下 = 回调插曲
            daily_bull = c[i] > ma20_4[i]
            s4 = c[i] > ma20_1[i]
            if daily_bull == s4:
                continue  # 不是插曲
            long_side = daily_bull
            entry = c[i]
            hit = 0
            for k in range(1, 25):
                if long_side:
                    if hi[i+k, sym] >= entry + a:
                        hit = 1; break
                    if lo[i+k, sym] <= entry - a:
                        hit = -1; break
                else:
                    if lo[i+k, sym] <= entry - a:
                        hit = 1; break
                    if hi[i+k, sym] >= entry + a:
                        hit = -1; break
            if hit == 1:
                wins += 1
            elif hit == -1:
                losses += 1
    nn = wins + losses
    print(f"{name}: 插曲策略 1:1 胜率 = {wins/nn*100:.1f}% (n={nn})", flush=True)
    return wins, losses

print("═══ 模拟对照: 纯随机游走下插曲策略 ═══")
ret1 = sim_rw()
run_strategy(ret1, "纯随机游走")
ret2 = sim_garch()
run_strategy(ret2, "随机游走+GARCH波动聚集")

print()
print("真实市场 (3年 20标的): 插曲 54.7-57.3%, 高波动插曲 62.7%")
