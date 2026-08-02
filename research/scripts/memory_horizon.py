#!/usr/bin/env python3
"""记忆窗口研究 — 当前市场状态受向前多大窗口影响 (3维度 × 3周期)

A. ACF 衰减曲线: 方向记忆 (收益) + 波动记忆 (|收益|/ATR)
   → 显著自相关的最长滞后 = 记忆长度
B. 波动增量预测: 窗口 W 的波动特征 → 未来24根波动 (1H)
   → 预测能力 vs W 曲线, 平台期 = 波动信息上限
C. (结构记忆单独一轮)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())

# ═══════════════ A. ACF 衰减曲线 ═══════════════
def acf_series(x, max_lag=400):
    x = x - x.mean()
    var = np.sum(x ** 2)
    if var <= 0:
        return np.zeros(max_lag)
    out = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        out[k-1] = np.sum(x[k:] * x[:-k]) / var
    return out

def memory_length(ac, n, threshold=1.96, hold=10):
    """记忆长度: 第一个"连续 hold 根不显著"后的断裂点 (避免长滞后偶然噪声)"""
    band = threshold / np.sqrt(n)
    sig = np.abs(ac) > band
    streak = 0
    for k in range(len(ac)):
        if not sig[k]:
            streak += 1
            if streak >= hold:
                return k + 1 - hold + 1  # 断裂点
        else:
            streak = 0
    return len(ac)

print("═══ A. ACF 记忆长度 (显著自相关最大滞后, 95%置信带) ═══", flush=True)
print(f"{'周期':<6} {'方向(收益)':>12} {'波动(|收益|)':>12} {'ATR':>8} {'样本':>8}")
for tf_name, df in [("1h", data["BTC/USDT:USDT"]["1h"]),
                    ("4h", data["BTC/USDT:USDT"]["4h"])]:
    c = df["close"].values
    ret = np.diff(c) / c[:-1] * 100
    n = len(ret)
    ac_dir = acf_series(ret, 400)
    ac_vol = acf_series(np.abs(ret), 400)
    atr = _atr_series(df)
    ac_atr = acf_series(atr[100:], 400)
    print(f"{tf_name:<6} {memory_length(ac_dir, n):>12} {memory_length(ac_vol, n):>12} "
          f"{memory_length(ac_atr, n-100):>8} {n:>8}", flush=True)

# 日线
df4 = data["BTC/USDT:USDT"]["4h"]
df1d = df4.resample("1D").agg(
    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
).dropna()
c = df1d["close"].values
ret = np.diff(c) / c[:-1] * 100
n = len(ret)
ac_dir = acf_series(ret, 200)
ac_vol = acf_series(np.abs(ret), 200)
atr = _atr_series(df1d)
ac_atr = acf_series(atr[60:], 200)
print(f"{'日线':<6} {memory_length(ac_dir, n):>12} {memory_length(ac_vol, n):>12} "
      f"{memory_length(ac_atr, n-60):>8} {n:>8}", flush=True)

# ACF 衰减形状 (1H, 波动)
print("\n1H 波动 ACF 衰减 (滞后 1/5/10/25/50/100/200/400):")
ac_vol_1h = acf_series(np.abs(np.diff(data["BTC/USDT:USDT"]["1h"]["close"].values)
                               / data["BTC/USDT:USDT"]["1h"]["close"].values[:-1] * 100), 400)
for lag in [1, 5, 10, 25, 50, 100, 200, 400]:
    print(f"  lag={lag:<4} ac={ac_vol_1h[lag-1]:+.4f}")

# ═══════════════ B. 波动增量预测 (1H, 信息上限) — 向量化 ═══════════════
print("\n═══ B. 波动增量预测 (1H BTC): 窗口W → 未来24根波动 ═══", flush=True)
c = data["BTC/USDT:USDT"]["1h"]["close"].values
ret = np.diff(c) / c[:-1] * 100
n = len(ret)
vol_series = np.abs(ret)

# 目标: tgt[i] = mean(vol[i+1:i+25]) = rolling(24) 在 vol 上的值平移
r24 = pd.Series(vol_series).rolling(24).mean().values

print(f"{'窗口W':>8} {'预测相关':>10} {'解释度R²':>10} {'n':>8}")
for W in [10, 20, 50, 100, 200, 500, 1000, 2000]:
    # 特征: feat[i] = mean(vol[i-W:i]) = rolling(W) 平移
    rW = pd.Series(vol_series).rolling(W).mean().values
    # i 从 W 到 n-26 → feat 索引 = i-1 (roll 在 i-1 处 = mean(vol[i-W:i]))
    feat = rW[W-1:n-25]
    tgt = r24[W+23:n-1]
    m = min(len(feat), len(tgt))
    feat = feat[:m]
    tgt = tgt[:m]
    r = np.corrcoef(feat, tgt)[0, 1]
    print(f"{W:>8} {r:>10.4f} {r*r:>10.4f} {m:>8}", flush=True)
