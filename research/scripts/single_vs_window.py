#!/usr/bin/env python3
"""单根K判定 vs 窗口组合判定 — 对"当前行情状态"的识别准确率

A. 合成数据 (ground truth 状态): 3分类准确率对比
   - 单根: 当前K收益分档 (噪声大)
   - 窗口: 10根动量 / MA20位置 / ADX+MA组合 (我们的系统)
B. 真实数据 (1H, 事后标签): 同样对比
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from market_phase import analyze_market_state, _atr_series, _adx_series

# ═══════════════ A. 合成数据 ═══════════════
def make_regime_series(segments, seed=0):
    rng = np.random.default_rng(seed)
    o = []; h = []; l = []; c = []; v = []
    price = 100.0
    for n, kind, vol in segments:
        for _ in range(n):
            if kind == "trend_up":
                drift = 0.35 * vol
            elif kind == "trend_down":
                drift = -0.35 * vol
            else:
                drift = rng.normal(0, 0.02 * vol)
            o.append(price)
            price += drift + rng.normal(0, 0.15 * vol)
            c.append(price)
            h.append(price + vol * (0.2 + abs(rng.normal(0, 0.1))))
            l.append(price - vol * (0.2 + abs(rng.normal(0, 0.1))))
            v.append(100)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})

segments = [(300, "trend_up", 1.0), (300, "range", 1.0),
            (300, "trend_down", 1.0), (300, "range", 1.0),
            (300, "trend_up", 1.0)]
df = make_regime_series(segments)
TRUE = []
for n, kind, vol in segments:
    TRUE.extend([kind] * n)
TRUE = np.array(TRUE)
c = df["close"].values
ret = np.diff(c) / c[:-1] * 100

# 真实状态 → 3分类 label (up/down/range)
def true_label(i):
    # i 是 ret 索引 → df 索引 i+1
    t = TRUE[i + 1]
    return "up" if t == "trend_up" else "down" if t == "trend_down" else "range"

# 判定方法
def judge_single(r, sig):
    """单根: 当前K收益分档"""
    if r > 0.5 * sig:
        return "up"
    if r < -0.5 * sig:
        return "down"
    return "range"

def judge_momentum(r_window, sig):
    """窗口: 10根累计动量"""
    m = np.sum(r_window[-10:])
    if m > 0.5 * sig:
        return "up"
    if m < -0.5 * sig:
        return "down"
    return "range"

def judge_ma(c_window):
    """窗口: 价格 vs MA20 位置"""
    ma = np.mean(c_window[-20:])
    if c_window[-1] > ma * 1.002:
        return "up"
    if c_window[-1] < ma * 0.998:
        return "down"
    return "range"

methods = {
    "单根K收益": [],
    "10根动量": [],
    "MA20位置": [],
    "ADX+MA(系统)": [],
}
sig_full = np.std(ret)

for i in range(120, len(ret) - 1):
    tl = true_label(i)
    # 单根
    m1 = judge_single(ret[i], sig_full)
    # 动量
    m2 = judge_momentum(ret[i-9:i+1], sig_full * np.sqrt(10))
    # MA20
    m3 = judge_ma(c[i-19:i+2])
    # 系统判定
    ms = analyze_market_state(df.iloc[max(0, i-119):i+2].reset_index(drop=True))
    st = ms.get("state", "")
    if st == "trend_up":
        m4 = "up"
    elif st == "trend_down":
        m4 = "down"
    else:
        m4 = "range"
    methods["单根K收益"].append(m1 == tl)
    methods["10根动量"].append(m2 == tl)
    methods["MA20位置"].append(m3 == tl)
    methods["ADX+MA(系统)"].append(m4 == tl)

print("═══ A. 合成数据: 当前状态识别准确率 (3分类) ═══")
for name, acc in methods.items():
    print(f"  {name:<16} {np.mean(acc)*100:.1f}% (n={len(acc)})")

# ═══════════════ B. 真实数据 (1H BTC, 事后标签) ═══════════════
print("\n═══ B. 真实数据 (1H BTC): 事后标签验证 ═══")
from backtest_engine import load_all
data = load_all(timeframes=["1h"])
df1 = data["BTC/USDT:USDT"]["1h"]
c1 = df1["close"].values
ret1 = np.diff(c1) / c1[:-1] * 100
n = len(ret1)
atr = _atr_series(df1)

# 事后标签: 未来30根净移动 (方向), 幅度决定趋势/震荡
def true_state_future(i):
    """i 是 ret 索引 → df 索引 i+1"""
    if i + 30 >= n or atr[i+1] <= 0:
        return None
    move = (c1[i+31] - c1[i+1]) / atr[i+1]
    if abs(move) > 7:
        return "up" if move > 0 else "down"
    if abs(move) < 3:
        return "range"
    return None  # 过渡期跳过

res = {name: [0, 0] for name in methods}
for i in range(120, n - 31, 3):
    tl = true_state_future(i)
    if tl is None:
        continue
    sig = np.std(ret1[i-500:i])
    m1 = judge_single(ret1[i], sig)
    m2 = judge_momentum(ret1[i-9:i+1], sig * np.sqrt(10))
    m3 = judge_ma(c1[i-19:i+2])
    ms = analyze_market_state(df1.iloc[max(0, i-119):i+2].reset_index(drop=True))
    st = ms.get("state", "")
    m4 = "up" if st == "trend_up" else "down" if st == "trend_down" else "range"
    for name, m in [("单根K收益", m1), ("10根动量", m2), ("MA20位置", m3), ("ADX+MA(系统)", m4)]:
        if m == tl:
            res[name][0] += 1
        res[name][1] += 1

print(f"{'方法':<16} {'准确率':>8} {'n':>8}")
for name, (w, t) in res.items():
    if t > 0:
        print(f"  {name:<16} {w/t*100:>7.1f}% {t:>8}")
