#!/usr/bin/env python3
"""波动状态 × 策略选择验证 (3年, BTC单标的先行)

波动状态定义: atr_ratio = ATR_1h[i] / ATR_1h 过去90根均值
  <0.7 低波动 | 0.7-1.3 常态 | >1.3 高波动

验证:
  A. 无条件: 波动分档 × 后续1:1 (双向)
  B. 插曲环境 (日线顺势+4H逆): 波动分档 × 沿日线方向1:1
  C. BB反转事件 (日线空/中性+1H BB上轨破轨): 波动分档 × 1:1 (TP0.5/SL0.5)
  D. (对照) 4H 波动分档 × 插曲1:1
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

SYM = None  # 全部标的
VOL_EDGES = [(0, 0.7, "低波动"), (0.7, 1.3, "常态"), (1.3, 99, "高波动")]

data = load_all(timeframes=["1h", "4h"])
df1 = data[SYM]["1h"]
df4 = data[SYM]["4h"]

# ── 日线状态 (4H 合成) ──
daily = df4.resample("1D").last()
ma20_d = daily["close"].rolling(20).mean()
daily_states = {}
for ts, row in daily.iterrows():
    if pd.isna(ma20_d[ts]):
        continue
    daily_states[ts.date()] = "bull" if row["close"] > ma20_d[ts] else "bear"

# ── 预计算 1H 指标 ──
c1 = df1["close"].values
h1 = df1["high"].values
l1 = df1["low"].values
o1 = df1["open"].values
n1 = len(df1)
atr1 = _atr_series(df1)
atr_ma = pd.Series(atr1).rolling(90).mean().values
atr_ratio = atr1 / np.where(atr_ma > 0, atr_ma, atr1)

# 4H 状态 (每 1H 采样点)
idx1 = df1.index.values.astype("datetime64[ns]")
idx4 = df4.index.values.astype("datetime64[ns]")
c4 = df4["close"].values
ma20_4 = pd.Series(c4).rolling(20).mean().values
atr4 = _atr_series(df4)

def t4_state(i):
    t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
    if t4 < 20 or np.isnan(ma20_4[t4]):
        return "N"
    return "bull" if c4[t4] > ma20_4[t4] else "bear"

def vol_bucket(i):
    r = atr_ratio[i]
    for lo, hi, name in VOL_EDGES:
        if lo <= r < hi:
            return name
    return "常态"

def bucket_stats(buckets, direction, tp, sl, w, long_side_fn=None):
    """统计: 每波动分档的 1:1 胜率 (direction: 'long'/'short'/'either')"""
    out = {name: [0, 0] for _, _, name in VOL_EDGES}
    for i in range(150, n1 - w - 1, 3):
        b = buckets[i]
        entry = c1[i]
        a = atr1[i]
        if a <= 0 or np.isnan(a):
            continue
        if long_side_fn is not None:
            long_side = long_side_fn(i)
            if long_side is None:
                continue
        hit = 0
        for k in range(1, w + 1):
            if direction == "long" or (long_side_fn is not None and long_side):
                if h1[i+k] >= entry + tp * a:
                    hit = 1; break
                if l1[i+k] <= entry - sl * a:
                    hit = -1; break
            else:
                if l1[i+k] <= entry - sl * a:
                    hit = 1; break
                if h1[i+k] >= entry + tp * a:
                    hit = -1; break
        if hit == 1:
            out[b][0] += 1
        elif hit == -1:
            out[b][1] += 1
    return out

def show_table(title, stats, tp, sl, w):
    print(f"\n{title} (TP={tp}/SL={sl}/W={w})")
    print(f"{'波动状态':<8} {'n':>8} {'胜率':>8}")
    for _, _, name in VOL_EDGES:
        w_ = stats[name][0]
        l_ = stats[name][1]
        n = w_ + l_
        if n < 100:
            print(f"{name:<8} {n:>8} 样本不足")
        else:
            print(f"{name:<8} {n:>8} {w_/n*100:>7.1f}%")

# ═══ A. 无条件: 波动分档 × 双向 1:1 (TP1/SL1/W24) ═══
print("═══ A. 无条件: 波动状态 × 双向 1:1 ═══")
buckets_a = [vol_bucket(i) for i in range(n1)]
stats_a = bucket_stats(buckets_a, "either", 1.0, 1.0, 24)
show_table("无条件(双向)", stats_a, 1.0, 1.0, 24)

# ═══ B. 插曲环境: 波动分档 × 沿日线方向 1:1 ═══
print("\n═══ B. 插曲环境 (日线顺势+4H逆) ═══")
def ep_long_fn(i):
    ts = pd.Timestamp(idx1[i])
    sd = daily_states.get(ts.date())
    if sd not in ("bull", "bear"):
        return None
    s4 = t4_state(i)
    if s4 == "N" or s4 == sd:
        return None
    return sd == "bull"

stats_b = bucket_stats(buckets_a, None, 1.0, 1.0, 24, long_side_fn=ep_long_fn)
show_table("插曲 (沿日线方向)", stats_b, 1.0, 1.0, 24)

# ═══ C. BB反转事件: 波动分档 × 1:1 (TP0.5/SL0.5/W48) ═══
print("\n═══ C. BB反转事件 (日线空/中性 + 1H BB上轨破轨) ═══")
ma20_1 = pd.Series(c1).rolling(20).mean().values
sd_1 = pd.Series(c1).rolling(20).std().values
stats_c = {name: [0, 0] for _, _, name in VOL_EDGES}
n_bb = 0
for i in range(150, n1 - 49, 3):
    if np.isnan(ma20_1[i]) or np.isnan(sd_1[i]) or sd_1[i] <= 0:
        continue
    ts = pd.Timestamp(idx1[i])
    sd = daily_states.get(ts.date())
    if sd == "bull":
        continue  # BB反转环境: 日线空/中性
    up = ma20_1[i] + 2.5 * sd_1[i]
    if c1[i] <= up:
        continue
    b = vol_bucket(i)
    entry = c1[i]
    a = atr1[i]
    if a <= 0:
        continue
    hit = 0
    for k in range(1, 49):
        if l1[i+k] <= entry - 0.5 * a:
            hit = 1; break
        if h1[i+k] >= entry + 0.5 * a:
            hit = -1; break
    n_bb += 1
    if hit == 1:
        stats_c[b][0] += 1
    elif hit == -1:
        stats_c[b][1] += 1
show_table(f"BB反转 ({n_bb} 事件)", stats_c, 0.5, 0.5, 48)

# ═══ D. 4H 波动分档 × 插曲 1:1 (对照) ═══
print("\n═══ D. 4H 波动分档 × 插曲 1:1 ═══")
atr4_ma = pd.Series(atr4).rolling(60).mean().values
stats_d = {name: [0, 0] for _, _, name in VOL_EDGES}
for i in range(150, n1 - 25, 6):
    t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
    if t4 < 60:
        continue
    r4 = atr4[t4] / atr4_ma[t4] if atr4_ma[t4] > 0 else 1.0
    b = "低波动" if r4 < 0.7 else "高波动" if r4 > 1.3 else "常态"
    ts = pd.Timestamp(idx1[i])
    sd = daily_states.get(ts.date())
    if sd not in ("bull", "bear"):
        continue
    s4 = t4_state(i)
    if s4 == "N" or s4 == sd:
        continue
    long_side = sd == "bull"
    entry = c1[i]
    a = atr1[i]
    if a <= 0:
        continue
    hit = 0
    for k in range(1, 25):
        if long_side:
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
        stats_d[b][0] += 1
    elif hit == -1:
        stats_d[b][1] += 1
show_table("4H波动×插曲", stats_d, 1.0, 1.0, 24)
