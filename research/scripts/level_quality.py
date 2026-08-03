#!/usr/bin/env python3
"""水平位质量定义 — 什么构成"好的关键位" (金标准: 贴锚胜率)

验证用户直觉:
  1. 多次打到并反转的位 = 优质 (touch 越多越强? 但次数太多会被打破?)
  2. 重要波段高低点 = 关键位
  3. 质量属性 (touch/年龄/幅度/聚类紧密) × 插曲贴锚胜率
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series
from support_resistance import find_swing_levels

data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20d = daily["close"].rolling(20).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]):
            continue
        d[ts.date()] = "bull" if row["close"] > ma20d[ts] else "bear"
    daily_states[sym] = d

# 收集: 插曲采样点 + 贴锚质量属性
# 样本: (touch, age, swing_amp, cluster_tight, is_support, out)
samples = []
n_near = 0
for sym in syms:
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    if df1 is None or df4 is None or len(df1) < 600:
        continue
    c = df1["close"].values
    h = df1["high"].values
    l = df1["low"].values
    n = len(df1)
    atr = _atr_series(df1)
    idx1 = df1.index.values.astype("datetime64[ns]")
    idx4 = df4.index.values.astype("datetime64[ns]")
    c4 = df4["close"].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    ds = daily_states.get(sym, {})

    for i in range(250, n - 25, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]):
            continue
        s4 = "bull" if c4[t4] > ma20_4[t4] else "bear"
        if s4 == sd:
            continue  # 插曲
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c[i]
        # 找贴住的水平位 (最近 300 根内的 swing 聚类)
        levels = find_swing_levels(df1.iloc[:i + 1].tail(300), 300)
        hit_level = None
        for lv in levels:
            dist = entry - lv.price if lv.side == "support" else lv.price - entry
            if 0 <= dist <= 0.5 * a:
                hit_level = lv
                break
        if hit_level is None:
            continue
        n_near += 1
        # 1:1 (沿日线方向)
        long_side = sd == "bull"
        hit = 0
        for k in range(1, 25):
            if long_side:
                if h[i+k] >= entry + a:
                    hit = 1; break
                if l[i+k] <= entry - a:
                    hit = -1; break
            else:
                if l[i+k] <= entry - a:
                    hit = 1; break
                if h[i+k] >= entry + a:
                    hit = -1; break
        if hit == 0:
            continue
        # 质量属性
        age = i - (len(df1.iloc[:i + 1]) - 300 + hit_level.last_touch_idx) if hit_level.last_touch_idx >= 0 else 999
        # 幅度: 该位附近 swing 的跨度 (用 level 与最近相反方向极值距离近似: 简化用 ATR)
        samples.append((hit_level.touch_count, age, hit_level.strength,
                        hit_level.side == "support", long_side, 1 if hit == 1 else 0))
    print(f"{sym}: 贴锚样本 {len(samples)}", flush=True)

print(f"\n贴锚样本总数: {len(samples)} (近位样本 {n_near})")
base = np.mean([s[5] for s in samples]) * 100
print(f"贴锚基线胜率: {base:.1f}%")

def show(tag, sub):
    w = sum(1 for s in sub if s[5] == 1)
    l = sum(1 for s in sub if s[5] == 0)
    nn = w + l
    if nn < 150:
        print(f"{tag:<44} n={nn:>5} 样本不足")
    else:
        print(f"{tag:<44} n={nn:>6} 胜率={w/nn*100:>5.1f}% Δ={w/nn*100-base:>+5.1f}pp")

print("\n═══ 直觉1: touch 次数 × 贴锚胜率 (次数太多会被打破?) ═══")
for lo, hi, name in [(1, 2, "touch=1"), (2, 3, "touch=2"), (3, 5, "touch=3-4"), (5, 99, "touch>=5")]:
    show(f"  {name}", [s for s in samples if lo <= s[0] < hi])

print("\n═══ 直觉2: 锚年龄 × 贴锚胜率 ═══")
for lo, hi, name in [(0, 5, "刚形成(0-5根)"), (5, 50, "5-50根"), (50, 200, "50-200根"), (200, 99999, "200+根")]:
    show(f"  年龄 {name}", [s for s in samples if lo <= s[1] < hi])

print("\n═══ 直觉3: touch × 年龄 交互 ═══")
for t_lo, t_hi, tn in [(1, 3, "touch<3"), (3, 99, "touch>=3")]:
    for a_lo, a_hi, an in [(0, 5, "年龄0-5"), (5, 200, "年龄5-200")]:
        show(f"  {tn} + {an}", [s for s in samples if t_lo <= s[0] < t_hi and a_lo <= s[1] < a_hi])

print("\n═══ 支持 vs 阻力 (方向正确性) ═══")
show("做多贴支撑", [s for s in samples if s[4] and s[3]])
show("做空贴阻力", [s for s in samples if not s[4] and not s[3]])
show("做多贴阻力(错位)", [s for s in samples if s[4] and not s[3]])
show("做空贴支撑(错位)", [s for s in samples if not s[4] and s[3]])

print("\n═══ touch 高端分档 (次数太多会被打破?) ═══")
for lo, hi, name in [(3, 5, "touch=3-4"), (5, 8, "touch=5-7"), (8, 11, "touch=8-10"), (11, 99, "touch>=11")]:
    show(f"  {name}", [s for s in samples if lo <= s[0] < hi])

print("\n═══ 年龄细分 (聚类位) ═══")
for lo, hi, name in [(5, 50, "5-50根"), (50, 100, "50-100根"), (100, 200, "100-200根"), (200, 400, "200-400根"), (400, 99999, "400+根")]:
    show(f"  年龄 {name}", [s for s in samples if lo <= s[1] < hi])
