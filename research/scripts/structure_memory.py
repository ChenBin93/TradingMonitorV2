#!/usr/bin/env python3
"""结构记忆维度 — 历史 swing 极值 (重大高低点) 的锚点效应

问题: 当前价格距"N根前形成的高点/低点"的距离, 对后续行为有没有影响?
     结构锚点是否超出记忆窗口 (方向~20根/波动~40-57天)?
验证:
  A. 锚点年龄效应: 距 5-100/100-250/250-500/500+ 根前极值的距离 × 插曲沿向1:1
  B. 距离效应: 贴锚(<0.5ATR) vs 近(0.5-1.5) vs 远(>1.5)
  C. 无条件对照
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

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

# 收集样本: (sym, i, long_side, out, 近高点年龄/距离, 近低点年龄/距离)
samples = []
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

    # 局部极值 (±2根) + 最近高点/低点索引 (向量化)
    h_roll = pd.Series(h).rolling(5, center=True).max().values
    l_roll = pd.Series(l).rolling(5, center=True).min().values
    is_high = h >= h_roll
    is_low = l <= l_roll
    idx_arr = np.arange(n)
    last_high = np.maximum.accumulate(np.where(is_high, idx_arr, -1))
    last_low = np.maximum.accumulate(np.where(is_low, idx_arr, -1))

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
            continue  # 插曲环境
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c[i]
        # 最近高点/低点 (年龄和距离)
        ih = last_high[i]
        il = last_low[i]
        age_hi = i - ih if ih >= 0 else 99999
        age_lo = i - il if il >= 0 else 99999
        dist_hi = (h[ih] - entry) / a if ih >= 0 else 99.0
        dist_lo = (entry - l[il]) / a if il >= 0 else 99.0
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
        samples.append((age_hi, dist_hi, age_lo, dist_lo, long_side, 1 if hit == 1 else 0))
    print(f"{sym}: {len(samples)} 样本", flush=True)

print(f"\n插曲样本总数: {len(samples)}")
base = np.mean([s[5] for s in samples]) * 100
print(f"基线胜率: {base:.1f}%")

# ═══ A. 锚点年龄 × 距离 (做多: 贴低点; 做空: 贴高点) ═══
print("\n═══ A. 结构锚点效应 (插曲环境, 沿日线方向) ═══")
AGE_BANDS = [(5, 100, "5-100根"), (100, 250, "100-250根"), (250, 500, "250-500根"), (500, 99999, "500+根")]

def show_band(title, cond):
    sub = [s for s in samples if cond(s)]
    w = sum(1 for s in sub if s[5] == 1)
    l = sum(1 for s in sub if s[5] == 0)
    nn = w + l
    if nn < 200:
        print(f"{title:<38} n={nn:>6} 样本不足")
    else:
        d = w / nn * 100 - base
        print(f"{title:<38} n={nn:>7} 胜率={w/nn*100:>5.1f}% Δ={d:>+5.1f}pp")

print("\n做多贴历史低点 (dist_lo < 0.5ATR, 按年龄):")
for lo_, hi_, name in AGE_BANDS:
    show_band(f"  低点年龄 {name}", lambda s, lo_=lo_, hi_=hi_: s[4] and s[2] >= lo_ and s[2] < hi_ and s[3] < 0.5)

print("\n做空贴历史高点 (dist_hi < 0.5ATR, 按年龄):")
for lo_, hi_, name in AGE_BANDS:
    show_band(f"  高点年龄 {name}", lambda s, lo_=lo_, hi_=hi_: not s[4] and s[0] >= lo_ and s[0] < hi_ and s[1] < 0.5)

print("\n═══ B. 距离效应 (贴锚 vs 近 vs 远) ═══")
for tag, cond in [
    ("做多 贴低点(<0.5ATR)", lambda s: s[4] and s[3] < 0.5),
    ("做多 低点近(0.5-1.5)", lambda s: s[4] and 0.5 <= s[3] < 1.5),
    ("做多 低点远(>1.5)", lambda s: s[4] and s[3] >= 1.5),
    ("做空 贴高点(<0.5ATR)", lambda s: not s[4] and s[1] < 0.5),
    ("做空 高点近(0.5-1.5)", lambda s: not s[4] and 0.5 <= s[1] < 1.5),
    ("做空 高点远(>1.5)", lambda s: not s[4] and s[1] >= 1.5),
]:
    show_band(tag, cond)

print("\n═══ C. 对照: 插曲基线分解 (做多/做空) ═══")
for tag, cond in [("做多 全部", lambda s: s[4]), ("做空 全部", lambda s: not s[4])]:
    show_band(tag, cond)
