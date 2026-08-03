#!/usr/bin/env python3
"""趋势线/通道线研究 — 斜线关键位的贴线效应 (1H, 插曲环境)

趋势线: 最近3个递增swing低点 (上升趋势线) / 递减高点 (下降趋势线) 拟合直线
通道线: 趋势线 + 平行线 (通过中间极值)
验证: 贴趋势线(0<=dist<0.5ATR) × 插曲 1:1 胜率 vs 基线
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


def swing_extrema(df, w=2):
    h = df["high"].values
    l = df["low"].values
    h_roll = pd.Series(h).rolling(2*w+1, center=True).max().values
    l_roll = pd.Series(l).rolling(2*w+1, center=True).min().values
    return (h >= h_roll), (l <= l_roll)


# 收集: 插曲样本 + 趋势线贴线状态
# 样本: (贴上升线, 贴下降线, 线上方距离, 线下方距离, long_side, out)
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
    mh, ml = swing_extrema(df1)
    idx_arr = np.arange(n)

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
            continue
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c[i]

        # ── 上升趋势线: 最近3个递增 swing low (近200根内) ──
        # 取近200根内的 swing low 位置
        lo_idx = np.where(ml[max(0, i-200):i])[0] + max(0, i-200)
        up_line_dist = None
        down_line_dist = None
        if len(lo_idx) >= 3:
            p1, p2, p3 = lo_idx[-3], lo_idx[-2], lo_idx[-1]
            v1, v2, v3 = l[p1], l[p2], l[p3]
            if v1 < v2 < v3:
                # 拟合直线 (用 p1,p3 两点, p2 验证)
                slope = (v3 - v1) / (p3 - p1)
                if 0 < slope < 0.5:  # 斜率合理 (不过陡)
                    line_at_i = v3 + slope * (i - p3)
                    up_line_dist = line_at_i - entry  # >0 = 价格在线上方
        # ── 下降趋势线: 最近3个递减 swing high ──
        hi_idx = np.where(mh[max(0, i-200):i])[0] + max(0, i-200)
        if len(hi_idx) >= 3:
            p1, p2, p3 = hi_idx[-3], hi_idx[-2], hi_idx[-1]
            v1, v2, v3 = h[p1], h[p2], h[p3]
            if v1 > v2 > v3:
                slope = (v3 - v1) / (p3 - p1)
                if -0.5 < slope < 0:
                    line_at_i = v3 + slope * (i - p3)
                    down_line_dist = entry - line_at_i  # >0 = 价格在线下方

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
        samples.append((up_line_dist, down_line_dist, long_side, 1 if hit == 1 else 0))
    print(f"{sym}: {len(samples)}", flush=True)

print(f"\n插曲样本: {len(samples)}")
base = np.mean([s[3] for s in samples]) * 100
print(f"基线胜率: {base:.1f}%")

def show(tag, sub):
    w = sum(1 for s in sub if s[3] == 1)
    l = sum(1 for s in sub if s[3] == 0)
    nn = w + l
    if nn < 100:
        print(f"{tag:<46} n={nn:>5} 样本不足")
    else:
        print(f"{tag:<46} n={nn:>6} 胜率={w/nn*100:>5.1f}% Δ={w/nn*100-base:>+5.1f}pp")

print("\n═══ 上升趋势线 (做多) ═══")
show("贴上升趋势线 (0<=dist<0.5ATR)", [s for s in samples if s[2] and s[0] is not None and 0 <= s[0] < 0.5])
show("接近趋势线 (0.5-1.0)", [s for s in samples if s[2] and s[0] is not None and 0.5 <= s[0] < 1.0])
show("远离趋势线 (>1.5)", [s for s in samples if s[2] and s[0] is not None and s[0] >= 1.5])
show("已跌破趋势线 (dist<0)", [s for s in samples if s[2] and s[0] is not None and s[0] < 0])

print("\n═══ 下降趋势线 (做空) ═══")
show("贴下降趋势线 (0<=dist<0.5ATR)", [s for s in samples if not s[2] and s[1] is not None and 0 <= s[1] < 0.5])
show("接近趋势线 (0.5-1.0)", [s for s in samples if not s[2] and s[1] is not None and 0.5 <= s[1] < 1.0])
show("远离趋势线 (>1.5)", [s for s in samples if not s[2] and s[1] is not None and s[1] >= 1.5])
show("已突破趋势线 (dist<0)", [s for s in samples if not s[2] and s[1] is not None and s[1] < 0])

print("\n═══ 与水平位对照 (参考) ═══")
from support_resistance import find_swing_levels
n_hl = 0

# 简化: 趋势线存在率
n_up = sum(1 for s in samples if s[2] and s[0] is not None)
n_dn = sum(1 for s in samples if not s[2] and s[1] is not None)
print(f"上升趋势线可检测率 (做多样本): {n_up}/{sum(1 for s in samples if s[2])*1.0:.0f} = {n_up/max(sum(1 for s in samples if s[2]),1)*100:.0f}%")
print(f"下降趋势线可检测率 (做空样本): {n_dn}/{sum(1 for s in samples if not s[2])*1.0:.0f} = {n_dn/max(sum(1 for s in samples if not s[2]),1)*100:.0f}%")
