#!/usr/bin/env python3
"""市场状态 × 关键位强度 × 方向 胜率矩阵 (贴而未破, 0<=dist<=0.5ATR)

市场状态: 日线(多/中性/空) × 4H(多/中性/空) = 9 组合
强度: 质量分 低(0-1.5) / 中(1.5-3) / 高(3+)
方向: 贴支撑→做多 / 贴阻力→做空 (独立 1:1)
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

# 日线 (4H 合成) 3 档状态
daily_data = {}
daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 300:
        continue
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(daily) < 200:
        continue
    daily_data[sym] = daily
    ma20d = daily["close"].rolling(20).mean()
    atr_d = (daily["high"] - daily["low"]).rolling(14).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]) or pd.isna(atr_d[ts]) or atr_d[ts] <= 0:
            continue
        dev = (row["close"] - ma20d[ts]) / atr_d[ts]
        if dev > 0.5:
            d[ts.date()] = "bull"
        elif dev < -0.5:
            d[ts.date()] = "bear"
        else:
            d[ts.date()] = "neut"
    daily_states[sym] = d

LOOKBACK = 600

def swing_extrema(df, w=2):
    h = df["high"].values
    l = df["low"].values
    h_roll = pd.Series(h).rolling(2*w+1, center=True).max().values
    l_roll = pd.Series(l).rolling(2*w+1, center=True).min().values
    return (h >= h_roll), (l <= l_roll)

sw4 = {}
swd = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None:
        continue
    sw4[sym] = swing_extrema(df4)
    if sym in daily_data:
        swd[sym] = swing_extrema(daily_data[sym])

def overlap_flags(sym, price, a1):
    flags = [False, False]
    df4 = data[sym].get("4h")
    if df4 is not None:
        h4 = df4["high"].values
        l4 = df4["low"].values
        mh4, ml4 = sw4[sym]
        idx4 = np.where(mh4 | ml4)[0]
        for j in idx4[-50:]:
            if abs(h4[j] - price) < 0.5 * a1 or abs(l4[j] - price) < 0.5 * a1:
                flags[0] = True
                break
    if sym in daily_data:
        daily = daily_data[sym]
        hd = daily["high"].values
        ld = daily["low"].values
        mhd, mld = swd[sym]
        idxd = np.where(mhd | mld)[0]
        for j in idxd[-20:]:
            if abs(hd[j] - price) < 0.5 * a1 or abs(ld[j] - price) < 0.5 * a1:
                flags[1] = True
                break
    return flags

def qband(age, touch, ov4, ovd):
    sc = 0
    if 200 <= age < 400:
        sc += 2
    elif age >= 100 or age >= 400:
        sc += 1
    if touch >= 5:
        sc += 1
    elif touch >= 3:
        sc += 0.5
    if ov4:
        sc += 1.5
    elif ovd:
        sc += 0.5
    if sc >= 3:
        return "高(3+)"
    if sc >= 1.5:
        return "中(1.5-3)"
    return "低(0-1.5)"

# 收集: (日线档, 4H档, 方向档, 质量档, out)  方向档: 'support'/'resist'
STATE9 = ["日多4H多", "日多4H中", "日多4H空",
          "日中4H多", "日中4H中", "日中4H空",
          "日空4H多", "日空4H中", "日空4H空"]
Q3 = ["低(0-1.5)", "中(1.5-3)", "高(3+)"]
res = {d: {s: {q: [0, 0] for q in Q3} for s in STATE9} for d in ["support", "resist"]}
n_scan = 0

for sym in syms:
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    if df1 is None or df4 is None or len(df1) < LOOKBACK + 400 or sym not in daily_states:
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
    atr4 = _atr_series(df4)
    ds = daily_states.get(sym, {})

    for i in range(LOOKBACK + 50, n - 25, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ("bull", "neut", "bear"):
            continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]):
            continue
        # 4H 档
        dev4 = (c4[t4] - ma20_4[t4]) / atr4[t4] if atr4[t4] > 0 else 0
        if dev4 > 0.5:
            s4 = "bull"
        elif dev4 < -0.5:
            s4 = "bear"
        else:
            s4 = "neut"
        st9 = {"bull": "多", "neut": "中", "bear": "空"}[sd] + "4H" + {"bull": "多", "neut": "中", "bear": "空"}[s4]
        st9 = "日" + st9
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c[i]
        n_scan += 1
        levels = find_swing_levels(df1.iloc[:i + 1].tail(LOOKBACK), LOOKBACK)
        # 贴支撑 (做多) / 贴阻力 (做空)
        for lv in levels:
            if lv.side == "support":
                dist = entry - lv.price
            else:
                dist = lv.price - entry
            if not (0 <= dist <= 0.5 * a):
                continue
            age = i - (len(df1.iloc[:i + 1]) - LOOKBACK + lv.last_touch_idx) if lv.last_touch_idx >= 0 else 999
            ov = overlap_flags(sym, lv.price, a)
            q = qband(age, lv.touch_count, ov[0], ov[1])
            long_side = (lv.side == "support")
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
                break  # 未触发 → 放弃该样本 (不能跳到下个贴位, 会改变方向判定)
            dkey = "support" if long_side else "resist"
            res[dkey][st9][q][0] += 1 if hit == 1 else 0
            res[dkey][st9][q][1] += 1
            break  # 每个样本只统计最近贴的位
    print(f"{sym}: done (扫描{n_scan})", flush=True)

print(f"\n扫描采样: {n_scan}")
for dkey, dname in [("support", "═══ 贴支撑做多 (0<=dist<=0.5ATR) ═══"),
                    ("resist", "═══ 贴阻力做空 (0<=dist<=0.5ATR) ═══")]:
    print(f"\n{dname}")
    hdr = "  状态      "
    for q in Q3:
        hdr += f"{'胜率(n)':>14}"
    print(hdr)
    for st in STATE9:
        line = f"  {st:<8}"
        for q in Q3:
            w, l = res[dkey][st][q]
            nn = w + l
            if nn < 50:
                line += f"{'--('+str(nn)+')':>14}"
            else:
                line += f"{w/nn*100:>7.1f}%({nn:>5})"
        print(line)
    # 按质量档汇总
    print("  [汇总]")
    for q in Q3:
        tot_w = sum(res[dkey][st][q][0] for st in STATE9)
        tot_l = sum(res[dkey][st][q][1] for st in STATE9)
        nn = tot_w + tot_l
        if nn > 0:
            print(f"    {q:<12}: n={nn:>6} 胜率={tot_w/nn*100:.1f}%")
