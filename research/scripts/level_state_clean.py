#!/usr/bin/env python3
"""市场状态 × 关键位 胜率分布 — 干净版 (覆盖全部 9 种状态组合)

对每个采样点独立测两个方向:
  贴支撑未破(0≤dist≤0.5ATR) → 做多 1:1
  贴阻力未破(0≤dist≤0.5ATR) → 做空 1:1
状态: 日线(多/中/空) × 4H(多/中/空), ±0.5ATR 三分类
质量: 年龄+touch+重叠 三档
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

# 日线合成 + 三分类状态
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
        d[ts.date()] = "多" if dev > 0.5 else "空" if dev < -0.5 else "中"
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
        return "高"
    if sc >= 1.5:
        return "中"
    return "低"

# 结果: dir('long'/'short') → state9 → q → [w, l]
STATE9 = ["日多4H多", "日多4H中", "日多4H空",
          "日中4H多", "日中4H中", "日中4H空",
          "日空4H多", "日空4H中", "日空4H空"]
Q3 = ["低", "中", "高"]
res = {d: {s: {q: [0, 0] for q in Q3} for s in STATE9} for d in ["long", "short"]}
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
        if sd is None:
            continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]):
            continue
        dev4 = (c4[t4] - ma20_4[t4]) / atr4[t4] if atr4[t4] > 0 else 0
        s4 = "多" if dev4 > 0.5 else "空" if dev4 < -0.5 else "中"
        st9 = "日" + sd + "4H" + s4
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c[i]
        n_scan += 1
        levels = find_swing_levels(df1.iloc[:i + 1].tail(LOOKBACK), LOOKBACK)

        # 贴支撑做多 与 贴阻力做空 — 独立检测
        for side_key, dir_key, cond in [
            ("support", "long", lambda lv: 0 <= entry - lv.price <= 0.5 * a),
            ("resistance", "short", lambda lv: 0 <= lv.price - entry <= 0.5 * a),
        ]:
            hit_lv = None
            for lv in levels:
                if lv.side == side_key and cond(lv):
                    hit_lv = lv
                    break
            if hit_lv is None:
                continue
            age = i - (len(df1.iloc[:i + 1]) - LOOKBACK + hit_lv.last_touch_idx) if hit_lv.last_touch_idx >= 0 else 999
            ov = overlap_flags(sym, hit_lv.price, a)
            q = qband(age, hit_lv.touch_count, ov[0], ov[1])
            # 1:1 判定 (沿贴位方向)
            long_side = (dir_key == "long")
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
            res[dir_key][st9][q][0] += 1 if hit == 1 else 0
            res[dir_key][st9][q][1] += 1
    print(f"{sym}: done", flush=True)

print(f"\n扫描采样: {n_scan}")
for dir_key, dname in [("long", "═══ 贴支撑未破 → 做多 ═══"),
                       ("short", "═══ 贴阻力未破 → 做空 ═══")]:
    print(f"\n{dname}")
    print(f"  {'状态':<8} {'全部':>12} {'低质量':>12} {'中质量':>12} {'高质量':>12}")
    for st in STATE9:
        line = f"  {st:<8}"
        for col, qs in [("全部", Q3), ("低", ["低"]), ("中", ["中"]), ("高", ["高"])]:
            w = sum(res[dir_key][st][q][0] for q in qs)
            l = sum(res[dir_key][st][q][1] for q in qs)
            nn = w + l
            if nn < 100:
                line += f"{'--('+str(nn)+')':>12}"
            else:
                line += f"{w/nn*100:>7.1f}%({nn:>4})"
        print(line)
    # 顺/逆日线汇总
    if dir_key == "long":
        ok_states = ["日多4H多", "日多4H中", "日多4H空"]
        ko_states = ["日空4H多", "日空4H中", "日空4H空"]
    else:
        ok_states = ["日空4H多", "日空4H中", "日空4H空"]
        ko_states = ["日多4H多", "日多4H中", "日多4H空"]
    for label, states in [("顺日线", ok_states), ("逆日线", ko_states)]:
        w = sum(res[dir_key][s][q][0] for s in states for q in Q3)
        l = sum(res[dir_key][s][q][1] for s in states for q in Q3)
        nn = w + l
        if nn > 0:
            print(f"  [汇总] {label}: n={nn:>6} 胜率={w/nn*100:.1f}%")
