#!/usr/bin/env python3
"""市场状态 × 关键位 胜率分布 — 向量化版 (性能优化)

替代 find_swing_levels 的逐点重算:
  预计算极值 mask + 对采样点 numpy 切片接近检测 → O(600) 向量化比较
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

t0 = time.time()
data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())

daily_data = {}
daily_states = {}
daily_times = {}
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
    d = []
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]) or pd.isna(atr_d[ts]) or atr_d[ts] <= 0:
            d.append("中")
            continue
        dev = (row["close"] - ma20d[ts]) / atr_d[ts]
        d.append("多" if dev > 0.5 else "空" if dev < -0.5 else "中")
    daily_states[sym] = np.array(d, dtype=object)
    daily_times[sym] = daily.index.values.astype("datetime64[ns]") + np.timedelta64(24, "h")

LOOKBACK = 600

# 预计算 4H/日线 swing 极值 (用于重叠标志)
sw4 = {}
swd = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None:
        continue
    h4 = df4["high"].values
    l4 = df4["low"].values
    sw4[sym] = np.where(
        (h4 >= pd.Series(h4).rolling(5, center=True).max().values) |
        (l4 <= pd.Series(l4).rolling(5, center=True).min().values)
    )[0]
    if sym in daily_data:
        daily = daily_data[sym]
        hd = daily["high"].values
        ld = daily["low"].values
        swd[sym] = np.where(
            (hd >= pd.Series(hd).rolling(5, center=True).max().values) |
            (ld <= pd.Series(ld).rolling(5, center=True).min().values)
        )[0]

STATE9 = ["日多4H多", "日多4H中", "日多4H空",
          "日中4H多", "日中4H中", "日中4H空",
          "日空4H多", "日空4H中", "日空4H空"]
Q3 = ["低", "中", "高"]
res = {d: {s: {q: [0, 0] for q in Q3} for s in STATE9} for d in ["long", "short"]}
n_scan = 0
t_collect = t0

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
    st_daily = daily_states.get(sym)
    close_daily = daily_times.get(sym)
    if st_daily is None:
        continue

    # 预计算 1H 极值 mask (向量化)
    is_low = l <= pd.Series(l).rolling(5, center=True).min().values
    is_high = h >= pd.Series(h).rolling(5, center=True).max().values

    # 4H/日线重叠: 预计算"最近 50 根 4H 极值集合"的滑动数组过于复杂,
    # 简化: 采样点内检查 (开销小)

    for i in range(LOOKBACK + 50, n - 25, 4):
        ts = idx1[i]
        jd = int(np.searchsorted(close_daily, ts, side="right")) - 1
        if jd < 0:
            continue
        sd = st_daily[jd]
        t4 = int(np.searchsorted(idx4, ts - np.timedelta64(240, "m"), side="right")) - 1
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
        tol = 0.5 * a

        # ── 向量化贴位检测: 近 LOOKBACK 根的极值接近比较 ──
        w_start = i - LOOKBACK
        # 低点 (支撑候选): 价格 ≤ entry, 且 entry - low ∈ [0, tol]
        lo_mask = is_low[w_start:i]
        if lo_mask.any():
            lo_prices = l[w_start:i][lo_mask]
            lo_age = (i - (w_start + np.where(lo_mask)[0]))  # 年龄 (距 i)
            # 贴支撑: 0 ≤ entry - low ≤ tol
            sup = (lo_prices <= entry) & (entry - lo_prices <= tol)
            if sup.any():
                sup_touch = int(sup.sum())
                sup_age = int(lo_age[sup].min())
                # 重叠检查
                ov4 = ovd = False
                if sym in sw4:
                    ext4 = sw4[sym]
                    j4 = ext4[(ext4 >= max(0, len(c4) - 50)) & (ext4 < len(c4))]
                    ov4 = bool(np.any(np.abs(c4[j4] - entry) <= tol)) if len(j4) else False
                if sym in swd:
                    extd = swd[sym]
                    jd = extd[(extd >= max(0, len(daily_data[sym]) - 20)) & (extd < len(daily_data[sym]))]
                    ovd = bool(np.any(np.abs(daily_data[sym]["low"].values[jd] - entry) <= tol)) if len(jd) else False
                sc = 0
                if 200 <= sup_age < 400:
                    sc += 2
                elif sup_age >= 100 or sup_age >= 400:
                    sc += 1
                if sup_touch >= 5:
                    sc += 1
                elif sup_touch >= 3:
                    sc += 0.5
                if ov4:
                    sc += 1.5
                elif ovd:
                    sc += 0.5
                q = "高" if sc >= 3 else "中" if sc >= 1.5 else "低"
                # 做多 1:1
                hit = 0
                for k in range(1, 25):
                    if h[i+k] >= entry + a:
                        hit = 1; break
                    if l[i+k] <= entry - a:
                        hit = -1; break
                if hit != 0:
                    res["long"][st9][q][0] += 1 if hit == 1 else 0
                    res["long"][st9][q][1] += 1

        # 高点 (阻力候选): 价格 ≥ entry, 且 high - entry ∈ [0, tol]
        hi_mask = is_high[w_start:i]
        if hi_mask.any():
            hi_prices = h[w_start:i][hi_mask]
            hi_age = (i - (w_start + np.where(hi_mask)[0]))
            res_ = (hi_prices >= entry) & (hi_prices - entry <= tol)
            if res_.any():
                res_touch = int(res_.sum())
                res_age = int(hi_age[res_].min())
                ov4 = ovd = False
                if sym in sw4:
                    ext4 = sw4[sym]
                    j4 = ext4[(ext4 >= max(0, len(c4) - 50)) & (ext4 < len(c4))]
                    ov4 = bool(np.any(np.abs(c4[j4] - entry) <= tol)) if len(j4) else False
                if sym in swd:
                    extd = swd[sym]
                    jd = extd[(extd >= max(0, len(daily_data[sym]) - 20)) & (extd < len(daily_data[sym]))]
                    ovd = bool(np.any(np.abs(daily_data[sym]["high"].values[jd] - entry) <= tol)) if len(jd) else False
                sc = 0
                if 200 <= res_age < 400:
                    sc += 2
                elif res_age >= 100 or res_age >= 400:
                    sc += 1
                if res_touch >= 5:
                    sc += 1
                elif res_touch >= 3:
                    sc += 0.5
                if ov4:
                    sc += 1.5
                elif ovd:
                    sc += 0.5
                q = "高" if sc >= 3 else "中" if sc >= 1.5 else "低"
                hit = 0
                for k in range(1, 25):
                    if l[i+k] <= entry - a:
                        hit = 1; break
                    if h[i+k] >= entry + a:
                        hit = -1; break
                if hit != 0:
                    res["short"][st9][q][0] += 1 if hit == 1 else 0
                    res["short"][st9][q][1] += 1
    print(f"{sym}: done ({time.time()-t_collect:.0f}s)", flush=True)

print(f"\n扫描采样: {n_scan}, 总耗时: {time.time()-t0:.0f}s")
for dir_key, dname in [("long", "═══ 贴支撑未破 → 做多 ═══"),
                       ("short", "═══ 贴阻力未破 → 做空 ═══")]:
    print(f"\n{dname}")
    print(f"  {'状态':<8} {'全部':>12} {'低质量':>12} {'中质量':>12} {'高质量':>12}")
    for st in STATE9:
        line = f"  {st:<8}"
        for col, qs in [("全部", Q3), ("低", ["低"]), ("中", ["中"]), ("高", ["高"])]:
            w = sum(res[dir_key][st][q][0] for q in qs)
            l = sum(res[dir_key][st][q][1] for q in qs)
            nn = l  # FIX: res[1] 已是命中总数(win+loss), 不能再加 w
            if nn < 100:
                line += f"{'--('+str(nn)+')':>12}"
            else:
                line += f"{w/nn*100:>7.1f}%({nn:>4})"
        print(line)
    if dir_key == "long":
        ok_states = ["日多4H多", "日多4H中", "日多4H空"]
        ko_states = ["日空4H多", "日空4H中", "日空4H空"]
    else:
        ok_states = ["日空4H多", "日空4H中", "日空4H空"]
        ko_states = ["日多4H多", "日多4H中", "日多4H空"]
    for label, states in [("顺日线", ok_states), ("逆日线", ko_states)]:
        w = sum(res[dir_key][s][q][0] for s in states for q in Q3)
        l = sum(res[dir_key][s][q][1] for s in states for q in Q3)
        nn = l  # FIX
        if nn > 0:
            print(f"  [汇总] {label}: n={nn:>6} 胜率={w/nn*100:.1f}%")
