#!/usr/bin/env python3
"""贴位方式敏感度 (v2): 用最近极值(last_low/last_high)距离分档
- 刚跌破 [-0.5,0) vs 贴而未破 [0,0.5] vs 接近 (0.5,1.5]  × 顺日线/插曲
- 插曲 × 质量分组合 (三条件叠加, 任务2)
- T={0.5,1.0,2.0} 对称1:1, W=24; 与 structure2 (贴=63%) 对齐对照
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

TARGETS = [0.5, 1.0, 2.0]
W = 24
W_MAX = 96
LOOKBACK = 600
MIN_N = 100

t0 = time.time()
data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())
print(f"[load {time.time()-t0:.0f}s]", flush=True)

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
    # 收盘时间 = 日线开盘 + 24h → 查询只用已收盘日线 (无未来函数)
    daily_times[sym] = daily.index.values.astype("datetime64[ns]") + np.timedelta64(24, "h")

sw4 = {}
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

ST9_ID = {"日多4H多": 0, "日多4H中": 1, "日多4H空": 2,
          "日中4H多": 3, "日中4H中": 4, "日中4H空": 5,
          "日空4H多": 6, "日空4H中": 7, "日空4H空": 8}

# key = mode*6 + grp*3 + q ; mode: 0=刚跌破 1=贴而未破 2=接近; grp: 0=顺日线 1=插曲
NB = 18
acc = {"long": {T: [np.zeros(NB, int), np.zeros(NB, int)] for T in TARGETS},
       "short": {T: [np.zeros(NB, int), np.zeros(NB, int)] for T in TARGETS}}


def hit_stats(H, L, E, A, T):
    up = H >= (E + T * A)[:, None]
    dn = L <= (E - T * A)[:, None]
    f_up = up.argmax(axis=1) + 1
    f_dn = dn.argmax(axis=1) + 1
    has_up = up.any(axis=1)
    has_dn = dn.any(axis=1)
    first = np.where(has_up & ~has_dn, 1,
                     np.where(~has_up & has_dn, -1,
                              np.where(has_up & has_dn & (f_up < f_dn), 1,
                                       np.where(has_up & has_dn & (f_dn < f_up), -1, 0))))
    bar = np.where(first == 1, f_up, np.where(first == -1, f_dn, 0))
    return first, bar


n_samples = 0
for sym in syms:
    t_sym = time.time()
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
    sw = sw4.get(sym)

    is_low = l <= pd.Series(l).rolling(5, center=True).min().values
    is_high = h >= pd.Series(h).rolling(5, center=True).max().values
    idx_arr = np.arange(n)
    last_low = np.maximum.accumulate(np.where(is_low, idx_arr, -1))
    last_high = np.maximum.accumulate(np.where(is_high, idx_arr, -1))

    i_s = np.arange(LOOKBACK + 50, n - W_MAX, 4)
    st_daily = daily_states.get(sym)
    if st_daily is None:
        continue
    close_daily = daily_times[sym]
    # 无未来函数: 只使用已收盘日线 (收盘时刻 <= 采样时刻)
    jd = np.searchsorted(close_daily, idx1[i_s], side="right") - 1
    ok_d = jd >= 0
    sd_vals = np.where(ok_d, st_daily[np.clip(jd, 0, None)], "")
    t4s = np.searchsorted(idx4, idx1[i_s] - np.timedelta64(240, "m"), side="right") - 1
    ok4 = (t4s >= 20) & ~np.isnan(ma20_4[t4s]) & (atr4[t4s] > 0)
    dev4 = np.where(ok4, (c4[t4s] - ma20_4[t4s]) / atr4[t4s], 0.0)
    s4 = np.where(dev4 > 0.5, "4H多", np.where(dev4 < -0.5, "4H空", "4H中"))
    st9 = np.array(["日" + sd + s for sd, s in zip(sd_vals.astype(str), s4)])
    st9_id = np.array([ST9_ID.get(x, -1) for x in st9])
    keep = ok_d & ok4 & (st9_id >= 0)
    i_s = i_s[keep]
    st9_id = st9_id[keep]
    if len(i_s) == 0:
        continue

    E = c[i_s]
    A = atr[i_s]
    H = h[i_s[:, None] + np.arange(1, W_MAX + 1)]
    L = l[i_s[:, None] + np.arange(1, W_MAX + 1)]

    long_j, long_key = [], []
    short_j, short_key = [], []
    for j, i in enumerate(i_s):
        a = A[j]
        if a <= 0 or np.isnan(a):
            continue
        entry = E[j]
        sid = st9_id[j]
        w_start = i - LOOKBACK

        # ── 支撑侧 (做多): 最近低点 ──
        il = last_low[max(0, i - 2)]  # 无未来函数: 极值需未来2根确认
        if il >= 0:
            d_lo = (entry - l[il]) / a
            if -0.5 <= d_lo <= 1.5:
                if d_lo < 0:
                    mode, q = 0, 0
                elif d_lo <= 0.5:
                    mode = 1
                    age = i - il
                    # touch: 窗口内贴近现价(±0.5ATR)的低点极值数
                    lm = is_low[w_start:i]
                    if lm.any():
                        lp = l[w_start:i][lm]
                        touch = int((((entry - lp) >= -0.5 * a) & ((entry - lp) <= 0.5 * a)).sum())
                    else:
                        touch = 0
                    sc = 0
                    if 200 <= age < 400:
                        sc += 2
                    elif age >= 100:
                        sc += 1
                    if touch >= 5:
                        sc += 1
                    elif touch >= 3:
                        sc += 0.5
                    if sw is not None:
                        j4 = sw[(sw >= max(0, len(c4) - 50)) & (sw < len(c4))]
                        if len(j4) and bool(np.any(np.abs(c4[j4] - entry) <= 0.5 * a)):
                            sc += 1.5
                    q = 2 if sc >= 3 else 1 if sc >= 1.5 else 0
                else:
                    mode, q = 2, 0
                if sid == 2:
                    grp = 1
                elif sid < 3:
                    grp = 0
                else:
                    grp = -1
                if grp >= 0:
                    long_j.append(j)
                    long_key.append(mode * 6 + grp * 3 + q)

        # ── 阻力侧 (做空): 最近高点 ──
        ih = last_high[max(0, i - 2)]  # 无未来函数
        if ih >= 0:
            d_hi = (h[ih] - entry) / a
            if -0.5 <= d_hi <= 1.5:
                if d_hi < 0:
                    mode, q = 0, 0
                elif d_hi <= 0.5:
                    mode = 1
                    age = i - ih
                    hm = is_high[w_start:i]
                    if hm.any():
                        hp = h[w_start:i][hm]
                        touch = int((((hp - entry) >= -0.5 * a) & ((hp - entry) <= 0.5 * a)).sum())
                    else:
                        touch = 0
                    sc = 0
                    if 200 <= age < 400:
                        sc += 2
                    elif age >= 100:
                        sc += 1
                    if touch >= 5:
                        sc += 1
                    elif touch >= 3:
                        sc += 0.5
                    if sw is not None:
                        j4 = sw[(sw >= max(0, len(c4) - 50)) & (sw < len(c4))]
                        if len(j4) and bool(np.any(np.abs(c4[j4] - entry) <= 0.5 * a)):
                            sc += 1.5
                    q = 2 if sc >= 3 else 1 if sc >= 1.5 else 0
                else:
                    mode, q = 2, 0
                if sid == 6:
                    grp = 1
                elif sid >= 6:
                    grp = 0
                else:
                    grp = -1
                if grp >= 0:
                    short_j.append(j)
                    short_key.append(mode * 6 + grp * 3 + q)

    for side, js, ks in (("long", long_j, long_key), ("short", short_j, short_key)):
        if not js:
            continue
        js = np.array(js)
        ks = np.array(ks)
        Hs, Ls, Es, As = H[js], L[js], E[js], A[js]
        for T in TARGETS:
            first, bar = hit_stats(Hs, Ls, Es, As, T)
            v = (first != 0) & (bar <= W)
            win = v & (first == 1)
            if side == "short":
                win = v & (first == -1)
            if v.any():
                acc[side][T][0] += np.bincount(ks[win], minlength=NB)
                acc[side][T][1] += np.bincount(ks[v], minlength=NB)
        n_samples += len(js)
    print(f"{sym}: pts={len(i_s)} ({time.time()-t_sym:.0f}s)", flush=True)

print(f"\n[scan {n_samples} samples, total {time.time()-t0:.0f}s]", flush=True)

MODE_N = ["刚跌破(-0.5~0)", "贴而未破(0~0.5)", "接近(0.5~1.5)"]
GRP_N = ["顺日线", "插曲"]
Q_N = ["低", "中", "高"]


def show(side, dname):
    print(f"\n═══ {dname} (W=24, 对称1:1) ═══")
    for mode, mn in enumerate(MODE_N):
        for grp, gn in enumerate(GRP_N):
            line = f"{mn}·{gn:<6}"
            for T in TARGETS:
                s = mode * 6 + grp * 3
                tot = int(acc[side][T][1][s: s + 3].sum())
                win = int(acc[side][T][0][s: s + 3].sum())
                if tot < MIN_N:
                    line += f"{'--('+str(tot)+')':>20}"
                else:
                    line += f"{win/tot*100:>7.1f}%({tot:>4})".rjust(20)
            print(line)
    print("\n  插曲 × 质量分 (仅贴而未破):")
    for q, qn in enumerate(Q_N):
        line = f"质量={qn:<6}"
        for T in TARGETS:
            key = 1 * 6 + 1 * 3 + q
            tot = int(acc[side][T][1][key])
            win = int(acc[side][T][0][key])
            if tot < MIN_N:
                line += f"{'--('+str(tot)+')':>20}"
            else:
                line += f"{win/tot*100:>7.1f}%({tot:>4})".rjust(20)
        print(line)


show("long", "贴支撑 → 做多")
show("short", "贴阻力 → 做空")
