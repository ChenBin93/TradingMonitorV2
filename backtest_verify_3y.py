#!/usr/bin/env python3
"""3年数据关键结论复测 (多空平衡验证)

A. 插曲 vs 全顺势 (做日线方向, 胜率矩阵)
B. BB 反转 2.5σ (严格口径 + 用户口径)
C. 日线顺/逆做多 vs 做空 1:1 (验证多头年份"逆日线做多")
D. 强K 短窗口
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all

data = load_all()
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    for ts, row in daily.iterrows():
        if pd.isna(ma20[ts]):
            continue
        daily_states.setdefault(sym, {})[ts.date()] = "bull" if row["close"] > ma20[ts] else "bear"

def t4_state(sym, ts):
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 70:
        return "N"
    idx = df4.index.values.astype("datetime64[ns]")
    pos = int(np.searchsorted(idx, np.datetime64(ts) - np.timedelta64(240, "m"), side="right")) - 1
    if pos < 60:
        return "N"
    c = df4["close"].values
    ma20 = np.mean(c[pos-19:pos+1])
    return "bull" if c[pos] > ma20 else "bear"

print("═══ A. 插曲 vs 全顺势 (做日线方向, 1:1, 同bar双命中跳过) ═══", flush=True)
A_W = [4, 24]
A_T = [0.5, 1.0]
A_ep = {t: {w: [0, 0] for w in A_W} for t in A_T}
A_fl = {t: {w: [0, 0] for w in A_W} for t in A_T}
for sym in syms:
    df4 = data[sym].get("4h")
    df1 = data[sym].get("1h")
    if df4 is None or df1 is None or len(df4) < 70 or len(df1) < 130:
        continue
    c1 = df1["close"].values; h1 = df1["high"].values; l1 = df1["low"].values
    n1 = len(df1)
    tr1 = np.zeros(n1)
    for i in range(1, n1):
        tr1[i] = max(h1[i]-l1[i], abs(h1[i]-c1[i-1]), abs(l1[i]-c1[i-1]))
    atr1 = np.zeros(n1); e = 0.0
    for i in range(n1):
        e = tr1[i] if i == 0 else (1/14)*tr1[i] + (13/14)*e
        atr1[i] = e
    idx1 = df1.index.values.astype("datetime64[ns]")
    for i in range(100, n1 - 25):
        ts = pd.Timestamp(idx1[i])
        sd = daily_states.get(sym, {}).get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        s4 = t4_state(sym, ts)
        if s4 == "N":
            continue
        if s4 == sd:
            group = A_fl
        elif s4 != sd:
            group = A_ep
        else:
            continue
        entry = c1[i]; a = atr1[i]
        if a <= 0:
            continue
        long_side = sd == "bull"
        for t in A_T:
            for w in A_W:
                wp = lp = skip = False
                for k in range(1, w+1):
                    if i+k >= n1:
                        break
                    if long_side:
                        hp = h1[i+k] >= entry + t*a
                        hl = l1[i+k] <= entry - t*a
                    else:
                        hp = l1[i+k] <= entry - t*a
                        hl = h1[i+k] >= entry + t*a
                    if hp and hl:
                        skip = True; break
                    if hp:
                        wp = True; break
                    if hl:
                        lp = True; break
                if skip:
                    continue
                if wp:
                    group[t][w][0] += 1
                elif lp:
                    group[t][w][1] += 1

print(f"{'目标':<6} {'窗口':<6} {'插曲胜率':>10} {'插曲n':>8} {'全顺势胜率':>10} {'全顺势n':>8} {'Δpp':>6}")
for t in A_T:
    for w in A_W:
        ew, el = A_ep[t][w]; fw, fl = A_fl[t][w]
        ep_wr = ew/(ew+el)*100 if ew+el else 0
        fl_wr = fw/(fw+fl)*100 if fw+fl else 0
        print(f"{t:<6} {w:<6} {ep_wr:>9.1f}% {ew+el:>8} {fl_wr:>9.1f}% {fw+fl:>8} {ep_wr-fl_wr:>+5.1f}", flush=True)

print()
print("═══ B. BB 反转 2.5σ (日线空/中性, TP0.5/SL0.5/W48) ═══", flush=True)
B_strict = [0, 0, 0]  # win, loss, both
B_user = [0, 0]
for sym in syms:
    df4 = data[sym].get("4h")
    df1 = data[sym].get("1h")
    if df4 is None or df1 is None or len(df4) < 70 or len(df1) < 130:
        continue
    o1 = df1["open"].values; c1 = df1["close"].values
    h1 = df1["high"].values; l1 = df1["low"].values
    n1 = len(df1)
    tr1 = np.zeros(n1)
    for i in range(1, n1):
        tr1[i] = max(h1[i]-l1[i], abs(h1[i]-c1[i-1]), abs(l1[i]-c1[i-1]))
    atr1 = np.zeros(n1); e = 0.0
    for i in range(n1):
        e = tr1[i] if i == 0 else (1/14)*tr1[i] + (13/14)*e
        atr1[i] = e
    idx1 = df1.index.values.astype("datetime64[ns]")
    c1s = pd.Series(c1)
    ma = c1s.rolling(20).mean().values
    sd = c1s.rolling(20).std().values
    for i in range(100, n1 - 50):
        if np.isnan(ma[i]) or np.isnan(sd[i]) or sd[i] <= 0:
            continue
        ts = pd.Timestamp(idx1[i])
        ds = daily_states.get(sym, {}).get(ts.date())
        if ds not in ("bear", "neut"):
            continue
        up = ma[i] + 2.5*sd[i]
        if c1[i] <= up:
            continue
        entry = c1[i]; a = atr1[i]
        if a <= 0:
            continue
        wp = lp = both = False
        for k in range(1, 49):
            if i+k >= n1:
                break
            hp = l1[i+k] <= entry - 0.5*a
            hl = h1[i+k] >= entry + 0.5*a
            if hp and hl:
                both = True; break
            if hp:
                wp = True; break
            if hl:
                lp = True; break
        if both:
            B_strict[2] += 1
            B_user[0] += 0.5; B_user[1] += 1
        elif wp:
            B_strict[0] += 1; B_user[0] += 1; B_user[1] += 1
        elif lp:
            B_strict[1] += 1; B_user[1] += 1
n_strict = B_strict[0] + B_strict[1]
print(f"严格口径 (双命中跳过): n={n_strict} 胜率={B_strict[0]/n_strict*100:.1f}%")
print(f"用户口径 (双命中50%): n={B_user[1]} 不亏率={B_user[0]/B_user[1]*100:.1f}%")

print()
print("═══ C. 日线顺/逆 做多 vs 做空 (1:1, 1ATR, 24bar) ═══", flush=True)
C = {"顺势多": [0, 0], "逆势多": [0, 0], "顺势空": [0, 0], "逆势空": [0, 0]}
for sym in syms:
    df4 = data[sym].get("4h")
    df1 = data[sym].get("1h")
    if df4 is None or df1 is None or len(df4) < 70 or len(df1) < 130:
        continue
    c1 = df1["close"].values; h1 = df1["high"].values; l1 = df1["low"].values
    n1 = len(df1)
    tr1 = np.zeros(n1)
    for i in range(1, n1):
        tr1[i] = max(h1[i]-l1[i], abs(h1[i]-c1[i-1]), abs(l1[i]-c1[i-1]))
    atr1 = np.zeros(n1); e = 0.0
    for i in range(n1):
        e = tr1[i] if i == 0 else (1/14)*tr1[i] + (13/14)*e
        atr1[i] = e
    idx1 = df1.index.values.astype("datetime64[ns]")
    for i in range(100, n1 - 25):
        ts = pd.Timestamp(idx1[i])
        sd = daily_states.get(sym, {}).get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        s4 = t4_state(sym, ts)
        if s4 == "N":
            continue
        entry = c1[i]; a = atr1[i]
        if a <= 0:
            continue
        if sd == "bull":
            # 顺势多 (日线多时做多) / 逆势空 (日线多时做空)
            for long_side, key in [(True, "顺势多"), (False, "逆势空")]:
                wp = lp = skip = False
                for k in range(1, 25):
                    if i+k >= n1:
                        break
                    if long_side:
                        hp = h1[i+k] >= entry + a; hl = l1[i+k] <= entry - a
                    else:
                        hp = l1[i+k] <= entry - a; hl = h1[i+k] >= entry + a
                    if hp and hl:
                        skip = True; break
                    if hp:
                        wp = True; break
                    if hl:
                        lp = True; break
                if skip:
                    continue
                if wp:
                    C[key][0] += 1
                elif lp:
                    C[key][1] += 1
        else:
            # 顺势空 (日线空时做空) / 逆势多 (日线空时做多)
            for long_side, key in [(True, "逆势多"), (False, "顺势空")]:
                wp = lp = skip = False
                for k in range(1, 25):
                    if i+k >= n1:
                        break
                    if long_side:
                        hp = h1[i+k] >= entry + a; hl = l1[i+k] <= entry - a
                    else:
                        hp = l1[i+k] <= entry - a; hl = h1[i+k] >= entry + a
                    if hp and hl:
                        skip = True; break
                    if hp:
                        wp = True; break
                    if hl:
                        lp = True; break
                if skip:
                    continue
                if wp:
                    C[key][0] += 1
                elif lp:
                    C[key][1] += 1

for k in ["顺势多", "逆势多", "顺势空", "逆势空"]:
    w, l = C[k]
    if w + l:
        print(f"{k}: n={w+l} 胜率={w/(w+l)*100:.1f}%", flush=True)
