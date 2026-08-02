#!/usr/bin/env python3
"""分析窗口精确化 — 判定窗口 W 的切换响应/稳态准确/判定有效性 (10-50根)

A. 合成数据 (有 ground truth): 切换后60根误判率 + 稳态误判率
B. 真实数据 (20标的 1H): 判定切换频率 (稳定性) + 插曲环境沿向1:1 (有效性)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from market_phase import analyze_market_state, _atr_series

W_LIST = [10, 15, 20, 25, 30, 40, 50]

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
switches = [i for i in range(1, len(TRUE)) if TRUE[i] != TRUE[i-1]]

print("═══ A. 合成数据: 切换响应 + 稳态准确 (ground truth) ═══", flush=True)
print(f"{'窗口':>5} {'切换后60根误判':>12} {'稳态误判':>10} {'综合分':>8}")
best = None
for W in W_LIST:
    preds = []
    for i in range(130, len(df)):
        ms = analyze_market_state(df.iloc[max(0, i - 120):i + 1].reset_index(drop=True), window=W)
        st = ms.get("state", "")
        if st == "trend_up":
            preds.append("trend_up")
        elif st == "trend_down":
            preds.append("trend_down")
        elif st == "range":
            preds.append("range")
        else:
            preds.append("transition")
    preds = np.array(preds)
    # 切换后 60 根误判
    resp_wrong = resp_n = 0
    for sw in switches:
        for j in range(sw, min(sw + 60, len(preds))):
            t = TRUE[j]
            p = preds[j]
            ok = (p in ("range", "transition")) if t == "range" else (p == t)
            resp_n += 1
            if not ok:
                resp_wrong += 1
    # 稳态误判 (切换 ±80 外)
    mask = np.ones(len(TRUE), bool)
    for sw in switches:
        mask[max(0, sw-80):sw+80] = False
    st_wrong = st_n = 0
    for j in range(len(preds)):
        if not mask[j]:
            continue
        t = TRUE[j]
        p = preds[j]
        ok = (p in ("range", "transition")) if t == "range" else (p == t)
        st_n += 1
        if not ok:
            st_wrong += 1
    resp = resp_wrong / resp_n * 100 if resp_n else 0
    stead = st_wrong / st_n * 100 if st_n else 0
    score = resp + stead  # 综合: 越低越好
    print(f"{W:>5} {resp:>11.1f}% {stead:>9.1f}% {score:>8.1f}")
    if best is None or score < best[0]:
        best = (score, W, resp, stead)
print(f"→ 合成最优: W={best[1]} (切换误判 {best[2]:.1f}% 稳态 {best[3]:.1f}%)")

# ═══════════════ B. 真实数据 (20标的 1H) ═══════════════
print("\n═══ B. 真实数据 (20标的 1H): 稳定性 + 插曲有效性 ═══", flush=True)
from backtest_engine import load_all
data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20[ts]):
            continue
        d[ts.date()] = "bull" if row["close"] > ma20[ts] else "bear"
    daily_states[sym] = d

def t4_state(sym, i, idx1, idx4, c4, ma20_4):
    t = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
    if t < 20 or np.isnan(ma20_4[t]):
        return "N"
    return "bull" if c4[t] > ma20_4[t] else "bear"

print(f"{'窗口':>5} {'状态切换率(次/100根)':>18} {'插曲沿向胜率':>12} {'插曲n':>7}")
for W in W_LIST:
    switch_cnt = 0
    total_i = 0
    ep_w = ep_l = 0
    for sym in syms:
        df1 = data[sym].get("1h")
        df4 = data[sym].get("4h")
        if df1 is None or df4 is None or len(df1) < 300:
            continue
        c = df1["close"].values
        hi = df1["high"].values
        lo = df1["low"].values
        n = len(df1)
        atr = _atr_series(df1)
        idx1 = df1.index.values.astype("datetime64[ns]")
        idx4 = df4.index.values.astype("datetime64[ns]")
        c4 = df4["close"].values
        ma20_4 = pd.Series(c4).rolling(20).mean().values
        ds = daily_states.get(sym, {})
        prev_state = None
        for i in range(200, n - 25, 6):
            ms = analyze_market_state(df1.iloc[max(0, i - 120):i + 1].reset_index(drop=True), window=W)
            st = ms.get("state", "unknown")
            total_i += 1
            if prev_state is not None and st != prev_state:
                switch_cnt += 1
            prev_state = st
            # 插曲环境: 日线顺势 + 4H逆 → 沿日线方向 1:1
            ts = pd.Timestamp(idx1[i])
            sd = ds.get(ts.date())
            if sd not in ("bull", "bear"):
                continue
            s4 = t4_state(sym, i, idx1, idx4, c4, ma20_4)
            if s4 == "N" or s4 == sd:
                continue
            a = atr[i]
            if a <= 0 or np.isnan(a):
                continue
            long_side = sd == "bull"
            entry = c[i]
            hit = 0
            for k in range(1, 25):
                if long_side:
                    if hi[i+k] >= entry + a:
                        hit = 1; break
                    if lo[i+k] <= entry - a:
                        hit = -1; break
                else:
                    if lo[i+k] <= entry - a:
                        hit = 1; break
                    if hi[i+k] >= entry + a:
                        hit = -1; break
            if hit == 1:
                ep_w += 1
            elif hit == -1:
                ep_l += 1
    switch_rate = switch_cnt / total_i * 100 if total_i else 0
    nn = ep_w + ep_l
    ep_wr = ep_w / nn * 100 if nn else 0
    print(f"{W:>5} {switch_rate:>16.2f} {ep_wr:>11.1f}% {nn:>7}")
