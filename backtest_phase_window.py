#!/usr/bin/env python3
"""趋势判定窗口 — 按周期(日线/4H/1H)真实窗口调优

真实窗口语义: analyze_market_state(df.tail(tw)) — tw 真正决定指标计算长度
评估: 沿判定方向做单 (trend_up→多, trend_down→空) 的 1:1 区分度
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import analyze_market_state

data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())
t0 = time.time()

# ═══ 日线合成 (每标的) ═══
daily_data = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 200:
        continue
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(daily) >= 200:
        daily_data[sym] = daily


def build_samples(df, interval, horizon, atr_ser):
    """采样点: (i, 后续1:1沿方向 out=1/-1/0)"""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    out = []
    for i in range(200, n - horizon - 1, interval):
        a = atr_ser[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c[i]
        res = 0
        for k in range(1, horizon + 1):
            if i + k >= n:
                break
            if h[i+k] >= entry + a:
                res = 1
                break
            if l[i+k] <= entry - a:
                res = -1
                break
        out.append((i, res))
    return out


def atr_seq(df, period=14):
    from market_phase import _atr_series
    return _atr_series(df, period)


def tune(tf_name, df_pool, samples, tws, horizon, atr_pool):
    print(f"\n═══ {tf_name} 趋势窗口 ═══")
    print(f"{'TW':>5} {'n_up':>7} {'up胜率':>7} {'n_dn':>7} {'dn胜率':>7} {'区分度':>7}")
    best = None
    for tw in tws:
        up_w = up_l = dn_w = dn_l = 0
        for sym, samples_sym in samples.items():
            df = df_pool[sym]
            atr = atr_pool[sym]
            for i, out in samples_sym:
                if i < tw:
                    continue
                ms = analyze_market_state(df.iloc[max(0, i - tw - 59):i + 1].reset_index(drop=True), window=tw)
                state = ms.get("state", "")
                if state == "trend_up":
                    if out == 1:
                        up_w += 1
                    elif out == -1:
                        up_l += 1
                elif state == "trend_down":
                    if out == -1:
                        dn_w += 1
                    elif out == 1:
                        dn_l += 1
        n_up = up_w + up_l
        n_dn = dn_w + dn_l
        if n_up < 300 or n_dn < 300:
            print(f"{tw:>5} 样本不足 ({n_up}/{n_dn})")
            continue
        up_wr = up_w / n_up * 100
        dn_wr = dn_w / n_dn * 100
        diff = up_wr - dn_wr
        print(f"{tw:>5} {n_up:>7} {up_wr:>6.1f}% {n_dn:>7} {dn_wr:>6.1f}% {diff:>+6.1f}")
        if best is None or diff > best[0]:
            best = (diff, tw, up_wr, dn_wr, n_up + n_dn)
    if best:
        print(f"→ 最优 TW={best[1]} 区分度={best[0]:+.1f}pp (up {best[2]:.1f}% / dn {best[3]:.1f}%, n={best[4]})")
    return best


# ═══ 1H ═══
print("构建 1H 采样...", flush=True)
df_1h = {s: data[s].get("1h") for s in syms}
atr_1h = {s: atr_seq(df_1h[s]) for s in syms if df_1h[s] is not None and len(df_1h[s]) > 220}
samp_1h = {s: build_samples(df_1h[s], 12, 24, atr_1h[s]) for s in atr_1h}
tune("1H", df_1h, samp_1h, [50, 70, 90, 120, 150], 24, atr_1h)
print(f"  [{time.time()-t0:.0f}s]", flush=True)

# ═══ 4H ═══
print("构建 4H 采样...", flush=True)
df_4h = {s: data[s].get("4h") for s in syms}
atr_4h = {s: atr_seq(df_4h[s]) for s in syms if df_4h[s] is not None and len(df_4h[s]) > 220}
samp_4h = {s: build_samples(df_4h[s], 3, 12, atr_4h[s]) for s in atr_4h}
tune("4H", df_4h, samp_4h, [50, 70, 90, 120, 150], 12, atr_4h)
print(f"  [{time.time()-t0:.0f}s]", flush=True)

# ═══ 日线 ═══
print("构建 日线采样...", flush=True)
atr_d = {s: atr_seq(daily_data[s]) for s in daily_data}
samp_d = {s: build_samples(daily_data[s], 1, 10, atr_d[s]) for s in daily_data}
tune("日线", daily_data, samp_d, [30, 50, 70, 90, 120], 10, atr_d)
print(f"  [{time.time()-t0:.0f}s]", flush=True)
