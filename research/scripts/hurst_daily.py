#!/usr/bin/env python3
"""Hurst 日线补全 — 20 标的日线 Hurst + 预测验证 (补 1H/4H 缺失的日线)"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

data = load_all(timeframes=["4h"])
syms = list(data.keys())
H_EDGES = [(0, 0.45, "回归态"), (0.45, 0.55, "随机"), (0.55, 2.0, "趋势态")]


def dfa_hurst(x, s_list=(4, 8, 16, 32, 64)):
    n = len(x)
    if n < 16:
        return 0.5
    y = np.cumsum(x - x.mean())
    F = []
    for s in s_list:
        nseg = n // s
        if nseg < 2:
            continue
        seg = y[:nseg * s].reshape(nseg, s)
        t2 = np.arange(s) - (s - 1) / 2.0
        denom = (t2 ** 2).sum()
        if denom <= 0:
            continue
        b = (seg - seg.mean(axis=1, keepdims=True)) @ t2 / denom
        trend = seg.mean(axis=1, keepdims=True) + b[:, None] * t2
        F.append(float(np.sqrt(((seg - trend) ** 2).mean())))
    if len(F) < 3:
        return 0.5
    return float(np.polyfit(np.log(np.array(s_list[:len(F)])), np.log(np.array(F)), 1)[0])


daily_data = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 400:
        continue
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(daily) >= 300:
        daily_data[sym] = daily

print(f"日线标的数: {len(daily_data)}")

for WIN in [128, 256]:
    print(f"\n{'='*56}\n═══ 日线 Hurst 窗口 {WIN} ═══\n{'='*56}", flush=True)
    all_h = []
    tier_n = {name: 0 for _, _, name in H_EDGES}
    tier_stats = {name: [0, 0] for _, _, name in H_EDGES}  # 动量向 w/l
    tier_against = {name: [0, 0] for _, _, name in H_EDGES}
    per_sym = {}
    for sym, df in daily_data.items():
        c = df["close"].values
        hi = df["high"].values
        lo = df["low"].values
        n = len(df)
        ret = np.diff(c) / c[:-1] * 100
        n_ret = len(ret)
        atr = _atr_series(df)
        hvals = []
        for i in range(WIN + 10, n_ret - 10, 3):
            h = dfa_hurst(ret[i-WIN:i])
            hvals.append(h)
            all_h.append(h)
            name = None
            for lo_, hi_, nm in H_EDGES:
                if lo_ <= h < hi_:
                    name = nm
                    break
            if name is None:
                continue
            tier_n[name] += 1
            j = i + 1
            a = atr[j]
            if a <= 0 or np.isnan(a):
                continue
            entry = c[j]
            mom_dir = 1 if ret[i-1] > 0 else -1
            # 沿动量
            w = l = 0
            for k in range(1, 11):
                if mom_dir == 1:
                    if hi[j+k] >= entry + a:
                        w = 1; break
                    if lo[j+k] <= entry - a:
                        l = 1; break
                else:
                    if lo[j+k] <= entry - a:
                        w = 1; break
                    if hi[j+k] >= entry + a:
                        l = 1; break
            if w:
                tier_stats[name][0] += 1
            elif l:
                tier_stats[name][1] += 1
            # 逆动量
            w = l = 0
            for k in range(1, 11):
                if mom_dir == -1:
                    if hi[j+k] >= entry + a:
                        w = 1; break
                    if lo[j+k] <= entry - a:
                        l = 1; break
                else:
                    if lo[j+k] <= entry - a:
                        w = 1; break
                    if hi[j+k] >= entry + a:
                        l = 1; break
            if w:
                tier_against[name][0] += 1
            elif l:
                tier_against[name][1] += 1
        per_sym[sym] = np.mean(hvals) if hvals else 0.5

    h_arr = np.array(all_h)
    print(f"\n日线 Hurst: 均值 {h_arr.mean():.3f} 中位 {np.median(h_arr):.3f} "
          f"p25 {np.percentile(h_arr,25):.3f} p75 {np.percentile(h_arr,75):.3f}")
    print(f"标的均值分布: {np.mean(list(per_sym.values())):.3f} (min {min(per_sym.values()):.3f}, "
          f"max {max(per_sym.values()):.3f})")
    print("\n分档 × 后续10天 1:1 (±1日线ATR):")
    print(f"{'分档':<6} {'n':>8} {'沿动量':>8} {'逆动量':>8}")
    for name in ["回归态", "随机", "趋势态"]:
        w, l = tier_stats[name]
        nn = w + l
        wa, la = tier_against[name]
        nna = wa + la
        wr = f"{w/nn*100:.1f}%" if nn >= 100 else f"--({nn})"
        wra = f"{wa/nna*100:.1f}%" if nna >= 100 else f"--({nna})"
        print(f"{name:<6} {tier_n[name]:>8} {wr:>8} {wra:>8}")
