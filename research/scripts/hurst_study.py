#!/usr/bin/env python3
"""滚动 Hurst 指数研究 — 趋势性 vs 均值回归状态 (BTC, 1H/4H)

DFA (去趋势波动分析) 滚动 Hurst:
  H>0.55 趋势态 (延续) | 0.45-0.55 随机 | H<0.45 回归态 (均值回归)

验证:
  A. Hurst 分布 + 窗口敏感性 (128/256/512 根)
  B. Hurst 分档 × 后续 1:1 (双向 + 动量方向)
  C. 与 market_phase 互补性 (交叉)
  D. 策略映射: Hurst 分档 × 插曲/BB反转
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

SYM = "BTC/USDT:USDT"
H_EDGES = [(0, 0.45, "回归态"), (0.45, 0.55, "随机"), (0.55, 2.0, "趋势态")]


def dfa_hurst(x: np.ndarray, s_list=(4, 8, 16, 32, 64)) -> float:
    """DFA 滚动 Hurst — 全向量化分段去趋势"""
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
        resid = seg - trend
        F.append(float(np.sqrt((resid ** 2).mean())))
    if len(F) < 3:
        return 0.5
    return float(np.polyfit(np.log(np.array(s_list[:len(F)])), np.log(np.array(F)), 1)[0])


def hurst_tier(h):
    for lo, hi, name in H_EDGES:
        if lo <= h < hi:
            return name
    return "随机"


data = load_all(timeframes=["1h", "4h"])
df1 = data[SYM]["1h"]
df4 = data[SYM]["4h"]

for tf_name, df, SAMP, HZ, ATR_PER in [
    ("1h", df1, 6, 24, 14),
    ("4h", df4, 2, 12, 14),
]:
    print(f"\n{'='*60}\n═══ {tf_name} (n={len(df)}) ═══\n{'='*60}", flush=True)
    c = df["close"].values
    ret = np.diff(c) / c[:-1] * 100
    n_ret = len(ret)
    atr = _atr_series(df)
    hi = df["high"].values
    lo = df["low"].values

    for WIN in [128, 256, 512]:
        # 滚动 Hurst (采样) — 同时记录 (df索引, tier)
        idxs = list(range(WIN + 10, n_ret, SAMP))
        hvals = np.array([dfa_hurst(ret[i-WIN:i]) for i in idxs])
        tiers = np.array([hurst_tier(h) for h in hvals])
        idx_df = [i + 1 for i in idxs]  # ret 索引 → df 索引

        print(f"\n── 窗口 {WIN} 根 ──")
        print(f"Hurst: 均值 {hvals.mean():.3f} 中位 {np.median(hvals):.3f} "
              f"p25 {np.percentile(hvals,25):.3f} p75 {np.percentile(hvals,75):.3f}")

        # 分布
        for name in ["回归态", "随机", "趋势态"]:
            cnt = (tiers == name).sum()
            print(f"  {name}: {cnt} ({cnt/len(tiers)*100:.1f}%)")

        # B. Hurst 分档 × 后续 1:1 (动量方向 + 双向)
        print(f"\n  后续 {HZ} 根 1:1 (±1ATR):")
        print(f"  {'分档':<6} {'n':>7} {'动量向胜率':>10} {'双向胜率':>9}")
        for name in ["回归态", "随机", "趋势态"]:
            sel = np.where(tiers == name)[0]
            w_m = l_m = w_b = l_b = 0
            for idx in sel:
                i = idxs[idx]
                j = i  # ret 索引 j → df 索引 j+1
                if j + HZ >= n_ret or atr[j+1] <= 0:
                    continue
                entry = c[j+1]
                a = atr[j+1]
                # 动量方向: 窗口内净移动方向
                mom_dir = 1 if ret[i-1] > 0 else -1  # 用窗口最后收益方向近似
                hit_m = 0
                hit_b = 0
                for k in range(1, HZ + 1):
                    if mom_dir == 1:
                        if hi[j+1+k] >= entry + a:
                            hit_m = 1; break
                        if lo[j+1+k] <= entry - a:
                            hit_m = -1; break
                    else:
                        if lo[j+1+k] <= entry - a:
                            hit_m = 1; break
                        if hi[j+1+k] >= entry + a:
                            hit_m = -1; break
                for k in range(1, HZ + 1):
                    if hi[j+1+k] >= entry + a:
                        hit_b = 1; break
                    if lo[j+1+k] <= entry - a:
                        hit_b = -1; break
                if hit_m == 1:
                    w_m += 1
                elif hit_m == -1:
                    l_m += 1
                if hit_b == 1:
                    w_b += 1
                elif hit_b == -1:
                    l_b += 1
            nn_m = w_m + l_m
            nn_b = w_b + l_b
            print(f"  {name:<6} {len(sel):>7} "
                  f"{w_m/nn_m*100 if nn_m else 0:>9.1f}% "
                  f"{w_b/nn_b*100 if nn_b else 0:>8.1f}%")

        if WIN == 256:
            # C. 与 market_phase 互补性 (ADX 判定交叉)
            print(f"\n  ── 与 market_phase 交叉 (窗口 {WIN}) ──")
            from market_phase import analyze_market_state
            cross = np.zeros((3, 3), dtype=int)
            n_cross = 0
            for idx in idx_df:
                i = idx
                if i < 120:
                    continue
                ms = analyze_market_state(df.iloc[max(0, i-120):i+1].reset_index(drop=True))
                st = ms.get("state", "")
                if st.startswith("trend"):
                    adx_tier = 2
                elif st == "range":
                    adx_tier = 0
                else:
                    adx_tier = 1
                kk = idx_df.index(idx)
                h_t = {"回归态": 0, "随机": 1, "趋势态": 2}[tiers[kk]]
                cross[h_t, adx_tier] += 1
                n_cross += 1
            names = ["回归态", "随机", "趋势态"]
            print(f"  (行=Hurst, 列=market_phase: 震荡/转换/趋势, n={n_cross})")
            for i in range(3):
                print(f"  {names[i]:<6} {' '.join(f'{cross[i,j]:>7}' for j in range(3))}")
