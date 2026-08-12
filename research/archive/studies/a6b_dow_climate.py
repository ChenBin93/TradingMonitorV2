#!/usr/bin/env python3
"""A6b 道氏段气候分解 (2026-08-04, 无未来函数, 4h, 30 GBM 种子)

用户理解: 统计状态 = 每根 K 线的局部气候 (波动/MA/动量); 道氏段 = 持续
的天气系统, 由许多小局部气候构成 (段内统计趋势态占比平均仅 39%)。

拆解 (真实 vs GBM):
  S1 段内气候构成: trend/transition/range 占比分布, 按段长分层
  S2 气候 × 段命运 (预注册):
     H1 段内 trend 占比高 → 段更长/幅度更大?
     H2 初始气候 (前 5 根 trend 占比) → 段寿命/幅度?
  S3 恢复率 × 气候: 回撤事件按所在段 trend 占比分层 × 恢复率
     (HL 后 48 根内越段内峰值) — 热段回撤恢复率是否更高

运行: python3 research/studies/a6b_dow_climate.py
"""
import os
import sys
import gc
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.state_features import state_series
from research.structures import dow_segments

TF = "4h"
N_GBM = 30
RECOVER_WIN = 48
HEAT_BINS = ("冷(<20%)", "中(20-60%)", "热(>60%)")


def heat_bin(v):
    return 0 if v < 0.20 else (1 if v < 0.60 else 2)


def collect(df, acc):
    d = dow_segments(df)
    states, _ = state_series(df)
    n = len(df)
    st_up = np.char.startswith(states, "trend_up")
    st_dn = np.char.startswith(states, "trend_down")
    st_tr = (states == "transition")
    hi = df["high"].values
    lo = df["low"].values
    # 回撤事件 → 所在段 (逐段扫描, 段数少)
    seg_of = np.full(n, -1)
    for si, s in enumerate(d["segs"]):
        seg_of[s["start"]:s["end"] + 1] = si
    for r in d["retraces"]:
        si = seg_of[r["bar"]]
        if si < 0:
            continue
        s = d["segs"][si]
        seg = slice(s["start"], s["end"] + 1)
        heat = (st_up[seg].sum() + st_dn[seg].sum()) / max(1, s["bars"])
        acc["retr_heat"][heat_bin(heat)].append(r)
        if r["bar"] + RECOVER_WIN < n:
            if r["direction"] == "up":
                rec = float(hi[r["bar"] + 1:r["bar"] + RECOVER_WIN + 1].max() > r["peak_val"])
            else:
                rec = float(lo[r["bar"] + 1:r["bar"] + RECOVER_WIN + 1].min() < r["trough_val"])
            acc["retr_rec"][heat_bin(heat)].append(rec)
    # 段统计
    for s in d["segs"]:
        if s["direction"] not in ("up", "down"):
            continue
        seg = slice(s["start"], s["end"] + 1)
        bars = s["bars"]
        trend_frac = (st_up[seg].sum() + st_dn[seg].sum()) / bars
        trans_frac = st_tr[seg].sum() / bars
        init5 = (st_up[s["start"]:s["start"] + 5].sum()
                 + st_dn[s["start"]:s["start"] + 5].sum()) / min(5, bars)
        acc["seg"].append(dict(bars=bars, amp=s["amp_atr"], heat=trend_frac,
                               trans=trans_frac, init=init5,
                               heat_bin=heat_bin(trend_frac)))


def report(acc, label):
    print(f"\n═══ {label} ═══")
    segs = acc["seg"]
    n_seg = len(segs)
    if not n_seg:
        print("  无段")
        return
    bars = np.array([x["bars"] for x in segs])
    heat = np.array([x["heat"] for x in segs])
    trans = np.array([x["trans"] for x in segs])
    amp = np.array([x["amp"] for x in segs])
    print(f"S1 段内气候构成 (n={n_seg}):")
    print(f"  trend 占比: 中位 {np.median(heat):.0%} 均值 {heat.mean():.0%} "
          f"P25 {np.percentile(heat, 25):.0%} P75 {np.percentile(heat, 75):.0%}")
    print(f"  transition 占比: 中位 {np.median(trans):.0%} 均值 {trans.mean():.0%}")
    for lo_b, hi_b in ((1, 15), (15, 40), (40, 10 ** 9)):
        m = (bars >= lo_b) & (bars < hi_b)
        if m.sum() < 20:
            continue
        print(f"  段长 {lo_b}-{hi_b if hi_b < 10 ** 9 else '+'}: n={m.sum()} "
              f"trend占比中位 {np.median(heat[m]):.0%}")
    print("S2 气候 × 段命运:")
    for b in range(3):
        m = [x["heat_bin"] == b for x in segs]
        m = np.array(m)
        if m.sum() < 20:
            continue
        print(f"  {HEAT_BINS[b]:<12} n={m.sum():>5} 段长中位 {np.median(bars[m]):>4.0f} 根 "
              f"幅度中位 {np.median(amp[m]):>5.2f} ATR")
    # 初始气候
    inits = np.array([x["init"] for x in segs])
    for b in range(3):
        m = np.array([heat_bin(x) == b for x in inits])
        if m.sum() < 20:
            continue
        print(f"  初始气候{HEAT_BINS[b]:<10} n={m.sum():>5} 段长中位 {np.median(bars[m]):>4.0f} 根 "
              f"幅度中位 {np.median(amp[m]):>5.2f} ATR")
    print("S3 恢复率 × 段气候:")
    for b in range(3):
        rec = acc["retr_rec"].get(b, [])
        rr = acc["retr_heat"].get(b, [])
        if len(rec) < 50:
            print(f"  {HEAT_BINS[b]:<12} n={len(rr)} 恢复率样本不足")
            continue
        print(f"  {HEAT_BINS[b]:<12} 回撤 n={len(rr)} 恢复率 {np.mean(rec):.1%}")


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    for b in range(3):
        r = np.array([x["bars"] for x in real["seg"] if x["heat_bin"] == b])
        g = np.array([x["bars"] for x in gbm["seg"] if x["heat_bin"] == b])
        rh = np.array([x["heat"] for x in real["seg"] if x["heat_bin"] == b])
        gh = np.array([x["heat"] for x in gbm["seg"] if x["heat_bin"] == b])
        if len(r) < 20 or len(g) < 20:
            continue
        print(f"  {HEAT_BINS[b]}: 段长中位 {np.median(r):.0f} vs {np.median(g):.0f} 根 "
              f"(Δ {np.median(r) - np.median(g):+.0f}) | trend占比中位 "
              f"{np.median(rh):.0%} vs {np.median(gh):.0%}")
    for b in range(3):
        r = real["retr_rec"].get(b, [])
        g = gbm["retr_rec"].get(b, [])
        if len(r) < 50 or len(g) < 50:
            continue
        print(f"  S3 {HEAT_BINS[b]}: 恢复率 真实 {np.mean(r):.1%} vs GBM {np.mean(g):.1%} "
              f"→ Δ {np.mean(r) - np.mean(g):+.1%} (n {len(r)}/{len(g)})")


def _load(tf):
    data = load_candles(timeframes=(tf,))
    out = []
    for sym, tfs in data.items():
        df = tfs.get(tf)
        if df is None:
            continue
        if verify(df, sym, tf):
            continue
        out.append(df)
    return out


def run_block(dfs, label=""):
    acc = defaultdict(list)
    acc["retr_heat"] = defaultdict(list)
    acc["retr_rec"] = defaultdict(list)
    for df in dfs:
        collect(df, acc)
        gc.collect()
    report(acc, label)
    return acc


def main():
    dfs = _load(TF)
    real = run_block(dfs, "真实")
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=11000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)


if __name__ == "__main__":
    main()
