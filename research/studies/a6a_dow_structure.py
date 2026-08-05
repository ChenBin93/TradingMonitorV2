#!/usr/bin/env python3
"""A6a 道氏趋势结构统计描述 (2026-08-04, 无未来函数, 4h 优先)

用户视角 (道氏): 上升趋势 = HH+HL 持续抬升, 跌破前低 (最近 HL) 结束;
"上升趋势回撤然后继续上升" = 段内确认新 HL (回撤事件) 后创新高 (恢复)。

统计 (真实 vs GBM 30 种子, 4h):
  S1 段生命周期: 时间占比 / 段数 / 时长分布 / 幅度分布 (ATR) / HH·HL 计数
  S2 回撤事件: 深度分布 (ATR) / 时长分布 / 段内位置
  S3 恢复率 (预注册): HL 确认后 48 根内 high > 段内峰值 → 恢复;
     down 段对称 (48 根内 low < 段内谷值) — 真实 vs GBM

运行: python3 research/studies/a6a_dow_structure.py
"""
import os
import sys
import gc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.structures import dow_segments

TF = "4h"
N_GBM = 30
RECOVER_WIN = 48


def collect(df):
    hi = df["high"].values
    lo = df["low"].values
    n = len(df)
    res = dow_segments(df)
    segs = [s for s in res["segs"] if s["direction"] in ("up", "down")]
    retr = res["retraces"]
    states = np.full(n, "range", dtype=object)
    for s in res["segs"]:
        states[s["start"]:s["end"] + 1] = s["direction"]
    # S3 恢复率: 48 根内越过段内峰值/谷值
    recover = []
    for r in retr:
        t = r["bar"]
        if t + RECOVER_WIN >= n:
            continue
        if r["direction"] == "up":
            rec = float(hi[t + 1:t + RECOVER_WIN + 1].max() > r["peak_val"])
        else:
            rec = float(lo[t + 1:t + RECOVER_WIN + 1].min() < r["trough_val"])
        recover.append(rec)
    return dict(n=n, segs=segs, retr=retr, recover=recover, states=states)


def run_block(dfs, label=""):
    print(f"\n{'═' * 70}\nA6a {label} ({TF})\n{'═' * 70}")
    n_bars = 0
    time_frac = {"up": 0, "down": 0, "range": 0, "warmup": 0}
    seg_lens = {"up": [], "down": []}
    seg_amps = {"up": [], "down": []}
    hh_hl = {"up": [], "down": []}
    retr_depth = {"up": [], "down": []}
    retr_dur = {"up": [], "down": []}
    rec_all = []
    for df in dfs:
        d = collect(df)
        n_bars += d["n"]
        for v in d["states"]:
            time_frac[str(v)] += 1
        for s in d["segs"]:
            seg_lens[s["direction"]].append(s["bars"])
            seg_amps[s["direction"]].append(s["amp_atr"])
            hh_hl[s["direction"]].append(s["n_hh"] + s["n_hl"])
        for r in d["retr"]:
            retr_depth[r["direction"]].append(r["depth_atr"])
            retr_dur[r["direction"]].append(r["dur_bars"])
        rec_all.extend(d["recover"])
        gc.collect()
    print("S1 段生命周期:")
    tot = max(1, n_bars)
    frac = " | ".join(f"{k}:{v / tot:.1%}" for k, v in time_frac.items())
    print(f"  时间占比: {frac}")
    for dr in ("up", "down"):
        ll = np.array(seg_lens[dr])
        aa = np.array(seg_amps[dr])
        hh = np.array(hh_hl[dr])
        print(f"  {dr}: 段数 {len(ll)} | 时长 中位 {np.median(ll):.0f} 均值 {ll.mean():.1f} "
              f"P90 {np.percentile(ll, 90):.0f} 根 | 幅度 中位 {np.median(aa):.2f} ATR | "
              f"HH+HL 中位 {np.median(hh):.0f}")
    print("S2 回撤事件:")
    for dr in ("up", "down"):
        dd = np.array(retr_depth[dr])
        du = np.array(retr_dur[dr])
        if not len(dd):
            print(f"  {dr}: 无样本")
            continue
        print(f"  {dr}: n={len(dd)} | 深度 中位 {np.median(dd):.2f} 均值 {dd.mean():.2f} "
              f"P90 {np.percentile(dd, 90):.2f} ATR | 时长 中位 {np.median(du):.0f} "
              f"均值 {du.mean():.1f} 根")
    print("S3 恢复率 (48 根内越过段内峰值/谷值):")
    if len(rec_all):
        print(f"  真实 {np.mean(rec_all):.1%} (n={len(rec_all)})")
    return dict(seg_lens=seg_lens, seg_amps=seg_amps, retr_depth=retr_depth,
                retr_dur=retr_dur, rec=rec_all)


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    for dr in ("up", "down"):
        rl, gl = np.array(real["seg_lens"][dr]), np.array(gbm["seg_lens"][dr])
        ra, ga = np.array(real["seg_amps"][dr]), np.array(gbm["seg_amps"][dr])
        print(f"  {dr} 段: 时长中位 Δ {np.median(rl) - np.median(gl):+.1f} 根 | "
              f"幅度中位 Δ {np.median(ra) - np.median(ga):+.2f} ATR "
              f"(n {len(rl)}/{len(gl)})")
        rd, gd = np.array(real["retr_depth"][dr]), np.array(gbm["retr_depth"][dr])
        rdu, gdu = np.array(real["retr_dur"][dr]), np.array(gbm["retr_dur"][dr])
        if len(rd) and len(gd):
            print(f"    回撤深度 中位 Δ {np.median(rd) - np.median(gd):+.2f} ATR | "
                  f"时长 Δ {np.median(rdu) - np.median(gdu):+.1f} 根")
    rr, rg = np.array(real["rec"]), np.array(gbm["rec"])
    print(f"  S3 恢复率: 真实 {rr.mean():.1%} vs GBM {rg.mean():.1%} "
          f"→ Δ {rr.mean() - rg.mean():+.1%} (n {len(rr)}/{len(rg)})")


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


def main():
    dfs = _load(TF)
    real = run_block(dfs, "真实")
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=10000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)


if __name__ == "__main__":
    main()
