#!/usr/bin/env python3
"""A6c 道氏段生命周期: 存活曲线 + 段剧本 (2026-08-04, 无未来函数, 4h, 30 GBM 种子)

用户关注: 趋势一般运行多久会失效; 趋势内部不同状态的分布。

输出:
  L1 存活曲线: P(段长 ≥ k), k=1..60 — "第 k 根还活着的概率"
  L2 段剧本: 段内归一化位置 (20 桶) × 同向趋势态/transition 占比
     — "开头热结尾冷?" 真实 vs GBM
  L3 回撤位置: 回撤事件 (HL/LH) 在段内的相对位置分布

运行: python3 research/studies/a6c_dow_lifecycle.py
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
BUCKETS = 20


def collect(df, acc):
    n = len(df)
    d = dow_segments(df)
    states, _ = state_series(df)
    st_up = np.char.startswith(states, "trend_up")
    st_dn = np.char.startswith(states, "trend_down")
    st_tr = (states == "transition")
    seg_of = np.full(n, -1)
    for si, s in enumerate(d["segs"]):
        seg_of[s["start"]:s["end"] + 1] = si
    for s in d["segs"]:
        if s["direction"] not in ("up", "down"):
            continue
        acc["len"].append(s["bars"])
        nb = s["bars"]
        for t in range(s["start"], s["end"] + 1):
            b = min(BUCKETS - 1, int((t - s["start"]) * BUCKETS / nb))
            p = acc["play"][s["direction"]][b]
            p["n"] += 1
            p["tr"] += int(st_tr[t])
            if s["direction"] == "up":
                p["same"] += int(st_up[t])
                p["opp"] += int(st_dn[t])
            else:
                p["same"] += int(st_dn[t])
                p["opp"] += int(st_up[t])
    for r in d["retraces"]:
        si = seg_of[r["bar"]]
        if si < 0:
            continue
        s = d["segs"][si]
        pos = (r["bar"] - s["start"]) / max(1, s["bars"])
        acc["retr_pos"][r["direction"]].append(pos)


def report(acc, label):
    print(f"\n═══ {label} ═══")
    lens = np.array(acc["len"])
    print(f"L1 存活曲线 P(段长 ≥ k):")
    line = "  "
    for k in (5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100):
        line += f"k={k:<3}{np.mean(lens >= k):.0%}  "
    print(line)
    print("L2 段剧本 (段内位置 × 同向趋势态/transition 占比):")
    for dr in ("up", "down"):
        print(f"  {dr}:")
        row_same, row_tr = [], []
        for b in range(BUCKETS):
            p = acc["play"][dr][b]
            if p["n"] < 50:
                row_same.append(None)
                row_tr.append(None)
                continue
            row_same.append(p["same"] / p["n"])
            row_tr.append(p["tr"] / p["n"])
        for name, row in (("同向趋势", row_same), ("transition", row_tr)):
            pts = " ".join(f"{v:.0%}" if v is not None else "--" for v in row[::4])
            print(f"     {name:<8} [首{'':>2}..{pts}..尾]")
    print("L3 回撤事件在段内位置 (分位):")
    for dr in ("up", "down"):
        v = np.array(acc["retr_pos"][dr])
        if not len(v):
            continue
        print(f"  {dr}: n={len(v)} P10 {np.percentile(v, 10):.2f} 中位 "
              f"{np.median(v):.2f} P90 {np.percentile(v, 90):.2f}")


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    rl, gl = np.array(real["len"]), np.array(gbm["len"])
    print(f"L1 存活率: ", end="")
    for k in (15, 25, 40):
        print(f"k={k}: {np.mean(rl >= k):.0%} vs {np.mean(gl >= k):.0%} "
              f"(Δ {np.mean(rl >= k) - np.mean(gl >= k):+.0%})  ", end="")
    print()
    print("L2 段剧本 Δ(真实−GBM) 同向趋势占比 [首→尾]:")
    for dr in ("up", "down"):
        d1 = []
        for b in range(BUCKETS):
            rp, gp = real["play"][dr][b], gbm["play"][dr][b]
            if rp["n"] < 50 or gp["n"] < 50:
                d1.append(None)
            else:
                d1.append(rp["same"] / rp["n"] - gp["same"] / gp["n"])
        pts = " ".join(f"{v:+.0%}" if v is not None else "--" for v in d1[::4])
        print(f"  {dr}: {pts}")
    for dr in ("up", "down"):
        r, g = np.array(real["retr_pos"][dr]), np.array(gbm["retr_pos"][dr])
        if len(r) and len(g):
            print(f"L3 {dr} 回撤位置中位: 真实 {np.median(r):.2f} vs GBM "
                  f"{np.median(g):.2f} (Δ {np.median(r) - np.median(g):+.2f})")


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
    acc = dict(len=[], play={dr: [dict(n=0, tr=0, same=0, opp=0)
                                 for _ in range(BUCKETS)] for dr in ("up", "down")},
               retr_pos={dr: [] for dr in ("up", "down")})
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
    rw = [gbm_dataframe(len(ref), sig, seed=12000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)


if __name__ == "__main__":
    main()
