#!/usr/bin/env python3
"""B5b 阶段混杂检验 (2026-08-04, 无未来函数, 1h 基准参数, 30 GBM 种子)

用户假设: 涨势触支撑 = 趋势末期反转的伴随现象 — 触碰事件系统性偏 late
阶段 (A2 的 late 判定含 dev, 回调触支撑 = dev 小/负 → 偏 late),
"逆势折返" (B5: 触位后沿趋势方向概率 46~47%) 可能不是关键位独立效应。

检验:
  S1 选择效应: 触碰时刻阶段分布 vs 全体 bar 阶段分布 (真实 vs GBM)
  S2 阶段×方向×侧 D1 矩阵 (真实−GBM 净差) — 逆势折返是否集中在 late
  S3 控制阶段净差 = S2 每阶段单元格的 真实−GBM — 关键位独立效应检验
  S4 段内位置: 触碰时刻距 trend 段起点的根数 (触支撑 vs 触阻力, 真实 vs
     GBM) — 后期偏置直接度量
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
from research.studies.b3_general_levels import (collect as b3collect, MIN_N, W)

MT, TOL = 2, 0.3
TF = "1h"
N_GBM = 30
W24 = 24


def stage_of(s):
    """states 元素 → (方向, 阶段)  例: trend_up:late → (up, late)"""
    if s.startswith("trend_up"):
        return "up", s.split(":")[1] if ":" in s else "accelerate"
    if s.startswith("trend_down"):
        return "dn", s.split(":")[1] if ":" in s else "accelerate"
    return "neu", None


def process_one(df, acc, dist_acc):
    d = b3collect(df, MT, TOL)
    c, h, l, atr, n = d["c"], d["h"], d["l"], d["atr"], d["n"]
    states, _ = state_series(df)
    up_all = np.char.startswith(states, "trend_up")
    dn_all = np.char.startswith(states, "trend_down")
    # S4: trend_up 段起点 (逐 bar 扫描, O(n))
    seg_start = np.full(n, -1)
    cur = -1
    for t in range(n):
        if up_all[t]:
            if cur < 0:
                cur = t
            seg_start[t] = cur
        else:
            cur = -1
    # S1: 全体阶段分布
    for mask, dr in ((up_all, "up"), (dn_all, "dn")):
        for s in states[mask]:
            stg = s.split(":")[1] if ":" in s else "accelerate"
            acc["full_dist"].append((dr, stg))
    for i, lv in enumerate(d["lvls"]):
        t_arr = d["ev"]["touch"][i]
        valid = (t_arr + W24 < n) & (t_arr >= 14) & (atr[t_arr] > 0)
        t_arr = t_arr[valid]
        if not len(t_arr):
            continue
        seg = c[t_arr[:, None] + np.arange(1, W24 + 1)]
        st = states[t_arr]
        up = np.char.startswith(st, "trend_up")
        dn = np.char.startswith(st, "trend_down")
        neu = ~(up | dn)
        c_t = c[t_arr]
        hi, lo = seg.max(axis=1), seg.min(axis=1)
        d1 = np.where(up, seg[:, -1] > c_t, np.where(dn, seg[:, -1] < c_t, seg[:, -1] > c_t))
        keys = np.where(up, "up", np.where(dn, "dn", "neu"))
        for m, dr in ((up, "up"), (dn, "dn"), (neu, "neu")):
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            for j in idx:
                s = st[j]
                stg = s.split(":")[1] if ":" in s else "accelerate"
                k = f"{dr}_{stg if dr != 'neu' else 'x'}_{lv.side}"
                acc[k].append((float(d1[j]),))
                acc["touch_dist"].append((dr, stg, lv.side))
            if dr == "up":
                ss = seg_start[t_arr[m]]
                dist_from_start = t_arr[m] - ss
                for j, dd in enumerate(dist_from_start):
                    dist_acc[f"up_{lv.side}"].append(float(dd))


def report(acc, dist_acc, label):
    print(f"\n═══ {label} ═══")
    # S1 阶段分布 (触碰 vs 全体)
    full = defaultdict(int)
    for dr, stg in acc["full_dist"]:
        full[f"{dr}_{stg}"] += 1
    touch = defaultdict(int)
    for dr, stg, side in acc["touch_dist"]:
        touch[f"{dr}_{stg}"] += 1
    nf = sum(full.values())
    nt = sum(touch.values())
    print("S1 阶段占比 (触碰 vs 全体 bar):")
    for k in sorted(full):
        f, t = full[k] / nf, touch.get(k, 0) / max(1, nt)
        print(f"  {k:<16} 全体 {f:>6.1%} | 触碰 {t:>6.1%} (触碰占比-全体: {t - f:+.1%})")
    # S2/S3 D1 矩阵
    print("S2/S3 D1 沿趋势方向概率 (阶段 × 方向 × 侧):")
    for k in sorted(acc):
        if k in ("full_dist", "touch_dist"):
            continue
        rows = acc[k]
        if len(rows) < MIN_N:
            print(f"  {k:<22} n={len(rows)} 不足")
            continue
        print(f"  {k:<22} n={len(rows):>7} D1={np.mean([x[0] for x in rows]):.1%}")
    # S4
    print("S4 触支撑 vs 触阻力 距 trend 段起点根数 (中位):")
    for k in sorted(dist_acc):
        v = np.array(dist_acc[k])
        if len(v) < MIN_N:
            print(f"  {k}: n={len(v)} 不足")
            continue
        print(f"  {k}: n={len(v):>7} 中位 {np.median(v):.1f} 根")


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    ra, rd = real
    ga, gd = gbm
    print(f"{'key':<22}{'n_真':>8}{'n_G':>8} {'ΔD1':>8}")
    for k in sorted(ra):
        if k in ("full_dist", "touch_dist"):
            continue
        r, g = ra[k], ga[k]
        if len(r) < MIN_N or len(g) < MIN_N:
            continue
        d = np.mean([x[0] for x in r]) - np.mean([x[0] for x in g])
        print(f"{k:<22}{len(r):>8}{len(g):>8} {d:>+8.1%}")
    print(f"\nS4 段内位置中位 (真实 vs GBM):")
    for k in sorted(rd):
        r, g = np.array(rd[k]), gd.get(k, [])
        if len(r) < MIN_N or len(g) < MIN_N:
            print(f"  {k}: n {len(r)}/{len(g)} 不足")
            continue
        print(f"  {k}: 真实 {np.median(r):.1f} vs GBM {np.median(g):.1f} 根 (Δ {np.median(r)-np.median(g):+.1f})")


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
    dist_acc = defaultdict(list)
    for df in dfs:
        process_one(df, acc, dist_acc)
        gc.collect()
    report(acc, dist_acc, label)
    return acc, dist_acc


def main():
    dfs = _load(TF)
    real = run_block(dfs, "真实")
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=8000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)


if __name__ == "__main__":
    main()
