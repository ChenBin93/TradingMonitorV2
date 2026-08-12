#!/usr/bin/env python3
"""B5c 趋势叙事三环节检验 (2026-08-04, 无未来函数, 1h 基准参数, 30 GBM 种子)

用户叙事模型: ①趋势早期突破关键位启动 → ②运行中触位折返 (短期回撤) →
③再次回到趋势方向。

检验:
  H1 突破启动: 确认突破事件在 trend 段起点 ±12 根内的密度 vs 全体
     (密度比 = 窗口内占比 / 窗口时间占比; >1 = 突破聚集在段起点)
  H3 回撤恢复: D1 沿趋势方向概率的窗口梯度 W ∈ {6,12,24,48,96} —
     短期 (≤24) 逆势、长期 (48~96) 转正/归零 → 折返=暂时回撤, 趋势恢复
  H4 角色转换: 触碰位带按"过去 60 根内是否被确认突破"分层 (刚破/未破)
     — 涨势触刚突破的支撑 (回踩位) → 反弹? vs 未破位 → 继续逆势?
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
from research.studies.b3_general_levels import (collect as b3collect, MIN_N)

MT, TOL = 2, 0.3
TF = "1h"
N_GBM = 30
WS = (6, 12, 24, 48, 96)
ROLE_WIN = 60
H1_WIN = 12


def process_one(df, acc):
    d = b3collect(df, MT, TOL)
    c, h, l, atr, n = d["c"], d["h"], d["l"], d["atr"], d["n"]
    states, _ = state_series(df)
    up_all = np.char.startswith(states, "trend_up")
    dn_all = np.char.startswith(states, "trend_down")
    # H1: trend 段起点
    starts = [t for t in range(n) if (up_all[t] or dn_all[t])
              and (t == 0 or not (up_all[t - 1] or dn_all[t - 1]))]
    starts = np.array(starts)
    if len(starts):
        all_conf = np.concatenate([d["ev"]["confirmed"][i] for i in range(len(d["lvls"]))
                                   if len(d["ev"]["confirmed"][i])])
        if len(all_conf):
            lo = np.searchsorted(all_conf, starts - H1_WIN, "left")
            hi = np.searchsorted(all_conf, starts + H1_WIN, "right")
            in_win = int((hi - lo).sum())
            win_frac = len(starts) * (2 * H1_WIN + 1) / max(1, n)
            conf_frac = in_win / len(all_conf)
            acc["h1_ratio"].append(conf_frac / max(1e-9, win_frac))
            acc["h1_n"].append(len(all_conf))
    # H3/H4: 触碰事件 (端点方向, 轻量)
    for i, lv in enumerate(d["lvls"]):
        t_arr = d["ev"]["touch"][i]
        valid = (t_arr + 96 < n) & (t_arr >= 14) & (atr[t_arr] > 0)
        t_arr = t_arr[valid]
        if not len(t_arr):
            continue
        st = states[t_arr]
        up = np.char.startswith(st, "trend_up")
        dn = np.char.startswith(st, "trend_down")
        c_t = c[t_arr]
        for W in WS:
            d1 = np.where(up, c[t_arr + W] > c_t, np.where(dn, c[t_arr + W] < c_t,
                                                          c[t_arr + W] > c_t))
            for m, dr in ((up, "up"), (dn, "dn")):
                if m.any():
                    acc[f"w{W}_{dr}_{lv.side}"].extend(d1[m].tolist())
        # H4: 角色转换 (W=24)
        conf = d["ev"]["confirmed"][i]
        if len(conf):
            lo = np.searchsorted(conf, t_arr - ROLE_WIN, "left")
            hi = np.searchsorted(conf, t_arr - 1, "right")
            has_break = (hi > lo)
            d1 = np.where(up, c[t_arr + 24] > c_t, np.where(dn, c[t_arr + 24] < c_t,
                                                            c[t_arr + 24] > c_t))
            for m, dr in ((up, "up"), (dn, "dn")):
                if not m.any():
                    continue
                idx = np.flatnonzero(m)
                for j in idx:
                    role = "刚破" if has_break[j] else "未破"
                    acc[f"role_{dr}_{role}_{lv.side}"].append(float(d1[j]))


def report(acc, label):
    print(f"\n═══ {label} ═══")
    if acc["h1_n"]:
        w = np.array(acc["h1_ratio"])
        print(f"H1 突破启动: 段起点±{H1_WIN}根 confirmed 密度比 "
              f"{np.mean(w):.2f} (n confirmed {int(np.sum(acc['h1_n']))})  (>1=聚集于段起点)")
    print("H3 窗口梯度 D1 沿趋势方向概率:")
    print(f"{'W':>4} {'up_res':>9} {'up_sup':>9} {'dn_res':>9} {'dn_sup':>9}")
    for W in WS:
        row = []
        for dr, side in (("up", "resistance"), ("up", "support"),
                         ("dn", "resistance"), ("dn", "support")):
            v = acc.get(f"w{W}_{dr}_{side}")
            row.append(f"{np.mean(v):.1%}" if v and len(v) >= MIN_N else "-")
        print(f"{W:>4} " + " ".join(f"{x:>9}" for x in row))
    print("H4 角色转换 D1 (过去60根内刚破 vs 未破):")
    for k in sorted(acc):
        if not k.startswith("role_"):
            continue
        v = acc[k]
        print(f"  {k:<24} n={len(v):>7} D1={np.mean(v):.1%}" if len(v) >= MIN_N
              else f"  {k:<24} n={len(v)} 不足")


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    ra, ga = real, gbm
    if ra["h1_n"] and ga["h1_n"]:
        print(f"H1 密度比: 真实 {np.mean(ra['h1_ratio']):.2f} vs GBM "
              f"{np.mean(ga['h1_ratio']):.2f}")
    print("H3 ΔD1 窗口梯度:")
    print(f"{'W':>4} {'up_res':>9} {'up_sup':>9} {'dn_res':>9} {'dn_sup':>9}")
    for W in WS:
        row = []
        for dr, side in (("up", "resistance"), ("up", "support"),
                         ("dn", "resistance"), ("dn", "support")):
            r = ra.get(f"w{W}_{dr}_{side}")
            g = ga.get(f"w{W}_{dr}_{side}")
            if r and g and len(r) >= MIN_N and len(g) >= MIN_N:
                row.append(f"{np.mean(r) - np.mean(g):+.1%}")
            else:
                row.append("-")
        print(f"{W:>4} " + " ".join(f"{x:>9}" for x in row))
    print("H4 ΔD1 (真实−GBM):")
    for k in sorted(ra):
        if not k.startswith("role_"):
            continue
        r, g = ra[k], ga.get(k)
        if g is None or len(r) < MIN_N or len(g) < MIN_N:
            continue
        print(f"  {k:<24} Δ {np.mean(r) - np.mean(g):+.1%} (n {len(r)}/{len(g)})")


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
    for df in dfs:
        process_one(df, acc)
        gc.collect()
    report(acc, label)
    return acc


def main():
    dfs = _load(TF)
    real = run_block(dfs, "真实")
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=9000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)


if __name__ == "__main__":
    main()
