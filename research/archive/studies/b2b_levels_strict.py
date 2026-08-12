#!/usr/bin/env python3
"""B2b 关键位参数收紧 — 位定义与突破确认的敏感性扫描 (2026-08-03, 无未来函数)

收紧方向:
  位定义: min_touch ∈ {2,3,5} (形成时聚类成员数), tolerance ∈ {0.3, 0.5}×ATR
  突破:   depth ∈ {0.5, 0.8}×ATR (穿透深度), hold ∈ {0.5, 0.7} (外侧维持比例)

核心指标 (每组合): 位数 / 拒绝率 / 穿透确认率(假突破率) / E1 触碰前后 ATR 比
对照: 随机游走 (主组合: min_touch=3, tol=0.5, depth=0.8, hold=0.7)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.data_loader import load_candles, verify
from research.levels import cluster_levels, levels_touch_class_all
from research.sim_market import gbm_dataframe

W = 24


def _load():
    data = load_candles(timeframes=("1h",))
    out = []
    for sym, tfs in data.items():
        df = tfs.get("1h")
        if df is None:
            continue
        if verify(df, sym, "1h"):
            continue
        out.append(df)
    return out


def combo_stats(dfs, min_touch, tol, depth, hold):
    """聚合统计: 位数/触碰/拒绝/突破/穿透/确认 + E1 (E1 只算首个标的, 上限 20 万)"""
    n_lvls = n_touch = n_reject = n_break = n_attempt = n_conf = 0
    pre_vals, post_vals = [], []
    e1_pending = True
    for df in dfs:
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        atr = _atr_series(df)
        n = len(c)
        lvls = cluster_levels(h, l, atr, min_touch=min_touch, tolerance_mult=tol)
        ev = levels_touch_class_all(lvls, c, h, l, atr, depth, W, hold)
        n_lvls += len(lvls)
        n_touch += sum(len(x) for x in ev["touch"])
        n_reject += sum(len(x) for x in ev["reject"])
        n_break += sum(len(x) for x in ev["breakout"])
        n_attempt += sum(len(x) for x in ev["attempt"])
        n_conf += sum(len(x) for x in ev["confirmed"])
        if e1_pending:
            for idxs in ev["touch"]:
                for t in idxs:
                    if 12 <= t < n - 12 and atr[t] > 0:
                        pre_vals.append(atr[t - 12:t] / atr[t])
                        post_vals.append(atr[t + 1:t + 13] / atr[t])
                        if len(pre_vals) >= 200000:
                            break
                if len(pre_vals) >= 200000:
                    break
            if len(pre_vals) >= 200000:
                e1_pending = False
    e1 = None
    if pre_vals:
        pre = np.mean(np.stack(pre_vals), axis=0).mean()
        post = np.mean(np.stack(post_vals), axis=0).mean()
        e1 = post / pre
    return dict(lvls=n_lvls, touch=n_touch, reject=n_reject, brk=n_break,
                attempt=n_attempt, conf=n_conf, e1=e1)


def fmt_row(label, s):
    rej = s["reject"] / s["touch"] if s["touch"] else float("nan")
    conf = s["conf"] / s["attempt"] if s["attempt"] else float("nan")
    e1 = f"{s['e1']:.3f}" if s["e1"] else "-"
    print(f"  {label:>36}: 位 {s['lvls']:>6} | 触碰 {s['touch']:>8} | 拒绝率 {rej:>6.1%} "
          f"| 穿透 {s['attempt']:>8} 确认率 {conf:>6.1%} | E1 {e1}")


def main():
    dfs = _load()
    print("═══ B2b 参数收紧 (真实, 1h) ═══\n")
    combos = [
        ("收紧位(3,0.5)×严格突破(0.8,0.7)", 3, 0.5, 0.8, 0.7),
        ("收紧位(3,0.5)×基准突破(0.5,0.5)", 3, 0.5, 0.5, 0.5),
        ("基准位(2,0.3)×严格突破(0.8,0.7)", 2, 0.3, 0.8, 0.7),
        ("基准位(2,0.3)×基准突破(0.5,0.5)", 2, 0.3, 0.5, 0.5),
    ]
    for label, mt, tol, dp, hd in combos:
        s = combo_stats(dfs, mt, tol, dp, hd)
        fmt_row(label, s)
    print()
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=500 + k) for k in range(10)]
    print("── 随机游走对照 ──")
    for label, mt, tol, dp, hd in combos:
        s = combo_stats(rw, mt, tol, dp, hd)
        fmt_row("GBM " + label, s)
    print()


if __name__ == "__main__":
    main()
