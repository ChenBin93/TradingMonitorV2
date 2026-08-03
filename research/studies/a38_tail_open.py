#!/usr/bin/env python3
"""A3.8 尾部开放 — 1ATR 止损截断亏损, 收益端不限幅 (2026-08-03, 无未来函数)

用户思路: SL 固定 1ATR (保持截断亏损的有利结构), 收益用峰值回撤跟踪
或不设限的持有去"尽可能扩大"。

变体 (early 入场, 固定 1×ATR 止损):
  A. late 即退            (基准: A3.5 = +0.40R)
  B. 峰值回撤退出          trail ∈ {1, 2, 3, 5}×ATR, 无 late
  C. 无退出 (超长窗口 w=768, 纯止损+超时) — 收益端完全不设限

对照: GBM (真实 OHLC 子步, 6 序列) — 净效应 = 真实 − GBM; 分年稳定
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.caliber import MIN_N
from research.data_loader import load_candles, verify
from research.hold_sim import simulate_holds
from research.sim_market import gbm_dataframe
from research.structures import structural_states

WARMUP = 100


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


def event_entries(states, target):
    n = len(states)
    out = np.zeros(n, bool)
    for t in range(n):
        if states[t] == target and (t == 0 or states[t - 1] != target):
            out[t] = True
    return out


def run_combo(dfs, direction, tag, exit_late, w, peak_trail, years_needed=True):
    rs, yrs = [], []
    for df in dfs:
        st = structural_states(df)[WARMUP:]
        c = df["close"].values[WARMUP:]
        h = df["high"].values[WARMUP:]
        l = df["low"].values[WARMUP:]
        atr = _atr_series(df)[WARMUP:]
        entries = event_entries(st, tag)
        trades = simulate_holds(c, h, l, atr, st, entries, direction,
                                "atr", exit_late, w, sl_mult=1.0, peak_trail=peak_trail)
        yrs_df = df.index.year.values[WARMUP:]
        for t in trades:
            rs.append(t.r_mult)
            yrs.append(yrs_df[t.entry_idx])
    rs = np.array(rs)
    yrs = np.array(yrs)
    if len(rs) == 0:
        return None
    by_year = {}
    for y in sorted(set(yrs)):
        m = yrs == y
        if m.sum() >= 50:
            by_year[y] = rs[m].mean()
    return dict(n=len(rs), exp=rs.mean(), by_year=by_year)


def print_row(label, res, gbm):
    if res is None:
        print(f"  {label:>34}: 无持仓")
        return
    net = res["exp"] - gbm["exp"]
    yr = " ".join(f"{y}:{v:+.2f}" for y, v in sorted(res["by_year"].items()))
    flag = "" if res["n"] >= MIN_N else " ⚠样本不足"
    print(f"  {label:>34}: 真实 {res['exp']:+.3f}R (n={res['n']}) | GBM {gbm['exp']:+.3f}R "
          f"| 净 {net:+.3f}R {yr}{flag}")


def main():
    dfs = _load()
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=300 + k) for k in range(6)]
    print("═══ A3.8 尾部开放 (1h, early 入场, 固定 1×ATR 止损) ═══\n")

    for direction, tag in [("long", "up:early"), ("short", "down:early")]:
        print(f"── {direction} ({tag}) ──")
        # A. 基准: late 即退
        res = run_combo(dfs, direction, tag, True, 96, None)
        gbm = run_combo(rw, direction, tag, True, 96, None)
        print_row("A late即退 w=96", res, gbm)
        # B. 峰值回撤 (无 late)
        for tr in (1, 2, 3, 5):
            res = run_combo(dfs, direction, tag, False, 384, tr)
            gbm = run_combo(rw, direction, tag, False, 384, tr)
            print_row(f"B 峰值回撤 {tr}×ATR w=384", res, gbm)
        # C. 无退出 (超长窗口)
        res = run_combo(dfs, direction, tag, False, 768, None)
        gbm = run_combo(rw, direction, tag, False, 768, None)
        print_row("C 无退出 w=768", res, gbm)
        print()


if __name__ == "__main__":
    main()
