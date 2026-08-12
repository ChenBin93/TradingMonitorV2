#!/usr/bin/env python3
"""A3.7 尾部利用 — 宽止损/长持有能否吃到真实市场的大K尾部 (2026-08-03, 无未来函数)

背景: A4 发现真实趋势收益分布偏度 (+2.49 up:late) 远大于随机游走 (+0.11),
但 A3.5 的 1ATR 止损 + 96根窗口下真实期望 ≈ GBM — 大K尾部被截断。
本实验: 扫描 止损距离(ATR×sl_mult) × 持有窗口(w), 看真实 − GBM 净效应
是否随"尾部空间"放大 — 若放大, 真实市场白送的大K被吃到了。

设计 (事件式, 无未来函数):
  入场: 结构状态 up:early (long) / down:early (short)
  止损: 固定 ATR×sl_mult (sl_mult=1/3/5 — 宽止损保留尾部)
  退出: late 状态即退 (趋势段保留到末期) / timeout (w 根)
  窗口: w=96/384 (更长窗口给尾部时间)
  对照: GBM (真实 OHLC 子步, 6 序列) — 净效应 = 真实 − GBM

附加 Q: 趋势状态内 top 5% K 的收益贡献占比 (尾部发动机量化, 真实 vs GBM)
"""
import os
import sys

import numpy as np
import pandas as pd

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


def run_combo(dfs, sl_mult, w, direction, tag):
    """期望 R 聚合 (含 timeout, 计入期望)"""
    rs = []
    for df in dfs:
        st = structural_states(df)[WARMUP:]
        c = df["close"].values[WARMUP:]
        h = df["high"].values[WARMUP:]
        l = df["low"].values[WARMUP:]
        atr = _atr_series(df)[WARMUP:]
        entries = event_entries(st, tag)
        trades = simulate_holds(c, h, l, atr, st, entries, direction,
                                "atr", True, w, sl_mult=sl_mult)
        for t in trades:
            rs.append(t.r_mult)
    rs = np.array(rs)
    if len(rs) == 0:
        return None
    return dict(n=len(rs), exp=rs.mean(),
                wr=float(np.mean(rs > 0)),
                win=np.mean(rs[rs > 0]) if (rs > 0).any() else 0.0,
                loss=np.mean(rs[rs <= 0]) if (rs <= 0).any() else 0.0)


def print_row(label, res, gbm):
    if res is None:
        print(f"  {label:>30}: 无持仓")
        return
    net = res["exp"] - gbm["exp"]
    flag = "" if res["n"] >= MIN_N else " ⚠样本不足"
    print(f"  {label:>30}: 真实 {res['exp']:+.3f}R (n={res['n']}) | GBM {gbm['exp']:+.3f}R "
          f"| 净 {net:+.3f}R{flag}")


def tail_contribution(dfs):
    """趋势状态内 top 5% K 的收益贡献占比 (绝对值收益排序)"""
    tot, top = [], []
    for df in dfs:
        st, _ = __import__("research.state_features", fromlist=["state_series"]).state_series(df)
        c = df["close"].values
        ret = np.zeros(len(c))
        ret[1:] = np.diff(c) / c[:-1]
        m = np.array([s.startswith("trend") for s in st])
        if m.sum() < 100:
            continue
        r = ret[m]
        sgn = np.where(np.array([s.startswith("trend_up") for s in st])[m], 1, -1)
        sr = r * sgn  # 顺趋势方向收益
        tot.append(sr.sum())
        k = max(1, int(len(sr) * 0.05))
        top.append(np.sort(sr)[-k:].sum())
    return np.mean(top) / np.mean(tot) if np.mean(tot) != 0 else np.nan


def main():
    dfs = _load()
    print("═══ A3.7 尾部利用 (1h, early 入场, 固定 ATR×sl_mult 止损, late 即退) ═══\n")

    # 附加: top 5% K 贡献
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=200 + k) for k in range(6)]
    t_real = tail_contribution(dfs)
    t_gbm = tail_contribution(rw)
    print(f"Q 趋势状态内 top 5% K 的顺趋势收益贡献: 真实 {t_real:.0%} vs GBM {t_gbm:.0%}\n")

    for direction, tag in [("long", "up:early"), ("short", "down:early")]:
        print(f"── {direction} ({tag}) ──")
        for sl_mult in (1, 3, 5):
            for w in (96, 384):
                res = run_combo(dfs, sl_mult, w, direction, tag)
                gbm = run_combo(rw, sl_mult, w, direction, tag)
                print_row(f"sl={sl_mult}×ATR w={w}", res, gbm)
        print()


if __name__ == "__main__":
    main()
