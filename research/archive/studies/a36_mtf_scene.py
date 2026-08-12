#!/usr/bin/env python3
"""A3.6 多周期状态组合 — 日线 bias × 4H 状态的 1:1 方向胜率 (2026-08-03, 无未来函数)

重验旧场景引擎核心假设 (LIVE_SYSTEM.md / market_state.py):
  插曲 (顺日逆时) 55-61% | 顺势 53-57% | 逆势 45-47%
  — 旧结论来自已作废研究 (未来函数), 必须无未来函数重验

设计 (无未来函数):
  - 日线: 4h 重采样, bar 开盘=当日00:00, 收盘=次日00:00
    → 1h bar t 只能用 open+24h <= t 的日线 bar (align_higher)
  - 4H: 已收盘 4h bar (align_higher: open+4h <= t)
  - bias/t4 = sign(close - ma20), 0=中性
  - 场景 = bias×t4 四象限: episode_long(b+1,t-1) 顺日逆时 / follow_* 顺势
  - 事件式 (场景进入第一根 bar 收盘), 1:1 T×ATR
  - 方向模式: with=顺日线方向 (插曲/顺势检验), against=逆日线 (逆势检验)
  - 分年 + Wilson CI + MIN_N + 随机游走对照
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.caliber import MIN_N, T, W
from research.data_loader import align_higher, daily_resample, load_candles, verify
from research.outcome import Outcome, evaluate_forward, report_wr, wilson_ci
from research.sim_market import gbm_dataframe

WARMUP = 60


def _load_pairs():
    data = load_candles(timeframes=("1h", "4h"))
    out = []
    for sym, tfs in data.items():
        if "1h" not in tfs or "4h" not in tfs:
            continue
        if verify(tfs["1h"], sym, "1h") or verify(tfs["4h"], sym, "4h"):
            continue
        out.append((tfs["1h"], tfs["4h"]))
    return out


def scene_series(df1h, df4h):
    """逐 1h bar: (scene, bias) — 只用已收盘日线/4H"""
    daily = daily_resample(df4h)
    daily["ma20"] = daily["close"].rolling(20).mean()
    daily = daily.dropna(subset=["ma20"])
    d4 = df4h.copy()
    d4["ma20"] = d4["close"].rolling(20).mean()
    idx = df1h.index
    dc = align_higher(daily[["close", "ma20"]], "1d", idx)
    bias = np.sign((dc["close"] - dc["ma20"]).values)
    t4c = align_higher(d4[["close", "ma20"]], "4h", idx)
    t4 = np.sign((t4c["close"] - t4c["ma20"]).values)
    n = len(idx)
    scene = np.array(["neutral"] * n, dtype=object)
    for i in range(n):
        b, t = bias[i], t4[i]
        if not (np.isfinite(b) and np.isfinite(t)) or b == 0 or t == 0:
            scene[i] = "neutral"
        elif b == 1 and t == -1:
            scene[i] = "episode_long"
        elif b == -1 and t == 1:
            scene[i] = "episode_short"
        elif b == 1 and t == 1:
            scene[i] = "follow_long"
        else:
            scene[i] = "follow_short"
    return scene, bias


def agg_outcomes(dfs, scene_target, mode):
    """场景进入事件 × 方向模式
    with:    顺日线方向 (bias>0 做多 / bias<0 做空) — 插曲/顺势检验
    against: 逆日线方向 (bias>0 做空 / bias<0 做多) — 逆势检验
    返回 (long_outcome, short_outcome, long_year, short_year) — 两方向分别统计
    """
    agg = {d: {"nw": 0, "nl": 0, "ne": 0, "ns": 0, "y": {}} for d in ("long", "short")}
    for df1h, df4h in dfs:
        c = df1h["close"].values
        h = df1h["high"].values
        l = df1h["low"].values
        o = df1h["open"].values
        atr = _atr_series(df1h)
        years = df1h.index.year.values
        scene, bias = scene_series(df1h, df4h)
        n = len(c)
        entries = {"long": np.zeros(n, bool), "short": np.zeros(n, bool)}
        for i in range(WARMUP, n):
            if scene[i] != scene_target:
                continue
            if scene[i] == scene[i - 1]:
                continue
            b = bias[i]
            if mode == "with":
                d = "long" if b > 0 else "short"
            else:
                d = "long" if b < 0 else "short"
            if np.isfinite(b) and b != 0:
                entries[d][i] = True
        for d in ("long", "short"):
            out, recs = evaluate_forward(c, h, l, atr, entries[d], d, T, W, open_px=o)
            a = agg[d]
            a["nw"] += out.n_win
            a["nl"] += out.n_loss
            a["ne"] += out.n_expired
            a["ns"] += out.n_skip
            for r in recs:
                if r.outcome not in ("win", "loss"):
                    continue
                y = years[r.entry_idx]
                wl = a["y"].setdefault(y, [0, 0])
                wl[0 if r.outcome == "win" else 1] += 1
    res = {}
    for d in ("long", "short"):
        a = agg[d]
        o = Outcome(a["nw"], a["nl"], a["ne"], a["ns"])
        by_year = {y: wl[0] / (wl[0] + wl[1]) for y, wl in a["y"].items() if wl[0] + wl[1] >= 100}
        res[d] = (o, by_year)
    return res


def print_row(label, o, base, by_year):
    ci = wilson_ci(o.n_win, o.n_eval)
    flag = "" if o.n_eval >= MIN_N else " ⚠样本不足"
    yr = " ".join(f"{y}:{v:.0%}" for y, v in sorted(by_year.items()))
    print(f"  {label:>26}: n={o.n_eval:>6} WR {o.win_rate:>6.1%} vs 基线 {base.win_rate:>5.1%} "
          f"Δ{o.win_rate - base.win_rate:>+6.1%}pp [CI {ci[0]:.1%}-{ci[1]:.1%}] {yr}{flag}")


def run_block(dfs, label=""):
    print(f"═══ 多周期状态组合 {label}═══\n")
    base_long = base_short = None
    for df1h, df4h in dfs:
        c = df1h["close"].values
        h = df1h["high"].values
        l = df1h["low"].values
        o = df1h["open"].values
        atr = _atr_series(df1h)
        one = np.ones(len(c), bool)
        o1, _ = evaluate_forward(c, h, l, atr, one, "long", T, W, open_px=o)
        o2, _ = evaluate_forward(c, h, l, atr, one, "short", T, W, open_px=o)
        if base_long is None:
            base_long, base_short = o1, o2
        else:
            base_long.n_win += o1.n_win
            base_long.n_loss += o1.n_loss
            base_short.n_win += o2.n_win
            base_short.n_loss += o2.n_loss
    print(f"  无条件基线 做多: {report_wr(base_long)}")
    print(f"  无条件基线 做空: {report_wr(base_short)}\n")

    all_scene = np.concatenate([scene_series(a, b)[0] for a, b in dfs])
    print("  场景占比:")
    for s in sorted(set(all_scene)):
        print(f"    {s:>16}: {np.mean(all_scene == s):.1%}")
    print()

    print("Q 顺日线方向 (with): 插曲/顺势 场景进入后顺 bias 做\n")
    for s in ["episode_long", "episode_short", "follow_long", "follow_short"]:
        res = agg_outcomes(dfs, s, "with")
        for d in ("long", "short"):
            o, yr = res[d]
            base = base_long if d == "long" else base_short
            print_row(f"{s} 顺日线×{d}", o, base, yr)
    print()

    print("Q 逆日线方向 (against): 逆势检验 (旧结论 45-47%)\n")
    for s in ["episode_long", "episode_short", "follow_long", "follow_short"]:
        res = agg_outcomes(dfs, s, "against")
        for d in ("long", "short"):
            o, yr = res[d]
            base = base_long if d == "long" else base_short
            print_row(f"{s} 逆日线×{d}", o, base, yr)
    print()


def main():
    pairs = _load_pairs()
    run_block(pairs)
    # 随机游走对照: GBM 1h + 4h 重采样
    ref = pairs[0][0]
    n_ref = len(ref)
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw_pairs = []
    for k in range(20):
        df1 = gbm_dataframe(n_ref, sig, seed=100 + k)
        df4 = df1.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                      "close": "last", "volume": "sum"}).dropna(subset=["close"])
        rw_pairs.append((df1, df4))
    run_block(rw_pairs, label="(随机游走)")


if __name__ == "__main__":
    main()
