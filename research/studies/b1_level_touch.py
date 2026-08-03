#!/usr/bin/env python3
"""B1 关键水平位触碰行为研究 (2026-08-03, 无未来函数)

问题 (用户经验: 波段高低点 + 多次触碰的水平位最重要):
  Q1 触碰关键位后, 反弹方向 1:1 胜率 vs 无条件基线 (支撑触碰做多 / 阻力触碰做空)
  Q2 收盘确认破位后, 破位方向 1:1 胜率 vs 基线 (破位延续 vs 回拉)
  Q3 支撑 vs 阻力 不对称性 (Q1 分 side 对比)
  Q4 随机游走对照: GBM 跑同一检测器 — 关键位在随机游走上天然存在,
     "触碰反弹"可能是条件极值的统计假象, 必须校准 (B 方向最大的坑)

口径: 事件式 (触碰进入 = 本根触及位带且前一根未触及); 入场 = 事件 bar 收盘;
  1:1 T×ATR (outcome 引擎); 分年 + Wilson CI + MIN_N
S/R 回看: 位形成距当前 ≤ 120/300/600 根 三档 (价位记忆长于状态, 规格例外条款)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.caliber import MIN_N, T, W
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.outcome import Outcome, evaluate_forward, report_wr, wilson_ci

LOOKBACKS = [120, 300, 600]


def _load(timeframes):
    data = load_candles(timeframes=timeframes)
    out = {}
    for sym, tfs in data.items():
        for tf in timeframes:
            df = tfs.get(tf)
            if df is None:
                continue
            if verify(df, sym, tf):
                continue
            out.setdefault(tf, []).append(df)
    return out


def level_events(df, atr, lookback, kind):
    """事件掩码列表 [(level, event_mask)]; kind=touch/breakout; 事件式进入"""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(c)
    lvls = cluster_levels(h, l, atr, min_touch=2)
    t_arr = np.arange(n)
    out = []
    for lv in lvls:
        usable = (t_arr >= lv.confirm_at) & (t_arr - lv.confirm_at < lookback)
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        if kind == "touch":
            m = (l <= p_hi) & (h >= p_lo) & usable
        else:
            m = (c < p_lo) & usable if lv.side == "support" else (c > p_hi) & usable
        e = m.copy()
        e[1:] &= ~m[:-1]
        out.append((lv, e))
    return out


def agg_outcomes(dfs, kind, lookback, direction_fn, base_long, base_short, side=None):
    n_win = n_loss = n_expired = n_skip = 0
    year_wl = {}
    for df in dfs:
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        atr = _atr_series(df)
        years = df.index.year.values
        for lv, e in level_events(df, atr, lookback, kind):
            if side is not None and lv.side != side:
                continue
            d = direction_fn(lv)
            if d is None:
                continue
            out, recs = evaluate_forward(c, h, l, atr, e, d, T, W)
            n_win += out.n_win
            n_loss += out.n_loss
            n_expired += out.n_expired
            n_skip += out.n_skip
            for r in recs:
                if r.outcome not in ("win", "loss"):
                    continue
                y = years[r.entry_idx]
                year_wl.setdefault(y, [0, 0])
                year_wl[y][0 if r.outcome == "win" else 1] += 1
    o = Outcome(n_win, n_loss, n_expired, n_skip)
    by_year = {y: wl[0] / (wl[0] + wl[1]) for y, wl in year_wl.items() if wl[0] + wl[1] >= 100}
    return o, by_year


def print_row(label, o, base, by_year):
    ci = wilson_ci(o.n_win, o.n_eval)
    flag = "" if o.n_eval >= MIN_N else " ⚠样本不足"
    yr = " ".join(f"{y}:{v:.0%}" for y, v in sorted(by_year.items()))
    print(f"  {label:>32}: n={o.n_eval:>6} WR {o.win_rate:>6.1%} vs 基线 {base.win_rate:>5.1%} "
          f"Δ{o.win_rate - base.win_rate:>+6.1%}pp [CI {ci[0]:.1%}-{ci[1]:.1%}] {yr}{flag}")


def run_b1(dfs_by_tf, label_extra=""):
    for tf, dfs in dfs_by_tf.items():
        print(f"═══ {tf} {label_extra}═══\n")
        ones = [np.ones(len(df), bool) for df in dfs]
        base_long = base_short = None
        for df, one in zip(dfs, ones):
            c = df["close"].values
            h = df["high"].values
            l = df["low"].values
            atr = _atr_series(df)
            o1, _ = evaluate_forward(c, h, l, atr, one, "long", T, W)
            o2, _ = evaluate_forward(c, h, l, atr, one, "short", T, W)
            if base_long is None:
                base_long, base_short = o1, o2
            else:
                base_long.n_win += o1.n_win
                base_long.n_loss += o1.n_loss
                base_short.n_win += o2.n_win
                base_short.n_loss += o2.n_loss
        print(f"  无条件基线 做多: {report_wr(base_long)}")
        print(f"  无条件基线 做空: {report_wr(base_short)}\n")

        for lb in LOOKBACKS:
            print(f"── S/R 回看 {lb} 根 ──")
            print("Q1 触碰后反弹方向 (支撑→做多 / 阻力→做空)")
            o, yr = agg_outcomes(dfs, "touch", lb,
                                 lambda lv: "long" if lv.side == "support" else "short",
                                 base_long, base_short, side="support")
            print_row(f"支撑触碰→做多×long", o, base_long, yr)
            o, yr = agg_outcomes(dfs, "touch", lb,
                                 lambda lv: "long" if lv.side == "support" else "short",
                                 base_long, base_short, side="resistance")
            print_row(f"阻力触碰→做空×short", o, base_short, yr)
            print("Q2 破位后延续方向 (支撑破位→做空 / 阻力破位→做多)")
            o, yr = agg_outcomes(dfs, "breakout", lb,
                                 lambda lv: "short" if lv.side == "support" else "long",
                                 base_long, base_short, side="support")
            print_row(f"支撑破位→做空×short", o, base_short, yr)
            o, yr = agg_outcomes(dfs, "breakout", lb,
                                 lambda lv: "short" if lv.side == "support" else "long",
                                 base_long, base_short, side="resistance")
            print_row(f"阻力破位→做多×long", o, base_long, yr)
            print()


def main():
    dfs_by_tf = _load(timeframes=("1h",))
    run_b1(dfs_by_tf)
    # ── 随机游走对照 ──
    rng = np.random.default_rng(0)
    ref = dfs_by_tf["1h"][0]
    n_ref = len(ref)
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw_dfs = []
    for _ in range(20):
        rets = rng.normal(0, sig, n_ref)
        close = 100 * np.exp(np.cumsum(rets))
        idx = ref.index
        rw_dfs.append(pd.DataFrame({"open": close, "high": close * (1 + 2 * sig),
                                    "low": close * (1 - 2 * sig), "close": close,
                                    "volume": 1.0}, index=idx))
    print("════════ 随机游走对照 (20 GBM) — 触碰/破位在无信息市场的行为基线 ════════\n")
    run_b1({"1h": rw_dfs}, label_extra="(随机游走)")


if __name__ == "__main__":
    main()
