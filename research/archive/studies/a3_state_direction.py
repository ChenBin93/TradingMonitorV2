#!/usr/bin/env python3
"""A3 状态倾向性研究 — 两套状态定义的条件方向胜率 vs 基线 (2026-08-03, 无未来函数)

两套状态 (并列对比, 不预设立场):
  ① 统计定义: state_features.state_series (8 趋势态) + vol_z_states (3 波动态)
  ② 结构定义: structures.structural_states (range / up|down × early|mid|late)

预注册问题:
  Q1 结构 vs 统计状态一致性 (方向判定一致率)
  Q2 各状态×方向 条件 1:1 胜率 vs 无条件基线 (事件式入场 = 状态进入第一根)
  Q3 H1: 结构趋势中期顺势胜率 > 基线?
  Q4 H2: 结构末期进入后, 未来 W 根内延续 vs 转 range 比例
  Q5 H3: 结构初期 (突破) 进入后, 方向延续 vs 假突破比例
  Q6 随机游走对照: 同参数 GBM 跑两套状态机 — 无信息基线, 分离真实信号与机械惯性

口径 (research/outcome.evaluate_forward, numpy 权威引擎):
  入场 = 状态进入 bar 收盘; 1:1 T×ATR; W 根窗口; 同 bar 双命中跳过
统计: Wilson CI + 分年稳定性 + MIN_N=200
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.caliber import MIN_N, T, W
from research.data_loader import load_candles, verify
from research.outcome import Outcome, evaluate_forward, report_wr, wilson_ci
from research.state_features import state_series, vol_z_states
from research.structures import structural_states

WARMUP = 750


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


def build_ctx(df):
    """预计算: 状态序列 + OHLC + atr (每标的只算一次)"""
    st, _ = state_series(df)
    atr = _atr_series(df)
    z120, _ = vol_z_states(atr, 120)
    stc = structural_states(df)
    return {
        "close": df["close"].values, "high": df["high"].values,
        "low": df["low"].values, "open": df["open"].values, "atr": atr,
        "stat": st[WARMUP:], "vol": z120[WARMUP:], "struct": stc[WARMUP:],
        "years": df.index.year.values[WARMUP:],
    }


def event_entries(states, target):
    n = len(states)
    out = np.zeros(n, bool)
    for t in range(n):
        if states[t] == target and (t == 0 or states[t - 1] != target):
            out[t] = True
    return out


def run_outcomes(ctxs, entries_list, direction, t_mult=T, w=W):
    """聚合所有标的的严格口径结果 (含分年)"""
    n_win = n_loss = n_expired = n_skip = 0
    year_wl = {}
    for ctx, entries in zip(ctxs, entries_list):
        out, recs = evaluate_forward(ctx["close"], ctx["high"], ctx["low"],
                                     ctx["atr"], entries, direction, t_mult=t_mult, w=w,
                                     open_px=ctx["open"])
        n_win += out.n_win
        n_loss += out.n_loss
        n_expired += out.n_expired
        n_skip += out.n_skip
        for r in recs:
            if r.outcome not in ("win", "loss"):
                continue
            y = ctx["years"][r.entry_idx]
            year_wl.setdefault(y, [0, 0])
            year_wl[y][0 if r.outcome == "win" else 1] += 1
    o = Outcome(n_win, n_loss, n_expired, n_skip)
    by_year = {y: wl[0] / (wl[0] + wl[1]) for y, wl in year_wl.items() if wl[0] + wl[1] >= 100}
    return o, by_year


def print_row(label, out, base, by_year):
    ci = wilson_ci(out.n_win, out.n_eval)
    yr = " ".join(f"{y}:{v:.0%}" for y, v in sorted(by_year.items()))
    flag = "" if out.n_eval >= MIN_N else " ⚠样本不足"
    print(f"  {label:>30}: n={out.n_eval:>6} WR {out.win_rate:>6.1%} vs 基线 {base.win_rate:>5.1%} "
          f"Δ{out.win_rate - base.win_rate:>+6.1%}pp [CI {ci[0]:.1%}-{ci[1]:.1%}] {yr}{flag}")


def state_followup(seqs, s_enter, w=W):
    """状态进入后 W 根内的演化 (延续/转 range/反向) — Q4/Q5"""
    cont = conv = total = 0
    for seq in seqs:
        for t in np.flatnonzero(event_entries(seq, s_enter)):
            window = seq[t + 1:min(t + 1 + w, len(seq))]
            if len(window) < w // 2:
                continue
            total += 1
            direction = s_enter.split(":")[0]  # up/down
            if np.any([x.startswith(direction) for x in window]):
                cont += 1
            elif np.any([x == "range" for x in window]) and not np.any([x.startswith(direction) for x in window]):
                conv += 1
    return cont, conv, total


def run_a3(dfs_by_tf):
    for tf, dfs in dfs_by_tf.items():
        print(f"═══ {tf} ═══")
        ctxs = [build_ctx(df) for df in dfs]
        n = len(ctxs)

        # ── 基线 ──
        ones = [np.ones(len(ctx["close"]) - WARMUP, bool) for ctx in ctxs]
        base_long, _ = run_outcomes(ctxs, ones, "long")
        base_short, _ = run_outcomes(ctxs, ones, "short")
        print(f"  无条件基线 做多: {report_wr(base_long)}")
        print(f"  无条件基线 做空: {report_wr(base_short)}\n")

        # ── Q1 一致性 ──
        print("Q1 结构 vs 统计 方向判定一致性\n")
        all_stat = np.concatenate([ctx["stat"] for ctx in ctxs])
        all_struct = np.concatenate([ctx["struct"] for ctx in ctxs])
        stat_dir = np.array([1 if s.startswith("trend_up") else -1 if s.startswith("trend_down") else 0 for s in all_stat])
        struct_dir = np.array([1 if s.startswith("up") else -1 if s.startswith("down") else 0 for s in all_struct])
        both = stat_dir != 0
        same = np.mean(stat_dir[both] == struct_dir[both])
        opp = np.mean(stat_dir[both] == -struct_dir[both])
        print(f"  统计有方向 {np.mean(both):.1%} | 结构有方向 {np.mean(struct_dir != 0):.1%}")
        print(f"  两者都有方向时: 一致 {same:.1%} | 反向 {opp:.1%}\n")

        # ── Q2 统计趋势态 ──
        print("Q2 统计趋势态 × 方向 (事件式入场)\n")
        stat_names = sorted(set(all_stat))
        for s in stat_names:
            dirs = ["long"] if s.startswith("trend_up") else ["short"] if s.startswith("trend_down") else ["long", "short"]
            for d in dirs:
                entries = [event_entries(ctx["stat"], s) for ctx in ctxs]
                out, yr = run_outcomes(ctxs, entries, d)
                print_row(f"{s}×{d}", out, base_long if d == "long" else base_short, yr)
        print()

        # ── Q2 波动态 ──
        print("Q2 波动态 × 方向\n")
        for v in ["低", "中", "高"]:
            for d in ["long", "short"]:
                entries = [event_entries(ctx["vol"], v) for ctx in ctxs]
                out, yr = run_outcomes(ctxs, entries, d)
                print_row(f"波动{v}×{d}", out, base_long if d == "long" else base_short, yr)
        print()

        # ── Q2 结构态 ──
        print("Q2 结构态 × 方向\n")
        struct_names = sorted(set(all_struct))
        for s in struct_names:
            if s == "warmup":
                continue
            dirs = ["long"] if s.startswith("up") else ["short"] if s.startswith("down") else ["long", "short"]
            for d in dirs:
                entries = [event_entries(ctx["struct"], s) for ctx in ctxs]
                out, yr = run_outcomes(ctxs, entries, d)
                print_row(f"{s}×{d}", out, base_long if d == "long" else base_short, yr)
        print()

        # ── Q4/Q5 状态演化 ──
        print("Q4/Q5 结构阶段进入后 W 根内状态演化 (H2 末期/H3 突破)\n")
        seqs_struct = [ctx["struct"] for ctx in ctxs]
        for s in ["up:early", "up:mid", "up:late", "down:early", "down:mid", "down:late"]:
            cont, conv, total = state_followup(seqs_struct, s)
            if total:
                print(f"  {s:>10}: 延续 {cont / total:.1%} | 转 range {conv / total:.1%} (n={total})")
        print()

        # ── Q6 随机游走对照 ──
        print("Q6 随机游走对照 (20 个 GBM, 同长度同波动率, 真实 OHLC 子步 — 无信息基线)\n")
        from research.sim_market import gbm_dataframe
        rng = np.random.default_rng(0)
        ref = dfs[0]
        n_ref = len(ref)
        sig = np.std(np.diff(np.log(ref["close"].values)))
        rw_dfs = [gbm_dataframe(n_ref, sig, seed=100 + k) for k in range(20)]
        rw_ctxs = [build_ctx(df) for df in rw_dfs]
        rw_ones = [np.ones(len(ctx["close"]) - WARMUP, bool) for ctx in rw_ctxs]
        rw_base, _ = run_outcomes(rw_ctxs, rw_ones, "long")
        print(f"  随机游走 无条件基线做多: {report_wr(rw_base)}")
        for s in ["trend_up:late", "transition", "up:mid", "up:early", "range"]:
            entries = [event_entries(ctx["stat"], s) for ctx in rw_ctxs]
            out, _ = run_outcomes(rw_ctxs, entries, "long")
            print(f"  随机游走 统计[{s}]做多: {report_wr(out)}")
        for s in ["up:mid", "up:early", "range"]:
            entries = [event_entries(ctx["struct"], s) for ctx in rw_ctxs]
            out, _ = run_outcomes(rw_ctxs, entries, "long")
            print(f"  随机游走 结构[{s}]做多: {report_wr(out)}")
        print()


if __name__ == "__main__":
    dfs_by_tf = _load(timeframes=("1h", "4h"))
    run_a3(dfs_by_tf)
