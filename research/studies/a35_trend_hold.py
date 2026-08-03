#!/usr/bin/env python3
"""A3.5 趋势跟踪策略验证 — early 入场持有到末期/破位 (2026-08-03, 无未来函数)

验证用户模型: 趋势初期入场 (ADX/偏离/量能可识别阶段, 已验证), 持有到末期,
方向由已确认结构跟随 (非预测)。策略 = 开放式持有 + 状态/止损退出。

设计:
  入场: 结构状态 up:early (做多) / down:early (做空) 进入时 bar 收盘
  4 组合: sl_mode (hl=跟踪最近HL / atr=固定1×ATR) × exit_late (False/True)
  统计: n / 胜率 / 平均R / 盈亏比 / 期望R / 分年 / timeout
  Q1: 趋势状态内顺趋势 K 比例 (用户直觉: 顺K多于逆K) vs 随机游走
  Q2: 4 组合的期望 R
  Q3: 随机游走对照 (20 GBM) — 真实期望 − GBM 基线 = 市场趋势性贡献

口径: 事件式入场 (每笔独立, 允许重叠); 止损触发按止损价成交 (保守);
  R 倍数 = (exit-entry)/初始止损距离; 无未来函数 (已收盘+已确认pivot)
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
from research.structures import structural_states

WARMUP = 100  # 结构态只需 pivot 确认 (~K*2) + 状态收敛


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


def event_entries(states, target):
    n = len(states)
    out = np.zeros(n, bool)
    for t in range(n):
        if states[t] == target and (t == 0 or states[t - 1] != target):
            out[t] = True
    return out


def run_combo(dfs, sl_mode, exit_late, direction, tag, w=96):
    """聚合所有标的: 期望 R / 胜率 / 分年"""
    rs, years, reasons = [], [], []
    for df in dfs:
        st = structural_states(df)[WARMUP:]
        c = df["close"].values[WARMUP:]
        h = df["high"].values[WARMUP:]
        l = df["low"].values[WARMUP:]
        atr = _atr_series(df)[WARMUP:]
        entries = event_entries(st, tag)
        trades = simulate_holds(c, h, l, atr, st, entries, direction,
                                sl_mode, exit_late, w)
        yrs = df.index.year.values[WARMUP:]
        for t in trades:
            if t.reason == "timeout":
                continue
            rs.append(t.r_mult)
            reasons.append(t.reason)
            years.append(yrs[t.entry_idx])
    rs = np.array(rs)
    years = np.array(years)
    reasons = np.array(reasons)
    if len(rs) == 0:
        return None
    wins = rs[rs > 0]
    losses = rs[rs <= 0]
    wr = len(wins) / len(rs)
    exp_r = rs.mean()
    pf = (wins.mean() * len(wins)) / (abs(losses.mean()) * len(losses)) if len(losses) else np.inf
    by_year = {}
    for y in sorted(set(years)):
        m = years == y
        if m.sum() >= 50:
            by_year[y] = rs[m].mean()
    flag = "" if len(rs) >= MIN_N else " ⚠样本不足"
    return dict(n=len(rs), wr=wr, exp_r=exp_r, pf=pf, by_year=by_year,
                n_win=len(wins), n_loss=len(losses), flag=flag)


def print_combo(res, label):
    if res is None:
        print(f"  {label:>34}: 无持仓")
        return
    yr = " ".join(f"{y}:{v:+.2f}R" for y, v in sorted(res["by_year"].items()))
    print(f"  {label:>34}: n={res['n']:>5} 胜率{res['wr']:6.1%} 期望R{res['exp_r']:+.3f} "
          f"盈亏比{res['pf']:5.2f} (win{res['n_win']}/loss{res['n_loss']}) {yr}{res['flag']}")


def pro_trend_ratio(dfs):
    """Q1: 趋势状态内顺趋势 K 比例 (up 态顺=涨, down 态顺=跌)"""
    ratios = []
    for df in dfs:
        st = structural_states(df)[WARMUP:]
        c = df["close"].values[WARMUP:]
        up = np.array([s.startswith("up:") for s in st])
        dn = np.array([s.startswith("down:") for s in st])
        d = np.diff(c)
        ok = up[1:] & (d > 0) | dn[1:] & (d < 0)
        m = up[1:] | dn[1:]
        if m.sum():
            ratios.append(ok[m].mean())
    return np.mean(ratios)


def main():
    dfs_by_tf = _load(timeframes=("1h", "4h"))
    for tf, dfs in dfs_by_tf.items():
        print(f"═══ {tf} ═══\n")
        print("Q1 趋势状态内顺趋势 K 比例 (用户直觉: 顺K应多于逆K)\n")
        r_real = pro_trend_ratio(dfs)
        print(f"  真实: {r_real:.1%} (50% = 无方向偏差)\n")

        print("Q2 趋势持有策略 4 组合 (early 入场)\n")
        for sl, sln in [("hl", "跟踪HL"), ("atr", "固定ATR")]:
            for el, eln in [(False, "破位退出"), (True, "late即退")]:
                for direction, tag in [("long", "up:early"), ("short", "down:early")]:
                    res = run_combo(dfs, sl, el, direction, tag)
                    print_combo(res, f"{sln}×{eln}×{direction}")
        print()

        # ── 随机游走对照 ──
        print("Q3 随机游走对照 (20 GBM, 同长度同波动率)\n")
        rng = np.random.default_rng(0)
        ref = dfs[0]
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
        r_rw = pro_trend_ratio(rw_dfs)
        print(f"  Q1 随机游走: 顺K比例 {r_rw:.1%}")
        for sl, sln in [("hl", "跟踪HL"), ("atr", "固定ATR")]:
            for el, eln in [(False, "破位退出"), (True, "late即退")]:
                for direction, tag in [("long", "up:early"), ("short", "down:early")]:
                    res = run_combo(rw_dfs, sl, el, direction, tag)
                    print_combo(res, f"GBM {sln}×{eln}×{direction}")
        print()


if __name__ == "__main__":
    main()
