#!/usr/bin/env python3
"""A1 窗口研究 — 各周期分析需要多大窗口 (2026-08-03, 重头开始, 无未来函数)

方法 (无未来函数):
  A1a 方向记忆: 过去 W 根收益符号 → 未来 W 根方向的条件命中率 vs 无条件基线
       特征只含 bar ≤ i 的已收盘数据 (c[i]-c[i-W]); 结果变量 (c[i+W]-c[i]) 仅用于判定
  A1b 波动记忆: log(ATR) 自相关 ACF → 半衰期; 对比 1h/4h 半衰期按时间 vs 按根数对齐
  A1c 指标收敛: ewm 指标 (ATR/ADX) 用 60/120/240 根截断历史 vs 全量收敛值
       (重验 live market_phase "指标历史 120 根即信息上限")

统计纪律: 无条件基线对照 + Wilson 95% CI + 分年稳定性 + MIN_N=200
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series, _adx_series
from research.caliber import MIN_N
from research.data_loader import load_candles, verify
from research.outcome import wilson_ci

WS = [2, 5, 10, 20, 40, 80, 160]


def _load(timeframes):
    data = load_candles(timeframes=timeframes)
    out = {}
    for sym, tfs in data.items():
        for tf in timeframes:
            df = tfs.get(tf)
            if df is None:
                continue
            problems = verify(df, sym, tf)
            if problems:
                print(f"跳过 {sym} {tf}: {problems}")
                continue
            out.setdefault(tf, []).append(df)
    return out


# ────────────────────────────────────────────────
# A1a 方向记忆
# ────────────────────────────────────────────────
def direction_memory(dfs, w):
    """聚合所有标的: 过去 W 根动量 → 未来 W 根方向 条件命中率"""
    past_all, fut_all, year_all = [], [], []
    for df in dfs:
        c = df["close"].values
        n = len(c)
        # i ∈ [w, n-w): past = 过去 W 根符号, fut = 未来 W 根符号
        past = np.sign(c[w:n - w] - c[0:n - 2 * w])
        fut = np.sign(c[2 * w:n] - c[w:n - w])
        m = (past != 0) & (fut != 0)
        past_all.append(past[m])
        fut_all.append(fut[m])
        year_all.append(df.index.year.values[w:n - w][m])
    past = np.concatenate(past_all)
    fut = np.concatenate(fut_all)
    year = np.concatenate(year_all)
    hit = past == fut
    n = len(hit)
    wr = hit.mean()
    base = (fut > 0).mean()
    lo, hi = wilson_ci(int(hit.sum()), n)
    by_year = {y: hit[year == y].mean() for y in sorted(set(year))}
    return n, wr, base, lo, hi, by_year


def run_a1a(dfs_by_tf):
    print("═══ A1a 方向记忆: 过去 W 根动量 → 未来 W 根方向 条件命中率 ═══")
    print("(基线 = 无条件上涨概率; Δpp = 条件-基线; 正 Δpp 才有预测力)\n")
    for tf, dfs in dfs_by_tf.items():
        print(f"── {tf} ──")
        print(f"{'W':>4} {'n':>8} {'条件WR':>7} {'基线':>6} {'Δpp':>6} {'95%CI':>16}  分年")
        for w in WS:
            n, wr, base, lo, hi, by_year = direction_memory(dfs, w)
            years = " ".join(f"{y}:{by_year[y]:.1%}" for y in sorted(by_year))
            flag = "" if n >= MIN_N else " ⚠样本不足"
            print(f"{w:>4} {n:>8} {wr:>7.1%} {base:>6.1%} {wr - base:>+6.1%} "
                  f"[{lo:.1%}-{hi:.1%}] {years}{flag}")
        print()


# ────────────────────────────────────────────────
# A1b 波动记忆: log(ATR) ACF 半衰期
# ────────────────────────────────────────────────
def atr_acf(df, max_lag):
    atr = _atr_series(df)
    x = np.log(atr[atr > 0])
    n = len(x)
    lags = np.arange(1, max_lag + 1)
    acf = np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in lags])
    half = np.where(acf < 0.5)[0]
    half_lag = int(half[0]) + 1 if len(half) else None
    return acf, half_lag


def run_a1b(dfs_by_tf):
    print("═══ A1b 波动记忆: log(ATR) 自相关 → 半衰期 ═══")
    print("(半衰期 = 自相关首次 < 0.5 的 lag; 若按时间对齐则 1h 半衰期 ≈ 4×4h 半衰期)\n")
    MAX_LAG = 168
    for tf, dfs in dfs_by_tf.items():
        acfs, halfs, beyond = [], [], 0
        for df in dfs:
            acf, half = atr_acf(df, MAX_LAG)
            acfs.append(acf)
            if half is None:
                beyond += 1
            else:
                halfs.append(half)
        acf_mean = np.mean(np.stack(acfs), axis=0)
        unit = "h" if tf == "1h" else "天" if tf == "4h" else ""
        factor = 1 if tf == "1h" else 6
        if halfs:
            half_mean = np.mean(halfs)
            span = f"[标的区间 {min(halfs)}-{max(halfs)} 根]"
        else:
            half_mean = float("nan")
            span = ""
        beyond_note = f"; {beyond}/{len(dfs)} 标的 168 根内未跌破 0.5" if beyond else ""
        print(f"── {tf}: 半衰期 ≈ {half_mean:.0f} 根 ({half_mean * factor / 24:.1f} 天) {span}{beyond_note}")
        row = "ACF @ lag: " + "  ".join(
            f"{k}根={acf_mean[k - 1]:.3f}" for k in [1, 2, 4, 8, 12, 24, 48, 96, 168] if k <= MAX_LAG)
        print("  " + row)
        print()


# ────────────────────────────────────────────────
# A1c 指标收敛: ewm 指标需要多少根历史
# ────────────────────────────────────────────────
def convergence(df, window):
    """指标在截断 window 根历史 vs 全量收敛值的相对误差 (抽样 5 点平均)"""
    atr_full = _atr_series(df)
    adx_full = _adx_series(df)
    n = len(df)
    pts = np.linspace(window + 50, n - 1, 5).astype(int)
    atr_err, adx_err = [], []
    for i in pts:
        if atr_full[i] <= 0 or not np.isfinite(atr_full[i]):
            continue
        atr_cut = _atr_series(df.iloc[i - window + 1:i + 1])[-1]
        adx_cut = _adx_series(df.iloc[i - window + 1:i + 1])[-1]
        atr_err.append(abs(atr_cut - atr_full[i]) / atr_full[i])
        if adx_full[i] > 1:
            adx_err.append(abs(adx_cut - adx_full[i]) / max(adx_full[i], 1))
    return np.mean(atr_err), np.mean(adx_err)


def run_a1c(dfs_by_tf):
    print("═══ A1c 指标收敛: ewm 指标截断历史 vs 全量收敛值的相对误差 ═══")
    print("(重验 live 假设'指标历史 120 根即信息上限'; 误差 <1% 视为收敛)\n")
    for tf, dfs in dfs_by_tf.items():
        print(f"── {tf} ──")
        print(f"{'窗口(根)':>8} {'ATR 相对误差':>14} {'ADX 相对误差':>14}")
        for w in (30, 60, 90, 120, 240):
            atr_errs, adx_errs = [], []
            for df in dfs:
                ae, xe = convergence(df, w)
                atr_errs.append(ae)
                adx_errs.append(xe)
            print(f"{w:>8} {np.mean(atr_errs):>13.2%} {np.mean(adx_errs):>13.2%}")
        print()


if __name__ == "__main__":
    dfs_by_tf = _load(timeframes=("1h", "4h"))
    run_a1a(dfs_by_tf)
    run_a1b(dfs_by_tf)
    run_a1c(dfs_by_tf)
