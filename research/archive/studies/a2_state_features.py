#!/usr/bin/env python3
"""A2 市场状态特征研究 — 市场有哪些状态, 各状态由哪些特征刻画 (2026-08-03, 无未来函数)

状态框架 (二维):
  趋势态: live market_phase.classify 逐 bar 复现 (trend_up/trend_down/range/transition + 阶段)
  波动态: ATR 滚动 z 三分位 (低/中/高) — 短分位 120 根 (判当前状态) vs 长背景 720 根

预注册问题:
  Q1 状态占比与平均持续时长 (根+时间), 分年稳定性 (2024/2025/2026)
  Q2 转移矩阵; 验证"低↔高波动必经中波动"层级结构 (A1.5 初步发现)
  Q3 趋势×波动独立性 (卡方 + Cramer V) — 决定二维框架是否必要
  Q4 特征画像: 每状态 ADX/斜率/偏离/动能/实体比/BB宽/波动z/量能比 分布
  Q5 阶段可分性: early/accelerate/late 的偏离/ADX转折/实体比 分离度
  Q6 转换状态量能: transition 期量能比是否显著放大

无未来函数: 特征逐 bar 用已收盘数据 (rolling/ewm 左对齐, 已过不变性测试);
  波动 z 用滚动窗口 (只含过去数据), 状态标签可直接给 A3 复用
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.state_features import state_series, vol_z_states

WARMUP = 730  # ma60+shift10+body13 + 波动长分位 720 根


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


def state_series(df):
    """逐 bar 趋势态 + 特征数组 (与 live analyze_market_state 语义一致, 向量化)"""
    c = pd.Series(df["close"].values)
    o = pd.Series(df["open"].values)
    v = pd.Series(df["volume"].values)
    atr = pd.Series(_atr_series(df))
    adx = pd.Series(_adx_series(df))
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    dev = (c - ma20) / atr
    slope = (ma20 - ma20.shift(10)) / atr
    spread = (ma20 - ma60) / atr
    mom = (c - c.shift(10)) / atr
    adx_prev = adx.shift(10)
    body = (c - o).abs()
    body_recent = body.rolling(3).mean()
    body_prior = body.rolling(13).mean().shift(3)
    vol_ratio = v / v.rolling(20).mean()
    bbw = c.rolling(20).std() / c.rolling(20).mean()

    states = []
    n = len(df)
    for i in range(n):
        r = classify(atr.iloc[i], adx.iloc[i], adx_prev.iloc[i], slope.iloc[i],
                     spread.iloc[i], dev.iloc[i], mom.iloc[i],
                     body_recent.iloc[i], body_prior.iloc[i])
        s = r["state"]
        if r["stage"] and s.startswith("trend"):
            s = f"{s}:{r['stage']}"
        states.append(s)

    feats = pd.DataFrame({
        "adx": adx.values, "dev": dev.values, "slope": slope.values,
        "mom": mom.values, "body_ratio": (body_recent / body_prior).values,
        "bbw": bbw.values, "vol_ratio": vol_ratio.values,
        "atr_c": atr.values / c.values,
    }, index=df.index)
    return np.array(states), feats


def vol_z_states(atr, window):
    s = pd.Series(atr)
    z = ((s - s.rolling(window).mean()) / s.rolling(window).std()).values
    return np.where(z < -0.5, "低", np.where(z > 0.5, "高", "中")), z


def run_stats(seq, hours_per_bar, years):
    """状态占比 + 平均持续时长 (根/时间) + 分年占比"""
    names = sorted(set(seq))
    occ = {s_: float(np.mean(seq == s_)) for s_ in names}
    dur = {}
    for s_ in names:
        mask = seq == s_
        runs = []
        cur, cnt = False, 0
        for m in mask:
            if m:
                cnt += 1
            else:
                if cnt:
                    runs.append(cnt)
                cnt = 0
        if cnt:
            runs.append(cnt)
        dur[s_] = np.mean(runs) if runs else np.nan
    by_year = {}
    for y in sorted(set(years)):
        m = years == y
        by_year[y] = {s_: float(np.mean(seq[m] == s_)) for s_ in names} if m.sum() else {}
    return occ, dur, by_year


def transition_matrix(seq):
    names = sorted(set(seq))
    idx = {s_: i for i, s_ in enumerate(names)}
    n = len(names)
    tm = np.zeros((n, n))
    for a, b in zip(seq[:-1], seq[1:]):
        tm[idx[a], idx[b]] += 1
    rs = tm.sum(axis=1, keepdims=True)
    return names, tm / np.maximum(rs, 1)


def feature_portrait(feats, seq, cols):
    names = sorted(set(seq))
    out = {}
    for s_ in names:
        m = seq == s_
        out[s_] = {c_: (feats.loc[m, c_].mean(), feats.loc[m, c_].std()) for c_ in cols}
    return out


def fmt_cell(v):
    return f"{v[0]:.2f}±{v[1]:.2f}"


def run_a2(dfs_by_tf):
    for tf, dfs in dfs_by_tf.items():
        hours = 1 if tf == "1h" else 4
        agg = {"trend": {}, "vol120": {}, "vol720": {}}
        # 聚合: 所有标的的 (状态序列, 年份, 特征, 波动z)
        seqs = {k: [] for k in agg}
        years_all = []
        feats_all = []
        z_all = {k: [] for k in ("vol120", "vol720")}
        for df in dfs:
            states, feats = state_series(df)
            atr = _atr_series(df)
            z120, _ = vol_z_states(atr, 120)
            z720, _ = vol_z_states(atr, 720)
            z120 = z120[WARMUP:]
            z720 = z720[WARMUP:]
            st = states[WARMUP:]
            years = df.index.year.values[WARMUP:]
            seqs["trend"].append(st)
            seqs["vol120"].append(z120)
            seqs["vol720"].append(z720)
            years_all.append(years)
            feats_all.append(feats.iloc[WARMUP:])
            z_all["vol120"].append(z120)
            z_all["vol720"].append(z720)

        # ── Q1 占比/持续/分年 ──
        print(f"═══ {tf} ═══")
        print("Q1 状态占比与平均持续时长 (分年稳定性)\n")
        for key, label in [("trend", "趋势态"), ("vol120", "波动态(z120)"), ("vol720", "波动态(z720)")]:
            seq = np.concatenate(seqs[key])
            years = np.concatenate(years_all)
            occ, dur, by_year = run_stats(seq, hours, years)
            parts = []
            for s_ in sorted(occ):
                parts.append(f"{s_}: 占{occ[s_]:.0%} 持续{dur[s_]:.0f}根({dur[s_]*hours:.0f}h)")
            print(f"  {label}: {' | '.join(parts)}")
            stable = all(len(v) == 4 and all(abs(v[k] - occ[k]) < 0.08 for k in v)
                         for y, v in by_year.items() if v)
            year_str = " ".join(f"{y}:{' '.join(f'{k}{v[k]:.0%}' for k in sorted(v))}"
                                for y, v in by_year.items())
            print(f"    分年: {year_str}")
        print()

        # ── Q2 转移矩阵 ──
        print("Q2 转移矩阵 (行=当前, 列=下一根)\n")
        for key, label in [("trend", "趋势态"), ("vol120", "波动态")]:
            seq = np.concatenate(seqs[key])
            names, tm = transition_matrix(seq)
            print(f"  {label}:")
            for i, s_ in enumerate(names):
                row = " ".join(f"{tm[i, j]:.2f}" for j in range(len(names)))
                print(f"    {s_:>12}: [{row}]")
        seq120 = np.concatenate(seqs["vol120"])
        names, tm = transition_matrix(seq120)
        idx = {s_: i for i, s_ in enumerate(names)}
        p_lh = tm[idx["低"], idx["高"]]
        p_lm = tm[idx["低"], idx["中"]]
        p_mh = tm[idx["中"], idx["高"]]
        print(f"    层级结构: 低→高直接 {p_lh:.3f} vs 经中路径 {p_lm * p_mh:.3f}")
        print()

        # ── Q3 独立性 ──
        print("Q3 趋势×波动独立性 (卡方)\n")
        seq_t = np.concatenate(seqs["trend"])
        seq_v = np.concatenate(seqs["vol120"])
        t_names = sorted(set(seq_t))
        v_names = ["低", "中", "高"]
        ct = np.zeros((len(t_names), 3))
        for i, t_ in enumerate(t_names):
            for j, v_ in enumerate(v_names):
                ct[i, j] = np.mean((seq_t == t_) & (seq_v == v_))
        chi2, p, dof, _ = chi2_contingency(ct * len(seq_t))
        cramer = np.sqrt(chi2 / (len(seq_t) * (min(ct.shape) - 1)))
        print(f"  卡方 p={p:.2e} (显著=不独立), Cramer V={cramer:.3f} (<0.1 弱关联)")
        print("  趋势态×波动态 联合占比:")
        for i, t_ in enumerate(t_names):
            row = " ".join(f"{ct[i, j]:.0%}" for j in range(3))
            print(f"    {t_:>12}: [{row}]")
        print()

        # ── Q4 特征画像 ──
        print("Q4 特征画像 (均值±std)\n")
        cols = ["adx", "dev", "slope", "mom", "body_ratio", "bbw", "atr_c", "vol_ratio"]
        for key, label in [("trend", "趋势态"), ("vol120", "波动态")]:
            seq = np.concatenate(seqs[key])
            feats = pd.concat(feats_all)
            portrait = feature_portrait(feats, seq, cols)
            print(f"  {label}:")
            print(f"    {'状态':>12} | {'ADX':>10} | {'偏离':>10} | {'斜率':>10} | {'动能':>10} | {'实体比':>9} | {'BB宽':>9} | {'波动率':>9} | {'量能比':>9}")
            for s_ in sorted(portrait):
                p = portrait[s_]
                print(f"    {s_:>12} | {fmt_cell(p['adx']):>10} | {fmt_cell(p['dev']):>10} | "
                      f"{fmt_cell(p['slope']):>10} | {fmt_cell(p['mom']):>10} | {fmt_cell(p['body_ratio']):>9} | "
                      f"{fmt_cell(p['bbw']):>9} | {fmt_cell(p['atr_c']):>9} | {fmt_cell(p['vol_ratio']):>9}")
        print()

        # ── Q5 阶段可分性 ──
        print("Q5 阶段可分性 (early/accelerate/late 的特征分离)\n")
        seq_t = np.concatenate(seqs["trend"])
        feats = pd.concat(feats_all)
        stages = [s_ for s_ in sorted(set(seq_t)) if ":" in s_]
        if stages:
            print(f"    {'阶段':>18} | {'偏离':>10} | {'ADX':>10} | {'实体比':>9}")
            for s_ in stages:
                m = seq_t == s_
                print(f"    {s_:>18} | {feats.loc[m,'dev'].mean():>10.2f} | {feats.loc[m,'adx'].mean():>10.1f} | "
                      f"{feats.loc[m,'body_ratio'].mean():>9.2f}")
        print()

        # ── Q6 转换状态量能 ──
        print("Q6 转换状态量能比 (transition vs 其他)\n")
        seq_t = np.concatenate(seqs["trend"])
        for s_ in ["transition", "range"]:
            m = seq_t == s_
            other = ~m & (seq_t != "unknown")
            v_t = feats.loc[m, "vol_ratio"].mean()
            v_o = feats.loc[other, "vol_ratio"].mean()
            print(f"  {s_}: 量能比 {v_t:.2f} vs 其余 {v_o:.2f} ({'放大' if v_t > v_o else '缩小'})")
        print()


if __name__ == "__main__":
    dfs_by_tf = _load(timeframes=("1h", "4h"))
    run_a2(dfs_by_tf)
