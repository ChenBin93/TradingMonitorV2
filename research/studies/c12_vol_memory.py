#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C12 波动长记忆 + 状态转移 (因果口径重验) (2026-08-13, 无未来函数, 1h/4h)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画市场事实 (波动记忆结构与
  状态转移结构), 无入场, 无交易含义, 无任何方向/收益结论。所有统计为事后
  描述; 若未来用作特征/条件, 必须经滚动口径重验。描述层发布门槛: 无胜率/
  期望/成本要求, 但必须有 GBM 无信息对照与数字可溯源。

============================================================
预注册假设 (运行前冻结, 结论逐条回应):
  H1: DFA-Hurst(1h) ≥ 0.70 且 4h ≥ 0.65 (各标的聚合, 中位数口径)
  H2: log(ATR) ACF @lag168 ≥ 0.5 (1h)
  H3: 低↔高直接转移率 ≤ 0.01 (必经中)
  H4: z120 三分位状态持续: 1h 低 50~60 根 / 高 40~50 根

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                            | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐  | bar 收盘后 | ctx 唯一对齐出口 (禁手动切片)
  atr              | causal ewm (ATR_PERIOD=14, 左对齐)  | bar 收盘后 | make_ctx 内置 (market_phase 语义)
  log(TR/close)    | 由 ctx.high/low/close 逐 bar 计算    | bar 收盘后 | H1 序列 (相对原始区间)
  log(ATR/close)   | ctx.atr / ctx.close                  | bar 收盘后 | H2/H3/H4 序列 (相对 ATR)
  z120 状态        | causal.rolling_percentile(w=120,     | 尾窗已收盘 | research.causal (禁全样本分位)
                   |   q=1/3, 2/3) 三分位                |            |
  DFA-H            | 自写 DFA (numpy, 线性去趋势, 向量化) | 全样本     | 探测器自检: 白噪声/GBM H≈0.5
  ACF@lag168       | 掩码互相关 (无切片)                  | 全样本     | 描述层
  GBM 无信息对照   | sim_market.gbm_matching (30 种子)    | 锚定真实   | 固定种子序列 0..29
  permutation 置换 | rng.permutation 打乱 log(ATR/close)  | 全样本     | H2 零假设 (见设计偏离)

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。

设计偏离说明 (相对波动口径 — 标定证据在 GATE/结论中引用):
  任务初稿指定绝对 log(ATR)。标定 (2026-08-13, 运行前) 发现:
  - 绝对 TR/ATR 与价格水平成比例 → log(绝对 TR) 对纯 GBM 的 ACF@lag1≈0.96,
    DFA-H≈1.4 — 价格随机游走水平污染, 不是波动记忆;
  - 白噪声经 ewm(1/14) 后 DFA-H≈0.83, GBM log(ATR/close) 30种子 H≈0.84 —
    ewm 平滑核自带 AR(1) 记忆, 使 GBM 零假设 H≈0.5 不可达 (A15 旧脚本已
    披露"ewm 平滑伪影", 用 |r| 替代)。
  故 H1 序列改为 log(TR/close) (原始相对区间, 无平滑; GBM 30种子 null H≈0.50),
  H2/H3/H4 序列用 log(ATR/close) (相对 ATR, 去除价格水平; lag168 处 ewm 核
  已衰减至 0.93^168≈0, 不影响 ACF 长记忆测量)。
  H2 零假设用 permutation (置换) 而非相位随机化/块 bootstrap: 后者保谱 →
  保留 ACF, 不能作 H2 的零假设; 置换彻底打乱时序 → ACF→0, 是唯一正确 null。

发布门槛自检 (描述层):
  - GATE 探测器: 白噪声 DFA-H∈[0.45,0.55]; 白噪声 ACF@168 |·|<0.05;
    GBM(logTR/c) 30种子 mean H∈[0.48,0.52] — 任一失败 SystemExit (违规即停)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义, 不涉及胜率/期望/成本 (描述层门槛)

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c12_vol_memory.py
  python3 research/studies/c12_vol_memory.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path (脚本以 `python3 research/studies/c12_vol_memory.py` 直接运行时,
# sys.path[0]=脚本目录, 需手动补根 — 模板同样缺少此行, 属模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.caliber import ATR_PERIOD, MIN_GBM_SEEDS, MIN_N
from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "warmup": 600,          # make_ctx 截断起点 (覆盖 atr ewm warm-up + 特征 head)
    "head_drop": 60,        # 截断后仍丢弃前 60 根 (tr[0]=0/atr 未收敛 → log 尖刺)
    "atr_period": ATR_PERIOD,
    "z_window": 120,
    "acf_lag": 168,
    "gbm_seeds": MIN_GBM_SEEDS,
    "dfa_min_scale": 16,
    "dfa_n_scales": 18,
    "perm_n": 5,
    "by_year_min_bars": 800,
    "rng_seed": 42,
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c12_vol_memory"


# ── 加载 ─────────────────────────────────────────────────────
def load(timeframes):
    data = load_candles(timeframes=timeframes)
    out = {}
    for sym, tfs in data.items():
        for tf in timeframes:
            df = tfs.get(tf)
            if df is None or verify(df, sym, tf):
                continue
            out.setdefault(tf, []).append(df)
    return out


# ── 序列构造 (因果, 无切片) ─────────────────────────────────
def build_series(ctx, head_drop=60):
    """相对波动序列 + 年份 (全部长度 = n, 统一 head_drop 掩码).

    返回 (logtr, logatr, logr, years) — 均已剔除不可用头部与非有限值:
      logtr   = log(TR/close)  原始相对区间 (H1)
      logatr  = log(ATR/close) 相对 ATR (H2/H3/H4)
      logr    = |log-return|   交叉验证 (a15 旧口径)
    """
    c = ctx.close
    n = len(c)
    cprev = np.roll(c, 1)
    m1 = np.arange(n) >= 1
    tr = np.zeros(n)
    tr[m1] = np.maximum(
        ctx.high[m1] - ctx.low[m1],
        np.maximum(np.abs(ctx.high[m1] - cprev[m1]),
                   np.abs(ctx.low[m1] - cprev[m1])))
    logtr = np.log(np.maximum(tr, 1e-12) / np.maximum(c, 1e-12))
    logatr = np.log(np.maximum(ctx.atr, 1e-12) / np.maximum(c, 1e-12))
    logr = np.abs(np.log(c / np.maximum(cprev, 1e-12)))
    fin = (np.isfinite(logtr) & np.isfinite(logatr) & np.isfinite(logr)
           & (np.abs(logr) > 0.0) & (np.arange(n) >= head_drop))
    return logtr[fin], logatr[fin], logr[fin], ctx.years[fin]


# ── DFA-Hurst (自写, numpy, 线性去趋势, 掩码无切片) ──────────
def dfa_hurst(x, min_scale=16, n_scales=18, min_blocks=10):
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    n = len(y)
    max_scale = max(32, n // 16)
    scales = np.unique(np.geomspace(min_scale, max_scale, n_scales).astype(int))
    fs, ss = [], []
    for s in scales:
        n_blocks = n // s
        if n_blocks < min_blocks:
            continue
        keep = np.arange(n) < n_blocks * s
        Y = y[keep].reshape(n_blocks, s)
        t = np.arange(s, dtype=float)
        t_dm = t - t.mean()
        Y_dm = Y - Y.mean(axis=1, keepdims=True)
        slope = (Y_dm * t_dm).sum(axis=1) / (t_dm * t_dm).sum()
        resid = Y_dm - slope[:, None] * t_dm
        fs.append(np.sqrt((resid * resid).mean()))
        ss.append(s)
    if len(ss) < 5:
        return float("nan")
    return float(np.polyfit(np.log(ss), np.log(fs), 1)[0])


# ── ACF (掩码互相关, 无切片) ────────────────────────────────
def acf_lag(x, lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    v = np.mean(x * x)
    if v <= 0:
        return float("nan")
    n = len(x)
    m1 = np.arange(n) < n - lag
    m2 = np.arange(n) >= lag
    return float(np.mean(x[m1] * x[m2]) / v)


# ── z120 状态 (rolling_percentile 三分位) ───────────────────
def z120_states(logatr, w=120):
    rp33 = rolling_percentile(logatr, w, 1.0 / 3.0)
    rp66 = rolling_percentile(logatr, w, 2.0 / 3.0)
    s = np.full(len(logatr), "中")
    s[logatr <= rp33] = "低"
    s[logatr > rp66] = "高"
    s[np.isnan(rp33) | np.isnan(rp66)] = "NA"
    return s


def transition_direct(states):
    n = len(states)
    lh = hl = tot = 0
    for i in range(n - 1):
        a = states[i]
        b = states[i + 1]
        if a == "NA" or b == "NA":
            continue
        tot += 1
        if a == "低" and b == "高":
            lh += 1
        elif a == "高" and b == "低":
            hl += 1
    return (lh / tot if tot else float("nan"),
            hl / tot if tot else float("nan"), tot)


def run_lengths(states):
    lens = {"低": [], "高": []}
    cur = states[0]
    cnt = 0
    for v in states:
        if v == cur:
            cnt += 1
        else:
            if cur in lens:
                lens[cur].append(cnt)
            cur = v
            cnt = 1
    if cur in lens:
        lens[cur].append(cnt)
    return lens


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_1h_df, params):
    """探测器自检 + GBM 30 种子零假设 (同管线重放 log(TR/close)), 失败 SystemExit"""
    p = params
    rng = np.random.default_rng(0)

    # 白噪声探测器 (DFA + ACF)
    wn = rng.normal(size=20000)
    h_wn = dfa_hurst(wn, p["dfa_min_scale"], p["dfa_n_scales"])
    acf_wn = acf_lag(wn, p["acf_lag"])
    if not 0.45 <= h_wn <= 0.55:
        raise SystemExit(f"GATE FAIL: 白噪声 DFA-H={h_wn:.3f} ∉ [0.45,0.55] — DFA 探测器错误")
    if abs(acf_wn) > 0.05:
        raise SystemExit(f"GATE FAIL: 白噪声 ACF@168={acf_wn:.3f} |·|>0.05 — ACF 探测器错误")

    # GBM 30 种子零假设: log(TR/close) DFA-H ≈ 0.5 (同管线重放)
    gbm_hs = []
    for seed in range(p["gbm_seeds"]):
        rw = gbm_matching(ref_1h_df, seed=seed)
        ctx = make_ctx(rw, p["warmup"], state_fns={})
        logtr, logatr, logr, years = build_series(ctx, p["head_drop"])
        gbm_hs.append(dfa_hurst(logtr, p["dfa_min_scale"], p["dfa_n_scales"]))
    gbm_mean = float(np.mean(gbm_hs))
    if not 0.48 <= gbm_mean <= 0.52:
        raise SystemExit(
            f"GATE FAIL: GBM 30种子 log(TR/c) DFA-H mean={gbm_mean:.3f} ∉ [0.48,0.52] — "
            f"零假设偏置, 停")
    return {"wn_h": h_wn, "wn_acf": acf_wn, "gbm_1h": np.array(gbm_hs), "gbm_mean": gbm_mean}


# ── 度量 ─────────────────────────────────────────────────────
def h1_hurst(dfs, params):
    hs = []
    for df in dfs:
        ctx = make_ctx(df, params["warmup"], state_fns={})
        logtr, logatr, logr, years = build_series(ctx, params["head_drop"])
        hs.append(dfa_hurst(logtr, params["dfa_min_scale"], params["dfa_n_scales"]))
    hs = np.array([h for h in hs if np.isfinite(h)])
    return hs


def h1_cross(dfs, params):
    hs = []
    for df in dfs:
        ctx = make_ctx(df, params["warmup"], state_fns={})
        logtr, logatr, logr, years = build_series(ctx, params["head_drop"])
        hs.append(dfa_hurst(logr, params["dfa_min_scale"], params["dfa_n_scales"]))
    return np.array([h for h in hs if np.isfinite(h)])


def h2_acf(dfs, params):
    vals = []
    for df in dfs:
        ctx = make_ctx(df, params["warmup"], state_fns={})
        logtr, logatr, logr, years = build_series(ctx, params["head_drop"])
        vals.append(acf_lag(logatr, params["acf_lag"]))
    return np.array(vals)


def state_metrics(dfs, params):
    lh1 = hl1 = tot1 = 0
    lh4 = hl4 = tot4 = 0
    runs1 = {"低": [], "高": []}
    runs4 = {"低": [], "高": []}
    for df in dfs[params["tf_list"][0]]:
        ctx = make_ctx(df, params["warmup"], state_fns={})
        logtr, logatr, logr, years = build_series(ctx, params["head_drop"])
        s = z120_states(logatr, params["z_window"])
        s = s[s != "NA"]
        a, b, t = transition_direct(s)
        lh1 += a * t
        hl1 += b * t
        tot1 += t
        r = run_lengths(s)
        runs1["低"].extend(r["低"])
        runs1["高"].extend(r["高"])
    for df in dfs[params["tf_list"][1]]:
        ctx = make_ctx(df, params["warmup"], state_fns={})
        logtr, logatr, logr, years = build_series(ctx, params["head_drop"])
        s = z120_states(logatr, params["z_window"])
        s = s[s != "NA"]
        a, b, t = transition_direct(s)
        lh4 += a * t
        hl4 += b * t
        tot4 += t
        r = run_lengths(s)
        runs4["低"].extend(r["低"])
        runs4["高"].extend(r["高"])
    return {
        "1h": {"lh": lh1 / tot1, "hl": hl1 / tot1, "n": tot1,
               "dur_low": float(np.mean(runs1["低"])), "dur_high": float(np.mean(runs1["高"])),
               "med_low": float(np.median(runs1["低"])), "med_high": float(np.median(runs1["高"]))},
        "4h": {"lh": lh4 / tot4, "hl": hl4 / tot4, "n": tot4,
               "dur_low": float(np.mean(runs4["低"])), "dur_high": float(np.mean(runs4["高"])),
               "med_low": float(np.median(runs4["低"])), "med_high": float(np.median(runs4["高"]))},
    }


def surrogate_acf(dfs, params, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    vals = []
    for df in dfs:
        ctx = make_ctx(df, params["warmup"], state_fns={})
        logtr, logatr, logr, years = build_series(ctx, params["head_drop"])
        for _ in range(params["perm_n"]):
            perm = rng.permutation(logatr)
            vals.append(acf_lag(perm, params["acf_lag"]))
    return np.array(vals)


def by_year_metrics(df, params, years_wanted=(2024, 2025, 2026)):
    ctx = make_ctx(df, params["warmup"], state_fns={})
    logtr, logatr, logr, years = build_series(ctx, params["head_drop"])
    out = {}
    for y in years_wanted:
        m = years == y
        if m.sum() < params["by_year_min_bars"]:
            continue
        out[y] = (dfa_hurst(logtr[m], params["dfa_min_scale"], params["dfa_n_scales"]),
                  acf_lag(logatr[m], params["acf_lag"]))
    return out


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, r, by_year_rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},warmup={},z_window={},acf_lag={},dfa_min_scale={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), p["warmup"], p["z_window"], p["acf_lag"],
            p["dfa_min_scale"], p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 口径: 描述层无入场, 1:1 WR 不适用; 无条件基线=DFA-H(logTR/c): "
        "真实1h中位 {:.3f} vs GBM30种子mean {:.3f} [PASS]; 白噪声探测器 {:.3f} [PASS]; "
        "MIN_N 描述层不适用 (样本bars见RESULTS)".format(
            p["gbm_seeds"], r["h1_1h_med"], g["gbm_mean"], g["wn_h"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 描述层无入场, 无交易含义",
        "",
        "[H1] DFA-H(log TR/close) 各标的聚合 (中位数口径):",
        "  1h: H med={:.3f} [min {:.3f}, max {:.3f}] (n_sym={})".format(
            r["h1_1h_med"], r["h1_1h_min"], r["h1_1h_max"], r["h1_1h_n"]),
        "  4h: H med={:.3f} [min {:.3f}, max {:.3f}] (n_sym={})".format(
            r["h1_4h_med"], r["h1_4h_min"], r["h1_4h_max"], r["h1_4h_n"]),
        "[H1-x] DFA-H(|log-ret|) 交叉验证 (a15 旧口径替代): 1h med={:.3f} | 4h med={:.3f}".format(
            r["h1x_1h_med"], r["h1x_4h_med"]),
        "[H1-null] GBM(logTR/c) 30种子 (gbm_matching 首标参考, 同管线): 1h mean={:.3f} "
        "[min {:.3f}, max {:.3f}]".format(r["gbm_1h_mean"], r["gbm_1h_min"], r["gbm_1h_max"]),
        "  4h mean={:.3f} [min {:.3f}, max {:.3f}]".format(
            r["gbm_4h_mean"], r["gbm_4h_min"], r["gbm_4h_max"]),
        "",
        "[H2] log(ATR/close) ACF@168 (1h): med={:.3f} mean={:.3f} 达标(≥0.5)标的 {}/{} "
        "[min {:.3f}, max {:.3f}]".format(
            r["h2_med"], r["h2_mean"], r["h2_ge5"], r["h2_n"], r["h2_min"], r["h2_max"]),
        "[H2-null] GBM(logATR/c) ACF@168 5种子: mean={:.4f} (ewm核 0.93^168≈0)".format(
            r["h2_gbm"]),
        "[H2-surr] permutation 置换 surrogate ACF@168 ({} 置换/标的): med={:.4f} mean={:.4f}".format(
            p["perm_n"], r["h2_surr_med"], r["h2_surr_mean"]),
        "",
        "[H3] z120 三分位直接转移率 (n 合计): 1h 低→高={:.4f} 高→低={:.4f} (n={}) | "
        "4h 低→高={:.4f} 高→低={:.4f} (n={})".format(
            r["s1_lh"], r["s1_hl"], r["s1_n"], r["s4_lh"], r["s4_hl"], r["s4_n"]),
        "[H4] z120 状态持续 (mean 根): 1h 低={:.1f} 高={:.1f} | 4h 低={:.1f} 高={:.1f}".format(
            r["s1_dur_low"], r["s1_dur_high"], r["s4_dur_low"], r["s4_dur_high"]),
        "[H4-med] z120 状态持续 (median 根): 1h 低={:.1f} 高={:.1f} | 4h 低={:.1f} 高={:.1f}".format(
            r["s1_med_low"], r["s1_med_high"], r["s4_med_low"], r["s4_med_high"]),
        "[对照-历史] a2(旧, 已作废) vol120 z-score 口径: 1h 低持续20根/高15根, 低→高直接0.002 "
        "(仅形状参照, 不作证据)",
        "[设计偏离-标定] 绝对口径标定 (运行前, 证据): 绝对log(TR)对纯GBM ACF@1=0.96 DFA-H=1.40; "
        "ewm(白噪声) H=0.83; GBM log(ATR/close) H=0.84 — GBM null≈0.5 不可达, 故 H1 用 "
        "log(TR/close), H2 用 log(ATR/close) (详见 docstring 设计偏离说明)",
    ]
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(by_year_rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    t0 = time.time()
    dfs_by_tf = load(PARAMS["tf_list"])
    if not dfs_by_tf or not dfs_by_tf.get("1h"):
        print("无数据, 退出")
        return 1

    # GATE 自检 (失败 SystemExit — 违规即停)
    g = gate(dfs_by_tf["1h"][0], PARAMS)
    print(f"[GATE] 白噪声 H={g['wn_h']:.3f} ACF@168={g['wn_acf']:.4f} | "
          f"GBM30种子 log(TR/c) H mean={g['gbm_mean']:.3f} [PASS]", flush=True)

    # H1: 真实 DFA-H + |r| 交叉 + 4h GBM null
    h1_1h = h1_hurst(dfs_by_tf["1h"], PARAMS)
    h1_4h = h1_hurst(dfs_by_tf["4h"], PARAMS)
    h1x_1h = h1_cross(dfs_by_tf["1h"], PARAMS)
    h1x_4h = h1_cross(dfs_by_tf["4h"], PARAMS)
    gbm_4h = np.array([
        dfa_hurst(build_series(make_ctx(gbm_matching(dfs_by_tf["4h"][0], seed=s),
                                        PARAMS["warmup"], state_fns={}),
                               PARAMS["head_drop"])[0],
                  PARAMS["dfa_min_scale"], PARAMS["dfa_n_scales"])
        for s in range(PARAMS["gbm_seeds"])])

    # H2: ACF@168 + GBM null + permutation surrogate
    h2 = h2_acf(dfs_by_tf["1h"], PARAMS)
    h2_gbm = np.array([
        acf_lag(build_series(make_ctx(gbm_matching(dfs_by_tf["1h"][0], seed=s),
                                      PARAMS["warmup"], state_fns={}),
                             PARAMS["head_drop"])[1], PARAMS["acf_lag"])
        for s in range(5)])
    h2_surr = surrogate_acf(dfs_by_tf["1h"], PARAMS, PARAMS["rng_seed"])

    # H3/H4: z120 状态
    sm = state_metrics(dfs_by_tf, PARAMS)

    # BY_YEAR (2024/2025/2026 分年 H 与 ACF)
    year_rows = []
    for tf in PARAMS["tf_list"]:
        agg = {}
        for df in dfs_by_tf[tf]:
            for y, (h, a) in by_year_metrics(df, PARAMS).items():
                agg.setdefault(y, []).append((h, a))
        for y in sorted(agg):
            hs = np.array([v[0] for v in agg[y] if np.isfinite(v[0])])
            ac = np.array([v[1] for v in agg[y] if np.isfinite(v[1])])
            year_rows.append("{} {} H={:.3f} ACF168={:.3f} (n_sym={})".format(
                tf, y, np.median(hs), np.median(ac), len(hs)))

    r = {
        "h1_1h_med": float(np.median(h1_1h)), "h1_1h_min": float(np.min(h1_1h)),
        "h1_1h_max": float(np.max(h1_1h)), "h1_1h_n": int(len(h1_1h)),
        "h1_4h_med": float(np.median(h1_4h)), "h1_4h_min": float(np.min(h1_4h)),
        "h1_4h_max": float(np.max(h1_4h)), "h1_4h_n": int(len(h1_4h)),
        "h1x_1h_med": float(np.median(h1x_1h)), "h1x_4h_med": float(np.median(h1x_4h)),
        "gbm_1h_mean": g["gbm_mean"], "gbm_1h_min": float(np.min(g["gbm_1h"])),
        "gbm_1h_max": float(np.max(g["gbm_1h"])),
        "gbm_4h_mean": float(np.mean(gbm_4h)), "gbm_4h_min": float(np.min(gbm_4h)),
        "gbm_4h_max": float(np.max(gbm_4h)),
        "h2_med": float(np.median(h2)), "h2_mean": float(np.mean(h2)),
        "h2_ge5": int((h2 >= 0.5).sum()), "h2_n": int(len(h2)),
        "h2_min": float(np.min(h2)), "h2_max": float(np.max(h2)),
        "h2_gbm": float(np.mean(h2_gbm)),
        "h2_surr_med": float(np.median(h2_surr)), "h2_surr_mean": float(np.mean(h2_surr)),
        "s1_lh": sm["1h"]["lh"], "s1_hl": sm["1h"]["hl"], "s1_n": sm["1h"]["n"],
        "s4_lh": sm["4h"]["lh"], "s4_hl": sm["4h"]["hl"], "s4_n": sm["4h"]["n"],
        "s1_dur_low": sm["1h"]["dur_low"], "s1_dur_high": sm["1h"]["dur_high"],
        "s4_dur_low": sm["4h"]["dur_low"], "s4_dur_high": sm["4h"]["dur_high"],
        "s1_med_low": sm["1h"]["med_low"], "s1_med_high": sm["1h"]["med_high"],
        "s4_med_low": sm["4h"]["med_low"], "s4_med_high": sm["4h"]["med_high"],
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, r, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
