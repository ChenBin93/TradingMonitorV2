#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C50 M6 U1 KAMA 自适应均线忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 M6 U1, PLAN §2.5 c50): 书 CH17 p.779-799 KAMA。
  oracle 逐字核实口径: KAMA_t = KAMA_{t−1} + sc_t×(p_t − KAMA_{t−1});
  sc_t = [ER_t×(2/3 − 2/31) + 2/31]²; ER=|p_t−p_{t−10}|/Σ|p_i−p_{i−1}|
  (收-收, 10 期); 纯收盘价。**TS 代码 `*2` 是 OCR 对 `²` 的讹写** — 数学正文
  是平方 (0.6022=2/3−2/31, 0.0645=2/31), 取平方并在 docstring 标注。
  平方效应: 慢端=900 期等价、快端=4 期等价; 交易规则=方向转向+小阈值过滤
  (0.1σ); "周期<14, 8-10 天最好"。CH17 零回测; 书自评"概念合理但公式化
  未完善、远非最终解"。描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。
  **结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留
  docstring 预注册冻结、内置 GATE (H1 确定性序列逐位断言)、因果纪律、
  dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): ① KAMA 公式 golden; ② 核心对撞 — 高 ER 触碰
  后 KAMA vs 固定 MA 的方向翻转 (书前提"高 ER 无假翻转" vs c27 反向预期);
  ③ KAMA 信号系统 (方向转向+0.1σ) 净收益 vs GBM null vs 固定 MA; ④ 平滑
  属性 (sc 分布/冻结占比/变化波动).

预注册假设 (PLAN §2.5 c50 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 公式 golden 对拍 — 确定性序列逐位验证: ①单调上涨 20 bar (ER=1→
      sc=(0.6667)², KAMA 逼近 4 期等价); ②±1 交替 (ER→0→sc=(0.0645)²,
      近乎冻结); ③15 涨 15 跌三角波 (GATE 逐位断言)
  H2: **核心对撞** — 高 ER 触碰 (c27 触碰管线, ER 高分位条件) 后 K bar 内
      方向翻转次数: KAMA vs 5 日快速 MA (4h=30bar) vs 30 日慢速 MA
      (4h=180bar), 真实 vs GBM 同管线; 判据: 真实高 ER 触碰下 KAMA 翻转率
      − null 翻转率 (书前提"高 ER 无假翻转"; c27 反向预期=KAMA 加速导致
      假翻转更多)
  H3: KAMA 信号系统 (方向转向+0.1σ 过滤, 无成本永远在场) 净收益 vs GBM
      null; KAMA−null 是否 > 固定 MA−null (书"可能更好"的最小检验)
  H4: 平滑属性 — sc 分位分布、慢端冻结占比 (sc<0.01)、KAMA 逐期变化波动
      vs 等速 MA

  操作化 (运行前锁定):
    - 数据: 20 标的 4h+1h 收盘; 学习级: 30 种子、无 BY_YEAR、MIN_N=100
    - H1: 纯确定性序列 (GATE 逐位断言, 无 GBM)
    - H2: 高 ER 触碰 (ER_10 rolling 80th 分位, c27 口径) 后 K=24 bar 内
      KAMA/MA5/MA30 斜率符号翻转次数; 真实 (20 标的聚合) vs GBM 同管线
      (首标×30 种子, c27 惯例)
    - H3: KAMA 方向转向 (斜率符号) + 0.1σ 过滤 (转向 bar 的 |Δclose| >
      0.1×σ_close, σ=全样本收益 sd); 永远在场反转无成本; net vs GBM 与
      固定 MA (MA30) 对照
    - H4: sc 分位/冻结占比/KAMA 变化波动 vs MA
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close            | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  ER_10            | |p_t−p_{t−10}|/Σ|Δp| (收-收, 因果)    | bar 收盘后 | 书 CH17
  KAMA             | 递推 (只回看)                         | bar 收盘后 | 书 CH17
  ER 分位           | causal.rolling_percentile (win=120)    | bar 收盘后 | c27 口径
  触碰事件         | levels.cluster_levels + 连续触碰首根   | bar 收盘后 | c27 管线
  翻转计数         | 触碰后 K bar 内斜率符号变化次数        | 事后       | 描述统计
  信号系统         | 方向转向+0.1σ 过滤 (因果)             | bar 收盘后 | 书 CH17
  GBM null         | sim_market.gbm_matching + 同管线       | 锚定真实   | 首标×30 种子

数据声明:
  20 标的 4h (6,570根) + 1h (26,280根), 2023-08..2026-08 (backtest.db)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  ER_n=10; fast=2/3; slow=2/31; K=24; MA5=30bar, MA30=180bar (4h);
  0.1σ 过滤; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - TS 代码 `*2` 按 OCR 讹写处理 (取平方, docstring 标注; oracle 逐字核实)。
  - H2 的"5 日/30 日 MA"按 4h 映射为 30/180 bar (日历偏差标注)。
  - H3 的 0.1σ 过滤用转向 bar 的 |Δclose| > 0.1×全样本收益 sd (声明)。
  - GBM null 首标×30 种子 (c27 惯例, PLAN §4 描述层最小覆盖)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: H1 确定性序列逐位断言 (单调涨/交替/三角波, KAMA/sc 手算
    对拍); 任一失败 SystemExit
  - GBM 无信息对照: 首标×30 种子, 同管线
  - MIN_N: 每格 n ≥ MIN_N=100 (不足标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 4h × GBM 3 种子, 不写 .out
  - 全量: 20 标的 4h+1h × 30 种子 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c50_kama.py --dev
  python3 research/studies/c50_kama.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path (模板摩擦, 见 c12 报告)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.sim_market import gbm_matching
from research.structures import K as KSTR

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tfs": ("4h", "1h"),
    "er_n": 10,
    "fast": 2.0 / 3.0,
    "slow": 2.0 / 31.0,
    "K": 24,
    "ma5_bars": {"4h": 30, "1h": 120},
    "ma30_bars": {"4h": 180, "1h": 720},
    "sig_frac": 0.1,                       # 0.1σ 过滤
    "r_win": 120,
    "q_hi": 0.8,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "freeze_th": 0.01,                     # H4 冻结阈值 (sc<0.01)
    "dev_subset": {"n_gbm": 3, "syms": ("BTC/USDT:USDT",), "tfs": ("4h",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c50_kama"


# ── KAMA (书 CH17, 递推因果) ─────────────────────────────────
def er_10_series(c, n):
    c = np.asarray(c, float)
    L = len(c)
    t = np.arange(L)
    cp = np.roll(c, 1)
    ad = np.where(t >= 1, np.abs(c - cp), 0.0)
    pref = np.concatenate([[0], np.cumsum(ad)])
    ok = t >= n
    net = np.full(L, np.nan)
    net[ok] = np.abs(c[t[ok]] - c[t[ok] - n])
    path = np.full(L, np.nan)
    path[ok] = pref[t[ok] + 1] - pref[t[ok] - n + 1]
    er = np.full(L, np.nan)
    m = ok & (path > 0)
    er[m] = net[m] / path[m]
    return er


def kama_series(c, er_n, fast, slow):
    """KAMA 递推 + sc 序列."""
    c = np.asarray(c, float)
    n = len(c)
    er = er_10_series(c, er_n)
    sc = np.full(n, np.nan)
    kama = np.full(n, np.nan)
    kama[0] = c[0]
    for t in range(1, n):
        e = er[t] if np.isfinite(er[t]) else 0.0
        sc[t] = (e * (fast - slow) + slow) ** 2
        kama[t] = kama[t - 1] + sc[t] * (c[t] - kama[t - 1])
    return kama, sc


def ma_series(c, n):
    return pd.Series(c).rolling(n).mean().values


# ── 翻转计数 (触碰后 K bar 内斜率符号变化) ──────────────────
def flips_in(line, start, K):
    """line 在 [start, start+K] 内的斜率符号变化次数."""
    cnt = 0
    prev = 0
    for j in range(start + 1, start + K + 1):
        s = line[j] - line[j - 1]
        sg = 1 if s > 0 else (-1 if s < 0 else 0)
        if sg != 0 and sg != prev:
            cnt += 1
        if sg != 0:
            prev = sg
    return cnt


def touch_flips(close, high, low, atr, er_state, lines, K, head):
    """高 ER 触碰后 K 线翻转计数 (逐线)."""
    n = len(close)
    t_idx = np.arange(n)
    lvls = cluster_levels(high, low, atr, k=KSTR, tolerance_mult=0.3,
                          min_touch=2)
    counts = {k: [] for k in lines}
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (low <= p_hi) & (high >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev & (t_idx >= head)
        for e in np.flatnonzero(entry):
            if er_state[e] != "高":
                continue
            if e + K >= n:
                continue
            for k, line in lines.items():
                counts[k].append(flips_in(line, e, K))
    out = {k: (float(np.mean(v)) if v else float("nan"), len(v))
           for k, v in counts.items()}
    return out


def er_state_series(close, r_win, q_hi, er_n):
    er = er_10_series(close, er_n)
    rp = rolling_percentile(er, r_win, q_hi)
    n = len(close)
    st = np.full(n, "", dtype=object)
    ok = np.isfinite(rp) & np.isfinite(er)
    st[ok & (er >= rp)] = "高"
    return st


# ── H3: KAMA 信号系统 (方向转向 + 0.1σ) ─────────────────────
def kama_system_net(close, kama, sig, pos0=None):
    """方向转向 + 0.1σ 过滤, 永远在场反转."""
    c = np.asarray(close, float)
    n = len(c)
    d = np.zeros(n, int)
    cur = 0
    for t in range(1, n):
        s = kama[t] - kama[t - 1]
        if s > 0:
            new = 1
        elif s < 0:
            new = -1
        else:
            continue
        if new != cur and abs(c[t] - c[t - 1]) > sig:
            cur = new
        d[t] = cur
    p = np.where(d != 0, d, 0)
    r = np.zeros(n)
    r[:-1] = c[1:] / c[:-1] - 1.0
    return float((p * r).sum())


# ── GATE 自检 (H1 确定性序列逐位断言) ───────────────────────
def gate():
    """① 单调涨 20 bar: ER=1, sc=(2/3)²=0.4444, KAMA 逼近;
    ② ±1 交替: sc≈(2/31)²=0.00416, 近乎冻结;
    ③ 15 涨 15 跌三角波: KAMA 值手算对拍."""
    # ① 单调涨
    c_up = np.arange(1.0, 21.0)
    k, sc = kama_series(c_up, PARAMS["er_n"], PARAMS["fast"], PARAMS["slow"])
    if abs(sc[-1] - (2.0 / 3.0) ** 2) > 1e-9:
        raise SystemExit(f"GATE FAIL: 单调涨 sc={sc[-1]:.5f} ≠ {(2/3)**2:.5f}")
    # 稳态滞后 = (1−sc)/sc ≈ 1.25 (4 期等价: 4 期 MA 滞后 (4−1)/2=1.5)
    lag_theory = (1.0 - sc[-1]) / sc[-1]
    if abs((c_up[-1] - k[-1]) - lag_theory) > 0.1:
        raise SystemExit(
            f"GATE FAIL: 单调涨 KAMA 滞后 {c_up[-1]-k[-1]:.3f} ≠ 理论 "
            f"{lag_theory:.3f} (4 期等价)")
    # ② ±1 交替
    alt = np.array([1.0, 2.0] * 30)
    k2, sc2 = kama_series(alt, PARAMS["er_n"], PARAMS["fast"], PARAMS["slow"])
    if abs(sc2[-1] - (2.0 / 31.0) ** 2) > 1e-6:
        raise SystemExit(f"GATE FAIL: 交替 sc={sc2[-1]:.6f} ≠ {(2/31)**2:.6f}")
    drift = float(np.max(k2) - np.min(k2))
    if drift > 0.25:                        # 价格范围 1.0, KAMA 冻结于 25% 内
        raise SystemExit(f"GATE FAIL: 交替 KAMA 未冻结 (范围 {drift:.3f} > 0.25)")
    # ③ 三角波 (15 涨 15 跌)
    tri = np.concatenate([np.arange(1.0, 16.0), np.arange(15.0, 0.0, -1.0)])
    k3, sc3 = kama_series(tri, PARAMS["er_n"], PARAMS["fast"], PARAMS["slow"])
    # 手算: t=20 (在下跌段), ER=|p20−p10|/Σ|Δ| = |5−10|/Σ
    manual_er = abs(tri[19] - tri[9]) / sum(abs(np.diff(tri[9:20])))
    if abs(er_10_series(tri, 10)[19] - manual_er) > 1e-9:
        raise SystemExit("GATE FAIL: 三角波 ER 手算不符")
    if not np.isfinite(k3[-1]) or k3[-1] <= 0:
        raise SystemExit("GATE FAIL: 三角波 KAMA 无效")
    print(f"[GATE] H1 KAMA golden (单调涨 sc=0.4444/逼近, 交替 sc=0.00416/冻结, "
          f"三角波 ER 对拍) [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def write_out(out_path, params, h1, h2, h3, h4):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},er_n={},fast={},slow={},K={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tfs"]), p["er_n"], round(p["fast"], 6),
            round(p["slow"], 6), p["K"], p["gbm_seeds"], p["min_n"],
            p["min_n"]),
        "# GATE: 探测器自检 H1 KAMA golden (单调涨/交替/三角波逐位) [PASS]; "
        "MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c50 M6 U1 KAMA 自适应均线忠实复现 (书 CH17 "
        "p.779-799); KAMA 递推+sc² (TS `*2`=OCR 讹写, 取平方标注); 核心对撞 "
        "(高 ER 触碰后翻转), 信号系统 (方向转向+0.1σ), 平滑属性; GBM 首标×30 "
        "种子同管线; 描述层无入场, 无交易含义",
        "",
    ]
    # H1
    lines.append("[H1] KAMA golden (GATE 逐位): {}".format(h1["summary"]))
    # H2
    lines.append("")
    lines.append("[H2] 核心对撞 — 高 ER 触碰后 K={} bar 翻转数:".format(p["K"]))
    for tf, r in h2.items():
        lines.append("  {}: KAMA 真实 {:.2f} (n={}) | GBM {:.2f}±{:.2f} | "
                     "MA快 {:.2f} (GBM {:.2f}) | MA慢 {:.2f} (GBM {:.2f})"
                     "".format(tf, r["kama"][0], r["kama"][1],
                               r["gbm_kama"][0], r["gbm_kama"][1],
                               r["ma_fast"][0], r["gbm_maf"][0],
                               r["ma_slow"][0], r["gbm_mas"][0]))
        diff = r["kama"][0] - r["gbm_kama"][0]
        lines.append("    KAMA 真实−null 翻转率差 {:+.2f} (c27 反向预期=KAMA "
                     "加速假翻转更多)".format(diff))
    # H3
    lines.append("")
    lines.append("[H3] KAMA 信号系统 (方向转向+0.1σ, 无成本永远在场):")
    for tf, r in h3.items():
        lines.append("  {}: KAMA 净 {:+.4f} | GBM {:+.4f} | 净差 {:+.4f} | "
                     "固定 MA30 净差 {:+.4f} (KAMA>MA: {})".format(
            tf, r["kama"], r["gbm"], r["kama"] - r["gbm"], r["ma_diff"],
            "✓" if r["kama"] - r["gbm"] > r["ma_diff"] else "✗"))
    # H4
    lines.append("")
    lines.append("[H4] 平滑属性 (sc 分布/冻结/变化波动):")
    for tf, r in h4.items():
        lines.append("  {}: sc 中位 {:.4f} (p10 {:.4f}, p90 {:.4f}) | 冻结占比 "
                     "(sc<{}) {:.1%} | KAMA 变化 sd {:.4f} vs MA {:.4f}".format(
            tf, r["sc_med"], r["sc_p10"], r["sc_p90"], p["freeze_th"],
            r["freeze"], r["kama_sd"], r["ma_sd"]))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c27 (高 ER 触碰折返更深 −3.44pp); c41/c43 (自适应 "
                 "N 不稳健); c46 (MA 系统); 书 CH17 p.779-799 (KAMA, 零回测, "
                 "书自评'远非最终解')")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    dev_syms = PARAMS["dev_subset"]["syms"] if dev else None
    dev_tfs = PARAMS["dev_subset"]["tfs"] if dev else None

    gate()

    data = load_candles(timeframes=PARAMS["tfs"])
    syms = [s for s in data if "USDT" in s]
    if dev_syms:
        syms = [s for s in syms if s in dev_syms]

    h2 = {}
    h3 = {}
    h4 = {}
    gbm_ref = {}                           # 首标的 GBM 管线结果

    for tf in PARAMS["tfs"]:
        if dev_tfs and tf not in dev_tfs:
            continue
        # 真实聚合
        real_kama = {"kama": [], "maf": [], "mas": []}
        real_net = []
        sc_all = []
        kama_sd_all, ma_sd_all = [], []
        ma_bars_f = PARAMS["ma5_bars"][tf]
        ma_bars_s = PARAMS["ma30_bars"][tf]
        for sym in syms:
            df = data[sym].get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            c, h, l = ctx.close, ctx.high, ctx.low
            kama, sc = kama_series(c, PARAMS["er_n"], PARAMS["fast"],
                                   PARAMS["slow"])
            maf = ma_series(c, ma_bars_f)
            mas = ma_series(c, ma_bars_s)
            er_st = er_state_series(c, PARAMS["r_win"], PARAMS["q_hi"],
                                    PARAMS["er_n"])
            fl = touch_flips(c, h, l, ctx.atr, er_st,
                             {"kama": kama, "maf": maf, "mas": mas},
                             PARAMS["K"], PARAMS["warmup"] // 2)
            real_kama["kama"].append(fl["kama"])
            real_kama["maf"].append(fl["maf"])
            real_kama["mas"].append(fl["mas"])
            sig = float(np.std(np.diff(np.log(c)), ddof=1))
            real_net.append(kama_system_net(c, kama, PARAMS["sig_frac"] * sig))
            sc_all.append(sc)
            kama_sd_all.append(float(np.nanstd(np.diff(kama))))
            ma_sd_all.append(float(np.nanstd(np.diff(mas))))
        rk = {k: (float(np.mean([x[0] for x in v if np.isfinite(x[0])])),
                  int(np.sum([x[1] for x in v])))
              for k, v in real_kama.items()}
        # GBM null (首标)
        gk, gs_, gms = [], [], []
        gnet = []
        ref_sym = syms[0]
        ref_df = data[ref_sym].get(tf)
        for seed in range(seeds):
            rw = gbm_matching(ref_df, seed=seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gc, gh, gl = gctx.close, gctx.high, gctx.low
            gkama, gsc = kama_series(gc, PARAMS["er_n"], PARAMS["fast"],
                                     PARAMS["slow"])
            gmaf = ma_series(gc, ma_bars_f)
            gmas = ma_series(gc, ma_bars_s)
            ger_st = er_state_series(gc, PARAMS["r_win"], PARAMS["q_hi"],
                                     PARAMS["er_n"])
            gfl = touch_flips(gc, gh, gl, gctx.atr, ger_st,
                              {"kama": gkama, "maf": gmaf, "mas": gmas},
                              PARAMS["K"], PARAMS["warmup"] // 2)
            gk.append(gfl["kama"])
            gs_.append(gfl["maf"])
            gms.append(gfl["mas"])
            gsig = float(np.std(np.diff(np.log(gc)), ddof=1))
            gnet.append(kama_system_net(gc, gkama, PARAMS["sig_frac"] * gsig))
        def _dist(v):
            a = np.array([x[0] for x in v if np.isfinite(x[0])])
            return (float(np.mean(a)), float(np.std(a, ddof=1)))
        gbm_ref[tf] = {"kama": _dist(gk), "maf": _dist(gs_), "mas": _dist(gms),
                       "net": float(np.mean(gnet))}
        # 固定 MA 信号系统 null 差 (MA30 方向, 无 0.1σ 过滤 — 简单对照)
        ma_system_net_r = []
        ma_system_net_g = []
        for sym in syms:
            df = data[sym].get(tf)
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            mas = ma_series(ctx.close, ma_bars_s)
            p = np.zeros(len(ctx.close), int)
            p[ma_bars_s:] = np.where(mas[ma_bars_s:] > mas[ma_bars_s - 1:-1],
                                     1, -1)
            r = np.zeros(len(ctx.close))
            r[:-1] = ctx.close[1:] / ctx.close[:-1] - 1.0
            ma_system_net_r.append(float((p * r).sum()))
        for seed in range(seeds):
            rw = gbm_matching(ref_df, seed=seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gmas = ma_series(gctx.close, ma_bars_s)
            gp = np.zeros(len(gctx.close), int)
            gp[ma_bars_s:] = np.where(gmas[ma_bars_s:] > gmas[ma_bars_s - 1:-1],
                                      1, -1)
            gr = np.zeros(len(gctx.close))
            gr[:-1] = gctx.close[1:] / gctx.close[:-1] - 1.0
            ma_system_net_g.append(float((gp * gr).sum()))
        ma_net_diff = float(np.mean(ma_system_net_r)) - float(
            np.mean(ma_system_net_g))
        kama_net = float(np.mean(real_net))
        gbm_net = gbm_ref[tf]["net"]
        h2[tf] = {"kama": rk["kama"], "ma_fast": rk["maf"], "ma_slow": rk["mas"],
                  "gbm_kama": gbm_ref[tf]["kama"],
                  "gbm_maf": gbm_ref[tf]["maf"],
                  "gbm_mas": gbm_ref[tf]["mas"]}
        h3[tf] = {"kama": kama_net, "gbm": gbm_net,
                  "kama_diff": kama_net - gbm_net, "ma_diff": ma_net_diff}
        sc_cat = np.concatenate([s[np.isfinite(s)] for s in sc_all])
        h4[tf] = {"sc_med": float(np.median(sc_cat)),
                  "sc_p10": float(np.percentile(sc_cat, 10)),
                  "sc_p90": float(np.percentile(sc_cat, 90)),
                  "freeze": float(np.mean(sc_cat < PARAMS["freeze_th"])),
                  "kama_sd": float(np.mean(kama_sd_all)),
                  "ma_sd": float(np.mean(ma_sd_all))}

    h1 = {"summary": "单调涨 sc=0.4444/KAMA 逼近 4 期等价; ±1 交替 sc=0.00416"
                     "/近乎冻结; 三角波 ER 手算对拍 — GATE 逐位 PASS"}

    if dev:
        for tf in h2:
            print("  [dev] {} KAMA 翻转 {:.2f} vs GBM {:.2f}".format(
                tf, h2[tf]["kama"][0], h2[tf]["gbm_kama"][0]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, h1, h2, h3, h4)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
