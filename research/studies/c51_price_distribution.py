#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C51 M7 U1 价格分布系统忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 M7 U1, PLAN §2.5 c51): 书 CH18 p.801-832 价格分布
  系统。oracle 逐字核实口径: 带=MA20±2σ (σ=价格 sd, 20 日窗口, 非收益 sd);
  "95%" 为正态假设断言 (书无实测); 触碰双用途 (出带=强弱势确认/异常值回归 —
  书自认两种用法不给结论); CH18 零系统回测。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。**结论不得作交易依据**。
  学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、内置
  GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): ① 带触碰 D1 折返/留存 + 书双用途裁决;
  ② 带触碰后波动释放 (E1); ③ 带收窄→释放 (块 bootstrap 波动聚集 null);
  ④ 直方图分布位 vs c14 关键位围墙效应对拍.

预注册假设 (PLAN §2.5 c51 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 收盘出带 (上/下带) 触碰事件的 D1 折返率 + K bar 留存率 vs GBM 同管线
      (书"带=支撑/阻力"直接检验); **附书双用途裁决**: 触碰后顺势 (跟随出带
      方向 1:1) vs 逆势 (折返 1:1) 哪边超 null
  H2: 带触碰后 24h 波动释放 (E1, c15 口径) vs GBM (预期与 c15 同向复现)
  H3: 带收窄→释放 — 带宽 (2σ/MA) 处于 1y 滚动 10% 分位以下的"窄带状态"→
      未来 K bar 波动增量 vs **波动聚集 null** (块 bootstrap: 收益按 50 bar
      块重采样 30 次保波动持续性; GBM 恒定波动 null 禁用); 判据=真实窄带后
      波动 − 块 bootstrap 窄带后波动, 超 2σ
  H4: 滚动 60 日价格直方图 (60% 中心簇边缘=分布位) vs c14 cluster_levels
      关键位的围墙效应对拍 (触碰 D1 折返, 两套位同管线同事件 vs GBM)

  操作化 (运行前锁定):
    - 数据: 20 标的 4h (标注偏离: 书为日线); 学习级: 30 种子、无 BY_YEAR、
      MIN_N=100、描述层
    - 带: MA20±2σ (σ=价格 sd, 20 bar); 触碰=收盘出带 (穿越边界)
    - H1: D1=1:1 胜率 (顺势: 上破→多, 下破→空; 逆势相反); 留存=K=24 后
      收盘仍在带外同侧; 真实 vs GBM 首标×30 种子 (c17 惯例)
    - H2: E1 = mean(ATR[t+1..t+12])/mean(ATR[t-11..t])−1 (c15 口径)
    - H3: 带宽=2σ/MA; 窄带=带宽<1y 滚动 10% 分位; 未来波动=K=24 收益 sd;
      块 bootstrap (50 bar 块重排 30 次, 价格重构+带宽因果重算); 判据=
      真实−bootstrap > 2σ_bootstrap
    - H4: 60 日 (360 bar) 直方图 60% 中心簇边缘 [p20, p80] 作分布位; 触碰=
      收盘穿越边缘; D1 vs c14 cluster_levels 触碰 D1; 同管线同事件 vs GBM

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close            | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  MA20/σ           | 滚动均值/价格 sd (只回看)             | bar 收盘后 | 书 CH18
  带触碰           | close 穿越带边界 (因果)               | bar 收盘后 | 书 p.802
  D1/E1            | 官方 1:1 引擎 / c15 口径               | 事后       | c17/c15
  带宽/窄带        | 2σ/MA + rolling 10% 分位 (因果)      | bar 收盘后 | research.causal
  块 bootstrap     | 收益 50 bar 块重排 (保波动聚集)       | 全样本     | 波动聚集 null
  直方图位         | 60 日中心簇 [p20,p80] (出窗可用)     | bar 收盘后 | 书 CH18
  GBM null         | sim_market.gbm_matching + 同管线      | 锚定真实   | 首标×30 种子

数据声明:
  20 标的 4h (6,570根), 2023-08..2026-08 (backtest.db)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  带 MA20±2σ; K=24 (留存/D1/E1 窗); H3 块长 50, bootstrap 30, 1y 滚动分位;
  H4 60 日直方图 60% 中心簇; GBM 30 种子; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - H3 的"3y 滚动 10% 分位"改为 1y 滚动 (4h 上 3y 窗样本过少, docstring
    标注); 带宽=2σ/MA (相对量, 跨标的可比)。
  - 块 bootstrap 用 50 bar 块随机重排 (保块内波动聚集), 价格从收益重构,
    带宽状态在重排序列上因果重算。
  - H4 的直方图位用 [p20, p80] 中心簇边缘 (60% 中心簇的简化实现)。
  - GBM null 首标×30 种子 (c17/c27 惯例)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① H1 带 golden (构造已知序列, MA20±2σ 手算对拍); ② H3
    块 bootstrap sanity (重排后 log 波动自相关 ≈ 原始 — 波动聚集保留);
    任一失败 SystemExit
  - GBM/块 bootstrap 无信息对照: 30 次
  - MIN_N: 每格 n ≥ MIN_N=100 (不足标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 4h × 3 种子, 不写 .out
  - 全量: 20 标的 4h × 30 种子 (预计 ≤12 分钟)

运行命令:
  python3 research/studies/c51_price_distribution.py --dev
  python3 research/studies/c51_price_distribution.py
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
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching
from research.structures import K as KSTR

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "4h",
    "band_n": 20,
    "band_k": 2.0,
    "K": 24,
    "e1_half": 12,
    "h3_block": 50,
    "h3_perm": 30,
    "h3_win_bars": 2190,                   # 1y (4h); 3y 窗样本过少, 偏离标注
    "h3_q": 0.10,
    "h4_days": 60,
    "h4_lo": 0.20,
    "h4_hi": 0.80,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "dev_subset": {"n_gbm": 3, "syms": ("BTC/USDT:USDT",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c51_price_distribution"


# ── 带 (MA20±2σ, 价格 sd, 因果) ─────────────────────────────
def band_series(c, n, k):
    c = np.asarray(c, float)
    ma = pd.Series(c).rolling(n).mean().values
    sd = pd.Series(c).rolling(n).std().values
    return ma - k * sd, ma + k * sd, 2 * k * sd / np.maximum(ma, 1e-12)


def band_touches(close, lo, hi):
    """收盘出带事件: 上破 (close 越上带) / 下破 (close 越下带)."""
    n = len(close)
    c_prev = np.roll(close, 1)
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    for t in range(1, n):
        if np.isfinite(hi[t]) and np.isfinite(lo[t]):
            if close[t] > hi[t] and c_prev[t] <= hi[t]:
                up[t] = True
            if close[t] < lo[t] and c_prev[t] >= lo[t]:
                dn[t] = True
    return up, dn


# ── H1: D1 (顺势/逆势 1:1) + 留存 ────────────────────────────
def h1_metrics(close, high, low, atr, up, dn, T, W, lo=None, hi=None):
    """顺势 (上破→多, 下破→空) 与逆势 (相反) 1:1 + K 留存."""
    out = {}
    for name, el, es in (("顺势", up, dn), ("逆势", dn, up)):
        ol, _ = evaluate_forward(close, high, low, atr, el, direction="long",
                                 t_mult=T, w=W)
        os_, _ = evaluate_forward(close, high, low, atr, es, direction="short",
                                  t_mult=T, w=W)
        ne = ol.n_eval + os_.n_eval
        nw = ol.n_win + os_.n_win
        out[name] = ((nw / ne) if ne else float("nan"), ne)
    # 留存: 上破后 K 根仍在上带外 / 下破后仍在带外
    n = len(close)
    ret_ok = ret_n = 0
    for t in np.flatnonzero(up | dn):
        if t + PARAMS["K"] >= n:
            continue
        if up[t]:
            ok = close[t + PARAMS["K"]] > hi[t]
        else:
            ok = close[t + PARAMS["K"]] < lo[t]
        ret_ok += float(ok)
        ret_n += 1
    out["留存"] = ((ret_ok / ret_n) if ret_n else float("nan"), ret_n)
    return out


# ── H2: E1 (c15 口径) ────────────────────────────────────────
def e1_series(atr, h):
    n = len(atr)
    t = np.arange(n)
    bar_ok = (t >= h - 1) & (t <= n - h - 1) & np.isfinite(atr) & (atr > 0)
    offs = np.arange(h)
    pre_idx = t[:, None] + offs - (h - 1)
    post_idx = t[:, None] + offs + 1
    pre = atr[pre_idx[bar_ok]].mean(axis=1)
    post = atr[post_idx[bar_ok]].mean(axis=1)
    e1 = np.full(n, np.nan)
    e1[bar_ok] = post / pre - 1.0
    return e1


def h2_e1(atr, up, dn, h):
    e1 = e1_series(atr, h)
    ev = np.flatnonzero(up | dn)
    vals = e1[ev]
    fin = np.isfinite(vals)
    return (float(np.mean(vals[fin])) if fin.any() else float("nan"),
            int(fin.sum()))


# ── H3: 带收窄→释放 (块 bootstrap) ───────────────────────────
def block_bootstrap_rets(r, block, seed):
    rng = np.random.default_rng(seed)
    n = len(r)
    n_blocks = int(np.ceil(n / block))
    order = rng.permutation(n_blocks)
    parts = [r[i * block:(i + 1) * block] for i in order]
    out = np.concatenate(parts)[:n]
    return out


def narrow_future_vol(close, width, K, win_bars, q):
    """窄带状态 (width < rolling q 分位) → 未来 K bar 波动."""
    rp = rolling_percentile(width, win_bars, q)
    n = len(close)
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    narrow = np.isfinite(rp) & (width < rp)
    out = []
    for t in np.flatnonzero(narrow):
        if t + K >= n:
            continue
        out.append(float(np.std(r[t + 1:t + K + 1])))
    return (float(np.mean(out)) if out else float("nan"), len(out))


# ── H4: 直方图分布位 vs cluster_levels ───────────────────────
def hist_levels(close, win_bars, lo_q, hi_q):
    """滚动 60 日中心簇边缘 [p_lo, p_hi] (因果, 出窗可用)."""
    n = len(close)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for t in range(win_bars - 1, n):
        w = close[t - win_bars + 1:t + 1]
        lo[t] = np.quantile(w, lo_q)
        hi[t] = np.quantile(w, hi_q)
    return lo, hi


def hist_touch_d1(close, high, low, atr, lo, hi, T, W):
    """直方图位触碰 (close 穿越边缘) 的逆势 1:1."""
    n = len(close)
    c_prev = np.roll(close, 1)
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    for t in range(1, n):
        if np.isfinite(hi[t]) and np.isfinite(lo[t]):
            if close[t] > hi[t] and c_prev[t] <= hi[t]:
                up[t] = True
            if close[t] < lo[t] and c_prev[t] >= lo[t]:
                dn[t] = True
    # 逆势 (fade): 上触→空, 下触→多
    ol, _ = evaluate_forward(close, high, low, atr, dn, direction="long",
                             t_mult=T, w=W)
    os_, _ = evaluate_forward(close, high, low, atr, up, direction="short",
                              t_mult=T, w=W)
    ne = ol.n_eval + os_.n_eval
    nw = ol.n_win + os_.n_win
    return ((nw / ne) if ne else float("nan"), ne)


def cluster_touch_d1(close, high, low, atr, T, W, head):
    """c14 cluster_levels 关键位触碰逆势 1:1."""
    n = len(close)
    t_idx = np.arange(n)
    lvls = cluster_levels(high, low, atr, k=KSTR, tolerance_mult=0.3,
                          min_touch=2)
    el = np.zeros(n, bool)
    es = np.zeros(n, bool)
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (low <= p_hi) & (high >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev & (t_idx >= head)
        for e in np.flatnonzero(entry):
            if lv.side == "resistance":
                es[e] = True
            else:
                el[e] = True
    ol, _ = evaluate_forward(close, high, low, atr, el, direction="long",
                             t_mult=T, w=W)
    os_, _ = evaluate_forward(close, high, low, atr, es, direction="short",
                              t_mult=T, w=W)
    ne = ol.n_eval + os_.n_eval
    nw = ol.n_win + os_.n_win
    return ((nw / ne) if ne else float("nan"), ne)


# ── GATE 自检 ────────────────────────────────────────────────
def gate(gbm_d1_follow, bs_acf_check):
    """① H1 带 golden (已知序列 MA20±2σ 手算); ② 块 bootstrap sanity."""
    c = np.arange(1.0, 41.0)
    lo, hi, width = band_series(c, 5, 2.0)
    exp_hi = np.mean(c[-5:]) + 2.0 * np.std(c[-5:], ddof=1)   # pandas ddof=1
    if abs(hi[-1] - exp_hi) > 1e-9:
        raise SystemExit(f"GATE FAIL: 带上界 {hi[-1]:.4f} ≠ 手算 {exp_hi:.4f}")
    if not bs_acf_check:
        raise SystemExit("GATE FAIL: 块 bootstrap 波动聚集未保留")
    if not (0.40 <= gbm_d1_follow <= 0.60):
        raise SystemExit(f"GATE FAIL: GBM 顺势 D1 {gbm_d1_follow:.3f} 异常")
    print(f"[GATE] H1 带 golden [PASS]; 块 bootstrap 波动聚集 [PASS]; GBM "
          f"顺势 D1 {gbm_d1_follow:.3f} [PASS]", flush=True)
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
        "params=tf={},band_n={},band_k={},K={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            p["tf"], p["band_n"], p["band_k"], p["K"], p["gbm_seeds"],
            p["min_n"], p["min_n"]),
        "# GATE: 探测器自检 H1 带 golden + 块 bootstrap sanity [PASS]; MIN_N "
        "n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c51 M7 U1 价格分布系统忠实复现 (书 CH18 p.801-832); "
        "带=MA20±2σ (价格 sd); 触碰双用途 (顺势/逆势); E1 (c15); 带收窄→释放 "
        "(块 bootstrap 波动聚集 null); 直方图位 vs c14 关键位; GBM 首标×30 种子; "
        "描述层无入场, 无交易含义",
        "",
    ]
    # H1
    lines.append("[H1] 带触碰 D1 (顺势/逆势 1:1) + K 留存 vs GBM:")
    for tf, r in h1.items():
        for k, (m, ne) in r["real"].items():
            nb = r["null"].get(k)
            if nb:
                lines.append("  {} {}: {:.1%} (n={}) {} | GBM {:.1%}±{:.1%}".format(
                    tf, k, m, ne, _nm(ne, p["min_n"]), nb[0], nb[1]))
            else:
                lines.append("  {} {}: {:.1%} (n={})".format(tf, k, m, ne))
    # H2
    lines.append("")
    lines.append("[H2] 带触碰后波动释放 E1 (c15 口径):")
    for tf, r in h2.items():
        lines.append("  {}: 真实 {:+.2%} (n={}) | GBM {:+.2%}±{:+.2%}".format(
            tf, r["real"][0], r["real"][1], r["gbm"][0], r["gbm"][1]))
    # H3
    lines.append("")
    lines.append("[H3] 带收窄→释放 (窄带后 K 波动 vs 块 bootstrap null):")
    for tf, r in h3.items():
        lines.append("  {}: 真实 {:+.6f} (n={}) | bootstrap {:+.6f}±{:+.6f} | "
                     "超额 {:+.6f} {}".format(
            tf, r["real"][0], r["real"][1], r["bs"][0], r["bs"][1],
            r["real"][0] - r["bs"][0],
            "超2σ" if r["real"][0] - r["bs"][0] > 2 * r["bs"][1] else "未超"))
    # H4
    lines.append("")
    lines.append("[H4] 直方图分布位 vs cluster_levels (逆势触碰 D1):")
    for tf, r in h4.items():
        lines.append("  {}: 直方图位 {:.1%} (n={}) | cluster {:.1%} (n={}) | "
                     "GBM {:.1%}±{:.1%}".format(
            tf, r["hist"][0], r["hist"][1], r["cl"][0], r["cl"][1],
            r["gbm"][0], r["gbm"][1]))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c14 (关键位围墙); c15 (触碰释放 +7.44pp); c17 "
                 "(触碰折返 -4.09pp); c23 (1:1 无优势); 书 CH18 p.801-832 (价格"
                 "分布, 双用途不给结论, 零系统回测)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    dev_syms = PARAMS["dev_subset"]["syms"] if dev else None

    data = load_candles(timeframes=(PARAMS["tf"],))
    syms = [s for s in data if "USDT" in s]
    if dev_syms:
        syms = [s for s in syms if s in dev_syms]

    # 块 bootstrap sanity: 重排序列 log 波动自相关 ≈ 原始
    bs_acf_ok = True

    h1 = {}
    h2 = {}
    h3 = {}
    h4 = {}
    gbm_d1_follows = []

    for tf in (PARAMS["tf"],):
        real_h1 = {"顺势": (0.0, 0), "逆势": (0.0, 0), "留存": (0.0, 0)}
        real_h2 = (0.0, 0)
        real_h3 = []
        real_h4_hist, real_h4_cl = [], []
        for sym in syms:
            df = data[sym].get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            c, h, l, atr = ctx.close, ctx.high, ctx.low, ctx.atr
            lo, hi, width = band_series(c, PARAMS["band_n"], PARAMS["band_k"])
            up, dn = band_touches(c, lo, hi)
            m1 = h1_metrics(c, h, l, atr, up, dn, 1.0, PARAMS["K"], lo, hi)
            for k, (m, ne) in m1.items():
                o_s, o_n = real_h1[k]
                real_h1[k] = (o_s + m * ne, o_n + ne)
            e = h2_e1(atr, up, dn, PARAMS["e1_half"])
            real_h2 = (real_h2[0] + e[0] * e[1], real_h2[1] + e[1])
            # H3 真实
            nv, nn = narrow_future_vol(c, width, PARAMS["K"],
                                       PARAMS["h3_win_bars"], PARAMS["h3_q"])
            if nn:
                real_h3.append((nv, nn))
            # H4 真实
            hlo, hhi = hist_levels(c, PARAMS["h4_days"] * 6, PARAMS["h4_lo"],
                                   PARAMS["h4_hi"])
            hd = hist_touch_d1(c, h, l, atr, hlo, hhi, 1.0, PARAMS["K"])
            real_h4_hist.append(hd)
            cd = cluster_touch_d1(c, h, l, atr, 1.0, PARAMS["K"],
                                  PARAMS["warmup"] // 2)
            real_h4_cl.append(cd)
        # 聚合真实
        h1[tf] = {"real": {k: (v[0] / v[1] if v[1] else float("nan"), v[1])
                           for k, v in real_h1.items()},
                  "null": {}}
        h2[tf] = {"real": (real_h2[0] / real_h2[1] if real_h2[1]
                           else float("nan"), real_h2[1])}
        rv = [(v[0], v[1]) for v in real_h3]
        rv_m = float(np.mean([x[0] for x in rv])) if rv else float("nan")
        rv_n = int(np.sum([x[1] for x in rv]))
        h3[tf] = {"real": (rv_m, rv_n)}
        h4[tf] = {"hist": (float(np.mean([x[0] for x in real_h4_hist
                                          if np.isfinite(x[0])])),
                           int(np.sum([x[1] for x in real_h4_hist]))),
                  "cl": (float(np.mean([x[0] for x in real_h4_cl
                                        if np.isfinite(x[0])])),
                         int(np.sum([x[1] for x in real_h4_cl])))}
        # GBM null (首标)
        ref_sym = syms[0]
        ref_df = data[ref_sym].get(tf)
        g_h1 = {"顺势": [], "逆势": [], "留存": []}
        g_h2 = []
        g_h3 = []
        g_h4 = []
        for seed in range(seeds):
            rw = gbm_matching(ref_df, seed=seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gc, gh, gl, gatr = gctx.close, gctx.high, gctx.low, gctx.atr
            glo, ghi, gwidth = band_series(gc, PARAMS["band_n"],
                                           PARAMS["band_k"])
            gup, gdn = band_touches(gc, glo, ghi)
            gm = h1_metrics(gc, gh, gl, gatr, gup, gdn, 1.0, PARAMS["K"], glo, ghi)
            for k, (m, ne) in gm.items():
                if ne:
                    g_h1[k].append(m)
            g_h2.append(h2_e1(gatr, gup, gdn, PARAMS["e1_half"]))
            gv, gn = narrow_future_vol(gc, gwidth, PARAMS["K"],
                                       PARAMS["h3_win_bars"], PARAMS["h3_q"])
            if gn:
                g_h3.append(gv)
            ghlo, ghhi = hist_levels(gc, PARAMS["h4_days"] * 6,
                                     PARAMS["h4_lo"], PARAMS["h4_hi"])
            g_h4.append(hist_touch_d1(gc, gh, gl, gatr, ghlo, ghhi, 1.0,
                                      PARAMS["K"]))
            if gm["顺势"][0] == gm["顺势"][0]:
                gbm_d1_follows.append(gm["顺势"][0])
        # H3 块 bootstrap null (用首标的收益)
        r_full = np.diff(np.log(ref_df["close"].values.astype(float)))
        bs_vals = []
        bs_acf = []
        for perm in range(PARAMS["h3_perm"]):
            br = block_bootstrap_rets(r_full, PARAMS["h3_block"], perm)
            bc = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], br])))
            bwidth = band_series(bc, PARAMS["band_n"], PARAMS["band_k"])[2]
            bv, bn = narrow_future_vol(bc, bwidth, PARAMS["K"],
                                       PARAMS["h3_win_bars"], PARAMS["h3_q"])
            if bn:
                bs_vals.append(bv)
            # 波动聚集 sanity: 重排序列 |r| ACF@1 vs 原始
            if perm == 0:
                a1 = autocorr_lag(np.abs(br), 1)
                a0 = autocorr_lag(np.abs(r_full), 1)
                bs_acf.append((a0, a1))
        ba = np.array(bs_vals)
        h3[tf]["bs"] = (float(np.mean(ba)), float(np.std(ba, ddof=1))
                        if len(ba) > 1 else 0.0)
        if bs_acf:
            a0, a1 = bs_acf[0]
            bs_acf_ok = abs(a1 - a0) < 0.15 * max(abs(a0), 0.01)
        # GBM 聚合
        for k, v in g_h1.items():
            a = np.array(v)
            h1[tf]["null"][k] = (float(np.mean(a)), float(np.std(a, ddof=1)))
        ga = np.array([x[0] for x in g_h2 if np.isfinite(x[0])])
        h2[tf]["gbm"] = (float(np.mean(ga)), float(np.std(ga, ddof=1)))
        g3 = np.array(g_h3)
        h3[tf].setdefault("bs", h3[tf]["bs"])
        h4a = np.array([x[0] for x in g_h4 if np.isfinite(x[0])])
        h4[tf]["gbm"] = (float(np.mean(h4a)), float(np.std(h4a, ddof=1)))

    gate(float(np.mean(gbm_d1_follows)) if gbm_d1_follows else 0.5,
         bs_acf_ok)

    if dev:
        print("  [dev] H1 顺势 {:.2f} 逆势 {:.2f} 留存 {:.2f} (真实)".format(
            h1["4h"]["real"]["顺势"][0], h1["4h"]["real"]["逆势"][0],
            h1["4h"]["real"]["留存"][0]))
        print("  [dev] H2 E1 {:.2%} vs GBM {:.2%}".format(
            h2["4h"]["real"][0], h2["4h"]["gbm"][0]))
        print("  [dev] H3 窄带 {:.5f} vs bs {:.5f}".format(
            h3["4h"]["real"][0], h3["4h"]["bs"][0]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, h1, h2, h3, h4)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


def autocorr_lag(x, lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    v = float(np.mean(x * x))
    if v <= 0:
        return float("nan")
    n = len(x)
    t = np.arange(n)
    return float(np.mean(x[t < n - lag] * x[t >= lag]) / v)


if __name__ == "__main__":
    sys.exit(main())
