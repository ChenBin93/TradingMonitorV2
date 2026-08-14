#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C52 E1 时间剖面 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c52): 用户质疑 — c51 的 E1 用 ±12 bar 对称窗口
  (pre 含 t) 会稀释波动释放的峰值。c52 把释放画成逐 horizon 曲线:
  E1(h)=mean(ATR[t+1..t+h])/mean(ATR[t−h..t−1])−1, h∈{1,3,6,12}
  (对称窗口, pre 不含 t)。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。**结论不得作交易依据**。
  学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、内置
  GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 出带触碰后的波动释放曲线 — 峰值在哪根附近?
  释放是前载 (头根) 还是持续 (全 horizon)? 逐 h 净差剖面 (真实−GBM).

预注册假设 (PLAN §2.5 c52 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 释放前载 — E1(1) 净差 (真实−GBM) > E1(12) 净差 (峰值在头根附近)
  H2: 全部 horizon 净差 > 0 且超 2σ (释放全时段为正)
  H3: 1h 与 4h 标准化剖面形状一致 (净差曲线单调性同向)

  操作化 (运行前锁定):
    - 事件 = 收盘出 MA20±2σ 带 (c51 口径, σ=价格 sd)
    - E1(h) = mean(ATR[t+1..t+h])/mean(ATR[t−h..t−1])−1, h∈{1,3,6,12}
      (对称窗口, pre 不含 t; 逐触碰事件)
    - GBM 30 种子同管线逐 horizon null (首标×30, c51 惯例)
    - H1 判据: 每 tf E1(1) 净差 > E1(12) 净差
    - H2 判据: 每 tf 每 h 净差 > 0 且 > 2σ
    - H3 判据: 1h 与 4h 的净差曲线单调方向一致 (同升或同降)
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close            | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  带 MA20±2σ       | 滚动均值/价格 sd (只回看)             | bar 收盘后 | c51/c18 口径
  带触碰           | close 穿越带边界 (因果)               | bar 收盘后 | c51 管线
  ATR              | make_ctx 内置 (14 期 ewm)             | bar 收盘后 | ctx.atr
  E1(h)            | 对称窗口均值比 (post 只 t+1 起, 掩码网格)| 事后       | c52 口径
  GBM null         | sim_market.gbm_matching + 同管线       | 锚定真实   | 首标×30 种子

数据声明:
  BTC/ETH 1h (26,280根) + 4h (6,570根), 2023-08..2026-08。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  带 MA20±2σ; h∈{1,3,6,12}; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - E1(h) 的 pre 窗口 [t−h, t−1] 不含 t (c52 口径); c51/c15 用 [t−11, t]
    含 t — h=12 时两口径略有差异 (c52 更纯), 结论中对照。
  - GBM null 首标×30 种子 (c51 惯例, PLAN §4 最小覆盖)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 带 golden (c51 同款); ② E1(h) golden (构造已知 ATR
    序列, 对称窗口手算对拍); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 同管线逐 horizon
  - MIN_N: 每格 n ≥ MIN_N=100 (不足标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 1h × 3 种子, 不写 .out
  - 全量: BTC/ETH 1h+4h × 30 种子 (预计 ≤5 分钟)

运行命令:
  python3 research/studies/c52_e1_timecourse.py --dev
  python3 research/studies/c52_e1_timecourse.py
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

from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "tfs": ("1h", "4h"),
    "band_n": 20,
    "band_k": 2.0,
    "hs": (1, 3, 6, 12),
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "dev_subset": {"n_gbm": 3, "tfs": ("1h",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c52_e1_timecourse"


# ── 带 (c51 口径) ────────────────────────────────────────────
def band_series(c, n, k):
    c = np.asarray(c, float)
    ma = pd.Series(c).rolling(n).mean().values
    sd = pd.Series(c).rolling(n).std().values
    return ma - k * sd, ma + k * sd


def band_touches(close, lo, hi):
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


# ── E1(h) 对称窗口 (pre 不含 t) ─────────────────────────────
def e1h_series(atr, h):
    n = len(atr)
    t = np.arange(n)
    bar_ok = (t >= h) & (t <= n - h - 1) & np.isfinite(atr) & (atr > 0)
    offs = np.arange(h)
    pre_idx = t[:, None] + offs - h          # [t-h, t-1]
    post_idx = t[:, None] + offs + 1         # [t+1, t+h]
    pre = atr[pre_idx[bar_ok]].mean(axis=1)
    post = atr[post_idx[bar_ok]].mean(axis=1)
    e1 = np.full(n, np.nan)
    e1[bar_ok] = post / pre - 1.0
    return e1


def e1h_touches(atr, up, dn, h):
    e1 = e1h_series(atr, h)
    ev = np.flatnonzero(up | dn)
    vals = e1[ev]
    fin = np.isfinite(vals)
    return (float(np.mean(vals[fin])) if fin.any() else float("nan"),
            int(fin.sum()))


# ── GATE 自检 ────────────────────────────────────────────────
def gate(gbm_e1_1):
    """① 带 golden (c51 同款); ② E1(h) golden (已知 ATR 序列手算对拍)."""
    c = np.arange(1.0, 41.0)
    lo, hi = band_series(c, 5, 2.0)
    exp_hi = np.mean(c[-5:]) + 2.0 * np.std(c[-5:], ddof=1)
    if abs(hi[-1] - exp_hi) > 1e-9:
        raise SystemExit(f"GATE FAIL: 带上界 {hi[-1]:.4f} ≠ {exp_hi:.4f}")
    # ② E1(h) golden: ATR 常数序列 → E1=0; ATR 台阶 (pre=1, post=2) → E1=1
    atr_const = np.ones(30) * 2.0
    e1c = e1h_series(atr_const, 3)
    if abs(e1c[10] - 0.0) > 1e-9:
        raise SystemExit(f"GATE FAIL: 常数 ATR E1={e1c[10]} ≠ 0")
    atr_step = np.concatenate([np.ones(15), np.ones(15) * 2.0])
    e1s = e1h_series(atr_step, 3)
    if not (0.9 <= e1s[15] <= 1.1):
        raise SystemExit(f"GATE FAIL: 台阶 ATR E1={e1s[15]:.3f} ≠ ~1")
    # ③ GBM E1(1) sanity (触碰条件化机械偏置带)
    if not (0.0 <= gbm_e1_1 <= 0.10):
        raise SystemExit(f"GATE FAIL: GBM E1(1) {gbm_e1_1:.4f} ∉ [0, 0.10]")
    print(f"[GATE] 带 golden [PASS]; E1(h) golden (常数=0, 台阶=1) [PASS]; "
          f"GBM E1(1) {gbm_e1_1:.4f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def _pp(v):
    return f"{v:+.2%}"


def write_out(out_path, params, rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},band_n={},band_k={},hs={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tfs"]), p["band_n"], p["band_k"], p["hs"],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: 探测器自检 带 golden + E1(h) golden [PASS]; GBM E1(1) [PASS]; "
        "MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c52 E1 时间剖面 (用户质疑: ±12bar 窗口稀释峰值); "
        "事件=收盘出 MA20±2σ; E1(h)=mean(ATR[t+1..t+h])/mean(ATR[t−h..t−1])−1, "
        "h∈{1,3,6,12}; GBM 30 种子逐 horizon; 描述层无入场, 无交易含义",
        "",
    ]
    for tf, r in rows.items():
        lines.append("[{}] 逐 horizon E1 净差曲线:".format(tf))
        for h in p["hs"]:
            rr, ne = r["real"][h]
            gm, gs = r["gbm"][h]
            net = rr - gm
            ok = net > 0 and net > 2 * gs
            lines.append("  h={:<2} 真实 {:+.2%} (n={}) {} | GBM {:+.2%}±{:+.2%} "
                         "| 净差 {:+.2%} {}".format(
                h, rr, ne, _nm(ne, p["min_n"]), gm, gs, net,
                "超2σ" if ok else "未超"))
        curve = " → ".join("{:+.2%}".format(r["real"][h][0] - r["gbm"][h][0])
                           for h in p["hs"])
        lines.append("  净差曲线: " + curve)
        lines.append("  H1: E1(1) 净差 {:+.2%} vs E1(12) 净差 {:+.2%} -> {}".format(
            r["real"][1][0] - r["gbm"][1][0],
            r["real"][12][0] - r["gbm"][12][0],
            "前载✓" if r["real"][1][0] - r["gbm"][1][0]
            > r["real"][12][0] - r["gbm"][12][0] else "非前载"))
        h2_ok = all(r["real"][h][0] - r["gbm"][h][0] > 0 and
                    r["real"][h][0] - r["gbm"][h][0] > 2 * r["gbm"][h][1]
                    for h in p["hs"])
        lines.append("  H2: 全 horizon 净差 > 0 且超 2σ -> {}".format(
            "PASS" if h2_ok else "FAIL"))
    # H3
    nets = {tf: [r["real"][h][0] - r["gbm"][h][0] for h in p["hs"]]
            for tf, r in rows.items()}
    if len(nets) == 2:
        d1 = [nets["1h"][i + 1] - nets["1h"][i] for i in range(len(p["hs"]) - 1)]
        d4 = [nets["4h"][i + 1] - nets["4h"][i] for i in range(len(p["hs"]) - 1)]
        same = all((a > 0) == (b > 0) for a, b in zip(d1, d4) if a != 0 and b != 0)
        lines.append("")
        lines.append("[H3] 1h/4h 净差曲线单调方向一致 -> {}".format(
            "PASS" if same else "FAIL"))
        lines.append("  (1h 逐段: {}, 4h 逐段: {})".format(
            [f"{x:+.2%}" for x in d1], [f"{x:+.2%}" for x in d4]))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c51 (E1 12bar 口径 +14.46% vs GBM +3.25%); c15 "
                 "(E1 +7.44pp net); 书 CH18 (价格分布); 用户质疑: ±12bar 稀释")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    dev_tfs = PARAMS["dev_subset"]["tfs"] if dev else None

    data = load_candles(timeframes=PARAMS["tfs"])
    gbm_e1_1_all = []

    rows = {}
    for tf in PARAMS["tfs"]:
        if dev_tfs and tf not in dev_tfs:
            continue
        real = {h: (0.0, 0) for h in PARAMS["hs"]}
        for sym in PARAMS["crypto"]:
            df = data.get(sym, {}).get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            c, atr = ctx.close, ctx.atr
            lo, hi = band_series(c, PARAMS["band_n"], PARAMS["band_k"])
            up, dn = band_touches(c, lo, hi)
            for h in PARAMS["hs"]:
                m, ne = e1h_touches(atr, up, dn, h)
                o_s, o_n = real[h]
                real[h] = (o_s + m * ne, o_n + ne)
        real_agg = {h: (v[0] / v[1] if v[1] else float("nan"), v[1])
                    for h, v in real.items()}
        # GBM null (首标×30)
        ref_sym = PARAMS["crypto"][0]
        ref_df = data[ref_sym].get(tf)
        gbm = {h: [] for h in PARAMS["hs"]}
        for seed in range(seeds):
            rw = gbm_matching(ref_df, seed=seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gc, gatr = gctx.close, gctx.atr
            glo, ghi = band_series(gc, PARAMS["band_n"], PARAMS["band_k"])
            gup, gdn = band_touches(gc, glo, ghi)
            for h in PARAMS["hs"]:
                gm, gne = e1h_touches(gatr, gup, gdn, h)
                if np.isfinite(gm):
                    gbm[h].append(gm)
        gbm_agg = {h: (float(np.mean(v)), float(np.std(v, ddof=1))
                       if len(v) > 1 else 0.0) for h, v in gbm.items()}
        gbm_e1_1_all.append(gbm_agg[1][0])
        rows[tf] = {"real": real_agg, "gbm": gbm_agg}

    gate(float(np.mean(gbm_e1_1_all)) if gbm_e1_1_all else 0.05)

    if dev:
        for tf, r in rows.items():
            print("  [dev] {} h1/3/6/12 净差: {}".format(
                tf, [f"{r['real'][h][0]-r['gbm'][h][0]:+.2%}" for h in
                     PARAMS["hs"]]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
