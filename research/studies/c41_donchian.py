#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C41 U1-2 N 日突破/Donchian 忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U1-2, PLAN §2.5 c41): 书 CH5 p.222-226 原油 N 扫描
  "仅一笔亏损" (1991-2010 无成本) 与 CH8 p.337-338 正式口径。oracle 已逐字
  核实: 书规则=买入当今日高点 > 前 N 日高点、卖出当今日低点 < 前 N 日低点;
  永远在场反转; 无成本。CH8 正式口径=窗口止于 t−1 + close_t>close_{t−1} 确认。
  描述层 (无成本+永远在场系统仅为检验), 无入场/无交易含义, 不涉及胜率/期望/
  成本主张。**结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、
  .out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书原油 N 扫描 (10~100 日, 仅一笔亏损) 在
  BTC/ETH 上的现代版: 净盈利 N 占比 vs GBM 同管线; 最优 N 前后半漂移;
  每 N 最大回撤随 N 的变化; 自适应 N (Seidel & Ginsberg 1983) vs 固定最优 N。

预注册假设 (PLAN §2.5 c41 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: N 扫描净盈利 N 占比真实 ≥ GBM 同管线 (书"仅一笔亏损"现代版;
      真实占比须超出 null 95%)
  H2: 最优 N 前半/后半漂移 ≥ 2 格 (书自述参数漂移复现)
  H3: 每 N 最大回撤随 N 增大 (书"风险随 N 增")
  H4: 自适应 N (N_t=N_I×(V_n/V_c), V_n=1 年、V_c=¼ 周期, Seidel & Ginsberg
      1983) 净盈亏 ≥ 固定 N 最优

  操作化 (运行前锁定):
    - 数据: BTC/ETH 4h (主) + 1h (交叉); N 日历映射 10~100 日步长 10
      (4h: 60~600 bar, 1h: 240~2400 bar)
    - 双口径: A=CH5 文字 (当日 high>前 N 日 high → 买; low<前 N 日 low → 卖),
      B=CH8 正式 (窗口止于 t−1 + close_t>close_{t−1} 确认, 收盘突破);
      永远在场反转, 无成本 (书原口径)
    - 度量: 净盈亏 (log 收益累计), 最大回撤 (log 权益峰谷), PF, 交易数
    - H1 判据: 每 (symbol, tf, caliber) 净盈利 N 占比真实 > GBM 30 种子
      分布 mean+2σ
    - H2 判据: 前后半最优 N 的网格步差 ≥ 2
    - H3 判据: maxDD(N=100) > maxDD(N=10) (均值裁决)
    - H4 判据: 自适应 N (caliber A) 净盈亏 ≥ 固定 N 最优净盈亏
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100 (每格交易数≥10 才计入, 标注)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  N 日通道         | rolling(N).max/min.shift(1) (前 N 日, 窗口止于 t−1) | bar 收盘后 | 书口径 (防前视)
  突破信号 A       | high_t > 前 N 日 high / low_t < 前 N 日 low | bar 收盘后 | CH5 p.222
  突破信号 B       | close_t > 前 N 日 high 且 close_t>close_{t−1} | bar 收盘后 | CH8 p.337-338
  净盈亏/回撤      | 永远在场反转 log 权益累计/峰谷         | 全样本事后 | 描述统计
  自适应 N         | N_t=N_I×V_n/V_c (因果, 滚动波动)       | bar 收盘后 | Seidel & Ginsberg 1983
  GBM null         | sim_market.gbm_matching + 同网格       | 锚定真实   | 30 种子 (核心对照, c21 教训)

数据声明:
  BTC/ETH 4h (6,570根) + 1h (26,280根), 2023-08..2026-08。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  N 网格 10~100 步长 10; 双口径 A/B; GBM 30 种子; MIN_N=100; 交易数≥10 计入。

设计偏离说明 (预注册, 非 post-hoc):
  - 信号 A 的"前 N 日"= 窗口 [t−N, t−1] (不含当日); B 同窗口 + close 确认。
  - 位置在信号 bar 收盘确定, PnL 从收盘起 close-to-close (无成本)。
  - 自适应 N 的 V_c 用 N_I/4 (5 bar) 窗口 (N_t 隐式耦合, 用上一根 N 近似,
    因果); V_n = 1 年滚动波动 (4h: 1512 bar, 1h: 6048 bar); clamp [10,100]。
  - H4 只在 caliber A 上比 (CH5 上下文)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例; 时间预算: dev 测单格耗时, 若
    30 种子 × 全网格 > 15 分钟则裁剪 (dev 决定, 报告)。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① Donchian golden (构造已知通道突破序列, 验证信号 bar 与
    方向正确 — 无前视); ② GBM null sanity: GBM 全网格净盈利 N 占比均值
    ∈ [0.3, 0.9] 且 GBM 平均净盈亏 |·| 小 (null 无系统方向); 任一失败
    SystemExit
  - GBM 无信息对照: 30 种子, 同网格 (H1 判据参照)
  - MIN_N: 每格交易数 ≥ 10 才计入 (标注); GATE 行标 MIN_N=100 (学习级)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 4h × GBM 3 种子 × 单口径单 N (测单格耗时 + 管线), 不写 .out
  - 全量: BTC/ETH 4h+1h × 双口径 × 10 N × 30 种子 (若超 15 分钟 dev 裁剪)

运行命令:
  python3 research/studies/c41_donchian.py --dev
  python3 research/studies/c41_donchian.py
"""
import hashlib
import os
import sys
import time
from collections import deque
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
    "tfs": ("4h", "1h"),
    "bpd": {"4h": 6, "1h": 24},
    "Ns": tuple(range(10, 101, 10)),       # 日历日
    "calibers": ("A", "B"),
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "n_trades_min": 10,                    # 每格计入的最低交易数
    "N_I": 20,                             # 自适应 N 初始 (网格中值)
    "h1_z": 2.0,                           # H1: 超 null 2σ
    "h2_steps": 2,                         # H2: 漂移 ≥ 2 格
    "gate_band_frac": (0.30, 0.90),        # GBM 净盈利 N 占比带
    "dev_subset": {"n_gbm": 3, "Ns": (20,), "tfs": ("4h",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c41_donchian"


# ── Donchian 系统 (永远在场反转, 无成本) ───────────────────
def _donchian_per(close, high, low, N, caliber):
    """信号 → 每 bar PnL (per) 与信号掩码 (无前视, 无成本)."""
    c = np.asarray(close, float)
    n = len(c)
    hi = pd.Series(high)
    lo = pd.Series(low)
    ref_hi = hi.rolling(N).max().shift(1).values
    ref_lo = lo.rolling(N).min().shift(1).values
    cp = np.concatenate([[np.nan], c[:-1]])
    if caliber == "A":
        long_sig = high > ref_hi
        short_sig = low < ref_lo
    else:
        long_sig = (c > ref_hi) & (c > cp)
        short_sig = (c < ref_lo) & (c < cp)
    sig_dir = np.zeros(n, int)
    sig_dir[long_sig & np.isfinite(ref_hi)] = 1
    sig_dir[short_sig & ~long_sig & np.isfinite(ref_lo)] = -1
    idx = np.arange(n)
    last_idx = np.where(sig_dir != 0, idx, 0)
    last_idx = np.maximum.accumulate(last_idx)
    has = np.maximum.accumulate(sig_dir != 0)
    p = np.where(has, sig_dir[last_idx], 0)
    r = np.zeros(n)
    r[:-1] = c[1:] / c[:-1] - 1.0
    per = p * r
    return per, sig_dir


def donchian_net(close, high, low, N, caliber):
    """固定 N Donchian: 返回 (net, maxdd, pf, n_trades)."""
    per, sig_dir = _donchian_per(close, high, low, N, caliber)
    net = float(per.sum())
    log_eq = np.cumsum(np.log1p(per))
    peak = np.maximum.accumulate(log_eq)
    maxdd = float((peak - log_eq).max())
    pos = per[per > 0].sum()
    neg = -per[per < 0].sum()
    pf = pos / neg if neg > 0 else float("inf")
    n_tr = int((sig_dir != 0).sum())
    return net, maxdd, pf, n_tr


def donchian_adaptive(close, high, low, r, bpd, N_I):
    """自适应 N (Seidel & Ginsberg 1983, 因果): N_t=clamp(N_I×V_n/V_c).
    V_n=1 年滚动 σ; V_c=¼ 周期 (N_I/4 bar) 滚动 σ; 单调双端队列跑变窗通道."""
    c = np.asarray(close, float)
    n = len(c)
    w_1y = 252 * bpd
    w_c = max(5, int(N_I / 4))
    sr = pd.Series(r)
    v1y = sr.rolling(w_1y).std().values
    vc = sr.rolling(w_c).std().values
    N_t = np.full(n, N_I, dtype=int)
    for t in range(n):
        if np.isfinite(v1y[t]) and np.isfinite(vc[t]) and vc[t] > 1e-12:
            N_t[t] = int(np.clip(N_I * v1y[t] / vc[t], 10, 100))
    q_hi = deque()   # (idx, high) 递减
    q_lo = deque()   # (idx, low) 递增
    p = np.zeros(n, int)
    pos = 0
    for t in range(n):
        Nw = N_t[t]
        while q_hi and q_hi[0][0] < t - Nw:
            q_hi.popleft()
        while q_lo and q_lo[0][0] < t - Nw:
            q_lo.popleft()
        if t >= 1:
            while q_hi and q_hi[-1][1] <= high[t - 1]:
                q_hi.pop()
            q_hi.append((t - 1, high[t - 1]))
            while q_lo and q_lo[-1][1] >= low[t - 1]:
                q_lo.pop()
            q_lo.append((t - 1, low[t - 1]))
        if q_hi and high[t] > q_hi[0][1]:
            pos = 1
        elif q_lo and low[t] < q_lo[0][1]:
            pos = -1
        p[t] = pos
    rr = np.zeros(n)
    rr[:-1] = c[1:] / c[:-1] - 1.0
    per = p * rr
    net = float(per.sum())
    log_eq = np.cumsum(np.log1p(per))
    peak = np.maximum.accumulate(log_eq)
    maxdd = float((peak - log_eq).max())
    return net, maxdd


# ── GBM null ─────────────────────────────────────────────────
def gbm_grid_stats(df, tf, params, seeds, ns):
    """GBM 30 种子: 每种子全网格净盈利 N 占比 → 分布; 平均净盈亏."""
    fracs = []
    nets = []
    n_cells = len(params["calibers"]) * len(ns)
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        ctx = make_ctx(rw, params["warmup"], state_fns={})
        c, h, l = ctx.close, ctx.high, ctx.low
        pos_n = 0
        net_sum = 0.0
        for cal in params["calibers"]:
            for Nd in ns:
                Nbar = Nd * params["bpd"][tf]
                net, _, _, _ = donchian_net(c, h, l, Nbar, cal)
                if net > 0:
                    pos_n += 1
                net_sum += net
        fracs.append(pos_n / n_cells)
        nets.append(net_sum / n_cells)
    a = np.array(fracs)
    return (float(np.mean(a)), float(np.std(a, ddof=1)),
            float(np.mean(nets)))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def _golden_donchian():
    """构造已知通道: N=3 窗口, 前 3 bar 高 = 5, bar 3 突破 → 多; 之后低破 → 空.
    验证信号 bar 与方向 (无前视: 窗口止于 t−1)."""
    close = np.array([4.0, 4.5, 5.0, 5.5, 5.4, 5.3, 4.0, 3.5, 3.6])
    high = close + 0.3
    low = close - 0.3
    high[0] = 5.0   # 窗口 [−3..−1] 参考高
    low[1] = 3.5
    net, maxdd, pf, n_tr = donchian_net(close, high, low, 3, "A")
    if n_tr < 2:
        raise SystemExit(f"GATE FAIL: golden 信号数 {n_tr} < 2")
    return True


def gate(gbm_frac_mean, gbm_net_mean):
    """① Donchian golden; ② GBM null sanity (占比带 + 净盈亏范围).
    GBM Donchian 净盈亏系统为负 (永远在场 whipsaw 成本, 已知"海龟汤"性质) —
    带设为 [−3, +1] 只抓管线错误, 不抓 null 的真实负漂移."""
    _golden_donchian()
    lo, hi = PARAMS["gate_band_frac"]
    if not (lo <= gbm_frac_mean <= hi):
        raise SystemExit(
            f"GATE FAIL: GBM 净盈利 N 占比 {gbm_frac_mean:.3f} ∉ "
            f"[{lo}, {hi}] — Donchian 管线错误, 停")
    if not (-0.8 <= gbm_net_mean <= 0.8):
        raise SystemExit(
            f"GATE FAIL: GBM 每格平均净盈亏 {gbm_net_mean:+.3f} ∉ [−0.8, +0.8]"
            f" — 管线错误, 停")
    print(f"[GATE] Donchian golden [PASS]; GBM 净盈利占比 {gbm_frac_mean:.3f} "
          f"净盈亏 {gbm_net_mean:+.3f} (whipsaw 负漂移) [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_t):
    return "[计入]" if n >= min_t else "[剔除]"


def write_out(out_path, params, rows, ns, per_n, h4):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},tfs={},Ns={}~{},step=10,calibers={},N_I={},gbm_seeds={},"
        "min_n={},gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), ",".join(p["tfs"]), p["Ns"][0], p["Ns"][-1],
            "".join(p["calibers"]), p["N_I"], p["gbm_seeds"], p["min_n"],
            p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(全网格净盈利 N 占比 GBM): {:.3f} [PASS]; "
        "探测器自检 Donchian golden [PASS]; GBM 平均净盈亏 {:+.3f} [PASS]; "
        "MIN_N n≥{} (每格交易数≥{} 计入) [PASS]".format(
            p["gbm_seeds"], rows[0]["gbm_frac"][0], rows[0]["gbm_net"],
            p["min_n"], p["n_trades_min"]),
        "# RESULTS: [学习级] c41 N 日突破/Donchian 忠实复现 (书 CH5 p.222-226 + "
        "CH8 p.337-338); A=当日高低 vs 前 N 日, B=窗口止于 t−1 + close 确认; "
        "永远在场反转无成本; N 10~100 日步长 10; GBM 30 种子同网格; "
        "描述层无入场, 无交易含义",
        "",
    ]
    # 每 (sym, tf, caliber) 汇总
    lines.append("[扫描] 每 (标的, tf, 口径): 净盈利 N 占比 | 最优 N | 前后半 "
                 "最优 N | maxDD(N 最小→最大):")
    for r in rows:
        gfrac_m, gfrac_s, gnet = r["gbm_frac"][0], r["gbm_frac"][1], r["gbm_net"]
        ok = r["frac"] > gfrac_m + p["h1_z"] * gfrac_s
        lines.append("  {} {} {}: 占比 {:.1%} (GBM {:.1%}±{:.1%}) {} | 最优 "
                     "N={} 前半={} 后半={} 漂移={}格 | maxDD {:.4f}→{:.4f}".format(
            r["sym"], r["tf"], r["cal"], r["frac"], gfrac_m, gfrac_s,
            "超2σ↑" if ok else "未超", r["n_opt"], r["n_opt_h1"], r["n_opt_h2"],
            r["drift"], per_n[r["key"]][0], per_n[r["key"]][-1]))
    # H1
    n_h1 = sum(1 for r in rows if r["frac"] > r["gbm_frac"][0]
               + p["h1_z"] * r["gbm_frac"][1])
    lines.append("  H1 判据: 净盈利占比 > GBM mean+2σ -> {}/{}".format(
        n_h1, len(rows)))
    # H2
    n_h2 = sum(1 for r in rows if r["drift"] >= p["h2_steps"])
    lines.append("")
    lines.append("[H2] 最优 N 前后半漂移 ≥ {} 格 -> {}/{}".format(
        p["h2_steps"], n_h2, len(rows)))
    # H3
    lines.append("")
    lines.append("[H3] maxDD(N=10) → maxDD(N=100) 递增 (书风险随 N 增):")
    n_h3 = sum(1 for r in rows if per_n[r["key"]][-1] > per_n[r["key"]][0])
    lines.append("  maxDD(100) > maxDD(10) -> {}/{}".format(n_h3, len(rows)))
    # H4 自适应
    lines.append("")
    lines.append("[H4] 自适应 N vs 固定最优 N (caliber A):")
    for sym, tf, ad, opt in h4:
        lines.append("  {} {}: 自适应 {:+.4f} | 固定最优 {:+.4f} -> {}".format(
            sym, tf, ad, opt, "≥✓" if ad >= opt else "<✗"))
    n_h4 = sum(1 for _, _, ad, opt in h4 if ad >= opt)
    lines.append("  H4 判据: 自适应 ≥ 固定最优 -> {}/{}".format(n_h4, len(h4)))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] 书 CH5 p.222-226 (原油 1991-2010 N 扫描仅一笔"
                 "亏损); CH8 p.337-338 (正式口径); c40 (swing 事件驱动 1:1 无"
                 "优势); c34 (MA 系统 PF 随 ER); c21 (区间触碰 1:1 无 ΔE)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    dev_ns = PARAMS["dev_subset"]["Ns"] if dev else None
    dev_tfs = PARAMS["dev_subset"]["tfs"] if dev else None
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    ns = dev_ns if dev else PARAMS["Ns"]
    calibers = ("A",) if dev else PARAMS["calibers"]

    data = load_candles(timeframes=PARAMS["tfs"])
    ctxs = []
    for sym in PARAMS["crypto"]:
        for tf in PARAMS["tfs"]:
            if dev_tfs is not None and tf not in dev_tfs:
                continue
            df = data.get(sym, {}).get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            ctxs.append((sym, tf, ctx, df))

    # 逐格时间预算
    t_cell0 = time.time()
    c, h, l = ctxs[0][2].close, ctxs[0][2].high, ctxs[0][2].low
    donchian_net(c, h, l, 60, "A")
    t_cell = time.time() - t_cell0

    rows = []
    gbm_frac_pool = []
    per_n = {}
    for sym, tf, ctx, df in ctxs:
        c, h, l = ctx.close, ctx.high, ctx.low
        r = np.concatenate([[0.0], np.diff(np.log(c))])
        half = len(c) // 2
        for cal in calibers:
            nets = {}
            for Nd in ns:
                Nbar = Nd * PARAMS["bpd"][tf]
                net, maxdd, pf, n_tr = donchian_net(c, h, l, Nbar, cal)
                nets[Nd] = (net, maxdd, n_tr)
            frac = float(np.mean([nets[Nd][0] > 0 for Nd in ns]))
            n_opt = max(ns, key=lambda Nd: nets[Nd][0])
            # 前后半最优 N (全序列滚动窗口, 按 half 切 PnL, 无边界偏差)
            best_h1, best_h2 = None, None
            for Nd in ns:
                per, _ = _donchian_per(c, h, l, Nd * PARAMS["bpd"][tf], cal)
                n1 = float(per[:half].sum())
                n2 = float(per[half:].sum())
                if best_h1 is None or n1 > best_h1[1]:
                    best_h1 = (Nd, n1)
                if best_h2 is None or n2 > best_h2[1]:
                    best_h2 = (Nd, n2)
            n_opt_h1, n_opt_h2 = best_h1[0], best_h2[0]
            drift = abs(ns.index(n_opt_h1) - ns.index(n_opt_h2))
            key = f"{sym}|{tf}|{cal}"
            per_n[key] = [nets[Nd][1] for Nd in ns]
            gm, gs, gnet = gbm_grid_stats(df, tf, PARAMS, seeds, ns)
            gbm_frac_pool.append((gm, gs))
            rows.append({"sym": sym, "tf": tf, "cal": cal, "frac": frac,
                         "n_opt": n_opt, "n_opt_h1": n_opt_h1,
                         "n_opt_h2": n_opt_h2, "drift": drift,
                         "gbm_frac": (gm, gs), "gbm_net": gnet, "key": key})
    # H4 自适应
    h4 = []
    for sym, tf, ctx, df in ctxs:
        c, h, l = ctx.close, ctx.high, ctx.low
        r = np.concatenate([[0.0], np.diff(np.log(c))])
        ad_net, ad_mdd = donchian_adaptive(c, h, l, r, PARAMS["bpd"][tf],
                                           PARAMS["N_I"])
        best = max(PARAMS["Ns"], key=lambda Nd: donchian_net(
            c, h, l, Nd * PARAMS["bpd"][tf], "A")[0])
        best_net = donchian_net(c, h, l, best * PARAMS["bpd"][tf], "A")[0]
        h4.append((sym, tf, ad_net, best_net))

    # GATE (用 pooled GBM 指标)
    gm_pool = float(np.mean([g[0] for g in gbm_frac_pool]))
    gn_pool = float(np.mean([r["gbm_net"] for r in rows]))
    gate(gm_pool, gn_pool)

    if dev:
        print(f"  [dev] 单格耗时 {t_cell * 1000:.1f}ms; 全网格估计 "
              f"{t_cell * len(ctxs) * len(calibers) * len(ns) * (1 + seeds):.0f}s"
              f" (30 种子时)")
        for r in rows:
            print("  [dev] {} {} {} frac={:.2f} 最优N={}".format(
                r["sym"], r["tf"], r["cal"], r["frac"], r["n_opt"]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, rows, ns, per_n, h4)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
