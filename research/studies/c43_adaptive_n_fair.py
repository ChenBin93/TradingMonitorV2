#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C43 自适应 N 的事前基准对比 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c43): c41 的 H4 判据"自适应 N ≥ 事后最优固定 N"
  被用户指出违反不可知性 (H2 已证最优 N 漂移, 事后最优=千里眼基准, 书 CH21
  自己反对这种比法)。c43 用公平基准重比: 走前法 (前半选 N → 后半比) + 经典
  N=20 日零优化对照 + GBM null 净差修正 (海龟汤: 永远在场反转 null 为负)。
  描述层 (无成本+永远在场系统仅为检验), 无入场/无交易含义, 不涉及胜率/期望/
  成本主张。**结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、
  .out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 自适应 N (Seidel & Ginsberg 1983) 是否真的
  有用? c41 H4 用事后最优固定 N 作基准 (不可知论违规) — c43 改用可实施基准:
  ① 走前法 (前半样本选最优 N, 分割点可知) → 后半比自适应 vs 该事前 N;
  ② 经典固定 N=20 日 (Donchian 4 周默认, 零优化) 全样本; ③ GBM null 净差
  修正 (海龟汤: 永远在场反转 null 为负, 比净差)。

预注册假设 (PLAN §2.5 c43 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 后半样本自适应 N 净盈亏 ≥ 前半最优固定 N (可实施性公平对比)
  H2: 自适应 N 净盈亏 ≥ N=20 日固定
  H3: 自适应 N 相对其 GBM null 的超额 ≥ 固定 N 相对其 null 的超额
      (海龟汤修正: 永远在场反转 null 为负, 比净差)

  操作化 (运行前锁定):
    - 数据: BTC/ETH 1h+4h, CH5 口径 A (c41 复现成功的口径)
    - 自适应 N: N_t=clamp(N_I×V_n/V_c, 10, 100); V_n=1 年滚动 σ, V_c=¼ 周期
      (N_I/4 bar) 滚动 σ; 全因果只回看; N_I=20 (书默认映射)
    - 走前法: 分割点=样本中点 (可知), 前半 N 扫描 (10~100 步长 10) 选最优
      固定 N → 后半只跑两个系统 (该事前 N vs 自适应 N)
    - 基准 2: 自适应 N vs 固定 N=20 日 (Donchian 4 周默认, 零优化) 全样本
    - GBM null: 30 种子同管线, 固定 N 与自适应 N 的 null 分别生成 (同口径
      净差); 走前法 null 复刻整条选择流程 (前半选 N → 后半净)
    - H1 判据: 每 (sym, tf) 后半自适应净 ≥ 后半事前固定 N 净
    - H2 判据: 自适应全样本净 ≥ N20 全样本净
    - H3 判据: (自适应净 − GBM自适应null) ≥ (固定N净 − GBM固定Nnull),
      固定 N 取 N=20 (主) 并报走前法版本
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  固定 N Donchian  | rolling(N).max/min.shift(1) (窗口止于 t−1)| bar 收盘后 | CH5 口径 A (c41 复用)
  自适应 N         | N_t=N_I×V_n/V_c, 滚动 σ 因果          | bar 收盘后 | Seidel & Ginsberg 1983
  走前法分割点     | 样本中点 (事前可知)                    | 全样本     | 预注册 (可实施)
  GBM null         | sim_market.gbm_matching + 同系统       | 锚定真实   | 30 种子 (分别生成)

数据声明:
  BTC/ETH 4h (6,570根) + 1h (26,280根), 2023-08..2026-08。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  Ns 10~100 步长 10; N_I=N20=20; 口径 A; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 固定 N 的 H3 主对照取 N=20 (零优化经典基准); 走前法选出的 N 作补充报告。
  - 走前法后半的滚动窗口用全样本 (PnL 按半切), 无边界偏差 (c41 H2 同法)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① Donchian golden (c41 同款: 已知通道突破序列, 信号 bar
    与方向正确); ② 自适应 golden (N_t 钳制 [10,100] 且净盈亏有限);
    ③ GBM null sanity: GBM 自适应/固定N20 全样本均值净 ∈ [−2.5, 0.5]
    (whipsaw 负漂移带); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 固定/自适应/走前法分别生成
  - MIN_N: 每配置交易数报告并标注 (学习级 MIN_N=100)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 1h × 3 种子, 不写 .out
  - 全量: BTC/ETH 1h+4h × 30 种子 (预计 ≤5 分钟)

运行命令:
  python3 research/studies/c43_adaptive_n_fair.py --dev
  python3 research/studies/c43_adaptive_n_fair.py
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
    "Ns": tuple(range(10, 101, 10)),
    "N20": 20,
    "N_I": 20,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "gate_band": (-2.5, 0.5),            # GBM 净盈亏带 (whipsaw 负)
    "dev_subset": {"n_gbm": 3, "tfs": ("1h",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c43_adaptive_n_fair"


# ── Donchian 口径 A (c41 复用, 无成本, 永远在场反转) ────────
def _donchian_per_A(close, high, low, N):
    """CH5 口径 A: 当日 high>前 N 日 high → 多; low<前 N 日 low → 空.
    返回 (per, sig_dir)."""
    c = np.asarray(close, float)
    n = len(c)
    hi = pd.Series(high)
    lo = pd.Series(low)
    ref_hi = hi.rolling(N).max().shift(1).values
    ref_lo = lo.rolling(N).min().shift(1).values
    long_sig = high > ref_hi
    short_sig = low < ref_lo
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


def fixed_net_A(close, high, low, N):
    per, sig = _donchian_per_A(close, high, low, N)
    return float(per.sum()), int((sig != 0).sum())


def adaptive_net_A(close, high, low, r, bpd, N_I):
    """自适应 N (Seidel & Ginsberg 1983, 因果): N_t=clamp(N_I×V_n/V_c).
    V_n=1 年滚动 σ; V_c=N_I/4 bar 滚动 σ; 单调双端队列变窗通道.
    返回 (net_full, per, n_trades)."""
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
    q_hi = deque()
    q_lo = deque()
    p = np.zeros(n, int)
    pos = 0
    n_tr = 0
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
            if pos != 1:
                n_tr += 1
            pos = 1
        elif q_lo and low[t] < q_lo[0][1]:
            if pos != -1:
                n_tr += 1
            pos = -1
        p[t] = pos
    rr = np.zeros(n)
    rr[:-1] = c[1:] / c[:-1] - 1.0
    per = p * rr
    return float(per.sum()), per, n_tr


def walk_forward(close, high, low, ns, bpd):
    """走前法: 前半扫描选最优 N → 后半净 (全样本滚动窗口, PnL 按半切).
    返回 (n_opt, second_net, first_net)."""
    c = np.asarray(close, float)
    n = len(c)
    half = n // 2
    best_n, best_net = None, None
    for Nd in ns:
        per, _ = _donchian_per_A(c, high, low, Nd * bpd)
        nh = float(per[:half].sum())
        if best_n is None or nh > best_net:
            best_n, best_net = Nd, nh
    per2, _ = _donchian_per_A(c, high, low, best_n * bpd)
    return best_n, float(per2[half:].sum()), best_net


# ── GBM null (30 种子, 固定/自适应/走前法分别生成) ──────────
def gbm_nulls(df, tf, params, seeds):
    a_full, a_second, n20_full, wf_second = [], [], [], []
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        ctx = make_ctx(rw, params["warmup"], state_fns={})
        c, h, l = ctx.close, ctx.high, ctx.low
        r = np.concatenate([[0.0], np.diff(np.log(c))])
        half = len(c) // 2
        ad_full, ad_per, _ = adaptive_net_A(c, h, l, r, params["bpd"][tf],
                                            params["N_I"])
        a_full.append(ad_full)
        a_second.append(float(ad_per[half:].sum()))
        n20_full.append(fixed_net_A(c, h, l, params["N20"] * params["bpd"][tf])[0])
        wf_second.append(walk_forward(c, h, l, params["Ns"],
                                     params["bpd"][tf])[1])
    out = {}
    for key, arr in (("a_full", a_full), ("a_second", a_second),
                     ("n20_full", n20_full), ("wf_second", wf_second)):
        a = np.array(arr)
        out[key] = (float(np.mean(a)), float(np.std(a, ddof=1)))
    return out


# ── GATE 自检 (违规即停) ────────────────────────────────────
def _golden_donchian():
    close = np.array([4.0, 4.5, 5.0, 5.5, 5.4, 5.3, 4.0, 3.5, 3.6])
    high = close + 0.3
    low = close - 0.3
    high[0] = 5.0
    low[1] = 3.5
    per, sig = _donchian_per_A(close, high, low, 3)
    if int((sig != 0).sum()) < 2:
        raise SystemExit("GATE FAIL: Donchian golden 信号数 < 2")


def gate(gbm_a_full, gbm_n20_full):
    """① Donchian golden; ② 自适应 golden (N_t 钳制 + 净有限);
    ③ GBM null sanity (whipsaw 负漂移带)."""
    _golden_donchian()
    lo, hi = PARAMS["gate_band"]
    if not (lo <= gbm_a_full <= hi) or not (lo <= gbm_n20_full <= hi):
        raise SystemExit(
            f"GATE FAIL: GBM 净盈亏 自适应 {gbm_a_full:+.3f} / N20 "
            f"{gbm_n20_full:+.3f} ∉ [{lo}, {hi}] — 管线错误, 停")
    print(f"[GATE] Donchian golden [PASS]; GBM null 自适应 {gbm_a_full:+.3f} "
          f"N20 {gbm_n20_full:+.3f} (whipsaw 负) [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pp(v):
    return f"{v:+.4f}"


def write_out(out_path, params, rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},tfs={},Ns={}~{},N_I={},N20={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), ",".join(p["tfs"]), p["Ns"][0], p["Ns"][-1],
            p["N_I"], p["N20"], p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(GBM 自适应全样本净): {:.3f} [PASS]; "
        "探测器自检 Donchian golden [PASS]; GBM null sanity (自适应 {:.3f} / "
        "N20 {:.3f} ∈ [{}, {}]) [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], rows[0]["gbm"]["a_full"][0],
            rows[0]["gbm"]["a_full"][0], rows[0]["gbm"]["n20_full"][0],
            p["gate_band"][0], p["gate_band"][1], p["min_n"]),
        "# RESULTS: [学习级] c43 自适应 N 事前基准对比 (c41 H4 不可知论修正); "
        "CH5 口径 A; 走前法 (前半选 N → 后半比) + N=20 零优化 + GBM null 净差; "
        "自适应 N_t=N_I×V_n/V_c (因果); 描述层无入场, 无交易含义",
        "",
    ]
    # 走前法 (H1)
    lines.append("[走前法] 每 (标的, tf): 前半最优 N | 后半: 事前固定N vs 自适应:")
    n_h1 = 0
    for r in rows:
        ok = r["ad_second"] >= r["wf_second"]
        if ok:
            n_h1 += 1
        lines.append("  {} {}: 前半最优 N={} | 后半 固定N {:+.4f} vs 自适应 "
                     "{:+.4f} -> {}".format(
            r["sym"], r["tf"], r["n_opt"], r["wf_second"], r["ad_second"],
            "自适应≥✓" if ok else "固定N≥✗"))
    lines.append("  H1 判据: 后半自适应 ≥ 事前固定 N -> {}/{}".format(
        n_h1, len(rows)))
    # H2 N=20
    lines.append("")
    lines.append("[H2] N=20 日固定 (零优化) vs 自适应 (全样本):")
    n_h2 = 0
    for r in rows:
        ok = r["ad_full"] >= r["n20_full"]
        if ok:
            n_h2 += 1
        lines.append("  {} {}: N20 {:+.4f} | 自适应 {:+.4f} -> {}".format(
            r["sym"], r["tf"], r["n20_full"], r["ad_full"],
            "自适应≥✓" if ok else "N20≥✗"))
    lines.append("  H2 判据: 自适应 ≥ N20 -> {}/{}".format(n_h2, len(rows)))
    # H3 null 超额
    lines.append("")
    lines.append("[H3] GBM null 净差超额 (海龟汤修正):")
    n_h3 = 0
    for r in rows:
        g = r["gbm"]
        ex_ad = r["ad_full"] - g["a_full"][0]
        ex_n20 = r["n20_full"] - g["n20_full"][0]
        ok = ex_ad >= ex_n20
        if ok:
            n_h3 += 1
        lines.append("  {} {}: 自适应超额 {:+.4f} | N20 超额 {:+.4f} -> "
                     "{}".format(r["sym"], r["tf"], ex_ad, ex_n20,
                                 "自适应≥✓" if ok else "N20≥✗"))
    lines.append("  H3 判据: 自适应超额 ≥ N20 超额 (full) -> {}/{}".format(
        n_h3, len(rows)))
    # 走前法 null 超额 (补充)
    lines.append("")
    lines.append("[H3-wf] 走前法 null 超额 (后半):")
    for r in rows:
        g = r["gbm"]
        ex_ad = r["ad_second"] - g["a_second"][0]
        ex_wf = r["wf_second"] - g["wf_second"][0]
        lines.append("  {} {}: 自适应超额 {:+.4f} | 走前固定N 超额 {:+.4f}"
                     "".format(r["sym"], r["tf"], ex_ad, ex_wf))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c41 H4 (事后最优基准: 自适应 0/4 未达 — 不可知论"
                 "违规); 书 CH21 (测试不用于发现); 书 CH5 (Donchian 4 周默认); "
                 "Seidel & Ginsberg 1983 (自适应 N)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    dev_tfs = PARAMS["dev_subset"]["tfs"] if dev else None
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    data = load_candles(timeframes=PARAMS["tfs"])
    rows = []
    for sym in PARAMS["crypto"]:
        for tf in PARAMS["tfs"]:
            if dev_tfs is not None and tf not in dev_tfs:
                continue
            df = data.get(sym, {}).get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            c, h, l = ctx.close, ctx.high, ctx.low
            r = np.concatenate([[0.0], np.diff(np.log(c))])
            half = len(c) // 2
            ad_full, ad_per, ad_n_tr = adaptive_net_A(c, h, l, r,
                                                      PARAMS["bpd"][tf],
                                                      PARAMS["N_I"])
            ad_second = float(ad_per[half:].sum())
            n20_full, n20_n_tr = fixed_net_A(c, h, l,
                                             PARAMS["N20"] * PARAMS["bpd"][tf])
            n_opt, wf_second, wf_first = walk_forward(c, h, l, PARAMS["Ns"],
                                                      PARAMS["bpd"][tf])
            gbm = gbm_nulls(df, tf, PARAMS, seeds)
            rows.append({"sym": sym, "tf": tf, "ad_full": ad_full,
                         "ad_second": ad_second, "n20_full": n20_full,
                         "n_opt": n_opt, "wf_second": wf_second,
                         "ad_n_tr": ad_n_tr, "n20_n_tr": n20_n_tr,
                         "gbm": gbm})

    gate(float(np.mean([r["gbm"]["a_full"][0] for r in rows])),
         float(np.mean([r["gbm"]["n20_full"][0] for r in rows])))

    if dev:
        for r in rows:
            print("  [dev] {} {} 走前N={} 后半 固定{:+.3f} vs 自适应{:+.3f} | "
                  "full N20 {:+.3f} 自适应 {:+.3f}".format(
                r["sym"], r["tf"], r["n_opt"], r["wf_second"], r["ad_second"],
                r["n20_full"], r["ad_full"]))
        print(f"[dev] 管线 OK ({len(rows)} 格 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
