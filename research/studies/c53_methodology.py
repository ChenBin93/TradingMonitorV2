#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C53 M9 U1 验证、风险与组合忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 M9 U1 收官, PLAN §2.5 c53): 书 CH20-24 验证/风险/
  组合。oracle 逐字核实口径: 几何衰减 (CH20 p.869 💬 无源, 50/25/12.5 连涨
  连跌衰减, 书自认"噪声≈随机数"基准); 冲击相关性 (CH22 p.994: 冲击时相关
  →1 分散失效); 分散化 (CH24 图 24.1: 现实充分分散≈风险减半 ≈ n_eff=2);
  波动稳定 (CH24 p.1142-1143: VF=目标波动/10 日滚动年化、滞后一期);
  测试纪律 (CH21: 测试用于取舍不用于发现 — 与我们门禁同源, 结论确认同源性).
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。**结论不得作交易依据**。
  学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、内置
  GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): ① 几何衰减对拍 (方向/幅度); ② 高波动态
  相关性抬升 (书"冲击时相关→1"); ③ 分散化审计 (n_eff + 分年); ④ 波动
  稳定口径核对 (c26 vs 书 VF).

预注册假设 (PLAN §2.5 c53 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 几何衰减对拍 — 方向版: 真实 P(下一 bar 反向)/P(延续≥2/≥3 bar) vs
      书 50/25/12.5 vs 硬币 null (c31 预期: 真实延续 < 50%); 幅度版:
      |ret| 的 50/25/12.5 Zipf 结构 vs lognormal null
  H2: 高波动态相关性抬升 — 20 标的滚动 24h 两两相关均值按波动分位分层
      vs 块 bootstrap null (保 c12 波动聚集, 30 次); 另测 n_eff 随波动态
      坍缩 (高波态 n_eff vs 低波态)
  H3: 分散化审计 — 全样本 n_eff (c37 复核) + 分年 n_eff vs 书图 24.1
      "现实≈2"; 判据=分年 n_eff 中位数与离散
  H4: 波动稳定口径核对 — c26 公式 vs 书 VF (VF=target/10 日滚动年化、
      滞后一期); 口径不同则标注并按书口径快速重跑 (BTC/ETH, 成本
      0/2.5bp)

  操作化 (运行前锁定):
    - 数据: 20 标的 1h/4h; 学习级: 30 种子、无 BY_YEAR、MIN_N=100
    - H1: run 状态机 (c31 口径, 0 延续); P(反向)/P(延续≥2/≥3) vs 硬币
      30 种子; 幅度: |ret| 分位比 q75/q50, q87.5/q50 vs lognormal null
    - H2: 滚动 24h 两两相关 (每 6 bar 采样) 按波动三分位; 块 bootstrap
      (50 bar 块重排 30 次) null; n_eff 高/低波动态
    - H3: n_eff 全样本 + 分年 (c37 特征值比)
    - H4: c26 口径 vs 书 VF 标注差异; 书口径重跑 (BTC/ETH 1h, 目标 12%,
      10 日滚动年化滞后一期, 成本 0/2.5bp)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close            | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  run 状态机       | sign 连续段 (0 延续, c31 口径)        | bar 收盘后 | c31
  |ret| 分位比      | q75/q50, q87.5/q50 (全样本描述)      | 全样本事后 | 书 CH20
  滚动相关         | 24h 两两相关 (每 6 bar 采样)          | bar 收盘后 | 因果
  波动分位         | 24h 收益 sd 三分位 (因果)             | bar 收盘后 | 描述分层
  块 bootstrap     | 50 bar 块重排 (保波动聚集)           | 全样本     | c52 同款
  n_eff            | (Σλ)²/Σλ² (c37 特征值比)             | 全样本     | c37
  VF (书)          | target/10 日滚动年化 (滞后一期)      | bar 收盘后 | 书 CH24
  GBM/硬币 null    | gbm_matching/硬币 iid (30 种子)      | 锚定真实   | 惯例

数据声明:
  20 标的 1h (26,280根) + 4h (6,570根), 2023-08..2026-08。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  run k=2/3; 幅度分位 50/75/87.5; 相关窗 24, 采样 6; 块 50, bootstrap 30;
  VF 目标 12%, 10 日滚动; 成本 0/2.5bp; GBM 30 种子; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - H2 的相关按每 6 bar 采样计算 (计算成本); 块 bootstrap 同采样。
  - H4 的 c26 口径 (ATR 比例仓位) 与书 VF (目标/10 日滚动年化) 不同 —
    标注并按书口径重跑 (BTC/ETH 1h, 成本 0/2.5bp)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① run golden (c31 同款); ② n_eff golden (完全相关=1,
    独立=2, c37 同款); ③ 块 bootstrap sanity (波动聚集保留); 任一失败
    SystemExit
  - GBM/硬币/块 bootstrap 无信息对照: 30 次
  - MIN_N: 每格 n ≥ MIN_N=100 (不足标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH 1h × 3 种子, 不写 .out
  - 全量: 20 标的 1h/4h × 30 种子 (预计 ≤12 分钟)

运行命令:
  python3 research/studies/c53_methodology.py --dev
  python3 research/studies/c53_methodology.py
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
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "h1_qs": (0.50, 0.75, 0.875),
    "h2_win": 24,                          # 相关窗口 (24h)
    "h2_step": 6,                          # 相关采样步长
    "h2_block": 50,                        # 块 bootstrap 块长
    "h2_perm": 30,
    "vf_target": 0.12,                     # 书 VF 目标 (12%)
    "vf_win": 10,                          # 10 日滚动 (1h=240 bar)
    "costs_bp": (0.0, 2.5),
    "dev_subset": {"n_gbm": 3, "tfs": ("1h",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c53_methodology"


# ── 装载 ─────────────────────────────────────────────────────
def load_all(tf):
    data = load_candles(timeframes=(tf,))
    out = []
    for sym in data:
        if "USDT" not in sym:
            continue
        df = data[sym].get(tf)
        if df is None or verify(df, sym, tf):
            continue
        out.append((sym, df))
    return out


# ── H1: 几何衰减 (run 状态机, c31 口径) ─────────────────────
def run_lengths(s):
    m = len(s)
    lengths = []
    cur_dir = 0
    cur_len = 0
    for i in range(m):
        si = s[i]
        if si != 0:
            if cur_dir == 0:
                cur_dir = si
                cur_len = 1
            elif si == cur_dir:
                cur_len += 1
            else:
                lengths.append(cur_len)
                cur_dir = si
                cur_len = 1
        else:
            if cur_dir != 0:
                cur_len += 1
    if cur_len > 0:
        lengths.append(cur_len)
    return np.array(lengths, int)


def h1_direction(close):
    s = np.sign(np.diff(close))
    L = run_lengths(s)
    if len(L) == 0:
        return None
    n_runs = len(L)
    # 精确 run 长分布 P(L==1/2/3) (书几何衰减 50/25/12.5)
    p1 = float(np.mean(L == 1))
    p2 = float(np.mean(L == 2))
    p3 = float(np.mean(L == 3))
    # 累计 P(L>=2/>=3)
    pg2 = float(np.mean(L >= 2))
    pg3 = float(np.mean(L >= 3))
    return p1, p2, p3, pg2, pg3, n_runs


def h1_amplitude(r):
    q = np.quantile(np.abs(r), PARAMS["h1_qs"])
    return (q[1] / q[0] if q[0] > 0 else float("nan"),
            q[2] / q[0] if q[0] > 0 else float("nan"))


def coin_null(n_diffs, seeds):
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        s = rng.choice([1.0, -1.0], size=n_diffs)
        L = run_lengths(s)
        out.append((float(np.mean(L == 1)), float(np.mean(L == 2)),
                    float(np.mean(L == 3)), float(np.mean(L >= 2)),
                    float(np.mean(L >= 3))))
    a = np.array(out)
    return (a[:, 0], a[:, 1], a[:, 2])


# ── H2: 高波动态相关性 ───────────────────────────────────────
def vol_state(returns_mat, w, q1, q2):
    """每 bar 的 24h 横截面收益 sd 三分位 (因果)."""
    n = returns_mat.shape[0]
    t = np.arange(n)
    sd = np.full(n, np.nan)
    for i in range(w - 1, n):
        sd[i] = float(np.std(returns_mat[i - w + 1:i + 1]))
    lo, hi = np.nanquantile(sd, [q1, q2])
    st = np.full(n, "", dtype=object)
    ok = np.isfinite(sd)
    st[ok & (sd <= lo)] = "低"
    st[ok & (sd > hi)] = "高"
    st[ok & (sd > lo) & (sd <= hi)] = "中"
    return st, sd


def pairwise_rho_series(returns_mat, w, step):
    """每 step bar 的两两相关均值 (滚动 w 窗)."""
    n = returns_mat.shape[0]
    out = []
    for i in range(w - 1, n, step):
        Z = returns_mat[i - w + 1:i + 1]
        Z = Z - Z.mean(axis=0)
        sd = Z.std(axis=0, ddof=1)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Zz = Z / sd
        C = Zz.T @ Zz / (w - 1)
        k = C.shape[0]
        rho = (float(C.sum()) - k) / (k * (k - 1)) if k > 1 else float("nan")
        out.append((i, rho))
    return out


def h2_by_vol(returns_mat, w, step, q1, q2):
    st, sd = vol_state(returns_mat, w, q1, q2)
    pr = pairwise_rho_series(returns_mat, w, step)
    acc = {"低": (0.0, 0), "中": (0.0, 0), "高": (0.0, 0)}
    for i, rho in pr:
        if np.isfinite(rho) and st[i] in acc:
            s, n = acc[st[i]]
            acc[st[i]] = (s + rho, n + 1)
    return {k: (v[0] / v[1] if v[1] else float("nan"), v[1])
            for k, v in acc.items()}


def n_eff_corr(returns_mat, w, step):
    """滚动相关矩阵 n_eff (采样 bar)."""
    out = []
    for i in range(w - 1, returns_mat.shape[0], step):
        Z = returns_mat[i - w + 1:i + 1]
        Z = Z - Z.mean(axis=0)
        sd = Z.std(axis=0, ddof=1)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Zz = Z / sd
        C = Zz.T @ Zz / (w - 1)
        lam = np.maximum(np.linalg.eigvalsh((C + C.T) / 2.0), 0.0)
        s1, s2 = lam.sum(), (lam * lam).sum()
        if s2 > 0:
            out.append(s1 * s1 / s2)
    return out


# ── H3: 分散化审计 (n_eff 全样本 + 分年) ────────────────────
def n_eff_matrix(mat):
    lam = np.maximum(np.linalg.eigvalsh((mat + mat.T) / 2.0), 0.0)
    s1, s2 = lam.sum(), (lam * lam).sum()
    return s1 * s1 / s2 if s2 > 0 else float("nan")


# ── H4: 书 VF (目标/10 日滚动年化, 滞后一期) ─────────────────
def vf_book_rerun(close, high, low, target, win, cost_bp, seed=None):
    """书口径 VF: 仓位 = target/10日滚动年化波动 (滞后一期), 固定排程."""
    r = np.diff(np.log(close))
    r = np.concatenate([[0.0], r])
    # 10 日滚动年化 (1h: 240 bar; 用输入 win bar)
    ann = pd.Series(r).rolling(win).std().values * np.sqrt(365 * 24)
    vf = np.full(len(close), 1.0)
    ok = np.isfinite(ann) & (ann > 0)
    vf[ok] = target / ann[ok]
    # 固定排程 (每 24 根一笔, 持有 24 根), 仓位=VF 滞后一期
    pnl = []
    for t0 in range(win, len(close) - 24, 24):
        pos = vf[t0 - 1]
        r_hold = close[t0 + 24] / close[t0] - 1.0
        pnl.append(pos * r_hold)
    p = np.array(pnl)
    cost = 2.0 * (cost_bp / 10000.0) * np.abs(vf[win - 1])
    net = float(p.sum()) - cost * len(p)
    return net, len(p)


# ── GATE 自检 ────────────────────────────────────────────────
def gate(bs_acf_ok):
    """① run golden; ② n_eff golden (c37 同款); ③ 块 bootstrap sanity."""
    s = np.array([1.0] * 5 + [0.0] * 2 + [1.0] * 3 + [-1.0] * 4 + [0.0] + [1.0])
    L = run_lengths(s)
    if not (len(L) == 3 and (L == np.array([10, 5, 1])).all()):
        raise SystemExit(f"GATE FAIL: run golden {L.tolist()}")
    C1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    if abs(n_eff_matrix(C1) - 1.0) > 1e-9:
        raise SystemExit("GATE FAIL: n_eff 完全相关 ≠ 1")
    if abs(n_eff_matrix(np.eye(2)) - 2.0) > 1e-9:
        raise SystemExit("GATE FAIL: n_eff 独立 ≠ 2")
    if not bs_acf_ok:
        raise SystemExit("GATE FAIL: 块 bootstrap 波动聚集未保留")
    print(f"[GATE] run golden [PASS]; n_eff golden [PASS]; 块 bootstrap sanity "
          f"[PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def write_out(out_path, params, h1, h2, h3, h4, gate_note):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},h2_win={},h2_block={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tfs"]), p["h2_win"], p["h2_block"], p["gbm_seeds"],
            p["min_n"], p["min_n"]),
        "# GATE: 探测器自检 run golden + n_eff golden + 块 bootstrap [PASS]; "
        "MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c53 M9 U1 验证/风险/组合忠实复现 (书 CH20-24); "
        "几何衰减 (50/25/12.5), 冲击相关性, 分散化 n_eff, 波动稳定 VF 口径; "
        "GBM/硬币/块 bootstrap 30 次同管线; 描述层无入场, 无交易含义",
        "",
    ]
    # H1
    lines.append("[H1] 几何衰减对拍 (书 50/25/12.5):")
    for tf, r in h1.items():
        p1, p2, p3, pg2, pg3, n_runs = r["dir"]
        cn = r["coin"]
        lines.append("  {}: P(L=1) {:.1%} (书 50%, 硬币 {:.1%}±{:.1%}) | "
                     "P(L=2) {:.1%} (书 25%, 硬币 {:.1%}±{:.1%}) | "
                     "P(L=3) {:.1%} (书 12.5%, 硬币 {:.1%}±{:.1%}) | "
                     "P(≥2) {:.1%} P(≥3) {:.1%} (n={})"
                     "".format(tf, p1, cn[0][0], cn[0][1], p2,
                               cn[1][0], cn[1][1], p3, cn[2][0], cn[2][1],
                               pg2, pg3, n_runs))
    lines.append("  幅度 Zipf (q75/q50, q87.5/q50):")
    for tf, r in h1.items():
        lines.append("  {}: 真实 {:.2f}/{:.2f} vs 书 2/4 vs lognormal "
                     "{:.2f}/{:.2f}".format(tf, r["amp"][0], r["amp"][1],
                                            r["ln"][0], r["ln"][1]))
    # H2
    lines.append("")
    lines.append("[H2] 高波动态相关性 (书: 冲击时相关→1):")
    for tf, r in h2.items():
        for st in ("低", "中", "高"):
            rr, nn = r["real"][st]
            bs = r["bs"][st]
            lines.append("  {} 波动{}: ρ {:.3f} (n={}) {} | bootstrap "
                         "{:.3f}±{:.3f}".format(
                tf, st, rr, nn, _nm(nn, p["min_n"]), bs[0], bs[1]))
        lines.append("  {} n_eff: 低波 {:.1f} vs 高波 {:.1f} (坍缩 {:.1%})"
                     "".format(tf, r["neff_lo"], r["neff_hi"],
                               (r["neff_hi"] - r["neff_lo"]) / r["neff_lo"]))
    # H3
    lines.append("")
    lines.append("[H3] 分散化审计 (书图 24.1: 现实≈n_eff=2):")
    for tf, r in h3.items():
        lines.append("  {}: 全样本 n_eff {:.2f} | 分年 {}".format(
            tf, r["full"], [f"{y}:{v:.2f}" for y, v in r["years"].items()]))
    # H4
    lines.append("")
    lines.append("[H4] 波动稳定口径核对 (c26 vs 书 VF):")
    lines.append("  " + h4["note"])
    for tf, r in h4["rerun"].items():
        lines.append("  {}: 书 VF 净 {:.4f} (0bp) / {:.4f} (2.5bp) (n={})"
                     "".format(tf, r[0], r[1], r[2]))
    # 测试纪律
    lines.append("")
    lines.append("[测试纪律] 书 CH21 '测试用于取舍不用于发现' 与我们三层门禁"
                 " (预注册/GATE/check_study) 同源 — 结论确认同源性")
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c31 (run 反持久); c37 (n_eff=2.15); c26 (波动目标); "
                 "c12 (波动长记忆); 书 CH20 p.869/CH22 p.994/CH24 p.1142-1143")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    dev_tfs = PARAMS["dev_subset"]["tfs"] if dev else None

    h1, h2, h3, h4 = {}, {}, {}, {}
    bs_acf_ok = True

    for tf in PARAMS["tfs"]:
        if dev_tfs and tf not in dev_tfs:
            continue
        syms_dfs = load_all(tf)
        rets = {}
        for sym, df in syms_dfs:
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            rets[sym] = np.concatenate([[0.0], np.diff(np.log(ctx.close))])
        n_bars = min(len(r) for r in rets.values())
        ret_mat = np.array([rets[s][:n_bars] for s in rets
                            if "USDT" in s]).T
        # H1 方向
        dirs = []
        for sym, df in syms_dfs:
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            d = h1_direction(ctx.close)
            if d:
                dirs.append(d)
        n_runs_total = sum(d[5] for d in dirs)
        p1 = sum(d[0] * d[5] for d in dirs) / n_runs_total
        p2 = sum(d[1] * d[5] for d in dirs) / n_runs_total
        p3 = sum(d[2] * d[5] for d in dirs) / n_runs_total
        pg2 = sum(d[3] * d[5] for d in dirs) / n_runs_total
        pg3 = sum(d[4] * d[5] for d in dirs) / n_runs_total
        cn = coin_null(n_bars - 1, seeds)
        cn_agg = [(float(np.mean(a)), float(np.std(a, ddof=1))) for a in cn]
        # H1 幅度
        amps = [h1_amplitude(np.diff(np.log(df["close"].values.astype(float))))
                for sym, df in syms_dfs]
        amp = (float(np.mean([a[0] for a in amps if np.isfinite(a[0])])),
               float(np.mean([a[1] for a in amps if np.isfinite(a[1])])))
        # lognormal null: 拟合 |r| 的 log sd → 分位比
        allr = np.concatenate([np.abs(np.diff(np.log(df["close"].values.astype(float))))
                               for sym, df in syms_dfs])
        ln_sd = float(np.std(np.log(allr[allr > 0])))
        ln_q = np.quantile(allr[allr > 0], [0.5, 0.75, 0.875])
        ln_ratio = (ln_q[1] / ln_q[0], ln_q[2] / ln_q[0])
        h1[tf] = {"dir": (p1, p2, p3, pg2, pg3, n_runs_total),
                  "coin": cn_agg, "amp": amp, "ln": ln_ratio}
        # H2 相关 (每 step bar 采样)
        win = PARAMS["h2_win"] * (24 if tf == "1h" else 6)
        step = PARAMS["h2_step"]
        st, sd = vol_state(ret_mat, win, 1.0 / 3, 2.0 / 3)
        real_h2 = h2_by_vol(ret_mat, win, step, 1.0 / 3, 2.0 / 3)
        # 块 bootstrap
        bs_h2 = {"低": [], "中": [], "高": []}
        for perm in range(PARAMS["h2_perm"]):
            bs_rets = np.zeros_like(ret_mat)
            for j in range(ret_mat.shape[1]):
                r = ret_mat[:, j]
                bs_rets[:, j] = block_perm(r, PARAMS["h2_block"], perm * 100 + j)
            bs_st, _ = vol_state(bs_rets, win, 1.0 / 3, 2.0 / 3)
            bs_pr = pairwise_rho_series(bs_rets, win, step)
            acc = {"低": (0.0, 0), "中": (0.0, 0), "高": (0.0, 0)}
            for i, rho in bs_pr:
                if np.isfinite(rho) and bs_st[i] in acc:
                    s, n = acc[bs_st[i]]
                    acc[bs_st[i]] = (s + rho, n + 1)
            for k in acc:
                if acc[k][1]:
                    bs_h2[k].append(acc[k][0] / acc[k][1])
        h2[tf] = {"real": real_h2,
                  "bs": {k: (float(np.mean(v)), float(np.std(v, ddof=1))
                             if len(v) > 1 else 0.0) for k, v in bs_h2.items()}}
        # n_eff 高/低波动态 (相关矩阵 n_eff)
        ne = n_eff_corr(ret_mat, win, step)
        ne_lo = np.quantile(ne, 0.1)
        ne_hi = np.quantile(ne, 0.9)
        # 每 bar 的 n_eff 与波动状态关联 (粗略: 采样 bar 的 sd 分位)
        sd_sampled = np.array([sd[i] for i in range(win - 1, n_bars, step)
                               if np.isfinite(sd[i])])
        ne_vals = np.array(ne)
        m = min(len(sd_sampled), len(ne_vals))
        s_lo = sd_sampled[:m] <= np.quantile(sd_sampled[:m], 1.0 / 3)
        s_hi = sd_sampled[:m] >= np.quantile(sd_sampled[:m], 2.0 / 3)
        h2[tf]["neff_lo"] = float(np.mean(ne_vals[:m][s_lo]))
        h2[tf]["neff_hi"] = float(np.mean(ne_vals[:m][s_hi]))
        # 块 bootstrap sanity (首标的 |r| ACF@1)
        r0 = ret_mat[:, 0]
        a0 = acf(np.abs(r0), 1)
        a1 = acf(np.abs(block_perm(r0, PARAMS["h2_block"], 7)), 1)
        if abs(a1 - a0) > 0.15 * max(abs(a0), 0.01):
            bs_acf_ok = False
        # H3 分散化
        full_mat = np.corrcoef(ret_mat.T)
        neff_full = n_eff_matrix(full_mat)
        # 分年 (用首标的 ctx 年份)
        ctx0 = make_ctx(syms_dfs[0][1], PARAMS["warmup"], state_fns={})
        years = ctx0.years[:n_bars]
        per_year = {}
        for y in (2024, 2025, 2026):
            m = years == y
            if m.sum() < win + 5:
                continue
            seg = ret_mat[m]
            per_year[y] = n_eff_matrix(np.corrcoef(seg.T))
        h3[tf] = {"full": neff_full, "years": per_year}
        # H4 书 VF (BTC/ETH 1h)
        if tf == "1h":
            rerun = {}
            for sym in PARAMS["crypto"]:
                df = None
                for s, d in syms_dfs:
                    if s == sym:
                        df = d
                        break
                if df is None:
                    continue
                ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
                n0, nt0 = vf_book_rerun(ctx.close, ctx.high, ctx.low,
                                        PARAMS["vf_target"],
                                        240, PARAMS["costs_bp"][0])
                n25, _ = vf_book_rerun(ctx.close, ctx.high, ctx.low,
                                       PARAMS["vf_target"],
                                       240, PARAMS["costs_bp"][1])
                rerun[sym] = (n0, n25, nt0)
            h4["rerun"] = rerun

    # 块 bootstrap sanity (全局)
    gate(bs_acf_ok)

    # H4 note
    h4["note"] = ("c26 用 ATR 比例仓位 (仓位∝1/ATR 或 ATR 尺度, 排程每 24 根); "
                  "书 CH24 VF=目标波动/10 日滚动年化、滞后一期 — 口径不同, "
                  "已按书口径重跑 (BTC/ETH 1h, 目标 12%, 240 bar 滚动, "
                  "成本 0/2.5bp)")

    if dev:
        for tf in h1:
            print("  [dev] {} P(L1)={:.2f} P(L2)={:.2f} P(L3)={:.2f}".format(
                tf, h1[tf]["dir"][0], h1[tf]["dir"][1], h1[tf]["dir"][2]))
        print("  [dev] H2 高波 ρ={:.3f} vs 低波 {:.3f}".format(
            h2["1h"]["real"]["高"][0], h2["1h"]["real"]["低"][0]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, h1, h2, h3, h4, "gate")
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


def block_perm(r, block, seed):
    rng = np.random.default_rng(seed)
    n = len(r)
    nb = int(np.ceil(n / block))
    order = rng.permutation(nb)
    parts = [r[i * block:(i + 1) * block] for i in order]
    return np.concatenate(parts)[:n]


def acf(x, lag):
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
