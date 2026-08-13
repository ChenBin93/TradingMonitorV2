#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C33 日线运行肥尾忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U0-2 补验, PLAN §2.5 c33): c31 用 1h/3y/GBM 证伪书
  CH1/CH8"运行序列肥尾"。oracle 复核: 书锚点是**日线、100 个交易日、对称硬币
  基准** (书"run of 6")。且 c18 证明方向由漂移主导 — 书用 50/50 硬币会把漂移
  误判为肥尾 → 必须双基准。本研究按书原口径忠实复现 + 漂移修正基准。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。结论标注 [学习级],
  **不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring
  预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH1/CH8"价格运行序列肥尾 — 同向连续 bar 比
  抛硬币长得多, 是趋势系统利润的唯一来源", 书语境为**日线** (书锚点 "run of
  6" = 100 交易日窗口内出现 6 连的概率)。c31 用 1h 数据 + GBM 基准证伪 — 口径
  错位 (PLAN §3 已确认)。本考证: 日线 + 对称硬币基准 (忠实版) + 漂移 GBM 基准
  (修正版, c18: 漂移会把硬币基准误判为肥尾)。

预注册假设 (PLAN §2.5 c33 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 传统市场 N(run≥6) 超出对称硬币 95% 区间 (书在其语境成立)
  H2: 加密 N(run≥6) 超出对称硬币 95% 区间
  H3: 漂移 GBM 基准下真实仍超出 (肥尾超出漂移可解释部分)

  操作化 (运行前锁定):
    - run = 纯符号 sign(close_t − close_{t−1}); **0 变化计延续** (声明):
      平 bar 延续当前 run 的方向, 计入 run 长度; 首个非零符号前不计
    - gap 规则: 日线间隔 > gap_thresh_days=7 天断 run (仅真实数据停更;
      **阈值校准见设计偏离** — 字面"2×1天"会在周末断 run, 使书锚点
      "run of 6" 结构性不可达)
    - 统计量: N(run≥k) = run 长度 ≥ k 的 run 数; k ∈ {6, 8, 10}, 主判据
      k=6 (书锚点 "run of 6"); 另报 N(run≥12) 与 P(run≥12)
    - 基准 1 (忠实版): 对称硬币 iid p=0.5, 30 种子, n = 各自 bar 数 (diff 数)
    - 基准 2 (修正版): 漂移=样本对数收益均值, σ=样本 sd 的 GBM, 30 种子
    - 95% 区间: 30 种子分布 2.5/97.5 分位
    - H1 判据: 5 传统标的 N6 > 硬币 95% 上界 (逐标的上报, 计数裁决)
    - H2 判据: BTC/ETH N6 > 硬币 95% 上界
    - H3 判据: 漂移 GBM 基准 95% 上界下真实仍超出
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close 日线 (传统)| data/control.db (ts=UTC epoch 秒,     | bar 收盘后 | 对照数据源; max 历史
                   |   表名特殊字符 → SQL 双引号包裹)      |            |   (SPY 1993-/^TNX 1962-)
  close 日线 (加密)| daily_resample 自 1h (c30 口径)       | 日线收盘后 | data_loader
  run 序列         | 逐 bar 收盘符号状态机 (顺序, 因果)     | bar 收盘后 | 只由 ≤t 的符号决定
                   |   (0 延续, gap 断, 无切片)            |            |
  N(run≥k)         | run 最终长度事后统计                  | 全样本事后 | 描述层 (非条件特征)
  硬币基准         | iid ±1, p=0.5, 30 种子, n 匹配        | 锚定书口径 | 书"对称硬币"忠实复现
  漂移 GBM 基准    | 对数收益 N(样本μ, 样本σ), 30 种子      | 锚定真实   | c18: 漂移修正 (书 50/50 会
                   |                                       |            |   把漂移误判为肥尾)

数据声明:
  data/control.db (gitignored): SPY_1d 8,442 (1993-), CL=F_1d 6,521 (2000-),
  GC=F_1d 6,512 (2000-), EURUSD=X_1d 5,890 (2003-), ^TNX_1d 16,141 (1962-);
  data/backtest.db: BTC/ETH 1h → 日线 (daily_resample, ~1,096 根/3y)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  k∈{6,8,10}, 主判据 k=6; gap_mult=2.0; GBM/硬币 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - **gap 阈值校准 (运行前标定)**: 预注册"日线间隔 > 2×标准间隔断"若按中位数
    间隔 (1 天) 字面实现, 会在每个周末 (3 天 > 2 天) 断 run — 使书锚点
    "run of 6" 在周频缺口数据上结构性不可达 (dev 实测 SPY N6=0)。预注册原文
    "日线序列无需 gap 断裂"表明该规则只是数据停更护栏; 实测日线间隔分布:
    正常 1~5 天 (周末/假日), 真停更 ≥7~18 天 (EURUSD 18 天)。故校准阈值 =
    gap_thresh_days=7 (间隔 > 7 天断), 只断真实数据停更。
  - 0 变化计延续 (声明, 与 c31 的"0 断且不计"不同): 书为日线连续序列, 平 bar
    按延续处理更贴近书口径; 日线平 bar 极少, 影响微小。
  - 硬币基准为纯 iid (无 gap 结构), 真实序列的 gap 断 run (罕见停更) 会压低
    真实长 run — H1/H2 为保守检验。
  - 漂移 GBM 基准取对数收益样本均值/σ (c18 衔接: 方向由漂移主导, 硬币基准
    会把漂移误判为肥尾, 必须扣除)。
  - 学习级: 无 BY_YEAR; 30 种子沿用 c30/c31/c32 惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① run 状态机 golden 对拍 (构造 0 延续 + gap 断的已知序列,
    逐位验证 run 长度); ② 硬币基准 sanity: 30 种子均值 N(run≥6) ≈ n/64
    (iid 理论, ±50% 带), 验证基准生成; 任一失败 SystemExit
  - 双基准无信息对照: 硬币 (书口径) + 漂移 GBM (修正) 各 30 种子
  - MIN_N: 每格 N(run≥k) ≥ MIN_N=100 (学习级) 逐格报告
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: SPY + BTC × 3 种子, 不写 .out (管线调试用)
  - 全量: 5 传统 + 2 加密 × 30 种子, sha256 锁定全量版本

运行命令:
  python3 research/studies/c33_daily_run_faithful.py --dev
  python3 research/studies/c33_daily_run_faithful.py
"""
import hashlib
import os
import sqlite3
import sys
import time
from datetime import date

# 仓库根入 path (模板摩擦, 见 c12 报告)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.data_loader import daily_resample, load_candles, verify

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "control": ("SPY", "CL=F", "GC=F", "EURUSD=X", "^TNX"),
    "control_db": "data/control.db",
    "ks": (6, 8, 10),
    "k_main": 6,
    "gap_mult": 2.0,
    "gap_thresh_days": 7.0,             # 日线 gap 校准: 间隔 > 7 天断 (仅数据停更)
    "gbm_seeds": 30,
    "min_n": 100,                       # 学习级 MIN_N
    "dev_subset": {"n_gbm": 3, "control": ("SPY",)},
    "data_range": "传统 max 历史 (SPY 1993-/^TNX 1962-) / 加密 2023-08..2026-08",
}

STUDY_ID = "c33_daily_run_faithful"


# ── 加载 ─────────────────────────────────────────────────────
def load_crypto_daily(symbols):
    data = load_candles(timeframes=("1h",))
    out = []
    for sym in symbols:
        df = data.get(sym, {}).get("1h")
        if df is None or verify(df, sym, "1h"):
            continue
        out.append((sym, daily_resample(df)))
    return out


def load_control_daily(symbols, db_path):
    conn = sqlite3.connect(db_path)
    out = []
    try:
        for sym in symbols:
            df = pd.read_sql_query(
                f'SELECT ts, open, high, low, close, volume FROM "{sym}_1d" '
                "ORDER BY ts", conn)
            if df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
            out.append((sym, df.set_index("ts").sort_index()))
    finally:
        conn.close()
    return out


def _epoch_seconds(idx):
    return (idx.values.astype("datetime64[ns]").astype("int64") // 10 ** 9)


# ── run 状态机 (c33 口径: 0 变化计延续, gap 断; 因果, 无切片) ─
def run_lengths(s, gaps):
    """s: 符号数组 (长度 m); gaps: 断 run 掩码 (长度 m).
    0 变化计延续 (声明): 平 bar 延续当前 run, 计入长度; 首个非零前不计.
    gap 处断 run. 返回 run 长度数组 (int)."""
    m = len(s)
    lengths = []
    cur_dir = 0
    cur_len = 0
    for i in range(m):
        if gaps[i]:
            if cur_len > 0:
                lengths.append(cur_len)
            cur_dir = 0
            cur_len = 0
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


def real_runs(c, ts, params):
    """日线 run: 0 变化计延续; 间隔 > gap_thresh_days 天断 (仅数据停更).

    gap 阈值校准 (运行前, 非 post-hoc): 预注册"间隔 > 2×标准间隔断"若按
    中位数间隔 (1 天) 字面实现, 会在每个周末 (3 天 > 2 天) 断 run — 使书
    锚点 "run of 6" 在周频缺口数据上结构性不可达 (已验证 SPY N6=0)。预注册
    原文"日线序列无需 gap 断裂"表明该规则只是数据停更护栏; 实测日线间隔
    分布: 正常 1~5 天 (周末/假日), 真停更 ≥7~18 天 (EURUSD)。故校准阈值 =
    7 天 (间隔 > 7 天断), 只断真实数据停更。"""
    m = len(c) - 1
    s = np.sign(np.diff(c))
    dt = np.diff(ts)
    thresh = params["gap_thresh_days"] * 86400
    gaps = (dt > thresh) & (np.arange(m) >= 1)
    return run_lengths(s, gaps)


def coin_runs(n_diffs, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice([1.0, -1.0], size=n_diffs)
    return run_lengths(s, np.zeros(n_diffs, bool))


def drift_gbm_runs(c_real, n_diffs, seed):
    lr = np.diff(np.log(c_real))
    fin = np.isfinite(lr)                    # CL=F 有负价数据瑕疵 → log=nan, 掩码剔除
    mu = float(np.mean(lr[fin]))
    sig = float(np.std(lr[fin], ddof=1))
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sig, size=n_diffs + 1)   # +1 → diff 数恰为 n_diffs
    c = 100.0 * np.exp(np.cumsum(r))
    s = np.sign(np.diff(c))
    return run_lengths(s, np.zeros(len(s), bool))


def n_ge(lengths, ks):
    return {k: int((lengths >= k).sum()) for k in ks}


def ci95(a):
    a = np.asarray(a, float)
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
            float(np.mean(a)), float(np.std(a, ddof=1)))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(dev_ref_n):
    """① run 状态机 golden 对拍 (0 延续 + gap 断);
    ② 硬币基准 sanity: 30 种子均值 N(run≥6) ≈ n/64 (iid 理论, ±50% 带)."""
    # ① golden: s = [1×5, 0×2, 1×3, -1×4, 0, gap, 1]
    s = np.array([1.0] * 5 + [0.0] * 2 + [1.0] * 3 + [-1.0] * 4 + [0.0] + [1.0])
    gaps = np.zeros(len(s), bool)
    gaps[len(s) - 1] = True
    lengths = run_lengths(s, gaps)
    exp = np.array([10, 5, 1])
    if not (len(lengths) == 3 and (lengths == exp).all()):
        raise SystemExit(
            f"GATE FAIL: golden run lengths {lengths.tolist()} ≠ [10, 5, 1] "
            f"— run 状态机错误")
    if n_ge(lengths, (6,))[6] != 1 or n_ge(lengths, (8,))[8] != 1 \
            or n_ge(lengths, (12,))[12] != 0:
        raise SystemExit("GATE FAIL: golden N(run≥k) 不符")
    # ② 硬币基准 sanity (参考 n)
    n = dev_ref_n
    n6s = []
    for seed in range(PARAMS["gbm_seeds"]):
        n6s.append(n_ge(coin_runs(n, seed), (6,))[6])
    mean6 = float(np.mean(n6s))
    theo = n / 64.0
    if not (0.5 * theo <= mean6 <= 1.5 * theo):
        raise SystemExit(
            f"GATE FAIL: 硬币基准 mean N6={mean6:.1f} ∉ [0.5,1.5]×n/64="
            f"{0.5 * theo:.1f}~{1.5 * theo:.1f} — 基准生成错误, 停")
    print(f"[GATE] run 状态机 golden [PASS]; 硬币基准 sanity mean N6={mean6:.1f} "
          f"vs n/64={theo:.1f} (n={n}) [PASS]", flush=True)
    return {"coin_mean6": mean6, "theo6": theo}


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def _hit(v, lo, hi, mode="upper"):
    if mode == "upper":
        return "超出↑" if v > hi else ("低于↓" if v < lo else "区间内")
    return "超出↑" if v > hi else ("低于↓" if v < lo else "区间内")


def write_out(out_path, params, g, rows, pool):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=control={},crypto={},k_main={},gap_mult={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["control"]), "+".join(p["crypto"]), p["k_main"],
            p["gap_mult"], p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(传统合并 N(run≥6)): 真实 {} | "
        "硬币均值 {} [PASS]; 探测器自检 run 状态机 golden [PASS]; 硬币基准 "
        "sanity (mean≈n/64) [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], pool["real6"], pool["coin_mean6"], p["min_n"]),
        "# RESULTS: [学习级] c33 日线运行肥尾忠实复现 (Q2 书原口径: 日线 + "
        "对称硬币 + run of 6); run=sign(close_t−close_{{t−1}}), 0 变化计延续, "
        "gap(>2×标准间隔)断; N(run≥k) 计数; 双基准: 对称硬币 (忠实) + 漂移 "
        "GBM (修正, c18); 描述层无入场, 无交易含义",
        "",
    ]
    # 表头行
    lines.append("[run≥k 计数] 每标的 N(run≥6/8/10) + N(run≥12):")
    for r in rows:
        lines.append("  {} (n_diff={}): N6={} N8={} N10={} N12={} | P12={:.4f} "
                     "({})".format(r["sym"], r["n"], r["N"][6], r["N"][8],
                                   r["N"][10], r["N"][12], r["P12"],
                                   _nm(r["N"][6], p["min_n"])))
    # H1 传统 vs 硬币
    lines.append("")
    lines.append("[H1] 传统日线 N(run≥6) vs 对称硬币 95% 区间 (30 种子):")
    cnt1 = 0
    for r in rows:
        if r["kind"] != "传统":
            continue
        lo, hi, mean, sd = r["coin_ci"]
        hit = _hit(r["N"][6], lo, hi)
        if r["N"][6] > hi:
            cnt1 += 1
        lines.append("  {}: 真实 N6={} | 硬币 [{:.0f}, {:.0f}] (mean {:.0f}, σ "
                     "{:.0f}) | {}".format(r["sym"], r["N"][6], lo, hi, mean,
                                           sd, hit))
    lines.append("  H1 判据: N6 超出硬币 95% 上界 -> {}/5".format(cnt1))
    # H2 加密
    lines.append("")
    lines.append("[H2] 加密日线 N(run≥6) vs 对称硬币 95% 区间 (30 种子):")
    cnt2 = 0
    for r in rows:
        if r["kind"] != "加密":
            continue
        lo, hi, mean, sd = r["coin_ci"]
        hit = _hit(r["N"][6], lo, hi)
        if r["N"][6] > hi:
            cnt2 += 1
        lines.append("  {}: 真实 N6={} | 硬币 [{:.0f}, {:.0f}] (mean {:.0f}, σ "
                     "{:.0f}) | {}".format(r["sym"], r["N"][6], lo, hi, mean,
                                           sd, hit))
    lines.append("  H2 判据: N6 超出硬币 95% 上界 -> {}/2".format(cnt2))
    # H3 漂移 GBM
    lines.append("")
    lines.append("[H3] 漂移 GBM 基准 (μ/σ 样本, 30 种子) 下真实仍超出:")
    cnt3 = 0
    for r in rows:
        lo, hi, mean, sd = r["gbm_ci"]
        hit = _hit(r["N"][6], lo, hi)
        if r["N"][6] > hi:
            cnt3 += 1
        lines.append("  {}: 真实 N6={} | 漂移GBM [{:.0f}, {:.0f}] (mean {:.0f}, "
                     "σ {:.0f}) | {}".format(r["sym"], r["N"][6], lo, hi,
                                             mean, sd, hit))
    lines.append("  H3 判据: 漂移 GBM 下真实仍超出 -> {}/7".format(cnt3))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c31 (2026-08-13): 1h/3y/GBM 口径 run 薄尾 (加密 "
                 "z8=-9.45, 传统 z8=-4.84) — 口径错位 (非书日线锚点); c18 "
                 "(2026-08-13): 4h 方向由无条件漂移主导 — 硬币基准会把漂移"
                 "误判为肥尾, 故 c33 双基准; 书 CH1/CH8: 日线 run of 6 超出"
                 "对称硬币 = 趋势利润来源")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    crypto = load_crypto_daily(PARAMS["crypto"])
    control = load_control_daily(PARAMS["control"], PARAMS["control_db"])
    if not crypto or not control:
        print("无数据, 退出")
        return 1

    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    if dev:
        control = [c for c in control if c[0] in PARAMS["dev_subset"]["control"]]

    g = gate(len(control[0][1]) - 1)

    rows = []
    for sym, df in control:
        c = df["close"].values
        ts = _epoch_seconds(df.index)
        n_diff = len(c) - 1
        lengths = real_runs(c, ts, PARAMS)
        N = n_ge(lengths, PARAMS["ks"])
        N[12] = int((lengths >= 12).sum())
        coin = [n_ge(coin_runs(n_diff, s), (6,))[6] for s in range(seeds)]
        cc = ci95(coin)
        gbm = [n_ge(drift_gbm_runs(c, n_diff, s), (6,))[6]
               for s in range(seeds)]
        gc = ci95(gbm)
        rows.append({"sym": sym, "kind": "传统", "n": n_diff, "N": N,
                     "P12": float((lengths >= 12).sum()) / len(lengths)
                     if len(lengths) else 0.0,
                     "coin_ci": cc, "gbm_ci": gc})
    for sym, df in crypto:
        c = df["close"].values
        ts = _epoch_seconds(df.index)
        n_diff = len(c) - 1
        lengths = real_runs(c, ts, PARAMS)
        N = n_ge(lengths, PARAMS["ks"])
        N[12] = int((lengths >= 12).sum())
        coin = [n_ge(coin_runs(n_diff, s), (6,))[6] for s in range(seeds)]
        cc = ci95(coin)
        gbm = [n_ge(drift_gbm_runs(c, n_diff, s), (6,))[6]
               for s in range(seeds)]
        gc = ci95(gbm)
        rows.append({"sym": sym, "kind": "加密", "n": n_diff, "N": N,
                     "P12": float((lengths >= 12).sum()) / len(lengths)
                     if len(lengths) else 0.0,
                     "coin_ci": cc, "gbm_ci": gc})

    if dev:
        for r in rows:
            lo, hi, m, sd = r["coin_ci"]
            print("  [dev] {} n={} N6={} | 硬币 [{:.0f},{:.0f}]".format(
                r["sym"], r["n"], r["N"][6], lo, hi))
        print(f"[dev] 管线 OK ({len(rows)} 标的 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    trad = [r for r in rows if r["kind"] == "传统"]
    cryp = [r for r in rows if r["kind"] == "加密"]
    pool = {
        "real6": sum(r["N"][6] for r in trad),
        "coin_mean6": sum(r["coin_ci"][2] for r in trad),
    }

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, rows, pool)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
