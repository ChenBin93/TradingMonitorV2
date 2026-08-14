#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C42 时代切片: 书时代 vs 后书时代市场结构 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c42): 用户感悟 — 书结论与实测出入来自市场环境变化
  (书时代 vs 后书时代)。定量化: 同一资产跨时代, 市场结构是否变了。
  度量: N(run≥6) vs 硬币 95% (c33 口径), ρ(1) (c38 口径), ER_10 中位数
  (c30 口径)。描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。
  结论标注 [学习级], **不得作交易依据**。学习级新协议: 不跑 pytest/
  check_study; 保留 docstring 预注册冻结、内置 GATE (SystemExit)、因果纪律、
  dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH1/CH2/CH8 的结论基于 1962-2012 数据
  (书时代)。本仓库实测 (2023-2026) 与其出入 — 市场结构是否随时代变了?
  ^TNX(1962-)/SPY(1993-)/CL=F(2000-)/GC=F(2000-)/EURUSD(2003-) 按时代切分,
  每时代测: 长 run 频率 (肥尾), ρ(1) (微观结构), ER_10 中位数 (噪声水平)。

预注册假设 (PLAN §2.5 c42 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 后书时代长 run 频率 z 值 < 前书时代 (肥尾衰减)
  H2: 后书时代 ρ(1) 更负 (做市/微观结构效应增强)
  H3: GBM 同管线各时代无差异 (null 不随时代变)

  操作化 (运行前锁定):
    - 时代切分: 书时代 ≤2010 vs 后书时代 2011+; ^TNX 另加 1962-1990 共三时代
    - 每时代度量 (真实 + GBM 30 种子同管线, μ/σ=各时代样本):
      a) N(run≥6) vs 硬币 95% 区间 (c33 口径: 纯符号, 0 计延续, gap>7 天断;
         硬币 iid ±1, n=各时代 diff 数, 30 种子) → z = (真实−硬币均值)/硬币σ
      b) ρ(1) = 对数收益 lag-1 自相关 (c38 口径)
      c) ER_10 中位数 (c30 口径)
    - H1 判据: 每标的 z(后书) < z(书时代) (计数裁决 + 均值)
    - H2 判据: 每标的 ρ(1)(后书) < ρ(1)(书时代) (计数裁决 + 均值)
    - H3 判据: GBM 各时代度量一致 (ρ(1) |跨时代差| < 0.02, ER |差| < 0.05,
      N6 相对 |差| < 30%)
    - 学习级: 30 种子、无 BY_YEAR (时代切片即分年)、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close 日线       | control.db 1d (双引号表名)            | bar 收盘后 | 对照数据源 max 历史
  时代切片         | 索引年份 ≤2010 / 2011+ (^TNX 三时代)  | 全样本     | 预注册 (时代即分年)
  run             | sign(close_t−close_{t−1}), 0 计延续,  | bar 收盘后 | c33 口径
                   |   gap>7 天断                         |            |
  ρ(1)            | 对数收益 lag-1 自相关 (中心化)        | 全样本事后 | c38 口径
  ER_10           | |C_t−C_{t−10}|/Σ|ΔC| 中位数           | bar 收盘后 | c30 口径
  硬币 null       | iid ±1, 30 种子, n=各时代 diff 数     | 锚定书口径 | c33 口径
  GBM null        | 各时代 μ/σ 样本, 30 种子              | 锚定真实   | 同管线分时代

数据声明:
  control.db 日线 max 历史: ^TNX 16,141 (1962-), SPY 8,442 (1993-),
  CL=F 6,521 (2000-), GC=F 6,512 (2000-), EURUSD=X 5,890 (2003-)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  k=6 (c33 锚点); gap_thresh_days=7; er_n=10; GBM/硬币 30 种子; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - 时代边界: 书时代 ≤2010 (书 2000-2012 主样本, oracle 校核取 ≤2010 以区分
    书/后书), 后书时代 2011+; ^TNX 1962-1990 为书时代前段 (三时代)。
  - GBM 用各时代 μ/σ 连续生成 (无 gap 结构, c33 同款); 真实侧 gap>7 天断 run。
  - 学习级: 无 BY_YEAR (时代切片即分年); 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① run golden (0 延续 + gap 断, c33 同款); ② 自相关 golden
    (交替 ρ1=−1, 斜坡 ρ1=1); ③ 硬币 sanity (mean N6 ≈ n/64); 任一失败
    SystemExit
  - GBM null sanity: 合并 GBM ρ(1) |均值| < 0.02 且 ER ∈ [0.1, 0.5]
  - MIN_N: 每时代每标的收益 n ≥ MIN_N=100
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: SPY + ^TNX × 3 种子, 不写 .out
  - 全量: 5 标的 × 各时代 × 30 种子

运行命令:
  python3 research/studies/c42_era_slices.py --dev
  python3 research/studies/c42_era_slices.py
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

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "control": ("^TNX", "SPY", "CL=F", "GC=F", "EURUSD=X"),
    "control_db": "data/control.db",
    "gap_thresh_days": 7.0,
    "er_n": 10,
    "k": 6,
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "eras_3": ("1962-1990", "1991-2010", "2011+"),
    "eras_2": ("书时代≤2010", "后书时代2011+"),
    "h3_bands": {"rho": 0.02, "er": 0.05, "n6": 0.30},
    "dev_subset": {"n_gbm": 3, "control": ("SPY", "^TNX")},
    "data_range": "control.db max 历史 (^TNX 1962-)",
}

STUDY_ID = "c42_era_slices"


# ── 装载 (control.db 日线全历史) ────────────────────────────
def load_daily(symbols, db_path):
    conn = sqlite3.connect(db_path)
    out = {}
    try:
        for sym in symbols:
            df = pd.read_sql_query(
                f'SELECT ts, close FROM "{sym}_1d" ORDER BY ts', conn)
            if df.empty:
                continue
            ts = pd.to_datetime(df["ts"], unit="s", utc=True)
            out[sym] = (df["close"].values.astype(float),
                        ts.dt.year.values.astype(int))
    finally:
        conn.close()
    return out


def era_slices(years, eras):
    """eras: [(y0, y1)] → [bool mask]"""
    out = []
    for y0, y1 in eras:
        m = np.ones(len(years), bool)
        if y0 is not None:
            m &= years >= y0
        if y1 is not None:
            m &= years <= y1
        out.append(m)
    return out


# ── run 状态机 (c33 口径: 0 计延续, gap 断) ─────────────────
def run_lengths(s, gaps):
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


def n6_of(close, ts=None, gap_days=7.0):
    c = np.asarray(close, float)
    if len(c) < 2:
        return 0, 0
    s = np.sign(np.diff(c))
    m = len(s)
    gaps = np.zeros(m, bool)
    if ts is not None and len(ts) == len(c):
        dt = np.diff(ts)
        gaps = (dt > gap_days * 86400) & (np.arange(m) >= 1)
    lengths = run_lengths(s, gaps)
    return int((lengths >= 6).sum()), len(lengths)


# ── 度量 (c38/c30 口径) ─────────────────────────────────────
def autocorr(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    v = float(np.mean(x * x))
    if v <= 0:
        return float("nan")
    n = len(x)
    t = np.arange(n)
    return float(np.mean(x[t < n - lag] * x[t >= lag]) / v)


def er_series(c, n):
    c = np.asarray(c, float)
    length = len(c)
    t = np.arange(length)
    c_prev = np.roll(c, 1)
    ad = np.where(t >= 1, np.abs(c - c_prev), 0.0)
    pref = np.concatenate([[0], np.cumsum(ad)])
    ok = t >= n
    net = np.full(length, np.nan)
    net[ok] = np.abs(c[t[ok]] - c[t[ok] - n])
    path = np.full(length, np.nan)
    path[ok] = pref[t[ok] + 1] - pref[t[ok] - n + 1]
    er = np.full(length, np.nan)
    m = ok & (path > 0)
    er[m] = net[m] / path[m]
    return er


def er_median(c, n):
    e = er_series(c, n)
    fin = np.isfinite(e)
    return float(np.nanmedian(e)) if fin.any() else float("nan")


def coin_n6(n_diffs, seeds):
    vals = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        s = rng.choice([1.0, -1.0], size=n_diffs)
        vals.append(int((run_lengths(s, np.zeros(n_diffs, bool)) >= 6).sum()))
    a = np.array(vals, float)
    return float(np.mean(a)), float(np.std(a, ddof=1))


def gbm_closes(n, mu, sig, seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, sig, size=n)
    return 100.0 * np.exp(np.cumsum(r))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(coin_mean_n6_ref, gbm_rho_mean, gbm_er_mean, n_ref):
    """① run golden (0 延续 + gap 断); ② 自相关 golden; ③ 硬币 sanity;
    ④ GBM null 幅度 sanity."""
    s = np.array([1.0] * 5 + [0.0] * 2 + [1.0] * 3 + [-1.0] * 4 + [0.0] + [1.0])
    gaps = np.zeros(len(s), bool)
    gaps[len(s) - 1] = True
    L = run_lengths(s, gaps)
    if not (len(L) == 3 and (L == np.array([10, 5, 1])).all()):
        raise SystemExit(f"GATE FAIL: run golden {L.tolist()} ≠ [10,5,1]")
    alt = np.array([1.0, -1.0] * 5000)
    if abs(autocorr(alt, 1) + 1.0) > 1e-9:
        raise SystemExit("GATE FAIL: 交替 ρ1 ≠ −1")
    theo = n_ref / 64.0
    if not (0.5 * theo <= coin_mean_n6_ref <= 1.5 * theo):
        raise SystemExit(f"GATE FAIL: 硬币 mean N6={coin_mean_n6_ref:.1f} "
                         f"∉ [0.5,1.5]×n/64")
    if abs(gbm_rho_mean) > 0.02 or not (0.1 <= gbm_er_mean <= 0.5):
        raise SystemExit(f"GATE FAIL: GBM ρ1={gbm_rho_mean:+.4f} 或 ER="
                         f"{gbm_er_mean:.3f} 幅度异常")
    print(f"[GATE] run/自相关 golden [PASS]; 硬币 sanity mean N6={coin_mean_n6_ref:.1f} "
          f"vs n/64={theo:.1f} [PASS]; GBM 幅度 ρ1={gbm_rho_mean:+.4f} "
          f"ER={gbm_er_mean:.3f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pp(v):
    return f"{v:+.3f}"


def write_out(out_path, params, rows, h1, h2, h3, gbm_pool):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=control={},k={},er_n={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["control"]), p["k"], p["er_n"], p["gbm_seeds"],
            p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(SPY 书时代 n): n={} [PASS]; 探测器"
        "自检 run/自相关 golden [PASS]; 硬币 sanity [PASS]; GBM null 幅度 "
        "ρ1 {:.4f} ER {:.3f} [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], gbm_pool["ref_n"], gbm_pool["rho"],
            gbm_pool["er"], p["min_n"]),
        "# RESULTS: [学习级] c42 时代切片 (书时代 vs 后书时代市场结构); "
        "N(run≥6) vs 硬币 95% (c33), ρ(1) (c38), ER_10 中位数 (c30); "
        "时代: ^TNX 三时代, 其余书≤2010/后书2011+; GBM 30 种子各时代同管线; "
        "描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[时代] 每标的×时代: N6 z (硬币) | ρ(1) | ER_10 中位数 | "
                 "GBM ρ(1)/ER:")
    for r in rows:
        lines.append("  {} {}: z={:+.2f} (N6={}) | ρ(1)={:.3f} | ER={:.3f} "
                     "(n={}) | GBM ρ1 {:.3f} ER {:.3f}".format(
            r["sym"], r["era"], r["z"], r["n6"], r["rho1"], r["er"],
            r["n"], r["gbm_rho"], r["gbm_er"]))
    # H1
    lines.append("")
    lines.append("[H1] 后书时代长 run z 值 < 书时代 (肥尾衰减):")
    for sym in h1:
        if sym.startswith("_"):
            continue
        z_pre, z_post = h1[sym]
        ok = z_post < z_pre
        lines.append("  {}: 书时代 z={:+.2f} → 后书 z={:+.2f} -> {}".format(
            sym, z_pre, z_post, "衰减✓" if ok else "未衰减"))
    lines.append("  H1 判据: z(后书) < z(书时代) -> {}/{}".format(
        h1["_n"], h1["_tot"]))
    # H2
    lines.append("")
    lines.append("[H2] 后书时代 ρ(1) 更负 (微观结构):")
    for sym in h2:
        if sym.startswith("_"):
            continue
        r_pre, r_post = h2[sym]
        ok = r_post < r_pre
        lines.append("  {}: 书时代 ρ1={:+.3f} → 后书 {:+.3f} -> {}".format(
            sym, r_pre, r_post, "更负✓" if ok else "未更负"))
    lines.append("  H2 判据: ρ(1)(后书) < ρ(1)(书时代) -> {}/{}".format(
        h2["_n"], h2["_tot"]))
    # H3
    lines.append("")
    lines.append("[H3] GBM 同管线各时代无差异:")
    for sym in h3:
        if sym.startswith("_"):
            continue
        eras_g = h3[sym]
        lines.append("  {}: 各时代 GBM ρ1 {} | ER {}".format(
            sym, [f"{g['rho']:+.3f}" for g in eras_g],
            [f"{g['er']:.3f}" for g in eras_g]))
    lines.append("  H3 判据: GBM 跨时代 ρ1|差|<0.02, ER|差|<0.05, N6 相对"
                 "<30% -> {}".format("PASS" if h3["_ok"] else "FAIL"))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c33 (日线 run 仅 ^TNX 肥尾); c38 (ρ(1) 近零/微负);"
                 " c30 (ER_10 频率持平); 书 CH1/CH2/CH8 (书时代 1962-2012)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    daily = load_daily(PARAMS["control"], PARAMS["control_db"])
    syms = PARAMS["dev_subset"]["control"] if dev else PARAMS["control"]
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    rows = []
    h1 = {"_n": 0, "_tot": 0}
    h2 = {"_n": 0, "_tot": 0}
    h3 = {}
    gbm_pool = {"rho": 0.0, "er": 0.0, "ref_n": 0}
    gbm_rho_all, gbm_er_all = [], []
    coin_ref = None

    for sym in syms:
        c_full, yarr = daily[sym]
        eras = ((1962, 1990), (1991, 2010), (2011, None)) if sym == "^TNX" \
            else ((None, 2010), (2011, None))
        emasks = era_slices(yarr, eras)
        sym_rows = []
        for i, m in enumerate(emasks):
            c = c_full[m]
            if len(c) < 110:
                continue
            era_label = (PARAMS["eras_3"] if sym == "^TNX"
                         else PARAMS["eras_2"])[i]
            n_diff = len(c) - 1
            # 真实度量 (负价数据瑕疵 → 有限对数收益掩码)
            n6, n_runs = n6_of(c)
            with np.errstate(invalid="ignore"):
                r = np.diff(np.log(c))
            fin = np.isfinite(r)
            r_fin = r[fin]
            rho1 = autocorr(r_fin, 1) if len(r_fin) >= 2 else float("nan")
            er = er_median(c, PARAMS["er_n"])
            # 硬币 null
            cm, cs = coin_n6(n_diff, seeds)
            z = (n6 - cm) / cs if cs > 0 else 0.0
            if coin_ref is None:
                coin_ref = (cm, n_diff)
            # GBM null (各时代 μ/σ, 有限值掩码)
            mu = float(np.mean(r_fin))
            sig = float(np.std(r_fin, ddof=1))
            g_n6, g_rho, g_er = [], [], []
            for seed in range(seeds):
                gc = gbm_closes(n_diff + 1, mu, sig, seed)
                g_n6.append(n6_of(gc)[0])
                g_rho.append(autocorr(np.diff(np.log(gc)), 1))
                g_er.append(er_median(gc, PARAMS["er_n"]))
            gb = {"n6": float(np.mean(g_n6)), "rho": float(np.mean(g_rho)),
                  "er": float(np.mean(g_er)), "n": n_diff}
            gbm_rho_all.append(gb["rho"])
            gbm_er_all.append(gb["er"])
            rows.append({"sym": sym, "era": era_label, "z": z, "n6": n6,
                         "rho1": rho1, "er": er, "n": n_diff,
                         "gbm_rho": gb["rho"], "gbm_er": gb["er"]})
            sym_rows.append((era_label, z, rho1, gb))
        if len(sym_rows) == 2 and sym != "^TNX":
            (pre, post) = sym_rows
            h1[sym] = (pre[1], post[1])
            h2[sym] = (pre[2], post[2])
            h1["_tot"] += 1
            h2["_tot"] += 1
            if post[1] < pre[1]:
                h1["_n"] += 1
            if post[2] < pre[2]:
                h2["_n"] += 1
        h3[sym] = [g for _, _, _, g in sym_rows]

    gbm_pool = {"rho": float(np.mean(gbm_rho_all)),
                "er": float(np.mean(gbm_er_all)),
                "ref_n": coin_ref[1] if coin_ref else 0}

    # H3 裁决 (N6 用率 N6/n, 原始计数随 n 缩放)
    h3_ok = True
    for sym, gs in h3.items():
        rhos = [g["rho"] for g in gs]
        ers = [g["er"] for g in gs]
        rates = [g["n6"] / max(1, g["n"]) for g in gs]
        if max(rhos) - min(rhos) > PARAMS["h3_bands"]["rho"]:
            h3_ok = False
        if max(ers) - min(ers) > PARAMS["h3_bands"]["er"]:
            h3_ok = False
        mr = np.mean(rates)
        if mr > 0 and (max(rates) - min(rates)) / mr > PARAMS["h3_bands"]["n6"]:
            h3_ok = False
    h3["_ok"] = h3_ok

    gate(coin_ref[0] if coin_ref else 0.0, gbm_pool["rho"], gbm_pool["er"],
         gbm_pool["ref_n"])

    if dev:
        for r in rows:
            print("  [dev] {} {} z={:+.2f} ρ1={:+.3f} ER={:.3f} (n={})".format(
                r["sym"], r["era"], r["z"], r["rho1"], r["er"], r["n"]))
        print(f"[dev] 管线 OK ({len(rows)} 格 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, rows, h1, h2, h3, gbm_pool)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
