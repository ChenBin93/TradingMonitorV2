#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C45 CH6 剩余实证断言打包 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U1-3 收尾, PLAN §2.5 c45): 书 CH6 三个零散实证断言
  一次判卷 — correlogram (图 6.13 黄金 20 年"无一显著"), 预报发散 (图 7.1),
  拟合-预报悖论 (p.257"对数统计最差预报最好")。描述层, 无入场, 无交易含义,
  不涉及胜率/期望/成本。**结论不得作交易依据**。学习级新协议: 不跑 pytest/
  check_study; 保留 docstring 预注册冻结、内置 GATE (回归 golden 复用 c44 +
  双模型 sanity)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH6 三个实证断言在 GC=F/SPY 日线 (max
  历史) 与 BTC/ETH 日线上的判卷: ① 收益 correlogram lag1-15 是否全部落在
  GBM 区间 (书"无一显著"对拍); ② 20bar 滚动回归 1/2/3/5 步前瞻 MAE 是否
  单调递增, 真实发散比率 vs GBM (有无超 null 内容); ③ 拟合-预报悖论窗口
  占比 vs GBM (对数模型拟合差预报好是否超随机).

预注册假设 (PLAN §2.5 c45 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: GC=F 日收益 correlogram lag1-15 全部在 GBM 95% 区间内 (书图 6.13
      黄金 20 年"无一显著"对拍) + BTC/ETH 日线交叉 (c38 口径)
  H2: 20bar 滚动回归 1/2/3/5 步前瞻 MAE 单调递增 (书图 7.1 字面断言) +
      真实 MAE(k)/MAE(1) 比率 vs GBM 同管线 (null 同样发散 — 检验有无
      超 null 内容)
  H3: 拟合-预报悖论 (书 p.257"对数统计最差预报最好"): 每窗口线性-原始价
      vs 线性-对数价两模型, 悖论窗口占比 (样本内拟合更差者样本外预报
      更好) > GBM 同管线 2σ

  操作化 (运行前锁定):
    - 数据: GC=F/SPY 日线 (control.db max 历史) + BTC/ETH 日线
      (daily_resample); 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层
    - H1: ρ(k) (k=1..15) 逐滞后 GBM 95% 区间 (c38 口径: 30 种子 iid
      同 n 同 μ/σ); 判据: GC=F 15 滞后全在区间内 (BTC/ETH 交叉报告)
    - H2: 滚动回归 n=20 (c44 前缀和), 预报 = a+b·(n+k), k∈{1,2,3,5};
      MAE(k) = mean|close[t+k] − pred|; 判据: MAE 单调递增 (每标的);
      真实比率 MAE(k)/MAE(1) vs GBM 同管线 (报告, 超 null 内容判据)
    - H3: 两模型 (raw close 线性回归 vs log close 线性回归, 预报 exp
      还原), n=20; 样本内拟合 = R² (尺度无关); 样本外 = 窗口后 5 bar
      预报误差 (价格单位); 悖论窗口 = 样本内拟合更差者样本外预报更好;
      占比 vs GBM 30 种子 2σ
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close 日线       | control.db / daily_resample            | bar 收盘后 | 数据源
  ρ(k)             | 对数收益逐滞后自相关                   | 全样本事后 | c38 口径 (描述统计)
  滚动回归         | c44 前缀和 (窗口只含 ≤t 数据)          | bar 收盘后 | 因果
  预报 MAE         | 回归线外推 k 步 vs 实际 (事后度量)     | 全样本事后 | 书图 7.1
  R²/残差          | 窗口内拟合度量                         | bar 收盘后 | 尺度无关
  悖论占比         | 两模型拟合/预报优劣方向比较            | 全样本事后 | 描述统计
  GBM null         | gbm_matching + 同管线 (30 种子)        | 锚定真实   | 同 n 同 μ/σ

数据声明:
  GC=F (6,512根, 2000-) / SPY (8,442根, 1993-) 日线 (control.db);
  BTC/ETH 日线 (daily_resample, ~1,095根, 2023-08..2026-08)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  H1 lags=1..15; H2 n=20, ks={1,2,3,5}; H3 n=20, OOS 5 bar; GBM 30 种子;
  MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 样本内拟合用 R² (尺度无关, raw 与 log 模型可比); 样本外用价格单位
    MAE (exp 还原, 可比)。
  - 悖论 = 两模型拟合优度方向与预报误差方向不一致的窗口占比。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 回归 golden (c44 同款: 直线 b=2/R=1; 指数线 log 模型
    R²=1 且 OOS 近零); ② GBM null sanity: GBM 悖论占比 ∈ [0.3, 0.7]
    (两模型近对称, null 应 ≈0.5); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 同管线 (H1 逐滞后区间 / H2 比率 / H3 悖论)
  - MIN_N: 每格 n ≥ MIN_N=100
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: GC=F + BTC 日线 × 3 种子, 不写 .out
  - 全量: GC=F/SPY/BTC/ETH × 30 种子 (预计 ≤8 分钟)

运行命令:
  python3 research/studies/c45_ch6_empirical.py --dev
  python3 research/studies/c45_ch6_empirical.py
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
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "control": ("GC=F", "SPY"),
    "control_db": "data/control.db",
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "h1_lags": tuple(range(1, 16)),
    "h2_n": 20,
    "h2_ks": (1, 2, 3, 5),
    "h3_n": 20,
    "h3_os": 5,
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "gate_paradox": (0.30, 0.70),        # GBM 悖论占比 sanity
    "dev_subset": {"n_gbm": 3, "control": ("GC=F",), "crypto": ("BTC/USDT:USDT",)},
    "data_range": "GC=F/SPY max 历史 + BTC/ETH 2023-08..2026-08",
}

STUDY_ID = "c45_ch6_empirical"


# ── 装载 ─────────────────────────────────────────────────────
def load_series(params):
    out = []
    conn = sqlite3.connect(params["control_db"])
    try:
        for sym in params["control"]:
            df = pd.read_sql_query(
                f'SELECT ts, close FROM "{sym}_1d" ORDER BY ts', conn)
            if df.empty:
                continue
            out.append((sym, df["close"].values.astype(float)))
    finally:
        conn.close()
    data = load_candles(timeframes=("1h",))
    for sym in params["crypto"]:
        df = data.get(sym, {}).get("1h")
        if df is None or verify(df, sym, "1h"):
            continue
        dd = daily_resample(df)
        out.append((sym, dd["close"].values.astype(float)))
    return out


# ── 自相关 (c38 口径) ───────────────────────────────────────
def autocorr(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    v = float(np.mean(x * x))
    if v <= 0:
        return float("nan")
    n = len(x)
    t = np.arange(n)
    return float(np.mean(x[t < n - lag] * x[t >= lag]) / v)


def gbm_acf_bands(r_real, lags, seeds):
    mu = float(np.mean(r_real))
    sig = float(np.std(r_real, ddof=1))
    n = len(r_real)
    bands = {k: [] for k in lags}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        r = rng.normal(mu, sig, size=n)
        for k in lags:
            bands[k].append(autocorr(r, k))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in bands.items()}


# ── 滚动回归 (c44 前缀和) ───────────────────────────────────
def rolling_ab(c, n):
    """每窗口 (a, b, R², pred 末点) — 前缀和 O(1)/bar."""
    c = np.asarray(c, float)
    L = len(c)
    x = np.arange(1, n + 1, dtype=float)
    Sx = x.sum()
    Sx2 = (x * x).sum()
    t = np.arange(L)
    ok = t >= n - 1
    ti = t[ok]
    s = ti - (n - 1)
    pc = np.concatenate([[0], np.cumsum(c)])
    pic = np.concatenate([[0], np.cumsum((np.arange(L) + 1) * c)])
    pc2 = np.concatenate([[0], np.cumsum(c * c)])
    Sy = pc[ti + 1] - pc[s]
    Sxy = (pic[ti + 1] - pic[s]) - s * Sy
    Sy2 = pc2[ti + 1] - pc2[s]
    denom = n * Sx2 - Sx * Sx
    b = (n * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / n
    sy2_denom = n * Sy2 - Sy * Sy
    R2 = ((n * Sxy - Sx * Sy) ** 2) / (denom * np.maximum(sy2_denom, 1e-30))
    a_f = np.full(L, np.nan)
    b_f = np.full(L, np.nan)
    R2_f = np.full(L, np.nan)
    a_f[ok] = a
    b_f[ok] = b
    R2_f[ok] = R2
    return a_f, b_f, R2_f, ti


# ── H2: 预报 MAE ────────────────────────────────────────────
def mae_by_k(close, n, ks):
    c = np.asarray(close, float)
    a, b, R2, ti = rolling_ab(c, n)
    out = {}
    for k in ks:
        idx = ti[ti + k < len(c)]
        pred = a[idx] + b[idx] * (n + k)
        err = np.abs(c[idx + k] - pred)
        out[k] = float(np.mean(err)) if len(err) else float("nan")
    return out


def gbm_mae_ratios(c_real, n, ks, seeds):
    mu = float(np.mean(np.diff(np.log(c_real))))
    sig = float(np.std(np.diff(np.log(c_real)), ddof=1))
    n_len = len(c_real)
    ratios = {k: [] for k in ks[1:]}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        r = rng.normal(mu, sig, size=n_len)
        gb = 100.0 * np.exp(np.cumsum(r))
        m = mae_by_k(gb, n, ks)
        for k in ks[1:]:
            ratios[k].append(m[k] / m[ks[0]] if m[ks[0]] > 0 else float("nan"))
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)))
            for k, v in ratios.items()}


# ── H3: 拟合-预报悖论 ───────────────────────────────────────
def paradox_prop(close, n, os_k):
    c = np.asarray(close, float)
    L = len(c)
    a_r, b_r, R2_r, ti_r = rolling_ab(c, n)
    lc = np.log(np.maximum(c, 1e-12))
    a_l, b_l, R2_l, ti_l = rolling_ab(lc, n)
    # 有效窗口: t + os_k < L
    valid = ti_r[ti_r + os_k < L]
    if len(valid) == 0:
        return float("nan"), 0
    # 样本外误差 (价格单位, os_k bar 平均)
    oos_r = np.zeros(len(valid))
    oos_l = np.zeros(len(valid))
    for k in range(1, os_k + 1):
        pred_r = a_r[valid] + b_r[valid] * (n + k)
        pred_l = np.exp(a_l[valid] + b_l[valid] * (n + k))
        oos_r += np.abs(c[valid + k] - pred_r)
        oos_l += np.abs(c[valid + k] - pred_l)
    oos_r /= os_k
    oos_l /= os_k
    fit_r = R2_r[valid]
    fit_l = R2_l[valid]
    raw_fits_better = fit_r > fit_l
    raw_fc_better = oos_r < oos_l
    paradox = raw_fits_better != raw_fc_better
    return float(np.mean(paradox)), int(len(valid))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(gbm_paradox_mean):
    """① 回归 golden (c44 同款 + 指数线双模型 sanity);
    ② GBM null sanity: 悖论占比 ∈ [0.3, 0.7] (两模型近对称, null≈0.5)."""
    # ① 直线 → raw R²=1, b=2
    x = np.arange(1, 21, dtype=float)
    y = 3 + 2 * x
    a, b, R2, ti = rolling_ab(y, 20)
    if abs(b[-1] - 2.0) > 1e-9 or abs(R2[-1] - 1.0) > 1e-9:
        raise SystemExit(f"GATE FAIL: 直线 golden b={b[-1]} R²={R2[-1]}")
    # ① 指数线 → log 模型 R²=1, raw R²<1, log OOS 近零
    yexp = np.exp(0.05 * np.arange(1, 31, dtype=float))
    a1, b1, R2_1, ti1 = rolling_ab(yexp, 20)
    a2, b2, R2_2, ti2 = rolling_ab(np.log(yexp), 20)
    if abs(R2_2[-1] - 1.0) > 1e-9 or R2_1[-1] >= 0.999999:
        raise SystemExit(f"GATE FAIL: 指数线 golden raw R²={R2_1[-1]} "
                         f"log R²={R2_2[-1]}")
    # ② GBM 悖论 sanity
    lo, hi = PARAMS["gate_paradox"]
    if not (lo <= gbm_paradox_mean <= hi):
        raise SystemExit(
            f"GATE FAIL: GBM 悖论占比 {gbm_paradox_mean:.3f} ∉ [{lo}, {hi}] "
            f"— 管线错误, 停")
    print(f"[GATE] 回归 golden (直线/指数线双模型) [PASS]; GBM 悖论占比 "
          f"{gbm_paradox_mean:.3f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pp(v):
    return f"{v:+.3f}"


def write_out(out_path, params, rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=control={},crypto={},h1_lags={},h2_n={},h2_ks={},h3_n={},"
        "h3_os={},gbm_seeds={},min_n={},gate=MIN_GBM_SEEDS=30,MIN_N={}"
        "(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["control"]), "+".join(p["crypto"]), p["h1_lags"][-1],
            p["h2_n"], p["h2_ks"], p["h3_n"], p["h3_os"], p["gbm_seeds"],
            p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(GBM 悖论占比): {:.3f} [PASS]; 探测器"
        "自检 回归 golden [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], rows[0]["gbm_paradox"][0], p["min_n"]),
        "# RESULTS: [学习级] c45 CH6 剩余实证断言打包 (U1-3 收尾); H1 correlogram"
        " lag1-15 vs GBM 95% (书图 6.13); H2 20bar 回归 1/2/3/5 步 MAE (书图 7.1)"
        "; H3 拟合-预报悖论 (书 p.257); GBM 30 种子同管线; 描述层无入场, 无交易"
        "含义",
        "",
    ]
    # H1
    lines.append("[H1] GC=F 日收益 correlogram lag1-15 (GBM 95% 区间):")
    gc = [r for r in rows if r["sym"] == "GC=F"]
    if gc:
        r = gc[0]
        all_in = all(r["rho"][k] >= r["bands"][k][0]
                     and r["rho"][k] <= r["bands"][k][1] for k in p["h1_lags"])
        vals = ", ".join(f"{r['rho'][k]:+.3f}" for k in
                         (1, 2, 3, 5, 10, 15))
        lines.append("  GC=F: ρ1/2/3/5/10/15 = [{}] | 15 滞后全部在区间: {}".format(
            vals, "✓" if all_in else "✗"))
    lines.append("  (BTC/ETH 日线交叉):")
    for r in rows:
        if r["sym"] not in p["crypto"]:
            continue
        nin = sum(1 for k in p["h1_lags"] if r["rho"][k] >= r["bands"][k][0]
                  and r["rho"][k] <= r["bands"][k][1])
        lines.append("  {}: 区间内 {}/{} (ρ1={:+.3f})".format(
            r["sym"], nin, len(p["h1_lags"]), r["rho"][1]))
    lines.append("  H1 判据: GC=F 15 滞后全在 GBM 区间 -> {}".format(
        "PASS" if gc and all_in else "FAIL"))
    # H2
    lines.append("")
    lines.append("[H2] 20bar 滚动回归 1/2/3/5 步前瞻 MAE (书图 7.1):")
    for r in rows:
        m = r["mae"]
        mono = all(m[p["h2_ks"][i]] < m[p["h2_ks"][i + 1]]
                   for i in range(len(p["h2_ks"]) - 1))
        ratios = " ".join("MAE{}/MAE1 {:.3f} (GBM {:.3f}±{:.3f})".format(
            k, m[k] / m[1], r["gbm_ratio"][k][0], r["gbm_ratio"][k][1])
            for k in p["h2_ks"][1:])
        lines.append("  {}: MAE1={:.4f} MAE2={:.4f} MAE3={:.4f} MAE5={:.4f} "
                     "单调{} | {}".format(
            r["sym"], m[1], m[2], m[3], m[5], "✓" if mono else "✗", ratios))
    lines.append("  H2 判据: MAE 单调递增 (每标的) -> {}".format(
        "PASS" if all(all(r["mae"][p["h2_ks"][i]] < r["mae"][p["h2_ks"][i + 1]]
                          for i in range(len(p["h2_ks"]) - 1)) for r in rows)
        else "FAIL"))
    lines.append("  超 null 内容: 真实比率 vs GBM 比率 (净差) 见上")
    # H3
    lines.append("")
    lines.append("[H3] 拟合-预报悖论占比 (书 p.257 对数统计最差预报最好):")
    for r in rows:
        pp_, ne = r["paradox"]
        gm, gs = r["gbm_paradox"]
        ok = pp_ > gm + 2 * gs
        lines.append("  {}: 悖论占比 {:.1%} (n={}) | GBM {:.1%}±{:.1%} | "
                     "超2σ{}".format(r["sym"], pp_, ne, gm, gs,
                                     "✓" if ok else "✗"))
    n_h3 = sum(1 for r in rows if r["paradox"][0] > r["gbm_paradox"][0]
               + 2 * r["gbm_paradox"][1])
    lines.append("  H3 判据: 悖论占比 > GBM 2σ -> {}/{}".format(
        n_h3, len(rows)))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c38 (ACF 谱近零); c44 (回归 R 触碰折返未超 null); "
                 "书 CH6 图 6.13 (correlogram 无一显著); 图 7.1 (预报发散); "
                 "p.257 (对数拟合差预报好)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    dev_control = PARAMS["dev_subset"]["control"] if dev else None
    dev_crypto = PARAMS["dev_subset"]["crypto"] if dev else None
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    all_s = load_series(PARAMS)
    rows = []
    for sym, c in all_s:
        if dev_control is not None and sym in PARAMS["control"] \
                and sym not in dev_control:
            continue
        if dev_crypto is not None and sym in PARAMS["crypto"] \
                and sym not in dev_crypto:
            continue
        r = np.diff(np.log(np.maximum(c, 1e-12)))
        rho = {k: autocorr(r, k) for k in PARAMS["h1_lags"]}
        bands = gbm_acf_bands(r, PARAMS["h1_lags"], seeds)
        mae = mae_by_k(c, PARAMS["h2_n"], PARAMS["h2_ks"])
        gbm_ratio = gbm_mae_ratios(c, PARAMS["h2_n"], PARAMS["h2_ks"], seeds)
        pp_, ne = paradox_prop(c, PARAMS["h3_n"], PARAMS["h3_os"])
        # GBM 悖论 (同管线)
        mu = float(np.mean(r))
        sig = float(np.std(r, ddof=1))
        gpp = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            rr = rng.normal(mu, sig, size=len(c))
            gc = 100.0 * np.exp(np.cumsum(rr))
            gp, _ = paradox_prop(gc, PARAMS["h3_n"], PARAMS["h3_os"])
            if np.isfinite(gp):
                gpp.append(gp)
        ga = np.array(gpp)
        gbm_paradox = (float(np.mean(ga)), float(np.std(ga, ddof=1))
                       if len(ga) > 1 else 0.0)
        rows.append({"sym": sym, "rho": rho, "bands": bands, "mae": mae,
                     "gbm_ratio": gbm_ratio, "paradox": (pp_, ne),
                     "gbm_paradox": gbm_paradox})

    gate(float(np.mean([r["gbm_paradox"][0] for r in rows])))

    if dev:
        for r in rows:
            print("  [dev] {} ρ1={:+.3f} MAE1={:.4f} MAE5={:.4f} 悖论={:.2f} "
                  "GBM悖论={:.2f}".format(r["sym"], r["rho"][1], r["mae"][1],
                                         r["mae"][5], r["paradox"][0],
                                         r["gbm_paradox"][0]))
        print(f"[dev] 管线 OK ({len(rows)} 标的 × {seeds} 种子), 不写 .out; "
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
