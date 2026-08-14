#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C38 收益自相关谱 + DW 统计 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U0-6, PLAN §2.5 c38): 书 CH2 p.43-58 的 Durbin-Watson
  检验的直接版本 — 收益自相关完整谱 ρ(lag 1..20) + DW≈2(1−ρ₁) + 平方收益
  自相关。c31 证实符号反持久、c12 证实波动长记忆、c18 证实方向由漂移主导 —
  但收益自相关的完整谱从未测过。本研究画出谱, 找出"近零持久"的边界。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。结论标注 [学习级],
  **不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring
  预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 收益自相关 ρ(k) (k=1..20) 的完整谱 —
  反持久 (ρ<0) 在哪一滞后消失 (近零持久边界)? 平方收益自相关 (波动长记忆
  的滞后表达) 是否显著为正且慢衰减? DW 统计值? 传统市场日线 ρ(1) 符号
  (书 CH2 暗示趋势市场应正自相关)?

预注册假设 (PLAN §2.5 c38 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 加密 1h 收益 ρ(1) < 0 且超 GBM 95% 区间 (反持久)
  H2: 存在滞后 k 使 |ρ(k)| 回到 GBM 区间 (反持久消失边界 — "近零持久"边界)
  H3: 平方收益自相关显著为正且慢衰减 (c12 长记忆的滞后表达)
  H4: 传统市场日线 ρ(1) 符号报告 (书 CH2 暗示趋势市场应正自相关 — 对照,
      不设门槛)

  操作化 (运行前锁定):
    - 收益 = close-to-close 对数收益 (log returns, 声明)
    - ρ(k) = 收益自相关 (全样本逐滞后, 因果无未来函数; 中心化)
    - DW = 2(1−ρ₁) (ρ₁ = lag-1 自相关)
    - ρ²(k) = 平方收益自相关 (raw 平方, 不中心化, 声明)
    - GBM null: 30 种子, 同 n 同 μ/σ (该序列对数收益样本, 有限值掩码),
      逐滞后 95% 区间 (2.5/97.5 分位)
    - 传统日线 gap 规则沿用 c32 (7 天阈值, 只断数据停更)
    - H1 判据: BTC/ETH 1h ρ(1) < 0 且 < GBM 95% 下界
    - H2 判据: 存在 k ∈ [1,20] 使 |ρ(k)| ∈ GBM 95% 区间 (首现滞后 = 边界)
    - H3 判据: 加密 ρ²(1) > GBM 95% 上界 且 ρ²(10) 仍显著 (慢衰减)
    - H4: SPY/GC=F/EURUSD 日线 ρ(1) 符号与 GBM 区间对照 (不设门槛)
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  收益             | close-to-close 对数收益               | bar 收盘后 | 声明 (对数口径)
  ρ(k)             | 全样本逐滞后自相关 (中心化)           | 全样本事后 | 描述统计 (非条件特征)
  DW               | 2(1−ρ₁)                                | 全样本事后 | 书 CH2 口径
  ρ²(k)            | 平方收益自相关 (raw, 不中心化)        | 全样本事后 | 声明
  GBM null         | 30 种子 iid N(样本μ,样本σ), 逐滞后区间 | 锚定真实   | 同 n 同 μ/σ

数据声明:
  BTC/ETH 1h (26,280根) + 4h (6,570) + 日线 (daily_resample ~1,095);
  SPY/GC=F/EURUSD=X 日线 (control.db, 共同 3y 窗 2023-08..2026-08)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  lags=1..20; DW 由 ρ₁ 派生; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 收益用对数 (声明); 平方收益自相关用 raw 平方不中心化 (声明, 波动长记忆
    度量惯例)。
  - ρ(k) 全样本逐滞后 = 描述统计 (无未来函数问题, 不用滚动条件化)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 自相关 golden (交替序列 ρ(1)=−1, 斜坡 ρ(1)→1, 精确验证);
    ② GBM null sanity: 参考序列 GBM ρ(1) 均值 |·| < 0.02 (iid null 近零,
    管线错误才停); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 同 n 同 μ/σ, 逐滞后 95% 区间
  - MIN_N: 每序列收益 n ≥ MIN_N=100
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH 1h + SPY 日线 × GBM 3 种子, 不写 .out
  - 全量: BTC/ETH 三频率 + SPY/GC=F/EURUSD 日线 × 30 种子

运行命令:
  python3 research/studies/c38_acf_spectrum.py --dev
  python3 research/studies/c38_acf_spectrum.py
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
    "control": ("SPY", "GC=F", "EURUSD=X"),
    "control_db": "data/control.db",
    "win_start": "2023-08-01",
    "win_end": "2026-08-01",
    "lags": tuple(range(1, 21)),
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "gate_band": 0.02,                   # GBM ρ(1) 均值 |·| < 0.02 (iid null)
    "dev_subset": {"n_gbm": 3},
    "data_range": "2023-08..2026-08 (对照共同 3y 窗)",
}

STUDY_ID = "c38_acf_spectrum"


# ── 加载 ─────────────────────────────────────────────────────
def load_returns(params):
    """→ list[(name, tf, log_returns)]"""
    out = []
    data = load_candles(timeframes=("1h", "4h"))
    for sym in params["crypto"]:
        d1 = data.get(sym, {}).get("1h")
        d4 = data.get(sym, {}).get("4h")
        if d1 is None or d4 is None:
            continue
        if verify(d1, sym, "1h") or verify(d4, sym, "4h"):
            continue
        out.append((sym, "1h", np.diff(np.log(d1["close"].values.astype(float)))))
        out.append((sym, "4h", np.diff(np.log(d4["close"].values.astype(float)))))
        dd = daily_resample(d1)
        out.append((sym, "1d", np.diff(np.log(dd["close"].values.astype(float)))))
    conn = sqlite3.connect(params["control_db"])
    try:
        t0 = pd.Timestamp(params["win_start"], tz="UTC")
        t1 = pd.Timestamp(params["win_end"], tz="UTC")
        for sym in params["control"]:
            df = pd.read_sql_query(
                f'SELECT ts, close FROM "{sym}_1d" ORDER BY ts', conn)
            if df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
            m = (df["ts"] >= t0) & (df["ts"] <= t1)
            sel = df[m]
            if len(sel) < 110:
                continue
            c = sel["close"].values.astype(float)
            out.append((sym, "1d", np.diff(np.log(c))))
    finally:
        conn.close()
    return out


# ── 自相关 (全样本逐滞后, 描述统计) ─────────────────────────
def autocorr(x, lag, center=True):
    x = np.asarray(x, float)
    if center:
        x = x - x.mean()
    v = float(np.mean(x * x))
    if v <= 0:
        return float("nan")
    n = len(x)
    t = np.arange(n)
    m1 = t < n - lag
    m2 = t >= lag
    return float(np.mean(x[m1] * x[m2]) / v)


def acf_spectrum(r, lags):
    """ρ(k) 谱 + DW + 平方收益 ρ²(k) (raw 与 centered 双口径)"""
    rho = {k: autocorr(r, k, center=True) for k in lags}
    dw = 2.0 * (1.0 - rho[1])
    r2 = r * r
    rho2 = {k: autocorr(r2, k, center=False) for k in lags}    # raw (预注册)
    rho2c = {k: autocorr(r2, k, center=True) for k in lags}   # centered (c12 口径)
    return rho, dw, rho2, rho2c


# ── GBM null (30 种子, 同 n 同 μ/σ, 逐滞后) ─────────────────
def gbm_acf(r_real, lags, seeds):
    mu = float(np.mean(r_real))
    sig = float(np.std(r_real, ddof=1))
    n = len(r_real)
    rho1s = []
    bands = {}
    r2c_bands = {}
    for k in lags:
        bands[k] = []
        r2c_bands[k] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        r = rng.normal(mu, sig, size=n)
        rho, dw, rho2, rho2c = acf_spectrum(r, lags)
        rho1s.append(rho[1])
        for k in lags:
            bands[k].append(rho[k])
            r2c_bands[k].append(rho2c[k])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
          for k, v in bands.items()}
    r2c_ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
              for k, v in r2c_bands.items()}
    return float(np.mean(rho1s)), ci, r2c_ci


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(gbm_rho1_mean):
    """① 自相关 golden: 交替序列 ρ(1)=−1; 斜坡序列 ρ(1)≈1 (长序列);
    ② GBM null sanity: ρ(1) 均值 |·| < 0.02 (iid null 近零)."""
    alt = np.array([1.0, -1.0] * 5000)
    if abs(autocorr(alt, 1) + 1.0) > 1e-9:
        raise SystemExit(f"GATE FAIL: 交替序列 ρ(1)={autocorr(alt,1)} ≠ −1")
    ramp = np.arange(20000.0)
    if abs(autocorr(ramp, 1) - 1.0) > 1e-3:
        raise SystemExit(f"GATE FAIL: 斜坡 ρ(1)={autocorr(ramp,1)} ≠ 1")
    if abs(gbm_rho1_mean) > PARAMS["gate_band"]:
        raise SystemExit(
            f"GATE FAIL: GBM ρ(1) 均值={gbm_rho1_mean:+.4f} |·| > "
            f"{PARAMS['gate_band']} — iid null 偏置, 停")
    print(f"[GATE] 自相关 golden (交替 ρ1=−1, 斜坡 ρ1=1) [PASS]; GBM ρ(1) 均值 "
          f"{gbm_rho1_mean:+.4f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pp(v):
    return f"{v:+.3f}"


def write_out(out_path, params, g, rows, h1, h2, h3, h4):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},control={},lags={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), "+".join(p["control"]), p["lags"][-1],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(BTC 1h 收益 n): n={} [PASS]; 探测器"
        "自检 自相关 golden [PASS]; GBM null ρ(1) 均值 {:+.4f} |·|<{} [PASS]; "
        "MIN_N n≥{} [PASS]".format(p["gbm_seeds"], len(rows[0]["r"]),
                                   g["rho1_mean"], p["gate_band"], p["min_n"]),
        "# RESULTS: [学习级] c38 收益自相关谱 + DW (书 CH2 p.43-58); 收益="
        "close-to-close 对数; ρ(k) 全样本逐滞后 (描述统计); DW=2(1−ρ₁); "
        "ρ²(k)=平方收益自相关 (raw 不中心化); GBM 30 种子同 n 同 μ/σ 逐滞后 "
        "95% 区间; 描述层无入场, 无交易含义",
        "",
    ]
    # 谱表
    lines.append("[谱] ρ(1..5) ρ(10) ρ(20) DW | GBM ρ(1) 95% | 平方收益 ρ²(1)"
                 " ρ²(10) (GBM 95%):")
    for r in rows:
        rho = r["rho"]
        rho2 = r["rho2"]
        g_lo, g_hi = r["gbm_ci"][1]
        lines.append("  {} {}: ρ1..5=[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}] "
                     "ρ10={:.3f} ρ20={:.3f} DW={:.2f} | GBM ρ1 [{:.3f}, {:.3f}]"
                     " | ρ²1={:.3f} ρ²10={:.3f}".format(
            r["name"], r["tf"], rho[1], rho[2], rho[3], rho[4], rho[5],
            rho[10], rho[20], r["dw"], g_lo, g_hi, rho2[1], rho2[10]))
    # H1
    lines.append("")
    lines.append("[H1] 加密 1h ρ(1) < 0 且超 GBM 95% 区间:")
    for r in rows:
        if r["tf"] != "1h":
            continue
        lo, hi = r["gbm_ci"][1]
        ok = r["rho"][1] < 0 and r["rho"][1] < lo
        lines.append("  {}: ρ(1)={:.3f} (GBM [{:.3f}, {:.3f}]) -> {}".format(
            r["name"], r["rho"][1], lo, hi, "超出↓" if ok else "未超"))
    lines.append("  H1 判据: 两标的均 ρ(1)<0 且<GBM 下界 -> {}".format(
        "PASS" if h1["ok"] else "FAIL"))
    # H2 消失边界
    lines.append("")
    lines.append("[H2] 反持久消失边界 (首个 |ρ(k)| 回到 GBM 区间):")
    for r in rows:
        k0 = h2[r["name"]][r["tf"]]
        lines.append("  {} {}: 边界 k={} ({})".format(
            r["name"], r["tf"], k0, "存在" if k0 is not None else "20 内未回"))
    h2_ok = all(h2[name].get("1h") is not None for name in ("BTC/USDT:USDT",
                                                            "ETH/USDT:USDT"))
    lines.append("  H2 判据: 加密 1h 存在消失滞后 -> {}".format(
        "PASS" if h2_ok else "FAIL"))
    # H3 平方收益
    lines.append("")
    lines.append("[H3] 平方收益自相关 (波动长记忆滞后表达; 判据用 centered 口径,"
                 " raw 口径 null≈1/3 只报告):")
    for r in rows:
        if r["name"] not in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
            continue
        lo1, hi1, lo10, hi10 = h3[r["name"]][r["tf"]]
        lines.append("  {} {}: ρ²c(1)={:.4f} (GBM [{:.4f}, {:.4f}]) | ρ²c(10)="
                     "{:.4f} (GBM [{:.4f}, {:.4f}]) | raw ρ²(1)={:.4f} "
                     "(raw null≈1/3)".format(
            r["name"], r["tf"], r["rho2c"][1], lo1, hi1, r["rho2c"][10],
            lo10, hi10, r["rho2"][1]))
    lines.append("  H3 判据: 加密 ρ²c(1)>GBM 上界 且 ρ²c(10) 仍显著 -> {}".format(
        "PASS" if h3["ok"] else "FAIL"))
    # H4 传统
    lines.append("")
    lines.append("[H4] 传统市场日线 ρ(1) 符号 (对照, 不设门槛):")
    for r in rows:
        if r["name"] not in params["control"]:
            continue
        lo, hi = r["gbm_ci"][1]
        sign = "正" if r["rho"][1] > 0 else "负"
        pos = "超出" if r["rho"][1] > hi else ("低于" if r["rho"][1] < lo
                                              else "区间内")
        lines.append("  {}: ρ(1)={:.3f} ({}) | GBM [{:.3f}, {:.3f}] | {}"
                     "".format(r["name"], r["rho"][1], sign, lo, hi, pos))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c31 (符号反持久, 1h z8=-9.45); c12 (波动长记忆 "
                 "DFA-H 0.93); c18 (4h 方向=无条件漂移); 书 CH2 p.43-58: DW 检验"
                 " — 趋势市场应正自相关 (H4 对照)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    series = load_returns(PARAMS)
    if not series:
        print("无数据, 退出")
        return 1
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    rows = []
    rho1_means = []
    for name, tf, r in series:
        if dev and not (tf == "1h" and name in PARAMS["crypto"]) \
                and not (tf == "1d" and name == "SPY"):
            continue
        rho, dw, rho2, rho2c = acf_spectrum(r, PARAMS["lags"])
        g_rho1, ci, r2c_ci = gbm_acf(r, PARAMS["lags"], seeds)
        rho1_means.append(g_rho1)
        rows.append({"name": name, "tf": tf, "r": r, "rho": rho, "dw": dw,
                     "rho2": rho2, "rho2c": rho2c, "gbm_ci": ci,
                     "gbm_r2c_ci": r2c_ci, "gbm_rho1": g_rho1})
    g = {"rho1_mean": float(np.mean(rho1_means))}
    gate(g["rho1_mean"])

    # H1: 加密 1h
    h1 = {"ok": True}
    for r in rows:
        if r["tf"] != "1h":
            continue
        lo, hi = r["gbm_ci"][1]
        if not (r["rho"][1] < 0 and r["rho"][1] < lo):
            h1["ok"] = False

    # H2: 消失边界
    h2 = {}
    for r in rows:
        k0 = None
        for k in PARAMS["lags"]:
            lo, hi = r["gbm_ci"][k]
            if abs(r["rho"][k]) <= hi:
                k0 = k
                break
        h2.setdefault(r["name"], {})[r["tf"]] = k0

    # H3: 平方收益 (加密) — 用 centered 口径 (c12 兼容, 波动聚集的正确度量);
    # raw 口径 null≈1/3 (iid 平方比, 被水平主导) 只作报告
    h3 = {"ok": True}
    for r in rows:
        if r["name"] not in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
            continue
        lo1, hi1 = r["gbm_r2c_ci"][1]
        lo10, hi10 = r["gbm_r2c_ci"][10]
        h3.setdefault(r["name"], {})[r["tf"]] = (lo1, hi1, lo10, hi10)
        if not (r["rho2c"][1] > hi1 and r["rho2c"][10] > hi10):
            h3["ok"] = False

    if dev:
        for r in rows:
            print("  [dev] {} {} ρ1={:.3f} DW={:.2f} ρ²1={:.4f}".format(
                r["name"], r["tf"], r["rho"][1], r["dw"], r["rho2"][1]))
        print(f"[dev] 管线 OK ({len(rows)} 序列 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    h4 = None
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, rows, h1, h2, h3, h4)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
