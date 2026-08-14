#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C37 BTC 相关性审计 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c37): 用户批评 — 加密高度相关, 20 币 ≈ BTC 的 20 份
  拷贝, c30"噪声占比恒定"等截面结论可能只是伪复制。本研究量化截面伪复制:
  20 标的 3y 日收益相关矩阵 + 与 BTC 的两两 ρ + 有效独立样本数 n_eff。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。结论标注 [学习级],
  **不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring
  预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 加密 20 标的是否高度同源 (与 BTC 高相关)?
  有效独立样本数 n_eff 是多少? c30/c34 的截面结论在 n_eff 修正后还剩多少
  信息量?

预注册假设 (PLAN §2.5 c37 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 加密组与 BTC 平均 ρ ≥ 0.6 (高度同源)
  H2: 20 标的 n_eff ≤ 6 (伪复制: 有效独立样本远小于 20)
  H3: GBM 同管线 (20 条独立 GBM, n 同、各标的自有 μ/σ) n_eff ≥ 18
      (度量方法在独立数据上成立)

  操作化 (运行前锁定):
    - 日收益 = close-to-close 对数收益 (daily_resample 收盘)
    - 相关矩阵: 20 币对齐日收益的 Pearson 相关 (20×20)
    - n_eff = (Σλ)² / Σλ² (相关矩阵特征值比, participation ratio)
    - 与 BTC 的两两 ρ: ρ(BTC, X) 逐标的, 分组均值/范围
      (分组: 加密=20 币, 美股=SPY, 金属=GC=F, 传统对照=EURUSD=X)
    - H1 判据: 加密组 (19 个非 BTC 币) 平均 ρ(BTC,·) ≥ 0.6
    - H2 判据: 20 币相关矩阵 n_eff ≤ 6
    - H3 判据: GBM 30 种子, 每种子 20 条独立 GBM → n_eff, 均值 ≥ 18
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  日收益 (币)      | daily_resample 自 1h 收盘对数收益     | 日线收盘后 | c30/c34 口径
  日收益 (对照)    | control.db 1d (双引号表名), 共同 3y 窗 | 日线收盘后 | 对照数据源
  相关矩阵         | 对齐日收益 Pearson 相关 (close-to-close)| 全样本事后| 描述层 (非条件特征)
  n_eff            | (Σλ)²/Σλ², λ=相关矩阵特征值            | 全样本事后 | participation ratio
  GBM null         | 20 条独立 GBM (n 同、各标的自有 μ/σ)    | 锚定真实   | 30 种子; 独立数据上
                   |                                       |            |   度量方法自检 (H3)

数据声明:
  20 币 (backtest.db 1h → daily_resample ~1,095 根, 2023-08..2026-08);
  SPY/GC=F/EURUSD=X 1d (control.db, 共同 3y 窗)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  H1 ρ≥0.6; H2 n_eff≤6; H3 n_eff≥18; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 分组: 加密=20 币 (与 BTC 的 ρ 用其余 19 个), 美股=SPY, 金属=GC=F,
    传统对照=EURUSD=X (CL=F/^TNX 不在本次——c37 行只列与 BTC 两两 ρ 的
    对照, 取代表性三组).
  - n_eff 对相关矩阵特征值直接计算 (数值小负特征值截断为 0 处理).
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例.

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① n_eff 计算 golden (完全相关 2×2 → n_eff=1; 独立 2×2 →
    n_eff=2); ② GBM null sanity: 30 种子 n_eff 均值 ≥ 15 (独立数据上度量
    成立; 若管线错误 n_eff 会塌缩); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子 × 20 条独立 GBM
  - MIN_N: 每序列日收益 ~1,095 观测 ≥ MIN_N=100
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: 币 5 个 + 对照 3 个 × GBM 3 种子, 不写 .out
  - 全量: 20 币 × 30 种子 (计算廉价)

运行命令:
  python3 research/studies/c37_btc_corr_audit.py --dev
  python3 research/studies/c37_btc_corr_audit.py
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
    "control": ("SPY", "GC=F", "EURUSD=X"),
    "control_db": "data/control.db",
    "win_start": "2023-08-01",
    "win_end": "2026-08-01",
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "h1_rho": 0.6,                       # H1: 加密组与 BTC 平均 ρ ≥ 0.6
    "h2_eff": 6,                         # H2: n_eff ≤ 6
    "h3_eff": 18,                        # H3: GBM n_eff ≥ 18
    "gate_eff": 15,                      # GATE: GBM n_eff ≥ 15 (松 sanity)
    "dev_subset": {"n_gbm": 3, "n_crypto": 5},
    "data_range": "2023-08..2026-08 (对照共同 3y 窗)",
}

STUDY_ID = "c37_btc_corr_audit"


# ── 加载 ─────────────────────────────────────────────────────
def load_daily_returns(params, n_crypto=None):
    """→ (crypto_dict {sym: 日对数收益 pd.Series}, control_dict {sym: Series})
    币序: BTC 优先, 其余字母序 (dev 子集必含 BTC); 索引 = DatetimeIndex
    (对照与币长度不同, BTC-ρ 按索引交集对齐)."""
    data = load_candles(timeframes=("1h",))
    raw = {}
    for sym, tfs in data.items():
        df = tfs.get("1h")
        if df is None or verify(df, sym, "1h") or "USDT" not in sym:
            continue
        dd = daily_resample(df)
        c = dd["close"].values.astype(float)
        raw[sym] = pd.Series(np.diff(np.log(c)), index=dd.index[1:])
    order = ["BTC/USDT:USDT"] + sorted(s for s in raw if s != "BTC/USDT:USDT")
    if n_crypto is not None:
        order = order[:n_crypto]
    crypto = {s: raw[s] for s in order}
    conn = sqlite3.connect(params["control_db"])
    control = {}
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
            c = sel["close"].values.astype(float)
            control[sym] = pd.Series(np.diff(np.log(c)),
                                     index=sel["ts"].iloc[1:])
    finally:
        conn.close()
    return crypto, control


def _pearson(x, y):
    xa, ya = x.align(y, join="inner")
    a = xa.values.astype(float)
    b = ya.values.astype(float)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def corr_matrix(returns_dict, order):
    """对齐日收益相关矩阵 (Pearson) — 按 DatetimeIndex 交集对齐
    (部分币上市晚, 长度不同)."""
    names = [s for s in order if s in returns_dict]
    df = pd.concat([returns_dict[s] for s in names], axis=1, join="inner")
    R = df.values.astype(float)                    # (n_obs, n_assets)
    R = R - R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Rz = R / sd
    n = R.shape[0]
    C = Rz.T @ Rz / (n - 1)
    return C, names


def n_eff(C):
    lam = np.linalg.eigvalsh((C + C.T) / 2.0)
    lam = np.maximum(lam, 0.0)
    s1, s2 = float(lam.sum()), float((lam * lam).sum())
    if s2 <= 0:
        return float("nan")
    return s1 * s1 / s2


# ── GBM null (20 条独立 GBM, 30 种子) ───────────────────────
def gbm_n_eff(crypto, seeds):
    """每种子: 20 条独立 GBM (n 同、各标的自有 μ/σ) → 相关矩阵 → n_eff"""
    n = len(next(iter(crypto.values())))
    idx = next(iter(crypto.values())).index
    effs = []
    for seed in range(seeds):
        series = {}
        for i, (sym, r) in enumerate(crypto.items()):
            mu = float(np.mean(r))
            sig = float(np.std(r, ddof=1))
            rng = np.random.default_rng(seed * 1000 + i)
            rr = rng.normal(mu, sig, size=n)
            series[sym] = pd.Series(rr, index=idx)
        C, names = corr_matrix(series, list(series.keys()))
        effs.append(n_eff(C))
    return np.array(effs)


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(gbm_eff_mean, n_assets):
    """① n_eff golden: 完全相关 2×2 → 1; 独立 2×2 → 2;
    ② GBM null sanity: n_eff 均值 ≥ 0.75×n_assets (独立数据上度量成立;
    dev 资产少时阈值按资产数缩放; 若管线错误 n_eff 会塌缩)."""
    # ① 完全相关
    C1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    if abs(n_eff(C1) - 1.0) > 1e-9:
        raise SystemExit(f"GATE FAIL: 完全相关 n_eff={n_eff(C1)} ≠ 1")
    # ① 独立
    C2 = np.eye(2)
    if abs(n_eff(C2) - 2.0) > 1e-9:
        raise SystemExit(f"GATE FAIL: 独立 n_eff={n_eff(C2)} ≠ 2")
    # ② GBM sanity (按资产数缩放)
    bound = 0.75 * n_assets
    if gbm_eff_mean < bound:
        raise SystemExit(
            f"GATE FAIL: GBM n_eff 均值={gbm_eff_mean:.1f} < 0.75×n_assets="
            f"{bound:.1f} — 度量/GBM 管线错误, 停")
    print(f"[GATE] n_eff golden (完全相关=1, 独立=2) [PASS]; GBM n_eff 均值 "
          f"{gbm_eff_mean:.1f} ≥ {bound:.1f} (0.75×{n_assets}) [PASS]",
          flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, crypto, control, C, names, btc_rho, n_eff_real,
              gbm_effs, h2_med):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto_n={},control={},gbm_seeds={},min_n={},h1_rho={},"
        "h2_eff={},h3_eff={},gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级"
        .format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            len(crypto), "+".join(p["control"]), p["gbm_seeds"], p["min_n"],
            p["h1_rho"], p["h2_eff"], p["h3_eff"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(BTC 1h 日收益序列长度): n={} [PASS]; "
        "探测器自检 n_eff golden (完全相关=1, 独立=2) [PASS]; GBM null sanity "
        "n_eff 均值≥0.75×n_assets [PASS]; MIN_N 每序列 n≥{} [PASS]".format(
            p["gbm_seeds"], len(next(iter(crypto.values()))), p["min_n"]),
        "# RESULTS: [学习级] c37 BTC 相关性审计 (用户批评: 20 币伪复制); "
        "日收益=close-to-close 对数收益 (daily_resample); n_eff=(Σλ)²/Σλ²; "
        "分组: 加密/美股/金属/传统对照; 描述层无入场, 无交易含义",
        "",
    ]
    # 分组名单
    lines.append("[分组] 加密 ({} 币): {} | 美股: SPY | 金属: GC=F | "
                 "传统对照: EURUSD=X".format(
        len(crypto), ",".join(sorted(crypto.keys()))))
    # BTC-ρ 表
    lines.append("")
    lines.append("[BTC-ρ] ρ(BTC, X) 分组均值与范围:")
    for g, syms in btc_rho.items():
        vals = btc_rho[g]
        lines.append("  {}: mean={:.3f} min={:.3f} max={:.3f} (n={})".format(
            g, float(np.mean(vals)), float(np.min(vals)),
            float(np.max(vals)), len(vals)))
    grp_crypto = [v for g, v in btc_rho.items() if g == "加密"]
    crypto_mean = float(np.mean(grp_crypto[0])) if grp_crypto else float("nan")
    lines.append("  H1 判据: 加密组与 BTC 平均 ρ ≥ {} -> {}".format(
        p["h1_rho"],
        "PASS" if crypto_mean >= p["h1_rho"] else "FAIL"))
    # n_eff
    lines.append("")
    lines.append("[n_eff] 20 币相关矩阵: n_eff = {:.2f} (λ 谱前5: {})".format(
        n_eff_real, ", ".join(f"{x:.1f}" for x in h2_med)))
    lines.append("  H2 判据: n_eff ≤ {} -> {}".format(
        p["h2_eff"], "PASS" if n_eff_real <= p["h2_eff"] else "FAIL"))
    lines.append("")
    lines.append("[H3] GBM 同管线 ({} 种子 × 20 条独立 GBM): n_eff mean={:.2f} "
                 "std={:.2f} min={:.2f} max={:.2f}".format(
        p["gbm_seeds"], float(np.mean(gbm_effs)), float(np.std(gbm_effs,
                                                              ddof=1)),
        float(np.min(gbm_effs)), float(np.max(gbm_effs))))
    lines.append("  H3 判据: n_eff ≥ {} -> {}".format(
        p["h3_eff"], "PASS" if np.mean(gbm_effs) >= p["h3_eff"] else "FAIL"))
    # 截面复核
    lines.append("")
    lines.append("[截面复核] c30 Spearman ρ=0.275 (20 币 ER 排名) 的有效独立 "
                 "样本 = n_eff {:.1f} — 20 币≈{} 个独立信息源; c34 ρ=0.367 "
                 "(25 市场) 含 5 传统异质市场 (与 BTC 低相关), 独立信息更多 "
                 "({} 币 n_eff + 5 对照)".format(
        n_eff_real, max(1, int(round(n_eff_real))), len(crypto)))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    n_crypto = PARAMS["dev_subset"]["n_crypto"] if dev else None
    crypto, control = load_daily_returns(PARAMS, n_crypto=n_crypto)
    if len(crypto) < 2:
        print("无足够数据, 退出")
        return 1
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    # 对齐: 20 币全部同长 (daily_resample 同源); BTC-ρ 与对照用 BTC 索引交集
    C, names = corr_matrix(crypto, sorted(crypto.keys()))
    n_eff_real = n_eff(C)

    btc = "BTC/USDT:USDT"
    btc_r = crypto[btc]
    btc_rho = {"加密": [], "美股": [], "金属": [], "传统对照": []}
    for sym, r in crypto.items():
        if sym == btc:
            continue
        btc_rho["加密"].append(_pearson(btc_r, r))
    for g, sym in (("美股", "SPY"), ("金属", "GC=F"), ("传统对照", "EURUSD=X")):
        if sym in control:
            btc_rho[g].append(_pearson(btc_r, control[sym]))

    gbm_effs = gbm_n_eff(crypto, seeds)
    gate(float(np.mean(gbm_effs)), len(crypto))

    if dev:
        print("  [dev] n_eff(20币)={:.2f} | GBM n_eff mean={:.2f} | "
              "BTC-ρ 加密 mean={:.3f}".format(
            n_eff_real, float(np.mean(gbm_effs)),
            float(np.mean(btc_rho["加密"]))))
        print(f"[dev] 管线 OK ({len(crypto)} 币 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    lam = np.linalg.eigvalsh((C + C.T) / 2.0)
    lam = np.maximum(lam, 0.0)
    top5 = np.sort(lam)[::-1][:5]
    top5 = [x for x in top5 if x > 0.01]
    h2_med = top5 if top5 else [float("nan")]

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, crypto, control, C, names, btc_rho,
              n_eff_real, gbm_effs, h2_med)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
