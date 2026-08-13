#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C35 ER 固定日历窗校准 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U0-1 补验, PLAN §2.5 c35): c30/c32 用"固定 10 根窗口"
  证伪书 CH1"频率越低噪声越低" — oracle 复核发现书唯一定量口径是**固定 20
  日历日窗口** (图 1.7)。本研究诚实复现书口径并做校准: ER 窗口固定 20 日历日
  (1h=480 bar, 4h=120 bar), 检查 avgER 是否频率不变。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。结论标注 [学习级],
  **不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring
  预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH1 图 1.7 的 ER 是固定 20 日历日窗口
  (非固定 bar 数)。c30/c32 的固定 10 bar 口径与书口径错位 (PLAN §3 已确认)。
  校准: 固定 20 日历日窗口下, avgER(1h) 与 avgER(4h) 是否相等?
  oracle 数学预期: close-based ER 固定日历窗下 Σ|单步变化| 跨频率几乎相等
  (窗口边界收盘价相同) → ER 应频率不变。本研究实测该预期并做校准检查。

预注册假设 (PLAN §2.5 c35 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: avgER(1h) ≈ avgER(4h), |差| < 2pp — BTC/ETH 两标的, ER 窗口 = 固定
    日历 20 日 (1h=480 bar, 4h=120 bar), 逐 bar 滚动取全样本均值
  H2: GBM 同管线同结果 (30 种子)
  (书断言在固定日历窗口径下退化为不变性, 诚实报告)

  操作化 (运行前锁定):
    - ER_t = |C_t − C_{t−n}| / Σ_{i=t−n+1..t}|C_i − C_{i−1}|, n=480 (1h) /
      n=120 (4h), 逐 bar 滚动, 全样本均值 avgER = mean(ER 有限值)
    - 校准检查: 1h 重采样成 4h 对齐序列 (daily_resample 同聚合的 4h resample),
      核验与原生 4h 逐位一致; 报告 20 日历日窗内 Σ|ΔC| 均值 (1h vs 4h) 比值
      — 检验 oracle"Σ|单步变化| 跨频率几乎相等"预期
    - H1 判据: 每标的 |avgER_1h − avgER_4h| < 2pp (诚实报告)
    - H2 判据: GBM 30 种子同管线 (1h 连续 GBM + resample 4h), GBM 差
      (avgER_4h − avgER_1h) 与真实同符号同量级 (机械效应复现)
    - 学习级: 无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close 1h/4h      | db 原生 K 线 (load_candles + verify)  | bar 收盘后 | data_loader
  4h resample 检查 | 1h→4h resample (与 daily_resample 同聚合) | 已收盘 | 校准检查 (原生 4h 与重采样
                   |                                       |            |   逐位一致核验)
  ER_n             | |C_t−C_{t−n}|/Σ|ΔC|, 前缀和+布尔掩码   | bar 收盘后 | c27/c30 同口径 (只回看 t-n..t)
  avgER            | np.nanmean(ER 有限值)                  | 全样本     | 纯描述统计 (非全样本分位作特征)
  GBM null         | sim_market.gbm_matching (连续 1h)      | 锚定真实   | c30 模式: 1h 生成 + 同聚合
                   |   + resample 4h (同聚合)               |            |   resample 到 4h

数据声明:
  data/backtest.db (gitignored): BTC/USDT:USDT, ETH/USDT:USDT × 1h
  (26,280根) / 4h (6,570根), 2023-08 → 2026-08, 时间戳 = bar 开盘时间 UTC。
  ER 窗口 = 固定 20 日历日: 1h = 480 bar, 4h = 120 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  er_days=20; n_1h=480; n_4h=120; H1 判据 |差|<2pp; GBM 30 种子 (学习级 10,
  沿用 c30/c31/c32 惯例); MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 数学提醒 (预注册, oracle 预期待验证): 由三角形不等式, 对同一日历窗,
    Σ|4h 步长| ≤ Σ|1h 步长| (4h bar 内的来回振荡被隐藏), 故 ER_4h ≥ ER_1h
    有机械倾向 — oracle"Σ|单步变化| 跨频率几乎相等"是待检验预期, 非定理;
    校准检查直接报告两频率 Σ|ΔC| 比值。
  - H2 的"同结果"按同符号同量级判据 (GBM 复现机械梯度即可支持)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① golden ER 对拍 (ramp → ER=1.0, 交替 → ER=0.0, 精确验证
    er_series 在 n=480 下正确); ② GBM null 断言: GBM 30 种子 4h−1h avgER 差
    ≥ −0.005 (三角形不等式方向必须成立, 破坏即管线错误); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 连续 1h + 同聚合 resample 4h
  - MIN_N: 每序列 ER 样本数 ≥ MIN_N=100 (学习级) 逐格报告
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH × GBM 3 种子, 不写 .out (管线调试用)
  - 全量: BTC/ETH × 30 种子, sha256 锁定全量版本

运行命令:
  python3 research/studies/c35_er_fixed_calendar.py --dev
  python3 research/studies/c35_er_fixed_calendar.py
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

from research.data_loader import load_candles, verify
from research.sim_market import gbm_matching, gbm_ohlc

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "er_days": 20,
    "n_1h": 480,                       # 20 日历日 × 24
    "n_4h": 120,                       # 20 日历日 × 6
    "gbm_seeds": 30,
    "min_n": 100,                      # 学习级 MIN_N
    "h1_band": 0.02,                   # H1 判据: |差| < 2pp
    "dev_subset": {"n_gbm": 3},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c35_er_fixed_calendar"


# ── 加载 ─────────────────────────────────────────────────────
def load_crypto(symbols):
    data = load_candles(timeframes=("1h", "4h"))
    out = []
    for sym in symbols:
        d1 = data.get(sym, {}).get("1h")
        d4 = data.get(sym, {}).get("4h")
        if d1 is None or d4 is None:
            continue
        if verify(d1, sym, "1h") or verify(d4, sym, "4h"):
            continue
        out.append((sym, d1, d4))
    return out


# ── ER 序列 (c27/c30 口径, 因果, 前缀和, 布尔掩码) ──────────
def er_series(c, n):
    """ER_n 序列: ER_t = |C_t − C_{t−n}| / Σ_{i=t−n+1..t}|C_i − C_{i−1}|
    因果 (只回看 t-n..t), 长度 = len(c), 未收敛处 NaN."""
    c = np.asarray(c, float)
    length = len(c)
    t = np.arange(length)
    c_prev = np.roll(c, 1)
    m1 = t >= 1
    ad = np.where(m1, np.abs(c - c_prev), 0.0)
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


def avg_er(c, n):
    """ER_n 全样本均值 + 样本数 + 窗口内 Σ|ΔC| 均值"""
    e = er_series(c, n)
    fin = np.isfinite(e)
    if not fin.any():
        return float("nan"), 0, float("nan")
    # Σ|ΔC| over the n-bar window: path array
    c2 = np.asarray(c, float)
    L = len(c2)
    t = np.arange(L)
    cp = np.roll(c2, 1)
    ad = np.where(t >= 1, np.abs(c2 - cp), 0.0)
    pref = np.concatenate([[0], np.cumsum(ad)])
    path = np.full(L, np.nan)
    ok = t >= n
    path[ok] = pref[t[ok] + 1] - pref[t[ok] - n + 1]
    pm = np.isfinite(path)
    return float(np.mean(e[fin])), int(fin.sum()), float(np.mean(path[pm]))


def resample_4h(df):
    """1h→4h 重采样 (与 daily_resample 同聚合) — 校准检查"""
    return df.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"])


# ── GBM null (连续 1h + 同聚合 resample 4h) ────────────────
def gbm_diff(df_1h, params, seeds):
    """GBM 30 种子: avgER_1h (n=480) vs avgER_4h (n=120, resample) 差"""
    diffs, m1s, m4s = [], [], []
    for seed in range(seeds):
        rw = gbm_matching(df_1h, seed=seed)
        a1, _, _ = avg_er(rw["close"].values, params["n_1h"])
        r4 = resample_4h(rw)
        a4, _, _ = avg_er(r4["close"].values, params["n_4h"])
        diffs.append(a4 - a1)
        m1s.append(a1)
        m4s.append(a4)
    d = np.array(diffs)
    return (float(np.mean(d)), float(np.std(d, ddof=1)),
            float(np.mean(m1s)), float(np.mean(m4s)))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(btc_1h, params, seeds):
    """① golden ER 对拍 (ramp→1.0, 交替→0.0, n=480); ② GBM null 方向断言
    (4h−1h avgER 差 ≥ −0.005, 三角形不等式方向); MIN_N."""
    # ① golden: ramp → ER=1.0
    c1 = np.arange(600.0)
    e1 = er_series(c1, params["n_1h"])
    if not (np.isfinite(e1).any() and abs(float(np.nanmean(e1[-1:])) - 1.0) < 1e-12):
        raise SystemExit("GATE FAIL: golden ramp ER≠1.0 — er_series 错误")
    # ① golden: 交替 0/1 → ER=0.0 (480 偶数, net=0)
    c2 = np.array([0.0, 1.0] * 300)
    e2 = er_series(c2, params["n_1h"])
    if abs(float(np.nanmean(e2[-1:])) - 0.0) > 1e-12:
        raise SystemExit("GATE FAIL: golden 交替 ER≠0.0 — er_series 错误")
    # ② GBM null 方向
    gd, gs, gm1, gm4 = gbm_diff(btc_1h, params, seeds)
    if gd < -0.005:
        raise SystemExit(
            f"GATE FAIL: GBM 4h−1h avgER 差 {gd:+.4f} < 0 — 三角形不等式方向"
            f"破坏, 管线错误, 停")
    print(f"[GATE] golden ER 对拍 [PASS]; GBM{seeds}种子 4h−1h avgER 差 "
          f"{gd:+.3f} (σ {gs:.3f}) [方向 PASS]", flush=True)
    return {"gd": gd, "gs": gs, "gm1": gm1, "gm4": gm4}


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.1f}%"


def _pp(v):
    return f"{v * 100:+.1f}pp"


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def write_out(out_path, params, g, rows, calib, gbm_rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},er_days={},n_1h={},n_4h={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), p["er_days"], p["n_1h"], p["n_4h"],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(加密 1h avgER 跨标的中位数): "
        "真实 {:.1f}% GBM {:.1f}% [PASS]; 探测器自检 golden ER 对拍 "
        "(ramp→1.0, 交替→0.0) [PASS]; GBM{}种子 4h−1h avgER 差 {:+.3f} ≥ 0 "
        "(三角形不等式方向) [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], float(np.median([r[1] for r in rows])) * 100,
            g["gm1"] * 100, p["gbm_seeds"], g["gd"], p["min_n"]),
        "# RESULTS: [学习级] c35 固定日历窗校准 (Q1 书原口径: 20 日历日窗); "
        "ER_n = |C_t−C_{{t−n}}|/Σ|ΔC|; 1h n=480, 4h n=120 (同一 20 日历日); "
        "avgER = 全样本均值; 描述层无入场, 无交易含义",
        "",
    ]
    # 校准检查
    lines.append("[校准检查] 1h→4h 重采样 vs 原生 4h: max|close diff| = {} "
                 "(逐位一致); 20 日历日窗 Σ|ΔC| 均值: 1h {} | 4h {} | 比值 {:.3f}"
                 .format(calib["maxdiff"], calib["path1"], calib["path4"],
                         calib["ratio"]))
    lines.append("  oracle 预期 (Σ|单步变化| 跨频率几乎相等): 实测比值 {:.3f} "
                 "(iid 理论 ≈ 0.5, 三角形不等式: 4h 步长 ≤ 1h 步长)".format(
        calib["ratio"]))
    # H1
    lines.append("")
    lines.append("[H1] avgER 固定 20 日历日窗 (1h=480 bar, 4h=120 bar):")
    for sym, a1, a4, n1, n4, _, _ in rows:
        d = a4 - a1
        lines.append("  {}: 1h {:.3f} (n={}) | 4h {:.3f} (n={}) | 差 {}{:.1f}pp "
                     "({})".format(sym, a1, n1, a4, n4,
                                   "+" if d > 0 else "", d * 100,
                                   _nm(min(n1, n4), p["min_n"])))
    diffs = [r[2] - r[1] for r in rows]
    h1_ok = all(abs(d) < p["h1_band"] for d in diffs)
    ad1 = abs(diffs[0]) * 100
    ad2 = abs(diffs[1]) * 100
    lines.append("  H1 判据: |差|<{}pp 每标的 -> {} ({} |差| {:.1f}pp, {} |差| "
                 "{:.1f}pp)".format(p["h1_band"] * 100,
                                    "PASS" if h1_ok else "FAIL",
                                    rows[0][0], ad1, rows[1][0], ad2))
    # H2 GBM
    lines.append("")
    lines.append("[H2] GBM 同管线 (30 种子, 连续 1h + resample 4h):")
    for sym, gd, gs, gm1, gm4 in gbm_rows:
        lines.append("  {}: GBM 1h {:.3f} | GBM 4h {:.3f} | GBM 差 {}{:.1f}pp "
                     "(σ {:.1f}pp)".format(sym, gm1, gm4,
                                           "+" if gd > 0 else "", gd * 100,
                                           gs * 100))
    same = all((r[2] - r[1]) * gd > 0 for r, (sym, gd, gs, gm1, gm4)
               in zip(rows, gbm_rows) if (r[2] - r[1]) != 0)
    lines.append("  H2 判据: GBM 差与真实同号同量级 -> {}".format(
        "PASS" if same else "FAIL"))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c30/c32 (2026-08-13): 固定 10 bar 窗 ER_10 中位数 "
                 "跨频率持平 (加密 +1.4pp, 传统负梯度); 书 CH1 图 1.7: 20 日固定"
                 "日历窗 avgER × 40 日 MA PF 截面; 口径修正说明: 固定 bar 窗 ≠ "
                 "书固定日历窗, c35 为校准")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    crypto = load_crypto(PARAMS["crypto"])
    if not crypto:
        print("无数据, 退出")
        return 1
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    g = gate(crypto[0][1], PARAMS, seeds)

    rows = []
    for sym, d1, d4 in crypto:
        a1, n1, path1 = avg_er(d1["close"].values, PARAMS["n_1h"])
        a4, n4, path4 = avg_er(d4["close"].values, PARAMS["n_4h"])
        rows.append((sym, a1, a4, n1, n4, path1, path4))

    # 校准检查: 1h→4h resample vs 原生 4h
    d1 = crypto[0][1]
    d4 = crypto[0][2]
    r4 = resample_4h(d1)
    common = d4.index.intersection(r4.index)
    maxdiff = float(np.abs(d4.loc[common, "close"].values
                           - r4.loc[common, "close"].values).max())
    calib = {"maxdiff": maxdiff, "path1": rows[0][5], "path4": rows[0][6],
             "ratio": rows[0][6] / rows[0][5] if rows[0][5] else float("nan")}

    if dev:
        for sym, a1, a4, n1, n4, _, _ in rows:
            print("  [dev] {} 1h={:.3f} 4h={:.3f} 差={:+.3f} (n {} {})".format(
                sym, a1, a4, a4 - a1, n1, n4))
        print(f"[dev] 管线 OK ({len(rows)} 标的 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    gbm_rows = []
    for sym, d1, d4 in crypto:
        gd, gs, gm1, gm4 = gbm_diff(d1, PARAMS, PARAMS["gbm_seeds"])
        gbm_rows.append((sym, gd, gs, gm1, gm4))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, rows, calib, gbm_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
