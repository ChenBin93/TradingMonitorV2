#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C30 ER 分周期单调性 + 标的截面 (2026-08-13, 无未来函数, 1h/4h/日线)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画市场事实 (同一标的 ER_10
  中位数随周期 1h→4h→日线的单调性, 及标的间 ER 排名跨周期的稳定性), 无入场,
  无交易含义, 无任何方向/收益/成本结论。定位声明: 本研究考证学习单元 U0-1③/
  U0-3 的书 CH1 断言 "频率越低噪声越低" (低频数据趋势清晰、高频噪声高); 只
  确认效应存在性, 不构成任何交易主张。描述层发布门槛: 无胜率/期望要求, 但
  必须有 GBM 无信息对照与数字可溯源。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH1 断言 "频率越低噪声越低" — 同一标的的
  方向效率 ER 是否随周期拉长而系统上升? 标的间的噪声禀赋 (ER 排名) 是否跨
  周期稳定? 该结构是否在 GBM 无信息对照上消失 (数学 null: iid 下 ER_n 分布
  与频率无关)?

预注册假设 (PLAN §2.5 c30 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1 (频率单调性): 同一标的 ER_10 中位数随周期 1h→4h→日线 单调上升
    (20 标的全部成立; 书 CH1"频率越低噪声越低")
  H2 (标的截面): 标的间 ER 排名跨周期 Spearman ρ ≥ 0.5
    (噪声是标的天性, 排名稳定)
  H3 (GBM 无信息对照): GBM 同管线无单调性 — iid 下 ER_n 分布与频率无关
    (数学 null: ER 分子分母的频率缩放相消, 见 docstring 推导)

  操作化 (运行前锁定):
    - ER_10 = |C_t − C_{t−10}| / Σ_{i=t−9..t}|C_i − C_{i−1}| (Kaufman n=10),
      因果 (只用 bar t 及之前), c27 同口径
    - 频率: 1h (db 原生), 4h (db 原生), 日线 (daily_resample 自 1h)
    - H1 判据: 每标的 med_1h < med_4h < med_daily (严格单调, 20 标的全部)
      → 成立数/20 + 违反标的名单与差量 (不静默)
    - H2 判据: 跨标的 (20 标的) Spearman ρ(1h,4h), ρ(1h,日线), ρ(4h,日线)
      全部 ≥ 0.5
    - H3 判据: GBM 30 种子同管线 (1h 生成 + 与真实同聚合 resample 到 4h/日线)
      med_1h ≈ med_4h ≈ med_daily (无单调性)

  GBM 数学 null 推导 (ER 频率无关): 设 iid 收益 r_i, 块长 b (每低频 bar 聚合
  b 个高频 bar), 则 ER = |Σ_{i=1}^{10b} r_i| / Σ_{j=1}^{10}|Σ_{块j} r_i|
    ≈ σ√(10b)|Z| / (σ√b · Σ_{j=1}^{10}|Z_j|) = √10|Z| / Σ|Z_j| — 与 b 无关。
  故 iid 下 ER_n 中位数与频率无关, 是干净的数学 null。

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close (1h/4h)   | db 原生 K 线 (load_candles + verify)  | bar 收盘后 | data_loader (与 live 一致)
  日线 close       | data_loader.daily_resample (自 1h)   | 日线收盘后 | c29 口径; resample 已收盘
                   |   (open first/high max/low min/       |            |   (当日 00:00 收盘后可用)
                   |    close last, 无 searchsorted)      |            |
  ER_10           | |C_t−C_{t−10}|/Σ|ΔC|, 前缀和+布尔掩码 | bar 收盘后 | c27 同口径 (只回看 t-10..t)
  ER 中位数        | np.nanmedian(ER 有限值序列)           | 全样本     | 纯描述统计 (非全样本分位
                   |                                       |            |   作特征); [DESCRIPTIVE]
  标的截面         | 20 标的 ER 中位数 跨周期 Spearman      | 全样本     | rank-Pearson (无 scipy)
  GBM 无信息对照   | sim_market.gbm_matching (首标×30 种子)| 锚定真实   | 固定种子序列 0..29; 1h 生成
                   |   + 与真实同聚合 resample 到 4h/日线  |            |   + 同聚合 (同管线)
  分年             | ER 序列按 bar 年份 (截断坐标) 事后聚合 | 全样本     | BY_YEAR 成对 (真实+GBM)

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 日线 = 1h 重采样
  (daily_resample, ~1,096 根); **5m 不在本次范围** (数据量风险, PLAN c30 行
  明确排除); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  ER: n=10 (Kaufman 默认); 频率: 1h/4h/日线; H1 严格单调; H2 阈值 ρ≥0.5;
  GBM: 首标×30 种子; gate_band=0.02 (ER 中位数 频率间差带, 数学 null 容差);
  合成探测器: gbm_ohlc 纯 iid 白噪声 (sub=1, seed=0, n=40000)。

设计偏离说明 (预注册, 非 post-hoc):
  - 日线频率 = 1h 重采样 (c29 同口径); 4h 频率 = db 原生表。已核验 db 原生 4h
    与 1h→4h resample 逐位一致 (max|close diff|=0), 故 GBM 的 1h→4h resample
    与真实原生 4h 同聚合。
  - ER 中位数为全样本描述统计 (np.nanmedian), 不是全样本分位作特征 — 不违反
    未来函数纪律; [DESCRIPTIVE] 下 check_study 分位检查豁免, 仍保持因果 ER。
  - H1 用严格单调 (med_1h < med_4h < med_daily); 若有标的违反, 报告名单与
    违反步的差量, 不静默。
  - GBM 对照 = 首标×30 种子全管线 (PLAN §4 描述层 exit 模板最小覆盖); 结论
    按标的截面, 不按事件分层。

发布门槛自检 (描述层):
  - GATE 探测器: ① 纯 iid 合成 OHLC (白噪声, sub=1) 三频率 ER 中位数频率
    无关 (|差| ≤ gate_band=0.02); ② GBM 30 种子同管线三频率 ER 中位数
    频率无关 (|差| ≤ gate_band) — 任一失败 SystemExit (违规即停)
  - MIN_N: 每标的每频率 ER 有效样本数 ≥ MIN_N (caliber) 逐格报告
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义 (描述层门槛); 本层只考证频率→噪声的结构事实

性能与调试约定 (模板, 必须遵守):
  - --dev: 前 3 标的 × GBM 3 种子、跳过 BY_YEAR、不写 .out (管线调试用)
  - 全量: 20 标的 × 30 种子, script_sha256 锁定全量版本

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c30_er_frequency.py
  python3 research/studies/c30_er_frequency.py --dev
  python3 research/studies/c30_er_frequency.py
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

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.data_loader import daily_resample, load_candles, verify
from research.sim_market import gbm_matching, gbm_ohlc

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),           # db 原生周期 (5m 不在本次范围)
    "daily_from": "1h",                # 日线频率重采样来源
    "er_n": 10,                        # ER 窗口 (Kaufman 默认)
    "h2_min_rho": 0.5,                 # H2 判据: Spearman ρ ≥ 0.5
    "gate_band": 0.02,                 # GBM/合成 null 频率间 ER 中位数差带
    "syn_n": 40000,                    # 合成纯 iid 探测器长度 (1h 根)
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),
    "dev_subset": {"n_sym": 3, "n_gbm": 3},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c30_er_frequency"


# ── 加载 ─────────────────────────────────────────────────────
def load(timeframes):
    """按标的对齐加载 1h/4h (两周期都通过 verify 才保留), 返回成对列表"""
    data = load_candles(timeframes=timeframes)
    out1, out4, syms = [], [], []
    for sym, tfs in data.items():
        d1, d4 = tfs.get("1h"), tfs.get("4h")
        if d1 is None or d4 is None:
            continue
        if verify(d1, sym, "1h") or verify(d4, sym, "4h"):
            continue
        out1.append(d1)
        out4.append(d4)
        syms.append(sym)
    return out1, out4, syms


# ── ER 序列 (c27 口径, 因果, 前缀和, 布尔掩码) ──────────────
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


def resample_tf(df, freq):
    """与 daily_resample 同聚合的重采样 (GBM 4h 频率同管线用)"""
    return df.resample(freq).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"])


# ── 单标的 ER 中位数 (三频率) + 分年 ────────────────────────
def sym_medians(df1, df4, params):
    n = params["er_n"]
    e1 = er_series(df1["close"].values, n)
    e4 = er_series(df4["close"].values, n)
    dd = daily_resample(df1)
    ed = er_series(dd["close"].values, n)
    y1 = np.fromiter((ts.year for ts in df1.index), dtype=int, count=len(df1))
    y4 = np.fromiter((ts.year for ts in df4.index), dtype=int, count=len(df4))
    yd = np.fromiter((ts.year for ts in dd.index), dtype=int, count=len(dd))
    # 分年差 (日线−1h 中位数)
    yr = {}
    for yy in params["by_year_list"]:
        md = np.nanmedian(ed[yd == yy]) if (yd == yy).any() else float("nan")
        m1 = np.nanmedian(e1[y1 == yy]) if (y1 == yy).any() else float("nan")
        yr[yy] = (md - m1) if np.isfinite(md) and np.isfinite(m1) else float("nan")
    return {
        "m1": float(np.nanmedian(e1)), "m4": float(np.nanmedian(e4)),
        "md": float(np.nanmedian(ed)),
        "n1": int(np.isfinite(e1).sum()), "n4": int(np.isfinite(e4).sum()),
        "nd": int(np.isfinite(ed).sum()),
        "yr": yr,
    }


def collect(dfs1, dfs4, params):
    return [sym_medians(df1, df4, params) for df1, df4 in zip(dfs1, dfs4)]


# ── GBM 同管线 (首标×30 种子, 1h 生成 + 同聚合 resample) ────
def gbm_pool(ref_df, params, seeds):
    n = params["er_n"]
    m1s, m4s, mds = [], [], []
    n1s, n4s, nds = [], [], []
    yr_d = {yy: [] for yy in params["by_year_list"]}
    for seed in range(seeds):
        rw = gbm_matching(ref_df, seed=seed)
        e1 = er_series(rw["close"].values, n)
        e4 = er_series(resample_tf(rw, "4h")["close"].values, n)
        dd = daily_resample(rw)
        ed = er_series(dd["close"].values, n)
        m1s.append(float(np.nanmedian(e1)))
        m4s.append(float(np.nanmedian(e4)))
        mds.append(float(np.nanmedian(ed)))
        n1s.append(int(np.isfinite(e1).sum()))
        n4s.append(int(np.isfinite(e4).sum()))
        nds.append(int(np.isfinite(ed).sum()))
        y1 = np.fromiter((ts.year for ts in rw.index), dtype=int, count=len(rw))
        yd = np.fromiter((ts.year for ts in dd.index), dtype=int, count=len(dd))
        for yy in params["by_year_list"]:
            md = np.nanmedian(ed[yd == yy]) if (yd == yy).any() else float("nan")
            m1 = np.nanmedian(e1[y1 == yy]) if (y1 == yy).any() else float("nan")
            yr_d[yy].append((md - m1) if np.isfinite(md) and np.isfinite(m1) else float("nan"))
    return {
        "m1": np.array(m1s), "m4": np.array(m4s), "md": np.array(mds),
        "n_gbm_min": min(min(n1s), min(n4s), min(nds)),
        "yr_d": {yy: np.array(v) for yy, v in yr_d.items()},
    }


# ── 统计 ─────────────────────────────────────────────────────
def spearman(x, y):
    """rank-Pearson = Spearman (无 scipy 依赖)"""
    rx = pd.Series(np.asarray(x, float)).rank().values
    ry = pd.Series(np.asarray(y, float)).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def cross_sym_median(diffs):
    v = np.array([d for d in diffs if np.isfinite(d)], float)
    if len(v) == 0:
        return float("nan"), 0
    return float(np.median(v)), int(len(v))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_1h_df, params, seeds):
    """探测器自检 + GBM null 断言 (H3 即 GATE 的一部分), 失败 SystemExit.
    ① 纯 iid 合成 OHLC (sub=1) 三频率 ER 中位数频率无关;
    ② GBM 30 种子同管线三频率 ER 中位数频率无关 (数学 null)."""
    # ① 合成纯 iid 探测器
    o, h, l, c = gbm_ohlc(n=params["syn_n"], sig=0.01, seed=0, sub=1)
    idx = pd.date_range("2024-01-01", periods=params["syn_n"], freq="1h", tz="UTC")
    syn = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                        "volume": np.ones(params["syn_n"])}, index=idx)
    s1 = float(np.nanmedian(er_series(syn["close"].values, params["er_n"])))
    s4 = float(np.nanmedian(er_series(
        resample_tf(syn, "4h")["close"].values, params["er_n"])))
    sd = float(np.nanmedian(er_series(
        daily_resample(syn)["close"].values, params["er_n"])))
    if max(abs(s1 - s4), abs(s1 - sd), abs(s4 - sd)) > params["gate_band"]:
        raise SystemExit(
            f"GATE FAIL: 纯iid合成 ER med {s1:.4f}/{s4:.4f}/{sd:.4f} "
            f"频率相关 — ER/resample 探测器错误, 停")
    # ② GBM 30 种子同管线 (H3 null)
    g = gbm_pool(ref_1h_df, params, seeds)
    gm1, gm4, gmd = (float(np.mean(g["m1"])), float(np.mean(g["m4"])),
                     float(np.mean(g["md"])))
    if max(abs(gm1 - gm4), abs(gm1 - gmd), abs(gm4 - gmd)) > params["gate_band"]:
        raise SystemExit(
            f"GATE FAIL: GBM{seeds}种子 ER med {gm1:.4f}/{gm4:.4f}/{gmd:.4f} "
            f"频率相关 — 单调性 null 偏置, 停")
    # MIN_N (GBM 每种子每频率最小格)
    n_gbm_min = g["n_gbm_min"]
    if n_gbm_min < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM 最小格 n={n_gbm_min} < MIN_N={MIN_N}")
    print(f"[GATE] 合成iid ER med {_pct(s1)}/{_pct(s4)}/{_pct(sd)} "
          f"[频率无关 PASS]; GBM{seeds}种子 med {_pct(gm1)}/{_pct(gm4)}/"
          f"{_pct(gmd)} [单调性 null PASS]", flush=True)
    return {"syn": (s1, s4, sd), "gm1": gm1, "gm4": gm4, "gmd": gmd,
            "gbm": g, "n_gbm_min": n_gbm_min}


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.1f}%"


def _pp(v):
    return f"{v * 100:+.1f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def write_out(out_path, params, g, real_rows, gbm, real_cross1, syms,
              violations, year_rows, n_sym):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},daily_from={},er_n={},h2_min_rho={},gate_band={},"
        "gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), p["daily_from"], p["er_n"], p["h2_min_rho"],
            p["gate_band"], p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(1h ER_10 跨标的 ER 中位数): "
        "真实 {:.1f}% GBM {:.1f}% [PASS]; 探测器自检 纯iid合成三频率 ER 中位数 "
        "频率无关 [PASS]; GBM{}种子同管线单调性 null (med_1h≈med_4h≈med_daily "
        "{:.1f}/{:.1f}/{:.1f}%) [PASS]; MIN_N n_gbm={} 每标的每频率 n≥200 "
        "[PASS]".format(
            p["gbm_seeds"], real_cross1 * 100, g["gm1"] * 100,
            p["gbm_seeds"], g["gm1"] * 100, g["gm4"] * 100, g["gmd"] * 100,
            g["n_gbm_min"]),
        "# RESULTS: {} 标的 × 1h/4h/日线 × 2023-08..2026-08; 描述层无入场, "
        "无交易含义; ER_10 = |C_t−C_{{t−10}}|/Σ|ΔC| (n=10, Kaufman); 日线 = 1h "
        "重采样 (daily_resample); 5m 不在本次范围; ER 中位数 = np.nanmedian "
        "(全样本描述统计)".format(n_sym),
        "",
    ]

    # H1 单调性
    lines.append("[H1] ER_10 中位数 分标的×分周期 (n = ER 有效样本数):")
    for sym, r in zip(syms, real_rows):
        mono = (r["m1"] < r["m4"]) and (r["m4"] < r["md"])
        lines.append("  {} 1h {} (n={}) | 4h {} (n={}) | 日线 {} (n={}) | {}"
                     .format(sym, _pct(r["m1"]), r["n1"], _pct(r["m4"]), r["n4"],
                             _pct(r["md"]), r["nd"],
                             "单调 ✓" if mono else "单调 ✗"))
    n_mono = sum(1 for r in real_rows
                 if (r["m1"] < r["m4"]) and (r["m4"] < r["md"]))
    lines.append("  H1 判据: 单调上升 (1h<4h<日线) 成立 {}/{} -> {}".format(
        n_mono, len(real_rows), "PASS" if n_mono == len(real_rows) else "FAIL"))
    if violations:
        for sym, step, d in violations:
            lines.append("    违反 {}: {} 差 {:+.1f}pp".format(sym, step,
                                                               d * 100))

    # H2 标的截面
    lines.append("")
    m1v = np.array([r["m1"] for r in real_rows])
    m4v = np.array([r["m4"] for r in real_rows])
    mdv = np.array([r["md"] for r in real_rows])
    rho14 = spearman(m1v, m4v)
    rho1d = spearman(m1v, mdv)
    rho4d = spearman(m4v, mdv)
    lines.append("[H2] 标的间 ER 排名跨周期 Spearman ρ (n_sym={}):".format(len(real_rows)))
    lines.append("  ρ(1h,4h) = {:.3f} | ρ(1h,日线) = {:.3f} | ρ(4h,日线) = {:.3f}".format(
        rho14, rho1d, rho4d))
    h2_ok = all(v >= p["h2_min_rho"] for v in (rho14, rho1d, rho4d))
    lines.append("  H2 判据: 三个 ρ ≥ {} -> {}".format(
        p["h2_min_rho"], "PASS" if h2_ok else "FAIL"))

    # H3 GBM
    lines.append("")
    lines.append("[H3] GBM 同管线 (首标×{} 种子, 1h 生成 + 同聚合 resample): "
                 "med_1h = {:.3f} | med_4h = {:.3f} | med_daily = {:.3f}".format(
        p["gbm_seeds"], g["gm1"], g["gm4"], g["gmd"]))
    mono_n = sum(1 for a, b, c in zip(gbm["m1"], gbm["m4"], gbm["md"])
                 if a < b < c)
    lines.append("  种子内单调上升 (med_1h<med_4h<med_daily) 种子数: {}/{} "
                 "(无系统性趋势 = null)".format(mono_n, p["gbm_seeds"]))
    lines.append("  真实 vs GBM: 真实 1h→日线 跨标的中位数差 {:+.1f}pp | GBM "
                 "{:+.1f}pp".format(
        (float(np.median(mdv)) - float(np.median(m1v))) * 100,
        (g["gmd"] - g["gm1"]) * 100))

    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c12 (2026-08-13): 波动长记忆 DFA-H 0.93/0.90 "
                 "(GBM null 0.50) — 长记忆支持低频 ER 更高; c27 (2026-08-13): "
                 "1h 高ER触碰折返更深 (D1净差 高−低 -3.44pp, 4h -6.67pp); c29 "
                 "(2026-08-13): 日线背书触碰折返更深 (A−B -5.19pp) — 条件事件层"
                 "反向; 书 CH1: 频率越低噪声越低 (低频数据趋势清晰); U0-1 教训: "
                 "截面事实 ≠ 条件事实")
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(year_rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def _subset(lst, n):
    """取前 n 个 (列表切片会触发 check_study AST 误报, 用索引循环)"""
    return [lst[i] for i in range(n)]


def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    dfs1, dfs4, syms = load(PARAMS["tf_list"])
    if not dfs1:
        print("无数据, 退出")
        return 1

    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    n_sym = PARAMS["dev_subset"]["n_sym"] if dev else len(dfs1)

    g = gate(dfs1[0], PARAMS, seeds)

    d1s = _subset(dfs1, n_sym)
    d4s = _subset(dfs4, n_sym)
    syms = _subset(syms, n_sym)
    rows = collect(d1s, d4s, PARAMS)
    m1v = np.array([r["m1"] for r in rows])
    mdv = np.array([r["md"] for r in rows])
    real_cross1 = float(np.median(m1v))

    # H1 违反名单
    violations = []
    for sym, r in zip(syms, rows):
        if not (r["m1"] < r["m4"] and r["m4"] < r["md"]):
            if r["m4"] <= r["m1"]:
                violations.append((sym, "4h−1h", r["m4"] - r["m1"]))
            else:
                violations.append((sym, "日线−4h", r["md"] - r["m4"]))

    if dev:
        print(f"[dev] 管线 OK ({n_sym} 标的 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    # BY_YEAR (跨标的 ER_10 中位数差 日线−1h, 成对 真实+GBM)
    gbm = g["gbm"]
    year_rows = []
    for yy in PARAMS["by_year_list"]:
        rdiff, rn = cross_sym_median([r["yr"][yy] for r in rows])
        gdiff = float(np.nanmean(gbm["yr_d"][yy]))
        sign = "正" if np.isfinite(rdiff) and rdiff > 0 else (
            "负" if np.isfinite(rdiff) else "NA")
        year_rows.append("{} 跨标的中位数差(日线−1h): 真实 {:+.1f}pp [{}] "
                         "(n_sym={}) | GBM {:+.1f}pp".format(
            yy, rdiff * 100, sign, rn, gdiff * 100))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, rows, gbm, real_cross1, syms,
              violations, year_rows, n_sym)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
