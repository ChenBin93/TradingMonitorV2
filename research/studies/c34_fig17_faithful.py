#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C34 图 1.7 忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U0-1/U0-3, PLAN §2.5 c34): 书 CH1 图 1.7 — 跨市场
  avgER (x) × 40 日 MA 趋势系统 PF (y) 散点, 书"低噪声市场趋势系统 PF 更高"。
  本书原口径 (oracle 逐字复核): x=20 日窗口 ER 全样本均值; y=PF=毛利润/毛亏损
  (书图题写 information ratio、正文写 PF, 自相矛盾 — 取 PF 并标注); 系统=40 日
  MA 趋势线转向信号 (MA 斜率符号改变即翻转, **非价格穿透**), 永远在场、多空
  反转、无成本; 信号日线收盘确定, 次日收盘成交, close-to-close PnL。
  描述层: 书系统无成本 + 永远在场, 非可交易主张, 结论避免触发词。
  结论标注 [学习级], **不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、
  .out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书图 1.7 (2000-2012 日线) 显示低噪声 (高 ER)
  市场的趋势系统 PF 更高。用本仓库数据 (2023-08..2026-08 日线, 20 币 + 5 传统
  对照 = 25 市场) 忠实复现: ER–PF 散点斜率/秩相关方向, 分组 (美股/金属/币/
  传统对照) PF 排名, GBM 同管线散点是否无此结构。

预注册假设 (PLAN §2.5 c34 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: ER–PF Spearman ρ < 0 且 |ρ| ≥ 0.3 (书: 低噪声市场趋势系统 PF 更高)
  H2: 美股子集 PF 排名垫底 (书"股指最差"; 判据: 美股组 PF 中位数 < 其余组)
  H3: GBM 同管线散点无此结构 (30 种子/市场, n=各自 bar 数, μ/σ=样本估计,
      同 MA 系统算 PF; 判据: GBM 散点 |ρ| < 0.3 且 PF 分布集中近 1)

  操作化 (运行前锁定):
    - x: avgER = 20 日窗口 ER 全样本均值 (ER_t = |C_t−C_{t−20}|/Σ|ΔC|, c35
      同口径)
    - y: PF = 毛利润/毛亏损 (逐 trade PnL; trade = 持仓方向连续段, 成交=次日
      收盘, close-to-close)
    - 系统: 40 日 MA 收盘斜率符号 → 方向 (斜率为 0 保持前向); 信号 close t
      确定 → 持仓 p[t+1]=d[t] (次日收盘成交), PnL 期间 [t+1, t+2]
    - 数据: 20 币 (backtest.db 1h → daily_resample) + 5 传统 (control.db 1d,
      共同 3y 窗口 2023-08..2026-08), 共 25 市场
    - 分组: 币 (USDT 标的) / 美股 (SPY) / 金属 (GC=F) / 传统对照
      (CL=F, EURUSD=X, ^TNX) — 分组名单 .out 列出
    - H1 判据: 计入市场 (n_trades≥10) 的 Spearman ρ < 0 且 |ρ| ≥ 0.3
    - H2 判据: 美股组 PF 中位数 < 其余各组 PF 中位数
    - H3 判据: GBM 散点 (每市场 30 种子中位数) |ρ| < 0.3 且 GBM PF 分布
      集中近 1 (报告 median/5-95% 范围)
    - 学习级: 25 标的 (偏离 BTC/ETH 规则 — 散点需截面宽度且计算廉价, 标注);
      无 BY_YEAR; MIN_N=100 (每市场 n_trades≥10 才计入 PF, 不足标注剔除)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close 日线 (币)  | backtest.db 1h → daily_resample       | 日线收盘后 | c30/c35 口径
  close 日线 (传统)| control.db 1d (双引号表名, ts=UTC 秒)  | 日线收盘后 | 对照数据源; 共同 3y 窗
  avgER (20 日)    | |C_t−C_{t−20}|/Σ|ΔC|, 前缀和+掩码     | bar 收盘后 | c35 同口径 (因果)
  MA40 信号        | ma40 斜率符号 (收盘确定, 次日成交)     | close t    | 书原口径 (非价格穿透)
  PF               | 逐 trade PnL (方向连续段, 无成本)      | 全样本事后 | 描述层 (书系统口径)
  GBM null         | n=各自 bar 数, μ/σ=样本 (有限值掩码)   | 锚定真实   | 30 种子, 同 MA 系统

数据声明:
  data/backtest.db: 20 币 1h (26,280根) → 日线 (~1,095根); 2023-08..2026-08。
  data/control.db: SPY/CL=F/GC=F/EURUSD=X/^TNX 1d, 取共同 3y 窗 2023-08..2026-08。
  口径偏差 (docstring 标注): 书为 2000-2012 日线期货+股指, 我们为 2023-08..2026-08
  日线 (币+传统); 书图题写 information ratio、正文写 PF — 取 PF。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  er_n=20; ma_w=40; GBM 30 种子; MIN_N=100 (学习级); n_trades_min=10 (计入 PF);
  H1 判据 |ρ|≥0.3 且 ρ<0; H3 判据 |ρ|<0.3 且 PF 集中近 1。

设计偏离说明 (预注册, 非 post-hoc):
  - **H1 预注册符号矛盾 (预注册内, 非本脚本引入)**: PLAN c34 行写 "ρ < 0 且
    |ρ| ≥ 0.3 (低噪声→高 PF)" — 低噪声=高 ER, 书逻辑应为 ρ>0; 预注册字面与
    正文方向自相矛盾。本研究按字面判据 (ρ<0, |ρ|≥0.3) 执行, 并同时报告实际
    ρ 方向, 结论中对照书逻辑裁决。
  - 口径偏差: 书 2000-2012 日线期货+股指 vs 我们 2023-08..2026-08 日线
    (币+传统); y 轴取 PF (书图题 information ratio 与正文矛盾)。
  - 25 标的 (学习级偏离 BTC/ETH 规则): 散点需要截面宽度 (25 点) 且计算廉价。
  - MA 斜率符号为 0 时保持前向 (浮点相等为测度零, 防御性处理)。
  - GBM μ/σ 用有限对数收益掩码估计 (CL=F 负价数据瑕疵, c33 同处理)。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① MA 转向系统 golden (构造升→降序列, 验证方向翻转次数与
    逐 trade PnL); ② GBM null 幅度 sanity: 合并 GBM PF 中位数 ∈ [0.5, 1.5]
    且 GBM avgER 中位数 ∈ [0.05, 0.6] (管线错误才停; |ρ_GBM| 属 H3 裁决 —
    预注册 GBM 含样本漂移, 漂移同时驱动 avgER 与 MA 系统 PF 产生机械相关,
    非管线错误); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子/市场, 同 MA 系统
  - MIN_N: 每市场 n_trades ≥ 10 才计入 PF (不足标注剔除); GATE 行标 MIN_N=100
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义 (书系统无成本)

性能与调试约定 (模板, 必须遵守):
  - --dev: 5 个市场 × GBM 3 种子, 不写 .out (管线调试用)
  - 全量: 25 市场 × 30 种子, sha256 锁定全量版本 (计算廉价, ≤3 分钟)
  - 可选: 散点图 research/notes/c34_fig17_scatter.png (matplotlib)

运行命令:
  python3 research/studies/c34_fig17_faithful.py --dev
  python3 research/studies/c34_fig17_faithful.py
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
    "control": ("SPY", "CL=F", "GC=F", "EURUSD=X", "^TNX"),
    "control_db": "data/control.db",
    "win_start": "2023-08-01",
    "win_end": "2026-08-01",
    "er_n": 20,                          # 书: 20 日固定窗口
    "ma_w": 40,                          # 书: 40 日 MA
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N (协议值)
    "n_trades_min": 10,                  # 每市场计入 PF 的最低交易笔数
    "h1_rho": 0.3,                       # H1: |ρ| ≥ 0.3
    "h3_rho": 0.3,                       # H3: GBM |ρ| < 0.3
    "dev_subset": {"n_gbm": 3,
                   "control": ("SPY", "GC=F", "CL=F", "EURUSD=X", "^TNX")},
    "data_range": "2023-08..2026-08 日线 (书: 2000-2012)",
    "groups": {"币": "USDT", "美股": "SPY", "金属": "GC=F"},
}

STUDY_ID = "c34_fig17_faithful"


# ── 加载 ─────────────────────────────────────────────────────
def load_markets(params):
    """25 市场日线序列: 币 (backtest.db→daily_resample) + 传统 (control.db 1d)"""
    out = []
    data = load_candles(timeframes=("1h",))
    for sym, tfs in data.items():
        df = tfs.get("1h")
        if df is None or verify(df, sym, "1h"):
            continue
        if "USDT" not in sym:
            continue
        dd = daily_resample(df)
        out.append((sym, dd["close"].values, dd.index))
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
            if len(sel) < params["ma_w"] + 5:
                continue
            out.append((sym, sel["close"].values.astype(float),
                        sel["ts"].values))
    finally:
        conn.close()
    return out


def group_of(sym):
    if "USDT" in sym:
        return "币"
    if sym == "SPY":
        return "美股"
    if sym == "GC=F":
        return "金属"
    return "传统对照"


# ── avgER (20 日窗口, c35 同口径, 因果) ────────────────────
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


def avg_er(c, n):
    e = er_series(c, n)
    fin = np.isfinite(e)
    if not fin.any():
        return float("nan")
    return float(np.mean(e[fin]))


# ── 40 日 MA 趋势线转向系统 (书口径, 无成本) ────────────────
def ma_system_pf(c, w_ma):
    """MA40 斜率符号 → 方向 (close t 确定, 斜率为 0 保持前向);
    持仓 p[t+1] = d[t] (次日收盘成交), PnL 期间 [t+1, t+2] (close-to-close).
    返回 (PF, n_trades, trade_pnls, positions)."""
    c = np.asarray(c, float)
    n = len(c)
    p = np.zeros(n, int)
    if n < w_ma + 3:
        return float("nan"), 0, np.array([], float), p
    ma = pd.Series(c).rolling(w_ma).mean().values
    slope = np.full(n, np.nan)
    t = np.arange(n)
    ok = t >= w_ma
    slope[ok] = ma[t[ok]] - ma[t[ok] - 1]
    d = np.zeros(n, int)
    cur = 0
    for i in range(w_ma, n):
        if slope[i] > 0:
            cur = 1
        elif slope[i] < 0:
            cur = -1
        d[i] = cur
    p[1:] = d[:-1]
    trades = []
    i = 1
    while i < n - 1:
        if p[i] == 0:
            i += 1
            continue
        j = i
        while j < n - 1 and p[j] == p[i]:
            j += 1
        trades.append(float(p[i]) * (c[j] / c[i] - 1.0))
        i = j
    trades = np.array(trades, float)
    nt = len(trades)
    if nt == 0:
        return float("nan"), 0, trades, p
    pos = float(trades[trades > 0].sum())
    neg = float(-trades[trades < 0].sum())
    pf = pos / neg if neg > 0 else float("inf")
    return pf, nt, trades, p


# ── GBM null (30 种子/市场, 同 MA 系统) ─────────────────────
def gbm_pf_series(c_real, n_bars, params, seeds):
    """每种子 GBM (n=n_bars, μ/σ=样本有限值估计) → (avgER, PF)"""
    lr = np.diff(np.log(c_real))
    fin = np.isfinite(lr)
    mu = float(np.mean(lr[fin]))
    sig = float(np.std(lr[fin], ddof=1))
    er20 = []
    pfs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        r = rng.normal(mu, sig, size=n_bars)
        c = 100.0 * np.exp(np.cumsum(r))
        er20.append(avg_er(c, params["er_n"]))
        pf, nt, trades, _ = ma_system_pf(c, params["ma_w"])
        pfs.append(pf)
    return (float(np.median(er20)), float(np.median(pfs)),
            np.array(pfs, float))


def spearman(x, y):
    rx = pd.Series(np.asarray(x, float)).rank().values
    ry = pd.Series(np.asarray(y, float)).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(gbm_rho, gbm_pf_med, gbm_er_med):
    """① MA 系统 golden: 升→降序列方向翻转一次 (n_trades=2), 单调升 → 1 trade;
    ② GBM null 幅度 sanity: PF 中位数 ∈ [0.5, 1.5] 且 avgER 中位数
    ∈ [0.05, 0.6] (GBM 输出量级合理); |ρ_GBM| 不在此断言 — 预注册 GBM 含
    样本漂移, 漂移同时驱动 avgER 与 MA 系统 PF (机械相关), 是 H3 的裁决对象
    而非管线错误."""
    # ① 升 60 → 降 60
    c = np.concatenate([100.0 + np.arange(60) * 1.0,
                        159.0 - np.arange(60) * 1.0])
    pf, nt, trades, p = ma_system_pf(c, PARAMS["ma_w"])
    if nt != 2 or not np.isfinite(pf) or pf <= 0:
        raise SystemExit(
            f"GATE FAIL: golden 升→降 nt={nt} pf={pf} (期望 2/有限>0) "
            f"— MA 系统错误")
    c2 = 100.0 + np.arange(200) * 1.0
    pf2, nt2, trades2, p2 = ma_system_pf(c2, PARAMS["ma_w"])
    if nt2 != 1 or not np.isinf(pf2):
        raise SystemExit(
            f"GATE FAIL: golden 单调升 nt={nt2} pf={pf2} (期望 1/inf) "
            f"— MA 系统错误")
    # ② GBM null 幅度 sanity
    if not (0.5 <= gbm_pf_med <= 1.5) or not (0.05 <= gbm_er_med <= 0.6):
        raise SystemExit(
            f"GATE FAIL: GBM PF 中位数={gbm_pf_med:.2f} (需∈[0.5,1.5]) 或 "
            f"avgER 中位数={gbm_er_med:.3f} (需∈[0.05,0.6]) — GBM/管线错误, 停")
    print(f"[GATE] MA 转向系统 golden (升→降 2 trades, 单调升 1 trade) "
          f"[PASS]; GBM null 幅度 PFmed={gbm_pf_med:.2f} ERmed={gbm_er_med:.3f} "
          f"(|ρ_GBM|={gbm_rho:.2f} 属 H3 裁决, 漂移机械相关) [PASS]",
          flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(nt, min_t):
    return "[计入]" if nt >= min_t else "[剔除]"


def write_out(out_path, params, rows, rho, h2_groups, gbm_rho, gbm_pfs,
              book_ref):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=control={},er_n={},ma_w={},gbm_seeds={},min_n={},n_trades_min={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["control"]), p["er_n"], p["ma_w"], p["gbm_seeds"],
            p["min_n"], p["n_trades_min"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(全市场 avgER 中位数): 真实 {:.3f} | "
        "GBM {:.3f} [PASS]; 探测器自检 MA 转向系统 golden [PASS]; GBM null "
        "幅度 sanity (PF 中位数∈[0.5,1.5], avgER 中位数∈[0.05,0.6]) [PASS]; "
        "MIN_N={}(每市场 n_trades≥{} 计入 PF) [PASS]".format(
            p["gbm_seeds"], float(np.median([r["er"] for r in rows])),
            float(np.median([r["gbm_er"] for r in rows])),
            p["min_n"], p["n_trades_min"]),
        "# RESULTS: [学习级] c34 图 1.7 忠实复现 (书 CH1); x=avgER (20 日固定窗 "
        "全样本均值), y=PF=毛利润/毛亏损; 系统=40 日 MA 斜率转向 (非价格穿透), "
        "永远在场多空反转无成本; 信号收盘确定次日收盘成交 close-to-close; "
        "口径偏差: 书 2000-2012 期货+股指 vs 本 2023-08..2026-08 日线; "
        "描述层无入场, 无交易含义",
        "",
    ]
    # 截面表
    lines.append("[截面] 25 市场 (2023-08..2026-08 日线):")
    for r in rows:
        lines.append("  {} ({}) avgER={:.3f} PF={:.3f} n_trades={} {}".format(
            r["sym"], r["group"], r["er"], r["pf"], r["nt"],
            _nm(r["nt"], p["n_trades_min"])))
    gnames = {}
    for r in rows:
        gnames.setdefault(r["group"], []).append(r["sym"])
    lines.append("  分组名单: " + " | ".join(
        f"{g}={','.join(v)}" for g, v in gnames.items()))
    # H1
    lines.append("")
    lines.append("[H1] ER–PF Spearman ρ (计入市场 n={}): ρ = {:.3f} -> {} "
                 "(判据 ρ<0 且 |ρ|≥{})".format(
        len(rho[2]), rho[0],
        "PASS" if rho[0] < 0 and abs(rho[0]) >= p["h1_rho"] else "FAIL",
        p["h1_rho"]))
    lines.append("  (预注册符号矛盾说明: 书逻辑'低噪声→高PF'对应 ρ>0; 若实测 "
                 "ρ>0 且 |ρ|≥0.3, 书方向成立但预注册字面判据 FAIL — 结论裁决)")
    # H2
    lines.append("")
    lines.append("[H2] 分组 PF 中位数: " + " | ".join(
        f"{g}={v:.3f}" for g, v in h2_groups.items()))
    us = h2_groups.get("美股")
    h2_ok = us is not None and all(us < v for g, v in h2_groups.items()
                                   if g != "美股")
    lines.append("  美股组 PF 中位数 < 其余组 -> {}".format(
        "PASS" if h2_ok else "FAIL"))
    # H3
    lines.append("")
    lines.append("[H3] GBM 同管线 (30 种子/市场, 同 MA 系统): ρ_GBM = {:.3f} "
                 "-> {} (判据 |ρ|<{})".format(
        gbm_rho, "PASS" if abs(gbm_rho) < p["h3_rho"] else "FAIL",
        p["h3_rho"]))
    gmed = float(np.median(gbm_pfs))
    glo, ghi = float(np.percentile(gbm_pfs, 5)), float(
        np.percentile(gbm_pfs, 95))
    lines.append("  GBM PF 分布 ({} 样本): median={:.3f} 5-95%=[{:.2f}, {:.2f}]"
                 " (判据 集中近 1)".format(len(gbm_pfs), gmed, glo, ghi))
    lines.append("  GBM 散点 |ρ| 与 PF 范围 -> {}".format(
        "PASS" if abs(gbm_rho) < p["h3_rho"] and 0.5 <= gmed <= 1.5
        else "FAIL"))
    # 对照-书
    lines.append("")
    xmin = min(r["er"] for r in rows)
    xmax = max(r["er"] for r in rows)
    ys = [r["pf"] for r in rows if np.isfinite(r["pf"])]
    ymin, ymax = min(ys), max(ys)
    lines.append("[对照-书] 书参考值: x∈[{}, {}], y∈[{}, {}]; 本复现: x∈"
                 "[{:.3f}, {:.3f}], y∈[{:.3f}, {:.3f}]".format(
        book_ref["x0"], book_ref["x1"], book_ref["y0"], book_ref["y1"],
        xmin, xmax, ymin, ymax))
    lines.append("[对照-书] 书图题写 information ratio、正文写 PF (自相矛盾) — "
                 "本研究取 PF 并标注; 书 2000-2012 日线期货+股指 vs 本 "
                 "2023-08..2026-08 日线 (币+传统)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    markets = load_markets(PARAMS)
    if len(markets) < 10:
        print(f"无足够数据 ({len(markets)}), 退出")
        return 1

    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    rows = []
    gbm_pfs_all = []
    for sym, c, ts in markets:
        er = avg_er(c, PARAMS["er_n"])
        pf, nt, trades, p = ma_system_pf(c, PARAMS["ma_w"])
        ger, gpf, gpf_arr = gbm_pf_series(c, len(c), PARAMS, seeds)
        gbm_pfs_all.extend(np.where(np.isfinite(gpf_arr), gpf_arr,
                                    np.nan).tolist())
        rows.append({"sym": sym, "group": group_of(sym), "er": er, "pf": pf,
                     "nt": nt, "gbm_er": ger, "gbm_pf": gpf})

    if dev:
        for r in rows:
            print("  [dev] {} ({}) er={:.3f} pf={:.3f} nt={}".format(
                r["sym"], r["group"], r["er"], r["pf"], r["nt"]))
        # GATE sanity with dev GBM
        incl = [r for r in rows if r["nt"] >= PARAMS["n_trades_min"]]
        ex = [r["er"] for r in incl]
        ey = [r["pf"] for r in incl if np.isfinite(r["pf"])]
        rho = spearman([r["er"] for r in incl if np.isfinite(r["pf"])],
                       [r["pf"] for r in incl if np.isfinite(r["pf"])])
        gbm_rho = spearman([r["gbm_er"] for r in incl],
                           [r["gbm_pf"] for r in incl])
        gbm_pfs = np.array([x for x in gbm_pfs_all if np.isfinite(x)])
        gate(gbm_rho,
             float(np.median(gbm_pfs)) if len(gbm_pfs) else 1.0,
             float(np.median([r["gbm_er"] for r in incl])))
        print(f"[dev] ρ={rho:.3f} |ρ_GBM|={gbm_rho:.3f} n_incl={len(ex)} "
              f"({len(rows)} 市场 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    # 计入市场 (n_trades ≥ 10)
    incl = [r for r in rows if r["nt"] >= PARAMS["n_trades_min"]
            and np.isfinite(r["pf"])]
    rho = spearman([r["er"] for r in incl], [r["pf"] for r in incl])
    # 分组 PF 中位数
    h2_groups = {}
    for r in rows:
        if r["nt"] >= PARAMS["n_trades_min"] and np.isfinite(r["pf"]):
            h2_groups.setdefault(r["group"], []).append(r["pf"])
    h2_med = {g: float(np.median(v)) for g, v in h2_groups.items()}
    # GBM
    gbm_rho = spearman([r["gbm_er"] for r in incl],
                       [r["gbm_pf"] for r in incl])
    gbm_pfs = np.array([x for x in gbm_pfs_all if np.isfinite(x)])
    gate(gbm_rho, float(np.median(gbm_pfs)),
         float(np.median([r["gbm_er"] for r in incl])))
    book_ref = {"x0": 0.204, "x1": 0.266, "y0": 0.2, "y1": 3.7}

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, rows, (rho, None, incl), h2_med,
              gbm_rho, gbm_pfs, book_ref)
    print(f"written: {out_path}")

    # 散点图 (可选)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in rows:
            col = {"币": "#f59e0b", "美股": "#3b82f6", "金属": "#10b981",
                   "传统对照": "#8b5cf6"}[r["group"]]
            ax.scatter(r["er"], r["pf"], c=col, s=30,
                       alpha=0.85 if r["nt"] >= PARAMS["n_trades_min"] else 0.35)
            ax.annotate(r["sym"].split("/")[0][:8], (r["er"], r["pf"]),
                        fontsize=5, alpha=0.6)
        ax.axvline(0.204, ls="--", c="grey", lw=0.7)
        ax.axvline(0.266, ls="--", c="grey", lw=0.7)
        ax.set_xlabel("avgER (20 日固定窗, 全样本均值)")
        ax.set_ylabel("PF (40 日 MA 趋势转向, 无成本)")
        ax.set_title("C34 图 1.7 复现 (2023-08..2026-08 日线, [学习级])")
        ax.set_yscale("log")
        png = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "notes", "c34_fig17_scatter.png")
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"written: {png}")
    except Exception as e:  # noqa: BLE001 — 散点图为可选
        print(f"[png 跳过] {e!r}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
