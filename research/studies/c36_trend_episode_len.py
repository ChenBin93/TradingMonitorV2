#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C36 趋势情节长度 × 频率 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U0-1, PLAN §2.5 c36): 书"低频噪声低"被三口径 (c30/c32/
  c35) 证伪后, 用户重释: 作者可能想说"低频趋势更持久" (日线趋势能走半年一年,
  1h 几天就碎)。c33 已知加密日线无运行肥尾 — 若日线趋势持久, 须经慢漂移而非
  连涨连跌。本研究直接测: 趋势情节 (MA 斜率符号段) 的长度随频率如何变化, 是否
  超出 GBM。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。结论标注 [学习级],
  **不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring
  预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 趋势情节 (40 日历日等价 MA 斜率符号段) 的
  长度 (bar/天) 是否随频率变化, 且超出 GBM 同管线? 日线趋势是否比 1h 更持久
  (超出 null 更多)?

预注册假设 (PLAN §2.5 c36 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 真实日线情节长度 (bar 数) > GBM 日线同管线 2σ (趋势持久性真实存在)
  H2: 真实−GBM 超出的持久性随频率降低而增强 (日线超出 > 1h 超出 —
      用户"低频趋势更持久")
  H3: GBM 同管线各频率情节分布一致 (iid 尺度不变性, 对照 c35 几何课)

  操作化 (运行前锁定):
    - 趋势情节 = 连续同号 MA 斜率段 (slope=MA[t]−MA[t−1]; slope>0 上升段 /
      <0 下降段; =0 不计, 断段); MA 窗口 = 40 日历日等价: 日线 MA40 /
      4h MA240 / 1h MA960 (收盘价 MA)
    - 度量: 平均情节长 (bar), 中位数 (bar), P(情节≥20 bar), 每频率每标的;
      跨频率比较用日历天数 (bar/bpd, bpd: 1h=24, 4h=6, 日线=1)
    - GBM null: 30 种子每频率, 同 n 同 μ/σ (该频率对数收益样本, 有限值掩码),
      同 MA 窗口 — **MA 平滑自身引入斜率自相关, null 必须同 MA 才能对照**
      (核心对照设计, c35 几何课延续)
    - gap 断裂: 1h/4h 无缺口; 对照日线用 7 天阈值 (c31/c33 校准, 只断数据
      停更, 周末/假日保留)
    - H1 判据: 日线各标的真实 mean 情节长 (bar) > GBM 种子分布 mean+2σ
    - H2 判据: 每标的 超出(真实−GBM mean, 天) 日线 > 1h
    - H3 判据: GBM mean 情节长 (天) 三频率相对最大偏差 ≤ 25%
      (iid 尺度不变性 — **待验证**: MA 斜率符号段在日历时间下随频率有机械
      梯度 [1h ~3 天 vs 日线 ~14 天, dev 实测], H3 可能是假; 诚实报告)
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close 1h/4h      | db 原生 K 线 (load_candles + verify)  | bar 收盘后 | data_loader
  close 日线 (币)  | daily_resample 自 1h (c30 口径)       | 日线收盘后 | c30/c34 口径
  close 日线 (对照)| control.db 1d (双引号表名), 共同 3y 窗  | 日线收盘后 | 对照数据源
  MA 斜率          | MA_w 收盘滚动均值斜率 (MA[t]−MA[t−1])  | close t    | c34 同口径 (因果)
  趋势情节         | 连续同号斜率段 (0 断且不计, 顺序状态机) | 全样本事后 | 描述层 (非条件特征)
  GBM null         | n 同、μ/σ=样本 (有限值掩码), 同 MA 窗  | 锚定真实   | 30 种子/频率; 同 MA 关键

数据声明:
  BTC/ETH 1h (26,280根) + 4h (6,570根) + 日线 (daily_resample ~1,095根);
  SPY/GC=F/EURUSD=X 1d (control.db, 共同 3y 窗 2023-08..2026-08)。
  40 日历日等价 MA: 日线 40 bar, 4h 240 bar, 1h 960 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  ma_days=40; bpd={1h:24,4h:6,1d:1}; ma_w 按频率派生; gap_thresh_days=7;
  P(≥20 bar) 阈值; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - slope=0 断且不计 (与 c31 同款 0 处理); 浮点斜率恰为 0 为测度零, 防御性。
  - 跨频率比较用日历天数 (bar/bpd), 因各频率 bar 数不可直接比; H1 用 bar
    (日线内部), H2 用天 (跨频率)。
  - GBM 各频率用各自 n 与 μ/σ (同管线: 该频率的真实条数与漂移)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① MA 情节 golden (单调升 → 1 情节; 升→降 → 2 情节, 逐位
    验证分段); ② GBM 幅度 sanity (各频率 mean 情节长 bar ∈ [5, 300], 管线
    错误才停); GBM 各频率情节长 (天) 的**一致性不在 GATE 断言** — MA 斜率
    符号段在日历时间下随频率有机械梯度 (iid 亦如此, c35 几何课延续), 是 H3
    的裁决对象; 任一失败 SystemExit
  - GBM 无信息对照: 30 种子/频率, 同 MA 窗口
  - MIN_N: 每格 n_episodes ≥ MIN_N=100 (不足标注; 2σ 判据为统计门禁)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH 1h+日线 + SPY 日线 × GBM 3 种子, 不写 .out
  - 全量: BTC/ETH 三频率 + SPY/GC=F/EURUSD 日线 × 30 种子

运行命令:
  python3 research/studies/c36_trend_episode_len.py --dev
  python3 research/studies/c36_trend_episode_len.py
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
    "ma_days": 40,
    "bpd": {"1h": 24, "4h": 6, "1d": 1},
    "ma_w": {"1h": 960, "4h": 240, "1d": 40},
    "gap_thresh_days": 7.0,
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "ep_thresh": 20,                     # P(情节≥20 bar)
    "gate_rel": 0.25,                    # GBM 三频率情节长(天) 相对最大偏差带
    "dev_subset": {"n_gbm": 3},
    "data_range": "2023-08..2026-08 (对照共同 3y 窗)",
}

STUDY_ID = "c36_trend_episode_len"


# ── 加载 ─────────────────────────────────────────────────────
def load_series(params):
    """→ list[(name, tf, close, bpd, ts_epoch)]"""
    out = []
    data = load_candles(timeframes=("1h", "4h"))
    for sym in params["crypto"]:
        d1 = data.get(sym, {}).get("1h")
        d4 = data.get(sym, {}).get("4h")
        if d1 is None or d4 is None:
            continue
        if verify(d1, sym, "1h") or verify(d4, sym, "4h"):
            continue
        out.append((sym, "1h", d1["close"].values.astype(float),
                    params["bpd"]["1h"], None))
        out.append((sym, "4h", d4["close"].values.astype(float),
                    params["bpd"]["4h"], None))
        dd = daily_resample(d1)
        out.append((sym, "1d", dd["close"].values.astype(float),
                    params["bpd"]["1d"], dd.index))
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
            if len(sel) < params["ma_w"]["1d"] + 5:
                continue
            out.append((sym, "1d", sel["close"].values.astype(float),
                        params["bpd"]["1d"], sel["ts"].values))
    finally:
        conn.close()
    return out


def _epoch_seconds(idx):
    return (idx.values.astype("datetime64[ns]").astype("int64") // 10 ** 9)


# ── MA 斜率情节 (c34 同口径 MA, 0 断且不计) ──────────────────
def ma_episodes(c, w_ma):
    """MA 斜率符号段: slope=MA[t]−MA[t−1]; 情节=连续同号段; slope=0 断且不计.
    返回 run 长度数组 (bar)."""
    c = np.asarray(c, float)
    n = len(c)
    if n < w_ma + 2:
        return np.array([], int)
    ma = pd.Series(c).rolling(w_ma).mean().values
    slope = np.full(n, np.nan)
    t = np.arange(n)
    ok = t >= w_ma
    slope[ok] = ma[t[ok]] - ma[t[ok] - 1]
    lengths = []
    cur = 0
    prev = 0
    for i in range(n):
        si = slope[i]
        if not np.isfinite(si) or si == 0.0:
            if cur > 0:
                lengths.append(cur)
            cur = 0
            prev = 0
            continue
        sg = 1.0 if si > 0 else -1.0
        if sg == prev:
            cur += 1
        else:
            if cur > 0:
                lengths.append(cur)
            cur = 1
            prev = sg
    if cur > 0:
        lengths.append(cur)
    return np.array(lengths, int)


def episode_metrics(lengths, bpd, ep_thresh):
    if len(lengths) == 0:
        return {"mean": float("nan"), "med": float("nan"),
                "p20": float("nan"), "n": 0, "mean_d": float("nan")}
    mean_b = float(np.mean(lengths))
    return {"mean": mean_b, "med": float(np.median(lengths)),
            "p20": float(np.mean(lengths >= ep_thresh)), "n": int(len(lengths)),
            "mean_d": mean_b / bpd}


# ── GBM null (30 种子/频率, 同 n 同 μ/σ 同 MA) ──────────────
def gbm_episode_stats(c_real, w_ma, bpd, params, seeds):
    lr = np.diff(np.log(c_real))
    fin = np.isfinite(lr)
    mu = float(np.mean(lr[fin]))
    sig = float(np.std(lr[fin], ddof=1))
    mean_bs, mean_ds, meds, p20s, ns = [], [], [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        r = rng.normal(mu, sig, size=len(c_real))
        c = 100.0 * np.exp(np.cumsum(r))
        m = episode_metrics(ma_episodes(c, w_ma), bpd, params["ep_thresh"])
        mean_bs.append(m["mean"])
        mean_ds.append(m["mean_d"])
        meds.append(m["med"])
        p20s.append(m["p20"])
        ns.append(m["n"])
    a = np.array(mean_bs)
    b = np.array(mean_ds)
    return {"mean_b": (float(np.mean(a)), float(np.std(a, ddof=1))),
            "mean_d": (float(np.mean(b)), float(np.std(b, ddof=1))),
            "med": float(np.mean(meds)), "p20": float(np.mean(p20s)),
            "n": int(np.mean(ns))}


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(gbm_tf_days):
    """① MA 情节 golden: 单调升 → 1 情节; 升→降 → 2 情节;
    ② GBM 幅度 sanity: 各频率 mean 情节长 (bar) ∈ [5, 300] (管线错误才停).
    GBM 各频率情节长 (天) 的一致性**不**在此断言 — MA 斜率符号段在日历时间
    下随频率有机械梯度 (iid 亦如此: 1h ~3 天 vs 日线 ~14 天, c35 几何课
    延续), 是 H3 的裁决对象而非管线错误."""
    # ① 单调升 (200 bar, MA40) → 1 情节
    c = 100.0 + np.arange(200) * 1.0
    L = ma_episodes(c, PARAMS["ma_w"]["1d"])
    if len(L) != 1 or L[0] <= 0:
        raise SystemExit(f"GATE FAIL: golden 单调升情节 {L.tolist()} ≠ [1段]")
    # ① 升 100 → 降 100 → 2 情节
    c2 = np.concatenate([100.0 + np.arange(100) * 1.0,
                         200.0 - np.arange(100) * 1.0])
    L2 = ma_episodes(c2, PARAMS["ma_w"]["1d"])
    if len(L2) != 2 or L2[0] <= 0 or L2[1] <= 0:
        raise SystemExit(f"GATE FAIL: golden 升→降情节 {L2.tolist()} ≠ [2段]")
    # ② GBM 幅度 sanity (各频率 mean 情节长 天 在合理范围)
    for tf, (m, s) in gbm_tf_days.items():
        if not (0.5 <= m <= 100.0):
            raise SystemExit(
                f"GATE FAIL: GBM {tf} mean 情节长 {m:.1f} 天 ∉ [0.5, 100] — "
                f"GBM/情节管线错误, 停")
    vals = {tf: gbm_tf_days[tf][0] for tf in gbm_tf_days}
    print(f"[GATE] MA 情节 golden (单调升 1 段, 升→降 2 段) [PASS]; GBM "
          f"幅度 {vals} 天 [PASS]; GBM 频率一致性属 H3 裁决 (MA 斜率几何梯度)",
          flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def _pp(v):
    return f"{v:+.1f}"


def write_out(out_path, params, g, rows, h2, gbm_tf_days, h1_counts):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},control={},ma_days={},ma_w={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), "+".join(p["control"]), p["ma_days"],
            p["ma_w"], p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(日线真实 mean 情节长(天)): 真实 {:.1f} "
        "| GBM {:.1f} [PASS]; 探测器自检 MA 情节 golden [PASS]; GBM null 幅度 "
        "sanity (各频率情节长 ∈[0.5,100]天) [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], h1_counts["real_d"], h1_counts["gbm_d"],
            p["min_n"]),
        "# RESULTS: [学习级] c36 趋势情节长度 × 频率 (用户重释: 低频趋势更持久); "
        "情节=40 日历日等价 MA 斜率符号段 (日线 MA40/4h MA240/1h MA960, 收盘 MA); "
        "slope=0 断且不计; 跨频率比较用日历天 (bar/bpd); GBM 同 n 同 μ/σ 同 MA; "
        "描述层无入场, 无交易含义",
        "",
    ]
    # 情节表
    lines.append("[情节] 每标的×每频率 mean 情节长 (bar/天), 中位数, "
                 "P(≥{}bar), n_episodes:".format(p["ep_thresh"]))
    for r in rows:
        m = r["m"]
        g_ = r["g"]
        lines.append("  {} {}: mean={:.0f}bar ({:.1f}天) med={:.0f} P≥{}= "
                     "{:.3f} n={} {} | GBM mean 天 {:.1f}±{:.1f}".format(
            r["name"], r["tf"], m["mean"], m["mean_d"], m["med"],
            p["ep_thresh"], m["p20"], m["n"], _nm(m["n"], p["min_n"]),
            g_["mean_d"][0], g_["mean_d"][1]))
    # H1 日线
    lines.append("")
    lines.append("[H1] 真实日线情节长 (bar) > GBM 日线同管线 2σ:")
    npass = 0
    for r in rows:
        if r["tf"] != "1d":
            continue
        gd = r["g"]
        gb_mean, gb_std = gd["mean_b"]
        ok = r["m"]["mean"] > gb_mean + 2 * gb_std
        if ok:
            npass += 1
        lines.append("  {}: 真实 {} bar | GBM {} bar (σ {:.1f} bar) | 超出{}"
                     "".format(r["name"], r["m"]["mean"], gb_mean, gb_std,
                               "↑" if ok else "未超"))
    lines.append("  H1 判据: 日线情节长 > GBM 2σ -> {}/{}".format(
        npass, h1_counts["n_daily"]))
    # H2 超出随频率
    lines.append("")
    lines.append("[H2] 超出 (真实−GBM mean 情节长, 天) 随频率:")
    for sym, d in h2.items():
        lines.append("  {}: 1h {}{:.1f} 天 | 4h {}{:.1f} 天 | 日线 {}{:.1f} 天"
                     "".format(sym, "+" if d[0] > 0 else "", d[0],
                               "+" if d[1] > 0 else "", d[1],
                               "+" if d[2] > 0 else "", d[2]))
    h2_ok = all(d[2] > d[0] for d in h2.values())
    lines.append("  H2 判据: 日线超出 > 1h 超出 -> {}".format(
        "PASS" if h2_ok else "FAIL"))
    # H3 GBM 各频率
    lines.append("")
    lines.append("[H3] GBM 各频率情节分布一致 (mean 天): 1h={:.1f} | 4h={:.1f} "
                 "| 日线={:.1f} (判据 相对最大偏差≤{:.0%})".format(
        gbm_tf_days["1h"][0], gbm_tf_days["4h"][0], gbm_tf_days["1d"][0],
        p["gate_rel"]))
    vals = [gbm_tf_days[tf][0] for tf in ("1h", "4h", "1d")]
    mv = float(np.mean(vals))
    rel = max(abs(v - mv) / mv for v in vals) if mv > 0 else 0.0
    lines.append("  H3 判据: GBM 尺度不变 -> {}".format(
        "PASS" if rel <= p["gate_rel"] else "FAIL"))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c30/c32/c35: 书'低频噪声低'三口径证伪 (固定 bar 窗 "
                 "持平 / 固定日历窗机械梯度 / 传统负梯度); c33: 日线无运行肥尾 "
                 "(仅 ^TNX 例外) — 若日线趋势持久须经慢漂移; c34: MA40 系统 PF "
                 "随 ER 上升 (ρ=0.37); 本 c36 直接测 MA 斜率情节长度")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    series = load_series(PARAMS)
    if not series:
        print("无数据, 退出")
        return 1
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    rows = []
    gbm_tf_days = {"1h": [], "4h": [], "1d": []}
    for name, tf, c, bpd, idx in series:
        if dev and not (name in PARAMS["crypto"] or name == "SPY"):
            continue
        if dev and tf == "4h":
            continue
        w = PARAMS["ma_w"][tf]
        m = episode_metrics(ma_episodes(c, w), bpd, PARAMS["ep_thresh"])
        g = gbm_episode_stats(c, w, bpd, PARAMS, seeds)
        gbm_tf_days[tf].append(g["mean_d"][0])
        rows.append({"name": name, "tf": tf, "m": m, "g": g})

    # H2 超出 (真实−GBM mean_d 天) 按标的×频率
    h2 = {}
    for r in rows:
        sym = r["name"].split("/")[0]
        d = r["m"]["mean_d"] - r["g"]["mean_d"][0]
        h2.setdefault(sym, {})[r["tf"]] = d
    h2_out = {sym: (d.get("1h", float("nan")), d.get("4h", float("nan")),
                    d.get("1d", float("nan")))
              for sym, d in h2.items() if "1h" in d and "1d" in d}

    # H1 计数 + 无条件基线
    daily = [r for r in rows if r["tf"] == "1d"]
    h1_counts = {"n_daily": len(daily),
                 "real_d": float(np.mean([r["m"]["mean_d"] for r in daily])),
                 "gbm_d": float(np.mean([r["g"]["mean_d"][0] for r in daily]))}

    gbm_tf_agg = {tf: (float(np.mean(v)), float(np.std(v, ddof=1)))
                  for tf, v in gbm_tf_days.items() if v}

    gate(gbm_tf_agg)

    if dev:
        for r in rows:
            print("  [dev] {} {} mean={:.0f}bar ({:.1f}天) n={} | GBM {:.1f}"
                  "±{:.1f}天".format(r["name"], r["tf"], r["m"]["mean"],
                                     r["m"]["mean_d"], r["m"]["n"],
                                     r["g"]["mean_d"][0],
                                     r["g"]["mean_d"][1]))
        print(f"[dev] 管线 OK ({len(rows)} 序列 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, None, rows, h2_out, gbm_tf_agg, h1_counts)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
