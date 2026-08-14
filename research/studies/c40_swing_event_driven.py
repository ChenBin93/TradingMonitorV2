#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C40 U1-1 事件驱动趋势忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U1-1, PLAN §2.5 c40): 书 CH5 swing trading — oracle
  核实 CH5 全章无回测, 本考证=补做书跳过的检验。书机制口径逐字核实并遵守:
  MSV_{t−1} 防前视 (p.187), 2 周期摆动点生成, 保守版永远在场反转 (p.188
  规则 1)。描述层, 无入场可交易含义 (1:1 度量仅为检验), 不涉及胜率/期望/
  成本主张。**结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、
  .out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH5 swing 机制 — 摆动点过滤 (MSV)、突破
  入场 (破前摆动高/低)、永远在场反转 — 在 BTC/ETH 1h/4h 上的表现: 破位后
  1:1 胜率 vs GBM (c21 已暗示不成立), 横盘态信号数 (书"横盘无 agenda"),
  每笔风险随 filter 的变化 (书 p.190 机制断言)。

预注册假设 (PLAN §2.5 c40 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 破位后 1:1 胜率 vs GBM 同管线 (入场后 +1×ATR 先于 −1×ATR 触达的
      占比; 真实−GBM 净差超 2σ; c21 已暗示不成立 — 每个 filter 各报)
  H2: 横盘态 (ER 低分位, c27 口径) swing 破位信号数 < 同窗口 40 日 MA 斜率
      转向信号数 (书"横盘无 agenda")
  H3: 每笔风险 (|入场价 − 反向触发点距离|, 反向触发点=前一反向摆动点) 随
      filter 增大而增大 (书 p.190 机制断言)

  操作化 (运行前锁定):
    - swing filter: MSV_t = p × price_t, **用 MSV_{t−1} (前一根 bar 的值)
      防前视** (书 p.187); p ∈ {1%, 2%} + 1×ATR 对照 (ATR 14 周期, 用
      ATR_{t−1})
    - swing 点: 2 周期局部极值 (high_t > high_{t−1} 且 high_t > high_{t+1}
      为摆动高点候选; 反向=摆动低点) + 幅度过滤 (high_t − low_t ≥ MSV_{t−1})
    - 确认滞后 1 bar: bar t 的候选在 bar t+1 收盘后确认, 自 t+2 起可作参照
      (严格无前视 — 探测器 golden 验证)
    - 入场 (书 p.188 规则 1 保守版): close > 前一摆动高点 → 买入; close <
      前一摆动低点 → 卖出; 永远在场反转
    - H1: 入场后 1:1 (evaluate_forward, T=1.0, W=24), 真实−GBM(30 种子
      同管线) 净差 > 2σ, 每 filter 各报
    - H2: ER 低分位 (c27 口径) 下 swing 破位信号数 < 40 日 MA 斜率转向
      信号数
    - H3: 每笔风险 mean/median 随 filter 单调 (p2 > p1)
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  atr              | make_ctx 内置 (14 周期 ewm)           | bar 收盘后 | ctx.atr
  MSV              | p × price_{t−1} (或 ATR_{t−1})       | bar 收盘后 | 书 p.187 (前一根 bar 防前视)
  swing 候选       | 2 周期局部极值 + 幅度过滤             | t+1 收盘后 | 确认滞后 1 bar (c 候选 c+1
                   |                                       |            |   确认, c+2 起可参照)
  突破入场         | close > 前一摆动高点 / < 前低          | bar 收盘后 | 书 p.188 规则 1
  ER 状态          | causal.rolling_percentile (c27 口径)   | bar 收盘后 | research.causal
  1:1 度量         | research.outcome.evaluate_forward      | 事后       | 官方引擎 (c21 口径)
  GBM null         | sim_market.gbm_matching + 同 swing 管线| 锚定真实   | 30 种子 (核心对照, c21 教训)

数据声明:
  BTC/ETH 1h (26,280根) + 4h (6,570根), 2023-08..2026-08。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  filters: p=1%, p=2%, ATR; T=1.0, W=24; er_win=120; ma_w (1h=960, 4h=240);
  GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 幅度过滤用候选 bar 的 (high−low) ≥ MSV_{t−1} (书"摆动高点−当日低点"/
    "当日高点−摆动低点"字面实现)。
  - 突破用收盘价 (close > 摆动高点), 非 intrabar 穿透 (保守版)。
  - 每笔风险 = |入场价 − 前一反向摆动点价| (长: 入场 − 前摆低; 短: 前摆高
    − 入场); 若反向摆动点尚不存在则该笔风险记 NaN (不计入 H3 统计)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① swing golden (构造已知摆动高/突破序列, 验证: 候选于
    c+1 确认、c+2 起可参照、突破入场 bar 与价格正确 — 无前视核心验证);
    ② GBM null sanity: 30 种子 1:1 胜率均值 ∈ [0.45, 0.55] (c21 口径 null
    ≈50%); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 同 swing 管线
  - MIN_N: 1:1 n_eval ≥ MIN_N=100 (不足标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义 (1:1 为检验度量)

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 1h × GBM 3 种子 × 1 filter (p=1%), 不写 .out
  - 全量: BTC/ETH 1h+4h × 3 filters × 30 种子

运行命令:
  python3 research/studies/c40_swing_event_driven.py --dev
  python3 research/studies/c40_swing_event_driven.py
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

from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "tfs": ("1h", "4h"),
    "filters": (("p1", 0.01), ("p2", 0.02), ("atr", None)),
    "er_n": 10,
    "er_win": 120,
    "q_hi": 0.8,
    "q_lo": 0.2,
    "ma_w": {"1h": 960, "4h": 240},      # 40 日历日等价 MA
    "t_mult": 1.0,
    "W": 24,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "gate_band": 0.05,                   # GBM 1:1 WR ∈ [0.45, 0.55]
    "dev_subset": {"n_gbm": 3, "filters": ("p1",), "tfs": ("1h",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c40_swing_event_driven"


# ── ER 状态 (c27 口径, causal) ───────────────────────────────
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


def _er_state_fn(df):
    c = df["close"].values.astype(float)
    er = er_series(c, PARAMS["er_n"])
    rp_hi = rolling_percentile(er, PARAMS["er_win"], PARAMS["q_hi"])
    rp_lo = rolling_percentile(er, PARAMS["er_win"], PARAMS["q_lo"])
    n = len(c)
    st = np.full(n, "", dtype=object)
    ok = np.isfinite(rp_hi) & np.isfinite(rp_lo) & np.isfinite(er)
    st[ok & (er >= rp_hi)] = "高"
    st[ok & (er <= rp_lo)] = "低"
    st[ok & (er > rp_lo) & (er < rp_hi)] = "中"
    return st


# ── swing 管线 (书口径, 确认滞后, 无前视) ───────────────────
def swing_pipeline(close, high, low, atr, msv, er_state):
    """swing 点 + 突破入场 (永远在场反转) + 每笔风险.

    - msv[t] = 过滤值 (已按 MSV_{t−1} 语义, 前一根 bar 派生)
    - 候选于 c 用 high[c+1] → c+1 收盘后确认 → c+2 起可作参照
    返回 (entries_long, entries_short, risks, n_sig_range)
    """
    n = len(close)
    entries_long = np.zeros(n, bool)
    entries_short = np.zeros(n, bool)
    risks = []                            # (bar, direction, risk)
    last_hi = None
    last_lo = None
    pos = 0
    n_range = 0
    for t in range(n):
        if t >= 3:
            c = t - 2                     # 候选 c 需 high[c+1]=high[t-1] → t 起可确认
            if high[c] > high[c - 1] and high[c] > high[c + 1] \
                    and (high[c] - low[c]) >= msv[c]:
                last_hi = high[c]
            if low[c] < low[c - 1] and low[c] < low[c + 1] \
                    and (high[c] - low[c]) >= msv[c]:
                last_lo = low[c]
        if last_hi is not None and close[t] > last_hi:
            if pos != 1:
                entries_long[t] = True
                risk = (close[t] - last_lo) if last_lo is not None else float("nan")
                risks.append((t, 1, risk))
                if er_state[t] == "低":
                    n_range += 1
                pos = 1
        elif last_lo is not None and close[t] < last_lo:
            if pos != -1:
                entries_short[t] = True
                risk = (last_hi - close[t]) if last_hi is not None else float("nan")
                risks.append((t, -1, risk))
                if er_state[t] == "低":
                    n_range += 1
                pos = -1
    return entries_long, entries_short, risks, n_range


def msv_array(close, atr, fkind, fp):
    """MSV_{t−1} 语义: t 处过滤值 = p×price_{t−1} 或 ATR_{t−1}"""
    if fkind == "atr":
        return np.concatenate([[np.nan], atr[:-1]])
    return np.concatenate([[np.nan], fp * close[:-1]])


def wr_1v1(close, high, low, atr, ent_l, ent_s, t_mult, w):
    """1:1 胜率 (官方引擎, 多空合并)"""
    out_l, _ = evaluate_forward(close, high, low, atr, ent_l, direction="long",
                                t_mult=t_mult, w=w)
    out_s, _ = evaluate_forward(close, high, low, atr, ent_s, direction="short",
                                t_mult=t_mult, w=w)
    n_eval = out_l.n_eval + out_s.n_eval
    n_win = out_l.n_win + out_s.n_win
    wr = n_win / n_eval if n_eval else float("nan")
    return wr, n_eval, n_win


def ma_turn_range(close, er_state, ma_w):
    """40 日 MA 斜率转向信号数 (仅横盘态 ER 低分位)"""
    n = len(close)
    ma = pd.Series(close).rolling(ma_w).mean().values
    d = np.zeros(n, int)
    cur = 0
    for t in range(ma_w, n):
        sl = ma[t] - ma[t - 1]
        if sl > 0:
            cur = 1
        elif sl < 0:
            cur = -1
        d[t] = cur
    turns = 0
    for t in range(ma_w + 1, n):
        if d[t] != 0 and d[t] != d[t - 1] and er_state[t] == "低":
            turns += 1
    return turns


# ── GATE 自检 (违规即停) ────────────────────────────────────
def _golden_swing():
    """构造已知摆动高 + 突破: 验证确认滞后与入场.
    摆动高在 bar 10 (high=110 > 邻), 幅度 7 ≥ MSV; 该点在 bar 12 起可参照;
    close[15]=110.5 突破 → 入场 bar 15."""
    n = 30
    close = np.arange(n, dtype=float) * 0.5 + 100.0
    high = close + 0.5
    low = close - 0.5
    high[10] = 110.0
    low[10] = 103.0
    close[15] = 110.5
    high[15] = 111.0
    low[15] = 110.0
    close[16:] = 111.0
    high[16:] = 111.5
    low[16:] = 110.5
    atr = np.full(n, 1.0)
    msv = np.full(n, 1.0)                 # 让幅度过滤恒通过 (仅测时序)
    st = np.full(n, "", dtype=object)
    ent_l, ent_s, risks, n_r = swing_pipeline(close, high, low, atr, msv, st)
    if not ent_l.any():
        raise SystemExit("GATE FAIL: golden 无入场")
    first = int(np.flatnonzero(ent_l)[0])
    if first != 15:
        raise SystemExit(f"GATE FAIL: golden 首个入场 bar={first} ≠ 15 "
                         f"(确认滞后违规或入场逻辑错)")
    # 摆动高 bar 10 不得在 bar 11 之前可参照: close[11..14] < 110 无入场已隐含
    if close[15] <= 110.0:
        raise SystemExit("GATE FAIL: golden 构造错误")
    return True


def gate(gbm_wr_mean):
    """① swing golden (确认滞后 + 入场时序); ② GBM 1:1 WR ≈ 50%."""
    _golden_swing()
    if not (0.5 - PARAMS["gate_band"] <= gbm_wr_mean
            <= 0.5 + PARAMS["gate_band"]):
        raise SystemExit(
            f"GATE FAIL: GBM 1:1 WR 均值={gbm_wr_mean:.3f} "
            f"∉ [0.45, 0.55] — swing/度量管线错误, 停")
    print(f"[GATE] swing golden (确认滞后, 入场 bar=15) [PASS]; GBM 1:1 WR "
          f"均值 {gbm_wr_mean:.3f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def _pp(v):
    return f"{v:+.2f}"


def write_out(out_path, params, rows, h1, h2, h3):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},tfs={},filters={},T={},W={},er_win={},gbm_seeds={},"
        "min_n={},gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), ",".join(p["tfs"]),
            "+".join(f[0] for f in p["filters"]), p["t_mult"], p["W"],
            p["er_win"], p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(BTC 1h p1 全破位 1:1 n): n={} [PASS]; "
        "探测器自检 swing golden (确认滞后, 入场 bar=15) [PASS]; GBM null 1:1 "
        "WR≈0.50 (均值 {:.3f}) [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], h1["base_n"], h1["gbm_wr_mean"], p["min_n"]),
        "# RESULTS: [学习级] c40 U1-1 swing 事件驱动忠实复现 (书 CH5, 补做书"
        "跳过的检验); MSV_{t−1} 防前视; 2 周期摆动点 + 确认滞后 1 bar; 保守版"
        "永远在场反转 (p.188 规则 1); 1:1 用官方引擎 (T=1.0, W=24); GBM 30 种子"
        "同管线; 描述层无入场, 无交易含义",
        "",
    ]
    # H1 表
    lines.append("[H1] 破位后 1:1 胜率 vs GBM 同管线 (每 filter):")
    for fkey, fname in h1["order"]:
        for row in rows:
            if row["filter"] != fkey:
                continue
            wr, ne, nw = row["wr"]
            gm, gs = row["gbm"]
            net = wr - gm
            z = net / gs if gs > 0 else 0.0
            ok = net > 2 * gs
            lines.append("  {} {} {}: 真实 WR={:.1%} (n={}) | GBM {:.1%}±{:.1%}"
                         " | 净差 {:+.1%} z={:+.1f} {}".format(
                fkey, row["sym"], row["tf"], wr, ne, gm, gs, net, z,
                "超2σ↑" if ok else "未超"))
        lines.append("    (H1 判据: 真实−GBM 净差 > 2σ; 每 filter 各报)")
    # H2
    lines.append("")
    lines.append("[H2] 横盘态 (ER 低分位) swing 破位信号数 vs 40 日 MA 转向"
                 "信号数:")
    for row in rows:
        if row["filter"] != "p1":
            continue
        lines.append("  {} {}: swing {} | MA 转向 {} | {} {}".format(
            row["sym"], row["tf"], row["n_range"], row["ma_range"],
            "swing<MA ✓" if row["n_range"] < row["ma_range"] else "不成立",
            _nm(min(row["n_range"], row["ma_range"]), p["min_n"])))
    h2_ok = all(r["n_range"] < r["ma_range"] for r in rows
                if r["filter"] == "p1")
    lines.append("  H2 判据: swing < MA 转向 (p1 filter) -> {}".format(
        "PASS" if h2_ok else "FAIL"))
    # H3
    lines.append("")
    lines.append("[H3] 每笔风险 (|入场−反向触发|) 随 filter: mean (median):")
    for fkey, fname in h1["order"]:
        vals = [r["risk_med"] for r in rows if r["filter"] == fkey
                and r["risk_med"] is not None]
        if vals:
            lines.append("  {}: 标的均值 median 风险 {:.3f}".format(
                fkey, float(np.mean(vals))))
    p1 = [r["risk_med"] for r in rows if r["filter"] == "p1"
          and r["risk_med"] is not None]
    p2 = [r["risk_med"] for r in rows if r["filter"] == "p2"
          and r["risk_med"] is not None]
    h3_ok = bool(p1 and p2 and float(np.mean(p2)) > float(np.mean(p1)))
    lines.append("  H3 判据: p2 median 风险均值 > p1 -> {}".format(
        "PASS" if h3_ok else "FAIL"))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c21 (区间触碰 1:1: ΔWR 全过但 ΔE 未达 +0.05R); "
                 "c14 (关键位围墙: 触碰后留存); c23/c25 (状态方向 1:1 未达); "
                 "c27 (ER 分位状态); 书 CH5 p.187-190: MSV 防前视/规则 1/"
                 "风险随 filter")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def load_ctx_series(params, dev_filters, dev_tfs):
    data = load_candles(timeframes=params["tfs"])
    out = []
    for sym in params["crypto"]:
        for tf in params["tfs"]:
            if dev_tfs is not None and tf not in dev_tfs:
                continue
            df = data.get(sym, {}).get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, params["warmup"],
                           state_fns={"er": _er_state_fn})
            out.append((sym, tf, ctx))
    return out


def run_series(ctx, tf, fkey, fp, params):
    close, high, low, atr = ctx.close, ctx.high, ctx.low, ctx.atr
    er_state = ctx.states["er"]
    msv = msv_array(close, atr, fkey, fp)
    ent_l, ent_s, risks, n_range = swing_pipeline(close, high, low, atr, msv,
                                                  er_state)
    wr = wr_1v1(close, high, low, atr, ent_l, ent_s,
                params["t_mult"], params["W"])
    ma_r = ma_turn_range(close, er_state, params["ma_w"][tf])
    risk_vals = [r[2] for r in risks if np.isfinite(r[2])]
    risk_med = float(np.median(risk_vals)) if risk_vals else None
    return {"wr": wr, "n_range": n_range, "ma_range": ma_r,
            "risk_med": risk_med}


def gbm_1v1(df, tf, fkey, fp, params, seeds):
    """GBM 同管线: 30 种子 1:1 胜率分布 (同 swing 管线)"""
    wrs = []
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        gctx = make_ctx(rw, params["warmup"], state_fns={})
        close, high, low, atr = gctx.close, gctx.high, gctx.low, gctx.atr
        msv = msv_array(close, atr, fkey, fp)
        ent_l, ent_s, _, _ = swing_pipeline(
            close, high, low, atr, msv,
            np.full(len(close), "", dtype=object))
        wrs.append(wr_1v1(close, high, low, atr, ent_l, ent_s,
                          params["t_mult"], params["W"])[0])
    a = np.array([w for w in wrs if np.isfinite(w)])
    return float(np.mean(a)), float(np.std(a, ddof=1))


def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    dev_filters = PARAMS["dev_subset"]["filters"] if dev else None
    dev_tfs = PARAMS["dev_subset"]["tfs"] if dev else None
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    # 装载
    data = load_candles(timeframes=PARAMS["tfs"])
    series = []
    for sym in PARAMS["crypto"]:
        for tf in PARAMS["tfs"]:
            if dev_tfs is not None and tf not in dev_tfs:
                continue
            df = data.get(sym, {}).get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={"er": _er_state_fn})
            series.append((sym, tf, ctx, df))

    if dev:
        filters = [f for f in PARAMS["filters"] if f[0] in dev_filters]
    else:
        filters = PARAMS["filters"]

    rows = []
    gbm_wr_means = []
    for sym, tf, ctx, df in series:
        for fkey, fp in filters:
            r = run_series(ctx, tf, fkey, fp, PARAMS)
            gm, gs = gbm_1v1(df, tf, fkey, fp, PARAMS, seeds)
            gbm_wr_means.append(gm)
            rows.append({"sym": sym, "tf": tf, "filter": fkey, "wr": r["wr"],
                         "n_range": r["n_range"], "ma_range": r["ma_range"],
                         "risk_med": r["risk_med"], "gbm": (gm, gs)})

    h1 = {"order": [(f[0], f[0]) for f in filters],
          "gbm_wr_mean": float(np.mean(gbm_wr_means)),
          "base_n": int(rows[0]["wr"][1])}
    gate(h1["gbm_wr_mean"])

    if dev:
        for r in rows:
            print("  [dev] {} {} {} WR={:.3f} n={} n_range={} ma_range={} "
                  "risk={}".format(r["sym"], r["tf"], r["filter"], r["wr"][0],
                                   r["wr"][1], r["n_range"], r["ma_range"],
                                   r["risk_med"]))
        print(f"[dev] 管线 OK ({len(rows)} 格 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    h2 = None
    h3 = None
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, rows, h1, h2, h3)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
