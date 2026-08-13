#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C18 方向 null + 触位 1:1 无增益 合并复核 (2026-08-13, 无未来函数, 1h/4h)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画两个市场事实
  (① state_features 状态格下的 E1 端点方向概率相对 GBM 无信息对照的偏差;
   ② 聚类位带触碰后 1:1 胜率相对 GBM 与真实无条件基线的净差), 无入场,
  无交易含义, 无任何方向/收益/成本结论, 不主张任何 edge。所有统计为事后
  描述; 若未来用作特征/条件, 必须经滚动口径重验。

============================================================
预注册假设 (运行前冻结, 结论逐条回应, 不得新造):

  c11 方向 null 基线 (合并 PLAN §4 c11):
    H1: 任一预注册条件格 (3 状态 × 2 周期 × 2 侧 = 12 格) 的 E1 端点方向
        概率 真实−GBM |Δ| ≤ 1pp
    H3: 分年 (2024/2025/2026) 主度量 |Δ| ≤ 1.5pp

  c18 触位 1:1 无增益 (合并 PLAN §4 c18):
    H1: 触碰 4 方向 (涨触阻/跌触阻/涨触撑/跌触撑) 1:1 胜率 真实−GBM
        |Δ| ≤ 1pp
    H2: GBM 对照 ≈ 无条件基线 (每格 |GBM_触位 − GBM_无条件(同方向)| ≤ 1pp)

  辅助判据 (非假设, 用于解释 H1 判定, B3c 教训):
    A1: 真实无条件 1:1 基线 (long/short) 与无条件 E1 作为"方向漂移参考";
        触碰条件化是否带来超出无条件漂移的增益, 由 真实_触位 − 真实_无条件
        (同方向) 判定

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/数据        | 计算方式                         | 可用时点   | 依据
  close/high/low/  | research.ctx.make_ctx 统一截断   | bar 收盘后 | ctx 唯一对齐出口
  open/atr/years   |  (内部 iloc[warmup:])            |            | (禁一切手动切片)
  状态序列         | state_features.state_series       | bar 收盘后 | 滚动/ewm 左对齐,
                   |  (make_ctx state_fns 契约, 截断   |            | 因果; 8 细分态
                   |   df 上计算)                      |            |
  状态格           | 合并态: trend_up / trend_down /   | bar 收盘后 | 3 状态 × 2 周期 ×
                   |  neutral(=range+transition)       |            | 2 侧 = 12 格
  状态进入事件     | is_t & ~roll(is_t,1) (布尔掩码)   | bar 收盘后 | 段首第一根
  E1 端点方向      | P(close[t+W] > close[t])         | 全样本事后 | 描述层; 方向度量
                   |  (long 侧; short 侧 = <)         |            | 一律以 close[t] 为
                   |                                  |            | 参照 (B3c 教训: 禁
                   |                                  |            | 位带中心参照); 符号
                   |                                  |            | 度量对数单调, 无
                   |                                  |            | Jensen 偏置
  聚类位带         | levels.cluster_levels 在线聚类    | confirm_at | 冻结后 price/band/
                   |  +冻结 (k=K, tol=0.3, min_touch=2)|            | confirm_at 不可变
                   |                                  |            | (levels R1/R2)
  触碰事件         | bar 区间重叠位带 & t>=confirm_at, | bar 收盘后 | 纯触碰事件, 不
                   |  段首首根; 触向分类: close[t] <   |            | 依赖 confirmed 标签
                   |  price=自下而上, > price=自上而下 |            |
  1:1 胜率         | outcome.evaluate_forward          | 已收盘 bar | 官方引擎 (对称
                   |  (t_mult=T, w=W 默认参数)         |            | t_mult 口径, 无
                   |                                  |            | t_target/t_stop)
  GBM 无信息对照   | sim_market.gbm_matching(ref_df,   | 锚定真实   | 固定种子序列 0..29
                   |  seed) 首标 × 30 种子同管线重放   |            | (MIN_GBM_SEEDS)
  分年             | ctx.years (截断坐标) 事后聚合     | 全样本     | BY_YEAR 成对
                   |                                  |            | (真实+GBM)

============================================================
数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。

设计偏离说明:
  - 预注册"约 12 格"落地为 3 合并状态 (trend_up/trend_down/neutral) × 2 周期
    × 2 侧 = 12 格; 细分阶段 (early/accelerate/late) 仅作 RESULTS 诊断行
    (不参与预注册格, 不进 BY_YEAR)。
  - c18 四方向映射 (B1 旧口径, 仅形状参照): 涨触阻=自下而上触阻力→空
    (B1 阻力触碰→做空); 跌触撑=自上而下触支撑→多 (B1 支撑触碰→做多);
    涨触撑=自下而上触支撑 (支撑破位后回测)→空 (B1 支撑破位→做空);
    跌触阻=自上而下触阻力 (阻力破位后回测)→多 (B1 阻力破位→做多)。
  - GBM 对照为"首标 × 30 种子全管线" (c15 同款, PLAN §4 描述层 exit 模板
    允许的最小覆盖); 分年真实侧聚合全部 20 标的, GBM 侧聚合首标 30 种子。

发布门槛自检 (描述层):
  - GATE 自检: ①1:1 无条件基线 GBM 30 种子同管线 WR ∈ [49%, 51%] (模板
    1:1 断言); ②c11 方向探测器 GBM 30 种子无条件 E1 ∈ [49%, 51%];
    ③GBM 样本 n ≥ MIN_N — 任一失败 SystemExit (违规即停)
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (同管线重放)
  - MIN_N 检查: 每个输出格含 n, 不足格标注 [MIN_N 不足] (全单元格输出)
  - BY_YEAR: 2024/2025/2026 主度量成对输出 (真实+GBM)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义, 不涉及成本 (描述层门槛)

运行命令:
  # 两道门禁: 引擎门禁 → 脚本门禁 → 运行
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c18_dir_null.py
  python3 research/studies/c18_dir_null.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path (脚本以 `python3 research/studies/c18_dir_null.py` 直接运行时,
# sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.caliber import MIN_GBM_SEEDS, MIN_N, T, W
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching
from research.state_features import state_series
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "combo": (2, 0.3),              # cluster_levels (min_touch, tolerance_mult) — 预注册
    "W": W,                          # 结果窗口 24 (caliber)
    "T": T,                          # 1:1 对称目标 1.0×ATR (caliber)
    "warmup": 600,                   # make_ctx 截断起点 (覆盖 atr ewm + 特征 warm-up)
    "states": ("trend_up", "trend_down", "neutral"),   # 3 合并态 (预注册)
    "sides": ("long", "short"),      # 多空两侧 (预注册)
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c18_dir_null"

# c18 4 方向: (名称, 掩码键, 交易方向, 语义)
TOUCH_CELLS = [
    ("涨触阻", "res_below", "short", "自下而上触阻力 (B1 阻力触碰→做空)"),
    ("跌触阻", "res_above", "long",  "自上而下触阻力, 阻力破位后回测 (B1 阻力破位→做多)"),
    ("涨触撑", "sup_below", "short", "自下而上触支撑, 支撑破位后回测 (B1 支撑破位→做空)"),
    ("跌触撑", "sup_above", "long",  "自上而下触支撑 (B1 支撑触碰→做多)"),
]

STAGE_KEYS = ("trend_up:early", "trend_up:accelerate", "trend_up:late",
              "trend_down:early", "trend_down:accelerate", "trend_down:late")


# ── 加载 ─────────────────────────────────────────────────────
def load(timeframes):
    data = load_candles(timeframes=timeframes)
    out = {}
    for sym, tfs in data.items():
        for tf in timeframes:
            df = tfs.get(tf)
            if df is None or verify(df, sym, tf):
                continue
            out.setdefault(tf, []).append(df)
    return out


# ── 状态 (state_features, 8 细分 → 3 合并) ──────────────────
def merged_states(states):
    """8 细分状态 (trend_up:stage / trend_down:stage / range / transition)
    → 3 预注册合并态 (trend_up / trend_down / neutral=range+transition)"""
    n = len(states)
    out = np.empty(n, dtype=object)
    for i in range(n):
        s = states[i]
        if isinstance(s, str) and s.startswith("trend_up"):
            out[i] = "trend_up"
        elif isinstance(s, str) and s.startswith("trend_down"):
            out[i] = "trend_down"
        else:
            out[i] = "neutral"
    return out


def make_ctx_states(df, warmup):
    """make_ctx + state_features 状态 (raw 8 态; merged 由调用方合并)"""
    return make_ctx(df, warmup, state_fns={"state": lambda d: state_series(d)[0]})


def state_entry_mask(states, target):
    """段首进入事件 (布尔掩码, 无切片)"""
    is_t = states == target
    prev = np.roll(is_t, 1)
    prev[0] = False
    return is_t & ~prev


# ── c11: E1 端点方向 ─────────────────────────────────────────
def e1_series(ctx, params):
    """全 bar 无条件 E1 方向 (long 侧): P(close[t+W] > close[t])
    返回 (ev, up, years) — 三数组等长"""
    t_idx = np.arange(ctx.n)
    ok = (t_idx + params["W"] < ctx.n) & np.isfinite(ctx.close)
    ev = np.flatnonzero(ok)
    up = ctx.close[ev + params["W"]] > ctx.close[ev]
    return ev, up, ctx.years[ev]


def c11_collect(ctx, params):
    """单 ctx 的 12 格 E1 事件 → {cell: {"dir": bool[], "y": int[]}}

    cell = (state, side); long 侧 dir=close[t+W]>close[t], short 侧取反;
    事件 = 状态段首 (entry) 且 t+W<n。
    """
    merged = merged_states(ctx.states["state"])
    t_idx = np.arange(ctx.n)
    ok = (t_idx + params["W"] < ctx.n) & np.isfinite(ctx.close)
    out = {}
    for state in params["states"]:
        entry = state_entry_mask(merged, state) & ok
        ev = np.flatnonzero(entry)
        if len(ev) == 0:
            for side in params["sides"]:
                out[(state, side)] = {"dir": np.array([], bool),
                                      "y": np.array([], int)}
            continue
        c_e = ctx.close[ev]
        up = ctx.close[ev + params["W"]] > c_e
        dn = ctx.close[ev + params["W"]] < c_e
        ys = ctx.years[ev]
        for side in params["sides"]:
            ind = up if side == "long" else dn
            out[(state, side)] = {"dir": ind, "y": ys}
    return out


def stage_e1_diag(ctx, params):
    """诊断: 细分阶段态段首 E1 (long 侧) — 解释合并态结果的构成"""
    states = ctx.states["state"]
    t_idx = np.arange(ctx.n)
    ok = (t_idx + params["W"] < ctx.n) & np.isfinite(ctx.close)
    out = {}
    for target in STAGE_KEYS:
        entry = state_entry_mask(states, target) & ok
        ev = np.flatnonzero(entry)
        if len(ev) == 0:
            out[target] = (0, float("nan"))
            continue
        up = ctx.close[ev + params["W"]] > ctx.close[ev]
        out[target] = (len(ev), float(up.mean()))
    return out


# ── c18: 触位 1:1 ────────────────────────────────────────────
def touch_cell_masks(ctx, combo, params):
    """单 ctx 的 4 触位格入场掩码 (全长度布尔)

    触碰 = bar 区间重叠位带 & t>=confirm_at, 段首首根; 触向分类:
      close[t] < lv.price → 自下而上 (涨触); close[t] > lv.price → 自上而下
      (跌触); close[t] == lv.price 精确相等不计入 (两侧均否)。
    """
    mt, tol = combo
    n = ctx.n
    t_idx = np.arange(n)
    lvls = cluster_levels(ctx.high, ctx.low, ctx.atr, k=K,
                          tolerance_mult=tol, min_touch=mt)
    cells = {t[1]: np.zeros(n, bool) for t in TOUCH_CELLS}
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        tm = (ctx.low <= p_hi) & (ctx.high >= p_lo) & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev
        ev = np.flatnonzero(entry)
        if len(ev) == 0:
            continue
        c = ctx.close[ev]
        if lv.side == "resistance":
            cells["res_below"][ev[c < lv.price]] = True
            cells["res_above"][ev[c > lv.price]] = True
        else:
            cells["sup_below"][ev[c < lv.price]] = True
            cells["sup_above"][ev[c > lv.price]] = True
    return cells


def run_1to1(ctx, entries, direction, params):
    """官方引擎 1:1 (对称 t_mult=T, w=W) → (n_win, n_loss, n_expired, n_skip,
    分年 year_wl: {y: [w, l]})"""
    out, recs = evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr, entries,
                                 direction=direction, t_mult=params["T"],
                                 w=params["W"], open_px=ctx.open)
    year_wl = {}
    for r in recs:
        if r.outcome in ("win", "loss"):
            y = ctx.years[r.entry_idx]
            wl = year_wl.setdefault(y, [0, 0])
            wl[0 if r.outcome == "win" else 1] += 1
    return out.n_win, out.n_loss, out.n_expired, out.n_skip, year_wl


def agg_outcomes(parts):
    """list of (nw, nl, ne, ns, year_wl) → 汇总"""
    nw = nl = ne = ns = 0
    year_wl = {}
    for w, l, e, s, yw in parts:
        nw += w
        nl += l
        ne += e
        ns += s
        for y, wl in yw.items():
            ywl = year_wl.setdefault(y, [0, 0])
            ywl[0] += wl[0]
            ywl[1] += wl[1]
    return nw, nl, ne, ns, year_wl


def pool_1to1_uncond(ctxs, direction, params):
    """无条件基线 (全 bar 入场) → (nw, nl, ne, ns, year_wl)"""
    parts = []
    for ctx in ctxs:
        one = np.ones(ctx.n, bool)
        parts.append(run_1to1(ctx, one, direction, params))
    return agg_outcomes(parts)


def pool_1to1_cells(ctxs, combo, params):
    """4 触位格 → {name: (nw, nl, ne, ns, year_wl)}

    每个 ctx 只聚类一次 (touch_cell_masks 缓存), 4 格共用 — 聚类是主耗时,
    避免重复聚类。
    """
    per_ctx = [touch_cell_masks(ctx, combo, params) for ctx in ctxs]
    res = {}
    for name, key, direction, _ in TOUCH_CELLS:
        parts = [run_1to1(ctx, masks[key], direction, params)
                 for ctx, masks in zip(ctxs, per_ctx)]
        res[name] = agg_outcomes(parts)
    return res


# ── 统计工具 ─────────────────────────────────────────────────
def wr_of(nw, nl):
    return float("nan") if nw + nl == 0 else nw / (nw + nl)


def _pct(v):
    return f"{v * 100:.2f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def cell_e1(pooled_cell):
    n = len(pooled_cell["dir"])
    return n, (float(pooled_cell["dir"].mean()) if n else float("nan"))


def cell_year_e1(pooled_cell, year):
    m = pooled_cell["y"] == year
    d = pooled_cell["dir"][m]
    n = int(m.sum())
    return n, (float(d.mean()) if n else float("nan"))


def year_wl_stats(year_wl, year):
    wl = year_wl.get(year, [0, 0])
    return wl[0] + wl[1], wr_of(wl[0], wl[1])


# ── GATE 自检 (违规即停) ─────────────────────────────────────
def gate(real_1h_ctxs, gbm_1h_ctxs, params):
    """①1:1 无条件基线 (真实 1h 全标的 long vs 首标×30种子 GBM long)
    ②c11 方向探测器 (GBM 30 种子无条件 E1 ≈ 50%) ③MIN_N — 任一失败 SystemExit"""
    real = pool_1to1_uncond(real_1h_ctxs, "long", params)
    gbm_parts = []
    for ctx in gbm_1h_ctxs:
        one = np.ones(ctx.n, bool)
        gbm_parts.append(run_1to1(ctx, one, "long", params))
    gbm = agg_outcomes(gbm_parts)
    real_wr = wr_of(real[0], real[1])
    gbm_wr = wr_of(gbm[0], gbm[1])
    gbm_e1s = [float(np.mean(e1_series(ctx, params)[1])) for ctx in gbm_1h_ctxs]
    gbm_e1 = float(np.mean(gbm_e1s))
    n_gbm = gbm[0] + gbm[1]
    print(f"[GATE] 无条件基线(1:1 long, 1h): 真实 {_pct(real_wr)} | "
          f"GBM {_pct(gbm_wr)} | E1探测器 GBM {_pct(gbm_e1)} "
          f"(gbm_seeds={len(gbm_1h_ctxs)})", flush=True)
    if len(gbm_1h_ctxs) < MIN_GBM_SEEDS:
        raise SystemExit(f"GATE FAIL: gbm_seeds={len(gbm_1h_ctxs)} < {MIN_GBM_SEEDS}")
    if not (49.0 <= gbm_wr * 100 <= 51.0):
        raise SystemExit(f"GATE FAIL: GBM 无条件 1:1 WR {gbm_wr:.4f} ∉ [49%, 51%] — 口径偏置, 停")
    if not (49.0 <= gbm_e1 * 100 <= 51.0):
        raise SystemExit(f"GATE FAIL: GBM 无条件 E1 {gbm_e1:.4f} ∉ [49%, 51%] — 方向探测器偏置, 停")
    if n_gbm < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM n={n_gbm} < MIN_N={MIN_N}")
    return {"real_wr": real_wr, "gbm_wr": gbm_wr, "gbm_e1": gbm_e1,
            "n_gbm": n_gbm}


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, c11_real, c11_gbm, c18_real, c18_gbm,
              uncond, by_year_rows, stage_real):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},states={},sides={},combo={},W={},T={},warmup={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), ",".join(p["states"]), ",".join(p["sides"]),
            p["combo"], p["W"], p["T"], p["warmup"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(1:1 1h long, 全bar入场) 真实 {:.2f}% "
        "GBM {:.2f}% [t1:1 PASS]; c11方向探测器 GBM30种子无条件E1 {:.2f}% "
        "[PASS]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], g["real_wr"] * 100, g["gbm_wr"] * 100,
            g["gbm_e1"] * 100, g["n_gbm"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 描述层无入场, 无交易含义; "
        "E1 = P(close[t+W] > close[t]) (long 侧) / P(close[t+W] < close[t]) "
        "(short 侧), W=24, 状态段首入场; 1:1 = outcome.evaluate_forward 对称口径 "
        "(T×ATR, W=24), 触碰段首入场 (levels.cluster_levels 在线聚类+冻结)",
        "",
    ]

    # ── c11: 12 状态格 ──
    lines.append("[H1-c11] 12 状态格 E1 端点方向概率 (真实 vs GBM 首标×30种子同管线):")
    for tf in p["tf_list"]:
        for state in p["states"]:
            for side in p["sides"]:
                rn, re = cell_e1(c11_real[(tf, state, side)])
                gn, ge = cell_e1(c11_gbm[(tf, state, side)])
                net = (re - ge) if np.isfinite(re) and np.isfinite(ge) else float("nan")
                lines.append("  {} {} {}: 真实 {} (n={}) | GBM {} (n={}) | "
                             "净差 {} {}".format(
                    tf, state, side, _pct(re), rn, _pct(ge), gn,
                    _pp(net) if np.isfinite(net) else "-", _nm(rn)))
    lines.append("")

    # ── c11 辅助: 无条件 E1 (漂移参考) ──
    lines.append("[A1-c11] 无条件 E1 (漂移参考, 全 bar):")
    for tf in p["tf_list"]:
        rn, re = cell_e1(c11_real[(tf, "ALL")])
        gn, ge = cell_e1(c11_gbm[(tf, "ALL")])
        net = (re - ge) if np.isfinite(re) and np.isfinite(ge) else float("nan")
        lines.append("  {} 无条件: 真实 {} (n={}) | GBM {} (n={}) | 净差 {} {}".format(
            tf, _pct(re), rn, _pct(ge), gn,
            _pp(net) if np.isfinite(net) else "-", _nm(rn)))
    lines.append("")

    # ── c11 诊断: 细分阶段段首 E1 (真实, long 侧) ──
    lines.append("[诊断-c11] 细分阶段段首 E1 (真实, long 侧, 解释合并态构成):")
    for tf in p["tf_list"]:
        parts = []
        for target in STAGE_KEYS:
            n, v = stage_real[(tf, target)]
            parts.append("{} {} (n={})".format(target, _pct(v), n))
        lines.append("  {}: {}".format(tf, " | ".join(parts)))
    lines.append("")

    # ── c18: 4 触位格 1:1 ──
    lines.append("[H1-c18] 4 触位格 1:1 胜率 (真实 vs GBM 首标×30种子; "
                 "无条件 = 同方向全bar入场基线):")
    for tf in p["tf_list"]:
        for name, key, direction, desc in TOUCH_CELLS:
            rnw, rnl, _, _, _ = c18_real[tf][name]
            gnw, gnl, _, _, _ = c18_gbm[tf][name]
            rn = rnw + rnl
            gn = gnw + gnl
            rw = wr_of(rnw, rnl)
            gw = wr_of(gnw, gnl)
            net = rw - gw
            base_r = uncond[(tf, direction)]
            base_g = uncond[(tf, direction + "_gbm")]
            rbr = wr_of(base_r[0], base_r[1])
            gbr = wr_of(base_g[0], base_g[1])
            rd = rw - rbr if np.isfinite(rbr) else float("nan")
            gd = gw - gbr if np.isfinite(gbr) else float("nan")
            lines.append("  {} {} ({}): 真实 {} (n={}) | GBM {} (n={}) | "
                         "净差 {} | 真实−无条件({}) {} | GBM−无条件({}) {} {}".format(
                tf, name, direction, _pct(rw), rn, _pct(gw), gn, _pp(net),
                direction, _pp(rd) if np.isfinite(rd) else "-",
                direction, _pp(gd) if np.isfinite(gd) else "-", _nm(rn)))
    lines.append("")

    # ── c18 辅助: 无条件 1:1 基线 ──
    lines.append("[A1-c18] 无条件 1:1 基线 (全 bar 入场, 真实 vs GBM):")
    for tf in p["tf_list"]:
        for direction in ("long", "short"):
            rnw, rnl, _, _, _ = uncond[(tf, direction)]
            gnw, gnl, _, _, _ = uncond[(tf, direction + "_gbm")]
            rn = rnw + rnl
            gn = gnw + gnl
            rw = wr_of(rnw, rnl)
            gw = wr_of(gnw, gnl)
            net = rw - gw
            lines.append("  {} {}: 真实 {} (n={}) | GBM {} (n={}) | 净差 {}".format(
                tf, direction, _pct(rw), rn, _pct(gw), gn, _pp(net)))
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(by_year_rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    t0 = time.time()
    dfs = load(PARAMS["tf_list"])
    if not dfs or not dfs.get("1h"):
        print("无数据, 退出")
        return 1

    # 构建 ctxs (state_features 状态): 真实全标的 + GBM 首标×30种子
    real_ctxs = {}
    gbm_ctxs = {}
    for tf in PARAMS["tf_list"]:
        real_ctxs[tf] = [make_ctx_states(df, PARAMS["warmup"])
                         for df in dfs[tf]]
        gbm_ctxs[tf] = [make_ctx_states(gbm_matching(dfs[tf][0], seed=s),
                                        PARAMS["warmup"])
                        for s in range(PARAMS["gbm_seeds"])]

    # GATE 自检 (1h; 失败 SystemExit — 违规即停)
    g = gate(real_ctxs["1h"], gbm_ctxs["1h"], PARAMS)

    # ── c11 池 (12 格 + 无条件 E1) ──
    c11_real = {}
    c11_gbm = {}
    for tf in PARAMS["tf_list"]:
        rcol = [c11_collect(ctx, PARAMS) for ctx in real_ctxs[tf]]
        gcol = [c11_collect(ctx, PARAMS) for ctx in gbm_ctxs[tf]]
        for state in PARAMS["states"]:
            for side in PARAMS["sides"]:
                rd = [c[(state, side)] for c in rcol]
                gd = [c[(state, side)] for c in gcol]
                c11_real[(tf, state, side)] = {
                    "dir": np.concatenate([d["dir"] for d in rd]),
                    "y": np.concatenate([d["y"] for d in rd])}
                c11_gbm[(tf, state, side)] = {
                    "dir": np.concatenate([d["dir"] for d in gd]),
                    "y": np.concatenate([d["y"] for d in gd])}
        re = [e1_series(ctx, PARAMS) for ctx in real_ctxs[tf]]
        ge = [e1_series(ctx, PARAMS) for ctx in gbm_ctxs[tf]]
        c11_real[(tf, "ALL")] = {
            "dir": np.concatenate([up for _, up, _ in re]),
            "y": np.concatenate([ys for _, _, ys in re])}
        c11_gbm[(tf, "ALL")] = {
            "dir": np.concatenate([up for _, up, _ in ge]),
            "y": np.concatenate([ys for _, _, ys in ge])}

    # ── c11 诊断: 细分阶段 (真实) ──
    stage_real = {}
    for tf in PARAMS["tf_list"]:
        for target in STAGE_KEYS:
            ns = []
            vs = []
            for ctx in real_ctxs[tf]:
                n, v = stage_e1_diag(ctx, PARAMS)[target]
                ns.append(n)
                if np.isfinite(v):
                    vs.append(v)
            stage_real[(tf, target)] = (sum(ns),
                                        float(np.mean(vs)) if vs else float("nan"))

    # ── c18 池 ──
    c18_real = {}
    c18_gbm = {}
    for tf in PARAMS["tf_list"]:
        c18_real[tf] = pool_1to1_cells(real_ctxs[tf], PARAMS["combo"], PARAMS)
        c18_gbm[tf] = pool_1to1_cells(gbm_ctxs[tf], PARAMS["combo"], PARAMS)

    # ── 无条件 1:1 基线 (真实 + GBM, long/short) ──
    uncond = {}
    for tf in PARAMS["tf_list"]:
        for direction in ("long", "short"):
            uncond[(tf, direction)] = pool_1to1_uncond(
                real_ctxs[tf], direction, PARAMS)
            gb = agg_outcomes([run_1to1(ctx, np.ones(ctx.n, bool), direction,
                                        PARAMS) for ctx in gbm_ctxs[tf]])
            uncond[(tf, direction + "_gbm")] = gb

    # ── BY_YEAR 成对 (真实 + GBM) ──
    year_rows = []
    for tf in PARAMS["tf_list"]:
        for state in PARAMS["states"]:
            for side in PARAMS["sides"]:
                rp = c11_real[(tf, state, side)]
                gp = c11_gbm[(tf, state, side)]
                for y in PARAMS["by_year_list"]:
                    rn, rv = cell_year_e1(rp, y)
                    gn, gv = cell_year_e1(gp, y)
                    if rn == 0 and gn == 0:
                        continue
                    year_rows.append(
                        "{} {} {} {} 真实 {} (n={}) GBM {} (n={})".format(
                            tf, state, side, y, _pct(rv), rn, _pct(gv), gn))
        for name, key, direction, _ in TOUCH_CELLS:
            for y in PARAMS["by_year_list"]:
                rn, rw = year_wl_stats(c18_real[tf][name][4], y)
                gn, gw = year_wl_stats(c18_gbm[tf][name][4], y)
                if rn == 0 and gn == 0:
                    continue
                year_rows.append(
                    "{} {} {} 真实 {} (n={}) GBM {} (n={})".format(
                        tf, name, y, _pct(rw), rn, _pct(gw), gn))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, c11_real, c11_gbm, c18_real, c18_gbm,
              uncond, year_rows, stage_real)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
