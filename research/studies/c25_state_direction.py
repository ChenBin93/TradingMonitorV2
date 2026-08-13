#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C25 状态条件化方向倾向 (2026-08-13, 无未来函数, 1h 为主 + 4h)

条件层 c2x: 只报告"条件 1:1 胜率与期望"的结构事实, 结论语言按 10 项发布门槛
裁决, 不达标一律写"未达发布门槛", 不主张任何可交易性。

============================================================
研究问题 (预注册, 运行前冻结): 状态单独 (无触碰条件) 是否可交易? trend_up
状态段首进入的逆势方向 1:1 胜率是否有真实−GBM 正净差 (c23 教训: 端点方向 ≠
1:1 命中, 必须直接测 1:1)?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1 (主端点): trend_up 状态段首进入, 逆势 (short), 对称 1:1 (T=1.0, W=24),
     1h, 胜率差 ≥ +3pp (c18 prior: E1 端点 +2.71pp — 若转移到 1:1 应 ≥ +3pp)
  H2 (状态对称检验): trend_down 状态段首进入, 逆势 (long), 1h,
     |胜率差| ≤ 1pp (c18 prior: trend_down 1h 无方向信息 +0.00pp — 若成立则
     效应是 trend_up 单侧的)
  H3 (顺势侧镜像): trend_up 状态段首进入, 顺势 (long), 1h, 胜率差 ≤ -2pp
     (c18 prior: -2.98pp)
  H4 (阶段分层): trend_up 内 early/accel/late 段首进入的逆势胜率差无梯度
     (c18 诊断: 全部 ≈46-47%, 预期 |梯度| < 2pp)
  漂移分解 (报告义务): 净差 = 真实−GBM 同管线; 另报 真实−真实无条件 (同方向
     1:1 全 bar 基线) 分解, 标明状态条件化增量
  成本模型预注册: taker 0.05%×2 + 滑点 1bp + funding 0.01%×3 (1h W=24 跨 3 个
     8h 周期) = 0.14% of notional; 成本后 E[R] 为正才可称达标
  4h 仅报净差 (c17/c18 教训: 4h 效应大部分=环境漂移)

操作定义 (冻结):
  - 状态 = state_features.state_series 语义的向量化复现 (trend_states_vec,
    与 state_series 在 GATE 逐位对拍错位 0); 8 细分态
    (trend_up:{early|accelerate|late} / trend_down:{...} / range / transition)
    与 3 合并态 (trend_up / trend_down / neutral=range+transition)
  - 段首进入事件 = 状态段首第一根 (is_t & ~roll(is_t,1)), 入场 = 该 bar 收盘
    close[i] (market), evaluate_forward 对称 1:1 (t_mult=T, w=W 默认参数),
    判定自 bar i+1 起 (官方引擎 open 出发语义)
  - 方向: trend_up → short (逆势, H1 主端点) / long (顺势镜像, H3);
    trend_down → long (逆势, H2)
  - 阶段 (H4): trend_up:early / trend_up:accelerate / trend_up:late 段首
    逆势 (short) 1:1; 梯度 = max(3 格 ΔWR) − min(3 格 ΔWR)
  - 期望 E[R] 单位 = ATR: 每笔 R ∈ {+1 (win), −1 (loss), 0 (expired/skip)};
    E[R] = (n_win − n_loss)/n_filled; n_filled = 成交且入场的交易数
    (win+loss+expired+skip; 末根截断不计入)
  - 漂移分解: 无条件层 = 真实无条件 − GBM无条件 (同方向, 全 bar 入场);
    状态增量层 = 真实状态格 − 真实无条件 (同方向)
  - 成本: cost_pct = taker 0.05%×2 + 滑点 1bp + funding 0.01%×3 = 0.14%;
    每笔成本 (ATR 单位) = close[entry] × 0.0014 / ATR[entry];
    E_cost = E[R] − mean(cost_atr)
  - HOLDOUT = 末 3 月 (2026-06..08), 主端点一次评估只报方向; 选参不存在
    (参数全部预注册), HOLDOUT 为主端点参数冻结下的一次评估

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/数据        | 计算方式                         | 可用时点   | 依据
  close/high/low/  | research.ctx.make_ctx 统一截断   | bar 收盘后 | ctx 唯一对齐出口
  open/atr/years   |  (内部 iloc[warmup:])            |            | (禁一切手动切片)
  月份 (HOLDOUT)   | df.index.month, keep 掩码对齐    | bar 收盘后 | 布尔掩码截断
                   |  (arange>=warmup 掩码, 无切片)   |            | (禁 iloc 切片)
  状态序列         | trend_states_vec 向量化复现      | bar 收盘后 | 滚动/ewm 左对齐,
                   |  (state_series 语义, GATE 对拍   |            | 因果; 与 state_series
                   |   错位 0)                        |            | 逐位一致 (GATE 断言)
  段首进入事件     | is_t & ~roll(is_t,1) (布尔掩码)  | bar 收盘后 | 段首第一根
  1:1 命中判定     | outcome.evaluate_forward         | 已收盘 bar | 官方引擎 (对称
                   |  (t_mult=T, w=W 默认)            |            | t_mult, 无 t_target)
  无条件基线       | 全 bar 入场 np.ones 同管线       | 已收盘 bar | 漂移分解/双方向 GATE
  GBM 无信息对照   | sim_market.gbm_matching(ref_df,  | 锚定真实   | 固定种子序列 0..29
                   |  seed) 首标 × 30 种子同管线重放  |            | (MIN_GBM_SEEDS)
  分年/HOLDOUT     | ctx.years + 月份掩码 事后聚合    | 全样本     | 成对输出

============================================================
数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  --dev 模式 (PARAMS dev_subset): 前 3 标的 × GBM 3 种子、跳过 BY_YEAR/
  HOLDOUT、不写 .out — 仅管线调试用; 最终 .out 必须全量版 (sha256 锁定)。

设计偏离说明:
  - c18 用 state_features.state_series (逐 bar Python 循环) 计算状态; 本
    研究按性能约定改用 trend_states_vec 向量化复现 (c16 同款, 已在 c16
    GATE 对拍错位 0), 与 state_series 在本研究 GATE 逐位对拍一致。
  - c18 的 E1 端点是 P(close[t+W] > close[t]) 符号度量; 本研究直接测 1:1
    命中率 (open 出发先碰语义 + 双命中 skip + expired), 不经过端点方向
    外推 (c23 教训)。
  - GBM 对照首标 × 30 种子同管线; 分年 GBM 侧聚合首标 30 种子。
  - multiprocessing 白名单已放开但本管线无 cluster_levels, 状态向量化后
    单线程全量 ≤ ~10 min, 未用 Pool (串行确定性优先)。

发布门槛自检 (条件层 10 项, 结论逐项填表):
  ① 真实−RW(30 种子) 超预注册下限 (H1: ΔWR≥+3pp)
  ② 分年 ≥2/3 为正、最差年 ≥−2pp
  ③ 每格 n≥MIN_N
  ④ GATE 条件组无偏 (GBM 侧绝对水平如实报告)
  ⑤ 跨周期+跨参数一致 (1h 主端点 + 4h 净差)
  ⑥ HOLDOUT 末 3 月一次评估方向不变
  ⑦ Holm 校正 (阶段 3 格 + 状态 3 格 = 6 格 ΔWR 单侧 p)
  ⑧ 成本核算后 > 0 (H1 主端点 E_cost)
  ⑨ 结论↔.out↔脚本三重一致 (sha256)
  ⑩ 负结果/未达标格全部记录

运行命令:
  # 两道门禁: 引擎门禁 → 脚本门禁 → 运行
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c25_state_direction.py
  python3 research/studies/c25_state_direction.py            # 全量
  python3 research/studies/c25_state_direction.py --dev      # 调试 (不写 .out)
"""
import hashlib
import os
import sys
import time
from datetime import date
from math import erf, sqrt

# 仓库根入 path (脚本以 `python3 research/studies/c25_state_direction.py` 直接
# 运行时, sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.caliber import MIN_GBM_SEEDS, MIN_N, T, W
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching
from research.state_features import state_series

DEV_MODE = "--dev" in sys.argv

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "main_tf": "1h",                     # 主端点周期
    "W": W,                              # 结果窗口 24 (caliber)
    "T": T,                              # 1:1 对称目标 1.0×ATR (caliber)
    "warmup": 600,                       # make_ctx 截断起点 (覆盖 atr/状态 warm-up)
    "gbm_seeds": 3 if DEV_MODE else MIN_GBM_SEEDS,
    "dev_n_sym": 3 if DEV_MODE else 20,  # dev 前 3 标的; 全量 20
    "by_year_list": (2024, 2025, 2026),
    "holdout": {"year": 2026, "months": (6, 7, 8)},   # 末 3 月
    "cost_taker": 0.0005,                # taker 费率 (单边)
    "cost_slip": 0.0001,                 # 滑点 1bp (总量)
    "cost_funding": 0.0001,              # funding 单周期 1bp
    "funding_periods": 3,                # W=24 (1h) 跨 3 个 8h 周期
    "data_range": "2023-08..2026-08",
}
STUDY_ID = "c25_state_direction"

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


def months_aligned(df, warmup):
    """截断对齐的月份数组 (长度 = ctx.n) — keep 布尔掩码, 无切片"""
    keep = np.arange(len(df)) >= warmup
    return np.asarray(df.index.month)[keep]


# ── 状态 (state_features.state_series 语义的向量化复现) ──────
def _atr_series(df, period=14):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    m = np.arange(n) >= 1
    if n > 1:
        cp = np.full(n, np.nan)
        cp[m] = c[np.arange(n)[m] - 1]
        tr[m] = np.maximum(h[m] - l[m],
                           np.maximum(np.abs(h[m] - cp[m]),
                                      np.abs(l[m] - cp[m])))
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values


def _adx_series(df, period=14):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    up = np.zeros(n)
    dn = np.zeros(n)
    m = np.arange(n) >= 1
    if n > 1:
        hp = np.full(n, np.nan)
        lp = np.full(n, np.nan)
        hp[m] = h[np.arange(n)[m] - 1]
        lp[m] = l[np.arange(n)[m] - 1]
        up[m] = h[m] - hp[m]
        dn[m] = lp[m] - l[m]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.zeros(n)
    if n > 1:
        cp = np.full(n, np.nan)
        cp[m] = c[np.arange(n)[m] - 1]
        tr[m] = np.maximum(h[m] - l[m],
                           np.maximum(np.abs(h[m] - cp[m]),
                                      np.abs(l[m] - cp[m])))
    alpha = 1 / period
    tr_s = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    pdi_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values
    mdi_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values
    s = pdi_s + mdi_s
    dx = np.where(s > 0, 100 * np.abs(pdi_s - mdi_s) / np.where(s > 0, s, 1), 0.0)
    adx = np.zeros(n)
    if n > period:
        tt = np.arange(n)
        m1 = tt < period
        m2 = tt >= period
        adx[m1] = dx[m1]
        tail = pd.Series(dx[tt >= period - 1])
        tw = tail.ewm(alpha=alpha, adjust=False).mean().values
        adx[m2] = tw[np.arange(n)[m2] - period + 1]
    else:
        adx = dx
    return adx


def trend_states_vec(df):
    """state_features.state_series 的向量化复现 (逐位一致, GATE 对拍自检).

    返回 np.ndarray[str]: trend_up:{early|accelerate|late} /
      trend_down:{...} / range / transition / unknown (atr≤1e-9)。
    """
    c = pd.Series(df["close"].values)
    o = pd.Series(df["open"].values)
    v = pd.Series(df["volume"].values)
    atr = pd.Series(_atr_series(df))
    adx = pd.Series(_adx_series(df))
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    dev = (c - ma20) / atr
    slope = (ma20 - ma20.shift(10)) / atr
    spread = (ma20 - ma60) / atr
    mom = (c - c.shift(10)) / atr
    adx_prev = adx.shift(10)
    body = (c - o).abs()
    body_recent = body.rolling(3).mean()
    body_prior = body.rolling(13).mean().shift(3)

    ADX_TREND = 25
    ADX_RANGE = 20
    EARLY_DEV = 0.6
    LATE_DEV = 2.0
    LATE_ADX_FALL = 3.0
    WEAK_BODY_RATIO = 0.6

    a = atr.values
    ad = adx.values
    ap = adx_prev.values
    sl = slope.values
    sp = spread.values
    dv = dev.values
    mo = mom.values
    br = body_recent.values
    bp = body_prior.values
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(bp > 0, br / np.maximum(bp, 1e-12), 1.0)

    trend_ok = (ad >= ADX_TREND) & (np.abs(sl) >= 0.15)
    range_ok = (ad < ADX_RANGE) & (np.abs(sl) < 0.15) & (np.abs(sp) < 0.5)
    state = np.where(trend_ok & (mo > 0), "trend_up",
                     np.where(trend_ok & (mo < 0), "trend_down",
                              np.where(range_ok, "range", "transition")))
    dev_abs = np.abs(dv)
    adx_turn = ad - ap
    late = ((dev_abs > LATE_DEV)
            | ((adx_turn < -LATE_ADX_FALL) & (ad >= ADX_TREND))
            | ((body_ratio < WEAK_BODY_RATIO) & (dev_abs > 1.0)))
    early = ((dev_abs < EARLY_DEV) & (ad >= ADX_TREND)) | \
            ((ap < ADX_TREND - 3) & (ad >= ADX_TREND))
    stage = np.where(late, "late", np.where(early, "early", "accelerate"))
    is_tr = (state == "trend_up") | (state == "trend_down")
    final = np.where(is_tr, np.char.add(np.char.add(state, ":"), stage), state)
    final = final.astype(object)
    final[a <= 1e-9] = "unknown"
    return final


def merged_states(states):
    """8 细分态 → 3 合并态 (trend_up / trend_down / neutral=range+transition)"""
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
    return make_ctx(df, warmup, state_fns={"state": trend_states_vec})


def state_entry_mask(states, target):
    """段首进入事件 (布尔掩码, 无切片)"""
    is_t = states == target
    prev = np.roll(is_t, 1)
    prev[0] = False
    return is_t & ~prev


# ── 无条件 E1 (GATE 探测器) ─────────────────────────────────
def e1_uncond(ctx, params):
    """全 bar 无条件 E1: P(close[t+W] > close[t])"""
    t_idx = np.arange(ctx.n)
    ok = (t_idx + params["W"] < ctx.n) & np.isfinite(ctx.close)
    ev = np.flatnonzero(ok)
    up = ctx.close[ev + params["W"]] > ctx.close[ev]
    return float(np.mean(up)), len(ev)


# ── 1:1 引擎封装 (含分年/HOLDOUT/成本) ──────────────────────
def run_1to1(ctx, months, entries, direction, params):
    """官方引擎 1:1 → (nw, nl, ne, ns, e, cost, e_cost, year_wl, ho)

    E[R] (ATR 单位): win=+1, loss=−1, expired/skip=0, 分母 n_filled=len(recs);
    成本: close[entry] × cost_pct / ATR[entry] (per filled trade);
    HOLDOUT: entry bar 在 2026-06..08 → (win, loss)。
    """
    out, recs = evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr, entries,
                                 direction=direction, t_mult=params["T"],
                                 w=params["W"], open_px=ctx.open)
    cost_pct = 2 * params["cost_taker"] + params["cost_slip"] + \
        params["cost_funding"] * params["funding_periods"]
    year_wl = {}
    ho = [0, 0]
    cost_sum = 0.0
    for r in recs:
        e_px = ctx.close[r.entry_idx]
        a_t = ctx.atr[r.entry_idx]
        cost_sum += e_px * cost_pct / a_t
        if r.outcome in ("win", "loss"):
            y = ctx.years[r.entry_idx]
            wl = year_wl.setdefault(y, [0, 0])
            wl[0 if r.outcome == "win" else 1] += 1
            if y == params["holdout"]["year"] and \
                    months[r.entry_idx] >= params["holdout"]["months"][0] and \
                    months[r.entry_idx] <= params["holdout"]["months"][1]:
                ho[0 if r.outcome == "win" else 1] += 1
    n_filled = len(recs)
    nw = out.n_win
    nl = out.n_loss
    ne = out.n_expired
    ns = out.n_skip
    e = (nw - nl) / n_filled if n_filled else float("nan")
    cost = cost_sum / n_filled if n_filled else float("nan")
    e_cost = e - cost if np.isfinite(e) else float("nan")
    return {"nw": nw, "nl": nl, "ne": ne, "ns": ns, "n_filled": n_filled,
            "e": e, "cost": cost, "e_cost": e_cost,
            "year_wl": year_wl, "ho": ho}


def agg_1to1(parts):
    """多标的/多种子汇总"""
    nw = nl = ne = ns = 0
    cost_sum = 0.0
    year_wl = {}
    ho = [0, 0]
    n_filled = 0
    for p in parts:
        nw += p["nw"]
        nl += p["nl"]
        ne += p["ne"]
        ns += p["ns"]
        n_filled += p["n_filled"]
        cost_sum += p["cost"] * p["n_filled"] if p["n_filled"] else 0.0
        for y, wl in p["year_wl"].items():
            ywl = year_wl.setdefault(y, [0, 0])
            ywl[0] += wl[0]
            ywl[1] += wl[1]
        ho[0] += p["ho"][0]
        ho[1] += p["ho"][1]
    wr = nw / (nw + nl) if nw + nl else float("nan")
    e = (nw - nl) / n_filled if n_filled else float("nan")
    cost = cost_sum / n_filled if n_filled else float("nan")
    e_cost = e - cost if np.isfinite(e) else float("nan")
    return {"nw": nw, "nl": nl, "ne": ne, "ns": ns, "n_filled": n_filled,
            "wr": wr, "e": e, "cost": cost, "e_cost": e_cost,
            "year_wl": year_wl, "ho": ho}


# ── 统计/格式化 ─────────────────────────────────────────────
def _pct(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:.2f}%"


def _pp(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:+.2f}pp"


def _e(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:+.4f}"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def wr_of(nw, nl):
    return float("nan") if nw + nl == 0 else nw / (nw + nl)


def z_norm_p(z):
    """单侧正态 p (H 方向: Δ>0) — math.erf 实现, 无 scipy 依赖"""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def holm_adjust(pvals):
    """Holm step-down: 升序 → q(i)=max_{j≤i} p(j)×(m−j+1), 同序返回"""
    order = sorted(range(len(pvals)), key=lambda k: pvals[k])
    m = len(pvals)
    out = [1.0] * len(pvals)
    running = 0.0
    for j, k in enumerate(order):
        v = min(1.0, pvals[k] * (m - j))
        running = max(running, v)
        out[k] = running
    return out


def delta_stats(real, gbm):
    """ΔWR + z/p (两样本比例, 单侧)"""
    rn = real["nw"] + real["nl"]
    gn = gbm["nw"] + gbm["nl"]
    if rn == 0 or gn == 0:
        return None
    pw = real["nw"] / rn
    gw = gbm["nw"] / gn
    dw = pw - gw
    se = sqrt(pw * (1 - pw) / rn + gw * (1 - gw) / gn)
    if se <= 0:
        return None
    return dw, dw / se, 1.0 - z_norm_p(dw / se)


def cell_line(tag, real, gbm, direction):
    """状态格 1 行: 真实/GBM/净差/漂移分解(真实−真实无条件)/E/cost"""
    rn = real["n_filled"]
    gn = gbm["n_filled"]
    dw = (real["wr"] - gbm["wr"]) if np.isfinite(real["wr"]) and np.isfinite(gbm["wr"]) else float("nan")
    de = (real["e"] - gbm["e"]) if np.isfinite(real["e"]) and np.isfinite(gbm["e"]) else float("nan")
    return ("  {}: 真实 {} (n={}) E{} | GBM {} (n={}) E{} | "
            "ΔWR {} ΔE {} E_cost 真实 {} {}".format(
                tag, _pct(real["wr"]), rn, _e(real["e"]),
                _pct(gbm["wr"]), gn, _e(gbm["e"]),
                _pp(dw), _e(de), _e(real["e_cost"]), _nm(rn)))


# ── GATE 自检 (违规即停) ─────────────────────────────────────
def gate(ref_1h_df, real_ctxs, gbm_ctxs, uncond_real, uncond_gbm, params):
    """①状态对拍 (trend_states_vec vs state_series 错位 0)
    ②1:1 双方向无条件基线 (真实+GBM, GBM long/short ∈ [49%,51%])
    ③E1 探测器 (GBM 无条件 E1 ∈ [49%,51%]) ④MIN_N — 任一失败 SystemExit
    (dev 模式跳过种子数检查, 其余断言保留)"""
    keep = np.arange(len(ref_1h_df)) >= params["warmup"]
    trunc = ref_1h_df.iloc[keep]
    st_ref, _ = state_series(trunc)
    st_vec = trend_states_vec(trunc)
    n_mis = int((st_ref != st_vec).sum())
    print(f"[GATE] 状态对拍: 错位 {n_mis} / {len(st_ref)}", flush=True)
    if n_mis != 0:
        raise SystemExit(f"GATE FAIL: trend_states_vec 与 state_series 错位 {n_mis} — 停")
    gl = uncond_gbm["1h"]["long"]["wr"]
    gs = uncond_gbm["1h"]["short"]["wr"]
    gbm_e1s = [e1_uncond(c[0], params)[0] for c in gbm_ctxs["1h"]]
    gbm_e1 = float(np.mean(gbm_e1s))
    n_gbm = uncond_gbm["1h"]["long"]["n_filled"]
    print(f"[GATE] 无条件基线(1:1 1h long/short): 真实 "
          f"{uncond_real['1h']['long']['wr'] * 100:.2f}%/"
          f"{uncond_real['1h']['short']['wr'] * 100:.2f}% | GBM "
          f"{gl * 100:.2f}%/{gs * 100:.2f}% | E1 {gbm_e1 * 100:.2f}%", flush=True)
    if not DEV_MODE and len(gbm_ctxs["1h"]) < MIN_GBM_SEEDS:
        raise SystemExit(f"GATE FAIL: gbm_seeds={len(gbm_ctxs['1h'])} < {MIN_GBM_SEEDS}")
    if not DEV_MODE and (not (49.0 <= gl * 100 <= 51.0) or not (49.0 <= gs * 100 <= 51.0)):
        raise SystemExit(f"GATE FAIL: GBM 无条件 WR long {gl:.3f} / short {gs:.3f} ∉ [49%, 51%] — 停")
    if not DEV_MODE and not (49.0 <= gbm_e1 * 100 <= 51.0):
        raise SystemExit(f"GATE FAIL: GBM 无条件 E1 {gbm_e1:.3f} ∉ [49%, 51%] — 方向探测器偏置, 停")
    if not DEV_MODE and n_gbm < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM n={n_gbm} < MIN_N={MIN_N}, 停")
    if DEV_MODE:
        print("[GATE] dev 模式: WR/E1 收敛性断言跳过 (30 种子性质); 对拍断言保留",
              flush=True)
    return {"n_mis": n_mis, "gbm_e1": gbm_e1, "n_gbm": n_gbm}


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR/HOLDOUT) ────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, results, by_year_rows, holdout_rows,
              holm_rows, holm_main):
    p = params
    tf1 = p["main_tf"]
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},main={},W={},T={},warmup={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], ",".join(p["tf_list"]), p["main_tf"],
            p["W"], p["T"], p["warmup"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(1:1 1h 全bar入场) long 真实 {:.2f}% "
        "GBM {:.2f}% short 真实 {:.2f}% GBM {:.2f}% [t1:1 PASS]; 状态对拍错位 "
        "{} [PASS]; E1探测器 GBM {:.2f}% [PASS]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"],
            results[tf1]["uncond_real"]["long"]["wr"] * 100,
            results[tf1]["uncond_gbm"]["long"]["wr"] * 100,
            results[tf1]["uncond_real"]["short"]["wr"] * 100,
            results[tf1]["uncond_gbm"]["short"]["wr"] * 100,
            g["n_mis"], g["gbm_e1"] * 100, g["n_gbm"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 条件层 c2x; "
        "状态 = state_features 语义向量化 (GATE 对拍错位 0); 段首入场 = 状态"
        "进入第一根收盘 (market); 1:1 = evaluate_forward 对称口径 (T×ATR, "
        "W=24); E[R] 单位=ATR (win=+1, loss=−1, expired/skip=0); GBM = 首标×30 "
        "种子同管线",
        "",
        "[门槛] H1: ΔWR≥+3pp (trend_up 逆势 short) | H2: |ΔWR|≤1pp (trend_down "
        "逆势 long) | H3: ΔWR≤-2pp (trend_up 顺势 long) | H4: 阶段 |梯度|<2pp | "
        "成本: taker 0.05%×2 + 滑点 1bp + funding 0.01%×3 = 0.14%",
        "",
    ]

    # H1 主端点
    r1 = results[tf1]["up_short"]
    g1 = results[tf1]["up_short_gbm"]
    d = delta_stats(r1, g1)
    lines.append("[H1主端点] 1h trend_up 段首 → short (逆势), 1:1 W=24:")
    lines.append(cell_line(f"1h trend_up→short", r1, g1, "short"))
    lines.append(f"     ΔWR z {d[1]:+.2f} p {d[2]:.3f}" if d else "")
    # 漂移分解
    ur = results[tf1]["uncond_real"]["short"]
    ug = results[tf1]["uncond_gbm"]["short"]
    lines.append("  [漂移分解] 无条件层 (真实−GBM, short): 真实 {} vs GBM {} "
                 "净差 {} | 状态增量层 (真实格−真实无条件): {} − {} = {} {}".format(
        _pct(ur["wr"]), _pct(ug["wr"]),
        _pp(ur["wr"] - ug["wr"]) if np.isfinite(ur["wr"]) and np.isfinite(ug["wr"]) else "-",
        _pct(r1["wr"]), _pct(ur["wr"]),
        _pp(r1["wr"] - ur["wr"]) if np.isfinite(r1["wr"]) and np.isfinite(ur["wr"]) else "-",
        _nm(r1["n_filled"])))
    lines.append("")

    # H2 / H3
    r2 = results[tf1]["dn_long"]
    g2 = results[tf1]["dn_long_gbm"]
    r3 = results[tf1]["up_long"]
    g3 = results[tf1]["up_long_gbm"]
    lines.append("[H2] 1h trend_down 段首 → long (逆势), |ΔWR|≤1pp:")
    lines.append(cell_line("1h trend_down→long", r2, g2, "long"))
    lines.append("[H3] 1h trend_up 段首 → long (顺势镜像), ΔWR≤-2pp:")
    lines.append(cell_line("1h trend_up→long", r3, g3, "long"))
    lines.append("")

    # H4 阶段
    lines.append("[H4] 1h trend_up 阶段段首 → short (逆势), 无梯度 (|梯度|<2pp):")
    stage_rows = []
    stage_net = []
    for st in ("trend_up:early", "trend_up:accelerate", "trend_up:late"):
        rr = results[tf1]["stage_" + st]
        gg = results[tf1]["stage_" + st + "_gbm"]
        dw = (rr["wr"] - gg["wr"]) if np.isfinite(rr["wr"]) and np.isfinite(gg["wr"]) else float("nan")
        stage_net.append(dw)
        lines.append(cell_line("1h {} → short".format(st), rr, gg, "short"))
    grad = (max(stage_net) - min(stage_net)) if np.isfinite(stage_net[0]) and np.isfinite(stage_net[1]) and np.isfinite(stage_net[2]) else float("nan")
    lines.append("     阶段 ΔWR 梯度: {} {}".format(
        _pp(grad) if np.isfinite(grad) else "-",
        "(<2pp 判据)" if np.isfinite(grad) else ""))
    lines.append("")

    # 成本
    lines.append("[成本] 主端点 (1h trend_up→short): cost 0.14% of notional "
                 "(taker 0.05%×2 + 滑点 1bp + funding 0.01%×3):")
    lines.append("  真实 E[R] {} cost {} E_cost {} | GBM E_cost {} | "
                 "净差E_cost {}".format(
        _e(r1["e"]), _e(r1["cost"]), _e(r1["e_cost"]),
        _e(g1["e_cost"]),
        _e(r1["e_cost"] - g1["e_cost"]) if np.isfinite(r1["e_cost"]) and np.isfinite(g1["e_cost"]) else "-"))
    lines.append("")

    # Holm (阶段 3 格 + 状态 3 格 = 6 格 ΔWR)
    lines.append("[Holm] 6 格 (状态 3 + 阶段 3) ΔWR 单侧 p + Holm 校正:")
    lines.extend(holm_rows)
    lines.append("  主端点 (trend_up→short): {}".format(holm_main))
    lines.append("")

    # 4h 仅报净差
    lines.append("[4h仅净差] (c17/c18: 4h 效应大部分=环境漂移, 不做门槛裁决):")
    for tag, key in (("4h trend_up→short", "up_short"),
                     ("4h trend_down→long", "dn_long"),
                     ("4h trend_up→long", "up_long")):
        rr = results["4h"][key]
        gg = results["4h"][key + "_gbm"]
        dw = (rr["wr"] - gg["wr"]) if np.isfinite(rr["wr"]) and np.isfinite(gg["wr"]) else float("nan")
        de = (rr["e"] - gg["e"]) if np.isfinite(rr["e"]) and np.isfinite(gg["e"]) else float("nan")
        lines.append("  {}: 真实 {} (n={}) | GBM {} (n={}) | ΔWR {} ΔE {}".format(
            tag, _pct(rr["wr"]), rr["n_filled"], _pct(gg["wr"]),
            gg["n_filled"], _pp(dw), _e(de)))
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(by_year_rows))
    lines.append("# HOLDOUT: " + " | ".join(holdout_rows))
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

    n_sym = PARAMS["dev_n_sym"]

    # ctxs: 真实 (前 n_sym 标的) + GBM (首标 × gbm_seeds), 逐 tf
    real_ctxs = {}
    gbm_ctxs = {}
    for tf in PARAMS["tf_list"]:
        real_ctxs[tf] = [
            (make_ctx_states(df, PARAMS["warmup"]),
             months_aligned(df, PARAMS["warmup"]))
            for df in dfs[tf][:n_sym]]
        ref = dfs[tf][0]
        gbm_ctxs[tf] = [
            (make_ctx_states(gbm_matching(ref, seed=s), PARAMS["warmup"]),
             months_aligned(gbm_matching(ref, seed=s), PARAMS["warmup"]))
            for s in range(PARAMS["gbm_seeds"])]

    # 无条件基线 (1:1, 全 bar 入场, long/short) — gate + 漂移分解
    def uncond(tf, direction):
        parts = [run_1to1(ctx, mon, np.ones(ctx.n, bool), direction, PARAMS)
                 for ctx, mon in real_ctxs[tf]]
        return agg_1to1(parts)

    def uncond_gbm(tf, direction):
        parts = [run_1to1(ctx, mon, np.ones(ctx.n, bool), direction, PARAMS)
                 for ctx, mon in gbm_ctxs[tf]]
        return agg_1to1(parts)

    uncond_real = {tf: {"long": uncond(tf, "long"), "short": uncond(tf, "short")}
                   for tf in PARAMS["tf_list"]}
    uncond_gbm = {tf: {"long": uncond_gbm(tf, "long"), "short": uncond_gbm(tf, "short")}
                  for tf in PARAMS["tf_list"]}

    # GATE 自检 (失败 SystemExit; dev 模式跳过种子数检查)
    ref1 = dfs["1h"][0]
    g = gate(ref1, real_ctxs, gbm_ctxs, uncond_real, uncond_gbm, PARAMS)

    # 状态格池化 (逐 tf)
    results = {}
    for tf in PARAMS["tf_list"]:
        cells_real = {}
        cells_gbm = {}
        for ctx, mon in real_ctxs[tf]:
            raw = ctx.states["state"]
            merged = merged_states(raw)
            ok = np.arange(ctx.n) < ctx.n - PARAMS["W"]
            m = {
                "up_short": state_entry_mask(merged, "trend_up") & ok,
                "up_long": state_entry_mask(merged, "trend_up") & ok,
                "dn_long": state_entry_mask(merged, "trend_down") & ok,
            }
            for st in ("trend_up:early", "trend_up:accelerate",
                       "trend_up:late"):
                m["stage_" + st] = state_entry_mask(raw, st) & ok
            dirmap = {"up_short": "short", "up_long": "long",
                      "dn_long": "long",
                      "stage_trend_up:early": "short",
                      "stage_trend_up:accelerate": "short",
                      "stage_trend_up:late": "short"}
            for key in m:
                cells_real.setdefault(key, []).append(
                    run_1to1(ctx, mon, m[key], dirmap[key], PARAMS))
        for ctx, mon in gbm_ctxs[tf]:
            raw = ctx.states["state"]
            merged = merged_states(raw)
            ok = np.arange(ctx.n) < ctx.n - PARAMS["W"]
            m = {
                "up_short": state_entry_mask(merged, "trend_up") & ok,
                "up_long": state_entry_mask(merged, "trend_up") & ok,
                "dn_long": state_entry_mask(merged, "trend_down") & ok,
            }
            for st in ("trend_up:early", "trend_up:accelerate",
                       "trend_up:late"):
                m["stage_" + st] = state_entry_mask(raw, st) & ok
            dirmap = {"up_short": "short", "up_long": "long",
                      "dn_long": "long",
                      "stage_trend_up:early": "short",
                      "stage_trend_up:accelerate": "short",
                      "stage_trend_up:late": "short"}
            for key in m:
                cells_gbm.setdefault(key, []).append(
                    run_1to1(ctx, mon, m[key], dirmap[key], PARAMS))
        results[tf] = {}
        for key in cells_real:
            results[tf][key] = agg_1to1(cells_real[key])
            results[tf][key + "_gbm"] = agg_1to1(cells_gbm[key])
        results[tf]["uncond_real"] = uncond_real[tf]
        results[tf]["uncond_gbm"] = uncond_gbm[tf]

    # Holm (1h: up_short/up_long/dn_long + 3 阶段 = 6 格 ΔWR)
    pvals = []
    cells6 = ["up_short", "up_long", "dn_long", "stage_trend_up:early",
              "stage_trend_up:accelerate", "stage_trend_up:late"]
    for key in cells6:
        d = delta_stats(results["1h"][key], results["1h"][key + "_gbm"])
        pvals.append(d[2] if d else 1.0)
    adj = holm_adjust(pvals)
    holm_rows = []
    holm_main = "-"
    for i, key in enumerate(cells6):
        d = delta_stats(results["1h"][key], results["1h"][key + "_gbm"])
        holm_rows.append("  {}: ΔWR {} p {:.3f} p_adj {:.3f}".format(
            key, _pp(d[0]) if d else "-", pvals[i], adj[i]))
        if key == "up_short":
            holm_main = "ΔWR {} p {:.3f} p_adj {:.3f}".format(
                _pp(d[0]) if d else "-", pvals[i], adj[i])

    # BY_YEAR (主端点 1h trend_up→short)
    by_year_rows = []
    if not DEV_MODE:
        r1 = results["1h"]["up_short"]
        g1 = results["1h"]["up_short_gbm"]
        for y in PARAMS["by_year_list"]:
            rw_ = wr_of(r1["year_wl"][y][0], r1["year_wl"][y][1])
            gw_ = wr_of(g1["year_wl"][y][0], g1["year_wl"][y][1])
            rn = r1["year_wl"][y][0] + r1["year_wl"][y][1]
            gn = g1["year_wl"][y][0] + g1["year_wl"][y][1]
            if rn == 0 and gn == 0:
                continue
            by_year_rows.append("1h trend_up→short {} 真实 {} (n={}) GBM {} (n={})".format(
                y, _pct(rw_), rn, _pct(gw_), gn))

    # HOLDOUT (主端点, 2026-06..08, 只报方向)
    holdout_rows = []
    if not DEV_MODE:
        r1 = results["1h"]["up_short"]
        g1 = results["1h"]["up_short_gbm"]
        rw_ = wr_of(r1["ho"][0], r1["ho"][1])
        gw_ = wr_of(g1["ho"][0], g1["ho"][1])
        rn = r1["ho"][0] + r1["ho"][1]
        gn = g1["ho"][0] + g1["ho"][1]
        holdout_rows.append(
            "主端点 trend_up→short 2026-06..08: 真实 {} (n={}) GBM {} (n={}) "
            "| 净差WR {} (只报方向: {})".format(
                _pct(rw_), rn, _pct(gw_), gn,
                _pp(rw_ - gw_) if np.isfinite(rw_) and np.isfinite(gw_) else "-",
                "正" if (np.isfinite(rw_) and np.isfinite(gw_) and rw_ > gw_)
                else "负/不可判"))

    if DEV_MODE:
        print("=== DEV 模式: 不写 .out ===")
        print("[H1] trend_up→short 1h:", cell_line("dev", results["1h"]["up_short"],
              results["1h"]["up_short_gbm"], "short"))
        print("[H2] trend_down→long:", cell_line("dev", results["1h"]["dn_long"],
              results["1h"]["dn_long_gbm"], "long"))
        print("[H3] trend_up→long:", cell_line("dev", results["1h"]["up_long"],
              results["1h"]["up_long_gbm"], "long"))
        for st in ("trend_up:early", "trend_up:accelerate", "trend_up:late"):
            print("[H4]", st, cell_line("dev", results["1h"]["stage_" + st],
                  results["1h"]["stage_" + st + "_gbm"], "short"))
        print(f"运行耗时: {time.time() - t0:.1f}s (dev)")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, results, by_year_rows, holdout_rows,
              holm_rows, holm_main)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
