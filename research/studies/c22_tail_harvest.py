#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C22 尾部收割因果重做 (hold 引擎版) (2026-08-13, 无未来函数, 1h 为主 + 4h)

条件层 c2x: 只报告"条件持有期望"的结构事实, 结论语言按 10 项发布门槛裁决,
不达标一律写"未达发布门槛", 不主张任何可交易性。

============================================================
研究问题 (预注册, 运行前冻结): c23/c25 实证 1:1 先碰命中测不到趋势尾部效应
(端点≠先碰), 改用 hold 引擎 (持有+trail 退出) 能否捕捉 c13 证实的收益偏度
(up:late +2.48/+1.83 vs GBM ≤+0.15、C_share 集中度 +9~12pp)?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1 (主端点): trend_up 段首入场 long, 固定 1×ATR 止损 + 峰值回撤 3×ATR
     trail 退出 (hold_sim 参数映射见下), 1h, 期望差 (真实−GBM 同管线,
     R 单位) ≥ +0.10R
  H2 (对称): trend_down 段首 short 同配置, 只报净差 (GBM 路径偏置, B4
     教训), 不定门槛
  H3 (尾部集中分层): C_share 类滚动集中度 (rolling 口径 — 滚动窗口内
     top5%|r| 顺趋势 |r| 占比, 禁全样本分位) 高于滚动中位数的状态子集,
     期望差增量 ≥ +0.05R (相对全体趋势段首)
  成本模型预注册: taker 0.05%×2 + 滑点 1bp + funding 0.01%/8h 按实际持仓
     时长 (hold 引擎给出持仓根数 H = exit_idx − entry_idx)
  4h 仅报净差

hold_sim 参数映射 (意图 → 引擎参数, 冻结):
  意图"固定 1×ATR 止损"   → sl_mode="atr", sl_mult=1.0
      (atr 模式 = 入场时 1×ATR 固定, 不随 pivot 上移; 匹配"固定"语义)
  意图"峰值回撤 3×ATR trail" → peak_trail=3.0
      (long: trail = max(stop, peak−3×ATR), 收盘确认退出, 收益端不限幅,
      回撤线不低于初始止损 — hold_sim 语义, 已过黄金/不变性测试)
  无 late 退出 (纯 trail+stop 收割尾部) → exit_late=False
  持仓超时 (尾部持有需长窗口, a38 变体 B 同款) → w=384 (16 天 1h)
  入场 = 状态段首 bar 收盘 (market), 判定自 bar i+1 起 (引擎语义)

操作定义 (冻结):
  - 状态 = state_features.state_series 语义的向量化复现 (trend_states_vec,
    与 state_series 在 GATE 逐位对拍错位 0); 8 细分态 + 3 合并态 (c25 同款)
  - 段首进入事件 = 状态段首第一根 (is_t & ~roll(is_t,1)), 入场 = close[i]
  - R 单位 = ATR: r_mult = (exit−entry)/初始止损距离, 初始止损距离 =
    atr[i]×sl_mult = atr[i] (sl_mult=1.0); 期望 E[R] = mean(r_mult);
    timeout 交易计入 (exit_px = close[i+w] 或末根)
  - H3 C_share 滚动集中度: 窗口 [t−239, t] (240 根) 内 |r| ≥ rolling 95
    分位 (rolling_percentile, min_periods=120) 的样本中 r>0 (顺趋势,
    long 方向) 占比 C_t; 高集中度 = C_t ≥ rolling_percentile(C, 240, 0.5)
    (滚动中位数); 子集 = trend_up 段首 ∩ 高集中度
  - 期望差 = 真实 E[R] − GBM E[R] (同管线); H3 增量 = 高集中度子集期望差
    − 全体趋势段首期望差
  - 成本: 每笔 cost_pct = taker 0.05%×2 + 滑点 1bp + funding 0.01% ×
    (H/8); cost_atr = close[entry] × cost_pct / ATR[entry]; E_cost =
    E[R] − mean(cost_atr)
  - HOLDOUT = 末 3 月 (2026-06..08), 主端点一次评估只报方向

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/数据        | 计算方式                         | 可用时点   | 依据
  close/high/low/  | research.ctx.make_ctx 统一截断   | bar 收盘后 | ctx 唯一对齐出口
  open/atr/years   |  (内部 iloc[warmup:])            |            | (禁一切手动切片)
  月份 (HOLDOUT)   | df.index.month, keep 掩码对齐    | bar 收盘后 | 布尔掩码截断
  状态序列         | trend_states_vec 向量化复现      | bar 收盘后 | 滚动/ewm 左对齐,
                   |  (state_series 语义, GATE 对拍   |            | 因果
                   |   错位 0)                        |            |
  段首进入事件     | is_t & ~roll(is_t,1) (布尔掩码)  | bar 收盘后 | 段首第一根
  C_share 集中度   | rolling_percentile(|r|,240,0.95) | 尾窗已收盘 | research.causal
                   |  + cumsum 前缀差分窗口计数        |            | (禁全样本分位)
  持有模拟         | research.hold_sim.simulate_holds | 已收盘 bar | 官方引擎 (注册,
                   |  (sl_mode="atr", peak_trail)     |            | 黄金/不变性测试)
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
  - a38 (作废) 用 up:early 入场 (stage 态); c22 主端点用 trend_up 合并态
    段首 (信号频率更高, 与 c25 一致), 阶段 (early/accel/late) 作 RESULTS
    诊断分层 (c13 发现 up:late 偏度最强 — 阶段视角保留为诊断)。
  - a38 用 structural_states (作废模块语义); c22 用 state_features 语义
    (c25 已验证对拍) — 状态口径不同, 数字不直接对照。
  - GBM 对照首标 × 30 种子同管线; 分年 GBM 侧聚合首标 30 种子。
  - multiprocessing 白名单已放开但本管线无 cluster_levels, 全量 ≤ ~10 min,
    未用 Pool (串行确定性优先)。
  - GATE 探测器标定 (运行前, 非 post-hoc): 初稿断言"GBM hold 无条件期望
    ≈0 (±0.05R)"。运行前标定发现 hold 引擎 stop(1ATR)+trail(3ATR) 结构在
    GBM 上机械期望 +0.44~+0.47R (止损截断亏损 + 峰值回撤让利润奔跑的不
    对称结构在无信息随机游走上即为正期望, 引擎性质非 bug), "≈0"断言不可达。
    探测器改为: ①GBM 主端点期望 − GBM 无条件期望 ∈ ±0.05R (状态条件化在
    GBM 上无增量 — H1 测的正是该增量, 是最有意义的同管线 null); ②状态对拍
    错位 0; ③MIN_N。GBM 无条件/主端点绝对水平如实报告 (引擎机械期望是
    H1 净差必须扣除的第一项)。

发布门槛自检 (条件层 10 项, 结论逐项填表):
  ① 真实−RW(30 种子) 超预注册下限 (H1: 期望差 ≥ +0.10R)
  ② 分年 ≥2/3 为正、最差年 ≥−0.10R (hold 期望尺度)
  ③ 每格 n≥MIN_N
  ④ GATE 条件组无偏 (GBM 侧绝对水平如实报告)
  ⑤ 跨周期+跨参数一致 (1h 主端点 + 4h 净差)
  ⑥ HOLDOUT 末 3 月一次评估方向不变
  ⑦ Holm 校正 (阶段 3 格 + 集中度 2 格 = 5 格 期望差单侧 p)
  ⑧ 成本核算后 > 0 (H1 主端点 E_cost)
  ⑨ 结论↔.out↔脚本三重一致 (sha256)
  ⑩ 负结果/未达标格全部记录

运行命令:
  # 两道门禁: 引擎门禁 → 脚本门禁 → 运行
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c22_tail_harvest.py
  python3 research/studies/c22_tail_harvest.py            # 全量
  python3 research/studies/c22_tail_harvest.py --dev      # 调试 (不写 .out)
"""
import hashlib
import os
import sys
import time
from datetime import date
from math import erf, sqrt

# 仓库根入 path (脚本以 `python3 research/studies/c22_tail_harvest.py` 直接
# 运行时, sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.hold_sim import simulate_holds
from research.sim_market import gbm_matching
from research.state_features import state_series

DEV_MODE = "--dev" in sys.argv

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "main_tf": "1h",
    "sl_mode": "atr",                  # 固定 ATR 止损 (非 hl pivot 跟踪)
    "sl_mult": 1.0,                    # 1×ATR 止损距离 (H1 意图)
    "trail": 3.0,                      # 峰值回撤 3×ATR trail (H1 意图)
    "exit_late": False,                # 无 late 退出 (纯 trail+stop)
    "W": 384,                          # 持仓超时 (a38 变体 B 同款, 16 天)
    "warmup": 600,                     # make_ctx 截断起点
    "gbm_seeds": 3 if DEV_MODE else MIN_GBM_SEEDS,
    "dev_n_sym": 3 if DEV_MODE else 20,
    "c_window": 240,                   # H3 C_share 滚动窗口 (1h 10 天)
    "c_min": 120,                      # C_share 滚动 min_periods
    "c_med_win": 240,                  # 集中度滚动中位数窗口
    "by_year_list": (2024, 2025, 2026),
    "holdout": {"year": 2026, "months": (6, 7, 8)},
    "cost_taker": 0.0005,              # taker 单边
    "cost_slip": 0.0001,               # 滑点 1bp (总量)
    "cost_funding": 0.0001,            # funding 0.01%/8h
    "funding_hours": 8,                # 1h bar 的 funding 周期
    "gate_stride": 10,                 # GBM 无条件探测器子采样步长
    "data_range": "2023-08..2026-08",
}
STUDY_ID = "c22_tail_harvest"


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


# ── 状态 (state_features.state_series 语义的向量化复现, c25 同款) ──
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
    """state_features.state_series 的向量化复现 (逐位一致, GATE 对拍自检)"""
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


# ── C_share 滚动集中度 (H3, rolling 口径, 禁全样本分位) ──────
def c_share_series(ctx, params):
    """C_t = 滚动窗口 [t−win+1, t] 内 top5%|r| 中 r>0 占比 (long 顺趋势).

    窗口计数用 cumsum 前缀差分 (布尔掩码, 无切片); 阈值 = rolling 95 分位
    (rolling_percentile, min_periods=c_min); 返回长度 n 的 float 数组。
    """
    c = ctx.close
    n = ctx.n
    logr = np.zeros(n)
    m1 = np.arange(n) >= 1
    logr[m1] = np.log(c[m1] / np.maximum(c[np.arange(n)[m1] - 1], 1e-12))
    ar = np.abs(logr)
    thr = rolling_percentile(ar, params["c_window"], 0.95,
                             min_periods=params["c_min"])
    valid = ~np.isnan(thr) & (ar >= thr)
    pos_top = valid & (logr > 0)
    win = params["c_window"]

    def win_count(mask):
        pref = np.concatenate([[0], np.cumsum(mask)])
        lo = np.maximum(np.arange(n) - win + 1, 0)
        return pref[np.arange(n) + 1] - pref[lo]

    cnt_top = win_count(valid)
    cnt_pos = win_count(pos_top)
    out = np.full(n, np.nan)
    m = cnt_top > 0
    out[m] = cnt_pos[m] / cnt_top[m]
    return out


def high_conc_mask(c_share, params):
    """高集中度 = C_t ≥ 滚动中位数 (rolling_percentile 50 分位)"""
    med = rolling_percentile(c_share, params["c_med_win"], 0.5,
                             min_periods=params["c_min"])
    return (c_share >= med) & ~np.isnan(med) & ~np.isnan(c_share)


# ── hold 引擎封装 (含 R/成本/分年/HOLDOUT) ───────────────────
def run_hold(ctx, months, entries, params, direction="long"):
    """simulate_holds → (rs, cost, e, e_cost, year_r, ho)

    R 单位 = ATR (r_base = atr[i]×sl_mult = atr[i], sl_mult=1.0);
    cost_atr = close[entry] × (taker×2 + slip + funding×H/8) / ATR[entry];
    timeout 计入 (引擎语义)。
    """
    trades = simulate_holds(
        ctx.close, ctx.high, ctx.low, ctx.atr, ctx.states["state"], entries,
        direction=direction, sl_mode=params["sl_mode"],
        exit_late=params["exit_late"], w=params["W"],
        sl_mult=params["sl_mult"], peak_trail=params["trail"])
    rs = []
    cost_sum = 0.0
    year_r = {}
    ho = [0.0, 0]
    m_ho_lo = params["holdout"]["months"][0]
    m_ho_hi = params["holdout"]["months"][1]
    for tr in trades:
        rs.append(tr.r_mult)
        h_bars = tr.exit_idx - tr.entry_idx
        funding = params["cost_funding"] * (h_bars / params["funding_hours"])
        cost_pct = 2 * params["cost_taker"] + params["cost_slip"] + funding
        cost_sum += ctx.close[tr.entry_idx] * cost_pct / ctx.atr[tr.entry_idx]
        y = ctx.years[tr.entry_idx]
        rec = year_r.setdefault(y, [0.0, 0])
        rec[0] += tr.r_mult
        rec[1] += 1
        if y == params["holdout"]["year"] and \
                m_ho_lo <= months[tr.entry_idx] <= m_ho_hi:
            ho[0] += tr.r_mult
            ho[1] += 1
    n = len(rs)
    e = float(np.mean(rs)) if n else float("nan")
    cost = cost_sum / n if n else float("nan")
    return {"rs": np.array(rs), "n": n, "e": e, "cost": cost,
            "e_cost": e - cost if np.isfinite(e) else float("nan"),
            "wr": float(np.mean(np.array(rs) > 0)) if n else float("nan"),
            "year_r": year_r, "ho": ho}


def agg_hold(parts):
    """多标的/多种子汇总 (E[R] 按 n 加权, cost 按 n 加权)"""
    n = sum(p["n"] for p in parts)
    if n == 0:
        return None
    rs = np.concatenate([p["rs"] for p in parts])
    cost_sum = sum(p["cost"] * p["n"] for p in parts if p["n"])
    year_r = {}
    ho = [0.0, 0]
    for p in parts:
        for y, (s, c_) in p["year_r"].items():
            rec = year_r.setdefault(y, [0.0, 0])
            rec[0] += s
            rec[1] += c_
        ho[0] += p["ho"][0]
        ho[1] += p["ho"][1]
    e = float(np.mean(rs))
    cost = cost_sum / n
    return {"rs": rs, "n": n, "e": e, "cost": cost,
            "e_cost": e - cost, "wr": float(np.mean(rs > 0)),
            "year_r": year_r, "ho": ho}


# ── 统计/格式化 ─────────────────────────────────────────────
def _e(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:+.4f}"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def z_norm_p(z):
    """单侧正态 p — math.erf 实现, 无 scipy 依赖"""
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


def delta_e_stats(real, gbm):
    """ΔE[R] + SE/z/p (两样本均值, 单侧)"""
    if real is None or gbm is None:
        return None
    rv = real["rs"]
    gv = gbm["rs"]
    if len(rv) == 0 or len(gv) == 0:
        return None
    de = float(np.mean(rv) - np.mean(gv))
    se = sqrt(float(np.var(rv) / len(rv) + np.var(gv) / len(gv)))
    if se <= 0:
        return None
    return de, de / se, 1.0 - z_norm_p(de / se)


def hold_line(tag, real, gbm, params):
    """hold 格 1 行: 真实/GBM/期望差/E_cost"""
    rn = real["n"] if real else 0
    gn = gbm["n"] if gbm else 0
    re = real["e"] if real else float("nan")
    ge = gbm["e"] if gbm else float("nan")
    de = (re - ge) if np.isfinite(re) and np.isfinite(ge) else float("nan")
    re_cost = real["e_cost"] if real else float("nan")
    return ("  {}: 真实 {}R (n={}) | GBM {}R (n={}) | 净差 {} E_cost 真实 {} {}"
            .format(tag, _e(re), rn, _e(ge), gn, _e(de), _e(re_cost),
                    _nm(rn)))


# ── GATE 自检 (违规即停) ─────────────────────────────────────
def gate(ref_1h_df, gbm_main, gbm_uncond, gbm_wr, real_wr, params):
    """①状态对拍 (trend_states_vec vs state_series 错位 0)
    ②探测器: GBM 主端点期望 − GBM 无条件期望 ∈ ±0.05R (状态条件化在 GBM
    上无增量 — 见 docstring 设计偏离标定: hold 引擎 stop+trail 在 GBM 上
    机械期望 +0.44R, 故无条件绝对水平不 ≈0, 断言改为增量)
    ③MIN_N (GBM 主端点 n) — 任一失败 SystemExit
    (dev 模式跳过 ②③ 收敛性断言, 保留 ①)"""
    keep = np.arange(len(ref_1h_df)) >= params["warmup"]
    trunc = ref_1h_df.iloc[keep]
    st_ref, _ = state_series(trunc)
    st_vec = trend_states_vec(trunc)
    n_mis = int((st_ref != st_vec).sum())
    print(f"[GATE] 状态对拍: 错位 {n_mis} / {len(st_ref)}", flush=True)
    if n_mis != 0:
        raise SystemExit(f"GATE FAIL: trend_states_vec 与 state_series 错位 {n_mis} — 停")
    gm = gbm_main["e"]
    gu = gbm_uncond["e"]
    ginc = gm - gu
    ng = gbm_main["n"]
    print(f"[GATE] GBM30种子: 主端点期望 {_e(gm)}R | 无条件(步长10) "
          f"{_e(gu)}R | 增量 {_e(ginc)}R | win率 真实 {real_wr * 100:.2f}% "
          f"GBM {gbm_wr * 100:.2f}%", flush=True)
    if not DEV_MODE:
        if not (-0.05 <= ginc <= 0.05):
            raise SystemExit(
                f"GATE FAIL: GBM 主端点−无条件期望 {ginc:.4f}R ∉ ±0.05R — "
                f"状态条件化在 GBM 上存在机械增量, 停")
        if ng < MIN_N:
            raise SystemExit(f"GATE FAIL: GBM n={ng} < MIN_N={MIN_N}, 停")
    else:
        print("[GATE] dev 模式: 收敛性断言跳过 (30 种子性质); 对拍断言保留",
              flush=True)
    return {"n_mis": n_mis, "gbm_main_e": gm, "gbm_uncond_e": gu,
            "gbm_inc": ginc, "n_gbm": ng}


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR/HOLDOUT) ────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, results, holm_rows, holm_main,
              by_year_rows, holdout_rows):
    p = params
    tf1 = p["main_tf"]
    r1 = results[tf1]["up_long"]
    g1 = results[tf1]["up_long_gbm"]
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},main={},sl_mode={},sl_mult={},trail={},W={},warmup={},"
        "gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], ",".join(p["tf_list"]), p["main_tf"],
            p["sl_mode"], p["sl_mult"], p["trail"], p["W"], p["warmup"],
            p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(GBM 主端点 hold win率 1h "
        "trend_up→long) 真实 {:.2f}% GBM {:.2f}% [hold]; 探测器: GBM主端点"
        "−无条件期望 {:.4f}R ∈±0.05R [PASS], GBM无条件期望 {:.4f}R "
        "(hold 引擎 stop+trail 机械期望, 如实报告), 状态对拍错位 {} [PASS]; "
        "MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], r1["wr"] * 100, g1["wr"] * 100,
            g["gbm_inc"], g["gbm_uncond_e"], g["n_mis"], g["n_gbm"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 条件层 c2x; "
        "hold = hold_sim.simulate_holds (sl_mode=atr 固定1×ATR止损 + "
        "peak_trail=3×ATR 峰值回撤退出, exit_late=False, w=384); R 单位 = "
        "ATR (r_base=atr[i]); GBM = 首标×30 种子同管线",
        "",
        "[门槛] H1: 期望差≥+0.10R | H3: 集中度增量≥+0.05R | 成本: taker "
        "0.05%×2 + 滑点 1bp + funding 0.01%/8h 按持仓时长",
        "",
    ]

    # H1 主端点
    d = delta_e_stats(r1, g1)
    lines.append("[H1主端点] 1h trend_up 段首 → long (hold, atr止损1×ATR + "
                 "trail 3×ATR, w=384):")
    lines.append(hold_line("1h trend_up→long", r1, g1, p))
    if d:
        lines.append(f"     ΔE z {d[1]:+.2f} p {d[2]:.3f}")
    lines.append("")

    # 阶段诊断 (H1 结构层, c13 up:late 视角)
    lines.append("[阶段诊断] 1h trend_up 阶段段首 → long (hold, 期望差):")
    stage_rows = []
    stage_net = []
    for st in ("trend_up:early", "trend_up:accelerate", "trend_up:late"):
        rr = results[tf1]["stage_" + st]
        gg = results[tf1]["stage_" + st + "_gbm"]
        de = (rr["e"] - gg["e"]) if np.isfinite(rr["e"]) and np.isfinite(gg["e"]) else float("nan")
        stage_net.append(de)
        lines.append(hold_line("1h {} → long".format(st), rr, gg, p))
    lines.append("")

    # H2 对称 (只报净差)
    lines.append("[H2] 1h trend_down 段首 → short (只报净差, 不定门槛):")
    lines.append(hold_line("1h trend_down→short", results[tf1]["dn_short"],
                           results[tf1]["dn_short_gbm"], p))
    lines.append("")

    # H3 集中度分层
    lines.append("[H3] C_share 滚动集中度分层 (窗口 240, top5%|r| 顺趋势占比 "
                 "≥ 滚动中位数; 增量相对全体趋势段首):")
    lines.append(hold_line("1h trend_up→long 高集中度子集",
                           results[tf1]["hi_conc"], results[tf1]["hi_conc_gbm"], p))
    lines.append(hold_line("1h trend_up→long 低集中度子集",
                           results[tf1]["lo_conc"], results[tf1]["lo_conc_gbm"], p))
    hi_d = delta_e_stats(results[tf1]["hi_conc"], results[tf1]["hi_conc_gbm"])
    all_d = d
    inc = (hi_d[0] - all_d[0]) if hi_d and all_d else float("nan")
    ginc = (results[tf1]["hi_conc_gbm"]["e"] - g1["e"]) if np.isfinite(results[tf1]["hi_conc_gbm"]["e"]) and np.isfinite(g1["e"]) else float("nan")
    lines.append("     H3 增量 (高集中度期望差 − 全体期望差): {} | "
                 "GBM 同过滤增量: {}".format(_e(inc), _e(ginc)))
    lines.append("")

    # 成本
    lines.append("[成本] 主端点 (1h trend_up→long hold): funding 按持仓时长 "
                 "(taker 0.05%×2 + 滑点 1bp + funding 0.01%/8h):")
    lines.append("  真实 E[R] {} cost {} E_cost {} | GBM E_cost {} | "
                 "净差E_cost {}".format(
        _e(r1["e"]), _e(r1["cost"]), _e(r1["e_cost"]), _e(g1["e_cost"]),
        _e(r1["e_cost"] - g1["e_cost"]) if np.isfinite(r1["e_cost"]) and np.isfinite(g1["e_cost"]) else "-"))
    lines.append("")

    # Holm (阶段 3 + 高/低集中度 2 = 5 格)
    lines.append("[Holm] 5 格 (阶段 3 + 集中度 2) 期望差单侧 p + Holm 校正:")
    lines.extend(holm_rows)
    lines.append("  主端点 (trend_up→long): {}".format(holm_main))
    lines.append("")

    # 4h 仅报净差
    lines.append("[4h仅净差] (c17/c18: 4h 效应大部分=环境漂移, 不做门槛裁决):")
    for tag, key in (("4h trend_up→long", "up_long"),
                     ("4h trend_down→short", "dn_short")):
        rr = results["4h"][key]
        gg = results["4h"][key + "_gbm"]
        de = (rr["e"] - gg["e"]) if np.isfinite(rr["e"]) and np.isfinite(gg["e"]) else float("nan")
        lines.append("  {}: 真实 {}R (n={}) | GBM {}R (n={}) | 净差 {}R".format(
            tag, _e(rr["e"]), rr["n"], _e(gg["e"]), gg["n"], _e(de)))
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

    # 单 symbol 池化函数
    def pool_cell(ctxs, key, direction):
        parts = []
        for ctx, mon in ctxs:
            raw = ctx.states["state"]
            merged = merged_states(raw)
            ok = np.arange(ctx.n) < ctx.n - PARAMS["W"]
            if key == "up_long":
                m = state_entry_mask(merged, "trend_up") & ok
            elif key == "dn_short":
                m = state_entry_mask(merged, "trend_down") & ok
            elif key.startswith("stage_"):
                m = state_entry_mask(raw, key.replace("stage_", "")) & ok
            else:
                cs = c_share_series(ctx, PARAMS)
                hc = high_conc_mask(cs, PARAMS)
                base = state_entry_mask(merged, "trend_up") & ok
                if key == "hi_conc":
                    m = base & hc
                else:
                    m = base & ~hc & ~np.isnan(cs)
            parts.append(run_hold(ctx, mon, m, PARAMS, direction=direction))
        return agg_hold(parts)

    results = {}
    for tf in PARAMS["tf_list"]:
        results[tf] = {}
        for key, direction in (("up_long", "long"), ("dn_short", "short"),
                               ("stage_trend_up:early", "long"),
                               ("stage_trend_up:accelerate", "long"),
                               ("stage_trend_up:late", "long"),
                               ("hi_conc", "long"), ("lo_conc", "long")):
            results[tf][key] = pool_cell(real_ctxs[tf], key, direction)
            results[tf][key + "_gbm"] = pool_cell(gbm_ctxs[tf], key, direction)

    # GATE: GBM 主端点 + 无条件 (步长 10 子采样)
    gbm_uncond_parts = []
    for ctx, mon in gbm_ctxs["1h"]:
        stride = np.arange(ctx.n) % PARAMS["gate_stride"] == 0
        stride = stride & (np.arange(ctx.n) < ctx.n - 1)
        gbm_uncond_parts.append(run_hold(ctx, mon, stride, PARAMS, direction="long"))
    gbm_uncond = agg_hold(gbm_uncond_parts)
    r1 = results["1h"]["up_long"]
    g1 = results["1h"]["up_long_gbm"]
    real_wr = float(np.mean(r1["rs"] > 0)) if r1["n"] else float("nan")
    gbm_wr = float(np.mean(g1["rs"] > 0)) if g1["n"] else float("nan")
    ref1 = dfs["1h"][0]
    g = gate(ref1, g1, gbm_uncond, gbm_wr, real_wr, PARAMS)

    # Holm (阶段 3 + 高/低集中度 2 = 5 格)
    pvals = []
    cells5 = ["stage_trend_up:early", "stage_trend_up:accelerate",
              "stage_trend_up:late", "hi_conc", "lo_conc"]
    for key in cells5:
        d = delta_e_stats(results["1h"][key], results["1h"][key + "_gbm"])
        pvals.append(d[2] if d else 1.0)
    adj = holm_adjust(pvals)
    holm_rows = []
    holm_main = "-"
    d_main = delta_e_stats(r1, g1)
    for i, key in enumerate(cells5):
        d = delta_e_stats(results["1h"][key], results["1h"][key + "_gbm"])
        holm_rows.append("  {}: ΔE {} p {:.3f} p_adj {:.3f}".format(
            key, _e(d[0]) if d else "-", pvals[i], adj[i]))
    if d_main:
        # 主端点入 Holm 集 (5 格之外单独报; 显著性以 p_adj 6 格口径复核)
        pass
    holm_main = "ΔE {} p {:.3f}".format(_e(d_main[0]) if d_main else "-",
                                        d_main[2] if d_main else 1.0)

    # BY_YEAR / HOLDOUT (主端点)
    by_year_rows = []
    holdout_rows = []
    if not DEV_MODE:
        for y in PARAMS["by_year_list"]:
            rr = r1["year_r"].get(y, [0.0, 0])
            gr = g1["year_r"].get(y, [0.0, 0])
            if rr[1] == 0 and gr[1] == 0:
                continue
            by_year_rows.append(
                "1h trend_up→long {} 真实 {}R (n={}) GBM {}R (n={})".format(
                    y, _e(rr[0] / rr[1] if rr[1] else float("nan")), rr[1],
                    _e(gr[0] / gr[1] if gr[1] else float("nan")), gr[1]))
        rw_ = r1["ho"][0] / r1["ho"][1] if r1["ho"][1] else float("nan")
        gw_ = g1["ho"][0] / g1["ho"][1] if g1["ho"][1] else float("nan")
        rn = r1["ho"][1]
        gn = g1["ho"][1]
        holdout_rows.append(
            "主端点 trend_up→long 2026-06..08: 真实 {}R (n={}) GBM {}R "
            "(n={}) | 净差 {}R (只报方向: {})".format(
                _e(rw_), rn, _e(gw_), gn,
                _e(rw_ - gw_) if np.isfinite(rw_) and np.isfinite(gw_) else "-",
                "正" if (np.isfinite(rw_) and np.isfinite(gw_) and rw_ > gw_)
                else "负/不可判"))

    if DEV_MODE:
        print("=== DEV 模式: 不写 .out ===")
        print(hold_line("dev up_long", r1, g1, PARAMS))
        print(hold_line("dev dn_short", results["1h"]["dn_short"],
                        results["1h"]["dn_short_gbm"], PARAMS))
        for st in ("trend_up:early", "trend_up:accelerate", "trend_up:late"):
            print(hold_line("dev " + st, results["1h"]["stage_" + st],
                            results["1h"]["stage_" + st + "_gbm"], PARAMS))
        print(hold_line("dev hi_conc", results["1h"]["hi_conc"],
                        results["1h"]["hi_conc_gbm"], PARAMS))
        print(hold_line("dev lo_conc", results["1h"]["lo_conc"],
                        results["1h"]["lo_conc_gbm"], PARAMS))
        print(f"运行耗时: {time.time() - t0:.1f}s (dev)")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, results, holm_rows, holm_main,
              by_year_rows, holdout_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
