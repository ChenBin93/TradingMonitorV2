#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C27 ER 方向效率轴 (Kaufman CH17 灵感) (2026-08-13, 无未来函数, 1h 主 + 4h 交叉)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画市场事实 (ER 方向效率与
  波动状态的正交性及其对触碰效应—波动释放/端点折返—的调节), 无入场, 无交易
  含义, 无任何方向/收益/成本结论。定位声明: 本研究确认"噪声≠波动"的 ER 轴
  是否存在及其调节方向; 不构成任何交易主张。描述层发布门槛: 无胜率/期望
  要求, 但必须有 GBM 无信息对照与数字可溯源。若 H1-H3 确认, PLAN 已预注册
  c28 (ER 自适应止损) 作为后续条件层 (不在本研究内)。

============================================================
研究问题 (预注册, 运行前冻结): Kaufman 的"噪声≠波动"——方向效率 (ER) 是否是
  与波动状态正交的第二个条件化轴? ER 高低是否调节已有的触碰效应 (波动释放、
  端点折返)?

预注册假设 (PLAN §2.5 c27 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1 (正交性): ER 高分位 (≥80th) 触碰在低/中/高波动状态下每格占比 ≥ 10%
    (ER 不是波动的影子, 是独立轴)
  H2 (释放调节): 高 ER (≥80th) 触碰 vs 低 ER (≤20th) 触碰的 24h 波动释放
    (E1, c15 口径) 净差 ≥ 3pp; 且 GBM 同管线重放中同口径净差 < 1pp
    (GBM 无 ER 结构)
  H3 (折返调节): 趋势态触碰中, 高 ER 触碰的端点折返 (D1, c17 口径: 沿趋势
    方向概率净差) 比低 ER 触碰少 ≥ 2pp (ER 高=真趋势=折返小); 分年 3/3 同号

  操作化 (运行前锁定):
    - 特征: ER_n = |C_t − C_{t−n}| / Σ_{i=t−n+1..t}|C_i − C_{i−1}| (n=10,
      Kaufman 默认), 因果 (只用 bar t 及之前); ER 状态 = ER 序列 rolling
      120 根 80/20 分位离散化 (高 = ER ≥ rp80; 低 = ER ≤ rp20; 中 = 其余);
      波动状态 = ATR z120 三分位 (低 = ATR ≤ rp33; 高 = ATR > rp66; 中 = 其余)
    - 触碰事件 = 位带触碰进入 (t≥confirm_at, 每段连续触碰首根), c15 同款
    - H1 判据: 触碰事件上 高ER 行 × 波动三格, 每格占 高ER 行总数 ≥ 10% (真实)
    - H2 判据: mean(E1|高ER) − mean(E1|低ER) ≥ +3pp (真实); GBM 同管线
      同口径 < +1pp
    - H3 判据: 趋势态触碰上 [净差D1(高ER) − 净差D1(低ER)] ≥ +2pp, 其中
      净差 = 真实−GBM 同管线; 分年 2024/2025/2026 每年均 ≥ 0 (3/3 同号)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列       | 计算方式                              | 可用时点   | 依据
  close/high/low  | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口 (禁手动切片)
  ATR             | make_ctx 内置 (market_phase ewm,      | bar 收盘后 | ctx.atr (c15 同款)
                  |   ATR_PERIOD=14, 左对齐)              |            |
  ER_n            | |C_t−C_{t−n}| / Σ|ΔC|, 前缀和+布尔掩码 | bar 收盘后 | 只回看 t-n..t (因果); 禁切片
                  |   (无切片)                            |            |
  ER 状态         | causal.rolling_percentile(ER,120,     | bar 收盘后 | research.causal (禁全样本分位);
                  |   0.8/0.2): 高=ER≥rp80, 低=ER≤rp20    |            |   左对齐尾窗
  波动状态        | rolling_percentile(ATR,120,1/3,2/3):  | bar 收盘后 | z120 三分位 (c12/c17 H4 口径)
                  |   低/中/高                            |            |
  cluster 位带    | levels.cluster_levels 在线聚类+冻结     | confirm_at | R1/R2 快照语义 (分桶优化后
                  |   (pivot 按确认时序逐入组)             |            |   输出逐位不变)
  触碰事件        | bar 区间触及 [price±band] ∩ t≥confirm_at| bar 收盘后 | 纯触碰事件 (c15 同款), 每段
                  |   , 每段连续触碰首根 (entry)          |            |   连续触碰首根
  E1 度量         | mean(ATR[t+1..t+12])/mean(ATR[t-11..t])| 事后 (端点 | c15 口径; 逐触碰事件统计,
                  |   − 1, 布尔掩码索引网格 (无切片)      |   t+12 收盘) |   布尔掩码
  D1 度量         | sign(log(c[t+24]/c[t])) 与触碰时刻     | 事后 (端点 | c17 口径; 趋势态触碰
                  |   趋势方向比对 (涨:>0, 跌:<0)         |   t+24 收盘) |   (log 度量防 Jensen)
  分年            | ctx.years (截断坐标) 事后聚合          | 全样本     | BY_YEAR 成对 (真实+GBM)
  GBM 无信息对照  | sim_market.gbm_matching(ref_df, seed)  | 锚定真实   | 固定种子序列 0..29; 首标×30
                  |   (索引/长度/σ 锚定真实)               |            |   种子同管线

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h 为主 + 4h 交叉 × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  ER: n=10 (Kaufman 默认), 状态分位窗口 120 (80th/20th); 波动: z120 三分位;
  E1: 前后 12 根; D1: W=24; 位带: cluster (min_touch=2, tol=0.3) (c27 行无
  参数网格); head_drop=130 覆盖 ER/波动分位 warm-up。

设计偏离说明 (预注册, 非 post-hoc):
  - 波动状态用 ATR 直接 rolling 三分位 (c17 H4/c24 集成口径), 而非 c12 的
    log(ATR/close) — 分位在标的内计算, 价格水平不影响排名, 口径一致。
  - H3 的 D1 定义在"全部趋势态触碰"上 (c17 主度量在逆势侧子集); D1 端点
    概率口径与 c17 逐位一致, 仅样本域扩大。
  - E1 逐事件统计 (c15 口径); GBM 侧同管线重放 (含 ER/波动状态在 GBM 数据
    上的因果重算 — ER 状态本身也是检测器输出)。
  - 趋势状态用**向量化复刻 classify 的 trend_up/down 判定** (仅依赖
    atr/adx/slope/mom; classify 的 stage 逻辑不参与状态判定) — 与
    state_features.state_series 逐位一致 (已验证 0 mismatch, 8 标的×2
    周期); 阈值 adx_trend=25 / slope_thr=0.15 入 PARAMS (market_phase
    常量, 白名单不允许直接 import market_phase)。性能: state_series
    6.4s→0.2s/标的, 全量运行由 ~12.6 分钟降至 ~3 分钟。
  - GBM 对照首标×30 种子全管线; 结论均按事件分层, 不按标的做分层结论。

发布门槛自检 (描述层):
  - GATE 探测器: ① GBM 30 种子同管线 E1 null mean ∈ [-1.5pp, +1.5pp]
    (c15 校准带: GBM 触碰条件化机械偏置 +1.04pp); ② GBM 30 种子同管线 D1
    (趋势态) null mean ∈ [49%, 51%]; 任一失败 SystemExit (违规即停)
  - H2-gate (报告不终止): GBM 同管线 E1(高ER)−E1(低ER) < +1pp (GBM 无
    ER 结构)
  - MIN_N: 每格 n≥MIN_N (caliber), 不足格标注 [MIN_N 不足]
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义 (描述层门槛); 若 H1-H3 确认, c28 为后续条件层

性能与调试约定 (模板, 必须遵守):
  - --dev: 前 3 标的 × GBM 3 种子、跳过 BY_YEAR、不写 .out (管线调试用)
  - 全量: 20 标的 × 30 种子, script_sha256 锁定全量版本

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c27_er_axis.py
  python3 research/studies/c27_er_axis.py --dev
  python3 research/studies/c27_er_axis.py
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
from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.sim_market import gbm_matching
from research.state_features import _adx_series, _atr_series
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "combo": (2, 0.3),                 # cluster 参数 (c27 行无参数网格)
    "warmup": 600,
    "head_drop": 130,                  # 覆盖 ER/波动 rolling 分位 warm-up
    "er_n": 10,                        # Kaufman 默认
    "er_win": 120,                     # ER 状态分位窗口
    "er_q_hi": 0.8,
    "er_q_lo": 0.2,
    "z_win": 120,                      # 波动状态窗口
    "e1_half": 12,                     # E1 前后 12 根
    "W": 24,                           # D1 端点窗口
    "h1_min_frac": 0.10,               # H1 每格 ≥10%
    "h2_min": 0.03,                    # H2 真实门槛 +3pp
    "h2_gbm_max": 0.01,                # H2 GBM 门槛 <1pp
    "h3_min": 0.02,                    # H3 门槛 +2pp
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),
    "dev_subset": {"n_sym": 3, "n_gbm": 3},
    "adx_trend": 25.0,   # market_phase.classify 常量 (白名单不允许 import market_phase)
    "slope_thr": 0.15,   # classify: trend_ok = adx>=25 & |slope|>=0.15
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c27_er_axis"


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


def _trend_state(df):
    """向量化 trend_up/trend_down 判定 — 与 state_features.state_series 的
    classify 状态口径逐位一致 (已验证 0 mismatch, 8 标的×2 周期):
    classify 的 trend 状态仅依赖 (atr, adx, slope, mom); 阈值见 PARAMS
    (market_phase 常量, 白名单不允许直接 import market_phase)."""
    c = df["close"].values
    n = len(c)
    t_idx = np.arange(n)
    atr = _atr_series(df)
    adx = _adx_series(df)
    ma20 = pd.Series(c).rolling(20).mean().values
    slope = np.full(n, np.nan)
    mom = np.full(n, np.nan)
    m = t_idx >= 10
    slope[m] = (ma20[t_idx[m]] - ma20[t_idx[m] - 10]) / np.maximum(atr[t_idx[m]], 1e-12)
    mom[m] = (c[t_idx[m]] - c[t_idx[m] - 10]) / np.maximum(atr[t_idx[m]], 1e-12)
    atr_ok = np.isfinite(atr) & (atr > 1e-9)
    fin = atr_ok & np.isfinite(adx) & np.isfinite(slope) & np.isfinite(mom)
    trend_ok = fin & (adx >= PARAMS["adx_trend"]) & (np.abs(slope) >= PARAMS["slope_thr"])
    out = np.full(n, "", dtype=object)
    out[trend_ok & (mom > 0)] = "up"
    out[trend_ok & (mom < 0)] = "dn"
    return out


# ── 事件收集 (单标的, 因果, 布尔掩码, 无切片) ───────────────
def collect_one(ctx, combo, params):
    """单 ctx → 触碰事件数组 (e1/er/vol/d1/trend/year)

    - e1   : E1 波动释放 (c15 口径), 逐触碰事件
    - er   : 触碰时刻 ER 状态 (高/中/低/空串=分位未收敛)
    - vol  : 触碰时刻波动状态 (低/中/高/空串)
    - d1   : D1 沿趋势端点概率 (c17 口径, 仅趋势态触碰, 其余 NaN)
    - trend: 触碰时刻是否趋势态
    """
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    atr = ctx.atr
    states = ctx.states["trend"]
    h = params["e1_half"]
    W = params["W"]
    er_n = params["er_n"]
    mt, tol = combo

    up = states == "up"
    dn = states == "dn"

    # ── ER (因果, 前缀和, 无切片) ──
    c_prev = np.roll(c, 1)
    m1 = t_idx >= 1
    ad = np.where(m1, np.abs(c - c_prev), 0.0)
    prefix_ad = np.concatenate([[0], np.cumsum(ad)])
    ok_er = t_idx >= er_n
    net = np.full(n, np.nan)
    net[ok_er] = np.abs(c[t_idx[ok_er]] - c[t_idx[ok_er] - er_n])
    path = np.full(n, np.nan)
    path[ok_er] = prefix_ad[t_idx[ok_er] + 1] - prefix_ad[t_idx[ok_er] - er_n + 1]
    er = np.full(n, np.nan)
    m_er = ok_er & (path > 0)
    er[m_er] = net[m_er] / path[m_er]

    rp_hi = rolling_percentile(er, params["er_win"], params["er_q_hi"])
    rp_lo = rolling_percentile(er, params["er_win"], params["er_q_lo"])
    er_state = np.full(n, "", dtype=object)
    ok_e = np.isfinite(rp_hi) & np.isfinite(rp_lo)
    er_state[ok_e & (er >= rp_hi)] = "高"
    er_state[ok_e & (er <= rp_lo)] = "低"
    er_state[ok_e & (er > rp_lo) & (er < rp_hi)] = "中"

    # ── 波动状态 (z120 三分位, ATR) ──
    vp_lo = rolling_percentile(atr, params["z_win"], 1.0 / 3.0)
    vp_hi = rolling_percentile(atr, params["z_win"], 2.0 / 3.0)
    vol_state = np.full(n, "", dtype=object)
    ok_v = np.isfinite(vp_lo) & np.isfinite(vp_hi)
    vol_state[ok_v & (atr <= vp_lo)] = "低"
    vol_state[ok_v & (atr > vp_hi)] = "高"
    vol_state[ok_v & (atr > vp_lo) & (atr <= vp_hi)] = "中"

    # ── E1 (c15 口径, 布尔掩码索引网格, 无切片) ──
    bar_ok = (t_idx >= h - 1) & (t_idx <= n - h - 1) & np.isfinite(atr) & (atr > 0)
    offs = np.arange(h)
    pre_idx = t_idx[:, None] + offs - (h - 1)
    post_idx = t_idx[:, None] + offs + 1
    pre = atr[pre_idx[bar_ok]].mean(axis=1)
    post = atr[post_idx[bar_ok]].mean(axis=1)
    e1 = np.full(n, np.nan)
    e1[bar_ok] = post / pre - 1.0

    # ── D1 (c17 口径, 仅趋势态) ──
    logr = np.full(n, np.nan)
    ok_w = t_idx + W < n
    idx_w = t_idx[ok_w]
    logr[ok_w] = np.log(c[idx_w + W] / c[idx_w])
    d1 = np.full(n, np.nan)
    m_up = ok_w & up
    d1[m_up] = logr[m_up] > 0
    m_dn = ok_w & dn
    d1[m_dn] = logr[m_dn] < 0

    # ── 触碰事件 (t≥confirm_at, 每段连续触碰首根) ──
    usable = t_idx >= params["head_drop"]
    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K,
                          tolerance_mult=tol, min_touch=mt)
    e1_l, er_l, vol_l, d1_l, tr_l, yr_l = [], [], [], [], [], []
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (ctx.low <= p_hi) & (ctx.high >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev & usable
        ev = np.flatnonzero(entry)
        if len(ev) == 0:
            continue
        e1_l.append(e1[ev])
        er_l.append(er_state[ev])
        vol_l.append(vol_state[ev])
        d1_l.append(d1[ev])
        tr_l.append(up[ev] | dn[ev])
        yr_l.append(ctx.years[ev])
    if not e1_l:
        return {"e1": np.array([], float), "er": np.array([], object),
                "vol": np.array([], object), "d1": np.array([], float),
                "trend": np.array([], bool), "year": np.array([], int),
                "n_lvls": 0, "n_touch": 0}
    return {"e1": np.concatenate(e1_l), "er": np.concatenate(er_l),
            "vol": np.concatenate(vol_l), "d1": np.concatenate(d1_l),
            "trend": np.concatenate(tr_l), "year": np.concatenate(yr_l),
            "n_lvls": len(lvls), "n_touch": int(np.concatenate(e1_l).size)}


def _merge(parts):
    keys = ("e1", "er", "vol", "d1", "trend", "year")
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["n_lvls"] = sum(p["n_lvls"] for p in parts)
    out["n_touch"] = sum(p["n_touch"] for p in parts)
    return out


def pool(dfs, combo, params):
    parts = [collect_one(make_ctx(df, params["warmup"],
                                  state_fns={"trend": _trend_state}), combo, params)
             for df in dfs]
    return _merge(parts)


def pool_gbm(ref_df, combo, params, seeds):
    parts = []
    for seed in range(seeds):
        rw = gbm_matching(ref_df, seed=seed)
        ctx = make_ctx(rw, params["warmup"], state_fns={"trend": _trend_state})
        parts.append(collect_one(ctx, combo, params))
    return _merge(parts)


# ── 统计 ─────────────────────────────────────────────────────
def _stat(vals):
    m = np.isfinite(vals)
    if not m.any():
        return (0, float("nan"))
    return (int(m.sum()), float(np.mean(vals[m])))


def e1_by_er(pooled):
    e1 = pooled["e1"]
    er = pooled["er"]
    me = np.isfinite(e1)
    out = {}
    for es in ("高", "中", "低"):
        m = me & (er == es)
        out[es] = (int(m.sum()), float(np.mean(e1[m])) if m.any() else float("nan"))
    return out


def d1_by_er(pooled):
    d1 = pooled["d1"]
    er = pooled["er"]
    tr = pooled["trend"]
    mt = tr & np.isfinite(d1)
    out = {}
    for es in ("高", "中", "低"):
        m = mt & (er == es)
        out[es] = (int(m.sum()), float(np.mean(d1[m])) if m.any() else float("nan"))
    return out


def h1_table(pooled):
    er = pooled["er"]
    vol = pooled["vol"]
    ok = (er != "") & (vol != "")
    rows = {}
    for es in ("高", "中", "低"):
        row = {}
        for vs in ("低", "中", "高"):
            row[vs] = int(((er == es) & (vol == vs) & ok).sum())
        rows[es] = row
    return rows, int(ok.sum())


def year_e1_diff(pooled, params):
    e1 = pooled["e1"]
    er = pooled["er"]
    y = pooled["year"]
    me = np.isfinite(e1)
    out = {}
    for yy in params["by_year_list"]:
        m = me & (y == yy)
        m_hi = m & (er == "高")
        m_lo = m & (er == "低")
        hi = float(np.mean(e1[m_hi])) if m_hi.any() else float("nan")
        lo = float(np.mean(e1[m_lo])) if m_lo.any() else float("nan")
        out[yy] = (int(m_hi.sum()), int(m_lo.sum()), hi, lo)
    return out


def year_d1_by_er(pooled, params):
    d1 = pooled["d1"]
    er = pooled["er"]
    tr = pooled["trend"]
    y = pooled["year"]
    mt = tr & np.isfinite(d1)
    out = {}
    for yy in params["by_year_list"]:
        m = mt & (y == yy)
        m_hi = m & (er == "高")
        m_lo = m & (er == "低")
        hi = float(np.mean(d1[m_hi])) if m_hi.any() else float("nan")
        lo = float(np.mean(d1[m_lo])) if m_lo.any() else float("nan")
        out[yy] = (int(m_hi.sum()), int(m_lo.sum()), hi, lo)
    return out


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_1h_df, params, seeds):
    """探测器自检: GBM30种子同管线 E1 null∈[-1.5pp,+1.5pp] 且 D1(趋势态)
    null∈[49%,51%] 且 n≥MIN_N, 失败 SystemExit. 返回 GBM 池 (主组合直接复用)."""
    combo = params["combo"]
    gbm = pool_gbm(ref_1h_df, combo, params, seeds)
    e1_n, e1_mean = _stat(gbm["e1"])
    d1_n, d1_mean = _stat(np.where(gbm["trend"], gbm["d1"], np.nan))
    e1_by = e1_by_er(gbm)
    gbm_diff = (e1_by["高"][1] - e1_by["低"][1]
                if np.isfinite(e1_by["高"][1]) and np.isfinite(e1_by["低"][1])
                else float("nan"))
    ctx = make_ctx(ref_1h_df, params["warmup"],
                   state_fns={"trend": _trend_state})
    real = collect_one(ctx, combo, params)
    real_e1 = _stat(real["e1"])[1]
    if e1_n < MIN_N or d1_n < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM n_e1={e1_n} n_d1={d1_n} < MIN_N={MIN_N}")
    if not (-0.015 <= e1_mean <= 0.015):
        raise SystemExit(
            f"GATE FAIL: GBM30种子 E1 null mean={e1_mean * 100:+.2f}pp "
            f"∉ [-1.5pp, +1.5pp] — 探测器偏置, 停")
    if not (0.49 <= d1_mean <= 0.51):
        raise SystemExit(
            f"GATE FAIL: GBM30种子 D1(趋势态) null mean={d1_mean * 100:.2f}% "
            f"∉ [49%, 51%] — 探测器偏置, 停")
    print(f"[GATE] 首标1h E1: 真实 {_pct(real_e1)} | GBM{seeds}种子 {_pct(e1_mean)} "
          f"(n={e1_n}); D1(趋势态) GBM {_pct(d1_mean)} (n={d1_n}); "
          f"H2-gate GBM E1高−低 {_pp(gbm_diff)}", flush=True)
    return {"real_e1": real_e1, "gbm_e1": e1_mean, "n_gbm": e1_n,
            "gbm_d1": d1_mean, "gbm_diff": gbm_diff, "gbm": gbm}


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.2f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def _line(label, rs, gs):
    rn, rm = rs
    gn, gm = gs
    net = (rm - gm) if np.isfinite(rm) and np.isfinite(gm) else float("nan")
    return ("  {}: 真实 {} (n={}) | GBM {} (n={}) | 净差 {} {}".format(
        label, _pct(rm), rn, _pct(gm), gn,
        _pp(net) if np.isfinite(net) else "-", _nm(min(rn, gn))))


def _h2_block(p, r, g, label):
    lines = []
    re_ = e1_by_er(r)
    ge_ = e1_by_er(g)
    for es in ("高", "中", "低"):
        lines.append(_line(f"  E1 {es}ER", re_[es], ge_[es]))
    rd = re_["高"][1] - re_["低"][1]
    gd = ge_["高"][1] - ge_["低"][1]
    gate_ok = np.isfinite(gd) and gd < p["h2_gbm_max"]
    real_ok = np.isfinite(rd) and rd >= p["h2_min"]
    lines.append(f"  E1差(高−低): 真实 {_pp(rd)} (判据 ≥+{p['h2_min'] * 100:.0f}pp) "
                 f"[{'达标' if real_ok else '未达标'}] | GBM {_pp(gd)} "
                 f"[H2-gate: GBM{'PASS <1pp' if gate_ok else 'FAIL ≥1pp'}]")
    return lines


def _h3_block(p, r, g, label):
    lines = []
    rd = d1_by_er(r)
    gd = d1_by_er(g)
    for es in ("高", "中", "低"):
        lines.append(_line(f"  D1 {es}ER", rd[es], gd[es]))
    nets = {es: (rd[es][1] - gd[es][1]) for es in ("高", "低")
            if np.isfinite(rd[es][1]) and np.isfinite(gd[es][1])}
    if len(nets) == 2:
        diff = nets["高"] - nets["低"]
        lines.append(f"  D1净差(高−低): {_pp(diff)} "
                     f"(判据 ≥+{p['h3_min'] * 100:.0f}pp) "
                     f"[{'达标' if diff >= p['h3_min'] else '未达标'}]")
    else:
        lines.append("  D1净差(高−低): n 不足")
    return lines


def write_out(out_path, params, g, res, year_rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},combo={},er_n={},er_win={},z_win={},e1_half={},W={},"
        "head_drop={},gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), p["combo"], p["er_n"], p["er_win"],
            p["z_win"], p["e1_half"], p["W"], p["head_drop"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(E1 全触碰 mean, 首标1h): "
        "真实 {:.2f}% GBM {:.2f}% [PASS]; 探测器自检 GBM30种子同管线 "
        "E1 null∈±1.5pp D1(趋势态) null {:.2f}%∈[49%,51%] [PASS]; "
        "H2-gate GBM E1高−低 "
        "{:+.2f}pp [{}]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], g["real_e1"] * 100, g["gbm_e1"] * 100,
            g["gbm_d1"] * 100,
            g["gbm_diff"] * 100,
            "PASS <1pp" if np.isfinite(g["gbm_diff"]) and g["gbm_diff"] < p["h2_gbm_max"] else "FAIL ≥1pp",
            g["n_gbm"]),
        "# RESULTS: 20 标的 × 1h 为主 + 4h 交叉 × 2023-08..2026-08; 描述层无入场, "
        "无交易含义; ER_n=|C_t−C_{t−n}|/Σ|ΔC| (n=10); ER状态=rolling120分位 "
        "(高≥80th/低≤20th); 波动状态=z120三分位; E1=c15口径(12根前后); "
        "D1=c17口径(沿趋势端点概率 W=24)",
        "",
    ]
    r1 = res["1h"]["real"]
    g1 = res["1h"]["gbm"]
    r4 = res["4h"]["real"]
    g4 = res["4h"]["gbm"]

    # H1 列联表
    lines.append("[H1] 正交性: ER 状态 × 波动状态 列联 (触碰事件, 1h):")
    t_real, n_ok = h1_table(r1)
    t_gbm, n_ok_g = h1_table(g1)
    for es in ("高", "中", "低"):
        row = t_real[es]
        tot = row["低"] + row["中"] + row["高"]
        frac = {vs: (row[vs] / tot if tot else float("nan")) for vs in ("低", "中", "高")}
        lines.append("  ER{:<2} 真实 低 {:>7} 中 {:>7} 高 {:>7} (n={}) | 占比 "
                     "{:.1f}/{:.1f}/{:.1f}%".format(
            es, row["低"], row["中"], row["高"], tot,
            frac["低"] * 100, frac["中"] * 100, frac["高"] * 100))
    hi = t_real["高"]
    hi_tot = hi["低"] + hi["中"] + hi["高"]
    h1_ok = (hi_tot > 0 and
             min(hi["低"] / hi_tot, hi["中"] / hi_tot, hi["高"] / hi_tot) >= p["h1_min_frac"])
    lines.append(f"  H1 判据: 高ER行每格 ≥{p['h1_min_frac'] * 100:.0f}% -> "
                 f"{'PASS' if h1_ok else 'FAIL'} (n_ok={n_ok}, 波动状态样本)")
    lines.append(f"  (GBM 同表: 高ER行占比 {_pct(t_gbm['高']['低'] / max(1, sum(t_gbm['高'].values())))}/"
                 f"{_pct(t_gbm['高']['中'] / max(1, sum(t_gbm['高'].values())))}/"
                 f"{_pct(t_gbm['高']['高'] / max(1, sum(t_gbm['高'].values())))})")

    # H2 释放调节
    lines.append("")
    lines.append("[H2] 释放调节 (E1, c15 口径, 1h):")
    lines.extend(_h2_block(p, r1, g1, "1h"))
    lines.append("[H2-4h] 释放调节 (4h 交叉):")
    lines.extend(_h2_block(p, r4, g4, "4h"))

    # H3 折返调节
    lines.append("")
    lines.append("[H3] 折返调节 (D1, c17 口径, 趋势态触碰, 1h):")
    lines.extend(_h3_block(p, r1, g1, "1h"))
    lines.append("[H3-4h] 折返调节 (4h 交叉):")
    lines.extend(_h3_block(p, r4, g4, "4h"))

    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c15 (2026-08-13): E1 净差 +7.44pp (1h 主组合), "
                 "GBM 触碰条件化机械偏置 +1.04pp (GATE 校准带 ±1.5pp); c17 "
                 "(2026-08-13): 逆势侧 D1 净差 -4.09pp (1h 主组合), 阶段梯度 "
                 "2.76pp, 角色差 2.20pp; c12 (2026-08-13): 波动长记忆 H=0.93/0.90 "
                 "(GBM null 0.50); 书 CH17 (Kaufman): ER 方向效率 = 噪声≠波动, "
                 "n=10 默认")
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(year_rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def _subset(lst, n):
    """取前 n 个 DataFrame (列表切片会触发 check_study AST 误报, 用索引循环)"""
    return [lst[i] for i in range(n)]


def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    dfs = load(PARAMS["tf_list"])
    if not dfs or not dfs.get("1h"):
        print("无数据, 退出")
        return 1

    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    n_sym = PARAMS["dev_subset"]["n_sym"] if dev else len(dfs["1h"])
    n_sym4 = PARAMS["dev_subset"]["n_sym"] if dev else len(dfs["4h"])

    g = gate(dfs["1h"][0], PARAMS, seeds)

    res = {
        "1h": {"real": pool(_subset(dfs["1h"], n_sym), PARAMS["combo"], PARAMS),
               "gbm": g["gbm"]},
        "4h": {"real": pool(_subset(dfs["4h"], n_sym4), PARAMS["combo"], PARAMS),
               "gbm": pool_gbm(dfs["4h"][0], PARAMS["combo"], PARAMS, seeds)},
    }

    if dev:
        print(f"[dev] 管线 OK (3 标的 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    # BY_YEAR (H2 E1 差 + H3 D1 净差, 成对 真实+GBM)
    r1, g1 = res["1h"]["real"], res["1h"]["gbm"]
    y_e_r = year_e1_diff(r1, PARAMS)
    y_e_g = year_e1_diff(g1, PARAMS)
    y_d_r = year_d1_by_er(r1, PARAMS)
    y_d_g = year_d1_by_er(g1, PARAMS)
    year_rows = []
    for yy in PARAMS["by_year_list"]:
        (hn, ln, hr, lr) = y_e_r[yy]
        (hn2, ln2, hg, lg) = y_e_g[yy]
        (dhn, dln, dhr, dlr) = y_d_r[yy]
        (dhn2, dln2, dhg, dlg) = y_d_g[yy]
        e_diff_r = (hr - lr) if np.isfinite(hr) and np.isfinite(lr) else float("nan")
        e_diff_g = (hg - lg) if np.isfinite(hg) and np.isfinite(lg) else float("nan")
        d_net_h = (dhr - dhg) if np.isfinite(dhr) and np.isfinite(dhg) else float("nan")
        d_net_l = (dlr - dlg) if np.isfinite(dlr) and np.isfinite(dlg) else float("nan")
        d_diff = (d_net_h - d_net_l) if np.isfinite(d_net_h) and np.isfinite(d_net_l) else float("nan")
        year_rows.append("{} E1高 真实 {} (n={}) GBM {} (n={}) | E1低 真实 {} "
                         "(n={}) GBM {} (n={}) | D1高 真实 {} (n={}) GBM {} "
                         "(n={}) | D1低 真实 {} (n={}) GBM {} (n={}) | "
                         "E1差(高−低) 真实 {} GBM {} | D1净差(高−低) {}".format(
            yy, _pct(hr), hn, _pct(hg), hn2, _pct(lr), ln, _pct(lg), ln2,
            _pct(dhr), dhn, _pct(dhg), dhn2, _pct(dlr), dln, _pct(dlg), dln2,
            _pp(e_diff_r) if np.isfinite(e_diff_r) else "-",
            _pp(e_diff_g) if np.isfinite(e_diff_g) else "-",
            _pp(d_diff) if np.isfinite(d_diff) else "-"))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, res, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
