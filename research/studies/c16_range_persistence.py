#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C16 区间延续性（因果区间定义重验）(2026-08-13, 无未来函数, 1h/4h)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画"触碰因果存活区间边界后,
  价格留在 [S,R] 双边界内的延续性"这一市场事实, 无入场, 无交易含义, 无任何
  方向/收益/成本结论。所有统计为事后描述; 若未来用作特征/条件, 必须经滚动
  口径重验。描述层发布门槛: 无胜率/期望要求, 但必须有 GBM 无信息对照与数字
  可溯源。

================================================================
研究问题 (预注册, 运行前冻结): 因果区间定义下, 触碰后留在 [S,R] 内
  (双边界) 真实−GBM ≥ +3pp 是否成立?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1: w=6 触碰后留存净差 (真实−GBM) ≥ +3pp (B3d 旧值为 +4.4~+9.1, 打折重验)
  H2: 全体区间 bar 留存净差 ≥ +5pp
  H3: 触碰后剩余存续 真实−GBM ≥ +1.5 根
  H4: 趋势分层反向概率净差 ≤ 0 (方向依旧无特异性)
  操作定义 (冻结, 与 B3d P1/P2/P4 同口径, 仅 alive 语义因果化):
    - 留存率 = 触碰/区间 bar 的 t 时刻后 w 根内 close 仍在 [S.price, R.price]
      内的根数占比 (双边界, 突破任一边即失留)
    - 剩余存续 = 触碰 bar 所在 alive 连续段的根数 (含触碰根)
    - 反向概率 (w=24): 支撑边界触碰 → P(close[t+24]>close[t]); 阻力边界
      触碰 → P(close[t+24]<close[t]); 按趋势状态分层 (B3d P4 同口径)

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。
  组合: 1h/4h × (min_touch=2, tol=0.3) / (3, 0.5), 共 4 组合; w ∈ {6,12,24};
  H4 反向窗口 W_REV=24; causal_confirmed 确认窗口 w=24 (与 level_breakdown
  一致, 默认 lag_lo=0, lag_hi=60 → conf∈[t-60,t-24])。

================================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口 (禁手动切片)
  atr              | make_ctx 内置 (market_phase ewm,       | bar 收盘后 | ctx.atr (ATR_PERIOD=14)
  cluster 位带     | levels.cluster_levels 在线聚类+冻结     | confirm_at | 冻结后 price/band/confirm_at
                   |   (pivot 按确认时序逐入组)             |            |   不可变 (levels R1/R2 快照)
  破位确认标签     | levels.level_breakdown (depth=0.5,      | 全样本     | 描述层事后标签; 仅经
                   |   w=24, hold=0.5) → confirmed           |            |   causal_confirmed 条件化
  存续条件化       | causal.causal_confirmed(conf, 24):      | conf 窗口  | research.causal (条件化唯一出口)
                   |   known[t]=1 ⟺ ∃conf∈[t-60,t-24]        | 已收盘闭合 |   默认 lag_lo=0, lag_hi=60
  触碰剔除         | [t-23,t] 内 confirmed 的触碰样本剔除     | 全样本事后 | causal_confirmed 语义: 刚确认/
                   |   (recent 掩码, cumsum 差分, 无切片)    |            |   突破样本不视为存活 (B2c/B2d
                   |                                       |            |   泄漏的干净替代)
  区间边界 S/R     | 逐 bar 最近活跃位: S=max{支撑价≤close},  | bar 收盘后 | confirm_at≤t 门控 (快照语义);
                   |   R=min{阻力价≥close}; 无宽度/LIFE 过滤 |            |   S≤close≤R 恒成立
  触碰事件         | 位带 intrabar 触及且前一根未触及 (entry) | bar 收盘后 | 触碰的位带 == 逐 bar 最近边界
                   |   ∩ t>=confirm_at, 仅计边界触碰        |            |   (B3d 定义更严格: 见设计偏离)
  留存度量         | lead 布尔掩码 (out[t]=arr[t+j], 无切片) | 全样本事后 | 描述层; 未来窗口为事后标签
  趋势状态         | state_features.state_series 语义的      | bar 收盘后 | 与 state_series 逐位对拍 (GATE
                   |   向量化复现 (rolling 特征 + classify)  |            |   等价性自检); 性能原因, 语义一致
  GBM 无信息对照   | sim_market.gbm_matching(ref_df, seed)   | 锚定真实   | 固定种子序列 0..29;
                   |   (索引/长度/σ 锚定真实)                |            |   首标×30 种子全管线
  分年             | ctx.years (截断坐标) 事后聚合           | 全样本     | 描述层 BY_YEAR (成对 真实+GBM)

设计偏离说明 (相对 B3d/B2c, 因果化与定义差异 — 结论对照时须考虑):
  - B3d 区间含宽度约束 (R−S ≤ 2.5×ATR) 与位带寿命 (LIFE<600); 本研究按任务
    定义: 区间 = 逐 bar 最近成对活跃位 (confirm_at≤t) + 双侧 causal_confirmed
    存续, 无宽度/LIFE 过滤。区间宽度分布以 [诊断] 行输出备查 (单位 ATR)。
  - B3d 的 alive 用 confirmed[c] (c<t) 直接判定 — 确认窗口 (c+24) 未闭合即
    泄漏; 本研究用 causal_confirmed (仅 conf∈[t-60,t-24] 计为已知), 且
    [t-23,t] 内 confirmed 的触碰样本剔除 (recent 掩码) 而非视为存活。
  - H4 趋势状态用 state_features.state_series 的语义 (向量化复现, GATE 逐位
    对拍一致); B3d P4 用 120 根收益方向 sign(log(c[t]/c[t-120])) — 分层变量
    不同, 方向结论只作形状参照。
  - 触碰样本限定为"区间边界触碰" (触碰位带 == 逐 bar 最近边界 S 或 R);
    B3d 计入区间内任意位带触碰 — 本研究样本定义更严格, n 偏小。
  - H1/H3 的触碰样本按 bar 计 (同一 bar 双侧触碰记 1 个样本); H4 按触碰侧
    计 (双侧触碰记 2 个样本, 各按自身反向方向)。
  - 趋势状态/ATR/ADX 的向量化复现为性能措施: _atr_series/_adx_series 与
    market_phase 逐字同式 (掩码改写, 无切片), trend_states_vec 与
    state_features.state_series 在 GATE 逐位对拍一致 (0 错位, 否则 SystemExit)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。

发布门槛自检 (描述层):
  - GATE 探测器: (1) 趋势状态向量化与 state_features.state_series 逐位一致
    (0 错位); (2) GBM 30 种子同管线 null: 留存率随 w 单调不升 (w6≥w12≥w24,
    touch 与 allbar 双侧), per-seed 触碰留存 w=6 全部有限且池均值 ∈ [0.25,0.75]
    合理带; (3) GBM 池 n ≥ MIN_N; 任一失败 SystemExit (违规即停)
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (同管线重放)
  - MIN_N 检查: 每个输出格含 n, 不足格标注 [MIN_N 不足] (全单元格输出)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义, 不涉及胜率/期望/成本 (描述层门槛)

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c16_range_persistence.py
  python3 research/studies/c16_range_persistence.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path (脚本以 `python3 research/studies/c16_range_persistence.py` 直接运行时,
# sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.causal import causal_confirmed
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels, level_breakdown
from research.sim_market import gbm_matching
from research.state_features import state_series
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "combos": ((2, 0.3), (3, 0.5)),      # (min_touch, tolerance_mult) — 预注册
    "w_list": (6, 12, 24),               # 留存窗口 (根) — 预注册
    "w_rev": 24,                         # H4 反向概率窗口 (根)
    "warmup": 600,                       # make_ctx 截断起点 (覆盖 atr/状态 warm-up)
    "brk_depth": 0.5,                    # level_breakdown depth (×ATR)
    "brk_w": 24,                         # level_breakdown 确认窗口
    "brk_hold": 0.5,                     # level_breakdown hold_ratio
    "allbar_stride": 5,                  # H2 全体区间 bar 采样步长 (B3d P3 同款)
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),  # 2023 为部分年, 不纳入
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c16_range_persistence"


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


# ── 趋势状态 (state_features.state_series 语义的向量化复现) ──
# _atr_series/_adx_series 与 market_phase 同式 (布尔掩码改写以规避 L3 禁切片,
# 已对拍 market_phase 逐位一致); trend_states_vec 与 state_series 在 GATE 对拍。
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


def trend_bucket(states):
    """趋势状态 → 4 桶 (up/down/range/transition) + unknown"""
    states = states.astype(str)
    b = np.full(len(states), "unknown", dtype=object)
    b[states == "range"] = "range"
    b[states == "transition"] = "transition"
    b[np.char.startswith(states, "trend_up")] = "up"
    b[np.char.startswith(states, "trend_down")] = "down"
    return b


def make_ctx_states(df, params):
    """make_ctx + 趋势状态 (state_fn 在截断 df 上计算, 长度 = n)"""
    return make_ctx(df, params["warmup"], state_fns={"trend": trend_states_vec})


# ── 布尔掩码辅助 (无切片: 禁 x[a:b], 只用掩码/整数索引) ──────
def lead_float(arr, j):
    """out[t] = arr[t+j] (t+j<n), 否则 NaN — 掩码实现, 无切片"""
    n = len(arr)
    out = np.full(n, np.nan)
    m = np.arange(n) < n - j
    out[m] = arr[np.arange(n)[m] + j]
    return out


def lag_bool(arr, j):
    """out[t] = arr[t-j] (t-j>=0), 否则 False — 掩码实现, 无切片"""
    arr = np.asarray(arr, bool)
    n = len(arr)
    out = np.zeros(n, bool)
    m = np.arange(n) >= j
    out[m] = arr[np.arange(n)[m] - j]
    return out


def recent_mask(confirmed, n, win):
    """recent[t] = ∃ confirmed[c], c∈[t-win+1, t] — cumsum 差分窗口标记, 无切片"""
    diff = np.zeros(n + 1, int)
    pos = np.flatnonzero(confirmed)
    pc = pos[pos < n]
    diff[pc] += 1
    ends = np.minimum(pc + win, n)
    diff[ends] -= 1
    return np.cumsum(diff)[:n] > 0


def run_len_forward(mask):
    """mask True 连续段自 t 起的长度 (含 t), False → 0 — flip 实现, 无切片"""
    n = len(mask)
    pos = np.arange(n)
    nxt = np.where(mask, n, pos)
    nxt = np.flip(np.minimum.accumulate(np.flip(nxt)))
    return nxt - pos


def interval_bounds(n, close, levels):
    """逐 bar 最近活跃位: S=max{支撑价≤close}, R=min{阻力价≥close} (confirm_at≤t).

    返回 (S, R, s_id, r_id) — 边界位带价格与 levels 索引 (-1 = 无)。
    """
    t = np.arange(n)
    S = np.full(n, np.nan)
    R = np.full(n, np.nan)
    s_id = np.full(n, -1, dtype=np.int64)
    r_id = np.full(n, -1, dtype=np.int64)
    for i, lv in enumerate(levels):
        m = t >= lv.confirm_at
        if lv.side == "support":
            m = m & (close >= lv.price)
            upd = np.isnan(S[m]) | (S[m] < lv.price)
            S[m] = np.where(upd, lv.price, S[m])
            s_id[m] = np.where(upd, i, s_id[m])
        else:
            m = m & (close <= lv.price)
            upd = np.isnan(R[m]) | (R[m] > lv.price)
            R[m] = np.where(upd, lv.price, R[m])
            r_id[m] = np.where(upd, i, r_id[m])
    return S, R, s_id, r_id


def persist_frac(m, close, S, R, w):
    """样本集 m 的留存率: 未来 w 根 close 仍在 [S,R] 内的占比 (样本处 t+w<n 已保证).

    返回长度 n 的数组, 样本处为留存率, 其余 NaN。
    """
    n = len(close)
    acc = np.zeros(n)
    for j in range(1, w + 1):
        cj = lead_float(close, j)
        acc = acc + ((cj >= S) & (cj <= R) & m)
    frac = np.full(n, np.nan)
    frac[m] = acc[m] / w
    return frac


def run_symbol(ctx, combo, params):
    """单标的完整管线: cluster_levels → 边界/存活 → 触碰样本 → 全部度量.

    返回 dict (池化用): 触碰样本 (ts_*) 与全体区间 bar 样本 (ab_*) 的值数组
    及宽度/位带诊断。
    """
    mt, tol = combo
    n = ctx.n
    t = np.arange(n)
    atr = ctx.atr
    close = ctx.close
    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K,
                          tolerance_mult=tol, min_touch=mt)
    S, R, s_id, r_id = interval_bounds(n, close, lvls)

    # alive (因果存续): 双侧边界 confirm_at≤t 且 conf∈[t-60,t-24] 无 confirmed
    not_known_s = np.ones(n, bool)
    not_known_r = np.ones(n, bool)
    # [t-23,t] 内 confirmed → 触碰样本剔除 (recent 掩码)
    not_recent_s = np.ones(n, bool)
    not_recent_r = np.ones(n, bool)
    for i, lv in enumerate(lvls):
        att, conf, outside, ratio = level_breakdown(
            lv, close, atr, params["brk_depth"], params["brk_w"],
            params["brk_hold"])
        if conf.sum() > 0:
            known, _ = causal_confirmed(conf, params["brk_w"])
        else:
            known = np.zeros(n, bool)
        recent = recent_mask(conf, n, params["brk_w"])
        if lv.side == "support":
            m = s_id == i
            not_known_s[m] = not_known_s[m] & ~known[m]
            not_recent_s[m] = not_recent_s[m] & ~recent[m]
        else:
            m = r_id == i
            not_known_r[m] = not_known_r[m] & ~known[m]
            not_recent_r[m] = not_recent_r[m] & ~recent[m]
    has_int = np.isfinite(S) & np.isfinite(R)
    alive = has_int & not_known_s & not_known_r

    # 触碰事件 (intrabar 触及位带且前一根未触及, confirm_at≤t 门控; 仅边界触碰)
    touch_s = np.zeros(n, bool)
    touch_r = np.zeros(n, bool)
    for i, lv in enumerate(lvls):
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        tm = (ctx.low <= p_hi) & (ctx.high >= p_lo) & (t >= lv.confirm_at)
        entry = tm & ~lag_bool(tm, 1)
        if lv.side == "support":
            touch_s = touch_s | (entry & (s_id == i))
        else:
            touch_r = touch_r | (entry & (r_id == i))

    # 样本集 (描述层样本; tail 保证 w=24 窗口完整)
    tail = t < n - params["w_rev"]
    valid = alive & tail & not_recent_s & not_recent_r
    m_ts = touch_s & valid
    m_tr = touch_r & valid
    m_t = m_ts | m_tr                        # H1/H3: 按 bar 计
    stride = (t % params["allbar_stride"]) == 0
    m_ab = alive & tail & stride             # H2: 全体区间 bar (步长 5)

    # 留存度量
    ts_persist = {w: persist_frac(m_t, close, S, R, w) for w in params["w_list"]}
    ab_persist = {w: persist_frac(m_ab, close, S, R, w) for w in params["w_list"]}

    # H3: 剩余存续 (alive 连续段长度, 含触碰根)
    rl = run_len_forward(alive)

    # H4: 反向概率 (w=24; 支撑触碰 → P(close[t+24]>close[t]),
    #               阻力触碰 → P(close[t+24]<close[t]))
    c24 = lead_float(close, params["w_rev"])
    rev = np.full(n, np.nan)
    rev[m_ts] = (c24[m_ts] > close[m_ts])
    rev[m_tr] = (c24[m_tr] < close[m_tr])

    # 趋势状态桶 (触碰样本)
    bucket = trend_bucket(ctx.states["trend"])

    # 区间宽度诊断 (alive 段, ATR 归一)
    width = (R - S) / np.maximum(atr, 1e-12)
    w_mask = alive & tail
    if w_mask.any():
        w_mean = float(np.mean(width[w_mask]))
        w_med = float(np.median(width[w_mask]))
    else:
        w_mean = float("nan")
        w_med = float("nan")

    out = {
        "ts_year": ctx.years[m_t],
        "ts_state": bucket[m_t],
        "ts_remain": rl[m_t],
        "ts_rev": rev[m_t],
        "ab_year": ctx.years[m_ab],
        "w_mean": w_mean,
        "w_med": w_med,
        "n_lvls": len(lvls),
    }
    for w in params["w_list"]:
        out["ts_p%d" % w] = ts_persist[w][m_t]
        out["ab_p%d" % w] = ab_persist[w][m_ab]
    return out


def merge(parts, params):
    """多标的/多种子结果池化 (数组拼接 + 标量求和)"""
    if not parts:
        return None
    out = {}
    for k in ("ts_year", "ts_state", "ts_remain", "ts_rev", "ab_year"):
        out[k] = np.concatenate([p[k] for p in parts])
    for w in params["w_list"]:
        out["ts_p%d" % w] = np.concatenate([p["ts_p%d" % w] for p in parts])
        out["ab_p%d" % w] = np.concatenate([p["ab_p%d" % w] for p in parts])
    out["n_lvls"] = sum(p["n_lvls"] for p in parts)
    out["n_ts"] = int(sum(p["ts_year"].size for p in parts))
    out["n_ab"] = int(sum(p["ab_year"].size for p in parts))
    out["w_mean"] = np.nanmean([p["w_mean"] for p in parts])
    out["w_med"] = np.nanmean([p["w_med"] for p in parts])
    return out


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_1h, params):
    """探测器自检 + GBM 30 种子同管线 null (1h 主组合 (2,0.3)), 失败 SystemExit.

    自检项:
      (1) trend_states_vec 与 state_features.state_series 逐位一致 (0 错位)
      (2) GBM 30 种子同管线 (首标 1h, 主组合): 触碰/全体留存率随 w 单调不升
          (w6≥w12≥w24), per-seed 触碰留存 w=6 全部有限且池均值 ∈ [0.25,0.75]
      (3) GBM 池 n ≥ MIN_N
    返回 GBM 池 (主组合 GBM 侧直接复用) + 真实首标结果 (主组合真实侧复用)。
    """
    # (1) 趋势状态等价性 (mask 截断, 禁切片)
    keep = np.arange(len(ref_1h)) >= params["warmup"]
    trunc = ref_1h.iloc[keep]
    st_ref, _ = state_series(trunc)
    st_vec = trend_states_vec(trunc)
    n_mis = int((st_ref != st_vec).sum())
    print(f"[GATE] 趋势状态向量化 vs state_series: 错位 {n_mis} / {len(st_ref)}",
          flush=True)
    if n_mis != 0:
        raise SystemExit(
            f"GATE FAIL: 趋势状态向量化与 state_series 不一致 (错位 {n_mis}) — 停")

    # (2) GBM 30 种子全管线 (主组合)
    combo = params["combos"][0]
    parts = []
    per_seed = []
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_1h, seed=seed)
        ctx = make_ctx_states(rw, params)
        r = run_symbol(ctx, combo, params)
        parts.append(r)
        per_seed.append(float(np.mean(r["ts_p6"])) if r["ts_p6"].size
                        else float("nan"))
    pool = merge(parts, params)
    per_seed = np.array(per_seed)
    gt6 = float(np.mean(pool["ts_p6"]))
    gt12 = float(np.mean(pool["ts_p12"]))
    gt24 = float(np.mean(pool["ts_p24"]))
    ga6 = float(np.mean(pool["ab_p6"]))
    ga12 = float(np.mean(pool["ab_p12"]))
    ga24 = float(np.mean(pool["ab_p24"]))
    print(f"[GATE] GBM30种子 触碰留存 w6/w12/w24: "
          f"{gt6 * 100:.1f}% / {gt12 * 100:.1f}% / {gt24 * 100:.1f}% | "
          f"全体 {ga6 * 100:.1f}% / {ga12 * 100:.1f}% / {ga24 * 100:.1f}%",
          flush=True)
    if not np.all(np.isfinite(per_seed)):
        raise SystemExit("GATE FAIL: GBM per-seed 触碰留存 w=6 含 NaN — 管线错位, 停")
    if not (gt6 >= gt12 >= gt24):
        raise SystemExit("GATE FAIL: GBM 触碰留存不随 w 单调不升 — 窗口错位, 停")
    if not (ga6 >= ga12 >= ga24):
        raise SystemExit("GATE FAIL: GBM 全体留存不随 w 单调不升 — 窗口错位, 停")
    if not (0.25 <= gt6 <= 0.75):
        raise SystemExit(f"GATE FAIL: GBM 触碰留存 w=6 池均值 {gt6:.1%} 越出 "
                         f"[25%, 75%] — 区间装配异常, 停")
    if pool["n_ts"] < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM n_ts={pool['n_ts']} < MIN_N={MIN_N}, 停")

    # 真实首标 (无条件基线)
    ctx_real = make_ctx_states(ref_1h, params)
    real_first = run_symbol(ctx_real, combo, params)
    real_a6 = float(np.mean(real_first["ab_p6"]))
    print(f"[GATE] 无条件基线 首标1h主组合 allbar-w6: 真实 {real_a6 * 100:.1f}% "
          f"| GBM30种子 {ga6 * 100:.1f}% (n={pool['n_ts']})", flush=True)
    return {
        "gbm_pool": pool,
        "real_first": real_first,
        "per_seed": per_seed,
        "gt": (gt6, gt12, gt24),
        "ga": (ga6, ga12, ga24),
        "real_a6": real_a6,
        "n_mis": n_mis,
    }


# ── 池化 ─────────────────────────────────────────────────────
def pool_gbm(ref_df, combo, params):
    """GBM 对照池 — 首标 × gbm_seeds 种子, 逐种子同管线重放"""
    parts = []
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_df, seed=seed)
        ctx = make_ctx_states(rw, params)
        parts.append(run_symbol(ctx, combo, params))
    return merge(parts, params)


# ── 统计/格式化 (全部数字带 n, MIN_N 标注) ──────────────────
def nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def pct(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:.1f}%"


def pp(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:+.1f}pp"


def num(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.1f}"


def fmt_combo_key(tf, combo):
    return f"{tf} (min_touch={combo[0]}, tol={combo[1]})"


def state_stats(ts_state, ts_rev):
    """趋势状态桶 → {桶: (n, mean 反向概率)}"""
    out = {}
    for bk in ("up", "down", "range", "transition", "unknown"):
        m = ts_state == bk
        out[bk] = (int(m.sum()),
                   float(np.mean(ts_rev[m])) if m.any() else float("nan"))
    return out


def by_year_lines(results, params):
    """BY_YEAR 成对 (真实 全标的 + GBM 首标30种子): 主度量逐年份"""
    rows = []
    for tf in params["tf_list"]:
        for combo in params["combos"]:
            rs = results[(tf, combo)]["real"]
            gs = results[(tf, combo)]["gbm"]
            for y in params["by_year_list"]:
                mr = rs["ts_year"] == y
                mg = gs["ts_year"] == y
                abr = rs["ab_year"] == y
                abg = gs["ab_year"] == y
                touch = (float(np.mean(rs["ts_p6"][mr])) if mr.any()
                         else float("nan"),
                         float(np.mean(gs["ts_p6"][mg])) if mg.any()
                         else float("nan"))
                allbar = (float(np.mean(rs["ab_p6"][abr])) if abr.any()
                          else float("nan"),
                          float(np.mean(gs["ab_p6"][abg])) if abg.any()
                          else float("nan"))
                remain = (float(np.mean(rs["ts_remain"][mr])) if mr.any()
                          else float("nan"),
                          float(np.mean(gs["ts_remain"][mg])) if mg.any()
                          else float("nan"))
                rev = (float(np.mean(rs["ts_rev"][mr])) if mr.any()
                       else float("nan"),
                       float(np.mean(gs["ts_rev"][mg])) if mg.any()
                       else float("nan"))
                rows.append(
                    "{} ({} ,{}) {} touch-w6 真实{} GBM{} | allbar-w6 "
                    "真实{} GBM{} | remain 真实{} GBM{} | rev 真实{} GBM{}".format(
                        tf, combo[0], combo[1], y,
                        pct(touch[0]), pct(touch[1]),
                        pct(allbar[0]), pct(allbar[1]),
                        num(remain[0]), num(remain[1]),
                        pct(rev[0]), pct(rev[1])))
    return rows


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, results, year_rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},combos={},w_list={},w_rev={},warmup={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], ",".join(p["tf_list"]), p["combos"],
            p["w_list"], p["w_rev"], p["warmup"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(首标1h主组合 allbar-w6 留存): "
        "真实 {:.1f}% GBM {:.1f}% [PASS]; 探测器: 状态对拍错位 0 [PASS], "
        "GBM 触碰留存 w6/w12/w24 {:.1f}%/{:.1f}%/{:.1f}% 单调不升 [PASS], "
        "GBM 全体留存 w6/w12/w24 {:.1f}%/{:.1f}%/{:.1f}% 单调不升 [PASS], "
        "per-seed 触碰w6 mean {:.1f}% [min {:.1f}%, max {:.1f}%] [PASS]; "
        "MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], g["real_a6"] * 100, g["ga"][0] * 100,
            g["gt"][0] * 100, g["gt"][1] * 100, g["gt"][2] * 100,
            g["ga"][0] * 100, g["ga"][1] * 100, g["ga"][2] * 100,
            float(np.mean(g["per_seed"])) * 100,
            float(np.min(g["per_seed"])) * 100,
            float(np.max(g["per_seed"])) * 100,
            g["gbm_pool"]["n_ts"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 描述层无入场, 无交易含义; "
        "留存率 = 样本 bar 后 w 根 close ∈ [S,R] 占比 (双边界); "
        "GBM = 首标×30 种子同管线; 触碰样本 = 因果存活区间边界触碰 ([t-23,t] 内 "
        "confirmed 的样本剔除)",
        "",
    ]
    for tf in p["tf_list"]:
        for combo in p["combos"]:
            r = results[(tf, combo)]
            rs, gs = r["real"], r["gbm"]
            key = fmt_combo_key(tf, combo)
            lines.append(f"[组合] {key} — 位带: 真实 {rs['n_lvls']} | "
                         f"GBM {gs['n_lvls']} (n_sym=20, GBM 首标×30种子)")
            for w in p["w_list"]:
                rv = rs["ts_p%d" % w]
                gv = gs["ts_p%d" % w]
                rn, gn = int(rv.size), int(gv.size)
                rm = float(np.mean(rv)) if rn else float("nan")
                gm = float(np.mean(gv)) if gn else float("nan")
                lines.append(
                    "  [H1] 触碰留存 w={}: 真实 {} (n={}) | GBM {} (n={}) | "
                    "净差 {} {}".format(w, pct(rm), rn, pct(gm), gn,
                                        pp(rm - gm) if rn else "-", nm(rn)))
            for w in p["w_list"]:
                rv = rs["ab_p%d" % w]
                gv = gs["ab_p%d" % w]
                rn, gn = int(rv.size), int(gv.size)
                rm = float(np.mean(rv)) if rn else float("nan")
                gm = float(np.mean(gv)) if gn else float("nan")
                lines.append(
                    "  [H2] 全体区间bar留存 w={}: 真实 {} (n={}) | GBM {} "
                    "(n={}) | 净差 {} {}".format(
                        w, pct(rm), rn, pct(gm), gn,
                        pp(rm - gm) if rn else "-", nm(rn)))
            rr = rs["ts_remain"]
            gr = gs["ts_remain"]
            rn, gn = int(rr.size), int(gr.size)
            rmean = float(np.mean(rr)) if rn else float("nan")
            gmean = float(np.mean(gr)) if gn else float("nan")
            rmed = float(np.median(rr)) if rn else float("nan")
            gmed = float(np.median(gr)) if gn else float("nan")
            lines.append(
                "  [H3] 剩余存续: 真实 mean {} 中位 {} (n={}) | GBM mean {} "
                "中位 {} (n={}) | Δmean {} 根 {}".format(
                    num(rmean), num(rmed), rn, num(gmean), num(gmed), gn,
                    num(rmean - gmean) if rn else "-", nm(rn)))
            rss = state_stats(rs["ts_state"], rs["ts_rev"])
            gss = state_stats(gs["ts_state"], gs["ts_rev"])
            rn_all = int(rs["ts_rev"].size)
            gn_all = int(gs["ts_rev"].size)
            rrev = float(np.mean(rs["ts_rev"])) if rn_all else float("nan")
            grev = float(np.mean(gs["ts_rev"])) if gn_all else float("nan")
            lines.append(
                "  [H4] 反向概率合计 w=24: 真实 {} (n={}) | GBM {} (n={}) | "
                "净差 {}".format(pct(rrev), rn_all, pct(grev), gn_all,
                                 pp(rrev - grev) if rn_all else "-"))
            for bk in ("up", "down", "range", "transition", "unknown"):
                rn, rv = rss[bk]
                gn, gv = gss[bk]
                net = (rv - gv) if (rn and np.isfinite(rv)
                                    and np.isfinite(gv)) else float("nan")
                lines.append(
                    "  [H4-{}] 反向概率 w=24: 真实 {} (n={}) | GBM {} (n={}) "
                    "| 净差 {} {}".format(bk, pct(rv), rn, pct(gv), gn,
                                          pp(net), nm(rn)))
            lines.append(
                "  [诊断] 区间宽 (ATR): 真实 mean {} 中位 {} | GBM mean {} "
                "中位 {}".format(num(rs["w_mean"]), num(rs["w_med"]),
                                 num(gs["w_mean"]), num(gs["w_med"])))
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(year_rows))
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

    # GATE 自检 (失败 SystemExit — 违规即停)
    g = gate(dfs["1h"][0], PARAMS)

    # 4 组合: 真实池 (全标的) + GBM 池 (首标×30 种子; 主组合复用 gate 池)
    results = {}
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            print(f"... 真实 {tf} {combo}", flush=True)
            parts = []
            for k, df in enumerate(dfs[tf]):
                if (tf, combo) == ("1h", PARAMS["combos"][0]) and k == 0:
                    parts.append(g["real_first"])
                else:
                    parts.append(run_symbol(
                        make_ctx_states(df, PARAMS), combo, PARAMS))
            real = merge(parts, PARAMS)
            if (tf, combo) == ("1h", PARAMS["combos"][0]):
                gbm = g["gbm_pool"]
            else:
                print(f"... GBM {tf} {combo}", flush=True)
                gbm = pool_gbm(dfs[tf][0], combo, PARAMS)
            results[(tf, combo)] = {"real": real, "gbm": gbm}

    year_rows = by_year_lines(results, PARAMS)
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, results, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
