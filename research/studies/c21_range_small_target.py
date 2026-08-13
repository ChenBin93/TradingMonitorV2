#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C21 区间触碰小目标 + limit 入场 (B4/B4e 因果重做) (2026-08-13, 无未来函数, 1h/4h)

条件层 c2x: 只报告"条件 1:1/非对称胜率与期望"的结构事实, 结论语言按 10 项发布
门槛裁决, 不达标一律写"未达发布门槛", 不主张任何 edge。入场/成本/胜率为描述
结构, 不含 live 可交易性主张。

============================================================
研究问题 (预注册, 运行前冻结): 因果区间内触下沿, limit 挂单入场 + 小目标/宽
止损非对称, 是否有真实−GBM 正净差, 且收盘入场版无净差 (追价伪影复现)?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1 (主端点): 区间内 (因果成对活跃位+双侧 causal_confirmed, c16 口径) 触
     下沿, limit buy @S.price (low≤S 触及成交), 目标 0.3×ATR / 止损 0.7×ATR,
     W=6, 胜率差 ≥ +3pp 且期望差 ≥ +0.05R (真实−GBM 同管线)
  H2 (伪影复现): 同条件收盘入场版 (evaluate_forward 非对称 t_target=0.3/
     t_stop=0.7) 净差 ≈0 (±1pp 内) — B4e 的"净差=追价伪影"因果复验
  H3 (空头侧): 只报净差 (GBM 路径偏置教训), 不做门槛裁决
  H4 (成本): 成本模型 (maker 0.02%×2 + 滑点 0.5bp) 后期望 > 0
  H5 (c24 集成): 触碰前 z120=低波动子集对主端点胜率差增量 ≥ +1.5pp; GBM
     同过滤无同向增量
  c15 教训 (报告义务): 触碰条件化在 GBM 上约 +1pp 机械偏置 — 净差口径必须
     扣除并如实报告 GBM 侧绝对水平

参数网格预注册: T∈{0.3,0.5}、S∈{0.7,1.0}、W∈{6,12}、(mt,tol)∈{(2,0.3),
(3,0.5)}; 开发/验证集分离: 只在 1h(2,0.3) 开发集上选参 (选优规则预注册:
按主端点期望差 [真实−GBM E[R]] 最大化), 其余 3 组合为验证集;
HOLDOUT=末 3 月 (2026-06~08) 参数冻结后一次评估只报方向。

操作定义 (冻结):
  - 区间 [S,R]: 逐 bar 最近活跃位 (confirm_at≤t, levels R1/R2 快照语义) +
    双侧 causal_confirmed 存续 (conf∈[t-60,t-24]; [t-23,t] 内 confirmed 的
    触碰样本剔除, recent 掩码) — c16 同款
  - 触下沿事件: 触碰位带 == 逐 bar 最近支撑 S (intrabar 触及, 段首首根,
    t>=confirm_at), close[t]∈[S,R] 恒成立 (interval_bounds 构造), 存活区间内
  - limit 入场: simulate_limit_entries (官方引擎): 挂单 bar t 起第一根
    low≤entry_px 成交, 成交价=挂单价; 判定自成交 bar 下一根 open 起,
    W 根内先碰者 (win/loss/expired/skip); 从未成交 → unfilled (不计入)
  - target/stop 为 per-event 数组: 目标=挂单价+T×ATR[t], 止损=挂单价−S×ATR[t]
    (ATR[t] = 挂单 bar 已收盘 ATR, 入场时已知)
  - 期望 E[R] 单位 = ATR: 每笔成交交易 R ∈ {+T (win), −S (loss), 0
    (expired/skip)}; E[R] = (T×n_win − S×n_loss)/n_filled; unfilled 无持仓
    不计入; "0.05R" = 0.05 ATR
  - 胜率 = n_win/(n_win+n_loss) (成交且命中, 与官方引擎 win_rate 语义一致)
  - H2 收盘版: 同事件集, 入场=触碰 bar 收盘 close[t], tp/sl 自 close[t] 起
    (evaluate_forward t_target/t_stop 非对称口径)
  - H5: z120 低波动 = atr[t] ≤ rolling_percentile(atr,120,1/3)[t] (因果尾窗);
    增量 = (低波动子集 真实−GBM 胜率差) − (全体 真实−GBM 胜率差) ≥ +1.5pp;
    GBM 同过滤增量 = GBM低波动 − GBM全体 ≤ 0 (无同向增量)
  - 选参: 开发集 1h(2,0.3) 8 个 (T,S,W) 格按 ΔE = E_real − E_gbm 最大化;
    冻结后验证 3 组合; HOLDOUT 用冻结参数在 2026-06..08 窗口一次评估

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/数据        | 计算方式                         | 可用时点   | 依据
  close/high/low/  | research.ctx.make_ctx 统一截断   | bar 收盘后 | ctx 唯一对齐出口
  open/atr/years   |  (内部 iloc[warmup:])            |            | (禁一切手动切片)
  月份 (HOLDOUT)   | df.index.month, keep 掩码对齐    | bar 收盘后 | 布尔掩码截断
                   |  (arange>=warmup 掩码, 无切片)   |            | (禁 iloc 切片)
  聚类位带         | levels.cluster_levels 在线聚类    | confirm_at | 冻结后 price/band/
                   |  +冻结 (k=K, tolerance_mult,     |            | confirm_at 不可变
                   |   min_touch)                     |            | (levels R1/R2)
  破位确认标签     | levels.level_breakdown (depth=0.5,| 全样本     | 条件层事后标签;
                   |  w=24, hold=0.5) → confirmed      |            | 仅经 causal_confirmed
  存续条件化       | causal.causal_confirmed(conf,24)  | conf 窗口  | research.causal (条件
                   |  known[t]=1 ⟺ ∃conf∈[t-60,t-24]   | 已收盘闭合 | 化唯一出口)
  触碰剔除         | [t-23,t] 内 confirmed 的触碰样本  | 全样本事后 | causal_confirmed 语义
                   |  剔除 (recent 掩码, 无切片)       |            | (B2c/B2d 泄漏替代)
  区间边界 S/R     | 逐 bar 最近活跃位 (confirm_at≤t)  | bar 收盘后 | S≤close≤R 恒成立
  触碰事件         | intrabar 重叠位带 ∩ 段首首根 ∩   | bar 收盘后 | 纯触碰, 不依赖
                   |  t>=confirm_at ∩ 存活区间         |            | confirmed 标签
  z120 低波动      | causal.rolling_percentile(atr,   | 尾窗已收盘 | research.causal
                   |  120, 1/3) → atr≤rp33            |            | (禁全样本分位)
  limit 成交+判定  | research.limit_sim.simulate_     | 已收盘 bar | 官方引擎 (注册)
                   |  limit_entries (per-event 数组)  |            |
  收盘入场对照     | research.outcome.evaluate_forward | 已收盘 bar | 官方引擎 (非对称
                   |  (t_target=T, t_stop=S)          |            | t_target/t_stop)
  GBM 无信息对照   | sim_market.gbm_matching(ref_df,   | 锚定真实   | 固定种子序列 0..29
                   |  seed) 首标 × 30 种子同管线重放   |            | (MIN_GBM_SEEDS)
  分年/HOLDOUT     | ctx.years + 月份掩码 事后聚合     | 全样本     | 成对输出

============================================================
数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。

设计偏离说明 (相对 B4/B4e, 因果化与定义差异 — 结论对照时须考虑):
  - B4/B4e 自写 _sim 与全样本分位 (sq 层) 已弃用; 入场/判定一律官方引擎,
    H5 波动过滤走 rolling_percentile。
  - B4e 的"区间内"结构用 b3_general_levels 的 confirmed (未来标签泄漏);
    本研究区间 = 双侧 causal_confirmed 因果存续 + [t-23,t] 内突破样本剔除。
  - E[R] 单位明确定义为 ATR (win=+T, loss=−S, expired/skip=0); B4e 的
    "期望"为 ±1 单位, 数值不可直接对照, 方向可参照。
  - 成本模型按预注册: maker 0.02%×2 + 滑点 0.5bp = 0.045% of notional,
    每笔成交交易成本 = entry_px × 0.00045 / ATR[t] (ATR 单位); 简化假设
    maker 双边 (stop 触发的市价单实际为 taker, 滑点 0.5bp 近似吸收);
    unfilled 无成本。
  - simulate_limit_entries 的成交搜索无时限 (至数据末尾); 判定窗口 W 自
    成交 bar 下一根起算 (官方引擎语义)。
  - 事件 tail 统一 t < n − max(W)=12 (保证 W 窗口完整); 成交在尾部的
    极端情况由引擎 n_truncated/expired 语义处理。

发布门槛自检 (条件层 10 项, 结论逐项填表):
  ① 真实−RW(30 种子) 超预注册下限 (H1: ΔWR≥+3pp 且 ΔE≥+0.05R)
  ② 分年 ≥2/3 为正、最差年 ≥−2pp
  ③ 每格 n≥MIN_N
  ④ GATE 条件组无偏 (GBM 侧绝对水平如实报告)
  ⑤ 跨周期+跨参数一致 (开发集选参 → 3 验证组合)
  ⑥ HOLDOUT 末 3 月参数冻结一次评估方向不变
  ⑦ Holm 校正后仍显著 (开发集 8 格 ΔE)
  ⑧ 成本核算后 >0 (H4)
  ⑨ 结论↔.out↔脚本三重一致 (sha256)
  ⑩ 负结果/未达标格全部记录

运行命令:
  # 两道门禁: 引擎门禁 → 脚本门禁 → 运行
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c21_range_small_target.py
  python3 research/studies/c21_range_small_target.py
"""
import hashlib
import os
import sys
import time
from datetime import date
from math import erf, sqrt

# 仓库根入 path (脚本以 `python3 research/studies/c21_range_small_target.py` 直接
# 运行时, sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.causal import causal_confirmed, rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels, level_breakdown
from research.limit_sim import simulate_limit_entries
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "combos": ((2, 0.3), (3, 0.5)),       # (min_touch, tolerance_mult)
    "t_list": (0.3, 0.5),                 # 目标 ×ATR
    "s_list": (0.7, 1.0),                 # 止损 ×ATR
    "w_list": (6, 12),                    # 判定窗口 (根)
    "main": {"T": 0.3, "S": 0.7, "W": 6},  # 预注册主端点
    "dev": ("1h", (2, 0.3)),              # 开发集 (选参唯一组合)
    "warmup": 600,                        # make_ctx 截断起点
    "brk_depth": 0.5,                     # level_breakdown depth (×ATR)
    "brk_w": 24,                          # level_breakdown 确认窗口
    "brk_hold": 0.5,                      # level_breakdown hold_ratio
    "z_window": 120,                      # H5 z120 窗口
    "z_q": 1.0 / 3.0,                     # H5 低波动 = ≤33 分位
    "cost_maker": 0.0002,                 # maker 费率 (单边)
    "cost_slip": 0.00005,                 # 滑点 0.5bp (总量)
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),
    "holdout": {"year": 2026, "months": (6, 7, 8)},  # 末 3 月
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c21_range_small_target"

# 逐事件结果码 (对齐 simulate_limit_entries / evaluate_forward 的 outcome 字符串)
RCODE = {"win": 1, "loss": -1, "expired": 2, "skip": 3, "unfilled": 0}


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


# ── 布尔掩码辅助 (无切片: 禁 x[a:b], 只用掩码/整数索引) ──────
def lag_bool(arr, j):
    """out[t] = arr[t-j] (t-j>=0), 否则 False"""
    arr = np.asarray(arr, bool)
    n = len(arr)
    out = np.zeros(n, bool)
    m = np.arange(n) >= j
    out[m] = arr[np.arange(n)[m] - j]
    return out


def recent_mask(confirmed, n, win):
    """recent[t] = ∃ confirmed[c], c∈[t-win+1, t] — cumsum 差分, 无切片"""
    diff = np.zeros(n + 1, int)
    pos = np.flatnonzero(confirmed)
    pc = pos[pos < n]
    diff[pc] += 1
    ends = np.minimum(pc + win, n)
    diff[ends] -= 1
    return np.cumsum(diff)[:n] > 0


def interval_bounds(n, close, levels):
    """逐 bar 最近活跃位: S=max{支撑价≤close}, R=min{阻力价≥close} (confirm_at≤t)"""
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


# ── 单标的管线 ───────────────────────────────────────────────
def run_symbol(ctx, months, combo, params):
    """cluster → 区间/存活 → 触碰事件 → 逐事件 meta (无切片)

    返回 dict: ev_s/ev_r (触碰 bar), atr_s/px_s/clo_s/yrs_s/mon_s/zlo_s (逐
    事件), touch_s (布尔掩码, 收盘版用), n_lvls。
    """
    mt, tol = combo
    n = ctx.n
    t = np.arange(n)
    atr, close = ctx.atr, ctx.close
    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K,
                          tolerance_mult=tol, min_touch=mt)
    S, R, s_id, r_id = interval_bounds(n, close, lvls)

    # 存活 (因果存续): 双侧 confirm_at≤t 且 conf∈[t-60,t-24] 无 confirmed
    nk_s = np.ones(n, bool)
    nk_r = np.ones(n, bool)
    # [t-23,t] 内 confirmed → 触碰样本剔除 (recent 掩码)
    nr_s = np.ones(n, bool)
    nr_r = np.ones(n, bool)
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
            nk_s[m] = nk_s[m] & ~known[m]
            nr_s[m] = nr_s[m] & ~recent[m]
        else:
            m = r_id == i
            nk_r[m] = nk_r[m] & ~known[m]
            nr_r[m] = nr_r[m] & ~recent[m]
    alive = np.isfinite(S) & np.isfinite(R) & nk_s & nk_r

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

    # 事件窗口 (tail 保证 max(W) 完整)
    tail = t < n - max(params["w_list"])
    valid = alive & tail & nr_s & nr_r
    ts = touch_s & valid
    tr = touch_r & valid
    ev_s = np.flatnonzero(ts)
    ev_r = np.flatnonzero(tr)

    # z120 低波动 (因果尾窗)
    rp = rolling_percentile(atr, params["z_window"], params["z_q"],
                            min_periods=60)
    zlo = (atr <= rp) & ~np.isnan(rp)

    out = {
        "touch_s": ts,
        "ev_s": ev_s,
        "ev_r": ev_r,
        "atr_s": atr[ev_s],
        "px_s": S[ev_s],
        "clo_s": close[ev_s],
        "yrs_s": ctx.years[ev_s],
        "mon_s": months[ev_s],
        "zlo_s": zlo[ev_s],
        "atr_r": atr[ev_r],
        "px_r": R[ev_r],
        "clo_r": close[ev_r],
        "yrs_r": ctx.years[ev_r],
        "mon_r": months[ev_r],
        "zlo_r": zlo[ev_r],
        "n_lvls": len(lvls),
    }
    return out


# ── 官方引擎逐事件码 ─────────────────────────────────────────
def lim_long_codes(r, ctx, T, S_, W, params):
    """触下沿 limit buy @S.price → 逐事件结果码 (对齐 ev_s)"""
    ev = r["ev_s"]
    if len(ev) == 0:
        return np.array([], int)
    out, recs = simulate_limit_entries(
        ev, ctx.open, ctx.high, ctx.low, r["px_s"],
        r["px_s"] + T * r["atr_s"], r["px_s"] - S_ * r["atr_s"], W)
    return np.array([RCODE[x.outcome] for x in recs], int)


def lim_short_codes(r, ctx, T, S_, W, params):
    """触上沿 limit sell @R.price → 逐事件结果码 (对齐 ev_r)"""
    ev = r["ev_r"]
    if len(ev) == 0:
        return np.array([], int)
    out, recs = simulate_limit_entries(
        ev, ctx.open, ctx.high, ctx.low, r["px_r"],
        r["px_r"] - T * r["atr_r"], r["px_r"] + S_ * r["atr_r"], W)
    return np.array([RCODE[x.outcome] for x in recs], int)


def close_long_codes(r, ctx, T, S_, W, params):
    """收盘入场版 (H2): 触下沿事件收盘 close[t] 入场, 非对称 t_target/t_stop"""
    out, recs = evaluate_forward(
        ctx.close, ctx.high, ctx.low, ctx.atr, r["touch_s"],
        direction="long", t_target=T, t_stop=S_, w=W, open_px=ctx.open)
    return np.array([RCODE[x.outcome] for x in recs], int)


def build_part(r, ctx, T, S_, W, params, version):
    """单标的单 (T,S,W,version) 格 → {"codes","yrs","mon","zlo","atr","px"}"""
    if version == "lim_long":
        codes = lim_long_codes(r, ctx, T, S_, W, params)
        return {"codes": codes, "yrs": r["yrs_s"], "mon": r["mon_s"],
                "zlo": r["zlo_s"], "atr": r["atr_s"], "px": r["px_s"]}
    if version == "lim_short":
        codes = lim_short_codes(r, ctx, T, S_, W, params)
        return {"codes": codes, "yrs": r["yrs_r"], "mon": r["mon_r"],
                "zlo": r["zlo_r"], "atr": r["atr_r"], "px": r["px_r"]}
    codes = close_long_codes(r, ctx, T, S_, W, params)
    return {"codes": codes, "yrs": r["yrs_s"], "mon": r["mon_s"],
            "zlo": r["zlo_s"], "atr": r["atr_s"], "px": r["clo_s"]}


# ── 池化/统计 ────────────────────────────────────────────────
def pool_cell(parts, T, S_, params, exclude_holdout=False):
    """多标的/多种子汇总 → stats dict (含分年/HOLDOUT/H5 子集/成本/R 值)

    exclude_holdout=True: 剔除 2026-06..08 事件后统计 (选参用, 防 HOLDOUT
    污染开发集选优); ho 字段始终基于全量事件集计算。
    """
    codes = np.concatenate([p["codes"] for p in parts])
    yrs = np.concatenate([p["yrs"] for p in parts])
    mon = np.concatenate([p["mon"] for p in parts])
    zlo = np.concatenate([p["zlo"] for p in parts])
    atr = np.concatenate([p["atr"] for p in parts])
    px = np.concatenate([p["px"] for p in parts])
    n_ev_all = len(codes)
    if n_ev_all == 0:
        return None
    # HOLDOUT (末 3 月) — 始终基于全量事件
    m_ho = (yrs == params["holdout"]["year"]) & \
        (mon >= params["holdout"]["months"][0]) & \
        (mon <= params["holdout"]["months"][1])
    ho = [int((m_ho & (codes == 1)).sum()),
          int((m_ho & (codes == -1)).sum())]
    if exclude_holdout:
        keep = ~m_ho
        codes = codes[keep]
        yrs = yrs[keep]
        mon = mon[keep]
        zlo = zlo[keep]
        atr = atr[keep]
        px = px[keep]
    cost_pct = 2 * params["cost_maker"] + params["cost_slip"]
    nw = int((codes == 1).sum())
    nl = int((codes == -1).sum())
    ne = int((codes == 2).sum())
    ns = int((codes == 3).sum())
    nu = int((codes == 0).sum())
    n_filled = nw + nl + ne + ns
    wr = nw / (nw + nl) if nw + nl else float("nan")
    e = (T * nw - S_ * nl) / n_filled if n_filled else float("nan")
    m_filled = codes != 0
    cost = (float(np.sum(px[m_filled] * cost_pct / atr[m_filled])) / n_filled
            if n_filled else float("nan"))
    e_cost = e - cost if np.isfinite(e) else float("nan")
    # 逐事件 R (ATR 单位, expired/skip=0) — Holm/SE 用
    rvals = np.where(codes == 1, T, np.where(codes == -1, -S_, 0.0))
    # 分年 (win/loss)
    year_wl = {}
    for yy in params["by_year_list"]:
        year_wl[yy] = [int(((yrs == yy) & (codes == 1)).sum()),
                       int(((yrs == yy) & (codes == -1)).sum())]
    # H5 低波动子集 (成交事件内)
    n_lo = int((zlo & (codes != 0)).sum())
    win_lo = int((zlo & (codes == 1)).sum())
    loss_lo = int((zlo & (codes == -1)).sum())
    wr_lo = (win_lo / (win_lo + loss_lo)
             if (win_lo + loss_lo) else float("nan"))
    return {
        "n_ev": len(codes), "n_filled": n_filled, "nw": nw, "nl": nl,
        "ne": ne, "ns": ns, "nu": nu, "wr": wr, "e": e, "cost": cost,
        "e_cost": e_cost, "rvals": rvals, "year_wl": year_wl, "ho": ho,
        "n_lo": n_lo, "wr_lo": wr_lo,
    }


# ── 统计/格式化 ─────────────────────────────────────────────
def wr_of(nw, nl):
    return float("nan") if nw + nl == 0 else nw / (nw + nl)


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


def combo_key(combo):
    return f"({combo[0]}, {combo[1]})"


def z_norm_p(z):
    """单侧正态 p (H1 方向: ΔE>0) — math.erf 实现, 无 scipy 依赖"""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def holm_adjust(pvals):
    """Holm 校正 (step-down): 排序升序 → q(i)=max_{j≤i} p(j)×(m−j+1),
    与输入同序返回校正 p。running 从 0 起 (step-down 单调约束)。"""
    order = sorted(range(len(pvals)), key=lambda k: pvals[k])
    m = len(pvals)
    out = [1.0] * len(pvals)
    running = 0.0
    for j, k in enumerate(order):
        v = min(1.0, pvals[k] * (m - j))
        running = max(running, v)
        out[k] = running
    return out


def d_e_stats(real, gbm):
    """ΔE 与 SE/z/p (两样本, 单侧) — 逐事件 R 值"""
    rv = real["rvals"]
    gv = gbm["rvals"]
    if len(rv) == 0 or len(gv) == 0:
        return None
    de = float(np.mean(rv) - np.mean(gv))
    se = sqrt(float(np.var(rv) / len(rv) + np.var(gv) / len(gv)))
    if se <= 0:
        return None
    return de, se, de / se, 1.0 - z_norm_p(de / se)


def de_str(s):
    if s is None:
        return ""
    de, se, z, p = s
    return " (z {:+.2f}, p {:.3f})".format(z, p)


# ── GATE 自检 (违规即停) ─────────────────────────────────────
def gate_dev(real_cells, gbm_cells, params):
    """开发组合 (1h(2,0.3)) 网格已算 — 从主端点 (0.3,0.7,6) 单元格断言, 无
    重复聚类. 任一失败 SystemExit:
      (1) GBM 主端点 WR ∈ [55%, 95%]  (limit 管线装配 sanity)
      (2) GBM 成交率 ≥ 50%
      (3) GBM close-long 版 WR ∈ [55%, 85%]  (H2 探测器; 盈亏平衡 70%)
      (4) GBM n ≥ MIN_N; gbm_seeds ≥ 30 (池规模在 pool_combo 保证)
    """
    r_main = real_cells[(0.3, 0.7, 6, "lim_long")]
    g_main = gbm_cells[(0.3, 0.7, 6, "lim_long")]
    g_close = gbm_cells[(0.3, 0.7, 6, "close_long")]
    gbm_wr = g_main["wr"]
    n_filled = g_main["n_filled"]
    fill_rate = (n_filled / g_main["n_ev"]
                 if g_main["n_ev"] else float("nan"))
    print(f"[GATE] GBM30种子 主端点(0.3,0.7,6): lim-long WR {gbm_wr * 100:.2f}% "
          f"(n={n_filled}, fill {fill_rate * 100:.1f}%) | close-long "
          f"WR {g_close['wr'] * 100:.2f}% (n={g_close['n_filled']})",
          flush=True)
    if not (0.55 <= gbm_wr <= 0.95):
        raise SystemExit(f"GATE FAIL: GBM lim-long WR {gbm_wr:.3f} ∉ [55%, 95%] "
                         f"— limit 管线装配异常, 停")
    if fill_rate < 0.5:
        raise SystemExit(f"GATE FAIL: GBM 成交率 {fill_rate:.2f} < 50% — 管线异常, 停")
    if not (0.55 <= g_close["wr"] <= 0.85):
        raise SystemExit(f"GATE FAIL: GBM close-long WR {g_close['wr']:.3f} "
                         f"∉ [55%, 85%] — H2 管线异常, 停")
    if n_filled < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM n={n_filled} < MIN_N={MIN_N}, 停")
    print(f"[GATE] 无条件基线(主端点 lim-long 1h(2,0.3)): 真实 "
          f"{r_main['wr'] * 100:.2f}% | GBM {gbm_wr * 100:.2f}%",
          flush=True)
    return {"real_first": r_main, "gbm_main": g_main, "gbm_close": g_close,
            "fill_rate": fill_rate}


# ── 主端点网格池 ─────────────────────────────────────────────
def pool_combo(ctxs, key, t_list, s_list, w_list, params):
    """单 (tf,combo) 组合全部 (T,S,W,version) 格池化.

    key = (tf, combo); ctxs[key] = [(ctx, months)]. 每个 symbol 只 run_symbol
    一次 (cluster 是主耗时), 全部格复用该 symbol 的事件与 meta。
    返回 (cells, parts): cells[(T,S,W,version)] = stats;
    parts[(T,S,W,version)] = 逐 symbol part 列表 (选参/HOLDOUT 复用, 免重算)。
    """
    syms = [run_symbol(ctx, mon, key[1], params) for ctx, mon in ctxs[key]]
    cells = {}
    parts = {}
    for T in t_list:
        for S_ in s_list:
            for W in w_list:
                for version in ("lim_long", "lim_short", "close_long"):
                    pp = []
                    for k, (ctx, _mon) in enumerate(ctxs[key]):
                        pp.append(build_part(syms[k], ctx, T, S_, W, params,
                                             version))
                    parts[(T, S_, W, version)] = pp
                    cells[(T, S_, W, version)] = pool_cell(pp, T, S_, params)
    return cells, parts


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR/HOLDOUT) ────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def fmt_grid_row(p, r, g, T, S_, W, params, label):
    rn = r["n_filled"] if r else 0
    gn = g["n_filled"] if g else 0
    rw = r["wr"] if r else float("nan")
    gw = g["wr"] if g else float("nan")
    re = r["e"] if r else float("nan")
    ge = g["e"] if g else float("nan")
    net_wr = (rw - gw) if np.isfinite(rw) and np.isfinite(gw) else float("nan")
    net_e = (re - ge) if np.isfinite(re) and np.isfinite(ge) else float("nan")
    return "{} T={} S={} W={}: 真实 {} (n={}) E{} | GBM {} (n={}) E{} | " \
        "ΔWR {} ΔE {}{}".format(
            label, T, S_, W, _pct(rw), rn, _e(re), _pct(gw), gn, _e(ge),
            _pp(net_wr), _e(net_e), _nm(rn))


def write_out(out_path, params, g, real_grid, gbm_grid, main_rows,
              select_rows, holm_rows, h4_rows, h5_rows, year_rows,
              holdout_rows, holm_main, holm_sel):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},combos={},t_list={},s_list={},w_list={},main={},dev={},"
        "warmup={},gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], ",".join(p["tf_list"]), p["combos"],
            p["t_list"], p["s_list"], p["w_list"], p["main"], p["dev"],
            p["warmup"], p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(主端点 lim-long 1h(2,0.3)): 真实 {:.2f}% "
        "GBM {:.2f}% [limit]; 探测器: GBM主端点WR∈[55%,95%] [PASS], GBM成交率 "
        "{:.1f}% [PASS], GBM close版WR∈[55%,85%] [PASS]; MIN_N n_gbm={} [PASS]"
        .format(p["gbm_seeds"], g["real_first"]["wr"] * 100,
                g["gbm_main"]["wr"] * 100, g["fill_rate"] * 100,
                g["gbm_main"]["n_filled"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 条件层 c2x; "
        "主端点 = 因果区间内触下沿 limit buy @S.price (simulate_limit_entries), "
        "target=T×ATR[t], stop=S×ATR[t], W 判定窗口; E[R] 单位=ATR "
        "(win=+T, loss=−S, expired/skip=0, unfilled 不计); GBM = 首标×30 种子 "
        "同管线; 开发集 1h(2,0.3) [DEV], 其余 3 组合验证 [VAL]",
        "",
        "[门槛] H1: ΔWR≥+3pp 且 ΔE≥+0.05R | H2: |ΔWR|≤1pp | "
        "H5: 增量≥+1.5pp 且 GBM 无同向增量 | H4: 成本后期望>0",
        "",
    ]

    # H1 主端点 (0.3,0.7,6) lim-long 4 组合 (真实 vs GBM)
    lines.append("[H1主端点] lim-long (T=0.3, S=0.7, W=6) 4 组合 "
                 "(真实全标的 vs GBM首标×30种子):")
    lines.extend(main_rows)
    lines.append("")

    # H1 网格 (lim-long 8 格 × 4 组合)
    lines.append("[H1网格] lim-long 全部 8 (T,S,W) 格 × 4 组合 (真实 vs GBM):")
    for tf in p["tf_list"]:
        for combo in p["combos"]:
            tag = "[DEV]" if (tf, combo) == p["dev"] else "[VAL]"
            for T in p["t_list"]:
                for S_ in p["s_list"]:
                    for W in p["w_list"]:
                        r = real_grid[(tf, combo, T, S_, W, "lim_long")]
                        g_ = gbm_grid[(tf, combo, T, S_, W, "lim_long")]
                        lines.append("  " + fmt_grid_row(
                            p, r, g_, T, S_, W, p,
                            "{} {}{} ".format(tf, combo_key(combo), tag)))
    lines.append("")

    # H2 收盘版
    lines.append("[H2] close-long (收盘入场, t_target/t_stop 非对称) 8 格 × 4 组合:")
    for tf in p["tf_list"]:
        for combo in p["combos"]:
            tag = "[DEV]" if (tf, combo) == p["dev"] else "[VAL]"
            for T in p["t_list"]:
                for S_ in p["s_list"]:
                    for W in p["w_list"]:
                        r = real_grid[(tf, combo, T, S_, W, "close_long")]
                        g_ = gbm_grid[(tf, combo, T, S_, W, "close_long")]
                        lines.append("  " + fmt_grid_row(
                            p, r, g_, T, S_, W, p,
                            "{} {}{} ".format(tf, combo_key(combo), tag)))
    lines.append("")

    # H3 空头
    lines.append("[H3] lim-short (触上沿 limit sell @R.price, 只报净差):")
    for tf in p["tf_list"]:
        for combo in p["combos"]:
            tag = "[DEV]" if (tf, combo) == p["dev"] else "[VAL]"
            for T in p["t_list"]:
                for S_ in p["s_list"]:
                    for W in p["w_list"]:
                        r = real_grid[(tf, combo, T, S_, W, "lim_short")]
                        g_ = gbm_grid[(tf, combo, T, S_, W, "lim_short")]
                        lines.append("  " + fmt_grid_row(
                            p, r, g_, T, S_, W, p,
                            "{} {}{} ".format(tf, combo_key(combo), tag)))
    lines.append("")

    # 选参/验证
    lines.append("[选参] 开发集 1h(2,0.3) lim-long 8 格 ΔE 排序 (选优规则: 期望差最大化):")
    lines.extend(select_rows)
    lines.append("")
    lines.append("[Holm] 开发集 8 格 ΔE 单侧 p + Holm 校正:")
    lines.extend(holm_rows)
    lines.append("  主端点 (0.3,0.7,6): {}".format(holm_main))
    lines.append("  选中格: {}".format(holm_sel))
    lines.append("")

    # H4 成本
    lines.append("[H4] 成本 (maker 0.02%×2 + 滑点 0.5bp = 0.045%):")
    lines.extend(h4_rows)
    lines.append("")

    # H5 波动过滤
    lines.append("[H5] z120 低波动过滤 (主端点 lim-long):")
    lines.extend(h5_rows)
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(year_rows))
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

    # 构建 ctx 池: 真实 (全标的) + GBM (首标×30 种子), 逐 (tf, combo)
    real_ctxs = {}
    gbm_ctxs = {}
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            key = (tf, combo)
            real_ctxs[key] = [
                (make_ctx(df, PARAMS["warmup"], state_fns={}),
                 months_aligned(df, PARAMS["warmup"])) for df in dfs[tf]]
            ref = dfs[tf][0]
            gbm_ctxs[key] = [
                (make_ctx(gbm_matching(ref, seed=s), PARAMS["warmup"],
                          state_fns={}),
                 months_aligned(gbm_matching(ref, seed=s), PARAMS["warmup"]))
                for s in range(PARAMS["gbm_seeds"])]

    # 各 (tf,combo) 网格池 (cells + parts); 开发组合先算 → GATE 复用
    dev_key = PARAMS["dev"]
    all_keys = [(tf, combo) for tf in PARAMS["tf_list"] for combo in PARAMS["combos"]]
    real_cells = {}
    gbm_cells = {}
    real_parts = {}
    gbm_parts = {}
    for key in ([dev_key] + [k for k in all_keys if k != dev_key]):
        rc, rp = pool_combo(real_ctxs, key, PARAMS["t_list"],
                            PARAMS["s_list"], PARAMS["w_list"], PARAMS)
        gc, gp = pool_combo(gbm_ctxs, key, PARAMS["t_list"],
                            PARAMS["s_list"], PARAMS["w_list"], PARAMS)
        for k2, v in rc.items():
            real_cells[key + k2] = v
        for k2, v in gc.items():
            gbm_cells[key + k2] = v
        real_parts[key] = rp
        gbm_parts[key] = gp
        if key == dev_key:
            # GATE 自检 (失败 SystemExit — 违规即停; 复用开发组合网格, 无重复聚类)
            g = gate_dev(rc, gc, PARAMS)

    # H1 主端点 4 组合行
    main_rows = []
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            r = real_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            g_ = gbm_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            is_dev = (tf, combo) == PARAMS["dev"]
            tag = "[DEV]" if is_dev else "[VAL]"
            net_wr = (r["wr"] - g_["wr"]) if np.isfinite(r["wr"]) and np.isfinite(g_["wr"]) else float("nan")
            net_e = (r["e"] - g_["e"]) if np.isfinite(r["e"]) and np.isfinite(g_["e"]) else float("nan")
            main_rows.append(
                "  {} {}{}: 真实 {} (n={}) E{} | GBM {} (n={}) E{} | "
                "ΔWR {} ΔE {} E_cost 真实 {} {}".format(
                    tf, combo_key(combo), tag, _pct(r["wr"]), r["n_filled"],
                    _e(r["e"]), _pct(g_["wr"]), g_["n_filled"], _e(g_["e"]),
                    _pp(net_wr), _e(net_e), _e(r["e_cost"]),
                    _nm(r["n_filled"])))

    # 选参: 开发集 lim-long 8 格按 ΔE 最大化 (剔除 HOLDOUT 窗口 — 防污染)
    dev_sel = []
    for T in PARAMS["t_list"]:
        for S_ in PARAMS["s_list"]:
            for W in PARAMS["w_list"]:
                r = pool_cell(real_parts[PARAMS["dev"]][(T, S_, W, "lim_long")],
                              T, S_, PARAMS, exclude_holdout=True)
                g_ = pool_cell(gbm_parts[PARAMS["dev"]][(T, S_, W, "lim_long")],
                               T, S_, PARAMS, exclude_holdout=True)
                s = d_e_stats(r, g_)
                net_e = (r["e"] - g_["e"]) if np.isfinite(r["e"]) and np.isfinite(g_["e"]) else float("nan")
                dev_sel.append((T, S_, W, r["e"], g_["e"], net_e, s))
    dev_sel.sort(key=lambda x: -x[5] if np.isfinite(x[5]) else -1e9)
    sel_T, sel_S, sel_W = dev_sel[0][0], dev_sel[0][1], dev_sel[0][2]
    select_rows = []
    for k, (T, S_, W, re_, ge_, net_e, s) in enumerate(dev_sel):
        mark = " ←选中" if k == 0 else ""
        select_rows.append(
            "  {}. T={} S={} W={}: 真实E {} GBM E {} ΔE {}{}{}".format(
                k + 1, T, S_, W, _e(re_), _e(ge_), _e(net_e),
                de_str(s) if s else "", mark))
    select_rows.append(f"  选中参数: T={sel_T}, S={sel_S}, W={sel_W}")

    # Holm (开发集 8 格 ΔE, 剔除 HOLDOUT)
    pvals = []
    for T in PARAMS["t_list"]:
        for S_ in PARAMS["s_list"]:
            for W in PARAMS["w_list"]:
                r = pool_cell(real_parts[PARAMS["dev"]][(T, S_, W, "lim_long")],
                              T, S_, PARAMS, exclude_holdout=True)
                g_ = pool_cell(gbm_parts[PARAMS["dev"]][(T, S_, W, "lim_long")],
                               T, S_, PARAMS, exclude_holdout=True)
                s = d_e_stats(r, g_)
                pvals.append(s[3] if s else 1.0)
    adj = holm_adjust(pvals)
    holm_rows = []
    idx = 0
    holm_main = holm_sel = "-"
    for T in PARAMS["t_list"]:
        for S_ in PARAMS["s_list"]:
            for W in PARAMS["w_list"]:
                r = pool_cell(real_parts[PARAMS["dev"]][(T, S_, W, "lim_long")],
                              T, S_, PARAMS, exclude_holdout=True)
                g_ = pool_cell(gbm_parts[PARAMS["dev"]][(T, S_, W, "lim_long")],
                               T, S_, PARAMS, exclude_holdout=True)
                s = d_e_stats(r, g_)
                p = s[3] if s else 1.0
                holm_rows.append("  T={} S={} W={}: ΔE {} p {:.3f} p_adj {:.3f}".format(
                    T, S_, W, _e(s[0]) if s else "-", p, adj[idx]))
                if (T, S_, W) == (0.3, 0.7, 6):
                    holm_main = "ΔE {} p {:.3f} p_adj {:.3f}".format(
                        _e(s[0]) if s else "-", p, adj[idx])
                if (T, S_, W) == (sel_T, sel_S, sel_W):
                    holm_sel = "ΔE {} p {:.3f} p_adj {:.3f}".format(
                        _e(s[0]) if s else "-", p, adj[idx])
                idx += 1

    # H4 成本 (主端点 4 组合)
    h4_rows = []
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            r = real_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            g_ = gbm_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            net_e_cost = (r["e_cost"] - g_["e_cost"]) if np.isfinite(r["e_cost"]) and np.isfinite(g_["e_cost"]) else float("nan")
            h4_rows.append(
                "  {} {}{}: 真实 E {} cost {} E_cost {} | GBM E_cost {} | "
                "净差E_cost {}".format(
                    tf, combo_key(combo),
                    "[DEV]" if (tf, combo) == PARAMS["dev"] else "[VAL]",
                    _e(r["e"]), _e(r["cost"]), _e(r["e_cost"]),
                    _e(g_["e_cost"]), _e(net_e_cost)))

    # H5 波动过滤 (主端点 lim-long 4 组合)
    h5_rows = []
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            r = real_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            g_ = gbm_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            rn_all, gn_all = r["n_filled"], g_["n_filled"]
            r_inc = (r["wr_lo"] - r["wr"]) if np.isfinite(r["wr_lo"]) and np.isfinite(r["wr"]) else float("nan")
            g_inc = (g_["wr_lo"] - g_["wr"]) if np.isfinite(g_["wr_lo"]) and np.isfinite(g_["wr"]) else float("nan")
            dh5 = ((r["wr_lo"] - g_["wr_lo"]) - (r["wr"] - g_["wr"])
                   if np.isfinite(r["wr_lo"]) and np.isfinite(g_["wr_lo"])
                   and np.isfinite(r["wr"]) and np.isfinite(g_["wr"])
                   else float("nan"))
            h5_rows.append(
                "  {} {}{}: 真实 全体 {} (n={}) / 低波动 {} (n={}) | GBM 全体 "
                "{} (n={}) / 低波动 {} (n={}) | 真实增量 {} GBM增量 {} | "
                "ΔH5 {}".format(
                    tf, combo_key(combo),
                    "[DEV]" if (tf, combo) == PARAMS["dev"] else "[VAL]",
                    _pct(r["wr"]), rn_all, _pct(r["wr_lo"]), r["n_lo"],
                    _pct(g_["wr"]), gn_all, _pct(g_["wr_lo"]), g_["n_lo"],
                    _pp(r_inc), _pp(g_inc), _pp(dh5)))

    # BY_YEAR (主端点 lim-long)
    year_rows = []
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            r = real_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            g_ = gbm_cells[(tf, combo, 0.3, 0.7, 6, "lim_long")]
            for y in PARAMS["by_year_list"]:
                rw_ = wr_of(r["year_wl"][y][0], r["year_wl"][y][1])
                gw_ = wr_of(g_["year_wl"][y][0], g_["year_wl"][y][1])
                rn = r["year_wl"][y][0] + r["year_wl"][y][1]
                gn = g_["year_wl"][y][0] + g_["year_wl"][y][1]
                if rn == 0 and gn == 0:
                    continue
                year_rows.append(
                    "{} {}{} {} 真实 {} (n={}) GBM {} (n={})".format(
                        tf, combo_key(combo),
                        "[DEV]" if (tf, combo) == PARAMS["dev"] else "[VAL]",
                        y, _pct(rw_), rn, _pct(gw_), gn))

    # HOLDOUT (选中参数, 2026-06..08; 只报方向)
    holdout_rows = []
    r_sel = pool_cell(real_parts[PARAMS["dev"]][(sel_T, sel_S, sel_W,
                                                "lim_long")],
                      sel_T, sel_S, PARAMS)
    g_sel = pool_cell(gbm_parts[PARAMS["dev"]][(sel_T, sel_S, sel_W,
                                               "lim_long")],
                      sel_T, sel_S, PARAMS)
    rw_ = wr_of(r_sel["ho"][0], r_sel["ho"][1])
    gw_ = wr_of(g_sel["ho"][0], g_sel["ho"][1])
    rn = r_sel["ho"][0] + r_sel["ho"][1]
    gn = g_sel["ho"][0] + g_sel["ho"][1]
    holdout_rows.append(
        "选中参数 T={} S={} W={} 2026-06..08: 真实 {} (n={}) GBM {} (n={}) "
        "| 净差WR {} (只报方向: {})".format(
            sel_T, sel_S, sel_W, _pct(rw_), rn, _pct(gw_), gn,
            _pp(rw_ - gw_) if np.isfinite(rw_) and np.isfinite(gw_) else "-",
            "正" if (np.isfinite(rw_) and np.isfinite(gw_) and rw_ > gw_)
            else "负/不可判"))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, real_cells, gbm_cells, main_rows,
              select_rows, holm_rows, h4_rows, h5_rows, year_rows,
              holdout_rows, holm_main, holm_sel)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

