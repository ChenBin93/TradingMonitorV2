#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C23 趋势逆势折返条件化 (2026-08-13, 无未来函数, 1h 主 + 4h 净差, 条件层 c2x)

[CONDITIONAL] 分区: 本研究为条件层 (c2x) — 有入场 (触碰 bar 收盘 market)、
  有胜率/期望/成本。结论语言必须按 PLAN §4 阶段 2 的 10 项发布门槛逐项裁决,
  不达标一律写"未达发布门槛"。定位声明: 效应基线预期仅 2-4pp, 本研究确认
  "结构发现"是否存在; 成本后 ≤0 只能写"结构发现"非 edge, 禁止交易主张。

============================================================
研究问题 (预注册, 运行前冻结): 趋势态触位后逆势入场 (涨触阻空/跌触撑多) 的
  1:1 胜率是否有真实−GBM 正净差, 成本后是否仍为正?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1: 对称 1:1 (evaluate_forward, T=1.0, W=24), 入场=触碰 bar 收盘 (market),
      1h, 胜率差 (真实−GBM 同管线) ≥ +3pp
  H2: 非对称 t_target=1.0/t_stop=0.7 (官方口径) 同条件胜率差 (探索, 不定门槛)
  H3: 阶段分层 — early 净差 ≈0、accel/late 净差更强 (c17 H2 复验于 1:1 口径);
      角色分层 — 未破角色净差 > 刚破角色 (c17 H3 复验)
  H4: 触碰前 z120=低波动 (滚动分位最低三分位) 子集的胜率差增量 ≥ +1.5pp
      (相对全体触碰); GBM 上同过滤无同向增量 (门禁)

  操作化 (运行前锁定):
    - 主度量: 逆势侧触碰 (涨触阻→short, 跌触撑→long) 入场=触碰 bar 收盘,
      对称 1:1 (T=1.0, W=24), 胜率 = n_win/(n_win+n_loss) (expired/skip/截断
      不计入, 官方引擎口径)
    - H1 判据: 1h 主组合 (2,0.3) 胜率差 (真实−GBM 同管线) ≥ +3pp
    - H3 判据: 阶段/角色分层净差方向与 c17 一致 (early 最弱≈0, 未破>刚破),
      数值报告不设硬门槛 (方向复验)
    - H4 判据: (低波动子集净差 − 全体净差) ≥ +1.5pp, 且 GBM 侧同增量 < +1.5pp
    - 漂移分解 (c18 教训, 报告义务): 净差=真实−GBM; 另报 真实−真实无条件
      (同方向 1:1 全 bar 基线), 标明触碰/状态条件化的增量
    - 成本模型 (预注册): taker 0.05%×2 + 滑点 1bp + funding 0.01%×3 (1h W=24
      跨 3 个 8h 周期) = 0.14% notional; cost_ATR = c_total/mean(ATR/close);
      E_R = T×(2WR−1) − cost_ATR; 盈亏平衡 WR* = (1+cost_ATR)/2; 成本后
      E_R>0 且净 E_R>0 才可称达标; 否则只写"结构发现"
    - 预注册声明: 若 H1 过但成本后 ≤0, 结论只能写"结构发现"非 edge

============================================================
无未来函数设计说明 (逐特征信息边界表, 沿用 c17 因果实现):
  特征/事件       | 计算方式                              | 可用时点   | 依据
  close/high/low  | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口 (禁手动切片)
  趋势状态/阶段   | state_features.state_series 因果状态机  | bar 收盘后 | state_fns 截断后计算 (不变性测试)
  cluster 位带    | levels.cluster_levels 在线聚类+冻结     | confirm_at | R1/R2 快照语义
  触碰事件        | bar 区间触及 [price±band] ∩ t≥confirm_at| bar 收盘后 | 纯触碰事件, 每段连续触碰首根
  z120 低波动     | causal.rolling_percentile(ATR,120,1/3):| bar 收盘后 | research.causal (禁全样本分位)
                  |   低波动 = ATR[t] ≤ rp33[t]           |            |
  角色层          | causal.causal_confirmed(confirmed, w=  | t 时刻已知 | causal 唯一条件化出口
                  |   24, lag_lo=0, lag_hi=60): 刚破/未破, |            |   (B5c 泄漏修复)
                  |   [t-23,t] 突破样本剔除               |            |
  入场/胜负判定   | outcome.evaluate_forward (close[i] 入场| 事后标签   | 官方引擎 (长度断言自动保护,
                  |   = 触碰 bar 收盘; i+1 起前向, open 出发|            |   open 出发语义, 对称+非对称)
                  |   语义; expired/skip 不计入胜率)      |            |
  成本            | 预注册常数: taker 0.05%×2 + 滑点 1bp +| 事后       | 预注册; ATR/close 在已评估
                  |   funding 0.01%×3 = 0.14% notional    |            |   入场 (win/loss) 上取均值
  GBM 无信息对照  | sim_market.gbm_matching 30 种子首标    | 锚定真实   | 固定种子 0..29 同管线
  holdout         | 末 3 月 (2026-06-01 起), 参数冻结后    | 全样本     | 门槛 ⑥ (只报方向不调参)
                  |   一次评估                            |            |
  分年            | ctx.years 事后聚合 (主度量成对)        | 全样本     | 门槛 ②

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h 为主 + 4h (仅报净差) ×
  2023-08 → 2026-08 (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘 UTC)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  组合: 主端点 = 1h (min_touch=2, tol=0.3); 跨参数 = 1h (3,0.5); 跨周期 = 4h
  (2,0.3) (门槛 ⑤)。W=24, T=1.0; H2 非对称 t_target=1.0/t_stop=0.7。

设计偏离说明 (预注册, 非 post-hoc):
  - c17 的可复用管线 (状态/触碰/阶段/角色) 以因果惯用法重写进本脚本
    (研究互 import 禁止); 事件定义与 c17 逐位一致 (逆势侧 = 涨触阻 + 跌触撑)。
  - H1 为 1:1 胜率端点 (TP/SL = T×ATR), 与 c17 的 D1 方向端点 (log(c[t+W]/
    c[t]) 符号) 不同度量 — 对照只比方向/量级, 不逐位对齐。
  - 成本模型用 ATR/close 均值换算 cost_ATR (ATR 单位), 已评估 (win/loss)
    入场上取值; expired 单边成本不在 E_R 中 (官方胜率分母不含 expired)。
  - GBM 对照首标×30 种子全管线; 条件结论均按事件分层, 不按标的做分层结论。

发布门槛自检 (10 项, 见结论表):
  - GATE (1:1 模板): GBM 30 种子同管线无条件 1:1 (全 bar, long+short 各一)
    WR ∈ [49%, 51%] 断言 + MIN_N, 失败 SystemExit (违规即停)
  - H4 GBM 门禁: GBM 侧低波动过滤增量 < +1.5pp (否则 H4 判负)
  - MIN_N: 每格 n≥200 (caliber.MIN_N), 不足格标注 [MIN_N 不足]
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 10 项门槛: ①净差超下限 ②分年 ③MIN_N ④GATE ⑤跨周期/参数 ⑥holdout
    ⑦Holm ⑧成本后>0 ⑨三重一致 ⑩负结果记录 — 逐项填表 (结论)

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c23_countertrend_trade.py
  python3 research/studies/c23_countertrend_trade.py
"""
import hashlib
import math
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
from research.causal import causal_confirmed, rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels, level_breakdown
from research.outcome import evaluate_forward, wilson_ci
from research.sim_market import gbm_matching
from research.state_features import state_series
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "combos": ((2, 0.3), (3, 0.5)),
    "primary": ("1h", (2, 0.3)),      # 主端点组合
    "warmup": 600,
    "head_drop": 120,                 # 截断后状态序列仍丢弃前 120 根
    "W": 24,                          # 1:1 窗口 (预注册)
    "T": 1.0,                         # 对称目标 ATR 倍数
    "t_target": 1.0,                  # H2 非对称目标
    "t_stop": 0.7,                    # H2 非对称止损
    "depth": 0.5,                     # level_breakdown 穿透深度 (B 系列同)
    "hold_ratio": 0.5,
    "role_w": 24,
    "role_lag_hi": 60,
    "role_excl_lo": 23,
    "z_win": 120,                     # H4 z120 窗口
    "z_q": 1.0 / 3.0,                 # 最低三分位
    "h4_inc": 0.015,                  # H4 增量门槛 +1.5pp
    "cost_fee": 0.0005,               # taker 0.05%/边
    "cost_slip": 0.0001,              # 滑点 1bp
    "cost_funding": 0.0001,           # funding 0.01%/8h
    "funding_periods": 3,             # 24h / 8h
    "by_year_list": (2024, 2025, 2026),
    "holdout_start": "2026-06-01",    # 末 3 月 holdout
    "gbm_seeds": MIN_GBM_SEEDS,
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c23_countertrend_trade"
STAGES = ("early", "accelerate", "late")
ROLES = ("刚破", "未破")
LAYER_NAMES = ("full", "early", "accelerate", "late", "刚破", "未破", "zlow",
               "y2024", "y2025", "y2026", "holdout")


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


def _trend_fn(df):
    return state_series(df)[0]


def _holdout_fn(df):
    hd = pd.Timestamp(PARAMS["holdout_start"], tz="UTC")
    return np.array([ts >= hd for ts in df.index], dtype=bool)


# ── 事件收集 (单标的, 因果, 布尔掩码, 无切片) ───────────────
def collect_one(ctx, combo, params):
    """单 ctx 逆势侧触碰事件 → 全长数组 (entry_short/entry_long/stage/role/
    zlow/year/holdout), 布尔掩码, 无切片."""
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    states = ctx.states["trend"]
    mt, tol = combo

    up = np.char.startswith(states, "trend_up")
    dn = np.char.startswith(states, "trend_down")
    stage = np.full(n, "", dtype=object)
    for i in range(n):
        s = states[i]
        if s.startswith("trend"):
            stage[i] = s.split(":")[1] if ":" in s else "accelerate"

    usable = t_idx >= params["head_drop"]
    lvls = cluster_levels(ctx.high, ctx.low, ctx.atr, k=K,
                          tolerance_mult=tol, min_touch=mt)

    entry_short = np.zeros(n, bool)
    entry_long = np.zeros(n, bool)
    role_arr = np.full(n, "", dtype=object)
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
        up_ev = up[ev]
        dn_ev = dn[ev]
        # 逆势侧: 涨触阻 → short; 跌触撑 → long
        if lv.side == "resistance":
            entry_short[ev[up_ev]] = True
        else:
            entry_long[ev[dn_ev]] = True
        # 角色 (causal_confirmed 唯一条件化出口)
        confirmed = level_breakdown(lv, c, ctx.atr, params["depth"],
                                    params["role_w"], params["hold_ratio"])[1]
        known, _ = causal_confirmed(confirmed, w=params["role_w"], lag_lo=0,
                                    lag_hi=params["role_lag_hi"])
        prefix = np.concatenate([[0], np.cumsum(confirmed)])
        epos = np.maximum(ev - params["role_excl_lo"], 0)
        recent = prefix[ev + 1] - prefix[epos]
        excl = recent > 0
        ro = np.where(excl, "excl", np.where(known[ev], "刚破", "未破"))
        role_arr[ev] = ro

    # z120 低波动 (滚动分位, 禁全样本分位)
    rp = rolling_percentile(ctx.atr, params["z_win"], params["z_q"])
    zlow = np.zeros(n, bool)
    zok = np.isfinite(rp)
    zlow[zok] = ctx.atr[zok] <= rp[zok]

    return {
        "entry_short": entry_short, "entry_long": entry_long,
        "stage": stage, "role": role_arr, "zlow": zlow,
        "years": ctx.years, "ho": np.asarray(ctx.states["holdout"], bool),
    }


def _classify(i, stage, role, zlow, years, ho):
    """入场 bar i 所属 layer 索引列表 (LAYER_NAMES 下标)"""
    out = [0]  # full
    st = stage[i]
    if st == "early":
        out.append(1)
    elif st == "accelerate":
        out.append(2)
    elif st == "late":
        out.append(3)
    ro = role[i]
    if ro == "刚破":
        out.append(4)
    elif ro == "未破":
        out.append(5)
    if zlow[i]:
        out.append(6)
    y = years[i]
    if y == 2024:
        out.append(7)
    elif y == 2025:
        out.append(8)
    elif y == 2026:
        out.append(9)
    if ho[i]:
        out.append(10)
    return out


def _acc_recs(recs, acc, coll, dirkey):
    """把 recs (win/loss) 计入分层计数 acc[layer][dirkey] = [win, loss]"""
    stage, role, zlow, years, ho = (coll["stage"], coll["role"],
                                    coll["zlow"], coll["years"], coll["ho"])
    for rec in recs:
        oc = rec.outcome
        if oc not in ("win", "loss"):
            continue
        w = 1 if oc == "win" else 0
        for li in _classify(rec.entry_idx, stage, role, zlow, years, ho):
            acc[li][dirkey][w] += 1


def symbol_counts(ctx, combo, params, need_uncond=False):
    """单标的: (acc, asym_full, uncond, atr_sum, atr_n)

    acc[layer][dir] = [win, loss]; asym_full = {dir: (win, loss)} (H2);
    uncond = {dir: (win, loss)} 全 bar 基线 (need_uncond 时);
    atr_sum/atr_n = 已评估入场 ATR/close 的累加 (成本模型).
    """
    n = ctx.n
    coll = collect_one(ctx, combo, params)
    acc = {i: {"short": [0, 0], "long": [0, 0]} for i in range(len(LAYER_NAMES))}
    atr_sum = 0.0
    atr_n = 0

    def _run(entries, direction, asym):
        kw = dict(t_mult=params["T"], w=params["W"], open_px=ctx.open)
        if asym:
            kw.update(t_target=params["t_target"], t_stop=params["t_stop"])
        return evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr,
                                entries, direction=direction, **kw)

    o_s, recs_s = _run(coll["entry_short"], "short", False)
    o_l, recs_l = _run(coll["entry_long"], "long", False)
    _acc_recs(recs_s, acc, coll, "short")
    _acc_recs(recs_l, acc, coll, "long")
    for rec in recs_s + recs_l:
        if rec.outcome in ("win", "loss"):
            atr_sum += ctx.atr[rec.entry_idx] / ctx.close[rec.entry_idx]
            atr_n += 1

    asym = {"short": [0, 0], "long": [0, 0]}
    o_s2, recs_s2 = _run(coll["entry_short"], "short", True)
    o_l2, recs_l2 = _run(coll["entry_long"], "long", True)
    for rec in recs_s2:
        if rec.outcome == "win":
            asym["short"][0] += 1
        elif rec.outcome == "loss":
            asym["short"][1] += 1
    for rec in recs_l2:
        if rec.outcome == "win":
            asym["long"][0] += 1
        elif rec.outcome == "loss":
            asym["long"][1] += 1

    uncond = None
    if need_uncond:
        all_e = np.ones(n, bool)
        o_su, _ = _run(all_e, "short", False)
        o_lu, _ = _run(all_e, "long", False)
        uncond = {"short": (o_su.n_win, o_su.n_loss),
                  "long": (o_lu.n_win, o_lu.n_loss)}
    return acc, asym, uncond, atr_sum, atr_n


def merge_acc(acc_list):
    tot = {i: {"short": [0, 0], "long": [0, 0]} for i in range(len(LAYER_NAMES))}
    for acc in acc_list:
        for li in tot:
            for d in ("short", "long"):
                tot[li][d][0] += acc[li][d][0]
                tot[li][d][1] += acc[li][d][1]
    return tot


def run_pool(dfs, combo, params, need_uncond=False):
    accs = []
    asym = {"short": [0, 0], "long": [0, 0]}
    uncond = {"short": (0, 0), "long": (0, 0)}
    atr_sum = 0.0
    atr_n = 0
    for df in dfs:
        ctx = make_ctx(df, params["warmup"],
                       state_fns={"trend": _trend_fn, "holdout": _holdout_fn})
        acc, asy, unc, a_s, a_n = symbol_counts(ctx, combo, params,
                                                need_uncond)
        accs.append(acc)
        for d in ("short", "long"):
            asym[d][0] += asy[d][0]
            asym[d][1] += asy[d][1]
            if unc is not None:
                uncond[d] = (uncond[d][0] + unc[d][0],
                             uncond[d][1] + unc[d][1])
        atr_sum += a_s
        atr_n += a_n
    return {
        "acc": merge_acc(accs),
        "asym": asym,
        "uncond": uncond if need_uncond else None,
        "atr_sum": atr_sum, "atr_n": atr_n,
    }


def pool_gbm(ref_df, combo, params):
    parts = []
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_df, seed=seed)
        ctx = make_ctx(rw, params["warmup"],
                       state_fns={"trend": _trend_fn, "holdout": _holdout_fn})
        acc, asy, unc, a_s, a_n = symbol_counts(ctx, combo, params, False)
        parts.append((acc, asy, a_s, a_n))
    accs = [p[0] for p in parts]
    asym = {"short": [0, 0], "long": [0, 0]}
    for p in parts:
        for d in ("short", "long"):
            asym[d][0] += p[1][d][0]
            asym[d][1] += p[1][d][1]
    return {
        "acc": merge_acc(accs),
        "asym": asym,
        "uncond": None,
        "atr_sum": sum(p[2] for p in parts),
        "atr_n": sum(p[3] for p in parts),
    }


# ── 统计 ─────────────────────────────────────────────────────
def layer_stats(acc, li):
    """layer li → pooled (n_win, n_loss, n_eval, wr)"""
    ws, wl = acc[li]["short"][0] + acc[li]["long"][0], \
        acc[li]["short"][1] + acc[li]["long"][1]
    ne = ws + wl
    if ne == 0:
        return (0, 0, 0, float("nan"))
    return (ws, wl, ne, ws / ne)


def dir_stats(acc, li, d):
    ws, wl = acc[li][d][0], acc[li][d][1]
    ne = ws + wl
    if ne == 0:
        return (0, 0, 0, float("nan"))
    return (ws, wl, ne, ws / ne)


def wr_line(label, rs, gs):
    """(n_win,n_loss,n_eval,wr) 两组的 真实/GBM/净差 行"""
    ws, wl, ne, wr = rs
    gs2, gl2, gne, gwr = gs
    if ne == 0 and gne == 0:
        return f"  {label}: 无样本"
    net = (wr - gwr) if ne and gne else float("nan")
    nm = "[MIN_N 通过]" if min(ne, gne) >= MIN_N else "[MIN_N 不足]"
    return ("  {}: 真实 {} (n={}) | GBM {} (n={}) | 净差 {} {}".format(
        label, _pct(wr), ne, _pct(gwr), gne,
        _pp(net) if np.isfinite(net) else "-", nm))


def _ztest_p(w1, n1, w2, n2):
    if n1 < 1 or n2 < 1:
        return float("nan")
    p1, p2 = w1 / n1, w2 / n2
    pp = (w1 + w2) / (n1 + n2)
    if pp <= 0 or pp >= 1:
        return 1.0
    se = math.sqrt(pp * (1 - pp) * (1.0 / n1 + 1.0 / n2))
    z = abs(p1 - p2) / se
    return 2.0 * 0.5 * math.erfc(z / math.sqrt(2.0))


def _holm(ps):
    order = sorted(range(len(ps)), key=lambda i: (ps[i], i))
    adj = [0.0] * len(ps)
    for rank, i in enumerate(order):
        adj[i] = min(1.0, ps[i] * (len(ps) - rank))
    return adj


# ── GATE 自检 (1:1 模板, 违规即停) ──────────────────────────
def gate(ref_1h_df, params):
    """GBM 30 种子同管线无条件 1:1 (全 bar, long+short 各一) WR ∈ [49%,51%]
    + MIN_N, 失败 SystemExit. 返回 GBM 无条件基线数字."""
    nw = nl = 0
    nw_s = nl_s = 0
    nw_l = nl_l = 0
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_1h_df, seed=seed)
        ctx = make_ctx(rw, params["warmup"],
                       state_fns={"trend": _trend_fn, "holdout": _holdout_fn})
        all_e = np.ones(ctx.n, bool)
        o_s, _ = evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr, all_e,
                                  direction="short", t_mult=params["T"],
                                  w=params["W"], open_px=ctx.open)
        o_l, _ = evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr, all_e,
                                  direction="long", t_mult=params["T"],
                                  w=params["W"], open_px=ctx.open)
        nw += o_s.n_win + o_l.n_win
        nl += o_s.n_loss + o_l.n_loss
        nw_s += o_s.n_win
        nl_s += o_s.n_loss
        nw_l += o_l.n_win
        nl_l += o_l.n_loss
    n_e = nw + nl
    if n_e < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM 无条件 n={n_e} < MIN_N={MIN_N}")
    wr_s = nw_s / (nw_s + nl_s)
    wr_l = nw_l / (nw_l + nl_l)
    if not (0.49 <= wr_s <= 0.51) or not (0.49 <= wr_l <= 0.51):
        raise SystemExit(
            f"GATE FAIL: GBM30种子 全bar 1:1 WR long={wr_l * 100:.2f}% "
            f"short={wr_s * 100:.2f}% ∉ [49%, 51%] — 探测器偏置, 停")
    ctx = make_ctx(ref_1h_df, params["warmup"],
                   state_fns={"trend": _trend_fn, "holdout": _holdout_fn})
    all_e = np.ones(ctx.n, bool)
    o_s, _ = evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr, all_e,
                              direction="short", t_mult=params["T"],
                              w=params["W"], open_px=ctx.open)
    o_l, _ = evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr, all_e,
                              direction="long", t_mult=params["T"],
                              w=params["W"], open_px=ctx.open)
    real_wr = (o_s.n_win + o_l.n_win) / (o_s.n_win + o_l.n_win +
                                         o_s.n_loss + o_l.n_loss)
    real_wr_l = o_l.n_win / (o_l.n_win + o_l.n_loss)
    real_wr_s = o_s.n_win / (o_s.n_win + o_s.n_loss)
    print(f"[GATE] 首标1h 无条件1:1: 真实 {_pct(real_wr)} "
          f"(L {_pct(real_wr_l)} / S {_pct(real_wr_s)}) | GBM30种子 "
          f"long {_pct(wr_l)} / short {_pct(wr_s)} (n={n_e}) [PASS]", flush=True)
    return {"real_wr": real_wr, "real_wr_l": real_wr_l, "real_wr_s": real_wr_s,
            "gbm_wr": nw / n_e, "gbm_wr_l": wr_l, "gbm_wr_s": wr_s,
            "n_gbm": n_e}


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.2f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _pct_plain(v):
    return f"{v * 100:.2f}"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def _cost_block(p, r, g):
    """成本模型行 (预注册): c_total=0.14% notional; cost_ATR; E_R; WR*."""
    c_total = 2 * p["cost_fee"] + p["cost_slip"] + \
        p["cost_funding"] * p["funding_periods"]
    lines = []
    lines.append(f"[成本] (预注册): taker {p['cost_fee'] * 100:.2f}%x2 + "
                 f"滑点 {p['cost_slip'] * 100:.2f}% + funding "
                 f"{p['cost_funding'] * 100:.2f}%x{p['funding_periods']} "
                 f"= {c_total * 100:.2f}% notional")
    for tag, pool in (("真实", r), ("GBM", g)):
        if pool["atr_n"] == 0:
            lines.append(f"  {tag}: 无已评估入场 (成本不可算)")
            continue
        rel = pool["atr_sum"] / pool["atr_n"]
        cost_atr = c_total / rel
        ws, wl, ne, wr = layer_stats(pool["acc"], 0)
        e_r = p["T"] * (2 * wr - 1) - cost_atr
        wr_star = (1 + cost_atr) / 2
        lines.append("  {}: mean(ATR/close)={:.4f} cost_ATR={:.3f} ATR "
                     "WR*={:.2f}% E_R={:+.3f} R (WR {} n={})".format(
            tag, rel, cost_atr, wr_star * 100, e_r, _pct(wr), ne))
    return lines


def write_out(out_path, params, g, res, year_rows, holdout_line):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},combos={},primary={},W={},T={},t_target={},t_stop={},"
        "z_win={},gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), p["combos"], p["primary"], p["W"], p["T"],
            p["t_target"], p["t_stop"], p["z_win"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(t1:1 全bar, 首标): "
        "真实 {:.2f}% GBM {:.2f}% [PASS]; long/short 分解 真实 L{:.2f}%/S{:.2f}% "
        "GBM L{:.2f}%/S{:.2f}%; 探测器自检 GBM30种子全bar 1:1 WR∈[49%,51%] "
        "[PASS]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], g["real_wr"] * 100, g["gbm_wr"] * 100,
            g["real_wr_l"] * 100, g["real_wr_s"] * 100,
            g["gbm_wr_l"] * 100, g["gbm_wr_s"] * 100, g["n_gbm"]),
        "# RESULTS: 20 标的 × 1h 为主 + 4h (仅报净差) × 2023-08..2026-08; "
        "条件层 (有入场); 入场=触碰 bar 收盘 (market); 胜率=n_win/n_eval "
        "(expired/skip/截断不计入, 官方引擎); 逆势侧=涨触阻(short)+跌触撑(long)",
        "",
    ]
    pr = res["primary"]
    # H1 主端点
    r0 = layer_stats(pr["real"]["acc"], 0)
    g0 = layer_stats(pr["gbm"]["acc"], 0)
    lines.append("[H1] 逆势触碰 1:1 (T=1.0, W=24, 1h (2,0.3)): "
                 "真实 {} (n={}) | GBM {} (n={}) | 净差 {} {}".format(
        _pct(r0[3]), r0[2], _pct(g0[3]), g0[2],
        _pp(r0[3] - g0[3]), _nm(min(r0[2], g0[2]))))
    lo, hi = wilson_ci(r0[0], r0[2])
    glo, ghi = wilson_ci(g0[0], g0[2])
    lines.append(f"  Wilson 95% CI: 真实 [{_pct_plain(lo)}%, "
                 f"{_pct_plain(hi)}%] | GBM [{_pct_plain(glo)}%, "
                 f"{_pct_plain(ghi)}%]")
    lines.append(wr_line("long(跌触撑)",
                         dir_stats(pr["real"]["acc"], 0, "long"),
                         dir_stats(pr["gbm"]["acc"], 0, "long")))
    lines.append(wr_line("short(涨触阻)",
                         dir_stats(pr["real"]["acc"], 0, "short"),
                         dir_stats(pr["gbm"]["acc"], 0, "short")))
    # 跨参数/跨周期
    rp = layer_stats(res["p1"]["real"]["acc"], 0)
    gp = layer_stats(res["p1"]["gbm"]["acc"], 0)
    lines.append(wr_line("[H1-p] 1h (3,0.5) 跨参数", rp, gp))
    rt = layer_stats(res["t4"]["real"]["acc"], 0)
    gt = layer_stats(res["t4"]["gbm"]["acc"], 0)
    lines.append(wr_line("[H1-t] 4h (2,0.3) 跨周期 (仅净差)", rt, gt))
    # H2 非对称
    a_r = pr["real"]["asym"]
    a_g = pr["gbm"]["asym"]
    nwr = a_r["short"][0] + a_r["long"][0]
    nlr = a_r["short"][1] + a_r["long"][1]
    nwg = a_g["short"][0] + a_g["long"][0]
    nlg = a_g["short"][1] + a_g["long"][1]
    lines.append("")
    er = nwr + nlr
    eg = nwg + nlg
    wr_r = nwr / er if er else float("nan")
    wr_g = nwg / eg if eg else float("nan")
    lines.append("[H2] 非对称 (t_target=1.0/t_stop=0.7, 1h 主组合): "
                 "真实 {} (n={}) | GBM {} (n={}) | 净差 {} {}".format(
        _pct(wr_r), er, _pct(wr_g), eg,
        _pp(wr_r - wr_g) if np.isfinite(wr_r) and np.isfinite(wr_g) else "-",
        _nm(min(er, eg))))
    # H3 阶段/角色
    lines.append("")
    lines.append("[H3] 阶段分层 (1h 主组合, 1:1):")
    for li, stg in ((1, "early"), (2, "accelerate"), (3, "late")):
        lines.append(wr_line(f"  {stg}",
                             layer_stats(pr["real"]["acc"], li),
                             layer_stats(pr["gbm"]["acc"], li)))
    lines.append("[H3-r] 角色分层 (1h 主组合, 1:1):")
    for li, rl in ((4, "刚破"), (5, "未破")):
        lines.append(wr_line(f"  {rl}",
                             layer_stats(pr["real"]["acc"], li),
                             layer_stats(pr["gbm"]["acc"], li)))
    # H4 波动过滤
    lines.append("")
    rf = layer_stats(pr["real"]["acc"], 0)
    gf = layer_stats(pr["gbm"]["acc"], 0)
    rz = layer_stats(pr["real"]["acc"], 6)
    gz = layer_stats(pr["gbm"]["acc"], 6)
    net_f = (rf[3] - gf[3]) if rf[2] and gf[2] else float("nan")
    net_z = (rz[3] - gz[3]) if rz[2] and gz[2] else float("nan")
    inc = net_z - net_f
    g_inc = (gz[3] - gf[3]) if gz[2] and gf[2] else float("nan")
    lines.append("[H4] z120 低波动过滤 (1h 主组合, 1:1):")
    lines.append(wr_line("  低波动子集(z120 最低1/3)",
                         layer_stats(pr["real"]["acc"], 6),
                         layer_stats(pr["gbm"]["acc"], 6)))
    lines.append(f"  全体净差 {_pp(net_f)} | 低波动净差 {_pp(net_z)} | "
                 f"增量(低波动−全体) {_pp(inc)}")
    lines.append(f"  GBM 门禁: GBM 侧过滤增量 {_pp(g_inc)} "
                 f"({'PASS <+1.5pp' if np.isfinite(g_inc) and g_inc < p['h4_inc'] else 'FAIL ≥+1.5pp'})")
    # 漂移分解
    lines.append("")
    u = pr["real"]["uncond"]
    u_ws = u["short"][0] + u["long"][0]
    u_ls = u["short"][1] + u["long"][1]
    u_wr = u_ws / (u_ws + u_ls)
    cond_inc = rf[3] - u_wr
    lines.append("[漂移分解] (1h 主组合, 同方向 pooled):")
    lines.append(f"  真实触碰 {_pct(rf[3])} (n={rf[2]}) | 真实无条件(全bar同方向) "
                 f"{_pct(u_wr)} (n={u_ws + u_ls}) | 条件化增量 {_pp(cond_inc)}")
    lines.append(f"  净差(真实−GBM 同管线) {_pp(net_f)} "
                 f"= 条件化增量 {_pp(cond_inc)} + (GBM 漂移调整)")
    # 成本
    lines.append("")
    lines.extend(_cost_block(p, pr["real"], pr["gbm"]))
    # Holm
    lines.append("")
    ra = pr["real"]["acc"]
    ga = pr["gbm"]["acc"]

    def _asym_stats(a):
        ws = a["short"][0] + a["long"][0]
        ls = a["short"][1] + a["long"][1]
        return (ws, ls, ws + ls, ws / (ws + ls)) if ws + ls else (0, 0, 0, float("nan"))

    test_names = ("H1 full", "H2 asym", "H3 early", "H3 accel", "H3 late",
                  "H3 刚破", "H3 未破", "H4 zlow")
    r_stats = [layer_stats(ra, 0), _asym_stats(pr["real"]["asym"]),
               layer_stats(ra, 1), layer_stats(ra, 2), layer_stats(ra, 3),
               layer_stats(ra, 4), layer_stats(ra, 5), layer_stats(ra, 6)]
    g_stats = [layer_stats(ga, 0), _asym_stats(pr["gbm"]["asym"]),
               layer_stats(ga, 1), layer_stats(ga, 2), layer_stats(ga, 3),
               layer_stats(ga, 4), layer_stats(ga, 5), layer_stats(ga, 6)]
    ps = [_ztest_p(r_[0], r_[2], g_[0], g_[2])
          for r_, g_ in zip(r_stats, g_stats)]
    adj = _holm(ps)
    lines.append("[统计] 两比例 z 检验 + Holm 校正 (k=8, α=0.05):")
    for name, pv, av in zip(test_names, ps, adj):
        if np.isfinite(pv):
            lines.append(f"  {name:<8} p={pv:.4f} Holm adj p={av:.4f} "
                         f"{'显著' if av < 0.05 else '不显著'}")
        else:
            lines.append(f"  {name:<8} n 不足")
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c17 (2026-08-13, 描述层): 逆势侧 D1 净差 -4.09pp "
                 "(1h 主组合, L6), 阶段梯度 2.76pp (early -1.92pp accel -4.68pp "
                 "late -4.48pp, L13-16), 角色差 2.20pp (刚破 -1.04pp 未破 -3.25pp, "
                 "L17-19); B5 (作废): 涨触阻力 D1=46.2% vs GBM 49.7% (Δ-3.5pp)")
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(year_rows))
    lines.append("# HOLDOUT: " + holdout_line)
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

    g = gate(dfs["1h"][0], PARAMS)

    primary = PARAMS["primary"]
    res = {}
    res["primary"] = {
        "real": run_pool(dfs["1h"], primary[1], PARAMS, need_uncond=True),
        "gbm": pool_gbm(dfs["1h"][0], primary[1], PARAMS),
    }
    res["p1"] = {
        "real": run_pool(dfs["1h"], (3, 0.5), PARAMS),
        "gbm": pool_gbm(dfs["1h"][0], (3, 0.5), PARAMS),
    }
    res["t4"] = {
        "real": run_pool(dfs["4h"], (2, 0.3), PARAMS),
        "gbm": pool_gbm(dfs["4h"][0], (2, 0.3), PARAMS),
    }

    # BY_YEAR (主度量 = H1 净差, 真实+GBM 成对)
    year_rows = []
    r_a = res["primary"]["real"]["acc"]
    g_a = res["primary"]["gbm"]["acc"]
    for yy in PARAMS["by_year_list"]:
        li = LAYER_NAMES.index(f"y{yy}")
        rr = layer_stats(r_a, li)
        gg = layer_stats(g_a, li)
        if rr[2] == 0 and gg[2] == 0:
            continue
        net = (rr[3] - gg[3]) if rr[2] and gg[2] else float("nan")
        year_rows.append("{} 真实 {} (n={}) GBM {} (n={}) 净差 {}".format(
            yy, _pct(rr[3]), rr[2], _pct(gg[3]), gg[2],
            _pp(net) if np.isfinite(net) else "-"))

    # HOLDOUT (末 3 月, 只报方向)
    rh = layer_stats(r_a, 10)
    gh = layer_stats(g_a, 10)
    rfull = layer_stats(r_a, 0)
    gfull = layer_stats(g_a, 0)
    full_net = (rfull[3] - gfull[3]) if rfull[2] and gfull[2] else float("nan")
    if rh[2] and gh[2] and np.isfinite(full_net):
        hnet = rh[3] - gh[3]
        holdout_line = ("2026-06-01 起: 真实 {} (n={}) GBM {} (n={}) 净差 {} "
                        "({})".format(
            _pct(rh[3]), rh[2], _pct(gh[3]), gh[2], _pp(hnet),
            "方向与全样本一致" if full_net * hnet > 0 else "方向与全样本相反"))
    else:
        holdout_line = "n 不足"

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, res, year_rows, holdout_line)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
