#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C17 趋势态触位后逆势折返 (因果口径重验) (2026-08-13, 无未来函数, 1h/4h)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画市场事实 (趋势态触碰聚类位带
  后 W=24 根端点沿趋势方向的概率及其阶段/角色结构), 无入场, 无交易含义, 无任何
  方向/收益/成本结论。定位声明: 效应基线预期仅 2-4pp, 本研究只确认效应存在性,
  不构成任何交易主张。所有统计为事后描述; 若未来用作特征/条件, 必须经滚动口径
  重验。描述层发布门槛: 无胜率/期望要求, 但必须有 GBM 无信息对照与数字可溯源。

============================================================
研究问题 (预注册, 运行前冻结): 趋势态触位后逆势折返 (-2~-4pp) 是否独立于
  阶段/角色、因果化后仍存在?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1: D1 沿趋势概率净差 ≤ -2pp (逆势, 涨趋势触阻力→后续更易向下), 4 组合同号
  H2: 阶段分层 (early/accel/late) 净差无梯度 (|Δ|<2pp, 复验 B5b"非末期混杂")
  H3: causal 角色层 (刚破/未破) 不产生额外方向信息

  操作化 (运行前锁定):
    - 主度量 D1 = 逆势侧触碰 (涨趋势触阻力 + 跌趋势触支撑) 后 W=24 根端点沿
      趋势方向概率, 对数度量 sign(log(close[t+W]/close[t])) 以 close[t] 为参照
    - H1 判据: 逆势侧 D1 净差 (真实−GBM 同管线) ≤ -2pp 且 4 组合 (2 周期×2 参数)
      全部同号 (负)
    - H2 判据: 逆势侧触碰按阶段 (early/accelerate/late) 分层净差, 每组合内
      三阶段净差 max−min < 2pp (无梯度)
    - H3 判据: 逆势侧触碰按 causal 角色 (刚破/未破) 分层, |净差_刚破 − 净差_未破|
      < 2pp 且两角色 n ≥ MIN_N (不产生额外方向信息)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  high/low/close   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口 (禁手动切片)
  趋势状态/阶段    | state_features.state_series (causal    | bar 收盘后 | state_fns 截断后计算, 截断
                   |   状态机, 逐 bar 只回看已收盘指标)     |            |   不变性 (invariance 测试)
  cluster 位带     | levels.cluster_levels 在线聚类+冻结     | confirm_at | 冻结后 price/band/confirm_at
                   |   (pivot 按确认时序逐入组)             |            |   不可变 (levels R1/R2)
  触碰事件         | bar 区间 (low<=price+band 且            | bar 收盘后 | 纯触碰事件 — 不依赖 confirmed
                   |   high>=price-band) ∩ t>=confirm_at,   |            |   标签; 每段连续触碰首根
                   |   取每段连续触碰首根 (entry)           |            |
  角色层           | causal.causal_confirmed(confirmed, w=  | t 时刻已知 | research.causal 唯一条件化
                   |   24, lag_lo=0, lag_hi=60): 刚破 =     |            |   出口 (B5c H4 泄漏修复);
                   |   ∃c∈[t-60,t-24] confirmed 且 c+24<=t; |            |   [t-23,t] 突破样本剔除
                   |   未破 = 无; [t-23,t] 突破样本剔除     |            |
  confirmed (事后) | levels.level_breakdown (depth=0.5,     | 事后标签   | 仅经 causal_confirmed 条件化
                   |   w=24, hold_ratio=0.5)                |            |   /剔除, 不直接条件化
  D1 度量          | sign(log(close[t+W]/close[t])) 与       | 事后 (端点  | 描述层端点概率; 对数度量防
                   |   触碰时刻趋势方向比对 (涨: >0, 跌:<0) |   t+W 收盘) |   Jensen/价格水平
  阶段分层         | state_series 输出的 causal stage       | bar 收盘后 | 因果状态机阶段; 段内位置只
                   |   (early/accelerate/late)              |            |   用已确认信息, 禁最终段长
                   |                                       |            |   归一 (a6d 教训)
  分年             | ctx.years (截断坐标) 事后聚合           | 全样本     | 描述层 BY_YEAR (成对 真实+GBM)
  GBM 无信息对照   | sim_market.gbm_matching(ref_df, seed)   | 锚定真实   | 固定种子序列 0..29; 首标×30
                   |   (索引/长度/σ 锚定真实)                |            |   种子全管线

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  组合: 1h/4h × (min_touch=2, tol=0.3) / (3, 0.5), 共 4 组合 (预注册)。
  W=24 端点窗口; 角色: causal_confirmed(w=24, lag_lo=0, lag_hi=60) +
  [t-23,t] 突破剔除; head_drop=120 覆盖状态指标 warm-up。

设计偏离说明 (预注册, 非 post-hoc):
  - D1 用对数度量 sign(log(c[t+W]/c[t])) 统一口径 (B5/B5b/B5c 用原始
    close[t+W]>close[t] 比较): sign 层面等价, 但对数化防价格水平/Jensen,
    预注册固定。
  - B5c H4 角色层用 searchsorted 直接条件化 confirmed (确认窗口未闭合 =
    未来标签泄漏, 该研究已作废); c17 一律走 causal_confirmed (conf∈[t-60,
    t-24] 且窗口闭合) + [t-23,t] 突破样本剔除 — 与 B5c 的数字差异含口径
    修复成分, 结论对照时注明。
  - 阶段定义沿用 state_series 的 causal 状态机阶段; 不做任何"段内位置/最终
    段长归一" (a6d 教训: 最终段长含未来信息)。
  - GBM 对照为"首标×30 种子全管线" (PLAN §4 描述层 exit 模板最小覆盖);
    结论均按事件分层, 不按标的做分层结论。

发布门槛自检 (描述层):
  - GATE 探测器: GBM 30 种子同管线 D1 (逆势侧) null mean ∈ [49%, 51%] 且
    n ≥ MIN_N, 任一失败 SystemExit (违规即停); 无条件基线 (真实+GBM) 同出
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (同管线重放)
  - MIN_N 检查: 每个输出格含 n, 不足格标注 [MIN_N 不足] (全单元格输出)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 定位: 效应基线仅 2-4pp, 本研究只确认效应存在性, 无入场/无交易含义

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c17_countertrend_touch.py
  python3 research/studies/c17_countertrend_touch.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path (脚本以 `python3 research/studies/c17_...py` 直接运行时,
# sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

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
    "combos": ((2, 0.3), (3, 0.5)),   # (min_touch, tolerance_mult) — 预注册
    "warmup": 600,                     # make_ctx 截断起点 (覆盖指标 warm-up)
    "head_drop": 120,                  # 截断后状态序列仍丢弃前 120 根 (指标收敛)
    "W": 24,                           # D1 端点窗口 (预注册)
    "depth": 0.5,                      # level_breakdown 穿透深度 (B 系列同)
    "hold_ratio": 0.5,                 # 确认突破保持比例 (B 系列同)
    "role_w": 24,                      # causal_confirmed 确认窗口 (B 系列同)
    "role_lag_hi": 60,                 # causal_confirmed 上界 (conf∈[t-60,t-24])
    "role_excl_lo": 23,                # [t-23, t] 突破样本剔除
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),  # 2023 为部分年, 不纳入
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c17_countertrend_touch"
STAGES = ("early", "accelerate", "late")
ROLES = ("刚破", "未破")


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


# ── 事件收集 (单标的, 因果, 布尔掩码, 无切片) ───────────────
def collect_one(ctx, combo, params):
    """单个 ctx 的趋势态触碰事件 → 事件数组 (d1/year/stage/role/dr/side)

    - d1   : 触碰后 W 根端点沿趋势方向指示 (涨: logr>0; 跌: logr<0; 中性: logr>0)
    - dr   : "up"/"dn"/"neu" (触碰时刻趋势方向)
    - stage: early/accelerate/late (趋势触碰; 中性为空串)
    - role : 刚破/未破/excl (causal_confirmed + [t-23,t] 突破剔除)
    """
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    states = ctx.states["trend"]
    W = params["W"]
    mt, tol = combo

    up = np.char.startswith(states, "trend_up")
    dn = np.char.startswith(states, "trend_down")
    neu = ~(up | dn)
    stage = np.full(n, "", dtype=object)
    for i in range(n):
        s = states[i]
        if s.startswith("trend"):
            stage[i] = s.split(":")[1] if ":" in s else "accelerate"

    # D1 (对数度量端点, 布尔掩码无切片)
    logr = np.full(n, np.nan)
    ok_t = t_idx + W < n
    idx_t = t_idx[ok_t]
    logr[ok_t] = np.log(c[idx_t + W] / c[idx_t])
    usable = ok_t & np.isfinite(logr) & (t_idx >= params["head_drop"])

    d1 = np.full(n, np.nan)
    m = usable & up
    d1[m] = logr[m] > 0
    m = usable & dn
    d1[m] = logr[m] < 0
    m = usable & neu
    d1[m] = logr[m] > 0

    lvls = cluster_levels(ctx.high, ctx.low, ctx.atr, k=K,
                          tolerance_mult=tol, min_touch=mt)

    d1_l, yr_l, st_l, ro_l, dr_l, side_l = [], [], [], [], [], []
    n_lvls = 0
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (ctx.low <= p_hi) & (ctx.high >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev
        ev = np.flatnonzero(entry & usable)
        if len(ev) == 0:
            continue
        n_lvls += 1
        d1_l.append(d1[ev])
        yr_l.append(ctx.years[ev])
        st_l.append(stage[ev])
        up_ev = up[ev]
        dn_ev = dn[ev]
        dr_l.append(np.where(up_ev, "up", np.where(dn_ev, "dn", "neu")))
        side_l.append(np.full(len(ev), lv.side, dtype=object))
        # 角色 (confirmed 事后标签 → causal_confirmed 唯一条件化出口)
        confirmed = level_breakdown(lv, c, ctx.atr, params["depth"],
                                    params["role_w"], params["hold_ratio"])[1]
        known, _ = causal_confirmed(confirmed, w=params["role_w"], lag_lo=0,
                                    lag_hi=params["role_lag_hi"])
        prefix = np.concatenate([[0], np.cumsum(confirmed)])
        epos = np.maximum(ev - params["role_excl_lo"], 0)
        recent = prefix[ev + 1] - prefix[epos]
        excl = recent > 0
        ro_l.append(np.where(excl, "excl", np.where(known[ev], "刚破", "未破")))
    if not d1_l:
        return {"d1": np.array([], float), "year": np.array([], int),
                "stage": np.array([], object), "role": np.array([], object),
                "dr": np.array([], object), "side": np.array([], object),
                "n_lvls": 0, "n_touch": 0}
    return {"d1": np.concatenate(d1_l), "year": np.concatenate(yr_l),
            "stage": np.concatenate(st_l), "role": np.concatenate(ro_l),
            "dr": np.concatenate(dr_l), "side": np.concatenate(side_l),
            "n_lvls": n_lvls, "n_touch": int(np.concatenate(d1_l).size)}


def _merge_pool(parts):
    keys = ("d1", "year", "stage", "role", "dr", "side")
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["n_lvls"] = sum(p["n_lvls"] for p in parts)
    out["n_touch"] = sum(p["n_touch"] for p in parts)
    return out


def pool(dfs, combo, params):
    """多标的 (真实) 触碰池 — 全部拼接"""
    parts = [collect_one(make_ctx(df, params["warmup"],
                                  state_fns={"trend": _trend_fn}), combo, params)
             for df in dfs]
    return _merge_pool(parts)


def pool_gbm(ref_df, combo, params):
    """GBM 对照池 — 首标 × gbm_seeds 种子, 逐种子同管线重放"""
    parts = []
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_df, seed=seed)
        ctx = make_ctx(rw, params["warmup"], state_fns={"trend": _trend_fn})
        parts.append(collect_one(ctx, combo, params))
    return _merge_pool(parts)


# ── 掩码与统计 ───────────────────────────────────────────────
def counter_mask(pooled):
    """逆势侧: 涨趋势触阻力 + 跌趋势触支撑"""
    dr = pooled["dr"]
    side = pooled["side"]
    return ((dr == "up") & (side == "resistance")) | \
           ((dr == "dn") & (side == "support"))


def with_mask(pooled):
    """顺势侧: 涨趋势触支撑 + 跌趋势触阻力"""
    dr = pooled["dr"]
    side = pooled["side"]
    return ((dr == "up") & (side == "support")) | \
           ((dr == "dn") & (side == "resistance"))


def stat(pooled, mask):
    """{掩码: (n, D1 mean)}"""
    d1 = pooled["d1"]
    m = mask
    if not m.any():
        return (0, float("nan"))
    return (int(m.sum()), float(np.mean(d1[m])))


def cell_stats(pooled):
    """方向×侧 4 格 + 中性 → {cell: (n, mean)}"""
    dr = pooled["dr"]
    side = pooled["side"]
    out = {}
    for d in ("up", "dn"):
        for s in ("resistance", "support"):
            out[f"{d}_{s}"] = stat(pooled, (dr == d) & (side == s))
    out["中性"] = stat(pooled, dr == "neu")
    return out


def stage_stats(pooled, ctr):
    out = {}
    st = pooled["stage"]
    for stg in STAGES:
        out[stg] = stat(pooled, ctr & (st == stg))
    return out


def role_stats(pooled, ctr):
    out = {}
    ro = pooled["role"]
    for r in ROLES:
        out[r] = stat(pooled, ctr & (ro == r))
    return out


def year_stats(pooled, ctr, params):
    out = {}
    y = pooled["year"]
    for yy in params["by_year_list"]:
        out[yy] = stat(pooled, ctr & (y == yy))
    return out


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_1h_df, params):
    """探测器自检: GBM 30 种子 (首标 1h, 主组合 (2,0.3)) 同管线 D1 (逆势侧)
    null mean ∈ [49%, 51%] 且 n ≥ MIN_N, 失败 SystemExit.

    返回 GBM 池 (主组合 GBM 侧直接复用, 免重复计算) + 真实/GBM 无条件基线。
    """
    combo = params["combos"][0]
    gbm = pool_gbm(ref_1h_df, combo, params)
    gbm_n, gbm_mean = stat(gbm, counter_mask(gbm))
    ctx = make_ctx(ref_1h_df, params["warmup"], state_fns={"trend": _trend_fn})
    real = collect_one(ctx, combo, params)
    real_n, real_mean = stat(real, counter_mask(real))
    print(f"[GATE] 首标1h主组合 D1(逆势侧): 真实 {_pct(real_mean)} (n={real_n}) | "
          f"GBM30种子 {_pct(gbm_mean)} (n={gbm_n}, ≥{MIN_GBM_SEEDS} 种子)",
          flush=True)
    if gbm_n < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM n={gbm_n} < MIN_N={MIN_N}")
    if not (0.49 <= gbm_mean <= 0.51):
        raise SystemExit(
            f"GATE FAIL: GBM30种子 D1 null mean={gbm_mean * 100:.2f}% "
            f"∉ [49%, 51%] — 探测器机械性偏置, 停")
    return {"real_mean": real_mean, "gbm_mean": gbm_mean,
            "n_gbm": gbm_n, "gbm": gbm, "combo": combo}


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.2f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def _net_line(label, rs, gs):
    rn, rm = rs
    gn, gm = gs
    net = (rm - gm) if np.isfinite(rm) and np.isfinite(gm) else float("nan")
    return ("  {}: 真实 {} (n={}) | GBM {} (n={}) | 净差 {} {}".format(
        label, _pct(rm), rn, _pct(gm), gn,
        _pp(net) if np.isfinite(net) else "-", _nm(rn)))


def fmt_combo_key(tf, combo):
    return f"{tf} (min_touch={combo[0]}, tol={combo[1]})"


def write_out(out_path, params, g, results, by_year_rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},combos={},W={},warmup={},head_drop={},role_w={},role_lag_hi={},"
        "gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), p["combos"], p["W"], p["warmup"],
            p["head_drop"], p["role_w"], p["role_lag_hi"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(首标1h主组合 D1 逆势侧沿趋势概率): "
        "真实 {:.2f}% GBM {:.2f}% [PASS]; 探测器自检 GBM30种子同管线 D1 "
        "null∈[49%,51%] [PASS]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], g["real_mean"] * 100, g["gbm_mean"] * 100,
            g["n_gbm"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 描述层无入场, 无交易含义; "
        "D1 = 触碰后 W=24 根端点沿趋势方向概率, 对数度量 sign(log(c[t+W]/c[t])), "
        "以 close[t] 为参照; 逆势侧 = 涨趋势触阻力 + 跌趋势触支撑",
        "",
    ]
    for tf in p["tf_list"]:
        for combo in p["combos"]:
            r = results[(tf, combo)]
            rs, gs = r["real"], r["gbm"]
            key = fmt_combo_key(tf, combo)
            lines.append(f"[组合] {key} — 位带/触碰: 真实 {rs['n_lvls']}/{rs['n_touch']} "
                         f"| GBM {gs['n_lvls']}/{gs['n_touch']}")
            lines.append(_net_line("D1 逆势侧", stat(rs, counter_mask(rs)),
                                   stat(gs, counter_mask(gs))))
            lines.append(_net_line("D1 顺势侧", stat(rs, with_mask(rs)),
                                   stat(gs, with_mask(gs))))
            lines.append(_net_line("D1 中性(无条件基线)", stat(rs, rs["dr"] == "neu"),
                                   stat(gs, gs["dr"] == "neu")))
            rc, gc = cell_stats(rs), cell_stats(gs)
            for cell in ("up_resistance", "up_support", "dn_resistance", "dn_support"):
                lines.append(_net_line(f"  {cell}", rc[cell], gc[cell]))
            # 阶段 (逆势侧)
            r_st = stage_stats(rs, counter_mask(rs))
            g_st = stage_stats(gs, counter_mask(gs))
            for stg in STAGES:
                lines.append(_net_line(f"  阶段 {stg}", r_st[stg], g_st[stg]))
            nets = [r_st[s][1] - g_st[s][1] for s in STAGES
                    if np.isfinite(r_st[s][1]) and np.isfinite(g_st[s][1])]
            if len(nets) == 3:
                grad = max(nets) - min(nets)
                lines.append(f"  阶段梯度(early/accel/late 净差 max-min): "
                             f"{grad * 100:.2f}pp")
            else:
                lines.append("  阶段梯度: n 不足")
            # 角色 (逆势侧)
            r_ro = role_stats(rs, counter_mask(rs))
            g_ro = role_stats(gs, counter_mask(gs))
            for rl in ROLES:
                lines.append(_net_line(f"  角色 {rl}", r_ro[rl], g_ro[rl]))
            nets = [r_ro[rl][1] - g_ro[rl][1] for rl in ROLES
                    if np.isfinite(r_ro[rl][1]) and np.isfinite(g_ro[rl][1])]
            if len(nets) == 2:
                lines.append(f"  角色差(刚破−未破 净差): "
                             f"{abs(nets[0] - nets[1]) * 100:.2f}pp")
            else:
                lines.append("  角色差: n 不足")
    lines.append("")
    lines.append("[对照-历史] B5/B5b/B5c (2026-08-12 整体作废, 仅形状参照): "
                 "B5 涨触阻力 D1=46.2% vs GBM 49.7% (净差-3.5pp), 中性 48.8% vs "
                 "50.2% (熊市漂移); B5b 阶段 up D1=45.6~46.7%, dn_early 49.5% "
                 "(跌早期无反效); B5c 角色(泄漏版) up未破_support D1=59.0% vs "
                 "GBM 69.0% (角色效应机械性)")
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

    # GATE 自检 (失败 SystemExit — 违规即停)
    g = gate(dfs["1h"][0], PARAMS)

    # 4 组合: 真实池 (全标的) + GBM 池 (首标×30 种子; 主组合复用 gate 池)
    results = {}
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            real = pool(dfs[tf], combo, PARAMS)
            if (tf, combo) == ("1h", g["combo"]):
                gbm = g["gbm"]
            else:
                gbm = pool_gbm(dfs[tf][0], combo, PARAMS)
            results[(tf, combo)] = {"real": real, "gbm": gbm}

    # BY_YEAR 成对 (主度量 = 逆势侧 D1, 真实 全标的 + GBM 首标30种子)
    year_rows = []
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            r = results[(tf, combo)]
            rs, gs = r["real"], r["gbm"]
            r_y = year_stats(rs, counter_mask(rs), PARAMS)
            g_y = year_stats(gs, counter_mask(gs), PARAMS)
            for yy in PARAMS["by_year_list"]:
                rn, rm = r_y[yy]
                gn, gm = g_y[yy]
                if rn == 0 and gn == 0:
                    continue
                year_rows.append("{} {} {} 真实 {} (n={}) GBM {} (n={})".format(
                    tf, combo, yy, _pct(rm), rn, _pct(gm), gn))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, results, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
