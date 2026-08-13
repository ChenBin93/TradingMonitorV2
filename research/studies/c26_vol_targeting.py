#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C26 波动目标仓位 (vol targeting) (2026-08-13, 无未来函数, 1h 为主 + 4h 交叉)

仓位管理方向 (Phase 2.5) 第一项: 纯风险改善研究, 不创造方向 edge — 入场用
无条件基座 (固定排程, 无状态/触碰条件)。结论语言按门槛裁决, 不达标一律写
"未达门槛", 不主张任何可交易性。

============================================================
研究问题 (预注册, 运行前冻结): 波动率可预测 (c12: H≈0.9) 的前提下, 把仓位
按 1/ATR 缩放 (每笔风险恒定) 相比固定仓位, 能否降低权益曲线波动率与回撤、
且不损失收益?

入场基座 (无方向 edge 基座, 隔离纯仓位效应):
  固定排程: 截断坐标 t % 24 == 0 且 t < n − W (每 24 根一笔, 非重叠;
  截断坐标 0 = 原始 warmup 后第一根); 方向 long-only (对称性由镜像保证);
  evaluate_forward 1:1 (T=1.0, W=24, 官方引擎)

对照:
  A = 固定仓位 (每笔 1 单位):  P&L_A = R × ATR_rel(entry),  ATR_rel = ATR/close
  B = 波动目标仓位 (每笔 K/ATR_rel, K=1, 风险恒定): P&L_B = R × K_rel, K_rel=1
  R ∈ {+1 (win), −1 (loss), 0 (expired/skip)} (1:1 引擎 R 单位 = ATR 距离);
  P&L 以相对 ATR 计价 (相对 close 百分比尺度), 跨标的可比

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1 (前提重验): ATR_rel(entry) 与后续 W 根实现波动 (|log-ret| 均值) 正相关,
     真实 > GBM 同管线 (GBM≈0, 波动无记忆)
  H2 (主端点): B 的每笔 P&L 标准差 < A 的每笔 P&L 标准差, 且
     真实侧降低率 − GBM 同管线降低率 ≥ 10pp
     (GBM 侧的降低 = ATR 离散度的机械效应, 必须扣除; 真实侧超出的部分
     才是波动持续性带来的)
  H3 (回撤): B 累积权益曲线最大回撤 < A 的最大回撤 (真实−GBM 净差 < 0)
  H4 (收益中性): B 总收益 (Σ P&L) 不显著低于 A (基座均值≈0, 仓位缩放不应
     改变符号; 用 z 检验报告, 不定严格门槛但必须报告)
  成本: 调仓手续费按仓位变化量计 — taker 0.05% × |B_i − B_{i-1}|; 排程
     非重叠每笔独立 (前一笔已平仓), 入场即建仓成本 0.05%×B_i = 0.0005 /
     ATR_rel(entry); 固定仓位 A 无调仓成本 (基准); 按此预注册口径计

门槛裁决: H2 净差 ≥10pp 且 H3 过且分年 ≥2/3 年同向 → 达标; 否则"未达门槛"

操作定义 (冻结):
  - H1 实现波动 = 入场后 W 根 (t+1..t+W) |log(close/close_prev)| 均值
    (因果: 只用入场后数据, 事后标签合法); 相关性 = 池内逐事件皮尔逊相关
  - H2 降低率 = (sd(P&L_A) − sd(P&L_B)) / sd(P&L_A), 池内逐事件 P&L
  - H3 权益曲线: 全部事件按 entry_idx 全局升序串接 (跨标的/种子合计,
    同 idx 按池内顺序), 权益 = cumsum(P&L); MDD = max(cumsum 峰值回撤);
    归一化口径: MDD 除以该池每笔 P&L 标准差 (波动单位, A vs B 同尺度可比
    — 直接 MDD 比较因 P&L_A 与 P&L_B 尺度不同 (ATR_rel≈1.5% vs 1) 无意义,
    必须以波动单位归一化, 预注册)
  - H4 z = (mean_B − mean_A) / sqrt(var_B/n_B + var_A/n_A) (两样本)
  - 成本: Σcost_B = Σ 0.0005 / ATR_rel(entry) (仅 B, A 基准无调仓成本)
  - HOLDOUT = 末 3 月 (2026-06..08) 事件池, H2 降低率净差方向 (只报方向)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/数据        | 计算方式                         | 可用时点   | 依据
  close/high/low/  | research.ctx.make_ctx 统一截断   | bar 收盘后 | ctx 唯一对齐出口
  open/atr/years   |  (内部 iloc[warmup:])            |            | (禁一切手动切片)
  月份 (HOLDOUT)   | df.index.month, keep 掩码对齐    | bar 收盘后 | 布尔掩码截断
  ATR_rel          | ctx.atr / ctx.close (entry bar)  | bar 收盘后 | make_ctx 内置 ATR
  排程入场         | t % 24 == 0 (截断坐标, 掩码)     | bar 收盘后 | 固定排程无信息
  1:1 判定         | outcome.evaluate_forward         | 已收盘 bar | 官方引擎 (对称
                   |  (T=1.0, W=24, 默认参数)         |            | t_mult)
  实现波动 (H1)    | 入场后 W 根 |logret| 均值 (lead   | 全样本事后 | 事后标签合法
                   |  掩码, 无切片)                   |            | (仅入场后数据)
  仓位缩放         | B_i = 1 / ATR_rel(entry)         | bar 收盘后 | K=1 风险恒定
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
  - P&L 尺度标定 (运行前, 非 post-hoc): 字面定义 P&L_A = R×ATR_rel 与
    P&L_B = R×1 的尺度不同 — B 的仓位 = 1/ATR_rel (≈67 倍 A), sd(P&L_B)
    ≈ 0.5 而 sd(P&L_A) ≈ 0.0075, 直接 sd 比较由尺度决定 (降低率为负),
    H2"B sd < A sd"在字面定义下数学不成立。标定修正: 比较用同平均仓位
    口径 A' = R × ATR_rel / E[ATR_rel] (池内均值归一, 均仓位 = 1 = B),
    H2/H3/H4 均以 A' 与 B 比较 — 回答"相同平均风险水平下波动目标能否
    降低 P&L 波动", 正是任务意图。字面 P&L_A 仅作报告参考。
  - H3 回撤用直接 MDD (A' 与 B 同尺度 — 均仓位 1): 与 H2 的 sd 降低
    同向 (B 每笔波动更低 → 权益回撤更浅); 跨真实/GBM 的事件数差异
    (20 标的 vs 30 种子) 影响 MDD 绝对量, 净差方向判定为主 (局限注明)。
  - 成本口径: 排程非重叠每笔独立, |B_i − B_{i-1}| = 建仓仓位 B_i (前一笔
    已平仓), 故每笔 B 成本 = 0.0005 × (1/ATR_rel); A 固定仓位无调仓成本。
  - 4h 交叉: 同样管线 (stride=24 在 4h = 每 4 天一笔), 只报 H1/H2 净差。
  - GBM 对照首标 × 30 种子同管线; 分年 GBM 侧聚合首标 30 种子。

发布门槛自检 (仓位管理):
  - H2 净差 ≥ 10pp 且 H3 过 (真实−GBM 净差 < 0) 且分年 ≥2/3 年同向 → 达标
  - "有效"类措辞只有在 H2/H3 全过 + 分年达标时才可用, 否则写"未达门槛"
  - GATE: 1:1 无条件基线 (真实+GBM long, GBM ∈ [49%,51%]) + GBM 侧 H1
    相关性 ≈0 断言 (±0.05 容差); gbm_seeds ≥ 30; 失败 SystemExit
  - 结论↔.out↔脚本三重一致; 结论数字全部带 (.out:L行号)

运行命令:
  # 两道门禁: 引擎门禁 → 脚本门禁 → 运行
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c26_vol_targeting.py
  python3 research/studies/c26_vol_targeting.py            # 全量
  python3 research/studies/c26_vol_targeting.py --dev      # 调试 (不写 .out)
"""
import hashlib
import os
import sys
import time
from datetime import date
from math import erf, sqrt

# 仓库根入 path (脚本以 `python3 research/studies/c26_vol_targeting.py` 直接
# 运行时, sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.caliber import MIN_GBM_SEEDS, MIN_N, T, W
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching

DEV_MODE = "--dev" in sys.argv

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "main_tf": "1h",
    "W": W,                            # 结果窗口 24 (caliber)
    "T": T,                            # 1:1 对称目标 1.0×ATR
    "stride": 24,                      # 排程间距 (每 24 根一笔)
    "warmup": 600,                     # make_ctx 截断起点
    "gbm_seeds": 3 if DEV_MODE else MIN_GBM_SEEDS,
    "dev_n_sym": 3 if DEV_MODE else 20,
    "cost_taker": 0.0005,              # taker 0.05% (建仓)
    "by_year_list": (2024, 2025, 2026),
    "holdout": {"year": 2026, "months": (6, 7, 8)},
    "data_range": "2023-08..2026-08",
}
STUDY_ID = "c26_vol_targeting"

# R 值映射 (1:1 引擎): win=+1, loss=−1, expired/skip=0
RCODE = {"win": 1, "loss": -1, "expired": 0, "skip": 0}


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


def lead_float(arr, j):
    """out[t] = arr[t+j] (t+j<n), 否则 NaN — 掩码实现, 无切片"""
    n = len(arr)
    out = np.full(n, np.nan)
    m = np.arange(n) < n - j
    out[m] = arr[np.arange(n)[m] + j]
    return out


# ── 单标的管线 ───────────────────────────────────────────────
def run_symbol(ctx, months, params):
    """排程入场 + 1:1 → 逐事件 (R, atr_rel, 实现波动, year, month)

    返回 dict: R (每笔), ar (ATR_rel[entry]), rv (后续 W 根 |logret| 均值),
    yr, mon (entry bar), e_idx (排序用)。
    """
    n = ctx.n
    t_idx = np.arange(n)
    sched = (t_idx % params["stride"] == 0) & (t_idx < n - params["W"])
    out, recs = evaluate_forward(ctx.close, ctx.high, ctx.low, ctx.atr, sched,
                                 direction="long", t_mult=params["T"],
                                 w=params["W"], open_px=ctx.open)
    atr_rel = ctx.atr / np.maximum(ctx.close, 1e-12)
    # 实现波动: 后续 W 根 |logret| 均值 (lead 累加, 掩码)
    logr = np.zeros(n)
    m1 = np.arange(n) >= 1
    logr[m1] = np.abs(np.log(
        ctx.close[m1] / np.maximum(ctx.close[np.arange(n)[m1] - 1], 1e-12)))
    acc = np.zeros(n)
    for j in range(1, params["W"] + 1):
        acc = acc + lead_float(logr, j)
    rv = acc / params["W"]

    Rs = []
    ars = []
    rvs = []
    yrs = []
    mons = []
    eids = []
    for r in recs:
        Rs.append(RCODE[r.outcome])
        eids.append(r.entry_idx)
        ars.append(atr_rel[r.entry_idx])
        rvs.append(rv[r.entry_idx])
        yrs.append(ctx.years[r.entry_idx])
        mons.append(months[r.entry_idx])
    return {"R": np.array(Rs, float), "ar": np.array(ars, float),
            "rv": np.array(rvs, float), "yr": np.array(yrs, int),
            "mon": np.array(mons, int), "e_idx": np.array(eids, int)}


def pool_events(parts):
    """多标的/多种子事件拼接 (跨标的合计, 权益按 e_idx 排序串接)"""
    R = np.concatenate([p["R"] for p in parts])
    ar = np.concatenate([p["ar"] for p in parts])
    rv = np.concatenate([p["rv"] for p in parts])
    yr = np.concatenate([p["yr"] for p in parts])
    mon = np.concatenate([p["mon"] for p in parts])
    eidx = np.concatenate([p["e_idx"] for p in parts])
    return {"R": R, "ar": ar, "rv": rv, "yr": yr, "mon": mon, "e_idx": eidx,
            "n": len(R)}


def pl_a(ev):
    return ev["R"] * ev["ar"]


def pl_a_same(ev, e_ar):
    """A 同平均仓位版: R × ATR_rel / E[ATR_rel] (均仓位 1, 与 B 同尺度)"""
    return ev["R"] * ev["ar"] / e_ar


def pl_b(ev):
    return ev["R"] * 1.0


# ── 度量 ─────────────────────────────────────────────────────
def corr_rel(ev):
    """H1: corr(ATR_rel(entry), 后续实现波动) — 池内逐事件"""
    m = np.isfinite(ev["ar"]) & np.isfinite(ev["rv"]) & (ev["rv"] > 0)
    if m.sum() < 3:
        return float("nan"), 0
    return float(np.corrcoef(ev["ar"][m], ev["rv"][m])[0, 1]), int(m.sum())


def sd_reduce(ev, e_ar):
    """H2: 降低率 = (sd_A' − sd_B) / sd_A' (A' 同平均仓位, 同尺度)"""
    a = pl_a_same(ev, e_ar)
    b = pl_b(ev)
    sd_a = float(np.std(a))
    sd_b = float(np.std(b))
    if sd_a <= 0:
        return float("nan"), float("nan"), float("nan")
    return sd_a, sd_b, (sd_a - sd_b) / sd_a


def max_dd(ev, pnl):
    """H3: 权益曲线 (按 e_idx 全局排序串接) 最大峰值回撤 (直接 P&L 单位)

    A' 与 B 同尺度 (均仓位 1), 直接 MDD 可比; 跨真实/GBM 的事件数差异
    (20 标的 vs 30 种子) 影响 MDD 绝对量, 净差方向由预注册口径判定
    (局限中注明)。
    """
    order = np.argsort(ev["e_idx"], kind="stable")
    p = pnl[order]
    eq = np.cumsum(p)
    peak = np.maximum.accumulate(eq)
    return float(np.max(peak - eq))


def h4_z(ev, e_ar=None):
    """H4: z = (mean_B − mean_A') / sqrt(var_B/n_B + var_A'/n_A') (A' 同尺度)"""
    if e_ar is None:
        return None
    a = pl_a_same(ev, e_ar)
    b = pl_b(ev)
    na = len(a)
    nb = len(b)
    if na < 2 or nb < 2:
        return None
    se = sqrt(float(np.var(b)) / nb + float(np.var(a)) / na)
    if se <= 0:
        return None
    z = (float(np.mean(b)) - float(np.mean(a))) / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))  # 双侧
    return z, p


def cost_b(ev):
    """B 每笔建仓成本 = 0.05% × (1/ATR_rel) (相对 close 计价)"""
    c = PARAMS["cost_taker"] / np.maximum(ev["ar"], 1e-12)
    return float(np.sum(c)), len(c)


def _e(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:+.4f}"


def _pct(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:.2f}%"


def _pp(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:+.2f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


# ── GATE 自检 (违规即停) ─────────────────────────────────────
def gate(real_wr, gbm_wr, gbm_corr, gbm_n, params):
    """①1:1 无条件基线 (真实+GBM long, GBM ∈ [49%,51%])
    ②GBM 侧 H1 相关性 ≈0 (±0.05 容差) ③MIN_N — 任一失败 SystemExit
    (dev 模式跳过收敛断言)"""
    print(f"[GATE] 1:1 无条件 long: 真实 WR {real_wr * 100:.2f}% | "
          f"GBM WR {gbm_wr * 100:.2f}% | GBM H1 相关性 {gbm_corr:+.4f}",
          flush=True)
    if not DEV_MODE:
        if not (49.0 <= gbm_wr * 100 <= 51.0):
            raise SystemExit(
                f"GATE FAIL: GBM 无条件 WR {gbm_wr:.4f} ∉ [49%, 51%] — 停")
        if not (-0.05 <= gbm_corr <= 0.05):
            raise SystemExit(
                f"GATE FAIL: GBM H1 相关性 {gbm_corr:.4f} ∉ ±0.05 — "
                f"波动记忆在 GBM 上存在, 停")
        if gbm_n < MIN_N:
            raise SystemExit(f"GATE FAIL: GBM n={gbm_n} < MIN_N={MIN_N}, 停")
    else:
        print("[GATE] dev 模式: 收敛断言跳过 (30 种子性质); 仅报告", flush=True)
    return {"gbm_corr": gbm_corr}


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR/HOLDOUT) ────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, res, year_rows, holdout_rows):
    p = params
    tf1 = p["main_tf"]
    r = res[tf1]["real"]
    gm = res[tf1]["gbm"]
    e_ar = float(np.mean(r["ar"])) if r["n"] else float("nan")
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},main={},W={},T={},stride={},warmup={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], ",".join(p["tf_list"]), p["main_tf"],
            p["W"], p["T"], p["stride"], p["warmup"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(1:1 long 排程入场) 真实 {:.2f}% "
        "GBM {:.2f}% [t1:1 PASS]; 探测器: GBM H1 相关性 {:.4f} ∈±0.05 "
        "[PASS]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], res[tf1]["real_wr"] * 100,
            res[tf1]["gbm_wr"] * 100,
            g["gbm_corr"], gm["n"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 仓位管理 (Phase 2.5); "
        "入场=无条件 long 固定排程 (每 24 根一笔, 非重叠, 无方向 edge 基座); "
        "A=固定仓位 P&L=R×ATR_rel, B=波动目标 P&L=R×1 (K=1 风险恒定); "
        "R 单位=ATR (1:1 evaluate_forward); GBM = 首标×30 种子同管线",
        "",
        "[门槛] H2 净差≥10pp 且 H3 过 (真实−GBM<0) 且分年 ≥2/3 年同向 → 达标",
        "",
    ]

    # H1
    rc, rn = corr_rel(r)
    gc, gn = corr_rel(gm)
    lines.append("[H1] 前提: corr(ATR_rel(entry), 后续W根实现波动) 1h:")
    lines.append("  真实 {} (n={}) | GBM {} (n={}) | 净差 {}{}".format(
        _e(rc), rn, _e(gc), gn, _e(rc - gc) if np.isfinite(rc) and np.isfinite(gc) else "-",
        _nm(rn)))
    lines.append("")

    # H2
    lines.append("[H2主端点] 每笔 P&L 标准差降低率 1h (sd_A − sd_B)/sd_A:")
    rsa, rsb, rr_ = sd_reduce(r, e_ar)
    gsa, gsb, gr_ = sd_reduce(gm, e_ar)
    net = (rr_ - gr_) if np.isfinite(rr_) and np.isfinite(gr_) else float("nan")
    lines.append("  真实 sd_A {} sd_B {} 降低率 {} (n={}) | GBM sd_A {} "
                 "sd_B {} 降低率 {} (n={}) | 净差 {}{}".format(
        _e(rsa), _e(rsb), _pct(rr_), r["n"], _e(gsa), _e(gsb), _pct(gr_),
        gm["n"], _pp(net), _nm(r["n"])))
    lines.append("")

    # H3
    lines.append("[H3] 累积权益最大回撤 / 每笔 sd (波动单位) 1h:")
    mdd_ra = max_dd(r, pl_a_same(r, e_ar))
    mdd_rb = max_dd(r, pl_b(r))
    mdd_ga = max_dd(gm, pl_a_same(gm, e_ar))
    mdd_gb = max_dd(gm, pl_b(gm))
    net3 = (mdd_rb - mdd_ra) - (mdd_gb - mdd_ga)
    lines.append("  真实: A' {:.3f} vs B {:.3f} (Δ {}) | GBM: A' {:.3f} vs "
                 "B {:.3f} (Δ {}) | 真实−GBM 净差 {}".format(
        mdd_ra, mdd_rb, _e(mdd_rb - mdd_ra), mdd_ga, mdd_gb,
        _e(mdd_gb - mdd_ga), _e(net3)))
    lines.append("")

    # H4
    lines.append("[H4] 收益中性: Σ P&L A vs B 1h (基座均值≈0, z 检验):")
    z4 = h4_z(r, e_ar)
    sum_a = float(np.sum(pl_a_same(r, e_ar)))
    sum_b = float(np.sum(pl_b(r)))
    lines.append("  ΣP&L_A {} | ΣP&L_B {} | mean_A {} mean_B {} | "
                 "z {}{}".format(
        _e(sum_a), _e(sum_b), _e(float(np.mean(pl_a_same(r, e_ar)))),
        _e(float(np.mean(pl_b(r)))),
        _e(z4[0]) if z4 else "-",
        " (p {:.3f})".format(z4[1]) if z4 else ""))
    lines.append("")

    # 成本
    cb, nb = cost_b(r)
    lines.append("[成本] B 建仓成本 (taker 0.05% × B_i, A 无调仓成本基准):")
    lines.append("  Σcost_B {} (n={}) | ΣP&L_B {} | cost/P&L 比 {}".format(
        _e(cb), nb, _e(sum_b),
        _e(cb / sum_b) if abs(sum_b) > 1e-12 else "-"))
    lines.append("")

    # 4h 交叉
    lines.append("[4h交叉] 同样管线 (stride=24 4h = 每 4 天一笔):")
    r4 = res["4h"]["real"]
    g4 = res["4h"]["gbm"]
    rc4, rn4 = corr_rel(r4)
    gc4, gn4 = corr_rel(g4)
    e_ar4 = float(np.mean(r4["ar"])) if r4["n"] else float("nan")
    _, _, rr4 = sd_reduce(r4, e_ar4)
    _, _, gr4 = sd_reduce(g4, e_ar4)
    net4 = (rr4 - gr4) if np.isfinite(rr4) and np.isfinite(gr4) else float("nan")
    lines.append("  H1 相关性: 真实 {} (n={}) vs GBM {} (n={}) | "
                 "H2 降低率: 真实 {} vs GBM {} | 净差 {}{}".format(
        _e(rc4), rn4, _e(gc4), gn4, _pct(rr4), _pct(gr4), _pp(net4),
        _nm(rn4)))
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

    n_sym = PARAMS["dev_n_sym"]

    def pool(tf):
        rparts = []
        for df in dfs[tf][:n_sym]:
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            rparts.append(run_symbol(ctx, months_aligned(df, PARAMS["warmup"]),
                                     PARAMS))
        ref = dfs[tf][0]
        gparts = []
        for s in range(PARAMS["gbm_seeds"]):
            rw = gbm_matching(ref, seed=s)
            ctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gparts.append(run_symbol(ctx, months_aligned(rw, PARAMS["warmup"]),
                                     PARAMS))
        return pool_events(rparts), pool_events(gparts)

    res = {}
    g = None
    for tf in PARAMS["tf_list"]:
        real, gbm = pool(tf)
        # GATE 用 1h: 排程入场 WR
        real_wr = float(np.mean(real["R"] > 0)) if len(real["R"]) else float("nan")
        gbm_wr = float(np.mean(gbm["R"] > 0)) if len(gbm["R"]) else float("nan")
        gc, gn = corr_rel(gbm)
        if tf == "1h":
            g = gate(real_wr, gbm_wr, gc, len(gbm["R"]), PARAMS)
        res[tf] = {"real": real, "gbm": gbm,
                   "real_wr": real_wr, "gbm_wr": gbm_wr,
                   "gbm_corr": gc, "n": len(real["R"])}

    # BY_YEAR (H2 降低率分年, 1h, 真实/GBM/净差 成对)
    by_year_rows = []
    holdout_rows = []
    if not DEV_MODE:
        r1 = res["1h"]["real"]
        g1 = res["1h"]["gbm"]
        e_ar1 = float(np.mean(r1["ar"])) if r1["n"] else float("nan")
        for y in PARAMS["by_year_list"]:
            mr = r1["yr"] == y
            mg = g1["yr"] == y
            if mr.sum() < 2 or mg.sum() < 2:
                continue
            sub_r = {"R": r1["R"][mr], "ar": r1["ar"][mr],
                     "rv": r1["rv"][mr], "yr": r1["yr"][mr],
                     "mon": r1["mon"][mr], "e_idx": r1["e_idx"][mr]}
            sub_g = {"R": g1["R"][mg], "ar": g1["ar"][mg],
                     "rv": g1["rv"][mg], "yr": g1["yr"][mg],
                     "mon": g1["mon"][mg], "e_idx": g1["e_idx"][mg]}
            _, _, rr_y = sd_reduce(sub_r, e_ar1)
            _, _, gr_y = sd_reduce(sub_g, e_ar1)
            net_y = (rr_y - gr_y) if np.isfinite(rr_y) and np.isfinite(gr_y) else float("nan")
            by_year_rows.append(
                "1h {} 真实降低率 {} (n={}) GBM降低率 {} (n={}) 净差 {}".format(
                    y, _pct(rr_y), int(mr.sum()), _pct(gr_y), int(mg.sum()),
                    _pp(net_y)))
        # HOLDOUT (末 3 月, H2 降低率净差方向)
        lo = PARAMS["holdout"]["months"][0]
        hi = PARAMS["holdout"]["months"][1]
        m_ho_r = (r1["yr"] == PARAMS["holdout"]["year"]) & (r1["mon"] >= lo) & (r1["mon"] <= hi)
        m_ho_g = (g1["yr"] == PARAMS["holdout"]["year"]) & (g1["mon"] >= lo) & (g1["mon"] <= hi)
        if m_ho_r.sum() >= 2 and m_ho_g.sum() >= 2:
            sub_r = {"R": r1["R"][m_ho_r], "ar": r1["ar"][m_ho_r],
                     "rv": r1["rv"][m_ho_r], "yr": r1["yr"][m_ho_r],
                     "mon": r1["mon"][m_ho_r], "e_idx": r1["e_idx"][m_ho_r]}
            sub_g = {"R": g1["R"][m_ho_g], "ar": g1["ar"][m_ho_g],
                     "rv": g1["rv"][m_ho_g], "yr": g1["yr"][m_ho_g],
                     "mon": g1["mon"][m_ho_g], "e_idx": g1["e_idx"][m_ho_g]}
            _, _, rr_ho = sd_reduce(sub_r, e_ar1)
            _, _, gr_ho = sd_reduce(sub_g, e_ar1)
            net_ho = (rr_ho - gr_ho) if np.isfinite(rr_ho) and np.isfinite(gr_ho) else float("nan")
            holdout_rows.append(
                "主端点 H2 降低率净差 2026-06..08: 真实 {} (n={}) GBM {} "
                "(n={}) | 净差 {} (只报方向: {})".format(
                    _pct(rr_ho), int(m_ho_r.sum()), _pct(gr_ho),
                    int(m_ho_g.sum()), _pp(net_ho),
                    "正" if (np.isfinite(net_ho) and net_ho > 0)
                    else "负/不可判"))

    if DEV_MODE:
        print("=== DEV 模式: 不写 .out ===")
        r1 = res["1h"]["real"]
        g1 = res["1h"]["gbm"]
        e_ar1 = float(np.mean(r1["ar"])) if r1["n"] else float("nan")
        rc, rn = corr_rel(r1)
        gc, gn = corr_rel(g1)
        print(f"[H1] 相关 真实 {rc:+.4f} (n={rn}) | GBM {gc:+.4f} (n={gn})")
        rsa, rsb, rr_ = sd_reduce(r1, e_ar1)
        gsa, gsb, gr_ = sd_reduce(g1, e_ar1)
        print(f"[H2] 降低率 真实 {rr_*100:.1f}% | GBM {gr_*100:.1f}% | 净差 {(rr_-gr_)*100:+.1f}pp")
        print(f"[H3] MDD 真实 A' {max_dd(r1, pl_a_same(r1, e_ar1)):.3f} B {max_dd(r1, pl_b(r1)):.3f}")
        z4 = h4_z(r1, e_ar1)
        print(f"[H4] z {z4[0]:+.2f} p {z4[1]:.3f}" if z4 else "[H4] 样本不足")
        print(f"运行耗时: {time.time() - t0:.1f}s (dev)")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, res, by_year_rows, holdout_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
