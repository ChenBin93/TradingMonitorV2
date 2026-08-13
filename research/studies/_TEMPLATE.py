#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C{id} {标题} ({date}, 无未来函数, {周期})

================================================================
预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1 {假设}: 若成立, 应观察到 {具体可证伪数字, 如 真实−GBM ≥ +3pp}
  H2 {假设}: 若成立, 应观察到 {具体可证伪数字}
  (运行前填写; 结论必须逐条回应: 支持 / 否定 / 无法判断)

================================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/数据        | 计算方式                       | 可用时点        | 依据
  close/high/low/  | research.ctx.make_ctx 统一     | bar 收盘后      | ctx 唯一截断对齐出口
  open/atr/years   |  截断对齐 (内部 iloc[warmup:])  |                 | (禁一切手动切片)
  滚动分位/排位     | causal.rolling_percentile/     | 尾窗已收盘      | research.causal
                   |  rolling_rank                   |                 |
  事后标签条件化   | causal.causal_confirmed         | conf∈[t-60,t-24]| research.causal
                   |  (conf 窗口内突破剔除)          |                 |
  在线聚类+冻结    | causal.frozen_cluster           | 冻结时刻        | research.causal
   状态序列         | ctx state_fns 在截断 df 上计算   | 已收盘 bar     | make_ctx 契约
   GBM 无信息对照   | sim_market.gbm_matching(ref,seed)| 锚定真实索引/长度 | 固定种子序列 ≥30

算法代码注意事项 (c12 试点摩擦, 已有惯例):
   - check_study 禁止一切数组切片 (拦截 WARMUP 对齐类 bug, A3 教训) — 算法内部
     移位/取窗请用布尔掩码 (如 x[m1]*x[m2]、y[keep].reshape()), 不要用 x[1:]/x[:-lag]
   - 描述层 c1x (无入场) 的 gate() 与 .out GATE 行需改写: 没有 1:1 WR, 用
     "探测器自检 (白噪声≈0.5) + GBM≥30种子同管线 null" 断言; .out GATE 行必须
     包含 `gbm_seeds=` / `无条件基线` / `MIN_N` 三个 token 才能过 check_study ④
     (描述层会触发 1 个 ④WARN, 属预期, 不 FAIL)

[DESCRIPTIVE] 分区 (仅描述层 c1x 需要时保留本段, 否则删除整段):
  以下结论为纯描述 (事后统计), 禁止进入交易含义:
  - {描述性统计项}: {说明}
  (注意: docstring 含 [DESCRIPTIVE] 会豁免 check_study 的 percentile/quantile
   全样本分位检查 — 描述分区之外的因果特征区仍必须走 rolling_percentile)

================================================================
数据声明:
  data/backtest.db (gitignored): 20 标的 × 5m/1h/4h × 3 年
  (2023-08 → 2026-08, 约 690 万行); 时间戳 = bar 开盘时间 (UTC);
  研究只用已收盘 bar; 与 live 的 data/history.db 完全独立。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。

发布门槛自检 (任何"正期望/edge/有效"结论必须全绿):
  - GATE 无条件基线 ≈50%±1pp (GBM ≥ MIN_GBM_SEEDS 种子, 同管线重放)
  - 真实 − RW 净差 > 0 且分年 ≥2/3 年同号
  - 每格 n ≥ MIN_N={MIN_N}
  - 结论每个数字带 (.out:L行号) 引用; 结论↔.out↔脚本 script_sha256 三重一致
  - 成本核算 (策略层必须; 成本后 ≤0 只能写"结构发现"非 edge)

运行命令:
  # 两道门禁: 引擎门禁 → 脚本门禁 → 运行
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/{id}.py
  python3 research/studies/{id}.py
"""
import hashlib
import os
import sys
from datetime import date

# 仓库根入 path (脚本以 `python3 research/studies/xxx.py` 直接运行时, sys.path[0]=脚本目录,
# 需手动补根 — c12 试点发现的模板硬伤)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.caliber import MIN_GBM_SEEDS, MIN_N, T, W
from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.outcome import Outcome, evaluate_forward, report_wr
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "1h",             # 周期 (5m/1h/4h)
    "W": W,                 # 结果窗口 (根)
    "T": T,                 # 对称目标 ×ATR (1:1 口径)
    "direction": "long",    # 方向 (1:1 研究一次跑一个方向)
    "warmup": 500,          # 特征 warm-up (覆盖 rolling 前 window-1 根 NaN)
    "gbm_seeds": MIN_GBM_SEEDS,
    "data_range": "2023-08..2026-08",  # 运行后按实际数据修正
}

STUDY_ID = "c{id}"


# ── 特征 / 入场 (研究在此实现) ────────────────────────────────
def make_features(ctx, params):
    """特征构建 — 全部走 ctx 对齐数组 (ctx.close/high/low/open/atr/years/states).

    只允许 causal.rolling_percentile/rolling_rank/rolling (尾窗), 禁 np.percentile;
    事后标签条件化一律 causal.causal_confirmed。
    """
    atr_z = rolling_percentile(ctx.atr, window=120, q=50, min_periods=60)
    return {"atr_z": atr_z}


def make_entries(ctx, feats, params):
    """入场 — 返回全长度布尔 (len == ctx.n).

    无条件基线: np.ones(ctx.n, bool);
    事件式入场: ctx.entries_from_events(states, target) (保证全长度布尔)。
    """
    entries = np.ones(ctx.n, bool)
    return entries


# ── 管线 (gate 与 main 共用同一管线 = 同管线重放) ────────────
def run_one(df, params):
    """单标的管线: make_ctx → 特征 → 入场 → evaluate_forward (1:1) → (out, 分年)"""
    ctx = make_ctx(df, params["warmup"], state_fns={})
    feats = make_features(ctx, params)
    entries = make_entries(ctx, feats, params)
    out, recs = evaluate_forward(
        ctx.close, ctx.high, ctx.low, ctx.atr, entries,
        direction=params["direction"], t_mult=params["T"], w=params["W"],
        open_px=ctx.open)
    year_wl = {}
    for r in recs:
        if r.outcome in ("win", "loss"):
            y = ctx.years[r.entry_idx]
            year_wl.setdefault(y, [0, 0])
            year_wl[y][0 if r.outcome == "win" else 1] += 1
    return out, year_wl


def aggregate(results):
    """多标的 Outcome 汇总 + 分年合计"""
    total = Outcome()
    year_wl = {}
    for out, by_year in results:
        total.n_win += out.n_win
        total.n_loss += out.n_loss
        total.n_expired += out.n_expired
        total.n_skip += out.n_skip
        for y, (w, l) in by_year.items():
            year_wl.setdefault(y, [0, 0])
            year_wl[y][0] += w
            year_wl[y][1] += l
    return total, year_wl


# ── GATE 自检 (违规即停) ─────────────────────────────────────
def gate(dfs, params):
    """与研究同管线重放的无条件基线: 真实 + GBM ≥MIN_GBM_SEEDS 固定种子序列.

    同一管线 (make_ctx→make_features→make_entries→evaluate_forward) 而非只调引擎,
    暴露研究脚本自身的管线级错位。GBM 无条件 WR ∉ [49%, 51%] → SystemExit。
    """
    def wr_of(df):
        out, _ = run_one(df, params)
        return out.win_rate

    real_wrs = [wr_of(df) for df in dfs]
    gbm_wrs = [wr_of(gbm_matching(dfs[0], seed=seed))
               for seed in range(params["gbm_seeds"])]
    real_wr = float(np.mean(real_wrs))
    gbm_wr = float(np.mean(gbm_wrs))
    print(f"[GATE] 无条件基线 真实 {real_wr:.1%} | GBM {gbm_wr:.1%} "
          f"(gbm_seeds={len(gbm_wrs)}, ≥{MIN_GBM_SEEDS})")
    if len(gbm_wrs) < MIN_GBM_SEEDS:
        raise SystemExit(f"GATE FAIL: gbm_seeds={len(gbm_wrs)} < {MIN_GBM_SEEDS}")
    if not 49.0 <= gbm_wr * 100 <= 51.0:
        raise SystemExit(
            f"GATE FAIL: GBM 无条件基线 {gbm_wr:.1%} ∉ [49%, 51%] — 口径偏置, 停")
    return real_wr, gbm_wr


# ── 分年成对 (真实 + GBM) ────────────────────────────────────
def run_by_year(dfs, params):
    """每条件结论的 真实分年 + GBM分年 成对输出 (GBM 逐种子同管线重放)."""
    def by_year(df):
        out, yw = run_one(df, params)
        return yw

    real, gbm = {}, {}
    for df in dfs:
        for y, (w, l) in by_year(df).items():
            real.setdefault(y, [0, 0])
            real[y][0] += w
            real[y][1] += l
    for seed in range(params["gbm_seeds"]):
        for y, (w, l) in by_year(gbm_matching(dfs[0], seed=seed)).items():
            gbm.setdefault(y, [0, 0])
            gbm[y][0] += w
            gbm[y][1] += l
    years = sorted(set(real) | set(gbm))
    return [(y, real.get(y, [0, 0]), gbm.get(y, [0, 0])) for y in years]


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _wr(wl):
    return float("nan") if not wl or wl[0] + wl[1] == 0 else wl[0] / (wl[0] + wl[1])


def write_out(out_path, params, real_wr, gbm_wr, total, year_pairs):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},W={},T={},direction={},warmup={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["tf"], p["W"], p["T"], p["direction"],
            p["warmup"], p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线 真实{:.1%} GBM{:.1%} "
        "[t1:1 PASS] MIN_N [PASS]".format(p["gbm_seeds"], real_wr, gbm_wr),
        "# RESULTS: n={} WR {:.1%} (win {} / loss {} / 过期 {} / 跳过 {})".format(
            total.n_eval, total.win_rate, total.n_win, total.n_loss,
            total.n_expired, total.n_skip),
    ]
    if year_pairs:
        parts = ["{} 真实{:.1%} GBM{:.1%}".format(y, _wr(ry), _wr(gy))
                 for y, ry, gy in year_pairs]
        lines.append("# BY_YEAR: " + " | ".join(parts))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"written: {out_path}")


# ── main ─────────────────────────────────────────────────────
def main():
    dfs = []
    data = load_candles(timeframes=(PARAMS["tf"],))
    for sym, tfs in data.items():
        df = tfs.get(PARAMS["tf"])
        if df is None or verify(df, sym, PARAMS["tf"]):
            continue
        dfs.append(df)
    if not dfs:
        print("无数据, 退出")
        return 1

    # GATE 自检 (失败 SystemExit — 违规即停)
    real_wr, gbm_wr = gate(dfs, PARAMS)

    # 全量结果
    results = [run_one(df, PARAMS) for df in dfs]
    total, year_wl = aggregate(results)

    # 分年成对 (真实+GBM)
    year_pairs = run_by_year(dfs, PARAMS)

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, real_wr, gbm_wr, total, year_pairs)
    print(report_wr(total, f"{STUDY_ID} {PARAMS['direction']}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
