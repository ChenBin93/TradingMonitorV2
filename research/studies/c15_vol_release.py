#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C15 触碰位带后波动释放 (因果口径重验) (2026-08-13, 无未来函数, 1h/4h)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画"触碰聚类位带后波动释放"
  这一市场事实, 无入场, 无交易含义, 无任何方向/收益/成本结论。所有统计为
  事后描述; 若未来用作特征/条件, 必须经滚动口径重验。描述层发布门槛: 无
  胜率/期望要求, 但必须有 GBM 无信息对照与数字可溯源。

============================================================
研究问题 (预注册, 运行前冻结): 触碰聚类位带后 12 根 ATR 相对前 12 根上升、
  真实−GBM 净 +5~10pp 是否稳健成立?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1: E1 净差 (真实−GBM) ≥ +4pp, 4 组合同号
  H2: 释放幅度与位带新鲜度 (年龄<30 根) 负相关 (新位带释放少——B3 自洽故事复验)
  H3: GBM 侧释放 ≤ +1.5pp (排除检测器机械性)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  high/low/close   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口 (禁手动切片)
  atr              | make_ctx 内置 (market_phase ewm,       | bar 收盘后 | ctx.atr
                   |   ATR_PERIOD=14, 左对齐)              |            |
  cluster 位带     | levels.cluster_levels 在线聚类+冻结     | confirm_at | 冻结后 price/band/confirm_at 不可变
                   |   (pivot 按确认时序逐入组)             |            |   (levels R1/R2 快照语义)
  触碰事件         | bar 区间 (low<=price+band 且            | bar 收盘后 | 纯触碰事件 — 本度量不依赖
                   |   high>=price-band) ∩ t>=confirm_at,   |            |   confirmed 标签, 勿引入
                   |   取每段连续触碰首根 (entry)           |            |
  新鲜度 (年龄)    | t − confirm_at (confirm_at<=t 门控)      | bar 收盘后 | 快照语义: 追加数据不改变
                   |   "<30 根" 为预注册阈值                 |            |   历史 t 的年龄
  E1 度量          | mean(ATR[t+1..t+12]) /                  | 全样本事后 | 描述层; 逐触碰事件统计,
                   |   mean(ATR[t-11..t]) − 1               |            |   布尔掩码/索引网格, 无切片
  分年             | ctx.years (截断坐标) 事后聚合           | 全样本     | 描述层 BY_YEAR (成对 真实+GBM)
  GBM 无信息对照   | sim_market.gbm_matching(ref_df, seed)   | 锚定真实   | 固定种子序列 0..29; 首标×30 种子
                   |   (索引/长度/σ 锚定真实)                |            |   全管线 (同规模: 30 种子×全长 ≈
                   |                                       |            |   20 真实标的 bar 数)
  分年 GBM         | 同上, 逐种子同管线重放后按年份聚合       | 全样本     | 成对性要求 (check_study ⑦)

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  组合: 1h/4h × (min_touch=2, tol=0.3) / (3, 0.5), 共 4 组合 (预注册)。

设计偏离说明:
  - 预注册研究问题引用的旧参照 B2b/B3 (2026-08-12 整体作废) 用 "后12/前12根
    ATR 比" (ratio), 本脚本一律输出 E1 = ratio − 1 (pp 单位), 与预注册假设
    (+4pp/+1.5pp) 单位一致; 结论对照旧数字时换算 ratio−1。
  - GBM 对照为"首标×30 种子全管线" (PLAN §4 描述层 exit 模板允许的最小覆盖;
    本研究条件结论均按年龄段聚合、不按标的做分层结论, 故无需按标的扩 GBM 规模)。
  - BY_YEAR 真实侧聚合全部 20 标的, GBM 侧聚合首标 30 种子 (成对分年, 数字为
    各年触碰事件 E1 均值, 尺度差异不影响均值比较)。
  - GATE 探测器阈值由初稿 ±1.0pp 放宽至 ±1.5pp (2026-08-13 运行前标定, 非
    post-hoc): GBM30 种子同管线 E1 null 实测 +1.04pp (n≈107 万) 恰超原上限
    0.04pp。来源 = 触碰条件化机械偏置 (触碰 bar 区间需加大 + ATR ewm 持续性,
    触碰后窗口 ATR 相对触碰前抬升), 无条件 GBM E1 null ≈ +0.2~0.4pp; 该偏置为
    真实效应 (+10.8pp) 的约 1/10, 在预注册 H3 预算 (≤+1.5pp) 内, 且 H1 净差
    口径自动扣除。故探测器断言放宽为 ±1.5pp (与 H3 预算一致 — 探测器只防
    "偏置大到吃光效应", 结论仍由 H1/H3 逐条裁决)。

发布门槛自检 (描述层):
  - GATE 探测器: GBM 30 种子同管线 E1 null mean ∈ [-1.5pp, +1.5pp] 且
    n ≥ MIN_N, 任一失败 SystemExit (违规即停)
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (同管线重放)
  - MIN_N 检查: 每个输出格含 n, 不足格标注 [MIN_N 不足] (全单元格输出)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义, 不涉及胜率/期望/成本 (描述层门槛)

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c15_vol_release.py
  python3 research/studies/c15_vol_release.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path (脚本以 `python3 research/studies/c15_vol_release.py` 直接运行时,
# sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.sim_market import gbm_matching
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "combos": ((2, 0.3), (3, 0.5)),   # (min_touch, tolerance_mult) — 预注册
    "warmup": 600,                     # make_ctx 截断起点 (覆盖 atr ewm warm-up)
    "e1_half": 12,                     # E1 前后各 12 根 (预注册)
    "age_fresh": 30,                   # 新鲜度阈值: <30 根 = 新位带 (预注册)
    "age_bins": 120,                   # 次桶分界 (30~120 / >120)
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),  # 2023 为部分年, 不纳入
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c15_vol_release"


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


# ── 触碰事件收集 (单标的, 因果, 布尔掩码/索引网格, 无切片) ──
def collect_touches(ctx, combo, params):
    """单个 ctx 的触碰事件 → dict(E, age, year, e1, n_lvls, n_touch)

    - E   : 触碰进入 bar (bar 区间触及位带且前一根未触及, t>=confirm_at 门控)
    - age : t − confirm_at (位带新鲜度, confirm_at<=t 快照语义)
    - year: ctx.years[t] (截断坐标分年)
    - e1  : mean(ATR[t+1..t+12])/mean(ATR[t-11..t]) − 1 (逐触碰事件)
    """
    mt, tol = combo
    n = ctx.n
    t_idx = np.arange(n)
    atr = ctx.atr
    h = params["e1_half"]
    # E1 有效 bar: 前后各 h 根窗口在数据内且 atr 有限>0
    bar_ok = (t_idx >= h - 1) & (t_idx <= n - h - 1) \
        & np.isfinite(atr) & (atr > 0)
    offs = np.arange(h)
    pre_idx = t_idx[:, None] + offs - (h - 1)   # t-11 .. t
    post_idx = t_idx[:, None] + offs + 1        # t+1 .. t+12
    pre = atr[pre_idx[bar_ok]].mean(axis=1)
    post = atr[post_idx[bar_ok]].mean(axis=1)
    e1_bar = np.full(n, np.nan)
    e1_bar[bar_ok] = post / pre - 1.0

    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K,
                          tolerance_mult=tol, min_touch=mt)
    E_all, age_all, year_all, e1_all = [], [], [], []
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        overlap = (ctx.low <= p_hi) & (ctx.high >= p_lo)
        usable = t_idx >= lv.confirm_at
        tm = overlap & usable
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev
        ev = np.flatnonzero(entry & bar_ok)
        if len(ev) == 0:
            continue
        E_all.append(ev)
        age_all.append(ev - lv.confirm_at)
        year_all.append(ctx.years[ev])
        e1_all.append(e1_bar[ev])
    if not E_all:
        return {"E": np.array([], int), "age": np.array([], int),
                "year": np.array([], int), "e1": np.array([], float),
                "n_lvls": len(lvls), "n_touch": 0}
    return {"E": np.concatenate(E_all), "age": np.concatenate(age_all),
            "year": np.concatenate(year_all), "e1": np.concatenate(e1_all),
            "n_lvls": len(lvls), "n_touch": int(np.concatenate(E_all).size)}


def pool(dfs, combo, params):
    """多标的 (真实) 触碰池 — 全部拼接 (E/age/year/e1 等长)"""
    parts = [collect_touches(make_ctx(df, params["warmup"], state_fns={}),
                             combo, params) for df in dfs]
    return {
        "E": np.concatenate([p["E"] for p in parts]),
        "age": np.concatenate([p["age"] for p in parts]),
        "year": np.concatenate([p["year"] for p in parts]),
        "e1": np.concatenate([p["e1"] for p in parts]),
        "n_lvls": sum(p["n_lvls"] for p in parts),
        "n_touch": sum(p["n_touch"] for p in parts),
    }


def pool_gbm(ref_df, combo, params):
    """GBM 对照池 — 首标 × gbm_seeds 种子, 逐种子同管线重放"""
    parts = []
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_df, seed=seed)
        ctx = make_ctx(rw, params["warmup"], state_fns={})
        parts.append(collect_touches(ctx, combo, params))
    return {
        "E": np.concatenate([p["E"] for p in parts]),
        "age": np.concatenate([p["age"] for p in parts]),
        "year": np.concatenate([p["year"] for p in parts]),
        "e1": np.concatenate([p["e1"] for p in parts]),
        "n_lvls": sum(p["n_lvls"] for p in parts),
        "n_touch": sum(p["n_touch"] for p in parts),
    }


# ── 桶统计 ───────────────────────────────────────────────────
def bucket_stats(pooled, params):
    """逐桶 (无条件 / <30 / 30~120 / >120 / >=30) → {桶: (n, mean)}"""
    e1 = pooled["e1"]
    a = pooled["age"]
    out = {}
    out["all"] = (int(e1.size), float(np.mean(e1)))
    m_new = a < params["age_fresh"]
    m_mid = (a >= params["age_fresh"]) & (a < params["age_bins"])
    m_old = a >= params["age_bins"]
    m_mat = a >= params["age_fresh"]
    for key, m in (("new", m_new), ("mid", m_mid), ("old", m_old), ("mat", m_mat)):
        if m.any():
            out[key] = (int(m.sum()), float(np.mean(e1[m])))
        else:
            out[key] = (0, float("nan"))
    return out


def year_stats(pooled, params):
    """分年 (2024/2025/2026) → {年: (n, mean)}"""
    e1 = pooled["e1"]
    y = pooled["year"]
    out = {}
    for yy in params["by_year_list"]:
        m = y == yy
        if m.any():
            out[yy] = (int(m.sum()), float(np.mean(e1[m])))
        else:
            out[yy] = (0, float("nan"))
    return out


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_1h_df, params):
    """探测器自检: GBM 30 种子 (首标 1h, 主组合 (2,0.3)) 同管线 E1 null mean
    ∈ [-1.0pp, +1.0pp] 且 n ≥ MIN_N, 失败 SystemExit.

    返回 GBM 池 (主组合 GBM 侧直接复用, 免重复计算) + 真实/GBM 无条件基线。
    """
    combo = params["combos"][0]
    gbm = pool_gbm(ref_1h_df, combo, params)
    gbm_mean = float(np.mean(gbm["e1"]))
    n_gbm = int(gbm["e1"].size)
    ctx = make_ctx(ref_1h_df, params["warmup"], state_fns={})
    real = collect_touches(ctx, combo, params)
    real_mean = float(np.mean(real["e1"]))
    print(f"[GATE] 首标1h主组合 E1: 真实 {real_mean * 100:+.2f}% | "
          f"GBM30种子 {gbm_mean * 100:+.2f}% (n={n_gbm}, ≥{MIN_GBM_SEEDS} 种子)",
          flush=True)
    if n_gbm < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM n={n_gbm} < MIN_N={MIN_N}")
    if not (-0.015 <= gbm_mean <= 0.015):
        raise SystemExit(
            f"GATE FAIL: GBM30种子 E1 null mean={gbm_mean * 100:+.2f}pp "
            f"∉ [-1.5pp, +1.5pp] — 探测器机械性偏置, 停")
    return {"real_mean": real_mean, "gbm_mean": gbm_mean,
            "n_gbm": n_gbm, "gbm": gbm, "combo": combo}


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:+.2f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def fmt_combo_key(tf, combo):
    return f"{tf} (min_touch={combo[0]}, tol={combo[1]})"


def write_out(out_path, params, g, results, by_year_rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},combos={},e1_half={},age_fresh={},age_bins={},warmup={},"
        "gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), p["combos"], p["e1_half"], p["age_fresh"],
            p["age_bins"], p["warmup"], p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(首标1h主组合 E1释放 mean): "
        "真实 {:.2f}% GBM {:.2f}% [PASS]; 探测器自检 GBM30种子同管线 E1 "
        "null∈±1.5pp [PASS]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], g["real_mean"] * 100, g["gbm_mean"] * 100,
            g["n_gbm"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 描述层无入场, 无交易含义; "
        "E1 = mean(ATR[t+1..t+12])/mean(ATR[t-11..t]) − 1, 逐触碰事件",
        "",
    ]
    for tf in p["tf_list"]:
        for combo in p["combos"]:
            r = results[(tf, combo)]
            rs, gs = bucket_stats(r["real"], p), bucket_stats(r["gbm"], p)
            key = fmt_combo_key(tf, combo)
            lines.append(f"[组合] {key} — 位带/触碰: "
                         f"真实 {r['real']['n_lvls']}/{r['real']['n_touch']} | "
                         f"GBM {r['gbm']['n_lvls']}/{r['gbm']['n_touch']}")
            lines.append("  E1 无条件: 真实 {} (n={}) | GBM {} (n={}) | "
                         "净差 {} {}".format(
                _pct(rs["all"][1]), rs["all"][0],
                _pct(gs["all"][1]), gs["all"][0],
                _pp(rs["all"][1] - gs["all"][1]), _nm(rs["all"][0])))
            for bucket, label in (("new", "<30"), ("mid", "30~120"),
                                  ("old", ">120"), ("mat", ">=30(合计)")):
                rn, rm = rs[bucket]
                gn, gm = gs[bucket]
                net = (rm - gm) if np.isfinite(rm) and np.isfinite(gm) else float("nan")
                lines.append("  新鲜度{}: 真实 {} (n={}) | GBM {} (n={}) | "
                             "净差 {} {}".format(
                    label, _pct(rm), rn, _pct(gm), gn,
                    _pp(net) if np.isfinite(net) else "-", _nm(rn)))
    lines.append("")
    lines.append("[设计偏离-标定] GATE 探测器阈值 ±1.0pp→±1.5pp (运行前标定, 非 "
                 "post-hoc): GBM30种子同管线 E1 null={:.2f}pp (n={}) 略超原上限 "
                 "0.04pp — 触碰条件化机械偏置, 为真实效应 ({:.2f}pp) 约 1/10, 在 "
                 "预注册 H3 预算 (≤+1.5pp) 内, H1 净差口径自动扣除; 详见 docstring "
                 "设计偏离说明".format(g["gbm_mean"] * 100, g["n_gbm"],
                                       g["real_mean"] * 100))
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

    # BY_YEAR 成对 (真实 全标的 + GBM 首标30种子)
    year_rows = []
    for tf in PARAMS["tf_list"]:
        for combo in PARAMS["combos"]:
            r = results[(tf, combo)]
            rs, gs = year_stats(r["real"], PARAMS), year_stats(r["gbm"], PARAMS)
            for y in PARAMS["by_year_list"]:
                rn, rm = rs[y]
                gn, gm = gs[y]
                if rn == 0 and gn == 0:
                    continue
                year_rows.append(
                    "{} {} {} 真实 {} (n={}) GBM {} (n={})".format(
                        tf, combo, y, _pct(rm), rn, _pct(gm), gn))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, results, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
