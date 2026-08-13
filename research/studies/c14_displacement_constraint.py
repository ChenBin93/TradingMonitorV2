#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C14 触碰聚类位带后位移约束 (因果条件化重验) (2026-08-13, 无未来函数, 1h/4h)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画市场事实 (触碰位带后的
  位移分布约束), 无入场, 无交易含义。M2P50/M2P90 是结果统计量 (事后描述),
  不做任何特征条件化 (禁全样本分位作特征 — 所有特征判定全部走因果口径)。

============================================================
预注册假设 (运行前冻结, 结论逐条回应):
  H1: M2P50Δ ≤ -0.4 ATR (位移约束净差下限), 全结构层同号
  H2: M2P90Δ ≤ -0.5 ATR
  H3: 泄漏敏感性对照 — 因果版 (causal_confirmed) 与 B3 原 alive_at 版
      (confirmed 直接条件化) 结果差异 ≤ 20% (差异>20% 则判定 B3 结论部分
      由泄漏驱动)

研究问题: 触碰聚类位带后 24 根最大位移被约束 (真实−GBM 大负) 在因果
  条件化下是否成立?

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/事件       | 计算方式                             | 可用时点       | 依据
  位带            | levels.cluster_levels 在线聚类+冻结  | 形成即冻结     | research.levels (R1/R2)
  触碰事件        | bar 区间触及 [price-band, price+band]| bar 收盘后     | 与 B3 同口径
  确认突破(事后)  | levels.level_breakdown (w=24)        | 事后标签       | 只用于条件化出口, 见下
  区间内判定-因果 | causal.causal_confirmed(conf, w=24,  | t 时刻已知    | research.causal 唯一条件化
                   |  lag_lo=24, lag_hi=60): known[t]=1  |               | conf∈[t-60,t-24] 且窗口关闭
                   |  ⟺ ∃c∈[t-60,t-24] confirmed[c]&    |               |
                   |  c+24<=t; [t-23,t] 内突破样本剔除   |               | (前缀和掩码, 与 causal 同技)
  M2 位移         | 触碰后 1..24 根 max(away) 位移       | 事后统计       | 参照 B3 精确定义
  GBM 无信息对照  | sim_market.gbm_matching (30 种子)    | 锚定真实       | 固定种子序列 0..29, 同管线
  MIN_N 口径      | 每层 n ≥ 200 才报分位数 (B3 同)      | —              | caliber.MIN_N

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。
  周期/参数组合 (4 组合): 1h/4h × (min_touch,tol) = (2,0.3)/(3,0.5)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。

设计偏离说明 (无 — 完全按预注册; 唯一的实现性选择):
  - "区间内" 的 [t-23,t] 剔除掩码用前缀和 (cumsum 差分) 实现 — 与
    causal_confirmed 内部同技术; causal_confirmed 不返回该掩码, 故脚本
    内实现并注明 (非 searchsorted, 非自写条件化出口)。
  - B3 旧版 alive_at 用 bisect 直接条件化 confirmed 位置 (窗口未关闭也计入)
    — c14 复现为 H3 的泄漏侧对照, 因果版与之一一对比。
  - GBM 对照: 首标 × 30 种子全管线 (位移分布是尺度不变度量, 标的选择影响
    微小); 条件结论按事件分层 (非按标的分层), GBM 覆盖同规模 (30×26k bars
    vs 真实 20×26k bars)。

 发布门槛自检 (描述层):
  - GATE: 无条件基线含 真实 1h(2,0.3) 全部触碰 M2P50 与 GBM 30 种子同管线 null
    (GBM mean ∈ [1.8, 2.6] 且触碰数 ≥10000, 失败 SystemExit); 探测器 = 同管线
    GBM null 本身
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义, 不涉及胜率/期望/成本 (描述层门槛)

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c14_displacement_constraint.py
  python3 research/studies/c14_displacement_constraint.py
"""
import hashlib
import os
import sys
import time
from bisect import bisect_right
from datetime import date

# 仓库根入 path (直接运行需要; 模板摩擦, 见 c12 报告)
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

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "mt_tol": ((2, 0.3), (3, 0.5)),
    "warmup": 600,
    "W": 24,                 # 触碰后窗口
    "depth": 0.5,            # level_breakdown 穿透深度 (B3 同)
    "hold_ratio": 0.5,       # 确认突破保持比例 (B3 同)
    "life": 600,             # 位带活跃期 (B3 同)
    "range_atr": 2.5,        # 区间内判定 (对侧距离, B3 同)
    "wide_atr": 5.0,         # 宽成对判定 (B3 同)
    "lag_lo": 24,            # causal_confirmed 因果窗口下界
    "lag_hi": 60,            # 因果窗口上界 (conf∈[t-60,t-24])
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year": (2024, 2025, 2026),
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c14_displacement_constraint"
LAYERS = ("区间内", "宽成对", "孤立", "孤立(对侧已破)")
ALL_LAYER = "全部触碰"


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


def alive_at_b3(conf_pos, t, range_bars=60):
    """B3 原 alive_at (泄漏对照): 最近 confirmed 位置 < t 且 < t-60 → alive.

    泄漏点: confirmed[c] 在 c+w 才可知, 本函数直接用 c<t 的 confirmed 判存活,
    未检查确认窗口是否在 t 前关闭 — 这正是 B3 被判定泄漏的条件化。
    """
    i = bisect_right(conf_pos, t - 1) - 1
    return not (i >= 0 and conf_pos[i] >= t - range_bars)


def run_symbol_events(df, mt, tol, tf, params):
    """单标的管线: 位带 → 触碰 → M2 → 因果/aliveat 结构层分类.

    返回事件列表 [{m2, layer_c, layer_b, year}]:
      layer_c = 因果版结构层; layer_b = B3 alive_at 版结构层。
    无切片 (全布尔掩码/花式索引); 全样本分位仅用于结果统计 (非特征)。
    """
    p = params
    ctx = make_ctx(df, p["warmup"], state_fns={})
    c, h, l, atr, years = ctx.close, ctx.high, ctx.low, ctx.atr, ctx.years
    n = len(c)
    idx = np.arange(n)
    lvls = cluster_levels(h, l, atr, min_touch=mt, tolerance_mult=tol)
    if not lvls:
        return []

    # ── 逐 bar 最近对侧位带 (B3 口径: confirm_at<=t<confirm_at+LIFE) ──
    sup = sorted([lv for lv in lvls if lv.side == "support"], key=lambda x: x.price)
    res = sorted([lv for lv in lvls if lv.side == "resistance"], key=lambda x: x.price)
    sup_p = np.array([lv.price for lv in sup], float)
    res_p = np.array([lv.price for lv in res], float)
    s_obj = np.full(n, None, dtype=object)
    r_obj = np.full(n, None, dtype=object)
    dn_dist = np.full(n, np.inf)
    up_dist = np.full(n, np.inf)
    for t in range(n):
        ct = c[t]
        k = bisect_right(sup_p, ct) - 1
        for i in range(k, -1, -1):
            lv = sup[i]
            if lv.confirm_at <= t < lv.confirm_at + p["life"]:
                s_obj[t] = lv
                dn_dist[t] = ct - lv.price
                break
        k = bisect_right(res_p, ct)
        for i in range(k, len(res)):
            lv = res[i]
            if lv.confirm_at <= t < lv.confirm_at + p["life"]:
                r_obj[t] = lv
                up_dist[t] = lv.price - ct
                break

    # ── 每位的 confirmed/known/recent (因果条件化素材) ──
    lv_known = {}
    lv_recent = {}
    lv_confpos = {}
    for lv in lvls:
        att, conf, out, ratio = level_breakdown(lv, c, atr, p["depth"], p["W"], p["hold_ratio"])
        known, _ = causal_confirmed(conf, p["W"], lag_lo=p["lag_lo"], lag_hi=p["lag_hi"])
        cum = np.concatenate([[0], np.cumsum(conf)])
        # recent[t] = confirmed[c] 存在 c ∈ [max(0,t-23), t] (前缀和, 与 causal 同技)
        recent = (cum[idx + 1] - cum[np.maximum(idx - 23, 0)]) > 0
        lv_known[id(lv)] = known
        lv_recent[id(lv)] = recent
        lv_confpos[id(lv)] = np.flatnonzero(conf)

    # ── 触碰事件 + M2 + 结构层分类 ──
    offs = np.arange(1, p["W"] + 1)
    events = []
    for lv in lvls:
        usable = idx >= lv.confirm_at
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        tm = (l <= p_hi) & (h >= p_lo) & usable
        touch = tm & ~np.roll(tm, 1)  # 进入首根 (B3 同; roll 仅污染 t=0, 已被 usable 排除)
        t_arr = np.flatnonzero(touch)
        valid = (t_arr + p["W"] < n) & (t_arr >= 13) & (atr[t_arr] > 0)
        t_arr = t_arr[valid]
        if not len(t_arr):
            continue
        seg = c[t_arr[:, None] + offs]
        if lv.side == "support":
            m2 = seg.max(axis=1) - c[t_arr]
            d_opp = up_dist[t_arr]
        else:
            m2 = c[t_arr] - seg.min(axis=1)
            d_opp = dn_dist[t_arr]
        m2 = m2 / atr[t_arr]
        known_lv = lv_known[id(lv)]
        recent_lv = lv_recent[id(lv)]
        confpos_lv = lv_confpos[id(lv)]
        for j, ti in enumerate(t_arr):
            t = int(ti)
            if lv.side == "support":
                opp = r_obj[t]
            else:
                opp = s_obj[t]
            pair_active = (opp is not None) and (d_opp[j] <= p["range_atr"] * atr[t])
            if pair_active:
                # 因果版: [t-23,t] 内突破 → 剔除; 否则 known 双 False → 区间内
                if recent_lv[t] or lv_recent[id(opp)][t]:
                    layer_c = "DROP"
                elif (not known_lv[t]) and (not lv_known[id(opp)][t]):
                    layer_c = "区间内"
                else:
                    layer_c = "孤立(对侧已破)"
                # B3 alive_at 版 (泄漏对照)
                a_own = alive_at_b3(confpos_lv, t)
                a_opp = alive_at_b3(lv_confpos[id(opp)], t)
                layer_b = "区间内" if (a_own and a_opp) else "孤立(对侧已破)"
            elif d_opp[j] <= p["wide_atr"] * atr[t]:
                layer_c = "宽成对"
                layer_b = "宽成对"
            else:
                layer_c = "孤立"
                layer_b = "孤立"
            if layer_c == "DROP":
                continue
            events.append({"m2": float(m2[j]), "layer_c": layer_c,
                           "layer_b": layer_b, "year": int(years[t])})
    return events


def m2_stats(m2):
    if len(m2) == 0:
        return float("nan"), float("nan")
    return float(np.median(m2)), float(np.percentile(m2, 90))


def pool_by_layer(events, key):
    out = {}
    for lv in LAYERS:
        out[lv] = [e["m2"] for e in events if e[key] == lv]
    out[ALL_LAYER] = [e["m2"] for e in events]
    return out


def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, combo_rows, h3_rows, year_rows):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},mt_tol={},W={},depth={},life={},lag_lo={},lag_hi={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            ",".join(p["tf_list"]), ";".join(f"{a}/{b}" for a, b in p["mt_tol"]),
            p["W"], p["depth"], p["life"], p["lag_lo"], p["lag_hi"],
            p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 口径: 描述层无入场, 1:1 WR 不适用; 无条件基线(全部触碰 "
        "M2P50 ATR归一): 真实1h(2,0.3) {:.3f} (n={}) vs GBM30种子 mean {:.3f} (n={}) "
        "[PASS] (探测器=同管线GBM null); MIN_N 描述层不适用(事件n见RESULTS)".format(
            p["gbm_seeds"], g["real_m2p50"], g["real_n_touch"],
            g["gbm_m2p50"], g["gbm_n_touch"]),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 4 组合; 描述层无入场, 无交易含义",
        "",
        "[H1/H2] M2P50Δ/M2P90Δ = 真实分位 − GBM分位 (ATR 归一; 因果版结构层; ΔP50≤-0.4, ΔP90≤-0.5 为达标)",
    ]
    for row in combo_rows:
        lines.append(f"组合 {row['name']}:")
        for lv in LAYERS + (ALL_LAYER,):
            lines.append("  {:<10} n={}/{:<7} ΔP50={:.3f} ΔP90={:.3f}{}".format(
                lv, row["n_r"][lv], row["n_g"][lv], row["dp50"][lv], row["dp90"][lv],
                "" if row["n_r"][lv] >= MIN_N and row["n_g"][lv] >= MIN_N
                else " [样本不足<{}]".format(MIN_N)))
    lines.append("")
    lines.append("[H3] 泄漏敏感性: 因果版 vs B3 alive_at 版 (区间内 ΔP50 相对差异 = "
                 "|Δ_causal−Δ_aliveat|/|Δ_causal|)")
    for h3 in h3_rows:
        lines.append("  {}: 因果 ΔP50={:.3f} vs aliveat ΔP50={:.3f} rel={:.1f}% "
                     "(n_因果={}/{} n_aliveat={}/{})".format(
            h3["name"], h3["dc"], h3["db"], h3["rel"] * 100.0,
            h3["nr_c"], h3["ng_c"], h3["nr_b"], h3["ng_b"]))
    lines.append("[对照-历史] B3(旧, 已作废, 泄漏条件化) 1h(2,0.3) 区间内 "
                 "M2P50Δ=-0.675 M2P90Δ=-1.212 (仅形状参照, 不作证据)")
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(year_rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    t0 = time.time()
    dfs_by_tf = load(PARAMS["tf_list"])
    if not dfs_by_tf or not dfs_by_tf.get("1h"):
        print("无数据, 退出")
        return 1

    p = PARAMS
    # ── GATE: 真实 1h(2,0.3) 全部触碰基线 + GBM 30 种子同管线 null (探测器自检) ──
    ref1 = dfs_by_tf["1h"][0]
    real_gate = []
    for df in dfs_by_tf["1h"]:
        real_gate.extend(run_symbol_events(df, 2, 0.3, "1h", p))
    r_m2 = np.array([e["m2"] for e in real_gate])
    r_med = float(np.median(r_m2)) if len(r_m2) else float("nan")
    gbm_gate = []
    for seed in range(p["gbm_seeds"]):
        rw = gbm_matching(ref1, seed=seed)
        gbm_gate.extend(run_symbol_events(rw, 2, 0.3, "1h", p))
    g_m2 = np.array([e["m2"] for e in gbm_gate])
    g_med = float(np.median(g_m2)) if len(g_m2) else float("nan")
    g = {"gbm_m2p50": g_med, "gbm_n_touch": int(len(g_m2)),
         "real_m2p50": r_med, "real_n_touch": int(len(r_m2))}
    if not 1.8 <= g_med <= 2.6:
        raise SystemExit(f"GATE FAIL: GBM 30种子 M2P50={g_med:.3f} ∉ [1.8,2.6] — 管线/归一错误, 停")
    if len(g_m2) < 10000:
        raise SystemExit(f"GATE FAIL: GBM 触碰数 {len(g_m2)} < 10000 — 管线未跑通, 停")
    print(f"[GATE] 真实1h(2,0.3) M2P50={r_med:.3f} vs GBM30种子 mean={g_med:.3f} "
          f"(n_touch 真实{len(r_m2)}/GBM{len(g_m2)}) [PASS]", flush=True)
    del r_m2, g_m2

    # ── 4 组合 × 真实 + GBM (每组合即时产出 combo/H3/BY_YEAR 行, 及时释放事件) ──
    combo_rows = []
    h3_rows = []
    year_rows = []
    for tf in p["tf_list"]:
        ref = dfs_by_tf[tf][0]
        for mt, tol in p["mt_tol"]:
            name = f"{tf} ({mt},{tol})"
            if name == "1h (2,0.3)":
                gbm_events = gbm_gate     # 复用 GATE 已跑
                real_events = real_gate   # 复用 GATE 已跑
            else:
                gbm_events = []
                for seed in range(p["gbm_seeds"]):
                    rw = gbm_matching(ref, seed=seed)
                    gbm_events.extend(run_symbol_events(rw, mt, tol, tf, p))
                real_events = []
                for df in dfs_by_tf[tf]:
                    real_events.extend(run_symbol_events(df, mt, tol, tf, p))
            rc = pool_by_layer(real_events, "layer_c")
            rb = pool_by_layer(real_events, "layer_b")
            gc = pool_by_layer(gbm_events, "layer_c")
            gb = pool_by_layer(gbm_events, "layer_b")
            row = {"name": name, "n_r": {}, "n_g": {}, "dp50": {}, "dp90": {}}
            for lv in LAYERS + (ALL_LAYER,):
                r50, r90 = m2_stats(rc[lv])
                g50, g90 = m2_stats(gc[lv])
                row["n_r"][lv] = len(rc[lv])
                row["n_g"][lv] = len(gc[lv])
                row["dp50"][lv] = r50 - g50
                row["dp90"][lv] = r90 - g90
            combo_rows.append(row)
            # H3: 区间内 因果 vs aliveat
            r50c, _ = m2_stats(rc["区间内"])
            g50c, _ = m2_stats(gc["区间内"])
            r50b, _ = m2_stats(rb["区间内"])
            g50b, _ = m2_stats(gb["区间内"])
            dc = r50c - g50c
            db = r50b - g50b
            rel = abs(dc - db) / max(abs(dc), 1e-9)
            h3_rows.append({"name": name, "dc": dc, "db": db, "rel": rel,
                            "nr_c": len(rc["区间内"]), "ng_c": len(gc["区间内"]),
                            "nr_b": len(rb["区间内"]), "ng_b": len(gb["区间内"])})
            # BY_YEAR (真实+GBM 成对, 全部触碰主度量 ΔP50)
            for y in p["by_year"]:
                ry = [e["m2"] for e in real_events if e["year"] == y]
                gy = [e["m2"] for e in gbm_events if e["year"] == y]
                if len(ry) < MIN_N or len(gy) < MIN_N:
                    continue
                year_rows.append(f"{name} {y} ΔP50={float(np.median(ry) - np.median(gy)):.3f} "
                                 f"(n={len(ry)}/{len(gy)})")
            print(f"[done] {name}: 区间内 n={len(rc['区间内'])}/{len(gc['区间内'])} "
                  f"ΔP50={row['dp50']['区间内']:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if name == "1h (2,0.3)":
                del real_gate, gbm_gate   # 释放 GATE 大列表 (后续组合不再引用)
            del rc, rb, gc, gb, real_events, gbm_events

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, combo_rows, h3_rows, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
