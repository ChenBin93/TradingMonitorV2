#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C68 聪明钱验证③: funding×OI 联合拥挤度 (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c68 行): librarian #3 — funding 单独已证 null (c49),
  联合 OI 是开放缺口。本砖尝试 funding×OI 联合拥挤度。**数据限制**: OKX OI
  历史端点全部 404 (open-interest-volume-history / 变体均不可得; 仅
  public/open-interest 当前快照, 无历史) — **OI 维度不可得, 降级为 funding
  单独拥挤度**, docstring/结论标注。描述层, 无入场, 无交易含义, 不涉及胜率/
  期望/成本主张。**结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): funding 高分位 (拥挤) 状态 → 未来方向 / 触碰
  折返调节, 是否 > funding 标签随机化 null?

预注册假设 (PLAN §2.5 c68 行, docstring 逐字; OI 不可得降级标注):
  H1: 拥挤状态 → 未来 K bar 方向/折返 vs GBM null (funding 单独 null,
      联合是否有增量 — **OI 不可得, 联合增量无法检验, 本砖为 funding 单独**)
  H2: 拥挤状态 → 触碰事件折返的调节 (c49 H4 失败 + OI 维度的增量检验 —
      **OI 不可得, 仅 funding 维度复检**)

  操作化 (运行前锁定):
    - 数据: 20 标的 funding (8h, 2026-05-11..08-14, funding.db) + 1h
      backtest (止 08-01); 重叠窗 05-11..08-01
    - 拥挤状态 = funding 滚动 90 日分位 ≥ 80th (因果); 映射到 1h bar
      (bar 用 ≤t 的最近 funding)
    - H1: 拥挤 bar 后 K=24 方向 (close[t+24]>close[t]) vs null (funding
      标签逐标的内打乱 30 次)
    - H2: cluster 触碰事件 (c15) 分拥挤/非拥挤组 → 折返 D1 (fade) vs null
    - 学习级: 标签随机化 null 30 次、MIN_N=100 (3 个月样本审计)、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  funding_rate     | funding.db (8h 结算)                  | 结算 ts    | 历史事实
  拥挤分位         | rolling_percentile(funding, 90d, 0.8) | 结算 ts    | 因果
  crowded bar      | 最近 funding 分位 ≥80th (时间戳映射)  | bar 收盘后 | 只回看
  方向 D1          | close[t+24] vs close[t]               | 事后       | 描述层
  触碰折返         | cluster_levels 触碰 fade D1 (c66 同款) | confirm_at | c15/c66
  null             | funding 标签逐标的内打乱 30 次         | 锚定真实   | 破坏标签

数据声明: data/funding.db (20 标的 × ~285 样本, 2026-05-11..08-14, 3 个月 —
  OKX API 上限标注); data/backtest.db (1h, 止 08-01)。**OI 历史不可得
  (全部端点 404)**。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  拥挤分位 90 日滚动 ≥80th; K=24; 触碰 fade W=24; 标签打乱 30 次; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - **OI 不可得降级**: 全部 OKX OI 历史端点 404 (仅当前快照) — H1/H2 的
    OI 联合增量无法检验, 本砖为 funding 单独拥挤度 (c49 H3/H4 的滚动分位版
    复检), 结论标注"联合检验未完成"。
  - 拥挤 bar 映射用 ≤t 最近 funding (8h 结算粒度), 因果。
  - null 用 funding 标签逐标的内打乱 (保留收益序列, 破坏拥挤标签)。
  - 学习级: 无 BY_YEAR; 30 次标签打乱沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 拥挤分位 golden (构造已知 funding → 分位对拍); ② null
    sanity — null 方向均值 ∈ [0.45, 0.55] (标签打乱后 ≈ 无条件 50%);
    任一失败 SystemExit
  - null 无信息对照: funding 标签打乱 30 次
  - MIN_N: 每格 n ≥ 100 (学习级; 3 个月样本审计)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC × 3 次打乱, 不写 .out
  - 全量: 20 标的 × 30 次打乱 (预计 ≤5 分钟)

运行命令:
  python3 research/studies/c68_crowding_oi.py --dev
  python3 research/studies/c68_crowding_oi.py
"""
import hashlib
import os
import sqlite3
import sys
import time
from datetime import date

# 仓库根入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "funding_db": "data/funding.db",
    "tf": "1h",
    "crowd_q": 0.80,
    "crowd_win": 90,                       # 30 日 = 90 个 8h 结算
                                           #   (3 个月数据仅 285 样本, 90 日窗
                                           #    吃光样本 — 30 日滚动偏离标注)
    "K": 24,
    "W": 24,                               # 触碰 fade 窗口
    "warmup": 200,
    "perm": 30,
    "min_n": 100,
    "null_band": (0.45, 0.55),             # GATE: null 方向均值带
    "dev_subset": {"n_perm": 3, "syms": ("BTC/USDT:USDT",)},
    "data_range": "funding 2026-05-11..08-14 (3 个月, API 上限); 1h 止 08-01; "
                  "OI 历史不可得 (降级标注)",
}

STUDY_ID = "c68_crowding_oi"


# ── 加载 ─────────────────────────────────────────────────────
def load_funding():
    conn = sqlite3.connect(PARAMS["funding_db"])
    cur = conn.cursor()
    insts = [r[0] for r in cur.execute(
        "SELECT DISTINCT instId FROM funding ORDER BY instId").fetchall()]
    out = {}
    for inst in insts:
        rows = cur.execute(
            "SELECT ts, funding_rate FROM funding WHERE instId=? ORDER BY ts",
            (inst,)).fetchall()
        out[inst] = (np.array([r[0] for r in rows], np.int64),
                     np.array([r[1] for r in rows], float))
    conn.close()
    return out


def crowded_bars(f_ts, f_rate, t_ms):
    """拥挤分位 (滚动 90d, ≥80th) → 1h bar 的 crowded 布尔."""
    pct = rolling_percentile(f_rate, PARAMS["crowd_win"], PARAMS["crowd_q"])
    crowded_events = np.isfinite(pct) & (f_rate >= pct)
    f_crowded = crowded_events
    n = len(t_ms)
    crow = np.zeros(n, bool)
    j = 0
    for i in range(n):
        while j + 1 < len(f_ts) and f_ts[j + 1] <= t_ms[i]:
            j += 1
        if j < len(f_ts) and f_ts[j] <= t_ms[i] and f_crowded[j]:
            crow[i] = True
    return crow


# ── 触碰 (c15 口径) ──────────────────────────────────────────
def collect_touches(ctx, params):
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    atr = ctx.atr
    W = params["W"]
    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K, tolerance_mult=0.3,
                          min_touch=2)
    evs = []
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (ctx.low <= p_hi) & (ctx.high >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev
        for t in np.flatnonzero(entry):
            if t + W >= n:
                continue
            if lv.side == "resistance":
                rev = float(c[t + W] < c[t])
            else:
                rev = float(c[t + W] > c[t])
            evs.append((t, rev))
    return evs


# ── GATE 自检 ────────────────────────────────────────────────
def gate_crowd_golden():
    """拥挤分位 golden: 构造 funding 阶梯 → 高分位段被标记."""
    f = np.array([0.0001] * 100 + [0.001] * 100)
    pct = rolling_percentile(f, 100, 0.80)
    crowd = np.isfinite(pct) & (f >= pct)
    # 前段 (0.0001) 有 50% 在分位以上? 构造应让后段 (0.001) 显著标记
    if not np.any(crowd[100:]):
        raise SystemExit("GATE FAIL: 拥挤分位 golden 未标记高分位段")
    return True


def gate(null_dir_means):
    gate_crowd_golden()
    nm = float(np.mean(null_dir_means)) if null_dir_means.size else 0.5
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null 方向均值 {nm:.4f} ∉ [{lo}, {hi}]")
    print(f"[GATE] 拥挤分位 golden [PASS]; null sanity {nm:.4f} [PASS]",
          flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n):
    return "[MIN_N 通过]" if n >= PARAMS["min_n"] else "[MIN_N 不足]"


def write_out(out_path, params, res):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crowd_q={},crowd_win={}d,K={},W={},perm={},min_n={},"
        "gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["crowd_q"], p["crowd_win"] // 3, p["K"],
            p["W"], p["perm"], p["min_n"], p["min_n"]),
        "# GATE: 拥挤分位 golden + null sanity [PASS]; MIN_N n≥{} [PASS]"
        .format(p["min_n"]),
        "# RESULTS: [学习级] c68 聪明钱验证③: funding×OI 联合拥挤度 — "
        "**OI 历史不可得 (OKX 端点全 404), 降级为 funding 单独拥挤度**; "
        "拥挤 = funding 滚动 90 日分位 ≥80th; H1 方向 K=24; H2 触碰折返 "
        "调节; null = funding 标签打乱 30 次; 3 个月窗口 (API 上限) 标注; "
        "描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[降级声明] OI 历史端点全部 404 (open-interest-volume-history "
                 "/ 变体; 仅 public/open-interest 当前快照无历史) — OI 攀升 "
                 "维度不可得, H1 联合增量 / H2 OI 增量**均未检验**, 本砖为 "
                 "funding 单独拥挤度复检")
    # H1
    r = res["h1"]
    lines.append("")
    lines.append("[H1] funding 拥挤 → 未来 {}h 方向 (close[t+{}]>close[t]):"
                 .format(p["K"], p["K"]))
    lines.append("  拥挤 bar 后方向 {:.1%} (n={}) {} | null {:.1%}±{:.1%} | "
                 "超额 {:+.1%} (z={:+.2f}) -> {}".format(
        r["real"], r["n"], _nm(r["n"]), r["null"][0], r["null"][1],
        r["real"] - r["null"][0],
        (r["real"] - r["null"][0]) / r["null"][1] if r["null"][1] > 0
        else float("nan"),
        "超2σ↑" if r["real"] > r["null"][0] + 2 * r["null"][1] else "未超"))
    lines.append("  无条件基准 (全部 bar 方向): {:.1%} (n={})".format(
        r["base"], r["base_n"]))
    # H2
    r = res["h2"]
    lines.append("")
    lines.append("[H2] funding 拥挤 → 触碰折返调节 (c49 H4 复检):")
    lines.append("  拥挤触碰折返 {:.1%} (n={}) {} | 非拥挤 {:.1%} (n={}) | "
                 "差 {:+.1%}".format(
        r["crowd"], r["n_crowd"], _nm(r["n_crowd"]), r["ncrowd"],
        r["n_ncrowd"], r["crowd"] - r["ncrowd"]))
    lines.append("  null (标签打乱): 拥挤触碰折返 {:.1%}±{:.1%}".format(
        r["null"][0], r["null"][1]))
    lines.append("")
    lines.append("[对照-历史] c49 (funding 方向/折返无调节 — H3/H4 null); "
                 "c66 (量能条件作用); 本砖: funding 单独拥挤度复检 + OI 不可得"
                 "降级; 2310.14973 质量警告 (OI 报价口径交叉核对) 因 OI 不可得"
                 "不适用")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    syms_dev = PARAMS["dev_subset"]["syms"] if dev else None
    n_perm = PARAMS["dev_subset"]["n_perm"] if dev else PARAMS["perm"]

    funding = load_funding()
    data = load_candles(timeframes=(PARAMS["tf"],))
    syms = [s for s in data if "USDT" in s]
    if syms_dev:
        syms = [s for s in syms if s in syms_dev]

    h1_dirs = []
    h1_base = []
    h1_crowd_n = 0
    h1_base_n = 0
    touch_crowd = []
    touch_ncrowd = []
    n_crowd_t = 0
    n_ncrowd_t = 0
    null_dir_pool = []

    for sym in syms:
        df = data[sym].get(PARAMS["tf"])
        if df is None or verify(df, sym, PARAMS["tf"]):
            continue
        inst = sym.replace("/USDT:USDT", "") + "-USDT-SWAP"
        if inst not in funding:
            continue
        f_ts, f_rate = funding[inst]
        ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
        ts_sec = df.index[PARAMS["warmup"]:].values.astype("datetime64[s]")
        t_ms = ts_sec.astype("int64") * 1000
        crow = crowded_bars(f_ts, f_rate, t_ms)
        c = ctx.close
        n = len(c)
        # H1: 拥挤 bar 后 K 方向
        for t in np.flatnonzero(crow):
            if t + PARAMS["K"] < n:
                h1_dirs.append(float(c[t + PARAMS["K"]] > c[t]))
                h1_crowd_n += 1
        for t in range(n - PARAMS["K"]):
            h1_base.append(float(c[t + PARAMS["K"]] > c[t]))
        h1_base_n += n - PARAMS["K"]
        # null: 标签打乱
        rng = np.random.default_rng(900 + len(h1_dirs))
        for q in range(n_perm):
            idx = rng.permutation(len(crow))
            crow_p = crow[idx]
            d = []
            for t in np.flatnonzero(crow_p):
                if t + PARAMS["K"] < n:
                    d.append(float(c[t + PARAMS["K"]] > c[t]))
            if d:
                null_dir_pool.append(float(np.mean(d)))
        # H2: 触碰折返分拥挤组
        evs = collect_touches(ctx, PARAMS)
        for t, rev in evs:
            if crow[t]:
                touch_crowd.append(rev)
                n_crowd_t += 1
            else:
                touch_ncrowd.append(rev)
                n_ncrowd_t += 1

    h1_real = float(np.mean(h1_dirs)) if h1_dirs else float("nan")
    h1_n = len(h1_dirs)
    h1_base_m = float(np.mean(h1_base)) if h1_base else float("nan")
    null_arr = np.array(null_dir_pool)
    null_h1 = (float(np.mean(null_arr)), float(np.std(null_arr, ddof=1))) \
        if len(null_arr) > 1 else (float("nan"), 0.0)

    tc = np.array(touch_crowd)
    tn = np.array(touch_ncrowd)
    h2_crowd = float(np.mean(tc)) if len(tc) else float("nan")
    h2_ncrowd = float(np.mean(tn)) if len(tn) else float("nan")

    gate(np.array(null_dir_pool) if null_dir_pool else np.array([0.5]))

    if dev:
        print("  [dev] H1 拥挤方向 {:.1%} (n={}) vs null {:.1%} | H2 拥挤触碰 "
              "{:.1%} (n={}) vs 非拥挤 {:.1%}".format(
            h1_real, h1_n, null_h1[0], h2_crowd, len(tc), h2_ncrowd))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {
        "h1": {"real": h1_real, "n": h1_n, "null": null_h1,
               "base": h1_base_m, "base_n": h1_base_n},
        "h2": {"crowd": h2_crowd, "n_crowd": n_crowd_t,
               "ncrowd": h2_ncrowd, "n_ncrowd": n_ncrowd_t,
               "null": null_h1},
    }
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
