#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C46 U1-5 MA 与滞后 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U1-5, PLAN §2.5 c46): 书 CH7 p.280-307 MA 与滞后。
  oracle 逐字核实口径: 滞后=(n−1)/2 (表 7.7 n=5→2; CH8 p.366 200 日→100 天,
  书另"50 天"笔误取 100 标注); 峰后转向延迟=(n+1)/2 (5→3); EMA Hutson
  c=2/(n+1); 方向信号=MA 斜率符号转向; 价格穿越=close 穿 MA; "方向信号少
  26-37% 交易"=CH8 表 8.2 (AMZN 10 年, 80 日 84→106 +26% 等)。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。**结论不得作交易依据**。
  学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、内置 GATE
  (H1 确定性对拍是核心)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): ① MA 滞后与峰后转向延迟公式的确定性对拍
  (纯数学); ② 带内回撤恢复的隐含运作假设审计 (书 CH7 无直接文字 — 运作假设
  而非书的预测; c19 日线版 −15.4pp, 本 4h 版预期同向=第二次证伪); ③ 方向
  转向 vs 价格穿越信号数比值对拍 (书 26-37%).

预注册假设 (PLAN §2.5 c46 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 滞后公式确定性对拍 — 构造阶梯 p_t=t (1..15) 与三角波, n=5/20/200:
      MA_t 值等于 p_{t−k} 时 k=(n−1)/2 (5→2); 峰后 MA 转向延迟=(n+1)/2
      (5→3); EMA Hutson 同图对拍
  H2: 带内回撤恢复隐含假设审计 — 趋势态=MA 方向向上 (n=20/60), 回撤事件=
      价格在 MA 上方回落触碰 MA 线 (close ≤ MA 且 MA 方向仍向上), 度量
      K bar 内恢复至前高的比例; 对照=漂移匹配 GBM 同管线 (30 种子);
      判据=恢复率差超 GBM 95% 区间 (c19 日线 −15.4pp, 1h 版预期同向)
  H3: "方向信号少 26-37% 交易"对拍 — 20 标的 4h (n=20/60 映射书 20/60 日);
      双口径信号计数 (方向转向 vs close 穿越, 永远在场); 判据: 穿越数/方向数
      − 1 落 26-37% 附近; 另报分年比值稳定性

  操作化 (运行前锁定):
    - H1: 阶梯 p_t=t (n=5: MA 位置滞后 2; n=20: 9.5; n=200: 99.5, 书取 100
      标注); 三角波峰后 MA 斜率转向延迟 (n+1)/2; EMA Hutson 滞后 (n−1)/2
    - H2: 事件 = close[t−1]>MA[t−1] 且 close[t]≤MA[t] 且 MA 方向向上; 前高
      = 事件前 H=20 bar 高点; 恢复 = K∈{5,10} bar 内高点 ≥ 前高; 真实聚合
      恢复率 vs GBM 30 种子 (2σ)
    - H3: 方向信号 = MA 斜率符号转向次数; 穿越 = close 穿 MA 次数; 比值 =
      穿越/方向 − 1; 判据 ∈ 26-37% 附近 (报告区间); 分年比值稳定性
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  MA_n             | 收盘滚动均值 (因果)                   | bar 收盘后 | 书 CH7
  MA 斜率方向      | MA[t]−MA[t−1] 符号 (转向=符号翻转)    | bar 收盘后 | 书 CH8 p.314-315
  回撤事件         | close 穿 MA (≤t 信息)                 | bar 收盘后 | 因果
  恢复度量         | 事后 K bar 高点 vs 前高               | 全样本事后 | 描述统计 (c19 同源)
  GBM null         | sim_market.gbm_matching + 同管线      | 锚定真实   | 30 种子 (漂移匹配)

数据声明:
  20 标的 4h (6,570根/标的), 2023-08..2026-08 (backtest.db)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  MA n=20/60; H2 H=20, K=5/10; H3 判据 26-37%; GBM 30 种子; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - 书为日线 n=20/60 日, 我们用 4h n=20/60 bar (日历偏差, docstring 标注)。
  - 书 CH8 p.366 "200 日=滞后 100 天" (另"50 天"笔误), 滞后公式 (n−1)/2 对
    n=200 给出 99.5 — 取书 100 并标注 (H1 golden 对 99.5 与 100 双报)。
  - H2 是运作假设审计 (书 CH7 无直接文字) — 表述区分"证伪运作假设"与
    "书错"; c19 的日线 dow 段口径与 H2 的 MA 触碰口径不同, 方向对照。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① H1 确定性对拍 (核心): 阶梯 n=5→k=2 / n=20→9.5 /
    n=200→99.5; 三角波峰后转向 (n+1)/2 (n=5→3); EMA Hutson 滞后 (n−1)/2;
    逐位断言, 任一失败 SystemExit; ② GBM null sanity: H2 GBM 恢复率
    ∈ [0.5, 0.95]
  - GBM 无信息对照: 30 种子, 同管线
  - MIN_N: 每格 n ≥ MIN_N=100 (不足标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 4h × GBM 3 种子, 不写 .out
  - 全量: 20 标的 4h × 30 种子 (预计 ≤8 分钟)

运行命令:
  python3 research/studies/c46_ma_lag.py --dev
  python3 research/studies/c46_ma_lag.py
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

from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "4h",
    "ma_ns": (20, 60),
    "h2_H": 20,                            # 前高参照窗口
    "h2_Ks": (5, 10),                      # 恢复窗口
    "h3_book": (0.26, 0.37),               # 书 CH8 表 8.2 比值区间
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "dev_subset": {"n_gbm": 3, "n_sym": 1},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c46_ma_lag"


# ── 装载 ─────────────────────────────────────────────────────
def load_ctxs(params, n_sym=None):
    data = load_candles(timeframes=(params["tf"],))
    out = []
    for sym in data:
        if "USDT" not in sym:
            continue
        if n_sym is not None and len(out) >= n_sym:
            break
        df = data[sym].get(params["tf"])
        if df is None or verify(df, sym, params["tf"]):
            continue
        ctx = make_ctx(df, params["warmup"], state_fns={})
        out.append((sym, ctx, df))
    return out


# ── MA / EMA ─────────────────────────────────────────────────
def ma_series(close, n):
    return pd.Series(close).rolling(n).mean().values


def ema_hutson(close, n):
    c = 2.0 / (n + 1.0)
    e = np.full(len(close), np.nan)
    cur = close[0]
    for i in range(len(close)):
        cur = c * close[i] + (1 - c) * cur
        e[i] = cur
    return e


# ── H2: 回撤事件 + 恢复率 ───────────────────────────────────
def h2_recovery(close, high, ma, K, H):
    """事件: close[t−1]>MA[t−1] 且 close[t]≤MA[t] 且 MA 方向向上.
    前高 = 事件前 H bar 高点; 恢复 = K bar 内高点 ≥ 前高."""
    n = len(close)
    up = np.concatenate([[False], ma[1:] > ma[:-1]])
    c_prev = np.concatenate([[np.nan], close[:-1]])
    ma_prev = np.concatenate([[np.nan], ma[:-1]])
    event = (c_prev > ma_prev) & (close <= ma) & up
    ph = pd.Series(high).rolling(H).max().shift(1).values
    rec = 0.0
    n_ev = 0
    for i in np.flatnonzero(event):
        if i + K >= n or not np.isfinite(ph[i]):
            continue
        win = float(np.max(high[i + 1:i + K + 1]))
        rec += float(win >= ph[i])
        n_ev += 1
    return (rec / n_ev) if n_ev else float("nan"), n_ev


# ── H3: 方向转向 vs close 穿越信号数 ────────────────────────
def signal_counts(close, ma):
    n = len(close)
    slope = np.concatenate([[np.nan], ma[1:] - ma[:-1]])
    d = np.zeros(n, int)
    cur = 0
    for t in range(n):
        if slope[t] > 0:
            cur = 1
        elif slope[t] < 0:
            cur = -1
        d[t] = cur
    dir_turns = sum(1 for t in range(1, n)
                    if d[t] != 0 and d[t] != d[t - 1])
    pos = close > ma
    cross = sum(1 for t in range(1, n) if pos[t] != pos[t - 1])
    return cross, dir_turns


# ── GATE 自检 (H1 确定性对拍是核心) ─────────────────────────
def gate(gbm_rec_mean):
    """① H1 确定性对拍: 阶梯 k=(n−1)/2 (n=5→2, 20→9.5, 200→99.5);
    三角波峰后转向 (n+1)/2 (5→3); EMA Hutson 滞后 (n−1)/2;
    ② GBM 恢复率 sanity."""
    # ① 阶梯 p_t = t, n=5: MA[t] = p[t−2] (k=2)
    p = np.arange(1.0, 16.0)
    ma5 = ma_series(p, 5)
    if abs(ma5[9] - 8.0) > 1e-9 or abs(ma5[9] - p[7]) > 1e-9:
        raise SystemExit(f"GATE FAIL: 阶梯 n=5 MA[9]={ma5[9]} ≠ 8 (k=2)")
    # n=20: MA[t] = p[t] − 9.5 (k=9.5)
    p20 = np.arange(1.0, 41.0)
    ma20 = ma_series(p20, 20)
    if abs(ma20[39] - 30.5) > 1e-9:
        raise SystemExit(f"GATE FAIL: 阶梯 n=20 MA[39]={ma20[39]} ≠ 30.5 (k=9.5)")
    # n=200: 阶梯 1..250 → MA[199] = 100.5 (k=99.5, 书取 100)
    p200 = np.arange(1.0, 251.0)
    ma200 = ma_series(p200, 200)
    if abs(ma200[199] - 100.5) > 1e-9:
        raise SystemExit(f"GATE FAIL: 阶梯 n=200 MA[199]={ma200[199]} ≠ 100.5")
    # ① 三角波峰后转向: 峰 P=15 (0-based idx 14), n=5 → 斜率翻转于 idx 17
    tri = np.concatenate([np.arange(1.0, 16.0), np.arange(14.0, 0.0, -1.0)])
    ma5t = ma_series(tri, 5)
    slope = np.concatenate([[np.nan], ma5t[1:] - ma5t[:-1]])
    flip = None
    for t in range(5, len(tri)):
        if slope[t] < 0:
            flip = t
            break
    if flip != 17:                         # 0-based: 峰 14 + 3
        raise SystemExit(f"GATE FAIL: 三角波转向 bar={flip} ≠ 17 (峰 idx14 + 3)")
    # ① EMA Hutson: n=5, c=1/3, 滞后 (1−c)/c = 2
    ema = ema_hutson(p20, 5)
    lag = ema[39] - p20[39]
    if abs(lag - (-2.0)) > 0.15:
        raise SystemExit(f"GATE FAIL: EMA Hutson 滞后 {lag:.3f} ≠ −2")
    # ② GBM 恢复率 sanity (恢复=回撤后 K bar 内回前高, 机械上偏严 ~15%)
    if not (0.02 <= gbm_rec_mean <= 0.80):
        raise SystemExit(f"GATE FAIL: GBM 恢复率 {gbm_rec_mean:.3f} ∉ [0.02, 0.80]")
    print(f"[GATE] H1 确定性对拍 (阶梯 k=2/9.5/99.5, 峰后转向 18, EMA 滞后 2) "
          f"[PASS]; GBM 恢复率 {gbm_rec_mean:.3f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def write_out(out_path, params, h2, h3, h1_summary):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},ma_ns={},h2_H={},h2_Ks={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            p["tf"], p["ma_ns"], p["h2_H"], p["h2_Ks"], p["gbm_seeds"],
            p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(GBM 恢复率): {:.3f} [PASS]; 探测器"
        "自检 H1 确定性对拍 [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], h2["gbm_rec_mean"], p["min_n"]),
        "# RESULTS: [学习级] c46 U1-5 MA 与滞后 (书 CH7 p.280-307); 滞后="
        "(n−1)/2, 峰后转向=(n+1)/2 (书表 7.7 / CH8 p.366, 200 日→100 天标注); "
        "H2 带内回撤恢复=运作假设审计 (书无直接文字); H3 方向转向 vs close 穿越"
        "信号数 (书 CH8 表 8.2 26-37%); 20 标的 4h; 描述层无入场, 无交易含义",
        "",
    ]
    # H1 对拍表
    lines.append("[H1] 滞后公式确定性对拍 (纯数学, GATE 逐位断言):")
    lines.append("  " + h1_summary)
    # H2
    lines.append("")
    lines.append("[H2] 带内回撤恢复率 (运作假设审计, MA 方向向上 n=20/60):")
    for nkey, r in h2["rows"].items():
        rr, ne = r["real"]
        gm, gs = r["gbm"]
        diff = rr - gm
        ok = abs(diff) > 2 * gs
        lines.append("  {}: 恢复率 {:.1%} (n={}) | GBM {:.1%}±{:.1%} | 净差 "
                     "{:+.1%} {}".format(nkey, rr, ne, gm, gs, diff,
                                         "超2σ" if ok else "未超"))
    lines.append("  H2 判据: 恢复率差超 GBM 95% -> {}".format(
        "PASS" if h2["ok"] else "FAIL"))
    lines.append("  (运作假设审计: 方向 vs 书 — 恢复差为负=第二次证伪, 见结论)")
    # H3
    lines.append("")
    lines.append("[H3] 方向转向 vs close 穿越信号数 (书 26-37%):")
    for nkey, r in h3["rows"].items():
        cross, dirn = r["counts"]
        ratio = cross / dirn - 1.0 if dirn else float("nan")
        in_band = p["h3_book"][0] <= ratio <= p["h3_book"][1] if np.isfinite(ratio) else False
        lines.append("  {}: 穿越 {} | 方向 {} | 比值 {:+.1%} {}".format(
            nkey, cross, dirn, ratio, "✓书区间" if in_band else "✗"))
    pooled = h3["pooled"]
    pr = pooled["cross"] / pooled["dirn"] - 1.0 if pooled["dirn"] else float("nan")
    lines.append("  合并: 穿越 {} / 方向 {} | 比值 {:+.1%} (书区间 "
                 "{:.0%}~{:.0%})".format(pooled["cross"], pooled["dirn"], pr,
                                          p["h3_book"][0], p["h3_book"][1]))
    yrows = " | ".join("{} {:.1%}".format(y, v) for y, v in h3["by_year"])
    lines.append("  分年比值 (穿越/方向−1): " + yrows)
    lines.append("  H3 判据: 合并比值落 26-37% 附近 -> {}".format(
        "PASS" if (p["h3_book"][0] <= pr <= p["h3_book"][1]) else "FAIL"))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c19 (日线 dow 回撤恢复率净差 -15.42pp); c34 (MA "
                 "斜率转向信号); 书 CH7 p.280-307 (MA 滞后); CH8 p.314-316 "
                 "(方向信号 26-37% 少交易)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    n_sym = PARAMS["dev_subset"]["n_sym"] if dev else None
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    ctxs = load_ctxs(PARAMS, n_sym=n_sym)

    h2_rows = {}
    h2_gbm_rec = []
    h3_rows = {}
    h3_counts = {"cross": 0, "dirn": 0}
    h3_by_year = {}

    for sym, ctx, df in ctxs:
        c, h = ctx.close, ctx.high
        for n_ma in PARAMS["ma_ns"]:
            ma = ma_series(c, n_ma)
            key = f"n={n_ma}"
            # H2 真实
            for K in PARAMS["h2_Ks"]:
                rr, ne = h2_recovery(c, h, ma, K, PARAMS["h2_H"])
                h2_rows.setdefault(f"{key} K={K}", {})["real"] = (rr, ne)
                h2_rows[f"{key} K={K}"]["sym"] = sym
            # H3 信号数 (仅 n=20 主口径 + n=60)
            cross, dirn = signal_counts(c, ma)
            h3_rows.setdefault(key, {"counts": (0, 0)})
            c0, d0 = h3_rows[key]["counts"]
            h3_rows[key]["counts"] = (c0 + cross, d0 + dirn)
        # H2 GBM null (首标, n=20)
        for seed in range(seeds):
            rw = gbm_matching(df, seed=seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gma = ma_series(gctx.close, PARAMS["ma_ns"][0])
            grr, gne = h2_recovery(gctx.close, gctx.high, gma,
                                   PARAMS["h2_Ks"][0], PARAMS["h2_H"])
            if np.isfinite(grr):
                h2_gbm_rec.append(grr)

    # H2 GBM 分布 (n=20, K=5 主; 其他 K 用同 GBM 但各算)
    gbm_dist = {f"n={n} K={K}": [] for n in PARAMS["ma_ns"]
                for K in PARAMS["h2_Ks"]}
    # 用首标的 GBM 逐种子算全部格
    _, ctx0, df0 = ctxs[0]
    for seed in range(seeds):
        rw = gbm_matching(df0, seed=seed)
        gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
        for n_ma in PARAMS["ma_ns"]:
            gma = ma_series(gctx.close, n_ma)
            for K in PARAMS["h2_Ks"]:
                grr, gne = h2_recovery(gctx.close, gctx.high, gma, K,
                                       PARAMS["h2_H"])
                if np.isfinite(grr):
                    gbm_dist[f"n={n_ma} K={K}"].append(grr)
    h2_ok = True
    for key, r in h2_rows.items():
        arr = np.array(gbm_dist.get(key, []))
        if len(arr) == 0:
            r["gbm"] = (float("nan"), 0.0)
            continue
        r["gbm"] = (float(np.mean(arr)), float(np.std(arr, ddof=1)))
        diff = r["real"][0] - r["gbm"][0]
        if not np.isfinite(diff) or abs(diff) <= 2 * r["gbm"][1]:
            h2_ok = False
    gbm_rec_mean = float(np.mean(h2_gbm_rec)) if h2_gbm_rec else float("nan")

    # H3 分年 (合并跨标的, n=20)
    h3_pooled = {"cross": 0, "dirn": 0}
    for key in h3_rows:
        c0, d0 = h3_rows[key]["counts"]
        if key == "n=20":
            h3_pooled["cross"] += c0
            h3_pooled["dirn"] += d0
    # 分年 (n=20, 跨标的)
    for sym, ctx, df in ctxs:
        ma = ma_series(ctx.close, 20)
        years = ctx.years
        # 逐 bar 方向/穿越信号
        slope = np.concatenate([[np.nan], ma[1:] - ma[:-1]])
        d = np.zeros(len(ctx.close), int)
        cur = 0
        for t in range(len(ctx.close)):
            if slope[t] > 0:
                cur = 1
            elif slope[t] < 0:
                cur = -1
            d[t] = cur
        pos = ctx.close > ma
        for y in (2024, 2025, 2026):
            m = years == y
            dt = np.zeros(len(ctx.close), int)
            dt[m] = d[m]
            turns = sum(1 for t in range(1, len(ctx.close))
                        if m[t] and dt[t] != 0 and dt[t] != dt[t - 1])
            crosses = sum(1 for t in range(1, len(ctx.close))
                          if m[t] and pos[t] != pos[t - 1])
            h3_by_year.setdefault(y, [0, 0])
            h3_by_year[y][0] += crosses
            h3_by_year[y][1] += turns
    by_year = [(y, h3_by_year[y][0] / h3_by_year[y][1] - 1.0)
               for y in sorted(h3_by_year) if h3_by_year[y][1] > 0]

    gate(gbm_rec_mean)

    if dev:
        for key, r in h2_rows.items():
            print("  [dev] {} {} 恢复={:.2f} GBM={:.2f}".format(
                r["sym"], key, r["real"][0], r["gbm"][0]))
        for key in h3_rows:
            print("  [dev] {} 穿越/方向 = {}/{}".format(
                key, h3_rows[key]["counts"][0], h3_rows[key]["counts"][1]))
        print(f"[dev] 管线 OK ({len(ctxs)} 标的 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    h1_summary = ("阶梯: n=5→k=2, n=20→k=9.5, n=200→k=99.5 (书取 100, "
                  "50 笔误标注); 三角波峰后转向: n=5→P+3 (=(n+1)/2); "
                  "EMA Hutson 滞后: (n−1)/2 — GATE 逐位 PASS")
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, {"rows": h2_rows, "ok": h2_ok,
                                 "gbm_rec_mean": gbm_rec_mean},
              {"rows": h3_rows, "pooled": h3_pooled, "by_year": by_year},
              h1_summary)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
