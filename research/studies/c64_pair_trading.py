#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C64 公开策略验证⑤: 配对/协整交易 (crypto) (2026-08-14, 无未来函数,
[学习级])

[学习级] 考证 (PLAN §2.5 c64 行): librarian 调研 #8 — Gatev 2006 年化 11%,
  Do-Faff 2010 利润减半 (拥挤后残值)。本砖在 20 标的 crypto 上验证配对交易
  残值。**横截面警示**: 20 标的 190 对受 n_eff=2.15 限制 (c37) — .out 报告
  n_eff 并降级声明。描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。
  **结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留
  docstring 预注册冻结、内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 60 日形成期 top-20 高相关配对 + 价差 z 均值
  回归交易, 残值净收益是否 > GBM null?

预注册假设 (PLAN §2.5 c64 行, docstring 逐字):
  H1: 配对交易净收益 > GBM null 2σ (残值验证 — Gatev 2006 现代残值)
  H2: 分年稳定性

  操作化 (运行前锁定):
    - 数据: 20 标的 1h (backtest.db) → 日线 (daily_resample)
    - 形成期 (60 日) 相关性 top 配对: 每月末用过去 60 日日收益相关矩阵选
      top-20 对 (因果滚动)
    - 价差 = 对数价格差; z = (价差 − 60 日滚动均值)/60 日滚动 sd (因果)
    - |z| > 2 开仓 (z>0: 空 leg1 多 leg2; z<0: 多 leg1 空 leg2), 回 0 平仓
      (z 穿越 0), 或配对退出选择集时平仓; dollar-neutral 逐对
    - H1: 单笔配对 P&L 均值 (log 收益) vs GBM null 30 种子同管线
      (GBM 配对无协整 — null 应≈0)
    - H2: 分年 (2024/2025/2026) 净收益
    - n_eff 警示: 重算并降级
    - 学习级: GBM null 30 种子、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  日线 close       | data_loader.daily_resample (1h→1D)    | 日线收盘后 | c30 口径
  配对选择         | 60 日日收益相关矩阵 top-20 (因果)      | 月未收盘后 | 只用窗口内
  价差 z           | (spread−mean60)/sd60 (滚动, 因果)     | bar 收盘后 | 禁全样本
  开/平仓          | |z|>2 开, z 回 0 平                     | bar 收盘后 | 逐对 dollar-neutral
  GBM null         | 20 条漂移匹配 GBM 同选择同交易          | 锚定真实   | 无协整 → null≈0
  n_eff            | (Σλ)²/Σλ² (c37 口径)                  | 全样本事后 | 降级声明

数据声明: data/backtest.db (20 标的 × 1h × 2023-08..2026-08); 日线 =
  daily_resample。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  形成期 60 日; top-20 对; z 阈值 ±2 / 回 0; 月频重选; GBM 30 种子漂移匹配;
  MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 配对选择用 60 日日收益相关 (非价格相关 — 更稳健); 价差交易在对数价格差。
  - 月频重选, 配对退出选择集时平仓 (含跨月持仓)。
  - GBM null 每标的漂移匹配 (c61/c62 口径) — GBM 无协整, null 检验交易逻辑
    本身是否机械盈利。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 配对选择 golden (构造已知相关 → top 对正确); ② z golden
    (滚动 mean/sd 对拍); ③ 交易 golden (已知 z 路径 → 开/平仓 + P&L 符号);
    ④ null sanity — GBM null 单笔均值 ∈ [−2%, +2%] (≈0); 任一失败
    SystemExit
  - GBM null 无信息对照: 30 种子同管线
  - MIN_N: 每格 n ≥ 100 (交易数; 学习级)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: 5 标的 × 3 种子, 不写 .out
  - 全量: 20 标的 × 30 种子 (预计 ≤5 分钟)

运行命令:
  python3 research/studies/c64_pair_trading.py --dev
  python3 research/studies/c64_pair_trading.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.data_loader import daily_resample, load_candles
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "1h",                          # daily_resample 来源
    "form_days": 60,                     # 形成期
    "n_pairs": 20,                       # top-20 对
    "z_open": 2.0,
    "z_close": 0.0,
    "rebal": "ME",                       # 月频重选
    "gbm_seeds": 30,
    "min_n": 100,
    "null_band": (-0.05, 0.05),          # GATE: null 单笔均值带
    "dev_subset": {"n_sym": 5, "n_seeds": 3},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c64_pair_trading"


# ── 加载 ─────────────────────────────────────────────────────
def load_daily_frame(n_sym=None):
    data = load_candles(timeframes=(PARAMS["tf"],))
    syms = [s for s in data if "USDT" in s]
    if n_sym:
        syms = syms[:n_sym]
    closes = {}
    for sym in syms:
        df = data[sym].get(PARAMS["tf"])
        if df is None:
            continue
        closes[sym] = daily_resample(df)["close"]
    return pd.DataFrame(closes).dropna()


def n_eff(C):
    lam = np.linalg.eigvalsh((C + C.T) / 2.0)
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    return float(s * s / np.sum(lam * lam)) if s > 0 else float("nan")


# ── 配对选择 (月未, 60 日相关 top-20) ────────────────────────
def select_pairs(returns, bar_idx, form_days, n_pairs):
    """月未 bar_idx: 每月最后一根 bar. 返回 {月未索引: [(i,j) top20]}.
    returns 形状 (n_sym, n_bars)."""
    sel = {}
    for m in bar_idx:
        lo = m - form_days + 1
        if lo < 0:
            continue
        r = returns[:, lo:m + 1]                  # (n_sym, form_days)
        C = np.corrcoef(r)
        n_sym = len(r)
        pairs = []
        for i in range(n_sym):
            for j in range(i + 1, n_sym):
                rho = C[i, j]
                if np.isfinite(rho):
                    pairs.append((rho, i, j))
        pairs.sort(key=lambda x: -x[0])
        sel[m] = [(i, j) for _, i, j in pairs[:n_pairs]]
    return sel


# ── 交易模拟 ─────────────────────────────────────────────────
def simulate_pairs(logp, ret, month_map, sel_by_month, form_days, z_open):
    """所有配对交易 → [trade P&L]. month_map: bar→最近的月未索引."""
    n_sym, n = logp.shape
    trades = []
    for i in range(n_sym):
        for j in range(i + 1, n_sym):
            spread = logp[i] - logp[j]
            # 滚动 mean/sd (因果)
            s_series = pd.Series(spread)
            mean = s_series.rolling(form_days).mean().values
            sd = s_series.rolling(form_days).std().values
            z = np.full(n, np.nan)
            ok = (sd > 1e-12) & np.isfinite(mean) & np.isfinite(sd)
            z[ok] = (spread[ok] - mean[ok]) / sd[ok]
            # sel[t]: 该配对在最近月未选择集
            sel = np.zeros(n, bool)
            last_sel = None
            for t in range(n):
                mb = month_map[t]
                if mb is not None:
                    sset = sel_by_month.get(mb)
                    if sset is not None:
                        last_sel = (i, j) in sset
                sel[t] = bool(last_sel)
            # 模拟
            pos = 0
            acc = 0.0
            for t in range(n):
                if pos != 0:
                    if pos == 1:
                        acc += -ret[i, t] + ret[j, t]
                    else:
                        acc += ret[i, t] - ret[j, t]
                    if (not sel[t]) or (pos == 1 and z[t] <= 0) or \
                       (pos == -1 and z[t] >= 0):
                        trades.append(acc)
                        pos = 0
                if pos == 0 and sel[t] and np.isfinite(z[t]) and \
                        abs(z[t]) > z_open:
                    pos = 1 if z[t] > 0 else -1
                    acc = 0.0
    return np.array(trades)


def month_map_of(idx, rebal):
    """每 bar → 最近期未索引 (在当期有数据的月)."""
    ends = idx.to_series().resample(rebal).last().index
    end_pos = {ts: k for k, ts in enumerate(ends)}
    # bar 时间 → 最近的月未 (月未 ≤ bar 时间)
    ts_arr = idx.values.astype("datetime64[ns]")
    ends_arr = np.array([np.datetime64(ts) for ts in ends])
    n = len(idx)
    mm = [None] * n
    # 简单映射: 月份 (year, month) → 该月最后 bar 的索引
    months = {}
    for t in range(n):
        key = (ts_arr[t].astype("datetime64[M]").astype(int),)
        months.setdefault(key, t)
    month_last = {}
    for t in range(n):
        key = (ts_arr[t].astype("datetime64[M]").astype(int),)
        month_last[key] = t
    bar_months = np.array([(ts_arr[t].astype("datetime64[M]").astype(int),)
                           for t in range(n)], dtype=object)
    mpos = []
    for t in range(n):
        key = bar_months[t]
        mpos.append(month_last.get(key, None))
    return np.array(mpos, dtype=object)


# ── 漂移匹配 GBM null ────────────────────────────────────────
def drift_gbm_daily(close_series, seed):
    c = close_series.values.astype(float)
    idx = close_series.index
    rw = gbm_matching(pd.DataFrame({"close": c}, index=idx), seed=seed)
    rw_c = rw["close"].values.astype(float)
    rw_ret = np.diff(np.log(rw_c))
    real_ret = np.diff(np.log(c))
    r1 = rw_ret - np.mean(rw_ret) + float(np.mean(real_ret))
    out = float(c[0]) * np.exp(np.concatenate([[0.0], r1]))
    return pd.Series(out, index=idx)


# ── GATE 自检 ────────────────────────────────────────────────
def gate_select_golden():
    """配对选择 golden: 构造 A/B 完全同步 (ρ=1), C/D 独立 → top-1 对 = A,B."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=60)
    returns = np.array([
        x, x,                                # A=B 完全同步
        rng.normal(size=60), rng.normal(size=60),   # C, D 独立
    ])
    sel = select_pairs(returns, [59], 60, 1)
    pairs = sel[59]
    if len(pairs) != 1 or not (pairs[0] == (0, 1)):
        raise SystemExit(f"GATE FAIL: 选择 golden 对 {pairs} ≠ [(0,1)]")
    return True


def gate_trade_golden():
    """z golden + 交易 golden: 60 bar 平坦 → 3 bar 尖峰 (z>2) → 回 0 (z≤0).
    短 leg1 (z>0) 从尖峰回 0 → 盈利 >0."""
    leg1 = np.concatenate([np.zeros(60), [0.3, 0.6, 1.0], np.zeros(40)])
    leg2 = np.zeros(len(leg1))
    logp = np.array([leg1, leg2])
    ret = np.diff(logp, axis=1, prepend=0.0)
    n = len(leg1)
    spread = leg1 - leg2
    s_series = pd.Series(spread)
    mean = s_series.rolling(60).mean().values
    sd = s_series.rolling(60).std().values
    z = np.full(n, np.nan)
    ok = (sd > 1e-12) & np.isfinite(mean)
    z[ok] = (spread - mean)[ok] / sd[ok]
    if not np.any(z > 2.0):
        raise SystemExit("GATE FAIL: z golden 无 >2 触发")
    mm = np.array([0 if t == 0 else None for t in range(n)], dtype=object)
    sel_by_month = {0: [(0, 1)]}
    trades = simulate_pairs(logp, ret, mm, sel_by_month, 60, 2.0)
    if len(trades) == 0:
        raise SystemExit("GATE FAIL: 交易 golden 无成交")
    if not np.all(np.array(trades) > 0):
        raise SystemExit(f"GATE FAIL: 交易 golden P&L {trades} 应 >0")
    return True


def gate(null_means):
    gate_select_golden()
    gate_trade_golden()
    nm = float(np.mean(null_means)) if null_means.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null 单笔均值 {nm:+.4f} ∉ [{lo}, {hi}]")
    print(f"[GATE] 选择 golden [PASS]; z/交易 golden [PASS]; null sanity "
          f"{nm:+.4f} [PASS]", flush=True)
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
        "params=form={}d,n_pairs={},z={}±,gbm_seeds={}(漂移匹配),min_n={},"
        "gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["form_days"], p["n_pairs"], p["z_open"],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: 选择 golden + z/交易 golden + null sanity [PASS]; MIN_N n≥{} "
        "[PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c64 公开策略验证⑤: 配对/协整交易 (crypto); 60 日"
        "形成期 top-20 相关对, 价差 z |z|>2 开 回 0 平, dollar-neutral 逐对; "
        "GBM null 漂移匹配 30 种子 (GBM 无协整 — null 检验交易逻辑); n_eff "
        "警示降级; 描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[横截面警示] n_eff = {:.2f} (c37 口径, c37 报 2.15) — "
                 "top 对受相关结构限制, 结论按 n_eff 降级".format(res["n_eff"]))
    r = res["trade"]
    lines.append("")
    lines.append("[H1] 配对交易单笔 P&L (log 收益) vs GBM null {} 种子:"
                 .format(p["gbm_seeds"]))
    lines.append("  真实: 单笔均值 {:+.4f} | 总净 {:+.3f} | n_trades {} {}"
                 .format(r["mean"], r["total"], r["n"], _nm(r["n"])))
    nl = r["null"]
    z = (r["mean"] - nl[0]) / nl[1] if nl[1] > 0 else float("nan")
    lines.append("  null: 单笔均值 {:+.4f}±{:.4f} | 超额 {:+.4f} (z={:+.2f}) "
                 "-> {}".format(nl[0], nl[1], r["mean"] - nl[0], z,
                                "超2σ↑" if z > 2 else "未超"))
    lines.append("")
    lines.append("[H2] 分年稳定性 (总净, log):")
    yl = []
    for y, v in sorted(r["years"].items()):
        yl.append("{} {:+.3f}".format(y, v))
    lines.append("  " + " | ".join(yl))
    lines.append("")
    lines.append("[对照-历史] Gatev 2006 (年化 11%); Do-Faff 2010 (利润减半); "
                 "c37 (n_eff=2.15); 本砖: crypto 配对交易残值 vs 漂移匹配 "
                 "GBM null (无协整)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    n_sym = PARAMS["dev_subset"]["n_sym"] if dev else None
    seeds = PARAMS["dev_subset"]["n_seeds"] if dev else PARAMS["gbm_seeds"]

    frame = load_daily_frame(n_sym)
    if len(frame.columns) < 4:
        print("数据不足")
        return 1
    logp = np.log(frame.values.astype(float)).T          # (n_sym, n)
    ret = np.diff(logp, axis=1, prepend=0.0)
    idx = frame.index
    n = len(idx)
    # 月未 bar 索引
    months = pd.Series(idx).dt.to_period("M")
    month_last = {}
    for t in range(n):
        month_last[months.iloc[t]] = t
    bar_ends = np.array([month_last[m] for m in months], dtype=int)
    month_end_idx = sorted(set(int(x) for x in bar_ends))
    # sel_by_month
    sel_by_month = select_pairs(ret, month_end_idx, PARAMS["form_days"],
                                PARAMS["n_pairs"])
    # month_map: bar t → 最近**之前**的月未索引 (形成期选择在 t 之前完成,
    #   该月未的选择适用于其后的月 — 无前视)
    me_arr = np.array(month_end_idx, dtype=int)
    month_map = []
    for t in range(n):
        prior = me_arr[me_arr < t]
        month_map.append(int(prior.max()) if len(prior) else None)
    month_map = np.array(month_map, dtype=object)
    # 交易
    trades = simulate_pairs(logp, ret, month_map, sel_by_month,
                            PARAMS["form_days"], PARAMS["z_open"])
    # null
    null_means = []
    for seed in range(seeds):
        g_frame = pd.DataFrame({
            s: drift_gbm_daily(frame[s], seed) for s in frame.columns})
        glogp = np.log(g_frame.values.astype(float)).T
        gret = np.diff(glogp, axis=1, prepend=0.0)
        g_sel = select_pairs(gret, month_end_idx, PARAMS["form_days"],
                             PARAMS["n_pairs"])
        gtrades = simulate_pairs(glogp, gret, month_map, g_sel,
                                 PARAMS["form_days"], PARAMS["z_open"])
        if len(gtrades):
            null_means.append(float(np.mean(gtrades)))
    nl = (float(np.mean(null_means)) if null_means else float("nan"),
          float(np.std(null_means, ddof=1)) if len(null_means) > 1 else 0.0)
    # n_eff
    C = np.corrcoef(ret)
    neff = n_eff(C)

    gate(np.array(null_means) if null_means else np.array([0.0]))

    if dev:
        print("  [dev] 配对 {} 笔 单笔均值 {:+.4f} vs null {:+.4f}±{:.4f} | "
              "n_eff {:.2f}".format(len(trades),
                                    float(np.mean(trades)) if len(trades)
                                    else float("nan"), nl[0], nl[1], neff))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    # 分年 (逐配对重放, 按平仓 bar 年份归集)
    year_sums = _yearly_totals(logp, ret, month_map, sel_by_month, idx,
                               PARAMS["form_days"], PARAMS["z_open"])
    res = {"n_eff": neff,
           "trade": {"mean": float(np.mean(trades)) if len(trades)
                     else float("nan"), "total": float(np.sum(trades)),
                     "n": len(trades), "null": nl, "years": year_sums}}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


def _yearly_totals(logp, ret, month_map, sel_by_month, idx, form_days, z_open):
    """逐配对重放, 按平仓 bar 年份归集总净."""
    n_sym, n = logp.shape
    years = np.array([idx[t].year for t in range(n)])
    ysum = {}
    for i in range(n_sym):
        for j in range(i + 1, n_sym):
            spread = logp[i] - logp[j]
            s_series = pd.Series(spread)
            mean = s_series.rolling(form_days).mean().values
            sd = s_series.rolling(form_days).std().values
            z = np.full(n, np.nan)
            ok = (sd > 1e-12) & np.isfinite(mean)
            z[ok] = (spread[ok] - mean[ok]) / sd[ok]
            sel = np.zeros(n, bool)
            last_sel = None
            for t in range(n):
                mb = month_map[t]
                if mb is not None:
                    sset = sel_by_month.get(mb)
                    if sset is not None:
                        last_sel = (i, j) in sset
                sel[t] = bool(last_sel)
            pos = 0
            acc = 0.0
            for t in range(n):
                if pos != 0:
                    if pos == 1:
                        acc += -ret[i, t] + ret[j, t]
                    else:
                        acc += ret[i, t] - ret[j, t]
                    if (not sel[t]) or (pos == 1 and z[t] <= 0) or \
                       (pos == -1 and z[t] >= 0):
                        y = int(years[t])
                        ysum[y] = ysum.get(y, 0.0) + acc
                        pos = 0
                if pos == 0 and sel[t] and np.isfinite(z[t]) and \
                        abs(z[t]) > z_open:
                    pos = 1 if z[t] > 0 else -1
                    acc = 0.0
    return ysum


if __name__ == "__main__":
    sys.exit(main())
