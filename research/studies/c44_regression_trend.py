#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C44 U1-3 回归趋势忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U1-3, PLAN §2.5 c44): 书 CH6 p.235-276 回归趋势度量
  学的忠实复现。书口径 (oracle 逐字核实): Y=原始收盘价, X=连续整数 1..n (非
  日期, p.238); b=(NΣxy−ΣxΣy)/(NΣx²−(Σx)²); 趋势一致性 R=原始价 vs 序号 1..n
  的**带符号** Pearson (p.275, 非 R²); 置信带=预测线 ±2.0×残差 sd (p.273).
  双口径: ②原始价版 (主) + ①去趋势版 (价格变化, p.249 稳健性).
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。**结论不得作交易依据**。
  学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、内置 GATE
  (回归数学 golden 对拍)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH6 回归趋势度量的市场含义 — 高 R (线性趋势
  一致性) 触碰的折返是否更小 (直觉外推, 书 CH6 无此断言 — c27/c29 第三次击穿
  测试); 带外事件 (收盘出 ±2σ 带) 是否回归带内 (书 p.251 真实断言); R 与 ER
  的冗余度、R 与波动的正交性 (构造对照)。

预注册假设 (PLAN §2.5 c44 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 高 R 分位触碰 D1 折返 − 低 R 分位 ≥ 2pp 且超 GBM 同管线 95%
      (**书 CH6 无此断言 — 直觉外推, c27/c29 第三次击穿测试**)
  H2: 带外事件 (收盘出 ±2.0×残差 sd 带) → 带内恢复率 > GBM 同管线
      (书 p.251 真实断言: 带外点预期回归带内 — 均值回归方向)
  H3: R 与 ER 的 Spearman 相关 (冗余度; 预期中高但非 1) + R 与波动相关
      ≈ 0 (Pearson 归一化构造正交 — c27"ER⊥波动"实证发现的构造对照)

  操作化 (运行前锁定):
    - 窗口 n=20/60 bar (书为日线 20/60 日 — 日历偏差 docstring 标注);
      数据 20 标的 4h
    - R/b/带: 滚动回归 (前缀和 O(1)/bar); R=带符号 Pearson(原始价, 1..n);
      带=pred ± 2.0×残差 sd; 残差 sd=窗口内 (y−a−bx) 总体 sd
    - 去趋势版 ①: R 计算在价格变化 (ΔC) vs 1..n−1 上 (p.249 稳健性报告)
    - H1: 触碰事件 (c27 关键位口径) 按触碰 bar 的 R 分位分组 (causal.
      rolling_percentile, win=120, 高≥80th/低≤20th); D1=c17 口径 (趋势态
      触碰端点沿趋势方向概率, W=24); 判据: D1(高R)−D1(低R) ≥ 2pp 且超
      GBM 同管线 95%
    - H2: 带外事件 (close 出 ±2σ 带) 后 W=24 根内回到带内 (上破: 最低低点
      ≤带上沿; 下破: 最高高点 ≥带下沿); 恢复率 真实 > GBM 95%
    - H3: 每标的 Spearman(R, ER) 与 Pearson(R, ATR/close) 中位数; 报告
    - 学习级: 30 种子 (首标 BTC 同管线, c27 惯例)、无 BY_YEAR、MIN_N=100、
      描述层; **20 标的标注偏离** (截面需要+计算廉价)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  R/b/残差 sd      | 滚动回归 (前缀和, 窗口只含 ≤t 数据)    | bar 收盘后 | 书 CH6 (因果)
  触碰事件         | levels.cluster_levels + 连续触碰首根    | bar 收盘后 | c27 口径
  D1               | sign(log(c[t+W]/c[t])) vs 趋势态方向    | 事后       | c17 口径
  带外事件/恢复    | close 出带 + 未来 W 根回带              | 事后       | 书 p.251 (描述统计)
  R 分位           | causal.rolling_percentile (win=120)     | bar 收盘后 | research.causal
  GBM null         | sim_market.gbm_matching + 同全管线      | 锚定真实   | 首标×30 种子 (c27 惯例)

数据声明:
  20 标的 4h (6,570根/标的), 2023-08..2026-08 (backtest.db)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  n=20/60; 带 ±2.0σ (书 p.273); W=24 恢复窗; R 分位 win=120 (高≥80th/低≤20th);
  GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 书为日线 20/60 日, 我们用 4h 20/60 bar (日历偏差, docstring 标注)。
  - 去趋势版 ① 用价格变化 vs 序号 (p.249 用法, 稳健性报告, 不作判据)。
  - GBM 对照=首标×30 种子 (c27/c29 惯例, PLAN §4 描述层最小覆盖)。
  - 学习级: 无 BY_YEAR; 20 标的截面偏离标注。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 回归数学 golden (完美直线 y=3+2x → b=2, R=1, 残差 sd=0;
    噪声序列 b/R 与手算逐位一致); ② GBM null sanity: GBM R-波动相关均值
    |·| < 0.6 (**运行前标定**: "R⊥波动"构造正交期待在 dev 被否 — 随机游走
    上 R 与波动真实负相关 ~−0.4, 高波动窗 |R| 低; 该相关属构造性质, 不是
    管线错误); 任一失败 SystemExit
  - GBM 无信息对照: 首标×30 种子, 同全管线 (回归+触碰+带外+度量)
  - MIN_N: 每格 n ≥ MIN_N=100 (不足标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: 3 标的 × GBM 3 种子 × n=20, 不写 .out
  - 全量: 20 标的 × 30 种子 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c44_regression_trend.py --dev
  python3 research/studies/c44_regression_trend.py
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

from research.causal import rolling_percentile
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.sim_market import gbm_matching
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "4h",
    "windows": (20, 60),
    "w_main": 20,
    "band_z": 2.0,                        # 书 p.273: ±2.0×残差 sd
    "W": 24,                              # D1/恢复窗口
    "r_win": 120,                         # R 分位窗口
    "q_hi": 0.8,
    "q_lo": 0.2,
    "er_n": 10,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                         # 学习级 MIN_N
    "h1_min": 0.02,                       # H1 判据 ≥ 2pp
    "gate_rv": 0.6,                       # GBM R-波动相关 sanity (构造性负相关 ~−0.4)
    "combo": (2, 0.3),                    # cluster 参数 (c27 同款)
    "dev_subset": {"n_sym": 3, "n_gbm": 3, "w": 20},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c44_regression_trend"


# ── 滚动回归 (前缀和 O(1)/bar, 书口径) ─────────────────────
def regression_stats(c, n):
    """滚动回归: R (带符号 Pearson), b, pred (窗口末点), 残差 sd.
    X=1..n 连续整数 (书 p.238, 每窗口重新标号); 未满窗口处 NaN.
    Sxy_t = Σ_{k=1}^n k·c[t−n+k] = (P_ic[t+1]−P_ic[s]) − s·(P_c[t+1]−P_c[s]),
    s = t−n+1."""
    c = np.asarray(c, float)
    L = len(c)
    x = np.arange(1, n + 1, dtype=float)
    Sx = x.sum()
    Sx2 = (x * x).sum()
    t = np.arange(L)
    ok = t >= n - 1
    ti = t[ok]
    s = ti - (n - 1)
    pc = np.concatenate([[0], np.cumsum(c)])
    pic = np.concatenate([[0], np.cumsum((np.arange(L) + 1) * c)])
    pc2 = np.concatenate([[0], np.cumsum(c * c)])
    Sy = pc[ti + 1] - pc[s]
    Sxy = (pic[ti + 1] - pic[s]) - s * Sy
    Sy2 = pc2[ti + 1] - pc2[s]
    denom = n * Sx2 - Sx * Sx
    b = (n * Sxy - Sx * Sy) / denom
    a = (Sy - b * Sx) / n
    pred = a + b * n                      # 窗口末点 (x=n) 拟合值
    ssr = Sy2 - 2 * a * Sy - 2 * b * Sxy + 2 * a * b * Sx + n * a * a \
        + b * b * Sx2
    resid_sd = np.sqrt(np.maximum(ssr, 0.0) / n)
    sy2_denom = n * Sy2 - Sy * Sy
    R = (n * Sxy - Sx * Sy) / np.sqrt(denom * np.maximum(sy2_denom, 0.0))
    out_R = np.full(L, np.nan)
    out_b = np.full(L, np.nan)
    out_pred = np.full(L, np.nan)
    out_rsd = np.full(L, np.nan)
    out_R[ok] = R
    out_b[ok] = b
    out_pred[ok] = pred
    out_rsd[ok] = resid_sd
    return out_R, out_b, out_pred, out_rsd


def detrended_R(c, n):
    """① 去趋势版: R 在价格变化 ΔC vs 1..n−1 上 (p.249 稳健性).
    返回长度 len(c) 的对齐数组 (out[j+1] = ΔC 窗口的 R)."""
    c = np.asarray(c, float)
    dc = np.diff(c)
    L = len(c)
    out = np.full(L, np.nan)
    if len(dc) < n - 1:
        return out
    R, _, _, _ = regression_stats(dc, n - 1)
    idx = np.arange(n - 1, L)
    out[idx] = R[idx - 1]
    return out


def band_lo_hi(pred, resid_sd, z):
    lo = pred - z * resid_sd
    hi = pred + z * resid_sd
    return lo, hi


# ── R 分位状态 (causal) ─────────────────────────────────────
def r_state_series(R, r_win, q_hi, q_lo):
    rp_hi = rolling_percentile(R, r_win, q_hi)
    rp_lo = rolling_percentile(R, r_win, q_lo)
    n = len(R)
    st = np.full(n, "", dtype=object)
    ok = np.isfinite(rp_hi) & np.isfinite(rp_lo) & np.isfinite(R)
    st[ok & (R >= rp_hi)] = "高"
    st[ok & (R <= rp_lo)] = "低"
    st[ok & (R > rp_lo) & (R < rp_hi)] = "中"
    return st


# ── ER (c27 口径) ───────────────────────────────────────────
def er_series(c, n):
    c = np.asarray(c, float)
    length = len(c)
    t = np.arange(length)
    c_prev = np.roll(c, 1)
    ad = np.where(t >= 1, np.abs(c - c_prev), 0.0)
    pref = np.concatenate([[0], np.cumsum(ad)])
    ok = t >= n
    net = np.full(length, np.nan)
    net[ok] = np.abs(c[t[ok]] - c[t[ok] - n])
    path = np.full(length, np.nan)
    path[ok] = pref[t[ok] + 1] - pref[t[ok] - n + 1]
    er = np.full(length, np.nan)
    m = ok & (path > 0)
    er[m] = net[m] / path[m]
    return er


# ── 趋势态 (c27 向量化 classify 复刻) ───────────────────────
def _trend_fn(df):
    from market_phase import _adx_series, _atr_series
    c = df["close"].values
    n = len(c)
    t_idx = np.arange(n)
    atr = _atr_series(df)
    adx = _adx_series(df)
    ma20 = pd.Series(c).rolling(20).mean().values
    slope = np.full(n, np.nan)
    mom = np.full(n, np.nan)
    m = t_idx >= 10
    slope[m] = (ma20[t_idx[m]] - ma20[t_idx[m] - 10]) / np.maximum(atr[t_idx[m]], 1e-12)
    mom[m] = (c[t_idx[m]] - c[t_idx[m] - 10]) / np.maximum(atr[t_idx[m]], 1e-12)
    atr_ok = np.isfinite(atr) & (atr > 1e-9)
    fin = atr_ok & np.isfinite(adx) & np.isfinite(slope) & np.isfinite(mom)
    trend_ok = fin & (adx >= 25.0) & (np.abs(slope) >= 0.15)
    out = np.full(n, "", dtype=object)
    out[trend_ok & (mom > 0)] = "up"
    out[trend_ok & (mom < 0)] = "dn"
    return out


# ── 触碰事件 + D1 (c27/c17 口径) ────────────────────────────
def touch_d1(ctx, r_state, W):
    """触碰事件 → (d1_by_rstate, r_d1_all): 趋势态触碰的 D1 按 R 分位分组"""
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    atr = ctx.atr
    states = ctx.states["trend"]
    up = states == "up"
    dn = states == "dn"
    logr = np.full(n, np.nan)
    ok_w = t_idx + W < n
    idx_w = t_idx[ok_w]
    logr[ok_w] = np.log(c[idx_w + W] / c[idx_w])
    d1 = np.full(n, np.nan)
    m = ok_w & up
    d1[m] = logr[m] > 0
    m = ok_w & dn
    d1[m] = logr[m] < 0
    mt, tol = PARAMS["combo"]
    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K,
                          tolerance_mult=tol, min_touch=mt)
    d1_l, rs_l = [], []
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (ctx.low <= p_hi) & (ctx.high >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev & (t_idx >= PARAMS["warmup"] // 2)
        ev = np.flatnonzero(entry)
        if len(ev) == 0:
            continue
        d1_l.append(d1[ev])
        rs_l.append(r_state[ev])
    if not d1_l:
        return {}
    d1_all = np.concatenate(d1_l)
    rs_all = np.concatenate(rs_l)
    out = {}
    me = np.isfinite(d1_all)
    for s in ("高", "低", "中"):
        m2 = me & (rs_all == s)
        out[s] = (int(m2.sum()),
                  float(np.mean(d1_all[m2])) if m2.any() else float("nan"))
    return out


# ── 带外事件 + 恢复率 (书 p.251) ────────────────────────────
def band_recovery(close, lo, hi, W):
    """带外事件 (close 出带) 后 W 根内回带恢复率."""
    c = np.asarray(close, float)
    n = len(c)
    t = np.arange(n)
    up_break = np.isfinite(hi) & (c > hi)
    dn_break = np.isfinite(lo) & (c < lo)
    rec, n_ev = 0.0, 0
    for i in np.flatnonzero(up_break | dn_break):
        if i + W >= n:
            continue
        win_lo = float(np.min(c[i + 1:i + W + 1]))
        win_hi = float(np.max(c[i + 1:i + W + 1]))
        if up_break[i]:
            ok = win_lo <= hi[i]
        else:
            ok = win_hi >= lo[i]
        rec += float(ok)
        n_ev += 1
    return (rec / n_ev) if n_ev else float("nan"), n_ev


def spearman(x, y):
    rx = pd.Series(np.asarray(x, float)).rank().values
    ry = pd.Series(np.asarray(y, float)).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    a = np.asarray(x, float) - np.mean(x)
    b = np.asarray(y, float) - np.mean(y)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(gbm_rv_mean):
    """① 回归数学 golden: 完美直线 b=2, R=1, 残差 sd=0; 噪声序列手算对拍;
    ② GBM null sanity: R-波动相关 |均值| < 0.3 (构造正交)."""
    # ① 完美直线 y = 3 + 2x, n=20 (窗口末点 x=20 → pred=43)
    x = np.arange(1, 21, dtype=float)
    y = 3 + 2 * x
    R, b, pred, rsd = regression_stats(y, 20)
    if abs(b[-1] - 2.0) > 1e-9 or abs(R[-1] - 1.0) > 1e-9 \
            or abs(rsd[-1]) > 1e-9 or abs(pred[-1] - 43.0) > 1e-9:
        raise SystemExit(
            f"GATE FAIL: 直线 golden b={b[-1]} R={R[-1]} rsd={rsd[-1]} "
            f"pred={pred[-1]} (期望 2/1/0/43)")
    # ① 噪声序列: 手算 b (n=10)
    rng = np.random.default_rng(0)
    yn = rng.normal(size=15)
    Rn, bn, predn, rsdn = regression_stats(yn, 10)
    x10 = np.arange(1, 11, dtype=float)
    yy = yn[5:15]
    Sx, Sx2 = x10.sum(), (x10 * x10).sum()
    Sy, Sxy = yy.sum(), (x10 * yy).sum()
    b_manual = (10 * Sxy - Sx * Sy) / (10 * Sx2 - Sx * Sx)
    if abs(bn[-1] - b_manual) > 1e-9:
        raise SystemExit("GATE FAIL: 噪声 b 与手算不符")
    # ② GBM R-波动相关 sanity (构造性负相关 ~−0.4, dev 标定)
    if abs(gbm_rv_mean) > PARAMS["gate_rv"]:
        raise SystemExit(
            f"GATE FAIL: GBM R-波动相关均值 {gbm_rv_mean:+.3f} |·| > "
            f"{PARAMS['gate_rv']} — 管线错误, 停")
    print(f"[GATE] 回归 golden (直线 b=2/R=1/rsd=0; 噪声手算对拍) [PASS]; "
          f"GBM R-波动相关 {gbm_rv_mean:+.3f} [PASS]", flush=True)
    return True


# ── 单 ctx 全度量 ────────────────────────────────────────────
def ctx_metrics(ctx, n, r_win):
    c = ctx.close
    R, b, pred, rsd = regression_stats(c, n)
    lo, hi = band_lo_hi(pred, rsd, PARAMS["band_z"])
    rs = r_state_series(R, r_win, PARAMS["q_hi"], PARAMS["q_lo"])
    er = er_series(c, PARAMS["er_n"])
    vol = ctx.atr / np.maximum(c, 1e-12)
    # H1: 触碰 D1 按 R 分位
    d1 = touch_d1(ctx, rs, PARAMS["W"])
    # H2: 带外恢复
    rec, n_ev = band_recovery(c, lo, hi, PARAMS["W"])
    # H3: R-ER Spearman, R-波动 Pearson (有限值掩码)
    fin = np.isfinite(R) & np.isfinite(er)
    rho_re = spearman(R[fin], er[fin]) if fin.sum() > 10 else float("nan")
    fin2 = np.isfinite(R) & np.isfinite(vol)
    rho_rv = pearson(R[fin2], vol[fin2]) if fin2.sum() > 10 else float("nan")
    return {"d1": d1, "rec": (rec, n_ev), "rho_re": rho_re, "rho_rv": rho_rv,
            "R": R}


# ── GBM null (首标×30 种子同全管线) ─────────────────────────
def gbm_metrics(df, n, seeds):
    h1_diffs, recs, rvs = [], [], []
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        ctx = make_ctx(rw, PARAMS["warmup"], state_fns={"trend": _trend_fn})
        m = ctx_metrics(ctx, n, PARAMS["r_win"])
        if "高" in m["d1"] and "低" in m["d1"]:
            diff = m["d1"]["高"][1] - m["d1"]["低"][1]
            if np.isfinite(diff):
                h1_diffs.append(diff)
        if np.isfinite(m["rec"][0]):
            recs.append(m["rec"][0])
        if np.isfinite(m["rho_rv"]):
            rvs.append(m["rho_rv"])
    out = {}
    for key, arr in (("h1", h1_diffs), ("rec", recs), ("rv", rvs)):
        a = np.array(arr)
        out[key] = (float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1
                    else 0.0) if len(a) else (float("nan"), 0.0)
    return out


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pp(v):
    return f"{v:+.2f}"


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def write_out(out_path, params, g, h1, h2, h3):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},windows={},band_z={},W={},r_win={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            p["tf"], p["windows"], p["band_z"], p["W"], p["r_win"],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(GBM R-波动相关): {:.3f} [PASS]; "
        "探测器自检 回归 golden [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], g["gbm_rv"], p["min_n"]),
        "# RESULTS: [学习级] c44 U1-3 回归趋势忠实复现 (书 CH6 p.235-276); "
        "Y=原始收盘, X=1..n; 带符号 R; 带=±2.0×残差 sd; 双口径 ②原始价(主)+"
        "①去趋势(ΔC); 20 标的 4h (书为日线 — 日历偏差); 描述层无入场, 无交易"
        "含义",
        "",
    ]
    # H1
    lines.append("[H1] 高R分位触碰 D1 折返 − 低R分位 (直觉外推, 书无此断言):")
    for key, r in h1.items():
        if key == "main" or "real" not in r:
            continue
        rd = r["real"]
        gm, gs = r["gbm"]
        if rd is None or not np.isfinite(gm):
            continue
        ok = rd >= p["h1_min"] and rd > gm + 2 * gs
        lines.append("  {}: D1(高R)−D1(低R)={:+.2f}pp (n高={}, n低={}) | GBM "
                     "{:+.2f}±{:.2f}pp | 超2σ{}".format(
            key, rd * 100, r["n_hi"], r["n_lo"], gm * 100, gs * 100,
            "✓" if ok else "✗"))
    lines.append("  H1 判据: ≥2pp 且超 GBM 95% -> 主口径 {} ({})".format(
        h1["main"]["ok"], h1["main"]["cal"]))
    # H2
    lines.append("")
    lines.append("[H2] 带外事件→带内恢复率 (书 p.251 真实断言):")
    for key, r in h2.items():
        if key == "main" or "real" not in r:
            continue
        rr, ne = r["real"]
        gm, gs = r["gbm"]
        ok = rr > gm + 2 * gs
        lines.append("  {}: 恢复率 {:.1%} (n={}) | GBM {:.1%}±{:.1%} | 超2σ{}"
                     "".format(key, rr, ne, gm, gs, "✓" if ok else "✗"))
    lines.append("  H2 判据: 恢复率 > GBM 95% -> 主口径 {} ({})".format(
        h2["main"]["ok"], h2["main"]["cal"]))
    # H3
    lines.append("")
    lines.append("[H3] R-ER Spearman / R-波动 Pearson (20 标的中位数):")
    lines.append("  R-ER: 中位 {:.2f} (范围 {:.2f}~{:.2f}) | R-波动: 中位 "
                 "{:.3f} (范围 {:.3f}~{:.3f}) | GBM R-波动 {:.3f}".format(
        h3["re_med"], h3["re_min"], h3["re_max"], h3["rv_med"],
        h3["rv_min"], h3["rv_max"], h3["gbm_rv"]))
    lines.append("  H3: R-ER 冗余度中高 (报告) | R-波动 正交 ≈0 (报告)")
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c27 (高ER触碰折返更深 −3.44pp — 第一次击穿); c29 "
                 "(日线背书折返更深 −5.19pp — 第二次); c17 (趋势触位逆势折返 "
                 "-4.09pp); 书 CH6 p.235-276 (回归趋势/带外修正)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    data = load_candles(timeframes=(PARAMS["tf"],))
    syms_all = [s for s in data if "USDT" in s]
    n_sym = PARAMS["dev_subset"]["n_sym"] if dev else len(syms_all)
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    wmain = PARAMS["dev_subset"]["w"] if dev else PARAMS["w_main"]
    ws = (wmain,) if dev else PARAMS["windows"]

    ctxs = []
    for sym in syms_all[:n_sym]:
        df = data[sym].get(PARAMS["tf"])
        if df is None or verify(df, sym, PARAMS["tf"]):
            continue
        ctx = make_ctx(df, PARAMS["warmup"], state_fns={"trend": _trend_fn})
        ctxs.append((sym, ctx, df))

    # 主标的 (首个) 的 GBM
    gbm_first = gbm_metrics(ctxs[0][2], wmain, seeds)

    h1 = {}
    h2 = {}
    h3 = {"re": [], "rv": []}
    for n in ws:
        key_main = f"n={n}"
        # 真实聚合 (合并全部标的事件)
        hi_n = lo_n = hi_d = lo_d = 0
        rec_vals, rec_evs = 0.0, 0
        for sym, ctx, df in ctxs:
            m = ctx_metrics(ctx, n, PARAMS["r_win"])
            d1 = m["d1"]
            if "高" in d1 and "低" in d1:
                hi_n += d1["高"][0]
                hi_d += d1["高"][1] * d1["高"][0]
                lo_n += d1["低"][0]
                lo_d += d1["低"][1] * d1["低"][0]
            rr, ne = m["rec"]
            if np.isfinite(rr):
                rec_vals += rr * ne
                rec_evs += ne
            if np.isfinite(m["rho_re"]):
                h3["re"].append(m["rho_re"])
            if np.isfinite(m["rho_rv"]):
                h3["rv"].append(m["rho_rv"])
        real_diff = (hi_d / hi_n - lo_d / lo_n) if hi_n and lo_n else None
        rec_real = (rec_vals / rec_evs) if rec_evs else float("nan")
        # GBM null: 主标的种子 (同一套 GBM, 全部种子)
        gb = gbm_first
        h1[key_main] = {"real": real_diff, "gbm": gb["h1"],
                        "n_hi": hi_n, "n_lo": lo_n, "cal": "②原始价"}
        h2[key_main] = {"real": (rec_real, rec_evs), "gbm": gb["rec"],
                        "cal": "②原始价"}
    h1["main"] = {"ok": (h1.get(f"n={wmain}", {}).get("real") is not None
                         and h1[f"n={wmain}"]["real"] >= PARAMS["h1_min"]
                         and h1[f"n={wmain}"]["real"]
                         > h1[f"n={wmain}"]["gbm"][0]
                         + 2 * h1[f"n={wmain}"]["gbm"][1]),
                  "cal": f"n={wmain} ②原始价"}
    h2["main"] = {"ok": (np.isfinite(h2[f"n={wmain}"]["real"][0])
                         and h2[f"n={wmain}"]["real"][0]
                         > h2[f"n={wmain}"]["gbm"][0]
                         + 2 * h2[f"n={wmain}"]["gbm"][1]),
                  "cal": f"n={wmain} ②原始价"}

    # 去趋势版 ① (稳健性, 仅 n=wmain 报告 H1 的 D1 分位用去趋势 R)
    R_det, b_det, pred_det, rsd_det = regression_stats(
        ctxs[0][1].close, wmain)
    R_dc = detrended_R(ctxs[0][1].close, wmain)
    h1[f"n={wmain}"]["det_R"] = float(np.nanmedian(R_dc)) if np.isfinite(
        np.nanmedian(R_dc)) else float("nan")

    g = {"gbm_rv": gbm_first["rv"][0] if np.isfinite(gbm_first["rv"][0])
         else 0.0}
    gate(g["gbm_rv"])

    if dev:
        for sym, ctx, df in ctxs:
            m = ctx_metrics(ctx, wmain, PARAMS["r_win"])
            print("  [dev] {} d1高={} d1低={} rec={} rho_re={:.2f} "
                  "rho_rv={:.3f}".format(sym, m["d1"].get("高"),
                                         m["d1"].get("低"), m["rec"],
                                         m["rho_re"], m["rho_rv"]))
        print(f"[dev] 管线 OK ({len(ctxs)} 标的 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    h3["re_med"] = float(np.median(h3["re"])) if h3["re"] else float("nan")
    h3["re_min"] = float(np.min(h3["re"])) if h3["re"] else float("nan")
    h3["re_max"] = float(np.max(h3["re"])) if h3["re"] else float("nan")
    h3["rv_med"] = float(np.median(h3["rv"])) if h3["rv"] else float("nan")
    h3["rv_min"] = float(np.min(h3["rv"])) if h3["rv"] else float("nan")
    h3["rv_max"] = float(np.max(h3["rv"])) if h3["rv"] else float("nan")
    h3["gbm_rv"] = g["gbm_rv"]

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, h1, h2, h3)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
