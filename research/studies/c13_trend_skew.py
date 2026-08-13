#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C13 趋势收益偏度特异性 (因果口径重验) (2026-08-13, 无未来函数, 1h/4h)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画"趋势状态内单根收益偏度
  与 top-5% 大 K 的方向集中"这一市场事实, 无入场, 无交易含义, 无任何方向/
  收益/成本结论。状态内组内偏度为事后统计 ([DESCRIPTIVE]), 禁止进入交易
  含义; 条件化分层 (状态标签) 本身是因果序列 (state_features 语义, 已过
  不变性测试)。若未来用作特征/条件, 必须经滚动口径重验。

================================================================
研究问题 (预注册, 运行前冻结): 趋势态 (尤其 late) 收益偏度真实显著 > GBM,
  是唯一趋势特异性? (PLAN §4 c13)

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1: up:late 单根收益偏度 ≥ +1.5 (GBM ≤ +0.3); dn:late ≤ -1.5
  H2: top 5% |r| 大 K 的顺趋势方向 |r| 占比 ≥ 60% (GBM ≈ 50%)
  H3: |偏度| early→accel→late 单调不降 (up 与 dn 两侧)
  操作定义 (冻结, 与 a4 同式, 因果化):
    - 收益 r = close[t]/close[t-1] − 1 (单根简单收益, 与 a4 Q2 同式)
    - 状态 = state_features.state_series 语义 (向量化复现, GATE 逐位对拍
      错位 0; 只用已收盘 bar 计算的因果状态序列)
    - 偏度 = 状态内 r 的组内偏度 (scipy.stats.skew, bias=True, 与 a4 同款;
      组内事后统计属 [DESCRIPTIVE])
    - H2: 状态内 |r| 最大 5% 的 bar (k=max(1, round(0.05n))) 中, 顺趋势方向
      (up: r>0; dn: r<0) 的 |r| 合计 / 该 5% 的 |r| 合计 — "top 5% 大 K 的
      移动中顺趋势占比" (GBM 两侧对称 → ≈50%); 另附诊断度量 C_share =
      top5% 顺趋势 |r| 合计 / 全部顺趋势 |r| 合计 (收益贡献份额版本)
  exit 含义: H1-H3 全过 → "尾部发动机"确认, 作为 c22 机制前提 (本结论只
  陈述市场事实, 不做交易主张)。

数据声明:
  data/backtest.db (gitignored): 20 标的 × 1h/4h × 2023-08 → 2026-08
  (1h 26,280根, 4h 6,570根, 时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  周期: 1h/4h; warmup=600 (make_ctx 截断); head_drop=120 (截断后丢弃头部,
  覆盖 rolling 特征 warm-up 的错标 bar); gbm_seeds=30; BY_YEAR 2024/2025/2026。

================================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close            | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口 (禁手动切片)
  收益 r           | close[t]/close[t-1]−1 (掩码实现)       | bar 收盘后 | 单根收益 (与 a4 同式)
  趋势状态         | state_features.state_series 语义的      | bar 收盘后 | 与 state_series 逐位对拍 (GATE
                   |   向量化复现 (rolling 特征 + classify)  |            |   等价性自检); 性能原因, 语义一致
  组内偏度         | scipy.stats.skew (状态内事后统计)        | 全样本     | [DESCRIPTIVE] 组内统计, 禁入交易含义
  top5% 选择       | |r| 降序 argsort 取前 5% (阈值截断)     | 全样本     | [DESCRIPTIVE] 事后描述; 无全样本分位
  分年             | ctx.years (截断坐标) 事后聚合           | 全样本     | 描述层 BY_YEAR (成对 真实+GBM)
  GBM 无信息对照   | sim_market.gbm_matching(ref_df, seed)   | 锚定真实   | 固定种子序列 0..29;
                   |   (索引/长度/σ 锚定真实)                |            |   首标×30 种子全管线

设计偏离说明 (相对 a4, 因果化与定义差异 — 结论对照时须考虑):
  - a4 在完整 df 上跑 state_series (rolling 用全长历史); 本脚本经 make_ctx
    截断 (warmup=600) + head_drop=120 再分层, 头部错标 bar 被剔除 — 状态
    标签与 a4 同语义 (向量化复现, GATE 逐位对拍), 但样本窗口略短。
  - a4 未直接测 H2 (top5% 大 K 方向集中); 本脚本预注册其操作定义 (见上),
    并附 C_share (收益贡献份额) 诊断度量 — 两个口径均在 .out 输出备查。
  - GBM 对照为首标×30 种子同管线 (PLAN §4 描述层 exit 模板允许的最小覆盖);
    本研究无按标的的分层结论 (均按状态聚合), 无需按标的扩 GBM 规模。
  - 趋势状态/ATR/ADX 的向量化复现为性能措施: _atr_series/_adx_series 与
    market_phase 同式 (布尔掩码改写, 无切片), trend_states_vec 与
    state_features.state_series 在 GATE 逐位对拍一致 (0 错位, 否则 SystemExit)。

发布门槛自检 (描述层):
  - GATE 探测器: (1) 偏度估计器自检 (200k 标准正态样本 skew ∈ ±0.1);
    (2) 趋势状态向量化与 state_series 逐位一致 (0 错位);
    (3) GBM 30 种子同管线 null: up:late 偏度 ≤ +0.3 且 dn:late ≥ -0.3
    (1h 与 4h 双侧); (4) GBM 池 n ≥ MIN_N; 任一失败 SystemExit (违规即停)
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (同管线重放)
  - MIN_N 检查: 每个输出格含 n, 不足格标注 [MIN_N 不足] (全单元格输出)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 无入场/无交易含义, 不涉及胜率/期望/成本 (描述层门槛)

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c13_trend_skew.py
  python3 research/studies/c13_trend_skew.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path (脚本以 `python3 research/studies/c13_trend_skew.py` 直接运行时,
# sys.path[0]=脚本目录, 需手动补根 — c12 试点记录的模板摩擦)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy.stats import skew

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.sim_market import gbm_matching
from research.state_features import state_series

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "warmup": 600,               # make_ctx 截断起点 (覆盖 atr/状态 warm-up)
    "head_drop": 120,            # 截断后仍丢弃头部 (rolling 特征 warm-up 错标 bar)
    "top5_frac": 0.05,           # H2: |r| 最大 5%
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),  # 2023 为部分年, 不纳入
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c13_trend_skew"

TREND_STATES = ("trend_up:early", "trend_up:accelerate", "trend_up:late",
                "trend_down:early", "trend_down:accelerate", "trend_down:late")
REF_STATES = ("range", "transition")
STATE_DIR = {  # 趋势方向: +1 up / -1 dn / 0 无方向
    "trend_up:early": 1, "trend_up:accelerate": 1, "trend_up:late": 1,
    "trend_down:early": -1, "trend_down:accelerate": -1, "trend_down:late": -1,
}


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


# ── 趋势状态 (state_features.state_series 语义的向量化复现) ──
# _atr_series/_adx_series 与 market_phase 同式 (布尔掩码改写以规避 L3 禁切片,
# 已对拍 market_phase 逐位一致); trend_states_vec 与 state_series 在 GATE 对拍。
def _atr_series(df, period=14):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    m = np.arange(n) >= 1
    if n > 1:
        cp = np.full(n, np.nan)
        cp[m] = c[np.arange(n)[m] - 1]
        tr[m] = np.maximum(h[m] - l[m],
                           np.maximum(np.abs(h[m] - cp[m]),
                                      np.abs(l[m] - cp[m])))
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values


def _adx_series(df, period=14):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    up = np.zeros(n)
    dn = np.zeros(n)
    m = np.arange(n) >= 1
    if n > 1:
        hp = np.full(n, np.nan)
        lp = np.full(n, np.nan)
        hp[m] = h[np.arange(n)[m] - 1]
        lp[m] = l[np.arange(n)[m] - 1]
        up[m] = h[m] - hp[m]
        dn[m] = lp[m] - l[m]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.zeros(n)
    if n > 1:
        cp = np.full(n, np.nan)
        cp[m] = c[np.arange(n)[m] - 1]
        tr[m] = np.maximum(h[m] - l[m],
                           np.maximum(np.abs(h[m] - cp[m]),
                                      np.abs(l[m] - cp[m])))
    alpha = 1 / period
    tr_s = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    pdi_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values
    mdi_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values
    s = pdi_s + mdi_s
    dx = np.where(s > 0, 100 * np.abs(pdi_s - mdi_s) / np.where(s > 0, s, 1), 0.0)
    adx = np.zeros(n)
    if n > period:
        tt = np.arange(n)
        m1 = tt < period
        m2 = tt >= period
        adx[m1] = dx[m1]
        tail = pd.Series(dx[tt >= period - 1])
        tw = tail.ewm(alpha=alpha, adjust=False).mean().values
        adx[m2] = tw[np.arange(n)[m2] - period + 1]
    else:
        adx = dx
    return adx


def trend_states_vec(df):
    """state_features.state_series 的向量化复现 (逐位一致, GATE 对拍自检).

    返回 np.ndarray[str]: trend_up:{early|accelerate|late} /
      trend_down:{...} / range / transition / unknown (atr≤1e-9)。
    """
    c = pd.Series(df["close"].values)
    o = pd.Series(df["open"].values)
    v = pd.Series(df["volume"].values)
    atr = pd.Series(_atr_series(df))
    adx = pd.Series(_adx_series(df))
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    dev = (c - ma20) / atr
    slope = (ma20 - ma20.shift(10)) / atr
    spread = (ma20 - ma60) / atr
    mom = (c - c.shift(10)) / atr
    adx_prev = adx.shift(10)
    body = (c - o).abs()
    body_recent = body.rolling(3).mean()
    body_prior = body.rolling(13).mean().shift(3)

    ADX_TREND = 25
    ADX_RANGE = 20
    EARLY_DEV = 0.6
    LATE_DEV = 2.0
    LATE_ADX_FALL = 3.0
    WEAK_BODY_RATIO = 0.6

    a = atr.values
    ad = adx.values
    ap = adx_prev.values
    sl = slope.values
    sp = spread.values
    dv = dev.values
    mo = mom.values
    br = body_recent.values
    bp = body_prior.values
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(bp > 0, br / np.maximum(bp, 1e-12), 1.0)

    trend_ok = (ad >= ADX_TREND) & (np.abs(sl) >= 0.15)
    range_ok = (ad < ADX_RANGE) & (np.abs(sl) < 0.15) & (np.abs(sp) < 0.5)
    state = np.where(trend_ok & (mo > 0), "trend_up",
                     np.where(trend_ok & (mo < 0), "trend_down",
                              np.where(range_ok, "range", "transition")))
    dev_abs = np.abs(dv)
    adx_turn = ad - ap
    late = ((dev_abs > LATE_DEV)
            | ((adx_turn < -LATE_ADX_FALL) & (ad >= ADX_TREND))
            | ((body_ratio < WEAK_BODY_RATIO) & (dev_abs > 1.0)))
    early = ((dev_abs < EARLY_DEV) & (ad >= ADX_TREND)) | \
            ((ap < ADX_TREND - 3) & (ad >= ADX_TREND))
    stage = np.where(late, "late", np.where(early, "early", "accelerate"))
    is_tr = (state == "trend_up") | (state == "trend_down")
    final = np.where(is_tr, np.char.add(np.char.add(state, ":"), stage), state)
    final = final.astype(object)
    final[a <= 1e-9] = "unknown"
    return final


def make_ctx_states(df, params):
    """make_ctx + 趋势状态 (state_fn 在截断 df 上计算, 长度 = n)"""
    return make_ctx(df, params["warmup"], state_fns={"trend": trend_states_vec})


# ── 状态内收益收集 (单标的, 因果, 布尔掩码, 无切片) ──────────
def collect_states(ctx, params):
    """单标的: 各状态内单根收益 + 年份 → {state: (ret, years)}.

    - r[t] = close[t]/close[t-1] − 1 (t≥1); 丢弃 head_drop 头部 (rolling
      特征 warm-up 错标 bar) 与 unknown 状态。
    """
    n = ctx.n
    states = ctx.states["trend"]
    m = np.arange(n) >= 1
    cprev = np.full(n, np.nan)
    cprev[m] = ctx.close[np.arange(n)[m] - 1]
    ret = ctx.close / cprev - 1.0
    use = (np.arange(n) >= params["head_drop"]) & np.isfinite(ret) \
        & (states != "unknown")
    r = ret[use]
    y = ctx.years[use]
    st = states[use]
    out = {}
    for key in TREND_STATES:
        mk = st == key
        out[key] = (r[mk], y[mk])
    for key in REF_STATES:
        mk = st == key
        out[key] = (r[mk], y[mk])
    # 聚合: up 合计 / dn 合计 (全部 trend 状态)
    st_str = st.astype(str)
    up = np.char.startswith(st_str, "trend_up")
    dn = np.char.startswith(st_str, "trend_down")
    out["up_agg"] = (r[up], y[up])
    out["dn_agg"] = (r[dn], y[dn])
    out["all"] = (r, y)
    return out


def merge(parts):
    """多标的/多种子池化: {state: (ret, years)} 数组拼接"""
    if not parts:
        return {}
    keys = parts[0].keys()
    out = {}
    for k in keys:
        rs = [p[k][0] for p in parts]
        ys = [p[k][1] for p in parts]
        out[k] = (np.concatenate(rs), np.concatenate(ys))
    return out


# ── 度量 ─────────────────────────────────────────────────────
def top5_dir_share(ret, trend_dir, frac=0.05):
    """H2 主度量 + 诊断.

    主度量 C_dir = Σ|r|(top5%|r| ∩ 顺趋势方向) / Σ|r|(top5%|r|)
      (GBM 对称 → ≈50%; 真实 late 大 K 顺趋势 → ≥60% 假设)
    诊断 C_share = Σ|r|(top5%|r| ∩ 顺趋势方向) / Σ|r|(全部顺趋势方向 bar)
      ("收益贡献份额" 版本)
    返回 (C_dir, C_share, k). trend_dir: +1 (up) / -1 (dn).
    """
    n = len(ret)
    if n < 20:
        return float("nan"), float("nan"), 0
    rabs = np.abs(ret)
    k = max(1, int(round(frac * n)))
    order = np.argsort(-rabs)
    thresh = rabs[order[k - 1]]
    top5 = rabs >= thresh
    if trend_dir == 1:
        td = ret > 0
    else:
        td = ret < 0
    s5 = float(rabs[top5].sum())
    s5_td = float(rabs[top5 & td].sum())
    s_all_td = float(rabs[td].sum())
    c_dir = s5_td / s5 if s5 > 0 else float("nan")
    c_share = s5_td / s_all_td if s_all_td > 0 else float("nan")
    return c_dir, c_share, k


def state_metrics(pr, py, trend_dir, params):
    """状态级度量: (n, mean, std, skew, C_dir, C_share, k)"""
    r = pr
    n = int(r.size)
    if n < 20:
        return {"n": n, "mean": float("nan"), "std": float("nan"),
                "skew": float("nan"), "c_dir": float("nan"),
                "c_share": float("nan"), "k": 0}
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    sk = float(skew(r))
    if trend_dir == 0:
        c_dir, c_share, k = float("nan"), float("nan"), 0
    else:
        c_dir, c_share, k = top5_dir_share(r, trend_dir,
                                           params["top5_frac"])
    return {"n": n, "mean": mean, "std": std, "skew": sk,
            "c_dir": c_dir, "c_share": c_share, "k": k}


def year_metrics(pr, py, trend_dir, params, years_wanted):
    """分年 (2024/2025/2026): {年: (n, skew, C_dir)}"""
    out = {}
    for y in years_wanted:
        mk = py == y
        r = pr[mk]
        n = int(r.size)
        if n < 20:
            out[y] = (n, float("nan"), float("nan"))
            continue
        sk = float(skew(r))
        if trend_dir == 0:
            c_dir = float("nan")
        else:
            c_dir, _, _ = top5_dir_share(r, trend_dir, params["top5_frac"])
        out[y] = (n, sk, c_dir)
    return out


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_1h, ref_4h, params):
    """探测器自检 + GBM 30 种子同管线 null, 失败 SystemExit.

    自检项:
      (1) 偏度估计器自检: 200k 标准正态样本 skew ∈ ±0.1
      (2) trend_states_vec 与 state_features.state_series 逐位一致 (0 错位, 1h/4h)
      (3) GBM 30 种子同管线 null: up:late 偏度 ≤ +0.3 且 dn:late ≥ -0.3
          (1h 与 4h 双侧 — H1 的 GBM 上界), GBM 池 n ≥ MIN_N
    返回 GBM 池 (1h/4h, 主结果直接复用) + 真实首标 (无条件基线)。
    """
    # (1) 偏度估计器自检
    rng = np.random.default_rng(0)
    wn_skew = float(skew(rng.normal(size=200000)))
    print(f"[GATE] 偏度估计器自检: 200k 正态样本 skew={wn_skew:.4f} "
          f"(∈ ±0.1)", flush=True)
    if abs(wn_skew) > 0.1:
        raise SystemExit(
            f"GATE FAIL: 偏度估计器 {wn_skew:.4f} ∉ ±0.1 — 度量错误, 停")

    # (2) 趋势状态等价性 (mask 截断, 禁切片)
    for name, ref in (("1h", ref_1h), ("4h", ref_4h)):
        keep = np.arange(len(ref)) >= params["warmup"]
        trunc = ref.iloc[keep]
        st_ref, _ = state_series(trunc)
        st_vec = trend_states_vec(trunc)
        n_mis = int((st_ref != st_vec).sum())
        print(f"[GATE] 趋势状态向量化 vs state_series ({name}): "
              f"错位 {n_mis} / {len(st_ref)}", flush=True)
        if n_mis != 0:
            raise SystemExit(
                f"GATE FAIL: 趋势状态向量化与 state_series 不一致 ({name} "
                f"错位 {n_mis}) — 停")

    # (3) GBM 30 种子全管线 (首标, 1h 与 4h)
    gbm_1h = pool_gbm(ref_1h, params)
    gbm_4h = pool_gbm(ref_4h, params)
    for name, pool in (("1h", gbm_1h), ("4h", gbm_4h)):
        m_up = state_metrics(*pool["trend_up:late"], 1, params)
        m_dn = state_metrics(*pool["trend_down:late"], -1, params)
        print(f"[GATE] GBM30种子 {name} up:late 偏度 {m_up['skew']:+.2f} "
              f"(n={m_up['n']}) | dn:late 偏度 {m_dn['skew']:+.2f} "
              f"(n={m_dn['n']})", flush=True)
        if m_up["skew"] > 0.3 or m_dn["skew"] < -0.3:
            raise SystemExit(
                f"GATE FAIL: GBM30种子 {name} up:late 偏度 {m_up['skew']:+.2f} "
                f"> +0.3 或 dn:late {m_dn['skew']:+.2f} < -0.3 — H1 的 GBM "
                f"上界不可达, 停")
        if m_up["n"] < MIN_N or m_dn["n"] < MIN_N:
            raise SystemExit(
                f"GATE FAIL: GBM30种子 {name} n_up={m_up['n']} n_dn="
                f"{m_dn['n']} < MIN_N={MIN_N}, 停")

    # 真实首标 (无条件基线)
    real_first = {}
    for name, ref in (("1h", ref_1h), ("4h", ref_4h)):
        real_first[name] = collect_states(
            make_ctx_states(ref, params), params)
    print("[GATE] 无条件基线 全体bar收益偏度: 真实1h {:.3f} vs GBM30种子 "
          "{:.3f} | 真实4h {:.3f} vs GBM {:.3f}".format(
              skew(real_first["1h"]["all"][0]),
              skew(gbm_1h["all"][0]),
              skew(real_first["4h"]["all"][0]),
              skew(gbm_4h["all"][0])), flush=True)
    return {"gbm_1h": gbm_1h, "gbm_4h": gbm_4h,
            "real_first_1h": real_first["1h"],
            "real_first_4h": real_first["4h"],
            "wn_skew": wn_skew}


# ── 池化 ─────────────────────────────────────────────────────
def pool_gbm(ref_df, params):
    """GBM 对照池 — 首标 × gbm_seeds 种子, 逐种子同管线重放"""
    parts = []
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_df, seed=seed)
        ctx = make_ctx_states(rw, params)
        parts.append(collect_states(ctx, params))
    return merge(parts)


def pool_real(dfs, params, first_part=None):
    """真实池 — 全标的 (可选复用 gate 首标结果)"""
    parts = []
    for k, df in enumerate(dfs):
        if k == 0 and first_part is not None:
            parts.append(first_part)
        else:
            parts.append(collect_states(make_ctx_states(df, params), params))
    return merge(parts)


# ── 统计/格式化 (全部数字带 n, MIN_N 标注) ──────────────────
def nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def pct(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:.1f}%"


def pp(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v * 100:+.1f}pp"


def skfmt(v):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:+.2f}"


def dir_of(key):
    if key in STATE_DIR:
        return STATE_DIR[key]
    if key == "up_agg":
        return 1
    if key == "dn_agg":
        return -1
    return 0


def by_year_lines(results, params):
    """BY_YEAR 成对 (真实 全标的 + GBM 首标30种子): 主度量逐年份"""
    rows = []
    for tf in params["tf_list"]:
        rs = results[tf]["real"]
        gs = results[tf]["gbm"]
        for y in params["by_year_list"]:
            r_up = year_metrics(*rs["trend_up:late"], 1, params,
                                (y,))[y]
            g_up = year_metrics(*gs["trend_up:late"], 1, params,
                                (y,))[y]
            r_dn = year_metrics(*rs["trend_down:late"], -1, params,
                                (y,))[y]
            g_dn = year_metrics(*gs["trend_down:late"], -1, params,
                                (y,))[y]
            rows.append(
                "{} {} up:late偏度 真实{} GBM{} | dn:late偏度 真实{} GBM{} "
                "| H2-up 真实{} GBM{} | H2-dn 真实{} GBM{}".format(
                    tf, y,
                    skfmt(r_up[1]), skfmt(g_up[1]),
                    skfmt(r_dn[1]), skfmt(g_dn[1]),
                    pct(r_up[2]), pct(g_up[2]),
                    pct(r_dn[2]), pct(g_dn[2])))
    return rows


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, g, results, year_rows):
    p = params
    up_share = float(np.mean(results["1h"]["real"]["all"][0] > 0))
    gup_share = float(np.mean(results["1h"]["gbm"]["all"][0] > 0))
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},warmup={},head_drop={},top5_frac={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], ",".join(p["tf_list"]), p["warmup"],
            p["head_drop"], p["top5_frac"], p["gbm_seeds"],
            MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(首标 全体bar收益偏度 1h): "
        "真实 {:.3f} GBM {:.3f} [PASS]; 顺K占比 真实 {:.1f}% GBM {:.1f}% "
        "[PASS]; 探测器: 偏度估计器 {:.3f}∈±0.1 [PASS], 状态对拍错位0 "
        "[PASS], GBM30种子 up:late偏度 {:.2f}≤+0.3 & dn:late偏度 {:.2f}≥-0.3 "
        "(1h/4h) [PASS]; MIN_N n_gbm_up={} n_gbm_dn={} [PASS]".format(
            p["gbm_seeds"], skew(g["real_first_1h"]["all"][0]),
            skew(g["gbm_1h"]["all"][0]), up_share * 100, gup_share * 100,
            g["wn_skew"],
            float(skew(g["gbm_1h"]["trend_up:late"][0])),
            float(skew(g["gbm_1h"]["trend_down:late"][0])),
            int(g["gbm_1h"]["trend_up:late"][0].size),
            int(g["gbm_1h"]["trend_down:late"][0].size)),
        "# RESULTS: 20 标的 × 1h/4h × 2023-08..2026-08; 描述层无入场, 无交易含义; "
        "收益 r=close[t]/close[t-1]−1; 偏度=状态内组内偏度(事后统计); "
        "H2=top5%|r| 大 K 的顺趋势方向 |r| 占比; GBM=首标×30 种子同管线",
        "[门槛] 预注册: H1 up:late≥+1.5 dn:late≤-1.5 (GBM≤+0.3) | H2 "
        "top5%|r|顺趋势方向占比≥60% (GBM≈50%) | H3 |偏度| early→accel→late "
        "单调不降",
        "",
    ]
    for tf in p["tf_list"]:
        rs = results[tf]["real"]
        gs = results[tf]["gbm"]
        lines.append(f"[{tf}] 状态内单根收益 (真实 | GBM30种子):")
        for key in REF_STATES + TREND_STATES + ("up_agg", "dn_agg", "all"):
            rm = state_metrics(*rs[key], dir_of(key), p)
            gm = state_metrics(*gs[key], dir_of(key), p)
            lines.append(
                "  {:<24} n 真实 {:>7} / GBM {:>7} | 均值 真实 {:+.4f}% "
                "GBM {:+.4f}% | 偏度 真实 {} GBM {} Δ {} {}".format(
                    key, rm["n"], gm["n"], rm["mean"] * 100,
                    gm["mean"] * 100, skfmt(rm["skew"]), skfmt(gm["skew"]),
                    skfmt(rm["skew"] - gm["skew"]), nm(rm["n"])))
        for side, state in (("up", "trend_up:late"), ("dn", "trend_down:late")):
            d = 1 if side == "up" else -1
            rm = state_metrics(*rs[state], d, p)
            gm = state_metrics(*gs[state], d, p)
            lines.append(
                "  [H1-{}] {} 偏度: 真实 {} (n={}) | GBM {} (n={}) | "
                "Δ {} {}".format(tf, state, skfmt(rm["skew"]), rm["n"],
                                 skfmt(gm["skew"]), gm["n"],
                                 skfmt(rm["skew"] - gm["skew"]), nm(rm["n"])))
        for side, state in (("up", "trend_up:late"), ("dn", "trend_down:late")):
            d = 1 if side == "up" else -1
            rm = state_metrics(*rs[state], d, p)
            gm = state_metrics(*gs[state], d, p)
            lines.append(
                "  [H2-{}] top5%|r| 顺趋势方向占比 {}: 真实 {} (n={}) | "
                "GBM {} (n={}) | 净差 {} {}".format(
                    tf, state, pct(rm["c_dir"]), rm["n"],
                    pct(gm["c_dir"]), gm["n"],
                    pp(rm["c_dir"] - gm["c_dir"]), nm(rm["n"])))
        lines.append(
            "  [H2d-{}] C_share 诊断 (top5%|r| 顺趋势 |r| / 全部顺趋势 |r|): "
            "up:late 真实 {} GBM {} | dn:late 真实 {} GBM {}".format(
                tf, pct(state_metrics(*rs["trend_up:late"], 1, p)["c_share"]),
                pct(state_metrics(*gs["trend_up:late"], 1, p)["c_share"]),
                pct(state_metrics(*rs["trend_down:late"], -1, p)["c_share"]),
                pct(state_metrics(*gs["trend_down:late"], -1, p)["c_share"])))
        us = [state_metrics(*rs[k], dir_of(k), p)["skew"] for k in
              ("trend_up:early", "trend_up:accelerate", "trend_up:late")]
        ds = [state_metrics(*rs[k], dir_of(k), p)["skew"] for k in
              ("trend_down:early", "trend_down:accelerate", "trend_down:late")]
        lines.append(
            "  [H3-{}] |偏度| 单调 (真实): up early {} accel {} late {} | "
            "dn early {} accel {} late {}".format(
                tf, skfmt(abs(us[0])), skfmt(abs(us[1])), skfmt(abs(us[2])),
                skfmt(abs(ds[0])), skfmt(abs(ds[1])), skfmt(abs(ds[2]))))
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(year_rows))
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
    g = gate(dfs["1h"][0], dfs["4h"][0], PARAMS)

    # 真实池 (全标的; 首标复用 gate) + GBM 池 (gate 已算, 复用)
    results = {}
    for tf in PARAMS["tf_list"]:
        real = pool_real(dfs[tf], PARAMS, g["real_first_" + tf])
        results[tf] = {"real": real, "gbm": g["gbm_" + tf]}

    year_rows = by_year_lines(results, PARAMS)
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, results, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
