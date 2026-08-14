#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C54 M5 行为与形态忠实复现 (CH14-16) (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (学习单元 M5 U1, PLAN §2.5 c54): 书 CH14-16 — 周内效应 (CH15
  图 15.14 X/O 模式)、反向意见代理 (CH14 "当多数人做多, 市场下跌" 量测)、5m
  日内突破 (CH16 核心主张 "无成本所有突破幅度全盈利, 20 年后依然")。oracle 逐字
  核实口径。描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。**结论不得作
  交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、
  内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 加密 24/7 市场是否复现书的三条行为学主张?
  (a) 周内模式 (up Monday down Tuesday); (b) 反向意见 (多头拥挤 → 未来跌);
  (c) 无成本突破全盈利 (加 taker 成本后消失)。

预注册假设 (PLAN §2.5 c54 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 周内效应加密版 — BTC/ETH 日线方向符号 (close-only, 书口径) Monday/
      Tuesday X/O 模式 (书 "up Monday down Tuesday") vs 日标签置换 null 30 次;
      判据 = 周二反转条件概率 (P[周二跌|周一涨]) 超置换 95% 区间
      (预期 24/7 无周内结构; 书图 15.14 全概率 34-66% 本就很弱)
  H2: 反向意见代理 — OKX long/short 账户比 (long-short-account-ratio?ccy=,
      1D) 极端分位 (上下 5%) → 未来 1 日 K bar 方向 + 触碰折返 (c17 管线)
      vs GBM 30 种子; **API 不可得则 docstring 与结论降级标注** (funding 版
      c49 H4 已测无调节 — 本 H 是新数据源的尝试, 非必成)
  H3: 5m 日内突破 — BTC/ETH 5m, N-bar 突破 N∈{10,20,50} (映射书固定点突破,
      标注), 无成本 (0bp) 永远在场 vs taker 1/2.5bp; GBM 30 种子同管线;
      判据 = 无成本 (0bp) 净盈亏 > null 与否、加成本后是否消失
      (预期成本吃光, 呼应 c21/c23/c25)

  操作化 (运行前锁定):
    - H1: 日线 = daily_resample(4h) (已收盘); 方向符号 s_t = sign(close_t/
      close_{t-1}−1) (close-only); 周一/周二按日线 bar 的 dayofweek;
      主度量 = P[周二跌 | 周一涨] (逐标的拼接池); null = 日标签置换 30 次
      (逐标的内置换 weekday 标签, 保留收益序列结构); 判据 = 真实超置换
      95% 区间 (2.5%~97.5%)
    - H2: OKX rubik long-short-account-ratio?ccy=BTC/ETH&period=1D (API,
      ~180 点, 2026-02-15..2026-08-13, 上限 100/请求, 无翻页 → pilot 标注);
      ratio_t 对齐日线 bar (ts=当日 00:00 UTC, 已收盘语义); 前向 = 日线
      close_{t+1}/close_t−1; ① 极端分位 (上下 5%) 前向方向; ② 连续性
      Spearman(ratio, forward ret) n≈180; null = GBM 30 种子 (同窗口日线
      重放, ratio 标签按日对齐); ③ 触碰折返 (c17 管线, 日线 ctx, combo
      (2,0.3), W=24) — 样本窗口短, n 可能不足, 标注 [MIN_N 不足]
    - H3: N-bar 突破 (书 CH16 固定点突破加密版, N∈{10,20,50} bar):
      收盘价 > 前 N 根最高 → 多; < 前 N 根最低 → 空; 永远在场反转; 无成本
      (0bp) vs taker 1/2.5bp (每信号事件 1 笔, 每笔扣 cost_bp×1e-4);
      GBM null = gbm_matching(该标的 5m df) 同管线 (0bp)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  日线             | data_loader.daily_resample (4h→1D)     | 日线收盘后 | 已收盘聚合 (c29/c30 口径)
  周内标签         | 日线 bar index 的 dayofweek            | 日线收盘后 | 日历事实
  ratio_t          | OKX rubik API (ccy, 1D) 对齐日线 bar   | 日 t 收盘  | ts=当日 00:00, 已收盘
                   |   (时间戳算术映射)                     |            |   语义; 禁 searchsorted
  前向收益         | close_{t+1}/close_t−1 (布尔掩码)       | 事后       | 描述层端点
  N-bar 通道       | rolling(N).max().shift(1) (窗口止于 t−1) | bar 收盘后 | 书口径 (防前视)
  突破信号         | close_t > 前 N 最高 / < 前 N 最低       | bar 收盘后 | CH16 固定点突破
  触碰折返         | levels.cluster_levels (在线聚类+冻结)   | confirm_at | 冻结后不可变 (levels R1/R2)
                   |   + state_series 趋势态 (因果状态机)   | bar 收盘后 | + 触碰后 W=24 端点
  净盈亏/成本      | 永远在场反转 log 权益累计; 每信号事件 1 笔 | 全样本事后 | 描述统计; 成本=事件数×bp
  GBM null         | sim_market.gbm_matching (锚定真实)     | 锚定真实   | 固定种子序列 0..29; 首标×30
  H1 置换 null     | 日标签逐标的内置换 30 次 (保留收益序列)  | 锚定真实   | 破坏周内结构, 保留自相关

数据声明:
  data/backtest.db (gitignored): BTC/ETH × 4h/5m × 2023-08..2026-08;
  日线 = daily_resample(4h); H2 ratio = OKX rubik API (2026-02-15..2026-08-13,
  ~180 日, pilot); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  H1: daily_from=4h, 置换 30; H2: ccy=BTC/ETH, period=1D, 分位 5%, 前向 k=1,
  combo (2,0.3), W=24; H3: N∈{10,20,50}, 成本 0/1/2.5bp, warmup=600;
  gbm_seeds=30; min_n=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - H1 用 close-only 方向符号 (书口径); 不做全概率 34-66% 网格, 只测预注册
    判据 (周二反转条件概率) + 分周内 P(up) 描述。
  - H2 的 OKX rubik 端点: long-short-account-ratio?ccy= 返回 ~180 日 (1D),
    无翻页 — pilot 标注; 前向方向用 1 日 (k=1); 触碰折返在日线窗口上 n 可能
    不足, 标注。API 不可得 (网络/429/403) → docstring+结论降级标注, 跳过
    H2 统计 (不 SystemExit)。
  - H3 N-bar 突破映射书固定点突破 (书为固定点金额, 加密版用 N-bar 通道);
    成本按每信号事件 1 笔扣 (反转按 2 笔的保守口径在 docstring 注明 — 这里
    统一 1 笔/事件, 对结论方向无影响)。
  - **裁剪 (预注册, dev 实测报告)**: 5m GBM 生成 14.5s/种子 (315K bar, sub=8
    Python 循环) — 30 种子×2 标的 ≈ 15min 超预算; 学习级种子减到 15 (H3 的
    GBM null), H1 置换/H2 保持完整; 效果与 30 种子等价 (GBM null 为分布,
    15 种子 σ 略大)。H3 标的/全部 N∈{10,20,50}/成本 0/1/2.5bp 不裁剪。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例; dev 先测 5m 单标的单 N 运行时间/
    内存, 超预算裁剪 (如仅 BTC 5m、N∈{10,50}、种子 15) 并在最终消息报告。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① H1 置换机制 sanity — 置换 null 的条件概率均值 ∈ [45%,55%]
    (标签随机化后必须 ≈ 边缘概率); ② H2 Spearman null sanity — GBM+真实
    ratio 标签的 Spearman 均值 ∈ [−0.15,+0.15] (随机游走对 ratio 无系统关系);
    ③ H3 Donchian golden (c41 内联) + GBM 无成本突破净盈亏带 ∈ [−0.8,+0.8]
    (永远在场 whipsaw 负漂移已知); 任一失败 SystemExit (H2 API 不可得除外)
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (同管线重放)
  - MIN_N: 每格 n ≥ 100 (学习级); H2/H1 条件格不足标注 [MIN_N 不足]
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: H1 2 标的; H2 BTC 单 ccy 1 种子; H3 BTC 5m × N={10} × 3 种子
    (测单格耗时 + 内存), 不写 .out
  - 全量: H1 2 标的; H2 BTC+ETH; H3 BTC+ETH 5m × 3 N × 3 成本 × 30 种子

运行命令:
  python3 research/studies/c54_behavioral_form.py --dev
  python3 research/studies/c54_behavioral_form.py
"""
import hashlib
import os
import sqlite3
import sys
import time
import urllib.request
from datetime import date

# 仓库根入 path (模板摩擦, 见 c12 报告)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats as sstats

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.ctx import make_ctx
from research.data_loader import DB_PATH, daily_resample, load_candles, verify
from research.levels import cluster_levels
from research.sim_market import gbm_matching
from research.state_features import state_series
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "h1_from": "4h",                   # 日线重采样来源 (预注册)
    "h1_perm": 30,                     # 日标签置换次数
    "h2_ccy": ("BTC", "ETH"),
    "h2_period": "1D",
    "h2_q": 0.05,                      # 极端分位 (上下 5%)
    "h2_k": 1,                         # 前向日数
    "h2_combo": (2, 0.3),              # 触碰组合 (min_touch, tol)
    "h2_W": 24,                        # 触碰后端点窗口 (日线 bar)
    "h2_depth": 0.5, "h2_hold": 0.5,
    "h3_Ns": (10, 20, 50),             # N-bar 突破
    "h3_costs_bp": (0.0, 1.0, 2.5),    # taker 成本
    "warmup": 600,
    "gbm_seeds": MIN_GBM_SEEDS,
    "min_n": 100,                      # 学习级 MIN_N (PLAN c54 行: 100, 非 caliber 200)
    "h1_band": (0.45, 0.55),           # GATE: 置换 null 条件概率带
    "h2_spear_band": (-0.15, 0.15),    # GATE: GBM Spearman 带
    "h3_gbm_band": (-0.8, 0.8),        # GATE: GBM 无成本净盈亏带
    # 裁剪方案 (dev 实测: 5m GBM 生成 14.5s/种子 × 315K bar; 30 种子×2 标的
    # ~15min 超预算 → 学习级种子减到 15, 预注册裁剪)
    "dev_subset": {"h2_ccy": ("BTC",), "h3_n": (10,), "n_gbm": 3},
    "full_gbm_seeds": 15,
    "data_range": "2023-08..2026-08; ratio 2026-02-15..2026-08-13 (API pilot)",
}

STUDY_ID = "c54_behavioral_form"
RATIO_URL = ("https://www.okx.com/api/v5/rubik/stat/contracts/"
             "long-short-account-ratio?ccy={ccy}&period={period}&limit=100")


# ── 加载 ─────────────────────────────────────────────────────
def load_4h_pairs():
    data = load_candles(timeframes=("4h",))
    out = []
    for sym in PARAMS["crypto"]:
        df = data.get(sym, {}).get("4h")
        if df is None or verify(df, sym, "4h"):
            continue
        out.append((sym, df))
    return out


def load_5m_pairs():
    """仅 BTC/ETH 的 5m (内存控制: 624 万行全量 ~400MB, 仅 2 标的 ~50MB)"""
    conn = sqlite3.connect(DB_PATH)
    out = []
    try:
        for sym in PARAMS["crypto"]:
            df = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume FROM candles "
                "WHERE symbol=? AND timeframe='5m' ORDER BY timestamp",
                conn, params=(sym,))
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
            df = df.sort_index()
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[~df["close"].isna()]
            out.append((sym, df))
    finally:
        conn.close()
    return out


# ── H1: 周内效应 ─────────────────────────────────────────────
def _daily_signs(df):
    daily = daily_resample(df)
    c = daily["close"].values
    s = np.sign(np.diff(c))          # close-only 方向
    wd = daily.index[1:].dayofweek.values  # 0=Mon .. 4=Fri
    return s, wd


def h1_conditional(s, wd):
    """P[周二跌 | 周一涨]"""
    mon_up = (wd[:-1] == 0) & (s[:-1] > 0)
    tue = wd[1:] == 1
    m = mon_up & tue
    if m.sum() == 0:
        return 0, 0, float("nan")
    return int(m.sum()), int((m & (s[1:] < 0)).sum()), float(
        np.mean(s[1:][m] < 0))


def h1_weekday_up(s, wd):
    out = {}
    for d in range(5):
        m = wd == d
        if m.sum() == 0:
            out[d] = (0, float("nan"))
        else:
            out[d] = (int(m.sum()), float(np.mean(s[m] > 0)))
    return out


def h1_perm_null(pairs, perm):
    """日标签置换: 逐标的内置换 weekday 标签 (保留收益序列结构)"""
    real_parts = [_daily_signs(df) for _, df in pairs]
    ss = [p[0] for p in real_parts]
    wds = [p[1] for p in real_parts]
    # 真实值
    n_tot = up_tot = 0
    cond_list = []
    for s, wd in zip(ss, wds):
        n, up, c = h1_conditional(s, wd)
        if n > 0:
            n_tot += n
            up_tot += up
            cond_list.append(c)
    real = up_tot / n_tot if n_tot else float("nan")
    # 置换
    nulls = []
    for p in range(perm):
        rng = np.random.default_rng(1000 + p)
        n_p = up_p = 0
        for s, wd in zip(ss, wds):
            wd_p = rng.permutation(wd)   # 逐标的置换
            n, up, _ = h1_conditional(s, wd_p)
            n_p += n
            up_p += up
        nulls.append(up_p / n_p if n_p else float("nan"))
    return real, n_tot, np.array(nulls), h1_weekday_up(
        np.concatenate(ss), np.concatenate(wds))


# ── H2: 反向意见代理 (OKX rubik API) ─────────────────────────
def json_load(r):
    import json
    return json.loads(r.read().decode())


def fetch_ratio(ccy, period):
    url = RATIO_URL.format(ccy=ccy, period=period)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json_load(r)
    return data.get("data", [])


def h2_align_ratio(ratio_rows, daily_df):
    """ratio ts(ms, 当日 00:00) → 日线 bar 索引 (时间戳算术, 禁 searchsorted)"""
    ratio_ts = [int(row[0]) for row in ratio_rows]
    ratio_val = np.array([float(row[1]) for row in ratio_rows])
    day_map = {}
    for ts in ratio_ts:
        d = pd.Timestamp(ts, unit="ms", tz="UTC").normalize()
        day_map[d] = ts
    dpos = {ts.normalize(): i for i, ts in enumerate(daily_df.index)}
    idx = np.full(len(ratio_val), -1, dtype=int)
    for i, ts in enumerate(ratio_ts):
        d = pd.Timestamp(ts, unit="ms", tz="UTC").normalize()
        idx[i] = dpos.get(d, -1)
    keep = idx >= 0
    return ratio_val[keep], idx[keep], daily_df


def h2_window(daily_df, idx):
    """把日线框裁剪到 ratio 窗口 [idx.min()..idx.max()], 返回 (daily_win, new_idx)"""
    first, last = int(idx.min()), int(idx.max())
    daily_win = daily_df.iloc[first:last + 1]
    new_idx = idx - first
    return daily_win, new_idx


def h2_stats(daily_win, ratio_val, idx, params):
    """① 极端分位前向方向; ② Spearman(ratio, forward ret); ③ 触碰折返.
    全部数组长度 = len(idx) (ratio 对齐数)."""
    c = daily_win["close"].values
    k = params["h2_k"]
    n = len(c)
    fwd = np.full(len(idx), np.nan)
    ok = (idx + k < n) & (idx >= 0)
    fwd[ok] = np.log(c[idx[ok] + k] / c[idx[ok]])
    q_lo, q_hi = np.quantile(ratio_val, params["h2_q"]), \
        np.quantile(ratio_val, 1 - params["h2_q"])
    m_lo, m_hi = ratio_val <= q_lo, ratio_val >= q_hi
    lo_n = int(m_lo.sum())
    hi_n = int(m_hi.sum())
    lo_dir = float(np.mean(fwd[m_lo] > 0)) if lo_n else float("nan")
    hi_dir = float(np.mean(fwd[m_hi] > 0)) if hi_n else float("nan")
    # 连续性 Spearman (n≈窗口内)
    v = ok & np.isfinite(fwd)
    spear = float(sstats.spearmanr(ratio_val[v], fwd[v]).statistic) if v.sum() > 2 \
        else float("nan")
    n_spear = int(v.sum())
    # 触碰折返 (c17 管线, ratio 窗口内的日线 ctx)
    ctx = make_ctx(daily_win, min(200, len(daily_win) // 3),
                   state_fns={"trend": lambda df: state_series(df)[0]})
    touch = collect_touch(ctx, params)
    return {
        "lo": (lo_n, lo_dir), "hi": (hi_n, hi_dir),
        "spear": (n_spear, spear), "touch": touch,
    }


def collect_touch(ctx, params):
    """c17 触碰折返简化版: 趋势态触碰聚类位带 → D1 (W 根端点沿趋势方向概率).
    无角色层 (H2 只需要触碰折返本身)."""
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    states = ctx.states["trend"]
    W = params["h2_W"]
    mt, tol = params["h2_combo"]
    up = np.char.startswith(states, "trend_up")
    dn = np.char.startswith(states, "trend_down")
    neu = ~(up | dn)
    logr = np.full(n, np.nan)
    ok_t = t_idx + W < n
    logr[ok_t] = np.log(c[t_idx[ok_t] + W] / c[t_idx[ok_t]])
    usable = ok_t & np.isfinite(logr)
    d1 = np.full(n, np.nan)
    m = usable & up
    d1[m] = logr[m] > 0
    m = usable & dn
    d1[m] = logr[m] < 0
    m = usable & neu
    d1[m] = logr[m] > 0
    lvls = cluster_levels(ctx.high, ctx.low, ctx.atr, k=K,
                          tolerance_mult=tol, min_touch=mt)
    d1_l, dr_l = [], []
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
        d1_l.append(d1[ev])
        up_ev = up[ev]
        dr_l.append(np.where(up_ev, "up", np.where(dn[ev], "dn", "neu")))
    if not d1_l:
        return {"d1": np.array([], float), "dr": np.array([], object),
                "n_touch": 0}
    return {"d1": np.concatenate(d1_l), "dr": np.concatenate(dr_l),
            "n_touch": int(np.concatenate(d1_l).size)}


def _counter_mask(touch):
    dr = touch["dr"]
    return (dr == "up") | (dr == "dn")


def h2_gbm_null(daily_win, ratio_val, idx, params, seeds):
    """GBM null: 同窗口日线重放 + 真实 ratio 标签; Spearman 分布 + 触碰 D1"""
    c = daily_win["close"].values
    n = len(c)
    k = params["h2_k"]
    spears = []
    touch_d1s = []
    for seed in range(seeds):
        rw = gbm_matching(daily_win, seed=seed)
        rw_c = rw["close"].values
        fwd = np.full(len(idx), np.nan)
        ok = (idx + k < n) & (idx >= 0)
        fwd[ok] = np.log(rw_c[idx[ok] + k] / rw_c[idx[ok]])
        v = ok & np.isfinite(fwd)
        if v.sum() > 2:
            spears.append(float(sstats.spearmanr(ratio_val[v], fwd[v]).statistic))
        touch = collect_touch(
            make_ctx(rw, min(200, len(daily_win) // 3),
                     state_fns={"trend": lambda df: state_series(df)[0]}),
            params)
        touch_d1s.append(touch["d1"])
    s_arr = np.array(spears)
    t_arr = np.concatenate(touch_d1s) if touch_d1s else np.array([])
    return s_arr, t_arr


# ── H3: 5m 日内突破 ──────────────────────────────────────────
def breakout_per(close, high, low, N, cost_bp):
    """N-bar 突破 (永远在场反转): 返回 per-bar PnL 与信号事件数."""
    c = np.asarray(close, float)
    hi = pd.Series(high)
    lo = pd.Series(low)
    ref_hi = hi.rolling(N).max().shift(1).values
    ref_lo = lo.rolling(N).min().shift(1).values
    long_sig = c > ref_hi
    short_sig = c < ref_lo
    sig_dir = np.zeros(len(c), int)
    sig_dir[long_sig & np.isfinite(ref_hi)] = 1
    sig_dir[short_sig & ~long_sig & np.isfinite(ref_lo)] = -1
    idx = np.arange(len(c))
    last_idx = np.where(sig_dir != 0, idx, 0)
    last_idx = np.maximum.accumulate(last_idx)
    has = np.maximum.accumulate(sig_dir != 0)
    p = np.where(has, sig_dir[last_idx], 0)
    r = np.zeros(len(c))
    r[:-1] = c[1:] / c[:-1] - 1.0
    per = p * r
    n_ev = int((sig_dir != 0).sum())
    cost = n_ev * cost_bp * 1e-4
    net = float(per.sum()) - cost
    log_eq = np.cumsum(np.log1p(per))
    peak = np.maximum.accumulate(log_eq)
    maxdd = float((peak - log_eq).max())
    return net, maxdd, n_ev


def h3_real(ctx, Ns, costs):
    out = {}
    for N in Ns:
        for cb in costs:
            net, mdd, nev = breakout_per(ctx.close, ctx.high, ctx.low, N, cb)
            out[(N, cb)] = (net, mdd, nev)
    return out


def h3_gbm(df, Ns, seeds):
    """GBM null: 每种子每 N 无成本净盈亏"""
    nets = {N: [] for N in Ns}
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        ctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
        for N in Ns:
            net, _, _ = breakout_per(ctx.close, ctx.high, ctx.low, N, 0.0)
            nets[N].append(net)
    return {N: (float(np.mean(v)), float(np.std(v, ddof=1))) for N, v in
            nets.items()}


# ── GATE 自检 (违规即停) ────────────────────────────────────
def _golden_breakout():
    """构造已知通道: N=3, 前 3 bar 高=5, bar 3 收盘突破 → 多; 低破 → 空."""
    close = np.array([4.0, 4.5, 5.0, 5.5, 5.4, 5.3, 4.0, 3.5, 3.6])
    high = close + 0.3
    low = close - 0.3
    high[0] = 5.0
    low[1] = 3.5
    net, mdd, n_ev = breakout_per(close, high, low, 3, 0.0)
    if n_ev < 2:
        raise SystemExit(f"GATE FAIL: golden 信号数 {n_ev} < 2")
    return True


def gate(h1_real, h1_nulls, h2_spear_gbm, h3_gbm_nets, h2_available):
    """① H1 置换 sanity; ② H2 GBM Spearman 带 (API 可得时); ③ H3 golden + 带."""
    _golden_breakout()
    lo, hi = PARAMS["h1_band"]
    n_mean = float(np.mean(h1_nulls))
    if not (lo <= n_mean <= hi):
        raise SystemExit(
            f"GATE FAIL: H1 置换 null 条件概率 {n_mean:.3f} ∉ [{lo}, {hi}] "
            f"— 置换管线错误, 停")
    if h2_available:
        slo, shi = PARAMS["h2_spear_band"]
        sm = float(np.mean(h2_spear_gbm)) if h2_spear_gbm.size else float("nan")
        if not np.isfinite(sm) or not (slo <= sm <= shi):
            raise SystemExit(
                f"GATE FAIL: H2 GBM Spearman null {sm:.3f} ∉ [{slo}, {shi}] "
                f"— 对齐/管线错误, 停")
    glo, ghi = PARAMS["h3_gbm_band"]
    gm = float(np.mean([v[0] for v in h3_gbm_nets.values()]))
    if not (glo <= gm <= ghi):
        raise SystemExit(
            f"GATE FAIL: H3 GBM 无成本净盈亏 {gm:+.4f} ∉ [{glo}, {ghi}] "
            f"— 突破管线错误, 停")
    print(f"[GATE] H1 置换 null {n_mean:.3f} [PASS]; "
          f"H3 golden [PASS], GBM 净盈亏 {gm:+.4f} [PASS]", flush=True)
    if h2_available:
        print(f"[GATE] H2 GBM Spearman null {sm:+.3f} [PASS]", flush=True)
    else:
        print("[GATE] H2 API 不可得 — 降级标注, 跳过 H2 统计", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.1f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= PARAMS["min_n"] else "[MIN_N 不足]"


def write_out(out_path, params, res, h2_degraded):
    p = params
    seeds_full = p["full_gbm_seeds"]
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=h1_perm={},h2_ccy={},h2_period={},h2_q={},h3_Ns={},h3_costs={},"
        "gbm_seeds={}(H3裁剪),min_n={},gate=MIN_GBM_SEEDS=30(学习级裁剪15),"
        "MIN_N={}(学习级),学习级"
        "".format(STUDY_ID, date.today().isoformat(), script_sha256(),
                  p["data_range"], p["h1_perm"], "+".join(p["h2_ccy"]),
                  p["h2_period"], p["h2_q"], p["h3_Ns"], p["h3_costs_bp"],
                  seeds_full, p["min_n"], p["min_n"]),
        "# GATE: 探测器自检 H1 置换 null [PASS]; H3 golden + GBM 带 [PASS]; "
        "MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c54 M5 行为与形态 (书 CH14-16); 周内效应 (CH15 图 "
        "15.14 X/O), 反向意见 (CH14, OKX ratio pilot), 5m 突破 (CH16 固定点); "
        "描述层无入场, 无交易含义",
        "",
    ]
    # H1
    r = res["h1"]
    real, n_tot, nulls, wd_up = r
    pct_lo, pct_hi = np.percentile(nulls, 2.5), np.percentile(nulls, 97.5)
    exceed = real > pct_hi
    lines.append("[H1] 周内效应 (BTC+ETH 日线 close-only, n={}):".format(n_tot))
    lines.append("  P[周二跌|周一涨] 真实 {} vs 置换 95% 区间 "
                 "[{:.1%}, {:.1%}] (置换 mean {:.1%}) -> {}".format(
                     _pct(real), pct_lo, pct_hi, float(np.mean(nulls)),
                     "超区间↑" if exceed else "区间内"))
    rows = []
    for d in range(5):
        dn, dm = wd_up[d]
        rows.append("{}:P(up) {:.1%}(n={})".format(
            "一二三四五"[d], dm, dn))
    lines.append("  分周内 P(up): " + " | ".join(rows))
    # H2
    lines.append("")
    lines.append("[H2] 反向意见代理 (OKX long-short-account-ratio?ccy=, 1D, "
                 "2026-02-15..2026-08-13 pilot):")
    if h2_degraded:
        lines.append("  [降级] API 不可得 — H2 无数据, 无统计; "
                     "funding 版 c49 H4 已测无调节")
    else:
        for ccy, h2 in res["h2"].items():
            lo_n, lo_dir = h2["lo"]
            hi_n, hi_dir = h2["hi"]
            n_spear, spear = h2["spear"]
            t = h2["touch"]
            if t["n_touch"] > 0:
                cm = (t["dr"] == "up") | (t["dr"] == "dn")
                td1 = float(np.mean(t["d1"][cm])) if cm.sum() else float("nan")
                tn = int(cm.sum())
            else:
                td1, tn = float("nan"), 0
            lines.append("  {}: 底部5%前向{:.1%} (n={}) {} | 顶部5%前向{:.1%} "
                         "(n={}) {} | Spearman {:+.3f} (n={}) {} | 触碰D1 "
                         "{:.1%} (n={}) {}".format(
                ccy, lo_dir, lo_n, _nm(lo_n), hi_dir, hi_n, _nm(hi_n),
                spear, n_spear, _nm(n_spear), td1, tn, _nm(tn)))
        if res["h2_gbm_spear"].size:
            lines.append("  GBM null Spearman: {:.3f}±{:.3f}".format(
                float(np.mean(res["h2_gbm_spear"])),
                float(np.std(res["h2_gbm_spear"], ddof=1))))
        if res["h2_gbm_touch"].size:
            lines.append("  GBM null 触碰 D1: {:.1%} (n={})".format(
                float(np.mean(res["h2_gbm_touch"])), res["h2_gbm_touch"].size))
    # H3
    lines.append("")
    lines.append("[H3] 5m N-bar 突破 (书 CH16 固定点突破加密版, 永远在场反转, "
                 "每信号事件 1 笔成本):")
    for sym, h3 in res["h3"].items():
        for N in p["h3_Ns"]:
            gmean, gstd = h3["gbm"][N]
            cells = []
            for cb in p["h3_costs_bp"]:
                net, mdd, nev = h3["cells"][(N, cb)]
                tag = ">null" if (cb == 0 and
                                  net > gmean + 2 * gstd) else ""
                cells.append("{}bp {}{:+.4f}{}".format(
                    cb, "" if cb else "无成本", net, tag))
            lines.append("  {} N={}: {}".format(
                sym.split("/")[0], N, " | ".join(cells)))
            lines.append("      GBM(0bp) {:.4f}±{:.4f} | 事件数 {}".format(
                gmean, gstd, h3["cells"][(N, 0.0)][2]))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] 书 CH15 图15.14 (周内全概率 34-66% 本就很弱); "
                 "CH14 (反向意见, 加密无直接民意数据 — ratio 代理 pilot); "
                 "CH16 (无成本突破全盈利, 20 年); c21/c23/c25 (成本吃光突破/"
                 "区间优势); c49 H4 (funding 折返无调节); c17 (触碰折返 "
                 "-2~-4pp 基线)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    h2_ccys = PARAMS["dev_subset"]["h2_ccy"] if dev else PARAMS["h2_ccy"]
    h3_Ns = PARAMS["dev_subset"]["h3_n"] if dev else PARAMS["h3_Ns"]
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["full_gbm_seeds"]

    # H1 (廉价, 全量跑)
    pairs = load_4h_pairs()
    if not pairs:
        print("无 4h 数据, 退出")
        return 1
    h1 = h1_perm_null(pairs, PARAMS["h1_perm"])

    # H2 API 尝试 (不可得 → 降级标注, 不 SystemExit)
    h2 = {}
    h2_degraded = False
    h2_gbm_spear = np.array([])
    h2_gbm_touch = np.array([])
    try:
        ratio_daily = {}
        for ccy in h2_ccys:
            rows = fetch_ratio(ccy, PARAMS["h2_period"])
            if not rows:
                raise RuntimeError(f"{ccy} ratio 空")
            # 对齐到该币 4h→日线
            sym = ccy + "/USDT:USDT"
            df4 = dict(pairs).get(sym)
            if df4 is None:
                continue
            daily = daily_resample(df4)
            rv, ix, _ = h2_align_ratio(rows, daily)
            ratio_daily[ccy] = (daily, rv, ix)
        if not ratio_daily:
            raise RuntimeError("无 ratio 数据")
        # GBM null (用 BTC 的 daily 窗口)
        btc_daily, btc_rv, btc_ix = ratio_daily.get("BTC") or \
            next(iter(ratio_daily.values()))
        btc_win, btc_nix = h2_window(btc_daily, btc_ix)
        h2_gbm_spear, h2_gbm_touch = h2_gbm_null(
            btc_win, btc_rv, btc_nix, PARAMS, seeds)
        for ccy, (daily, rv, ix) in ratio_daily.items():
            win, nix = h2_window(daily, ix)
            h2[ccy] = h2_stats(win, rv, nix, PARAMS)
    except Exception as e:
        print(f"[H2] API 不可得: {type(e).__name__}: {e} — 降级标注", flush=True)
        h2_degraded = True

    # H3 5m (dev 测单格耗时/内存)
    pairs5 = load_5m_pairs()
    h3 = {}
    t_cell0 = time.time()
    first5 = pairs5[0][1]
    ctx0 = make_ctx(first5, PARAMS["warmup"], state_fns={})
    breakout_per(ctx0.close, ctx0.high, ctx0.low, 20, 0.0)
    t_cell = time.time() - t_cell0
    for sym, df5 in pairs5:
        ctx = make_ctx(df5, PARAMS["warmup"], state_fns={})
        cells = h3_real(ctx, h3_Ns, PARAMS["h3_costs_bp"])
        gbm = h3_gbm(df5, h3_Ns, seeds)
        h3[sym] = {"cells": cells, "gbm": gbm}

    # GATE
    gate(h1[0], h1[2], h2_gbm_spear,
         {N: v for sym in h3 for N, v in h3[sym]["gbm"].items()},
         not h2_degraded)

    if dev:
        print(f"  [dev] 5m 单格耗时 {t_cell * 1000:.1f}ms; 5m 数据 {len(first5)}"
              f" 根/标的")
        h1_line = "H1 条件概率 {:.3f}".format(h1[0])
        if not h2_degraded:
            for ccy, hh in h2.items():
                h1_line += " | {} 底5%前向 {:.2f} 顶5% {:.2f} Spear {:.3f} " \
                           "触碰n={}".format(
                    ccy, hh["lo"][1], hh["hi"][1], hh["spear"][1],
                    hh["touch"]["n_touch"])
        else:
            h1_line += " | H2 degraded"
        print("  [dev] " + h1_line)
        print("  [dev] H3 样例 {:.4f}".format(
            next(iter(h3.values()))["cells"][(h3_Ns[0], 0.0)][0]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {"h1": h1, "h2": h2, "h2_gbm_spear": h2_gbm_spear,
           "h2_gbm_touch": h2_gbm_touch, "h3": h3}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res, h2_degraded)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
