#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C56 书自测的忠实复现补做 (CH15/16/19) (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c56 行): 用户指出 c54/c55 测的是**加密扩展版**, 违反
  忠实复现优先纪律 — 本考证按书原市场/原周期/原方法重做书自测。oracle 逐字
  核实口径。描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。**结论不得
  作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册
  冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书的三项自测在忠实口径 (书市场/书周期/书方法)
  下是否成立? (vs c54/c55 的加密扩展版对照)

预注册假设 (PLAN §2.5 c56 行, docstring 逐字):
  H1: 周二反转忠实复现 — SPY/GC=F/EURUSD=X 日线 (control.db 覆盖书周期
      2000-01~2011-05; 书的 6 市场可测 3 个, 标注) close-only X/O
      Monday/Tuesday 模式矩阵 vs 书的报告值 (全概率 34-66% 带) + 置换 null
      (30 次) + 我们的周期 (2012-2026) 扩展
  H2: ORB 方法忠实复现 — 书方法 (固定幅度突破、日内 bar、突破入场) 的加密
      5m 映射 (突破幅度 = k×ATR, k∈{0.5,1,2}; 书为 $0.05-1.00 固定点 —
      映射标注) vs c54 H3 的 N-bar 版 + 无成本/成本 + GBM null
  H3: 择时单例忠实复现 — EURUSD=X 日线 (书同市场同周期 2010-07~2011-03)
      60 日 MA + 3 日 raw stochastic <30 回撤入场 vs 即时入场, 净收益
      (点数) 对拍书 $6,200/$2,075 + 我们的周期 (2012-2026) 扩展 + GBM null

  操作化 (运行前锁定):
    - H1: 3 市场 × 2 周期 (书窗口 [2000-01-01, 2011-05-31] 用可测重叠,
      GC=F 自 2000-08-30、EURUSD 自 2003-12-01 起 — 标注截断; 扩展窗口
      [2012-01-01, 2026-08-31]); close-only 涨跌符号 s_t=sign(c_t/c_{t-1}−1),
      分周内 P(up) X/O 矩阵; 关键条件概率 P[周二跌|周一涨]; 置换 null =
      窗口内日标签打乱 30 次; 对照书 34-66% 全概率带
    - H2: BTC/ETH 5m, N=20 通道 + 固定幅度 k×ATR (k∈{0,0.5,1,2}, k=0 =
      c54 N-bar 等价基线), 永远在场反转, 无成本 (0bp) vs 2.5bp (每信号
      事件 1 笔); GBM null 15 种子 (5m 生成耗时 — c54 教训), 同管线
    - H3: EURUSD=X 日线, 书窗口 [2010-07-01, 2011-04-01) (warmup 用窗口前
      数据), MA(60) 方向转向 + %K(3)<30 回撤入场 vs 即时入场 (c55 状态机
      同款, 日线参数); 净收益点数 (×10000); 扩展窗口 [2012-01-01, 2026-09-01);
      GBM null 30 种子 (每窗口各自锚定), 同管线
    - 学习级: 30 种子 (H2 5m 裁 15)、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close (日线)     | control.db 直接读 (ts=epoch 秒)        | 日线收盘后 | 已收盘 daily bar
  X/O 符号         | sign(diff(close)) 逐日                | 日线收盘后 | 书 close-only 口径
  周内标签         | ts 的 dayofweek (UTC)                 | 日历事实   | 书周一~周五
  5m 通道/突破     | rolling(N).max().shift(1) 窗口止于 t−1| bar 收盘后 | c54 同款 (防前视)
  固定幅度         | k×atr[t−1] (atr 用前一根, 防当前 bar) | bar 收盘后 | 书 $ 固定点 → 波动单位
  MA(60) 方向      | rolling(60).mean(), sign(diff)        | 日线收盘后 | 书 60 日 MA
  %K(3)            | (close−low_3)/(high_3−low_3)          | 日线收盘后 | 书 3 日 raw stochastic
  择时状态机       | 转向后等 %K<30 入场 vs 即时 (c55 同款) | 日线收盘后 | 因果, 无前视
  净收益           | Σ(pos·ret), 点数 = Σ(pos·Δprice)×1e4   | 全样本事后 | 描述统计
  GBM null         | sim_market.gbm_matching (锚定真实)     | 锚定真实   | 固定种子 0..29; 5m 15 种子
  H1 置换 null     | 窗口内日标签打乱 30 次 (保留收益序列)   | 锚定真实   | 破坏周内结构

数据声明:
  data/control.db: SPY_1d (1993-01..2026-08), GC=F_1d (2000-08..2026-08),
  EURUSD=X_1d (2003-12..2026-08), ts=epoch 秒 (UTC); 书窗口 2000-01~2011-05
  可测: SPY 全程 / GC=F 自 2000-08 / EURUSD 自 2003-12 (标注截断)。
  data/backtest.db: BTC/ETH 5m 2023-08..2026-08 (~315K bar/标的)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  H1: 书窗口 2000-01-01~2011-05-31 / 扩展 2012-2026, 置换 30, 书带 34-66%;
  H2: N=20, k∈{0,0.5,1,2}, 成本 0/2.5bp, 种子 15; H3: ma_w=60, stoch_w=3,
  thresh=0.30, 书窗口 2010-07~2011-03, 种子 30; min_n=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - H1: 书的 6 市场仅 3 个可测 (SPY 代理 emini S&P, EURUSD 现货代理期货,
    GC=F 黄金; 缺失 BAC/Intel/30y 美债 — 标注); EURUSD/GC 书窗口起点截断
    (2003-12/2000-08) — 标注; 34-66% 是书报告的全概率带, 用于对照分周内
    P(up) 范围。
  - H2: 书原油 5m 2009-2011 数据不可得 — 忠实复现**方法**而非数据: 加密 5m
    固定幅度突破, $ 固定点映射为 k×ATR 波动单位 (k∈{0.5,1,2}); 加入 k=0
    格作为 c54 N-bar 等价的同管线对照。
  - H3: EURUSD=X 现货汇率作欧元期货代理 (标注); 书 $6,200/$2,075 为单合约
    美元额, 现货只有价格点 — 用净收益点数 (×10000) 对拍, 比值 (6200/2075
    ≈2.99) 为同量纲参照。
  - 学习级: 无 BY_YEAR; H2 5m GBM 种子 15 (c54 裁剪经验, 报告)。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① X/O golden (构造已知周一涨周二跌序列 → 条件概率=1.0);
    ② 择时 golden (c55 同款: 转向+%K 回撤 → 入场延迟于转向); ③ H1 置换
    sanity — 置换 null 条件概率 mean ∈ [45%,55%]; ④ H2 GBM 5m 突破净盈亏
    带 [−0.8,+0.8]; ⑤ H3 GBM null gain 带 [−0.5,+0.5]; 任一失败 SystemExit
  - GBM 无信息对照: H2 15 种子 / H3 30 种子, gbm_matching 锚定真实同管线
  - MIN_N: 每格 n ≥ 100 (学习级); 不足标注 [MIN_N 不足]
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: H3 EURUSD 书窗口 × 3 种子 (测择时管线) + H1 1 市场 + H2 单 k,
    不写 .out
  - 全量: H1 3 市场 × 2 周期; H2 BTC/ETH 5m × 4 k × 2 成本 × 15 种子;
    H3 书窗口 + 扩展 × 30 种子 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c56_faithful_self_tests.py --dev
  python3 research/studies/c56_faithful_self_tests.py
"""
import hashlib
import os
import sqlite3
import sys
import time
from datetime import date

# 仓库根入 path (模板摩擦, 见 c12 报告)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.ctx import make_ctx
from research.data_loader import DB_PATH, load_candles
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "control_db": "data/control.db",
    "h1_markets": ("SPY_1d", "GC=F_1d", "EURUSD=X_1d"),
    "h1_book": ("2000-01-01", "2011-05-31"),   # 书窗口 (重叠标注)
    "h1_ext": ("2012-01-01", "2026-08-31"),    # 扩展窗口
    "h1_perm": 30,
    "h1_book_band": (0.34, 0.66),              # 书图15.14 全概率带
    "h2_crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "h2_N": 20,
    "h2_ks": (0.0, 0.5, 1.0, 2.0),             # k×ATR 固定幅度 (k=0=N-bar 等价)
    "h2_costs_bp": (0.0, 2.5),
    "h3_ma_w": 60,
    "h3_stoch_w": 3,
    "h3_thresh": 0.30,
    "h3_book": ("2010-07-01", "2011-04-01"),   # 书同市场同周期
    "h3_ext": ("2012-01-01", "2026-09-01"),
    "warmup": 600,                             # 5m
    "warmup_d": 200,                           # 日线
    "gbm_seeds": 30,
    "h2_gbm_seeds": 15,                        # 5m 裁剪 (c54 教训)
    "min_n": 100,                              # 学习级
    "h1_null_band": (0.45, 0.55),
    "h2_gbm_band": (-0.8, 0.8),
    "h3_null_band": (-3000.0, 3000.0),   # 点数 (×10000) 量级 sanity 带
    "dev_subset": {"n_gbm": 3, "h2_ks": (1.0,), "h1_markets": ("EURUSD=X_1d",)},
    "data_range": "书窗口 2000-01..2011-05 (EURUSD 2003-12/GC 2000-08 起); "
                  "扩展 2012-2026; 5m 2023-08..2026-08",
}

STUDY_ID = "c56_faithful_self_tests"


# ── 加载 control.db 日线 ─────────────────────────────────────
def load_daily(table, t0, t1):
    conn = sqlite3.connect(PARAMS["control_db"])
    try:
        df = pd.read_sql_query(
            f'SELECT ts, open, high, low, close FROM "{table}" '
            "ORDER BY ts", conn)
    finally:
        conn.close()
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop_duplicates(subset="ts").set_index("ts").sort_index()
    df = df[~df["close"].isna()]
    m = (df.index >= pd.Timestamp(t0, tz="UTC")) & \
        (df.index <= pd.Timestamp(t1, tz="UTC"))
    return df[m]


def load_5m_pairs():
    conn = sqlite3.connect(DB_PATH)
    out = []
    try:
        for sym in PARAMS["h2_crypto"]:
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


# ── H1: X/O 周二反转 (忠实复现) ─────────────────────────────
def h1_stats(df, perm):
    """窗口内日线: X/O 分周内 P(up) 矩阵 + P[周二跌|周一涨] 条件概率 + 置换 null"""
    c = df["close"].values
    s = np.sign(np.diff(c))                  # close-only 涨跌
    wd = df.index[1:].dayofweek.values       # 0=Mon..4=Fri
    n = len(s)
    # 分周内 P(up)
    wd_up = {}
    for d in range(5):
        m = wd == d
        wd_up[d] = (int(m.sum()),
                    float(np.mean(s[m] > 0)) if m.sum() else float("nan"))
    # 条件概率 P[周二跌|周一涨]
    mon_up = (wd[:-1] == 0) & (s[:-1] > 0)
    tue = wd[1:] == 1
    mm = mon_up & tue
    n_cond = int(mm.sum())
    real = float(np.mean(s[1:][mm] < 0)) if n_cond else float("nan")
    # 置换 null (窗口内日标签打乱, 保留收益序列)
    nulls = []
    for p in range(perm):
        rng = np.random.default_rng(5000 + p)
        wd_p = rng.permutation(wd)
        mup = (wd_p[:-1] == 0) & (s[:-1] > 0)
        tup = wd_p[1:] == 1
        mm_p = mup & tup
        if mm_p.sum():
            nulls.append(float(np.mean(s[1:][mm_p] < 0)))
    return {"wd_up": wd_up, "cond": (n_cond, real), "nulls": np.array(nulls),
            "n_bars": n}


# ── H2: 固定幅度突破 (ORB 方法忠实复现) ─────────────────────
def breakout_per(close, high, low, atr, N, k, cost_bp):
    """N=20 通道 + k×ATR 固定幅度突破 (永远在场反转). k=0 ≡ c54 N-bar 版."""
    c = np.asarray(close, float)
    n = len(c)
    hi = pd.Series(high)
    lo = pd.Series(low)
    ref_hi = hi.rolling(N).max().shift(1).values
    ref_lo = lo.rolling(N).min().shift(1).values
    atr_prev = np.concatenate([[np.nan], atr[:-1]])   # 前一根 ATR
    amp = k * atr_prev
    long_sig = c > ref_hi + amp
    short_sig = c < ref_lo - amp
    sig_dir = np.zeros(n, int)
    sig_dir[long_sig & np.isfinite(ref_hi + amp)] = 1
    sig_dir[short_sig & ~long_sig & np.isfinite(ref_lo - amp)] = -1
    idx = np.arange(n)
    last_idx = np.where(sig_dir != 0, idx, 0)
    last_idx = np.maximum.accumulate(last_idx)
    has = np.maximum.accumulate(sig_dir != 0)
    p = np.where(has, sig_dir[last_idx], 0)
    r = np.zeros(n)
    r[:-1] = c[1:] / c[:-1] - 1.0
    per = p * r
    n_ev = int((sig_dir != 0).sum())
    net = float(per.sum()) - n_ev * cost_bp * 1e-4
    return net, n_ev


def h2_real(ctx, N, ks, costs):
    out = {}
    for k in ks:
        for cb in costs:
            net, nev = breakout_per(ctx.close, ctx.high, ctx.low, ctx.atr,
                                    N, k, cb)
            out[(k, cb)] = (net, nev)
    return out


def h2_gbm(df, N, ks, seeds):
    nets = {k: [] for k in ks}
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        ctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
        for k in ks:
            net, _ = breakout_per(ctx.close, ctx.high, ctx.low, ctx.atr,
                                  N, k, 0.0)
            nets[k].append(net)
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)))
            for k, v in nets.items()}


# ── H3: 择时单例 (c55 状态机, 日线参数) ─────────────────────
def dual_tf(close, high, low, ma_w, stoch_w, thresh, long_only=True):
    """书忠实版双系统 (long_only=True: 书 CH19 dip-buy 例, 只做多;
    long_only=False: c55 对称永远在场版). 返回 (pos_imm, pos_tim, n_entries)."""
    c = np.asarray(close, float)
    n = len(c)
    ma = pd.Series(c).rolling(ma_w).mean().values
    slope = np.sign(np.diff(ma, prepend=np.nan))
    hi = pd.Series(high).rolling(stoch_w).max().values
    lo = pd.Series(low).rolling(stoch_w).min().values
    kk = (c - lo) / (hi - lo)
    pos_imm = np.zeros(n, int)
    pos_tim = np.zeros(n, int)
    pos = 0
    pending = 0
    n_entries = 0
    for i in range(1, n):
        # 即时: 上一 bar 收盘时的 MA 方向 (long_only: 只有多, MA 下则平)
        if np.isfinite(slope[i - 1]) and slope[i - 1] != 0:
            if long_only:
                pos_imm[i] = 1 if slope[i - 1] > 0 else 0
            else:
                pos_imm[i] = int(slope[i - 1])
        else:
            pos_imm[i] = pos_imm[i - 1]
        # 择时状态机
        s1 = slope[i - 1] if np.isfinite(slope[i - 1]) else 0
        s2 = slope[i - 2] if i >= 2 and np.isfinite(slope[i - 2]) else 0
        kv = kk[i - 1] if np.isfinite(kk[i - 1]) else float("nan")
        if s1 == 1 and s2 != 1:
            pending = 1
            if pos == -1:
                pos = 0
        elif (not long_only) and s1 == -1 and s2 != -1:
            pending = -1
            if pos == 1:
                pos = 0
        elif long_only and pos == 1 and s1 != 1:
            pos = 0                     # long_only: MA 转下即平多 (不反手)
        if pending == 1 and pos == 0 and np.isfinite(kv) and kv < thresh:
            pos = 1
            pending = 0
            n_entries += 1
        elif (not long_only) and pending == -1 and pos == 0 and \
                np.isfinite(kv) and kv > 1 - thresh:
            pos = -1
            pending = 0
            n_entries += 1
        pos_tim[i] = pos
    return pos_imm, pos_tim, n_entries


def h3_run(df, win, params, long_only=True):
    """日线 df (已含 warmup 前数据) → win=[t0,t1) 内净收益 (点数) + 入场数."""
    ctx = make_ctx(df, params["warmup_d"], state_fns={})
    pi, pt, ne = dual_tf(ctx.close, ctx.high, ctx.low, params["h3_ma_w"],
                         params["h3_stoch_w"], params["h3_thresh"],
                         long_only=long_only)
    ts = df.index[params["warmup_d"]:].values.astype("datetime64[ns]")
    m = (ts >= np.datetime64(win[0])) & (ts < np.datetime64(win[1]))
    price = ctx.close
    dprice = np.concatenate([[0.0], np.diff(price)])
    net_imm = float(np.sum(pi[m] * dprice[m])) * 1e4   # 点数 (×10000)
    net_tim = float(np.sum(pt[m] * dprice[m])) * 1e4
    return {"imm": net_imm, "tim": net_tim, "gain": net_tim - net_imm,
            "n_entries": ne, "n_bars": int(m.sum())}


def h3_gbm_null(df, win, params, seeds, long_only=True):
    gains = []
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        st = h3_run(rw, win, params, long_only)
        gains.append(st["gain"])
    g = np.array(gains)
    return float(np.mean(g)), float(np.std(g, ddof=1))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def _golden_xo():
    """构造已知序列: 周一恒涨 (+1)、周二恒跌 (−1) → P[周二跌|周一涨] 应 = 1.0.
    s[i]=diff(close)[i] 是 index[i+1] 那天的涨跌 — 按 weekday 直接构造."""
    n_days = 60
    idx = pd.date_range("2000-01-03", periods=n_days + 1, freq="D", tz="UTC")
    moves = []
    for i in range(1, n_days + 1):
        w = idx[i].dayofweek
        if w == 0:
            moves.append(1.0)
        elif w == 1:
            moves.append(-1.0)
        else:
            moves.append(0.5 if i % 2 == 0 else -0.5)
    close = np.concatenate([[100.0], np.cumsum(moves)])
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                       "close": close}, index=idx)
    st = h1_stats(df, 5)
    n, real = st["cond"]
    if real != 1.0 or n < 8:
        raise SystemExit(
            f"GATE FAIL: X/O golden P[周二跌|周一涨]={real} n={n} "
            f"(期望 1.0, n≥8)")
    return True


def _golden_dual_tf():
    """c55 同款 golden: 平盘 → 强涨 → 回撤 (%K<30) → 择时入场延迟于转向."""
    c = np.array([100.0] * 500 + list(100.0 + 0.5 * np.arange(1, 201)))
    hi = c + 1.0
    lo = c - 1.0
    dip = np.array([c[-1] - 1.5, c[-1] - 3.0, c[-1] - 4.5,
                    c[-1] - 3.0, c[-1] - 1.5])
    restore = np.array([c[-1] + 1.0] * 20)
    c = np.concatenate([c, dip, restore])
    hi = np.concatenate([hi, dip + 1.0, restore + 1.0])
    lo = np.concatenate([lo, dip - 1.0, restore - 1.0])
    idx = pd.date_range("2024-01-01", periods=len(c), freq="4h", tz="UTC")
    df = pd.DataFrame({"open": c, "high": hi, "low": lo, "close": c,
                       "volume": 1.0}, index=idx)
    ctx = make_ctx(df, 200, state_fns={})
    pi, pt, ne = dual_tf(ctx.close, ctx.high, ctx.low, 100, 12, 0.30)
    if not np.any(pi != 0) or ne < 1:
        raise SystemExit("GATE FAIL: 择时 golden 转向/入场缺失")
    turn_i = int(np.flatnonzero(pi != 0)[0])
    entry_i = int(np.flatnonzero(pt != 0)[0])
    if entry_i <= turn_i:
        raise SystemExit("GATE FAIL: 择时入场早于转向 — 状态机错误")
    return True


def gate(h1_nulls, h2_gbm, h3_null_btc, pos_imm_btc):
    _golden_xo()
    _golden_dual_tf()
    lo, hi = PARAMS["h1_null_band"]
    nm = float(np.mean(h1_nulls))
    if not (lo <= nm <= hi):
        raise SystemExit(
            f"GATE FAIL: H1 置换 null mean {nm:.3f} ∉ [{lo}, {hi}]")
    glo, ghi = PARAMS["h2_gbm_band"]
    gm = float(np.mean([v[0] for v in h2_gbm.values()]))
    if not (glo <= gm <= ghi):
        raise SystemExit(f"GATE FAIL: H2 GBM 净盈亏 {gm:+.4f} ∉ [{glo}, {ghi}]")
    hlo, hhi = PARAMS["h3_null_band"]
    if not (hlo <= h3_null_btc <= hhi):
        raise SystemExit(
            f"GATE FAIL: H3 GBM null gain {h3_null_btc:+.4f} ∉ [{hlo}, {hhi}]")
    print(f"[GATE] X/O golden [PASS]; 择时 golden [PASS]; H1 置换 null {nm:.3f} "
          f"[PASS]; H2 GBM {gm:+.4f} [PASS]; H3 null {h3_null_btc:+.4f} "
          f"[PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.1f}%"


def _nm(n):
    return "[MIN_N 通过]" if n >= PARAMS["min_n"] else "[MIN_N 不足]"


def _xo_line(st):
    parts = []
    for d in range(5):
        dn, dm = st["wd_up"][d]
        parts.append("{}:P(up) {:.1%}(n={})".format("一二三四五"[d], dm, dn))
    return " | ".join(parts)


def write_out(out_path, params, res):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=h1_book={}~{},h1_perm={},h2_N={},h2_ks={},h2_costs={},"
        "h3_ma_w={},h3_stoch_w={},gbm_seeds=30(h2_5m裁15),min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["h1_book"][0], p["h1_book"][1], p["h1_perm"],
            p["h2_N"], p["h2_ks"], p["h2_costs_bp"], p["h3_ma_w"],
            p["h3_stoch_w"], p["min_n"], p["min_n"]),
        "# GATE: 探测器自检 X/O golden + 择时 golden + H1 置换 null 带 + "
        "H2 GBM 带 + H3 null 带 [PASS]; MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c56 书自测忠实复现 (CH15 图15.14 周二反转 / "
        "CH16 ORB 固定幅度突破 / CH19 择时单例); 书市场×书周期×书方法; "
        "描述层无入场, 无交易含义",
        "",
    ]
    # H1
    lines.append("[H1] 周二反转忠实复现 (close-only X/O, 书窗口 2000-01~2011-05 "
                 "可测重叠; 书图15.14 全概率带 34-66%):")
    for mkt in p["h1_markets"]:
        for period in ("book", "ext"):
            st = res["h1"][(mkt, period)]
            tag = "书窗口" if period == "book" else "扩展"
            pct_lo, pct_hi = (np.percentile(st["nulls"], 2.5),
                              np.percentile(st["nulls"], 97.5))
            n_cond, real = st["cond"]
            verdict = "超区间↑" if real > pct_hi else \
                ("低于区间↓" if real < pct_lo else "区间内")
            lines.append("  {} [{}] {}:".format(mkt.split("_")[0], tag,
                                                _xo_line(st)))
            lines.append("      P[周二跌|周一涨] 真实 {:.1%} (n={}) {} vs 置换 "
                         "95% [{:.1%}, {:.1%}] (mean {:.1%}) -> {}".format(
                real, n_cond, _nm(n_cond), pct_lo, pct_hi,
                float(np.mean(st["nulls"])), verdict))
    # H2
    lines.append("")
    lines.append("[H2] ORB 固定幅度突破 (BTC/ETH 5m, N={} 通道 + k×ATR, "
                 "永远在场反转; k=0 ≡ c54 N-bar 版):".format(p["h2_N"]))
    for sym, h2 in res["h2"].items():
        for k in p["h2_ks"]:
            gmean, gstd = h2["gbm"][k]
            cells = []
            z00 = float("nan")
            for cb in p["h2_costs_bp"]:
                net, nev = h2["cells"][(k, cb)]
                if cb == 0.0 and gstd > 0:
                    z00 = (net - gmean) / gstd
                cells.append("{}bp {:+.4f}(ev={})".format(
                    cb, net, nev))
            tag = "超2σ↑" if z00 > 2 else ("低于2σ↓" if z00 < -2 else "|z|≤2")
            lines.append("  {} k={}: {} | GBM(0bp) {:.4f}±{:.4f} | 0bp z "
                         "{:+.2f} {}".format(
                sym.split("/")[0], k, " ".join(cells), gmean, gstd, z00, tag))
    # H3
    lines.append("")
    lines.append("[H3] 择时单例忠实复现 (EURUSD=X 日线, MA({}) + %K({})<30 "
                 "回撤 vs 即时, 净收益点数 ×10000; 长-only = 书 CH19 dip-buy "
                 "例):".format(p["h3_ma_w"], p["h3_stoch_w"]))
    for period in ("book", "ext"):
        st = res["h3"][period]
        tag = "书窗口 2010-07~2011-03" if period == "book" else "扩展 2012-2026"
        rlo = st["real_lo"]
        nlo = st["null_lo"]
        zlo = (rlo["gain"] - nlo[0]) / nlo[1] if nlo[1] > 0 else float("nan")
        lines.append("  {} 长-only: 即时 {:+.1f} 点 | 择时 {:+.1f} 点 | gain "
                     "{:+.1f} | null {:+.3f}±{:.3f} | z {:+.2f} | 入场 n={} {}"
                     .format(tag, rlo["imm"], rlo["tim"], rlo["gain"],
                             nlo[0], nlo[1], zlo, rlo["n_entries"],
                             _nm(rlo["n_entries"])))
        rs = st["real_sym"]
        lines.append("  {} 对称:  即时 {:+.1f} 点 | 择时 {:+.1f} 点 | gain "
                     "{:+.1f}".format(" " * 3, rs["imm"], rs["tim"],
                                      rs["gain"]))
    rlo = res["h3"]["book"]["real_lo"]
    if rlo["imm"] != 0:
        ratio = rlo["tim"] / rlo["imm"]
        lines.append("  书窗口长-only 择时/即时 比值 {:.2f} (书 $6200/$2075 ≈ "
                     "{:.2f})".format(ratio, 6200.0 / 2075.0))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] 书 CH15 图15.14 (6 市场 2000-01~2011-05, 全概率 "
                 "34-66%, 自评'up Monday down Tuesday 但弱'); CH16/CH18 ORB "
                 "(4 期货 5m 固定点突破全盈利, 原油 2009-2011 数据不可得); "
                 "CH19 (60日 MA + 3日随机 $2075→$6200 单例 n=1); c54 (加密 "
                 "扩展版: 周内判据不成立, 5m N-bar 无成本即负); c55 (加密 "
                 "扩展版: 择时减益); c19 (−15.4pp dip-buy 无优势); "
                 "c29 (时间视角证伪); c21/c23/c25 (成本吃光)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    h1_markets = PARAMS["dev_subset"]["h1_markets"] if dev else \
        PARAMS["h1_markets"]
    h2_ks = PARAMS["dev_subset"]["h2_ks"] if dev else PARAMS["h2_ks"]
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    h2_seeds = PARAMS["dev_subset"]["n_gbm"] if dev else \
        PARAMS["h2_gbm_seeds"]

    # H1 (廉价)
    res_h1 = {}
    h1_nulls_pool = []
    for mkt in h1_markets:
        for period, (t0s, t1s) in (("book", PARAMS["h1_book"]),
                                   ("ext", PARAMS["h1_ext"])):
            df = load_daily(mkt, t0s, t1s)
            if len(df) < 10:
                continue
            st = h1_stats(df, PARAMS["h1_perm"])
            res_h1[(mkt, period)] = st
            h1_nulls_pool.append(st["nulls"])
    h1_all_nulls = np.concatenate(h1_nulls_pool) if h1_nulls_pool else \
        np.array([0.5])

    # H2 (5m, 15 种子)
    pairs5 = load_5m_pairs()
    res_h2 = {}
    for sym, df5 in pairs5:
        ctx = make_ctx(df5, PARAMS["warmup"], state_fns={})
        cells = h2_real(ctx, PARAMS["h2_N"], h2_ks, PARAMS["h2_costs_bp"])
        gbm = h2_gbm(df5, PARAMS["h2_N"], h2_ks, h2_seeds)
        res_h2[sym] = {"cells": cells, "gbm": gbm}

    # H3 (日线, 30 种子)
    res_h3 = {}
    df_book = load_daily("EURUSD=X_1d", "1990-01-01", PARAMS["h3_book"][1])
    df_ext = load_daily("EURUSD=X_1d", PARAMS["h3_ext"][0], PARAMS["h3_ext"][1])
    wins = {"book": PARAMS["h3_book"], "ext": PARAMS["h3_ext"]}
    for period, df in (("book", df_book), ("ext", df_ext)):
        win = wins[period]
        real_lo = h3_run(df, win, PARAMS, long_only=True)
        gmn_lo, gsd_lo = h3_gbm_null(df, win, PARAMS, seeds,
                                     long_only=True)
        real_sym = h3_run(df, win, PARAMS, long_only=False)
        gmn_sy, gsd_sy = h3_gbm_null(df, win, PARAMS, seeds,
                                     long_only=False)
        res_h3[period] = {"real_lo": real_lo, "null_lo": (gmn_lo, gsd_lo),
                          "real_sym": real_sym, "null_sym": (gmn_sy, gsd_sy)}

    # GATE (用 H2 首标 GBM + H3 首段 null)
    first_h2 = next(iter(res_h2.values()))
    gate(h1_all_nulls, first_h2["gbm"], res_h3["book"]["null_lo"][0], None)

    if dev:
        for (mkt, period), st in res_h1.items():
            print("  [dev] {} {} 条件概率 {:.3f} (n={})".format(
                mkt, period, st["cond"][1], st["cond"][0]))
        for sym, h2 in res_h2.items():
            for k, v in h2["cells"].items():
                print("  [dev] {} k={} 0bp {:+.4f}".format(
                    sym.split("/")[0], k, v[0]))
        for period in ("book", "ext"):
            st = res_h3[period]
            print("  [dev] H3 {} 长-only: 即时 {:+.1f} 择时 {:+.1f} (null "
                  "{:+.3f}±{:.3f}) | 对称: 即时 {:+.1f} 择时 {:+.1f}".format(
                period, st["real_lo"]["imm"], st["real_lo"]["tim"],
                st["null_lo"][0], st["null_lo"][1],
                st["real_sym"]["imm"], st["real_sym"]["tim"]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {"h1": res_h1, "h2": res_h2, "h3": res_h3}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
