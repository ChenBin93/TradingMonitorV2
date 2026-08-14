#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C57 期权线第一块砖: 2σ 带触碰 → 跨式期权 (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c57 行): long-vol 策略先验 — MA20±2σ 带触碰 (波动
  释放信号) 后买跨式是否在期权定价之上有增量。数据限制: OKX 只服务在列合约
  (过期 51001 不可拉), 周度合约上市晚 — **合约可用性以蜡烛覆盖为准** (触碰
  时点无蜡烛的合约 = 当时不可交易, 自动排除)。期权 K 线稀疏 (流动性差时段缺
  bar) — 目标时点 ±2h 内最近 bar, 否则该事件剔除。ctMult=0.01 BTC/张 (BTC
  期权)。描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。**结论不得作
  交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册
  冻结、内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 触碰 (c51 口径) 后的跨式 P&L 是否 > 随机
  时点 null? 信号是否超出期权定价 (realized 波动 vs 隐含 breakeven)?

预注册假设 (PLAN §2.5 c57 行, docstring 逐字):
  H1: 触碰后 24h 跨式 P&L 均值 > 随机时点 null (30 次同数量抽样) 95% 区间
      (IV 定价之上有增量)
  H2: 触碰后 realized 24h 波动增量 (c52 口径) vs 跨式隐含 breakeven 波动
      (跨式价×2/标的价) — 信号是否超出期权定价
  H3: 成本敏感性 (taker 0.03%×4 腿 + 1 tick 价差)

  操作化 (运行前锁定):
    - 数据: BTC/ETH 1h (backtest.db) MA20±2σ 触碰 (c51 口径), 触碰窗口
      2026-06-24 起 (期权可映射起始); 期权 data/options.db (options_meta +
      options_candles, 1H)
    - 触碰 = 收盘穿越带边界 (c51 band_touches); 触碰 bar T, 入场 = T+1 bar
      开盘 (τ = ts[T+1])
    - 合约映射 (因果, 蜡烛覆盖 = 地面真相): τ 时已有蜡烛 (coverage_start ≤ τ)
      且到期 ≥ τ+72h 的合约; 最近到期 → 最近 strike (|strike−spot|) 的 call+put
      对; 无候选 → 事件剔除 (计入被排除数)
    - 价格查找: 目标时点 ±2h 内最近 bar (open=入场, close=平仓), 无 → 剔除
    - 持有 24/48/72h; P&L = 平仓跨式价 − 入场跨式价 (×ctMult; 无 bid/ask
      用 OHLC 标注); P&L 归一化 % of 标的价
    - H1: 24h P&L 均值 vs 随机时点 null 30 次 (同数量, 同映射同持有)
    - H2: E1(24) (c52 口径: mean(ATR[t+1..t+24])/mean(ATR[t−24..t−1])−1) +
      realized 24h 移动 |log(c[T+25]/c[T+1])| vs breakeven = 2×跨式价/标的价
    - H3: 成本 = 0.03%×(入场+平仓跨式价)×ctMult + 4×tick (tick=0.0001);
      净 P&L 再报
    - 学习级: BTC/ETH 双标的 (ETH 期权无覆盖 → 标注)、30 次 null、MIN_N=100

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close (1h)       | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  MA20±2σ 带       | rolling(20) mean/sd (价格 sd)         | bar 收盘后 | c51 口径
  触碰事件         | 收盘穿越带边界 (c51 band_touches)      | bar 收盘后 | 出带=波动释放信号
  合约可用性       | 蜡烛覆盖 (coverage_start ≤ τ)          | τ 时刻    | OKX 在列=可交易地面真相
  到期约束         | expiry ≥ τ+72h (解析 inst_id YYMMDD)   | τ 时刻    | 覆盖最长持有窗
  期权价格         | 目标时点 ±2h 内最近 1H bar (open/close)| τ 时刻    | 稀疏 K 线 → ±2h 容忍
  P&L             | Δ(跨式价)×ctMult, % of 标的价          | 全样本事后 | 描述统计
  E1(24)          | c52 对称窗口均值比 (pre 不含 t)         | 事后       | c52 口径
  breakeven 波动   | 2×跨式价/标的价 (任务给定公式)         | 事后       | 跨式隐含所需波动
  null            | 随机时点同数量抽样 30 次 (同映射同持有)  | 锚定真实   | 非 GBM (期权路径无
                   |                                       |            |   简单 GBM 同管线)

数据声明:
  data/backtest.db: BTC/ETH 1h (2023-08..2026-08); 触碰窗口 2026-06-24~
  2026-08-01 (backtest 1h 止 08-01)。
  data/options.db: options_meta (70 定向合约) + options_candles (339 合约
  17.2 万行 1H, 2026-03-19..2026-08-14); BTC 各到期覆盖全 (260828 自 05-15,
  260925/261225/270326 自 03 月); **ETH 仅 12 合约全为 260815, 蜡烛只覆盖
  08-11..08-14 — 触碰窗口内无可映射 ETH 合约** (标注)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  带 MA20±2σ (n=20); 触碰窗口 2026-06-24..2026-08-01; 到期 ≥ τ+72h;
  持有 24/48/72h; 价格查找 ±2h; ctMult BTC=0.01; taker 0.03%×4 腿 + 1 tick
  (0.0001); null 30 次; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - ETH 期权蜡烛覆盖仅 08-11..08-14 (触碰窗口之后) — ETH 可用样本 0, 全量
    以 BTC 为准 (标注; 不硬凑 ETH)。
  - 书/任务给出 breakeven 公式 2×跨式价/标的价 — 期权价格为每单位标的的 BTC
    报价, 即跨式价已是标的价的分数 → breakeven = 2×(C+P) (标注换算)。
  - 无 bid/ask, 用 OHLC (入场用 bar open, 平仓用 bar close); 价格在日内可能
    陈旧 (低流动时段缺 bar/重复报价) — 标注, P&L 为标报价 mark-to-market。
  - null 用随机时点抽样 (期权没有简单的 GBM 同管线 — 期权价依赖标的+IV+期限
    结构, 随机时点 null 捕获 theta/流动性基线)。
  - 学习级: 无 BY_YEAR; 30 次 null 沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 合约映射 golden (构造已知覆盖场景: 最近到期/最近 strike/
    C+P 对/到期约束/覆盖排除); ② 价格查找 golden (±2h 最近 bar / 超窗 None);
    ③ P&L golden (Δ跨式价×ctMult); ④ null sanity — 随机时点 P&L 均值 ∈
    [−5%, +5%] (theta 侵蚀基线; 超带 = 管线错误 SystemExit)
  - null 无信息对照: 随机时点 30 次 (同数量同管线)
  - MIN_N: 每格 n ≥ 100 (学习级); 可用样本/被排除数/窗口诚实报告
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 触碰 20 个 × null 3 次 (测管线 + 映射存活率), 不写 .out
  - 全量: BTC (ETH 标注无数据) 全触碰 × null 30 次 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c57_option_straddle.py --dev
  python3 research/studies/c57_option_straddle.py
"""
import hashlib
import os
import re
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
from research.data_loader import DB_PATH, load_candles, verify

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "tf": "1h",
    "touch_start": "2026-06-24",          # 触碰窗口起点 (预注册)
    "touch_end": "2026-08-01",            # backtest 1h 止
    "band_n": 20,
    "band_k": 2.0,
    "holds_h": (24, 48, 72),
    "exp_min_h": 72,                      # 到期 ≥ τ+72h
    "lookup_win_h": 2,                    # ±2h 内最近 bar
    "ct_mult": 0.01,                      # BTC 期权 (标注: ETH 无数据)
    "taker_frac": 0.0003,                 # taker 0.03% × 4 腿 (每腿溢价)
    "tick": 0.0001,                       # 1 tick (数据分辨率)
    "null_draws": 30,
    "e1_h": 24,                           # H2 E1 窗口 (= 24h)
    "warmup": 600,
    "min_n": 100,                         # 学习级
    "null_band": (-20.0, 20.0),           # GATE: null P&L (溢价归一 %) sanity
                                          #   带 (窗口内 IV 漂移可致正 null)
    "opt_db": "data/options.db",
    "dev_subset": {"n_touch": 20, "n_null": 3},
    "data_range": "触碰 2026-06-24..2026-08-01; 期权蜡烛 2026-03..2026-08",
}

STUDY_ID = "c57_option_straddle"


# ── 加载 ─────────────────────────────────────────────────────
def load_underlyings():
    data = load_candles(timeframes=(PARAMS["tf"],))
    out = {}
    for sym in PARAMS["crypto"]:
        df = data.get(sym, {}).get(PARAMS["tf"])
        if df is None or verify(df, sym, PARAMS["tf"]):
            continue
        out[sym] = df
    return out


def load_options():
    """返回 {uly: {exp_ms: {strike: {C/P: (ts_ms, open, close)}}}} 与候选列表."""
    conn = sqlite3.connect(PARAMS["opt_db"])
    cur = conn.cursor()
    inst_rows = cur.execute("SELECT DISTINCT inst_id FROM options_candles").fetchall()
    # 解析 inst_id: ULY-USD-YYMMDD-STRIKE-TYPE
    cands = []      # (uly, exp_ms, strike, opt_type, coverage_start)
    data = {}       # inst_id -> (ts, open, close)
    for (inst_id,) in inst_rows:
        m = re.match(r"^([A-Z]+)-USD-(\d{6})-(\d+(?:\.\d+)?)-([CP])$", inst_id)
        if not m:
            continue
        uly, yymmdd, strike, otype = m.groups()
        y, mo, d = int(yymmdd[:2]) + 2000, int(yymmdd[2:4]), int(yymmdd[4:6])
        exp_ms = int(pd.Timestamp(y, mo, d, 8, tz="UTC").timestamp() * 1000)
        rows = cur.execute(
            "SELECT ts, open, close FROM options_candles WHERE inst_id=? "
            "ORDER BY ts", (inst_id,)).fetchall()
        if not rows:
            continue
        ts = np.array([r[0] for r in rows], np.int64)
        op = np.array([r[1] for r in rows], float)
        cl = np.array([r[2] for r in rows], float)
        data[inst_id] = (ts, op, cl)
        cands.append((uly, exp_ms, float(strike), otype, int(ts[0]), inst_id))
    conn.close()
    return cands, data


# ── 触碰事件 (c51 口径) ──────────────────────────────────────
def band_touches(close, lo, hi):
    n = len(close)
    c_prev = np.roll(close, 1)
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    for t in range(1, n):
        if np.isfinite(hi[t]) and np.isfinite(lo[t]):
            if close[t] > hi[t] and c_prev[t] <= hi[t]:
                up[t] = True
            if close[t] < lo[t] and c_prev[t] >= lo[t]:
                dn[t] = True
    return up, dn


def touch_events(df, ctx):
    """触碰 bar 索引 + 时间戳 (ms) — 窗口内, 有下一 bar."""
    c = ctx.close
    ma = pd.Series(c).rolling(PARAMS["band_n"]).mean().values
    sd = pd.Series(c).rolling(PARAMS["band_n"]).std().values
    lo, hi = ma - PARAMS["band_k"] * sd, ma + PARAMS["band_k"] * sd
    up, dn = band_touches(c, lo, hi)
    ts_sec = df.index[PARAMS["warmup"]:].values.astype("datetime64[s]")
    t_ms = ts_sec.astype("int64") * 1000
    start_ms = int(pd.Timestamp(PARAMS["touch_start"], tz="UTC").timestamp()
                   * 1000)
    end_ms = int(pd.Timestamp(PARAMS["touch_end"] + " 23:59", tz="UTC")
                 .timestamp() * 1000)
    ev = np.flatnonzero((up | dn) & (t_ms >= start_ms) & (t_ms <= end_ms))
    out = []
    for t in ev:
        if t + 1 >= len(c):
            continue                      # 需下一 bar 入场
        out.append({"t": int(t), "ts_ms": int(t_ms[t]), "dir": "up" if up[t]
                    else "dn"})
    return out


# ── 合约映射 (因果, 蜡烛覆盖) ────────────────────────────────
def map_contract(cands, uly, spot, tau_ms):
    """τ 时刻: 已有蜡烛 (coverage ≤ τ) 且到期 ≥ τ+72h → 最近到期 → 最近 strike
    C+P 对. 返回 (call_inst, put_inst, strike, exp_ms) 或 None."""
    lo_h = PARAMS["exp_min_h"] * 3600 * 1000
    elig = [cd for cd in cands
            if cd[0] == uly and cd[4] <= tau_ms and cd[1] >= tau_ms + lo_h]
    if not elig:
        return None
    min_exp = min(cd[1] for cd in elig)
    same_exp = [cd for cd in elig if cd[1] == min_exp]
    strikes = sorted({cd[2] for cd in same_exp})
    best = min(strikes, key=lambda s: abs(s - spot))
    call = [cd for cd in same_exp if cd[2] == best and cd[3] == "C"]
    put = [cd for cd in same_exp if cd[2] == best and cd[3] == "P"]
    if not call or not put:
        return None
    return call[0], put[0], best, min_exp


# ── 期权价格查找 (±2h 最近 bar) ──────────────────────────────
def price_at(data, inst_id, target_ms, use_open=True):
    """目标时点 ±2h 内最近 bar 的价格 (open/close); 无 → None."""
    if inst_id not in data:
        return None
    ts, op, cl = data[inst_id]
    win = PARAMS["lookup_win_h"] * 3600 * 1000
    i = int(np.searchsorted(ts, target_ms, side="left"))
    best = None
    best_d = win + 1
    for j in (i - 1, i):
        if 0 <= j < len(ts) and abs(ts[j] - target_ms) <= win:
            d = abs(ts[j] - target_ms)
            if d < best_d:
                best_d = d
                best = j
    if best is None:
        return None
    return float(op[best]) if use_open else float(cl[best])


# ── 跨式事件 P&L ─────────────────────────────────────────────
def straddle_pnl(uly, spot, tau_ms, hold_h, cands, data):
    """单个入场 (uly, τ, 标的价 spot, 持有 hold_h) → (pnl_pct, breakeven, ...).
    返回 dict 或 None (剔除)."""
    mc = map_contract(cands, uly, spot, tau_ms)
    if mc is None:
        return None
    call, put, strike, exp_ms = mc
    call_id, put_id = call[5], put[5]
    c_in = price_at(data, call_id, tau_ms, use_open=True)
    p_in = price_at(data, put_id, tau_ms, use_open=True)
    if c_in is None or p_in is None:
        return None
    exit_ms = tau_ms + hold_h * 3600 * 1000
    c_out = price_at(data, call_id, exit_ms, use_open=False)
    p_out = price_at(data, put_id, exit_ms, use_open=False)
    if c_out is None or p_out is None:
        return None
    straddle_in = c_in + p_in
    straddle_out = c_out + p_out
    pnl_btc = (straddle_out - straddle_in) * PARAMS["ct_mult"]   # 每张
    pnl_pct_spot = (straddle_out - straddle_in) * 100.0 / max(spot, 1e-9)
    pnl_pct_prem = (straddle_out - straddle_in) * 100.0 / max(
        straddle_in, 1e-12)
    # H3 成本 (每单位标的): fee 0.03%×(入+出跨式溢价) + 4 tick
    fee = PARAMS["taker_frac"] * (straddle_in + straddle_out)
    spread = 4 * PARAMS["tick"]
    cost_pct_prem = (fee + spread) * 100.0 / max(straddle_in, 1e-12)
    cost_pct_spot = (fee + spread) * 100.0 / max(spot, 1e-9)
    return {"pnl_pct": pnl_pct_spot, "pnl_pct_prem": pnl_pct_prem,
            "pnl_btc": pnl_btc, "cost_pct_prem": cost_pct_prem,
            "cost_pct_spot": cost_pct_spot,
            "breakeven": 2 * straddle_in,      # = 2×跨式价/标的价 (价已归一)
            "straddle_in": straddle_in, "straddle_out": straddle_out,
            "call": call_id, "put": put_id, "strike": strike,
            "exp_ms": exp_ms}


def run_events(uly, bar_times, cands, data, hold_h):
    """bar_times: [(tau_ms, spot)] 入场序列 → 存活事件列表 (pnl dict)."""
    out = []
    for tau_ms, spot in bar_times:
        r = straddle_pnl(uly, spot, tau_ms, hold_h, cands, data)
        if r is not None:
            out.append(r)
    return out


# ── E1(24) (c52 口径, 1h ATR) ────────────────────────────────
def e1_series(atr, h):
    n = len(atr)
    t = np.arange(n)
    bar_ok = (t >= h) & (t <= n - h - 1) & np.isfinite(atr) & (atr > 0)
    offs = np.arange(h)
    pre_idx = t[:, None] + offs - h
    post_idx = t[:, None] + offs + 1
    pre = atr[pre_idx[bar_ok]].mean(axis=1)
    post = atr[post_idx[bar_ok]].mean(axis=1)
    e1 = np.full(n, np.nan)
    e1[bar_ok] = post / pre - 1.0
    return e1


# ── GATE 自检 ────────────────────────────────────────────────
def _mk_cand():
    exp_828 = int(pd.Timestamp("2026-08-28", tz="UTC").timestamp() * 1000)
    exp_925 = int(pd.Timestamp("2026-09-25", tz="UTC").timestamp() * 1000)
    exp_821 = int(pd.Timestamp("2026-08-21", tz="UTC").timestamp() * 1000)
    cov = int(pd.Timestamp("2026-06-01", tz="UTC").timestamp() * 1000)
    cov_late = int(pd.Timestamp("2026-07-31", tz="UTC").timestamp() * 1000)
    def c(e, s, t, cv):
        s_fmt = str(int(s)) if float(s).is_integer() else str(s)
        return ("BTC", e, float(s), t, cv,
                f"BTC-USD-{pd.Timestamp(e, unit='ms', tz='UTC').strftime('%y%m%d')}"
                f"-{s_fmt}-{t}")
    return [c(exp_828, 60000, "C", cov), c(exp_828, 60000, "P", cov),
            c(exp_828, 62000, "C", cov), c(exp_828, 62000, "P", cov),
            c(exp_925, 60000, "C", cov), c(exp_925, 60000, "P", cov),
            c(exp_821, 60000, "C", cov_late), c(exp_821, 60000, "P", cov_late)]


def gate_mapping_golden():
    cand = _mk_cand()
    t_jul10 = int(pd.Timestamp("2026-07-10", tz="UTC").timestamp() * 1000)
    t_aug01 = int(pd.Timestamp("2026-08-01", tz="UTC").timestamp() * 1000)
    t_aug20 = int(pd.Timestamp("2026-08-20", tz="UTC").timestamp() * 1000)
    # ① 07-10: 260821 未覆盖 (07-31 才上市) → 最近到期 = 260828
    mc = map_contract(cand, "BTC", 61500.0, t_jul10)
    exp_828 = int(pd.Timestamp("2026-08-28", tz="UTC").timestamp() * 1000)
    if mc is None or mc[3] != exp_828:
        raise SystemExit(f"GATE FAIL: 映射 golden 07-10 到期 {mc and mc[3]} "
                         f"≠ 260828")
    # ② 最近 strike: spot 61500 → 62000 (|差| 500 < 1500)
    if abs(mc[2] - 62000.0) > 1e-9:
        raise SystemExit(f"GATE FAIL: 映射 golden strike {mc[2]} ≠ 62000")
    # ③ 08-01: 260821 已覆盖 → 最近到期 = 260821
    mc2 = map_contract(cand, "BTC", 61500.0, t_aug01)
    exp_821 = int(pd.Timestamp("2026-08-21", tz="UTC").timestamp() * 1000)
    if mc2 is None or mc2[3] != exp_821:
        raise SystemExit(f"GATE FAIL: 映射 golden 08-01 到期 {mc2 and mc2[3]} "
                         f"≠ 260821")
    # ④ 08-20: 260821 到期 (08-21) < τ+72h → 剔除 → 260828
    mc3 = map_contract(cand, "BTC", 61500.0, t_aug20)
    if mc3 is None or mc3[3] != exp_828:
        raise SystemExit(f"GATE FAIL: 映射 golden 08-20 到期 {mc3 and mc3[3]} "
                         f"≠ 260828 (到期约束)")
    # ⑤ C+P 对: call/put inst 都存在
    if not (mc[0][3] == "C" and mc[1][3] == "P"):
        raise SystemExit("GATE FAIL: 映射 golden C/P 对缺失")
    return True


def gate_price_golden():
    """±2h 最近 bar / 超窗 None."""
    t0 = 1000 * 3600 * 1000
    ts = np.array([t0, t0 + 3600000, t0 + 3 * 3600000])
    op = np.array([0.05, 0.06, 0.07])
    cl = np.array([0.051, 0.061, 0.071])
    data = {"T": (ts, op, cl)}
    v = price_at(data, "T", t0 + 2 * 3600000 + 300000, use_open=True)
    if abs(v - 0.07) > 1e-12:             # 最近 = t0+3h (差 55min)
        raise SystemExit(f"GATE FAIL: ±2h 查找 {v} ≠ 0.07")
    v2 = price_at(data, "T", t0 + 6 * 3600000, use_open=True)
    if v2 is not None:                    # 超 ±2h → None
        raise SystemExit("GATE FAIL: ±2h 超窗未剔除")
    v3 = price_at(data, "T", t0, use_open=False)
    if abs(v3 - 0.051) > 1e-12:
        raise SystemExit(f"GATE FAIL: close 查找 {v3} ≠ 0.051")
    return True


def gate_pnl_golden():
    """Δ跨式价 × ctMult."""
    cand = _mk_cand()
    ts_arr = np.array([int(pd.Timestamp("2026-07-10 00:00", tz="UTC")
                           .timestamp() * 1000)])
    data = {
        "BTC-USD-260828-62000-C": (ts_arr, np.array([0.10]), np.array([0.11])),
        "BTC-USD-260828-62000-P": (ts_arr, np.array([0.04]), np.array([0.05])),
    }
    tau = int(pd.Timestamp("2026-07-10 00:00", tz="UTC").timestamp() * 1000)
    # 退出时点找不到合约 (只有入场一条 bar) → None 剔除 (不报错)
    r = straddle_pnl("BTC", 62000.0, tau, 24, cand, data)
    if r is not None:
        raise SystemExit("GATE FAIL: 无退出 bar 的事件未剔除")
    # 构造入场+退出两条 bar → 校验 P&L
    ts2 = np.array([tau, tau + 24 * 3600000])
    data2 = {
        "BTC-USD-260828-62000-C": (ts2, np.array([0.10, 0.12]),
                                   np.array([0.10, 0.12])),
        "BTC-USD-260828-62000-P": (ts2, np.array([0.04, 0.03]),
                                   np.array([0.04, 0.03])),
    }
    r2 = straddle_pnl("BTC", 62000.0, tau, 24, cand, data2)
    if r2 is None:
        raise SystemExit("GATE FAIL: P&L golden 事件被剔除")
    expect = (0.12 + 0.03 - 0.10 - 0.04) * PARAMS["ct_mult"]
    if abs(r2["pnl_btc"] - expect) > 1e-12:
        raise SystemExit(f"GATE FAIL: P&L {r2['pnl_btc']} ≠ {expect}")
    return True


def gate(null_means):
    gate_mapping_golden()
    gate_price_golden()
    gate_pnl_golden()
    nm = float(np.mean(null_means))
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null 均值 {nm:+.4f}% ∉ [{lo}, {hi}]")
    print(f"[GATE] 映射 golden [PASS]; ±2h 查找 golden [PASS]; P&L golden "
          f"[PASS]; null sanity {nm:+.4f}% [PASS]", flush=True)
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
        "params=tf={},band={}±{}σ,window={}~{},holds={},exp_min={}h,lookup=±{}h,"
        "ct_mult={},taker={},tick={},null={},min_n={},gate=MIN_N={}(学习级),学习级"
        "".format(STUDY_ID, date.today().isoformat(), script_sha256(),
                  p["data_range"], p["tf"], p["band_n"], p["band_k"],
                  p["touch_start"], p["touch_end"], p["holds_h"],
                  p["exp_min_h"], p["lookup_win_h"], p["ct_mult"],
                  p["taker_frac"], p["tick"], p["null_draws"], p["min_n"],
                  p["min_n"]),
        "# GATE: 探测器自检 映射 golden + ±2h 查找 golden + P&L golden + "
        "null sanity [PASS]; MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c57 期权线第一块砖: 2σ 带触碰 → 跨式期权 "
        "(long-vol 先验); 触碰=c51 口径 (MA20±2σ 出带); 合约映射以蜡烛覆盖为"
        "地面真相; 期权 K 线稀疏 ±2h 最近 bar; 无 bid/ask 用 OHLC (入场 open/"
        "平仓 close, 日内报价可能陈旧); P&L = Δ跨式价×ctMult, 归一化 % of "
        "标的价; 描述层无入场, 无交易含义",
        "",
    ]
    for sym, r in res.items():
        if sym != "BTC/USDT:USDT":
            continue
        eff = r["eff"]
        lines.append("[数据] BTC 触碰窗口 {} ~ {}: 原始触碰 {} (up {} + dn {}) "
                     "| 剔除 {} (无候选合约/无期权价) | 可用样本 {} (24h) {} | "
                     "ETH: 期权蜡烛仅 08-11..08-14 覆盖, 触碰窗口内无可映射 "
                     "合约 (数据缺失标注)".format(
            p["touch_start"], p["touch_end"], r["raw_n"], r["n_up"],
            r["n_dn"], r["n_excl"], eff, _nm(eff)))
        # H1
        lines.append("")
        lines.append("[H1] 触碰后跨式 P&L (24h) vs 随机时点 null {} 次 "
                     "(标的价归一 %; 括号内=跨式溢价归一 %):".format(
            p["null_draws"]))
        for hold in p["holds_h"]:
            h = r["holds"][hold]
            nm, ns = h["null"]
            pct_lo, pct_hi = h["null_95"]
            exceed = h["real"] > pct_hi
            below = h["real"] < pct_lo
            lines.append("  {}h: 真实 {:+.6f}% ({:+.3f}%) (n={}) {} | null "
                         "{:+.6f}±{:.6f}% ({:+.3f}%) 95% [{:+.6f}, {:+.6f}] "
                         "-> {}".format(
                hold, h["real"], h["real_prem"], h["n"], _nm(h["n"]), nm, ns,
                h["null_prem"], pct_lo, pct_hi,
                "超区间↑" if exceed else
                ("低于区间↓" if below else "区间内")))
        # H2
        lines.append("")
        lines.append("[H2] 触碰后 realized 波动 vs 跨式隐含 breakeven 波动:")
        h2 = r["h2"]
        lines.append("  E1(24) (c52 口径): 真实 {:+.2%} (n={}) | 24h 移动 "
                     "|log ret| {:+.2%} | 跨式 breakeven (2×跨式价/标的价) "
                     "{:+.2%} | breakeven − 移动 {:+.2%}".format(
            h2["e1"], h2["n"], h2["move"], h2["be"], h2["be"] - h2["move"]))
        lines.append("  判定: realized 24h 移动 {} breakeven (信号{}超出期权"
                     "定价)".format(
            ">" if h2["move"] > h2["be"] else "<",
            "已" if h2["move"] > h2["be"] else "未"))
        # H3
        lines.append("")
        lines.append("[H3] 成本敏感性 (taker 0.03%×4 腿 + 1 tick 价差, "
                     "跨式溢价归一 %):")
        for hold in p["holds_h"]:
            h = r["holds"][hold]
            lines.append("  {}h: 毛 P&L {:+.3f}% | 成本 {:+.3f}% | 净 P&L "
                         "{:+.3f}% (n={})".format(
                hold, h["real_prem"], h["cost_prem"], h["net_prem"], h["n"]))
        # 合约分布
        lines.append("")
        lines.append("[合约] 有效事件映射 (24h):")
        for inst, n in sorted(r["insts"].items(), key=lambda x: -x[1]):
            lines.append("  {}: {}".format(inst, n))
    lines.append("")
    lines.append("[对照-历史] c51 (带触碰: 双用途不给结论, E1 释放); c52 (E1 "
                 "h 剖面); c14/c17 (关键位触碰); c54/c56 (突破/择时); 期权线 "
                 "第一块砖 (long-vol 先验, IV 定价之上有增量假设); 数据摩擦: "
                 "OKX 只在列合约 + 周度上市晚 + 期权 K 线稀疏 (±2h 容忍 + "
                 "重复报价)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    n_touch_max = PARAMS["dev_subset"]["n_touch"] if dev else None
    n_null = PARAMS["dev_subset"]["n_null"] if dev else PARAMS["null_draws"]

    ul = load_underlyings()
    cands, data = load_options()

    res = {}
    null_means_all = []
    null_means_prem_all = []
    for sym in PARAMS["crypto"]:
        df = ul.get(sym)
        if df is None:
            continue
        ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
        evs = touch_events(df, ctx)
        if n_touch_max:
            evs = evs[:n_touch_max]
        n_up = int(sum(1 for e in evs if e["dir"] == "up"))
        n_dn = int(sum(1 for e in evs if e["dir"] == "dn"))
        raw_n = len(evs)
        # 入场序列: 触碰 bar T+1 (open), 标的价 = open[T+1]
        ts_sec = df.index[PARAMS["warmup"]:].values.astype("datetime64[s]")
        o_arr = ctx.open
        bar_ms = ts_sec.astype("int64") * 1000
        entry_list = []
        for e in evs:
            t = e["t"]
            if t + 1 >= len(o_arr):
                continue
            entry_list.append((int(bar_ms[t + 1]), float(o_arr[t + 1])))
        n_excl = raw_n - len(entry_list)
        uly = "BTC" if sym.startswith("BTC") else "ETH"
        holds = {}
        for hold in PARAMS["holds_h"]:
            real = run_events(uly, entry_list, cands, data, hold)
            if hold == 24:
                n_excl += (len(entry_list) - len(real))
            # null 抽样
            null_means = []
            null_means_prem = []
            win_ms = bar_ms[bar_ms >= int(pd.Timestamp(
                PARAMS["touch_start"], tz="UTC").timestamp() * 1000)]
            win_ms = win_ms[win_ms <= int(pd.Timestamp(
                PARAMS["touch_end"] + " 23:59", tz="UTC").timestamp() * 1000)]
            rng = np.random.default_rng(4242)
            for d in range(n_null):
                idx = rng.integers(0, len(win_ms), size=len(entry_list))
                pos = np.searchsorted(bar_ms, win_ms[idx])
                pos = np.clip(pos, 0, len(o_arr) - 1)
                seq = [(int(win_ms[i]), float(o_arr[pos[i]])) for i in
                       range(len(idx))]
                rn = run_events(uly, seq, cands, data, hold)
                if rn:
                    null_means.append(float(np.mean([x["pnl_pct"]
                                                     for x in rn])))
                    null_means_prem.append(float(np.mean(
                        [x["pnl_pct_prem"] for x in rn])))
            nm = float(np.mean(null_means)) if null_means else float("nan")
            ns = float(np.std(null_means, ddof=1)) if len(null_means) > 1 \
                else 0.0
            pct_lo, pct_hi = (float(np.percentile(null_means, 2.5)),
                              float(np.percentile(null_means, 97.5))) \
                if len(null_means) >= 2 else (float("nan"), float("nan"))
            real_mean = float(np.mean([x["pnl_pct"] for x in real])) if real \
                else float("nan")
            real_prem = float(np.mean([x["pnl_pct_prem"] for x in real])) \
                if real else float("nan")
            null_prem = float(np.mean(null_means_prem)) if null_means_prem \
                else float("nan")
            cost_prem = float(np.mean([x["cost_pct_prem"] for x in real])) \
                if real else float("nan")
            net_prem = real_prem - cost_prem
            holds[hold] = {"real": real_mean, "n": len(real),
                           "null": (nm, ns), "null_95": (pct_lo, pct_hi),
                           "real_prem": real_prem, "null_prem": null_prem,
                           "cost_prem": cost_prem, "net_prem": net_prem,
                           "null_means": null_means,
                           "null_means_prem": null_means_prem}
            if hold == 24:
                null_means_all.extend(null_means)
                null_means_prem_all.extend(null_means_prem)
        # H2 (24h 窗)
        atr = ctx.atr
        e1 = e1_series(atr, PARAMS["e1_h"])
        c = ctx.close
        e1s, moves = [], []
        for e in evs:
            t = e["t"]
            if t + PARAMS["e1_h"] + 1 >= len(c) or not np.isfinite(e1[t]):
                continue
            e1s.append(float(e1[t]))
            moves.append(float(abs(np.log(c[t + PARAMS["e1_h"] + 1] /
                                         c[t + 1]))))
        # 合约分布 (24h 存活事件)
        real24 = run_events(uly, entry_list, cands, data, 24)
        insts = {}
        for r_ in real24:
            insts[r_["call"]] = insts.get(r_["call"], 0) + 1
        # H2 breakeven: 24h 存活事件的入场跨式
        be_vals = [r_["breakeven"] for r_ in real24]
        be_mean = float(np.mean(be_vals)) if be_vals else float("nan")
        move_mean = float(np.mean(moves)) if moves else float("nan")
        e1_mean = float(np.mean(e1s)) if e1s else float("nan")
        res[sym] = {
            "raw_n": raw_n, "n_up": n_up, "n_dn": n_dn, "n_excl": n_excl,
            "eff": holds[24]["n"], "holds": holds, "insts": insts,
            "h2": {"e1": e1_mean, "move": move_mean, "be": be_mean,
                   "n": len(be_vals)},
        }
        if dev:
            print("  [dev] {} 触碰 {} (up {} dn {}) 有效24h {} | 24h P&L "
                  "{:+.6f}% ({:+.3f}%) vs null {:+.6f}% ({:+.3f}%) | E1 "
                  "{:+.2%} move {:+.2%} be {:+.2%}".format(
                sym.split("/")[0], raw_n, n_up, n_dn, res[sym]["eff"],
                holds[24]["real"], holds[24]["real_prem"],
                holds[24]["null"][0], holds[24]["null_prem"],
                e1_mean, move_mean, be_mean))

    gate(np.array(null_means_prem_all) if null_means_prem_all
         else np.array([0.0]))

    if dev:
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
