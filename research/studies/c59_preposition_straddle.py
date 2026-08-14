#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C59 期权线第三块砖: 事前预置 (高 ER 逼近关键位买跨式) (2026-08-14,
无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c59 行): c57 显示触碰后买跨式买在 IV 高位 (亏)。
  本砖检验**事前**信号: 高 ER (趋势性, c27 rolling 分位) + 价格逼近关键位
  (c14 cluster_levels) 且未触碰时**预置**买跨式 — 能否领先 IV 抬升?
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。**结论不得作交易
  依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、
  内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 高 ER 逼近关键位的预置买跨式是否 > 随机
  时点 null? 是否规避 c57 触碰后买入的 IV 高估?

预注册假设 (PLAN §2.5 c59 行, docstring 逐字):
  H1: 预置跨式 P&L 均值 > 随机时点 null (30 次) 95% 区间
      (事前信号领先 IV 抬升)
  H2: 预置 vs c57 触碰后买入 (同窗口同合约口径) — 事前是否规避 IV 高估
  H3: 成本敏感性

  操作化 (运行前锁定):
    - 事件: BTC 1h; 预置 bar = 高 ER 分位 (ER(10) ≥ rolling 120 分位 0.80,
      c27 口径) + 价格逼近关键位 (距最近同侧 cluster_levels 位 < 1×ATR,
      阻力在上/支撑在下, 未触碰) 且未触碰的 bar; 同一逼近序列只取首个 bar
    - 入场: 预置 bar 下一 bar 开盘买跨式 (strike 最近、到期 ≥ τ+72h 最近、
      蜡烛覆盖=可交易)
    - 持有: 至触碰发生 (标的触及该关键位) 或 48h, 先到先平; 平仓价 = 平仓
      bar 收盘 (±2h 最近期权 bar)
    - H1: P&L 均值 vs 随机时点 null 30 次 (同数量同规则: 随机 bar → 最近同侧
      位 → 同持有逻辑)
    - H2: 同窗口同口径 c57 触碰后买入 (触碰事件 → 24h 买跨式) 内联对拍
    - H3: 成本 = 0.03%×(入+出跨式价)+4 tick (溢价归一)
    - 事件数审计: 高 ER+逼近同现可能稀少, n<100 标注降级
    - 学习级: BTC、30 次 null、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  ER(10)           | |c[t]−c[t−n]|/Σ|Δc| (c27 前缀和)      | bar 收盘后 | c27 口径
  高 ER            | ER ≥ rolling_percentile(ER,120,0.80)  | bar 收盘后 | research.causal
                   |   (禁全样本分位)                       |            |   (B4e 教训)
  关键位           | levels.cluster_levels (在线聚类+冻结)  | confirm_at | 冻结后不变 (c14 R1/R2)
  逼近             | 距同侧位 < 1×ATR 且未触碰              | bar 收盘后 | 同侧=阻力在上/支撑在下
  去重             | 连续逼近序列取首个 bar                 | bar 收盘后 | 预注册
  持有/平仓        | 触碰位或 48h 先到先平, 平仓 bar 收盘   | bar 收盘后 | 收盘价成交
  合约映射/价格    | c57 同款 (蜡烛覆盖/±2h/最近 strike)    | τ 时刻    | 同 c57
  null            | 随机 bar → 最近同侧位 → 同持有逻辑 30 次 | 锚定真实   | 同规则对照

数据声明: data/backtest.db (BTC 1h, 止 2026-08-01); data/options.db (BTC
  期权蜡烛 2026-03..2026-08); 同 c57/c58。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  ER(10), 分位窗 120, q=0.80; 逼近阈值 1×ATR; 持有上限 48h; 窗口
  2026-06-24..2026-08-01; 到期 ≥ τ+72h; ±2h 查找; ctMult=0.01; taker
  0.03%×4 腿 + 1 tick; null 30 次; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - null 的"同规则"= 随机 bar 映射最近同侧位 → 触碰或 48h 先到先平 (随机
    bar 距位远, 多数 48h 平仓 — 与真实预置的短持有形成对照)。
  - 平仓 bar 的期权价用 close (收盘价成交); 触碰 = intrabar 触及位价。
  - H2 的 c57 对拍内联计算 (触碰事件 → 24h 买跨式, 同映射同 P&L 口径)。
  - 学习级: 无 BY_YEAR; 30 次 null 沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① ER golden (已知序列: 单调涨 → ER=1 高; 交替 → ER≈0);
    ② 逼近 golden (构造已知位 + 逼近路径: 首 bar 检测 + 去重 + 同侧 + 未触
    碰 + 触碰退出); ③ 映射/±2h 查找 golden (c57 同款); ④ null sanity —
    随机 null 均值 ∈ [−20%, +20%] (溢价归一)
  - null 无信息对照: 随机时点 30 次 (同规则)
  - MIN_N: 每格 n ≥ 100 (学习级); 预置事件稀少则标注降级
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 预置事件上限 20 × null 3 次, 不写 .out
  - 全量: BTC 全预置 × null 30 次 (预计 ≤5 分钟)

运行命令:
  python3 research/studies/c59_preposition_straddle.py --dev
  python3 research/studies/c59_preposition_straddle.py
"""
import hashlib
import os
import re
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
from research.data_loader import DB_PATH, load_candles, verify
from research.levels import cluster_levels
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT",),
    "tf": "1h",
    "win_start": "2026-06-24",
    "win_end": "2026-08-01",
    "er_n": 10,
    "er_win": 120,
    "er_q": 0.80,
    "appr_thr": 1.0,                     # 距同侧位 < 1×ATR
    "hold_max_h": 48,
    "exp_min_h": 72,
    "lookup_win_h": 2,
    "ct_mult": 0.01,
    "taker_frac": 0.0003,
    "tick": 0.0001,
    "null_draws": 30,
    "warmup": 600,
    "min_n": 100,
    "null_band": (-20.0, 20.0),
    "opt_db": "data/options.db",
    "dev_subset": {"n_ev": 20, "n_null": 3},
    "data_range": "触碰 2026-06-24..2026-08-01; 期权蜡烛 2026-03..2026-08",
}

STUDY_ID = "c59_preposition_straddle"


# ── 加载 (c57 同款) ──────────────────────────────────────────
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
    conn = sqlite3.connect(PARAMS["opt_db"])
    cur = conn.cursor()
    inst_rows = cur.execute("SELECT DISTINCT inst_id FROM options_candles").fetchall()
    cands = []
    data = {}
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


def map_contract(cands, uly, spot, tau_ms):
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


def price_at(data, inst_id, target_ms, use_open=True):
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


# ── ER (c27 口径) ────────────────────────────────────────────
def er_series(c, n):
    c = np.asarray(c, float)
    N = len(c)
    t_idx = np.arange(N)
    c_prev = np.roll(c, 1)
    m1 = t_idx >= 1
    ad = np.where(m1, np.abs(c - c_prev), 0.0)
    prefix_ad = np.concatenate([[0], np.cumsum(ad)])
    ok = t_idx >= n
    net = np.full(N, np.nan)
    net[ok] = np.abs(c[t_idx[ok]] - c[t_idx[ok] - n])
    path = np.full(N, np.nan)
    path[ok] = prefix_ad[t_idx[ok] + 1] - prefix_ad[t_idx[ok] - n + 1]
    er = np.full(N, np.nan)
    m_er = ok & (path > 0)
    er[m_er] = net[m_er] / path[m_er]
    return er


# ── 关键位逼近 ───────────────────────────────────────────────
def ahead_level(lvls, t, close_t):
    """最近同侧已确认位 (阻力在上 / 支撑在下). 返回 (side, price, dist) 或 None."""
    best = None
    for lv in lvls:
        if lv.confirm_at > t:
            continue
        if lv.side == "resistance" and lv.price > close_t:
            d = lv.price - close_t
            if best is None or d < best[2]:
                best = ("R", lv.price, d)
        elif lv.side == "support" and lv.price < close_t:
            d = close_t - lv.price
            if best is None or d < best[2]:
                best = ("S", lv.price, d)
    return best


def preposition_events(close, high, low, atr, er_hi, lvls, t_ms, start_ms,
                       end_ms, thr):
    """预置 bar: 高 ER + 逼近 (同侧 < thr×ATR, 未触碰) + 连续序列首个.
    返回 [(t, side, price)]."""
    n = len(close)
    ev = []
    prev_ok = False
    for t in range(1, n):
        if not (start_ms <= t_ms[t] <= end_ms):
            prev_ok = False
            continue
        if not er_hi[t] or not np.isfinite(atr[t]) or atr[t] <= 0:
            prev_ok = False
            continue
        al = ahead_level(lvls, t, close[t])
        if al is None or al[2] >= thr * atr[t]:   # 预注册: 距 < 1×ATR (严格)
            prev_ok = False
            continue
        side, price, dist = al
        if side == "R" and high[t] >= price:   # 已触碰 → 不是逼近
            prev_ok = False
            continue
        if side == "S" and low[t] <= price:
            prev_ok = False
            continue
        if not prev_ok:                        # 连续序列首个
            ev.append((t, side, price))
        prev_ok = True
    return ev


def exit_idx(t, side, price, high, low, n, max_h):
    """从 t+1 起扫描: 触碰 (R: high≥price / S: low≤price) 或 max_h bar."""
    end = min(t + max_h, n - 1)
    for tp in range(t + 1, end + 1):
        if side == "R" and high[tp] >= price:
            return tp
        if side == "S" and low[tp] <= price:
            return tp
    return end


# ── 买侧跨式 P&L (变持有) ────────────────────────────────────
def straddle_buy(uly, spot, tau_ms, exit_ms, cands, data):
    """τ 开盘买跨式, exit_ms 收盘平仓 (先到先平). 返回 dict 或 None."""
    mc = map_contract(cands, uly, spot, tau_ms)
    if mc is None:
        return None
    call, put, strike, exp_ms = mc
    call_id, put_id = call[5], put[5]
    c_in = price_at(data, call_id, tau_ms, use_open=True)
    p_in = price_at(data, put_id, tau_ms, use_open=True)
    if c_in is None or p_in is None:
        return None
    c_out = price_at(data, call_id, exit_ms, use_open=False)
    p_out = price_at(data, put_id, exit_ms, use_open=False)
    if c_out is None or p_out is None:
        return None
    straddle_in = c_in + p_in
    straddle_out = c_out + p_out
    gross = straddle_out - straddle_in
    pnl_btc = gross * PARAMS["ct_mult"]
    pnl_pct_spot = gross * 100.0 / max(spot, 1e-9)
    pnl_pct_prem = gross * 100.0 / max(straddle_in, 1e-12)
    fee = PARAMS["taker_frac"] * (straddle_in + straddle_out)
    spread = 4 * PARAMS["tick"]
    cost_pct_prem = (fee + spread) * 100.0 / max(straddle_in, 1e-12)
    return {"pnl_pct": pnl_pct_spot, "pnl_pct_prem": pnl_pct_prem,
            "pnl_btc": pnl_btc, "cost_pct_prem": cost_pct_prem,
            "breakeven": 2 * straddle_in,
            "call": call_id, "put": put_id, "strike": strike,
            "exp_ms": exp_ms}


def run_prep_events(uly, events, bar_ms, o_arr, close, high, low, cands,
                    data, max_h):
    """预置事件序列 → (tau_ms, exit_ms, spot) → 存活 P&L."""
    out = []
    for (t, side, price) in events:
        if t + 1 >= len(o_arr):
            continue
        tau = int(bar_ms[t + 1])
        spot = float(o_arr[t + 1])
        ex = exit_idx(t + 1, side, price, high, low, len(close), max_h)
        exit_ms = int(bar_ms[ex])
        r = straddle_buy(uly, spot, tau, exit_ms, cands, data)
        if r is not None:
            r["hold_h"] = (ex - (t + 1))
            out.append(r)
    return out


# ── GATE 自检 ────────────────────────────────────────────────
def gate_er_golden():
    """单调涨 → ER≈1; 交替 → ER≈0; 高 ER 掩码识别单调段."""
    c_up = np.arange(100.0, 100.0 + 200.0, 1.0)   # 单调
    er = er_series(c_up, 10)
    if not np.isfinite(er[-1]) or abs(er[-1] - 1.0) > 1e-6:
        raise SystemExit(f"GATE FAIL: 单调 ER {er[-1]} ≠ 1")
    c_alt = np.array([100.0] + [100.0 + (1 if i % 2 else -1)
                                for i in range(1, 200)])
    er2 = er_series(c_alt, 10)
    if np.isfinite(er2[-1]) and er2[-1] > 0.2:
        raise SystemExit(f"GATE FAIL: 交替 ER {er2[-1]} 应 ≈0")
    return True


class _Lv:
    def __init__(self, side, price, confirm_at):
        self.side = side
        self.price = price
        self.confirm_at = confirm_at
        self.band = 0.0


def gate_approach_golden():
    """逼近检测: 构造阻力位 R=110 (confirm=0), 路径接近 → 首 bar 检测 +
    同侧 + 未触碰; 触碰后不再逼近."""
    lvls = [_Lv("resistance", 110.0, 0), _Lv("support", 90.0, 0)]
    close = np.array([105.0] * 5 + [108.0, 109.0, 109.5, 109.8, 110.1,
                                    110.6, 111.0])
    high = close + 0.1                      # intrabar 不提前触位
    low = close - 0.1
    atr = np.full(len(close), 1.0)
    er_hi = np.zeros(len(close), bool)
    er_hi[5:] = True                       # 6 起高 ER
    t_ms = np.arange(len(close)) * 3600000
    ev = preposition_events(close, high, low, atr, er_hi, lvls, t_ms, 0,
                            len(close) * 3600000, 1.0)
    # 预期: bar 6 (close 109, 距 1×ATR 恰好 → 严格 < 不触发); bar 7 (距 0.5
    # ×ATR) 触发; bar 8 连续 → 去重; bar 9 (high 110.2 ≥ 110) 已触碰 → 不触发
    triggers = [t for t, s, p in ev]
    if triggers != [7]:
        raise SystemExit(f"GATE FAIL: 逼近触发 {triggers} ≠ [7]")
    # 退出: 从 bar 8 起扫描 R 触碰 → bar 9 (high 110.2 ≥ 110)
    ex = exit_idx(7, "R", 110.0, high, low, len(close), 48)
    if ex != 9:
        raise SystemExit(f"GATE FAIL: 退出 bar {ex} ≠ 9")
    return True


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


def gate_map_price_golden():
    cand = _mk_cand()
    t_jul10 = int(pd.Timestamp("2026-07-10", tz="UTC").timestamp() * 1000)
    t_aug01 = int(pd.Timestamp("2026-08-01", tz="UTC").timestamp() * 1000)
    exp_828 = int(pd.Timestamp("2026-08-28", tz="UTC").timestamp() * 1000)
    exp_821 = int(pd.Timestamp("2026-08-21", tz="UTC").timestamp() * 1000)
    mc = map_contract(cand, "BTC", 61500.0, t_jul10)
    if mc is None or mc[3] != exp_828 or abs(mc[2] - 62000.0) > 1e-9:
        raise SystemExit("GATE FAIL: 映射 golden")
    mc2 = map_contract(cand, "BTC", 61500.0, t_aug01)
    if mc2 is None or mc2[3] != exp_821:
        raise SystemExit("GATE FAIL: 映射 golden 08-01")
    t0 = 1000 * 3600 * 1000
    ts = np.array([t0, t0 + 3600000, t0 + 3 * 3600000])
    op = np.array([0.05, 0.06, 0.07])
    cl = np.array([0.051, 0.061, 0.071])
    data = {"T": (ts, op, cl)}
    v = price_at(data, "T", t0 + 2 * 3600000 + 300000, use_open=True)
    if abs(v - 0.07) > 1e-12:
        raise SystemExit("GATE FAIL: ±2h 查找")
    v2 = price_at(data, "T", t0 + 6 * 3600000, use_open=True)
    if v2 is not None:
        raise SystemExit("GATE FAIL: ±2h 超窗")
    return True


def gate(null_means_prem):
    gate_er_golden()
    gate_approach_golden()
    gate_map_price_golden()
    nm = float(np.mean(null_means_prem)) if null_means_prem.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null 均值 {nm:+.3f}% ∉ [{lo}, {hi}]")
    print(f"[GATE] ER golden [PASS]; 逼近/退出 golden [PASS]; 映射/±2h golden "
          f"[PASS]; null sanity {nm:+.3f}% [PASS]", flush=True)
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
        "params=tf={},er={}/{}/{}thr,appr_thr={}ATR,hold_max={}h,window={}~{},"
        "exp_min={}h,lookup=±{}h,ct_mult={},taker={},tick={},null={},min_n={},"
        "gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["tf"], p["er_n"], p["er_win"], p["er_q"],
            p["appr_thr"], p["hold_max_h"], p["win_start"], p["win_end"],
            p["exp_min_h"], p["lookup_win_h"], p["ct_mult"], p["taker_frac"],
            p["tick"], p["null_draws"], p["min_n"], p["min_n"]),
        "# GATE: 探测器自检 ER golden + 逼近/退出 golden + 映射/±2h golden + "
        "null sanity [PASS]; MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c59 期权线第三块砖: 事前预置 (高 ER 逼近关键位买"
        "跨式); 事件=ER(10)≥rolling 0.80 分位 + 距同侧 cluster 位 <1×ATR 且未"
        "触碰, 连续序列取首; 持有至触碰或 48h 先到先平, 收盘价成交; 合约映射/"
        "±2h 同 c57; 描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[数据] BTC 预置事件 {} (逼近 {}/高ER {}) | 映射后可用 {} "
                 "{} | 平均持有 {:.1f}h | 触碰退出 {:.0%} (n={})".format(
        res["raw_n"], res["n_appr"], res["n_er"], res["eff"], _nm(res["eff"]),
        res["avg_hold"], res["touch_frac"], res["eff"]))
    # H1
    lines.append("")
    lines.append("[H1] 预置跨式 P&L vs 随机时点 null {} 次 (标的价归一 %; "
                 "括号=跨式溢价归一 %):".format(p["null_draws"]))
    h = res["h1"]
    nm, ns = h["null"]
    pct_lo, pct_hi = h["null_95"]
    exceed = h["real"] > pct_hi
    below = h["real"] < pct_lo
    lines.append("  预置: 真实 {:+.6f}% ({:+.3f}%) (n={}) {} | null "
                 "{:+.6f}±{:.6f}% ({:+.3f}%) 95% [{:+.6f}, {:+.6f}] -> {}".format(
        h["real"], h["real_prem"], h["n"], _nm(h["n"]), nm, ns, h["null_prem"],
        pct_lo, pct_hi, "超区间↑" if exceed else
        ("低于区间↓" if below else "区间内")))
    # H2 vs c57
    lines.append("")
    lines.append("[H2] 预置 vs c57 触碰后买入 (同窗口同合约口径):")
    lines.append("  预置 {:+.3f}% (n={}) | c57 触碰买入 {:+.3f}% (n={}) | "
                 "预置 − c57 触碰 {:+.3f}% {}".format(
        res["h1"]["real_prem"], res["eff"], res["c57"]["real_prem"],
        res["c57"]["n"],
        res["h1"]["real_prem"] - res["c57"]["real_prem"],
        "规避IV高估↑" if res["h1"]["real_prem"] > res["c57"]["real_prem"]
        else "未规避"))
    # H3
    lines.append("")
    lines.append("[H3] 成本敏感性 (taker 0.03%×4 腿 + 1 tick, 溢价归一 %):")
    lines.append("  预置: 毛 {:+.3f}% | 成本 {:+.3f}% | 净 {:+.3f}% (n={})"
                 .format(h["real_prem"], res["cost_prem"], res["net_prem"],
                         res["eff"]))
    lines.append("  c57 触碰买入: 毛 {:+.3f}% | 成本 {:+.3f}% | 净 {:+.3f}% "
                 "(n={})".format(res["c57"]["real_prem"], res["c57"]["cost"],
                                 res["c57"]["net"], res["c57"]["n"]))
    lines.append("")
    lines.append("[对照-历史] c57 (触碰后买跨式 −4.177% vs null +2.070%, IV "
                 "高位); c58 (触碰后卖跨式 +4.177% vs null −2.000%, 收割 IV "
                 "回复); c59 (本砖: 事前预置买 — 能否领先 IV 抬升); c27 (高 "
                 "ER 趋势性); c14 (cluster 关键位); 期权线: c57 买=c58 卖的 "
                 "镜像, c59 测事前")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    n_ev_max = PARAMS["dev_subset"]["n_ev"] if dev else None
    n_null = PARAMS["dev_subset"]["n_null"] if dev else PARAMS["null_draws"]

    ul = load_underlyings()
    cands, data = load_options()
    sym = PARAMS["crypto"][0]
    df = ul.get(sym)
    if df is None:
        print("无数据")
        return 1
    ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
    c, h, l, atr = ctx.close, ctx.high, ctx.low, ctx.atr
    n = len(c)
    ts_sec = df.index[PARAMS["warmup"]:].values.astype("datetime64[s]")
    bar_ms = ts_sec.astype("int64") * 1000
    start_ms = int(pd.Timestamp(PARAMS["win_start"], tz="UTC").timestamp()
                   * 1000)
    end_ms = int(pd.Timestamp(PARAMS["win_end"] + " 23:59", tz="UTC")
                 .timestamp() * 1000)

    # ER + 高 ER 掩码
    er = er_series(c, PARAMS["er_n"])
    rp = rolling_percentile(er, PARAMS["er_win"], PARAMS["er_q"])
    er_hi = np.isfinite(rp) & (er >= rp)

    # 关键位
    lvls = cluster_levels(h, l, atr, k=K, tolerance_mult=0.3, min_touch=2)

    # 预置事件
    evs = preposition_events(c, h, l, atr, er_hi, lvls, bar_ms, start_ms,
                             end_ms, PARAMS["appr_thr"])
    n_appr = len(evs)
    n_er = int(np.sum(er_hi[(bar_ms >= start_ms) & (bar_ms <= end_ms)]))
    if n_ev_max:
        evs = evs[:n_ev_max]
    raw_n = len(evs)

    o_arr = ctx.open
    real = run_prep_events("BTC", evs, bar_ms, o_arr, c, h, l, cands, data,
                           PARAMS["hold_max_h"])
    avg_hold = float(np.mean([r["hold_h"] for r in real])) if real else \
        float("nan")
    touch_frac = float(np.mean([r["hold_h"] < PARAMS["hold_max_h"]
                                for r in real])) if real else float("nan")

    # null (随机 bar → 最近同侧位 → 同持有)
    win_idx = np.flatnonzero((bar_ms >= start_ms) & (bar_ms <= end_ms))
    rng = np.random.default_rng(6464)
    null_means = []
    null_means_prem = []
    for d in range(n_null):
        idx = rng.integers(0, len(win_idx), size=len(evs))
        rand_t = [int(win_idx[i]) for i in range(len(idx))]
        null_evs = []
        for t in rand_t:
            al = ahead_level(lvls, t, c[t])
            if al is None:
                continue
            if (al[0] == "R" and h[t] >= al[1]) or \
               (al[0] == "S" and l[t] <= al[1]):
                continue
            null_evs.append((t, al[0], al[1]))
        rn = run_prep_events("BTC", null_evs, bar_ms, o_arr, c, h, l, cands,
                             data, PARAMS["hold_max_h"])
        if rn:
            null_means.append(float(np.mean([x["pnl_pct"] for x in rn])))
            null_means_prem.append(float(np.mean([x["pnl_pct_prem"]
                                                  for x in rn])))
    nm = float(np.mean(null_means)) if null_means else float("nan")
    ns = float(np.std(null_means, ddof=1)) if len(null_means) > 1 else 0.0
    pct_lo, pct_hi = (float(np.percentile(null_means, 2.5)),
                      float(np.percentile(null_means, 97.5))) \
        if len(null_means) >= 2 else (float("nan"), float("nan"))
    real_mean = float(np.mean([x["pnl_pct"] for x in real])) if real \
        else float("nan")
    real_prem = float(np.mean([x["pnl_pct_prem"] for x in real])) if real \
        else float("nan")
    null_prem = float(np.mean(null_means_prem)) if null_means_prem \
        else float("nan")
    cost_prem = float(np.mean([x["cost_pct_prem"] for x in real])) if real \
        else float("nan")
    net_prem = real_prem - cost_prem

    # H2: c57 触碰后买入 (内联对拍: 触碰事件 → 24h 买跨式)
    c57_events = _c57_touch_events(df, ctx, bar_ms, start_ms, end_ms)
    c57_real = _c57_buy(bar_ms, o_arr, c57_events, cands, data)
    c57_prem = float(np.mean([x["pnl_pct_prem"] for x in c57_real])) \
        if c57_real else float("nan")
    c57_cost = float(np.mean([x["cost_pct_prem"] for x in c57_real])) \
        if c57_real else float("nan")

    gate(np.array(null_means_prem) if null_means_prem else np.array([0.0]))

    if dev:
        print("  [dev] 预置 {} (逼近 {} 高ER {}) 可用 {} | P&L {:+.6f}% "
              "({:+.3f}%) vs null {:+.6f}% ({:+.3f}%) | 持有 {:.1f}h 触碰退出 "
              "{:.0%} | c57 触碰买入 {:+.3f}%".format(
            raw_n, n_appr, n_er, len(real), real_mean, real_prem, nm,
            null_prem, avg_hold, touch_frac, c57_prem))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {"raw_n": raw_n, "n_appr": n_appr, "n_er": n_er, "eff": len(real),
           "avg_hold": avg_hold, "touch_frac": touch_frac,
           "h1": {"real": real_mean, "real_prem": real_prem, "n": len(real),
                  "null": (nm, ns), "null_95": (pct_lo, pct_hi),
                  "null_prem": null_prem},
           "cost_prem": cost_prem, "net_prem": net_prem,
           "c57": {"real_prem": c57_prem, "cost": c57_cost,
                   "net": c57_prem - c57_cost, "n": len(c57_real)}}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


# ── c57 触碰对拍 (内联) ──────────────────────────────────────
def _c57_touch_events(df, ctx, bar_ms, start_ms, end_ms):
    c = ctx.close
    ma = pd.Series(c).rolling(20).mean().values
    sd = pd.Series(c).rolling(20).std().values
    lo, hi = ma - 2 * sd, ma + 2 * sd
    c_prev = np.roll(c, 1)
    up = np.zeros(len(c), bool)
    dn = np.zeros(len(c), bool)
    for t in range(1, len(c)):
        if np.isfinite(hi[t]) and np.isfinite(lo[t]):
            if c[t] > hi[t] and c_prev[t] <= hi[t]:
                up[t] = True
            if c[t] < lo[t] and c_prev[t] >= lo[t]:
                dn[t] = True
    ev = np.flatnonzero((up | dn) & (bar_ms >= start_ms) & (bar_ms <= end_ms))
    return [int(t) for t in ev if t + 1 < len(c)]


def _c57_buy(bar_ms, o_arr, events, cands, data):
    out = []
    for t in events:
        tau = int(bar_ms[t + 1])
        spot = float(o_arr[t + 1])
        exit_ms = tau + 24 * 3600 * 1000
        r = straddle_buy("BTC", spot, tau, exit_ms, cands, data)
        if r is not None:
            out.append(r)
    return out


if __name__ == "__main__":
    sys.exit(main())
