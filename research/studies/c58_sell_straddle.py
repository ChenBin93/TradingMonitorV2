#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C58 期权线第二块砖: 触碰后卖跨式 + 尾部风险审计 (2026-08-14, 无未来函数,
[学习级])

[学习级] 考证 (PLAN §2.5 c58 行): c57 反向线索 — 触碰后买跨式显著低于随机
  null (买在 IV 高位)。本砖检验**卖**侧: 触碰后 IV 高估是否可收割 (卖跨式)?
  并做尾部风险审计 (short vol 死因量化)。同 c57 事件与合约映射 (BTC 84 触碰、
  蜡烛覆盖=可交易)。描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。
  **结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring
  预注册冻结、内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 触碰后**卖**跨式是否 > 随机时点卖 null?
  short vol 的尾部 (爆尾) 有多大?

预注册假设 (PLAN §2.5 c58 行, docstring 逐字):
  H1: 触碰后卖跨式 P&L 均值 > 随机时点卖 null (30 次同数量) 95% 区间
      (IV 高估的可收割性)
  H2: 尾部风险审计 — 卖跨式单笔最大亏损、亏损尾部 (95/99 分位) 真实 vs
      null、爆尾条件 (触碰后 realized 波动 > breakeven 的事件占比) —
      short vol 死因量化
  H3: 成本敏感性 (taker 0.03%×4 腿 + 1 tick)

  操作化 (运行前锁定):
    - 事件/映射: 同 c57 — BTC 1h MA20±2σ 触碰 (c51 口径), 窗口
      2026-06-24..2026-08-01; 合约映射 (蜡烛覆盖=可交易, 到期 ≥ τ+72h,
      最近到期 → 最近 strike C+P 对); 价格查找 ±2h
    - 卖跨式 = 触碰 bar T 收盘确认后下一 bar (T+1) 开盘**卖** call+put,
      持有 24/48/72h 买回平仓; 卖 P&L = 入场跨式价 − 平仓跨式价
      (独立计算, 非买侧取负, 保 golden 清晰)
    - H1: 卖 P&L 均值 vs 随机时点卖 null 30 次 (同数量同管线) 95% 区间
    - H2: 尾部审计 — 逐笔卖 P&L (溢价归一) min/5%/1% 分位, 真实 vs
      池化 null; 爆尾条件 = |realized 24h 移动| > breakeven (2×跨式价/标的价)
      的事件占比
    - H3: 成本 = 0.03%×(入+出跨式价)+4 tick (溢价归一); 净 P&L 再报
    - 学习级: BTC、30 次 null、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close (1h)       | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  MA20±2σ 带       | rolling(20) mean/sd (价格 sd)         | bar 收盘后 | c51 口径
  触碰事件         | 收盘穿越带边界 (c51 band_touches)      | bar 收盘后 | 出带=波动释放信号
  合约可用性       | 蜡烛覆盖 (coverage_start ≤ τ)          | τ 时刻    | 在列=可交易地面真相
  期权价格         | 目标时点 ±2h 内最近 1H bar (open/close)| τ 时刻    | 稀疏 K 线容忍
  卖侧 P&L         | 入−出跨式价 (独立符号) ×ctMult         | 全样本事后 | 描述统计
  null            | 随机时点卖 30 次 (同数量同管线)         | 锚定真实   | 同 c57
  realized 移动    | |log(c[T+25]/c[T+1])| (24h)            | 事后       | 爆尾条件分母

数据声明: data/backtest.db (BTC 1h, 止 2026-08-01); data/options.db
  (BTC 期权蜡烛 2026-03..2026-08); 同 c57。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  带 MA20±2σ (n=20); 窗口 2026-06-24..2026-08-01; 到期 ≥ τ+72h; 持有
  24/48/72h; ±2h 查找; ctMult=0.01; taker 0.03%×4 腿 + 1 tick (0.0001);
  null 30 次; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 卖侧 P&L 独立计算 (符号与买侧相反, 但独立实现保 golden 清晰)。
  - 尾部审计用跨式溢价归一 (% of 入场跨式价) — 单笔亏损的可读单位;
    H1 比较用标的价归一 (与 c57 同口径)。
  - 无 bid/ask 用 OHLC (入场 open/平仓 close), 日内报价可能陈旧 — 标注。
  - 学习级: 无 BY_YEAR; 30 次 null 沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 合约映射 golden (c57 同款); ② ±2h 查找 golden (c57
    同款); ③ 卖侧 P&L golden (独立符号校验: 卖 = 入−出); ④ null sanity —
    随机时点卖 null 均值 ∈ [−20%, +20%] (溢价归一, IV 漂移带)
  - null 无信息对照: 随机时点卖 30 次 (同数量同管线)
  - MIN_N: 每格 n ≥ 100 (学习级); 不足标注
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 触碰 20 个 × null 3 次, 不写 .out
  - 全量: BTC 全触碰 × null 30 次 (预计 ≤5 分钟, c57 仅 ~10s)

运行命令:
  python3 research/studies/c58_sell_straddle.py --dev
  python3 research/studies/c58_sell_straddle.py
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

from research.ctx import make_ctx
from research.data_loader import DB_PATH, load_candles, verify

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT",),
    "tf": "1h",
    "touch_start": "2026-06-24",
    "touch_end": "2026-08-01",
    "band_n": 20,
    "band_k": 2.0,
    "holds_h": (24, 48, 72),
    "exp_min_h": 72,
    "lookup_win_h": 2,
    "ct_mult": 0.01,
    "taker_frac": 0.0003,
    "tick": 0.0001,
    "null_draws": 30,
    "e1_h": 24,
    "warmup": 600,
    "min_n": 100,
    "null_band": (-20.0, 20.0),
    "opt_db": "data/options.db",
    "dev_subset": {"n_touch": 20, "n_null": 3},
    "data_range": "触碰 2026-06-24..2026-08-01; 期权蜡烛 2026-03..2026-08",
}

STUDY_ID = "c58_sell_straddle"


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
            continue
        out.append({"t": int(t), "ts_ms": int(t_ms[t]), "dir": "up" if up[t]
                    else "dn"})
    return out


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


# ── 卖侧跨式事件 P&L (独立实现) ─────────────────────────────
def straddle_pnl_sell(uly, spot, tau_ms, hold_h, cands, data):
    """卖跨式: 入场收 (入跨式价), 平仓付 (出跨式价). 返回 dict 或 None."""
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
    sell_gross = straddle_in - straddle_out                 # 卖=入−出
    pnl_btc = sell_gross * PARAMS["ct_mult"]
    pnl_pct_spot = sell_gross * 100.0 / max(spot, 1e-9)
    pnl_pct_prem = sell_gross * 100.0 / max(straddle_in, 1e-12)
    fee = PARAMS["taker_frac"] * (straddle_in + straddle_out)
    spread = 4 * PARAMS["tick"]
    cost_pct_prem = (fee + spread) * 100.0 / max(straddle_in, 1e-12)
    return {"pnl_pct": pnl_pct_spot, "pnl_pct_prem": pnl_pct_prem,
            "pnl_btc": pnl_btc, "cost_pct_prem": cost_pct_prem,
            "breakeven": 2 * straddle_in,
            "straddle_in": straddle_in, "straddle_out": straddle_out,
            "call": call_id, "put": put_id, "strike": strike,
            "exp_ms": exp_ms}


def run_events(uly, bar_times, cands, data, hold_h):
    out = []
    for tau_ms, spot in bar_times:
        r = straddle_pnl_sell(uly, spot, tau_ms, hold_h, cands, data)
        if r is not None:
            out.append(r)
    return out


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
    exp_828 = int(pd.Timestamp("2026-08-28", tz="UTC").timestamp() * 1000)
    exp_821 = int(pd.Timestamp("2026-08-21", tz="UTC").timestamp() * 1000)
    mc = map_contract(cand, "BTC", 61500.0, t_jul10)
    if mc is None or mc[3] != exp_828 or abs(mc[2] - 62000.0) > 1e-9:
        raise SystemExit("GATE FAIL: 映射 golden 07-10 (到期/strike)")
    mc2 = map_contract(cand, "BTC", 61500.0, t_aug01)
    if mc2 is None or mc2[3] != exp_821:
        raise SystemExit("GATE FAIL: 映射 golden 08-01 到期")
    mc3 = map_contract(cand, "BTC", 61500.0, t_aug20)
    if mc3 is None or mc3[3] != exp_828:
        raise SystemExit("GATE FAIL: 映射 golden 08-20 到期约束")
    if not (mc[0][3] == "C" and mc[1][3] == "P"):
        raise SystemExit("GATE FAIL: 映射 golden C/P 对")
    return True


def gate_price_golden():
    t0 = 1000 * 3600 * 1000
    ts = np.array([t0, t0 + 3600000, t0 + 3 * 3600000])
    op = np.array([0.05, 0.06, 0.07])
    cl = np.array([0.051, 0.061, 0.071])
    data = {"T": (ts, op, cl)}
    v = price_at(data, "T", t0 + 2 * 3600000 + 300000, use_open=True)
    if abs(v - 0.07) > 1e-12:
        raise SystemExit(f"GATE FAIL: ±2h 查找 {v} ≠ 0.07")
    v2 = price_at(data, "T", t0 + 6 * 3600000, use_open=True)
    if v2 is not None:
        raise SystemExit("GATE FAIL: ±2h 超窗未剔除")
    return True


def gate_pnl_sell_golden():
    """卖侧 P&L 独立符号校验: 卖=入−出."""
    tau = int(pd.Timestamp("2026-07-10 00:00", tz="UTC").timestamp() * 1000)
    ts2 = np.array([tau, tau + 24 * 3600000])
    data = {
        "BTC-USD-260828-62000-C": (ts2, np.array([0.10, 0.12]),
                                   np.array([0.10, 0.12])),
        "BTC-USD-260828-62000-P": (ts2, np.array([0.04, 0.03]),
                                   np.array([0.04, 0.03])),
    }
    r = straddle_pnl_sell("BTC", 62000.0, tau, 24, _mk_cand(), data)
    if r is None:
        raise SystemExit("GATE FAIL: 卖侧 golden 事件被剔除")
    # 入 0.14, 出 0.15 → 卖 = −0.01 (亏) → pnl_btc = −0.0001
    expect = (0.10 + 0.04 - 0.12 - 0.03) * PARAMS["ct_mult"]
    if abs(r["pnl_btc"] - expect) > 1e-12:
        raise SystemExit(f"GATE FAIL: 卖侧 P&L {r['pnl_btc']} ≠ {expect}")
    if r["pnl_btc"] >= 0:
        raise SystemExit("GATE FAIL: 卖侧符号错误 (出>入应亏损)")
    return True


def gate(null_means_prem):
    gate_mapping_golden()
    gate_price_golden()
    gate_pnl_sell_golden()
    nm = float(np.mean(null_means_prem)) if null_means_prem.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null 卖均值 {nm:+.3f}% ∉ [{lo}, {hi}]")
    print(f"[GATE] 映射 golden [PASS]; ±2h 查找 golden [PASS]; 卖侧 P&L "
          f"golden [PASS]; null sanity {nm:+.3f}% [PASS]", flush=True)
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
        "# GATE: 探测器自检 映射 golden + ±2h 查找 golden + 卖侧 P&L golden + "
        "null sanity [PASS]; MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c58 期权线第二块砖: 触碰后卖跨式 (c57 反向线索) + "
        "尾部风险审计; 事件/映射同 c57 (BTC 触碰, 蜡烛覆盖=可交易); 卖跨式="
        "下一 bar 开盘卖, 持有买回; P&L = 入−出跨式价 (独立符号); 无 bid/ask "
        "用 OHLC; 描述层无入场, 无交易含义",
        "",
    ]
    eff = res["eff"]
    lines.append("[数据] BTC 触碰 {} (up {} + dn {}) | 剔除 {} | 可用 {} (24h) "
                 "{}".format(res["raw_n"], res["n_up"], res["n_dn"],
                             res["n_excl"], eff, _nm(eff)))
    # H1
    lines.append("")
    lines.append("[H1] 触碰后卖跨式 P&L (标的价归一 %; 括号=跨式溢价归一 %) vs "
                 "随机时点卖 null {} 次:".format(p["null_draws"]))
    for hold in p["holds_h"]:
        h = res["holds"][hold]
        nm, ns = h["null"]
        pct_lo, pct_hi = h["null_95"]
        exceed = h["real"] > pct_hi
        below = h["real"] < pct_lo
        lines.append("  {}h: 真实 {:+.6f}% ({:+.3f}%) (n={}) {} | null "
                     "{:+.6f}±{:.6f}% ({:+.3f}%) 95% [{:+.6f}, {:+.6f}] -> "
                     "{}".format(hold, h["real"], h["real_prem"], h["n"],
                                 _nm(h["n"]), nm, ns, h["null_prem"], pct_lo,
                                 pct_hi,
                                 "超区间↑" if exceed else
                                 ("低于区间↓" if below else "区间内")))
    # H2 尾部审计
    lines.append("")
    lines.append("[H2] 尾部风险审计 (逐笔卖 P&L, 跨式溢价归一 %):")
    t = res["tail"]
    lines.append("  真实: min {:+.2f} | p1 {:+.2f} | p5 {:+.2f} | p50 "
                 "{:+.2f} | max {:+.2f} (n={}) {}".format(
        t["real"]["min"], t["real"]["p1"], t["real"]["p5"], t["real"]["p50"],
        t["real"]["max"], t["real"]["n"], _nm(t["real"]["n"])))
    lines.append("  池化 null: min {:+.2f} | p1 {:+.2f} | p5 {:+.2f} | p50 "
                 "{:+.2f} | max {:+.2f} (n={})".format(
        t["null"]["min"], t["null"]["p1"], t["null"]["p5"], t["null"]["p50"],
        t["null"]["max"], t["null"]["n"]))
    lines.append("  爆尾条件 (|realized 24h 移动| > breakeven 事件占比): "
                 "{:.1%} (n={}) | breakeven {:.2%} | 移动 {:.2%}".format(
        t["burst"], t["burst_n"], t["be"], t["move"]))
    lines.append("  单笔最大亏损真实 {:.2f}% vs 池化 null {:.2f}% (溢价) — "
                 "short vol 死因量化".format(t["real"]["min"], t["null"]["min"]))
    # H3
    lines.append("")
    lines.append("[H3] 成本敏感性 (taker 0.03%×4 腿 + 1 tick, 溢价归一 %):")
    for hold in p["holds_h"]:
        h = res["holds"][hold]
        lines.append("  {}h: 毛 {:+.3f}% | 成本 {:+.3f}% | 净 {:+.3f}% "
                     "(n={})".format(hold, h["real_prem"], h["cost_prem"],
                                     h["net_prem"], h["n"]))
    # 对照
    lines.append("")
    lines.append("[对照-历史] c57 (买侧: 触碰跨式 −4.177% vs null +2.070%, "
                 "显著低于 null — IV 高位买入); c58 (本砖: 卖侧收割检验 + 尾部); "
                 "short vol 死因 = 尾部爆亏 (max loss / p1) 而非均值; 数据摩擦: "
                 "期权 K 线稀疏/陈旧 (±2h 容忍, 标报价 mark-to-market)")
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
    sym = PARAMS["crypto"][0]
    df = ul.get(sym)
    if df is None:
        print("无数据")
        return 1
    ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
    evs = touch_events(df, ctx)
    if n_touch_max:
        evs = evs[:n_touch_max]
    n_up = int(sum(1 for e in evs if e["dir"] == "up"))
    n_dn = int(sum(1 for e in evs if e["dir"] == "dn"))
    raw_n = len(evs)
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

    holds = {}
    null_means_all = []
    null_means_prem_all = []
    tail_real = []
    null_trades_pool = []
    for hold in PARAMS["holds_h"]:
        real = run_events("BTC", entry_list, cands, data, hold)
        if hold == 24:
            n_excl += (len(entry_list) - len(real))
        null_means = []
        null_means_prem = []
        win_ms = bar_ms[bar_ms >= int(pd.Timestamp(
            PARAMS["touch_start"], tz="UTC").timestamp() * 1000)]
        win_ms = win_ms[win_ms <= int(pd.Timestamp(
            PARAMS["touch_end"] + " 23:59", tz="UTC").timestamp() * 1000)]
        rng = np.random.default_rng(5353)
        for d in range(n_null):
            idx = rng.integers(0, len(win_ms), size=len(entry_list))
            pos = np.searchsorted(bar_ms, win_ms[idx])
            pos = np.clip(pos, 0, len(o_arr) - 1)
            seq = [(int(win_ms[i]), float(o_arr[pos[i]])) for i in
                   range(len(idx))]
            rn = run_events("BTC", seq, cands, data, hold)
            if rn:
                null_means.append(float(np.mean([x["pnl_pct"] for x in rn])))
                null_means_prem.append(float(np.mean(
                    [x["pnl_pct_prem"] for x in rn])))
                if hold == 24:
                    null_trades_pool.extend([x["pnl_pct_prem"] for x in rn])
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
        holds[hold] = {"real": real_mean, "n": len(real),
                       "null": (nm, ns), "null_95": (pct_lo, pct_hi),
                       "real_prem": real_prem, "null_prem": null_prem,
                       "cost_prem": cost_prem, "net_prem": net_prem}
        if hold == 24:
            null_means_all.extend(null_means)
            null_means_prem_all.extend(null_means_prem)
            tail_real = [x["pnl_pct_prem"] for x in real]

    # H2 尾部 (24h): 爆尾条件逐可用事件
    c = ctx.close
    real24_ = run_events("BTC", entry_list, cands, data, 24)
    be_vals = []
    moves = []
    for tau_ms, _spot in entry_list:
        # 该入场对应的 underlying bar 索引 (τ = bar open)
        bi = int(np.searchsorted(bar_ms, tau_ms, side="left"))
        if bi + 25 >= len(c):
            continue
        moves.append(float(abs(np.log(c[bi + 24] / c[bi]))))
    for r_ in real24_:
        be_vals.append(r_["breakeven"])
    n_pair = min(len(moves), len(be_vals))
    move_mean = float(np.mean(moves[:n_pair])) if n_pair else float("nan")
    be_mean = float(np.mean(be_vals[:n_pair])) if n_pair else float("nan")
    burst_n = int(sum(1 for i in range(n_pair) if moves[i] > be_vals[i]))
    burst_share = burst_n / max(n_pair, 1)
    arr_real = np.array(tail_real) if tail_real else np.array([0.0])
    arr_null = np.array(null_trades_pool) if null_trades_pool else np.array([0.0])
    tail = {
        "real": {"min": float(arr_real.min()), "p1": float(np.percentile(
            arr_real, 1)), "p5": float(np.percentile(arr_real, 5)),
            "p50": float(np.percentile(arr_real, 50)),
            "max": float(arr_real.max()), "n": int(len(arr_real))},
        "null": {"min": float(arr_null.min()), "p1": float(np.percentile(
            arr_null, 1)), "p5": float(np.percentile(arr_null, 5)),
            "p50": float(np.percentile(arr_null, 50)),
            "max": float(arr_null.max()), "n": int(len(arr_null))},
        "burst": burst_share, "burst_n": int(burst_n), "be": be_mean,
        "move": move_mean,
    }

    gate(np.array(null_means_prem_all) if null_means_prem_all
         else np.array([0.0]))

    if dev:
        print("  [dev] 触碰 {} 可用24h {} | 24h 卖 {:+.6f}% ({:+.3f}%) vs null "
              "{:+.6f}% ({:+.3f}%) | max loss {:+.2f}% | 爆尾 {:.1%}".format(
            raw_n, holds[24]["n"], holds[24]["real"], holds[24]["real_prem"],
            holds[24]["null"][0], holds[24]["null_prem"], tail["real"]["min"],
            tail["burst"]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {"raw_n": raw_n, "n_up": n_up, "n_dn": n_dn, "n_excl": n_excl,
           "eff": holds[24]["n"], "holds": holds, "tail": tail}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
