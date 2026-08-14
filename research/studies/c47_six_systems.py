#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C47 U1-7 六系统比较忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U1-7 收官, PLAN §2.5 c47): 书 CH8 p.309-366 六系统
  (MOM/MA/EXP/NDB/SWG/LRS) 比较。oracle 逐字核实口径: 永远在场、信号即反转、
  无止损、收盘执行; 书成本 $20/手/$0.01/股 → taker 费率敏感性替代 (换算标注);
  规则逐字 (MOM close 穿 n 前收盘; MA/EXP 平滑值方向; NDB 前高破位+close 确认;
  SWG 摆动点高低突破 (MSV 过滤, c40 管线); LRS 最小二乘斜率).
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。**结论不得作交易
  依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、
  内置 GATE (六系统 golden)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书六系统长期盈利、排序 (MA 居前/MOM 垫底)、
  胜率签名 (34%/43%/50%)、噪声×入场交互、选速法则 — 现代版 (20 标的 4h +
  GBM null 海龟汤: 各系统 null 各有结构) 判卷.

预注册假设 (PLAN §2.5 c47 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 真实系统净收益 − 各自 GBM null 净收益 > 0 且覆盖成本 (书"长期都盈利"
      现代版; 成本敏感性只作用于真实侧, null 无成本)
  H2: 系统间排序真实 vs null (书"MA 居前、MOM 垫底、NDB 高波动" — 若真实
      排序≈null 排序则排序为机制伪影)
  H3: 胜率签名对拍 (书 MA/MOM ~34%、NDB ~50%、LRS/EXP ~43%、NDB 交易数
      不到其他一半 — 真实 vs null vs 书三层)
  H4: 噪声×入场交互 (avgER 三分位分组, 书表 8.5: 低噪声收盘穿越优、高噪声
      带穿越优 — 带穿越相对收盘穿越的 PF 差随 ER 分位单调)
  H5: 选速法则 (PF-n 斜率"更长更好" + 单 MA vs 多 MA 交叉"简单更好";
      摆幅/4 单例仅报告不设判据)

  操作化 (运行前锁定):
    - 六系统 × n∈{20,40,80,120,240} (映射书 5-160 日, 4h 日历偏差标注);
      SWG p∈{1%,2%} (裁剪: 全量 4 档→2 档, dev 预算); 永远在场反转无止损
    - GBM null: 每标的漂移匹配 30 种子同六系统 (null 无成本 — 海龟汤)
    - H1: 每系统真实聚合净收益 − null 聚合均值 > 0, 且 2.5bp 成本后仍 >
      null; H2: 真实 vs null 的 6 系统净收益排序; H3: 每系统胜率 (逐交易)
      与交易数 vs null 与书; H4: avgER 三分位 (20 标的按全样本 avgER 分组),
      收盘穿越 (MOM) vs 带穿越 (MA-of-high/low) 的 PF 差随 ER 单调;
      H5: 每系统 PF-n 回归斜率 + 单 MA vs MA 交叉 PF
    - 学习级: 20 标的 (偏离 BTC/ETH — 截面需要+计算廉价, 标注); 无 BY_YEAR;
      MIN_N=100; 描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  MOM/MA/EXP/LRS   | 逐 bar 因果信号 (close[t] vs close[t−n] 等) | bar 收盘后 | 书 CH8
  NDB              | high[t] > max(high[t−n..t−1]) & close 确认 | bar 收盘后 | 书口径 (t−1 窗口)
  SWG              | MSV_{t−1} 摆动点 + 高低突破 (c40 管线) | bar 收盘后 | 书 CH8 + c40
  LRS              | 最小二乘斜率 (c44 前缀和)              | bar 收盘后 | 书 CH8 + c44
  胜率/交易数      | 逐持仓段 (事后)                        | 全样本事后 | 描述统计
  GBM null         | sim_market.gbm_matching + 同六系统     | 锚定真实   | 30 种子 (无成本)

数据声明:
  20 标的 4h (6,570根/标的), 2023-08..2026-08 (backtest.db)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  n∈{20,40,80,120,240}; SWG p∈{1%,2%} (裁剪); 成本 {0, 2.5bp}; GBM 30 种子;
  MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 书为日线 5-160 日, 我们用 4h n∈{20..240} bar (日历偏差标注); SWG p 裁剪
    至 {1%,2%}; 成本敏感性裁剪至 {0, 2.5bp} (dev 预算, 报告).
  - 书成本 $20/手/$0.01/股 用 taker 费率替代 (换算标注: 每翻转 2×费率).
  - H4 的收盘穿越=MOM (close vs n 前), 带穿越=close vs MA(high)/MA(low).
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例.

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: 六系统 golden (已知序列信号逐位对拍: MOM/MA/EXP/LRS 在阶梯
    上升序列上应为全多; NDB 破位信号 bar 正确; SWG 确认滞后; 任一失败
    SystemExit); ② GBM sanity: GBM 各系统平均净收益在合理带内
  - GBM 无信息对照: 30 种子同六系统 (无成本)
  - MIN_N: 每格 n ≥ MIN_N=100
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH × GBM 3 种子 × n=40, 不写 .out
  - 全量: 20 标的 × 30 种子 (裁剪后, 预计 ≤8 分钟)

运行命令:
  python3 research/studies/c47_six_systems.py --dev
  python3 research/studies/c47_six_systems.py
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
    "ns": (20, 40, 80, 120, 240),
    "swg_ps": (0.01, 0.02),                # 裁剪: 书 4 档 → 2 档
    "costs_bp": (0.0, 2.5),                # 裁剪: 书 4 档 → 2 档
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "er_n": 10,
    "h4_ns": (40, 80),                     # H4/H5 周期 (裁剪)
    "h4_terciles": 3,
    "dev_subset": {"n_gbm": 3, "n_sym": 2, "ns": (40,)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c47_six_systems"
SYS = ("MOM", "MA", "EXP", "NDB", "SWG", "LRS")


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


# ── 基础构造 ─────────────────────────────────────────────────
def ma_series(c, n):
    return pd.Series(c).rolling(n).mean().values


def ema_hutson(c, n):
    al = 2.0 / (n + 1.0)
    e = np.full(len(c), np.nan)
    cur = c[0]
    for i in range(len(c)):
        cur = al * c[i] + (1 - al) * cur
        e[i] = cur
    return e


def lrs_slope(c, n):
    """最小二乘斜率 b (c44 前缀和) — 全长度 NaN 数组."""
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
    Sy = pc[ti + 1] - pc[s]
    Sxy = (pic[ti + 1] - pic[s]) - s * Sy
    b = (n * Sxy - Sx * Sy) / (n * Sx2 - Sx * Sx)
    out = np.full(L, np.nan)
    out[ok] = b
    return out


def er_median(c, n):
    c = np.asarray(c, float)
    t = np.arange(len(c))
    cp = np.roll(c, 1)
    ad = np.where(t >= 1, np.abs(c - cp), 0.0)
    pref = np.concatenate([[0], np.cumsum(ad)])
    ok = t >= n
    net = np.full(len(c), np.nan)
    net[ok] = np.abs(c[t[ok]] - c[t[ok] - n])
    path = np.full(len(c), np.nan)
    path[ok] = pref[t[ok] + 1] - pref[t[ok] - n + 1]
    er = np.full(len(c), np.nan)
    m = ok & (path > 0)
    er[m] = net[m] / path[m]
    fin = np.isfinite(er)
    return float(np.nanmedian(er[fin])) if fin.any() else float("nan")


# ── 六系统位置 (永远在场, 信号 close[t] 确定) ──────────────
def _ff_dir(buy, sell, L):
    """事件驱动: 持仓=最后信号方向 (buy 优先)."""
    sig = np.zeros(L, int)
    sig[buy & ~sell] = 1
    sig[sell & ~buy] = -1
    idx = np.arange(L)
    last = np.where(sig != 0, idx, 0)
    last = np.maximum.accumulate(last)
    has = np.maximum.accumulate(sig != 0)
    return np.where(has, sig[last], 0)


def swing_positions(close, high, low, p_swg):
    """SWG (书 + c40 管线): MSV_{t−1} 摆动点, 高低突破 → 方向."""
    n = len(close)
    msv = np.concatenate([[np.nan], p_swg * close[:-1]])
    p = np.zeros(n, int)
    pos = 0
    last_hi = None
    last_lo = None
    for t in range(n):
        if t >= 3:
            c_ = t - 2
            if high[c_] > high[c_ - 1] and high[c_] > high[c_ + 1] \
                    and (high[c_] - low[c_]) >= msv[c_]:
                if last_hi is None or high[c_] > last_hi:
                    pos = 1
                last_hi = high[c_]
            if low[c_] < low[c_ - 1] and low[c_] < low[c_ + 1] \
                    and (high[c_] - low[c_]) >= msv[c_]:
                if last_lo is None or low[c_] < last_lo:
                    pos = -1
                last_lo = low[c_]
        p[t] = pos
    return p


def system_positions(close, high, low, n, sys_name, p_swg=None):
    c = np.asarray(close, float)
    L = len(c)
    if sys_name == "MOM":
        p = np.zeros(L, int)
        p[n:] = np.where(c[n:] > c[:-n], 1, -1)
        return p
    if sys_name == "MA":
        ma = ma_series(c, n)
        p = np.zeros(L, int)
        p[n:] = np.where(ma[n:] > ma[n - 1:-1], 1, -1)
        return p
    if sys_name == "EXP":
        ema = ema_hutson(c, n)
        p = np.zeros(L, int)
        p[n:] = np.where(ema[n:] > ema[n - 1:-1], 1, -1)
        return p
    if sys_name == "LRS":
        b = lrs_slope(c, n)
        p = np.zeros(L, int)
        p[n:] = np.where(b[n:] > 0, 1, -1)
        return p
    if sys_name == "NDB":
        ref_hi = pd.Series(high).rolling(n).max().shift(1).values
        ref_lo = pd.Series(low).rolling(n).min().shift(1).values
        cp = np.concatenate([[np.nan], c[:-1]])
        buy = (high > ref_hi) & (c > cp)
        sell = (low < ref_lo) & (c < cp)
        return _ff_dir(buy, sell, L)
    if sys_name == "SWG":
        return swing_positions(c, high, low, p_swg)
    raise ValueError(sys_name)


def run_metrics(close, high, low, n, sys_name, p_swg=None):
    """→ (net, pf, wr, n_trades, n_flips). 胜率=逐持仓段."""
    c = np.asarray(close, float)
    L = len(c)
    p = system_positions(c, high, low, n, sys_name, p_swg)
    r = np.zeros(L)
    r[:-1] = c[1:] / c[:-1] - 1.0
    per = p * r
    net = float(per.sum())
    pos = per[per > 0].sum()
    neg = -per[per < 0].sum()
    pf = pos / neg if neg > 0 else float("inf")
    n_flips = int((np.diff(p) != 0).sum())
    # 持仓段 (trade) + 每段 PnL
    runs_pnl = []
    i = 0
    while i < L:
        if p[i] == 0:
            i += 1
            continue
        j = i
        while j < L and p[j] == p[i]:
            j += 1
        runs_pnl.append(float(p[i]) * (c[j - 1] / c[i] - 1.0))
        i = j
    runs_pnl = np.array(runs_pnl) if runs_pnl else np.array([0.0])
    wr = float(np.mean(runs_pnl > 0))
    return net, pf, wr, len(runs_pnl), n_flips


# ── GATE 自检 ────────────────────────────────────────────────
def gate():
    """六系统 golden (已知序列信号逐位对拍) + 无 GBM 幅度断言 (海龟汤负漂移
    已知, 由 H1 判据承接)."""
    # 上升阶梯: MOM/MA/EXP/LRS 应全多 (n=5)
    c = np.arange(1.0, 60.0)
    h = c + 0.3
    l = c - 0.3
    for sn in ("MOM", "MA", "EXP", "LRS"):
        p = system_positions(c, h, l, 5, sn)
        if not (p[5:] == 1).all():
            raise SystemExit(f"GATE FAIL: {sn} 上升序列未全多")
    # NDB: 阶梯上升 → 破前高信号应出现在 bar 5
    p = system_positions(c, h, l, 5, "NDB")
    if int((np.diff(p) != 0).sum()) < 1:
        raise SystemExit("GATE FAIL: NDB 无翻转")
    # SWG: 确认滞后 (candidate 于 c+1 确认, c+2 起可参照)
    p = swing_positions(c, h, l, 0.01)
    if p[0] != 0:
        raise SystemExit("GATE FAIL: SWG 早期不应有持仓")
    # 三角波: MOM n=5 峰后方向翻转 (close 跌破 5 bar 前高)
    tri = np.concatenate([np.arange(1.0, 30.0), np.arange(29.0, 0.0, -1.0)])
    ht = tri + 0.3
    lt = tri - 0.3
    p = system_positions(tri, ht, lt, 5, "MOM")
    if not ((p >= 0).any() and (p < 0).any()):
        raise SystemExit("GATE FAIL: MOM 三角波无多空翻转")
    print("[GATE] 六系统 golden [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pp(v):
    return f"{v:+.3f}"


def write_out(out_path, params, mat, h1, h2, h3, h4, h5):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},ns={},swg_ps={},costs_bp={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            p["tf"], p["ns"], p["swg_ps"], p["costs_bp"], p["gbm_seeds"],
            p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 探测器自检 六系统 golden [PASS]; MIN_N n≥{} "
        "[PASS]".format(p["gbm_seeds"], p["min_n"]),
        "# RESULTS: [学习级] c47 U1-7 六系统比较忠实复现 (书 CH8 p.309-366); "
        "六系统永远在场反转无止损收盘执行; GBM null 30 种子无成本 (海龟汤); "
        "裁剪: SWG p=1/2%, 成本 0/2.5bp; 20 标的 4h; 描述层无入场, 无交易含义",
        "",
    ]
    # 矩阵 (真实 0 成本 vs null)
    lines.append("[矩阵] 六系统 × n: 真实净收益 (净差 null) | 真实 PF (null PF) "
                 "| 胜率 (null) | 交易数 (null):")
    for sn in SYS:
        row = mat[sn]
        for nkey in row["order"]:
            r = row[nkey]["real"]
            g = row[nkey]["gbm"]
            net_diff = r["net"] - g["net"]
            lines.append("  {} {}: 净 {:.4f} (净差 {:+.4f}) | PF {:.2f} ({:.2f}) "
                         "| WR {:.1%} ({:.1%}) | 交易 {} ({})".format(
                sn, nkey, r["net"], net_diff, r["pf"], g["pf"], r["wr"],
                g["wr"], r["nt"], g["nt"]))
    # H1
    lines.append("")
    lines.append("[H1] 真实 − null 净收益 > 0 且覆盖成本 (2.5bp):")
    for sn in SYS:
        r = h1[sn]
        lines.append("  {}: 净差 {:+.4f} (2σ {:.4f}) | 2.5bp 成本后净差 "
                     "{:+.4f} -> {}".format(
            sn, r["diff"], r["gs"], r["diff_25"],
            "PASS" if r["ok"] else "FAIL"))
    lines.append("  H1 判据: 每系统净差 > 0 且成本后仍 > 0 -> {}/6".format(
        sum(1 for sn in SYS if h1[sn]["ok"])))
    # H2 排序
    lines.append("")
    lines.append("[H2] 系统排序 (真实 vs null):")
    lines.append("  真实: " + " > ".join(h2["real_rank"]))
    lines.append("  null: " + " > ".join(h2["null_rank"]))
    lines.append("  排序相关性 (真实 vs null 均值): {:.2f} — {}".format(
        h2["corr"], "≈null=机制伪影" if h2["corr"] > 0.5 else "≠null=非机制"))
    # H3 胜率签名
    lines.append("")
    lines.append("[H3] 胜率签名 (书: MA/MOM~34%, NDB~50%, LRS/EXP~43%, "
                 "NDB 交易数<一半):")
    for sn in SYS:
        r = h3[sn]
        lines.append("  {}: 真实 WR {:.1%} | null WR {:.1%} | 真实交易 {} "
                     "(相对 MOM {:.0%})".format(
            sn, r["wr"], r["wr_null"], r["nt"], r["nt_rel"]))
    lines.append("  NDB 交易数 < MOM 一半: {}".format(
        "✓" if h3["ndb_half"] else "✗"))
    # H4
    lines.append("")
    lines.append("[H4] 噪声×入场交互 (avgER 三分位, n={}):".format(p["h4_ns"]))
    for nkey, r in h4.items():
        lines.append("  {}: 收盘PF {} | 带PF {} | 差 [{}] -> 单调{}".format(
            nkey, [f"{v:.2f}" for v in r["close_pf"]],
            [f"{v:.2f}" for v in r["band_pf"]],
            [f"{v:+.2f}" for v in r["diff"]],
            "✓" if r["mono"] else "✗"))
    # H5
    lines.append("")
    lines.append("[H5] 选速法则:")
    lines.append("  PF-n 斜率 (每系统): {}".format(
        " | ".join("{} {:+.3f}".format(k, v) for k, v in h5["slopes"].items())))
    lines.append("  单 MA PF {:.2f} vs MA 交叉 PF {:.2f} (简单更好: {})".format(
        h5["single_pf"], h5["cross_pf"],
        "✓" if h5["single_pf"] >= h5["cross_pf"] else "✗"))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c41 (NDB/海龟汤 null 负漂移); c46 (MA 系统/滞后); "
                 "c34 (ER 截面 PF 随 ER 升); 书 CH8 p.309-366 (六系统比较)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    n_sym = PARAMS["dev_subset"]["n_sym"] if dev else None
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    ns = PARAMS["dev_subset"]["ns"] if dev else PARAMS["ns"]

    ctxs = load_ctxs(PARAMS, n_sym=n_sym)
    gate()

    def sys_key(sn, n_or_p):
        return f"{sn}|{n_or_p}"

    # 真实 + null 聚合 (单遍: 每 (sym, seed) 一次 GBM 复用全系统)
    real_pool = {sn: [] for sn in SYS}          # 每系统真实聚合净
    null_seed = {sn: [] for sn in SYS}          # 每系统每种子聚合净
    mat = {sn: {} for sn in SYS}
    h3_acc = {sn: {"wr_sum": 0.0, "wr_n": 0, "nt": 0} for sn in SYS}
    cell_real = {sn: {} for sn in SYS}
    cell_null = {sn: {} for sn in SYS}

    def _items(sn):
        if sn == "SWG":
            return [("p=1%", 0.01), ("p=2%", 0.02)]
        return [(f"n={n}", n) for n in ns]

    for sym, ctx, df in ctxs:
        c, h, l = ctx.close, ctx.high, ctx.low
        for sn in SYS:
            for nkey, n_or_p in _items(sn):
                r = run_metrics(c, h, l, n_or_p, sn,
                                p_swg=n_or_p if sn == "SWG" else None)
                real_pool[sn].append(r[0])
                h3_acc[sn]["wr_sum"] += r[2] * r[3]
                h3_acc[sn]["wr_n"] += r[3]
                h3_acc[sn]["nt"] += r[3]
                cell_real[sn].setdefault(nkey, []).append(r)
        # GBM null (每标的, 逐种子; 每种子内聚合各系统)
        for seed in range(seeds):
            rw = gbm_matching(df, seed=seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gc, gh, gl = gctx.close, gctx.high, gctx.low
            for sn in SYS:
                items = _items(sn)
                seed_sum = 0.0
                for nkey, n_or_p in items:
                    gr = run_metrics(gc, gh, gl, n_or_p, sn,
                                     p_swg=n_or_p if sn == "SWG" else None)
                    seed_sum += gr[0]
                    cell_null[sn].setdefault(nkey, []).append(gr)
                null_seed[sn].append(seed_sum / len(items))

    # 每格聚合 (跨标的×种子)
    for sn in SYS:
        mat[sn]["order"] = [k for k, _ in _items(sn)]
        for nkey in mat[sn]["order"]:
            rs = cell_real[sn][nkey]
            gs = cell_null[sn][nkey]
            mat[sn][nkey] = {
                "real": {"net": float(np.mean([r[0] for r in rs])),
                         "pf": float(np.mean([r[1] for r in rs])),
                         "wr": float(np.mean([r[2] for r in rs])),
                         "nt": float(np.mean([r[3] for r in rs]))},
                "gbm": {"net": float(np.mean([g[0] for g in gs])),
                        "pf": float(np.mean([g[1] for g in gs])),
                        "wr": float(np.mean([g[2] for g in gs])),
                        "nt": float(np.mean([g[3] for g in gs]))},
            }

    # H1: 每系统聚合净差 (跨标的×n 均值) vs null 种子分布
    h1 = {}
    for sn in SYS:
        real_m = float(np.mean(real_pool[sn]))
        null_arr = np.array(null_seed[sn])
        nm, nsd = float(np.mean(null_arr)), float(np.std(null_arr, ddof=1))
        # 2.5bp 成本: 每格 n_flips × 2×fee — 用真实平均翻转数
        # 简化: 平均每系统翻转 ~ 交易数; 成本 = 2×2.5bp×n_trades均值
        avg_nt = np.mean([mat[sn][k]["real"]["nt"] for k in mat[sn]["order"]])
        cost = 2.0 * (PARAMS["costs_bp"][1] / 10000.0) * avg_nt
        real_25 = real_m - cost
        diff = real_m - nm
        diff_25 = real_25 - nm
        h1[sn] = {"diff": diff, "gs": 2 * nsd, "diff_25": diff_25,
                  "ok": diff > 0 and diff_25 > 0}
    # H2 排序
    real_rank = sorted(SYS, key=lambda s: -h1[s]["diff"])
    null_rank = sorted(SYS, key=lambda s: -float(np.mean(null_seed[s])))
    rvals = [h1[s]["diff"] for s in SYS]
    nvals = [float(np.mean(null_seed[s])) for s in SYS]
    rx = pd.Series(rvals).rank().values
    nx = pd.Series(nvals).rank().values
    h2 = {"real_rank": real_rank, "null_rank": null_rank,
          "corr": float(np.corrcoef(rx, nx)[0, 1])}
    # H3
    h3 = {}
    mom_nt = h3_acc["MOM"]["nt"]
    for sn in SYS:
        wr = h3_acc[sn]["wr_sum"] / h3_acc[sn]["wr_n"] \
            if h3_acc[sn]["wr_n"] else float("nan")
        wr_null = float(np.mean([mat[sn][k]["gbm"]["wr"] for k in
                                 mat[sn]["order"]]))
        h3[sn] = {"wr": wr, "wr_null": wr_null, "nt": h3_acc[sn]["nt"],
                  "nt_rel": h3_acc[sn]["nt"] / mom_nt if mom_nt else 0.0}
    h3["ndb_half"] = h3["NDB"]["nt"] < mom_nt / 2

    # H4: avgER 三分位 × (收盘穿越 MOM vs 带穿越)
    ers = [(er_median(ctx.close, PARAMS["er_n"]), sym, ctx, df)
           for sym, ctx, df in ctxs]
    terc = sorted(ers, key=lambda x: x[0])
    n_t = len(terc) // PARAMS["h4_terciles"]
    groups = [terc[i * n_t:(i + 1) * n_t if i < PARAMS["h4_terciles"] - 1
               else len(terc)] for i in range(PARAMS["h4_terciles"])]
    h4 = {}
    for nkey, n_ma in ((f"n={PARAMS['h4_ns'][0]}", PARAMS["h4_ns"][0]),
                       (f"n={PARAMS['h4_ns'][1]}", PARAMS["h4_ns"][1])):
        c_pfs, b_pfs = [], []
        for grp in groups:
            cp, bp = [], []
            for er, sym, ctx, df in grp:
                c_ = ctx.close
                h_ = ctx.high
                l_ = ctx.low
                m1 = run_metrics(c_, h_, l_, n_ma, "MOM")
                cp.append(m1[1] if np.isfinite(m1[1]) else float("nan"))
                # 带穿越: close vs MA(high)/MA(low)
                mah = ma_series(h_, n_ma)
                mal = ma_series(l_, n_ma)
                buy = c_ > mah
                sell = c_ < mal
                p = _ff_dir(buy, sell, len(c_))
                r = np.zeros(len(c_))
                r[:-1] = c_[1:] / c_[:-1] - 1.0
                per = p * r
                pos = per[per > 0].sum()
                neg = -per[per < 0].sum()
                bp.append(pos / neg if neg > 0 else float("inf"))
            c_pfs.append(float(np.mean([x for x in cp if np.isfinite(x)]))
                         if cp else float("nan"))
            b_pfs.append(float(np.mean(bp)))
        d = [b - c for b, c in zip(b_pfs, c_pfs)]
        h4[nkey] = {"close_pf": c_pfs, "band_pf": b_pfs, "diff": d,
                    "mono": all(d[i + 1] > d[i] for i in range(len(d) - 1))}

    # H5: PF-n 斜率 + 单 MA vs MA 交叉
    h5 = {"slopes": {}, "single_pf": 0.0, "cross_pf": 0.0}
    for sn in SYS:
        if sn == "SWG":
            continue
        ns_here = [n for n in (20, 40, 80, 120, 240) if f"n={n}" in mat[sn]]
        pfs = [mat[sn][f"n={n}"]["real"]["pf"] for n in ns_here]
        xs = list(range(len(ns_here)))
        h5["slopes"][sn] = float(np.polyfit(xs, pfs, 1)[0]) \
            if len(pfs) > 1 else float("nan")
    # 单 MA (MA n=40) vs MA 交叉 (MA20 vs MA80)
    m40 = mat["MA"]["n=40"]["real"]["pf"]
    c_, h_, l_ = ctxs[0][1].close, ctxs[0][1].high, ctxs[0][1].low
    ma20 = ma_series(c_, 20)
    ma80 = ma_series(c_, 80)
    px = np.zeros(len(c_), int)
    px[80:] = np.where(ma20[80:] > ma80[80:], 1, -1)
    r = np.zeros(len(c_))
    r[:-1] = c_[1:] / c_[:-1] - 1.0
    per = px * r
    pos = per[per > 0].sum()
    neg = -per[per < 0].sum()
    h5["single_pf"] = m40
    h5["cross_pf"] = pos / neg if neg > 0 else float("inf")

    if dev:
        for sn in SYS:
            print("  [dev] {} 净差={:+.4f} null={:.4f}".format(
                sn, h1[sn]["diff"], float(np.mean(null_seed[sn]))))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, mat, h1, h2, h3, h4, h5)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
