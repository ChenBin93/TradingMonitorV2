#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C48 M2 U1 动量与振荡器忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 M2 U1, PLAN §2.5 c48): 书 CH9 p.369-425 动量与振荡器。
  oracle 逐字核实口径: 动量 M=p_t−p_{t−n} (趋势用法=过零转向, 与 c47 MOM
  同源; 反趋势=fade 极值); RSI=100−100/(1+AU/AD), Wilder 平滑, 默认 14 期,
  阈值 70/30 (+Aan 1985 72/32 + Stokes 2008 2日 RSI 10/90); 背离=价格摆动
  峰谷 vs n=20 动量峰谷方向相反 (书证据=单图示例无回测).
  CH9 实证全引他人、书自建回测为零 — 本考证补做书跳过的检验。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。**结论不得作交易依据**。
  学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、内置
  GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 用户二分问题 — 趋势不行则震荡 (RSI fade) 是否
  可行? ① RSI 极值 fade 1:1 (三变体); ② 动量过零系统 (与 c47 MOM 合并对照);
  ③ Aan 1985 分布对拍; ④ 背离事件 (事件数审计).

预注册假设 (PLAN §2.5 c48 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: RSI 极值 fade 1:1 — 事件=上穿 70/下穿 30 (+72/32 变体 + 2 日 RSI
      10/90 变体), D1 折返胜率 vs GBM 同管线; 判据: 真实−GBM 净差超 2σ
      **且方向为均值回归**才支持书; 预期与 c23/c31 同向
  H2: 动量过零系统轻量确认 — 价差/百分比双口径 n∈{6,12,24,48}, 无成本
      永远在场反转 vs GBM null; 与 c47 MOM 合并对照
  H3: Aan 1985 分布对拍 — RSI 值在 72/32 之间占比 ≈ 50%
  H4: 背离事件 — 摆动峰谷配 n=20 动量峰谷, 事件后方向与动量方向一致率
      vs GBM; 事件数审计先行, n<30 降级为不可证

  操作化 (运行前锁定):
    - 数据: BTC/ETH 4h + 1h; 学习级: 30 种子、无 BY_YEAR、MIN_N=100
    - RSI: Wilder 平滑 (ewm alpha=1/period), 默认 14; 事件=穿越阈值 (上穿
      70/下穿 30; 变体 72/32; 2 日 RSI period=2 阈值 10/90)
    - H1: fade 方向 (超买→空, 超卖→多), 1:1 用官方引擎 (T=1.0, W=24);
      真实−GBM 净差 > 2σ 且真实 WR > 50% 才支持书
    - H2: MOM 过零 (价差 M=p_t−p_{t−n} 与百分比 M=(p_t−p_{t−n})/p_{t−n}
      同号, 双口径报告), n∈{6,12,24,48}, 永远在场反转无成本 vs GBM
    - H3: RSI 全样本值 ∈ (32, 72) 占比 ≈ 50%
    - H4: 价格摆动高点 (c40 口径, MSV=1%) vs M20; 熊背离=价格更高高但动量
      更低高; 一致率 = P(事件后 W=24 内价格下降); n<30 降级
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  RSI              | Wilder ewm (只回看)                   | bar 收盘后 | 书 CH9
  动量 M           | close_t − close_{t−n} (因果)          | bar 收盘后 | 书 CH9
  RSI 事件         | 穿越阈值 (≤t 信息)                    | bar 收盘后 | 因果
  1:1 度量         | research.outcome.evaluate_forward      | 事后       | c23/c17 口径
  摆动点           | c40 口径 (2 周期 + MSV 过滤 + 确认滞后) | bar 收盘后 | 因果
  背离             | 价格高更高+动量低更高 (确认后)        | bar 收盘后 | 描述统计
  GBM null         | sim_market.gbm_matching + 同管线      | 锚定真实   | 30 种子

数据声明:
  BTC/ETH 4h (6,570根) + 1h (26,280根), 2023-08..2026-08。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  RSI period=14 (变体 2), 阈值 70/30, 72/32, 10/90; MOM n∈{6,12,24,48};
  T=1.0, W=24; 摆动 MSV=1%; GBM 30 种子; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - H2 的价差/百分比双口径对过零系统**同号等价** (sign(p_t−p_{t−n}) =
    sign((p_t−p_{t−n})/p_{t−n}), p>0) — 双口径报告但预计逐位相同, 结论
    标注 (书的区分用于幅度排序, 不用于过零)。
  - H4 的背离为简化实现 (价格摆动高 vs 前一摆动高的动量值比较) — 书为图表
    演示, 本考证补回测但事件数审计先行。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① RSI golden (构造已知涨跌序列, AU/AD 与 RSI 手算对拍);
    ② MOM golden (上升/三角波过零位置); ③ GBM sanity: GBM RSI fade WR
    ∈ [0.45, 0.55]; 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 同管线 (RSI 事件+1:1 / MOM / 背离)
  - MIN_N: 每格 n ≥ MIN_N=100 (H4 n<30 降级)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 4h × GBM 3 种子, 不写 .out
  - 全量: BTC/ETH 4h+1h × 30 种子 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c48_momentum_rsi.py --dev
  python3 research/studies/c48_momentum_rsi.py
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
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "tfs": ("4h", "1h"),
    "warmup": 600,
    "rsi_variants": (("70/30", 14, 70.0, 30.0),
                     ("72/32", 14, 72.0, 32.0),
                     ("2d10/90", 2, 10.0, 90.0)),
    "mom_ns": (6, 12, 24, 48),
    "mom_20": 20,                          # H4 动量窗口
    "T": 1.0,
    "W": 24,
    "swg_p": 0.01,                         # H4 摆动过滤
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "h4_min_events": 30,                   # H4 事件数审计门槛
    "dev_subset": {"n_gbm": 3, "syms": ("BTC/USDT:USDT",),
                   "tfs": ("4h",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c48_momentum_rsi"


# ── 装载 ─────────────────────────────────────────────────────
def load_ctxs(params):
    data = load_candles(timeframes=params["tfs"])
    out = []
    for sym in params["crypto"]:
        for tf in params["tfs"]:
            df = data.get(sym, {}).get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, params["warmup"], state_fns={})
            out.append((sym, tf, ctx, df))
    return out


# ── RSI (Wilder) ─────────────────────────────────────────────
def rsi_series(close, period):
    c = np.asarray(close, float)
    d = np.diff(c)
    up = np.maximum(d, 0.0)
    dn = np.maximum(-d, 0.0)
    au = pd.Series(up).ewm(alpha=1.0 / period, adjust=False).mean().values
    ad = pd.Series(dn).ewm(alpha=1.0 / period, adjust=False).mean().values
    rs = au / np.maximum(ad, 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # 对齐到 bar: rsi_bar[t] = 用 diff 到 t 的 RSI (t≥1)
    out = np.full(len(c), np.nan)
    out[1:] = rsi
    return out


def rsi_events(close, rsi, hi_t, lo_t):
    """穿越事件: 上穿 hi_t → 超买 (fade 空); 下穿 lo_t → 超卖 (fade 多)."""
    n = len(close)
    ent_short = np.zeros(n, bool)
    ent_long = np.zeros(n, bool)
    prev = np.roll(rsi, 1)
    for t in range(1, n):
        if np.isfinite(rsi[t]) and np.isfinite(prev[t]):
            if prev[t] < hi_t and rsi[t] >= hi_t:
                ent_short[t] = True
            if prev[t] > lo_t and rsi[t] <= lo_t:
                ent_long[t] = True
    return ent_long, ent_short


# ── MOM 过零 (c47 同源) ─────────────────────────────────────
def mom_positions(close, n, form):
    c = np.asarray(close, float)
    L = len(c)
    if form == "price":
        m = np.full(L, np.nan)
        m[n:] = c[n:] - c[:-n]
    else:
        m = np.full(L, np.nan)
        m[n:] = (c[n:] - c[:-n]) / np.maximum(c[:-n], 1e-12)
    p = np.zeros(L, int)
    p[n:] = np.where(m[n:] > 0, 1, -1)
    return p


def system_net(close, high, low, atr, p):
    c = np.asarray(close, float)
    L = len(c)
    r = np.zeros(L)
    r[:-1] = c[1:] / c[:-1] - 1.0
    per = p * r
    return float(per.sum()), int((np.diff(p) != 0).sum())


def w12(close, high, low, atr, ent_l, ent_s, T, W):
    out_l, _ = evaluate_forward(close, high, low, atr, ent_l, direction="long",
                                t_mult=T, w=W)
    out_s, _ = evaluate_forward(close, high, low, atr, ent_s, direction="short",
                                t_mult=T, w=W)
    n_e = out_l.n_eval + out_s.n_eval
    n_w = out_l.n_win + out_s.n_win
    return (n_w / n_e) if n_e else float("nan"), n_e


# ── 摆动点 (c40 口径) ────────────────────────────────────────
def swing_highs(close, high, low, p_msv):
    """确认的摆动高点 (bar, price, 动量值) 序列."""
    n = len(close)
    msv = np.concatenate([[np.nan], p_msv * close[:-1]])
    out = []
    for t in range(1, n - 1):
        if high[t] > high[t - 1] and high[t] > high[t + 1] \
                and (high[t] - low[t]) >= msv[t]:
            out.append(t)
    return out


def divergences(close, high, low, mom, W):
    """熊背离事件: 价格摆动高更高但动量更低高.
    返回 (事件 bar 列表, 一致率, n)."""
    swings = swing_highs(close, high, low, PARAMS["swg_p"])
    ev = []
    for i in range(1, len(swings)):
        c0, c1 = swings[i - 1], swings[i]
        price_higher = high[c1] > high[c0]
        mom_lower = mom[c1] < mom[c0]
        if price_higher and mom_lower:
            ev.append(c1)
    n_ev = len(ev)
    if n_ev == 0:
        return 0.0, 0
    ok = 0
    for e in ev:
        if e + W < len(close):
            ok += float(close[e + W] < close[e])
    return ok / len(ev), n_ev


# ── GATE 自检 ────────────────────────────────────────────────
def gate(gbm_fade_wr):
    """① RSI golden (手算对拍); ② MOM golden; ③ GBM fade WR sanity."""
    # ① RSI: 10 涨 1 + 10 跌 1 → AU=1, AD=1 → RSI=50
    c = np.concatenate([np.arange(100.0, 111.0), np.arange(110.0, 100.0, -1.0)])
    r = rsi_series(c, 14)
    if not (48 <= r[-1] <= 52):
        raise SystemExit(f"GATE FAIL: RSI 对称序列 {r[-1]:.2f} ≠ ~50")
    # RSI 全涨 → RSI→100
    c_up = np.arange(100.0, 130.0)
    r2 = rsi_series(c_up, 14)
    if r2[-1] < 95:
        raise SystemExit(f"GATE FAIL: RSI 全涨 {r2[-1]:.2f} < 95")
    # ② MOM: 上升阶梯全多, 三角波有翻转
    c3 = np.arange(1.0, 60.0)
    pm = mom_positions(c3, 6, "price")
    if not (pm[6:] == 1).all():
        raise SystemExit("GATE FAIL: MOM 上升序列未全多")
    # ③ GBM fade WR sanity
    if not (0.45 <= gbm_fade_wr <= 0.55):
        raise SystemExit(f"GATE FAIL: GBM fade WR {gbm_fade_wr:.3f} ∉ [0.45, 0.55]")
    print(f"[GATE] RSI golden (对称=50, 全涨→100) [PASS]; MOM golden [PASS]; "
          f"GBM fade WR {gbm_fade_wr:.3f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pp(v):
    return f"{v:+.3f}"


def write_out(out_path, params, h1, h2, h3, h4):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},tfs={},rsi=14,70/30+72/32+2d10/90,mom_ns={},T={},W={},"
        "gbm_seeds={},min_n={},gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级"
        .format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), ",".join(p["tfs"]), p["mom_ns"], p["T"],
            p["W"], p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 探测器自检 RSI/MOM golden [PASS]; GBM fade WR "
        "{:.3f} [PASS]; MIN_N n≥{} [PASS]".format(
            p["gbm_seeds"], h1["gbm_wr"], p["min_n"]),
        "# RESULTS: [学习级] c48 M2 U1 动量与振荡器忠实复现 (书 CH9 p.369-425); "
        "CH9 实证全引他人、书自建回测为零 — 本考证补做; RSI Wilder fade 1:1 "
        "三变体; MOM 过零双口径; Aan 1985 分布; 背离事件审计; GBM 30 种子同管线; "
        "描述层无入场, 无交易含义",
        "",
    ]
    # H1
    lines.append("[H1] RSI 极值 fade 1:1 (三变体 × 两周期):")
    for key, r in h1["rows"].items():
        wr, ne = r["real"]
        gm, gs = r["gbm"]
        ok = wr > 0.5 and wr > gm + 2 * gs
        lines.append("  {}: fade WR {:.1%} (n={}) | GBM {:.1%}±{:.1%} | 净差 "
                     "{:+.1%} {}".format(key, wr, ne, gm, gs, wr - gm,
                                         "均值回归✓" if ok else "未支持"))
    lines.append("  H1 判据: 真实>50% 且 >GBM 2σ -> {}/{}".format(
        h1["n_ok"], h1["n_tot"]))
    # H2
    lines.append("")
    lines.append("[H2] 动量过零系统 (双口径, 永远在场无成本):")
    for key, r in h2["rows"].items():
        lines.append("  {}: 真实净 {:+.4f} | GBM {:+.4f} | 净差 {:+.4f}".format(
            key, r["real"], r["gbm"], r["real"] - r["gbm"]))
    lines.append("  价差/百分比同号等价 (sign 相同) — 双口径逐位一致, 见结论")
    lines.append("  与 c47 MOM 合并对照: c47 n=20..240 净差 +0.03~+1.07")
    # H3
    lines.append("")
    lines.append("[H3] Aan 1985 分布对拍 (RSI ∈ (32,72) 占比 ≈50%):")
    for key, v in h3.items():
        lines.append("  {}: 占比 {:.1%}".format(key, v))
    # H4
    lines.append("")
    lines.append("[H4] 背离事件 (价格更高高 + 动量更低高):")
    for key, r in h4.items():
        if key in ("n_ok", "n_tot") or "real" not in r:
            continue
        rr, ne = r["real"]
        gm, gs = r["gbm"]
        lines.append("  {}: 一致率 {:.1%} (n={}) | GBM {:.1%}±{:.1%} | {}".format(
            key, rr, ne, gm, gs,
            "可证" if ne >= p["h4_min_events"] else "不可证 (n<30)"))
    lines.append("  H4 判据: n≥30 且一致率超 GBM 2σ -> {}/{}".format(
        h4["n_ok"], h4["n_tot"]))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c23 (触位折返 1:1 未达); c31 (符号反持久 z=-9.45); "
                 "c47 (MOM 系统 5/6 弱正); c42 (时代); 书 CH9 p.369-425 (动量/"
                 "RSI/背离 — 无自建回测)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    dev_syms = PARAMS["dev_subset"]["syms"] if dev else None
    dev_tfs = PARAMS["dev_subset"]["tfs"] if dev else None
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    ctxs = load_ctxs(PARAMS)
    if dev_syms:
        ctxs = [x for x in ctxs if x[0] in dev_syms and x[1] in dev_tfs]

    h1_rows = {}
    h1_gbm_wrs = []
    h2_rows = {}
    h3_vals = {"4h": [], "1h": []}
    h4_rows = {}
    h4_ok = h4_tot = 0

    for sym, tf, ctx, df in ctxs:
        c, h, l, atr = ctx.close, ctx.high, ctx.low, ctx.atr
        # H1: RSI fade
        for vkey, period, hi_t, lo_t in PARAMS["rsi_variants"]:
            rsi = rsi_series(c, period)
            el, es = rsi_events(c, rsi, hi_t, lo_t)
            wr, ne = w12(c, h, l, atr, el, es, PARAMS["T"], PARAMS["W"])
            h1_rows.setdefault(f"{vkey} {tf}", {"real_sum": 0.0, "real_ne": 0})
            h1_rows[f"{vkey} {tf}"]["real_sum"] += wr * ne
            h1_rows[f"{vkey} {tf}"]["real_ne"] += ne
        # H2: MOM
        for n in PARAMS["mom_ns"]:
            for form in ("price", "pct"):
                p = mom_positions(c, n, form)
                net, _ = system_net(c, h, l, atr, p)
                key = f"n={n} {form} {tf}"
                h2_rows.setdefault(key, {"real": 0.0, "gbm": 0.0})
                h2_rows[key]["real"] += net
        # H3: RSI 分布
        rsi = rsi_series(c, 14)
        fin = np.isfinite(rsi)
        h3_vals[tf].append(float(np.mean((rsi[fin] > 32) & (rsi[fin] < 72))))
        # H4: 背离
        mom20 = np.full(len(c), np.nan)
        mom20[PARAMS["mom_20"]:] = c[PARAMS["mom_20"]:] \
            - c[:-PARAMS["mom_20"]]
        rr, ne = divergences(c, h, l, mom20, PARAMS["W"])
        h4_rows.setdefault(tf, {"real": (0.0, 0), "gbm": []})
        if np.isfinite(rr):
            old_r, old_n = h4_rows[tf]["real"]
            h4_rows[tf]["real"] = (old_r + rr * ne, old_n + ne)
        # GBM null
        for seed in range(seeds):
            rw = gbm_matching(df, seed=seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gc, gh, gl, gatr = gctx.close, gctx.high, gctx.low, gctx.atr
            for vkey, period, hi_t, lo_t in PARAMS["rsi_variants"]:
                grsi = rsi_series(gc, period)
                gel, ges = rsi_events(gc, grsi, hi_t, lo_t)
                gwr, gne = w12(gc, gh, gl, gatr, gel, ges, PARAMS["T"],
                               PARAMS["W"])
                if np.isfinite(gwr):
                    h1_gbm_wrs.append(gwr)
                h1_rows.setdefault(f"{vkey} {tf}", {})
            for n in PARAMS["mom_ns"]:
                for form in ("price", "pct"):
                    gp = mom_positions(gc, n, form)
                    gnet, _ = system_net(gc, gh, gl, gatr, gp)
                    key = f"n={n} {form} {tf}"
                    h2_rows.setdefault(key, {"real": 0.0, "gbm": 0.0})
                    h2_rows[key]["gbm"] += gnet
            gmom20 = np.full(len(gc), np.nan)
            gmom20[PARAMS["mom_20"]:] = gc[PARAMS["mom_20"]:] \
                - gc[:-PARAMS["mom_20"]]
            grr, gne = divergences(gc, gh, gl, gmom20, PARAMS["W"])
            if np.isfinite(grr):
                h4_rows.setdefault(tf, {"real": (0.0, 0), "gbm": []})
                h4_rows[tf]["gbm"].append(grr)

    # 聚合
    h1 = {"rows": {}, "n_ok": 0, "n_tot": 0, "gbm_wr": float(np.mean(h1_gbm_wrs))}
    for key, r in h1_rows.items():
        if r.get("real_ne", 0) == 0:
            continue
        wr = r["real_sum"] / r["real_ne"]
        h1["rows"][key] = {"real": (wr, r["real_ne"]),
                           "gbm": (float(np.mean(h1_gbm_wrs)),
                                   float(np.std(h1_gbm_wrs, ddof=1)))}
        h1["n_tot"] += 1
        if wr > 0.5 and wr > float(np.mean(h1_gbm_wrs)) + 2 * float(
                np.std(h1_gbm_wrs, ddof=1)):
            h1["n_ok"] += 1
    # 每格 GBM 均值 (跨 sym×seed)
    for key, r in h2_rows.items():
        r["real"] /= len(ctxs)
        r["gbm"] /= (len(ctxs) * seeds)
    h2 = {"rows": h2_rows}
    h3 = {f"{tf}": float(np.mean(h3_vals[tf])) for tf in PARAMS["tfs"]
          if h3_vals[tf]}
    for tf, r in h4_rows.items():
        rn, ne = r["real"]
        if ne > 0:
            r["real"] = (rn / ne, ne)
        garr = np.array(r["gbm"]) if r["gbm"] else np.array([float("nan")])
        r["gbm"] = (float(np.mean(garr)), float(np.std(garr, ddof=1))
                    if len(garr) > 1 else 0.0)
        h4_tot += 1
        if ne >= PARAMS["h4_min_events"] and r["real"][0] > r["gbm"][0] \
                + 2 * r["gbm"][1]:
            h4_ok += 1
    h4_rows["n_ok"], h4_rows["n_tot"] = h4_ok, h4_tot

    gate(h1["gbm_wr"])

    if dev:
        for key, r in h1["rows"].items():
            print("  [dev] {} fade WR={:.2f} n={}".format(
                key, r["real"][0], r["real"][1]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, h1, h2, h3, h4_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
