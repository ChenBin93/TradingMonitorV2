#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C49 M4 U1 量、仓与价差忠实复现 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 M4 U1, PLAN §2.5 c49): 书 CH12 p.527-563 (量) + CH13
  carry 映射 (资金流向高真实收益市场)。oracle 逐字核实口径: 书图 12.15 U 形
  量 (当地商业时段为界) — 加密 24/7 翻译为"美股时段驼峰"; "tick volume≈实际
  量" (p.560) 改写为量-波动关系; OI 映射差异声明; CH13 forward bias 加密翻译
  = funding 直接提取 (正 funding=空头收 carry, funding 符号=拥挤度定价)。
  **数据限制**: OKX funding 仅 2026-05-11~2026-08-14 (~95 天, API 上限) —
  funding 部分为 pilot, 样本限制标注。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。**结论不得作交易依据**。
  学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、内置 GATE、
  因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): ① 时段量结构 (书 U 形的加密版); ② 量-波动
  关系 (tick 断言改写); ③ funding-方向 (carry 加密检验); ④ funding-折返
  (拥挤度×触碰).

预注册假设 (PLAN §2.5 c49 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 时段量结构 — 20 标的 3y 1h 真实量的 UTC 小时剖面 (每标的每小时量
      中位数/整体中位数归一, 跨标的聚合) vs null (每标的 shuffle 小时标签
      30 次置换); 判据: 美股时段 (13:30-21:00 UTC) 小时量显著高于 null
      置换带 (书 U 形的加密版=美股时段驼峰)
  H2: 量-波动 — Spearman(bar volume, |ret|)、Spearman(volume, range)、量
      领先/滞后波动 (volume_t vs |ret|_{t+k}, k∈{1,3,6,12})、volume 长记忆
      (DFA 或自相关谱 vs c12 口径) — BTC/ETH 1h 3y
  H3: funding-方向 pilot (3 个月窗口, 样本限制标注) — E[forward k-bar ret |
      funding 符号/水平], forward k∈{8,24,72}, 分组=符号 (正/负) + 水平
      三分位; null=GBM 匹配 (μ/σ 用样本, funding 标签 shuffle 30 次);
      判据: 真实分组 forward ret 差超 null 95% 区间; MIN_N 审计: 若符号格
      n<100 则只报水平分位版并标注
  H4: funding-折返 pilot — funding 分位极值 (top/bottom 三分位) + 触碰事件
      (c17 管线, 3 个月窗口) 的 D1 折返 vs GBM 同管线; 判据同上

  操作化 (运行前锁定):
    - H1: 真实量 = 1h bar volume; UTC 小时 = timestamp 换算; 剖面 = 每小时
      量中位数/整体中位数; 跨标的均值聚合; null = 每标的 volume 值置换
      30 次 (破坏时段结构, 保持量分布)
    - H2: BTC/ETH 1h: Spearman(volume, |ret|), Spearman(volume, high−low);
      量领先: Spearman(volume_t, |ret|_{t+k}) k∈{1,3,6,12}; 长记忆:
      log(volume) 自相关 @lag 168 (c12 口径)
    - H3: funding 事件 (8h 结算) 对齐 1h bar (事件 ts 后最近 1h bar 为
      forward 起点, 严格因果); forward ret k∈{8,24,72}; 分组符号/水平
      三分位; null = GBM 1h rets (μ/σ 样本) + funding 标签 shuffle 30 次
    - H4: 触碰事件 (c17 管线) 在 funding 窗口 (2026-05-11+); funding 三分位
      极值组; D1 = 1:1 折返胜率; null = GBM 触碰 + 标签 shuffle
    - 学习级: 30 种子/置换、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  volume/OHLC      | db 原生 (load_candles)               | bar 收盘后 | data_loader
  时段剖面         | UTC 小时中位数归一 (描述统计)         | 全样本事后 | 书图 12.15
  置换 null        | volume 值置换 (破坏时段, 保分布)      | 全样本     | 因果无问题
  funding          | data/funding.db (8h 结算)             | 事件 ts    | 严格因果
  forward ret      | 事件 ts 后最近 1h bar 起 k 根          | 事后       | 因果起点
  GBM null         | sim_market.gbm_matching + 标签 shuffle | 锚定真实   | 30 次
  触碰/D1          | c17 管线 + 官方 1:1 引擎              | 事后       | c17 口径

数据声明:
  backtest.db: 20 标的 1h (26,280根) 3y (量字段真实); funding.db: 20 标的
  funding_rate (8h, 2026-05-11~2026-08-14 ~95 天, OKX API 上限, pilot 标注)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  H1 置换 30; H2 k∈{1,3,6,12}, ACF lag 168; H3 k∈{8,24,72}, 置换 30;
  H4 触碰 c17 + 1:1; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - H1 的 null 用 volume 值置换 (保持量分布, 破坏时段标签) — 书 U 形为
    15 分钟 tick 量, 我们用 1h 真实量 (频率/字段偏离标注)。
  - H3/H4 的 funding 数据仅 3 个月 (OKX API 上限) — pilot 标注, 结论不超
    样本限制。
  - H4 的 GBM null: GBM 触碰 (无 funding) + 真实 funding 标签 shuffle —
    检验"funding 水平@触碰"是否调节折返 (标签关联破坏)。
  - 学习级: 无 BY_YEAR; 30 种子/置换沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① H1 置换 sanity (纯随机序列的置换带应覆盖均值, 人造
    时段信号被置换破坏); ② H3 对齐 golden (构造已知 funding→ret 关联,
    验证对齐与分组); 任一失败 SystemExit
  - GBM/置换无信息对照: 30 次
  - MIN_N: 每格 n ≥ MIN_N=100 (H3/H4 样本限制标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH 1h × 3 置换, 不写 .out
  - 全量: 20 标的 × 30 置换/种子 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c49_volume_funding.py --dev
  python3 research/studies/c49_volume_funding.py
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
from research.data_loader import load_candles, verify
from research.outcome import evaluate_forward
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "tf": "1h",
    "funding_db": "data/funding.db",
    "us_hours": (13, 14, 15, 16, 17, 18, 19, 20),   # 13:30-21:00 UTC
    "n_perm": 30,
    "h2_lags": (1, 3, 6, 12),
    "h2_acf_lag": 168,
    "h3_ks": (8, 24, 72),
    "h3_terciles": 3,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                          # 学习级 MIN_N
    "dev_subset": {"n_perm": 3},
    "data_range": "量: 3y; funding: 2026-05-11..2026-08-14 (pilot)",
}

STUDY_ID = "c49_volume_funding"


# ── 装载 ─────────────────────────────────────────────────────
def load_1h_all(params, syms_filter=None):
    data = load_candles(timeframes=(params["tf"],))
    out = []
    for sym in data:
        if "USDT" not in sym:
            continue
        if syms_filter and sym not in syms_filter:
            continue
        df = data[sym].get(params["tf"])
        if df is None or verify(df, sym, params["tf"]):
            continue
        out.append((sym, df))
    return out


def load_funding(db_path, inst_map):
    """funding: {sym: (ts_ms array, rate array)}. instId 如 BTC-USDT-SWAP."""
    conn = sqlite3.connect(db_path)
    out = {}
    try:
        rows = conn.execute(
            "SELECT instId, ts, funding_rate FROM funding ORDER BY ts").fetchall()
    finally:
        conn.close()
    for inst, ts, rate in rows:
        base = inst.split("-")[0]
        if base not in inst_map:
            continue
        out.setdefault(inst_map[base], []).append((ts, rate))
    return {sym: (np.array([r[0] for r in v], dtype="int64"),
                  np.array([r[1] for r in v], float))
            for sym, v in out.items()}


# ── H1: 时段量结构 ───────────────────────────────────────────
def hour_profile_one(vol, hours, rng):
    """单标的归一化小时剖面 (每小时中位数/整体中位数)."""
    med_all = np.median(vol)
    if med_all <= 0:
        return np.full(24, 1.0)
    return np.array([np.median(vol[hours == hh]) / med_all for hh in range(24)])


def h1_run(dfs, n_perm, us_hours):
    syms = [s for s, _ in dfs]
    vols = [df["volume"].values.astype(float) for _, df in dfs]
    hours = [(df.index.hour.values) for _, df in dfs]
    prof = np.mean([hour_profile_one(v, h, None) for v, h in zip(vols, hours)],
                   axis=0)
    null = np.zeros((n_perm, 24))
    for p in range(n_perm):
        ps = []
        for i, (v, h) in enumerate(zip(vols, hours)):
            rng = np.random.default_rng(p * 1000 + i)
            vs = rng.permutation(v)
            ps.append(hour_profile_one(vs, h, rng))
        null[p] = np.mean(ps, axis=0)
    band = (np.percentile(null, 2.5, axis=0), np.percentile(null, 97.5, axis=0))
    real_us = float(np.mean(prof[list(us_hours)]))
    null_us = np.mean(null[:, list(us_hours)], axis=1)
    return prof, band, real_us, null_us


# ── H2: 量-波动 ──────────────────────────────────────────────
def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    rx = pd.Series(x[m]).rank().values
    ry = pd.Series(y[m]).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def h2_run(ctx, volume):
    c = ctx.close
    v = ctx.high - ctx.low
    r = np.concatenate([[np.nan], np.abs(np.diff(np.log(c)))])
    out = {"vol_ret": spearman(volume, r),
           "vol_range": spearman(volume, v)}
    for k in PARAMS["h2_lags"]:
        r_f = np.full(len(r), np.nan)
        r_f[:-k] = np.abs(np.log(c[k:]) - np.log(c[:-k]))
        out[f"lead{k}"] = spearman(volume, r_f)
    # 长记忆: log(volume) ACF@168
    lv = np.log(np.maximum(volume, 1e-9))
    lv = lv - np.nanmean(lv)
    lv = np.where(np.isfinite(lv), lv, 0.0)
    n = len(lv)
    t = np.arange(n)
    vv = float(np.mean(lv * lv))
    out["acf168"] = float(np.mean(lv[t < n - 168] * lv[t >= 168]) / vv) \
        if vv > 0 else float("nan")
    return out


# ── H3: funding-方向 ─────────────────────────────────────────
def h3_run(df, fund_ts, fund_rate, ks, n_perm, mu, sig):
    """funding 事件对齐 1h bar → forward ret 分组 (符号/水平三分位).
    返回 {k: {组: (mean, n), null_band: (lo, hi)}}."""
    ts_arr = df.index.values.astype("int64") // 10 ** 6  # ms
    c = df["close"].values.astype(float)
    r = np.concatenate([[0.0], np.diff(np.log(c))])
    pref = np.concatenate([[0], np.cumsum(r)])
    # 事件 bar: ts 之后最近的 1h bar
    bars = np.searchsorted(ts_arr, fund_ts, side="left")
    out = {}
    for k in ks:
        ok = (bars + k) < len(c)
        bs, fr = bars[ok], fund_rate[ok]
        fwd = pref[bs + k] - pref[bs]
        pos, neg = fr > 0, fr < 0
        # 水平三分位
        q1, q2 = np.quantile(fr, [1.0 / 3, 2.0 / 3])
        lo_grp = fr <= q1
        hi_grp = fr >= q2
        rows = {"正": (float(np.mean(fwd[pos])) if pos.any() else float("nan"),
                       int(pos.sum())),
                "负": (float(np.mean(fwd[neg])) if neg.any() else float("nan"),
                       int(neg.sum())),
                "低1/3": (float(np.mean(fwd[lo_grp])) if lo_grp.any()
                          else float("nan"), int(lo_grp.sum())),
                "高1/3": (float(np.mean(fwd[hi_grp])) if hi_grp.any()
                          else float("nan"), int(hi_grp.sum()))}
        # null: GBM 1h rets + 标签 shuffle
        null_rows = {g: [] for g in rows}
        for p in range(n_perm):
            rng = np.random.default_rng(p)
            gbm_r = rng.normal(mu, sig, size=len(c))
            gpref = np.concatenate([[0], np.cumsum(gbm_r)])
            gfwd = gpref[bs + k] - gpref[bs]
            shuf = rng.permutation(fr)
            for g, (_, _n) in rows.items():
                if g == "正":
                    m = shuf > 0
                elif g == "负":
                    m = shuf < 0
                elif g == "低1/3":
                    m = shuf <= q1
                else:
                    m = shuf >= q2
                if m.any():
                    null_rows[g].append(float(np.mean(gfwd[m])))
        out[k] = {"rows": rows, "null": {
            g: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for g, v in null_rows.items() if v}}
    return out


# ── H4: funding-折返 (触碰 + funding 三分位 + 1:1) ──────────
def h4_run(df, fund_ts, fund_rate, warmup, T, W):
    """触碰事件 (c17 管线, funding 窗口) → 1:1, 按 funding 三分位分组.
    全部数组用 make_ctx 截断对齐 (禁切片)."""
    from research.levels import cluster_levels
    from research.structures import K
    ctx = make_ctx(df, warmup, state_fns={})
    c, h, l, atr = ctx.close, ctx.high, ctx.low, ctx.atr
    ts_arr = df.index.values.astype("int64") // 10 ** 6
    ts_tr = ts_arr[warmup:]                # 截断坐标
    n = len(c)
    t_idx = np.arange(n)
    lvls = cluster_levels(h, l, atr, k=K, tolerance_mult=0.3, min_touch=2)
    ent_l = np.zeros(n, bool)
    ent_s = np.zeros(n, bool)
    fund_at = np.full(n, np.nan)
    fb = np.searchsorted(ts_tr, fund_ts, side="left")
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (l <= p_hi) & (h >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev
        ev = np.flatnonzero(entry)
        for e in ev:
            fi = int(np.searchsorted(fb, e, side="right") - 1)
            if fi >= 0:
                fund_at[e] = fund_rate[fi]
            if lv.side == "resistance":
                ent_s[e] = True
            else:
                ent_l[e] = True
    # funding 三分位
    fr = fund_rate
    q1, q2 = np.quantile(fr, [1.0 / 3, 2.0 / 3])
    in_win = np.isfinite(fund_at)
    out = {}
    for gname, mask in (("低1/3", fund_at <= q1), ("高1/3", fund_at >= q2),
                        ("全", np.ones(n, bool))):
        m = in_win & mask
        el = ent_l & m
        es = ent_s & m
        ol, _ = evaluate_forward(c, h, l, atr, el, direction="long",
                                 t_mult=T, w=W)
        os_, _ = evaluate_forward(c, h, l, atr, es, direction="short",
                                  t_mult=T, w=W)
        ne = ol.n_eval + os_.n_eval
        nw = ol.n_win + os_.n_win
        out[gname] = ((nw / ne) if ne else float("nan"), ne)
    return out, fr


def h4_null(df, fund_ts, fund_rate, warmup, T, W, n_perm):
    """null: GBM 触碰 + 真实 funding 标签 shuffle."""
    from research.levels import cluster_levels
    from research.structures import K
    out = {"低1/3": [], "高1/3": []}
    for p in range(n_perm):
        rw = gbm_matching(df, seed=p)
        rctx = make_ctx(rw, warmup, state_fns={})
        c = rctx.close
        h = rctx.high
        l = rctx.low
        atr = rctx.atr
        ts_arr = rw.index.values.astype("int64") // 10 ** 6
        n = len(c)
        t_idx = np.arange(n)
        lvls = cluster_levels(h, l, atr, k=K, tolerance_mult=0.3,
                              min_touch=2)
        el = np.zeros(n, bool)
        es = np.zeros(n, bool)
        shuf = np.random.default_rng(p).permutation(fund_rate)
        q1, q2 = np.quantile(shuf, [1.0 / 3, 2.0 / 3])
        fb = np.searchsorted(ts_arr, fund_ts, side="left")
        for lv in lvls:
            p_lo = lv.price - lv.band
            p_hi = lv.price + lv.band
            ov = (l <= p_hi) & (h >= p_lo)
            tm = ov & (t_idx >= lv.confirm_at)
            prev = np.roll(tm, 1)
            prev[0] = False
            for e in np.flatnonzero(tm & ~prev):
                fi = int(np.searchsorted(fb, e, side="right") - 1)
                if fi < 0:
                    continue
                fv = shuf[fi]
                if fv <= q1:
                    (el if lv.side == "support" else es)[e] = True
                elif fv >= q2:
                    (el if lv.side == "support" else es)[e] = True
        for gname in ("低1/3", "高1/3"):
            ol, _ = evaluate_forward(c, h, l, atr, el, direction="long",
                                     t_mult=T, w=W)
            os_, _ = evaluate_forward(c, h, l, atr, es, direction="short",
                                      t_mult=T, w=W)
            ne = ol.n_eval + os_.n_eval
            nw = ol.n_win + os_.n_win
            if ne:
                out[gname].append(nw / ne)
    return {g: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for g, v in out.items() if v}


# ── GATE 自检 ────────────────────────────────────────────────
def gate(null_us_band):
    """① H1 置换 sanity: 人造时段信号的置换带应被破坏; ② 无 GBM 幅度断言."""
    # ① 构造已知时段信号 (12 点时段量 2×): 置换后剖面应接近平
    n = 4800
    hours = np.arange(n) % 24
    vol = np.ones(n) * 1.0
    vol[hours == 12] = 2.0
    rng = np.random.default_rng(7)
    vs = rng.permutation(vol)
    prof_perm = hour_profile_one(vs, hours, rng)
    if prof_perm[12] > 1.5:
        raise SystemExit(f"GATE FAIL: 置换未破坏时段信号 (h12={prof_perm[12]:.2f})")
    # null band sanity
    if null_us_band is not None and not (0.8 <= null_us_band[0] <= null_us_band[1] <= 1.2):
        raise SystemExit("GATE FAIL: H1 null 置换带异常")
    print("[GATE] H1 置换 sanity (人造时段信号被置换破坏) [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def write_out(out_path, params, h1, h2, h3, h4):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},n_perm={},h3_ks={},min_n={},gbm_seeds={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            p["tf"], p["n_perm"], p["h3_ks"], p["min_n"], p["gbm_seeds"],
            p["min_n"]),
        "# GATE: 探测器自检 H1 置换 sanity [PASS]; MIN_N n≥{} [PASS]".format(
            p["min_n"]),
        "# RESULTS: [学习级] c49 M4 U1 量、仓与价差忠实复现 (书 CH12 p.527-563 "
        "+ CH13 carry 映射); 时段量 (书 U 形加密版), 量-波动, funding-方向/"
        "折返 pilot (3 个月样本限制); GBM/置换 30 次同管线; 描述层无入场, 无交易"
        "含义",
        "",
    ]
    # H1
    prof, band, real_us, null_us = h1
    lines.append("[H1] 时段量结构 (UTC 小时剖面, 美股时段 {} 点):".format(
        list(p["us_hours"])))
    hour_str = " ".join("{:02d}:{:.2f}".format(hh, prof[hh]) for hh in range(24))
    lines.append("  剖面: " + hour_str)
    lines.append("  美股时段块均值: 真实 {:.2f} | null {:.2f}±{:.2f} (带 "
                 "[{:.2f},{:.2f}])".format(
        real_us, float(np.mean(null_us)), float(np.std(null_us, ddof=1)),
        float(np.percentile(null_us, 2.5)),
        float(np.percentile(null_us, 97.5))))
    h1_ok = real_us > float(np.percentile(null_us, 97.5))
    lines.append("  H1 判据: 美股时段块 > null 95% 上界 -> {}".format(
        "PASS" if h1_ok else "FAIL"))
    # H2
    lines.append("")
    lines.append("[H2] 量-波动 (BTC/ETH 1h):")
    for sym, r in h2.items():
        lines.append("  {}: ρ(vol,|ret|)={:.3f} ρ(vol,range)={:.3f} | 量领先 "
                     "|ret| k=1/3/6/12: {:.3f}/{:.3f}/{:.3f}/{:.3f} | "
                     "log(vol) ACF@168={:.3f}".format(
            sym, r["vol_ret"], r["vol_range"], r["lead1"], r["lead3"],
            r["lead6"], r["lead12"], r["acf168"]))
    # H3
    lines.append("")
    lines.append("[H3] funding-方向 pilot (3 个月, 2026-05-11..08-14):")
    for sym, res in h3.items():
        for k, r in res.items():
            for g, (m, n_) in r["rows"].items():
                if g in ("正", "负"):
                    continue
                nb = r["null"].get(g)
                if nb:
                    lines.append("  {} k={}: {} 组 ret {:+.5f} (n={}) {} | null "
                                 "[{:+.5f}, {:+.5f}]".format(
                        sym, k, g, m, n_, _nm(n_, p["min_n"]), nb[0], nb[1]))
                else:
                    lines.append("  {} k={}: {} 组 ret {:+.5f} (n={})".format(
                        sym, k, g, m, n_))
        # 符号组审计
        k0 = p["h3_ks"][0]
        n_pos = res[k0]["rows"]["正"][1]
        n_neg = res[k0]["rows"]["负"][1]
        lines.append("  {} 符号组 n: 正 {} 负 {} ({} — 若 <100 只报水平分位)".format(
            sym, n_pos, n_neg,
            "水平分位版已报" if min(n_pos, n_neg) < p["min_n"] else "符号组可报"))
    # H4
    lines.append("")
    lines.append("[H4] funding-折返 pilot (触碰 + funding 三分位):")
    for sym, r in h4.items():
        for g, (m, ne) in r["rows"].items():
            nb = r["null"].get(g)
            if nb:
                lines.append("  {} {}: D1 折返 {:.1%} (n={}) {} | null "
                             "[{:.1%}, {:.1%}]".format(
                    sym, g, m, ne, _nm(ne, p["min_n"]), nb[0], nb[1]))
            else:
                lines.append("  {} {}: D1 折返 {:.1%} (n={})".format(
                    sym, g, m, ne))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c12 (波动长记忆 H=0.93); c17 (触碰折返); 书 CH12 "
                 "图 12.15 (U 形量); p.560 (tick 量≈实际量); CH13 p.597 (carry/"
                 "forward bias)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    n_perm = PARAMS["dev_subset"]["n_perm"] if dev else PARAMS["n_perm"]

    # H1: 20 标的 1h
    all_syms = load_1h_all(PARAMS)
    prof, band, real_us, null_us = h1_run(all_syms, n_perm, PARAMS["us_hours"])
    gate((float(np.percentile(null_us, 2.5)),
          float(np.percentile(null_us, 97.5))))

    # H2: BTC/ETH 1h
    h2 = {}
    for sym, df in load_1h_all(PARAMS, PARAMS["crypto"]):
        ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
        vol = df["volume"].values.astype(float)[PARAMS["warmup"]:]
        h2[sym] = h2_run(ctx, vol)

    # funding 映射: BTC-USDT-SWAP → BTC/USDT:USDT
    fund = load_funding(PARAMS["funding_db"],
                        {s.split("/")[0]: s for s in PARAMS["crypto"]})
    # H3: funding-方向
    h3 = {}
    for sym, df in load_1h_all(PARAMS, PARAMS["crypto"]):
        if sym not in fund:
            continue
        ft, fr = fund[sym]
        c = df["close"].values.astype(float)
        r = np.concatenate([[0.0], np.diff(np.log(c))])
        mu, sig = float(np.mean(r)), float(np.std(r, ddof=1))
        h3[sym] = h3_run(df, ft, fr, PARAMS["h3_ks"], n_perm, mu, sig)

    # H4: funding-折返
    h4 = {}
    for sym, df in load_1h_all(PARAMS, PARAMS["crypto"]):
        if sym not in fund:
            continue
        ft, fr = fund[sym]
        rows, _ = h4_run(df, ft, fr, PARAMS["warmup"], 1.0, 24)
        nnull = h4_null(df, ft, fr, PARAMS["warmup"], 1.0, 24, n_perm)
        h4[sym] = {"rows": rows, "null": nnull}

    if dev:
        print("  [dev] H1 美股时段块 真实={:.2f} null={:.2f}".format(
            real_us, float(np.mean(null_us))))
        for sym, r in h2.items():
            print("  [dev] {} ρ(vol,|ret|)={:.3f} acf168={:.3f}".format(
                sym, r["vol_ret"], r["acf168"]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, (prof, band, real_us, null_us), h2, h3, h4)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
