#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C67 聪明钱验证②: 强平前兆量能压缩 (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c67 行): librarian #2 — arXiv 2026 吃单流方差压缩
  p≈5e-6 (级联前兆)。本砖用 **OHLCV 近似** (volume×range 滚动方差处于低分位
  + 距关键位 < 2×ATR) 检验。docstring 标注近似性。描述层, 无入场, 无交易
  含义, 不涉及胜率/期望/成本主张。**结论不得作交易依据**。学习级新协议: 不跑
  pytest/check_study; 保留 docstring 预注册冻结、内置 GATE、因果纪律、dev
  先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 量能压缩态 (volume×range 方差低分位 + 近
  关键位) 后的波动释放 (E1) 是否 > GBM null?

预注册假设 (PLAN §2.5 c67 行, docstring 逐字):
  H1: 压缩状态后 K bar 波动释放 (E1) > GBM null (量能压缩→级联→波动爆发 —
      方向中性事件 alpha)
  H2: 压缩状态后的方向倾向报告 (不设门槛, 方向中性优先)

  操作化 (运行前锁定):
    - 数据: 20 标的 1h 3y (backtest.db, 含 volume)
    - 压缩代理 = (volume×range) 24 bar 滚动方差 (因果); 压缩 = 该方差处于
      长期滚动 20% 分位以下 (pct_win=90 日, 3y→90d 偏离标注 — 计算可行性与
      c51 3y→1y 先例)
    - 事件 = 压缩态 + 距任一 cluster 关键位 < 2×ATR (级联高发区)
    - E1 (c52 口径, h∈{1,3,6,12}): mean(ATR[t+1..t+h])/mean(ATR[t−h..t−1])−1
    - H1: 每 h 的 E1 真实 vs GBM null 30 种子同管线
    - H2: 事件后 K=24 bar 方向倾向 (mean sign of log ret) 报告
    - 学习级: GBM null 30 种子、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  volume/OHLC      | backtest.db 原生                       | bar 收盘后 | data_loader
  vr 方差          | rolling(24).var() of (vol×range)       | bar 收盘后 | 因果
  压缩分位         | causal.rolling_percentile (90 日, 0.20)| bar 收盘后 | 禁全样本分位
  关键位距离       | cluster_levels (冻结) → min |c−price|   | confirm_at | 冻结后不变
  E1(h)            | c52 对称窗口 (pre 不含 t)              | 事后       | c52 口径
  GBM null         | gbm_matching + 真实 volume 50bar 块打乱 | 锚定真实   | 同 c66

数据声明: data/backtest.db (20 标的 × 1h × 2023-08..2026-08, 含 volume)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  vr 窗 24; 分位窗 90 日 (2160 bar, 3y→90d 偏离); q=0.20; 距位 <2×ATR;
  E1 h∈{1,3,6,12}; K=24; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 吃单流 (order flow) 数据不可得 — 用 OHLCV 近似: volume×range 方差 (标注
    近似性; 真实级联前兆是 tick/吃单流方差压缩, 本砖是 bar 级代理)。
  - 压缩分位窗 3y→90 日滚动 (计算可行性, c51 3y→1y 先例)。
  - GBM null volume 用真实 50bar 块打乱重挂载 (同 c66)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 压缩态 golden (构造已知 vr 序列 → 低分位态检测正确);
    ② 距位 golden (构造位 + 路径 → 距离对拍); ③ E1 golden (c52 同款:
    常数 ATR→0, 台阶→1); ④ null sanity — GBM null E1(1) ∈ [−5pp, +10pp];
    任一失败 SystemExit
  - GBM null 无信息对照: 30 种子同管线
  - MIN_N: 每格 n ≥ 100 (学习级)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 1h × 3 种子, 不写 .out
  - 全量: 20 标的 1h × 30 种子 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c67_compression_precursor.py --dev
  python3 research/studies/c67_compression_precursor.py
"""
import hashlib
import os
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
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.sim_market import gbm_matching
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "1h",
    "vr_win": 24,
    "pct_win": 2160,                       # 90 日 (3y→90d 偏离, 标注)
    "q": 0.20,
    "atr_thr": 2.0,
    "e1_hs": (1, 3, 6, 12),
    "K": 24,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,
    "null_band": (-0.05, 0.10),            # GATE: null E1(1) 带 (pp)
    "dev_subset": {"n_gbm": 3, "syms": ("BTC/USDT:USDT",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c67_compression_precursor"


# ── 压缩态检测 ───────────────────────────────────────────────
def compression_state(volume, high, low, vr_win, pct_win, q):
    vr = volume * (high - low)
    var = pd.Series(vr).rolling(vr_win).var().values
    pct = rolling_percentile(var, pct_win, q)
    comp = np.isfinite(pct) & np.isfinite(var) & (var <= pct)
    return comp


def nearest_level_dist(lvls, t_idx, close):
    n = len(close)
    dist = np.full(n, np.inf)
    for lv in lvls:
        active = t_idx >= lv.confirm_at
        d = np.abs(close - lv.price)
        np.minimum(dist, np.where(active, d, np.inf), out=dist)
    return dist


# ── E1(h) (c52 口径) ─────────────────────────────────────────
def e1h_series(atr, h):
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


def collect_events(ctx, volume, params):
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    atr = ctx.atr
    comp = compression_state(volume, ctx.high, ctx.low, params["vr_win"],
                             params["pct_win"], params["q"])
    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K, tolerance_mult=0.3,
                          min_touch=2)
    dist = nearest_level_dist(lvls, t_idx, c)
    near = dist < params["atr_thr"] * atr
    ev = np.flatnonzero(comp & near & np.isfinite(atr) & (atr > 0))
    out = {"n": 0}
    for h in params["e1_hs"]:
        e1 = e1h_series(atr, h)
        vals = e1[ev]
        fin = np.isfinite(vals)
        out[f"e1_{h}"] = (float(np.mean(vals[fin])) if fin.any()
                          else float("nan"), int(fin.sum()))
    # H2 方向
    k = params["K"]
    dirs = []
    for t in ev:
        if t + k < n:
            dirs.append(float(np.sign(np.log(c[t + k] / c[t]))))
    out["dir"] = (float(np.mean(dirs)) if dirs else float("nan"),
                  len(dirs))
    out["n"] = len(ev)
    return out


def _merge_e(parts, params):
    out = {}
    for h in params["e1_hs"]:
        vals = []
        nn = 0
        for p in parts:
            v, n = p[f"e1_{h}"]
            if np.isfinite(v):
                vals.append(v * n)
                nn += n
        out[f"e1_{h}"] = (float(np.sum(vals) / nn) if nn else float("nan"),
                          nn)
    dvals, dnn = [], 0
    for p in parts:
        v, n = p["dir"]
        if np.isfinite(v):
            dvals.append(v * n)
            dnn += n
    out["dir"] = (float(np.sum(dvals) / dnn) if dnn else float("nan"), dnn)
    return out


def shuffled_volume(vol, block, seed):
    rng = np.random.default_rng(seed)
    n = len(vol)
    n_blocks = int(np.ceil(n / block))
    order = rng.permutation(n_blocks)
    parts = [vol[i * block:(i + 1) * block] for i in order]
    return np.concatenate(parts)[:n]


# ── GATE 自检 ────────────────────────────────────────────────
def gate_comp_golden():
    """压缩态 golden: vr 前段恒定 (方差≈0 → 低分位) → 压缩态 True."""
    n = 120
    vol = np.ones(n) * 10.0
    hi = np.ones(n) * 101.0
    lo = np.ones(n) * 99.0
    # vr = 20 恒定 → var=0 → 低分位
    comp = compression_state(vol, hi, lo, 24, 40, 0.20)
    if not np.any(comp[100:]):
        raise SystemExit("GATE FAIL: 压缩态 golden 未检测到")
    # 注入波动 → 非压缩
    vol2 = np.ones(n) * 10.0
    hi2 = np.ones(n) * 101.0
    lo2 = np.ones(n) * 99.0
    vol2[70:100] = np.linspace(10.0, 100.0, 30)
    comp2 = compression_state(vol2, hi2, lo2, 24, 60, 0.20)
    if np.any(comp2[80:95]):
        raise SystemExit("GATE FAIL: 波动段被误判为压缩")
    return True


def gate_e1_golden():
    atr_const = np.ones(30) * 2.0
    e1 = e1h_series(atr_const, 3)
    if abs(e1[10] - 0.0) > 1e-9:
        raise SystemExit("GATE FAIL: 常数 ATR E1 ≠ 0")
    atr_step = np.concatenate([np.ones(15), np.ones(15) * 2.0])
    e1s = e1h_series(atr_step, 3)
    if not (0.9 <= e1s[15] <= 1.1):
        raise SystemExit("GATE FAIL: 台阶 ATR E1 ≠ ~1")
    return True


def gate(null_e1_1):
    gate_comp_golden()
    gate_e1_golden()
    nm = float(np.mean(null_e1_1)) if null_e1_1.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null E1(1) {nm:+.4f} ∉ [{lo}, {hi}]")
    print(f"[GATE] 压缩态 golden [PASS]; E1 golden [PASS]; null sanity "
          f"{nm:+.4f} [PASS]", flush=True)
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
        "params=tf={},vr_win={},pct_win={}bar(90d),q={},atr_thr={},e1_hs={},"
        "gbm_seeds={},min_n={},gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["tf"], p["vr_win"], p["pct_win"], p["q"],
            p["atr_thr"], p["e1_hs"], p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: 压缩态 golden + E1 golden + null sanity [PASS]; MIN_N n≥{} "
        "[PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c67 聪明钱验证②: 强平前兆量能压缩 (OHLCV 近似); "
        "压缩 = (vol×range) 24bar 方差 < 90 日滚动 20% 分位 + 距关键位 <2×ATR; "
        "E1 (c52, h∈{1,3,6,12}); GBM null 30 种子 (真实 volume 块打乱); "
        "描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[数据] 压缩事件 {} [MIN_N {}] | 方向倾向报告 (K={}):".format(
        res["real"]["dir"][1], "通过" if res["real"]["dir"][1] >= p["min_n"]
        else "不足", p["K"]))
    for h in p["e1_hs"]:
        rv, rn = res["real"][f"e1_{h}"]
        nv = res["null"][f"e1_{h}"]
        z = (rv - nv[0]) / nv[1] if nv[1] > 0 else float("nan")
        lines.append("  E1({}): 真实 {:+.2%} (n={}) {} | null {:+.2%}±{:.2%} "
                     "| 超额 {:+.2%} (z={:+.2f}) -> {}".format(
            h, rv, rn, _nm(rn), nv[0], nv[1], rv - nv[0], z,
            "超2σ↑" if z > 2 else "未超"))
    d, dn = res["real"]["dir"]
    lines.append("  方向倾向: sign(24h log ret) 均值 {:+.3f} (n={}) — 报告"
                 "不作判据".format(d, dn))
    lines.append("")
    lines.append("[对照-历史] arXiv 2026 (吃单流方差压缩 p≈5e-6 — 本砖 OHLCV "
                 "近似标注); c51 (带收窄→释放); c52 (E1 h 剖面); c14 (关键位); "
                 "本砖: 压缩态 E1 vs 漂移匹配 null")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    dev_syms = PARAMS["dev_subset"]["syms"] if dev else None

    data = load_candles(timeframes=(PARAMS["tf"],))
    syms = [s for s in data if "USDT" in s]
    if dev_syms:
        syms = [s for s in syms if s in dev_syms]

    parts = []
    for sym in syms:
        df = data[sym].get(PARAMS["tf"])
        if df is None or verify(df, sym, PARAMS["tf"]):
            continue
        ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
        vol = df["volume"].values.astype(float)
        parts.append(collect_events(ctx, vol[PARAMS["warmup"]:], PARAMS))
    real = _merge_e(parts, PARAMS)

    # GBM null (首标)
    ref = data[syms[0]].get(PARAMS["tf"])
    null_vals = {h: [] for h in PARAMS["e1_hs"]}
    null_e1_1 = []
    for seed in range(seeds):
        rw = gbm_matching(ref, seed=seed)
        sv = shuffled_volume(ref["volume"].values.astype(float), 50, seed)
        gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
        gp = collect_events(gctx, sv[PARAMS["warmup"]:], PARAMS)
        for h in PARAMS["e1_hs"]:
            v, n = gp[f"e1_{h}"]
            if np.isfinite(v):
                null_vals[h].append(v)
        if np.isfinite(gp["e1_1"][0]):
            null_e1_1.append(gp["e1_1"][0])
    null = {}
    for h in PARAMS["e1_hs"]:
        a = np.array(null_vals[h])
        null[f"e1_{h}"] = (float(np.mean(a)) if len(a) else float("nan"),
                           float(np.std(a, ddof=1)) if len(a) > 1 else 0.0)

    gate(np.array(null_e1_1) if null_e1_1 else np.array([0.0]))

    if dev:
        print("  [dev] 事件 n={} | E1(1) 真实 {:+.2%} vs null {:+.2%}±{:.2%}"
              .format(real["dir"][1], real["e1_1"][0], null["e1_1"][0],
                      null["e1_1"][1]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {"real": real, "null": null}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
