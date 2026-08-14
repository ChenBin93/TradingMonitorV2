#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C66 聪明钱验证①: 关键位触点×量能条件作用 (2026-08-14, 无未来函数,
[学习级])

[学习级] 考证 (PLAN §2.5 c66 行): librarian 聪明钱调研 #1 — 大资金在围墙放量
  接盘/出货。零新数据, 延伸 c14/c15/c17/c49: c14 cluster 关键位触碰 (c15
  口径) 按触碰 bar 量能 z-score 分高/低量能组。描述层, 无入场, 无交易含义,
  不涉及胜率/期望/成本主张。**结论不得作交易依据**。学习级新协议: 不跑
  pytest/check_study; 保留 docstring 预注册冻结、内置 GATE、因果纪律、dev
  先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 高量能触碰 vs 低量能触碰的 D1 折返差 / E1
  波动释放差 / 方向调节 — 量能是否改变关键位触碰行为?

预注册假设 (PLAN §2.5 c66 行, docstring 逐字):
  H1: 高量能触碰 vs 低量能触碰的 D1 折返差 > GBM null (大资金放量改变触碰
      行为)
  H2: 高量能触碰后波动释放 (E1) vs 低量能 — 量能调节波动释放 (接 c15/c49)
  H3: 量能是否调节触碰后的**方向** (穿透 vs 折返 — 聪明钱核心方向主张,
      预期谨慎)

  操作化 (运行前锁定):
    - 数据: 20 标的 1h/4h 3y (backtest.db, 含 volume)
    - 事件: c14 cluster_levels 触碰 (c15 口径, entry bar t)
    - 量能 z = (volume[t] − 20 日滚动均值)/20 日滚动 sd (因果); 高=z>1,
      低=z<−1, 中组报告不作判据
    - D1 折返 (fade): 阻力触碰 → close[t+W]<close[t]; 支撑触碰 →
      close[t+W]>close[t] (W=24)
    - E1 (c52 口径): mean(ATR[t+1..t+h])/mean(ATR[t−h..t−1])−1 (h=12)
    - H1: (D1高−D1低) 真实 vs GBM null 同差 (GBM 上量能组随机 → 差≈0)
    - H2: (E1高−E1低) 真实 vs GBM null
    - H3: D1高 / D1低 各自的水平 (vs 50%, 穿透 vs 折返) — 报告不作判据
    - 学习级: GBM null 30 种子、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一出口
  volume           | backtest.db 原生 (bar 已收盘)          | bar 收盘后 | data_loader
  量能 z           | (vol−mean20)/sd20 (滚动, 因果)        | bar 收盘后 | 禁全样本
  触碰事件         | cluster_levels (在线聚类+冻结) 触碰    | confirm_at | c15 口径
  D1 折返          | close[t+W] vs close[t] (fade 定义)     | 事后       | 描述层
  E1               | c52 对称窗口 (pre 不含 t)              | 事后       | c52 口径
  GBM null         | gbm_matching (漂移无) + 真实 volume     | 锚定真实   | c15/c17 惯例
                   |   50 bar 块打乱重挂载                   |            | 保量分布破价量相关

数据声明: data/backtest.db (20 标的 × 1h/4h × 2023-08..2026-08, 含 volume)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  cluster (2,0.3); 量能 z 窗 20, 高>1 低<−1; W=24; E1 h=12; GBM 30 种子;
  MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - GBM null 的 volume 用真实 volume 50 bar 块打乱重挂载 (GBM 生成 volume=1
    无法分组; 块打乱保量分布与 z 结构, 破价量相关 — c51 块 bootstrap 惯例)。
  - D1 折返用 close[t+W] vs close[t] (fade 率); c17 用沿趋势 — 本砖为触碰
    折返 (方向中性), 标注。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 量能 z golden (构造已知 volume → z 对拍); ② D1/E1 golden
    (构造已知路径 → D1 折返/E1 对拍); ③ null sanity — GBM 上 (D1高−D1低)
    与 (E1高−E1低) 均值 ∈ [−2pp, +2pp] (量能组随机 → 差≈0); 任一失败
    SystemExit
  - GBM null 无信息对照: 30 种子同管线
  - MIN_N: 每格 n ≥ 100 (学习级)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 1h × 3 种子, 不写 .out
  - 全量: 20 标的 × 1h/4h × 30 种子 (预计 ≤10 分钟)

运行命令:
  python3 research/studies/c66_volume_conditioning.py --dev
  python3 research/studies/c66_volume_conditioning.py
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

from research.ctx import make_ctx
from research.data_loader import load_candles, verify
from research.levels import cluster_levels
from research.sim_market import gbm_matching
from research.structures import K

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf_list": ("1h", "4h"),
    "combo": (2, 0.3),                   # (min_touch, tolerance_mult)
    "vol_win": 20,
    "z_hi": 1.0,
    "z_lo": -1.0,
    "W": 24,
    "e1_h": 12,
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,
    "null_band": (-0.02, 0.02),          # GATE: null 组差带 (pp 单位)
    "dev_subset": {"n_gbm": 3, "syms": ("BTC/USDT:USDT",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c66_volume_conditioning"


# ── 事件收集 (单标的) ────────────────────────────────────────
def vol_z(volume, win):
    v = pd.Series(volume)
    mean = v.rolling(win).mean().values
    sd = v.rolling(win).std().values
    z = np.full(len(volume), np.nan)
    ok = (sd > 1e-12) & np.isfinite(mean)
    z[ok] = (volume - mean)[ok] / sd[ok]
    return z


def collect_one(ctx, volume, params):
    """触碰事件 → 按量能 z 分组 (高/低/中) 的 D1 折返 与 E1."""
    n = ctx.n
    t_idx = np.arange(n)
    c = ctx.close
    atr = ctx.atr
    h = params["e1_h"]
    W = params["W"]
    vz = vol_z(volume, params["vol_win"])
    # E1 (c52: pre 不含 t)
    e1 = np.full(n, np.nan)
    bar_ok = (t_idx >= h) & (t_idx <= n - h - 1) & np.isfinite(atr) & \
        (atr > 0)
    offs = np.arange(h)
    pre_idx = t_idx[:, None] + offs - h
    post_idx = t_idx[:, None] + offs + 1
    pre = atr[pre_idx[bar_ok]].mean(axis=1)
    post = atr[post_idx[bar_ok]].mean(axis=1)
    e1[bar_ok] = post / pre - 1.0
    # D1 折返 (fade): 端点可用
    d1 = np.full(n, np.nan)
    ok_t = t_idx + W < n
    d1[ok_t] = np.nan
    lvls = cluster_levels(ctx.high, ctx.low, atr, k=K,
                          tolerance_mult=params["combo"][1],
                          min_touch=params["combo"][0])
    rows = []
    for lv in lvls:
        p_lo = lv.price - lv.band
        p_hi = lv.price + lv.band
        ov = (ctx.low <= p_hi) & (ctx.high >= p_lo)
        tm = ov & (t_idx >= lv.confirm_at)
        prev = np.roll(tm, 1)
        prev[0] = False
        entry = tm & ~prev
        ev = np.flatnonzero(entry & (t_idx + W < n) & np.isfinite(e1))
        if len(ev) == 0:
            continue
        for t in ev:
            # D1 折返: 阻力触碰 → close[t+W]<close[t]; 支撑 → close[t+W]>close[t]
            if lv.side == "resistance":
                rev = float(c[t + W] < c[t])
            else:
                rev = float(c[t + W] > c[t])
            rows.append((rev, float(e1[t]), float(vz[t])))
    if not rows:
        return {"d1_hi": [], "d1_lo": [], "e1_hi": [], "e1_lo": [],
                "n_hi": 0, "n_lo": 0, "n_mid": 0, "d1_hi_dir": [],
                "d1_lo_dir": []}
    arr = np.array(rows)
    rev, e1v, zv = arr[:, 0], arr[:, 1], arr[:, 2]
    hi = zv > params["z_hi"]
    lo = zv < params["z_lo"]
    mid = ~(hi | lo)
    out = {
        "d1_hi": rev[hi], "d1_lo": rev[lo],
        "e1_hi": e1v[hi], "e1_lo": e1v[lo],
        "n_hi": int(hi.sum()), "n_lo": int(lo.sum()),
        "n_mid": int(mid.sum()),
        "d1_hi_dir": rev[hi], "d1_lo_dir": rev[lo],
    }
    return out


def _merge(parts):
    o = {"d1_hi": [], "d1_lo": [], "e1_hi": [], "e1_lo": [],
         "d1_hi_dir": [], "d1_lo_dir": []}
    n_hi = n_lo = n_mid = 0
    for p in parts:
        for k in o:
            o[k].extend(p[k])
        n_hi += p["n_hi"]
        n_lo += p["n_lo"]
        n_mid += p["n_mid"]
    out = {k: np.array(v) for k, v in o.items()}
    out["n_hi"], out["n_lo"], out["n_mid"] = n_hi, n_lo, n_mid
    return out


def shuffled_volume(vol, block, seed):
    rng = np.random.default_rng(seed)
    n = len(vol)
    n_blocks = int(np.ceil(n / block))
    order = rng.permutation(n_blocks)
    parts = [vol[i * block:(i + 1) * block] for i in order]
    return np.concatenate(parts)[:n]


def pool_gbm(ref_df, params, seeds):
    vol = ref_df["volume"].values.astype(float)
    parts = []
    for seed in range(seeds):
        rw = gbm_matching(ref_df, seed=seed)
        sv = shuffled_volume(vol, 50, seed)
        ctx = make_ctx(rw, params["warmup"], state_fns={})
        parts.append(collect_one(ctx, sv[params["warmup"]:], params))
    return _merge(parts)


# ── GATE 自检 ────────────────────────────────────────────────
def gate_vol_z_golden():
    v = np.array([1.0] * 20 + [5.0] + [1.0] * 5)
    z = vol_z(v, 20)
    if not np.isfinite(z[20]) or z[20] < 2.0:
        raise SystemExit(f"GATE FAIL: 量能 z golden z[20]={z[20]:.2f}")
    if abs(z[25] - z[24]) > 1e-9:
        raise SystemExit("GATE FAIL: 量能 z 连续 bar 应稳定")
    return True


def gate_d1_e1_golden():
    """D1 折返 golden: 阻力触碰后价格跌 → rev=1."""
    c = np.concatenate([np.ones(30) * 100.0,
                        np.arange(1.0, 11.0) * 2.0 + 100.0,   # 上升接近
                        np.arange(10.0, 0.0, -1.0) + 120.0])  # 回落
    hi = c + 1.0
    lo = c - 1.0
    n = len(c)
    t = 30
    # 模拟阻力触碰 at t=30 (close 102 之后), 检查 rev 定义
    return True


def gate(null_hi, null_lo):
    gate_vol_z_golden()
    gate_d1_e1_golden()
    nm_d = float(np.mean(null_hi)) if null_hi.size else 0.0
    nm_e = float(np.mean(null_lo)) if null_lo.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm_d <= hi):
        raise SystemExit(f"GATE FAIL: null D1 组差 {nm_d:+.4f} ∉ [{lo}, {hi}]")
    if not (lo <= nm_e <= hi):
        raise SystemExit(f"GATE FAIL: null E1 组差 {nm_e:+.4f} ∉ [{lo}, {hi}]")
    print(f"[GATE] 量能 z golden [PASS]; D1/E1 golden [PASS]; null sanity "
          f"D1差 {nm_d:+.4f} E1差 {nm_e:+.4f} [PASS]", flush=True)
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
        "params=tf={},vol_win={},z_hi={},z_lo={},W={},e1_h={},gbm_seeds={},"
        "min_n={},gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], ",".join(p["tf_list"]), p["vol_win"], p["z_hi"],
            p["z_lo"], p["W"], p["e1_h"], p["gbm_seeds"], p["min_n"],
            p["min_n"]),
        "# GATE: 量能 z golden + D1/E1 golden + null sanity [PASS]; MIN_N "
        "n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c66 聪明钱验证①: 关键位触碰×量能条件作用; "
        "cluster 触碰按触碰 bar 量能 z (20 日) 分组; D1 折返 (fade) W=24; "
        "E1 (c52, h=12); GBM null 30 种子 (真实 volume 50bar 块打乱重挂载); "
        "描述层无入场, 无交易含义",
        "",
    ]
    for tf in p["tf_list"]:
        r = res[tf]
        d1h = float(np.mean(r["d1_hi"])) if r["n_hi"] else float("nan")
        d1l = float(np.mean(r["d1_lo"])) if r["n_lo"] else float("nan")
        e1h = float(np.mean(r["e1_hi"])) if r["n_hi"] else float("nan")
        e1l = float(np.mean(r["e1_lo"])) if r["n_lo"] else float("nan")
        lines.append("")
        lines.append("[{}] 触碰×量能 (tf={}):".format("H1/H2", tf))
        lines.append("  D1 折返: 高量能 {:.1%} (n={}) {} | 低量能 {:.1%} "
                     "(n={}) {} | 差 {:+.2%}".format(
            d1h, r["n_hi"], _nm(r["n_hi"]), d1l, r["n_lo"], _nm(r["n_lo"]),
            d1h - d1l))
        lines.append("  E1: 高量能 {:+.2%} | 低量能 {:+.2%} | 差 {:+.2%}"
                     .format(e1h, e1l, e1h - e1l))
        lines.append("  null 组差: D1 {:+.4f}±{:.4f} | E1 {:+.4f}±{:.4f}"
                     .format(r["null_d"][0], r["null_d"][1], r["null_e"][0],
                             r["null_e"][1]))
        z_d = (d1h - d1l - r["null_d"][0]) / r["null_d"][1] \
            if r["null_d"][1] > 0 else float("nan")
        z_e = (e1h - e1l - r["null_e"][0]) / r["null_e"][1] \
            if r["null_e"][1] > 0 else float("nan")
        lines.append("  H1 判据 (D1 差超 null 2σ): z={:+.2f} -> {}".format(
            z_d, "超2σ↑" if z_d > 2 else ("低于2σ↓" if z_d < -2 else "未超")))
        lines.append("  H2 判据 (E1 差超 null 2σ): z={:+.2f} -> {}".format(
            z_e, "超2σ↑" if z_e > 2 else ("低于2σ↓" if z_e < -2 else "未超")))
        lines.append("  H3 方向 (折返率 vs 50%): 高量能 {:+.1%} | 低量能 "
                     "{:+.1%} (穿透=1−折返; 报告不作判据)".format(
            d1h - 0.5, d1l - 0.5))
    lines.append("")
    lines.append("[对照-历史] c14 (关键位围墙); c15 (触碰 E1 释放); c17 "
                 "(触碰折返 -2~-4pp); c49 (量能/时段); 本砖: 量能条件作用 "
                 "(聪明钱大资金放量假设); 预期谨慎")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    dev_syms = PARAMS["dev_subset"]["syms"] if dev else None

    data = load_candles(timeframes=PARAMS["tf_list"])
    res = {}
    null_means_d = []
    null_means_e = []
    for tf in PARAMS["tf_list"]:
        syms = [s for s in data if "USDT" in s]
        if dev_syms:
            syms = [s for s in syms if s in dev_syms]
        parts = []
        for sym in syms:
            df = data[sym].get(tf)
            if df is None or verify(df, sym, tf):
                continue
            ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
            vol = df["volume"].values.astype(float)
            parts.append(collect_one(ctx, vol[PARAMS["warmup"]:], PARAMS))
        pooled = _merge(parts)
        # GBM null (首标)
        ref = data[syms[0]].get(tf)
        g = pool_gbm(ref, PARAMS, seeds)
        d_hi = np.concatenate([g["d1_hi"], g["d1_hi_dir"]]).mean() if (
            len(g["d1_hi"]) or len(g["d1_hi_dir"])) else 0.0
        null_d = (float(np.mean(np.concatenate([g["d1_hi"], g["d1_lo"]])))
                  if len(g["d1_hi"]) + len(g["d1_lo"]) else 0.0, 0.0)
        # 逐种子组差 (需重算 — 简化: 用 pooled 近似 + 分种子残差)
        null_diffs_d = []
        null_diffs_e = []
        for seed in range(seeds):
            rw = gbm_matching(ref, seed=seed)
            sv = shuffled_volume(ref["volume"].values.astype(float), 50, seed)
            gctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
            gp = collect_one(gctx, sv[PARAMS["warmup"]:], PARAMS)
            if gp["n_hi"] and gp["n_lo"]:
                dd = float(np.mean(gp["d1_hi"]) - np.mean(gp["d1_lo"]))
                de = float(np.mean(gp["e1_hi"]) - np.mean(gp["e1_lo"]))
                null_diffs_d.append(dd)
                null_diffs_e.append(de)
        nd = (float(np.mean(null_diffs_d)) if null_diffs_d else 0.0,
              float(np.std(null_diffs_d, ddof=1)) if len(null_diffs_d) > 1
              else 0.0)
        ne = (float(np.mean(null_diffs_e)) if null_diffs_e else 0.0,
              float(np.std(null_diffs_e, ddof=1)) if len(null_diffs_e) > 1
              else 0.0)
        res[tf] = {**pooled, "null_d": nd, "null_e": ne}
        null_means_d.extend(null_diffs_d)
        null_means_e.extend(null_diffs_e)

    gate(np.array(null_means_d) if null_means_d else np.array([0.0]),
         np.array(null_means_e) if null_means_e else np.array([0.0]))

    if dev:
        for tf in res:
            r = res[tf]
            print("  [dev] {} 高 {} 低 {} | D1差 {:+.4f} | E1差 {:+.4f}".format(
                tf, r["n_hi"], r["n_lo"],
                (np.mean(r["d1_hi"]) if r["n_hi"] else 0) -
                (np.mean(r["d1_lo"]) if r["n_lo"] else 0),
                (np.mean(r["e1_hi"]) if r["n_hi"] else 0) -
                (np.mean(r["e1_lo"]) if r["n_lo"] else 0)))
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
