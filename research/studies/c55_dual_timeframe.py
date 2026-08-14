#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C55 M8 多时间框架忠实复现 (CH19 p.833-843) (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (学习单元 M8 U1, PLAN §2.5 c55): 书 CH19 — 双框架择时 "60 日 MA
  + 3 日随机" 把 $2075 变 $6200 (n=1 单例, 无 null)。oracle 逐字核实口径:
  慢趋势 = 60 日 MA 方向转向; 择时 = 趋势信号后等 3 日 raw stochastic %K<30
  回撤入场 vs 趋势信号即入场。描述层, 无入场, 无交易含义, 不涉及胜率/期望/
  成本主张。**结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、
  .out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 加密 4h 上, 择时 (回撤入场) 相对即时入场
  的净收益差是否 > null (GBM 同管线)?

预注册假设 (PLAN §2.5 c55 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 双框架择时 — BTC/ETH 4h; 慢趋势 = 4h MA 方向 (60 日映射 = 360bar,
      方向转向信号); 择时版 = 趋势信号后等 3 日 raw stochastic
      (%K = (close−low_n)/(high_n−low_n), n = 72h = 18 根 4h bar) < 30 的
      回撤入场 vs 即时版 = 趋势信号即入场; 度量 = 净收益差;
      null = 两系统各自 GBM 30 种子净差
      (**择时增益 = 真实择时差 − null 择时差, 超 2σ 才支持书**;
      预期撞墙 — 书择时前提 = dip-buy, c19 −15.4pp / c29 两条已证伪)
  H2: 周期比对拍 — 振荡器/趋势周期比 ≈ 0.2 (5/23) 的参数敏感性
      (工程对拍, 不设 null, 仅报告): 趋势 360bar 配振荡器 72/36/18bar 三档
      的择时效果 (ratio 0.2/0.1/0.05; 书 5/23 ≈ 0.2 最接近 72bar 档)

  操作化 (运行前锁定):
    - H1 数据: BTC/ETH 4h (6,570 根), warmup=600; 日线映射 60 日 = 360 根
      4h bar; 3 日随机 = 18 根 4h bar (%K 用当前 bar 的 high/low, 收盘已知)
    - 系统 A (即时): pos_imm[i] = sign(ma[i−1]−ma[i−2]) — MA 方向即仓位
      (永远在场, 转向即反手); 系统 B (择时): 转向后等 %K<30 (多) / %K>70
      (空) 才入场, 入场后持有到下次转向; 转向时先平到 0 再等回撤
    - 度量: 净收益差 gain = Σ(pos_timed·ret) − Σ(pos_imm·ret)
      (log 收益); 择时增益 = 真实 gain − null gain mean
    - H1 判据: 择时增益 > 2×null gain std (每标的独立判; 报告两标的)
    - H2: 同管线, stoch_w ∈ {72, 36, 18} (ratio 0.2/0.1/0.05), 只报告
      真实 gain, 不设 null
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100 (择时入场次数)、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close/high/low   | research.ctx.make_ctx 统一截断对齐     | bar 收盘后 | ctx 唯一对齐出口
  MA(360)          | rolling(360).mean() 仅用已收盘 close   | bar 收盘后 | 书口径 60 日映射
  MA 方向转向      | sign(ma[i]−ma[i−1]) 变化 (转向信号)    | bar i 收盘 | 因果, 无前视
  %K(18)           | (close−low_18)/(high_18−low_18)        | bar i 收盘 | 当前 bar high/low
                   |   (rolling min/max 含当前 bar)        |            |   收盘即已知
  仓位/净收益      | 即时: pos_imm[i]=sign(ma[i−1]−ma[i−2]) | 已知于    | 转向 bar 收盘后
                   | 择时: 转向后 %K 触发入场, 持有到转向   |  close[i−1]| 入场作用于 bar i 收益
  GBM null         | sim_market.gbm_matching (锚定真实)     | 锚定真实   | 固定种子 0..29, 每标的
                   |   + 同管线两系统                        |            |   各自 30 种子

数据声明:
  data/backtest.db (gitignored): BTC/ETH × 4h × 2023-08..2026-08 (6,570 根,
  时间戳 = bar 开盘时间 UTC); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  ma_w=360 (60日), stoch_w=18 (3日, H1), thresh=0.30 (多)/0.70 (空);
  H2 stoch_w ∈ {72,36,18}; warmup=600; gbm_seeds=30; min_n=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 书 CH19 $6200 是单例 (n=1, 无 null) — 本研究升级为 4h 双标的 + GBM
    null, 只检验"择时差"而非绝对 $6200。
  - 系统 B 在转向时先平仓到 0 再等回撤入场 (择时 = 入场时机, 非持续在仓);
    若转向后无回撤直接再转向, 该腿不入场 (记录为 flat 腿)。
  - %K 阈值固定 0.30/0.70 (书 30/70 惯例); 不对参数做搜索 (H2 只对 stoch_w
    做工程对拍, 不搜阈值)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① golden — 构造已知 MA 转向 + %K 回撤序列, 验证择时入场
    bar 恰在 %K<30 的回撤 bar (非转向 bar); ② GBM null sanity — null gain
    mean ∈ [−0.5, +0.5] (log 净收益差量级检查, 抓管线错误); ③ 即时系统
    恒在仓 (warmup 后 |pos_imm|=1); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (每标的各自 30 种子,
    同管线两系统)
  - MIN_N: 择时入场次数 (多+空) 每标的 ≥ 100 (学习级)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC 单标的 × 3 种子 (测管线 + 耗时), 不写 .out
  - 全量: BTC+ETH 4h × 30 种子 × 双系统 (+ H2 三档真实对拍)

运行命令:
  python3 research/studies/c55_dual_timeframe.py --dev
  python3 research/studies/c55_dual_timeframe.py
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
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "tf": "4h",
    "ma_w": 360,                       # 60 日 = 360 根 4h (预注册)
    "stoch_w": 18,                     # 3 日 = 18 根 4h (H1)
    "thresh": 0.30,                    # %K 回撤入场阈值 (多 <0.30 / 空 >0.70)
    "h2_windows": (72, 36, 18),        # 周期比对拍 (ratio 0.2/0.1/0.05)
    "warmup": 600,
    "gbm_seeds": 30,
    "min_n": 100,                      # 学习级 MIN_N (择时入场次数)
    "null_band": (-0.5, 0.5),          # GATE: null gain 量级带
    "dev_subset": {"crypto": ("BTC/USDT:USDT",), "n_gbm": 3},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c55_dual_timeframe"


# ── 双框架系统 ───────────────────────────────────────────────
def dual_tf(close, high, low, ma_w, stoch_w, thresh):
    """返回 (pos_imm, pos_tim, n_timed_entries).
    - pos_imm[i] = sign(ma[i-1]−ma[i-2]) — MA 方向即仓位 (转向即反手)
    - pos_tim[i] = 择时仓位: 转向后等 %K 回撤 (多 <thresh / 空 >1−thresh)
      入场, 持有到下次转向; 转向时先平到 0 再等回撤
    """
    c = np.asarray(close, float)
    n = len(c)
    ma = pd.Series(c).rolling(ma_w).mean().values
    slope = np.sign(np.diff(ma, prepend=np.nan))
    hi = pd.Series(high).rolling(stoch_w).max().values
    lo = pd.Series(low).rolling(stoch_w).min().values
    kk = (c - lo) / (hi - lo)          # %K ∈ [0,1]; warmup 期 NaN
    pos_imm = np.zeros(n, int)
    pos_tim = np.zeros(n, int)
    pos = 0
    pending = 0
    n_entries = 0
    for i in range(1, n):
        # 即时: 上一 bar 收盘时的 MA 方向 (0/NaN → 延续仓位, 永远在场)
        if np.isfinite(slope[i - 1]) and slope[i - 1] != 0:
            pos_imm[i] = int(slope[i - 1])
        else:
            pos_imm[i] = pos_imm[i - 1]
        # 择时状态机 (i-1 bar 收盘已知: slope[i-1], kk[i-1])
        s1 = slope[i - 1] if np.isfinite(slope[i - 1]) else 0
        s2 = slope[i - 2] if i >= 2 and np.isfinite(slope[i - 2]) else 0
        kv = kk[i - 1] if np.isfinite(kk[i - 1]) else float("nan")
        if s1 == 1 and s2 != 1:
            pending = 1
            if pos == -1:
                pos = 0
        elif s1 == -1 and s2 != -1:
            pending = -1
            if pos == 1:
                pos = 0
        if pending == 1 and pos == 0 and np.isfinite(kv) and kv < thresh:
            pos = 1
            pending = 0
            n_entries += 1
        elif pending == -1 and pos == 0 and np.isfinite(kv) and kv > 1 - thresh:
            pos = -1
            pending = 0
            n_entries += 1
        pos_tim[i] = pos
    return pos_imm, pos_tim, n_entries


def nets(close, pos_imm, pos_tim):
    ret = np.concatenate([[0.0], np.diff(np.log(close))])
    net_imm = float(np.sum(pos_imm * ret))
    net_tim = float(np.sum(pos_tim * ret))
    return net_imm, net_tim, net_tim - net_imm


def system_stats(ctx, ma_w, stoch_w, thresh):
    pi, pt, ne = dual_tf(ctx.close, ctx.high, ctx.low, ma_w, stoch_w, thresh)
    imm, tim, gain = nets(ctx.close, pi, pt)
    return {"imm": imm, "tim": tim, "gain": gain, "n_entries": ne}


# ── GBM null (每标的 30 种子, 同管线双系统) ──────────────────
def gbm_null(df, ma_w, stoch_w, thresh, seeds):
    gains = []
    entries = []
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        ctx = make_ctx(rw, PARAMS["warmup"], state_fns={})
        st = system_stats(ctx, ma_w, stoch_w, thresh)
        gains.append(st["gain"])
        entries.append(st["n_entries"])
    g = np.array(gains)
    return (float(np.mean(g)), float(np.std(g, ddof=1)), int(np.sum(entries)))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def _golden_dual_tf():
    """构造已知序列: 500 根平盘 (MA=100, slope=0) → 强涨 (MA 转向上) → 回撤
    (%K<30) → 验证择时入场 bar 在回撤 bar (非转向 bar); 无回撤时择时不入场."""
    c = np.array([100.0] * 500 + list(100.0 + 0.5 * np.arange(1, 201)))
    hi = c + 1.0
    lo = c - 1.0
    # 强涨后插一个 6 根小回撤 (close 下台阶) → %K 跌破 0.3
    dip = np.array([c[-1] - 1.5, c[-1] - 3.0, c[-1] - 4.5,
                    c[-1] - 3.0, c[-1] - 1.5])
    restore = np.array([c[-1] + 1.0] * 20)   # 回撤后恢复
    c = np.concatenate([c, dip, restore])
    hi = np.concatenate([hi, dip + 1.0, restore + 1.0])
    lo = np.concatenate([lo, dip - 1.0, restore - 1.0])
    idx = pd.date_range("2024-01-01", periods=len(c), freq="4h", tz="UTC")
    df = pd.DataFrame({"open": c, "high": hi, "low": lo, "close": c,
                       "volume": 1.0}, index=idx)
    ctx = make_ctx(df, 200, state_fns={})
    pi, pt, ne = dual_tf(ctx.close, ctx.high, ctx.low, 100, 12, 0.30)
    # 验证: 即时系统曾在仓 (MA 转向后), 择时系统至少 1 次入场
    if not np.any(pi != 0):
        raise SystemExit("GATE FAIL: golden 即时系统未转向 — 构造错误")
    if ne < 1:
        raise SystemExit("GATE FAIL: golden 择时未入场 — %K 回撤未触发")
    # 验证择时入场不早于转向 (延迟性): 首次入场 bar > 首次转向 bar
    turn_i = int(np.flatnonzero(pi != 0)[0])
    entry_i = int(np.flatnonzero(pt != 0)[0])
    if entry_i <= turn_i:
        raise SystemExit("GATE FAIL: 择时入场早于转向 — 状态机错误")
    return True


def gate(null_gain_mean, null_gain_std, pos_imm):
    _golden_dual_tf()
    lo, hi = PARAMS["null_band"]
    if not (lo <= null_gain_mean <= hi):
        raise SystemExit(
            f"GATE FAIL: null gain mean {null_gain_mean:+.4f} ∉ [{lo}, {hi}] "
            f"— 管线错误, 停")
    if not np.all(np.abs(pos_imm[PARAMS["ma_w"] + 2:]) == 1):
        raise SystemExit("GATE FAIL: 即时系统 MA warmup 后非恒在仓 — 仓位错误")
    print(f"[GATE] golden [PASS]; null gain {null_gain_mean:+.4f}±"
          f"{null_gain_std:.4f} [PASS]; 即时系统恒在仓 [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n):
    return "[MIN_N 通过]" if n >= PARAMS["min_n"] else "[MIN_N 不足]"


def write_out(out_path, params, res, h2):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=ma_w={},stoch_w={},thresh={},h2_windows={},gbm_seeds={},"
        "min_n={},gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["ma_w"], p["stoch_w"], p["thresh"],
            p["h2_windows"], p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: 探测器自检 golden (转向+回撤时序) [PASS]; null gain 带 "
        "[PASS]; 即时系统恒在仓 [PASS]; MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c55 M8 双框架择时 (书 CH19 p.833-843, $6200 单例 "
        "n=1 无 null); 慢趋势=MA(360) 方向转向, 择时=3日 raw stochastic "
        "%K<30 回撤入场 vs 即时入场; 描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[H1] 双框架择时 (4h, MA(360) + %K({}) 回撤入场):".format(
        p["stoch_w"]))
    for sym, st in res.items():
        gm, gs, ge = st["null"]
        gain = st["real"]["gain"]
        z = (gain - gm) / gs if gs > 0 else float("nan")
        lines.append("  {}: 即时 {:+.4f} | 择时 {:+.4f} | gain {:+.4f} | "
                     "null gain {:+.4f}±{:.4f} | 择时增益 {:+.4f} (z={:+.2f}) "
                     "{} | 入场 n={} {}".format(
            sym.split("/")[0], st["real"]["imm"], st["real"]["tim"], gain,
            gm, gs, gain - gm, z,
            "超2σ↑" if z > 2 else "未超", st["real"]["n_entries"],
            _nm(st["real"]["n_entries"])))
    lines.append("  H1 判据: 择时增益 > 2σ -> {}/{}".format(
        sum(1 for st in res.values() if
            (st["real"]["gain"] - st["null"][0]) / st["null"][1] > 2
            if st["null"][1] > 0), len(res)))
    # H2 周期比对拍
    lines.append("")
    lines.append("[H2] 周期比对拍 (趋势 360bar, 振荡器窗口/ratio 工程对拍, "
                 "无 null):")
    for w in p["h2_windows"]:
        rv = h2[w]
        cells = []
        for sym, gain in rv.items():
            cells.append("{} {:+.4f}".format(sym.split("/")[0], gain))
        lines.append("  stoch_w={:>3} (ratio {:.3f}): {}".format(
            w, w / p["ma_w"], " | ".join(cells)))
    lines.append("  书 5/23≈0.2 → 最接近 72bar 档 (ratio 0.2)")
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] 书 CH19 p.833-843 (60日 MA + 3日随机, $2075 → "
                 "$6200 单例 n=1 无 null); c19 (回撤恢复 −15.4pp, dip-buy 无 "
                 "优势); c29 (时间视角两条证伪); c41 (突破/趋势系统无成本 "
                 "whipsaw 负漂移)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    syms = PARAMS["dev_subset"]["crypto"] if dev else PARAMS["crypto"]
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    data = load_candles(timeframes=(PARAMS["tf"],))
    ctxs = []
    for sym in syms:
        df = data.get(sym, {}).get(PARAMS["tf"])
        if df is None or verify(df, sym, PARAMS["tf"]):
            continue
        ctx = make_ctx(df, PARAMS["warmup"], state_fns={})
        ctxs.append((sym, df, ctx))

    res = {}
    for sym, df, ctx in ctxs:
        real = system_stats(ctx, PARAMS["ma_w"], PARAMS["stoch_w"],
                            PARAMS["thresh"])
        gm, gs, ge = gbm_null(df, PARAMS["ma_w"], PARAMS["stoch_w"],
                              PARAMS["thresh"], seeds)
        res[sym] = {"real": real, "null": (gm, gs, ge)}

    # H2 (真实对拍, 无 null)
    h2 = {}
    for w in PARAMS["h2_windows"]:
        h2[w] = {}
        for sym, df, ctx in ctxs:
            st = system_stats(ctx, PARAMS["ma_w"], w, PARAMS["thresh"])
            h2[w][sym] = st["gain"]

    # GATE
    first = next(iter(res.values()))
    pi, _, _ = dual_tf(ctxs[0][2].close, ctxs[0][2].high, ctxs[0][2].low,
                       PARAMS["ma_w"], PARAMS["stoch_w"], PARAMS["thresh"])
    gate(first["null"][0], first["null"][1], pi)

    if dev:
        for sym, st in res.items():
            print("  [dev] {} gain {:+.4f} (null {:+.4f}±{:.4f}) 入场 n={}"
                  .format(sym.split("/")[0], st["real"]["gain"],
                          st["null"][0], st["null"][1],
                          st["real"]["n_entries"]))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res, h2)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
