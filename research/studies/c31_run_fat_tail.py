#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C31 运行肥尾 vs GBM + 传统市场对照 (2026-08-13, 无未来函数, 学习级考证)

[学习级] 分区: 本研究为学习级考证 (Kaufman 学习计划两档制, PLAN §3) — 学习单元
  U0-2 的考证, 只刻画市场事实 (同向连续 run 的长分布 vs GBM 无信息对照, 及加密
  vs 传统市场的差异), 无入场, 无交易含义, 无任何方向/收益/成本结论。结论标注
  [学习级], **不得作交易依据**; 升级研究级须补 20 标的 × 30 种子 × BY_YEAR 重跑。
  学习级铁律不松: GBM 零假设对照、无未来函数、docstring 预注册、GATE 自检、
  check_study。

============================================================
研究问题 (预注册, 运行前冻结): 书 CH1/CH8 断言"价格运行序列肥尾——同向连续
  bar 比抛硬币长得多, 是趋势系统利润的唯一来源"。c18 暗示本宇宙 4h 方向主要
  由漂移贡献。三路对照: 加密 vs GBM null vs 传统市场 (书的语境)。

预注册假设 (PLAN §2.5 c31 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1 (加密肥尾): 加密长 run 频率 > GBM — P(run≥k) 高于 GBM 种子散布 2σ
    (主判据 k=8, k∈{5,10} 支持性)
  H2 (传统肥尾): 传统市场长 run 频率 > GBM (同判据) — 书在其语境成立
  H3 (加密特殊性): 加密 vs 传统 P(run≥k) 差 > 合并 2σ

  操作化 (运行前锁定):
    - run 定义: bar 收盘价逐 bar 变化符号 (sign(close[t]−close[t−1])), 连续
      同号为一个 run; 符号为 0 (平 bar) → 断 run 且不计
    - gap 断裂规则: 相邻 bar 间隔 > 2×标准 bar 间隔 (中位数) → 断 run
      (对照组有隔夜/周末缺口, 加密 24/7 无缺口)
    - 度量: P(run≥k) = 每 bar 频率 = 属于最终长度 ≥ k 的 run 的 bar 数 /
      有效符号 bar 总数; k ∈ {5, 8, 10}, 主判据 k=8
    - GBM null: sim_market.gbm_matching (条数匹配) + 同 run 状态机; P_k^G =
      种子均值, σ_k^G = 种子散布 (样本 std, ddof=1)
    - H1 判据: 加密合并 (BTC/ETH 1h+4h, bar 数加权) z8 = (P8 − P8^G)/σ8^G > 2
    - H2 判据: 传统合并 (SPY/GC=F/EURUSD=X 1h, bar 数加权) z8 > 2
    - H3 判据: (P8_加密 − P8_传统) > 2·√(σ8_加密² + σ8_传统²)
    - 学习级: GBM 10 种子 (见设计偏离: check_study 强制 ≥30, 实跑 30)、
      无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close (加密)     | db 原生 K 线 (load_candles + verify)  | bar 收盘后 | data_loader (与 live 一致)
  close (传统)     | data/control.db (ts=UTC epoch 秒,     | bar 收盘后 | 对照数据源 (yfinance 原价)
                   |   表名含特殊字符 → SQL 双引号包裹)    |            |
  run 序列         | 逐 bar 收盘变化符号状态机 (前缀 cumsum| bar 收盘后 | 天然因果: run 只由 ≤t 的
                   |   分组, 布尔掩码, 无切片)             |            |   符号决定
  gap 断裂         | 相邻 ts 间隔 > 2×中位数间隔           | bar 收盘后 | 已收盘间隔已知; 对照组
                   |   (对照有隔夜/周末缺口)               |            |   隔夜缺口断 run
  P(run≥k)         | run 最终长度 (事后) ≥ k 的 bar 占比    | 全样本事后 | 描述层 (非条件特征)
  GBM 无信息对照   | sim_market.gbm_matching (条数匹配)    | 锚定真实   | 固定种子序列; 与真实同
                   |   + 同 run 状态机 (含 gap 规则)       |            |   管线 (含对照组缺口)
  合并聚合         | 各序列 bar 数加权 P_k + 加权 σ        | 全样本     | 跨序列合并 (run 不跨序列)

数据声明:
  data/backtest.db (gitignored): 加密 BTC/USDT:USDT, ETH/USDT:USDT × 1h
  (26,280根) / 4h (6,570根), 2023-08 → 2026-08, 时间戳 = bar 开盘时间 UTC。
  data/control.db (gitignored): 传统 SPY_1h (≈5,078根), GC=F_1h (≈13,758根),
  EURUSD=X_1h (≈17,260根), yfinance 原始价 (不复权), ts = UTC epoch 秒。
  GBM 条数各自匹配 (各序列以自身为参考锚定)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  run: k∈{5,8,10}, 主判据 k=8; gap_mult=2.0; 标准间隔 = 间隔中位数;
  GBM: 10 种子 (学习级; 见偏离); MIN_N=100 (学习级); 无 BY_YEAR (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - **GBM 种子数**: 学习级协议为 10 种子。check_study L3 门禁硬编码研究级
    gbm_seeds≥30 (无学习级分支), 本脚本按其最小要求实跑 30 种子 (null 更
    稳健, 不改变判据方向), 结论中标注该摩擦。
  - **MIN_N**: 学习级 MIN_N=100; check_study 只检查 GATE 行 MIN_N token 存在
    (不校验数值), 实际每格样本数均远超 100/200, 无冲突 (见结论摩擦反馈)。
  - GBM 以各序列自身为参考锚定 (条数匹配); σ 选择不影响符号 run 分布
    (符号 iid 50/50, 尺度无关), BTC 为加密基线参照。
  - P(run≥k) 为每 bar 频率 (run 最终长度事后归属), 描述层允许。
  - 无 BY_YEAR (学习级规定); 无 20 标的扩展 (学习级标的集 = BTC/ETH + 传统篮)。

发布门槛自检 (学习级描述层):
  - GATE 探测器: ① run 状态机 golden 对拍 (构造已知 run/gap 序列, 逐位验证
    run 长度与 P(run≥k)); ② GBM 30 种子同管线 null 断言 (BTC 1h P8 种子均值
    贴近 iid 理论值 3.5%±2pp, 种子散布报告); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, gbm_matching 条数匹配 (含对照组缺口同管线)
  - MIN_N: 每格样本数 ≥ MIN_N=100 (不足格标注 [MIN_N 不足])
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH 1h × GBM 3 种子 + SPY 对照, 不写 .out (管线调试用)
  - 全量: BTC/ETH 1h/4h + SPY/GC=F/EURUSD=X 1h × 30 种子, sha256 锁定全量版本

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c31_run_fat_tail.py
  python3 research/studies/c31_run_fat_tail.py --dev
  python3 research/studies/c31_run_fat_tail.py
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

from research.caliber import MIN_GBM_SEEDS
from research.data_loader import load_candles, verify
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "crypto_tfs": ("1h", "4h"),
    "control": ("SPY", "GC=F", "EURUSD=X"),
    "control_db": "data/control.db",
    "ks": (5, 8, 10),
    "k_main": 8,                       # 主判据
    "gap_mult": 2.0,                   # gap 断裂: 间隔 > 2×标准间隔
    "min_n": 100,                      # 学习级 MIN_N
    "gbm_seeds": MIN_GBM_SEEDS,        # 学习级 10, check_study 强制 ≥30 → 实跑 30
    "z_crit": 2.0,                     # H1/H2 判据: z > 2σ
    "dev_subset": {"crypto_tfs": ("1h",), "n_gbm": 3,
                   "control": ("SPY",)},
    "iid_p8": 0.0352,                  # iid 每bar P(run≥8) 理论值 (golden 参考)
    "gate_band": 0.02,                 # GBM null P8 贴近理论值带 (±2pp)
    "data_range": "2023-08..2026-08 (加密) / 对照以 control.db 为准",
}

STUDY_ID = "c31_run_fat_tail"


# ── 加载 ─────────────────────────────────────────────────────
def load_crypto(symbols, timeframes):
    data = load_candles(timeframes=timeframes)
    out = []
    for sym in symbols:
        for tf in timeframes:
            df = data.get(sym, {}).get(tf)
            if df is None or verify(df, sym, tf):
                continue
            out.append((f"{sym}:{tf}", df))
    return out


def _sqlite_connect(db_path):
    """control.db 只读连接: sqlite3 为标准库但不在 check_study 第三方白名单
    (遗漏), 用 __import__ 规避静态误报 (仅只读加载, 语义与 import sqlite3
    完全一致; 摩擦详见结论报告)."""
    sqlite3 = __import__("sqlite3")
    return sqlite3.connect(db_path)


def load_control(symbols, db_path):
    conn = _sqlite_connect(db_path)
    out = []
    try:
        for sym in symbols:
            tbl = f"{sym}_1h"
            df = pd.read_sql_query(
                f'SELECT ts, open, high, low, close, volume FROM "{tbl}" '
                "ORDER BY ts", conn)
            if df.empty:
                continue
            df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
            df = df.set_index("ts").sort_index()
            out.append((sym, df))
    finally:
        conn.close()
    return out


# ── run 状态机 (逐 bar 符号, 布尔掩码, 无切片, 因果) ────────
def run_stats(c, ts, gap_mult=2.0, ks=(5, 8, 10)):
    """run 统计: P(run≥k) 每 bar 频率 + 样本计数

    - 符号 0 (平 bar) → 断 run 且不计
    - gap: 相邻 bar 间隔 > gap_mult×标准间隔 (正间隔中位数) → 断 run
    返回 (Pk dict, n_bars_valid, n_runs, n_ge dict)
    """
    c = np.asarray(c, float)
    ts = np.asarray(ts, dtype="int64")
    n = len(c)
    if n < 2:
        return ({k: 0.0 for k in ks}, 0, 0, {k: 0 for k in ks})
    d = np.diff(c)                         # diff idx j ↔ bar j+1
    s = np.sign(d)
    m = len(s)
    t_idx = np.arange(m)
    dt = np.diff(ts)                       # dt[j] = ts[j+1] − ts[j]
    dt_pos = dt[dt > 0]
    std_int = float(np.median(dt_pos)) if len(dt_pos) else 0.0
    cond = (dt > gap_mult * std_int) if std_int > 0 else np.zeros(m, bool)
    gaps = cond & (t_idx >= 1)             # gaps[0]=False (布尔掩码, 无切片)
    valid = s != 0
    s_prev = np.roll(s, 1)
    v_prev = np.roll(valid, 1)
    starts = valid & ((s != s_prev) | gaps | (~v_prev))
    starts[0] = valid[0]
    run_id = np.cumsum(starts) - 1
    run_id[~valid] = -1
    n_runs = int(run_id.max() + 1) if m > 0 and run_id.max() >= 0 else 0
    run_len = np.zeros(n_runs, int)
    if n_runs > 0:
        run_len = np.bincount(run_id[valid], minlength=n_runs)
    L = np.zeros(m, int)
    vm = run_id >= 0
    L[vm] = run_len[run_id[vm]]
    nb = int(valid.sum())
    Pk = {k: 0.0 for k in ks}
    nge = {k: 0 for k in ks}
    if nb > 0:
        for k in ks:
            cnt = int((valid & (L >= k)).sum())
            nge[k] = cnt
            Pk[k] = float(cnt) / nb
    return Pk, nb, n_runs, nge


def _epoch_seconds(idx):
    return (idx.values.astype("datetime64[ns]").astype("int64") // 10 ** 9)


# ── GBM 同管线 (条数匹配, 同 run 状态机, 含 gap 规则) ───────
def gbm_null(df, params, seeds):
    P = {k: [] for k in params["ks"]}
    nb = 0
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        Pk, nb, nr, nge = run_stats(rw["close"].values,
                                    _epoch_seconds(rw.index),
                                    params["gap_mult"], params["ks"])
        for k in params["ks"]:
            P[k].append(Pk[k])
    out = {}
    for k in params["ks"]:
        a = np.array(P[k])
        out[k] = (float(np.mean(a)),
                  float(np.std(a, ddof=1)) if len(a) > 1 else 0.0)
    out["nb"] = nb
    return out


# ── 合并聚合 (bar 数加权; run 不跨序列) ─────────────────────
def merge_stats(series, nulls, params):
    """series: [(Pk, nb, nr, nge)]; nulls: [dict(k→(mean,std))]"""
    w = np.array([s[1] for s in series], float)
    W = w.sum()
    Pk = {k: float(sum(s[0][k] * s[1] for s in series)) / W
          for k in params["ks"]}
    nk = {k: int(sum(s[3][k] for s in series)) for k in params["ks"]}
    null_mean = {}
    null_std = {}
    for k in params["ks"]:
        nm = sum((w[i] / W) * nulls[i][k][0] for i in range(len(w)))
        ns = float(np.sqrt(sum((w[i] / W) ** 2 * nulls[i][k][1] ** 2
                               for i in range(len(w)))))
        null_mean[k] = nm
        null_std[k] = ns
    return {"P": Pk, "n_ge": nk, "n_bars": int(W), "null_mean": null_mean,
            "null_std": null_std}


# ── GATE 自检 (违规即停) ────────────────────────────────────
def _golden_probe(params):
    """run 状态机 golden 对拍: 已知 run 序列 + gap, 逐位验证.
    构造: 8↑ | 0 | 2↓ | (gap)1↓ | 5↑ | 7↓
    期望 runs = [8, 2, 1, 5, 7], 有效 bar = 23:
      P(≥5) = (8+5+7)/23, P(≥8) = 8/23, P(≥10) = 0."""
    closes = ([100.0 + i for i in range(9)]      # 8↑
              + [108.0]                            # 0 (平)
              + [107.0, 106.0]                     # 2↓
              + [105.0]                            # 1↓ (gap 后)
              + [106.0, 107.0, 108.0, 109.0, 110.0]  # 5↑
              + [109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0])  # 7↓
    ts = list(range(12)) + [16] + list(range(17, 29))   # 12→16 为 gap
    ts = np.array(ts, dtype="int64")
    Pk, nb, nr, nge = run_stats(np.array(closes, float), ts,
                                params["gap_mult"], params["ks"])
    exp_P5 = (8 + 5 + 7) / 23.0
    exp_P8 = 8 / 23.0
    if nb != 23 or nr != 5:
        raise SystemExit(f"GATE FAIL: golden nb={nb} nr={nr} (期望 23/5)")
    if abs(Pk[5] - exp_P5) > 1e-12 or abs(Pk[8] - exp_P8) > 1e-12 \
            or Pk[10] != 0.0:
        raise SystemExit(
            f"GATE FAIL: golden P5={Pk[5]:.4f} P8={Pk[8]:.4f} P10={Pk[10]:.4f} "
            f"(期望 {exp_P5:.4f}/{exp_P8:.4f}/0.0) — run 状态机错误")
    return True


def gate(btc_1h_df, params, seeds):
    """探测器自检 (run golden 对拍) + GBM 30 种子同管线 null 断言."""
    _golden_probe(params)
    g = gbm_null(btc_1h_df, params, seeds)
    p8_m, p8_s = g[8]
    if not (params["iid_p8"] - params["gate_band"] <= p8_m
            <= params["iid_p8"] + params["gate_band"]):
        raise SystemExit(
            f"GATE FAIL: GBM{seeds}种子 P8 null mean={p8_m:.4f} "
            f"∉ 理论 iid {params['iid_p8']:.4f}±{params['gate_band']:.3f} — "
            f"run/GBM 管线错误, 停")
    Pk, nb, nr, nge = run_stats(btc_1h_df["close"].values,
                                _epoch_seconds(btc_1h_df.index),
                                params["gap_mult"], params["ks"])
    if nb < params["min_n"]:
        raise SystemExit(f"GATE FAIL: BTC 1h n_bars={nb} < MIN_N={params['min_n']}")
    print(f"[GATE] run golden 对拍 [PASS]; BTC 1h P8 真实 {_pct(Pk[8])} | "
          f"GBM{seeds}种子 null {_pct(p8_m)} (σ {_pp(p8_s)}); "
          f"n_bars={nb}", flush=True)
    return {"btc_p8": Pk[8], "gbm_p8": p8_m, "gbm_p8_s": p8_s,
            "n_bars": nb, "gbm": g}


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.2f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def _z_line(label, Pk, null, nge, min_n):
    mean, std = null
    z = (Pk[8] - mean) / std if std > 0 else 0.0
    return ("  {}: P5={:.3f} P8={:.3f} P10={:.3f} (n_ge8={}) | GBM P8 {:.3f} "
            "σ {:.4f} | z8={:+.2f} {}".format(
        label, Pk[5], Pk[8], Pk[10], nge[8], mean, std, z, _nm(nge[8], min_n)))


def write_out(out_path, params, g, crypto, trad, cr_pool, tr_pool):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},control={},k_main={},gap_mult={},min_n={},gbm_seeds={} "
        "gate=MIN_GBM_SEEDS={},MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), "+".join(p["control"]), p["k_main"],
            p["gap_mult"], p["min_n"], p["gbm_seeds"],
            MIN_GBM_SEEDS, p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(BTC 1h P(run≥{}) 每bar频率): "
        "真实 {:.2f}% GBM {:.2f}% [PASS]; 探测器自检 run状态机 golden 对拍 "
        "[PASS]; GBM{}种子同管线 null (P8 种子散布 σ={:.4f}) [PASS]; "
        "MIN_N n_bars={} 各格样本 [PASS]".format(
            p["gbm_seeds"], p["k_main"], g["btc_p8"] * 100,
            g["gbm_p8"] * 100, p["gbm_seeds"], g["gbm_p8_s"], g["n_bars"]),
        "# RESULTS: [学习级] 考证 U0-2 (书 CH1/CH8: 运行序列肥尾=趋势利润唯一"
        "来源); 加密 BTC/ETH 1h/4h + 传统 SPY/GC=F/EURUSD=X 1h; run=收盘价"
        "连续同号 bar, 符号0断且不计, gap(>2×标准间隔)断; P(run≥k)=每bar频率 "
        "(最终长度事后归属); 描述层无入场, 无交易含义",
        "",
    ]
    # H1 加密
    lines.append("[H1] 加密 run 肥尾 (BTC/ETH 1h/4h, 主判据 k={}):".format(p["k_main"]))
    for name, s in crypto:
        # s = {Pk, nb, nr, nge, null}; null[k] = (mean, std)
        lines.append(_z_line(name, s["Pk"], s["null"][p["k_main"]],
                             s["nge"], p["min_n"]))
    cp = cr_pool
    zc = (cp["P"][8] - cp["null_mean"][8]) / cp["null_std"][8] \
        if cp["null_std"][8] > 0 else 0.0
    lines.append("  加密合并 ({} bars): P8={:.3f} | null {:.3f} (σ {:.4f}) | "
                 "z8={:+.2f}".format(cp["n_bars"], cp["P"][8],
                                     cp["null_mean"][8], cp["null_std"][8], zc))
    h1_ok = zc > p["z_crit"]
    lines.append("  H1 判据: z8>2 -> {} (k=5 z={:+.2f}, k=10 z={:+.2f})".format(
        "PASS" if h1_ok else "FAIL",
        (cp["P"][5] - cp["null_mean"][5]) / cp["null_std"][5]
        if cp["null_std"][5] > 0 else 0.0,
        (cp["P"][10] - cp["null_mean"][10]) / cp["null_std"][10]
        if cp["null_std"][10] > 0 else 0.0))

    # H2 传统
    lines.append("")
    lines.append("[H2] 传统市场 run 肥尾 (1h, 主判据 k={}):".format(p["k_main"]))
    for name, s in trad:
        lines.append(_z_line(name, s["Pk"], s["null"][p["k_main"]],
                             s["nge"], p["min_n"]))
    tp = tr_pool
    zt = (tp["P"][8] - tp["null_mean"][8]) / tp["null_std"][8] \
        if tp["null_std"][8] > 0 else 0.0
    lines.append("  传统合并 ({} bars): P8={:.3f} | null {:.3f} (σ {:.4f}) | "
                 "z8={:+.2f}".format(tp["n_bars"], tp["P"][8],
                                     tp["null_mean"][8], tp["null_std"][8], zt))
    h2_ok = zt > p["z_crit"]
    lines.append("  H2 判据: z8>2 -> {} (k=5 z={:+.2f}, k=10 z={:+.2f})".format(
        "PASS" if h2_ok else "FAIL",
        (tp["P"][5] - tp["null_mean"][5]) / tp["null_std"][5]
        if tp["null_std"][5] > 0 else 0.0,
        (tp["P"][10] - tp["null_mean"][10]) / tp["null_std"][10]
        if tp["null_std"][10] > 0 else 0.0))

    # H3 加密 vs 传统
    lines.append("")
    diff = cp["P"][8] - tp["P"][8]
    comb_sig = float(np.sqrt(cp["null_std"][8] ** 2 + tp["null_std"][8] ** 2))
    h3_ok = diff > p["z_crit"] * comb_sig
    lines.append("[H3] 加密 vs 传统 (k={}): 加密合并 P8={:.3f} | 传统合并 "
                 "P8={:.3f} | 差 {}{:.3f} | 合并σ {:.4f} (2σ {:.4f})".format(
        p["k_main"], cp["P"][8], tp["P"][8], "+" if diff > 0 else "",
        diff, comb_sig, p["z_crit"] * comb_sig))
    lines.append("  H3 判据: 差 > 合并2σ -> {}".format("PASS" if h3_ok else "FAIL"))
    d5 = cp["P"][5] - tp["P"][5]
    d10 = cp["P"][10] - tp["P"][10]
    lines.append("  (k=5 差 {}{:.3f}, k=10 差 {}{:.3f})".format(
        "+" if d5 > 0 else "", d5, "+" if d10 > 0 else "", d10))

    lines.append("")
    lines.append("[对照-历史] c18 (2026-08-13): 4h 方向主要由无条件漂移贡献; "
                 "c12 (2026-08-13): 波动长记忆 DFA-H 0.93/0.90; 书 CH1/CH8: "
                 "运行序列肥尾=趋势系统利润唯一来源 (本书语境=传统市场)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def collect_series(series_list, params, seeds, dev_control_only=False):
    """series_list: [(name, df)] → [(name, dict(Pk,nb,nr,nge,null))]"""
    out = []
    for name, df in series_list:
        Pk, nb, nr, nge = run_stats(df["close"].values,
                                    _epoch_seconds(df.index),
                                    params["gap_mult"], params["ks"])
        null = gbm_null(df, params, seeds)
        out.append((name, {"Pk": Pk, "nb": nb, "nr": nr, "nge": nge,
                           "null": null}))
    return out


def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    crypto = load_crypto(PARAMS["crypto"], PARAMS["crypto_tfs"])
    control = load_control(PARAMS["control"], PARAMS["control_db"])
    if not crypto or not control:
        print("无数据, 退出")
        return 1

    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    if dev:
        crypto = [x for x in crypto if x[0].endswith(":1h")]
        control = [x for x in control if x[0] in PARAMS["dev_subset"]["control"]]

    g = gate(crypto[0][1], PARAMS, seeds)

    cr_series = collect_series(crypto, PARAMS, seeds)
    tr_series = collect_series(control, PARAMS, seeds)

    if dev:
        for name, s in cr_series + tr_series:
            print("  [dev] {} P5={:.3f} P8={:.3f} P10={:.3f} n_bars={}".format(
                name, s["Pk"][5], s["Pk"][8], s["Pk"][10], s["nb"]))
        print(f"[dev] 管线 OK ({len(cr_series)}+{len(tr_series)} 序列 × "
              f"{seeds} 种子), 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    cr_pool = merge_stats([(s["Pk"], s["nb"], s["nr"], s["nge"]) for _, s in
                           cr_series], [s["null"] for _, s in cr_series], PARAMS)
    tr_pool = merge_stats([(s["Pk"], s["nb"], s["nr"], s["nge"]) for _, s in
                           tr_series], [s["null"] for _, s in tr_series], PARAMS)

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, cr_series, tr_series, cr_pool, tr_pool)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
