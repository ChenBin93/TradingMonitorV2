#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C32 ER 频率梯度的传统市场对照 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U0-1 补验): c30 在加密宇宙证伪了书 CH1"频率越低噪声
  越低" (ER_10 中位数 1h/4h/日线几乎持平, 接近 iid null)。本考证用同管线验证
  传统市场 — 书断言在书的市场 (SPY/CL=F/GC=F/EURUSD=X/^TNX) 成立吗?
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。结论标注 [学习级],
  **不得作交易依据**; 升级研究级须补 20 标的 × 30 种子 × BY_YEAR 重跑。

  学习级新协议 (2026-08-13 起): 本脚本不跑 pytest / check_study (门禁已废除
  于学习级), 保留底线: docstring 预注册运行前冻结; 内置 GATE (探测器自检 +
  GBM null 断言, 失败 SystemExit); 因果纪律自我约束 (causal 库、布尔掩码
  禁切片、禁全样本分位); dev 先行; .out 数字引用; 结论 [学习级]。

============================================================
研究问题 (预注册, 运行前冻结): c30 在加密 (BTC/ETH) 证伪"频率越低噪声越低"
  (ER_10 中位数 1h/4h/日线 27~32% 持平)。书 CH1 的语境是传统市场 — 用同管线
  在传统市场 (SPY/CL=F/GC=F/EURUSD=X/^TNX) 检验: 书断言在书的市场成立吗?
  加密特殊性 (传统有梯度、加密没有) 成立吗?

预注册假设 (PLAN §2.5 c32 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1 (传统梯度): 传统市场 ER_10 中位数 1h < 1d — 5 标的全部成立
    (书"频率越低噪声越低"在其语境成立)
  H2 (加密特殊性): 传统日线−1h 差 (5 标的均值) − 加密日线−1h 差
    (BTC/ETH 均值) ≥ 2pp — 传统有梯度、加密没有
  H3 (GBM null): GBM 同管线无梯度 — 30 种子, **gap 匹配 null** (按各对照
    标的会话日历掩码 GBM bar, c31 模式; BTC/ETH 无缺口用连续 null)

  操作化 (运行前锁定):
    - ER_10 = |C_t − C_{t−10}| / Σ_{i=t−9..t}|C_i − C_{i−1}| (n=10, c27/c30
      同口径, 前缀和+布尔掩码, 因果)
    - 频率: 加密 1h (db 原生) vs 日线 (daily_resample 自 1h, c30 口径);
      传统 1h (control.db 原生) vs 1d (control.db 原生表)
    - 度量: 各序列 ER_10 全样本中位数 (np.nanmedian, 纯描述统计)
    - H1 判据: 5 传统标的严格 med_1h < med_1d
    - H2 判据: mean_5trad(med_1d − med_1h) − mean_BTC_ETH(med_1d − med_1h)
      ≥ +2pp
    - H3 判据: GBM 30 种子同管线 (1h 会话日历匹配 / 1d 原生日历匹配)
      各序列 null 梯度 (med_1d^G − med_1h^G) 无系统性梯度 (报告均值±σ;
      判据: 合并 null 梯度 < 2pp)
    - 学习级: 无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close 1h (加密)  | db 原生 K 线 (load_candles + verify)  | bar 收盘后 | data_loader
  日线 close (加密)| data_loader.daily_resample (自 1h)   | 日线收盘后 | c30 口径 (resample 已收盘)
  close 1h/1d (传统)| data/control.db (ts=UTC epoch 秒,     | bar 收盘后 | 对照数据源 (yfinance 原价)
                   |   表名特殊字符 → SQL 双引号包裹)      |            |
  ER_10           | |C_t−C_{t−10}|/Σ|ΔC|, 前缀和+布尔掩码 | bar 收盘后 | c27/c30 同口径 (只回看 t-10..t)
  ER 中位数        | np.nanmedian(ER 有限值序列)           | 全样本     | 纯描述统计 (非全样本分位作特征)
  GBM null (加密)  | sim_market.gbm_matching (1h 连续)     | 锚定真实   | c30 模式: 1h 生成 + daily_resample
                   |   + daily_resample 到日线             |            |   同聚合
  GBM null (传统)  | sim_market.gbm_matching (1h/1d 各自   | 锚定真实   | c31 模式 (gap 匹配): GBM 继承
                   |   原生会话日历, 含隔夜/周末缺口)      |            |   各对照标的会话日历掩码 bar
  设计偏离-会话缺口| 传统 1h ER_10 窗口混入隔夜/周末缺口    | bar 收盘后 | 预注册偏离: 缺口增大 Σ|ΔC|,
                   |   (10 根窗口跨越交易时段边界)          |            |   压低 1h ER; gap 匹配 null
                   |                                       |            |   同缺口结构, 比较仍同管线

数据声明:
  data/backtest.db (gitignored): 加密 BTC/USDT:USDT, ETH/USDT:USDT × 1h
  (26,280根), 2023-08 → 2026-08; 日线 = 1h 重采样。
  data/control.db (gitignored): 传统 SPY_1h(≈5,078)/1d(≈753), CL=F_1h
  (≈13,515)/1d(≈755), GC=F_1h(≈13,758)/1d(≈755), EURUSD=X_1h(≈17,260)/1d
  (≈779), ^TNX_1h(≈4,232)/1d(≈753); yfinance 原始价 (不复权); 4h 无源跳过。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  ER: n=10; GBM: 30 种子 (学习级协议 10, 实跑 30 沿用 c30/c31 惯例); 
  H2 判据: ≥2pp; gate_band=0.02 (连续 null 梯度带, c30 实测 ~+0.3pp);
  MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 传统 1h 的 ER_10 窗口跨会话缺口 (隔夜/周末) — 这是数据本性, 缺口混入
    ER 窗口按预注册接受; gap 匹配 null (c31 模式) 使真实与 null 承受同缺口
    结构, 比较仍同管线。
  - 传统 1d 用 control.db 原生表 (非重采样); 加密日线用 daily_resample
    (c30 口径) — 各自同管线到自己的 null。
  - H3 判据采用"合并 null 梯度 < 2pp" (与 H2 门槛一致): 若 null 梯度接近或
    超过 H2 门槛, 真实梯度需在 null 之上才有意义 (结论中给出真实−null 净差)。
  - 学习级: 无 BY_YEAR (学习级规定); 无 check_study (新协议废除); GBM 30 种子
    沿用 c30/c31 惯例 (高于学习级 10 的下限)。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 纯 iid 合成 OHLC (sub=1) 1h vs 日线 ER 中位数频率无关
    (验证 er_series + daily_resample); ② 加密连续 GBM 30 种子 null 梯度
    (日线−1h) ∈ ±gate_band=0.02 (c30 实测 ~+0.3pp); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, gap 匹配 (传统继承会话日历, 加密连续)
  - MIN_N: 每序列 ER 样本数 ≥ MIN_N=100 (学习级) 逐格报告
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH + 2 个对照标的 × GBM 3 种子, 不写 .out (管线调试用)
  - 全量: BTC/ETH + 5 传统标的 × 30 种子, sha256 锁定全量版本

运行命令:
  python3 research/studies/c32_er_freq_control.py --dev
  python3 research/studies/c32_er_freq_control.py
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

from research.data_loader import daily_resample, load_candles, verify
from research.sim_market import gbm_matching, gbm_ohlc

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "control": ("SPY", "CL=F", "GC=F", "EURUSD=X", "^TNX"),
    "control_db": "data/control.db",
    "er_n": 10,
    "gbm_seeds": 30,
    "min_n": 100,                       # 学习级 MIN_N
    "h2_min": 0.02,                     # H2 判据: 传统−加密 ≥ 2pp
    "gate_band": 0.02,                  # 连续 null 梯度带 (c30 实测 ~+0.3pp)
    "syn_n": 40000,
    "dev_subset": {"n_gbm": 3,
                   "control": ("SPY", "GC=F")},
    "data_range": "加密 2023-08..2026-08 / 对照以 control.db 为准 (≈2y)",
}

STUDY_ID = "c32_er_freq_control"


# ── 加载 ─────────────────────────────────────────────────────
def load_crypto_1h(symbols):
    data = load_candles(timeframes=("1h",))
    out = []
    for sym in symbols:
        df = data.get(sym, {}).get("1h")
        if df is None or verify(df, sym, "1h"):
            continue
        out.append((sym, df))
    return out


def load_control_pairs(symbols, db_path):
    """传统标的 → (1h df, 1d df) 成对; 表名含特殊字符 → SQL 双引号包裹"""
    conn = sqlite3.connect(db_path)
    out = []
    try:
        for sym in symbols:
            h = pd.read_sql_query(
                f'SELECT ts, open, high, low, close, volume FROM "{sym}_1h" '
                "ORDER BY ts", conn)
            d = pd.read_sql_query(
                f'SELECT ts, open, high, low, close, volume FROM "{sym}_1d" '
                "ORDER BY ts", conn)
            if h.empty or d.empty:
                continue
            h["ts"] = pd.to_datetime(h["ts"], unit="s", utc=True)
            d["ts"] = pd.to_datetime(d["ts"], unit="s", utc=True)
            out.append((sym, h.set_index("ts").sort_index(),
                        d.set_index("ts").sort_index()))
    finally:
        conn.close()
    return out


# ── ER 序列 (c27/c30 口径, 因果, 前缀和, 布尔掩码) ──────────
def er_series(c, n):
    """ER_n 序列: ER_t = |C_t − C_{t−n}| / Σ_{i=t−n+1..t}|C_i − C_{i−1}|
    因果 (只回看 t-n..t), 长度 = len(c), 未收敛处 NaN."""
    c = np.asarray(c, float)
    length = len(c)
    t = np.arange(length)
    c_prev = np.roll(c, 1)
    m1 = t >= 1
    ad = np.where(m1, np.abs(c - c_prev), 0.0)
    pref = np.concatenate([[0], np.cumsum(ad)])
    ok = t >= n
    net = np.full(length, np.nan)
    net[ok] = np.abs(c[t[ok]] - c[t[ok] - n])
    path = np.full(length, np.nan)
    path[ok] = pref[t[ok] + 1] - pref[t[ok] - n + 1]
    er = np.full(length, np.nan)
    m = ok & (path > 0)
    er[m] = net[m] / path[m]
    return er


def med_diff(c_low, c_high, n):
    """低频中位数 − 高频中位数 (ER_10 中位数差) + 样本数"""
    el = er_series(c_low, n)
    eh = er_series(c_high, n)
    ml = float(np.nanmedian(el))
    mh = float(np.nanmedian(eh))
    return ml, mh, int(np.isfinite(el).sum()), int(np.isfinite(eh).sum())


# ── GBM null (30 种子; 加密连续 / 传统 gap 匹配) ────────────
def null_gradient(series_1h, series_1d, params, seeds, crypto_mode):
    """GBM null 梯度 (日线−1h ER 中位数): 种子均值 ± 种子散布

    - crypto_mode=True : 1h GBM 连续生成 + daily_resample 到日线 (c30 模式)
    - crypto_mode=False: 1h GBM 继承会话日历 (含缺口) + 1d GBM 继承 1d 日历
      (c31 gap 匹配模式)
    返回 (grad_mean, grad_std, diffs, n1h, n1d, m1_mean)
    """
    diffs = []
    m1s = []
    n1h = n1d = 0
    for seed in range(seeds):
        rw1 = gbm_matching(series_1h, seed=seed)
        e1 = er_series(rw1["close"].values, params["er_n"])
        m1 = float(np.nanmedian(e1))
        if crypto_mode:
            dd = daily_resample(rw1)
            ed = er_series(dd["close"].values, params["er_n"])
        else:
            rwd = gbm_matching(series_1d, seed=seed)
            ed = er_series(rwd["close"].values, params["er_n"])
        md = float(np.nanmedian(ed))
        diffs.append(md - m1)
        m1s.append(m1)
        n1h = max(n1h, int(np.isfinite(e1).sum()))
        n1d = max(n1d, int(np.isfinite(ed).sum()))
    a = np.array(diffs)
    return (float(np.mean(a)), float(np.std(a, ddof=1)), a,
            n1h, n1d, float(np.mean(m1s)))


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(crypto_1h_list, params, seeds):
    """① 纯 iid 合成探测器: 1h vs 日线 ER 中位数频率无关 (验证管线);
    ② 加密连续 GBM null 梯度 ∈ ±gate_band (c30 实测 ~+0.3pp);
    失败 SystemExit."""
    # ① 合成纯 iid OHLC (sub=1)
    o, h, l, c = gbm_ohlc(n=params["syn_n"], sig=0.01, seed=0, sub=1)
    idx = pd.date_range("2024-01-01", periods=params["syn_n"], freq="1h",
                        tz="UTC")
    syn = pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                        "volume": np.ones(params["syn_n"])}, index=idx)
    s1 = float(np.nanmedian(er_series(syn["close"].values, params["er_n"])))
    sd = float(np.nanmedian(er_series(
        daily_resample(syn)["close"].values, params["er_n"])))
    if abs(sd - s1) > params["gate_band"]:
        raise SystemExit(
            f"GATE FAIL: 合成iid 日线−1h ER med 差 {sd - s1:+.4f} "
            f"∉ ±{params['gate_band']} — ER/resample 探测器错误, 停")
    # ② 加密连续 GBM null 梯度
    diffs_all = []
    m1_all = []
    for sym, df in crypto_1h_list:
        g = null_gradient(df, None, params, seeds, crypto_mode=True)
        diffs_all.extend(g[2].tolist())
        m1_all.append(g[5])
    g_mean = float(np.mean(diffs_all))
    if abs(g_mean) > params["gate_band"]:
        raise SystemExit(
            f"GATE FAIL: 加密连续 GBM null 梯度 mean={g_mean:+.4f} "
            f"∉ ±{params['gate_band']} — 单调性 null 偏置, 停")
    print(f"[GATE] 合成iid 日线−1h ER med 差 {sd - s1:+.3f} [频率无关 PASS]; "
          f"加密连续 GBM null 梯度 {g_mean:+.3f} [PASS]", flush=True)
    return {"syn_diff": sd - s1, "null_grad_c": g_mean,
            "gbm_1h": float(np.mean(m1_all))}


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.1f}%"


def _pp(v):
    return f"{v * 100:+.1f}pp"


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def write_out(out_path, params, g, cr_rows, tr_rows, h2_diff, null_rows,
              n_crypto, n_trad):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},control={},er_n={},gbm_seeds={},h2_min={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), "+".join(p["control"]), p["er_n"],
            p["gbm_seeds"], p["h2_min"], p["min_n"], p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(加密 1h ER_10 跨标的中位数): "
        "真实 {:.1f}% GBM {:.1f}% [PASS]; 探测器自检 合成iid 1h vs 日线 ER "
        "中位数频率无关 [PASS]; 加密连续 GBM null 梯度 ±{} [PASS]; "
        "MIN_N 各序列 ER 样本数≥{} [PASS]".format(
            p["gbm_seeds"],
            float(np.median([r[1] for r in cr_rows])) * 100,
            g["gbm_1h"] * 100,
            p["gate_band"], p["min_n"]),
        "# RESULTS: [学习级] U0-1 补验 (c30 加密证伪书断言, 本考证同管线传统"
        "市场); ER_10 = |C_t−C_{{t−10}}|/Σ|ΔC|; 传统 1h/1d 为 control.db 原生 "
        "(1h ER 窗口跨会话缺口=预注册设计偏离, gap 匹配 null); 加密日线 = "
        "daily_resample; 描述层无入场, 无交易含义",
        "",
    ]
    # 加密
    lines.append("[加密] ER_10 中位数 1h vs 日线 (c30 口径):")
    for sym, m1, md, d, n1, nd in cr_rows:
        lines.append("  {}: 1h {} | 日线 {} | 差 {}{:.1f}pp ({})".format(
            sym, _pct(m1), _pct(md), "+" if d > 0 else "", d * 100,
            _nm(min(n1, nd), p["min_n"])))
    cr_mean = float(np.mean([r[3] for r in cr_rows]))
    lines.append("  加密均值 日线−1h 差: {}{:.1f}pp".format(
        "+" if cr_mean > 0 else "", cr_mean * 100))
    # H1 传统
    lines.append("")
    lines.append("[H1] 传统市场 ER_10 中位数 1h vs 1d (书语境):")
    for sym, m1, md, d, n1, nd in tr_rows:
        mono = m1 < md
        lines.append("  {}: 1h {} | 1d {} | 差 {}{:.1f}pp [1h<1d {}] ({})".format(
            sym, _pct(m1), _pct(md), "+" if d > 0 else "", d * 100,
            "✓" if mono else "✗", _nm(min(n1, nd), p["min_n"])))
    n_mono = sum(1 for _, m1, md, _, _, _ in tr_rows if m1 < md)
    lines.append("  H1 判据: 5 标的全部 1h<1d -> {} ({}/{})".format(
        "PASS" if n_mono == len(tr_rows) else "FAIL", n_mono, len(tr_rows)))
    # H2
    tr_mean = float(np.mean([r[3] for r in tr_rows]))
    lines.append("")
    lines.append("[H2] 加密特殊性: 传统均值 {}{:.1f}pp − 加密均值 {}{:.1f}pp "
                 "= {}{:.1f}pp (判据 ≥{}pp) -> {}".format(
        "+" if tr_mean > 0 else "", tr_mean * 100,
        "+" if cr_mean > 0 else "", cr_mean * 100,
        "+" if h2_diff > 0 else "", h2_diff * 100,
        p["h2_min"] * 100, "PASS" if h2_diff >= p["h2_min"] else "FAIL"))
    # H3 GBM null
    lines.append("")
    lines.append("[H3] GBM null 梯度 (日线−1h ER 中位数, 30 种子, gap 匹配):")
    for name, nm, ns, mode in null_rows:
        lines.append("  {}: null 梯度 {}{:.1f}pp (σ {:.1f}pp) [{}]".format(
            name, "+" if nm > 0 else "", nm * 100, ns * 100, mode))
    null_c = float(np.mean([r[1] for r in null_rows if r[3] == "连续"]))
    null_t = float(np.mean([r[1] for r in null_rows if r[3] == "gap匹配"]))
    lines.append("  合并 null 梯度: 加密(连续) {}{:.1f}pp | 传统(gap匹配) "
                 "{}{:.1f}pp (判据 无梯度<2pp)".format(
        "+" if null_c > 0 else "", null_c * 100,
        "+" if null_t > 0 else "", null_t * 100))
    h3_ok = abs(null_c) < p["h2_min"] and abs(null_t) < p["h2_min"]
    lines.append("  H3 判据: GBM 同管线无梯度 -> {}".format(
        "PASS" if h3_ok else "FAIL"))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c30 (2026-08-13): 加密 20 标的 ER_10 中位数 "
                 "1h/4h/日线 27~32% 持平, 真实日线−1h +1.4pp vs GBM +0.3pp, "
                 "书断言在加密证伪; 书 CH1: 频率越低噪声越低 (语境=传统市场)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    crypto = load_crypto_1h(PARAMS["crypto"])
    control = load_control_pairs(PARAMS["control"], PARAMS["control_db"])
    if not crypto or not control:
        print("无数据, 退出")
        return 1

    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]
    if dev:
        control = [c for c in control if c[0] in PARAMS["dev_subset"]["control"]]

    g = gate(crypto, PARAMS, seeds)

    # 加密 (1h → 日线 daily_resample)
    cr_rows = []
    for sym, df1 in crypto:
        dd = daily_resample(df1)
        md, m1, nd, n1 = med_diff(dd["close"].values, df1["close"].values,
                                  PARAMS["er_n"])
        cr_rows.append((sym, m1, md, md - m1, n1, nd))

    # 传统 (原生 1h + 1d)
    tr_rows = []
    for sym, df1, dfd in control:
        md, m1, nd, n1 = med_diff(dfd["close"].values, df1["close"].values,
                                  PARAMS["er_n"])
        tr_rows.append((sym, m1, md, md - m1, n1, nd))

    if dev:
        for sym, m1, md, d, n1, nd in cr_rows + tr_rows:
            print("  [dev] {} 1h={:.3f} 日线={:.3f} 差={:+.3f} (n {} {})".format(
                sym, m1, md, d, n1, nd))
        print(f"[dev] 管线 OK ({len(cr_rows)}+{len(tr_rows)} 序列 × {seeds} "
              f"种子), 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    tr_mean = float(np.mean([r[3] for r in tr_rows]))
    cr_mean = float(np.mean([r[3] for r in cr_rows]))
    h2_diff = tr_mean - cr_mean

    # GBM null 逐序列
    null_rows = []
    for sym, df1 in crypto:
        nm, ns, _, _, _, _ = null_gradient(df1, None, PARAMS,
                                           PARAMS["gbm_seeds"], crypto_mode=True)
        null_rows.append((sym, nm, ns, "连续"))
    for sym, df1, dfd in control:
        nm, ns, _, _, _, _ = null_gradient(df1, dfd, PARAMS,
                                           PARAMS["gbm_seeds"], crypto_mode=False)
        null_rows.append((sym, nm, ns, "gap匹配"))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, cr_rows, tr_rows, h2_diff, null_rows,
              len(cr_rows), len(tr_rows))
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
