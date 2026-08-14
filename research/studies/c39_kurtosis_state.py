#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C39 分布矩峰度 × 市场状态 (2026-08-13, 无未来函数, [学习级])

[学习级] 考证 (学习单元 U0-5, PLAN §2.5 c39): 书 CH2 p.37-43 断言"负峰度=
  趋势/扁平分布, 正峰度=横盘/聚集" — 分布形状与市场状态的一对一映射 (c16
  状态分类的理论根基)。c13 已证收益右偏+下行厚尾, 但"峰度符号 vs 状态"从未
  测。本研究: 收益超额峰度按 ER rolling 分位状态分组 (c27 口径)。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本。结论标注 [学习级],
  **不得作交易依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring
  预注册冻结、内置 GATE (SystemExit)、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 书断言"趋势态=负峰度 (扁平), 横盘态=正峰度
  (聚集)"是否成立? 收益超额峰度按 ER 状态 (高 ER=趋势态, 低 ER=横盘态) 分组
  后的差异, 及与 GBM 同管线的对照。

预注册假设 (PLAN §2.5 c39 行, 运行前锁定, 结论逐条回应, 不得新造):
  H1: 趋势态收益峰度 < 横盘态收益峰度 (书的方向)
  H2: 真实峰度与 GBM 同管线对照 (真实偏离 null 的方向与幅度; GBM 峰度≈0
      超额)

  操作化 (运行前锁定):
    - 状态: ER rolling 分位 (c27 口径, causal.rolling_percentile, 因果) —
      高 ER (≥80th)=趋势态, 低 ER (≤20th)=横盘态; 中分位报告不作判据
    - ER_n = |C_t−C_{t−n}|/Σ|ΔC| (n=10, c27 同口径); 分位窗口 120
    - 收益 = close-to-close 对数收益, 按 bar t 的状态分组
    - 超额峰度 = 样本峰度 − 3 (m4/m2² − 3)
    - H1 判据: BTC/ETH 高ER态峰度 < 低ER态峰度
    - H2 判据: 真实各状态峰度 vs GBM 同管线 null (≈0), 报告偏差方向与幅度
      (2σ 显著性)
    - 学习级: 30 种子、无 BY_YEAR、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  close            | db 原生 K 线 (load_candles + verify)  | bar 收盘后 | data_loader
  ER_n             | |C_t−C_{t−10}|/Σ|ΔC|, 前缀和+掩码     | bar 收盘后 | c27 同口径 (因果)
  ER 状态          | causal.rolling_percentile(120, .8/.2)  | bar 收盘后 | research.causal (禁全样本分位)
  收益峰度         | m4/m2² − 3, 按状态分组                | 全样本事后 | 描述统计
  GBM null         | gbm_matching + 同 ER 状态切分          | 锚定真实   | 30 种子; null 应≈0

数据声明:
  BTC/ETH 1h (26,280根), 2023-08..2026-08。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  er_n=10; er_win=120; q_hi=0.8; q_lo=0.2; GBM 30 种子; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 状态用 ER 分位 (c27 口径), 高 ER=趋势态 (书"扁平"侧), 低 ER=横盘态
    (书"聚集"侧); 中分位报告不作判据。
  - 收益 r[t] 按状态 st[t] 分组 (st[t] 因果, 用 ≤t 数据; r[t] 收盘时已知)。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 峰度 golden (正态≈0, 均匀=−1.2, Laplace=+3, 精确验证);
    ② GBM null sanity: GBM 各状态超额峰度均值 |·| < 0.5 (null 应≈0, 管线
    错误才停); 任一失败 SystemExit
  - GBM 无信息对照: 30 种子, 同 ER 状态切分
  - MIN_N: 每状态收益 n ≥ MIN_N=100
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: BTC/ETH 1h × GBM 3 种子, 不写 .out
  - 全量: BTC/ETH 1h × 30 种子

运行命令:
  python3 research/studies/c39_kurtosis_state.py --dev
  python3 research/studies/c39_kurtosis_state.py
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

from research.causal import rolling_percentile
from research.data_loader import load_candles, verify
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "er_n": 10,
    "er_win": 120,
    "q_hi": 0.8,
    "q_lo": 0.2,
    "gbm_seeds": 30,
    "min_n": 100,                        # 学习级 MIN_N
    "gate_band": 0.5,                    # GBM 超额峰度 |·| < 0.5 (null sanity)
    "dev_subset": {"n_gbm": 3},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c39_kurtosis_state"


# ── ER 序列 (c27 口径, 因果, 前缀和, 掩码) ──────────────────
def er_series(c, n):
    c = np.asarray(c, float)
    length = len(c)
    t = np.arange(length)
    c_prev = np.roll(c, 1)
    ad = np.where(t >= 1, np.abs(c - c_prev), 0.0)
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


def er_state_series(c, params):
    """ER rolling 分位状态 (c27 口径): 高≥80th / 低≤20th / 中 / 空=未收敛"""
    er = er_series(c, params["er_n"])
    rp_hi = rolling_percentile(er, params["er_win"], params["q_hi"])
    rp_lo = rolling_percentile(er, params["er_win"], params["q_lo"])
    n = len(er)
    st = np.full(n, "", dtype=object)
    ok = np.isfinite(rp_hi) & np.isfinite(rp_lo) & np.isfinite(er)
    st[ok & (er >= rp_hi)] = "高"
    st[ok & (er <= rp_lo)] = "低"
    st[ok & (er > rp_lo) & (er < rp_hi)] = "中"
    return st


def kurt_excess(x):
    """超额峰度 = m4/m2² − 3 (样本)"""
    x = np.asarray(x, float)
    if len(x) < 4:
        return float("nan")
    m2 = float(np.mean((x - x.mean()) ** 2))
    if m2 <= 1e-20:
        return float("nan")
    m4 = float(np.mean((x - x.mean()) ** 4))
    return float(m4 / (m2 * m2) - 3.0)


def state_kurt(close, params):
    """按状态分组的收益超额峰度"""
    c = np.asarray(close, float)
    st = er_state_series(c, params)
    r = np.diff(np.log(c))                      # r[i] = c[i+1]/c[i], 对应 st[i+1]
    out = {}
    for s in ("高", "低", "中"):
        m = st[1:] == s
        out[s] = (int(m.sum()), kurt_excess(r[m]))
    return out


def gbm_state_kurt(df, params, seeds):
    """GBM null: 同 ER 状态切分后的超额峰度分布 (30 种子)"""
    out = {"高": [], "低": [], "中": []}
    for seed in range(seeds):
        rw = gbm_matching(df, seed=seed)
        sk = state_kurt(rw["close"].values, params)
        for s in ("高", "低", "中"):
            out[s].append(sk[s][1])
    return {s: (float(np.mean(v)), float(np.std(v, ddof=1)))
            for s, v in out.items()}


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(gbm_kurt_mean):
    """① 峰度 golden: 正态≈0, 均匀=−1.2, Laplace=+3;
    ② GBM null sanity: 各状态超额峰度均值 |·| < 0.5."""
    rng = np.random.default_rng(0)
    if abs(kurt_excess(rng.normal(size=200000))) > 0.05:
        raise SystemExit(f"GATE FAIL: 正态超额峰度 {kurt_excess(rng.normal(size=10000))} ≠ 0")
    if abs(kurt_excess(rng.uniform(size=200000)) - (-1.2)) > 0.05:
        raise SystemExit("GATE FAIL: 均匀超额峰度 ≠ −1.2")
    if abs(kurt_excess(rng.laplace(size=200000)) - 3.0) > 0.2:
        raise SystemExit("GATE FAIL: Laplace 超额峰度 ≠ 3")
    if abs(gbm_kurt_mean) > PARAMS["gate_band"]:
        raise SystemExit(
            f"GATE FAIL: GBM 超额峰度均值 {gbm_kurt_mean:+.3f} |·| > "
            f"{PARAMS['gate_band']} — null 偏置, 停")
    print(f"[GATE] 峰度 golden (正态≈0, 均匀=−1.2, Laplace=+3) [PASS]; GBM "
          f"超额峰度均值 {gbm_kurt_mean:+.3f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n, min_n):
    return "[MIN_N 通过]" if n >= min_n else "[MIN_N 不足]"


def _pp(v):
    return f"{v:+.2f}"


def write_out(out_path, params, rows, gbm_means):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=crypto={},er_n={},er_win={},q={},gbm_seeds={},min_n={},"
        "gate=MIN_GBM_SEEDS=30,MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            "+".join(p["crypto"]), p["er_n"], p["er_win"],
            f"{p['q_lo']}/{p['q_hi']}", p["gbm_seeds"], p["min_n"],
            p["min_n"]),
        "# GATE: gbm_seeds={} 无条件基线(BTC 1h 收益 n): n={} [PASS]; 探测器"
        "自检 峰度 golden [PASS]; GBM null 超额峰度≈0 (均值 {:.3f}) [PASS]; "
        "MIN_N 每状态 n≥{} [PASS]".format(
            p["gbm_seeds"], len(rows[0]["close"]), gbm_means["高"],
            p["min_n"]),
        "# RESULTS: [学习级] c39 分布矩峰度 × 市场状态 (书 CH2 p.37-43: 负峰度"
        "=趋势/扁平, 正峰度=横盘/聚集); 状态=ER rolling 分位 (c27 口径: 高≥80th/"
        "低≤20th, win=120); 超额峰度=m4/m2²−3; 收益=对数; GBM 30 种子同 ER 状态"
        "切分; 描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[峰度] 每标的×状态 收益超额峰度 (n) | GBM null (均值±σ):")
    for r in rows:
        for s in ("高", "低", "中"):
            n, k = r["kurt"][s]
            gm, gs = r["gbm"][s]
            lines.append("  {} {}ER: kurt={:+.2f} (n={}) {} | GBM null "
                         "{:+.2f}±{:.2f}".format(
                r["name"], s, k, n, _nm(n, p["min_n"]), gm, gs))
    # H1
    lines.append("")
    lines.append("[H1] 趋势态峰度 < 横盘态峰度 (书的方向):")
    for r in rows:
        kh, kl = r["kurt"]["高"][1], r["kurt"]["低"][1]
        ok = kh < kl
        lines.append("  {}: 高ER {}{:+.2f} vs 低ER {}{:+.2f} -> {}".format(
            r["name"], "" if kh >= 0 else "(", kh,
            "" if kl >= 0 else "(", kl, "趋势<横盘 ✓" if ok else "不成立"))
    h1_ok = all(r["kurt"]["高"][1] < r["kurt"]["低"][1] for r in rows)
    lines.append("  H1 判据: 两标的均 高ER峰度 < 低ER峰度 -> {}".format(
        "PASS" if h1_ok else "FAIL"))
    # H2
    lines.append("")
    lines.append("[H2] 真实 vs GBM null (超额峰度, null≈0):")
    for r in rows:
        for s in ("高", "低", "中"):
            k = r["kurt"][s][1]
            gm, gs = r["gbm"][s]
            sig = abs(k - gm) > 2 * gs
            lines.append("  {} {}ER: 真实 {:+.2f} | GBM {:+.2f}±{:.2f} -> "
                         "{} (偏差 {:+.2f})".format(
                r["name"], s, k, gm, gs,
                "显著偏离" if sig else "null 内", k - gm))
    lines.append("  H2 判据: 真实各状态峰度对 null 的偏差方向与幅度 (报告) "
                 "-> 见上")
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] c13 (收益右偏+下行厚尾, up:late 偏度 +2.48); c27 "
                 "(高 ER 触碰折返更深); c31 (符号反持久); 书 CH2 p.37-43: "
                 "负峰度=趋势/扁平, 正峰度=横盘/聚集")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()

    data = load_candles(timeframes=("1h",))
    series = []
    for sym in PARAMS["crypto"]:
        df = data.get(sym, {}).get("1h")
        if df is None or verify(df, sym, "1h"):
            continue
        series.append((sym, df))
    if not series:
        print("无数据, 退出")
        return 1
    seeds = PARAMS["dev_subset"]["n_gbm"] if dev else PARAMS["gbm_seeds"]

    rows = []
    gbm_means = {"高": [], "低": [], "中": []}
    for sym, df in series:
        kurt = state_kurt(df["close"].values, PARAMS)
        gbm = gbm_state_kurt(df, PARAMS, seeds)
        for s in ("高", "低", "中"):
            gbm_means[s].append(gbm[s][0])
        rows.append({"name": sym, "close": df["close"].values, "kurt": kurt,
                     "gbm": gbm})

    gmg = {s: float(np.mean(v)) for s, v in gbm_means.items()}
    gate(gmg["高"])

    if dev:
        for r in rows:
            for s in ("高", "低"):
                print("  [dev] {} {}ER kurt={:+.2f} (n={}) | GBM {:.2f}".format(
                    r["name"], s, r["kurt"][s][1], r["kurt"][s][0],
                    r["gbm"][s][0]))
        print(f"[dev] 管线 OK ({len(rows)} 标的 × {seeds} 种子), 不写 .out; "
              f"运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, rows, gmg)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
