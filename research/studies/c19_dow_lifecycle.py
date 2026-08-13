#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C19 道氏段生命周期因果重做 (2026-08-13, 无未来函数, 4h + 日线重采样)

[DESCRIPTIVE] 分区: 本研究为描述层 (c1x) — 只刻画市场事实 (道氏段存活曲线、
  回撤恢复率及其气候分层), 无入场, 无交易含义, 无任何方向/收益/成本结论。
  存活曲线 P(段长≥k) 为事后描述统计 (段长来自因果状态机的最终完成值)。
  定位声明: 本研究只确认效应存在性, 不构成任何交易主张。描述层发布门槛:
  无胜率/期望要求, 但必须有 GBM 无信息对照与数字可溯源。

============================================================
研究问题 (预注册, 运行前冻结): 修复 a6 的恢复窗口起点违规与条件变量未来
  泄漏后, 存活曲线与"恢复率真实<GBM"是否仍成立?

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1: 4h 道氏段存活 P(段长≥25根) 真实−GBM ≥ +2pp
  H2: 回撤恢复率净差 ≤ -8pp (真实更脆弱; a6b/a6d 旧值 -12~-17pp 打折重验)
  H3: 段内气候条件 (统计趋势占比等) 全部用滚动 120 根重算后方向不变

  操作化 (运行前锁定):
    - H1 判据: 4h 道氏段 (up+down 合并) 段长≥25 根占比, 真实−GBM ≥ +2pp
      (分方向也报告; a6c 旧值 +4pp, 打折至 +2pp)
    - H2 判据: 回撤事件 (HL/LH pivot 于 j, 确认于 j+K) 后 [j+K+1, j+K+48]
      窗口内 high>段内峰值 (up) / low<段内谷值 (down) 的恢复率,
      真实−GBM ≤ -8pp (a6a 旧值 -12.6pp, 打折重验)
    - H3 判据: 回撤事件按确认时刻滚动 120 根 trend 占比分层
      (冷<20% / 中 20-60% / 热>60%), 热段恢复率净差 ≤ 冷段恢复率净差
      (方向与 a6b 一致: 净差随热度单调下降/热段最负); 真实侧热段恢复率
      ≤ 冷段恢复率 + 1pp (平坦性方向不变)

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  high/low/close   | dow_segments 内部取原始数组            | bar 收盘后 | research.structures (audited,
                   |   (脚本不切片, 全布尔掩码)             |            |   invariance 测试锁定)
  道氏段/回撤事件  | structures.dow_segments (K=3 pivot    | 确认 bar   | test_dow 确认时序: HL pivot@j
                   |   确认时序锁定, 无未来函数)            |   j+K 后   |   在 j..j+K-1 不可见 (黄金测试)
  回撤恢复窗口     | 起点 = 回撤低点 pivot 确认 bar (j+K)   | j+K 收盘后 | a6a 违规起点 j+1 修复; 窗口
                   |   + 1, 即 [j+K+1, j+K+W], W=48       |            |   [j+K+1, j+K+W] (W=48 预注册)
  trend 气候       | state_features.state_series 因果状态机 | bar 收盘后 | 状态机只回看已收盘指标
                   |   (trend_up/trend_down 指示)          |            |   (invariance 测试)
  段内气候条件     | 事件确认时刻滚动 120 根 trend 占比      | 确认时刻   | a6b 整段占比(未来信息)修复;
                   |   (前缀和, 左对齐尾窗)                |            |   禁最终段长/整段归一
  日线层           | data_loader.daily_resample + 已收盘   | 日线收盘后 | a6d 语义: 当日日线 bar 不
                   |   日线对齐 (align_higher 语义, 无      |            |   可用; 段位置不归一 (a6d D3
                   |   searchsorted — 时间戳算术映射)       |            |   教训)
  存活曲线         | P(段长≥k), 段长 = 段 start..end 根数  | 事后描述   | [DESCRIPTIVE] 事后统计
  GBM 无信息对照   | sim_market.gbm_matching(ref_df, seed)  | 锚定真实   | 固定种子序列 0..29; 首标×30
                   |   (索引/长度/σ 锚定真实)               |            |   种子同管线 (4h + 日线重采样)
  分年             | 事件确认时刻年份, 事后聚合             | 全样本     | 描述层 BY_YEAR (成对 真实+GBM)

数据声明:
  data/backtest.db (gitignored): 20 标的 × 4h × 2023-08 → 2026-08
  (4h 6,570根, 时间戳 = bar 开盘时间 UTC); 日线 = 4h 重采样聚合
  (data_loader.daily_resample, resample("1D")); 只用已收盘 bar。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  4h 主周期 + 日线重采样; W=48 恢复窗口; climate_win=120 滚动气候;
  head_drop=200 覆盖状态指标 warm-up (气候事件门槛); gbm_seeds=30。

设计偏离说明 (预注册, 非 post-hoc):
  - 恢复窗口起点修复: a6a/a6b/a6d 从 r["bar"]+1 (pivot bar+1) 起算 — 违规
    (pivot 在 j+K 才确认); c19 一律从确认 bar (j+K) 之后起算, 即
    [j+K+1, j+K+48]。恢复率数字与 a6 系不可逐格对照 (窗口整体后移 K=3 根),
    只对照净差方向/量级。
  - 气候变量修复: a6b 段内 trend 占比用整段 (未来信息); c19 用回撤事件
    确认时刻的滚动 120 根 trend 占比 (因果, 前缀和)。a6d 日线位置/一致性
    中段位置用最终日线段长归一 (未来信息) — c19 只保留"已收盘日线方向
    一致性"层, 不做位置层。
  - 存活曲线是事后描述 [DESCRIPTIVE] (段长来自因果状态机最终完成值), 与
    恢复率分层 (事件时刻因果) 分区标注。
  - 日线对齐用时间戳算术映射 (dpos 字典), 不用 searchsorted (check_study
    AST 禁止), 语义 = daily_resample + align_higher 的已收盘语义。
  - GBM 对照为"首标×30 种子全管线" (含 4h 道氏段 + 日线重采样), PLAN §4
    描述层 exit 模板最小覆盖; 结论均按事件分层, 不按标的做分层结论。

发布门槛自检 (描述层):
  - GATE 探测器: ① dow_segments 对拍 (test_dow 黄金场景语义内联: 回撤事件
    bar=39 确认于 42, j..j+K-1 不可见) 失败 SystemExit; ② GBM 30 种子同管线
    恢复率 null ∈ [70%, 90%] 且存活 P(≥25) ∈ [30%, 60%] (a6a/a6c 旧 GBM 值
    83.3%/45% 边际), 任一失败 SystemExit; ③ n ≥ MIN_N
  - GBM 无信息对照: 30 种子, gbm_matching 锚定真实 (同管线重放)
  - MIN_N 检查: 每个输出格含 n, 不足格标注 [MIN_N 不足] (全单元格输出)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - 定位: 效应量级预期 2~17pp, 本研究只确认效应存在性, 无入场/无交易含义

运行命令:
  python3 -m pytest research/tests -q
  python3 research/check_study.py research/studies/c19_dow_lifecycle.py
  python3 research/studies/c19_dow_lifecycle.py
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

from research.caliber import MIN_GBM_SEEDS, MIN_N
from research.data_loader import daily_resample, load_candles, verify
from research.sim_market import gbm_matching
from research.state_features import state_series
from research.structures import K, dow_segments

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "4h",                          # 主周期 (预注册)
    "W": 48,                             # 恢复窗口 (预注册)
    "climate_win": 120,                  # 滚动气候窗口 (预注册)
    "heat_lo": 0.20,                     # 气候桶: 冷 <20% / 中 20-60% / 热 >60%
    "heat_hi": 0.60,
    "head_drop": 200,                    # 气候事件门槛 (状态指标 warm-up)
    "k_surv": 25,                        # H1 存活阈值 (预注册)
    "surv_ks": (15, 25, 40, 60),         # 存活曲线 k 集合
    "gbm_seeds": MIN_GBM_SEEDS,
    "by_year_list": (2024, 2025, 2026),  # 2023 为部分年, 不纳入
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c19_dow_lifecycle"
RECOVER_START_OFFSET = K + 1             # 恢复窗口起点 = 确认 bar (j+K) + 1


# ── 加载 ─────────────────────────────────────────────────────
def load(timeframes):
    data = load_candles(timeframes=timeframes)
    out = []
    for sym, tfs in data.items():
        for tf in timeframes:
            df = tfs.get(tf)
            if df is None or verify(df, sym, tf):
                continue
            out.append(df)
    return out


# ── GATE: dow_segments 对拍 (test_dow 黄金场景语义内联) ──────
def _mk(closes, spikes=None, hi_margin=0.5, lo_margin=0.5):
    n = len(closes)
    highs = [c + hi_margin for c in closes]
    lows = [c - lo_margin for c in closes]
    for i, (hi, lo) in (spikes or {}).items():
        if hi is not None:
            highs[i] = max(highs[i], hi)
        if lo is not None:
            lows[i] = min(lows[i], lo)
    opens = [closes[i - 1] if i > 0 else closes[0] for i in range(len(closes))]
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": 1.0}, index=idx)


def _build_golden():
    """test_dow 黄金场景: 45 根, up 段 (start 18, end 44), 回撤事件 @39 (确认@42)"""
    closes = [100.0] * 18
    closes += [105.5, 106.5, 107.5, 108.5]
    closes += [107.5, 106.5]
    closes += [105.5, 105.5, 106.0, 106.5]
    closes += [107.5, 108.5, 109.5, 110.5]
    closes += [111.5, 110.5, 109.5]
    closes += [109.0, 108.5, 108.0, 107.5]
    closes += [107.2, 107.8, 108.3, 108.5]
    closes += [107.5, 105.5]
    spikes = {3: (105.0, None), 8: (None, 96.0), 21: (110.0, None),
              24: (None, 103.0), 32: (114.0, None), 39: (None, 106.0)}
    return _mk(closes, spikes)


def gate_dow_check():
    """dow_segments 对拍 (test_dow 语义): 回撤事件 @39 确认于 j+K=42;
    j..j+K-1 内不得出现该事件; 恢复窗口起点常量 = j+K+1."""
    df = _build_golden()
    res = dow_segments(df)
    assert len(res["retraces"]) == 1, "黄金场景回撤事件数应为 1"
    r = res["retraces"][0]
    assert r["bar"] == 39 and r["direction"] == "up", "回撤事件 bar/direction 不符"
    assert not any(x["bar"] == 39 for x in dow_segments(df.iloc[np.arange(42)])["retraces"]), \
        "确认 bar 前事件提前出现 (确认时序违规)"
    assert any(x["bar"] == 39 for x in dow_segments(df.iloc[np.arange(43)])["retraces"]), \
        "确认 bar 后事件未出现"
    assert RECOVER_START_OFFSET == K + 1, "恢复窗口起点必须 = j+K+1"
    return True


# ── 事件收集 (单标的, 因果, 布尔掩码/滑动窗口, 无切片) ───────
def collect(df, params):
    n = len(df)
    W = params["W"]
    cwin = params["climate_win"]
    res = dow_segments(df)
    hi = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    years = np.fromiter((ts.year for ts in df.index), dtype=int, count=n)
    states, _ = state_series(df)
    tr = (np.char.startswith(states, "trend_up")
          | np.char.startswith(states, "trend_down"))

    # 滚动 cwin 根 trend 占比 (因果, 前缀和, 无切片)
    prefix = np.concatenate([[0], np.cumsum(tr)])
    t_idx = np.arange(n)
    frac = np.full(n, np.nan)
    ok = t_idx >= cwin - 1
    frac[ok] = (prefix[t_idx[ok] + 1] - prefix[t_idx[ok] - cwin + 1]) / cwin

    # 滑动窗口最大/最小 (恢复判定, 无切片)
    win_max_h = win_min_l = None
    if n > W:
        from numpy.lib.stride_tricks import sliding_window_view
        win_max_h = sliding_window_view(hi, W).max(axis=1)
        win_min_l = sliding_window_view(lo, W).min(axis=1)

    # 日线层: daily_resample + 已收盘对齐 (时间戳算术, 无 searchsorted)
    daily = daily_resample(df)
    dd = dow_segments(daily)
    dstates = dd["states"]
    dpos = {ts.normalize(): i for i, ts in enumerate(daily.index)}
    one_day = pd.Timedelta(days=1)
    day_idx = np.full(n, -1, dtype=int)
    for i, ts in enumerate(df.index):
        day_idx[i] = dpos.get((ts - one_day).normalize(), -1)

    # H1 存活: 段长 (up+down)
    seg_bars, seg_dir, seg_year = [], [], []
    for s in res["segs"]:
        if s["direction"] in ("up", "down"):
            seg_bars.append(float(s["bars"]))
            seg_dir.append(s["direction"])
            seg_year.append(years[s["start"]])
    # H2/H3 回撤恢复
    rec, rdir, rheat, ryear, rday = [], [], [], [], []
    for r in res["retraces"]:
        j = r["bar"]
        conf = j + K
        if conf + W >= n or win_max_h is None:
            continue
        if r["direction"] == "up":
            recd = float(win_max_h[conf + 1] > r["peak_val"])
        else:
            recd = float(win_min_l[conf + 1] < r["trough_val"])
        rec.append(recd)
        rdir.append(r["direction"])
        ryear.append(years[conf])
        h = frac[conf]
        rheat.append(float(h) if np.isfinite(h) else float("nan"))
        di = day_idx[conf]
        ds = str(dstates[di]) if 0 <= di < len(dstates) else "warmup"
        if ds == r["direction"]:
            rday.append("一致")
        elif ds in ("up", "down"):
            rday.append("相反")
        else:
            rday.append("无日线趋势")
    return {
        "seg_bars": np.array(seg_bars, float), "seg_dir": np.array(seg_dir, object),
        "seg_year": np.array(seg_year, int),
        "rec": np.array(rec, float), "rdir": np.array(rdir, object),
        "rheat": np.array(rheat, float), "ryear": np.array(ryear, int),
        "rday": np.array(rday, object),
        "n_seg": len(seg_bars), "n_retr": len(rec),
    }


def _merge_pool(parts):
    out = {}
    for k in ("seg_bars", "seg_dir", "seg_year", "rec", "rdir", "rheat",
              "ryear", "rday"):
        out[k] = np.concatenate([p[k] for p in parts])
    out["n_seg"] = sum(p["n_seg"] for p in parts)
    out["n_retr"] = sum(p["n_retr"] for p in parts)
    return out


def pool(dfs, params):
    return _merge_pool([collect(df, params) for df in dfs])


def pool_gbm(ref_df, params):
    parts = []
    for seed in range(params["gbm_seeds"]):
        rw = gbm_matching(ref_df, seed=seed)
        parts.append(collect(rw, params))
    return _merge_pool(parts)


# ── 统计 ─────────────────────────────────────────────────────
def surv(pooled, k):
    b = pooled["seg_bars"]
    return (int(b.size), float(np.mean(b >= k)))


def surv_dir(pooled, k, d):
    b = pooled["seg_bars"]
    m = pooled["seg_dir"] == d
    if not m.any():
        return (0, float("nan"))
    return (int(m.sum()), float(np.mean(b[m] >= k)))


def rec_rate(pooled):
    r = pooled["rec"]
    return (int(r.size), float(np.mean(r)))


def rec_rate_dir(pooled, d):
    r = pooled["rec"]
    m = pooled["rdir"] == d
    if not m.any():
        return (0, float("nan"))
    return (int(m.sum()), float(np.mean(r[m])))


def heat_buckets(pooled, params):
    r = pooled["rec"]
    h = pooled["rheat"]
    lo, hi = params["heat_lo"], params["heat_hi"]
    out = {}
    for key, m in (("冷(<20%)", (h < lo) & np.isfinite(h)),
                   ("中(20-60%)", (h >= lo) & (h < hi)),
                   ("热(>60%)", h >= hi)):
        if not m.any():
            out[key] = (0, float("nan"))
        else:
            out[key] = (int(m.sum()), float(np.mean(r[m])))
    return out


def day_buckets(pooled):
    r = pooled["rec"]
    out = {}
    for key in ("一致", "相反", "无日线趋势"):
        m = pooled["rday"] == key
        if not m.any():
            out[key] = (0, float("nan"))
        else:
            out[key] = (int(m.sum()), float(np.mean(r[m])))
    return out


def year_stats(pooled, params):
    out = {}
    for yy in params["by_year_list"]:
        m = pooled["ryear"] == yy
        if not m.any():
            out[yy] = (0, float("nan"))
        else:
            out[yy] = (int(m.sum()), float(np.mean(pooled["rec"][m])))
    return out


def surv_year(pooled, params, k):
    out = {}
    b = pooled["seg_bars"]
    y = pooled["seg_year"]
    for yy in params["by_year_list"]:
        m = y == yy
        if not m.any():
            out[yy] = (0, float("nan"))
        else:
            out[yy] = (int(m.sum()), float(np.mean(b[m] >= k)))
    return out


# ── GATE 自检 (违规即停) ────────────────────────────────────
def gate(ref_4h_df, params):
    gate_dow_check()
    gbm = pool_gbm(ref_4h_df, params)
    rec_n, rec_mean = rec_rate(gbm)
    surv_n, surv_mean = surv(gbm, params["k_surv"])
    if rec_n < MIN_N:
        raise SystemExit(f"GATE FAIL: GBM 回撤 n={rec_n} < MIN_N={MIN_N}")
    if not (0.70 <= rec_mean <= 0.90):
        raise SystemExit(
            f"GATE FAIL: GBM30种子 恢复率 null={rec_mean * 100:.2f}% "
            f"∉ [70%, 90%] — 探测器/窗口错误, 停")
    if not (0.30 <= surv_mean <= 0.60):
        raise SystemExit(
            f"GATE FAIL: GBM30种子 存活 P(≥25) null={surv_mean * 100:.2f}% "
            f"∉ [30%, 60%] — 探测器错误, 停")
    real = collect(ref_4h_df, params)
    real_rec = rec_rate(real)[1]
    print(f"[GATE] 首标4h 恢复率: 真实 {_pct(real_rec)} | GBM30种子 "
          f"{_pct(rec_mean)} (n={rec_n}, ≥{MIN_GBM_SEEDS} 种子); "
          f"存活P(≥25) GBM {_pct(surv_mean)}; dow 对拍 [PASS]", flush=True)
    return {"real_mean": real_rec, "gbm_mean": rec_mean, "n_gbm": rec_n,
            "surv_gbm": surv_mean, "gbm": gbm}


# ── .out 写出 (meta/GATE/RESULTS/BY_YEAR 四区块) ─────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _pct(v):
    return f"{v * 100:.2f}%"


def _pp(v):
    return f"{v * 100:+.2f}pp"


def _nm(n):
    return "[MIN_N 通过]" if n >= MIN_N else "[MIN_N 不足]"


def _net_line(label, rs, gs):
    rn, rm = rs
    gn, gm = gs
    net = (rm - gm) if np.isfinite(rm) and np.isfinite(gm) else float("nan")
    return ("  {}: 真实 {} (n={}) | GBM {} (n={}) | 净差 {} {}".format(
        label, _pct(rm), rn, _pct(gm), gn,
        _pp(net) if np.isfinite(net) else "-", _nm(rn)))


def write_out(out_path, params, g, r, by_year_rows):
    p = params
    rs, gs = r["real"], r["gbm"]
    # H1 存活
    r_s, g_s = surv(rs, p["k_surv"]), surv(gs, p["k_surv"])
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=tf={},W={},climate_win={},heat={},head_drop={},k_surv={},"
        "gbm_seeds={} gate=MIN_GBM_SEEDS={},MIN_N={}(描述层不适用)".format(
            STUDY_ID, date.today().isoformat(), script_sha256(), p["data_range"],
            p["tf"], p["W"], p["climate_win"],
            (p["heat_lo"], p["heat_hi"]), p["head_drop"], p["k_surv"],
            p["gbm_seeds"], MIN_GBM_SEEDS, MIN_N),
        "# GATE: gbm_seeds={} 无条件基线(GBM30种子同管线恢复率 null): "
        "真实 {:.2f}% GBM {:.2f}% [PASS]; 探测器自检: dow_segments 对拍 "
        "(golden 时序, 回撤@39 确认@42) [PASS], GBM 恢复率∈[70%,90%] "
        "存活P(≥25)∈[30%,60%] [PASS]; MIN_N n_gbm={} [PASS]".format(
            p["gbm_seeds"], g["real_mean"] * 100, g["gbm_mean"] * 100,
            g["n_gbm"]),
        "# RESULTS: 20 标的 × 4h + 日线重采样 × 2023-08..2026-08; 描述层无入场, "
        "无交易含义; 恢复窗口 = 回撤低点 pivot (j) 确认 bar (j+K) +1 起 48 根 "
        "(a6a 违规修复); 气候 = 事件确认时刻滚动 120 根 trend 占比 (a6b 整段"
        "占比修复); 日线 = daily_resample + 已收盘对齐 (a6d 语义)",
        "",
        "[H1] 存活 P(段长≥{}): 真实 {} (n_seg={}) | GBM {} (n_seg={}) | "
        "净差 {} {}".format(p["k_surv"], _pct(r_s[1]), r_s[0],
                            _pct(g_s[1]), g_s[0], _pp(r_s[1] - g_s[1]),
                            _nm(r_s[0])),
    ]
    # 存活曲线
    krow = []
    for k in p["surv_ks"]:
        rv, gv = surv(rs, k)[1], surv(gs, k)[1]
        krow.append(f"k={k} 真实 {_pct(rv)} GBM {_pct(gv)} "
                    f"净差 {_pp(rv - gv)}")
    lines.append("  存活曲线: " + " | ".join(krow))
    for d in ("up", "down"):
        lines.append(_net_line(f"  {d} 存活 P(≥{p['k_surv']})",
                               surv_dir(rs, p["k_surv"], d),
                               surv_dir(gs, p["k_surv"], d)))
    # H2 恢复率
    r_r, g_r = rec_rate(rs), rec_rate(gs)
    lines.append("")
    lines.append("[H2] 恢复率 (窗口 [j+K+1, j+K+48]): 真实 {} (n={}) | GBM {} "
                 "(n={}) | 净差 {} {}".format(
        _pct(r_r[1]), r_r[0], _pct(g_r[1]), g_r[0],
        _pp(r_r[1] - g_r[1]), _nm(r_r[0])))
    lines.append(_net_line("  up", rec_rate_dir(rs, "up"), rec_rate_dir(gs, "up")))
    lines.append(_net_line("  down", rec_rate_dir(rs, "down"),
                           rec_rate_dir(gs, "down")))
    # H3 气候
    lines.append("")
    lines.append("[H3] 滚动{}根气候分层恢复率 (事件确认时刻):".format(p["climate_win"]))
    rh, gh = heat_buckets(rs, p), heat_buckets(gs, p)
    nets = []
    for key in ("冷(<20%)", "中(20-60%)", "热(>60%)"):
        rv, gv = rh[key], gh[key]
        lines.append(_net_line(f"  {key}", rv, gv))
        if np.isfinite(rv[1]) and np.isfinite(gv[1]):
            nets.append((key, rv[1] - gv[1]))
    if len(nets) >= 2:
        lo_k, lo_v = min(nets, key=lambda x: x[1])
        hi_k, hi_v = max(nets, key=lambda x: x[1])
        lines.append(f"  净差极值: 最负 {lo_k} {_pp(lo_v)} | 最正 {hi_k} "
                     f"{_pp(hi_v)} | 极差 {_pp(hi_v - lo_v)}")
    # 日线层 (补充, a6d 因果版)
    lines.append("")
    lines.append("[H3-x] 日线一致性分层恢复率 (已收盘日线, a6d 因果版):")
    rd, gd = day_buckets(rs), day_buckets(gs)
    for key in ("一致", "相反", "无日线趋势"):
        lines.append(_net_line(f"  {key}", rd[key], gd[key]))
    # 对照-历史
    lines.append("")
    lines.append("[对照-历史] a6a/a6b/a6c/a6d (2026-08-12 整体作废, 仅形状参照): "
                 "a6a 恢复率 真实 70.7% vs GBM 83.3% (Δ-12.6pp); a6b 气候恢复率 "
                 "真实 70.0~71.6% 平坦 vs GBM 79.0~86.8% 单调 (热段净差-16.8pp); "
                 "a6c 存活 k=25 真实 49% vs GBM 45% (+4pp); a6d 日线一致性恢复率 "
                 "真实 66.4/77.3/75.7% vs GBM 82.8/84.7/83.9% (顺风净差-16.5pp)")
    lines.append("")
    lines.append("# BY_YEAR: " + " | ".join(by_year_rows))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    t0 = time.time()
    dfs = load((PARAMS["tf"],))
    if not dfs:
        print("无数据, 退出")
        return 1

    g = gate(dfs[0], PARAMS)

    real = pool(dfs, PARAMS)
    gbm = g["gbm"]

    # BY_YEAR (主度量: 恢复率 + 存活)
    r_y, g_y = year_stats(real, PARAMS), year_stats(gbm, PARAMS)
    r_sy, g_sy = surv_year(real, PARAMS, PARAMS["k_surv"]), \
        surv_year(gbm, PARAMS, PARAMS["k_surv"])
    year_rows = []
    for yy in PARAMS["by_year_list"]:
        rn, rm = r_y[yy]
        gn, gm = g_y[yy]
        rsn, rsm = r_sy[yy]
        gsn, gsm = g_sy[yy]
        if rn == 0 and gn == 0:
            continue
        year_rows.append("{} 恢复率 真实 {} (n={}) GBM {} (n={}) | 存活 "
                         "真实 {} (n={}) GBM {} (n={})".format(
            yy, _pct(rm), rn, _pct(gm), gn, _pct(rsm), rsn, _pct(gsm), gsn))

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, g, {"real": real, "gbm": gbm}, year_rows)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
