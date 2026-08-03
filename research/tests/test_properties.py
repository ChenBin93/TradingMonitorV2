#!/usr/bin/env python3
"""性质测试 (physics checks) — 验证引擎"设计正确性", 不依赖手算答案

2026-08-03 引入 (B1 36%/64% 假象的教训):
1. 无信息市场 (GBM, 真实 OHLC) 无条件 1:1 基线必须 ≈ 理论值 50% (±0.5pp)
   — 旧简化构造测出 51.3% 即为此类偏差漏网
2. 多空镜像恒等式: 同批入场, long WR + short WR = 100% (±0.1pp)
3. 合成市场恢复真值: 已知触碰后反弹率 p_true 的市场, 引擎输出必须 ≈ p_true
   — 这是"设计+实现"整体的端到端验证 (行业标准做法)
4. 跳空语义: open 已越过目标时按越界判定 (旧判定在"跳空+both"场景错误)
"""
import numpy as np
import pytest

from research.outcome import evaluate_forward
from research.sim_market import gbm_ohlc

D = 1.0  # 目标距离 (ATR×T)


# ── 1. 无信息市场基线 ≈ 50% ──
def test_rw_baseline_is_50():
    n = 120000
    sig = 0.002
    o, h, l, c = gbm_ohlc(n, sig, seed=0)
    atr = np.full(n, D)
    entries = np.ones(n, bool)
    out_l, _ = evaluate_forward(c, h, l, atr, entries, "long", open_px=o, w=96)
    out_s, _ = evaluate_forward(c, h, l, atr, entries, "short", open_px=o, w=96)
    assert out_l.n_eval > 50000
    assert out_s.n_eval > 50000
    # 关键判据: 无信息市场 1:1 胜率必须 ≈ 理论 50% (修正确认)
    assert abs(out_l.win_rate - 0.50) < 0.005, f"long 基线 {out_l.win_rate:.3f} ≠ 50%"
    assert abs(out_s.win_rate - 0.50) < 0.005, f"short 基线 {out_s.win_rate:.3f} ≠ 50%"


# ── 2. 多空镜像恒等式 ──
def test_mirror_identity():
    n = 60000
    sig = 0.002
    o, h, l, c = gbm_ohlc(n, sig, seed=1)
    atr = np.full(n, D)
    rng = np.random.default_rng(3)
    entries = rng.random(n) < 0.3
    out_l, _ = evaluate_forward(c, h, l, atr, entries, "long", open_px=o, w=96)
    out_s, _ = evaluate_forward(c, h, l, atr, entries, "short", open_px=o, w=96)
    assert abs((out_l.win_rate + out_s.win_rate) - 1.0) < 0.001, \
        f"镜像恒等式破坏: {out_l.win_rate:.4f} + {out_s.win_rate:.4f}"


# ── 3. 合成市场恢复真值 ──
def _known_market(n_seg, p_true, d, seed=0, period=20, noise=0.05):
    """构造已知市场: 每 period 根一个入场 (close=d 目标带), 入场后第一根
    以 p_true 先碰 +d (反弹路径), 否则先碰 -d (下跌路径); 之后噪声填充

    反弹路径 bar: 子步先涨 d*1.3 再回落 → high ≥ +d 且 low > -d (单侧)
    下跌路径 bar: 子步先跌 d*1.3 再回升 → low ≤ -d 且 high < +d (单侧)
    """
    rng = np.random.default_rng(seed)
    base = 100.0
    n = n_seg * period
    o = np.empty(n)
    h = np.empty(n)
    l = np.empty(n)
    c = np.empty(n)
    entries = np.zeros(n, bool)
    for s in range(n_seg):
        i = s * period
        # 入场 bar
        o[i] = base
        h[i] = base + noise * d
        l[i] = base - noise * d
        c[i] = base
        entries[i] = True
        # 结果 bar (i+1)
        up = rng.random() < p_true
        if up:
            path = [base + d * 1.3, base + d * 0.9, base + d * 1.1, base + d * 0.3,
                    base + d * 0.4, base - 0.3 * d, base - 0.2 * d, base + 0.1 * d]
        else:
            path = [base - d * 1.3, base - d * 0.9, base - d * 1.1, base - d * 0.3,
                    base - d * 0.4, base + 0.3 * d, base + 0.2 * d, base - 0.1 * d]
        o[i + 1] = base
        h[i + 1] = max(path)
        l[i + 1] = min(path)
        c[i + 1] = path[-1]
        # 噪声填充
        for k in range(2, period):
            j = i + k
            o[j] = c[j - 1]
            v = base + rng.normal(0, 0.2 * d)
            h[j] = max(o[j], v) + 0.1 * d
            l[j] = min(o[j], v) - 0.1 * d
            c[j] = v
    return o, h, l, c, entries


@pytest.mark.parametrize("p_true", [0.4, 0.5, 0.6, 0.7])
def test_known_market_recovers_truth(p_true):
    n_seg = 4000
    o, h, l, c, entries = _known_market(n_seg, p_true, D, seed=int(p_true * 100))
    atr = np.full(len(c), D)
    out, _ = evaluate_forward(c, h, l, atr, entries, "long", open_px=o, w=4)
    assert out.n_eval > 3000
    # 引擎必须恢复已知真值 (±2pp)
    assert abs(out.win_rate - p_true) < 0.02, \
        f"合成市场 p_true={p_true} → 引擎 {out.win_rate:.3f}"


# ── 4. 跳空语义 ──
def test_gap_open_determines_side():
    """open 已越过 SL 但 bar 内回穿 TP → 应按跳空侧判定 (loss, 旧判定错误 skip)"""
    c = np.array([100.0, 95.0])
    h = np.array([100.0, 116.0])   # 回穿 TP 上方
    l = np.array([100.0, 94.0])    # 且低于 SL
    o = np.array([100.0, 85.0])    # 跳空低开越过 SL=90
    atr = np.array([10.0, 10.0])
    entries = np.array([True, False])
    out, recs = evaluate_forward(c, h, l, atr, entries, "long", open_px=o, w=4)
    assert (out.n_win, out.n_loss, out.n_skip) == (0, 1, 0), f"{out}"
    assert recs[0].exit_px == 90.0


def test_gap_up_through_tp_is_win():
    """open 已越过 TP → win (跳空上穿)"""
    c = np.array([100.0, 105.0])
    h = np.array([100.0, 106.0])
    l = np.array([100.0, 85.0])    # 回穿 SL 下方
    o = np.array([100.0, 112.0])   # 跳空高开越过 TP=110
    atr = np.array([10.0, 10.0])
    entries = np.array([True, False])
    out, recs = evaluate_forward(c, h, l, atr, entries, "long", open_px=o, w=4)
    assert (out.n_win, out.n_loss, out.n_skip) == (1, 0, 0), f"{out}"
    assert recs[0].exit_px == 110.0
