#!/usr/bin/env python3
"""黄金测试 — 手工构造已知答案的合成 K 线, 验证严格口径双实现

每个用例: 手工算好的胜/负/跳过/过期答案
numpy 参考引擎 == 手算答案, vectorbt 实现 == numpy 参考引擎
"""
import numpy as np
import pytest

from research.outcome import Outcome, evaluate_forward, evaluate_forward_vbt

ATR = 10.0


def mk(candles, atr=ATR):
    """candles: [(o,h,l,c), ...] → (close, high, low, atr)"""
    o, h, l, c = (np.array([x[k] for x in candles], float) for k in range(4))
    return c, h, l, np.full(len(c), atr)


def assert_outcome(engine, candles, expected, direction="long", t_mult=1.0, w=24, **kw):
    c, h, l, a = mk(candles)
    out, recs = engine(c, h, l, a, np.array([True] + [False] * (len(c) - 1)),
                       direction=direction, t_mult=t_mult, w=w)
    assert (out.n_win, out.n_loss, out.n_expired, out.n_skip) == \
           (expected.n_win, expected.n_loss, expected.n_expired, expected.n_skip), \
        f"{engine.__name__}: {out}"
    return out, recs


NUCLEAR = [evaluate_forward, evaluate_forward_vbt]

# ── 基础: TP 先中 ──
def test_tp_first():
    candles = [(100, 100, 100, 100),
               (101, 105, 99, 103),
               (103, 112, 102, 110)]
    for eng in NUCLEAR:
        out, recs = assert_outcome(eng, candles, Outcome(1, 0, 0, 0))
        r = recs[0]
        assert r.outcome == "win" and r.exit_idx == 2
        assert r.tp == 110 and r.sl == 90 and r.entry_px == 100


# ── SL 先中 ──
def test_sl_first():
    candles = [(100, 100, 100, 100),
               (101, 105, 88, 100)]
    for eng in NUCLEAR:
        out, recs = assert_outcome(eng, candles, Outcome(0, 1, 0, 0))
        assert recs[0].outcome == "loss" and recs[0].exit_idx == 1


# ── 同 bar 双命中 → 跳过 ──
def test_same_bar_double_hit_skip():
    candles = [(100, 100, 100, 100),
               (101, 112, 88, 100)]  # high>=110 且 low<=90
    for eng in NUCLEAR:
        assert_outcome(eng, candles, Outcome(0, 0, 0, 1))


# ── 跳空跌破 SL → 负 (gap) ──
def test_gap_through_sl_loss():
    candles = [(100, 100, 100, 100),
               (85, 95, 80, 88)]  # 开盘即低于 SL 90
    for eng in NUCLEAR:
        assert_outcome(eng, candles, Outcome(0, 1, 0, 0))


# ── 超时未命中 → 过期 (不计入) ──
def test_expire_within_window():
    candles = [(100, 100, 100, 100)] + [(100, 105, 95, 100)] * 24
    for eng in NUCLEAR:
        out, recs = assert_outcome(eng, candles, Outcome(0, 0, 1, 0), w=24)
        assert recs[0].exit_idx == 24


# ── 入场 bar 自身不算: 信号 bar 的 high 触 TP 不构成胜 ──
def test_entry_bar_high_does_not_count():
    candles = [(100, 115, 90, 100),   # 信号 bar: high>=110 但入场在收盘
               (100, 105, 95, 100)]
    for eng in NUCLEAR:
        assert_outcome(eng, candles, Outcome(0, 0, 1, 0))


# ── 窗口边界: j=i+W 命中算胜, j=i+W+1 算过期 ──
def test_window_boundary_hit_at_w():
    base = [(100, 100, 100, 100)]
    hit_at_w = base + [(100, 105, 95, 100)] * 2 + [(103, 111, 102, 108)]  # W=3, 命中在 j=3
    for eng in NUCLEAR:
        assert_outcome(eng, hit_at_w, Outcome(1, 0, 0, 0), w=3)
    hit_after_w = base + [(100, 105, 95, 100)] * 3 + [(103, 111, 102, 108)]  # 命中在 j=4 > W
    for eng in NUCLEAR:
        assert_outcome(eng, hit_after_w, Outcome(0, 0, 1, 0), w=3)


# ── 做空镜像 ──
def test_short_mirror():
    win = [(100, 100, 100, 100), (99, 102, 88, 95)]       # TP=90 先中
    loss = [(100, 100, 100, 100), (101, 115, 99, 110)]    # SL=110 先中
    both = [(100, 100, 100, 100), (101, 112, 88, 95)]     # 双命中 → 跳过
    gap_win = [(100, 100, 100, 100), (85, 90, 82, 87)]    # 跳空跌破 TP=90 → 胜
    for eng in NUCLEAR:
        assert_outcome(eng, win, Outcome(1, 0, 0, 0), direction="short")
        assert_outcome(eng, loss, Outcome(0, 1, 0, 0), direction="short")
        assert_outcome(eng, both, Outcome(0, 0, 0, 1), direction="short")
        assert_outcome(eng, gap_win, Outcome(1, 0, 0, 0), direction="short")


# ── T 乘数生效 ──
def test_t_multiplier():
    candles = [(100, 100, 100, 100), (103, 108, 101, 106)]  # T=1: TP=110 未中; T=0.5: TP=105 中
    c, h, l, a = mk(candles)
    out_t1, _ = evaluate_forward(c, h, l, a, np.array([True, False]), t_mult=1.0)
    out_t05, _ = evaluate_forward(c, h, l, a, np.array([True, False]), t_mult=0.5)
    assert out_t1.n_expired == 1 and out_t05.n_win == 1


# ── 多个入场独立判定 ──
def test_multiple_entries_independent():
    candles = [(100, 100, 100, 100),
               (101, 105, 99, 103),
               (103, 112, 102, 110),   # 第 1 个入场: 胜
               (110, 110, 110, 110),
               (111, 115, 108, 113),
               (113, 122, 111, 120)]   # 第 2 个入场(bar3 close=110, TP=120): 胜
    entries = np.array([True, False, False, True, False, False])
    for eng in NUCLEAR:
        c, h, l, a = mk(candles)
        out, recs = eng(c, h, l, a, entries)
        assert (out.n_win, out.n_loss, out.n_expired, out.n_skip) == (2, 0, 0, 0)
        assert [r.entry_idx for r in recs] == [0, 3]


# ── 两实现结果完全一致 (逐笔) ──
# column-per-entry 模拟: 每个入场独立一列, 无阻塞/无重叠, 可任意密集
@pytest.mark.parametrize("seed", [1, 7, 42])
def test_vbt_matches_numpy_random(seed):
    rng = np.random.default_rng(seed)
    n = 800
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    h = np.maximum(c * (1 + rng.uniform(0, 0.005, n)), c)
    l = np.minimum(c * (1 - rng.uniform(0, 0.005, n)), c)
    atr = 100 * 0.005 * np.ones(n)
    entries = rng.random(n) < 0.05  # 密集随机入场
    for direction in ("long", "short"):
        for w in (12, 24):
            o1, r1 = evaluate_forward(c, h, l, atr, entries, direction, w=w)
            o2, r2 = evaluate_forward_vbt(c, h, l, atr, entries, direction, w=w)
            assert o1 == o2, f"Outcome mismatch {direction} w={w}: {o1} vs {o2}"
            assert len(r1) == len(r2)
            for a, b in zip(r1, r2):
                assert a.entry_idx == b.entry_idx
                assert a.outcome == b.outcome
                if a.outcome in ("win", "loss"):
                    assert a.exit_idx == b.exit_idx
                    # 成交价差异仅来自跳空: 胜→vbt 按开盘价更优; 负→vbt 按开盘价更差
                    if a.outcome == "win":
                        ok = b.exit_px >= a.exit_px - 1e-6 if direction == "long" \
                            else b.exit_px <= a.exit_px + 1e-6
                    else:
                        ok = b.exit_px <= a.exit_px + 1e-6 if direction == "long" \
                            else b.exit_px >= a.exit_px - 1e-6
                    assert ok, f"exit_px 方向不一致: {a} vs {b}"
                assert abs(a.entry_px - b.entry_px) < 1e-6
