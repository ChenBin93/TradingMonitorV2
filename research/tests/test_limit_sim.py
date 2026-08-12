#!/usr/bin/env python3
"""limit_sim.py 测试 — 手算黄金 + 未成交 + 不变性 (PLAN.md §2 L2)

黄金场景: intrabar 触及成交 (成交价=挂单价) / 下一根 open 出发判定
(跳空越界、前向先碰、同 bar 双命中 skip) / W 根 timeout / 未成交。
"""
import numpy as np
import pytest

from research.limit_sim import LimitOutcome, simulate_limit_entries


def run(t_arr, bars, entry_px, target, stop, w=5, as_arrays=True):
    """bars: [(o,h,l), ...] → open_px/high/low 数组"""
    o = np.array([b[0] for b in bars], float)
    h = np.array([b[1] for b in bars], float)
    l = np.array([b[2] for b in bars], float)
    return simulate_limit_entries(np.array(t_arr, int), o, h, l, entry_px, target, stop, w)


def test_intrabar_fill_then_forward_win():
    """成交: 挂单 bar intrabar 触及 (low<=90<=high), 成交价=90; 下一根 open 出发判定"""
    # bar0 触及挂单价 90 → 成交@bar0 价 90; 判定从 bar1 open 起
    # bar2 high=102 >= target=100 → 前向先碰 → win
    bars = [(100, 105, 90),   # 成交 bar (low 90 <= 90)
            (91, 94, 88),     # open 91 在带内, 无触碰
            (93, 102, 90)]    # high 102 >= 100 → win
    out, recs = run([0], bars, entry_px=90, target=100, stop=85, w=5)
    assert (out.n_win, out.n_loss, out.n_expired, out.n_skip, out.n_unfilled) == (1, 0, 0, 0, 0)
    r = recs[0]
    assert r.outcome == "win"
    assert r.entry_idx == 0 and r.entry_px == 90.0     # 成交价 = 挂单价
    assert r.exit_idx == 2 and r.exit_px == 100.0
    assert r.tp == 100.0 and r.sl == 85.0


def test_unfilled():
    """未成交: 数据内始终 low > 挂单价 → unfilled"""
    bars = [(100, 105, 95), (99, 103, 93), (98, 102, 94)]
    out, recs = run([0], bars, entry_px=80, target=100, stop=85, w=5)
    assert (out.n_win, out.n_loss, out.n_expired, out.n_skip, out.n_unfilled) == (0, 0, 0, 0, 1)
    r = recs[0]
    assert r.outcome == "unfilled"
    assert r.entry_idx == 0 and r.exit_idx == -1 and np.isnan(r.exit_px)


def test_resting_order_fills_later():
    """挂单休眠到未来 bar 成交: 成交 bar 前的高/low 不参与判定"""
    # bar0/1 未触及 88; bar2 low=86 触及 → 成交@bar2 价 88; bar4 触 target
    bars = [(100, 105, 95),
            (99, 103, 92),
            (95, 96, 86),    # 成交 bar (low 86 <= 88 <= 96)
            (89, 97, 85),    # open 89 在 [84,100] 内, 无触碰
            (90, 101, 88)]   # high 101 >= 100 → win
    out, recs = run([0], bars, entry_px=88, target=100, stop=84, w=5)
    assert out.n_win == 1
    r = recs[0]
    assert r.entry_idx == 2 and r.entry_px == 88.0 and r.exit_idx == 4


def test_gap_through_target_is_win():
    """成交后下一根 open 越过 target → 跳空判定 win"""
    bars = [(100, 105, 88),   # 成交 (low 88 <= 90)
            (99, 101, 90)]    # open 99 >= target 95 → 跳空上穿 → win
    out, recs = run([0], bars, entry_px=90, target=95, stop=88, w=5)
    assert (out.n_win, out.n_loss, out.n_skip) == (1, 0, 0)
    r = recs[0]
    assert r.outcome == "win" and r.exit_idx == 1 and r.exit_px == 95.0


def test_gap_through_stop_is_loss():
    """成交后下一根 open 越过 stop → 跳空判定 loss"""
    bars = [(100, 105, 88),   # 成交
            (82, 90, 80)]     # open 82 <= stop 85 → 跳空下穿 → loss
    out, recs = run([0], bars, entry_px=90, target=100, stop=85, w=5)
    assert (out.n_win, out.n_loss, out.n_skip) == (0, 1, 0)
    r = recs[0]
    assert r.outcome == "loss" and r.exit_idx == 1 and r.exit_px == 85.0


def test_single_touch_stop_is_loss():
    """同 bar 单侧触碰 stop → loss (open 出发必先碰下界)"""
    bars = [(100, 105, 90),   # 成交
            (92, 95, 84),     # open 92 在带内; low 84 <= 85 → loss
            (93, 96, 90)]
    out, recs = run([0], bars, entry_px=90, target=100, stop=85, w=5)
    assert out.n_loss == 1
    r = recs[0]
    assert r.outcome == "loss" and r.exit_idx == 1 and r.exit_px == 85.0


def test_same_bar_double_hit_skip():
    """同 bar 双命中 (open 在带内, low<=stop 且 high>=target) → skip"""
    bars = [(100, 105, 90),   # 成交
            (92, 102, 84)]    # open 92 在 [85,100]; low 84<=85 且 high 102>=100 → skip
    out, recs = run([0], bars, entry_px=90, target=100, stop=85, w=5)
    assert (out.n_win, out.n_loss, out.n_skip) == (0, 0, 1)
    r = recs[0]
    assert r.outcome == "skip" and r.exit_idx == 1


def test_timeout_expired():
    """W 根内无触碰 → expired (exit_idx = fill_j + w)"""
    bars = [(100, 105, 90), (91, 94, 88), (92, 95, 89), (93, 96, 90)]  # 成交@0, w=3
    out, recs = run([0], bars, entry_px=90, target=100, stop=85, w=3)
    assert (out.n_win, out.n_loss, out.n_expired, out.n_skip) == (0, 0, 1, 0)
    r = recs[0]
    assert r.outcome == "expired" and r.exit_idx == 3 and np.isnan(r.exit_px)


def test_short_side_loss_at_hi_stop():
    """做空型 (target < entry < stop): hi_bound=stop → 触 hi 为 loss"""
    # entry_px=110, target=100, stop=115; bar0 high 112 >= 110 → 成交@110
    # bar1 high 118 >= 115 → 单侧触 hi (stop) → loss
    bars = [(100, 112, 98),
            (105, 118, 102)]
    out, recs = run([0], bars, entry_px=110, target=100, stop=115, w=5)
    assert (out.n_win, out.n_loss) == (0, 1)
    r = recs[0]
    assert r.outcome == "loss" and r.entry_px == 110.0 and r.exit_px == 115.0


def test_short_side_gap_to_target_win():
    """做空型: 下一根 open 跌破 target → 跳空判定 win"""
    bars = [(100, 112, 98),   # 成交@110
            (99, 105, 95)]    # open 99 <= target 100 → 跳空下穿 → win
    out, recs = run([0], bars, entry_px=110, target=100, stop=115, w=5)
    assert out.n_win == 1
    r = recs[0]
    assert r.outcome == "win" and r.exit_idx == 1 and r.exit_px == 100.0


def test_multiple_independent_orders():
    """多事件独立判定 (允许同 bar 多单)"""
    bars = [(100, 105, 90),
            (91, 94, 88),
            (93, 102, 90),
            (100, 105, 95),
            (96, 104, 92)]
    out, recs = run([0, 3], bars, entry_px=90, target=100, stop=85, w=5)
    # 单0: 成交@0 → win@2; 单3: bar3 low95>90 未成交 → unfilled
    assert (out.n_win, out.n_unfilled) == (1, 1)
    assert [r.outcome for r in recs] == ["win", "unfilled"]


def test_invariance_truncated_vs_full():
    """无未来函数不变性: 截断数据上成交/判定与全量一致

    比较规则: 全量中成交 bar j < cut 的单 → 截断中同 bar 同价成交;
    判定 bar (win/loss/skip 命中 bar) < cut 的 → 截断中结果相同;
    全量中 expired 且 fill_j + w < cut → 截断中也 expired。
    """
    rng = np.random.default_rng(11)
    n = 400
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(c, o) * (1 + rng.uniform(0, 0.006, n))
    l = np.minimum(c, o) * (1 - rng.uniform(0, 0.006, n))
    t_arr = np.arange(0, n - 40, 7)
    entry_px = c[t_arr] * 0.985          # 挂单价低于市价 (买限价)
    target = c[t_arr] * 1.01
    stop = c[t_arr] * 0.96
    w = 12
    full_out, full_recs = simulate_limit_entries(t_arr, o, h, l, entry_px, target, stop, w)
    for cut in (200, 300):
        mask = t_arr < cut
        trunc_out, trunc_recs = simulate_limit_entries(t_arr[mask], o[:cut], h[:cut], l[:cut],
                                                       entry_px[mask], target[mask], stop[mask], w)
        assert len(trunc_recs) == trunc_out.n_total
        for k, r in enumerate(trunc_recs):
            fr = full_recs[k]
            if fr.outcome == "unfilled":
                assert r.outcome == "unfilled"
                continue
            fill_j = fr.entry_idx
            if fill_j >= cut:
                continue  # 成交发生在截断之外 → 不可比
            # 截断中同样成交 bar、同价
            assert r.entry_idx == fill_j and abs(r.entry_px - fr.entry_px) < 1e-12
            if fr.outcome in ("win", "loss", "skip"):
                if fr.exit_idx < cut:
                    assert r.outcome == fr.outcome, f"k={k}: {r} vs {fr}"
                    assert r.exit_idx == fr.exit_idx
            elif fr.outcome == "expired":
                if fill_j + w < cut:
                    assert r.outcome == "expired"
