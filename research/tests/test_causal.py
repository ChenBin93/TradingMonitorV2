#!/usr/bin/env python3
"""causal.py 测试 — 手算黄金 + 追加不变性 (PLAN.md §2 L2)

每个 helper: 手算黄金用例 + 追加数据不改变历史值/组集合。
"""
import numpy as np
import pytest

from research.causal import (FrozenGroup, align_events, causal_confirmed,
                             frozen_cluster, rolling_percentile, rolling_rank)


# ── rolling_percentile ──
def test_rolling_percentile_golden():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    # window=3, q=0.5, min_periods=3 (默认): i<3 → NaN
    got = rolling_percentile(x, 3, 0.5)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert np.allclose(got[2:], [2.0, 3.0, 4.0])
    # q=0.25: 线性插值 pct=1.5/2.5/3.5
    got = rolling_percentile(x, 3, 0.25)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert np.allclose(got[2:], [1.5, 2.5, 3.5])


def test_rolling_percentile_min_periods_partial():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    # window=3, min_periods=1: 部分窗口也出值
    got = rolling_percentile(x, 3, 0.5, min_periods=1)
    assert np.allclose(got, [1.0, 1.5, 2.0, 3.0, 4.0])


def test_rolling_percentile_nan_skipped():
    x = [1.0, np.nan, 3.0, 4.0, 5.0]
    # i=2: 窗口 [1,nan,3], 有效 [1,3], 中位数 2.0; min_periods=2
    got = rolling_percentile(x, 3, 0.5, min_periods=2)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert np.allclose(got[2:], [2.0, 3.5, 4.0])


# ── rolling_rank ──
def test_rolling_rank_golden():
    # x=[3,1,4,2,5], window=3: 平均秩
    x = [3.0, 1.0, 4.0, 2.0, 5.0]
    got = rolling_rank(x, 3)
    assert np.isnan(got[0]) and np.isnan(got[1])  # 前 window-1 根 NaN
    assert np.allclose(got[2:], [1.0, 0.5, 1.0])  # 4→max; 2→中位; 5→max


def test_rolling_rank_min_max_ties():
    got = rolling_rank([5.0, 3.0, 1.0], 3)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert got[2] == 0.0  # min→0
    # 全并列: 平均秩 (1+2+3)/3=2 → (2-1)/(3-1)=0.5
    got = rolling_rank([2.0, 2.0, 2.0], 3)
    assert np.isnan(got[0]) and np.isnan(got[1])
    assert np.allclose(got[2:], [0.5])


# ── 追加不变性: 追加数据不改变历史值 ──
def test_rolling_invariance_append():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    for fn in (lambda a, w=9: rolling_percentile(a, w, 0.5),
               lambda a, w=9: rolling_rank(a, w)):
        full = fn(x)
        trunc = fn(x[:100])
        assert len(trunc) == 100
        # 历史位置逐位相等 (NaN 对齐)
        a, b = full[:100], trunc
        same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, equal_nan=True)
        assert np.all(same), f"{fn}: 追加数据改变了历史值"


# ── causal_confirmed ──
def test_causal_confirmed_golden_cases():
    """手算 3 例: c=t-1 不可用; c=t-24 边界可用; c=t-60 左边界可用"""
    n = 200
    c = 10
    confirmed = np.zeros(n, bool)
    confirmed[c] = True
    known, usable = causal_confirmed(confirmed, w=24)  # 默认 lag_lo=0, lag_hi=60
    assert usable == [c]
    # c=t-1 (t=11): 确认窗口 [11,34] 未收完 → 不可用
    assert not known[c + 1]
    # c=t-24 (t=34): c+24=34 <= 34 → 边界可用
    assert known[c + 24]
    # c=t-60 (t=70): 左边界可用
    assert known[c + 60]
    # 窗口外: t=71 时 c 已不在 [t-60,t] → 不可用
    assert not known[c + 61]
    # 未到确认完成: t=33 → c+24=34 > 33 → 不可用
    assert not known[33]


def test_causal_confirmed_confirm_cost():
    n = 200
    confirmed = np.zeros(n, bool)
    confirmed[10] = True
    known, usable = causal_confirmed(confirmed, w=24, confirm_cost=5)
    # c+29 <= t → 最早 t=39
    assert not known[38] and known[39]
    assert usable == [10]


def test_causal_confirmed_lag_lo():
    n = 200
    confirmed = np.zeros(n, bool)
    confirmed[50] = True
    known, _ = causal_confirmed(confirmed, w=24, lag_lo=5, lag_hi=60)
    # c=50 ∈ [t-60, t-5] → t ∈ [55,110]; 且 t >= 74 → known 从 74 起
    assert not known[73] and known[74] and known[110] and not known[111]


def test_causal_confirmed_multiple_events():
    n = 100
    confirmed = np.zeros(n, bool)
    confirmed[[10, 50]] = True
    known, usable = causal_confirmed(confirmed, w=24)
    assert usable == [10, 50]
    assert known[34] and known[74] and not known[11]
    assert known[99]  # t=99: 窗口 [39,99] 含 c=50


# ── frozen_cluster ──
def test_frozen_cluster_golden():
    events = [(0, 10.0, "A"), (1, 10.5, "A"), (2, 11.0, "A")]
    groups = frozen_cluster(events, tol_fn=lambda t: 1.0, min_touch=2)
    assert len(groups) == 1
    g = groups[0]
    assert g.key == "A"
    assert abs(g.value - 10.25) < 1e-12  # median([10,10.5]) 形成时刻
    assert g.confirm_at == 1             # 第 2 个事件 (min_touch=2)
    assert g.n_touch == 3                # 形成后并入 1 个 touches
    assert g.touches == [(2, 11.0)]      # 不改 value/confirm_at, 只记日志


def test_frozen_cluster_key_separation():
    events = [(0, 10.0, "A"), (1, 10.5, "B")]  # 同值不同 key → 不合并
    groups = frozen_cluster(events, tol_fn=lambda t: 1.0, min_touch=2)
    assert groups == []  # 两组 pending 各 1 个, 都不够 min_touch


def test_frozen_cluster_invariance_append():
    """追加新事件不改变历史 FrozenGroup 集合与 value (关键性质)"""
    rng = np.random.default_rng(3)
    events = []
    # 5 组事件流 (key, 中心, 噪声), confirm_at 升序
    for key, center in [("A", 100.0), ("B", 200.0), ("C", 300.0),
                        ("D", 400.0), ("E", 500.0)]:
        for k in range(12):
            confirm = 10 * len(events) + k
            events.append((confirm, center + rng.normal(0, 2.0), key))
    events.sort(key=lambda e: e[0])
    tol = lambda t: 5.0
    full = frozen_cluster(events, tol, min_touch=3)
    assert len(full) == 5  # 每组各形成 1 个组
    for cut in (30, 60, 90):
        prev = frozen_cluster(events[:cut], tol, min_touch=3)
        # 历史组集合是前缀: key/value/confirm_at 逐位相等
        assert [g.key for g in full[:len(prev)]] == [g.key for g in prev]
        for a, b in zip(full[:len(prev)], prev):
            assert abs(a.value - b.value) < 1e-12
            assert a.confirm_at == b.confirm_at
        # 历史 touches 日志也是前缀 (只增不改)
        for a, b in zip(full[:len(prev)], prev):
            assert a.touches[:len(b.touches)] == b.touches


# ── align_events ──
def test_align_events_golden():
    pos = np.array([1, 5, 9, 12, 20])
    got = align_events(pos, t=10, lag_lo=2, lag_hi=8)   # 窗口 [2, 8]
    assert np.array_equal(got, [5])
    got = align_events(pos, t=10, lag_lo=0, lag_hi=5)   # 窗口 [5, 10]
    assert np.array_equal(got, [5, 9])
    got = align_events(pos, t=10, lag_lo=10, lag_hi=12)  # 窗口 [-2, 0]
    assert len(got) == 0


def test_align_events_sorted_assert():
    with pytest.raises(AssertionError):
        align_events(np.array([3, 1, 2]), t=5, lag_lo=0, lag_hi=10)
