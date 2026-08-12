#!/usr/bin/env python3
"""L1 引擎加固测试 (PLAN.md §1 引擎层 / §2 官方非对称口径扩展)

覆盖:
1. 入口长度断言: entries 与 close 长度不符 → ValueError (三个引擎)
2. 末根入场计入 n_truncated (数据截断, 独立于 expired), 不再静默丢弃
3. gbm_matching: 索引锚定 ref 首根 / 长度相同 / σ = ref 对数收益样本 std
4. check_continuity: 挖掉中间一根 → 报出缺口位置; 并入 verify()
5. 非对称口径 (t_target/t_stop):
   - 手算黄金用例 (t_target<t_stop 与 t_target>t_stop, 含跳空/双命中/超时)
   - GBM 性质: 非对称期望 ≈ 0 (±0.05R) + 多空镜像恒等式
   - 真实数据 numpy vs vbt 对拍
"""
import os

import numpy as np
import pandas as pd
import pytest

from research.caliber import MIN_GBM_SEEDS
from research.data_loader import check_continuity, verify
from research.hold_sim import simulate_holds
from research.outcome import Outcome, evaluate_forward, evaluate_forward_vbt, report_wr
from research.sim_market import gbm_dataframe, gbm_matching, gbm_ohlc

ATR = 10.0
NUCLEAR = [evaluate_forward, evaluate_forward_vbt]

DB = "data/backtest.db"
REAL = pytest.mark.skipif(not os.path.exists(DB), reason="backtest.db 不存在")


# ══ 1. 入口长度断言 ══
def _mk(n=40, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    h = np.maximum(c * (1 + 0.002), c)
    l = np.minimum(c * (1 - 0.002), c)
    a = np.full(n, ATR)
    o = np.concatenate([[c[0]], c[:-1]])
    return c, h, l, a, o


@pytest.mark.parametrize("engine", NUCLEAR)
def test_length_mismatch_entries_raises(engine):
    c, h, l, a, o = _mk()
    entries = np.zeros(len(c) - 5, bool)   # 长度不符 → 必须抛错, 不静默产出
    with pytest.raises(ValueError, match="make_ctx"):
        engine(c, h, l, a, entries)


@pytest.mark.parametrize("engine", NUCLEAR)
def test_length_mismatch_prices_raises(engine):
    c, h, l, a, o = _mk()
    entries = np.zeros(len(c), bool)
    with pytest.raises(ValueError, match="make_ctx"):
        engine(c, h, l[:-3], a, entries)   # low 被截短


def test_hold_sim_length_mismatch_raises():
    c, h, l, a, o = _mk()
    states = np.array(["range"] * len(c))
    entries = np.zeros(len(c) + 3, bool)   # 长度不符
    with pytest.raises(ValueError, match="make_ctx"):
        simulate_holds(c, h, l, a, states, entries)


def test_hold_sim_states_mismatch_raises():
    c, h, l, a, o = _mk()
    states = np.array(["range"] * (len(c) - 1))
    entries = np.zeros(len(c), bool)
    with pytest.raises(ValueError, match="make_ctx"):
        simulate_holds(c, h, l, a, states, entries)


# ══ 2. 末根入场 → n_truncated ══
@pytest.mark.parametrize("engine", NUCLEAR)
def test_last_bar_entry_counted_truncated(engine):
    c, h, l, a, o = _mk()
    entries = np.zeros(len(c), bool)
    entries[-1] = True                       # 末根入场: 无前向 bar
    out, recs = engine(c, h, l, a, entries)
    assert out.n_truncated == 1
    assert out.n_total == 0                  # 独立于 eval/expired/skip
    assert len(recs) == 0                    # 截断不产记录


@pytest.mark.parametrize("engine", NUCLEAR)
def test_truncated_mixed_with_normal(engine):
    c, h, l, a, o = _mk()
    entries = np.zeros(len(c), bool)
    entries[0] = True
    entries[-1] = True
    out, recs = engine(c, h, l, a, entries, w=4)
    assert out.n_truncated == 1
    assert out.n_total == 1                  # 只有正常入场进统计
    assert len(recs) == 1


def test_report_wr_shows_truncated():
    s = report_wr(Outcome(1, 0, 0, 0, n_truncated=3))
    assert "截断 3" in s


# ══ 3. gbm_matching ══
def _mk_ref(n=500, seed=7):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    idx = pd.date_range("2023-08-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "volume": np.ones(n)}, index=idx)


def test_gbm_matching_index_and_length():
    ref = _mk_ref(500)
    g = gbm_matching(ref, seed=42)
    assert g.index[0] == ref.index[0]        # 索引起点锚定 ref 首根
    assert g.index.equals(ref.index)         # 完整索引一致
    assert len(g) == len(ref)
    assert np.isfinite(g["close"]).all()


def test_gbm_matching_sigma_from_ref(monkeypatch):
    ref = _mk_ref(3000)
    sig_ref = float(np.std(np.diff(np.log(ref["close"].values)), ddof=1))
    captured = {}
    def fake_ohlc(n, sig, seed, sub, start):
        captured["sig"] = float(sig)
        captured["start"] = float(start)
        rng = np.random.default_rng(seed)
        c = start * np.exp(np.cumsum(rng.normal(0, sig / np.sqrt(sub), n)))
        return c, c, c, c
    monkeypatch.setattr("research.sim_market.gbm_ohlc", fake_ohlc)
    g = gbm_matching(ref, seed=42)
    assert captured["sig"] == pytest.approx(sig_ref)          # σ = ref 对数收益 std
    assert captured["start"] == pytest.approx(float(ref["close"].iloc[0]))
    assert len(g) == len(ref)


def test_gbm_dataframe_default_unchanged():
    df = gbm_dataframe(10, 0.01, seed=1)
    assert df.index[0] == pd.Timestamp("2023-01-01", tz="UTC")
    assert df.index[1] - df.index[0] == pd.Timedelta(hours=1)  # 默认 1h 行为不变


def test_gbm_dataframe_custom_start_freq():
    df = gbm_dataframe(10, 0.01, seed=1, freq="4h", start_time="2023-08-01")
    assert df.index[0] == pd.Timestamp("2023-08-01", tz="UTC")
    assert df.index[1] - df.index[0] == pd.Timedelta(hours=4)


# ══ 4. check_continuity ══
def test_check_continuity_reports_gap_position():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": np.arange(6.0)}, index=idx)
    df = df.drop(idx[3])                     # 挖掉 03:00 → 04:00 处间隔 2h
    gaps = check_continuity(df, "1h")
    assert len(gaps) == 1
    pos, ts, prev_ts, gap = gaps[0]
    assert pos == 3
    assert ts == idx[4] and prev_ts == idx[2]
    assert gap == pd.Timedelta(hours=2)


def test_check_continuity_clean_and_tfs():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": np.arange(10.0)}, index=idx)
    assert check_continuity(df, "1h") == []
    idx5 = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    df5 = pd.DataFrame({"close": np.arange(5.0)}, index=idx5)
    assert check_continuity(df5, "5m") == []
    idx4 = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    df4 = pd.DataFrame({"close": np.arange(5.0)}, index=idx4)
    assert check_continuity(df4, "4h") == []
    assert check_continuity(df4.iloc[:1], "4h") == []          # 单根无缺口


def test_verify_reports_gap():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    n = len(idx)
    df = pd.DataFrame({"open": np.arange(n, dtype=float),
                       "high": np.arange(n, dtype=float) + 1,
                       "low": np.arange(n, dtype=float) - 1,
                       "close": np.arange(n, dtype=float),
                       "volume": np.ones(n)}, index=idx)
    df = df.drop(idx[3])
    problems = verify(df, "SYM", "1h")
    assert any("缺口" in p for p in problems)
    assert check_continuity(df, "1h")[0][0] == 3


# ══ 5. 非对称口径: 手算黄金用例 ══
def _mk_oc(candles, atr=ATR):
    """candles: [(o,h,l,c), ...] → (close, high, low, atr); open 由引擎
    _default_open 取 prev close (与 test_golden 同口径)"""
    o, h, l, c = (np.array([x[k] for x in candles], float) for k in range(4))
    return c, h, l, np.full(len(c), atr)


def _assert_asym(engine, candles, expected, t_target, t_stop, direction="long", w=24):
    c, h, l, a = _mk_oc(candles)
    out, recs = engine(c, h, l, a, np.array([True] + [False] * (len(c) - 1)),
                       direction=direction, t_target=t_target, t_stop=t_stop, w=w)
    assert (out.n_win, out.n_loss, out.n_expired, out.n_skip) == \
           (expected.n_win, expected.n_loss, expected.n_expired, expected.n_skip), \
        f"{engine.__name__} t_target={t_target} t_stop={t_stop}: {out}"
    return out, recs


# 5a. t_target < t_stop: long TP=105 / SL=85 (ATR=10, entry=100)
@pytest.mark.parametrize("engine", NUCLEAR)
def test_asym_target_lt_stop_long(engine):
    tt, ts = 0.5, 1.5
    win = [(100, 100, 100, 100), (101, 108, 100, 107)]   # 单侧触 TP
    loss = [(100, 100, 100, 100), (101, 104, 84, 90)]    # 单侧触 SL
    skip = [(100, 100, 100, 100), (101, 110, 80, 95)]    # 同 bar 双命中
    gap_loss = [(100, 100, 100, 100), (80, 90, 75, 85)]  # 跳空跌破 SL
    gap_win = [(100, 100, 100, 100), (110, 115, 105, 112)]  # 跳空上穿 TP
    expire = [(100, 100, 100, 100), (101, 104, 96, 102)] * 3   # 带内 → 超时
    _assert_asym(engine, win, Outcome(1, 0, 0, 0), tt, ts)
    _assert_asym(engine, loss, Outcome(0, 1, 0, 0), tt, ts)
    _assert_asym(engine, skip, Outcome(0, 0, 0, 1), tt, ts)
    _assert_asym(engine, gap_loss, Outcome(0, 1, 0, 0), tt, ts)
    _assert_asym(engine, gap_win, Outcome(1, 0, 0, 0), tt, ts)
    _assert_asym(engine, expire, Outcome(0, 0, 1, 0), tt, ts)


# 5b. t_target > t_stop: long TP=120 / SL=95
@pytest.mark.parametrize("engine", NUCLEAR)
def test_asym_target_gt_stop_long(engine):
    tt, ts = 2.0, 0.5
    win = [(100, 100, 100, 100), (101, 125, 100, 122)]
    loss = [(100, 100, 100, 100), (99, 102, 92, 96)]
    skip = [(100, 100, 100, 100), (101, 125, 92, 110)]
    gap_loss = [(100, 100, 100, 100), (90, 100, 85, 92)]
    gap_win = [(100, 100, 100, 100), (125, 130, 120, 127)]
    _assert_asym(engine, win, Outcome(1, 0, 0, 0), tt, ts)
    _assert_asym(engine, loss, Outcome(0, 1, 0, 0), tt, ts)
    _assert_asym(engine, skip, Outcome(0, 0, 0, 1), tt, ts)
    _assert_asym(engine, gap_loss, Outcome(0, 1, 0, 0), tt, ts)
    _assert_asym(engine, gap_win, Outcome(1, 0, 0, 0), tt, ts)


# 5c. short 镜像: t_target=0.5 / t_stop=1.5 → TP=95 / SL=115
@pytest.mark.parametrize("engine", NUCLEAR)
def test_asym_short_mirror(engine):
    tt, ts = 0.5, 1.5
    win = [(100, 100, 100, 100), (99, 102, 92, 94)]    # 单侧触 TP (下)
    loss = [(100, 100, 100, 100), (101, 118, 100, 115)]  # 单侧触 SL (上)
    skip = [(100, 100, 100, 100), (101, 118, 92, 100)]   # 同 bar 双命中
    _assert_asym(engine, win, Outcome(1, 0, 0, 0), tt, ts, direction="short")
    _assert_asym(engine, loss, Outcome(0, 1, 0, 0), tt, ts, direction="short")
    _assert_asym(engine, skip, Outcome(0, 0, 0, 1), tt, ts, direction="short")


# ══ 6. 非对称口径: GBM 性质测试 ══
def _expected_r_stats(recs, tt, ts):
    """(n_eval, n_win, EV_R) — R 归一化到止损距离: win = +t_target/t_stop R, loss = -1 R"""
    ev, cnt, wins = 0.0, 0, 0
    for r in recs:
        if r.outcome == "win":
            ev += tt / ts
            cnt += 1
            wins += 1
        elif r.outcome == "loss":
            ev += -1.0
            cnt += 1
    return cnt, wins, (ev / cnt if cnt else float("nan"))


def test_asym_gbm_properties():
    """零漂移 GBM (30 固定种子, 固定种子序列禁换): 非对称口径期望 ≈0 (±0.05R)

    多空镜像对非对称口径成立的形式 (过程对称, 两侧目标距离同为 tt×ATR):
    - WR_long(tt,ts) == WR_short(tt,ts) (同参数, 引擎无方向偏置)
    - WR_long(tt,ts) + WR_short(ts,tt) ≈ 1 (参数互换才是真互补镜像)
    容差 ±0.05R 与阈值取 ratio≤3 组合 (PLAN 实际用例 c21 0.3/0.7、c23 1.0/0.3
    同量级); 更大 ratio 的远目标胜率被有限样本漂移主导, 单测不可靠。
    """
    n = 5000
    sig = 0.002
    atr = 1.0
    seeds = range(30)
    combos = [(0.5, 1.5), (1.5, 0.5)]
    stats = {}
    for tt, ts in combos:
        ev_num_l = ev_num_s = 0.0
        cnt_l = cnt_s = wins_l = wins_s = 0
        for s in seeds:
            o, h, l, c = gbm_ohlc(n, sig, seed=s)
            atr_arr = np.full(n, atr)
            entries = np.ones(n, bool)
            out_l, recs_l = evaluate_forward(c, h, l, atr_arr, entries, "long",
                                             t_target=tt, t_stop=ts, open_px=o, w=96)
            out_s, recs_s = evaluate_forward(c, h, l, atr_arr, entries, "short",
                                             t_target=tt, t_stop=ts, open_px=o, w=96)
            cl, wl, el = _expected_r_stats(recs_l, tt, ts)
            cs, ws, es = _expected_r_stats(recs_s, tt, ts)
            cnt_l += cl; wins_l += wl; ev_num_l += el * cl
            cnt_s += cs; wins_s += ws; ev_num_s += es * cs
        stats[(tt, ts, "long")] = (cnt_l, wins_l, ev_num_l / cnt_l)
        stats[(tt, ts, "short")] = (cnt_s, wins_s, ev_num_s / cnt_s)
    # 期望 ≈ 0 (±0.05R) — 非对称结构在无信息市场无 edge
    for k, (cnt, wins, ev) in stats.items():
        assert cnt > 50000, f"{k}: n 不足 {cnt}"
        assert abs(ev) < 0.05, f"{k}: EV {ev:+.4f}R"
    # 同参数多空一致 (引擎无方向偏置)
    for tt, ts in combos:
        wr_l = stats[(tt, ts, "long")][1] / stats[(tt, ts, "long")][0]
        wr_s = stats[(tt, ts, "short")][1] / stats[(tt, ts, "short")][0]
        assert abs(wr_l - wr_s) < 0.005, \
            f"({tt},{ts}) long/short WR 不一致: {wr_l:.4f} vs {wr_s:.4f}"
    # 参数互换互补镜像
    wr_l = stats[(0.5, 1.5, "long")][1] / stats[(0.5, 1.5, "long")][0]
    wr_s = stats[(1.5, 0.5, "short")][1] / stats[(1.5, 0.5, "short")][0]
    assert abs(wr_l + wr_s - 1.0) < 0.005, \
        f"镜像破坏: {wr_l:.4f}+{wr_s:.4f}"


@REAL
def test_asym_cross_check_real_data():
    """真实 1h 数据: numpy 与 vbt 在非对称口径下必须一致 (对拍)"""
    from market_phase import _atr_series
    from research.data_loader import load_candles
    data = load_candles(timeframes=("1h",))
    assert data, "backtest.db 无 1h 数据"
    checked = 0
    for sym, tfs in list(data.items())[:6]:
        df = tfs["1h"]
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        atr = _atr_series(df)
        n = len(c)
        entries = np.zeros(n, bool)
        entries[::97] = True                 # 确定性抽样入场
        for tt, ts in ((0.5, 1.5), (1.5, 0.5)):
            for direction in ("long", "short"):
                o1, r1 = evaluate_forward(c, h, l, atr, entries, direction,
                                          t_target=tt, t_stop=ts, w=24)
                o2, r2 = evaluate_forward_vbt(c, h, l, atr, entries, direction,
                                              t_target=tt, t_stop=ts, w=24)
                assert o1 == o2, f"{sym} {direction} ({tt},{ts}): {o1} vs {o2}"
                assert len(r1) == len(r2), f"{sym} {direction}: 记录数不一致"
                for a, b in zip(r1, r2):
                    assert a.entry_idx == b.entry_idx
                    assert a.outcome == b.outcome, f"{sym} {direction} ({tt},{ts}): {a} vs {b}"
                    if a.outcome in ("win", "loss"):
                        assert abs(a.exit_idx - b.exit_idx) <= 1
        checked += 1
    assert checked >= 3, f"对拍覆盖不足: 只有 {checked} 个标的"


# ══ 7. caliber 常量 ══
def test_min_gbm_seeds_constant():
    assert MIN_GBM_SEEDS == 30
