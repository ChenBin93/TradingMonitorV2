#!/usr/bin/env python3
"""多环境参数扫描 v3 — 内存优化: 收集时直接算20组合特征, 不存全序列

环境: 插曲/顺势/逆势/BB反转 × MA{10,15,20,25,30} × 动量{5,10,15,20}
评估: 组合特征 (偏离MA + 0.5×动量, 方向对齐) 的分组胜率差
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20d = daily["close"].rolling(20).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]):
            continue
        d[ts.date()] = "bull" if row["close"] > ma20d[ts] else "bear"
    daily_states[sym] = d

MA_GRID = [10, 15, 20, 25, 30]
MOM_GRID = [5, 10, 15, 20]
N_FEAT = len(MA_GRID) * len(MOM_GRID)
env_data = {name: [] for name in ["插曲", "顺势", "逆势", "BB反转"]}

for sym in syms:
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    if df1 is None or df4 is None or len(df1) < 400:
        continue
    c1 = df1["close"].values
    hi = df1["high"].values
    lo = df1["low"].values
    n1 = len(df1)
    atr = _atr_series(df1)
    idx1 = df1.index.values.astype("datetime64[ns]")
    idx4 = df4.index.values.astype("datetime64[ns]")
    c4 = df4["close"].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    ma20_1 = pd.Series(c1).rolling(20).mean().values
    sd_1 = pd.Series(c1).rolling(20).std().values
    ds = daily_states.get(sym, {})
    ret1 = np.diff(c1) / c1[:-1] * 100

    # 预计算 (float32): 5 MA + cumsum 动量
    ma_all = {p: pd.Series(c1).rolling(p).mean().values.astype(np.float32) for p in MA_GRID}
    cum = np.concatenate([[0.0], np.cumsum(ret1)])
    # 过去 p 根累计收益: cum[i] - cum[i-p]  (i>=p) — 严禁未来数据
    mom_all = {}
    for p in MOM_GRID:
        m = np.full(n1, np.nan, dtype=np.float32)
        m[p:] = (cum[p:] - cum[:-p]).astype(np.float32)
        mom_all[p] = m

    def sim1(long_side, i, a, entry, w=24, tp=1.0, sl=1.0):
        hit = 0
        for k in range(1, w + 1):
            if long_side:
                if hi[i+k] >= entry + tp * a:
                    hit = 1; break
                if lo[i+k] <= entry - sl * a:
                    hit = -1; break
            else:
                if lo[i+k] <= entry - sl * a:
                    hit = 1; break
                if hi[i+k] >= entry + tp * a:
                    hit = -1; break
        return hit

    def feats(i, a, s):
        """20 个组合特征 (方向对齐后)"""
        f = np.empty(N_FEAT, dtype=np.float32)
        k = 0
        for ma_p in MA_GRID:
            dev = (c1[i] - ma_all[ma_p][i]) / a
            for mom_p in MOM_GRID:
                m = mom_all[mom_p][i] / (a / c1[i] * 100)
                f[k] = dev * s + m * s * 0.5
                k += 1
        return f

    for i in range(200, n1 - 49, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]):
            continue
        s4 = "bull" if c4[t4] > ma20_4[t4] else "bear"
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c1[i]
        s_follow = 1 if sd == "bull" else -1
        s_counter = -s_follow
        if s4 != sd:
            hit_f = sim1(sd == "bull", i, a, entry)
            if hit_f != 0:
                env_data["插曲"].append((feats(i, a, s_follow), 1 if hit_f == 1 else 0))
            hit_c = sim1(sd != "bull", i, a, entry)
            if hit_c != 0:
                env_data["逆势"].append((feats(i, a, s_counter), 1 if hit_c == 1 else 0))
        else:
            hit_f = sim1(sd == "bull", i, a, entry)
            if hit_f != 0:
                env_data["顺势"].append((feats(i, a, s_follow), 1 if hit_f == 1 else 0))

    # BB 反转 (密集)
    for i in range(200, n1 - 49):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd != "bear":
            continue
        if np.isnan(ma20_1[i]) or np.isnan(sd_1[i]) or sd_1[i] <= 0:
            continue
        if c1[i] <= ma20_1[i] + 2.5 * sd_1[i]:
            continue
        a = atr[i]
        if a <= 0 or np.isnan(a):
            continue
        entry = c1[i]
        hit = sim1(False, i, a, entry, w=48, tp=0.5, sl=0.5)
        if hit != 0:
            env_data["BB反转"].append((feats(i, a, -1.0), 1 if hit == 1 else 0))
    print(f"{sym}: 插曲{len(env_data['插曲'])} 顺势{len(env_data['顺势'])} 逆势{len(env_data['逆势'])} BB反转{len(env_data['BB反转'])}", flush=True)

print("\n═══ 多环境参数扫描 (分组胜率差 pp) ═══")
for env_name in ["插曲", "顺势", "逆势", "BB反转"]:
    rows = env_data[env_name]
    if len(rows) < 600:
        print(f"\n{env_name}: 样本不足 ({len(rows)})")
        continue
    F = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    base = np.mean(y) * 100
    print(f"\n── {env_name} (n={len(y)}, 基线 {base:.1f}%) ──")
    hdr = "  MA\\动量"
    for mom_p in MOM_GRID:
        hdr += f" {mom_p:>7}"
    print(hdr)
    max_d = (0, None)
    k = 0
    for ma_p in MA_GRID:
        line = f"  {ma_p:>4}"
        for mom_p in MOM_GRID:
            fv = F[:, k]
            k += 1
            med = np.median(fv)
            hi_idx = fv >= med
            lo_idx = fv < med
            if hi_idx.sum() < 50 or lo_idx.sum() < 50:
                line += "     --"
                continue
            wr_hi = np.mean(y[hi_idx]) * 100
            wr_lo = np.mean(y[lo_idx]) * 100
            d = wr_hi - wr_lo
            if abs(d) > max_d[0]:
                max_d = (abs(d), (ma_p, mom_p, wr_hi, wr_lo))
            line += f" {d:>+6.1f}"
        print(line)
    if max_d[1]:
        ma_p, mom_p, wh, wl = max_d[1]
        print(f"  → 最大判别: MA={ma_p} 动量={mom_p} 高分组 {wh:.1f}% vs 低分组 {wl:.1f}% (Δ{max_d[0]:.1f}pp)")
