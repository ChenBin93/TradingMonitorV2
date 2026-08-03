#!/usr/bin/env python3
"""目标×窗口网格: 贴支撑做多/贴阻力做空 在 6目标×5窗口 的胜率矩阵
+ 固定SL(=1ATR) EV 附送 + 未贴位对照 + 插曲状态单列(P1 闭合)

性能设计 (README 纪律):
- 复用 level_state_fast.py: 预计算极值 mask + 接近检测 (无 find_swing_levels)
- 判定全向量化: 每 symbol 每 T 一次构造 (n,96) 窗口数组 + argmax first-hit,
  5 个窗口由 "hit_bar <= W" 截断一次覆盖 → 30 格零 Python 内层循环
- 汇总用 np.bincount → 无逐样本累加
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

TARGETS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
WINDOWS = [6, 12, 24, 48, 96]
W_MAX = max(WINDOWS)
LOOKBACK = 600
MIN_N = 100

t0 = time.time()
data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())
print(f"[load {time.time()-t0:.0f}s] {len(syms)} syms", flush=True)

# ── 日线状态 (三分类 ±0.5ATR) ──
daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 300:
        continue
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(daily) < 200:
        continue
    ma20d = daily["close"].rolling(20).mean()
    atr_d = (daily["high"] - daily["low"]).rolling(14).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]) or pd.isna(atr_d[ts]) or atr_d[ts] <= 0:
            continue
        dev = (row["close"] - ma20d[ts]) / atr_d[ts]
        d[ts.date()] = "多" if dev > 0.5 else "空" if dev < -0.5 else "中"
    daily_states[sym] = d

# ── 4H swing 极值 (重叠标志) ──
sw4 = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None:
        continue
    h4 = df4["high"].values
    l4 = df4["low"].values
    sw4[sym] = np.where(
        (h4 >= pd.Series(h4).rolling(5, center=True).max().values) |
        (l4 <= pd.Series(l4).rolling(5, center=True).min().values)
    )[0]

# ── 累加器: [T][W] -> wins(27), tots(27); 27 = st9*3 + q ──
def new_acc():
    return {ti: {wi: [np.zeros(27, int), np.zeros(27, int)]
                 for wi in range(len(WINDOWS))}
            for ti in range(len(TARGETS))}

# 贴位: 对称1:1 / 固定SL1  (long=贴支撑做多, short=贴阻力做空)
s_acc = {"long": new_acc(), "short": new_acc()}
sl_acc = {"long": new_acc(), "short": new_acc()}
# 未贴位对照
c_acc = {"long": new_acc(), "short": new_acc()}

ST9_ID = {"日多4H多": 0, "日多4H中": 1, "日多4H空": 2,
          "日中4H多": 3, "日中4H中": 4, "日中4H空": 5,
          "日空4H多": 6, "日空4H中": 7, "日空4H空": 8}


def hit_stats(H, L, E, A, T, SL_T=None):
    """向量化: 每样本窗口 W_MAX 内, 先到 +T*ATR(+1) 还是 -SL_T*ATR(-1), 命中 bar (0=未命中)"""
    sl = SL_T if SL_T else T
    up = H >= (E + T * A)[:, None]
    dn = L <= (E - sl * A)[:, None]
    f_up = up.argmax(axis=1) + 1
    f_dn = dn.argmax(axis=1) + 1
    has_up = up.any(axis=1)
    has_dn = dn.any(axis=1)
    first = np.where(has_up & ~has_dn, 1,
                     np.where(~has_up & has_dn, -1,
                              np.where(has_up & has_dn & (f_up < f_dn), 1,
                                       np.where(has_up & has_dn & (f_dn < f_up), -1, 0))))
    bar = np.where(first == 1, f_up, np.where(first == -1, f_dn, 0))
    return first, bar


def accumulate(acc, key, first, bar):
    """按 (T,W) 把 (win,total) bincount 进 27 桶"""
    for ti, T in enumerate(TARGETS):
        ft = first[ti]
        bt = bar[ti]
        for wi, W in enumerate(WINDOWS):
            v = (ft != 0) & (bt <= W)
            if v.any():
                w_ = v & (ft == 1)
                acc[ti][wi][0] += np.bincount(key[w_], minlength=27)
                acc[ti][wi][1] += np.bincount(key[v], minlength=27)


n_samples = 0
for sym in syms:
    t_sym = time.time()
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    if df1 is None or df4 is None or len(df1) < LOOKBACK + 400 or sym not in daily_states:
        continue
    c = df1["close"].values
    h = df1["high"].values
    l = df1["low"].values
    n = len(df1)
    atr = _atr_series(df1)
    idx1 = df1.index.values.astype("datetime64[ns]")
    idx4 = df4.index.values.astype("datetime64[ns]")
    c4 = df4["close"].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    atr4 = _atr_series(df4)
    sw = sw4.get(sym)

    # 预计算 1H 极值 mask
    is_low = l <= pd.Series(l).rolling(5, center=True).min().values
    is_high = h >= pd.Series(h).rolling(5, center=True).max().values

    # ── 采样点 + 9 状态 (向量化) ──
    i_s = np.arange(LOOKBACK + 50, n - W_MAX, 4)
    ds_series = pd.Series(daily_states[sym])
    dkey = np.array(pd.DatetimeIndex(idx1[i_s]).date)
    sd_vals = ds_series.reindex(pd.Index(dkey)).values
    ok_d = ~pd.isna(sd_vals)
    t4s = np.searchsorted(idx4, idx1[i_s] - np.timedelta64(240, "m"), side="right") - 1
    ok4 = (t4s >= 20) & ~np.isnan(ma20_4[t4s]) & (atr4[t4s] > 0)
    dev4 = np.where(ok4, (c4[t4s] - ma20_4[t4s]) / atr4[t4s], 0.0)
    s4 = np.where(dev4 > 0.5, "4H多", np.where(dev4 < -0.5, "4H空", "4H中"))
    st9 = np.array(["日" + sd + s for sd, s in zip(sd_vals.astype(str), s4)])
    st9_id = np.array([ST9_ID.get(x, -1) for x in st9])
    keep = ok_d & ok4 & (st9_id >= 0)
    i_s = i_s[keep]
    st9_id = st9_id[keep]
    if len(i_s) == 0:
        continue

    # 向量化窗口数组
    H = h[i_s[:, None] + np.arange(1, W_MAX + 1)]
    L = l[i_s[:, None] + np.arange(1, W_MAX + 1)]
    E = c[i_s]
    A = atr[i_s]

    # ── 未贴位对照: 无条件同状态采样 (方向: 日多→做多, 日空→做空) ──
    c_dir = np.where(st9_id < 3, 0, np.where(st9_id >= 6, 1, -1))  # 0=long 1=short -1=日中跳过
    c_keep = c_dir >= 0
    if c_keep.any():
        ck = np.where(c_keep)[0]
        c_first = []
        c_bar = []
        for T in TARGETS:
            f, b = hit_stats(H[ck], L[ck], E[ck], A[ck], T)
            c_first.append(f)
            c_bar.append(b)
        for side_id, side in ((0, "long"), (1, "short")):
            m = c_dir[ck] == side_id
            if not m.any():
                continue
            key27 = (st9_id[ck][m] * 3 + 0)  # 对照不按质量分 → q=0
            firsts = np.array([c_first[ti][m] for ti in range(len(TARGETS))])
            bars = np.array([c_bar[ti][m] for ti in range(len(TARGETS))])
            if side == "short":
                firsts = -firsts  # FIX: 做空赢 = 下跌先到
            for ti in range(len(TARGETS)):
                ft = firsts[ti]
                bt = bars[ti]
                for wi, W in enumerate(WINDOWS):
                    v = (ft != 0) & (bt <= W)
                    if v.any():
                        w_ = v & (ft == 1)
                        c_acc[side][ti][wi][0] += np.bincount(key27[w_], minlength=27)
                        c_acc[side][ti][wi][1] += np.bincount(key27[v], minlength=27)

    # ── 贴位检测 (per-sample, 极值 mask 切片) ──
    long_i, short_i = [], []
    long_st9, short_st9 = [], []
    long_q, short_q = [], []
    for j, i in enumerate(i_s):
        a = A[j]
        if a <= 0 or np.isnan(a):
            continue
        entry = E[j]
        tol = 0.5 * a
        sid = st9_id[j]
        w_start = i - LOOKBACK
        # 支撑 (贴而未破): 0 ≤ entry-low ≤ tol
        lo_mask = is_low[w_start:i]
        if lo_mask.any():
            lo_p = l[w_start:i][lo_mask]
            sup = (lo_p <= entry) & (entry - lo_p <= tol)
            if sup.any():
                sup_touch = int(sup.sum())
                sup_age = int(i - (w_start + np.where(sup)[0][0]))
                ov4 = False
                if sw is not None:
                    j4 = sw[(sw >= max(0, len(c4) - 50)) & (sw < len(c4))]
                    ov4 = bool(np.any(np.abs(c4[j4] - entry) <= tol)) if len(j4) else False
                sc = 0
                if 200 <= sup_age < 400:
                    sc += 2
                elif sup_age >= 100:
                    sc += 1
                if sup_touch >= 5:
                    sc += 1
                elif sup_touch >= 3:
                    sc += 0.5
                if ov4:
                    sc += 1.5
                q = 2 if sc >= 3 else 1 if sc >= 1.5 else 0
                long_i.append(j)
                long_st9.append(sid)
                long_q.append(q)
        # 阻力 (贴而未破): 0 ≤ high-entry ≤ tol
        hi_mask = is_high[w_start:i]
        if hi_mask.any():
            hi_p = h[w_start:i][hi_mask]
            res_ = (hi_p >= entry) & (hi_p - entry <= tol)
            if res_.any():
                res_touch = int(res_.sum())
                res_age = int(i - (w_start + np.where(res_)[0][0]))
                ov4 = False
                if sw is not None:
                    j4 = sw[(sw >= max(0, len(c4) - 50)) & (sw < len(c4))]
                    ov4 = bool(np.any(np.abs(c4[j4] - entry) <= tol)) if len(j4) else False
                sc = 0
                if 200 <= res_age < 400:
                    sc += 2
                elif res_age >= 100:
                    sc += 1
                if res_touch >= 5:
                    sc += 1
                elif res_touch >= 3:
                    sc += 0.5
                if ov4:
                    sc += 1.5
                q = 2 if sc >= 3 else 1 if sc >= 1.5 else 0
                short_i.append(j)
                short_st9.append(sid)
                short_q.append(q)

    # ── 贴位判定 (向量化) ──
    for side, idxs, st9s, qs in (("long", long_i, long_st9, long_q),
                                 ("short", short_i, short_st9, short_q)):
        if not idxs:
            continue
        idxs = np.array(idxs)
        key = np.array(st9s) * 3 + np.array(qs)
        Hs, Ls, Es, As = H[idxs], L[idxs], E[idxs], A[idxs]
        n_s = len(idxs)
        first_all = np.zeros((len(TARGETS), n_s), int)
        bar_all = np.zeros((len(TARGETS), n_s), int)
        first1_all = np.zeros((len(TARGETS), n_s), int)
        bar1_all = np.zeros((len(TARGETS), n_s), int)
        for ti, T in enumerate(TARGETS):
            f, b = hit_stats(Hs, Ls, Es, As, T)
            if side == "short":
                f = -f  # FIX: 做空赢 = 下跌先到 (ft=-1), 取反统一为 ft==1 判赢
            first_all[ti] = f
            bar_all[ti] = b
            f1, b1 = hit_stats(Hs, Ls, Es, As, T, SL_T=1.0)
            if side == "short":
                f1 = -f1  # FIX 同上 (固定SL)
            first1_all[ti] = f1
            bar1_all[ti] = b1
        accumulate(s_acc[side], key, first_all, bar_all)
        accumulate(sl_acc[side], key, first1_all, bar1_all)
        n_samples += n_s

    print(f"{sym}: {len(i_s)} pts ({time.time()-t_sym:.0f}s)", flush=True)

print(f"\n[scan {n_samples} touch samples, total {time.time()-t0:.0f}s]", flush=True)

# ── 聚合输出 ──
ST9_RANGE = np.arange(27)
Q3 = [(0, "低"), (1, "中"), (2, "高")]


def masked(acc, side, st9_set, qs=None):
    """st9_set: list of st9 ids; qs: None=全部 or list of q"""
    m = np.zeros(27, bool)
    for s in st9_set:
        if qs is None:
            m[s * 3:(s + 1) * 3] = True
        else:
            for q in qs:
                m[s * 3 + q] = True
    wins = np.zeros((len(TARGETS), len(WINDOWS)), int)
    tots = np.zeros((len(TARGETS), len(WINDOWS)), int)
    for ti in range(len(TARGETS)):
        for wi in range(len(WINDOWS)):
            wins[ti, wi] = acc[side][ti][wi][0][m].sum()
            tots[ti, wi] = acc[side][ti][wi][1][m].sum()
    return wins, tots


def show(title, wins, tots, ev=False, sl=1.0):
    print(f"\n{title}")
    hdr = "        " + "".join(f"{('W='+str(W)):>16}" for W in WINDOWS)
    print(hdr)
    for ti, T in enumerate(TARGETS):
        line = f"T={T:<5}"
        for wi in range(len(WINDOWS)):
            tot = tots[ti, wi]
            w = wins[ti, wi]
            if tot < MIN_N:
                line += f"{'--('+str(tot)+')':>16}"
            elif ev:
                wr = w / tot
                e = wr * T - (1 - wr) * sl
                line += f"{e:>7.2f}EV({tot:>4})".rjust(16)
            else:
                line += f"{w/tot*100:>7.1f}%({tot:>4})".rjust(16)
        print(line)
    # 汇总行
    line = "W=全部  "
    for wi in range(len(WINDOWS)):
        tot = int(tots[:, wi].sum())
        w = int(wins[:, wi].sum())
        if tot < MIN_N:
            line += f"{'--('+str(tot)+')':>16}"
        else:
            line += f"{w/tot*100:>7.1f}%({tot:>4})".rjust(16)
    print(line)


LONG_OK = [0, 1, 2]   # 顺日线做多
LONG_KO = [6, 7, 8]   # 逆日线做多
SHORT_OK = [6, 7, 8]
SHORT_KO = [0, 1, 2]
LONG_EPI = [2]        # 日多4H空 = 插曲(沿日线做多)
SHORT_EPI = [6]       # 日空4H多 = 插曲(沿日线做空)

for side, dname, ok, ko, epi in (("long", "═══ 贴支撑未破 → 做多 ═══", LONG_OK, LONG_KO, LONG_EPI),
                                 ("short", "═══ 贴阻力未破 → 做空 ═══", SHORT_OK, SHORT_KO, SHORT_EPI)):
    print(f"\n{dname}")
    wins, tots = masked(s_acc, side, ok)
    show("对称1:1 · 顺日线", wins, tots)
    wins, tots = masked(s_acc, side, ko)
    show("对称1:1 · 逆日线", wins, tots)
    wins, tots = masked(s_acc, side, epi)
    show("对称1:1 · 插曲(沿日线)", wins, tots)
    wins, tots = masked(sl_acc, side, ok)
    show("固定SL=1ATR · 顺日线 EV", wins, tots, ev=True)
    wins, tots = masked(c_acc, side, ok)
    show("对照·未贴位 · 顺日线", wins, tots)
    for q, qn in Q3:
        wins, tots = masked(s_acc, side, ok, qs=[q])
        show(f"对称1:1 · 顺日线 · 质量={qn}", wins, tots)
