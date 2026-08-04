#!/usr/bin/env python3
"""B3 一般水平关键位特性泛化研究 (2026-08-04, 无未来函数)

动机: B2c/B2d 的约束/锚定结论仅限成对区间位带 (间距≤2.5×ATR)。本实验把
行为/约束度量推广到全部聚类位带 (区间内/宽成对/孤立), 并分层考察位带
属性 (触次/年龄/结构) 对约束强度的调节, 全部对标随机游走 (GBM)。

分层 (触碰时刻可知, 无未来函数):
  结构层: 区间内 (对侧≤2.5×ATR 且双方 60 根内无确认突破, 对齐 B2d)
         | 宽成对 (2.5<对侧≤5×ATR) | 孤立 (>5×ATR 或无对侧)
  触次层: 2~4 / 5~8 / ≥9        (文献: Chung-Bellotti 2021 触次增强)
  年龄层: <30 / 30~120 / >120   (文献: 同, 年龄衰减)
  波动层: 触碰时刻 vol_z(atr,120) 低/中/高

度量 (触碰后 24 根窗口, ATR 归一):
  M1 收盘仍在本侧比例 (R3 泛化)   M2 最大位移 P50/P90 (R2 泛化)
  M3 后12/前12根 ATR 比 (E1 按层) M5 拒绝率 (B2b 定义)
  M6 触碰时刻波动分布 vs 无条件   M4 约束场: 贴近(±0.5ATR) vs 远离 ATR 比

预注册假设:
  H1 约束普适: M1/M2 净差 (真实−GBM) 在孤立位也 >0, 强度随夹持减弱
  H2 触次增强: 触次越高 M1 净差越大
  H3 年龄衰减: 位越新 M1 净差越大
  H4 机械性: M4/M6 在 GBM 上同样存在 → 位-低波动关联是检测器机械产物

运行: python3 research/studies/b3_general_levels.py (约 30-60 分钟)
"""
import os
import sys
from bisect import bisect_right
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.data_loader import load_candles, verify
from research.levels import cluster_levels, levels_touch_class_all
from research.sim_market import gbm_dataframe
from research.state_features import vol_z_states

W = 24
DEPTH = 0.5
HOLD = 0.5
LIFE = 600
RANGE_ATR = 2.5
WIDE_ATR = 5.0
RANGE_BARS = 60
MIN_N = 200
PARAMS = [(2, 0.3), (3, 0.5)]   # (min_touch, tolerance_mult)
TIMEFRAMES = ["1h", "4h"]


def _load(tf):
    data = load_candles(timeframes=(tf,))
    out = []
    for sym, tfs in data.items():
        df = tfs.get(tf)
        if df is None:
            continue
        if verify(df, sym, tf):
            continue
        out.append(df)
    return out


def alive_at(conf, t):
    """位带过去 RANGE_BARS 根内无确认突破 (conf 已排序)"""
    i = bisect_right(conf, t - 1) - 1
    return not (i >= 0 and conf[i] >= t - RANGE_BARS)


def collect(df, mt, tol):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = _atr_series(df)
    n = len(c)
    lvls = cluster_levels(h, l, atr, min_touch=mt, tolerance_mult=tol)
    ev = levels_touch_class_all(lvls, c, h, l, atr, DEPTH, W, HOLD)
    sup = sorted([lv for lv in lvls if lv.side == "support"], key=lambda x: x.price)
    res = sorted([lv for lv in lvls if lv.side == "resistance"], key=lambda x: x.price)
    idx_of = {id(lv): k for k, lv in enumerate(lvls)}
    conf_of = {id(lv): ev["confirmed"][k] for k, lv in enumerate(lvls)}
    st, z = vol_z_states(atr, 120)

    # 逐 bar: 下方最近活跃支撑 / 上方最近活跃阻力 (对象 + 距离 + 存活)
    sup_p = np.array([lv.price for lv in sup])
    res_p = np.array([lv.price for lv in res])
    s_obj = np.full(n, None, dtype=object)
    r_obj = np.full(n, None, dtype=object)
    dn_dist = np.full(n, np.inf)
    up_dist = np.full(n, np.inf)
    active = np.zeros(n, bool)
    for t in range(n):
        k = bisect_right(sup_p, c[t]) - 1
        for i in range(k, -1, -1):
            lv = sup[i]
            if lv.confirm_at <= t < lv.confirm_at + LIFE:
                s_obj[t] = lv
                dn_dist[t] = c[t] - lv.price
                break
        k = bisect_right(res_p, c[t])
        for i in range(k, len(res)):
            lv = res[i]
            if lv.confirm_at <= t < lv.confirm_at + LIFE:
                r_obj[t] = lv
                up_dist[t] = lv.price - c[t]
                break
        if s_obj[t] is not None and r_obj[t] is not None:
            active[t] = (r_obj[t].price - s_obj[t].price) <= RANGE_ATR * atr[t]

    return dict(c=c, h=h, l=l, atr=atr, n=n, lvls=lvls, ev=ev, sup=sup, res=res,
                idx_of=idx_of, conf_of=conf_of, st=st, s_obj=s_obj, r_obj=r_obj,
                dn_dist=dn_dist, up_dist=up_dist, active=active)


BUCKETS = {"m1": [], "m2": [], "m3": [], "m5": []}
LAYER_KEYS = ("structure", "touch", "age", "vol")


def bucket_key(rec):
    """事件分层 → dict {layer: bucket} (触次/年龄/结构/波动)

    触次 = 当时累积触碰次数 = 聚类形成时 pivot 数 + 触碰事件序号
    (无未来函数; 注意 B2 的 ≥n 分层用全样本触碰总数, 含未来信息, 不可信)
    """
    t, lv, d, seq = rec["t"], rec["lv"], rec["d"], rec["seq"]
    s = {}
    # 结构层
    if lv.side == "support":
        dist = d["up_dist"][t]
    else:
        dist = d["dn_dist"][t]
    if d["active"][t]:
        opp = d["r_obj"][t] if lv.side == "support" else d["s_obj"][t]
        both_alive = (alive_at(d["conf_of"][id(lv)], t)
                      and alive_at(d["conf_of"][id(opp)], t))
        s["structure"] = "区间内" if both_alive else "孤立(对侧已破)"
    elif dist <= WIDE_ATR * d["atr"][t]:
        s["structure"] = "宽成对"
    else:
        s["structure"] = "孤立"
    # 触次层
    tc = lv.touch_count + seq
    s["touch"] = "2~4" if tc <= 4 else ("5~8" if tc <= 8 else "9+")
    # 年龄层
    age = t - lv.confirm_at
    s["age"] = "<30" if age < 30 else ("30~120" if age < 120 else ">120")
    # 波动层
    s["vol"] = d["st"][t]
    return s


def run_symbols(dfs, mt, tol, label=""):
    print(f"\n{'═' * 62}\nB3 {label} 参数(min_touch={mt}, tol={tol})\n{'═' * 62}")
    acc = {lk: defaultdict(list) for lk in LAYER_KEYS}
    for lk in LAYER_KEYS:
        for b in ("区间内", "宽成对", "孤立", "孤立(对侧已破)", "2~4", "5~8", "9+",
                  "<30", "30~120", ">120", "低", "中", "高", "未知"):
            acc[lk][b] = {k: [] for k in BUCKETS}
    vol_touch = defaultdict(int)
    vol_uncond = defaultdict(int)
    n_touch = n_near = n_far = 0
    sum_near = sum_far = 0.0
    n_level = n_reject = n_break = 0

    for df in dfs:
        d = collect(df, mt, tol)
        c, atr, n = d["c"], d["atr"], d["n"]
        # 全局: 位数/拒绝率 (B2b 口径)
        for i, lv in enumerate(d["lvls"]):
            n_level += 1
            n_reject += len(d["ev"]["reject"][i])
            n_break += len(d["ev"]["breakout"][i])
        # 无条件波动分布
        for v in d["st"]:
            vol_uncond[v] += 1
        # 约束场 (M4): 每位带活跃期 near/far
        for i, lv in enumerate(d["lvls"]):
            t0 = lv.confirm_at
            t1 = min(t0 + LIFE, n)
            if t1 <= t0:
                continue
            near_mask = np.abs(c[t0:t1] - lv.price) <= 0.5 * atr[t0:t1]
            a = atr[t0:t1]
            if near_mask.any():
                n_near += int(near_mask.sum())
                sum_near += float(a[near_mask].sum())
            far_mask = ~near_mask
            if far_mask.any():
                n_far += int(far_mask.sum())
                sum_far += float(a[far_mask].sum())
        # 触碰事件
        for i, lv in enumerate(d["lvls"]):
            t_arr = d["ev"]["touch"][i]
            if not len(t_arr):
                continue
            valid = (t_arr + W < n) & (t_arr >= 13) & (atr[t_arr] > 0)
            t_arr = t_arr[valid]
            if not len(t_arr):
                continue
            seg = c[t_arr[:, None] + np.arange(1, W + 1)]
            rej = d["ev"]["reject"][i]
            is_rej = np.isin(t_arr, rej)
            if lv.side == "support":
                m1 = (seg[:, -1] > lv.price)
                m2 = seg.max(axis=1) - c[t_arr]
            else:
                m1 = (seg[:, -1] < lv.price)
                m2 = c[t_arr] - seg.min(axis=1)
            m2 = m2 / atr[t_arr]
            a_prev = atr[t_arr[:, None] + np.arange(-12, 0)]
            a_next = atr[t_arr[:, None] + np.arange(1, 13)]
            m3 = a_next.mean(axis=1) / a_prev.mean(axis=1)
            n_touch += len(t_arr)
            for j, t in enumerate(t_arr):
                vol_touch[d["st"][t]] += 1
                rec = {"t": int(t), "lv": lv, "d": d, "seq": j + 1}
                bk = bucket_key(rec)
                for lk in LAYER_KEYS:
                    b = bk[lk]
                    acc[lk][b]["m1"].append(float(m1[j]))
                    acc[lk][b]["m2"].append(float(m2[j]))
                    acc[lk][b]["m3"].append(float(m3[j]))
                    acc[lk][b]["m5"].append(float(is_rej[j]))

    # ── 输出 ──
    print(f"位数 {n_level} | 触碰 {n_touch} | 拒绝率 {n_reject / max(1, n_touch):.1%} "
          f"| 突破率 {n_break / max(1, n_touch):.1%}")
    for lk in LAYER_KEYS:
        print(f"── {lk} ──")
        for b in acc[lk]:
            d = acc[lk][b]
            nn = len(d["m1"])
            if nn < MIN_N:
                print(f"  {b:<12} n={nn} 样本不足(<{MIN_N})")
                continue
            m1 = np.mean(d["m1"])
            m2 = np.asarray(d["m2"])
            m3 = np.mean(d["m3"])
            m5 = np.mean(d["m5"])
            print(f"  {b:<12} n={nn:>7} | M1 {m1:.1%} | M2P50 {np.median(m2):.2f} "
                  f"P90 {np.percentile(m2, 90):.2f} | E1 {m3:.3f} | 拒绝 {m5:.1%}")
    total_vol = max(1, sum(vol_touch.values()))
    print("── vol (触碰时刻分布) ──")
    print("  " + " | ".join(f"{k}: {v / total_vol:.1%}" for k, v in
                            sorted(vol_touch.items(), key=lambda x: -x[1])))
    print("  无条件: " + " | ".join(f"{k}: {v / max(1, sum(vol_uncond.values())):.1%}"
                                    for k, v in sorted(vol_uncond.items(), key=lambda x: -x[1])))
    near_mean = sum_near / n_near if n_near else float("nan")
    far_mean = sum_far / n_far if n_far else float("nan")
    print(f"── M4 约束场 ──")
    print(f"  贴近位带(±0.5ATR) ATR {near_mean:.3f} vs 远离 {far_mean:.3f} "
          f"| 比率 {near_mean / far_mean if far_mean else float('nan'):.3f} "
          f"(near {n_near} bars, far {n_far} bars)")
    return acc, dict(vol_touch=vol_touch, vol_uncond=vol_uncond,
                     n_reject=n_reject, n_touch=n_touch, n_level=n_level)


NET_LAYERS = [("structure", ["区间内", "宽成对", "孤立", "孤立(对侧已破)"]),
              ("touch", ["2~4", "5~8", "9+"]),
              ("age", ["<30", "30~120", ">120"])]


def diff_block(real, gbm, label=""):
    print(f"\n── 净差矩阵 (真实−GBM) {label} ──")
    print(f"{'层':<10}{'桶':<12}{'n_真实':>8}{'n_GBM':>8} {'M1Δ':>7} {'M2P50Δ':>8} "
          f"{'M2P90Δ':>8} {'E1Δ':>7} {'拒绝Δ':>7}")
    for lk, buckets in NET_LAYERS:
        for b in buckets:
            rd, gd = real[0][lk][b], gbm[0][lk][b]
            nn, gn = len(rd["m1"]), len(gd["m1"])
            if nn < MIN_N or gn < MIN_N:
                print(f"{lk:<10}{b:<12}{nn:>8}{gn:>8}   n 不足")
                continue
            d1 = np.mean(rd["m1"]) - np.mean(gd["m1"])
            d2p = np.median(rd["m2"]) - np.median(gd["m2"])
            d2q = np.percentile(rd["m2"], 90) - np.percentile(gd["m2"], 90)
            d3 = np.mean(rd["m3"]) - np.mean(gd["m3"])
            d5 = np.mean(rd["m5"]) - np.mean(gd["m5"])
            print(f"{lk:<10}{b:<12}{nn:>8}{gn:>8} {d1:>7.1%} {d2p:>8.3f} {d2q:>8.3f} "
                  f"{d3:>7.3f} {d5:>7.1%}")


def main():
    for mt, tol in PARAMS:
        for tf in TIMEFRAMES:
            print(f"\n########## 真实 {tf} (min_touch={mt}, tol={tol}) ##########")
            dfs = _load(tf)
            real = run_symbols(dfs, mt, tol, f"真实 {tf}")
            ref = dfs[0]
            sig = np.std(np.diff(np.log(ref["close"].values)))
            rw = [gbm_dataframe(len(ref), sig, seed=900 + (tf == "4h") * 100 + k)
                  for k in range(10)]
            gbm = run_symbols(rw, mt, tol, f"GBM {tf}")
            diff_block(real, gbm, f"{tf} 参数({mt},{tol})")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
