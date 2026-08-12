#!/usr/bin/env python3
"""B2 关键水平位 — 形成与效果基础分析 (2026-08-03, 无未来函数, 流式聚合防 OOM)

定义 (与用户确认):
  穿透: close 越出位带外侧 ≥ 0.5×ATR
  真突破 (失效): 穿透后未来 24 根内 close 保持外侧比例 ≥ 50%
  假突破: 穿透但未确认 (位保持有效)
  有效拒绝: 触碰后 24 根内无确认突破

位质量分层: 形成时触次 ≥2 / ≥3 / ≥5 — 检验"多触碰的位是否更有效"
分析 (描述性口径): F1 形成环境 / F3 位密度 / E1 触碰前后波动 /
  E2 拒绝后反弹位移 / E4 真/假突破后位移; 全部带随机游走对照
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.data_loader import load_candles, verify
from research.levels import cluster_levels, levels_touch_class_all
from research.sim_market import gbm_dataframe
from scipy.stats import skew

W = 24
DEPTH = 0.5
HOLD = 0.5
MIN_TIERS = [2, 3, 5]


def _load():
    data = load_candles(timeframes=("1h",))
    out = []
    for sym, tfs in data.items():
        df = tfs.get("1h")
        if df is None:
            continue
        if verify(df, sym, "1h"):
            continue
        out.append(df)
    return out


def flatten_idxs(list_of_arrays):
    return np.concatenate(list_of_arrays) if list_of_arrays else np.array([])


def max_rev_vec(c, atr, idxs, w, cap=2000000):
    idxs = np.asarray(idxs)
    idxs = idxs[(idxs >= 0) & (idxs + w < len(c)) & (atr[idxs] > 0)]
    if len(idxs) == 0:
        return np.array([])
    if len(idxs) > cap:
        rng = np.random.default_rng(1)
        idxs = rng.choice(idxs, cap, replace=False)
    rows = np.arange(w)[None, :] + 1 + idxs[:, None]
    seg = c[rows]
    return (seg.max(axis=1) - c[idxs]) / atr[idxs]


def fwd_disp_vec(c, atr, idxs, w, cap=200000):
    idxs = np.asarray(idxs)
    idxs = idxs[(idxs >= 0) & (idxs + w < len(c)) & (atr[idxs] > 0)]
    if len(idxs) == 0:
        return np.array([])
    if len(idxs) > cap:
        rng = np.random.default_rng(0)
        idxs = rng.choice(idxs, cap, replace=False)
    return (c[idxs + w] - c[idxs]) / atr[idxs]


def run_block(dfs, label=""):
    print(f"═══ B2 关键水平位分析 {label}═══\n")
    all_atr = np.concatenate([_atr_series(df) for df in dfs])
    a1, a2 = np.quantile(all_atr, [1 / 3, 2 / 3])
    del all_atr

    # 每层统计
    layers = {t: dict(cnt=dict(touch=0, reject=0, breakout=0, attempt=0,
                               confirmed=0, false_break=0, lvls=0),
                      env={"低": 0, "中": 0, "高": 0},
                      pre=[], post=[], rej=[], conf=[], fals=[], gaps=[])
              for t in MIN_TIERS}

    for df in dfs:
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        atr = _atr_series(df)
        n = len(c)
        lvls = cluster_levels(h, l, atr, min_touch=2)
        for t in MIN_TIERS:
            ly = layers[t]
            ev = levels_touch_class_all(lvls, c, h, l, atr, DEPTH, W, HOLD)
            # 分层按"该位总触碰次数"(含未来触碰, 事后质量标签, 合法)
            keep = [i for i, x in enumerate(ev["touch"]) if len(x) >= t]
            if not keep:
                continue
            sub = [lvls[i] for i in keep]
            ev = {k: [ev[k][i] for i in keep] for k in ev}
            ly["cnt"]["lvls"] += len(sub)
            for k in ("touch", "reject", "breakout", "attempt", "confirmed", "false_break"):
                ly["cnt"][k] += sum(len(x) for x in ev[k])
            for lv in sub:
                if lv.confirm_at >= 60:
                    m = np.mean(atr[lv.confirm_at - 60:lv.confirm_at])
                    ly["env"]["低" if m < a1 else "高" if m > a2 else "中"] += 1
            cts = sorted(lv.confirm_at for lv in sub)
            if len(cts) >= 2:
                ly["gaps"].append(np.diff(cts))
            for idxs in ev["touch"]:
                for tt in idxs:
                    if 12 <= tt < n - 12 and atr[tt] > 0:
                        ly["pre"].append(atr[tt - 12:tt] / atr[tt])
                        ly["post"].append(atr[tt + 1:tt + 13] / atr[tt])
            ly["rej"].append(max_rev_vec(c, atr, flatten_idxs(ev["reject"]), W))
            ly["conf"].append(fwd_disp_vec(c, atr, flatten_idxs(ev["confirmed"]), W))
            ly["fals"].append(fwd_disp_vec(c, atr, flatten_idxs(ev["false_break"]), W))

    for t in MIN_TIERS:
        ly = layers[t]
        cnt = ly["cnt"]
        print(f"── 位质量层: 形成触次 ≥{t} (n={cnt['lvls']} 位) ──")
        if cnt["touch"]:
            print(f"  触碰 {cnt['touch']}: 拒绝 {cnt['reject']} ({cnt['reject']/cnt['touch']:.1%}) "
                  f"| 突破 {cnt['breakout']} ({cnt['breakout']/cnt['touch']:.1%})")
        if cnt["attempt"]:
            print(f"  穿透 {cnt['attempt']}: 确认 {cnt['confirmed']} "
                  f"({cnt['confirmed']/cnt['attempt']:.1%}) | 假突破 {cnt['false_break']} "
                  f"({cnt['false_break']/cnt['attempt']:.1%})")
        tot_env = sum(ly["env"].values())
        if tot_env:
            print(f"  F1 形成环境: 低 {ly['env']['低']/tot_env:.1%} | 中 {ly['env']['中']/tot_env:.1%} "
                  f"| 高 {ly['env']['高']/tot_env:.1%}")
        if ly["gaps"]:
            g = np.concatenate(ly["gaps"])
            print(f"  F3 形成间隔中位 {np.median(g):.0f} 根")
        if ly["pre"]:
            pre = np.mean(np.stack(ly["pre"]), axis=0)
            post = np.mean(np.stack(ly["post"]), axis=0)
            print(f"  E1 触碰前后 ATR: {pre.mean():.3f} → {post.mean():.3f} "
                  f"(后12根/前12根 = {post.mean()/pre.mean():.3f})")
        rej = np.concatenate(ly["rej"]) if ly["rej"] else np.array([])
        if len(rej) > 100:
            print(f"  E2 拒绝后 24根反向位移: {rej.mean():.3f} ATR (偏度 {skew(rej):.2f})")
        conf = np.concatenate(ly["conf"]) if ly["conf"] else np.array([])
        fals = np.concatenate(ly["fals"]) if ly["fals"] else np.array([])
        if len(conf) > 50:
            print(f"  E4 真突破后 24根位移: {conf.mean():+.3f} ATR (n={len(conf)})")
        if len(fals) > 50:
            print(f"  E4 假突破后 24根位移: {fals.mean():+.3f} ATR (n={len(fals)})")
        print()

    # 无条件基线 (每 24 根抽 1 点) — 供 E2 对照
    base_all = []
    for df in dfs:
        c = df["close"].values
        atr = _atr_series(df)
        t = np.arange(60, len(c) - W, 24)
        if len(t):
            rows = np.arange(W)[None, :] + 1 + t[:, None]
            base_all.append(((c[rows].max(axis=1) - c[t]) / atr[t]))
    if base_all:
        b = np.concatenate(base_all)
        print(f"  无条件 24根反向位移: {b.mean():.3f} ATR (n={len(b)})\n")


def main():
    dfs = _load()
    run_block(dfs)
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=400 + k) for k in range(10)]
    run_block(rw, label="(随机游走对照)")


if __name__ == "__main__":
    main()
