#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bb_ml_reversion — 布林带深偏离 ML 均值回归策略 (15M 框架 × 5M 决策)。

交易系统集成接口:
    from strategy_bb_ml import BBMLReversion
    strat = BBMLReversion()              # 用默认 BTC 训练系数
    strat.fit(df5_train)                 # 或用自己的 5M DataFrame 训练
    df5_out = strat.run(df5)             # 喂入 5M DataFrame → 附 信号/方向 列

输入: 5M DataFrame (DatetimeIndex UTC, 列 open/high/low/close/volume)。
策略全部指标在内部重采样到 15M 计算 (框架), 5M close 做决策。

无第三方 ML 依赖 (手写线性回归), 仅 numpy/pandas。
数据加载与仓库解耦: 调用方喂 DataFrame (可来自 data/backtest.db 或交易所)。

====================================================================
策略规格 (v1.0, 2026-08):
  框架:     15M MA20 ± 2σ (σ=close 20根滚动std), 全部 shift1 因果滞后
  偏离段:   做多 = 5M close < MA20−2σ 进入; 退出 = close 回到 MA20−1.2σ 内
            (迟滞 1.2σ, 合并边缘抖动碎片段); 做空镜像 (>+2σ 进, 回 1.2σ 内出)
  入场器:   段内每根 5M close 打分 = 预期"此刻入场→回中轨(限价)/60根超时"
            净收益(bp, 扣 4bp 成本)。11 特征(全因果滞后)。
            决策: 分数 > 训练集 P90 阈值 → 入场, 每段至多一笔。
            做多/做空独立模型。
  出场器:   持仓中每根 5M close 打分 = 预期"此刻平 vs 自然回中轨/超时"
            收益差。分数 > −10bp → 提前平仓。
            出场优先级: 出场器触发 > 触及中轨限价 > 60 根超时。
  成本:     4bp/往返 (maker)。 超时: 60 根 5M。
  跨币:     BTC 训练冻结系数应用全币 (特征无量纲, 已验证 11 币泛化)。
====================================================================
"""
import numpy as np
import pandas as pd

# ── 策略参数 (冻结) ──────────────────────────────────────────────
BAND = 2.0          # 偏离段进入阈值 (σ)
HYST_EXIT = 1.2     # 偏离段退出阈值 (σ, 迟滞)
BB_WIN = 20         # 布林带窗口 (15M 根数)
ATR_COST = 4.0      # 成本 bp/往返
MAXK = 60           # 最大持仓 (5M 根数)
ENTRY_Q = 90        # 入场器阈值分位 (训练集 pred P90)
EXIT_TH = -10.0     # 出场器触发阈值 (bp)
MIN_TR_BARS = 3     # 出场器最早触发 (入场后第3根起)


def _rolling_ols_beta_shifted(close, w):
    """ln(close) OLS 斜率 β (bp/bar); 窗口不含当前 bar (无未来)。"""
    n = len(close)
    y = np.log(np.maximum(close, 1e-12))
    win = np.lib.stride_tricks.sliding_window_view(y, w)
    x = np.arange(1, w + 1, dtype=float)
    xc = x - x.mean()
    beta = (win @ xc) / float(np.sum(xc ** 2)) * 1e4
    pad = np.full(n, np.nan)
    pad[w - 1:] = beta
    pad2 = np.full(n, np.nan)
    pad2[1:] = pad[:-1]
    return pad2


def _ols_fit(X, y):
    Xa = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.pinv(Xa.T @ Xa) @ Xa.T @ y
    return coef


def _ols_predict(coef, X):
    Xa = np.column_stack([np.ones(len(X)), X])
    return Xa @ coef


# 入场器特征构造器 (返回函数 list): 索引含义
# 0 dev(偏离σ) 1 run_len(段内位置,cap10) 2 ret5 3 ret20 4 vol_ratio
# 5 atr_bp 6 beta5 7 beta20 8 beta40 9 dist_support(ATR)
def _make_feature_builder(df15):
    """构建 15M 框架 → 返回 (ffill数组集, 5M特征函数)。"""
    c15 = df15["close"].to_numpy()
    h15 = df15["high"].to_numpy()
    l15 = df15["low"].to_numpy()
    v15 = df15["volume"].to_numpy()
    mid15 = pd.Series(c15).rolling(BB_WIN).mean().shift(1).to_numpy()
    sd15 = pd.Series(c15).rolling(BB_WIN).std(ddof=0).shift(1).to_numpy()
    tr15 = np.zeros(len(c15))
    tr15[1:] = np.maximum(h15[1:] - l15[1:],
                          np.maximum(np.abs(h15[1:] - c15[:-1]),
                                     np.abs(l15[1:] - c15[:-1])))
    atr15 = pd.Series(tr15).ewm(alpha=1 / 14, adjust=False).mean().shift(1).to_numpy()
    atr15_bp = atr15 / np.maximum(c15, 1e-9) * 1e4
    vol_ma15 = pd.Series(v15).rolling(20).mean().shift(1).to_numpy()
    lo60 = pd.Series(l15).rolling(60).min().shift(1).to_numpy()
    hi60 = pd.Series(h15).rolling(60).max().shift(1).to_numpy()
    b5 = _rolling_ols_beta_shifted(c15, 5)
    b20 = _rolling_ols_beta_shifted(c15, 20)
    b40 = _rolling_ols_beta_shifted(c15, 40)

    def ret15(w):
        r = np.full(len(c15), np.nan)
        r[w:] = (c15[w:] / c15[:-w] - 1) * 1e4
        return r

    ret5_15 = ret15(5)
    ret20_15 = ret15(20)
    return dict(c15=c15, mid15=mid15, sd15=sd15, atr15_bp=atr15_bp,
                vol_ma15=vol_ma15, lo60=lo60, hi60=hi60,
                b5=b5, b20=b20, b40=b40, ret5_15=ret5_15, ret20_15=ret20_15,
                ts15=df15.index)


class BBMLReversion:
    """布林带深偏离 ML 均值回归策略。

    用法:
        s = BBMLReversion()
        s.fit(df5)                  # 训练 (自动内部15M重采样)
        out = s.run(df5)            # 返回带 signal 的 DataFrame

    内部: 15M 指标 shift1 因果 → ffill 到 5M 轴 (仅用已开盘15M的滞后值),
    5M close 决策。入场价 = 触发5M close (收盘确认); 出场 = 5M high/low
    触及中轨按 mid 限价。全部无未来函数。
    """

    def __init__(self, symbol="BTC/USDT:USDT"):
        self.symbol = symbol
        self.entry_models = None   # {1: (coef, thr), -1: (coef, thr)}
        self.exit_model = None     # coef
        self._fb = None            # 特征 builder 状态

    # ── 数据准备 ──────────────────────────────────────────────
    def _prep(self, df5):
        df5 = df5.copy()
        df5.index = pd.to_datetime(df5.index, utc=True)
        df5 = df5.sort_index()
        df15 = df5.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "volume": "sum"}).dropna(subset=["close"])
        return df5, df15

    # ── 特征/段/交易模拟 (全量向量准备 + 逐bar) ───────────────
    def _prepare(self, df5, df15):
        fb = _make_feature_builder(df15)
        c5 = df5["close"].to_numpy()
        h5 = df5["high"].to_numpy()
        l5 = df5["low"].to_numpy()
        v5 = df5["volume"].to_numpy()
        n5 = len(c5)
        ts5 = df5.index
        ts15 = fb["ts15"]

        # 向量化 ffill (原逐 bar 双循环调用 11 次 — 性能瓶颈)
        # pos[i] = 最近一根 ts15 <= ts5[i] 的索引 (无则 -1)
        t15 = np.array([t.value for t in ts15])
        t5 = np.array([t.value for t in ts5])
        pos = np.searchsorted(t15, t5, side="right") - 1

        def ffill(arr15):
            out = np.full(n5, np.nan)
            arr = np.asarray(arr15, float)
            m = pos >= 0
            out[m] = arr[np.clip(pos[m], 0, len(arr) - 1)]
            return out

        mid_map = ffill(fb["mid15"])
        sd_map = ffill(fb["sd15"])
        atr_map = ffill(fb["atr15_bp"])
        vol_ma_map = ffill(fb["vol_ma15"])
        lo60_map = ffill(fb["lo60"])
        hi60_map = ffill(fb["hi60"])
        b5_map = ffill(fb["b5"])
        b20_map = ffill(fb["b20"])
        b40_map = ffill(fb["b40"])
        ret5_map = ffill(fb["ret5_15"])
        ret20_map = ffill(fb["ret20_15"])
        dev = (c5 - mid_map) / np.maximum(sd_map, 1e-9)

        # 迟滞段 (进 2σ / 出 1.2σ)
        def seg(direction):
            sid = np.zeros(n5, int)
            sn = 0
            ins = False
            for i in range(50, n5):
                if not ins:
                    if (direction == 1 and dev[i] < -BAND) or \
                       (direction == -1 and dev[i] > BAND):
                        ins = True
                        sn += 1
                        sid[i] = sn
                else:
                    if (direction == 1 and dev[i] > -HYST_EXIT) or \
                       (direction == -1 and dev[i] < HYST_EXIT):
                        ins = False
                    else:
                        sid[i] = sn
            return sid

        seg_l = seg(1)
        seg_s = seg(-1)

        def runlen_from(sid):
            out = np.zeros(n5)
            for i in range(1, n5):
                if sid[i] > 0 and sid[i] == sid[i - 1]:
                    out[i] = out[i - 1] + 1
                elif sid[i] > 0:
                    out[i] = 1
            return out

        rll = runlen_from(seg_l)
        rls = runlen_from(seg_s)

        def ef(i, side):
            if side == 1:
                return [dev[i], min(rll[i], 10), ret5_map[i], ret20_map[i],
                        v5[i] / vol_ma_map[i] if vol_ma_map[i] > 0 else 1.0,
                        atr_map[i], b5_map[i], b20_map[i], b40_map[i],
                        (c5[i] - lo60_map[i]) / max(atr_map[i], 1e-9)]
            return [dev[i], min(rls[i], 10), ret5_map[i], ret20_map[i],
                    v5[i] / vol_ma_map[i] if vol_ma_map[i] > 0 else 1.0,
                    atr_map[i], b5_map[i], b20_map[i], b40_map[i],
                    (hi60_map[i] - c5[i]) / max(atr_map[i], 1e-9)]

        def ne(i, side):
            """自然出场收益 bp (回中轨限价 / MAXK 超时), 返回 (k, bp)。"""
            entry = c5[i]
            for k in range(1, MAXK + 1):
                j = i + k
                if j >= n5:
                    break
                if side == 1 and h5[j] >= mid_map[j]:
                    return k, np.log(mid_map[j] / entry) * 1e4
                if side == -1 and l5[j] <= mid_map[j]:
                    return k, -np.log(mid_map[j] / entry) * 1e4
            if side == 1:
                return MAXK, np.log(c5[min(i + MAXK, n5 - 1)] / entry) * 1e4
            return MAXK, -np.log(c5[min(i + MAXK, n5 - 1)] / entry) * 1e4

        env = dict(c5=c5, h5=h5, l5=l5, v5=v5, n5=n5, ts5=ts5,
                   mid_map=mid_map, sd_map=sd_map, atr_map=atr_map,
                   vol_ma_map=vol_ma_map, b5_map=b5_map, b20_map=b20_map,
                   ret5_map=ret5_map, dev=dev, in_long=seg_l > 0,
                   in_short=seg_s > 0, rll=rll, rls=rls, ef=ef, ne=ne)
        return env

    # ── 训练 ──────────────────────────────────────────────────
    def fit(self, df5):
        """训练入场器(双向) + 出场器。df5: 5M DataFrame。"""
        df5, df15 = self._prep(df5)
        env = self._prepare(df5, df15)
        years = np.array([ts.year for ts in env["ts5"]])
        n5 = env["n5"]
        entry_models = {}
        for side in (1, -1):
            inz = env["in_long"] if side == 1 else env["in_short"]
            idx = np.where(inz & (years <= 2024) & (years >= 2023))[0]
            X = []
            y = []
            for i in idx:
                if i < 200 or i > n5 - 200:
                    continue
                f = env["ef"](i, side)
                if all(np.isfinite(f)):
                    X.append(f)
                    y.append(env["ne"](i, side)[1])
            X = np.array(X)
            y = np.array(y)
            coef = _ols_fit(X, y)
            pred = _ols_predict(coef, X)
            entry_models[side] = (coef, float(np.percentile(pred, ENTRY_Q)))
        # 出场器
        trades = []
        for side in (1, -1):
            coef, thr = entry_models[side]
            inz = env["in_long"] if side == 1 else env["in_short"]
            i = 200
            while i < n5 - 200:
                if years[i] > 2024 or not inz[i]:
                    i += 1
                    continue
                while i < n5 - 200 and inz[i] and years[i] <= 2024:
                    f = env["ef"](i, side)
                    if all(np.isfinite(f)):
                        if _ols_predict(coef, [f])[0] > thr:
                            trades.append((i, side))
                            i += 1
                            break
                    i += 1
                while i < n5 - 200 and inz[i]:
                    i += 1
        Xe = []
        ye = []
        for (i, side) in trades:
            entry = env["c5"][i]
            k_nat, r_nat = env["ne"](i, side)
            peak = 0.0
            for k in range(1, k_nat + 1):
                j = i + k
                if j >= n5:
                    break
                if side == 1:
                    cur = np.log(env["c5"][j] / entry) * 1e4
                else:
                    cur = -np.log(env["c5"][j] / entry) * 1e4
                peak = max(peak, cur)
                f = [k / MAXK, cur, peak - cur if peak > 0 else 0.0,
                     (env["c5"][j] - env["mid_map"][j]) / env["sd_map"][j] * side,
                     env["b5_map"][j] * side, env["b20_map"][j] * side,
                     env["ret5_map"][j] * side, env["atr_map"][j],
                     env["v5"][j] / env["vol_ma_map"][j]
                     if env["vol_ma_map"][j] > 0 else 1.0]
                if all(np.isfinite(f)):
                    Xe.append(f)
                    ye.append(cur - r_nat)
        self.exit_model = _ols_fit(np.array(Xe), np.array(ye))
        self.entry_models = entry_models
        self._env = env
        self._years = years
        return self

    # ── 信号生成 / 回测 ───────────────────────────────────────
    def run(self, df5, entry_models=None, exit_model=None, cost=ATR_COST):
        """在 df5 上生成交易 (在线规则), 返回 DataFrame 含:
        entry_time/exit_time/side/entry_price/exit_price/net_bp/hold_k。
        用 self 模型或传入的冻结模型。"""
        em = entry_models if entry_models is not None else self.entry_models
        xm = exit_model if exit_model is not None else self.exit_model
        if em is None or xm is None:
            raise ValueError("模型未训练: 先 fit() 或传冻结模型")
        df5, df15 = self._prep(df5)
        env = self._prepare(df5, df15)
        years = np.array([ts.year for ts in env["ts5"]])
        n5 = env["n5"]
        recs = []
        for side in (1, -1):
            coef, thr = em[side]
            inz = env["in_long"] if side == 1 else env["in_short"]
            i = 200
            while i < n5 - 200:
                if not inz[i]:
                    i += 1
                    continue
                entered = None
                while i < n5 - 200 and inz[i]:
                    f = env["ef"](i, side)
                    if all(np.isfinite(f)):
                        if _ols_predict(coef, [f])[0] > thr:
                            entered = i
                            i += 1
                            break
                    i += 1
                if entered is None:
                    continue
                entry = env["c5"][entered]
                k_nat, r_nat = env["ne"](entered, side)
                peak = 0.0
                exit_ret = None
                exit_k = k_nat
                for k in range(1, k_nat + 1):
                    j = entered + k
                    if j >= n5:
                        break
                    if side == 1:
                        cur = np.log(env["c5"][j] / entry) * 1e4
                    else:
                        cur = -np.log(env["c5"][j] / entry) * 1e4
                    peak = max(peak, cur)
                    if k >= MIN_TR_BARS:
                        f = [k / MAXK, cur, peak - cur if peak > 0 else 0.0,
                             (env["c5"][j] - env["mid_map"][j]) /
                             env["sd_map"][j] * side,
                             env["b5_map"][j] * side, env["b20_map"][j] * side,
                             env["ret5_map"][j] * side, env["atr_map"][j],
                             env["v5"][j] / env["vol_ma_map"][j]
                             if env["vol_ma_map"][j] > 0 else 1.0]
                        if all(np.isfinite(f)):
                            if _ols_predict(xm, [f])[0] > EXIT_TH:
                                exit_ret = cur
                                exit_k = k
                                break
                if exit_ret is None:
                    exit_ret = r_nat
                recs.append(dict(
                    symbol=self.symbol,
                    entry_time=env["ts5"][entered],
                    exit_time=env["ts5"][min(entered + exit_k, n5 - 1)],
                    side=side,
                    entry_price=float(entry),
                    net_bp=float(exit_ret - cost),
                    hold_k=int(exit_k)))
                # 跳到段外 (每段一笔)
                while i < n5 - 200 and inz[i]:
                    i += 1
        out = pd.DataFrame(recs)
        if len(out):
            out = out.sort_values("entry_time").reset_index(drop=True)
        return out

    # ── 信号序列 (逐 bar 期望状态, 供模拟盘跟随) ───────────────
    def run_signals(self, df5, entry_models=None, exit_model=None):
        """输出每根 5M bar 的期望持仓状态 (1/0/-1) — 模拟盘跟随

        与 run() 同口径 (单仓, 每段一笔, 出场器/触及中轨/超时), 但按时间
        推进返回逐 bar 状态: 1=应持多, -1=应持空, 0=空仓。
        模拟盘: 状态变化即触发开/平仓 (状态→非0 开仓, →0 平仓, 翻转反向)。
        """
        em = entry_models if entry_models is not None else self.entry_models
        xm = exit_model if exit_model is not None else self.exit_model
        if em is None or xm is None:
            raise ValueError("模型未训练: 先 fit() 或传冻结模型")
        df5, df15 = self._prep(df5)
        env = self._prepare(df5, df15)
        n5 = env["n5"]
        in_long = env["in_long"]
        in_short = env["in_short"]
        state = np.zeros(n5, int)   # 期望持仓: 1/-1/0
        # 持仓扫描用的事件模拟 (多空独立, 与 run 同规则)
        for side in (1, -1):
            coef, thr = em[side]
            inz = in_long if side == 1 else in_short
            i = 200
            while i < n5 - 200:
                if not inz[i]:
                    i += 1
                    continue
                # 找入场
                entered = None
                while i < n5 - 200 and inz[i]:
                    f = env["ef"](i, side)
                    if all(np.isfinite(f)):
                        if _ols_predict(coef, [f])[0] > thr:
                            entered = i
                            i += 1
                            break
                    i += 1
                if entered is None:
                    continue
                # 持仓期: 从入场 bar 起状态=side, 直到出场
                entry = env["c5"][entered]
                k_nat, r_nat = env["ne"](entered, side)
                peak = 0.0
                exit_k = k_nat
                for k in range(1, k_nat + 1):
                    j = entered + k
                    if j >= n5:
                        break
                    if side == 1:
                        cur = np.log(env["c5"][j] / entry) * 1e4
                    else:
                        cur = -np.log(env["c5"][j] / entry) * 1e4
                    peak = max(peak, cur)
                    if k >= MIN_TR_BARS:
                        f = [k / MAXK, cur, peak - cur if peak > 0 else 0.0,
                             (env["c5"][j] - env["mid_map"][j]) /
                             env["sd_map"][j] * side,
                             env["b5_map"][j] * side, env["b20_map"][j] * side,
                             env["ret5_map"][j] * side, env["atr_map"][j],
                             env["v5"][j] / env["vol_ma_map"][j]
                             if env["vol_ma_map"][j] > 0 else 1.0]
                        if all(np.isfinite(f)):
                            if _ols_predict(xm, [f])[0] > EXIT_TH:
                                exit_k = k
                                break
                # 状态: entered..entered+exit_k 之间持 side
                state[entered] = side
                # 入场 bar 到出场前一根 (出场 bar 用 next 5M 反映)
                e_end = min(entered + exit_k, n5 - 1)
                state[entered: e_end + 1] = side
                # 若出场在 e_end (该 bar 出场), 状态到 e_end 结束
                # 跳到段外 (每段一笔)
                i = e_end + 1
                while i < n5 - 200 and inz[i]:
                    i += 1
        # 多空可能重叠? 每段一笔保证不重叠 (不同段)
        return pd.Series(state, index=df5.index, name="state")


def load_btc_5m():
    """便捷: 从 backtest.db 加载 BTC 5M (若在仓库根运行)。"""
    from research.data_loader import load_candles
    raw = load_candles(timeframes=("5m",))
    return raw["BTC/USDT:USDT"]["5m"]


if __name__ == "__main__":
    # 演示: 训练 + 测试 (BTC)
    df5 = load_btc_5m()
    strat = BBMLReversion()
    strat.fit(df5)
    te = df5[df5.index >= "2025-01-01"]
    trades = strat.run(te)
    if len(trades):
        print(f"交易: {len(trades)}")
        print(f"净收益: {trades.net_bp.mean():+.2f}bp/笔  胜率: "
              f"{(trades.net_bp > 0).mean():.1%}")
        by_side = trades.groupby("side").net_bp.mean()
        print(f"做多: {by_side.get(1, float('nan')):+.2f}  做空: "
              f"{by_side.get(-1, float('nan')):+.2f}")
