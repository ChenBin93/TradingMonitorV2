#!/usr/bin/env python3
"""BBML Reversion 策略 — 增量式引擎 (live_trader paper/okx 共用)

从 bb_ml_reversion/strategy_bb_ml.py 的全量回放改为逐 bar 增量决策。
口径与回测逐位一致 (README §5 审计要点):
- 15M 框架指标全部 shift(1) 因果滞后; 5M close 决策时 ffill 用最新
  已开盘 15M bar 的滞后值
- 偏离段状态机: 做多段 close < MA20−2σ 进, 回 MA20−1.2σ 内出 (迟滞)
- 入场器: 段内每 5M close 打分 > 训练 P90 阈值 → 入场 (每段至多一笔)
- 出场器: 持仓第 3 根起打分 > −10bp → 提前平; 否则触及中轨限价 /
  60 根 5M 超时
- 入场价 = 信号 bar close (收盘确认); 出场 = mid 触及价 / close

用法:
    engine = BBMLIncremental(models_path)
    engine.on_bar(bar_dict)   # 每根新 5M bar 收盘调用
    → {"action": "none"|"enter_long"|"enter_short"|"exit", ...}
"""
import json
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("trader.bbml")

# ── 参数 (冻结, 与 strategy_bb_ml.py 一致) ─────────────────
BAND = 2.0
HYST_EXIT = 1.2
BB_WIN = 20
MAXK = 60
ENTRY_Q = 90
EXIT_TH = -10.0
MIN_TR_BARS = 3


def _ols_predict(coef, feats):
    """coef[0]=截距, 后续=特征系数"""
    return coef[0] + np.dot(coef[1:], feats)


def _rolling_ols_beta_shifted(close_vals, w):
    """ln(close) OLS 斜率 β (bp/bar) — 回测 shift1 语义

    与 strategy_bb_ml 一致: 滚动窗口不含当前 bar (shift(1)),
    即用最近 w 根但不含最新已收盘块。
    """
    n = len(close_vals)
    y = np.log(np.maximum(close_vals, 1e-12))
    if n < w + 2:
        return np.nan
    win = y[-(w + 1):-1]  # 不含最新块, 取之前 w 根 (shift1)
    x = np.arange(1, w + 1, dtype=float)
    xc = x - x.mean()
    return float(np.dot(win, xc) / np.sum(xc ** 2) * 1e4)


class BBMLIncremental:
    """增量状态机 — 每根 5M bar 收盘调用 on_bar()"""

    def __init__(self, models_path: str | None = None, warmup_15m: int = 200):
        if models_path is None:
            models_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "bb_ml_reversion", "models_btc.json")
        with open(models_path) as f:
            m = json.load(f)
        self.entry_models = {
            int(k): (np.array(v["coef"]), float(v["threshold"]))
            for k, v in m["entry_models"].items()}
        self.exit_model = np.array(m["exit_model"]["coef"])
        self.warmup_15m = warmup_15m

        # 5M bar 累积 (只留尾部足够计算 15M 指标)
        self._ts5 = []
        self._c5 = []
        self._h5 = []
        self._l5 = []
        self._v5 = []

        # 15M bar 累积 (重采样后)
        self._ts15 = []
        self._c15 = []
        self._h15 = []
        self._l15 = []
        self._v15 = []

        # 当前 15M 聚合块 (未完成)
        self._cur15 = None
        self._cur15_ts = None

        # 偏离段状态
        self.seg_side = 0       # 1 / -1 / 0
        self.seg_bars = 0       # 段内 5M 根数
        self.entered_this_seg = False  # 本段是否已入场

        # 持仓状态
        self.position = None    # {side, entry_price, entry_idx, entry_time}

    # ── 15M 重采样 (按 15min 边界) ──────────────────────────
    def _append_5m(self, bar: dict):
        ts = bar["timestamp"]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        self._ts5.append(ts)
        self._c5.append(float(bar["close"]))
        self._h5.append(float(bar["high"]))
        self._l5.append(float(bar["low"]))
        self._v5.append(float(bar.get("volume", 0.0)))
        # 裁剪: 保留最近 ~3000 根 5M (= 足够 40+ 根 15M + 余量)
        if len(self._c5) > 3000:
            self._ts5 = self._ts5[-3000:]
            self._c5 = self._c5[-3000:]
            self._h5 = self._h5[-3000:]
            self._l5 = self._l5[-3000:]
            self._v5 = self._v5[-3000:]
        # 15M 聚合 (增量)
        self._acc_15m(ts, bar)

    def _acc_15m(self, ts: pd.Timestamp, bar: dict):
        """增量 15M 聚合 — OKX 5M bar 开盘分钟 %15==10 时, 前 15M 块完成

        块 [t, t+15) 含 5M 开盘 t/t+5/t+10; 收到开盘 t+10 的 5M 收盘后块完成。
        聚合: open=第1根open, high=max, low=min, close=第3根close, vol=sum。
        """
        minute = ts.minute
        # 当前 5M 属于哪个 15M 块 (块开盘时刻)
        block_start = ts.floor("15min")
        is_block_3rd = (minute % 15 == 10)  # xx:10/25/40/55 → 块内第3根

        # 累计到当前块
        if (self._cur15_ts is None or
                self._cur15_ts != block_start):
            # 新块开始
            self._cur15_ts = block_start
            self._cur15 = {"open": float(bar["open"]), "high": float(bar["high"]),
                           "low": float(bar["low"]), "close": float(bar["close"]),
                           "vol": float(bar.get("volume", 0.0))}
        else:
            c = self._cur15
            c["high"] = max(c["high"], float(bar["high"]))
            c["low"] = min(c["low"], float(bar["low"]))
            c["close"] = float(bar["close"])
            c["vol"] += float(bar.get("volume", 0.0))

        # 块完成 → 提交
        if is_block_3rd and self._cur15 is not None:
            self._ts15.append(self._cur15_ts)
            self._c15.append(self._cur15["close"])
            self._h15.append(self._cur15["high"])
            self._l15.append(self._cur15["low"])
            self._v15.append(self._cur15["vol"])
            self._cur15 = None
            self._cur15_ts = None
            if len(self._ts15) > 1000:
                self._ts15 = self._ts15[-1000:]
                self._c15 = self._c15[-1000:]
                self._h15 = self._h15[-1000:]
                self._l15 = self._l15[-1000:]
                self._v15 = self._v15[-1000:]

    # ── 15M 指标 (回测 shift1 口径) ─────────────────────────
    def _ind15(self):
        """返回最新已收盘 15M 块的 shift1 指标 (决策时刻可见, 与回测一致)

        回测: 指标 = rolling().shift(1), ffill 到 5M。即对最新已收盘块
        last, 指标用不含 last 的窗口 (mid = 前20根均值, 不含 last)。
        引擎: 收到第3根5M (xx:10) 收盘 → 提交 last 块 → 此处算 shift1
        指标 (窗口 arr[-(w+1):-1] 不含 last)。ATR ewm 同回测全序列。
        """
        n = len(self._c15)
        if n < BB_WIN + 6:
            return None
        c = np.array(self._c15)
        h = np.array(self._h15)
        l = np.array(self._l15)
        v = np.array(self._v15)
        # shift1: 窗口不含最新块 last (= arr[-(w+1):-1])
        def ma(arr, w):
            return float(np.mean(arr[-(w + 1):-1])) if len(arr) >= w + 1 else np.nan
        mid = ma(c, BB_WIN)
        sd = float(np.std(c[-(BB_WIN + 1):-1], ddof=0)) if len(c) >= BB_WIN + 1 else np.nan
        # ATR: 回测是 rolling(TR).ewm 全序列后 shift1 — 这里 ewm 到 last-1
        pc = np.empty(n)
        pc[0] = c[0]
        pc[1:] = c[:-1]
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        atr_series = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean()
        atr = float(atr_series.iloc[-2])  # shift1: last-1 的值
        atr_bp = atr / max(mid, 1e-9) * 1e4 if mid else np.nan
        vol_ma = float(np.mean(v[-(BB_WIN + 1):-1])) if len(v) >= BB_WIN + 1 else np.nan
        # β (shift1: 不含 last)
        b5 = _rolling_ols_beta_shifted(c, 5)
        b20 = _rolling_ols_beta_shifted(c, 20)
        b40 = _rolling_ols_beta_shifted(c, 40)
        # ret (不含 last: c[-1] 是 last, 用 c[-2])
        def ret(w):
            if len(c) < w + 2:
                return np.nan
            return (c[-2] / c[-2 - w] - 1) * 1e4
        ret5 = ret(5)
        ret20 = ret(20)
        # lo60/hi60: 回测 rolling(60).shift(1) → 不含 last
        lo60 = float(np.min(l[-61:-1])) if len(l) >= 61 else np.nan
        hi60 = float(np.max(h[-61:-1])) if len(h) >= 61 else np.nan
        return {"mid": mid, "sd": sd, "atr_bp": atr_bp, "vol_ma": vol_ma,
                "b5": b5, "b20": b20, "b40": b40,
                "ret5": ret5, "ret20": ret20, "lo60": lo60, "hi60": hi60}

    # ── 偏离段状态机 (每 5M bar 更新) ───────────────────────
    def _update_seg(self, dev: float):
        """dev = (close - mid)/sd (当前5M), 更新段状态"""
        if self.seg_side == 0:
            if dev < -BAND:
                self.seg_side = 1
                self.seg_bars = 1
                self.entered_this_seg = False
            elif dev > BAND:
                self.seg_side = -1
                self.seg_bars = 1
                self.entered_this_seg = False
        elif self.seg_side == 1:
            if dev > -HYST_EXIT:
                self.seg_side = 0
                self.seg_bars = 0
            else:
                self.seg_bars += 1
        elif self.seg_side == -1:
            if dev < HYST_EXIT:
                self.seg_side = 0
                self.seg_bars = 0
            else:
                self.seg_bars += 1

    # ── 主入口 ──────────────────────────────────────────────
    def on_bar(self, bar: dict) -> dict:
        """新 5M bar 收盘 → 更新状态 → 返回决策 dict

        bar: {timestamp, open, high, low, close, volume}
        返回:
          {"action": "none"}
          {"action": "enter_long"/"enter_short", "price": entry_close,
           "side": 1/-1}
          {"action": "exit", "price": exit_price, "reason": ...}
        """
        # 1) 追加 bar + 更新 15M
        self._append_5m(bar)
        ind = self._ind15()
        if ind is None or not np.isfinite(ind["mid"]) or ind["sd"] <= 0:
            return {"action": "none"}

        close = float(bar["close"])
        dev = (close - ind["mid"]) / ind["sd"]

        # 2) 持仓管理: 出场判定优先
        if self.position is not None:
            return self._check_exit(ind, dev, close, bar)

        # 3) 偏离段状态更新 (无持仓时)
        self._update_seg(dev)
        if self.seg_side == 0 or self.entered_this_seg:
            return {"action": "none"}

        # 4) 入场器打分
        side = self.seg_side
        return self._check_entry(ind, dev, close, bar, side)

    # ── 入场判定 ────────────────────────────────────────────
    def _check_entry(self, ind, dev, close, bar, side: int) -> dict:
        # 特征 (11个, 与回测 ef() 一致)
        if side == 1:
            dist = (close - ind["lo60"]) / max(ind.get("atr_bp", 1) * max(ind["mid"], 1e-9) / 1e4, 1e-9) \
                if np.isfinite(ind.get("lo60", np.nan)) else np.nan
        else:
            dist = (ind["hi60"] - close) / max(ind.get("atr_bp", 1) * max(ind["mid"], 1e-9) / 1e4, 1e-9) \
                if np.isfinite(ind.get("hi60", np.nan)) else np.nan
        f = [
            dev, min(self.seg_bars, 10), ind["ret5"], ind["ret20"],
            (bar.get("volume", 0) / ind["vol_ma"]) if ind["vol_ma"] > 0 else 1.0,
            ind["atr_bp"], ind["b5"], ind["b20"], ind["b40"], dist,
        ]
        if not all(np.isfinite(f)):
            return {"action": "none"}
        coef, thr = self.entry_models[side]
        score = _ols_predict(coef, f)
        if score > thr:
            self.entered_this_seg = True
            self.position = {
                "side": side, "entry_price": close, "entry_idx": len(self._c5),
                "entry_time": bar["timestamp"], "entry_score": score,
                "peak_pnl": 0.0, "bars_held": 0,
            }
            return {"action": "enter_long" if side == 1 else "enter_short",
                    "price": close, "side": side, "score": score}
        return {"action": "none"}

    # ── 出场判定 ────────────────────────────────────────────
    def _check_exit(self, ind, dev, close, bar, ) -> dict:
        pos = self.position
        side = pos["side"]
        bars_held = pos["bars_held"] + 1
        pos["bars_held"] = bars_held

        # 自然出场: 触及中轨 (限价, high/low 覆盖 mid)
        mid = ind["mid"]
        h5 = float(bar["high"])
        l5 = float(bar["low"])
        touched = (side == 1 and h5 >= mid) or (side == -1 and l5 <= mid)

        # 浮盈
        if side == 1:
            cur = np.log(close / pos["entry_price"]) * 1e4
        else:
            cur = -np.log(close / pos["entry_price"]) * 1e4
        pos["peak_pnl"] = max(pos["peak_pnl"], cur)
        drawdown = pos["peak_pnl"] - cur if pos["peak_pnl"] > 0 else 0.0

        # 出场器 (第3根起)
        if bars_held >= MIN_TR_BARS:
            dev_signed = dev * side
            f = [
                bars_held / MAXK, cur, drawdown, dev_signed,
                ind["b5"] * side, ind["b20"] * side, ind["ret5"] * side,
                ind["atr_bp"],
                (bar.get("volume", 0) / ind["vol_ma"]) if ind["vol_ma"] > 0 else 1.0,
            ]
            if all(np.isfinite(f)):
                exit_score = _ols_predict(self.exit_model, f)
                if exit_score > EXIT_TH:
                    return self._exit("exiter", exit_score, close)

        if touched:
            return self._exit("mid", 0.0, mid)

        # 超时
        if bars_held >= MAXK:
            return self._exit("timeout", 0.0, close)

        return {"action": "none"}

    def _exit(self, reason, score, price) -> dict:
        pos = self.position
        self.position = None
        self.entered_this_seg = True  # 平仓后本段不再入场 (每段一笔)
        return {"action": "exit", "reason": reason, "score": score,
                "price": price, "side": pos["side"]}
