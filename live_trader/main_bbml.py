#!/usr/bin/env python3
"""BBML Reversion 策略 — paper 模拟盘 (run_signals 信号跟随)

方案: 每根新 5M bar, 用累积历史 (窗口 ~1500 根 ≈ 5天) 调
strategy_bb_ml.run_signals() 重算期望持仓状态, 对比上一根的状态:
  状态 0→±1: 开仓 (entry = 状态变化那根 5M 的 close, 与回测口径一致)
  状态 ±1→0: 平仓 (exit = 当前 bar close; 出场器/触及中轨由 run_signals 内部判定)
口径 100% 与回测一致 (复用回测引擎)。

用法:
  python3 live_trader/main_bbml.py --feed replay --start 2026-05-01   # 回放
  python3 live_trader/main_bbml.py --feed okx                         # OKX paper
"""
import argparse
import json
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import yaml

from bb_ml_reversion.strategy_bb_ml import BBMLReversion
from live_trader.executor import PaperBroker
from live_trader.state import TraderState
from live_trader.monitor import Monitor
from live_trader.data_feed import ReplayFeed, OkxFeed

log = logging.getLogger("trader.bbml")

WIN_BARS = 1500      # 重算窗口 (5M ≈ 5 天, 含 15M 预热)
WARMUP = 400         # 前 400 根只累积不交易


class BBMLTrader:
    def __init__(self, cfg: dict, feed: str = "", start: str = ""):
        self.cfg = cfg
        symbol = cfg["symbol"]
        self.symbol = symbol["name"]
        self.tf = symbol.get("timeframe", "5m")
        self.state_dir = cfg.get("state_dir", "live_trader/state_bbml")

        with open(os.path.join(ROOT, "bb_ml_reversion", "models_btc.json")) as f:
            m = json.load(f)
        self.entry_models = {
            int(k): (np.array(v["coef"]), float(v["threshold"]))
            for k, v in m["entry_models"].items()}
        self.exit_model = np.array(m["exit_model"]["coef"])
        self.strat = BBMLReversion()

        self.broker = PaperBroker(cfg.get("paper", {}), self.state_dir)
        self.state = TraderState(self.state_dir)
        self.monitor = Monitor(cfg.get("health", {}).get("port", 8095), self.status)

        self.history = pd.DataFrame()
        self.bars_seen = 0
        self.cur_state = 0       # 期望持仓: 1/-1/0
        self.pnl_bps = []        # 已平仓收益 bp

        if feed == "replay":
            self.feed = ReplayFeed(self.symbol, self.tf, start=start,
                                   warmup_bars=WIN_BARS + WARMUP)
            self.feed.speed = 999999
        elif feed == "okx":
            self.feed = OkxFeed(self.symbol, self.tf)
        else:
            self.feed = None

    def fetch_candles(self):
        if self.feed is not None:
            if isinstance(self.feed, OkxFeed):
                return self.feed.refresh()
            bar = self.feed.next_bar()
            if bar is None:
                raise StopIteration("回放结束")
            return bar
        raise NotImplementedError("需要 --feed (replay/okx)")

    # ── 每根 5M 收盘 ────────────────────────────────────────
    def on_bar(self, bar: dict):
        ts = bar.get("timestamp", bar.get("ts"))
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        row = pd.DataFrame([{
            "open": float(bar["open"]), "high": float(bar["high"]),
            "low": float(bar["low"]), "close": float(bar["close"]),
            "volume": float(bar.get("volume", 0.0))}], index=[ts])
        self.history = pd.concat([self.history, row]).tail(WIN_BARS + WARMUP)
        self.bars_seen += 1
        if self.bars_seen < WARMUP or len(self.history) < WIN_BARS:
            return

        # 重算信号 (窗口 = 尾部 WIN_BARS 根)
        window = self.history.tail(WIN_BARS)
        sig = self.strat.run_signals(window, entry_models=self.entry_models,
                                     exit_model=self.exit_model)
        target = int(sig.iloc[-1]) if len(sig) else 0
        self._apply_state(target, window, ts)

    def _apply_state(self, target: int, window, now_ts):
        """状态变化 → 交易动作"""
        cur = self.cur_state
        pos = self.broker.get_position(self.symbol)
        if target == cur and pos is not None:
            return
        # 目标与当前不一致
        if target != 0 and pos is None:
            # 开仓
            side = "buy" if target == 1 else "sell"
            px = float(window["close"].iloc[-1])  # 状态变化那根 close
            qty = self._position_size(px)
            self.broker.open_position(self.symbol, side, px, qty,
                                      {"side": "long" if target == 1 else "short",
                                       "state": target})
            log.info("开仓 %s %s @ %.2f", self.symbol,
                     "long" if target == 1 else "short", px)
            self.cur_state = target
        elif target == 0 and pos is not None:
            # 平仓
            px = float(window["close"].iloc[-1])
            rec = self.broker.close_position(self.symbol, px, "signal_exit")
            if rec and self.state.position:
                entry_px = self.state.position.get("entry_price") or px
                side = 1 if rec.get("side") == "long" else -1
                bp = side * (np.log(px / entry_px) * 1e4) - 4
                self.pnl_bps.append(bp)
                log.info("平仓 %s @ %.2f bp=%.1f", self.symbol, px, bp)
            self.cur_state = 0
        elif target != 0 and pos is not None and target != cur:
            # 反向 → 先平后开 (简化: 平仓)
            px = float(window["close"].iloc[-1])
            rec = self.broker.close_position(self.symbol, px, "signal_reverse")
            self.cur_state = 0
            log.info("反向平仓 %s @ %.2f", self.symbol, px)

    def _position_size(self, price: float) -> float:
        eq = self.broker.get_equity()
        notional = eq * 0.1  # 10% 名义
        return max(notional / price, 1e-8)

    def status(self) -> dict:
        return {
            "symbol": self.symbol,
            "equity": round(self.broker.get_equity(), 4),
            "position": self.broker.get_position(self.symbol),
            "state": self.cur_state,
            "bars_seen": self.bars_seen,
            "closed": len(self.pnl_bps),
            "avg_bp": round(sum(self.pnl_bps) / len(self.pnl_bps), 2) if self.pnl_bps else 0,
        }

    def run(self, duration_sec: float = 0):
        self.monitor.start()
        log.info("BBML paper 启动 (signal跟随): %s", self.symbol)
        start = time.time()
        replay = isinstance(getattr(self, "feed", None), ReplayFeed)
        while True:
            if duration_sec and time.time() - start > duration_sec:
                break
            try:
                bar = self.fetch_candles()
                self.on_bar(bar)
            except StopIteration:
                log.info("回放结束")
                break
            except Exception as e:
                log.exception("循环异常: %s", e)
                break
            if replay:
                continue
            time.sleep(2)
        if self.pnl_bps:
            log.info("=== 完成: %d 笔, 平均 %.2fbp/笔, 胜率 %.1f%% ===",
                     len(self.pnl_bps), sum(self.pnl_bps) / len(self.pnl_bps),
                     100 * sum(1 for x in self.pnl_bps if x > 0) / len(self.pnl_bps))
        self.monitor.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="live_trader/config_bbml.yaml")
    ap.add_argument("--feed", default="replay")
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--duration", type=float, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    t = BBMLTrader(cfg, feed=args.feed, start=args.start)
    t.run(args.duration)


if __name__ == "__main__":
    main()
