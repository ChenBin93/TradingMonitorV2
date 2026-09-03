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
from live_trader.executor import PaperBroker, OkxBroker
from live_trader.state import TraderState
from live_trader.monitor import Monitor
from live_trader.data_feed import ReplayFeed, OkxFeed

log = logging.getLogger("trader.bbml")

WIN_BARS = 1200      # 重算窗口 (5M ≈ 4.2 天; demo 行情仅 ~5天历史)
WARMUP = 300         # 前 300 根只累积不交易


class BBMLTrader:
    def __init__(self, cfg: dict, mode: str = "paper", feed: str = "",
                 start: str = "", secrets: dict | None = None):
        self.cfg = cfg
        self.mode = mode
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

        # Broker: paper (本地模拟) / okx (demo 真实下单)
        if mode == "okx":
            if not secrets or not secrets.get("okx"):
                raise SystemExit("okx 模式需要密钥 (live_trader/secrets_bbml.yaml)")
            self.broker = OkxBroker({"sandbox": secrets["okx"].get("sandbox", True)},
                                    secrets)
            log.info("Broker: OKX demo (sandbox)")
        else:
            self.broker = PaperBroker(cfg.get("paper", {}), self.state_dir)
            log.info("Broker: PAPER (本地模拟)")
        self.state = TraderState(self.state_dir)
        self.monitor = Monitor(cfg.get("health", {}).get("port", 8095), self.status)

        self.history = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.hist_1h = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.hist_4h = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.bars_seen = 0
        self.cur_state = 0       # 期望持仓: 1/-1/0
        self.pnl_bps = []        # 已平仓收益 bp

        # 行情源: mode=okx 用 broker; 否则 replay(回放) 或 None
        if mode == "okx":
            # 用 OkxBroker 拉历史热身 (demo 下单同市场行情) + 多周期
            df = self.broker.get_candles(self.symbol, self.tf, 300)
            # 分页补足窗口
            pages = [df]
            oldest = df.index[0]
            while sum(len(p) for p in pages) < WIN_BARS + WARMUP:
                before = self._older_candles(oldest)
                if before is None or before.empty:
                    break
                pages.append(before)
                oldest = before.index[0]
            full = pd.concat(pages).sort_index().tail(WIN_BARS + WARMUP)
            for ts, r in full.iterrows():
                self.history.loc[ts] = [float(r.open), float(r.high),
                                        float(r.low), float(r.close),
                                        float(r.volume)]
                self.bars_seen += 1
            # 1h/4h 历史 (v1.1 趋势特征需要)
            for tf_name, hist in (("1h", self.hist_1h), ("4h", self.hist_4h)):
                try:
                    hdf = self.broker.get_candles(self.symbol, tf_name, 300)
                    for ts, r in hdf.iterrows():
                        hist.loc[ts] = [float(r.open), float(r.high),
                                        float(r.low), float(r.close),
                                        float(r.volume)]
                except Exception as e:
                    log.warning("拉 %s 历史失败: %s", tf_name, e)
            self.feed = None
            self.is_replay = False
            log.info("OKX demo warmup: 5m=%d 1h=%d 4h=%d",
                     len(self.history), len(self.hist_1h), len(self.hist_4h))
        elif feed == "replay":
            # paper 回放模式
            self.feed = ReplayFeed(self.symbol, self.tf, start=start,
                                   warmup_bars=WIN_BARS + WARMUP)
            self.feed.speed = 999999
            self.is_replay = True
        else:
            self.feed = None
            self.is_replay = False

    def _older_candles(self, before_ts):
        """拉 before_ts 之前的 K 线 (分页补历史)"""
        try:
            ohlcv = self.broker.ex.fetch_ohlcv(
                self.symbol, timeframe=self.tf, limit=300,
                params={"after": str(int(before_ts.value // 1e6))})
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low",
                                              "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df.set_index("ts")
        except Exception as e:
            log.warning("分页拉历史失败: %s", e)
            return None

    def _refresh_multitf(self, now_ts):
        """增量刷新 1h/4h 历史 (v1.1 趋势特征) — 从 broker 拉最新追加"""
        for tf_name, hist in (("1h", self.hist_1h), ("4h", self.hist_4h)):
            try:
                df = self.broker.get_candles(self.symbol, tf_name, 300)
                last_ts = hist.index[-1] if len(hist) else None
                new = df[df.index > last_ts] if last_ts is not None else df
                for ts, r in new.iterrows():
                    hist.loc[ts] = [float(r.open), float(r.high),
                                    float(r.low), float(r.close),
                                    float(r.volume)]
                if len(hist) > 500:
                    # 保留足够 (4h 需覆盖 5m 窗口 ~5天=30根4h; 留 300 冗余)
                    hist.drop(hist.index[:-300], inplace=True)
            except Exception as e:
                log.debug("刷新 %s 失败: %s", tf_name, e)

    def fetch_candles(self):
        if self.feed is not None:
            bar = self.feed.next_bar()
            if bar is None:
                raise StopIteration("回放结束")
            return bar
        raise NotImplementedError("需要 --feed (replay/okx) 或 okx 模式")

    # ── 每根 5M 收盘 ────────────────────────────────────────
    def on_bar(self, bar: dict):
        ts = bar.get("timestamp", bar.get("ts"))
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        self.history.loc[ts] = [float(bar["open"]), float(bar["high"]),
                                float(bar["low"]), float(bar["close"]),
                                float(bar.get("volume", 0.0))]
        self.history = self.history.sort_index().tail(WIN_BARS + WARMUP)
        self.bars_seen += 1
        if self.mode == "okx":
            self._refresh_multitf(ts)  # 增量刷新 1h/4h
        if self.bars_seen < WARMUP or len(self.history) < WIN_BARS:
            return
        window = self.history.tail(WIN_BARS)
        sig = self.strat.run_signals(
            window, df_1h=self.hist_1h, df_4h=self.hist_4h,
            entry_models=self.entry_models, exit_model=self.exit_model)
        target = int(sig.iloc[-1]) if len(sig) else 0
        self._apply_state(target, window, ts)

    def _apply_state(self, target: int, window, now_ts):
        """状态变化 → 交易动作 (paper: 本地模拟; okx: demo 市价单)"""
        okx = self.mode == "okx"
        cur = self.cur_state
        pos = self.broker.get_position(self.symbol)
        if target == cur and pos is not None:
            return
        # 目标与当前不一致
        if target != 0 and pos is None:
            # 开仓 (市价)
            side = "buy" if target == 1 else "sell"
            qty = self._position_size(float(window["close"].iloc[-1]))
            if okx:
                pos = self.broker.open_market(
                    self.symbol, side, qty,
                    {"side": "long" if target == 1 else "short", "state": target})
                px = pos.get("entry_price", float(window["close"].iloc[-1]))
            else:
                px = float(window["close"].iloc[-1])
                pos = self.broker.open_position(self.symbol, side, px, qty,
                                                {"side": "long" if target == 1 else "short",
                                                 "state": target})
            self.state.set_position(pos)
            log.info("开仓 %s %s @ %.4f qty=%.4f", self.symbol,
                     "long" if target == 1 else "short", px, qty)
            self.cur_state = target
        elif target == 0 and pos is not None:
            # 平仓 (市价)
            qty = pos.get("qty", 0)
            if okx and qty > 0:
                rec = self.broker.close_market(self.symbol, qty, {"reason": "signal_exit"})
                px = rec.get("exit_price", float(window["close"].iloc[-1])) if rec else None
            else:
                px = float(window["close"].iloc[-1])
                rec = self.broker.close_position(self.symbol, px, "signal_exit")
            spos = self.state.position or {}
            entry_px = spos.get("entry_price") or 0
            pside = spos.get("side") or (rec or {}).get("side")
            if rec and entry_px and px:
                side = 1 if pside == "long" else -1
                bp = side * (np.log(px / entry_px) * 1e4) - 4
                self.pnl_bps.append(bp)
                log.info("平仓 %s @ %.4f bp=%.1f", self.symbol, px, bp)
            self.state.clear_position()
            self.cur_state = 0
        elif target != 0 and pos is not None and target != cur:
            # 反向 → 平仓
            qty = pos.get("qty", 0)
            if okx and qty > 0:
                rec = self.broker.close_market(self.symbol, qty, {"reason": "signal_reverse"})
                px = rec.get("exit_price", float(window["close"].iloc[-1])) if rec else None
            else:
                px = float(window["close"].iloc[-1])
                rec = self.broker.close_position(self.symbol, px, "signal_reverse")
            self.state.clear_position()
            self.cur_state = 0
            log.info("反向平仓 %s @ %.4f", self.symbol,
                     px if px else float(window["close"].iloc[-1]))

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
        log.info("BBML 启动 (signal跟随): %s mode=%s", self.symbol, self.mode)
        start = time.time()
        replay = self.is_replay
        last_ts = self.history.index[-1] if len(self.history) else None
        while True:
            if duration_sec and time.time() - start > duration_sec:
                break
            try:
                if replay:
                    bar = self.fetch_candles()
                    self.on_bar(bar)
                elif self.mode == "okx":
                    # 轮询: 拉最新行情, diff 新 bar
                    df = self.broker.get_candles(self.symbol, self.tf, 300)
                    new = df[df.index > last_ts] if last_ts is not None else df
                    for ts, r in new.iterrows():
                        self.on_bar({"ts": ts, "open": r.open, "high": r.high,
                                     "low": r.low, "close": r.close,
                                     "volume": r.volume})
                        last_ts = ts
                    if len(new):
                        log.debug("新 bar %d 根", len(new))
                    time.sleep(2)
                else:
                    raise NotImplementedError("未知模式")
            except StopIteration:
                log.info("回放结束")
                break
            except Exception as e:
                log.exception("循环异常: %s", e)
                break
            if replay:
                continue
        if self.pnl_bps:
            log.info("=== 完成: %d 笔, 平均 %.2fbp/笔, 胜率 %.1f%% ===",
                     len(self.pnl_bps), sum(self.pnl_bps) / len(self.pnl_bps),
                     100 * sum(1 for x in self.pnl_bps if x > 0) / len(self.pnl_bps))
        self.monitor.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="paper", choices=["paper", "okx"],
                    help="paper=本地模拟; okx=OKX demo 真实下单")
    ap.add_argument("--config", default="live_trader/config_bbml.yaml")
    ap.add_argument("--secrets", default="live_trader/secrets_bbml.yaml")
    ap.add_argument("--feed", default="replay", help="仅 paper 模式用")
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--duration", type=float, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    secrets = None
    if args.mode == "okx":
        secrets = yaml.safe_load(open(args.secrets)) if os.path.exists(args.secrets) else {}
    t = BBMLTrader(cfg, mode=args.mode, feed=args.feed, start=args.start,
                   secrets=secrets)
    t.run(args.duration)


if __name__ == "__main__":
    main()
