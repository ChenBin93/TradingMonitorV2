# 持仓监控 + 止损策略

import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
from loguru import logger


@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    pnl_pct: float
    pnl_abs: float
    inst_id: str
    open_time: datetime | None = None
    peak_price: float = 0.0
    exit_stage: int = 0


# =============================================================================
# 主类
# =============================================================================

class PositionMonitor:
    def __init__(self, config: dict, notifier, secrets: dict):
        self.config = config
        self.notifier = notifier  # 飞书客户端（由 main.py 注入）
        self.secrets = secrets

        pos_cfg = config.get("position", {})
        self._interval = pos_cfg.get("interval_seconds", 60)
        self._pnl_loss = pos_cfg.get("pnl_loss_pct", 5.0)
        self._pnl_profit = pos_cfg.get("pnl_profit_pct", 10.0)
        self._max_hours = pos_cfg.get("max_hours", 4)
        self._auto_exit = pos_cfg.get("auto_exit", False)
        self._initial_sl = pos_cfg.get("initial_sl_pct", 3.0)

        okx_cfg = secrets.get("okx", {})
        self._api_key = okx_cfg.get("api_key", "")
        self._api_secret = okx_cfg.get("api_secret", "")
        self._passphrase = okx_cfg.get("passphrase", "")
        self._testnet = okx_cfg.get("testnet", False)
        self._active = bool(self._api_key)

        self._lock = threading.Lock()
        self._last_positions: dict[str, Position] = {}
        self._last_alert_time: dict[str, datetime] = {}
        self._silence_seconds = 600
        self._running = False

    def run(self):
        """主循环（在独立线程中运行）"""
        if not self._active:
            logger.info("Position monitor: no API key, disabled")
            return

        self._running = True
        logger.info(f"Position monitor started (interval={self._interval}s)")
        while self._running:
            try:
                self._check_positions()
            except Exception as e:
                logger.error(f"Position check error: {e}")
            time.sleep(self._interval)

    def stop(self):
        self._running = False

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        msg = f"{timestamp}{method}{path}{body}"
        mac = hmac.new(
            self._api_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        )
        import base64
        return base64.b64encode(mac.digest()).decode()

    def _fetch_positions(self) -> list[dict]:
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        path = "/api/v5/account/positions"
        sig = self._sign(ts, "GET", path)
        base_url = "https://www.okx.com"

        try:
            resp = requests.get(
                f"{base_url}{path}",
                headers={
                    "OK-ACCESS-KEY": self._api_key,
                    "OK-ACCESS-SIGN": sig,
                    "OK-ACCESS-TIMESTAMP": ts,
                    "OK-ACCESS-PASSPHRASE": self._passphrase,
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            data = resp.json()
            if data.get("code") != "0":
                logger.warning(f"OKX positions API error: {data.get('msg')}")
                return []
            return data.get("data", [])
        except Exception as e:
            logger.warning(f"Fetch positions error: {e}")
            return []

    def _is_silent(self, key: str) -> bool:
        last = self._last_alert_time.get(key)
        if not last:
            return False
        return (datetime.now() - last).total_seconds() < self._silence_seconds

    def _alert(self, key: str, msg: str):
        if self._is_silent(key):
            return
        self._last_alert_time[key] = datetime.now()
        self.notifier.send(msg)

    def _check_positions(self):
        raw = self._fetch_positions()
        current_positions: dict[str, Position] = {}

        for p in raw:
            size = float(p.get("pos", 0))
            if size == 0:
                continue

            symbol = p.get("instId", "").replace("-USDT-SWAP", "/USDT")
            side = "long" if p.get("posSide") == "long" else "short"
            entry = float(p.get("avgPx", 0))
            mark = float(p.get("markPx", 0))
            upl = float(p.get("upl", 0))
            pnl_pct = upl / (entry * size) * 100 if entry > 0 and size > 0 else 0

            pos = Position(
                symbol=symbol, side=side, size=size,
                entry_price=entry, current_price=mark,
                pnl_pct=pnl_pct, pnl_abs=upl,
                inst_id=p.get("instId", ""),
                open_time=datetime.fromtimestamp(int(p.get("cTime", 0)) / 1000) if p.get("cTime") else None,
            )
            current_positions[symbol] = pos

        with self._lock:
            prev_symbols = set(self._last_positions.keys())

            # 新增持仓
            for sym, pos in current_positions.items():
                prev = self._last_positions.get(sym)

                # 更新 peak price
                if prev:
                    if pos.side == "long":
                        pos.peak_price = max(prev.peak_price, pos.current_price)
                    else:
                        pos.peak_price = max(prev.peak_price, pos.current_price) if prev.peak_price == 0 else min(prev.peak_price, pos.current_price)
                else:
                    pos.peak_price = pos.current_price

                if not prev or sym not in prev_symbols:
                    self._alert(f"pos_new_{sym}", f"[持仓新增] {sym} {pos.side} 仓位:{pos.size}手 入场:{pos.entry_price:.4f}")

                # 浮盈/浮亏告警
                if pos.pnl_pct >= self._pnl_profit:
                    self._alert(f"pnl_profit_{sym}", f"[浮盈] {sym} {pos.side} +{pos.pnl_pct:.1f}% (${pos.pnl_abs:.0f})")
                elif pos.pnl_pct <= -self._pnl_loss:
                    self._alert(f"pnl_loss_{sym}", f"[浮亏] {sym} {pos.side} {pos.pnl_pct:.1f}% (${pos.pnl_abs:.0f})")

                # 持仓时间过长
                if pos.open_time:
                    hours = (datetime.now() - pos.open_time).total_seconds() / 3600
                    if hours > self._max_hours:
                        self._alert(f"pos_age_{sym}", f"[持仓过久] {sym} {pos.side} 已持仓{hours:.0f}小时")

                # 自动止损
                if self._auto_exit and prev:
                    pos.exit_stage = prev.exit_stage
                    self._check_auto_exit(pos, prev)

            # 已平仓
            for sym in prev_symbols:
                if sym not in current_positions:
                    prev = self._last_positions[sym]
                    self._alert(f"pos_closed_{sym}", f"[持仓平仓] {sym} {prev.side} 最终盈亏:{prev.pnl_pct:.1f}%")

            self._last_positions = current_positions

    def _check_auto_exit(self, pos: Position, prev: Position):
        """4 阶段止损出场"""
        # Stage 1: 初始化止损
        if pos.exit_stage == 0:
            pos.exit_stage = 1
            return

        entry = pos.entry_price
        peak = pos.peak_price
        pnl = pos.pnl_pct

        if pos.side == "long":
            # Stage 1 → 2: 盈利达到初始止损的 2 倍，移动止损到保本
            if pos.exit_stage == 1 and pnl >= self._initial_sl * 2:
                pos.exit_stage = 2
                self._alert(f"exit_stage_{pos.symbol}", f"[止损升级] {pos.symbol} 推保本 (盈利{pnl:.1f}%)")
            # Stage 2 → 3: 盈利扩大，追踪止盈前高回撤 30%
            elif pos.exit_stage == 2:
                drawdown_pct = (peak - pos.current_price) / peak * 100
                if drawdown_pct > self._initial_sl * 0.3:
                    self._alert(f"exit_signal_{pos.symbol}", f"[止损触发] {pos.symbol} long 回撤{drawdown_pct:.1f}% @ {pos.current_price:.4f}")
        else:
            # 做空同理
            if pos.exit_stage == 1 and pnl >= self._initial_sl * 2:
                pos.exit_stage = 2
                self._alert(f"exit_stage_{pos.symbol}", f"[止损升级] {pos.symbol} 推保本 (盈利{pnl:.1f}%)")
            elif pos.exit_stage == 2:
                if pos.current_price > peak:
                    drawdown_pct = (pos.current_price - peak) / peak * 100
                    if drawdown_pct > self._initial_sl * 0.3:
                        self._alert(f"exit_signal_{pos.symbol}", f"[止损触发] {pos.symbol} short 回撤{drawdown_pct:.1f}% @ {pos.current_price:.4f}")
