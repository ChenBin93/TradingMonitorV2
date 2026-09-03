#!/usr/bin/env python3
"""执行引擎 — paper 模式 + ccxt 真实接口

设计:
- PaperBroker: 模拟撮合 (行情驱动: 前一根 bar high/low 覆盖挂单价即成交),
  完整模拟资金/持仓/费用/滑点, 状态可审计
- OkxBroker: ccxt 同步 REST (真实/测试网), 轮询行情 + 限价挂单 + 查询成交
- 统一 Broker 接口: get_candles / place_limit / cancel / get_position / close

成交判定口径 (与回测一致):
- 挂单: 做多挂 mid-2σ, 做空挂 mid+2σ
- 成交: 最新已收盘 bar 的 low<=挂单价 (多) / high>=挂单价 (空) → 成交
- 成交价 = 挂单价 (更优价成交)
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

log = logging.getLogger("trader.executor")


# ── 统一成交记录 ─────────────────────────────────────────────
@dataclass
class Fill:
    symbol: str
    side: str            # "buy" / "sell"
    price: float
    qty: float
    ts: float
    fee: float = 0.0
    meta: dict = field(default_factory=dict)


# ── Paper 撮合 ───────────────────────────────────────────────
class PaperBroker:
    """模拟交易所 — 行情驱动成交, 无网络

    持仓以名义价值记账: qty 为 USDT 数量 (1x 价值), price 为标记价。
    """

    def __init__(self, cfg: dict, state_dir: str):
        self.cfg = cfg
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.equity = float(cfg.get("initial_equity", 100.0))
        self.positions: dict[str, dict] = {}   # symbol -> position
        self.pending: dict[str, dict] = {}     # symbol -> 挂单
        self.fills: list[Fill] = []
        self.trades_log: list[dict] = []
        self._load()

    # ── 持久化 ───────────────────────────────────────────────
    def _state_path(self):
        return os.path.join(self.state_dir, "paper_state.json")

    def _load(self):
        p = self._state_path()
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            self.equity = d.get("equity", self.equity)
            self.positions = d.get("positions", {})
            self.pending = d.get("pending", {})
            self.fills = [Fill(**x) for x in d.get("fills", [])]
            log.info("paper 状态已恢复: equity=%.2f pos=%d pending=%d",
                     self.equity, len(self.positions), len(self.pending))

    def save(self):
        with open(self._state_path(), "w") as f:
            json.dump({
                "equity": self.equity,
                "positions": self.positions,
                "pending": self.pending,
                "fills": [vars(x) for x in self.fills],
            }, f, indent=2, ensure_ascii=False)

    # ── 行情 ─────────────────────────────────────────────────
    def get_candles(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        """由主循环注入行情缓存; paper 模式下从外部 data source 取"""
        raise NotImplementedError("PaperBroker 行情由主循环注入 (on_kline)")

    # ── 下单 ─────────────────────────────────────────────────
    def place_limit(self, symbol: str, side: str, price: float,
                    qty: float, meta: dict | None = None) -> str:
        """挂限价单; 返回 order_id"""
        oid = f"paper_{int(time.time()*1000)}_{len(self.fills)}"
        self.pending[symbol] = {
            "id": oid, "side": side, "price": price, "qty": qty,
            "meta": meta or {}, "ts": time.time(),
        }
        log.info("[paper] 挂单 %s %s %s @ %.2f qty=%.4f", oid, symbol, side, price, qty)
        self.save()
        return oid

    def open_position(self, symbol: str, side: str, price: float, qty: float,
                      meta: dict | None = None) -> dict:
        """信号成交直接开仓 — 成交价 = 挂单价 (信号 bar 已覆盖挂单价)"""
        fee = qty * price * self.cfg.get("fee_rate", 0.0004) / 2
        self.equity -= fee
        pos = {
            "side": "long" if side == "buy" else "short",
            "entry_price": price,
            "qty": qty,
            "entry_time": time.time(),
            "entry_meta": meta or {},
        }
        # 合并 meta (stop_price/bars_held 等出场判定字段直接进 position)
        for k, v in (meta or {}).items():
            pos[k] = v
        self.positions[symbol] = pos
        self.fills.append(Fill(symbol, side, price, qty, time.time(), fee))
        log.info("[paper] 成交开仓 %s %s @ %.2f qty=%.4f fee=%.4f",
                 symbol, side, price, qty, fee)
        self.save()
        return self.positions[symbol]

    def cancel(self, symbol: str) -> bool:
        if symbol in self.pending:
            del self.pending[symbol]
            log.info("[paper] 撤单 %s", symbol)
            self.save()
            return True
        return False

    def on_bar(self, symbol: str, bar: dict):
        """新 bar 收盘后撮合: 检查挂单成交 + 更新持仓"""
        # 挂单成交判定
        pend = self.pending.pop(symbol, None)
        if pend:
            hit = False
            if pend["side"] == "buy" and bar["low"] <= pend["price"]:
                hit = True
            elif pend["side"] == "sell" and bar["high"] >= pend["price"]:
                hit = True
            if hit:
                fee = pend["qty"] * pend["price"] * self.cfg.get("fee_rate", 0.0004) / 2
                self.equity -= fee
                self.positions[symbol] = {
                    "side": "long" if pend["side"] == "buy" else "short",
                    "entry_price": pend["price"],
                    "qty": pend["qty"],
                    "entry_time": time.time(),
                    "entry_meta": pend["meta"],
                }
                self.fills.append(Fill(symbol, pend["side"], pend["price"],
                                       pend["qty"], time.time(), fee))
                log.info("[paper] 成交 %s %s @ %.2f qty=%.4f fee=%.4f",
                         symbol, pend["side"], pend["price"], pend["qty"], fee)
            else:
                self.pending[symbol] = pend  # 未成交保留
        self.save()

    # ── 平仓 ─────────────────────────────────────────────────
    def close_position(self, symbol: str, price: float, reason: str = "") -> Optional[dict]:
        pos = self.positions.get(symbol)
        if not pos:
            return None
        side = 1 if pos["side"] == "long" else -1
        pnl = side * (price - pos["entry_price"]) * pos["qty"]
        fee = pos["qty"] * price * self.cfg.get("fee_rate", 0.0004) / 2
        self.equity += pnl - fee
        rec = {
            "symbol": symbol, "side": pos["side"],
            "entry_price": pos["entry_price"], "exit_price": price,
            "qty": pos["qty"], "pnl": pnl, "fee": fee,
            "reason": reason, "ts": time.time(),
        }
        self.trades_log.append(rec)
        del self.positions[symbol]
        log.info("[paper] 平仓 %s %s pnl=%.4f reason=%s", symbol, pos["side"], pnl, reason)
        self.save()
        return rec

    def get_position(self, symbol: str) -> Optional[dict]:
        return self.positions.get(symbol)

    def get_pending(self, symbol: str) -> Optional[dict]:
        return self.pending.get(symbol)

    # ── 账户 ─────────────────────────────────────────────────
    def get_equity(self) -> float:
        # 简单记账: 持仓浮盈不计入 (保守); 平仓时才更新
        return self.equity


# ── ccxt 真实接口 ────────────────────────────────────────────
class OkxBroker:
    """OKX ccxt 同步 REST — 真实/测试网

    轮询: 每 5m 对齐收盘拉已收盘 bar; 限价单 + 查询 + 撤单。
    """

    def __init__(self, cfg: dict, secrets: dict):
        import ccxt
        self.cfg = cfg
        self.ex = ccxt.okx({
            "apiKey": secrets["okx"]["api_key"],
            "secret": secrets["okx"]["api_secret"],
            "password": secrets["okx"]["passphrase"],
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        if cfg.get("sandbox", False):
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        log.info("OKX 连接就绪: %s (sandbox=%s)", self.ex.name, cfg.get("sandbox", False))

    # ── 行情 ─────────────────────────────────────────────────
    def get_candles(self, symbol: str, tf: str, limit: int) -> pd.DataFrame:
        ohlcv = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    def get_ticker(self, symbol: str) -> dict:
        t = self.ex.fetch_ticker(symbol)
        return {"bid": t["bid"], "ask": t["ask"], "last": t["last"]}

    # ── 下单 ─────────────────────────────────────────────────
    def _contract_size(self, symbol: str) -> float:
        """每张合约的币数量 (BTC-USDT-SWAP = 0.01 BTC; SOL = 1 SOL)"""
        m = self.ex.market(symbol)
        return float(m.get("contractSize") or 1.0)

    def _qty_to_contracts(self, symbol: str, qty: float) -> float:
        """币数量 → 合约张数 (OKX 永续按张下单)"""
        cs = self._contract_size(symbol)
        return qty / cs

    def _contracts_to_qty(self, symbol: str, contracts: float) -> float:
        """合约张数 → 币数量"""
        cs = self._contract_size(symbol)
        return contracts * cs

    def place_limit(self, symbol: str, side: str, price: float,
                    qty: float, meta: dict | None = None) -> str:
        """挂限价单 — qty 为币数量, 内部转合约张数"""
        contracts = self._qty_to_contracts(symbol, qty)
        o = self.ex.create_limit_order(symbol, side, contracts, price)
        log.info("[okx] 挂单 %s %s %s @ %.2f qty=%s币(%s张) -> %s",
                 o["id"], symbol, side, price, qty, contracts, o["status"])
        return str(o["id"])

    def open_position(self, symbol: str, side: str, price: float, qty: float,
                      meta: dict | None = None) -> dict:
        """信号成交开仓 — 限价挂单 (挂单价已由信号 bar 覆盖确认)"""
        contracts = self._qty_to_contracts(symbol, qty)
        o = self.ex.create_limit_order(symbol, side, contracts, price)
        log.info("[okx] 开仓挂单 %s %s %s @ %.2f qty=%s币(%s张) -> %s",
                 o["id"], symbol, side, price, qty, contracts, o["status"])
        # 立即查询成交 (限价单可能未立即成交, 由主循环撮合/状态同步)
        pos = {
            "side": "long" if side == "buy" else "short",
            "entry_price": price,
            "qty": qty,
            "entry_time": time.time(),
            "entry_meta": meta or {},
            "okx_order_id": str(o["id"]),
        }
        for k, v in (meta or {}).items():
            pos[k] = v
        return pos

    def open_market(self, symbol: str, side: str, qty: float,
                    meta: dict | None = None) -> dict:
        """市价开仓 (BBML: 按信号时刻市价成交, 回测口径为信号 close)"""
        contracts = self._qty_to_contracts(symbol, qty)
        o = self.ex.create_market_order(symbol, side, contracts)
        px = float(o.get("average") or 0.0)
        log.info("[okx] 市价开仓 %s %s qty=%s币(%s张) avg=%.2f -> %s",
                 symbol, side, qty, contracts, px, o["id"])
        # 若返回无均价, 从持仓查询真实入场价
        if px <= 0:
            time.sleep(1.5)
            rp = self.get_position(symbol)
            px = float(rp.get("entry_price") or 0.0) if rp else 0.0
        pos = {"side": "long" if side == "buy" else "short",
               "entry_price": px, "qty": qty,
               "entry_time": time.time(),
               "entry_meta": meta or {}, "okx_order_id": str(o["id"])}
        for k, v in (meta or {}).items():
            pos[k] = v
        return pos

    def close_market(self, symbol: str, qty: float,
                     meta: dict | None = None) -> Optional[dict]:
        """市价平仓 (按当前持仓方向反向)"""
        pos = self.get_position(symbol)
        if not pos:
            return None
        side = "sell" if pos["side"] == "long" else "buy"
        contracts = self._qty_to_contracts(symbol, qty)
        o = self.ex.create_market_order(symbol, side, contracts)
        px = float(o.get("average") or 0.0)
        # 若返回无均价, 查订单详情
        if px <= 0:
            try:
                time.sleep(1.5)
                od = self.ex.fetch_order(str(o["id"]), symbol)
                px = float(od.get("average") or od.get("price") or 0.0)
            except Exception:
                pass
        rec = {"symbol": symbol, "side": pos["side"], "exit_price": px,
               "qty": qty, "reason": (meta or {}).get("reason", ""),
               "order": str(o["id"]), "ts": time.time()}
        log.info("[okx] 市价平仓 %s %s qty=%s avg=%.2f", symbol, side, qty, px)
        return rec

    def cancel(self, symbol: str, order_id: str) -> bool:
        try:
            self.ex.cancel_order(order_id, symbol)
            log.info("[okx] 撤单 %s %s", symbol, order_id)
            return True
        except Exception as e:
            log.warning("[okx] 撤单失败 %s %s: %s", symbol, order_id, e)
            return False

    # ── 持仓/账户 ────────────────────────────────────────────
    def get_position(self, symbol: str) -> Optional[dict]:
        """查询持仓 — 返回币数量 qty + 出场判定所需字段

        注意: 出场判定需要 stop_price/bars_held, 这些来自策略状态
        (state.py), 由 main.py 在成交时持久化; 此处返回交易所实际持仓。
        """
        try:
            pos = self.ex.fetch_position(symbol)
        except Exception as e:
            log.warning("[okx] 持仓查询失败 %s: %s", symbol, e)
            return None
        if pos and pos.get("contracts"):
            contracts = float(pos["contracts"])
            return {
                "side": "long" if pos["side"] == "long" else "short",
                "entry_price": float(pos.get("entryPrice") or pos.get("average") or 0),
                "qty": self._contracts_to_qty(symbol, contracts),
                "contracts": contracts,
                "notional": float(pos.get("notional") or 0),
                "entry_time": time.time(),
            }
        return None

    def get_pending(self, symbol: str) -> Optional[dict]:
        """查询未成交挂单 (第一个 open 状态的限价单)"""
        try:
            orders = self.ex.fetch_open_orders(symbol)
        except Exception as e:
            log.warning("[okx] 挂单查询失败 %s: %s", symbol, e)
            return None
        if orders:
            o = orders[0]
            return {
                "id": str(o["id"]), "side": o["side"], "price": o["price"],
                "qty": o["amount"], "ts": o["timestamp"],
            }
        return None

    def get_equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            return float(bal.get("USDT", {}).get("total", 0.0))
        except Exception as e:
            log.warning("[okx] 余额查询失败: %s", e)
            return 0.0

    # ── 平仓 ─────────────────────────────────────────────────
    def close_position(self, symbol: str, price: float, reason: str = "") -> Optional[dict]:
        pos = self.get_position(symbol)
        if not pos:
            return None
        side = "sell" if pos["side"] == "long" else "buy"
        # 市价平仓 (contracts 张数)
        try:
            o = self.ex.create_market_order(symbol, side, pos["contracts"])
        except Exception as e:
            log.warning("[okx] 平仓失败 %s %s: %s", symbol, side, e)
            return None
        return {"symbol": symbol, "exit_price": price, "reason": reason,
                "order": o["id"], "qty": pos["qty"], "ts": time.time()}
