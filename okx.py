# OKX 数据客户端 — REST + WebSocket + 内存缓存

import asyncio
import json
import threading
from datetime import datetime
from dataclasses import dataclass
from typing import Callable

import ccxt
import pandas as pd
from loguru import logger


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class KlineCache:

    def __init__(self, max_candles: int = 500):
        self._data: dict[str, dict[str, list[Candle]]] = {}
        self._max = max_candles
        self._lock = threading.RLock()

    def update(self, symbol: str, timeframe: str, candle: Candle):
        with self._lock:
            tf_map = self._data.setdefault(symbol, {})
            candles = tf_map.setdefault(timeframe, [])
            if candles and candles[-1].timestamp == candle.timestamp:
                candles[-1] = candle
            else:
                candles.append(candle)
                if len(candles) > self._max:
                    candles.pop(0)

    def get_closed(self, symbol: str, timeframe: str) -> list[Candle]:
        with self._lock:
            return list(self._data.get(symbol, {}).get(timeframe, []))

    def get_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        candles = self.get_closed(symbol, timeframe)
        if not candles:
            return pd.DataFrame()
        return pd.DataFrame([{
            "timestamp": c.timestamp, "open": c.open, "high": c.high,
            "low": c.low, "close": c.close, "volume": c.volume,
        } for c in candles]).sort_values("timestamp").reset_index(drop=True)


class OKXClient:

    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = "", testnet: bool = False):
        self._exchange = ccxt.okx({
            "apiKey": api_key,
            "secret": api_secret,
            "password": passphrase,
            "enableRateLimit": True,
        })
        if testnet:
            self._exchange.set_sandbox_mode(True)
        self._ws = None
        self._running = False

    # --- REST ---
    def get_top_symbols(self, n: int, quote: str = "USDT") -> list[str]:
        markets = self._exchange.load_markets()
        swaps = [
            s for s, m in markets.items()
            if m.get("swap") and m.get("quote") == quote and m.get("active")
        ]
        tickers = self._exchange.fetch_tickers(swaps)
        ranked = sorted(swaps, key=lambda s: tickers.get(s, {}).get("quoteVolume", 0) or 0, reverse=True)
        return ranked[:n]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list[dict]:
        ohlcv = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return [{
            "timestamp": datetime.fromtimestamp(c[0] / 1000),
            "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5],
        } for c in ohlcv]

    # --- WebSocket ---
    async def ws_connect(self, symbols: list[str], timeframes: list[str],
                         on_kline: Callable, on_trade: Callable | None = None):
        import websockets
        self._running = True
        self._ws = await websockets.connect(
            "wss://ws.okx.com:8443/ws/v5/business", ping_interval=30
        )
        logger.info("WebSocket connected")

        # 订阅
        for sym in symbols:
            for tf in timeframes:
                inst_id = sym.replace("/", "-").replace(":USDT", "-SWAP")
                tf_okx = f"candle{tf.upper()}" if "H" in tf.upper() else f"candle{tf.lower()}"
                await self._ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [{"instId": inst_id, "channel": tf_okx}],
                }))

        # 读取循环
        asyncio.create_task(self._read_loop(on_kline, on_trade))

    async def _read_loop(self, on_kline: Callable, on_trade: Callable | None):
        while self._running and self._ws:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=60)
                data = json.loads(msg)
                arg = data.get("arg", {})
                channel = arg.get("channel", "")
                if channel.startswith("candle"):
                    sym = self._parse_inst_id(arg.get("instId", ""))
                    tf = channel.replace("candle", "").lower()
                    for d in data.get("data", []):
                        candle = Candle(
                            timestamp=datetime.fromtimestamp(int(d[0]) / 1000),
                            open=float(d[1]), high=float(d[2]),
                            low=float(d[3]), close=float(d[4]), volume=float(d[5]),
                        )
                        on_kline(sym, tf, candle)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._running:
                    logger.warning(f"WS read error: {e}")
                break

    def _parse_inst_id(self, inst_id: str) -> str:
        if inst_id.endswith("-USDT-SWAP"):
            base = inst_id.replace("-USDT-SWAP", "")
            return f"{base}/USDT:{base}"
        if inst_id.endswith("-USD-SWAP"):
            base = inst_id.replace("-USD-SWAP", "")
            return f"{base}/USD:{base}"
        return inst_id

    async def ws_close(self):
        self._running = False
        if self._ws:
            await self._ws.close()
