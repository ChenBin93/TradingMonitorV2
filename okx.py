# OKX 数据客户端 — REST + WebSocket + 内存缓存

import asyncio
import json
import threading
import time
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
        self._last_update: dict[str, dict[str, float]] = {}  # {symbol: {tf: epoch_seconds}}
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
            sym_ts = self._last_update.setdefault(symbol, {})
            sym_ts[timeframe] = time.time()

    def get_data_age_minutes(self, symbol: str, timeframe: str) -> float | None:
        """返回该 symbol/tf 缓存数据上次更新的时间 (分钟前), None=从未更新"""
        with self._lock:
            sym_ts = self._last_update.get(symbol, {})
            ts = sym_ts.get(timeframe)
            if ts is None:
                return None
            return (time.time() - ts) / 60

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
        # OKX ccxt 不提供 quoteVolume — 用 baseVolume × 现价估算
        def _est_vol(s):
            t = tickers.get(s, {})
            bv = t.get("baseVolume", 0) or 0
            p = t.get("last", 0) or 0
            return bv * p
        ranked = sorted(swaps, key=_est_vol, reverse=True)
        return ranked[:n]

    def get_24h_volume(self, symbols: list[str]) -> dict[str, float]:
        """返回币种的 24h 估计成交量 (USDT), baseVolume × 现价"""
        result = {}
        for s in symbols:
            try:
                t = self._exchange.fetch_ticker(s)
                base_vol = t.get("baseVolume", 0) or 0
                price = t.get("last", 0) or 0
                result[s] = base_vol * price
            except Exception:
                result[s] = 0
        return result

    def fetch_funding_rate(self, symbol: str) -> dict | None:
        import requests
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")
        try:
            resp = requests.get(
                "https://www.okx.com/api/v5/public/funding-rate",
                params={"instId": inst_id}, timeout=10)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                d = data["data"][0]
                return {
                    "funding_rate": float(d.get("fundingRate", 0)),
                    "next_funding_time": d.get("nextFundingTime", ""),
                }
        except Exception:
            pass
        return None

    def fetch_oi(self, symbol: str) -> float | None:
        import requests
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")
        try:
            resp = requests.get(
                "https://www.okx.com/api/v5/public/open-interest",
                params={"instId": inst_id}, timeout=10)
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                return float(data["data"][0].get("oi", 0))
        except Exception:
            pass
        return None

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list[dict]:
        import requests
        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")
        bar_map = {"5m": "5m", "1h": "1H", "4h": "4H", "15m": "15m"}
        bar = bar_map.get(timeframe, timeframe)
        params = {"instId": inst_id, "bar": bar, "limit": str(limit)}
        resp = requests.get(
            "https://www.okx.com/api/v5/market/candles",
            params=params, timeout=15)
        data = resp.json()
        if data.get("code") != "0" or not data.get("data"):
            logger.warning(f"fetch_ohlcv failed for {symbol} {timeframe}: {data.get('msg', '')}")
            return []
        bars = []
        for b in reversed(data["data"]):
            bars.append({
                "timestamp": datetime.fromtimestamp(int(b[0]) / 1000),
                "open": float(b[1]), "high": float(b[2]),
                "low": float(b[3]), "close": float(b[4]), "volume": float(b[5]),
            })
        return bars

    def fetch_ohlcv_extended(self, symbol: str, timeframe: str, min_bars: int = 500) -> list[dict]:
        """迭代下载历史 K 线，使用 OKX REST API 直接分页"""
        import time, requests

        inst_id = symbol.replace("/", "-").replace(":USDT", "-SWAP")
        bar_map = {"15m": "15m", "1h": "1H", "4h": "4H", "5m": "5m"}
        bar = bar_map.get(timeframe, timeframe)

        all_bars = []
        after_ts = None

        while len(all_bars) < min_bars:
            params = {"instId": inst_id, "bar": bar, "limit": "300"}
            if after_ts is not None:
                params["after"] = str(after_ts)

            # 先尝试历史端点（支持更早的数据）
            try:
                resp = requests.get(
                    "https://www.okx.com/api/v5/market/history-candles",
                    params=params, timeout=15)
                data = resp.json()
            except Exception:
                break

            if data.get("code") != "0" or not data.get("data"):
                # 回退到实时端点
                try:
                    resp = requests.get(
                        "https://www.okx.com/api/v5/market/candles",
                        params=params, timeout=15)
                    data = resp.json()
                except Exception:
                    break

            if data.get("code") != "0" or not data.get("data"):
                break

            batch = []
            for b in data["data"]:
                ts = datetime.fromtimestamp(int(b[0]) / 1000)
                batch.append({
                    "timestamp": ts,
                    "open": float(b[1]), "high": float(b[2]),
                    "low": float(b[3]), "close": float(b[4]), "volume": float(b[5]),
                })

            if not all_bars:
                all_bars = batch
            else:
                existing_ts = {b["timestamp"] for b in all_bars}
                new_batch = [b for b in batch if b["timestamp"] not in existing_ts]
                if not new_batch:
                    break
                all_bars = sorted(batch + all_bars, key=lambda x: x["timestamp"])

            # 下一次请求: after = 最老 K 线 ts（毫秒）
            oldest_ts_ms = int(batch[-1]["timestamp"].timestamp() * 1000)
            after_ts = oldest_ts_ms - 1
            time.sleep(0.2)

        return sorted(all_bars, key=lambda x: x["timestamp"])

    # --- WebSocket ---
    async def ws_connect(self, symbols: list[str], timeframes: list[str],
                         on_kline: Callable, on_trade: Callable | None = None):
        self._symbols = symbols
        self._timeframes = timeframes
        self._on_kline = on_kline
        self._on_trade = on_trade
        self._running = True
        self._ws = None
        asyncio.create_task(self._read_loop())

    async def _read_loop(self):
        while self._running:
            try:
                import websockets
                self._ws = await websockets.connect(
                    "wss://ws.okx.com:8443/ws/v5/business", ping_interval=30
                )
                logger.info("WebSocket connected")

                # 订阅
                for sym in self._symbols:
                    for tf in self._timeframes:
                        inst_id = sym.replace("/", "-").replace(":USDT", "-SWAP")
                        tf_okx = f"candle{tf.upper()}" if "H" in tf.upper() else f"candle{tf.lower()}"
                        await self._ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [{"instId": inst_id, "channel": tf_okx}],
                        }))

                # 读取消息
                while self._running:
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
                                self._on_kline(sym, tf, candle)
                    except asyncio.TimeoutError:
                        continue
            except Exception as e:
                if self._running:
                    logger.warning(f"WS disconnected: {e}, reconnecting in 30s...")
                await asyncio.sleep(30)
            finally:
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

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
