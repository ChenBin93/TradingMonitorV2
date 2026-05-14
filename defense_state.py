# 防线状态追踪 — 记录哪些防线被突破过，供 retest 信号查询

import time
from collections import defaultdict


class DefenseState:
    """追踪防线突破记录，支持 retest 信号"""
    def __init__(self, max_age_hours: int = 48):
        self._store: dict[str, list[dict]] = defaultdict(list)
        self._max_age = max_age_hours * 3600

    def record_break(self, symbol: str, price: float, side: str, break_dir: str):
        """记录防线被突破: side=resistance/support, break_dir=up/down"""
        self._clean(symbol)
        self._store[symbol].append({
            "price": price,
            "side": side,
            "dir": break_dir,
            "ts": time.time(),
        })

    def was_broken(self, symbol: str, price: float, side: str, atr: float) -> dict | None:
        """检查某个防线是否在最近被突破过（同侧同价附近）"""
        self._clean(symbol)
        for rec in self._store.get(symbol, []):
            if rec["side"] == side and abs(rec["price"] - price) <= atr * 0.5:
                return rec
        return None

    def get_recent_breaks(self, symbol: str) -> list[dict]:
        """获取某个币种最近的突破记录"""
        self._clean(symbol)
        return list(self._store.get(symbol, []))

    def _clean(self, symbol: str):
        now = time.time()
        self._store[symbol] = [
            r for r in self._store.get(symbol, [])
            if now - r["ts"] < self._max_age
        ]


# 全局单例
_global_state = DefenseState()


def get_defense_state() -> DefenseState:
    return _global_state
