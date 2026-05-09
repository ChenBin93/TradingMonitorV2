# TradingMonitor V2 — 精简架构设计文档

> 版本: v2.0-draft  
> 日期: 2026-05-09  
> 状态: 设计阶段  

---

## 1. 设计哲学

V1 的问题不是"缺少什么"，而是**包装层太多**。30+ 文件、多层抽象，但核心逻辑不超过 2000 行。

V2 的原则：

1. **一个模块做一件事** — 不搞 ABC 抽象直到真的有第二个实现
2. **信号就是代码，不是配置** — YAML 定义信号反而限制灵活性
3. **参数集中，逻辑分散** — 阈值放一起，检测函数各管各
4. **扩展靠复制+适配，不靠继承** — 加交易所就 copy 一个 client 文件

---

## 2. 架构总览

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  OKX WS  │    │  缓存    │    │  指标    │    │  信号    │
│ 实时行情  │───▶│ K线Cache │───▶│ 计算     │───▶│ 检测     │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                     │
                     ┌───────────────────────────────┤
                     ▼                               ▼
              ┌──────────┐                   ┌──────────┐
              │  去重    │                   │  TOP5    │
              │  过滤    │                   │  排序    │
              └──────────┘                   └──────────┘
                     │                               │
                     └───────────┬───────────────────┘
                                 ▼
                          ┌──────────┐    ┌──────────┐
                          │  飞书    │    │  持仓    │
                          │  推送    │    │  监控    │
                          └──────────┘    └──────────┘

后台:
┌──────────┐    ┌──────────┐
│ 历史数据  │    │ 历史统计  │    (异步，不阻塞主循环)
│ 下载     │───▶│ 计算     │
└──────────┘    └──────────┘
```

---

## 3. 目录结构

```
v2/
├── main.py            # 入口 + 主循环 + 飞书推送格式化
├── config.yaml        # 扁平配置文件
├── secrets.yaml       # API Key（不提交）
├── okx.py             # OKX REST + WebSocket + 缓存（一个文件搞定）
├── indicators.py      # 指标计算（从 V1 复用）
├── signals.py         # 信号定义 + 检测函数
├── notify.py          # 飞书客户端
├── history.py         # 历史数据下载 + 统计 + SQLite
├── position.py        # 持仓监控 + 止损策略
├── utils.py           # 日志 + 健康检查
├── requirements.txt
└── README.md
```

**共计 11 个文件**，相比 V1 的 31 个，每个文件职责清晰不嵌套。

---

## 4. 模块设计

### 4.1 `main.py` — 编排中心（~300 行）

```python
# 伪代码: 主循环
def main():
    config = load_yaml("config.yaml")
    secrets = load_yaml("secrets.yaml")
    feishu = Feishu(secrets["feishu"])
    cache = KlineCache(max_candles=500)
    okx = OKXClient(secrets["okx"])
    symbols = okx.get_top_symbols(config["top_n"])
    
    # WebSocket 连接
    okx.ws_connect(symbols, timeframes=["15m","1h","4h"],
                   on_kline=cache.update, on_trade=cache.update_price)
    
    # 后台历史数据
    thread = start_history_download(okx, symbols, config["history"])
    
    # 主扫描循环
    while True:
        sleep(120)  # 每 2 分钟
        alerts = []
        for sym in symbols:
            for tf in ["15m", "1h", "4h"]:
                df = cache.get_df(sym, tf)
                if len(df) < 30: continue
                
                ind = indicators.compute(df, config["indicators"][tf])
                state = SignalState(sym, tf, ind)
                
                for sig_def in SIGNALS:
                    result = sig_def.check(state)
                    if result:
                        alerts.append(Alert(sym, tf, sig_def.id, **result))
        
        # 去重 + 推送
        filtered = dedup_and_filter(alerts)
        if filtered:
            feishu.send(format_report(filtered))
            feishu.send(format_top5(rank(filtered)))
```

**数据流向**: WS → Cache → 指标 → 信号 → 去重 → 飞书。一条直线，没有跳转。

### 4.2 `okx.py` — 数据源（~350 行）

合并现在的 `exchange.py` + `websocket.py` + `cache.py` + `price_cache.py`（4 文件 575 行 → 1 文件 ~350 行）。

```python
class OKXClient:
    """OKX 数据客户端，封装 REST + WebSocket"""
    
    # --- REST ---
    def get_top_symbols(n: int) -> list[str]
    def fetch_ohlcv(symbol, timeframe, limit) -> list[dict]
    
    # --- WebSocket ---
    async def ws_connect(symbols, timeframes, on_kline, on_trade)
    async def ws_close()
    
    # --- 内置缓存 ---
    def get_candles(symbol, timeframe) -> list
    def get_df(symbol, timeframe) -> pd.DataFrame  # 直接返回 DataFrame

class KlineCache:
    """K 线内存缓存，线程安全"""
    def update(symbol, tf, candle)
    def get_closed(symbol, tf) -> list      # 已完成K线
    def get_df(symbol, tf) -> pd.DataFrame
```

**加新交易所**: 复制 `okx.py` → `binance.py`，实现相同接口。main.py 里 `client = BinanceClient()`。不需要 ABC。

### 4.3 `indicators.py` — 指标计算（~450 行）

**从 V1 完整复用**，纯函数无副作用。

```python
def compute(df, params) -> dict:
    """返回 {close, rsi, adx, bb_width, roc, macd_hist, ...}"""
```

### 4.4 `signals.py` — 信号系统（~300 行）

核心设计：**信号 = 检查函数 + 参数定义 + 显示信息**

```python
from dataclasses import dataclass

@dataclass
class SignalState:
    symbol: str
    timeframe: str
    ind: dict          # indicators.compute() 的返回值
    bbw_rank: float    # 由 main.py 填入（需要历史数据）

@dataclass
class SignalDef:
    id: str
    name: str          # 中文名
    check: Callable    # (state) -> dict | None
    params: dict       # 可调阈值
    tag: str           # 短标签，用于 TOP5 报告

# --- 检测函数 ---

def check_bb_squeeze(state):
    """布林带压缩"""
    bb = state.ind.get("bb_width")
    rank = state.bbw_rank
    if not bb or not rank: return None
    if rank <= state.params["threshold"]:
        return {"severity": "critical" if rank <= 10 else "high",
                "evidence": f"压缩位{rank:.0f}%", "score": 0.7}

def check_rsi_extreme(state):
    """RSI 极值"""
    rsi = state.ind.get("rsi")
    if not rsi: return None
    if rsi <= state.params["oversold"]:
        return {"direction": "long", "severity": "critical" if rsi <= 25 else "high",
                "evidence": f"RSI={rsi:.0f}", "score": 0.8}
    if rsi >= state.params["overbot"]:
        return {"direction": "short", "severity": "critical" if rsi >= 80 else "high",
                "evidence": f"RSI={rsi:.0f}", "score": 0.8}
    return None

# ... 其他信号同理

# --- 信号注册表（列表，不是注册器）---

SIGNALS: list[SignalDef] = [
    SignalDef("bb_squeeze",  "BB压缩",  check_bb_squeeze,
              {"threshold": 25}, "BB"),
    SignalDef("rsi_extreme", "RSI极值", check_rsi_extreme,
              {"oversold": 30, "overbot": 70}, "RSI"),
    SignalDef("ma_converge", "MA汇聚",  check_ma_converge,
              {"threshold": 0.5}, "MA"),
    SignalDef("volume_spike","量能爆发", check_volume_spike,
              {"threshold": 3.0}, "VOL"),
    SignalDef("ttm_squeeze", "TTM压缩", check_ttm_squeeze,
              {"min_bars": 5}, "TTM"),
    SignalDef("rsi_divergence", "RSI背离", check_rsi_divergence,
              {"min_dist": 2.0}, "RSI背"),
    SignalDef("macd_divergence","MACD背离", check_macd_divergence,
              {"min_dist": 2.0}, "MACD背"),
    SignalDef("volume_breakout","量价突破", check_volume_breakout,
              {"threshold": 1.5}, "突破"),
]
```

**加新信号**：

```python
# 1. 写检测函数（~15 行）
def check_my_signal(state):
    ...

# 2. 注册（1 行）
SIGNALS.append(SignalDef("my_signal", "我的信号", check_my_signal, {...}, "MY"))
```

对比 V1：不需要改 7 个地方。

### 4.5 `notify.py` — 通知（~100 行）

从 V1 复用 `notification/feishu.py`。

```python
class Feishu:
    def __init__(app_id, app_secret, chat_id)
    def send(text: str)
```

**加新渠道**: 复制 → `discord.py`，实现 `send(text)`。main.py 里 `notifiers = [Feishu(...), Discord(...)]`。

### 4.6 `history.py` — 历史数据（~400 行）

合并 V1 的 `history_db.py` + `history_downloader.py` + `background_tasks.py`（3 文件 723 行 → 1 文件 ~400 行）。

```python
class HistoryDB:
    """SQLite 存储 + 分位数查询"""
    def get_bbw_rank(symbol, tf, lookback, bb_width) -> float
    def get_volatility_rank(symbol, tf, lookback) -> float

def download_history(client, symbols, timeframes, days):
    """后台下载历史 K 线"""
    
def compute_stats(db, symbols, timeframes):
    """计算统计分位数，写入 DB"""
```

### 4.7 `position.py` — 持仓监控（~300 行）

从 V1 复用 `core/position_monitor.py`。精简掉独立的 FeishuClient 初始化，改为接收外部传入的 `notifier`。

```python
class PositionMonitor:
    def __init__(config, notifier)   # notifier 由 main.py 传入
    def start()
    def stop()
```

### 4.8 `utils.py` — 工具（~50 行）

```python
def setup_logging(level, file)   # 日志初始化
def start_health_server(port)    # 健康检查 HTTP 端点
```

### 4.9 `config.yaml` — 扁平配置

```yaml
# 不再嵌套 Pydantic 模型
top_n: 100
timeframes: [15m, 1h, 4h]
scan_interval: 120

indicators:
  15m: {roc_period: 5, rsi_period: 14, adx_period: 14, ...}
  1h:  {roc_period: 10, rsi_period: 14, adx_period: 14, ...}
  4h:  {roc_period: 20, rsi_period: 14, adx_period: 14, ...}

alert:
  dedup_minutes: 30
  min_confidence: 0.65

history:
  lookback_days: [90, 180, 365]
  db_path: data/history.db

health:
  port: 8080
```

---

## 5. 与 V1 对比

| | V1 | V2 |
|---|---|---|
| 源文件 | 31 个 | 11 个 |
| 总行数 | ~6000 | ~2500 |
| 信号定义 | YAML → dataclass → register_func → check | 1 个函数 + 1 行注册 |
| 加信号改动点 | 7-10 处 | 2 处 |
| 数据源 | 4 个文件（exchange + ws + cache + price） | 1 个文件 |
| 历史数据 | 3 个文件（db + downloader + bg_tasks） | 1 个文件 |
| 配置 | Pydantic 10 个 Model 191 行 | 扁平 YAML + dict 访问 |
| 去重 | 3 套实现 | 1 个 dict |
| 双扫描器 | realtime + batch 两套 | 1 套 WebSocket 实时 |
| 监控 | loguru + health endpoint | 同 V1 |

---

## 6. 扩展指南

### 加新交易所（如 Binance）

1. 复制 `okx.py` → `binance.py`
2. 修改 REST API 调用和 WebSocket 消息解析
3. `main.py`: `client = BinanceClient()`

### 加新信号

1. `signals.py`: 写 `check_xxx(state)` 函数
2. `signals.py`: 在 SIGNALS 列表追加一行
3. 完。

### 加新推送渠道（如 Discord）

1. `notify.py`: 加 `class Discord: def send(text)`
2. `main.py`: `notifiers.append(Discord(...))`

### 加新市场（如 A 股）

1. 复制 `okx.py` → `astock.py`，适配新浪/东方财富 API
2. `main.py`: 开第二个 async loop 或按交易时间调度
3. `signals.py`: 信号逻辑复用，可单独建一组 SIGNALS_ASTOCK 调不同阈值

---

## 7. 从 V1 复用清单

以下模块直接复制到 V2，基本不修改：

| V1 文件 | V2 对应 | 修改 |
|---------|---------|------|
| `signals/indicators.py` | `indicators.py` | 无 |
| `notification/feishu.py` | `notify.py` | 无 |
| `core/position_monitor.py` | `position.py` | 注入 notifier，去掉独立 FeishuClient |
| `data/cache.py` | 合并进 `okx.py` | KlineCache 类保留 |
| `data/websocket.py` | 合并进 `okx.py` | OKX WS 逻辑保留 |
| `data/exchange.py` | 合并进 `okx.py` | REST 逻辑保留 |
| `data/history_db.py` | 合并进 `history.py` | 保留 SQLite 操作 |
| `data/history_downloader.py` | 合并进 `history.py` | 保留下载逻辑 |
| `utils/logging.py` | `utils.py` | 无 |
| `utils/health.py` | `utils.py` | 无 |

---

## 8. 待定事项

- [ ] 持仓监控的 `subprocess + curl` 调用是否改为 `requests` 库（更简洁）
- [ ] 历史数据 `lookback_days` 是否默认 90/180/365 还是精简
- [ ] 飞书格式化是否需要保持 V1 的报告样式
