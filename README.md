# TradingMonitor V2

精简版 OKX 永续合约实时监控 + 持仓预警系统。

## 快速开始

```bash
pip install -r requirements.txt
cp secrets.yaml.example secrets.yaml
python main.py
```

## 目录

| 文件 | 职责 |
|------|------|
| main.py | 入口 + 主循环 + 飞书报告格式化 |
| okx.py | OKX REST + WebSocket + 内存缓存 |
| indicators.py | 技术指标计算（纯函数） |
| signals.py | 信号定义 + 检测函数 |
| notify.py | 飞书推送 |
| history.py | 历史数据下载 + 统计分位数 |
| position.py | 持仓监控 + 止损策略 |
| utils.py | 日志 + 健康检查 |
| config.yaml | 配置文件 |
| secrets.yaml | API Key (不提交) |

## 信号列表

BB压缩 / RSI极值 / MA汇聚 / 量能爆发 / TTM压缩 / RSI背离 / MACD背离 / 量价突破

加新信号：signals.py 写一个 check 函数 + 注册一行。
