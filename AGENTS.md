# AGENTS.md

## 这是什么
TradingMonitor V2 — OKX 永续合约实时监控 + 量化研究系统。Python 3.10，仓库根目录平铺的顶层 .py 模块（无包结构），无测试框架、无 CI、无 lint。docs/ 是策略知识库；README.md 的信号清单已过时，以 docs/LIVE_SYSTEM.md 为准。

## 常用命令
- 启动 live: `cp secrets.yaml.example secrets.yaml`（填密钥）→ `python3 main.py`
- 重启: `bash start.sh`（杀旧进程 → nohup 启动；日志 logs/main.log，启动输出 /tmp/v2.log）
- 健康检查: `curl localhost:8080`（端口见 config.yaml health.port）
- 重建自选列表并重启: `bash update_watchlist.sh`（crontab 建议: 每周一 `0 8 * * 1`）

## Live 系统
- 入口 main.py `async_main()`；每 5 分钟对齐已收盘 5m bar 扫描：indicators.compute → relative_strength → signals → main.py `_enrich_alert`（富化）→ market_state.scene_of（场景引擎）→ 飞书推送
- 信号 = signals.py 的 `_check_*` 函数 + 在 SIGNALS 注册；**信号本身无 edge（≈50%），edge 全部来自场景引擎**——改信号/场景后必须回测验证，不能只看单信号命中率
- 富化标记逻辑集中在 main.py `_enrich_alert`（改标记去那里）
- config.yaml 控制自选列表/指标参数/持仓监控（position.enabled=false 默认关，需 okx api_key）
- 美股代币仅美盘 UTC 13:30-21:00 推送，金属/币 24h；显示时间均为北京时间

## 回测与研究（重点）
- 自研回测引擎（backtest_*.py）已整体删除（2026-08-03），后续回测验证改用第三方框架（待定）
- `data/backtest.db`（gitignored，保留）：20 标的 × 5m/1h/4h × 3 年（2023-08→2026-08，约 690 万行）原始 K 线，可作为第三方框架的数据源；与 live 的 data/history.db 完全独立。如需补充/重下数据可参考 history.py 的 OKX history-candles 分页逻辑（after 参数翻向过去）
- **未来函数是本仓库第一大禁忌**：K线时间戳 = bar 开盘时间；研究只能用已收盘 bar。此前整批研究（level/episode 体系等）因未来函数污染作废，两天工作白费（git log: "reset: delete all research (contaminated by future-function)"）。**研究已重头开始（2026-08-03）**，旧研究结论一律不可直接采信，必须用无未来函数方法重验
- 新研究：用第三方回测框架搭建，脚本必须写明无未来函数设计；结论落地后提交（提交前缀风格: `research:` / `live:` / `fix:` / `docs:`）

## 文档
- docs/LIVE_SYSTEM.md — 当前 live 系统规格（2026-08-03 更新；注意部分验证结论基于已作废研究，落地标记待重验）
- docs/SESSION_SUMMARY.md — 研究结论速查（同样待重验；其"测试工具"一节引用的 backtest_*.py 已删除）
- 多处引用 docs/STRATEGY_VALIDATION.md（45章）但该文件不存在——以代码和重验结果为准

## 注意
- 无测试/无 lint，验证 = 运行脚本读输出
- secrets.yaml 必须存在（gitignored）；勿提交 data/*.db、logs/*
