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
- **研究权威文档 = `research/PLAN.md`**（C 系列协议，2026-08-12 定稿）。A/B 系列（2026-08-03~05）29 项研究因未来函数/口径违规于 2026-08-12 整体作废，已移入 `research/archive/`——**旧研究结论一律不得引用**（作废原因见 research/archive/README.md）
- `data/backtest.db`（gitignored，保留）：20 标的 × 5m/1h/4h × 3 年（2023-08→2026-08，约 690 万行）原始 K 线；与 live 的 data/history.db 完全独立。如需补充/重下数据可参考 history.py 的 OKX history-candles 分页逻辑（after 参数翻向过去）
- **未来函数是本仓库第一大禁忌**：K线时间戳 = bar 开盘时间；研究只能用已收盘 bar；MTF 对齐走 `align_higher`；结构检测必须在线聚类+冻结（全样本聚类=未来函数）。**每次研究前必跑两道门禁，任一失败即停**：
  1. 引擎门禁：`python3 -m pytest research/tests -q`（约 4 分钟；黄金测试 + 未来函数不变性测试 + MTF 边界 + 真实数据对拍 + 性质测试）
  2. 研究脚本门禁：`python3 research/check_study.py <script>`（import 白名单/禁止模式 AST 检查/.out 头部 sha256 校验/GATE 区块校验/数字指纹/成对性/发布门槛强制，任一违规 FAIL 并报文件+行号）
- **分层编号**：c1x 描述层（市场事实，纯描述，无入场）/ c2x 条件层（条件 1:1 胜率/倾向）/ c3x 策略层（策略期望 + 成本核算 + vectorbt 二层 + live 试点）。脚本写 `research/studies/`，输出 `research/notes/`
- **研究脚本模板**：docstring 预注册（标题 + 预注册假设 H1.. 运行前锁定、结论逐条回应 + 无未来函数设计逐特征信息边界表 + 数据/参数声明 + 发布门槛自检 + 运行命令），check_study 校验；`PARAMS` 唯一参数源；GATE 自检内置（GBM ≥30 种子、与同管线重放的无条件基线 ≈50%±1pp，失败 SystemExit）
- **发布门槛**（任何"正期望/edge/有效"结论必须全绿）：真实 − 正确构造的随机游走基线（GBM，固定种子序列 ≥30，禁换种子刷结果）> 0 且分年稳定且 n ≥ MIN_N；条件层加 GATE≈50% + 分年 + 净效应 + 成本行；策略层加成本核算。结论每个数字带 `(.out:L行号)` 引用，禁止新引入 .out 没有的分层
- **因果模块**（研究基础设施唯一出口，禁止自写替代）：
  - `research/causal.py` — rolling_percentile / rolling_rank（禁全样本分位）/ causal_confirmed（事后标签因果可用版，conf∈[t-60,t-24]，[t-23,t] 内突破剔除）/ frozen_cluster（在线聚类+冻结）
  - `research/ctx.py` — make_ctx 是截断对齐的唯一构造路径（内部统一 `iloc[warmup:]`）；entries_from_events 保证全长度布尔
  - 引擎（caliber.py / outcome.py / sim_market.py / data_loader.py）+ research/tests/ 经 2026-08-12 审计无未来函数，保留；研究脚本禁止硬编码口径
- **禁止项**（check_study AST 拦截）：自写 outcome 引擎（同现 tp/sl 变量 + 逐 bar 循环）；价格数组手动切片（必须走 ctx.make_ctx）；`np.percentile/quantile` 作用于特征（必须 rolling_percentile）；`searchsorted(conf,...)` 事后标签条件化（必须 causal_confirmed）；研究脚本互 import
- **研究纪律**：改脚本 = 重跑 + 结论重写（sha256 校验）；每研究一个 commit（脚本+notes+.out）；禁止从脏工作树运行；负结果同等归档；提交前缀风格: `research:` / `live:` / `fix:` / `docs:`

## 文档
- research/PLAN.md — **研究权威文档**（C 系列协议：三层门禁/编号分层/脚本模板/发布门槛/路线图）
- **docs/RESEARCH_C_SERIES_SUMMARY.md — C 系列最终总结（2026-08-13）：13 项研究结果总表、可信事实清单、Phase 2 四候选全负结论与四条方法学教训；研究数字的唯一速查入口**
- docs/LIVE_SYSTEM.md — 当前 live 系统规格；**运行规格类内容（端口/启动/推送时间窗/live 数据流）仍有效**；涉及研究数字的段落（SCENE_WR 胜率表、关键位质量分、"类型A 贴位"等）已加作废戳与行内标注（2026-08-12），不采信
- docs/SESSION_SUMMARY.md — 已加作废戳（2026-08-12 批次作废），研究结论速查不再可采信
- docs/MARKET_STATE_STRATEGY_MAP.md — 已加作废戳，不采信
- 多处引用 docs/STRATEGY_VALIDATION.md（45章）但该文件不存在——以代码和 research/PLAN.md 为准

## 注意
- 无测试/无 lint，验证 = 运行脚本读输出
- secrets.yaml 必须存在（gitignored）；勿提交 data/*.db、logs/*
- **research/levels.py 是 live 依赖，不得归档**：main.py:581 → key_levels.py:19 → `research.levels.cluster_levels`（关键位监控核心）。A/B 批次未提交的 levels batch helpers 改动已存档于 `research/archive/patches/levels_batch_helpers_uncommitted.patch`（研究全作废不留用）；levels.py 修复（PLAN.md §2 R1/R2）须在 HEAD 基础上重写
- A/B 批次作废原因（一句话）：A3 索引错位、confirmed 未来标签泄漏、B4 结论不可复现、B4e 全样本分位等致命缺陷，详见 research/archive/README.md 与 research/PLAN.md
