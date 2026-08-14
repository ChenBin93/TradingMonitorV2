# Kaufman《Trading Systems and Methods》学习计划

边学书、边用自有数据考证。每个概念一个学习单元，五段格式；考证研究走 C 系列协议（PLAN 预注册 / GATE / check_study，编号 c3x 延续）。

## 单元格式
0. **忠实复现基线** — 先按书原口径复现（原频率/窗口/基线/判据），复现出结果后才允许扩展（换市场/换周期）；复现失败先查口径错位清单（窗口定义、频率锚点、基线选择、成本假设、信号定义），口径逐字对齐后仍失败才构成"书结论不成立"
1. **大白话讲解** — 概念在说什么、解决什么问题
2. **书的原始逻辑** — Kaufman 为什么这么想、公式思路
3. **对接我们的研究** — 与 C 系列的印证 / 矛盾 / 空白
4. **实证考证** — 预注册研究（可选）
5. **修正理解** — 数据判卷后的最终版本（考证完成后回写）

## 模块
| 模块 | 书对应章节 | 状态 |
|---|---|---|
| 0 地基 | CH1-2 | **✅ 完成（6 单元，2026-08-13）** |
| 1 趋势家族 | CH5-8 | **✅ 完成（7 单元，2026-08-13）** |
| 2 动量与振荡器 | CH9 | 未开始 |
| 3 时间与季节 | CH10-11 | 未开始 |
| 4 成交量与持仓 | CH12 | 未开始（需先确认数据） |
| 5 行为与形态 | CH14-15 | 未开始 |
| 6 自适应技术 | CH17 | 已学一半（c27/c29） |
| 7 价格分布区带 | CH18 | 未开始 |
| 8 多时间框架 | CH19 | 未开始 |
| 9 测试方法论与风险 | CH21/23 | 已学一半（三层门禁 / c26） |

## 规则
- 考证研究编号沿用 C 系列（c30、c31…）
- **两档制**（详见 research/PLAN.md 学习级考证节）：跟书学习用**学习级**（BTC/ETH + 传统对照篮、GBM 30 种子、无分年、MIN_N=100、**跳过 pytest/check_study 两道门禁**、结论标 `[学习级]`、不作交易依据）；有交易含义才升级**研究级**（20 标的 × BY_YEAR + 两道门禁全绿）
- 两档都不松的底线：GBM 零假设对照（脚本内置 GATE）+ 无未来函数纪律（causal 库/掩码/禁切片，自我约束）+ docstring 预注册 + .out 数字引用
- **截面纪律（c37 起）**：加密截面研究必须报告有效独立样本数 n_eff（相关矩阵特征值比）——20 币 ≈ 2 个独立信息源，结论置信度按 n_eff 而非标的数计；跨市场截面优先混入低相关传统市场（control.db）扩面
- **传统市场对照篮**：data/control.db（SPY/CL=F/GC=F/EURUSD=X/^TNX × 1d/1h，control_data.py 下载）。书结论先在传统市场验证成立，加密证伪才有"加密特殊性"语义
- 学习单元讲解存放在 `docs/LEARNING/moduleN/`
- 每个单元的"修正理解"在考证完成后回写该单元文件
- 书原文提取文本在 `/tmp/opencode/TSaM.txt`（gitignored 的 PDF 在 research/TSaM.pdf）

## 单元索引
- 模块 0：`module0/unit01_noise_and_er.md`（噪声与效率比 ER，✅ c27/c29/c30/c32/c34/c35/c36/c37）
- 模块 0：`module0/unit02_run_fat_tail.md`（随机游走与运行肥尾，✅ c31/c33）
- 模块 0：`module0/unit03_frequency_and_style.md`（数据频率与交易风格，✅ 收口于 c30/c32/c35/c36）
- 模块 0：`module0/unit04_returns_and_annualization.md`（收益计算与年化，方法论，无考证）
- 模块 0：`module0/unit05_moments_kurtosis.md`（分布矩与峰度，✅ c39）
- 模块 0：`module0/unit06_autocorrelation_dw.md`（自相关与 DW 统计，✅ c38）
- 模块 1：`module1/unit01_event_driven_swing.md`（事件驱动趋势与 swing 点位，✅ c40）
- 模块 1：`module1/unit02_donchian.md`（N 日突破与 Donchian 4 周规则，✅ c41/c43）
- 模块 1：`module1/unit03_regression_trend.md`（回归分析与斜率趋势，✅ c44/c45）
- 模块 1：`module1/unit04_time_series_components.md`（时间序列分解，方法论收口，无考证）
- 模块 1：`module1/unit05_ma_and_lag.md`（移动平均与滞后，✅ c46）
- 模块 1：`module1/unit06_trend_profit_source.md`（趋势系统的利润来源，✅ 收口于 c31/c33/c42）
- 模块 1：`module1/unit07_system_comparison.md`（趋势系统比较与选速，✅ c47）
- 模块 2：`module2/unit01_momentum_rsi.md`（动量与振荡器，✅ c48）
- 模块 4：`module4/unit01_volume_funding.md`（成交量、持仓与价差，✅ c49；funding pilot 受 3 个月数据限制）
- 模块 6：`module6/unit01_kama.md`（KAMA 自适应均线，✅ c50）
