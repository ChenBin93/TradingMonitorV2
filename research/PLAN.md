# 研究重启方案（C 系列）

> 2026-08-12 定稿。A/B 系列 29 项研究经严格审计发现致命缺陷（A3 750 根索引错位、confirmed 未来标签泄漏线、B4 结论不可复现等），研究结论整体作废。本方案保留经审计验证的基础设施（62 项测试全过、无未来函数），重建研究协议与路线图。

## 0. 裁决与原则

- **基础设施保留加固**：caliber.py / outcome.py / sim_market.py / data_loader.py + 62 项测试经审计无未来函数，保留；补齐缺口。
- **旧研究归档作废**：A/B 系列脚本+结论移入 `research/archive/`，数字一律不得引用；可信描述性规律仅作为"待重验 prior"，不是事实。
- **分层推进**：描述层（c1x）→ 条件层（c2x）→ 策略层（c3x），每层独立发布门槛。
- **核心教训**：旧门禁只测引擎、不测研究脚本——A3 的致命 bug 因此混过。新协议下研究脚本本身进入验证范围。

## 1. 三层门禁体系

### L1 引擎层（现有 62 测试保留 + 加固）

1. **引擎入口长度断言**（outcome.evaluate_forward / evaluate_forward_vbt / hold_sim.simulate_holds）：`len(entries) != len(close)` 即抛 ValueError——A3 类索引错位第一次运行即死，不再产出 .out 后才被发现。
2. **末根入场语义**：`i+1>=n` 不再静默丢弃，计入独立字段 `n_truncated`（非 expired）。
3. **data_loader 连续性检查**：`check_continuity(df, tf)` 报告 bar 间隔缺口，纳入 `verify()`。
4. **sim_market 时间锚定**：新增 `gbm_matching(ref_df, seed)`——索引锚定真实数据首根、长度相同、σ 由真实对数收益估计（修复 GBM 2023-01 vs 真实 2023-08 年份错位；替代各研究手写的 σ 估计模式）。
5. **caliber.py 新增 `MIN_GBM_SEEDS = 30`**（a38 6 种子、B3c 10 种子 ±2.6pp 的教训，统一取 30）。

### L2 模块层（每个公共 API 必须黄金+不变性+时序测试）

- **test_dow.py（dow_segments，当前零测试）**：手算道氏段黄金（确认 pivot→突破→HH→回撤→段结束逐事件断言）、多空镜像、真实+合成数据 cut 不变性、确认时序（HL pivot 确认前不得产生回撤事件——锁定"恢复窗口起点"类问题的结构侧）、amp_atr 无 pivot 段返回 NaN（顺手修 a6b 已披露 bug）。
- **test_levels_batch.py**：breakdown_all 按 `confirm_at` 门控（位形成前突破不计）、batch helpers 与已测单级函数对拍、`active_levels` 的 age_bars 追加数据后历史值不变（R1 修复锁）。
- **test_causal.py / test_ctx.py**：每个 helper 手算黄金 + 追加不变性。

### L3 研究脚本层（`research/check_study.py`，研究前第二道门禁）

`pytest research/tests -q` 通过后执行，任一违规即 FAIL（输出文件+行号）。检查项：

1. **import 白名单（AST）**：只允许 research 公共模块 + 第三方白名单；`from research.studies.xxx import`（研究互 import，B4e/B5c 模式）→ FAIL。
2. **禁止模式（AST）**：价格数组手动切片（必须走 `ctx.make_ctx`）；`np.percentile/nanpercentile/quantile` 作用于特征（除非 `[DESCRIPTIVE]` 分区）；自写 outcome（函数内同现 tp/sl 变量与逐 bar 循环，且不在引擎注册表）；`searchsorted(conf, ...)` 事后标签条件化（必须走 `causal.causal_confirmed`）。
3. **.out 头部校验**：meta 区块含 `study_id/script_sha256/data_range/params/gate`，sha256 与当前脚本一致（脚本改过未重跑 → FAIL）。
4. **GATE 区块校验**：存在无条件基线（真实+GBM）；GBM 种子数 ≥ MIN_GBM_SEEDS；1:1 口径研究 GBM 无条件胜率 ∈ 50%±1pp；MIN_N 检查结果打印。
5. **数字指纹**：结论文件中所有数字模式必须在对应 .out 中可查，找不到 → FAIL（B5c 的"1.00"即被此项击落）。
6. **成对性**：每个 .conclusion.md 必须有同名 .out，反之亦然。
7. **发布门槛强制**：结论含"正期望/edge/有效/可交易"等词时，必须全部门槛绿（真实−RW>0 且 .out BY_YEAR 成对分年可查 + n≥MIN_N + 成本核算小节），否则 FAIL。

**历史反例拦截对照（验收矩阵）**：

| 问题实例 | 拦截层 | 机制 |
|---|---|---|
| A3 索引错位 | L1 + L3 | 引擎长度断言（首次运行抛错）+ AST 禁手动切片 |
| B4e 全样本分位 | L2 + L3 | rolling_percentile 不变性测试 + AST 黑名单 |
| B5c 结论与脚本不符 | L3 | sha256 + 数字指纹 |
| confirmed 泄漏 | L2 | causal_confirmed 为唯一条件化出口 |
| B4 自写 _sim | L1/L3 | 官方非对称口径（见 §2）+ 引擎注册表 AST |
| a38 种子不足 | L1 + L3 | MIN_GBM_SEEDS=30 常量 + GATE 区块校验 |

## 2. 因果模式库与模块新增

**research/causal.py**（所有"当特征用"的结构统计唯一出口，全部过不变性测试）：

- `rolling_percentile(x, window, q, min_periods)` —— 滚动分位（禁全样本分位）。
- `rolling_rank(x, window)` —— 滚动百分位排位。
- `causal_confirmed(confirmed, w, lag_lo, lag_hi, confirm_cost)` —— 事后标签的因果可用版：`known[t]=1 ⟺ 存在 c∈[t-60, t-24] 满足 confirmed[c] 且确认窗口完全收在 t 之前`。默认语义 conf∈[t-60, t-24]；[t-23,t] 内突破的样本**剔除**而非视为存活（B2c/B2d 泄漏的正确替代）。
- `frozen_cluster(events, tol_fn, min_touch)` —— 在线聚类+冻结通用封装（levels.cluster_levels 泛化），研究脚本禁止自写聚类。
- `align_events(event_positions, t, lag_lo, lag_hi)` —— 因果窗口查询（bisect 封装）。

**research/ctx.py**：`make_ctx(df, warmup, state_fns)` 是截断对齐的唯一构造路径（内部统一 `iloc[warmup:]`，脚本永远见不到未对齐组合）；`entries_from_events` 保证全长度布尔。

**research/limit_sim.py**：`simulate_limit_entries(...)` limit 挂单成交模拟（B4e 语义合法化），注册引擎，黄金+性质+对拍测试。

**outcome.py 官方非对称口径扩展**：`evaluate_forward(..., t_target, t_stop)`——黄金测试 + 性质测试（GBM 上非对称结构优势必须真实/GBM 一致）+ vbt 对拍。消除 B4 自写 `_sim` 的根因：非对称需求存在但此前无官方口径。

**levels.py 修复**（**live 依赖，不得归档**：key_levels.py:19 ← main.py:581）：

- R1（:120 last_touch_idx 全样本突变）：Level 内部改为事件日志，`age_bars` 在 `active_levels(levels, t)` 内按确认时序重建快照；形成后字段不可变。live 只用最终带，输出语义不变。
- R2（:202-203/:331 未按 confirm_at 门控）：attempt/confirmed 增加 `t >= confirm_at` 掩码。
- 未提交 diff 处置：`git diff research/levels.py > research/archive/patches/levels_batch_helpers_uncommitted.patch` → `git checkout -- research/levels.py` 恢复 HEAD → 在 HEAD 基础上按 R1/R2 重写并补测。

## 3. 研究脚本与结论模板

### docstring 预注册格式（固定字段，check_study 校验）

```
"""C{id} {标题} ({date}, 无未来函数, {周期})

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1 ...: 若成立, 应观察到 {具体可证伪数字}
无未来函数设计说明 (逐特征信息边界表): 特征 | 计算方式 | 可用时点 | 依据
  ([DESCRIPTIVE] 事后统计单独成段并标注, 其结论禁止进入交易含义)
数据声明 / 参数声明 (集中于 PARAMS) / 发布门槛自检 / 运行命令
"""
```

### 脚本骨架

import 白名单 + `PARAMS` 唯一参数源 + **GATE 自检**（模板内置：GBM ≥30 种子、与研究同管线重放的无条件基线断言 ≈50%，失败 `SystemExit`——"违规即停"的技术落地）。自检用同一管线重放而非只调引擎，可暴露研究脚本自己的管线级错位。

**性能与调试约定（2026-08-13）**：①小批量调试先行——脚本带 `--dev`（前 3 标的 × GBM 3 种子、跳过 BY_YEAR/HOLDOUT，~1-3 分钟），管线 debug 用它，最终 .out 必须全量（sha256 锁定全量版）；②pytest 迭代期只跑相关文件，全量门禁只在最终运行前跑一次；③GBM 种子循环用 multiprocessing（白名单已放开）、研究内确定性重算用 functools.lru_cache；④rolling_percentile（pandas 后端 ~400x）与 cluster_levels（分桶 ~65x）已优化，研究脚本禁止自写逐根循环/线性扫描替代；⑤机器 2 核/3.3GB，峰值内存敏感，chunk 化处理大矩阵。

### .out 格式

```
# meta: study_id date script_sha256 data params
# GATE: gbm_seeds=30 无条件基线 真实..% GBM..% [PASS] MIN_N [PASS]
# RESULTS: ...
# BY_YEAR: 每个条件结论的 真实分年 + GBM分年 (成对出现)
```

### 结论模板

每个数字带 `(.out:L行号)` 引用；末尾固定发布门槛自查表；假设逐条回应（支持/否定/无法判断），禁止结论中新引入 .out 没有的分层或发现。

### 编号分层

| 层 | 编号 | 内容 | 门槛 |
|---|---|---|---|
| 描述层 | c1x | 市场事实，纯描述 | 无入场；[DESCRIPTIVE] 与因果特征分区；GBM 对照 |
| 条件层 | c2x | 条件 1:1 胜率/倾向 | GATE≈50% + 分年 + 净效应 + 成本行 |
| 策略层 | c3x | 策略期望（hold/limit 变体） | 全门槛 + vectorbt 二层 + live 试点 |

### 学习级考证（2026-08-13 新增：Kaufman 学习计划两档制）

跟书学习的考证用"学习级"，有交易含义的结论才升级"研究级"：

| | 学习级 | 研究级 |
|---|---|---|
| 标的 | BTC/ETH（+传统对照篮） | 20 标的 |
| GBM | 30 种子（成本可忽略） | 30 种子 |
| 分年 | 不做 | 必做 |
| MIN_N | 100 | 200 |
| pytest 引擎门禁 | **跳过**（引擎审计后未改动，学习级不碰引擎） | 必跑 |
| check_study 门禁 | **跳过**（2026-08-13 起；脚本纪律靠自我约束） | 必跑 |

两档都不松的底线（防自欺，几乎零成本）：

- **GBM 零假设对照**（脚本内置 GATE：同管线 null + 探测器自检，失败 SystemExit）
- **无未来函数纪律**（causal 库、布尔掩码禁切片、禁全样本分位——自我约束替代 check_study AST 检查）
- docstring 预注册（运行前冻结）+ dev 模式先行 + .out 数字引用 + 结论标注 `[学习级]`

- 学习级结论不得作交易依据；升级研究级须补 20 标的 × BY_YEAR 重跑 + 两道门禁全绿。
- **传统市场对照篮**（data/control.db，root 级 control_data.py 下载，gitignored）：SPY / CL=F / GC=F / EURUSD=X / ^TNX × 1d(3y) / 1h(≈2y)，yfinance 原始价（不复权）。
- 对照逻辑：书结论先在传统市场验证成立（书在其语境正确），加密再证伪/证成才有"加密特殊性"语义——**对照成立是加密特殊性结论的前提**。对照组口径与加密同管线、同 GBM null。

## 4. C 系列路线图

### 阶段 0：前置条件（加固 + 归档）

- P0-1 levels.py 未提交改动处置（patch 存档 → checkout → 重写）。
- P0-2 dow_segments / hold_sim 黄金测试补齐。
- P0-3 官方非对称口径 `evaluate_forward(t_target, t_stop)`。
- P0-4 A/B 归档 + 文档作废戳 + AGENTS.md 研究章节重写（见 §7）。
- P0-5 数据补充（可选）：历史延长至 4 年（history.py 分页）、OKX funding rate 历史（8h）。

### 阶段 1：描述层 c1x（prior 验尸）

通用 exit 模板：①GATE 过；②效应达预注册下限且 4 组合（2 参数×2 周期）同号；③真实−GBM 净差>0 且 GBM 侧无同类效应；④分年 ≥2/3 年同号；⑤每单元格 n≥MIN_N，全单元格输出（含不足者）。

| 编号 | 研究问题 | 关键假设（下限已打折） | 旧参照 | 优先级 |
|---|---|---|---|---|
| c11 | 方向不可预测 null 基线（Phase2 方向类对照锚） | H1: 各条件格方向概率 真实−GBM |Δ|≤1pp | a36+a1+B3c | 低（可并入 c18） |
| c12 | 波动长记忆+状态转移（因果口径） | H1: DFA-H≥0.70(1h)/0.65(4h)；H3: 低↔高直接转移率≤0.01（必经中）；z120 用滚动分位 | a15/a2 | 高（地基） |
| c13 | 趋势收益偏度特异性 | H1: up:late 偏度≥+1.5 且 GBM≤+0.3；H2: top5% K 贡献≥60% | a4 | 高（尾部线地基） |
| c14 | 位移约束普适（因果条件化） | H1: M2P50Δ≤-0.4ATR 全结构层同号；H3: 泄漏敏感性对照（因果版 vs B3 alive_at 版差异≤20%） | B3 | 高 |
| c15 | 触碰后波动释放 | H1: 净差≥+4pp 4 组合同号；H3: GBM 侧≤+1.5pp | B2b/B3 | 高 |
| c16 | 区间延续性（因果区间+双边界） | H1: w=6 留存净差≥+3pp；H4: 方向层净差≤0 | B3d | 高 |
| c17 | 趋势逆势折返（线索级） | H1: 沿趋势概率净差≤-2pp；H2: 阶段无梯度；H3: causal 角色层无额外信息 | B5/B5b/B5c | 中 |
| c18 | 触位 1:1 无增益复核 | H1: 4 方向 |Δ|≤1pp | B1 | 低（与 c11 合并） |
| c19 | 道氏段生命周期因果重做 | H1: 4h 存活 P(≥25) 真实−GBM≥+2pp；H2: 恢复率净差≤-8pp（打折重验）；恢复窗口起点=回撤低点确认 bar+1；气候条件全滚动 | a6 系 | 中（c22 不用道氏可砍） |
| c1a | 趋势持有正期望=纯结构 | H1: |期望差|<0.05R 全止损模式 | a35 | 低（默认砍，作 Phase3 sanity） |

交付顺序：**c12 → {c14, c15, c16} 并行 → c13 → c17 → {c11+c18 合并} → c19**。预计 10 项有效研究。

### 阶段 2：条件层 c2x（2026-08-13 amend：c23+c21 先行，c24 集成其中，c22 暂缓，新增 c25）

| 编号 | 候选 | 前提 | 主端点预注册要点（Phase 1 证据修正后） |
|---|---|---|---|
| c23 | 趋势逆势折返条件化 | c17 ✅（-3.6~-4.1pp） | 逆势入场（涨触阻空/跌触撑多），入场=触碰 bar 收盘（market）；**主端点**：对称 1:1（evaluate_forward，T=1.0，W=24）1h 胜率差≥+3pp；**探索端点**：非对称 t_target=1.0/t_stop=0.7（官方口径）；**成本模型预注册**：taker 0.05%×2+滑点 1bp+funding 0.01%×3（1h W=24 跨 3 个 8h 周期）；位带相对止损（位带外+0.3ATR）列为后续工作（引擎为标量 ATR 倍数，不支持逐入场止损距离）。1h 为主、4h 仅报净差（c17：4h 效应≈环境漂移）；**新增预注册分层**：阶段（early≈0 vs accel/late 强，c17 H2）、角色（未破更强，c17 H3）；**漂移分解**（c18 教训）：净差=真实−GBM 同管线，另报 真实−真实无条件（同方向）分解；预注册声明：成本后≤0 只能写"结构发现"非 edge |
| c21 | 区间触碰小目标 + limit 入场（B4/B4e 因果重做） | c14+c16 ✅（触碰留存 +4~7pp）；P0-3 就位 | limit buy @S.price（intrabar 触及成交），目标 0.3×ATR / 止损 0.7×ATR，W=6，胜率差≥+3pp 且期望差≥+0.05R；收盘入场版净差≈0（复现 B4e 追价伪影）；空头只报净差（GBM 路径偏置）；成本（maker 0.02%×2+滑点 0.5bp）后期望>0。**修正**：区间定义=因果成对活跃位+双侧 causal_confirmed（c16 口径，不用 B3d 宽度约束）；c15 教训：触碰条件化 GBM 机械偏置约 +1pp 需在净差口径扣除。参数网格预注册 + 开发/验证集分离（1h(2,0.3) 开发，其余验证） |
| c25 | 状态条件化方向倾向（新，c18/c11 发现） | c18 ✅（trend_up 进入 E1 -3~-4.6pp） | 预注册主端点：trend_up 状态段首进入，逆势（short）1:1，1h，W=24，胜率差≥+3pp；分层：阶段（early/accel/late，c18 诊断显示全部 ≈46-47%）、状态（trend_down 对称检验）；**漂移分解**（c18：无条件漂移层 + 状态增量层必须分离报告）；4h 仅报净差。定位：与 c23 的区别=无触碰条件（状态单独是否可交易），若有 edge 则信号频率远高于 c23 |
| c24 | 波动压缩→触碰→释放 择时过滤 | c12+c15 ✅ | **改为集成到 c21/c23**：作为两研究内的预注册子分析（触碰前 z120=低波动子集净差增量 ≥+1.5pp；GBM 上无同向增量门禁），不单独立项 |
| c22 | 尾部收割重做（a38） | c13 部分（up:late 偏度 +2.48/+1.83 全过、C_share 集中度 +9~12pp；dn:late 4h 未达） | **重做口径（2026-08-13 amend）**：c23/c25 实证 1:1 命中测不到端点/尾部效应，故改用 hold 引擎（hold_sim.simulate_holds）而非 1:1。H1（主端点）：trend_up 段首入场 long（early/accel 优先），固定 1×ATR 止损 + 峰值回撤 3×ATR trail 退出，1h，期望差（真实−GBM 同管线）≥ +0.10R；H2（对称）：trend_down 段首 short 只报净差（GBM 路径偏置，B4 教训）；H3（尾部集中分层）：C_share 类滚动集中度（rolling 口径，禁全样本分位）高于中位数的子集期望差增量 ≥ +0.05R；成本=taker 0.05%×2+滑点 1bp+funding 按持仓时长（8h 周期 0.01%）；GBM 30 种子；4h 仅报净差 |

**Phase 2 执行序**：c23 + c21 并行 → c25 → c22。每项按 10 项门槛清单裁决。

### 阶段 2.5：仓位管理方向（2026-08-13 重开，用户指定）

背景：Phase 2 证明"状态→入场方向"无 edge；但 c12 证明波动率可预测（H≈0.9），
状态作为**仓位调节器**（state-as-modulator）完全未测且证据支持度最高。

| 编号 | 研究 | 前提 | 预注册要点 |
|---|---|---|---|
| c26 | 波动目标仓位（vol targeting） | c12（H≈0.9）；c15（触碰后释放） | 入场=无条件 long 固定排程（每 W=24 根一笔，非重叠，无方向 edge 基座——纯隔离仓位效应）；对照 A=固定仓位（每笔 1 单位）、B=波动目标仓位（每笔 K/ATR_entry，风险恒定，K=1）；**美元 P&L 用相对 ATR（ATR/close）计价**（跨标的可比）。H1（前提）：ATR_entry 与后续 W 根实现波动正相关，真实 > GBM 同管线（GBM≈0）；H2（主端点）：B 的每笔 P&L 标准差比 A 低，且真实侧降低率 − GBM 同管线降低率 ≥ 10pp（GBM 侧降低=ATR 离散度的机械效应，必须扣除）；H3（回撤）：B 累积权益曲线最大回撤 < A（真实−GBM 净差 < 0）；H4（收益中性）：B 总收益不显著低于 A（无 edge 基座均值≈0 不变——纯风险改善，不创造收益）；成本=调仓手续费按仓位变化量计（taker 0.05%×|Δ仓位|）；GBM 30 种子同管线；1h 为主、4h 交叉；分年成对；门槛=H2 净差≥10pp 且 H3 过且分年 ≥2/3 年同向 |
| c27 | ER 方向效率轴（Kaufman CH17 灵感，书 P1） | c12/c14/c15/c17 + 书"噪声≠波动" | **已完成（2026-08-13）**：H1 正交性 ✅（高ER触碰在低/中/高波动三格 23.2/24.8/51.9%，均≥10%）；H2 释放调节 ✅（高ER vs 低ER 触碰的波动释放差真实 +16.27pp、GBM +0.26pp——ER 是释放的强调节轴，分年全正）；H3 折返调节 ❌ **方向相反**（高ER触碰端点折返更多：D1净差 −3.44pp/1h、−6.67pp/4h，分年 3/3 负——高效率趋势触位后更容易折返，与 Kaufman"高ER=真趋势=折返小"直觉相反，但与书 CH12"量 spike=耗尽（人人上船=船沉）"逻辑一致）。**c28 前提作废**：ER 自适应止损（Knapp 规则）假设高ER趋势更稳，被 H3 否定——c28 取消；反向（高ER触位折返加深）如需利用须新预注册且注意 c23/c25 的 1:1 死胡同 |
| c29 | ER 折返效应的日线背景条件化（用户假设：书 vs 实测矛盾=时间视角不同） | c27（H3 反转）+ c19（日线层口径） | 1h 触碰事件按日线背景分组：A=日线 ER 高分位（日线序列 rolling 分位，daily_resample + 已收盘对齐，c19 模式）且日线净位移方向与触碰方向一致（大周期真趋势背书）；B=其余（无背书）。H1：A 组触碰的端点折返（D1，c27 口径）比 B 组少 ≥ 2pp（书的直觉在 A 组成立）；H2：A 组触碰后波动释放（E1，c15 口径）比 B 组低 ≥ 3pp（真趋势温和通过，不炸）；H3：GBM 同管线无此 A/B 差异（< 1pp，排除机械性）。描述层；GBM 30 种子；1h 为主 4h 交叉 | **已完成（2026-08-13）**：用户"时间视角"假设**证伪**——日线真趋势背书（A 组）触碰折返更深（D1 A 45.02% vs B 50.21%，A−B −5.19pp；4h −3.73pp；分年 3/3 负），书直觉在日线尺度也不恢复；H2 不支持（1h −1.10pp 不足门槛、4h +2.20pp 反向）；H3 ✅（GBM A/B 差 +0.27pp 无机械性）。三条管线（1h ER/日线 ER/日线顺风）收敛：趋势质量越高触碰越反 |
| c30 | ER 分周期单调性 + 标的截面（学习单元 U0-1③/U0-3 考证：频率→噪声） | c27（ER 口径）/c29（daily_resample） | 同一标的 ER_10 中位数随周期 1h→4h→日线单调上升（20 标的全部成立，书 CH1"频率越低噪声越低"）；标的间 ER 排名跨周期 Spearman ρ ≥ 0.5（噪声是标的天性）；GBM 同管线无单调性（iid 下 ER_n 分布与频率无关——数学 null）。数据 1h/4h/日线（**5m 不在本次范围**，数据量风险）；描述层；GBM 30 种子；BY_YEAR=每年跨标的 ER_10 中位数差（日线−1h）符号 | **已完成（2026-08-13）**：H1 证伪（单调上升仅 12/20 成立，8 违反；三频率中位数同挤 27~32%，真实日线−1h 差 +1.4pp vs GBM +0.3pp，分年符号不稳）；H2 证伪（Spearman 0.275/−0.008/0.487 全 <0.5）；H3 ✅（GBM 30 种子频率平坦 27.9/28.0/28.2，数学 null 成立）。书"频率越低噪声越低"在固定窗口口径下不成立；未测版本（固定日历窗、图 1.7 跨标的 PF 截面）留待模块 1（U1-6/U1-7） |
| c32 | ER 频率梯度的传统市场对照（U0-1 补验：书断言在书的市场成立吗？用户要求） | c30 同管线 + c31 gap 匹配 null | 同 c30 管线：ER_10 中位数 1h vs 1d。加密 BTC/ETH 1h+日线（c30 口径）；传统对照 SPY/CL=F/GC=F/EURUSD=X/^TNX（control.db，1h 原始序列 + 1d；4h 无源跳过；会话缺口混入 ER 窗口=设计偏离，docstring 注明）。H1：传统市场 ER_10 中位数 1h < 1d（5 标的全部，书"频率越低噪声越低"在其语境成立）；H2：传统日线−1h 差（5 标的均值）− 加密日线−1h 差（BTC/ETH 均值） ≥ 2pp（传统有梯度、加密没有 → 加密特殊性）；H3：GBM 同管线无梯度（30 种子，**gap 匹配 null**：按各对照标的会话日历掩码 GBM bar，c31 模式）。学习级：无 BY_YEAR、MIN_N=100、描述层、[学习级] | **已完成（2026-08-13，[学习级]）**：H1 不支持且反向（0/5：传统 5 标的 1h ER 全部高于日线，−0.3~−8.1pp——隔夜跳空在 10 根 1h 窗口占主导把 ER 顶高，设计偏离已注明）；H2 不支持（传统 −4.3pp − 加密 +1.9pp = −6.2pp，符号相反）；H3 ✅（GBM gap 匹配 null 梯度 +0.2~+0.4pp 平坦，负梯度非管线伪影）。书断言在传统市场同样证伪——**c30+c32 双市场闭环：未发现该断言可成立的市场**；加密特殊性不成立（方向反转）。ER 中位数度量在会话缺口数据上有局限，后续可用 log 收益 run/自相关度量或剔隔夜 bar（新预注册候选） |
| c31 | 运行肥尾 vs GBM + 传统市场对照（学习单元 U0-2：书 CH1/CH8"运行序列肥尾=趋势系统利润唯一来源"） | U0-2 框架 + c12 | 加密 BTC/ETH 1h/4h 同向连续 run 长分布 vs GBM；传统对照 SPY/GC=F/EURUSD=X 1h 同口径同管线。H1：加密长 run 频率 > GBM（P(run≥k) 高于 GBM 种子散布 2σ，主判据 k=8，k∈{5,10} 支持性）；H2：传统市场长 run 频率 > GBM（同判据，书在其语境成立）；H3：加密 vs 传统 P(run≥k) 差 > 合并 2σ（加密特殊性）。run 断裂规则：相邻 bar 间隔 > 2×标准 bar 间隔则断 run（对照组有隔夜/周末缺口）。**学习级**：GBM 30 种子、无 BY_YEAR、MIN_N=100、描述层、结论标 [学习级] | **已完成（2026-08-13，[学习级]）**：H1 不支持且**反向**（加密薄尾：合并 z8=−9.45，BTC 1h −6.05 / 4h −4.18、ETH 1h −6.02 / 4h −1.67——连涨连跌比抛硬币还少）；H2 不支持（传统合并 z8=−4.84：EURUSD −6.31、GC=F −0.25 持平、SPY k=8 结构性不可测——会话 ~10bar/日）；H3 不支持（加密 P8 0.018 vs 传统 0.018，无特殊性——两市场同被符号反持久主导）。书"运行肥尾=趋势利润唯一来源"在两市场 1h 均证伪；日线口径留待后续新预注册（control.db 已有 1d 数据） |

**阶段 2 进展记录（2026-08-13）**：c23 未达（1:1 +0.27pp，c17 端点效应不转移；角色层发现保留）；c21 未达（ΔWR 全过但 ΔE 未达 +0.05R，GBM 填充偏置吸收一半；成本后为正）；c25 未达（状态增量 −0.18pp≈0，净差全为无条件漂移；1:1 成本结构性不可达）；c22 未达（hold 引擎 GBM 机械期望 +0.54R，真实均值回归净差 −0.14R；C_share 分层增量 +0.10R 但绝对仍负）。**Phase 2 四个预注册候选全部未达发布门槛——按 exit criteria 归档全部候选，不进入 Phase 3。**共同教训：①端点方向 ≠ 1:1 先碰命中（c18 的 E1 效应在 1:1 下消失）；②一切条件化结构在 GBM 上都有机械偏置（触碰填充 +1pp、hold+trail +0.54R），净差化是唯一防线；③1h/4h 尺度下真实市场的均值回归主导（趋势尾部、逆势折返都不足以覆盖 taker 成本）；④描述层事实（围墙/波动释放/留存/脆弱性）真实存在，但至今未找到把它们转化为正期望入场的方式——**"没有 edge 也是一种结论"**。

**Phase2 发布门槛清单（10 项，缺一不可）**：①真实−RW(30 种子) 超预注册下限；②分年 ≥2/3 为正、最差年 ≥-2pp；③每格 n≥MIN_N；④GATE 条件组无偏；⑤跨周期+跨参数一致；⑥holdout（末 3 月，参数冻结后一次评估）方向不变；⑦Holm 校正后仍显著；⑧成本核算后 >0；⑨结论↔.out↔脚本三重一致；⑩负结果/未达标格全部记录。

### 阶段 3：策略层 c3x

**c31 双层验证**：口径层（事件级 edge 是否存在）→ vectorbt 全回测层（可交易性：重叠持仓、资金管理、成本逐项注入，每步衰减可解释）。一致性验证：(a) 零成本单开无重叠下 vbt 与口径层逐笔对拍；(b) R 分布两层差异 ≤ 预注册容差（vbt 已知坑由测试锁定，不许"修"）；(c) 成本衰减曲线报告，单项成本吃掉 >50% 净差即单独立项。

**c32 live 试点**：影子模式（只推送不成交）观察 ≥30 信号事件 且 ≥4 周。6 项检查：事件频率偏差≤±30%、历史重放信号重合率≥95%、特征/参数/时间戳语义逐项对拍、limit 成交率偏差≤±10pp、事件延迟≤1 bar、影子 R 序列落在预回归 95% 区间。最小仓位：每笔风险 0.1~0.5% 净值，连亏 5R 退回影子，每周对账。

### 阶段 1 完成记录（2026-08-13，8 项全部归档于 research/notes/）

| 编号 | 结论 | prior 裁决 |
|---|---|---|
| c12 波动长记忆+状态转移 | H1 长记忆 H=0.93/0.90 确认（GBM null 0.50）；H3 必经中确认 | H4 持续期 prior 否定（16/13 根，非 50~60） |
| c14 位移约束 | H1/H2 全 16 格过（ΔP50 -0.63~-1.06 ATR）；H3 泄漏敏感度 0% | 确认 |
| c15 波动释放 | H1 净差 +6.3~+7.4pp 4 组合；H2 年龄单调；H3 GBM 机械偏置 +1.04pp 在预算内 | 确认 |
| c16 区间延续 | H1 触碰留存 +4~+7pp；H3 存续 +2.6~+4.4 根；H4 方向无特异性 | H2 全体 bar 留存否定（≈0，B3d 的 +8~12pp 是宽度约束效应） |
| c13 收益偏度 | H1 up:late 全过；H3 3/4 单调 | H1 dn:late 4h 未达；H2 top5% 方向占比为机械产物（GBM 亦 ~89%） |
| c17 逆势折返 | H1 -3.6~-4.1pp 4 组合（强于 prior） | H2 阶段无梯度否定（early≈0，accel/late 强）；H3 角色层携带信息（未破更强） |
| c18 方向 null+触位 1:1 | c18 触位无增益成立（相对无条件漂移） | **c11 方向 null 推翻**：trend_up 状态进入 E1 -3~-4.6pp（状态条件化携带真实方向信息）；4h 效应大部分=环境漂移 |
| c19 道氏生命周期 | H1 存活 +4.33pp；H2 恢复率 -12.35pp；H3 热段 -17.45pp | a6 系四组修复全部未改变净差符号——"真实趋势更脆弱"因果口径下成立 |

**Phase1→2 出口判定**：8 项完成且 GATE 全过 ✅；核心 prior 确认 ≥6（c12/c14/c15/c16/c17/c19）✅；c1a 按默认砍 ✅。**进入 Phase 2 的条件已满足。**

对 Phase 2 的影响：c23 前提强化（c17+c18 双支撑）；c22 前提弱化（c13 部分）；c21/c24 前提齐备（c14/c15/c16/c12）；新增候选方向=状态条件化方向倾向（c18 c11 发现，需 amend 预注册后才能进 c2x）。

### Exit criteria

| 跃迁 | 条件 |
|---|---|
| Phase1→2 | ≥8 项完成且 GATE 全过；核心 prior 确认 ≥6 条（**c14/c15/c16 至少 2 条 + c12 必过**）；未确认 prior 按失败归档写明原因；若 c14-c17 全部不确认 → 停 level 线，Phase2 只留 c22/c24 |
| Phase2→3 | ≥1 候选过全部 10 项门槛，且机制与 Phase1 事实自洽；0 个通过 → 归档全部候选，返回 Phase1 扩展，明示"没有 edge 也是结论" |
| Phase3→实盘 | 影子 6 项全过 + 成本后期望>0 + holdout 未衰减 + 用户确认仓位/止损/暂停规则 |

## 5. 资源预算

| 阶段 | 估算 |
|---|---|
| Phase0 加固 | 3-5 人日（引擎断言→causal/ctx/limit_sim→levels 重写→check_study→三反例回归演示） |
| Phase1 | 8-10 项 × (pytest 4min + 运行 10-90min)；level 系 30-60min/项；**10-20 机器时，4-5 人日** |
| Phase2 | 4 家族 × (开发+验证+30 种子+holdout+成本) ≈ 3-6h/家族；**15-25 机器时，4-6 人日** |
| Phase3 | 全回测 1-3h/配置；影子模式 4 周墙钟；3-5 人日 |
| 数据 | backtest.db 够 Phase1/2 主体；建议补 4 年历史（1 天下载）+ funding 历史（Phase2 成本必需） |

## 6. 风险与反模式防线

| 反模式（历史教训） | C 系列防线 |
|---|---|
| 未来函数（整批作废之根） | 已收盘 bar；MTF align_higher；在线聚类+冻结；特征过 invariance；禁 center 窗口/shift(-1) |
| confirmed 泄漏（B 系列贯穿缺陷） | 条件化一律 causal_confirmed（conf∈[t-60,t-24]）；c14 H3 泄漏敏感性对照实验 |
| 全样本分位（B4e） | 一律 rolling_percentile；全样本统计量仅限事后标签且禁参与条件化 |
| post-hoc（B4 结论对应不存在的运行） | docstring 预注册冻结；三重一致为提交前置；脚本改动=重跑+结论重写 |
| 选择性报告 | 全单元格输出；负结果同等归档；"发现"措辞必须过门槛清单 |
| 样本内过拟合 | 参数网格预注册+开发/验证分离；GBM 固定种子序列（禁换种子刷结果）；holdout 一次评估；跨周期一致强制 |
| GBM 种子不足（a38 6 种子） | MIN_GBM_SEEDS=30，GATE 自动检查 |
| 引擎语义坑（自写 _sim、vbt 伪影） | 禁自写 outcome；非对称走官方扩展；vbt 坑测试锁定不许修 |
| 成本幻觉（B4 净差 0.07R < 成本 0.3-0.5R） | Phase2 结论附成本行；统一成本模型；成本后≤0 标"结构发现" |
| 文档不一致（LIVE_SYSTEM 旧段引用作废研究） | 文档三级状态标注（已确认/待重验/作废）；修正声明连带更新所有引用处 |
| 复现性（levels.py 未提交、B5c 修复未落地） | 每研究一个 commit（脚本+notes+.out）；禁止从脏工作树运行 |
| 方向参照偏置（位带中心参照 GBM 63%≠50%） | 方向度量一律以 close[t] 为参照；对数化防 Jensen；GATE 无条件 ≈50% 强制 |

## 7. 归档与迁移方案

### git 操作序列（按序，勿合并）

```bash
# 1. levels.py 未提交改动存档（B2+ 依赖的 batch helpers 研究全作废 → 存档不留用）
mkdir -p research/archive/patches
git diff research/levels.py > research/archive/patches/levels_batch_helpers_uncommitted.patch
git checkout -- research/levels.py          # 恢复 HEAD（live 依赖的稳定版）

# 2. 归档（保留历史，勿删）
git mv research/studies research/archive/studies
git mv research/notes   research/archive/notes

# 3. 重建空目录 + archive/README.md 作废公告

# 4. 确认 live 不受影响后提交
grep -rn "research.studies\|research.notes" --include="*.py" .   # 应只剩 archive 内部互引
git commit -m "research: archive all A/B studies (contaminated) + restart as C series"
```

注意：`research/levels.py` **不能归档**（live 依赖）；archive 内部互 import 断裂可接受（旧研究不再运行，tests/ 无 studies 依赖已核实）。

### 文档处置

| 文件 | 处置 |
|---|---|
| docs/SESSION_SUMMARY.md、docs/MARKET_STATE_STRATEGY_MAP.md、archive/notes/phase_summary.md、CONTEXT_RESTORE.md | 不删，头部加作废戳：研究结论基于 2026-08-03~05 批次，2026-08-12 整体作废，数字不可采信；有效结论唯一来源 = research/notes/（C 系列） |
| docs/LIVE_SYSTEM.md | 运行规格（端口/启动/推送时间窗）保留；SCENE_WR 等研究数字段加戳 |
| AGENTS.md | "回测与研究"整节重写：C 系列协议（编号/模板/两道门禁 `pytest research/tests -q` + `python3 research/check_study.py`/发布门槛/archive 制度/levels patch 存档与 live 依赖说明） |

## 8. 执行顺序（最小步骤 + 验收）

1. **归档**（git 序列 §7）→ 验收：git status 干净、live 冒烟不报错、旧测试全绿、无残留 import。
2. **L1 引擎加固** → 验收：新测试绿、62 用例不回归、用 A3 旧脚本演示断言触发。
3. **L2 模块**（causal/ctx/limit_sim + 测试 → levels R1/R2 重写 + test_levels_batch → test_dow 含 amp NaN 修复）→ 验收：全绿（预计 ~75 用例）、live 冒烟 key_levels 输出不变。
4. **L3 门禁**（check_study 七类检查 + 三模板）→ 验收：A3/B4e/B5c 三反例回归演示，输出"FAIL+原因+行号"。
5. **C1 试点**：c12（波动记忆+状态转移）——描述层无入场、风险最低，dogfood 全模板 → 验收：check_study 过、GATE 50%±1pp、结论数字全可溯源。
6. **分层推进**：c1x 队列 → c2x 候选 → c3x 全回测+试点，每层按 exit criteria 裁决。

---

**一句话总纲**：门禁从"测引擎"升级为"测模块+验脚本"三层；Phase1 用因果口径给旧结论验尸（确认 6 条即过关），Phase2 只允许 4 个预注册候选走完整门槛（过不了就是没有），Phase3 口径层与 vectorbt 层、回测与 live 双重对拍——宁可收敛到"关键位=围墙、无方向 edge、尾部收割待成本验证"的诚实结论，也不许再产出一个 B4。
