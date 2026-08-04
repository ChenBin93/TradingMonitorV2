# 业界/学术调研：关键水平位 (Support & Resistance)

> 2026-08-04 调研，为 B3 泛化研究提供假设来源与方法参照。
> 仅作研究参考；标注"存疑"的结论不采信，需用无未来函数方法重验。

## 一、学术研究（性质与机制）

### 1. Osler (2000) "Support for Resistance: Technical Analysis and Intraday Exchange Rates"
- 来源: NY Fed Economic Policy Review; 数据 = 6 家外汇交易公司每日发布的 S/R 位 + 1 分钟 Reuters 报价
- 发现:
  - 价格触碰发布位后**反弹概率显著高于随机生成的位**（bounce 效应）
  - 预测力持续最多 5 个交易日
  - **预测力随公司与汇率而异 → 位质量存在异质性**（支持质量分层研究）
- 与我们对照: 用"随机位"对照与我们 GBM 同流程对照思路一致; 位质量异质性 → B4 质量分的合理性

### 2. Osler (2003) "Currency Orders and Exchange Rate Dynamics" (Journal of Finance 58(5))
- 机制研究（首批个人止损/止盈订单数据）:
  - **止损单与止盈单聚集在 S/R 位附近（尤其整位 round numbers）**
  - 聚集解释两条技术分析预测: ①趋势在 S/R 处反转（止盈单集群被触发 → 反方向动能）
    ②趋势穿越 S/R 后异常快速移动（**止损单触发 → 止损推动价格**）
- 对我们研究的对应:
  - **为 B2b E1"触碰后波动释放 +5pp"提供机制解释**（位附近订单聚集 → 触碰触发止损/止盈 → 波动放大）
  - 提示突破后短期加速在 FX 存在; 但我们在 crypto 24 根窗口测到"真突破后无延续"(B2 E4)
    → 差异待解: crypto 无订单簿聚集 / 时间尺度不同 / 突破定义不同（候选后续研究，需 5m 数据测分钟级加速）

### 3. Chung & Bellotti (2021) "Evidence and Behaviour of Support and Resistance Levels in Financial Time Series" (arXiv:2101.07410)
- 启发式 SR 检测算法 + 行为检验:
  - **反弹次数越多的位 → 再次反弹概率越高（触次单调增强有效性）**
  - **SR 有效性随年龄衰减（decay）**
- 对我们研究的对应:
  - 直接支持 B3 触次层与年龄层设计假设
  - 与我们用户经验"波段高低点 + 多次触碰"一致
  - 注意: 该方法学与基线构造细节未审（可能有未来函数），只取假设方向，不采信数字

### 4. Garzarelli, Cristelli, Zaccaria, Pietronero (2011) "Memory effects in stock price dynamics: evidences of technical trading" (arXiv:1110.5197)
- 提出 S/R 检测准则; 价格更可能**回弹而非穿越** S/R 位（自证预言 self-fulfilling prophecy）
- 注意: 方法细节存疑，仅概念参考; 我们 B 系列用 GBM 对照检验同一问题（B2d: 回弹倾向非特异，
  约束/留存才是特异性）

### 5. Lo, Mamaysky, Wang (2000) "Foundations of Technical Analysis" (Journal of Finance)
- kernel 回归平滑识别形态（含 S/R 类形态）; 道指上形态具有统计意义但经济显著性弱
- 仅背景参考

### 6. DeepSupp (2025) "Attention-Driven ... SR Levels Identification" (arXiv:2507.01971)
- 深度学习（multi-head attention + 特征工程 + DBSCAN 聚类提取价位）检测 S&P 500 SR 位
- 业界趋势: ML 替代手工画线; 与我们"在线聚类+冻结"思路同源; 无严谨对照基线（仅 6 基线方法对比），
  结论不采信

## 二、业界画法谱系（如何准确画关键位）

| 方法 | 说明 | 我们的对应/可行性 |
|------|------|-----------------|
| Swing 高低点 + 右移确认 | 波段高低点需右侧 K 根确认 | ✓ cluster_levels 的 pivot（K=3 同款） |
| 聚类成带 | 多个价位聚成带（非线）; ATR 比例带宽 | ✓ cluster_levels（tolerance×ATR 聚类） |
| 触碰确认 | 2 触候选 / 3 触确认（"第三次测试"） | ✓ 触次分层（B3）; 触次增强有文献支撑 |
| 圆整数位（心理价位） | 订单聚集于整位（Osler 2003 实证）; 独立于图表形态 | **未测** — B 系列后续候选: 整位 vs 聚类位触碰行为对比 |
| Volume Profile / POC | 成交量分布密集区 | 需 tick 数据，暂不可行 |
| 枢轴点 (Pivot Points) | (H+L+C)/3 数学位 | 衍生品从业者常用，实证证据弱，不优先 |
| 斐波那契回撤/扩展 | 比例位 | 实证证据弱，不优先 |
| VWAP/动态位 | 日内动态 S/R | 动态位，与水平位范畴不同 |
| 多周期共振 | 多周期同位更强 | 有 A3.6 MTF 基建，可扩展（B 系列后续候选） |

## 三、对 B3 及其后的直接影响

1. **B3 分层有文献背书**: 触次层（Chung-Bellotti 触次增强）、年龄层（同文献衰减）
2. **位质量异质性**（Osler 2000）→ B4 质量分研究（年龄/触次/结构 × 约束强度）合理
3. **E1 波动释放的机制假设**: 位附近订单聚集 → 触碰触发止损/止盈 → 波动放大（Osler 2003）
4. **后续候选**:
   - 圆整数位 vs 聚类位（crypto 整位效应检验，5m/1h 均可）
   - 突破后分钟级加速（止损推动机制 vs B2 E4 24 根无延续）
   - 多周期共振关键位（MTF 重叠 × 约束强度）
