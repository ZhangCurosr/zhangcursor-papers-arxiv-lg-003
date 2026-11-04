---
title: "Zero-shot-time-series-foundation-models-against-the-Khepri-v"
source: https://arxiv.org/pdf/2611.02895v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:46:13"
field: "时间序列预测评估方法"
keywords: ["时间序列基础模型", "零样本预测", "碳排放强度", "预注册评估", "污染控制", "Diebold-Mariano检验"]
innovations: ["回测+后发布holdout双窗口污染控制设计", "MDE前置锁定与not-determined判定规则", "OpenTimestamps完整审计链与deviation log机制"]
benchmarks: ["Khepri v1.5 Nordic Bidding Zones", "CarbonX 214 grids"]
---

# 论文速读：Zero-shot-time-series-foundation-models-against-the-Khepri-v

## 一句话总结
本研究以前瞻性注册方式，将三个零样本时间序列基础模型（Timestral-5M、Timestral-209M、Chronos-2）与 Khepri v1.5 的 SARIMA 和 GBM 基线进行对比，用于北欧九个竞价区域 96 小时碳排放强度预测；回测结果显示所有 TSFM 均显著优于基线，但小规模后发布 holdout（14 个起源日）因统计功效不足而判定为"未确定"，且未发现污染证据。

## 研究问题与动机
- **碳排放强度预测的决策价值**：精确的日内至数日级 CI 预测可引导灵活用电时段向低碳时段转移，是碳感知调度系统的核心组件。
- **现有基线的局限与比较对象**：Khepri v1.5 已发布 SARIMA 与 GBM 基线，但两者为有监督/局部拟合方法，需要逐区域训练；TSFM 的零样本能力使其可直接复用，但缺乏严格对比。
- **污染问题（contamination）**：TSFM 的预训练语料未公开披露，测试数据可能进入预训练集，导致回测高估真实泛化性能，亟需一种污染控制设计。
- **评估透明度不足**：多数 TSFM 评测为静态历史回溯，缺少事前注册、锁定分析代码与事后验证机制，本工作引入 OpenTimestamps + GPG 签名的前瞻性注册来规范评估流程。

## 核心贡献（创新点）
1. **首个在北欧 9 个竞价区域上对零样本 TSFM 与 Khepri v1.5 基线的前瞻性注册对比**：研究设计、分析代码和最小可检测效应（MDE）在所有 TSFM 预测之前已通过 Bitcoin-confirmed OpenTimestamps 时间戳固化。
2. **回测 + 后发布 holdout 的双测试设计以控制污染**：361 个起源日的回测提供统计功效，14 个起源日的 holdout（仅在 Timestral 发布次日之后）作为污染控制，MDE 在预测前锁定，差值小于 MDE 一律判为"未确定"而非等效。
3. **污染指示器（contamination indicator）的定量构造**：以回测优势减去 holdout 优势，配合 HAC 标准误计算 z 统计量，无一项超过阈值 1.96，表明未发现显著污染信号。
4. **SARIMA 概率预测的实用构造方法**：针对 Khepri v1.5 原生输出不含分位数的限制，从拟合模型的 Gaussian 预测分布中推导 21 个分位数，使 SARIMA 可与 TSFM 在 CRPS 上进行公平比较。
5. **完整的可复现证据链**：所有预测、CI 系列、锁定分析代码及每项偏差均以 sha256 和 OpenTimestamps 存档于 Zenodo（DOI 10.5281/zenodo.23881315），并附有日期标注的 deviation log。

## 方法详解
- **数据与 CI 系列**：使用 ENTSO-E A75 每小时发电类型数据，按 Khepri v1.5 规则（AR5 生命周期排放因子、ADR-0002 物质性阈值、抽水蓄能时段设为缺失、Waste/Other 排除、 hourly mean、gap filling 插值）计算 gCO2-eq/kWh 的 CI 系列；2026 年新增数据于 2026-11-02 通过 entsoe-py 0.8.0 获取。
- **预测起源与窗口**：每日 00:00 UTC 起源，96 小时预测 horizon；回测窗口为 2025 全年（361 个起源日/区域），holdout 窗口为 2026-10-16 至 2026-10-29（14 个起源日/区域）。
- **预测臂（arms）**：Seasonal naive（周）、SARIMA（Khepri v1.5 不变：SARIMAX(2,0,1)(1,1,0,24)）、GBM（LightGBM，200 棵树，31 叶，lr=0.05）、Timestral-5M、Timestral-209M、Chronos-2；所有模型均使用相同 8192 步历史上下文，无外生协变量。
- **损失函数**：主评估指标为 Khepri v1.5 MAPE（96 小时平均绝对百分比误差，|actual|≤1e-6 的时段排除）；概率校准用 CRPS 的 21 分位数近似：$\text{CRPS} \approx \frac{2}{21}\sum_k \text{pinball}_{\tau_k}(y,q_k)$，以及 80% 和 90% 中心区间的经验覆盖率。
- **统计检验**：Diebold–Mariano 检验 + Harvey–Leybourne–Newbold 小样本修正，h=4 处理每日起源重叠的 96 小时预测；长程方差使用矩形核（lags 0..3），辅以 Bartlett 权重（若不正定）；Holm 校正 p 值作为次要保护。
- **MDE 计算**：基于 SARIMA−GBM 在 2025 测试上的损失差序列，$\sigma^2_{\text{LR}}=7.6455$ pp²，n=14 时 MDE=2.989 pp，n=361 时 MDE=0.413 pp；holdout 中 $|\text{mean } d|<\text{MDE}$ 一律判为"not determined"。
- **污染指示器**：$A_t = \text{loss}(\text{baseline})_t - \text{loss}(\text{TSFM})_t$，指示器 = $\bar{A}_{\text{retro}} - \bar{A}_{\text{holdout}}$，z = 指示器 / SE；z > 1.96 标记为"可能污染"。
- **敏感性分析 S1**：排除抽水蓄能填补缺失较多的区域（NO2、NO3、NO5），仅对 NO1、NO4、SE1–SE4 六个区域重复 DM 检验。

## 实验与结果
- **数据集与区域**：北欧 9 个竞价区域 NO1–NO5 和 SE1–SE4，ENTSO-E A75 数据，2019–2026。
- **基线**：Khepri v1.5 SARIMA 和 GBM（代码与超参完全不变），Seasonal naive（周）参考臂。
- **回测结果（Table 1，361 起源日，Zone-averaged）**：
  - Timestral-209M vs SARIMA：MAPE 6.213% vs 7.561%，diff = −1.347 pp，HLN = −12.77，p < 0.0001（最强提升）
  - Timestral-5M vs SARIMA：diff = −1.308 pp，HLN = −13.30，p < 0.0001
  - Chronos-2 vs GBM：diff = −0.952 pp，HLN = −7.79，p < 0.0001
  - 所有 TSFM 在 9/9 区域均击败 SARIMA，H1 假设得到支持。
- **Holdout 结果（Table 3，14 起源日，MDE = 2.989 pp）**：
  - 所有 TSFM 的点估计仍优于基线（diff 范围 −0.729 至 −1.326 pp），但全部落在 MDE 范围内，故一律判为"not determined"；四项 p < 0.05，但因功效不足不能下结论。
- **校准结果（Table 4，Holdout）**：
  - SARIMA CRPS = 1.4165，Timestral-209M CRPS = 1.0733（最优），Chronos-2 CRPS = 1.1198，Timestral-5M CRPS = 1.1114；所有 TSFM vs SARIMA 的 DM 检验 p < 0.0001。
  - 80% 覆盖率：TSFM 0.806–0.838，SARIMA 0.835；90% 覆盖率：TSFM 0.905–0.921，SARIMA 0.911。
- **污染指示器（RQ5）**：z 值介于 −0.68 至 1.75 之间，无任何比较超过 1.96 阈值，未发现污染证据。
- **NO4/NO5 专项（RQ3）**：NO4 中 GBM 在 v1.5 中劣于 SARIMA（+2.92 pp），TSFM 在该区域仍为最优（Timestral-209M 6.133% vs SARIMA 7.242%）；NO5 holdout 中 CI 恒为 24.0 gCO2-eq/kWh，除 GBM 外所有臂 MAPE 为 0.000%，此为该窗口数据特性而非模型能力差异。

## 相关工作脉络
- **DACF（Maji et al. 2022a）**：用独立神经网络预测各发电源产量再结合排放因子，区域仅含瑞典（国家级别）和德国，非竞价区级别，与本研究的零样本 TSFM 路线不同。
- **CarbonCast（Maji et al. 2022b）**：扩展至 96 小时，但同样只覆盖瑞典国家级别，且未在模型发布后进行前瞻性测试。
- **EnsembleCI（Yan et al. 2025）**：堆叠 LightGBM/CatBoost/XGBoost/神经网络集成，区域限于瑞典国家，与本研究竞价区粒度和零样本范式无直接可比性。
- **CarbonX（Maji et al. 2025）**：在 214 个电网上评测五种 TSFM 做 CI 零样本预测，但评估全为历史数据且未讨论污染问题，挪威和瑞典均仅以国家粒度出现；本文将其推进到竞价区级并引入污染控制。
- **Impermanent（Garza et al. 2026）**：GitHub 活动序列的实时预测基准，核心动机同为"训练数据可能与评估数据重叠"，但聚焦社交信号而非能源领域；本文的 holdout 设计受其启发但不构成 live benchmark。
- **Khepri v1.5（Botten Nicolaysen 2026）**：本研究的基础对比平台，提供 SARIMA/GBM 基线及 CI 计算管线，本文保持其代码与超参完全不变以确保可比性。

## 局限性与未来方向
- **回测污染无法排除**：Timestral 未公开训练数据与 cut-off，Chronos-2 虽列出预训练数据集但不含 ENTSO-E，Nordic 数据仍可能通过聚合数据集渗入，回测优势可能被高估。
- **Holdout 统计功效极低**：仅 14 个起源日，MDE 高达 2.989 pp，即使观测到与回测同向的优势也无法做出确定性判断。
- **NO5 区域在 holdout 中信息量极有限**：该区域 CI 在整个 holdout 窗口恒为 24.0，几乎所有臂 MAPE=0，削弱了跨区域比较的有效性。
- **SARIMA 分位数为近似推导**：从高斯预测分布构造而非原生概率预测，CRPS 差异可能部分反映 SARIMA 区间质量而非 TSFM 本身优势。
- **部分设置暂缓**：32768 步上下文、Timestral-28M/123M、协变量输入等均推迟至 v2，当前版本仅使用 8192 步上下文。

## 研究启发与可借鉴点
- **双窗口设计（回测 + 后发布 holdout）可作为 TSFM 评估的标准范式**：高功效回测承载统计推断，短 holdout 承担污染控制，两者结合兼顾发现力与可信度，适用于任何存在预训练数据重叠风险的模型评测场景。
- **MDE 前置锁定 + "not determined"判定规则**：将最小可检测效应预先计算并写入注册文件，比传统 p 值报告更能避免小样本过解读，值得在数据受限的前瞻性评估中推广。
- **OpenTimestamps + GPG 签名 + Deviation Log 的完整审计链**：从注册到代码变更的全部操作均有时间戳和签名，为竞争激烈的模型评测领域提供了可复制的透明度模板。
- **高斯分布推导分位数的通用技巧**：对于缺少原生概率输出的基线模型，可从拟合分布中构造分位数以支持与深度模型在 CRPS 上的公平比较，降低了评估门槛。
- **与团队方向的结合机会**：本研究采用的前瞻性注册管线与污染检测框架可直接迁移至团队在能源/气象等领域的时间序列预测工作，尤其适合需要与既有基线做严格对比的场景。

## 关键术语表
- **Zero-shot TSFM（零样本时间序列基础模型）**：在大规模多域时间序列上预训练、可直接应用于新序列预测而无需目标序列微调的模型，如 Timestral 和 Chronos-2。
- **Carbon Intensity（CI，碳排放强度）**：单位电量（gCO2-eq/kWh）的温室气体当量排放，反映电网实时碳密集程度。
- **Khepri v1.5**：作者开源的北欧 CI 预测基准系统，提供 SARIMA 与 GBM 基线及完整的 CI 计算管线。
- **Minimum Detectable Effect（MDE，最小可检测效应）**：在给定样本量和统计功效（此处 80%）下，检验能够识别的最小效应量，此处 holdout 为 2.989 pp。
- **Diebold–Mariano–HLN 检验**：比较两个预测模型损失差异显著性的检验，Harvey–Leybourne–Newbold 修正针对小样本和重叠预测窗口。
- **Contamination（数据污染）**：测试数据意外进入模型预训练集，导致静态回测高估真实泛化性能的问题。
- **HAC 标准误**：Newey–West 类型的异方差自相关稳健标准误，使用矩形核（lags 0..3）估计重叠预测序列的长程方差。
- **CRPS（连续排序概率得分）**：衡量概率预测校准质量的严格评分规则，此处用 21 个分位数的 pinball loss 近似。

## 可复现要素
- **数据集**：ENTSO-E Transparency Platform A75 数据；Khepri v1.5 原始数据文件未直接分发，sha256 经 OpenTimestamps 签名，派生 CI 系列已发布（Zenodo，parquet 格式）。
- **代码**：已开源（GitHub: github.com/avalyset/khepri-tsfm，Zenodo DOI: 10.5281/zenodo.23881315），Apache-2.0 许可。
- **模型权重**：Timestral 权重以 Modified MIT License 单独发布（含收入限制条款）；Chronos-2 以 Apache-2.0 发布于 Hugging Face（amazon/chronos-2）。
- **关键超参**：Timestral-5M/209M 使用库默认配置；Chronos-2 使用 predict_quantiles、prediction_length=96；SARIMA(2,0,1)(1,1,0,24)，enforce_stationarity=False，maxiter=50；GBM LightGBM 200 树/31 叶/lr=0.05/3 seeds。
- **计算环境**：Python 3.12.13，uv.lock 锁定依赖，CPU（Apple M1，16 GB），torch.set_num_threads(8)，HF hub offline 模式。
