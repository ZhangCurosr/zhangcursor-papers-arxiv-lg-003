---
title: "Zero-shot-time-series-foundation-models-against-the-Khepri-v"
source: https://arxiv.org/pdf/2611.02895v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:46:09"
---

# 论文速读：Zero-shot-time-series-foundation-models-against-the-Khepri-v

## 一句话总结
本文预先注册并对比了三种零样本时间序列基础模型（Timestral-5M/209M、Chronos-2）与Khepri v1.5基线（SARIMA、GBM），在北欧9个竞价区进行96小时逐小时碳排放强度预测；通过历史回溯测试保证统计功效、发布后预留样本控制污染风险，证明零样本TSFMs在点预测精度与概率校准上均显著优于传统基线。

## 研究问题与动机
- **核心问题**：零样本TSFMs能否在Nordic竞价区的96小时碳排放强度（CI）预测中超越已发表的SARIMA与GBM基线？
- **污染风险未知**：TSFMs的预训练语料未公开训练截止期，静态历史基准测试可能因数据重叠而高估实际泛化能力。
- **区域粒度缺失**：现有CI预测工作（DACF、CarbonCast、EnsembleCI、CarbonX）多仅覆盖国家级别（如Sweden whole country），缺乏NO1–NO5与SE1–SE4等竞价区级别的精细评估。
- **评估透明度不足**：多数TSFM对比研究未在预测生成前锁定分析流程，缺乏可复现的污染控制设计与决策阈值。

## 核心贡献（创新点）
- **预先注册的全流程锁定机制**：研究设计、分析代码与MDE均在首次TSFM预测前通过OpenTimestamps（Bitcoin确认）与GPG签名固化，与多数“先跑模型后写报告”的TSFM评估本质不同。
- **双窗口污染控制架构**：以高功效历史测试（361起源）承载统计检验，以低功效发布后预留测试（14起源）专门监控预训练数据渗漏，突破了静态基准仅能报告单点指标的传统范式。
- **基线代码冻结与驱动封装**：将Khepri v1.5原样字节级复用，仅通过外部driver保留每小时预测结果，确保TSFM与传统模型在完全一致的上下文、数据管道与评估函数下公平对比。
- **开放前瞻性每日预测日志**：自注册日起每日自动生成并加盖时间戳的预测日志，为Foundation Model的持续无偏验证提供了可累积的数据基座，区别于一次性离线评测。

## 方法详解
- **数据与CI生成**：使用ENTSO-E A75实际发电数据，按Khepri v1.5流程计算9区逐小时CI（gCO2-eq/kWh），包含AR5生命周期排放因子、ADR-0002显著性阈值、抽水蓄能缺失值标记、`interpolate(limit=6).ffill().bfill()`填充，且填充后序列同时作为模型输入与真实标签。
- **预测臂设定**：全部为单变量无协变量设置。基线SARIMA为`SARIMAX(2,0,1)(1,1,0,24)`（`enforce_stationarity=False`），GBM为LightGBM（200 estimators、31 leaves、lr=0.05、3 seed平均、特征含星期/月份/当前CI/24阶滞后/ horizon）；TSFMs统一取最近8192小时上下文，点预测取0.5分位数，TSFM原生输出21个分位数（0.01~0.99），SARIMA通过其高斯预测分布构造分位数。
- **双测试窗口设计**：
  - **回溯测试（Retro）**：2025年361个日起源，统计功效高，但标注“contamination not ruled out”。
  - **预留测试（Holdout）**：2026-10-16起14个起源（均在Timestral/Chronos-2公开发布之后），作为污染控制，MDE预设为2.989 pp。
- **统计检验与决策规则**：MAPE差异采用带Harvey–Leybourne–Newbold校正的Diebold–Mariano检验（h=4，矩形核lag 0~3，HAC长程方差）；CRPS采用21分位数pinball score近似；若`|mean d| < MDE`则统一判为“not determined”，绝不解读为等价；污染指示器为`Advantage_retro − Advantage_holdout`，z>1.96标记为“possible contamination”。
- **敏感性分析**：剔除有抽水蓄能填充的5区，仅在NO1、NO4、SE1–SE4共6个无填充区执行S1检验，区组选择在设计阶段固定。

## 实验与结果
- **数据集与基线**：北欧9竞价区CI序列；基线为Khepri v1.5原样复现的SARIMA与GBM；TSFMs为Timest
