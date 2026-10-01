---
title: "QIYAO-M-MULTIMODAL-TIME-SERIES-FOUNDATIONMODEL-WITH-ROLE-AWA"
source: https://arxiv.org/pdf/2609.34842v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:55:09"
field: "多模态时间序列预测"
keywords: ["time series forecasting", "multimodal foundation model", "endogenous/exogenous modality", "retrieval enhancement", "role-aware modeling"]
innovations: ["角色感知建模：区分内生/外生模态的预测角色并分别设计演化建模与检索增强策略", "内生模态显式演化监督：通过预测未来内生表示并监督对齐实现时序演化学习", "代理训练解决外生数据稀缺：用内生模态动态替代外生模态训练检索增强器"]
benchmarks: ["GIFT-Eval", "TIME", "Time-MMD", "MoTime", "FinMultiTime"]
---

# 论文速读：QIYAO-M: MULTIMODAL TIME SERIES FOUNDATION MODEL WITH ROLE-AWARE MODELING OF ENDOGENOUS AND EXOGENOUS MODALITIES

## 一句话总结
本文提出 QiYao-M，一种角色感知的多模态时间序列基础模型（TSFM），通过区分内生模态（endogenous）与外生模态（exogenous）的预测角色，分别采用显式演化建模与检索增强策略，显著提升跨域多模态时间序列预测性能。

## 研究问题与动机
1. **现有方法忽视模态角色差异**：当前多模态 TSFM 对内生/外生模态采用 largely shared mechanisms 建模，未考虑二者在预测中的本质差异——内生模态与时间序列内在耦合，外生模态来自外部且跨域/跨模态类型存在异质性。
2. **内生模态演化缺乏显式监督**：数值预测损失仅评估最终预测值，无法提供模态级别的演化监督信号；现有方法仅利用内生信息丰富历史上下文，未显式学习其从历史到未来的演化过程。
3. **外生模态预训练数据稀缺**：相同外生模态在不同域中预测效果不同（如同样的卫星云图在气象预测中表示降雨增加，在交通预测中却降低流量），且预测场景中可用的外生模态类型/数量差异大，难以学习可迁移的外生知识。

## 核心贡献（创新点）
1. **提出角色感知多模态 TSFM 框架**：首次根据预测角色差异分别建模内生/外生模态，区别于现有 role-agnostic 的共享机制设计。
2. **Endo-Multimodal Predictor + Endo-Multimodal Supervision**：通过预测未来内生模态表示并监督其与 ground-truth 演化对齐，实现内生信息的显式时序演化建模，而非仅作为历史上下文增强。
3. **Exo-Multimodal Retrieval Enhancer + Endo-Modality Proxy Training**：检索增强模块在推理时无需更新 TSFM 参数即可适应未知域与任意模态组合；通过内生模态代理训练解决外生预训练数据稀缺问题，区别于依赖大量外生数据的端到端预训练方案。

## 方法详解
**整体架构**：输入序列经归一化与 patch 化后，依次通过 Endo-Multimodal Fusion（patch 级融合）、Transformer 骨干、可选的 Exo-Multimodal Retrieval Enhancer、Endo-Multimodal Predictor（输出数值+内生模态表示）。

**1. Endo-Multimodal Fusion**
- **内生模态构造**：每个 patch 生成两种内生模态——Endo-Text（统计特征描述，如趋势、波动率、事件等确定性模板）和 Endo-Image（224×224 RGB 图，三通道分别编码原始值、一阶差分、二阶差分）。
- **编码与融合**：冻结预训练 CLIP 编码文本/图像，通过 MLP 分阶段融合 $H^{TS}$、$H^{Text}$、$H^{Image}$ 表示：$H_1 = \text{MLP}_1(\text{Concat}(H^{TS}, H^{Text}))$，$H_2 = \text{MLP}_2(\text{Concat}(H^{TS}, H^{Image}))$，$H = \text{MLP}_3(H_1 + H_2 + H^{TS})$。

**2. Exo-Multimodal Retrieval Enhancer**
- **Cases Analysis**：估计各外生模态的 Forecasting Contribution Weight $W^m$，通过检索历史案例的未来响应评估其预测效用：$e_i^m = \text{MSE}(\hat{Y}_i^m, Y_i^{fut})$，$W^m = \frac{1}{N_R}\sum_i \text{Softmax}_m(-e_i^m)$。
- **Modality-Independent Retrieval**：每种模态独立检索 top-K 案例，合并后按贡献加权分数重排：$S_{qj} = \sum_m W^m s_{qj}^m$。
- **Candidate-Aware Enhancer**：通过 cross-attention 将候选案例融入骨干表示：$H^{Enh} = \text{CrossAttn}(Q, K, V)$，其中 $K,V$ 由候选表示加贡献权重编码得到。

**3. Endo-Multimodal Predictor**
- **时间序列解码**：MLP 多分位数头输出 $\hat{Y} \in \mathbb{R}^{H \times Q}$。
- **内生模态投影**：将预测窗口表示投影到 CLIP 空间，预测未来 Endo-Text 和 Endo-Image 表示：$F^{Text} = \text{Proj}_T(H_{pred}^{Enh})$，$F^{Image} = \text{Proj}_{Image}(H_{pred}^{Enh})$。

**4. 三阶段训练策略**
- **Stage I（Numerical-only）**：仅训练数值预测路径，损失为 Pinball loss $\mathcal{L}_{TS}$。
- **Stage II（Endo-Multimodal）**：冻结前16层 Transformer，训练内生多模态模块，联合优化 $\mathcal{L}_{Multi} = \mathcal{L}_{TS} + \lambda_{Text}\mathcal{L}_{Text} + \lambda_{Image}\mathcal{L}_{Image}$，其中 $\mathcal{L}_{Text/Image}$ 为 Frobenius 范数表示级对齐损失。
- **Stage III（Retrieval Training with Proxies）**：冻结主干与内生模块，仅训练检索增强器；动态采样内生模态组合作为外生代理，优化检索-to-response 学习过程。

## 实验与结果
**数据集**：单模态基准 GIFT-Eval（23数据集）和 TIME（50数据集）；多模态基准 Time-MMD（TS+Text）、MoTime（TS+Text+Image）、FinMultiTime（TS+Text+Image+Table）。

**基线**：GPT4MTS、CALF、Time-VLM、TATS（端到端）；Zeus、Chronos-2、Toto-2、PatchTST-FM-r2、TiRex-2（单模态TSFM）；ChatTime、Aurora（多模态TSFM）。

**主要结果**：
- **单模态**：QiYao-M 在 GIFT-Eval 和 TIME 上均取得最佳 MASE 和 CRPS；相比 TiRex-2-Pretrained 降低 MASE 1.0%、CRPS 1.3%。
- **多模态**：结合外生模态后平均 MSE 降低 5.1%；较各数据集最强单模态 TSFM 平均降低 4.3%；相比现有最多多模态 TSFM（仅支持文本）平均降低 14.6% MSE。
- **消融**：完整模型在全部基准上最优；Endo-Fusion、Endo-Sup、Retrieval Enhancer、Cases Analysis 逐级贡献提升。

## 相关工作脉络
1. **GPT4TS、TEST、Time-LLM、CALF**：通过 reprogramming/对齐/交叉模态交互桥接语言与时序表示，但依赖 task-specific adaptation，跨域迁移受限。
2. **CMIN、Modality-aware Transformer、GPT4MTS、Time-MMD**：将文本信息纳入预测，但未解决多模态类型/数量异质性的泛化问题。
3. **UniTS、TimesFM、Chronos、Toto-2、TiRex-2、ZEUS**：大规模数值预训练学习可迁移时序模式，但不建模多模态信息。
4. **ChatTime、STRIDE、Aurora**：引入多模态预训练的 TSFM，但仅支持外生文本且采用 role-agnostic 共享机制。
5. **本文定位**：首次显式区分内生/外生模态的预测角色，分别设计演化建模与检索增强策略，弥补上述工作在模态角色感知与跨模态泛化上的不足。

## 局限性与未来方向
1. **检索模块对 bank 大小和 top-K 的敏感性**：虽已验证鲁棒性，但在极端场景下仍需调参。
2. **外生模态的语义鸿沟**：代理训练使用内生模态替代外生模态，未能完全模拟真实外生信息的语义内容，可能限制跨模态知识迁移上限。
3. **模型规模有限**：375.63M 参数相比 Toto-2.0（scaling era）偏小，未来可扩展至更大规模。
4. **推理时检索开销**：需维护历史案例库，在实时/流式场景下可能成为瓶颈。

## 研究启发与可借鉴点
1. **角色感知建模思路**：将模态按预测角色分类（内生 vs 外生）并差异化设计，可迁移至其他多模态预测任务（如多模态异常检测、缺失值填补）。
2. **代理训练策略（Proxy Training）**：用易得数据替代稀缺数据训练检索/适配模块，是低资源多模态学习的通用范式，可结合对比学习进一步扩展。
3. **确定性模板生成多模态数据**：基于统计特征与差分曲线自动生成文本/图像，无需人工标注即可构建大规模预训练数据集，适用于其他时间序列多模态场景。
4. **检索增强无需更新参数**：推理时仅通过 cross-attention 融合历史案例，保持 TSFM 冻结，适合部署场景下的快速 adaptation。

## 关键术语表
**Endogenous Modality（内生模态）**：源于时间序列自身的辅助信息（如统计描述、差分图像），提供同一时序动态的多视角表征。

**Exogenous Modality（外生模态）**：来自外部的额外信息（如文本、图像、表格），可能影响目标序列的未来演化。

**Forecasting Contribution Weight（预测贡献权重）**：评估各外生模态检索历史案例后对未来响应的预测效用，用于加权重排候选案例。

**Endo-Modality Proxy Training（内生模态代理训练）**：动态采样内生模态组合替代外生模态，训练检索增强器而无需外生预训练数据。

**Exo-Multimodal Retrieval Enhancer（外生多模态检索增强器）**：独立检索各外生模态的历史案例并融合，实现零参数更新的下游适应。

**Endo-Multimodal Supervisor（内生多模态监督）**：将预测的内生表示与 ground-truth 未来内生表示对齐，显式约束时序演化学习。

**Pinball Loss（分位数损失）**：用于概率预测的多分位数优化目标，衡量预测分布与真实值的校准程度。

## 可复现要素
- **数据集**：GIFT-Eval、TIME、Time-MMD、MoTime、FinMultiTime 均为公开基准；预训练语料含 GIFT-Eval Pretrain、Chronos Corpus 及自构造的 ~5M patch-level 内生样本和 ~10M sequence-level 代理样本。
- **代码/权重**：论文未提及开源代码与模型权重。
- **关键超参**：Patch size=32，骨干24层 Transformer（前16层 Stage II 冻结），CLIP ViT-B/32，主模型 375.63M 参数（含 CLIP 共 526.90M）。
