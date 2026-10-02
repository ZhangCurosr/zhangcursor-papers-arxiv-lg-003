---
title: "Synthetic-Data-Characterization-via-Training-Dynamics"
source: https://arxiv.org/pdf/2609.39447v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:39:08"
field: "合成数据质量评估"
keywords: ["synthetic data", "training dynamics", "learnability", "data cartography", "LLM-generated data", "dataset characterization"]
innovations: ["基于训练动力学（confidence/variability/correctness）的样本级合成数据表征框架", "扩展数据地图至多标签分类、序列标注和树预测任务，揭示合成与人类数据的系统性可学习性差异"]
benchmarks: ["SST-5", "SemEval-2018 EC", "UD 2.15 English EWT", "Spanish GSD", "French Sequoia"]
---

# 论文速读：Synthetic Data Characterization via Training Dynamics

## 一句话总结
本文提出基于训练动力学（confidence、variability、correctness）从样本级别表征 LLM 生成合成数据的方法，系统比较了不同 LLM 家族/规模下合成数据的可学习性分布，揭示了合成数据与人类数据在困难样本生成上的系统性差异。

## 研究问题与动机
- 合成数据已成为 NLP 预训练/指令微调的重要组成，但其多样性、真实性和可学习性仍存疑。
- 现有合成数据评估方法将数据集视为静态整体分布，依赖特征空间相似度指标，忽视了样本在训练过程中的个体动态行为。
- 需要一种能从样本级别诊断合成数据质量的框架，以区分不同 LLM 家族、规模及提示策略下的训练动力学差异。

## 核心贡献（创新点）
1. **样本级训练动力学表征框架**：将 Swayamdipta 的数据地图扩展至多标签分类、序列标注、树预测任务，使合成数据可按"易/难/模糊"样本进行定量划分。
2. **四种提示策略的对比设计**：从朴素到困难感知的四级提示（Naive / Exemplified / Difficulty-aware / Ambiguous-targeted），揭示提示复杂度对训练动力学分布无显著影响。
3. **跨 LLM 家族的分布一致性结论**：Wasserstein 距离显示不同模型在同一任务上的训练动力学分布比跨任务的同一模型更相似，且 Qwen 与 Gemma 最接近人类分布。
4. **难样本选择的有效性与局限性**：基于正确性（α）筛选的子集在 OOD 上表现更稳定，而高置信度/高变异性子集的增益不如人类数据显著。

## 方法详解
- **个体置信度 λ**：在每个训练步 t，根据任务类型定义样本的预测概率，分类任务用单类概率，多标签用各类概率均值，序列标注用 token 级均值，树预测用边级均值。
- **核心三指标**：
  - **置信度 γ(x,y)** = 训练过程中 Pθₜ(y|x) 的期望值。
  - **变异性 σ(x,y)** = γ 在训练过程中的标准差，衡量学习不稳定性。
  - **正确性 α(x,y)** = 训练过程中模型预测正确的平均频次（多标签/序列标注/树预测使用 Jaccard 或边匹配率）。
- **数据地图（Data Map）**：将每个样本映射到 (γ, σ, α) 三维空间，划分出易样本（高 γ、低 σ）、难样本（低 γ、低 σ）和模糊样本（高 σ）三类区域。
- **分布对比度量**：使用 Wasserstein 距离（W₁）度量分布形状差异，使用 Cliff's delta 作为非参数效应量比较正确性分布。
- **提示策略**：(i) Naive：仅描述任务；(ii) Exemplified：加入两个参考示例；(iii) Difficulty-aware：加入来自不同学习分区（易/难/模糊）的示例；(iv) Ambiguous-targeted：明确指示生成模糊样本。

## 实验与结果
- **数据集**：SST-5（情感分类）、SemEval-2018 EC（多标签情绪分类）、UD 2.15 English EWT（POS 标注 + 依存句法分析），并扩展至西班牙语 GSD 和法语 Sequoia。
- **LLM 家族**：Qwen-3 8B、Llama-3.1 8B、Olmo-3 7B、Gemma-3（4B/12B/27B）。
- **编码器**：RoBERTa-large（主）、BERT、DistilBERT、LLM2Vec（基于 Llama-3.1-8B + LoRA）。
- **关键数值**：
  - SST-5 上人类数据最佳 ID 准确率 58.1–59.3%，合成数据最高约 44.1%（Gemma-27B，最大正确性子集）。
  - dependency parsing（EWT）上人类数据 LAS 达 93.2%，合成数据最高约 38.9%。
  - W₁ 距离：SST-5 上 LLaMA 与 Qwen 差异仅 0.03；dependency parsing 跨模型差异显著增大。
  - Cliff's delta 对正确性：多数合成数据在 SST-5 和 EC 上为正（更易样本），在 dependency parsing 上为强负（更难样本）。
- **最强结果**：基于最大正确性（max α）子集的 RoBERTa 在 SST-5 OOD（Yelp）上达到 53.9%，优于随机选择与高置信度子集。

## 相关工作脉络
1. **Swayamdipta et al. (2020)**：原始数据集制图方法，仅在单标签分类任务上用 RoBERTa，本文将其推广至多标签、序列标注和树预测。
2. **Kang et al. (2025)**：分析合成数据对 LLM 预训练的影响，本文聚焦下游分类/标注任务，并提供样本级诊断视角。
3. **Mekala et al. (2024)**：用小模型学习动力学过滤预训练语料，本文用编码器（非自回归模型）做细粒度样本级分析。
4. **Sucholutsky & Schonlau (2021)、Maekawa et al. (2024)**：将训练动力学作为优化目标（梯度/嵌入匹配），本文将其作为诊断工具。
5. **Muñoz-Ortiz et al. (2024)、Zamaraeva et al. (2025)**：从语言学角度对比合成与人类文本，本文从可学习性视角补充了训练行为维度的诊断。

## 局限性与未来方向
- LLM 输出的近重复率高，为匹配人类数据集大小需生成 3–25× 原始样本。
- 复杂结构化任务（依存解析）的 LLM 生成质量较低，需后处理修正，且修正不影响分布结论但存在人工干预风险。
- 主观标注任务的合成等价性难以可靠验证，本文依赖高标注一致性的基准来缓解。
- 计算资源受限，仅使用开源模型，未探索商业 LLM 的扩展。
- 未来可探索跨语言迁移、结合模型崩溃（model collapse）分析、以及将训练动力学用于合成数据混合配比优化。

## 研究启发与可借鉴点
1. **可扩展的数据图方法**：本框架可将数据地图概念推广至任何有标签的训练任务，可作为后续合成数据质量评估的标准化工具。
2. **困难度感知的数据筛选策略**：基于正确性（α）而非置信度（γ）筛选样本的效果更稳定，可作为数据选择优先级指标。
3. **跨编码器稳健性验证**：用多种编码器（RoBERTa/BERT/LLM2Vec）对比可增强合成数据评估的可信度，减少单一架构偏差。
4. **提示策略对动力学影响有限的启示**：表明合成数据的可学习性特征主要由 LLM 自身决定而非提示模板，为数据生成流程设计提供简化依据。
5. **与团队方向结合机会**：可将训练动力学指标用于指导合成数据清洗/配比，或与其他质量度量（如多样性、语义相似度）联合优化数据选择策略。

## 关键术语表
**Data Map**：将训练过程中样本的 confidence、variability、correctness 三维可视化，用于诊断数据集特性。
**Confidence (γ)**：模型在整个训练过程中对样本正确预测概率的期望值，反映平均学习难度。
**Variability (σ)**：训练过程中 confidence 的标准差，反映样本预测不稳定程度。
**Correctness (α)**：模型预测正确的平均频次，衡量样本的正确识别率。
**Wasserstein Distance (W₁)**：用于度量两个训练动力学分布之间地理距离的非参数化统计量。
**Cliff's Delta**：非参数效应量，用于比较两个正确性分布的相对大小。
**Learning Trajectory**：单个样本在训练过程中的置信度和正确性演变曲线。

## 可复现要素
- **代码**：已开源，见论文公开仓库链接。
- **数据集**：SST-5、SemEval-2018 EC、UD 2.15 English EWT（公开）；西班牙语 GSD 和法语 Sequoia 亦为公开 UD 树库。
- **LLM 模型**：Qwen-3 8B、Llama-3.1 8B、Olmo-3 7B、Gemma-3（4B/12B/27B）均为开源模型。
- **编码器**：RoBERTa-large（主）、BERT、DistilBERT、LLM2Vec（基于 Llama-3.1-8B + LoRA）。
- **关键超参**：top-k=50、top-p=0.95、temperature 因任务/模型而异（SA: 0.8–1.2；PoS/DP: 0.5–1.0）；MinHash Jaccard 阈值 0.4；近重复过滤阶段。
- **生成过采样比例**：为匹配人类数据集大小，PoS/DP 任务需生成 3–25× 原始样本。
