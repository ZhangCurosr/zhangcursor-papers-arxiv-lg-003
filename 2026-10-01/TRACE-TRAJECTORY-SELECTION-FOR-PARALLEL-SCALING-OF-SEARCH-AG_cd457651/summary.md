---
title: "TRACE-TRAJECTORY-SELECTION-FOR-PARALLEL-SCALING-OF-SEARCH-AG"
source: https://arxiv.org/pdf/2609.39912v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:40:17"
field: "强化学习与Agent系统"
keywords: ["Search Agents", "Trajectory Selection", "Parallel Scaling", "Graph Neural Networks", "Evidence Aggregation", "Efficient Reasoning"]
innovations: ["提出TRACE选择器，利用跨轨迹证据传播进行轻量级轨迹选择", "构建保持发生的查询-证据图，连接不同轨迹的共享证据", "实现答案条件读取，结合局部轨迹背景与全局证据信息"]
benchmarks: ["WebQA", "BrowseComp-Plus", "FRAMES", "GAIA"]
---

# 论文速读：TRACE: TRAJECTORY SELECTION FOR PARALLEL SCALING OF SEARCH AG

## 一句话总结
论文提出了 **TRACE (Trajectory Ranking with Aggregated Cross-Rollout Evidence)**，一种轻量级的轨迹选择器，用于在并行搜索智能体的候选池中筛选最优轨迹。它通过构建保持检索来源的证据图并利用跨轨迹信息传播，在不增加额外搜索或自回归生成的情况下，显著提升了最终答案的准确率，且效率远高于基于大语言模型的生成式聚合方法。

## 研究问题与动机
1.  **核心问题**：在搜索智能体的并行扩展（Parallel Scaling）中，如何从 K 个已完成的搜索轨迹（候选池）中有效地选择一个最终答案？
2.  **现有方法的不足**：
    *   **多数投票（Majority Voting）**：仅比较最终答案的一致性，忽略了产生答案的搜索过程和检索证据。
    *   **独立的验证器（Independent Verifiers）**：通常独立评估每个候选方案，未能利用候选池中的共享证据。
    *   **生成式聚合（Generative Aggregation，如 AggAgent）**：虽然能结合多个轨迹的信息，但引入了额外的自回归推理阶段。当正确的答案已在候选池中时，这种额外的生成是不必要的，可能引入新的错误或增加高昂的计算成本。
3.  **动机**：不同的搜索轨迹可能通过不同的子查询检索到相同的文本片段，或访问相同的文档但暴露出不同的段落。这些“共享证据关系”为评估候选轨迹提供了有用的信号。有效的问题在于如何设计一个既能利用跨轨迹证据，又能保留每个搜索过程来源信息的表示方法。

## 核心贡献（创新点）
1.  **提出轨迹选择作为并行扩展的后处理任务**：将轨迹选择与轨迹生成解耦，提供了一个替代投票和生成式聚合的轻量级选择方案。
2.  **引入保持发生的查询-证据图（Occurrence-Preserving Query-Evidence Graph）**：保留每个搜索查询和证据的具体发生情况，并通过共享内容块（WebQA）或文档身份（长视距浏览）连接跨轨迹的证据。
3.  **设计基于图神经网络的跨轨迹证据传播机制**：通过关系特定的 GNN 在证据节点之间传播信息，使每个轨迹能利用其他相关轨迹的证据来更新自身状态。
4.  **提出答案条件轨迹读取（Answer-Conditioned Trajectory Readout）**：每个候选答案只查询其自身轨迹更新后的状态，实现了局部检索背景与全局相关搜索信息的结合。
5.  **证明了 TRACE 的可迁移性和高效性**：一个选择器无需针对特定生成器微调，即可跨多种 Rollout 策略和智能体骨干网络使用，并在 WebQA 和长视距数据集上均优于投票和大型 LLM 聚合器，且处理吞吐量高出至少 10 倍。

## 方法详解
TRACE 的整体流程包括三个阶段：图构建、证据传播和轨迹评分。

1.  **发生保持的查询-证据图构建**：
    *   将每个完成的搜索轨迹解析为搜索查询（Subquery）、检索到的证据（Evidence）、来源引用（Source）和最终答案（Answer）。
    *   每个具体的查询和证据都作为独立的节点，保留其检索历史。
    *   **跨轨迹连接**：如果两个证据节点包含相同的文本块（WebQA）或来自同一个文档（长视距浏览），则在它们之间建立连接。连接保留在边关系上，而不是合并节点，从而保持了各自的来源。

2.  **跨轨迹证据传播（Cross-Rollout Evidence Propagation）**：
    *   使用冻结的 **Qwen3-Embedding-8B** 模型生成所有节点（问题、Subquery、Evidence、Answer）的 4096 维文本嵌入。
    *   可训练的编码器将嵌入映射到 $d=256$ 维的隐藏状态。答案节点的初始状态还包含了答案频率特征。
    *   使用 **4 层关系特定的 GraphSAGE** 进行信息传播。对于每个节点，根据边的关系类型（如检索关系、时间关系、跨轨迹共享证据关系）分别聚合邻居信息并更新自身状态。
    *   此阶段，Query 和 Answer 状态不参与图消息传递。

3.  **答案条件轨迹读取与评分（Answer-Conditioned Trajectory Readout & Scoring）**：
    *   对于每个候选答案 $a_i$，使用其对应的轨迹 $\tau_i$ 中更新后的 Subquery 和 Evidence 状态作为 Key 和 Value，以答案嵌入作为 Query，通过一个 **QFormer（跨注意力机制）** 进行读取。
    *   生成的候选表示 $\mathbf{h}_{a_i}^{\star}$ 同时保留了该轨迹的局部上下文和来自其他轨迹的全局信息。
    *   通过可学习的投影将问题和候选答案映射到共同的评分空间，计算余弦相似度作为轨迹的得分（Score）。
    *   最终选择得分最高的已有候选答案 $\hat{a}$ 作为输出。

4.  **训练目标**：
    *   在冻结文本嵌入的基底上进行训练。
    *   损失函数由三部分组成：加权二元交叉熵（BCE，学习候选正确性）、列表排序损失（Listwise，集中概率质量在正确轨迹上）和硬排名损失（Hard-ranking，拉开最强正确候选与最强错误候选的差距）。

## 实验与结果
*   **数据集**：
    *   **WebQA**：NQ, HotpotQA, TriviaQA, PopQA, 2WikiMultiHopQA, MuSiQue, Bamboogle (共 3,125 个问题)。
    *   **长视距搜索**：BrowseComp-Plus, FRAMES, GAIA。
*   **评估基线**：
    *   单条轨迹 (Single rollout)
    *   多数投票 (Majority Voting)
    *   置信度加权投票 (Weighted Voting)
    *   最少工具调用 (Fewest Tools)
    *   生成式聚合 (SolAgg, SummAgg, AggAgent - 使用 Qwen3-32B)
*   **主要结果**：
    *   **WebQA (K=16, Qwen2.5-14B)**：
        *   TRACE 在 Base 和 SFT 上分别达到 **45.2%** 和 **49.2%** 的 EM，优于多数投票 (42.6%/47.2%) 和最强的生成式聚合器 SolAgg (43.9%/48.0%)。
        *   在 6 种不同的 Rollout 策略下，TRACE 均能提升最终答案选择效果。
    *   **长视距搜索 (K=16)**：
        *   TRACE 平均准确率达到 **78.6%**，优于多数投票 (75.5%) 3.1 个百分点，在所有 6 个基准测试-骨干组合中有 5 个达到最高或并列最高。
    *   **效率**：
        *   TRACE 的处理速度远快于生成式聚合。在 SFT WebQA 池上，TRACE 仅需 **0.271 GPU-hours**，而 SolAgg, SummAgg, AggAgent 分别需要 3.28, 32.96, 69.93 GPU-hours，即 TRACE 的效率是它们的 **12倍、122倍和258倍**。
    *   **鲁棒性**：
        *   TRACE 可以从 8 个轨迹中选择，其性能接近在 64 个轨迹上进行多数投票，表明更好的选择可以减少所需的采样预算。

## 相关工作脉络
1.  **并行扩展与轨迹聚合**：与 ParallelMuse, AggAgent 等工作不同，TRACE 不生成新的答案，而是直接从现有候选中选择一个，避免了额外的生成成本。
2.  **验证与候选选择**：与 Best-of-N 和独立的验证器（如 Cobbe et al.）不同，TRACE 将轨迹质量视为关系性的，允许一个轨迹的得分依赖于其他轨迹中发现的证据。
3.  **基于图结构的证据推理**：与在单一轨迹内部组织证据的多跳 QA 或事实核查方法（如 GEAR）不同，TRACE 在多个已完成的搜索轨迹之间构建图，用于跨轨迹的证据传播。
4.  **测试时计算扩展**：遵循通过重复采样和并行探索来提高模型性能的大方向（如 Brown et al., Snell et al.），但专注于搜索智能体这一特定场景的合并阶段。

## 局限性与未来方向
*   **依赖预训练嵌入**：TRACE 的性能依赖于 Qwen3-Embedding-8B 等预训练文本嵌入的质量，可能需要针对特定领域进行优化。
*   **图的构建规则**：当前的图连接规则（如基于精确文本匹配或文档 ID）可能无法捕捉所有语义上的相似证据，未来可以探索更灵活的匹配策略。
*   **泛化性**：虽然展示了跨不同 Rollout 策略和模型规模的泛化性，但在更加复杂或未知的搜索环境中表现如何仍需进一步验证。
*   **未来方向**：可以探索与其他类型的验证信号（如过程监督）的结合，或者将 TRACE 的设计思路应用到更广泛的 Agent 任务中。

## 研究启发与可借鉴点
1.  **发生保持的图表示**：保留查询和证据的多次发生，而不是简单地合并，有助于捕捉搜索过程的具体背景，这对于评估检索增强任务的轨迹质量很有启发。
2.  **分离推理与选择**：将轨迹生成和最终答案选择解耦，通过轻量级的选择器来利用跨轨迹信息，是一种高效且可扩展的方案。
3.  **关系特定的图神经网络**：在不同类型的关系（如检索关系、时间关系、共享证据关系）上分别进行消息传递，可以更精细地建模任务结构。
4.  **跨轨迹证据传播**：利用候选池中其他轨迹的证据来辅助当前轨迹的评分，可以有效弥补单条轨迹信息不足的缺陷。
5.  **效率优先的设计**：在追求性能的同时，充分考虑计算成本，证明了轻量级选择器可以媲美甚至超越重型生成式聚合器。

## 关键术语表
*   **TRACE (Trajectory Ranking with Aggregated Cross-Rollout Evidence)**: 论文提出的轻量级轨迹选择器，用于在并行搜索轨迹中进行最佳选择。
*   **Parallel Scaling**: 通过生成多个候选轨迹来提高搜索智能体性能的策略。
*   **Occurrence-Preserving Query-Evidence Graph**: 一种图结构，保留了每个搜索查询和证据的具体发生情况，并通过共享证据关系连接不同轨迹。
*   **Cross-Rollout Evidence Propagation**: 利用图神经网络在不同轨迹的共享证据节点之间传播信息，以更新轨迹表示。
*   **Answer-Conditioned Trajectory Readout**: 使用候选答案作为查询，从其自身轨迹更新后的节点状态中读取信息，生成最终表示。
*   **Generative Aggregation**: 使用另一个大语言模型来综合多个轨迹信息并生成最终答案的方法（如 AggAgent）。
*   **Rollout**: 指智能体为解决一个查询而生成的一个完整的搜索轨迹。

## 可复现要素
*   **数据集**：WebQA (NQ, HotpotQA 等), BrowseComp-Plus, FRAMES, GAIA。论文声明代码可用。
*   **代码/权重**：
    *   代码已开源：https://github.com/Jaasssoooonnnnn/TRACE
    *   使用预训练文本嵌入模型：**Qwen3-Embedding-8B**。
*   **关键超参**：
    *   图神经网络层数：L = 4
    *   隐藏维度：d = 256
    *   优化器：AdamW
    *   学习率：$3 \times 10^{-4}$
    *   Dropout：0.1
    *   Batch size：WebQA 为 256，长视距为 4
    *   训练轮数：3 轮
