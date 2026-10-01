---
title: "RACE-RELATION-LEVEL-COUNTERFACTUAL-EXPLA-NATIONS-FOR-HETEROG"
source: https://arxiv.org/pdf/2609.37650v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:47:16"
field: "图神经网络可解释性"
keywords: ["counterfactual explanation", "heterogeneous GNN", "graph interpretability", "relation-level attribution", "exact search"]
innovations: ["实例级精确关系搜索提供确定性对抗性解释", "层级化关系→边缘细化管道保证单边缘恢复不可约性", "显式不可行性报告避免强行归因提升可信度"]
benchmarks: ["ACM", "Cora-derived", "ogbn-mag", "ogbn-arXiv", "DBLP"]
---

# 论文速读：RACE-RELATION-LEVEL-COUNTERFACTUAL-EXPLA-NATIONS-FOR-HETEROG

## 一句话总结
RACE 提出了一种面向异构图神经网络的**关系级对抗性解释**框架，通过对每个实例在关系类型层面进行精确枚举搜索，找到能翻转预测的最小关系删除集，并在离散模型上进行单边恢复验证；该方法在 ACM、Cora、ogbn-mag 等数据集上显著提升了翻转成功率并减少了删除边数。

## 研究问题与动机
- **核心问题**：现有 GNN 对抗性解释方法在异构图上需先将图坍缩为无类型边，导致无法回答领域专家真正关心的问题——"哪个关系类型驱动了该预测？"。
- **现有方法缺陷**：CF-GNNExplainer、CF²、RCExplainer 等基线方法输出的是散乱的个体边集合，其语义不可聚合为人类可读的"该预测依赖于某类关系"这样的陈述；且它们无法提供实例级别的确定性解释与不可行性报告。
- **非单调性问题**：作者证明某些预测在部分关系删除时发生翻转，但在完全删除时反而不翻转，说明聚合级的翻转率搜索不能替代实例级问题。
- **解释可信度需求**：当模型未学到可解释机制时，现有方法仍会强行归因；RACE 通过显式报告不可行性来保持解释可信度。

## 核心贡献（创新点）
1. **实例级关系对抗性解释的精确搜索**：对每个实例枚举所有 2^K 关系子集，返回认证的最小关系删除集或显式不可行报告，与软掩码基线相比具有确定性且成功率波动更小（6–8 pp）。
2. **经验证的层级化关系→边缘细化管道**：关系级答案被细化为单边缘恢复不可约的认证类型边缘集，在离散模型上通过逐边恢复验证，提供单一边缘恢复不可约性证书。
3. **保真度度量框架与可识别性刻画**：采用 NF/SF 作为描述性保真度指标，刻画了关系级归因可识别的条件，并通过合成研究证明搜索能恢复模型实际依赖的关系而非虚构归因。

## 方法详解
**整体流程**：
- **Stage 1**：构建关系感知异构图 Transformer 骨干网络并冻结
- **Stage 2**：实例级精确关系搜索（Eq. 5-6）
- **Stage 3**：层级化关系→边缘细化
- **Stage 4**：NF/SF 保真度评估与稳定性分析

**关键公式**：
- **边缘级连续优化**（Gumbel-Sigmoid 松弛）：
  $\mathcal{L} = \mathbb{E}_v[\max(0, M_v(\mathcal{G} \odot m) + \kappa)] + \lambda_{sp}\frac{1}{E}\sum(1-m_e^r) + \beta\frac{1}{E}\sum m_e^r(1-m_e^r)$
- **实例级精确搜索**：$S_v^* = \arg\min_S C_v(S)$ s.t. $M_v(\mathcal{G} \backslash S) \leq -\kappa$，代价函数为字典序：先最少关系数，再最小局部边比例，最后最强翻转 margin。
- **NF/SF 保真度**：$NF_v(S) = \mathbf{1}[f(\mathcal{G}\backslash S)_v \neq \hat{y}_v]$，$SF_v(S) = \mathbf{1}[f(S)_v = \hat{y}_v]$。

**关系→边缘细化**：从 $\mathcal{G} \backslash S_v^*$ 开始，按显著性顺序逐一恢复关系内的边，若翻转保持则接受恢复，直到不动点；返回单边缘恢复不可约集（或报告预算受限状态）。

## 实验与结果
**数据集**：ACM（2关系）、ogbn-mag（2/4关系）、Cora-derived（2关系）、ogbn-arXiv（2关系）、DBLP（2关系）及合成 SCM。

**主要结果**（同冻结骨架、实例目标协议，10 seeds）：
- **CSR 提升**：RACE-hier 相对最强基线 CF²-hetero 在 ACM/Cora/ogbn-mag 上分别提升 +0.8/+2.7/+1.4 pp（p=0.002，Holm 校正 ≤0.062）
- **边成本降低**：减少 -1.9/-1.0/-2.7 pp
- **跨骨架泛化**：在 ogbn-arXiv 上4种骨架（RACE-bb/HAN/HGT/R-GCN）均复现 CSR 优势（+0.9 至 +2.6 pp）
- **稳定性**：跨种子关系集一致率达 0.78–0.89；5% 随机边扰动下 $S_v^*$ 一致率 0.93–0.98
- **覆盖率**：ACM 0.18、Cora 0.34、ogbn-mag 0.17（多数正确分类实例无可行关系删除，显式报告不可行）

## 相关工作脉络
1. **GNNExplainer / PGExplainer**： factual 解释器，仅学习保留预测的 mask，无法回答 counterfactual 查询，且无关系粒度。
2. **CF-GNNExplainer / CF² / RCExplainer**：图对抗性解释的开山之作，但 operate on homogeneous pairwise graphs，不保留关系类型。
3. **NSEG / InduCE / C2Explainer**：近期 counterfactual baselines，NSEG 几乎从不翻转（CSR ≤ 0.01），INDUCE 不稳定。
4. **HAN / HGT / R-GCN**：异构 GNN backbone，本文在其上实现关系级解释。
5. **概率因果框架（Pearl, Tian & Pearl）**：本文借用 NF/SF 作为描述性指标，将因果语言保留给显式生成机制的合成数据。

## 局限性与未来方向
- **关系数量限制**：精确枚举在 K ≤ 10 时可行（≤10s），K=20 时需依赖 branch-and-bound 近似
- **边缘细化非全局最优**：单边缘恢复不可约性证书不等于全局最小删除集，存在 gap（Appendix I 显示 15% 案例偏离1条边）
- **覆盖率低**：ACM/Cora/ogbn-mag 覆盖率仅 18%/34%/17%，大部分实例无可行关系删除
- **未来方向**：扩展至 meta-path 级别干预、基于结构因果模型的因果可识别性分析、可扩展到更大 K 的证明近似界、人类用户评估

## 研究启发与可借鉴点
1. **实例级精确搜索替代软掩码优化**：对于小关系集合（K≤10），枚举搜索提供确定性保证，避免软方法因随机 ordering 导致 6–8 pp 成功率波动，可作为基线设计参考。
2. **层级化细化策略**：关系级→边缘级的两步管道设计思路可迁移至其他多粒度解释任务（如特征→子结构、节点→社区）。
3. **不可行性报告机制**：显式报告"无可行解释"而非强行归因，提升了可信度，该设计值得在医疗/金融等高风险领域解释器中采用。
4. **离散模型逐阶段验证**：在离散模型上每一步操作后评估翻转状态，而非仅在优化后进行阈值判断，提高了验证可靠性。

## 关键术语表
**Counterfactual Explanation（对抗性解释）**：通过最小输入扰动（如边删除）使模型预测翻转的解释方法。
**Relation-level Search（关系级搜索）**：在关系类型集合上枚举所有子集以寻找最小翻转集的方法。
**Necessity Fidelity (NF)**：删除某集合后预测是否翻转的指标（必要性保真度）。
**Sufficiency Fidelity (SF)**：保留某集合后预测是否保持的指标（充分性保真度）。
**Single-edge Restoration Irreducibility（单边缘恢复不可约性）**：任意单条边恢复都会导致预测不再翻转的性质，是解释稳定性的局部证书。
**Coverage（覆盖率）**：存在可行关系删除集的实例比例，反映解释的可应用范围。
**Non-monotonicity（非单调性）**：部分关系删除导致翻转，但完全删除反而不翻转的现象。
**Lexicographic Cost（字典序代价）**：优先最小化关系数，其次最小化局部边比例，最后最大化翻转 margin 的多级代价函数。

## 可复现要素
- **数据集**：ACM、ogbn-mag、ogbn-arXiv、Cora、DBLP 均为公开基准；合成 SCM 由论文 Sec. 4.2 描述
- **代码**：匿名实现开源，地址 https://anonymous.4open.science/r/race\_official-68A1
- **超参**：骨干网络 hidden dim=32、2层、dropout=0.5、weight decay=5×10⁻⁴；解释器 150 mask steps、Adam lr=0.1、λ_sp=0.05、Gumbel τ=1.0；关系搜索 κ=0
- **统计**：10 seeds（跨骨架/额外数据集为5 seeds），配对符号翻转检验，Holm 校正
