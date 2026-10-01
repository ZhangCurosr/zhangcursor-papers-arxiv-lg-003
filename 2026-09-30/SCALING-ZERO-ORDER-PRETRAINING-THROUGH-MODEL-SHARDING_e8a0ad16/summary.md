---
title: "SCALING-ZERO-ORDER-PRETRAINING-THROUGH-MODEL-SHARDING"
source: https://arxiv.org/pdf/2609.37899v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:16:55"
field: "零阶优化与语言模型预训练"
keywords: ["zero-order optimization", "language model pretraining", "mixture of experts", "SPSA", "gradient variance"]
innovations: ["独立损失机制理论证明方差降低1/N", "SOMA架构实现ZO预训练的计算效率优势"]
benchmarks: ["FineWeb-Edu", "WikiText-103"]
---

# 论文速读：SCALING-ZERO-ORDER-PRETRAINING-THROUGH-MODEL-SHARDING

## 一句话总结
论文提出SOMA架构，通过将零阶优化（ZO）训练中的模型划分为多个独立专家，利用可分离损失函数显著降低梯度估计方差，从而在固定计算预算下实现比单体ZO方法更高效的预训练效果。

## 研究问题与动机
- 零阶优化（ZO）适用于前向硬件和非可微损失场景，但梯度估计方差随扰动维度线性增长，阻碍大模型训练。
- 在固定扰动数量下，单体SPSA方法的相对梯度方差缩放比例为 $M/n_{\mathrm{pert}}$（$M$ 为参数总数），控制误差需 $n_{\mathrm{pert}} \propto M$，导致每步计算复杂度为 $O(M^2)$。
- 现有改进方法（如EGGROLL的低秩结构）仅优化扰动评估效率，未从架构层面改变估计问题的规模。
- 需要一种新架构，使总模型规模增长时，每个独立估计块的参数维度和梯度方差不随之扩大。

## 核心贡献（创新点）
1. **提出SOMA架构**：基于CBTM思想的独立专家训练框架，每个LSTM专家在独立数据集群上用SPSA训练，无需交换梯度、激活值或优化器状态。
2. **独立损失机制的理论证明**：证明在可分离目标下，独立损失将相对梯度方差降至共享损失估计器的约 $1/N$（$N$ 为专家数），实验测得4.02×降低。
3. **训练计算效率优势**：在约8.44M参数和150 aggregate GPU-hours预算下，SOMA N=2达到1.76 test nats/byte，优于Monolithic SPSA（2.00–2.11）和EGGROLL（2.21）。
4. **推理吞吐量收益**：重度分片（N=256）在top-k路由（k=4）下实现2.36M tokens/s，是SOMA N=8的9.19倍，且测试损失更低（1.68 vs 1.71）。

## 方法详解
- **架构设计**：SOMA由共享的embedding/decoder矩阵E（权重重绑定）和N个独立LSTM专家组成，每个专家包含2个残差块（LSTM+MLP），宽度 $d_t=32$。
- **路由机制**：使用tf-idf词频加权、128维SVD降维和平衡球面k-means聚类，推理时选择最近k=min(4,N)个专家，路由固定不通过混合过程微分。
- **训练流程**：先以N=1种子模型训练1000步学习共享embedding/decoder；将100B字节语料按词级tf-idf聚类分配给各专家；专家复制种子主体并独立训练。
- **更新规则**：身体参数使用SPSA估计梯度，解码器权重使用delta-rule精确计算梯度（不通过循环体反向传播），公式为 $\nabla_E L_{\mathrm{dec}} = \frac{1}{BT}\sum_{b,t}(p_{b,t}-\mathrm{onehot}(y_{b,t+1}))h_{b,t}^\top$。
- **独立损失**：每个专家使用自身损失 $L_e$ 独立估计梯度，消除跨专家扰动噪声，理论相对方差比约为 $\frac{M/N-1}{M-1} \xrightarrow{M\to\infty} \frac{1}{N}$。

## 实验与结果
- **数据集**：FineWeb-Edu（100B字节，byte vocabulary V=256），外部验证集WikiText-103。
- **评估指标**：test nats/byte，路由使用256字节观察上下文，预测后续768字节目标。
- **主要结果**（~8.44M参数，150 GPU-hours）：
  - SOMA N=2：1.76 test nats/byte
  - Monolithic SPSA (n_pert=64)：2.00
  - Monolithic SPSA (n_pert=256)：2.11
  - Monolithic SPSA (n_pert=1024)：2.00
  - EGGROLL：2.21
- **WikiText-103冻结验证**：SOMA N=2达2.07，Monolithic SPSA为2.25–2.36，EGGROLL为2.49。
- **独立损失消融**：SOMA N=4在1000步后独立损失比加和损失降低0.035 nats/byte，相对方差降低4.02×。
- **梯度方差测量**：SOMA N=256相对中心方差比N=1低118倍，验证理论预测。

## 相关工作脉络
- **CBTM（Gururangan et al., 2023）**：训练独立Transformer专家并按语义聚类合并，SOMA采用类似组织但针对ZO优化问题，研究局部学习信号能否补偿联合适应的缺失。
- **EGGROLL（Sarkar et al., 2025）**：使用低秩进化策略训练循环模型，SOMA对比实验显示相同基准下性能更优。
- **MeZO及变体（Malladi et al., 2023; Chen et al., 2024b）**：针对微调的ZO方法，使用控制变量或稀疏选择，SOMA聚焦从头预训练架构设计。
- **KronZO（Allaire et al., 2026）**：使用Kronecker结构化扰动预训练GPT-2 Small，损失函数和数据集不同，无法直接比较。
- **DiLoCo（Douillard et al., 2023）与联邦平均**：通过通信摊销或模型平均协调分布式训练，SOMA完全无通信需求。

## 局限性与未来方向
- 仅使用单一byte-level语料库和小LSTM专家，局限于8.44M参数规模。
- 大多数缩放运行采用单seed，尽管关键比较使用三seed验证。
- 独立训练牺牲了跨领域联合表示学习，路由固定不随专家学习而适应。
- 未与BPTT进行完整公平比较，早期预算下BPTT损失更低（表5）。
- 短提示生成未评估，需要显式冷启动路由策略。

## 研究启发与可借鉴点
1. **独立损失机制的可迁移性**：对于任何需要使用ZO优化的大型模型，可考虑将参数划分为独立子块并使用分离损失，理论保证方差降低1/N。
2. **路由-训练一致性设计**：使用相同聚类管线同时指导专家分配和推理路由，简化系统且保持语义一致性。
3. **解码器路径的delta-rule更新**：保持嵌入/解码器权重冻结或通过精确梯度更新，避免将其纳入ZO扰动维度，有效缩小估计空间。
4. **GPU时与聚合计算的区分**：明确区分墙钟时间和aggregate GPU-hours，为公平比较分布式训练成本提供清晰框架。

## 关键术语表
- **Zero-order optimization (ZO)**：无需梯度信息、仅通过函数值估计优化的方法，适用于前向硬件和非可微损失场景。
- **SPSA（Simultaneous Perturbation Stochastic Approximation）**：通过随机扰动方向估计梯度的ZO方法，每步仅需两次函数评估。
- **SOMA（Sharded Optimization Mixture of Assemblies）**：本文提出的独立专家集成架构，专为ZO预训练设计。
- **Relative gradient variance**：梯度估计误差与真实梯度范数的比值，衡量ZO估计质量。
- **Fully Sharded Optimization (FSO)**：专家完全分布式训练模式，无梯度、激活或优化器状态交换。
- **Distributed Data and Perturbation Parallelism (DDPP)**：单体模型的并行训练模式，将批次和扰动分发到多GPU。
- **nats/byte**：自然单位的信息论损失度量，表示每字节的平均交叉熵。

## 可复现要素
- **数据集**：FineWeb-Edu（100B字节），公开可用。
- **代码与权重**：论文声明释放所有训练和评估代码及检查点，支持完整复现。
- **关键超参**：$n_{\mathrm{pert}}=64$，$B=64$，context length=1024，learning rate初始$10^{-3}$，$\varepsilon=10^{-3}$，$\beta_1=0.9$，$\beta_2=0.999$，weight decay=$10^{-4}$，probe density=0.5。
- **硬件**：RTX 5090 GPU，FP32权重，TF32内核。
