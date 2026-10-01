---
title: "SCALING-ZERO-ORDER-PRETRAINING-THROUGH-MODEL-SHARDING"
source: https://arxiv.org/pdf/2609.37899v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:16:52"
field: "零阶优化与语言模型预训练"
keywords: ["零阶优化", "zero-order optimization", "SPSA", "Mixture of Experts", "模型分片", "语言模型预训练", "梯度方差"]
innovations: ["提出 SOMA 架构，通过独立专家训练与独立损失将 ZO 梯度相对方差降至共享损失的 1/N", "证明并实验验证独立损失机制在固定预算下显著优于加和损失", "揭示轻度分片优化训练效率、重度分片优化推理吞吐的分离设计原则"]
benchmarks: ["FineWeb-Edu 100B bytes", "WikiText-103"]
---

# 论文速读：SCALING-ZERO-ORDER-PRETRAINING-THROUGH-MODEL-SHARDING

## 一句话总结
论文提出 SOMA（Sharded Optimization Mixture of Assemblies），一种面向零阶优化（ZO）的专家集合架构，通过将模型拆分为多个独立训练的 LSTM 专家，以独立损失消除跨专家扰动噪声，从而在相同总参数量和训练计算预算下显著提升 ZO 预训练的算力效率与推理吞吐。

## 研究问题与动机
1. **零阶优化梯度方差随参数维度线性增长**：固定扰动数量 $n_{\mathrm{pert}}$ 下，相对梯度方差 $\propto M/n_{\mathrm{pert}}$（$M$ 为参数总数），导致大模型 ZO 训练难以收敛。
2. **现有改进路线的局限**：EGGROLL 等方法通过低秩结构提升扰动评估效率，但未改变架构本身；增大 $n_{\mathrm{pert}}$ 虽降低方差，但在固定计算预算下会减少更新步数。
3. **缺乏对独立损失机制的系统性验证**：CBTM 等 MoE 思路在反向传播框架下有效，但在纯前向的 ZO 预训练中，独立专家能否补偿共享表征与联合适应的损失尚不明确。
4. **推理效率与训练效率存在分离考量**：现有工作多聚焦训练损耗，未深入分析架构分解对推理吞吐的额外收益。

## 核心贡献（创新点）
1. **提出 SOMA 架构实现独立 ZO 专家训练**：每个专家使用独立损失在专属数据簇上训练，不交换梯度、激活或优化器状态，本质区别在于将梯度估计的维度从全模型 $M$ 降为单个专家 $M/N$。
2. **证明并验证独立损失可将相对梯度方差降至共享损失的约 $1/N$**：在可分目标下严格推导 $\frac{M/N-1}{M-1} \to 1/N$，实验测得 SOMA N=256 相对方差较 N=1 低 118×，独立损失比加和损失低 4.02×。
3. **揭示训练效率与推理吞吐的分离优化路径**：轻度分片（N=2）在约 150 GPU-hour 预算下获得更低测试损耗（1.76 vs 2.00–2.11 nats/byte），重度分片（N=256）则提供 9.19× 推理吞吐增益（top-k=4 路由）。

## 方法详解
1. **架构设计**：SOMA 是一个 LSTM 专家集合，每个专家含两个残差块（LSTM + MLP，宽 $d_t=32$），共享一个冻结的权重绑定嵌入/解码矩阵 $E \in \mathbb{R}^{V \times d_E}$（$V=256$ 字节词汇）。推理时固定路由（基于词级 tf–idf + 128维截断 SVD + 平衡球形 k-means 聚类）选择 top-k（$k=\min(4,N)$）个专家，平均其预测概率：$p(y_t|x_{<t}) = \frac{1}{|S|}\sum_{e\in S}p_e(y_t|x_{<t})$。
2. **训练配方**：先在 10B 字节子集上训练 N=1 seed（1,000 次更新），学习主体与共享头；再将 100B FineWeb-Edu 语料按 tf–idf 聚类划分为 N 个簇，每个专家复制 seed 主体并在各自簇上独立训练。
3. **独立损失 SPSA 更新**：专家 $e$ 的参数 $\theta_e \in \mathbb{R}^{d_e}$ 使用独立损失 $L_e$ 和稀疏 Rademacher 扰动 $z_j$ 估计梯度：$\hat{g}_e = \frac{1}{n_{\mathrm{pert}}}\sum_{j=1}^{n_{\mathrm{pert}}}\frac{L_e(\theta_e+\varepsilon z_j)-L_e(\theta_e-\varepsilon z_j)}{2\varepsilon}z_j$，配合 Adam 动量更新。**解码头仅用 delta-rule 精确更新**（不经过 SPSA 扰动），公式为 $\nabla_E L_{\mathrm{dec}} = \frac{1}{BT}\sum_{b,t}(p_{b,t}-\mathrm{onehot}(y_{b,t+1}))h_{b,t}^\top$。
4. **分布式训练策略**：完全分片优化（FSO）每专家独占一 GPU，无通信；分布数据与扰动并行（DDPP）用于基线单片模型的公平对比，将一批次的扰动和 batch 分散到多 GPU 并在更新前聚合梯度估计。
5. **方差理论保证**：Proposition 1 证明在固定小批量下，独立损失使相对扰动方差比为 $\frac{M/N-1}{M-1} \xrightarrow{M\to\infty} \frac{1}{N}$；Corollary 2 推导了收敛到平稳点的步长上界，独立分片允许更大学习率。

## 实验与结果
- **数据集**：FineWeb-Edu（100B 字节，byte-level tokenizer $V=256$），WikiText-103 外部验证；上下文长度 1,024 字节，评估窗口 256 输入 + 768 目标。
- **主要结果（~8.44M 参数，~150 GPU-hour）**：SOMA N=2 达到 **1.76 nats/byte** 测试损耗，优于单片 SPSA（$n_{\mathrm{pert}}=64$: 2.00；256: 2.11；1,024: 2.00）和 EGGROLL（2.21）；在 WikiText-103 上分别为 2.07 vs 2.25–2.36 vs 2.49。
- **三种子超参搜索**：SOMA N=2 在所有种子下均优于单片 SPSA，平均降低 0.0396 nats/byte。
- **独立损失消融**：固定架构 N=4，独立损失比加和损失测试损耗降低 **0.035 nats/byte**（1,000 步后，三种子一致），测量方差降低 4.02×。
- **方差测量**：N=256 相对 N=1 降低 **118×** 相对中心方差； doubling $n_{\mathrm{pert}}$ 恰好减半方差。
- **推理吞吐**：SOMA N=256（top-k=4）达 **2.36M tokens/s**，是 SOMA N=8（257k tokens/s）的 **9.19×**，测试损耗 1.68 vs 1.71，但需 59.9× 更多训练计算。
- **BPTT 对照**：相同预算下 BPTT 仍显著优于 ZO（Table 5），本文结论限于 ZO 领域内比较。

## 相关工作脉络
1. **CBTM（Gururangan et al., 2023）**：基于聚类-分支-训练-合并的 Transformer MoE，SOMA 借鉴其专家分离思路但将其适配到 ZO 场景，核心差异是 SOMA 不使用联合训练的路由器，专家完全独立训练。
2. **EGGROLL（Sarkar et al., 2025）**：基于低秩演化策略的 recurrent LLM 预训练，SOMA 在相同测试集上复现对比，SOMA 在同等预算下损耗更低（1.76 vs 2.21 nats/byte），且不需低秩近似假设。
3. **MeZO / Sparse MeZO / MeZO-SVRG**：面向已有 BP 预训练模型的零阶微调方法，SOMA 是面向从头 ZO 预训练的架构设计，目标和问题设置不同。
4. **SmallTalk LM（Filippova et al., 2025）**：独立模型训练+前缀路由，SOMA 在路由机制（tf–idf + SVD clustering）和数据划分策略上与有差异，且 SOMA 专注于验证独立损失对 ZO 梯度的方差抑制效应。
5. **KronZO（Allaire et al., 2026）**：基于 Kronecker 结构化扰动的 GPT-2 Small 预训练，与 SOMA 正交——前者保持单片架构但结构化扰动，后者改变架构为分片专家。

## 局限性与未来方向
1. **当前规模有限**：仅研究 8.44M 参数的小模型和单一 LSTM 架构，未扩展到大规模语言模型或 transformer 架构（因 KV cache 内存开销过大）。
2. **未超越 BPTT**：相同预算下 BPTT 仍显著优于所有 ZO 基线，SOMA 的优势仅在 ZO 内部成立，未解决 ZO 整体不及 BP 的根本问题。
3. **路由固定不学习**：tf–idf + SVD 聚类路由器在训练前固定，不随专家学习动态调整；seed 训练的字节解码器也未扩展到更大词表。
4. **单语料与单一协议**：仅用 FineWeb-Edu，不同语料和评估协议下结论可能不同；多数 scaling 曲线为单 seed，统计显著性受限。
5. **重度分片训练成本极高**：N=256 需 41.9k GPU-hour，训练预算与推理收益之间存在巨大 trade-off。

## 研究启发与可借鉴点
1. **独立损失机制可作为 ZO 通用技巧**：任何需要并行估计梯度的场景（如进化策略、ES variants）均可通过分离损失消除跨参数块噪声，理论界 $1/N$ 方差缩减具有通用性。
2. **路由与训练解耦的工程范式**：固定路由器 + 独立专家训练避免了联合训练中的通信瓶颈，适合分布式/异构硬件部署，可迁移到联邦学习和分布式强化学习场景。
3. **解码头与本体分离更新策略**：用 delta-rule 精确更新 tied head 而用 SPSA 估计本体梯度，巧妙规避了高维 tied embedding 的扰动维度膨胀，是可复用的架构-优化协同设计模式。
4. **训练效率与推理效率的分离度量框架**：本文明确区分 aggregate GPU-hours、wall-clock time 和 inference throughput 三个维度，为后续工作提供了清晰的评估基准和分析框架。
5. **与低资源/专用硬件的结合前景**：随着 forward-only 硬件（如 analog compute、memristor arrays）的发展，SOMA 的纯前向特性使其成为此类平台上预训练语言模型的有力候选方案。

## 关键术语表
**Zero-order optimization (ZO)**：无需计算梯度（反向传播），仅通过扰动参数并评估标量 loss 来估计更新方向的优化范式。
**SPSA（Simultaneous Perturbation Stochastic Approximation）**：Spall 提出的 ZO 方法，每次随机采样一个扰动方向 $z$，通过 $L(\theta+\varepsilon z)-L(\theta-\varepsilon z)$ 估计梯度方向。
**SOMA（Sharded Optimization Mixture of Assemblies）**：本文提出的面向 ZO 的专家集合架构，专家独立训练、独立损失、推理时 top-k 路由平均。
**Independent loss vs. summed loss**：独立损失指每个专家用自己的 loss 估计梯度；加和损失指将所有专家 loss 求和后统一扰动估计，前者消除跨专家噪声。
**FSO（Fully Sharded Optimization）**：每专家独占 GPU、无通信的分布式训练模式；**DDPP（Distributed Data and Perturbation Parallelism）**：将单个模型的 batch 和扰动分散到多 GPU 并聚合梯度。
**Relative gradient variance**：ZO 梯度估计误差方差与真实梯度范数平方的比值，衡量估计质量。
**Top-k routing（tf–idf + SVD clustering）**：基于上下文前缀的词汇特征聚类，选择距离最近的 k 个专家参与推理。
**Aggregate GPU-hours vs. wall-clock time**：前者是所有 GPU 耗时之和（衡量总计算量），后者是最慢 GPU 的耗时（衡量实际训练时长）。

## 可复现要素
- **数据集**：FineWeb-Edu（100B 字节，HuggingFace `sample-100BT`），WikiText-103（官方 split）；数据集公开可用。
- **代码/权重**：论文声明释放全部训练与评估代码及 checkpoint，含附录和 zip 文件中的所有复现指南；CSV 数据与 source 一并开源。
- **关键超参**：$n_{\mathrm{pert}}=64$，$B=64$，$\varepsilon=\mathrm{lr}=10^{-3}$（ plateau 衰减），$\beta_1=0.9$，$\beta_2=0.999$，weight decay $=10^{-4}$，probe density=0.5，context length=1,024 bytes，SVD 维度=128，max_features=50,000；硬件：NVIDIA RTX 5090，精度 FP32/TF32。
