---
title: "SCALING-INFLUENCE-FUNCTIONS-IN-LLMS-THROUGH-EIGENBASIS-CORRE"
source: https://arxiv.org/pdf/2609.37842v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:16:41"
field: "LLM可解释性与数据归因"
keywords: ["influence functions", "gradient compression", "LLM attribution", "EK-FAC", "one-bit quantization", "data selection"]
innovations: ["证明了半白化梯度的PCA是固定维度线性影响压缩的最优解", "提出两阶段投影（EK-FAC子空间+子空间PCA）逼近理论最优表示", "一比特量化使存储降至<16KB/样本且几乎无损"]
benchmarks: ["GPT-2 WikiText-2 LDS", "OLMo 2 SFT NDCG@20 vs EK-FAC", "Counterfactual Retraining Perplexity"]
---

# 论文速读：SCALING-INFLUENCE-FUNCTIONS-IN-LLMS-THROUGH-EIGENBASIS-CORRE

## 一句话总结
本文提出 **EOGP**（Eigenbasis-corrected One-bit Gradient Projection），通过特征基修正的一比特梯度投影，将LLM训练梯度压缩至每样本 <16 KB 的极小存储，同时保持高保真的影响函数计算；在 GPT-2 和 OLMo 2（1B–32B）上均优于现有压缩基线，节省存储超 100 倍。

## 研究问题与动机
1. **存储瓶颈**：Influence functions 用于分析训练数据对 LLM 行为的影响，需重用训练梯度进行多次归因查询；但 8B 模型单样本半精度梯度需 16 GB，大规模场景不可行。
2. **压缩方向未知**：现有压缩方法（随机投影、曲率感知投影）无法预先知道未来影响查询的方向，压缩需保留"跨查询通用"的信息。
3. **理论缺失**：缺乏对"哪种 k 维线性表示最能保留影响估计"的严格刻画，导致设计依赖启发式。
4. **可扩展性缺口**：直接对全空间梯度做 PCA 计算代价过高，亿级参数模型无法承受。

## 核心贡献（创新点）
1. **理论最优性刻画**：证明了在半白化训练梯度上做 top-k PCA 是固定维度线性表示的最优解（最小化归一化最坏情况影响误差）——与 LoGra 等随机/启发式方法本质不同，后者无理论保障。
2. **两阶段投影框架**：先用 EK-FAC 构造候选子空间（低计算成本），再在子空间内做 PCA 修正压缩基——克服了全空间 PCA 不可行的问题，理论与效率兼顾。
3. **一比特量化存储**：对投影坐标做 sign 量化 + 共享尺度，使固定预算下可保留更多坐标；实验证明对 EOGP 表示几乎无损（LDS 保持），而基线方法（LoGra、GraSS）一比特后性能显著下降。
4. **大规模实证验证**：在 GPT-2（WikiText-2）和 OLMo 2 SFT（1B–32B）上系统验证，以 <16 KB/示例 的存储超越存储 >100× 的基线，建立影响力估计压缩的新 SOTA。

## 方法详解
**目标函数**：定义归一化最坏情况影响误差
$$\mathcal{E}(V, M) = \mathbb{E}_i\left[\sup_{\nabla_\theta f \in \mathcal{B}} \left|\nabla_\theta f^\top(H_\lambda^{-1} - MV^\top)\nabla_\theta \ell_i(\theta^*)\right|^2\right]$$
其中 $\mathcal{B} = \{q : \|q\|_{H_\lambda^{-1}} \leq 1\}$ 为 $H_\lambda^{-1}$-加权单位球。

**Proposition 1（半白化 PCA 最优性）**：令 $\tilde{t}_i = H_\lambda^{-1/2}\nabla_\theta \ell_i(\theta^*)$，则最优线性压缩为 $V = M = H_\lambda^{-1/2}U$，即存储 $U^\top\tilde{t}_i$（$U$ 为 $\Sigma = \mathbb{E}[\tilde{t}_i\tilde{t}_i^\top]$ 的前 k 大特征向量）。影响估计退化为 $\langle U^\top\tilde{q}, U^\top\tilde{t}_i \rangle$。

**EOGP 两阶段投影**：
- **Stage 1（EK-FAC 子空间选择）**：利用已有 EK-FAC 分解 $G \approx Q\Lambda Q^\top$，取修正特征值最大的 m 个特征向量构成 $Q_m$，投影得到 $\tilde{t}_i^{(1)} = (\Lambda_m + \lambda I)^{-1/2}Q_m^\top\nabla_\theta \ell_i$。
- **Stage 2（子空间 PCA 修正）**：对 $\tilde{t}_i^{(1)}$ 计算其二阶矩 $\Sigma^{(1)}$ 的前 k 大特征向量 $P$，最终坐标 $\tilde{t}_i^{(2)} = P^\top\tilde{t}_i^{(1)}$。Proposition 2 证明此步的均方重建误差 ≤ 直接用前 k 个 EK-FAC 轴。

**一比特量化**（§3.3）：
$$b_{i,u} = \mathrm{sign}(\tilde{t}_{i,u}^{(2)}) \in \{\pm 1\}^{k_u}, \quad s_{i,u} = \frac{1}{k_u}\|\tilde{t}_{i,u}^{(2)}\|_1$$
查询时坐标不量化，影响估计为 $\hat{\mathcal{T}}(i) = \sum_u s_{i,u}\langle \tilde{q}_u^{(2)}, b_{i,u} \rangle$。

**变体 EOGP-R**：用 SRHT（subsampled randomized Hadamard transform）替代 Stage 2 的 PCA 矩阵 P，避免存储 $m \times k$ 修正矩阵，适合更大 k。

## 实验与结果
**数据集与模型**：
- GPT-2 on WikiText-2；OLMo 2 SFT checkpoints（1B、7B、13B、32B）on Tulu 3 mixture。

**评估指标**：
- LDS（Linear Data-Modeling Score）：影响估计与重训练结果的 Spearman 秩相关。
- 反事实重训练：移除最高影响样本后 validation perplexity 增量。
- NDCG@20 / Spearman correlation vs. EK-FAC 参考排名。

**主要结果**：
- **GPT-2**（Figure 2）：EOGP 在 96 KB 的 LDS 超过所有基线在 1,536 KB 的表现（节省 16× 存储）；在 384 KB 时 LDS 接近未压缩 K-FAC，节省 400× 存储。
- **OLMo 2 1B–32B**（Figure 4）：<16 KB/示例 时 EOGP 的 NDCG@20 和 Spearman 均显著优于 LoGra、GraSS、LoRIF，后者被分配 >100× 存储仍不及。
- **消融**（Figure 5）：PCA 修正 > 直接使用 EK-FAC 轴；一比特与 FP16 在固定坐标数下 NDCG@20 相近，证明量化几乎无损。
- **最强提升**：EOGP at 96 KB vs. LoGra (PCA) at 1,536 KB，LDS 提升显著（图 2 左）；32B 模型在 <16 KB 下 NDCG@20 接近基线 1,536 KB 水平。

**计算开销**（Table 3）：单卡 B200 上每样本构建时间 ~0.247s（32B），与 LoGra（0.278s）相当；PCA 拟合一次性开销 18.4 min（32B）。

## 相关工作脉络
1. **Koh & Liang (2017)**：原始 influence function 定义；本文在其扩展至 LLM 压缩存储方向的工作。
2. **Grosse et al. (2023)**：EK-FAC 用于 LLM 影响分析；本文复用其曲率估计，在此基础上做梯度压缩。
3. **Choe et al. (2025) LoGra**：Kronecker 因子投影 + 随机/PCA 初始化；本文在理论最优性和压缩比上超越。
4. **Hu et al. (2025) GraSS**：梯度稀疏化 + 稀疏随机投影；本文方法一比特后仍远优于 GraSS。
5. **Li et al. (2026) LoRIF**：低秩因子存储 + 截断 SVD 近似逆曲率；本文在匹配存储下 LDS/NDCG 均领先。
6. **Park et al. (2023) TRAK**：随机投影用于数据归因；本质不同——TRAK 不做曲率矫正，本文基于半白化理论最优。

## 局限性与未来方向
1. **仅针对线性层归因**：embedding、LM head、norm 参数未纳入；未来可扩展至全参数。
2. **PCA 拟合需缓存数据**：一阶坐标需约 10K 样本做 PCA fit；对流式/在线训练场景适用性待探索。
3. **一比特量化理论分析不足**：实验显示有效，但缺乏对量化误差的严格界。
4. **未评估对预训练数据选择/去毒的实际下游任务影响**：当前以重训练和排名 fidelity 为主要代理指标。
5. **共享 PCA 矩阵存储成本**：32B 模型约 481 GB；虽随样本数摊销，但对小 store 不友好。

## 研究启发与可借鉴点
1. **理论驱动压缩设计**：从"最小化最坏情况影响误差"出发推导最优表示，而非依赖启发式——可作为其他梯度压缩任务的通用设计范式。
2. **两阶段近似策略**：先用结构化近似（EK-FAC）降维，再在子空间做精确优化（PCA）——平衡了理论最优性与计算可行性，可迁移至其他大规模矩阵分解场景。
3. **一比特量化 + 尺度共享的存储技巧**：对"半白化后坐标幅度较均匀"的表示有效；可探索对其他曲率预处理后的梯度是否同样适用。
4. **重训练验证 + 排名 fidelity 双评估**：Figure 3 证明 NDCG@20/Spearman 与 LDS 高相关（ρ=0.95/0.84），为大模型场景下用排名相关性代理重训练提供了依据。
5. **可与本团队方向结合**：若团队做数据选择（data selection）或毒性数据去除，EOGP 的低存储影响函数可作为高效归因模块；SRHT 变体 EOGP-R 更适合资源受限部署。

## 关键术语表
**Influence Functions**：估计单个训练样本对模型行为（如预测、 loss）的影响程度，用于数据归因。
**EK-FAC**：Kronecker-Factored Approximate Curvature 的特征值修正版本，用于高效近似 LLM 的 GGN/Fisher 矩阵。
**Half-whitening**：用 $H_\lambda^{-1/2}$ 对梯度做预处理，使影响估计转化为内积形式，PCA 在此空间最优。
**One-bit Quantization**：仅存储梯度的符号（±1）和一个共享尺度，大幅压缩存储但保留方向信息。
**NDCG@20**：Normalized Discounted Cumulative Gain at rank 20，衡量 Top-20 影响样本排名的准确性。
**LDS（Linear Data-Modeling Score）**：用线性模型预测重训练结果，衡量影响估计与真实重训练变化的相关性。
**Counterfactual Retraining**：移除最高影响样本后重训练模型，比较 perplexity 变化作为归因质量的 ground truth。
**SRHT**：Subsampled Randomized Hadamard Transform，一种低复杂度随机投影，用于替代 PCA 修正矩阵。

## 可复现要素
- **数据集**：WikiText-2（GPT-2）、Tulu 3 SFT mixture（OLMo 2）——均为公开数据集。
- **代码/权重**：论文未明确说明开源，但使用了公开 checkpoint（allenai/OLMo-2-*、gpt2）和开源工具 Kronfluence、LogIX。
- **关键超参**：相对阻尼 $\lambda_u = 0.1 \cdot \mathrm{tr}(\Lambda_u)/d_u$；PCA 拟合集 10K 条对话、最多 2,048 tokens；EOGP 的 $m_u = 262{,}144$，$k_u \in \{256, 512, \ldots, 8192\}$。
- **硬件**：NVIDIA B200。
