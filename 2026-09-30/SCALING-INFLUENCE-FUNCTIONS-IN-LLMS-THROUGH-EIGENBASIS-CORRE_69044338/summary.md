---
title: "SCALING-INFLUENCE-FUNCTIONS-IN-LLMS-THROUGH-EIGENBASIS-CORRE"
source: https://arxiv.org/pdf/2609.37842v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:16:22"
field: "LLM可解释性与数据归因"
keywords: ["影响函数", "梯度压缩", "大语言模型", "特征基修正", "单比特量化", "可解释性"]
innovations: ["证明半白化梯度top-k PCA为线性压缩最优解", "两阶段EK-FAC+子空间PCA近似框架", "单比特量化结合缩放因子实现高保真压缩"]
benchmarks: ["GPT-2 WikiText-2 LDS", "OLMo 2 NDCG@20", "反事实重训练 perplexity increase"]
---

# 论文速读：SCALING-INFLUENCE-FUNCTIONS-IN-LLMS-THROUGH-EIGENBASIS-CORRE

## 一句话总结
本文提出EOGP（特征基修正单比特梯度投影）方法，通过两阶段线性投影与单比特量化，将LLM训练梯度压缩至原来1/16以下，同时保持影响函数估计的高保真度，使大规模模型的可复用梯度存储变得可行。

## 研究问题与动机
- **存储成本瓶颈**：8B参数模型单个训练样本的全精度梯度需16GB半精度存储，支持多次归因查询时成本无法承受
- **未来查询不可知**：压缩时无法预知后续归因查询的具体形式，需保留对多种查询均通用的信息
- **现有方法局限**：随机投影（TRAK、LoGra）和曲率感知投影（GraSS、LoRIF）在固定存储预算下难以同时兼顾方向选择与精度分配
- **理论目标缺失**：缺乏对"k维线性表示中最优压缩方向"的形式化刻画，无法指导压缩策略设计

## 核心贡献（创新点）
- **最优性证明**：在归一化最坏情况影响误差准则下，证明半白化训练梯度的top-k PCA坐标是k维线性表示中的最优解（Proposition 1）
- **两阶段近似框架**：针对LLM规模下直接PCA不可行的问题，提出先用EK-FAC降维、再在保留子空间内做PCA的两阶段近似方法（EOGP）
- **特征基修正理论保证**：证明在m个候选轴张成的子空间内做PCA修正，其重建误差小于等于直接用前k个EK-FAC轴（Proposition 2）
- **单比特量化兼容性**：结合缩放单比特量化，在固定存储预算下可保留更多坐标，EOGP在96KB下超越对比方法在1,536KB下的表现（GPT-2上16倍存储优势）
- **大模型可扩展性**：在OLMo 2（1B-32B参数）上验证，即使每样本<16KB也能与分配超100倍存储的对比方法竞争

## 方法详解
- **压缩目标函数**：定义归一化最坏情况影响误差 $\mathcal{E}(V,M) = \mathbb{E}_i[\sup_{\nabla_\theta f \in \mathcal{B}} |\nabla_\theta f^\top(H_\lambda^{-1} - MV^\top)\nabla_\theta \ell_i(\theta^*)|^2]$，其中查询梯度被限制在 $H_\lambda^{-1}$-加权单位球内
- **半白化变换**：定义 $\tilde{t}_i = H_\lambda^{-1/2}\nabla_\theta \ell_i(\theta^*)$ 和 $\tilde{q} = H_\lambda^{-1/2}\nabla_\theta f$，使影响分数转化为内积 $\langle \tilde{q}, \tilde{t}_i \rangle$
- **第一阶段（EK-FAC子空间）**：利用EK-FAC近似 $G \approx Q\Lambda Q^\top$，选取前$m$个最大修正特征值对应的特征向量构成候选子空间，投影得到 $\tilde{t}_i^{(1)} = (\Lambda_m + \lambda I)^{-1/2}Q_m^\top \nabla_\theta \ell_i(\theta^*)$
- **第二阶段（子空间PCA修正）**：对第一阶段坐标计算二阶矩 $\Sigma^{(1)} = \mathbb{E}_i[\tilde{t}_i^{(1)}(\tilde{t}_i^{(1)})^\top]$，取其前$k$大特征值对应的特征向量$P$，最终坐标为 $\tilde{t}_i^{(2)} = P^\top \tilde{t}_i^{(1)}$
- **单比特量化**：对每个模块$u$，存储符号向量 $b_{i,u} = \text{sign}(\tilde{t}_{i,u}^{(2)}) \in \{\pm1\}^{k_u}$ 和缩放因子 $s_{i,u} = \frac{1}{k_u}\|\tilde{t}_{i,u}^{(2)}\|_1$，每模块仅需$k_u$比特+2字节
- **查询评分**：对查询梯度应用相同投影但保持未量化，影响估计为 $\hat{\mathcal{T}}(i) = \sum_u s_{i,u}\langle \tilde{q}_u^{(2)}, b_{i,u}\rangle$
- **EOGP-R变体**：用隐式SRHT（子采样随机Hadamard变换）替代PCA修正矩阵，避免存储dense $P \in \mathbb{R}^{m\times k}$，适合大$k$场景

## 实验与结果
- **GPT-2重训练验证**：在WikiText-2上比较LDS和反事实重训练指标，EOGP在96KB下LDS超越所有对比方法在1,536KB下的表现；384KB时接近未压缩K-FAC参考，存储仅为162MB的1/400+
- **OLMo 2保真度评估**：在1B-32B参数SFT模型上，以EK-FAC影响排名为参考，测量NDCG@20和Spearman相关；EOGP在各模型尺寸和小存储预算下均显著超越LoGra、GraSS、LoRIF
- **存储压缩比**：32B模型在$k_u=2,048$时每样本仅需~116KB，对比未压缩半精度梯度62.4GB，压缩比达$5.4\times10^5$
- **单比特量化稳健性**：EOGP在FP16与单比特存储下NDCG@20相近，Pearson相关达0.89-0.92，显著优于LoGra/GraSS（部分配置相关<0.1或负）
- **与重训练指标的关联**：NDCG@20与LDS的Spearman相关达0.95，Spearman保真度与LDS相关0.84，验证了EK-FAC保真度作为归因质量代理的有效性

## 相关工作脉络
- **K-FAC/EK-FAC**：Martens & Grosse (2015)提出K-FAC利用层结构近似Fisher矩阵；George et al. (2018)的EK-FAC修正特征值同时保留块结构，是本文曲率近似的基础
- **LoGra**：Choe et al. (2025)使用Kronecker因子的随机或PCA初始化投影，本文证明EOGP在匹配存储下系统性优于LoGra两种初始化变体
- **GraSS**：Hu et al. (2025)结合梯度稀疏化与稀疏随机投影，与EOGP相比缺乏对投影方向的最优性理论保证
- **LoRIF**：Li et al. (2026)存储低秩因子并近似逆曲率，使用截断SVD，本文显示EOGP在小预算下更优
- **TRAK/LDS**：Park et al. (2023)提出线性数据建模评分和TRAK随机投影方法，为本文重训练验证提供评估基准
- **反事实重训练**：Bae et al. (2024)提出通过移除高影响样本并重训练来验证归因质量的协议，本文沿用该评估范式

## 局限性与未来方向
- **PCA fitting成本**：需要~10,000样本的缓存梯度进行PCA拟合，32B模型耗时18.4分钟，对超大规模存储构建构成瓶颈
- **共享矩阵存储**：EOGP的PCA修正矩阵$P_u \in \mathbb{R}^{m_u\times k_u}$按FP16存储，32B模型总计约481GB，虽不随样本数增长但绝对成本高
- **只适用于线性层**：当前仅对attention和MLP线性层（含bias）做归因，embedding、LM head和norm参数被排除
- **单比特量化假设**：缩放单比特量化假设坐标幅度相对均匀，对非均匀分布可能引入额外失真
- **固定查询分布依赖**：最优性基于训练梯度的二阶矩估计，若实际查询分布偏离训练数据分布，性能可能下降

## 研究启发与可借鉴点
- **理论指导实践**：从最坏情况误差准则推导出PCA最优性，为梯度压缩提供了明确的理论目标而非启发式规则，值得在其他表示学习任务中借鉴
- **两阶段近似策略**：先粗筛（EK-FAC）再精修（子空间PCA）的思路，平衡了计算可行性与表示质量，可扩展至其他高维投影场景
- **半白化提升量化兼容性**：理论分析和实验均表明，曲率预白化使坐标幅度更均匀，显著提升单比特量化的保真度，这一发现对量化感知训练有参考价值
- **保真度-重训练关联验证**：证明EK-FAC排名保真度与LDS高度相关（ρ=0.95），为大规模模型中用廉价代理指标替代昂贵重训练提供了依据
- **SRHT作为PCA替代**：EOGP-R用隐式SRHT避免dense修正矩阵，为存储受限场景提供了可扩展变体，可在其他需要低秩近似的任务中尝试

## 关键术语表
- **影响函数（Influence Functions）**：估计单个训练样本对模型预测或行为的影响程度的数学工具，形式化为查询梯度与训练梯度的加权和
- **EK-FAC**：Kronecker-Factored Approximate Curvature的特征值修正版本，在保留层-wise Kronecker结构的同时修正对角二阶矩，用于近似Fisher/GGN矩阵
- **半白化（Half-Whitening）**：用 $H_\lambda^{-1/2}$ 对梯度进行变换，使影响分数转化为标准内积形式，同时均衡不同曲率方向的重要性
- **单比特量化（One-bit Quantization）**：仅存储梯度坐标的符号位（±1）和共享缩放因子，将存储需求从浮点数降至每坐标1比特
- **NDCG@20**：归一化折损累积增益，衡量压缩方法在Top-20位置与参考排名的重合度，用于评估归因排名的精确性
- **LDS（Linear Data Modeling Score）**：通过线性模型预测重训练结果并计算Spearman相关，评估影响分数预测重训练outcome的能力
- **SRHT（Subsampled Randomized Hadamard Transform）**：隐式应用的随机投影矩阵，保持范数与内积的近似，避免存储dense投影矩阵
- **反事实重训练（Counterfactual Retraining）**：移除被认为高影响的训练样本后重新训练模型，通过验证损失变化验证归因质量的实验协议

## 可复现要素
- **数据集**：GPT-2使用WikiText-2（publicly available），OLMo 2使用Tulu 3 SFT mixture（allenai/tulu-3-sft-olmo-2-mixture-0225，公开）
- **模型权重**：GPT-2（openai/gpt2）、OLMo 2 1B/7B/13B/32B SFT checkpoints（allenai/OLMo-2-*, 公开）
- **代码开源**：论文未明确声明代码开源，但提到基于LogIX实现LoGra、使用Kronfluence拟合曲率统计
- **关键超参**：阻尼系数 $\lambda_u = 0.1 \cdot \text{tr}(\Lambda_u)/d_u$（相对阻尼），PCA fitting样本数10,000，序列长度截断2,048 tokens
- **实验硬件**：NVIDIA B200单卡，每样本构建时间0.056-0.247秒（EOGP vs LoGra可比）
- **评估指标**：LDS（100个子集，50个验证query）、NDCG@20、Spearman相关、反事实重训练（移除50-300样本，5个seed平均）
