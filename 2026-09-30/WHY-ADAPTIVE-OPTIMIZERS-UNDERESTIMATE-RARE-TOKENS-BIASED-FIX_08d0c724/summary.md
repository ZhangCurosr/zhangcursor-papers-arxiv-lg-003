---
title: "WHY-ADAPTIVE-OPTIMIZERS-UNDERESTIMATE-RARE-TOKENS-BIASED-FIX"
source: https://arxiv.org/pdf/2609.37535v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:54:58"
field: "优化算法理论分析"
keywords: ["adaptive optimizers", "rare tokens", "fixed point bias", "softmax output layer", "Adam", "RMSProp", "conservation laws", "language models"]
innovations: ["揭示了坐标自适应优化器在 softmax 层导致稀有 token 固定点偏置的机制", "建立了输出嵌入均值守恒性的理论框架并分类了优化器", "提出了量化偏置的参数 κ 并推导了 RMSProp 固定点的闭式解"]
benchmarks: ["Unigram model with Zipf frequencies", "Small language model on Markov chain"]
---

# 论文速读：WHY ADAPTIVE OPTIMIZERS UNDERESTIMATE RARE TOKENS

## 一句话总结
本文揭示了常见坐标自适应优化器（如 Adam、RMSProp）在 Softmax 输出层对稀有 token 存在系统性低估的机制，并通过理论推导与实验验证，量化了该偏置与数据频率、学习率、动量参数及批量大小之间的关系。

## 研究问题与动机
- **核心问题**：在语言模型训练中，token 频率呈重尾分布，而主流优化器（如 Adam）对每个坐标使用独立的二阶矩估计进行归一化。这种机制如何影响稀有 token 的 logits 学习动态？
- **现有方法的不足**：
  1. 坐标自适应优化器（Adam 等）的二阶矩估计在 token 出现后最大，随后在漫长的间隔期内衰减，导致出现后的正向更新被强烈抑制，而间隔期内的负向更新被逐步放大，造成固定点偏移。
  2. 尽管“输出嵌入平均偏移”（common shift）现象已被研究（如 Stollenwerk & Stollenwerk, 2025），但针对**单个稀有 token 预测概率的局部固定点偏置**缺乏理论刻画。
  3. 现有修复方法（如 Coupled Adam、AMSGrad）多基于经验观察，缺乏统一的理论解释框架。

## 核心贡献（创新点）
1. **建立了输出层均值嵌入守恒性的理论框架**：证明了更新为历史梯度线性组合的优化器（SGD、Momentum、Shampoo、Muon、Coupled Adam）均守恒输出嵌入均值；而坐标自适应方法（Adam、RMSProp、Adafactor、Sign descent、Lion）会破坏该守恒律。
2. **揭示了稀有 token 的有偏固定点机制**：从理论上证明了对于出现频率低于一半 minibatch 的 token，Sign descent 会以恒定速率降低其 logit；对于 RMSProp，推导出周期性到达情况下固定点的闭式解，证明其平衡概率严格低于数据频率。
3. **提出了无偏固定点的统一视角**：证明了 SGD 和 AMSGrad（以及 Coupled Adam）在相同 unigram 模型下能保持无偏固定点，为理解不同优化器的行为差异提供了清晰基准。
4. **提供了定量化的分析工具 κ**：引入参数 $\kappa = (1 - \beta_2)/(q_i B)$，将稀有程度、动量衰减率和批量大小统一到一个度量中，并推导出 RMSProp 固定点与 κ 的渐近关系 $\rho(\kappa) = \kappa / (2(e^{\kappa/2} - 1))$。

## 方法详解
- **理论模型**：采用 unigram 模型简化分析，仅保留输出 bias 参数 $b \in \mathbb{R}^V$，预测为 $p(b) = \text{softmax}(b)$。在每一步，从分布 $q$ 中独立采样 minibatch，若 token $i$ 出现 $c_{t,i}$ 次，则梯度 $g_{t,i} = p_i - c_{t,i}/B$。
- **守恒性分析（Proposition 1 & 2）**：
  - 对于更新形式为 $U_t = \sum_{s \leq t} \alpha_{t,s} G_s$ 的方法（SGD、momentum），由于 $\mathbf{1}^\top G_s = 0$，均值嵌入 $\bar{\mathbf{w}}$ 守恒。
  - 对于坐标自适应方法（Adam、RMSProp），Proposition 2 给出了每步均值变化的精确公式：$\bar{\mathbf{w}}_{t+1,j} - \bar{\mathbf{w}}_{t,j} = -\eta_t \text{Cov}_i(d_{t,ij}, M_{t,ij})$，其中协方差项源于稀有 token 的正动量与其大缩放因子的正相关性。
- **有偏固定点分析（Theorem 3 & 4）**：
  - **Sign descent**：对于出现概率 $\pi_i < 1/2$ 的 token，每个时间步的期望 logit 变化 $\mathbb{E}[\Delta b_{t,i}] \leq -\eta(1 - 2\pi_i)$，以恒定速率下降。
  - **RMSProp（周期性到达）**：假设 token 每 $N$ 步出现一次，推导出二阶矩估计 $v_t$ 的周期性序列，并证明固定点 $p^* < q_i$，且比值 $p^*/q_i = N/x^*$ 仅依赖于 $N$ 和 $\beta_2$，与学习率 $\eta$ 无关。
  - **渐近行为**：当 $N \to \infty$ 且 $\beta_2 = 1 - \kappa/N$ 时，$p^*/q_i \to \rho(\kappa) = \kappa/(2(e^{\kappa/2} - 1))$，该函数随 $\kappa$ 单调递减。
- **关键参数 $\kappa$**：定义为 $\kappa = N(1-\beta_2) = (1-\beta_2)/(q_i B)$，表征 token 两次出现之间的平均步数与二阶矩时间常数 $1/(1-\beta_2)$ 的比值。$\kappa > 1$ 时偏置显著。

## 实验与结果
- **数据集与模型**：
  1. **守恒性测试**：Softmax 回归，$V=512$ 类（128 类永不出现），$d=32$，$B=64$，float64 精度，300 步。
  2. **单 token 与 unigram 模型**：$V=5000$ 个 token，Zipf 频率 $q_i \propto i^{-1.2}$，初始化于无偏点 $b = \log q$，运行 $3\times10^5$ 步。
  3. **小型语言模型**：基于 2048 token 的一阶 Markov 链生成数据，$V=4096$（一半永不出现），embedding 宽 64，残差 MLP（隐藏层 256），untied 输出层，$B=256$，训练 $2\times10^4$ 步。
- **主要结果**：
  1. **守恒性（Table 1）**：SGD、Heavy ball、Nesterov、Shampoo、Muon、Coupled Adam 的 $\|\Delta \bar{\mathbf{w}}\|_2$ 和 $|\Delta \bar{b}|$ 均在 $10^{-15}$ 量级（数值误差）；而 Adam、RMSProp、AMSGrad、Adafactor、Lion、Sign descent 的偏移量在 $0.1-1.4$ 量级。
  2. **单 token 周期性到达**：RMSProp 实测固定点与 Theorem 4(iii) 闭式解吻合，最大对数偏差仅 0.009。
  3. **Unigram 模型随机到达（Figure 1b）**：对于 $\kappa \lesssim 1$，RMSProp 和 Adam 的偏置遵循理论曲线 $\log \rho(\kappa)$；当 $\kappa \geq 2$ 时，实测偏置（平均约 -5.88）远大于理论预测，表明周期性公式在随机到达且 $\kappa$ 较大时低估了偏置。
  4. **小型语言模型（Table 2）**：
     - $\beta_2 = 0.95$ 时，Adam、RMSProp、AdamW 对稀有 token（$e_y < 0.05$）的平均对数偏置分别为 -2.72、-2.65、-2.25，而 SGD 为 -0.22（几乎无偏）。
     - 将 $\beta_2$ 提高至 0.999 可将 Adam 的偏置降至 -0.06。
     - 偏置与 KL 散度正相关：$\beta_2=0.95$ 时 Adam/RMSProp 的 KL（0.2278/0.2190）显著高于 SGD（0.0940）、AMSGrad（0.1241）和 Coupled Adam（0.1238）。
- **最强结果与提升**：使用 Coupled Adam（共享词汇表二阶矩）或提高 $\beta_2$ 至 0.999 可大幅缓解偏置，使 RMSProp/Adam 的稀有 token 偏置从 -2.7 量级降至 -0.06 量级，KL 散度相应降低约 0.09-0.10。

## 相关工作脉络
1. **Symmetries and conservation laws (Kunin et al., 2021)**：本文扩展了其守恒律分析至 Kronecker-factored（Shampoo）和正交化（Muon）优化器，明确了它们属于守恒类。
2. **Output embedding shift (Gao et al., 2019; Bis et al., 2021; Stollenwerk & Stollenwerk, 2025)**：先前工作关注“共同偏移”现象及其对 embedding 退化的影响；本文揭示的**局部 fixed-point bias** 是该现象的深层原因之一，且指出中心化（centering）无法消除对稀有 token 的偏置。
3. **Coupled Adam (Stollenwerk & Stollenwerk, 2025)**：经验性提出通过共享二阶矩 across vocabulary 来稳定训练；本文理论证明该方法（更新形式 d）能守恒均值嵌入，并解释其为何能避免有偏固定点。
4. **Output logit divergence (Wortsman et al., 2024; Stollenwerk et al., 2026)**：观察到 LLM 训练中的 logit 发散现象；本文机制为其提供了优化器层面的解释，并建议通过高 $\beta_2$ 或 Coupled Adam 缓解。
5. **Adaptive methods for rare classes (Kunstner et al., 2024; Land & Bartolo, 2024)**：指出自适应优化在重尾类别不平衡下的优势；本文补充表明，这种“更快进展”可能以牺牲稀有 token 的校准概率为代价。
6. **Non-convergence of Adam (Reddi et al., 2018)**：AMSGrad 的动机；本文证明在 rare token 的梯度模式（间歇性大梯度）下，AMSGrad 确实能保持无偏固定点，从另一角度验证其有效性。

## 局限性与未来方向
- **理论简化**：unigram 模型假设 token 出现独立同分布且 log-partition 恒定，忽略了真实语言模型中上下文依赖、token 共现模式及梯度相关性。
- **周期性假设**：Theorem 4 的闭式解依赖周期性到达，随机到达下（尤其 $\kappa > 1$）偏置更大且无法精确刻画。
- **小规模实验**：语言模型实验仅为小型 proxy model（embedding 64，MLP 隐藏层 256），结论在大规模 LLM 中的可推广性待验证。
- **未来方向**：
  1. 将分析扩展至 tied embeddings 场景，此时共同偏移会影响损失。
  2. 研究在预训练（百万级 batch size，$\kappa$ 极小）vs. 微调（小 batch，$\kappa$ 较大）中偏置程度的差异。
  3. 探索该偏置对下游罕见词生成、长尾分类校准及模型整体 perplexity 的实际影响。

## 研究启发与可借鉴点
1. **守恒律分析作为优化器设计的指导原则**：可将“输出嵌入均值守恒”作为验证新优化器在 softmax 层行为是否合理的理论基准，避免引入系统性偏置。
2. **参数 κ 的统一诊断价值**：该参数将频率、批量、动量系数统一，可用于快速评估任意 token 在给定训练配置下可能面临的偏置严重程度，为超参选择（如 $\beta_2$、batch size）提供理论依据。
3. **实验设计借鉴**：使用已知生成分布的小型模型评估优化器行为，通过测量与生成分布的 KL 散度来量化偏置的下游影响，这一范式可直接迁移至其他序列建模任务（如机器翻译）中优化器的公平性评估。
4. **修复策略的对比实验**：论文系统对比了 AMSGrad、Coupled Adam、高 $\beta_2$、SGD 等多种缓解方案，并给出了定量排序，这种全面对比为后续研究提供了清晰的基线参考。
5. **低精度训练的影响**：论文提及有限精度会破坏零和梯度性质，可启发研究者将固定点偏置分析与低精度训练（如混合精度、量化）相结合，探索更鲁棒的训练策略。

## 关键术语表
- **Adaptive Optimizers (Adam, RMSProp)**：为每个参数坐标维护独立的二阶矩估计，并以此归一化梯度更新的优化器家族。
- **Biased Fixed Point**：由于优化器更新规则的不对称性（稀有 token 出现后二阶矩最大），导致模型预测概率无法收敛至数据真实频率的稳定状态。
- **Conservation Law (Mean Output Embedding)**：某些优化器（如 SGD、Shampoo）在更新过程中保持所有 token 输出嵌入的均值不变，源于 softmax 梯度的零和性质。
- **Kronecker-Factored Methods (Shampoo)**：使用梯度外积的 Kronecker 因子分解来近似二阶信息，其更新形式能守恒输出嵌入均值。
- **Orthogonalized Methods (Muon)**：对动量矩阵进行正交化（如极分解）的优化器，同样属于守恒类。
- **Parameter κ**：定义为 $\kappa = (1-\beta_2)/(q_i B)$，综合衡量 token 稀有程度（$q_i$）、批量大小（$B$）和二阶矩记忆长度（$1-\beta_2$）的关键无量纲参数。
- **Unigram Model**：假设 token 独立同分布出现的简化概率模型，用于隔离分析单个 token 的学习动态。
- **Output Embedding Centering**：从输出权重矩阵的每一行减去其均值，以抵消共同偏移现象的技术。

## 可复现要素
- **数据集**：合成数据（未公开第三方数据集）；unigram 模型使用 Zipf 分布 $q_i \propto i^{-1.2}$；语言模型基于自定义的一阶 Markov 链生成。
- **代码/权重**：论文声明所有实验代码已作为补充材料提供（`run experiments.py`），在 CPU 上运行总时间不超过两小时。权重文件未提及开源。
- **关键超参**：
  - Unigram 模型：$V=5000$，学习率 $\eta=4\times10^{-3}$（自适应方法），$\epsilon=10^{-12}$，训练步数 $3\times10^5$。
  - 语言模型：$V=4096$，$B=256$，embedding 宽度 64，MLP 隐藏层 256，warmup 200 步后恒定学习率 $\eta=3\times10^{-3}$（$\beta_1=0.9$，$\epsilon=10^{-8}$），AdamW 权重衰减 $\lambda=0.1$。
