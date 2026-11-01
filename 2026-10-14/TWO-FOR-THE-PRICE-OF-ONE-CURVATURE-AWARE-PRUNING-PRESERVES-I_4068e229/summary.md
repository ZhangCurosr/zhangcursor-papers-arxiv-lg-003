---
title: "TWO-FOR-THE-PRICE-OF-ONE-CURVATURE-AWARE-PRUNING-PRESERVES-I"
source: https://arxiv.org/pdf/2610.18516v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:10:24"
field: "大语言模型压缩与高效推理"
keywords: ["model pruning", "influence functions", "curvature-aware compression", "ADMM", "sparse plus low-rank", "LLM compression"]
innovations: ["CACTUS: ADMM solver for curvature-aware KFAC pruning with closed-form Sylvester solve in cached eigenbases", "Unified framework for pure pruning and sparse+low-rank decomposition under curvature-aware objective", "First demonstration that curvature-aware pruning significantly improves influence function preservation and data selection"]
benchmarks: ["C4 perplexity", "WikiText-2 perplexity", "BIG-Bench Hard accuracy", "Llama-3.2-1B/3B, Qwen3-4B, Llama-3-8B, Qwen3.8-27B"]
---

# 论文速读：TWO-FOR-THE-PRICE-OF-ONE-CURVATURE-AWARE-PRUNING-PRESERVES-I

## 一句话总结
本文提出 CACTUS，一个基于 ADMM 的曲率感知（curvature-aware）剪枝与稀疏+低秩（S+L）分解高效优化器，同时实现高质量模型压缩与影响函数（influence function）梯度保真；在 Qwen3-4B 50% 剪枝下，C4 困惑度差距较基线缩小 45%，压缩模型与稠密模型的梯度余弦相似度从 0.670 提升至 0.755，且比 LLM-Surgeon 快 28 倍以上。

## 研究问题与动机
1. **后训练剪枝的梯度盲区问题**：现有主流方法（SparseGPT、WANDA、ALPS）仅最小化层重建误差 $\|XW - X\widehat{W}\|_F^2$，只利用输入激活 $X$，完全忽略反向传播梯度，因此无法保留压缩模型与稠密模型之间的梯度对齐。
2. **曲率感知目标计算代价过高**：LLM-Surgeon 等同时利用输入激活与下游梯度，最小化 KFAC 损失 $\|X(\widehat{W}-W)A^\top\|_F^2$，但每模型需数小时 GPU 时间，限制了实际应用。
3. **影响函数在 LLM 尺度下难以使用**：全量 influence function 估计需每样本梯度与逆 Hessian 向量积，计算和存储均不可行；现有压缩代理方法（如 IProX）仅限纯低秩分解，无法覆盖结构化剪枝场景。
4. **两线程独立发展的割裂**：模型压缩与影响函数估计两条研究脉络长期各自演进，缺乏统一算法将其共同桥接。

## 核心贡献（创新点）
1. **CACTUS：面向 KFAC 曲率感知目标的高效 ADMM 求解器**——通过变量分裂将稀疏与低秩约束解耦，S-update 利用两个缓存特征基将 Sylvester 方程降为逐元素标量除法，L-update 用带权 IPSVD，整体复杂度降至 $O(d_{in}d_{out}(d_{in}+d_{out}))$ 每次迭代。
2. **统一支持纯剪枝与 (S+L) 分解**——与 LLM-Surgeon（仅纯剪枝）和 IProX（仅纯低秩）不同，CACTUS 在同一框架下同时处理两种结构约束，扩展了曲率感知压缩的适用范围。
3. **理论收敛保证**——定理 1 证明在正定 H、G 及固定有限运动界 $C_L$ 下，ADMM 迭代收敛至可行稀疏+低秩分解，连续变化与共识残差均为 $O(1/\rho^{(t-1)})$。
4. **验证了曲率感知剪枝对影响函数保真的下游收益**——在 Llama-3.2-3B 上，用 CACTUS 选出的 5% 数据微调得到 BIG-Bench Hard 49.44% 准确率，接近稠密模型选择（49.54%），显著高于 ALPS（47.22%）。

## 方法详解
**目标函数**：对每个线性层 $W \in \mathbb{R}^{d_{in} \times d_{out}}$，利用 $N_{cal}$ 条校准数据得到输入激活协方差 $H = X^\top X$ 和下游梯度协方差 $G = A^\top A$，最小化 KFAC 损失：
$$\mathcal{L}_{kfac}(\widehat{W}) = \|X(\widehat{W}-W)A^\top\|_F^2 = \text{Tr}(H(\widehat{W}-W)G(\widehat{W}-W)^\top)$$

**ADMM 变量分裂 formulation**：引入 $\Delta \in \mathcal{C}_S$ 作为稀疏副本，构造增广 Lagrangian：
$$\mathcal{L}_\rho = \text{Tr}(H(W-S-L)G(W-S-L)^\top) + \frac{\rho}{2}\|S-\Delta+V\|_F^2 - \frac{\rho}{2}\|V\|_F^2$$

**四个更新步骤**：
- **(i) S-update**：固定 L、Δ、V，对 S 求梯度为零得到广义 Sylvester 方程 $HSG + \frac{\rho}{2}S = RHS$。在 H、G 的特征基（$Q_H\Lambda_H Q_H^\top$、$Q_G\Lambda_G Q_G^\top$）下做变量替换 $S = Q_H \tilde{S} Q_G^\top$，方程逐元素解为 $\tilde{S}_{ij} = \widetilde{RHS}_{ij} / (\lambda_{H,i}\lambda_{G,j} + \rho/2)$，避免 $O((d_{in}d_{out})^3)$ 的 Kronecker 求解。
- **(ii) L-update**：固定 S、Δ、V，对 L 做带权 rank-r SVD：$L^\star = H^{-1/2} \text{SVD}_r(H^{1/2} R_S G^{1/2}) G^{-1/2}$，并用运动界 $\|L^\star - L^{(t)}\|_F \leq C_L \|S^{(t+1)} - S^{(t)}\|_F$ 控制步长。
- **(iii) Δ-update**：对 $S+V$ 做精确稀疏投影（按幅值保留 top-k 或 n:m 块内 top-n）。
- **(iv) 对偶与罚参数更新**：$\rho$ 非递减且满足 $\sum 1/\rho^{(t)} < \infty$ 以保证收敛。

**预处理**：H、G 分别加 ridge 阻尼 $H \leftarrow H + \lambda_H \text{mean}(\text{diag}(H)) I$，$G \leftarrow G + \lambda_G \text{mean}(\text{diag}(G)) I$，默认 $\lambda_H=0.01, \lambda_G=0.1$。

## 实验与结果
**模型与数据集**：Llama-3.2-1B/3B、Llama-2-7B、Llama-3-8B、Qwen3-4B、Qwen3.8-27B；校准集为 C4 前 shard 的 128 条序列（长度 2048）；评估 PPL 在 C4 和 WikiText-2（长度 2048）上。

**关键结果**：
- **Qwen3-4B 50% 剪枝**（Table 2）：CACTUS-prune 取得 C4 PPL=21.85，较 LLM-Surgeon(5sh)（23.50）缩小困惑度差距约 45%；$\cos_{te}$（TracIn）从 0.670 提升至 0.755；运行时间 42m18s vs. LLM-Surgeon 20h13m（2×A100），快约 28×，单卡 L40 即可。
- **Llama-3.2-1B 50% 剪枝**（Table 7）：CACTUS-prune C4 PPL=19.43，较 LLM-Surgeon(5sh)（21.39）差距缩小约 25%；$\cos_{tr}$（TracIn）从 0.445（LLM-Surgeon）提升至 0.524。
- **Qwen3-4B (S+L)**：CACTUS-SLR 在 50% 压缩下 C4 PPL=21.38，优于 3BASiL（21.63）和 OATS（23.97）。
- **Llama-3-8B (2:4+r256)**（Figure 3/Table 11）：CACTUS-SLR TracIn $\rho_{qry}=0.236$，较 3BASiL（0.170）提升约 38%。
- **数据选择（Table 3）**：CACTUS-prune 在 Llama-3.2-3B 上选出 5% 数据微调，BBH EM=49.44%，与稠密 LESS 选择（49.54%）几乎持平，高于 ALPS（47.22%）；与稠密选集的 overlap 达 54.13%。
- **LoGRA 集成（Figure 4）**：Qwen3-4B MLP 层 2:4 剪枝+rank-64 校正下，CACTUS-SLR 保留 $\cos_{tr}=0.711$ 的 LoGRA 投影余弦对齐。
- **Qwen3.8-27B 2:4 剪枝（Table 14）**：CACTUS-prune C4 PPL=13.42 vs. SparseGPT 14.17，困惑度差距缩小超 25%。

## 相关工作脉络
1. **SparseGPT / WANDA / ALPS**（层重建剪枝）：仅用激活协方差 $H$，目标为 $\|X(W-\widehat{W})\|_F^2$，梯度盲；CACTUS 在此基础上引入梯度协方差 $G$，形成双边曲率感知目标。
2. **LLM-Surgeon**（曲率感知纯剪枝）：同样最小化 KFAC 损失，但用粗糙的权重选择+相关校正多轮迭代；CACTUS 用 ADMM 统一求解，速度快一个数量级以上。
3. **IProX**（影响保真低秩代理）：用 IPSVD 两阶段构造低秩代理以保梯度；CACTUS 将 IPSVD 作为 ADMM 的一个子步，同时覆盖纯剪枝和 (S+L)。
4. **3BASiL / OATS**（S+L 分解）：基于激活协方差的 sparse-plus-low-rank 方法；CACTUS-SLR 在相同框架下引入梯度协方差，PPL 和 IF 指标双优。
5. **TRAK / LoGRA**（梯度投影）：方向 (ii) 的效率方法，CACTUS 与其兼容，分别用 TRAK JL 散列和 LoGRA 架构感知投影验证 IF 保真。
6. **YAQA**（曲率感知量化）：概念上更接近 LLM-Surgeon 的错误补偿思路，而非本文的 ADMM 分裂策略。

## 局限性与未来方向
1. **仅覆盖剪枝与 (S+L) 分解**，未扩展到量化或混合量化-稀疏设置；若涉及重参数化（如旋转），梯度几何会变化，ADMM 框架需重新设计。
2. **均匀压缩预算**：所有层使用统一预算，未研究非均匀分配稀疏度和 rank 的自适应策略。
3. **无后压缩微调**：未涉及全模型或 LoRA 风格的 post-compression fine-tuning，可能在不同维度权衡部署质量与梯度保真。
4. **校准集规模固定**：实验使用 128 条序列，更多数据有小幅改善（Table 15），但超大规模校准的成本未深入探索。

## 研究启发与可借鉴点
1. **Sylvester 方程的特征基解法是高效曲率感知优化的关键技巧**：将双边耦合 $(H S G)$ 通过预计算特征分解降为逐元素除法，复杂度从 $O((dm)^3)$ 降至 $O(dm(d+m))$，可迁移到任何含双边二次型的矩阵优化问题。
2. **KFAC 损失作为"桥梁目标"同时服务两个研究子领域**：用同一目标连接模型压缩质量与影响函数保真，避免了分阶段设计的精度损失，这种统一视角值得在其他代理任务中借鉴。
3. **运动界约束（movement bound）控制低秩更新步长**：公式 (8) 的 $C_L$ 阈值机制防止 ADMM 迭代中低秩分量剧烈震荡，在类似多块交替优化中可作为稳定化正则化手段。
4. **LoGRA 探针与 (S+L) 分解的拼接构造**（Concat-LoGRA）：将 SVD 因子拼接到随机投影矩阵中，使压缩模型的梯度评估与稠密模型在同一致射空间可比，这一接口设计可直接复用于其他结构化压缩方法的影响函数评估。
5. **ridge 阻尼参数的鲁棒性**（Table 4）：在较大范围内 $\lambda_H, \lambda_G$ 的变化对 PPL 影响较小，表明该方法对超参不敏感，降低了工程落地门槛。

## 关键术语表
**CACTUS**：本文提出的基于 ADMM 的曲率感知模型压缩算法，同时支持纯剪枝和稀疏+低秩（S+L）分解。
**KFAC loss**：Kronecker-Factored Approximate Curvature 损失，形式为 $\|X(\widehat{W}-W)A^\top\|_F^2$，同时利用输入激活和下游梯度协方差的双边曲率感知目标。
**Influence Function (IF)**：影响函数，估计训练样本对模型预测的(counterfactual)影响，形式为 $\nabla\ell(z')^\top H^{-1}\nabla\ell(z)$。
**TracIn**：一种影响函数近似，使用原始 per-sample 梯度向量做内积比较，不经 Hessian 加权。
**LoGRA**：Low-rank Gradient Approximation，通过在每个线性层附加低秩随机探针，在不物化全量 per-sample 梯度情况下高效估计影响。
**S+L (Sparse + Low-rank)**：稀疏+低秩分解，将压缩权重表示为 $\widehat{W}=S+L$，其中 S 满足稀疏约束，L 为低秩修正项。
**ADMM**：Alternating Direction Method of Multipliers，一种将复杂约束优化分解为多个易解子问题的交替方向乘子法。
**CAL**：Calibration（校准）数据集，用于收集激活值和梯度协方差的小规模数据子集（本文为 C4 前 shard 128 条序列）。

## 可复现要素
- **数据集**：C4（Calibration，第一 shard）、WikiText-2（评估 PPL）、BIG-Bench Hard（评估数据选择）、FLAN-v2/CoT/Dolly/OpenAssistant 混合（270,679 条 LESS 候选集）。
- **模型**：Llama-3.2-1B/3B、Llama-2-7B、Llama-3-8B、Qwen3-4B、Qwen3.8-27B（公开权重）。
- **代码/权重开源**：论文未明确声明代码开源状态（有 ArXiv 链接，作者在 AI Use Statement 中提及代码开发有 AI 辅助）。
- **关键超参**：$\lambda_H=0.01$、$\lambda_G=0.1$、$C_L=10$（S+L）、$\eta=0.3$（低秩预算比例）、初始 $\rho_0=0.1$、投影维度 $d=8192$、校准序列数 128、序列长度 2048。
