---
title: "TWO-FOR-THE-PRICE-OF-ONE-CURVATURE-AWARE-PRUNING-PRESERVES-I"
source: https://arxiv.org/pdf/2610.18516v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:10:35"
field: "大语言模型高效压缩与可解释性"
keywords: ["LLM 剪枝", "曲率感知压缩", "影响函数保留", "ADMM", "稀疏+低秩分解", "CACTUS", "LLM-Surgeon", "LoGRA"]
innovations: ["提出 CACTUS 算法，以 ADMM 高效优化 KFAC 曲率感知目标，同步提升剪枝质量与影响函数保留", "通过 Sylvester 方程特征基闭式求解，将曲率感知优化速度提升至 LLM-Surgeon 的 28 倍以上", "将曲率感知压缩代理用于数据选择任务，在 BIG-Bench Hard 上达到接近密集模型的效果"]
benchmarks: ["C4 perplexity", "WikiText-2 perplexity", "BIG-Bench Hard accuracy", "TracIn/KFAC cosine similarity", "LoGRA-projected gradient alignment"]
---

# 论文速读：TWO-FOR-THE-PRICE-OF-ONE-CURVATURE-AWARE-PRUNING-PRESERVES-I

## 一句话总结
本文提出 **CACTUS** 算法，通过 ADMM 优化曲率感知目标（同时利用层激活与反向传播梯度），在 LLM 后训练剪枝及稀疏+低秩分解中，以远低于 LLM‑Surgeon 的计算代价，同步提升压缩模型的下游语言质量（PPL）与对密集模型影响函数的保留能力。

## 研究问题与动机
- 现有主流剪枝方法（SparseGPT、ALPS 等）仅利用输入激活协方差进行层 wise 重建，**对下游梯度盲视**，导致压缩后模型性能下降且无法保留影响函数所需的梯度对齐信息。
- 曲率感知目标（LLM‑Surgeon、IProX 等所采用）能同时改善模型质量与影响函数保留，但其优化过程通常需要多 GPU 小时，**计算开销过大，难以普及**。
- 在 LLM 尺度上直接计算影响函数（需 per‑sample 梯度与 Hessian 逆）成本极高；虽然已有 IProX 等压缩代理思路，但仅局限于低秩分解，**未覆盖纯剪枝与稀疏+低秩联合场景**。
- 需要一个**统一、高效**的优化框架，使曲率感知目标能同时服务于高质量剪枝与影响函数保留，实现“一体两得”。

## 核心贡献（创新点）
1. **提出 CACTUS 算法**：基于 ADMM 的曲率感知压缩求解器，同时支持纯剪枝（CACTUS‑prune）与稀疏+低秩分解（CACTUS‑SLR），提供了收敛性保证。与 LLM‑Surgeon 等同类方法相比，核心区别在于**通过 Sylvester 方程的闭式特征基求解，将两步耦合目标降至矩阵乘法量级**。
2. **计算效率大幅提升**：在 Qwen3‑4B 50% 剪枝设置下，CACTUS‑prune 的运行速度比 LLM‑Surgeon 快 **28× 以上**，且仅需单张 L40 GPU，而非多台 A100。
3. **压缩质量与影响函数保留双优**：在 Llama‑3.2‑1B 50% 压缩下，CACTUS 将 C4 PPL 差距相对最强基线缩小 25%（纯剪枝）/28%（S+L）；在 WikiText‑2 上将压缩‑密集 per‑sample 梯度余弦相似度从 0.445 提升至 **0.512**。
4. **影响函数保留的直接下游收益**：利用 CAIUS‑prune 模型选出 5% 训练数据，微调 fresh Llama‑3.2‑3B 后在 BIG‑Bench Hard 上达到 **49.44%** 准确率，接近密集模型选数据的 49.54%，且与密集选择的重合度达 54.13%，优于 ALPS 的 47.22%。
5. **与 LoGRA 等梯度投影策略无缝集成**：在 Qwen3‑4B 的 MLP 层应用 2:4 稀疏 + rank‑64 修正，使用相同 LoGRA 投影时，压缩模型与密集模型的 C4 梯度余弦对齐仍保持 **0.711**，验证了方法的通用兼容性。

## 方法详解
- **曲率感知目标（KFAC 损失）**：对每一线性层权重 \(\mathbf{W}\)，收集校准数据的输入激活 \(\mathbf{X}\) 与下游反向传播梯度 \(\mathbf{A}\)，构造协方差 \(\mathbf{H}=\mathbf{X}^\top\mathbf{X}\)、\(\mathbf{G}=\mathbf{A}^\top\mathbf{A}\)，最小化  
  \(\mathcal{L}_{\text{kfac}}(\widehat{\mathbf{W}}) = \|\mathbf{X}(\widehat{\mathbf{W}}-\mathbf{W})\mathbf{A}^\top\|_F^2\)。  
  该目标同时捕捉激活域与梯度域的曲率信息。
- **约束类型**：① 纯剪枝（无结构稀疏或 n:m 结构稀疏）；② 稀疏+低秩分解 \(\widehat{\mathbf{W}}=\mathbf{S}+\mathbf{L}\)，其中 \(\mathbf{S}\) 满足稀疏约束，\(\mathbf{L}\) 满足秩约束。
- **ADMM 框架**：引入辅助变量 \(\mathbf{\Delta}\)（稀疏副本）与 \(\mathbf{S}\)（光滑副本），构建增广拉格朗日函数，交替更新四个块：
  1. **S‑update**：求解广义 Sylvester 方程 \(\mathbf{H}\mathbf{S}\mathbf{G}+\frac{\rho}{2}\mathbf{S}=\mathbf{RHS}\)。通过在 \(\mathbf{H}\)、\(\mathbf{G}\) 的预计算特征基 \(\mathbf{Q}_H\)、\(\mathbf{Q}_G\) 中对角化，将矩阵方程转化为逐元素除法，复杂度为 \(O(d_{in}d_{out}(d_{in}+d_{out}))\)，避免不可行的 Kronecker 直接求解。
  2. **L‑update**：对剩余部分 \(\mathbf{R}_S=\mathbf{W}-\mathbf{S}\) 作加权秩‑r 近似，闭式解为 \(\mathbf{L}^*= \mathbf{H}^{-1/2}\text{SVD}_r(\mathbf{H}^{1/2}\mathbf{R}_S\mathbf{G}^{1/2})\mathbf{G}^{-1/2}\)，并施加重移动约束保证稳定性。
  3. **Δ‑update**：对 \(\mathbf{S}+\mathbf{V}\) 执行精确稀疏投影（保留 top‑k 元素或每 m 个元素中 top‑n）。
  4. **对偶更新**：缩放对偶变量 \(\mathbf{V}\)，惩罚参数 \(\rho\) 按支持集变化间隔单调递增，满足收敛所需的求和条件。
- **预处理**：对 \(\mathbf{H}\)、\(\mathbf{G}\) 分别施加 Ridge 阻尼（\(\lambda_H=0.01\)，\(\lambda_G=0.1\) 为默认）改善条件数；所有层的 \(\mathbf{G}\) 在一次前向‑反向传递中预计算并存储于 CPU，\(\mathbf{H}\) 随层序构建。
- **收敛性**：定理 1 证明，在正定 \(\mathbf{H}\)、\(\mathbf{G}\)、精确光滑求解与有界移动约束下，迭代序列收敛至可行稀疏+低秩分解，残差以 \(O(1/\rho^{(t-1)})\) 衰减。

## 实验与结果
- **数据集**：C4（校准与 PPL 评估）、WikiText‑2（PPL 与梯度评估）、BIG‑Bench Hard（BBH，数据选择下游评估）。
- **基线方法**：纯剪枝（SparseGPT、ALPS、LLM‑Surgeon）；纯低秩（SVD‑LLM(W)、IPSVD、IProX）；稀疏+低秩（OATS、3BASiL）。
- **主要结果（Qwen3‑4B，50% 压缩）**：
  - CACTUS‑prune：C4 PPL 21.85（密集 19.85），WT‑2 PPL 16.27；TracIn \(\cos_{tr}=0.762\)，\(\cos_{te}=0.755\)；运行时间 42m 18s（L40），比 LLM‑Surgeon(5sh)（20h 13m，双 A100）快 **28×**。
  - CACTUS‑SLR：C4 PPL 21.38，WT‑2 PPL 15.59；TracIn \(\cos_{tr}=0.782\)，\(\cos_{te}=0.790\)。
  - 相对最强基线，C4 PPL 差距缩减 **45%**，梯度余弦相似度从 0.670 提升至 **0.755**。
- **数据选择实验（Llama‑3.2‑3B）**：使用 LESS 流程，由剪枝代理选出 5% 指令数据微调新密集模型。CACTUS‑prune 选出的子集与密集选择重合 **54.13%**，BBH 准确率达 **49.44%**，高于 ALPS 的 47.22% 与密集模型自身的 49.54%（几乎持平）。
- **与 LoGRA 集成**：Qwen3‑4B 的 MLP 层应用 2:4 稀疏 + rank‑64 修正，使用相同 LoGRA 投影时，压缩模型的 TracIn \(\cos_{tr}=0.711\)，显著优于其他剪枝方法，且 \(\rho_{qry}=0.804\)、\(R_{qry@5\%}=0.597\)。
- **跨模型/压缩率**：在 Llama‑3.2‑1B（50%）、Llama‑3‑8B（2:4+r256）、Qwen3.8‑27B（2:4）等设置下，CACTUS 均取得最优或接近最优的 PPL 与梯度保留指标，且运行时间大幅低于多 shot 的 LLM‑Surgeon。

## 相关工作脉络
- **层 wise 重建剪枝**（SparseGPT、ALPS、3BASiL 等）：仅使用输入激活协方差 \(\mathbf{H}\)，梯度盲；CACTUS 通过引入 \(\mathbf{G}\) 将目标升级为曲率感知，提升梯度对齐能力。
- **曲率感知压缩**（LLM‑Surgeon、YAQA）：同样采用 KFAC 损失，但 LLM‑Surgeon 依赖多次权重补偿迭代，耗时长；YAQA 面向量化；CACTUS 以 ADMM 统一求解剪枝与 S+L，效率与适用性更强。
- **影响函数压缩代理**（IProX）：仅构造低秩代理（IPSVD+梯度对齐微调）；CACTUS 将同一曲率目标拓展至纯剪枝与 S+L，并直接作为代理无需额外微调。
- **高效影响函数估计**（TRAK、LoGRA、Arnoldi‑IF）：关注梯度投影降维；CACTUS 与之正交且兼容，可在不同投影（TRAK/JL、LoGRA）下验证保留效果。
- **稀疏+低秩分解**（OATS、3BASiL、Hassle‑free）：利用激活二阶统计量联合优化稀疏与低秩分量；CACTUS 在同等框架上叠加梯度协方差 \(\mathbf{G}\)，使分解同时服务质量与梯度保留。

## 局限性与未来方向
- **仅限剪枝与 S+L**：未推广至量化（尤其是整数量化）或量化‑稀疏混合场景，扩展时需处理重参数化带来的梯度几何变化。
- **均匀压缩预算**：各层分配相同的稀疏/秩比例，未来可探索非均匀预算分配策略。
- **无压缩后微调**：未研究 full‑model 或 LoRA 风格的压缩后微调，该类微调可能以不同方式权衡部署质量与梯度对齐保真度。
- **校准数据量**：默认使用 128 条 C4 序列；更多序列可轻微提升 PPL，但计算成本增加，实际中需权衡。

## 研究启发与可借鉴点
- **曲率感知目标可作为统一桥梁**：同一 KFAC 损失可同时优化下游语言质量与影响函数保留，为剪枝算法的设计提供了兼具性能与可解释性的新视角。
- **Sylvester 方程的特征基闭式求解**：将两步耦合优化转化为对角空间中的逐元素操作，是加速曲率感知 ADMM 的关键技巧，可迁移至其他涉及 \(\mathbf{H}\mathbf{W}\mathbf{G}\) 型目标的压缩问题。
- **影响函数保留的下游验证闭环**：不仅报告梯度相似度，还通过数据选择任务（BBH 准确率）证明保留能力的实际价值，为方法评估提供了更全面的范式。
- **与 LoGRA 等投影策略的即插即用兼容**：算法不依赖特定投影，只需更换投影探针即可适配不同 influence‑function 估算流水线，提高了方法的通用性。
- **轻量化硬件基准测试**：全部运行时间以单张 L40 GPU 为标准报告，并明确标注 LLM‑Surgeon 的双 A100 配置，使效率对比更透明、可复现。

## 关键术语表
- **CACTUS**：基于 ADMM 的曲率感知剪枝/稀疏‑低秩分解算法，通过 Sylvester 方程闭式求解实现高效优化。
- **KFAC 损失**：\(\|\mathbf{X}(\widehat{\mathbf{W}}-\mathbf{W})\mathbf{A}^\top\|_F^2\)，同时惩罚激活域与梯度域的误差，近似 Hessian 二阶信息。
- **影响函数（Influence Function）**：估计训练样本对模型预测的反事实影响，形式为 \(\nabla\ell(z')^\top \mathbf{H}^{-1}\nabla\ell(z)\)。
- **ADMM**：交替方向乘子法，将原问题拆分为多个可闭式求解的子问题并交替迭代。
- **稀疏+低秩分解（S+L）**：将权重矩阵分解为稀疏主干 \(\mathbf{S}\) 与低秩修正 \(\mathbf{L}\) 之和，以兼顾 sparsity 与 expressiveness。
- **LoGRA**：低秩梯度近似策略，为每层附加随机投影适配器，在不显式计算全梯度矩阵的前提下估计 per‑sample 梯度。
- **LESS**：基于影响函数的数据选择流水线，利用代理模型选择对下游任务最有价值的训练样本。
- **TracIn / KFAC 表示**：两种 per‑sample 梯度表示方式，前者使用原始梯度向量，后者用 \(\mathbf{H}^{-1/2}\nabla\mathbf{G}^{-1/2}\) 加权，更贴近 KFAC 近似下的内积。

## 可复现要素
- **数据集**：C4（公开）、WikiText‑2（公开）、BIG‑Bench Hard（公开）；校准数据为 C4 前 128 条序列（长度 2048）。
- **代码/权重**：论文未明确声明开源仓库；模型权重为公开 LLM（Llama、Qwen 系列）。
- **关键超参**：Ridge 阻尼 \(\lambda_H=0.01\)、\(\lambda_G=0.1\)；ADMM 初始惩罚 \(\rho_0=0.1\)；低秩移动约束常数 \(C_L=10\)；S+L 秩分配比例 \(\eta=0.3\)；TRAK 投影维度 \(d=8192\)；校准序列数 128；评估序列长度 512（影响函数）/2048（PPL）。
