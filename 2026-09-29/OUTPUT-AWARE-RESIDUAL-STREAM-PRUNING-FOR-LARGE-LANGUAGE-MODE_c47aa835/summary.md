---
title: "OUTPUT-AWARE-RESIDUAL-STREAM-PRUNING-FOR-LARGE-LANGUAGE-MODE"
source: https://arxiv.org/pdf/2609.35579v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:52:38"
field: "大语言模型压缩与剪枝"
keywords: ["residual stream pruning", "output-aware compression", "sensitivity-aware pruning", "SliceGPT", "KL divergence approximation", "structured pruning"]
innovations: ["基于二阶KL散度近似的输出敏感性残差流剪枝目标", "通过谱上界将耦合目标转化为可高效求解的特征分解问题", "提出tau网格候选搜索策略实现激活协方差与输出敏感度的权衡"]
benchmarks: ["WikiText-2", "WinoGrande", "MMLU", "ARC-Easy", "ARC-Challenge", "HellaSwag", "OpenBookQA", "PIQA", "GSM8K", "IFEval"]
---

# 论文速读：OUTPUT-AWARE-RESIDUAL-STREAM-PRUNING-FOR-LARGE-LANGUAGE-MODE

## 一句话总结
本文提出一种**输出敏感性感知**的残差流剪枝方法（Output-Aware Residual Stream Pruning），通过在方向选择目标中同时考虑激活协方差与模型输出的局部敏感度（二阶 KL 散度近似），显著优于仅依赖激活重建误差的 SliceGPT，尤其在较高剪枝率下表现更优。

## 研究问题与动机
- **核心问题**：残差流剪枝（Residual Stream Pruning）通过旋转低重要性方向并将其裁剪，从而缩减 Transformer 模型的隐藏维度；但现有方法（如 SliceGPT）仅以最小化激活重建误差为目标，**忽略了被删方向对下游输出的敏感度差异**。
- **问题成因**：两个幅度相近的激活扰动可能因方向不同而对输出分布产生截然不同的 KL 散度影响，仅凭激活能量（activation energy）无法捕捉这一输出敏感性。
- **现有方法不足**：SliceGPT 本质是残差流激活的非中心化 PCA，完全忽略曲率/敏感度信息；一阶或纯激活基方法在较高剪枝率下性能退化更明显。
- **目标**：推导一个层级的替代目标，在仅使用少量矩阵运算的前提下，将**输出敏感度**纳入残差流方向选择，并保留与 SliceGPT 相当的工程简洁性。

## 核心贡献（创新点）
1. **输出敏感的剪枝目标**：基于原始与剪枝模型输出分布的 KL 散度的二阶近似，推导出结合激活协方差 $C$ 与输出敏感度矩阵 $H$ 的目标 $\mathcal{L}(U)=\mathrm{Tr}(U^\top C U U^\top H U)$，揭示"激活能量×敏感度"的耦合效应。
   - 与 SliceGPT 的本质区别：SliceGPT 仅优化 $\mathrm{Tr}(U^\top C U)$（忽略敏感度），本文目标同时度量扰动对输出分布的影响。

2. **可计算的谱上界与方向选取算法**：证明 rank-one 情形的精确谱等价（Proposition 1），并扩展到 $k$ 维子空间时给出上界 $B_\tau(U)=\frac{1}{4}\mathrm{Tr}(U^\top M_\tau^2 U)$（其中 $M_\tau=\tau C+\tau^{-1}H$），通过 5 个 $\tau$ 值的网格搜索即可得到高质量候选基。
   - 与 Iterative Stiefel 优化的区别：无需在流形上迭代，仅需少量独立特征分解，工程上可直接嵌入 SliceGPT 流水线。

3. **理论保证与正则化解释**：证明联合优化上界等价于最小化原目标加上正则项 $\|U^\top C U\|_F \|U^\top H U\|_F$（Proposition 3），从理论上说明 $\tau$ 调节了两类信息的权衡。
   - 与直接优化原目标的差距：原目标非凸且难以直接优化，本文的上界在保持可解性的同时给出了有理论支撑的近似。

4. **系统化实验验证**：在 Llama、Mistral、Phi 三大模型系列的多尺度剪枝率（10%–30%）下全面评估，一致降低校准/测试 KL 散度与 perplexity，并在数学推理（GSM8K）和指令遵循（IFEval）任务上稳定提升。
   - 与前作的定位差异：不仅对比 SliceGPT，还对比 LLM-Pruner、OSSCAR、Wanda-sp、FLAP 等结构化剪枝方法，验证残差流剪枝在整体压缩范式中的竞争力。

## 方法详解
### 3.1 残差流剪枝回顾
- 利用 Transformer 残差流对正交变换的不变性：若 $Q$ 为正交矩阵，则可将 $Q$ 吸收进相邻线性层的权重中，而不改变前向计算结果（RMSNorm 满足正交等变性；LayerNorm 需重写）。
- 设旋转矩阵 $Q=[V \quad U]\in\mathbb{R}^{d\times d}$，其中 $V\in\mathbb{R}^{d\times(d-k)}$ 保留方向、$U\in\mathbb{R}^{d\times k}$ 为待删方向；删去 $U$ 坐标后，通过过渡矩阵 $(V^{\ell+1})^\top V^\ell$ 在各层残差连接上保持一致。

### 4.1 敏感性感知方向选择
- 理想目标：最小化校准集上原始模型与剪枝模型输出分布的期望 KL 散度 $\mathbb{E}_{s\sim\mathcal{D}_{\mathrm{cal}}}[D_{\mathrm{KL}}(p_\theta(\cdot|s)\|p_{\hat{\theta}(\mathcal{U})}(\cdot|s))]$。
- 单 token 扰动下 KL 的二阶展开：$D_i(\hat{x}_i;s)\approx\frac{1}{2}(\hat{x}_i-x_i)^\top H_i(s)(\hat{x}_i-x_i)$，其中 $H_i(s)$ 为关于干预激活的 Fisher 信息矩阵。
- 忽略跨 token 与跨层曲率交互，定义层共享曲率 $H=\mathbb{E}[\frac{1}{|s|}\sum_i H_i(s)]$。
- 剪枝导致的扰动为 $\hat{x}_i=(I-UU^\top)x_i$，代入后得到层级目标：
  $$\mathcal{L}(U)=\mathrm{Tr}(U^\top C U U^\top H U),$$
  其中 $C=\mathbb{E}[\frac{1}{|s|}\sum_i x_i x_i^\top]$ 为激活二阶矩矩阵。当 $H=I$ 时退化为 SliceGPT 目标。

### 4.2 高效近似与理论分析
- **Proposition 1（rank-one 精确解）**：$\min_{\|u\|=1}(u^\top C u)(u^\top H u)=\frac{1}{4}\inf_{\tau>0}[\lambda_{\min}(\tau C+\tau^{-1} H)]^2$，最优 $u$ 为 $\tau_\star C+\tau_\star^{-1}H$ 最小特征值对应的特征向量。
- **Proposition 2（$k$-dim 上界）**：$\mathcal{L}(U)\leq B_\tau(U)=\frac{1}{4}\mathrm{Tr}(U^\top M_\tau^2 U)$，$M_\tau=\tau C+\tau^{-1}H$；固定 $\tau$ 下由 $M_\tau$ 最小 $k$ 个特征值对应的特征向量达成。
- **Proposition 3（正则化等价）**：$\frac{1}{4}\inf_\tau\sum_{j=1}^k\lambda_j(M_\tau)^2=\min_U[\frac{1}{2}\mathcal{L}(U)+\frac{1}{2}\|U^\top C U\|_F\|U^\top H U\|_F]$，说明谱上界的最小化等价于原目标加上激活能量与敏感度在子空间上的乘积正则项。
- **候选选择流程**：给定 $\tau$ 网格（实验中 $\tau\in\{1,7,10,30,70\}$），对每个 $\tau_j$ 计算 $U_j=\mathrm{eig}_{\min,k}(M_{\tau_j})$，再在候选中选择使 $\mathcal{L}(U_j)$ 最小者。
- **$H$ 的估计**：每个校准 token 从模型输出分布采样 1 个 token，通过对数概率求梯度得到 $H_i(s)$ 的近似，再按 token 平均得到层共享 $H$；全部采用单精度累加，$C$ 与特征分解使用双精度。

## 实验与结果
- **模型与剪枝率**：Llama-3.2 3B、Llama-3.1 8B、Mistral 7B v0.3、Mistral Nemo、Phi-3 Medium，剪枝率 10%、20%、25%、30%。
- **校准数据**：语言建模/常识推理使用 1024 条 FineWeb-Edu 序列（长度 2048）；生成任务使用 1024 条 Tulu3 示例，敏感度仅在 assistant token 上计算。
- **评测指标**：KL 散度、perplexity（WikiText-2）、平均常识推理准确率（WinoGrande、MMLU、ARC-E/C、HellaSwag、OpenBookQA、PIQA）、GSM8K（数学推理）、IFEval（指令遵循）。
- **主要结果（Table 1）**：在所有模型与所有剪枝率下，本文方法的 $D_{\mathrm{KL}}$ 与 PPL 均低于 SliceGPT。例如 Llama-3.2-3B 在 30% 剪枝率下：SliceGPT PPL=57.17、Acc=47.40；本文 PPL=30.14、Acc=49.43。
- **数学推理与指令遵循（Table 2）**：在 Llama-3.2-3B、Llama-3.1-8B、Mistral-Nemo 三个模型上，10%–30% 剪枝率下 IFEval（prompt/instruction 准确率）与 GSM8K 准确率均优于 SliceGPT，且差距随剪枝率增大而扩大（如 Llama-3.1-8B 在 30% 下 IFEval prompt 准确率 44.2 vs 42.7，GSM8K 55.1 vs 52.8）。
- **校准数据量敏感性（Figure 3）**：128→1024 样本提升显著，1024 以上收益趋缓。
- **额外优化开销（Table 3）**：在选定子空间上进行 10000 步额外优化仅带来微小收益，说明谱候选已足够好。
- **校准时间（Table 4）**：在 Llama-3.1-8B 上灵敏度估计耗时 9 分钟（占总时间 12%），总时长从 SliceGPT 的 01:03 增至 01:14，工程可控。
- **与其他结构化剪枝对比（Table 5）**：在 Llama-3.1-8B 上，本文方法在各剪枝率下与 LLM-Pruner、OSSCAR、Wanda-sp、FLAP 等竞争，体现残差流剪枝范式的整体竞争力。
- **最强提升**：Llama-3.2-3B 在 30% 剪枝率下 PPL 从 57.17 降至 30.14（降低约 47%），平均常识推理准确率从 47.40 提升至 49.43。

## 相关工作脉络
- **SliceGPT (Ashkboos et al., 2024)**：本文的直接基线，利用 Transformer 残差流旋转不变性进行 PCA 式残差流降维；本文在其基础上引入输出敏感度矩阵 $H$。
- **LLM-Pruner (Ma et al., 2023)** / **LLM-Surgeon (van der Ouderaa et al., 2024)**：结构化剪枝，分别使用一阶 Taylor 近似与 Kronecker 因式化 Fisher 矩阵评估重要性；本文关注残差流维度压缩而非单层权重剪枝。
- **SparseGPT (Frantar & Alistarh, 2023)** / **Wanda (Sun et al., 2024)**：针对权重矩阵的一次性剪枝，产生非结构化稀疏；本文输出结构化残差流压缩，兼容标准硬件。
- **YAQA (Tseng et al., 2026)** / **EvoPress (Sieberling et al., 2025)** / **Tyr-the-pruner (Li et al., 2025)**：输出分布感知压缩方法，分别针对量化与剪枝配置优化 KL；本文与 YAQA 最接近，但专注残差流方向选择而非逐权重量化。
- **GFWSVD (Chekalina et al., 2026)** / **SVD-LLM (Wang et al., 2025)** / **ASVD (Yuan et al., 2025)**：基于低秩分解的权重压缩方法；本文通过旋转吸收避免额外低秩参数。
- **FLAP (An et al., 2024)** / **OSSCAR (Meng et al., 2024)** / **ZipLM (Kurtic et al., 2023)**：针对 attention head / MLP channel 的结构化剪枝；本文作用于整个残差流维度，剪枝模式不同。

## 局限性与未来方向
- **数据集平均曲率的近似**：用共享矩阵 $H$ 替代 token 级 $H_i(s)$，忽略了激活外积与对应曲率之间的相关性；附录 G 虽通过 rescale 实验表明两者在相对排序上高度一致，但在敏感度跨 token 变化剧烈的场景下可能失真。
- **跨层曲率交互被忽略**：层 wise 处理未建模相邻层扰动传播的耦合效应，可能在高剪枝率下累积误差。
- **过渡矩阵的额外存储与计算**：层间过渡矩阵 $(V^{\ell+1})^\top V^\ell$ 带来 $\mathcal{O}((d-k)^2)$ 额外开销，作者承认未来可通过结构化旋转或跨层参数共享来降低。
- **与非结构化/量化方法的比较不对等**：Table 5 对比的其他结构化剪枝方法的 sparsity 统计口径与残差流剪枝不同（后者额外引入旋转参数），限制直接公平性。
- **未来方向**：探索参数高效的跨层共享变换、联合优化残差流与注意力头/MLP 通道剪枝、以及将输出敏感度目标推广至多模态模型。

## 研究启发与可借鉴点
- **二阶 KL 散度近似作为裁剪/量化目标**：将输出敏感度显式建模为 Fisher 信息矩阵 $H$，可为其他模型压缩任务（如权重剪枝、量化）提供可复用的评估准则；与本团队在"输出分布保真"方向的工作高度可结合。
- **谱上界的参数化构造技巧**：通过引入标量 $\tau$ 将乘积型目标 $(u^\top Cu)(u^\top Hu)$ 转化为 $\lambda_{\min}(\tau C+\tau^{-1}H)$ 的最优化，是一种将非凸耦合目标拆解为特征分解的可迁移技巧，可推广至其他矩阵乘积优化问题。
- **校准数据分层设计**：针对不同任务类型（语言建模 vs 生成）使用不同校准数据，并在生成任务中仅在 assistant token 上计算敏感度，这一设计可有效避免 free-form 生成带来的敏感度估计偏差，值得在后续多任务压缩实验中借鉴。
- **双精度累积 + 单精度敏感度估计的工程折中**：$C$ 用双精度确保 PCA 方向稳定，$H$ 用单精度降低显存与计算负担，这一混合精度策略在大规模 LLM 校准中具有实用参考价值。
- **候选网格搜索替代流形优化**：用固定 $n=5$ 个 $\tau$ 值的谱候选网格搜索代替复杂的 Stiefel 流形迭代，在几乎不损失性能的前提下大幅降低工程复杂度，可作为后续高效结构搜索的基准范式。

## 关键术语表
- **Residual Stream Pruning**：通过正交旋转将低重要性残差流方向对齐到可删坐标轴，随后切除相应行列权重以缩减模型宽度的结构化剪枝范式。
- **SliceGPT**：基于残差流激活非中心化 PCA 的一次性剪枝方法，以最小化 $\mathrm{Tr}(U^\top C U)$ 为准则选择待删方向。
- **Fisher Information Matrix (输出敏感度)**：关于干预激活 $x_i$ 的输出 KL 散度二阶导数，刻画该方向上扰动对模型预测分布的敏感程度。
- **Activation Covariance $C$**：校准集上激活二阶矩 $\mathbb{E}[x_ix_i^\top]$，衡量各残差方向的激活能量。
- **Curvature Matrix $H$**：层共享的输出敏感度矩阵，通过对所有校准 token 的 $H_i(s)$ 取平均得到。
- **Sensitivity-weighted Covariance $M_\tau$**：加权组合 $\tau C+\tau^{-1}H$，其特征向量用于构造剪枝候选子空间。
- **Transition Matrix**：相邻层保留子空间之间的正交映射 $(V^{\ell+1})^\top V^\ell$，用于在残差连接上保持一致性。
- **Output-Aware Compression**：以输出分布的 KL 散度（或其近似）为优化目标而非仅靠激活/权重幅值的压缩方法。

## 可复现要素
- **数据集**：FineWeb-Edu（Penedo et al., 2024）、Tulu3（Lambert et al., 2025）、WikiText-2、WinoGrande、MMLU、ARC-E/C、HellaSwag、OpenBookQA、PIQA、GSM8K、IFEval；FineWeb-Edu 与 Tulu3 为公开数据集。
- **代码/权重**：论文未明确声明开源仓库链接（PDF 正文与附录未提及 GitHub 地址），需关注作者页面或 arXiv 附属信息；基础模型（Llama-3.x、Mistral、Phi-3）均为公开权重。
- **关键超参**：
  - 校准序列数：1024
  - 序列长度：2048
  - $\tau$ 网格：$\{1, 7, 10, 30, 70\}$（共 5 个候选）
  - $H$ 估计：每 token 从模型输出分布采样 1 个 token
  - 数值精度：$C$ 与特征分解使用双精度，$H$ 累加使用单精度
  - 额外优化步数（消融实验）：10,000 步/剪枝位
  - 评估设置：常识推理 zero-shot；GSM8K 8-shot with CoT；IFEval zero-shot
