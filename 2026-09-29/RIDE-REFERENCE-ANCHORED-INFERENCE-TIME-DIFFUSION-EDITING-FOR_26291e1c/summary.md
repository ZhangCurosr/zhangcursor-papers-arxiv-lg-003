---
title: "RIDE-REFERENCE-ANCHORED-INFERENCE-TIME-DIFFUSION-EDITING-FOR"
source: https://arxiv.org/pdf/2609.35623v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:56:13"
field: "分子生成与药物设计"
keywords: ["scaffold hopping", "diffusion model", "inference-time editing", "molecular generation", "drug discovery"]
innovations: ["提出参考锚定推理时扩散编辑框架 RIDE，实现无需重训练的可控骨架生成", "引入价值引导采样策略，联合优化 2D 新颖性和 3D 形状保留", "通过噪声轨迹恢复与扰动编辑，建立受参考约束的生成分布"]
benchmarks: ["CrossDocked2020", "60 human disease-related protein-ligand complexes"]
---

# 论文速读：RIDE-REFERENCE-ANCHORED-INFERENCE-TIME-DIFFUSION-EDITING-FOR-SCAFFOLD-HOPPING

## 一句话总结
本文提出了 RIDE（Reference-anchored Inference-time Diffusion Editing），一个用于 scaffold hopping 的推理时扩散编辑框架，通过恢复参考扩散噪声轨迹、选择最优轨迹段进行噪声扰动，并结合价值引导的骨架采样，实现低 2D 相似性、高 3D 相似性的新骨架生成，无需重新训练基础模型即可提升现有扩散模型的药物发现能力。

## 研究问题与动机
- **核心问题**：Scaffold hopping 旨在发现与参考配体共享关键功能基团和相似 3D 形状，但 2D 结构显著不同的新分子骨架。
- **现有方法不足**：
  - 现有扩散基方法将问题建模为给定功能基团的骨架条件生成，依赖采样随机性多样化结果，缺乏 principled 机制来联合保证 2D 结构新颖性和 3D 形状保留。
  - 基于结构的药物设计（SBDD）方法可结合蛋白-配体结合信息，但缺乏相对于已知配体的显式 2D 结构新颖性保障。
  - 基于配体的药物设计（LBDD）方法可探索结构不同的骨架，但缺乏蛋白-配体结合的显式指导。
- **关键需求**：需要一种统一框架，在保持结合亲和力和 3D 形状的同时，显式优化 2D 相似性降低。

## 核心贡献（创新点）
1. **提出 RIDE 框架**：将 scaffold hopping 重新表述为从参考骨架到新生成骨架的扩散编辑过程，通过三步（噪声轨迹恢复、最优轨迹段选择、价值引导采样）实现可控编辑。
2. **参考锚定轨迹恢复**：利用逆向扩散方法恢复参考骨架在基础扩散模型下的噪声轨迹，建立受参考控制的生成分布。
3. **价值引导的推理时编辑**：引入非梯度价值函数估计中间状态的预期未来奖励，在扰动轨迹段内进行贪心搜索，同时优化 2D 新颖性和 3D 形状保留。
4. **无需重新训练的即插即用框架**：RIDE 可直接适配预训练的 SBDD 扩散模型，无需额外训练即可实现可控 scaffold hopping。

## 方法详解
### 4.1 问题定义
- 配体表示为 $M = (S, F)$，其中 $S$ 为骨架，$F$ 为功能基团，蛋白口袋为 $P$。
- Scaffold hopping 目标：用新生成的骨架 $S$ 替换参考骨架 $S^{\text{ref}}$，生成 $M = (S, F^{\text{ref}})$。
- 优化目标：最大化 $\mathbb{E}_{S \sim \tilde{p}_\theta}[\mathcal{R}(S, S^{\text{ref}})]$，奖励函数 $\mathcal{R} = \lambda(1 - \text{Sim}_{2\text{D}}) + (1-\lambda)\text{Sim}_{3\text{D}}$。

### 4.2 参考锚定骨架生成
**Step 1: 参考噪声轨迹恢复**
- 对参考骨架 $S^{\text{ref}}$ 使用 DDIM 逆转向（Eq. 1）恢复位置轨迹 $\{X_t^{\text{ref}}\}$ 和类型轨迹 $\{V_t^{\text{ref}}\}$。
- 通过 Eq. 2 和 Eq. 4 恢复对应的噪声序列 $\{\epsilon_t^{\text{ref}}\}$ 和 $\{g_t^{\text{ref}}\}$。

**Step 2: 轨迹噪声扰动编辑**
- 在 $t_1$ 到 $t_2$ 的轨迹段上，用采样噪声替换参考噪声：
  - 位置噪声：$\mathcal{N}(0, I)$
  - 类型噪声：Gumbel(0, 1)
- 通过 Monte Carlo 采样评估不同 $t_1$ 的得分 $J(t_1)$（Eq. 13），选择最优起始时间点 $t_1^*$。

**Step 3: 价值引导骨架采样**
- 使用 Theorem 1 的价值估计公式（Eq. 16）计算中间状态 $S_t$ 的预期奖励。
- 在扰动段内执行价值引导的贪心搜索（Eq. 18-19），选择最优噪声。
- 最终通过参考噪声完成从 $t_2^*$ 到 0 的完整轨迹。

## 实验与结果
### 数据集与设置
- **数据集**：CrossDocked2020 测试集的子集，共 60 个与人类疾病相关蛋白靶点的复合物，58 个独特配体。
- **基线模型**：
  - conDitar-a、IPDiff-a（SBDD 模型适配）
  - DiffHopp（结构基 scaffold hopping 模型）
  - ShEPhERD（配体基 scaffold hopping 模型）

### 主要结果（Table 1）
- **整体提升**：RIDE$^{(1)}$ 相比基线模型，$\text{Sim}_{2\text{D}}$ 平均降低 **11.7%**，$\text{Sim}_{3\text{D}}$ 平均提升 **7.3%**。
- **最佳结果**：
  - conDitar-a 基线：$\text{Sim}_{2\text{D}}=0.398$，$\text{Sim}_{3\text{D}}=0.795$
  - RIDE$^{(1)}$ with conDitar-a：$\text{Sim}_{2\text{D}}=0.354$，$\text{Sim}_{3\text{D}}=0.884$
  - RIDE$^{(2)}$ with conDitar-a：$\text{Sim}_{2\text{D}}=0.325$，$\text{Sim}_{3\text{D}}=0.893$
- **Vina S**：RIDE$^{(1)}$ 相比 conDitar-a 和 DiffHopp 平均提升 **39.1%**。
- **连接率**：RIDE$^{(1)}$ 平均提升 **5.0%**。

### 消融实验
- **价值引导 vs 随机扰动**：价值引导采样平均降低 $\text{Sim}_{2\text{D}}$ **8.3%**，提升 $\text{Sim}_{3\text{D}}$ **1.6%**（Table 2）。
- **最优 $t_1^*$ vs 固定 $t_1$**：参考最优的 $t_1^*$ 在所有测试中取得最佳或接近最佳的 $\text{Sim}_{2\text{D}}$ 和 $\text{Sim}_{3\text{D}}$。
- **可迁移奖励**：当奖励替换为 QED 时，RIDE 仍能保持高于基线的 $\text{Sim}_{3\text{D}}$，说明参考轨迹有助于隐式保留 3D 形状。

## 相关工作脉络
1. **SBDD 扩散模型**：IPDiff（Huang et al., 2024）、conDitar（Gao et al., 2026）利用蛋白口袋信息生成分子，但不显式优化 2D 新颖性。
2. **LBDD 扩散模型**：ShEPhERD（Adams et al., 2025）专注于 3D 形状相似生成，但缺乏蛋白-配体结合指导。
3. **Scaffold hopping 专用模型**：DiffHopp（Torge et al., 2023）是首个结构基 scaffold hopping 扩散模型，但未显式优化 2D 相似度。
4. **逆扩散编辑方法**：DDIM inversion（Song et al., 2020）、Prompt-to-Prompt（Hertz et al., 2022）等用于图像编辑，RIDE 将其扩展至分子生成领域。
5. **推理时优化**：Value-based methods（Li et al., 2024; Kim et al., 2026）通过价值函数估计引导采样，RIDE 适配该方法至分子骨架编辑。

## 局限性与未来方向
- **原子数限制**：当前 RIDE 生成的骨架原子数与参考相同，可能限制结构多样性。
- **多跳累积误差**：RIDE$^{(2)}$ 在多跳过程中 Vina S 有所下降，提示属性偏差可能累积。
- **计算开销**：价值估计需要大量 Monte Carlo 采样（$M=1000$），推理成本较高。
- **未来方向**：扩展到更多分子属性优化、开发自适应扰动策略、提升多跳稳定性。

## 研究启发与可借鉴点
1. **推理时编辑范式**：无需重新训练即可适配预训练扩散模型，为其他分子生成任务提供通用框架。
2. **价值引导采样**：非梯度价值函数估计结合 Monte Carlo 估计，适用于不可微的分子性质优化。
3. **参考轨迹恢复**：通过逆扩散恢复参考样本的噪声轨迹，可作为通用技巧用于可控生成。
4. **奖励函数灵活性**：同一框架可适配不同奖励（Sim$_{2\text{D}}$/Sim$_{3\text{D}}$/QED），证明方法的可扩展性。

## 关键术语表
- **Scaffold hopping**：药物发现任务，指在保持功能基团和 3D 形状相似的同时，生成 2D 结构不同的新分子骨架。
- **Reference-anchored**：以参考分子为锚点，通过恢复其噪声轨迹建立受控生成分布。
- **Value-guided sampling**：基于价值函数估计中间状态预期奖励，引导贪心搜索最优生成路径。
- **Sim$_{2\text{D}}$**：分子 2D 结构相似性，用 Tanimoto 距离度量指纹相似性。
- **Sim$_{3\text{D}}$**：分子 3D 形状相似性，用 ROCS 工具计算形状重叠度。
- **Inference-time editing**：在推理阶段修改扩散模型的采样过程，而非重新训练模型参数。
- **Vina S/M/D**：AutoDock Vina 预测的结合亲和力指标，分别对应原始得分、能量最小化后得分和 docking 后得分。

## 可复现要素
- **数据集**：CrossDocked2020（公开），测试集来自 Gao et al. (2026)。
- **代码**：已开源，地址 https://anonymous.4open.science/r/RIDE-C8A0。
- **关键超参**：
  - $T=1000$（conDitar-a, IPDiff-a），$T=500$（DiffHopp）
  - 扰动段长度 $L=100$（默认）
  - Monte Carlo 采样数 $K=100$（轨迹选择），$M=1000$（价值估计）
  - 奖励权重 $\lambda$：RIDE$^{(1)}$ 使用 1，RIDE$^{(2)}$ 使用 0.5-0.8
- **基线权重**：conDitar、IPDiff、DiffHopp、ShEPhERD 均使用公开 checkpoint。
