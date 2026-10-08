---
title: "Sparse-Planning-in-Visual-World-Models-via-Cost-Gradients"
source: https://arxiv.org/pdf/2610.10274v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:08:31"
field: "视觉世界模型的高效推理与规划"
keywords: ["visual world models", "sparse planning", "token selection", "gradient-based attribution", "model-based reinforcement learning", "CEM planning", "AdaLN conditioning"]
innovations: ["提出COSTGRAD，一种基于规划成本梯度范数的免训练目标条件token选择器，在50%稀疏度下匹配或超越全token规划", "揭示选择器-架构兼容性现象：COSTGRAD在AdaLN-Zero预测器上有效但在matched-concat上失效，差异由action-pathway drift解释"]
benchmarks: ["PointMaze", "Wall", "PushT", "MetaWorld"]
---

# 论文速读：Sparse-Planning-in-Visual-World-Models-via-Cost-Gradients

## 一句话总结
本文提出 COSTGRAD，一种免训练、目标条件的 token 选择器，通过计算规划成本对输入 token 的梯度范数来排序并保留关键 token，从而在视觉世界模型（DINO-WM/JEPA-WMs）的 CEM 规划中实现稀疏计算。在 50% 稀疏度下，该方法在 4 个连续控制基准中的 3 个上匹配或超越全 token 规划，并提供 2.6× wall-clock 加速；结合减少的 CEM 迭代，可实现约 5× 总加速且成功率更高。

## 研究问题与动机
- 基于 token 的世界模型在进行 CEM 规划时，每一步需对全量 256 个空间 token 执行数千次前向计算（约 $1.4 \times 10^7$ 次 token 级运算），但控制任务中真正相关的 token（智能体、物体、障碍物、目标）只占极小比例，大量背景 token 浪费计算资源。
- 直观方案（复用预测器自身的 attention 权重或预测误差）存在**目标不匹配**：attention 受预测损失欠定，预测误差衡量视觉不可预测性而非控制重要性——对控制最关键的对象（如 PushT 中随动作易预测运动的推块）恰恰因"易于预测"而被 prediction-error 低估。
- 已有稀疏规划方法（Sparse Imagination）依赖随机 token 选择 + 稀疏训练，无法针对当前目标动态选择；本文目标是让固定预训练预测器在无额外训练的前提下，依据下游控制目标自适应地稀疏化输入。

## 核心贡献（创新点）
- **提出 COSTGRAD，一种基于规划成本梯度的免训练目标条件 token 选择器**：通过对零动作探针下预测结果与目标编码的 L2 距离求关于各输入 token 的梯度范数来评分，保留 Top-K 关键 token。与 PREDATTN/PREDGRAD 等基于预测信号的选择的本质区别在于：选取标准直接来自下游控制目标而非预测损失。
- **在 50% 稀疏度下达到甚至超越全 token 规划性能**：在 AdaLN-Zero 预测器上，COSTGRAD 在 PointMaze（−2.0 pp，噪声范围内）、Wall（+7.3 pp）、PushT（+2.8 pp）、MetaWorld（+6.9 pp）上取得平均 +3.8 pp 的提升；与 Random（−19.2 pp）和 PREDATTN（平均低 12.1 pp）形成显著差距。
- **发现选择器-架构兼容性这一新设计维度**：在配对 AdaLN-vs-concat 对比实验中，COSTGRAD 在 AdaLN 上显著优于 Random（+25.0 pp），但在 matched concat 上完全失去优势（−5.1 pp vs Full）。该差异通过 action-pathway drift（KL 散度诊断）得到解释——梯度选择移除在 AdaLN 上比随机移除产生更小漂移（0.29×），而在 concat 上反而更大（1.33×）。
- **揭示稀疏规划可超越全 token 性能的机制**：稀疏化不仅节省计算，还能通过去除背景噪声使规划代价景观在可控对象和目标相关区域更加陡峭；在更退化的 L1 代价下，COSTGRAD 相对 Full 的提升从 +7.3 pp 扩大至 +22.9 pp。

## 方法详解
**Token 选择（每规划步一次）**：
- 给定上下文编码 $z$ 和目标编码 $z_g$，在 probe 动作 $\mathbf{a}_0 = \mathbf{0}$ 下执行一步前向传播，计算规划代价：$\mathcal{L}_{\text{plan}}(\mathbf{a}_0; z, z_g) = \|\text{predictor}(z, \mathbf{a}_0) - z_g\|_2^2$。
- 单次反向传播获得每个 token 的分数：$s_j = \|\nabla_{z_j} \mathcal{L}_{\text{plan}}\|_2$，保留 Top-K 个 token 构成集合 $\mathcal{S}$（默认 $K = P/2 = 128$）。
- 梯度范数衡量的是 token 对规划代价的局部敏感度，包括其对其他位置预测的间接影响——即使某 token 自身预测值接近目标，只要它对 goal-relevant 区域的预测有关键上下文作用，仍能获高分。

**稀疏规划（CEM 在选定 token 集上运行）**：
- 保留 token 维持原始位置索引，共享于 context frames 和 goal；选择探针仅用于打分，CEM 采样和动作优化使用自己的动作序列。
- CEM 在每个环境步中对 horizon $H=6$ 采样 300 条动作序列，取 top-30 elites 重拟合分布，迭代 30 轮；最终代价为视觉特征 L2（权重 1）与 proprioceptive 特征 L2（权重 0.1）的加权和。
- 选定 token 子集在整个 CEM 搜索过程中固定不变，仅在下一观测帧重新选择。

**计算开销**：
- 额外开销为一次全 token forward + backward pass（约 27.53 ms on H100），摊销到 30 轮 CEM 迭代后仅约 1 ms/iter，远低于每轮节省的 ~1820 ms。
- Token 减半使 attention FLOPs 降 4×（$O(N^2)$）、MLP FLOPs 降 2×（$O(N)$），合计产生 2.6× wall-clock 加速。

## 实验与结果
- **数据集与基准**：四个连续控制仿真环境——PointMaze（MZ）、Wall、PushT（PT）、MetaWorld（MW Reach），源自 DINO-WM / JEPA-WMs 评测套件。
- **评估指标**：规划成功率（%），3 个 evaluation seed × 96 episodes/seed。
- **主要结果（Table 1，50% token，AdaLN-Zero 预测器）**：

| Method | PointMaze | Wall | PushT | MetaWorld | Avg |
|---|---|---|---|---|---|
| Full (100%) | 88.2±8.7 | 84.7±2.6 | 62.4±4.4 | 58.7±1.8 | 73.5 |
| **COSTGRAD** | **86.2±6.1** | **92.0±1.8** | **65.2±6.5** | **65.6±4.7** | **77.3** |
| PREDATTN | 86.7±9.7 | 67.3±1.7 | 62.0±8.0 | 44.8±6.1 | 65.2 |
| Random | 77.0±9.4 | 67.0±3.6 | 41.3±5.9 | 46.9±3.1 | 58.1 |

- **最强结果**：Wall 环境达到 92.0% 成功率（Full: 84.7%），提升 +7.3 pp；平均成功率 77.3%，超越 Full 的 73.5%（+3.8 pp）。
- **速度-性能权衡**：COSTGRAD 50% tokens + 15 CEM iterations 达到 92.0% 成功率，耗时 ~17 s/step，相比 Full（84.7% at 88.6 s）实现 **5.2× 总加速**，且成功率更高。
- **稀疏度扫描**：在 25% token 时，Wall/PushT/MetaWorld 三个环境中仍接近 Full 性能；低于 ~25% 时所有稀疏方法性能骤降。
- **公开 checkpoint 验证**：在未微调的 JEPA-WMs 官方 checkpoint 上同样有效（COSTGRAD 优于 Random 11.3–46.3 pp）。

## 相关工作脉络
- **DINO-WM / JEPA-WMs** [Zhou et al., 2025; Terver et al., 2025]：本文的基础框架，冻结 DINOv2 编码器 + transformer 预测器，在 token 级 latent 空间中进行 CEM 规划；COSTGRAD 在不改变预测器的前提下减少规划时 token 数量。
- **Sparse Imagination** [Chun et al., 2025]：结合训练期随机分组 attention 与规划期随机 token 选择；与本文互补——稀疏训练提升对缺失 token 的容忍度，COSTGRAD 从当前目标识别规划相关 token。
- **Scope-WM** [Li et al., 2026]：同期工作，将 prediction-loss sensitivity 蒸馏为 action-conditioned 选择器并配合轻量 background update 训练；与 COSTGRAD 的关键区别在于后者完全免训练、直接从当前目标的规划成本梯度选择。
- **Token merging / pruning** [Bolya et al., 2023; Rao et al., 2021; Liu et al., 2023]：ViT 效率优化方法，面向单次前向感知任务；COSTGRAD 面对的是 CEM 中多次 action-conditioned rollout 的需求，对 token subset 的动态稳定性和任务相关性要求更高。
- **Gradient saliency / attribution** [Simonyan et al., 2014; Sundararajan et al., 2017; Selvaraju et al., 2017]：从输入梯度可视化到 Integrated Gradients 和 Grad-CAM；COSTGRAD 将梯度显著性应用于下游控制目标的 token 选择而非分类解释。
- **Objective mismatch / value equivalence** [Lambert et al., 2020; Farahmand et al., 2017; Grimm et al., 2020]：区分预测保真度与下游决策有用性；COSTGRAD 将此原则延伸至推理时 token 选择，由规划目标决定哪些表征支持动作搜索。

## 局限性与未来方向
- 评测仅限四个仿真环境、单一编码器（DINOv2 ViT-S/14）和单一规划器（CEM）；物理系统中的感知噪声、校准误差和模型失配尚未验证。
- Selector–architecture 兼容性发现仅针对测试的 AdaLN 和 concat 两种 action-conditioning 方式；FiLM、cross-attention、prefix-style conditioning 等机制的通用性未检验。
- 选择器在当前环境下表现不均：PointMaze 和 PushT 上 attention-based 选择已具竞争力，Wall 和 MetaWorld 上差距更大；环境依赖性仍需探索。
- 当前使用单步 probe（zero action）进行 token 选择，而 CEM 评估更长 horizon 动作序列；更长 horizon 的 probe 是否能显著提升选择质量值得研究。
- 所有实验使用固定预训练预测器和推理时选择；在线适应（online adaptation）和 learned selector 留作未来工作。

## 研究启发与可借鉴点
- **规划目标驱动的 token 选择范式可迁移至其他视觉-动作联合推理任务**：任何需要在 latent space 中进行多步 roll-out 的环境（如 VLA 模型推理、robotic manipulation planning）均可借鉴"用下游任务成本而非预测误差指导稀疏化"的思路。
- **Selector–architecture compatibility 作为新的评估维度**：本文揭示了 token 选择效果强烈依赖预测器如何注入动作条件——这对于设计高效 world model 系统具有重要启示：稀疏规划不仅要选对 token，还需确保预测器架构在稀疏输入下保持动作响应的稳定性。
- **Action-pathway drift 诊断指标（KL 散度）**：通过比较完整与稀疏输入下动作效应的空间分布差异来量化架构对 token 移除的敏感度，这一诊断工具可复用于评估其他稀疏化方法对模型行为的影响。
- **Speed-success Pareto 前沿的系统刻画**：本文同时扫描 token 预算（K）和 CEM 迭代数两个独立计算轴，展示了二者互补性——这种 multi-axis 效率优化视角对构建实时规划系统具有参考价值。
- **Easy-but-critical token 分析框架**：将 prediction error 与 planning gradient 做 quadrant 分析，揭示了"易于预测但控制关键"的 token 类型（PushT 中推块）；这种交叉分析可用于诊断和比较不同选择策略的偏向。

## 关键术语表
**COSTGRAD**：本文提出的免训练 token 选择器，通过规划成本对输入 token 的梯度范数排序来选择关键 token。
**Token-based world model**：以冻结视觉编码器输出的空间 token 网格为 latent 表示的世界模型，支持在 token 级进行 goal-conditioned 规划。
**CEM (Cross-Entropy Method)**：在 latent 空间中通过采样、评估和重拟合候选动作序列来优化规划代价的采样式搜索算法。
**Action-pathway drift**：稀疏 token 选择导致预测器对不同动作的响应空间分布相对于全 token 输入的偏移程度，用 KL 散度量化。
**Objective mismatch**：模型预测精度与下游控制决策有用性之间的根本性差异，是本文选择规划梯度而非预测误差的核心动机。
**AdaLN-Zero**：在 transformer block 中以 adaptive layer normalization 方式注入动作条件的架构，动作参数通过 scale/shift/gate 调制每个 token。
**Concat conditioning**：将动作特征 channel-wise 拼接到每个视觉 token 后输入自注意力的动作条件注入方式。
**Probe action**：用于计算 token 选择分数的固定动作输入（默认零动作），不用于实际规划搜索。

## 可复现要素
- **数据集**：PointMaze、Wall、PushT、MetaWorld（Reach）；来自 DINO-WM / JEPA-WMs 评测套件，代码开源（JEPA-WMs GitHub）。
- **代码/权重**：项目页面 https://ycxuyingchen.github.io/costgrad/；使用 JEPA-WMs 开源代码库；公开 checkpoint 来自 Terver et al. (2025)。
- **关键超参**：Token 编码器 DINOv2 ViT-S/14（16×16 = 256 tokens，dim=384）；预测器 6-block transformer，16 heads，embedding=400；CEM：horizon H=6，300 candidates，30 iterations，10 elites；COSTGRAD 默认 K=128（50% sparsity）；probe action 默认 zero；规划代价 L2 visual（权重 1）+ L2 proprioceptive（权重 α=0.1）。
- **训练细节**：预测器用 AdamW，lr=5e-4，batch_size=8/GPU，20 epochs；COSTGRAD 本身免训练。
