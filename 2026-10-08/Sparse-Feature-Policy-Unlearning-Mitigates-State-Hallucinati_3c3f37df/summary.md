---
title: "Sparse-Feature-Policy-Unlearning-Mitigates-State-Hallucinati"
source: https://arxiv.org/pdf/2610.09496v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:07:46"
field: "具身智能与机器人策略可解释性"
keywords: ["VLA", "state hallucination", "sparse autoencoder", "policy unlearning", "mechanistic interpretability", "robot manipulation"]
innovations: ["首次将状态幻觉形式化为VLA中可识别的反复失败模式并提供精确定义", "提出基于稀疏特征的选择性策略遗忘方法SOUL，以幻觉特征为遗忘目标、成功特征为保留目标", "证明区域引导的特征选择在策略unlearning中显著优于全局/token级选择"]
benchmarks: ["LIBERO-Plus", "RoboCasa", "Franka Real-world"]
---

# 论文速读：Sparse-Feature-Policy-Unlearning-Mitigates-State-Hallucinati

## 一句话总结
本文研究视觉-语言-动作（VLA）模型中反复出现的"状态幻觉"失败模式，即机器人在未实际达成目标状态时仍继续执行后续动作；通过稀疏自编码器（SAE）分析识别出与状态幻觉相关的内部稀疏特征，并据此提出 SOUL（Sparse feature pOlicy UnLearning）方法，有选择性地抑制幻觉特征、增强成功特征，从而在不损害原有操作能力的前提下显著降低幻觉率并提升任务成功率。

## 研究问题与动机
1. **状态幻觉是 VLA 部署中的反复出现模式**：VLA 在真实机器人操作中频繁出现不可靠行为，其中一种典型失败是机器人在未实际抓住物体的情况下就闭合夹爪，并基于这一虚假状态继续执行运输和放置动作，导致局部错误在多个阶段持续传播。
2. **现有工作主要关注推理时干预或恢复，而非直接修改策略知识**：已有方法（中间推理、故障检测与恢复、测试时采样/验证）侧重于部署时的策略输出改进，而非从参数层面持久性地消除与反复失败相关的策略知识。
3. **机器学习中 unlearning 在分类和语言任务已有探索，但在机器人策略中尚未研究**：挑战在于如何在移除特定失败模式的同时不破坏与其他紧密耦合技能共享的表示；直接对幻觉样本做梯度上升会严重损害任务成功率。
4. **稀疏自编码器为 mechanistic interpretability 提供了可操作的内部表示分解工具**：SAE 可将密集激活分解为稀疏潜在特征，使研究者能将个体特征与更局部化的行为模式相关联，从而为选择性策略修改提供明确目标。

## 核心贡献（创新点）
1. **首次将"状态幻觉"形式化为 VLA 中一类反复出现的可识别失败模式**，并区分了抓取幻觉、运输幻觉和放置幻觉三种子类型，为后续分析提供了统一的标签体系。与以往仅描述"不稳定动作"不同，本文给出了精确的判定标准。
2. **通过视觉注意力分析与稀疏特征表征，揭示状态幻觉与任务相关区域注意力减弱之间的关联**：幻觉执行中策略对机械臂和目标物体的注意力显著弱化，且稀疏特征在幻觉事件前后呈现时间局域化的强激活模式，这是已有工作未曾量化的内部表征差异。
3. **提出 SOUL——基于稀疏特征的选择性策略遗忘方法**，以幻觉特征作为显式"遗忘目标"、成功特征作为"保留目标"，通过特征级损失函数同时抑制坏行为和增强好行为。与已有 unlearning 基线（如梯度上升）相比，本文方法在降低幻觉率的同时保持 89% 的原有成功行为。
4. **构建了区域引导的稀疏特征选择机制**，证明保留空间结构（region-guided vs. global/tokenized）对策略 unlearning 效果至关重要，幻觉率从 45.8%（global）降至 33.5%（region-guided），任务成功率从 42.7% 提升至 52.2%。

## 方法详解
**1. SAE 特征提取**：收集 VLA  rollout 过程中各 transformer 层视觉 token 的残差流激活，展平后作为 SAE 训练样本；使用 Top-k SAE（k=100，辅助 k_aux=512，隐层维度 4096），冻结 SAE 后用于提取稀疏特征激活 $z_{p,j}(x)$。

**2. 区域条件特征激活**：基于语义区域掩码（仿真提供或人工标注），对视觉 token 按区域聚合：
$$z_{r,j}(x) = \frac{1}{|\mathcal{R}_r(x)|}\sum_{p\in\mathcal{R}_r(x)} z_{p,j}(x)$$
对每个 rollout 取时间维最大值 $m_{e,r,j}$，再在各执行结果类型内平均得 $\bar{m}_{c,r,j}$。

**3. 特征关联分数与分类**：定义两类分数：
$$s_{r,j}^{\mathrm{hall}} = \bar{m}_{\mathrm{HF},r,j} - \tfrac{1}{2}(\bar{m}_{\mathrm{CS},r,j}+\bar{m}_{\mathrm{NF},r,j})$$
$$s_{r,j}^{\mathrm{succ}} = \bar{m}_{\mathrm{CS},r,j} - \tfrac{1}{2}(\bar{m}_{\mathrm{HF},r,j}+\bar{m}_{\mathrm{NF},r,j})$$
$s_{r,j}^{\mathrm{hall}}$ 高的特征归为**幻觉特征（forget set $\mathcal{F}$）**，$s_{r,j}^{\mathrm{succ}}$ 高的归为**成功特征（retain set $\mathcal{R}$）**；遗忘目标还额外要求高区分度 $s_{r,j}^{\mathrm{sep}}=\bar{m}_{\mathrm{HF}}-\bar{m}_{\mathrm{NF}}$，最终 $s_{r,j}^{\mathcal{F}}=s_{r,j}^{\mathrm{hall}}+(s_{r,j}^{\mathrm{sep}}-s_{r,j}^{\mathrm{succ}})$。

**4. 特征引导的策略遗忘损失**：SAE 编码器冻结，仅优化 LoRA 适配器参数 $\theta$。遗忘损失（对 HF 样本最小化幻觉特征激活的平方）：
$$\mathcal{L}_{\mathrm{forget}} = \mathbb{E}_{x\in\mathcal{D}_{\mathrm{HF}}}\left[\frac{1}{|\mathcal{F}|}\sum_{(r,j)\in\mathcal{F}}(z_{r,j}(x;\theta))^2\right]$$
保留损失（对 CS 样本最大化成功特征激活）：
$$\mathcal{L}_{\mathrm{retain}} = -\mathbb{E}_{x\in\mathcal{D}_{\mathrm{CS}}}\left[\frac{1}{|\mathcal{R}|}\sum_{(r,j)\in\mathcal{R}}z_{r,j}(x;\theta)\right]$$
联合损失：$\mathcal{L}_{\mathrm{unlearn}}=\mathcal{L}_{\mathrm{forget}}+\lambda_{\mathrm{r}}\mathcal{L}_{\mathrm{retain}}$。

## 实验与结果
**数据集与基线**：
- 仿真基准：LIBERO-Plus（OpenVLA）、RoboCasa（$\pi_{0.5}$），聚焦 pick-and-place 任务
- 真实场景：Franka Research 3 七自由度机械臂，各模型 50 rollouts
- 基线：原始预训练策略、梯度上升 unlearning（在 HF 样本上最大化动作预测损失）

**主要结果（TABLE I，两基准平均）**：
| 方法 | 幻觉率↓ | 幻觉失败率↓ | 总成功率↑ | 干净成功率↑ |
|---|---|---|---|---|
| 原始策略 | 55.0% | 48.6% | 34.0% | 27.5% |
| 梯度上升 | 72.0% | 71.5% | 2.4% | 1.9% |
| **SOUL** | **33.5%**（↓21.7%p） | **29.5%** | **56.2%**（↑22.1%p） | **52.2%** |

- **真实机器人**：SOUL 平均提升干净成功率 20%p，降低幻觉失败率 22%p
- **行为保留与修正**：保留率 89%，将 40% 原有失败执行修正为成功，修正 78% 幻觉事件
- **各幻觉类型均下降**：抓取/运输/放置幻觉同步降低，非单一阶段修复

## 相关工作脉络
1. **VLA 部署可靠性工作**（Safe-VLA、Failsafe、Self-Correcting VLA 等）：关注推理时的故障检测/恢复或约束学习，而本文从参数层面持久性地消除反复出现的幻觉行为，二者互补而非替代。
2. **VLA 测试时干预**（Haon et al., Mechanistic Interpretability for Steering VLA；Khan et al., Sparse Latent Directions）：这些工作在推理时对 feed-forward 方向或稀疏特征进行实时干预；本文将干预编码进策略参数，实现无需重复干预的持久行为修正。
3. **机器遗忘（Machine Unlearning）**（Tofu、Influence-based unlearning）：主要在分类和语言任务中研究"删除特定数据影响"；本文首次将其应用于机器人策略，并面临更严峻的"技能耦合"挑战。
4. **稀疏自编码器（SAE）**（Huben et al., Gao et al.）：SAE 此前用于语言模型可解释性分析或测试时行为引导；本文首次将 SAE 特征作为显式遗忘/保留目标指导策略参数更新。
5. **LIBERO-Plus / RoboCasa**：前者提供多级别视觉扰动下的鲁棒性评测，后者提供多样化家庭环境；本文首次在 LIBERO-Plus 上建立状态幻觉评测协议。

## 局限性与未来方向
1. **仅在 pick-and-place 这一基础原语上验证**，尚未扩展至更长 horizon 或更复杂的多步骤组合任务，状态幻觉在复杂任务中的表现未知。
2. **SAE 特征依赖于有限层（5 个层位）**，可能遗漏其他层中与幻觉相关的表征模式。
3. **仅针对一类反复出现的失败模式**，对于 VLA 的其他不可靠行为（如动作不可行、长期漂移等）的有效性未验证。
4. **遗忘目标选择仅取 top-5 区域-特征对**，特征数量较少，可能未覆盖所有幻觉相关模式。
5. **未讨论 unlearning 过程中的灾难性遗忘边界**：当 $\lambda_r$ 设置不当或 forget set 过大时，可能侵蚀原本健康的技能表征。

## 研究启发与可借鉴点
1. **区域引导的稀疏特征选择**：将视觉 token 按语义区域聚合后再计算特征关联分数，比全局池化或 token 级选择效果显著更好（幻觉率 45.8%→33.5%），这一"空间 grounding + 特征分析"的思路可迁移至其他具身模型的可解释性与干预研究。
2. **遗忘-保留双目标设计**：同时构造显式的 forget set 和 retain set，用组合损失平衡"删除坏行为"与"保留好行为"，避免了纯梯度上升 unlearning 导致的全面性能崩溃，可作为具身 unlearning 的标准范式。
3. **特征激活的时间局域化分析**：幻觉特征并非全程高激活，而是在幻觉事件前后短暂强烈激活；这一发现提示后续工作可探索时间窗口感知的特征干预而非全 rollout 惩罚。
4. **真实机器人验证的价值**：论文同时在仿真和 Franka 真实设备上验证，真实场景中 clean success 提升 20%p，证明了方法从仿真到实物的泛化能力，为后续工作提供了可行的 eval 流程参考。
5. **LoRA 微调 + 特征约束 unlearning 的工程路径**：先对目标任务做 LoRA 微调，再冻结主权重仅用 LoRA 做特征级 unlearning，这一轻量且可逆的参数更新策略适合机器人快速部署场景。

## 关键术语表
**State Hallucination（状态幻觉）**：VLA 在机器人-物体未达到某物理状态时，仍基于虚假状态继续执行后续动作的反复出现行为模式。
**Clean Success（干净成功）**：无状态幻觉事件且成功完成任务的 rollout。
**Hallucination Failure（HF）**：包含至少一次状态幻觉事件且最终任务失败的 rollout。
**Non-hallucination Failure（NF）**：无状态幻觉但任务失败的 rollout。
**Sparse Autoencoder（SAE）**：将密集模型激活分解为稀疏潜在特征的自编码器，用于 mechanistic interpretability 分析。
**Forget Score / Retain Score**：分别衡量某稀疏特征与幻觉失败/干净成功之间关联强度的量化指标，用于选择遗忘和保留目标。
**Region-guided Feature Selection**：在语义图像区域内聚合视觉 token 激活后再分析稀疏特征的方法，保留空间定位信息。
**Policy Unlearning（策略遗忘）**：在不重新从头训练的情况下，有选择性地移除模型中与特定行为相关的内部知识。

## 可复现要素
- **数据集**：LIBERO-Plus（公开）、RoboCasa（公开）；仿真环境可复现
- **模型权重**：OpenVLA 和 $\pi_{0.5}$ 均提供公开发布的预训练权重
- **代码**：论文未明确声明开源，需关注作者主页或后续 release
- **关键超参**：SAE 隐层维度 4096，k=100，k_aux=512；每 rollout 最多采样 20 个时间步；选 top-5 区域-特征对作为遗忘/保留目标；使用 LoRA 适配器更新策略参数
- **硬件**：Franka Research 3（7-DoF）真实机器人环境
