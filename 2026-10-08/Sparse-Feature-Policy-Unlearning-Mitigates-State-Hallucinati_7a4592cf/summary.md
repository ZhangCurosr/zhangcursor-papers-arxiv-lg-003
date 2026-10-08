---
title: "Sparse-Feature-Policy-Unlearning-Mitigates-State-Hallucinati"
source: https://arxiv.org/pdf/2610.09496v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:11:01"
field: "机器人基础模型可靠性与可解释性"
keywords: ["Vision-Language-Action", "Machine Unlearning", "Sparse Autoencoder", "State Hallucination", "Mechanistic Interpretability", "Robot Policy"]
innovations: ["提出SOUL特征级策略遗忘方法，选择性抑制幻觉相关稀疏特征同时保留成功特征", "首次将状态幻像形式化为VLA反复失败模式并提供特征级诊断与干预框架", "区域引导的稀疏特征选择在机器人策略修改中显著优于全局/token级选择"]
benchmarks: ["LIBERO-Plus", "RoboCasa", "Franka Real-World"]
---

# 论文速读：Sparse-Feature-Policy-Unlearning-Mitigates-State-Hallucinati

## 一句话总结
论文提出了 **SOUL（Sparse feature pOlicy UnLearning）**，一种基于稀疏自编码器的特征级策略遗忘方法：通过识别与"状态幻像"相关的稀疏特征作为遗忘目标、与成功执行相关的特征作为保留目标，选择性抑制幻觉行为的内部表征，在减少VLA状态幻像的同时保持操作能力。

## 研究问题与动机
- **状态幻像（state hallucination）** 是VLA中一种反复出现的失败模式：模型在未达到实际机器人-物体状态时仍继续执行后续动作（如未抓到物体就执行搬运和放置），导致局部错误在多阶段传播。
- 现有方法主要关注推理时干预（如中间推理、测试时采样/验证）或故障检测与恢复，而非直接修改策略中导致反复失败的内部知识。
- 简单基于梯度上升的遗忘方法（最大化幻觉失败样本的动作预测损失）会严重损害任务成功率（降至0%），说明直接"忘记"失败样本会破坏紧密耦合的操作技能。
- 缺乏对VLA内部失败机制的精细化理解，难以定位可修改的明确目标以选择性消除有害行为。

## 核心贡献（创新点）
1. **首次系统定义并形式化"状态幻像"问题**：将幻像细分为抓握、搬运、放置三种类型，并建立CS（清洁成功）、HF（幻觉失败）、NF（非幻觉失败）的分类体系，区别于以往仅关注整体失败率的评估。
2. **通过视觉注意力与稀疏特征的双重分析揭示幻像的内在表征模式**：发现幻像执行伴随任务相关区域注意力减弱、空间分布更分散，且稀疏特征激活呈现可区分的时空模式，为机制可解释性到行动干预搭建了桥梁。
3. **提出SOUL特征级策略遗忘框架**：将稀疏特征显式分为"遗忘目标"和"保留目标"，通过联合优化 forgetting loss（抑制幻觉特征）和 retention loss（增强成功特征），实现选择性策略修改而非全参数微调。
4. **在两种VLA架构（OpenVLA、π₀.₅）及真实机器人上验证有效性**：平均幻觉率降低21.7%p、任务成功率提升22.1%p，同时保留89%的原有成功行为，证明了特征级遗忘在机器人基础模型中的可行性。

## 方法详解
**SAE特征提取**：
- 使用 Top-k SAE 对策略残差流中的 visual-token 激活进行分解，latent维度=4096，活跃特征数 k=100，辅助活跃数 k_aux=512。
- SAE编码器在训练后冻结，仅用于提取稀疏特征激活。

**区域条件特征激活**：
- 基于语义区域掩码（仿真提供或人工标注），计算特征 j 在区域 r 内的平均激活：
  $$z _ { r , j } ( x ) = \frac { 1 } { | \mathcal { R } _ { r } ( x ) | } \sum _ { p \in \mathcal { R } _ { r } ( x ) } z _ { p , j } ( x )$$
- 对每次执行取时间维度最大值 m_{e,r,j}，再按执行结果类型（CS/NF/HF）求平均得 \bar{m}_{c,r,j}。

**特征关联分数与特征选择**：
- 幻觉关联分数：$s_{r,j}^{\mathrm{hall}} = \bar{m}_{\mathrm{HF},r,j} - \frac{1}{2}(\bar{m}_{\mathrm{CS},r,j} + \bar{m}_{\mathrm{NF},r,j})$
- 成功关联分数：$s_{r,j}^{\mathrm{succ}} = \bar{m}_{\mathrm{CS},r,j} - \frac{1}{2}(\bar{m}_{\mathrm{HF},r,j} + \bar{m}_{\mathrm{NF},r,j})$
- 遗忘分数：$s_{r,j}^{\mathcal{F}} = s_{r,j}^{\mathrm{hall}} + s_{r,j}^{\mathrm{agg}}$，其中 $s_{r,j}^{\mathrm{agg}} = s_{r,j}^{\mathrm{sep}} - s_{r,j}^{\mathrm{succ}}$，$s_{r,j}^{\mathrm{sep}}$ 衡量与NF的区分度
- 选取 top-5 区域-特征对作为遗忘集 $\mathcal{F}$，top-5 作为保留集 $\mathcal{R}$

**策略更新（LoRA微调）**：
- 冻结原始VLA参数，仅优化LoRA适配器
- 遗忘损失：$\mathcal{L}_{\mathrm{forget}} = \mathbb{E}_{x \in D_{\mathrm{HF}}} \left[ \frac{1}{|\mathcal{F}|} \sum_{(r,j) \in \mathcal{F}} (z_{r,j}(x;\theta))^2 \right]$
- 保留损失：$\mathcal{L}_{\mathrm{retain}} = -\mathbb{E}_{x \in D_{\mathrm{CS}}} \left[ \frac{1}{|\mathcal{R}|} \sum_{(r,j) \in \mathcal{R}} z_{r,j}(x;\theta) \right]$
- 总损失：$\mathcal{L}_{\mathrm{unlearn}} = \mathcal{L}_{\mathrm{forget}} + \lambda_r \mathcal{L}_{\mathrm{retain}}$

## 实验与结果
**数据集与基线**：
- LIBERO-Plus（含多严重度视觉扰动）、RoboCasa（多样厨房环境）、Franka Research 3真实机器人（各50次rollout）
- 基线：原始预训练策略、Gradient Ascent遗忘（最大化HF样本动作损失）
- 模型：OpenVLA（LIBERO-Plus）、π₀.₅（RoboCasa）

**主要结果**：
| 方法 | 幻觉率↓ | 幻觉失败↓ | 总成功率↑ | 清洁成功率↑ |
|------|---------|-----------|-----------|-------------|
| OpenVLA原始 | 70.0% | 60.0% | 26.0% | 16.0% |
| Gradient Ascent | 84.0% | 84.0% | 0.0% | 0.0% |
| **SOUL** | **44.0%** | **38.0%** | **58.0%** | **52.0%** |
| π₀.₅原始 | 40.0% | 37.1% | 41.9% | 39.0% |
| Gradient Ascent | 60.0% | 59.0% | 4.8% | 3.8% |
| **SOUL** | **22.9%** | **21.0%** | **54.3%** | **52.4%** |

- 跨设置平均：幻觉率从55.0%降至33.5%（↓21.7%p），清洁成功率从27.5%升至52.2%（↑22.1%p）
- 真实机器人：清洁成功率平均提升20%p，幻觉失败减少22%p
- 行为保留率89%，成功纠正40%的原有失败执行，78%的幻像事件被修正

**消融实验**：
- 区域引导特征选择优于全局 pooling 和 tokenized 选择（幻觉率33.5% vs 45.8%/41.5%）
- 遗忘分数中的聚合项 $s^{agg}$ 和保留损失缺一不可：去掉任一项均导致清洁成功率下降

## 相关工作脉络
1. **VLA可靠性改进**（[7]-[13]）：聚焦推理时中间推理、故障检测恢复、测试时采样验证，属"输出层"干预；本文转向策略参数层面的选择性知识修改。
2. **机制可解释性干预**（[24][25]）：通过干预feed-forward方向或稀疏特征在推理时引导VLA；本文将其扩展为策略参数的持久化修改，无需每次推理干预。
3. **机器学习遗忘**（[26]-[31]）：主要在分类和LLM factual knowledge上研究；本文首次将feature-level selective unlearning引入机器人策略，面临技能强耦合的新挑战。
4. **稀疏自编码器**（[32]-[35]）：用于LLM/VLM特征分解与解释；本文创新性地将其用于机器人策略的"遗忘目标"显式化，实现从诊断到干预的跨越。
5. **机器人故障处理**（[12][21]）：failsafe和self-correcting方法依赖在线检测与恢复；本文从根源上减少幻觉行为的发生概率。

## 局限性与未来方向
- 仅在 pick-and-place 这一基础原语上验证，复杂多步骤任务的泛化性待检验。
- 语义区域标注依赖仿真引擎提供的mask或人工标注，真实场景的自动区域分割未讨论。
- 仅针对状态幻像这一种失败模式，其他幻像类型（如[13]提出的action hallucination）未涉及。
- 超参数（$\lambda_r$、遗忘/保留特征数量）缺乏系统性扫描，实际部署时需调优。
- 未讨论对策略long-horizon规划能力或其他副作用的影响。

## 研究启发与可借鉴点
1. **机制可解释性→策略修改的范式迁移**：将SAE特征分析从"诊断工具"升级为"行动指南"，为其他VLA失败模式（如drift、infeasible actions）的特征级干预提供了方法论模板。
2. **显式遗忘/保留目标设计**：通过关联分数同时满足"强关联幻觉"、"弱关联成功"、"与NF可区分"三个条件来选择遗忘特征，避免了简单阈值筛选的粗糙性，可推广至其他选择性遗忘场景。
3. **区域引导的空间结构保留**：消融表明 region-guided 特征选择显著优于 global pooling，说明空间定位信息对精确定位失败表征至关重要，后续工作可在更细粒度（如物体子部件）上探索。
4. **LoRA+特征损失的轻量策略更新**：仅微调适配器即可实现显著行为修正，为部署级策略迭代提供了低成本的工程路径。

## 关键术语表
**State Hallucination（状态幻像）**：VLA在未达到实际机器人-物体状态（如未抓握成功）时仍继续执行后续操作序列的反复失败模式。
**Sparse Autoencoder (SAE)**：将模型密集激活分解为稀疏潜在表示的自编码器，使单个特征可与可解释的行为模式关联。
**Forget Feature（遗忘特征）**：在幻觉失败执行中强烈激活、在成功执行中激活较弱的稀疏特征，作为策略遗忘的目标。
**Retain Feature（保留特征）**：在清洁成功执行中强烈激活的稀疏特征，作为策略保留的目标以维持操作能力。
**SOUL（Sparse feature pOlicy UnLearning）**：本文提出的特征级策略遗忘方法，通过抑制遗忘特征、增强保留特征来缓解状态幻像。
**Clean Success (CS)**：无任何状态幻像事件的完整任务成功执行。
**Hallucination Failure (HF)**：包含至少一次状态幻像事件且最终任务失败的执行。
**Feature Association Score（特征关联分数）**：衡量稀疏特征在不同执行结果类型间激活差异的量化指标，用于特征分类与选择。

## 可复现要素
- **数据集**：LIBERO-Plus（开源）、RoboCasa（开源）、Franka真实实验（论文未提供独立数据集链接）
- **代码/权重**：使用OpenVLA和π₀.₅公开预训练权重；SAE及SOUL代码论文未明确提及开源状态
- **关键超参**：SAE latent dim=4096，k=100，k_aux=512；遗忘/保留特征各选top-5对；每rollout均匀采样最多20个时间步；LoRA微调；λ_r值论文未具体说明
