---
title: "ON-POLICY-SELF-DISTILLATION-FOR-MULTI-TURN-IMAGE-EDITING"
source: https://arxiv.org/pdf/2609.35611v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:51:39"
field: "多轮图像编辑"
keywords: ["multi-turn image editing", "on-policy self-distillation", "train-test mismatch", "flow matching", "exposure bias", "image editing benchmark"]
innovations: ["提出MT-OPSD自蒸馏框架，无需多轮标注即可训练模型在自身生成的退化状态上保持编辑能力", "构建LME-Bench基准，包含100个十轮混合局部/全局编辑会话及SR/CR评测协议", "设计自适应rollout课程+门控教师晋升机制，以漂移量为信号动态推进训练难度"]
benchmarks: ["LME-Bench", "MSE-Bench", "ImgEdit"]
---

# 论文速读：ON-POLICY SELF-DISTILLATION FOR MULTI-TURN IMAGE EDITING

## 一句话总结
本文提出 MT-OPSD，一种基于 on-policy 自蒸馏的多轮图像编辑框架，通过将预训练模型在自身生成的退化状态上以干净源图像为参考进行蒸馏，无需多轮标注即可显著提升长程多轮编辑的稳定性。同时引入 LME-Bench 基准（100 个十轮编辑会话），系统揭示了现有开源编辑模型在多轮迭代中的快速退化问题。

## 研究问题与动机
- **多轮编辑退化普遍存在**：当前基于指令的图像编辑模型在单次编辑上表现优异，但在递归编辑（每次以上一轮输出为输入）时，微小的模型诱导误差会逐轮累积，导致高频色噪、结构碎片化和严重的身份漂移。
- **条件分布的训练–测试不匹配**：模型仅在干净源图像上训练，推理时却必须在其自身不完美的输出上继续编辑，这一 conditioning distribution mismatch 是多轮坍缩的根本原因。
- **多轮标注数据稀缺且端到端优化成本极高**：直接对多轮 rollout 进行端到端反向传播需跨越数百步去噪和多轮编辑，计算代价不可行。
- **无训练方法存在局限**：Emu Edit 等训练无偏方法通过像素回退缓解局部编辑漂移，但无法处理全局变换（重绘风格、换光等），因为全局变换下几乎所有像素都预期发生变化。

## 核心贡献（创新点）
- **揭示了多轮坍缩的归因机制**：系统性证明多轮退化是单轮训练范式的普遍缺陷而非模型特有问题，并将其归因于 conditioning distribution 的训练–测试不匹配（类比自回归视频生成中的 exposure bias）。
- **MT-OPSD：无需多轮标注的 on-policy 自蒸馏框架**：利用模型自身干净条件下的编辑行为作为教师信号，在自生成的 rollout 状态上进行蒸馏，无需外部教师或多轮真值图像，与 MT-EditFlow（依赖 RL + 外部奖励）形成本质区别。
- **LME-Bench 基准**：构建包含 100 个十轮会话（共 1000 条指令，混合 6 次局部 + 4 次全局编辑）的评测基准，提供 SR@k 和 CR@k 两个互补指标，填补了长程多轮鲁棒性评估的空白。
- **跨三架构的显著增益**：在 Qwen-Image-Edit-2511、FireRed-Image-Edit、FLUX.2-klein-base 三个 backbone 上均实现 SR@10 提升 0.25~0.41、CR@10 降至 ≤0.04，同时几乎不损失 ImgEdit 单轮编辑质量。

## 方法详解
MT-OPSD 由四个核心组件构成：

**1. 自生成 Rollout 状态（Self-Generated Rollout States）**
从干净源图像 $I^{(0)}$ 出发，递归应用当前学生模型并注入恒等指令 $e_{\text{id}}$（如"Make everything unchanged"），构造包含模型诱导误差的退化状态：
$$\tilde{I}^{(k)} = G_{\theta_S}(\tilde{I}^{(k-1)}, e_{\text{id}}), \quad \tilde{I}^{(0)} = I^{(0)}$$
使用恒等指令而非真实编辑指令，确保 rollout 过程中目标语义内容不变，使 $\tilde{I}^{(k)}$ 与 $I^{(0)}$ 的差异仅反映模型诱导误差的累积，从而隔离出需要学习的误差分量。

**2. 双分支训练目标（Two-Branch Training Objective）**
对每个 rollout 状态 $\tilde{I}^{(k)}$，以 2:1 比例采样两个互补分支：

- **恒等分支（Identity Branch）**：防止误差进一步放大，以 rollout 状态自身为目标的 flow matching 损失（而非恢复到干净源图像，因恢复 k 步累积误差难度过高）：
$$\mathcal{L}_{\text{id}} = \mathbb{E}_{t,\epsilon}\left[\|v_{\theta_S}(\tilde{x}_t, t, \tilde{I}^{(k)}, e_{\text{id}}) - (\epsilon - \tilde{x}_0)\|_2^2\right]$$

- **编辑分支（Editing Branch）**：以干净源图像为条件的冻结教师 $\theta_T$ 作为参考，对学生在 $\tilde{I}^{(k)}$ 条件下的编辑行为进行稀疏查询式速度匹配（sparse query-based velocity matching）：
$$\mathcal{L}_{\text{edit}} = \mathbb{E}_{q \sim p_q}\left[\|v_{\theta_S}(\bar{x}_{t_q}, t_q, \tilde{I}^{(k)}, e) - v_{\theta_T}(\bar{x}_{t_q}, t_q, I^{(0)}, e)\|_2^2\right]$$
其中 $\bar{x}_{t_q} = \text{sg}(x_{t_q})$ 为 stop-gradient 的学生查询状态，查询步骤从 Beta 分布采样（Qwen/FireRed 用 Beta(5,5)，FLUX 用 Beta(2,5)）。教师保留干净条件，学生接受退化条件，两者在相同 CFG scale 下匹配。

**3. 自适应 Rollout 课程（Adaptive Rollout Curriculum）**
以 rollout 漂移（$\tilde{I}^{(k)}$ 与 $I^{(0)}$ 的均值像素偏差）为监控指标：当漂移低于阈值 $\tau$ 持续 $P$ 步后，深度 $k$ 递增 1。这一机制避免初期以过深深度暴露于重度退化状态导致优化失稳。实践中课程通常在约 4 轮时饱和，但仍能泛化至 10 轮测试。

**4. 门控教师晋升（Gated Teacher Promotion）**
异步评估学生 checkpoint，由 VLM judge 在 held-out gate set（23 个十轮会话）上比较候选与当前教师在各轮次的 $\Delta\text{SR}$ 和 $\Delta\text{CR}$，满足成功/稳定阈值（$\alpha, \beta$ 因 backbone 而异）则晋升。教师晋升仅用于选择参考，不作为训练目标或奖励信号。

## 实验与结果
**数据集与基准**：
- **LME-Bench**（本文提出）：100 个十轮会话，1024×1024 分辨率，Z-Image-Turbo 生成，10 类语义均匀分布，每会话含 6 次局部 + 4 次全局编辑（硬/软各半），使用 GPT-4o 作为 judge 评估 prompt following 和 consistency。
- **MSE-Bench**（Qu et al., 2026）：100 个五轮会话，以局部编辑为主。
- **ImgEdit**（Ye et al., 2026a）：标准单轮编辑评测。

**实验设置**：基于 OmniEdit 的约 2,000 对 source-image/instruction 训练，LoRA rank=32、α=64，训练分辨率 512²，8× NVIDIA H100/H200 GPU。

**主要结果（LME-Bench，Table 1）**：

| Backbone | 方法 | SR@10 ↑ | CR@10 ↓ |
|---|---|---|---|
| Qwen-Image-Edit-2511 | Base | 0.03 | 0.55 |
| | + Emu Edit | 0.13 | 0.30 |
| | + VAE-LFA | 0.10 | 0.36 |
| | **+ MT-OPSD** | **0.44** | **0.02** |
| FireRed-Image-Edit | Base | 0.15 | 0.61 |
| | + VAE-LFA | 0.37 | 0.33 |
| | **+ MT-OPSD** | **0.52** | **0.03** |
| FLUX.2-klein-base | Base | 0.12 | 0.25 |
| | + VAE-LFA | 0.16 | 0.24 |
| | **+ MT-OPSD** | **0.38** | **0.04** |

- MT-OPSD 将三个 open-source backbone 的 SR@10 从 0.03–0.15 提升至 0.38–0.52，CR@10 降至 ≤0.04（基线最高 0.61），显著提升长程鲁棒性。
- 超越 GPT-Image-1（SR@10=0.01, CR@10=0.69），但弱于更新的 GPT-Image-2 和 Nano Banana Pro。
- **MSE-Bench**（Table 2）：Qwen 从 SR@5=0.24 提升至 0.51；FireRed 从 0.40 提升至 0.69；FLUX 维持在 0.49。
- **ImgEdit**：单轮编辑质量基本无损（Qwen: 4.51→4.49，FireRed: 4.56→4.52，FLUX: 4.20→4.28）。

## 相关工作脉络
- **InstructPix2Pix (Brooks et al., 2023) / OmniEdit (Wei et al., 2025) / Qwen-Image-Edit-2511 (Wu et al., 2025a)**：单轮指令编辑的代表性工作，本文在其基础之上解决多轮鲁棒性问题，定位为"在强单轮能力上的扩展"。
- **Emu Edit (Sheynin et al., 2024) / FreqEdit (Liao et al., 2026) / VAE-LFA (Wang et al., 2026)**：训练无偏的多轮修正方法，通过像素回退、高频恢复、VAE 低频对齐缓解退化，但不改变模型行为本身，对全局变换无效——本文方法在能力和适用性上更全面。
- **VINCIE (Qu et al., 2026) / AnchorEdit (Xu et al., 2026) / MT-EditFlow (Huang et al., 2026a)**：训练型多轮编辑方法。VINCIE 依赖完整编辑历史 conditioning，本文仅用上一轮输出即可达到更高 SR@5/SR@10；MT-EditFlow 使用 RL + 外部奖励，本文用自蒸馏无需额外奖励设计。
- **On-Policy Self-Distillation (D-OPSD, OPSD-V, DanceOPD, DiffusionOPD)**：视觉生成领域的 on-policy 蒸馏方法依赖配对目标图或 reward gradient，本文的 on-policy state 是模型自身的 rollouted 图像条件，无需外部配对数据，是最独特的适配形式。

## 局限性与未来方向
- **Rollout 课程饱和深度有限**：自适应课程通常在约 4 轮时饱和，但评测延伸至 10 轮，长期稳定性仍依赖蒸馏效果的外推，更深 rollouts 的训练效率未充分探索。
- **训练数据规模较小**：仅使用 2,000 对 source-image/instruction 进行微调，在更大数据量下的泛化能力有待验证。
- **依赖 VLM judge 的评估偏差**：SR@k 和 CR@k 均基于 GPT-4o 打分，judge 的主观性和 prompt 设计可能影响结果可靠性。
- **需要强预训练初始化**：方法依赖预训练模型已具备可靠的单轮编辑能力，对弱初始化模型的适用性未讨论。
- **全局编辑的挑战**：虽然 MT-OPSD 能处理全局变换（这是优于训练无偏方法的关键），但 Table 1 显示 Nano Banana 在全局编辑上的较低 SR 主要源于早期全局编辑失败，暗示全局变换仍是更难的子任务。

## 研究启发与可借鉴点
- **身份 Rollout 隔离误差的设计**：用恒等指令让模型"徒劳地"尝试复制自身输入，从而累积纯模型误差——这一思路可迁移至任何自回归/迭代生成任务中的 train-test mismatch 诊断与缓解（如视频生成、多步推理）。
- **自蒸馏中"学生状态 + 教师上下文"的不对称设计**：学生接受退化条件，教师接受干净条件，两者共享同一模型权重但条件不同——这是一种无需额外参数的零成本教师构建策略，适用于 diffusion/flow matching 之外的更多生成范式。
- **基于漂移量的自适应课程**：以 mean pixel deviation 为监控信号自动推进训练难度，比固定 schedule 更高效且免调参，可推广至其他 curriculum learning 场景。
- **门控晋升的异步评估机制**：VLM judge 仅用于 checkpoint 选择而非训练目标，避免了 reward hacking 风险，是评估驱动模型选择的实用范式。
- **LME-Bench 的会话结构设计**：硬性约束（全局编辑不相邻、最后两轮为局部、有效指令六大家族规则）确保每个 turn 都有信息量——这一评测构造方法论对其他多轮生成基准有参考价值。

## 关键术语表
**On-Policy Self-Distillation (OPSD)**：学生在自身策略生成的状态下训练，同时由同一模型在更丰富条件下的版本提供教师监督，无需外部教师。
**Rollout（身份 Rollout）**：从干净源图像出发递归应用恒等指令生成的一系列退化状态，用于模拟多轮编辑中的误差累积。
**Sparse Query-Based Velocity Matching**：在学生去噪轨迹的少量随机查询步上，匹配教师在同一时刻的预测速度，实现高效的蒸馏监督。
**SR@k（Success Rate at turn k）**：前 k 轮全部成功的会话比例，衡量累积编辑正确性。
**CR@k（Collapse Rate at turn k）**：到第 k 轮已发生持续视觉退化的会话比例（连续两轮被判定为退化即视为 collapse）。
**Gated Teacher Promotion**：异步评估学生 checkpoint 并在满足多轮成功/低坍缩阈值时用其替换当前教师，促进蒸馏参考的动态升级。
**LME-Bench**：Long Multi-turn Image Editing Bench，包含 100 个十轮会话的长程多轮编辑评测基准。
**Conditional Distribution Mismatch**：模型训练时以干净图像为条件、推理时以自身退化输出为条件的分布不匹配，是多轮编辑退化的根本原因。

## 可复现要素
- **数据集**：LME-Bench 代码和权重将在项目页面公开（https://liangbingzhao.github.io/MT-OPSD/）；训练数据为 OmniEdit 的约 2,000 对样本；源图像由 Z-Image-Turbo 生成（种子编号保证可复现）。
- **代码/权重**：论文声明项目页面将提供，代码开源状态需见最终发布；LoRA 权重为适配方式。
- **关键超参**：LoRA rank=32、α=64、训练分辨率 512²、Identity loss weight=10、初始 rollout 深度=2、最大深度 $K_{\max}=10$、漂移阈值 τ=9~10、patience=10、query 分布 Beta(5,5) 或 Beta(2,5)、学习率 1~3×10⁻⁴。
