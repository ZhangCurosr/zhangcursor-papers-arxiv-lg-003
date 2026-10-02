---
title: "Where-the-Model-Changes-Its-Mind-Hindsight-Divergence-Locali"
source: https://arxiv.org/pdf/2609.36864v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:56:39"
field: "大语言模型强化学习"
keywords: ["RLVR", "reinforcement learning", "branch-point selection", "hindsight divergence", "prefix reuse", "agentic RL", "group-relative policy optimization"]
innovations: ["提出 hindsight-divergence score 定位模型改主意的关键决策点", "基于反馈后验重打分的分支点选择无需额外训练目标", "Prefix reuse 配合局部后缀 loss 实现 rollout 成本大幅缩减"]
benchmarks: ["AIME24/25/26", "HMMT February 2026", "Minerva Math", "OlympiadBench", "LiveCodeBench v5/v6", "ScienceWorld"]
---

# 论文速读：Where-the-Model-Changes-Its-Mind-Hindsight-Divergence-Locali

## 一句话总结
本文提出** hindsight-divergence Localization（HDL）**方法，利用验证器反馈后的反思信号来定位模型在推理过程中需要"改主意"的关键决策点，从这些分支点重新采样续接轨迹，从而以更少生成token实现高效的 RLVR 训练，在数学、代码和 Agent 任务上均取得性能提升。

## 研究问题与动机
- **RLVR  rollout 成本高昂**：Group-relative 方法（如 GRPO）对每个 prompt 独立采样多条完整轨迹，rollout 生成是训练的主要开销，尤其对长推理链和 Agent 任务更为突出。
- **现有方法无法同时兼顾成本与探索质量**：Token-selective 方法（如 Wang et al., 2025）仅筛选高熵 token 更新，但不减少完整轨迹的生成成本；TreeRL/BPO 等基于策略不确定性的分支方法未利用最终结果反馈来重新评估之前的决策。
- **已完成轨迹的反馈揭示了值得重采样的决策点**：Reflection-based 方法（PivoARL、R³L）引入额外训练目标来学习反思能力，而本文希望仅通过后验重打分来定位分支点，无需额外目标。
- **核心问题**：能否在生成阶段将 rollout 预算定向分配到选定位置的替代续接，同时复用其前缀，以减少生成成本并聚焦关键决策的探索？

## 核心贡献（创新点）
1. **提出 HDL（Hindsight-Divergence Localization）框架**：通过反馈驱动的反思信号重打分 token log-likelihood，定位模型"改主意"的分支点，而非依赖熵或显式反思提示。
2. **基于 hindsight-divergence score 的分支点选择机制**：定义 $s_i = |\log p_i^H(y_i) - \log p_i^0(y_i)|$，用绝对 log-likelihood 变化量排序候选位置，捕捉反馈前后模型对某 token 评估的变化程度。
3. **Prefix reuse + 局部续接采样策略**：从选定的分支点复用根轨迹前缀并采样新鲜后缀，使每个续接仅对新生成后缀施加策略损失，大幅降低生成开销同时保持 group-relative 目标不变。
4. **在 Math/Code/Agent 三领域均验证有效性**：相比 GRPO 节省 35–61% token、加速 18–45% rollout 时间的同时提升性能；相比 Entropy 和 Reflection 等替代定位信号方法全面领先。

## 方法详解
- **整体流程**：对每个问题 $x$，HDL 先采样 $M < G$ 条完整根轨迹（默认 $M=2$），对每条根轨迹通过 verifier feedback $f$ 生成反思 $r$，组成 hindsight 上下文 $h = [f; r]$，再据此选择分支点并构造训练组。
- **Hindsight-conditioned scoring**：对根轨迹中每个 token $y_i$，分别在原始上下文（仅前缀 $y_{<i}$）和 hindsight 上下文（$h + y_{<i}$）下计算 next-token 分布 $p_i^0$ 和 $p_i^H$，两者使用相同策略参数 $\theta_{\text{old}}$，仅上下文不同。
- **Branch-point selection**：定义 hindsight-divergence score $s_i = |\log p_i^H(y_i) - \log p_i^0(y_i)|$，对每条根轨迹按 $s_i$ 排序选取最高分的位置作为分支点（默认每条根选 2 个分支点）。
- **Localized group construction**：给定根轨迹 $y$ 和分支点 $i$，复用前缀 $y_{<i}$ 并在原始任务上下文 $x$ 下重新采样后缀 $\tilde{y}_{\geq i} \sim \pi_{\theta_{\text{old}}}(\cdot | x, y_{<i})$，形成完整轨迹 $\tilde{y} = y_{<i} \| \tilde{y}_{\geq i}$。每组包含 $M$ 条根 + $(G-M)$ 条续接。
- **训练目标**：延续 GRPO 的 group-relative 优势函数 $A_j = R_j - \frac{1}{G}\sum_k R_k$；根轨迹对所有生成 token 计算 loss，续接轨迹仅对新生成的后缀部分计算 loss，避免重复计数共享前缀。

## 实验与结果
- **实验设置**：三个模型（Qwen3-4B、Qwen3-8B、Llama3.1-Nemotron-Nano-8B），三个领域（Math/DeepMath-103K、Code/DeepCoder+rStar-Coder、Agent/ScienceWorld），每组 $G=16$，200 步优化，学习率 $10^{-6}$。
- **Rollout 效率**：相比 GRPO，HDL 减少 35–61% 生成 token，加速 18–45% rollout 壁钟时间；相比 DAPO 减少 43–75% token，加速 1.52×–3.18×。
- **性能提升**：
  - **Math**：Qwen3-8B 上 HDL 平均 54.16%，优于 GRPO（53.17%）和 DAPO（53.47%）；Qwen3-4B 上 HDL 51.86%，接近 DAPO 52.48%（后者消耗近 4 倍 token）。
  - **Code**：Qwen3-4B 和 Qwen3-8B 上 HDL 均取得最高准确率（54.15% 和 55.56%）。
  - **Agent**：Qwen3-4B 上 HDL 67.44% vs GRPO 57.76%（+9.68pt）；Qwen3-8B 上 HDL 71.96% vs GRPO 59.50%（+12.46pt），领先幅度最大。
- **定位信号对比**：HDL 在三个领域均优于 Entropy 和 Reflection 基线，Agent 任务上领先优势最大（vs Entropy +5.67pt，vs Reflection +6.54pt）。
- **分支配置消融**：默认 2×2（每条根选 2 个分支点，分配 3+4 条续接）表现最佳；减少分支点至 1 个仅降 0.87pt 但省 15% 时间；增加根数至 4 条反而降 4.09pt 且增 cost。

## 相关工作脉络
1. **GRPO（Shao et al., 2024）**：Group-relative 方法的基础，独立采样完整轨迹；HDL 保留相同目标但通过 prefix reuse 减少生成量。
2. **TreeRL（Hou et al., 2025）/ BPO（He et al., 2026）**：基于策略熵的分支方法；HDL 与之区别在于利用结果反馈后的 hindsight 重打分而非仅靠模型不确定性。
3. **PivoARL（Guo et al., 2026）/ R³L（Shi et al., 2026）**：基于显式反思定位重试点，需额外训练目标；HDL 无需额外目标，直接通过后验 log-likelihood 变化量定位。
4. **SDPO/RLSD/SRPO/HINT-SD/HSD**：On-policy self-distillation 方法，利用 hindsight 进行 token 级监督；HDL 借鉴其"比较有无 hindsight 的 token 概率"思路，但用途不同——用于分支点排名而非监督信号。
5. **Token-selective RL（Wang et al., 2025）**：仅在高熵 token 上更新；HDL 与之正交，聚焦于从何处分支而非更新哪些 token。
6. **DAPO（Yu et al., 2025）**：动态采样过滤均匀奖励组；HDL 在相同 group size 下进一步减少生成开销。

## 局限性与未来方向
- **对小模型依赖性强**：Qwen3-1.7B 上 HDL 的反思 outcome 对齐率仅 76.0%（Math）和 56.1%（Code），HDL 在 Agent 任务上退化为与 GRPO 持平（45.70% vs 45.66%），说明模型需具备可靠理解反馈的能力。
- **Reflection 生成的位置可能重复或无法解析**：Reflection 基线平均仅得 3.22 个有效分支点（最多 4 个），而 HDL 为 3.95 个。
- **未来方向**：探索更鲁棒的分支点选择机制以适应更小模型；将 HDL 扩展到更长 horizon 的 agentic 任务；研究分支点数量与续接分配的自动调度策略。

## 研究启发与可借鉴点
1. **Hindsight-divergence score 可作为通用的"改主意"定位指标**：该信号不依赖显式反思格式，只需后验重打分，可迁移至其他基于 rollout 的 RL 方法（如 PPO、Reinforce+）中用于智能采样。
2. **Prefix reuse + 局部 loss 的设计范式**：复用根前缀仅对后缀施加 policy gradient loss，兼顾 exploration 效率和 credit assignment 准确性，可推广至任何需要 tree-structured rollout 的场景。
3. **实验设计值得借鉴**：同 group size、同训练步骤、同硬件条件下的公平对比；同时对 rollout efficiency（token 数、wall-clock time）和 task performance 双维度报告，完整呈现 trade-off。
4. **Agent 任务的 credit assignment 挑战**：HDL 在 ScienceWorld 上取得最大增益（+12.46pt），说明在长 horizon 多步决策中定位关键转折点具有更大价值，可启发团队在类似交互任务中应用此思路。
5. **定位信号的系统化对比**：Entropy vs Reflection vs HDL 的分支点分布分析（如图 5/6 中 verb/argument 选择性）提供了深入理解不同信号行为的方法论，值得在后续工作中复用。

## 关键术语表
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：利用可验证奖励信号对 LLM 进行强化学习训练的方法，典型应用于数学推理、代码生成等场景。
- **Group-relative methods（群相对方法）**：通过同 prompt 下多条轨迹的奖励相对差异（如优势函数）提供学习信号，代表方法有 GRPO。
- **Hindsight-divergence score**：衡量某一 token 在有无 hindsight 上下文时 log-likelihood 的绝对变化量，用于定位模型"改主意"的决策点。
- **Prefix reuse**：续接轨迹复用根轨迹的前缀部分，仅对新生成的后缀施加策略损失，降低生成成本。
- **Reflection（反思）**：利用 verifier feedback 让模型生成对自身轨迹的解释，HDL 中的反思用于构建 hindsight 上下文。
- **Credit assignment**：在长轨迹中确定哪些决策对最终奖励负有责任的问题，Agent 任务中的核心挑战。
- **On-policy self-distillation**：模型在自身采样轨迹上以 hindsight 条件版本作为教师进行自蒸馏训练。
- **ScienceWorld**：文本交互式 Agent 评测环境，要求模型通过自然语言动作完成科学实验任务。

## 可复现要素
- **数据集**：DeepMath-103K、DeepCoder（TACO+PrimeIntellect）、rStar-Coder seed_testcase、ScienceWorld——均为公开数据集。
- **代码/权重**：slime 框架开源（https://github.com/THUDM/slime）；模型为 Qwen3-4B/8B 和 Llama3.1-Nemotron-Nano-8B-v1（公开权重）；论文未声明 HDL 代码单独开源。
- **关键超参**：每组 $G=16$ 条轨迹，根轨迹 $M=2$，每根选 2 个分支点，分配 3+4 条续接；优化步数 200，学习率 $10^{-6}$，温度 1.0；Math/Code 截断 32768 token，Agent Qwen3 截断 8192 token、Llama3.1 截断 16384 token；不对称 clipping 阈值 0.2/0.28，无 KL penalty。
