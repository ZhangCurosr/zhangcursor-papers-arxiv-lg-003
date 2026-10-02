---
title: "Where-the-Model-Changes-Its-Mind-Hindsight-Divergence-Locali"
source: https://arxiv.org/pdf/2609.36864v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:56:25"
field: "大语言模型强化学习"
keywords: ["RLVR", "HDL", "后视分歧", "rollout 效率", "组相对策略优化", "分支点定位", "前缀复用"]
innovations: ["利用事后反馈引发的 token 对数似然变化定位分支点，替代纯熵或反射方法", "前缀复用 + 后缀 loss 聚焦的高效组构建，在不减小组规模前提下削减 35–61% 生成 token", "将事后自蒸馏范式从 token 级监督迁移到 rollout 预算分配，保持 GRPO 目标不变"]
benchmarks: ["AIME24/25/26", "HMMT February 2026", "Minerva Math", "OlympiadBench", "LiveCodeBench v5/v6", "ScienceWorld"]
---

# 论文速读：Where-the-Model-Changes-Its-Mind-Hindsight-Divergence-Locali

## 一句话总结
论文提出 **HDL（Hindsight-Divergence Localization）**，利用事后反馈引发的 token 对数似然变化来定位关键分支点，通过复用前缀 + 局部续接采样的方式，在不减少训练组规模的前提下将 RLVR 的 rollout 生成成本降低 35–61%，同时在数学、代码和 agent 任务上实现性能提升。

## 研究问题与动机
- **独立完整轨迹采样成本过高**：Group-relative RLVR 方法（如 GRPO）每次为同一 prompt 独立采样 G 条完整轨迹，对长推理轨迹和 agentic 任务尤其昂贵。
- **已有分支方法未利用结果反馈重新评估早期决策**：TreeRL/BPO 用策略熵选择分支点，但未考虑 verifier 反馈；Reflection-based 方法（PivoARL、R³L）虽能识别重试点，但需引入额外训练目标培养反思能力。
- **Token 贡献不均但生成成本未缩减**：Wang et al. (2025) 表明仅需对高熵 token 做更新即可匹配全量更新，但这类方法仍未减少完整轨迹的生成开销。
- **事后重评分能揭示"模型何时改变主意"**：事后自蒸馏方法证明利用反馈后的 token 概率变化可以定位关键决策，本文将其用于 rollout 预算分配。

## 核心贡献（创新点）
1. **提出以事后分歧分数定位分支点**：通过比较原始上下文与事后上下文（反馈+反思）下同一 token 的对数似然绝对差值来排名候选位置，本质区别在于利用已发生结果重新评估决策，而非仅依赖策略不确定性。
2. **前缀复用 + 续接采样降低 rollout 成本**：从少数完整根轨迹中选取分支点，其余槽位用相同前缀在原始任务上下文中采样新鲜后缀，训练 loss 仅作用于新后缀，避免重复计算前缀。
3. **保留组相对目标不变，仅改变组构建策略**：续接轨迹与根轨迹共享同一 GRPO 目标，无需引入额外损失或训练阶段，与 TreeRL/BPO/PivoARL 等依赖不同训练信号的 baselines 形成对比。
4. **在数学、代码、agent 三域同步提升性能并压缩开销**：最大节省 61% 生成 token、1.8× 加速 rollout 时延，同时 agent 任务上较 GRPO 提升 12.5 分。

## 方法详解
- **事后条件评分**：给定问题 $x$、根轨迹 $y$、verifier 反馈 $f$ 及策略生成的反思 $r$，构造事后上下文 $h = [f; r]$。对每个 sampled token $y_i$，分别计算原始分布 $p_i^0(\cdot) = \pi_{\theta_{\text{old}}}(\cdot \mid x, y_{<i})$ 和事后分布 $p_i^H(\cdot) = \pi_{\theta_{\text{old}}}(\cdot \mid x, h, y_{<i})$。
- **事后分歧分数**：$s_i = |\log p_i^H(y_i) - \log p_i^0(y_i)|$，取绝对值同时捕获成功轨迹中的强化和失败轨迹中的削弱。
- **分支点选择**：在每个根轨迹内按 $s_i$ 降序排名，选取最高分位置作为分支点（默认每根 2 个分支点）。
- **局部组构建**：每步采样 $M=2$ 条完整根轨迹，其余 $G-M=14$ 个槽位由各根的续接轨迹填充（每个分支点分配 3 或 4 条续接）。续接公式：$\tilde{y}_{\ge i} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x, y_{<i})$，复用前缀 $y_{<i}$ 并采样新鲜后缀。
- **训练目标**：组内 GRPO advantage $A_j = R_j - \frac{1}{G}\sum_k R_k$；根轨迹在所有 token 上计算 loss，续接轨迹仅在新生成的后缀上计算 loss。
- **反射 prompt**（Appendix B.1）：要求模型用自己的话总结轨迹方法、成功/失败原因及正确做法，限定 60–120 tokens，禁止引用原文或提及行号。

## 实验与结果
- **数据集**：Math — DeepMath-103K（过滤后 4,555 题）；Code — DeepCoder（TACO + PrimeIntellect）与 rStar-Coder（seed_testcase），共 4,063 题；Agent — ScienceWorld（1,856 任务-变体对）。
- **模型**：Qwen3-4B、Qwen3-8B、Llama3.1-Nemotron-Nano-8B-v1。
- **训练设置**：slime 框架，4 节点 × GB200 GPU，每步 128 题、每组 $G=16$、200 步、lr=$10^{-6}$、temperature=1.0；Math/Code 上限 32,768 token，Agent 上限 8,192/16,384 token。
- **效率提升**：相对 GRPO，HDL 减少 35–61% 生成 token（Qwen3-8B Math 从 27.28M 降至 10.59M/步）、18–45% 端到端 rollout 墙钟时间；相对 DAPO 节省 43–75% token、34–69% 时间。
- **性能提升**：
  - Math：Qwen3-8B Avg 达 54.16%，优于 GRPO（53.17%）和 DAPO（53.47%）。
  - Code：Qwen3-4B 54.15%、Qwen3-8B 55.56%，均为三者最高。
  - Agent：Qwen3-4B 67.44%（较 GRPO +9.68 pts）、Qwen3-8B 71.96%（较 GRPO +12.46 pts），均大幅领先 DAPO。
- **定位信号对比**（Table 4，Qwen3-8B）：HDL 在所有三域均优于 Entropy 和 Reflection，Agent 上分别高出 5.67 和 6.54 分。
- **分支配置对比**（Table 5）：默认 $2 \times 2$（每根 2 点、共 16 条）得分最高（71.96%）；$2 \times 1$ 提速 15% 但降 0.87 分；$4 \times 2$ 反而降 4.09 分且增 token。

## 相关工作脉络
- **GRPO（Shao et al., 2024）**：组相对策略优化基线，独立采样完整轨迹，HDL 在其基础上仅修改组构建策略。
- **TreeRL（Hou et al., 2025）/ BPO（He et al., 2026）**：基于策略熵选择分支点，不利用 verifier 反馈重新评估，HDL 用事后重评分替代纯熵。
- **PivoARL（Guo et al., 2026）/ R³L（Shi et al., 2026）**：通过显式反思识别重试点，需额外训练目标；HDL 无需额外训练，直接复用 GRPO 目标。
- **SDPO / RLSD / HSD（Hußotter et al., 2026; Yang et al., 2026; Li et al., 2026b）**：事后自蒸馏类工作，用 hindsight 指导 token 级监督；HDL 仅用分歧分数定位分支点，不做 token 级重加权。
- **Token-selective RL（Wang et al., 2025）**：仅在高熵 token 上做更新但不减少生成成本；HDL 从生成阶段即通过前缀复用压缩预算。
- **AReL / DAPO（Fu et al., 2025; Yu et al., 2025）**：异步/动态采样系统；HDL 与其正交，可组合使用。

## 局限性与未来方向
- **小模型适用性受限**：Qwen3-1.7B 上 outcome-label agreement 仅 76%（Math）和 56%（Code），HDL 在 ScienceWorld 上与 GRPO 持平（45.70% vs 45.66%），落后 Entropy（47.54%），说明依赖模型可靠解释反馈的前提在小模型上失效。
- **反思质量直接影响分支排名**：当模型幻觉或误判成功/失败时，事后重评分可能引入噪声。
- **默认配置 $M=2$ 未充分探索**：Section 4.5 仅对比了少数配置，最优 $M$ 随任务/模型规模的变化规律尚不明确。
- **未来方向**：① 设计对反思噪声鲁棒的分支点估计；② 探索 $M$ 的动态调度策略；③ 将 HDL 与异步 RL（如 AReL）或小模型专用变体结合。

## 研究启发与可借鉴点
- **事后分歧作为分支点信号的有效性与普适性**：HDL 将事后自蒸馏的思路从"token 级监督"转向"rollout 预算分配"，这一范式迁移可直接复用到其他 group-relative RLVR 场景。
- **前缀复用 + 后缀 loss 聚焦的高效组构建模式**：续接轨迹仅对新生成后缀计算 loss 的设计避免了前缀重复计算，可迁移至任何需要局部重采样的序列生成任务。
- **定位信号对比实验设计值得借鉴**：Section 4.4 在严格等控制条件下比较 HDL/Entropy/Reflection，并深入分析分支点分布差异（Figure 5–6），这种"信号选择 ablation + 行为分析"的组合为后续工作提供标准对照范式。
- **小模型脆弱性的显式披露**：Appendix A 公开了 1.7B 模型的失败案例，为团队选择方法时的规模门槛提供了定量参考（建议 $\ge$ 4B 或确保反思准确率 >95%）。
- **与动态采样/异步 RL 的兼容性**：HDL 不改写优化目标，可与 DAPO、AReL 等方法叠加，为多优化路径融合提供接口。

## 关键术语表
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：使用可自动验证正确性的奖励信号（如答案是否匹配、测试是否通过）进行策略优化的强化学习范式。
- **HDL（Hindsight-Divergence Localization）**：本文提出的方法，通过事后反馈引发的 token 对数似然变化来定位 rollouts 的分支点。
- **GRPO（Group-Relative Policy Optimization）**：组内相对比较的策略梯度方法，通过计算组内平均 reward 为中心的优势函数实现无 baseline 的 policy update。
- **Hindsight-divergence score**：同一 token 在原始上下文与事后上下文（反馈+反思）下对数似然差值的绝对值，用于排名候选分支点。
- **Root trajectory**：每步中完整采样的轨迹，作为续接轨迹的前缀来源。
- **Continuation trajectory**：复用根轨迹前缀并从分支点重新采样后缀生成的轨迹，仅对新后缀计算策略 loss。
- **Reflection**：由策略生成的对轨迹成功/失败原因的自然语言解释，与 verifier feedback 共同构成事后上下文。
- **Branch point**：根据事后分歧分数选出的关键决策位置，续接轨迹从此处开始独立采样。

## 可复现要素
- **数据集**：DeepMath-103K（公开）、ScienceWorld（公开）、DeepCoder TACO/PrimeIntellect 子集（开源）、rStar-Coder seed_testcase（开源）；过滤后规模见实验设置。
- **代码/权重**：slime 框架开源（https://github.com/THUDM/slime）；模型使用 Qwen3-4B/8B 与 Llama3.1-Nemotron-Nano-8B-v1（均开源）。
- **关键超参**：$G=16$、$M=2$、每步 128 题、200 优化步、lr=$10^{-6}$、temperature=1.0；数学/代码 max 32,768 token，Agent max 8,192（Qwen3）/16,384（Llama3.1）；不对称裁剪 [−0.2, 0.28]，无 KL 惩罚。
