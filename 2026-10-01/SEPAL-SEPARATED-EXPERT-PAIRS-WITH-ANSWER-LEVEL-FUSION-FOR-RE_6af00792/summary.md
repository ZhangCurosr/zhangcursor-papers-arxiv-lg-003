---
title: "SEPAL-SEPARATED-EXPERT-PAIRS-WITH-ANSWER-LEVEL-FUSION-FOR-RE"
source: https://arxiv.org/pdf/2609.39645v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:36:59"
field: "大语言模型推理与协作"
keywords: ["multi-agent collaboration", "actor-critic", "preference optimization", "self-consistency", "chain-of-thought", "answer-level fusion", "DPO"]
innovations: ["隔离角色对+答案级投票融合避免跨候选错误传播", "续写价值偏好学习训练私有Critic修正各角色轨迹", "三角色（Direct/Evidence/Verification）结构化多样性设计"]
benchmarks: ["MMLU", "BoolQ", "BBH", "SciQ", "ARC"]
---

# 论文速读：SEPAL-SEPARATED-EXPERT-PAIRS-WITH-ANSWER-LEVEL-FUSION-FOR-RE

## 一句话总结
SEPAL 提出将三个独立的 Actor-Critic 角色对（直接推理、证据 grounding、验证）隔离运行并通过最终答案级多数投票融合，避免了共享讨论中错误跨候选传播的风险，在 5 个开源基座和 5 个 QA 基准上平均提升 1.81 个百分点的宏观准确率。

## 研究问题与动机
1. **共享讨论导致错误传染**：现有 multi-agent 讨论方法中，早期错误可能被多个 agent 看到并在后续候选中复现，破坏投票所需的多样性。
2. **Self-consistency 缺乏纠正反馈**：自一致性通过独立采样保留多样性，但无法通过反馈纠正错误答案。
3. **单对 Actor-Critic 无法生成多个候选**：ACC-Collab 能通过学习反馈改进单一答案，但不产生可供投票的多条候选轨迹。
4. **投票收益取决于候选如何共同失败**：投票的有效性不仅取决于各候选个体精度，还取决于它们在反馈后的错误相关性；若修正使所有候选趋向同一错误前提，投票增益将被削弱。

## 核心贡献（创新点）
1. **提出隔离角色对 + 答案级后期融合的协作框架**：将 Critic 反馈限制在各自团队内部，仅在最终答案阶段进行投票，避免跨候选的错误传播——与 ACC-Collab 单对迭代、共享历史辩论的本质区别在于"局部修复、晚期聚合"。
2. **引入三种具不同推理目标的私有角色（Direct / Evidence / Verification）**：通过角色特定的 SFT 初始化赋予结构化的推理差异，而非仅靠采样随机性产生多样性——与 CoMM 等仅用角色 prompt 的区别在于每个角色有独立的 adapter 和训练轨迹。
3. **将 ACC-Collab 的续写价值偏好学习规则独立复制给三个角色**：使用 DPO 训练 Critic 和 Actor，以续写正确率为反馈质量指标，训练顺序为 Critic DPO → Actor DPO，每个角色完全独立——与 Multiagent Finetuning 的区别在于本文保持共享 backbone 和训练数据，仅独立 adapter。
4. **提供对修正动态的系统量化分析**：发现 R1（首轮 Critic 条件修订）捕获了 89.2% 的最终增益，后续轮次存在"修正 vs 倒退"的此消彼长，揭示固定 R4 协议下存在早停优化空间。

## 方法详解
**整体架构**：三个私有 Actor-Critic 团队并行运行，每队执行 4 轮修订后输出最终答案，系统对三个答案做多数投票（平票时退回 Direct 答案）。

**角色定义**：
- **Direct（D）**：直接推导，识别决定性事实或计算。
- **Evidence（E）**： grounding，基于相关定义/事实/段落证据选择答案。
- **Verification（V）**：独立重新求解，验证已选选项与主要替代方案的对比。

**局部生成依赖（公式 1）**：
$$a_i^0 \sim A_i(\cdot|x),\quad c_i^t \sim C_i(\cdot|x, a_i^t),\quad a_i^{t+1} \sim A_i(\cdot|x, a_i^t, c_i^t)$$
其中 $i \in \{D, E, V\}$，$t = 0, \ldots, 3$，每队内不接收任何其他队的响应。

**平衡角色初始化（§3.3）**：
- 对每个 backbone，在 10,000 道 MMLU 辅助训练题上以温度 {0.4, 0.7, 1.0} 为每个角色生成候选。
- 仅保留提取答案正确且未被截断的目标，每道题-角色对最多保留一个目标，三个角色取 question ID 交集。
- Critic 从未经微调的基础模型初始化。

**续写价值偏好学习（§3.4）**：
- 对 Actor 状态 $(x, a)$，采样自然反馈 $c^0$、引导正确答案的 $c^+$、引导错误答案的 $c^-$。
- 用 $K=10$ 次 Actor 续写估计反馈质量（公式 2）：
$$\widehat{R}(c|x,a) = \frac{1}{K}\sum_{k=1}^{K} \mathbf{1}[g(a_k') = y],\quad a_k' \sim A_i(\cdot|x,a,c)$$
- 有序保留规则：若 $\widehat{R}(c^+) - \widehat{R}(c^0) \geq \epsilon = 0.6$，保留 $(c^+, c^0)$；否则若 $\widehat{R}(c^0) - \widehat{R}(c^-) \geq \epsilon$，保留 $(c^0, c^-)$；否则丢弃。
- 训练顺序：先构建并训练 Critic（Critic DPO），再用训练后的 Critic 构建并训练 Actor（Actor DPO）。
- DPO 损失（公式 3）：$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\log\sigma\left(\beta[\log\frac{\pi_\theta(u^+|s)}{\pi_{\text{ref}}(u^+|s)} - \log\frac{\pi_\theta(u^-|s)}{\pi_{\text{ref}}(u^-|s)}]\right) + \lambda\mathcal{L}_{\text{NLL}}$，其中 $\beta=0.1$，$\lambda=1$。

**无 Judge 后期融合（§3.5，公式 4）**：
$$\hat{y} = \begin{cases} m, & |\{i: z_i = m\}| \geq 2 \\ z_D, & \text{otherwise} \end{cases}$$
- 无学习型 Judge，平票时固定退回 Direct 答案。

## 实验与结果
**数据集**：MMLU（训练，14,042 题）、BoolQ（3,270 题）、BBH（1,260 题，22 类分层采样）、SciQ（1,000 题）、ARC Easy+Challenge（3,548 题）。后四个为 transfer 集，训练仅用 MMLU。

**模型基座**：Meta-Llama-3-8B-Instruct、Qwen2.5-3B-Instruct、Gemma-2-2B-it、Phi-4-mini-instruct、Mistral-7B-Instruct-v0.3。

**评估基线**：
- Direct（未微调模型单次生成）
- Debate（未训练的 Actor-Critic 五轮协议）
- SoM-2 / SoM-4（2/4 个对称无训练 agent，可互相观测）
- ACC（单对 Actor-Critic，使用相同 MMLU 源数据和超参）

**主要结果（Table 2）**：
- SEPAL 在全部 5 个 backbone 上均超越匹配的 ACC 单对，**宏观准确率平均提升 1.81 个百分点**，提升幅度 1.06–2.22 点。
- 25 个 model-dataset 单元格中 SEPAL 赢下 24 个，仅 Gemma-2-2B BoolQ 微降 0.15 点。
- Mistral-7B 获得最大增益（+2.22 点），Phi-4-mini 达到最高绝对宏观准确率（80.90%）。
- 在全部 4 个 transfer 集上均超过 ACC；MMLU 也在所有 backbone 上提升。

**决策诊断（§4.3）**：
- 投票准确率（76.31%）在 21/25 单元格中超过最强单角色，平均高出 0.62 点。
- 2/3 多数覆盖率平均 96.41%，Fallback 仅用 3.59%。
- Oracle-any-role 准确率为 86.12%，暴露 9.81 点的选择 gaps（为未来路由/验证器提供明确上限）。
-  pairwise agreement 约 78.7–79.2%，说明约 1/5 题目保留不同答案。

**消融（Table 3）**：
- Critic 反馈贡献最大：Full-R0 → Full-R4 提升 +3.72 点（23/25 单元格改善）。
- 无反馈的 SFT-only → Full-R0 仅提升 +0.26 点（17/25 单元格改善）。
- Base-C → Trained-C 平均 +0.57 点；Trained-C → Full-R4 仅 +0.01 点。
- 完整 pipeline（Full-R4）相对 SFT-only 提升 +3.97 点（24/25 单元格）。

**修订动态（§4.5）**：
- R1 捕获 89.2% 的最终增益（+3.32 点），R1→R4 净增仅 +0.40 点。
- R1 时错误→正确转换 6.50%，正确→错误倒退 3.19%；R4 时为 7.79% vs 4.07%。
- 不同数据集表现不同：BoolQ 从 R1 到 R4 增益持续扩大，BBH 则在 R1 后反而收窄。

## 相关工作脉络
1. **Self-consistency (Wang et al., 2023)**：独立采样多条推理路径后聚合；SEPAL 的相同点在于候选 histories 保持分离，不同点在于每候选有独立学习的 Critic 进行修正。
2. **ACC-Collab (Estornell et al., 2025)**：单对 Actor-Critic 通过续写正确率学习反馈；SEPAL 继承其偏好学习规则并独立复制给三个角色。
3. **Multi-agent Debate (Du et al., 2024; Liang et al., 2024)**：agent 间共享对话历史，可能传播错误；SEPAL 的关键差异在于反馈在团队内私有化，不在决策前跨团队共享。
4. **CoMM (Chen et al., 2024b)**：为 agent 分配不同角色 prompt；SEPAL 不仅用角色 prompt，还通过独立 SFT/DPO 训练不同 adapter，使角色具有实质性的行为差异。
5. **Tree of Thoughts (Yao et al., 2023)**：树搜索中保留多个部分解并用模型评估扩展；SEPAL 不剪枝，而是保留完整候选轨迹至投票接口。
6. **Multiagent Finetuning (Subramaniam et al., 2025)**：独立微调多个 agent 以保留推理链多样性；SEPAL 与之一致但额外引入 role-local Critic 修正和后期投票融合。
7. **LLM-Blender (Jiang et al., 2023b)**：学习配对 ranker + generative fuser 融合异构模型输出；SEPAL 用固定规则投票替代学习型选择层。

## 局限性与未来方向
1. **仅报告单 seed**：每个 model-method-dataset 单元格仅一次运行，重复训练 seed 的结果未报告，鲁棒性待验证。
2. **未做等计算量对比**：与 SoM/Debate 等基线比较时未统一 token 预算，公平性受限。
3. **仅三对结构**：三的奇数设计便于严格多数投票，但未探索更多角色对的边际收益。
4. **短答案 QA 任务限定**：当前仅评估选择题/是与否任务，复杂推理任务的泛化未知。
5. **固定 R4 修订轮次**：不同数据集的最优停止点不同，未来需基于 held-out 验证集设计早停策略。
6. **Oracle gap 9.81 点**：暴露路由/验证器的明确改进空间，提示未来可训练分类器选择最优候选。

## 研究启发与可借鉴点
1. **"局部反馈 + 晚期投票"的设计范式**可迁移至任何需要纠错但不希望错误传染的 multi-agent 场景，是共享讨论与 self-consistency 之间的有效折中。
2. **续写价值偏好学习（continuation-valued preference）**：用下游任务正确率而非人工标注来训练 Critic，是一种无需额外 reward model 的高效反馈学习方法，可复用于其他修正任务。
3. **角色 SFT 的"同一 question ID 交集"策略**：确保不同角色获得相同的训练题但不同视角的标注，是构造互补角色的实用技巧。
4. **R1 捕获 89.2% 增益的发现**提示在资源受限场景下可大幅减少修订轮数以换取性价比，未来值得设计动态早停机制。
5. **Oracle-any-role gap 作为可度量天花板**：该指标为未来路由器的研究提供了清晰的基准上限，可用于横向比较不同选择策略。

## 关键术语表
**SEPAL**：Separated Expert Pairs with Answer-Level Fusion，本文提出的三对隔离 Actor-Critic 团队+答案级投票融合框架。
**Actor-Critic Collaboration (ACC-Collab)**：以续写正确率为反馈质量指标的 Actor-Critic 成对学习方法，SEPAL 的核心训练组件。
**Continuation-valued preference**：通过对候选反馈进行 $K$ 次 Actor 续写并统计正确率来评估反馈质量的偏好构造方式。
**Majority coverage**：三个角色中至少两个达成一致的比例，衡量投票稳定性的指标。
**Oracle-any-role accuracy**：假设完美选择器能从三个角色中选到最优答案时的准确率，反映选择瓶颈的上限。
**Direct fallback**：当三角色无两票一致时，固定返回 Direct 角色答案的决策规则。
**SFT-only**：仅完成角色特定监督微调、未经 DPO 训练的 Actor，用于消融分析中衡量预训练增益。
**Regressed fraction**：从 R0 到后续修订轮次中由正确变为错误的样本比例，衡量过度修订的风险。

## 可复现要素
- **数据集**：MMLU（公开）、BoolQ（公开）、BBH（公开，使用 1,260 题分层采样子集）、SciQ（公开）、ARC（公开）。
- **代码/权重**：代码及完整结果已开源，地址 https://github.com/zhansan114514/SEPAL；包含 source code、test suite、50 个训练/评估配置、CSV 结果矩阵、125 个 per-example 决策文件、以及 2 个无需推理即可复算结果的脚本。
- **关键超参**：LoRA rank=256, scaling=512；DPO $\beta=0.1$, $\lambda=1.0$；SFT lr=$5.0\times10^{-5}$，DPO lr=$1.41\times10^{-5}$；margin $\epsilon=0.6$，续写采样数 $K=10$；温度 0.7，top-p 0.9；max new tokens 1,024；DPO/NLL total token limit 4,096；SFT epochs=1，DPO epochs=3。
- **硬件**：vLLM，bfloat16，最多 4 块 80GB NVIDIA A800 GPU。
- **Seed**：base seed=42，三角色分别偏移 0/10,000/20,000。
