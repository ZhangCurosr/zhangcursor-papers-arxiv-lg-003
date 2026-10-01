---
title: "RLTL-DR-Self-improvement-by-Internalizing-Self-generated-Fee"
source: https://arxiv.org/pdf/2609.37633v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:11:43"
field: "LLM 强化学习与自我改进"
keywords: ["RLVR", "GRPO", "self-distillation", "insight internalization", "learning barrier", "SFTL;DR"]
innovations: ["顺序自生成洞察打破 advantage collapse", "仅对 17 token 洞察做 SFT 损失即可内化任务→洞察映射", "简化到 SFTL;DR：仅 4k (任务,洞察) 元组训练恢复全性能"]
benchmarks: ["Appworld test-challenge", "Synthetic-API (SAPI)", "Leetcode frontier-difficult"]
---

# 论文速读：RLTL;DR: Self-improvement by Internalizing Self-generated Feedback

## 一句话总结
本文提出 RLTL;DR，一种针对极难任务（Pass@128=0）的自我改进强化学习方法：每次失败后让策略生成一句 TL;DR 洞察，并在后续尝试中将其注入上下文；同时通过 SFT 损失对洞察 token 反向传播，将"任务→洞察"映射内化到模型权重中。核心发现是仅训练 (任务, 洞察) 元组（SFTL;DR）即可恢复几乎全部性能。

## 研究问题与动机
- **学习信号枯竭**：标准 RLVR（如 GRPO）在 Pass@128=0 的任务上，所有 rollouts 均失败，优势函数退化为零，梯度消失（advantage collapse / learning cliff）。
- **无教师模型可蒸馏**：自我改进场景假设智能体处于能力前沿，没有更强 teacher 或黄金示例可用于 supervised distillation。
- **并行 i.i.d. 采样不够**：传统 GRPO 对每个任务并行采样 K 条轨迹，在极难任务上无法跳出零信号区。
- **测试时无法依赖上下文提示**：如果在训练时依赖 insight 才能解题，但评估时不注入 insight，性能会回落至基线水平。

## 核心贡献（创新点）
1. **顺序自生成洞察（Sequential Self-generated Insights）**：将并行 i.i.d. rollout 改为按失败次数顺序注入 prior insights，使策略在后续尝试中获得可学习的信号，Pass@k 从 ~6% 提升至 ~57%。
2. **洞察内化（Insight Internalization）**：引入 SFT 损失 ℒ_SFT 对 insight tokens（约 17 token/条）反向传播，以任务描述 g 为条件预测洞察，将"对这类任务要注意什么"映射内化到权重。
3. **简化到 SFTL;DR**：发现关闭 GRPO 损失、仅保留 ℒ_SFT 仍能恢复几乎全部性能；进一步去重为 4k 个 (g, f) 元组训练，仅用 68k backward tokens 即逼近完整 RLTL;DR。
4. **去混淆评估（Deconfounded Evaluation）**：明确区分含/不含 insight 的 rollout，报告仅在无 insight 条件下的 Pass@1，确保指标可比。

与已有工作的本质区别：RLTF-SD 对完整 rollout 做 self-distillation，而本文只预测 ~17 token 的高层抽象洞察；SGE 仅注入 prior attempts 的摘要而不反思失败原因；标准 GRPO 依赖 i.i.d. 并行采样无法突破学习瓶颈。

## 方法详解
### 3.1 背景：GRPO
对任务 g 并行采样 K 条轨迹 {τ_k}，优势 Â_k = r_k - Σr_i/K，GRPO 损失为：
ℒ_GRPO(θ) = E[ min(ρ·Â, clip_ε(ρ)·Â) ]，其中 ρ = π_θ(a_t|h_t)/π_θ_old(a_t|h_t)，ε=0.2。

### 3.2 顺序自生成洞察
- 第 1 次尝试正常生成 τ_1。
- 若验证器返回失败，让策略基于完整 rollout 及失败测试输出生成 JSON 格式的反思，提取最后一句 TL;DR 洞察 f_1。
- 当当前批次成功率 ≤ 50% 时，将历史洞察 {f_i} 作为附加 user message 插入下一轮 rollout 的上下文：h̃_t = (g, f_1, ..., f_I, ...)。
- 最多保留 16 条洞察；K 次尝试视为一个 GRPO group。

### 3.3 洞察内化损失
定义 SFT 损失：
ℒ_SFT(θ) = -log π_θ(f_1,...,f_I | g)
总损失：ℒ = ℒ_GRPO + λ·ℒ_SFT，λ=0.5 为默认值。

实现技巧：由于洞察 token 已出现在上下文 h̃_3 中，ℒ_SFT 无需额外 forward，只需修改 backprop mask 翻转入度即可。

### 辅助技巧
- 去除优势函数的标准差除法（借鉴 Dr. GRPO）。
- Positive-ratio filtering：确保 75% rollout 有正奖励，稳定训练。

## 实验与结果
### 数据集（均过滤至 Pass@128=0）
- **SAPI**（Proprietary）：458 训练 + 184 较难任务 + 1624 eval
- **Appworld**：34 train/dev/test-normal + 417 test-challenge
- **Leetcode**：123 frontier-difficult

### 基线
- **GRPO**：Qwen 3.5 9B Thinking，全部任务 Pass@1 稳定在 0–1%
- **SGE**（Strategy-guided Exploration, Szot et al. 2026）：注入 prior attempts 摘要，但无反思，同样无法学习
- **RLTF-SD**（Song et al. 2026）：对 rollout 做 self-distillation

### 主要结果（Table 1，测试集 Pass@1，无 insight 条件）
| 方法 | SAPI | Appworld | Leetcode |
|------|------|----------|----------|
| GRPO | 57.8% | 33.7% | 55.1% |
| SGE | 57.7% | 33.5% | 56.4% |
| RLTF-SD | 53.0% | 38.9% | 42.2% |
| **RLTL;DR** | **91.1%** | **61.7%** | **49.1%** |

### 消融（Table 2，SAPI frontier split）
- RLTL;DR (λ=0.5)：Train 21.5%，Eval 18.9%
- RLTL;DR (λ=0)：Train 6.1%，Eval 13.2% → 证明 ℒ_SFT 是关键
- 仅 ℒ_SFT（关闭 GRPO）：Train 20.0%，Eval 17.0% → 几乎恢复全性能
- SFTL;DR (deduplicated, 4k tuples)：Train 19.1%，Eval 16.9%，仅 68k backward tokens

### 洞察质量（Table 3）
- 去掉 failed unit tests 信息：Pass@1 跌至 3.4%（最关键）
- 详细诊断段落反而降低 generalize 能力（过度具体含 session-specific 细节）
- GLM 5.2 teacher 生成洞察有 modest 提升（Eval 20.2% vs 19.2%）

## 相关工作脉络
1. **RLVR / GRPO**（Lambert et al. 2024; Shao et al. 2024; Yu et al. 2026）：并行 i.i.d. 采样，优势 collapse 在 frontier tasks 上是结构性失效。
2. **RLTF-SD**（Song et al. 2026）：对 rollout 做 self-distillation，本文对比并指出只蒸馏完整 rollout 不如蒸馏高层洞察。
3. **SGE**（Szot et al. 2026）：注入 prior attempts 摘要引导探索，但缺乏对失败原因的反思，无法突破学习壁垒。
4. **Context Distillation**（Askell et al. 2021; Snell et al. 2022）：训练模型模拟有上下文的行为；本文是其"极简版"——只内化 17 token 洞察。
5. **Advantage Collapse / Learning Cliff**（Xia et al. 2026; Agrawal et al. 2026; Agashe et al. 2026）：全正/全负奖励导致梯度消失，本文通过顺序 insight 打破此困境。
6. **ECHO**（Shrivastava et al. 2026）、**Lu et al. 2026a**：类似发现——对 user-message 中的 token 反向传播可内化知识；本文扩展至"任务→抽象洞察"格式。

## 局限性与未来方向
- **领域适用性**：对高度特异性任务（如数学证明需特定技巧）可能无效；Leetcode 结果不佳暗示代码生成中洞察泛化受限。
- **预训练能力门槛**：内化依赖模型参数空间的"语义平滑性"，需要足够大的预训练基础。
- **洞察生成难度**：在某些领域（如自动科研），生成好洞察可能需要与生成解同等能力，此时顺序采样策略可能失效。
- **过拟合风险**：训练曲线显示 RLTL;DR 和 RLTF-SD 均存在一定过拟合迹象，尤其在 Leetcode 上。
- **评估公平性**：SAPI 训练集包含部分 Appworld test-normal 任务，不应引用为严格 benchmark score。

## 研究启发与可借鉴点
1. **极简自蒸馏信号**：仅对 ~17 token 的高层抽象反馈做 SFT 损失即可大幅提升难任务性能，无需蒸馏完整 rollout——可作为通用自改进范式。
2. **顺序采样打破 advantage collapse**：将 i.i.d. 并行尝试改为条件顺序尝试，配合 50% 成功率启发式注入洞察，是突破稀疏奖励的有效策略。
3. **去混淆评估设计**：明确区分"含/不含 insight 的 rollout"并分别报告 macro-averaged Pass@1，避免上下文带来的指标虚高。
4. **元组去重训练**：SFTL;DR 证明 4k 个 (任务, 洞察) 唯一元组即足以恢复性能，提示未来可在更小规模数据上验证该类方法。
5. **正比例过滤（Positive-ratio filtering）**：确保 75% rollout 有正奖励，可有效防止 entropy explosion 导致 policy collapse。

## 关键术语表
- **RLVR**（Reinforcement Learning with Verifiable Rewards）：通过可验证奖励（如单元测试通过）进行强化学习，成功 reward=1、失败 reward=-1。
- **GRPO**（Group Relative Policy Optimization）：组内相对策略优化，优势函数 Â_k = r_k - mean(r)，无 KL 正则。
- **Advantage Collapse / Learning Cliff**：当一组 rollout 全部成功或全部失败时优势退化为零，梯度消失。
- **TL;DR Insight**：对失败尝试的高层抽象反思，通常一句话（~17 token），如"Remember to paginate search results"。
- **SFTL;DR**：仅训练 (任务, 洞察) 元组的 SFT 版本，关闭 GRPO 损失和 rollout 反向传播。
- **Deconfounded Evaluation**：报告不含 insight 上下文时的 Pass@1，以公平比较不同方法的真实泛化能力。
- **Positive-ratio Filtering**：训练时过滤掉全负奖励的 batch，保留至少 75% 正奖励 rollout 以稳定训练。
- **Insight Reliance**（Xia et al. 2026）：衡量成功 trajectory 对 insight 的依赖程度，本文测得从 +0.009 升至 +0.034，属低依赖。

## 可复现要素
- **数据集**：Appworld（开源）、Leetcode Dataset（开源）、SAPI（proprietary，未公开）；筛选标准为 Qwen 3.5 9B Thinking Pass@128=0。
- **代码/权重**：论文未提供开源代码或模型权重；提供了 Appendix B 的详细复现步骤（prompt、mask 翻转方法）。
- **关键超参**：λ=0.5（SFT 损失权重）、ε=0.2（PPO clip）、lr=3×10⁻⁶（Appworld/SAPI）/ 6×10⁻⁶（Leetcode）、minibatch=4/1、gradient accumulation=8/4、8×B200 GPU。
- **模型**：Qwen 3.5 9B Thinking（Qwen Team 2026）。
- **训练规模**：SAPI frontier split ≈ 170k environment interactions；Appworld ≈ 13k steps per epoch。

---
