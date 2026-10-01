---
title: "RLTL-DR-Self-improvement-by-Internalizing-Self-generated-Fee"
source: https://arxiv.org/pdf/2609.37633v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:11:46"
field: "LLM Agent 自我改进与强化学习"
keywords: ["RLVR", "GRPO", "self-improvement", "insight internalization", "SFTL", "context distillation", "agent reinforcement learning"]
innovations: ["将失败后的 TL;DR 洞察作为顺序 rollout 上下文并内化 task→洞察映射", "揭示仅靠 (task, insight) 元组的 SFT 即可恢复大部分性能", "系统消融洞察格式/生成器/数量对泛化的影响"]
benchmarks: ["Synthetic-API (SAPI)", "Appworld", "LeetCode"]
---

# 论文速读：RLTL;DR: Self-improvement by Internalizing Self-generated Feedback

## 一句话总结
本文提出 RLTL;DR，一种针对极难任务（初始 Pass@128=0）的自改进强化学习方法：在每次失败后让策略生成一条"TL;DR"洞察（一句话总结），将洞察反馈到下一轮尝试，并通过 SFT 损失将"任务→洞察"映射内化到模型参数中，从而突破标准 GRPO 在此类任务上无法学习的瓶颈。

## 研究问题与动机
- **学习信号枯竭**：在自改进场景下，任务极难，策略在 128 次尝试中一次都未成功，GRPO 因所有 rollout 获得相同负奖励而陷入"优势坍缩"，梯度消失。
- **缺乏教师/示例**：自改进设定假设策略处于能力前沿，不存在更强教师模型或黄金示例可供蒸馏。
- **已有引导方法不足**：现有注入先验信息的方法依赖外部强教师或参考答案，不适用于无外部反馈的自改进。
- **测试时无法利用上下文**：训练时借助洞察可提升性能，但测试时要求单次尝试（Pass@1），需将知识从上下文迁移到权重。

## 核心贡献（创新点）
1. **提出 RLTL;DR 框架**：在失败后将验证器输出和尝试历史交还给策略，让其生成一条 TL;DR 洞察，并在下一轮以顺序 rollout 方式注入上下文，从而打破学习信号枯竭。
2. **引入洞察内化目标（SFT loss on insights）**：以任务描述为输入、让模型直接预测已生成的洞察，仅通过 backpropagation mask 翻转实现，无需额外前向计算，将"任务→洞察"映射从上下文迁移到权重。
3. **揭示并简化至 SFTL;DR**：发现纯 SFT 损失作用于洞察 token 即可恢复 RLTL;DR 几乎全部性能；进一步去重后仅需 4k 个 (task, insight) 元组即可复现，证明"on this sort of task, keep this sort of thing in mind"是真正关键的训练范式。
4. **系统性消融洞察设计**：证明简洁的 TL;DR 洞察（平均 17 词）优于更详细的诊断段落/修正代码；最关键的是洞察生成器必须能看到失败的单元测试，否则完全失效。

## 方法详解
- **任务建模**：将 agent 交互形式化为 POMDP，策略 π_θ 生成动作 a_t（含 think trace 与代码块），经验轨迹 τ = (h_T, a_T, o_T, r)，最终由代码验证器给出二元奖励 r。
- **顺序洞察条件采样**：对同一任务依次采样 K 次尝试；若当前批次成功率 ≤ 50%，则将此前 I 条洞察 {f_i} 作为额外 user message 插入上下文；每条洞察约 17 token。
- **洞察生成**：每次失败后，将完整 rollout 历史、环境观测（截断）与验证器报错送入 judge prompt，让策略先思考（4096 token think），再生成 JSON 格式的 summary/feedback/wrong_step_id/corrected_step/TL;DR hint，仅取最终单句 hint 作为洞察。
- **双损失联合优化**：总损失 L = L_GRPO + λ·L_SFT，其中 L_GRPO 为标准 GRPO 策略梯度（ clip ε=0.2，去除了除以标准差）；L_SFT 为对洞察 token 的 next-token 监督损失，仅通过翻转 backpropagation mask 在第三轮尝试的上下文中实现，不增加运行时开销。默认 λ=0.5。
- **正比例过滤（positive-ratio filtering）**：为保证稳定性，在计算 advantage 后过滤 rollout 使 75% 具有正奖励，避免熵爆炸与策略崩溃。
- **SFTL;DR 简化**：完全去掉 GRPO，仅对 (g, f_1, …, f_I) 元组做 SFT；去重后构建 4k 唯一 (task, insight) 对，每对训练一次。

## 实验与结果
- **数据集与过滤**：Appworld（90 train/57 dev/168 test-normal/417 test-challenge）、私有 Synthetic-API（16k train，过滤后 458 frontier 任务+184 hard）、LeetCode（2641，过滤后 123）。均只保留 Qwen 3.5 9B Thinking 在 128 次尝试中零成功的任务（Pass@128=0）。
- **基线**：标准 GRPO、RLTF-SD（自蒸馏）、SGE（策略引导探索）。
- **主要结果（Pass@1，无洞察上下文评估）**：

| 方法 | SAPI | Appworld | Leetcode |
|---|---|---|---|
| GRPO | 57.8% | 33.7% | 55.1% |
| SGE | 57.7% | 33.5% | 56.4% |
| RLTF-SD | 53.0% | 38.9% | 42.2% |
| **RLTL;DR** | **91.1%** | **61.7%** | 49.1% |

- GRPO/SGE 在 frontier 集上停滞于 0–1%；RLTL;DR 达到 12–13% Pass@1（不含洞察），含洞察时训练期 Pass@1 达 14–31%、Pass@k 达 14–59%。
- 消融表明 λ 从 0.5 降至 0.01 或 0 会严重损害性能；去掉 GRPO 仅用 L_SFT 仍可恢复 17.0% eval Pass@1（RLTL;DR 为 18.9%）。
- 仅对 4k 去重 (task, insight) 元组做 SFTL;DR-deduplicated，eval Pass@1 达 16.9%，训练仅 68k backward tokens（对比 RLTL;DR 的 12M）。
- 详细诊断（含完整错误信息/修正代码）不如 TL;DR 通用；洞察生成器是否具备 thinking 影响不大，但**能否访问失败的单元测试至关重要**（关闭后 Pass@1 降至 3.4%）。

## 相关工作脉络
- **RLVR/GRPO 的局限**：Lambert et al. (2024)、Shao et al. (2024)、Yu et al. (2026)、Liu et al. (2025b)；本文指出当 K 个 rollout 同获零/全奖励时 advantage 坍缩，属于该现象的 frontier 变体。
- **利用特权信息引导探索**：Zhang et al. (2025, 2026b)、Agrawal et al. (2026) 引入强教师或参考答案；本文强调**无任何外部指导**，仅依赖验证器输出的失败信息自生成洞察。
- **Self-distillation/上下文蒸馏**：Askell et al. (2021)、Snell et al. (2022)、Wang et al. (2026)、Hübotter et al. (2026, RLTF-SD)、Song et al. (2026, RLTF)；区别在于本文只预测约 17 token 洞察而非整个 rollout，且训练目标是 task→洞察映射。
- **Context bootstrapping**：Agashe et al. (2026)、Xia et al. (2026)；本文与之共性问题相关，但通过洞察内化转移到了无需检索的端到端训练。
- **平滑性与跨格式泛化**：Nakkiran et al. (2026)、Shrivastava et al. (2026, ECHO)、Lu et al. (2026a)、Cook et al. (2026)；本文现象与之呼应，即反向传播将洞察知识路由到语义相似的 task→rollout 生成。

## 局限性与未来方向
- **过度具体的任务难以迁移**：若解决方案依赖极其特异的技巧（如某道数学证明的唯一技巧），则洞察难以泛化；LeetCode 结果弱于 Appworld/SAPI 即印证此点。
- **依赖预训练平滑性**：(task, insight) 训练可能是涌现能力，需要足够强的预训练平滑性；对弱基座模型可能失效。
- **洞察生成门槛**：在科学发现等前沿领域，生成高质量洞察本身可能几乎与生成解决方案同等困难，即便内化有效，前端也可能卡住。
- **测试仅报告 Pass@128=0 frontier 任务上的提升**：在普通难度集上 GRPO 已能学习，RLTL;DR 未见显著增益（除 LeetCode 略有下降），适用范围仍需扩大验证。
- **计算代价**：顺序采样使采样阶段比并行 GRPO 慢约 4.5×（K=8 时），整体 walltime 增加约 1.5×。

## 研究启发与可借鉴点
1. **"任务→洞察"内化是一种高效的知识压缩训练范式**：本团队可在自身 agent/代码生成任务上复用——收集失败轨迹、提取单句洞察、以 L_SFT 内化，无需完整 rollout 反向传播即可获得显著提升。
2. **洞察必须保持抽象与可迁移**：避免将 session-specific 细节（如 object_id、access_token）混入洞察；可通过限制洞察生成 prompt 来过滤。
3. ** privilieged information（失败单元测试输出）是关键信号**：若团队场景缺少细粒度验证器输出，可考虑模拟/构造近似反馈。
4. **正比例过滤与去标准差 GRPO 改进**：团队可借鉴其稳定 GRPO 的训练技巧（75% 正比例过滤、去掉标准差归一化）以改善自身极难任务的训练曲线。
5. **SFTL;DR 可作为独立训练方案**：在算力受限场景下，仅维护一个 (task, insight) 数据库并用 SFT 训练，比完整 RL 循环更轻量，值得探索。

## 关键术语表
- **RLTL;DR**：本文提出的方法，通过顺序注入自我生成的 TL;DR 洞察并内化 task→洞察映射来突破 RLVR 学习屏障。
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：以可验证奖励（如单元测试通过率）驱动 LLM agent 强化学习的范式。
- **GRPO（Group Relative Policy Optimization）**：Shao et al. (2024) 提出的组相对策略优化，以组内平均奖励为基准计算 advantage。
- **Advantage collapse / 学习悬崖**：当组内所有 rollout 同获零或全奖励时，advantage 退化为零，梯度消失，训练停滞。
- **Insight internalization**：将洞察 token 的 SFT 损失通过翻转 backpropagation mask 施加到模型参数上，使"任务→洞察"映射被内化。
- **SFTL;DR**：RLTL;DR 的极简版本，完全去掉 GRPO，仅对 (task, insight) 元组做 SFT。
- **Deconfounded evaluation**：训练期间始终只在无洞察上下文的 rollout 上计算指标，以公平比较方法真实能力。
- **Positive-ratio filtering**：过滤 rollout 保证 75% 具有正奖励，防止极难任务下熵爆炸与策略崩溃。

## 可复现要素
- **数据集**：Appworld（公开，ACL 2024）、LeetCode dataset（公开，Xia et al. 2025）；Synthetic-API（SAPI）为 Apple 私有数据，**未公开**。
- **代码/权重**：论文**未提供**开源代码与模型权重；附录 B 给出了三步实现细节（judge prompt、洞察插入格式、mask 翻转），可据此复现。
- **关键超参**：λ=0.5、KL 无正则、clip ε=0.2、正比例过滤 75%、洞察生成 think budget 4096 token、洞察平均 17 token、K=8 顺序尝试、成功率阈值 50% 触发洞察注入、学习率 3e-6（Appworld/SAPI）或 6e-6（LeetCode）、minibatch=4 或 1、梯度累积 8 或 4、8×B200 GPU。
