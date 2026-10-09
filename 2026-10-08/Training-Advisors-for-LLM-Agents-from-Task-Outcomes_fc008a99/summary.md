---
title: "Training-Advisors-for-LLM-Agents-from-Task-Outcomes"
source: https://arxiv.org/pdf/2610.09858v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:55:27"
field: "LLM Agent 训练与反馈机制"
keywords: ["LLM Agent", "Critic Training", "Reinforcement Learning", "Outcome-Based Reward", "Multi-Step Reasoning", "Tool Use", "Cross-Model Transfer"]
innovations: ["提出 Caddie：从任务最终结果训练 critic，无需步骤级标注或参考 critiques", "仅用 Qwen3-4B 训练的 critic 可跨模型迁移至 30B/27B/Kimi K3 并提升成功率", "在 MuSiQue 上训练的两个 critic 可零样本迁移至 DeepDive 和 τ³ 跨任务域"]
benchmarks: ["MuSiQue", "ALFWorld", "DeepDive", "tau3-bench"]
---

# 论文速读：Training-Advisors-for-LLM-Agents-from-Task-Outcomes

## 一句话总结
本文提出 **Caddie**，一种从任务最终结果出发训练 critic 的方法，使其能在 LLM agent 执行多步任务的过程中提供自然语言分析与建议；无需步骤级标注或参考 critiques，仅通过强化学习（DAPO）优化 critic，即可显著提升不同规模基础模型在域内及跨域任务上的成功率，且具备跨模型、跨任务的迁移能力。

## 研究问题与动机
1. **多步 agent 容易卡死或偏离目标**：LLM-based agent 在执行多步任务时，可能因重复无效动作或丢失原始目标而失败，需要外部反馈来引导修正。
2. **现有过程奖励模型（PRMs）依赖昂贵标注**：传统 PRMs 需要步骤级人工或自动标注，成本高且易引入噪声；步骤正确性判断在交互式环境中本身具有模糊性。
3. **自然语言反馈更具可解释性但训练困难**：critique-based 方法可提供直接指导下一步行动的自然语言反馈，减少搜索/分支开销，但如何有效训练此类 critic 仍不清晰。
4. **已有 critique 训练方法需要额外监督**：如 CTRL 需合成参考 critiques，Critique-RL 需两阶段训练（先评判再优化），流程复杂且依赖额外数据。

## 核心贡献（创新点）
1. **提出 Caddie 方法**：直接从 agent 轨迹的最终任务结果训练生成式 critic，无需步骤级标注或参考 critiques；与 CTRL/Critique-RL 等需要预训练监督或两阶段优化的方法本质不同，训练流程更简洁。
2. **展示跨模型迁移能力**：仅在 Qwen3-4B 上训练的一个 critic，无需微调即可显著提升 Qwen3-30B-A3B、Qwen3.8-27B、Kimi K3 等三个不同架构/规模基础模型的成功率；与 Advisor Models 仅报告跨学生迁移不同，本文进一步验证了跨任务域迁移。
3. **证明跨任务域迁移有效性**：在 MuSiQue 上训练的 critic 可直接用于 DeepDive 和 τ³ 等未见过的任务域并带来增益；揭示了 feedback 的有效性依赖于 generator 规模与任务类型（如小模型更需要定向搜索建议，大模型更需要停止搜索的建议）。
4. **揭示两种训练方式的互补性**：在 generator 已用 GRPO 训练至收敛后，再训练 critic 可进一步带来 13.88 个百分点的提升，说明 critic 反馈与 generator 直接优化是可叠加的正交信号。

## 方法详解
1. **Agent-环境交互框架**：冻结的基础策略 π_b 与环境 E 交互，状态 s_t 包含系统提示、任务、历史动作与观测；每步采样 a_t ~ π_b(·|s_t)，环境返回 o_t ~ E(·|s_t, a_t)，轨迹 τ 获得最终任务奖励 R(τ)。
2. **Critic 干预协议**：Critic π_c^θ 在状态 s_t 生成自然语言 critique c_t，追加到状态后形成 s̃_t = s_t ⊕ c_t，agent 继续执行；研究三种协议：(i) Windowed——在固定窗口内随机采样干预步；(ii) Probabilistic——每步以概率 p 独立干预；(iii) Self-call——base model 通过 call_critic 工具自主决定是否请求帮助（推理时使用此协议）。
3. **Outcome-based 奖励定义**：对每个采样状态前缀 s_k，critic 生成 G 条 critique，每条 appended 后运行 N 次独立 continuation，取平均最终奖励 R_i = (1/N) Σ_n R(τ^(i,n)) 作为该 critique 的评分，反馈价值完全由下游任务表现决定。
4. **DAPO 优化 critic**：采用 GRPO 变体 DAPO，计算组内归一化优势 Â_i = (R_i - μ_R)/σ_R，使用不对称 clipping 的 clipped surrogate objective；仅更新 critic 参数，base model 保持冻结；token 级 loss 聚合，动态采样丢弃零方差组并补充新组；格式违规的 critique 直接赋予 -1 奖励。

## 实验与结果
- **数据集与评估**：域内评估使用 MuSiQue（attached/wiki 两种检索设置）和 ALFWorld；跨域评估使用 DeepDive 和 τ³（Retail/Airline）；每条件生成 8 次独立 rollout，报告均值 ± SEM。
- **基线方法**：No critic、Prompted Qwen3-4B critic、Prompted Kimi K3 critic、Agent-RRM（生成式 reward model）。
- **主要结果**：
  - MuSiQue-attached：No critic 21.56% → Caddie **46.88%**（+25.31 pp）
  - MuSiQue-wiki：10.38% → **21.13%**（+10.75 pp）
  - ALFWorld：23.79% → **49.72%**（+25.93 pp）
  - Caddie 在全部三项均超越 Prompted Qwen3-4B 和 Prompted Kimi K3 基线。
- **跨模型迁移**：同一 Qwen3-4B critic 在 MuSiQue-attached 上提升 Qwen3-30B-A3B（28.44%→41.63%）、Qwen3.8-27B（46.63%→55.94%）、Kimi K3（38.56%→51.00%），后者甚至超越了 Kimi K3 自身无 critic 的基准。
- **跨任务迁移**：在 τ³ Retail 上 Qwen3.8-27B 从 50.31% 提升至 58.12%；Airline 从 74.38% 提升至 78.12%；DeepDive 从 48.83% 提升至 49.80%。
- **两阶段训练**：Qwen3-4B generator 经 GRPO 训练至 49.69% 后，再用其训练 critic 可进一步提升至 63.56%（+13.88 pp）。

## 相关工作脉络
1. **RL4F / CTRL**：通过监督初始化的 critiques 训练 critic（RL4F 用人工/程序生成反馈，CTRL 用代码执行合成 critiques）；Caddie 无需任何预训练监督，直接从最终 outcome 端到端优化。
2. **Critique-RL**：两阶段训练——先学正确性判断，再优化反馈质量；Caddie 跳过评判阶段，直接用任务结果作为唯一训练信号，流程更简洁。
3. **Advisor Models（Asawa et al., 2026）**：同样使用 GRPO 训练小 advisor，但 advice 在完整尝试前/后或固定步数（每 5 步）提供；Caddie 从采样中间状态分叉并比较 G 条 critique 的 continuation 结果，鼓励修正正在进行中的 episode，且 agent 自主调用。
5. **Welleck et al. (2023) / GLoRe / Retroformer**：训练 corrector 直接重写 generator 输出或在失败后反思；Caddie 在 agent 执行过程中提供下一步建议，而非事后修正。
6. **ECHO / PivoARL / ICRL**：联合训练 critic 与 agent 或交替训练；Caddie 保持 base model 完全冻结，仅更新 critic。

## 局限性与未来方向
1. **任务结构限制**：在自包含的单步任务（如纯数学推理）中，critic 可能学会直接生成完整解决方案让 base model 复制，丧失跨模型/跨任务迁移能力；本文聚焦多步工具使用任务。
2. **训练成本较高**：每个采样状态需要 N×G 次 base model continuation，对长轨迹、大模型或昂贵 tool call 场景开销显著；环境状态恢复也可能困难（如需 checkpoint 文件/进程）。
3. **Critic 规模未探索**：仅训练 Qwen3-4B  critic，更大 critic 的效果及训练稳定性尚未验证。
4. **与标量验证器的对比缺失**：未证明自然语言 feedback 优于 scalar verification（如 PRM/RRM 的打分选择），两者可在 matched inference 条件下公平比较。
5. **未来方向**：探索 base policy 与 critic 联合训练、fine-tuning base model 以更好利用 trained critic、结合无 critique baseline subtraction 或 counterfactual reward 等替代奖励定义。

## 研究启发与可借鉴点
1. **Outcome-based critic training 范式**：用任务最终结果作为唯一训练信号，避免昂贵步骤级标注，可迁移至其他需要 online feedback 的 agent 场景（如代码生成、embodied navigation）。
2. **Self-call 干预协议的设计**：将 call_critic 作为 base model 的可选工具，由 agent 自主决定何时求助，避免了固定步长/概率干预的任务敏感性问题，是一种优雅的推理时灵活控制机制。
3. **两阶段训练的互补性**：先训练 generator（GRPO）再训练 critic 可带来额外增益，说明 generator 优化与 critic 指导是正交信号；可探索交替训练或 joint training 策略。
4. **跨模型/跨任务迁移的实证价值**：小模型训练的 critic 可惠及大模型和不同任务域，提示 feedback 的质量可能比具体模型架构更通用；可进一步研究何种任务/配置下迁移性更强。
5. **Advice 类型分析框架**：通过拆解 advice 内容（搜索指导 vs. 停止回答 vs. 状态验证）并与 generator 规模关联，揭示了 "what works for whom" 的机制；此类分析可复用于其他 feedback 方法的可解释性研究。

## 关键术语表
**Caddie**：论文提出的方法名，指一种从任务最终结果训练 critic 的框架，使 critic 能在 agent 执行过程中提供自然语言分析与建议。
**DAPO**：DeepSeek 提出的开源 RL 训练系统，基于 GRPO 改进，支持不对称 clipping、token-level loss 聚合与动态采样，本文用于优化 critic。
**Self-call intervention**：一种干预协议，base model 通过 call_critic 工具自主决定是否请求 critic 帮助，而非由外部固定调度触发。
**Outcome reward model（ORM）**：仅对完整轨迹给出最终奖励的模型，区别于过程奖励模型（PRM）；本文用 ORM 信号训练 critic。
**Process reward model（PRM）**：对中间步骤进行正确性评估的模型，通常需要步骤级标注；本文避免使用此类标注。
**Group-normalized advantage**：将一组 critique 的奖励减去组内均值再除以标准差，得到归一化优势值，用于 GRPO/DAPO 中的策略梯度更新。
**Windowed intervention**：在预设的时间窗口 [k_min, k_max] 内均匀采样干预步，critic 仅在该步被调用。
**Agent-RRM**：一种生成式 reward model，通过 teacher 生成的 reasoning、critiques 和 trajectory scores 训练，用作本文跨任务迁移的基线之一。

## 可复现要素
- **数据集**：MuSiQue（Trivedi et al., 2022）、ALFWorld（Shridhar et al., 2021）、DeepDive（Lu et al., 2025）、τ³-bench（Sierra Research, 2026）；论文使用各数据集的公开训练/验证/测试 split。
- **代码/权重**：论文使用了开源模型 Qwen3-4B-Instruct-2507、Qwen3-30B-A3B-Instruct-2507、Qwen3.8-27B、Kimi K3；训练系统使用 DAPO（开源）；代码与 critic checkpoint 的开源状态论文未明确声明，需关注作者后续发布。
- **关键超参**：G=16（每组 critique 数），N=1（每条 critique 的 continuation 数），batch size=128，learning rate=10⁻⁶，clip range [0.20, 0.28]，context length=8192（域内）/32768（跨域），critic output cap=512 tokens，训练步数 MuSiQue=2000、ALFWorld=800；硬件为单节点 8×H100 80GB。
