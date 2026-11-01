---
title: "TEV-SCALING-LONG-HORIZON-AGENTIC-ENVIRONMENTS-WITH-DENSE-PRO"
source: https://arxiv.org/pdf/2610.18868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:07:42"
field: "长视界智能体强化学习"
keywords: ["long-horizon agent", "reinforcement learning", "reward shaping", "synthetic data", "terminal agent", "progress potential", "group-relative policy optimization"]
innovations: ["提出潜世界优先的合成方法确保任务-验证器-答案严格对齐", "构建TEV-GYM首个带密集渐进奖励的长视界训练数据集", "提出PPRS渐进势奖励塑形缓解GRPO稀疏奖励困境"]
benchmarks: ["ALE-CM", "Terminal-Bench 3.0-CM", "Terminal-Bench 4.0-CM", "Terminal-Bench 2.1"]
---

# 论文速读：TEV: SCALING LONG-HORIZON AGENTIC ENVIRONMENTS WITH DENSE PROGRESS REWARDS

## 一句话总结
本文提出了一种"潜世界优先"的数据合成方法与渐进势奖励塑形（PPRS）技术，构建了首个具有密集渐进奖励的长视界智能体训练数据集 TEV-GYM，使开源模型在长期复杂任务（如 ALE-CM、Terminal-Bench）上的表现大幅超越基线模型，逼近甚至匹敌前沿闭源模型。

## 研究问题与动机
- **长视界任务的训练数据稀缺**：现有开源智能体训练语料（如 SWE-Smith、TMAX、SETA）多为短视界任务（平均耗时 1-2.4 分钟），而真正长视界任务仅存在于人工构建的评测集中（如 ALE、Terminal-Bench），数量稀少且无法直接用于训练。
- **长视界任务奖励稀疏**：传统终端仅依赖最终提交结果给出 pass/fail 信号，导致 GRPO 等组相对策略优化方法中，全失败组的梯度为零，严重浪费训练资源。
- **任务合成与验证的困难**：长任务需要足够复杂以提供有意义训练信号，同时又需精确可验证；人工合成难以规模化，而纯 LLM 生成的任务存在"工件漂移"（artifact drift）问题。
- **开放模型能力差距大**：当前开源模型在 Terminal-Bench 4.0 等基准上的 Pass@1 仅为 4-9%，与 GPT-6 Astra、Claude Opus 5 等前沿模型差距显著。

## 核心贡献（创新点）
1. **提出"潜世界优先"的任务合成方法**：首先构建一个完全正确的潜在世界（latent world），再自顶向下渲染为任务输入和验证器，确保任务、验证器与正确答案严格对齐，消除工件漂移。
2. **构建 TEV-GYM 数据集**：首个具有可验证、密集渐进奖励的长视界智能体训练数据集，包含 1,170 个跨越 7 个领域的任务环境，任务平均耗时 14.8 分钟，约为之前数据集的 6 倍。
3. **提出 PPRS（Progress-Potential Reward Shaping）**：将潜在世界的逐步进展量化为连续奖励信号，使得即使所有 rollout 均未完成，组内仍存在梯度差异，有效缓解组相对优化的稀疏奖励困境。
4. **实现 TEV-27B 与 TEV-320B 的训练**：TEV-27B 在 ALE-CM 上 Pass@5 达到 0.28，匹配 GPT-6 Astra 与 Claude Opus 5；在 Terminal-Bench 4.0-CM 上 Pass@1 达 0.10，超越 GLM 5.2 与 Gemini 3.7 Flash。

## 方法详解
**1. 潜世界优先合成流程（Latent-World-First Synthesis）**
- **Stage 0**：从 Agents' Last Exam (ALE-CM) 的 7 个计算与数学领域随机采样任务规范。
- **Stage 1**：使用种子化程序生成完整的潜在世界 W，并验证其不变量约束。
- **Stage 2**：将任务工作流编译为有向无环图（DAG），定义任务视界长度与步骤依赖。
- **Stage 3**：从 W 渲染公共输入文件（CSV、SQLite、safetensors 等格式），确保渲染可逆。
- **Stage 4**：按预设缺陷计划对文件进行可控损坏（schema drift、重复记录、缓存过期等），生成智能体面对的初始工作空间 s₀。
- **Stage 5-6**：生成公开任务描述与隐藏验证套件（包括变换不变性检查与反事实失效检查）。

**2. 渐进势函数 Φ_W 的定义**
每个任务提供一套进度节点 V，每个节点 v 为工作空间的可直接检查状态，权重 w_v > 0 且 Σw_v = 1。部分评分函数 c_v(s) ∈ [0,1] 将当前工作空间 s 与潜世界对比，信用需满足前置条件：
$$\tilde{c}_v(s) = \min_{u \in \text{pre}(v) \cup \{v\}} c_u(s), \quad \Phi(s) = \sum_{v \in V} w_v \tilde{c}_v(s) \in [0,1]$$

**3. PPRS 奖励塑形公式**
每步奖励为势函数变化量 r_t = Φ(s_t) - Φ(s_{t-1})，rollout 总奖励：
$$R(\tau) = \mathbb{1}[\tau \text{ passes}] + \lambda \sum_{t=1}^{T} r_t = \mathbb{1}[\tau \text{ passes}] + \lambda \Phi(s_T)$$
其中 λ = 0.25。通过每步归一化优势（turn-index normalization）与 γ=1 的 return-to-go，使得即使全部失败组也能获得非零梯度。

**4. 资格门控机制**
每个环境必须通过 8 项独立于基准的资格门：奖励因果性、反作弊性（14 种游戏策略均得分 0）、角色独立性、求解器一致性、渐进有效性、变换不变性、六种子多样性、公共契约卫生。

## 实验与结果
**数据集与评估基准**
- TEV-GYM：1,170 个环境，7 个领域（软件工程、数据工程、AI/CS 研究、数学与运筹学、安全与取证、基础设施管理、量子计算）
- ALE-CM：18 个专家编写任务（计算与数学科学子集）
- Terminal-Bench：TB 3.0-CM（22 个任务）、TB 4.0-CM（18 个任务）、TB 2.1（89 个中视界任务）

**主要结果**
| 模型 | ALE-CM P@5 | TB 3.0-CM P@1 | TB 4.0-CM P@1 | TB 2.1 P@1 |
|------|-----------|--------------|--------------|-----------|
| TEV-27B | 0.28 | 0.14 (+56%) | 0.10 (+150%) | 0.794 |
| TEV-9B | 0.11 | — | — | 0.227 (+28%) |
| Qwen3.8-27B (base) | 0.17 | 0.09 | 0.04 | 0.764 |
| GPT-6 Astra | 0.28 | 0.41 | 0.46 | — |
| Claude Opus 5 | 0.28 | 0.53 | 0.52 | — |
| GLM 5.2 | — | 0.04 | 0.06 | — |
| Gemini 3.7 Flash | — | 0.07 | 0.09 | — |

**关键提升**
- TEV-27B 在 ALE-CM 上 P@5 达 0.28，与 GPT-6 Astra 和 Claude Opus 5 持平
- 在 TB 4.0-CM 上 Pass@1 达 0.10，超过 GLM 5.2 和 Gemini 3.7 Flash
- 零样本迁移至 TB 2.1，Pass@1 达 0.794，匹配 Claude Opus 4.8 水平
- 扩展至 GLM-5.3-Flash 得到 TEV-320B，TB 4.0-CM Pass@1 达 0.37

**消融实验**
- 去掉潜世界改为 LLM 直接生成：全失败组比例从 0% 升至 88%
- 仅用终端奖励（无 PPRS）：27B 模型几乎无法完成优化器一步
- PPRS 每轮归一化优于逐 rollout 归一化

## 相关工作脉络
1. **代码/终端智能体训练语料**：SWE-Smith、TMAX、SETA、Endless Terminals 等使用 LLM 生成或挖取真实仓库，但任务尺度小（平均 <3 分钟），且仅提供终端 pass/fail 信号；TEV-GYM 首次将任务时长扩展至 ~15 分钟并提供密集奖励。
2. **反向合成方法**：Denoising、bug 注入、指令后翻译等单对象逆向方法；TEV 首次将整个潜世界的多格式渲染作为统一锚点，避免工件漂移。
3. **组相对策略优化中的稀疏奖励问题**：DAPO、TMAX 等丢弃全失败组或重采样；PPRS 从根本上解决该问题，使全失败组仍能产生梯度。
4. **过程奖励模型**：Process Reward Model 依赖人类标注或训练裁判模型；PPRS 由潜世界直接计算，无需额外模型。
5. **长视界基准**：ALE、Terminal-Bench、TheAgentCompany 等为人工评测集；本文首次将其转化为可规模化的训练资源。

## 局限性与未来方向
- **领域覆盖有限**：当前 TEV-GYM 仅覆盖 ALE-CM 的 7 个领域，可推广至能源工程、金融等其他 ALE 领域
- **与前沿模型仍有差距**：在 TB 3.0/4.0 上 TEV-27B 的 P@1 仅 0.10-0.14，与 Claude Opus 5（0.52-0.53）差距较大，需要更强的基础模型
- **合成成本高**：每个环境需 ~30 秒 CPU 时间，且需人工审查生成器
- **仅支持 Linux Docker 容器**：排除需要 Windows VM 或完整虚拟机的任务（如二进制分析）
- **未来方向**：简化新工作流支持、扩展领域覆盖、与更强开源模型结合

## 研究启发与可借鉴点
1. **"潜世界优先"设计范式**：可迁移至其他长任务领域（如科学发现、自动化研究），通过先构建正确世界再渲染任务，确保验证一致性。
2. **渐进势奖励塑形**：PPRS 的数学形式（基于工作空间状态的单调势函数）可推广至任何具有可验证中间状态的多阶段任务，不仅限于终端代理。
3. **资格门控机制**：8 项自包含门控（反作弊、角色独立、变换不变性等）可作为合成数据集质量保障的标准协议。
4. **覆盖数组控制多样性**：strength-2 covering array 方法可确保任务在多个轴向上的平衡，避免模板锁定。
5. **组内归一化改进**：per-turn-index group normalization 比 per-rollout normalization 更有效，值得在其他 RL 训练中尝试。

## 关键术语表
**Latent World（潜世界）**：任务领域的完整正确状态表示，包含所有输入数据、中间状态和正确答案，作为任务生成的唯一锚点。
**Progress Potential Φ_W**：基于潜世界的渐进势函数，衡量智能体当前工作空间相对于正确状态的完成程度，取值 [0,1]。
**PPRS（Progress-Potential Reward Shaping）**：渐进势奖励塑形，将每步势函数变化量作为密集奖励信号，与终端奖励结合训练。
**GRPO（Group-Relative Policy Optimization）**：组相对策略优化，通过对同一任务的多条 rollout 计算组内归一化优势来更新策略。
**Artifact Drift（工件漂移）**：任务、验证器和输入文件分别生成时出现的不一致问题，导致验证失效。
**Strength-2 Covering Array**：保证任意两个维度上的所有合法取值对至少在一个任务中出现的多样性控制方法。
**Turn-index Normalization**：按步数索引在组内归一化优势，保留后期步骤的信用而非与整个 rollout 均值比较。
**Return-to-go**：从当前步到 rollout 结束的未来累积奖励，γ=1 时完整传播渐进进展的信用。

## 可复现要素
- **数据集**：TEV-GYM，生成程序为确定性种子函数，可精确复现；论文未提及公开下载链接，但提供了附录中的详细构造说明
- **代码**：论文未明确提及代码开源状态，但声明所有结果可从论文复现
- **模型权重**：TEV-27B、TEV-9B 及 TEV-320B 未提及开源
- **关键超参**：G=24 rollouts/task，8 tasks/step，响应预算 65,536 tokens，每步最多 64 次工具调用，λ=0.25，学习率 1e-6，total variation bound 0.1，no KL anchor
