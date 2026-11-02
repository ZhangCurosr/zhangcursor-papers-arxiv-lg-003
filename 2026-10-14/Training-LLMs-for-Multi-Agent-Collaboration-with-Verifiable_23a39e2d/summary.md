---
title: "Training-LLMs-for-Multi-Agent-Collaboration-with-Verifiable"
source: https://arxiv.org/pdf/2610.17893v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:46:02"
field: "多智能体LLM协作训练"
keywords: ["multi-agent collaboration", "reinforcement learning", "information reward", "self-play", "LLM post-training", "team surprisal"]
innovations: ["提出基于团队惊奇度减少的可计算信息奖励，区分共享/请求/使用三种行为", "构建贝叶斯评估器精确计算部分观测下的后验概率", "通过自博弈在简化环境中训练可迁移的协作行为"]
benchmarks: ["HiddenBench", "SILO-BENCH", "SplitQA", "MathChat", "BenchForm", "Einstein Puzzles", "Collaboration Tax"]
---

# 论文速读：Training-LLMs-for-Multi-Agent-Collaboration-with-Verifiable

## 一句话总结
本文提出 CoSIR 方法，通过在简化的评分表环境中使用可计算的信息奖励（基于团队惊奇度减少量）训练 LLM 多智能体学会共享、请求和使用信息，并在七个未见过的自然语言协作任务上验证了方法的迁移有效性。

## 研究问题与动机
- 多智能体协作中，最终任务结果无法提供个体消息价值的细粒度反馈，现有 RL 方法仅奖励正确答案导致训练信号稀疏。
- 自然语言任务中事实分布、贡献难以验证，无法直接计算信息价值；已有方法（如 MA-PoRL）侧重答案正确性和讨论修正，但未显式奖励"有用信息的共享/请求"。
- 推理时增强方法（ToM、TalkHier）无法改变模型本身的协作行为，需从训练层面培养共享、请求和整合证据的能力。

## 核心贡献（创新点）
- **可计算的信息奖励机制**：提出基于团队惊奇度减少量的共享和请求奖励，使信息价值可量化，区别于仅依赖最终结果的正统 RL 方法。
- **贝叶斯评估器设计**：构建基于部分观测表的精确后验概率计算框架，用动态规划避免组合爆炸，为奖励提供理论基础。
- **自博弈训练范式**：在评分表环境中通过自博弈将协作行为训练与通用能力保持结合，证明简单环境可培养可迁移的协作策略。
- **跨任务迁移验证**：在七个未见过的自然语言协作基准上证明 CoSIR 训练的模型在所有任务上均超越基线，且通用能力几乎无损。

## 方法详解
- **分布式信息任务环境**：生成包含 k 个选项、m 个属性的评分表 V ∈ {-1,0,+1}^{k×m}，每个 agent 仅获得部分单元格（briefing），最优选项 y* 为总分最高者。
- **贝叶斯评估器**：对 agent i 的信息记录 K_i = (B_i, R_i) 计算正确选项的后验概率 P(y*|K_i)，使用动态规划精确计算：P(y|K_i) = b_{iy} / Σ_o b_{io}，其中 b_{iy} 基于行和分布 F_{io} 计算。
- **团队惊奇度**：Φ(K) = -Σ_i log₂ P̄(y*|K_i)，P̄ 为下限为 10⁻⁴ 的概率。
- **共享奖励**：I_share(M) = ΔΦ(M) = Φ(K_before) - Φ(K_after)，对消息整体求值后用 Shapley 值分配给各陈述，仅支持性陈述可获得正奖励，伪造陈述受罚。
- **请求奖励**：I_request(q,c) = [ΔΦ(x_{q,c})]₊，假设 truthful 回答后计算惊喜度减少，乘以剩余轮次概率 p_next(q) 和折扣因子 δ_c，上限 2 比特。
- **私有答案奖励**：r_i^answer = P̄(ŷ_i|K_i) + (1/k)Σ_o 1[ŝ_io = s_i^K_o]，鼓励使用接收到的信息做选择和计算。
- **训练过程**：每轮 60 局游戏，16 个候选响应，使用 PPO 变体优化，附加 kl 惩罚和通用提示模仿损失保持基础能力。

## 实验与结果
- **模型与基线**：Qwen3-8B、Qwen3-4B、Gemma-3-4B-it；基线包括 Base、ToM、TalkHier、Outcome-RL（仅奖励最终结果）。
- **测试任务**：HiddenBench、SILO-BENCH、SplitQA（自建）、MathChat、BenchForm、Einstein Puzzles、Collaboration Tax。
- **主要结果**：
  - Qwen3-8B：CoSIR SUM=384.44，比第二的 TalkHier（364.65）高 19.79 分，比 Base 提升 31.85 分。
  - Qwen3-4B：CoSIR SUM=357.87，比 Outcome-RL（334.92）高 22.95 分。
  - Gemma-3-4B-it：CoSIR SUM=235.94，比 Outcome-RL（211.08）高 24.86 分。
  - CoSIR 在所有三个模型的七个任务上均超越 Base，而推理时方法在较小模型上 SUM 反而下降。
- **行为分析**：训练后私有事实披露率提升（Qwen3-8B 从 68.5%→74.3%），忽略队友输入失败率从 26.3% 降至 22.8%。
- **通用能力**：GSM8K/MMLU/IFEval 变化在 ±0.8 个百分点内，基本无损。

## 相关工作脉络
- **MA-PoRL**（Park et al., 2025）：奖励正确答案和讨论修正，但不区分单条消息的信息价值；CoSIR 显式奖励共享/请求行为。
- **MAGRPO**（Liu et al., 2026b）：将协作建模为合作 MARL，奖励最终结果；CoSIR 提供细粒度的信息级奖励。
- **SPIRAL/MARS**：从零和/合作游戏中学习推理能力并迁移；本文聚焦协作行为本身而非通用推理。
- **ToM/TalkHier**：推理时增强方法，不修改模型权重；CoSIR 通过 post-training 改变模型内在协作能力。
- **SPIRAL**：使用自博弈学习推理，但目标是零和游戏中的策略；本文将自博弈应用于协作信息共享。
- **Hipo**（Chen et al., 2026）：约束 RL 训练指令遵循；本文方法可扩展至开放域协作任务的训练信号设计。

## 局限性与未来方向
- 评估仅限三个骨干模型和七个任务，未覆盖更多架构和更复杂的真实场景。
- 训练未能消除所有协作失败类型（如重复步骤 85%+、信息保留 18.7%）。
- 奖励依赖已知事实的玩具环境，扩展到开放任务需解决"完整状态不可知时如何评估信息价值"的问题。
- 未来可将贝叶斯评估器与 LLM 推理结合，或探索在真实多智能体系统中的应用。

## 研究启发与可借鉴点
- **可计算奖励设计**：通过构造简化环境使信息价值可计算，为多智能体训练提供可复用的框架思路。
- **Shapley 值分配**：将消息整体价值分配给个体陈述，解决 credit assignment 问题，可迁移至其他多轮对话训练场景。
- **保留通用能力机制**：参考模型 KL 惩罚 + 通用提示模仿损失的组合，平衡专项训练与能力保持。
- **行为分析辅助评估**：不仅报告任务分数，还统计披露率、回应率、失败类型等，为协作训练提供多维评估。
- **自建基准 SplitQA**：用 HotpotQA 构建需要跨文档整合信息的任务，验证自由形式答案的迁移能力，值得借鉴。

## 关键术语表
- **CoSIR**：Collaboration via Surprisal-based Information Rewards，基于团队惊奇度减少的信息奖励协作训练方法。
- **团队惊奇度**：所有 agent 对正确选项概率的负对数之和，衡量团队整体不确定性。
- **贝叶斯评估器**：基于部分观测评分表计算各选项为最优的后验概率的精确推理器。
- **Self-play**：同一模型同时扮演所有 agent 进行自博弈训练。
- **Shapley 分配**：基于边际贡献的平均值将消息整体价值分配给各陈述。
- **MAST**：Multi-Agent System Failure Taxonomy，多智能体系统失败分类体系。
- **Outcome-RL**：仅奖励最终团队结果的对比强化学习方法。
- **Surprisal reduction**：信息价值度量，以比特为单位衡量惊喜度的减少量。

## 可复现要素
- **数据集**：自建评分表环境（五种配置家族），SplitQA 测试集 70 题；其他基准 HiddenBench、SILO-BENCH、MathChat 等公开。
- **代码/权重**：论文未明确声明开源，但提到附录提供详细实现。
- **关键超参**：采样温度 1.2，消息限制 1024 tokens，每轮 60 局游戏，16 候选响应，PPO clipping [0.8, 1.28]，双裁剪 bound 3，学习率 1e-5→2e-5，AdamW8bit，梯度范数限制 1.0，参考 KL 系数 0.1。
- **硬件**：2×NVIDIA H100 + 4×Intel Gaudi 2，每轮约 5 小时，共 3 轮约 15 小时。
