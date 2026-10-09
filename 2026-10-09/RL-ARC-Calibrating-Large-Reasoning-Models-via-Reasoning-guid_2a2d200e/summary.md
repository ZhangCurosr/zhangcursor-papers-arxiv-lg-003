---
title: "RL-ARC-Calibrating-Large-Reasoning-Models-via-Reasoning-guid"
source: https://arxiv.org/pdf/2610.11352v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:33:26"
field: "大模型可靠性与校准"
keywords: ["大语言模型校准", "推理置信度", "强化学习", "可验证奖励", "分布外泛化", "过度自信缓解"]
innovations: ["首次联合建模推理置信度与答案置信度进行双阶段校准训练", "设计条件依赖的差异化优化策略（正则化/惩罚）缓解准确性-校准权衡", "显著提升OOD场景下的置信度校准性能并保持推理准确率"]
benchmarks: ["Big-Math", "GSM8K", "MATH-500", "AMC23", "AIME24", "AIME25", "StrategyQA", "GPQA", "HotpotQA", "SimpleQA", "NQ-Open", "TriviaQA"]
---

# 论文速读：RL-ARC-Calibrating-Large-Reasoning-Models-via-Reasoning-guid

## 一句话总结
本文提出RL-ARC，一种结合推理置信度与答案置信度的校准感知强化学习训练框架，通过在正确预测上使用推理引导正则化、在错误预测上施加过度自信惩罚，在保持推理准确率的同时显著改善模型校准，尤其在下分布迁移场景下效果突出。

## 研究问题与动机
- **RLVR训练导致过度自信**：现有大推理模型普遍采用可验证奖励强化学习(RLVR)训练，仅关注答案正确性，忽略置信度校准，导致模型对错误答案也给出高置信度。
- **现有校准方法存在准确性-校准权衡**：已知的校准感知训练方法（如RLCR）虽然改善了置信度校准，但在分布外(OOD)场景下仍出现过度自信，且会明显牺牲推理性能。
- **缺乏对推理过程可信度的考量**：现有方法仅校准最终答案的置信度，未考虑推理过程本身的可靠性，导致模型可能基于错误推理给出高置信度答案。
- **高 stakes 场景对可靠性要求严苛**：医疗、金融等高风险领域要求模型不仅准确，还需能准确评估自身不确定性。

## 核心贡献（创新点）
- **提出RL-ARC双置信度校准框架**：首次同时建模推理置信度(c_r)和答案置信度(c_a)，将其统一纳入强化学习训练目标。
- **设计条件依赖的差异化优化策略**：对正确预测使用推理引导正则化（最小化c_a与c_r的偏差），对错误预测使用过度自信惩罚（惩罚高c_r的情况）。
- **缓解准确性-校准权衡**：实验证明该方法在大幅提升校准指标（ECE降低最多达40%+）的同时，保持与标准RLVR相当的推理准确率。
- **增强OOD泛化能力**：在复杂推理和事实问答等OOD基准上，校准性能显著优于现有方法，且置信度分布更加合理。

## 方法详解
- **双置信度 elicitation**：通过特定提示模板，让模型在推理过程中同时输出推理置信度(c_r)和答案置信度(c_a)。
- **基础奖励函数**：沿用RLVR的二元奖励R_a = 1_{y=y*}，确保推理性能不受影响。
- **校准感知奖励设计**：
  - R_total(z, c_a, c_r) = R(z, c_a) - Ω_arc(z, c_a, c_r)
  - 其中R(z, c_a) = z - (z - c_a)²，基于Brier Score改进
- **条件依赖的正则化/惩罚项**：
  - 当z=1（预测正确）：Ω_arc = λ_pos × (c_a - c_r)²，鼓励答案置信度与推理置信度一致
  - 当z=0（预测错误）：Ω_arc = λ_neg × c_r²，惩罚基于错误推理的高置信度
- **超参数设置**：默认λ_pos=0.4, λ_neg=0.15，正样本权重更高，体现"奖励正确+惩罚错误"的不对称策略。

## 实验与结果
- **训练数据**：Big-Math（30K样本），评估覆盖算术推理ID基准（Big-Math, GSM8K, MATH-500, AMC23, AIME24/25）和OOD基准（StrategyQA, GPQA, HotpotQA, SimpleQA, NQ-Open, TriviaQA）。
- **模型基线**：Qwen2.5-7B, Qwen3-8B, Llama-8B (DeepSeek-R1蒸馏版)。
- **对比方法**：Base, RLVR w/Confidence, RLVR w/Probability, RLVR w/Post-hoc, Behavioral Calibration, RLCR。
- **Qwen2.5-7B主结果**：
  - ID：Acc=56.53%, AUROC=0.60, Brier=0.23, ECE=0.18（ECE较RLCR降低18%）
  - OOD：Acc=47.27%, AUROC=0.54, Brier=0.26, ECE=0.16（ECE较RLCR降低20%）
- **Qwen3-8B主结果**：
  - ID：Acc=66.10%, AUROC=0.83, Brier=0.13, ECE=0.15
  - OOD：Acc=51.61%, AUROC=0.69, Brier=0.23, ECE=0.17
- **核心结论**：RL-ARC在保持与RLVR相近准确率的同时，在所有校准指标上显著优于现有方法，尤其在小覆盖率区域的风险-覆盖曲线表现最佳。

## 相关工作脉络
- **RLCR (Damani et al., 2026)**：代表性校准感知RL方法，仅利用答案置信度，未考虑推理过程可靠性；RL-ARC通过引入推理置信度实现更优校准。
- **Behavioral Calibration (Wu et al., 2026)**：聚合推理过程中所有 verbalized 置信度进行校准；RL-ARC采用更精细的条件依赖策略。
- **C²GSPG (Liu et al., 2025)**：通过校准梯度序列实现自洽推理；RL-ARC聚焦于训练阶段的置信度联合优化。
- **置信度提取方法**：包括概率方法（平均token概率）、后验方法（外部置信度估计器）、准确率方法（多采样一致性）；RL-ARC采用verbalized confidence并进一步区分推理与答案层面。
- **定位差异**：本文首次系统性地论证推理置信度对答案校准的辅助价值，并提出可兼容不同模型架构的通用训练框架。

## 局限性与未来方向
- **训练设置有限**：仅在两种训练分布（数学推理/复杂推理）和单一RL算法（GRPO）下验证，未测试更强RL算法或其他训练分布。
- **模型范围限制**：主要验证Qwen系列模型，扩展至多模态推理模型将是重要方向。
- **推理正确性评估依赖LLM-as-a-Judge**：可能存在评估偏差，需要更可靠的推理正确性自动评估方法。
- **超参数敏感性**：λ_pos和λ_neg的取值需要仔细调优，当前默认值未必是最优配置。

## 研究启发与可借鉴点
- **双置信度解耦思路**：将"过程可信度"与"结果可信度"分离建模的思想可迁移至其他需要可靠性评估的生成任务（如代码生成、对话系统）。
- **条件依赖的训练目标设计**：根据预测正确性动态调整正则化/惩罚强度的策略，可作为通用模块嵌入其他RL训练框架。
- **验证置信度质量的方法**：控制实验（随机打乱置信度保留分布）可有效区分"分布平坦化"与"真实校准改善"，值得借鉴。
- **与团队的结合点**：可将推理置信度信号融入团队现有的长思维链训练 pipeline，或在评测阶段利用推理置信度进行选择性输出过滤，提升系统可靠性。

## 关键术语表
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：基于可验证奖励的强化学习训练范式，仅以答案正确性作为二元奖励信号。
- **Answer Confidence (c_a)**：模型对自身最终答案正确性的自评估置信度。
- **Reasoning Confidence (c_r)**：模型对推理过程正确性的自评估置信度。
- **ECE (Expected Calibration Error)**：期望校准误差，衡量模型置信度与真实准确率之间偏差的指标。
- **Brier Score**：二次损失函数，衡量预测概率分布与真实标签之间的差异。
- **OOD (Out-of-Distribution)**：分布外，指测试数据与训练数据分布不一致的场景。
- **AURC (Area Under Risk-Coverage Curve)**：风险-覆盖曲线下的面积，衡量选择性预测性能。
- **Shallow Reasoning**：浅层推理，指模型给出正确答案但推理过程错误或无关的现象。

## 可复现要素
- **训练数据集**：Big-Math（公开），30K问题子集用于训练；评估数据集均公开（GSM8K, MATH-500, AMC23, AIME24/25, StrategyQA, GPQA, HotpotQA, SimpleQA, NQ-Open, TriviaQA）。
- **代码/权重**：论文未明确声明代码开源状态。
- **关键超参数**：λ_pos=0.4, λ_neg=0.15, 学习率=5e-6, batch_size=1536, rollout_group_size=8, rollout_temperature=0.7, max_input_length=1024, max_output_length=4096, warmup_ratio=0.2。
- **基座模型**：Qwen2.5-7B, Qwen3-8B, Llama-8B (DeepSeek-R1蒸馏版)。
- **RL算法**：GRPO。
