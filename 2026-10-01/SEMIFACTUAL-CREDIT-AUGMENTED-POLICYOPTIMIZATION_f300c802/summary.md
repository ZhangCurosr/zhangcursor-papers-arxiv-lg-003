---
title: "SEMIFACTUAL-CREDIT-AUGMENTED-POLICYOPTIMIZATION"
source: https://arxiv.org/pdf/2609.40360v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:36:43"
field: "大语言模型强化学习"
keywords: ["RLVR", "GRPO", "token-level credit assignment", "semifactual intervention", "reasoning robustness", "policy optimization"]
innovations: ["提出半事实信用增强策略优化SCAPO，通过token级概率漂移信号细化GRPO信用分配", "设计负向-only校正机制，仅惩罚不稳定token而不额外奖励稳定性", "证明无需过程监督或外部奖励模型即可实现细粒度token级信用分配"]
benchmarks: ["AIME 2024-2026", "AMC 2023-2025", "HMMT 2025-2026", "GPQA-Diamond", "NoOp-AIME", "ThinkBench-AIME"]
---

# 论文速读：SEMIFACTUAL-CREDIT-AUGMENTED-POLICY-OPTIMIZATION

## 一句话总结
本文提出 SCAPO（半事实信用增强策略优化），通过将半事实提示干预下的 token 级概率漂移信号引入 GRPO 的信用分配机制，实现对相对稳定 token 的优势惩罚，从而提升 LLM 数学推理的准确性与分布外泛化能力。在 Qwen3-4B-Base 和 Qwen3-1.7B-Base 上，SCAPO 分别在 AIME 2024–2026 上超越 GRPO +5.63 和 +4.17 个百分点。

## 研究问题与动机
- **RLVR 的鲁棒性缺陷**：尽管 RLVR 显著提升了 LLM 推理能力，但模型预测仍对与任务无关的提示特征敏感，存在虚假相关性依赖。
- **GRPO 信用分配的粗糙性**：GRPO 对响应中每个有效 token 分配相同的结果派生优势，无法区分 token 级对无关特征的敏感性，可能强化虚假依赖。
- **缺少细粒度信用信号**：现有方法缺乏无需过程监督或外部奖励模型的 token 级别信用校正机制。

## 核心贡献（创新点）
- **揭示 token 级半事实敏感性**：通过固定响应的半事实提示干预，发现不同 token 类别对无关扰动敏感性差异显著（反思标记漂移为整体均值 2.74×，数学符号仅 0.47×）。
- **提出负向-only 信用校正机制**：SCAPO 将半事实稳定性转化为相对稳定性分数，仅取其负值部分附加到 GRPO 优势上，不额外奖励稳定性本身。
- **高效且无需额外标注**：无需过程监督信号或外部奖励模型，只需 teacher-force 采样响应并测量概率漂移即可实现 token 级信用细化。

## 方法详解
- **半事实扰动构造**：对每个训练 prompt 生成 4 种扰动（paraphrase、typo noise、irrelevant scenario wrapping、appended irrelevant context），保持数学问题和答案不变。
- **Token 概率漂移测量**：对固定响应在原始提示和 K 个扰动提示下进行 teacher-forcing，计算 bounded-symmetric distance：$d_{i,t}^{(k)} = \frac{|p_{i,t}^{(0)} - p_{i,t}^{(k)}|}{(p_{i,t}^{(0)} + p_{i,t}^{(k)})/2}$。
- **组内相对稳定性估计**：对漂移取负后做组内 z-score 标准化，跨 K 个扰动类型取平均后再做一次 z-score，保留负值部分 $u_{i,t}^- = \min(u_{i,t}, 0)$。
- **信用增强策略优化**：token 级优势更新为 $\tilde{A}_{i,t} = A_i + \lambda \cdot \text{sg}(u_{i,t}^-)$，其中 $\lambda$ 在前 $N_0$ 步非零，之后归零；梯度在此修正项上阻断。
- **计算开销极低**：半事实探测仅占训练运行时间的 1.8%，因复用采样响应并通过 teacher-forcing 避免重新生成。

## 实验与结果
- **数据集**：DAPO-Math-17K（约 17,000 道竞赛级数学题）。
- **基线方法**：GRPO、GSPO、SAPO、CF-GRPO、FIPO。
- **主要结果**：
  - Qwen3-4B-Base：AIME 24–26 准确率 37.40% vs GRPO 31.77%（+5.63pp），在所有 8 个数学基准上领先。
  - Qwen3-1.7B-Base：AIME 24–26 准确率 11.88% vs GRPO 7.71%（+4.17pp），在所有 8 个数学基准上领先。
  - 分布外泛化：GPQA-Diamond（科学问答）4B 提升 +6.19pp，1.7B 提升 +3.83pp。
  - 加速收敛：SCAPO 在不到 GRPO 一半的优化步数内达到 GRPO 的最终 AIME 准确率。
- **消融验证**：随机打乱信用信号、反事实扰动替代均显著劣于 SCAPO，支持信用应与半事实敏感性对齐；仅使用正向校正或全符号校正均低于负向-only 设置。

## 相关工作脉络
- **RLVR 方法演进**：GRPO（Shao et al., 2024）→ 后续改进归一化/采样（Liu et al., 2025; Yu et al., 2025）→ 重要性权重与裁剪改进（Zheng et al., 2025; Gao et al., 2025b）。
- **细粒度信用分配**：过程监督方法（Lightman et al., 2024）、蒙特卡洛估计（Kazemnejad et al., 2025）、token 熵/置信度/资格迹（Wang et al., 2025b; Mou et al., 2026; Ma et al., 2026）。
- **扰动/梯度归因方法**：CF-GRPO（Khandoga et al., 2026）通过 counterfactual masking 估计 token 级信用；本文与之区别在于使用 semifactual 而非 counterfactual 扰动（保持答案不变）。
- **鲁棒性研究**：NoOp/AIME 扰动（Mirzadeh et al., 2025）、ThinkBench（Huang et al., 2025）评估输入扰动鲁棒性；本文从训练信用分配角度解决类似问题。

## 局限性与未来方向
- **模型规模限制**：仅在 1.7B 和 4B 稠密 Qwen3 模型上验证，未测试更大模型或 MoE/混合架构。
- **数据规模限制**：训练数据仅 DAPO-Math-17K（约 17K 题），大规模数据和更多样化领域效果待验证。
- **领域泛化有限**：GPQA-Diamond 展示了一定跨域转移能力，但数学推理之外的其他领域需进一步探索。
- **未来方向**：扩展到更大模型规模、更多架构类型、更广泛训练域。

## 研究启发与可借鉴点
- **半事实干预作为诊断工具**：无需训练即可量化 token 级敏感性，可用于快速评估模型鲁棒性。
- **负向-only 校正设计**：仅惩罚不稳定 token 而不奖励稳定 token，避免引入虚假正反馈，设计简洁且有效。
- **teacher-forcing 高效探测**：复用采样响应避免重新生成，将额外计算开销控制在 1.8%，适合集成到现有 RLVR pipeline。
- **早期训练阶段介入**：信用增强仅在初始 $N_0$ 步生效，之后回归标准 GRPO，平衡了探索引导与最终优化。

## 关键术语表
- **RLVR**：Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习，用于训练 LLM 推理能力。
- **GRPO**：Group Relative Policy Optimization，通过组内相对优势进行策略优化的 RLVR 方法。
- **Semifactual intervention**：半事实干预，修改提示的无关特征但保持问题和答案不变的扰动方式。
- **Bounded-symmetric distance**：有界对称距离，用于衡量两个概率值之间相对差异的度量，取值范围 [0, 2]。
- **Token-level credit assignment**：Token 级信用分配，为响应中每个 token 分配独立的学习信号。
- **Teacher-forcing**：教师强制，在推理时强制模型使用预定义的正确 token 序列而非自回归生成。
- **Stop-gradient (sg)**：梯度阻断操作，阻止梯度通过某项传递。
- **Out-of-distribution (OOD)**：分布外，指训练数据分布之外的评估场景或数据。

## 可复现要素
- **数据集**：DAPO-Math-17K（公开）、AIME/AMC/HMMT 等基准（公开竞赛题）。
- **代码**：开源，GitHub: https://github.com/DtYXs/SCAPO，基于 EasyR1 框架构建。
- **模型**：Qwen3-4B-Base 和 Qwen3-1.7B-Base（公开预训练权重）。
- **关键超参**：$\lambda_0 = 0.01$，$N_0 = 120$（4B）/200（1.7B）步，学习率 $10^{-6}$，rollout batch size=128，每 prompt 8 个采样，最大响应长度 16,384 tokens。
