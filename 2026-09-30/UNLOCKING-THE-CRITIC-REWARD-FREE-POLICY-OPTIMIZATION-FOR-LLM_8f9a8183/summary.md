---
title: "UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM"
source: https://arxiv.org/pdf/2609.37119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:36:46"
field: "大语言模型强化学习"
keywords: ["reward-free RL", "critic-based optimization", "LLM post-training", "chain-of-thought reasoning", "length bias debiasing"]
innovations: ["发现 critic 不稳定性源于优化方法而非网络本身，单步更新可恢复稳定", "单一冻结 critic 同时担任奖励、GAE 基线和未完成前缀预测器三重角色", "通过类别内长度去偏与二值化阻止策略利用 critic 长度偏差"]
benchmarks: ["AIME 2025", "AIME 2026", "AMC 2023", "GPQA-Diamond"]
---

# 论文速读：UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM

## 一句话总结
论文提出 RFPO（Reward-Free Policy Optimization）方法，将预训练的 critic 冻结并校准后直接用作奖励信号与 GAE 基线，在数学推理任务上达到与监督 PPO 相当的性能，但训练循环中无需任何外部验证器标签，显著降低计算与显存开销。

## 研究问题与动机
- 当前大模型 RL 后训练趋势是移除 critic 以避免其高昂的显存占用与训练不稳定性，但 critic 在训练中已学会预测轨迹的最终结果，其预测能力被浪费。
- 现有无奖励替代方案（多数投票、自奖励、隐式偏好优化等）仍依赖外部标注或容易产生 reward hacking，无法从根本上消除对 verifier 的依赖。
- 长链思考推理中，稀疏的终端正确性奖励导致信用分配困难，且等待每条轨迹完成会产生高昂的计算成本。
- 需要一种方法：既能利用 critic 已学得的轨迹成功概率估计，又不需要在策略优化阶段引入外部奖励信号。

## 核心贡献（创新点）
- **发现 critic 不稳定性是优化 artifact**：通过保持策略更新小且低方差（每批次仅一次梯度更新），可恢复稳定收敛，而非必须移除 critic。
- **单一冻结 critic 承担三重角色**：同一网络同时作为 rollout 级奖励、GAE 基线和未完成前缀的成功预测器，提供密集的前缀级学习信号。
- **二值化去偏得分阻止策略利用长度偏差**：通过校准去除 critic 分数的长度偏差后二值化，避免策略通过改变长度而非正确性来 exploit 奖励。
- **零标签训练匹配监督 PPO 性能**：在 5,120-token 预算下，RFPO 在 AIME 和 GPQA 上达到与监督 PPO 相当的水平，且无需训练循环中的任何验证器标签。
- **支持截断轨迹的高效训练**：critic 可对未完成的 rollout 进行准确评分，使训练无需等待轨迹完成即可分配奖励，特别适合长时域推理场景。

## 方法详解
- **Critic 预训练**：在数学推理数据上运行监督 PPO，使用 γ=λ=1 拟合 critic，使其学习 V_φ(x, y≤t) ≈ Pr(success | x, y≤t)，即从轨迹前缀预测最终成功的后验概率。
- **冻结与校准**：在累计步 800 提取 critic 后冻结，在 512 个 on-policy rollout 上拟合长度去偏项 b(ℓ) 和阈值 τ，公式为 r̂(x,y) = 1[v(x,y) - b(ℓ(y)) > τ]，其中 b(ℓ) 在正确性类别内估计长度十分位的偏移。
- **三重角色利用**：冻结的 critic 同时提供：(1) rollout 级奖励 r̂；(2) GAE 基线 V_φ(x, y≤t)；(3) 未完成前缀的成功预测。advantage 计算简化为 A_t = r̂ - V_φ(x, y≤t)。
- **部分监督混合**：引入监督分数 p，以概率 p 替换 r̂ 为真实 verifier 标签，p=0 为完全 reward-free，p=1 恢复监督 PPO。
- **单步更新规则**：每个批次仅执行一次策略梯度更新，mini-batch 等于完整 batch，无内部循环，显著降低更新方差。

## 实验与结果
- **数据集与模型**：Qwen3-4B-Base，在 OpenR1-Math-220k（45k 条）上 SFT，然后在 DAPO-Math-17k 上预训练 critic。
- **评估基准**：AIME 2025/2026、AMC 2023、GPQA-Diamond，宏平均作为主要指标。
- **主要结果**（300 步后）：RFPO 零标签在四个基准平均 pass@1 达 41.1%，监督 PPO 为 41.8%，差距仅 0.7 点；AIME 2026 上 RFPO 峰值 25.7% vs PPO 25.0%。
- **截断场景表现**：在 4,096-token 预算下约半数 rollout 未完成的条件下，RFPO 宏平均 47.6% 仍高于监督 PPO 的 46.6%，且 GPU 小时数减少 19%（264 vs 327）。
- **成本节约**：每步时间从 582 秒降至 421 秒（-28%），每 GPU 峰值显存减少 9.4 GB，吞吐提升 35%。
- **连续性奖励的突破与崩溃**：连续奖励可达 AIME 2026 pass@1 29.2%（高于监督 PPO 的 25.0%），但后期因策略 exploit 长度偏差而性能崩溃，二值化是维持稳定的关键。

## 相关工作脉络
- **Process Reward Models**（Lightman et al., 2024; Wang et al., 2024）：提供密集步骤级信号但需步骤标注，critic 则从结果学习并预测未来。
- **Implicit reward 方法**（DPO、PRIME）：训练时仍需标签，RFPO 的奖励在策略优化前已固定且经校准。
- **Label-free 替代方案**（多数投票、自奖励、模型置信度）：缺乏 grounded supervision 易导致 reward hacking，RFPO 基于 verifier 结果且受校准约束。
- **Critic-free 推理训练**（DeepSeek-R1、Kimi、Tulu 3）：完全移除价值网络，RFPO 证明保留 critic 可提供更丰富的信号。
- **现有 critic 修复工作**（EVPO、Generative Critics）：仍把 critic 仅用作 baseline，奖励来自外部，RFPO 进一步将 critic 用作奖励本身。

## 局限性与未来方向
- 实验仅限 4B 参数模型和数学推理任务，尚未验证更大模型或其他任务类型。
- 训练成本较高，单次 300 步 RL 运行需 264-388 GPU 小时（8×A100-80GB），限制了对更大规模模型的探索。
- 连续奖励虽能达到更高准确率峰值，但缺乏有效的停止规则防止后期崩溃，需要未来研究开发 validated stopping criterion。
- 当 rollout 与 critic 训练分布差异过大时（如 1,024-token 极端截断），critic 评分精度急剧下降，存在适用下限。

## 研究启发与可借鉴点
- **单步更新规则**：在长链思考 RL 中，每批次仅一次梯度更新可显著稳定 critic-based 训练，避免熵坍塌或发散，可作为默认配置。
- **Explained variance 作为 critic 质量指标**：训练时可低成本监控 explained variance，与离线 within-problem AUC 高度相关（r=0.91），便于实时评估 critic 可靠性。
- **长度偏差分离技术**：在正确性类别内估计长度偏移可分离信号与偏差，为其他基于 value 的奖励方法提供防 exploit 框架。
- **未完成轨迹的早期评分**：利用 critic 对 prefix 的预测能力，可在轨迹未完成时分配奖励，大幅缩短训练等待时间，特别适合长时域 agentic 任务。
- **部分监督混合策略**：引入监督分数 p 作为连续调节器，可在 reward-free 稳定性和 verifier 信号之间灵活平衡。

## 关键术语表
- **RFPO**：Reward-Free Policy Optimization，利用冻结 critic 作为奖励信号的策略优化方法，训练循环中无需外部标签。
- **GAE**：Generalized Advantage Estimation，结合多步回报与 value baseline 的 advantage 估计方法，此处简化为 A_t = r̂ - V_φ。
- **Explained Variance**：值网络对回报方差的解释比例，用于在线监控 critic 训练质量。
- **Within-problem AUC**：在同一问题内比较不同回答的 critic 评分区分正确/错误的能力，排除问题难度干扰。
- **Debiasing Term b(ℓ)**：在正确性类别内估计的长度十分位偏移，用于去除 critic 分数中的长度偏差。
- **Supervision Fraction p**：以概率 p 用真实 verifier 标签替换 critic 奖励，提供部分监督保护。
- **Pass@k**：在 k 次采样中至少一次正确的概率估计，用于评估推理任务性能。
- **Length Exploitation**：策略利用奖励中的长度偏差而非提高正确性的行为，通过二值化校准防止。

## 可复现要素
- **数据集**：OpenR1-Math-220k（SFT 用 45k）、DAPO-Math-17k（critic 预训练）、AIME 2025/2026、AMC 2023、GPQA-Diamond（评估）；论文未明确声明公开状态。
- **代码/权重**：附录 H 提供所有训练日志映射、校准 artifacts 和重现脚本，但未提及代码仓库 URL；模型权重未公开。
- **关键超参**：actor lr=10⁻⁶、critic lr=10⁻⁵（仅监督基线）、clip=0.2、γ=λ=1、temperature=1.0（采样）、128 prompts × 8 responses、训练 cap 5,120/4,096 tokens、验证 cap 12,288 tokens。
- **硬件**：8×A100-80GB GPU 节点，使用 verl 框架、bf16、FSDP、gradient checkpointing。
