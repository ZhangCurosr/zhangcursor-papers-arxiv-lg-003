---
title: "RL-ARC-Calibrating-Large-Reasoning-Models-via-Reasoning-guid"
source: https://arxiv.org/pdf/2610.11352v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:26:28"
field: "大模型推理与校准"
keywords: ["calibration", "reinforcement learning", "reasoning models", "uncertainty estimation", "overconfidence", "OOD generalization"]
innovations: ["提出 RL-ARC 框架，联合推理置信度与答案置信度进行条件化校准", "对正确/错误预测采用差异化正则化策略，缓解 accuracy-calibration trade-off", "在 ID 与 OOD 场景下实现校准与性能的联合优化"]
benchmarks: ["Big-Math", "GSM8K", "MATH-500", "AMC23", "AIME24", "AIME25", "StrategyQA", "GPQA", "HotpotQA", "SimpleQA", "NQ-Open", "TriviaQA"]
---

# 论文速读：RL-ARC: Calibrating Large Reasoning Models via Reasoning-guided Uncertainty

## 一句话总结
本文提出 RL-ARC，一个校准感知的强化学习框架，通过联合利用推理置信度（reasoning confidence）和答案置信度（answer confidence），在不显著牺牲推理性能的前提下有效改善大推理模型（LRM）的校准性，尤其在分布外（OOD）场景下表现突出。

## 研究问题与动机
- **RLVR 导致过度自信**：标准强化学习可验证奖励（RLVR）仅基于预测正确性提供奖励，不考量模型输出置信度，易强化过度自信的推理过程。
- **现有校准方法在 OOD 下失效**：已有的校准感知训练方法虽改善了校准，但在分布外场景下仍存在过度自信问题，且常以牺牲推理性能为代价（accuracy-calibration trade-off）。
- **缺乏对推理过程可靠性的关注**：现有方法仅校准答案置信度，未考虑推理过程本身的可信度，导致模型可能基于错误推理给出高置信度错误预测。

## 核心贡献（创新点）
1. **提出 RL-ARC 框架**：首次将推理置信度作为辅助信号引入校准感知强化学习，联合优化推理过程与答案预测的置信度对齐。
2. **条件化置信度正则化策略**：对正确预测使用推理置信度进行正则化（鼓励推理与答案一致），对错误预测施加过度自信惩罚（利用低推理置信度抑制高答案置信度）。
3. **缓解 accuracy-calibration trade-off**：实验证明 RL-ARC 在显著改善校准的同时，保持与纯 RLVR 相当的推理准确率，突破了已有校准方法的性能瓶颈。
4. **通用性验证**：在 Qwen2.5/3、Llama-8B 等多个模型架构及数学/复杂推理/事实问答等多种任务分布下验证了方法的有效性。

## 方法详解
**置信度提取（Confidence Elicitation）**：
- 提示模型在推理过程中同时输出推理置信度 $c_r$ 和答案置信度 $c_a$（verbalized confidence）。
- 推理置信度反映模型对自身推理过程正确性的评估，答案置信度反映对最终预测正确性的评估。

**校准感知优化目标（Calibration-aware Optimization）**：
- 总奖励函数：$R_{\mathrm{total}}(z, c_a, c_r) = R(z, c_a) - \Omega_{\mathrm{arc}}(z, c_a, c_r)$
- 其中 $z = \mathbf{1}_{y=y^*}$ 为预测是否正确，$R(z, c_a) = z - (z - c_a)^2$ 为基于 Brier Score 的校准奖励。
- 关键正则项 $\Omega_{\mathrm{arc}}$ 根据预测正确性条件化：
  - **正确情况（z=1）**：$\Omega_{\mathrm{arc}} = \lambda_{\mathrm{pos}}(c_a - c_r)^2$，鼓励答案置信度与推理置信度对齐（推理引导正则化）。
  - **错误情况（z=0）**：$\Omega_{\mathrm{arc}} = \lambda_{\mathrm{neg}} c_r^2$，当推理置信度 $c_r$ 较高时施加惩罚（过度自信惩罚）。
- 超参数 $\lambda_{\mathrm{pos}}=0.4$，$\lambda_{\mathrm{neg}}=0.15$（默认值）。

## 实验与结果
**数据集与基线**：
- 训练数据：Big-Math（30K 样本）、HotpotQA（消融实验）
- ID 评测：Big-Math、GSM8K、MATH-500、AMC23、AIME24/25
- OOD 评测：StrategyQA、GPQA、HotpotQA、SimpleQA、NQ-Open、TriviaQA
- 基线：Base、RLVR、w/Confidence、w/Probability、w/Post-hoc、Behavioral Calibration、RLCR

**主要结果（Qwen2.5-7B）**：
| 设置 | 方法 | Acc.(↑) | AUROC(↑) | Brier(↓) | ECE(↓) |
|------|------|---------|----------|----------|--------|
| **ID (Avg.)** | RLVR | 56.86 | 0.45 | 0.41 | 0.40 |
| | RLCR | 54.82 | 0.58 | 0.25 | 0.22 |
| | **RL-ARC** | **56.53** | **0.60** | **0.23** | **0.18** |
| **OOD (Avg.)** | RLVR | 47.58 | 0.50 | 0.50 | 0.50 |
| | RLCR | 46.02 | 0.54 | 0.28 | 0.20 |
| | **RL-ARC** | **47.27** | **0.54** | **0.26** | **0.16** |

- RL-ARC 在 OOD 上 ECE 较 RLCR 降低 20%（0.16 vs 0.20），Acc. 保持相近。
- Qwen3-8B 上 OOD ECE 达 **0.17**（最佳），Acc. 51.61%。
- Llama-8B 上 OOD AURC 达 **0.25**（最佳），较 RLCR 提升显著。
- 在事实 QA（SimpleQA/NQ/TriviaQA）OOD 任务上，RL-ARC 同时获得最高 Acc. 与最低 ECE。

## 相关工作脉络
1. **RLVR (Lambert et al., 2025)**：标准推理强化学习，仅基于答案正确性提供二元奖励，是本文基准对比方法。
2. **RLCR (Damani et al., 2026)**：首个将 verbalized confidence 引入 RLVR 的校准感知方法，仅校准答案置信度，本文主要对比对象。
3. **Behavioral Calibration (Wu et al., 2026)**：聚合推理过程中的 verbalized confidence 进行校准，未区分推理与答案置信度，本文认为其未能有效利用推理过程信息。
4. **Post-hoc 校准 (Li et al., 2025)**：训练外部置信度估计器，在 ID 上表现良好但 OOD 泛化受限，本文指出其依赖 ID 数据训练的局限性。
5. **概率基校准 (Kuhn et al., 2023)**：基于 token 概率计算语义不确定性，在开放生成场景适用性有限，需额外推理开销。

## 局限性与未来方向
- **训练数据有限**：仅使用数学推理（Big-Math）和复杂推理（HotpotQA）两种分布，未覆盖更多样化的任务类型。
- **RL 算法单一**：所有方法统一使用 GRPO，未来可探索更多强 RL 算法（如 PPO、PPO-Clip 等）下的泛化性。
- **模型家族局限**：主要聚焦 Qwen 系列，多模态推理模型的校准尚未探索。
- **推理正确性评估误差**：使用 LLM-as-a-judge 评估推理过程正确性可能引入偏差，需更精确的评估方法。

## 研究启发与可借鉴点
1. **条件化正则化策略**：根据预测正确性动态调整正则化目标（正样本对齐 vs 负样本惩罚），可有效平衡校准与性能，该思路可迁移至其他校准场景。
2. **双置信度联合建模**：将推理过程置信度与答案置信度分离并联合优化，为"过程监督"提供新思路，可探索在 CoT 推理中的延伸应用。
3. **OOD 校准评估标准化**：本文系统对比 ID/OOD 下多种校准指标（ECE/Brier/AUROC/AURC/Risk-Coverage），为后续工作提供完整评测范式。
4. **超参数敏感性分析**：通过对 $\lambda_{\mathrm{pos}}/\lambda_{\mathrm{neg}}$ 的系统扫描发现正主导设置（$\lambda_{\mathrm{pos}} > \lambda_{\mathrm{neg}}$）更优，为类似框架的超参调优提供参考。
5. **Shallow Reasoning 度量**：定义并量化"正确预测+错误推理"的浅层推理比例，为推理可靠性评估提供新视角。

## 关键术语表
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：基于可验证奖励的强化学习，通过答案正确性提供二元奖励以训练推理模型。
- **Reasoning Confidence ($c_r$)**：模型对自身推理过程正确性的自我评估置信度（verbalized）。
- **Answer Confidence ($c_a$)**：模型对最终预测答案正确性的自我评估置信度。
- **Brier Score**：衡量预测概率与真实标签之间差异的严格评分规则，本文用于校准奖励设计。
- **ECE (Expected Calibration Error)**：按置信度分箱计算的期望校准误差，衡量预测置信度与实际准确率的偏差。
- **AURC (Area Under Risk-Coverage Curve)**：选择性预测场景下的风险-覆盖率曲线下面积，越低表示高置信预测越可靠。
- **Shallow Reasoning (SR)**：模型给出正确预测但推理过程错误/无关的比例，用于量化推理可靠性。
- **OOD (Out-of-Distribution)**：测试数据分布与训练分布不同的场景，用于评估模型泛化与校准稳健性。

## 可复现要素
- **训练数据集**：Big-Math（30K 样本）、HotpotQA（消融实验）——论文未提供公开链接，需自行获取。
- **代码开源**：论文未明确声明代码开源状态。
- **权重开源**：未开源，使用 Qwen2.5-7B、Qwen3-8B、Llama-8B（DeepSeek-R1 蒸馏）作为 backbone。
- **关键超参**：$\lambda_{\mathrm{pos}}=0.4$，$\lambda_{\mathrm{neg}}=0.15$；学习率 5e-6；batch size 1536；rollout group size 8；temperature 0.7；epochs 1。
- **训练硬件**：4× NVIDIA H100/A100。
- **评估工具**：math-verify（数学答案验证）、LLM-as-a-judge（非数学任务评估）。
