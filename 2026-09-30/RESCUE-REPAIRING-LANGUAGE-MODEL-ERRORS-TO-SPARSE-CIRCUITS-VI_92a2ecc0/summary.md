---
title: "RESCUE-REPAIRING-LANGUAGE-MODEL-ERRORS-TO-SPARSE-CIRCUITS-VI"
source: https://arxiv.org/pdf/2609.36813v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:48:32"
field: "LLM 机制可解释性与模型修复"
keywords: ["mechanistic interpretability", "error circuit", "sparse circuit editing", "reinforcement learning", "language model repair", "circuit localization"]
innovations: ["RL-based 生成式电路精炼与组相对策略梯度", "两阶段 ablation 验证 + 限定微调闭环", "跨重述迁移证明因果稳健性"]
benchmarks: ["GSM8K", "MATH-500", "MedMCQA", "ARC-Challenge", "HellaSwag", "WinoGrande", "PIQA", "TruthfulQA-MC1", "MMLU-HS Math/Stats", "BBH Multi-Step Arithmetic"]
---

# 论文速读：RESCUE-REPAIRING-LANGUAGE-MODEL-ERRORS-TO-SPARSE-CIRCUITS-VI

## 一句话总结
论文提出 **RESCUE**（Reasoning-Error Sparse-Circuit Uncovering and Editing），将稀疏电路定位与强化学习结合，精准修复 LLM 的多步推理错误。仅调参约 **1.4%** 密度即可将 GSM8K 准确率从 80% → **91%**，医学 QA 从 0% → **81%**，同时在八大非目标基准上保留 **99.4%** 基线能力。

---

## 研究问题与动机
1. **现有电路研究偏向"功能保持"**：已有工作主要定位事实检索、推理等成功行为的稀疏子图，缺乏对错误成因的系统性归因。
2. **SFT-based mask 优化在多步生成任务中失效**：早期推理偏差导致前缀偏离监督参考，使基于固定纠错参考的掩码搜索忽略真正参与错误生成的电路。
3. **安全导向方法范围有限**：Hallucination/Backdoor 归因只覆盖单点神经元，未扩展到面向广泛任务失败的精准修复。
4. **缺乏可泛化的错误定位框架**：现有方法要么依赖昂贵的手动干预搜索，要么只关注行为解释而非修复。

---

## 核心贡献（创新点）
1. **生成式电路定位新范式**：首次将错误归因从"解释"转向"修复"，把错误电路视为可抑制的因果源。
2. **RL-based 掩码精炼机制**：用组相对策略梯度在多次 masked-model rollout 上优化掩码，避免单条参考响应带来的信息瓶颈。
3. **两级闭环验证**：先 ablation 验证抑制效果，再 circuit-restricted tuning 微调参数，确保定位与实际修复因果一致。
4. **跨表述迁移证据**：同一电路在无重定位、无重调参条件下对改写输入仍提升 8–66pp，证明捕获的是计算机制而非表面词汇模式。
5. **极简参数修复**：仅修约 **1.4%** MLP 输出行，对比 LoRA 全参数低秩微调在多数场景下损失更小、效果更稳。

---

## 方法详解

### 4.1 数据构造（双路径监督）
- **清洁集** $\mathcal{D}_{\text{clean}} = \{(x_j, y_j)\}$：模型答对的原始生成。
- **纠错集** $\mathcal{D}_{\text{err}} = \{(x_i, y_i^-, y_i^+)\}$：模型答错的原回答 $y_i^-$，由外部 LLM（Gemini 3.1 Pro）按**最小干预原则**改写到正确答案 $y_i^+$，保留有效中间步骤、仅修改错误部分。

### 4.2 错误电路发现（三阶段）

**① 候选电路搜索（SFT-based mask optimization）**
- 每个 MLP 输出行 $(l,p,j)$ 对应一个可训练 mask logit $z_u$，定义 keep-mask：
  $$c_u = \operatorname{clip}_{[0,1]}\!\left(2\sigma\!\left(\frac{z_0 - z_u}{\tau}\right)-1\right), \quad m_u = 1 - c_u$$
- 初始化 $z_u = z_0$ 使 $m_u = 1$（全保留）。
- 目标函数：
  $$\mathcal{L}_{\text{cand}} = \mathcal{L}_{\text{corr}} + \lambda_{\text{clean}}\mathcal{L}_{\text{clean}} + \lambda_{\text{keep}}\frac{1}{|\mathcal{P}|}\sum_q\frac{1}{|\mathcal{U}_q|}\sum_{u\in\mathcal{U}_q} m_u + \mathcal{L}_{\text{sens}}$$
- 采用 **STE（Straight-Through Estimator）**：前向用二值掩码 $b_u$，反向通过连续 $m_u$ 传梯度；每模块设删除预算 $\rho_0$。

**② RL-based 电路精炼**
- 对每道错题采样 $K=4$ 个 masked-model 生成 $\{\hat{y}_{i,k}^-\}$。
- 组内归一化优势：
  $$r_{i,k} = \lambda_{\text{ans}} r_{\text{ans}}(\hat{y}_{i,k}, a_i) + \lambda_{\text{rea}} r_{\text{rea}}(x_i, \hat{y}_{i,k}, a_i), \quad A_{i,k} = \frac{r_{i,k} - \bar{r}_i}{\max(s_i, \epsilon)}$$
- 序列级目标：
  $$\mathcal{L}_{\text{RL}} = -\mathbb{E}_i\!\left[\frac{1}{K}\sum_{k=1}^K A_{i,k}\,\frac{1}{T_{i,k}}\sum_{t=1}^{T_{i,k}}\log p_{\theta,\mathbf{b}^{\text{ste}}}(\hat{y}_{i,k,t}\mid x_i, \hat{y}_{i,k,<t})\right]$$
- 辅助稳定：纠错参考 NLL（在组内 reward 均匀时加重）、清洁 NLL、梯度冲突时投影消去对立分量、最后可行掩码回退机制。

**③ 电路剪枝**
- 按关闭强度 $c_u$ 由弱到强逐组重新开放，判断标准：
  $$S_{\text{err}}(\mathbf{b}') \ge S_{\text{err}}(\mathbf{b}^{\text{RL}}) - \epsilon_{\text{err}}, \quad S_{\text{clean}}(\mathbf{b}') \ge S_{\text{clean}}(\mathbf{b}^{\text{RL}}) - \epsilon_{\text{clean}}$$
- 保留仅抑制必要单元，得到最终错误电路 $\mathcal{C}^* = \{u\mid b_u^*=0\}$。

### 4.3 电路验证与限定微调
- **Ablation 验证**：比较 $\mathcal{G}$ 与 $\mathcal{G}\setminus\mathcal{C}^*$ 在错题集/清洁集上的 exact-match 准确率，验证"抑制=纠错"且"清洁不变=保留能力"。
- **Circuit-Restricted Tuning**：仅在 $\mathcal{C}^*$ 对应行引入零初始化可训练 delta $\Delta\theta(\mathcal{C}^*)$，冻结其余参数；目标：
  $$\min_{\Delta\theta(\mathcal{C}^*)} -\mathbb{E}_{(x,y^+)\sim\mathcal{D}_{\text{err}}}\!\left[\frac{\sum_t w_t \log p_{\theta+\Delta\theta(\mathcal{C}^*)}(y_t^+\mid x, y_{<t}^+)}{\sum_t w_t}\right]$$
  其中 $w_t$ 加权最终答案 token，训练后 merge 最佳 delta。

---

## 实验与结果

### 数据集与模型
- **主模型**：Qwen3-8B、Llama-3.1-8B-Instruct
- **目标任务**：GSM8K（数学推理，5 种模式特定修复集 + 异质修复集）、MedMCQA（医学 QA，跨域验证）
- **非目标基准（8 个）**：ARC-C、HellaSwag、WinoGrande、PIQA、TruthfulQA-MC1、MMLU-HS Math/Stats、BBH-多步算术 → 汇总为 **Aggregate Evaluation**
- **泛化基准**：MATH-500

### 关键结果
| 模型 | 基线 GSM8K | RESCUE 修复后 | 提升 | 对照 LoRA | 对照 Supervised-only |
|---|---|---|---|---|---|
| Qwen3-8B | 80% | **91%** | +11pp | 88%（-3pp） | 87%（-4pp） |
| Llama-3.1-8B | 73% | **77%** | +4pp | 50%（-27pp） | 73%（无提升） |

- **电路密度**：Qwen3-8B 用 1.49% 行，Llama-3.1-8B 用 1.40% 行
- **全集修复准确率**：Qwen3-8B 0%→50.5%（ablation）， tuning 后 53.5%；Llama-3.1-8B 6.0%→69.0%→75.5%
- **能力保留**：Aggregate Evaluation 平均保留 **99.4%** 基线
- **跨重述迁移**：Llama-3.1-8B 在 Cumulative Target Gap 上 25%→60%（+35pp），Percentage Risk 26%→92%（+66pp）
- **随机电路对照**：10 个同密度随机 mask 最高仅 30%，RESCUE 达 90–95%，证明**归因必要性**
- **顺序修复**：5 阶段累积准确率 Qwen3-8B 1.8%→90.8%，Llama 19.6%→92.6%，前期修复最多下降 8pp
- **MedMCQA 跨域**：RESCUE 修复集 0–3%→78.5–81%，Aggregate 保留优于 LoRA

---

## 相关工作脉络
1. **Circuit Discovery**（Path Patching、ACDC、Attribution Patching、IBCircuit、Sparse Feature Circuits）—— 侧重行为保持/解释，本文转向错误归因与修复。
2. **Error Attribution**（CCS、SAT Probe、FAITH、ReDeEP）—— 多为隐藏状态探针或单神经元关联，缺乏因果干预闭环。
3. **Inference-time Intervention**（ITI、TruthX、DoLa）—— 解码阶段外部干预，本文在模型参数层面做稀疏修复。
4. **LoRA / 全参数微调**—— 本文证明限定电路微调在保留非目标能力上显著优于通用适配器。
5. **Hallucination 归因**（H-Neurons）—— 仅指出错误神经元存在，未建立"定位→ablation→tuning"的完整修复流程。
6. **安全电路**（Safeseek、Backdoor attribution）—— 面向对齐/后门，与通用任务错误修复定位不同。

---

## 局限性与未来方向
1. **依赖外部 LLM 生成纠错参考**，引入潜在污染风险；可探索自监督或 reward-model 信号替代。
2. **仅在 8B 模型验证**，大尺度模型（175B+）的电路密度与可修复性待验证。
3. **多错误模式交互**未充分建模，顺序修复最多损失 8pp，长期叠加行为需追踪。
4. **RL 阶段组相对奖励在高 variance 时不稳定**，需更多 rollout 或更好的 baseline。
5. **仅覆盖 MLP 投影行**，未探索 attention head / 残差路径的错误电路。
6. 未来可拓展至代码生成、多轮对话、agent 规划等更长推理链场景。

---

## 研究启发与可借鉴点
1. **组相对策略梯度 + 生成式反馈**的组合可直接迁移至任何需要"从失败样例中定位关键组件"的任务（如代码 linting、格式错误修复）。
2. **STE + 模块删除预算**的稀疏掩码优化框架，可用于其他需要结构化稀疏约束的模型编辑问题。
3. **ablation-before-tuning 的两步验证**可作为电路类研究的标配，增强因果结论可信度。
4. **跨重述迁移测试**（不重定位、不重调参）是检验电路因果稳健性的有效手段，建议在后续工作中复用。
5. **0.2 初始 mask logit + 可行性回退**的稳健训练技巧，对任何带 hard constraint 的 masked 优化都有参考价值。

---

## 关键术语表
**Sparse Circuit**：模型中仅占约 1.4% 的 MLP 输出行子图，抑制后即可纠正特定错误而基本不损伤其他能力。

**Keep-Mask**：可训练 sigmoid 映射值 $m_u\in[0,1]$，1 保留、0 关闭，用于软控制每个电路单元的开关。

**Straight-Through Estimator (STE)**：前向用二值掩码，反向通过连续值传梯度，使离散掩码优化可微。

**Group-Relative Advantage**：同一问题多个 rollout 的奖励减去组内均值、除以标准差，消除绝对尺度依赖。

**Circuit-Restricted Tuning**：仅在定位到的错误电路上引入零初始化 delta 并微调，其余参数冻结。

**Heterogeneous Repair**：混合多种错误模式的修复数据集，检验电路定位的泛化能力。

**Minimum-Intervention Correction**：由外部 LLM 按"仅改错误部分、保留有效推理"原则生成参考答案。

---

## 可复现要素
- **代码**：已开源 https://github.com/chuanpupig/RESCUE
- **数据**：GSM8K / MedMCQA 官方训练集，错误参考由 Gemini 3.1 Pro 生成，与公开评测集不相交
- **精度与种子**：BF16，固定 seed=42
- **关键超参**：mask 初始 logit=0.2，$\tau$ 温度系数，监督优化 10 epoch，RL 精炼 6 epoch，每组 rollout $K=4$
- **优化器**：详见附录 A.3（AdamW 变体）
- **评测工具**：lm-evaluation-harness v0.4.3
