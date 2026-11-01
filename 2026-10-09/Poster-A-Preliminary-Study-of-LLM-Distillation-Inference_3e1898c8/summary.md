---
title: "Poster-A-Preliminary-Study-of-LLM-Distillation-Inference"
source: https://arxiv.org/pdf/2610.12137v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:06:05"
field: "大模型安全与版权保护"
keywords: ["distillation inference", "knowledge distillation detection", "shadow models", "likelihood ratio test", "LLM security", "membership inference"]
innovations: ["将蒸馏推断形式化为带影子模型校准的假设检验，输出可解释 p-value", "从 per-instance 弱信号聚合为 per-model 强统计量，实现完全群体分离"]
benchmarks: ["CoT-Collection", "AUC/TPR@FPR", "balanced accuracy"]
---

# 论文速读：Poster: A Preliminary Study of LLM Distillation Inference

## 一句话总结
本文提出**蒸馏推断（distillation inference）**框架：将"嫌疑模型是否从目标教师模型蒸馏而来"建模为假设检验，通过训练蒸馏/独立两类影子模型来校准似然比统计量，最终输出带统计保证的 p-value。在 Qwen2.5-7B 教师、Llama-3.2-3B 嫌疑的初步实验中，该方法在 α=0.02 下实现 TPR=1.0、FPR≈0.02。

## 研究问题与动机
1. **非授权蒸馏攻击激增**：Google（2026.2）和 Anthropic（2026.2）报告 API 滥用型蒸馏攻击大幅上升，涉及知识产权与国家安全风险。
2. **现有方法缺乏统计保证**：已有工作仅给出嫌疑模型与教师的相似度打分或排名，无法为特定教师-嫌疑对提供带 p-value 的执法级决策。
3. **单样本信号过弱**：逐实例行为信号（per-instance）在蒸馏/独立类间重叠严重，AUC 仅 0.64–0.67，无法满足低 FPR 场景。
4. **审计场景设定**：假设教师持有嫌疑方的 API 查询日志，可作为审计集（D_audit ⊆ D_train），这是现实可行的切入点。

## 核心贡献（创新点）
1. **将蒸馏推断形式化为假设检验**：H₀（独立训练）vs H₁（从教师蒸馏），与现有方法的本质区别在于输出可解释的 p-value 而非相似度分数。
2. **影子模型校准的似然比检验**：引入蒸馏影子（用教师 CoT 微调）和独立影子（用参考答案微调）构建两个行为分布，将嫌疑模型得分转化为 calibrated log-likelihood ratio。
3. **三种蒸馏检测信号的统一提取**：log-probability（灰盒）、predictive entropy（灰盒）、token agreement（黑盒），均可用于同一框架。
4. **全分离的初步结果**：在 M=50 影子模型下，聚合后的 Λ 统计量使蒸馏/独立群体完全分离（AUC=1.0），并分析规避策略（paraphrase CoT）下的检测表现。
5. **同家族鲁棒性验证**：Qwen→Qwen 的同号词器对产生更大分布间隔，证明跨家族检测结果不受输出格式差异混淆。

## 方法详解

**Stage 1 — 训练影子模型**：从同一基础模型出发，随机取 D_audit 的一半：
- 蒸馏影子 f_m^dist：LoRA（rank=8, scale=16, dropout=0.05）微调，目标是复现教师 CoT t_i。
- 独立影子 f_m^ind：相同结构与超参，目标是参考答案 y_i。

**Stage 2 — 提取蒸馏信号**：对审计集每个实例 (x_i, t_i)，让影子模型做一次教师 CoT 的前向 pass，在每个位置 ℓ 计算条件分布 p_{i,ℓ}^f = f(· | x_i, t_i,<ℓ)，得到三个信号：
- **Log-probability**（灰盒）：g_lp(f,i) = (1/L_i) Σ_ℓ log p_{i,ℓ}^f(t_{i,ℓ})，衡量模型对教师 token 的保留程度。
- **Predictive entropy**（灰盒）：g_ent(f,i) = (1/L_i) Σ_ℓ H(p_{i,ℓ}^f)，衡量模型输出的确定性。
- **Token agreement**（黑盒）：g_ta(f,i) = (1/L_i) Σ_ℓ 1[ê_{i,ℓ}^f = t_{i,ℓ}]，衡量贪婪解码与教师的一致性。

**Stage 3 — 假设检验与校准**：
- 对每个实例 i，用 M 个蒸馏影子和 M 个独立影子的信号值拟合高斯分布 N(μ_dist,i, σ²_dist,i) 和 N(μ_ind,i, σ²_ind,i)。
- 计算实例级对数似然比（LLR）：λ_i(f) = log P(g(f,i) | N_dist) / P(g(f,i) | N_ind)。
- 聚合为模型级统计量：Λ(f) = Σ_i λ_i(f)。
- 使用留一法（leave-one-out）从影子模型获取 Λ 在 H₀ 和 H₁ 下的经验分布，避免自引用偏差。
- 最终 p-value：p = (1 + #{m : Λ(f_m^ind) ≥ Λ(f_sus)}) / (M + 1)。
- 决策：ŷ = 1[p < α]。

## 实验与结果
- **教师**：Qwen2.5-7B-Instruct；**嫌疑/影子基础模型**：Llama-3.2-3B。
- **数据集**：CoT-Collection [5]，N=500 审计实例；M=50 影子模型/类。
- **训练**：LoRA rank=8, scale=16, dropout=0.05，3 epochs，取验证集最低 loss checkpoint。

| 指标 | Per-instance (MIA baseline) | Per-model (本文) |
|---|---|---|
| AUC | 0.64–0.67 | 1.000 |
| Balanced Accuracy | ~0.51–0.52 | 1.000 |
| TPR @ 5% FPR | < 0.10 | 1.000 |
| TPR @ 0% FPR | ≤ 0.003 | 1.000 |

- **关键结果**：逐实例信号几乎无法区分，但跨审计集聚合后两群体完全分离；M=50 时 p-value 下限 = 1/(M+1)=0.020，FPR ≈ 1/M = 0.020。
- **缓解实验**：对 CoT 做 paraphrase（教师生成 / Mistral 生成）后检测统计量下降，但仍能在 α=0.05 下全部检出；用匹配 paraphrase 训练的蒸馏影子可恢复审计能力。

## 相关工作脉络
1. **LiRA（Shokri et al., 2017/Carlini et al., 2022）**：membership inference 的经典似然比攻击，本文直接借鉴其 per-instance 构造思路，将其迁移至模型级蒸馏推断。
2. **Distillation Quantification（Lee et al., 2025, ACL）**：量化蒸馏规模，但未提供带统计保证的个体模型检测。
3. **KD Detection for Open-weights（Shi et al., 2025, NeurIPS）**：检测开放权重模型是否经历蒸馏，但无 p-value 或 FPR 控制。
4. **Who Taught You That?（Wadhwa et al., 2025, ACL Findings）**：追踪教师来源，侧重排名而非假设检验。
5. **Shadow Model 方法论（Carlini et al., 2022; Nasr et al.）**：用影子模型估计不可观测分布，本文将其用于蒸馏推断而非成员推断。

## 局限性与未来方向
1. **影子模型训练成本高**：需要微调 M 个影子模型，对前沿大模型不现实。
2. **审计集需是训练集子集**：依赖教师持有嫌疑方 API 日志，现实中难以保证。
3. **假设嫌疑与影子使用相同训练流程**：未考虑不同 LoRA 超参、epoch、数据量等变体。
4. **未来方向**：放松审计集假设（D_audit 不一定是 D_train 子集）、减少影子模型数量、提升对抗 paraphrase 的鲁棒性。

## 研究启发与可借鉴点
1. **影子模型校准范式可直接迁移**：任何"检测模型是否学习过某数据分布"的问题（数据泄露、版权追溯）均可套用此框架。
2. **从 per-instance 到 per-model 聚合是关键设计**：逐样本信号弱不等于整体无用，LLR 聚合+高斯校准是通用提升策略。
3. **Leave-one-out 校准避免自引用偏差**：影子模型不应在自己参与拟合的分布上评估自己，此设计对 membership inference 类问题同样适用。
4. **黑盒信号（token agreement）同样有效**：即使没有 next-token probabilities，仅凭贪婪输出即可检测蒸馏，降低了审计门槛。

## 关键术语表
- **Distillation Inference**：判断嫌疑模型是否从已知教师模型蒸馏而来的假设检验任务。
- **Shadow Model**：与嫌疑模型同结构、同训练集但监督目标不同的模拟模型，用于估计不可观测的行为分布。
- **Chain-of-Thought (CoT)**：模型输出中间推理步骤的 token 序列，蒸馏的核心内容来源。
- **Log-Likelihood Ratio (LLR)**：嫌疑模型在两个假设分布下的得分比值取对数，作为检验统计量的基础。
- **LiRA（Likelihood Ratio Attack）**：基于影子模型的 membership inference 攻击，本文将其思想迁移到蒸馏检测。
- **p-value (校准)**：在 H₀ 下观测到 ≥ 嫌疑模型统计量的概率，反映执法决策的可信度。
- **Leave-one-out Calibration**：每个影子模型用其余影子拟合分布，避免评估时用自己参与定义的分布。
- **Paraphrase Evasion**：对教师 CoT 做语义保持改写以混淆蒸馏检测的规避策略。

## 可复现要素
- **数据集**：CoT-Collection [5]（论文提及，需确认是否公开）。
- **代码**：论文未提及开源代码。
- **权重**：Qwen2.5-7B-Instruct 和 Llama-3.2-3B 均为开源模型。
- **关键超参**：LoRA rank=8, scale=16, dropout=0.05, 3 epochs, M=50 影子/类, N=500 审计实例。
