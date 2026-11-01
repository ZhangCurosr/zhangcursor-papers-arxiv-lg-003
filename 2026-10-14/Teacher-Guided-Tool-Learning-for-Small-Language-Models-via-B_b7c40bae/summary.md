---
title: "Teacher-Guided-Tool-Learning-for-Small-Language-Models-via-B"
source: https://arxiv.org/pdf/2610.18459v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:10:29"
field: "小型语言模型工具使用增强"
keywords: ["小语言模型", "工具学习", "强化学习", "知识蒸馏", "双向KL", "教师指导"]
innovations: ["选择性反向KL教师指导与持续前向KL演示监督相结合的双向KL正则化框架", "基于任务表现阈值的 token 级 advantage 修正机制，区分错误偏好并精细调整"]
benchmarks: ["BFCL-v4", "API-Bank", "GSM8K", "HotpotQA"]
---

# 论文速读：Teacher-Guided-Tool-Learning-for-Small-Language-Models-via-B

## 一句话总结
论文提出选择性教师指导策略优化（SeTPO）方法，通过**双向KL约束**（反向KL用于低奖励轨迹的 token 级教师反馈、前向KL用于持续演示监督）协调小语言模型的自主探索与教师指导，显著提升其工具使用能力。

## 研究问题与动机
1. **SLMs自主RL探索受限**：小语言模型通过强化学习独立发现有效工具使用行为的成功率低，难以获得足够的优质训练信号。
2. **SFT演示分布不匹配**：监督微调基于教师演示训练，但推理时学生自生成轨迹与演示分布存在偏差（covariate shift）。
3. **OPD可能传播错误偏好**：在线蒸馏虽提供 token 级反馈，但教师参考答案可能错误，直接模仿会传递错误知识。
4. **师生能力差异阻碍知识迁移**：简单复制教师偏好无法保证有效学习，需要结合学生自主探索与选择性教师干预。

## 核心贡献（创新点）
1. **提出SeTPO框架**：整合选择性反向KL教师指导与持续前向KL演示监督，同时支持自主探索与知识保留。
2. **反向KL token 级精细反馈**：仅在低奖励轨迹上应用教师评分，通过 log-probability 差异调整 advantage，实现决策级修正而非轨迹级。
3. **前向KL持续结构保持**：辅助SFT全程激活，确保工具调用格式和响应结构不在学习过程中退化。
4. **理论保证双向KL优化性质**：证明最优策略期望奖励不低于教师，且保留教师概率质量的确定下界（$\frac{b}{b+\Delta_R}$）。

## 方法详解
**整体架构**：SeTPO = GRPO任务奖励优化 + 选择性反向KL教师指导 + 辅助前向KL演示监督

**1. 选择性干预机制**：
- 定义干预指示器 $\mathbb{I}_j = \mathbb{I}[R_j < \gamma]$，仅对低于阈值 $\gamma$ 的低奖励轨迹启用教师指导
- 同一 prompt 的不同采样轨迹可获差异化指导

**2. 反向KL教师反馈**：
- 在固定学生前缀 $h_{j,t} = (x, y_{j,<t})$ 下，计算 token 级 log-probability 差异：
  $d_{j,t} = \log \pi_{\theta_{old}}(y_{j,t}|h_{j,t}) - \log \pi_T(y_{j,t}|h_{j,t})$
- 该期望等于反向KL散度 $D_{KL}(p_{old}||p_T)$
- 修正后的 advantage：$A'_{j,t} = A_j - \mathbb{I}_j \beta d_{j,t}$
- 教师概率更高的 token 获得更大 advantage，反之降低

**3. 前向KL演示监督**：
- 冻结教师为每个 prompt 生成参考轨迹 $\tau^T$，构建 $\mathcal{D}_{distill}$
- 最小化交叉熵损失：$\mathcal{L}_{SFT} = -\mathbb{E}[\sum_t \log \pi_\theta(\tau^T_t|x, \tau^T_{<t})]$
- 等价于最小化 $D_{KL}(q_\mathcal{D}||\pi_\theta)$，保持教师行为覆盖

**4. 总目标函数**：
$\mathcal{L}_{total} = \mathcal{L}_{SeTPO} + \lambda \mathcal{L}_{SFT}$

**5. 理论分析**（定理1）：
- 在双向KL正则化目标下，最优策略满足 $\mathbb{E}_{p^*}[R] \geq \mathbb{E}_q[R]$
- 对任意响应集合 $\mathcal{E}$：$p^*(\mathcal{E}) \geq \frac{b}{b+\Delta_R} q(\mathcal{E})$

## 实验与结果
**数据集**：BFCL-v4（工具使用综合基准）、API-Bank（API调用基准）、GSM8K、HotpotQA（通用推理）

**模型配置**：Qwen2.5-Instruct 学生（0.5B/1.5B/3B）vs 教师（1.5B/7B/14B）

**主要结果**（Table 1）：
- **0.5B学生**：BFCL-v4 达18.99%，较GRPO提升8.32pp，较OPD提升2.82pp
- **1.5B学生**：BFCL-v4 达24.13%，较OPD提升1.96pp；API-Bank 达62.14%
- **3B学生**：BFCL-v4 达25.39%，API-Bank 达63.82%（**超过14B教师62.48%**）

**关键发现**：
- SeTPO在所有学生规模上均取得最优性能
- 选择性指导优于 always-on（18.99 vs 16.51 on BFCL-v4）和 random intervention（13.29）
- 1.5B教师对0.5B学生效果最佳，说明师生容量差距需平衡
- 通用推理能力同步提升（GSM8K: 38.51%, HotpotQA: 35.0%）

**消融实验**（Table 4）：
- 移除 token 级反馈（改用轨迹级）：BFCL-v4 下降11.20pp，API-Bank 下降13.56pp
- 移除在线教师：BFCL-v4 下降8.21pp
- 移除辅助SFT：BFCL-v4 下降2.94pp，API-Bank 下降8.71pp

**超参数敏感性**（Table 7）：
- $\gamma=3.99$（接近最大奖励4）效果最佳
- $\beta=1.0$ 适中，过大（2.0）会削弱任务驱动更新
- $\lambda=0.1$ 适中，无SFT损失导致API-Bank大幅下降

## 相关工作脉络
1. **On-Policy Distillation (OPD)**：Gu et al. (2024) 提出的在线蒸馏方法，直接在 student-generated prefixes 上进行 teacher feedback；本文区别在于 OPD 对全轨迹统一应用指导且可能传播错误，SeTPO 选择性干预并区分正确/错误偏好。

2. **Group Relative Policy Optimization (GRPO)**：Shao et al. (2024) 的组相对策略优化；本文在其基础上引入教师反向KL修正 advantage，弥补低奖励轨迹信号不足。

3. **ToolRL**：Qian et al. (2025) 的 ToolRL 框架；本文在相同 reward 设定下，补充教师指导信号以提升小模型探索效率。

4. **Reward/Failure-based Distillation**：如 DyME (Liu et al., 2026) 和 RSTG (Li et al., 2026a) 在 group 失败时触发指导；本文机制更细粒度（token-level）且结合持续性演示监督。

5. **Knowledge Distillation for SLMs**：MiniLLM (Gu et al., 2024) 等离线蒸馏方法；本文强调 on-policy 反馈与 demonstration 监督的双向KL互补。

## 局限性与未来方向
1. **模型范围限制**：仅测试 Qwen2.5 系列（0.5B-3B），其他模型家族泛化性待验证。
2. **工具域局限**：仅在 BFCL-v4 和 API-Bank 两个基准评估，未测试长 horizon 多轮交互任务。
3. **开放域扩展困难**：当前 reward 依赖确定性 checker（需 reference call），无法直接应用于无标准答案的开放环境。
4. **计算开销**：在线反馈需访问冻结教师的 token 概率，teacher scoring 和 reference generation 增加训练成本。
5. **师生配对固定**：教师规模影响效果（1.5B教师最优），需进一步探索动态教师选择策略。

## 研究启发与可借鉴点
1. **双向KL正则化设计范式**：反向KL（push学生远离错误偏好）+ 前向KL（保持教师行为覆盖）的互补组合，可推广至其他 distillation-RL 融合场景。
2. **选择性干预机制**：基于任务表现阈值动态切换指导策略，避免过度约束高表现轨迹，对 agent 训练有参考价值。
3. **Token 级 advantage 修正**：将教师 feedback 细化到 token 层次而非 trajectory 层次，提供更精细的学习信号。
4. **冷启动SFT+持续辅助监督**：先用 expert data 初始化，再在RL全程保留 demonstration supervision，平衡 exploration 与结构保持。
5. **理论下界分析思路**：通过双向KL目标证明 reward improvement 与 behavior preservation 的 trade-off 边界，为方法可信性提供保障。

## 关键术语表
**SeTPO**：Selective Teacher-guided Policy Optimization，本文提出的选择性教师指导策略优化方法。
**反向KL（Reverse KL）**：$D_{KL}(p_{old}||p_T)$，衡量学生分布与教师分布的差异，用于 token 级 advantage 修正。
**前向KL（Forward KL）**：$D_{KL}(q_\mathcal{D}||\pi_\theta)$，衡量演示分布与学生分布的差异，用于保持教师行为覆盖。
**GRPO**：Group Relative Policy Optimization，Shao et al. (2024) 提出的组相对策略优化算法。
**OPD**：On-Policy Distillation，Agarwal et al. (2024) 提出的在线蒸馏方法。
**BFCL-v4**：Berkeley Function Calling Leaderboard v4，评估 LLM 工具使用能力的综合基准。
**API-Bank**：Li et al. (2023) 提出的 API 调用能力评估基准。
**Composite Checker**：确定性评价器，结合 format reward 和 correctness reward 计算轨迹总分。

## 可复现要素
- **数据集**：ToolRL dataset（400 cold-start + 4000 training），BFCL-v4，API-Bank，GSM8K，HotpotQA — **论文未提及公开状态**
- **代码/权重**：论文未提及开源，仅声明使用 Qwen2.5-Instruct 官方模型
- **关键超参**：
  - LoRA rank r=32, alpha α=64
  - Learning rate $2\times10^{-5}$, Cosine scheduler
  - GRPO group size G=4, RL epochs=15
  - Intervention threshold γ=3.99, teacher weight β=1.0, SFT weight λ=0.1
  - Max prompt length=2048, max response length=1024
  - Hardware: 1-2× NVIDIA L40S/RTX PRO 6000 Blackwell GPUs
