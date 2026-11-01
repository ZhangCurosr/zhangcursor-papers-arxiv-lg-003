---
title: "Teacher-Guided-Tool-Learning-for-Small-Language-Models-via-B"
source: https://arxiv.org/pdf/2610.18459v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:10:32"
field: "小语言模型工具使用学习"
keywords: ["Tool Learning", "Small Language Models", "Reinforcement Learning", "Knowledge Distillation", "KL Divergence", "Policy Optimization"]
innovations: ["提出 SeTPO 框架，结合选择性反向 KL 教师指导与持续前向 KL 演示监督", "在低奖励轨迹上引入 token 级教师反馈修正 GRPO 优势", "理论证明双向 KL 正则下期望奖励不低于教师且保留教师行为概率下界"]
benchmarks: ["BFCL-v4", "API-Bank", "GSM8K", "HotpotQA"]
---

# 论文速读：Teacher-Guided Tool Learning for Small Language Models via Bidirectional KL Constraints

## 一句话总结
论文提出 **SeTPO（Selective Teacher-guided Policy Optimization）**，面向小语言模型（SLMs）的工具使用学习问题。该方法通过**选择性反向KL教师指导**（针对低奖励探索轨迹）与**持续前向KL演示监督**（辅助完整教师示范）相结合，使SLMs在自主探索不足时获得有效的外部纠正信号，同时在 RL 训练全程保持教师支持的结构化行为分布。

## 研究问题与动机
1. **SLMs 工具学习的“能力瓶颈”**：外部工具可扩展 SLMs 的能力，但受限于模型本身能力，仅靠强化学习（RL）难以发现有效的工具使用行为。
2. **纯监督微调（SFT）的分布错配**：基于演示的 SFT 训练分布与推理时的自生成分布不一致，限制了迁移效果。
3. **纯强化学习（RL）探索效率低**：SLMs 探索成功率低，有效经验稀缺，导致 RL 信号稀疏，难以指导工具选择与参数构造等关键决策。
4. **在线蒸馏（OPD）的错误传播风险**：直接使用教师反馈可能将教师自身的错误偏好传播给学生，且未考虑师生能力差距导致的行为分布不匹配。

## 核心贡献（创新点）
1. **提出 SeTPO 框架**：将“选择性教师指导”与“持续演示监督”统一于双向 KL 正则目标中，分别对应反向 KL（改进学生尝试）与前向 KL（保留教师行为）。
2. **选择性反向 KL 教师反馈**：仅在低奖励轨迹上引入 token 级反向 KL 差异作为优势修正，避免对已成功尝试的过度干扰。
3. **理论保证**：在理想化双向 KL 目标下证明，最优策略期望奖励不低于教师，且对学生策略保留教师概率质量的保证下界。
4. **实证全面领先**：在 BFCL-v4 和 API-Bank 上，跨三种学生规模（0.5B、1.5B、3B）均取得优于 SFT、GRPO、KD、OPD 等基线的工具使用准确率。
5. **错误分析与泛化验证**：揭示 SeTPO 显著减少工具选择与参数构造错误，并在 GSM8K、HotpotQA 等通用推理任务上保持或提升性能，说明所学技能可迁移。

## 方法详解
**整体架构**：SeTPO 同时维护两个学习信号——在线任务奖励驱动的 GRPO 探索 + 选择性教师反向 KL 指导，以及离线教师完整演示的前向 KL 监督（辅助 SFT 损失）。

**1. 选择性干预指示器**
$$
\mathbb{I}_j = \mathbb{I}[R_j < \gamma]
$$
仅对奖励低于阈值 $\gamma$ 的轨迹触发教师打分，高奖励轨迹保持纯任务驱动更新。

**2. Token 级反向 KL 反馈**
在固定学生前缀 $h_{j,t}$ 下，计算行为策略 $\pi_{\theta_{old}}$ 与冻结教师 $\pi_T$ 的概率对数差异：
$$
d_{j,t} = \log p_{old}(y_{j,t}) - \log p_T(y_{j,t})
$$
该差异是反向 KL $D_{KL}(p_{old} \| p_T)$ 的单样本估计。

**3. 优势函数修正**
$$
A'_{j,t} = A_j - \mathbb{I}_j \beta d_{j,t}
$$
教师分配更高概率的 token 获得更大优势提升，反之降低。随后使用 clip 策略目标更新：
$$
\mathcal{L}_{SeTPO}(\theta) = -\mathbb{E}_{j,t}[\min(\rho_{j,t}(\theta) A'_{j,t},\; \bar{\rho}_{j,t}(\theta) A'_{j,t})]
$$

**4. 辅助前向 KL 监督**
训练前由冻结教师为每个 prompt 生成一条参考轨迹，构成演示数据集 $\mathcal{D}_{distill}$。全程最小化：
$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,\tau^T)\sim\mathcal{D}_{distill}}\left[\sum_t \log \pi_\theta(\tau^T_t \mid x, \tau^T_{<t})\right]
$$
等价于最小化前向 KL $D_{KL}(q_\mathcal{D}(\cdot|x) \| \pi_\theta(\cdot|x))$，用于维持教师支持的结构化响应格式。

**5. 总损失**
$$
\mathcal{L}_{total}(\theta) = \mathcal{L}_{SeTPO}(\theta) + \lambda \mathcal{L}_{SFT}(\theta)
$$

**6. 理论分析（理想化双向 KL 目标）**
假设目标 $\mathcal{F}(p)=\mathbb{E}_{y\sim p}[R(y)] - a D_{KL}(p\|q) - b D_{KL}(q\|p)$，可证：
- 最优策略期望奖励 $\geq$ 教师期望奖励。
- 对任意响应集 $\mathcal{E}$，$p^*(\mathcal{E}) \geq \frac{b}{b+\Delta_R} q(\mathcal{E})$，即保留教师概率质量的下界。

## 实验与结果
**数据集与模型**：
- 工具使用基准：**BFCL-v4**、**API-Bank**
- 通用推理基准：GSM8K、HotpotQA
- 学生模型：Qwen2.5-Instruct 0.5B、1.5B、3B
- 教师模型：对应冻结的 1.5B、7B、14B Qwen2.5-Instruct

**主要结果（Table 1）**：
- **0.5B 学生**：SeTPO 在 BFCL-v4 达到 **18.99**，较 GRPO（10.67）提升 **+8.32pp**，较 OPD（16.17）提升 **+2.82pp**；在 API-Bank 达到 **46.06**，优于所有基线。
- **1.5B 学生**：BFCL-v4 **24.13**，API-Bank **62.14**。
- **3B 学生**：BFCL-v4 **25.39**，API-Bank **63.82**，**超越其 14B 教师**（62.48）。

**干预策略对比（Table 2）**：SeTPO（选择性低奖励干预）在 BFCL-v4（18.99）和 API-Bank（46.06）上均优于始终干预（Always-on）和随机干预（Random）。

**消融实验（Table 4）**：
- 移除 token 级信号改用轨迹级信号：BFCL-v4 下降 **-11.20pp**。
- 移除辅助 SFT 损失：BFCL-v4 下降 **-2.94pp**，API-Bank 下降 **-8.71pp**。
- 移除在线教师：BFCL-v4 下降 **-8.21pp**，API-Bank 下降 **-6.70pp**。

**错误分析（Figure 5）**：SeTPO 显著减少工具选择、参数键、参数值/类型三类错误，尤其在参数值/类型上纠正最多。

**泛化能力（Table 3）**：在 GSM8K 和 HotpotQA 上，三个学生尺度中 SeTPO 均取得最高或接近最高的通用推理准确率，说明工具学习训练未损害基础推理能力。

## 相关工作脉络
1. **SFT 工具学习**：如 ToolACE、Hammer、xLAM 等利用合成或专家数据微调 SLMs。本文与之区别在于：不只依赖静态演示，而是通过 RL 探索 + 选择性教师反馈动态改进。
2. **GRPO 类 RL 方法**：ToolRL 等工作纯用任务奖励优化工具调用格式与功能正确性。本文补充：纯 RL 探索效率低时，引入教师指导填补信号空白。
3. **离线知识蒸馏（KD）**：直接从教师轨迹做 SFT，忽略学生生成历史的前缀分布差异。本文强调：需在 student-generated prefixes 下评估教师反馈。
4. **在线策略蒸馏（OPD）**：对自生成轨迹进行 token 级教师打分。本文与 OPD 的关键差异：① 仅在低奖励轨迹上应用（选择性）；② 结合持续的前向 KL 演示监督以维持响应结构；③ 理论给出双向 KL 正则下的奖励与行为保留保证。
5. **失败重放/自适应引导**：如 Distill where you fail、DYME、RSTG 等按失败情况触发教师介入。本文与之相似但更形式化：用显式 reward gate + 反向 KL 差异定义干预机制，并给出理论下界。

## 局限性与未来方向
1. **实验范围有限**：仅验证 Qwen2.5 系列 0.5B–3B 模型，在其他模型架构（如 LLaMA、Mistral）和更大尺度（≥7B）上的适用性待检验。
2. **工具域与任务长度**：当前基准侧重单轮/短多轮函数调用，对长视界交互式工具链（long-horizon agentic tasks）或开放域工具的使用泛化能力未充分评估。
3. **奖励设计依赖确定性检查器**：需要参考调用（reference calls）计算精确格式与参数正确性；难以直接扩展到无标准答案的开放式工具环境。
4. **计算开销**：在线教师打分需访问冻结教师的 token 概率，增加训练时间与显存负担；教师生成交互轨迹亦有额外成本。
5. **未来方向**：可扩展至无参考答案的开放工具环境（设计无需 ground-truth 的奖励函数）；探索不同师生规模匹配策略；研究动态阈值 $\gamma$ 与权重 $\beta,\lambda$ 的自动调节机制。

## 研究启发与可借鉴点
1. **选择性干预思想**：并非所有轨迹都需要教师指导；可根据任务奖励表现动态区分“需要纠正”与“已足够好”的尝试，适用于其他 RL+蒸馏的混合训练场景。
2. **双向 KL 正则的理论设计**：反向 KL 修正学生分布、前向 KL 保留教师支持行为，二者结合可提供理论上的奖励非退化与行为保留保证，该方法论可迁移至其他 agent 训练任务。
3. **Token 级优势修正**：将教师对单个 token 的概率差异融入 GRPO 优势函数，而非单纯替换整个轨迹标签，提供了比传统 OPD 更细粒度的纠正信号。
4. **辅助 SFT 的持续作用**：在 RL 过程中始终保持对完整教师演示的监督，有助于稳定输出格式，防止纯 RL 导致的结构性崩塌；这一技巧可用于需要严格格式的工具调用、代码生成等任务。
5. **理论 + 实证并重**：本文给出理想化目标下的严格定理，同时展示真实实验中的显著提升，为后续工作提供了“可证明性质 + 实用框架”的研究范式参考。

## 关键术语表
**SeTPO**：Selective Teacher-guided Policy Optimization，本文提出的结合选择性教师指导与辅助演示监督的混合策略优化方法。

**反向 KL（Reverse KL）**：$D_{KL}(p_{old} \| p_T)$，衡量学生当前策略与教师策略的差异，本文用于在低奖励轨迹上调整 token 级优势。

**前向 KL（Forward KL）**：$D_{KL}(q_\mathcal{D} \| \pi_\theta)$，衡量教师演示分布与学生策略的差异，本文通过辅助 SFT 损失持续最小化以保留结构化行为。

**GRPO**：Group Relative Policy Optimization，一种基于组内相对奖励归一化的强化学习算法，本文用于工具使用任务的策略更新。

**OPD**：On-Policy Distillation，在线策略蒸馏，即对自生成轨迹使用冻结教师进行 token 级监督。

**BFCL-v4**：Berkeley Function Calling Leaderboard v4，评估 LLM 工具调用能力的综合基准。

**API-Bank**：面向 API 调用评估的 benchmark，包含 L1/L2/L3 三级难度，评分依据软匹配计算。

**Intervention Threshold（$\gamma$）**：触发教师指导的奖励阈值，当轨迹奖励低于 $\gamma$ 时启动选择性反向 KL 指导。

## 可复现要素
- **数据集**：ToolRL 数据集（含 400 个冷启动专家示例 + 3920 个训练/80 个验证的 ToolACE/Hammer/xLAM 数据）；评估基准 BFCL-v4、API-Bank、GSM8K、HotpotQA。论文未明确说明原始数据集是否公开，但引用了各源论文。
- **代码**：论文未提供代码链接声明（在提供的文本中未见 GitHub 链接），需在 arXiv 页面或项目主页确认。
- **权重**：使用 Qwen2.5-Instruct 开源模型（0.5B/1.5B/3B/7B/14B）作为基础，教师参数冻结。
- **关键超参**：
  - LoRA rank $r=32$，alpha $\alpha=64$
  - 学习率 $2\times10^{-5}$，Cosine scheduler，3 epochs cold-start SFT
  - GRPO group size $G=4$，epoch 数 15，max steps 105
  - 干预阈值 $\gamma=3.99$，教师指导权重 $\beta=1.0$，辅助 SFT 权重 $\lambda=0.1$
  - 最大 prompt 长度 2048，最大 response 长度 1024
- **硬件**：0.5B/1.5B 实验使用单卡/双卡 NVIDIA L40S，3B 学生+14B 教师使用双卡 NVIDIA RTX PRO 6000 Blackwell。
