---
title: "WHEN-KL-REGULARIZATION-MISFIRES-IN-GROUPPOLICY-OPTIMIZATION"
source: https://arxiv.org/pdf/2610.12161v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:03:55"
field: "大语言模型强化学习"
keywords: ["group policy optimization", "KL regularization", "RLVR", "policy gradient", "conditional KL", "zero-sum calibration", "reward clipping"]
innovations: ["系统分析 KL-奖励交互的七种失配模式", "提出 ZCPO：利用条件 KL 漂移进行组内零和系数校准", "最大绝对奖励归一化替代标准差归一化以提升稳定性"]
benchmarks: ["AIME 2024", "AIME 2025", "DocMath", "LongBench-V2", "GLM-4-9B math"]
---

# 论文速读：WHEN-KL-REGULARIZATION-MISFIRES-IN-GROUPPOLICY-OPTIMIZATION

## 一句话总结
本文分析了独立参考策略 KL 正则化与组相对奖励更新之间的七种失配模式，并提出 ZCPO（Zero-Sum Calibrated Policy Optimization），利用条件 KL 漂移在组内校准联合系数，有效缓解残余 KL 梯度、长度偏差和采样噪声等问题。

## 研究问题与动机
1. **核心观察**：移除参考策略 KL 正则化在某些组相对策略优化方法中反而提升了训练效果（如 PAPO 在 Qwen2.5-VL 上将 avg@8 从 47.92% 提升至 50.18%；Open-Reasoner-Zero 也报告无 KL 的 PPO 优于加 KL 的变体）。
2. **设计困惑**：既然移除 KL 可能有益，那么参考策略信息应如何更合理地嵌入组相对优化中，而非简单作为独立损失项叠加？
3. **现有方法不足**：DAPO、GMPO 等近期工作省略了显式参考 KL 惩罚，但未系统分析 KL 与奖励裁剪、组内梯度抵消等机制的交互失效模式。
4. **理论缺口**：此前 KL 估计器分析多聚焦梯度偏差恢复，未深入讨论独立 KL 与奖励 clipping / 组过滤之间的耦合失效问题。

## 核心贡献（创新点）
1. **系统归纳 KL 与 GRPO 奖励更新的七种潜在失配模式**（F1–F7），刻画了奖励梯度消失后的残余 KL 更新、KL 长度依赖、信号集中抑制以及 k1 采样噪声等机制性假说。
2. **提出 ZCPO（Zero-Sum Calibrated Policy Optimization）**，将条件 KL 漂移作为组内零和校准系数，与基础代理目标结合；相较于独立 KL 损失，ZCPO 使参考信息进入优化的方式从"逐 token 独立惩罚"变为"组内相对校准"。
3. **从凸优化角度推导联合系数的最优性**：在固定输入与组内零和约束下，ZCPO 系数是唯一最优解，实现了贴近奖励系数与偏好低漂移之间的最优权衡，且具有组内公共偏移不变性。
4. **多维度实验验证**：在 Qwen2.5-32B 数学推理任务上，ZCPO 在 AIME24 上达到 56.8%（avg@32），显著优于 DAPO+标准KL、RPG、GVPO 和 GOPO 等基线；长上下文与跨模型（GLM-4-9B）实验支持方法的泛化性。

## 方法详解
- **基础设定**：对于同一 prompt $q$，采样策略 $\pi_{\text{old}}$ 生成 $G \geq 2$ 条响应，定义组内奖励均值 $\bar{R}$、标准差 $\sigma_q$ 和最大绝对中心化奖励 $s_q = \max_i |R_i - \bar{R}|$。
- **组门控**：$B_q = \mathbf{1}\{G \geq 2\} \cdot \mathbf{1}\{s_q > 0\}$，跳过奖励全等的组（对应 F3）。
- **条件 KL 计算**：每个 token 位置的条件 KL 为 $\kappa_{i,t} = \text{KL}(\pi_\theta(\cdot|h_{i,t}) \| \pi_{\text{ref}}(\cdot|h_{i,t}))$，通过完整词汇表分布求和得到。
- **长度归一化与组内中心化**：$\widetilde{D}_i = D_i / T_i$（平均条件 KL），$K_i = \widetilde{D}_i - \frac{1}{G}\sum_j \widetilde{D}_j$（组内去中心化漂移）。
- **联合系数构造**：$C_i = \frac{A_i}{s_q} - \beta K_i$，其中 $A_i = R_i - \bar{R}$，除以 $s_q$ 而非 $\sigma_q$ 以固定最强奖励信号幅度。
- **损失函数**：$\mathcal{L}_q = -\sum_i \sum_t w_{i,t} \, S_{\text{base}}(r_{i,t}; \text{sg}(C_i))$，系数 $C_i$ 在当前反向传播中被 stop-gradient 冻结，梯度仅通过概率比 $r_{i,t}$ 传播。
- **最优性证明**（Appendix D）：最小化 $\min_{\mathbf{c}:\sum c_i=0} \{\frac{1}{2}\sum_i(c_i - A_i/s_q)^2 + \beta \sum_i c_i \widetilde{D}_i\}$，通过 Lagrange 乘子法得唯一最优解 $C_i$，Hessian 为单位矩阵保证严格凸性。
- **公共偏移不变性**：对所有 $\widetilde{D}_i$ 加相同常数不改变 $K_i$ 和 $C_i$，因此共享漂移分量不影响系数。

## 实验与结果
- **主实验**：Qwen2.5-32B，训练数据 DAPO-Math-17K（~17K 整数答案 prompt），评估 AIME24 和 AIME25（avg@32）。
- **ZCPO vs 基线**（AIME24 / AIME25）：
  - DAPO without KL（β=0）：47.3% / 37.1%
  - DAPO + 标准 KL（β=0.001）：44.5% / 34.7%
  - DAPO + 标准 KL（β=0.04）：31.2% / 18.9%（过高 β 严重损害性能）
  - **ZCPO（β=4）：56.8% / 48.5%**（最强结果，相对 DAPO without KL 提升 +9.5pp / +11.4pp）
  - RPG：50.7% / 42.5%；GVPO：52.6% / 41.1%；GOPO：48.5% / 39.9%
- **消融实验**：
  - 无 KL 中心化（$K_i \leftarrow D_i/T_i$）：45.6% / 36.7%（最大降幅，破坏零和性质）
  - 无 KL 长度归一化：53.9% / 43.5%
  - 用 k1 替代条件 KL：54.5% / 47.1%
  - 标准差奖励归一化：55.4% / 44.9%
- **长上下文实验**（Qwen3-4B，GoLongRL 子集，6 个基准）：GRPO-ZCPO 在 DocMath（63.8）、LBV2（48.2）、Frames（68.2）、MRCR（66.0）、CorpusQA（69.6）等上全面优于 GRPO+标准KL 和 GRPO without KL。
- **跨模型实验**（GLM-4-9B-0414）：DAPO without KL 24.5% → ZCPO 36.2%（+11.7pp），趋势一致。
- **梯度诊断**：ZCPO（β=4）的 KL/总梯度范数中位比为 7.0%，而 vanilla（β=0.04）为 5.8%，但前者任务性能显著更好；ZCPO 在 β=20 时仍保持 30.1% 的 KL 占比并维持强性能。
- **F1 控制**：在 clipping 位置禁用 KL 使 AIME24 达 46.1%，高于 vanilla（β=0.001）的 44.5%。

## 相关工作脉络
1. **DAPO**（Yu et al., 2025）：引入 clip-higher、动态采样和 token-level 聚合，省略显式 KL 惩罚；ZCPO 在此基础上通过条件 KL 校准重新引入参考信息，而非完全移除。
2. **GMPO**（Zhao et al., 2026）：使用几何均值聚合；ZCPO 关注 KL 与奖励的交互失效，而非聚合策略本身。
3. **GOPO**（Zixian, 2026）：用二次概率比惩罚替代显式 KL；ZCPO 保留 KL 参考信息但以组内相对校准形式融入。
4. **GVPO / RPG**：使用随训练更新的旧策略作为参考；本文指出这无法消除条件采样波动，且组内中心化不能保证消除所有噪声。
5. **Amini et al. (2025)**：使用 per-prefix 条件 KL 降低估计方差；但未研究与组内中心化、长度归一化和奖励激活门控相结合的联合系数设计。
6. **Vassoyan et al. (2025)**：按参考策略熵加权 per-token KL 以放松高不确定性位置的约束；但未解决参考策略过于自信的场景。

## 局限性与未来方向
1. **条件 KL 计算的数值精度**：在 bf16 训练下，当当前策略与参考策略接近时，条件 KL 是分布扰动的二阶量，易受舍入误差影响；论文通过在 fp32 下重算输出层投影并做全词汇表 KL 约减来缓解，但带来约 0.5% 额外 GPU 计算开销。
2. **长度效应未完全消除**：归一化后的 $K_i$ 仍与响应长度 $T_i$ 相关，意味着长度相关的偏差并未被彻底消除。
3. **评估范围局限**：主要实验集中在数学推理任务，虽然附录中有长上下文和跨模型验证，但其他任务类型（如对话、代码生成）的泛化性尚待验证。
4. **ZCPO 系数最优性仅针对固定输入**：实际训练中系数随参数更新而变化，最优性结果不直接保证每步训练都改善奖励或降低 KL。

## 研究启发与可借鉴点
1. **组内相对校准思想可迁移**：将参考信息从"绝对惩罚"转为"组内相对比较系数"的思路，可推广到其他依赖参考策略的 RLHF/RLVR 方法中，尤其是当独立正则化项与奖励信号冲突时。
2. **停止梯度冻结系数 + 联合代理目标**：先计算校准系数并 stop-gradient，再将其嵌入基础 surrogate 的设计模式简洁且计算高效，可作为通用模块接入不同基线算法。
3. **最大绝对奖励归一化 vs 标准差归一化**：论文发现除以 $s_q = \max_i|A_i|$ 比除以 $\sigma_q$ 更优，因为它固定了最强奖励信号幅度，避免其随组内分布波动——这一归一化策略可在其他组相对方法中验证。
4. **七种失配模式的系统性分析框架**：F1–F7 的分类方法可作为诊断 KL-奖励交互问题的通用分析工具，帮助研究者在新方法设计中提前识别潜在失效点。
5. **与团队方向的结合机会**：若团队关注长上下文 RLVR 或低资源对齐，ZCPO 的条件 KL 校准和组门控设计可直接集成到 DAPO/GMPO 等基线中，有望在保持训练稳定性的同时提升奖励信号利用率。

## 关键术语表
- **Group-relative policy optimization（组相对策略优化）**：通过组内奖励比较进行策略更新的方法，无需值函数，代表作为 GRPO 及其变体。
- **Conditional KL（条件 KL）**：在给定前缀上下文 $h_{i,t}$ 下，当前策略与参考策略在完整词汇表上的 KL 散度，消除了单 token 采样的随机性。
- **Zero-sum calibrated coefficient（零和校准系数）**：在组内加和为零的联合更新系数 $C_i$，由奖励项与 KL 漂移项线性组合而成。
- **KL–reward mismatch（KL-奖励失配）**：独立 KL 正则化项与组相对奖励更新在不同机制下（如 clipping、梯度抵消）产生的非同步行为。
- **Reward clipping gate（奖励裁剪门）**：基于概率比是否超出 clipping 边界决定奖励梯度是否为零的二值门控函数 $m_{\text{base}}(r, \hat{A})$。
- **k1 / k3 KL**：k1 为基于采样 token 的对数概率比（含采样噪声），k3 为基于完整分布的条件 KL（无采样噪声）。
- **Group gating（组门控）**：跳过组内所有奖励相同的 prompt 组（$s_q = 0$），避免无意义的 KL 驱动更新。
- **Common-shift invariance（公共偏移不变性）**：组内所有响应的平均条件 KL 增加相同常数时，中心化漂移 $K_i$ 和联合系数 $C_i$ 保持不变的性质。

## 可复现要素
- **数据集**：DAPO-Math-17K（~17K prompts with integer answers）用于主实验；GoLongRL 子集（8,000 条）用于长上下文实验；AIME 2024/2025 用于评估。数据集开源性论文未明确声明。
- **代码**：论文声明"supplementary material provides code that can be run directly to reproduce the results"，附录提供可复现代码。
- **关键超参**：β=4（ZCPO 主实验）、β=0.001 和 β=0.04（vanilla KL 对比）、clip_low=0.2、clip_high=0.28、lr=1e-6、G=16 responses/prompt、max_length=20,480 tokens。
