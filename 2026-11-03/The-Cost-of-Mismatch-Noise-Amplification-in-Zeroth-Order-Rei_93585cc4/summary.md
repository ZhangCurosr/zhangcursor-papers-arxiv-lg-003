---
title: "The-Cost-of-Mismatch-Noise-Amplification-in-Zeroth-Order-Rei"
source: https://arxiv.org/pdf/2611.03260v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:40:44"
field: "大语言模型高效后训练与零阶优化"
keywords: ["Zeroth-Order Optimization", "Reinforcement Learning with Verifiable Rewards", "Noise Amplification", "BOLT", "LLM Post-Training", "Variance Analysis"]
innovations: ["建立通用方差理论框架，揭示ZO梯度估计可靠性由两次配对评估的噪声相关性结构决定", "证明on-policy RL失配数据导致1/μ²预言机噪声放大，BOLT匹配数据消除该放大", "提出ZO-BOLT算法，在保持近推理显存的同时实现RLVR对齐优化"]
benchmarks: ["GSM8K", "MATH"]
---

# 论文速读：The Cost of Mismatch: Noise Amplification in Zeroth-Order Reinforcement Learning

## 一句话总结
本文从方差理论角度揭示了零阶（ZO）优化在强化学习后训练中的根本性限制：on-policy RL 因数据失配导致 $1/\mu^2$ 预言机噪声放大，构成不可消除的噪声底；而 BOLT 通过匹配数据机制消除了该放大。据此提出 ZO-BOLT，在保持近推理显存的同时实现了可靠的 RLVR 对齐优化。

## 研究问题与动机
- **核心问题**：零阶优化在内存高效的 LLM 后训练中已被证明可行（如 SFT、偏好优化），但其应用于强化学习可验证奖励（RLVR）场景的可靠性尚属空白——什么因素限制了 ZO 优化在 RLVR 中的可靠应用？
- **现有方法不足**：
  - 对 on-policy RL 直接应用 ZO 时，两次函数评估基于不同策略生成的轨迹，数据天然失配，导致噪声方差随扰动尺度 $\mu$ 减小而呈 $1/\mu^2$ 爆炸。
  - 轨迹复用可通过冻结分布代理降低方差，但本质改变了优化目标；重要性加权校正的轨迹级权重二阶矩随序列长度指数增长，在长程推理中极不稳定。
  - PPO/GRPO 等虽采用冻结分布代理使配对评估可复用同一批数据，但仍是以改变优化目标为代价，并未解决结构化失配问题。

## 核心贡献（创新点）
1. **通用方差理论框架**：建立了噪声 ZO 优化的通用分析框架，将估计误差精确分解为无噪声项与预言机噪声项（Lemma 1），揭示了 ZO 梯度估计可靠性由两次配对评估间噪声相关性结构决定，统一刻画了不同数据 regimes 下的噪声行为。
2. **失配与匹配数据 regime 的方差理论刻画**：证明 on-policy RL 在失配数据 regime 下产生 $1/\mu^2$ 预言机噪声放大（Prop. 1），而 BOLT 在匹配数据 regime 下消除该放大、仅保留有界的 mini-batch 方差项（Prop. 2）；两大结果统一为预言机噪声分离定理（Thm. 1），并导出具显式常数的收敛保证（Thm. 2）。
3. **经验验证与 ZO-BOLT 实践**：通过扰动尺度 $\mu$ 扫测和匹配/失配端到端对比实验，实证验证了理论的噪声放大效应；提出 ZO-BOLT 算法，在三个模型变体与两个数学推理基准上验证了其有效性和近推理级显存效率。

## 方法详解
- **方差理论框架**：将含噪 ZO 梯度估计量分解为 $\hat{g} = g_F + \frac{e(\theta+\mu u, \xi_+) - e(\theta, \xi_-)}{\mu}u$，其中 $g_F$ 为无噪声项，第二项为预言机噪声项。关键结论（Lemma 1）：噪声项的二阶矩上界为 $d\nu^2/\mu^2$，其中 $\nu^2$ 控制两次评估噪声差的二阶矩。
- **失配数据 regime（on-policy RL）**：两次评估分别用 $P_\theta$ 和 $P_{\theta+\mu u}$ 采样的独立轨迹，噪声条件独立，故 $e_+$ 与 $e_-$ 不相关，产生 $1/\mu^2$ 放大（Prop. 1: 上界 $2d\sigma_{\mathrm{roll}}^2/\mu^2$）。论文还分析了轨迹复用和重要性加权的结构性限制：前者改变优化目标，后者在长序列下二阶矩指数增长（Cor. 2: $\leq \rho^T$）。
- **匹配数据 regime（BOLT）**：BOLT 目标 $F_{\mathrm{BOLT}}(\theta) = \frac{1}{M}\sum_i \ell_i(\theta)$ 在固定离线数据集上定义，两次评估使用同一 mini-batch $\mathcal{B}$，导致噪声完全相关并在差分操作中相消（Prop. 2: 上界 $(\sqrt{\frac{(d+2)\sigma_{\mathrm{mb}}^2}{B}} + L\mu\sqrt{d(d+2)(d+4)})^2$，无 $1/\mu^2$ 项）。
- **ZO-BOLT 算法**：在 BOLT 固定加权 SFT 目标上使用对称有限差分估计器，结合 ZoAR（查询复用，保留 $N_h$ 历史对）和 R-AdaZO（自适应一阶矩平滑与二阶矩预处理）。具体实现：每次迭代采样 mini-batch $\mathcal{B}_t$ 和随机方向 $\mathbf{u}_t$，在同一 batch 上评估 $F_{\mathcal{B}_t}(\theta_{t-1}+\mu\mathbf{u}_t)$ 与 $F_{\mathcal{B}_t}(\theta_{t-1}-\mu\mathbf{u}_t)$，再通过历史查询复用和自适应更新计算梯度。

## 实验与结果
- **数据集与模型**：GSM8K（小学数学）和 MATH（竞赛级数学）两个推理基准；Qwen2.5-1.5B、Qwen2.5-7B、DeepSeek-R1-Distill-Qwen-1.5B 三个模型；LoRA（$r=8, \alpha=16$）。
- **噪声方差扫测（Table 1）**：$\mu \in \{10^{-4}, 3\times10^{-4}, 10^{-3}, 3\times10^{-3}, 10^{-2}\}$，匹配 BOLT 的归一化预言机噪声方差保持恒定（$\sim 2.1\times10^{-2}$），而失配 BOLT 和 on-policy RL 均按 $1/\mu^2$ 增长（$\mu=10^{-4}$ 时达 $2.77\times10^6$ / $2.44\times10^6$），精确验证理论预测。
- **端到端对比（Table 2）**：ZO-BOLT Matched 最佳准确率达 63.78%（+11.54 over baseline 52.24%），ZO-BOLT Mismatched 反降至 49.35%（-2.89）；ZO-GRPO Matched 达 59.00%，Mismatched 仅 53.08%，匹配优势一致。
- **性能对比（Table 3）**：ZO-BOLT 在全部 6 个设置下均超越预训练基线。GSM8K 上 Qwen2.5-1.5B 达到 63.78%，恢复一阶 BOLT 增益的 98.2%；7B 模型恢复 92.2%；DeepSeek-R1-Distill-Qwen-1.5B 恢复 60.4%。MATH 上恢复比例较低（79.9%/69.2%/21.0%）。
- **显存效率（Table 4）**：ZO-BOLT 在 1.5B 上峰值显存 4,201 MB（较一阶 BOLT 降低 59.5%，较推理仅增加 920 MB）；7B 上 16,014 MB（较一阶 BOLT 降低 43.7%，较推理仅增加 1,114 MB）。

## 相关工作脉络
- **MeZO 与 ZO SFT**（Malladi et al., 2023; Zhang et al., 2024）：建立了 ZO 用于 LLM 微调的可行性，但聚焦固定数据 SFT，未涉及 on-policy RL。本文补充了 ZO 在 RLVR 下的理论边界分析。
- **BOLT**（Shu et al., 2026）：本文依托的RLVR对齐目标。BOLT 将 KL 正则化 RLVR 目标重写为离线加权 SFT，实现数据生成与策略优化解耦；本文从 ZO 视角揭示了其匹配数据结构对噪声放大的根本性消除作用。
- **PPO/GRPO**（Schulman et al., 2017; Shao et al., 2024）：冻结分布代理方法，本质上改变了优化目标以换取匹配数据机会；本文明确区分了"改变目标"与"结构性解决失配"的不同。
- **ZoAR / R-AdaZO**（Qiu et al., 2025; Shu et al., 2025）：ZO-BOLT 采用的查询复用与自适应优化组件，提升了单点 ZO 估计的采样效率。
- **稀疏 ZO / 子空间 ZO**（Liu et al., 2025; Yu et al., 2025; Chen & Shu, 2026）：低维扰动与稀疏更新路线，与本文从数据 regime 角度出发的正交思路互补。

## 局限性与未来方向
- 实验仅限于数学推理任务和 7B 及以下模型规模，未覆盖更长序列或更复杂任务域。
- ZO-BOLT 与一阶 BOLT 的性能差距在更难的 MATH 基准上更为显著（如 DeepSeek-R1-Distill-Qwen-1.5B 上仅恢复 21.0%），说明当前 ZO 策略对高难度任务优化效率仍有提升空间。
- 未来方向包括：更广泛的任务覆盖、更大模型规模（>7B）、改进 ZO 优化策略以缩小与一阶方法的性能差距。

## 研究启发与可借鉴点
- **理论分析的普适性**：方差分解框架（Lemma 1）可作为分析任意含噪 ZO 估计器的通用工具，适用于其他在线/离策略学习场景的噪声传播分析。
- **数据 regime 作为设计原则**：匹配 vs. 失配数据的分析揭示了算法设计的根本约束——在需要减少噪声的场景中，优先考虑数据复用/结构化匹配而非单纯增大 batch size 或调整 $\mu$。
- **实验验证范式**：扰动尺度 $\mu$ 扫测 + 匹配/失配对照的两层验证策略（先验证估计器层面的噪声分离，再验证端到端优化性能），为后续工作提供了严谨的理论-实验对接范例。
- **ZO-BOLT 的模块化设计**：ZoAR 查询复用与 R-AdaZO 自适应预处理的组合可直接迁移到其他基于固定数据的 ZO 后训练场景（如 RLHF 偏好优化）。

## 关键术语表
**Zeroth-Order (ZO) Optimization**：零阶优化，仅通过前向函数评估而非反向传播梯度来估计梯度方向并进行参数更新的优化范式，显著降低训练显存。
**Oracle-Noise Amplification**：预言机噪声放大，指 ZO 有限差分估计中因两次函数评估引入的随机噪声被除以小尺度 $\mu$ 而导致的方差 $1/\mu^2$ 放大效应。
**Matched-Data Regime**：匹配数据 regime，指 ZO 优化的两次配对函数评估在相同数据子集上进行，使噪声相关并可相消，从而避免 $1/\mu^2$ 放大。
**Mismatched-Data Regime**：失配数据 regime，指 ZO 优化的两次配对评估在不同数据上进行（如 on-policy RL 中不同策略生成的轨迹），噪声独立叠加导致 $1/\mu^2$ 放大。
**BOLT (Boltzmann Targeted SFT)**：Boltzmann 目标监督微调，将 KL 正则化 RLVR 目标转化为固定离线加权 SFT 问题的方法，通过参考采样和 prompt 归一化权重构造不依赖当前策略的固定数据集。
**On-Policy RL**：在线策略强化学习，策略在每一步生成轨迹时实时更新，导致不同参数下采样分布不同，天然产生数据失配。
**ZoAR**：Zeroth-Order query Reuse，通过缓存历史查询对并复用来降低单点 ZO 估计方差的技术。
**R-AdaZO**：Refined Adaptive Zeroth-Order optimizer，对 ZO 梯度估计进行一阶矩平滑与自适应二阶矩预处理的优化器。

## 可复现要素
- **数据集**：GSM8K（7,473 训练，1,319 测试）和 MATH（5,000 测试）均为公开数据集。
- **代码**：已公开，GitHub: https://github.com/lianmin-chen/the_cost_of_mismatch
- **模型权重**：使用公开预训练模型 Qwen2.5-1.5B、Qwen2.5-7B、DeepSeek-R1-Distill-Qwen-1.5B。
- **关键超参**：LoRA $r=8, \alpha=16$；ZO-BOLT $\mu \in \{10^{-3}, 10^{-2}\}$，$N_h=40$（GSM8K）/ $30$（MATH），学习率 $10^{-4}$；ZoAR + R-AdaZO（$\beta_1=0.9, \beta_2=0.99, \varepsilon=10^{-8}$）。
- **硬件**：NVIDIA L40 GPU（48GB VRAM）。
