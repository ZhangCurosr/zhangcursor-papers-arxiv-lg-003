---
title: "What-Drives-the-Zero-to-One-Jump-Diagnosing-and-Steering-Har"
source: https://arxiv.org/pdf/2611.02747v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:42:41"
field: "大语言模型推理强化学习"
keywords: ["RLVR", "GRPO", "negative learning", "zero-to-one jump", "generation entropy", "Fisher information", "policy optimization"]
innovations: ["提出早期分叉驱动函数 d_early-ppl，以前置熵模式区分接近可解失败与死端", "设计 NAPO 算法，在全负组内按驱动值分级施加负优势，注入结构化梯度", "建立 Fisher 信息与生成熵的理论桥梁，论证均匀负惩罚的信息浪费问题"]
benchmarks: ["AIME24", "AIME2025", "AMC23", "Math-Beyond", "LiveCodeBench"]
---

# 论文速读：What-Drives-the-Zero-to-One-Jump-Diagnosing-and-Steering-Hard-Prompt-Learning-in-RLVR

## 一句话总结
论文研究了 RLVR 训练中硬提示"零到一跳变"（zero-to-one jump）的动力学机制，发现失败 rollout 的**早期生成熵**（front-loaded entropy）能区分"接近可解的失败"与"死胡同"，并据此提出 NAPO（Negative-Augmented Policy Optimization），通过驱动函数对全负组内的负样本进行梯度加权，在 AIME24 上 Pass@256 提升 10.0 个百分点，AIME2025 提升 6.6 个百分点。

## 研究问题与动机
- **问题一：二元监督下零到一跃迁的诊断困境。** GRPO 等方法在全负组（all-negative group）中因缺乏正负对比而产生接近零的优势值，导致硬提示在大量训练步内停滞，随后突然解决；现有 pass@k 指标无法定位"首次解决"的时刻，群体级奖励也无法区分即将突破的失败与真正的死端。
- **问题二：全负组内异质负样本的梯度利用不足。** 现有改进方法（如 w-REINFORCE、NGRPO 等）对负样本施加近似均匀的惩罚，未能识别组内不同负 rollout 的潜在学习价值差异，造成优化预算浪费在低信息量的刚性失败上。
- **核心目标：** 回答"什么驱动了 N→P 跃迁？能否被主动引导？"，并提出一种可诊断、可干预的训练策略。

## 核心贡献（创新点）
1. **事件驱动的零到一动力学分析框架。** 首次引入 TTFP（Time-to-First-Positive）和 N→P/N→N 标记，以事件视角刻画硬提示首次可解的转化过程，而非仅依赖聚合型 pass@k。
2. **早期分叉驱动函数（early-branching driver）。** 定义 $d_{\text{early-ppl}}$，以响应前 20% token 的困惑度均值与全局均值之比度量"不确定性前置程度"，在 batch 内（Claim A）和跨 replay （Claim B）两个层面均有效分离 N→P 与 N→N 负样本。
3. **Fisher 信息视角的理论解释。** 证明 token 级 Shannon 熵与局部 Fisher 信息迹正相关，说明高熵失败路径对应更大的策略可塑性，为基于熵的负样本加权提供了理论依据（Lemma D.1）。
4. **NAPO 算法。** 将驱动函数嵌入 GRPO，在全负组中按驱动分数 min-max 归一化后赋予分级负优势，实现"解了用标准 GRPO，没解时用驱动加权 NSR"的分段更新策略，几乎不增加计算开销（每步仅多约 0.41s）。

## 方法详解
- **驱动函数定义（Eq. 6）：**
  $$d_{\text{early-ppl}} = \frac{\frac{1}{T_\alpha}\sum_{t=1}^{T_\alpha} \rho_t^{(k)}}{\frac{1}{T}\sum_{t=1}^{T} \rho_t^{(k)}}, \quad \rho_t^{(k)} = \exp(H_t^{(k)})$$
  其中 $H_t^{(k)}$ 是 top-k 截断 next-token 分布的 Shannon 熵，$\alpha=0.2$ 时分离效果最佳。
- **Fisher 信息代理（Lemma 4.1 / D.1）：** $\mathrm{Tr}(\mathbf{F}_t) = 1 - \sum_k p_{t,k}^2$，与 $H_t$ 正相关；低熵失败接近梯度饱和（saddle points），均匀惩罚无效。
- **NAPO 优势分配（Eq. 8–9）：**
  - 若组内有正样本（$p(x)=1$）：使用标准 GRPO 优势 $A_{\text{GRPO}}$。
  - 若无正样本（$p(x)=0$）：对每个负 rollout 计算归一化驱动 $\tilde{d}$，赋优势 $A = -\tilde{d}$，等价于加权 NSR。
- **理论最优性（Theorem D.4）：** 在"均匀成对相关性假设"下，Gram 矩阵最优权重 $w^* \propto \mathbf{G}\mathbf{1}$ 的排序与轨迹级 Fisher 信息迹（即熵）单调相关，驱动函数是其可计算的近似。
- **计算开销：** 仅需从已产出的 logits 中提取 per-token 标量，每步增加约 0.41s（占端到端时长的 ~0.03%）。

## 实验与结果
- **数据集：** 训练集 DAPO-17k（Qwen2.5-Math-7B 初始通过率 <0.2）；验证集 AMC23、AIME2025、AIME24、Math-Beyond；代码实验 LiveCodeBench v5/v6（Qwen3-4B）。
- **基线：** GRPO、W-REINFORCE（w-REINFORCE）、Uniform(−0.1) 控制、Uniform(−1) 控制。
- **主要结果（NAPO02 vs GRPO）：**
  - AIME24 Pass@256：**+10.0 pp**（0.9750 → 新结果）；Pass@8 **+27.28 pp**。
  - AIME2025 Pass@256：**+6.6 pp**；Pass@1 **+10.48 pp**。
  - AMC23 Pass@1 **+14.40 pp**；Math-Beyond Pass@1 **+7.43 pp**。
- **NAPO02 vs NAPO05：** NAPO02 在 AIME24 和 AIME2025 中段表现更强，NAPO05 在 AMC23 和高 k 区域更优，说明负更新强度需按基准难度校准。
- **Within-group 差异化 vs Uniform(−0.1)：** 在所有四个数据集上 Pass@256 均正向，证明梯度分化比常数负反馈更有效。
- **扩展验证：** Llama-3.1-8B-Instruct 上 AIME24 Pass@256 +4.45pp，Math-Beyond +1.29pp，部分退化表明超参需模型级校准；代码任务（LiveCodeBench）同样观察到 N→P 组 msk 更高的信号。

## 相关工作脉络
- **GRPO / DeepSeekMath（Shao et al., 2024; Guo et al., 2025）：** 本方法的基线框架，全负组优势消失是本工作要解决的核心缺陷。
- **NGRPO（Nan et al., 2025）与 Negative-Augmented 系列：** 已有工作利用负样本，但采用均匀权重；本文通过早期熵信号实现组内差异化加权。
- **Entropy-guided advantage shaping（Le et al., 2026）：** 关注 zero-variance prompts 的熵引导，但与本文的"早期分叉"模式和 N→P 事件诊断视角不同。
- **NSR / 负样本强化（Zhu et al., 2025）：** 提出 PSR/NSR 分解，但未解决全负组内负样本的异质性排序问题。
- **Dynamic rollout allocation（Fang et al., 2026）：** 动态调整 rollout 分配，与本文的"在固定 rollout 预算内重新分配负样本梯度权重"形成互补。
- **Selective rollouts（Zheng et al., 2025b）：** 发现低成功率提示存在结构化时序动态，本文为此动态提供了可操作的诊断信号和干预手段。

## 局限性与未来方向
- 当前分析仅限于**数学推理**场景与**二元可验证奖励**，在部分奖励、噪声奖励或主观反馈 setting 下驱动函数行为可能不同，需重新校准。
- 方法依赖 rollout 级统计量，效果可能受 **rollout 预算、组大小、解码温度、基础策略质量**等因素影响；在极低成功率（几乎从不接近正确解）或极高成功率场景下增益有限。
- Llama-3.1-8B 上的混合结果表明驱动窗口和优势幅度需要跨模型族调参，泛化性有待进一步验证。
- 代码任务的初步验证显示最优 $\alpha$ 不同（0.5 vs 0.2），需发展跨域自适应的驱动定义。

## 研究启发与可借鉴点
- **事件驱动的诊断思维：** 用 TTFP 和 N→P/N→N 标记代替静态 pass@k 来刻画训练动力学，可在本团队的 RL 训练监控中复用，用于定位"瓶颈期"和识别可干预的训练阶段。
- **"可塑性代理"的设计范式：** 用生成熵（或 Fisher 信息下界）作为策略可塑性的廉价代理，无需额外模型前向即可在 training time 获取，该方法论可迁移至其他基于 rollouts 的 RLVR/RLHF 场景。
- **分段更新策略（solved vs unsolved groups）：** "有正样本走标准 GRPO，无正样本走加权 NSR"的分段逻辑简单且稳定，可组合进任何 group-relative 变体。
- **计算开销极低的梯度重加权：** 仅增加每步 ~0.41s（0.03%），性价比极高，值得在团队的大规模 RL 训练管线中尝试集成。
- **驱动函数超参（$\alpha$）的基准难度敏感性：** NAPO02 vs NAPO05 的分化提示我们：未来设计熵相关模块时应考虑"难度自适应窗口"，而非固定超参。

## 关键术语表
- **Zero-to-one jump（零到一跳变）：** 硬提示在长期无进展后突然从全负组跃迁至至少包含一个正确 rollouts 的离散转变事件。
- **N→P / N→N transition：** 分别指"负→正"（从未解到首次解）和"负→负"（持续未解）的提示状态转移类型。
- **TTFP（Time-to-First-Positive）：** 提示 $x$ 在训练步 $t$ 首次出现至少一个正 rollout 的时刻，用于量化"首次解决"速度。
- **Early-branching driver（$d_{\text{early-ppl}}$）：** 响应前 $\alpha$  fraction token 的平均困惑度与全局平均困惑度之比，值越大表示不确定性越前置。
- **NAPO（Negative-Augmented Policy Optimization）：** 将驱动加权负优势嵌入 GRPO 的算法，全负组内按驱动值分级施加负奖励。
- **PSR / NSR（Positive/Negative Sample Reward）：** RLVR 目标的两种分解项，分别强化正确 rollout 和抑制错误 rollout。
- **Fisher information trace（Fisher 信息迹）：** 局部 token 分布的信息量度量，$1 - \sum_k p_k^2$，与 Shannon 熵正相关，表征策略在该点的可塑性。
- **Uniform-correlation assumption（均匀成对相关性假设）：** 同提示不同 rollout 的梯度向量间余弦相似度近似相等，使 Gram 矩阵最优权重可由对角项（Fisher 迹）近似排序。

## 可复现要素
- **数据集：** DAPO-17k（训练）；AMC23、AIME2025、AIME24、Math-Beyond（验证）；LiveCodeBench v5/v6（代码实验）。论文未明确说明 DAPO-17k 的公开状态，但 DAPO 系统本身已开源。
- **代码：** 已开源，见 https://github.com/jameszhu-pixel-time/Z2O。
- **权重：** 基于 Qwen2.5-Math-7B 和 Llama-3.1-8B-Instruct 开源权重训练，论文未提供独有预训练权重。
- **关键超参：** rollout 数 $G=8$，temperature $T=1.0$，max context 4096，learning rate $1\times10^{-6}$，KL penalty $0.0$，entropy penalty $0.0$，PPO minibatch size 256，early-fraction $\alpha=0.2$（NAPO02）或 $0.5$（NAPO05），top_k=-1，top_p=1.0，训练 1 epoch（Qwen: 200 steps；Llama: 100 steps）。
