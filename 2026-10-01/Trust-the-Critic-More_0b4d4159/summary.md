---
title: "Trust-the-Critic-More"
source: https://arxiv.org/pdf/2609.39247v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:41:44"
field: "大语言模型强化学习"
keywords: ["Actor-Critic", "LLM Reinforcement Learning", "Credit Assignment", "Value Function", "GAE", "Mathematical Reasoning", "FineProofs-RL", "IMO-ProofBench"]
innovations: ["首次将GAE λ=0引入LLM RLVR，解除对终端奖励的强制依赖", "提出问题级Local Readiness门控，动态安全启用Critic信用分配", "结合参考解上下文注入与10k token动作分块显著提升Critic估值可靠性"]
benchmarks: ["FineProofs-RL", "IMO-ProofBench"]
---

# 论文速读：Trust-the-Critic-More

## 一句话总结
本文提出 Actor-Critic with Action Chunking (AC2)，通过引入局部置信度门控、参考解上下文注入与千级 token 动作分块，首次让 LLM RL 训练摆脱对每条轨迹终端奖励的强制依赖；在数学证明任务上，AC2 以更少的解码 FLOPs 和训练步数显著超越 GRPO，证明已学习的 Critic 可以被更积极地信任。

## 研究问题与动机
- **信用分配粗糙**：主流 LLM RL 方法（如 GRPO）将整条 rollout 视为单一动作，所有 token 共享由终端奖励决定的相同优势值，细微的错误（如末尾计算失误）会惩罚整条有效推理链。
- **Critic 利用率低**：尽管 actor-critic 可提供细粒度信用分配，但现有 RLVR 管线普遍认为已学习 critic 不够准确，仅将其用作基线（baseline），优势估计仍完全依赖终端奖励（λ≈1）。
- **算力浪费**：强制每条轨迹 rollout 至终止不仅消耗大量解码 FLOPs，也无法支持基于历史轨迹前缀的状态重置（state resetting）与 off-policy 学习。
- **设计空间未被探索**：若能在 critic 足够可靠时安全地跳过终端奖励，将打开 LLM RL 算法的全新参数与调度设计空间。

## 核心贡献（创新点）
1. **AC2 算法框架**：首次将 GAE 的 λ=0 优势估计引入 LLM RLVR，使策略更新完全依赖 critic 对动作块终点的估值，无需观测终端奖励。与 GRPO 的本质区别在于放弃 λ≈1 的蒙特卡洛终态反馈，转向纯值函数 bootstrap。
2. **Local Readiness（局部置信度）**：提出按问题维度动态判定 critic 可靠性的门控机制，仅在 critic 误差低于阈值且该问题已出现过正确解时才启用 critic-based 更新；这与现有工作全局固定插值或无条件使用 critic 的做法形成对比。
3. **参考解上下文注入（Reference Solution Conditioning）**：当历史上已获得正确答案时，将参考证明追加至 critic prompt，实证表明可显著降低 critic MAE，这是现有 actor-critic LLM RL 方法普遍缺失的 privileged information 利用方式。
4. **Action Chunking（动作分块）**：以 10k token 为信用分配单元替代单 token 或整条轨迹，在保持 critic 评估稳定性的同时实现更密集的中间监督；较 VinePPO/SPO 等依赖额外 Monte-Carlo rollout 的片段估值更为高效。
5. **理论-实证双重验证**：证明移除终端奖励依赖后仍可保持训练稳定性与性能提升，并系统消融揭示三个核心组件的必要性，为后续 LLM RL 算法设计提供可复用的调度范式。

## 方法详解
- **状态与动作定义**：将 MDP 中的动作视为长度为 `b` 的连续 token 块（action chunk），状态 `s` 为问题 + 部分 CoT 前缀。策略 `π_θ` 与价值网络 `V_θ^π` 共享参数。
- **Replay Buffer 采样**：维护 FIFO 缓冲区 `B`（容量 256 条完整轨迹）。每步采样 `n_refill` 个新问题（全 rollout 入队）与 `n_batch` 条历史轨迹，随机切割出前缀 `s`。
- **Local Readiness 判定**：问题满足三条件即进入 `ready` 集合：① 最近 5 步所有前缀的 critic 误差均值 ε < τ_global (0.20)；② 上一步该问题的 ε < τ_local (0.18)；③ 已至少产生过一条正确轨迹。一旦 ready 则永久保留。
- **Advantage 估计（λ=0 vs λ=1）**：
  - 非 ready 问题：生成完整轨迹，端点值 `v_i = r_i`，优势 `Â_i = r_i - mean_j(r_j)`（等价 GRPO）。
  - Ready 问题：仅生成最多 `b=10,000` token 的块，端点值 `v_i = V_θ^π(s·c_i)`，优势 `Â_i = v_i - mean_j(v_j)`。
  - Auditing 机制：对 α=1/4 的 ready 问题仍执行全 rollout，用于监测 critic 真实误差并维持 critic 训练目标的可度量性。
- **Critic 训练**：将前缀 `s` 的组均值 `mean_i(v_i)` 四舍五入至离散网格 `{0, 0.1, ..., 1}` 作为目标，以 next-token-prediction 损失拟合。有参考解时同时训练含/不含参考的两种 prompt（各权重 0.5）。Critic buffer 容量 1,920 对，每次采样最多 768 对更新。
- **Actor 更新**：采用 PPO-style clipped surrogate（Eq.5），advantage 沿整个 chunk 保持恒定，replay prefix 仅作为 conditioning 不计算 loss。结合自适应熵控制动态调整 `ε_high`。

## 实验与结果
- **数据集与评测**：训练集 FineProofs-RL（约 5,200 道国际竞赛几何/数论证明题）；验证集 IMO-ProofBench（60 道，带细粒度 rubric）；裁判模型 DeepSeek-V4-Flash（0–7 分制）。
- **基线**：GRPO（group size 16）、Prefix GRPO（相同 replay 前缀系统但仍 rollout 至终点并使用终端奖励）。
- **主结果**：
  - AC2 在 **90 步**即超越 GRPO 峰值验证均分 **18.50%**，而 GRPO 需 120 步；AC2 最终峰值达 **20.57%**。
  - 达到 18.50% 时，AC2 仅需 **0.79×10²⁰ Decoding FLOPs**，GRPO 需 **1.99×10²⁰**，算力效率提升 **2.5×**。
  - Critic 误差随训练单调下降，约在第 7 步跨越全局阈值，第 200 步 ready 问题占比达 ~70%。
- **关键消融**：
  - 去除 Local Readiness → 学习停滞，均分从 13.56% 降至 13.10%（step 40）。
  - 块长减至 2k token → 第 50 步落后（16.03% vs 16.41%），第 100 步峰值仅 16.62%，第 120 步坍塌至 13.57%。
  - 仅保留正确轨迹（≥6/7分）入队 → 早期超前（step 60: 17.35% vs 16.88%），后期严重退化（step 110: 14.67%）。
  - 高 stale replay buffer（每 10 步刷新一次）→ 仍可快速提升并超越 GRPO，表明对 off-policy 前缀的容忍度高。
  - 无 Group & Audit 单块变体 → 性能与 AC2 相当，验证了单步 bootstrap 的可行性。

## 相关工作脉络
- **GRPO 及其变体**：以组内终端奖励均值作基线（λ=1），强制全轨迹 rollout；AC2 通过 λ=0 彻底解耦优势估计与终态奖励。
- **VinePPO / SPO**：引入中间价值估计但依赖额外 Monte-Carlo 全 rollout 计算段价值，仍属 λ≈1 范畴；AC2 直接用单次 critic 前向完成估值，计算代价更低。
- **Le Critique / EVPO / BPCO / POISE / JustRL 2 / V / GenAC / VAPO**：同期工作均训练 critic，但优势估计仍绑定终端奖励或仅作基线；本文是首个明确以 λ=0 运行且大规模验证有效的 LLM RLVR 方法。
- **A3C / SAC**：经典离散/连续控制中的 actor-critic，通过 n-step return 缩短 horizon；AC2 思想同源，但面向 LLM 长 CoT 与可验证奖励场景重构了 state-resetting 与 readiness 调度。
- **Setlur et al. (Reuse your FLOPs)**：提出 off-policy prefix 训练但依赖正确轨迹过滤；本文证明混合正确/错误前缀并结合 readiness 门控反而更稳定。

## 局限性与未来方向
- **Local Readiness 依赖 epoching**：需累积完整轨迹以获取终端奖励 ground truth 来校准 critic，在单 epoch 或无限数据流场景下难以直接复用。
- **Critic 误差在部分难问题上仍偏高**：即使 ready 后，完全无参考解时的 critic MAE 仍在 0.06–0.10 区间波动，极端分布偏移可能导致早期信用分配噪声。
- **Chunk 长度固定为 10k**：未针对任务复杂度自适应调节，对极短推理或超长证明可能存在粒度错配。
- **未来方向**：设计无需终端 reward 的 critic 自校准机制（如自一致性bootstrap、reward model free uncertainty quantification）；探索 beam search / MCTS 与 AC2 训练的无缝衔接；将 readiness 扩展至 batch/loss-weight 软门控而非硬切换。

## 研究启发与可借鉴点
- **Critic 信任门控范式**：Local Readiness 的“问题级误差阈值+历史正确性要求”可直接迁移至其他 verifier-based RL 任务（代码生成、工具调用），作为安全启用 value-based 更新的开关。
- **Reference Solution Conditioning**：将已验证答案作为 privileged context 注入 critic prompt 是一种低成本、高回报的评估校准手段，适用于任何拥有少量 expert trajectory 的领域。
- **Chunk-level Credit Assignment**：10k token 块的设计平衡了 critic 预测稳定性与监督密度，可在多模态推理、长文档摘要等序列任务中替代单 token advantage，降低优化方差。
- **State Resetting + Off-policy Prefix**：从历史轨迹随机切割前缀并续写的范式，比从头 rollout 更高效；结合 stale buffer 实验表明该方法对策略偏移具有较强鲁棒性，可推广至离线 RL 预热阶段。
- **Decoding FLOPs 作为硬件无关算力指标**：本文用解码 FLOPs 替代 GPU hours 进行跨实验对齐，方法透明且可复现，值得在 RL 系统论文中推广为默认算力口径。

## 关键术语表
- **AC2 (Actor-Critic with Action Chunking)**：本文提出的 LLM RL 算法，通过动作分块与 critic 估值替代终端奖励进行策略梯度更新。
- **Local Readiness**：按问题维度动态评估 critic 可靠性的门控条件，满足误差阈值与正确轨迹要求后才启用 λ=0 更新。
- **GAE(λ)**：Generalized Advantage Estimation，λ 控制 bootstrap 深度；本文在 ready 问题上取 λ=0，完全依赖当前 critic 而非终端奖励。
- **Action Chunking**：将连续 token 序列聚合为固定长度块（本文 10k）作为信用分配单位，替代单 token 或整条轨迹的优势分配。
- **Auditing**：对部分 ready 问题仍执行全 rollout 并记录终端奖励，用于在线监测 critic 真实误差并防止训练漂移。
- **FineProofs-RL**：约 5,200 道国际数学竞赛证明题的训练数据集，配套精细评分 rubric。
- **IMO-ProofBench**：60 道 Olympiad 级证明题的独立验证集，由 DeepSeek-V4-Flash 裁判打分。
- **Decoding FLOPs**：仅统计策略 rollout 阶段的解码计算量，排除预填充、评估与参数更新，作为跨设备可比的学习效率指标。

## 可复现要素
- **数据集**：FineProofs-RL（公开，ArXiv:2604.04898）、IMO-ProofBench（公开，EMNLP 2025）。
- **代码/权重**：论文未提及开源仓库或模型权重发布。
- **关键超参**：`g=16`（组大小）、`b=10,000`（chunk 长度）、`α=1/4`（auditing 比例）、`τ_global=0.20`、`τ_local=0.18`、actor LR `2×10⁻⁶`、critic 初始 LR `2√2×10⁻⁶`（降至 floor `5√2×10⁻⁷`）、critic gradient-norm clip `0.2`、replay buffer 容量 `256`、critic buffer 容量 `1,920`、评估 temperature `0.8` / top-p `0.95` / top-k `20`。
