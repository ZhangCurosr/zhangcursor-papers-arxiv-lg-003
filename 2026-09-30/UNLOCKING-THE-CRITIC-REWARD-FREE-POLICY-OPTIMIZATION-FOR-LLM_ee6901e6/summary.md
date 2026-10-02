---
title: "UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM"
source: https://arxiv.org/pdf/2609.37119v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:37:14"
field: "大语言模型强化学习后训练"
keywords: ["reward-free RL", "critic reuse", "LLM post-training", "chain-of-thought reasoning", "proximal policy optimization", "length debiasing"]
innovations: ["证明 critic 不稳定性源于优化配方而非网络本身，单步低方差更新可稳定训练", "单一冻结校准 critic 同时担任 rollout 奖励、GAE 基线、未完成前缀预测器三重角色", "通过长度十分位去偏加二值化阈值关闭策略 exploit 通道，实现零标签匹配监督 PPO"]
benchmarks: ["AIME 2025", "AIME 2026", "AMC 2023", "GPQA-Diamond"]
---

# 论文速读：UNLOCKING-THE-CRITIC-REWARD-FREE-POLICY-OPTIMIZATION-FOR-LLM

## 一句话总结
论文提出 RFPO（Reward-Free Policy Optimization），将预训练的 Critic 冻结并校准后，直接用作 rollout 级奖励、GAE 基线和不完整前缀的成功预测器，实现无需外部验证器标签的 RL 后训练，在长跨度数学推理任务中以更低计算/内存开销达到与监督 PPO 相当的性能。

## 研究问题与动机
- **Critic 被盲目丢弃**：当前 LLM RL 后训练趋势正移除 Critic（以多采样 baseline 替代），理由是内存开销大、训练不稳定；但 Critic 已学会预测轨迹成功率，其信号被白白浪费。
- **Reward 标注成本高**：人类反馈、结果/过程奖励模型均需独立标注工程；可验证 reward 要求每条 prompt 有参考答案，均无法随 RL 规模线性扩展。
- **长轨迹等待成本**：在 long-horizon 推理中生成占主导成本，稀疏终端 reward 导致训练必须等所有轨迹完成后才能更新，浪费严重。
- **核心科学问题**：若一个 Critic 已足够可靠作为 GAE baseline，它是否也足够可靠作为 reward 本身？

## 核心贡献（创新点）
1. **Critic 不稳定性是优化配方问题**：长 Chain-of-Thought 中 critic-based RL 的不稳定源于多 inner update 的高方差更新，保持单步、低方差更新即可稳定收敛——这是对"移除 Critic"主流叙事的基础性反驳。
2. **单一固定 Critic 三合一**：一个冻结校准后的 Critic 同时承担 rollout 级奖励、GAE 基线、未完成前缀成功预测器三重角色，训练循环中无需 verifier、无需训练价值网络。
3. **二值化去偏奖励防 exploit**：连续使用 Critic 评分会被策略利用长度偏差（响应变短或变长），通过在校准集上拟合长度十分位偏移 b(ℓ) 并设置分位数阈值 τ 进行二值化，关闭 exploit 通道。
4. **零标签匹配监督 PPO，并节省算力**：在 5,120-token 上限下 RFPO（0% 标签）pass@1 均值 41.1% vs 监督 PPO 41.8%；在更极端的 4,096-token 上限（约半数 rollout 未完成）下 RFPO 平均 47.6% 超过 PPO 的 46.6%，且 GPU-hours 从 327 降至 264（−19%）。

## 方法详解
- **Critic 预训练**：从 SFT checkpoint 出发，在 DAPO-Math-17k 上做两阶段 supervised PPO（第一阶段 8,192-token 预算 500 步，第二阶段 5,120-token 上限 300 步），在累计步 800 提取 critic V̄φ（解释方差 ≈ 0.55，within-problem AUC = 0.837）。
- **校准（唯一离线步骤）**：从初始策略 πθ₀ 采样 512 条 on-policy rollout，按长度十分位分组，在正确/错误两类内分别计算分数偏移 b(ℓ)；阈值 τ 取校准集上使正样本率与 verifier 正样本率（53.5%）对齐的分位数，得 τ = 0.5215，与 verifier 一致率达 93.4%。
- **奖励公式**：r̂(x,y) = 1[v(x,y≤T) − b(ℓ(y)) > τ]，其中 v 是冻结 critic 的终端值，ℓ(y) 为响应长度。
- **GAE 坍缩**：γ = λ = 1 时，优势 A_t = r̂ − V̄φ(x, y≤t)，无需额外 backward。
- **训练规则**：每个 batch 仅做一次策略梯度更新（mini-batch = full batch，无 inner loop），clip ratio = 0.2，actor lr = 10⁻⁶，critic 冻结不更新。
- **部分监督可选**：以概率 p 混入真实 verifier label，p = 0 为完全 reward-free，p = 1 退化为监督 PPO。

## 实验与结果
- **模型与数据**：Qwen3-4B-Base，SFT 于 OpenR1-Math-220k（45k 条 CoT），critic 预训练于 DAPO-Math-17k（≈ 563k 条 verifier-labeled rollout）。
- **评估基准**：AIME 2025/2026、AMC 2023、GPQA-Diamond；每 10 步评估，8 samples/problem，temp=0.7，12,288-token 验证预算。
- **主要结果（300 步，5,120-token cap）**：
  - RFPO 0% labels 在四基准平均 pass@1 达 41.1%，监督 PPO 为 41.8%，AIME 2026 pass@1 持平（21.7% vs 21.7%），GPQA 领先 0.7 点。
  - 峰值 AIME 2026 pass@1：RFPO 三份 run 最高 26.2% vs PPO 25.0%。
- **关键对比（4,096-token cap，约半数 rollout 未完成）**：
  - RFPO 宏观均值 47.6% > PPO 46.6%；GPU-hours 264 vs 327（−19%）；每步耗时 395s vs 490s（−28%）；峰值显存降 9.4 GB/GPU。
- **连续 vs 二值化**：连续 reward 可超监督 PPO 至 29.2（AIME 2026 pass@1），但后续崩溃（最终 16.2%），响应长度漂移 +33.6%；二值化牺牲峰值换取稳定。
- **Critic 预测能力**：在 1,024-token 前缀上 within-problem AUC = 0.60，4,096-token 达 0.84，说明未完成轨迹仍可被有效打分。

## 相关工作脉络
- **DeepSeek-R1 / Kimi k1.5/k2 / MiniMax-m1**：采用无 critic、GRPO-style 多采样 baseline 方案；本文反向论证 critic 价值。
- **Vapo / EVPO / Seed1.5-thinking**：尝试修复 critic（生成式 critic、解释方差加权），但仍需外部 reward；本文让 critic 兼任 reward。
- **DPO / PRIME / Implicit Reward**：仍依赖训练期标签；本文 reward 完全来自冻结 verifier 前的离线校准。
- **Self-rewarding / Majority voting / Generated rubrics**：无 grounded supervision，易 reward hacking；本文 reward 有 verifier  grounded 且有校准边界。
- **Process Reward Models**：需 step-level 标注，判已写步骤；Critic 预测轨迹终点并从结果学习，无需细粒度标注。
- **Sparrow / Truncation-based 方法**：仅截断 rollout 或稀疏 attention，无法给未完成轨迹打分；Critic 提供这一能力。

## 局限性与未来方向
- 实验仅限于 Qwen3-4B-Base 与数学推理，未验证更大模型或其他任务（如 agentic、code）。
- 单步更新规则虽稳定，但对更复杂任务/更长 horizon 的泛化性待检验。
- 连续 reward 的更高精度未被安全利用：存在"高天花板但会崩溃"的两难，需开发 validated stopping rule。
- 校准依赖一次 512 rollout 的离线拟合，跨域/跨分布迁移时可能需重新校准。

## 研究启发与可借鉴点
- **Critic 重用范式**：任何已训练的价值网络均可尝试冻结复用为 reward，避免额外 reward model 标注与推理开销。
- **去偏+二值化防 exploit**：对任何带连续 bias（如长度、复杂度）的 proxy reward，分群去偏后二值化是关闭 exploit 通道的有效通用技巧。
- **单步低方差更新**：在 critic-based RL 中，以单步全 batch 更新替代多 inner update，可系统性缓解长 CoT 训练不稳定。
- **未完成轨迹奖励**：利用价值网络的早期预测给 truncated rollout 打分，可在生成预算受限场景下显著提升训练效率。
- **解释方差作为训练指标**：EV 与 offline AUC 高度相关（r=0.91），可在训练中低成本监控 critic 质量。

## 关键术语表
- **RFPO（Reward-Free Policy Optimization）**：本文提出的方法，用冻结校准后的 critic 替代 verifier 作为训练 reward。
- **GAE（Generalized Advantage Estimation）**：actor-critic 中用于估计优势函数的无偏估计，本文因 γ=λ=1 而简化为 r̂ − V。
- **Within-problem AUC**：固定题目下 critic 区分正确/错误 response 的排名能力，衡量 reward 可用性而非仅难度识别。
- **Length bias**：Critic 原始分数与响应长度负相关的系统性偏差，策略可利用该偏差缩短/拉长响应而不改善正确率。
- **Calibration**：在 512 条 on-policy rollout 上拟合并部署长度偏移 b(ℓ) 与阈值 τ，使二值化 reward 的正样本率对齐 verifier。
- **Explained Variance（EV）**：critic 预测对 returns 方差的解释比例，本文用作训练时低成本 critic 质量代理指标。
- **Single-update rule**：每个 batch 仅执行一次策略梯度更新（mini-batch = full batch），是保证 critic-based RL 稳定的关键配方。
- **Reward hacking**：策略 exploit 奖励函数漏洞而非真正提升任务性能，本文二值化校准正是为防范此类行为。

## 可复现要素
- **数据集**：OpenR1-Math-220k（SFT 用 45k 子集）、DAPO-Math-17k（critic 预训练）；评估集 AIME 2025/2026、AMC 2023、GPQA-Diamond——均为公开数据集。
- **代码/权重**：训练框架使用 verl；论文声明所有实验 artifact（训练日志、critic 校准文件、benchmark 输出、绘图脚本）均已附于 submission 附录（data/logs/, data/critic/baselines/ 等），可再生成全部图表。
- **关键超参**：actor lr=10⁻⁶，critic lr=10⁻⁵（仅监督 baseline 使用），clip=0.2，γ=λ=1，temperature=1.0（生成）/0.7（评估），prompt cap=2,048，训练生成 cap=5,120（主实验）/4,096（消融），验证 cap=12,288，batch=128 prompts × 8 responses=1,024 rollouts，单步更新，无 KL/entropy penalty。
