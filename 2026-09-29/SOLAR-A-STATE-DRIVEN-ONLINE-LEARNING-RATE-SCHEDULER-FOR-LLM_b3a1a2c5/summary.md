---
title: "SOLAR-A-STATE-DRIVEN-ONLINE-LEARNING-RATE-SCHEDULER-FOR-LLM"
source: https://arxiv.org/pdf/2609.34681v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:10:00"
field: "LLM预训练优化"
keywords: ["learning rate scheduling", "large language models", "online optimization", "reinforcement learning", "pretraining", "PPO", "residual scheduling"]
innovations: ["以基线调度为锚、学习有界残差乘法修正的在线LR控制器", "轻量状态表征+分组squashed Gaussian动作+progress-aware奖励", "60M代理策略冻结后跨尺度/跨4倍基线LR复用无需目标PPO更新"]
benchmarks: ["C4 Llama 2 60M-1B AdamW/Muon", "The Pile Qwen2-MoE 1B", "Dolma 3 Mix DeepSeek-V2 MoE 3B"]
---

# 论文速读：SOLAR: A State-Driven Online Learning Rate Scheduler for LLM Pretraining

## 一句话总结
论文提出 SOLAR，一种基于强化学习（RL）的在线学习率调度器，通过在固定基线调度（如 Cosine）之上学习有界、状态依赖的残差修正，实现 LLM 预训练过程中每参数组的动态 LR 自适应；在 60M–1B 密集模型与 1B/3B MoE 模型上均显著降低最终困惑度（PPL）。

## 研究问题与动机
- **手调调度已主导 LLM 预训练但缺乏适应性**：工业界仍依赖 Warmup-Cosine-Decay / Warmup-Stable-Decay 等开环预设方案，一旦开始训练即无法随优化动力学（loss 波动、梯度范数漂移等）调整。
- **现有自适应优化器未解决全局 LR 问题**：Adam/Muon 等仅在参数级别做更新缩放，全局学习率轨迹仍需人工预设，因此调度层面的自适应仍然必要。
- **在线 L2O 调度在 LLM 规模上极不稳定**：可用信号（loss、梯度范数）高方差、非平稳、单步信息量弱；次优决策产生长视距静默退化，激进调整可在数步内引发灾难性发散，现有 RL 调度器难以在完整 LLM 预训练中稳定运行。
- **已有 learned LR 控制器均未在目标预训练同一路径内在线学习**：多数方法依赖离线元训练、独立目标 episode 或只在小型网络验证，缺少"在同一条全量 autoregressive LLM 预训练中同步训练并提升目标模型"的证明。

## 核心贡献（创新点）
1. **提出 SOLAR 框架：以基线调度为锚、学习有界残差修正**。与 GNS/GANNO/已往 RL 调度器直接预测绝对 LR 不同，SOLAR 的策略只输出相对于当前基线 LR 的 multiplicative residual，每步重新 anchor 到基线，避免探索误差累积成全新的 schedule。
2. **轻量状态表征 + 分组残差动作空间**。状态仅用 loss 统计、梯度范数、参数范数等低维通用信号（共 10 维），动作采用因式化 squashed Gaussian，每组独立均值、共享可学习 σ，并通过 action-scale warmup 抑制早期扰动；区别于 GANNO 的层间分类动作与递归绝对 LR 乘法器。
3. **Progress-aware 奖励设计 + 熔断恢复机制（Circuit-Breaker）**。奖励由即时改进项、EMA 长趋势项、分组稳定性 shaping 项组成；熔断器以 loss/EMA 比率阈值触发强制 PPO 更新并回滚至最近安全 checkpoint，解决单次鲁棒性失败。
4. **在 AdamW 与 Muon 上跨密集/稀疏大尺度实证**。60M–1B Llama 2（C4）与 1B/3B MoE（The Pile/Dolma）均获得一致 PPL 下降；1B AdamW 上相比 Cosine 降低超 10%，且 60M 代理策略冻结后可跨 4 倍基线 LR 范围复用，无需目标侧 PPO 更新。

## 方法详解
- **状态表征**：受控参数组 $g$ 的状态为全局与局部拼接，$s_{t,g}=[s_t^{\text{global}};s_{t,g}^{\text{local}}]$。全局特征包括归一化训练进度 $\tau_t$、$\log L_t$、短窗 loss 波动 $\nu_t$、长短 EMA 差异 $\Delta \text{EMA}_t$；局部特征包括 $\log \eta_{t,g}^{\text{base}}$、$\log\|\nabla_{t,g}\|$、上一动作 $a_{t-1,g}$、层深 $d_g$、$\log\|w_{t,g}\|$、梯度范数变化 $\Delta\log\|\nabla_{t,g}\|$，全部经 EMA 归一化（decay=0.99）。整体低维、脱离优化器内部统计。
- **动作空间**：每组采用 factorized squashed Gaussian，$u_{t,g}\sim\mathcal{N}(\mu_{t,g},\sigma^2)$，$\tilde a_{t,g}=\tanh(u_{t,g})$，clip 后执行；共享单标量 $\sigma$。残差 LR 调制：$\eta_{t,g}=\eta_{t,g}^{\text{base}}\cdot\exp(\alpha_t a_{t,g})$，$\alpha_t$ 在前 10% 步线性 warmup 至上限 $\alpha=1.3$，保证早期扰动有界。
- **奖励函数**：$r_{t+1,g}=r_{t+1}^{\text{perf}}+r_{t+1}^{\text{trend}}-p_{t+1,g}^{\text{stab}}$，其中性能项 $20\log(L_t/(L_{t+1}+10^{-10}))$，趋势项 $2(\text{EMA}^{\text{long}}-L_{t+1})/(\text{EMA}^{\text{long}}+10^{-8})$，稳定性项以组级梯度范数相对 EMA 比率 $q_{t+1,g}$ 打分，超阈值时施加额外惩罚。
- **Circuit-Breaker**：若 $L_{t+1}>1.5\cdot L_{t+1}^{\text{ema}}$ 则触发，施加 $\lambda_{\text{cb}}=100$ 全局惩罚、立即执行截断轨迹上的 PPO 更新，并向外层训练循环广播 abort 信号，回滚至最近安全 checkpoint（保留更新后的 PPO 状态）。
- **训练协议**：两隐层 MLP actor-critic（hidden=256）+ PPO；每 50 步更新一次、$K=4$ 个 epoch；$\gamma=0.99$、$\epsilon_{\text{clip}}=0.2$、$\lambda_v=0.5$、$\lambda_e=0.05$、GAE $\lambda=0.95$；由 rank-0 计算并广播 action，其余 worker 同步；新增开销约 1% 级 step time。

## 实验与结果
- **数据集/模型/优化器**：Llama 2 60M–1B 在 C4（1.4B–13.1B tokens），优化器 AdamW 与 Muon；另测 Qwen2-MoE 1B 在 The Pile，以及 DeepSeek-V2-style 3B MoE 在 Dolma 3 Mix（Megatron/32×A800）。
- **主要 PPL 数字**：1B AdamW + Cosine 基线 16.52 → SOLAR-online 14.83（降幅约 10.4%）；1B Muon + Cosine 14.36 → SOLAR-online 13.79；350M AdamW 18.31 → 17.23；1B frozen 进一步降至 14.63。MoE 1B（The Pile）9.61 → 9.34；3B MoE 10.73 → 10.38。
- **迁移实验**：60M 代理在线训练后冻结，应用于 130M/350M/1B 无目标 PPO 更新，在所有测试尺度及 0.5×/1×/2× 基线 LR 下均优于匹配 Cosine；2× 场景下 Cosine 在 ~4K–6K 步发散，SOLAR-frozen 仍可完成。
- **机械对照**：全局递归 PPO 27.09 → 锚定+有界 23.74（+3.35 PPL）→ 分组控制 22.87（再 +0.87 PPL）；AvgLR Replay 25.61、静态分组 profile 24.19，说明时空耦合与在线状态 conditioning 缺一不可。
- **开销**：1B AdamW 在线模式增加 1.23% wall-clock（step time 1.27%），frozen 增加 0.76%（step time 0.81%）；不引入额外 LLM forward/backward。

## 相关工作脉络
- **Hand-crafted schedules（Cosine/WSD/CLR/Blockwise LR）**：SOLAR 接受其作为 base，但不依赖其"一次性预设"属性；通过残差修正使同一 base 在不同轨迹中实时适配。
- **AutoLRS / MECHANIC / Prodigy / Schedule-Free**：前者仍是静态搜索或单标量缩放；SOLAR 是闭环多组残差控制器，且在 AdamW 和 Muon 上均有效，证明调度层控制独立于底层优化器几何。
- **GANNO（Tessera et al., 2022/2023）**：层间绝对 LR 分类动作 + 离线元训练；SOLAR 改为组级残差乘法动作 + 在线同轨学习，且实验显示 GANNO 在 130M 匹配设置下 24.95 PPL 劣于 SOLAR-online 22.79。
- **GNS（Xiong et al., 2022）**：图网络编码层状态、跨 episode 训练；SOLAR 不使用图结构，仅依赖低维标量统计，状态表征更轻、可直接在同一预训练 run 内在线学习。
- **L2O/learned optimizers（Andrychowicz, Li & Malik, VeLO 等）**：学习完整更新规则，参数空间大、跨尺度迁移难；SOLAR 学习更低维的调度修正，保留原始优化器更新规则，因此易于在 60M 代理上获得后复用到更大模型。
- **Adaptive optimizers（AdamW、Muon、Adafactor、Lion 等）**：仅作用于参数级别内部分子；SOLAR 在它们之外提供组级 LR 调控，两者正交可叠加。

## 局限性与未来方向
- 验证主要局限于 1B 以下密集模型与补充的 3B MoE；更大参数规模（如 7B/70B 级）下的行为与可扩展性尚未检验。
- 当前仅测试 C4 与 The Pile/Dolma 三类语料；跨语种、代码、多模态等更广泛数据分布上的泛化未知。
- 参数量分组策略为"每个可训练张量一组"（1B 模型 219 组），更细粒度（子张量级）或语义分组（如按 attention head、expert）的探索未见。
- 熔断器在所有主实验中未被触发，安全性主要依靠 action bound 与 reward shaping 隐式保障；极端分布偏移下的鲁棒性存疑。
- 在线模式的 PPO 更新周期（50 步）与超参在代理尺度选定后固定传递，目标尺度的自适应调节能力受限（frozen 模式尤其明显）。

## 研究启发与可借鉴点
- **"锚定基线 + 有界残差"范式**：将强先验（warmup-decay profile）放在控制器外部、让 RL 只学局部修正，可大幅缓解 L2O 的探索脆弱性；该思路可迁移到 warmup 比例、weight decay、gradient clipping 等超参的在线调节。
- **分组残差乘法 vs 递归绝对乘法**：本文对照显示，unanchored 递归乘法会缓慢漂移至灾难区间（α=2.6 时达到 5.7× 均值且 PPL=79.58），而每步 re-anchor 则稳定；任何在开放环境中学习乘法因子的 controller 都应优先采用该结构。
- **State 极简主义**：仅用 loss/gradient norm/parameter norm/进度/EMA 差异五个标量即达成显著收益，避免了引入优化器内部状态（如 Adam 的 m̂、v̂）带来的不可移植性；这一设计可直接复用至其它优化控制任务（batch size scheduling、mixed-precision 策略等）。
- **Froze-and-transfer 协议**：在 60M 代理完成 full-length 在线训练后冻结策略并跨 4 倍 LR 范围复用，无需目标侧再调参；这为"小模型代理→大模型生产"的 L2O 部署提供了可操作的工程路径。
- **Reward 三要素的解耦贡献**：即时性能项贡献最大（移除 +1.94 PPL）、趋势项次之（+0.90）、稳定性 shaping 最小（+0.30）但移除后唯一触发熔断；提示在其它 RL 优化控制任务中，"安全兜底项"虽对均值贡献小，却是必要组件。

## 关键术语表
- **SOLAR**：State-driven Online Learning rAte scheduleR，本文提出的基于 RL 的在线学习率调度框架。
- **Residual LR modulation**：在基线 LR 上乘 $\exp(\alpha a)$ 形式的有界乘法修正，使策略只需学习偏离基线的局部调整而非完整 schedule。
- **Base anchoring（重锚定）**：每步 action 乘子作用在当前基线 LR 而非上一时刻的有效 LR，阻断递归误差累积。
- **Action-scale warmup**：前 10% 训练步中线性放大残差动作上限 $\alpha_t$，避免早期不可靠策略造成剧烈扰动。
- **Circuit-Breaker**：基于 loss/EMA 比值的紧急触发器，命中后强制 PPO 更新并回滚至安全 checkpoint，防止单次 bad action 污染整条轨迹。
- **Progress-aware reward**：由即时 loss 改进、EMA 长趋势和组级梯度稳定性三项加权合成的奖励，分别对应短期下降、中期趋势与防发散。
- **Squashed Gaussian policy**：对高斯采样经 tanh 压缩并 clip 到 $[-1+\epsilon,1-\epsilon]$，再作 log-probability 变量变换校正的有界连续动作分布。
- **L2O（Learning to Optimize）**：用学习到的规则替代手工设计优化器的研究范式；本文落在 L2O 的"学习超参数调度器"子分支。

## 可复现要素
- **数据集**：C4（Raffel et al., 2020，公开）、The Pile（Gao et al., 2020，公开）、Dolma 3 Mix（OLMo 团队发布，公开）；均已在相应工作中开源。
- **代码/权重**：**论文未提及**代码或权重是否开源；实验复现需自行基于 PyTorch 实现，并在 Appendix A–B 的配置下对齐。
- **关键超参**：actor 两层 MLP hidden=256；$\alpha=1.3$，action warmup 占比 10%；PPO $\gamma=0.99$、$\epsilon_{\text{clip}}=0.2$、$\lambda_v=0.5$、$\lambda_e=0.05$、GAE $\lambda=0.95$、每 50 步更新一次、$K=4$ epoch；actor-critic 输入 dim=10、共享 σ 初始化 logσ=0；Circuit-Breaker 阈值 $\kappa_L=1.5$、惩罚 $\lambda_{\text{cb}}=100$。
- **模型/数据规模**：60M（11K 步/1.4B tokens）、130M（20K/2.6B）、350M（60K/7.8B）、1B（100K/13.1B）；MoE 1B 与 3B 配置见 Appendix B.4/C.12。
