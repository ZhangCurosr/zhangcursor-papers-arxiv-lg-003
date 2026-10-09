---
title: "SAFE-ON-AVERAGE-UNSAFE-IN-THE-TAIL-WHEN-IS-THE-EPISODIC-COST"
source: https://arxiv.org/pdf/2610.09508v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:04:29"
---

# 论文速读：SAFE ON AVERAGE, UNSAFE IN THE TAIL: WHEN IS THE EPISODIC-COST TAIL CONTROLLABLE?

## 一句话总结
本文指出标准安全强化学习仅约束期望阶段成本会掩盖极端 episode 的尾部风险，提出用 CVaR₀.₁（最差 10% episode 的平均代价）作为统一后训练评估指标，并系统检验了五种标准算法与四类强约束族在导航与 locomotion 任务中能否在保持回报的前提下控制成本尾部。

## 研究问题与动机
- 标准 CMDP 安全 RL 仅要求 𝔼[C(τ)] ≤ d，但同一期望值下代价分布可能极不均匀，少量 episode 的代价可远超预算，而现有基准普遍只报告聚合均值，无法识别此类尾部违规。
- 即便采用更强的约束形式（分布 critic、可行域门控、预算状态化、尾部驱动乘子），在密集障碍导航任务中收紧约束往往导致回报坍塌至零，无法实现“持回报控尾”。
- 尚不清楚何种任务结构或策略行为模式允许在维持高任务回报的同时将最坏情况控制在预算内。
- 缺乏跨算法、跨任务家族的统一尾部评估协议，难以横向比较“均值合规”与“尾部安全”之间的真实差距。

## 核心贡献（创新点）
1. **共享后训练尾部评估协议**：同时度量期望成本与 CVaR₀.₁，首次在统一实验下暴露五种标准期望成本算法在导航任务中的严重尾部违规（最高达 18.2× 预算）。
2. **四类强约束族的对照研究**：系统测试可行性门控、预算状态化、尾部驱动拉格朗日、逐状态分位数-CVaR critic，证明在密集危险导航中收紧约束无法打破回报-尾部权衡。
3. **OQ-SAC 与尾部可控性经验判据**：提出融合乐观探索与分位数-CVaR 惩罚的安全控制器，在 locomotion 上实现种子级尾安全；总结出“安全且高回报的 episode 必须频繁出现”这一可观测指标，并给出导航任务中尾部不可控的 episode 层面解释。

## 方法详解
- **评估定义**：CVaR₀.₁(C) 为代价分布上尾 10% 的均值；经验估计为 100 条确定性 rollout 中代价排序后最高的 10 条均值。尾安全判定为 𝑪𝒗𝒂𝑹₀.₁ ≤ d=25。
- **四类约束族设计**：
  - *Feasibility gating*：学习 max-over-time 可达避免 critic V_h，结合标量/状态依赖乘子及 RESPO 风格门控，在预估可行状态追求回报、否则最小化 V_h。
  - *Budget as state*：将归一化剩余预算 z_t 拼接为观测（z_{t+1}=z_t−c_t/d），累计代价超过 d 时替换任务奖励为负惩罚。
  - *Tail-driven Lagrangian*：保留 SAC-Lag 标量 critic，但乘子由最近 10 个训练 episode 的 empirical CVaR₀.₁ 或 C̄+kσ_C 更新；replay 中 25% 采自一步代价最高的 10% 转移。
  - *Per-state quantile-CVaR critic*：用 32 分位数模型拟合折扣成本回报 Z^C(s,a)，actor 惩罚取最大的 [αN] 个分位数均值；critic 训练同样偏向高代价转移。
- **OQ-SAC**：继承 ORAC 的乐观探索机制并与分位数-CVaR 目标策略学习结合。使用 E=3 个奖励 critic 与 E=3 个含 N=16 分位数的成本 critic；数据收集时基于 UCB 奖励与 LCB 成本的梯度偏移执行探索动作；目标策略 loss 为 𝓛_π=𝔼[α_ent log π − q̄ + λ c̄]，其中 q̄/c̄ 为集成均值；λ 按任务固定（导航 0.06–0.15，locomotion 20–30）；critic minibatch 中 25% 采样自历史一步代价 Top-10% 转移以提升尾部估计精度。

## 实验与结果
- **基线算法**：SAC-Lagrangian, TRPO-Lagrangian, FOCOPS, PPO-Lagrangian, CPO（OmniSafe）。
- **导航任务**（PointGoal1/2, PointButton1, CarButton1, PointPush1）：所有方法均不尾安全；高回报策略 mean cost 47–55 且 CVaR₀.₁ 达 110–229（4.4–9.2×d）；低回报策略 mean cost 19 但 CVaR₀.₁ 仍达 139（5.6×d）；四个强约束族 38 个训练 run 的 17 个 operating point 全部落入回报坍塌区。
- **OQ-SAC 在 locomotion**（HalfCheetah, Ant, Walker2d, Hopper）：HalfCheetah（Return 2868±46, CVaR 0±0）、Ant（2656±329, 0.9±0.6）、Hopper（877±556, 1.0±1.3）在 3 个 seed 上均尾安全；Walker2d 4/5 seed 安全（余下 28.3）。对比 SAC-Lag/PPO-Lag/WCSAC，OQ-SAC 在 HalfCheetah 与 Hopper 是唯一双任务种子级安全的算法。
- **消融结论**：Mean-cost 变体与 No-shift 变体在 HalfCheetah/Hopper 各自独立即可达到尾安全，说明单组件非必需但组合更稳健。
- **关键发现**：尾部可控性与“安全 episode 的频率及回报”强相关；locomotion 中安全高回报 episode 占比高（HalfCheetah 100%），而导航中安全 episode 几乎无回报或占比极低（PointButton1 仅 7%）。

## 相关工作脉络
- *标准约束 RL*（CPO, PPO-Lag, TRPO-Lag, SAC-Lag, FOCOPS）：优化期望成本，本文证明其无法识别/控制尾部风险，且在稠密导航
