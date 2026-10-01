---
title: "Prompts-Live-on-an-Arc-Gaussian-Curricula-in-Fisher-Rao-Coor"
source: https://arxiv.org/pdf/2609.38018v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:26:06"
field: "大语言模型强化学习"
keywords: ["GRPO", "prompt selection", "Fisher-Rao geometry", "curriculum learning", "RLVR", "rollout efficiency", "Kalman filter"]
innovations: ["揭示GRPO的Fisher-Rao弧长几何，证明在该坐标下更新均匀、零方差组为高斯边界层、证据同方差、目标梯度为高斯", "提出ARCUS算法：弧长Kalman滤波+目标匹配高斯核评分+frontier pacing，留空GRPO损失不变", "建立sampler-objective对应关系（Corollary 1），证明任何prompt选择器等价于对某目标函数的梯度上升"]
benchmarks: ["AIME24", "AIME25", "AMC23", "MATH500", "Minerva", "OlympiadBench"]
---

# 论文速读：Prompts Live on an Arc: Gaussian Curricula in Fisher-Rao Coordinates for Rollout-Efficient GRPO

## 一句话总结
本文揭示了 GRPO 优化过程内在的 Fisher–Rao 弧长几何结构，并据此提出 ARCUS——一种 rollouts 高效的 prompt 课程采样方法，通过 Kalman 滤波在弧长坐标下追踪 prompt 难度、按目标匹配的高斯核评分，并将训练目标动态推进至最难的可行目标。在六个数学推理基准上，ARCUS 相比 GRPO 平均提升 2.8–2.9 点，相比 Dynamic Sampling 提升 1.1–1.2 点，同时减少 48–57% 的 rollout 开销。

## 研究问题与动机
1. **零方差组的浪费问题**：GRPO 在采样一组 response 后，若所有 response 全部正确或全部错误，则组内方差为零，advantage 全为 0，不产生梯度，但仍需支付全部 G 次 rollout 的计算成本；此类组约占均匀采样 batch 的一半，且随策略提升占比增长。
2. **现有 prompt 选择方法的启发式缺陷**：现有方法（如 DS、GCS、FG-ExPO 等）在三个关键选择上均依赖手工设定：比较 pass rate 的坐标（raw p 或 logit）、偏好区域的宽度、以及目标值（通常固定为 p=0.5），缺乏从优化器几何中导出的依据。
3. **不同选择器的隐式目标不明确**：各种 sampler 会隐式改变 GRPO 所上升的目标函数，但未被显式建模，导致不同方法的比较和对齐困难。
4. **需要一种无需预测 rollout 的高效选择机制**：现有 rollout-free 预测方法（如 KGPS、MoPPS）在零方差边界处存在校准问题，logit 坐标下的不确定性估计会严重高估置信度。

## 核心贡献（创新点）
1. **揭示 GRPO 的弧长几何**：证明 ψ = arcsin√p 是 Bernoulli Fisher–Rao 流形的自然坐标，在该坐标下 GRPO 的期望更新是均匀的（除边界两层 ramp），为零方差组提供了精确的高斯边界层 bound。
2. **建立 sampler–objective 对应关系**：证明任何 prompt 选择器等价于对某个目标函数 F_w(ψ) 的梯度上升，高斯 sampler 对应 pass@k/pass^k 目标族中的某一成员。
3. **提出 ARCUS 算法**：基于该几何设计完整的 prompt 课程框架——Kalman 滤波在弧长坐标下更新信念、目标匹配的高斯核评分、仅保留 informative group 进行更新、目标沿 pass^k ↔ pass@k 族动态推进。
4. **理论与实验双重验证**：给出四条命题（均匀更新、高斯边界层、同方差证据、高斯目标梯度）的严格证明，并在三个 backbone、六个基准上验证 ARCUS 优于九种基线。

## 方法详解
### 几何基础
- **弧长坐标**：ψ = arcsin√p ∈ [0, π/2]，其中 p 为 prompt 的 pass rate。Bernoulli(p) 在 ψ 参数化下的 Fisher information 恒为 4，Fisher–Rao 距离为 2|ψ − ψ'|，Jeffreys prior 在 ψ 上均匀。
- **Proposition 1（GRPO 在弧长下均匀）**：对二元 reward 和 population-statistics 归一化，E[ĝ_q] = 2ω_G(p_q)∇_θψ_q，其中 ω_G → 1（当 G→∞ 且在 (0,1) 紧子集上），仅在边界 ψ→0 和 ψ→π/2 处有 ramp。
- **Proposition 2（零方差组为高斯边界层）**：P(全对或全错) = z_G(ψ) = cos^{2G}ψ + sin^{2G}ψ ≤ exp(−Gψ²) + exp(−G(π/2−ψ)²)。若 ψ ~ N(μ,P)，可闭式计算该 bound 的期望。
- **Proposition 3（证据同方差）**：对 0 < m < n 次成功，Laplace 近似的 mean 为 arcsin√(m/n)，variance 为 1/(4n)，与 m 无关（Anscombe 变换）；而 logit 坐标下的 delta-method variance 为 1/(np(1−p))，在 p→0,1 时发散。
- **Proposition 4（目标为高斯）**：pass@k 的梯度 H_k(ψ) = 2k sinψ cos^{2k−1}ψ 是 log-concave 密度，mode 在 ψ_k^* = arcsin(1/√(2k))，Laplace 近似为 N(ψ_k^*, 1/(4k))。

### ARCUS 算法流程
1. **弧长信念（Arc-length Beliefs）**：每个 prompt q 维护 ψ_q ~ N(μ_q, P_q)，初始化为 Jeffreys 矩 N(π/4, π²/48)。每次更新前按 drift-diffusion 模型传播：
   - μ_q ← Π[μ_q + v_t sin(2μ_q)]（漂移项建模策略整体能力提升，在两端 vanishes）
   - P_q ← P_q + Q_t（扩散项吸收 prompt 特异变化）
   观测更新采用 Kalman filter，observation (z, R) 按 Proposition 3 设置：
   - 0 < m < G：z = arcsin√((m+3/8)/(G+3/4))，R = 1/(4G+2)
   - m = 0：z = 0，R = 1/(2G)（零方差边界层 likelihood）
   - m = G：z = π/2，R = 1/(2G)

2. **目标匹配评分（Objective-matched Scoring）**：对目标 ψ*，kernel 为 N(ψ*, σ²) 其中 σ = min(sinψ*, cosψ*)/√2。score = κ_q(ψ*) × ŷ_q：
   - κ_q = √(σ²/(σ²+P_q)) · exp(−(μ_q−ψ*)²/(2(σ²+P_q))) 表示信念与目标的 overlap
   - ŷ_q = 1 − [exp(−Gμ_q²/a_q) + exp(−G(π/2−μ_q)²/a_q)]/√a_q，其中 a_q = 1+2GP_q，表示 informative group 的闭式概率下界

3. **候选采样（Candidate Selection）**：按 score 的 1/T 次幂做 Gumbel-top-M 无放回采样，M = ⌈(1+ρ)B⌉。

4. **前沿推进（Frontier Pacing）**：对 41 个目标 ψ ∈ Ψ（从 pass@G 到 pass^G），预测候选 yield Ȳ_t(ψ) = (1/M)∑π_q(ψ)ŷ_q，选择最难的可行目标：
   ψ̃_t = min{ψ ∈ Ψ : Ȳ_t(ψ) ≥ max_{ψ'} Ȳ_t(ψ') − ε}
   受限于 |ψ_t^* − ψ_{t−1}^*| ≤ Δ，warm-up 阶段固定 ψ^* = π/4。

5. **后采样选择（Post-rollout Selection）**：所有候选更新信念后，按更新后的 score 选取 top-B 个 informative group 进入不变的 GRPO 更新。

## 实验与结果
- **数据集**：训练池 DAPO-Math-17k（去重后的数学 prompt）；评估基准 AIME24/25、AMC23、MATH500、Minerva、OlympiadBench。
- **模型**：Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen3-4B-Base。
- **基线**：Uniform GRPO、GCS、AdaRFT、CurES、MoPPS、DPS、KGPS、GRESO、DS（共 9 种）。
- **关键结果（Table 1，平均六基准 accuracy）**：
  - Qwen2.5-Math-1.5B：GRPO 34.8 → ARCUS 37.6（+2.8），DS 36.5 → ARCUS 37.6（+1.1），rollout 倍率 ARCUS ×1.25 vs DS ×2.94。
  - Qwen2.5-Math-7B：GRPO 44.1 → ARCUS 46.9（+2.8），DS 45.7 → ARCUS 46.9（+1.2），rollout 倍率 ARCUS ×1.25 vs DS ×2.41。
  - Qwen3-4B-Base：GRPO 46.3 → ARCUS 49.2（+2.9）。
- **效率（Table 2，7B 模型）**：ARCUS 生成 rollout 0.77M（vs DS 1.48M，−48%），tokens 0.66B（vs DS 1.33B，−50%），wall-clock 24.1h（vs DS 41.6h），update batch 中 informative share 达 97%（DS 仅 100% 但生成 yield 仅 41%）。
- **消融（Table 3，7B）**：去掉 arc 坐标（改用 raw p）降 0.69 点；logit Kalman 降 0.96 点；去掉 dead-zone factor ŷ 降 1.03 点；固定目标（pass@1）降 0.62 点；Greedy 采样降 0.53 点。
- **泛化性（Table 11）**：在 DeepSeek-R1-Distill-Qwen-1.5B（长 CoT）上 +2.4 vs GRPO；在 Llama-3.2-3B-Instruct 上 +1.9；换用 MATH-train 池 +2.2；替换为 Dr. GRPO/DAPO/RLOO loss 仍保持 +1.0~1.2 over DS。

## 相关工作脉络
1. **Dynamic Sampling (DS, Yu et al., 2025)**：evaluate-then-filter 方法，生成后丢弃零方差组，需 2.4–2.9× rollout；ARCUS 在相同或更少 rollout 下达到更高 accuracy，且理论分析了 DS 等价于 arcsine 方向的纯缩放。
2. **KGPS (Zhu et al., 2026) / MoPPS (Qu et al., 2026a) / DPS (Mao et al., 2026a)**：rollout-free 预测方法，分别在 logit 空间做 Kalman filter、Beta posterior + Thompson sampling、HMM 动态；ARCUS 指出 logit 坐标在零方差边界处 noise 发散，而弧长坐标下 Kalman filter 校准更优。
3. **FG-ExPO (Lin et al., 2026)**：在 raw p 空间使用固定高斯 kernel (μ=0.5, σ=0.35)；ARCUS 证明高斯 kernel 需在弧长坐标下且宽度由目标 k 决定，raw p 上的高斯宽度会随 ψ 变化（σ_p/sin2ψ），形状由参数化而非目标决定。
4. **CurES (Zeng et al., 2025a) / Learnability sampling (Foster et al., 2025)**：分别使用 √(p(1−p)) 和 p(1−p) 作为分数；ARCUS 将其解释为 pass@1 kernel 的变体（前者为 ½H_1，后者为 ¼H_1²），但宽度未与目标匹配。
5. **Davis & Recht (2025) / Mroueh (2025)**：前者证明 GRPO 等价于对 arcsin√p 的梯度上升；后者分析 GRPO normalization 放大 rare success 的机制；ARCUS 在此基础上进一步将 sampler 的选择与隐式目标显式对齐（Corollary 1）。
6. **Advantage reshaping (pass@k 方法, Walder & Karkhanis 2025; Chen et al. 2025b)**：通过修改 advantage 来优化 pass@k；ARCUS 指出 sampler 同样能塑造目标，且能避免为 zero-variance 组支付 rollout 成本，两种路径可组合。

## 局限性与未来方向
1. **二元 reward 假设**：ARCUS 当前针对二元 verifier reward 推导；连续 reward 需二值化或寻找新的方差稳定变换（论文 Appendix C 讨论了 Dr. GRPO 等非标准归一化的适配方案，效果有部分保留）。
2. **prompt 间梯度独立性假设**：sampler–objective 对应关系假设各 prompt 有独立梯度，忽略了 prompt 间的知识迁移；当前用全局 drift 一阶建模，强异质迁移场景下可能欠拟合。
3. **目标推进的单一偏好**：pacing rule 编码了"在 yield slack ε 内选最硬目标"这一偏好；重视 reliability 而非 coverage 的应用可能需要其他 trade-off。
4. **实验规模限制**：当前实验覆盖到 7B 模型、G ≤ 16，code/agent 任务和更大 scale 尚未验证；代码智能体的零方差查询回收（Coelho et al., 2026）是可组合的方向。
5. **极小组 size 的挑战**：当 G ≤ 3 时两个 dead zone 覆盖大部分弧长，任何 selector 都难以有效工作。

## 研究启发与可借鉴点
1. **从优化器几何出发推导 sampler 设计**：本文的核心洞察是从 GRPO 的 normalization 结构反推其自然坐标（Fisher–Rao arc length），这一"先分析优化器几何再设计算法"的思路可推广到其他 RLVR 算法（如 RLOO、BNPO、REINFORCE++）的 prompt 选择器设计。
2. **零方差组的边界层 likelihood 而非丢弃**：传统 DS 方法丢弃零方差组浪费数据；本文用高斯边界层 likelihood 将其纳入 Kalman 观测，使 filter 在边界处依然校准，这一思路可用于其他带有确定性边界的统计推断场景。
3. **目标族（pass^k ↔ pass@k）的参数化课程**：将 curriculum 定义为可微的目标族参数 k，而非手工设定的难度 schedule，使 easy-to-hard 模式从 waste constraint 中 emergent 而非预设，这一机制化课程思想可迁移至 agent training、code generation 等任务。
4. **漂移项建模历史偏差（stale belief bias）**：Proposition 5 形式化证明了 history-based selectors 的系统性偏差方向（偏向过时的 easy prompts），并提出 mobility-weighted drift 作为一阶修正；这一偏差分析和修正框架可用于诊断其他在线学习系统的信念校准问题。
5. **closed-form 评分避免预测 rollout**：ARCUS 的所有评分量（kernel overlap、informative probability、pacing yield）均为 closed-form scalar，不依赖额外 forward pass 或预测模型，这种"几何推导代替经验调参"的轻量设计在 compute-constrained 场景下具有实践价值。

## 关键术语表
**Fisher–Rao arc length (ψ)**：Bernoulli 分布族在 Fisher information metric 下的弧长坐标，ψ = arcsin√p，使 Fisher information 恒为常数 4，是该流形的自然参数化。

**Zero-variance group**：GRPO 中一次采样的 G 个 response 全部正确或全部错误的组，此时组内 variance 为 0，advantage 全为 0，不产生梯度但消耗全部 rollout 成本。

**Informative group**：包含至少一个正确和一个错误 response 的组（0 < m < G），能产生非零 advantage，是 GRPO 的有效训练信号。

**Pass@k / Pass^k**：pass@k 表示 k 次尝试中至少一次正确的概率（1−(1−p)^k）；pass^k 表示 k 次尝试全部正确的概率（p^k）；两者是 GRPO 可优化的目标族的两个端点。

**Arc-length Kalman filter**：在 ψ = arcsin√p 坐标下运行的线性–Gaussian 状态估计器，利用 Anscombe 变换实现同方差观测，避免了 logit 坐标下零方差边界的噪声发散问题。

**Frontier pacing**：ARCUS 中沿目标族 ψ^* 的动态推进机制，每一步选择使预测 yield 不低于最佳 yield 减去 slack ε 的最难目标，形成 emergent 的 easy-to-hard curriculum。

**Drift-diffusion belief update**：ARCUS 中信念的预测步骤，drift 项 v_t sin(2μ) 建模策略整体能力提升（在两端 vanishes），diffusion 项 Q_t 吸收 prompt 特异变化，两者在线由 innovation 矩匹配估计。

**Gumbel-top-M sampling**：ARCUS 的候选 prompt 选择机制，按 score^{1/T} 的概率无放回采样 M = ⌈(1+ρ)B⌉ 个候选，实现带温度的 exploration。

## 可复现要素
- **数据集**：训练池 DAPO-Math-17k（公开）；评估基准 AIME24/25、AMC23、MATH500、Minerva、OlympiadBench（均公开）。
- **代码/权重**：论文声明理论验证脚本为 code/theory_checks.py；框架为 verl + vLLM；完整开源信息需在 arxiv 源码中确认。
- **关键超参**：B=256, G=8, 300 updates, lr=10⁻⁶, clip=0.2, 无 KL；ARCUS 参数：ρ=0.25, ε=0.03, T=0.3, Δ=0.005 rad, |Ψ|=41, warm-up ≈54 steps, β=0.1。
- **硬件**：8×H100 80GB。
