---
title: "Prompts-Live-on-an-Arc-Gaussian-Curricula-in-Fisher-Rao-Coor"
source: https://arxiv.org/pdf/2609.38018v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:25:57"
field: "大语言模型强化学习与课程学习"
keywords: ["GRPO", "Fisher-Rao arc length", "prompt selection", "rollout efficiency", "curriculum learning", "RLVR"]
innovations: ["揭示GRPO的Fisher-Rao弧长坐标下更新均匀、零方差高斯边界、证据齐次与目标高斯化", "ARCUS: 弧长Kalman信念+目标匹配高斯核+yield-pacing的rollout高效提示选择", "证明任意sampler隐式决定目标,统一DS/CurES/FG-ExPO/KGPS为特例"]
benchmarks: ["AIME24/25", "AMC23", "MATH500", "Minerva", "OlympiadBench", "DAPO-Math-17k"]
---

# 论文速读：Prompts Live on an Arc: Gaussian Curricula in Fisher–Rao Coordinates for Rollout-Efficient GRPO

## 一句话总结
论文提出了 ARCUS（Arc-length Curriculum Sampling），一种基于 Fisher–Rao 几何的 rollouts 高效提示选择方法，利用 GRPO 在伯努利分布上的自然坐标 ψ = arcsin√p（弧长），将提示课程形式化为弧长空间中的高斯核，并在不变 GRPO 更新的前提下将平均准确率提升 2.8–2.9 分、rollout 数较动态采样减少 48–57%。

## 研究问题与动机
- GRPO 中零方差组造成大量 rollouts 浪费：当一组 G 个采样响应全部正确或全部错误时，组内优势全为零，但该组的 G 个 rollout 已被计入成本；均匀采样下这类组约占一半，且随策略改进占比继续上升。
- 现有提示选择方法在“坐标—宽度—目标”三个关键超参上都依赖独立启发式：raw p/logit 坐标、带宽设定、以及通常固定在 p=0.5 的目标。
- 不同选择并非等价：GRPO 的标准化本身隐含了特定的几何（弧长上的均匀更新），但先前工作未将坐标/宽度/目标统一到这一几何上，导致例如 logit 空间 Kalman 在零方差附近过自信、原始 p 上高斯宽度随位置变化而非由目标决定。
- 目标：在保持 rollout-free 的预测-选择范式下，从 GRPO 的几何中推导出坐标、带宽与目标的一致性设计，并用闭式评分与 pacing 实现高效采样。

## 核心贡献（创新点）
1. **揭示 GRPO 的弧长几何**：证明在 ψ = arcsin√p 下，GRPO 期望更新在内部近似均匀、零方差组概率被高斯边界层有界、通过证据噪声齐次，且 pass@k/pass^k 的梯度为闭式高斯；这是与 Davis & Recht、Mroueh 等工作的本质区别——后者关注 advantage 重塑，本文揭示 sampler 同样隐式决定目标。
2. **ARCUS 框架**：把提示课程定义为弧长空间中的高斯核 × 信息组概率，用 Kalman 滤波跟踪每个提示的 (μ, P)，只让信息组进入未修改的 GRPO 更新；与已有工作的区别在于坐标/带宽/目标三者均由几何导出而非启发式。
3. **目标 pacing 规则**：以闭式预测的 yield 约束在 ε 内选最困难目标，形成 pass^k ↔ pass@k 族上的同伦，避免 DS 等事后过滤的浪费。
4. **统一视角与消融**：证明 DS 为 objective-neutral（仅在 arcsine 方向放大步长）、learnability/CurES/FG-ExPO/KGPS 等为 ARCUS 的特例（固定目标/错配宽度/错配坐标）。
5. **实测收益**：在三架构六基准上均优于九类基线，较 DS 以 48–57% 更少 rollout 达到同等或更高准确率。

## 方法详解
- **弧长坐标**：ψ = arcsin√p ∈ [0, π/2]，Bernoulli 的 Fisher 信息为 4（与 ψ 无关），Jeffreys 先验在此坐标下均匀；∇p = sin2ψ · ∇ψ。
- **Proposition 1（GRPO 均匀性）**：对二元奖励与总体统计标准化，E[ĝ_q] = 2 ω_G(p_q) ∇ψ_q，ω_G → 1 一致于任何紧子集，仅在两端 ramp（宽约 1/√G）处下降。
- **Proposition 2（零方差边界层）**：零方差概率 z_G(ψ) = cos^{2G}ψ + sin^{2G}ψ ≤ e^{-Gψ²} + e^{-G(π/2−ψ)²}；对高斯信念的期望可闭式积分。
- **Proposition 3（齐次噪声）**：Anscombe 估计 ψ̂_A = arcsin√((m+3/8)/(n+3/4)) 的方差为 1/(4n) + O(n^{-2})；零/全成功时的似然以均值 0 或 π/2、方差 1/(2n) 的高斯有界；logit 空间方差 1/(np(1−p)) 在端点无界。
- **Proposition 4（目标高斯化）**：pass@k 目标梯度 H_k(ψ) = 2k sinψ cos^{2k−1}ψ 为 log-concave 密度，mode ψ_k^* = arcsin(1/√(2k))，Laplace 近似为 N(ψ_k^*, 1/(4k))；pass^k 为其镜像。
- **Corollary 1（采样器即目标）**：若 inclusion prob 为 w(ψ)，则总更新为 ∇_θ Σ_q F_w(ψ_q)，F_w' = 2w·ω_G；高斯 w 对应 smoothed count 目标。
- **ARCUS 信念更新（Algorithm 1/2）**：每个提示维护 ψ_q ~ N(μ_q, P_q)；漂移项 v_t sin2μ_q 建模全局能力增益（mobility 在两端消失）；观测按 (3) 给出：信息组用 Anscombe，零/全组用边界层 (z=0/π/2, R=1/(2G))。
- **评分**：κ_q(ψ*) 为信念与目标核 N(ψ*, σ²) 的闭式重叠，ŷ_q 为信息组概率下界；s_q = κ_q · ŷ_q。
- **Pacing**：在 |Ψ|=41 目标格点上预测候选 yield Ȳ_t(ψ)，取满足 Ȳ_t ≥ max Ȳ_t − ε 的最困难 ψ*，受 |ψ_t^* − ψ_{t−1}^*| ≤ Δ 限速；warm-up 持 ψ^* = π/4。
- **后采样选择**：Gumbel-top-M 抽取 M = ⌈(1+ρ)B⌉ 候选， rollout 后更新信念，仅前 B 个信息组进入未改动的 GRPO。
- **成本**：每步闭式标量计算，N≈17k、|Ψ|=41 下 <0.1s；额外开销仅为 ρBG 候选 margin。

## 实验与结果
- **设置**：Qwen2.5-Math-1.5B/7B、Qwen3-4B-Base；DAPO-Math-17k 训练；B=256, G=8, 300 次更新，lr=1e-6, clip=0.2；评测 AIME24/25、AMC23、MATH500、Minerva、OlympiadBench（六基准平均）。
- **主结果（Table 1）**：
  - Qwen2.5-Math-1.5B：ARCUS 37.6 vs GRPO 34.8（Δ=+2.8），DS 36.5（×2.94 rollout）。
  - Qwen2.5-Math-7B：ARCUS 46.9 vs GRPO 44.08（Δ=+2.8），DS 45.70（×2.41 rollout）。
  - Qwen3-4B-Base：ARCUS 49.2 vs GRPO 46.27（Δ=+2.9），DS 48.0（×2.63 rollout）。
  - 相对最强 baseline DS 提升 +1.1~+1.2。
- **效率（Table 2, Qwen2.5-7B）**：ARCUS 生成 rollout 0.77M、tokens 0.66B；DS 为 1.48M / 1.33B；ARCUS 生成 yield 79%、更新批次 97% 信息；耗时 24.1h vs DS 41.6h；达 DS 最终精度仅需 0.45M rollout（少 70%）。
- **消融（Table 3）**：
  - 坐标影响最大：换 raw-p 高斯 −0.69，logit Kalman −0.96。
  - 零方差因子 ŷ 最关键（−1.03），否则硬目标会拉入全失败层。
  - Pacing 优于固定目标：固定 pass@1 −0.62，直接跳到最终目标 −0.76，reward-feedback pacing −0.85。
- **机制分析**：目标随 G 自动找到闭式 frontier（G=8→p*=0.29；G=4→0.43；G=16→0.18）；信念校准误差 0.09（vs Beta 0.12、logit-KF 0.17）；历史方法（MoPPS/KGPS）信念易老化偏向 p≈0.75–0.8，ARCUS 漂移项维持中位≈0.5。
- **泛化（Appendix F）**：R1-Distill-1.5B (+2.4 vs GRPO, +1.1 vs DS)、Llama-3.2-3B-Instruct (+1.9/+0.8)、MATH pool (+2.2/+0.9)；换 Dr. GRPO/DAPO/RLOO 损失仍领先 DS +1.06~+1.15。

## 相关工作脉络
1. **Dynamic Sampling (DS, Yu et al., 2025)**：事后过滤零方差组，buy 更干净的批次但不改变目标（arcsine 方向放大）；ARCUS 从几何导出目标，避免为被丢弃组付费。
2. **FG-ExPO (Lin et al., 2026) / GCS**：在 raw p 上做高斯权重；论文证明 raw-p 高斯在弧长上宽度随位置变化（σ_p/sin2ψ），不与任何 pass^k/pass@k 目标匹配。
3. **KGPS (Zhu et al., 2026)**：logit 空间 Kalman + E[p(1−p)] 评分；logit 坐标下零方差证据噪声无界，导致过自信（Figure 1c），且其目标为固定 p=0.5 附近。
4. **CurES (Zeng et al., 2025a) / Learnability sampling**：score ∝ √p(1−p) 或 p(1−p) 对应 pass@1 核的缩放版本，宽度未按目标闭式匹配；ARCUS 将其定位为目标宽度 σ = min(sin ψ*, cos ψ*)/√2 的特例。
5. **Davis & Recht (2025) / Mroueh (2025)**：揭示 GRPO 目标为 arcsin√p、标准化放大罕见成功；本文在此基础上证明 sampler 的选择同样构成目标，且能通过抽样避开零方差组的代价。
6. **Advantage reshaping（pass@k 优化）**：Walder & Karkhanis、Chen 等通过改 advantage 实现；本文走 sampler 侧，同目标但省去零方差组的 rollout 开销。

## 局限性与未来方向
- 理论针对二元奖励与 GRPO 的 std 标准化推导；Dr. GRPO 等去标准化损失的 native 坐标为 p，需对核除以 sin2ψ（Appendix C 已给出适配且保留增益）。
- 连续奖励需二值化或新方差稳定化映射；小 G（≤3）时两端死区覆盖大部分弧长，任何选择器均难。
- 采样器—目标对应假设提示梯度分离，忽略跨提示 transfer；全局漂移为一阶近似，强异质 transfer 时或需 per-cluster 漂移。
- 实验限于数学推理、模型 ≤7B、G ≤ 16；代码/agent 任务与更大规模待验证。
- pacing 编码单一偏好（yields within ε 内最硬目标）；重可靠性轻覆盖的场景可选其他目标。

## 研究启发与可借鉴点
1. **从优化器几何反推采样几何**：GRPO 标准化隐含 Fisher-Rao 弧长坐标；对任意 group-normalized 奖励可先求其 native 坐标，再在该坐标上设计采样/信念模型，避免坐标错配导致的噪声异方差与过自信。
2. **零方差组作为边界层似然**：不用抛弃而用闭式高斯界吸收进信念更新，使 filter 在 dead zone 仍校准；这一处理可迁移至任何二值 verifier 的 rollout 选择器。
3. **目标族 pacing 替代人工 curriculum**：在 pass^k ↔ pass@k 族上以 yield 约束自动推进目标硬度，无需手工调度 easy-to-hard；可与 bandit/贝叶斯优化课程结合形成更通用 pacing。
4. **Kalman 漂移项校正老化信念**：mobility-weighted drift v sin2μ 抵消"上次观测中间难度，现已变易"的系统偏差；该漂移项结构（依赖当前信念位置）可复用于其他在线难度估计。
5. **闭式 score = 核重叠 × 信息组概率**：避免训练辅助预测器；在成本敏感场景下（token/时间预算严格）可直接替换 DS/GRESO 等昂贵的事后过滤模块。

## 关键术语表
- **Fisher–Rao arc length ψ**：Bernoulli 分布族上的测地线弧长坐标，ψ = arcsin√p，使 Fisher 信息为常数 4。
- **ARCUS**：Arc-length CUrriculum Sampling，基于弧长高斯核与 Kalman 信念的 rollouts 高效 GRPO 提示选择器。
- **Zero-variance group**：组内 G 个响应全对或全错的组，GRPO 优势全零，但消耗 G 个 rollout。
- **Dead-zone boundary layer**：ψ 靠近 0 或 π/2 时零方差概率呈高斯衰减的区域，宽约 1/√G。
- **Homoscedastic evidence**：在 ψ 坐标下，单组观察的噪声方差恒为 1/(4n)，不随 p 变化。
- **Pass@k / pass^k**：k 次独立尝试中至少 1 次成功的指标与 reliability 对应目标，其弧长梯度为 N(arcsin(1/√(2k)), 1/(4k)) 附近的高斯。
- **Objective pacing**：沿目标族自动选最困难且 yield 不低于阈值的目标 ψ*，作为 curriculum 动态推进。
- **Corollary 1**：任何固定 inclusion prob w(ψ) 的采样器等价于在 GRPO 上沿 F_w(ψ) 做上升；因此采样器即目标控制器。

## 可复现要素
- **数据集**：DAPO-Math-17k（训练池），AIME24/25、AMC23、MATH500、Minerva、OlympiadBench（评测）；公开。
- **代码/权重**：论文声明 analytic figures 与 Table 4 由 code/ 目录脚本生成；正文未给出 GitHub 链接（以 arXiv 为准，发布时常见同步开源）。
- **关键超参**：B=256, G=8, 300 updates, lr=1e-6, clip=0.2, KL=0；ARCUS: ρ=0.25, ε=0.03, T=0.3, |Ψ|=41, Δ=0.005 rad, warm-up ≈⌈N/M⌉；框架 verl+vLLM, bf16, 8×H100。
