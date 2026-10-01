---
title: "ReSPO-Reshaped-Sequence-Policy-Optimization-for-Gradient-Sta"
source: https://arxiv.org/pdf/2609.35433v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:57:08"
field: "大语言模型强化学习优化"
keywords: ["policy optimization", "off-policy learning", "gradient starvation", "sequence-level importance weighting", "alpha-divergence", "RLVR", "rollout reuse"]
innovations: ["揭示 clipping policy optimization 中符号依赖的梯度饥饿机制并提出序列级两分支核函数设计原则", "基于alpha-散度变分目标与KL指数倾斜推导可解析确定的正负分支光滑重塑核", "在密集与MoE模型上系统验证rollout复用场景下ReSPO对训练效率与泛化性能的提升"]
benchmarks: ["DAPO-MATH-17k", "AIME 2025", "AIME 2024", "AMC 2023", "OlympiadBench", "MinervaMath", "MATH-500"]
---

# 论文速读：ReSPO-Reshaped-Sequence-Policy-Optimization-for-Gradient-Sta

## 一句话总结
论文针对离策略策略优化中的梯度饥饿问题，提出 ReSPO（Reshaped Sequence Policy Optimization），用基于 α-散度的光滑两分支序列级核函数替代 GRPO/GSPO 的硬截断（clipping），在 rollout 复用导致策略漂移的场景下显著提升数学推理训练效率与最终泛化性能。

## 研究问题与动机
- **离策略漂移加剧**：RLVR（如 GRPO）为节约计算成本而复用同一批 rollout 进行多次策略更新，当前策略 π_θ 与生成数据的旧策略 π_old 差异累积，序列重要性权重 W = π_θ(o|q) / π_old(o|q) 偏离 1，导致传统 clipped policy optimization 失效。
- **符号依赖的梯度饥饿**：对正响应（Â > 0），低 W 尾部（当前策略更不倾向生成）的梯度权重随 w_t 线性缩小至 0；对负响应（Â < 0），高 W 尾部（当前策略更倾向生成）未被 clipping 限制，梯度权重无界放大，两者分别造成"信息丰富的正向长轨迹失学"与"极端负向轨迹主导更新"。
- **既有方法的尾部缺陷**：GRPO/GSPO 的 hard clip 无法同时满足 φ^(+)(0) > 0 与 φ^(-)(0) = φ^(-)(∞) = 0；VESPO 虽两端都趋于 0，但正分支在 W→0 处同样为零，仍会饿死低 W 正向响应。
- **序列级失配**：token 级裁剪将序列级奖励信号拆解到每个 token，随着序列长度增加噪声放大；GSPO 的长度归一化 s = W^(1/|o|) 引入长度相关的偏差（短/长序列在相同平均 log-ratio 下获得相同权重，但真实序列权重相差指数倍）。

## 核心贡献（创新点）
- **揭示离策略场景下的符号依赖梯度饥饿现象**：明确刻画 clipping 优化中正/负响应在重要性权重双尾部的梯度分配失衡机制，指出其根因在于线性权重与硬边界无法同时满足四个尾部条件。
- **基于 α-散度变分推导的两分支光滑核**：从 dual-proximity objective 出发，利用 α-divergence 生成 power-mean 插值核 φ_0，再经 KL 投影施加二阶矩约束导出指数倾斜，得到可微、无硬截断的序列级重塑核 φ(W)。
- **正/负分支的针对性尾部设计**：正分支取 α^(+) = 2 使 φ^(+)(W) = (1+W)/2 · exp(1-W)，满足 φ^(+)(0) = e/2 > 0 且严格递减至 φ^(+)(∞) = 0；负分支取 α^(-) = 1 极限形式 φ^(-)(W) = √W · exp(2(1-√W))，满足 φ^(-)(0) = φ^(-)(∞) = 0，两端均被抑制。
- **参数解析可推导而非纯调参**：β^(±) = 0.5 由对称插值确定，λ^(+) = 1/(1-β^(+)) = 2 由 φ^(+)'(0)=0 的唯一驻点条件确定，λ^(-) = 2 由两分支在 W=1 处一阶导数匹配确定，三者均可从尾部分析与光滑性要求解析给出。
- **在密集与 MoE 模型上系统验证 rollout 复用鲁棒性**：在 Qwen3-1.7B 与 Qwen3-30B-A3B 上，rollout 复用比 N ∈ {8, 16, 32}，ReSPO 在 12 个模型–N 组合中 10 个取得最早期峰值与最晚期均值的最优或接近最优，并在 AIME/AMC 等高难度竞赛基准上实现最大提升。

## 方法详解
- **问题形式化**：在 GRPO 设定下，给定 prompt q 采样 G 个 response，计算组内归一化优势 Â_i。序列重要性权重 W_i = ∏_t w_{i,t}。现有 GRPO/GSPO 在 unclipped 区域梯度正比于 Â·ρ·∇logπ_θ，其中 ρ 为 token 级或长度归一化序列级权重，线性放大双尾的不均衡。
- **权重重塑的测度解释**：将 raw importance weight W 替换为 φ(W)，等价于定义未归一化测度 τ(o) = μ(o)·φ(W(o))，其中 μ = π_old。目标梯度变为 E_μ[φ(W)·Â·∇logπ_θ]，设计 φ 即设计测度 τ 的梯度分配。
- **α-散度变分推导**：采用 Bregman generator f(x) = x^α / [α(α-1)]，求解 min_τ≥0 (1-β)D_f(τ‖μ) + βD_f(τ‖π)，得到幂均值插值 τ_0^*(o) = [(1-β)μ(o)^{α-1} + βπ(o)^{α-1}]^{1/(α-1)}，对应的无约束核 φ_0(W) = [(1-β) + βW^{α-1}]^{1/(α-1)}，满足 φ_0(1)=1。
- **KL 投影的指数倾斜**：在泛化 KL 散度 D_KL(τ‖τ_0^*) 下，施加二阶矩约束 Σ_o τ(o)φ_0(W(o)) ≤ C_1 与质量约束 Σ_o τ(o) ≤ C_2，通过 KKT 条件得到 τ^*(o) ∝ τ_0^*(o)·exp(-λφ_0(W(o)))，还原为重塑核 φ(W) = φ_0(W)·exp(λ(1-φ_0(W)))。α-divergence 决定插值形状，KL 投影提供平滑乘法倾斜而非硬边界。
- **两分支核的具体形式**：
  - 正分支（Â ≥ 0，α^(+) = 2，β^(+) = 0.5）：φ_0^(+)(W) = (1+W)/2，φ^(+)(W) = [(1+W)/2]·exp(1-W)，φ^(+)(0) = e/2 ≈ 1.359，φ^(+)(1)=1，φ^(+)(∞)=0，在 W=0 处取全局最大值。
  - 负分支（Â < 0，α^(-) = 1，β^(-) = 0.5）：φ_0^(-)(W) = W^{0.5} = √W，φ^(-)(W) = √W·exp(2(1-√W))，φ^(-)(0)=φ^(-)(∞)=0，峰值位于 W* = 1/4，φ^(-)(1)=1。
  - 两分支在 W=1 处函数值与一阶导数均匹配（φ^(+)'(1) = φ^(-)'(1) = -1/2），保证 on-policy 附近的光滑过渡。
- **策略梯度估计**：最终 ReSPO 梯度为 ∇_θ I_ReSPO = E[ (1/G) Σ_i sgn(φ(W_i; Â_i)) · Â_i · Σ_t ∇_θ log π_θ(o_{i,t}|q, o_{i,<t}) ]，其中 φ 经 stop-gradient 截断后作为序列级标量缩放系数，避免对权重本身求梯度的数值不稳定。
- **数值实现**：log W_i 先 clamp 至 [-20, 20]，正分支直接计算 (1+W)/2·exp(1-W)；负分支在 log 空间计算 log φ^(-) = β·log W + λ(1 - exp(β·log W))，避免浮点溢出。GPU 上同时计算两分支并用 sign(Â) mask 选取。

## 实验与结果
- **实验设置**：基于 verl 框架，异步 vLLM rollout，mini-batch M=32，每组 G=8 个 rollout，GRPO 优势估计，学习率 1e-6，无 KL 惩罚。训练数据 DAPO-MATH-17k，提示上限 1024 tokens，响应上限 Qwen3-1.7B 为 15360、Qwen3-30B-A3B 为 8192（30B 加软长度惩罚：≤4096 无惩罚，4096–8192 线性降至 -1）。rollout 复用比 N ∈ {8, 16, 32}，总策略更新 1024 步。评估基准：AIME 2025、AIME 2024、AMC 2023、OlympiadBench、MinervaMath、MATH-500，temperature=1.0、top-p=0.7，每题生成 16（竞赛题）或 4（其他）个 completion，用 Math-Verify 评分。
- **基线对比**：GRPO（clip ε=0.2）、GSPO（ε_low=3e-4、ε_high=4e-4）、VESPO（β^(+)=2、λ^(+)=3、β^(-)=3、λ^(-)=2）。
- **训练性能（ normalized training score = (score+1)/2 ）**：
  - 早期峰值（前 256 步）：ReSPO 在 1.7B 的 N=8/16/32 分别达到 0.192/0.166/0.157，10B 在 N=16/32 分别达到 0.445/0.418（N=8 时 GRPO 以 0.478 略优 1.0 pp）。
  - 晚期均值（最后 128 步）：ReSPO 在全部 12 个 model–N 组合中的 10 个取得最高或统计重叠的最高，相对最强 clipped 基线的提升幅度：1.7B 的 N=32 达 7.9 pp（0.276 vs 0.197），30B 的 N=32 达 8.4 pp（0.588 vs 0.504）。
- **评估性能（pass@1 macro average over 6 benchmarks）**：
  - N=8 时各方法差异在 bootstrap 误差范围内；N=16/32 时 ReSPO 全面领先。
  - 1.7B：N=16 均值 0.343±0.015，N=32 均值 0.345±0.015，均超过最强基线。
  - 30B：N=16 均值 0.525±0.018，N=32 均值 0.512±0.018。
  - 竞赛题（AIME25+AIME24+AMC23）平均：30B 在 N=8/16/32 相对最强基线分别提升 6.3/5.8/3.4 pp。
- **按响应长度分桶准确率**：10 分位数桶（Q1–Q9 无截断、Q10 为 16384 上限）下，ReSPO 在 30/30 个桶中有 24 个最高；N=32 时 Q1–Q7 领先基线 8.4–24.4 pp，表明优势来自学习质量而非仅长度分布偏移。
- **多种子鲁棒性（N=32 三 seed）**：1.7B 晚期均值 27.07%±0.54 pp，分别超过 VESPO/GRPO/GSPO 4.07/7.37/8.66 pp；30B 晚期均值 57.92%±0.95 pp，分别超过 5.62/7.52/8.64 pp；评估均值 41.94%±0.47 pp，分别超过 3.17/3.51/7.24 pp，标准差远小于性能增益。
- **与 Routing Replay（R3）的组合**：在 30B N=16 设置下，ReSPO+R3 晚期均值 0.582±0.008，较单独 ReSPO 的 0.563±0.021 提升 1.84 pp，时间波动从 2.11 pp 降至 0.75 pp，响应长度从 1992 降至 1213，截断率从 3.505% 降至 0.011%，表明两方法正交可叠加。

## 相关工作脉络
- **GRPO / DAPO / REINFORCE++**：均为 token 级 clipped or normalized policy optimization，关注 group-relative advantage 计算与采样策略，但未显式处理序列级重要性权重在离策略漂移下的尾部失配问题；ReSPO 聚焦序列级重塑，保留 IS 显式处理 staleness。
- **GSPO**：首次将序列级几何平均权重 s = W^(1/|o|) 引入 MoE 训练以提升稳定性，但硬 clip 仍保留梯度饥饿；ReSPO 明确指出 s 的长度归一化引入长度相关偏差，改用未归一化的 W 避免此问题。
- **VESPO**：基于 KL 散度导出 φ_KL(W) = W^β·exp(λ(1-W))，两端均趋于 0，适合负分支但不满足正分支 φ^(+)(0)>0 的要求；ReSPO 将 VESPO 的 KL 变分框架推广至 α-divergence 族，并证明单一 α 无法同时满足正负分支条件。
- **REAL（Zhai et al., 2026）**：将梯度饥饿重构为分类问题，放弃 IS；ReSPO 保留 IS 以维持 off-policy staleness 的显式编码，适应 rollout 复用场景下每步漂移累积的需求。
- **Routing Replay（Ma et al., 2025）/ 截断 IS（Liu et al., 2025）**：前者对齐训练-推理 router 决策，后者限制 IS 的尾部，均为系统层稳定性方案；ReSPO 与二者正交，专注于梯度权重分配的算法层重塑。
- **TRPO / PPO / 概率平滑 / 自适应门控**：通过 KL 约束或 soft clip 控制信任域；ReSPO 采用可分离的 power-Bregman 几何，在序列级构造两分支 kernel，不依赖额外的 value function 或硬截断边界。

## 局限性与未来方向
- **计算预算限制**：30B 实验仅使用 8192-token 响应上限与软长度惩罚，未测试更长响应与更多更新步数；扩展至更长上下文与更大 N 需数周 8×H200 GPU。
- **超参未穷举搜索**：α、β、λ 主要基于尾部分析与导数匹配解析确定，未做网格搜索；不同模型规模、复用比、响应长度 regime 可能存在更优组合。
- **仅评估数学推理任务**：实验集中在 DAPO-MATH 与 AIME/AMC 等数学基准，未验证于代码生成、指令遵循或多模态 RLVR 场景。
- **未覆盖极端长尾漂移**：log W clamp 至 [-20, 20]，当漂移极大时 kernel 行为由有限范围近似决定，理论尾部性质与实际数值实现存在微小偏差。
- **MoE 场景仅验证单组消融**：Routing Replay 组合实验仅在 30B N=16 下完成，未系统研究 ReSPO 参数与路由策略超参的联合调优。

## 研究启发与可借鉴点
- **尾部条件驱动 kernel 设计的思路**：从 φ^(+)(0)>0、φ^(-)(0)=φ^(-)(∞)=0 四个尾部不等式出发反推核函数形式，而非经验试错；该"需求→解析构造"范式可迁移至其他 IS-based 优化问题（如 off-policy RL、离线强化学习）的权重重塑设计。
- **Bregman 散度族的核函数推导流程**：先以 α-divergence 求解 unconstrained interpolation，再以 KL projection 施加矩约束导出指数倾斜，两步分离几何设计与方差控制，可复用于构建其他光滑 trust-region kernel。
- **正负分支不对称参数的必要性**：同一 α 无法兼顾正负尾部的相反需求，必须按 Â 的符号选择不同分支；这一洞察提示在 advantage-sign-dependent weighting 场景中，对称参数化可能掩盖结构性缺陷。
- **实验设计上对 roll-out reuse ratio N 的系统扫描**：将 N 作为衡量离策略漂移程度的控制变量，展示方法在漂移加剧时的衰减曲线；该设计可作为评估任何 off-policy correction 方法稳定性的标准协议。
- **与 Routing Replay 等系统级优化的正交性验证**：在 MoE 实验中单独消融 R3，证明算法层权重重塑与工程层路由对齐可独立叠加；提示后续工作可探索"算法+系统"双层组合的规模化收益。

## 关键术语表
- **RLVR（Reinforcement Learning from Verifiable Rewards）**：利用确定性标量奖励（如代码执行、数学答案比对）替代人工奖励模型进行 LLM 策略优化的训练范式。
- **Gradient Starvation（梯度饥饿）**：clipped importance-weighted objective 在重要性权重低尾抑制正向响应梯度、在高尾放任负向响应梯度放大的符号依赖失配现象。
- **Sequence-level Importance Weight（W）**：整条 response 的 token 级比例乘积 W = π_θ(o|q) / π_old(o|q)，度量当前策略与数据生成策略在序列粒度的偏移程度。
- **α-divergence（α-散度）**：由 Bregman generator f(x)=x^α/[α(α-1)] 诱导的距离族，本文用于构造行为策略与目标策略之间的幂均值插值测度。
- **Exponential Tilt（指数倾斜）**：通过对 unconstrained kernel 施加二阶矩约束并进行 KL 投影，得到的乘法修正因子 exp(λ(1-φ_0(W)))，实现平滑方差控制。
- **Rollout Reuse Ratio（N）**：同一批 old-policy 生成的 response 被连续用于 N 个 mini-batch 的策略更新次数，N 越大 off-policy drift 越严重。
- **Stop-gradient（sg）**：对 reshaping kernel φ(W) 的结果断开梯度传播，使其仅作为标量缩放系数作用于策略梯度，避免权重本身的梯度噪声。
- **Power-mean Interpolation（幂均值插值）**：由 α-divergence 一阶条件导出的 τ_0^* = [(1-β)μ^{α-1} + βπ^{α-1}]^{1/(α-1)}，在 μ 与 π 之间形成介于几何平均（α→1）与算术平均（α=2）之间的测度过渡。

## 可复现要素
- **数据集**：DAPO-MATH-17k（训练），AIME 2025、AIME 2024、AMC 2023、OlympiadBench、MinervaMath、MATH-500（评估）；均公开可用。
- **模型权重**：Qwen3-1.7B-Base、Qwen3-30B-A3B-Base（开源底座）。
- **代码/权重**：论文声明附带的 code artifact 包含 ReSPO 实现与 dense/MoE 实验启动配置；附录 A–G 提供完整推导、log-space 实现、超参表与复现脚本说明。
- **关键超参**：α^(+)=2、α^(-)=1、β^(+)=β^(-)=0.5、λ^(+)=λ^(-)=2；GRPO clip ε=0.2；GSPO ε_low=3e-4、ε_high=4e-4；VESPO β^(+)=2、λ^(+)=3、β^(-)=3、λ^(-)=2；学习率 1e-6，M=32，G=8，N∈{8,16,32}，总更新 1024 步。
