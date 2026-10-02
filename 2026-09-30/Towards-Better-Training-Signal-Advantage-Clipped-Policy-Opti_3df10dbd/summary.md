---
title: "Towards-Better-Training-Signal-Advantage-Clipped-Policy-Opti"
source: https://arxiv.org/pdf/2609.36816v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:35:39"
field: "大语言模型强化学习后训练"
keywords: ["RL post-training", "policy optimization", "importance sampling", "advantage clipping", "PPO", "GRPO", "mirror descent", "LLM reasoning"]
innovations: ["提出ACPO，直接对IS ratio与优势函数乘积进行截断以稳定梯度估计", "建立ACPO截断与梯度截断Policy Mirror Descent的理论联系并证明收敛性", "在数学推理基准上统一验证ACPO对PPO/GRPO的精度提升与训练加速效果"]
benchmarks: ["MATH500", "Minerva Math", "OlympiadBench", "AIME-like"]
---

# 论文速读：Towards-Better-Training-Signal-Advantage-Clipped-Policy-Opti

## 一句话总结
本文提出 ACPO（Advantage Clipped Policy Optimization），通过直接对重要性采样比率与优势函数的乘积 `r_t(θ)·Â_t` 进行截断，替代 PPO/GRPO 仅对 IS ratio 截断的做法，从而获得更稳定的梯度估计；理论证明了其梯度二阶矩不高于 PPO，并在多个数学推理基准上实现了稳定且高效的政策优化提升。

## 研究问题与动机
- **IS-ratio 截断缺乏原理性解释**：PPO 及其变体普遍采用对 IS ratio 单独截断的方式稳定训练，但策略更新的梯度估计实际由 `r_t(θ)·Â_t` 这一整体乘积决定，现有方法仅约束其中一个因子，对截断有效性的理解仍有限。
- **易题/难题导致梯度长尾扰动**：通过对不同难度提示下 `r_t(θ)·Â_t` 分布的分析发现，难题产生明显正向长尾、易题产生明显负向长尾，而中等难度提示的梯度贡献更集中，传统截断无法针对性抑制这两类极端信号。
- **训练稳定性与样本效率的矛盾**：RL post-training 依赖 on-policy 数据，复用 off-policy 数据引入不稳定；仅裁剪 IS ratio 难以充分抑制梯度方差，制约了训练效率和最终性能。
- **截断机制与优化理论的联系缺失**：梯度截断在深度优化中广泛应用，但其在策略优化（如 PPO/GRPO）中的角色缺乏从 Policy Mirror Descent 视角的系统解释。

## 核心贡献（创新点）
1. **提出 ACPO 截断目标**：将 IS ratio 与 advantage 的乘积直接截断至 `[−α, α]`，与仅截断 IS ratio 的 PPO/GRPO 在机制上存在本质区别——直接控制了每个样本梯度贡献的标量系数。
2. **梯度二阶矩理论保证**：在等保留概率假设下证明 `E[‖G_ACPO‖²] ≤ E[‖G_PPO‖²]`；若进一步假设两者梯度期望范数相近，则可推出 ACPO 的方差上界不高于 PPO，提供方差削减的严格保证。
3. **建立与梯度截断 PMD 的理论桥梁**：从带梯度截断的策略镜像下降出发，通过重要性采样近似、去掉显式 KL 正则项，导出 ACPO 目标函数，将优化领域的梯度截断与 RL post-training 截断统一到一个框架内。
4. **收敛性分析**：在标准 RL 设定下证明带梯度截断的 PMD 在有强凸正则化时线性收敛、无正则化时达到 `O(1/N)` 次线性收敛。
5. **实验验证与效率提升**：在 Qwen2.5-Math-7B 与 Qwen3（1.7B/4B/8B）上系统对比 PPO 和 GRPO，ACPO 在所有设置下均取得更高加权精度，同时实现显着的训练加速（相对 PPO 提速 6.25×，相对 GRPO 提速 1.7×）。

## 方法详解
- **ACPO 目标函数**：
  定义重要性采样比率 `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`，ACPO 目标为
  `I^ACPO(θ) = E_{(s_t,a_t)~π_θ_old}[ clip(r_t(θ)·Â_t, −α, α) ]`，
  其中 `Â_t` 可使用 PPO 型 GAE 或 GRPO 型组内归一化优势。该截断直接作用于每个 token 的梯度贡献标量系数。
- **与 PPO 目标的对比**：PPO 为 `min(r_t·Â_t, clip(r_t, 1−ε, 1+ε)·Â_t)`，仅限制 IS ratio 偏离范围；ACPO 则直接限制 `r_t·Â_t` 本身的大小，更贴合策略梯度幅度的实际调控需求。
- **Proposition 1（梯度二阶矩）**：在 `E[1_PPO] = E[1_ACPO]` 且梯度范数条件期望为常数 `C` 的假设下，证明 `E[‖G_ACPO‖²] ≤ E[‖G_PPO‖²]`；若两者期望梯度范数相近，则有 `Var(G_ACPO) ≤ Var(G_PPO)`。
- **PMD 视角的推导**：从带梯度截断的 PMD 更新出发，令 `λ_k(s) = α / max{α, ‖A^{π_k}(s,·)‖}` 控制步长缩放；通过重要性采样将状态层面优势转换为样本层面 `r_t(θ)·Â_t`，去除显式 KL 惩罚后得到 ACPO 目标，揭示两种截断的同构性。
- **收敛定理**：
  - **Theorem 1（有强凸正则）**：常数步长下目标间隙以 `ρ^N` 线性衰减，`ρ = max{γ, 1/(κ(1+ημ))}`；自适应步长下类似结果成立。
  - **Theorem 2（无正则）**：目标间隙满足 `F_N ≤ C_0 / ((1−γ)N)`，即 `O(1/N)` 次线性收敛。
- **超参选择依据**：基于 Figure 1 观察，难题（准确率 0–30%）呈正长尾、易题（70–100%）呈负长尾，取对称截断区间 `[−α, α]` 使优化行为趋近中等难度提示下的稳定状态；实验中设 `α = 2~3`。

## 实验与结果
- **数据集**：DAPO 训练集（约 17.4k 数学问题，来自 AoPS 及官方竞赛首页），使用 Math-Verify 自动判题。
- **模型**：Qwen2.5-Math-7B、Qwen3-1.7B、Qwen3-4B、Qwen3-8B。
- **评估基准**：MATH500、Minerva Math、OlympiadBench、AIME-like（含 AIME24/25、HMMT24/25、BRUMO25、AMC23、CMIMC25 共 230 题）；报告 Pass@1 加权精度。
- **关键超参**：PPO/GRPO 截断 ε=[0.2, 0.28]；PPO-AC α=3；GRPO-AC α=3（7B）或 2（Qwen3）；Prompt batch size=256/512；rollouts：PPO=1，GRPO=4（Qwen3）/8（7B）；Actor lr=1e−6（Qwen3-1.7B PPO 用 5e−7）。
- **主要结果（Table 1，最佳加权精度）**：
  - **Qwen2.5-7B**：PPO-AC 较 PPO 提升 +1.6pp；GRPO-AC 较 GRPO 提升 +1.4pp。
  - **Qwen3-1.7B**：PPO-AC 较 PPO 提升 +2.4pp（59.7% vs 57.3%）；GRPO-AC 提升 +1.5pp。
  - **Qwen3-4B**：PPO-AC 较 PPO 提升 +2.7pp（69.4% vs 66.7%）；GRPO-AC 提升 +0.6pp。
  - **Qwen3-8B**：PPO-AC 较 PPO 提升 +4.0pp（71.3% vs 67.3%）；OLYMPIAD 上 PPO-AC 达 70.5% vs PPO 63.7%（+6.8pp），AIME-like 48.8% vs 43.6%（+5.2pp）。
  - **最强结果**：Qwen3-8B+PPO-AC 在 MATH500 达 71.3%，在 OLYMPIAD 达 70.5%，均显著超越所有基线。
- **训练效率**：ACPO 平均训练步数加速 6.25×（相对 PPO）和 1.7×（相对 GRPO）；PPO-AC 在 Qwen3-4B/8B 上约 40 步即可达到 PPO 200 步的性能。
- **熵稳定性**：ACPO 有效缓解策略熵坍缩；GRPO-AC 熵在整个训练中保持稳定，PPO-AC 熵下降速度远低于 PPO。
- **框架实现**：基于 VERL 框架，所有实验关闭 KL 惩罚项（与近期工作一致）。

## 相关工作脉络
- **PPO（Schulman et al., 2017）**：原始 IS ratio 截断方法；ACPO 在机制上更直接地控制梯度贡献幅度，而非仅约束 ratio 偏离度。
- **GRPO（Shao et al., 2024）**：无 critic 的组内归一化优势方法；ACPO 可无缝集成到 GRPO 中形成 GRPO-AC，进一步利用 token 级截断稳定性。
- **DAPO（Yu et al., 2025）**：提出 Clip-Higher 策略（不对称截断 `[0.2, 0.28]`）鼓励探索；ACPO 采用对称截断，从梯度二阶矩角度提供不同维度的稳定保证。
- **CISPO（MiniMax et al., 2025）**：对 IS 权重截断但保留对应梯度贡献；与 ACPO 的本质差异在于 ACPO 直接截断 `r_t·Â_t` 乘积，而非先截权重再补偿。
- **GAPO（Jia et al., 2026）**：根据轨迹优势自适应调整 IS 截断范围；ACPO 则从固定截断区间出发，通过理论分析证明其梯度性质更优。
- **Policy Mirror Descent（Tomar et al., 2022；Song et al., 2026）**：前者提出 MDPO 近似求解信任域子问题，后者通过 Bellman 方程重构 PMD；本文的差异化贡献在于用 PMD 框架解释并推导出 ACPO 的截断机制，而非直接导出新的算法形式。

## 局限性与未来方向
- **截断阈值 α 需经验调参**：不同模型规模和数据集下 α 的最优值存在差异（如 Qwen3 用 α=2，Qwen2.5-7B 用 α=3），缺乏自适应选择 α 的理论指导。
- **忽略 KL 正则项的实验设置**：为凸显截断机制的作用，实验中去除了 KL 惩罚项，这在部分场景下可能与实际需求不符，其影响范围有待进一步探讨。
- **优势估计依赖**：PPO 侧使用价值网络估计 token 级优势，引入额外方差来源；GRPO 侧使用组内归一化，可能在高方差场景下低估极端优势信号。
- **理论假定的现实性**：Proposition 1 要求等保留概率与梯度范数条件期望为常数，这些假设在实际训练中难以精确满足，方差削减结论依赖近似的额外假设。
- **仅验证于数学推理任务**：目前实验集中在数学推理领域，对代码生成、对话等其他 reasoning-type 任务的泛化能力尚待验证。
- **未来方向**：（1）设计自适应 α 调度策略；（2）结合难度感知 curriculum 进行动态截断；（3）将 ACPO 扩展至多轮对话和工具调用场景；（4）与 DAPO/Clip-Higher 等非对称截断机制进行更系统的比较研究。

## 研究启发与可借鉴点
- **梯度贡献乘积截断思想可迁移**：将 IS ratio 与优势的乘积作为统一截断单元的思路，可推广到其他依赖重要性采样的离线 RL 或混合 on/off-policy 算法中，作为通用稳定化组件。
- **难度感知的截断动机具有方法论价值**：通过可视化不同难度提示下的梯度系数分布来指导算法设计，这一"诊断→干预"的研究范式值得在其他 RL post-training 工作中复用。
- **PMD 框架为截断机制提供理论解释**：本文通过 Policy Mirror Descent 推导 ACPO 目标，展示了如何用优化理论统一解释经验性工程技巧，该方法论可用于分析其他截断变体（如 DAPO、GAPO）的理论性质。
- **熵坍缩分析与截断机制的关联**：ACPO 能缓解策略熵快速衰减的现象，提示未来可将被控熵作为辅助目标与截断机制联合设计，以平衡 exploitation 与 exploration。
- **可复用的实验对照设计**：在同一训练框架（VERL）、相同数据集和相同超参（除截断方式外）下对比 PPO/GRPO/ACPO 变体，消除了混杂因素，该对照方案可直接借鉴于后续截断方法的实验评估。

## 关键术语表
- **IS ratio（重要性采样比率）** `r_t(θ)`：当前策略与新策略在相同状态-动作对上的概率比值，用于 off-policy 数据的重要性加权。
- **Advantage（优势函数）** `Â_t`：衡量在状态 s 采取动作 a 相对于策略平均表现的增量收益，决定策略更新的方向和幅度。
- **ACPO（Advantage Clipped Policy Optimization）**：直接对 IS ratio 与 advantage 的乘积进行对称截断的策略优化算法。
- **PMD（Policy Mirror Descent）**：将镜像下降扩展到 MDP 策略优化的理论框架，以 Bregman 散度度量策略邻域。
- **Gradient clipping（梯度截断）**：优化中限制梯度范数防止爆炸的常用技术；本文将其与策略优化的截断机制建立理论联系。
- **Clip-Higher（DAPO）**：使用非对称截断范围 `[ε_low, ε_high]` 以鼓励探索的 PPO 改进策略。
- **GAE（Generalized Advantage Estimation）**：通过 λ 加权 TD 误差序列估计优势函数的方法，兼顾偏差与方差。
- **Entropy collapse（熵坍缩）**：策略分布过早收敛到高置信度单一模式，导致探索能力丧失的训练现象。

## 关键超参汇总
- **ACPO 截断阈值 α**：Qwen3 系列设为 2，Qwen2.5-Math-7B 设为 3。
- **PPO/GRPO 基线截断 ε**：ε_low=0.2，ε_high=0.28（GRPO）。
- **Actor 学习率**：AdamW 1e−6（Qwen3-1.7B PPO 用 5e−7）。
- **Rollouts**：PPO=1，GRPO=4（Qwen3）/ 8（Qwen2.5-7B）。
- **Prompt batch size**：256（Qwen3 GRPO）/ 512（其余）。
- **Actor mini-batch size**：64（Qwen3 GRPO）/ 128（其余）。
- **KL 惩罚**：全部关闭。
- **训练步数**：100–300 步（视模型规模而定）。

## 可复现要素
- **数据集**：DAPO 训练集（约 17.4k 数学问题，来自 AoPS 及官方竞赛首页）——论文中引用，是否完全开源需另行核实。
- **代码**：基于 VERL 框架实现，论文未明确公开 ACPO 独立代码仓库。
- **权重**：未开源微调后的模型权重。
- **环境**：8× NVIDIA H100（7B/8B）、4× H100（4B）、2× H100（1.7B），参数与优化器卸载开启。
- **关键超参**：见上节"关键超参汇总"，均可从 Table 2 直接复现。
