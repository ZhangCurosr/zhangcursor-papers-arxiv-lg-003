---
title: "PERSISTENT-PARTNERS-RAISE-PRICESAMONG-LEARNING-AGENTS"
source: https://arxiv.org/pdf/2609.35402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:54:03"
field: "算法合谋与多智能体定价"
keywords: ["algorithmic collusion", "pricing agents", "multi-agent reinforcement learning", "platform design", "spontaneous coupling", "LLM agents", "Bertrand duopoly"]
innovations: ["首个预注册随机化实验证明配对持久性 causal 地提高学习代理价格", "提出净执行度量 E_net 并证明约 1/3-1/2 表观惩罚实为静态最佳响应", "在对手价格隐藏且惩罚不可能的条件下仍观察到价格上升"]
benchmarks: ["Calvano et al. (2020) calibrated Bertrand duopoly"]
---

# 论文速读：PERSISTENT PARTNERS RAISE PRICES AMONG LEARNING AGENTS

## 一句话总结
本文通过预注册随机实验证明：在Bertrand双寡头定价环境中，平台维持代理与同一对手的长期配对（而非每轮随机重配）会显著提高学习代理达成的价格水平，即便在对手价格完全不可见、报复惩罚无从发生时该效应依然存在。

## 研究问题与动机
1. **核心问题**：平台决定"谁面对谁"的匹配机制是否 causal 地影响学习代理的定价结果，以及这一效应是否伴随可检测的"惩罚"行为。
2. **现有方法不足**：此前算法合谋研究主要变化算法类型（由 firms 选择），从未有人随机化"对手配对"这一由平台控制的 lever，导致因果关系无法确立。
3. **监管盲区**：当前合谋审计多依赖"惩罚模式"检测，而隐藏对手价格的条件下惩罚机制完全不可能存在，仅检测惩罚会系统性漏检。
4. **LLM 可迁移性未经验证**：Fish et al. (2024) 展示了未训练 LLM 可达超竞争价格，但无法区分"训练过程"的贡献，且未探究环境结构因素的影响。

## 核心贡献（创新点）
1. **首个预注册随机化实验测试"匹配 persistence"作为平台 lever**：通过固定 vs 随机重配的对仗设计，在所有20对 run 和25个复现块中一致证实效应存在，本质区别于既往观察性研究。
2. **揭示"无惩罚能力下仍可涨价"的因果效应**：在对手价格隐藏条件下，价格模块根本无法感知降价行为，更无法实施报复，但残留价格仍显著上升且持续至训练结束。
3. **提出净执行度量 E_net 及其静态基准校正**：从原始强制偏差探针中扣除静态最佳响应（static best-responder）的贡献，证明约 1/3 至 1/2 的"表观惩罚"实为普通最优反应；二者联合拒绝作为 H6 判定标准。
4. **通过事前承诺预测检验自发耦合（spontaneous coupling）机制**：两个持续相遇的学习者从同一价格序列中学习，价值估计同步漂移，耦合签名（价值表相关性差分）在隐藏对手条件下更强，支持但不证明该机制。
5. **扩展至未训练 LLM 的探索性发现**：Qwen2.5-7B/14B 在对手价格隐藏时复现效应（7B 在 fresh blocks 上复制成功），但 Mistral-7B 和 OLMo-2-7B 不显示任何效应，揭示模型族差异。

## 方法详解
- **环境**：Calvano et al. (2020) 校准的差异化产品 Bertrand 双寡头，logit 需求，差异化参数 μ=0.25，33 档价格网格（静态 Nash 在 bin 6，联合垄断在 bin 26），16 个并行市场，8000 轮，延续概率 0.98。
- **Agent 架构**：两部分组成——(1) Qwen2.5-0.5B-Instruct 语言骨干 + LoRA（rank 16）+ 线性价格 head（logits 绝对值上限 c=5.5）；(2) 4×4 表格价值模块（对 bucketed 上一轮对手价格和己方价格），另加1个无历史状态和4个对手隐藏状态，采用 Calvano 规则的离线 Q-learning（η_Q=0.15, δ=0.95）。行动 logit = head输出 + 16.5[onehot(argmax_a Q(s,a)) + 0.5·Q̃(s,·)]。探索率 ε=0.3×0.9994^t。
- **三因素实验设计**：每个因子为二水平 availability 变量——(a) **配对持久性**：fixed（全程同一对手）vs shuffled（每 episode 末随机重配）；(b) **可观察性**：visible（对手价格进入 prompt 和表格状态）vs hidden（均不出现）；(c) **通信**：none vs free_text（每轮6 token 消息）。8 个 cell × 5 个 seed block = 40 runs。
- **强制偏差探针（forced-deviation probe）**：在每个 checkpoint 冻结策略，从4个起始历史出发各玩40轮 greedy 到达休息点，然后分出 sham 分支（无干预）和 deviation 分支（一方被强制降价 k=4 bin），观察25轮。水平度量 Δ = (π - π_Nash)/(π_mon - π_Nash)，范围 [0,1]。
- **净执行度量 E_net**：E_net^r = E − E_myo pic^r，其中 E 为原始偏差分支与 sham 分支之间的 signed AUC（bins·rounds），E_myo pic^r 为对相同偏差路径逐轮施加静态最佳响应（BR₁：只看单步；BR_K：看 K=5 轮历史窗口）得到的面积。定理证明：若对手价格等于匹配静态参考经因果绝对可和线性滤波（静态增益 G=1），则极限下 E_net=0。H6 要求两个基准同时拒绝才判定。
- **统计推断**：以 block 为单位做 sign-flip randomization test（2^20 种模式精确枚举），学生化统计量 T=mean_b(C_b)/(sd_b(C_b)/√5)。H1（α_Δ=0.030，单侧）检验 C_Δ>0；H6（α_E=0.010）要求双基准同时拒绝。SESOI_Δ=0.10，SESOI_E=4 bins·rounds。

## 实验与结果
- **主结果（H1，预注册）**：C_Δ = +0.274，95% CI [+0.195, +0.350]，p=0.00034，20 对 run 全部为正，判定为 substantial。
- **执行结果（H6，预注册）**：E_net^BR1 = +1.48，CI [−1.50, +4.75]，p=0.145；E_net^BRK = +1.36，CI [−1.25, +4.50]，p=0.156，双基准均未拒绝，判定为 inconclusive（计划功效约 0.5）。
- **纯表格学习器复现（25 block）**：C_Δ = +0.306，CI [+0.270, +0.340]，全部 25 块为正。一对一永久伙伴对比约三个伙伴：+0.229 vs +0.048，表明"与单一对手持续相遇"是关键。
- **隐藏对手条件**：resting price 上升幅度不小于可见条件（最终 checkpoint Δ=+0.52 vs +0.15），且上升持续至训练结束；可见对手条件下最终 checkpoint 效应与零无显著差异。
- **净执行校正效果**：在 fixed/visible/none cell，静态最佳响应解释原始 E=6.99 中的 48%（净值 3.61）；在 shuffled/visible/free_text cell，原始 E=5.16 反转为 −2.04。
- **Swap opponent 检验**：固定 agent 与未合作过的对手配对，Δ 下降 −0.177，p=0.0003（Holm 阈值 0.005），但不能分离"对特定伙伴的适应"与"面对陌生人的成本"。
- **LLM 扩展**：Qwen2.5-7B 在对手价格隐藏时 C_Δ=+0.384（8/8 block 为正），fresh blocks 上复现为 +0.442；Mistral-7B（垄断天花板 0.97）和 OLMo-2-7B（0.84）均不显示效应。
- **耦合签名**：固定配对下双方价值表相关性高于非伙伴配对（隐藏对手：+0.17，可见对手：+0.013）。

## 相关工作脉络
1. **Calvano et al. (2020)**：开创性工作，证明表格 Q-learning 在校准双寡头中可达超竞争价格并读出 reward-punishment；本文在其实验设置上首次随机化"谁面对谁"，并引入净执行校正。
2. **Fish et al. (2024)**：展示未训练 LLM 在 Bertrand 双寡头中达超竞争价格并读出惩罚-奖励模式；本文补充了训练过程的因果检验，并证明配对持久性是独立于算法类型的平台 lever。
3. **Banchio & Mantegazza (2022)**：提出 spontaneous coupling 机制解释无威胁下的超竞争价格；本文通过配对持久性 vs 随机重配的对比和耦合签名预测提供因果证据支持（虽未完全确立）。
4. **Hu et al. (2020) / Lupu et al. (2021) / Strouse et al. (2021)**：MARL 中的 cross-play 评估与合作失败文献；本文反转其结论方向——固定伙伴raising价格而非降低 generalization 能力。
5. **Hansen et al. (2021) / Waltman & Kaymak (2008)**：独立算法在 Cournot/无对手观测下达超竞争价格；本文扩展至 Bertrnad 并系统比较可见/隐藏对手条件。
6. **Weis et al. (2026)**：通过 in-context co-player inference 实现合作；本文预测其可替代 persistent partner 且效应更小，为反向预测。

## 局限性与未来方向
1. 所有对比均条件于单一环境（M=4、8000轮、0.5B模型、单一 prompt），推广至更大模型、更多对手、真实市场受限。
2. 持久性与 per-partner 经验和更新频率混杂（尽管表格学习器和 zero-head 消融部分分离）；无法完全排除学习频率 confound。
3. H6 功效不足（约 0.5），净执行度量在不同计算机上可能变号；无法确立惩罚是否真正存在。
4. swap opponent 测试无法分离"伙伴特定适应"与"面对陌生人的成本"。
5. 隐藏对手条件同时缩小了表格状态空间，效应可能部分源于状态压缩而非纯配对持久性。
6. 未建模重匹配的福利成本和实际部署成本。

## 研究启发与可借鉴点
1. **平台 re-matching 作为反合谋 lever**：实验证明随机重配可有效降低超竞争价格，尤其当对手信息隐藏时效应更持久；为平台设计提供直接政策建议。
2. **强制偏差探针的静态基准校正方法**：E_net 定义及 BR₁/BR_K 双基准设计可复用于其他算法合谋审计场景，避免将普通最优反应误判为惩罚。
3. **事前承诺预测（committed predictions）验证机制**：在探索性分析前提交预测并记录 commit hash，为"数据窥探"风险提供解决方案，可在本团队后续机制研究中复用。
4. **耦合签名作为机制诊断工具**：价值表相关性差分（partners vs non-partners）是一种轻量级的机制诊断指标，可迁移至其他 MARL 合作/合谋研究。
5. **模型族敏感性差异的发现**：Qwen2.5 复现效应而 Mistral/OLMo 不复现，提示在 LLM 代理研究中需报告模型族，不能假设泛化性。

## 关键术语表
**Spontaneous coupling（自发耦合）**：两个持续相遇的学习者从同一价格序列中学习，各自价值估计同步漂移上升，无需显式威胁或反应即可产生高价。
**E_net（净执行度量）**：强制偏差探针中对手价格响应的面积减去静态最佳响应对应面积，用于分离真正的策略性惩罚与普通最优反应。
**Bertrand duopoly（伯川德双寡头）**：两个厂商同时选择价格、消费者选择低价产品的差异化产品竞争模型，静态纳什均衡价格为边际成本。
**Forced-deviation probe（强制偏差探针）**：冻结策略后从休息点出发，强制一方大幅降价并观察25轮的探测方法，用于评估对手是否报复。
**SESOI（最小感兴趣效应量）**：预注册中设定的"值得检测的最小效应"阈值，用于区分 substantial/precise null/inconclusive 判决。
**Randomization inference（随机化推断）**：以 block 为单位做 sign-flip permutation test，在 sharp null 下精确、在 weak null 下渐近有效。
**Static best-responder（静态最佳响应基准）**：逐轮对偏离者的实际价格路径施加无记忆的最优反应函数，作为"非策略性"参照。
**Calvano rule**：表格 Q-learning 的离线更新规则 Q(s,a) ← (1−η)Q(s,a) + η(r + δ max_{a'} Q(s',a'))，是本文价值模块的核心学习算法。

## 可复现要素
- **数据集**：Calvano et al. (2020) 校准的 Bertrand 双寡头仿真环境（非真实数据集，为标准 benchmark）。
- **代码/权重**：分析代码、40 次确认 run 输出、探索性 run 输出、block launch scripts（含 seed 分配）、机制测试预测及 commit hash、figure scripts 均在会议投稿补充档案中，可向作者索取；预注册代码冻结于 OSF（https://osf.io/98bx5，注册日 2026-09-08），SHA-256: b8821d797ba6ea451c6ac38a006ec02c8689608b9cf24ad619ed91b708de5a4b（currently under embargo）。
- **关键超参**：Q-learning 学习率 η_Q=0.15，折扣 δ=0.95；表格权重 16.5；head logits 上限 c=5.5；ε 衰减 ε=0.3×0.9994^t；LoRA rank=16，scaling=32；policy gradient LR=10⁻⁴（LoRA）/3×10⁻³（head）；entropy bonus=0.01；8000 轮；16 并行市场；continuation=0.98。
- **模型**：Qwen2.5-0.5B-Instruct（训练 agent），Qwen2.5-7B/14B-Instruct、Mistral-7B-Instruct-v0.3、OLMo-2-1124-7B-Instruct（prompted 扩展）。
