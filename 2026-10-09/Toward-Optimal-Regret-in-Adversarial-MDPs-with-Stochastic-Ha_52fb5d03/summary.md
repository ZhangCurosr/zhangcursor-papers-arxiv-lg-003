---
title: "Toward-Optimal-Regret-in-Adversarial-MDPs-with-Stochastic-Ha"
source: https://arxiv.org/pdf/2610.12153v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:12:04"
field: "安全强化学习与在线决策"
keywords: ["constrained MDP", "adversarial loss", "safe exploration", "regret lower bound", "Slater margin", "online learning"]
innovations: ["改进S-OPS将regret从1/d^2降至1/d", "提出MA-OPS实现对T,d,ρ联合最优的regret界"]
benchmarks: ["episodic adversarial CMDP with stochastic hard constraints"]
---

# 论文速读：Toward Optimal Regret in Adversarial MDPs with Stochastic Hard Constraints

## 一句话总结
本文针对对抗性损失与随机硬约束下的受限MDP（CMDP），提出MA-OPS算法，通过乐观搜索+悲观评估安全地学习高Slater边际策略，实现了对T、d、ρ三者均最优的regret界 $\widetilde{\mathcal{O}}(\sqrt{T}/\rho + 1/(d\rho))$，并给出匹配的下界证明。

## 研究问题与动机
1. **核心问题**：在已知初始严格可行策略（margin为d）但Slater边际ρ未知的CMDP中，设计一个算法，在**每期都严格满足硬约束**的前提下，达到regret对T、d、ρ的最优联合依赖。
2. **S-OPS的不足**：Stradi et al. (2025a) 的S-OPS在d≤1时regret为 $\widetilde{\mathcal{O}}(\sqrt{T}/d^2)$，而理论下界为 $\Omega(\sqrt{T}/\rho)$，两者之间存在$d$与$\rho$之间的显著差距；且其分析将baseline使用cost与探索uncertainty解耦，导致$d$的依赖出现平方放大。
3. **d与ρ分离的必要性**：实际场景中初始提供的安全策略可能远非最优（d≪ρ），但若算法能在安全探索中自适应发现更接近ρ的边际，则regret可从$1/d$改善为$1/\rho$。
4. **硬约束的额外挑战**：与累积违规允许的情况不同，每期零违规（zero constraint violation per episode）要求搜索过程中的每次采样都不能突破阈值，这引入了比纯stochastic CMDP更严格的探索代价。

## 核心贡献（创新点）
1. **S-OPS线性1/d改进**：改进了S-OPS的regret分析，将 baseline 使用regret从 $\widetilde{\mathcal{O}}(\sqrt{T}/d^2)$ 降至 $\widetilde{\mathcal{O}}(\sqrt{T}/d)$，核心技巧是将执行概率 $(1-\lambda_{t-1})$ 嵌入 constraint violation 加权分析（Lemma 3.1）。
2. **MA-OPS算法设计**：首次提出一种在常数时间内安全学习 Slater 边际并同时满足每期约束的算法，通过乐观 max–min 估计 $\overline{\rho}_n$ 与悲观 cost upper bound 估计 $\underline{\rho}_n$ 构建自适应停止规则。
3. **紧 regret 上界**：MA-OPS 达到 $\widetilde{\mathcal{O}}(\sqrt{T}/\rho + 1/(d\rho))$ 的 regret，主项 $1/\rho$ 消除了对已知初始边际 d 的依赖，加法项 $1/(d\rho)$ 刻画了从零 margin 到发现 $\rho/2$ 边际的不可避免探索代价。
4. **最优下界**：证明对任意满足每期零违规的算法，存在实例使 regret 达到 $\Omega(\sqrt{T}/\rho + 1/(d\rho))$，上下界在 T、d、ρ 的依赖上至多差对数因子，即联合最优。

## 方法详解
- **S-OPS 改进分析**：关键在于公式 (4) 的改写：$\lambda_{t-1}(\alpha_{i_t} - \beta_{i_t}) = (1-\lambda_{t-1})(w_{t,i_t} - \alpha_{i_t})$，由于 $\alpha_{i_t} - \beta_{i_t} \geq d$，可得 $\lambda_{t-1} \leq (1-\lambda_{t-1})e_t / d$，从而 baseline 使用的 regret 被 $(1-\lambda_{t-1})e_t$ 加权控制，而非无权重上界 $L/d$，最终得到线性 $1/d$ 依赖。
- **MA-OPS 两阶段结构**：第一阶段为 margin search（Algorithm 1 第5–23行），第二阶段调用 S-OPS。
- **乐观边际搜索**：每收集一条探索轨迹后，求解 LP (B.2) 得 $\overline{\rho}_n = \min_i[\alpha_i - (\widehat{g}_{n-1,i} - \xi_{n-1})^\top \widehat{q}_n]$，在置信事件上保证 $\overline{\rho}_n \geq \rho$。
- **悲观策略评估**：对选定策略 $\widehat{\pi}_n$，计算 $w_{n,i} = \min\{L, \max_{P'\in\mathcal{P}_{n-1}}(\widehat{g}_{n-1,i}+\xi_{n-1})^\top q^{P',\widehat{\pi}_n}\}$，从而得到下界 $\underline{\rho}_n = \min_i(\alpha_i - w_{n,i})$，满足 $\underline{\rho}_n \leq \text{true margin} \leq \rho \leq \overline{\rho}_n$。
- **自适应停止规则**：若 $2d \geq \overline{\rho}_n$ 则返回原 baseline（边际已足够）；若 $2\underline{\rho}_n \geq \overline{\rho}_n$ 则返回新策略（ pessimistic 估计已接近乐观上界，真实边际 $\geq \rho/2$）。
- **安全探索混合**：搜索阶段沿用 S-OPS 的混合概率 $\lambda_n = \max_i(w_{n,i}-\alpha_i)/(w_{n,i}-\beta_i)$，确保 $G^\top q_t \leq \lambda_n\beta + (1-\lambda_n)w_n \leq \alpha$。

## 实验与结果
- 本文为纯理论工作，**无数值实验部分**，所有结论均以定理形式给出。
- **主要理论结果**：
  - Theorem 1：S-OPS regret 改进为 $\widetilde{\mathcal{O}}(L^2|X|\sqrt{|A|T}/d + L|X|^3|A|/d)$。
  - Theorem 2：MA-OPS regret $\widetilde{\mathcal{O}}(L^2|X|\sqrt{|A|T}/\rho + L^2|X|^3|A|/(d\rho))$，每期约束满足。
  - Theorem 3：搜索阶段安全终止，收集 $\widetilde{\mathcal{O}}(L^2|X|^2|A|/\rho^2)$ 条探索轨迹，总搜索时长 $\widetilde{\mathcal{O}}(L^2|X|^2|A|/(d\rho))$ 期。
  - Theorem 4：下界 $\Omega(\sqrt{T}/\rho + 1/(d\rho))$，匹配上界。
- 对比 S-OPS（$\widetilde{\mathcal{O}}(\sqrt{T}/d^2)$）和 CV-OPS（允许 violation），MA-OPS 在 leading term 上将 $1/d$ 替换为 $1/\rho$，在 strict zero violation 条件下达成最优。

## 相关工作脉络
1. **S-OPS (Stradi et al., 2025a)**：同设定下最早的工作，给出 $\widetilde{\mathcal{O}}(\sqrt{T}/d^2)$ regret 及 $\Omega(\sqrt{T}/\rho)$ 下界；本文改进了其 $1/d$ 依赖并闭合了这一 gap。
2. **CV-OPS (Stradi et al., 2025a)**：无需初始 feasible policy，但允许搜索阶段产生 $\widetilde{\mathcal{O}}(1/\rho^4)$ 的累积违规；本文在要求零违规的同时达到更优的 $\rho$ 依赖。
3. **OptPess-LP / DOPE+ (Liu et al., 2021; Yu et al., 2025)**：针对 stochastic CMDP 的工作，leading term 均为线性 $1/d$，但有 additive $1/d^2$ 项；本文将其推广至 adversarial losses 并消除 additive quadratic term。
4. **SOLB (Genalti et al., 2025)**：adversarial constrained bandits，假设已知最大边际策略（即 $d = \rho$）；本文首次显式分离 d 与 ρ，量化二者不同时的探索代价。
5. **OPB (Pacchiano et al., 2021)**：stochastic bandits 下给出 $\widetilde{\mathcal{O}}(\sqrt{T}/d)$；本文的工作可视为其在 episodic MDP + adversarial losses  setting 下的拓展与深化。

## 局限性与未来方向
1. **依赖问题维度**： regret 界中包含 $|X|^3|A|$ 的加法项，在高维状态/动作空间下可能较大；如何压缩这一项是开放问题。
2. **仅适用于 loop-free（有向无环层状）CMDP**：虽可通过时间展开处理一般有限 horizon MDP，但对循环结构或 discounted setting 尚未覆盖。
3. ** Slater 条件依赖**：Assumption 1 要求已知严格可行策略；在 general Slater condition（无已知 baseline）下的零违规学习仍为开放问题。
4. **扩展至函数近似**：当前为 tabular setting，在线性函数近似或 neural function approximation 下的安全探索 regret 分析尚未开展。

## 研究启发与可借鉴点
1. **加权 regret 分解技巧**：将 baseline 使用概率 $(1-\lambda_{t-1})$ 引入 violation 上界的控制，是处理 hard constraint 下 safe exploration 的通用分析工具，可迁移至其他 constrained RL 算法。
2. **乐观–悲观成对构造**：MA-OPS 中 $\overline{\rho}_n$ 与 $\underline{\rho}_n$ 的对称设计——一个上界保证不低估，一个下界保证不高估——构成了一种可复用的"双侧边界收敛"范式，适用于 unknown parameter 的安全估计问题。
3. **自适应停止规则（$2\underline{\rho} \geq \overline{\rho}$）**：当悲观估计达到乐观估计的一半时即停止，避免了精确收敛所需的过多次采样，这一比率型停止准则可推广到其他参数学习场景。
4. **下界构造中的"互换动作代价"技巧**：证明 $1/(d\rho)$ 项时，通过交换两个 low-loss 动作的代价分布构造两个 instance，并利用 KL 散度限制采样信息量，是信息论下界在 constrained RL 中的经典论证模式。

## 关键术语表
- **CMDP（Constrained MDP）**：在标准 MDP 基础上引入多条代价约束，要求累积期望代价不超过给定阈值。
- **Slater margin ρ**：所有可行策略中能达到的最大公共约束 slack，衡量问题的"安全性余量"。
- **Initial margin d**：已知 baseline 策略的实际安全裕度，满足 $d \leq \rho$。
- **Hard constraint / Zero violation per episode**：要求每一期的期望累积代价都不超过阈值，不允许累积抵消。
- **Occupancy measure $q^{P,\pi}$**：策略 π 在转移核 P 下访问各 state–action–next-state 三元组的概率分布。
- **Confidence set $\mathcal{P}_t$**：基于历史观测构建的转移核不确定性集合，用于 optimistic/pessimistic 评估。
- **Online mirror descent（OMD）**：在 occupancy 空间中通过相对熵投影实现带约束的在线策略更新。
- **Implicit exploration（IE）**：通过在 loss estimator 分母中加入 $\gamma$ 来隐式鼓励探索，避免显式 exploration rate schedule。

## 可复现要素
- **数据集**：无（理论论文）。
- **代码/权重**：论文未开源（截至 arXiv 提交时），但算法（Algorithm 1, 2）与公式细节完整，可据此复现。
- **关键超参**：学习率 $\eta = \gamma = \sqrt{L\ln(8m|X|^2|A|T/\delta)/(T|X||A|)}$；置信参数 $\delta_0 = \delta/2$ 分配至搜索与 S-OPS 两阶段；logarithmic parameter $u = \log(8m|X|^2|A|T/\delta)$。
