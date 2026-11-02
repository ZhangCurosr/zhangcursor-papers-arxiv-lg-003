---
title: "Transfer-Learning-in-Bandits-from-Unknown-Adaptive-Policies"
source: https://arxiv.org/pdf/2610.21582v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-11-02 08:50:03"
---

# 论文速读：Transfer-Learning-in-Bandits-from-Unknown-Adaptive-Policies

## 一句话总结
本文研究多臂老虎机（MAB）在**过程生成先验样本**（process-generated samples）设定下的转移学习问题，提出 **KL-UCB-Transfer** 算法，推导了适配未知自适应源策略的实例依赖渐近下界与对数 regret 上界，并以有效转移水平 $\mu^\dagger$ 统一刻画单侧偏差条件下的三区制鲁棒性。

## 研究问题与动机
- 传统离线迁移假设先验数据为独立同分布或固定样本，但实际历史决策常由**未知且自适应的策略** $\pi'$ 动态生成，样本量 $N_k'$ 随机且与目标分布独立。
- 现有转移学习 bandit 算法无法直接处理源-目标均值仅满足有界偏差 $|\mu_k - \mu_k'| < L_k$ 且分布为非参数有界 $\mathcal{P}([0,1])$ 的情形。
- 需在不对称先验信息下建立信息论下界，并设计既能吸收先验收益、又能在先验失效时退化为纯在线算法的探索策略。
- 目标分布仅假设均值有界，缺乏参数结构，要求方法具备强非参数鲁棒性。

## 核心贡献（创新点）
- **提出 KL-UCB-Transfer 算法**：将 Prevost et al. (2025) 的高斯/固定样本版本推广至有界非参数分布与随机先验样本情形；本质区别在于显式建模过程生成样本的随机性，并在索引中嵌入平移散度项。
- **推导适应未知自适应源策略的实例依赖渐近下界（Theorem 2）**：揭示先验均值 $\mu_a'$ 相对 $\mu_1 - L_a$ 的位置决定次优臂采样下界；与固定离线下界相比，允许 $\mathbb{E}[N_a']/\ln T$ 趋于非零常数。
- **建立 regret 上界并证明一阶最优性（Theorem 3）**：算法在对数 regret 意义下一阶不劣于纯在线 KL-UCB，且先验均值远离次优臂时可显著降低采样。
- **引入有效转移水平 $\mu^\dagger$ 统一误设定鲁棒性（Theorem 5）**：将对称偏差条件放松为单侧 $\mu_k - \mu_k' < L_k$，以 $\mu^\dagger$ 为枢纽统一规范、半规范与不利三种区制的 regret 表达。
- **推广至 $M-1$ 个独立先验源的多源设定（Theorem 6）**：证明多源贡献可叠加，$M=2$ 时退化回单源结果， regret 上界保持相同对数结构。

## 方法详解
- **问题设定**：$K$ 臂 MAB，目标奖励分布 $\nu_a \in \mathcal{P}([0,1])$，源先验分布 $\nu_a'$ 来自未知自适应策略，已知均值偏差上界 $L_a$；先验样本量 $N_a'$ 为随机变量且与目标分布 $\nu$ 独立。
- **索引构造**：$U_a(t) = \sup\left\{\mu \,\middle|\, N_a(t)\mathcal{K}_{inf}(\hat{\nu}_a(t),\mu) + N_a'\mathcal{K}_{inf}(\hat{\nu}_a'(t),\mu - L_a) \leq \delta_t\right\}$，其中 $\delta_t = \ln t + \ln\ln t$，$\mathcal{K}_{inf}(\nu,\mu) = \inf_{\tilde{\nu}:\tilde{\mu}>\mu}\mathrm{KL}(\nu,\tilde{\nu})$。
- **决策规则**：首轮每臂探索一次，之后 $a_t = \arg\max_a U_a(t)$；第二项将先验估计均值平移 $L_a$ 后再计算散度，体现“源信息需满足偏差容忍”的设计逻辑。
- **多源扩展**：将 $N_a'$ 替换为 $\sum_{i=2}^M N_a^{(i)}$，索引结构不变，各源贡献在 regret 上界中以乘积型偏差累积项叠加。
- **有效转移水平与三区制**：选取 $\mu^\dagger \in (\mu_a, \mu_1]$ 满足 $\mu^\dagger < \mu_1' + L_1$，统一 regret 上界 $\mathbb{E}[N_a(T)] \leq \frac{\ln T}{\mathcal{K}_{inf}(\nu_a, \mu^\dagger)} + h(T)$，$h(T)=O((\ln T)^{4/5}(\ln\ln T)^{1/5})$；当 $\mu_a' + L_a < \mu^\dagger$ 时出现带 $(\cdot)_+$ 项的改进形式。

## 实验与结果
- 论文第 9 节提供模拟实验验证过程生成历史数据的实际收益，但分段笔记未披露具体数据集名称、对比基线数值及主要指标数字；代码与权重开源情况未提及。
- 理论评估显示：在先验均值满足 $\mu_a' < \mu_1 - L_a$ 时，算法对数 regret 系数显著低于标准 KL-UCB；当先验信息不足或存在单侧偏差时，退化为与纯在线算法相同的 $O(\ln T)$ 上界。
- 最强结果为 Theorem 3 与 Theorem 5 的理论 regret 上界，在规范区制且浓度条件 $\mathbb{E}[|N_a' - \mathbb{E}[N_a']|] = o(\ln T)$ 成立时（Proposition 4），上下界匹配至一阶。

## 相关工作脉络
- **KL-UCB / KL-UCB-Switch (Cappé et al. 2013; Garivier et al. 2022)**：在线基础算法；本文将其索引扩展至含源偏移项的非参数迁移设定，并引入 $\mathcal{K}_{inf}$ 浓度引理（Lemma 7-8）控制估计偏差。
- **Prevost et al. (2025)**：直接前身，处理高斯分布与固定样本；本文推广至有界非参数分布与随机过程生成样本，并建立新的信息论下界。
- **转移学习 bandit (Zhang & Bareinboim 2017; Bellot et al. 2023; Cai et al. 2024; Deng et al. 2025)**：侧重因果可迁移性与多环境迁移；本文聚焦未知自适应源策略下的统计 regret 界，不依赖因果图假设。
- **离线到在线学习 (Cheung & Lyu 2024; Sentenac et al. 2025)**：处理带偏离线数据集；本文的过程生成样本具有时序自适应依赖性，区别于 i.i.d. 离线假设。
- **Thompson Sampling (Russo & Van Roy 2016; Kaufmann et al. 2012)**：贝叶斯基准；本文采用频率主义 UCB 框架，更适配非参数有界分布与最坏情况分析。
- **Routine Bandits (Saber et al. 2021)**：研究重复问题的最小 regret；本文解决单次跨环境迁移，场景更为一般。

## 局限性与未来方向
- 下界（Theorem 2）在边界情形 $\mu_a' = \mu_1 - L_a$ 处与上界未完全匹配，论文指出该情形仍为开放问题。
- 理论假设 $N_a'$ 与目标分布 $\nu$ 独立，实际自适应策略可能产生隐式依赖，影响 regret 界的有效性。
- 仅考虑均值偏差有界条件，未讨论奖励分布形状差异（如方差、支持集重叠度）对转移收益的影响。
- 多源扩展（Theorem 6）假设各源独立，未处理源间相关性或选择性观测偏差。
- 模拟实验细节未在笔记中提供，算法在实际自适应历史策略下的收敛速度与超参敏感性有待验证。

## 研究启发与可借鉴点
- **索引构造范式**：将源先验信息以平移散度项 $N_a'\mathcal{K}_{inf}(\cdot, \mu - L_a)$ 嵌入 UCB 索引，可直接迁移至线性 bandit 或 Contextual bandit 的迁移场景。
- **有效转移水平 $\mu^\dagger$ 的统一刻画**：用单一参数统合三种区制的 regret 上界，适用于需要鲁棒性分析的跨环境迁移理论证明。
- **过程生成样本
