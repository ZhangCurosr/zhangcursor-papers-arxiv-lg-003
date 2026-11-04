---
title: "The-Staged-Pandora-s-Box-Reservation-Indices-for-Multi-Fidel"
source: https://arxiv.org/pdf/2611.02722v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 18:22:05"
---

# 论文速读：The-Staged-Pandora’s-Box-Reservation-Indices-for-Multi-Fidel

## 一句话总结
本文提出**分阶潘多拉魔盒**（Staged Pandora’s Box）理论框架，为具有有限阶梯、仅部分阶段可采集结果且观测成本依赖自身历史的候选搜索问题给出了精确的认购期权（covered-call）递归形式，证明了独立情形下的最优预留索引策略，并给出了任意策略的精确损失分解、收集权限/成本的比较静态定价，以及在臂间信息共享下的近似规划 regret 上界。

## 研究问题与动机
- **单阶段索引无法刻画多保真/阶梯评估结构：** 传统 Pandora’s box 与 PBGI 仅处理一次性观测，而超参训练、代理评估与学习曲线 checkpoint 等场景天然呈多阶段阶梯结构，且只有特定阶段的结果可被采纳为最终候选。
- **仅部分阶段可采集（intermediate collection）改变最优决策规则：** 中间读数若可作为最终结果保留，将同时影响“前进哪条臂”与“何时停止”两个决策，现有嵌套索引与终端仅收集模型无法直接覆盖该设定。
- **共享先验/潜变量破坏独立臂假设：** 在联合代理（如多任务 BO 的共享核）下，一条臂的观测会改变其他臂的预测分布；仅凭各臂边际预测计算的索引策略是否具有理论保证尚不清楚。
- **缺乏对偏离策略的精确代价量化：** 现有 BO 停止/提前策略（如 freeze-thaw、Hyperband）缺少可归因的性能损失分解，难以支撑算法诊断与改进。

## 核心贡献（创新点）
- **提出兼容任意采集集与历史相关成本的认购期权递归：** 通过 `W/Z/G` 嵌套逆推将分阶搜索转化为金融期权定价结构，无需离散化或单调性假设即可得到唯一有限预留索引。
- **建立任意策略的精确损失分解定理：** 将停止损失与每步前进损失表示为被覆盖调用值与竞争臂状态的泛函，实现策略偏离最优的完全可归因，而非仅提供上界。
- **给出收集权限与成本变化的显式比较静态定价：** 证明扩大采集集或降低成本使问题价值提升的 overshoot 公式，并严格刻画“误用较小收集集规划”时的上下界可达性。
- **揭示信息交织下独立索引的极限并提供有界 regret 保证：** 构造相同边际律但联合结构不同的反例证明普适独立规则不存在；在相干联合路径律下导出基于局部值变化与预测总变差（TV）色散的统一 regret 上界。

## 方法详解
- **模型设定：** 固定候选池 $\mathcal{X}=\{x_1,\dots,x_N\}$，臂 $i$ 有长度 $T_i$ 的有限阶梯；每次前进耗资 $c(s_i)$ 并采样 $Y \sim P_i(\cdot|s_i)$；预设采集集 $K_i \subseteq \{1,\dots,T_i\}$，仅 $t \in K_i$ 的读数计入当前 incumbent $\tilde{y}$（否则为 $-\infty$）。目标为 $\sup_\pi \mathbb{E}[\tilde{y}_{\tau_\pi} - K_{\tau_\pi} \mid S_0]$。
- **认购期权递归（Covered-Call Recursion）：** 对剩余路径状态 $s$，定义 $W(s) = \max\{\tilde{Y}, Z(s')\}$，$Z(s) = \min\{G(s), W(s)\}$，其中 $G(s)$ 由预留方程 $\mathbb{E}[(W(s)-G(s))^+] = c(s)$ 唯一确定。$W$ 为整条剩余路径上的随机变量，$Z$ 为被价格 $G$ 封顶的期权值，$G=-\infty$ 当且仅当无剩余采集阶段。
- **最优索引策略（Theorem 1）：** 在独立臂假设下，最优值满足 $V^*(S)+K = \mathbb{E}[\max\{q, \max_i Z_i(s_i)\}]$。策略为：当 $\max_i G_i(s_i) > q$ 时前进索引最大的任意臂；否则停止。该策略在有限阶梯下可达全局最优。
- **精确损失分解（Theorem 2）：** 停止损失 $\Delta_{\text{stop}}(S) = \mathbb{E}[(\max_j Z_j - q)^+]$；前进臂 $i$ 的损失 $\Delta_i(S) = c(s_i) - \mathbb{E}[(W_i - \max\{O_i, G_i\})^+]$，其中 $O_i$ 为其余臂与 incumbent 提供的 outside option。任意策略的累积损失等于轨迹上各步损失之和。
- **共享信息下的鲁棒界（Prop 7 & Cor 2）：** 定义未观测臂局部值变化度量 $\varepsilon_i(S)$，若沿所有可行历史一致满足 $\varepsilon_i \leq \bar{\varepsilon}$，则索引策略 regret $\leq 2H\bar{\varepsilon}$（$H$ 为总剩余阶梯数）。若共享流经由公共潜变量 $\Theta$，进一步控制为 $\eta_i(S) \sum_{j \neq i} B_j(S) \Lambda_j(S)$，其中 $\eta_i$ 为一次新观测的预测 TV 色散，$\Lambda_j$ 为臂 $j$ 对 $\Theta$ 的路径敏感度。

## 实验与结果
- **验证设置：** 纯理论精确验证，使用有理算术求解联合 Bellman 方程作为 ground truth；未引入真实 ML 基准或学习代理。
- **独立臂验证：** 192 个独立二元树实例与 660 个阶梯网格单元格中，索引最优动作、价值表示、所有停止/前进损失均与联合 Bellman 完全吻合；指定偏离策略（如 stop-after-one、lowest-index-first）的效用损失精确等于累积动作损失。
- **比较静态与回退值：** 1,763+60 对受控问题对验证价值差等于认购期权 overshoot；终端仅收集策略在中间采集集下损失落在 $[0, V^{*'} - V^*]$ 内，两端均在各网格点可达；无
