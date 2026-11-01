---
title: "TRAIN-FOR-MANY-UPDATE-WITH-ONE-IMPROVING-MULTI-SAMPLE-REASON"
source: https://arxiv.org/pdf/2610.17735v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:09:28"
field: "大模型推理与多样本采样"
keywords: ["多样本推理", "Pass@K", "Fenchel-Bregman优化", "变分后验学习", "有限K后验学习", "递归推理", "数学推理"]
innovations: ["将Fenchel共轭重公式化与Bregman EMA追踪结合，实现每次更新单次后验采样的有限K后验学习", "问题相关梯度权重K*F^(K-1)的自然涌现，使训练自动聚焦于其他候选难以解决的问题"]
benchmarks: ["Sudoku-Extreme", "Maze-Hard", "ARC-AGI-1", "ARC-AGI-2", "MATH500", "OlymMATH", "AMC23", "AIME24", "AIME25"]
---

# 论文速读：TRAIN FOR MANY, UPDATE WITH ONE: IMPROVING MULTI-SAMPLE REASONING VIA FENCHEL–BREGMAN POSTERIOR LEARNING

## 一句话总结
本文提出有限K后验学习（finite-K posterior learning）框架，将候选覆盖率（Pass@K）作为训练目标，并基于Fenchel-Bregman对偶重公式化和闭式EMA权重追踪，实现每次更新仅需单次后验采样的优化算法FBI，在递归推理与自回归数学推理任务上同时提升了候选覆盖率并显著降低了训练开销。

## 研究问题与动机
- **核心问题**：多样本推理（multi-sample reasoning）的目标是最大化Pass@K（即K个候选中至少一个正确的概率），但现有方法通常最小化单个采样的平均损失，这与多样本推理的候选覆盖率目标不对齐。
- **现有不足**：直接蒙特卡洛估计器Direct-K虽能无偏估计有限K全失败概率，但每次更新需K次独立后验采样，计算成本高；而简单重用单次采样无法保持$K$次幂目标的无偏性（因为对于二值失败指示器$F$，$\mathbb{E}[F^K]=\bar{F}$而非$\bar{F}^K$）。
- **动机**：训练一个过求解器参数的分布$q_\phi(\theta)$，使K个独立后验抽样的候选池在推断时获得更高的覆盖率，同时避免每次更新消耗K次采样。

## 核心贡献（创新点）
- **有限K后验学习的形式化**：提出将候选池的全失败概率$\bar{F}_\phi(x)^K$直接作为训练目标，与现有单模型多候选生成方法本质不同——本文训练的是参数分布而非单一模型。
- **Fenchel重公式化实现单次采样更新**：利用Fenchel共轭将凸函数$r^K$分解为加权线性函数，将问题相关的梯度权重$\lambda_x$与后验更新解耦，使得固定权重下的后验更新可仅依赖一次采样。
- **Bregman追踪实现闭式权重更新**：基于Fenchel共轭函数$\psi_K$的Bregman散度推导出一阶指数移动平均（EMA）权重追踪公式，避免了额外后验采样，同一张采样同时用于后验更新和下一轮权重更新。
- **实例化为FBI（Fenchel–Bregman IVON）**：将上述方法接入IVON变分优化器，形成端到端算法，在TRM/FPRM递归模型和Qwen3-4B自回归模型上均验证有效性。

## 方法详解

**有限K目标**：对于问题$x$，定义$F(\theta,x)\in\{0,1\}$为单次失败的指示器，后验下失败概率$\bar{F}_\phi(x)=\mathbb{E}_{\theta\sim q_\phi}[F(\theta,x)]$，则K个独立候选全部失败的联合概率为$\bar{F}_\phi(x)^K$。训练目标为最小化期望全失败概率加KL正则：
$$\min_\phi \mathbb{E}_x[\bar{F}_\phi(x)^K] + \tau_{KL} D_{KL}(q_\phi\|p_0)$$

**问题相关梯度权重**：对Eq.2数据项求梯度可得$\nabla_\phi \bar{F}_\phi(x)^K = \underbrace{K\bar{F}_\phi(x)^{K-1}}_{\lambda_x^*}\nabla_\phi \bar{F}_\phi(x)$，即困难问题（其他候选不太可能解决）获得更大权重，易解问题获得更小权重。

**Fenchel重公式化**：对$K\ge2$，利用恒等式$r^K=\max_{\lambda\in[0,K]}\{\lambda r-\psi_K(\lambda)\}$，其中$\psi_K(\lambda)=(K-1)(\lambda/K)^{K/(K-1)}$，最优权$\lambda_x^*=Kr^{K-1}=K\bar{F}_\phi(x)^{K-1}$。原问题转化为鞍点形式：
$$\min_\phi\max_{\lambda_x}\mathbb{E}_x[\mathbb{E}_{\theta\sim q_\phi}[\lambda_x F(\theta,x)]] - \psi_K(\lambda_x) + \tau_{KL}D_{KL}(q_\phi\|p_0)$$

**固定权重的单次采样更新**：当$\lambda_x$固定时，后验子问题的数据项退化为加权单期望，单次采样即可无偏估计梯度，且$\lambda_x=\lambda_x^*$时恰好恢复有限K梯度。

**Bregman追踪（闭式EMA）**：对对偶权重$\lambda_x$施加Bregman proximal更新，等价于带Bregman散度的镜像上升，其闭式解为：
$$\hat{F}_{x,t+1}=\beta\hat{F}_{x,t}+(1-\beta)F_t,\quad \lambda_{x,t+1}=K\hat{F}_{x,t+1}^{K-1}$$
其中$\beta=1/(1+\eta)$控制历史失败估计的保留程度，$\hat{F}$为观察失败分数的指数移动平均，无需额外采样。

**FBI算法流程**：每步从D抽取小批量，用当前EMA估计构造$\lambda_{x,t}$，单次抽样$\theta_t\sim q_{\phi_t}$计算加权损失$\mathcal{L}_t^{data}$，用IVON更新后验参数，再用相同失败观测更新EMA，形成交替的随机 primal-dual优化。

## 实验与结果

**递归推理（TRM/FPRM主干）**：
- **Sudoku-Extreme**：FBI达到Pass@10=99.55%、Selected Accuracy=99.53%，超越PTRM（97.38%）和W-PTRM（98.84%）。
- **Maze-Hard**：FBI达到Pass@10=91.30%、Acc=88.00%，较W-PTRM的89.80%提升1.5个百分点。
- **ARC-AGI-1/2**：FBI在多个K预算下达到或持平最高Pass@K。
- 跨主干验证：FPRM上FBI达到Pass@10=94.80%（vs W-FPRM 88.70%）。
- 跨贝叶斯方法：FBI（91.3% Pass@10）和FBE（92.7% Pass@10）均超越IVON（90.8%）、EVON（91.1%）、LA和SWAG。

**自回归数学推理（Qwen3-4B-Base，$K_{train}=8$）**：
- **Table 2汇总Pass@8**：FBI以60.44%的平均Pass@8位列第一，比DMPO（59.60%）高0.84个百分点，比Pass@K Training（59.55%）高0.89个百分点。在AMC23（85.20%）和AIME25（33.70%）上取得单项最高。
- **覆盖率曲线**：K=1时FBI落后DMPO（40.22% vs 43.01%），K=4时追平，K=8时反超0.84个百分点至60.44%，K=16时达66.06%超过Pass@K Training 1.90个百分点。

**训练效率（匹配比较）**：
- FBI每次更新仅需1次后验采样，Direct-10需10次；在相同GPU和种子下，FBI达到与Direct-10可比的Pass@10，但训练时间仅为Direct-10的12.3%，FLOPs仅为10.4%，峰值GPU显存仅为0.611GB vs 3.707GB。
- $K_{train}$对角效应：$K_{train}=K$在每个评估预算下均获得最高Pass@K。
- 问题相关权重消融：移除后Pass@10从91.3%降至90.7%。
- 首候选失败后的恢复率：FBI达32.5%（40/123），高于Direct-10（27.2%）和W-PTRM（22.1%）。

## 相关工作脉络
- **Pass@K Training (Chen et al., 2025b)**：优化组级别成功概率，但训练单个模型，本文训练参数分布以匹配固定候选池预算。
- **DMPO (Li et al., 2026)**：通过分布匹配保留多样性，关注group distribution；本文直接优化全失败概率，问题加权策略不同。
- **MMPO (Zhang et al., 2026)**：最小化失败概率的多阶矩，以目标截断首次成功时间；本文仅用一阶矩但结合问题相关权重形成等效的组级目标。
- **Pass@K Policy Optimization (Walder & Karkhanis, 2025)**：强化学习视角优化组成功率；本文基于变分贝叶斯框架，后验学习视角更统一。
- **IVON (Shen et al., 2024)**：用于高维网络的对角高斯后验变分优化器，本文作为FBI的后验更新内核，扩展至多样本推理场景。
- **EVON (Minut et al., 2026)**：结构化（Kronecker分解）高斯后验；本文的FB框架可平移至EVON得FBE，展示泛化性。

## 局限性与未来方向
- **代理不匹配**：实际使用连续失败分数$f(\theta,x)$替代二值指示器$F$，导致$\bar{f}_\phi^K$仅为$\bar{F}_\phi^K$的代理，校准偏差可能影响权重精确度。
- **随机梯度噪声**：单次采样估计梯度方差较高，在高K值下估计质量可能下降。
- **后验漂移**：EMA追踪的失败估计依赖于历史观测，快速变化的后验可能导致权重滞后。
- **未来方向**：改进代理校准（如温度调度）、采用预测性或方差缩减的权重追踪、自适应/关联候选池以更好地对齐推断、为未见过问题引入摊销估计、在更广泛任务和更多后验家族上验证。

## 研究启发与可借鉴点
- **Fenchel重公式化分离权重与采样**：将组级目标中的高次幂项通过共轭分解，实现了"一次采样做后验更新、EMA做权重更新"的解耦设计，该思路可推广到其他组级/复合优化目标。
- **Bregman-EMA等价性**：将Bregman proximal更新与指数移动平均建立严格等价关系，为在线权重追踪提供了理论保证，而非经验启发式设计。
- **问题相关加权的学习信号**：$K\bar{F}^{K-1}$的权重分配自然地将训练注意力集中在"其他候选难以解决"的问题上，这一现象可直接迁移到reward shaping或curriculum learning中。
- **跨主干与跨后验的通用性**：FBI在TRM/FPRM等不同递归骨架以及IVON/EVON等不同变分后验上均有效，说明FB框架具有较好的泛化性，可借鉴到其他推理模型。
- **$K_{train}=K$的对角对齐规律**：训练目标候选数与评估预算对齐时表现最优，为实际部署中的超参选择提供了直接指导。

## 关键术语表
**Pass@K**：多样本推理的核心评估指标，表示K次独立生成中至少一次得到正确答案的概率。
**Finite-K Posterior Learning**：本文提出的学习框架，直接优化K候选池的全失败概率作为训练目标。
**Fenchel Reformulation**：利用凸函数的Fenchel共轭将$r^K$分解为线性-凸函数形式，使问题相关权重与后验更新解耦。
**Bregman Update**：基于Fenchel共轭函数生成的Bregman散度的proximal更新，在此文中等价于闭式EMA。
**IVON (Iterative Variational Online Learning)**：用于深度网络的变分贝叶斯优化器，学习对角高斯后验的均值和对角精度。
**FBI (Fenchel–Bregman IVON)**：本文提出的具体算法，将Fenchel重公式化与Bregman追踪集成到IVON框架中。
**Failure Score**：连续可微的失败近似分数$f(\theta,x)\in[0,1]$，作为二值失败指示器的代理用于梯度估计。
**Problem-Specific Weight**：问题相关的梯度缩放系数$\lambda_x=K\bar{F}^{K-1}$，困难问题获得更高权重以强调学习。

## 可复现要素
- **数据集**：Sudoku-Extreme、Maze-Hard、ARC-AGI-1、ARC-AGI-2（递归推理）；MATH7.5K（训练）、MATH500/OlymMATH/AMC23/AIME24/AIME25（数学推理评估）——部分数据集为公开基准。
- **代码/权重**：论文未明确声明开源代码或模型权重，引用了公开检查点（TRM、FPRM、Qwen3-4B-Base）。
- **关键超参**：$K_{train}\in\{1,2,5,10\}$、EMA系数$\beta\in\{0.9,0.99\}$、KL强度（通过IVON ESS控制）、failure-score温度$T$、学习率（TRM路径约$10^{-4}$，Qwen路径0.5）、batch size（128/32/768）、ESS（1.5万~10亿）——详细配置见Appendix B Tables 5-7及B.4节。
