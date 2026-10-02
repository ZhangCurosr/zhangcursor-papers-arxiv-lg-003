---
title: "Scaling-Laws-for-Looped-Mixture-of-Experts"
source: https://arxiv.org/pdf/2609.40316v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:38:30"
---

# 论文速读：Scaling-Laws-for-Looped-Mixture-of-Experts

## 一句话总结
本文提出了首个联合建模循环深度（recurrence）与稀疏性（MoE sparsity）的扩展定律（Loop Scaling Laws），通过引入有界且受活跃参数比例调节的循环映射，更准确地预测了循环MoE模型的泛化损失，并为计算与显存双重约束下的架构寻优提供了统一原则。

## 研究问题与动机
- **单一轴建模的局限**：现有扩展定律仅单独刻画循环（如Parcae、Iso-Depth）或稀疏性（如Unified MoE、Joint MoE），缺乏对两者交互行为的统一理论框架。
- **循环收益的饱和特性未被捕捉**：循环Transformer通过权重复用增加计算深度，但边际收益天然递减；先前的线性或幂律映射假设增益无界，导致外推预测严重乐观。
- **前作未解耦R与E的联合缩放**：近期循环MoE研究（Sparse Layers、SMELT）在主扩展分析中将循环次数固定为 $R=2$，未能揭示不同稀疏度下循环深度与性能的动态权衡。
- **缺乏资源约束下的设计准则**：工程部署需在训练计算量（FLOPs）与权重显存之间取舍，现有工作未提供可操作的联合寻优公式。

## 核心贡献（创新点）
1. **提出Loop Scaling Laws统一框架**：首次将循环次数 $R$ 与专家数 $E$ 作为显式缩放轴纳入扩展定律，并严格证明其可还原为标准稠密定律、循环稠密定律与非循环MoE定律为特例。
2. **设计有界且稀疏条件化的循环映射**：用指数衰减形式替代无界线性/幂律映射，刻画循环收益的渐近饱和；通过活跃比例 $m$ 调节映射的幅度与收敛速率，量化稀疏性对有效容量的提升作用。
3. **建立计算-显存联合优化准则**：基于拟合定律推导固定FLOPs预算下的最优循环 $R^\star$、固定显存预算下的最优专家数 $E^\star$，以及两者的联合帕累托最优配置。
4. **实证万亿token规模的互补收益**：证明稀疏性提供约3倍活跃参数效率，循环提供约2倍总参数推理效率；在匹配计算量下，0.3B活跃/1.3B总参的LoopMoE可匹敌0.6B活跃/2.9B总参的非循环MoE，并支持测试时按需扩展。

## 方法详解
- **基础缩放定律**：标准形式 $\mathcal{L}(N, D) = A N^\alpha + B D^\beta + c$，训练计算 $F_{\text{train}} = 6 N_{\text{unroll}}(R) D$。
- **有界循环映射**：有效参数 $N_{\text{eff}}^{\text{bounded}}(R) = N + \kappa_1 N_{\text{loop}} (1 - e^{-(R-1)/\kappa_2})$，其中 $\kappa_1$ 控制渐近增益上限，$\kappa_2$ 控制收敛速度；$R=1$ 时退回非循环基线，$R \to \infty$ 时增益有界。
- **稀疏条件化映射**：定义活跃比例 $m = N_{\text{act}}/N_{\text{total}}$（$m$ 越小越稀疏），令 $\kappa_j(m) = \kappa_j m^{-\theta}$。稀疏度越高，渐近线上移且收敛变缓，反映不同循环路径可访问更多专家参数。
- **MoE循环扩展定律**：
  $$\mathcal{L}(N_{\text{act}}, D, R, E, m) = A \hat{E}^\delta N_{\text{eff}}(R,m)^{\alpha+\gamma \ln \hat{E}} + B \hat{E}^\omega D^{\beta+\zeta \ln \hat{E}} + c$$
  其中 $\hat{E}$ 为专家数的单调变换。该式在 $E=1$ 时退化为稠密循环定律，$R=1$ 时退化为标准MoE定律，$R=E=1$ 时退化为Chinchilla定律。
- **架构优化目标**：
  - 计算最优循环：$R^\star = \arg\min_R \mathcal{L}(\cdot)$，s.t. $6 N_{\text{unroll}}(R) D = \bar{F}_{\text{train}}$ 且 $\Delta \mathcal{L}(R) \geq \epsilon$（$\epsilon$ 取拟合RMSE）。
  - 显存最优专家数：$E^\star = \arg\min_E \mathcal{L}(\cdot)$，s.t. 计算预算固定且权重显存 $\mathcal{M}_{\text{weight}} \leq M_{\text{budget}}$。
  - 联合最优：在二维预算约束下遍历 $(N_{\text{act}}, E, R)$ 候选空间，选取预测损失最低的配置。

## 实验与结果
- **缩放实验设置**：$N_{\text{act}} \in \{0.3, 0.6, 1.0\}$B，$D \in \{100, 200, \dots, 500\}$B，$E \in \{1, 2, 4, 8, 16\}$，$R \in \{1, 2, 3, 4, 6, 8\}$；采用中间块循环（middle-cycle looping），sigmoid gating + top-k routing。
- **预测精度对比**：有界映射在held-out $R/N/D$ 上的RMSE显著优于无界映射（held-out $R$: 0.0092 vs 线性0.2566 / 幂律0.0313），准确捕捉了损失 plateau 现象。
- **IsoFLOP 前沿分析**：固定计算下，提升稀疏度（$E=1 \to 8$）直接压低损失；固定稀疏度下，增加循环可在相同 $N_{\text{act}}$ 上获得更低损失，且高稀疏度场景更倾向于选择更高 $R^\star$。
- **下游基准（14个，覆盖推理/科学/常识/阅读/知识）**：稀疏性带来 $\sim 3\times$ 活跃参数效率，循环带来 $\sim 2\times$ 总参数推理效率；联合缩放 consistently 超越单一轴基线。
- **万亿token实测案例**：A0.3B-1.3B LoopMoE ($E=8, R^\star=5$) 与 A0.6B-2.9B MoE ($E=8, R=1$) 在 matched compute ($\sim 1.5 \times 10^{22}$ FLOPs) 下训练。LoopMoE在推理基准（BBH, GSM8K）上匹配大模型，且测试时可动态调整 $R$（1→5），Overall 提升 +10.3 分（36.6 →
