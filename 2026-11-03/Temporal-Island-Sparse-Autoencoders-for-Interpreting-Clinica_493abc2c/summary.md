---
title: "Temporal-Island-Sparse-Autoencoders-for-Interpreting-Clinica"
source: https://arxiv.org/pdf/2611.03420v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-11-04 18:23:06"
---

# 论文速读：Temporal-Island-Sparse-Autoencoders-for-Interpreting-Clinical-EHR

## 一句话总结
提出 TI-SAE（Temporal Island Sparse Autoencoder），一种完全 post-hoc 的可解释框架，从冻结的 EHR 预测 backbone 表示中学习“时间高斯岛 + 稀疏变量路由”双组件词典，生成具备时间连贯性、跨变量结构对齐且患者共享的临床概念，在多个 backbone 与三类预测任务上显著超越传统归因与概念基线方法。

## 研究问题与动机
- EHR 深度学习模型（Discrete、HyMaTE、BAT、STraTS 等）预测性能优异，但内部表示不可解释，阻碍临床信任与落地。
- 现有 post-hoc 归因方法（Integrated Gradients、SHAP/TimeSHAP、Dynamask、TimeX 等）仅输出逐 cell 重要性分数，**缺乏时间连贯性、跨变量结构语义与跨患者可复用性**。
- 标准 SAE 倾向于恢复 token-specific 或高度局部的特征，与临床时间跨度长、结构化共现的概念不对齐（Bhalla et al., 2026）。
- Concept-based 方法（CBM、TCAV）通常需修改 backbone 架构或依赖人工定义概念方向，难以从稀疏网格数据中自动发现具有学习时间范围的潜在概念。

## 核心贡献（创新点）
1. **提出双组件时间-变量解耦词典结构**：每个 latent 由 M 个时间高斯岛（定义“何时激活”）与稀疏变量路由向量 w_k（定义“涉及哪些变量”）构成。*与已有工作的本质区别：不同于标准 SAE 的纯向量激活或 CBM 的人工概念，显式建模临床概念的时间轮廓与变量共现模式。*
2. **设计 IslandParamNet 与四组件架构**：包含 encoder、IslandParamNet（μ/σ 估计）、island selector（能量门控）、decoder，引入 dead latent revival、μ-repulsion、decoder diversity、mask coverage、channel purity 等正则化策略。*与已有工作的本质区别：专为稀疏临床时序网格设计的参数化高斯岛估计与预算门控机制，避免 islands 坍缩并控制选择开销。*
3. **分离 Stage A（字典学习）与 Stage B（预测感知选择器）**：Stage A 在冻结 backbone 上学习患者共享词典；Stage B 独立训练 softmax 选择器，通过 logit 拟合与 completeness 损失路由至 n_global^pred=64 个最相关 island。*与已有工作的本质区别：解耦概念发现与任务特定解释，保证词典泛化性同时提供可调控的 faithfulness。*
4. **系统性验证与临床可用性评测**：在 5 个 backbone × 3 个预测任务 × 2 个数据集上全面评估，并通过临床医生 Likert 评分与 LLM judge 验证概念的可理解性与可操作性。*与已有工作的本质区别：首次在不同架构 backbone 上统一对比 post-hoc 归因、动态掩码、概念瓶颈等方法，建立 EHR 可解释性的系统基准。*

## 方法详解
- **字典参数化**：latent k 对应 M 个高斯岛 {μ_{k,m}, σ_{k,m}}_{m=1}^M 与稀疏变量路由 w_k ∈ R^V。岛生成时间注意力轮廓 g_{k,m}(t) = exp(-(t-μ_{k,m})²/(2σ_{k,m}²))，w_k 经 top-k_dec 稀疏化与行归一化。
- **四组件架构**：
  1. `encoder`：将 backbone 隐藏状态 H_grid ∈ R^{B×K×T} 编码。
  2. `IslandParamNet`：两层 1D Conv (H=32, kernel=25, zero-padding) 提取时序特征；逐点投影生成 attention logits 计算 μ_{k,m}；σ_{k,m} 通过 EM 估计（t=3），无 learnable 参数，截断于 [0.5, 12.0]。
  3. `island selector`：能量门控评分 s_{k,m} = (Σ_t A_0[b,k,t]·g_{k,m}(t)) / √(Σ_t g_{k,m}(t)) · gate_{k,m}，gate 由独立 linear layer 产出（梯度仅经
