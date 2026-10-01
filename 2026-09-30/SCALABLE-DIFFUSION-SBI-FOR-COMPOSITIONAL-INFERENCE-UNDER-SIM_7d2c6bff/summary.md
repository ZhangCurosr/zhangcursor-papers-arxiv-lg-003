---
title: "SCALABLE-DIFFUSION-SBI-FOR-COMPOSITIONAL-INFERENCE-UNDER-SIM"
source: https://arxiv.org/pdf/2609.36950v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 17:16:05"
---

# 论文速读：SCALABLE-DIFFUSION-SBI-FOR-COMPOSITIONAL-INFERENCE-UNDER-SIM

## 一句话总结
本文提出了一种统一的扩散基近似贝叶斯计算（SBI）框架，将预训练联合扩散模型作为可复用推理引擎，在采样阶段动态实现多源异构观测的组合聚合与层次结构解耦，并结合路径正则化微调有效适配模拟器误差设定，无需针对新场景重新训练。

## 研究问题与动机
1. **组合推断的计算瓶颈**：传统SBI方法（如JAC、GAUSS）在处理大量异构观测时依赖显式Jacobian或辅助协方差估计，计算复杂度随观测数 $n$ 快速增长。
2. **层次结构的建模割裂**：真实科学数据（如细胞通路）同时包含共享参数与组特定潜变量，现有方法多采用pooled采样将观测合并为单一后验，丢失组间异质性与独立latent mode。
3. **模拟器误差设定（Misspecification）**：预训练模型基于理想模拟器生成，与真实观测存在分布偏移；全量重训成本高昂，亟需轻量级适配机制。
4. **可移植性缺失**：缺乏一套通用范式能同时支持组合聚合、层次划分与似然迁移，导致“每换实验场景必重新训练”的重复劳动。

## 核心贡献（创新点）
1. **推导n依赖连续时间扩散系数（F-NPSE-SDE）**：将离散F-NPSE扩展至高效SDE采样，理论推导避开JAC的Jacobian与GAUSS的辅助协方差估计，扩散系数随 $n$ 自然收敛至 $1/n$（契合Bernstein–von Mises渐近性）。
2. **提出层次块式扩散采样（HBDS）**：仅依赖单一预训练联合扩散模型，在采样时通过交替更新与分数合成实现共享参数与组特定潜态的解耦推断，无需重训层级估计器。
3. **设计路径正则化微调（Path-regularized finetuning）**：借助Implicit Diffusion学习隐式变换 $T:\mathcal{V}_{\text{sim}}\to\mathcal{V}_{\text{obs}}$，通过共享token embeddings与FiLM实现likelihood→posterior转移，微调仅作用于轻量模块。
4. **构建路径空间Fisher几何（Path-space Fisher geometry）**：基于条件反向SDE路径律定义局部Fisher–Rao度量，利用Girsanov定理量化PT↔FT路径散度，
