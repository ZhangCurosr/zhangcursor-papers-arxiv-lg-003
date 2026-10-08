---
title: "Sparse-Planning-in-Visual-World-Models-via-Cost-Gradients"
source: https://arxiv.org/pdf/2610.10274v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:11:38"
---

# 论文速读：Sparse Planning in Visual World Models via Cost Gradients

## 一句话总结
提出 COSTGRAD，一种免训练、目标条件驱动的视觉世界模型稀疏规划 token 选择器，通过规划代价对输入 token 的梯度范数排名并保留 Top-K，在 50% 稀疏度下匹配或超越全 token 规划成功率，同时带来 2.6× 至约 5× 的端到端加速。

## 研究问题与动机
1. **计算瓶颈**：基于 token 的视觉世界模型（如 DINO-WM/JEPA-WMs）在推理时需在完整空间网格上进行 CEM 动作搜索，单步规划涉及约 1.4×10^7 次 token 级前向计算，开销极高。
2. **目标错配**：现有稀疏方法（如 Sparse Imagination）依赖随机选择或预测导向信号，未解决“预测目标与控制目标不一致”的客观错配（objective mismatch）问题。
3. **信号不可靠**：直接使用预测器的注意力权重或预测误差选 token 存在缺陷——注意力受训练损失制约且跨训练种子不稳定（IoU 仅 0.44），而预测误差高估的是视觉可预测性而非控制重要性（例如运动轨迹可预测的物体恰是规划核心）。

## 核心贡献（创新点）
1. **提出 COSTGRAD 选择器**：首次将下游规划目标本身作为 token 重要性来源，通过单次反向传播计算规划代价梯度范数实现免训练、目标条件驱动的稀疏选择；与 PREDATTN/PREDGRAD 等预测导向基线的本质区别在于直接对齐控制目标而非预测保真度。
2. **验证稀疏规划的速度-成功率帕累托优势**：在 AdaLN-Zero 预测器上 50% 稀疏度时，四点三
