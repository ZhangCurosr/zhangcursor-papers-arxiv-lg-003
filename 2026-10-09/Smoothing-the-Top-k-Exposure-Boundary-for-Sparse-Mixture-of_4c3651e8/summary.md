---
title: "Smoothing-the-Top-k-Exposure-Boundary-for-Sparse-Mixture-of"
source: https://arxiv.org/pdf/2610.11575v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:57:21"
---

# 论文速读：Smoothing-the-Top-k-Exposure-Boundary-for-Sparse-Mixture-of-Experts

## 一句话总结
本文针对稀疏 MoE 模型静态 top-k 路由在训练时形成的刚性曝光边界问题，提出 Elastic Expert Routing。该方法在训练阶段围绕目标 expert 数量 k 进行局部对称离散采样，将陡峭的阈值函数平滑为渐进的选择概率分布，在严格保持推理计算预算不变的前提下，显著改善了边界附近专家的梯度反馈，提升了监督微调与从零预训练的性能。

## 研究问题与动机
1. **静态 top-k 的刚性边界导致优化脆弱**：传统训练固定激活 top-k 个专家，将连续 routing 分布转化为刚性阶跃函数。排名恰在 cutoff 附近的专家会因微小分数波动被任意划分到全监督区或零反馈区，造成梯度断层。
2. **动态路由难以直接部署**：现有 adaptive routing 方法通过动态调整激活数量或修改模型结构来打破阈值，但往往改变部署预算或依赖复杂训练技巧，难以适配现有模型服务基础设施。
3. **边界附近竞争性专家未获合理训练**：static top-k 使紧邻 cutoff 的专家无法获得稳定的任务损失反馈，限制了 expert 优先级的精细排序与容量挖掘。
4. **k 仅被视作计算约束而未被重新建模**：现有工作多将 k 当作固定超参，未从“训练曝光分布塑造”的角度系统性改进路由优化过程。

## 核心贡献（创新点）
1. **提出 Elastic Expert Routing，将硬阈值平滑为渐进概率分布**：通过在训练期围绕目标 k 对激活 expert 预算进行局部对称采样软化曝光边界，与已有工作的本质区别在于不修改路由架构与推理规则，仅改变训练时的预算分布。
2. **在 SFT 与从零预训练中均实现稳定性能增益**：OLMoE-1B-7B 与 Qwen3-30B-A3B 微调分别提升 +0.84 和 +2.02 个 Avg(9) 点；1.42B MoE 从零预训练平均提升 1.6 点，证明该方法作为通用训练期修改的迁移价值。
3. **建立 Rank-Neighborhood Smoothing 的理论解释**：严格推导离散预算分布与 expert 曝光概率的关系，证明期望激活数守恒且 cutoff 两侧曝光转移平衡，为弹性路由提供可解释的
