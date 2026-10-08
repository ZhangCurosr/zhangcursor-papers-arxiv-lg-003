---
title: "SoftSEEPS-improves-ML-based-precipitation-forecasting"
source: https://arxiv.org/pdf/2610.09752v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:07:54"
---

# 论文速读：SoftSEEPS-improves-ML-based-precipitation-forecasting

## 一句话总结
本文提出了SEEPS评分的可微近似损失SoftSEEPS，使降水预报模型能够直接针对气象业务中广泛使用的干/轻/重分类指标进行端到端训练，并在0.1° IMERG数据集上验证了其与RMSE联合优化可在几乎不牺牲连续精度前提下带来显著的分类评分提升。

## 研究问题与动机
1. **核心问题**：灾害预警与气候适应业务高度依赖SEEPS（Stable Equitable Error in Probability Space）评估降水预报的干湿重分类质量，但该指标基于分段常数阈值划分，数学上不可微，无法直接作为ML模型的训练目标。
2. **现有方法不足-指标错配**：主流ML天气预报模型（如FourcastNet、GraphCast、Pangu-Weather等）普遍采用RMSE或对数RMSE训练，但降水分布具有强偏态与间歇性，RMSE无法真实反映高空间分辨率下防灾业务关心的分类准确率。
3. **现有方法不足-缺乏可微近似**：尽管学界已指出不应将RMSE作为降水训练目标，但Rasp等WeatherBench系列相关工作均未提供SEEPS的可微分近似，导致评估与训练目标长期脱节。
4. **动机**：打通“业务分类评分→梯度优化”的链路，使模型在训练中内嵌气候学阈值先验，直接面向early-warning系统所需的干湿重判断能力进行学习。

## 核心贡献（创新点）
1. **提出SoftSEEPS可微损失**：通过Sigmoid函数对硬阈值三分类进行平滑松弛，构造完全可微的SEEPS近似损失，使位置与日历年依赖的罚矩阵S可直接参与反向传播。（与以往仅用连续误差训练的ML预报模型本质不同，首次将业务equitable评分转化为训练信号）
2. **设计τ动态退火策略**：不固定平滑参数，而是结合验证集离散SEEPS的Plateau Scheduler自动降低τ，在梯度条件性与阈值近似精度之间实现自适应平衡。（区别于Gumbel-Softmax等固定温度松弛策略）
3. **揭示联合优化的Pareto前沿**：系统扫描λ证明
