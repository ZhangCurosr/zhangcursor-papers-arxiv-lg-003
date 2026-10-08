---
title: "RoBART-Bayesian-Additive-Regression-Trees-with-Tree-Specific"
source: https://arxiv.org/pdf/2610.10214v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:08:21"
---

# 论文速读：RoBART: Bayesian Additive Regression Trees with Tree-Specific Rotations

## 一句话总结
本文提出 RoBART，为贝叶斯加性回归树（BART）的每一棵树分配独立的正交旋转矩阵，在旋转后的坐标空间内保持轴对齐分割与常数叶节点；同时设计了可逆的联合 Givens 旋转与切割点更新算法，并在理论上证明了其对含各向异性 Hölder 光滑性的加性函数的后验收缩率，且从理论上证明了标准轴对齐 BART 无法达到相同的收缩速度。

## 研究问题与动机
- **核心问题**：标准 BART 仅支持沿原始预测变量轴的轴对齐分割，当真实回归决策边界与坐标轴不平行（如斜线、椭圆等高线）时，需要极深的树与大量分割才能近似，导致表达冗余与采样效率低下。
- **现有方法不足**：
  1. 轴对齐 BART 无法自适应学习潜在的低维倾斜结构，后验质量受限于固定坐标网格，理论收缩率在高维各向异性真值下显著劣化。
  2. Oblique BART 等变体为每个内部节点独立学习超平面法向量，缺乏全局结构约束，且缺乏完整的后验收缩理论保证。
  3. 随机旋转集成（如 Random Rotation Ensembles）在拟合前预固定旋转矩阵，无法根据数据后验联合优化。
  4. 现有文献缺少对“连续旋转参数 + 离散切割点网格”联合 MCMC 更新的细致平衡证明，阻碍了理论分析与工程落地的结合。

## 核心贡献（创新点）
1. **模型架构创新**：提出 RoBART，每棵树共享一个 $p \times p$ 的 $\mathrm{SO}(p)$ 旋转矩阵，在旋转坐标系内进行轴对齐分割并保持常数叶节点，兼顾斜切表达能力与经典树结构的可解释性。
2. **MCMC 算法创新**：设计基于 Givens 旋转序列的联合提议机制，
