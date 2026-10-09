---
title: "TRANSFER-UNCERTAINTY-TRANSFER-FROM-SMALL-TO-LARGE-NETWORKS-U"
source: https://arxiv.org/pdf/2610.11668v1.pdf
model: agnes-2.5-flash
chunks: 3
summarized_at: "2026-10-09 17:01:35"
---

# 论文速读：TRANSFER-UNCERTAINTY-TRANSFER-FROM-SMALL-TO-LARGE-NETWORKS-U

## 一句话总结
本文提出 σTransfer，在最大更新参数化（μP）框架下构造宽度归一化先验 $S_n=T_nT_n^\top$，使 Laplace 近似的先验核随网络宽度稳定收敛，从而实现先验精度与不确定性衍生决策（主动学习选点、OOD 检测、拒答门控）从小网络到大网络的零样本迁移，无需在目标大模型上重新验证扫描。

## 研究问题与动机
- Laplace 近似预测不确定性高度依赖先验精度 $\lambda$，传统做法需在目标模型验证集上进行网格扫描，对十亿参数级网络计算代价极高。
- 在标准参数化（SP）下，相同 $\lambda$ 在不同网络宽度下产生显著不同的不确定性估计，导致小模型选定的超参无法直接迁移至大模型。
- 现有宽度无关理论（如 μP）主要面向训练动态与损失 landscape 的迁移，尚未系统解决 Laplace 后验在宽度扩展下的谱稳定性与跨尺度复用问题。
- 缺乏无需重构目标大模型后验即可直接复用不确定性信号的工程化路径，制约了 Bayes 近似在低成本代理→大模型迁移场景中的落地。

## 核心贡献（创新点）
- 提出 σTransfer 先验重参数化：将各向同性先验 $\lambda I$ 替换为 $\lambda S_n^{-1}$，以 μP 宽度缩放因子 $T_n$ 构造 $S_n$，在不引入新超参的前提下实现先验核宽度归一化。
- 建立“先验核稳定→后验均值/协方差稳定→精度与决策一致”的收敛理论链条，给出有限步 ReLU MLP、全 GGN、验证 NLL 曲线与连续评分规则下的误差界（定理 1-4）。
- 设计 Regime A（精度迁移）与 Regime B（决策迁移）双模式：前者实现 $\lambda^*_n$ 零样本复用，后者跳过目标后验构造直接驱动主动学习/OOD/拒答等下游决策。
- 在回归、图像分类与 Transformer readout 任务上验证有效性，十亿参数 1B→7B 迁移平均 NLL 增加 <10⁻⁴，扫描加速最高达 5000×，显著优于 SP 与 μP isotropic 基线。

## 方法详解
- **先验协方差归一化**：标准 Laplace 先验为 $\mathcal{N}(\theta_0,\lambda^{-1}I)$。σTransfer 将其改为 $\mathcal{N}(\theta_0,\lambda^{-1}S_n^{-1})$，其中 $S_n=T_nT_n^\top$，$T_n$ 为 μP 规定的宽度相关缩放矩阵，从而对齐不同宽度下的梯度尺度与 Hessian 特征谱。
- **先验核稳定化机制**：归一化后先验核 $K_n$ 随网络宽度 $n\to\infty$ 收敛至极限核 $K_\infty$，避免宽度过度增加导致的后验置信区间膨胀或收缩失真。
- **Regime A（精度迁移）**：在源小网络（宽度 $n$）验证集上扫描得到最优 $\lambda^*_n$，直接作为目标大网络（宽度 $N$）的 Laplace 先验精度，避免在大模型上重复扫描。
- **Regime B（决策迁移）**：完全跳过目标大网络的 Laplace 后验求解，利用小网络已计算的后验不确定性信号（预测方差、熵、置信区间）直接指导主动学习样本选择、OOD 判别阈值或拒答门控。
- **理论支撑**：定理 1-4 分别从固定深度 ReLU MLP、完整 GGN 近似、验证 NLL 曲线单调性及连续评分规则角度证明收敛性，excess NLL 上界为 $2\sup|\text{NLL}_n-\text{NLL}_N|$。

## 实验与结果
- **数据集与任务**：回归（ESOL、FreeSolv、Lipophilicity）、图像分类（MNIST、FMNIST、PenDigits、Letter）、主动学习（AG News）、大规模 LLM（公开 1B→7B 多任务）。
- **精度迁移性能**：MNIST 128→4096 扫描加速约 5000×（1.5s vs 7560s），目标 NLL 退化仅 0.002；公开 1B→7B 模型中位加速 ~2.3×（最高 ~330×），十任务平均 NLL 增加 <10⁻⁴，七任务选到相同精度。
- **回归任务对比**：σTransfer 的 $|\Delta\lambda|$ 仅 0.4–0.9 bits（SP isotropic 为 1.5–5.0 bits）；$|\Delta\text{NLL}|$ 低一个数量级以上（FreeSolv: 0.461 vs 452.8×10⁻³）。
- **分类与 OOD 检测**：MNIST/FMNIST 128→4096 下 σTransfer 的 $|\Delta\text{NLL}|$ 最小（最坏 0.6×10⁻²）；width=4096 时 OOD AUROC 达 0.950，与目标调优一致（SP 降 0.072，μP isotropic 降 0.327）。
- **主动学习（决策迁移）**：AG News 128→2048，Spearman $\rho$ 达 0.75（SP isotropic 仅
