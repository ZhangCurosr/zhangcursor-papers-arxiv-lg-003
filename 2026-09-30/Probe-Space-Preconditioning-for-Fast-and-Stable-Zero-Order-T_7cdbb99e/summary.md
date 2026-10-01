---
title: "Probe-Space-Preconditioning-for-Fast-and-Stable-Zero-Order-T"
source: https://arxiv.org/pdf/2609.38095v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:26:05"
---

# 论文速读：Probe-Space-Preconditioning-for-Fast-and-Stable-Zero-Order-T

## 一句话总结
论文提出 1.5-SPSA，一种面向大模型后训练的零阶优化求解器，通过在探针空间引入廉价的对角曲率预条件与α饱和加权，在严格保持推理模式内存占用的前提下，将收敛步数从 MeZO 的 10 万步压缩至百步以内，并在 OPT-13B/30B 上取得超越 MeZO 甚至 BP+Adam 的测试精度。

## 研究问题与动机
- **显存瓶颈**：BP+Adam 训练 OPT-30B 需约 600GB 显存，迫使依赖昂贵集群与复杂分片；ZOO 可在推理模式下训练，显存仅需约 60GB，但收敛极慢。
- **病态损失地貌**：神经网络损失面的 Hessian 特征值跨度极大（实测可达 $10^8$ 量级），传统 ZOO 缺乏曲率感知，导致后期训练不稳定甚至发散。
- **计算预算分配不明**：现有 ZOO 工作多将前向传播预算投向“多步小扰动”，未系统探索步数、effective batch size 与扰动数之间的最优配比。
- **内存与预条件的矛盾**：Adam 等对角预条件器需 $O(d)$ 状态存储，与推理模式内存约束直接冲突，亟需一种 $O(1)$ 额外内存的曲率估计方案。

## 核心贡献（创新点）
- **预算重分配发现**：在固定前向传播预算下，将计算资源从“多步小batch”转向“少步大 effective batch + 大量并行扰动”，可使 1SPSA 在数十步内超越 MeZO。
- **1.5-SPSA 曲率预条件**：每步额外复用一次干净前向传播，以三步模板估计探针方向标量曲率，并通过 α 饱和加权压制高曲率方向，实现无状态开销的稳定快速收敛。
- **JL 引理的理论推广**：从 Johnson-Lindenstrauss 引理出发，形式化证明随机扰动子空间可高概率保留局部曲率信息，为探针空间预条件提供几何理论支撑。
- **端到端系统工程**：结合 8-bit 打包随机生成器、Triton 融合解包/应用核与种子广播分布式策略，在单节点 8×A100 上实现 OPT-30B 级模型的快速推理模式训练，整体加速约 2.76×。

## 方法详解
- **基础梯度估计**：沿用 1SPSA 中央差分公式 $\hat{g}(\theta) = \frac{1}{2n_{pert}} \sum_{i=1}^{n_{pert}} \frac{L(\theta+\epsilon z_i) - L(\theta-\epsilon z_i)}{2\epsilon} z_i^{-1}$，扰动 $z_i$ 采样自 Rademacher 分布（$\pm 1$），实现极简内存占用。
- **探针空间曲率估计**：利用共享的中心点 $L(\theta)$，对每个扰动三元组
