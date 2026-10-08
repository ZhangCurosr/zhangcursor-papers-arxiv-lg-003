---
title: "Stability-and-Diversity-of-Networked-Self-Consuming-Generati"
source: https://arxiv.org/pdf/2610.09409v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:09:38"
---

# 论文速读：Stability-and-Diversity-of-Networked-Self-Consuming-Generati

## 一句话总结
本文首次构建了网络化自消耗生成模型的理论框架，将多模型交互抽象为有向加权图，并从理论上严格刻画了合成数据混合比例与交互图拓扑如何共同决定系统的长期稳定性与输出多样性。

## 研究问题与动机
- 现有自消耗训练（self-consuming training）研究主要局限于单一模型递归训练或简化的双模型耦合系统，无法刻画真实生成生态中多模型通过复杂、有向、加权路径共享合成数据的网络交互现象。
- 随着合成数据广泛渗透训练管道，跨模型依赖（如 Orca、Phi-3、DALL·E 3、PixArt-α）已成常态，但合成数据在网络中的传播如何系统性影响整体生态的渐近行为尚缺乏统一理论。
- 已有工作缺乏对任意交互拓扑、非对称边权及异构模型架构的分析框架，难以量化“局部放大效应”与“全局传播效应”的耦合机制。
- 实际部署亟需可诊断分布漂移、模型崩溃与多样性丧失风险的理论工具，以指导数据配比、图结构设计及缓解策略制定。

## 核心贡献（创新点）
- **统一图动力学框架**：将异构生成模型集合建模为有向加权交互图节点，形式化任意跨模型合成数据流动与迭代重训练映射，严格一般化现有单模型/双模型理论。
- **谱半径稳定性判据**：导出固定点 Jacobian 的块上界，定义局部放大因子 $\beta_i$ 与比较矩阵 $\mathbf{D}$，证明 $\rho(\mathbf{D})<1$ 保证局部渐近稳定，并给出基于图谱范数 $\|\mathbf{W}\|_2$ 的易验证充分条件。
- **拓扑依赖的多样性收缩理论**：推导固定点输出表示的闭式线性系统，证明合成数据消费会引发依赖图拉普拉斯的平滑效应，严格量化网络连通性（代数连通度 $\mu_2$）对系统多样性的压制上界。
- **理论-实证闭环验证**：在合成 8-Gaussian 数据集与真实 CIFAR-10 数据集上，使用 DDPM/CFM/GAN 等多类生成模型及多种交互拓扑，验证了稳定性阈值、环增益主导性及多样性衰减的定性规律。

## 方法详解
- **迭代重训练动力学**：设 $K$ 个模型 $\mathcal{V}$，交互矩阵 $\mathbf{W}$ 行随机（$w_{ij}\ge0, \sum_j w_{ij}=1$）。第 $t$ 轮模型 $i$ 的更新为：
  $\pmb{\theta}_i^t \in \arg\max_{\pmb{\theta}} \mathbb{E}_{p_i^{\text{data}}}[\log p_\pmb{\theta}(x)] + \lambda_i \sum_{j=1}^K w_{ij} \mathbb{E}_{p_{\pmb{\theta}_j^{t-1}}}[\log p_\pmb{\theta}(x)]$
- **固定点与正则假设**：固定点 $\bar{\Theta}$ 满足一阶最优性条件。假设局部强凹性（$\alpha_i>0$）与 Hessian 分布偏移 Lipschitz 连续性（常数 $L_i$），定义分布间隙 $\varepsilon_i = \max_{j:w_{ij}>0} d
