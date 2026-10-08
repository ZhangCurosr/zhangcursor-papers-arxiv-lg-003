---
title: "Sparsifying-Stochasticity-Not-Capacity-Partial-Stochasticity"
source: https://arxiv.org/pdf/2610.09886v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:11:30"
field: "贝叶斯深度学习"
keywords: ["Bayesian neural networks", "partial stochasticity", "deep weight factorization", "MMD", "universal conditional density approximation", "type-II MAP"]
innovations: ["将深度权重分解应用于先验尺度以实现随机性稀疏化而非容量稀疏化", "证明 MMD 二次核下函数先验拟合退化为矩匹配并给出 UCDA 线性时间证书", "刻画混合推理为 type-II MAP 随机近似并证明耦合步长的非消失跟踪误差"]
benchmarks: ["UCI CONCRETE", "UCI WINE", "UCI BIKESHARING", "UCI BOSTON", "UCI BANKNOTE", "UCI HTRU2", "UCI ONLINESHOPPERS", "UCI CREDIT"]
---

# 论文速读：Sparsifying-Stochasticity-Not-Capacity-Partial-Stochasticity

## 一句话总结
本文提出将深度权重分解（Deep Weight Factorization, DWF）应用于贝叶斯神经网络先验标准差，通过 MMD 拟合函数先验至高斯过程，自动学习参数中随机/确定的划分；随机性被稀疏化而网络容量保留——落入阈值以下先验尺度的参数变为推理时可优化的确定性参数。在 UCI 基准上，该方法以约一半参数为确定性的代价，性能与全随机网络相当。

## 研究问题与动机
- **核心问题**：贝叶斯神经网络不需要所有参数都随机化即可获得良好不确定性估计（Sharma et al., 2023），但"哪些参数应随机化"的选择问题仍开放，组合搜索空间巨大且不可行。
- **现有方法的不足**：
  - spike-and-slab 等稀疏方法通过置零参数剪枝，同时削减了网络容量；
  - MAP -based 子网选择方法（如 SNI-PSNN）依赖完整网络训练结果，且确定性参数被固定而非继续优化；
  - 部分随机网络的混合推理方案（采样+优化并行）缺乏理论刻画，其耦合步长会留下不随步长衰减的跟踪误差。

## 核心贡献（创新点）
1. **DWF 作用于先验尺度以实现"随机性稀疏化"**：对先验尺度做 DWF 参数化，优化时低于阈值的尺度坍缩为零使参数变为确定性；与非零尺度之间存在理论间隙（Theorem 4.1，当 $D \geq 3$ 时），与剪枝类方法保留容量的本质区别在于确定性参数在推理阶段仍自由优化。
2. **MMD 拟合函数先验至 GP**：用二次多项式核的 MMD 作为目标，等价于匹配函数输出的一阶与二阶矩（Proposition 4.3）；KL 散度的 Pythagorean 分解解释了稀疏网络最多只能匹配矩（Lemma A.3）。
3. **线性时间可检查的 UCDA 证书及最小修复**：给出一个可 $O(|\Theta^{(1)}|)$ 时间内验证 universal conditional density approximation 的充分条件（Proposition 4.7），不满足时通过 Corollary A.9 修复不超过 34 个参数。
4. **混合推理的理论刻画**：证明混合方案是 type-II MAP 目标上的随机近似 EM（Theorem 4.9），并证明耦合步长会产生不随步长消失的跟踪误差，需解耦（Corollary 4.10）。

## 方法详解
- **先验尺度参数化**：对每个参数 $\theta$ 引入 DWF 因子 $(\omega_\theta^{(i)})_{i=1}^D$，令 $u_\theta = \prod_i \omega_\theta^{(i)}$，$\sigma_\theta = H(u_\theta) = (h(u_\theta)-h(0))^2$，其中 $h(u)=\mathrm{softplus}(u)$。先验设为 $\theta \sim \mathcal{N}(\mu_\theta, \sigma_\theta^2)$，$\mu_\theta$ 和 $\omega_\theta$ 均为可训练参数。
- **优化目标**：$\min_{\mu, \omega} \mathcal{L}_k(\mu, H(\mathbf{u})) + \sum_l \frac{\lambda_l}{D} \sum_{\theta \in \Theta^{(l)}} \sum_i (\omega_\theta^{(i)})^2$，其中 $\mathcal{L}_k$ 为 MMD 期望，用随机测量点上的函数样本经验估计。
- **MMD 到矩匹配的简化**：二次多项式核下 $\mathrm{MMD}_2^2(P,Q) = 2\|\mu_P-\mu_Q\|^2 + \|M_P-M_Q\|_F^2$，权重由核固定无需调参。
- **测量点分布**：$\mathcal{G} = 0.7\hat{\mathbb{P}}_{\mathrm{train}} + 0.3\mathcal{U}$，均匀分量强制训练集外区域的矩匹配，对 OOD 不确定性有直接影响。
- **阈值与稀疏性**：优化后低于 cutof $c$ 的 $\sigma_\theta$ 设为 0（确定性）；当 $D\geq 3$ 时，非零先验尺度在驻点处有下界 $H(-t_l)>0$，与零值之间存在间隙（Theorem 4.1），使稀疏划分稳定。
- **UCDA 证书**：若第一层随机偏置数 $U(S) \geq d_L$ 且全确定性神经元数 $|T| \geq d_0$，则掩码网络具有 universal conditional density approximation 性质。
- **推理**：确定性参数用 Adam 在 $\mathcal{N}(\hat{\mu}_d, \tau^2 I)$ 先验下优化（$\tau=\gamma\log 2$ 控制漂移幅度）；随机参数用 SGHMC 采样；耦合步长会导致不消失的跟踪误差，因此优化器步长需快于采样器衰减。

## 实验与结果
- **数据集**：UCI 8 个数据集（回归：CONCRETE、WINE、BIKESHARING、BOSTON；分类：BANKNOTE、HTRU2、ONLINESHOPPERS、CREDIT），每数据 10 次随机 9:1 划分。
- **基线**：ALS-BNN（全随机）、FLS-BNN（仅第一层随机）、RAND-PSNN（随机掩码同预算）、SNI-PSNN（MAP-based 选择）。
- **主要结果（Table 3 平均秩）**：
  - 分类：DWF-PSNN 在 ACC（2.38）与 NLL（2.63）上均排第一，ECE（2.88）次之；显著优于 SNI-PSNN。
  - 回归：DWF-PSNN 以 RMSE 2.00 排第一（ALS-BNN 3.25），NLL 2.25 与 ALS-BNN 并列。
  - **关键数字**：DWF-PSNN 约 47-61% 参数为确定性（中层约保留一半随机），性能与 ALS-BNN（全随机）相当或更优。
- **Bimodal target 实验**：学习到的划分在 6.5%-99.6% 确定性比例范围内 $W_1 \approx 0.06$ 保持稳定；跨层随机置换比同预算随机划分最差高两个数量级。cutof $c$ 在超过一个数量级范围内变化对模型影响微乎其微。

## 相关工作脉络
1. **Sharma et al. (2023)** 证明部分随机 MLP 可 universal conditional density approximate，但理论未指导具体选择——本文从"如何选择"出发。
2. **Tran et al. (2022)** 用 sliced Wasserstein 距离拟合函数先验至 GP；本文改用 MMD（二次核退化为矩匹配），目标更精确且无超参调权。
3. **Kolb et al. (2025)** 提出深度权重分解用于 $\ell_{2/D}$ quasi-norm 稀疏学习；本文首次将其应用于先验尺度而非网络权重，实现稀疏随机性而非稀疏容量。
4. **Daxberger et al. (2021) (SNI-PSNN)**：先训练全网络再 MAP 选择子网；本文的划分来自函数先验拟合，无需先训练，且确定性参数仍参与推理优化。
5. **Neal (1996) ARD**：通过超先验连续压缩尺度实现软稀疏，但划分从不显式；本文 DWF 产生显式稀疏。
6. **Li et al. (2024)** 稀疏子空间 VI 直接剪除参数、减少容量；本文确定性参数保持容量不变。

## 局限性与未来方向
- **仅限 MLP**：证书仅覆盖带固定底层堆栈的 head（卷积/Transformer），底层共享权重不在分析范围内；修复要求 head 宽度超过特征维度，多数架构不满足。
- **闭环动力学未覆盖**：Adam 更新使用当前样本，存在优化器与采样器之间的反馈回路，当前理论仅分析开环跟踪误差，闭环分析留待未来。
- **高维输入依赖修复**：输入维度大的数据集（如 CREDIT，$d_0=23$）证书频繁失效，需修复多个参数；修复后效果虽可接受但偏离原始学习目标。

## 研究启发与可借鉴点
1. **DWF 应用范式创新**：将深度权重分解从"权重稀疏化"迁移至"先验尺度稀疏化"，思路可直接移植到其他稀疏先验学习任务，或与变分推断结合。
2. **MMD 二次核=矩匹配的工程简化**：无需调权重的矩匹配目标可作为函数先验拟合的通用替代方案，尤其适合需要控制模型表达力的场景。
3. **UCDA 证书的检查与修复机制**：线性时间可验证的充分条件 + 最小修复策略，可作为其他概率神经网络建模的安全保障工具。
4. **混合推理的步长解耦设计**：耦合步长产生不可收敛跟踪误差的发现，指导了后续实验中优化器学习率比采样器快 10 倍的设置，可推广至其他混合采样-优化算法。
5. **与团队方向结合机会**：本方法在不确定性感知的稀疏贝叶斯深度学习中具有潜力，可探索在大规模时序/表格数据上的扩展，或将 MMD 矩匹配目标与扩散模型的 score matching 结合。

## 关键术语表
- **DWF (Deep Weight Factorization)**：将参数分解为多个因子的 Hadamard 积，使 $\ell_{2/D}$ quasi-norm 正则化可通过可微方式实现，$D\geq3$ 时产生强于 $\ell_1$ 的稀疏性。
- **Prior scale**：参数先验分布的标准差 $\sigma_\theta$，本文对其施加 DWF 参数化，其坍缩与否决定参数是否随机化。
- **MMD (Maximum Mean Discrepancy)**：衡量两个分布差异的核均值距离；二次多项式核下 MMD² 退化为均值与二阶矩的 Frobenius 距离。
- **UCDA (Universal Conditional Density Approximation)**：模型在 mild 拓扑条件下可任意逼近任意条件分布的能力。
- **Type-II MAP**：对随机参数边际化后，对确定性参数优化的 $\log Z(\theta_d) - \frac{1}{2\tau^2}\|\theta_d-\hat\mu_d\|^2$ 目标，本文证明混合方案等价于其随机近似。
- **Hybrid inference scheme**：部分随机网络中同时对随机参数采样、对确定性参数进行梯度优化的联合推理框架。
- **Functional prior**：通过参数先验诱导的网络函数空间上的先验分布 $p(\mathbf{f};\Psi)=\int p(\mathbf{f}|\Theta)p(\Theta;\Psi)d\Theta$。
- **Tracking error**：采样器分布追随移动中目标分布时产生的误差，耦合步长下该误差不随步长趋于零。

## 可复现要素
- **数据集**：UCI ML Repository（Kelly et al., 2023），均已公开；bimodal target 为文中构造的合成数据。
- **代码**：基于 Tran et al. (2022) 代码库扩展，原文仓库许可证未声明，代码未单独开源。
- **关键超参**：MLP 两层各 100 单元、tanh 激活；DWF 深度 $D=3$；cutof $c=0.01$；$\lambda_l \propto l$（层权重比 3:4:5）；SGHMC 学习率 0.01-0.03，Adam 0.001-0.003，Adam 衰减更快（$\alpha_\mathrm{Adam}=0.4 > \alpha_\mathrm{SGHMC}=0.2$）；$\gamma=2.5$；测量点混合比例 $\rho_M=0.7$。
