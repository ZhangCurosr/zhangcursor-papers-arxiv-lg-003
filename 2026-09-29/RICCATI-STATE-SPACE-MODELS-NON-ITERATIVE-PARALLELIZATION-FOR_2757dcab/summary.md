---
title: "RICCATI-STATE-SPACE-MODELS-NON-ITERATIVE-PARALLELIZATION-FOR"
source: https://arxiv.org/pdf/2609.35441v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:56:09"
field: "状态空间模型与序列建模"
keywords: ["状态空间模型", "Mobius变换", "并行扫描", "Riccati方程", "非线性序列建模", "液体递归网络", "长序列预测"]
innovations: ["利用Riccati方程精确离散流为Mobius变换的性质，实现非线性状态依赖动态的单次并行扫描评估", "推导稳定参数化保证有界不变区间与状态收缩性，避免极点发散", "证明RiccatiSSM与LrcSSM的二阶Taylor对应关系，分离截断误差与离散化误差"]
benchmarks: ["UEA长序列分类（Heartbeat, SCP1, SCP2, Ethanol, Motor, Worms）", "PPG-DaLiA回归", "Weather长期预测"]
---

# 论文速读：RICCATI STATE SPACE MODELS: NON-ITERATIVE PARALLELIZATION FOR NONLINEAR SEQUENCE MODELING

## 一句话总结
论文提出 RiccatiSSM，一种非线性状态空间模型，通过将每个状态维度建模为输入条件化的 Riccati 微分方程，利用其精确离散流为 Mobius 变换的性质，使状态更新在 $2\times 2$ 矩阵乘法下闭合，从而实现单次关联并行扫描的精确非迭代序列评估，相比迭代非线性 LrcSSM 减少 22–33% 运行时间。

## 研究问题与动机
1. **线性 SSM 缺乏状态依赖动态**：现有 SSM（如 Mamba、S⁶）的状态更新为仿射形式，局部 Jacobian 不依赖当前状态，无法捕捉状态依赖的时间常数与收缩率变化。
2. **非线性 RNN 难以并行化**：LTC、LrcSSM 等液体递归模型引入状态依赖动态，但破坏了组合闭合性，需通过 Newton/Picard 迭代线性化（如 DEER、Quasi-DEER）反复扫描，计算开销大且收敛不确定。
3. **非线性与非迭代并行的矛盾**：如何在保持状态依赖非线性动态的同时，实现精确的单次并行扫描评估，仍是未解决问题。

## 核心贡献（创新点）
1. **提出 RiccatiSSM**：将每个状态维度建模为输入条件化的 Riccati ODE，其精确零阶保持离散流为分式线性（Mobius）变换，满足组合闭合性。
2. **单次并行扫描的非迭代评估**：利用 Mobius 变换对应 $2\times 2$ 矩阵乘法的性质，通过单次关联前缀扫描计算完整状态轨迹，无需 Newton 迭代或定点方程求解。
3. **稳定参数化设计**：推导约束参数化保证有界不变区间 $[-B, B]$、状态收缩性（Jacobian $\leq -\varepsilon s_{\min} < 0$），避免分母为零的极点发散。
4. **与 LrcSSM 的理论关联**：证明 Riccati 动态是 LRC 向量场在 $x=0$ 处的二阶 Taylor 近似，数值实验表明离散化误差主要来自 LrcSSM 的显式 Euler 积分而非二次截断。
5. **高效的长序列预测性能**：在分类、回归、预测基准上与 Transformer、SSM、神经 ODE 等方法竞争，同时比 LrcSSM 快 22–33%。

## 方法详解
1. **输入条件化 Riccati 动态**：每个状态维度 $i$ 的连续时间动态为 $\dot{x}_i = \varepsilon_i(u)[\alpha_i(u) + \beta_i(u)x_i + \gamma_i(u)x_i^2]$，其中系数由输入 $u$ 的投影头生成，局部 Jacobian $\partial \dot{x}_i/\partial x_i = \varepsilon_i(\beta_i + 2\gamma_i x_i)$ 显式依赖当前状态。
2. **射影提升与精确离散化**：将标量 Riccati ODE 提升为二维线性系统 $\frac{d}{dt}(p, q)^\top = L_t(p, q)^\top$，其中 $L_t = \varepsilon_t\begin{pmatrix}\beta_t/2 & \alpha_t \\ -\gamma_t & -\beta_t/2\end{pmatrix}$ 是无迹矩阵。在零阶保持下，精确一步解为 $M_t = \exp(\Delta t L_t) = \cosh(\omega_t\Delta t)I + \frac{\sinh(\omega_t\Delta t)}{\omega_t}L_t$，诱导状态更新 $x_t = \frac{a_t x_{t-1}+b_t}{c_t x_{t-1}+d_t}$，即 Mobius 变换。
3. **关联并行扫描**：由于 Mobius 变换复合对应矩阵乘法，定义前缀积 $P_t = M_t M_{t-1}\cdots M_1$，通过单次 $O(\log T)$ 深度的并行扫描计算所有 $P_t$，最终状态由 $x_t = \frac{P_t^{(11)}x_0 + P_t^{(12)}}{P_t^{(21)}x_0 + P_t^{(22)}}$ 恢复，总计算量 $O(TD)$。
4. **稳定参数化**：通过网络输出 $(\hat{\alpha}, \hat{\beta}, \hat{\gamma}, \hat{\varepsilon})$，经约束映射保证：$\varepsilon = \sigma(\hat{\varepsilon})\in(0,1)$，$s = s_{\min}+\text{softplus}(\hat{\beta})$，$\beta = -(s + 2|\gamma|B)$，$\alpha = (sB+|\gamma|B^2)\tanh(\hat{\alpha})$，确保判别式 $\beta^2/4 - \alpha\gamma \geq s^2/4 > 0$，轨迹始终在有界区间内且收缩。

## 实验与结果
- **数据集**：UEA 六组长序列分类（Heartbeat 405、SCP1 896、SCP2 1152、Ethanol 1751、Motor 3000、Worms 17984）、PPG-DaLiA 回归、Weather 预测（720 步上下文预测 720 步）。
- **基线**：Transformer/RFormer、NRDE、NCDE、Log-NCDE、LRU、S⁵、Mamba、S⁶、LinOSS-IMEX/IM、LrcSSM。
- **分类结果**：RiccatiSSM 在 SCP1（87.4%±4.3）、Motor（59.6%±3.1）、Worms（90.6%±1.4）上达到最高或接近最高；与 LrcSSM 总体竞争力相当。
- **回归结果**：PPG-DaLiA 上 MSE 7.15±1.13×10⁻²，优于 LrcSSM（10.89±0.96）与 LinOSS-IM（6.40±0.23）。
- **预测结果**：Weather MAE 0.5681，优于 LrcSSM（0.5888）与 S⁴（0.5783），接近 LinOSS-IMEX（0.5081）。
- **运行时间**：在相同 6 层架构、状态维度 64 下，RiccatiSSM 比 LrcSSM 快 22–33%（Figure 2），在长序列 EigenWorms 上加速最显著。
- **消融**：移除输入依赖速度因子 $\varepsilon(u)$ 使 MSE 上升 1.3×10⁻²；限制 $\beta,\gamma,\varepsilon$ 为输入独立使 MSE 升至 8.71±1.10；LRC-tied 参数化 MSE 8.47±0.47，劣于自由参数化（7.15±1.01）。

## 相关工作脉络
1. **DEER/ELK 系列**：将序列评估视为非线性轨迹求解，通过 Newton 迭代线性化后并行扫描，需 $K$ 次迭代（数据依赖），RiccatiSSM 通过选择组合闭合的动态族避免迭代。
2. **LrcSSM**：液体电阻-电容递归模型，对角 Jacobian 设计降低迭代成本，但仍需约 3 次 quasi-DEER 迭代；RiccatiSSM 与之竞争性能的同时实现单次扫描。
3. **Affine SSM（S⁵、Mamba、S⁶）**：仿射状态更新支持并行扫描，但局部 Jacobian 不依赖状态；RiccatiSSM 通过二次项引入状态依赖时间常数。
4. **LTC/CfC**：液体时间常数网络，状态依赖动态但序列递归仍为顺序，无法并行化；RiccatiSSM 保留状态依赖特性同时实现并行。
5. **Kalman Linear Attention**：使用 Mobius 递归于不确定性统计量，状态线性演化；RiccatiSSM 将 Mobius 作用于状态本身，产生状态依赖动力学。
6. **Fixed-Point RNNs**：将密集线性递归表达为可并行对角系统的定点；RiccatiSSM 通过精确组合闭合避免定点迭代。

## 局限性与未来方向
1. **动态形式受限**：RiccatiSSM 仅适用于二阶多项式形式（$\alpha+\beta x+\gamma x^2$），高阶或非多项式非线性无法直接扩展。
2. **系数不能依赖状态**：$\alpha,\beta,\gamma,\varepsilon$ 只能依赖输入 $u$，若允许依赖状态 $x$ 会破坏组合闭合性。
3. **长序列优化敏感性**：在 EigenWorms 数据集上 seed-to-seed 变异较大（77.8%–91.7%），可能反映对优化路径敏感。
4. **未来方向**：可扩展至多变量 Riccati 系统、探索其他组合闭合非线性族（如线性分式其他子类）、与选择性机制结合、探索在语言建模等任务上的适用性。

## 研究启发与可借鉴点
1. **组合闭合动态族的设计思路**：通过数学结构分析（Mobius 变换的矩阵表示）将非线性问题转化为线性代数问题，为设计可并行非线性和非迭代评估模型提供范式。
2. **射影提升技巧**：将一维非线性 ODE 提升为二维线性系统再通过投影恢复，类似技巧可应用于其他具有精确解的非线性方程族。
3. **稳定参数化的设计原则**：通过约束网络输出映射到保证有界区间和收缩性的参数空间，避免数值不稳定，此策略可迁移至其他连续时间模型。
4. **与 LRC 的动力学对应分析**：通过 Taylor 展开建立与液体模型的联系，并用匹配动态实验分离截断误差与离散化误差，分析方法值得借鉴。
5. **单次并行扫描 vs 迭代扫描的权衡**：实验表明消除迭代扫描可获得 22–33% 加速，提示在序列建模中优先选择组合闭合动态的价值。

## 关键术语表
**RiccatiSSM**：论文提出的非线性状态空间模型，每个状态维度遵循输入条件化的 Riccati 微分方程，精确离散流为 Mobius 变换，支持单次并行扫描评估。
**Mobius 变换（分式线性变换）**：形式为 $f(x)=(ax+b)/(cx+d)$ 的映射，可通过 $2\times 2$ 矩阵表示且复合对应矩阵乘法，具有组合闭合性。
**关联并行扫描（Associative Parallel Scan）**：利用算子结合律在 $O(\log T)$ 深度、$O(T)$ 工作量下计算前缀积的并行算法，适用于仿射或 Mobius 更新。
**零阶保持（Zero-Order Hold, ZOH）**：假设系数在一个时间步内保持恒定，用于连续时间动力学的离散化，RiccatiSSM 在此假设下获得精确闭式解。
**LrcSSM**：液体电阻-电容状态空间模型，通过迭代 quasi-DEER 并行化非线性递归，对角 Jacobian 设计降低每次迭代成本。
**DEER**：通过 Newton 迭代求解非线性序列轨迹的并行方法，每步线性化后执行并行扫描，需多次迭代至收敛。
**射影提升（Projective Lift）**：将标量 Riccati ODE 映射为二维线性系统的表示技巧，通过 $x=p/q$ 恢复非线性动态，使组合闭合成为可能。
**稳定参数化**：对 Riccati 系数施加约束，保证状态轨迹有界不变、转移收缩且无极点，通过软 plus 和 tanh 等映射实现。

## 可复现要素
- **数据集**：UEA 时间序列分类基准（公开）、PPG-DaLiA（公开）、Weather（公开）。
- **代码/权重**：论文未提供公开代码仓库链接，但声明代码基于 Rusch & Rus (2025) 和 Farsang & Grosu (2025) 的实现构建；依赖 Discretax 库（Nazari et al., 2025）。
- **关键超参**：学习率 $\{10^{-5}, 10^{-4}, 10^{-3}\}$，隐藏维度 $\{16, 64, 128\}$，状态空间维度 $\{16, 64, 256\}$，层数 $\{2, 4, 6\}$；各数据集最优配置见 Table 7。
- **硬件**：NVIDIA A40/A100 GPU。
- **稳定参数化超参**：收缩边界 $B>0$、最小收缩率 $s_{\min}>0$、时间步长 $\Delta t$；论文未明确具体数值。
