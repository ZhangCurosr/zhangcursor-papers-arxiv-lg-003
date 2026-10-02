---
title: "SCALABLE-COX-REGRESSION-VIA-GROUPED-RISK-SETS-AND-SHARPER-LO"
source: https://arxiv.org/pdf/2609.40120v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:36:27"
field: "大规模生存分析与凸随机优化"
keywords: ["Cox regression", "LogSumExp optimization", "stochastic approximation", "grouped risk sets", "softplus surrogate", "asymptotic equivalence", "survival analysis"]
innovations: ["对 softplus LogSumExp 代理建立 O(T^{-1/2}) 均值率与 O(log T/T) 最后迭代率", "基于嵌套风险集分组的二阶梯度误差控制与曲率转移", "压缩估计器在显式调度下保持全数据 Cox 估计器的渐近正态性"]
benchmarks: ["SUPPORT2", "NWTCO", "Correlated 100k", "Correlated 1M", "Independent 100k"]
---

# 论文速读：SCALABLE-COX-REGRESSION-VIA-GROUPED-RISK-SETS-AND-SHARPER-LO

## 一句话总结
本文针对大规模 Cox 回归的计算瓶颈，提出了一种基于**分组风险集**与**softplus 代理**的可伸缩随机优化方法；该方法为每个风险集组引入一个辅助标量，实现了无偏单样本梯度，并在光滑凸 LogSumExp 目标上建立了 $O(T^{-1/2})$ 的均值收敛率与 $\tilde{O}(T^{-1})$ 的最后迭代收敛率，最终将端到端计算误差控制在 $O((\log T/T)^{4/5})$，且压缩估计器保留了全数据的渐近正态性。

## 研究问题与动机
- **大规模 Cox 回归的瓶颈**：传统 Cox 偏似然为 LogSumExp 型目标，风险集呈嵌套结构，精确求解依赖全数据遍历，难以应对分布式/流式/超出内存的大规模数据。
- **mini-batch 估计的偏差**：直接用小批次替换风险集会引入系统性的梯度偏差，使优化器收敛到一个与原始全数据目标不同的“批次依赖”目标。
- **现有无偏方法的代价**：变分/辅助变量方法虽可得到无偏梯度，但通常需为每个事件保留一个辅助变量（状态 $O(m)$），且在强凸正则下缺乏明确常数的先验收敛界。
- **缺乏统一误差分解框架**：现有工作多将近似误差与随机优化误差分开处理，缺少对“分组近似 + softplus 近似”联合误差的定量控制与转移定理。

## 核心贡献（创新点）
1. **更 sharp 的软凸 LogSumExp 分析**：对光滑凸 LogSumExp 目标建立 $O(T^{-1/2})$ 均值目标界；在原变量加强凸正则时，在不要求辅助变量方向强凸的前提下，给出 $\tilde{O}(\log T / T)$ 的最后迭代平方误差界。
   - **本质区别**：相比 Gladin 等 (2025) 的 $O(T^{-1/4})$，利用期望平滑性获得更优速率；同时给出带显式常数的先验界（非轨迹累积方差依赖）。
2. **嵌套风险集的确定性分组压缩**：将相邻失败按风险集大小相对变化 $\le \delta$ 分组，每组共享一个辅助位移；证明分组带来的梯度扰动为二阶（在组内风险集变化中为 $O(L_q r_\delta^2)$），曲率保持由不变量的 $\kappa$ 与 $L_q$ 控制。
   - **本质区别**：不同于子采样/一次校正（改变目标分布）或 MCMC 近似（需递增精度），此处保留固定全数据目标的确定性一致逼近。
3. **端到端计算速率与统计转移定理**：结合优化界与分组/softplus 界，得到关于全 Cox 解的均方误差 $O((\log T/T)^{4/5})$；在样本增大下，只要 $\sqrt{N}(L_{q,N}^2 \delta_N^2 + \rho_N \kappa_N)\to 0$，压缩估计器与全数据估计器同分布。
   - **本质区别**：将优化误差、分组误差与统计渐近统一在相同框架，并给出子线性辅助状态数 $K_N = o_p(N)$ 的构造。
4. **公开实现与一致评测**：提供开源代码与复现实验协议，在合成与真实生存数据上与 Batch LSE、Minibatch Cox、BigSurv、Cox-CC 在匹配向量预算下比较。
   - **本质区别**：以正则化全 Cox 训练 gap 与测试 C-index/loss 为统一度量，并在多 seed 下报告中位数与四分位。

## 方法详解
- **通用 LogSumExp 目标**：$J(\theta)=R(\theta)+\sum_k p_k \log \mathbb{E}_k e^{L_k(X,\theta)}$。引入 softplus 代理 $h_\rho(u)=\rho^{-1}\log(1+\rho e^u)$ 与辅助位移 $s_k$，形成联合目标 $G_\rho(\theta,s)$ 及其轮廓 $J_\rho(\theta)=\min_s G_\rho(\theta,s)$。
- **无偏单样本梯度**：每次均匀采样组 $k$ 与样本 $X$，计算权重 $w=1/(\rho+\exp(s_k-L_k(X,\theta)))$，更新 $\nabla R(\theta)+Kp_k w\nabla L_k$ 与 $g_{s,k}=Kp_k(1-w)$，其余位移不变。
- **加权投影步**：在加权范数 $\|z\|_P^2=\|\theta\|^2+\sum p_k s_k^2$ 下执行投影梯度，$\theta$ 用普通投影，采样位移 $s_k$ 剪切至 $[-B-1,B]$。
- **收敛分析关键工具**：期望平滑性（Lemma A.2）给出梯度二阶矩与目标差的线性关系；强凸情形用受限割线不等式（Lemma A.4）得到几何收缩。
- **Cox 分组规则**：按时间排序失败，贪心合并相邻失败到组 $\mathcal{I}_k$，使组内最大/最小心风险集大小比 $\le 1+\delta$；设 $r_\delta=\delta/(1+\delta)$，每个组内均匀抽失败 $I$ 与风险集内受试者 $J$。
- **数据不变量**：$\kappa$ 为归一化指数矩上界（风险权重分散）；$L_q$ 为短区间内离开风险集的受试者平均风险权重与原风险集平均之比的上界，二者独立于分组。
- **梯度/曲率控制定理（Theorem 4.1）**：在 $L_q r_\delta\le 1/2$ 与 $\rho\kappa\le 1/9$ 下，梯度误差上界 $X(L_q^2 r_\delta^2+4\rho\kappa)$，Hessian 损失被 $X^2(2.5 L_q^2 r_\delta^2+20\rho\kappa)I$ 控制；若原目标 Hessian $\succeq \nu I$，则压缩后目标亦保持 $\nu/2$ 强凸。
- **调度选择**：取 $\delta_T=(\log T/T)^{1/5}$、$\rho_T\propto (\log T)/(\delta_T T)$、$\eta_T\propto (\log T)/T$，平衡随机误差 $O(\log T/(\delta_T T))$ 与分组偏差平方 $O(\delta_T^4)$，得 $\mathbb{E}\|\beta_{T+1}-\beta^\star\|^2=O((\log T/T)^{4/5})$。
- **统计转移（Theorem 4.3）**：若 $\sqrt{N}(L_{q,N}^2\delta_N^2+\rho_N\kappa_N)\xrightarrow{p}0$，则 $\sqrt{N}(\hat\beta_N^{\rho,\delta}-\hat\beta_N)\xrightarrow{p}0$，继承全数据 Cox 估计器的渐近正态性；取 $\delta_N=N^{-1/4}/\ell_N$、$\rho_N=N^{-1/2}/\ell_N$ 即可。

## 实验与结果
- **数据集**：SUPPORT2 ($N{=}8{,}873$, $d{=}22$, $m_{tr}{=}3{,}621$)、NWTCO ($N{=}4{,}028$, $d{=}11$, $m_{tr}{=}342$)、Correlated 100k/1M、Independent 100k 合成数据。
- **基线**：Batch LSE、Minibatch Cox (Zeng et al. 2026)、BigSurv (Tarkhan & Simon 2020)、Cox-CC (Kvamme et al. 2019)。
- **设置**：60/20/20 分层划分；所有方法共享 $\lambda{=}10^{-3}$、$\beta_0{=}0$、Breslow 处理打结；向量预算 $U_{HPO}{=}\max\{3\cdot10^6,100N\}$、$U_{final}{=}1.5U_{HPO}$；12 配置网格跨两个 tuning seed 选优。
- **优化精度**：在匹配预算下，本文方法在所有 5 个数据集上达到**最低中位数终端正则化全 Cox 训练 gap**（Table 4）。例如 SUPPORT2 为 $3.93\cdot10^{-5}$，Correlated 1M 为 $1.39\cdot10^{-6}$。BigSurv 次之，差距 1.22–26.15 倍（Table 4）。
- **目标达成**：在阈值 $10^{-5}$ 下，仅本文方法在 Correlated 100k（7/10）与 Correlated 1M（10/10）达到并保持（Table 5）。
- **预测性能**：均值 Test Cox loss 在 NWTCO、Correlated 100k、Independent 100k 上最低；Test C-index 在 NWTCO 与 Correlated 100k 上最高（Tables 6–7）。
- **近似诊断**：在 $\beta_{ref}$ 处测得 grouping 梯度误差对数斜率 1.37–1.87，softplus 误差对数斜率 0.997–1.000，验证理论阶次（Figure 2, Table 8）。
- **压缩幅度**：$\delta{=}0.05$ 时，辅助位移数相比每事件一个减少 22.8–337.8 倍（Table 1）。

## 相关工作脉络
- **精确 Cox 优化**：Simon et al. (2011) 利用嵌套风险集与坐标下降做全数据遍；本文面向随机/分布访问场景，保留相同目标。
- **MCMC 近似**：Achab et al. (2015) 在风险集中做 MCMC 并用方差缩减；本文避免 MCMC，用确定性分组与 softplus 保证无偏梯度。
- **最优子采样/一次校正**：Zhang et al. (2024)、Wang et al. (2024) 用子样恢复全数据估计；本文对梯度进行一致逼近而非改换目标分布。
- **嵌套 case-control**：Goldstein & Langholz (1992) 有大样本保证；Kvamme et al. (2019) 将其用于神经网络；本文在凸线性框架下给出显式常数与曲率转移。
- **LogSumExp 优化**：Wang & Yang (2022, 2025) 研究耦合复合优化；SCENT (Wei et al. 2026b) 用变分形式也达 $O(T^{-1/2})$ 但依赖轨迹方差；本文给出固定参数的先验界与最后迭代速率。
- **位置差异**：本文区分计算近似与统计模型误差，给出联合误差分解、确定性分组定理与渐近等价性证明。

## 局限性与未来方向
- 分析假设线性预测、协方差有界与均匀曲率条件（$\nabla^2\mathcal{L}\succeq\nu I$）；高异质性下 $\kappa$、$L_q$ 可能较大。
- 统计转移定理针对**互异失败时间**；实现用 Breslow 处理打结，但理论未直接覆盖。
- 未讨论时变协变量、非线性预测器与打结时间的渐近理论。
- 实验未证明相对于已优化的累计和全 Cox 求解器的端到端运行时间优势；本方法面向随机/流式访问场景。
- 未来可在变分近似、自适应分组与分布式实现上拓展，并纳入打结与时间依赖扩展的理论框架。

## 研究启发与可借鉴点
- **软凸代理的期望平滑分析**可迁移到其它带外层 $\log\mathbb{E}e^{\cdot}$ 结构的优化（如 softmax、对比学习、分布鲁棒优化）。
- **分组误差的二阶控制**思想（梯度扰动为组内变化之积）可用于具有嵌套/滑动窗口的复合目标。
- **显式常数先验界**与轨迹无关的收敛分析值得在大规模 compositional 优化中推广。
- **联合优化-统计转移范式**（将计算误差与统计误差在同一条件下合成）可复用于其他近似 M-估计器。
- 开源协议与多 seed 中位数报告可作为后续生存分析随机优化实验的标准对照。

## 关键术语表
- **LogSumExp 目标**：形如 $\log\sum e^{f_i(\theta)}$ 的光滑凸目标，广泛出现在 softmax、风险度量与生存模型中。
- **softplus 代理**：$h_\rho(u)=\rho^{-1}\log(1+\rho e^u)$ 逼近 $e^u$，引入辅助位移后可得无偏梯度。
- **风险集**：在某一失败时刻仍“处于风险中”的受试者集合，Cox 似然中用于归一化。
- **分组风险集**：将相邻失败按风险集大小相对变化 $\delta$ 合并，共享一个归一化变量以降低状态维度。
- **$\kappa$（归一化指数矩）**：衡量风险权重在风险集内的相对分散程度，控制 softplus 近似误差。
- **$L_q$（局部风险权重比）**：衡量在短区间内离开风险集的受试者权重与原集合权重之比，控制分组误差。
- **最后迭代收敛**：直接对最后一轮迭代给出误差界，而非对 iterate 平均，便于实际使用。
- **渐近等价**：压缩估计器与全数据估计器在 $\sqrt{N}$ 尺度下差趋于零，共享同一极限正态分布。

## 可复现要素
- **代码**：公开于 https://github.com/elizkaveta/Cox-Regression-Analysis/
- **数据集**：SUPPORT2、NWTCO 为公开数据；合成数据生成细节在附录 D.1。
- **关键超参**：$\lambda{=}10^{-3}$、$\rho{=}10^{-4}$、$\delta{=}0.05$、$\eta_t{=}\eta_0(1+t/1000)^{-1/2}$、初始位移 $s_{k,0}{=}\log(1-\rho)$；批大小 $b$ 与 $\eta_0$ 经验证选择。
- **预处理**：事件分层 60/20/20 划分，训练集拟合插补/标准化，Breslow 打结处理。
- **评估**：向量预算 $U_{HPO}{=}\max\{3\cdot10^6,100N\}$、$U_{final}{=}1.5U_{HPO}$；指标为正则化全 Cox 训练 gap、测试 Cox loss、Harrell C-index。
