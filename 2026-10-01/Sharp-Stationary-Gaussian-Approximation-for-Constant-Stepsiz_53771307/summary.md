---
title: "Sharp-Stationary-Gaussian-Approximation-for-Constant-Stepsiz"
source: https://arxiv.org/pdf/2609.39144v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:38:52"
field: "随机优化理论"
keywords: ["常数步长 SGD", "Markovian 噪声", "高斯近似", "Wasserstein 距离", "Poisson 方程", "平稳分布", "Lindeberg 插值"]
innovations: ["将常数步长SGD平稳分布的W_1高斯近似上界从O(sqrt(alpha)log(1/alpha))改进至O(sqrt(alpha))", "构造四状态Markov噪声下界实例，揭示三阶时序混合矩是主导修正阶的来源", "双Poisson方程方法同时处理Markov噪声均值与条件协方差的时序依赖"]
---

# 论文速读：Sharp-Stationary-Gaussian-Approximation-for-Constant-Stepsize-SGD

## 一句话总结
本文针对外生一致遍历 Markov 链驱动的常数步长 SGD，证明了中心化的迭代量按步长平方根缩放后，其平稳分布与 Ornstein-Uhlenbeck 极限高斯分布之间的 1-Wasserstein 距离为 $O(\sqrt{\alpha})$；并构造了一个四状态噪声例子给出了匹配的 $\Omega(\sqrt{\alpha})$ 下界，揭示了高阶时序依赖（而非噪声边际偏置）是导致这一修正阶的主导因素。

## 研究问题与动机
1. **常数步长 SGD 平稳分布的高斯近似精度尚未明确**：现有工作（Wang et al., 2026）仅给出 $O(\sqrt{\alpha}\log(1/\alpha))$ 的 $W_1$ 上界，精度不够"锐利"。
2. **Markovian 噪声的时序依赖如何影响高斯近似的误差阶**：虽然噪声边际对称且所有非零滞后自协方差为零，但相邻高阶混合矩仍能产生 $O(\sqrt{\alpha})$ 阶修正，现有理论未能刻画这一机制。
3. **从有限块到平稳律的转移缺乏紧致的耦合分析**：已有方法多依赖集中不等式或 Lyapunov 技术，难以在 Wasserstein-1 框架下给出直接与低界匹配的结果。
4. **目标函数曲率与噪声时序结构的交互作用尚不明确**：强凸性与 Hessian Lipschitz 条件对 $O(\sqrt{\alpha})$ 误差的贡献分解缺乏系统刻画。

## 核心贡献（创新点）
1. **将高斯近似的 $W_1$ 上界从 $O(\sqrt{\alpha}\log(1/\alpha))$ 改进至 $O(\sqrt{\alpha})$（Theorem 1）**：通过块wise Gaussian comparison 与长期收缩结合，消除了对数因子，与下界匹配。
2. **构造四状态 Markov 噪声实例给出匹配的 $\Omega(\sqrt{\alpha})$ 下界（Theorem 2）**：该例中噪声边际对称、所有非零滞后自协方差为零，但主导修正来自相邻三阶混合矩 $\mathbb{E}[\xi(Z_0)\xi(Z_1)^2] = -9/4$，揭示了独立同分布假设下不可见的时序效应。
3. **提出双 Poisson 方程解耦方法：噪声均值修正 + 条件协方差修正**：第一个 Poisson 方程将 Markov 噪声分解为鞅差加 coboundary；第二个方程将可预测的条件协方差波动 telescope 为边界项与变分和，两者共同刻画时序依赖进入高斯近似的两个精确入口。
4. **建立联合过程的收缩性证明（Proposition 4）**：构造带尺度因子 $\sqrt{\alpha}$ 的度量 $d_{\alpha,\kappa}$，利用强凸性和 coalescent coupling 证明单块映射为严格收缩映射，从而将块级近似转移至平稳分布。

## 方法详解

**问题设定**：考虑递归 $X_{k+1} = X_k - \alpha\nabla f(X_k) + \alpha\xi(Z_k)$，其中 $f$ 为 $m$-强凸、$L$-光滑且 Hessian 为 $M$-Lipschitz，$(Z_k)$ 为一致遍历 Markov 链（收敛速率 $\rho^k$），$\xi$ 有界且 $\pi$-零均值。

**平稳高斯基准**：设 $H = \nabla^2 f(x_\star)$，长 Run 协方差 $\Sigma_M = \sum_{\ell\in\mathbb{Z}}\Gamma_\ell$，目标高斯分布为 $\mathcal{N}(0,\Sigma)$，其中 $\Sigma$ 满足 Lyapunov 方程 $H\Sigma + \Sigma H = \Sigma_M$。

**尺度变换**：定义 $Y_k = (X_k - x_\star)/\sqrt{\alpha}$，则更新变为 $Y_{k+1} = \mathcal{T}_\alpha(Y_k) + \sqrt{\alpha}\,\xi(Z_k)$，其中 $\mathcal{T}_\alpha(y) = y - \sqrt{\alpha}\nabla f(x_\star+\sqrt{\alpha}y)$。

**单块长度**：取 $n = \lceil T/\alpha\rceil$，使确定性收缩因子约 $e^{-mT}$，一步收缩率为 $1-m\alpha$，$n$ 步后收缩为固定常数。

**双 Poisson 方程**：
- 噪声修正：$V = \sum_{j\geq 0} P^j\xi$，满足 $V-PV=\xi$，定义鞅差 $D_{k+1} = V(Z_{k+1}) - PV(Z_k)$，从而 $\xi(Z_k) = D_{k+1} + V(Z_k) - V(Z_{k+1})$。
- 协方差修正：$\mathcal{C}(z) = P(VV^\top)(z) - (PV)(z)(PV)(z)^\top$，满足 $\int\mathcal{C}\,d\pi = \Sigma_M$；令 $W = \sum_{j\geq 0}P^j(\mathcal{C}-\Sigma_M)$，则 $W-PW = \mathcal{C}-\Sigma_M$，实现条件协方差波动的 telescoping。

**证明路径（五步）**：
1. **漂移线性化**：用 $A = I_d - \alpha H$ 线性化，非线性余项贡献 $O(\sqrt{\alpha})$（利用 Hessian Lipschitz 和块内矩有界）。
2. **Markov 和 → 鞅和**：通过首个 Poisson 方程的求和分部，余余项 $R_{V,n}$ 满足 $\|R_{V,n}\|\leq C_T\sqrt{\alpha}$。
3. **鞅增量 → 高斯增量（Lindeberg 插值）**：以 Gaussian 初始条件 $Y_0\sim\gamma$ 提供的平滑性（Hessian 有界且 Lipschitz），逐位置替换 $B_rD_r \to B_rN_r$，二次 Taylor 余项合计 $O(\sqrt{\alpha})$；第二个 Poisson 方程使条件协方差波动项 telescope 为边界项 $O(\alpha)$ 与变分项 $O(\sqrt{\alpha})$。
4. **协方差识别**：离散协方差 $\Sigma_{\alpha,n}$ 与连续解 $\Sigma$ 的差异满足 $\|\Sigma_{\alpha,n}-\Sigma\|_F \leq C_T\alpha$，对应 $W_1$ 误差 $O(\alpha)$。
5. **平稳转移**：利用 coalescent coupling（会合时间几何尾部）+ 强凸收缩，证明联合过程在度量 $d_{\alpha,\kappa}$ 下为压缩映射，从而将单块误差 $O(\sqrt{\alpha})$ 传递至不变测度，得 $W_1(\mathcal{L}(Y_\alpha),\gamma)\leq C\sqrt{\alpha}$。

**下界构造（四状态例）**：取 $f(x)=x^2/2$，$Z_k=(\eta_{k-1},\eta_k)$ 为 i.i.d. Rademacher 对的 Markov 链，$\xi(Z_k)=\eta_k(3-\eta_{k-1})/2$。该例边际对称（取值 $\{-2,-1,1,2\}$ 等概率），$\mathbb{E}[\xi(Z_0)\xi(Z_\ell)]=0\ (\ell\neq 0)$，$\Sigma_M=5/2$，但 $\mathbb{E}[\xi(Z_0)\xi(Z_1)^2]=-9/4\neq 0$。通过两状态特征函数递归直接计算得 $\mathbb{E}[\sin(Y_\alpha)]=\frac{3}{8}e^{-5/8}\sqrt{\alpha}+O(\alpha)$，结合 $h(x)=\sin x$ 的 1-Lipschitz 性质给出 $W_1\geq c_0\sqrt{\alpha}$。

## 实验与结果
- **主要理论结果**：$W_1(\mathcal{L}((X_\alpha-x_\star)/\sqrt{\alpha}),\mathcal{N}(0,\Sigma))\leq C\sqrt{\alpha}$（Theorem 1），且存在常数 $c_0>0$ 使得 $W_1(\mathcal{L}(Y_\alpha),\mathcal{N}(0,5/4))\geq c_0\sqrt{\alpha}$（Theorem 2），上下界匹配，阶次最优。
- **最强结果**：$O(\sqrt{\alpha})$ 上界相较此前 Wang et al. (2026) 的 $O(\sqrt{\alpha}\log(1/\alpha))$ 消除了对数因子，且下界实例确认不可改进。
- **数值基准**：本文以理论分析为主，未含数值实验；下界通过解析特征函数递归验证（Appendix G）。

## 相关工作脉络
1. **Pflug (1986)**：提出常数步长随机近似的扩散逼近框架，本文沿用其 OU 基准 $\mathcal{N}(0,\Sigma)$ 但给出了定量 Wasserstein 误差界。
2. **Dieuleveut et al. (2020)、Chen et al. (2022)**：研究常数步长 SGD 的平稳分布与弱极限；本文在这些渐近结果基础上给出 $O(\sqrt{\alpha})$ 的**非渐近**定量上界。
3. **Wang et al. (2026)**：给出 $O(\sqrt{\alpha}\log(1/\alpha))$ 的 $W_1$ 上界及尾部估计；本文的核心改进在于消除对数因子并给出匹配下界。
4. **Benveniste et al. (2012)、Kushner & Yin (2003)、Fort (2015)**：经典 Poisson 方程方法处理 Markovian 随机近似；本文将该方法精细推广至 Gaussian approximation 场景，引入双 Poisson 方程分别处理均值与协方差。
5. **Huo et al. (2024, 2026)、Allmeier & Gast (2024)、Merad & Gaïfas (2025)**：研究 Markovian 噪声下的平稳偏置与前极限耦合；本文聚焦于 Gaussian 近似的速率，揭示了三阶时序矩的作用。
6. **Zhang & Xie (2026a,b,c)、Haque et al. (2026)、Wei et al. (2025)**：近期关于 Gaussian approximation 的若干工作；Appendix A 对比指出本文的证明路线（直接 Poisson-Lindeberg 插值 + Gaussian 初始条件平滑）与 Zhang & Xie 的 CLT 路线、Haque et al. 的下界路线各有侧重，三方共同确立了 $O(\sqrt{\alpha})$ 最优性。

## 局限性与未来方向
1. **假设较为理想化**：要求噪声有界、Markov 链一致遍历（几何收敛）、Hessian Lipschitz；对更一般的亚指数噪声或非一致遍历场景尚未覆盖。
2. **仅处理加法噪声**：递归模型假设为加性 Markovian 噪声 $\alpha\xi(Z_k)$，乘性噪声或多项式收敛链的情形需另作分析。
3. **未讨论非平稳初始化**：分析基于平稳分布；从任意起点出发的收敛速率（transient phase）未在本文展开。
4. **下界仅对特定四状态例成立**：虽证明 $\sqrt{\alpha}$ 阶不可改进，但对一般噪声结构的下界刻画仍是开放问题。
5. **高维依赖未显式量化**：常数 $C$ 依赖维度 $d$，但具体函数形式未展开，对高维应用的 scaling 不够清晰。

## 研究启发与可借鉴点
1. **双 Poisson 方程策略可迁移**：将 Markov 噪声分解为鞅差 + coboundary，再对条件协方差波动施加第二个 Poisson 修正，这一"双重解耦"思路可推广至其他含时序依赖的随机优化收敛分析。
2. **Gaussian 初始条件提供平滑性**：以 $\mathcal{N}(0,\Sigma)$ 作为块起点，利用 Gaussian 卷积使 Lipschitz 测试函数获得有界 Hessian，从而允许二阶 Lindeberg 展开——此技巧可适用于其他鞅近似场景。
3. **协方差 telescope 的技术细节值得复现**：第二个 Poisson 方程将 $\sum_r\mathbb{E}\langle K_r,\mathcal{C}(Z_{r-1})-\Sigma_M\rangle_F$ 化为边界项与变分项，核心在于利用 $K_r$ 对前序信息的可预测性；该手法可推广至非线性 SDE 离散化误差分析。
4. **下界构造的"反例设计"范式**：通过精心设计四状态 Markov 链，使低阶矩（对称性、零自协方差）看起来像 i.i.d.，但高阶混合矩保留依赖信号，从而凸显独立同分布假设的不足——这一构造思想可用于检验其他 Gaussian approximation 结果的紧致性。
5. **与贝叶斯后验近似的关联**：Mandt et al. (2017) 将 SGD 平稳分布解释为后验近似，本文的 $O(\sqrt{\alpha})$ 速率可为后验质量评估提供新基准，值得进一步探索。

## 关键术语表
**Constant-stepsize SGD**：步长 $\alpha$ 保持不变的随机梯度下降，在无无偏噪声时收敛，在有噪声时形成非退化的平稳分布。
**Invariant law / 平稳分布**：递归过程 $(X_k,Z_k)$ 在 $k\to\infty$ 时收敛到的联合概率分布 $\Pi_\alpha$，本文研究的对象。
**Ornstein-Uhlenbeck 近似**：在最优解 $x_\star$ 处线性化漂移后得到的连续时间 OU 过程，其平稳分布为 $\mathcal{N}(0,\Sigma)$，满足 Lyapunov 方程 $H\Sigma+\Sigma H=\Sigma_M$。
**Long-run covariance $\Sigma_M$**：Markovian 噪声的稳态协方差，含所有滞后自协方差之和 $\sum_{\ell\in\mathbb{Z}}\Gamma_\ell$，决定高斯极限的协方差矩阵。
**1-Wasserstein 距离 $W_1$**：Kantorovich-Rubinstein 对偶定义的度量，衡量两个一阶矩有限概率分布的差异，本文的主要误差度量。
**Poisson 方程修正项 $V$ 和 $W$**：$V$ 将 Markov 噪声分解为鞅差与 coboundary；$W$ 将条件协方差波动 telescope 为边界项，两者共同编码时序依赖对高斯近似的修正。
**Lindeberg 插值**：逐步将鞅差序列中的每个增量替换为独立高斯变量，通过二阶 Taylor 展开控制每步替换的误差，总误差累加为 $O(\sqrt{\alpha})$。
**Coalescent coupling**：利用 Markov 链的几何遍历性构造的两个链的相遇耦合，相遇时间具有几何尾部，用于证明联合过程的收缩性。

## 可复现要素
- **数据集**：论文未使用具体数据集，为纯理论分析工作。
- **代码/权重**：论文未提及代码开源。
- **关键超参**：步长 $\alpha$ 需满足 $\alpha\leq\alpha_\star$（依赖 $d,m,L,M,b,C_Z,\rho,\Sigma_M$）；块长参数 $T>0$ 需满足 $e^{-mT}\leq 1/4$。
- **下界实例**：$f(x)=x^2/2$，四状态 Markov 链 $Z_k=(\eta_{k-1},\eta_k)$，$\xi(Z_k)=\eta_k(3-\eta_{k-1})/2$，可自行复现 Appendix G 的解析计算。
