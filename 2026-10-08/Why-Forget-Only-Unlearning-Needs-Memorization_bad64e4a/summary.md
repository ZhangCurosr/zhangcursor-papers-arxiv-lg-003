---
title: "Why-Forget-Only-Unlearning-Needs-Memorization"
source: https://arxiv.org/pdf/2610.10519v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:10:29"
field: "机器遗忘与信息论"
keywords: ["machine unlearning", "forget-only unlearning", "Rényi divergence", "information-theoretic lower bound", "memorization", "shared-deletion example", "teaching dimension"]
innovations: ["提出共享删除示例概念并证明 forget-only 遗忘的信息论不可能性", "建立基于互信息的遗忘所需记忆量下界 I(M;S)>=ΣH(W_i|历史)-K·err", "将下界实例化于阈值ERM/PCA/岭回归/矩阵完成等多个经典算法"]
benchmarks: ["无（理论论文）"]
---

# 论文速读：Why-Forget-Only-Unlearning-Needs-Memorization

## 一句话总结
本文证明了**纯遗忘（forget-only）机器学习遗忘在理论上是不可行的——并非对所有学习算法都能实现**；进一步推导出：**若要支持未来的任意删除请求，已训练模型必须比常规训练存储更多数据信息**（以互信息 $I(M;S)$ 量化），揭示了"遗忘"与"记忆"之间的基本权衡。

## 研究问题与动机
1. **背景**：随着模型部署普及，满足 GDPR 等隐私法规的"被遗忘权"需求日益增长，重训成本过高，故催生机器遗忘（machine unlearning）研究。
2. **核心问题**：在**forget-only**设置下（删除时仅能访问已训练模型 $M$ 和待删除集 $U$，无法获取保留数据、训练日志等额外信息），是否总存在一个遗忘算法，其输出分布与从保留数据重训的模型难以区分？
3. **现有方法不足**：当前主流经验方法（如基于遗忘集梯度上升、NPO 等）在某些设定下有效，但在其他设定下会失败，缺乏系统性理论解释。
4. **关键缺口**：已有理论工作（如 Cherapanamjeri et al., 2025）基于假设类维度（eluder dimension）下界，不针对 forget-only 设置，且不适用于回归、PCA 等任务；本文首次为严格 forget-only 遗忘建立信息论下界。

## 核心贡献（创新点）
1. **引入"共享删除示例"（shared-deletion example）概念并证明不可能性**：定义了一个学习算法特有的性质——当多个数据集产生相同预训练模型，但删除同一 $U$ 后重训目标彼此远离时，forget-only 遗忘在信息论上不可能达到小的 $\varepsilon$（Theorem 3.1）。
2. **推导出遗忘所需记忆量的互信息下界**（Theorem 4.3）：证明若 $(\mathcal{A}, \bar{\mathcal{A}})$ 满足 $(\alpha,\varepsilon)$-Rényi 遗忘，则 $I(M;S) \gtrsim \sum_i H(W_{\pi(i)} | \cdots) - K \cdot \text{err}$，其中 $W_i$ 是各删除请求对应的重训目标，即模型必须记住足以区分所有可能重训目标的信息。
3. **具体实例化多个经典学习算法的下界**：针对规范阈值 ERM、坐标 PCA、岭回归、行式因子分解矩阵完成等给出显式下界，揭示普通学习只需输出一个简单统计量，但支持遗忘需存储 $\Omega(m \log(N/n))$ 比特。
4. **建立教学维度（teaching dimension）与遗忘不可行性的联系**（Proposition 3.2）：证明对任何零一损失下的 ERM，若假设类教学维度有限且 $|\mathcal{H}| \geq N$，则必然存在共享删除示例，从而 forget-only 遗忘永远不可能无代价实现。

## 方法详解
- **Rényi 遗忘定义**（Definition 1）：$(\mathcal{A}, \bar{\mathcal{A}})$ 满足 $(\alpha, \varepsilon)$-RU 若对任意数据集 $S$ 和 $|U| \leq m$ 的删除请求，有 $d_\alpha(\bar{\mathcal{A}}(\mathcal{A}(S), U), \bar{\mathcal{A}}(\mathcal{A}(S\setminus U), \emptyset)) \leq \varepsilon$，其中 $d_\alpha$ 为对称 $\alpha$-Rényi 散度。
- **空请求效用条件**（Definition 2）：$\bar{\mathcal{A}}(w, \emptyset)$ 须以高概率保留原模型，防止退化恒等输出的 trivial 解。
- **共享删除示例**（Definition 3, s.d.e.）：$N$ 个数据集 $S_i$ 满足：① $U$ 为公共删除集；② $\mathcal{A}(S_i)$ 分布彼此接近（$d_\alpha \leq \delta_\alpha$）；③ 删除后重训目标 $\mathcal{A}(S_i \setminus U)$ 落在相互距离 $\geq \Delta$ 的离散集合内。
- **不可能性定理**（Theorem 3.1）：若存在 s.d.e.，则对精确碰撞（$\delta_\alpha=0$）有 $\varepsilon \geq \sup_r \max\{\log N + \frac{1}{\gamma}\log(1-u(r)), \cdots\}$，其中 $\gamma = (\alpha-1)/\alpha$，$u(r) = \kappa + \Gamma(r)$。
- **互信息下界**（Theorem 4.3）：对离散有限模型空间，任意排列 $\pi$ 下：
$$I(M; S) \geq \sum_{i=1}^K H(W_{\pi(i)} | W_{\pi(<i)}, U_{\pi(\leq i)}) - K\left[h(\tilde{\beta}) + \tilde{\beta}\log(|\mathcal{W}|-1)\right]$$
其中 $\tilde{\beta}$ 为经 Fano 不等式界定的恢复误差上界，第一项为各重训目标逐步揭示的**新信息熵**，第二项为近似遗忘带来的惩罚。
- **关键构造技术**：利用条件 Fano 不等式（对每个请求 $W_i$ 给定历史条件下至多 $q_i$ 个可能值）和 Markov 图推导互信息链式下界（Proposition 4.1 → Theorem 4.3）。

## 实验与结果
> 注：本文为理论论文，不含数值实验，以下为理论下界的实例化结果。

1. **规范阈值 ERM**（Proposition 4.4）：数据集大小 $n$，定义域 $[N]$，$N=nq$，删除预算 $m$（$n \geq 2m$）：普通学习 $I(M;S)_{\text{bare}} = \log(N/n)$ 比特；支持遗忘需 $I(M;S) \geq m[\log(N/n) - \beta_q \log(N/n - 1) - h(\beta_q)]$，当 $q$ 大时与 bare 输出呈 $\Omega(m)$ 倍差距。
2. **坐标 PCA**（Proposition 4.5）：维度 $d=nq$，普通学习仅需 $\log(d/n)$ 比特；支持遗忘需 $\Omega(m \log(d/n))$ 比特。
3. **硬间隔 SVM**（Corollary 1.1 实例化）：$d \geq 2$ 维下，对任意固定有限 $\varepsilon$，不存在具有非平凡效用的 forget-only 遗忘算法满足统一 $(\alpha,\varepsilon)$-RU 保证。
4. **岭回归**（Proposition 4.6）：$d=2b, n=8b$，普通拟合系数恒为零（$I(M;S)_{\text{bare}}=0$）；支持遗忘需 $I(M;S) \geq \frac{n}{8}[\log 3 - h(\beta_3) - \beta_3 \log 2]$ 比特，说明仅靠系数无法支持遗忘。
5. **行式因子分解矩阵完成**（Proposition 4.7）：普通拟合矩阵确定（$I(M;S)_{\text{bare}}=0$）；支持遗忘需存储 $\log(r_\star!)$ 量级的隐藏置换信息。

## 相关工作脉络
1. **Cao & Yang (2015)** 开创性提出机器遗忘概念，以重训为目标定义 indistinguishability，但未讨论 forget-only 限制下的信息瓶颈。
2. **Sekhari et al. (2021, NeurIPS)** 提出带辅助信息的认证遗忘算法，允许访问保留数据或训练状态；本文与之对比，指出 forget-only 设置下这类辅助不可用，信息瓶颈更加严格。
3. **Cherapanamjeri et al. (2025, COLT)** 基于 eluder dimension 给出学习-遗忘算法的空间复杂度下界，但不针对 forget-only，且无法覆盖回归/PCA 等非分类任务；本文与之互补——本文下界**依赖具体学习算法和删除请求**，而非仅假设类。
4. **Empirical gradient-ascent 方法**（Jang et al. 2023; Zhang et al. 2024; Fan et al. 2025; Mavrothalassitis et al. 2025）：在 LLM 遗忘中广泛使用，但实证表明有时失败；本文理论解释了"何时失败及为何失败"——当共享删除示例存在且模型未记忆足够信息时必然失效。
5. **Source-free/retain-free 近似方法**（Ahmed et al. 2025; Chen et al. 2025; Lee et al. 2025）：通过代理数据或生成器重建保留信息；本文理论表明，这些方法的代价在于**在隐式地记忆更多信息**，与本文下界结论一致。
6. **Differential Privacy 下界技术**（Hardt & Talwar 2010）：本文 s.d.e. 的 packing 条件构造借鉴了 DP 下界中的经典思路，但应用于完全不同的遗忘场景。

## 局限性与未来方向
1. **理论边界与实战线之间存在 gap**：下界针对标准学习算法推导，但现代大模型（如 LLM）的 forget-only 遗忘在实践中部分成功，本文指出这通常依赖任务结构（request benign、模型局部稳定等）或弱化遗忘定义，但未量化这些假设的边界。
2. **未指明应存储何种信息**：下界给出了所需信息量的下界，但未指定信息的具体形式（哪些特征/统计量、如何编码存储），对实际算法设计指导性有限。
3. **未处理连续/无限模型空间**：主要下界（Theorem 4.3）针对离散有限模型空间，对神经网络等连续参数空间的直接推广需要额外技术。
4. **未来方向**：（i）将下界推广至现代架构和大模型；（ii）探索"过度参数化模型中已观察到的记忆"是否可被有效利用以支持认证遗忘；（iii）研究在非 forget-only 设置下（允许附加信息）的权衡关系。

## 研究启发与可借鉴点
1. **信息论下界方法可直接迁移**：本文的"共享删除示例 → 互信息下界"框架可用于分析其他受限访问模型操作（如模型编辑、后门去除、数据 poisoning 逆转）的信息论可行性。
2. **条件 Fano 不等式 + 排列分解**：对多个删除请求逐次揭示新信息的建模方式（按排列 $\pi$ 分解条件熵）是一个精巧的技巧，可复用于分析多步交互式遗忘或动态数据删除场景。
3. **与团队方向的结合机会**：若团队关注 LLM 安全/隐私，本文的可迁移价值在于：在部署 forget-only 遗忘方法时，可先检查对应学习算法是否存在 s.d.e.，若存在则应预期该方法在特定分布下会失败，需转向 retain-free 或附加状态方案。
4. **实证评估启示**：论文指出 TOFU/MUSE 等基准在独立 forget/retain 查询假设下可能高估进展（Section A），提示团队在评估遗忘算法时应考虑查询耦合效应。

## 关键术语表
**Forget-only unlearning**：删除时遗忘算法仅接收已训练模型 $M$ 和待删除集 $U$，无法访问保留数据或其他训练状态的最严格遗忘设置。

**Rényi unlearning（RU）**：以对称 $\alpha$-Rényi 散度 $d_\alpha$ 度量遗忘输出与重训目标分布之间的距离，要求 $d_\alpha \leq \varepsilon$ 的遗忘保证。

**Shared-deletion example（s.d.e.）**：$N$ 个数据集经同一学习算法产生相近预训练模型，但删除相同 $U$ 后重训目标分布在相互距离 $\geq \Delta$ 的集合中的构造，是遗忘不可行性的核心障碍。

**Teaching dimension**：假设类中每个假设所需唯一标识的最小标注样本数，有限教学维度意味着存在大量数据集产生相同 ERM 输出，导致 s.d.e. 必然存在。

**Mutual information memorization**：以 $I(M;S)$ 量化训练模型对训练数据的信息保留量，作为衡量"模型为支持遗忘必须记忆多少"的理论指标。

**Recovery error（$\beta_{\text{un}}$）**：遗忘输出 $\overline{W}$ 与重训目标 $W$ 不同值的最大概率上界，由 $\varepsilon$ 和空请求效用噪声共同决定。

## 可复现要素
- **数据集**：本文无数值实验，所有"数据集"均为理论构造（如 canonical threshold、hard-margin SVM 构造、one-hot ridge regression 构造等），未使用公开数据集。
- **代码/权重**：论文未提及开源代码或权重。
- **关键超参**：$\alpha > 1$（Rényi 阶数）、$\varepsilon \geq 0$（遗忘精度）、$m$（删除预算）、$\Delta$（重训目标分离距离）、$\delta_\alpha$（预训练模型碰撞容忍度）、$\kappa$（重训目标分布集中度）。
