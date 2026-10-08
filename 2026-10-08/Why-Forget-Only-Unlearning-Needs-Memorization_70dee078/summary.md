---
title: "Why-Forget-Only-Unlearning-Needs-Memorization"
source: https://arxiv.org/pdf/2610.10519v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:15:38"
field: "机器学习理论/机器遗忘"
keywords: ["machine unlearning", "forget-only unlearning", "Rényi divergence", "memorization", "information-theoretic lower bound", "mutual information", "teaching dimension"]
innovations: ["提出共享删除示例(s.d.e.)并证明forget-only unlearning并非对所有学习算法可行", "推导模型为支持unlearning需保留的训练数据信息量下界I(M;S)", "将下界实例化至阈值/PCA/岭回归/SVM/矩阵补全等多种学习算法"]
benchmarks: ["理论下界（无数值benchmark）"]
---

# 论文速读：Why Forget-Only Unlearning Needs Memorization

## 一句话总结
本文从理论上证明：**forget-only unlearning（仅凭训练模型和被遗忘样本来执行删除）并非对所有学习算法都可行**，并推导出模型为支持未来删除请求必须记住训练数据的**信息量下界**，揭示标准训练丢弃的信息可能在删除时被重新需要。

## 研究问题与动机
- **核心问题**：在forget-only设置下（删除时仅能访问训练模型 $M$ 和遗忘集 $U$，无保留数据或其他训练信息），机器遗忘是否总是可行的？
- **现有不足**：实证方法（如基于遗忘集梯度上升的方法）在某些场景成功但在其他场景失败，缺乏理论解释；现有理论工作（如Cherapanamjeri et al., 2025）主要关注假设类复杂度（eluder dimension），而非具体学习算法与信息保留之间的关系。
- **理论空白**：为何某些 forget-only unlearning 方法会失败？成功时需要模型记住多少训练数据信息？

## 核心贡献（创新点）
1. **不可行性下界**：提出"共享删除示例"(shared-deletion example)概念，证明当多个数据集产生相同预训练模型但删除后重训练目标分离时，forget-only unlearning 无法满足小的 Rényi 发散参数 $\varepsilon$（定理 3.1）。
2. **记忆必要性定理**：首次系统推导支持 forget-only unlearning 所需的最小记忆量下界，即模型与训练数据之间的互信息 $I(M;S)$ 必须满足的具体不等式（定理 4.3）。
3. **算法实例化**：将理论下界具体应用于多种标准学习算法——阈值学习器、经验中位数、硬间隔SVM、坐标PCA、岭回归、行分解仿射矩阵补全——揭示标准训练保留的信息量远低于 unlearning 所需。
4. **教学维度关联**：证明对于有限教学维度(teaching dimension)的假设类，任何经验风险极小化(ERM)学习器都存在共享删除示例，从而建立不可行性的普遍性（命题 3.2）。

## 方法详解
- **Rényi unlearning 定义**：$(\mathcal{A}, \bar{\mathcal{A}})$ 满足 $(\alpha, \varepsilon)$-Rényi unlearning (RU) 若对任意数据集 $S$ 和删除请求 $U$，有 $d_\alpha(\bar{\mathcal{A}}(\mathcal{A}(S), U), \bar{\mathcal{A}}(\mathcal{A}(S\setminus U), \emptyset)) \leq \varepsilon$，其中 $d_\alpha$ 为对称 $\alpha$-Rényi 发散。
- **空请求效用**：要求 $\bar{\mathcal{A}}(w, \emptyset)$ 以高概率接近输入 $w$（定义 2），防止trivial解（始终输出同一分布）。
- **共享删除示例**：$N$ 个数据集 $S_1,\dots,S_N$ 共享遗忘集 $U$，预训练模型分布相近（$d_\alpha \leq \delta_\alpha$），但删除后重训练目标分布在距离至少 $\Delta$ 的不相交集合上（定义 3）。
- **记忆下界核心公式**（定理 4.3）：
$$I(M;S) \geq \sum_{i=1}^{K} H(W_{\pi(i)} | W_{\pi(<i)}, U_{\pi(\leq i)}) - K \cdot \text{err}(\alpha, \varepsilon)$$
其中第一项为各重训练目标的**新信息熵**（考虑已处理请求的条件依赖），第二项为近似 unlearning 误差惩罚。对任意排列 $\pi$ 成立，可选最优排序得最强下界。
- **推导技术**：结合数据处理不等式、Rényi 发散的事件传递引理（引理 B.1）和 Fano 不等式，将互信息下界与 unlearning 恢复误差 $\beta_{un}$ 关联。

## 实验与结果
本文为纯理论工作，无数值实验，但给出多个理论实例化的显式下界：

| 学习算法 | 标准训练保留信息 $I(M;S)_{\text{bare}}$ | Unlearning 所需下界 | 关键结论 |
|---------|----------------------------------------|---------------------|---------|
| 规范阈值 ERM（$N=nq$ 点） | $\log(N/n)$ | $\geq m[\log(N/n) - \beta_q \log(N/n - 1) - h(\beta_q)]$ | 需保留约 $m$ 倍信息 |
| 坐标 PCA（维度 $d=nq$） | $\log(d/n)$ | $\geq m[\log(d/n) - \beta_q \log(d/n - 1) - h(\beta_q)]$ | 同理 |
| 岭回归（$d=2b, n=8b$） | $0$（系数恒为零） | $\geq \frac{n}{8}[\log 3 - h(\beta_3) - \beta_3 \log 2]$ | 标准输出**完全不够** |
| 行分解仿射矩阵补全 | $0$（拟合矩阵确定性） | $\geq \log(r_\star!) - \sum [h(\beta_s) + \beta_s \log(s-1)]$ | 需记住隐式置换 |
| 硬间隔 SVM（$d \geq 2$） | — | 对任意有限 $\varepsilon$，**不存在**满足非平凡效用的 forget-only unlearner | Corollary 1.1 |

最强结果：**Corollary 1.1** 表明在 $d \geq 2$ 维空间中，硬间隔 SVM 的 forget-only unlearning 对任何固定有限 $\varepsilon$ 均不可能（在 Gaussian 空请求后处理下）。阈值学习器的下界显示：对于规模 $n$、定义域 $N$ 的数据，标准训练仅保留 $\log(N/n)$ 比特，但支持大小为 $m$ 的删除请求需 $\Omega(m \log(N/n))$ 比特。

## 相关工作脉络
1. **Cao & Yang (2015)**：机器遗忘的奠基性工作，定义重训练等价性；本文在其框架内研究更严格的 forget-only 设置。
2. **Neel et al. (2021), Sekhari et al. (2021)**：允许访问保留数据或二阶信息的 certified unlearning；本文对比指出这些方法通过额外信息绕过了本文的障碍。
3. **Cherapanamjeri et al. (2025)**：基于 eluder dimension 下界学习-unlearning 算法空间复杂度；本文与之互补——本文下界依赖具体学习算法和删除请求，适用于严格 forget-only 设置，并涵盖回归、PCA 等任务。
4. **Feldman et al. (2025), Brown et al. (2021)**：用互信息 $I(M;S)$ 量化记忆；本文沿用此度量并推导 unlearning 场景下的下界。
5. **Empirical forget-only 方法**（Jang et al. 2023, Zhang et al. 2024, Fan et al. 2025 等）：LLM 遗忘中的梯度上升/NPO 方法；本文理论解释了这些方法为何有时失效——并非实现缺陷，而是信息论障碍。
6. **Source-free/retain-free unlearning**（Chen et al. 2025, Lee et al. 2025 等）：通过代理数据/生成器重建保留信息；本文指出这等价于在模型外存储了本文下界所要求的记忆量。

## 局限性与未来方向
- **理论下限的形式**：下界给出"需要多少信息"，但未指出**以何种形式**存储这些信息（哪些特征/统计量），也未给出构造性算法。
- **大规模架构的外推**：结论在标准学习算法上实例化，但对当代过参数化模型（如 LLM）和大规模训练管道，区分算法性失败与信息论性失败仍有待研究。
- **记忆利用问题**：过参数化模型中已观察到的记忆能否被有效访问以支持 certified forget-only unlearning，仍是开放问题。
- **非均匀删除请求**：下界对每个数据集和每个允许删除请求一致成立，但实际中删除请求可能集中在特定子集，可能降低要求。

## 研究启发与可借鉴点
1. **共享删除示例构造技巧**：通过构造"预训练不可区分但删除后可区分"的数据集对，证明不可行性——该方法可迁移到其他隐私/记忆相关的下界证明。
2. **互信息-熵下界框架**：定理 4.3 的排列分解技术（利用条件熵计只有信息、用 Fano 界处理近似误差）可作为分析记忆需求的通用工具。
3. **教学维度与 ERM 联系**（命题 3.2）：有限教学维度类必然存在共享删除示例，这一洞见可用于快速判断某假设类是否 inherently 不支持 forget-only unlearning。
4. **实验设计启发**：对阈值/PCA/岭回归的实例化展示了如何用简单的概率分布构造（均匀分桶、随机置换）得到紧的下界，值得在类似理论分析中借鉴。
5. **与团队方向结合机会**：若团队研究 LLM 遗忘或数据删除，本文结论提示：实现 certified forget-only unlearning 需在训练中有意保留额外统计量（如删除请求对应的候选重训练目标的编码），而非仅保存最终模型。

## 关键术语表
- **Forget-only unlearning**：删除时仅能访问训练模型 $M$ 和遗忘集 $U$，无保留数据或训练辅助信息的机器遗忘设置。
- **Rényi unlearning (RU)**：用对称 $\alpha$-Rényi 发散 $d_\alpha$ 度量 unlearning 输出与重训练目标分布的接近程度，要求 $d_\alpha \leq \varepsilon$。
- **Shared-deletion example (s.d.e.)**：$N$ 个数据集共享同一遗忘集 $U$，预训练模型分布相近但删除后重训练目标分布在距离 $\Delta$ 以上不相交集合上的实例，用于构造不可行性下界。
- **Teaching dimension**：假设类 $\mathcal{H}$ 中任一假设被唯一确定的最小标注样本数；有限教学维度的 ERM 学习器必然存在 s.d.e.。
- **Mutual information $I(M;S)$**：模型 $M$ 与训练数据 $S$ 之间的互信息，本文用于量化模型对训练数据的记忆量。
- **Recovery error $\beta_{un}$**：unlearning 输出 $\overline{W}$ 与重训练目标 $W$ 不同的上界概率，由 $\varepsilon$、空请求噪声 $\eta$ 和学习器随机性共同控制。
- **Coordinate PCA**：返回最大化样本二阶矩的坐标方向（即 $e_j$ 使得 $\sum_{x \in S} x_j^2$ 最大）的简化 PCA 变体。
- **Row-wise factorized affine matrix completion**：用行因子化 $x_i = u_i v_i$ 最小化仿射约束损失和因子范数和的矩阵补全方法。

## 可复现要素
- **数据集**：理论论文，无实证数据集；实例化使用人工构造的分布 $\mathcal{P}_S$（均匀分桶、随机置换等）。
- **代码/权重**：论文未提供代码。
- **关键超参**：$\alpha > 1$（Rényi 参数）、$\varepsilon \geq 0$（unlearning 精度）、$\delta_\alpha, \kappa$（s.d.e. 参数）、$\lambda$（岭回归正则化）、$\sigma$（Gaussian 噪声方差）。
