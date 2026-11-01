---
title: "Score-Based-Learning-of-Cluster-DAGs-from-Interventions"
source: https://arxiv.org/pdf/2610.11947v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:32:10"
---

# 论文速读：Score-Based-Learning-of-Cluster-DAGs-from-Interventions

## 一句话总结
提出 `coarse`，首个基于分数的干预数据聚类DAG学习方法。在线性高斯假设下，利用干预导致的边缘分布偏移模式（intervention signature）直接确定变量聚类与块级因果顺序，再通过跨环境池化BIC驱动单块局部搜索恢复边结构；在边恢复精度与RePaRe持平的情况下，将搜索速度提升达两个数量级。

## 研究问题与动机
- **细粒度因果发现的可扩展性与可解释性瓶颈**：当变量维度达数百至数千（如像素、神经元、转录本）时，DAG搜索空间呈超指数增长，且细粒度结构难以用于高层推理与干预决策。
- **聚类DAG（Cluster DAG）学习缺乏数据驱动方法**：现有因果抽象工作多为理论刻画（如函数视角的abstraction hierarchy、图形视角的C-DAG与transit cluster），学习聚类与边缘的方法要么预设划分、要么依赖强参数假设。
- **唯一现有方法 RePaRe 依赖脆弱的CI检验**：Madaleno等提出的RePaRe采用两阶段约束法（先聚类后定边），每步依赖条件独立性检验；高阶检验功效低、误差跨阶段累积，且多目标干预下原方法不可识别。
- **分数法尚未迁移至粗粒度异构场景**：GES、GIES、FLOP等成功规避了CI检验，但它们针对单一细粒度DAG的全局/局部搜索无法直接复用，因聚类DAG的目标是“划分+粗化边”而非单DAG，且环境分布异质性未被利用。

## 核心贡献（创新点）
1. **将干预粗化扩展至未知多目标软干预**：证明仅凭边缘偏移签名即可识别 $\mathcal{G}^{\varepsilon}$，而Madaleno的单目标定义在多目标下不可识别（$\Pi^{\mathbb{I}}$ 不一定可辨），本文对象更具一般性。
2. **干预签名同时解耦聚类与因果排序**：签名沿有向边单调递增（Lem. 10），据此定义的支撑偏序 $<_{\text{supp}}$ 忠实编码块级祖先关系，直接将候选父池限定为 $\text{pa}^{\star}(\pi)$，彻底消除顺序枚举搜索。
3. **构建块级弯曲指数族与池化BIC**：证明粗化块模型为 curved exponential family，推导可分解、得分等价的 pooled BIC；证明其局部一致性（单环境的 $O(n)$ 似然增益恒压倒 $O(\log n)$ 惩罚）。
4. **多项式时间一致算法与实证加速**：给出最坏复杂度 $\mathcal{O}(d^2n + |\mathcal{E}|q^3d^3)$（Thm. 20），并在合成与 Light-tunnel 真实数据上验证与 SOTA 边恢复相当、边阶段加速 9×~510×。

## 方法详解
- **第一阶段：聚类与父池构建**
  - 对每个变量 $v$ 与环境 $e\neq 0$，使用 Welch t 检验比较 $X_v^{(e)}$ 与 $X_v^{(0)}$ 的边缘均值，得到 $\hat{M}_{v,e}=\mathbb{1}[p_{v,e}<\alpha]$；也可替换为 KS、能量距离等双样本检验。
  - 按 $\hat{M}$ 的行等价类贪婪分裂初始划分 $\{V\}$，得到 $\hat{\Pi}^{\varepsilon}$；每个块 $\pi$ 的支撑 $\widehat{\text{supp}}(\pi)=\{e:\hat{M}_{v,e}=1, \forall v\in\pi\}$。
  - 定义父候选池 $\widehat{\text{pa}}^{\star}(\pi_j)=\{\pi_k:\widehat{\text{supp}}(\pi_k)\subsetneq\widehat{\text{supp}}(\pi_j)\}$。定理13(ii)保证 $\text{pa}_{\mathcal{G}^{\varepsilon}}(\pi_j)\subseteq\widehat{\text{pa}}^{\star}(\pi_j)\subseteq\text{nd}_{\mathcal{G}^{\varepsilon}}(\pi_j)$，满足受限马尔可夫边界坍缩至真实父集的充分条件。
- **第二阶段：分块分数搜索**
  - **块级线性高斯模型**：$X_j^{(e)} = B_j^{(e)} X_{\text{Pa}_j}^{(e)} + \varepsilon_j^{(e)}$，残差 $\varepsilon_j^{(e)}\sim\mathcal{N}(0,\Sigma_j^{(e)})$ 块间独立。该模型为弯曲指数族，维度 $d_j=r_js_j+r_j(r_j+1)/2$。
  - **Pooled BIC**：$\text{BIC}_j(\pi_j,\text{Pa}_j)=\sum_{e\in\mathcal{E}}[-n_e\log|\widehat{\Sigma}_j^{(e)}|-\lambda\log(n_e)d_j]$，其中 $\widehat{\Sigma}_j^{(e)}$ 为环境 $e$ 下块回归的残差样本协方差。得分沿块完全分解。
  - **GrowShrink 局部搜索**：对每个块 $j$，在 $\widehat{\text{pa}}^{\star}(\pi_j)$ 内运行非贪婪分数版 GrowShrink（Alg. 1）：扩张阶段依次加入使 pooled BIC 提升的候选块，收缩阶段移除使得分上升的冗余块，直至稳定。Cor. 14(ii) 证明输出即为块级真实父集。
- **有限样本变体**：针对 $d$ 大或 $n_e$ 小的场景，提出 CV-coarse（交叉验证选 $\alpha$）、kPC-coarse（观测环境 PCA 降维至 $k_j$ 维）、avg-coarse（块内标准化均值聚合），均仅修改第二阶段的输入维度，不改算法骨架。

## 实验与结果
- **数据集与设置**：合成数据采用 `sempler` 生成 Erdős–Rényi (ER) 与 Scale-Free (SF) 图，$\rho\in\{0.2,0.5,0.8\}$，$d\in[10,1000]$，$m=5$ 个软干预环境（目标未知，每目标加 $\mathcal{N}(2,1)$ 噪声），$n_e\in[10^3,10^5]$；真实数据为 Light-tunnel Causal Chamber（20 个执行器/传感器，1 观测 + 5 干预环境）。评估指标为 ARI（聚类）、Precision/Recall/F1（边），运行时间为单核墙钟时间。
- **一致性与扩展性**：随 $n_e$ 增大，ARI 与 F1 单调上升；Oracle 划分下 $d\le 500$ 且 $n_e\ge 5\times10^3$ 时 F1 接近 1.0；纯端到端时 $d=1000$ F1 降至 0.67，瓶颈主要源于 $d\cdot m$ 次 Welch 检验的多重比较负担。
- **与 RePaRe 对比（Oracle 划分）**：$n_e=10^3$ 时 RePaRe 在高维 dense 图（$d=300,500$）因 CI 检验功效不足领先；$n_e\ge 5\times10^3$ 后 coarse F1 $\ge 0.99$ 全面反超。加速比：$n_e=10^3$ 时 9×~44×，$n_e=10^5$ 时 51×~510×。
- **真实数据（Light-tunnel）**：coarse 与 RePaRe 均取得 ARI=1.00，块级 F1=0.89（8/10 真实粗边，无假阳性），变量级 F1=0.74；GnIES 因数据中心化抵消了均值偏移导致 F1≈0.02，UT-IGSP 仅 0.44。coarse fit 时间 0.02s，RePaRe 0.10s。
- **最强结果**：Oracle 划分下 $d=500, n_e=10^5$ 时 F1=$1.00\pm0.00$，相对 RePaRe 加速 510×。

## 相关工作脉络
- **Causal Abstraction / Cluster DAGs**：Rubenstein
