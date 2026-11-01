---
title: "TAMING-SOFTMAX-AT-THE-LIMITS-OF-MUON-AUNIFIED-APPROACH-THROU"
source: https://arxiv.org/pdf/2610.17588v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:06:40"
field: "深度学习优化器设计"
keywords: ["Hilbert距离", "Muon优化器", "softmax模块", "RowNorm", "Compositional Muon", "概率分布稳定性", "优化几何"]
innovations: ["提出Hilbert距离作为softmax模块的统一更新约束度量", "从概率分布稳定性推导RowNorm和Compositional Muon的数学基础", "建立Hilbert距离与CE损失界、TV界的严格关系"]
---

# 论文速读：TAMING-SOFTMAX-AT-THE-LIMITS-OF-MUON-AUNIFIED-APPROACH-THROUGH-HILBERT-GEOMETRY

## 一句话总结
本文提出使用Hilbert距离（Hilbert projective metric）作为softmax模块的更新约束度量，统一推导了RowNorm（LM head优化）和Compositional Muon（attention优化）的数学基础，揭示了二者共享的概率分布稳定原则。

## 研究问题与动机
- Muon优化器通过控制线性层输出的RMS移动来约束权重更新，但其几何度量适用于纯线性层，未能考虑后续非线性激活（如softmax）对输出分布的影响
- RMS位移与softmax输出分布变化存在几何不匹配：相同RMS约束下，不同更新方向可能导致截然不同的概率重分配（图1示例）
- 现有大型Muon训练食谱中，LM head、embeddings等通常仍使用AdamW，缺乏针对softmax模块的统一优化几何设计
- Kimi K2引入的MuonClip通过事后重缩放缓解注意力分数过度增长，但未从概率分布稳定性角度提供系统性约束

## 核心贡献（创新点）
- **统一Hilbert距离框架**：提出用Hilbert距离直接约束softmax输出分布变化，该度量同时控制成对几率比、总变差概率重分配和交叉熵损失变化
- **LM head的RowNorm推导**：将Hilbert距离约束转化为行直径约束，通过common-ball近似导出RowNorm闭式解，并给出与最优解的近似误差上界（Jung定理）
- **Attention的Compositional Muon推导**：将同一原则应用于scaled dot-product attention，导出QK矩阵一阶变化的谱范数约束， recover Compositional Muon的最陡下降问题
- **理论完备性**：建立了Hilbert距离与softmax不变性（平移不变）、TV界、CE损失界的严格数学关系，并证明总变差界的尖锐性

## 方法详解
**Hilbert距离定义**：对严格正向量 $x, y \in \mathbb{R}_{>0}^n$，定义为 $\mathcal{H}(x,y) = \log \frac{\max_i x_i/y_i}{\min_j x_j/y_j}$

**关键性质**：
1. **对softmax闭式表达**：$\mathcal{H}(\text{softmax}(z), \text{softmax}(z+\Delta z)) = \max_{i,j}|\Delta z_i - \Delta z_j| = \|\Delta z\|_{\text{var}}$（variation seminorm）
2. **控制交叉熵损失变化**：$|\Delta \ell_i| \leq \mathcal{H}(p,q) \leq \varepsilon$
3. **控制概率重分配**：$\text{TV}(p,q) \leq \tanh(\mathcal{H}(p,q)/4)$，且该界是尖锐的

**最陡下降问题形式化**：
$$\min_{\Delta W} \langle G, \Delta W \rangle_F \quad \text{s.t.} \quad \sup_{x \in \mathcal{X}} \mathcal{H}(p_W(x), p_{W+\Delta W}(x)) \leq \varepsilon$$

**LM head应用**：
- Hilbert距离约束等价于行直径约束：$D(\Delta W) = \max_{i,j}\|\Delta w_i - \Delta w_j\|_2 \leq \rho_{\text{head}}$
- 精确对偶是向量最优运输问题，变量数随词表大小平方增长
- Common-ball近似：要求所有更新行落在半径 $\rho_{\text{head}}/2$ 的同一球内，导出RowNorm更新：$\Delta W_{\text{RN}} = -\frac{\rho_{\text{head}}}{2}\text{RN}(G)$
- 近似误差界：$0 \leq \frac{\Delta_\star - \Delta_{\text{ball}}}{\Delta_\star} \leq 1 - \sqrt{\frac{k+1}{2k}}$，其中 $k = \text{rank}(G)$

**Attention应用**：
- 固定隐藏状态时，Hilbert距离约束导出：$\|\Delta M\|_{\text{op}} \leq \rho_{\text{QK}}$，其中 $M = W_Q W_K^\top$
- 一阶近似：$\|\Delta W_Q W_K^\top + W_Q \Delta W_K^\top\|_{\text{op}} \leq \rho_{\text{QK}}$
- 该约束与Compositional Muon的一阶QK最陡下降问题一致

## 实验与结果
本文主要为理论分析工作，**未提供新的实验结果**。作者明确声明："We leave that question to future experimental work and make no empirical claim of improved training speed or final model quality."

文中引用的已有实验证据：
- Lau & Su (2026) 的RowNorm在语言模型预训练中报告了改进
- Keigwin et al. (2026) 的Compositional Muon在预训练中报告了改进
- Kimi K2的MuonClip作为工程实践被提及

图1展示了RMS与Hilbert距离的不匹配示例（32维，中心化扰动）：
- Blue: RMS=1, H=4, TV≈0.514
- Purple: RMS=1, H=2, TV≈0.004
- Red: RMS=0.5, H=4, TV不可见但分布显著改变

## 相关工作脉络
- **Muon (Jordan et al., 2024)**：控制线性层输出RMS移动，导出谱范数约束；本文指出其对softmax模块的几何不匹配
- **Modula项目 (Bernstein, 2025b)**：倡导按模块功能设计更新几何；本文扩展至softmax复合模块
- **Scion (Pethick et al., 2025)**：系统框架，基于$p \to q$算子范数约束；RowNorm是其特例
- **RowNorm (Lau & Su, 2026; Pethick et al., 2025)**：从置换和平移对称性推导；本文从Hilbert距离独立推导
- **Compositional Muon (Keigwin et al., 2026)**：约束QK乘积一阶变化的谱范数；本文证明其源于概率分布稳定性
- **自然梯度 (Amari, 1998)**：用Fisher度量；本文用Hilbert度量，关注输出分布而非参数分布
- **TRPO (Schulman et al., 2015)**：约束KL散度；本文约束Hilbert距离，保证更严格的CE损失界

## 局限性与未来方向
- **主要局限**：纯理论框架，缺乏实证验证训练加速或模型质量提升
- **Hilbert距离的相对性**：对低概率事件的大相对变化敏感（如从$10^{-5}$到$10^{-6}$），即使概率质量移动很小
- **未来方向1**：探索保留有用保证但对低概率事件不敏感的替代度量
- **未来方向2**：将统一框架推广至其他复合模块（如SwiGLU MLP、MoE路由器）
- **计算复杂度**：精确行直径问题的对偶随词表大小平方增长，需近似处理

## 研究启发与可借鉴点
- **模块感知的优化几何设计**：不同架构模块（线性、softmax、attention）需要不同的更新度量，而非一刀切的全局优化器
- **概率分布稳定性作为训练目标**：直接约束输出分布变化（而非中间激活）可能提供更稳定的训练动力学
- **Hilbert距离的工程应用**：其闭式表达（variation seminorm）和与softmax平移不变性的契合，值得在其他概率输出模块中探索
- **近似与精度的权衡分析**：Jung定理给出的RowNorm近似误差界提供了理论保证，类似分析可用于评估其他优化器的近似质量
- **统一视角的价值**：将独立发展的方法（RowNorm、Compositional Muon）统一到Hilbert距离框架，有助于发现新的优化设计机会

## 关键术语表
**Hilbert距离**：投影几何中的度量，对正向量定义为对数几率比的范围，适合衡量概率分布变化
**Variation seminorm**：向量最大值与最小值之差， softmax的Hilbert距离等价于预激活扰动的该范数
**RowNorm**：对LM head梯度按行归一化的优化更新，本文从Hilbert距离推导
**Compositional Muon**：约束QK矩阵一阶变化谱范数的attention优化器，本文证明其与Hilbert距离框架一致
**总变差（TV）**：概率分布间重分配程度的度量，受Hilbert距离界控制
**最陡下降（Steepest descent）**：在给定度量下寻找损失最大一阶下降方向的优化问题
**Modula项目**：倡导按模块功能设计更新几何的优化器设计范式
**RMS单位球**：Muon使用的输入有界集，对应隐藏状态欧氏范数不超过$\sqrt{d}$

## 可复现要素
- **数据集**：论文未进行新实验，无数据集声明
- **代码/权重**：论文未开源代码，仅理论分析
- **关键超参**：$\rho_{\text{head}}$（行直径约束）、$\rho_{\text{QK}}$（QK谱约束）、$\varepsilon$（Hilbert距离上限）、动量系数$\beta$、数值稳定项$\delta_{\text{num}}$；论文未提供具体数值
- **复现说明**：Appendix A.5给出了RowNorm的实际更新公式（含动量和数值稳定），Appendix C给出了图1的复现参数
