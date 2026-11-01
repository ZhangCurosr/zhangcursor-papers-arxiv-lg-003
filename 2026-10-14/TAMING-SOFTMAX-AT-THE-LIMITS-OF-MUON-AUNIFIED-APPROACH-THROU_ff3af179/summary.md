---
title: "TAMING-SOFTMAX-AT-THE-LIMITS-OF-MUON-AUNIFIED-APPROACH-THROU"
source: https://arxiv.org/pdf/2610.17588v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:06:53"
field: "深度学习优化器理论"
keywords: ["Muon优化器", "Hilbert距离", "softmax模块优化", "RowNorm", "Compositional Muon", "最陡下降", "概率分布几何", "深度学习优化器设计"]
innovations: ["以Hilbert距离统一LM head的RowNorm与attention的Compositional Muon，证明两者均由softmax输出分布的Hilbert距离约束导出", "建立Hilbert距离与交叉熵损失变化、总变差的sharp界，为概率分布稳定性提供可计算的参数级约束"]
---

# 论文速读：TAMING-SOFTMAX-AT-THE-LIMITS-OF-MUON-A-UNIFIED-APPROACH-THROUGH-HILBERT-GEOMETRY

## 一句话总结
本文以Hilbert距离（Hilbert projective metric）为核心工具，统一了LM head的RowNorm优化器与注意力机制的Compositional Muon方法，证明两者均可由"约束softmax输出分布变化"这一共同原则导出。

## 研究问题与动机
- **Muon对softmax的几何失配**：Muon通过控制线性层输出的RMS运动来约束权重更新，但softmax后的概率分布变化无法被preactivation的RMS位移忠实反映——相同RMS扰动可导致完全不同的概率重新分配（图1）。
- **缺乏模块感知的统一优化框架**：现有大规模Muon训练需对LM head、embeddings和attention分别采用AdamW或额外 stabilization 手段（如Kimi K2的MuonClip），缺少基于模块功能的统一几何设计。
- **RMS边界与概率边界的不等价性**：即使对centered扰动，固定RMS界仍允许Hilbert距离随输出维度$O(\sqrt{n})$增长（附录A.1），意味着RMS约束可能掩盖大规模softmax模块的剧烈分布偏移。
- **已有独立方法的理论溯源需求**：RowNorm（Pethick et al., 2025）与Compositional Muon（Keigwin et al., 2026）分别针对LM head和attention设计，缺乏统一的数学基础，难以指导新模块的优化器设计。

## 核心贡献（创新点）
1. **提出以Hilbert距离为约束的softmax模块最陡下降框架**：将模块输出分布的Hilbert距离$\mathcal{H}(p_W(x), p_{W+\Delta W}(x)) \leq \varepsilon$作为更新大小的度量，取代Muon的RMS/谱范数约束。
2. **统一RowNorm的推导**：对LM head，Hilbert距离约束等价于更新矩阵的行直径约束$D(\Delta W) \leq \rho_{\text{head}}$；通过common-ball近似结合Jung定理，严格证明RowNorm方向在相对gap$1-\sqrt{(k+1)/(2k)}$内达到最优（$k=\text{rank}(G)$）。
3. **统一Compositional Muon的推导**：对scaled dot-product attention，Hilbert距离约束等价于QK乘积的一阶谱范数约束$\|\Delta W_Q W_K^\top + W_Q \Delta W_K^\top\|_{\text{op}} \leq \rho_{\text{QK}}$，与Keigwin et al. (2026)的推导完全一致。
4. **建立Hilbert距离与交叉熵损失/总变差的显式界**：证明$\mathcal{H}(p,q)\leq\varepsilon \implies |\Delta\ell_i|\leq\varepsilon$且$\text{TV}(p,q)\leq\tanh(\varepsilon/4)$（后者为sharp bound），为概率分布稳定性提供可计算的保障。

## 方法详解
**Hilbert距离的定义与性质**
对严格正向量$x,y\in\mathbb{R}_{>0}^n$，Hilbert距离为$\mathcal{H}(x,y)=\log\frac{\max_i x_i/y_i}{\min_j x_j/y_j}=\max_{i,j}|\log\frac{q_i/q_j}{p_i/p_j}|$，度量所有pairwise odds ratio的最大对数变化。

**对softmax的闭式表达**
对softmax$ p=\text{softmax}(z) $，有$\mathcal{H}(p,\text{softmax}(z+\Delta z)) = \max_{i,j}|\Delta z_i-\Delta z_j|=\|\Delta z\|_{\text{var}}$，即preactivation扰动的variation seminorm（最大坐标差），自然满足soft-max的平移不变性。

**最陡下降问题的统一形式**
$$\min_{\Delta W}\langle\nabla_W\mathcal{L},\Delta W\rangle_F \quad\text{s.t.}\quad \sup_{x\in\mathcal{X}}\mathcal{H}(p_W(x),p_{W+\Delta W}(x))\leq\varepsilon$$

**LM head：行直径约束→RowNorm**
- 对隐藏状态$\|h\|_2\leq R$，有$\sup\mathcal{H}=R\cdot D(\Delta W)$，其中$D(\Delta W)=\max_{i,j}\|\Delta w_i-\Delta w_j\|_2$为更新行的直径。
- 精确对偶是含$\binom{V}{2}$对、每对带$d$维流变量的最优传输问题，计算不可行。
- **Common-ball近似**：要求所有更新行落在半径$\rho_{\text{head}}/2$的欧氏球内，经平移不变性简化为每行在$B(0,\rho_{\text{head}}/2)$内独立优化。
- **RowNorm解**：$\Delta W_{\text{RN}}=-\frac{\rho_{\text{head}}}{2}\text{RN}(G)$，即逐行归一化梯度。
- **近似保证**：由Jung定理，相对gap$\frac{\Delta_\star-\Delta_{\text{ball}}}{\Delta_\star}\leq 1-\sqrt{\frac{k+1}{2k}}$，对$k=1$精确，对任意$k$至少保留$1/\sqrt{2}$的最优下降。
- **实用版本（Appendix A.5）**：对一阶矩估计$B_t$做mean-row投影$P$、数值稳定归一化（除以$\|\cdot\|_2+\delta_{\text{num}}$）、再投影$P$，学习率$\eta_t\leq\varepsilon/(2R)$保证Hilbert距离界。

**Attention：QK谱约束→Compositional Muon**
- 固定输入下，第$i$行attention分布的Hilbert距离为$\mathcal{H}(p_i,p_i^+)=\frac{1}{\sqrt{d_h}}\max_{j,k\in\mathcal{I}_i}|h_i^\top\Delta M(h_j-h_k)|$。
- 对所有$\|h_\ell\|_2\leq R$取上确界，得$\sup\max_i\mathcal{H}=\frac{2R^2}{\sqrt{d_h}}\|\Delta M\|_{\text{op}}$。
- 令其$\leq\varepsilon$导出$\|\Delta M\|_{\text{op}}\leq\rho_{\text{QK}}$，代入一阶近似$\Delta M\approx\Delta W_Q W_K^\top+W_Q\Delta W_K^\top$得到Compositional Muon的最陡下降问题（Equation 14）。

**前向/反向扰动界（Appendix B）**
- LM head：后向梯度变化$\|\lambda_h^+-\lambda_h\|_2\leq \varepsilon/R+\tau D(W)$，$\tau=\tanh(\varepsilon/4)$。
- Attention：后向梯度变化界为$\frac{R}{\sqrt{d_h}}(\|\Delta M\|_{\text{op}}+4\|M\|_{\text{op}}\tau)\sum_i\text{osc}(r_i)$。

## 实验与结果
- **定位声明**：本文是纯理论工作，**不做实证训练速度的声称**（Section 6 Limitations）。
- **引用已有实验证据**：RowNorm（Lau & Su, 2026）和Compositional Muon（Keigwin et al., 2026）各自报告的预训练提升，为本文框架的实际相关性提供间接支持。
- **数值示例（Appendix C，复现Figure 1）**：$n=32$维，固定初始分布$p$，三种centered扰动均满足$\text{RMS}(\Delta z)\leq 1$：
  - Blue：$\text{RMS}=1,\ \mathcal{H}=4,\ \text{TV}\approx0.514$（分布剧烈重分配）
  - Purple：$\text{RMS}=1,\ \mathcal{H}=2,\ \text{TV}\approx0.004$（分布几乎不变）
  - Red：$\text{RMS}=0.5,\ \mathcal{H}=4,\ \text{TV}$相近Blue（以一半RMS实现同等Hilbert距离）
- 该示例直接验证：**相同RMS下Hilbert距离可相差一倍，且RMS减半仍可能维持大Hilbert距离**，印证RMS与softmax分布变化的几何失配。
- **不等式数值验证（Appendix A.1）**：对centered扰动满足$2\|\Delta z\|_{\text{RMS}}\leq\mathcal{H}(p,q)\leq\sqrt{2n}\|\Delta z\|_{\text{RMS}}$，上界因子$\sqrt{2n}$是sharp的（等号在$\Delta z=(a,-a,0,\ldots)$时达到）。

## 相关工作脉络
1. **Muon (Jordan et al., 2024; Bernstein, 2025a)**：以RMS约束线性层输出变化，导出谱范数约束的最陡下降。本文指出其不适用于softmax模块，并在此基础上扩展几何到概率分布空间。
2. **Modula / Scion (Bernstein, 2025b; Pethick et al., 2025)**：倡导"优化几何应反映模块功能"的Modular duality理念，Scion系统化形式化为$p\to q$算子范数约束的LMO框架。本文延续此精神，专门处理linear+softmax复合模块。
3. **RowNorm (Pethick et al., 2025; Lau & Su, 2026)**：Scion中提出的LM head优化器，经置换与共享平移对称性推导。本文从Hilbert距离出发独立重推，揭示其概率分布稳定性的深层几何根源，并给出Jung定理保证的近似gap上界。
4. **Compositional Muon (Keigwin et al., 2026)**：对attention施加QK乘积一阶变化的谱约束。本文证明该约束等价于约束attention分布的Hilbert距离，为独立设计提供统一解释。
5. **MuonClip (Kimi Team et al., 2026)**：Kimi K2的工程干预——在Muon更新后监测最大attention logits并条件性重缩放Q权重。本文从原理层面解释：此类stabilization的必要性正源于RMS与softmax概率变化之间的几何缺口。
6. **自然梯度 / TRPO (Amari, 1998; Schulman et al., 2015)**：以Fisher几何/KL散度约束策略分布变化。本文选用Hilbert距离作为替代几何，优势在于闭合形式与参数空间的直接映射，且控制pairwise odds而非相对熵。
7. **Hilbert几何与概率单纯形 (Birkhoff, 1957; Nielsen & Sun, 2018, 2023; Cohen & Fausti, 2024)**：Hilbert距离在正锥上的收缩理论与矩阵缩放、聚类中的先验工作。本文首次将其系统引入深度学习优化器设计，并借用Cohen & Fausti (2024)的sharp TV bound建立与损失/概率变化的显式联系。

## 局限性与未来方向
- **无实证验证**：作者明确声明本文不做训练速度或最终模型质量的经验声称，Hilbert距离框架的实际效能有待大规模实验检验（Section 6）。
- **低概率事件的敏感性**：Hilbert距离基于pairwise odds，对极低概率事件的相对变化极度敏感——概率从$10^{-5}$升到$10^{-6}$（相对变化1000倍）可产生$\mathcal{H}\approx6.91$，而总变差仅$\sim10^{-7}$。过紧的$\varepsilon$约束可能过度排斥合理的微小概率质量移动（Section 6）。
- **一阶近似的固有误差**：对attention，$\Delta M$的一阶近似忽略$\Delta W_Q\Delta W_K^\top$二阶项，当更新幅度较大时可能低估实际谱范数变化。
- **输入集合假设**：推导基于$\|h\|_2\leq R$的球形假设；实际训练中hidden state分布可能非各向同性，影响常数因子的 Tightness。
- **未来方向**：（1）探索对低概率事件更鲁棒的距离度量；（2）扩展到其他softmax-like复合模块（如SwiGLU router、MoE门控）；（3）集成实证评估。

## 研究启发与可借鉴点
1. **"输出几何→参数几何"的推导范式**：将模块功能输出空间（如概率单纯形）的度量（Hilbert距离）作为更新约束的源头，是设计模块感知优化器的系统方法，可复用于其他复合模块（如LayerNorm+softmax、normalized attention）。
2. **Jung定理在优化器近似中的运用**：将原始 diameter 约束替换为 common-ball 约束时，用Jung定理定量控制近似gap（$1-\sqrt{(k+1)/(2k)}$），为"可解近似 vs. 最优严格形式"的 trade-off 提供了可量化的设计原则。
3. **平移不变性的显式利用**：softmax对共同shift的不变性直接导致梯度行求和为零（$\mathbf{1}^\top G=0$），使约束对常向量平移无限制，从而可将问题等价地约束在zero-mean子空间——这是RowNorm推导中"减去mean row"操作的理论根基。
4. **sharp总变差界作为稳定性量化**：$\text{TV}(p,q)\leq\tanh(\mathcal{H}(p,q)/4)$给出了从参数约束到输出分布变化的直接定量翻译，可在其他需要概率稳定性的场景中借用。
5. **与前向/后向扰动联合分析**：Appendix B同时给出前向输出变化和后向梯度变化的界，这种双向稳定性分析范式可作为后续工作的模板，确保优化器改进不牺牲反向传播的健康性。

## 关键术语表
**Hilbert距离（Hilbert projective metric）**：定义在正向量/概率单纯形上的度量$\mathcal{H}(p,q)=\max_{i,j}|\log\frac{q_i/q_j}{p_i/p_j}|$，刻画所有pairwise odds ratio的最大对数变化。
**Variation seminorm（变分半范）**：向量$\Delta z$的最大坐标减最小坐标$\max_i\Delta z_i-\min_i\Delta z_i$；对softmax而言等于Hilbert距离。
**Row diameter（行直径）**：矩阵更新行集合$\{\Delta w_1,\ldots,\Delta w_V\}$的两两欧氏距离最大值$D(\Delta W)=\max_{i,j}\|\Delta w_i-\Delta w_j\|_2$。
**Jung定理**：任何直径为$D$、位于$k$维空间的点集必可被半径$\leq D\sqrt{k/(2(k+1))}$的球包含；此处用于界定RowNorm近似的gap。
**Compositional Muon**：Keigwin et al. (2026)提出的attention优化器，约束QK乘积一阶变化的谱范数；本文证明其等价于Hilbert距离约束。
**Total variation (TV) bound**：$\text{TV}(p,q)\leq\tanh(\mathcal{H}(p,q)/4)$，将概率重新分配幅度与Hilbert距离直接关联。
**MuonClip**：Kimi K2提出的工程稳定技术，在Muon更新后监测attention logits最大值并重缩放Q/K权重；本文从原理上解释其必要性。
**Modular duality / Scion**：Bernstein等人提出的模块化优化框架，主张为每个模块定制反映其功能的更新几何。

## 可复现要素
- **数据集**：理论工作，不涉及特定数据集；Figure 1复现使用固定的32维二次profile初始logits（Appendix C给出确定性公式）。
- **代码/权重开源**：论文未提及开源代码或权重；实验部分仅引用已有工作的实证结果（RowNorm、Compositional Muon）。
- **关键超参**：$\varepsilon$（Hilbert距离上界）、$\rho_{\text{head}}=\varepsilon/R$、$\rho_{\text{QK}}=\varepsilon\sqrt{d_h}/(2R^2)$、momentum系数$\beta$、数值稳定项$\delta_{\text{num}}$、学习率$\eta_t\leq\varepsilon/(2R)$；论文未给出具体推荐值。
