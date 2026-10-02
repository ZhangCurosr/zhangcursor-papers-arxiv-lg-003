---
title: "VISUAL-BRANCH-IS-WHAT-YOU-NEED-FOR-CLIP-BASED-CLASS-INCREMEN"
source: https://arxiv.org/pdf/2609.37888v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:35:02"
field: "类增量学习"
keywords: ["类增量学习", "CLIP", "视觉-语言预训练", "闭式分类器", "无回放学习", "LS-SVM", "模态间隙"]
innovations: ["提出完全去除CLIP文本分支的纯视觉增量分类方法Vis", "通过多层CLIP特征残差融合增强视觉表示并在基座会话固化", "设计基于可加充分统计量的闭式增量核化LS-SVM分类器"]
benchmarks: ["CIFAR100", "FGVCAircraft", "StanfordCars", "ImageNet-R", "CUB200", "UCF101", "Food101", "ObjectNet", "SUN397"]
---

# 论文速读：VISUAL BRANCH IS WHAT YOU NEED FOR CLIP-BASED CLASS-INCREMENTAL LEARNING

## 一句话总结
本文重新审视了 CLIP 类增量学习（CIL）中广泛使用的文本分支，发现文本嵌入与视觉类别分布存在模态偏差，不利于分类器优化；据此提出 **Vis（Visual Incremental SVM）**，一种完全去掉文本分支、仅在视觉空间中构建增量分类器的方法，结合多层视觉特征残差融合与闭式核化增量 LS-SVM，在九项基准上均达到 SOTA。

## 研究问题与动机
1. **核心问题**：CLIP-based CIL 普遍将 CLIP 文本编码器生成的类别模板嵌入作为分类器权重（文本头），这一设计是否真正有效？
2. **现有方法不足**：CLIP 的共享嵌入空间仍存在图像-文本模态间隙（modality gap），文本分类器权重与视觉类别分布几何错位；在相同任务增量训练下，文本初始化分类器比视觉原型初始化产生更高的训练损失和更低的准确率。
3. **关键观察**：类别级模态间隙 $g_c = 1 - \mu_c^\top t_c$ 与文本头性能退化 $\Delta_c$ 呈正相关，表明视觉原型作为分类器权重更可靠。
4. **研究问题**：既然 CLIP 已提供强大的视觉编码器，增量分类器是否仍有必要依赖部署阶段的文本分支？

## 核心贡献（创新点）
1. **提出纯视觉增量分类框架 Vis**：彻底移除 CLIP 文本分支，在纯视觉空间构建增量分类器，与主流 CLIP-based CIL 方法形成本质差异。
2. **多层视觉特征残差融合**：仅利用基座会话数据，将 CLIP 多层的 CLS token 特征经 MLP 残差融合到最终层特征，使视觉表示更具任务适应性，而 CLIP 主干保持冻结。
3. **闭式增量核化 LS-SVM 分类器**：使用显式固定非线性特征映射（随机投影+ReLU）构造有限维 kernel 空间，LS-SVM 权重由可加的充分统计量 $(G_t, Q_t, \mathbf{s}_t)$ 以闭式求解，新增类别时通过累加统计量重算所有已见类别权重，无需存储旧样本即实现无回放增量学习。
4. **系统性诊断 CLIP 文本嵌入的缺陷**：从几何（t-SNE 可视化、模态间隙度量）和优化（训练损失曲线、梯度余弦分布）两个角度，首次系统揭示文本分类器权重在 CIL 场景下的固有劣势。

## 方法详解
**整体架构**：Vis 由两个模块组成——任务自适应视觉表示（第 4.1 节）和增量核化 LS-SVM 分类器（第 4.2 节），所有更新均在视觉特征空间完成。

**4.1 任务自适应视觉表示**
- 冻结 CLIP 视觉编码器（ViT-B/16，$L$ 个 Transformer block），提取每层的 CLS token：$\mathbf{h}_\ell(\boldsymbol{x}) = \mathbf{h}_{\text{cls}}^\ell(\boldsymbol{x}) \in \mathbb{R}^{d_v}$。
- 将选定的多层 CLS 特征拼接后送入轻量级两阶段 MLP（Residual Mixer $\mathcal{M}$）生成残差修正：
$$\mathcal{M}(\mathbf{m}(\boldsymbol{x})) = U \cdot \text{GELU}(V\mathbf{m}(\boldsymbol{x}) + \mathbf{b}_V) + \mathbf{b}_U$$
- 增强后的视觉表示为：$\mathbf{u}(\boldsymbol{x}) = \mathbf{h}_L(\boldsymbol{x}) + \mathcal{M}(\mathbf{m}(\boldsymbol{x}))$，最终线性层零初始化保证起始等价于原始 CLIP 输出。
- 仅在基座会话用辅助线性分类器训练 $\mathcal{M}$，损失函数为：
$$\mathcal{L}_{\text{adapt}} = \frac{1}{|\mathcal{D}_1|}\sum_{(\boldsymbol{x},\boldsymbol{y})\in\mathcal{D}_1}\left[\ell_{\text{ce}}(g_\omega(\mathbf{u}(\boldsymbol{x})),\boldsymbol{y}) + \lambda_{\text{adapt}}\|\mathbf{u}(\boldsymbol{x}) - \mathbf{h}_L(\boldsymbol{x})\|_2^2\right]$$
- 同时通过基座验证集联合搜索最优核特征维度 $D$、正则系数 $\lambda$ 和视觉层子集 $B$，全部确定后固化。

**4.2 增量核化 LS-SVM 分类器**
- 显式非线性特征映射（固定随机投影矩阵 $R$，ReLU 激活）：$\phi(\boldsymbol{x}) = \sigma(R^\top \mathbf{u}(\boldsymbol{x})) \in \mathbb{R}^D$，加上偏置项：$\tilde{\phi}(\boldsymbol{x}) = [\phi(\boldsymbol{x}); 1] \in \mathbb{R}^{D+1}$。
- 多分类 LS-SVM 目标（one-vs-all $\pm 1$ 编码）：
$$\tilde{W}^\star = \left(\Gamma + C_{\text{svm}} \tilde{\Phi}^\top \tilde{\Phi}\right)^{-1}\left(C_{\text{svm}} \tilde{\Phi}^\top Y\right)$$
- 维护可加的充分统计量：
$$G_t = \sum_{s=1}^t \tilde{\Phi}_s^\top \tilde{\Phi}_s,\quad Q_t = \sum_{s=1}^t \tilde{\Phi}_s^\top Y_s^{(t)},\quad \mathbf{s}_t = \sum_{s=1}^t \tilde{\Phi}_s^\top \mathbf{1}$$
- 当新会话引入 $m_t$ 个新类别时，将 $Q_{t-1}$ 扩展（追加历史样本对新类别的 $-1$ 贡献）：
$$Q_{t-1}^{\uparrow} = [Q_{t-1},\; -\mathbf{s}_{t-1}\mathbf{1}_{m_t}^\top]$$
然后累加更新 $G_t, Q_t, \mathbf{s}_t$，并以闭式重算全类别分类器权重 $\tilde{W}_t^\star$，无需访问历史数据。推理时仅用视觉分支：$\hat{y} = \arg\max_c \tilde{W}_t^\star[:,c]^\top \tilde{\phi}(\boldsymbol{x})$。

## 实验与结果
**数据集**：九个 CLIP-based CIL 基准——FGVCAircraft、CIFAR100、StanfordCars、Food101、UCF101、CUB200、ObjectNet、ImageNet-R、SUN397，采用标准 B-m Inc-n 协议（含零基座 B0 和半基座 B50/B100）。

**主要结果（Table 1，所有方法使用相同 CLIP ViT-B/16 预训练权重）**：

| 数据集 / 协议 | Vis $\bar{A}$ | 最优基线 $\bar{A}$ | 提升 |
|---|---|---|---|
| Aircraft B0 Inc10 | **75.21** | BOFA 70.96 | +4.25 |
| Aircraft B50 Inc10 | **71.77** | BOFA 66.09 | +5.68 |
| CIFAR100 B0 Inc10 | **90.64** | ACIL 89.41 | +1.23 |
| CIFAR100 B50 Inc10 | **87.17** | ACIL 86.56 | +0.61 |
| Cars B0 Inc10 | **94.47** | BOFA 94.21 | +0.26 |
| Cars B50 Inc10 | **92.55** | BOFA 92.13 | +0.42 |
| ImageNet-R B0 Inc20 | **86.05** | BOFA 84.53 | +1.52 |
| ImageNet-R B100 Inc20 | **82.61** | BOFA 81.60 | +1.01 |
| UCF B0 Inc10 | **99.23** | RanPAC 98.28 | +0.95 |
| UCF B50 Inc10 | **98.75** | RanPAC 98.05 | +0.70 |

Vis 在全部九个数据集的所有协议设置下均达到最高平均准确率（$\bar{A}$）和最终阶段准确率（$A_B$）。消融实验验证了各组件贡献：ZS-CLIP → Visual LS-SVM → +Kernel Map → +Residual Fusion 逐步提升；与 ACIL、RanPAC 等闭式方法相比优势显著；加入文本原型的控制实验（Vis+Text）未带来稳定增益。遗忘度量 $F_B$ 在 8 组设置中的 5 组最低，表明充分统计量机制有效防止灾难性遗忘。

## 相关工作脉络
1. **CLIP-based CIL 方法**（SimpleCIL, CLG-CBM, PROOF, BOFA, RAPF）：这些方法仍依赖或融合 CLIP 文本分支进行增量分类器构建或跨模态引导；Vis 的本质区别在于完全剔除部署阶段的文本分支，论证了文本分支并非必需。
2. **Prompt-based CIL**（DualPrompt, CODA-Prompt）：通过在冻结 backbone 上学习/选择提示令牌实现任务适配；Vis 不引入可学习提示，仅训练基座会话的一个轻量残差融合模块，参数效率更高。
3. **解析类 CIL 方法**（ACIL, RanPAC）：同样采用闭式/递归更新、无需回放的增量分类器；Vis 的关键差异在于：(a) 通过残差融合先增强视觉表示，(b) 采用 one-vs-all $\pm1$ LS-SVM 而非 one-hot Ridge，配合充分统计量 $\mathbf{s}_t$ 实现类别扩展，并在 CLIP 基础上验证。
4. **CLIP 模态间隙研究**（Liang et al., 2022）：揭示了 CLIP 共享空间中图像-文本嵌入的几何分离现象；本文将其量化为类别级模态间隙 $g_c$，并首次证明其对 CIL 文本头性能的负面预测力。
5. **表示不变性/多阶段 ViT 特征研究**（Raghu et al., 2021; Ghiasi et al., 2022）：证实 ViT/CLIP 中间层蕴含有用信息；本文将这一观察操作化，设计残差融合策略在 CIL 场景下充分利用多层 CLS 特征。

## 局限性与未来方向
1. **视觉表示冻结**：基座会话之后残差融合模块和特征映射均固定，无法在后续增量任务中进一步适配表征；限制了模型对分布漂移的适应能力。
2. **固定随机投影的隐含假设**：显式特征映射 $\phi(\boldsymbol{x})$ 依赖一次性随机采样 $R$，虽然实验证明结果对 $R$ 的初始化种子不敏感，但在极端任务序列下可能存在表达能力上限。
3. **未来方向**（论文自述）：探索在保持闭式统计量分类器高效性的前提下，允许视觉表示在增量阶段自适应更新。

## 研究启发与可借鉴点
1. **文本分支非必需的设计思路**：本文通过系统的几何+优化诊断，推翻了一个看似自然的 CLIP CIL 设计惯念，这种"反直觉"分析范式值得借鉴——对任何预训练模型的多模态下游应用，都应重新审视跨模态对齐的实际效用。
2. **多层特征残差融合策略**：仅用基座数据训练一个轻量 MLP 来增强最终层表示，避免在增量阶段改变 backbone，兼顾表示适应性与稳定性，可迁移至其他冻结骨干的增量学习场景。
3. **闭式增量 LS-SVM + 充分统计量机制**：将 one-vs-all 分类器与增量统计量累加相结合，无回放、无优化、闭式求解，在参数效率和隐私保护（不存储旧样本）方面具有独特优势，可推广至其他预训练模型的增量分类任务。
4. **控制变量评估文本信息的影响**：Vis+Text 对照实验清晰隔离了文本信息的边际贡献，这种"控制变量"的实验设计是验证多模态组件价值的典范，值得在同类研究中复用。

## 关键术语表
**Class-Incremental Learning (CIL)**：类增量学习，要求模型在不断接收新类别数据时保持对已学类别的分类能力，而不遗忘旧知识。

**Modality Gap**：模态间隙，指 CLIP 共享嵌入空间中图像特征与文本特征的几何分离现象，本文量化为 $g_c = 1 - \mu_c^\top t_c$。

**Residual Fusion**：残差融合，将 CLIP 多层的 CLS token 特征经轻量 MLP 学习残差修正量，叠加到最终层视觉特征上，增强任务适应性。

**LS-SVM（Least Squares Support Vector Machine）**：最小二乘支持向量机，将 SVM 的不等式约束替换为等式残差，使分类器权重可通过闭式求解，适合增量统计量累积。

**Sufficient Statistics**：充分统计量，指 $G_t$（特征相关性）、$Q_t$（特征-标签相关性）、$\mathbf{s}_t$（特征累积和），用于闭式增量更新分类器而无需存储历史数据。

**One-vs-All $\pm 1$ Coding**：一对一多编码，每个类别的学习目标为 +1，其余所有类别为 -1，区别于 one-hot 编码中非目标类别为 0。

**Exemplar-free CIL**：无样例类增量学习，不允许存储或回放历史样本的增量学习协议。

**Kernel-induced Feature Map**：核诱导特征映射，通过固定随机投影 + 非线性激活将原始特征映射到高维空间，使线性分类器等价于原始空间中的非线性决策边界。

## 可复现要素
- **数据集**：CIFAR100、CUB200、ObjectNet、ImageNet-R、FGVCAircraft、StanfordCars、Food101、SUN397、UCF101，均为公开研究数据集。
- **代码/权重**：论文未明确提供开源代码仓库链接；使用 LAION-400M 预训练的 CLIP ViT-B/16（OpenCLIP），公开可得。
- **关键超参**：基座会话训练 5 epochs、SGD lr=0.01、batch size=64、$\lambda_{\text{adapt}}=0.01$、$C_{\text{svm}}=10$、$D=15000$、隐藏维度 $d_h=256$、选层子集 $B=\{6,8,10,12\}$；LS-SVM 正则系数 $\lambda$ 从 16 个候选值中搜索；层子集和 $\lambda$ 仅在基座会话验证集上选定。
