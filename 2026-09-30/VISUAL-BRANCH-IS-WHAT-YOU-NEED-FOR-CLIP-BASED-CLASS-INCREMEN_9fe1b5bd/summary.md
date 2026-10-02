---
title: "VISUAL-BRANCH-IS-WHAT-YOU-NEED-FOR-CLIP-BASED-CLASS-INCREMEN"
source: https://arxiv.org/pdf/2609.37888v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:35:33"
field: "持续学习/增量学习"
keywords: ["类别增量学习", "CLIP", "解析分类器", "最小二乘SVM", "视觉表征", "无回放学习"]
innovations: ["提出纯视觉增量分类器框架Vis，移除CLIP文本分支", "设计多尺度视觉层残差融合增强最终表征", "基于加法充分统计量的核化增量LS-SVM闭式更新"]
benchmarks: ["CIFAR100", "FGVCAircraft", "StanfordCars", "ImageNet-R", "CUB200", "UCF101"]
---

# 论文速读：VISUAL-BRANCH-IS-WHAT-YOU-NEED-FOR-CLIP-BASED-CLASS-INCREMENTAL-LEARNING

## 一句话总结
论文挑战了基于CLIP的类别增量学习（CIL）中依赖文本分支的常见设计，提出了一种纯视觉的增量学习方法Vis，通过增强CLIP视觉表征并采用核化增量最小二乘SVM，在无需文本分支的情况下实现了SOTA性能。

## 研究问题与动机
- CLIP-based CIL方法通常通过文本编码器生成类别模板嵌入作为分类器权重，但这种设计假设文本特征能与视觉类别分布对齐
- 论文通过几何分析和优化实验发现CLIP中存在**模态间隙**：文本嵌入与视觉原型的分布存在明显分离，导致文本分类器权重偏离实际视觉类别分布
- 在相同cosine分类器训练条件下，使用视觉原型初始化比文本特征初始化可获得更低损失和更高增量精度，说明文本分支可能并非必需
- 核心问题：能否在部署时完全移除文本分支，仅在视觉空间中构建高效的增量分类器？

## 核心贡献（创新点）
- **提出纯视觉增量学习框架Vis**：彻底移除部署时的文本分支，将增量分类器完全构建在视觉空间中，与现有依赖文本引导的CLIP-CIL方法形成本质区别
- **多尺度视觉层特征残差融合**：仅使用基础会话数据，通过轻量级MLP残差融合模块增强CLIP最终视觉表征，整合多层ViT特征提供更任务自适应的视觉表示
- **核化增量最小二乘SVM分类器**：采用显式非线性特征映射构建有限维核空间，使线性分类器等价于核空间中的非线性决策边界，同时保持闭式解的可增量更新性
- **加法充分统计量的闭式增量更新**：维护特征相关性$G_t$、特征-标签相关性$Q_t$和累积特征和$\mathbf{s}_t$三个充分统计量，新类别到来时通过矩阵扩展和重新求解获得所有已见类别的分类器权重，无需存储历史样本

## 方法详解
**残差融合模块（Residual Fusion）**：
- 从冻结的CLIP视觉编码器提取各层CLS token：$\mathbf{h}_\ell(\pmb{x}) = \mathbf{h}_{\text{cls}}^\ell(\pmb{x}) \in \mathbb{R}^{d_v}$
- 拼接多层特征并通过两阶段MLP生成残差修正：$\mathcal{M}(\mathbf{m}(\pmb{x})) = U \cdot \text{GELU}(V\mathbf{m}(\pmb{x}) + \mathbf{b}_V) + \mathbf{b}_U$
- 增强视觉表示：$\mathbf{u}(\pmb{x}) = \mathbf{h}_L(\pmb{x}) + \mathcal{M}(\mathbf{m}(\pmb{x}))$
- 基础会话训练目标：$\mathcal{L}_{\text{adapt}} = \frac{1}{|\mathcal{D}_1|}\sum [\ell_{\text{ce}}(g_\omega(\mathbf{u}(\pmb{x})), \pmb{y}) + \lambda_{\text{adapt}}\|\mathbf{u}(\pmb{x}) - \mathbf{h}_L(\pmb{x})\|_2^2]$
- 初始化为零输出，确保从最终层特征开始

**核化增量LS-SVM分类器**：
- 显式非线性特征映射：$\phi(\pmb{x}) = \sigma(\pmb{R}^\top \mathbf{u}(\pmb{x})) \in \mathbb{R}^D$，其中$\pmb{R}$随机采样后固定，$\sigma$为ReLU激活
- 吸收偏置项：$\tilde{\phi}(\pmb{x}) = [\phi(\pmb{x}); 1] \in \mathbb{R}^{D+1}$
- 多类LS-SVM目标函数：$\min_{\tilde{W}} \frac{1}{2}\text{tr}(\tilde{W}^\top \Gamma \tilde{W}) + \frac{C_{\text{svm}}}{2}\|\tilde{\Phi}\tilde{W} - Y\|_F^2$，其中$Y \in \{-1, +1\}^{N \times C}$为one-vs-all标签矩阵
- 闭式解：$\tilde{W}^\star = (\Gamma + C_{\text{svm}}\tilde{\Phi}^\top\tilde{\Phi})^{-1}(C_{\text{svm}}\tilde{\Phi}^\top Y)$

**增量充分统计量更新**：
- 维护：$G_t = \sum_{s=1}^t \tilde{\Phi}_s^\top \tilde{\Phi}_s$（特征相关性），$Q_t = \sum_{s=1}^t \tilde{\Phi}_s^\top Y_s^{(t)}$（特征-标签相关性），$\mathbf{s}_t = \sum_{s=1}^t \tilde{\Phi}_s^\top \mathbf{1}$（特征累积和）
- 新类别$m_t$到来时扩展$Q$：$Q_{t-1}^\uparrow = [Q_{t-1}, -\mathbf{s}_{t-1}\mathbf{1}_{m_t}^\top]$，新增列表示历史样本对新类别的-1贡献
- 更新规则：$G_t = G_{t-1} + \tilde{\Phi}_t^\top \tilde{\Phi}_t$，$Q_t = Q_{t-1}^\uparrow + \tilde{\Phi}_t^\top Y_t^{(t)}$，$\mathbf{s}_t = \mathbf{s}_{t-1} + \tilde{\Phi}_t^\top \mathbf{1}$
- 分类器重算：$\tilde{W}_t^\star = (\Gamma + C_{\text{svm}}G_t)^{-1}(C_{\text{svm}}Q_t)$
- 推理：$\hat{y} = \arg\max_{c} \tilde{W}_t^\star[:, c]^\top \tilde{\phi}(\pmb{x})$

## 实验与结果
**数据集**：9个基准（Aircraft、CIFAR100、Cars、ImageNet-R、CUB、UCF、Food、SUN397、ObjectNet），采用B-m Inc-n协议（如B0 Inc10、B50 Inc10等）

**主要结果**（B0 Inc10协议，平均准确率$\bar{A}$与最终会话准确率$\mathcal{A}_B$）：
- **Aircraft**：Vis $\bar{A}=75.21$, $\mathcal{A}_B=66.46$，领先第二BOFA（70.96/60.43）约4-6个百分点
- **CIFAR100**：Vis $\bar{A}=90.64$, $\mathcal{A}_B=85.26$，领先RanPAC（89.3/83.18）约1.3-2个百分点
- **Cars**：Vis $\bar{A}=94.47$, $\mathcal{A}_B=91.43$，小幅领先BOFA（94.21/90.20）
- **ImageNet-R**（B0 Inc20）：Vis $\bar{A}=86.05$, $\mathcal{A}_B=80.58$
- **CUB**（B0 Inc20）：Vis $\bar{A}=87.59$, $\mathcal{A}_B=82.02$
- **UCF**（B0 Inc10）：Vis $\bar{A}=98.37$, $\mathcal{A}_B=99.09$
- **Food**（B0 Inc10）：Vis $\bar{A}=86.99$, $\mathcal{A}_B=80.25$

**消融实验**：ZS-CLIP → Visual LS-SVM → w/ Kernel Map → w/ Residual Fusion (Full)，各组件逐级贡献正向收益

**对比基线**：ACIL、RanPAC、DualPrompt、CODA-Prompt、SimpleCIL、RAPF、CLG-CBM、PROOF、BOFA

** forgetting分析**：Vis在8组实验中5组取得最低遗忘率，如UCF B0 Inc10仅0.72（第二RanPAC为1.32）

**可控文本实验**：Vis + Text（引入文本原型）在Aircraft上$\bar{A}$下降2.03点，证明视觉 formulation 为主因

**训练效率**：Vis总训练时间1.72分钟（Cars B0 Inc10），推理延迟3.12ms/图像，参数量紧凑

## 相关工作脉络
- **Prompt-based CIL**（L2P、DualPrompt、CODA-Prompt）：通过在冻结Transformer上学习/选择prompt tokens适配不同任务；Vis完全不依赖prompt机制，直接在视觉特征空间构建解析分类器
- **Adapter-based CIL**（Fukuda等、Yu等）：在骨干网络中插入轻量模块；Vis同样冻结CLIP视觉编码器，仅训练附加的轻量残差融合模块
- **Analytic/Random-feature classifiers**（ACIL、RanPAC）：用闭式或递归更新替代梯度优化；Vis与之同属解析学习路线，但Vis采用one-vs-all ±1目标而非one-hot，且通过累加特征和$\mathbf{s}_t$支持新类别展开
- **CLIP-based CIL**（RAPF、CLG-CBM、PROOF、BOFA）：利用文本原型、prompt tuning或跨模态对齐进行增量识别；Vis证明完全不需要文本分支即可实现更强性能
- **代表闭式分类器改进**：Vis相比ACIL的one-hot Ridge采用LS-SVM的±1 formulation，相比RanPAC在表征增强阶段引入多尺度特征融合

## 局限性与未来方向
- **视觉表征冻结限制**：基础会话后残差融合模块和核特征空间固定不变，无法进一步适应后续任务的分布漂移
- **潜在改进方向**：探索在保持闭式统计分类器的同时，对视觉空间进行自适应更新的机制（论文自述）
- **超参数敏感性问题**：虽然实验显示对$\lambda_{\text{adapt}}$和$D$有一定鲁棒性，但层子集选择$B$和正则化系数$\lambda$仍需基础会话验证集调优

## 研究启发与可借鉴点
- **模态间隙诊断方法**：通过类级别差距$g_c = 1 - \pmb{\mu}_c^\top \pmb{t}_c$量化文本-视觉对齐程度，并关联到文本头退化指标，为多模态模型部署提供可复用的诊断框架
- **闭式充分统计量增量更新范式**：$G_t$、$Q_t$、$\mathbf{s}_t$的设计思路可迁移到其他需要避免梯度遗忘的增量学习场景，尤其是结合解析分类器时
- **显式随机特征映射+核方法**：用固定$\pmb{R}$和$\sigma$构造$\phi(\pmb{x})$实现有限维核近似，避免隐式核矩阵计算，这一设计在保持非线性容量的同时维持了增量更新的高效性
- **残差融合增强预训练表征**：仅用基础会话数据训练轻量修正模块，不改变主干编码器，这一策略可推广到其他需要适配下游任务但不希望破坏预训练知识的场景
- **可控变量实验设计**：Vis vs Vis+Text的对比实验隔离了文本信息的独立贡献，为验证"视觉优先"假设提供了严谨的实证依据

## 关键术语表
**Class-Incremental Learning (CIL)**：类别增量学习，要求模型在持续学习新类别的同时不遗忘已有类别，且通常不允许存储历史样本（无回放协议）
**Modality Gap**：模态间隙，指CLIP共享嵌入空间中视觉特征与文本特征之间的分布偏差，本文量化为$g_c = 1 - \pmb{\mu}_c^\top \pmb{t}_c$
**Least Squares SVM (LS-SVM)**：最小二乘支持向量机，将SVM的 hinge 约束替换为等式残差，得到闭式解且保留one-vs-all判别能力
**Sufficient Statistics**：充分统计量，此处指$G_t$、$Q_t$、$\mathbf{s}_t$三个可加量，用于无重建原始数据的情况下累积所有已见类别的信息
**Residual Fusion**：残差融合，通过轻量MLP从多层视觉特征学习对最终层CLS特征的修正量，增强下游任务适应性
**Kernel-induced Feature Map**：核诱导特征映射，用$\phi(\pmb{x}) = \sigma(\pmb{R}^\top \pmb{u}(\pmb{x}))$将视觉表示映射到高维空间，使线性分类器等价于核空间中的非线性决策
**Base Session / Incremental Session**：基础会话指首个任务（包含全部类别的子集），增量会话指后续到达的新类别任务序列

## 可复现要素
- **数据集**：CIFAR100、FGVCAircraft、StanfordCars、ImageNet-R、CUB200、UCF101、Food101、SUN397、ObjectNet，均为公开数据集
- **代码/权重**：论文未明确声明开源，但提到基于C3Box toolbox复现；使用LAION-400M预训练的CLIP ViT-B/16 backbone公开可用
- **关键超参**：$\lambda_{\text{adapt}}=0.01$、$C_{\text{svm}}=10$、$D=15000$、$d_h=256$、训练5 epochs、batch size=64、SGD lr=0.01；层子集$B=\{6,8,10,12\}$
- **协议**：B-m Inc-n标准协议，类顺序seed=1993
