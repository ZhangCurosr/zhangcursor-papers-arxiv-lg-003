---
title: "Zatom-2-Multitask-Pretraining-on-Atomistic-Data-for-Generati"
source: https://arxiv.org/pdf/2610.11454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:05:42"
field: "原子级生成建模与跨域迁移学习"
keywords: ["原子级生成模型", "多任务预训练", "条件流匹配", "力条件控制", "跨域迁移学习", "蛋白质设计", "分子生成", "材料生成"]
innovations: ["统一atom1/atom14多尺度Transformer架构，联合生成、结构预测与力/能量预测多任务预训练", "引入系统平均力范数作为全局条件控制生成样本的松弛状态", "联合生成-预测预训练显著提升低数据蛋白质长度外推可设计性（67.8%→74.8%）"]
benchmarks: ["QM9", "GEOM-Drugs", "MP20", "SCOPe-2k protein generation", "OMol25/OMat24 structure prediction"]
---

# 论文速读：Zatom-2: Multitask Pretraining on Atomistic Data for Generative Modeling across Domains

## 一句话总结
Zatom-2是一个统一的原子级生成模型，通过在OMol25（4M）和OMat24（1M）数据集上联合预训练生成、结构预测与力/能量预测任务，实现跨分子、周期性材料和蛋白质的生成建模，并在低数据蛋白质生成场景中通过联合生成-预测预训练显著提升可设计性。

## 研究问题与动机
1. **跨域建模割裂**：现有原子级生成方法高度专业化，分子/材料模型与蛋白质模型各自独立开发，未能探索单一模型在分子、周期材料和生物分子上的统一训练与微调。
2. **预训练数据利用不足**：OMol25（100M+ DFT计算）和OMat24（110M+ DFT计算）等大规模量子化学数据集包含丰富的化学组成、构象和能量多样性，但现有方法未充分利用其进行通用预训练。
3. **生成-预测分离**：机器学习原子势（MLIP）模型已被证明能学到可迁移的几何表征，但如何将物理约束（力/能量预测）与生成任务联合训练尚待探索。
4. **低数据生物分子生成**：蛋白质等生物大分子的结构数据稀缺，如何利用小分子/材料领域的高保真DFT数据实现有效迁移学习是关键挑战。

## 核心贡献（创新点）
1. **统一多任务架构Zatom-2**：提出单一Transformer架构支持分子/材料生成、结构预测、能量/力预测及蛋白质生成，无需为不同科学领域开发独立模型。
2. **atom1 + atom14多尺度标记化**：引入统一的原子级tokenization方案（分子用atom1、蛋白质用atom14），使原子坐标隐含元素/序列信息，无需单独的扩散轨迹处理原子类型。
3. **力条件控制（Force Conditioning）**：首次将系统平均力范数作为全局条件输入，实现对生成样本松弛状态（低力/高力 regime）的显式控制。
4. **联合生成-预测预训练促进迁移**：证明在OMol25/OMat24上联合预训练生成+结构预测+力/能量预测，可将蛋白质设计的长度外推可设计性从67.8%提升至74.8%。
5. **数据与模型扩展性验证**：系统验证模型容量和数据覆盖率对各项任务的互补增益，表明向完整OMol25/OMat24（各100M）扩展具有明确价值。

## 方法详解
**架构设计**：
- **U形多尺度Transformer**：输入初始化器 → 原子编码器（局部注意力，128 keys）→ 下投影到token → Transformer主干（全局token注意力）→ 上投影回atom → 原子解码器。原子特征$Q_L$编码局部几何交互，token特征$Q_I$负责全局信息交换。
- **时间门控多任务损失**：$\mathcal{L} = \lambda_{FM}\mathcal{L}_{FM} + \lambda_{lDDT}\mathcal{L}_{lDDT} + g_{1Å}(t)\sum\lambda_k\mathcal{L}_k + m_E\lambda_E\mathcal{L}_E + m_F\lambda_F\mathcal{L}_F + \mathbf{1}_{MLIP}\lambda_{eq}\mathcal{L}_{eq}$，其中$g_\delta(t)$在接近端点时门控辅助任务，防止目标力泄漏。
- **条件流匹配（Conditional Flow Matching）**：采用速度预测范式，$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$，损失为$\mathbb{E}[\|\hat{\mathbf{v}}_\theta(\mathbf{x}_t, t) - \mathbf{v}_t\|_2^2]$，推理时使用EDM-style采样。
- **力条件定义**：$s_F = \log_{10}\left(\frac{1}{L}\sum_{\ell=1}^L \|\mathbf{F}_\ell\|_2\right)$，作为全局描述符嵌入到输入初始化器。
- **能量重构**：$\widehat{E}_b = \sum_{\ell \in b} \varepsilon_{d, z_\ell} - n_b(\mu_d + s_d \widehat{e}_b)$，通过域归一化的残差能量预测物理能量。
- **Recycling机制**：将clean坐标估计$\widehat{\mathbf{x}}_1$转换为token级distogram，更新token-pair conditioning进行第二轮前向传播（默认N=2 pass）。

## 实验与结果
**预训练数据**：OMol25 4M（3,986,754例）+ OMat24 1M（1,009,850例），联合采样，80 epoch。

**核心结果**：
- **QM9生成**：Zatom-2（joint QM9+MP20）validity达95.72%，优于Zatom-1-L（95.26%）。
- **GEOM-Drugs生成**：validity 96.90%，Unique 99.61%，PB-valid 91.47%，AFD 0.016。
- **MP20材料生成**：MetaSUN达4.92%，远超ADiT（1.0）、Crystalformer（3.1）及Zatom-1-L（0.64）。
- **力条件一致性**：AUROC达0.817（OMol25），MAE约0.616 eV Å⁻¹，能可靠区分低/高力 regime。
- **蛋白质迁移（SCOPe-2k）**：
  - 50-128残基（分布内）：Zatom-2(V)+FC designability 89.09%，优于scratch（88.22%）和RFdiffusion3（91.07%）。
  - **129-256残基（长度外推）**：Zatom-2(V)+FC designability **74.80%**，较无预训练（67.81%）提升**6.99个百分点**；Per-sequence success达45.79%，均显著优于对比方法。
- **Scaling**：Full dataset + Large模型在各项指标上持续优于小数据/小模型配置。

## 相关工作脉络
1. **RFdiffusion3 / Emyx**：蛋白质all-atom生成模型，采用atom14表示和U形架构；Zatom-2借鉴其多尺度思想但统一扩展到小分子和材料领域。
2. **TABASCO / Zatom-1**：Zatom-1已实现分子-材料统一生成；Zatom-2在此基础上增加结构预测、力/能量预测任务和力条件控制。
3. **Proteina / La-Proteina**：流匹配蛋白质生成模型；Zatom-2证明跨域预训练可迁移到蛋白质生成，弥补了蛋白质领域数据稀缺问题。
4. **MLIPs（UMA / eSEN / AllScAIP）**：专用原子势模型，学习高精度能量/力预测；Zatom-2将MLIP目标作为辅助任务而非主任务，验证了物理监督对表征质量的增益。
5. **REPA / CrystalREPA**：通过表征对齐将MLIP知识迁移到生成模型；Zatom-2采用端到端多任务联合训练，无需额外对齐损失。
6. **GEOM-Drugs / OMol25 / OMat24**：大规模量子化学数据集；本文首次将这些数据集用于统一生成模型的预训练。

## 局限性与未来方向
1. **数据规模受限**：仅使用OMol25/OMat24的4M/1M子集（完整数据集各100M），未充分利用全量数据。
2. **力条件精度不足**：虽能区分低/高力 regime，但对特定子集的精准力匹配仍有挑战，未来可探索逐原子力向量条件。
3. **缺乏精确旋转等变性**：未强制能量-力梯度一致性或严格旋转等变性，限制了其在高精度分子动力学中的应用。
4. **蛋白质实验规模小**：仅在SCOPe-2k（2,000个结构域）上验证，未来需扩展到大規模蛋白质数据集。
5. **MLIP精度有限**：作为非专用模型，能量/力预测误差显著高于UMA等专用势函数，不支持高精度物理模拟。

## 研究启发与可借鉴点
1. **联合生成-预测预训练范式**：将MLIP类预测目标（力/能量）作为辅助任务与生成任务联合训练，可显著提升下游迁移学习效果，值得在其他跨域生成任务中验证。
2. **隐式类型推断机制**：通过atom1标记化使坐标隐含元素/序列信息，避免了多模态扩散的复杂性，设计更简洁高效。
3. **力条件作为全局松弛描述符**：用平均力范数控制生成样本的物理状态，为生成模型引入物理先验提供了一个新颖且有效的条件接口。
4. **回收机制（Recycling）的distogram反馈**：将clean坐标估计转换为pair conditioning进行第二轮前向传播，可在不显著增加计算成本的前提下提升表示质量。
5. **跨域预训练迁移到数据稀缺领域**：利用化学/材料领域的高保真DFT数据预训练，再迁移到蛋白质设计等低数据场景，是解决生物大分子数据瓶颈的有效策略。

## 关键术语表
**Atom1 / Atom14标记化**：统一原子级表示方案，分子每原子一个token（atom1），蛋白质每残基14个坐标槽（atom14），使不同域共享同一Transformer架构。
**条件流匹配（Conditional Flow Matching）**：生成建模方法，学习从噪声分布到数据分布的连续OT路径，以速度预测为训练目标。
**Force Conditioning（力条件）**：将系统平均力范数作为全局条件输入，控制生成样本的松弛程度（低力/高力 regime）。
**MLIP（Machine Learning Interatomic Potential）**：机器学习原子势，学习预测原子间相互作用能量和力的模型。
**lDDT损失（smoothed lDDT）**：结构预测的局部距离差异测试损失，评估生成结构与参考结构的局部几何相似性。
**MetaSUN**：材料生成评估指标，衡量生成的材料满足"可验证、独特、新颖、亚稳态（$E_{hull} \leq 0.1$ eV/atom）"的比例。
**Recycling机制**：将第一轮输出的clean坐标估计转换为distogram，更新token-pair conditioning进行第二轮前向传播。
**AFD（Atomistic Frechet Distance）**：应用于原子级结构的Frechet Inception Distance变体，评估生成分布与真实分布的相似性。

## 可复现要素
- **数据集**：OMol25（4M子集）、OMat24（1M子集）、QM9、MP20、GEOM-Drugs、SCOPe-2k；OMol25/OMat24为CC BY 4.0，其余为CC0/MIT。
- **代码**：开源于 https://github.com/Zatom-AI/nucleus，含完整文档、数据加载、训练/推理脚本。
- **模型权重**：论文未明确声明权重开源，需查看项目页面。
- **关键超参**：Token trunk blocks（S/base/L: 9/18/36）、Atom/token宽度（128/768）、Attention heads（4/12）、$\sigma_{data}=16$ Å、学习率1e-4、Adam (0.9, 0.95)、batch token budget（7,168/3,584/512）、80 epochs、EMA decay 0.999。
- **训练硬件**：NVIDIA A100 80GB GPU，4-16节点分布式训练。
