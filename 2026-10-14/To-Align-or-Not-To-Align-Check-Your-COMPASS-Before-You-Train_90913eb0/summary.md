---
title: "To-Align-or-Not-To-Align-Check-Your-COMPASS-Before-You-Train"
source: https://arxiv.org/pdf/2610.18638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:35:31"
field: "模型组合与迁移学习"
keywords: ["模型兼容性", "跨模态对齐", "模型合并", "k-NN重叠", "表示收敛", "COMPASS"]
innovations: ["提出双轴诊断框架Ease×Ceiling预测模型对兼容性", "理论证明局部k-NN重叠可预测有限时间收敛", "在中位阈值四象限划分提供可操作的决策指南"]
benchmarks: ["Flickr30k", "COCO", "Task-Arithmetic Benchmark"]
---

# 论文速读：To Align or Not To Align: Check Your COMPASS Before You Train

## 一句话总结
本文提出 COMPASS，一种在模型对齐或合并前即可快速评估模型对兼容性的诊断方法，通过"难度（Ease）"和"天花板（Ceiling）"双轴打分，避免昂贵试错；在跨模态对齐（72对实验）中 Pearson 相关系数达 0.91，在任务算术合并（392对）中达 0.70，显著优于全局 CKA（分别为 0.63/0.68 和 0.25）。

## 研究问题与动机
- **核心问题**：在将两个预训练模型组合（对齐或合并）前，如何高效筛选出兼容性好的模型对？当前做法需先跑完昂贵的对齐/合并实验才能判断兼容性，搜索成本高。
- **现有方法不足**：
  1. 以往可迁移性指标（如 Task2vec、LEEP、LogME）仅针对单模型选择，未解决模型对兼容性问题。
  2. 全局表示相似性度量（如 CKA）在跨模态和合并任务中与下游表现的关联较弱（对齐仅 0.63/0.68，合并仅 0.25）。
  3. 模型对兼容性只能在完成对齐/合并后才能得知，缺乏前置诊断工具。

## 核心贡献（创新点）
1. **提出 COMPASS 诊断框架**：首次将模型对兼容性分解为"难度（Ease）"和"天花板（Ceiling）"两个廉价可计算的维度，实现训练前的兼容性排名。
2. **理论动机**：基于局部 k-NN 重叠构建有限时间收敛定理（Theorem 4.1）和天花板上界定理（Theorem 4.2），从理论上证明局部重叠为何能预测兼容性。
3. **双轴相乘排名策略**：将 Ease 与 Ceiling 相乘得到连续评分，并通过中位数阈值划分四个行动区间（act now、budget-limited、ceiling-limited、incompatible），提供可解释的决策指导。
4. **与已有工作的本质区别**：不同于 prior works 依赖训练后评估，COMPASS 仅需在探针集上做前向传播（成本降低两个数量级），即可在训练前预测兼容性；且聚焦局部邻居重叠而非全局表示对齐。

## 方法详解
- **Probe Set**：跨模态对齐使用 WIT 数据集中 1024 对图文样本；模型合并使用 MNIST 的 2048 个样本。
- **Ease 得分**：衡量配对初始状态下的局部一致性。对每对模型 X、Y，计算共同 k-NN 重叠率 $O(k)=\frac{1}{N}\sum o_i^{(k)}$，其中 $o_i^{(k)}=|\text{kNN}_X^{(k)}(i)\cap \text{kNN}_Y^{(k)}(i)|/k$。在跨模态对齐中为预训练后的重叠；在合并中为 fine-tune 前的重叠。
- **Ceiling 得分**：衡量范式层面的兼容性上限。对齐任务中，为 vision encoder 在所有 text encoder 上的平均重叠；合并任务中，定义为 $1-\frac{1}{|L|-1}\sum_{v\in L}O(b,v)$，反映预训练范式的专业化程度。
- **综合评分**：COMPASS = Ease × Ceiling，越高越好。按中位数划分四象限：
  - act now（高 Ease、高 Ceiling）
  - budget-limited（低 Ease、高 Ceiling）
  - ceiling-limited（高 Ease、低 Ceiling）
  - incompatible（低 Ease、低 Ceiling）

## 实验与结果
- **数据集**：Flickr30k、COCO（跨模态对齐）；8 个图像分类任务（MNIST、SVHN 等，合并）；WIT（probe）、MS-COCO（对齐训练）。
- **评估基线**：Global CKA、Task-vector cosine、Paradigm distance、元数据预测器等。
- **主要结果**：
  - 跨模态对齐：COMPASS 在 Flickr30k I2T R@1 上与下游 Pearson r=0.90，COCO 上 r=0.91；最佳对 Top-1 准确率 0.635，Regret=0%；Spearman ρ=0.91/0.93。
  - 模型合并：COMPASS 在 mean retention 上 r=0.70；Top-1 保留率 0.947，Budget to best=5/392。
  - Global CKA 对齐仅 r=0.63/0.68，合并仅 r=0.25，显著弱于 COMPASS。
- **最强结果**：跨模态对齐 COMPASS 在 Flickr30k 达到 r=0.91（vs CKA 的 0.63），提升 44%。

## 相关工作脉络
- **Platonic/Aristotelian Representation Hypothesis**：前者主张全局收敛，后者指出局部邻居重叠才是真正的兼容性来源；本文基于 Aristotelian 假设构建诊断。
- **I-Con 框架 (Alshammari et al., 2025)**：统一对比、非对比和聚类目标，通过 KL 散度刻画训练目标对局部拓扑的影响，支撑 Ceiling 设计。
- **CSA (Li et al., 2025)**：用于跨模态对齐的闭式 adapter，本文作为下游评估管道。
- **Task Arithmetic (Ilharco et al., 2023)**：模型合并的经典方法，本文在其 14 backbone × 8 task 基准上验证 COMPASS。
- **Global CKA (Kornblith et al., 2019)**：全局表示相似性基线，本文证明其远不如局部 k-NN 重叠有效。
- **定位差异**：相比 prior 工作仅做事后评估，COMPASS 提供前置诊断；相比单一相似度指标，提出双轴正交分解。

## 局限性与未来方向
- **Ceiling 定义依赖领域知识**：Ceiling 无法用一个通用公式表达，需针对不同应用单独实例化。
- **理论假设为充分条件**：有限时间收敛定理的假设（A1-A3）非必要条件，某些训练动态可能不满足。
- **扩展范围待验证**：目前仅在跨模态对齐和任务算术合并上验证，其他双模型交互场景（如风格迁移、模型蒸馏）的适用性未知。
- **低-ease+低-ceiling 并非绝对不可能**：在极大训练预算或不同范式下仍可能成功，四象限仅描述有限预算下的效率而非不可能性。

## 研究启发与可借鉴点
1. **双轴正交分解思路**：将兼容性拆分为"初始距离"和"范式上限"，便于理解和决策，可迁移至其他模型组合场景。
2. **局部 k-NN 重叠优于全局度量**：在表示相似性度量中应优先选用局部邻居指标，对跨模态、多任务场景鲁棒。
3. **探针集复用降低成本**：每个模型只需一次前向传播，复杂度随模型数线性而非成对增长，适合大规模模型库筛选。
4. **四象限决策框架**：用中位数阈值划分行动区间（优先/加预算/换范式/放弃），为工程实践提供可操作指南。

## 关键术语表
**COMPASS**：COMPatibility ASsessment Score，通过 Ease 和 Ceiling 双轴评估模型对兼容性的前置诊断工具。
**Ease**：配对特定得分，衡量模型对在训练/合并前的初始局部 k-NN 重叠程度。
**Ceiling**：范式级别得分，衡量两模型所属训练范式在最优情况下可支持的最大重叠。
**Local k-NN Overlap**：局部 k-近邻重叠率，两编码器对同一样本集的近邻集合交集比例。
**Cross-modal Alignment**：跨模态对齐，将视觉和语言表示映射到共享空间的任务。
**Task Arithmetic**：任务算术，通过权重空间中任务向量叠加合并多个微调检查点。
**CKA**：Centered Kernel Alignment，全局表示相似性度量。
**WIT / MS-COCO / Flickr30k**：多模态探针数据集和图像描述语料库，分别用于探针和下游评估。

## 可复现要素
- **数据集**：WIT（CC BY-SA 3.0）、MS-COCO（CC BY 4.0）、Flickr30k（Flickr 许可）、MNIST 等 8 个分类任务数据集均公开可用。
- **代码/权重**：代码将在接受后开源，GitHub 链接 https://github.com/mlbio-epfl/COMPASS；模型来自 timm 公开 checkpoint。
- **关键超参**：k=10（k-NN 邻居数）、Canonical dimension=512（对齐）、学习率 backbone 1e-4 / head 1e-3、Epochs=20、Batch size=32、AdamW + cosine schedule、Seed=42。
