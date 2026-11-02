---
title: "zPocket-Better-Pocket-Finding-with-Frozen-Co-Folding-Feature"
source: https://arxiv.org/pdf/2610.17696v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:49:22"
---

# 论文速读：zPocket-Better-Pocket-Finding-with-Frozen-Co-Folding-Feature

## 一句话总结
本文提出 **zPocket**，一种在冻结的共折叠模型（Protenix/ESMFold2）内部 pair 特征上训练的轻量残基级口袋预测头，配合新构建的时间切分基准 **PocketWatch**，在序列相似度、结构相似度与隐蔽性全分层下显著超越传统几何检测器与已有深度学习基线，并在晶体学碎片筛选与人类蛋白质组新颖口袋假设发现中验证了实用价值。

## 研究问题与动机
- **核心问题**：蛋白质-配体结合口袋（pocket）预测是结构基础药物设计的起点，但现有深度学习方法高度依赖输入结构的显式几何/物理化学特征，难以直接利用共折叠模型已学习到的丰富序列-结构联合表征。
- **基准缺陷**：AlphaFold3 时代共折叠模型的训练数据截止日通常早于近年 PDB 沉积高峰，而现有公开基准（UniSite、LIGYSIS 等）的测试口袋多已在模型训练期内出现，导致评估无法区分模型是“真正泛化”还是“记忆训练分布”。
- **分层需求**：口袋预测面临序列相似性泄漏、结构同源性泄漏以及隐蔽口袋（apo 结构未开放、holo 结构才显现）等多重挑战，亟需一套按序列 identity、结构覆盖度（max_qcov）与隐蔽性严格分层的评估协议。
- **应用缺口**：碎片筛选与全蛋白质组 novelty 挖掘需要高召回且可重排序的口袋预测工具，而现有单链几何检测器在多口袋蛋白与浅表肽结合界面处表现有限。

## 核心贡献（创新点）
1. **构建 PocketWatch 时间切分基准**：以 PDB 沉积日期为硬截止（2021-09-30 前为训练/验证，2024-01-01 后为测试），按 UniProt 序列聚合配体接触残基并以 5 Å 单连通聚类定义离散口袋位点，首次提供可分离“泛化”与“记忆”的评估平台。
2. **提出 zPocket 冻结 trunk + 轻量 head 范式**：仅使用前向缓存的 co-folding trunk pair 表示 $z \in \mathbb{R}^{L \times L \times C}$，训练参数量仅 26 万~33 万的残基级分类头，无需微调大模型即可输出每个残基的口袋概率。
3. **配对型 head 架构设计**：在共享 readout（row/col mean + 对角线拼接 → LayerNorm + MLP）前引入 0~2 个 AlphaFold 式三角形 pair mixing 块，证明 pocket 信号主要编码在 pair 表示中而非单 residue 表示。
4. **双解码策略（聚类 + 重排序）**：支持将残基概率聚类为质心，或直接用 head 对 fpocket/P2Rank/Lacuna/VN-EGNN 等外部候选腔体按 lining 残基概率聚合重排序，实现“粗筛定位 + 精排打分”的解耦。
5. **跨 trunk 泛化与无参集成**：在 7 种开源共折叠模型（Protenix-v1/v2、ESMFold2、OpenFold3-preview、Boltz-1、Chai-1、RF3）上统一训练，证明 pocket 信号普遍存在且 trunk 差异仅 0.024 AUPRC；简单 logits 平均即可稳定提点至 0.777。

## 方法详解
- **特征提取**：对每条蛋白序列运行一次冻结的 Protenix-v2 或 ESMFold2（含 MSA，10 轮 recycling），缓存半精度 pair 表示 $z$、单 residue 表示 $s$ 及 distogram logits，计算一次后全程复用。
- **Head 架构**：
  - `pooled`：直接对 $z$ 每残基取 row mean、column mean 与对角线拼接，过 LayerNorm+MLP 输出标量 logit。
  - `axial`：先将 $z$ 投影至窄宽度（32），执行一次行→列自注意力块。
  - `pairformer1/2`：在 pooled 基础上前置 1/2 个 AlphaFold 式三角形注意力块（pair width=64），使残基预测可捕获非直接接触的几何上下文。
- **损失函数**：掩码二元交叉熵（Focal 形式）$\mathcal{L} = \frac{\sum_i m_i \ell_i}{\sum_i m_i}$，其中 $\ell_i = \mathrm{BCE}_{w,\gamma}(\hat{y}_i, y_i)$，经验证集优选取 $w=1, \gamma=0$（即标准 BCE）；掩码 $m_i$ 屏蔽已见口袋、界面口袋及UNK残基。
- **训练协议**：单蛋白前向，梯度累积 8 蛋白/步，Adam 优化器（lr=2e-3），20 epoch（single head 40 epoch），每 2 epoch 评估，以验证集 per-target AUPRC 选 checkpoint，仅在测试集评估一次。
- **解码器**：
  - *聚类解码*：阈值 $\tau=0.3$ 筛选高置信残基，以 Cα 坐标为种子做加权均值漂移（bandwidth=14 Å，最小簇大小 $m=3$），密度峰即口袋质心。
  - *重排序解码*：保留外部检测器候选，用 zPocket head 对 covering shell（5/8/10 Å 或原生 lining list）内残基概率做聚合（mean/max/sum/Top-k），重新打分排序。
  - *混合解码*：合并 zPocket 聚类质心与外部候选去重后，统一用 head 重排序。

## 实验与结果
- **数据集规模**：PocketWatch 含 Train 11,016 / Val 1,553 / Test 1,566 条 UniProt chain；有效测试集经 mask 后为 **1,275 蛋白 / 1,562 口袋**，正残基率 4.86%。
- **残基级预测（per-target AUPRC）**：
  - zPocket (ESMFold2 pairformer2)：**0.745**（AUROC 0.951，Rp 0.682）
  - VN-EGNN (PW)：0.598；UniSite-1D (PW)：0.595
  - VN-EGNN (pub)：0.559；UniSite-1D (pub)：0.531
  - P2Rank：0.477；随机基线：0.049
- **分层泛化**：在序列 identity、max_qcov、crypticity 所有 stratum 中 zPocket 均领先；max_qcov < 20%（高度新颖）时 AUPRC 降至 0.327，但仍优于同期基线。
- **Trunk 对比**：Protenix-v2 pairformer2 达 0.757，ESMFold2 为 0.745；7 种 trunk 差距仅 0.024，head 架构选择比 trunk 选择更重要（pairformer2 vs pooled 增益 0.08~0.12）。集成全部 7 种 trunk logits 达 **0.777**。
- **质心预测（DCC$_{N+2}$, 4 Å）**：zPocket 自身聚类精度最高（60% 命中），但平均仅发射 1.1 个质心，recall 受限；重排序 fpocket/Lacuna 等基线平均提升 0.090；混合解码（zPocket + VN-EGNN PW）最佳配置达 **0.675**。
- **应用验证**：
  - *碎片筛选*（12 蛋白/35 位点）：zPocket AUPRC 0.494 vs P2Rank 0.459；重排序 fpocket 后 DCC$_{N+2}$ 提升至 0.391（受限于多口袋蛋白与片段低亲和力）。
  - *人类蛋白质组扫描*：对 ~20,000 reviewed UniProt 预测 ~22,000 口袋，经几何/ novelty/ motif 过滤后保留 **155 个高置信新颖口袋假设**（如 DET1 降解体招募位点、COP1 肽结合浅 groove）。

## 相关工作脉络
- **传统几何空腔检测器**（fpocket、P2Rank、DeepPocket）：依赖溶剂可及表面枚举与局部理化描述符，zPocket 完全数据驱动且无需任何几何先验，证明 trunk pair 特征已隐式编码空腔信息。
- **图/体素深度学习检测器**（VN-EGNN、DeepSite）：VN-EGNN 读取结构图+虚拟节点，UniSite 读取序列 mask；zPocket 表明冻结 co-folding trunk 的表征质量已超越从头学习的图卷积/序列 Transformer，且参数量仅其 1/10。
- **共折叠特征利用先例**（AF2Bind、Boltz-2）：AF2Bind 需注入 20 个“诱饵”氨基酸间接学习结合信号；zPocket 直接消费 trunk 内部 $z$ 表示，范式更简洁、训练成本更低，且无需修改 trunk 输入。
- **基准与评估体系**
