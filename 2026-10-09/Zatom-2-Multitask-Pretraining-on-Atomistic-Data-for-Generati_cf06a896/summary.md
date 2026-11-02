---
title: "Zatom-2-Multitask-Pretraining-on-Atomistic-Data-for-Generati"
source: https://arxiv.org/pdf/2610.11454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:05:44"
field: "原子级生成建模与跨域迁移学习"
keywords: ["atomistic generation", "flow matching", "multitask pretraining", "protein design", "machine learning interatomic potential", "force conditioning", "cross-domain transfer"]
innovations: ["统一atom1/atom14分词支持分子/材料/蛋白质跨域生成", "力条件生成控制样本弛豫状态", "生成-预测联合预训练提升低数据蛋白质迁移能力"]
benchmarks: ["OMol25", "OMat24", "QM9", "MP20", "GEOM-Drugs", "SCOPe-2k", "LeMat-GenBench"]
---

# 论文速读：Zatom-2-Multitask-Pretraining-on-Atomistic-Data-for-Generati

## 一句话总结
Zatom-2 是一个跨域统一原子级生成模型，通过在 OMol25（4M）和 OMat24（1M）电子结构数据集上进行多任务预训练（生成、结构预测、能量/力预测），实现了分子、材料和蛋白质的联合建模与生成，并在蛋白质生成低数据场景中展现出强大的迁移学习能力。

## 研究问题与动机
1. **学科分割问题**：现有原子级生成模型高度专业化于特定科学领域（化学 vs. 生物学），无法在分子、周期性材料和生物大分子间统一训练和微调单一模型。
2. **数据利用不足**：现有方法未充分利用高容量有机（分子）和无机体（材料）数据用于通用预训练，限制了表示学习的泛化能力。
3. **表征学习缺口**：尽管量子化学数据集（如 OMol25、OMat24）包含丰富的能量和力标签，但尚未被有效用于训练跨域可迁移的原子级表示。
4. **低数据场景需求**：生物分子生成面临结构数据稀缺问题，需要从小规模原子级数据中学习可迁移的几何和相互作用表征。

## 核心贡献（创新点）
1. **统一原子级架构**：提出 atom1 + atom14 混合分词方案，使分子、材料和全原子蛋白质生成任务可在单一标准 Transformer 架构中实现。
   - *区别*：不同于 RFdiffusion3 和 BoltzGen 等仅针对蛋白质的专用 tokenization，本文统一了化学与生物系统的表示。

2. **多任务联合预训练框架**：引入生成、结构预测与 MLIP 辅助预测（能量/力）的联合预训练课程，利用物理约束优化生成模型的潜在表示。
   - *区别*：不同于 standalone MLIP（如 UMA），本文测试生成与预测目标共存时共享特征的有效性。

3. **力条件生成控制**：首次将系统平均力范数作为全局描述子，实现对生成样本弛豫状态的显式控制。
   - *区别*：现有分子生成模型通常不区分 rattled（高力）与 relaxed（低力）构型，本文提供新的 force consistency 评估指标。

4. **跨域迁移至蛋白质生成**：证明原子级预训练可迁移到低数据蛋白质生成场景，长度外推下 designability 从 67.8% 提升至 74.8%。
   - *区别*：RFdiffusion3 在分布内蛋白质生成中表现更好，但在 OOD 长度外推场景下 Zatom-2 显著优于从头训练基线。

## 方法详解
1. **多尺度 Transformer 主干**：采用 U 形 atom-token-atom 架构，atom encoder 学习局部几何交互（Q_L），token trunk 进行全局信息交换（Q_I），atom decoder 细化融合特征。使用 pair-biased attention 和 DiT-style adaptive normalization。

2. **Flow Matching 生成框架**：采用速度预测形式，定义插值路径 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ 和归一化速度目标 $\mathbf{v}_t = (\mathbf{x}_1 - \mathbf{x}_0)/\sigma_{data}$，训练目标为 $\mathbb{E}[\|\widehat{\mathbf{v}}_\theta - \mathbf{v}_t\|_2^2]$。使用 EDM-style 采样策略提升样本质量。

3. **多任务损失函数**：
   $$\mathcal{L} = \lambda_{FM}\mathcal{L}_{FM} + \lambda_{lDDT}\mathcal{L}_{lDDT} + g_{1Å}(t)\sum_{k \in \mathcal{K}}\lambda_k\mathcal{L}_k + m_E\lambda_E\mathcal{L}_E + m_F\lambda_F\mathcal{L}_F + \mathbf{1}_{MLIP}\lambda_{eq}\mathcal{L}_{eq}$$
   其中 $\mathcal{K} = \{\text{element, sequence, fractional, lattice}\}$，时间门控函数 $g_\delta(t)$ 仅在 $t \to 1$ 时激活结构预测任务，防止 target-force 泄漏。

4. **力条件机制**：使用系统平均力范数作为全局条件：
   $$s_F = \log_{10}\left(\frac{1}{L}\sum_{\ell=1}^{L}\|\mathbf{F}_\ell\|_2\right)$$
   该条件通过 initializer 嵌入，显式告知模型系统的弛豫程度。

5. **自 recycling 机制**：从去噪网络获得干净坐标估计 $\widehat{\mathbf{x}}_1$ 后，将其转换为 token-level distogram，更新 token-pair 条件进行第二轮前向传播（N=9/18/36 trunk layers）。

6. **能量重建公式**：
   $$\widehat{E}_b = \sum_{\ell \in b}\varepsilon_{d,z_\ell} - n_b(\mu_d + s_d\widehat{e}_b)$$
   其中 $\varepsilon_{d,z}$ 为元素参考能量，$\mu_d, s_d$ 为归一化参数。

## 实验与结果
1. **预训练配置对比**：从小型 Zatom-2 (I) 到最终大型 Zatom-2 (VI)，逐步增加结构预测、OMat24 数据、力/能预测任务和模型容量（70M → 140M → 270M 参数）。

2. **分子生成质量（OMol25）**：Zatom-2 (VI) 达到 RDKit Validity 85.0%，AFD 0.016，Mean |F| 1.269 eV Å⁻¹，优于 Zatom-1（Validity 94.1%，AFD 0.047）。

3. **力条件一致性**：在 OMol25 结构预测任务上，AUROC 达 0.76-1.00，MAE 0.69-0.93 eV Å⁻¹，成功区分低/高力区间。

4. **GEOM-Drugs 生成**：Validity 96.90%，PB-valid 91.47%，接近 TABASCO（97.6%）但更高 novelty。

5. **MP20 材料生成**：MetaSUN yield 达 4.92%，显著超越所有基线（ADiT 1.0%，Crystalformer 3.1%）。

6. **蛋白质迁移学习（SCOPe-2k）**：
   - 分布内（50-128残基）：Zatom-2 (V) 设计成功率 63.39%， diversity 453.7 clusters
   - 长度外推（129-256残基）：Designability 从 67.81%（无预训练）提升至 74.80%，显著优于 RFdiffusion3（54.04%）

7. **缩放实验**：完整数据集 vs. 每域 500k 样本，全量训练在所有指标上 consistently 更优；模型容量扩大带来额外增益，尤其在 OMat24 结构预测上。

## 相关工作脉络
1. **RFdiffusion3 / BoltzGen**：蛋白质专用 tokenization（atom14）+ 坐标扩散，但无法跨分子/材料领域；Zatom-2 统一 atom1/atom14 支持跨域生成。

2. **TABASCO / EQGAT-diff**：分子生成专用模型，缺乏材料/蛋白质扩展能力；Zatom-2 在同一架构下支持三域。

3. **All-atom Diffusion Transformers (Joshi et al., 2025)**：统一分子/材料生成，但未探索蛋白质迁移；Zatom-2 在此基础上增加力条件与 MLIP 辅助训练。

4. **Proteina / La-Proteina**：蛋白质生成模型，依赖序列-结构分离；Zatom-2 通过原子级预训练学习几何表征，无需单独 diffusion over amino acid types。

5. **MLIP 模型（UMA, eSEN, Orb）**：专注性质预测，生成能力有限；Zatom-2 证明 MLIP 目标可作为生成模型的辅助预训练任务，提升表征质量。

6. **Crystalite / SymmCD**：材料生成专用，缺乏分子/蛋白质扩展；Zatom-2 以统一架构实现三域生成。

## 局限性与未来方向
1. **数据规模限制**：仅使用 OMol25 4M 和 OMat24 1M 子集，完整数据集各含 100M 样本，未充分挖掘大数据潜力。

2. **力条件精细度不足**：虽能区分低/高力区间，但区间内的精确力匹配仍具挑战，未来可探索逐原子力向量条件。

3. **MLIP 精度局限**：作为非专用模型，能量/力预测误差远高于专门 MLIP（如 UMA），不支持高精度分子动力学模拟。

4. **蛋白质数据规模小**：SCOPe-2k 仅含 2,000 蛋白结构域，需扩展至大规模蛋白质数据集验证泛化性。

5. **对称性约束缺失**：未强制执行能量-梯度一致性和精确旋转等变性，依赖旋转数据增强近似。

6. **双用途风险**：生成化学与蛋白质设计能力可能被滥用，需负责任的应用评估与安全筛查。

## 研究启发与可借鉴点
1. **生成-预测联合预训练范式**：将 MLIP 辅助目标（能量/力预测）融入生成模型预训练，可显著提升下游迁移学习效果，值得在材料科学、药物设计等领域验证。

2. **力条件生成机制**：以系统平均力范数控制生成样本弛豫状态的思路，可迁移至其他需要控制构型质量的场景（如反应路径采样、过渡态生成）。

3. **统一 atom1/atom14 分词方案**：通过 residue-level（蛋白质）和 atom-level（分子/材料）灵活 tokenization，实现跨尺度原子级建模，为未来统一生物-化学生成模型提供架构参考。

4. **时间门控多任务学习**：近端点（t→1）才激活结构预测任务的设计，有效防止 target-force 泄漏，此策略可推广至其他扩散/flow matching 多任务训练。

5. **自 recycling distogram 更新**：利用去噪估计的干净坐标生成 distogram 反馈至 token-pair 条件，增强长程几何理解，可借鉴于需要精确空间关系的生成任务。

## 关键术语表
**Flow Matching**：连续概率传输方法，通过 ODE 将噪声分布映射至数据分布，本文采用速度预测形式训练。

**Atom1 / Atom14 Tokenization**：atom1 表示每个原子为一个 token（分子/材料），atom14 表示每个残基占 14 个坐标槽位（蛋白质），支持统一架构处理不同尺度系统。

**MLIP (Machine Learning Interatomic Potential)**：基于机器学习的原子间势能面模型，提供能量和力的预测，本文作为辅助预训练目标提升生成模型表征质量。

**Force Conditioning**：以系统平均力范数 $s_F$ 作为全局条件，控制生成样本的弛豫程度，区分 rattled（高力）与 relaxed（低力）构型。

**lDDT Loss**：smoothed local Distance Difference Test loss，用于评估生成结构与参考结构的几何一致性，替代 RMSD 进行结构预测评估。

**AFD (Atomistic Frechet Distance)**：将 FID 应用于原子级结构的分布相似度度量，使用预训练 CT 编码器提取 256 维图级 embedding 计算 Fréchet 距离。

**MetaSUN**：MP20 材料生成评估指标，衡量 valid + unique + novel + metastable（$0 < E_{hull} \leq 0.1$ eV/atom）材料的比例。

**Designability**：蛋白质生成质量指标，定义为至少一个 SolubleMPNN 设计序列经 ESMFold2 重构后 Cα scRMSD < 1.5 Å 的 backbone 比例。

## 可复现要素
- **数据集**：OMol25（CC BY 4.0，4M 子集）、OMat24（CC BY 4.0，1M 子集）、QM9（CC0）、MP20（MIT）、GEOM-Drugs（CC0）、SCOPe（ freely available）
- **代码开源**：https://github.com/Zatom-AI/nucleus，包含数据加载、训练/推理代码及文档
- **关键超参**：模型规模 S/base/L 分别为 70M/140M/270M 参数；Token-trunk blocks 9/18/36；atom/token 宽度 128/768；注意力头数 4/12；坐标先验尺度 $\sigma_{data}=16$ Å；训练精度 bf16-mixed；Adam lr $10^{-4}$；EMA decay 0.999
- **训练配置**：80 epochs，每 epoch 约 5M 样本采样；Base 模型 per-GPU token budget 3,584（ Large 7,168）；梯度累积保持 ~458,752 tokens/update
