---
title: "zPocket-Better-Pocket-Finding-with-Frozen-Co-Folding-Feature"
source: https://arxiv.org/pdf/2610.17696v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:47:09"
field: "蛋白质小分子结合位点预测"
keywords: ["protein-ligand binding site prediction", "co-folding model", "pocket finding", "frozen trunk feature", "PocketWatch benchmark", "AF3 generalization"]
innovations: ["冻结共折叠trunk的极轻量per-residue head（pairformer架构）在全新时间切分基准上超越所有结构基线", "构建PocketWatch时间切分+结构/隐秘性三维分层基准以公平评估AF3模型泛化", "将共折叠特征用于片段筛选与人类蛋白质组新口袋假说发现的端到端验证"]
benchmarks: ["PocketWatch", "CryptoBench", "Runs N' Poses"]
---

# 论文速读：zPocket-Better-Pocket-Finding-with-Frozen-Co-Folding-Feature

## 一句话总结
本文提出了**zPocket**——一种基于冻结共折叠模型（co-folding model）内部特征的口袋残基预测头，在构建的**PocketWatch**时间切分基准上全面超越P2Rank、VN-EGNN、UniSite等现有方法；同时将方法应用于晶体学片段筛选和人类蛋白质组规模化新口袋假说发现。

## 研究问题与动机
1. **基准缺失**：已有口袋预测基准的蛋白质结构沉积时间早于AF3时代共折叠模型的训练截止日期，导致无法公平评估模型对新口袋的泛化能力（存在"记忆"而非"泛化"的嫌疑）。
2. **共折叠特征未被利用**：AF2Bind等前人工作表明AlphaFold2 trunk特征含口袋信号，但AF3时代共折叠模型的开放权重与丰富pair表示尚未被系统地用于口袋残基预测。
3. **结构依赖性限制**：传统几何腔检测器（fpocket、P2Rank）严重依赖输入结构的几何质量，而共折叠模型直接从序列提取表征，有望绕过这一瓶颈。
4. **隐秘口袋（cryptic pocket）识别难**：结合apo–holo RMSD标准定义的隐秘口袋对静态结构方法是难题，而共折叠特征是否包含构象变化信号值得检验。

## 核心贡献（创新点）
1. **PocketWatch数据集**：从PDB截至2025年底的结构中按UniProt整合生物相关口袋，按最早口袋沉积日期做时间切分，并叠加max_qcov（结构相似性）与crypticity（隐秘性）三维分层，首次允许对AF3共折叠模型做"未见口袋"评估。
2. **zPocket骨架-头部范式**：将Protenix-v2/ESMFold2等共折叠trunk完全冻结，在其pair表示$z \in \mathbb{R}^{L \times L \times c_z}$上训练极轻量head（26万参数），通过row/col mean + diagonal拼接后MLP输出每残基口袋概率。
3. **Trunk无关的口袋信号确认**：在7个开源共折叠trunk（Protenix-v2/v1、ESMFold2、OpenFold3-preview、Chai-1、Boltz-1、RF3）上以同一recipe训练pairformer2 head，AUPRC仅差0.024，证明口袋信号是共折叠表征的通用属性而非单模型特例。
4. **两种解码器设计**：聚类解码器（加权mean-shift过Cα坐标）生成精准但稀疏的 centroid；重排序解码器（聚合lining残基概率）可提升任意候选检测器的DCC$_{N+2}$，并提供"zPocket proposals + 其他方法 proposals"的hybrid融合方案。
5. **两个应用验证**：（a）12蛋白35个片段的In-silico筛选任务中zPocket仍领先；（b）对~20,000条人源Reviewed蛋白扫描，经多层novelty几何过滤后产出155个候选新口袋（DET1、COP1），展示从基础预测到靶点发现的端到端流程。

## 方法详解
- **特征缓存**：每条蛋白序列用MSA折叠一次（Protenix-v2 10轮recycle / ESMFold2 num_loops=10），只保留half-precision的$z$（pair representation）、$s$（single track）、distogram logits；trunk全程冻结，后续head训练不再重跑。
- **Head架构（递增expressivity）**：
  - `pooled`：直接对$z$做row mean / col mean / diag拼接，LayerNorm + MLP→标量，无pair mixing（264k参数）。
  - `axial`：先投影到窄width（32），再接一行×一列self-attention块。
  - `pairformerK`：接K个AlphaFold-style triangular pair block（64宽），让残基得分依赖非直接接触的几何邻域；K=1/2。
  - `single`：仅读最终层ESMC-6B语言模型embedding（ESMFold2特有），作为无结构输入的对照。
- **损失函数**：掩码per-residue binary cross-entropy，引入focal形式$\text{BCE}_{w,\gamma}(\hat{y}, y) = -[w(1-\hat{y})^\gamma y \log \hat{y} + \hat{y}^\gamma (1-y) \log(1-\hat{y})]$；实验选用$w=1, \gamma=0$（普通BCE），按每target val AUPRC选checkpoint，Adam LR=$2\times10^{-3}$，每优化步累积8条蛋白梯度。
- **聚类解码**：阈值$\tau=0.3$的残基种子加权mean-shift（带宽14 Å，最小3残基），收敛判据0.05 Å或100次迭代；蛋白全低于阈值时回退top-10残基。
- **重排序解码**：对已有detector的每个候选cavity，以其lining残基（5/8/10 Å shell或detector自带列表）上的head概率作aggregate（mean/max/sum/top-k），从28种配置中选最优；union池去重2 Å后也可重排。
- **PocketWatch分层标注**：
  - 序列相似性：MMseqs2对齐，≥50残基且覆盖一半序列即计为hit，取identity=max(对齐%长度, 1/短序列长度)。
  - 结构相似性max_qcov：基于PLINDER-style qcov公式，以Foldseek预截断库（≤2021-09-30沉积）作参考，最大权重二分图匹配避免自匹配膨胀。
  - Crypticity：继承CryptoBench/IsoDDE的apo–holo口袋RMSD定义（≤2 Å backbone / ≤2.5 Å all-atom），分cryptic / refuted / unknown三类。

## 实验与结果
- **基准与数据集**：PocketWatch effective test set = 1,275蛋白 / 1,562口袋（已掩码seen界面口袋）；正残基率4.86%。
- **每残基AUPRC（Table 1，ESMFold2 trunk）**：
  - zPocket pairformer2：**0.745**（per-target）/ 0.702（per-pocket）；AUROC 0.951。
  - 最强基线 VN-EGNN (PW)：0.598 / 0.559；UniSite-1D (PW)：0.595 / 0.552。
  - P2Rank：0.477 / 0.455；random ~0.05。
  - 仅读ESMC-6B single embedding即达0.611，仍超所有结构基线。
- **Trunk对比（Table 2）**：pairformer2在7个trunk上0.733–0.757；Protenix-v2略优（0.757），但架构差异远大于trunk差异（pairformer2比pooled高0.08–0.12）。七trunk等权集成至**0.777**。
- **分层鲁棒性**：无论按max_qcov或crypticity分层，zPocket pairformer2在每个bin都领先（图1C、图3）。
- **口袋centroid（Figure 4 / Table A7）**：
  - zPocket clusters ceiling=0.524（平均仅1.1个/蛋白），但精度最高（60%命中真位点）。
  - 用zPocket重排任意detector proposals平均提升DCC$_{N+2}$ **0.090**；加入zPocket proposals再重排再提**0.108**。
  - VN-EGNN (PW) + zPocket proposals 最佳达**0.675** DCC$_{N+2}$。
  - 低max_qcov（<80%）时所有方法骤降，DCC$_{N+2}$上限0.37，说明 novelty仍是难点。
- **片段筛选（12蛋白35位点）**：zPocket per-residue AUPRC 0.494 vs P2Rank 0.459；fpocket+zPocket重排最佳0.391 DCC$_{N+2}$，但区间宽（[0.16, 0.65]）不显著。
- **人源蛋白质组扫描**：21,646→359（去接口）→155个通过novelty+几何+confidence多重过滤的候选口袋，示例DET1（degrader-recruiter site）与COP1（肽结合热点亚位点）。

## 相关工作脉络
1. **AF2Bind [12]**：首次用AlphaFold2 pair rep+20个"bait"氨基酸间接预测口袋；zPocket改用更直接的per-residue head读frozen trunk特征，无需bait。
2. **P2Rank [19]** / **fpocket [21]** / **DeepPocket [1]** / **DeepSite [16]**：经典/几何腔检测+ML重排序或3D-CNN重排；zPocket与之定位不同——它不产生腔proposal，而是用纯序列→trunk→head路线给出per-residue概率来**重排**或**补充**这些方法。
3. **VN-EGNN [27]** / **UniSite [11]**：图神经网络/Transformer直接预测口袋；zPocket与它们的关键差异是trunk冻结+无结构输入（不需要先折叠），且zPocket可端到端训练于新数据。
4. **Lacuna [23]**：在构象ensemble上检测口袋；zPocket单帧fold即可，但zPocket可作为Lacuna proposals的下游ranker。
5. **CryptoBench [28]** / **IsoDDE [15]**：形式化隐秘口袋benchmark；本文借用其定义并叠加时间切分，揭示predicted fold精度比cryptic标签更能解释性能。
6. **Runs N' Poses [29]** / **PLINDER [10]**：训练-测试泄漏度量；本文采用其qcov思想但修正为搜索式（foldseek+bipartite matching）并放宽序列identity阈值。

## 局限性与未来方向
1. **真 novelty 口袋稀缺**：71.6% test pockets位于max_qcov≥80% bin，整体AUPRC被"熟悉口袋"主导；真正未见口袋的泛化仍待更大沉积库支撑。
2. **仅评估单体口袋**：界面口袋（interface-only）被mask以对齐单链baseline，多链场景未来需扩展。
3. **predicted site ≠ functional site**：高置信度口袋未必有生物学功能，但作者认为随着degrader/molecular glue/radioligand兴起，"小分子可结合位点"本身已具治疗价值。
4. **聚类解码recall有限**：平均1.1个/蛋白难以覆盖多口袋蛋白（>1 pocket时AUPRC从0.751降至0.599）；需探索更丰富proposal生成机制。
5. **Protenix-v2权重转专有许可**：主文结果均基于ESMFold2 trunk，Protenix-v2对比仅在附录；若无法复现则影响可重复性。
6. **片段口袋与药物口袋差异**：片段尺寸小、亲和力低，zPocket在片段集上的DCC$_{N+2}$（0.391）远低于PocketWatch（0.68），提示模态gap。

## 研究启发与可借鉴点
1. **"frozen trunk + tiny head"范式可直接迁移**：任何具备pair/single rep的预训练biomolecular模型（AF3、Chai-1、Boltz系）均可按相同recipe做下游任务适配，无需重训trunk。
2. **时间切分+多维分层构成可复用的generalization benchmark设计模板**：max_qcov（结构相似）与序列相似分层可推广至affinity prediction、interaction prediction等任务。
3. **Re-ranking作为模块化提升手段**：将学习式ranker叠加于几何检测器之上（DeepPocket思路的广义化）是一条低成本高收益的工程路线，可复用于任意提供候选proposal的方法。
4. **Crypticity vs. fold-accuracy交叉分析**：本文发现predicted fold RMSD比apo-holo cryptic标签更能解释性能，提示后续工作应直接用fold质量做stratification而非仅依赖实验标签。
5. **Proteome-scale scan的filter pipeline**：direct annotation → homology-propagated (Pfam/clan) → motif-propagated (Foldseek/Folddisco) → geometric plausibility → confidence的多级漏斗值得移植到其他"寻找未知功能位点"的场景。

## 关键术语表
- **Co-folding model**：同时对蛋白质序列与共存分子（核酸/配体/其他蛋白）做联合折叠的深度学习模型（如AF3、Protenix、ESMFold2），其trunk内部表征蕴含跨分子相互作用信息。
- **PocketWatch**：本文构建的时间切分口袋基准，按UniProt合并PDB中生物相关配体口袋，按最早沉积日期分train/val/test，并叠加结构相似与隐秘性分层。
- **max_qcov**：基于PLINDER-style qcov的结构相似指标，衡量held-out口袋残基被pre-cutoff口袋覆盖的最大比例，用于量化"泛化 vs 记忆"。
- **Cryptic pocket**：在apo（无配体）实验结构中未呈现为开放构象的口袋；本文同时记录predicted fold与holo口袋的RMSD以交叉分析。
- **Pair representation $z$**：共折叠trunk输出的$L \times L \times c_z$三元组张量，编码残基对间的几何/接触关系，是zPocket的主要特征源。
- **DCC$_{N+2}$**：Top-$N{+}2$ recall的centroid度量，$N$为蛋白真实口袋数，预测centroid距真centroid≤4 Å即视为命中。
- **Triangular pair block**：AlphaFold trunk中的残基对混合模块，通过边角更新规则让每个残基对考虑几何邻域中非直接接触的三方关系。
- **Weighted mean-shift decoder**：以阈值以上残基的概率为权重，在其Cα坐标上做密度峰寻找，每个峰对应一个预测口袋centroid。

## 可复现要素
- **数据集**：PocketWatch已开源（HuggingFace `genesisml/PocketWatch`）。
- **代码**：`https://github.com/genesistherapeutics/zpocket`（MIT许可）。
- **权重**：zPocket head权重开源（HuggingFace `genesisml/zPocket`）；主文使用的ESMFold2 trunk开源；Protenix-v2 trunk在实验期间开放，现已转专有许可（仅附录用）。
- **关键超参**：Adam LR=$2\times10^{-3}$，batch累积8蛋白/步，20 epoch（single head 40 epoch），$\tau=0.3$、mean-shift带宽14 Å、最小cluster 3残基，focal $w=1, \gamma=0$；trunk推理recycle=10（Protenix）/ num_loops=10（ESMFold2）。
