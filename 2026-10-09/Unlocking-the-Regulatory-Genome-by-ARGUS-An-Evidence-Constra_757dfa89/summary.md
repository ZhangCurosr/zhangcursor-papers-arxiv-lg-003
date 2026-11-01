---
title: "Unlocking-the-Regulatory-Genome-by-ARGUS-An-Evidence-Constra"
source: https://arxiv.org/pdf/2610.12281v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:13:47"
field: "AI for Biology / Genomic Interpretation"
keywords: ["agentic genomics", "noncoding variant interpretation", "LLM hallucination", "evidence-constrained reasoning", "regulatory genomics", "variant effect prediction"]
innovations: ["Deterministic-science + agentic-reasoning architecture that separates biological classification from LLM reasoning", "Hypothesis-directed investigation loop with observation-dependent branching trajectories", "Structured evidence-type tagging preventing computational evidence from being reported as experimental confirmation"]
benchmarks: ["rs6983267 at 8q24 cancer locus (FOXA1, KLF6, RAD21, SP1)", "ADASTRA allele-specific binding data", "AlphaGenome Atlas cross-reference"]
---

# 论文速读：Unlocking-the-Regulatory-Genome-by-ARGUS-An-Evidence-Constrained-Agentic-Framework-for-Interpreting-Single-Nucleotide-Variants

## 一句话总结
ARGUS是一个证据约束的智能体框架，通过严格分离**确定性生物计算**与**LLM推理**，对单核苷酸变异（SNV）的调控功能进行假设导向的调查，避免LLM幻觉并生成可审计、可追溯的解释轨迹。

## 研究问题与动机
- **核心问题**：GWAS发现的疾病相关变异中超过90%位于非编码调控区，如何准确解释其功能机制仍是基因组医学的核心瓶颈。
- **现有预测模型的局限**：序列-based深度学习模型（如DeepVRegulome、AlphaGenome Atlas）已能预测变异影响，但**预测与解释是两个不同问题**；预测仅给出模型输出，未整合组织背景、独立实验证据及对不确定性的诚实评估。
- **LLM直接解释的幻觉风险**：在早期原型中，使用Qwen 2.5 7B解释rs6983267时，LLM将两个远低于结合阈值（ref=0.007, alt=0.0009）的概率之差解读为“结合增益”（log_odds_ratio=3.02），**从统计上不显著的噪声中幻觉出生物学意义**。
- **缺乏系统化的验证框架**：现有agentic AI系统多侧重于自动化分析流程，但未建立**“何时能够拒绝断言”**的机制，容易过度自信地报告未经验证的假设。

## 核心贡献（创新点）
1. **提出“确定性科学 + 智能体推理”的架构原则**：所有生物分类由基于规则的、可审计的代码执行，LLM仅负责选择调查动作或撰写报告，不能覆盖预分类的事实。
   - *与已有工作的本质区别*：不同于让LLM直接解释原始模型输出的做法，ARGUS将分类权完全交给确定性模块，LLM被约束在证据已分类的事实之上。
2. **构建假设导向的调查循环（Hypothesis-directed investigation loop）**：Planner根据当前不确定性选择证据源，Verifier确定性解释每次观察，中间结果动态改变调查路径。
   - *与已有工作的本质区别*：区别于固定优先级的流水线（依次调用所有工具），ARGUS的轨迹是**观测依赖的分支结构**，能在证据充分时提前停止，避免无效查询。
3. **实现四维证据状态的明确区分**：引入supported / contradicted / rescued / abstained等状态，并通过数据字段（如`CAN_CLAIM_EXPERIMENTAL`）强制标记证据来源的性质。
   - *与已有工作的本质区别*：多数agentic系统不区分计算证据与实验证据，ARGUS在数据结构层面 enforcement，防止将 motif 分析或调控元件重叠上报为实验验证。
4. **在真实变异rs6983267上展示四种不同TF的分歧轨迹**：同一Planner对FOXA1（3步rescued）、KLF6（8步abstained）、RAD21（8步abstained）、SP1（8步abstained）生成不同路径，全部基于真实数据（无模拟观察）。
   - *与已有工作的本质区别*：首次在同一框架下展示**相同的推理引擎如何因中间观察不同而产生结构上不同的调查轨迹**，验证了观察驱动决策的有效性。

## 方法详解
ARGUS分为两个阶段：**确定性预测阶段**与**智能体调查阶段**，两者通过LangGraph编排。

### 预测阶段（Prediction Stage）
- **输入路由**：用户查询通过确定性正则匹配分为四种模式（variant / gene / region / TF），无LLM参与。
- **模型库**：选用458个基于DNABERT的TF结合预测模型，依据GTEx组织表达、ENCODE/ChIP-Atlas ChIP-seq峰（193M+ peaks）及疾病相关调控注释筛选候选TF。
- **序列处理**：每个TF模型接收以变异为中心、长度510 bp的序列，分别代入参考等位基因（ref）和替代等位基因（alt）。
- **确定性分类器**：基于0.5的结合阈值输出四种标签之一：
  1. **LOF/GOF**：一等位基因≥0.5，另一等位基因<0.5（跨越阈值）；
  2. **binding strengthened/weakened**：两等位基因均≥0.5但定量偏移；
  3. **binding retained**：两等位基因均≥0.5且变化微小；
  4. **no confident binding**：两等位基因均<0.5。
- **饱和标记**：当两概率均>0.95时标记为**饱和**，提示模型处于动态范围上限，无法区分等位基因差异。该机制从根本上防止了之前Qwen原型中将微小低概率比值误判为结合增益的幻觉。

### 调查阶段（Investigation Stage）：状态机循环
每次分类结果初始化一个`HypothesisState`对象，跟踪TF-变异身份、DVR预测、认识状态（active / supported / contradicted / rescued / abstained / partially_supported）、已收集证据及完整决策轨迹。

调查循环为**状态依赖的周期**，非固定有向无环图，每步形式化如下：
$$
a_t = \pi(S_t, \mathcal{T}), \quad o_t = T_{a_t}(S_t), \quad S_{t+1} = V(S_t, o_t)
$$
其中 $\pi$ 为规划器策略，$T_{a_t}$ 为选定的证据工具，$V$ 为确定性验证器。

**五大模块**：
1. **State**：存储科学事实，不决策下一步动作。
2. **Planner**：基于当前不确定性选择证据源。默认实现为**固定优先级策略**：
   - 第一优先级：ADASTRA（实验性等位基因特异性结合数据）
   - 第二优先级：JASPAR（等位基因特异性motif评分）
   - 第三优先级：ENCODE cCRE（调控元件上下文）
   - 若无可用证据源则返回ABSTAIN。
   - 同时实现了**LLM-mediated planner**（Claude Sonnet 4.6），通过结构化工具使用提出下一步行动及显式理由。
   - 假设优先级按风险排序：饱和预测 > LOF/GOF > 定量偏移 > retained binding > null预测。
3. **Tools**：
   - **ADASTRA**：本地查询批量TSV文件（数千个ChIP-seq实验的等位基因特异性结合事件），返回FDR校正p值、实验数、阅读覆盖度、偏好等位基因及motif一致性。**缺失记录为“证据不可用”而非“无结合”**。
   - **JASPAR**：使用JASPAR CORE脊椎动物PSSM库进行等位基因特异性motif评分；相对分数差绝对值>0.05时与DVR预测方向比较。**该证据为计算性、间接性，不能独立确立或反驳实验性TF结合**。
   - **ENCODE cCRE**：本地tabix查询GRCh38 ENCODE SCREEN候选顺式调控元件；报告变异是否重叠注释的调控元件（启动子样、增强子样、CTCF结合或DNase-only）。**失败查询同样记录为“不可用”**。
   - 458个TF模型中445个有匹配的ADASTRA表格。
4. **Verifier**：确定性解释每个工具的原始输出并更新状态。以ADASTRA为例，实现10分支决策树，处理：无数据（保持active）、显著ASB且模型饱和（contradicts DVR null → **rescued**）、显著ASB与null预测矛盾（→ contradicted）、显著ASB且方向匹配预测（→ supported）、显著ASB且方向不匹配（→ contradicted）、无显著性但有足够功效（→ 中立）、无显著性且功效不足（→ 保持active）等情形。
5. **Stopping rule**：当Planner返回ABSTAIN时决定最终状态：一致间接支持 → partially_supported；混合间接证据（部分支持、部分反驳） → abstained；功效不足的直接证据 → abstained；无决定性证据 → abstained。**只有Verifier能对直接实验证据赋予supported/contradicted/rescued状态**。

### 证据约束报告器（Evidence-Constrained Reporter）
默认配置下，**Reporter是系统中唯一使用LLM的组件**。接收来自完成调查的结构化事实块（包含变异注释、分类结合结果、带判决和来源标记的证据记录、以及`CAN_CLAIM_EXPERIMENTAL`标志）。LLM仅基于预分类事实生成叙述文本，**不能覆盖任何判决、重新分类结合事件或在`CAN_CLAIM_EXPERIMENTAL=False`时声称实验确认**。

## 实验与结果
- **评估数据集**：变异rs6983267（chr8:127401060 G>T, GRCh38），位于8q24癌症风险位点；涉及四个TF：FOXA1、KLF6、RAD21、SP1。
- **评估基线**：
  1. 固定优先级Planner（ADASTRA → JASPAR → cCRE）
  2. LLM-mediated Planner（Claude Sonnet 4.6，每假设最多3次工具调用）
  3. 前沿LLM直接解释（GPT-4o、Claude 4.6 Sonnet、Gemini 2.5 Pro，非受控对比）
- **主要结果数字**：
  - **FOXA1**：DVR预测饱和retained（p_ref=0.995, p_alt=0.995）；ADASTRA返回显著ASB（FDR=0.030, 15个实验, 475 reads）；Verifier判定contradicts DVR null → **rescued**；总步数**3步**。
  - **KLF6**：DVR预测strengthened（p_ref=0.596, p_alt=0.915）；ADASTRA无数据（unavailable）；JASPAR motif相反（Δ = -0.147）；cCRE显示重叠proximal enhancer-like signature with CTCF；混合间接证据 → **abstained**；总步数**8步**。
  - **RAD21**：DVR预测饱和retained（p_ref=0.999, p_alt=0.9995）；ADASTRA返回功效充足但无显著性的ASB（FDR=0.65, 5个实验, 104 reads）；JASPAR无匹配motif；cCRE提供上下文； → **abstained**；总步数**8步**。
  - **SP1**：DVR预测饱和retained（p_ref=p_alt=0.9995）；ADASTRA无数据；JASPAR motif方向未裁决（非定向预测）；cCRE仅提供上下文； → **abstained**；总步数**8步**。
- **最强结果与提升**：FOXA1在**3步**内从饱和retained被**rescued**，展示了框架识别模型假阴性的能力；LLM-mediated Planner在所有五个假设上达到与固定Planner**完全相同的判决**，但仅用**9次工具调用**（固定为13次），节省了4次对cCRE的查询，证明LLM可学会拒绝无法解析主张的证据。
- **AlphaGenome Atlas交叉验证**：rs6983267的AVI分数为0.570（高预测调控影响）；FOXA1的最大|Δ|=0.463（与ADASTRA发现一致），KLF6的Δ=-0.240（与JASPAR motif discordance方向一致）。
- **LLM直接解释的对比**：前沿LLM将8q24/MYC已知生物学与变异特异性TF结合主张混淆，引用实验文献但未验证变异水平解析度，且均未主动承认知识界限或 abstain 从不可验证主张。

## 相关工作脉络
1. **DeepVRegulome (Dutta et al., 2025)**：序列-based DNABERT预测框架，ARGUS沿用其458个TF模型作为上游预测源，但ARGUS解决的是下游解释问题，不竞争预测性能。
2. **AlphaGenome Atlas (Cheng et al., 2026)**：预计算90亿SNV的变异影响预测；ARGUS将其视为外部计算源（未纳入当前调查循环），强调预测≠解释。
3. **Li et al. (2026) – Agentic AI for biomedical research**：综述agentic AI在生物医学研究中的应用；ARGUS属于此类，但针对性解决LLM在生物学推理中的幻觉问题。
4. **Corpas et al. (2026) – Agentic genomics**：提出从流水线自动化到自主验证的演进；ARGUS的具体化实现，引入假说导向调查循环与确定性验证器。
5. **Lin et al. (2025) – Benchmarking LLMs for cancer variant classification**：显示通用LLM易产生不准确分类并过度依赖弱证据；ARGUS通过架构约束直接针对此失效模式。
6. **Zhou & Troyanskaya (2015) – DeepSEA**：早期序列-based非编码变异效应预测模型；ARGUS可集成此类预测作为输入，但提供解释层。
7. **ARGUS相对定位**：不替代预测模型，而是为任何上游预测提供**证据驱动的验证层**；通过数据结构的证据类型区分（experimental=True/False）防止计算证据被误报为实验确认。

## 局限性与未来方向
- **评估规模有限**：仅评估两个位点上的五个TF假设，未构成预测准确性基准或受控的幻觉减少测量。
- **非端到端系统**：DVR输出为预计算，循环未作为自然语言到推断的端到端系统测试。
- **规划器策略静态**：当前固定优先级策略，未实现基于信息增益的自适应规划或学习策略。
- **证据源有限**：仅三个证据源（ADASTRA、JASPAR、ENCODE cCRE）；可扩展至eQTL（GTEx）、染色质可及性（ATAC-seq）、文献证据（PubMed）、独立模型预测（AlphaGenome Atlas CHIP_TF）。
- **模型盲点**：DNABERT在结合概率高时的饱和问题及核心motif位置外变异的敏感性有限；饱和救援路径仅解决最严重情况。
- **组织匹配与实验聚合**：模型训练上下文与证据源的组织匹配、ADASTRA的实验聚合、参考/替代等位基因标准化仍存限制。
- **数据获取问题**：ENCODE cCRE因SCREEN API不可用而使用本地索引BED文件，缺乏API提供的biosample特异性活性注释。
- **未来方向**：自适应规划、基于期望信息增益的形式化停止准则、跨TF依赖的调控假说图、整合TF特异性方向预测作为第四证据源。

## 研究启发与可借鉴点
1. **架构级约束优于提示级指令**：ARGUS证明，通过数据结构和状态机强制执行“什么可以做、什么不可以做”，比依赖LLM遵循“只报告显著发现”等提示更有效，可迁移至其他科学LLM应用（如药物发现、代谢通路分析）。
2. **观测依赖的分支调查循环**：在证据收集阶段引入状态反馈（`S_{t+1} = V(S_t, o_t)`）可使系统动态调整调查路径，这一模式可应用于需要多源证据整合的其他解释任务（如蛋白质变异效应、细胞类型特异性调控）。
3. **证据类型区分的结构化标记**：`experimental=True/False`、`direct_to_claim=True/False`、`CAN_CLAIM_EXPERIMENTAL`等字段设计，可防止计算证据被过度解读，为多模态科学报告器提供通用模板。
4. **饱和标记与假阴性救援**：对模型输出范围顶端的预测添加特殊标记（如saturation flag），并设计专用验证路径（如查询实验数据），可帮助识别并纠正模型盲点，适用于其他基于阈值分类的预测系统。
5. **LLM作为规划器而非分类器**：实验表明LLM Planner在工具选择上与固定策略等价，且可减少无用查询；可将LLM限定在“选择下一步行动”角色，而将“解释观察结果”保留给确定性模块，平衡灵活性与可靠性。

## 关键术语表
- **DVR (Deterministic Variant Interpreter)**：ARGUS中的确定性预测管道，基于458个DNABERT模型输出结合概率并分类为四种标签。
- **ADASTRA**：大型等位基因特异性结合数据库，整合数千个ChIP-seq实验数据，提供FDR校正p值、实验数、阅读覆盖度等。
- **JASPAR**：开放访问的转录因子结合motif数据库，ARGUS用于等位基因特异性motif评分，作为计算性间接证据。
- **ENCODE cCRE**：ENCODE SCREEN候选顺式调控元件注释，提供变异是否位于调控元件（启动子样、增强子样等）的上下文信息。
- **Rescued**：当直接实验证据（如ADASTRA显著ASB）与DVR预测的null结果矛盾时，判定为模型可能的假阴性，赋予更高层级状态。
- **Abstained**：当无决定性证据、混合间接证据或功效不足时，系统拒绝做出肯定或否定断言，保持中立状态。
- **Saturation flag**：当ref和alt预测概率均>0.95时设置的标记，提示模型处于动态范围上限，无法区分等位基因差异。
- **CAN_CLAIM_EXPERIMENTAL**：布尔标志，仅当存在直接实验证据支持假说时为True，用于约束报告器不得将计算证据宣传为实验验证。

## 可复现要素
- **数据集**：公开数据库（ADASTRA v6.1.1、JASPAR 2024 release、ENCODE SCREEN cCRE、GTEx、ChIP-Atlas），评估变异rs6983267（GRCh38坐标chr8:127401060 G>T）。
- **代码**：已开源于 https://github.com/duttaprat/ARGUS。
- **权重**：458个DNABERT-based TF结合模型；AlphaGenome Atlas SDK版本0.9.0（评估时）。
- **关键超参**：结合阈值0.5；饱和阈值0.95；JASPAR motif分数差绝对阈值0.05；ADASTRA FDR校正；LLM Planner预算3次工具调用/假说。
- **硬件/环境**：论文未详细提及；API调用成本约$0.01/假说（按当前定价）。
