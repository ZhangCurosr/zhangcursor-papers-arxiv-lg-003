---
title: "Unlocking-the-Regulatory-Genome-by-ARGUS-An-Evidence-Constra"
source: https://arxiv.org/pdf/2610.12281v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:02:45"
field: "计算基因组学/非编码变异功能解释"
keywords: ["noncoding variant interpretation", "agentic genomics", "allele-specific binding", "hallucination reduction", "evidence-constrained reasoning", "DNABERT", "ADASTRA", "transcription factor binding"]
innovations: ["确定性分类与LLM推理的架构级分离防止生物学幻觉", "假设驱动的observation-dependent调查循环实现同planner divergent trajectories", "三层证据源优先级与CAN_CLAIM_EXPERIMENTAL flag区分实验与计算证据"]
benchmarks: ["rs6983267 8q24 locus", "AlphaGenome Atlas AVI cross-reference", "Frontier LLM qualitative baseline (GPT-4o/Claude 4.6/Gemini 2.5 Pro)"]
---

# 论文速读：Unlocking-the-Regulatory-Genome-by-ARGUS-An-Evidence-Constra

## 一句话总结
论文提出 ARGUS（Agentic Regulatory Genomics for an Uncertainty-aware Scientist）框架，通过严格分离确定性生物学计算与 LLM 推理，解决非编码调控区单核苷酸变异（SNV）功能解释中大模型幻觉问题，实现可审计、可追溯的假设驱动调查。

## 研究问题与动机
- **核心瓶颈**：GWAS 中发现的 >90% 疾病相关变异位于非编码调控区域，其功能解释仍为基因组医学开放难题，现有工具擅长预测效应却缺乏独立证据验证。
- **LLM 幻觉风险**：大语言模型在解释 TF 结合变化时容易编造实验支持、将统计上不显著的数值比（如两个极低概率的对数比值）误判为生物学显著事件。
- **预测≠解释**：AlphaGenome Atlas 等预计算了 9B 种 SNV 效应，但从未区分"a model predicts an effect"与"independent evidence supports this effect"。
- **缺乏不确定性表达**：现有 agentic 系统倾向于给出自信结论，不会在证据不足时主动 abstain。

## 核心贡献（创新点）
1. **提出"确定性科学、代理式推理"设计原则**：所有生物学分类由规则代码完成，LLM 仅负责规划行动或撰写报告，无法覆写预分类事实。
2. **构建假设驱动的调查循环（observation-dependent branching）**：Planner 根据当前 epistemic status 动态选择下一个证据源，实现了相同 planner 对不同中间观测产生 divergent trajectories（如 FOXA1 3 步 vs KLF6 8 步）。
3. **三层证据源与确定性验证器（verifier）**：ADASTRA（实验级 ASB）、JASPAR（motif 计算）、ENCODE cCRE（调控上下文）各有明确可信度标签，verifier 实现 10 分支决策树处理 ADASTRA 结果。
4. **展示饱和预测的 rescue 路径**：同一位点 rs6983267 上，FOXA1 因真实 ADASTRA 数据（FDR=0.030）推翻模型饱和保留预测而被 rescued，而 SP1 因无直接证据仅 abstain，证明框架能识别假阴性。
5. **LLM planner 优于固定优先级策略**：在保持相同最终结论前提下，LLM-mediated planner 仅需 9 次 tool calls 而非固定策略的 13 次，主动放弃无法解析声明的 cCRE 查询。

## 方法详解
- **Prediction Stage（预测阶段）**：用户查询经确定性 regex 路由至四种模式（variant/gene/region/TF）。ARGUS 从 458 个 DNABERT-based TF binding 模型库中选取候选 TF，对每个变异输入 510-bp 序列（ref/alt 替换），输出 $p_{\text{ref}}$ 与 $p_{\text{alt}}$。确定性分类器以 0.5 为阈值划分四标签：LOF/GOF（跨过阈值）、strengthened/weakened（定量偏移）、binding retained（双等位均≥0.5 且变化小）、no confident binding（双<0.5）；同时标记 saturation（双>0.95），防止 SP1 类幻觉。
- **Investigation Stage（代理调查阶段）**：状态方程为 $a_t = \pi(S_t, \mathcal{T})$, $o_t = T_{a_t}(S_t)$, $S_{t+1} = V(S_t, o_t)$，其中 $\pi$ 为 planner 策略，$T$ 为工具，$V$ 为确定性验证器。五模块严格分离职责：
  - **State**：存储科学事实（预测、证据记录、epistemic status、轨迹事件），不做决策。
  - **Planner**：优先级策略 ADASTRA→JASPAR→ENCODE cCRE，按风险排序假设（saturation > LOF/GOF > quantitative shift > retained > null）。可选 LLM-mediated planner（Claude Sonnet 4.6，结构化 tool use 输出 rationale）。
  - **Tools**：ADASTRA（Mabel v6.1.1，本地 TSV，返回 FDR、experiment count、read coverage、preferred allele）、JASPAR 2024（motif PSSM，$\Delta$ 阈值 0.05）、ENCODE cCRE（tabix 查询 GRCh38，报告重叠的调控元件类型）。
  - **Verifier**：ADASTRA 10 分支决策（无数据→active、显著 ASB+饱和模型→rescued、显著 ASB 与预测方向一致→supported、显著 ASB 相反→contradicted、非显著但充分 power→neutral、非显著 power 不足→active）；JASPAR 标记 computational/indirect；cCRE 仅提供 context。
  - **Stopping Rule**：综合间接证据一致→partially_supported；混合间接证据或 power 不足的直接证据→abstained；仅 verifier 可对直接实验证据赋予 supported/contradicted/rescued。
- **Evidence-Constrained Reporter**：默认配置下唯一使用 LLM 的组件，接收预分类 facts block（含 CAN_CLAIM_EXPERIMENTAL flag），只能叙事化输出而不能覆写 verdict。

## 实验与结果
- **数据集**：rs6983267（chr8:127401060 G>T，GRCh38）8q24 癌症风险位点；四个 TF 假设（FOXA1、KLF6、RAD21、SP1）。
- **评估基线**：固定优先级 planner vs LLM-mediated planner（Claude Sonnet 4.6，budget=3）；Frontier LLM 基线（GPT-4o、Claude 4.6 Sonnet、Gemini 2.5 Pro）定性对比；AlphaGenome Atlas AVI score 交叉验证。
- **主要结果**：
  - FOXA1 3 步即 resolved：ADASTRA 返回显著 ASB（FDR=0.030，15 experiments，475 reads），verifier 赋 rescued。
  - KLF6 8 步后 abstain：ADASTRA unavailable → JASPAR 方向相反（$\Delta=-0.147$）→ cCRE 支持（pELS+CTCF）；混合间接证据触发 abstain。
  - RAD21 8 步后 abstain：ADASTRA 非显著（FDR=0.65，5 experiments，104 reads）但 power 充足 → neutral；JASPAR 无匹配 motif；cCRE 仅提供 context。
  - SP1 8 步后 abstain：ADASTRA unavailable；JASPAR 虽找到 motif 但因 non-directional arm 不 adjudicate；cCRE context only。
  - Frontier LLM 产生自信但混淆已知 8q24/MYC 生物学与位点特异性 TF 结合声明，未区分预测与证据，未 abstain。
  - AlphaGenome Atlas AVI=0.570，FOXA1 max $|\Delta|=0.463$、KLF6 $\Delta=-0.240$ 方向与 ARGUS 一致，但因其为 variant-level 统一分数不参与 planner 分支。
  - LLM planner 与固定 planner 达成相同 5/5 verdicts；LLM planner 用 9 次 vs 固定 13 次 tool calls（节省 cCRE 对 abstaining 假设的无效查询）；每次 LLM planner call 约 1,200 input + 250 output tokens（~$0.01/hypothesis）。
- **最强结果**：FOXA1 的 rescued 路径证明了框架能识别 DNABERT 饱和假阴性，并通过真实实验数据纠正预测。

## 相关工作脉络
1. **DeepVRegulome / DNABERT-based models**（Zhou & Troyanskaya 2015; Dutta et al. 2025）：上游预测引擎，ARGUS 在其输出之上增加证据约束解释层，两者正交互补。
2. **AlphaGenome Atlas**（Cheng et al. 2026）：预计算 9B SNV 效应，ARGUS 强调"预测≠解释"并引入独立实验证据验证，AVI 分数因 variant-level 统一性无法提供 observation-dependent branching。
3. **Agentic genomics 系统**（Li et al. 2026; Corpas et al. 2026）：同属代理式生物研究框架，但 ARGUS 将"生物分类由确定性代码执行"作为架构硬约束，而非 prompt-level 指令。
4. **ADASTRA 数据库**（Abramov et al. 2021）：提供数千 ChIP-seq 实验的等位基因特异性结合 TSV，ARGUS 将其作为首选实验证据源（experimental=True, direct_to_claim=True）。
5. **JASPAR motif 数据库**（Rauluseviciute et al. 2024）：motif scoring 作为间接计算证据（experimental=False），ARGUS 明确其不能独立确立或反驳实验 TF 结合。
6. **ENCODE SCREEN cCRE**（ENCODE Project Consortium 2020）：提供调控元件上下文，ARGUS 将其定位为辅助 context 证据，不能解析 TF-specific binding 方向。

## 局限性与未来方向
- 评估仅覆盖 5 个 TF 假设、2 个 locus，非预测准确率基准测试，也未控制衡量幻觉减少幅度。
- 当前 planner 为固定证据优先级，未引入基于预期信息增益（expected information gain）或 value-of-information 的自适应策略。
- 证据源仅限三种，可扩展至 eQTL（GTEx）、染色质开放性（ATAC-seq）、文献证据（PubMed）、独立模型预测（AlphaGenome Atlas CHIP_TF）。
- DNABERT 模型的盲点（饱和、motif 核心位置外敏感性低）虽通过 rescue 路径缓解，但系统性 failure mode 表征仍在进行中。
- 生物解释受限于组织匹配（模型训练 context vs 证据来源）、ADASTRA assay 聚合、参考/ alternate 等位基因标准化等问题。
- ENCODE cCRE 查询因 API 不可达而使用本地 BED 索引，缺少 biosample-specific 活性注释。

## 研究启发与可借鉴点
1. **"确定性分类+代理推理"架构范式**：可将生物分类硬约束推广至其他高幻觉风险领域（如药物相互作用预测、临床指南推理），避免 LLM 将统计噪声误报为显著发现。
2. **observation-dependent branching 设计**：停止条件由中间观测决定而非固定深度，此模式可迁移至需要动态调整实验方案或数据检索策略的科研代理系统。
3. **evidence hierarchy 与 CAN_CLAIM_EXPERIMENTAL flag**：通过数据结构层强制区分 experimental/computational 与 direct/indirect 证据，防止将 motif 一致性包装为实验验证——此 flag 设计可直接复用于多模态科学报告生成。
4. **饱和预测的 rescue 路径**：识别模型动态范围天花板并通过独立实验数据纠正假阴性，可启发对任何 deep learning 预测器"high-confidence wrong"情形的系统检测策略。
5. **LLM planner 与固定策略的比较实验**：论证 LLM 在避免无效查询上的效率优势（9 vs 13 calls），为后续研究中规划层选型提供实证依据。

## 关键术语表
- **DVR (Deterministic Variant Reader)**：确定性变异读取器，基于 DNABERT 输出 ref/alt 结合概率并赋予四类标签（LOF/GOF/strengthened/weakened/retained/no confident binding）及饱和标记。
- **ASB (Allele-Specific Binding)**：等位基因特异性结合，ADASTRA 数据库核心测量，提供 FDR-corrected p-value、experiment count、read coverage、preferred allele。
- **FDR (False Discovery Rate)**：错误发现率，ADASTRA 多检验校正指标，文中显著阈值对应 FDR=0.030。
- **cCRE (candidate Cis-Regulatory Element)**：候选顺式调控元件，来自 ENCODE SCREEN 注释，分为 promoter-like、enhancer-like（含 pELS）、CTCF-bound、DNase-only 类型。
- **Rescued**：当直接实验证据（如 ADASTRA 显著 ASB）与 DVR 饱和保留预测矛盾时，标记为模型可能的假阴性而非预测正确。
- **Abstained**：系统在混合间接证据、power 不足的直接证据或无 resolving evidence 时主动拒绝做出 supported/contradicted 声明。
- **LOF/GOF (Loss/Gain of Function)**：功能丧失或获得，指变异使结合概率跨过 0.5 阈值（一 allele≥0.5、另一<0.5）。
- **CAN_CLAIM_EXPERIMENTAL**：证据源元数据 flag，仅当存在 direct experimental evidence 支持假设时为 True，约束 reporter 不得将计算一致性包装为实验确认。

## 可复现要素
- **数据集及公开性**：ADASTRA（Mabel v6.1.1 本地 TSV）、JASPAR 2024 release、ENCODE cCRE（本地 tabix 索引 BED）、GTEx 组织表达、ENCODE/ChIP-Atlas 193M+ ChIP-seq peaks；论文未声明独立数据集归档，依赖本地索引文件。
- **代码/权重开源**：代码仓库 https://github.com/duttaprat/ARGUS；458 个 DNABERT-based TF binding 模型为预计算输出，论文未提供单独模型权重下载链接。
- **关键超参**：active-binding 阈值 0.5、saturation 阈值 0.95、JASPAR motif $\Delta$ 绝对阈值 0.05、ADASTRA FDR 显著性（文中示例 0.030）、LLM planner budget=3 tool calls/hypothesis。
- **运行环境**：LangGraph（LangChain 2024）、Claude Sonnet 4.6（Anthropic API credits）、SDK 0.9.0 for AlphaGenome Atlas。
