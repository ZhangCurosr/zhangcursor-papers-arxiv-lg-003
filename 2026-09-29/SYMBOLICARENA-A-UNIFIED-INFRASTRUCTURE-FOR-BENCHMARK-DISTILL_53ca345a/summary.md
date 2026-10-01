---
title: "SYMBOLICARENA-A-UNIFIED-INFRASTRUCTURE-FOR-BENCHMARK-DISTILL"
source: https://arxiv.org/pdf/2609.35113v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:21:03"
field: "符号回归评估基准"
keywords: ["符号回归", "基准蒸馏", "多轴评估", "紧凑基准", "统一执行协议", "可解释AI"]
innovations: ["将紧凑SR基准构建形式化为带约束的蒸馏优化问题并导出Core50", "提供统一执行协议标准化异构SR算法的跨范式比较", "提出六轴多轴评估体系揭示数值/符号/搜索行为的正交差异"]
benchmarks: ["Core50", "Full Task Set (664)", "Calibration Set (200)", "SRBench", "SRSD", "LLM-SRBench"]
---

# 论文速读：SYMBOLICARENA-A-UNIFIED-INFRASTRUCTURE-FOR-BENCHMARK-DISTILL

## 一句话总结
SymbolicArena 提出了一种统一的基础设施，将符号回归（SR）基准构建形式化为蒸馏问题，从 664 个异构任务中蒸馏出经过验证的 50 任务紧凑基准 Core50，并同时提供标准化执行协议与多轴评估体系（ID/OOD 数值质量、SYM/MIN 符号质量、EFF/STAB 搜索行为），在降低 92.5% 评估成本的同时保持与全量基准的高度一致性。

## 研究问题与动机
- **评估成本与基准有效性的权衡**：现有 SR 基准中，大规模任务池评估成本高昂，而紧凑基准缺乏系统性证据证明其保留了任务多样性和算法可区分性。
- **跨范式比较缺乏统一协议**：SR 算法采用多样化搜索机制与输出表示（演化搜索、神经策略搜索、结构化搜索、LLM 辅助等），现有基准主要强调任务收集和最终性能测量，缺乏一致的执行与评估标准来刻画不同算法行为。
- **紧凑基准构建缺乏方法论**：可靠的紧凑基准需验证的不仅是任务多样性，还包括算法区分度与从大规模任务总体中保持结论的一致性，目前尚缺系统化的构建策略。

## 核心贡献（创新点）
1. **将紧凑 SR 基准构建形式化为蒸馏问题并导出 Core50**：从 664 个可执行任务中选取 50 个任务构成经验证的紧凑基准，通过显式平衡约束保留任务覆盖与算法区分度。
2. **提供统一执行协议以标准化异构 SR 算法**：对 15 种代表性 SR 算法使用公共 Wrapper 运行，统一记录输出与搜索轨迹，支持可复现的跨范式比较。
3. **提出多轴评估（Multi Axis Evaluation）体系**：将指标组织为数值质量（ID/OOD）、符号质量（SYM/MIN）和搜索行为（EFF/STAB）六个正交维度，揭示仅靠最终数值误差无法表征的 SR 性能差异。

## 方法详解

### Benchmark Distillation（基准蒸馏）
- **Full Task Set 构建**：从超 800 个候选任务中选取 664 个通过验证（表达式可解析、变量/目标一致、固定数据划分可用、元数据完整）的任务，记为 $\mathcal{D}_{664}$，包含 $f_i^*, D_i^{\mathrm{train}}, D_i^{\mathrm{ID}}, D_i^{\mathrm{OOD}}, m_i$。
- **Calibration Set 挖掘**：以 PySR 和 LLM-SR 为初始探针，计算每个任务在 ID 和 OOD 上的性能差距 $G_i = (|g_i^{\mathrm{id}}| + |g_i^{\mathrm{ood}}|)/2$，选取高分歧（140 个）、单侧可解（20 个）和中分歧（40 个）共 200 个任务构成 Calibration Set。
- **Probe 选择**：在 12 种候选算法中选取 4 个探针 $\mathcal{A}_4 = \{\mathrm{DSO}, \mathrm{PyOperon}, \mathrm{iMCTS}, \mathrm{uDSR}\}$，覆盖 Neural Policy Search、Evolutionary Search 和 Structured Search 三个主要范式，最大化校准可用性 $H_\alpha(a;\tau)$。
- **Core50 构建目标函数**：
$$J(S) = 0.45\,\mathrm{Coverage}(S) + 0.35\,\mathrm{MeanInfo}(S) + 0.20\,\mathrm{Balance}(S)$$
其中 Coverage 由 Structural Coverage（0.6）和 Response Coverage（0.4）加权；MeanInfo 为任务信息量 $\mathrm{Info}_i = \mathrm{Disc}_i \cdot \mathrm{Stab}_i$ 的均值；Balance 衡量难度分布与失败模式的代表性。采用贪心初始化 + 可行单次交换优化。

### Unified Evaluation Protocol
- 每个算法通过隔离环境（Conda/Docker）中的共享 Wrapper 运行，输入训练集与特征名，ID/OOD 测试集和真实表达式留作评估。
- 每 1 分钟记录一次检查点（180 次），记录原生最优表达式及数值质量，支持 EFF 和 STAB 指标计算。

### Multi Axis Evaluation
- **ID / OOD（数值质量）**：基于 NMSE 的裁剪对数变换 $\phi(x)$，边界 $\ell_{\min}=-12, \ell_{\max}=2$，值域 [0,1]。
- **SYM（符号保真度）**：若符号等价则得 1；否则为 $0.5\cdot(S_{\mathrm{tree}}\cdot F_v\cdot F_o)^{1/3}$，其中 $S_{\mathrm{tree}}$ 为树结构相似度，$F_v, F_o$ 为变量/算子 F1 分数。
- **MIN（表达式极简性）**：$\min(1, C^{\mathrm{ref}}/C^{\mathrm{pred}})$，衡量预测表达式相对于参考表达式的节点数比。
- **EFF（搜索效率）**：每分钟检查点的归一化质量累积，$m^{\mathrm{EFF}} = \frac{1}{T}\sum_{t=1}^{T} \frac{q(t)}{q^*}$，其中 $q^*$ 为当次运行的最大观测质量。
- **STAB（稳定性）**：几何平均 $(NVC)^{1/3}$，结合数值一致性（N）、有效运行比例（V）、结构一致性（C）三个组件。

### Opus 5 辅助符号后处理
- 搜索完成后由 Claude Opus 5 对最终表达式进行化简和等价性判定，判定结果在聚合前冻结，不参与搜索过程。

## 实验与结果

### 数据集
- **Full Task Set**：664 个来自 8 个来源（LLM-SRBench 240, SRSD 232, SRBench 1.0 133, Keijzer 15, Nguyen 12, Korns 12, SRBench 2025 12, Vladislavleva 8）的标准化任务，覆盖有理函数（282）、三角（137）、多项式（124）等类型，分简单/中等/复杂三级。
- **Core50**：50 个蒸馏任务的集合，来源分布为 SRSD 15、LLM-SRBench 16、SRBench 1.0 9、Nguyen 3、Keijzer 2、Korns 2、SRBench 2025 2、Vladislavleva 1。

### 评估基线
- 6 个替代选择器：Uniform random-50、Smoothed family stratified、Metadata diverse-50、Response K-medoids-50、Top information-50、Difficulty balanced-50。
- 15 种 SR 算法：iMCTS、QLattice、JAXSR、PySR、PyOperon、gplearn、SymbolFit、uDSR、DSO、FePySR、RAG-SR、E2ESR、TPSR、DrSR、LLM-SR。
- 3 个构造外验证算法：FePySR、SymbolFit、JAXSR（未参与基准构建）。

### 主要结果
- **保真度验证**：Core50 的聚合分数 MAE 为 0.1388，相比其他 6 个选择器降低 72.6%–86.7%（如 Top information-50 的 MAE 为 1.0339）。
- **排名一致性**：9 个算法在 Full Task Set 与 Core50 间的 Spearman 相关系数 ID 为 0.9833、OOD 为 0.9667；3 个验证算法的 OOD 排序完全保持。
- **诊断分析**：无任何算法在所有六个维度上占优——iMCTS 在数值质量上领先（ID=77.91, OOD=73.35）但 SYM 仅 42.15；FePySR 在 SYM=48.13 和 MIN=89.55 上领先；JAXSR 在 EFF=99.99 和 STAB=97.43 上领先。
- **数值近似与符号恢复的鸿沟**：PySR 在 Nguyen-9 上 NMSE 约 $10^{-15}$ 但仍被判为 not_equivalent（SYM=0.4886）。
- **训练噪声鲁棒性**：iMCTS 在 5% 噪声下仍保持最高 OOD（40.73），DSO 的绝对下降最小（-1.30）。

## 相关工作脉络
1. **SRBench (La Cava et al., 2021)**：252 任务，侧重算法比较，无搜索轨迹记录和无紧凑基准验证；SymbolicArena 在此基础上增加统一执行协议与经验证的紧凑蒸馏基准。
2. **SRBench 2025 (Imai Aldeia et al., 2025)**：24 任务，关注精度和复杂度，有 HP 和 CP 但无 VC 和 ST；本文相比后者任务规模更大且提供跨范式的诊断评估。
3. **SRSD (Matsubara et al., 2024)**：240 任务，聚焦科学方程重发现，无 HP/CP/VC/ST；本文覆盖更广的任务来源并引入紧凑基准保真度验证。
4. **LLM-SRBench (Shojaee et al., 2025b)**：240 任务，专注 LLM 辅助方程发现，无 HP/CP/VC/ST；本文将 LLM-SR 纳入统一评估框架并与传统算法横向对比。
5. **tiny-Benchmarks / Anchor Points / SubLIME**：NLP/ML 领域的紧凑基准选择工作，但 SR 因任务异质性和算法响应多样性未被充分探索；本文填补了这一空白。
6. **Dynabench / Dynaboard / DataPerf**：动态与多维评估的基础性工作，强调基准随模型发展而演化；本文的多轴评估继承了多维评估思想并针对 SR 特有的符号与搜索维度进行了专门设计。

## 局限性与未来方向
- **基准范围与保真度**：Core50 仅在当前任务分布和所代表算法范式内验证了保真度，新增任务类型或搜索范式需重新验证；科学领域和算子族中不在当前 664 任务内的部分仍未覆盖。
- **算法配置**：评估使用官方默认或推荐配置，未包含算法级超参数优化，绝对分数和相对排名可能在专门调优后发生变化。
- **Opus 5 判定的不一致性**：审计发现 3 个 clean、2 个 1% 噪声和 3 个 5% 噪声算法-任务组存在不一致的成对符号判定，限制了 STAB 结构一致性的解释力。
- **未来方向**：可扩展至更多科学领域任务、支持增量式动态更新、探索自动探针选择机制的泛化。

## 研究启发与可借鉴点
1. **蒸馏问题的形式化方法可迁移**：将紧凑基准构建建模为带约束的优化问题（Coverage + MeanInfo + Balance），并通过贪心+单任务交换进行搜索，这一思路可迁移到其他需要精简评测集的领域（如代码推理、数学推理基准选择）。
2. **多轴评估设计的启示**：将评估分解为相互独立的正交维度（数值/符号/搜索行为），并通过实际实验证明维度间的低相关性（如 EFF 与 ID 相关系数为 -0.08），这种方法论可用于设计更全面的模型评估框架。
3. **LLM 辅助符号后处理与搜索隔离**：将 LLM（Opus 5）仅用于后处理判定而非参与搜索过程，且结果在聚合前冻结，这一"评估辅助但不可泄露"的设计原则可借鉴于需要人工/模型辅助判定的评测任务中，防止评估污染。
4. **探针选择的稳定性分析框架**：通过 1000 次校准样本重采样和权重扰动分析验证探针选择的稳健性，并揭示稳定的竞争集合，这一审计方法可为其他基准选择流程提供可复用的验证范式。

## 关键术语表
- **Symbolic Regression (SR)**：从数据中自动发现简洁且可解释的数学表达式，用于科学方程发现与符号建模。
- **Core50**：从 664 个 Full Task Set 中蒸馏出的 50 任务紧凑基准，经验证可保持与全量基准的高度一致性。
- **Full Task Set ($\mathcal{D}_{664}$)**：标准化后的 664 个可执行符号回归任务的集合，包含固定数据划分和真实表达式。
- **Multi Axis Evaluation**：六轴评估体系，涵盖数值质量（ID、OOD）、符号质量（SYM、MIN）和搜索行为（EFF、STAB）。
- **Calibration Set**：200 个用于筛选探针和高信息任务的子集，通过双探针（PySR vs LLM-SR）分歧挖掘。
- **Probe（探针）**：用于评估任务信息量的参考算法，选定为 $\{\mathrm{DSO}, \mathrm{PyOperon}, \mathrm{iMCTS}, \mathrm{uDSR}\}$。
- **NMSE**：Normalized Mean Squared Error，归一化均方误差，用于数值质量的原始度量。
- **OPUS 5 辅助后处理**：使用 Claude Opus 5 对搜索完成的最终表达式进行化简和符号等价判定，结果冻结后参与指标聚合。

## 可复现要素
- **数据集**：664 个任务的 task manifest 和 Core50 成员列表见论文 Table 7；代码仓库 https://github.com/scientific-intelligent-modelling。
- **代码/权重**：论文声明提供 manifests、scripts、configurations 和冻结的符号判定以支持复现（URL 在摘要中给出）。
- **关键超参**：Core50 目标函数权重 $w=(0.45, 0.35, 0.20)$；探针选择参数 $\alpha=0.5, \tau=100$；评估时间预算 180 分钟；随机种子 $\{520, 521, 522\}$；Opus 5 配置 temperature 自适应、extra high effort、max_tokens=65536。
