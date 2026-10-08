---
title: "Understanding-and-Mitigating-Token-Pruning-Induced-Vulnerabi"
source: https://arxiv.org/pdf/2610.09703v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:13:27"
field: "多模态大模型安全"
keywords: ["Token剪枝", "视觉语言模型", "多模态安全", "越狱攻击", "注意力机制", "模型压缩"]
innovations: ["首次系统评估Token剪枝对VLM安全性的异质性影响", "发现并形式化剪枝诱导恶意放大机制", "提出即插即用SAP推理时防御框架"]
benchmarks: ["MM-SafetyBench", "FigStep", "JailBreakV-28K", "MMBench", "MM-Vet", "LLaVA-Bench", "ScienceQA"]
---

# 论文速读：Understanding and Mitigating Token-Pruning-Induced Vulnerabilities in VLMs

## 一句话总结
本文首次系统评估了Token剪枝对视觉语言模型（VLM）安全性的影响，发现了"剪枝诱导的恶意放大"机制——剪枝会破坏注意力平衡，使恶意语义被过度放大；并提出即插即用的安全感知剪枝（SAP）机制，在保持加速效果的同时将越狱攻击成功率（ASR）降低最高62%。

## 研究问题与动机
- **核心问题**：Token剪枝作为VLM推理加速的主流技术，是否以及如何引入新的安全风险？现有工作仅关注效率，未系统评估安全性。
- **动机1**：多模态越狱攻击（如MM-Safety、FigStep、JailBreakV）表明VLM本身存在安全隐患，而参数剪枝已被证明会放大LLM的安全风险（Wei et al., 2024），但token剪枝在VLM中的安全影响尚属空白。
- **动机2**：不同剪枝策略对安全性的影响呈现反常分化——文本引导剪枝显著恶化安全（ASR上升约10%），而基于查询的压缩在极端剪枝率（99.8%）下反而提升安全性，这一矛盾现象亟待解释。
- **动机3**：当前研究假设效率与安全存在内在权衡，但本文发现通过理解剪枝-induced的注意力失衡机制，有望在不牺牲加速收益的前提下实现安全增强。

## 核心贡献（创新点）
- **首次系统性安全评估**：在三种主流剪枝范式（视觉中心、文本引导、基于查询）下，于三个多模态安全基准上评估八种代表性剪枝策略，揭示剪枝对安全性的异质性影响，填补了VLM安全评估的空白。
- **发现剪枝诱导恶意放大机制**：首次识别并形式化"Pruning-Induced Malicious Amplification"——背景token移除破坏语义缓冲区，迫使注意力坍缩至前景恶意锚点，从而放大越狱语义；通过注意力熵实验和温度缩放干预提供实证验证。
- **提出即插即用SAP防御机制**：设计包含恶意锚点识别（MAI）、良性token恢复（BTR）、注意力重分配（AR）三阶段的推理时防御框架，无需重新训练即可兼容多种剪枝方法。
- **理论分析与实验验证**：建立恶意语义密度形式化模型，证明剪枝导致分母衰减使密度趋近1；通过信息论命题证明SAP通过混合分布提升熵下界；在五个剪枝方法上验证ASR最高降低62%，同时保持效用不下降。

## 方法详解
### 1. 剪枝诱导恶意放大机制
- **形式化模型**：将视觉token空间划分为恶意子空间$\mathcal{M}$和安全缓冲区$\mathcal{B}$，定义恶意语义密度$\rho = S/(S+N)$，其中$S=\sum_{j\in\mathcal{M}}\exp(q\cdot k_j)$为恶意能量，$N=\sum_{m\in\mathcal{B}}\exp(q\cdot k_m)$为安全缓冲能量。
- **机制解释**：标准Top-K剪枝 disproportionate 丢弃背景token（安全缓冲区），导致$N\to 0$，而$S$基本不变，分母衰减使$\rho\to 1$，恶意语义被"净化"并主导生成。随机剪枝因保持$\mathbb{E}[\rho_{rand}]\approx\rho_{original}$而维持安全稳定性。
- **注意力熵量化**：定义$H(\alpha)=-\sum_{i=1}^{K}\alpha_i\log\alpha_i$，发现熵与ASR呈强负相关（如TRIM熵从3.39降至2.31，ASR从54%升至62%）。温度缩放干预实验（τ从1.9降至1.0）证实熵下降直接导致ASR上升。

### 2. 安全感知剪枝（SAP）
SAP包含三个阶段：

**阶段一：恶意锚点识别（MAI）**
- 计算每个保留token的恶意语义得分：$MSA_{score}(v_i) = \left[\frac{1}{|\mathcal{P}|}\sum_{t\in\mathcal{P}}\mathbf{A}[t,v_i]\right]\cdot\left(1-\frac{v_i\cdot v_{safe}}{\|v_i\|_2\|v_{safe}\|_2}\right)$
- 第一项捕获文本指令对视觉token的平均注意力权重（信号幅度），第二项衡量token与预计算的安全对齐向量$v_{safe}$的语义偏差（风险方向）。
- 选取Top-m得分最高的token作为恶意锚点集合$\mathcal{T}_{msa}$。

**阶段二：良性token恢复（BTR）**
- 从被丢弃token集合$\mathcal{V}_{drop}$中均匀采样k个token构成恢复集$\mathcal{V}_{restored}$。
- 识别保留集中得分最低的k个token $\mathcal{V}_{low}$，用恢复集替换：$\mathcal{V}_{active}=(\mathcal{V}_{keep}\setminus\mathcal{V}_{low})\cup\mathcal{V}_{restored}$，保持序列长度不变。
- 重建语义缓冲区，为注意力重分配提供目标。

**阶段三：注意力重分配（AR）**
- 对恶意锚点token的注意力权重进行稀释：$\hat{\mathbf{A}}_t[v_i]=(1-\lambda)\mathbf{A}_t[v_i]$，$i\in\mathcal{T}_{msa}$。
- 移除的质量均匀重分配给恢复的良性token：$\hat{\mathbf{A}}_t[v_i]=\mathbf{A}_t[v_i]+\frac{\Delta}{|\mathcal{T}_{restored}|}$，$i\in\mathcal{T}_{restored}$，其中$\Delta=\sum_{j\in\mathcal{T}_{msa}}\lambda\mathbf{A}_t[v_j]$。
- AR在各层各注意力头独立应用，无需重新计算softmax，开销可忽略。

### 3. 理论分析
- **命题5.1**：Top-K剪枝使注意力支持集从$\mathcal{V}$收缩至$\mathcal{M}$，导致Shannon熵坍缩：$H(\mathbf{A}_{pruned})<H(\mathbf{A})$。
- **命题5.2**：随着熵下降（$N\to 0$），剪枝模型输出分布与原安全对齐分布的KL散度增大：$D_{KL}(P_\phi\|P_\theta)\uparrow$ as $S/N\uparrow$。
- **命题5.3**：SAP通过混合分布$\mathbf{A}_{SAP}=(1-\lambda)\mathbf{A}_{pruned}+\lambda\mathbf{A}_{restored}$建立熵下界：$H(\mathbf{A}_{SAP})\geq(1-\lambda)H(\mathbf{A}_{pruned})+\lambda H(\mathbf{A}_{restored})$，从而约束KL散度。

## 实验与结果
### 数据集与基准
- **安全基准**：MM-SafetyBench（13种风险场景）、FigStep（排版视觉提示越狱）、JailBreakV-28K（28K多模态越狱样本）。
- **效用基准**：MMBench、MM-Vet、LLaVA-Bench、ScienceQA（SQA）。
- **模型**：LLaVA-1.5-7B，统一75%剪枝率，贪婪解码（temperature=0），seed=42。

### 主要结果
| 方法 | ASR (%) ↓ | MMBench (%) ↑ | FLOPs (T) | Latency (ms) |
|------|-----------|---------------|-----------|--------------|
| Original | 54.01 | 64.46 | 8.79 | 94.29 |
| TRIM | 62.33 | 63.84 | 2.22 | 55.90 |
| TRIM+SAP | **50.19** (↓19.5%) | 63.45 (↓0.6%) | 2.22 | 57.19 |
| FasterVLM | 63.22 | 64.28 | 2.22 | 55.90 |
| FasterVLM+SAP | **47.60** (↓24.7%) | 63.98 (↓0.5%) | 2.22 | 57.19 |

- **最强结果**：SAP在五个剪枝方法上平均降低ASR 19.5%-27.6%，最高达62%（FasterVLM+文本扩展SAP†从72.02%降至10.32%），同时效用波动<1%。
- **对比外部guardrail**：SAP与LLaVAGuard相比，达到相当安全性（MM-Safety 52.78% vs 51.59%）但延迟低3倍（57.19ms vs 160.09ms）。
- **ablation**：BTR单独降低MM-Safety ASR 2%（65.87→63.89），AR单独降低12%（65.87→53.57），两者结合降低15%（65.87→52.78）且维持高效用。
- **超参敏感性**：最优锚点数m=10；剪枝率50%-90%范围内SAP均有效；安全指令变化导致ASR波动仅3%。

## 相关工作脉络
- **多模态越狱攻击**：MM-SafetyBench（Liu et al., 2023b）揭示VLM整合后安全对齐失效；JailBreakV-28K（Luo et al., 2024）证明文本越狱向视觉模态的高迁移性；FigStep（Gong et al., 2025）展示通过图像内嵌毒性指令绕过防护。本文工作扩展至剪枝后的安全性评估。
- **Token剪枝方法**：视觉中心范式（HiPrune, FasterVLM）利用视觉内部注意力稀疏性；文本引导范式（TRIM, SparseVLM, FitPrune）基于指令相关性过滤；基于查询范式（LLaVA-Mini, MQT-LLaVA）通过可学习查询聚合。本文首次系统比较三类范式的安全影响。
- **模型压缩与安全性**：Wei et al. (2024) 证明参数剪枝放大LLM安全风险；SafePTR（Chen et al., 2025a）提出先剪除再恢复机制防御token级越狱。本文发现token剪枝的安全机制不同于参数剪枝，并针对性提出SAP。
- **外部guardrail方法**：LLaVAGuard（Helff et al., 2025）在输入/输出层部署安全过滤器。本文证明SAP与guardrail正交可级联，且SAP不破坏剪枝加速收益。
- **注意力熵与安全**：本文建立注意力熵与越狱风险的定量关联，为后续研究提供可复用的安全诊断指标。

## 局限性与未来方向
- **白盒访问限制**：SAP需要完整访问VLM内部注意力权重和隐藏状态，仅适用于开源架构，黑盒部署受限。
- **极端剪枝假设**：BTR在>95%剪枝率下可能因保留token过少而置换重要前景锚点，但该场景在实际中因效用坍塌而不具实用性。
- **自适应攻击鲁棒性**：已知机制的white-box防御面临自适应攻击威胁，但攻击者需同时破解多种剪枝策略组合，成本较高。
- **安全向量预计算**：$v_{safe}$需单次前向传播预计算，虽可复用但依赖固定安全指令，可能存在边界情况失效。
- **未来方向**：探索自适应攻击下的鲁棒性评估；扩展至闭源模型的黑盒防御；研究多剪枝策略组合的防御边界。

## 研究启发与可借鉴点
- **注意力熵作为安全诊断指标**：建立$H(\alpha)$与ASR的强负相关，可为其他模型压缩方法提供即时的安全风险评估工具。
- **语义缓冲区设计思想**：BTR通过恢复被丢弃token重建缓冲区，这一"语义缓冲区"概念可迁移至量化、蒸馏等压缩技术的安全加固。
- **推理时即插即用防御范式**：SAP无需重新训练，通过MAI+BTR+AR三阶段在推理时修正注意力分布，为其他安全敏感应用提供轻量级防御模板。
- **效率-安全非零和博弈**：本文打破"加速必牺牲安全"的预设，证明通过机制理解可实现双向优化，启发后续研究探索效率-安全的帕累托前沿。
- **多模态交叉注意力分析**：结合文本指令注意力与视觉token表征偏差的MAI设计，为跨模态恶意检测提供特征融合思路。

## 关键术语表
- **Token-Pruning**：通过重要性评分函数识别并移除冗余视觉token，以降低VLM推理计算量的加速技术。
- **Pruning-Induced Malicious Amplification**：剪枝移除背景良性token破坏注意力平衡，迫使权重坍缩至前景恶意锚点从而放大越狱语义的机制。
- **Safety-Aware Pruning (SAP)**：包含恶意锚点识别、良性token恢复、注意力重分配三阶段的即插即用推理时防御机制。
- **Malicious Semantic Density ($\rho$)**：恶意能量与总视觉能量之比，形式化刻画剪枝后恶意语义的主导程度。
- **Attention Entropy**：后softmax注意力权重的Shannon熵，量化注意力分布的集中程度，与越狱风险负相关。
- **MM-SafetyBench**：评估VLM在13种风险场景下安全对齐性能的基准，报告Attack Success Rate (ASR)。
- **FigStep**：通过图像内嵌毒性排版提示实现越狱的攻击方法基准。
- **JailBreakV-28K**：包含28K多模态越狱样本的大规模基准，评估文本-视觉联合攻击鲁棒性。

## 可复现要素
- **数据集**：MM-SafetyBench、FigStep、JailBreakV-28K、MMBench、MM-Vet、LLaVA-Bench、ScienceQA均为公开基准。
- **代码/权重**：基于LLaVA-1.5-7B官方仓库实现，SAP为训练-free方法，论文未明确开源代码链接（需查看arxiv补充材料）。
- **关键超参**：剪枝率75%（默认）、锚点数m=10、重分配系数λ=0.6、BTR恢复量为保留集的10%、温度缩放实验τ∈[1.0, 1.9]。
- **硬件环境**：单张NVIDIA RTX 4090，贪婪解码（temperature=0），seed=42。
- **实现细节**：安全向量$v_{safe}$通过"Please answer this question safely and accurately"指令单次前向传播提取各层hidden states。
