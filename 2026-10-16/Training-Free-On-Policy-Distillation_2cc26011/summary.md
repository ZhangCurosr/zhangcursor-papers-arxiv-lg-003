---
title: "Training-Free-On-Policy-Distillation"
source: https://arxiv.org/pdf/2610.22236v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:05:29"
field: "大语言模型高效微调与合并"
keywords: ["on-policy distillation", "model merging", "training-free", "reinforcement learning", "knowledge fusion", "regmean"]
innovations: ["提出训练-free的on-policy蒸馏TF-OPD，仅前向传播即可完成多专家合并", "发现rollout数据筛选（剔除重复循环答案）比求解器设计对硬问题表现影响更大", "证明当专家共享基座时层-wise蒸馏存在闭合最小二乘解，退化为RegMean形式"]
benchmarks: ["AIME25", "MATH-500", "OlympiadBench", "HumanEval+", "MBPP+", "AIME24", "HMMT25"]
---

# 论文速读：Training-Free On-Policy Distillation

## 一句话总结
本文提出训练-free 的 on-policy 蒸馏方法（TF-OPD），仅需前向传播即可将从同一基座模型训练出的多个领域专家（如数学、代码）合并为一个通用模型；融合模型在平均意义上比任一专家更准确，关键在于对rollout数据的筛选而非求解器设计。

## 研究问题与动机
- 强化学习（RL）已从同一基座模型训练出多个领域专家（数学、编程等），但用户希望在一个模型中获得多种能力，需将专家合并为通用模型。
- 现有"训练型"on-policy 蒸馏虽能在学生自生成文本上监督，但需大量滚动、教师推理和梯度更新步骤，成本高昂。
- 现有"无训练"权重算术方法（Task Arithmetic、TIES、DARE）忽略模型实际处理的数据分布，在 RL 专家上常损失专家特有能力；RegMean 虽用输入统计量，但取自监督训练语料，而非推理模型生成的文本。
- 关键问题：当专家共享基座模型时，是否必须通过训练才能合并？rollout 数据中的哪些成分决定最终效果？

## 核心贡献（创新点）
1. **首次证明训练-free 聚合可行**：展示通过仅前向传播即可聚合多个专家能力，融合模型平均精度超过任一专家（Olmo 3 7B 上七个基准实验）。
2. **将层-wise 蒸馏转化为闭合最小二乘问题**：利用 RegMean 的数学结构，以教师模型 rollouts 的二阶矩（Gram 矩阵）代替监督训练，每个线性层一次精确线性求解，无需反向传播。
3. **发现 rollout 筛选比求解器设计更关键**：数学专家 rollouts 中少量陷入重复循环的回答（truncated in repetition loops）携带大量 token，若不剔除则融合模型在 AIME25 上与 Task Arithmetic 持平而非专家水平；剔除后即在相同格式下匹配专家表现。
4. **系统性消融分析**：对比学生/教师 rollout、域权重调节、 mined hard problems 等策略，得出"数据构成决定硬问题表现、数值精度影响显著、student rollouts 无提升"等关键结论。

## 方法详解
**核心思想**：将对多个教师的蒸馏目标限制到单个线性层时，是一个关于权重 W 的二次凸优化问题，解的形式与 RegMean 相同，但输入统计量来自模型自身生成的 rollout。

- **目标函数**：对每个线性层求解
$$W_\star = \arg\min_W \sum_{d=1}^{K} w_d \mathbb{E}_{x \sim \mu_d} \|Wx - W_d x\|_2^2$$
其中 $x$ 是教师 $d$ 在该层接收到的输入，$\mu_d$ 是教师处理自身答案时的输入分布。

- **闭合解**：整理得 $W_\star = W_b + \left(\sum_d \Delta_d \widetilde{G}_d\right)\left(\sum_d \widetilde{G}_d\right)^{-1}$，其中 $\widetilde{G}_d = w_d[\alpha G_d + (1-\alpha)\text{diag}(G_d)]$，$\alpha=0.9$ 为收缩系数，$G_d = \mathbb{E}[xx^\top]$ 为输入 Gram 矩阵。

- **统计量收集**（仅前向传播）：每个专家回答若干训练题（数学 256 题、代码 256 题），经筛选后，专家以 teacher-forced 模式过一遍保留答案，前向 hook 累积每层 $X^\top X$ 并按 token 数归一化得到 $G_d$。

- **层分组**：自动检测同架构层（如 query/key/value、gate/up、attention output、down），共享同一 Gram 矩阵。

- **筛选规则**（Section 3.4）：排除生成截断（未到长度限制即停止）且检测到重复循环（末尾 20,000 字符 DEFLATE 压缩比 < 5%）的回答；数学答案额外要求正确；代码答案要求通过所有测试用例。

- **精度策略**：融合权重存入 float32、评估用 float16（因 bfloat16 会因离散化失真闭合更新量）。

## 实验与结果
- **基座与专家**：Olmo-3-1025-7B（基座）；Olmo-3.1-7B-RL-Zero-Math 和 Olmo-3.1-7B-RL-Zero-Code（数学/代码专家，均由 RL 从零训练）。
- **训练数据**：数学 256 题（DAPO-Math-17k），代码 256 题（DeepCoder 训练集）；筛选后保留 175 个正确数学答案、92 个通过测试的代码答案。
- **评测基准**（7 个）：数学 MATH-500、OlympiadBench、AIME24、AIME25、HMMT25；代码 HumanEval+、MBPP+。
- **主要结果（Table 2）**：

| 模型 | MATH-500 | OlympiadBench | AIME24 | AIME25 | HMMT25 | HumanEval+ | MBPP+ | Mean |
|------|----------|---------------|--------|--------|--------|------------|-------|------|
| Base | 63.4 | 33.4 | 15.8 | 10.0 | 8.3 | 45.7 | 44.2 | 31.6 |
| Math specialist | 91.8 | 68.7 | 50.0 | 43.3 | 30.8 | 71.3 | 62.4 | 59.8 |
| Code specialist | 85.6 | 62.5 | 26.7 | 30.0 | 15.0 | 86.0 | 69.3 | 53.6 |
| **TF-OPD** | **90.8** | **69.4** | **50.0** | 38.3 | 27.5 | **84.1** | **67.5** | **61.1** |

- **最强结果**：TF-OPD 在 7 个基准的平均精度 **61.1%** 最高，超过数学专家（59.8%）和代码专家（53.6%）；在 MATH-500（90.8% vs 91.8%）、OlympiadBench（69.4% > 68.7%）、HumanEval+（84.1% vs 86.0%）、MBPP+（67.5% vs 69.3%）上与对应专家持平或接近。
- **AIME25 关键发现**（Table 5）：使用全部答案的 TF-OPD 仅 35.0%，与 Task Arithmetic（34.2%）相近，低于数学专家（bfloat16: 43.3%，float16: 39.2%）；剔除循环答案后升至 **40.0%**，逼近 float16 格式下的专家表现。
- **计算成本**（Table 4）：两个专家合计约 **10 GPU-hours**（生成 64 题 × 1 GPU-hour，统计收集 40 min，线性求解 25 min），全程无梯度计算。

## 相关工作脉络
1. **On-policy 蒸馏**（Ross et al., 2011; Agarwal et al., 2024 ICLR; Gu et al., 2024 ICLR）：监督学生模型在其自生成文本上的行为，需多轮训练；TF-OPD 保留"匹配教师输出"和"on-policy 统计"两个要素，去除优化循环，得到闭合解。
2. **权重算术/模型合并**（Task Arithmetic: Ilharco et al., 2023 ICLR; TIES-Merging: Yadav et al., 2023 NeurIPS; DARE: Yu et al., 2024 ICML; Model Soups: Wortsman et al., 2022 ICML）：仅操作参数，不感知输入分布；TF-OPD 在共享基座前提下降维到这些方法的特例（当输入分布相同时退化为权重平均），但在输入分布正交时可无损重现各教师。
3. **RegMean**（Jin et al., 2023 ICLR）：引入输入二阶矩的闭合形式权重融合，但统计量取自各任务监督训练数据；TF-OPD 将其移植到 RL 推理场景，统计量来自模型自生成 rollout。
4. **Fisher Merging**（Matena & Raffel, 2022 NeurIPS）：用 Fisher 信息加权参数重要性；TF-OPD 与之竞争的角度不同，后者关注输入分布匹配而非参数重要性。
5. **无监督蒸馏与知识融合**（Zhao et al., 2026; Wan et al., 2024 ICLR）：涉及 self-distillation 和多源融合；TF-OPD 定位为这类工作的"无优化版本"。

## 局限性与未来方向
- **共享基座的假设**：方法依赖专家与融合模型共享基座权重；若教师架构或大小不同则不适用（论文 Section 4.6 明确说明）。
- **rollout 规模限制**：当前实验仅用各 256 题，规模进一步扩大时的收敛性未知。
- **浮点精度敏感**：bfloat16 存储导致 fused update 失真，需 float32/float16，增加了存储和部署成本。
- **重复循环检测启发式**：DEFLATE 压缩比阈值（5%）为经验设置，可能不适配所有生成模式。
- **未验证超过 2 个专家的场景**：论文仅实验数学+代码两个专家；扩展到 3+ 专家时的理论保证和实证效果未探索。

## 研究启发与可借鉴点
1. **数据筛选优先于优化设计**：在蒸馏/模型合并任务中，rollout 数据的"干净度"（剔除退化样本）可能比求解器复杂度更重要；可迁移到任何基于生成数据的知识融合场景。
2. **前向-only 融合作为快速原型工具**：TF-OPD 以 ~10 GPU-hours 完成两个 7B 专家的合并，可作为后续需要精确训练蒸馏的快速 baseline 或消融起点。
3. **层-wise 最小二乘的通用性**：任何共享权重的多专家聚合任务均可套用此框架，只需替换统计量来源（如替换为学生 rollout、混合教师-学生数据）。
4. **数值精度对低秩更新的敏感性**：RL 微调步长极小时，参数更新在 bfloat16 精度下被量化噪声淹没；在研究参数编辑、合并类工作时需关注存储格式选择。
5. **与硬问题挖掘结合的空间**：论文发现 mined hard problems 和 student rollout 均无显著提升，提示未来可在"何种条件下学生自生成数据有用"这一问题上深入——这为设计更高效的数据选择策略留下方向。

## 关键术语表
- **TF-OPD**：Training-Free On-Policy Distillation，本文提出的无需训练的 on-policy 蒸馏方法，通过前向传播收集统计量并以闭合最小二乘解合并专家。
- **On-policy distillation**：在模型自身生成的文本上进行的蒸馏，区别于 off-policy（固定语料）蒸馏。
- **Task Arithmetic**：将各专家相对基座的权重增量直接相加（$W_b + \sum_d \Delta_d$）的模型合并方法。
- **RegMean**：Jin et al. (2023) 提出的基于输入二阶矩（Gram 矩阵）的闭合形式权重融合方法。
- **Repetition loop**：长文本生成中模型进入重复句子的退化状态，产生大量低信息 token，污染统计量。
- **Gram 矩阵**（$G_d$）：输入二阶矩 $\mathbb{E}[xx^\top]$ 的样本估计，在本工作中记录教师处理答案时各层的输入分布。
- **Teacher forcing**：解码时强制使用 Ground Truth 上下文而非模型自回归生成，用于快速收集激活统计量。
- **Residual stream**：Transformer 中每一层的残差连接累积信号，是模型内部表征的关键载体。

## 可复现要素
- **数据集**：DAPO-Math-17k（数学训练集）、DeepCoder 训练集（代码训练集）、EvalPlus（代码评测）、MathArena（AIME24/AIME25/HMMT25）、MATH-500、OlympiadBench、HumanEval+、MBPP+；论文使用开源数据集。
- **代码/权重开源**：Olmo 3 系列模型以 Apache 2.0 许可发布（allenai/Olmo-3-1025-7B、allenai/Olmo-3.1-7B-RL-Zero-Math、allenai/Olmo-3.1-7B-RL-Zero-Code）；论文附录 B 提供 Algorithm 1 伪代码和完整实现细节，但未提及开源代码仓库。
- **关键超参**：收缩系数 $\alpha = 0.9$，域权重 $w_d = 1$（等权重），数学 rollout 256 题/温度 0.7/top-p 0.9/最大 31,744 token，代码 rollout 256 题/温度 1.0/top-p 1.0/最大 32,768 token，循环检测阈值 DEFLATE 压缩比 < 5%（末尾 20,000 字符）。
