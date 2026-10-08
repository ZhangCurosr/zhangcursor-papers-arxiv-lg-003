---
title: "Using-Small-Language-Models-to-Reverse-Engineer-Machine-Lear"
source: https://arxiv.org/pdf/2610.10261v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:14:19"
---

# 论文速读：Using-Small-Language-Models-to-Reverse-Engineer-Machine-Learning-Pipelines-Structures-Potential-and-Limits

## 一句话总结
本文系统性评估了小型语言模型（SLMs）在从Jupyter Notebook代码中提取ML Pipeline阶段结构的潜力，发现SLMs虽取得良好分类性能（DS-Pipelines数据集加权F1=0.912），但未超越现有基线分类器，且分类方法的选择会显著影响对数据科学家实践的理解。同时揭示了推理成本高和分类体系措辞敏感性的局限。

## 研究问题与动机
1. **核心问题**：ML Pipeline结构高度多样且快速演化，如何准确从源代码中自动提取pipeline阶段结构？
2. **现有方法局限-静态映射**：基于预定义函数库映射的方法（如DS-Pipelines）覆盖有限，仅能支持30.66%的代码指令，且需手动维护映射表。
3. **现有方法局限-监督分类器**：现有监督分类器受限于训练数据的范围（如CORAL仅支持7个库），泛化能力不足。
4. **动机**：SLMs具备代码理解和分类能力，且资源需求较低，但其在ML Pipeline逆向工程中的潜力尚未被系统性评估。

## 核心贡献（创新点）
1. **首次系统性评估SLMs用于ML Pipeline逆向工程**：通过对比7个开源代码模型，发现SLMs无需微调即可实现良好分类性能（DS-Pipelines加权F1=0.912），这是首次将SLMs系统应用于该任务的研究。
2. **揭示分类体系措辞对SLM性能的关键影响**：通过同义词替换实验，发现171对比较中82.46%存在显著差异，F1分数最大差异达0.108，MCC最大差异达0.154。
3. **建立统一的ML Pipeline分类体系**：整合DS-Pipelines（11阶段）和DASWOW（10阶段）分类体系，经3位专家协商形成包含6个阶段的统一分类体系T_unified，并提供了详细的映射关系。
4. **量化分类方法对ML实践理解的影响**：发现不同分类器导致阶段分布估计的系统性偏差（SLM_best对Data Preparation过度分类+20%，DS-Pipelines低估-35.23%），以及转移概率的重大差异。
5. **提出可持续的研究方法**：遵循《哥本哈根宣言》，使用开源/受限权重的小模型（<8B参数），报告详细的能耗数据（107Wh），为可复现研究提供范例。

## 方法详解
**1. SLMs筛选与选择**
- 候选模型池：基于HumanEval、BigCodeBench、EvalPlus三个基准测试筛选
- 筛选条件：参数量<8B、开放/受限权重、代码任务专用
- 最终选择7个模型：Qwen2.5-Coder-7B-Instruct、IBM Granite 4.0 Tiny、Gemma 3 4B、Phi-3-mini-128K-Instruct、CodeQwen1.5-7B-Chat、Magicoder-S-DS-6.7B、Artigenz-Coder-DS-6.7B
- 最佳模型：Qwen2.5-Coder-7B-Instruct（加权平均F1=0.754，显著优于其他模型，p<0.001）

**2. Prompt设计**
- **结构化输出**：使用JSON Schema约束输出，定义enum类型的"class"字段限定可选阶段
- **代码Prompt**：将分类任务表述为Python函数，阶段定义写入docstring
- **Chain-of-Thought**：将CoT表达为条件分支（if-else结构）
- **In-Context Learning**：提供8个分类示例（针对表现较差的类别如Model Evaluation）
- **位置优化**：将notebook代码置于system prompt作为上下文，提升MCC 0.024

**3. 实验配置**
- temperature=0, top_p=1（限制随机性）
- frequency_penalty=0, presence_penalty=0（防止类别变化）
- max_tokens=15（适应taxonomy头词长度+JSON开销）
- 推理服务器：vLLM v0.10.2，使用Nvidia A100 80GB GPU

**4. 统计检验方法**
- Cochran's Q检验：比较多个SLM之间的性能差异
- McNemar检验：配对比较SLM与基线分类器
- Holm-Bonferroni校正：控制多重比较的Type I错误
- G-test（拟合优度）：比较阶段分布和转移概率
- Cohen's ω：测量效应大小（small=0.10, medium=0.30, large=0.50）

**5. 统一分类体系构建**
- 建立关系y ⊆ T_ds-pipelines × T_daswow，包含所有等价术语对
- 识别连通分量（CC），将相关术语分组
- 定义映射函数w: CC → T_unified
- 专家协商解决分歧（Fleiss' κ = 0.89，几乎完美一致）

**6. 同义词替换实验**
- 每个阶段选取3个数据科学领域同义词（如Data Collection→Data Acquisition/Data Gathering/Load Data）
- 18种taxonomy变体（1个原始+17个变异）
- 在DS-Pipelines数据集上评估性能变化

## 实验与结果
**数据集**：
- DS-Pipelines: 104个notebook，4,975条指令（单标签），原始支持率仅30.66%
- DASWOW: 470个notebook，test split共1,887个cell（多标签），类别高度不平衡（Data Preparation占49.5%，Save Results仅1.2%）

**评估基线**：
- DS-Pipelines: 静态映射规则（基于404个函数/类映射，加权F1未明确报告）
- DASWOW: 随机森林分类器（加权平均F1=0.71，MCC=0.587）

**主要结果**：

1. **RQ2 - 分类体系措辞影响**（Finding 1）：
   - Cochran's Q检验：Q = 4112.01, p < 0.001
   - 171对比较中140对（82.46%）存在显著差异
   - 最大OR = 9.3
   - F1分数最大差异：0.108（Data Modeling→Model Building与Model Deployment→Model Prediction之间）
   - MCC最大差异：0.154

2. **RQ1 - SLM与基线对比**（Finding 2）：
   
   DS-Pipelines数据集（Table 8）：
   - 加权平均F1: 0.912，MCC: 0.838
   - Data Preparation表现最佳（F1=0.94, MCC=0.83）
   - Model Evaluation较弱（F1=0.66, MCC=0.65）
   - 假阳性率14.14%，主要集中在Data Preparation类别
   
   DASWOW数据集（Table 9）：
   - 加权平均F1: 0.748（略优于基线0.729）
   - 但MCC仅0.509（低于基线0.587）
   - 假阳性率64%（Data Preparation），假阴性率47%（Save Results）、36%（Data Modeling）
   
   推理成本：
   - DASWOW测试集：20分钟（批量不变）vs 基线300毫秒
   - 能耗：107Wh（批量不变）或29Wh（批量查询）
   - P99延迟：6.77秒（批量不变）vs 0.41秒（基线）

3. **RQ3 - 分类方法对实践理解的影响**：
   
   阶段分布差异（Figure 10）：
   - SLM_best: 对Data Preparation过度分类（+20% vs ground truth）
   - DS-Pipelines: 对Data Preparation低估（-35.23% vs ground truth）
   - DASWOW相对准确（M = -3.19%）
   
   分布检验（Figure 11，Finding 4）：
   - SLM_best: χ²(5, N=2518) = 179.95, ω = 0.267
   - DASWOW: χ²(5, N=1574) = 144.14, ω = 0.303
   - DS-Pipelines: χ²(5, N=860) = 77.35, ω = 0.300
   
   转移概率检验（Figure 12，Finding 5）：
   - SLM_best: χ²(46, N=1728) = 584.45, ω = 0.582（大效应）
   - DS-Pipelines: χ²(46, N=789)
