---
title: "YANchor-4B-Effective-Long-Horizon-Reasoning-in-O-N-Time-with"
source: https://arxiv.org/pdf/2610.10118v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:16:18"
field: "高效长上下文语言模型"
keywords: ["循环语言模型", "长程推理", "O(1)内存", "显式记忆机制", "RWKV", "数学推理", "线性时间生成"]
innovations: ["独立记忆锚点机制实现O(N)时间O(1)内存的长程推理", "读写分离投影与优先级保留策略提升记忆管理能力", "四阶段训练(M0-R-S-RLVR)系统化发展循环模型的显式记忆能力"]
benchmarks: ["AIME 2024-2026", "HMMT Feb 2026", "MATH-500", "MMLU-Pro", "HumanEval", "LiveCodeBench v6", "DROP", "MuSR", "MMBench", "POPE", "MME"]
---

# 论文速读：YANchor-4B-Effective-Long-Horizon-Reasoning-in-O-N-Time-with

## 一句话总结
YANchor-4B 是一种通用型循环架构语言模型，通过独立的"记忆锚点"机制在 O(N) 生成时间与 O(1) 固定内存下保留关键历史信息，实现了高效的长程推理能力，在 AIME 数学竞赛等任务上显著超越更大参数的线性时间-恒定状态基线模型。

## 研究问题与动机
- **长程推理的信息依赖问题**：复杂推理（如数学竞赛解题）需在生成过程中持续访问早期定义、约束和中间结果，而完整历史注意力（Full-history attention）的 KV cache 随序列增长，导致计算和存储成本不可控。
- **循环模型精度退化问题**：现有循环/状态空间模型（如 RWKV、Mamba）用固定维度状态压缩历史，重复更新会导致精确信息的丢失，难以保留可供检索的细粒度关联。
- **局部注意力的视野限制**：局部窗口注意力仅保留近期细节，远距离信息随窗口滑动而遗忘，无法支撑跨长段的依赖推理。
- **恒定状态与能力的矛盾**：如何在固定容量历史状态下同时兼顾高效推理和高质量长程推理，是当前线性时间模型面临的核心难题。

## 核心贡献（创新点）
1. **独立记忆锚点机制（ANchor）**：提出与循环状态和局部注意力正交的独立读写记忆模块，将关键历史信息以锚点形式持久化，区别于仅依赖状态压缩或局部窗口的现有方案。
2. **O(N) 时间 + O(1) 内存的三重历史记录架构**：组合 Gated DeltaNet（循环压缩）、SWA（1024 局部窗口）与独立 Memory，三种路径在固定容量下互补，实现可证明的线性生成时间。
3. **优先级保留策略（Priority-based retention）**：引入自适应 admission network 对 KV 记录评分，每 256-token 块独立选 Top-K 保留，保证高价值信息在离开局部窗口后仍可被检索。
4. **四阶段训练流程（M0→R→S→RLVR）**：从记忆模块初始化到联合适配、监督微调再到可验证强化学习（RLVR），系统性地发展模型的写入、检索和使用记忆的能力。
5. **在 4B 参数规模下超越 13B 基线**：AIME 2024–2026 均值 pass@1 达 82.93%，对比 RWKV-7 G1j 13.3B 的 18.89% 实现大幅领先，同时保持数倍于 Transformer 基线的批处理吞吐。

## 方法详解
**架构概述**：以 Qwen3.5-4B 为骨干，32 层分为 8 组，每组 3 个 Gated DeltaNet (GDN) 层 + 1 个结合局部注意力与独立记忆的层：
$$[\text{GDN, GDN, GDN, SWA}_{1024} \parallel \text{Memory}] \times 8$$
每层输出经 MLP 子层后送入残差流。

**历史表示三路合一**：
- GDN 层：固定维度循环矩阵压缩全局上下文
- SWA（Sliding Window Attention）：暴露最近 1024 个 token 的精确细节
- Memory：独立保留可寻址的远处记录

**独立写入与准入（Independent Writing & Admission）**：
- 写入路径：$w_t = u_t + W_{\text{down}}[\text{SiLU}(W_{\text{gate}}u_t) \odot W_{\text{up}}u_t]$，生成待存储表示
- 查询从 $u_t$ 独立投影，写/读投影分离，使写入侧重内容塑造、读取侧重当前检索需求
- Admission network：$a_t = W_{a,2}\text{SiLU}(W_{a,1}u_t)$，每个 KV 组独立评分
- 每 256-token 块边界，各组从候选中合并并选择 Top-1024 记录持久化

**分组查询检索与整合（Grouped-query Retrieval）**：
- 每 Memory 模块：16 查询头 / 4 KV 组 / head dim=256
- 检索公式：$e_{t,h} = \sum_{j \in \mathcal{M}_{g(h)}} \frac{\exp(q_{t,h}^\top k_{j,g(h)} / \sqrt{d})}{1 + \sum_{i \in \mathcal{M}_{g(h)}} \exp(q_{t,h}^\top k_{i,g(h)} / \sqrt{d})} v_{j,g(h)}$
- 分母含单位项（零 logit、零 value），空记忆时输出中性，并允许注意力远离低匹配记录
- 查询条件门控 $g_{t,h}$ 调制各头输出后再经独立投影融合进残差流

**线性时间复杂度证明**：每 token 更新固定大小循环状态，读最多 W 个局部位置和 K 个记忆记录，块提交从 K+C 记录中选，总自回归工作量 $\sum_{t=1}^{N} O(1 + W + K + \frac{(K+C)\log(K+C)}{C}) = O(N)$，历史状态恒为 O(1)。

## 实验与结果
**数学推理（核心亮点）**：
| 模型 | AIME '24 | AIME '25 | AIME '26 | HMMT '26 | MATH-500 | GSM8K |
|------|----------|----------|----------|----------|---------|-------|
| YANchor-4B | 86.72 | 77.45 | 84.64 | 63.64 | 97.49 | 94.45 |
| RWKV-7 G1j 13.3B | 17.50 | 27.92 | 11.25 | 12.12 | 77.00 | 94.30 |

- AIME 2024–2026 三年均值 pass@1：**82.93%**（vs. 最强基线 18.89%）
- HMMT Feb 2026：**63.64%**（vs. 12.12%）
- MATH-500：**97.49%**，接近 Qwen3.5-4B 的 98.00%
- 平均推理长度约 18,000–25,000 token（AIME），HMMT 约 30,189 token

**通用能力（24 benchmark 均值）**：
- YANchor-4B：**78.64**，领先 16 个恒定状态基线最强（RWKV-7 G1j 13.3B: 56.29）22.35 分
- 在 20/24 个任务上领先所有bounded-state基线
- MMLU-Pro：79.60，超越 MiniCPM4.1-8B（65.90）、Gemma4-E4B（68.60）等更大模型
- HumanEval：96.72，MBPP：87.81

**推理效率（H100 80GB 单卡）**：
- 长序列解码稳定约 **213 tokens/s**（128K 输入 + 128K 输出）
- 批量吞吐：比 Qwen3.5-4B 高 **4.12–6.58×**，比 Gemma4 E4B 高 **3.25–5.66×**
- Batch=512 时纯解码吞吐 **12,854 tokens/s**
- 固定历史缓存 **48.90 GiB**（不随上下文增长），可同时驻留 **512 条独立长序列**

**VL 能力**：六项基准均值 **84.94%**（MMBench EN: 88.28, MMBench CN: 89.00, POPE: 85.43, MME: 82.98, ChartQA: 83.92, AI2D: 80.05）

**长程记忆保留**：远距离信息检索准确率 **94.14%**；关闭 Memory 分支后降至 **0.39%**，验证其必要性。

## 相关工作脉络
1. **RWKV-7 / Mamba / xLSTM**：固定维度循环状态的经典方案，YANchor 在此基础上引入独立可寻址记忆，显著提升了复杂推理能力。
2. **AHN-GDN**：将循环压缩与局部注意力结合，但未引入独立记忆锚点；YANchor 通过优先级保留和分组查询检索实现了更精细的信息管理。
3. **RADLADS / ARWKV**：将预训练 Transformer 能力蒸馏/转换到循环架构；YANchor 采用架构适配（替换注意力层）+ 四阶段训练的策略，而非直接转换。
4. **HOLA**：探索有界精确存储以弥补循环状态遗忘；YANchor 的 Memory 机制在写入/读取分离、分组查询和优先级保留方面更为系统化。
5. **PromptCoT-Mamba**：研究循环模型中的推理；YANchor 进一步证明了在恒定状态约束下可达成接近甚至超越全注意力模型的大规模数学推理能力。
6. **超二次架构理论（Hartl et al.）**：将次二次架构能力与状态跟踪/记忆动力学联系；YANchor 的工作为该理论提供了实证支持。

## 局限性与未来方向
- **变量绑定任务仍较弱**：在跨多关联的信息整合任务上正确率仅 53.13%，是长程记忆中最难诊断的类别。
- **参数规模偏小**：4B 参数在部分指令遵循（IFEval 88.03 vs. Qwen3.5-4B 90.02）和代码生成（LCB v6 61.63 vs. Qwen3.5-4B 76.20）上落后于同等参数量的全局注意力模型。
- **VL 能力非重点优化**：视觉编码器直接沿用 Qwen3.5-4B，未针对多模态长程记忆进行专门优化。
- **四阶段训练成本高**：M0→R→S→RLVR 的完整流程需要较多计算资源和数据，可能限制大规模部署的复现门槛。
- **固定记忆容量上限**：1024 条记录/组的容量对于极端长上下文（>130K token）可能存在信息竞争和冲突。

## 研究启发与可借鉴点
1. **读写分离投影设计**：Memory 写入和查询使用独立投影（$w_t$ vs. $u_t$），使存储内容与检索请求解耦，该设计可迁移到其他需要显式记忆的循环架构中。
2. **优先级保留机制**：admission network 基于写入时可用信息评分，每块独立选 Top-K，此策略可在其他状态空间模型（如 Mamba、RWKV）的长上下文扩展中复用。
3. **四阶段训练范式**：M0（记忆初始化）→ R（联合适配）→ S（监督微调）→ RLVR（强化学习验证）的渐进训练流程，为开发具备显式记忆能力的循环模型提供了可复用的训练模板。
4. **长程记忆能力诊断基准**：论文设计的六种长程记忆任务（精确检索、状态更新解释、跨文档组合、变量绑定等）为评估模型的远距离信息利用能力提供了系统性评测方法。
5. **O(1) 内存 + 高吞吐的工程实践**：YANchor 在 H100 上实现 batch=512、512 条 64K 序列常驻的推理性能，其 CUDA Graph + Fused Kernel + 共享物理存储的工程优化方案对大规模部署具有参考价值。

## 关键术语表
- **Gated DeltaNet (GDN)**：一种改进的 Mamba2 循环层，通过 Delta 规则提升信息传输效率，是 YANchor 骨干中的主要循环组件。
- **SWA (Sliding Window Attention)**：固定窗口局部注意力，保留最近 1024 个 token 的精确细节交互。
- **Memory Anchor（记忆锚点）**：被 admission network 选中并持久化存储的关键 KV 记录，供后续查询检索。
- **Admission Network**：为每条候选记录生成优先级分数的轻量网络，决定哪些信息被保留在长期记忆中。
- **Grouped-query Retrieval**：将多个查询头映射到共享 KV 组的检索机制，减少存储开销的同时支持多路检索请求。
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：基于可验证任务结果（如数学答案正确性）的强化学习微调阶段。
- **pass@1**：单次生成样本的正确率，用于评估数学推理等任务的成功率。
- **TTFT (Time to First Token)**：从发送请求到返回第一个输出 token 的时间，衡量长序列推理的首字延迟。

## 可复现要素
- **数据集**：AIME 2024–2026、HMMT Feb 2026、MATH-500、GSM8K、MMLU/MMLU-Pro/MMLU-Redux、CMMLU、C-Eval、GPQA-Diamond、SuperGPQA、HumanEval、MBPP、LiveCodeBench v6、IFEval、IFBench、DROP、MuSR、LogiQA2、MMBench EN/CN、POPE、MME、ChartQA、AI2D（均为公开基准）
- **代码**：github.com/RocoreMatrix/YANchor
- **模型权重**：huggingface.co/HuishanJi/YANchor-4B
- **关键超参**：温度 0.6、top-p 0.95、top-k 20；响应预算 128K token；Memory 模块 8 个、每组 1024 条记录、head dim=256、256-token 块大小；局部窗口 1024 token
- **训练硬件**：论文未提及具体训练集群配置
- **推理测试硬件**：单张 H100 80GB
