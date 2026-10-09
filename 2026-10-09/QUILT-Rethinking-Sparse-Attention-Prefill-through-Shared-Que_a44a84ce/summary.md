---
title: "QUILT-Rethinking-Sparse-Attention-Prefill-through-Shared-Que"
source: https://arxiv.org/pdf/2610.11134v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:32:31"
field: "长上下文 LLM 推理系统优化"
keywords: ["sparse attention", "LLM inference", "prefill optimization", "cross-query reuse", "kernel design", "Ascend NPU"]
innovations: ["Shift-and-Compare Set Decomposition（SCSD）将不规则集合运算转化为数据并行原语", "Cascaded sharing 通过层级分解捕获多级粒度的跨查询 KV 共享", "Pipelined SCSD 与 attention 计算流水线重叠隐藏额外开销"]
benchmarks: ["LongBench", "GLM-5.3", "DeepSeek-3.2", "Ascend 910C NPU"]
---

# 论文速读：QUILT-Rethinking-Sparse-Attention-Prefill-through-Shared-Que

## 一句话总结
本文提出 **QUILT**，一种面向长上下文 LLM prefill 阶段稀疏注意力计算的**跨查询复用执行机制**，通过分组并行执行相邻 query、利用 Shift-and-Compare Set Decomposition（SCSD）高效分解共享/私有 KV 集合，并结合级联共享与 tile-aware 裁剪，将稀疏注意力 kernel 延迟最高降低 **55.1%**、处理 KV 数据量最高降低 **55.9%**，TTFT 最高降低 **36.8%**，同时几乎不损失精度。

---

## 研究问题与动机

1. **现有稀疏注意力 kernel 是 workload-agnostic 的**：OPS-Transformer 等主流实现采用 query-parallel 执行，每个 query 独立加载、反量化、处理其 Top-K 选定的 KV 条目，完全忽略相邻 query 之间的相关性。
2. **相邻 query 的 sparse workload 高度重叠**：论文在 GLM-5.3 上 profiling 发现，group size=2 时相邻 query 的 KV 交集比例始终 >85.91%（Qasper）和 >78.06%（GovReport），第一层更超过 90.03%，存在大量重复的 HBM 加载与反量化开销。
3. **重复处理造成显著性能浪费**：在 Ascend 910C NPU 上的 profiling（Figure 2）显示，KV loading + dequantization 占现有 sparse-attention kernel 执行时间的很大比例，说明跨查询复用潜力巨大。
4. **稀疏性破坏了 dense attention 的高效 multi-query packing**：dense attention 可将多个 query 打包到同一个 KV tile 上联合计算；而 dynamic sparse attention 中每个 query 选择不同的 KV 条目，使得这种打包难以直接应用，现有 kernel 被迫退回到 per-query 独立执行。

---

## 核心贡献（创新点）

1. **识别并量化了稀疏注意力中的跨查询 KV 复用机会**：通过系统 profiling 揭示相邻 query 高度重叠的规律，并指出这与现有 query-parallel 执行的 mismatch，是本文的出发点。
2. **提出 Shift-and-Compare Set Decomposition（SCSD）**：将不规则的集合运算（交集/差集）转化为排序 + 移位 + 比较 + 布尔原语的数据并行算法，适配现代加速器并行单元，比 naive Torch 实现快约 96×。
3. **设计 pipelined SCSD 执行**：将 SCSD 嵌入注意力计算管线，利用矩阵计算（QK/PV）期间向量单元的闲置算力并行构造未来 segment，使额外开销在关键路径上几乎不可见。
4. **引入 cascaded sharing 捕获多级粒度共享**：从大 query group 开始逐级提取共有的 KV 条目，递归进入子 group，避免固定 group size 只能捕获单一粒度共享的缺陷。
5. **提出 tile-aware clipping + importance-aware 裁剪**：对未对齐 attention tile 大小的尾部 segment 做精确 push-down 以提升 tile 利用率；在最终 query-specific 层级对低重要性残差做近似裁剪，消除未充分利用的 tail tile，同时保持数值保真度（head-level cosine similarity 均值 1.0000，显著优于 random clipping 的 0.9999）。

---

## 方法详解

### 整体架构：Group-based Query Execution
QUILT 不再逐 query 独立执行，而是将相邻 query 分组（默认 group size=8），联合处理其 correlated sparse workloads，使共享 KV 条目只需从 HBM 加载和反量化一次，再被组内所有 query 复用。

### 3.1 Shift-and-Compare Set Decomposition（SCSD）
给定两个 Top-K 索引列表 A 和 B（各含 K 个唯一索引），SCSD 的计算流程如下：

1. **拼接 + 排序**：$(C, T) = \text{Sort}(A \| B)$，其中 $C$ 是排序后的索引序列，$T_i \in \{0,1\}$ 标记来源。
2. **双向移位 + 比较**：构造右移序列 $C_r$ 和左移序列 $C_l$，计算：
   - $D = (C = C_r)$
   - $F = (C = C_l)$
3. **提取交集**：$E = C[D] = A \cap B$（保留 $D$ 中匹配的其中一份副本）。
4. **提取私有部分**：$G = \neg(D \lor F)$ 标记仅出现在一个列表中的元素，再通过来源标签 $T$ 分离：
   - $A - E = C[G \land (T=0)]$
   - $B - E = C[G \land (T=1)]$

SCSD 的优势：全部操作均为规则的数据并行原语（shift、element-wise comparison、Boolean、masked select），无串行标量循环和分支，可充分利用加速器并行单元。

### 3.2 Pipelined Execution
若将 SCSD 作为独立预处理器，其延迟 $T_{SCSD}$ 会直接加到关键路径上（总时间 $T_{SCSD} + T_{Attn}$）。QUILT 的解决方案：

- **关键观察**：现代加速器的矩阵引擎（QK/PV 计算）和向量单元（softmax、SCSD）可并发执行；稳态期间向量侧负载通常短于矩阵侧，存在空闲向量槽位。
- **流水线策略**：仅在实际消费前构造所需的稀疏 segment。先计算少量初始 segment 启动管线（打破 producer-consumer 依赖），稳态后矩阵侧执行当前 segment 的 QK/PV，向量侧同时构造未来 segment 的共享/私有部分。
- **细粒度分阶段**：将完整 SCSD 任务拆分为多个小阶段，分布到连续的 attention pipeline slot 中，更好地利用空闲向量资源。

### 3.3 Cascaded Sharing
固定 group size 的局限性：
- 小 group（如 2）：局部重叠率高，但跨组共享被遗漏。
- 大 group（如 8）：只捕获所有 query 共同选中的 KV，许多 subgroup 级别的共享也被遗漏。

Cascaded sharing 的层级分解（默认起始 group size=8）：
1. 提取当前 group 所有 query 的交集 KV，一次性加载/反量化，供组内复用。
2. 从各 query 的 Top-K 集中移除已共享条目。
3. 将剩余 entries 递归划分为更小的子 group，继续提取共享部分。
4. 最终剩余条目作为 query-specific residual 独立处理。

示例（Figure 7）：4 个 query 的 Top-4 集合分别为 $\{A,B,C,E\}, \{A,B,C,F\}, \{A,B,D,G\}, \{A,B,D,H\}$。Cascaded sharing 先提取 4-way 交集 $\{A,B\}$，再在子 group $\{Q_1,Q_2\}$ 中提取 $\{C\}$，在 $\{Q_3,Q_4\}$ 中提取 $\{D\}$，最后各自处理私有残差 $E,F,G,H$。总处理 KV 数从 naive 的 16 降至 8（降低 50%），优于固定 group=2 或 group=4 的 10。

### 3.4 Tile-Aware Clipping
Cascaded sharing 产生的 segment 大小通常不与原生 attention tile size $T$ 对齐，导致低利用率的 tail tile。QUILT 的策略：

1. **Tile-utilization threshold $\tau$**（默认 50%）：对 shared segment 的尾部，若剩余元素数 $r \geq \tau T$，则 pad 至整 tile 在当前层级处理（mask 填充部分）；若 $r < \tau T$，则 push-down 到下一层 cascade，与更细粒度 workload 合并以提升 tile 利用率。此步为精确操作。
2. **Importance-aware clipping**（唯一近似步骤）：在最终 query-specific 层级，观察到残差条目主要集中在原始 Top-K 排名的尾部（低重要性区）。因此保留 $q_i T$ 个最高 indexer score 的条目，裁剪掉 $r_i$ 个最低重要性残差，消除最后一个未充分利用的 tile。

数值保真度验证（Figure 15）：importance-aware clipping 相比 random clipping 在 head-level 最小 cosine similarity 从 0.312 提升至 0.406，hidden-level 从 0.664 提升至 0.719，mean 分别从 0.9999 和 0.9998 提升至 1.0000。

---

## 实验与结果

**测试平台**：16× Ascend 910C NPU（64GB HBM/卡），CANN 9.1.0，XYServe 推理引擎。

**模型**：GLM-5.3（Top-2048 DSA，支持跨层索引复用）、DeepSeek-3.2（Top-2048 Lightning Indexer，每层独立选 KV）。

**基线**：OPS-Transformer 优化 sparse-attention kernel（当前 Ascend NPU 最先进实现）。

**数据集**：LongBench 21 个数据集（QA、Document Summarization、Code Generation 等）。

### 主要结果（图 9、表 3）

| 模型 | 平行方式 | 平均 kernel 延迟降低 | 平均处理 KV 降低 |
|------|----------|---------------------|-----------------|
| GLM-5.3 | TP | **53.8%**（最高 55.1% on GO） | **54.7%**（最高 55.9% on GO） |
| GLM-5.3 | SP | 43.4% | — |
| DeepSeek-3.2 | TP | 28.7%（最高 37.9% on QA） | 30.4%（最高 39.6% on QA） |
| DeepSeek-3.2 | SP | 17.1% | — |

### TTFT 提升（图 11）
- GLM-5.3 / TP：平均降低 **35.6%**（最高 36.8% on Musique）
- GLM-5.3 / SP：平均降低 10.4%
- DeepSeek-3.2 / TP：平均降低 **23.0%**（最高 30.2% on QA）
- DeepSeek-3.2 / SP：平均降低 3.4%

### 精度保持（表 2）
- GLM-5.3：21 个 LongBench 数据集平均绝对误差 **0.42 分**（相对误差 0.90%）
- DeepSeek-3.2：平均绝对误差 **0.75 分**（相对误差 2.01%）
- 两者均属于 negligible accuracy degradation。

### 消融实验（图 12–15）
- **SCSD vs Naive**：Naive 实现比基线慢 21.84×（TP）/ 33.65×（SP），SCSD 降低延迟 96.1%（TP）/ 97.3%（SP）。
- **Pipeline**：进一步降低 45.9%（GLM-5.3/TP）。
- **Cascaded vs 固定 group**：GLM-5.3 下比 group=2/4/8 分别减少处理 KV 30.8%/20.6%/22.9%。
- **Tile-aware clipping**：GLM-5.3/TP 额外降低 23.71%，DeepSeek-3.2/TP 降低 24.44%。
- **共享长度敏感性**：sharing length 从 0→2048，GLM-5.3/TP 延迟降低 42.74%，DeepSeek-3.2/TP 降低 41.03%。

---

## 相关工作脉络

1. **训练-free 稀疏注意力**：MInference [17]、FlexPrefill [20]、XAttention [33] 通过轻量代理计算在推理时识别重要 KV 条目，不修改模型训练。QUILT 与其正交，可直接应用于这些方法生成的 sparse workload。
2. **Model-native 稀疏注意力**：NSA [36]、DeepSeek V3.2/V4 [9,10,11]、GLM-5 [15]、Hunyuan-4 [29] 将可训练 indexer 嵌入模型架构。QUILT 对 indexer 生成策略 agnostic，适用于所有上述模型。
3. **解码阶段稀疏 KV 管理**：H₂O [37]、StreamingLLM [32]、Scissorhands [24]、SnapKV [22]、Quest [28]、InfLLM [31]、ArkVale [4]。这些方法在 decode 阶段做 KV cache 压缩/动态选择；QUILT 聚焦 prefill 阶段的跨查询复用，目标互补。
4. **硬件高效注意力 kernel**：FlashAttention [7]、FlashAttention-2 [6]、FlashAttention-3 [26]、FlashInfer [34]。这些工作优化 dense attention 的 tile-based 执行和 on-chip 复用；QUILT 在此基础上针对 sparse attention 特有的跨查询冗余做进一步优化。
5. **Prefill 优化系统**：FlashPrefill [12] 做硬件感知的 pattern discovery；QUILT 在其上叠加跨查询 sharing 优化。
6. **解码阶段跨步 KV 缓存**：RetroInfer [5]、LiteCache [35]、SparseServe [39] 利用相邻 decode step 的 KV 选择相似性做 HBM 缓存；QUILT 在 prefill 阶段的 attention kernel 内部做类似但更底层的复用。

---

## 局限性与未来方向

1. **实验仅在 Ascend NPU 上验证**：虽然作者声称方法适用于 GPU（Tensor Core + SIMT core 架构类似），但未在 GPU 平台上做实测，跨硬件泛化性有待验证。
2. **只对 model-native 稀疏注意力（DSA）做评估**：training-free 稀疏方法（如 MInference、FlexPrefill）产生的 overlap 模式可能与 DSA 不同，QUILT 在这些场景下的收益未验证。
3. **tile-aware clipping 是唯一的近似操作**：虽然精度损失 negligible（cosine similarity 均值 1.0000），但在对精度极度敏感的应用场景（如数学推理、代码生成）中仍需评估 worst-case 影响。
4. **固定起始 group size（默认 8）**：不同数据集/层/模型的工作负载特征差异大，自适应选择最优 group size 和 threshold $\tau$ 的策略未探索。
5. **未评估 multi-device 跨设备 sharing**：当前只在单 device 内做跨 query 共享；在多 NPU 的 sequence parallelism 配置下，跨设备 KV 复用潜力未挖掘。

---

## 研究启发与可借鉴点

1. **SCSD 的可迁移性**：将不规则集合运算转化为排序 + 移位 + 比较 + 布尔原语的思路，可推广到其他需要高效 set intersection/difference 的 kernel 场景（如 MoE expert routing、RAG recall 阶段的向量检索）。
2. **流水线隐藏额外开销的设计模式**："priming + steady-state producer-consumer" 的管线策略适用于任何有预处理开销但后续计算可并行的 kernel 优化。
3. **Cascaded sharing 的多粒度思想**：从粗到细的层级分解可用于其他需要平衡"共享范围"与"共享条件"的系统问题（如 shared memory cache management、batch scheduling）。
4. **importance-aware clipping 的 trade-off 设计**：在精度-性能 trade-off 中，利用已有打分器（indexer score）做裁剪决策比随机裁剪显著更优，这一原则可迁移到 KV cache eviction、token pruning 等领域。
5. **跨查询相关性 profiling 的发现方法**：通过系统性 profiling（不同 group size、不同层、不同数据集）量化 workload correlation，为后续 kernel 设计提供定量依据，这一研究方法本身值得借鉴。

---

## 关键术语表

**Sparse Attention**：仅让每个 query token attend 到少量重要 KV 条目（Top-K），而非全部上下文，以降低 prefill 阶段计算量和内存流量。

**Cross-query KV Reuse**：相邻 query 的稀疏选中集合高度重叠，同一 KV 条目被多个 query 独立重复加载和反量化的现象，是 QUILT 优化的核心目标。

**Shift-and-Compare Set Decomposition（SCSD）**：将两个 Top-K 索引列表拼接排序后，通过双向移位和元素比较提取交集与差集的数据并行算法，避免标量循环和分支。

**Cascaded Sharing**：从大 query group 开始逐级提取共有的 KV 条目，递归进入子 group，捕获多级粒度共享，避免固定 group size 的粒度局限。

**Pipelined SCSD Execution**：将 SCSD 分阶段嵌入注意力计算管线，利用矩阵计算期间向量单元的闲置算力并行构造未来 segment，使额外开销在关键路径上不可见。

**Tile-aware Clipping**：根据 attention tile 对齐情况决定是否将未充分利用的尾部 segment push-down 到更细粒度层级，并在最终层级对低重要性残差做裁剪以消除低利用率 tail tile。

**Time-to-First-Token（TTFT）**：prefill 阶段结束到第一个输出 token 生成完成的时间，直接决定用户感知的响应延迟。

**Model-native Sparse Attention**：将可训练轻量 indexer 嵌入模型架构的稀疏注意力设计（如 DeepSeek DSA），在训练和推理阶段动态选择重要 KV 条目。

---

## 可复现要素

- **数据集**：LongBench（21 个数据集，官方公开）。
- **模型**：GLM-5.3、DeepSeek-3.2（开源权重，引用 [9][15]）。
- **代码/权重**：基线使用 OPS-Transformer（GitHub: https://github.com/hicann/ops-transformer）；QUILT 代码论文中未明确开源声明（"论文未提及"）。
- **硬件**：16× Ascend 910C NPU，CANN 9.1.0，XYServe 推理引擎。
- **关键超参**：默认 group size=8；tile-utilization threshold $\tau$=50%；Top-K=2048（GLM-5.3 和 DeepSeek-3.2 均使用）。
- **并行配置**：Tensor Parallelism（TP）和 Sequence Parallelism（SP）均有评测。

---
