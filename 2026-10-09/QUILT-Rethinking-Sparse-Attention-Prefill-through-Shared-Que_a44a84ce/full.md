# QUILT: Rethinking Sparse-Attention Prefill through Shared Query Execution

Zhenduo Zhao<sup>∗</sup>   
zhaozhenduo@huawei.com   
Huawei Technologies Co., Ltd. China

Zhiyi Chen chenzhiyi17@huawei.com Huawei Technologies Co., Ltd. China

Zequn Gong gongzequn1@huawei.com Huawei Technologies Co., Ltd. China

Qihui Zhou<sup>∗</sup> zhouqihui3@huawei.com Huawei Technologies Co., Ltd. China

Chuangguan Ye   
yechuangguan@huawei.com   
Huawei Technologies Co., Ltd. China

Jing Li joy.lijing@huawei.com Huawei Technologies Co., Ltd. China

Guoping Long robin3@huawei.com Huawei Technologies Co., Ltd. China

Mingcong Song   
songmingcong@huawei.com   
Huawei Technologies Co., Ltd. China

Fengfan Hou houfengfan@huawei.com Huawei Technologies Co., Ltd. China

Hongjie Si sihongjie@huawei.com Huawei Technologies Co., Ltd. China

## Abstract

Sparse attention has emerged as a promising approach to reduce the rapidly growing cost of long-context attention. Existing sparse-attention kernels typically adopt query-parallel execution, where diferent queries independently load, dequantize, and process their selected KV entries. However, such execution is largely workload-agnostic and overlooks the strong correlation among sparse workloads across queries. We observe that neighboring queries frequently select substantially overlapping KV entries. Consequently, processing them independently repeatedly loads and dequantizes the same KV data, resulting in significant redundant memory trafic and computation.

To solve this issue, we present QUILT, a new workloadaware sparse-attention execution mechanism with groupbased query execution that explicitly exploits cross-query reuse. QUILT jointly processes neighboring queries and reuses KV entries shared across their sparse workloads, reducing redundant KV data processing and improving hardware utilization. To eficiently realize such execution, QUILT introduces Shift-and-Compare Set Decomposition (SCSD), a new data-parallel algorithm that transforms irregular set operations into regular data-parallel primitives that can be efficiently executed on modern accelerators. QUILT further pipelines SCSD with attention computation to hide its remaining overhead. To accommodate heterogeneous sharing patterns, QUILT is equipped with cascaded sharing to capture cross-query reuse hierarchically at multiple granularities. Finally, QUILT adopts a tile-aware execution strategy that jointly considers sharing granularity and hardware tile utilization, while selectively removing low-importance query-specific tails to eliminate underutilized tiles.

![](images/c0a7ee4b19e98d48b26afd1ce1636397af8aa2593b8439402ffdd98615ea2bdb.jpg)

![](images/99cc222611e121fa10ad6cce1a969c4c3bef91b414c6ff382b6ee34a7f5270da.jpg)

![](images/560745faad8bd882b3a581d3eb0fb7833e88ad6c483ced77a123baacb0b72f64.jpg)

![](images/a6ac5e122d2f3e2ed0f482bbcadbd37d9c4230a76a0be23a0269641044baf0ef.jpg)  
Figure 1. Top-K KV overlap ratios among neighboring queries across model layers and query-group sizes for GLM-5.3 [15] on the Qasper and GovReport datasets in LongBench.

We evaluate QUILT on LongBench using GLM-5.3 and DeepSeek-3.2 under both tensor and sequence parallelism. Compared with the state-of-the-art sparse-attention kernel, QUILT reduces average kernel latency by up to 55.1% and the amount of processed KV data by up to 55.9%, while reducing time-to-first-token (TTFT) latency by up to 36.8% with negligible accuracy degradation.

## 1 Introduction

The rapid expansion of context windows has enabled large language models (LLMs) to tackle increasingly complex and long-horizon workloads, including long-document understanding [1, 2], repository-level code analysis [18, 23], and agentic applications that maintain extensive interaction histories [25, 38]. However, supporting such long contexts comes at a substantial inference cost. In particular, during the prefill phase, the model must process the entire input context before generating the first output token, making prefill latency directly visible to users as time-to-first-token (TTFT). As sequence length increases, attention incurs rapidly growing computation and data movement over the expanding keyvalue (KV) context. Consequently, attention constitutes an increasing fraction of prefill latency at long context lengths, motivating more eficient attention execution for responsive long-context LLM inference.

Sparse attention has emerged as a promising approach to mitigate the growing cost of long-context attention [4, 5, 9, 17, 20, 24, 28, 32, 33, 37]. Instead of attending to the entire KV context, each query token employs a lightweight indexer to dynamically identify a small subset of important KV entries and performs attention only over the selected Top-� entries. Since � is typically much smaller than the context length and can remain bounded as the context grows, sparse attention substantially reduces the computation and memory trafic of long-context prefill. To further reduce the memory footprint and bandwidth cost of KV accesses, sparse attention is often combined with KV quantization, where KV entries are stored in a low-precision format and dequantized on demand when selected for attention computation.

However, sparsity fundamentally changes how attention can be eficiently mapped onto accelerators. In conventional dense attention, multiple queries can be packed together and jointly processed against regular KV tiles on the same compute core, forming suficiently large matrix operations for eficient execution [6, 34]. In dynamic sparse attention, diferent queries generally select diferent KV entries, introducing query-specific access patterns that make such eficient multiquery packing largely impractical. Existing sparse-attention kernels [3, 21] therefore typically adopt a query-parallel execution strategy, where diferent queries are mapped to separate compute cores and independently gather, dequantize, and process their selected KV entries.

While query-parallel execution efectively exploits the massive parallelism of modern accelerators, it treats the KV accesses of diferent queries independently and overlooks an important workload characteristic: neighboring queries often exhibit highly correlated sparse attention patterns and select substantially overlapping KV entries. To quantify this behavior, we profile GLM-5.3 [15] on LongBench [2] across diferent datasets, layers, and query-group sizes. As shown in Figure 1, substantial cross-query overlap consistently exists across model layers and group sizes, and is particularly pronounced in early layers and for smaller query groups. Specifically, when the group size is two, the overlap ratio consistently exceeds 85.91% and 78.06% across all layers on Qasper [8] and GovReport [16], respectively, and further reaches over 90.03% in the first layer. As the group size increases, the fraction of KV entries shared by all queries in the group gradually decreases, yet considerable overlap persists across most layers. These results reveal substantial opportunities for cross-query KV reuse during prefill, while also showing that the degree of sharing varies across datasets, layers and sharing granularities.

![](images/22dcb9303cf3615f0d7836333f8ff1a126662496262691f47b2e6cfc5709c4f6.jpg)  
Figure 2. Latency breakdown of the existing sparseattention kernel for GLM-5.3 and DeepSeek-4.1 on an Ascend 910C NPU across diferent prefill sequence lengths (SP) and query heads (TP).

Unfortunately, existing query-parallel execution cannot exploit such cross-query reuse. The root cause is that existing sparse-attention kernels are workload-agnostic: they execute each query according to its own Top-� selection without considering the correlation or sharing structure of sparse workloads across queries. Consequently, KV entries shared by neighboring queries are repeatedly loaded from HBM into on-chip memory and independently dequantized on diferent compute cores, resulting in redundant data movement and computation. As shown in Figure 2, our profiling on an Ascend 910C NPU shows that KV loading and dequantization together account for a large portion of the execution time of the existing sparse-attention kernel across diferent prefill sequence lengths on GLM-5.3 [15] and DeepSeek-4.1 [11]. This substantial overhead suggests that explicitly exploiting cross-query KV reuse can provide significant opportunities to further improve sparse-attention execution eficiency.

To exploit these cross-query reuse opportunities, we propose QUILT, a new workload-aware sparse-attention execution mechanism with group-based query execution. Instead of processing each query independently, QUILT groups neighboring queries and jointly executes their correlated sparse workloads according to the sharing structure of their

KV accesses. KV entries shared by multiple queries are loaded from HBM and dequantized only once before being reused across the group, thereby reducing redundant KV data processing. Moreover, jointly processing neighboring queries increases the efective computation granularity of sparse attention and improves accelerator utilization. However, translating such cross-query sharing into actual performance gains is non-trivial and introduces several key design challenges.

Firstly, group-based execution introduces additional overhead for identifying the shared and query-specific KV entries among neighboring queries. A naive implementation based on scalar traversal, explicit loops, or pairwise membership checking can incur substantial set-processing overhead, which may ofset or even exceed the benefit of cross-query sharing. To address this challenge, we propose Shift-and-Compare Set Decomposition (SCSD), a data-parallel algorithm for eficiently computing the intersections and differences of sparse KV sets. SCSD concatenates and sorts the KV indices, transforming irregular set operations into regular shift, comparison, and Boolean primitives that map eficiently onto the parallel execution units of modern accelerators. Moreover, we overlap SCSD with attention computation by embedding it into the existing kernel pipeline, largely hiding its latency behind useful computation. As a result, the kernel can expose cross-query sharing opportunities with near-zero additional overhead on the critical path.

Secondly, the degree of cross-query KV sharing varies substantially across datasets, layers, and sharing granularities, making a fixed query-group size inherently suboptimal. A small group size captures strong local overlap but misses reuse opportunities across groups, whereas a large group size requires KV entries to be shared by more queries and therefore retains only a smaller common subset. To address such heterogeneous sharing patterns, we introduce cascaded sharing, which hierarchically decomposes sparse KV workloads across multiple granularities. Starting from a relatively large query group, the mechanism first extracts KV entries shared by the entire group, then recursively identifies additional sharing within smaller subgroups, and finally processes the remaining query-specific entries. This hierarchical decomposition captures both coarse-grained and fine-grained reuse without committing to a single fixed sharing granularity.

Thirdly, cascaded sharing produces shared and queryspecific segments whose sizes are often misaligned with the native attention tile size, causing poorly utilized tail tiles. To address this issue, we introduce tile-aware clipping for the sharing process. For each shared segment, we examine the occupancy of its tail tile. If the occupancy falls below a configurable threshold, we push the tail down to the next cascade level, where it is merged with finer-grained workloads to improve tile utilization. Otherwise, we pad the tail to a full tile and process it at the current sharing level. This push-down is exact and trades a small amount of sharing for improved tile utilization. At the final query-specific level, where no further push-down is possible, we observe that the remaining KV entries are predominantly concentrated near the tail of the original Top-� ranking and thus tend to have lower importance scores. We therefore apply importanceaware clipping, pruning the lowest-ranked residual entries to eliminate the final underutilized tile. This approximation further reduces attention-tile execution while introducing negligible accuracy degradation.

We evaluate QUILT on LongBench [2] using GLM-5.3 [15] and DeepSeek-3.2 [9] under both tensor and sequence parallelism. Compared with the state-of-the-art sparse-attention kernel, QUILT reduces the average kernel execution latency by up to 55.1% and the amount of processed KV data by up to 55.9%. These kernel-level improvements translate into substantial end-to-end benefits, reducing the average time-tofirst-token (TTFT) latency by up to 36.8% while introducing negligible accuracy degradation. These results demonstrate that exploiting cross-query workload structure can efectively translate the inherent correlation in sparse attention into tangible system-level performance gains. To summarize, this paper makes the following contributions:

• We identify substantial cross-query reuse opportunities in sparse-attention workloads and reveal a mismatch between their correlated workload characteristics and existing query-parallel execution, which processes queries independently and results in redundant KV data processing and ineficient hardware utilization.

• We propose QUILT, a new workload-aware sparse-attention mechanism with group-based query execution to exploit cross-query reuse. We develop Shift-and-Compare Set Decomposition for eficient data-parallel set processing and pipeline it with attention computation to hide its overhead. We further introduce cascaded sharing to capture reuse at multiple granularities and tile-aware execution to jointly optimize sharing granularity and hardware utilization.

• We conduct comprehensive experiments on representative long-context models and workloads under diferent parallel configurations. Compared with the state-of-theart sparse-attention implementation, QUILT substantially reduces processed KV data and kernel execution latency, translating these improvements into significant TTFT improvement with negligible accuracy degradation.

## 2 Background and Motivation

## 2.1 Generative LLM Inference Basics

Transformer. Modern generative LLMs are typically built from stacks of Transformer layers, each consisting of a selfattention module and a feed-forward network (FFN). Given an input sequence $X = [ x _ { 1 } , x _ { 2 } , \dots , x _ { N } ]$ , each layer projects the hidden states into query, key, and value matrices �, �,

![](images/eefc5cd9c8bcf7e5ec0062fd7ed6eeaad35ae9aece89363d5594f5e797855e5c.jpg)  
Figure 3. Illustration of the sparse attention workflow.

and � , and computes

$$
S = { \frac { Q K ^ { T } } { \sqrt { d } } } + M , \quad P = \operatorname { s o f t m a x } ( S ) , \quad O = P V ,
$$

where � is the causal mask and $P _ { i , j }$ denotes the attention weight from query token � to key token �. The output is then processed by the FFN and passed to the next layer.

Prefill and decoding. LLM inference consists of a prefill phase and a decode phase. During prefill, all input tokens are processed in parallel to produce the first output token, and the prefill latency directly determines the time-to-firsttoken (TTFT). Under causal self-attention, each query token attends to the key and value vectors of all preceding tokens. Therefore, for an input sequence of length �, dense prefill attention incurs $O ( N ^ { 2 } d )$ computation together with rapidly increasing KV data movement as the context grows. After prefill, decoding generates new tokens autoregressively, one token at a time. This work focuses on accelerating the atten tion computation in the prefill phase.

## 2.2 Sparse Attention

The workflow of sparse attention. Sparse attention reduces the cost of long-context attention by restricting each query token to attend to only a small subset of important KV entries. As shown in Figure 3, a lightweight indexer first estimates the relevance of previous KV entries to the current query and selects a Top-� subset, where � is typically much smaller than the total context length. Attention is then computed only over the selected KV entries, substantially reducing both computation and memory trafic compared with dense attention. To further reduce memory footprint and bandwidth consumption, the KV cache is often stored in a low-precision format and the selected entries are dequantized on demand before attention computation.

Importantly, sparsification fundamentally changes the execution workflow of the attention kernel. As illustrated in Figure 4, dense attention operates on regular KV tiles that can be jointly processed by multiple queries, whereas sparse attention introduces query-specific and irregular KV selections. Consequently, sparse kernels typically need to independently gather, dequantize, and process the selected KV entries for each query, leading to an execution structure substantially diferent from that of dense attention.

![](images/97dd2becc5d7eab325d04c436e6dc15cdfede43c9bc49feaec9a28cff20d5291.jpg)  
Multi-query packing · Regular KV access ery parallelism · Per-query KV access  
Figure 4. Comparison of the execution workflows of dense and sparse attention kernels.

Training-free sparsification. Early approaches such as MInference [17], FlexPrefill [20], and XAttention [33] sparsify pretrained LLMs without modifying their original training procedure. Instead, they exploit the intrinsic structure of attention maps at inference time and use lightweight proxy computations to identify important KV entries. For example, MInference detects recurring structured sparsity patterns and constructs sparse attention indices accordingly, while FlexPrefill adaptively selects sparse patterns and computation budgets based on the observed attention characteristics of individual inputs and heads. XAttention further uses lightweight antidiagonal scoring to estimate block importance and select the most relevant KV blocks for attention.

Model-native sparse attention. More recently, sparse attention has increasingly been incorporated directly into model training and architecture. Native Sparse Attention (NSA) [36] demonstrates that sparse attention can be trained end-to-end, while DeepSeek-V3.2-Exp [9] further integrates DeepSeek Sparse Attention (DSA) as a native component of a large-scale LLM, using a trainable lightweight indexer to dynamically select important KV entries during both training and inference. Similar designs have since been adopted and extended by recent models, including the GLM-5 series [15], DeepSeek-V4 series [10, 11], and Hunyuan-4 series [29], suggesting an emerging trend toward model-native sparse attention for eficient long-context processing.

Importantly, recent models are adopting increasingly sophisticated sparsification strategies. DeepSeek-4.1 [11], for example, combines sparse-index generation, refinement over restricted candidate sets, and cross-layer index reuse. Despite such increasingly complex mechanisms, the attention kernel ultimately operates on a query-specific set of selected KV entries. QUILT operates directly on these resulting sparse workloads and is therefore agnostic to how sparse indices are generated, refined, or reused. Consequently, QUILT remains applicable to a broad range of existing and emerging sparse-attention architectures, including both training-free and model-native designs, without requiring changes to their sparsification strategies.

## 2.3 Architecture of Modern Accelerators

Hardware components and functions. Modern AI accelerators typically integrate heterogeneous compute units specialized for diferent classes of operations. Matrix-intensive operations, such as GEMM and the QK/PV computations in attention, are executed on high-throughput matrix engines, e.g., Tensor Cores on GPUs and Cube units on Ascend NPUs. In contrast, element-wise operations, reductions, comparisons, and other lightweight computations are mainly handled by general-purpose execution units following SIMD/SIMT execution models, such as CUDA cores on GPUs and Vector units on Ascend NPUs. These compute units are backed by a hierarchical memory system, where data are transferred from of-chip HBM to faster on-chip memories before being consumed by the compute units. Importantly, matrix and vector units can execute concurrently when their data dependencies permit, enabling kernels to pipeline heterogeneous computations.

Tile-based execution. Modern accelerator kernels typically execute large tensor operations in a tile-based manner. Large tensors are first partitioned into smaller tiles that can be dis tributed across compute cores and eficiently accommodated by the on-chip memory hierarchy. Within each tile, computation is further mapped to the native execution granularity of the underlying hardware. Matrix engines operate on fixed or constrained matrix fragments, whereas vector units process a fixed number of elements per SIMD/vector instruction. Therefore, the efective tile shape is jointly determined by hardware instruction granularity, data type and alignment requirements, on-chip memory capacity, and available paral lelism. Operations on successive tiles can be organized into a software pipeline, allowing data movement, matrix computation, and vector computation to proceed concurrently.

## 2.4 Opportunities and Challenges

Mismatch between workload and execution. As discussed in Section 1, neighboring queries in sparse attention frequently select highly overlapping KV entries. However, existing query-parallel kernels process each query independently and therefore fail to exploit such cross-query reuse. As illustrated in Figure 4, dense attention can jointly process multiple queries against regular KV tiles, whereas sparse attention typically falls back to query-independent execution because each query selects a diferent set of KV entries. Consequently, the same KV entries may be repeatedly loaded from HBM into on-chip memory and independently dequantized across diferent queries, introducing redundant data movement and computation. Moreover, executing each sparse query in isolation often results in narrow and fragmented matrix operations, limiting the utilization of the high-throughput compute units of modern accelerators.

This mismatch between the correlated sparse workload and query-independent execution motivates us to rethink the execution paradigm of sparse-attention kernels. Rather than treating each query as an isolated workload, we advocate a workload-aware execution strategy that explicitly exploits the structural correlation across queries to improve data reuse, reduce redundant computation, and better utilize the heterogeneous compute resources of modern accelerators.

Design challenges. Translating cross-query sharing opportunities into actual kernel-level speedup is non-trivial and requires addressing several key challenges:

❶ Set-decomposition overhead: Exploiting cross-query sharing requires decomposing neighboring Top-� lists into shared and query-specific KV entries. Naive implementations based on scalar traversal, explicit loops, or irregular membership checking can incur substantial overhead, which may ofset or even exceed the savings from KV reuse. Moreover, even after accelerating these set operations, executing them as a standalone preprocessing stage still exposes the remaining overhead on the kernel critical path. Eficient cross-query sharing therefore requires both a highly parallel, hardware-friendly set-decomposition algorithm and an execution strategy that overlaps its computation with the original attention pipeline.

❷ Heterogeneous sharing granularity: Cross-query sharing naturally exists at diferent granularities. Small query groups capture strong local overlap but miss reuse across groups, whereas large groups only capture KV entries shared by all queries and can therefore miss substantial subgroup-level reuse. Eficient execution must exploit multiple sharing granularities rather than committing to a single fixed group size.

❸ Tile underutilization: Reducing the number of logically processed KV entries does not necessarily translate into proportional hardware savings. The shared and queryspecific segments produced by group-based execution are often misaligned with the native attention tile size, resulting in poorly utilized tail tiles. Eficient execution must therefore jointly consider sharing granularity and hardware tile utilization.

## 3 The Design of QUILT

To address the workload execution mismatch identified in Section 2.4, we propose QUILT, which redesigns sparseattention execution around cross-query workload sharing. Instead of processing each query independently, QUILT groups neighboring queries and jointly executes their correlated sparse workloads, enabling shared KV entries to be reused across queries while improving accelerator utilization.

![](images/e6e857d36bc1db445541788872e6f544a294191b7d9a3a808a689ea32a61525f.jpg)  
Figure 5. A running example of Shift-and-Compare Set Decomposition (SCSD) with two input KV index sets.

Realizing such workload-aware execution mechanism requires addressing the three challenges discussed above. For Challenge 1, we propose Shift-and-Compare Set Decomposition (SCSD) to transform irregular set operations into hardware-friendly vector operations, and further develop pipelined SCSD execution to overlap the remaining set-processin overhead with the original attention pipeline. For Challenge 2, we introduce cascaded sharing, which hierarchically captures cross-query reuse at multiple sharing granularities instead of relying on a fixed query-group size. For Challenge 3, we develop tile-aware cascaded execution to jointly optimize sharing granularity and hardware tile utilization, and apply importance-aware clipping to eliminate poorly utilized query-specific tail tiles with negligible accuracy impact.

## 3.1 Shift-and-Compare Set Decomposition

Group-based query execution requires eficiently decomposing the Top-� selections of neighboring queries into shared and query-specific KV entries. A straightforward implemen tation may rely on scalar merge, explicit loops, or pairwise membership checking between Top-� lists. However, sparse attention already performs a relatively small amount of computation per query, leaving little room for additional bookkeeping overhead. Consequently, a naive set-decomposition <sup>ft</sup>implementation can introduce substantial serialized control flow and irregular comparisons, whose cost may ofset or even exceed the savings from eliminating redundant KV load ing and dequantization. Eficient set decomposition is therefore critical to realizing the benefits of cross-query sharing. To address this challenge, we propose Shift-and-Compare Set Decomposition (SCSD), which transforms irregular set operations into regular sorting, shifting, and equality-comparison primitives that can be eficiently parallelized on modern accelerators.

Figure 5 shows a running example of SCSD with two Top-� index lists, � and �, each containing � unique KV indices. We use the two-way case for clarity, while the same principle naturally generalizes to multi-way set decomposition. SCSD first concatenates the two input lists into a 2�-element sequence and jointly sorts the indices while tracking the source of each element:

$$
( C , T ) = { \mathrm { S o r t } } ( A \| B ) ,\tag{1}
$$

where � denotes the sorted indices and $T _ { i } \in \{ 0 , 1 \}$ indicates whether $C _ { i }$ originates from � or �. Since each Top-� list contains unique indices, an index appearing in both � and � occurs exactly twice in �. After sorting, these two occurrences are guaranteed to be adjacent.

SCSD therefore identifies shared entries through bidirectional shift-and-compare operations. Specifically, we construct the right- and left-shifted sequences $C _ { r }$ and $C _ { l } ,$ and compute

$$
D = \left( C = C _ { r } \right)\tag{2}
$$

$$
F = ( C = C _ { l } )\tag{3}
$$

For each shared index, the two adjacent copies are captured by � and � , respectively. Selecting one copy using � therefore directly yields the intersection:

$$
E = C [ D ] = A \cap B\tag{4}
$$

An index that matches neither of its neighbors must appear in only one input list. We thus identify all query-specific entries using

$$
G = \neg ( D \lor F )\tag{5}
$$

and further separate them according to their source labels:

$$
A - E = C [ G \land ( T = 0 ) ]\tag{6}
$$

$$
B - E = C [ G \land ( T = 1 ) ]\tag{7}
$$

This formulation is particularly well suited to modern accelerators. SCSD consists of a fixed sequence of regular data-parallel primitives, including shifts, element-wise comparisons, Boolean operations, and masked selections, which can be executed concurrently across the Top-� sequence. Unlike conventional set-intersection implementations based on explicit loops, scalar pointer advancement, or branchheavy membership checks, SCSD avoids serialized scalar execution and irregular control flow. As a result, it exposes abundant data-level parallelism and substantially reduces the bookkeeping overhead of cross-query sharing.

## 3.2 Pipelined Execution

Although SCSD converts irregular set operations into hardwarefriendly vector operations, executing it as a standalone preprocessing stage still places its latency directly on the critical path. In a straightforward implementation, the shared and query-specific Top-� segments are fully constructed before the attention computation starts. The resulting execution

![](images/5c7329960a5f9d681215f8eee995cd14643c36497699674876a43fe5c404e29a.jpg)  
Figure 6. Illustration of pipelined SCSD execution within the attention pipeline.

time therefore becomes the sum of set-decomposition and attention latencies, which can substantially diminish the benefit obtained from cross-query KV sharing.

We observe that the attention kernel provides an opportunity to hide this overhead. Modern accelerators typically execute matrix-intensive and vector-intensive operations on separate compute pipelines. In sparse attention, operations such as �� and �� are dominated by matrix computation, whereas auxiliary operations, including softmax/update and our SCSD, mainly use vector resources. Importantly, during the steady-state attention pipeline, the vector-side workload is often shorter than the concurrent matrix computation, leaving otherwise underutilized vector execution slots. Since SCSD consists primarily of sorting, shifting, comparisons, and Boolean operations, it can naturally utilize these idle vector resources.

Based on this observation, we pipeline SCSD with the attention computation instead of executing all set decomposition upfront. The key idea is to construct each sparse segment only before it is consumed by the corresponding attention stage. We first compute a small number of initial Top-� segments to prime the pipeline and break the immediate producer–consumer dependency between SCSD and attention. Once the pipeline reaches the steady state, while the matrix pipeline performs �� or �� computation for the current sparse segment, the vector pipeline simultaneously constructs the shared or private segment required by a future attention stage. In this way, SCSD computation is shifted away from the critical path and overlapped with useful attention computation.

To further improve the overlap, we divide the construction of a Top-� segment into multiple finer-grained stages. A complete set-decomposition task may be too large to fit within a single vector-side slack window. We therefore split it into several smaller stages and schedule them across consecutive attention pipeline slots. Each stage consumes only a fraction of the available vector capacity, allowing the kernel to better utilize otherwise idle vector resources without delaying the matrix pipeline. After the warmup phase, SCSD and attention thus progress as a producer–consumer pipeline: attention consumes previously prepared sparse segments while SCSD constructs future segments in parallel.

This staged pipelining largely hides the additional bookkeeping introduced by cross-query sharing. Rather than paying $T _ { \mathrm { S C S D } } + T _ { \mathrm { A t t n } }$ sequentially, the efective SCSD latency is overlapped with the existing attention pipeline, making the

![](images/3900320ae3d4a0d4c748c65e40b8c529b86e51ae615b90c95cf2b594682fa814.jpg)  
Figure 7. A running example of cascaded sharing across multiple query-group granularities.

Table 1. Number of processed KV entries under diferent sharing strategies for the running example in Figure 7.
<table><tr><td>Strategy</td><td>Processed KVs</td><td>Reduction</td></tr><tr><td>No sharing</td><td>16</td><td></td></tr><tr><td>Group = 2</td><td>10</td><td>37.5%</td></tr><tr><td>Group = 4</td><td>10</td><td>37.5%</td></tr><tr><td>Cascaded</td><td>8</td><td>50.0%</td></tr></table>

additional set-decomposition overhead largely invisible on the critical path.

## 3.3 Cascaded Sharing

Although grouping neighboring queries enables cross-query KV reuse, using a fixed sharing granularity cannot fully ex ploit the heterogeneous overlap structure of sparse-attention workloads. A small query group typically exhibits a high Top-� overlap ratio, but misses reuse opportunities across diferent groups. In contrast, a large group covers a wider sharing scope, yet only KV entries selected by all queries in the group can be directly shared, causing many entries that are shared by only a subset of queries to fall back to redundant execution.

To address this limitation, we propose cascaded sharing, which hierarchically decomposes the sparse workload and captures KV reuse at multiple granularities. Starting from a relatively large query group (8 by default), we first extract the KV entries shared by all queries in the group and process them jointly. These shared entries are then removed from the Top-� sets of individual queries, and the remaining entries are recursively partitioned into smaller subgroups to identify additional reuse opportunities. The decomposition continues until no further sharing is possible, at which point the remaining entries are processed as query-specific residuals. In this way, each KV entry is processed at the largest query granularity over which it can be shared, avoiding the need to commit to a single fixed group size.

![](images/7968b17f0a7c597981569984998d56359c4795e55ea2a46e38bfc2d65bc3a691.jpg)  
Figure 8. The ranking distributions of query-specific KVs.

Figure 7 illustrates cascaded sharing with four neighboring queries whose Top-� sets are $S _ { 1 } ~ = ~ \{ A , B , C , E \} , S _ { 2 } ~ =$ $\{ A , B , C , F \} , S _ { 3 } = \{ A , B , D , G \}$ , and $S _ { 4 } ~ = ~ \{ A , B , D , H \}$ . Cascaded sharing first extracts the four-way intersection $\{ A , B \}$ which is loaded and dequantized once and reused by all four queries. After removing these entries, the residual workloads become {�, �}, {�, � }, {�, �}, and {�, �}, respectively. The four-query group is then divided into two subgroups, $\{ Q _ { 1 } , Q _ { 2 } \}$ and $\{ Q _ { 3 } , Q _ { 4 } \}$ , where the additional shared entries � and � are extracted and reused within each subgroup. Finally, only the query-specific residuals �, � , �, and � are processed independently.

Table 1 quantifies the benefit of such multi-granularity sharing in terms of the number of processed KV entries. With a fixed group size of two, the globally shared entries � and � are redundantly processed by both subgroups, resulting in 10 processed KV entries. Increasing the group size to four eliminates this inter-group redundancy, but fails to capture the subgroup-level sharing of� and �, again resulting in 10 processed KV entries. In contrast, cascaded sharing captures both four-way and two-way reuse and reduces the total number of processed KV entries to 8, equal to the number of distinct KV entries in this example.

This hierarchical organization therefore avoids committing to a single fixed sharing granularity and allows each KV entry to be processed at the largest query granularity over which it can be shared.

## 3.4 Tile-Aware Clipping

Cascaded sharing partitions the Top-� workload into mul tiple shared and query-specific segments whose sizes are generally not aligned with the native attention tile size. Consequently, even a few remaining KV entries may trigger an additional QK/PV tile and lead to poor hardware utilization. We therefore make cascaded sharing aware of the underlying tile granularity.

Let � denote the number of KV entries per tile and $| S | =$ $q T + r \left( 0 \leq r < T \right)$ for a shared segment �. We introduce a tile-utilization threshold � (50% by default). If $r \geq \tau T$ , we pad the tail to a full tile and process it at the current sharing level, masking the padded entries to preserve exact semantics. Otherwise, we push the tail down to the next cascade level, where it is merged with lower-level workloads to form betterutilized tiles. This push-down is exact and trades a small amount of sharing for higher tile utilization.

At the final query-specific level, no further push-down is possible. Figure 8 shows the rank distribution of these queryspecific KV entries after cascaded sharing. We observe that the remaining residual entries are predominantly concentrated near the tail of the original Top-� ranking, suggesting that highly ranked KV entries are more likely to have already been captured by shared segments at earlier cascade levels. Based on this observation, we perform importance-aware clipping: for $\left| R _ { i } \right| = q _ { i } T + r _ { i } .$ , we retain the �<sub>�</sub>� KV entries with the highest indexer scores and prune the $r _ { i }$ least important ones, thereby eliminating the final underutilized tile. This is the only approximate step in our tile-aware execution and minimizes accuracy degradation by discarding only low-importance residual entries.

## 4 Performance Evaluation

## 4.1 Experimental Setup

Testbed. We conduct our experiments on 16 Ascend 910C NPUs, each equipped with 64 GB of HBM. The NPUs are interconnected through a high-speed scale-up fabric, providing up to 784 GB/s bidirectional D2D bandwidth. We use CANN 9.1.0 as the underlying software stack for kernel development and execution. Although our evaluation is conducted on Ascend NPUs, our design does not rely on Ascend-specific architectural features. Instead, it builds on general characteristics of modern AI accelerators, including separate matrix and general-purpose/vector compute units, tile-based execution, and heterogeneous compute pipelines. Therefore, the proposed techniques can also be applied to GPUs, where similar functionality is provided by Tensor Cores and general-purpose SIMT cores.

Models and Workloads. We evaluate our design using two representative long-context LLMs, DeepSeek-3.2 [9] and GLM-5.3 [15], both of which employ model-native sparse attention. DeepSeek-3.2 uses a trainable Lightning Indexer in each attention layer to select the Top-2048 KV entries independently for each query. GLM-5.3 adopts a similar Top-2048 DSA mechanism, while further allowing neighboring layers to share indexer selections. For workloads, we use datasets from LongBench [2], covering a diverse set of longcontext tasks, which includes Question Answering, Document Summarization, and Code Generation. These workloads cover diverse input structures and attention behaviors, allowing us to evaluate the efectiveness of our design across diferent long-context scenarios.

Baseline and implementation. We compare QUILT against the optimized sparse-attention kernel implementation provided by OPS-Transformer [3], which represents a highly optimized implementation for Ascend NPUs. To ensure a fair comparison, we integrate both the OPS-Transformer baseline kernel and our workload-aware sparse-attention kernel into the same XYServe [27] inference engine. Both implementations therefore share the same model execution framework, request scheduling, memory management, and runtime environment, with the sparse-attention kernel being the only major diference. Unless otherwise specified, all reported kernel-level and end-to-end results are measured under this unified serving setup.

Table 2. Accuracy comparison of sparse-attention kernels across 21 LongBench datasets (%).
<table><tr><td rowspan="2">Dataset</td><td colspan="3">GLM-5.3</td><td colspan="3">DeepSeek-3.2</td></tr><tr><td>Baseline</td><td>QUILT</td><td>Δ</td><td>Baseline</td><td>QUILT</td><td>Δ</td></tr><tr><td>qmsum</td><td>22.48</td><td>22.60</td><td>+0.12</td><td>22.86</td><td>23.02</td><td>+0.16</td></tr><tr><td>gov_report</td><td>30.40</td><td>30.60</td><td>+0.20</td><td>29.85</td><td>29.60</td><td>-0.25</td></tr><tr><td>narrativeqa</td><td>32.20</td><td>31.36</td><td>-0.84</td><td>31.94</td><td>31.76</td><td>-0.18</td></tr><tr><td>samsum</td><td>45.81</td><td>46.09</td><td>+0.28</td><td>43.94</td><td>44.69</td><td>+0.75</td></tr><tr><td>dureader</td><td>28.64</td><td>28.89</td><td>+0.25</td><td>26.11</td><td>26.84</td><td>+0.73</td></tr><tr><td>lsht</td><td>57.50</td><td>57.50</td><td>0.00</td><td>47.00</td><td>47.75</td><td>+0.75</td></tr><tr><td>trec</td><td>78.50</td><td>78.00</td><td>-0.50</td><td>58.88</td><td>58.25</td><td>-0.63</td></tr><tr><td>2wikimqa</td><td>60.11</td><td>59.87</td><td>-0.24</td><td>66.43</td><td>67.57</td><td>+1.14</td></tr><tr><td>hotpotqa</td><td>63.76</td><td>64.18</td><td>+0.42</td><td>67.99</td><td>66.54</td><td>-1.45</td></tr><tr><td>passage_retrieval_zh</td><td>100.00</td><td>100.00</td><td>0.00</td><td>100.00</td><td>100.00</td><td>0.00</td></tr><tr><td>qasper</td><td>46.39</td><td>47.05</td><td>+0.66</td><td>49.57</td><td>49.80</td><td>+0.23</td></tr><tr><td>vcsum</td><td>12.30</td><td>12.19</td><td>-0.11</td><td>14.66</td><td>14.44</td><td>-0.22</td></tr><tr><td>multifieldqa_zh</td><td>62.11</td><td>62.80</td><td>+0.69</td><td>65.28</td><td>66.32</td><td>+1.04</td></tr><tr><td>triviaqa</td><td>91.07</td><td>91.60</td><td>+0.53</td><td>93.98</td><td>92.63</td><td>-1.35</td></tr><tr><td>passage_count</td><td>29.50</td><td>30.08</td><td>+0.58</td><td>7.50</td><td>8.50</td><td>+1.00</td></tr><tr><td>multifieldqa_en</td><td>50.45</td><td>51.45</td><td>+1.00</td><td>55.80</td><td>56.14</td><td>+0.34</td></tr><tr><td>lcc</td><td>74.35</td><td>73.44</td><td>-0.91</td><td>73.61</td><td>74.09</td><td>+0.48</td></tr><tr><td>musique</td><td>51.40</td><td>50.55</td><td>-0.85</td><td>51.35</td><td>54.06</td><td>+2.71</td></tr><tr><td>repobench-p</td><td>69.69</td><td>70.22</td><td>+0.53</td><td>42.54</td><td>41.39</td><td>-1.15</td></tr><tr><td>passage_retrieval_en</td><td>100.00</td><td>100.00</td><td>0.00</td><td>98.50</td><td>99.50</td><td>+1.00</td></tr><tr><td>multi_news</td><td>26.84</td><td>26.74</td><td>-0.10</td><td>23.38</td><td>23.26</td><td>-0.12</td></tr></table>

## 4.2 Main Results

## 4.2.1 Accuracy Retainment.

We first evaluate the accuracy impact of our workload-aware sparse-attention execution. We compare XYServe using the original OPS-Transformer sparse kernel with XYServe using QUILT across 21 LongBench datasets, following the oficial evaluation metric for each dataset. The results are shown in Table 2. For both GLM-5.3 and DeepSeek-3.2, our kernel achieves nearly identical accuracy to the baseline, with only negligible diferences across individual tasks and in the average score. Specifically, across all datasets, the mean absolute score diference between QUILT and the baseline is only 0.75 points for DeepSeek-3.2 and 0.42 points for GLM-5.3. Relative to the baseline scores, the corresponding mean absolute percentage diferences are 2.01% and 0.90%, respectively. These results confirm that our exact execution optimizations preserve the original sparse attention semantics, while the importance-aware clipping in our tile-aware execution introduces only negligible accuracy variance.

![](images/78fd83f72ace42a58ac74e3c3a009cfe6a22a8a3a9cb46f472b22c237ddfac56.jpg)  
Figure 9. Average sparse-attention kernel latency reduction achieved by QUILT over OPS-Transformer on Qasper (QA), GovReport (GO), Musique (MU), and NarrativeQA (NA) from LongBench under SP and TP.

Table 3. Reduction in processed KV entries achieved by QUILT over OPS-Transformer on Qasper (QA), GovReport (GO), Musique (MU), and NarrativeQA (NA).
<table><tr><td>Workload</td><td>GLM-5.3</td><td>DeepSeek-3.2</td></tr><tr><td>QA</td><td>53.8%</td><td>39.6%</td></tr><tr><td>GO</td><td>55.9%</td><td>31.9%</td></tr><tr><td>MU</td><td>54.6%</td><td>25.2%</td></tr><tr><td>NA</td><td>54.3%</td><td>24.8%</td></tr></table>

## 4.2.2 Average Kernel Latency Reduction.

We next evaluate the kernel-level execution eficiency of QUILT on GLM-5.3 and DeepSeek-3.2. We compare QUILT against the OPS-Transformer sparse kernel on four representative LongBench workloads, including Qasper (QA) [8], GovReport (GO) [16], Musique (MU) [30], and NarrativeQA (NA) [19], under both tensor parallelism (TP) and sequence parallelism (SP). Figure 9 reports the reduction in average prefill-stage sparse-attention kernel latency achieved by QUIL relative to OPS-Transformer. QUILT consistently reduces kernel latency across both models, all four workloads, and both parallel configurations. Specifically, for GLM-5.3, under TP, our kernel reduces the average sparse-attention latency by 53.2%, 55.1%, 54.5%, and 52.4% on the four datasets, respectively, corresponding to an average reduction of 53.8%. Under SP, the reductions are 44.6%, 45.0%, 42.3%, and 41.9%, with an average reduction of 43.4%. For DeepSeek-3.2, our kernel achieves latency reductions of 37.9%, 29.6%, 24.5%, and 22.8% under TP and 25.9%, 17.5%, 9.9%, and 15.0% under SP, corresponding to average reductions of 28.7% and 17.1%, respectively.

The reductions in kernel latency are accompanied by substantial decreases in KV processing. Table 3 reports the reduction in processed KV entries for each workload and model. Since TP and SP only change how the query workload is partitioned across devices, they do not afect the total number of KV entries processed by QUILT. Across the four workloads, QUILT reduces the number of processed KV entries by 54.7% on average for GLM-5.3 and 30.4% for DeepSeek-3.2. These results demonstrate that our design provides robust kernel-level improvements across diferent model architectures, workloads, and parallelization strategies by exploiting cross-query KV reuse and improving hardware utilization.

![](images/e1dc270c5ffe75808a40a405abb6e4ed32da270b12c16fdc5d281b08c177164f.jpg)

![](images/0aad45b05ea2eeaddb7264866957796a008afe5abfe0e058d30411e5fc7fbb89.jpg)

Figure 10. Average sparse-attention kernel latency with varying sharing lengths under SP and TP, normalized to OPS-Transformer.  
![](images/7e71ebe5b2bd68f11b71dc5233729ca8ad62fd452d57f81063f184281bb48cfe.jpg)

![](images/d2744c5eb5d24e9730f83a9c0b4db64c45536afc82ab53539a70a6a082e43e67.jpg)  
Figure 11. Average time-to-first-token latency (TTFT) reduction achieved by QUILT over OPS-Transformer on Qasper (QA), GovReport (GO), Musique (MU), and NarrativeQA (NA) from LongBench under SP and TP.

## 4.2.3 Performance Sensitivity to Sharing Length.

We further quantify the relationship between cross-query sharing ratio and the kernel performance using synthetic sparse-attention workloads. Specifically, we synthetically construct Top-� index lists with controlled sharing lengths among neighboring queries while keeping the overall Top-� size unchanged. This allows us to isolate the impact of cross-query sharing from other workload characteristics. We vary the sharing length and measure the average sparseattention kernel latency of QUILT and OPS-Transformer for both GLM-5.3 and DeepSeek-3.2 under tensor parallelism (TP) and sequence parallelism (SP). Figure 10 shows that the kernel latency consistently decreases as the sharing length increases across both models and parallel configurations. With more shared KV entries, our workload-aware execution can eliminate more redundant KV processing and reuse the corresponding data across neighboring queries. Specifically, increasing the sharing length from 0 to 2048 reduces the kernel latency by 42.74% and 33.25% for GLM-5.3 under TP and SP, respectively, and by 41.03% and 29.53% for DeepSeek-3.2. We also observe a small latency overhead when cross-query sharing is limited, mainly due to the set-decomposition work performed before the attention pipeline is fully overlapped.

![](images/e88e6400e1af25f544f64e50a0ecf6b25fb031e71411142c1d86e8f545762595.jpg)

![](images/1ac0f4d52a1d4891ccf64dd6d9de041d11c99cacad11665d8b977f9b18552526.jpg)  
Figure 12. Average sparse-attention kernel latency under TP and SP, normalized to OPS-Transformer. Naive uses a Torch-based set-intersection implementation; Vectorized replaces it with SCSD; Pipeline further enables pipelined SCSD execution.

Table 4. Average number of processed KV entries with fixed query-group sizes and cascaded sharing (CS), normalized to OPS-Transformer.
<table><tr><td>Strategy</td><td>GLM-5.3</td><td>DeepSeek-3.2</td></tr><tr><td>Group 2</td><td>65.7%</td><td>78.9%</td></tr><tr><td>Group 4</td><td>57.2%</td><td>82.9%</td></tr><tr><td>Group 8</td><td>59.0%</td><td>88.4%</td></tr><tr><td>CS</td><td>45.5%</td><td>70.1%</td></tr></table>

This overhead is more visible under SP because sequence partitioning shortens the per-kernel sequence length, leaving less computation to amortize the pipeline warm-up cost. As sharing increases, the reuse benefits quickly dominate this fixed overhead. These results confirm that the performance gains of QUILT increase with the amount of cross-query sharing, as greater reuse progressively outweighs the fixed set-processing overhead.

## 4.2.4 Average TTFT Reduction.

We further evaluate whether the kernel-level improvements of QUILT translate into end-to-end prefill speedups on GLM-5.3 and DeepSeek-3.2. We compare QUILT with OPS-Transformer in XYServe on the same four LongBench workloads under both TP and SP, and report the reduction in average timeto-first-token latency (TTFT) relative to OPS-Transformer in Figure 11. QUILT consistently reduces TTFT across both models, all evaluated workloads, and both parallel configurations. For GLM-5.3, the latency reductions under TP are 34.6%, 36.5%, 36.8%, and 34.5%, corresponding to an average reduction of 35.6%. Under SP, the corresponding reductions are 7.0%, 11.0%, 11.9%, and 11.7%, averaging 10.4%. DeepSeek 3.2 exhibits a similar trend, with reductions of 30.2%, 23.7%, 20.0%, and 18.1% under TP, and 3.4%, 3.6%, 2.4%, and 4.3% under SP, corresponding to average reductions of 23.0% and 3.4%, respectively. These end-to-end results confirm that the benefits of our workload-aware kernel generalize across different sparse-attention models and translate into substantial reductions in user-visible prefill latency.

![](images/4e07f2375a5a8e3ea05a1986b75f6f6ce8f697787310459570834ce36ad87c4c.jpg)

![](images/4cfd8f2b97415f22f4e8e8dd3113f831f29e94056cac0ed4da5dc00e5ef5130b.jpg)  
Figure 13. Average sparse-attention kernel latency of cascaded sharing (CS) and fixed query-group sizes of 2, 4, and 8 under TP and SP, normalized to OPS-Transformer.

## 4.3 Ablation Studies

We conduct a series of ablation studies on requests from the same four LongBench workloads, namely Qasper (QA), GovReport (GO), Musique (MU), and NarrativeQA (NA), to quantify the individual contributions of the key optimizations in QUILT, including Shift-and-Compare Set Decomposition (SCSD), Pipelined Execution, Cascaded Sharing, and Tile-Aware Clipping.

4.3.1 Efectiveness of SCSD and Pipelined Execution. We first evaluate the efectiveness of SCSD and pipelined execution by incrementally enabling them on top of the OPS-Transformer baseline. We compare four variants: (1) the original baseline kernel without cross-query sharing; (2) Naive, which enables group-based query execution but computes set intersections using a naive Torch-based implementation; (3) Vectorized, which replaces the naive set operations with the proposed SCSD; and (4) Pipeline, the full design that further overlaps SCSD with attention computation. We evaluate all variants on GLM-5.3 and DeepSeek-3.2 across the same four LongBench workloads under both TP and SP. The results are shown in Figure 12.

The naive implementation incurs substantial overhead across both models and parallel configurations. For GLM-5.3, it is 21.84× and 33.65× slower than OPS-Transformer under TP and SP, respectively. Replacing the naive set operations with SCSD reduces the kernel latency by 96.1% under TP and 97.3% under SP, while further enabling pipelined execution provides additional reductions of 45.9% and 36.9%, respectively. DeepSeek-3.2 exhibits a similar trend. The naive implementation is 22.18× and 27.53× slower than the baseline under TP and SP, respectively; SCSD reduces the latency by 95.8% and 96.4%, and pipelined execution further reduces it by 22.9% and 16.2%, respectively. Overall, the full design outperforms the baseline by 35.5% on average across both models and both parallel configurations. The consistent improvements demonstrate that eficient set decomposition and pipeline overlap are both necessary to translate cross-query sharing into actual performance gains.

## 4.3.2 Efectiveness of Cascaded Sharing.

![](images/25ff4edaba949ac6aa0ac9f7647e1d3b640134672b608d29fe0dc66fce38354e.jpg)

![](images/4d3d40b1bdef8e3a1e1f44d25b49171de64e4b56f0936e708280a5c6762de26b.jpg)

Figure 14. Average sparse-attention kernel latency with tile-aware clipping under TP and SP, normalized to the corresponding execution without clipping.  
![](images/533d46838cc7bece73a791a692561c5c3db9876bcdeab63c640f2f2669da3918.jpg)

![](images/b964b57a8b5cfcbb457d493289434147e9568740318d25119249c132567aad50.jpg)  
Figure 15. Cosine similarity between the attention output tensors produced by random and importance-aware clipping and the corresponding unclipped reference.

We next evaluate the efectiveness of cascaded sharing by comparing it with fixed query-group sizes of 2, 4, and 8. All variants use the same SCSD and pipelined execution and difer only in how cross-query sharing is organized. We evaluate GLM-5.3 and DeepSeek-3.2 on the same four Long-Bench workloads under both TP and SP. Figure 13 reports the normalized sparse-attention kernel latency, while Table 4 reports the corresponding number of processed KV entries, also normalized to OPS-Transformer.

For GLM-5.3, cascaded sharing reduces the number of processed KV entries by 30.8%, 20.6%, and 22.9% compared with fixed group sizes of 2, 4, and 8, respectively. These reductions translate into kernel-latency improvements of 29.6%, 19.2%, and 21.6% under TP, and 18.8%, 6.8%, and 10.9% under SP. DeepSeek-3.2 shows a similar trend: cascaded sharing reduces processed KV entries by 11.2%, 15.4%, and 20.7% over group sizes of 2, 4, and 8, respectively, while reducing kernel latency by 9.7%, 14.0%, and 19.4% under TP and by 0.5%, 4.5%, and 11.0% under SP. Cascaded sharing captures both by hierarchically exploiting reuse at multiple granularities, converting more cross-query Top-� overlap into reduced data movement and lower execution latency.

## 4.3.3 Efectiveness of Tile-Aware Clipping.

Latency Comparison. We finally evaluate tile-aware clipping from both performance and numerical-fidelity perspectives. For performance, we compare the full QUILT design against an exact variant that retains all query-specific residual KV entries and therefore performs no clipping. Both variants otherwise use the same SCSD, cascaded sharing, and pipelined execution. As shown in Figure 14, tile-aware clipping consistently reduces sparse-attention kernel latency under both TP and SP. For GLM-5.3, the latency reductions are 23.71% under TP and 22.68% under SP, while DeepSeek-3.2 achieves reductions of 24.44% and 14.13%, respectively. These results demonstrate that eliminating poorly utilized tail tiles provides consistent performance benefits across both models and parallel configurations.

Output Similarity Comparison. We next examine whether importance-aware clipping preserves the attention output more efectively than arbitrary pruning with GLM-5.3. Using the unclipped execution as the reference, we compare two strategies that remove the same number of KV entries from the query-specific residuals: random clipping, which removes residual entries uniformly at random, and our importanceaware clipping, which removes entries with the lowest indexer scores. We measure the cosine similarity between the clipped and unclipped attention outputs as

$$
\mathrm { S i m } ( O , \hat { O } ) = \frac { \langle O , \hat { O } \rangle } { \| O \| _ { 2 } \| \hat { O } \| _ { 2 } } .
$$

where � and $\hat { O }$ denote the corresponding unclipped and clipped output vectors, respectively. We evaluate this similarity at two granularities. For the head-level metric, cosine similarity is computed independently for each �-dimensional attention-head output. For the hidden-level metric, all � heads are flattened into a single ��-dimensional vector before computing cosine similarity. For both metrics, we report the mean and minimum similarity across the evaluated outputs, capturing the overall numerical fidelity and the worst-case deviation, respectively.

As shown in Figure 15, importance-aware clipping consistently preserves the attention output more accurately than random clipping. Specifically, importance-aware clipping achieves a head-level mean cosine similarity of 1.0000, compared with 0.9999 for random clipping, while improving the head-level minimum similarity from 0.312 to 0.406. At the hidden level, the mean similarity increases from 0.9998 to 1.0000, and the minimum similarity improves from 0.664 to 0.719. These results show that pruning residual KV entries according to their indexer scores substantially reduces the numerical perturbation caused by clipping, allowing QUILT to eliminate underutilized tail tiles while closely preserving the unclipped attention output.

## 5 Related Work

Sparse attention for prefill. A growing body of work exploits attention sparsity to accelerate the prefill phase of long-context LLM inference [9, 12, 13, 17, 20, 33]. MInference [17] identifies several recurring sparse attention patterns and dynamically constructs sparse indices for eficient prefill computation. FlexPrefill [20] further adapts the sparse pattern and computation budget to individual inputs and attention heads, while XAttention [33] employs lightweight scoring to identify important attention blocks. SeerAttention [13] replaces hand-crafted sparsity patterns with a learnable mechanism for predicting block-level attention sparsity. Diferent from these post-hoc or inference-time sparsification approaches, DeepSeek Sparse Attention (DSA) [9] incorporates fine-grained sparse attention directly into model training and employs a lightweight indexer to dynamically select important KV entries during both training and inference. Following DSA, model-native sparse attention has been increasingly adopted and extended by recent LLMs, including the GLM series [15], Hunyuan series [29], and DeepSeek-4 [10, 11]. Meanwhile, systems such as FlashPrefill [12] further improve sparse prefill through hardwareaware pattern discovery and execution. These approaches focus on determining which KV entries should participate in attention or on eficiently executing the sparse workload of individual queries. In contrast, QUILT operates on the resulting query-specific sparse KV sets and exploits sharing across neighboring queries, which is orthogonal to these sparsification techniques, and can be applied on top of their designs to further reduce redundant cross-query KV accesses and computation.

Sparse attention for decoding. A large body of work exploits attention sparsity to reduce the KV-cache access cost during autoregressive decoding. $_ \mathrm { H _ { 2 } O }$ [37], StreamingLLM [32], Scissorhands [24], FastGen [14], and SnapKV [22] compress the KV cache by retaining only tokens considered important for future attention computation, thereby reducing both memory consumption and attention cost. To avoid permanently discarding potentially useful KV entries, dynamic sparse-attention methods such as Quest [28], InfLLM [31], and ArkVale [4] dynamically select query-relevant KV entries at runtime while retaining or recovering the full historical context. Several systems further exploit the temporal locality of sparse KV accesses across consecutive decoding steps. RetroInfer [5], LiteCache [35], and SparseServe [39] leverage the high similarity of KV selections between neighboring decoding queries to cache frequently accessed KV entries in HBM, thereby reducing costly KV transfers between host DRAM and HBM. These approaches primarily exploit cross-query locality at the memory-management level during decoding. In contrast, our QUILT targets concurrently processed queries during prefill and exploits their overlapping sparse workloads inside the attention kernel, eliminating redundant KV movement across the HBM–onchip-memory boundary as well as redundant dequantization and computation.

Hardware-eficient attention kernels. A rich line of work optimizes attention kernels by better exploiting the memory hierarchy and parallel compute resources of modern accelerators [6, 7, 26, 34]. FlashAttention [7] introduces an IO-aware tiled attention algorithm that keeps intermediate attention states in on-chip memory and employs online softmax to avoid materializing the full attention matrix in HBM, substantially reducing HBM accesses. FlashAttention-2 [6] further improves hardware utilization through better work partitioning across thread blocks and warps, increased parallelism, and reduced non-matrix-multiplication overhead. FlashAttention-3 [26] exploits emerging GPU features such as asynchronous matrix computation and data movement, using warp specialization and pipelining to overlap memory accesses, matrix multiplication, and softmax computation, while also supporting low-precision execution. Flash-Infer [34] provides optimized and customizable attention kernels for diverse LLM inference workloads and KV-cache layouts. These works establish general principles for efi cient attention execution, including maximizing on-chip data reuse, improving compute-unit utilization, and overlapping computation with data movement. Our QUILT builds upon these principles but targets a complementary source of inefficiency specific to sparse attention: it exploits the sharing structure of Top-K KV selections across neighboring queries to expose additional cross-query data reuse that conventional attention kernels do not capture.

## 6 Conclusion

In this work, we revisit sparse-attention execution for longcontext LLM prefill and identify substantial cross-query KV reuse that is largely overlooked by existing workloadagnostic, query-parallel execution. Based on this observation, we propose QUILT, a workload-aware sparse-attention execution mechanism that jointly processes correlated queries to exploit cross-query sharing. QUILT introduces Shift-and-Compare Set Decomposition (SCSD) for eficient data-parallel set processing, pipelines SCSD with attention computation to hide its overhead, and employs cascaded sharing to capture reuse across multiple query granularities. It further incorporates tile-aware execution to improve hardware utilization, together with importance-aware clipping to retain high accuracy. Evaluations on GLM-5.3 and DeepSeek-3.2 show that QUILT consistently reduces sparse-attention kernel latency and translates these improvements into substantial end-toend prefill speedups across diverse long-context workloads and parallel configurations. Overall, our results demonstrate the importance of exploiting cross-query workload structure, rather than treating sparse queries independently, for eficient long-context LLM inference.

## References

[1] Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024. L-Eval: Instituting Standardized Evaluation for Long Context Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024. Association for Computational Linguistics, 14388–14411. doi:10.18653/V1/2024.ACL-LONG.776

[2] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 3119–3137. doi:10.18653/V1/2024.ACL-LONG.172

[3] CANN. 2025. OPS-Transformer: Advanced Transformer Operator Library for Ascend. htps://github.com/hicann/ops-transformer. Accessed: 2026.

[4] Renze Chen, Zhuofeng Wang, Beiquan Cao, Tong Wu, Size Zheng, Xiuhong Li, Xuechao Wei, Shengen Yan, Meng Li, and Yun Liang. 2024. ArkVale: Eficient Generative LLM Inference with Recallable Key-Value Eviction. In Advances in Neural Information Processing Systems. htps://dblp.org/rec/conf/nips/ChenWCW0LWYL024

[5] Yaoqi Chen, Jinkai Zhang, Baotong Lu, Qianxi Zhang, Chengruidong Zhang, Jing Liu, Jingjia Luo, Di Liu, Huiqiang Jiang, Qi Chen, Bailu Ding, Xiao Yan, Jiawei Jiang, Chen Chen, Mingxing Zhang, Cheng Li, Yuqing Yang, Fan Yang, and Mao Yang. 2026. RetroInfer: A Vector Storage Engine for Scalable Long-Context LLM Inference. Proc. VLDB Endow. 19, 5 (2026), 1016–1031. htps://dblp.org/rec/journals/pvldb/ ChenZLZZLLLJCDYJCZLYYY26

[6] Tri Dao. 2024. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In The Twelfth International Conference on Learning Representations, ICLR 2024. htps://dblp.org/rec/conf/iclr Dao24

[7] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022. htps://dblp.org/rec/conf/nips/DaoFERR22

[8] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 4599–4610.

[9] DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. CoRR abs/2512.02556 (2025). doi:10.48550/ ARXIV.2512.02556

[10] DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Eficient Million-Token Context Intelligence. CoRR abs/2606.19348 (2026). arXiv:2606.19348 doi:10.48550/ARXIV.2606.19348

[11] DeepSeek-AI. 2026. DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression. CoRR abs/2609.19969 (2026). arXiv:2609.19969

[12] Qihang Fan, Huaibo Huang, Zhiying Wu, Juqiu Wang, Bingning Wang, and Ran He. 2026. FlashPrefill: Instantaneous Pattern Discovery and Thresholding for Ultra-Fast Long-Context Prefilling. CoRR abs/2603.06199 (2026). doi:10.48550/ARXIV.2603.06199

[13] Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden K. H. So, Ting Cao, Fan Yang, and Mao Yang. 2025. SeerAttention: Self-distilled Attention Gating for Eficient Long context Prefilling. In Advances in Neural Information Processing Systems 39: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025. htps://dblp.org/rec/conf/nips/GaoZDCZQLSCYY25

[14] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2024. Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs. In The Twelfth International Conference on Learning Representations, ICLR 2024. htps://dblp.org/rec/conf/iclr/ Ge0LZ0024

[15] GLM-5 Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, et al. 2026. GLM-5: from Vibe Coding to Agentic Engineering. arXiv:2602.15763 [cs.CL]

[16] Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. 2021. Eficient attentions for long document summarization. In Proceedings of the 2021 conference of the north American chapter of the association for computational linguistics: Human language technologies. 1419–1436.

[17] Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xu fang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. MInference 1.0: Ac celerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024. htps://dblp.org/rec/conf/nips/JiangLZWLAHA0L024

[18] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-world Github Issues?. In The Twelfth International Conference on Learning Representations, ICLR 2024. htps: //dblp.org/rec/conf/iclr/JimenezYWYPPN24

[19] Tomáš Kočisky, Jonathan Schwarz, Phil Blunsom, Chris Dyer,\` Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The narrativeqa reading comprehension challenge. Transactions of the Association for Computational Linguistics 6 (2018), 317–328.

[20] Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. 2025. FlexPrefill: A Context-Aware Sparse Attention Mechanism for Eficient Long-Sequence Inference. In The Thirteenth International Conference on Learning Representations, ICLR 2025. OpenReview.net. htps:// openreview.net/forum?id=OfjIlbelrT

[21] Jiashi Li and Shengyu Liu. 2025. FlashMLA: Eficient Multi-head Latent Attention Kernels. htps://github.com/deepseek-ai/FlashMLA.

[22] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Lo catelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. SnapKV: LLM Knows What You are Looking for Before Generation. In Advances in Neural Information Processing Systems. htps://dblp.org/rec/conf/nips/LiHYVLYCLC24

[23] Tianyang Liu, Canwen Xu, and Julian J. McAuley. 2024. RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems. In The Twelfth International Conference on Learning Representations, ICLR 2024. htps://dblp.org/rec/conf/iclr/0003XM24

[24] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. CoRR abs/2305.17118 (2023). doi:10.48550/ARXIV.2305.17118

[25] Joon Sung Park, Joseph C. O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, UIST 2023. ACM, 2:1–2:22. doi:10.1145/3586183.3606763

[26] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. 2024. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024. htps://dblp.org/rec/ conf/nips/ShahBZTRD24

[27] Mingcong Song, Xinru Tang, Fengfan Hou, Jing Li, Wei Wei, Yipeng Ma, Runqiu Xiao, Hongjie Si, Dingcheng Jiang, Shouyi Yin, Yang Hu,

and Guoping Long. 2026. XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads. In Proceedings ofthe 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, ASPLOS 2026. ACM, 314– 329. doi:10.1145/3760250.3762228

[28] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. 2024. QUEST: Query-Aware Sparsity for Eficient Long-Context LLM Inference. In Proceedings of the 41st International Conference on Machine Learning, ICML 2024. 47901–47911. htps: //dblp.org/rec/conf/icml/TangZZXKH24

[29] Tencent Hunyuan. 2026. Hy4-preview. htps://github.com/Tencent-Hunyuan/Hy4-preview. Accessed: 2026.

[30] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Trans. Assoc. Comput. Linguistics 10 (2022), 539–554. doi:10.1162/TACL\_A\_00475

[31] Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. 2024. InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Eficient Context Memory. In Advances in Neural Information Processing Systems. htps://dblp.org/rec/conf/nips/XiaoZ0XLZ0024

[32] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Eficient Streaming Language Models with Attention Sinks. In The Twelfth International Conference on Learning Representations, ICLR 2024. htps://dblp.org/rec/conf/iclr/XiaoTCHL24

[33] Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. 2025. XAttention: Block Sparse Attention with Antidiagonal Scoring. CoRR abs/2503.16428 (2025). doi:10.48550/ARXIV.2503.16428

[34] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. FlashInfer: Eficient and Customizable Attention Engine for LLM Inference Serving. In Proceedings ofMachine Learning and Systems 7, MLSys 2025. htps://dblp.org/rec/ conf/mlsys/00010LLZW0KGKC25

[35] Jiawei Yi, Ping Gong, Youhui Bai, Zewen Jin, Shengnan Wang, Jiaqi Ruan, Jia He, Jiaan Zhu, Pengcheng Wang, Haibo Wang, Weiguang Wang, Xia Zhu, and Cheng Li. 2026. LiteCache: A Query Similarity-Driven, GPU-Centric KVCache Subsystem for Eficient LLM Inference. arXiv:2511.14510 [cs.LG]

[36] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. 2025. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, ACL 2025, Volume 1: Long Papers. 23078–23097. doi:10.18653/V1/2025.ACL-LONG.1126

[37] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2O: Heavy-Hitter Oracle for Eficient Generative Inference of Large Language Models. In Advances in Neural Information Processing Systems. htps://dblp. org/rec/conf/nips/Zhang00CZC0TRBW23

[38] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing Large Language Models with Long-Term Memory. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024. AAAI Press, 19724–19731. doi:10.1609/AAAI.V38I17. 29946

[39] Qihui Zhou, Peiqi Yin, Pengfei Zuo, and James Cheng. 2025. SparseServe: Unlocking Parallelism for Dynamic Sparse Attention in Long-Context LLM Serving. CoRR abs/2509.24626 (2025). doi:10.48550/ ARXIV.2509.24626