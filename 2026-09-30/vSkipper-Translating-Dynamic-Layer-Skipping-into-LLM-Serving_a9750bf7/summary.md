---
title: "vSkipper-Translating-Dynamic-Layer-Skipping-into-LLM-Serving"
source: https://arxiv.org/pdf/2609.37062v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:35:53"
field: "LLM 推理系统优化"
keywords: ["LLM serving", "dynamic layer skipping", "inference optimization", "system virtualization", "continuous batching"]
innovations: ["提出 RUN/Project-Only 虚拟化接口，将动态层跳过策略以插件形式接入现代推理引擎", "设计混合深度 cohort 执行机制，在捕获图中支持 per-token 不同深度的正确执行", "基于 roofline 模型推导盈利能力阈值，实现 prefetch-aware 的模式切换"]
benchmarks: ["GSM8K", "BBH", "CoQA"]
---

# 论文速读：vSkipper: Translating Dynamic Layer Skipping into LLM Serving Gains

## 一句话总结
vSkipper 是一个服务化虚拟化层，将逐 token 的动态层跳过（dynamic layer skipping）无缝集成到现代 LLM 推理引擎（SGLang）中，通过 RUN/Project-Only 接口和混合深度执行，在保持连续批处理、KV 缓存和捕获图完整性的同时，将 FlexiDepth 等跳层方法的节省计算量转化为实际的延迟降低与吞吐提升。

## 研究问题与动机
- 现有动态层跳过方法（如 FlexiDepth）虽然在研究级 PyTorch 路径上可节省 FLOPs，但在现代推理引擎中无法转化为服务效率增益；实验复现显示 FlexiDepth 在标准生成循环中解码吞吐反而下降 14.6%–21.0%。
- 根本原因：现代推理引擎（SGLang、vLLM 等）依赖规则批处理执行、连续批处理、paged KV 缓存和捕获 GPU 图，假设批次内所有 token 遍历相同层；动态跳过打破这一规律性。
- 节省的 FLOPs 不等于节省的时间：在 batch=1 时路由开销抵消了节省的工作量；在 batch=8 时批次仍需遍历 token 所需层的并集，拆分批次牺牲了批宽度。
- 现有跳层方法与引擎调度器、内存管理之间缺少接口层，导致无法复用现有的 continuous batching、prefix reuse 等优化。

## 核心贡献（创新点）
1. **RUN/Project-Only 虚拟化接口**：将逐 token 跳层策略以插件形式接入现代推理引擎，无需重写调度器或内存管理器，使跳层器成为可插拔组件。与已有工作相比，FlexiDepth 等使用专用 PyTorch 路径，而 vSkipper 通过统一接口使跳层器可在 SGLang 等引擎中直接运行。
2. **混合深度批次执行（Route-Guided Cohort Execution）**：在捕获的 CUDA 图中按路由决策将 token 打包为 RUN 和 Project-Only 两个 cohort，分别执行全层计算和仅投影操作，再通过 device-side gather/scatter 散回原批次位置，保留 paged KV 缓存和 prefix cache 正确性。
3. **盈利能力感知模式切换（Profitability-Aware Mode Switching）**：基于 roofline 模型推导阈值公式，仅在预期节省超过固定路由开销时才启用 routed 模式；decode 模式可在 prefill 后从 dense 单向升级为 routed，避免在不必要的负载下引入额外开销。
4. **受控的公平评估**：在完全匹配的 prompt、到达分布、输出长度和 launch 设置下对比 vSkipper 与上游 SGLang，分离 checkpoint 质量损失与服务效应；同时在 9 种合成跳层策略、2 个 Qwen3 跳层器、3 种工作负载和 3 款 GPU 上验证泛化性。

## 方法详解
- **RUN/Project-Only 接口**：跳层器对每个 token 在每个被路由层输出 RUN（正常执行）或 Project-Only（仅执行投影）决策；Project-Only 行跳过 attention 和 FFN，但仍通过 skipper 提供的 projector 写出本层 KV 状态。
- **路由磁带（Route Tape）**：每层的决策写入设备端 route tape（per-row mask），captured graph 直接将其作为数据消费，路由可从 step 到 step 变化而无需 host 同步或重新捕获图。
- **打包 cohort 执行**：从 tape 导出两个 cohort（RUN cohort 大小 $n_r$、Project-Only cohort 大小 $n_p$），行 gather 折叠进 operand load，加权 scatter 折叠进 epilogue；GEMM 的 bound 是 cohort 大小而非整个 batch 大小，attention kernel 则通过 mask 跳过 Project-Only 行的 KV 读取。
- **不变量保证**：(1) 每层必须写出自己的 KV——Project-Only 行仍执行 key/value 投影写入 KV cache，保证 paged KV 和 prefix reuse 的语义不变；(2) 行元数据永不陈旧——所有基于 route 的 map（attention metadata、page mapping、gather/scatter map）均由当前 tape 在 device 端实时派生。
- **盈利能力切换阈值**：Decode 模式下，当 resident token 数 $V > V^* = \frac{\tau \cdot \mathrm{BW}}{s L_r b}$ 时路由有利（A100 上 $V^* \approx 158\text{k}$ tokens；enter 阈值 200k，exit 阈值 160k）。Prefill 模式：prompt ≥ 1,536 tokens 且估计 Project-Only 比例 > 0.35 时启用路由，每 64 个 admitted batch 探测一次。
- **Capture 兼容**：dense 和 routed 两种模式均在固定 batch size（最大 1,024 rows on A100）下捕获图；routed 图中额外包含一个 device-side conditional CUDA-graph node，当本步无 token 跳过时走全 RUN 快速路径。

## 实验与结果
- **实验设置**：模型 Llama-3-8B（FlexiDepth checkpoint，路由最后 16/32 层）和自训练的 Qwen3-4B/8B 跳层器；工作负载 GSM8K（3,600 requests，116 output tokens）、BBH（4,000 requests，255 output tokens）、CoQA（4,000 requests，164 output tokens）；GPU：A100-SXM4-80GB、H100 80GB、RTX A6000 48GB。
- **主要结果（A100，knee 处 $0.95 \times Q^*$）**：
  - GSM8K：mean E2E 降低 **36.8%**（p99 -37.3%），TTFT 降低 66.0%，makespan 降低 1.4%。
  - BBH：mean E2E 降低 **13.6%**（p99 -11.4%），TPOT 降低 14.6%。
  - CoQA：mean E2E 降低 4.0%（不显著），但 makespan 降低 7.0%。
- **饱和吞吐（$1.25 \times Q^*$）**：GSM8K RPS 提升 **11.3%**（TPS +10.4%），BBH RPS 提升 **7.4%**（TPS +4.1%），CoQA 无显著变化。
- **质量评估（2×2 DiD 设计）**：serving 引入的质量损失在统计上 unresolved——GSM8K DiD = $-0.23 \pm 0.68$ pp，BBH DiD = $+0.45 \pm 0.55$ pp，CoQA DiD = $+0.07 \pm 0.12$ pp，区间均包含零。
- **泛化性**：9 种 RandomSkip 合成策略下，per-policy 阈值设置使最优臂达到 E2E 降低 51.0%；Qwen3-8B skip=0.416 时 E2E 降低 6.6% 且质量无损；H100 knee 处 E2E 降低 39.4%，RTX A6000 knee 处降低 30.0%，均无需 workload-specific 调参。
- **机制消融**：GSM8K 的收益主要来自 prefill 路由（单独 prefill-only 臂 -35.6%），BBH 的收益来自 decode 路由（-16.0%），CoQA 的 tail 请求加速使得 makespan 改善 7.0–8.1%。

## 相关工作脉络
- **FlexiDepth [15]**：提出 per-token 动态层跳过，添加轻量级路由器，但仅在 PyTorch 研究路径上报告 FLOPs 节省，未集成到现代 serving engine；vSkipper 在其 checkpoint 之上提供 serving 接口。
- **AdaSkip [10] / DASH [25]**：自适应子层或整层跳过，同样使用专用执行路径，缺乏与 continuous batching、paged KV 缓存的集成。
- **Early-exit 系统（EE-LLM [4], LayerSkip [8], DREX [13], Apparate [7]）**：在特定层提前退出，需重计算、多深度缓存或跨深度状态传输；vSkipper 保留完整 depth 但跳过中间层，满足 Invariant 1（每层写 KV）。
- **MoE 系统 [21] / Multi-adapter LoRA 服务 [3, 11, 22]**：在层内路由 token 到不同 expert/adapter，但仍遍历全部层；vSkipper 路由 across depth，保持 batch 和捕获图不变。
- **Serving 引擎（SGLang [28], vLLM [12], Orca [26]）**：假设固定深度执行；vSkipper 以虚拟化层方式在不修改这些引擎核心架构的前提下增强其能力。
- **Mixture-of-Depths [18]**：动态分配 compute 给不同层，但属于训练时架构变更；vSkipper 作用于已训练 checkpoint，不改模型结构。

## 局限性与未来方向
- 当前仅支持整层跳过（whole-layer skipping），不支持子层级跳过如 AdaSkip。
- 仅在 SGLang 上实现，未扩展到其他 serving engine（如 vLLM、Orca）或不同 attention backend。
- 分布式 serving（tensor parallelism、pipeline parallelism）尚未支持；pipeline parallelism 被认为是最有前景的扩展方向。
- Prefill 和 decode 的切换阈值基于 roofline 模型的粗粒度估算，实际负载动态波动时可能需要更精细的在线自适应。
- Qwen3 跳层器仅使用单阶段训练（alignment only），未达到两阶段训练可能达到的最优质量-效率权衡。
- 某些 prompt 下 checkpoint 会出现重复生成（repetition）现象，采样惩罚调参留待未来工作。

## 研究启发与可借鉴点
- **Virtualization 层设计思想**：将模型侧算法（跳层策略）与系统侧执行（serving engine）解耦，通过固定接口（RUN/Project-Only）实现即插即用，这一模式可迁移到其他模型压缩/加速技术（如 early-exit、sublayer pruning）的服务化集成。
- **Device-side route tape + cohort packing**：路由决策以 device-resident mask 存储，避免 host-device 同步；cohort 打包使 GEMM bound 由 cohort 大小而非 batch 大小决定，这一 pattern 适用于任何需要 heterogeneous execution 的 serving 场景。
- **Invariant 驱动的 correctness 保证**：明确定义并验证关键不变量（每层写 KV、行元数据不陈旧），为动态深度执行提供形式化正确性基础，可推广至其他变深度推理系统。
- **Roofline-based 阈值推导**：用硬件带宽和算术强度推导出 profitable routing 的解析阈值，无需 per-workload 网格搜索，这一方法论可复用于其他 serving 优化开关（如 speculative decoding、chunked prefill）的自适应控制。
- **Matched-work 评估协议**：冻结输出长度以分离执行速度与生成行为差异，这一实验设计对公平比较 serving 系统的执行效率具有参考价值。

## 关键术语表
- **Dynamic Layer Skipping**：根据每个 token 的需求动态选择执行哪些层，跳过冗余层的推理计算。
- **RUN / Project-Only**：两种逐行执行模式；RUN 执行完整层（attention + FFN），Project-Only 仅执行投影并写 KV，跳过 attention 和 FFN 计算。
- **Route Tape**：设备端驻留的 per-row mask 数组，记录每个 token 在各路由层的跳过决策，供 captured graph 直接消费。
- **Cohort Execution**：将具有相同路由决策的行打包为 contiguous cohort，以 cohort 大小为 bound 执行 GEMM，减少 kernel launch 开销。
- **Profitability-Aware Mode Switching**：基于 roofline 模型推导的阈值，仅在 resident token 数超过盈亏平衡点时启用 routed 模式，避免低负载下的路由开销。
- **Continuous Batching**：现代 LLM 推理引擎的核心调度策略，允许不同长度的请求在同一 batch 中并发执行。
- **Paged KV Caching**：将 KV 缓存按 page 管理（类似操作系统内存分页），支持动态长度的 batch 和 prefix reuse。
- **Difference-in-Differences (DiD)**：2×2 实验设计中的因果推断方法，用于分离 serving 系统本身引入的质量变化与 checkpoint 固有质量损失。

## 可复现要素
- **数据集**：GSM8K、BBH（BIG-Bench Hard）、CoQA；训练数据 allenai/tulu-3-sft-mixture——论文未明确说明开源状态，但基准数据集均为公开数据集。
- **代码**：已开源，SGLang fork 位于 https://github.com/AKafakA/sglang-vskipper/tree/vskipper-ref。
- **权重/Checkpoint**：FlexiDepth-Llama-3-8B-Instruct（xuan-luo 发布）；两个 Qwen3 跳层器 checkpoint 发布于 https://huggingface.co/asdwb/vskip-flexidepth-qwen-3。
- **关键超参**：fp16（Llama）/ bf16（Qwen）；static memory fraction 0.8；Triton attention；decode graph 最大 capture batch size 1,024 rows（A100）；prefill floor 1,536 tokens；Project-Only share 阈值 0.35；Qwen3 训练 lr 1e-4、global batch 32、seq length 2,048、最多 1 epoch。
