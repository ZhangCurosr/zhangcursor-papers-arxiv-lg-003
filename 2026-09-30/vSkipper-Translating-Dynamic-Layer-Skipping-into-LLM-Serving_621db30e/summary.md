---
title: "vSkipper-Translating-Dynamic-Layer-Skipping-into-LLM-Serving"
source: https://arxiv.org/pdf/2609.37062v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:36:12"
---

# 论文速读：vSkipper-Translating-Dynamic-Layer-Skipping-into-LLM-Serving

## 一句话总结
vSkipper 提出了一种面向现代 LLM Serving 引擎的虚拟化层，将动态逐 token 层跳跃（dynamic layer skipping）无缝集成到 SGLang 中，通过 RUN/Project-Only 混合深度执行与基于 roofline 的收益感知模式切换，成功将理论 FLOP 节省转化为实际的端到端延迟下降与饱和吞吐提升，且在 GSM8K 与 BBH 上实现了无统计显著质量损失的服务加速。

## 研究问题与动机
- **核心问题**：现有动态层跳跃方法（如 FlexiDepth）虽能减少模型计算量，但其专用生成循环无法与现代 Serving 引擎的连续批处理、固定形状批次、分页 KV 缓存及捕获解码图（captured graphs）兼容，导致“节省的 FLOP 无法转化为节省的服务延迟”。
- **动机 1**：论文复现发现，FlexiDepth 平均跳过 8/32 层，但在标准生成循环中 decode 吞吐反而下降 14.6%–21.0%，路由开销在低负载时抵消了节省的计算，在高负载时拆散批次宽度同样拖累性能。
- **动机 2**：Serving 引擎默认假设批次内所有 token 遍历相同层数；动态跳跃打破该规整性后，若直接接入会破坏 batching 效率、KV 状态一致性与图捕获优化。
- **动机 3**：亟需一个引擎侧的虚拟化接口，使 skipper 的逐 token 决策能以 cohort 打包形式在捕获图中执行，并仅在计算收益超过路由固定开销时启用。

## 核心贡献（创新点）
1. **Serving 虚拟化层与 RUN/Project-Only 接口**：将 skipper 策略与执行、引擎状态彻底解耦，作为可插拔组件接入现代 serving 引擎，无需重设计调度器或内存管理器。*与已有工作本质区别： prior skipper 多停留在模型侧 FLOP 分析或专用 PyTorch 路径，本文首次提供 engine-side 标准化接口。*
2. **混合深度批次执行（Mixed-depth execution）**：引入 route tape 与 cohort 打包机制，在同一 captured graph 内将 RUN 与 Project-Only 行分组，通过 gather/scatter 折叠进出 operand 与 epilogue。*区别于 MoE/LoRA 多 adapter 服务（仍固定遍历每层），本文在 depth 维度进行动态路由，且严格维持逐层 KV 缓存不变量。*
3. **收益感知模式切换（Profitability-aware mode switching）**：基于 roofline 模型推导盈亏阈值 $V^* = \frac{\tau \cdot \mathrm{BW}}{s L_r b}$，设置 hysteresis band（exit/enter），decode 阶段支持单向晋升，避免低负载反噬。*与固定 early-exit 或静态 depth 策略不同，阈值由硬件带宽、模型常量与 skipper 权重动态算出，具备跨设备可移植性。*
4. **首个实现内部逐 token 层跳过 Serving 加速的系统**：在 SGLang 上验证 FlexiDepth、自训练 Qwen3 skippers 及九种合成策略，跨 A100/H100/RTX A6000 无需 workload-specific tuning 即获显著延迟/吞吐收益。*定位：填补“模型侧 skipper”与“系统侧 serving”之间的执行接口空白。*

## 方法详解
- **SKIPPER 接口**：策略输出每行（token）的 `RUN` 或 `PROJECT_ONLY` 决策及 projector 权重。Project-Only 路径跳过注意力与 FFN 主计算，仅通过 skipper 自带 projector 轻量投影，并**必须**写入当前层自身的 KV 投影（满足 Invariant 1）。
- **Route Tape 与 Cohort 执行**：每层决策记录为 device-resident 的 route tape。decode 阶段在 captured graph 内从 tape 派生 RUN/Project-Only cohort 的索引与大小（设备标量），将 gather 折叠进 operand load，weighted scatter 折叠进 epilogue。Decode 使用 count-bounded Triton GEMMs，Prefill 使用标准 cuBLAS GEMMs，均与原始引擎 forward pass 兼容。
- **正确性不变量**：
  - **Invariant 1（KV 完整性）**：即使 token 被跳过，其所在层的 key/value 仍由层自身投影权重生成并写入 KV cache，保障 batched attention、paged KV 与 prefix reuse 的有效性。
  - **Invariant 2（Metadata 新鲜度）**：所有 row→memory 映射（attention metadata、page mapping、gather/scatter map）均从当前 tape 重新计算，杜绝 stale map 导致跨请求 KV 泄漏。
- **模式切换规则**：
  - **Decode**：基于 roofline 盈亏点推导，A100 上 $V^* \approx 158\mathrm{k}$ resident tokens。设置 hysteresis band：进入阈值 200k，退出阈值 160k。切换为单向（dense→routed 可晋升，生成中途不可逆转）。
  - **Prefill**：固定阈值 ≥1,536 prompt tokens 且 Project-Only share > 0.35 时启用 routed 模式，每 64 次 admission 轮询一次。
- **引擎兼容**：保持 continuous batching、admission control、paged KV management、radix prefix cache 不变；mode 作为 prefix reuse 的 key 之一，dense/routed 命名空间分离。

## 实验与结果
- **实验设置**：Baseline 为 upstream SGLang，测试模型 Llama-3-8B（FlexiDepth checkpoint，路由最后 16 层）与自训练 Qwen3-4B/8B skippers。工作负载 GSM8K、BBH、CoQA。硬件 A100-80GB、H100-80GB、RTX A6000-48GB。采用 matched-work 协议（记录 baseline 输出长度后 replay），隔离执行成本与生成行为差异。
- **核心性能（A100, matched-work）**：
  - GSM8K 在 knee（$0.95 \times Q^*$）：Mean E2E latency **↓36.8%**，TPOT ↓35.6%，TTFT ↓66.0%。
  - BBH 在 knee：Mean E2E latency **↓13.6%**，TPOT ↓14.6%。
  - 饱和负载（$1.25 \times Q^*$）：RPS **↑11.3%** (GSM8K)，**↑7.4%** (BBH)。
  - CoQA：整体无显著 resolution，但 makespan ↓7.0–7.8
