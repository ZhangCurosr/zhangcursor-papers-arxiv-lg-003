---
title: "V-CoLA-Vision-Token-Compression-with-Linear-Attention"
source: https://arxiv.org/pdf/2610.11251v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:13:35"
field: "多模态大模型推理优化"
keywords: ["视觉Token压缩", "线性注意力", "VLM加速", "免训练压缩", "混合架构"]
innovations: ["提出基于线性注意力状态更新的免训练Token重要性估计", "设计长短期联合的去冗余重要性准则", "实现与chunk-wise并行兼容的自适应分块Token合并"]
benchmarks: ["MME", "MMB", "GQA", "SQA", "POPE", "MMStar"]
---

# 论文速读：V-CoLA-Vision-Token-Compression-with-Linear-Attention

## 一句话总结
本文提出 V-CoLA，一种专为混合架构（含线性注意力，如 Qwen3.5）设计的免训练视觉 Token 压缩框架，通过线性注意力的内在信息保留信号识别关键 Token 并进行自适应分块合并，在仅保留 50% 视觉 Token 时保持 99.5% 性能，12.5% 时仍保持超 88%，预填充加速 1.86×–6.15×。

## 研究问题与动机
- 视觉语言模型（VLM）中视觉 Token 数量远超语言 Token，造成巨大的计算与显存负担。
- 既有视觉 Token 压缩方法（FastV、DART 等）主要针对 softmax 注意力设计，在引入线性注意力（Gated DeltaNet）的混合架构中泛化失败——实验表明它们甚至不及随机裁剪基线。
- 线性注意力的有界循环状态存在信息天花板（Data Processing Inequality），但状态更新的选择性保留行为本身编码了 Token 重要性信号，为适配线性注意力的压缩提供了新思路。
- 当前缺乏同时兼容线性注意力 chunk-wise 并行、且能自适应不同语义区域压缩率的免训练框架。

## 核心贡献（创新点）
1. **系统性揭示既有压缩方法在混合架构中的结构性失效**：证明基于 softmax 注意力分数和基于特征相似度的方法在 Qwen3.5 等线性注意力 VLM 中均大幅退化，并给出信息论层面的解释。
2. **提出 V-CoLA 免训练压缩框架**：首次利用线性注意力的内在更新动力学估计 Token 重要性，而非依赖外部注意力图或相似度，与已有工作本质区别在于"信号来源从注意力输出转向状态更新本身"。
3. **Uniqueness-aware 重要性准则**：联合考量 Token 对最终状态的全局长期贡献（$\mathcal{R}^{\text{imp}}$）与其短期唯一性/去冗余能力（$\mathcal{R}^{\text{uni}}$），避免仅选显著区域导致的冗余选择。
4. **Adaptive Chunk-wise Token Merging 策略**：按重要性密度动态划分 Token 序列为 chunk，并在各 chunk 内以 Softmax 加权进行聚合，相比硬性裁剪更充分保留信息。
5. **实现级优化兼容线性注意力加速**：将重要性计算与 Token 合并适配至 chunk-wise 并行，Extended Gated DeltaRule 在 4-query 下实现 3.85× 加速，额外开销仅占前向 1.6%。

## 方法详解
**1) 线性注意力作为内在重要性指示器**
- 基于 Gated DeltaNet 的关联记忆视角：状态 $\mathbf{S}_t \in \mathbb{R}^{d \times d}$ 将全部历史压缩为固定容量，其状态更新的选择性本身编码 Token 重要性。
- 浅层状态的视觉信息保留率较高且跨层一致，深层层则急剧下降（<20%），支持"在浅层压缩视觉 Token"的策略。

**2) 去冗余的重要性估计（Eq. 6–8）**
- 长期重要性：$\mathcal{R}_t^{\text{imp}} = 1 - \psi(\mathbf{S}_L \boldsymbol{k}_t, \boldsymbol{v}_t)$，其中 $\psi$ 为归一化余弦距离，衡量 Token 信息被最终状态保留的程度。
- 短期唯一性：$\mathcal{R}_t^{\text{uni}} = \psi(\mathbf{S}_t \boldsymbol{Q}_t^{*\top}, \mathbf{S}_{t-1} \boldsymbol{Q}_t^{*\top})$，用邻近窗口内的伪查询 $\boldsymbol{Q}_t^*$ 探测状态变化幅度，反映该 Token 对去冗余的贡献。
- 加权融合：$\mathcal{R}_t = (1-\lambda)\mathcal{R}_t^{\text{imp}} + \lambda\mathcal{R}_t^{\text{uni}}$，默认 $\lambda=0.1$。

**3) 自适应分块 Token 合并（Eq. 9–10）**
- 目标：将序列划分为 $L'=L\times\gamma$ 个 chunk，使每 chunk 重要性之和均衡：
  $$\{\mathcal{C}_i^*\} = \arg\min_{\{\mathcal{C}_i\}}\sum_i\left|\bar{\mathcal{R}}-\sum_{t=t_i}^{t_{i+1}}\mathcal{R}_t\right|$$
- 合并权重：$w_t = \exp(\mathcal{R}_t/\tau)/\sum_k\exp(\mathcal{R}_k/\tau)$，$\tau=1.0$。
- 相比硬裁剪，此策略在密集重要性区分配更细粒度、在稀疏区更粗粒度，自适应调节局部压缩率。

**4) 视觉 Token 早退机制（Section 3.4）**
- PPL 分析表明：在 layer-24 之后移除所有视觉 Token 对 perplexity 几乎无影响。
- MME/MMB/SQA 验证：layer-24 后视觉 Token 可安全丢弃，释放预算用于浅层视觉理解。

**5) 并行实现（Appendix A）**
- Extended Gated DeltaRule：支持多查询注入，仅增加 $O(Nd^2)$ 少量计算。
- 自适应分块：基于累积重要性曲线的平行矩阵操作实现高效划分。

## 实验与结果
**模型与基准**
- 主实验：Qwen3.5-9B（3:1 混合架构）；扩展实验：Qwen3.5-27B、InfiniteVL。
- 基准：MME、MMB、GQA、SQA、T-VQA、POPE、VizWiz、MMStar。
- 对比方法：FastV、SparseVLM、DART、VisionZip、DTP（专门针对线性注意力的方法）。

**主要结果（Qwen3.5-9B）**

| 保留率 | 方法 | MME | MMB | GQA | SQA | POPE | MMStar | 平均保持率 |
|--------|------|-----|-----|-----|-----|------|--------|----------|
| 50% | V-CoLA | 2398.5 | 85.0 | 61.0 | 92.7 | 89.8 | 49.8 | **99.5%** |
| 75% | V-CoLA | 2268.7 | 84.4 | 60.1 | 92.4 | 88.9 | 47.2 | **94.8%** |
| 87.5% | V-CoLA | 2012.0 | 77.9 | 57.5 | 90.9 | 88.3 | 41.4 | **>88.0%** |

- 最强结果：50% 保留率下平均性能保持 99.5%，全面超越所有基线（FastV 97.5%、DTP 99.0%）。
- 12.5% 极端压缩下仍保持超 88%，优于 DTP（75.7%）约 12 个百分点。

**效率结果**
- Prefill 加速：1.86×（50%）→ 6.15×（12.5%），8K Token 场景下 6.15×。
- 额外开销：压缩部分仅占前向 1.6%；Adaptive Token Merging 仅 0.86ms。
- 端到端延迟（INT4 TensorRT）：TTFT 降低 38.6–64.6%，E2E 加速最高 1.733×。

**泛化性**：在 Qwen3.5-27B 和 InfiniteVL 上同样取得最优，超参无需模型微调。

## 相关工作脉络
1. **FastV（Chen et al., 2024a）**：基于 softmax 注意力分数的一步裁剪；本文证明其在混合架构中因浅层注意力快速衰减而失效。
2. **DART（Wen et al., 2025）**：基于特征相似度的去冗余选择；本文指出线性注意力的状态更新破坏了特征空间的相似度结构，使其 pivot token 选择不稳定。
3. **VisionZip（Yang et al., 2025b）**：金字塔式逐层裁剪，依赖 softmax 注意力分布；同样不适配线性注意力架构。
4. **DTP（Park et al., 2025）**：专为 Mamba 架构设计的两阶段裁剪；利用中间输出排名，性能次优（50% 下 99.0% vs V-CoLA 99.5%），且未考虑去冗余与长期保留的联合建模。
5. **SparseVLM（Zhang et al., 2024b）**：文本引导的稀疏化；局限于 softmax 架构的注意力分布假设。
6. **Visual-Word Tokenizer（Gee et al., 2024）**：编码器阶段的静态 patch 分组，与本方法的骨干内部重要性信号正交可互补。

## 局限性与未来方向
- **架构覆盖有限**：当前仅验证了 Qwen3.5、InfiniteVL 等 3:1 混合架构，纯 Mamba 或全线性注意力 VLM 尚未覆盖；这类架构缺少周期性 softmax 层，重要性信号的分布特性可能不同。
- **线性注意力公式范围**：方法主要基于 Gated DeltaNet 推导，对其他 gated 线性注意力变体（如 Mamba-2、RWKV）的系统性验证不足。
- **信息天花板下的极限压缩**：理论分析指出状态容量 $\mathcal{O}(d^2)$ 与信息需求 $\mathcal{O}(Ld)$ 的差距随 $L$ 增长，在极低保留率（<12.5%）时性能仍有下降空间。
- **未来方向**：扩展至纯 Mamba/全线性架构；研究与 Visual-Word 等编码器阶段压缩方法的联合；探索可微分的重要性估计替代 heuristic 打分。

## 研究启发与可借鉴点
1. **"信号内生化"思路**：将压缩判断从外部可观测信号（注意力分数、特征相似度）转向架构内在动力学（状态更新量），为适配其他新架构（如扩散 Transformer）提供范式参考。
2. **长短期重要性联合建模**：$\mathcal{R}^{\text{imp}}$（全局长期）与 $\mathcal{R}^{\text{uni}}$（短期去冗余）的加权融合设计简洁有效，可迁移到文本 Token 压缩或视频 Token 管理。
3. **自适应分块合并替代硬裁剪**：按重要性密度均衡划分 chunk 再 Softmax 加权聚合，比 top-k 保留在极低预算下更稳健，值得在任意序列压缩任务中尝试。
4. **早退机制与压缩协同**：深层层视觉 Token 可安全丢弃的发现，支持"浅层密集压缩 + 深层零预算"的两阶段策略，节约整体推理成本。
5. **实现级并行兼容**：将压缩操作适配至 chunk-wise 并行框架（Extended Gated DeltaRule），为工程落地点提供直接参考。

## 关键术语表
- **V-CoLA**：Vision token Compression with Linear Attention，本文提出的免训练视觉 Token 压缩框架。
- **Gated DeltaNet**：Qwen3.5 采用的线性注意力变体，通过选择性写入/擦除/遗忘门控更新有界状态 $\mathbf{S}_t$。
- **Uniqueness-aware 重要性**：联合长时信息保留（$\mathcal{R}^{\text{imp}}$）与短时去冗余（$\mathcal{R}^{\text{uni}}$）的 Token 打分准则。
- **Adaptive Chunk-wise Token Merging**：按重要性密度均衡划分序列分块并在块内 Softmax 加权聚合的压缩策略。
- **信息天花板（Information Ceiling）**：线性注意力有界状态容量 $\mathcal{O}(d^2)$ 与序列信息需求 $\mathcal{O}(Ld)$ 之间的信息丢失下界。
- **Early Exit（视觉 Token 早退）**：在深层层（layer-24 后）丢弃全部视觉 Token 的加速机制。
- **Prefill Speedup**：压缩后语言模型预填充阶段的推理加速倍数（1.86×–6.15×）。
- **Chunk-wise Parallelism**：线性注意力加速器常用的分块并行策略，V-CoLA 的实现与该策略兼容。

## 可复现要素
- **数据集/基准**：MME、MMB、GQA、SQA、T-VQA、POPE、VizWiz、MMStar（均为公开基准）；长对话评估使用 MMDU，小目标评估使用 V*-Bench。
- **模型**：Qwen3.5-9B（主实验）、Qwen3.5-27B、InfiniteVL（公开模型）。
- **代码/权重**：论文未明确声明开源，附录 A 提供了算法伪代码与并行实现细节；超参数完整列出（$\lambda=0.1, (w_1,w_2)=(0,4), \tau=1.0$）。
- **硬件**：NVIDIA A100 GPU。
- **评估框架**：LMMs-Eval（Zhang et al., 2024a）。
- **关键超参**：压缩层位置为 layer-1 后；早退层为 layer-24；余弦距离作为 $\psi$；窗口 $(w_1,w_2)=(0,4)$。
