---
title: "ResidualQuant-KV-Cache-Quantization-for-Looped-Transformers"
source: https://arxiv.org/pdf/2610.10381v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:03:06"
field: "低比特量化与高效推理"
keywords: ["KV Cache Quantization", "Looped Transformer", "Low-bit Quantization", "Residual Quantization", "INT2 Quantization", "Inference Throughput"]
innovations: ["利用末loop残差重建实现INT2级别KV压缩", "Last-loop参考与LS缩放结合降低重建误差", "混合精度锚点分配提升精度-存储权衡"]
benchmarks: ["GSM8K", "MATH500", "HumanEval", "MBPP"]
---

# 论文速读：ResidualQuant-KV-Cache-Quantization-for-Looped-Transformers

## 一句话总结
本文提出 ResidualQuant，一种面向 Looped Transformer 的 KV Cache 量化方法，利用各 loop 间 KV 状态的高度相似性，以最后一个 loop 的 KV 作为锚点、用 INT2 残差表示其余 loop，结合最小二乘缩放、正交旋转与循环混合精度，在 KV 存储减少 80.7% 的同时保持接近 BF16 精度，并在 RTX 5090 上将解码吞吐最高提升 4.15×。

## 研究问题与动机
1. Looped Transformer 虽通过参数共享降低存储，但 KV Cache 随循环次数线性增长，成为制约批量大小与推理吞吐的关键瓶颈。
2. 既有 KV Cache 共享/复用方案（如 MoR、PLT、MELT）会丢弃 loop 特有的表征信息，在低比特下精度退化严重。
3. 现有旋转-based 量化方法（OptR、OSCAR 等）在 INT2 等极端低精度下仍存在显著精度损失，且仍需维护 BF16 缓冲区。
4. Looped Transformer 具有独特机会：共享投影矩阵使得不同 loop 的 KV 状态高度相关，残差分布更集中、异常值更少，适合激进低比特量化。

## 核心贡献（创新点）
1. 提出 Last-loop 残差量化框架，以末 loop KV 为锚点存储低精度残差，避免 Previous-loop 链式重建带来的内存访问开销；与仅依赖统计旋转的方法本质不同，利用了循环结构的冗余性。
2. 引入最小二乘（LS）缩放与残差旋转（OptR-H）的组合策略，使 INT2 量化更加鲁棒；与单独使用缩放或旋转相比，二者在残差空间协同可显著降低预测 MSE。
3. 提出 loop-wise 混合精度分配（[2,2,2,4]），将较高精度集中在共享锚点以提升整体重建质量；与统一精度方案相比，在同等有效位宽下取得更好的精度-存储权衡。
4. 实现完整的定制 CUDA 注意力内核与 vLLM 集成，验证压缩可直接转化为固定 batch 与峰值吞吐提升（最高 2.73× 与 4.15×）。
5. 将残差重建原则扩展到 W4A4 激活量化，在与 LoopQ、FlatQuant 等方法结合时一致提升精度，证明方法的通用性。

## 方法详解
1. **残差量化基本形式**：对第 r 个 loop 的 KV 向量 X_r，取末 loop 锚点 R_r = X̂_L，经 LS 缩放后量化残差：X̂_r = α_r R_r + Q(X_r - α_r R_r)，其中 α_r = ⟨X_r, R_r⟩ / ‖R_r‖²。
2. **参考策略选择**：对比 Previous-loop、First-loop、Last-loop 与 Next-loop 四种策略，Last-loop 在所有 loop 上平均重建误差最低，且仅需一次锚点读取即可重建任意非锚点 loop，避免链式依赖。
3. **残差旋转**：对残差 X_r - α_r R_r 施加正交旋转 U，量化旋转变换后的残差并在重建时应用 U^T，以重新分布残留异常值；K 与 V 分别使用独立旋转矩阵，跨 loop 共享。
4. **混合精度配置**：锚点 loop 使用 INT4（group size g=32），其余 loop 使用 INT2（g=16 或 32），典型配置为 [2,2,2,4]，平均位宽提升至约 2.5 bit，有效位宽 b_eff 约 3.1 bit（含元数据与 LS 系数）。
5. **内核与 vLLM 集成**：采用 tile-by-tile 重建策略，避免将完整 BF16 KV 写入显存；利用共享内存查找表预计算去量化值，异步加载与计算重叠；decode 时当前 token 的 KV 暂存于 BF16，待末 loop 完成后才量化并写入压缩缓存。

## 实验与结果
1. **模型与基准**：Ouro-1.4B（4 loops）与 Huginn-3.5B（32 loops，划分为 8 组、每组 4 loops）；在 GSM8K、MATH500、HumanEval、MBPP 上评估 pass@1 或精确匹配。
2. **主要精度结果**：在混合 INT2/4（g=32）下，Ouro-1.4B 的 MATH500 达 76.0%，与 BF16 的 75.0% 持平；相比 OptR-H 基线（55.4%）提升 20.6 个百分点，相对提升约 37%。
3. **存储压缩**：混合精度配置下理论 KV 存储减少 80.7%，有效位宽 b_eff 约 3.1 bit，远低于 BF16 的 16 bit。
4. **吞吐提升**：在 RTX 5090 上，8k 上下文固定 batch=4 时吞吐从 111.4 提升至 192.5 tokens/s（1.73×）；16k 上下文最大可行 batch 从 2 增至 4，峰值吞吐从 34.0 提升至 141.3 tokens/s（4.15×）。
5. **激活量化扩展**：在 W4A4 设置下，残差量化使 Naive、FlatQuant 与 LoopQ 在 GSM8K/MATH500 等任务上均有 1~8% 的绝对提升，证明方法可迁移至激活域。

## 相关工作脉络
1. **KV Cache 量化**：KIVI、KVQuant 提出面向 K/V 分布的分组仿射量化；QuaRot、SpinQuant、RotateKV 引入正交旋转再分布异常值；OSCAR 与 OptR 进一步用 attention-aware 目标优化旋转，但仍需保留 BF16 缓冲区；本文利用循环冗余将 INT2 残差与 INT4 锚点结合，无需额外 BF16 缓冲区。
2. **Looped Transformer KV 复用**：MoR 复用首 loop KV；PLT 结合全局 KV 与局部滑动窗口；MELT 通过学习门控更新共享 KV；本文保留各 loop 独立 KV 但通过残差压缩降低存储，不涉及模型重训练或权重更新。
3. **Looped Transformer 优化**：LoopSpec 通过自投机解码加速；SMELT 在匹配 FLOPs 与参数条件下验证稀疏 MoE 循环的效率；本文聚焦 KV 压缩，可与这些优化方向正交组合。
4. **激活量化**：FlatQuant 学习仿射变换展平分布；LoopQ 针对 loop 依赖分布偏移设计 LAS/SLT 组件；本文残差量化可直接叠加于上述方法之上，进一步提升低比特激活压缩精度。
5. **并发工作 FlashLoop**：同样基于循环冗余动机，但采用 Previous-loop 参考与 INT4 残差，精度与吞吐不及本文；两者正交可组合（表 10 验证）。

## 局限性与未来方向
1. 旋转矩阵需离线校准，不同模型/任务可能需要独立优化，增加部署复杂度。
2. 目前主要在数学推理与代码生成任务上验证，对其他领域（如多轮对话、长文本摘要）的泛化性尚待探索。
3. 对 32 loop 的 Huginn，采用每组 4 loop 的分组策略，组间残差关联可能被削弱，更大组或自适应分组可能进一步优化。
4. LS 系数计算与旋转矩阵乘法引入额外运算开销，虽被内存访问减少所补偿，但在极端低功耗场景下仍需权衡。
5. 仅讨论后训练量化（PTQ）路径，未涉及量化感知训练（QAT）或在线校准策略，可能仍有精度提升空间。

## 研究启发与可借鉴点
1. **结构感知量化**：利用模型架构固有特性（如循环冗余、注意力沉积水）设计量化策略，而非仅依赖数据分布统计，可突破通用量化的精度瓶颈。
2. **参考策略对系统效率的影响**：Last-loop 共享锚点策略避免了 Previous-loop 链式重建导致的内存访问爆炸，对实际部署具有重要参考价值。
3. **混合精度分配的优先级原则**：在共享锚点与独立残差之间，应将更高精度分配给影响全局重建质量的共享部分，这一原则可迁移至其他层次化量化场景。
4. **残差重建的通用性**：将残差量化从 KV Cache 扩展至激活量化，证明了"参考+低精度差异"范式的广泛适用性，可与现有激活量化管线无缝集成。

## 关键术语表
**Looped Transformer**：通过重复应用同一组 Transformer 模块处理输入，以增加计算深度而不增加参数量的模型架构，如 Ouro 与 Huginn。
**KV Cache**：注意力计算中缓存的 key 与 value 张量，用于避免在 decode 阶段重复计算历史 token 的表征，其大小随序列长度、batch 与循环次数线性增长。
**残差量化（Residual Quantization）**：不直接量化原始张量，而是先选择一个参考值并用低精度存储两者之差，重建时再加回参考值，适合分布高度相关的场景。
**Last-loop 策略**：以最后一个 loop 的 KV 状态作为所有其他 loop 的共享参考锚点，实现 O(1) 次缓存读取重建任意 loop 的 KV。
**Least-Square（LS）缩放**：通过计算标量 α = ⟨X_r, R_r⟩ / ‖R_r‖² 最小化残差范数，使量化前的预测误差更小，从而降低量化噪声。
**OptR-H**：以 Hadamard 矩阵初始化的头级正交旋转方法，通过最小化 attention 输出误差来学习旋转矩阵，用于改善 INT2 量化下的残差分布。
**Effective Bitwidth（b_eff）**：包含量化码、组级 scale/offset 元数据及 LS 系数在内的平均每 KV 元素存储位数，用于公平比较不同方法的实际内存占用。
**W4A4 量化**：权重与激活均采用 4 位整数量化的技术，本文将其与残差量化结合以进一步压缩计算过程中的中间表示。

## 可复现要素
- **数据集**：GSM8K、MATH500、HumanEval、MBPP（均为公开基准）。
- **代码/权重**：论文提供项目页面 https://seal-kaist.github.io/projects/ResidualQuant，但未明确声明代码是否开源；模型使用 ByteDance/Ouro-1.4B 与 tomg-group-umd/huginn-0125 checkpoint。
- **关键超参**：INT2 group size g ∈ {16, 32}，INT4 group size g = 32，Mixed 配置 [2,2,2,4]，LS 系数按内积公式计算，OptR-H 旋转矩阵 FP32 校准 80 步、学习率 0.02。
- **硬件环境**：NVIDIA RTX 5090 与 RTX PRO 6000 Blackwell（96GB），单卡评估。
