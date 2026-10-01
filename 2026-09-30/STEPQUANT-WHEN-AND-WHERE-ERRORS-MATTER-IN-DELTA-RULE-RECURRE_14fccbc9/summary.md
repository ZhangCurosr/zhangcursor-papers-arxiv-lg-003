---
title: "STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE"
source: https://arxiv.org/pdf/2609.38169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:49:09"
---

# 论文速读：STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE

## 一句话总结
提出 STEPQuant，一种针对 Delta-rule 循环状态的空间-时间后训练量化（PTQ）框架，通过生命周期感知位分配与键行感知双轴拟合，在名义 6-bit 预算下使 Qwen3.8-27B 和 Kimi-Linear-48B-A3B-Instruct 的精度匹配 FP32 基线，并实现超 5× 循环状态压缩与最高 68.7% 的服务内存节省。

## 研究问题与动机
1. 线性注意力用固定大小的循环状态替代增长型 KV 缓存，但在并发推理时每个请求需独立持久状态，总状态池内存随并发数线性增长，在 Qwen 中超过 BF16 权重内存。
2. 直接对循环状态应用均匀量化会引发严重精度下降，因为量化误差通过 Delta 规则递归更新不断累积传播，在 INT4/INT6 下平均精度骤降。
3. 量化误差的影响沿两个互补维度变化：时间上，长寿命记忆单元累积更大误差；空间上，不同键行对输出误差的贡献不同，且状态矩阵在两轴上均存在显著离群值，单轴缩放无法兼顾。

## 核心贡献（创新点）
1. **时空误差传播的理论刻画**：推导条件误差传播公式 $E_t = A_t E_{t-1} + \varepsilon_t$，证明门控衰减与 Delta 修正共同作用使误差不放大但可持久，且长半衰期头/通道与累积误差呈强正相关（Spearman ρ≈0.80）。
2. **生命周期感知位分配**：提出基于门控保留寿命与重构失真的混合精度分配策略，在固定预算下将更多比特分配给误差大且持久的单元，与 SPQR/SqueezeLLM 等仅考虑幅值的权重混合精度本质不同。
3. **键行感知双轴拟合**：设计键行与值列独立缩放因子的双轴量化方案，行因子同时编码当前行幅值与经校准的输出影响权重，区别于传统单轴 outlier 平滑或 Q-Mamba 的双轴但无空间影响感知的方案。
4. **SGLang 集成与高效内核**：将重建-更新-读取融合为单一 CUDA kernel，并在独立流上并行执行尺度拟合与打包回写，实现 2.91× 更快的状态更新与 5.03× 状态压缩。

## 方法详解
1. **错误传播建模**：证明 $\|A_t\|_2 \leq \|D_t\|_2 \leq 1$，表明量化误差不放大；长期误差主要由正交于当前 key 的分量经门控衰减决定，保留率接近 1 时误差可持续多个 decode step。
2. **寿命权重计算**：对分配单元 $u$（Qwen 为整头，KDA 为键行），计算平均对数保留率 $\ell_u = \mathbb{E}[\log r_{t,u}]$，导出 $H$ 步寿命权重 $L_u = \sum_{j=0}^{H-1} \exp(2j\ell_u)$。
3. **混合精度分配优化**：在 $\sum n_u b_u \leq \bar{b}\sum n_u$ 约束下最小化 $\sum_u L_u d_u(b_u)$，Qwen 用多选择动态规划、KDA 用拉格朗日分配+离散修复；高风险单元保留为 FP16 稀疏枢轴（Qwen 32 头 / KDA 512 行，约 1.39%）。
4. **键行影响得分**：定义 $\omega_i = \mathbb{E}_{cal}[(A_t^\top q_t)_i^2]$ 衡量第 $i$ 键行对读取误差的贡献；将行按 $\omega_i$ 分八组后单独 INT4 量化验证，高影响组困惑度下降显著更大。
5. **双轴缩放因子拟合**：行因子 $r_i = m_i^{1/2} w_i^{-1/2}$，其中 $m_i$ 为行幅值、$w_i \propto \omega_i^{\gamma/2}$（$\gamma=0.25$）为影响权重；列因子 $c_j$ 通过最小化 $\sum_{i,j} w_i^2(X_{ij}-r_i c_j z_{ij})^2$ 求解，对高影响行给予更大误差惩罚。
6. **解码流程**：每步重建压缩状态→计算 Delta 更新 $X_t$→发出 $\hat{y}_t=X_t^\top q_t$→融合内核完成尺度拟合与打包回写；枢轴行保持 FP16 不参与共享列尺度拟合。

## 实验与结果
- **模型/硬件**：Qwen3.8-27B (GDN) 与 Kimi-Linear-48B-A3B-Instruct (KDA)，BF16 与 4-bit AWQ 权重，4× NVIDIA A800，SGLang v0.5.12。
- **校准**：32 段 WikiText-2，每段 2048 tokens。
- **长生成基准**（7 任务）：Live-CodeBench v6, EvalPlus, AIME 2026, MATH-500, HMMT Feb 2026, GPQA Diamond, IFBench。
- **短生成基准**（6 任务）：MMLU, ARC-C, OpenBookQA, HellaSwag, WinoGrande, LAMBADA。

关键数字：
- **Qwen @6bit**：STEPQuant 平均 80.59% vs FP32 80.60%（几乎无损），INT6 仅 45.04%；AIME 87.24% vs 33.96%，LCB 85.42% vs 30.43%。
- **Qwen @4bit**：STEPQuant 80.51% vs FP32 80.60%，INT4 仅 12.73%。
- **Kimi @6bit**：STEPQuant 61.47% vs FP32 61.52%，INT6 仅 45.70%。
- **Kimi @4bit**：STEPQuant 58.52% vs FP32 61.52%，INT4 仅 21.63%。
- **与 W4A16 兼容**：Qwen @6bit 79.27% vs FP32 79.32%；Kimi @6bit 58.62% vs FP32 58.95%。
- **消融**：Spatial only @4bit 73.95%，Temporal only @4bit 12.87%，加枢轴后 Temporal 提升显著（+6.70 点 @4bit）；组合后 @4bit 达 84.72%。
- **内存/吞吐**：@6bit 压缩 5.03× (Qwen) / 5.08× (Kimi)；总服务内存减 68.7% (Qwen) / 53.7% (Kimi)；decode 吞吐 Qwen @512 提升 20.53% (6040→7280 tokens/s)。
- **生成长度**：均匀 INT4 在 Kimi 上平均生成 63.4K (AIME) / 64.2K (HMMT) tokens 且精度近零（过思考）；STEPQuant 长度与 FP32 相当。

## 相关工作脉络
1. **线性注意力架构**（RetNet, Gated LA, GDN, KDA, Mamba/Mamba-2）：本文聚焦这些架构的循环状态压缩，区别于纯权重/激活量化路线。
2. **SSM 量化**（Quamba/Quamba2, MambaQuant, Q-Mamba）：前两者主要量化权重与激活；Q-Mamba 提出双轴状态量化但面向 Mamba，本文针对 GDN/KDA 的 Delta-rule 更新与时空误差特性设计。
3. **KV 缓存量化**（KIVI, IntactKV, HVQ）：解决传统 softmax 注意力中增长型 KV 的压缩，本文处理的是固定大小但并发敏感的循环状态。
4. **PTQ 方法**（SmoothQuant, GPTQ, OmniQuant, SPQR, SqueezeLLM）：主要面向权重/激活，本文首次将混合精度与双轴拟合系统引入循环状态 PTQ。
5. **并发工作 DAMP**：基于衰减持久性选择 FP16 通道并 INT8 量化其余（9.9 bits），本文通过时空联合设计在 6.3/4.3 bits 达到相近/更好精度保留（100.51%/92.91% vs 100.99%）。
6. **混合精度量化**（SPQR, SqueezeLLM, OMPQ）：主要在空间维度分配比特，本文额外引入时间维度（生命周期权重），使分配同时感知误差幅度与持久性。

## 局限性与未来方向
1. 生命周期权重仅通过门控衰减近似误差持久性，未完整建模时变且 key 依赖的状态转移，对长期误差积累捕捉不够充分。
2. KDA 在 BF16 权重 @4bit 时长生成任务精度下降（58.52% vs 61.52%），且生成长度增加约 4.4%；W4A16 下 Qwen @4bit 精度损失仅 0.34 点但生成长度增加约 31.6%。
3. 实验仅覆盖 GDN/KDA 两种架构及固定 A800 硬件，其他架构（如纯 Mamba）与动态服务负载下的泛化性待验证。
4. 未评估不同上下文长度分布、极端并发模式下的鲁棒性与内存收益。

## 研究启发与可借鉴点
1. **时空双维度分析范式**：将量化误差分解为时间持久性与空间影响分布，可迁移至 State Space Models、Linear Attention 及其他循环中间表示的压缩研究。
2. **校准数据的跨域稳定性**：WikiText 校准得到的半衰期排序在 C4 和 LiveCodeBench 上 Spearman 相关达 0.98+，说明轻量校准即可捕获结构性敏感度。
3. **双轴离群结构的普适发现**：状态矩阵在两轴上的最大/中位 RMS 对比达 10.3× 与 19.4×（98.6% 样本超 3×），可启发对其他矩阵型中间表示（如 V 矩阵、选择性 SSMs 的 $B/C$ 矩阵）的量化设计。
4. **稀疏枢轴的高性价比**：仅保留 1.39% 的 FP16 单元即可在 @4bit 提升 6.7 个点，证明"少而精"的超高精度保护比均匀低比特更有效。
5. **内核融合与流重叠的系统优化**：重建-更新-读取三合一 + 尺度拟合在独立流上并行，对低比特 serving 系统的通用设计具有参考值。

## 关键术语表
1. **Gated DeltaNet (GDN)**：逐头标量门控 + Delta 规则更新的线性注意力架构，Qwen3.8 系列采用。
2. **Kimi Delta Attention (KDA)**：GDN 的扩展，采用逐通道门控，用于 Kimi Linear / K3 系列模型。
3. **Lifetime-aware Bit Allocation**：结合重构失真与门控保留寿命的混合精度位分配策略，长寿命高误差单元获更多比特。
4. **Key-Row-Aware Dual-Axis Fitting**：分别为键行与值列拟合独立缩放因子的空间量化方法，行因子同时编码幅值与输出影响权重。
5. **FP16 Pivots**：量化过程中以 FP16 保留的高风险状态单元（约 1.39%），用于保护对精度最敏感的通道。
6. **Recurrent State**：线性注意力中 summarizing past tokens 的固定大小矩阵 $S_t \in \mathbb{R}^{d_k \times d_v}$，替代传统 KV cache。
7. **Post-Training Quantization (PTQ)**：训练完成后直接量化的方法，本文聚焦循环状态的 PTQ 而非权重/激活。
8. **Overthinking**：量化模型生成异常长输出但精度未提升的现象，均匀 INT4 在长生成任务上尤为严重。

## 可复现要素
- **数据集**：校准用 32 段 WikiText-2（2048 tokens/段）；基准测试均公开（Live-CodeBench v6, EvalPlus, AIME 2026, MATH-500, HMMT Feb 2026, GPQA Diamond, IFBench, MMLU, ARC-C, OpenBookQA, HellaSwag, WinoGrande, LAMBADA）。
- **代码**：开源，https://github.com/Dreamer-Toby/STEPQuant。
- **模型权重**：Qwen3.8-27B、Kimi-Linear-48B-A3B-Instruct（官方发布）；W4A16 权重使用 AWQ 量化版本。
- **关键超参**：
  - 候选比特集：@4bit {2,4,6,8}，@6bit {4,6,8}
  - FP16 枢轴：Qwen 32 heads，KDA 512 key rows
  - 行影响指数 $\gamma = 0.25$（KDA）
  - 寿命权重历史步数 $H$：论文附录未明确给出，需查阅源码
  - 硬件：4× NVIDIA A800，TP4
  - SGLang 版本：0.5.12
- **基线**：FP32、uniform rowwise-absmax INT4/INT6/
