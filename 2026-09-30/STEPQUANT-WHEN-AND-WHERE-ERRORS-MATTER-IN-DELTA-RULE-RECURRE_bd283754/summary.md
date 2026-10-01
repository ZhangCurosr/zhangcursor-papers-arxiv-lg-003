---
title: "STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE"
source: https://arxiv.org/pdf/2609.38169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:49:11"
field: "低比特量化与高效推理"
keywords: ["recurrent state quantization", "post-training quantization", "delta-rule attention", "GDN", "KDA", "STEPQuant", "mixed-precision", "key-row impact"]
innovations: ["提出寿命感知位分配，将门控保留时长建模为误差累积权重以指导混合精度", "提出键行感知双轴拟合，把 readout 敏感分数注入行/列缩放联合优化", "在 SGLang 中以融合 tile kernel 实现压缩状态的 decode 流水线，实测 5× 压缩与近无损 6-bit 精度"]
benchmarks: ["LiveCodeBench v6", "EvalPlus", "AIME 2026", "MATH-500", "HMMT Feb 2026", "GPQA Diamond", "IFBench", "MMLU", "ARC-C", "HellaSwag", "WinoGrande", "LAMBADA"]
---

# 论文速读：STEPQUANT-WHEN-AND-WHERE-ERRORS-MATTER-IN-DELTA-RULE-RECURRE

## 一句话总结
本文提出 STEPQuant，一种针对 Gated Delta-rule 线性注意力模型循环状态的后训练量化框架，通过时间维度的"生命周期感知位分配"与空间维度的"键行感知双轴拟合"联合优化，在名义 6-bit 预算下使 Qwen3.8-27B 和 Kimi-Linear-48B-A3B-Instruct 的循环状态精度几乎无损，并将 6-bit 配置下的循环状态压缩 5× 以上、服务内存最高降低 68.7%。

## 研究问题与动机
- **核心问题**：Gated Delta-rule 线性注意力（如 GDN、KDA）用固定大小的循环状态替代 KV Cache，但在高并发推理场景下，每个请求的持久化状态池仍随并发数线性增长，成为与权重内存相当的瓶颈。
- **均匀量化失效**：直接对循环状态施加 INT4/6/8 均匀量化会导致推理精度断崖式下跌，且在 4/6-bit 下性能严重退化，说明量化误差的传播机制不同于常规权重或激活。
- **误差传播的两个维度**：时间上，Delta 更新递归地把量化误差反馈进下一次状态，长寿命（高门控保留）的单元累积误差更大；空间上，不同 key 行对输出的影响差异显著，且状态矩阵在 key 行与 value 列两个轴上都存在数量级差异巨大的离群值，单轴缩放无法同时适配。
- **现有方法不足**：面向 Mamba 类 SSM 的量化方法（如 Q-Mamba、Quamba2）未考虑 Delta-rule 特有的递归误差积累与键行-输出相关性；同期工作 DAMP 在 9.9-bit 下仅能维持 INT8，而本文证明在 6-bit 即可实现近 FP32 精度。

## 核心贡献（创新点）
- **误差传播的理论刻画**：形式化证明了循环状态量化误差的递归传播（Proposition 1），揭示门控保留因子 $\|A_t\|_2 \leq \|D_t\|_2 \leq 1$ 决定了误差累积的上界，并将"记忆寿命"与"累积量化误差"在 2304 个 Qwen 头上验证出 Spearman $\rho_S \approx 0.80$ 的强相关性。
- **寿命感知位分配（Temporal）**：首次将混合精度思想引入循环状态，定义基于平均对数保留率的寿命权重 $L_u = \sum_{j=0}^{H-1} \exp(2j\ell_u)$，在固定比特预算下最小化寿命加权重建失真，并保留 1.39% 高风险单元为 FP16 pivot。
- **键行感知双轴拟合（Spatial）**：发现状态矩阵同时存在 key 行和 value 列双轴的离群结构（最大/中位 RMS 分别为 10.3× 与 19.4×），提出 $r_i = m_i^{1/2} w_i^{-1/2}$ 的行缩放与以 $w_i^2$ 加权的列缩放联合拟合，将行影响分数 $\omega_i = \mathbb{E}[g_{t,i}^2]$ 嵌入量化网格。
- **系统与算子集成**：在 SGLang 中实现融合 tile 重建 + Delta 更新 + 当前 readout 的 GPU kernel，并在独立 CUDA stream 上重叠 scale 拟合与 packed writeback；实测 Qwen@6bit 循环状态压缩 5.03×、更新加速 2.91×，总服务内存降低 68.7%。
- **低比特可行性证明**：在 BF16 与 4-bit AWQ 权重下均验证 STEPQuant 的有效性，4-bit 配置下均匀 INT4 在 Qwen 上平均精度跌至 12.73%，而 STEPQuant@4bit 仍达 80.51%，证明循环状态可压至 4~6-bit 而不崩塌。

## 方法详解
### 前置：Gated Delta-rule 线性注意力
单头状态更新公式（Eq.1/Eq.2）：
$$S_t = D_t S_{t-1} + \beta_t k_t (v_t^\top - k_t^\top D_t S_{t-1}) = A_t S_{t-1} + B_t$$
其中 $A_t = (I - \beta_t k_t k_t^\top)D_t$，$B_t = \beta_t k_t v_t^\top$。GDN 用标量门 $D_t = \alpha_t I$，KDA 用通道级对角门。输出 $y_t = S_t^\top q_t$。

### 时间维度：寿命感知位分配
- **误差递归（Prop.1）**：令 $E_t = \hat{S}_t - S_t$、$\varepsilon_t = Q_t(X_t) - X_t$，有 $E_t = A_t E_{t-1} + \varepsilon_t$，即旧误差被 $A_t$ 衰减传播，新误差逐层叠加。
- **寿命估计**：在校准集上采样每个 unit $u$ 的门控保留率 $r_{t,u}$，计算对数保留 $\ell_u = \mathbb{E}[\log r_{t,u}]$，近似第 $j$ 步误差保留因子为 $\exp(j \ell_u)$，寿命权重 $L_u = \sum_{j=0}^{H-1} \exp(2j\ell_u)$。
- **位分配优化**：在候选码本 $B_{\bar{b}}$ 中为每个 unit 选 $b_u$，最小化 $\sum_u L_u \cdot d_u(b_u)$ 受限于 $\sum_u n_u b_u \leq \bar{b} \sum_u n_u$。Qwen 采用多选择 DP，KDA 采用 Lagrangian + 离散修正。
- **FP16 pivot**：保留风险最高的少量 unit（Qwen 32 heads、KDA 512 rows）以 FP16 存盘，不参与整数 bit 预算。

### 空间维度：键行感知双轴拟合
- **行影响分数**：$\omega_i = \mathbb{E}_{cal}[(A_t^\top q_t)_i^2]$ 衡量 key 行 $i$ 对 readout 误差的贡献，将行按 $\omega_i$ 分组做 INT4 消融验证其可预测性（Fig.2(a)）。
- **行/列双轴缩放**：重构值 $\hat{X}_{ij} = r_i c_j z_{ij}$。
  - 行缩放 $r_i = m_i^{1/2} w_i^{-1/2}$：$m_i$ 为行均值幅值（决定量化范围），$w_i$ 为归一化行影响（决定精度粒度），高影响行得到更细的量化步长。
  - 列缩放：最小化加权重构误差 $\min_{c_j>0} \sum_{i,j} w_i^2 (X_{ij} - r_i c_j z_{ij})^2$，使列轴适应剩余方差，同时对高影响行施加更大惩罚。
- **解码流程**（表 C2）：① 从 packed 整数码与 scale 重建上一状态；② 计算 $X_t$；③ 立即产出 $\hat{y}_t$；④ 同步进行行/列 scale 拟合；⑤ packed writeback，老表示保持有效直到下游读者消费完毕。

### 硬件加速
- 融合 tilewise 重建 + Delta update + readout 为单次 kernel launch，避免整态 FP32 shadow 拷贝。
- 在 SGLang 0.5.12 上以 radix cache 方式将 packed state page 映射到 request slot；prefill 解包、decode 直接在整数页上做 FP-tile 更新。

## 实验与结果
### 设定
- 模型：Qwen3.8-27B（GDN）、Kimi-Linear-48B-A3B-Instruct（KDA）
- 硬件：4× NVIDIA A800，TP=4
- 权重配置：BF16、4-bit AWQ
- 校准：32 段 WikiText-2 × 2048 tokens
- 基线：FP32、INT4/6/8 均匀量化、Q-Mamba@DSQ
- 评测：7 个长生成推理 + 6 个短生成理解任务

### 主要结果
**长生成（BF16 权重，表 1）**：
- Qwen 平均精度：FP32 80.60%，INT6 45.04%，STEPQuant@6bit 80.59%（差 0.01）、@4bit 80.51%（差 0.09）；Kimi 对应 61.52 / 45.70 / 61.47 / 58.52。
- INT4 在 AIME/HMMT 上几乎归零，STEPQuant@4bit 保留 86.25%/73.86%。

**短生成（表 2）**：
- Qwen STEPQuant@4bit 平均 87.63% vs FP32 87.78%（-0.15）；Kimi @4bit 68.11% vs 68.36%（-0.25）。
- INT4 在 Qwen WinoGrande 跌至 46.57%、HellaSwag 60.51%，STEPQuant@4bit 均保持在 89.50%、93.08%。

**与 W4 权重兼容（表 3）**：
- Qwen STEPQuant@6bit + W4：79.27%（FP32 79.32%，-0.05）；Kimi 58.62% vs 58.95%（-0.33）。

**消融（表 4 + App.E.4）**：
- 4-bit Qwen：Spatial only 73.95%，Temporal only 12.87%，Temporal w/o pivots 6.17%，STEPQuant 84.72%；FP32 84.61%，证明双轴协同与 pivot 保护互补。
- Q-Mamba@4bit 仅 7.64%，远低于本文 spatial only 的 73.95%，体现键行影响分数对 Mamba 式双轴方法的必要性。

**效率（Fig.3/Fig.F4）**：
- Qwen@6bit：循环状态内存从 150.99 MiB/request 降至 29.99 MiB（5.03×，80.1%），总服务内存从 419.73 GiB 降至 131.18 GiB（-68.7%，B=512）。
- KDA@6bit：总内存 149.70 → 69.36 GiB（-53.7%），循环状态压缩 5.08×（80.3%），更新延迟 8.48 → 4.76 ms（-43.9%）。
- 吞吐：Qwen B=512 从 6040 → 7280 tokens/s（+20.53%）；KDA 21241 → 23748 tokens/s（+11.80%）。

**生成长度（Fig.3/E.2）**：
- INT4 在 KDA 上 AIME 平均 63.40K tokens、HMMT 64.22K tokens，逼近 65536 上限却近乎 0 准确率，出现"过思考"。STEPQuant@4bit 平均 13.01K / 20.55K tokens，长度与 FP32 相近。

## 相关工作脉络
- **线性注意力 / Delta-rule**：RetNet、GLA、GDN、KDA、Mamba/Mamba-2 均以固定状态替代 KV Cache；本文聚焦 GDN/KDA 这一分支的循环状态量化，与上述工作的差异在于处理门控遗忘与 Delta 残差共同作用下的误差传播。
- **PTQ 权重/激活量化**：GPTQ、SmoothQuant、QuaRot、OmniQuant 等面向 Transformer 权重的 PTQ；本文方法不能直接迁移，因为状态矩阵兼具"跨步递归依赖"与"键行×值列双轴异方差"特性。
- **KV Cache 量化**：KIVI（2-bit 非对称 KV 分轴）、H2O（heavy-hitter 保留）、IntactKV（pivot token 保留）等；本文将"pivot 保留 + 双轴 scale"思路转嫁到循环状态上，但引入时间维度（寿命权重）。
- **SSM/状态量化**：Quamba/Quamba2（Mamba 权重与 8-bit 缓存）、MambaQuant（方差对齐旋转）、Q-Mamba（DSQ + selectivity 重构）；本文对比发现 Q-Mamba@4bit 在 GDN 上仅 7.64%，而本文同预算达 84.72%，核心差异在于显式建模键行对 readout 的平方影响 $g_{t,i}^2$。
- **同期工作 DAMP**：使用 Hadamard 变换 + 基于衰减的 channel 选择保留 FP16，剩余量化至 INT8，有效位 9.9；本文 6.3-bit 即达同等保留率，证明 GDN/KDA 状态可在更低精度工作。
- **推理加速系统**：SGLang（radix cache、tile-wise 融合内核）、TurboQuant（在线向量量化）；本文实现与 SGLang 0.5.12 深度集成。

## 局限性与未来方向
- **寿命权重的近似性**：$L_u$ 以标量对数保留率近似，忽略了时间变化门控与 key 相关的状态转移，不能完全刻画长期误差。
- **4-bit 在 KDA 上的精度差距**：W4 权重下 KDA@4bit 相对 FP32 仍有 2.29 点下降，且生成长度上升约 4.4%，Qwen 则几乎无损，说明 KDA 的通道级门比 GDN 的标量门更难在极低比特下稳定。
- **评估范围有限**：仅在两个 GDN/KDA 模型、固定 A800 配置与静态 workload 下验证；动态批处理、跨架构、跨语言/代码任务的泛化有待进一步检验。
- **FP16 pivot 的开销**：pivot 占比虽低（Qwen 1.39%），但对极端小 batch 或极低比特场景的边际收益未充分讨论。
- **未来方向**：可探索 (i) 在线自适应位分配，而非离线校准固定；(ii) 联合优化权重与状态的混合精度；(iii) 将行影响分数扩展至 key-query 交互的全局敏感性；(iv) 在更长上下文（百万 token）与多模态/ agent 场景中验证。

## 研究启发与可借鉴点
- **双维度拆解量化误差**："时间（误差传播持久性）× 空间（对下游任务影响）"的分析范式可直接迁移到其他递归结构（如 RNN/LSTM/Transformer 的 recurrent 模块、世界模型）的量化设计。
- **键行影响分数 $\omega_i = \mathbb{E}[(A^\top q)^2]$**：用一行向量内积平方刻画状态行对输出的敏感度，计算零成本（只需校准期记录 $g_t$），适合复用于其他矩阵型状态（如 MoE 的 gates、SSM 的 B/C 投影）的量化先验。
- **FP16 pivot + 密集整数混合**：仅保护极少高风险单元（~1%）即可显著托底 4-bit 精度，是一种高 ROI 的低成本兜底策略；在权重量化、MoE 专家选择、attention head mask 等稀疏保护场景均可能复用。
- **双轴 scale 联合拟合**：与 SmoothQuant 的单轴 outlier 平滑不同，本文同时沿行/列拟合且以 $w_i^2$ 加权，对"非各向同性 + 跨轴异方差"矩阵（如状态矩阵、选择机制的 $B_t C_t$ 乘积）有通用参考价值。
- **SGLang 融合 kernel 设计模式**："读 → compute → 同时跑 scale 拟合 + packed writeback 到另一 stream"的双轨流水线，可作为同类状态更新算子的工程模板。

## 关键术语表
- **Gated Delta-rule 线性注意力（GDN/KDA）**：以固定维度矩阵状态替代 KV Cache 的线性注意力变体，通过门控遗忘与当前 key/value 的 Delta 残差联合更新状态。
- **STEPQuant**：本文提出的空间-时间联合后训练量化框架，包含寿命感知位分配与键行感知双轴拟合两大组件。
- **Lifetime-aware Bit Allocation**：基于平均对数门保留率构造寿命权重，在固定比特预算下为循环状态单元分配不同 bit 宽度的混合精度策略。
- **Key-Row-Aware Dual-axis Fitting**：利用键行对 readout 误差的平方影响 $\omega_i$ 构造行缩放，并与列缩放联合最小化以适配状态矩阵的双轴离群结构。
- **FP16 pivot**：在量化过程中保留的一小部分高风险状态单元（头部或行），以 FP16 精度存盘，防止精度崩塌。
- **Readout error propagation**：在 Delta-rule 更新中，上一步的量化误差通过 $A_t$ 矩阵被递推传播至后续 step 的输出误差。
- **RMS outlier contrast**：状态矩阵某轴（行或列）的最大 RMS 相对于该轴中位 RMS 的比值，本文观测到 key 行 10.3×、value 列 19.4×。
- **Recall / retention gate $D_t$**：控制状态记忆留存比例的矩阵或标量，GDN 为 $\alpha_t I$，KDA 为对角阵，其谱范数决定误差传播衰减程度。

## 可复现要素
- **代码**：已开源，https://github.com/Dreamer-Toby/STEPQuant
- **模型与权重**：Qwen3.8-27B、Kimi-Linear-48B-A3B-Instruct，需从官方渠道下载；BF16 与 4-bit AWQ 权重均使用。
- **数据集**：校准集为 32 段 WikiText-2 × 2048 tokens；评测集包括 C4、LiveCodeBench v6、EvalPlus、AIME 2026、MATH-500、HMMT Feb 2026、GPQA Diamond、IFBench、MMLU、ARC-C、OpenBookQA、HellaSwag、WinoGrande、LAMBADA，均为公开 benchmark。
- **关键超参**：
  - 预算：名义 4-bit / 6-bit（含 scale 开销后 Qwen 实际 4.62/6.36 bits/value，KDA 4.30/6.30）
  - 候选码本：@4bit {2,4,6,8}，@6bit {4,6,8}
  - Pivot 数量：Qwen 32 heads、KDA 512 rows（约 1.39%）
  - 行影响幂次：$\gamma=0.25$（App.C.1）
  - 校准段长度：2048 tokens，32 段
- **硬件与并行**：4× A800，TP=4；SGLang 0.5.12；batch 最大 512，prompt 128 tokens，每 sample 生成 1024 tokens。
- **开源声明**：论文明确说明作者贡献、代码与实验复现细节均在附录提供；AI 仅用于语言润色、LaTeX 检查与 debug。
