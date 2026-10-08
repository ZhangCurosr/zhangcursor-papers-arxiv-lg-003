---
title: "ResidualQuant-KV-Cache-Quantization-for-Looped-Transformers"
source: https://arxiv.org/pdf/2610.10381v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:07:38"
field: "大模型推理效率优化"
keywords: ["KV Cache Quantization", "Looped Transformers", "Low-bit Quantization", "Residual Quantization", "Inference Optimization", "INT2 Quantization"]
innovations: ["提出 Last-loop 残差量化框架，以末环 KV 为锚点存储跨环 INT2 残差，结合 LS 缩放显著降低极低精度下的量化误差", "将正交旋转（OptR-H）作用于残差而非原始 KV，实现残差量化与旋转 outlier 缓解的正交互补", "设计环路混合精度 [2,2,2,4] 策略，将 INT4 分配给共享锚点、INT2 分配给残差，在 80.7% 存储压缩下保持 BF16 级精度"]
benchmarks: ["GSM8K", "MATH500", "HumanEval", "MBPP"]
---

# 论文速读：ResidualQuant-KV-Cache-Quantization-for-Looped-Transformers

## 一句话总结
本文提出 ResidualQuant，一种针对循环 Transformer（Looped Transformers）的 KV Cache 量化方法，利用跨循环 KV 状态的高相似性，将各循环的 KV 表示为末循环参考状态的 INT2 残差，结合最小二乘缩放、正交旋转与环路混合精度，在 KV 存储降低 80.7% 的同时保持接近 BF16 的推理精度。

## 研究问题与动机
- 循环 Transformer（如 Ouro、Huginn）通过共享块重复计算提升深度而不增加参数量，但 KV Cache 随循环数线性增长，成为内存瓶颈，限制了 batch size 和推理吞吐。
- 现有 KV 共享方案（MoR、PLT、MELT）用全局/首循环 KV 替代后续循环，会丢失循环特异性信息并导致精度显著下降。
- 现有旋转类 KV 量化方法（OptR-H、OSCAR 等）在 INT2/4 极低精度下仍出现严重精度退化，难以同时兼顾低比特压缩与准确性。
- 关键观察：循环 Transformer 中同一层/token/head 处的 KV 状态因共享投影矩阵而高度相似，跨环残差比原始 KV 本身 outlier 更少，更适合极低精度量化。

## 核心贡献（创新点）
- **残差量化框架（Last-loop + LS Scaling）**：以末循环 KV 为共享参考锚点，存储其余各环 KV 的残差而非独立量化全量 KV；与直接量化的本质区别在于利用循环间冗余大幅缩小量化误差分布。
- **残差正交旋转（Residual + OptR-H）**：将 OptR-H 旋转矩阵作用于残差而非原始 KV，进一步打散残差中的 outlier；与已有旋转工作的区别在于旋转目标从"全量 KV"转为"跨环差值"，配合残差结构产生互补增益。
- **环路混合精度 [2,2,2,4]**：将更高精度（INT4）分配给作为共享参考的末环锚点，其余环保持 INT2；与均匀精度的区别在于锚点质量直接影响所有残差重建精度，因此非对称分配带来更优精度–存储权衡。
- **系统级 kernel 与 vLLM 集成**：设计 tile-by-tile 的自定义 CUDA attention kernel，避免重建后写入/回读显存；将延迟量化整合进 vLLM 调度；实测在 RTX 5090 上 decode 吞吐最高提升 4.15×。
- **扩展到 Activation 量化**：残差思想同样适用于 W4A4 激活量化，对 Naive、FlatQuant、LoopQ 均有稳定提升，证明跨循环冗余利用的通用性。

## 方法详解
- **基本设定**：对第 $r$ 环 KV 向量 $X_r$，以末环重建锚点 $\widehat{X}_L$ 为参考，引入标量系数 $\alpha_r$，重建公式：
  $$\widehat{X}_r = \alpha_r R_r + Q(X_r - \alpha_r R_r), \quad R_r = \widehat{X}_L$$
- **参考策略选择**：比较四种策略（Previous-loop / First-loop / Last-loop / Next-loop）。Last-loop 策略使早期环的 NMSE 最低（平均 69.80×10⁻³ vs. First-loop 82.57×10⁻³），且无需链式读取，逻辑缓存读取次数从 $O(L^2)$ 降至 $2L-1$。
- **LS 缩放系数**：$\alpha_r = \langle X_r, R_r \rangle / \|R_r\|_2^2$，最小化 $\|X_r - \alpha_r R_r\|_2^2$；相比 $\alpha_r=1$ 可将 key 预测 MSE 最多降低 24.3%。
- **残差旋转**：对残差施加正交旋转 $U$（OptR-H，Hadamard 初始化），量化后做 $U^\top$ 逆变换重建；旋转矩阵按层/头学习、跨循环共享，存储不随序列长度增长。
- **混合精度分配**：配置为 $[2,2,2,4]$（Ouro-1.4B 共 4 环），有效位宽 $b_{\text{eff}} \approx 3.1$ bit（含 FP8 元数据 + BF16 $\alpha_r$ 系数），相较 BF16 存储减少 80.7%。
- **Effective bitwidth 计算**：$b_{\text{eff}} = \frac{1}{L}\sum_r(b_r + 16/g_r) + c_\alpha$，其中 $c_\alpha = \frac{16(L-1)}{LD}$ 为 LS 系数的摊销成本。
- **Kernel 设计**：tile-by-tile 重建锚点与残差，直接用于 QK dot-product 与 V 加权累加；利用 shared-memory lookup table 预计算 INT2 四种码值避免重复 scale-offset 计算；异步加载掩盖显存访问延迟。
- **vLLM 集成**：Prefill 阶段各环使用 BF16 prompt KV；末环结束后量化锚点并计算其余环残差存储；Decode 阶段当前 token 的 KV 在末环完成前保持 BF16，完成后写入压缩缓存。

## 实验与结果
- **模型**：Ouro-1.4B（4 环）、Huginn-3.5B（32 环，按 8 组×4 环分组应用）。
- **基准**：GSM8K、MATH500（数学推理）；HumanEval、MBPP（代码生成）；评测硬件 RTX 5090 / A100-SXM4-40GB。
- **关键结果（Ouro-1.4B，Mixed INT2/4, g=32）**：
  - BF16 Avg = 52.94；OptR-H = 47.17；**ResidualQuant = 52.37**（接近 BF16，差距仅 0.57）。
  - MATH500：BF16=75.0，OptR-H=63.0，**ResidualQuant=76.0**（反超 BF16）。
  - 相较 OptR-H 在同等预算下最高提升 **13.0%**（Ouro-1.4B MATH500: 71.6 vs 55.4）。
- **关键结果（Huginn-3.5B，Mixed INT2/4, g=32）**：
  - ResidualQuant + OptR-H Avg=52.37 vs BF16 Avg=52.94。
- **吞吐量（Ouro-1.4B, RTX 5090, g=32）**：
  - 固定 batch 场景：16k context 下 decode 吞吐提升 **2.73×**（34.0→93.1 tok/s）。
  - 最大可行 batch 场景：8k context 下最大 batch 从 4 增至 16，峰值吞吐提升 **3.27×**；16k context 下从 2 增至 4，峰值吞吐提升 **4.15×**。
- **Activation 量化扩展（W4A4）**：对 Ouro-1.4B GSM8K，Naive 从 49.51% → 56.86%（+7.35%），FlatQuant 从 68.31% → 70.20%（+1.89%），LoopQ 从 65.88% → 68.31%（+2.43%）；Huginn-3.5B 同理稳定提升。

## 相关工作脉络
- **KV Cache 量化**：KIVI、KVQuant 针对 KV 分布特性设计量化；QuaRot、SpinQuant、RotateKV 引入旋转；OSCAR 与 OptR-H 优化至 INT2 并保留部分 BF16 锚定 token——本文在此基础上利用循环冗余实现全量 INT2 残差量化，无需额外 BF16 buffer。
- **循环/递归 Transformer**：Universal Transformer、ACT、PonderNet 探索参数共享与自适应计算深度；MoR、PLT、MELT 尝试跨环 KV 复用/共享——本文不修改模型结构或训练流程，仅针对 KV 存储做后训练量化。
- **Looped Transformer 优化**：LoopSpec 用 self-speculative decoding 加速；FlashLoop（并发工作）也基于跨环冗余做 INT4 量化 + 稀疏更新——本文方法在 INT2 精度下仍保持高精度，且与 FlashLoop 的稀疏机制正交可叠加。
- **权重-激活量化**：FlatQuant 学习平坦化变换；LoopQ 专门针对循环 Transformer 的分布漂移设计——本文证明残差思想可正交附加于上述方法之上进一步提升精度。

## 局限性与未来方向
- **校准依赖**：OptR-H 旋转矩阵需要校准数据集（GSM8K/MBPP），不同任务/领域需重新校准，未见零校准方案。
- **循环数分组限制**：Huginn-3.5B 的 32 环被分为 8 组×4 环独立应用，组间无残差关联；更大分组可降低锚点开销但可能影响精度（表 8 显示 16 环分组精度略降）。
- **仅评估 post-training 量化**：未探索与训练感知量化的联合优化，可能存在进一步精度提升空间。
- **当前仅支持固定循环数模型**：对自适应循环深度（如 Huginn 的 dynamic depth）的适配未讨论。
- **未来方向**：探索零校准旋转、自适应分组策略、与 FlashLoop 稀疏机制的系统级联合部署、扩展至其他循环架构（Deep Equilibrium Models 等）。

## 研究启发与可借鉴点
- **跨周期冗余利用范式**：残差量化思路可迁移至其他具有重复计算结构的模型（如 RNN、DEQ、状态空间模型），只要存在跨步/跨迭代相似性即可应用。
- **锚点-残差混合精度策略**：将更高精度分配给"共享参考"而非"差异信号"的设计原则，对多版本/多分支模型量化具有普适参考价值。
- **Kernel 级 tile-by-tile 重建**：避免完整反量化后再计算的 design pattern，可直接复用于其他低比特 KV 量化系统的 attention kernel 实现。
- **残差 + 旋转的正交组合**：证明残差表示与旋转 outlier 缓解两个技术维度的互补性，提示未来工作可系统化搜索残差构造方式与旋转目标的组合空间。
- **与稀疏更新的正交性**：本文与 FlashLoop 的组合实验表明，结构级稀疏与数值级量化可叠加优化，为系统层联合压缩提供思路。

## 关键术语表
**Looped Transformer**：通过重复应用共享 Transformer 块来增加计算深度而不增加参数量的模型架构，代表工作有 Ouro、Huginn。
**Last-loop 参考策略**：以末循环 KV 作为所有其他循环重建的共享锚点，避免 Previous-loop 的链式读取开销。
**Least-Square (LS) Scaling**：通过 $\alpha_r = \langle X_r, R_r \rangle / \|R_r\|_2^2$ 最优缩放参考 KV，使残差范数最小化。
**OptR-H**：基于注意力输出误差优化的头级正交旋转方法，Hadamard 初始化，用于打散 KV outlier。
**Effective Bitwidth ($b_{\text{eff}}$)**：计入量化码值、FP8 元数据（scale/offset）及 LS 系数后的平均每 KV 元素存储位数。
**Mixed INT2/4 Precision**：将 INT4 分配给末环锚点、INT2 分配给其余残差环的非对称量化配置 $[2,2,2,4]$。
**W4A4 Quantization**：权重与激活均为 INT4 的后训练量化设置。
**KV Reconstruction NMSE**：归一化均方误差，衡量重建 KV 与原始 BF16 KV 的偏差。

## 可复现要素
- **数据集**：GSM8K、MATH500、HumanEval、MBPP（均为公开基准）。
- **模型权重**：Ouro-1.4B（ByteDance/Ouro-1.4B）、Huginn-3.5B（tomg-group-umd/huginn-0125），均来自 HuggingFace，公开可用。
- **代码**：项目页面 https://seal-kaist.github.io/projects/ResidualQuant；FlashLoop 对比实验使用其公开代码库。
- **关键超参**：INT2 group size $g=16/32$；INT4 group size $g=32$；OptR-H 学习率 0.02、80 步、Adam、FP32；校准样本数 128（Ouro）/1024（Huginn）；混合精度配置 $[2,2,2,4]$。
- **硬件**：NVIDIA RTX 5090、RTX PRO 6000 Blackwell（96GB）、A100-SXM4-40GB。
- **开源声明**：论文未明确声明代码仓库 URL，仅给出项目页面链接；模型权重公开可下载。
