---
title: "ROUNDING-IN-PRECONDITIONER-SPACE-REDESIGNING4-BIT-ADAMW-OPTI"
source: https://arxiv.org/pdf/2610.12444v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:26:58"
field: "低精度优化器与高效训练"
keywords: ["optimizer quantization", "4-bit AdamW", "stochastic rounding", "preconditioner space", "low-precision training"]
innovations: ["在预处理器空间定义舍入坐标以减小更新误差", "提出零包含+预处理器空间SR和零排除+EDEN校准两种4-bit AdamW配置", "定位LM-head一阶矩在训练尾部的定向偏置并提出阶段性SR切换"]
benchmarks: ["GPT-style 162M-2.7B pretraining", "Llama-style 130M-1.1B pretraining", "Qwen3-8B SFT", "Llama-3.2-3B SFT"]
---

# 论文速读：ROUNDING-IN-PRECONDITIONER-SPACE-REDESIGNING4-BIT-ADAMW-OPTI

## 一句话总结
本文从"舍入坐标空间"视角重新设计 AdamW 的 4-bit 优化器状态量化，提出两种配置（ZIP-SR 与 ZE-EDEN），在保持与 TorchAO 4-bit 相同的持久存储成本下，显著缩小了与 32-bit AdamW 的验证损失差距（最大降低 70%）。

## 研究问题与动机
- **量化误差通过矩递推传播**：AdamW 的二阶矩存储在步骤间循环更新，量化误差会不断传播并扰动后续的自适应更新。
- **零点位失效率（zero-point failure）**：Li et al. (2023) 发现将小的正二阶矩映射为 0 会产生过大的更新步长；TorchAO 因此将 0 排除在二阶矩码本外，但引入了正下界（positive floor）问题。
- **状态空间舍入≠预处理器空间保真**：AdamW 对二阶矩施加非线性变换（取倒数平方根）后才缩放更新，因此在状态空间中无偏的舍入，仍可能扭曲最终更新方向。
- **一阶矩码本选择被忽视**：TorchAO 使用 SDyn4 作为一阶矩格式，但 NF4（基于正态分布分位数）在该分布上的重建误差更低。

## 核心贡献（创新点）
1. **首次系统分析舍入坐标对优化动力学的影响**：通过局部零单元分析和标量二次构造，证明"状态空间舍入"与"预处理器空间舍入"会产生本质不同的优化动态，这是与 TorchAO 等工作的本质区别——后者仅关注状态空间重建误差。
2. **提出 ZIP-SR（零包含预处理器空间随机舍入）**：保留 0 在二阶矩码本中，但在预处理器空间计算随机舍入概率，显著降低有害零选择的概率；与以往零包含方案的区别在于通过改变舍入坐标而非码本结构来解决零点位失败。
3. **提出 ZE-EDEN（零排除 EDEN 校准）**：沿袭零排除路线，但引入 EDEN 块校准来缓解正下界导致的预处理器失真；与以往工作的区别在于首次将 EDEN 校准适配到 AdamW 二阶矩量化场景并量化其收益。
4. **发现 LM-head 一阶矩在训练尾部的定向偏置问题**：通过匹配检查点干预定位到 NF4-RTN 在 LM-head 的一阶矩上产生系统性径向收缩，提出最后 10% 训练中仅对该状态切换至 NF4-SR。

## 方法详解
**基本设定**：对 AdamW 的 working moments $\tilde{\mathbf{m}}_t, \tilde{\mathbf{v}}_t$ 在写入存储前进行量化，下一轮递推从解码后的值出发。定义当前步预处理器映射 $h_t(v) = 1/(\sqrt{v/(1-\beta_2^t)} + \epsilon)$ 和下一步预处理器映射 $r_{t+1}(v) = h_{t+1}(\beta_2 v + (1-\beta_2)g_{t+1}^2)$。

**ZIP-SR（第二部分二阶矩）**：
- 码本：零包含 Dyn4（包含 0）
- 舍入坐标：$\phi(v) = h_t(v)$（预处理器空间），而非 $\phi(v)=v$（状态空间）
- 随机舍入概率：$P(Q=v_a) = \frac{h_t(v) - h_t(b)}{h_t(a) - h_t(b)}$，保证 $\mathbb{E}[h_t(Q(\tilde{v}_t))] = h_t(\tilde{v}_t)$
- 理论分析（Proposition 1）：Update-SR 的下一步预处理器误差有界（$\Delta_r = -\Theta(1)$），而 State-SR 的预处理器误差随 $\epsilon \to 0$ 发散为 $\Theta(\epsilon^{-1})$；在零邻域单元格内，Update-SR 舍入到 0 的概率为 $\Theta(\epsilon)$，远低于 State-SR 的 $\Theta(1)$。
- 标量二次实验（Proposition 2）：State-RTN 收敛到非零二周期，而 Update-RTN 几何收敛到最小值；在最优解附近，State-SR 局部扩张而 Update-SR 局部收缩。

**ZE-EDEN（第二部分二阶矩）**：
- 码本：零排除 Dyn4-NZ（不含 0，最小正值 $0.00325s$）
- 对量化块 $\mathbf{x}$ 应用 EDEN 校准：$\tilde{\mathbf{x}} = c_\tau Q(\mathbf{x})$，其中 $c_\tau = \|\mathbf{x}\|_2^2 / \max\{\langle \mathbf{x}, Q(\mathbf{x}) \rangle, \tau s^2\}$，$\tau=10^{-12}$
- 校准因子 $c_\tau < 1$ 时降低解码块尺度，从而压低正下界，部分恢复被抑制的大预处理器尾部

**一阶矩设计**：
- 格式：NF4（均值平方重建误差 $1.056\times10^{-3}$，优于 SDyn4 的 $2.495\times10^{-3}$ 和 FP4 的 $1.739\times10^{-3}$）
- 默认舍入：NF4-RTN
- 特殊策略：LM-head 一阶矩在训练最后 10% 切换为 NF4-SR（通过匹配检查点干预定位失败源）

**存储成本**：block size=128，每参数 1.0625 字节（vs FP32 的 8 字节，压缩比 7.53×）。

## 实验与结果
**预训练实验**：GPT-style（162M–2.7B）和 Llama-style（130M–1.1B）模型，FineWeb-Edu 数据，20 tokens/parameter。

- **834M 组件归因**（Table 3）：相较 TorchAO（gap=0.0246），NF4 一阶矩贡献最大单步提升（33.7% 缩减），ZIP-SR 最终 gap=0.0140（较 TorchAO 降低 43.2%），ZE-EDEN gap=0.0177（降低 28.2%）。
- **跨规模预训练**（Table 4）：两种方法在所有模型尺寸上均缩小 TorchAO 的 gap；GPT-style 1.4B 达到最大相对缩减 **70.1%**（TorchAO gap 0.0521→ZIP-SR 0.0156）。GPT-style 2.7B 上 TorchAO 在 cooldown 末期出现剧烈发散（gap=1.5435），而 ZIP-SR（0.0343）和 ZE-EDEN（0.0535）保持稳定。
- **SFT 实验**（Table 5）：在 Qwen3-8B 和 Llama-3.2-3B 上全参数微调，两种方法均取得低于 TorchAO 的验证损失；ZE-EDEN 在 Qwen3-8B 上 gap≈0，下游任务（MMLU、GSM8K、HumanEval、IFEval）分数与 32-bit AdamW 接近。

## 相关工作脉络
1. **Li et al. (2023)**：首次将优化器状态扩展到 4-bit，发现零点位失败并提出排除 0 的零排除策略；本文沿袭此思路但分别探索零包含+预处理器舍入和零排除+EDEN 校准两条路线。
2. **TorchAO (Or et al., 2025)**：工业级实现，使用 Lin4-NZ（线性零排除码本）+ SDyn4 一阶矩；本文指出 NF4 在一阶矩上的重建误差更优，并提出预处理器空间舍入的替代方案。
3. **Xu et al. (2025) / SOLO**：研究未量化状态的信号淹没和方差放大；本文聚焦舍入坐标选择对预处理器误差的影响，与其互补。
4. **Topollai & Choromanska (2026)**：研究状态陈旧性和周期性重置；本文从舍入空间角度解释零点位失败，两者从不同层面理解量化动力学。
5. **Vargaftik et al. (2022) / EDEN**：原始工作针对联邦学习的分布式均值估计；本文首次将其适配到优化器状态量化场景，用于校准二阶矩的正下界失真。
6. **Zhan et al. (2026) / BReD**：针对有偏循环舍入引起的持续漂移；本文的 Update-SR 从根源上减小了预处理器空间的偏差，与 BReD 的纠偏策略形成对比。

## 局限性与未来方向
- 理论分析仅针对条件单步预处理器偏差和玩具标量二次模型，尚未建立量化 AdamW 的通用端到端收敛性保证。
- LM-head 一阶矩定向偏置的成因及其与学习率衰减的关系仍未完全阐明。
- 方法尚未扩展到 4-bit 以下（如 2-bit）的优化器状态量化。
- 是否适用于其他自适应优化器（如 AdaGrad、RMSProp）尚待研究。

## 研究启发与可借鉴点
1. **"舍入坐标"是一个被忽视的设计自由度**：不仅适用于 AdamW，任何涉及非线性变换（如根号、倒数）的自适应优化器都可考虑在变换后空间进行舍入，以获得更小的更新误差。
2. **匹配检查点干预（matched checkpoint interventions）是一种强力的故障定位技术**：本文通过独立切换不同层/区域的量化规则并对比 trajectories，精确定位到 LM-head NF4-RTN 是 2.7B 训练失稳的根源，值得在大规模系统调试中推广。
3. **NF4 在一阶矩上的优势**：FP32 一阶矩在 block-normalized 分布下近似正态，NF4 的分位数设计天然匹配；这一观察可推广到其他使用 EMA 结构的场景。
4. **EDEN 校准的思路可迁移**：对于任何引入正下界的量化方案（如零排除码本），可通过内积校准因子动态调整尺度，缓解分布尾部失真，不适用于仅依赖 absmax 缩放的场景。
5. **阶段性切换舍入策略**：在训练最后阶段将特定层的 RTN 切换为 SR，以抑制累积定向偏置；这种"warm-start + late-stage correction"模式可与其他训练稳定化技术结合。

## 关键术语表
**Preconditioner（预处理器）**：AdamW 中由二阶矩计算的缩放因子 $1/\sqrt{v_t}$，决定每个参数的自适应学习率。
**Zero-point failure（零点位失败）**：二阶矩量化时将小正数映射为 0，导致预处理器异常增大、更新步长失控。
**Stochastic Rounding (SR)（随机舍入）**：以与距离成比例的概率随机选择相邻量化级别，期望上无偏。
**Round-to-Nearest (RTN)（最近舍入）**：确定性选择距离输入最近的量化级别。
**Dyn4 / Dyn4-NZ**：动态指数（DE）4-bit 码本；Dyn4 包含 0，Dyn4-NZ 排除 0。
**NF4（NormalFloat 4-bit）**：基于标准正态分布分位数构建的 4-bit 浮点格式，包含精确的 0。
**EDEN calibration（EDEN 校准）**：通过归一化内积自适应调整量化块尺度，缓解正下界带来的分布失真。
**Radial bias（径向偏置）**：量化误差在输入向量方向上的投影，正值为扩张、负值为收缩。

## 可复现要素
- **数据集**：FineWeb-Edu（预训练）、Tulu-3（SFT），均公开可用。
- **代码/权重**：PyTorch 实现已开源，链接 https://github.com/nubank/adamw4bit；训练脚本包含在内。
- **关键超参**：block size=128，$\beta_1=0.9$，$\beta_2=0.95$，$\epsilon=10^{-8}$，梯度裁剪=1.0，学习率 schedule= WSD（warmup-stable-decay），LM-head SR 切换点=训练最后 10%。
- **硬件**：H200 GPU（部分实验用 B200）。
- **并行**：PyTorch DDP / DeepSpeed ZeRO-2。
