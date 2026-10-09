---
title: "ROUNDING-IN-PRECONDITIONER-SPACE-REDESIGNING4-BIT-ADAMW-OPTI"
source: https://arxiv.org/pdf/2610.12444v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:33:40"
field: "低比特优化器状态量化"
keywords: ["optimizer quantization", "4-bit AdamW", "stochastic rounding", "preconditioner space", "low-precision training", "model compression"]
innovations: ["提出预条件器空间随机舍入Update-SR，在含零码本下避免零点失败并使预条件器误差有界", "将EDEN块级校准引入排除零的二阶矩量化以缓解正下界对大预条件器尾部的压制", "通过匹配checkpoint干预定位LM-head一阶矩确定性舍入导致的晚期不稳定性并给出靶向SR方案"]
benchmarks: ["FineWeb-Edu预训练", "Tulu-3 SFT", "MMLU", "GSM8K", "HumanEval", "IFEval"]
---

# 论文速读：ROUNDING IN PRECONDITIONER SPACE: REDESIGNING 4-BIT ADAMW OPTIMIZER-STATE QUANTIZATION

## 一句话总结
论文重新从"舍入空间（rounding space）"视角设计了4-bit AdamW优化器状态量化方法，提出ZIP-SR（含零的预条件器空间随机舍入）和ZE-EDEN（排除零的EDEN校准）两种配置，在不同规模的GPT/Llama预训练中显著缩小了4-bit与32-bit AdamW的验证损失差距（最大减少70.1%），并在SFT中取得更低验证损失。

## 研究问题与动机
- **内存瓶颈**：AdamW的FP32一阶/二阶矩缓冲区需8字节/参数，在大模型训练中带来显著存储开销， motivate 了4-bit等低比特优化器状态量化研究。
- **零点失败（zero-point failure）**：Li et al. (2023) 发现，将小的正二阶矩值映射到零会导致下一步预条件器异常放大，产生过大参数更新；后续工作（包括TorchAO）均采用"排除零"的二阶矩码本。
- **正下界的副作用**：排除零的码本引入了正量化下界，会压制小二阶矩对应的大预条件器条目，同样损害优化动态。
- **核心洞察**：舍入坐标的选择决定了含零码本是否会导致零点失败——AdamW在缩放更新前对二阶矩做了非线性变换，状态空间无偏的舍入仍可能扭曲实际更新；将舍入决策放在预条件器空间可大幅降低有害零选择的概率。

## 核心贡献（创新点）
1. **提出预条件器空间舍入的分析框架**：形式化了舍入坐标φ对一步预条件器偏差Δ_r的影响，证明在零相邻单元中Update-SR可将零选择概率从Θ(1)降至Θ(ε)，使预条件器误差有界。
2. **ZIP-SR方法**：保留含零的二阶矩Dyn4码本，但在预条件器空间$h_t(v)=1/\sqrt{v/(1-\beta_2^t)}+\epsilon$中计算随机舍入概率，与State-SR相比避免了零点失败。
3. **ZE-EDEN方法**：沿用排除零的Dyn4-NZ码本，引入EDEN块级校准$\tilde{\mathbf{x}}=c_\tau Q(\mathbf{x})$以缓解正下界对大预条件器尾部的压制。
4. **第一矩量化策略**：实证表明NF4优于TorchAO使用的SDyn4；并通过匹配checkpoint干预定位到LM-head一阶矩的确定性舍入是2.7B模型冷却期不稳定的根源，提出在最后10%训练切换为NF4-SR。

## 方法详解
- **AdamW矩递推与存储约定**：工作矩$\tilde{\mathbf{m}}_t=\beta_1 Q(\tilde{\mathbf{m}}_{t-1})+(1-\beta_1)\mathbf{g}_t$，$\tilde{\mathbf{v}}_t=\beta_2 Q(\tilde{\mathbf{v}}_{t-1})+(1-\beta_2)\mathbf{g}_t^2$，参数更新使用存储前的工作矩。
- **舍入坐标定义**：状态空间舍入用φ(v)=v，预条件器空间舍入用φ(v)=$h_t(v)=1/(\sqrt{v/(1-\beta_2^t)}+\epsilon)$。
- **Update-SR概率**：在相邻量化级$a<b$间，对$\tilde{v}_t\in(a,b)$，以概率$\frac{h_t(\tilde{v}_t)-h_t(b)}{h_t(a)-h_t(b)}$选a。当a=0时，零选择概率为$\frac{\epsilon(1-\sqrt{\tilde{v}_t/b})}{\sqrt{\tilde{v}_t/(1-\beta_2^t)}+\epsilon}=\Theta(\epsilon)$。
- **Proposition 1**：证明Update-SR的下一步预条件器期望不大于真实值（$\mathbb{E}[r_{t+1}(Q_U)]\leq r_{t+1}(\tilde{v}_t)$），而State-SR不小于；且当$\epsilon\to0$时，State-SR/State-RTN的预条件器误差发散为$\Theta(\epsilon^{-1})$，Update-SR保持有界。
- **ZE-EDEN校准公式**：$c_\tau=\|\mathbf{x}\|_2^2/\max\{\langle\mathbf{x},Q(\mathbf{x})\rangle,\tau s^2\}$，当内积超过能量时$c_\tau<1$，下调块尺度以缓解正下界压制。
- **第一矩设计**：默认NF4-RTN；在GPT-style 2.7B中发现NF4-RTN在冷却期产生持续的径向负偏差（ inward bias），导致梯度爆炸；通过匹配checkpoint干预确认LM-head切换为NF4-SR即可恢复稳定轨迹。

## 实验与结果
- **数据集**：FineWeb-Edu；**评估基准**：预训练验证损失（GPT-style 162M~2.7B，Llama-style 130M~1.1B），SFT在Tulu-3上用Qwen3-8B和Llama-3.2-3B评估MMLU/GSM8K/HumanEval/IFEval。
- **组件归因（834M）**：NF4替换SDyn4是最大单项提升（gap从0.0312降至0.0207，-33.7%）；在此基础上EDEN再降14.7%，Update-SR再降32.5%；相对TorchAO最终gap降低43.2%。
- **预训练结果**：两种方法在所有模型尺寸上均缩小TorchAO相对32-bit的gap；GPT-style 1.4B处最大相对减少**70.1%**；GPT-style 2.7B在冷却期末TorchAO出现剧烈发散（gap 1.5435），而ZE-EDEN和ZIP-SR在发散前已分别降低37.6%和62.1%。
- **SFT结果**：ZE-EDEN在Qwen3-8B上验证损失gap接近零（-0.00003），在Llama-3.2-3B上亦为负（-0.00035）；下游任务分数均接近32-bit基线。
- **存储成本**：所有4-bit方法统一使用block size 128，每参数1.0625字节（7.53×压缩），与TorchAO持平。

## 相关工作脉络
- **TorchAO/Li et al. (2023)**：提出4-bit AdamW并识别零点失败，采用排除零的Lin4-NZ码本；本文指出其舍入仍在状态空间，未考虑预条件器扭曲。
- **SOLO (Xu et al., 2025)**：分析无符号状态的信号淹没与方差放大；本文从其"零排除"路线出发引入EDEN校准作为互补方案。
- **FlashOptim (Gonzalez Ortiz et al., 2026)**：对二阶矩施加平方根压缩后再8-bit量化；本文保持原始尺度不变，仅改变舍入坐标。
- **Full-Stack FP4 (Ding et al., 2026)**：在更完整的低精度训练栈中使用transformed NVFP4；本文聚焦于舍入空间这一独立自由度。
- **BReD (Zhan et al., 2026)**：研究递归舍入偏差导致的持久漂移；本文理论层面分析了一步条件偏差，两者视角互补。
- **Q-Adam-mini (Han et al., 2025)**：对embedding层动量使用SR解决权重范数爆炸；本文将其思想推广至LM-head并在理论上解释。

## 局限性与未来方向
- 理论仅给出条件单步预条件器偏差分析与标量二次型的动力学比较，**未建立一般端到端收敛性保证**。
- LM-head SR有效缓解晚期回归，但**方向性一阶矩偏差的根源及其与学习率衰减的关系仍未阐明**。
- 实验仅在GPT/Llama架构上验证，**未扩展到非自回归架构或其他自适应优化器**（如AdaBelief、Muon）。
- 未探索**低于4-bit（如3-bit）**的进一步优化器状态量化。
- 论文承认使用了ChatGPT辅助表述Propositions和润色，存在AI辅助声明。

## 研究启发与可借鉴点
1. **舍入坐标作为独立设计自由度**：量化器不应仅优化状态空间重建误差，而应追踪该误差经下游非线性变换后的实际影响；此思路可迁移至权重/激活量化。
2. **匹配checkpoint干预（matched continuation）的定位方法**：通过冻结权重、优化器状态、数据顺序仅改变单一量化规则来隔离失效原因，是高成本训练实验中定位问题的有效范式。
3. **径向偏差（radial bias）诊断指标**：$b_{\mathrm{rad}}(\tilde{\mathbf{m}}_t)=\langle Q(\tilde{\mathbf{m}}_t)-\tilde{\mathbf{m}}_t,\tilde{\mathbf{m}}_t\rangle/\|\tilde{\mathbf{m}}_t\|_2^2$ 可有效捕捉确定性舍入在EMA递推中的累积方向性漂移。
4. **分层差异化量化策略**：对敏感层（LM-head、embedding）与非敏感层采用不同码本/舍入规则的组合，比全局统一策略更有效。
5. **EDEN校准的低成本适配**：将EDEN的内积归一化思想应用于任何引入正下界的量化场景（如exponent-only格式），均可缓解尾部压制。

## 关键术语表
- **Preconditioner space rounding（预条件器空间舍入）**：以AdamW更新公式中的预条件器$h_t(v)=1/(\sqrt{v}+\epsilon)$作为舍入坐标，使舍入决策直接优化实际参数更新而非状态值本身。
- **Zero-inclusive codebook（含零码本）**：量化码本中包含0这一重建级别，与排除零的码本相对；本文证明在正确舍入规则下可安全使用。
- **Stochastic rounding (SR)**：以与输入到相邻量化级距离成正比的概率随机选择较低或较高级别，使期望无偏。
- **Update-SR**：在预条件器空间$h_t$中计算SR概率的随机舍入，保证当前步预条件器的条件期望无偏。
- **Zero-point failure（零点失败）**：将小的正二阶矩量化为零后，下一步预条件器$1/\sqrt{v}$异常放大导致参数更新过大的病态现象。
- **EDEN calibration（EDEN校准）**：基于块内积与原向量能量的比值$c_\tau$对量化块进行尺度重校准，缓解正下界引起的系统性压制。
- **NF4 (NormalFloat 4-bit)**：基于标准正态分布分位数构造的4-bit量化码本，相比SDyn4/FP4在高密度区域分配更多重建级。
- **Radial bias（径向偏差）**：量化误差在原始向量方向上的投影归一化度量，用于诊断EMA递推中确定性舍入的累积方向性漂移。

## 可复现要素
- **代码**：PyTorch实现（ZIP-SR与ZE-EDEN及训练脚本）已开源：https://github.com/nubank/adamw4bit
- **数据集**：FineWeb-Edu（公开）；SFT使用Tulu-3（公开）
- **模型**：GPT-style（162M/405M/834M/1.4B/2.7B）与Llama-style（130M/350M/1.1B）架构见论文Table 6；SFT用Qwen3-8B-Base与Llama-3.2-3B
- **关键超参**：$\beta_1=0.9,\ \beta_2=0.95,\ \epsilon=10^{-8}$，梯度裁剪norm=1.0，block size=128，BF16混合精度+FP32主参数；预训练每参数20 tokens，WSD调度；SFT峰值lr=$10^{-5}$，1 epoch
- **硬件**：H200 GPU（8卡/节点），GPT 2.7B用B200
- **随机种子**：预训练3个配对种子，SFT 5个配对种子（42-46）
