---
title: "THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIO"
source: https://arxiv.org/pdf/2610.17636v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-11-01 19:09:37"
field: "扩散语言模型安全对齐"
keywords: ["扩散语言模型", "安全引导", "激活干预", "去噪过程", "零样本对齐", "选择性门控"]
innovations: ["首次将激活方向引导系统引入扩散语言模型推理阶段，提出SADDER框架", "设计选择性二值门控机制，根据去噪动态得分仅在有害偏向位置施校正，平衡安全与质量", "提出Warm-up+Selective两阶段策略，并结合位置级安全分布理论分析"]
benchmarks: ["RealToxicityPrompts", "JailbreakBench", "AdvBench", "HarmBench", "XSTest", "BeaverTails"]
---

# 论文速读：THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIONAL-ACTIVATION

## 一句话总结
本文提出 **SADDER**（Steering via Activation Difference for Diffusion LLMs），一种面向扩散语言模型推理阶段的安全引导方法：通过构建有害/安全响应的隐藏层方向，在去噪过程中对特定位置施加选择性激活校正，从而在不依赖额外微调的前提下显著降低生成内容的有害性。

## 研究问题与动机
- **扩散语言模型的推理安全问题**：扩散语言模型（如 LLaDA、Dream）在续写和指令遵循中仍会产生有害内容，现有后训练对齐方法（RLHF/DPO）成本高且需重新训练。
- **激活控制（Activation Control）方法在扩散 LLM 上的适用性空白**：既有激活干预方法多针对自回归 LLM，扩散模型的去噪过程具有逐步去噪的独特时序结构，无法直接套用。
- **全位置引导的质量退化**：在全部 masked 位置无差别施加引导（Full SAD）虽能降低有害性，但会显著劣化 benign 质量（PPL 上升 75%+），需要位置选择机制。
- **门控信号的设计依据不足**：去噪过程中各位置的表征信息密度是动态演变的，已有工作未系统刻画这一特性并利用其指导干预位置的选择。

## 核心贡献（创新点）
1. **提出 SADDER 框架**：首次将基于激活方向的引导策略系统引入扩散语言模型的推理阶段，在零样本（no fine-tuning）条件下实现安全增强。
2. **选择性引导门（Selective Gating）**：设计二值门控 $a_{k,l}^{(t)}$，仅在中心化得分 $s_{k,l}^{(t)} > 0$ 的可编辑 masked 位置上施加引导，相比 Full SAD 在保持安全性的同时显著减少对 benign 质量的损害。
3. **Warm-up + Selective 的混合策略**：在前 $w$ 步使用 Full SAD（充分探索安全子空间），之后切换为 Selective（精细校准），进一步平衡安全与质量。
4. **位置级安全分布的理论分析（定理 A.1）**：证明位置级安全目标分布与全局安全分布的去噪器均值相等的充要条件是剩余位置安全权重与当前 token 的条件协方差为零，为位置级引导提供理论依据。
5. **系统性基准评测**：在 RTP、JailbreakBench、AdvBench、HarmBench、XSTest 等多个安全基准上评估，展示 SADDER 相比已有方法（CSD、FK、BoN）的竞争力。

## 方法详解
**核心思想**：给定 prompt $c_i$，收集有害响应 $\{x_i^{\text{harm}}\}$ 和安全响应 $\{x_i^{\text{safe}}\}$，在每一层 $k$ 计算两类响应的平均隐藏状态（centroid）$\bar{h}_k^{\text{harm}}$ 和 $\bar{h}_k^{\text{safe}}$，构建从安全指向有害的单位方向 $\hat{v}_k = \frac{\bar{h}_k^{\text{harm}} - \bar{h}_k^{\text{safe}}}{\|\bar{h}_k^{\text{harm}} - \bar{h}_k^{\text{safe}}\|}$，以及中点 $m_k = \frac{\bar{h}_k^{\text{harm}} + \bar{h}_k^{\text{safe}}}{2}$。

**引导得分**：对去噪步骤 $t$、位置 $l$ 的隐藏状态 $h_{k,l}^{(t)}$，计算中心化得分 $s_{k,l}^{(t)} = \langle h_{k,l}^{(t)} - m_k, \hat{v}_k \rangle$，正值表示该位置表征偏向有害侧。

**选择性引导门**：$a_{k,l}^{(t)} = \mathbb{1}[s_{k,l}^{(t)} > 0 \land l \in \text{editable}]$，仅对得分为正且处于可编辑（masked）状态的位置施加强度 $\eta \cdot \beta(\boldsymbol{x}_t)$ 的校正。

**完整 SAD（Full SAD）**：不区分正负得分，对所有 eligible masked 位置施加引导。

**Warm-up 策略**：前 $w=8$ 步使用 Full SAD 以确保充分的安全探索，之后切换为 Selective SADDER 进行精细引导。

**理论支撑**：定理 A.1 表明位置级安全去噪只需在剩余位置安全权重与当前 token 条件无关时等价于全局安全去噪，解释了为何位置级引导是合理的近似。

**关键恒等式**：$\langle h - m, \hat{v} \rangle > 0 \iff 2\langle h, \bar{h}^{\text{harm}} - \bar{h}^{\text{safe}} \rangle > \|\bar{h}^{\text{harm}}\|^2 - \|\bar{h}^{\text{safe}}\|^2$，避免显式计算单位向量。

## 实验与结果
**数据集**：
- **RTP**（RealToxicityPrompts）：600 条 benign + unsafe 池（Perspective≥0.8 & Llama-Guard 判定）
- **BeaverTails**：600 条含 unsafe 标签指令-响应对
- **WildJailbreak**：357 条 jailbreak unsafe 参考
- **JailbreakBench（JBB）**：100 种有害行为开发集 + 100–519 transfer 集
- **AdvBench**：对抗性 prompt
- **HarmBench**：held-out offset 100–199
- **XSTest**：250 safe + 200 unsafe，检测过度拒绝

**基座模型**：LLaDA-8B-Base / LLaDA-8B-Instruct、Dream-7B-Base / Dream-7B-Instruct

**攻击方式**：DIJA（adversarial prompt masked span 填充）、PAD（生成区插入固定 anchor）

**关键结果（LLaDA-Base, RTP 有害集, NFE=32）**：

| 方法 | Unsafe %↓ | Harmful PPL↓ | Benign PPL↓ | Benign BERT↑ |
|------|-----------|-------------|-------------|-------------|
| Unmodified | 38.6 | 21.1 | 17.5 | 84.65 |
| Full SAD | 32.4 | 29.3 | 30.0 | 83.92 |
| **SADDER (selective)** | **29.4** | 42.1 | 18.4 | 84.59 |
| SADDER + warm-up | 31.2 | 31.6 | 44.3 | 83.01 |

→ Selective SADDER 在 Unsafe% 上达到最优（29.4%，相对 unmodified 降低 23.8%），同时 Benign PPL 几乎不变（18.4 vs 17.5），体现选择性门控的有效性。

**Duo / UDLM 均匀状态续写结果（NFE=128）**：
- SAD（η=1.5×10⁵）在 Duo 上 Unsafe% 降至 41.1，但在 UDLM 上达 49.5；
- S_ADDER（k=12, η=5×10⁴）在 Duo 上 Unsafe%=43.0、UDLM 上 51.9，Benign BERT 保持在 81.86/80.69，显示出更好的质量-安全权衡。

**最佳超参**：LLaDA 用 k=31（近输出层）、η=100（selective）、窗口 [0,15]；Dream-Base 用 k=27、η=10000。

**最强结果**：Selective SADDER 在 LLaDA-8B-Base / RTP 上以 29.4% Unsafe% 和 84.59 BERTScore 的组合为最佳安全-质量平衡点。

## 相关工作脉络
1. **激活控制（Activation Intervention/Editing）**：已有工作（如 activation steering、directed pressure）主要面向自回归 LLM 的解码过程，本文将其适配到扩散模型的逐步去噪范式，是方法迁移的核心差异。
2. **CSD（Conditional Score Distillation）**：对比基线之一，CSD 需额外训练/微调条件去噪器，而 SADDER 是零样本推理期引导，无需修改模型权重。
3. **Filter-Kernel（FK）与 Best-of-N（BoN）**：这些方法通过重采样/多候选选择提升安全性，计算开销高（需 N 倍前向传递）；SADDER 仅需单次前向即可引导，NFE 不变。
4. **Dream / LLaDA 等扩散 LLM**：本文选取的代表模型，区别于传统文本扩散模型（如 DiffText），这些模型在 token 空间做扩散，激活引导策略需适配其架构特性。
5. **安全对齐方法（RLHF/DPO/GRPO）**：后训练对齐需大量计算和数据；SADDER 作为推理时方法，可作为对齐的补充手段而非替代。

## 局限性与未来方向
- **方向向量的静态性**：$\hat{v}_k$ 基于离线收集的安全/有害响应 centroid 计算，未考虑 prompt 内容的动态差异，对未见过的 prompt 分布可能泛化不足。
- **门控阈值的离散性**：当前使用严格二值门（sigmoid 软化未带来一致收益），在边界模糊位置上可能过于粗暴。
- **超参数敏感**：η 在不同模型/数据集间差异巨大（100→10⁵ 量级），缺乏自适应性；k（干预层）的选择也需手动调优。
- **Remasking 策略的取舍**：本文关闭 remasking（允许重新生成已输出 token），可能限制对早期错误 token 的修正能力，是未来可探索的方向。
- **评估覆盖度**：主要在英文基准上验证，对多语言、长文本、复杂指令场景的安全性尚未充分测试。

## 研究启发与可借鉴点
1. **去噪过程中的动态门控信号**：open-gate ratio 随去噪推进逐步上升这一发现具有重要方法论价值——未来可将其推广至其他扩散模型任务（如图像、多模态）的动态干预调度。
2. **位置级 vs 全局级安全分布的理论等价条件**：定理 A.1 给出了精确的可交换条件，可推广到多类（不止 safe/harmful）或细粒度安全属性的位置级控制。
3. **Warm-up + Selective 的两阶段策略**：先全量探索再精细校准的思路可迁移到其他需要平衡探索/利用的生成任务（如创意写作、代码生成）。
4. **零样本推理期安全的工程可行性**：SADDER 无需微调即可部署，适合快速响应新出现的有害 prompt 模式，可与在线安全 classifier 结合形成闭环。
5. **与对抗攻击防御的结合点**：DIJA、PAD 等攻击下 SADDER 仍保持一定鲁棒性，可探索将其作为对抗训练的正则项或防御后处理模块。

## 关键术语表
**SADDER**：Steering via Activation Difference for Diffusion LLMs，本文提出的扩散语言模型安全引导框架。
**Centroid（质心）**：同一类别（有害/安全）响应在某层隐藏状态的平均向量，用于定义安全-有害方向。
**Open-gate ratio**：去噪步中得分 $s_{k,l}^{(t)} > 0$ 的位置占总 masked 位置的比例，随去噪推进逐步上升。
**Selective SAD**：仅在正得分的可编辑位置上施加激活引导的门控版本。
**Full SAD**：在所有 eligible masked 位置统一施加引导，不区分得分正负。
**Warm-up**：前 $w$ 步使用 Full SAD 以充分探索安全子空间，之后切换为 Selective 的策略。
**NFE（Number of Function Evaluations）**：生成器前向传递次数，衡量推理计算开销。
**Over-refusal（过度拒绝）**：模型在 benign prompt 下也拒绝回答的安全过度敏感现象，由 XSTest 评估。

## 可复现要素
- **数据集**：RTP（公开）、BeaverTails（公开）、WildJailbreak（公开）、JailbreakBench（公开）、AdvBench（公开）、HarmBench（公开）、XSTest（公开）
- **代码/权重**：论文未明确声明开源状态；基座模型 LLaDA-8B 和 Dream-7B 有公开发布
- **关键超参**：k=31（LLaDA）/ k=27（Dream）、η=100（selective）/ η=3000（+warm-up）、w=8、32 new tokens、32 reverse steps、temperature=1
- **随机种子**：42, 43, 44；95% CI 通过 5000 次 prompt-cluster bootstrap 估计
