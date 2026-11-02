---
title: "Your-LR-Schedule-May-Not-Be-Principled-Effective-Learning-Ra"
source: https://arxiv.org/pdf/2610.40362v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:07:42"
field: "大语言模型预训练优化"
keywords: ["learning rate scheduling", "weight normalization", "effective learning rate", "large language model pretraining", "MoE", "optimization"]
innovations: ["揭示 ELR 轨迹是 WN 优于 WD 的真正驱动因素，匹配后训练动态几乎坍缩", "从方向对齐最大化理论导出 1/√t 最优衰减，构造 AIR 仿射调度器", "发现 late-stage 调度占最终损失差距约 90%，shared tail 可关闭 42%-94% 差距"]
benchmarks: ["MMLU", "MMLU-Pro", "AGIEval", "WinoGrande", "BBH", "DROP", "GSM8k", "MATH", "HumanEval", "MBPP+", "TriviaQA"]
---

# 论文速读：Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight-Normalized Training

## 一句话总结
本文发现 Weight-Normalized (WN) 训练相对于 Weight Decay (WD) 的优势本质上来自**有效学习率（ELR）轨迹**的不同，而非归一化本身；据此提出 AIR（Affine Inverse-square-Root decay）调度器，在 128B-token MoE 预训练中相比 Cosine 衰减将终端训练损失降低约 **0.01**，并显著提升下游性能与超参鲁棒性。

## 研究问题与动机
1. **WN 与 WD 训练动态差异大但机制不清**：相同 LR 调度下，WN 在 Cosine 衰减时最终优于 WD，但在 WSD 下持续落后，为何如此？
2. **ELR 是驱动因素而非名义 LR**：有效学习率 $\eta^{\text{eff}} = \eta / r$（$r$ 为参数块范数）决定更新方向的变化幅度；匹配 ELR 轨迹后，WN 与 WD 的训练损失动态几乎完全收敛（Figure 2）。
3. **现有调度设计仅针对名义 LR，不够 principled**：WD 下名义 LR 与参数范数耦合，无法直接复现目标 ELR 轨迹；WN 下 $r_k = R$ 固定，ELR 正比于名义 LR，是最干净的设计环境。
4. **Late-stage 调度对最终损失的影响被低估**：共享 late 阶段可大幅缩小不同早期调度造成的最终损失差距（90% 的差距来自 late 阶段），解释了为何后期 LR 设计尤为关键。

## 核心贡献（创新点）
1. **揭示 ELR 轨迹是 WN 优于 WD 的真正驱动因素**：通过 norm replay 和 blockwise LR rescaling 两种 ELR 匹配实验，证明匹配后训练动态几乎坍缩；与已有工作（仅指出 ELR 重要）的本质区别在于**定量展示了匹配后动态坍缩并据此指导调度设计**。
2. **提出 AIR（Affine Inverse-square-Root decay）调度**：从 WN 下最大化方向对齐的理论推导得出 $1/\sqrt{t}$ 最优衰减，再用仿射变换适配有限训练窗口的峰值/终端 LR；与同类方法的区别在于**从优化目标（方向对齐）出发而非经验启发**。
3. **发现并理论解释 late-stage scheduling 的关键作用**：通过 stage swap 实验和理论界证明，shared late schedule 能衰减早期差异（约 90% 的差距被消除），揭示了调度设计应优先关注后期阶段。
4. **128B-token MoE 大规模实证验证**：在 110M/4.7B 和 790M/35.7B 两个尺度上验证 AIR 优于 Cosine 及其他候选调度族，终端损失降约 0.01，下游平均提升显著。

## 方法详解

**有效学习率定义：**
$$\eta_k^{\text{eff}} = \frac{\eta_k}{r_k}$$
其中 $r_k = \|\pmb{\theta}_k\|$ 为参数块范数。

**ELR 匹配实验：**
- **Via Norm**：在 WN 训练中 replay WD 的范数轨迹，保持名义 LR 不变，使 ELR 等于 WD 的 ELR。
- **Via LR**：保持 WN 固定范数 $R$，按 $\eta_k^{\text{WN}} = R \cdot \eta_k^{\text{WD}} / r_k^{\text{WD}}$ 调整各块名义 LR。

**AIR 调度推导：**
- 在切向更新假设（$\pmb{\theta}_k^\top \pmb{u}_k = 0$）下，方向对齐 $m(t)$ 满足连续 ODE：
$$\dot{m}(t) = a\eta(t) - b\eta(t)^2 m(t)$$
- 最大化终端对齐 $m(h)$ 得到无约束最优解：$\eta^\star(t) = \eta_0 / \sqrt{1 + 2b\eta_0^2 t}$，即 $1/\sqrt{t}$ 型衰减。
- **AIR 实用形式**（仿射变换，满足有限窗口内指定峰值 $p$ 和终端 $\eta_{\min}$）：
$$q(k) = \left(1 + \frac{k}{\alpha}\right)^{-1/2},\quad f(k) = \frac{q(k) - q(K)}{1 - q(K)},\quad \eta_{\text{AIR}}(k) = \eta_{\min} + (p - \eta_{\min}) f(k)$$
其中 $\alpha = K/8$ 为时间尺度超参。

**Late-stage 理论分析：**
- 若共享 late schedule $[K_s, K]$，则早期差异以因子 $H(K, K_s) = \exp[-b\int_{K_s}^K \eta(k)^2 dk] \leq 1$ 衰减，符号不变。
- 后期 LR 差异对终端对齐的影响更大（因为后续衰减因子更弱）。

## 实验与结果

**实验设置：**
- **模型**：MoE，两档规模——110M/4.7B（128B tokens，16 GPU，48h/次）和 790M/35.7B（500B tokens，32 GPU，168h/次）。
- **优化器**：Muon，WN 约束，Frobenius 范数。
- **比较控制**：所有调度共享相同 warmup、终端 LR 和 cumulative LR（LR 总和）；peak LR 按离散 LR 求和校准（Appendix B.6）。
- **对比基线**：Cosine、Linear、Half-cosine、Rear-cosine、Cycloid、Double-cosine、$(1-t)^2$、$1-t^\beta$、$1-\log t$、$e^{-t}$、$1/t^2$、$1/t$、$t^{-\beta(t)}$。

**主要结果（Table 1）：**

| 模型规模 | Schedule | Train Loss ↓ | Eval Loss ↓ | Avg Downstream ↑ |
|---|---|---|---|---|
| 110M/4.7B | **AIR** | **1.730** | **2.174** | **29.97** |
| 110M/4.7B | Cosine | 1.740 | 2.185 | 29.24 |
| 790M/35.7B | **AIR** | **1.421** | **1.832** | **57.78** |
| 790M/35.7B | Cosine | 1.431 | 1.843 | 56.22 |

- AIR 较 Cosine 终端训练损失降低约 **0.01**（两档规模均一致）。
- 下游平均 MMLU/MMLU-Pro/AGIEval/WinoGrande/BBH/DROP/GSM8k/MATH/HumanEval/MBPP+/TriviaQA 均最优或接近最优。
- **超参鲁棒性**（Table 5）：AIR 的 LR 敏感度最低（Train Sensitivity 0.0018，Eval 0.0018），在宽 peak LR 范围内保持低损失（Figure 4）。

**Late-stage 干预结果（Table 2, Figure 5）：**
- 共享 AIR late 阶段可关闭 Cosine 等调度与 AIR 之间 **42%–83%**（训练损失）和 **45%–94%**（评估损失）的差距。
- AIR 与 Cycloid 共享 peak/terminal/cumulative LR 时，loss gap 约 0.005 几乎完全来自 late 阶段差异。

## 相关工作脉络
1. **Weight Decay vs. Weight Normalization**：WD [20] 通过衰减系数收缩参数，WN [33] 固定参数范数；本文指出二者差异本质是 ELR 轨迹不同。
2. **ELR 驱动训练动态**：D'Angelo et al. [5] 和 Kosson et al. [13] 已指出 ELR 的重要性；本文进一步展示 ELR 匹配可使 WN 与 WD 动态坍缩，并据此指导调度设计。
3. **Concurrent ELR 研究**：Liu et al. [17]、Xiao et al. [34]、Zhou et al. [38] 同期独立报告相似发现，但未从优化目标推导衰减律；本文从方向对齐最大化导出 $1/\sqrt{t}$ 并构造 AIR。
4. **LR Schedule 设计传统**：Cosine [19]、WSD [10]、power decay [24] 等均仅针对名义 LR；本文将其对象升级为 ELR。
5. **River-valley 分析与 convex bounds**：Wen et al. [32]、Schaipp et al. [24] 用不同理论框架分析 LR 调度，但未处理参数范数演化带来的耦合问题；本文在 WN 解耦环境下给出精确理论。

## 局限性与未来方向
1. **理论假设简化**：方向对齐分析基于固定理想方向 $\pmb{v}$、切向更新假设和常系数 $a,b$；实际训练中更新统计量和目标方向会持续演化，理论与实际的定量联系仍需深化。
2. **仅在小/中规模 MoE 上验证**：实验在 110M 和 790M activated 参数上进行，尚未在千亿级参数模型上验证 AIR 的扩展性。
3. **未覆盖继续预训练与微调**：结论主要针对 pretraining；continued training (CT)、SFT 和 RL 阶段的 ELR 调度尚未探索。
4. **其他架构/优化器泛化待验证**：当前实验基于 Muon + WN，未测试 AdamW + WD 等其他组合。

## 研究启发与可借鉴点
1. **ELR 作为调度设计的第一性原理**：将调度目标从名义 LR 升级为 ELR，适用于任何参数范数变化的训练场景（含 WD），值得在其他优化器/架构上复现验证。
2. **AIR 的仿射 $1/\sqrt{t}$ 构造可迁移**：利用 $f(k) = (q(k)-q(K))/(1-q(K))$ 将理论最优曲线适配有限窗口的方法，可推广到其他衰减形状的设计。
3. **Late-stage 敏感度分析作为调试工具**：通过 stage swap 实验分离 early/middle/late 各阶段贡献，是一种系统性的调度诊断方法，可用于解释不同调度性能差异的来源。
4. **离散 LR sum 校准技术（Appendix B.6）**：通过求解离散线性方程确定 peak LR 以匹配指定 cumulative LR，确保了跨调度比较的公平性，可在任何需要 LR 预算匹配的场景中复用。

## 关键术语表
- **Effective Learning Rate (ELR)**：名义学习率与参数范数的比值 $\eta/r$，刻画每次更新对参数方向的相对影响强度。
- **Weight Normalization (WN)**：每步更新后将参数块范数重新缩放至固定值 $R$ 的优化技术，与 weight decay 互为替代。
- **AIR (Affine Inverse-square-Root decay)**：本文提出的 LR 调度，基于 $1/\sqrt{t}$ 衰减律经仿射变换适配有限训练窗口。
- **Directional Alignment**：参数向量与理想目标方向之间的余弦相似度，是本文理论分析的核心目标量。
- **Cumulative LR**：训练全程所有步骤 LR 的总和，用于公平跨调度比较的 LR 预算约束。
- **Muon Optimizer**：用于本实验的_hidden layer_ 优化器，结合 WN 取得更好效果 [11]。
- **LR Sensitivity**：衡量调度性能对 peak LR 选择的敏感程度，定义为平均损失超优值 per e-fold peak LR 变化。
- **Stage Swap**：将两个调度在不同时间段（early/middle/late）的 LR 曲线互换的实验干预方法。

## 可复现要素
- **数据集**：Mixture-of-Experts 预训练数据（128B/500B tokens），论文未明确公开数据集名称。
- **代码/权重**：论文未提及开源；作者单位含 ByteDance Seed，代码未公开声明。
- **关键超参**：
  - AIR 时间尺度 $\alpha = K/8$（$K$ 为 decay 总步数）
  - 终端 LR $\eta_{\min} = 10^{-5}$
  - Warmup 步数：3,040（小模型）
  - Global batch size：128
  - Normalization epsilon：$10^{-8}$
  - 优化器：Muon

---
