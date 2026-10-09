---
title: "Understanding-and-Mitigating-Token-Pruning-Induced-Vulnerabi"
source: https://arxiv.org/pdf/2610.09703v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:07:20"
---

# 论文速读：Understanding-and-Mitigating-Token-Pruning-Induced-Vulnerabi

## 一句话总结
本文首次系统评估了 Token-Pruning 对视觉语言模型（VLMs）在多模态越狱场景下的安全性影响，揭示了“剪枝诱导恶意放大”机制（背景良性 token 被剔除导致注意力坍塌至前景恶意 token），并提出免训练的推理期即插即用防御框架 SAP，在几乎不损失加速收益与任务性能的前提下，将越狱攻击成功率（ASR）最高降低 62%。

## 研究问题与动机
- Token-Pruning 虽能显著降低 VLM 推理成本，但其安全性影响长期被忽视；已有参数剪枝研究提示剪枝可能放大安全风险，但视觉-文本联合架构下的 Token 级剪枝是否会引入新型漏洞尚不明确。
- 主流剪枝策略（Vision-Centric、Text-Guided）在提升压缩率时，ASR 显著上升，且安全性退化并非由模型容量下降引起，而是与剪枝算法的选择性偏差密切相关。
- 实验中发现 Query-based Compression（如 LLaVA-Mini 在 99.8% 压缩率下）反而意外提升安全性，表明效率与安全性并非天然 trade-off，亟需阐明不同剪枝范式重塑安全行为的底层机制。

## 核心贡献（创新点）
1. **首次系统性揭示 Token-Pruning 的安全脆弱性**：建立 8 种代表性剪枝策略在三大安全基准上的评测基线，指出除 Query-based 外多数方法会因注意力坍塌显著加剧越狱风险。*与已有工作的区别*：打破了“剪枝仅影响推理效率”的传统认知，将安全视角引入 VLM 压缩领域。
2. **提出“剪枝诱导恶意放大”（Pruning-Induced Malicious Amplification）机制**：从信息论与能量竞争角度形式化证明，标准 Top-K 剪枝剔除安全缓冲 token 后，注意力质量被迫汇聚至残留恶意锚点，导致恶意语义密度 $\rho \to 1$。*与已有工作的区别*：超越经验性安全评测，给出可量化的注意力失衡理论解释。
3. **设计免训练推理期防御模块 SAP**：通过恶意锚点识别、背景良性 token 恢复、注意力重分配三步即可插即用修复安全性，无需重新微调或改变模型架构。*与已有工作的区别*：区别于外部 Guardrail 或训练期对齐，直接在推理时动态修正注意力分布，兼顾零成本部署与原生加速收益。

## 方法详解
- **理论建模**：将视觉 token 空间划分为恶意子空间 $\mathcal{M}$ 与安全缓冲子空间 $\mathcal{B}$，定义恶意能量 $S=\sum_{j\in\mathcal{M}}\exp(q\cdot k_j)$、缓冲能量 $N=\sum_{m\in\mathcal{B}}\exp(q\cdot k_m)$ 与恶意语义密度 $\rho=S/(S+N)$。标准剪枝物理移除 $\mathcal{B}$ 导致 $N\to 0$，$\rho$ 趋近 1，注意力分布熵骤降，越狱风险飙升。
- **SAP 三阶段设计**：
  1. **Malicious Anchor Identification (MAI)**：联合注意力强度与安全语义偏离度计算锚点得分 $MSA_{score}(v_i) = \left[\frac{1}{|\mathcal{P}|}\sum_{t\in\mathcal{P}}\mathbf{A}[t,v_i]\right]\cdot\left(1-\frac{v_i\cdot v_{\text{safe}}}{\|v_i\|_2\|v_{\text{safe}}\|_2}\right)$，其中 $v_{\text{safe}}$ 为单次前向传播预计算的安全对齐隐层向量。取 Top-$m$ 得分最高 token 构成恶意锚点集 $\mathcal{T}_{msa}$。
  2. **Benign Token Restoration (BTR)**：从被丢弃集合 $\mathcal{V}_{drop}$ 中均匀采样 $k$ 个背景 token 作为 $\mathcal{V}_{restored}$，替换当前保留集 $\mathcal{V}_{keep}$ 中得分最低的 $k$ 个 token，保持序列长度 $| \mathcal{V}_{active} | = | \mathcal{V}_{keep} |$ 不变，重建语义缓冲。
  3. **Attention Reallocation (AR)**：在每个解码步对恶意锚点的 post-softmax 注意力施加衰减因子 $\lambda$，释放的质量 $\Delta=\sum_{j\in\mathcal{T}_{msa}}\lambda\mathbf{A}_t[v_j]$ 均匀重分配至恢复的良性 token，数学上等价于构建混合分布 $\mathbf{A}_{SAP}=(1-\lambda)\mathbf{A}_{pruned}+\lambda\mathbf{A}_{restored}$，理论证明可抬高 Shannon 熵下界并约束 KL 散度扩张。

## 实验与结果
- **数据集与基准**：安全基准 MM-SafetyBench、FigStep、JailBreakV-28K；效用基准 MMBench、MM-Vet、LLaVA-Bench、ScienceQA (SQA)。基座模型 LLaVA-1.5-7B，默认剪枝率 75%。
- **主要结果**：
  - 未防御状态下，Text-Guided 类剪枝（TRIM、FitPrune、SparseVLM）平均 ASR 较原始模型上升约 8–10%，Vision-Centric 类同样显著恶化。
  - 引入 SAP 后，五种主流剪枝方法安全性全面回升，平均 ASR 绝对降幅最高达 62%（SAP† 扩展至文本模态后，FasterVLM 在 MM-Safety 上从 72.02% 降至 10.32%）；纯视觉 SAP 亦实现 19%–27% 的相对降幅。
  - 效用与效率几乎无损：MMBench 等基准波动控制在 ±2% 以内，理论 FLOPs 不变，单卡 RTX 4090 延迟仅增加约 1.3 ms（TRIM 55.90 ms → TRIM+SAP 57.19 ms）。
  - 对比外部 Guardrail LLaVAGuard，SAP 达成相近安全性但延迟低 3
