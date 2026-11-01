---
title: "WHEN-KL-REGULARIZATION-MISFIRES-IN-GROUPPOLICY-OPTIMIZATION"
source: https://arxiv.org/pdf/2610.12161v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:14:57"
---

# 论文速读：WHEN-KL-REGULARIZATION-MISFIRES-IN-GROUPPOLICY-OPTIMIZATION

## 一句话总结
本文系统分析了群相对策略优化（GPO）中独立KL正则项与奖励更新交互的七种失效模式，并提出零和校准策略优化（ZCPO）。ZCPO利用全词表条件KL漂移构建组内零和联合系数，将参考策略信息从独立惩罚项转化为组内相对偏好信号，在数学推理与长上下文任务上显著提升训练效果与稳定性。

## 研究问题与动机
- **现象驱动的设计疑问**：DAPO、GMPO等近期方法去除显式reference KL惩罚反而提升训练效果（如PAPO去掉KL后Qwen2.5-VL-3B/7B avg@8提升约2-3个百分点），引发“参考策略信息应如何合理融入群相对更新”的核心问题。
- **独立KL与group-relative结构的结构性错配**：传统独立per-token KL正则项不与奖励系数的clip gate或组内归一化联动，导致reward梯度消失或抵消时KL梯度仍残留，形成训练噪声或方向冲突。
- **长度与采样带来的尺度失衡**：独立KL随响应长度累积，长样本受到不成比例的惩罚；同时基于采样log-ratio（k1）的估计存在单步条件采样噪声，组内中心化无法完全消除。
- **KL信号集中压制有益更新**：KL差异往往集中在少数token（如首token），当该局部拉力与奖励优化方向相反时，会压制模型在关键位置的有效策略改变。

## 核心贡献（创新点）
- **形式化七种KL失效模式（F1–F7）**：统一刻画了clip后残留、梯度抵消不同步、同奖组无效更新、长度累积失衡、聚合贡献偏移、信号集中压制及k1采样噪声等机制缺陷。与既有工作仅关注梯度偏置或异步训练不同，本文首次将KL与奖励clip/归一化/过滤的交互进行结构化诊断。
- **提出ZCPO零和校准系数**：用条件KL构建组内相对漂移 $K_i$，与最大绝对奖励归一化项线性组合得到联合系数 $C_i$，并通过stop-gradient冻结后替换基算法的优势系数。本质区别在于将KL从独立正则项转为组内相对校准信号，而非直接惩罚策略偏离。
- **给出系数最优性的闭式解**：证明在固定输入与组内零和约束下，$C_i$ 是二次校准问题的唯一全局最优解，具有共平移不变性，且能恢复被独立KL压制的弱奖励信号方向。
- **机制控制与多维度实验验证**：提供F1（clip处KL门控）、F3（同奖组过滤）等因果控制实验，并在数学推理、长上下文、跨模型（Qwen/Glm）三场景验证，展示ZCPO在性能与训练稳定性上的双重优势。

## 方法详解
- **条件KL计算**：对每个token位置 $t$ 与前缀 $h_{i,t}$，计算当前策略与参考策略在完整词表上的条件KL $\kappa_{i,t} = \text{KL}(\pi_\theta(\cdot|h_{i,t}) || \pi_{\text{ref}}(\cdot|h_{i,t}))$，再通过前向输出复用与fp32投影重算保证bf16训练精度，额外算力开销约0.5%。
- **相对漂移与归一化**：响应级平均条件KL $\widetilde{D}_i = D_i/T_i$，组内中心化得到 $K_i = \widetilde{D}_i - \frac{1}{G}\sum_j \widetilde{D}_j$；奖励优势用最大绝对值归一化 $A_i/s_q$（$s_q=\max_i|R_i-\bar{R}|$）替代标准差归一化，固定强信号峰值尺度。
- **零和联合系数**：$C_i = \text{sg}(A_i/s_q - \beta K_i)$，当 $G<2$ 或 $s_q=0$ 时通过门控 $B_q$ 直接跳过该组，避免无效更新。
- **代理目标集成**：将冻结的 $C_i$ 代入基算法（如DAPO）的surrogate objective，$\mathcal{L}_q = -\sum_i\sum_t w_{i,t} S_{\text{base}}(r_{i,t}; \text{sg}(C_i))$，梯度仅通过概率比 $r_{i,t}$ 传播，确保KL校准不干扰当前步的策略更新动力学。
- **最优性推导**：最小化 $\min_{\sum c_i=0}\{\frac{1}{2}\sum_i(c_i - A_i/s_q)^2 + \beta\sum_i c_i \widetilde{D}_i\}$ 的拉格朗日一阶条件直接导出 $c_i^\star = C_i$，严格凸性保证解的唯一性。

## 实验与结果
- **主实验设置**：基于DAPO框架，使用 Qwen2.5-32B 在 DAPO-Math-17K 上训练，评测 AIME24/AIME25 (avg@32)；基线含 DAPO±KL、RPG、GVPO、GOPO。
- **核心性能**：ZCPO 取得 AIME24 **56.8%**、AIME25 **48.5%**，较 DAPO without KL (47.3%/37.1%) 分别提升 **9.5/11.4** 个百分点，全面优于标准KL加扰基线（$\beta=0.04$ 时仅 31.2%/18.9%）及其他GPO变体（RPG 50.7%/42.5%，GVPO 52.6%/41.1%，GOPO 48.5%/39.9%）。
- **关键消融**：去除KL中心化（$K_i\leftarrow \widetilde{D}_i$）下降最剧烈（AIME24: 45.6%）；去除长度归一化（53.9%）、用k1替代条件KL（54.5%）、改用标准差归一化（55.4%）均劣于全模型，验证各组件协同必要性。
- **机制诊断结果**：
  - **F1**：vanilla在reward clip关闭的token处KL梯度占比约51%，过分配达68–172倍；ZCPO在自身gate关闭时双支梯度均为零。禁用clip处KL的F1控制实验准确率高于独立KL基线。
  - **F2**：按reward梯度抵消率分桶后，vanilla高/低抵消组间KL梯度份额相差约524倍，ZCPO仅3.45倍，跨组差异显著缩小。
  - **F4/F5**：vanilla平均响应长度随$\beta$上升而缩短（$\beta=0.04$时774，$\beta=4$时611），ZCPO在$\beta=4$时保持1695；长短组KL份额差异vanilla为12倍，ZCPO仅0.24个百分点。
  - **F6/F7**：ZCPO在首token处漂移幅度接近纯DAPO，中心化后负KL估计比例降至0%；k1对照在步骤3–10仍有42%–59%响应出现负估计，RMS残差约0.0026–0.0062。
- **跨模型与长上下文**：GLM-4-9B-0414上ZCPO从24.5%提升至36.2%；Qwen3-4B长上下文实验中ZCPO在DocMath/LBV2/Frames/MRCR/CorpusQA/LBV1-QA六项基准全面领先，含组门控（$B_q$）略优于全开门控版本。

## 相关工作脉络
- **GRPO族算法（DAPO/GMPO/GSPO等）**：既有工作聚焦clip策略、动态采样、几何平均聚合等结构改进，本文切入KL项与group-relative更新的交互失效，指出“保留参考信息但重构引入方式”的设计空间。
- **KL估计与放置分析（Vojnovic & Yun 2025; Tang & Munos 2025; Shah et al. 2026）**：集中于梯度偏置、leave-one-out基线或异步训练评估，未涉及将条件
