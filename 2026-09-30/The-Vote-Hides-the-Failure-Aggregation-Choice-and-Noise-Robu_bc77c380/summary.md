---
title: "The-Vote-Hides-the-Failure-Aggregation-Choice-and-Noise-Robu"
source: https://arxiv.org/pdf/2609.37161v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:34:37"
---

# 论文速读：The Vote Hides the Failure: Aggregation Choice and Noise Robustness in Heart Murmur Detection

## 一句话总结
本文通过对照实验揭示，PCG 杂音检测模型的噪声鲁棒性结论高度依赖窗口/患者级聚合规则与评估指标的选择；在 CirCor DigiScope 上，BiLSTM 整体优于 HMS-Net，但聚合机制可能掩盖个体预测的真实退化，甚至使噪声增强训练后的 MV 准确率出现“虚假提升”。

## 研究问题与动机
- **核心问题**：现有 PCG 检测研究多聚焦架构性能，却忽视“聚合方式”与“评估指标”如何塑造噪声鲁棒性的最终结论。
- **现有不足 1**：多数工作将患者级投票视为工程收尾直接套用，未将其作为可被压力测试的独立变量。
- **现有不足 2**：单一总体准确率无法区分“聚合层面的表观稳定”与“窗口/片段级的真实失效”。
- **现有不足 3**：类高度不平衡（Absent 占 73.8%）场景下，Accuracy 与 W.acc 等指标对同一训练干预可能给出相反的统计显著性结论。

## 核心贡献（创新点）
1. **提出聚合规则与评估指标的联合压力测试框架**：在统一协议下对比 native 聚合与架构无关 MV，证明评估选择本身可决定鲁棒性结论。与以往仅报告端到端指标的惯例不同，本文首次将聚合机制置于噪声谱系中进行归因。
2. **发现并命名“聚合掩蔽效应”**：HMS-Net 的 native 规则在盐椒噪声下保持高准确率，实为窗口级分歧被其冗余规则吸收；BiLSTM 经噪声增强训练后 MV 准确率上升，但个体预测并未改善。与现有“指标提升即模型变强”的默认假设形成本质区别。
3. **揭示指标依赖性导致的统计结论翻转**：同一噪声增强干预在 Accuracy 下显著损害泛化，换用 W.acc 则完全不显著，原因在于 Absent 类 recall 下降被低权重平滑。区别于多数仅选用单一指标的同类工作，本文强调医疗评估需多指标交叉验证。
4. **提供可复现的双管线对照基准**：独立完成 HMS-Net 与 BiLSTM 的结构/行为对齐，并公开完整噪声注入、细分割与聚合对比协议，填补 CirCor 挑战赛背景下缺乏控制噪声多严重度跨架构对比的空白。

## 方法详解
- **数据集与划分**：使用 CirCor DigiScope（PhysioNet Challenge 2022），942 患者、3046 段录音，类别分布 Present 19.0% / Unknown 7.2% / Absent 73.8%。按患者 ID 字典序 round-robin 分 5 折，严格避免同患者录音跨折泄漏。
- **架构复现与校验**：HMS-Net（层级多尺度 CNN）与 BiLSTM（双向 LSTM）。前者通过 forward shape tracing 与 state-dict 映射对齐；后者逐行复现参考代码并核对层/参数计数。BiLSTM 改用患者级划分替代原片的 patch 级划分，属更保守的防泄漏设计。
- **噪声注入协议**：在原始波形上、管线预处理前注入四类噪声——AWGN、pink、uniform（{20, 10, 5, 0} dB）与 salt-and-pepper（{1, 3, 5, 10}%，振幅按局部标准差 k=3 缩放）。发现预处理会反向改变实际输入 SNR（BiLSTM +5~+12 dB，HMS-Net -4~-9 dB），因此保留全链路评估而非仅对特征输入加噪。
- **双聚合规则**：
  - *Native*：HMS-Net 按时间持续时间判段（Unknown 占比>0.8 或 Present≥3.0s），患者级 any-Present-wins；BiLSTM 选 P(Present) 最高段的 argmax。两者均由设计引入 Present-biased 倾向。
  - *MV*：架构无关的平坦投票，对所有片段预测简单取多数。
- **噪声增强微调**：每样本随机采样噪声类型与连续严重度（uniform-ablation 覆盖全范围；vulnerability-informed 将 AWGN/uniform 截断至 0~10 dB，salt-and-pepper 作为 held-out 泛化测试保留）。AdamW lr=1e-4，label smoothing 0.1，batch 128，20 epochs，ReduceLROnPlateau。
- **统计评估**：报告 Accuracy、Balanced Accuracy、W.acc（按 PhysioNet 2022 权重：Present×5 + Unknown×3 + Absent×1）。采用 per-fold 配对 t 检验（df=4）与 Bonferroni 校正（α=0.00625，8 cells）。

## 实验与结果
- **Clean 基线**：HMS-Net W.acc = 0.8021±0.0228，BiLSTM = 0.7464，贴近原论文 reported 0.81 与 0.757。
- **跨管线对比（匹配 MV 聚合）**：BiLSTM 在全部 16 个噪声条件下 Accuracy 与 W.acc 均领先 HMS-Net（16/16）；Balanced Accuracy 下 13/16 占优（3 个轻度例外<1.1pp）。低 SNR 差距剧烈，如 AWGN 0dB 时 HMS-Net Accuracy 仅 0.212，BiLSTM 达 0.809。
- **聚合掩蔽实证**：
  - HMS-Net：native 在盐椒噪声下准确率稳定，但实际为窗口级分歧被 native 冗余规则吸收；clean 时 native 比 MV 高 2.9pp（Present recall +30.7pp、Unknown recall -19.1pp），噪声加重后差距扩至 7.3~16.8pp，native 在 59/60 cells 中优于 MV（98.3%）。
  - BiLSTM：噪声增强后 MV W.acc 在所有条件下提升（+2.1~+5.3pp，2 cell 显著），native 精度持平或下降（-0.05~-3.3pp），sign test p≈0.008，表明训练仅改变了投票边际分布而未提升个体预测。
- **指标依赖性翻转**：噪声增强对 HMS-Net 在 mild salt-and-pepper 下的 Accuracy 显著下降 8.3pp（p<0.001），换用 W.acc 仅下降 0.74pp（p=0.37，不显著）。归因于 Absent recall 从 0.765 降至 0.630、Present recall 升至 0.834，W.acc 对 Absent 的最低权重（×1）使其损失被平滑掩盖。

## 相关工作脉络
- Patwa et al. [6] 对比 1D-CNN/LSTM 但剔除纯噪声段，未分析聚合规则误差来源；本文将其置于受控噪声与双聚合对照下进行压力测试。
- Oakden-Rayner et al. [7] 提出 hidden stratification 可掩盖子集失效；本文进一步指出 PCG 的窗口级分歧可通过聚合规则被“数学平滑”，机制不同但警示一致。
- Raghunathan et al. [8] 探讨鲁棒性-准确性权衡；本文发现噪声增强可提升聚合准确率却无益于个体预测，揭示“虚假鲁棒”这一特殊退化形态。
- Noman et al. [11] 指出 MV 可能遗漏低频病理事件；本文补充证明噪声会放大该缺陷，而 native 的 present-bias 反而形成被动补偿。
- Shariat Panah et al. [13] 发现噪声空间分布影响分类；本文延伸指出聚合选择与指标选择可与噪声分布共同决定最终结论，三者需联合考量。

## 局限性与未来方向
- 噪声均为合成注入，未验证真实临床干扰（环境音、探头脱落、患者运动）的泛化性。
- 未报告各严重度下逐录音的预测类别分布，仅呈现聚合后宏观指标。
- 每折仅用单一 seed，无法分离 seed 方差与 fold 方差。
- Unknown 类样本稀少（每折 10-16 患者），其 recall 估计噪声较大。
- Native 聚合均存在 present-biased 结构倾向，本文侧重分析其随噪声的变化量而非绝对值。
- 噪声增强后的 MCD vs. Deep Ensembles 校准
