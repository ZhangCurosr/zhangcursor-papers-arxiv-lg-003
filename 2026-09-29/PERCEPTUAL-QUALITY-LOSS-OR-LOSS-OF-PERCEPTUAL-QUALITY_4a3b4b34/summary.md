---
title: "PERCEPTUAL-QUALITY-LOSS-OR-LOSS-OF-PERCEPTUAL-QUALITY"
source: https://arxiv.org/pdf/2609.35054v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:53:20"
field: "语音增强评估"
keywords: ["speech enhancement", "PESQ loss", "perceptual quality", "metric optimization", "listening experiment", "Goodhart's law"]
innovations: ["系统对比可微分PESQ损失与GAN度量损失在SE中的实际效果", "主观听觉实验揭示无PESQ损失模型在不匹配条件下显著更优", "量化复合指标中PESQ的主导性（最高91.1%方差解释）"]
benchmarks: ["VB-DMD", "EARS-WHAM v2"]
---

# 论文速读：PERCEPTUAL-QUALITY-LOSS-OR-LOSS-OF-PERCEPTUAL-QUALITY

## 一句话总结
本文系统评估了语音增强（SE）模型训练中引入 PESQ 损失函数的实际效果，发现虽然优化 PESQ 损失确实能显著提升测试集上的 PESQ 分数，但在多数其他客观指标和主观听觉实验中并未带来改进，甚至在不匹配数据上反而有害。

## 研究问题与动机
- **现状**：当代深度 SE 模型常在损失函数中加入可微分 PESQ 或 GAN 式度量损失作为辅助项，以期提升感知质量。
- **问题**：更高的 PESQ 分数并不必然对应更好的听觉体验，甚至可能因 Goodhart's law（"当指标成为目标，便不再是好的指标"）产生反效果。
- **已知局限**：先前工作（如 PESQetarian）已证明极端优化 PESQ 会导致听觉性能崩溃，但实践中 PESQ 损失通常与其他距离损失（MSE/SI-SDR）混合使用，其真实收益尚不明确。
- **研究空白**：缺乏对两类代表性 PESQ 损失方案（可微分 PESQ 与 GAN 度量损失）在完整指标套件和主观实验下的系统性对比分析。

## 核心贡献（创新点）
1. **系统性对比两种 PESQ 损失方案**：分别基于 SB-SGMSE+（可微分 PESQ 损失）和 SEMamba（GAN 度量损失）评估不同 PESQ 权重对 SE 性能的影响，揭示了混合训练场景下 PESQ 损失的真实作用。
2. **主观听觉实验验证客观结果**：通过盲测配对听觉实验（12 位音频专家，双数据集）发现无 PESQ 损失模型在主观偏好上显著优于含 PESQ 损失模型，尤其在不匹配条件下差距扩大至约 15 个百分点。
3. **复合指标中 PESQ 的主导性量化**：方差分解分析表明 PESQ 在 CSIG/CBAK/COVL 中分别贡献 59.2%/69.0%/91.1% 的解释方差，揭示了复合指标过度依赖 PESQ 的问题。
4. **不匹配数据上的负效应发现**：GAN 式 PESQ 损失在不匹配数据集（EARS-WHAM）上不仅未能提升泛化性能，反而导致 PESQ 和其他指标全面下降，归因于判别器在小规模 VB-DMD 上过拟合。

## 方法详解
**两种 PESQ 损失实现方式：**

1. **可微分 PESQ 损失（Differentiable PESQ Loss）**
   - 基于 torch-pesq 包，跳过原始 PESQ 的时间对齐，通过 IIR 滤波进行电平对齐。
   - 在 SB-SGMSE+ 扩散模型中加入加权 $L_1$ + PESQ 损失，超参 $\alpha_P \in \{0, 0.0005, 0.001\}$ 对应 M5（无 PESQ）、M7（PESQ-mid）、M6（PESQ-high）三个模型。
   - 作者建议使用 SI-SDR 损失配合 PESQ 损失以避免不良结果。

2. **GAN 度量损失（GAN Metric Loss）**
   - 继承 MetricGAN 思路，用判别器 D 将 PESQ 作为黑盒预测目标（输出缩放到 [0,1]）。
   - 判别器损失：$\mathcal{L}_D = \mathbb{E}[\|D(X_m, X_m) - 1\|^2] + \mathbb{E}[\|D(X_m, \hat{X}_m) - Q_{\text{PESQ}}\|^2]$
   - 生成器度量损失：$\mathcal{L}_{\text{Metric}} = \mathbb{E}[\|D(X_m, \hat{X}_m) - 1\|^2]$
   - 在 SEMamba 模型上比较有无该损失的两种配置。

**实验设置：**
- 数据集：VB-DMD（匹配，824 样本，SNR 2.5~17.5 dB）和 EARS-WHAM v2（不匹配，886 样本）。
- 客观指标：PESQ、POLQA、ESTOI、segSNR、SI-SDR、CSIG/CBAK/COVL、DistillMOS、SCOREQ、WAcc（QuartzNet 15x5 和 Parakeet CTC 0.6B）。
- 主观实验：盲测配对偏好测试，12 位专家，每样本 3 次评估，Bradley-Terry 模型排序。

## 实验与结果
**匹配数据集（VB-DMD）：**
| 模型 | PESQ | CSIG | COVL | ESTOI | SI-SDR |
|------|------|------|------|-------|--------|
| SB-SGMSE+ M5 (no PESQ) | 2.91 | 4.03 | 3.53 | **0.88** | **19.43** |
| SB-SGMSE+ M7 (PESQ-mid) | 3.56 | 4.63 | 4.21 | 0.87 | 13.21 |
| SB-SGMSE+ M6 (PESQ-high) | **3.73** | **4.69** | **4.33** | 0.86 | 7.70 |
| SEMamba w/o metric loss | 3.35 | 4.71 | 4.13 | **0.89** | **20.11** |
| SEMamba w/ metric loss | **3.42** | **4.72** | **4.17** | **0.89** | 19.70 |

**不匹配数据集（EARS-WHAM）：**
- SB-SGMSE+ M5 在 segSNR、SI-SDR、ESTOI 上显著优于 M6（最高 SI-SDR 12.67 dB vs 3.23 dB）。
- SEMamba w/o metric loss 在几乎所有指标上优于 w/ metric loss（如 SI-SDR 12.06 dB vs 9.96 dB，ESTOI 0.78 vs 0.76）。

**主观听觉实验：**
- VB-DMD：M5（no PESQ）在 Bradley-Terry 排序中得分 0.534，M7 得 0.371，M6 仅 0.095。
- EARS-WHAM：无 PESQ 模型胜率领先约 15 个百分点；SEMamba w/o metric loss 显著优于 w/（binomial test, p=0.030）。

**复合指标方差分析：**
PESQ 在 CSIG、CBAK、COVL 中的方差解释占比分别为 59.2%、69.0%、91.1%，WSS 贡献不足 1%。

## 相关工作脉络
- **MetricGAN (Fu et al., ICML 2019)**：开创 GAN 度量损失范式，将非可微指标转为可微代理，本文在其基础上进一步验证 PESQ 作为度量目标的局限性。
- **CMGAN (Cao et al., Interspeech 2022)**：采用 MSE + 多种度量损失，使用与本文类似的复合指标套件评估，本文揭示其评估套件中 PESQ 主导方差的问题。
- **MP-SENet (Lu et al., Interspeech 2023) & SEMamba (Chao et al., SLT 2024)**：使用 GAN 度量损失的现代 SE 架构，本文通过消融实验验证此类设计的实际收益有限。
- **SB-SGMSE+ (Richter et al., ICASSP 2025)**：扩散模型 SE，引入可微分 PESQ 损失，本文展示 PESQ 权重对 SI-SDR/ESTOI 的负面影响。
- **PESQetarian (de Oliveira et al., Interspeech 2024)**：极端优化 PESQ 的对照实验，本文将其结论扩展到更现实的"混合损失"设置。

## 局限性与未来方向
- 主观实验仅 12 位音频专家参与，样本量有限，可能影响统计效力。
- 仅针对两个代表性模型（SB-SGMSE+ 和 SEMamba）进行评估，结论的普适性需更多模型验证。
- GAN 判别器在 VB-DMD 上同时训练，可能因数据量小和 SNR 范围有限而过拟合，泛化能力存疑。
- 未来工作可探索更鲁棒的感知损失设计、更多样化的受试者池、以及在不平衡数据集上的验证。

## 研究启发与可借鉴点
1. **多指标综合评估的重要性**：单一 PESQ 优化可能损害 SI-SDR、ESTOI 等关键指标，建议评估时覆盖声学质量、可懂度、ASR 性能等多维度。
2. **复合指标的重新审视**：CSIG/CBAK/COVL 过度依赖 PESQ（最高 91.1% 方差），可考虑引入更均衡的感知度量（如 POLQA、DistillMOS、SCOREQ）替代或补充。
3. **不匹配条件下的敏感性分析**：实际部署常面临域偏移，论文展示了在 EARS-WHAM 上不匹配条件下 PESQ 损失的负面影响更显著，建议在泛化基准上验证损失设计。
4. **GAN 度量训练的域适应风险**：判别器与生成器同步训练时易过拟合训练分布，未来可探索冻结判别器、正则化或域扩展策略。

## 关键术语表
- **PESQ**：Perceptual Evaluation of Speech Quality，ITU-T P.862 标准，基于参考信号的语音感知质量客观评估指标，得分范围 1~5。
- **POLQA**：Perceptual Objective Listening Quality Analysis，ITU-T P.863 标准，PESQ 的继任者，支持全带音频和现代通信系统。
- **SI-SDR**：Scale-Invariant Signal-to-Distortion Ratio，尺度不变信号失真比，衡量增强信号与参考信号之间的失真程度。
- **ESTOI**：Extended Short-Time Objective Intelligibility，扩展短时客观可懂度，基于时间包络频谱相关性的可懂度预测指标。
- **Goodhart's law**：古德哈特定律，指"当一项指标成为目标时，它便不再是一个好指标"，常用于描述指标优化导致的意外后果。
- **CSIG/CBAK/COVL**：CMGAN 评估套件中的三个复合感知指标，分别预测语音清晰度、背景噪音感和整体语音质量。
- **Bradley-Terry 模型**：用于配对比较数据的统计排序模型，将 pairwise 胜率转换为各模型的绝对能力分数。
- **SCOREQ**：基于对比学习的深度语音质量评估模型，支持参考模式（REF）和非参考模式（NR）两种评估方式。

## 可复现要素
- **数据集**：VB-DMD 和 EARS-WHAM v2，论文声明可用（EARS-WHAM 为公开基准）。
- **代码/权重**：SB-SGMSE+ 权重由作者提供；SEMamba 使用官方代码（作者提供），batch size=12，learning rate=0.001。
- **关键超参**：PESQ 损失权重 α_P ∈ {0, 0.0005, 0.001}（SB-SGMSE+）；SEMamba 使用默认损失组合。
