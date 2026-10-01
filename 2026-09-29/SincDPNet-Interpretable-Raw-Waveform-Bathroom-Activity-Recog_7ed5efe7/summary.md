---
title: "SincDPNet-Interpretable-Raw-Waveform-Bathroom-Activity-Recog"
source: https://arxiv.org/pdf/2609.34907v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:21:35"
field: "边缘声学事件分类与可解释 TinyML"
keywords: ["Acoustic Event Classification", "Raw-Waveform Learning", "SincNet", "Interpretable Deep Learning", "TinyML", "Bathroom Activity Recognition", "Multi-Objective Bayesian Optimization", "Edge AI"]
innovations: ["提出 SincDPNet：将可学习 sinc 带通前端与深度可分离卷积体结合，在仅 2,848 参数下实现可解释的原始波形浴室事件分类", "发布 SnaanGhar7：首个含门/助行器/非目标类、采用环境不交叉划分的卫生间声学事件公开数据集", "以 MOBO 探索 accuracy–size Pareto 前沿，证明结构化 sinc 前端在千参数量级下显著优于无约束卷积前端"]
benchmarks: ["SnaanGhar7 (environment-disjoint split)", "LOEO cross-validation across 5 environments", "Raspberry Pi Zero 2 W deployment (RTF evaluation)"]
---

# 论文速读：SincDPNet-Interpretable-Raw-Waveform-Bathroom-Activity-Recog

## 一句话总结
本文提出了**SincDPNet**，一种面向辅助生活场景中卫生间声学事件识别的可解释轻量级原始波形分类器；同时开源了首个包含门、助行器等多类别及非目标类的**SnaanGhar7**数据集（21,387 个标注片段），在环境不交叉划分下实现了最优 80.2% 准确率/0.760 macro-F1（仅 14,040 参数）与最高效率点 75.7% 准确率/0.661 macro-F1（仅 2,848 参数）。

## 研究问题与动机
- **隐私敏感场景缺乏专用音频基准**：ESC-50、UrbanSound8K、AudioSet 等通用数据集几乎不含卫生间活动类，已有卫生间语料（Chen et al.、Öztürk et al.）主要覆盖水流类事件，缺少门开关、助行器/拐杖等辅助生活相关类别，也缺少显式的非目标背景类。
- **模型可解释性与误判归因需求强**：辅助监测场景不仅要高精度，还需要能解释"为何把冲水误判为开水龙头"——现有 MFCC+CNN 管线缺乏物理含义映射。
- **边缘部署的体积/精度双重约束**：DCASE 低复杂度任务要求 ≤128 kB / ≤128 k INT8 参数，现有工作多为"先做大再剪枝/量化"，缺少"从数千参数起点"的一体化设计范式。
- **评估协议易被房间特性污染**：随机 clip 级划分会让同房间片段出现在训练/测试中，导致性能虚高；需要 session 与 environment 双重隔离的划分方式。

## 核心贡献（创新点）
1. **SnaanGhar7 数据集**：5 个真实卫生间环境、7 类（含非目标 Unknown）共 21,387 个 1.5 s 窗口片段，公开并附 baseline；**与既有卫生间语料（如 Öztürk 460 min/11 类）的本质区别在于首次把门、助行器/拐杖与非目标类纳入同一基准，并提供环境不交叉划分协议。**
2. **SincDPNet 架构**：可学习 sinc 带通滤波器组（仅 $2N_f$ 个频率参数）接深度可分离卷积体；**与 SincNet 的本质区别在于 sinc 之后的骨干由全连接+标准卷积改为极轻量 separable body，使整体参数从 ~13 万降至几千量级。**
3. **基于多目标贝叶斯优化的设计探索管线**：以 24 种配置搜索 accuracy–size / accuracy–latency Pareto 前沿，并用 quality-based coreset（top-25% clip）加速候选评估；**定位上不是提出新 BO 算法，而是把 MOBO 作为工程选型工具，提供可复现的 TinyML 音频模型搜索范式。**
4. **声学分析–模型误差的桥梁验证**：训练前计算类间 Bhattacharyya 重叠与 class-level 难度分 $D_c$，训练后与混淆矩阵对照；**发现 $D_c$ 对 class-wise 错误弱相关（$\rho=-0.07$），而 pairwise 重叠与 off-diagonal 混淆正相关（$\rho=0.39$），为"何时依赖频域、何时依赖时序"提供可检验结论。**

## 方法详解
- **数据划分**：先按 recording session 和环境 ID 分组，再在组内生成 1.51125 s（30,225 采样，hop=10,225，~66% 重叠）的窗口；训练/验证/测试严格按环境隔离（E01–E03 训、E04 验、E05 测），并给出 LOEO 五折结果。
- **可学习 Sinc 前端**：每个滤波器由下限 $f_1^{(i)}$ 和带宽 $b^{(i)}$ 两个可训量参数化，截断到核长 $L$ 后乘 Hamming 窗（式 4–6），消除频谱泄漏；前端总参数仅 $2N_f$（与 $L$ 无关）。池化前依次做幅度取绝对值、BatchNorm、ReLU、时间 max-pooling。
- **深度可分离卷积体**：$N_f$ 路带通输出重塑为 $\mathbb{R}^{N_f \times T \times 1}$ 的"类声谱图"，经 $B$ 个 DS-conv block，每块 DS-conv + BN + ReLU + 可选 2×2 池化；通道数 $C_b=\min(\alpha 2^b, 8\alpha)$；末层 GAP 后线性 softmax（式 11），交叉熵损失。
- **MOBO 搜索设置**：6 个设计变量（$N_f \in\{16..128\}$、$L \in\{101..401\}$、$\alpha \in\{2..12\}$、$B \in\{3..6\}$、peak LR、pooling stride）；Study A 优化 (macro-F1, size)；Study B 优化 (accuracy, latency on Raspberry Pi)；qEHVI 采集函数 + Matérn-5/2 核，Sobol 初始 6 点，共 24 次评估；表 8 显示不同 acquisition/kernel 组合最终 HV 差异 <0.09%，说明搜索结果稳健。
- **部署转换**：将训练时解析式 sinc 滤波器等价固化成固定卷积核，再做 FP32 TFLite 导出，原始 Keras → frozen → TFLite 三级预测一致率 100%，排除转换引入的误差。

## 实验与结果
- **数据集**：SnaanGhar7，7 类、5 环境（E01–E05）、21,387 窗口、采样 20 kHz/32-bit mono，最大类别不平衡比 1.2×；平台为 Raspberry Pi Zero 2 W + Waveshare WM8960 Audio HAT。
- **基线**：DS-CNN、CRNN、TinyCNN（MFCC）；SincNet、ACDNet、DPNet（raw waveform）。
- **环境不交叉主结果（表 10）**：
  - 最佳 SincDPNet：14,040 参数，**Acc 80.2%、Bal Acc 0.847、macro-F1 0.760、MCC 0.716**。
  - Rank-1 加权配置：3,408 参数，macro-F1 0.673、MCC 0.714。
  - Compact $N_f=25$：2,848 参数，**Acc 75.7%、macro-F1 0.661、MCC 0.716**。
  - 相对 DPNet（同量级轻量 backbone）：用 Sinc 前端即换来 macro-F1 +0.049、MCC +0.071、参数 -42%。
  - 最强基线 SincNet（132,327 参数）macro-F1 0.712；SincDPNet best-F1 在其 **28× 更小参数下高出 +0.048 macro-F1**。
- **LOEO**：论文未给全量数字，但讨论区强调跨环境误差模式与固定划分一致。
- **硬件推理（表 13，Raspberry Pi Zero 2 W）**：Compact 模型 435.6 ms/clip（RTF=0.29）、best-F1 920.4 ms（RTF=0.61）；全部 RTF<1，满足始终在线监控。
- **混淆分析（图 24、表 14）**：1,209 条错误中 774 条来自瞬态冲击重叠（Door↔Walker/Crutch）、406 条来自宽带水流歧义；Basin Tap 在 compact 配置下召回仅 8.2%（多被判为 Bathroom Tap/Shower），最佳配置提升至 85.2%。

## 相关工作脉络
- **SincNet（Ravanelli & Bengio, 2018）**：原始波形 + 可学习 sinc 带通前端，但后端为标准卷积+全连接，~10^5 参数；本文在其前端基础上替换为深度可分离体，实现 28× 压缩。
- **DPNet（Chowdhury et al., 2026）**：同团队的轻量浴室声分类基线，使用普通卷积前端；本文以相同参数量级证明"结构化的 sinc 前端 > 无约束卷积前端"。
- **DS-CNN / TinyCNN / Hello Edge（Zhang et al., 2017；Mittermaier et al., 2020）**：面向 MCU 关键词唤醒，强调 depthwise-separable 与量化；本文沿其轻量化路线但面向连续声学事件分类并强调可解释性。
- **DCASE 2021–2025 低复杂度任务**：规定 ≤128 kB / ≤128 k INT8 参数预算与设备信息划分；本文 compact 模型仅 ~2.8 k 参数，约为 DCASE 预算的 1/45，且报告 RTF 指标。
- **Öztürk et al.（2025）**：公共洗手间语料 460 min/11 细粒度类、RegNetY + mel-spectrogram 达到 97.8% 精度；本文指出其缺少门/助行器/非目标类，且使用随机划分易受房间特征污染，对比需在同一环境不交叉协议下进行。
- **Bittner et al.（2025）**：对角状态空间模型实现可解释原始音频分类；本文与之并列代表"可解释 + 轻量原始波形"两路线，差异在于 SincDPNet 保持经典 CNN 结构、可接入成熟部署工具链。

## 局限性与未来方向
- **Water 类间的固有频谱重叠无法仅靠前端解决**：Flush/Bathroom Tap/Shower 的 Bhattacharyya 重叠是数据本身的属性（图 12 线性判别仍交叠），需更长时序上下文或联合活动序列建模。
- **Door↔Walker/Crutch 的瞬态混淆暴露池化代价**：为使体部轻量而使用的 2×2 时间池化丢弃了关键瞬态细节（讨论节自承"这是我们未曾注意到的效率代价"）。
- **数据集规模偏小、环境数仅 5**：泛化到更多房型/麦克风/录制条件仍需验证；LOEO 结果未见完整数字。
- **仅评估 FP32 TFLite，未做 INT8 量化**：实际 MCU 部署的延迟与能耗尚未实测。
- **论文声明的未来方向**：① 用更长时序窗口或二级分类器缓解 Door/Walker 混淆；② 扩展至更多房间与设备以检验 sinc 频带可迁移性；③ 部署到 MCU 级别硬件评估内存/能耗。

## 研究启发与可借鉴点
- **"先声学、后建模"的可检验假设闭环**：训练前计算类间 Bhattacharyya 重叠 + 类难度分 $D_c$，训练后用混淆矩阵对照；本文发现 pairwise 重叠比单类难度分更能预测错误（$\rho=0.39$ vs $-0.07$），该流程可直接迁移到其它小样本音频分类任务。
- **Sinc 前端 + 深度可分离体的"结构化稀疏"范式**：在参数预算极低（<5k）时，给前端施加物理先验（带通形状）比扩大骨干收益更大——这为 TinyML 音频任务提供了"把容量花在刀刃上"的设计法则。
- **Quality-based coreset 加速 MOBO 选型**：按 SNR/削波/静音/谱平坦/类内离群复合打分保留 top-25%，验证集排序与全量训练高度一致，可在资源受限的架构搜索中复用。
- **环境/session 双重隔离划分 + RTF 指标**：报告 RTF<1 的部署证据对边缘 AI 审稿人很有说服力；环境不交叉协议也应作为同类工作默认基线。
- **可复现交付物清单完整**：公开 learned sinc 滤波器参数、TFLite 冻结图、代码与数据申请链接，便于下游做跨设备迁移或二次开发。

## 关键术语表
- **SincDPNet**：由可学习 sinc 带通前端与深度可分离卷积体组成的原始波形分类器，前端仅 $2N_f$ 参数且频带可直接以 Hz 解读。
- **SnaanGhar7**：面向辅助生活的 7 类卫生间声学事件数据集，5 环境、21,387 个 1.5 s 窗口，含显式非目标 Unknown 类。
- **环境不交叉划分（environment-disjoint split）**：训练/验证/测试按录音环境完全隔离，避免同房间声学特征泄漏导致虚高指标。
- **Bhattacharyya 重叠（BC）**：基于高斯近似的类间分布相似性度量，BC∈[0,1] 越大表示两类越难区分；本文用于量化 pairwise 声学歧义。
- **多目标贝叶斯优化（MOBO）**：以 qEHVI 为采集函数、GP 为代理模型，在 accuracy–size / accuracy–latency 二维空间搜索 Pareto 最优配置的工程工具。
- **深度可分离卷积（DS-conv）**：将标准卷积分解为逐通道 depthwise 与 1×1 pointwise 两步，参数/乘加量约为标准的 $1/C_{out} + 1/k^2$。
- **Real-Time Factor（RTF）**：推理时长与音频片段时长的比值，RTF<1 表示模型能在一个采集周期内完成分类，适合 always-on 部署。
- **Quality-based coreset**：按 SNR、削波率、静音率、谱平坦度与类内离群度综合打分，在每类内保留 top-25% clip 用于加速架构搜索。

## 可复现要素
- **数据集**：SnaanGhar7，可通过 Dataset Access Form 申请获取（论文称可公开访问）。
- **代码/权重**：源码、模型配置与实验脚本已在 GitHub 公开（https://github.com/debolina-34/SnaanGhar7），learned sinc 滤波器参数亦随实现一并发布。
- **关键超参**：
  - 采样率 20 kHz、窗口 30,225 样本（1.51125 s）、hop 10,225；
  - 前端：核长 $L \in \{101,251,401\}$，滤波器数 $N_f \in \{16..128\}$，Hamming 窗；
  - 骨干：DS-conv block 数 $B \in \{3..6\}$，宽度乘子 $\alpha \in \{2..12\}$，GAP + 线性 softmax；
  - 训练：class-balanced 交叉熵、振幅增广 ±25%（仅训练集）、三种子均值±标准差报告；
  - MOBO：qEHVI + Matérn-5/2、Sobol 初始 6 点、24 次评估。
