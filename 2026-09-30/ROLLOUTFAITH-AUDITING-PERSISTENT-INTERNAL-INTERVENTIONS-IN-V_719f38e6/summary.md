---
title: "ROLLOUTFAITH-AUDITING-PERSISTENT-INTERNAL-INTERVENTIONS-IN-V"
source: https://arxiv.org/pdf/2609.36843v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:13:03"
field: "世界模型可解释性与干预评估"
keywords: ["world models", "interpretability", "intervention auditing", "activation editing", "persistent effects", "RolloutFaith", "LoReFT"]
innovations: ["提出 RolloutFaith 框架与 ISG/SSG 双指标，分离即时修复与持续效益", "系统诊断拟合编辑器在量级、方向、跨源泛化上的三重失败模式", "提出 Delayed LoReFT 通过未来过渡时间监督提升持续干预效果"]
benchmarks: ["Crafter", "DMC Cartpole", "Procgen CoinRun"]
---

# 论文速读：ROLLOUTFAITH — AUDITING PERSISTENT INTERNAL INTERVENTIONS IN VISUAL WORLD MODELS

## 一句话总结
本文提出 **RolloutFaith** 框架，通过配对 rollout 机制将世界模型内部干预的"即时语义修复"与"持续语义收益"分离评估，揭示现有拟合编辑器在纠正量级、方向和跨源泛化上的三重局限，并提出 **Delayed LoReFT**（在冻结的未来过渡上优化低秩干预）作为改进方向。

---

## 研究问题与动机

1. **世界模型（World Models, WMs）中内部干预的持久性问题**：WMs 利用自身预测作为后续输入，一次内部纠正必须能在编辑停止后仍然有效，但现有解释性方法（探针、激活修补等）仅测量当前输出的修复，无法回答"干预效果是否会随时间衰减"。
2. **缺乏统一的时间尺度评估标准**：既有 WM 解释性评测（如 WorldModelLens、WRBench、WorldSimProbe、Intervention Gap）要么只测当前步骤的输出一致性，要么测试外部观察变化下的行为，没有一种方法对单次内部干预后的自运行未来进行度量。
3. **拟合编辑器性能不稳定且难以泛化**：现有编辑器（LoReFT、PCA+ridge、Manifold 等）在校验集上的正向收益在向未见测试数据泛化时常常失效，甚至变为有害——原因尚未被系统分解。
4. **缺乏对"效果通过何种状态组件传播"的结构化理解**：不同架构（扩散、循环、Transformer）在保留干预效果时的路径差异未被刻画。

---

## 核心贡献（创新点）

1. **RolloutFaith 框架与 ISG/SSG 双指标**：提出即时语义增益（ISG）和持续语义增益（SSG@H）两个时间尺度指标，通过固定事件、动作、预测噪声的配对 rollout 实现控制比较。*本质区别在于：此前工作只评估干预时刻或外部变化下的输出，本文首次评估单次内部纠正后自生成的持续效益。*

2. **揭示拟合编辑器的三重失败模式（量级、方向、泛化）**：通过 RAP-matched-by-norm 和反向对照实验，将干预失败的根源系统分离为纠正量级不足、纠正方向偏差、跨源泛化失败三个独立因素。*与已有工作相比，本文不只报告"编辑器差"，而是提供可分解的诊断工具。*

3. **架构级组件恢复分析揭示持久化传播路径**：提出组件恢复（component restoration）协议，逐分量替换后追踪效果消失程度，发现 DIAMOND 通过最新生成帧、DreamerV3 通过循环状态、STORM 通过两者联合传播效果。*这是首个针对三种 WM 架构的干预持久化路径的实证对比。*

4. **Delayed LoReFT：通过未来过渡进行时间监督的编辑器训练改进**：将 LoReFT 的优化目标从当前编辑帧扩展至四个冻结未来过渡帧的图像 MSE 损失之和，在多个任务组合上提升了持续干预效果。*本质区别：现有编辑器只优化当前输出，本文使训练奖励未来的后果。*

---

## 方法详解

### 3.1 模型与数据集
- **三个世界模型**：DIAMOND（4 帧扩散上下文）、DreamerV3（递归随机动力学 RSSM）、STORM（带 Transformer 缓存的随机隐状态）。
- **三个任务**：Crafter（瓦片语义，7×9 网格 19 类 one-hot）、DMC Cartpole（关节几何，body-mask IoU）、Procgen CoinRun（地形，8×8 网格两类地形比例）。
- 测试集：133 个独立来源、229 个可评分事件（38 Crafter episodes、48 Cartpole episodes、47 CoinRun levels）。

### 3.2 干预方法
一般干预公式：
$$a'_m = a + \text{cap}_{\rho}^{(m)}\left(\alpha \, d_m(a, r, e)\right)$$
其中 $\rho$ 为训练纠正长度的 90th 百分位数，$\alpha$ 由验证集选择。

- **Reference Activation Patching (RAP)**：理想对照，将激活 $a$ 直接替换为事件匹配的参考激活 $a^\star$，测量接口能力上限（不参与排名）。
- **分析方向类**：Mean Delta、Difference of Means、Inverse Probe（$P^\dagger e$）、Normalized Probe（单位读取出方向）。
- **监督参数类**：PCA+ridge、Uncentered Carrier（$\mu=0$ 的线性载体）、Nonlinear MLP（两隐层 tanh）、Immediate LoReFT（式 3：$d = R^\top(Wa + Ur + b - Ra)$，正交子空间 + 请求条件仿射映射，以编辑帧图像 MSE 为损失）、Concept DAS（双向分布匹配，JSD 损失，式 8）。
- **非参数几何类**：Manifold（基于训练锚点的最近邻图最短路径）。

### 3.3 评估指标
$$\text{ISG}_C(m) = \mathbb{E}_i\!\left[q_i^m(0) - q_i^C(0)\right]$$
$$\text{SSG}_C@H(m) = \mathbb{E}_i\!\left[\frac{\sum_{h=1}^{H} v_i(h)(q_i^m(h) - q_i^C(h))}{\sum_{h=1}^{H} v_i(h)}\right], \quad H=8$$
要求至少 6 个可评分未来帧。分数先对 seed 和事件取均值再等权平均 sources，避免多评估来源过度加权。

### 3.4 评估协议
噪声耦合（noise coupling）：编辑和对比分支共享事件、动作、预测噪声和可评分掩码，仅干预不同。每个事件生成 4 个未编辑预测后，在第 5 步施加一次干预，之后自运行 $h=1,\dots,8$。共 2,061 个 model–event–seed 组合。

### 3.5 组件恢复协议
在步骤 1 将某一候选下游组件（latent / memory）替换回原始未编辑值，追踪后续 24 步的增益变化。恢复损失 $L_c = G(I) - G(R_c \circ I)$，正值表示该组件承载了有效纠正。

### 3.6 Delayed LoReFT
$$\mathcal{L}_I(\theta) = \ell(\hat{o}_t^{\mathcal{T}_\theta}, o_t^\star), \qquad \mathcal{L}_D(\theta) = \frac{1}{K}\sum_{k=1}^{K} \ell(\hat{o}_{t+k}^{\mathcal{T}_\theta}, o_{t+k}^\star), \quad K=4$$
冻结世界模型，仅训练 LoReFT 参数，对四个未来过渡帧的图像 MSE 求平均。

---

## 实验与结果

### 主要数值结果（Table 2）

| 模型 | 任务 | RAP ISG | RAP SSG@8 | 最佳拟合编辑器 SSG@8 |
|---|---|---|---|---|
| DIAMOND | Crafter | **+15.23** | **+10.76** | Mean: +0.31 |
| DIAMOND | Cartpole | **+5.97** | **+9.01** | Delayed LoReFT: +3.24 |
| DIAMOND | CoinRun | **+7.62** | **+4.14** | NormProbe: +2.25 |
| DreamerV3 | Crafter | **+2.71** | **+1.24** | NormProbe: +0.32 |
| DreamerV3 | Cartpole | **+0.52** | **+0.92** | **Immediate LoReFT: +3.22**（唯一超过 RAP）|
| DreamerV3 | CoinRun | **+1.06** | **+1.20** | Carrier: +0.38 |
| STORM | Crafter | **+8.32** | **+3.62** | InvProbe: +2.06 |
| STORM | Cartpole | **+12.05** | **+8.90** | Delayed LoReFT: +0.75 |
| STORM | CoinRun | **+6.17** | **+3.35** | NormProbe: +0.44 |

### 关键发现
- **RAP 在全部 9 个组合中 SSG@8 均为正**，说明所有接口均有纠正潜力；8/9 组合中 RAP 优于最佳拟合编辑器。
- **唯一例外**：DreamerV3 Cartpole，Immediate LoReFT（+3.22 pp）超过 RAP（+0.92 pp），配对 95% CI [−2.68, −1.90]。
- **验证-测试泛化缺口**：90 个拟合组合中 68 个测试表现低于验证；25 个从正验证增益转为负测试增益。
- **方向/量级分解**（Figure 3）：DIAMOND Crafter 失败主因量级不足；DreamerV3 Crafter 失败主因方向偏差。
- **组件恢复**：DIAMOND 持久化通过**最新生成帧**（Crafter 恢复 latent 损失 +5.42 pp）；DreamerV3 通过**循环状态**（Cartpole 恢复 memory 损失 +4.68 pp）；STORM **两者兼有**（Cartpole 分别损失 +3.10 / +2.80 pp）。
- **Delayed LoReFT**（Table 3）：在 9 组合中，最终增益（steps 17–24）优于 Immediate LoReFT 的有 6/9 个；DIAMOND Cartpole 的 SSG@8 从 +0.32 提升至 **+3.24**（+10 倍），Final gain 从 +0.01 升至 **+1.20**。

---

## 相关工作脉络

1. **WM 解释性评测**：Zhang (2026) Atari probing、Challagundla et al. (2026) WorldModelLens、Lu et al. (2026a) WRBench、Co et al. (2026) WorldSimProbe、Vakalis (2026) Intervention Gap——均评测当前步骤或外部变化下的行为一致性，**不评估单次内部干预后的自运行持续性**。
2. **通用干预方法**：Activation Addition / RepE（Turner et al. 2023; Zou et al. 2023）、DAS / LoReFT（Geiger et al. 2024; Wu et al. 2024）、Concept DAS（Bao et al. 2026）、Manifold Steering（Wurgaft et al. 2026）——均优化当前输出，**无时间维度损失**。
3. **WM 专用干预**：Feedback steering（Hong et al. 2026）依赖重复反馈；CGSReg（Lu et al. 2026b）修复任务关键区域；Low-rank carriers（Liu & Chen 2026）无参考预测——本文将它们统一置于 RolloutFaith 下对比，首次揭示其持续性缺口。
4. **扩散模型解释**：Revelio（Kim et al. 2025）、Prompt Conditioned Intervention（Görgün et al. 2026）——关注单次生成内部的去噪时间步，**非环境时间的未来预测**。
5. **世界模型训练增强**：Diffusion Forcing（Chen et al. 2024）、Self Forcing（Huang et al. 2026）——扩展生成超过训练 horizon 以减少自回归暴露偏差；本文将类似思想迁移至**内部编辑器训练**。

---

## 局限性与未来方向

1. **编辑器训练数据预算有限**：所有拟合编辑器仅使用开发集（每个任务 15–36 个 source），结果"不能确立充分数据缩放编辑器的能力上限"。
2. **仅评估三个 WM 架构**：DIAMOND（扩散）、DreamerV3（循环）、STORM（Transformer）——其他架构（如 DreamerGen、Gaia-1）未被覆盖。
3. **组件恢复存在混合状态解释局限**（Appendix H.3）：恢复单一组件后产生 hybrid state，不能严格视为因果分解，只能给出局部操作性的路径描述。
4. **SSG 在长 horizon 上普遍衰减**：RAP 在 5/9 组合中 SSG 随 horizon 下降，说明自主动态并非总能保留纠正。
5. **未来方向**：延长时间监督 horizon（$K>4$）、开发无需测试参考的跨源泛化编辑器、直接编辑承载纠正的状态组件以使其效果持久。

---

## 研究启发与可借鉴点

1. **配对 rollout 噪声耦合协议**：编辑和对比分支共享同一 actions + noise seed，仅干预不同——这是消除环境变异噪声的干净设计，可直接迁移至任何需要因果对比的序列模型评测。
2. **ISG/SSG 双层指标体系**：将"当下修复"和"未来持续"分开度量，避免单一指标掩盖关键行为（如 Figure 1 中 RAP ISG=+24.25 但 8 步后降至 +14.73）——值得推广至任何干预后行为追踪任务。
3. **幅度/方向解耦对照**：RAP-matched-by-norm 与 fitted-at-reference-magnitude 两类对照实验，能分离量级不足与方向偏差——是诊断干预失败的可复用方法论。
4. **组件恢复的跨架构对比思路**：对不同 WM 架构统一抽象为"(latent, memory)"二元结构并进行恢复对照，给出可操作的"传播路径"描述——可复用到其他序列生成模型的干预分析。
5. **Delayed LoReFT 的时间监督范式**：将编辑器训练损失从单步扩展到未来 $K$ 步的冻结 rollout——可直接迁移到 LoReFT/REFT 家族的其他任务（如语言模型的未来 token 监督）。

---

## 关键术语表

**RolloutFaith**：论文提出的统一配对 rollout 评估框架，同时衡量内部干预的即时修复（ISG）与持续效益（SSG）。

**ISG（Immediate Semantic Gain）**：干预步骤 $h=0$ 处编辑预测相对于未编辑预测的语义质量提升（期望差值）。

**SSG@H（Sustained Semantic Gain）**：干预停止后 $h=1$ 至 $H$ 步自主预测中编辑预测的平均持续质量增益。

**RAP（Reference Activation Patching）**：将模型某处激活直接替换为真实观察对应激活的理想化对照干预，用于测量接口纠正容量上限。

**Delayed LoReFT**：将 LoReFT 的优化目标从当前编辑帧扩展至 $K=4$ 个未来冻结过渡帧的图像 MSE 损失平均，引入时间监督。

**Component Restoration**：在干预后的步骤 1 将某一状态分量（latent/memory）替换回原始未编辑值，通过增益消失量识别该分量是否承载持久化效果。

**Noise Coupling（噪声耦合）**：编辑与对比分支使用完全相同的预测噪声 seed，确保二者仅在干预上存在差异。

**Source-level Aggregation**：先在每个 source 内对 seed 和 event 取均值，再等权平均 sources，防止多评估 source 过度影响全局均值。

---

## 可复现要素

- **数据集**：Crafter、DMC Cartpole、Procgen CoinRun（标准基准）；测试事件来自冻结 checkpoint 独立采样。论文未声明公开链接，但 appendix A.1 详述了事件筛选流程与发育/测试 split。
- **代码/权重**：论文未提供官方 GitHub 链接；附录注明"所有拟合编辑器使用纯开发集拟合与选择"，checkpoint 来自各模型官方发布版本（DIAMOND/ DreamerV3/ STORM 原始论文）。
- **关键超参**：
  - SSG horizon $H=8$，要求至少 6 个可评分未来帧
  - Delayed LoReFT 的 $K=4$
  - 量级 cap：训练纠正长度的 90th 百分位数
  - 剂量网格：{0.01, 0.03, 0.1, 0.25, 0.5, 1}
  - LoReFT 秩搜索：{4, 8}，学习率 {1e-4, 3e-5}，每组合 256 次更新
  - PCA/ridge rank=8，惩罚 $10^{-3}$
  - 非线性 MLP：2 隐层，宽度 64，tanh，LR=1e-3，weight decay=1e-3，梯度裁剪=10，512 次更新
  - Concept DAS：Adam LR=1e-3，梯度裁剪=1，256 次更新，rank=8
  - Manifold：初始 $k=6$，Dijkstra 最短路径
  - 预测 seed：{69000, 69001, 69002}
  - 统计置信区间：10,000 次 source bootstrap

---
