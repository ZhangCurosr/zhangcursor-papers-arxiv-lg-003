---
title: "TERRA-LEARNING-TRANSPORTABLE-LATENT-AC-TIONS-THROUGH-TEMPORA"
source: https://arxiv.org/pdf/2610.09509v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:10:51"
---

# 论文速读：TERRA: LEARNING TRANSPORTABLE LATENT ACTIONS THROUGH TEMPORAL EFFECT REPRESENTATION AND RELATIONAL ALIGNMENT

## 一句话总结
本文提出 TERRA，通过引入窗口内低阶动力学分量构建紧凑的时序效应表示，并利用效应锚定迁移（EAT）机制在同源不同初始状态下将解码效应的方向锚定至源端观测，从而学到既保留动作时序结构又具备跨上下文一致性的可迁移连续潜动作表示。

## 研究问题与动机
1. **时序信息选择性保留的张力**：现有连续潜运动模型仅以帧对端点差为输入，丢失窗口内运动展开的细节（如图 1a，终点相同但快慢节奏不同的动作会被映射为同一潜码）；若保留完整时序序列，又极易混入与动作无关的外观变化和终态异质性。
2. **跨上下文效应一致性缺失**：标准自重建目标仅在匹配的“状态–转换”对上优化，潜变量会编码上下文特定噪声；当该潜码被应用在其他初始状态时，其解码效应会发生漂移，动作语义无法稳定迁移（图 1b）。
3. **端到端 VLA 预训练中的表征对齐瓶颈**：离散潜动作（LAPA/UniVLA）依赖量化码本，连续潜动作（CoMo/RotVLA）缺乏显式的跨状态一致性约束，导致下游策略对视觉干扰敏感且在长视野/多变场景下表现受限。

## 核心贡献（创新点）
1. **时序效应表示（TER）**：将过渡概括为净分量（累积特征变化）与动力学分量（DCT-II 最低频非恒定模式）两部分，以极低维度的时序证据替代完整序列。*与 CoMo 等仅依赖端点差的连续潜运动模型本质不同，TER 显式注入窗口内时序分布先验，使动作轮廓线性可恢复。*
2. **效应锚定迁移（EAT）**：在同一个效应空间内同时监督“潜码保留什么”与“潜码在新状态下能做什么”，通过固定描述符将迁移后的效应方向锚定至源端观测。*与 LAPA/UniVLA 仅依赖状态-转换重建的目标不同，EAT 直接约束潜码的跨上下文行为而非静态编码内容。*
3. **系统级可控验证与匹配规模对比**：拆解 TER 分量与 EAT 四项正则，证明各自对动作对齐、抗干扰鲁棒性与迁移稳定性的独立贡献；在相同预训练数据规模下，完整系统以 93.4% 的 LIBERO 平均成功率超越 UniVLA（91.8%）。

## 方法详解
TERRA 由两部分组成：**时序效应表示（TER）** 与 **基于效应锚定的关系对齐（RA/EAT）**，全程使用冻结的 DINOv2 视觉特征，无需机器人动作标签。

- **时序证据构建**：对片段 $x_i=(o_{i,0},...,o_{i,K})$ 提取冻结 DINOv2 特征并空间池化为 $F_{i,k} \in \mathbb{R}^{S \times C}$，计算增量 $\Delta F_{i,k}=F_{i,k+1}-F_{i,k}$。定义两类低阶分量：
  - 净分量 $N(Y)=\sum_{k=0}^{K-1} Y_k$（等价于端点差 $F_K-F_0$，捕捉累积变化）；
  - 动力学分量 $A_1(Y)=\sum_{k=0}^{K-1} b_1(k) Y_k$（$b_1$ 为 DCT-II 最低频非恒定基，捕捉窗口内粗粒度时序趋势）。
  将 $F_0$、$N(\Delta F)/\sigma_\Delta$、$A_1(\Delta F)/\sigma_\Delta$ 拼接为结构化 token，经 6 层 Transformer 与 4 个学习查询聚合，输出连续潜码 $z=E_\theta(x) \in \mathbb{R}^{4\times 64}$。

- **自重建损失**：条件解码器 $T_\phi$ 预测归一化增量序列 $\hat{D}=T_\phi(F_0,z) \in \mathbb{R}^{K \times S \times C}$，训练目标为 $\mathcal{L}_{\text{self}} = \mathbb{E}[\frac{1}{KSC}\|\hat{D} - \Delta F/\sigma_\Delta\|_F^2]$。

- **效应锚定迁移（EAT）四项约束**：
  1. **效应锚定** $\mathcal{L}_{\text{effect}}$：使用无参数描述符 $\Psi(Y)=\text{std}(R^\top \xi(Y))$（$R$ 为固定高斯随机投影，$\xi=[N,A_1]$），最小化迁移效应与源端观测效应的余弦距离 $\mathbb{E}[1-\cos(\Psi(Y_{j\to i}), \text{sg}[\Psi(\Delta F_i)])]$，仅约束方向，Magnitude 允许随上下文自由变化。
  2. **相对效应一致性** $\mathcal{L}_{\text{rel}}$：维护 EMA 目标潜码 $\bar{z}$ 的队列，优化 $\mathbb{E}[1-\cos(r_j^{i,i'}, r_{j'}^{i,i'})]$，稳定不同潜码效应在跨接收状态下的相对关系，直接正则化共享解码器。
  3. **过渡分布匹配** $\mathcal{L}_{\text{dist}}$：采用无偏 MMD 匹配真实与生成增量投影后的经验分布，作用于过渡空间而非潜空间。
  4. **无关变量不变性** $\mathcal{L}_{\text{inv}}$：BYOL 风格，在线潜码接收时空裁剪+亮度/对比度扰乱的视图，EMA 潜码接收干净视图，最小化余弦差异。扰动仅用于此项，效应约束始终在原始未增强特征上计算。

- **联合优化**：$\mathcal{L}_{\text{EAT}} = \mathcal{L}_{\text{self}} + \lambda_e \mathcal{L}_{\text{effect}} + \lambda_r \mathcal{L}_{\text{rel}} + \lambda_d \mathcal{L}_{\text{dist}} + \lambda_i \mathcal{L}_{\text{inv}}$。预训练 65k 步自重建后，进入 15k 步 EAT 精修；精修后冻结 $E_\theta$ 用于下游策略。

## 实验与结果
- **数据集与设置**：预训练基于 Open X-Embodiment 混合数据集（29 个源
