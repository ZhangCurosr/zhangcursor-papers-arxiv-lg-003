---
title: "TERRA-LEARNING-TRANSPORTABLE-LATENT-AC-TIONS-THROUGH-TEMPORA"
source: https://arxiv.org/pdf/2610.09509v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:12:30"
field: "机器人视觉-语言-动作预训练"
keywords: ["latent action", "robot learning", "visual representation", "cross-context transfer", "vision-language-action", "self-supervised learning"]
innovations: ["时序效应表示：用净分量和动态分量统一概括转换的紧凑表征，保留窗口内时序结构", "效应锚定迁移：通过固定效应描述符在跨上下文迁移时锚定运输效应方向", "相对效应一致性：用EMA队列和分离目标latent稳定不同隐式动作间的相对关系"]
benchmarks: ["LIBERO", "Open X-Embodiment"]
---

# 论文速读：TERRA-LEARNING-TRANSPORTABLE-LATENT-ACTIONS-THROUGH-TEMPORAL-EFFECT-REPRESENTATION-AND-RELATIONAL-ALIGNMENT

## 一句话总结
TERRA 提出了一种可迁移的连续隐式动作表示方法，通过同时捕捉视觉转换中的净特征变化（net component）和窗口内动态趋势（dynamics component），并在跨上下文迁移时锚定效应方向，使得隐式动作在不同初始状态下保持一致性，在 LIBERO 上以匹配预训练规模超过 UniVLA（93.4% vs 91.8%）。

## 研究问题与动机
- **选择性地保留时序信息**：连续隐式运动模型仅基于帧对特征差（endpoint difference）计算，无法区分相同首尾帧但运动节奏不同的执行（如快-慢 vs 慢-快）；而保留完整时序序列又会引入过多无关外观变化。
- **跨上下文效应一致性**：标准重建目标仅观察状态-转换匹配对，可能将上下文特定变异编码进隐式变量；当应用于不同初始状态时，相同隐式动作可能解码出不同效应。
- **统一表征空间**：现有工作将时序隐式构建与跨上下文一致性视为独立问题，本文主张用同一紧凑效应空间同时解决"保留什么"和"如何在新上下文中表现"两个问题。

## 核心贡献（创新点）
1. **时序效应表示（TER）**：在帧对特征差基础上增加低阶窗口内动态分量（DCT-II 基函数 $A_1$），使隐式动作保留"变化如何展开"的时序结构，而非仅记录端点差异。
   → 与 CoMo 等纯帧对方法的区别在于显式编码了窗口内的非稳态趋势，使 action profile 可线性恢复。

2. **效应锚定迁移（EAT）**：利用同一效应空间监督隐式动作在新初始状态下的解码行为，通过固定效应描述符 $\Psi$ 将运输后效应的方向锚定到源转换的观察效应。
   → 与现有离散码方法（如 LAPA、UniVLA）的区别在于直接约束"隐式动作做什么"而非"它是什么"，且无需额外训练迁移网络。

3. **相对效应一致性正则化**：维护 EMA 目标 tokenizer 并通过队列提供分离的 donor latent，约束不同 latent 效应间的相对关系在不同 recipient 状态下保持稳定。
   → 与前作（如 Olaf-World、cycle-based 方法）的区别在于在固定效应描述符空间而非原始特征空间约束相对关系。

4. **系统级验证**：在匹配的预训练规模下，完整 TERRA 系统在 LIBERO 上达到 93.4% 平均成功率，较 UniVLA 提升 1.6 个百分点，且在视觉干扰下衰减更慢（nuisance drop 17.9% vs 23.0%）。

## 方法详解
**Temporal Effect Representation (TER)**：
- 使用冻结的 DINOv2 特征（$C=768$），空间池化到 $4\times4$ 网格（$S=16$）。
- 对增量序列 $Y = (\Delta F_0, \ldots, \Delta F_{K-1})$，定义净分量 $N(Y) = \sum_k Y_k$（等价于 $F_K - F_0$）和动态分量 $A_1(Y) = \sum_k b_1(k) Y_k$，其中 $b_1$ 为最低频非恒定 DCT-II 模式。
- 结构化 token：$[F_0 \mid N(\Delta F)/\sigma_\Delta \mid A_1(\Delta F)/\sigma_\Delta]$，经 6 层 Transformer + 4 个 learned query 压缩为 $z \in \mathbb{R}^{4\times64}$。
- 自重建损失：$\mathcal{L}_{\text{self}} = \mathbb{E}[\|\hat{D} - \Delta F/\sigma_\Delta\|_F^2]$，其中 $\hat{D} = T_\phi(F_0, z)$。

**Effect-Anchored Transport (EAT)**：
- **效应锚定损失**：固定效应描述符 $\Psi(Y) = \text{std}(R^\top \xi(Y))$（$R$ 为固定高斯随机投影），$\mathcal{L}_{\text{effect}} = \mathbb{E}[1 - \cos(\Psi(Y_{j\to i}), \text{sg}[\Psi(\Delta F_i)])]$，对齐运输效应方向。
- **相对效应一致性**：$\mathcal{L}_{\text{rel}} = \mathbb{E}[1 - \cos(r_j^{i,i'}, r_{j'}^{i,i'})]$，其中 $r_s^{i,i'} = e_s^i - e_s^{i'}$，稳定不同 latent 间效应的相对关系。
- **过渡分布匹配**：MMD 损失 $\mathcal{L}_{\text{dist}}$ 匹配生成与真实特征增量的分布。
- **干扰不变性**：BYOL-style 损失 $\mathcal{L}_{\text{inv}}$，对在线 tokenizer 施加时空一致的图片增强（裁剪+亮度/对比度扰动）。
- 联合优化：$\mathcal{L}_{\text{EAT}} = \mathcal{L}_{\text{self}} + \lambda_e \mathcal{L}_{\text{effect}} + \lambda_r \mathcal{L}_{\text{rel}} + \lambda_d \mathcal{L}_{\text{dist}} + \lambda_i \mathcal{L}_{\text{inv}}$，权重均为 1（$\lambda_r=0.5$）。

## 实验与结果
- **数据集**：Open X-Embodiment 混合数据集（29 个源，约 819 万 clip）；下游评估使用 LIBERO（4 个子集：Spatial、Object、Goal、Long）。
- **基线**：LAPA 离散 VQ 基线（同 DINO 特征空间）、重训练的 UniVLA*（匹配预训练规模）。
- **主要结果**：
  - LIBERO 平均成功率：TERRA 93.4% vs UniVLA* 91.8%（+1.6%）；Object (+2.6%)、Long (+1.9%) 提升最大。
  - 线性读者 Action NMSE：TERRA 0.723 vs UniVLA* 0.785 vs LAPA 0.819。
  - 视觉干扰鲁棒性（$\alpha=1$）：TERRA drop 17.9% vs UniVLA* 23.0% vs LAPA 更高。
  - 动态分量贡献：加入 $A_1$ 后 Profile NMSE 从 0.983 降至 0.909，nuisance drop 从 33.6% 降至 22.4%。
  - EAT 贡献：在 15k 步更新预算下，EAT 较纯重建进一步降低 Action NMSE 至 0.723，transport 距离敏感性降低 3 倍。
- **迁移稳定性**：随 donor-recipient 距离增大，full EAT 的 donor-action NMSE 上升仅 0.087，vs 重建-only 的 0.267。

## 相关工作脉络
- **LAPA / UniVLA**（Ye et al., 2025; Bu et al., 2025）：离散量化隐式动作，任务中心但缺乏跨上下文一致性的显式约束。
- **CoMo**（Yang et al., 2026）：连续隐式运动，仅用帧对特征差（net component），无法区分同端点不同节奏的执行。
- **RotVLA**（Li et al., 2026）：结构化连续动作空间，引入时序组合但未见跨上下文迁移的直接正则化。
- **Olaf-World**（Jiang et al., 2026）：用冻结视频编码器的时序特征差定向隐式动作，但未统一表征构建与迁移。
- **LAOM**（Nikulin et al., 2025）：揭示干扰物会退化重建学习的隐式动作，提出线性探测评估，本文在此基础上直接正则化。
- **cycle-based 方法**（Chen et al., 2025a）：通过生成转换正则化隐式身份，本文与之区别在于在固定效应描述符空间约束而非原始特征。

## 局限性与未来方向
- **同源限制**：运输目标与度量定义在同一数据源内（共享机械臂和相机几何），跨 embodiment 迁移需要感知对应关系的描述符。
- **时序分辨率**：效应空间仅保留两个低阶时序分量在粗粒度空间网格上，以鲁棒性换取精细时序细节。
- **度量性质**：运输度量衡量的是 donor 动作语义是否在另一状态下存活，而非解码转换是否匹配唯一反事实未来。
- **规模限制**：闭环结果在缩减预训练规模（10% OXE）下报告，更大规模预训练和真实机器人部署为未来工作。
- **部分源退化**：roboturk 源上 EAT 在远距离 quartile 上 NMSE 从 0.822 升至 1.224，显示在某些域上仍需改进。

## 研究启发与可借鉴点
1. **低阶时序分量的显式建模**：使用 DCT-II 基函数提取窗口内动态趋势是一种简洁有效的结构化先验，可迁移至其他视频/序列表征学习任务。
2. **固定描述符的统一监督**：$\Psi$ 无学习参数且与表征构建使用相同公式，使"表征内容"与"迁移行为"在同一空间对齐，避免了对齐鸿沟。
3. **EMA 队列 + 相对关系约束**：通过 detached target latent 和 FIFO 队列实现相对效应一致性，以低成本正则化共享解码器，适用于任何需要跨上下文一致性的表征学习。
4. **干扰不变性的时空一致性增强**：clip 内共享单次增强（而非逐帧噪声）更贴近真实视觉干扰模式，设计值得借鉴。
5. **匹配规模的系统对比**：在相同 VLM backbone、相同预训练数据子集、相同 fine-tuning 协议下对比完整系统，结论更具说服力。

## 关键术语表
**Latent Action（隐式动作）**：从视觉转换中推断出的类动作表示，作为视觉输入与机器人策略之间的中间监督信号。

**Temporal Effect Representation（时序效应表示）**：用净分量（累积特征变化）和动态分量（窗口内低阶趋势）两个低阶组件概括转换的紧凑表征。

**Effect-Anchored Transport（效应锚定迁移）**：将 donor latent 解码到 recipient 初始状态后，通过固定效应描述符将其运输效应方向锚定到 donor 观察效应的正则化方法。

**Net Component（净分量）**：增量序列的求和 $N(Y) = \sum_k Y_k$，等价于端点特征差 $F_K - F_0$。

**Dynamics Component（动态分量）**：用最低频非恒定 DCT-II 模式加权求和 $A_1(Y) = \sum_k b_1(k) Y_k$，捕捉窗口内的粗粒度时序趋势。

**Effect Descriptor $\Psi$（效应描述符）**：无参数的固定变换 $\Psi(Y) = \text{std}(R^\top \xi(Y))$，将高维效应向量投影并标准化到低维空间用于方向比对。

**Relative-Effect Consistency（相对效应一致性）**：约束不同 latent 的运输效应在不同 recipient 状态下的差异方向保持一致的正则化项。

**Nuisance Invariance（干扰不变性）**：通过 BYOL-style 损失使 latent 对时空一致的视觉扰动（裁剪、亮度/对比度）保持不变。

## 可复现要素
- **数据集**：Open X-Embodiment（公开），LIBERO（公开）；论文使用 OXE 的 10% 子集进行 latent-VLM 预训练。
- **代码**：论文未明确声明开源，但提供了完整的训练细节与超参数（Appendix A.2）。
- **权重**：使用 OpenVLA-7B 作为 backbone（公开），DINOv2 特征缓存（公开）。
- **关键超参**：$\lambda_e=1, \lambda_r=0.5, \lambda_d=1, \lambda_i=1$；$\beta_{\text{EMA}}=0.99$；队列大小 16 个 minibatch（~1024 latents）；$K=4$ 步增量；$S=16$ 空间网格（$4\times4$）；$C=768$ 特征维度；latent 维度 $4\times64$。
