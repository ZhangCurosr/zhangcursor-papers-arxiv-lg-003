---
title: "Probe-Space-Preconditioning-for-Fast-and-Stable-Zero-Order-T"
source: https://arxiv.org/pdf/2609.38095v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:25:20"
field: "零阶优化与大模型高效微调"
keywords: ["zero-order optimization", "SPSA", "large language model fine-tuning", "inference-mode training", "curvature preconditioning", "derivative-free optimization"]
innovations: ["提出1.5-SPSA，在推理模式下通过单次clean forward pass估计探针空间曲率预条件子，显著加速收敛", "证明预算应从多step转向大批量+多扰动以在少步内达到更高精度", "结合8-bit打包扰动与Triton融合kernel实现OPT-30B级别模型的单卡推理模式训练"]
benchmarks: ["GLUE/SuperGLUE (SST-2, RTE, BoolQ, WSC, WiC)", "Stable ToolBench"]
---

# 论文速读：Probe-Space-Preconditioning-for-Fast-and-Stable-Zero-Order-T

## 一句话总结
本文提出 **1.5-SPSA**，通过在推理模式（inference-mode）零阶优化中引入单次 clean forward pass 估计探针空间内的廉价对角预条件子，实现对高曲率方向的下加权；该方法在 OPT-13B/30B 和 Qwen3 系列模型上实现了超过 MeZO 与 BP+Adam 的精度，且优化步数从数万步降至几十步。

## 研究问题与动机
- **BP 内存税过大**：训练 OPT-30B 用 Adam 需约 600GB GPU 显存，迫使大规模分布式集群，单卡训练不可行。
- **ZOO 收敛太慢**：现有零阶方法（如 MeZO）虽仅需推理模式（~60GB），但通常需要 100,000 步才能收敛，计算效率远低于 BP。
- **噪声与曲率波动**：SPSA 同时引入 batch noise 与 perturbation noise，且神经网络 loss 曲率在随机方向上跨度可达 8 个数量级（$-10^8$ 到 $10^8$），固定步长会导致高曲率方向不稳定。
- **缺乏低开销的预条件机制**：Adam 类对角预条件需要 $O(d)$ 额外内存存储二阶矩，与推理模式内存约束冲突。

## 核心贡献（创新点）
- **发现预算重分配策略**：在相同 forward-pass 预算下，将优化步骤分配给更大有效 batch size 和更多 perturbations per step（而非更多 steps），1SPSA 可在数十步内超越 MeZO 精度。
- **提出 1.5-SPSA 曲率预条件**：在 1SPSA 基础上每步仅增加一次 clean forward pass，利用三点差分估计每个探针方向的标量曲率 $\hat{c}_i$，并通过 $\alpha$-饱和权重 $w_i = |\hat{c}_i|^{\alpha}$ 下加权高曲率方向，稳定训练并允许更大学习率。
- **提供严格的 JL 引理证明**：从 Johnson-Lindenstrauss 嵌入保内积性质出发，证明随机扰动子空间中的三点曲率估计能高概率保持真实方向曲率，为 probe-space 预条件化提供理论保障。
- **系统级工程优化**：结合 8-bit 打包 Rademacher 扰动生成、Triton fused unpack/apply kernel 与分布式 seed-based 并行，使 OPT-30B 可在单节点 8×A100 上推理模式训练，相较 PyTorch 原生实现获得 2.76× step 加速。

## 方法详解
- **预算分析**：定义总 forward-pass 预算 $F_{\text{1SPSA}} = s \times a \times 2 \times n_{\text{pert}}$，在 OPT-13B SST-2 上固定 $F=500{,}000$  Sweep $(a, n_{\text{pert}})$ 发现最优配置为 effective batch size=128、$n_{\text{pert}}=160$，仅需 80 步达到 94.2%（MeZO 用相似预算仅 91.4%）。
- **λ=ε 绑定**：学习率 λ 与扰动半径 ε 绑定时数值最稳定，两项在更新规则中相消，避免 λ>ε（步长超出测量范围）或 λ<ε（收敛缓慢）。
- **三点曲率估计**：$\hat{c}_i = \frac{L(\theta+\epsilon z_i) - 2L(\theta) + L(\theta-\epsilon z_i)}{\epsilon^2} \approx z_i^\top \nabla^2 L(\theta) z_i$，其中 $L(\theta)$ 为每步共享的 clean forward pass。
- **α-饱和权重**：$w_i = \frac{1}{\max(\lambda_{\text{reg}}, |\hat{c}_i|^\alpha)}$，其中 $\lambda_{\text{reg}}=1$ 防除零，$\alpha=0.1$ 在所有架构/任务上表现最稳定；$\alpha=0$ 退化为标准 1SPSA，$\alpha=1$ 为扰动空间内完整对角 Newton 步。
- **更新公式**：$\Delta\theta = \frac{1}{2n_{\text{pert}}}\sum_{i=1}^{n_{\text{pert}}} \frac{L(\theta+\epsilon z_i)-L(\theta-\epsilon z_i)}{\max(\lambda_{\text{reg}},|\hat{c}_i|^\alpha)} z_i$，等价于矩阵形式 $\Delta\theta = -\eta Z W \Delta\ell$。
- **分布式 seed 方案**：Rank 0 广播 θ→生成种子分散到各 rank→各 rank 就地生成 packed 扰动→执行三次 forward（±ϵ 及中心）→仅传回标量 loss→Rank 0 重放扰动并更新 θ→广播新 θ，通信开销极低。

## 实验与结果
- **数据集**：OPT-13B/30B 在 GLUE/SuperGLUE（SST-2、RTE、BoolQ、WSC、WiC）上进行 post-training；Qwen3-8B 在同一 GLUE 集；Qwen3-1.7B 在 Stable ToolBench 上做 tool-use 微调。
- **主要结果（OPT-13B）**：1.5-SPSA 在 SST-2 上达到 **94.5%**，较 MeZO（91.4%）提升 **+3.1%**，较 BP+Adam（92.0%）提升 **+2.5%**，仅需 70 步 vs MeZO 的 100,000 步；RTE 达 77.7%（MeZO 66.1%）。
- **OPT-30B**：1.5-SPSA SST-2 达 94.5%（MeZO/Prefix 90.6%），证明方法可平滑扩展至更大模型。
- **Qwen3-8B**：SST-2 达 94.7%，较 1SPSA 提升 +0.2%。
- **Qwen3-1.7B Stable ToolBench**：1.5-SPSA 达 79.0%（lr=5e-4），较 BP+Adam（77.0%）提升 +2.0%，且允许使用更大 lr（1SPSA 在 lr=1e-3/5e-4 时发散）。
- **Stiff Paraboloid**：条件数 κ 增大时 1.5-SPSA 收敛速度显著快于 1SPSA，最高可达 **6× 加速**，验证改善幅度与 Hessian condition number 成正比。
- **DNC overfitting stress test**：模型规模至 1B 参数，1.5-SPSA 在多数情况下收敛步数比 1SPSA 少最多 **6×**，部分配置下快于 BPTT。
- **系统加速**：Triton bit-packed kernel 相较原生 PyTorch 实现实现 4.9× probe gen 加速、2.76× step 加速。

## 相关工作脉络
- **MeZO (Malladi et al., 2023)**：首个将 1SPSA 适配 LLM post-training 的 in-place ZOO 方法，每步 2 次 forward，需 100,000 步；本文 1.5-SPSA 在更少步数下超越其精度。
- **LeZO (Wang et al., 2024)**：层内结构化扰动降低不同层参数尺度混叠问题；本文采用全局 Rademacher 扰动 + 曲率重加权达到类似稳定效果，且无需层内设计。
- **ZO-AdaMM (Chen et al., 2019)**：维护动量/自适应学习率，但引入 $O(d)$ 优化器状态内存；本文明确拒绝此类存储以维持推理模式。
- **DeepZero (Chen et al., 2024)**：稀疏化 ZO 更新以扩展至大模型；本文保持全参数更新，通过曲率预条件而非稀疏化来提升效率。
- **Evolution Strategies (Salimans et al., 2017; Wierstra et al., 2008)**：种群式梯度估计适合并行系统；本文 1SPSA 框架同样支持并行 perturbations，但更简洁且只需 2 次 forward/perturbation。
- **2SPSA (Spall, 1997)**：尝试估计 Hessian 预条件，但 $O(d^2)$ 存储不可行；本文退而求其次仅在扰动子空间内估计标量曲率，避免高维 Hessian。

## 局限性与未来方向
- **BP 基线未充分调优**：作者承认未在极端少步/大批量配置下对 BP+Adam 进行 exhaustive tuning，故"全面超越 BP"的声明需谨慎理解。
- **α 和 λ 依赖超参搜索**：虽然 α=0.1 跨任务稳健，但仍需手动 sweep；学习率 λ 的最佳值随 batch size 和 n_pert 变化，缺乏自动调节策略。
- **曲率估计无跨 batch/扰动归一化**：当前方案在每步内独立估计 $w_i$，未做跨 batch 或跨扰动方向的相对归一化，可能存在分布偏移。
- **不适用于需要 optimizer state 的分布式策略**：如 ZeRO-3 等需要额外通信和状态同步的场景，本文方案的优势未在此类配置下验证。

## 研究启发与可借鉴点
- **预算重分配思路可迁移**：在资源受限的优化场景（如边缘设备在线学习、机器人控制）中，将 forward-pass 预算向"大步长+多扰动"倾斜而非"多步+少扰动"，可能是更高效的策略。
- **probe-space 预条件化范式**：在高维优化中直接估计全 Hessian 不现实，但在随机投影子空间内估计低维曲率并做权重调节是低成本可行的思路，可推广至其他 black-box 优化场景（如强化学习策略梯度）。
- **λ=ε 绑定技巧**：将学习率与扰动半径绑定可同时满足数值稳定性和步长-测量范围对齐，这一原则在有限差分类估计器中具有普适性。
- **8-bit packed Rademacher + fused kernel**：扰动向量以 1 bit/参数存储、Triton 内联 unpack/apply 的工程方案，对任何需要高频随机扰动的大规模优化（如 NEP、ES）均有复用价值。
- **JL 引理支撑的理论保障**：用随机投影保曲率的方法为 black-box 优化提供了可证明的预条件有效性，为后续工作建立理论基准。

## 关键术语表
- **Zero-Order Optimization (ZOO)**：无需梯度信息、仅通过函数值估计进行参数更新的优化范式，训练全程处于推理模式，无需存储激活值和梯度。
- **1SPSA (Simultaneous Perturbation Stochastic Approximation)**：Spall 提出的零阶优化算法，用单个随机扰动向量 $z$ 通过两次前向传播估计梯度近似，维度无关。
- **MeZO**：将 1SPSA 适配到大语言模型 post-training 的 in-place 零阶优化方法，每步 2 次 forward，无需 optimizer state。
- **Inference-Mode Training**：训练过程不使用反向传播、不存储中间激活、不维护 optimizer states，显存占用与推理阶段相同。
- **Directional Curvature Estimator**：通过三点差分 $[L(\theta+\epsilon z), L(\theta), L(\theta-\epsilon z)]$ 估计损失沿扰动方向 $z$ 的标量曲率 $\hat{c} \approx z^\top \nabla^2 L \, z$。
- **α-Saturation Weighting**：用 $w_i = |\hat{c}_i|^\alpha$（夹在正则下界内）对高曲率方向施加弱非线性下加权，避免牛顿步的过激衰减。
- **Johnson-Lindenstrauss (JL) Lemma**：高维空间中的点集可以低维随机线性映射几乎等距嵌入，本文扩展证明该性质也保持方向曲率内积。
- **Bit-Packed Perturbation**：将 Rademacher 扰动向量以每参数 1 bit 压缩存储，配合 Triton fused kernel 即时解包并应用，消除 $O(d)$ 显存占用。

## 可复现要素
- **数据集**：GLUE/SuperGLUE（SST-2、RTE、BoolQ、WSC、WiC）、Stable ToolBench；均为公开基准。
- **代码/权重开源**：论文未明确声明开源状态（以 arXiv 版本为准，通常代码会在项目页面或 GitHub 发布，需关注后续更新）。
- **关键超参**：$\alpha = 0.1$、$\lambda_{\text{reg}} = 1$、$\lambda = \epsilon \in \{10^{-3}, 5\times10^{-4}, \ldots, 10^{-7}\}$ sweep、$n_{\text{pert}} \in \{40, 60, 100\}$、effective batch size ∈ {128, 256}、sequence length=256、plateau schedule（连续 10 次无改进则 $\lambda,\epsilon$ 减半）。
- **硬件**：8×A100 GPU 集群。
- **模型**：OPT-13B、OPT-30B、Qwen3-8B、Qwen3-1.7B。
