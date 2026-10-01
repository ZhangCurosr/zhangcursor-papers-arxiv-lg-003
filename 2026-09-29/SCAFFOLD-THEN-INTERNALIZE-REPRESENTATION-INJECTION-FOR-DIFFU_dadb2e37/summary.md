---
title: "SCAFFOLD-THEN-INTERNALIZE-REPRESENTATION-INJECTION-FOR-DIFFU"
source: https://arxiv.org/pdf/2609.35292v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:58:29"
field: "生成式人工智能/扩散模型高效训练"
keywords: ["Diffusion Transformer", "Representation Injection", "REPI", "Scaffold-to-Internalization", "Flow Matching", "Visual Encoder"]
innovations: ["提出REPI将预训练编码器表征注入扩散Transformer参与去噪，与REPA形成互补", "设计脚手架-内化训练策略，160K步匹配vanilla SiT 7M步，加速43.5倍", "证明K/V联合注入优于单一Q或V注入，且对注入层位置和超参数鲁棒"]
benchmarks: ["ImageNet 256x256", "ImageNet 512x512", "MS-COCO"]
---

# 论文速读：SCAFFOLD-THEN-INTERNALIZE-REPRESENTATION-INJECTION-FOR-DIFFU

## 一句话总结
本文提出 **REPI（REpresentation Injection）**，将预训练视觉编码器的表征通过"脚手架-内化"策略注入扩散 Transformer 参与去噪过程，并在推理时移除。REPI 与 REPA 高度互补，二者结合仅需 160K 训练步即可匹配 SiT 7M 步的 FID，加速超过 **43.5×**。

---

## 研究问题与动机
1. **扩散 Transformer 训练效率低**：SiT 等模型需要大量训练步（如 7M）才能达到高样本质量，核心瓶颈在于从噪声中学习语义结构化视觉表征困难。
2. **现有 REPA 方法的单向性**：REPA 将扩散 Transformer 的隐状态投影到编码器空间进行对齐，而本文探索一个反向且互补的方向——将编码器表征**注入**扩散 Transformer，直接参与去噪计算。
3. **简单反转设计的失败**：直接将编码器输出替换 Transformer 中间层的隐藏状态会导致训练崩溃（FID=212.5），因为会切断来自噪声输入的信息流。因此需要探索更精细的注入点设计。
4. **推理时无法获取干净图像表征**：Oracle 设置下（推理时可直接使用干净图像编码器表征），30K 步即可超越 vanilla SiT 7M 步，但实际生成中不可行，需将外部表征**内化**为模型自身能力。

---

## 核心贡献（创新点）
1. **反向注入范式**：提出将编码器表征注入扩散 Transformer 内部（而非投影到编码器空间），使外部语义表征直接参与去噪，这与 REPA 形成本质不同的知识传递方向。
2. **"脚手架-内化"训练策略（Scaffold-to-Internalization）**：编码器表征仅在训练早期作为临时脚手架使用，随后移除并通过内化损失鼓励模型自行复现兼容的 K/V 表征，使模型最终在推理时完全无需编码器。
3. **与 REPA 的高度互补性**：REPA 提供"扩散→编码器"的对齐监督，REPI 提供"编码器→扩散"的注入机制；二者组合在 SiT-XL 上 160K 步达到 FID=8.22，匹配 vanilla SiT 7M 步的 8.30，速度提升 **43.5×**。
4. **广泛的通用性验证**：REPI 在多种视觉编码器（DINOv2、MAE、MoCoV3、CLIP 等 10 种）、多种骨干网络（SiT、DiT、DiG）及分辨率（256、512）上均优于 REPA，且对注入层位置和超参数具有鲁棒性。

---

## 方法详解

### 1. 表示注入（Representation Injection）
- 在训练初期，从冻结的预训练视觉编码器 $f$ 的最终自注意力层提取干净的键值对 $(K_*, V_*)$。
- 通过两个可训练的线性投影层 $h_{\phi_K}$ 和 $h_{\phi_V}$ 将其映射到扩散 Transformer 目标层的 K/V 空间：$\bar{K} = h_{\phi_K}(K_*),\ \bar{V} = h_{\phi_V}(V_*)$。
- 将目标层的原生 K/V **替换**为 $\bar{K}, \bar{V}$，但保留原生 Query $Q$（来自噪声输入），注意力输出为：
$$Z_{\text{inj}} = \text{softmax}\left(\frac{Q\bar{K}^\top}{\sqrt{d}}\right)\bar{V}$$
- 设计要点：保留 $Q$ 确保噪声输入的信息流不被切断，$K$ 控制检索哪些外部信息，$V$ 提供内容。

### 2. 内化与脚手架移除（Internalization and Scaffold Removal）
- **脚手架阶段**：前 20K 步（默认）使用上述注入，投影层与扩散 Transformer 联合优化。
- **内化阶段**：移除脚手架，恢复原生的 K/V 计算，并引入内化损失：
$$\mathcal{L}_{\text{int}} = \frac{1}{|K|}\|K - \text{sg}[\bar{K}]\|_F^2 + \frac{1}{|V|}\|V - \text{sg}[\bar{V}]\|_F^2$$
其中 sg[·] 为 stop-gradient 算子。该损失引导原生 K/V 向脚手架阶段的值靠拢，实现平滑过渡。
- **总损失**：$\mathcal{L} = \mathcal{L}_{\text{velocity}} + \lambda \mathcal{L}_{\text{int}}$，默认 $\lambda = 2.0$。
- **推理时**：视觉编码器和投影层完全丢弃，模型架构与原始扩散 Transformer 一致。

### 3. 与 REPA 的结合
- REPA 对齐损失贯穿整个训练过程（脚手架+内化两阶段）。
- REPI 注入层位于 REPA 对齐层**之前**（layer 4 vs layer 8），保证注入后的表征先被 REPA 对齐监督，从而避免相互干扰。
- 结合时 $\lambda = 0.25$。

---

## 实验与结果

- **数据集**：ImageNet 256×256（主实验）、512×512（高分辨率）、MS-COCO（文本生成）。
- **骨干网络**：SiT-B/L/XL、DiT-L、DiG-L。
- **视觉编码器**：DINOv2-B/14（主实验），另有 MAE、MoCoV3、CLIP 等 10 种编码器验证通用性。
- **评估基线**：REPA、iREPA、sREPA、Stable Velocity。
- **关键结果**：

| 模型/方法 | 训练步数 | FID（无 CFG） |
|---|---|---|
| Vanilla SiT-XL/2 | 7M | 8.30 |
| + REPA | 100K | 19.40 |
| + REPI | 100K | 14.51 |
| **+ REPA + REPI** | **160K** | **8.22** |
| + REPA + REPI | 400K | 6.33 |
| + REPI | 400K | 7.21 |

- **最强结果**：SiT-XL + REPA + REPI，160K 步 FID=8.22，匹配 vanilla SiT 7M 步（FID=8.30），速度提升 **43.5×**；400K 步 FID 达 **6.33**。
- **文本生成**：MS-COCO 上 REPI + REPA 在有无 CFG 下均优于 REPA 和 REPI 单独使用。
- **消融结论**：K/V 联合注入效果最佳（FID=14.51 vs V-only=15.53）；脚手架持续时间 10K–30K 均稳定；内化损失权重 $\lambda \in [1.0, 4.0]$ 表现稳健。

---

## 相关工作脉络
1. **REPA（Yu et al., 2025）**：将扩散 Transformer 隐状态投影到编码器空间对齐，是本文方法的直接对比基线；本文走反向注入路线，形成互补。
2. **iREPA（Singh et al., 2026）**：强调空间结构对齐，与 REPA 同属"扩散→编码器"范式。
3. **sREPA（Xu et al., 2026）**：对齐关系几何结构，同样是辅助对齐损失方法。
4. **Stable Velocity（Yang et al., 2026）**：从方差视角优化 flow matching，与表征对齐正交。
5. **DiG（Zhu et al., 2024）**：门控线性注意力扩散模型，本文验证 REPI 同样适用于此类架构。
6. **SiT（Ma et al., 2024）**：Scalable Interpolant Transformer，作为主要评估骨架网络。

---

## 局限性与未来方向
1. **推理时完全丢弃编码器**：虽然消除了部署开销，但内化是否完全等价于保留编码器注入仍存疑问。
2. **仅验证了图像生成**：扩展到视频生成等其他领域有待探索（作者已提及为未来方向）。
3. **注入层选择仍依赖实验调优**：尽管对层位置鲁棒（4–14 层差异不大），但未给出理论指导。
4. **未与基于 VAE 特征的自对齐方法对比**：如 SRA²（Wang et al., 2026）。

---

## 研究启发与可借鉴点
1. **"脚手架-内化"范式可迁移**：此策略（先用外部知识快速初始化，再引导模型自行复现）可推广至其他预训练知识迁移场景（如大语言模型、音频生成等）。
2. **注意力的 K/V 分离设计**：保留 Q 以维持输入依赖性、替换 K/V 注入外部语义——这一设计原则对其他基于注意力机制的模型同样适用。
3. **方法论层面的"正反双路"启发**：当某一方法（如 REPA）已建立后，系统性地探索其反向操作（反向映射）可能发现互补甚至更强的新路径。
4. **实验设计严谨**：包含 Oracle 诊断实验（证明思想可行性）、消融实验（各组件贡献）、多编码器/多架构/多分辨率泛化验证，值得借鉴。
5. **与现有最优方法天然兼容**：REPI 与 REPA 可直接叠加，无需重新设计整体框架，这种模块化设计利于实际应用集成。

---

## 关键术语表
**REPI（REpresentation Injection）**：将预训练编码器表征注入扩散 Transformer 并逐步内化的训练框架，与 REPA 方向相反且互补。
**Scaffold-to-Internalization**：训练前期将外部表征作为临时脚手架引导模型学习，后期移除脚手架并通过损失函数促使模型自行复现同类表征的策略。
**REPA（Representation Alignment）**：通过将扩散 Transformer 隐状态投影到预训练编码器空间并对齐，以加速训练的已有方法。
**Internalization Loss**：$\mathcal{L}_{\text{int}} = \frac{1}{|K|}\|K - \text{sg}[\bar{K}]\|_F^2 + \frac{1}{|V|}\|V - \text{sg}[\bar{V}]\|_F^2$，用于引导原生 K/V 逼近脚手架阶段的编码器投影值。
**Oracle Setting**：假设推理时可直接获取干净图像编码器表征的理想化实验设置，用于验证注入思路的可行性。
**Flow Matching**：一种扩散模型训练范式，通过最小化速度预测误差 $\|\mathbf{v}_\theta(\mathbf{x}_t, t, c) - \mathbf{v}_t\|^2$ 来训练去噪器。

---

## 可复现要素
- **数据集**：ImageNet（公开）、MS-COCO（公开）；论文遵循 REPA 的相同协议。
- **代码**：作者声明代码将在 https://jeneveuxpas.github.io/REPI 开源。
- **权重**：未提及预训练权重开源计划。
- **关键超参**：$\lambda = 2.0$（独立使用）、$\lambda = 0.25$（结合 REPA）；学习率 $10^{-4}$，batch size 256，AdamW，scaffold 持续 20K 步，fp16 混合精度，EMA；SiT 系列用 Euler-Maruyama 采样器（250 NFE），DiT/DiG 用 Improved DDPM。
- **硬件**：4× NVIDIA H200 GPU。

---
