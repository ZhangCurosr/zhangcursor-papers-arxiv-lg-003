---
title: "Unlocking-Any-Order-Generation-in-Pretrained-Autoregressive"
source: https://arxiv.org/pdf/2610.39293v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:06:46"
field: "自回归图像生成与任意顺序建模"
keywords: ["autoregressive image generation", "any-order generation", "future-aware positional embedding", "ordering curriculum", "image inpainting", "discrete diffusion", "KV caching"]
innovations: ["提出轻量适配方案将预训练光栅扫描AR图像模型转为任意顺序生成器，无需从头训练", "引入未来感知位置嵌入解决因果任意顺序生成的可辨识性问题，保持单流Transformer架构", "设计顺序课程学习平滑插值训练分布，避免灾难性遗忘并稳定优化路径"]
benchmarks: ["ImageNet-1k class-conditional generation", "ImageNet inpainting", "ImageNet outpainting"]
---

# 论文速读：Unlocking-Any-Order-Generation-in-Pretrained-Autoregressive

## 一句话总结
论文提出一种轻量级适配方案，将已预训练的光栅扫描顺序自回归(AR)图像生成模型转换为任意顺序生成器，仅需增加约13–33%的算力成本，同时保持KV caching与vLLM等标准AR推理栈的兼容性，在ImageNet inpainting/outpainting任务上显著超越MaskGIT。

## 研究问题与动机
- 现有AR图像生成模型以固定光栅扫描顺序训练，硬编码了单一因子分解方式，推理时无法对非连续上下文进行条件生成（如inpainting、outpainting、编辑）。
- 现有任意顺序生成方案（如RandAR、离散扩散）均需从头训练，计算成本约为标准光栅扫描训练的3倍，且离散扩散因双向注意力无法复用KV caching与AR推理框架。
- 论文认为预训练光栅扫描AR模型已学到图像联合分布，所缺的只是"非顺序条件路径的表达力"，轻量适配即可解锁，无需重新训练。
- 直接对预训练模型施加任意顺序损失会引发两类问题：（1）可辨识性问题——相同上下文对应不同预测目标，梯度冲突；（2）优化不稳定——随机排列样本的高损失主导梯度，导致灾难性遗忘。

## 核心贡献（创新点）
- **轻量适配框架，避免从头训练**：证明预训练光栅扫描AR模型只需约1.25×原始预算即可解锁任意顺序能力，相比从头训练节省约2.4倍算力；与RandAR/MAR等从头方案在成本与生态兼容性上形成本质区分。
- **未来感知位置嵌入（future-aware positional embeddings）**：在输入层添加一个可学习的嵌入表，将下一目标位置σ(t+1)的身份注入当前token表示，以极小额外参数解决因果任意顺序生成的可辨识性问题；与XLNet双流注意力相比，不翻倍计算量、不破坏单流Transformer与KV cache。
- **顺序课程学习（ordering curriculum）**：训练排列从纯光栅扫描平滑插值至均匀随机，α线性从0增长到1；与单纯降低学习率不同，课程修改的是梯度组成而非步长，使优化路径始终停留在预训练盆地附近，避免灾难性遗忘。
- **保持AR推理生态兼容**：适配后模型仍为标准因果单流Transformer，可直接复用vLLM、并行解码等已有推理加速手段。

## 方法详解
- **任意顺序目标**：训练时采样排列σ∼P_α，最大化期望对数似然
  L_any-order(θ) = E_{σ}[Σ_{t=1}^{n} log p_θ(x_{σ(t)} | x_{σ(<t)})]。
- **可辨识性问题**：在变序下，给定相同观测上下文x_S预测不同目标x_j vs x_k会产生冲突梯度，目标不可区分。
- **未来感知位置嵌入**：引入E∈R^{n×d}，序列位置t的输入表示增广为 h̃_t = h_t + E_{σ(t+1)}（末步用零嵌入E_∅）。嵌入仅在输入层叠加一次，共享于所有序列位置与排列。
- **位置编码处理**：无论LlamaGen（RoPE）还是RAR（绝对位置表），注意力/位置查表均使用空间索引σ(t)而非序列步t，保留预训练学到的空间关系先验。
- **顺序课程**：P_α = (1−α)·δ_{σ_raster} + α·Uniform(S_n)，α(s)=min(s/S_ramp, 1)线性调度。早期以光栅扫描梯度为主稳定参数，后期逐步增加随机排列比重。
- **训练目标**：L(θ) = E_{σ~P_α}[Σ −log p_θ(x_{σ(t)} | x_{σ(<t)}, E_{σ(t+1)})]。
- **推理流程**：用户指定生成顺序σ（可将已知token置于首位），依次用h̃_t = h_t + E_{σ(t+1)}作为输入，标准因果forward；KV cache可直接复用。

## 实验与结果
- **数据集与基线**：ImageNet-1k 256×256类条件生成；适配4个基座模型（LlamaGen-L/XL、RAR-L/XL）；对比MaskGIT（cosine/Halton调度）、MAR-L、VAR、DiT-XL等。
- **类条件生成（FID↓ IS↑ Prec↑ Rec↑）**：
  - LlamaGen-L：FID 3.07→4.24（+1.17），IS 256.06→231.62
  - LlamaGen-XL：FID 2.63→3.91（+1.28），IS 244.08→252.94
  - RAR-L：FID 1.70→2.68（+0.98），IS 299.5→283.13
  - RAR-XL：FID 1.50→2.68（+1.18），IS 306.9→285.16
  - Precision/Recall基本维持。
- **Inpainting/Outpainting（50K样本）**：
  - Inpainting：Ours FID=6.21，优于MaskGIT cosine 10.44、Halton 7.82，接近MAR-L 6.63。
  - Outpainting：Ours FID=5.84，优于MaskGIT cosine 12.17、Halton 7.91，接近MAR-L 6.86。
- **训练效率**：LlamaGen-XL适配总成本约1.25×原始预训练；相比从头训练的RandAR（≈3×）节省约2.4倍；对已有预训练checkpoint的边际成本为预训练的25%，比从头便宜12倍。
- **并行解码**：LlamaGen-XL从576步降至64步（9×前向传播减少），FID 3.91→5.03。
- **消融**：移除未来感知嵌入→loss上升无法收敛；移除课程（α=1直接训练）→FID升至5.86；嵌入学习率需×10否则收敛不良；加teacher KL正则反而使FID升至9.62。

## 相关工作脉络
- **XLNet [28]**：解决语言任意顺序可辨识性，采用双流注意力；本文用单流+轻量输入嵌入替代，保留KV cache。
- **RandAR [18]**：图像域任意顺序自回归，需从头训练约300 epoch；本文从预训练光栅扫描模型轻量适配，成本大幅更低。
- **MAR [3]**：用扩散头预测连续token实现任意顺序，需从头训练，且不在AR推理生态内；本文保持离散token+因果Transformer。
- **MaskGIT / 离散扩散 [2,23]**：以双向注意力迭代去噪实现任意顺序，无法复用KV cache与vLLM；本文维持因果单流。
- **ControlNet [36]**：面向单一控制信号的任务适配；本文支持任意观测token子集的条件生成，通用性更强。
- **RAR [5]**：随机化自回归图像生成基座；本文在其预训练权重上进一步解锁任意顺序能力。

## 局限性与未来方向
- 实验仅限ImageNet类条件生成与256×256分辨率，未验证到text-to-image AR模型或更高分辨率。
- 仅适配基于token的AR模型，MAR等非token/连续token变体不在适用范围内。
- 编辑评估仅覆盖区域补全，未测试指令跟随、身份保留等更复杂编辑场景。
- 随机初始化对照实验在计算量上匹配但未匹配训练调度，对比并非严格公平。

## 研究启发与可借鉴点
- **"适配而非重训"范式**：证明预训练表征的顺序无关性可通过轻量干预解锁，该思路可推广至其他预训练生成模型（如视频、3D）的能力扩展。
- **轻量条件嵌入解决可辨识性**：输入级目标身份注入比双流架构经济得多，类似技巧可用于任意序列生成任务中的目标定位问题。
- **课程式损失混合控制优化路径**：通过平滑调整训练分布组成而非单纯调学习率来防止灾难性遗忘，方法论可迁移至多目标/多模态微调。
- **与现有推理栈兼容的设计优先**：保持单流因果Transformer不变以复用KV cache/vLLM，对工业部署路径极具参考价值。
- **可与本团队结合的创新机会**：将本适配流程应用于团队已有的文本/多模态AR预训练模型，探索inpainting、region editing、parallel decoding的统一接口；或将顺序课程思想迁移至多步Diffusion/Dit的微调策略。

## 关键术语表
**Autoregressive (AR) image generation**：将图像经VQTokenizer离散化为token序列后按因果顺序预测的图像生成范式，继承语言模型的可扩展性与推理生态。
**Any-order generation**：不固定生成顺序，允许模型以任意观测token子集为条件预测剩余token的能力，一次训练覆盖inpainting/outpainting/editing。
**Future-aware positional embeddings**：在输入层将下一目标空间位置σ(t+1)的嵌入加到当前token表示上，使因果模型能区分不同预测目标，解决可辨识性问题。
**Ordering curriculum**：训练排列从纯光栅扫描平滑插值到均匀随机分布的课程调度，避免直接从预训练参数跳入高方差任意顺序优化。
**Identifiability problem**：在任意顺序生成中，相同观测上下文对应不同预测目标时产生冲突梯度的问题。
**KV caching**：自回归生成中缓存已生成token的键值状态以加速后续步；本文方法因保持单流因果架构而完整兼容。
**Catastrophic forgetting**：从预训练初始化直接施加高方差任意顺序梯度时，原有表征被迅速破坏的现象。

## 可复现要素
- **数据集**：ImageNet-1k（公开），适配用1.28M训练集，inpainting/outpainting评估用50K验证集。
- **代码**：https://github.com/Ordered-AR/unlockedadaptation（开源）。
- **权重**：基于公开预训练checkpoint（LlamaGen MIT、RAR Apache 2.0）适配；适配后权重随论文发布。
- **关键超参**：AdamW (β₁=0.9, β₂=0.95)，weight decay=10⁻⁴；嵌入表学习率为基础学习率的10倍；课程α线性ramp，S_ramp=10–30 epoch；适配轮数LlamaGen-L 100 / LlamaGen-XL 25 / RAR-L 100 / RAR-XL 50；bf16混合精度；top-k解码，temperature=1.0，CFG scale=1.75；序列长度256/576 token。
