---
title: "Unlocking-Any-Order-Generation-in-Pretrained-Autoregressive"
source: https://arxiv.org/pdf/2610.39293v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:06:25"
field: "生成式视觉模型"
keywords: ["自回归图像生成", "任意顺序生成", "预训练模型适配", "未来感知位置嵌入", "排序课程学习", "图像修复", "推理效率"]
innovations: ["提出未来感知位置嵌入解决因果任意顺序生成的标识符冲突", "设计排序课程学习实现从预训练到任意顺序的稳定适配", "以13-33%额外计算成本解锁任意顺序生成并保持AR推理生态兼容"]
benchmarks: ["ImageNet-1k 256x256", "Inpainting FID", "Outpainting FID", "Parallel Decoding Efficiency"]
---

# 论文速读：Unlocking Any-Order Generation in Pretrained Autoregressive Image Models

## 一句话总结
本文提出一种轻量适配方案，将已训练的栅格扫描自回归（AR）图像生成模型转换为支持任意顺序生成的模型，仅需增加约 13–33% 的计算成本，即可实现无损、非顺序上下文条件下的图像修复、外扩与编辑，且完全兼容现有 AR 推理生态（KV 缓存、vLLM）。

## 研究问题与动机
- **固定顺序限制**：标准 AR 图像模型以固定光栅扫描顺序训练并推理，无法在推理时处理非连续上下文（如 inpainting、outpainting），限制了模型灵活性。
- **从scratch训练的昂贵代价**：现有任意顺序生成方法（如 MaskGIT、RandAR、离散扩散模型）需从零开始训练，计算开销约为栅格扫描训练的 3 倍。
- **推理生态不兼容**：离散扩散模型采用双向注意力与迭代细化解码，无法利用 KV 缓存、投机解码及 vLLM 等 AR 推理加速框架。
- **知识冗余假设**：预训练栅格扫描模型已学习图像 token 的联合分布，只需通过轻量适配解锁"任意条件路径表达能力"，无需重新学习分布知识。

## 核心贡献（创新点）
- **未来感知位置嵌入（Future-Aware Positional Embeddings）**：通过在输入层叠加目标位置嵌入解决因果任意顺序生成中的标识符冲突问题，仅增加 (n+1)·d 参数，且不破坏单流 Transformer 架构。
- **排序课程学习策略（Ordering Curriculum）**：设计从栅格扫描到均匀随机排列的线性插值课程，避免直接从预训练初始化跳入高方差梯度导致的灾难性遗忘。
- **轻量级适配框架**：提出"预训练模型→任意顺序生成"的适配配方，相比从零训练节省约 2.4× 端到端计算成本（仅需 25% 额外预算）。
- **保持 AR 推理兼容性**：适配后模型仍支持 KV 缓存、并行解码及 vLLM 部署，实现零任务特定训练即可执行 inpainting、outpainting 和区域条件编辑。

## 方法详解
- **目标函数**：任意顺序训练最大化不同置换 σ 下的对数似然期望，即 $\mathcal{L}_{\text{any-order}}(\theta) = \mathbb{E}_{\sigma \sim P}\left[\sum_{t=1}^{n} \log p_\theta(x_{\sigma(t)} | x_{\sigma(<t)})\right]$。
- **标识符问题**：相同上下文需预测不同目标 token 时产生冲突梯度；方法通过添加 $E_{\sigma(t+1)}$ 使每个步骤的目标位置可识别。
- **位置嵌入设计**：在输入层添加学习型嵌入表 $E \in \mathbb{R}^{n \times d}$，对序列位置 t 的表示做加法注入 $\tilde{h}_t = h_t + E_{\sigma(t+1)}$，最后一位使用空嵌入 $E_\emptyset$。
- **课程策略**：采样分布为 $P_\alpha = (1-\alpha)\delta_{\sigma_{\text{raster}}} + \alpha \cdot \text{Uniform}(S_n)$，α 线性从 0 增至 1，逐步引入随机置换。
- **超参数细节**：嵌入表零初始化，学习率设为基础学习率的 10 倍；空间位置信息用于 RoPE/RAR 的位置编码，保持预训练的空间关系先验不变。

## 实验与结果
- **数据集**：ImageNet-1k（ILSVRC-2012），256×256 分辨率，适配使用 1.28M 训练集，评估使用 50,000 验证集。
- **基线模型**：LlamaGen-L/XL、RAR-L/XL，两类架构共四个规模。
- **分类条件生成**：LlamaGen-XL 适配后 FID 从 2.63 升至 3.91（+1.28），IS/Precision/Recall 基本持平。
- **图像修复（Inpainting）**：FID 6.21 显著优于 MaskGIT（10.44）与 MaskGIT-Halton（7.82），略优于 MAR-L（6.63）。
- **图像外扩（Outpainting）**：FID 5.84 优于 MaskGIT（12.17）与 MaskGIT-Halton（7.91），略优于 MAR-L（6.86）。
- **计算效率**：适配成本仅 1.25× 原始预训练预算（LlamaGen-XL），远低于 RandAR/MAR 的 3–4×；未来嵌入仅增加 0.74M 参数（0.095%）。
- **消融验证**：移除未来感知嵌入导致损失不收敛；无课程（α=1 从零开始）导致 FID 恶化至 5.86；嵌入学习率提升 10× 对稳定收敛至关重要。
- **并行解码**：在 LlamaGen-XL 上，64 步（较原生 576 步减少 9×）仅造成 1.12 FID 损耗。

## 相关工作脉络
- **标准 AR 图像生成**（LlamaGen、VAR）：固定栅格扫描顺序，推理高效但缺乏灵活性；本文保持其架构与推理特性。
- **MaskGIT/离散扩散**（MaskGIT、Muse、MaskBit）：支持任意顺序但依赖双向注意力与迭代细化，无法复用 KV 缓存与 AR 推理栈。
- **随机顺序 AR**（RandAR）：从零训练任意顺序生成器，成本约 3× 栅格扫描训练；本文证明可从小量适配获得类似能力。
- **MAR**（自回归扩散模型）：预测连续 token，不在本文适配范畴；本文针对离散 token 的 AR 模型。
- **XLNet 两流注意力**：语言领域的任意顺序解决方案，需双倍注意力计算；本文用轻量加法嵌入替代，保留单流架构。
- **预训练模型适配**（LoRA、VIT）：传统适配针对新任务/域；本文针对同一数据分布的"因式分解方式"进行适配，概念新颖。

## 局限性与未来方向
- **评估范围受限**：仅在 ImageNet 256×256 分类条件生成上验证，未测试文本到图像的 AR 模型或更高分辨率。
- **模型类型限制**：仅适配基于 token 的 AR 模型，连续 token 预测模型（如 MAR）超出适用范围。
- **编辑任务单一**：仅评估区域条件编辑（region completion），未覆盖指令遵循或身份保持等更复杂编辑场景。
- **消融控制不全**：随机初始化对照组仅匹配计算量而未匹配训练调度，可能存在公平性偏差。
- **未来方向**：扩展至 text-to-image 场景、更高分辨率、结合 ControlNet 等任务特定适配器、探索非 token 化架构的适配方案。

## 研究启发与可借鉴点
- **"知识已存在，只需解锁表达"的思路**：预训练模型的表征具有顺序无关性，轻量适配即可激活新能力，避免重复昂贵预训练。
- **未来感知嵌入的简洁性**：用单次加法注入替代两流注意力，以极小参数代价解决标识符问题，值得在其他序列建模任务中借鉴。
- **课程学习作为适配稳定器**：将 Curriculum 视为 continuation method，逐步扩展梯度分布，避免从预训练权重的直接跳到不稳定区域。
- **推理生态兼容性作为设计约束**：在追求功能扩展的同时保留 KV 缓存与 vLLM 支持，为实际部署提供可行路径。
- **跨架构泛化验证**：在 LlamaGen 与 RAR 两种不同位置编码（RoPE vs. 加法绝对位置）上均有效，证明方法的通用性。

## 关键术语表
- **Autoregressive (AR) Generation**：自回归生成，按顺序逐个预测离散 token 的图像生成范式。
- **Raster-Scan Order**：光栅扫描顺序，从左到右、从上到下的标准图像 token 预测顺序。
- **Any-Order Generation**：任意顺序生成，允许模型根据任意置换 σ 预测 token，支持非连续上下文条件。
- **Future-Aware Positional Embedding**：未来感知位置嵌入，在输入层注入下一个目标位置信息的可学习嵌入，解决因果任意顺序生成的标识符冲突。
- **Ordering Curriculum**：排序课程，从栅格扫描逐步过渡到随机置换的训练策略，避免预训练权重被破坏。
- **Discrete Diffusion**：离散扩散，将任意顺序生成建模为迭代掩码去噪过程的模型家族，如 MaskGIT。
- **KV Caching**：键值缓存，AR 推理中缓存已生成 token 的中间表示以加速后续预测的技术。
- **Catastrophic Forgetting**：灾难性遗忘，快速更新导致预训练表示被破坏而丧失原有能力的现象。

## 可复现要素
- **数据集**：ImageNet-1k（ILSVRC-2012），公开可用。
- **代码开源**：https://github.com/Ordered-AR/unlockedadaptation。
- **模型权重**：适配后的 LlamaGen-L/XL 与 RAR-L/XL 权重将随论文发布（继承基座模型的 MIT/Apache 2.0 许可）。
- **关键超参**：嵌入表学习率为基础学习率的 10 倍；课程系数 α 线性调度；适配步数 T_ft 占预训练 epochs 的 12.5%–33.3%；使用 AdamW（β₁=0.9, β₂=0.95）、weight decay 10⁻⁴–5×10⁻⁵、gradient clip=1.0、bf16 混合精度。
