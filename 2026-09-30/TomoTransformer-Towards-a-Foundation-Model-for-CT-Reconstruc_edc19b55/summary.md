---
title: "TomoTransformer-Towards-a-Foundation-Model-for-CT-Reconstruc"
source: https://arxiv.org/pdf/2609.37605v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:35:00"
field: "医学影像重建"
keywords: ["sparse-view CT", "foundation model", "transformer reconstruction", "disentangled back-projection", "geometric attention", "zero-shot denoising"]
innovations: ["在解耦后投影空间中提出协议无关的 transformer 重建基础模型，支持任意输入/输出视角数和探测器尺寸", "设计几何 tokenizer 与 GAM 注意力机制，将物理先验直接编码入注意力偏置", "单次前向传播即可实现零样本去噪，结合重掩码策略无需微调提升重建质量"]
benchmarks: ["LoDoPaB-CT", "KiTS23", "COVID-19 CTSpine1K", "MSD-T10", "COLONOG", "HNSCC", "FFHQ", "ImageNet", "SLS 纳米脑组织 PXCT"]
---

# 论文速读：TomoTransformer-Towards-a-Foundation-Model-for-CT-Reconstruc

## 一句话总结
论文提出 TomoTransformer，一个基于 transformer 的 CT 重建基础模型，通过将解耦后投影空间中的局部滤波投影编码为 token 并利用自注意力预测缺失视角，实现了对输入投影数量、角度配置和探测器尺寸的完全协议无关性。

## 研究问题与动机
- 现有监督深度学习重建模型通常针对固定投影数量、固定角度范围和固定探测器尺寸训练，在分布偏移时性能急剧下降，部署灵活性受限。
- FBP 在稀疏视角下会产生条纹伪影并丢失精细结构，而基于迭代优化的传统方法计算成本高。
- ViewTrans 等近期多用途方法虽尝试提升泛化能力，但仍受限于固定探测器尺寸和图像分辨率，无法真正处理任意尺度的数据采集。
- 真实应用场景（如纳米级 X 射线成像）中数据采集协议差异极大，缺乏一个统一的通用重建基础模型。

## 核心贡献（创新点）
- 提出解耦后投影空间中的视图预测框架，将每个单视角后投影视为独立 token，使视图插值在几何上自然对齐且对探测器尺寸不变。
- 设计几何 tokenizer 和 decoder，通过沿条纹方向采样将 2D 局部后投影块压缩为 1D 特征向量，显著降低计算复杂度并实现内存高效处理。
- 引入几何注意力机制（GAM），将成对角度差以学习函数形式注入 attention logits，利用物理先验增强重建质量。
- 在涵盖多种医学 CT 解剖结构和自然图像的大规模异构数据集上预训练，展示零样本迁移至纳米级脑组织 PXCT 数据的强大泛化能力。
- 证明单次前向传播即可实现零样本去噪，并结合盲斑重掩码和互补重掩码策略进一步提升重建质量。

## 方法详解
- **解耦后投影空间**：将 sinogram 沿角度方向分离，每个视角 r 的后投影 $\mathbf{b}_r \in \mathbb{R}^{N \times N}$ 通过公式 (3) 独立计算，形成张量 $\mathbf{B} = [\mathbf{b}_1, \ldots, \mathbf{b}_R] \in \mathbb{R}^{N \times N \times R}$，使不同视角在同一空间网格上对齐。
- **局部 patch 提取与几何 tokenizer**：从每个单视角后投影中提取 $P \times P$ 局部块（$P=32$），沿条纹法线方向对角线采样 $K=\lceil P\sqrt{2} \rceil=46$ 个等距点得到 1D 特征向量 $\mathbf{p}_r \in \mathbb{R}^K$，再通过线性层投影至嵌入维度 $D=512$。
- **角度位置编码**：将每个投影角度 $\theta_r$ 映射为 $F=32$ 个频率的傅里叶位置编码 $[\sin(k\theta_r), \cos(k\theta_r)]_{k=1}^F$，并可通过可学习门控标量自适应加权，支持任意连续角度。
- **几何注意力机制（GAM）**：在标准 dot-product attention 基础上，将成对角度差 $\Delta_{rr'} = \theta_r - \theta_{r'}$ 的 learned function $g_\phi(\sin\Delta, \cos\Delta)$ 加入 attention logits，公式 (6) 体现了物理感知的注意力偏置设计。
- **MAE 式编解码架构**：encoder 仅处理观测 token，decoder 接收 encoder 输出与可学习 mask token 的完整序列；两阶段均使用 4 层 transformer block（宽度 512，8 头注意力）。
- **训练损失**：对未观测目标视角的 patch 计算 $\ell_1$ 损失（公式 7），使用 AdamW 优化器，余弦学习率衰减，训练 $10^6$ 步。

## 实验与结果
- **数据集**：约 65 万张 2D 切片（57 万医学 CT 来自 LoDoPaB-CT、COVID-19 CTSpine1K、KiTS23、MSD-T10、COLONOG、HNSCC；8 万自然图像来自 FFHQ、ImageNet），均在原始分辨率下使用，无需重采样。
- **固定稀疏重建**（64→256 视角）：TomoTransformer (fixed) 在 8 个数据集上平均 PSNR 达 39.66 dB，SSIM 0.931，优于所有 CNN/Transformer 基线（NAFNet 39.37/0.926、Restormer 39.47/0.927）；TomoTransformer (varied) 平均 PSNR 38.32 dB 仍接近专用基线。
- **可变稀疏重建**（LoDoPaB-CT）：128→256 时 TomoTransformer 达 42.40 dB / 0.959，显著高于 ViewTrans 的 37.75 dB / 0.914；在极稀疏 64→256 下优势更明显（37.68 dB / 0.910 vs. 33.07 dB / 0.832），ViewTrans 出现严重漩涡状伪影。
- **零样本去噪**：在 45 dB 高斯噪声和泊松光子计数噪声下，预训练模型无需微调即可显著抑制 FBP 条纹噪声；结合盲斑重掩码（BS）和互补重掩码（CR）策略进一步提升质量。
- **真实实验数据**：在瑞士光源 SLS 获取的纳米级脑组织 PXCT 数据（768×768，448 层）上，128→256 视角重建 PSNR 35.8 dB / SSIM 0.837，128→512 达 38.8 dB / 0.916，成功恢复细胞级精细结构；ViewTrans 因固定分辨率限制无法应用，专用基线出现虚假纹理和块状伪影。

## 相关工作脉络
- **U-Net 类 CNN 基线**（FBPConvNet、DRUNet）：采用端到端图像域后处理，受限于固定输入尺度和单一解剖结构，缺乏协议泛化能力。
- **全变分/稀疏正则迭代方法**（Sidky & Pan 2008）：基于手工先验抑制伪影，但优化缓慢且难以捕捉复杂纹理结构。
- **双域 Transformer**（DuDoTrans、CTTR）：直接在 sinogram 和图像域联合建模，仍绑定固定探测器分辨率，无法处理任意角度配置。
- **ViewTrans（Chen et al., 2026）**：首个尝试多用途 sinogram 补全的 transformer，但 token 维度等于探测器宽度 M，架构与分辨率强耦合；本文工作从根本上消除了这一限制。
- **扩散模型逆问题求解**（Chung et al., 2023）：通过先验解耦实现协议灵活性，但需数百至数千次分数评估，推理成本高昂。
- **物理感知注意力设计**：GAM 将 Radon 变换的几何先验直接编码进 attention 机制，区别于纯数据驱动的位置编码方案。

## 局限性与未来方向
- 论文仅验证了 2D 平行束几何，三维锥束 CT（Cone-beam CT）的扩展待研究（作者在 Conclusion 中明确提及）。
- 训练数据虽涵盖多种解剖结构和自然图像，但未包含金属植入物、运动伪影等临床常见干扰模态。
- 推断时采用固定 patch 分块策略（$P=32$），对于极端高分辨率卷可能引入边界拼接伪影。
- 零样本去噪效果依赖噪声分布与训练数据的距离，未见对极端强噪声（<30 dB）的量化评估。
- 模型参数量约 12.7M，在超大规模数据集上的缩放行为尚未充分研究。

## 研究启发与可借鉴点
- **解耦表示策略**：将多视角观测分解为空间对齐的单视角后投影 token，为任何多视图重建任务提供通用的几何对齐表征范式，可迁移至 MRI 重建、SAR 成像等领域。
- **几何注意力机制**：将成对相对坐标（角度差、空间距离）以学习偏置形式注入 attention，是物理先验与 transformer 结合的高效低开销方案，可在地震波成像、声学层析等正演算子已知的逆问题中复用。
- **局部 patch 训练 + 全局推理**：训练时仅处理小 patch 但保持模型尺度无关性，既降低显存占用又天然支持任意分辨率推断，适合大规模预训练基础设施。
- **零样本去噪作为附加能力**：在干净数据上预训练的视图补全模型可隐式学习平滑先验，结合重掩码策略实现无需额外训练的去噪，为多任务统一架构设计提供新思路。
- **异构数据联合预训练**：医学 CT 与自然图像混合训练可增强纹理归纳偏置，该策略适用于跨模态医学基础模型的构建。

## 关键术语表
- **Filtered Back-Projection (FBP)**：基于 Fourier Slice Theorem 的经典解析重建方法，通过对 sinogram 各列进行 ramp 滤波后反投影求平均得到重建图像。
- **Disentangled Back-projection Space**：将 sinogram 按视角分解为独立单视角后投影张量 $\mathbf{B} \in \mathbb{R}^{N\times N\times R}$，使不同视角在同一空间坐标上对齐。
- **Geometric Tokenizer**：沿单视角后投影条纹法线方向对角线采样，将 2D 局部块压缩为 1D 特征向量的几何感知编码模块。
- **Geometric Attention Mechanism (GAM)**：在 self-attention logits 中加入成对角度差的 learned function 偏置，显式编码视角间几何关系。
- **Blind-spot Re-masking (BS)**：将测量视角划分为互不重叠子集并逐轮掩码预测，以迭代方式去除测量数据中的噪声。
- **Complementary Re-masking (CR)**：将首轮预测的缺失视角作为输入反向预测原始测量视角，平均两轮结果实现双向去噪。
- **Ptychographic X-ray Computed Tomography (PXCT)**：利用扫描衍射成像测量样品折射率复分布的高分辨率纳米级 X 射线层析技术。
- **Foundation Model（基础模型）**：在大规模异构数据上预训练、可通过零样本或少量微调适配多种下游任务与条件的统一模型。

## 可复现要素
- **数据集**：LoDoPaB-CT、COVID-19 CTSpine1K、KiTS23、MSD-T10、COLONOG、HNSCC、FFHQ、ImageNet 均为公开数据集；纳米级脑组织数据来自 Swiss Light Source (SLS) 合作者提供（Bosch et al., 2025），论文未声明独立开源。
- **代码**：论文未提及开源仓库链接。
- **权重**：论文未声明预训练权重公开方式。
- **关键超参**：patch 大小 $P=32$，采样点数 $K=46$，嵌入维度 $D=512$，注意力头数 $H=8$，傅里叶频率 $F=32$，优化器 AdamW（$\beta_1=0.9, \beta_2=0.95$，weight decay 0.05），基础学习率 $3\times10^{-4}$ 余弦衰减，梯度裁剪 norm=1.0，训练 $10^6$ 步单卡 A100。
