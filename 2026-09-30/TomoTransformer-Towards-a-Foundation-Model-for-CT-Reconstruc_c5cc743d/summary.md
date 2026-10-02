---
title: "TomoTransformer-Towards-a-Foundation-Model-for-CT-Reconstruc"
source: https://arxiv.org/pdf/2609.37605v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:34:40"
field: "医学图像重建"
keywords: ["稀疏视图CT重建", "Transformer", "基础模型", "解耦反投影", "零样本泛化"]
innovations: ["提出解耦反投影空间的Transformer架构，实现任意投影数和角度的统一重建", "几何注意力机制将物理先验编码到自注意力中", "局部patch token化实现分辨率无关的重建"]
benchmarks: ["LoDoPaB-CT", "KiTS23", "CTSpine1K", "ImageNet", "FFHQ"]
---

# 论文速读：TomoTransformer-Towards-a-Foundation-Model-for-CT-Reconstruction

## 一句话总结
本文提出了TomoTransformer，一个基于Transformer的CT重建基础模型，通过在解耦反投影空间中将每个局部滤波投影视为独立token，实现了对任意数量投影、任意角度配置和任意探测器尺寸的统一重建，无需重新训练即可处理稀疏视图CT重构任务。

## 研究问题与动机
- 现有监督深度学习方法通常需要针对特定采集几何（固定投影数和角度范围）进行训练，无法泛化到不同协议
- 现有模型设计依赖于固定的图像和探测器尺寸，限制了实际部署灵活性
- 模型泛化能力有限，通常在单一窄数据分布上训练（如单一解剖区域），难以跨分布泛化
- 最近尝试如ViewTrans虽使用Transformer，但仍受限于固定探测器尺寸和图像分辨率

## 核心贡献（创新点）
- 提出TomoTransformer架构，同时摆脱对投影数量和角度配置以及探测器尺寸的依赖，消除了现有监督方法的协议依赖性
- 在解耦空间中 formulate 了sinogram补全问题，使视角插值在几何上更自然，产生了灵活可扩展的架构
- 在大规模医学CT解剖和自然图像数据集上训练，证明单一权重集可跨解剖结构、材料和分辨率泛化，包括零样本迁移到真实实验数据
- 在固定稀疏视图重建任务上匹配或超越针对每个稀疏级别单独训练的协议专用基线模型
- 显著优于同类型多用途基线ViewTrans，并在零样本去噪和真实实验数据上展示实用价值

## 方法详解
- **解耦反投影空间**：将滤波反投影每个视角的贡献隔离，形成张量 $\mathbf{B} = [\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_R] \in \mathbb{R}^{N \times N \times R}$，其中每个视角$\mathbf{b}_r$在相同$N \times N$图像网格上共享坐标系
- **局部token化**：从完整反投影中提取$P \times P$局部块（$P=32$像素）作为输入token，通过双线性插值沿对角线采样$K=\lceil P\sqrt{2}\rceil=46$个点形成1D特征向量
- **几何tokenizer和decoder**：几何tokenizer将局部反投影块映射到$D=512$维嵌入，几何decoder从预测嵌入恢复$P \times P$输出块
- **位置编码**：每个角度$\theta_r$映射到傅里叶位置嵌入$\{\sin(k\theta_r), \cos(k\theta_r)\}_{k=1}^{32}$，支持任意连续角度
- **几何注意力机制(GAM)**：在注意力logits中加入成对角度差$\Delta_{rr'}=\theta_r-\theta_r'$的可学习函数，编码视角间的相对几何关系
- **编码器-解码器架构**：遵循MAE设计，编码器仅处理观测token，解码器接收编码器输出和可学习的查询token，两者均由4个transformer块组成，宽度$D=512$，8个注意力头
- **损失函数**：采用$\ell_1$距离，仅在未观测目标视角上计算：$\mathcal{L} = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}}\|\hat{\mathbf{t}}_q - \mathbf{t}_q\|_1$

## 实验与结果
- **训练数据集**：约65万2D切片，包括57万CT切片（来自LoDoPaB-CT、CTSpine1K、KiTS23等）和8万自然图像（FFHQ、ImageNet）
- **固定稀疏视图重建**（64→256视图）：在8个数据集上的平均PSNR达39.66 dB，优于所有基线（NAFNet 39.37 dB，Restormer 39.47 dB），SSIM达0.931
- **可变稀疏视图重建**：在LoDoPaB-CT上，128→256视图时PSNR达42.40 dB / SSIM 0.959，远超ViewTrans的37.75 dB / 0.914；64→256视图时达37.68 dB / 0.910，较ViewTrans的33.07 dB / 0.832有显著提升
- **零样本去噪**：在无噪声训练的模型上直接应用于高斯噪声(45 dB)和泊松噪声数据，无需微调即实现有效去噪
- **真实实验数据**：在瑞士光源SLS采集的纳米级脑组织PXCT数据（768×768像素，448轴向切片）上零样本测试，128→256视图达到35.8 dB / 0.837，128→512视图达到38.8 dB / 0.916，显著优于协议专用基线

## 相关工作脉络
- **CNN基线方法**：U-Net、DRUNet、NAFNet等图像域后处理方法，均为协议专用且依赖固定分辨率
- **Transformer方法**：CTformer、Restormer等，虽具有更强建模能力但仍受限于固定采集协议
- **ViewTrans**：最近提出的多用途sinogram修复Transformer，但仅支持固定探测器尺寸
- **扩散模型方法**：如DPSS等生成先验方法，虽理论可适配任意协议但推理成本高且需特定分布先验
- **物理启发重建**：如Learned Primal-Dual等方法将Radon算子嵌入网络，但仍需针对特定协议训练

## 局限性与未来方向
- 论文明确提到未来工作将探索将该架构扩展到其他断层成像模态，包括锥束CT
- 当前模型基于平行束几何设计，对实际医学CT中常见的锥束几何需要适配
- 在极稀疏视图（如<32视图）下的性能未充分评估
- 训练依赖大量合成数据，真实测量数据的域偏移可能影响性能
- 零样本去噪能力虽强但非显式优化，可能在某些噪声分布下表现不如专门去噪模型

## 研究启发与可借鉴点
- **解耦表示学习**：将投影几何与局部结构分离的思想可迁移到其他inverse problem，如MRI重建、超声成像等
- **几何注意力机制**：将物理先验（角度关系）直接编码到注意力中的方法值得在其他视觉任务中探索
- **局部patch处理策略**：固定尺寸patch处理任意分辨率图像的技巧可应用于大规模遥感、病理图像等场景
- **多模态预训练策略**：结合医学图像与自然图像进行预训练以提升泛化能力的思路可推广到其他医学AI任务
- **零样本泛化评估**：在真实实验数据上的零样本测试方案为模型实用性评估提供了示范

## 关键术语表
**Filtered Back-Projection (FBP)**：计算机断层扫描中经典的解析重建方法，通过滤波和反投影将投影数据转换为图像
**Disentangled Back-projection Space**：将每个视角的反投影结果隔离存储在同一空间坐标系的表示方式
**Geometric Attention Mechanism (GAM)**：将视角间角度差编码到注意力logits中的机制，显式利用投影几何信息
**Sinogram**：CT投影数据的二维表示，横轴为探测器位置，纵轴为投影角度
**Ptychographic X-ray Computed Tomography (PXCT)**：一种基于相干衍射成像的高分辨率X射线断层扫描技术
**Zero-shot Generalization**：模型在未见过的新分布或任务上直接应用而无需微调的能力

## 可复现要素
- **数据集**：训练使用LoDoPaB-CT、CTSpine1K、KiTS23、MSD、COLONOG、HNSCC、FFHQ、ImageNet等公开数据集
- **代码开源**：论文未明确提及代码开源情况
- **权重开源**：论文未明确提及模型权重开源情况
- **关键超参**：patch大小$P=32$像素，embedding维度$D=512$，注意力头数$H=8$，傅里叶频率数$F=32$，训练迭代$10^6$次，learning rate $3\times10^{-4}$ cosine decay，AdamW优化器
