---
title: "Utility-Aware-Generative-Recommendation-Utility-Enriched-Sem"
source: https://arxiv.org/pdf/2611.02981v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:16:45"
---

# 论文速读：Utility-Aware-Generative-Recommendation-Utility-Enriched-Sem

## 一句话总结
本文提出 UAGR，通过在生成式推荐的 SID 构建阶段以位分配方式静态注入离散化效用信号，并在束搜索解码阶段融合多行为效用预测分，解决传统 SID 仅表征语义相似性而忽视曝光后效用、导致高价值候选在解码中途被提前剪枝的问题。

## 研究问题与动机
1. **SID 构建仅依赖语义相似性**：传统残差量化生成的语义标识符对“语义相近但曝光后效用差异大”的物品区分度不足。
2. **基于 likelihood 的束搜索易误剪高效用路径**：有限宽度的 beam search 会在中间前缀阶段不可逆地剪枝，高效用候选一旦丢失无法在后续重排序或训练对齐中恢复。
3. **协作信号注入的时效性与结构扰动问题**：现有融入协同信号的 SID 方法会改变已学习的语义结构，且协同信号时变性导致码本快速过时，难以支撑工业级长期部署。
4. **训练目标与下游多目标效用脱节**：标准 next-token prediction 仅拟合历史交互 likelihood，未显式建模长观看、完播、点赞、分享等 heterogeneous post-exposure outcomes。

## 核心贡献（创新点）
1. **提出 UE-SID（Utility-Enriched Semantic Codes）**：在保留前 L-1 层分层语义结构的前提下，将最后一层部分 bit 释放并编码离散化后的物品级效用，不增加 SID 长度与词表大小。*与已有工作的本质区别在于：以静态位分配方式注入效用信号，避免端到端联合训练量化器带来的语义结构扰动与协作信号时效性衰减。*
2. **提出 UGD（Utility-Guided Decoding）**：从生成骨干的中间 decoder hidden states 提取共享低维表征，通过多行为专用头预测曝光反馈，并将归一化效用分与累计 likelihood 融合指导束搜索剪枝。*与已有工作的本质区别在于：效用信号仅在推理期用于 beam pruning，且通过 stop-gradient 完全隔离训练梯度，不干扰骨干网络的原始生成能力。*
3. **设计双时间尺度在线更新机制**：小时级局部效用统计刷新 + 周级全局桶边界校准，支持增量更新而不需重训练语义量化器。*与已有工作的本质区别在于：兼顾工业部署对码本新鲜度与稳定性的双重需求，实现“训练免扰动、线上可热更”。*
4. **提供轻量、可插拔的多目标效用对齐范式**：支持灵活配置不同行为权重，推理端无额外骨干前向计算开销。*与已有工作的本质区别在于：在“模型轻量化、工程低风险、业务可配置”之间取得平衡，而非依赖重型序列级对齐或事后重排。*

## 方法详解
1. **UE-SID 构建（仅改造第 L 层）**
   - 前 $L{-}1$ 层保持残差 K-means 不变，第 $L$ 层替换为 LFQ（Look-Up-Free Quantization）。
   - 残差投影与二值化：$\mathbf{z}_i = f_{\text{proj}}(\mathbf{r}_i^{(L-1)})$，逐维 $\widehat{z}_{i,j} = \text{sgn}(z_{i,j})$ 映射为 0/1 码。
   - LFQ 训练损失：$\mathcal{L}_{\text{LFQ}} = \mathcal{L}_{\text{com}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$，其中 $\mathcal{L}_{\text{com}}$ 为 commitment loss，$\mathcal{L}_{\text{ent}}$ 为熵正则以鼓励 code 均匀利用、缓解 code collapse。
   - 位分配策略：穷举剔除 3 个对 SID 区分度贡献最弱的 bit，将释放的 $B_u{=}3$ 位用于编码 CTR 离散化后的效用桶索引（8 个 bucket），剩余 $B_s{=}10$ 位打包为语义码，最终 token 为 $\text{Concat}(c_{s,i}, c_{\text{u},i})$。

2. **UGD 效用估计与束搜索融合**
   - 隐藏状态提取与投影：$\mathbf{h}^{(l)} = f_\theta(\mathcal{H}_u, \mathbf{s}^{(<l)})$，$\mathbf{g}^{(l)} = P_\psi(\text{sg}(\mathbf
