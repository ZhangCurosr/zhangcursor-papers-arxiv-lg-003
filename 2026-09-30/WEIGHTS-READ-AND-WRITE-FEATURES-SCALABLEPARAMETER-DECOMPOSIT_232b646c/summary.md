---
title: "WEIGHTS-READ-AND-WRITE-FEATURES-SCALABLEPARAMETER-DECOMPOSIT"
source: https://arxiv.org/pdf/2609.37731v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:52:54"
field: " mechanistic interpretability"
keywords: ["mechanistic interpretability", "parameter decomposition", "sparse autoencoder", "weight editing", "IOI circuit", "activation grounding"]
innovations: ["提出ASPD联合激活与参数空间分解，用共享稀疏表示实现权重组件的激活接地", "设计内部重建损失提供局部学习信号，使参数分解可扩展至Qwen-3-8B等大规模模型", "构建读-写交互评分框架，将权重组件组合为参数级机制电路并恢复IOI电路"]
benchmarks: ["GPT-2 small", "Gemma-2-2B", "Qwen-3-8B"]
---

# 论文速读：WEIGHTS READ AND WRITE FEATURES: SCALABLE PARAMETER DECOMPOSITION GROUNDED IN ACTIVATION SPACE

## 一句话总结
论文提出激活支持参数分解（ASPD）方法，通过将激活空间与参数空间联合分解，使每个权重组件通过其读取或写入的激活特征获得语义接地，从而在大型预训练语言模型中实现可解释、可因果编辑的参数分解，并在 Qwen-3-8B 上验证。

## 研究问题与动机
- 现有可解释性方法分别研究激活空间和参数空间，缺乏两者之间的连接分析。
- 激活空间分解（如 SAE）揭示了可解释特征，但无法直接揭示哪些权重创建、消耗或变换这些特征。
- 现有参数分解方法虽能分解权重矩阵，但学到的权重组件未显式接地到模型操作的激活特征上，且难以扩展到大型预训练模型。
- 权重重分解本身具有多解性，需要激活信息作为约束条件，同时内部激活还提供局部学习信号，避免仅通过最终输出变化推断机制。

## 核心贡献（创新点）
- 提出激活接地参数分解框架，通过权重组件读取/写入的激活特征来表征组件，并设计 ASPD 联合学习激活特征与权重机制。
- ASPD 使用共享稀疏表示和内部重建信号替代昂贵的成对特征-组件干预，实现在 Qwen-3-8B 级别模型中的可扩展参数分解。
- 构建读-写交互框架，将权重组件组合为参数级机制电路，并在 IOI 电路恢复和语义变换追踪中展示效果。

## 方法详解
- **权重组件的读-写解释**：将权重矩阵 W 分解为 C 个秩-1 组件 P_c = u_c v_c^T，配合因果重要性函数 g_{t,c}(X) 控制组件何时参与计算；v_c 决定读取方向，u_c 决定写入方向。
- **激活接地目标**：通过消融组件对下游激活特征的影响（write 效应）或消融特征对组件激活的影响（read 效应）建立因果关联，理想目标为使每个组件与少量激活特征建立稀疏因果关联。
- **ASPD 联合学习**：设 C = F，学习共享稀疏编码器 g^s 和激活解码器 d，用同一隐坐标同时定义激活特征 c 和门控权重组件 c；损失函数包含内部重建损失 L_internal（直接重建权重矩阵输出）、激活重建损失 L_act（重建输入激活）和稀疏损失 L_sparse。
- **读-写交互评分**：组件 c1 写到 e_{t,c1} u_{c1}，组件 c2 沿 v_{c2} 读取，交互强度通过 E[g_{t,c2} e_{t,c1}] <u_{c1}, v_{c2}> 评分，用于连接跨层跨矩阵的机制。
- **工程实现**：使用 BatchTopK 编码器实现稀疏性，φ(s) = I[s > 0] 作为门控函数，接入残差流激活。

## 实验与结果
- **数据集与模型**：GPT-2 small（OpenWebText）、Gemma-2-2B（Pile）、Qwen-3-8B（Pile），每个模型训练 2B tokens。
- **基线方法**：VPD（adversarial parameter decomposition）、PD Transcoder（Transcoder 的参数分解版本）、VPD + internal（加入内部重建损失）。
- **可解释性**：Intruder 测试（LLM 法官），ASPD 在 GPT-2 small 达到 0.68±0.04，Gemma-2-2B 达到 0.62±0.03，Qwen-3-8B 达到 0.57±0.04；VPD 在大型模型上接近随机（0.22）。
- **多样性**：平均配对 Jaccard 重叠，ASPD 在三个模型上均为 0.03，显著低于 VPD 的 0.18-0.44。
- **语义定位**：Matching score，ASPD 在三个模型上均取得正值（GPT-2: 1.02，Gemma-2: 0.30，Qwen-3: 0.61）。
- **因果编辑定位**：ASPD 在 Qwen-3-8B 单目标编辑达到 6.7× 随机基线，多目标达到 11.1×；VPD 仅 0.9× 和 0.8×。
- **消融实验**：内部学习信号确实改善 VPD 可解释性但不足以超越 ASPD；VPD 的 L_param 和 L_ablate 损失对 PD Transcoder 有负面影响。
- **机制案例**：在 GPT-2 small 上完整分解 72 个权重矩阵，恢复 IOI 电路中 induction、duplicate-token、S-inhibition、name movement 等组件及其跨头交互。

## 相关工作脉络
- **Sparse Autoencoders (SAE)**：激活空间分解的基础工作，本文将其思想扩展至参数空间，用共享表示桥接两空间。
- **VPD（Adversarial Parameter Decomposition）**：现有最强参数分解方法，但仅在小模型上验证且不可解释，本文提出激活接地解决此问题。
- **Transcoder**：原为激活分解方法，本文证明其可视为参数分解的特例（PD Transcoder），并指出 ASPD 通过解耦读方向和特征编码器更灵活。
- **Feature Circuits**：激活空间因果图方法，本文在参数空间构建等价机制电路，补充了"哪些权重实现哪些计算"的视角。
- **Knowledge Editing / Weight Steering**：参数修改工作，本文的因果可编辑组件为这类任务提供更精细的编辑单元。
- **Mechanistic Interpretability of IOI**：Wang et al. (2022) 的 IOI 电路分析在激活/头级别，本文将其细化到权重组件级别。

## 局限性与未来方向
- 分解粒度有限，部分机制仍较宽泛（如 H5.5 未学到"John 之后的 token"特定信号），可能需要更多组件。
- 存在特征分裂（feature splitting）现象，多个行为相似的组件难以完全区分。
- 未在 ABC 提示词上探索 IOI 电路，可能遗漏额外机制。
- L_param 和 L_ablate 损失虽对 PD Transcoder 有害，但其保证的模块化编辑性质仍有价值，值得进一步研究。
- 参数 diffing（比较微调前后机制变化）是潜在应用方向，尚未探索。

## 研究启发与可借鉴点
- **共享稀疏表示桥接两空间**：用同一隐坐标同时定义激活特征和控制权重门控，是一种简洁有效的接地设计，可迁移到其他需要连接表示与计算的任务。
- **内部重建损失提供局部学习信号**：直接在目标权重矩阵输出上施加重建监督，避免信号经多层传播衰减，对深层大模型参数分解有普适价值。
- **读-写交互评分公式**：E[g_{c2} e_{c1}] <u_{c1}, v_{c2}> 将数据共现和几何对齐结合，是一种可复用的跨组件依赖度量，可用于其他机制发现任务。
- **Intruder 评估协议**：用 LLM 法官进行组件可解释性量化评估，配合多样性（Jaccard 重叠）和语义定位（Matching score）三项互补指标，评估体系设计严谨。
- **与团队方向结合**：若团队从事知识编辑或电路压缩，ASPD 提供的细粒度因果可编辑组件可直接作为编辑原子；参数 diffing 方向可借鉴联合分解框架。

## 关键术语表
- **ASPD（Activation-Supported Parameter Decomposition）**：激活支持参数分解，本文提出的联合激活-参数分解方法。
- **Read-Write 组件**：权重秩-1 分解组件，v_c 方向读取输入信息，u_c 方向写入输出信息，g_{t,c} 控制激活时机。
- **Causal Importance Function g_{t,c}**：因果重要性函数，指示组件 c 在 token t 是否参与计算。
- **Attribution Patching**：归因补丁技术，通过扰动单点激活测量其对目标输出的因果影响。
- **IOI Circuit**：间接对象识别电路，GPT-2 small 中完成"I and John went to the store, John gave a drink to ___"填空的经典机制电路。
- **BatchTopK**：批量 Top-K 稀疏函数，直接强制每 token 恰好 K 个特征激活。
- **L_internal**：内部重建损失，要求分解后的组件加权输出来自目标权重矩阵的实际输出。
- **Meaning Localization**：语义定位，评估权重组件的输出效应是否与独立训练的激活特征语义一致。

## 可复现要素
- **数据集**：OpenWebText（GPT-2 small）、Pile（Gemma-2-2B、Qwen-3-8B），均为公开数据集。
- **代码**：论文未提及代码开源状态，但附录提供了详细的超参和损失系数表。
- **关键超参**：C = F = 24576（GPT-2s）、36864（Gemma-2-2B、Qwen-3-8B）；平均稀疏度 L_0 = 32；学习率 3×10^{-4}；训练 2B tokens；BatchTopK k=32。
