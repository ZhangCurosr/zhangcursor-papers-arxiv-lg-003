---
title: "WEIGHTS-READ-AND-WRITE-FEATURES-SCALABLEPARAMETER-DECOMPOSIT"
source: https://arxiv.org/pdf/2609.37731v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:53:12"
field: " mechanistic interpretability of LLMs"
keywords: ["mechanistic interpretability", "parameter decomposition", "sparse autoencoder", "weight editing", "activation grounding", "IOI circuit"]
innovations: ["提出激活锚定参数分解 ASPD，用共享稀疏坐标统一建模激活特征与权重门控，避免 O(F×C) 成对因果干预", "引入内部重建损失 L_internal，在目标权重矩阵输出处提供局部监督，显著提升大模型参数分解的可解释性与编辑定位性", "构建读-写交互度量并组装参数级机制电路，在 IOI 电路上无监督还原 induction/duplicate-name/S-inhibition/name-mover 等已知机制"]
benchmarks: ["GPT-2 small MLP_in layer 0", "Gemma-2-2B MLP_out layer 13", "Qwen-3-8B Attention O layer 17"]
---

# 论文速读：WEIGHTS READ AND WRITE FEATURES: SCALABLE PARAMETER DECOMPOSITION GROUNDED IN ACTIVATION SPACE

## 一句话总结
论文提出 Activation-Supported Parameter Decomposition (ASPD)，一种联合分解激活空间与参数空间的机制可解释性方法：通过共享稀疏表示将权重分量与激活特征对齐，并以内部重建作为局部监督信号，使权重分量在语义上可读可写、可因果编辑；在 GPT-2 Small、Gemma-2-2B、Qwen-3-8B 上均显著优于 VPD 等基线，并在 IOI 电路中还原出诱导头、重复名检测、S-inhibition 等参数级机制。

## 研究问题与动机
- **激活空间与参数空间割裂**：现有可解释方法大多只研究一方——SAE 等激活分解揭示稀疏特征，但不说明哪些权重"读出/写入"这些特征；VPD 等参数分解能把权重拆成若干分量，但各分量缺乏激活语义锚定。
- **参数分解的非唯一性与不可扩展性**：同一权重矩阵允许多种秩-1 分解，仅靠权重本身无法确定哪些分量在真实激活分布上被实际启用；现有无监督参数机制发现方法难以扩展到十亿/百亿级预训练模型。
- **缺乏局部学习信号**：VPD 等通过最终输出的消融误差传播来监督内部分量，信号在深层网络中衰减严重；若能在目标权重矩阵的输出处直接提供重建信号，则可避免跨层传播导致的监督稀释。
- **权重分量之间难以构成机制电路**：单个分量的读/写语义若不被明确建模，就无法跨层、跨头拼接成有因果含义的参数级电路。

## 核心贡献（创新点）
- **激活锚定参数分解（Activation-grounded parameter decomposition）**：首次将权重分量用其所"读取/写入"的激活特征来刻画，并以共享稀疏坐标替代昂贵的成对因果干预，使参数分解具备语义锚定。
- **ASPD 联合学习与内部重建监督**：提出 ASPD 框架，联合学习激活稀疏特征与权重秩-1 分量；内部重建损失要求分量在目标权重处直接复现变换，避免 VPD 式的端到端输出监督。
- **读-写交互构成参数级机制电路**：基于 `Interact(c₁,c₂)=E[g_{c₂}·e_{c₁}]·⟨u_{c₁},v_{c₂}⟩` 等形式，把跨头/跨层的分量连接成参数电路；在 IOI 电路中无监督还原出 induction、duplicate-name、S-inhibition、name-mover 等已知机制及其跨头传递关系。
- **在 Qwen-3-8B 上实现可扩展的参数分解**：在 2B tokens、平均 L₀=32 设置下，于 8B 模型 Attention O 矩阵上完成可解释、低冗余且可因果编辑的参数分解。

## 方法详解
- **权重分量与读-写语义**：对权重矩阵 W∈R^{d_out×d_in}，用 C 个秩-1 分量 P_c=u_c v_c^⊤ 近似；引入因果重要性门控 g_{t,c}(X)，有效贡献 e_{t,c}=g_{t,c}(v_c^⊤x_t)，重建输出 ŷ_t=Σ_c e_{t,c}u_c。其中 v_c 决定读取方向、g 决定何时激活、u_c 决定写入下游的方向。
- **理想激活锚定目标（非直接优化）**：定义写效应 A^{write}_{i,c}=a_i(r_t)−a_i(r_t|g_{t,c}→0) 与读效应 A^{read}_{i,c}=g_{t,c}(X)−g_{t,c}(X|a_i(r_t)→0)，并设置 L_ground 鼓励每个分量只与少量特征存在强因果关联。由于 O(F×C) 成对干预不可行，改由结构代理实现。
- **共享稀疏表示的结构代理**：令 C=F，学习共享 sparse encoder g^s:R^{T×d_act}→R^{T×C} 与激活解码器 d，并设 a(R)=g^s(R)、g_{t,c}(X)=φ(g^s_{t,c}(R))。同一坐标既定义第 c 个激活特征、又门控第 c 个权重分量，强制分量只能读/写到激活空间中的某一位点。
- **ASPD 联合优化目标**：
  - 内部重建损失 L_internal=E[(1/T)Σ_t||y_t−ŷ_t||₂²]，直接在目标权重矩阵输出处约束分量复现变换。
  - 激活重建损失 L_act=E[(1/T)||R−d(g^s(R))||₂²]，确保共享坐标构成激活空间的稀疏分解。
  - 稀疏损失 L_sparse(g^s) 鼓励门控稀疏（实现中用 BatchTopK，目标 L₀=32）。
  - 总目标 L_ASPD=L_internal+λ_act·L_act+λ_sparse·L_sparse，实际中 λ_act=λ_internal=1、辅助 loss 系数 0.03125。
- **读-写交互与参数级电路**：对顺序组件（OV、MLP_in→MLP_out、跨层残差），评分 Interact(c₁,c₂)=E_{X,t}[g_{t,c₂}·e_{t,c₁}]·⟨u_{c₁},v_{c₂}⟩；对 QK 自注意力头内交互按头分解，用 ⟨u^h_{c₁},u^h_{c₂}⟩ 度量共线性。据此可排序跨组件依赖关系并组装机制电路。
- **实现细节**：g^s 用 per-token BatchTopK；φ 取指示函数 I[g>0]；L_act 中加入 Matryoshka loss（前缀比 [0.0625,0.0625,0.125,0.25,0.5]）；用 auxiliary loss（coeff=0.03125）复活死分量；共享门控在 residual-stream-pre（Q/K/V、MLP 输入前）或 residual-stream-mid（O 矩阵后）处取值。

## 实验与结果
- **评测设置**：三模型×单矩阵——GPT-2 small 的 MLP_in（layer 0）、Gemma-2-2B 的 MLP_out（layer 13）、Qwen-3-8B 的 Attention O（layer 17）；均在 2B tokens、L₀=32 下训练，分别得 24576/36864/36864 个分量。独立训练输出 SAE 用于 Meaning Localization 与 Editing 评估。
- **可解释性与多样性（Table 1）**：ASPD 在 GPT-2 上 Intruder=0.68±0.04、Sim=0.01±0.00；Gemma-2-2B 上 0.62±0.03 / 0.03±0.00；Qwen-3-8B 上 0.57±0.04 / 0.03±0.00。VPD 在大模型上接近随机（Interp≈0.20–0.22、Sim 偏高）。
- **Meaning Localization（Table 2）**：ASPD 在 GPT-2  Matching=1.02±0.14、Gemma 0.30±0.11、Qwen 0.61±0.14，显著优于 VPD（大模型上 ≤0.17 或为负）。
- **因果权重编辑定位（Table 3）**：ASPD 在 Qwen-3-8B 单目标 ratio=6.7±0.9、localization=0.053±0.006；多目标 ratio=11.1±0.6、localization=0.044±0.002。VPD 在相同设置下 ratio≈0.9/0.8，几乎等于随机。
- **消融**：在 VPD 中加入 L_internal 后，三项指标均有改善，但仍未追上 ASPD；添加 VPD 的 L_param/L_ablate 对 PD Transcoder 普遍有害，对 VPD 效果混杂，故 ASPD 未引入。
- **机制案例**：ASPD 在 GPT-2 small 全 72 个矩阵上联合分解（共 442,368 个分量），无监督还原 IOI 电路：H5.5 的 164Q/5897K 识别"名 + 其后词"、228O 输出诱导信号；H3.0 的 224O 识别重复名；H8.6 的 2861V/101V 接收上游信号并强化 S-inhibition；H9.6/H11.10 的 QK 与 OV 组件分别实现 name-mover / negative name-mover。直接消融 QK 对可显著抑制对应注意力模式。

## 相关工作脉络
- **SAE/Transcoder 等激活空间分解**（Bricken et al., 2023; Lieberum et al., 2024; Dunefsky et al., 2024）：刻画激活表征，但不揭示实现这些表征的权重机制；本文把此类成果作为激活锚定与损失设计的基石。
- **VPD**（Bushnaq et al., 2025, 2026）：最早尝试无监督参数分解的代表方法，依靠最终输出层消融误差反传监督内部分量；在 4 层 toy model 有效，但在更大模型上退化为不可解释、编辑性差；本文以内部重建 + 激活锚定解决该可扩展性瓶颈。
- **Sparse weight decomposition / circuit extraction**（Yan et al., 2026; Braun et al., 2025; Chrisman et al., 2025; Vigouroux & Sharkey, 2026）：同属参数分解路线，但缺乏激活语义接地；本文强调"权重为何被使用"需要激活特征约束。
- **Knowledge editing / weight steering / vocabulary-based 分析**（Meng et al., 2022; Geva et al., 2021; Fierro & Roger, 2026）：侧重修改或定位已有参数效应；本文侧重从权重出发逆向发现底层机制。
- **Feature circuits 与 activation-space 电路**（Marks et al., 2024; Lindsey et al., 2025）：在激活空间定位功能特征并构建电路；本文把视角下沉到权重层面，实现"特征–权重–下游特征"的闭环链路。
- **SAE-guided parameter editing / cross-space interaction**（Gur-Arieh et al., 2025; Chen et al., 2025）：证明两空间可互用，但未给出机制解释；本文进一步使权重分量在结构上与激活特征绑定，达到可解释的因果编辑。

## 局限性与未来方向
- 当前 IOI 案例揭示的某些分量语义偏粗粒度（如 H5.5 学到"名字之后的 token"而非"John 之后的 token"），更细粒度可能需要更多分量或分层分解。
- 部分具有相似激活模式的并行分量（如多个"follow a name"组件）尚未充分区分，可能涉及 feature splitting 或更深层语义。
- 未在 ABC prompt 等变体上系统检验电路稳健性；跨模型、跨任务的通用性仍需扩展验证。
- 未结合 Parameter Diffing 方向：文中提示可将"激活 diffing + 参数分解"结合用于分析微调前后机制变化，留作未来工作。
- L_param / L_ablate 等 VPD 模块化损失虽理论上有助于编辑正交性，但实验中表现混杂；如何在不破坏性能前提下引入仍需探索。
- 共享门控目前仅在残差流某一点取值，未系统比较不同锚定位（如 layer norm 前后、不同层之间）对读/写语义的影响。

## 研究启发与可借鉴点
- **内部重建信号替代端到端输出监督**：对任意深层模块做参数分解时，在目标模块输出处加 L2 重建损失可显著缓解跨层信号衰减，这一思路可直接迁移到 MLP 注意力中间层分解等任务。
- **共享稀疏坐标绑定激活特征与权重门控**：用同一坐标同时决定"特征是什么"与"哪个权重被使用"，天然实现语义锚定且避免了 O(F×C) 因果干预；可推广至视觉 Transformer、多模态模型等更宽泛的参数分解场景。
- **读-写交互度量 Interact=E[g·e]·⟨u,v⟩**：把组件对的共激活与几何对齐分离成两个因子，能更好地反映真实因果通量；可作为通用"参数电路边权重"定义，用于发现跨层/跨模块的潜在通路。
- **Meaning Localization 评测范式**：通过独立 SAE + attribution patching + LLM judge 评估"分量写出的信息与目标特征是否同义"，弥补了传统 Intruder/Diversity 指标的不足；可成为后续参数分解论文的标准评测环节。
- **与团队方向的结合机会**：本工作把 SAE 可解释性与参数编辑打通，适合延伸至 (1) 微调前后的参数级 diffing；(2) 多模态模型中视觉/文本通道的跨模态权重通路追踪；(3) 面向推理过程（chain-of-thought）的阶段级参数分解，定位每一步的机制变化。

## 关键术语表
- **Activation-Supported Parameter Decomposition (ASPD)**：本文提出的联合激活与参数空间分解框架，用共享稀疏表示把权重分量锚定到激活特征。
- **Read-write component**：将每个秩-1 权重分量理解为"在某条件下读取某方向信息、并写入另一方向"的上下文依赖机制，三元组为 (v_c, g_c, u_c)。
- **Causal importance gate g_{t,c}**：标量门控，决定分量 c 在 token t 是否参与计算；在 ASPD 中与共享稀疏坐标绑定。
- **Internal reconstruction loss L_internal**：约束分量在目标权重矩阵的输出端直接复现原矩阵变换的局部监督损失。
- **Meaning Localization / Matching score**：用独立 SAE 和 attribution patching 估计分量对下游特征的因果影响，并由 LLM judge 打分衡量分量"写入"信息与目标特征的语义一致性。
- **Weight editing localization (ratio)**：通过移除 top-k 分量后对比目标特征激活变化与非目标变化的比值，评估编辑的可定位性；ratio>1 表示优于随机编辑。
- **Interaction score Interact(c₁,c₂)**：衡量上游分量 c₁ 对下游分量 c₂ 的因果贡献，等于平均共激活乘以读写方向的余弦相似性。
- **BatchTopK sparse encoder**：每 token 取前 K 个最大坐标作为激活的稀疏函数，用于 ASPD 的门控实现，确保精确控制 L₀ 稀疏度。

## 可复现要素
- **数据集**：GPT-2 small 用 apollo-research/Skylion007-openwebtext-tokenizer-gpt2；Gemma-2-2B 与 Qwen-3-8B 用 monology/pile-uncopyrighted（论文 Table 4）；评估编辑/localization 使用 10⁶ 随机抽样 token。
- **代码/权重开源状态**：论文正文及附录未明确给出 GitHub/模型权重开源链接（仅引用了 VPD/BatchTopK 等开源基线实现），建议在 arXiv 提交页或补充材料中确认；本笔记依原文仅如实记录"论文未明确声明开源"。
- **关键超参**：C=F=24576（GPT-2s）/36864（Gemma-2-2B、Qwen-3-8B）；训练 token 数 2B；平均 L₀=32（BatchTopK k=32）；λ_internal=1、λ_act=1、λ_aux=0.03125；Adam lr=3×10⁻⁴（β₁=0.9, β₂=0.99）；Matryoshka 前缀比 [0.0625, 0.0625, 0.125, 0.25, 0.5]。
