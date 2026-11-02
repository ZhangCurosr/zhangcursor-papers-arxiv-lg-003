---
title: "Why-don-t-Graph-Transformers-attend-to-the-same-token"
source: https://arxiv.org/pdf/2610.19324v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:13:37"
field: "图深度学习"
keywords: ["Graph Transformer", "attention sink", "GRIT", "dot-product attention", "gated scoring", "molecular graphs", "over-smoothing"]
innovations: ["首次系统研究 Graph Transformer 中的 attention sink 现象并证明打分函数是决定性因素", "发现图 Transformer 与 LLM 相反的 sink-尺寸趋势（图越大 sink 越少）", "提出三个 attention 上界（对称性/局部性/打分散布）并实证其在真实分子上均不充分"]
benchmarks: ["ZINC-12k", "Peptides-func"]
---

# 论文速读：Why-don-t-Graph-Transformers-attend-to-the-same-token

## 一句话总结
本文系统研究了 Graph Transformer 在分子图上是否形成 attention sink，发现与 LLM 相反的趋势——图大小增加时 sink 反而更少；**决定 sink 形成的关键是注意力打分函数（gated vs dot-product）**，而非图结构本身的对称性或局部性约束。

## 研究问题与动机
- **LLM 的 attention sink 现象在图 Transformer 中是否存在？** LLM 中近 80% 的注意力头集中在 <BOS> 令牌以抵御 over-mixing；图 Transformer 面临类似的 over-smoothing 压力，但图中没有天然的"第一个 token"。
- **现有方法未能解释 sink 形成的根本原因。** 现有工作记录了 attention sink 在双向扩散语言模型、注意力 GNN 中的存在，但缺乏对"图中 sink 何处能形成、哪个架构组件决定其是否形成"的系统分析。
- **对称性和局部性能否阻止 sink？** 图的 permutation equivariance 要求同轨道节点获得相同注意力，但真实分子几乎无对称原子；同时注意力高度非局部，局部性边界也过松。
- **GRIT 与 GraphGPS 的 sink 率差异极大（0.09 vs 0.42），根源是什么？** 两者在 heads、PE、dropout、message passing 和打分函数上均有差异，需逐组件消融定位。

## 核心贡献（创新点）
1. **首次系统刻画 Graph Transformer 上的 attention sink 行为。** 证明了图结构层面的三个上界（对称性、局部性、打分散布），并实证表明这些边界在真实分子上均不充分。
2. **发现与 LLM 相反的尺寸趋势。** GRIT 在更大图（Peptides-func 最大四分位，均值 245 节点）上的 sink 率（0.045）远低于小图（0.238），而 LLM 中更长上下文带来更强 sink。
3. **精确定位打分函数为 sink 形成的决定性因素。** 仅将 GRIT 的 gated logit 替换为 dot-product，sink 率从 0.090 跃升至 0.680，且该效应在准确率匹配的控制实验中依然成立。
4. **揭示 sink 的功能差异：dot-product sink 近似"noop"，GRIT 的高注意力节点对预测有实质贡献。** GRIT 的 ablation ratio 为 2.06，而 dot-product 模型仅为 1.00–1.22。

## 方法详解
- **Sink 度量定义：** 节点 sink 分数 $s_j(A) = \frac{1}{n}\sum_u A_{uj}$（所有 query 的平均接收注意力）；当 $s_j > \varepsilon = 0.3$ 时视为 sink；sink rate 为共享同一 sink 节点的 head 比例最大值（峰值 sink 层平均）。
- **理论界（Proposition 3.1）：** 对任意节点 $j$ 和半径 $r$，$s_j(A) \leq \min\left\{\frac{1}{|\mathcal{O}(j)|},\; \lambda_r(A)+(1-\lambda_r(A))\frac{|B_G(j,r)|}{n},\; \frac{e^\Delta}{e^\Delta+n-1}\right\}$，分别对应对称性、局部性和打分散布上界。
- **GRIT vs dot-product 打分函数对比：**
  - GRIT：$z_{uj}^{\text{GRIT}} = a^\top \min\{5,\text{ReLU}(\text{ssq}((q_u+k_j)\odot w_{uj})+b_{uj}))\}$，每坐标 clip 到 [0,5]，logit 对节点范数不敏感。
  - Dot-product：$z_{uj}^{\text{dot}} = q_u^\top k_j / \sqrt{h}$，随被注意力节点范数线性增长，无上界。
- **消融策略（逐一组件替换）：** 替换 logit（GRIT-dotlogit）、移除消息传递（GraphGPS-none）、移除图结构（GRIT-blind）、添加虚拟节点（GraphGPS + virtual node）、匹配准确率的对照实验等。
- **范数干预实验：** 在层 $\ell \in \{3,5,8\}$ 将某节点输入放大 $\rho$ 倍，统计成为 sink 的 head 比例（Table 4）。

## 实验与结果
- **数据集：** ZINC-12k（回归，~23 节点）；Peptides-func（分类，四分位均值 56–245 节点），每四分位训练 2,023 张图，batch size 16。
- **GRIT（ZINC）：** sink rate = **0.090 ± 0.028**，Test MAE = 0.064 ± 0.009，ablation ratio = 2.06。
- **GraphGPS（ZINC）：** sink rate = **0.415 ± 0.194**，Test MAE = 0.072 ± 0.005，ablation ratio ≈ 1.00。
- **GRIT-dotlogit（仅改打分）：** sink rate 飙升至 **0.680 ± 0.098**（高于 GraphGPS），验证打分函数是主因。
- **Peptides-func 尺寸趋势：** GRIT sink rate 从最小四分位 0.238 降至最大四分位 0.045（反向于 LLM）。
- **范数干预（Table 4）：** 32× 放大时，GRIT 仅 1.3% head 形成 sink，GRIT-noclamp 为 54.0%，GRIT-dotlogit 为 66.9%。
- **虚拟节点：** GraphGPS + virtual node sink rate = 0.559，但 sink 主要落在原子而非虚拟节点；GRIT + virtual node 仍保持低 sink rate（0.095）。
- **最强结果对比：** GRIT 在 ZINC 达到 Test MAE 0.064（接近原文发表结果），同时 sink rate 极低（0.090），是"高性能+低 sink"的典型。

## 相关工作脉络
1. **Xiao et al. (2024) — Attention Sinks in LLMs。** 发现 LLM 中 <BOS> 作为 sink 保护深层模型免于 over-mixing；本文将其概念迁移至图 Transformer，发现图场景下规律不同。
2. **Gu et al. (2025) / Barbero et al. (2025) — Sink 量化与成因。** 提出 sink metric 和阈值定义；本文在其框架下扩展评测对象至图 Transformer，并分离出打分函数这一新变量。
3. **Ma et al. (2023) — GRIT 原始论文。** 提出无消息传递的 Graph Transformer，使用 RRWP 和 gated 打分；本文发现其低 sink 特性源于打分函数设计。
4. **Rampášek et al. (2022) — GraphGPS。** 混合消息传递与 dot-product 注意力的图 Transformer；本文发现其高 sink 率源于 dot-product 打分而非消息传递缺失。
5. **Bini et al. (2024) — 注意力 GNN 中的 massive activations。** 记录图注意力的极端激活现象；本文从 sink 视角重新解释该现象，归因于打分函数。
6. **Liu et al. (2026)（同期工作）— 图缺乏自然 sink。** 论证图缺少类似 <BOS> 的天然 sink 候选；本文进一步指出即使加入虚拟节点，gated 打分仍阻止 sink 形成。

## 局限性与未来方向
- **仅研究单一架构（GRIT）对抗 dot-product 基线**，未覆盖 EGT、Graphormer、SAN 等其他 Graph Transformer 变体。
- **仅限分子图**，非分子图（如社交网络、知识图谱）上的 sink 行为未知。
- **模型深度仅 10 层**，较浅；Larger-scale 和更深网络下的行为未验证。
- **虚拟节点相关结果存在 seed 依赖性**，稳定性待验证。
- **测量的是"接收注意力"而非实际"混合程度"**（Appendix A.2 已说明 sink score ≠ mixing coefficient），sink 是否真正起防护作用尚待进一步验证。
- **未来方向：** 扩展至更多架构和非分子图；区分 sink 与 over-smoothing 的关系；探索 gated scoring 在其他图学习任务中的泛化性。

## 研究启发与可借鉴点
1. **"单组件消融"的研究范式值得借鉴：** 本文通过逐一替换组件（logit、MPNN、PE 等）精准定位因果因素，避免了多变量混淆，为后续研究提供了可复用的实验设计模板。
2. **虚拟节点作为"可控 sink 候选"的引入策略：** 模拟 LLM 的 <BOS> token，为图 Transformer 中 sink 形成机制的研究提供了一个干净的控制变量工具。
3. **打分函数的非线性设计具有正则化/防 sink 的附带收益：** GRIT 的 clip + gated 机制天然限制了 logit 对节点范数的敏感度，这一设计洞察可迁移至其他需要抑制注意力过度集中的场景。
4. **"sink rate vs 任务性能"的解耦分析：** 本文证明 sink 的形成与模型 accuracy 无关（匹配准确率的对照实验中 sink 率仍差异显著），提示 sink 可能是架构属性而非优化缺陷，这对后续研究的设计有指导意义。
5. **结合本团队方向的创新机会：** 若团队研究图 Transformer 的可解释性或稳定性，可将 gated scoring 机制引入现有模型以抑制病态注意力集中；或基于本文的 three bounds 框架诊断其他图学习模型中的注意力异常。

## 关键术语表
- **Attention Sink（注意力汇聚）：** 某一 token/node 收到异常大量注意力的现象，在 LLM 中通常集中于 <BOS>，在图中表现为某节点 receive attention 远超均匀水平。
- **GRIT（Graph Inductive Biases in Transformers）：** Ma et al. (2023) 提出的无消息传递的 Graph Transformer，使用 RRWP 初始化和 gated 打分函数。
- **Dot-product Attention（点积注意力）：** 标准 Transformer 打分方式 $q^\top k/\sqrt{h}$，logit 随节点范数线性增长，无上界。
- **Gated Logit（门控 logit）：** GRIT 的打分函数，通过 pair representation 门控 query+key 和，经 signed square root 和 clip 后读出，对节点范数不敏感。
- **Sink Rate（汇聚率）：** 峰值 sink 层中共享同一 sink 节点的 head 比例，是衡量模型注意力集中程度的核心指标。
- **Permutation Equivariance（置换等变性）：** 图模型的固有属性，要求节点排列后输出同步排列，导致同轨道节点获得相同注意力。
- **Virtual Node（虚拟节点）：** 与所有真实节点相连的额外节点，作为图中类似 <BOS> 的 sink 候选进行实验。
- **Over-mixing / Over-smoothing（过度混合/平滑）：** 深层图神经网络中节点表示趋于一致的现象，attention sink 被认为可能延缓此过程。

## 可复现要素
- **数据集：** ZINC-12k（公开）、Peptides-func（公开，来自 Long Range Graph Benchmark）。
- **代码/权重：** 论文未明确声明开源仓库，但使用了 GRIT 和 GraphGPS 的官方实现进行修改；超参数见 Appendix B Table 2。
- **关键超参：** GRIT-ZINC：dim=64, heads=8, layers=10, LR=1e-3, epochs=2000, batch=32, PE=RRWP-21；GraphGPS-ZINC：dim=64, heads=4, layers=10, LR=1e-3, epochs=1500, batch=32, PE=RWSE-28；优化器 AdamW + cosine annealing + gradient clipping。
- **训练环境：** 单 GPU；Peptides-func 图截断至 300 节点。
