---
title: "RELMEM-LEARNING-RECURRENT-MEMORY-FORLONGITUDINAL-EHR-MODELIN"
source: https://arxiv.org/pdf/2609.37587v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:48:03"
field: "纵向医疗记录建模"
keywords: ["longitudinal EHR", "recurrent memory", "context compression", "large language models", "attention alignment", "curriculum learning", "medical AI"]
innovations: ["固定容量循环记忆框架，通过多粒度优化（中间注意力对齐+最终预测监督）学习可替换的逐层KV记忆", "仅在memory token位置激活的token-conditioned低秩适配器，实现极轻量参数更新", "课程学习策略支持长程历史稳定更新，从短序列逐步过渡到完整历史"]
benchmarks: ["MIMIC-IV medication prediction", "MIMIC-IV next-visit diagnosis prediction"]
---

# 论文速读：RELMEM: LEARNING RECURRENT MEMORY FOR LONGITUDINAL EHR MODELING

## 一句话总结
论文提出 ReLMem 框架，通过学习固定容量的循环患者记忆，使冻结的 LLM 能够高效处理纵向电子健康记录（EHR）；在多粒度优化策略下，药物预测任务仅用 2.9% 的历史存储量即逼近全历史基线性能，同时显著降低 GPU 内存与推理延迟。

## 研究问题与动机
1. **纵向 EHR 建模中的历史膨胀问题**：患者每次就诊都会累积新的临床记录，直接使用 LLM 处理完整历史会导致注意力计算呈二次方增长、KV cache 线性膨胀，难以在实际临床系统中部署。
2. **现有压缩方法的局限**：传统上下文压缩方法（如 KV 剪枝、文本摘要）在纵向场景中面临两难——要么每次重新压缩需重读早期记录，要么逐条压缩后累积仍会导致上下文过载。
3. **循环更新中的信息丢失风险**：固定容量下的逐次更新可能逐层累积信息损失，而仅依赖最终预测监督无法保证中间记忆状态保留关键历史证据。
4. **资源效率与预测性能的权衡需求**：临床场景要求模型在有限计算预算下持续整合新就诊信息，同时保留足够历史上下文以支持下游预测。

## 核心贡献（创新点）
1. **固定容量循环记忆框架 ReLMem**：为冻结的任务适配 LLM 配备轻量级压缩适配器，在每个就诊节点将新记录与上一轮记忆整合，输出等容量更新状态，无需重读早期记录。与已有工作本质区别在于直接替换而非累积压缩表示，实现真正的固定容量循环更新。

2. **多粒度联合优化策略**：设计中间注意力对齐损失（$L_{\text{inter}}$）与最终预测监督损失（$L_{\text{pred}}$）的联合目标。与仅依赖最终答案监督的方法不同，该方法通过对齐压缩记忆与完整历史的注意力输出，显式约束中间状态的历史保留能力。

3. **分层 KV 状态的循环更新机制**：将记忆表示为逐层 KV 对（而非单嵌入向量），利用 token-conditioned 低秩适配器仅在 memory token 位置启用训练参数，实现精细化的历史信息压缩。与 RMT 等基于最后一层 hidden state 的方法相比，保留了多层注意力信息。

4. **课程学习策略支持长程历史建模**：从短序列开始训练，逐步引入更长就诊历史的样本，帮助模型学习跨多轮更新的稳定性。相比一次性暴露所有长度，该策略显著提升了长历史场景下的性能。

5. **系统性效率-性能评估**：在药物预测任务上，ReLMem 将平均历史存储降低 97.1%，峰值 GPU 内存减少约 49%，更新延迟降低 26.3%，同时在相同记忆预算下较最强基线提升 macro-F1 4.66pp、micro-F1 4.75pp。

## 方法详解
**整体架构**：采用两阶段训练流程。第一阶段使用 LoRA 在完整患者历史上对 LLM 进行任务适配（SFT），获得任务特定的 backbone 参数 $\theta$；第二阶段冻结 $\theta$，仅训练记忆 token 嵌入与压缩适配器参数 $\phi$。

**记忆表示**（公式 3）：记忆 $M_t$ 定义为逐层 KV 对集合 $\{(K_t^\ell, V_t^\ell)\}_{\ell=1}^L$，其中每层每头固定 $B$ 个 memory slot，形状为 $\mathbb{R}^{H_{\text{kv}} \times B \times d_h}$。

**记忆条件编码**（公式 4）：冻结 backbone 将新就诊 $v_t$ 以 $M_{t-1}$ 为历史上下文进行编码，得到当前就诊的逐层 KV 对 $P_t = \text{KV}_\theta(v_t | M_{t-1})$。

**固定容量更新**（公式 5）：将 $B$ 个 memory token 追加至就诊序列末尾，通过 token-conditioned 低秩适配器（仅在这些位置激活）处理 $[M_{t-1}; P_t]$ 前缀 KV cache，输出新的 $B$ 个 memory token 的 KV 对作为 $M_t$，替换旧记忆。

**中间注意力对齐损失**（公式 6-8）：
- 独立用冻结 backbone 编码完整历史 $H_s$ 得到参考 $F_s = \text{KV}_\theta(H_s)$。
- 用下一就诊或任务查询生成共享 query $Q_s$，分别读取 $M_s$ 和 $F_s$ 得到输出 $O_s^M$ 和 $O_s^H$。
- 最小化归一化平方差：$\mathcal{L}_{\text{inter}}(\phi; s) = \frac{1}{|\mathcal{S}| H_q} \sum_{\ell \in \mathcal{S}} \sum_{h=1}^{H_q} \frac{\|O_{s,\ell,h}^M - O_{s,\ell,h}^H\|_F^2}{\|O_{s,\ell,h}^H\|_F^2 + \epsilon n_s d_h}$。
- 为降低开销，每个样本仅均匀采样一个中间边界 $s$ 进行对齐计算。

**最终预测监督损失**（公式 9）：
$\mathcal{L}_{\text{pred}}(\phi) = -\frac{1}{N} \sum_{j=1}^N \log p_\theta(y_j | M_T, q, y_{<j})$，仅优化 $\phi$。

**联合目标**（公式 10）：
$\min_\phi \mathbb{E}_{(H_T,q,y)\sim\mathcal{D}}[\mathcal{L}_{\text{pred}}(\phi) + \frac{\lambda}{T}\sum_{s=1}^T \mathcal{L}_{\text{inter}}(\phi; s)]$，其中 $\lambda = 0.1$。

**课程学习**（公式 11）：训练阶段 $e$ 使用就诊数阈值 $\tau_e$ 筛选样本集 $\mathcal{D}_e = \{(H_T,q,y) \in \mathcal{D} : T \leq \tau_e\}$，药物预测的 $\tau_e$ 依次为 $(4, 6, \infty, \infty, \infty)$，诊断预测为 $(6, 8, \infty, \infty, \infty)$。

**位置编码策略**：记忆 token 不推进历史 token 计数器，采用 RoPE 并遵循 skip-position 约定，确保压缩记忆与完整历史在相同 query 下可比。

## 实验与结果
**数据集**：MIMIC-IV 回顾性电子健康记录，构建两个任务——药物预测（基于已完成就诊+目标就诊诊断/操作，预测首24小时内 ATC level-3 药物类别）和下一次就诊诊断预测（仅基于已完成就诊预测后续诊断类别）。训练/验证/测试集按患者划分，无重叠。药物预测测试集含300例、107类药物类别；诊断预测测试集含300例、584个诊断类别。

**评估指标**：Macro-F1、Micro-F1、P@k/R@k（$k\in\{5,10\}$）、历史存储量、峰值 GPU 内存、更新延迟、预测延迟。

**基线方法**：Full History（无压缩）、LLM-Rsum（递归文本摘要）、SnapKV、KVzip、Attention Matching、RMT、CCM-merge。

**主要结果（药物预测，Qwen3-4B，B=1024）**：
- Full History：Macro-F1 = 62.22%，Micro-F1 = 63.32%，历史预算 = 35,279 tokens。
- ReLMem：Macro-F1 = 61.93%，Micro-F1 = 63.24%，历史预算 = 1,024 tokens（仅 Full History 的 2.9%）。
- 较最强压缩基线 CCM-merge（Macro-F1=57.27%，Micro-F1=58.49%）提升 +4.66pp / +4.75pp。
- 资源效率：峰值 GPU 内存从 17.44 GiB 降至 9.24 GiB（-47%），更新延迟从 0.4676s/visit 降至 0.3446s/visit（-26.3%）。

**关键分析结论**：
- 仅用预测监督（无中间对齐）时 Macro-F1 暴跌至 0.37%，表明中间对齐对信息保留至关重要。
- 移除预测监督仅保留对齐时 Macro-F1 仅 37% → 说明两者互补。
- 记忆容量从 1024 降至 128（8倍压缩），Macro-F1 仅下降 2.81pp，仍优于 KVzip 在 2048 slots 上的表现。
- 模型扩展性：在 Qwen3-8B 和 Llama-3.1-8B 上 ReLMem 与 Full History 的 F1 差距均小于 2.5pp。
- 诊断预测任务：ReLMem（Macro-F1=35.43%，Micro-F1=35.85%）与 Full History（36.30% / 37.41%）差距仅 0.87pp。
- 长历史测试：将就诊数翻倍至平均 16.72 次，ReLMem 仍保持 Macro-F1=62.20%，大幅优于 CCM-merge 的 40.71%。

## 相关工作脉络
1. **Transformer-XL / Compressive Transformer**：开创分段级循环记忆范式，但前者扩展固定上下文长度、后者压缩历史后重建注意力。ReLMem 定位为面向 EHR 纵向场景的专用循环记忆框架，强调任务相关信息保留而非通用序列建模。
2. **RMT (Recurrent Memory Transformer)**：使用最后一层 hidden state 承载跨就诊记忆。ReLMem 与之本质区别在于保留逐层 KV 状态，实验表明隐藏状态表示存在表征瓶颈（RMT Macro-F1 仅 47.28% vs ReLMem 61.93%）。
3. **CCM (Compressed Context Memory)**：在线交互场景的循环压缩方法，采用累积平均策略合并跨轮记忆。ReLMem 采用直接替换策略配合中间对齐监督，避免历史信息被稀释。
4. **KV 缓存压缩（SnapKV / KVzip / Attention Matching）**：针对单次长上下文压缩设计，通过注意力分数选择保留 KV 条目。ReLMem 定位差异在于将其 adaptation 到循环场景后性能显著下降，说明直接的上下文压缩策略无法适配纵向信息累积。
5. **LLM-Rsum / LLMLingua**：基于文本摘要的压缩方法。ReLMem 实验表明 KV 状态保留比文本摘要更具预测价值（KVzip 52.53% vs LLM-Rsum 18.97%），体现了结构化历史信息的优势。
6. **纵向 EHR 建模（BEHRT / TransformEHR / EHR-R1）**：使用预训练 Transformer 处理患者历史记录。ReLMem 补充分支在于引入固定容量循环记忆以解决扩展性问题，而非仅提升单次预测精度。

## 局限性与未来方向
1. **数据局限性**：仅在单一中心 MIMIC-IV 上 retrospective 评估，未在跨机构或 prospective 场景中验证，临床泛化性存疑。
2. **模型规模与家族覆盖不足**：实验仅覆盖 Qwen 和 Llama 的 4B/8B 模型，更大规模（如 70B+）或其他架构（如 MoE）的表现未验证。
3. **任务适配方式**：当前每个任务需单独适配 backbone 和记忆模块，缺乏 task-agnostic 的通用记忆表示；多任务共享记忆的研究未探索。
4. **中间对齐的计算开销**：虽采用单点采样降低开销，但 full-history reference 的训练计算仍随历史长度增加，在线部署时的实时对齐策略有待优化。
5. **未来方向**：扩展至更多 EHR 任务（如风险预测、生存分析）、开发跨任务共享记忆、探索更高效的 memory update 机制、在真实临床工作流中进行 prospective 验证。

## 研究启发与可借鉴点
1. **中间监督对于循环压缩至关重要**：仅依赖最终任务监督会导致信息快速退化，设计中间层的对齐/蒸馏目标可有效缓解循环更新中的信息丢失。该思路可迁移至任何递归压缩场景（如长对话记忆、时序文档更新）。
2. **逐层 KV 状态比单嵌入更适合作为循环记忆载体**：实验证明保留多层 attention 信息（而非仅最后一层 hidden state）显著优于 RMT，提示在需要精细历史检索的场景中应优先保留 KV 结构。
3. **课程学习对长程循环建模的增益**：从短序列逐步过渡到长序列的训练策略帮助模型学习稳定的更新规则，可推广至其他需要跨多步累积信息的任务（如持续学习、在线推理）。
4. **Token-conditioned 稀疏适配器设计**：仅在 memory token 位置启用训练参数、其余位置冻结 backbone，实现了极低的参数量开销（仅需训练 $B$ 个 token 嵌入 + 低秩适配器），为资源受限的循环记忆更新提供了高效范式。
5. **归一化注意力对齐损失的稳定性设计**：采用目标输出范数进行归一化（公式 8）避免了梯度尺度问题，该技巧可复用于其他基于 attention output 对齐的蒸馏任务。

## 关键术语表
**Longitudinal EHR Modeling**：整合患者多次就诊的时序电子健康记录以支持临床预测的建模任务。
**Recurrent Memory**：在连续信息片段间传递固定容量状态的记忆机制，避免历史累积导致的存储爆炸。
**Multi-granularity Optimization**：结合中间层对齐监督与最终任务预测监督的联合训练策略，兼顾过程保真与结果准确。
**KV Cache Compression**：通过选择性保留或压缩 attention key-value 对来降低 LLM 上下文存储与计算开销的技术。
**Curriculum Learning**：从简单样本（短历史）开始训练、逐步引入复杂样本（长历史）的学习策略，提升模型对长程依赖的适应能力。
**Token-Conditional Low-Rank Adapter**：仅在特定 token 位置（如 memory token）激活的低秩微调模块，其余位置保持 backbone 参数冻结。
**Attention Alignment**：通过比较压缩记忆与完整历史在相同 query 下的 attention 输出，约束压缩过程保留关键历史信息。
**ATC Level-3**：解剖学治疗学及化学分类系统的第三级分类，用于标准化药物类别标注。

## 可复现要素
- **数据集**：MIMIC-IV（PhysioNet，需 credentialing 申请），论文提供了详细的数据构建流程（附录 A、B）。
- **代码/权重**：论文声明 "Code will be publicly released upon acceptance"，当前未公开。
- **关键超参**：Memory slots $B = 1024$，对齐权重 $\lambda = 0.1$，LoRA rank=8 / $\alpha$=16（task adaptation）和 rank=8 / $\alpha$=8（memory learning），学习率 $10^{-4}$（SFT）和 $3\times10^{-4}$（memory learning），BF16 精度，8×A800-80GB GPU。
- **骨干模型**：Qwen3-4B（默认），额外评估 Qwen3-8B 和 Llama-3.1-8B。
