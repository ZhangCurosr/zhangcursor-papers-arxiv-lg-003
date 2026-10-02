---
title: "Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM"
source: https://arxiv.org/pdf/2609.37169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:36:09"
field: "大语言模型训练与计算分配"
keywords: ["mid-training", "model merging", "trajectory averaging", "compute scaling", "weight averaging", "LLM"]
innovations: ["将 mid-training 计算预算从单条轨迹延长改为分配到多条 recipe-perturbed 独立分支，提出 Trajectory Soup 两阶段合并", "通过局部偏差-方差分析分离 intra/inter 两阶段合并对验证 loss 的不同贡献，证明均匀权重方差最优并推导有限最优 checkpoint 数", "在 7.9B MoE 模型上验证 Extended 模式优于单轨迹最强 merge（68.96 vs 68.55），且优势经 SFT 后仍保持"]
benchmarks: ["ARC-Easy/Challenge", "AGIEval", "MMLU-Pro", "GSM8K", "HumanEval", "LiveCodeBench", "C-Eval"]
---

# 论文速读：Trajectory-Soup-Pushing-the-Compute-Scaling-Frontier-of-LLM

## 一句话总结
论文提出 **Trajectory Soup**，将 mid-training 的计算预算分配到多个从同一预训练 checkpoint 分叉的独立轨迹上，通过在每条轨迹内选择最佳 checkpoint 并跨轨迹合并，以在相同计算预算下超越单条长轨迹的训练效果，从而扩展 mid-training 的计算缩放边界。

## 研究问题与动机
1. **Mid-training 计算饱和**：对 LLM 进行 mid-training 时，下游性能随训练 token 数先快速上升，随后趋于饱和甚至退化；继续延长单条训练轨迹带来的收益有限。
2. **现有合并方法局限**：既有方法通常只关注单次运行内的 checkpoint 平均（intra-trajectory）或跨轨迹端点平均（inter-trajectory），尚未系统探索两者联合使用的效果。
3. **计算分配视角缺失**：mid-training 的计算预算分配（trajectory 数量 vs. 长度）作为一个缩放轴未被充分研究，轨迹数在单条轨迹长度饱和后仍可能持续产生收益。

## 核心贡献（创新点）
1. **将 mid-training 计算分配重新定义为"轨迹数"而非"单条轨迹长度"**，并发现 recipe 微调产生的独立分支在参数空间中保持几何方向差异，插值融合可进一步降低验证损失。
2. **提出 Trajectory Soup 两阶段合并框架**：先在每条轨迹内按验证 loss 排序选择 Top-K checkpoint 进行平均得到分支 anchor，再对所有分支 anchor 均匀平均得到最终模型；通过局部偏差-方差分析分离两个平均层级的贡献，证明均匀权重在方差最小化上是最优的，并推导出选择 bias 决定了最优 checkpoint 数量有限。
3. **在多模型规模、学习率调度、token 预算和轨迹数量下的系统实验验证**：在 Ling-3.0-Tiny（7.9B MoE）上，Trajectory Soup（Extended）取得 Overall Average 68.96，较最强单轨迹合并提升约 0.41 点；优势在共享 SFT 后得以保持；同时用 2B 小模型复现，验证方法鲁棒性。

## 方法详解
1. **分支生成**：从共享预训练 checkpoint $\theta_0$ 出发，基于不同训练配方（data shuffle seed、peak learning rate、global batch size、learning-rate schedule、optimizer momentum）并行生成 $N$ 条独立分支，每条分支训练相同 token 数 $t$，总预算 $T = Nt$。
2. **兼容性筛选**：每条分支需满足训练 loss 相对基准不超过阈值 $\varepsilon = 0.01$，以保证各分支优化质量可比。
3. **阶段一（Intra-Trajectory Merging）**：在每条分支内，按验证 accuracy 对保存的 checkpoint 排序，选取验证 loss 最低的 Top-$K$ 个 checkpoint，均匀平均得到该分支 anchor：
$$\bar{\theta}_n(t, K) = \frac{1}{K} \sum_{i \in \mathcal{I}_n(t, K)} \theta_n(i)$$
4. **阶段二（Inter-Trajectory Merging）**：将所有分支 anchor 均匀平均得到最终模型：
$$\theta_{\text{Traj-Soup}}(N, t, K) = \frac{1}{N} \sum_{n=1}^{N} \bar{\theta}_n(t, K) = \frac{1}{NK} \sum_{n=1}^{N} \sum_{i \in \mathcal{I}_n(t, K)} \theta_n(i)$$
5. **预算模式**：
- **Limited（计算守恒）**：$N$ 条分支共享固定总预算 $T$，每分支仅得 $T/N$ tokens。
- **Extended（预算扩展）**：固定每条分支预算 $t$，逐步增加分支数 $N$。
6. **理论分解**（局部二次 loss 模型 + 偏差-方差恒等式）：
$$\mathbb{E}[\mathcal{L}_Q(\theta_{\text{Traj-Soup}})] - \mathcal{L}^* = B_{\text{soup}}(K) + \frac{V_{\text{intra}}}{N \cdot K_{\text{eff}}(K)} + \frac{V_{\text{inter}}}{N}$$
其中 $K_{\text{eff}}(K)$ 为有效 checkpoint 数（受时间相关性影响，通常 $K_{\text{eff}} \leq K$）；bias $B_{\text{soup}}(K)$ 随 $K$ 增大而增大，解释了为何 checkpoint 数量存在最优上限。

## 实验与结果
- **模型**：主实验使用 Ling-3.0-Tiny（7.9B 总参数，1.3B active MoE，24 层，128 experts/layers）；鲁棒性实验使用 2B MoE（32 experts）。
- **Mid-training 评测**：41 项 benchmark，5 个能力类别（General Knowledge & Reasoning、Language Modeling、Professional Knowledge、Math、Code）。
- **主要结果（Tab. 2）**：
  - Single-Trajectory Merge: **68.55**
  - Model Soup (Extended): **68.67**
  - Trajectory Soup (Limited): **68.72**
  - Trajectory Soup (Extended): **68.96**（最强，较单轨迹提升 +0.41）
- **SFT 后结果（Tab. 3）**：Trajectory Soup (Extended) 在 SFT 后取得 Overall Average **61.52**，较 Single-Trajectory Merge（61.11）提升 +0.41，优势保持。
- **2B 小模型（Tab. 4）**：Trajectory Soup (Extended) 达 **50.07**，单轨迹仅 49.55。
- **Scaling 趋势（Fig. 5/8）**：随着轨迹数 $N$ 增加性能单调提升（但边际递减），每条轨迹贡献的 checkpoint 数 $K$ 在约 10–16 时性能最优，过多则振荡下降。
- **可控对照（Fig. 7）**：当候选池大小和合并 checkpoint 数完全匹配时（单轨迹密集采样 vs. 多轨迹稀疏采样），Trajectory Soup 仍全面胜出，证明收益来源于轨迹多样性而非采样密度。
- **方向多样性预测增益（Fig. 9）**：分支间 Leading Direction 的余弦相似度与插值增益呈强负相关（Pearson $r = -0.80$，Spearman $\rho = -0.94$）。

## 相关工作脉络
1. **SWA（Izmailov et al., 2018）**：对单条轨迹内的权重进行平均；本文在此基础上扩展，引入跨分支平均，二者在偏差-方差分解中对应不同项（$V_{\text{intra}}$ vs. $V_{\text{inter}}$）。
2. **Model Soups（Wortsman et al., 2022）**：对多条轨迹的最终 checkpoint 进行平均；本文在 Model Soup 基础上增加了分支内 Top-K checkpoint 选择与平均，减少了 end-point bias。
3. **Branch-Train-Merge（Li et al., 2022）**：并行训练专家后集成；本文关注同架构共享初始化的兼容分支，不依赖 MoE 路由结构。
4. **Extra-Merge（Zhou et al., 2026）**：发现后期轨迹近似秩-1结构并用于外推；本文利用该结构作为方向多样性分析的依据，但方法上更通用。
5. **WSM（Tian et al., 2026）**：通过 checkpoint 合并实现无衰减学习率调度；本文侧重 mid-training 阶段的计算分配与多分支融合。
6. **Scaling Laws（Kaplan et al., 2020; Hoffmann et al., 2022）**：将 trajectory 数量作为与参数规模和 token 数同等维度的计算分配轴，扩展了 scaling law 的讨论维度。

## 局限性与未来方向
1. **计算成本核算不完整**：当前仅以训练 token 数衡量 compute，未计入 checkpoint 存储、validation pass、超参搜索及加速器利用率差异。
2. **理论基于局部二次模型**：偏差-方差分解在局部二次 loss 假设下成立，实际 loss landscape 可能偏离此假设。
3. **轨迹多样性与兼容性需要经验调参**：recipe 扰动过强会使分支进入不可合并区域；目前轨迹数 $N$ 和 checkpoint 数 $K$ 依赖 validation 搜索，尚未实现训练中的自适应分配。
4. **强相关分支存在误差下界**：若分支间高度相关（如仅改变 data seed），则 $V_{\text{inter}}$ 无法有效降低，合并收益有限。
5. **未来方向**：建立 trajectory 成本模型以进行优化选择、在训练过程中自适应调整 $N$ 和 $K$、探索更广泛的数据混合变化作为分支扰动来源。

## 研究启发与可借鉴点
1. **轨迹多样性作为计算分配轴**：在 mid-training（乃至 continued pretraining / domain adaptation）中，可系统性地将计算预算分配到多个 recipe-perturbed 分支而非延长单条训练，这一视角具有通用迁移价值。
2. **两阶段合并 + bias-variance 分析**：先 intra 后 inter 的两层平均设计有清晰的理论解释；该分析框架可用于指导其他模型合并场景（如 LoRA 合并、多 SFT 模型融合）。
3. **方向余弦相似度预测合并增益**：分支间 Leading Direction 的 cosine similarity 可作为合并前 cheaply computable 的选择指标（$r = -0.80$），可用于自动筛选最有价值的分支对。
4. **平衡分配的 importance**：消融表明 Top-K per branch + 均匀权重 + 对称轨迹代表是最强配置；这提示在模型合并任务中应优先保证轨迹贡献的均衡性，而非依赖复杂加权。
5. **预算扩展（Extended）vs. 重分配（Limited）的对比实验设计**：通过两种预算模式分离"更多分支"与"更长单分支"的效应，为 scaling study 提供了清晰的消融范式。

## 关键术语表
- **Mid-training**：介于 pretraining 与 post-training（SFT/RLHF）之间的训练阶段，用于赋予模型专业能力和推理能力。
- **Trajectory Soup**：本文提出的两阶段 checkpoint 合并方法，先在每条分支内选择最佳 checkpoint 平均，再跨分支平均。
- **Intra-trajectory merging**：单条训练轨迹内的 checkpoint 平均，用于衰减沿优化路径的随机波动（$V_{\text{intra}}$）。
- **Inter-trajectory merging**：跨多条独立分支的 anchor 平均，用于融合不同优化方向的互补信息（$V_{\text{inter}}$）。
- **Branch anchor**：单条分支内 Top-$K$ checkpoint 均匀平均后的模型参数。
- **Compatibility screen**：要求分支的训练 loss 不超过基准一定容忍度（$\varepsilon=0.01$），确保合并的兼容性。
- **Limited vs. Extended budget**：Limited 为固定总 token 预算分配给多条轨迹；Extended 为每条轨迹固定预算、逐步增加轨迹数量。
- **$K_{\text{eff}}(K)$**：有效 checkpoint 数，表征 intra-trajectory 平均的实际方差缩减效率，受时间相关性和排名依赖影响。

## 可复现要素
- **数据集**：使用高质量 mid-training 语料（论文附录未公开具体数据），评测使用 41 项标准 benchmark；训练数据未公开。
- **代码/权重**：论文未明确声明开源代码仓库；模型 Ling-3.0-Tiny 托管于 HuggingFace（https://huggingface.co/inclusionAI/Ling-3.0-tiny）。
- **关键超参**：per-branch horizon $t = 600\text{B}$ tokens；checkpoint 保存间隔 $25\text{B}$ tokens；兼容性阈值 $\varepsilon = 0.01$；默认 $N=3$（主实验）、$K=4$（Extended）；序列长度 $262{,}144$。
