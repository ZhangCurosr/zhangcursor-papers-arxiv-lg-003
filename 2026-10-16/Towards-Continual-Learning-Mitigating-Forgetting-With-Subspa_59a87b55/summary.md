---
title: "Towards-Continual-Learning-Mitigating-Forgetting-With-Subspa"
source: https://arxiv.org/pdf/2610.22293v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:05:08"
field: "大语言模型持续学习与参数高效微调"
keywords: ["catastrophic forgetting", "subspace tuning", "continual learning", "LLM fine-tuning", "singular value decomposition", "on-policy distillation"]
innovations: ["对优化器最终更新做双侧奇异子空间投影以严格限制更新方向", "提出并系统化 basis refresh 机制作为适应-保留控制旋钮", "证明对于 Adam 等有状态优化器必须投影更新而非梯度才能精确约束子空间"]
benchmarks: ["GPQA-Diamond", "MMLU-Pro", "GSM8K", "HumanEval", "IFEval", "FinQA", "MedMCQA"]
---

# 论文速读：Towards-Continual-Learning-Mitigating-Forgetting-With-Subspa

## 一句话总结
论文提出 Subspace Tuning (SST)，通过将每个权重矩阵的优化器更新投影到非主奇异子空间（保留 top-99% 奇异方向不变），在适配新任务的同时大幅缓解 LLM 的灾难性遗忘；该方法与优化器无关，可无缝组合 SFT、SDFT 及 Muon 等训练配方。

## 研究问题与动机
- **核心问题**：LLM 经全参数微调（SFT）学习新任务时，预训练获取的能力（代码生成、推理、指令遵循等）会被严重破坏（灾难性遗忘），而原始预训练数据通常不可获得，阻碍了持续学习能力。
- **现有洞察**：预训练权重矩阵的主奇异方向（leading singular directions）与模型重要功能强相关；RLVR/on-policy RL 的更新集中在非主方向，因而遗忘较少；SFT 则沿主方向集中更新，易导致遗忘。
- **关键假设**：保护预训练权重的「主奇异子空间」可保留既有能力，同时在非主子空间留出容量以适应新任务。
- **已有方法不足**：SDFT 等通过 on-policy 数据间接实现该效果，对长推理链场景仍显著遗忘；Sculpting/OPLoRA 等子空间约束方法仅部分移除有效块且未设计刷新机制，遗忘抑制有限。

## 核心贡献（创新点）
1. **提出 SST（Subspace Tuning）**：对优化器最终更新做双侧投影 $\Pi(\Delta W)=(I-U_kU_k^\top)\Delta W(I-V_kV_k^\top)$，将更新严格限制在非主奇异子空间，使 top-k 奇异三元组完全不变。
   - **本质区别**：直接约束「到达权重的更新量」而非梯度本身，因此对 Adam 等有状态优化器天然兼容；现有 Sculpting 投影梯度只能移除 4 个块中的第 (1) 块，遗漏 (2)(3) 跨块导致 69% 更新仍泄漏到受保护方向。
2. **严格的对照实验确立投影位置的最优性**：在 Toy 实验中比较投影梯度/投影更新/重投影累积变化三种方案，证明仅投影应用更新能同时消除泄漏且保持泛化 NLL 几乎不变。
   - **本质区别**：此前工作（Sculpting）默认梯度投影即足够，本文证明因 Adam 对角预条件不与投影交换，梯度投影等价于近似方案，会产生严重泄漏。
3. **引入 Basis Refresh 机制（每 K 步重算 SVD）**：将 $W_{\text{ref}}$ 锚定到当前权重并重新计算保护基，缓解长训练时主奇异方向旋转带来的约束僵化，作为适应-保留控制旋钮。
   - **本质区别**：Sculpting/OPLoRA 均未考虑保护子空间的动态刷新；SST 将 refresh 作为方法论组成部分并系统扫过 $K \in \{\infty, 100, 10\}$。
4. **系统性实证：SST 在模型规模、数据规模、多阶段连续微调三个维度上均呈现一致收益**。
   - **本质区别**：与「遗忘是学习的必然代价」的直觉相反，本文证明遗忘本质上是更新落在权重空间何处的问题，可通过更新几何直接控制。

## 方法详解
- **SVD 分解与分块**：对权重矩阵 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，其 SVD 给出奇异值 $\sigma_i$ 与左右奇异向量 $u_i,v_i$。令 $U_k,V_k$ 为 top-k 向量，$P_U=U_kU_k^\top,P_V=V_kV_k^\top$ 为投影，$Q_U=I-P_U,Q_V=I-P_V$ 为补空间。任意更新 $\Delta W$ 可分解为四个块：
  $$\Delta W=\underbrace{P_U\Delta W P_V}_{(1)}+\underbrace{P_U\Delta W Q_V}_{(2)}+\underbrace{Q_U\Delta W P_V}_{(3)}+\underbrace{Q_U\Delta W Q_V}_{(4)}$$
  其中块 (1) 同时触碰受保护的输入/输出方向，(2)(3) 在一侧触碰保护方向，(4) 完全在自由子空间。保留块 (4) 即实现保护。
- **投影公式**：SST 应用的更新为 $\Pi(\Delta W)=Q_U \Delta W Q_V=(I-U_kU_k^\top)\Delta W(I-V_kV_k^\top)$，使得 $W_{\text{ref}}+\Pi(\Delta W)$ 在所有受保护输入方向上作用与 $W_{\text{ref}}$ 完全相同，且不产生新的受保护输出分量。
- **投影位置决定于优化器更新而非梯度**：对 Adam 等状态优化器，对角预条件矩阵 $D=\text{diag}(1/(\sqrt{\hat v}+\epsilon))$ 与 $\Pi$ 不交换，$\Pi(g)$ 经 Adam 后产生的更新仍含受保护方向分量。SST 采用方案 (2)：先让 Adam 用全梯度更新动量并产出让步 $\Delta W$，再对 $\Delta W$ 做双侧投影替换该矩阵的更新分量。
- **Rank fraction $r$**：$k=\lfloor r\cdot\min(d_{\text{out}},d_{\text{in}})\rfloor$，文中默认 $r=0.99$，保护 99% 方向，仅留 1% 低能尾部用于适配。
- **Refresh 机制**：每 $K$ 步把 $W_{\text{ref}}^\ell \leftarrow W_{t+1}^\ell$ 并基于当前权重重算 $U_k^\ell,V_k^\ell$；使用 Gram-based 特征分解加速。$K=\infty$ 为最严格保护，$K=10/100$ 逐步放松。
- **投影范围**：对每个 attention 线性层（q,k,v,o）和 MLP 线性层（gate,up,down）应用，embedding/layer norm 不受约束。

## 实验与结果
- **数据集与模型**：主实验基于 Qwen3-8B（thinking 模式），在 Science、Finance (FinQA)、Pathology、Medicine 四个领域微调，数据含显式 $\texttt{<think>}...\texttt{</think>}$ 推理链；另在 Qwen2.5-7B-Instruct、Olmo-3-7B-Think、Qwen3-1.7B/4B/14B/32B 上做规模测试。
- **评估指标**：ID accuracy 衡量目标任务学习；Prior capability 为 GPQA-Diamond、MMLU-Pro、GSM8K、HumanEval、IFEval 五榜平均相对 base 的变化；WikiText-2 上的 KL 漂移度量分布接近度。
- **主要结果（Qwen3-8B，no-refresh 版）**：
  - **Science**：SFT ID=77.6、LM avg 降至 55.5（$\Delta=-25.5$）；SFT+SST ID=80.1、LM avg=80.0（$\Delta=-1.0$）。
  - **FinQA**：SFT ID=72.3、LM avg=23.7（$\Delta=-57.3$）；SFT+SST ID=73.0、LM avg=82.3（$\Delta=+1.3$）。
  - **Pathology**：SFT ID=78.3、LM avg=34.7（$\Delta=-46.3$）；SFT+SST ID=77.3、LM avg=77.2（$\Delta=-3.8$）。
  - **Medicine**：SFT ID=74.8、LM avg 大幅下降；SFT+SST ID=72.4、LM avg=78.9（$\Delta=-2.1$）。
- **缩放实验**：模型从 1.7B→14B 在 Science 上 5 epoch，SFT 丢失 30–40pp prior capability；SST 在 8B+ 几乎归零（$\Delta\approx-1$），且新任务精度更高。数据规模从 9k×5 到 45k 单次暴露，SFT 漂移单调增大，SST 保持平坦且 ID 持续上升。
- **KL 漂移**：SST 在 WikiText-2 上的 $KL(\pi_{\text{base}}\|\pi_\theta)$ 比 SFT 低 2–4×；ID 提升约 3× 更少的分布偏移。
- **多阶段连续微调**：两阶段（A→B）和 FinQA→Pathology→Science 三阶段链中，SST+R10 各阶段技能在学习后维持不变，SFT 在未训练阶段分别下降 28/18pp。
- **对比 Sculpting/OPLoRA**：四领域均值 LM avg 分别为 54.9/63.1 vs. SST 79.6；OPLoRA 即使开 $k=0.99d$ 仍在 3/4 领域落后。
- **自蒸馏场景**：Olmo-3-7B-Think 在 answer-only 医疗数据上，SFT 使推理长度急剧缩短，SDFT 会捏造"reference solution"；SST+SDFT 大幅降低幻觉率并保留推理长度。
- **反向 KL 正则**：$\mathcal{L}=\text{CE}+\lambda\text{KL}(\pi_\theta\|\pi_{\text{base}})$ 在 $\lambda\in\{0.001,0.01,0.1\}$ 均无法恢复 prior capability（最强仅 64.3 vs. SST 80+）。
- **计算开销**：在 8×L40S 上，SFT step=1.73s；SST 投影增加 26.2ms/step（1.5%）；R100/R10 refresh 额外增加 62.4ms/623.5ms；GPU 分裂后开销降至 0.6%/4.7%。

## 相关工作脉络
1. **Sculpting（ICLR 2026）**：冻结每个权重矩阵 top 奇异向量并在每次 AdamW 前投影梯度；只移除分块 (1) 而遗漏 (2)(3)，且作用于梯度而非更新，导致 69% 泄漏。SST 投影最终更新，彻底清除所有三类泄漏并引入 refresh。
2. **OPLoRA（AAAI 2026）**：冻结权重、训练 LoRA 因子 $BA$ 并通过固定投影 $\Delta W=Q_U BA Q_V$ 限制到低秩子空间；本质是「固定投影下的低秩适配」，与 SST「全参 Dense 更新投影」相对；其固定投影通过链式法则自动投影更新，但学习率设定在因子空间而非权重空间，且未研究 refresh。
3. **SDFT（Self-Distillation Fine-Tuning，Thinking Machines Lab 2025）**：通过 on-policy 自蒸馏使监督信号贴近模型输出分布以间接减小遗忘；属于「数据侧」方法，遗忘抑制有限（长推理场景仍显著遗忘），而 SST 与 SDFT 正交可组合（表 4 中 SDFT+SST 显著优于单独 SDFT）。
4. **RLVR / On-policy RL 遗忘分析**：RLVR 更新集中在非主方向、KL 漂移小（Zhu et al., 2025）；on-policy RL 接近 base policy 因而遗忘少（Shenfeld et al., 2026）；SST 从更新几何角度提供与这些观察一致的显式实现。
5. **经典 continual learning（A-GEM、EWC、GEOM）等**：依赖历史数据或梯度近似 Fisher 重要方向，LLM 规模下无法获得数据且无明确任务边界；SST 只需权重重 SVD，无需外部信息。
6. **PiSSA / MiLoRA / CorDA**：PEFT 类基于初始 SVD 构造 adapter；PiSSA 优先训练高能量方向、MiLoRA/CorDA 采用固定低秩分解；SST 直接在 dense 更新上做投影，无需 adapter 分解或任务协方差估计。

## 局限性与未来方向
- 约束改变了可行更新几何，最优学习率与 unconstrained SFT 不同，完全调好的 learning–retention Pareto 前沿尚未建立。
- 尚未扩展到百亿级以上模型和更长任务序列的连续微调（仅测试到 32B、3 阶段）。
- Refresh 采用固定间隔 $K$；基于每 batch 更新信息动态调整保护策略是下一步效率优化方向。
- 论文实验主要覆盖 reasoning/问答类任务，未覆盖生成型（如写作、翻译）等更广泛下游。
- 对 embedding / layernorm 不受约束，可能仍是少量泄漏源；是否需扩展到全部参数未讨论。

## 研究启发与可借鉴点
1. **「投影优化器更新」而非梯度**：对有状态优化器做精确子空间约束时，必须投影最终更新，这是本文最核心的方法学洞见；可作为其他参数高效微调方法的通用检查项。
2. **Refresh 作为适应-保留控制旋钮**：固定基的严格保护与周期刷新之间存在显式 trade-off；后续可设计自适应 refresh 策略（例如基于当前梯度在保护方向上的投影幅度触发重算）。
3. **用 NLL probe 而非仅靠 accuracy 验证遗忘**：Toy 实验显示，仅看目标 loss 无法区分三种投影方案的差异，必须借助 held-out 通用句子的 NLL 作为敏感探针；该评估设计值得推广到更多 continual learning 工作。
4. **SST 与 on-policy 方法可组合**：SDFT+SST 在四领域均优于单独 SDFT；表明「数据侧 on-policy + 更新侧子空间约束」正交互补的思路可复用至 RL 微调、on-policy distillation 等新场景。
5. **模型合并视角**：独立训练的 SST experts 合并时干扰远低于 SFT experts，说明低频子空间更新具有天然正交性；可启发多专家路由、task arithmetic 等方向的改进。

## 关键术语表
- **Subspace Tuning (SST)**：将全参数微调中每个权重矩阵的优化器更新双侧投影到其 top-k 奇异向量补空间的方法，使更新严格停留在低能尾部。
- **Catastrophic forgetting**：模型在学习新任务时，先前已掌握能力出现急剧退化的现象。
- **Principal singular directions**：权重矩阵最大奇异值对应的左右奇异向量张成的子空间，与模型核心功能高度相关。
- **On-policy distillation (SDFT)**：用模型自身改写训练数据构造监督信号，使训练分布贴近模型当前输出分布以降低遗忘。
- **Refresh (basis refresh)**：每隔 $K$ 步以当前权重重新计算 $U_k,V_k$，使保护子空间跟随权重演化、释放被旋转过的新自由方向。
- **Rank fraction ($r$)**：受保护奇异方向占总奇异方向的比例，$r=0.99$ 表示保留 top 99% 方向不动。
- **KL drift**：微调后模型策略与预训练 base 策略之间的 $KL(\pi_{\text{base}}\|\pi_\theta)$，度量分布整体偏移程度。
- **Toy controlled study**：在极简 next-token 任务上用 NLL 漂移和泄漏比例作为判别指标，用于隔离投影位置设计差异的受控实验。

## 可复现要素
- **数据集**：Science（open-r1/Mixture-of-Thoughts）、FinQA/ConvFinQA、MedMCQA；均由 HuggingFace 公开。合成推理链由 Claude Sonnet 4.6 生成。
- **代码/权重**：论文未声明开源；附录 B.14 提供了训练超参（lr=$5\times10^{-5}$、batch=16、DeepSpeed ZeRO-3）与评估细节（vLLM 推理、答案抽取器）。
- **关键超参**：rank fraction $r=0.99$（Medical 域因尾部太小用 $r=0.9$ 的特例），refresh 间隔 $K\in\{\infty,100,10\}$，投影矩阵为 attention q/k/v/o 与 MLP gate/up/down。
- **模型**：Qwen3-8B 为主测，另含 Qwen3 系列 1.7B/4B/14B/32B、Qwen2.5-7B-Instruct、Olmo-3-7B-Think。
