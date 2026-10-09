---
title: "UNROLLED-FLOW-MODELS-FOR-REASONING"
source: https://arxiv.org/pdf/2610.09759v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:07:04"
field: "流匹配语言模型推理"
keywords: ["Flow Matching", "Reasoning", "Rollout Training", "Sphere Retraction", "Structured Reasoning", "Unrolled Flow Model"]
innovations: ["提出终端 rollout 损失训练 flow 模型，使额外 Euler 步骤能协同提升推理性能", "引入球面收缩稳定长 rollout 动态，让小规模 flow 模型超越参数多三倍的标准基线"]
benchmarks: ["ProsQA", "Sudoku-Hard", "Sudoku-Extreme", "Maze-Hard"]
---

# 论文速读：UNROLLED-FLOW-MODELS-FOR-REASONING

## 一句话总结
本文提出 **Unrolled Flow Model (UFM)**，通过**终端 rollout 损失**训练 flow 模型，使额外积分步骤能够逐步构建推理信息；结合**球面收缩**稳定长 rollout 动态，在 ProsQA、Sudoku、Maze 等结构化推理任务上显著超越参数更多倍的 standard flow 基线，并给出图可达性问题的理论存在性证明。

## 研究问题与动机
- **核心问题**：Flow matching 能在少数步骤内生成文本，但增加 Euler 积分步骤是否能让 flow 语言模型更好地推理？现有工作表明标准 flow 语言模型（FLM、S‑FLM）在推理任务上准确率几乎不随步数增加而提升。
- **现有方法不足**：标准 flow 训练目标（公式 4）独立地监督每个随机采样时间点的 denoiser 预测，**从未评估模型自身前一更新产生的状态是否能为后续更新所用**，因此中间步骤无法协同完成推理。
- **理论动机**：Zhu et al. (2025) 证明两層 Transformer 可通过 latent superposition 逐步扩展可达顶点集合，但那是针对 autoregressive 潜在链，未说明 flow 框架下能否学到类似机制。
- **实践动机**：若能让额外 Euler 步骤真正服务于推理，flow 模型即可像 chain‑of‑thought 一样在推理时灵活分配计算量（adaptive test‑time compute）。

## 核心贡献（创新点）
1. **图可达性的 flow 存在性构造**：证明存在一个两層 Transformer（第一层缓存边/候选/起点，第二层为线性注意力传播头）定义的 flow，其 Euler rollout 每步将可达顶点集合扩展一跳，所需步数等于目标顶点到根的距离。
2. **终端 rollout 损失训练（UFM）**：仅在 rollout 终点施加词表解码交叉熵损失，让梯度流经连续更新步骤，使早期更新产出的状态能被后续更新利用；相比标准 flow 的逐点独立监督，本质区别在于**训练目标直接衡量步骤间的协同能力**。
3. **球面收缩（Sphere Retraction）稳定长 rollout**：对 Sudoku/Maze 等长序列任务，潜在范数会剧烈增长导致动态失稳；通过在每次更新前将球面投影到固定半径（位置级 $\psi(z)=\sqrt{d}\,z/\|z\|_2$），控制输入尺度，使 8.4M 参数的 UFM 在参数 28.6M 的 baseline 上获得大幅超越。
4. **多 rollout 采样 + 参数自由 margin 选择**：推理时采样多条轨迹，用输出 logits 间距的平均值（无参数）挑选最佳答案，在 Sudoku‑Extreme 上接近 Pass@100，但在更长答案的 Maze‑Hard 上选择仍具挑战。
5. **从理论构造到可训练机制的衔接**：虽未证明梯度下降能精确恢复理论构造，但通过 logit 分析显示训练后的 UFM 中间状态同样呈现“ reachable 顶点集体右移、最优顶点最先分离”的模式，与理论中的逐跳扩展定性一致。

## 方法详解
### 1. 基础 flow matching 设定
- 给定提示 $c$ 和长度为 $L$ 的答案，flow 将 $t=0$ 处的高斯噪声 $\boldsymbol{x}_0\sim\mathcal{N}(0,I)$ 映射到 $t=1$ 处的干净表示 $\boldsymbol{x}_1$。
- 采用线性插值路径 $I_t=(1-t)\boldsymbol{x}_0+t\boldsymbol{x}_1$，denoiser $\widehat{\boldsymbol{x}}_\theta(\boldsymbol{x},t,c)$ 预测 $\boldsymbol{x}_1$ 的条件期望，对应速度场为 $\boldsymbol{v}_\theta(\boldsymbol{x},t)=(\widehat{\boldsymbol{x}}_\theta(\boldsymbol{x},t,c)-\boldsymbol{x})/(1-t)$（公式 2）。
- 用 Euler 方法离散积分：$z_{t_{k+1}}=z_{t_k}+\Delta t_k\,\boldsymbol{v}_\theta(z_{t_k},t_k,c)$。

### 2. UFM 的 latent rollout 训练
- **潜在空间表示**：状态 $\boldsymbol{z}_t\in\mathbb{R}^{L\times d}$（$d$ 为隐藏宽），DiT 预测同维度的 latent target $\widehat{\boldsymbol{x}}_\theta$。
- **单次解码**：词汇表解码器 $W_{\text{dec}}\in\mathbb{R}^{V\times d}$ **仅在终点**应用：$\hat{\boldsymbol{y}}=\operatorname{softmax}(z_{t_N}W_{\text{dec}}^\top)$（公式 7）。ProsQA 用 softmax，Sudoku/Maze 用 StableMax 以增强数值稳定性。
- **终端 rollout 损失**：训练时从随机采样时间点 $t_0$ 开始 rollout，只在 $t_N=1$ 计算交叉熵 $\mathcal{L}_{\text{UFM}}$（公式 7）。早期状态**不施加词表监督**。
- **截断反向传播**：为节省显存，仅对最后 $N_{\text{back}}$ 步记录梯度，前 $N_{\text{train}}-N_{\text{back}}$ 步执行 `sg()` 停止梯度（Algorithm 1）。

### 3. 球面收缩（Sphere Retraction）
- 在 Sudoku/Maze 长 rollout 中，latent norm 会指数增长。对每个答案位置独立执行投影：
  $$\psi(\boldsymbol{z})=\sqrt{d}\,\frac{\boldsymbol{z}}{\|\boldsymbol{z}\|_2}$$
- 更新规则（Algorithm 2）：DiT 仍读取原始状态 $\boldsymbol{z}_{t_k}$，但**用收缩后的版本作为 Euler 步的起点**，即 $y_{t_k}=\psi(z_{t_k})$，然后 $z_{t_{k+1}}=y_{t_k}+\frac{\Delta t_k}{1-t_k}(\widehat{\boldsymbol{x}}_\theta(z_{t_k},t_k,c)-y_{t_k})$。
- 效果：限制每步输入尺度，防止范数爆炸；消融实验显示去掉收缩后 Sudoku‑Extreme 准确率下降约 12%。

### 4. 推理与 rollout 选择
- 推理从纯噪声 $t_0=0$ 开始，使用 EMA 权重，沿均匀时间网格执行 $N_{\text{eval}}$ 步 Euler 更新。
- 采样 $K=100$ 条独立轨迹，用 **margin 评分**选择最佳答案：
  $$\widehat{\delta}=\frac{1}{L}\sum_{i=1}^L\bigl(\ell_i^{(1)}-\ell_i^{(2)}\bigr)$$
  其中 $\ell_i^{(1)},\ell_i^{(2)}$ 为位置 $i$ 上最大与次大原始 logit（Section 5.4）。该分数无需额外参数。

## 实验与结果
### 数据集与评估协议
- **ProsQA**：有向图可达性，候选答案二选一，答案距离根 ≤4 跳；14,785 训练 / 257 验证 / 419 测试。
- **Sudoku‑Hard**：30 个已知数的数独，48k 训练 / 2k 报告集 / 另 2k 验证集用于 UFM 选择。
- **Sudoku‑Extreme**：422,786 测试题，训练用 1k 基础题 ×1000 增强。
- **Maze‑Hard**：30×30 迷宫，最短路径 >110 格，1k 训练 / 1k 测试。

### 主要结果（Pass@1，$N_{\text{eval}}=128$）
| 基准 | FLM (28.6M) | S‑FLM (28.6M) | **UFM (8.4M)** |
|------|-------------|---------------|----------------|
| ProsQA | ~12%（N=1） | ~12% | **97%**（N=5 训练，长 rollout 提升） |
| Sudoku‑Hard | 51.9% | 50.9% | **86.9±0.9%** |
| Sudoku‑Extreme | 10.7% | 9.4% | **74.4±0.0%** |
| Maze‑Hard | 40.4% | 49.9% | **89.3±0.5%** |

- **步数 Scaling**：FLM/S‑FLM 准确率几乎不随 $N$ 变化（ProsQA 图 2、Sudoku‑Hard Table 6），而 UFM 从 $N=1$ 的 12% 提升至约 97%，证明额外 Euler 步骤真正被用于推理。
- **多 rollout 选择**：UFM 在 Sudoku‑Extreme 上 $K=100$ 经 margin 选择达到 **98.6%**（Table 5），接近 Pass@100 的 98.65%；Maze‑Hard 选择效果较弱（92.0% vs Pass@100 95.0%），因长答案中大量共享单元格稀释了 logit 差距。
- **训练效率**：UFM 参数量仅为 baselines 的 ~1/3，训练时间也更短（Sudoku‑Hard 1.1h vs 1.6–1.7h）。

## 相关工作脉络
1. **Recursive / Looped Reasoners**（HRM, TRM, EqR, GRAM, FPRM）：同样反复应用小网络更新潜在状态；UFM 与之本质区别在于使用 **flow 速度场**而非 recurrent 权重 tying，且通过终端 loss 而非局部 reward 训练。
2. **Latent Chain of Thought**（Coconut, Zhu et al. 2025）：Coconut 将 LLM hidden state 反馈为下一轮 embedding；Zhu et al. 证明两層 Transformer 可维护可达顶点 superposition 并逐跳扩展。**UFM 的构造直接继承其图搜索思想**，但将其嵌入 flow matching 框架并证明 Euler 离散化可实现逐跳传播。
3. **Flow Language Models**（FLM, S‑FLM, ELF）：标准 flow 模型在每个采样时间点独立监督 denoiser；UFM 与之对比表明，**独立监督无法让中间步骤服务于后续推理**，必须通过终端 rollout loss 建立步骤间协同。
4. **Differentiating through Generation**（DRaFT）：DRaFT 将可微 reward 通过 diffusion sampler 回传，并提出截断至最后 $K$ 步的技巧；UFM 采用相同训练机制，但**从零预训练**而非 fine‑tune，且目标为词表交叉熵。
5. **Flow Reasoning Models (FRM)**：FRM 为 flow 添加循环自条件与 fixed‑point forcing，并报告 standard FLM 增加步数无法提升数独表现——与本文 ProsQA 观察一致；但 FRM 使用 carry 并停止梯度，而 UFM 梯度流经最后若干步。
6. **Looped Flows**（Suleymanzade et al., 2026）：并发工作，使用局部去噪损失并停止更新间梯度；UFM 则用单一终端损失并通过截断反向传播训练，两者设计哲学不同。

## 局限性与未来方向
- **模型规模受限**：实验仅使用 8M–16M 参数的小模型；rollout 训练需要存储反向传播后缀，限制了可扩展性。扩展到大模型需要更内存高效的连续更新联合训练策略。
- **理论构造与训练 Gap**：Section 3.1 的存在性证明依赖正交嵌入、无 softmax 线性传播头等理想假设，并未证明梯度下降能恢复该速度场；训练模型的动态仅与理论构造**定性相似**。
- **长答案 rollout 选择困难**：Maze‑Hard 等长序列任务上，参数自由 margin 评分选择性较差；需引入**学习的 halting / reward head**（如 HRM、TRM 所用）或 latent reward model。
- **任务范围限于结构化合成基准**：尚未验证 UFM 在非结构化语言推理（如数学证明、代码生成）上的有效性。
- **未来方向**：① 设计内存高效的长 rollout 训练（如checkpointing、并行 rollout）；② 集成 learned scorer/halting mechanism 提升多 trajectory 选择可靠性；③ 将 UFM 扩展至更大语言模型与更一般的 reasoning 任务。

## 研究启发与可借鉴点
1. **终端 rollout loss 思想可迁移**：任何需要多步迭代更新的生成模型（扩散、流、循环网络），若希望中间状态具备“可继续利用”的特性，均可尝试用最终任务损失反向穿过整条轨迹进行训练，而非仅监督瞬时预测。
2. **球面收缩作为动态稳定器**：当潜在状态范数在连续应用中易发散时，位置级/维度级投影到固定超球面是一种简单有效的正则化手段，可嵌入任意 Euler‑type 更新流程。
3. **截断反向传播的工程实践**：只对最后 $N_{\text{back}}$ 步保留梯度、前期步骤仅做前向传递，能在几乎不损失性能的前提下大幅降低显存，适合长 rollout 训练。
4. **无参数 margin 选择作为 baseline**：在多 trajectory 采样场景下，直接使用输出 logits 间距均值进行筛选，无需额外训练选择器即可初步验证“多数投票/最佳选择”的收益上限。
5. **理论构造指导经验模型分析**：先给出一个理想化的可解释机制（图搜索的逐跳扩展），再观察训练后模型是否涌现类似动态，为黑箱模型的 mechanistic interpretability 提供可操作的验证路径。

## 关键术语表
- **Flow Matching**：通过定义从噪声分布到数据分布的概率流，并学习该流的速度场，从而在少量积分步内生成样本的生成建模方法。
- **Unrolled Flow Model (UFM)**：本文提出的模型，通过将终端交叉熵损失反向传播穿过整个 Euler rollout 来训练 flow 语言模型，使中间更新步骤协同服务于最终答案。
- **Euler Rollout**：使用欧拉方法对连续时间 flow 方程进行离散化，逐步更新潜在状态的过程。
- **Sphere Retraction**：在每次 Euler 更新前将潜在状态投影到固定半径的超球面上，以控制范数增长、稳定长 rollout 动态的技术。
- **Terminal Rollout Loss**：仅在 rollout 终点计算的任务损失，梯度流经所有已执行的更新步骤。
- **Recursive Reasoner**：通过反复应用同一组权重的小网络（如 RNN、循环 Transformer）更新内部潜在状态来完成推理的模型族。
- **Latent Chain of Thought**：在连续潜在空间中维护推理中间表示，而非逐步生成文本 token 的推理范式。
- **Margin Score**：无参数选择分数，取各输出位置最大与次大 logit 之差均值，用于在多条 rollout 轨迹中挑选最可靠答案。

## 可复现要素
- **数据集**：ProsQA（Hao et al., 2026）、Sudoku‑Hard / Sudoku‑Extreme（Deschenaux & Gulcehre, 2026；Wang et al., 2025）、Maze‑Hard（Wang et al., 2025）。论文未提供数据下载链接，但均引用自已有公开工作。
- **代码/权重**：论文声明“implementation can be found here”（链接未在摘录文本中给出），代码应已开源；权重未明确说明是否发布。
- **关键超参数**：
  - 模型：两层 DiT，隐藏宽度 $d=448$，8 个注意力头，参数 8.4M；pre‑norm RMSNorm、SwiGLU（expansion 4）、adaLN‑Zero、2D Rotary PE。
  - 训练：AdamW ($\beta=(0.9,0.95)$)，梯度裁剪 1.0，dropout 0.1，EMA decay 0.999；学习率余弦退火 $10^{-4}\to10^{-5}$。
  - Rollout：$N_{\text{train}}=24$，截断反向传播 $N_{\text{back}}=6$；时间网格在 $(t_0,1)$ 内独立采样排序，$t_0\sim\mathcal{U}[0, t_{0,\max}]$（$t_{0,\max}$ 从 0.6 线性降至 0.2）。
  - 解码：ProsQA 用 softmax，Sudoku/Maze 用 StableMax + RMSNorm + 线性投影。
  - 推理：$N_{\text{eval}}=128$ 均匀网格，$K=100$ 条轨迹，margin 选择。
