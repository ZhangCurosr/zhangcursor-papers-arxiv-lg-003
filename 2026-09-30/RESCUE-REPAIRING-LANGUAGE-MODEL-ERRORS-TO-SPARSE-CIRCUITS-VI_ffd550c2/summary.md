---
title: "RESCUE-REPAIRING-LANGUAGE-MODEL-ERRORS-TO-SPARSE-CIRCUITS-VI"
source: https://arxiv.org/pdf/2609.36813v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 17:11:18"
field: "大语言模型机理可解释性与模型修复"
keywords: ["mechanistic interpretability", "sparse circuit", "error repair", "reinforcement learning", "circuit pruning", "circuit-restricted tuning", "GSM8K", "MedMCQA"]
innovations: ["将电路定位从行为保持扩展至错误归因，提出两阶段 SFT+RL mask 精炼框架", "组内相对优势驱动的 generation-guided mask 精炼结合双约束 accept/reject 机制", "密度匹配随机电路对照+跨重述迁移评估，验证稀疏错误电路的因果泛化性"]
benchmarks: ["GSM8K", "MATH-500", "MedMCQA", "ARC-Challenge", "HellaSwag", "Wino-Grande", "PIQA", "TruthfulQA-MC1", "MMLU", "BBH"]
---

# 论文速读：RESCUE-REPAIRING-LANGUAGE-MODEL-ERRORS-TO-SPARSE-CIRCUITS-VI

## 一句话总结
RESCUE 提出了一套"定位–修编"范式：先用 SFT 基于教师修正答案优化 keep-mask 初步定位与任务失败相关的稀疏电路，再通过 RL 在多条 masked-model  rollout 的组内相对优势信号上精炼 mask，最后剪枝并仅在稀疏错误电路上做参数微调，从而以极小的参数更新量（~1.4%）修复模型错误并保持其他能力。

## 研究问题与动机
- 现有 mechanistic interpretability 研究多聚焦于"保留能力/解释安全"的稀疏电路（task circuits、trustworthy circuits），对更广泛任务中的失败机制与定向修复缺乏有效方法。
- SFT-based mask optimization 依赖离线教师修正参考，而多步推理/长输出任务中前缀偏移会导致参考答案偏离模型生成，从而遗漏真正导致推理错误的内部组件。
- 直接全参数微调（如 LoRA）在修复特定错误时往往引发非目标能力的显著退化，需要一种稀疏、可因果归因、仅扰动相关参数的修复路径。
- 单神经元/隐藏状态层面的归因（如 H-Neurons、CCS 等）多停留在相关性与可解码性，缺少可因果验证的 circuit-level 定位及其对泛化错误模式的迁移性检验。

## 核心贡献（创新点）
1. **将 circuit 视角从"能力保留"扩展到"错误归因与修复"**：定义 error-associated circuit 为"抑制后可纠正失败并保持干净输入行为"的稀疏单元集合，而非传统的功能保持子图。
2. **两阶段 mask 优化：SFT-based candidate search + RL-based generation-guided refinement**：第一阶段用教师修正答案与干净样本联合训练 keep-mask logits（STE+per-module budget），第二阶段用多条 masked-model  rollout 的组内相对优势（答案正确度 + 推理质量）更新 mask，缓解离线参考与自由生成之间的分布失配。
3. **电路剪枝 + 电路受限微调（circuit-restricted tuning）形成完整闭环**：按 closure strength 由弱到强逐步重新打开冗余 closed units，并在满足 error-correction 与 clean-preservation 双约束的前提下，仅对选定投影行引入零初始化 trainable delta 并合并回原参数。
4. **提供跨重述迁移与顺序修复的系统评估**：证明定位到的错误电路在不同表面表述下仍保持因果效应，且可串行累积修复而不显著抹除前序修复。
5. **密度匹配随机电路对照组**：同一稀疏比例与优化流程下随机 mask 几乎无效，说明修复收益来自功能相关的归因而非稀疏性本身。

## 方法详解
### 4.1 数据构造
- 在每道题 $x$ 上运行目标模型，按最终答案正误划分：
  - $\mathcal{D}_{\text{clean}} = \{(x_j, y_j)\}$：正确样本保留原始生成。
  - $\mathcal{D}_{\text{err}} = \{(x_i, y_i^-, y_i^+)\}$：错误样本保留原始错误回答 $y_i^-$，并用外部 LLM（Gemini 3.1 Pro）按最小干预原则生成修正回答 $y_i^+$（仅修改错误计算及下游结果，保留有效推理步骤）。
- 修正回答用于 error-correction 监督，干净样本用于行为保持监督。

### 4.2 错误电路发现
**Circuit units**：以 Transformer MLP 投影的输出维度为粒度，对每层 $l$ 与投影 $p \in \{\text{gate, up, down}\}$ 的每一输出行 $j$ 定义 unit $u=(l,p,j)$，输出 $n_{l,p,j}(\mathbf{h}) = \mathbf{w}_{l,p,j}^\top \mathbf{h}$。

**Keep-mask 参数化**：
$$c_u = \mathrm{clip}_{[0,1]}\!\Big(2\sigma\!\big(\tfrac{z_0-z_u}{\tau}\big)-1\Big),\quad m_u = 1-c_u$$
$z_u=z_0$ 时全开；$m_u=1$ 保留，$m_u=0$ 关闭。

**候选电路搜索**（冻结原模型，只训练 mask logits）：
$$\mathcal{L}_{\text{cand}} = \mathcal{L}_{\text{corr}} + \lambda_{\text{clean}}\mathcal{L}_{\text{clean}} + \lambda_{\text{keep}}\frac{1}{|\mathcal{P}|}\sum_q\frac{1}{|\mathcal{U}_q|}\sum_{u\in\mathcal{U}_q}m_u + \mathcal{L}_{\text{sens}}$$
- STE 把连续 $m_u$ 转为二值 mask $b_u$：$b_u^{\text{ste}}=m_u+\mathrm{sg}(b_u-m_u)$，前向使用离散干预，反向通过 $m_u$ 回传梯度；每模块设删除预算 $\rho_0$。
- 训练 10 epochs，保留最佳 feasible checkpoint，得到候选电路 $\mathcal{C}_{\text{cand}}$。

**RL-based 精炼**（从 $\mathbf{b}^{(0)}$ 初始化，冻结权重）：
- 对每题采样 $K=4$ 条 masked-model rollout $\{\hat{y}_{i,k}^-\}$，用答案正确性 $r_{\text{ans}}$ 与外部推理评判 $r_{\text{rea}}$（Qwen2.5-14B-Instruct）得到组内归一化优势：
$$A_{i,k}=\frac{\lambda_{\text{ans}}r_{\text{ans}}+\lambda_{\text{rea}}r_{\text{rea}}-\bar{r}_i}{\max(s_i,\epsilon)}$$
- 序列级目标（group-relative policy gradient）：
$$\mathcal{L}_{\text{RL}}=-\mathbb{E}_i\!\left[\frac{1}{K}\sum_k A_{i,k}\frac{1}{T_{i,k}}\sum_t \log p_{\theta,\mathbf{b}^{\text{ste}}}(\hat{y}_{i,k,t}\mid x_i,\hat{y}_{i,k,<t})\right]$$
- 辅助：对 $y_i^+$ 加 NLL（当组内奖励方差小时提供稳定信号）；对 $\mathcal{D}_{\text{clean}}$ 加 NLL；梯度冲突时投影掉冲突分量；更新仅在同时满足纠错与干净性能约束时接受，否则回滚到上一 feasible mask。
- 精炼后期渐进收紧删除预算以缩小电路。

**Circuit Pruning**：按 $c_u$ 由弱到强分组测试 reopen，条件：
$$S_{\text{err}}(\mathbf{b}')\ge S_{\text{err}}(\mathbf{b}^{\text{RL}})-\epsilon_{\text{err}},\quad S_{\text{clean}}(\mathbf{b}')\ge S_{\text{clean}}(\mathbf{b}^{\text{RL}})-\epsilon_{\text{clean}}$$
剩余 closed units 构成最终电路 $\mathcal{C}^*=\{u\mid b_u^*=0\}$。

### 4.3 电路验证与受限微调
**Ablation 验证**：比较原模型与 $\mathcal{G}\setminus\mathcal{C}^*$ 在 error/clean 集上的 exact-match 准确率。
**Circuit-restricted tuning**：仅对选定行的 gate/up/down 投影引入零初始化 delta（含 bias），冻结全部原参数；在 $\mathcal{D}_{\text{err}}$ 上以修正回答监督，末端答案 token 赋予更高权重 $w_t$：
$$\min_{\Delta\theta(\mathcal{C}^*)}-\mathbb{E}_{(x,y^+)\sim\mathcal{D}_{\text{err}}}\frac{\sum_t w_t\log p_{\theta+\Delta\theta(\mathcal{C}^*)}(y_t^+\mid x,y_{<t}^+)}{\sum_t w_t}$$
最佳 delta 合并回原参数。

## 实验与结果
### 设置
- 模型：Qwen3-8B、Llama-3.1-8B-Instruct（BF16，seed=42）。
- 目标数据：GSM8K 自定义 5 类 pattern-specific 集 + 1 个 heterogeneous 集；MedMCQA 跨域集；100 条 rephrased 示例用于迁移测试。
- 基线：base、LoRA、supervised-only baseline（跳过 RL 精炼与剪枝）。
- 控制：10 个密度匹配的随机电路 mask。
- 评估：GSM8K、MATH-500、MedMCQA + 8 个非目标基准（ARC-Challenge、HellaSwag、Wino-Grande、PIQA、TruthfulQA-MC1、MMLU-HS Math/Stats、BBH Multi-Step Arithmetic），汇总为 Aggregate Evaluation。

### 主要结果
**Heterogeneous GSM8K 修复（Sec 5.2）**
- 消融：Qwen3-8B 0%→50.5%，Llama-3.1-8B 6.0%→69.0%；电路占比 1.49% / 1.40%。
- 微调后 GSM8K：Qwen3-8B 80%→91%（+11pp，LoRA 88%，supervised-only 87%）；Llama-3.1-8B 73%→77%（supervised-only 停于 73%，LoRA 反降至 50%）。
- MATH-500：Qwen3-8B 75%→78%；Llama-3.1-8B 48%→47%（supervised-only 42%，LoRA 29%）。
- Aggregate Evaluation：Qwen3-8B 66.63%→66.63%（完全保持）；Llama-3.1-8B 62.38%→61.25%（supervised-only 60.0%，LoRA 56.6%）。

**Pattern-specific 修复（Sec 5.3, Table 1）**
- Qwen3-8B 5 类：0–4%→76–94%，增益 74–91pp，电路 0.26–0.44%；Aggregate 偏差 ≤0.75pp。
- Llama-3.1-8B 5 类：1–36%→92–95%，增益 56–94pp，电路 0.60–0.74%；Aggregate 偏差 ≤1.25pp。
- 随机电路控制最高仅 30%，均值 0–13.5%，证明修复来自功能归因而非稀疏性。

**跨重述迁移（Sec 5.4, Table 2）**
- 不重定位、不重新微调，直接用原电路在重述集上消融：
  - Qwen3-8B 平均 +15.0pp（+8~+23pp）；Llama-3.1-8B 平均 +46.2pp（+18~+66pp），说明电路因果效应跨表层表述迁移。
- 重定位后可达 94–96%，但直接迁移已显著优于 base，证明并非仅学到词汇模板。

**顺序修复（Sec 5.5, Fig 4）**
- Qwen3-8B 从 1.8% 累进到 90.8%；Llama-3.1-8B 从 19.6%→92.6%；前期修复最大下降 ≤8pp。

**跨域 MedMCQA（Sec 5.6, Table 3）**
- 消融：Qwen3-8B 3.0%→78.5%，Llama-3.1-8B 0.0%→81.0%，电路 1.47%/1.44%。
- 微调后 MedMCQA：两者均 60%/59%，Aggregate 分别为 66.75%/59.75%，LoRA 反而降低至 64.75%/58.88%。GSM8K 交叉保持：Qwen3-8B 80%→86%，Llama-3.1-8B 73%→72%（LoRA 分别为 85%、71%）。
- 全设定 Aggregate Retention 均值 99.4%。

## 相关工作脉络
1. **Circuit Discovery（Path Patching/ACDC/Attribution Patching/IBCircuit/Sparse Feature Circuits）**：侧重行为保持与能力复现的稀疏子图发现；RESCUE 转攻"错误归因+定向修复"，避免穷举干预。
2. **Task circuits（Yao et al. 2024; Hong et al. 2025; Hou et al. 2023）**：定位维持成功行为的电路；本文相反——定位导致失败的行为并抑制/修编它。
3. **Trustworthy/Safety circuits（Yu et al. 2026; Zhao et al. 2025; Yu et al. 2025b）**：聚焦对齐与后门；RESCUE 覆盖更广泛的数学/医学推理失败。
4. **幻觉/错误神经元归因（H-Neurons Gao et al. 2025; SAT Probe; FAITH; ReDeEP）**：多停留在相关性或单元级；本文以输出维度级 row 粒度构造稀疏电路并在多 rollout RL 下验证因果效应。
5. **推理期干预（ITI/TruthX/DoLa）**：通过内部信号在 decoding 阶段修正输出；本文通过稀疏电路微调改变模型内部计算结构，属 parametric repair。
6. **LoRA / 直接微调**：作为参数效率基线，但本文证明其会在跨域与非目标基准上造成更大退化，凸显稀疏归因修复的优势。

## 局限性与未来方向
- 修正数据依赖外部 LLM（Gemini 3.1 Pro）的最小干预改写，其质量直接影响 $\mathcal{D}_{\text{err}}$ 的可靠性；对复杂多步推理，"最小修改"的边界不够清晰。
- RL 精炼需对每题生成 $K=4$ 条 rollout，成本高于纯 SFT，且 reward 由外部评判器给出，存在评估器偏差风险。
- 实验集中在 8B 级指令模型与两类特定模式（数学/医学），在更大参数规模、更多样任务（代码、对话、长程规划）下的可迁移性待验证。
- 电路层深分布呈现模型特异性（Llama 偏中后期，Qwen 较分散），尚缺统一的层级组织理论。
- 跨重述迁移效果差异大（如某些模式直接迁移增益小），暗示同一电路对词汇变异鲁棒性不均。
- 顺序修复虽能累积，但仍出现 ≤8pp 的先前修复退化，多轮迭代后的稳定性需进一步刻画。

## 研究启发与可借鉴点
1. **SFT+RL 两阶段 mask 优化范式**可迁移至其他"定位并修复某类行为"的 interpretability 任务：先用参考答案做冷启动，再用生成级相对优势精炼，缓解 offline reference 的分布偏移。
2. **组内归一化优势 + 双约束（纠错/保持）的 accept/reject 机制**是稳定稀疏电路学习的实用技巧，可推广至安全电路、偏好矫正等场景。
3. **密度匹配随机电路对照**是证明"归因有效性"而非"稀疏性有效性"的标准实验设计，建议在同类稀疏干预工作中标配。
4. **跨重述迁移评估**为电路的因果泛化提供了比 single-instance ablation 更强的证据链，可复用于验证其他 circuit 方法的语义抽象程度。
5. **Circuit-restricted zero-initialized delta 的微调形式**兼顾参数效率与可组合性（sequential repair），适合后续与 PEFT、model merging、capability editing 等工作结合。

## 关键术语表
- **RESCUE**：Reasoning-Error Sparse-Circuit Uncovering and Editing，本文提出的错误电路定位与修复框架。
- **Keep-mask / closure strength**：$m_u\in[0,1]$ 表示保留强度，$c_u=1-m_u$ 为关闭强度；连续松弛配合 STE 实现可微的稀疏电路搜索。
- **Circuit-restricted tuning**：仅在定位到的错误电路对应投影行上引入零初始化 trainable delta，冻结其余参数。
- **Group-relative advantage**：对同题生成的 $K$ 条 rollout 进行奖励组内归一化 $(r_i-\bar{r}_i)/\max(s_i,\epsilon)$，作为 RL 精炼信号。
- **Aggregate Evaluation**：8 个非目标基准（ARC、HellaSwag、Wino-Grande、PIQA、TruthfulQA、MMLU-Math/Stats、BBH-MSA）的无加权宏平均。
- **Straight-Through Estimator (STE)**：前向用二值 mask、反向通过连续 logit 传梯度，保证干预与优化的离散一致性。
- **Cross-rephrase transfer**：在原表述上定位电路后，直接在词汇重写但逻辑不变的测试例上消融，检验因果效应的表层不变性。
- **Sequential repair**：按阶段依次定位并修编新模式错误，每步以前一步输出模型为起点，检验修复可累积性。

## 可复现要素
- **数据集**：GSM8K、MATH-500、MedMCQA、ARC-Challenge、HellaSwag、Wino-Grande、PIQA、TruthfulQA-MC1、MMLU（HS Math/Stats）、BBH（MSA）均为公开基准；文中自定义的 5 类 pattern-specific 修复集与 rephrased 集由 Gemini 3.1 Pro 生成，论文未声明独立开源数据集。
- **代码/权重**：代码已开源（匿名仓库，见 Reproducibility Statement）；包含 SFT mask 优化、RL 精炼、剪枝、ablation、circuit-restricted tuning 与评测脚本。**预训练/基础模型权重**为公开模型（Qwen3-8B、Llama-3.1-8B-Instruct），**定位所得电路 mask 与 tuned delta 未声明单独开源**。
- **关键超参**：mask temperature $\tau$、阈值 $\eta_0$、每模块删除预算 $\rho_0$；SFT 候选搜索 10 epochs，RL 精炼 6 epochs，每题 $K=4$ 条 rollout；$\lambda_{\text{ans}},\lambda_{\text{rea}}$ 平衡答案与推理奖励；剪枝容忍 $\epsilon_{\text{err}},\epsilon_{\text{clean}}$；最终 $\Delta\theta(\mathcal{C}^*)$ 微调以答案 token 加权 NLL 为目标。精确数值见附录与开源仓库。
