---
title: "VACE-Validation-Gated-Alternating-Co-Evolution-of-Agent-Mode"
source: https://arxiv.org/pdf/2609.37105v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:41:33"
field: "Agent RL 与系统协同优化"
keywords: ["agentic reinforcement learning", "model-harness co-evolution", "validation gating", "harness optimization", "LLM agents", "trajectory-driven refinement"]
innovations: ["提出验证门控交替共进化框架，将 agentic RL 与轨迹驱动 Harness 精炼闭环连接", "引入 checkpoint-specific 验证门控机制，仅在更新后模型权重下严格改进才采纳修订", "系统性对比五种基线与调度变体，揭示 17/44 修订导致退化并需门控拦截"]
benchmarks: ["OfficeQA", "AutomationBench"]
---

# 论文速读：VACE: Validation-Gated Alternating Co-Evolution of Agent Models and Harnesses

## 一句话总结
VACE 提出了一种验证门控交替共进化框架，通过交替执行智能体强化学习（RL）与轨迹驱动的 Harness 精炼，并在每个 RL 阶段结束后对候选 Harness 更新以更新后的模型权重为基准进行验证，仅当验证表现提升时才采纳。在 Qwen3.5-9B 上，VACE 于 OfficeQA 和 Automation-Bench 均显著超越纯权重 RL 与非门控交替方法。

## 研究问题与动机
- **模型权重与 Harness 存在耦合交互**：Agent 性能同时取决于模型权重（决定如何推理和决策）和 Harness（指令、技能、工具接口等执行系统），单独优化其一会忽略两者的协同效应——例如训练后更新的模型可能对早期设计的 Harness 产生误用，而更优的 Workflow 可能暴露模型尚未学会的决策。
- **离散 Harness 修订不一定提升性能**：基于轨迹诊断提出的 Harness 修改（如要求更多检索步骤）在更新后的模型 checkpoint 下可能反而降低验证得分（如消耗交互预算），因此需要独立的"修订生成"与"采纳决策"两个阶段。
- **已有方法缺乏同步验证机制**：WHALE 等交替优化方法直接提交每轮 Harness 输出，未在新模型权重下验证；SIA 等方法先优化 Harness 再固定执行 RL，未实现真正交替。
- **缺少在统一实验设定下对比单组件优化与联合优化的系统实验**：本文在相同权重/ Harness 优化器、基座模型和评测集下对比四种条件（静态、Harness-only、RL-only、VACE 等），量化了协同优化的真实收益。

## 核心贡献（创新点）
- **提出 VACE 交替共进化框架**：将 agentic RL 与基于轨迹的 Harness 精炼闭环连接，每轮依次执行模型训练→轨迹提取修订→验证门控采纳。与 SIA/Co-Harness 等相比，VACE 的核心差异在于每个修订都在更新后的 checkpoint 上显式验证后再决定是否进入下一轮 RL。
- **引入 checkpoint-specific 验证门控机制**：在相同更新后权重下公平比较 incumbent 与 candidate Harness 的验证得分，仅严格改进时采纳（平局保留 incumbent）。与 WHALE 风格的无门控交替相比，有效避免了有害修订污染后续训练。
- **系统性实验对比五种基线与调度变体**：在统一设置下证明联合适应优于单组件优化（RL-only 高 6.43/9.09 pp；Harness-only 高 9.98/13.58 pp），以及门控交替优于无门控交替（WHALE-style 高 4.59/6.95 pp）。
- **揭示并量化了 17/44 的 Harness 修订在对应 checkpoint 上导致验证退化**：通过 Phase-level 轨迹分析证明"修订生成"和"采纳决策"应当分离的设计必要性。

## 方法详解
- **Agent 状态表示**：将 Agent 建模为二元组 $(W, H)$，其中 $W$ 为模型权重，$H$ 为 Harness 配置。任务 $x$ 在 $(W, H)$ 下产生轨迹 $\tau \sim p_{W,H}(\cdot|x)$，优化目标为最大化期望回报 $J(W, H) = \mathbb{E}_{x \sim \mathcal{P}}\mathbb{E}_{\tau \sim p_{W,H}(\cdot|x)}[r(x,\tau)]$。
- **交替更新循环**（Algorithm 1）：第 $t$ 轮执行：
  1. **模型训练阶段**：在固定 $H_t$ 下，模型优化器 $\mathcal{M}$ 对 $D_{\text{train}}$ 执行 agentic RL，产出 $(W_{t+1}, \mathcal{T}_t)$，$\mathcal{T}_t$ 包含该轮执行的轨迹及失败样本。
  2. **Harness 修订提议**：Harness 优化器 $S$ 复用 $\mathcal{T}_t$（无需额外收集）基于失败轨迹的诊断证据提出修订 $H_t' = S(H_t, \mathcal{T}_t)$，来源于 HarnessEvolve 的轨迹驱动精炼。
  3. **验证门控**：固定 $W_{t+1}$，在相同验证集 $\mathcal{V}$ 上比较：$b_t = \widehat{R}_\mathcal{V}(W_{t+1}, H_t)$ 与 $c_t = \widehat{R}_\mathcal{V}(W_{t+1}, H_t')$，若 $c_t > b_t$ 则采纳 $H_{t+1} = H_t'$，否则保留 $H_{t+1} = H_t$。
- **实现细节**：使用 Uni-Agent 连接执行、轨迹收集、RL 训练与 Harness 精炼；RL 采用 GRPO + DAPO-style oversampling/零优势过滤，仅使用 outcome rewards（OfficeQA 二值正确性，AutomationBench 密集 partial credit）。Harness 编辑范围限定于技能（skills）和执行指导（execution guidance），评估器、工具语义和环境转移规则固定。
- **验证集复用机制**：验证集用于每轮 Harness 采纳决策，测试集不参与优化循环。

## 实验与结果
- **数据集与任务**：
  - **OfficeQA**（文档推理）：84 train / 53 val / 109 test，Binary accuracy，基于 Qwen3.5-9B。
  - **AutomationBench**（工作流）：HR 65 / Marketing 72 / Finance 72 共 209 题，取自 HR 26/15/24、Marketing 27/22/23、Finance 27/21/24；评估部分信用（partial credit）。
- **基线方法**：Static agent、Harness-only、RL-only、SIA-style（先优化 Harness 固定后做 RL）、WHALE-style（交替但无门控）。
- **主要结果（Table 2）**：
  - **OfficeQA 测试准确率**：VACE **45.26%** vs RL-only 38.83%（+6.43 pp）vs WHALE-style 40.67%（+4.59 pp）vs Static 32.11%（+13.15 pp）。
  - **AutomationBench 平均 partial credit**：VACE **75.19%** vs RL-only 66.10%（+9.09 pp）vs WHALE-style 68.25%（+6.95 pp）vs SIA-style 65.35%（+9.85 pp）vs Static 50.26%（+24.94 pp）。
  - VACE 在 AutomationBench 的 HR 域得分 75.18%，超过所有基线。
- **验证门控有效性（Table 3）**：OfficeQA 12 次 Harness 试验中 7 采纳/2 平局/3 拒绝；AutomationBench 32 次中 18 采纳/0 平局/14 拒绝（**17/44 修订导致验证退化被门控拦截**）。
- **验证曲线（Figure 3）**：OfficeQA 从 30.28% 升至最高 60.37%（保留 47.17%）；AutomationBench 从 53.08% 升至最高 81.42%；优化轨迹非单调，权重阶段可能导致验证分下降。

## 相关工作脉络
- **SIA [Hebbar et al., 2026]**：使用反馈代理更新 Harness 和模型权重，先在 Harness 侧优化再做 RL。VACE 与之区别：真正交替而非先 Harness 后权重，且每轮修订均在更新后的 checkpoint 验证。
- **Co-Harness [Chen et al., 2026b]**：交替进行基于失败的 Harness 精炼与成功轨迹监督训练，也有验证环节。VACE 与之区别：基于 agentic RL（而非 SFT）更新权重，且使用严格改进门控（Co-Harness 为失败驱动）。
- **WHALE [Kim et al., 2026]**（并发独立工作）：交替 online rejection-sampling FT 与 executable-harness search，有 Meta-Harness 内部评估。VACE 与之区别：VACE 在更新后 checkpoint 做显式验证门控，WHALE 直接提交每轮 Harness 输出，无门控。
- **HarnessEvolve [Jiang et al., 2026]**：基于轨迹诊断与参考引导错误分析提出 Harness 修订。VACE 将其作为 $S$ 的实例化，并新增验证门控和与 RL 的交替闭环。
- **GEPA [Agrawal et al., 2025] / Reflexion [Shinn et al., 2023] / ExpeL [Zhao et al., 2023]**：纯 Harness/prompt 优化，不涉及模型权重更新。VACE 将这些思路与 agentic RL 联合训练结合。
- **Agent Lightning [Luo et al., 2025] / Search-R1 [Jin et al., 2025] / WebAgent-R1 [Wei et al., 2025]**：agentic RL 权重训练方法。VACE 在此基础上引入 Harness 协同优化。

## 局限性与未来方向
- **单一模型规模与评测集**：仅使用 Qwen3.5-9B 和两个 agent benchmark，Harness 编辑局限于技能与执行指导，更广泛的模型规模和任务环境有待探索。
- **计算开销**：Harness 精炼和验证步骤增加了额外计算成本。
- **验证集复用与自适应选择偏差**：Repeated use of validation set 可能引入 adaptive selection effects（Dwork et al., 2015）。
- **固定交替调度**：当前采用固定轮次交替，未来可探索基于任务结果和训练进度驱动的动态调度。
- **评估噪声**：严格改进门控依赖点估计，存在 noisy evaluation 导致的误判风险。

## 研究启发与可借鉴点
- **验证门控设计可迁移**：任何涉及离散参数/配置优化的框架（prompt engineering、tool selection、workflow scheduling）均可借鉴"修订生成→checkpoint 特定验证→采纳决策"的两阶段分离设计，避免有害修改污染后续训练。
- **轨迹复用兼顾 RL 数据与诊断信息**：将 RL 阶段收集的轨迹直接用于 Harness 修订诊断，无需额外 rollout 即可获取失败模式证据，提升样本效率——此思路可扩展至其他 LLM 系统优化场景。
- **交叉评估设计（same-checkpoint cross-evaluation）**：在相同模型权重下公平比较不同 Harness 变体的策略，可精确分离"模型改进"与"配置改进"的贡献，值得在其他 agent 优化论文中作为标准分析手段。
- **与团队方向结合机会**：本团队的 agent 训练 pipeline 若引入 VACE 式交替共进化（特别是验证门控模块），有望在当前 RL-only 或 prompt-tuning 基础上进一步压榨性能；可优先在 OfficeQA/AutomationBench 类似的工具调用密集型任务上验证。
- **调度优化研究空间**：固定交替调度 → 动态调度（由任务反馈决定何时更新哪个组件）是一个自然的后续研究方向，可与现有 RL 调度策略（如 curriculum learning）结合。

## 关键术语表
- **VACE**：Validation-Gated Alternating Co-Evolution 的缩写，本文提出的交替共进化框架，通过验证门控连接模型权重 RL 更新与 Harness 精炼。
- **Harness**：驱动 agent 执行任务的外部系统配置，包括指令、可复用技能（skills）、工具接口和执法规则，区别于模型权重本身。
- **Validation-gated harness update**：在更新后的模型 checkpoint 上比较 incumbent 与 candidate Harness 的验证得分，仅严格改进时采纳，平局保留 incumbent。
- **Trajectory-driven harness refinement**：利用 RL 阶段收集的执行轨迹（含失败样本）进行诊断分析，自动提议对 Harness 的针对性修订。
- **Agentic RL**：面向 agent 的强化学习，以任务 outcome 为 reward 信号，通过多轮环境交互优化模型权重。
- **GRPO**：Group Relative Policy Optimization，本文采用的 RL 算法，结合 DAPO-style 的 group 零优势过滤机制。
- **OfficeQA**：Databricks 发布的文档 grounded 推理 benchmark，binary correctness 评分。
- **AutomationBench**：涵盖 HR/Marketing/Finance 领域工作流任务的 agent benchmark，使用 dense partial credit 评分。

## 可复现要素
- **数据集**：OfficeQA（来自 Databricks AI Research blog，公开）；AutomationBench（arXiv:2604.18934，公开）；论文使用了自定义划分（OfficeQA: 84/53/109；AutomationBench: 80/58/71）。
- **代码/权重**：使用了 Uni-Agent 框架（GitHub: verl-project/uni-agent），论文未明确声明 VACE 代码开源。
- **关键超参**：RL 使用 GRPO + DAPO-style oversampling/过滤；仅用 outcome rewards；Harness 编辑范围限于 skills 和 execution guidance；基座模型 Qwen3.5-9B。其他超参（如 $B_W$、$B_H$、rollout 数、学习率）论文未详细列出，标注为"论文未提及"。
- **评估方式**：每个方法选取验证集最高分对应的 checkpoint 和 harness，报告三次独立测试评估的均值；相同 task IDs 和 evaluator 下进行 incumbent/candidate 配对比较。
