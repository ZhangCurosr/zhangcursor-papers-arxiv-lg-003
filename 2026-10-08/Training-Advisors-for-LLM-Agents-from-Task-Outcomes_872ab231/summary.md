---
title: "Training-Advisors-for-LLM-Agents-from-Task-Outcomes"
source: https://arxiv.org/pdf/2610.09858v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:14:02"
field: "LLM Agent 训练与迁移"
keywords: ["LLM agent", "critic training", "outcome-based RL", "transfer learning", "multi-step reasoning", "DAPO", "natural language feedback"]
innovations: ["只用最终任务结果训练生成式批评者，无需步骤级标注与参考批评", "单模型训练的 critic 跨不同基础模型与跨任务域仍能带来显著增益", "与 actor GRPO 微调正交，可在已收敛 generator 上叠加 critic 进一步提升成败率"]
benchmarks: ["MuSiQue", "ALFWorld", "DeepDive", "tau^3 Retail", "tau^3 Airline"]
---

# 论文速读：Training-Advisors-for-LLM-Agents-from-Task-Outcomes

## 一句话总结
提出 **Caddie**，一种让大语言模型（LLM）智能体在任务执行过程中获得自然语言批评与建议的训练方法：只依赖智能体的**最终任务结果**作为奖励信号，通过强化学习（DAPO）优化批评者（critic）模型，而冻结基础模型（base model）的权重。一个在 MuSiQue 上训练的 4B 批评者无需重新训练，即可把多种不同规模与架构的基础模型的成败率提升 10+ 个百分点，并在跨任务域（DeepDive、τ³）实现无微调增益。

## 研究问题与动机
- **LLM 智能体在执行多步任务时容易卡住或迷失目标**：模型在交互循环中可能重复无效动作、错过关键证据，或丢失对初始目标的跟踪。
- **已有反馈方法对标注或搜索代价敏感**：过程奖励模型（PRM）需要大量步骤级人工标注或自动标签；在交互式环境中，"某一步是否正确"本身很难界定（某些看起来"低效"的动作可能是探索环境所必需的）。
- **纯提示式批评效果有限**：即使给基础模型同版本提示词，没有经过基于下游结果的强化优化的批评者，其建议质量仍不稳定。
- **跨基础模型与跨任务域的泛化目标未经验证**：此前训练批评者的工作多数仅在单一模型/任务上评估，缺乏对迁移性的系统论证。

## 核心贡献（创新点）
1. **提出 Caddie，用最终任务结果直接训练生成式批评者**：仅通过冻结基础模型在干预后产生的轨迹终态奖励来更新批评者，不需要任何步骤级人工标注，也不依赖合成参考批评。
2. **首次系统验证"单模型训练的批评者可跨多种基础模型与跨任务域迁移"**：用一个基于 Qwen3-4B 训练的 critic，同时把 Qwen3-30B-A3B、Qwen3.8-27B、Kimi K3 在 MuSiQue 上提升 9–13pp，并在 DeepDive、τ³ Retail/Airline 上实现多处显著提升。
3. **展示批评干预可与基础模型微调（GRPO）互补**：在已将 Qwen3-4B 用 DAPO/GRPO 训练到 plateau 的情况下再训练 critic，于 MuSiQue-attached 再提升 13.88pp（49.69% → 63.56%），说明两种训练范式并非互斥。
4. **比较并给出三种干预协议的定位**：窗口化、概率化与自调用（self-call）三个协议各有优劣；本文证明自调用协议最利于跨模型/跨任务迁移，因其不依赖外部固定调度。

## 方法详解
- **智能体–环境交互设定**：任务 $x \sim \mathcal{D}$，冻结基础策略 $\pi_b$ 在环境 $\mathcal{E}$ 中按 $a_t \sim \pi_b(\cdot \mid s_t)$、$o_t \sim \mathcal{E}(\cdot \mid s_t, a_t)$ 采样生成轨迹 $\tau$，在 $T$ 步结束时获得任务奖励 $R(\tau)$。
- **批评者策略**：可训练参数为 $\theta$ 的 LM $\pi_c^\theta$，在选定的中间状态 $s_k$ 采样 $G$ 条自然语言批评 $c^{(i)}$。
- **三套干预协议**：
  1. 窗口化（windowed）：从预设窗口 $\mathcal{K}$ 中均匀采样 $k$，到达该步强制干预。
  2. 概率化（probabilistic）：每步以概率 $p$ 独立插入批评。
  3. 自调用（self-call）：基础模型动作空间扩展出 `call_critic` 工具，模型自行决定何时求助，最多 3 次/轨迹。
- **基于结果的奖励估计**：每条批评 $c^{(i)}$ 后拼接进 $s_k$，由冻结基础模型产生 $N$ 条独立延续轨迹 $\tau^{(i,n)}$，以均值 $R_i = \frac{1}{N}\sum_n R(\tau^{(i,n)})$ 作为该批评的奖励。
- **DAPO 优化**：采用 GRPO 类优势归一化 $\hat{A}_i = (R_i - \mu_R)/\sigma_R$，在批评 token 上计算 clipped surrogate objective，仅更新 $\theta$；丢弃组内方差为零的 group，动态采样补齐批次。
- **格式惩罚**：批评输出若不符合"Analysis / Advices for next steps"两节结构则给 $-1$ 奖励且不运行基础模型延续，防止批评者退化为通用 filler。
- **避免退化为"直接解题"**：在需要多步工具交互的任务上，批评者难以一次给出完整可执行序列，从而被迫学习"诊断+建议"而非"代答"；但作者也明确警告，在单步可解任务中这一风险会显现。

## 实验与结果
- **数据集/环境**：MuSiQue（attached/wiki 两检索设置）、ALFWorld；跨域评估用 DeepDive、τ³ Retail 与 τ³ Airline。
- **训练设置**：critic 均以 Qwen3-4B-Instruct-2507 初始化，各任务单独训练；MuSiQue 训练至 2000 步、ALFWorld 至 800 步后固化 checkpoint，不在测试时重选。8 GPU H100，batch=128，学习率 $10^{-6}$，clip $(0.20, 0.28)$。
- **主要结果（Qwen3-4B 基础模型）**：
  - MuSiQue-attached：no critic 21.56% → **Caddie 46.88%**（+25.31pp）；wiki：10.38% → 21.13%（+10.75pp）；ALFWorld：23.79% → **49.72%**（+25.93pp）。
  - Caddie 4B critic 全面超越同参数 prompted Qwen3-4B 与 kimi K3 prompted critic。
- **跨基础模型（MuSiQue 训练、其他基础模型评估，不重训）**：
  - attached：Qwen3-30B-A3B 28.44%→41.63%（+13.19pp）、Qwen3.8-27B 46.63%→55.94%（+9.31pp）、Kimi K3 38.56%→51.00%（+12.44pp）。
  - wiki：Kimi K3 20.06%→26.25%（+6.19pp）、Qwen3-30B-A3B 19.94%→20.75%（+0.81pp）。
- **跨任务域（MuSiQue 训练、DeepDive/τ³ 直接评估）**：
  - Qwen3.8-27B + wiki critic：DeepDive 48.83%→49.80%、τ³ Retail 50.31%→58.12%（+7.81pp）、τ³ Airline 74.38%→78.12%（+3.74pp）。
  - 6 项跨任务组合中有 5 项以 Caddie 取得最高均值。
- **对已微调基础模型的二次增益**：Qwen3-4B 经 DAPO/GRPO 训至 49.69% 后，再配 Caddie critic 达到 **63.56%**（+13.88pp）。

## 相关工作脉络
- **RL4F / CTRL / Critique-RL**：均要求预训练阶段引入人工或程序合成参考批评、或在"判断→生成"两阶段优化，Caddie 去掉这两类监督，直接用终态结果反向更新批评 token。
- **Advisor Models（Asawa et al., 2026）**：同样仅用学生最终任务奖励，但在单次提示后即给分，且建议频率固定（每 5 步一次）；Caddie 改为从中间状态分叉比较 $G$ 条续路，且允许 self-call 自主决策时机。
- **Agent-RRM**：教师生成的 reasoning/critique 训练 generative reward model，属于"事后评价+排序"范式，而 Caddie 属于"在线指导+条件生成"。
- **Steer, Don't Solve / Welleck 等修正方法**：训练 separate corrector 直接改写已生成输出，不面向持续交互中的实时建议。
- **GLRe / Retroformer**：偏全局修订或对失败轨迹的反思（post-hoc），与 Caddie 在同一 attempt 中途介入形成对比。
- **ECHO / PivoARL / ICRL**：训练过程中 actor 与 critic 联合演化，Caddie 坚持冻结 actor，降低训练耦合与部署复杂度。

## 局限性与未来方向
- **单步/封闭任务下的退化风险**：当任务可用一次完整解答直接得分时，critic 会被优化成"替代 base 直接解题"的策略，损失跨模型/跨任务的迁移性。
- **训练代价依赖基座延续数**：每一步需 $N \times G$ 条独立基座 rollout；对长轨迹、贵工具调用、大基座模型，算力与状态回滚成本较高。
- **环境恢复限制**：多分支比较要求能恢复到相同中间状态，在具外部副作用的工具场景中可能难以实现。
- **critic 规模未验证**：全文只用 Qwen3-4B 作 critic，更大 critic 的迁移/效率关系未知。
- **未比较 generative 批评 vs. scalar 判别式验证**：两者在相同推理预算与调用策略下谁更优尚无定论。
- **未来方向**：联合微调 actor、探索 critic 规模、开发 counterfactual/baseline 变体 reward、以及将 scalar verifier 与 generative advisor 融合。

## 研究启发与可借鉴点
1. **用终端任务结果做 critic 的强化学习信号是最简可行的通用路径**：不需要步骤标注、也不需要参考文本，只要任务本身有可复现的终态评判（success/failure）。
2. **冻结 actor + 仅训练 critic 有利于跨模型部署**：同一 critic checkpoint 可直接用于不同规模、不同架构的 base model，无需重新适配。
3. **self-call 工具接口是跨任务迁移的关键设计**：比固定窗口/概率触发更能适应不同模型的决策节奏，避免"在不需要帮助时打扰"与"该求助时未求助"两类错误。
4. **与已有 Actor 微调正交可叠加**：GRPO 训完 generator 后再接 critic 能再拿 +13.88pp，提示可形成"先调 actor、再配 advisor"的分阶段 pipeline。
5. **格式奖励（format penalty）能抑制无效输出**：对 malformed 批评给 -1 且不运行基座延续，在实验中明显引导出"Analysis / Advices"结构化输出。

## 关键术语表
- **Caddie**：本文方法名，指"以任务终态为奖励、对冻结基础模型生成式批评"的训练框架。
- **Critic / Advisor**：在 agent 轨迹中间给出自然语言诊断与建议的辅助模型，区别于事后评分器。
- **Self-call intervention**：基础模型通过新增 `call_critic` 工具自行决定是否求助的交互协议。
- **DAPO**：基于 GRPO 的 open-source LLM 强化学习优化器，使用非对称 clip、token 级损失聚合与动态 batch 采样。
- **Group-normalized advantage**：将同一状态下采样的 $G$ 条批评奖励做均值/标准差归一化，得到每个批评的优势值 $\hat{A}_i$。
- **Terminal outcome reward**：以整条轨迹的最终二值成败作为批评的标量奖励，忽略中间过程。
- **Intervention protocol**：决定在轨迹的哪些步骤插入批评的策略（窗口/概率/自调用三种）。
- **OOD transfer**：指批评器在训练集之外（新 base model、新任务域）仍能带来性能增益的能力。

## 可复现要素
- **数据集**：MuSiQue（训练 split 由作者在文中提供）、ALFWorld（json_2.1.1）、DeepDive、τ³ Retail/Airline（继承自 τ²-bench 预定义测试集）。
- **代码/权重**：论文以 DAPO 为优化基座，但未在正文声明公开 critic checkpoint 或完整训练脚本；具体 prompt、超参在 Appendix D/G 给出（learning rate $10^{-6}$、clip 0.20/0.28、batch 128、temperature 1、top-p 1、critic 输出上限 512 token、上下文 8192/32768 等），理论上可复现但需自行实现环境回放与多分支 rollout。
- **关键超参**：$G=16$、$N=1$（主实验）、mu/sigma 归一化、训练步数 2000（MuSiQue）/800（ALFWorld）。
