---
title: "SELFSEARCH-REWARD-FREE-SEARCH-FOR-SELF-IMPROVING-AGENTS"
source: https://arxiv.org/pdf/2609.37968v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:31:51"
field: "自主代理与自动化 agent 设计"
keywords: ["self-improving agents", "reward-free search", "LLM agent", "automated agent design", "experience-driven evolution", "SWE-bench", "Terminal-Bench"]
innovations: ["提出无需下游评估反馈的自我改进搜索流程，以历史 episode 记录作为改进信号", "双谱系（能力/自适应）共享只读记录并交叉复用工具的演化架构", "跨模型族可迁移且搜索成本仅 $4.03 的高成功率 harness"]
benchmarks: ["SWE-bench Verified", "SWE-bench Multilingual", "Terminal-Bench 2.1"]
---

# 论文速读：SELFSEARCH: REWARD-FREE SEARCH FOR SELF-IMPROVING AGENTS

## 一句话总结
论文提出 SelfSearch，一种无需下游任务评估的代理自我改进搜索流程，通过让代理基于历史自我改进 episode 记录（包含推理轨迹、工具操作与结果）自行修改自身代码与工具，在不依赖外部奖励信号的情况下持续提升任务解决能力与执行效率，并在 Terminal-Bench 2.1 上以仅 $4.03 搜索成本达到与最强公开 harness Codex 持平的 82.0% 成功率。

## 研究问题与动机
- 现有 LLM agent 自我改进方法依赖下游评估（如 SWE-bench 开发集）反复测试候选代理以引导搜索，导致搜索成本随任务数量和轮次增长而显著增加，且改进方向被绑定到特定评估任务分布上。
- 自我改进过程本身产生丰富经验记录（工具调用、推理路径、失败与修复过程），但现有工作未系统利用这些记录作为后续自我改进的引导信号，而是主要依赖外部评估反馈。
- 需要在不消耗下游评估预算的前提下，探索代理是否能从"如何改进自身"的经验中迁移到更好的任务求解能力和更高效的操作流程。
- 验证 reward-free 自我改进生成的 harness 是否具有跨模型泛化能力（不同模型族下是否仍能保持改进效果）。

## 核心贡献（创新点）
- **经验驱动的自我改进搜索机制**：提出 SelfSearch，让代理仅依靠历史 episode 记录（而非下游评估分数）进行迭代修改；与 DGM、HGM 等工作依赖外部评估选优的本质区别在于，改进信号来源于自我改进过程自身留下的轨迹证据。
- **双谱系共享记录的演化架构**：设计 capability（能力提升）与 adaptive（适应改进）两条并行谱系，各自接收另一谱系的 episode 记录并交叉复用已发现的工具与修复策略；相比 Hyperagents 中分离的 meta-agent 与 task agent，本文用同一代理同时承担任务求解与自我改进角色。
- **跨模型可迁移的无奖励改进结果**：在 GPT 和 DeepSeek 两种配置下，在 SWE-bench Verified、SWE-bench Multilingual 与 Terminal-Bench 2.1 上均实现群体平均成功率提升，且进化后的 harness 在未见过的执行模型下仍显著优于初始 agent；与评估引导搜索相比，在 110 题 SWE-bench Verified 和 SWE-bench Multilingual 上达到持平或更优成绩的同时，搜索成本降低 13.4%-53.2%。

## 方法详解
- **自我改进 episode 形式化**：第 k 代代理 $B_k$ 接收可编辑仓库副本与只读的历史 episode 记录集 $\mathcal{E}_k$，执行 `SelfImprove(B_k; E_k, d)` 产生第 $k+1$ 代代理 $B_{k+1}$ 与该 episode 记录 $e_k$；底层模型权重固定，仅仓库中的指令、工具、执行逻辑和代码组织结构可被修改。
- **episode 记录内容**：$e_k$ 包含推理轨迹（工具调用序列、调用参数与返回结果）、中间检查/验证结果与代码 diff；这些记录作为后续 episode 的经验输入，而非当前 episode 的执行环境。
- **双谱系搜索过程**：维护 capability 方向（识别能力不足并开发可复用工具/流程）与 adaptive 方向（改进失败时的策略调整与恢复机制）；第 1 代先分别以两个方向运行初始代理一次，保留 episode 记录但丢弃生成结果，使两条谱系从同一起点 $B_0$ 出发；此后每代将上一代两条谱系的记录共同注入，允许交叉复用工具与修复方案。
- **运行环境与隔离**：模型-工具交互循环运行在仓库外的固定 runtime 中，负责控制推理设置、token 上限、tool-step 预算与执行时间（每个 episode 最多 513 次 model call、512 次 tool step、4 小时），代理仅可通过 `self.respond(message)` 接口调用该循环；runtime 同时记录 trajectory 并保存为只读证据供后续 episode 访问。
- **搜索方向提示（design guidance）**：能力方向强调产生跨任务通用的可复用机制而非仅靠 instruction；自适应方向强调在证据与假设冲突时保留灵活性，避免固化工作流；两方向均要求改动需提高实际能力或可靠性，而非仅保持包可运行。
- **候选归档与评估分离**：搜索阶段不引入下游评测结果，所有世代产生的代理均存入候选库 $\mathcal{C}$；最终从第 10 代的两条谱系各取一个代理单独在下游基准（120 题 SWE-bench Verified、60 题 SWE-bench Multilingual、全量 89 题 Terminal-Bench 2.1）上评估，不基于开发集分数做选择。
- **成本度量**：平均执行成本 $\bar{c}(B) = \frac{1}{|\mathcal{T}|}\sum_{i\in\mathcal{T}} c_i(B)$ 计入成功与失败任务；对初始与进化代理共同解决的子集 $S_B$ 定义 $R_{\text{shared}}(B)$ 用于衡量纯执行侧成本变化，两者均不含搜索成本。

## 实验与结果
- **基准与模型配置**：GPT 配置使用 gpt-5.6-sol 作自我改进、gpt-5.6-luna 作下游执行；DeepSeek 配置使用 deepseek-v4-pro-0813 作改进、deepseek-v4-flash-0731 作下游；两配置均在 medium reasoning effort、temperature=1 下运行。
- **总体成功率提升**：六组模型-基准配置中，SelfSearch 群体均值均超过初始代理；Terminal-Bench 2.1 上 GPT 由 43.8% 升至 55.1%、DeepSeek 由 65.2% 升至 73.0%（个体最大提升 11.2pp）；SWE-bench Multilingual 最大增益 GPT 为 6.7pp、DeepSeek 为 5.0pp；SWE-bench Verified 上 DeepSeek 两谱系均由 81.7% 升至 86.7%。
- **执行成本下降**：DeepSeek 能力谱系在 Terminal-Bench 2.1 上提升成功率的同时降低平均成本 12.1%，自适应谱系降 16.9%；在 SWE-bench Multilingual 上 DeepSeek 自适应谱系提升 5.0pp 成功率的同时，对共同解决任务的执行成本降低 38.5%；SWE-bench Verified 上 DeepSeek 共同任务成本降幅达 19.7%-36.1%。
- **对比评估引导搜索**：在线性搜索（linear search）和档案搜索（archive search）的 Ten-task SWE-bench 开发集引导下十轮搜索中，SelfSearch 在 110 题 SWE-bench Verified 和 SWE-bench Multilingual 上持平或超越两者；搜索成本方面，GPT 下降低 13.4%-47.2%，DeepSeek 下降低 49.0%-53.2%（SelfSearch 仅 $4.03/$6.52 vs 基线 $7.53-$12.35）。
- **跨模型迁移**：用 GPT-5.6 Sol 搜索出的 harness 在 DeepSeek V4 Flash 上达到 83.3%/85.0%，优于初始 81.7%；用 DeepSeek V4 Pro 搜索出的 harness 在 GPT-5.6 Luna 上达到 79.2%/76.7%，优于初始 75.0%；跨族迁移时两谱系均持续优于基线。
- **与其他 harness 对比**：在 Apache Maka 公布的九 harness 比较（Terminal-Bench 2.1、DeepSeek V4 Flash、xhigh reasoning、扩大执行限制）中，SelfSearch 生成的 $B^c$ 以 $4.03$ 搜索成本解决 73/89 题（82.0%），与排名第一的 Codex 持平。
- **消融结果**：移除 episode 记录使 GPT/DeepSeek 群体均值分别下降 2.1pp/2.9pp；固定改进器（始终用 $B_0$ 执行所有修改）使群体均值分别下降 1.7pp/2.9pp；完整 SelfSearch 达到最高均值（GPT 76.2%、DeepSeek 86.7%）。
- **工具迁移到下游任务**：在 269 题下游任务中，GPT 与 DeepSeek 代理分别以 8.6%/23.6% 的任务调用 text-search、以 40.3%/33.8% 调用 line-range file viewing，表明自我改进阶段开发的工具被复用于真实任务验证与代码定位。

## 相关工作脉络
- **DGM (Zhang et al., 2026a)**：基于 GitHub issue 的 coding agent 自我改进，通过诊断流程提议修改；SelfSearch 与其本质区别在于不使用外部评估分数引导搜索，且记录包含自我改进全过程而非仅最终结果。
- **HGM (Wang et al., 2026)**：独立诊断模块提议对 agent 的修改；SelfSearch 将任务代理与改进代理统一为同一可编辑实体，并通过 episode 记录实现跨代经验传递。
- **Hyperagents (Zhang et al., 2026b)**：在可编辑程序中显式定义任务代理与 meta-agent 两层；SelfSearch 则用单一代理完成两角色，并利用跨谱系共享记录实现经验复用。
- **SICA (Robeyns et al., 2025)**：自我改进 coding agent 并在单次任务内修改 scaffold；SelfSearch 聚焦于跨代持续改进，并在完全隔离的 runtime 中保留推理设置以保证可比性。
- **Group-evolving agents (Weng et al., 2026)**：多代理间的经验共享与开放 ended 自我改进；SelfSearch 将"共享"限定为两谱系 episode 记录的交叠复用，并强调无需下游评估的 reward-free 特性。
- **Mendel Gödel machine (Liu et al., 2026)**、**Meta-Harness (Lee et al., 2026)** 等递归自改进或元优化框架：以不同形式对改进过程自身进行优化；SelfSearch 更强调经验记录的直接利用而非对搜索算法的结构化重写。

## 局限性与未来方向
- 实验仅在 GPT 与 DeepSeek 两族模型与三类基准上验证，缺乏对更大规模模型族、更长搜索世代与更多谱系的系统性扩展评估。
- Reward-free 搜索虽降低成本，但未探索与评估引导搜索的混合策略（如用 episode 记录生成候选后再用少量评估预算精筛）在相同总预算下的上限。
- Episode 记录仅以只读形式被后续代理消费，未引入主动压缩、摘要或检索索引，当记录体积随世代累积时可能带来检索效率瓶颈。
- 跨模型迁移实验仅报告了反向方向（搜索与执行不同族），正向方向的稳定性与边界条件未充分讨论。
- 论文作者自述"需要进一步工作检验这些收益在独立搜索中的稳定性"，说明单次搜索结果的波动性与可复现性仍需更多随机种子验证。

## 研究启发与可借鉴点
- **自我改进轨迹即数据**：可将 agent 修改自身的推理轨迹、失败诊断与修复步骤视为高质量元数据源，用于构建自我监督的改进先验或训练轻量级"改进策略网络"。
- **双谱系交叉复用架构**：capability/adaptive 两类目标 + 共享只读记录的设计具有通用性，可迁移到强化学习、神经架构搜索或工作流自动化等需要"过程经验"的场景。
- **执行成本-成功率前沿的联合优化**：论文通过 $R_{\text{shared}}$ 与 accuracy-cost frontier 提供了一套可同时比较质量与效率的评估范式，值得在后续 agent 效率研究中沿用。
- **Runtime 与可编辑仓库的解耦**：将推理设置、token 上限、执行预算锁定在不可编辑 runtime 中、仅开放仓库层面的可编辑性，可为可控自我演化提供工程模板。
- **低成本高迁移性的 harness 发现**：$4.03 搜索成本即追平顶级公开 harness 的结果提示，reward-free 的经验驱动搜索可作为后续大规模 harness 自动设计的基线方案。

## 关键术语表
- **SelfSearch**：无需下游评估反馈、以历史自我改进 episode 记录为引导的代理迭代修改流程。
- **Episode record**：某次自我改进过程中代理的推理轨迹、工具调用、返回结果与代码 diff 的只读汇总。
- **Capability lineage**：以"识别能力短板并开发通用工具"为目标的搜索谱系。
- **Adaptive lineage**：以"改进失败响应与策略切换机制"为目标的搜索谱系。
- **Reward-free search**：搜索阶段不引入下游任务得分或外部评估信号，仅凭过程经验驱动修改的搜索范式。
- **Population-mean success**：两条谱系最终代理成功率 arithmetic mean，用于表征整体改进效果。
- **Shared-success cost reduction**：仅对比初始与进化代理共同解决任务的执行成本变化，衡量纯执行侧效率改进。
- **Harness**：指包含指令、工具与执行逻辑的 agent 实现封装，可在不同模型下以相同接口运行。

## 可复现要素
- **数据集**：SWE-bench Verified（120 题，来自 DGM small/medium 各 60 题加 large 子集抽样的 60 题）、SWE-bench Multilingual（seed=42 抽样 60 题，覆盖 8 种语言）、Terminal-Bench 2.1（全部 89 题）；基准数据本身公开，论文提供了具体 task IDs 列表（Table 6、Table 7）。
- **代码/权重**：论文提供执行框架与附录 A-F 的实现细节、搜索提示模板与模型配置；未声明独立仓库链接，原始代码需向作者索取或参考 DGM 发布仓库。
- **关键超参**：十代搜索；每 episode 上限 513 次 model call、512 次 tool step、4 小时容器执行；每 call 输出限制 GPT 8192 token、DeepSeek 16384 token；temperature=1，medium reasoning effort；搜索成本约 $6.52（GPT）与 $4.03（DeepSeek）。
