---
title: "SELFSEARCH-REWARD-FREE-SEARCH-FOR-SELF-IMPROVING-AGENTS"
source: https://arxiv.org/pdf/2609.37968v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:31:51"
field: "自我改进智能体"
keywords: ["reward-free search", "self-improving agents", "LLM agents", "agent evolution", "experience sharing"]
innovations: ["用回合记录替代下游评估实现奖励无关搜索", "双谱系（能力/适应）协同进化并通过共享记录跨谱系迁移发现"]
benchmarks: ["SWE-bench Verified", "SWE-bench Multilingual", "Terminal-Bench 2.1"]
---

# 论文速读：SELFSEARCH: REWARD-FREE SEARCH FOR SELF-IMPROVING AGENTS

## 一句话总结
论文提出 SelfSearch，一种无需下游任务评估反馈的奖励无关搜索程序：智能体利用先前自我改进回合的记录（推理轨迹、工具调用、代码变更）指导自身实现的重写，在不增加搜索成本的前提下持续提升 Agent 的任务求解能力与执行效率。

## 研究问题与动机
- 现有自我改进 Agent 方法依赖重复的下游评测来引导搜索，每次候选 Agent 都需要在开发集上执行评测，累积成本随评测任务数线性增长。
- 搜索信号与特定任务绑定，导致进化出的 Agent 难以跨模型、跨基准迁移。
- 自我修改过程本身会产生大量经验信号（如何识别局限、如何构造修订、如何检验效果），但这些信号未被系统化复用。
- 需要一种**不依赖下游奖励信号**、仅凭自我改进经历即可持续进化 Agent 的搜索范式。

## 核心贡献（创新点）
1. **SelfSearch 框架**：用回合记录替代下游评估作为搜索指导信号，实现纯奖励无关的 Agent 自我改进搜索。与 DGM/HGM 等基于下游分数的搜索相比，搜索阶段不再需要任何任务评测。
2. **双谱系协同进化设计**：引入能力谱系（capability, B^c）与适应谱系（adaptive, B^a）两条并行进化线，分别探索"任务求解局限修复"与"修订策略改进"两类方向，并通过共享回合记录实现跨谱系知识迁移。
3. **跨模型泛化与成本优势**：进化出的 Agent 在 DeepSeek-V4-Flash 上以仅 4.03 美元搜索成本达到 Terminal-Bench 2.1 的 82.0% 成功率，与九 harness 公开对比中的最高分 Codex 持平；且在不同模型间保持提升。
4. **机制剖析与消融**：系统拆解回合记录与动态改进者各自贡献（各带来约 2-3 个百分点的提升），并追踪工具在改进过程中如何诞生、修复、并在下游任务中复用（如文本搜索工具在 23.6% 的 DeepSeek 任务中被调用）。

## 方法详解
**核心公式**：第 k 代回合中，Agent B_k 根据历史记录 E_k 与搜索方向 d 生成后继与记录：
$$(B_{k+1}, e_k) = \text{SelfImprove}(B_k; \mathcal{E}_k, d)$$
其中 $e_k$ 包含推理轨迹、工具调用及其结果、代码变更快照。

**双谱系结构**：维护两条谱系（能力 c、适应 a），每代从相同基线 Agent B_0 出发，分别接收上代的回合记录（交叉共享），并行执行自我改进。搜索方向 d 为定性指令：
- **能力方向**：识别任务求解中的局限（如低效行为、失败操作），开发可复用的工具或流程。
- **适应方向**：改进 Agent 在面对矛盾证据或失败时的修订策略。

**运行时隔离**：Agent 的整个仓库（指令、工具、执行逻辑）均可编辑，但模型推理循环、资源限制、执行预算固定在外部运行时中，通过 `respond(message)` 接口调用，确保搜索环境稳定可控。

**搜索代价**：DeepSeek 配置下 10 代搜索总成本为 4.03 美元（832 次调用，36.89M 输入 token），GPT 配置为 6.52 美元。评测在搜索冻结后独立进行，不计入搜索成本。

## 实验与结果
**模型配置**：
- GPT 组：gpt-5.6-sol（搜索）+ gpt-5.6-luna（下游）
- DeepSeek 组：deepseek-v4-pro-0813（搜索）+ deepseek-v4-flash-0731（下游）
- 均使用 medium reasoning effort，temperature=1

**评测基准**：
- SWE-bench Verified：120 个真实 GitHub Issue 修复任务
- SWE-bench Multilingual：60 个任务，覆盖 8 种编程语言（C/C++/Go/Java/JS/PHP/Ruby/Rust）
- Terminal-Bench 2.1：89 个命令行交互任务

**主要结果（表 1）**：
- **Terminal-Bench 2.1**：DeepSeek B^c 从 65.2% 提升至 73.0%（+7.8pp），GPT B^c 从 43.8% 提升至 55.1%（+11.3pp）
- **SWE-bench Multilingual**：DeepSeek B^a 从 68.3% 提升至 73.3%（+5.0pp），且在初始与进化 Agent 共同解决的 39 个任务上执行成本降低 **38.5%**
- **SWE-bench Verified**：DeepSeek 双谱系均从 81.7% 提升至 86.7%（+5.0pp）
- **搜索成本对比（表 2）**：SelfSearch 比线性搜索低 49.0%–53.2%（DeepSeek）/ 13.4%–47.2%（GPT）

**最强结果**：以 DeepSeek V4 Flash 执行时，SelfSearch 在 Terminal-Bench 2.1 上达到 **82.0%**（73/89 任务），与公开九 harness 对比中排名第一的 Codex 持平，且搜索成本仅 4.03 美元。

**跨模型迁移（表 3）**：在 SWE-bench Verified 上，GPT 搜索得到的 Agent 用 DeepSeek V4 Flash 执行达 83.3%/85.0%，DeepSeek 搜索得到的 Agent 用 GPT-5.6 Luna 执行达 79.2%/76.7%，均高于各自基线。

**消融（表 4）**：移除回合记录使 DeepSeek 均值下降 2.9pp，固定改进者为 B_0 使均值下降 2.9pp，full SelfSearch 均取得最高成绩。

## 相关工作脉络
1. **DGM (Zhang et al., 2026a)**：可进化任务 Agent，但依赖下游评测引导搜索，未实现奖励无关搜索。
2. **HGM (Wang et al., 2026)**：分离诊断与编辑流程，同样需下游评估反馈。
3. **Hyperagents (Zhang et al., 2026b)**：定义独立的 task agent 与 meta-agent，角色分离而非统一。
4. **SICA (Robeyns et al., 2025)**：统一角色设计，但仍依赖下游奖励。
5. **SelfSearch 定位**：唯一同时满足"可进化任务 Agent + 可进化 meta-agent + 统一角色 + 奖励无关搜索"的方法（表 13）。
6. **经验共享工作 (Weng et al., 2026; Liu et al., 2026)**：关注多 Agent/多任务间的经验传递，本文聚焦于单 Agent 自身的改进历程记录复用。
7. **SWE-Agent (Yang et al., 2024)**：Agent-Computer Interface 设计，SelfSearch 进化出的线范围查看、精确文本替换等操作与此类接口设计相似。

## 局限性与未来方向
- **评测范围有限**：仅在编程/命令行类基准上验证，未测试自然语言理解或多模态场景。
- **搜索成本仍未趋零**：DeepSeek 配置需 4.03 美元、GPT 需 6.52 美元，对大规模迭代仍构成约束。
- **短期进化**：仅 10 代，未探索长期开 ended 进化的稳定性与性能天花板。
- **增益存在波动**：消融显示移除记录仅损失 2-3pp，说明回合记录的价值虽显著但非决定性。
- **作者自述**：不同独立搜索中增益的一致性尚待验证；未来可尝试将 SelfSearch 与评估引导搜索结合，在相同总预算下探索混合策略。

## 研究启发与可借鉴点
1. **回合记录作为廉价信号**：用改进过程的轨迹记录替代昂贵的下游评测，为其他需要搜索优化的 Agent 系统提供了低成本替代方案。
2. **双谱系差异化探索**：能力方向与适应方向的并行设计值得借鉴——前者修"工具"，后者修"策略"，二者通过共享记录交叉受益，避免了单方向过早收敛。
3. **进化工具的下游复用性**：本文发现的文本搜索、线范围查看等工具在下游任务中分别被调用 23.6% 和 33.8% 的任务，证明自我改进过程中开发的工具具有通用价值，可作为 Agent 工具箱的可持续积累资产。
4. **运行时隔离架构**：将推理循环、资源限制固定在外部运行时、Agent 仓库仅可编辑业务逻辑的设计，兼顾了安全控制与灵活性，适合需要可控搜索边界的生产场景。
5. **与本团队方向结合机会**：若团队关注 Agent 自动化工作流生成（参考 AFlow, Zhang et al., 2025）或元学习优化，SelfSearch 的回合记录机制可直接嵌入为"改进历史"模块，减少对工作流评估的依赖。

## 关键术语表
- **SelfSearch**：无需下游评估反馈、仅凭自我改进回合记录指导 Agent 自身实现的搜索过程。
- **Episode record**：单个自我改进回合的完整轨迹，含推理过程、工具调用及结果、代码变更前后的快照。
- **Capability lineage (B^c)**：聚焦识别任务求解局限并开发可复用工具/流程的进化谱系。
- **Adaptive lineage (B^a)**：聚焦改进 Agent 面对矛盾证据或失败时的修订策略的进化谱系。
- **Population mean**：两条谱系成功率 arithmetic mean，用于汇总进化效果。
- **Reward-free search**：搜索阶段不使用下游任务评测分数作为指导或选择信号。
- **Search direction**：定性指令（capability/adaptive），引导 Agent 在不同维度上调查改进方向。

## 可复现要素
- **数据集**：SWE-bench Verified（120 任务）、SWE-bench Multilingual（60 任务，seed=42 抽样）、Terminal-Bench 2.1（89 任务）；均为公开数据集。
- **代码/权重**：论文未声明开源；初始 Agent 基于 DGM 公开实现修改；搜索提示词、模型配置见 Appendix B–C。
- **关键超参**：10 代搜索、2 谱系、medium reasoning effort、temperature=1、每调用输出上限 8192/16384 token、每回合最多 513 次模型调用/512 次工具步骤/4 小时执行时间。
- **成本**：DeepSeek 搜索 4.03 美元，GPT 搜索 6.52 美元（附录 D）。
