---
title: "When-Non-Adversarial-Noise-Hurts-Learning-Benchmarking-Skill"
source: https://arxiv.org/pdf/2610.19910v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:12:29"
field: "Agent 鲁棒性与自演化"
keywords: ["Skill Self-Evolution", "Agent Robustness", "Non-Adversarial Noise", "Benchmark", "RSE-Bench", "RSEClaw", "LLM Agent"]
innovations: ["首个面向非对抗性噪声的技能自演化鲁棒性基准 RSE-Bench，正交分离学习侧与评估侧噪声", "提出任务保持型四类噪声注入协议（指令/证据/记录/绑定），覆盖自演化循环全入口", "构建方法无关的统一评测框架 RSEClaw，内置过程诊断（IER/CCS/版本追踪）与 paired-replay audit"]
benchmarks: ["RSE-Bench", "SpreadsheetBench", "SP-DocVQA", "MP-DocVQA", "ALFWorld", "SkillFlow", "ContinualSkillBench"]
---

# 论文速读：When Non-Adversarial Noise Hurts Learning: Benchmarking Skill Self-Evolution in Agents

## 一句话总结
本文提出了 **RSE-Bench**（首个针对技能自演化鲁棒性的非对抗性噪声基准）和 **RSEClaw**（方法无关的统一评测框架），系统评估 LLM Agent 在经历指令、环境证据、执行记录和结果-记录配对四类噪声时，从经验中蒸馏可复用技能的能力。实验表明，四类噪声学习条件下，模型仅保留干净自演化所获增益的 25.5%–42.1%，其中证据噪声和记录/绑定噪声尤为致命。

## 研究问题与动机
- **核心问题**：LLM Agent 通过经验蒸馏构建持久技能库实现自演化，但现实中部署环境充斥冗余指令、无关文件、碎片化日志和异步反馈错位等普通非对抗性噪声；这些噪声若被蒸馏进持久技能，将影响后续任务，而现有工作对此缺乏系统性理解。
- **已有评测的不足**：现有 Agent 噪声基准（如 GSM-IC、AgentNoiseBench、StressWeb）仅在执行时注入扰动并评估**静态策略**，不涉及从噪声 episode 中更新技能；鲁棒性探测（如 RethinkSkill）仅在演化完成后扰动已冻结的技能；安全领域工作（如 AgentPoison、MemEvoBench）关注恶意攻击而非日常部署噪声。
- **关键空白**：缺乏对"非对抗性噪声进入学习循环后，如何被蒸馏为持久技能并影响下游任务"的受控评估。
- **实践动机**：真实部署中 Agent 会遇到过时表格、重复文件、异步返回的错位反馈等，导致错误经验被固化，而当前自演化方法均未对此进行验证。

## 核心贡献（创新点）
- **RSE-Bench 基准**：首个面向技能自演化的非对抗性噪声鲁棒性基准，覆盖 4 个领域、6 个数据集，受控注入四类噪声，并通过跨条件设计分离"学习侧噪声"与"评估侧噪声"的影响，区别于所有仅评估静态执行或攻击成功的既有基准。
- **任务保持型噪声注入协议**：四类噪声（N-Inst/N-Evid/N-Rec/N-Bind）均满足任务目标、真值答案、评分规则和证据可达性不变的条件，是首个在同一自演化循环中联合覆盖输入侧与运行时两类噪声位置的受控协议。
- **RSEClaw 统一框架**：提供固定接口解耦方法与执行/评分/噪声注入，支持多方法 plug-and-play 评测，并内置过程诊断（实际噪声暴露率 IER、候选技能相似度 CCS、版本选择追踪），弥补仅凭最终测试分无法区分"鲁棒"与"未遇到噪声"的缺陷。
- **系统性实验与机制分析**：在三方法两 Backbone 上的大规模实验揭示噪声危害的量化程度，并结合案例研究提炼六类失败机制（技能与噪声互斥、防御性规则越界、污染与有用经验共存等），为后续鲁棒自演化研究提供可解释的诊断视角。

## 方法详解
- **四类噪声定义**（各自独立作用于一个入口点）：
  - **N-Inst（指令噪声）**：向用户请求中注入无关填充文本、相关但非决定性信息、误导性建议，但有效请求不变，正确路径仍可求解。
  - **N-Evid（环境证据噪声）**：向执行空间注入干扰材料（无关文件、近似副本、重复呈现），正确证据保持可达，交互预算不变。
  - **N-Rec（执行记录噪声）**：在 episode 结束后、提供给学习者前，随机删除约 25% 的助手文本和工具输出字符（无标记残留），原始执行和评分不受影响。
  - **N-Bind（结果-记录绑定噪声）**：模拟异步学习管道中的反馈错位，将某 episode 的真实 outcome 与同 batch 内另一 attempt 的真实 record 配对，仅改变对应关系。
- **任务保持约束**：所有噪声条件均满足——任务目标不变、正确证据可达、真值 outcome 不变、评分语义不变、无答案泄漏或提示。通过自动化检查（字节级比对、渲染重放）+ 模型审阅（温度 0）+ 人工抽检三层验证。
- **学习/评估正交设计**：将噪声分为两个轴（进入位置 × 作用阶段），形成交叉条件矩阵：
  - **Immunity（免疫性）**：噪声学习 → 干净评估（N→C），衡量学到的技能在干净任务上是否仍可靠。
  - **Adaptation（适应性）**：噪声学习 → 噪声评估（N→N）vs 干净学习 → 噪声评估（C→N），衡量噪声中学习是否帮助适应部署时的同类噪声（仅 N-Inst/N-Evid 有此变体）。
- **RSEClaw 框架核心设计**：
  - **固定进化循环**：从 evolution-train 取任务 → 方法检索相关技能 → 原生环境执行 → 原生评分 → 返回 episode 给方法决定是否更新 → 下轮使用更新后技能库。
  - **方法接口契约**：`retrieve_skills(task)` / `consume_episode(episode)` / `should_update()` / `get_skill_store()`，禁止方法重评分、访问 final-test 分区或感知当前条件。
  - **噪声接缝（Noise Seams）**：N-Rec 嵌入在执行与反思之间，N-Bind 嵌入在评估结果与 attempt 记录之间，接口强制保证不变量在运行时被校验。
  - **过程诊断**：记录实际噪声暴露率（IER）、选定的技能版本（v*）、候选技能文本相似度（CCS），支持 paired-replay audit。

## 实验与结果
- **评测设置**：三方法（ACE、SkillOpt、RethinkSkill）× 两 Backbone（GPT-5.2、DeepSeek-V4.1-Flash）× 六数据集 × 五条件（1 干净 + 4 噪声学习）。每个 cell 为 3 次独立运行均值。
- **主要数字**：
  - 干净学习下，自演化将 6 个数据集的宏观平均提升 **2.6–6.5 个百分点**。
  - 四类噪声学习条件下，仅保留干净自演化增益的 **25.5%–42.1%**（平均相对损失：N-Evid 4.9%、N-Rec 4.5%、N-Bind 4.4%、N-Inst 3.9%）。
  - **证据噪声（N-Evid）危害最大**：学习侧相对损失 4.9%，评估侧升至 6.3%；最大单格下降为 ACE+GPT-5.2 在 SkillFlow 上从 31.33 降至 17.33（**-44.7%**）。
  - **N-Rec 和 N-Bind 的代价与输入侧噪声相当**，尽管它们不改变原始执行和评分。
  - **SkillOpt+DeepSeek-V4.1-Flash 表现最强**：干净分 77.71，N-Bind 下仅损失 1.9%。
  - **GPT-5.5 缓解但未消除**：学习侧相对损失从 15.83pp 降至 1.3%–6.6%，但所有噪声条件仍低于干净学习分 95.00。
- **跨轮次扩展加剧差距**：从 1 轮到 3 轮，干净学习从 85.00→90.00，N₂ 学习从 76.25→73.75，干净-噪声差距从 8.75pp 扩大到 16.25pp。
- **匹配噪声学习无适应性收益**：在 SpreadsheetBench 上，用 N-Inst 噪声学习 + N-Inst 噪声评估得 71.25，而干净学习 + N-Inst 噪声评估得 77.50，差距 6.25pp。
- **最弱场景**：ContinualSkillBench 下 ACE+DeepSeek 在 N-Bind 时相对损失 22.1%；SkillFlow 下 ACE+GPT-5.2 在 N-Evid 评估时下降 44.7%。

## 相关工作脉络
- **Skill 自演化方法**（Voyager、Expel、Reflexion、Dynamic Cheatsheet、SkillWeaver 等）及其评测基准（StreamBench、LifelongAgentBench、Evo-Memory、SkillFlow、SkillLearnBench、ContinualSkillBench 等）：均在无噪声条件下训练和选择技能，本文首次将噪声引入学习循环本身。
- **Agent 噪声鲁棒性评测**（GSM-IC、AgentNoiseBench、StressWeb）：仅评估静态 Agent 在运行时遭遇噪声的执行鲁棒性，不更新技能，本文扩展至自演化场景。
- **Agent 记忆污染/安全**（AgentPoison、MINJA、MemEvoBench、SkillMisevo-Bench）：注入恶意内容测量攻击成功率（ASR），本文注入非对抗性日常噪声，测量 held-out 任务上的能力损失。
- **反馈关联问题**（Attrimem、Romerl）：识别记忆中奖励错配问题，本文 N-Bind 将其抽象为可控注入扰动，首次独立测量其对技能演化的影响。
- **RethinkSkill**：评估演化后技能的鲁棒性，但仅扰动已冻结技能（输入侧），本文进一步覆盖学习侧噪声并正交化两阶段。

## 局限性与未来方向
- **只评估非对抗性噪声**：未涉及恶意注入或 adversarial attack，实际部署可能同时面临两种威胁。
- **仅测试三种方法和两个 Backbone**：方法覆盖面有限，更广泛的 agent 架构（如 tool-use、multi-agent）的鲁棒性有待验证。
- **噪声强度固定**：当前 N-Rec 使用固定删除率（25%）、N-Bind 使用固定窗口（3 attempts），未系统探索噪声强度的连续变化。
- **案例研究规模有限**：机制分析主要基于 SpreadsheetBench 的子集，其他数据集的失败模式可能不同。
- **未来方向**：可扩展至 adversarial noise 对比、探索细粒度（clause-level）技能隔离与修正、研究动态噪声强度下的鲁棒性曲线、将诊断框架应用于更多 agent 架构。

## 研究启发与可借鉴点
- **过程诊断的设计思路可迁移**：IER（实际噪声暴露率）+ CCS（候选文本相似度）+ 版本选择追踪的组合，可用于任何自演化/在线学习系统的可解释性评估，帮助区分"真正鲁棒"与"从未遭遇噪声"。
- **任务保持噪声构造方法**：答案感知的干扰物生成（answer-aware distractor generation）和运行时 seam 级注入的解耦设计，可作为构建其他领域鲁棒性基准的通用范式。
- **正交化学习侧/评估侧噪声的实验设计**：该设计可被复用到 RAG 系统、fine-tuning pipeline、online RL 等场景中，分离"训练时数据质量"与"部署时推理鲁棒性"的影响。
- **六类失败机制的分类框架**：可为后续研究提供诊断标签体系，指导防御策略设计（如 clause-level 隔离、元规则可信度评估）。
- **开源完整代码与数据**（GitHub + HuggingFace Dataset），可直接用于复现和扩展。

## 关键术语表
- **Skill Self-Evolution（技能自演化）**：LLM Agent 不更新权重，而是将执行经验蒸馏为持久、可复用的技能资产（技能库、自然语言洞察、工作流记忆等）。
- **RSE-Bench**：本文提出的首个面向非对抗性噪声下技能自演化鲁棒性的基准，覆盖 4 领域 6 数据集，支持学习任务侧与评估侧噪声的正交组合。
- **RSEClaw**：方法无关的统一评测框架，通过固定接口解耦方法实现与执行/评分/噪声注入，内置过程诊断日志。
- **N-Inst / N-Evid / N-Rec / N-Bind**：四类噪声入口——指令噪声、环境证据噪声、执行记录噪声、结果-记录绑定噪声。
- **Immunity vs Adaptation**：两大评估目标——Immunity 衡量噪声学习后在干净任务上的保持能力；Adaptation 衡量噪声中学习是否能帮助适应部署时同类噪声。
- **Task-Preserving（任务保持）**：噪声注入的核心约束，要求任务目标、真值答案、评分规则和证据可达性在干净与噪声版本间保持一致。
- **IER（Input Exposure Rate）**：优化器输入中包含注入扰动的比例，用于量化噪声是否真正到达学习方法。
- **CCS（Candidate-to-Clean Similarity）**：噪声运行与干净运行产生的候选技能之间的 TF cosine similarity，揭示相同分数下技能文本的潜在差异。

## 可复现要素
- **数据集**：RSE-Bench 包含 6 个数据集（SpreadsheetBench、SP-DocVQA、MP-DocVQA、ALFWorld、SkillFlow、ContinualSkillBench）的各 5 个版本（1 干净 + 4 噪声），已开源至 HuggingFace（https://huggingface.co/datasets/Kirito-Lab/RSE-Bench）。
- **代码**：RSEClaw 框架、噪声生成器、评测 harness 全部开源（https://github.com/yutao1024/RSEClaw）。
- **关键超参**：每 episode 40 步工具调用上限、60s/步超时、600s 总墙钟时间；N-Rec 删除率 25%（severity 0.25）、N-Bind 窗口 3 attempts、最大延迟 6 位置；每方法 3 轮训练（R=3），每轮 50 个训练任务；Seed = 42。
- **模型配置**：DeepSeek-V4.1-Flash（主实验与分析）、GPT-5.2（主实验）、GPT-5.5（Backbone 消融）；Kimi-K2.8 专用于 N-Inst/N-Evid 生成与审阅（不参与演化运行）。
- **验证流程**：自动化检查 + 两轮模型审阅（温度 0）+ 人工抽检（每基准 30 个样本 × 3 人独立标注），详细见 Appendix F。
