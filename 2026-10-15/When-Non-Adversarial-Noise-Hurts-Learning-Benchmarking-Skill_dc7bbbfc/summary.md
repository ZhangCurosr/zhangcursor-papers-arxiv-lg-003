---
title: "When-Non-Adversarial-Noise-Hurts-Learning-Benchmarking-Skill"
source: https://arxiv.org/pdf/2610.19910v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:12:46"
field: "LLM Agent Robustness and Reliability"
keywords: ["Skill Self-Evolution", "Agent Robustness", "Non-Adversarial Noise", "Benchmark", "RSE-Bench", "RSEClaw", "Continuous Learning"]
innovations: ["提出了首个评估非对抗性噪声下技能自我进化鲁棒性的基准 RSE-Bench，解耦学习/评估阶段噪声。", "设计了覆盖指令、证据、执行记录、结果配对四类注入点的任务保持型噪声协议及运行时框架 RSEClaw。", "揭示了噪声学习仅保留 25.5%-42.1% clean 增益，且证据噪声危害最大，防御性规则易溢出至干净环境的发现。"]
benchmarks: ["RSE-Bench", "SpreadsheetBench", "SP-DocVQA", "MP-DocVQA", "ALFWorld", "SkillFlow", "ContinualSkillBench"]
---

# 论文速读：When Non-Adversarial Noise Hurts: Benchmarking Skill Self-Evolution in Agents

## 一句话总结
本文提出了首个针对**技能自我进化（skill self-evolution）过程中非对抗性噪声鲁棒性**的基准测试 **RSE-Bench** 及其评估框架 **RSEClaw**，通过解耦学习阶段与评估阶段的噪声注入，系统量化了指令、证据、执行记录及结果-记录配对等四类常见噪声对智能体从经验中提炼可复用技能的破坏程度。

## 研究问题与动机
1.  **现有评估的盲区**：当前关于 LLM Agent 技能自我进化的研究主要关注干净环境下的性能提升，而部署场景中存在大量冗余指令、干扰文件、碎片化日志等非对抗性噪声，这些噪声可能污染学习过程并固化到持久技能中。
2.  **噪声影响的持续性**：不同于静态 Agent 仅在单次执行时受到噪声干扰，自我进化 Agent 的错误经验可能被蒸馏为可复用规则，从而影响后续多个任务的执行，这种“长尾”损害未被充分研究。
3.  **缺乏受控的解耦评估**：现有基准（如 AgentNoiseBench、StressWeb）主要评估执行时的噪声鲁棒性，或在进化完成后对冻结技能进行测试；缺乏一种能隔离“学习阶段噪声”与“评估阶段噪声”影响、且保持任务可解性的控制协议。
4.  **真实部署的可靠性担忧**：异步学习管线中可能出现结果与执行记录错位配对（outcome-record misbinding）等问题，导致智能体从错误的反馈中学习，需要评估这类运行时噪声对技能可靠性的具体影响。

## 核心贡献（创新点）
1.  **首个面向噪声鲁棒技能自我进化的基准 RSE-Bench**：区别于仅评估静态执行鲁棒性或对抗性内存投毒的工作，本文聚焦于非对抗性噪声进入学习循环后对持久技能质量的长期影响。
2.  **首套覆盖四类学习环路注入点的任务保持型噪声协议**：定义了 N-Inst（指令）、N-Evid（证据）、N-Rec（记录）、N-Bind（配对）四种注入方式，所有噪声均保持任务目标、正确答案和评分规则不变，确保比较的公平性。
3.  **方法无关的评估框架 RSEClaw**：通过固定接口解耦评估方法、执行、评分与噪声注入，支持 ACE、SkillOpt、RethinkSkill 等多种自我进化方法在同一套噪声条件下进行可比评估，并提供过程诊断工具（如噪声暴露率 IER、候选技能相似度 CCS）。
4.  **揭示噪声学习的严重损害与机制**：实验表明， noisy learning 仅保留了 clean self-evolution 25.5%–42.1% 的增益；并发现证据噪声（N-Evid）危害最大，而执行记录噪声（N-Rec）和配对噪声（N-Bind）即便不改变原始执行过程，也能造成与输入侧噪声相当的平均损失。

## 方法详解
1.  **噪声注入分类与约束**：
    *   **N-Inst (Instruction Noise)**：在用户请求中插入无关填充、看似相关但非决定性的信息或误导性建议，有效请求不变。
    *   **N-Evid (Evidence Noise)**：在智能体执行期间观察到的环境中添加无关文件、页面或重复观测，正确证据仍可达。
    *   **N-Rec (Record Noise)**：在学习器读取执行记录时，随机删除约 25% 的助手文本和工具输出字符，不留下标记。
    *   **N-Bind (Binding Noise)**：将异步管线中的结果与不同尝试的执行记录错位配对，两者均为真实数据，仅对应关系被破坏。
2.  **任务保持原则 (Task-Preserving)**：所有噪声注入必须保持任务目标、所需交付物、Ground-truth 答案、原生评分规则和交互预算不变，确保噪声增加的是寻找相关信息的难度而非改变任务本身。
3.  **解耦评估协议**：将学习噪声（Noisy Learning）与评估噪声（Noisy Evaluation）分开。免疫性（Immunity）评估在干净任务上测试从噪声经验中学到的技能；适应性（Adaptation）评估在匹配噪声下测试学习是否提升了应对同类噪声的能力。
4.  **RSEClaw 框架设计**：
    *   **固定接口**：方法只需实现检索技能、消费 episode、决定是否更新、暴露技能库的契约，无法重评分或访问最终测试集。
    *   **运行时噪声接缝**：N-Rec 和 N-Bind 在 execution 与 reflection 之间、evaluation 与 attempt 的接缝处注入。
    *   **过程诊断**：记录噪声暴露率 (IER)、被选中的技能版本、候选技能文本相似度 (CCS)，以及优化器请求日志，用于区分“真正抵抗了噪声”与“根本未遇到噪声”。
5.  **基准构成**：涵盖四个领域（电子表格操作 SpreadsheetBench、文档理解 SP/MP-DocVQA、交互决策 ALFWorld、序列技能复用 SkillFlow 和 ContinualSkillBench），共六个基准，每个基准分为 evolution-train、selection 和 held-out final-test 三个固定划分。

## 实验与结果
1.  **设置**：评估三种方法 ACE、SkillOpt、RethinkSkill，结合两个 Backbone GPT-5.2 和 DeepSeek-V4.1-Flash，在每个基准上进行 3 轮进化训练。
2.  **主要结果（Table 2）**：
    *   **Clean evolution 提升显著**：在所有方法-骨架组合上，clean learning 将六基准宏观平均得分提高了 2.6–6.5 个百分点。
    *   **噪声学习侵蚀增益**：四种学习侧噪声条件仅保留了 clean self-evolution **25.5%–42.1%** 的增益。
    *   **噪声影响排名**：平均相对损失为 N-Evid (4.9%) > N-Rec (4.5%) ≈ N-Bind (4.4%) > N-Inst (3.9%)。值得注意的是，N-Rec 和 N-Bind 仅扰动传给学习器的经验，却造成了与输入侧噪声相当的损失。
    *   **最坏情况**：证据噪声（N-Evid）在两个阶段均造成最大平均损失；在 SkillFlow 上，ACE+GPT-5.2 在 evaluation-side N-Evid 下得分从 31.33 降至 17.33（相对损失 44.7%）。
3.  **方法差异**：SkillOpt 在 DeepSeek-V4.1-Flash 上表现最稳健（最小平均相对损失 2.5%），而在 GPT-5.2 上 ACE 的相对损失最小（3.9%）。
4.  **基准差异**：SkillFlow (10.9%) 和 SpreadsheetBench (5.7%) 损失最大，SP-DocVQA (1.4%) 损失最小。
5.  **过程诊断（Section 5.3）**：
    *   **实际暴露率 (IER)**：ACE 在 N-Inst/N-Rec 下 IER 为 100%，N-Evid 为 97%，N-Bind 仅为 21%（32/150）。RethinkSkill 的 N-Bind IER 为 15%。
    *   **技能版本选择**：不同噪声条件下选出的最终技能版本（v*）不同。
    *   **文本相似度 (CCS)**：所有 noisy-to-clean CCS 均低于 clean-to-clean 参考均值，表明噪声导致技能文本显著变化，即使最终分数差异不大（如 RethinkSkill+N-Bind 分数变化仅 0.84pp，但 CCS 从 0.901 降至 0.783）。
6.  **消融分析（Section 5.4）**：
    *   **Clean evolution 的逆效果**：在 N-Inst/N-Evid 评估下，cleanly evolved agent 得分低于 initial agent（无技能），说明学到的技能在噪声环境下反而成为负担。
    *   **多轮训练扩大差距**：随着训练轮数增加（1→3轮），clean-noisy 差距从 8.75pp 扩大到 16.25pp。
    *   **匹配噪声学习无效**：在 N-Inst/N-Evid 评估下，用匹配噪声学习得到的技能（N→N）比干净学习得到的技能（C→N）得分更低，证明学习阶段的噪声暴露并不能提升部署时的适应性。
    *   **更强 Backbone 的缓解作用**：GPT-5.5 减少了噪声导致的绝对损失（如 ACE 最大损失从 15.83pp 降至 6.25pp），但未消除相对损失（1.3%-6.6%）。
7.  **案例分析（Appendix G）**：揭示了噪声导致技能失效的机制，包括：技能与噪声互斥（G.1）、防御性子句消耗交互预算（G.2）、防御规则溢出到干净环境（G.3）、污染与有用经验混合在同一规则中难以剥离（G.4）、错位配对诱发虚构规则并被固化（G.5）、归因从错误转移到观察通道（G.6）等。

## 相关工作脉络
1.  **Skill Self-Evolution Benchmarks (Evo-Memory, LifelongAgentBench, SkillFlow, ContinualSkillBench)**：这些工作评估 Agent 从经验中进化技能/记忆的能力，但均在**干净**环境下训练和选择技能，未考察学习过程本身的噪声鲁棒性。本文填补了这一空白。
2.  **Agent Noise Robustness Benchmarks (GSM-IC, AgentNoiseBench, StressWeb)**：主要评估**静态** Agent 在执行时的抗干扰能力（execution-time robustness），不涉及技能更新或持久化。本文将这些评估延伸至自我进化循环内部。
3.  **Adversarial Memory Poisoning (AgentPoison, MINJA, SkillMisevoBench, MemEvoBench)**：研究恶意攻击者污染 Agent 内存或自我进化过程的安全性问题，关注攻击成功率 (ASR) 或安全退化。本文关注**非对抗性**（ordinary）噪声，评估其对任务能力 (Capability) 的影响。
4.  **Unified Evaluation Frameworks (SkillMisevo, SkillLearnBench)**：部分工作提供了统一接口比较不同进化方法，但仅限于干净条件。RSEClaw 在此基础上增加了统一的噪声注入和诊断能力。
5.  **Feedback Dynamics / Reflection (RethinkSkill)**：研究进化技能的反馈动态，但未系统性地引入和执行时噪声注入到学习循环中。本文通过 N-Rec/N-Bind 首次精确控制了反馈链路的噪声。

## 局限性与未来方向
1.  **噪声模型的简化**：当前的 N-Rec（随机删除字符块）和 N-Bind（简单错位配对）是对真实世界日志丢失和异步管线错误的简化建模，可能未覆盖所有复杂的噪声形态（如部分正确反馈、延迟反馈等）。
2.  **评估范围的局限性**：目前仅评估了三种主流自我进化方法（ACE, SkillOpt, RethinkSkill）和两个 Backbone，未涵盖所有类型的技能表示（如纯代码技能、向量存储记忆）或其他架构的 Agent。
3.  **适应性评估的缺失**：由于 N-Rec 和 N-Bind 仅在 learning phase 定义，无法测试“学习阶段遭遇 N-Rec/N-Bind 是否提升了对该噪声的适应性”，仅能通过 N-Inst/N-Evid 的 N→N 评估来检验适应性，且结果显示适应性并未提升。
4.  **计算成本**：RSE-Bench 包含六个基准和多种噪声条件的交叉组合，加上过程诊断和多次独立运行，计算开销较大，可能限制大规模方法的快速迭代评估。
5.  **未来方向**：可以探索更复杂的噪声合成策略（如基于 LLM 的语义级噪声）、扩展至多模态或更真实的 Web/Office 环境、研究能够抵御此类噪声的新型技能精炼算法（如细粒度规则过滤、溯源验证）。

## 研究启发与可借鉴点
1.  **学习-评估噪声解耦设计**：将噪声严格区分为“进入学习循环的噪声”和“部署时的噪声”，并设计交叉评估协议（如 Table 11 所示），是评估自我进化系统鲁棒性的严谨范式，值得借鉴用于其他持续学习场景。
2.  **过程诊断指标**：引入 IER（实际噪声暴露率）、CCS（技能文本相似度）和选定版本追踪等诊断指标，有助于区分“鲁棒性”与“未暴露”，避免被表面性能迷惑，可作为评估 Agent 学习稳定性的标准配置。
3.  **N-Rec/N-Bind 的运行时注入机制**：通过框架层面的“seams”（接缝）在运行时注入记录噪声和配对噪声，而非预生成数据，使得无法预先感知噪声的方法也能被公平评估，这是一种灵活且强制性的基准构建技术。
4.  **防御性规则溢出问题**：案例分析揭示的“防御性规则在干净环境中同样被检索并阻碍正确操作”的现象，提示我们在设计自我进化系统时，需考虑技能的**情境感知检索**或**条件化规则**，而非简单的语义相似度检索。
5.  **污染-有用经验共生问题**：发现噪声引入的错误修正与真实有用经验往往交织在同一条规则中（G.4），这启示未来的技能精炼模块需要支持**子句级（clause-level）的溯源和过滤**，而非仅对整个规则进行接受/拒绝。

## 关键术语表
**RSE-Bench**: Reward-Skill Evolution Benchmark 的缩写（根据上下文推断，原文未明确全称，但指代本文提出的噪声鲁棒技能自我进化基准）。
**Skill Self-Evolution**: LLM Agents 在没有权重更新的情况下，通过将任务执行经验蒸馏为可复用的持久性工件（如技能库、洞察、工作流记忆）来提升自身能力的过程。
**N-Inst / N-Evid / N-Rec / N-Bind**: 分别代表指令噪声、环境证据噪声、执行记录噪声和结果-记录配对噪声，是本文定义的四种非对抗性噪声注入类型。
**Task-Preserving**: 噪声注入必须保持的任务不变量，包括任务目标、正确答案、原生评分规则、正确证据的可及性以及交互预算。
**Immunity (免疫性)**: 评估指标，指 Agent 在干净任务上，使用从噪声经验中学到的技能所保持的性能，衡量对历史学习噪声的抵抗能力。
**Adaptation (适应性)**: 评估指标，指 Agent 在噪声任务上，使用从同类噪声经验中学到的技能相比干净学习技能的提升幅度，衡量从噪声中学习的获益能力。
**IER (Input Exposure Rate)**: 输入暴露率，指优化器输入中包含注入噪声的 episode 的比例，用于量化方法实际接触到噪声的程度。
**CCS (Candidate-to-Clean Similarity)**: 候选技能与干净运行中候选技能的文本相似度（TF-IDF cosine similarity），用于衡量噪声对生成技能文本的影响程度。

## 可复现要素
*   **数据集**：RSE-Bench 数据公开于 HuggingFace (https://huggingface.co/datasets/Kirito-Lab/RSE-Bench)，包含六个基准的干净及四种噪声变体数据。
*   **代码**：RSEClaw 框架及噪声生成器开源在 GitHub (https://github.com/yutao1024/RSEClaw)。
*   **关键超参**：
    *   进化轮数 R=3。
    *   N-Rec 删除严重程度 0.25（约删除 25% 字符），作用于 assistant text 和 tool outputs 两个通道。
    *   N-Bind 窗口大小 3 attempts，延迟均匀采样至 6 positions。
    *   每任务工具调用步数上限 40，单步超时 60s，总墙钟时间 600s。
    *   模型：GPT-5.2, DeepSeek-V4.1-Flash (Actor/Optimizer), Kimi-K2.8 (噪声生成/审查)。
    *   随机种子：42。
