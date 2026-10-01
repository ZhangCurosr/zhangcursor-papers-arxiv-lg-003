---
title: "SRHARNESS-A-HARNESS-FOR-AGENTIC-SYMBOLIC-REGRESSION"
source: https://arxiv.org/pdf/2609.35501v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:20:59"
field: "符号回归与科学发现"
keywords: ["symbolic regression", "agentic AI", "LLM harness", "scientific discovery", "equation recovery", "runtime support"]
innovations: ["提出与搜索策略无关的领域特定harness框架，解耦运行时支撑与搜索逻辑", "设计组合科学动作机制，支持基于表达式的视图实现异构操作统一接口", "建立持久科学状态与模型朝向视图分离机制，缓解长程agent的context瓶颈"]
benchmarks: ["LLM-SRBench", "LSR-Synth", "LSR-Transform", "LSR-Transform-Anon"]
---

# 论文速读：SRHARNESS-A-HARNESS-FOR-AGENTIC-SYMBOLIC-REGRESSION

## 一句话总结
论文提出SRHarness，一个面向代理符号回归（agentic symbolic regression）的领域特定运行时支撑框架，通过组合科学动作、持久科学状态和轨迹生命周期管理三大机制，在LLM-SRBench基准上显著提升了数值泛化与符号恢复性能，证明了高效符号回归不仅依赖模型能力或工具丰富度，更需要结构化的运行时支撑。

## 研究问题与动机
- **核心问题**：现有代理符号回归方法将LLM作为搜索控制器自主选择科学操作，但其性能不仅取决于底层模型和搜索策略，还强烈依赖于支撑科学搜索的运行时基础设施，而现有系统的harness通常与特定搜索策略耦合，缺乏可复用的独立支撑层。
- **现有方法不足1**：传统LLM引导的SR方法（如LLM-SR、IGSR）仅将LLM作为候选方程生成器或变异算子，嵌入预定义的进化/搜索流程中，缺乏自适应的科学操作选择能力。
- **现有方法不足2**：近期代理方法（如SR-Scientist、KeplerAgent）虽赋予LLM更多控制权，但其支持机制（工具接口、状态管理）围绕各自搜索策略设计，不同系统间的比较往往混淆了搜索策略差异与harness差异（Zhang et al., 2026c）。
- **现有方法不足3**：现有科学信息处理harness（如Beaver、Ref）针对多模态证据合成或游戏环境设计，未考虑符号回归中候选方程需共享一致表示与评估语义、中间分析结果需跨阶段复用的特殊需求。

## 核心贡献（创新点）
1. **提出SRHarness，一个与搜索策略无关的领域特定运行时支撑框架**：将执行环境、认知管理和治理三个维度实例化为符号回归专用机制，使runtime支持独立于具体搜索逻辑。
2. **设计组合科学动作（composable scientific actions）机制**：通过`r = A(q; κ)`形式分离智能体指定的科学参数与harness管理的执行上下文，支持基于表达式的视图（raw变量、变换视图、候选派生视图），使异构分析/拟合/搜索过程通过统一接口和评估语义运行。
3. **建立持久科学状态（persistent scientific state）管理**：将已评估假设及其证据持久化存储于对话上下文之外，并通过投影策略`v_t = P_ρ(S_t)`向模型暴露紧凑视图（Pareto前沿、Top-k），避免完整归档重复进入对话。
4. **实现轨迹生命周期管理（trajectory lifecycle management）**：协调延续、分支、重启和终止操作，通过R×C×L×K调度器（重启轮数×分支数×细化深度×本地响应采样）组织长程搜索，并跨转换记录溯源信息。
5. **在LLM-SRBench上验证harness的独立贡献**：与Codex对照实验证明，相同DeepSeek-v4-flash-0731骨干下SRHarness达72.97% SA vs Codex 20.72%，且为Codex提供相同科学工具无法复现优势，表明结构化运行时支撑是关键性能来源。

## 方法详解
**1. 组合科学动作（Section 3.2）**
- 动作形式：`r = A(q; κ)`，其中q为智能体选择的科学参数（指定操作类型和目标量），κ为harness管理的执行上下文（数据绑定、训练-验证划分、运行时配置）。
- **表达式视图机制**：q不必局限于固定数据集列，可指定为当前数据的表达式视图——原始变量`x_i`、变换视图（如`log x_i`、`x_i/x_j`）、候选派生视图（如残差`y - f(x)`），使中间构造视图无需物化新数据集即可作为后续动作输入。
- **候选-证据契约**：产生候选的动作遵循统一合约，记录表达式、评估值、复杂度、诊断证据和溯源信息；harness在 admitted 前应用统一的有效性检查和评估语义。
- **结果分离**：每个动作将机器可读结果与返回模型的紧凑观察分离，保留详细证据而不填充对话上下文。

**2. 持久科学状态（Section 3.3）**
- 状态存储：候选方程在 admitted 后与定量评估、复杂度、支持证据、溯源信息一起存储在对话上下文之外的持久状态`S_t`中，跨对话轨迹保留。
- **模型朝向视图**：`v_t = P_ρ(S_t)`，投影策略ρ生成紧凑子集而非完整归档；实现包括：
  - **拟合-复杂度Pareto视图**：突出候选方程在拟合优度与复杂度之间的权衡
  - **Top-k视图**：呈现历史领先候选
- 被排除在当前视图外的候选仍保留在持久状态中，可在后续视图中纳入。

**3. 轨迹生命周期管理（Section 3.4）**
- **四种转换**：延续（advance existing context）、分支（create alternative continuations from common prefix）、重启（initialize new conversation from model-facing view）、终止（end search）。
- **R×C×L×K调度器**（Algorithm 1）：
  - R：重启轮数
  - C：独立对话分支数
  - L：细化深度（每分支最多步数）
  - K：本地响应采样数
- 流程：每次重启从持久状态取Top-k初始化新对话；每分支内最多L步细化，模型接收Pareto视图；当K>1时在每步采样多个响应，按候选质量选一个延续，其余的诊断结果整合入上下文；全程记录中间结果，最佳候选在早停/中断/失败时保留。

**4. 动作集（Appendix A.4）**
- 统计分析、关系分析、阅读技能、评估公式、提交公式、常数拟合、调用PySR、调用SINDy、代码执行器。
- 首次响应前自动运行统计与关系诊断并加载符号定律发现技能。
- 默认配置：R1-C1-L30-K1（1轮重启、1分支、最多30步、每步1响应）。

## 实验与结果
**数据集**：LLM-SRBench（Shojaee et al., 2025b），包含240个问题：
- LSR-Synth：128个合成问题（物理44、化学36、生物24、材料25），评估ID/OOD数值泛化
- LSR-Transform：111个问题，由已知科学方程变换而来，侧重精确符号恢复
- LSR-Transform-Anon：作者构建的匿名变体，移除科学描述和变量语义，保留数值观测

**评估基线**：
- 非LLM：PySR（Cranmer, 2023）
- LLM引导：LLM-SR（Shojaee et al., 2025a）、IGSR（Saveliev et al., 2026）
- 代理方法：SR-Scientist（Xia et al., 2026）
- Codex对照实验

**主要结果**：
- **LSR-Synth（DeepSeek-v4-flash-0731）**：SRHarness在物理/化学/生物/OOD上分别达84.46%/78.00%、95.24%/78.87%、95.14%/87.09%，显著优于LLM-SR、IGSR、SR-Scientist（均低于70%）；符号准确率SA达6.20%，而基线均低于1%；搜索效率最高（平均11.11分钟/问题）。
- **LSR-Transform（DeepSeek-v4-flash-0731）**：SRHarness达93.69% SA，vs SR-Scientist 62.16%；数值精度相当（NMSE ~10⁻¹⁴），但SRHarness表达式复杂度最低（14.64 vs LLM-SR 64.46、IGSR 42.14），表明能恢复更简洁的结构等价方程。
- **LSR-Transform-Anon（移除语义先验）**：SRHarness保持72.97% SA，vs SR-Scientist 39.64%，保留78%原始准确率（LLM-SR/IGSR仅保留28%）；搜索时间几乎不变（12.34分钟），而基线需更长轨迹但恢复更少方程。
- **Codex对照**：相同DeepSeek-v4-flash-0731骨干，SRHarness 72.97% SA vs Codex 20.72%；SRHarness+DeepSeek性能媲美Codex+GPT-5.5（68.47%）；向Codex提供相同科学工具后SA反降至64.86%，证明harness集成而非工具可用性是关键。
- **骨干依赖性**：SR性能与长程agent基准（TerminalBench、DeepSWE）高度相关（Spearman ρ=0.90/1.00），而非模型规模或GPQA科学问答分数；DeepSeek-v4.1-flash（16B激活参数）整体优于DeepSeek-v4-pro-0813（49B激活参数）。

## 相关工作脉络
1. **LLM引导的符号回归**（LLM-SR、LaSR、SGA、SR-LLM、DrSR、ProAug、IGSR）：LLM作为候选生成器/变异算子嵌入预定义搜索流程，SRHarness将其角色提升为自主控制器，并通过harness支撑长程反馈驱动搜索。
2. **代理符号回归**（SR-Scientist、KeplerAgent、LLM-PySR、MOT-SR、DE、A-SR）：赋予LLM更多搜索控制权，但支持机制围绕特定策略设计；SRHarness提供不预设搜索策略的可复用运行时支撑。
3. **科学agent harness**（Ref gaming harness、Beaver科学策展、Ref多模态报告生成）：针对环境交互或证据合成设计；SRHarness面向异构分析/拟合/搜索过程，强调候选方程共享一致表示与评估语义。
4. **传统符号回归**（PySR、遗传编程）：基于进化算法搜索表达式空间；SRHarness结合LLM推理与代码执行，实现自适应科学操作选择。
5. **基准评估**（LLM-SRBench、NewtonBench）：SRHarness在LLM-SRBench上验证，并贡献了LSR-Transform-Anon匿名变体以分离语义先验与数据驱动发现。

## 局限性与未来方向
- **模型规模非线性效应**：更大/更强模型不总是表现更好（如DeepSeek-v4-pro-0813 vs DeepSeek-v4.1-flash），表明需更深入理解agent场景下的模型能力匹配。
- **领域难度差异显著**：材料科学问题因变量范围平滑、低阶多项式近似有效而接近饱和，物理/化学/生物更具挑战性，harness在不同领域的通用性需进一步验证。
- **轨迹调度超参敏感**：消融显示深化细化比增加重启/分支更有效，但R×C×L×K最优配置可能依赖任务特性，缺乏自适应调度机制。
- **工具扩展的复杂性权衡**：添加拟合动作提升SA至79.28%但复杂度增至33.16，表明动作集扩展需平衡发现能力与搜索效率。
- **未来方向**：开发自适应调度策略、探索更多领域特定动作、研究harness与模型能力的协同优化机制。

## 研究启发与可借鉴点
1. **harness作为一等公民**：将运行时支撑（执行环境、状态管理、轨迹治理）从搜索策略中解耦，作为独立可复用层设计，适用于其他long-horizon agent科学发现任务（如实验设计、假设检验）。
2. **表达式视图机制**：支持动作参数指定为原始/变换/候选派生视图，使中间分析结果无缝流动，可迁移至其他需要多阶段数据分析的agent系统。
3. **模型朝向视图压缩**：通过投影策略`P_ρ`将持久状态压缩为Pareto/Top-k等紧凑视图，缓解长程agent的context window瓶颈，值得在其他agent harness中应用。
4. **R×C×L×K调度抽象**：将重启、分支、细化深度、响应采样统一为可调调度维度，为长程搜索的资源分配提供结构化接口，可适配不同计算预算。
5. **匿名化评估设计**：LSR-Transform-Anon通过移除语义先验隔离纯数据驱动发现能力，为评估agent是否真正理解科学结构而非依赖预训练知识提供了可复用的评估范式。

## 关键术语表
**Symbolic Regression (符号回归)**：从观测数据中发现可解释数学表达式的任务，目标是恢复底层科学定律而非仅拟合数据。

**Agentic Symbolic Regression (代理符号回归)**：将LLM从候选生成器提升为搜索控制器，自主根据中间证据选择科学操作的符号回归方法。

**Composable Scientific Actions (组合科学动作)**：通过统一接口`r = A(q; κ)`分离智能体决策与执行上下文，支持基于表达式的视图使异构操作可组合。

**Persistent Scientific State (持久科学状态)**：跨对话轨迹保留已评估假设、证据和溯源的持久存储，与模型朝向的紧凑视图分离。

**Trajectory Lifecycle Management (轨迹生命周期管理)**：协调延续、分支、重启、终止的长程搜索治理机制，通过R×C×L×K调度器组织资源分配。

**LSR-Transform-Anon (匿名化变换基准)**：保留数值观测但移除科学描述和变量语义的评估变体，用于分离语义先验与数据驱动发现。

**Symbolic Accuracy (符号准确率, SA)**：测量的发现表达式是否与参考方程结构等价（代数重排、有效域恒等式允许），而非仅数值接近。

**Pareto View (Pareto视图)**：展示候选方程在拟合优度与复杂度之间权衡的模型朝向视图，促进多样性探索。

## 可复现要素
- **数据集**：LLM-SRBench（LSR-Synth 128题、LSR-Transform 111题、LSR-Transform-Anon匿名变体），论文未提及是否公开，但基准本身来自Shojaee et al. (2025b)
- **代码**：已开源，地址https://github.com/tsinghua-fib-lab/SRHarness（Reproducibility statement明确声明）
- **权重**：未单独提供，使用开源模型DeepSeek-v4-flash-0731、GLM-5.3-flash等通过OpenRouter访问
- **关键超参**：默认R1-C1-L30-K1（1重启轮、1分支、30细化步、1本地采样）；随机分割seed=42；实验seed=26091320；DeepSeek输出4096 tokens，GLM输出8192 tokens；.temperature=1.0，reasoning disabled for DeepSeek，low effort for GLM
- **动作集**：统计分析、关系分析、阅读技能、评估公式、提交公式、常数拟合、调用PySR、调用SINDy、代码执行器
