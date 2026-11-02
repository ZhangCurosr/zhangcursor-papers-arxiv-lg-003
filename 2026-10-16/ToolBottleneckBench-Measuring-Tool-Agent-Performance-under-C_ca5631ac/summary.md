---
title: "ToolBottleneckBench-Measuring-Tool-Agent-Performance-under-C"
source: https://arxiv.org/pdf/2610.21098v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:15:57"
field: "LLM Agent Evaluation"
keywords: ["tool agent benchmark", "harness effect", "failure diagnosis", "controlled pressure", "completion cap truncation", "repair evaluation"]
innovations: ["受控单压力基准设计，固定任务族每次仅变一个压力以分离因果效应", "失败记录联合标签（义务+limit）区分模型失败与系统限制", "修复特异性检验协议（L/Γ/Σ/Δ）与配置切换帕累托比较"]
benchmarks: ["ToolBottleneckBench", "ToolMenuBench", "Harness-Bench", "ToolSandbox"]
---

# 论文速读：ToolBottleneckBench: Measuring Tool-Agent Performance under Controlled Pressures and Fixed Harness Budgets

## 一句话总结
本文提出 **ToolBottleneckBench**，一个在固定 harness 预算下对工具代理施加单一可控压力的基准测试，通过精确评分和失败记录揭示：可见工具菜单膨胀导致的性能下降主要源于 completion cap 截断，且大多数失败实际由系统限制而非模型能力引起。

## 研究问题与动机
- 现有工具代理基准通常只用单一成功率指标比较，但代理运行在含工具菜单、返回编码、token/调用/轮次预算的 harness 中，失败原因难以归因于模型还是 harness 限制。
- 现有基准中任务间的 catalog、workflow、observation format 等多个负载同时变化，无法隔离单一压力效果。
- 已有失败诊断方法只能定位失败阶段，但不能验证标签是否准确，也无法判断压力与 harness limit 的关联。
- 实际部署中配置切换（换模型或 provider route）是常见应对策略，但缺乏在相同压力下对修复与配置切换的系统比较。

## 核心贡献（创新点）
- **受控单压力基准设计**：固定任务族，每次仅改变一个压力（R/V/W/D），通过配对差异估计压力效应，区别于同时变化多因素的前人工作。
- **失败记录（Failure Record）**：记录首个未满足义务与触发的 harness limit，区分模型失败与系统限制，此前工作缺乏此类联合标签。
- **修复激活与配置切换对比协议**：定义 L/Γ/Σ/Δ 四个量化指标检验修复是否在压力下特异性激活，并在相同压力下比较修复与切换配置的帕累托优势。
- **十六倍 completion cap 扫掠实验**：直接验证截断是菜单惩罚的主要机制，证明多数菜单损失可被预算放宽消除。
- **开源完整基准设施**：发布确定性任务生成器、精确评分器、所有评分数据与分析代码，支持可复现研究。

## 方法详解
- **任务族设计**：36 个任务族嵌套于 12 个合成企业意图（会计、邮件、文件、日历），每个任务固定请求、事实值、决策标签、证据规则、工具身份，记录 ID 不透明防止答案复用。
- **四种压力层级**：
  - **注册目录 R**：$R \in \{32, 512, 2048\}$ 个工具，通过冻结检索器返回 16 工具短列表，测量可用性 A 与选择性 S 的 ITT 成功（$A \cdot S$）。
  - **可见菜单 V**：$V \in \{8, 32, 128\}$ 个 schema，操作工具始终可见，增加的路由邻居工具在 workspace/region/collection 上 differ，schema 字节从 3,481 增至 55,176。
  - **认证工作流 W**：low/medium/high 分别要求 2/4/6 次读取加一次可逆调用，程序长度、依赖深度、证据广度同步增长。
  - **返回状态编码 D**：raw JSON / compact（字段字典）/ typed（类型字典）/ linked（身份链接），四种编码解码为相同值。
- **固定 Harness 合同**：completion cap = 1,024 tokens/请求，episode budget = 128,000 tokens，call limit = 认证最小值 +2，turn limit = call limit +2，触发 infeasibility stop 时提前终止。
- **精确评分九项义务**：$Y = A \land S \land P \land Q \land E \land X \land G \land B \land C$，按序检查可用性、选择、工作流合法性、答案精确性、证据集、安全性、治理、预算、完成完整性，无任何模型裁判。
- **统计方法**：配对 family-level 差异（pp），10,000 次 cluster bootstrap（聚类 36 任务族），Holm 多重检验校正，主效应与交互效应通过因子设计系数计算。

## 实验与结果
- **评估规模**：21 个配置（7 个机构：Anthropic/OpenAI/Google/Alibaba/DeepSeek/Tencent/Z.ai），26,576 个评分条件，20,408 个精确 VTS 评分。
- **主要结果**：
  - 可见菜单从 8→128 导致 VTS 下降 **−12.3 pp** [−14.6, −9.9]（Holm $p < .001$），34/36 任务族退化；全菜单可见时降 **−17.5 pp**。
  - 工作流压力低→高仅 **+0.8 pp** [−1.2, 2.8]，无显著 pooled 效应；不同配置效应符号相反（Gemini 3.6 Flash: −30.6 pp vs HY4 Preview: +34.7 pp）。
  - 编码压力中 compact 降 **−3.7 pp**（Holm $p = .003$），typed/linked 无显著效应。
- **截断机制验证**：V=128 时截断率从 1.6% 升至 13.6%，391/411 次截断发生在首次调用前；跨配置截断率与菜单惩罚 Pearson $r = -0.95$；十六倍 cap（16,384 tokens）使 6 个高截断配置菜单惩罚从 −43.7 pp 降至 −2.7 pp。
- **失败归因**：3,725 次失败中 **91.5%** 以 harness limit 结束（completion cap 28.7%、预算/调用限制 50.0%、infeasibility 12.8%），仅 5.5% 为错误结论；所有 849 次"selection"失败均无调用记录。
- **修复效果**：connector-first router 激活（ITT 提升 58.1 pp）、routed menu 激活（$\Gamma = 4.9$ pp）、public-handoff controller 激活（结构保留集提升 96.6 pp）；两个 prompt 修复均未激活。仅 router 和 controller 在成功上优于所有配置切换。

## 相关工作脉络
- **Tool-use benchmarks**（API-Bank, ToolLLM, BFCL, AppWorld, τ-Bench）：评分工具调用正确性或真实环境交互，但多因素同时变化，无法隔离单一压力效应。
- **受控工具负载研究**（ToolMenuBench, Tool-Sandbox, APIFlow-Bench）：分别研究检索、菜单大小、长链调用，但未记录失败时的 harness limit，本文补充此维度。
- **Harness 效应研究**（Harness-Bench, HAL, Model or Harness?）：报告每配置的 harness 影响、发现 harness 可重排模型排名，本文进一步量化压力效应中与 limit 共现的比例。
- **失败诊断与修复**（DoVer, REFLECT, Tool-RobustBench）：定位失败阶段或修补 harness 缺陷，本文通过配对修复测试与错误层对照验证修复特异性。
- **上下文长度效应**（Lost in the middle, LOCA-bench）：证明长上下文本身有害，本文区分菜单膨胀的截断效应与信息过载效应。
- **Benchmark 有效性审计**（StableToolBench, LayerRAG-Bench）：强调评分器自身需验证，本文提供生成器/评分器的静态与运行时审计。

## 局限性与未来方向
- 发现仅在固定 harness 预算下成立，episode budget 和 call limit 未被变量化（仅 completion cap 被扫掠）。
- 十六倍 cap 扫掠仅在 routed-menu 通道的 8 个配置上进行，残差菜单惩罚仍在未变的 episode budget 下测量。
- 任务为合成数据、精确评分无部分 credit，英语固定提示，结果描述特定模型-provider 配对，未必泛化到真实部署。
- 生成器与任务已公开，后续训练模型可能接触过测试数据。
- 未来可变量化 episode budget/call limit，测试 live MCP server，扩展多意图跨域验证。

## 研究启发与可借鉴点
- **压力-限制解耦框架**：将模型能力压力（菜单/工作流/编码）与系统限制（cap/budget/call limit）分离测量，为 agent 评估提供因果归因新思路。
- **失败记录联合标签设计**：同时记录"首个未满足义务"与"触发 limit"，比单一诊断标签更具可操作性，可直接复用于其他 agent 基准。
- **修复特异性检验协议**：L/Γ/Σ/Δ 四指标结合 intersection-union test 与 Holm 校正，可推广至任何 intervention 评估场景。
- **预算扫掠作为机制验证**：通过变量化单一预算参数（如 completion cap）来隔离截断效应，为性能下降归因提供干净实验。
- **配置切换基准比较**：在相同压力下比较修复与换配置，为工程实践提供成本-成功 Pareto 前沿，值得纳入 agent 评估标准流程。

## 关键术语表
- **Verified Task Success (VTS)**：九项义务合取的精确评分，任何一项失败即判定整集失败，无需模型裁判。
- **Assigned Pressure**：通过确定性编译器在固定任务族上施加的可控负载（R/V/W/D），用于因果效应估计。
- **Failure Record**：首个未满足义务与触发 harness limit 的配对标签，用于区分模型失败与系统限制。
- **Completion Cap**：单次请求最大生成 token 数（1,024），截断发生在首次响应未完成时。
- **Repair Activation**：修复在压力下特异性生效，需同时满足Assigned Loss(L)>0、Stress Gain(Γ)>0、Selectivity(Σ)>0。
- **Configuration**：模型与 provider route 及 harness 设置的组合，共 21 个配置来自 7 个机构。
- **Intention-to-Treat (ITT) Success**：catalog 压力下的评分，工具未检索到时直接计为失败（$A \cdot S$）。
- **Routed Menu**：首次执行操作工具后菜单收窄至仅操作工具的 oracle-assisted 策略。

## 可复现要素
- **数据集**：36 个任务族、26,576 个评分条件，公开于 https://github.com/zichuanxu/tool-bottleneck-bench（MIT License）。
- **代码**：确定性生成器、精确评分器、分析代码、frozen plans 均开源。
- **权重**：使用托管推理端点（21 个配置），未提供本地权重，需 provider 凭据运行新 episode。
- **关键超参**：completion cap=1,024 tokens，episode budget=128,000 tokens，call limit=认证最小+2，turn limit=call limit+2，parallel_tool_calls=false，seed condition-bound。
