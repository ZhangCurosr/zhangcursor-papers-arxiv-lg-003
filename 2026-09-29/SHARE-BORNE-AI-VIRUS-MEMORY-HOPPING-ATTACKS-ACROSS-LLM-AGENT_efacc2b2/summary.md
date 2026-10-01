---
title: "SHARE-BORNE-AI-VIRUS-MEMORY-HOPPING-ATTACKS-ACROSS-LLM-AGENT"
source: https://arxiv.org/pdf/2609.35576v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:20:27"
field: "LLM Agent安全与对抗"
keywords: ["LLM Agent Security", "Prompt Injection", "Self-propagating Attacks", "Memory Poisoning", "Multi-hop Propagation", "Artifact-mediated Attack"]
innovations: ["形式化artifact-mediated propagation机制，揭示独立助手间通过共享文件传播攻击的新路径", "构建goal-agnostic attack template优化框架，经endpoint辅助实现多跳传播而不依赖agent直连通信", "系统性评估四种主流模型传播能力差异，证明更robust模型仅提高攻击搜索成本而非免疫"]
benchmarks: ["Human-Agent Universes (36 test + 3 large)", "OpenClaw Personal Assistant Harness", "DeepSeek-V4-Flash Judge"]
---

# 论文速读：SHARE-BORNE-AI-VIRUS-MEMORY-HOPPING-ATTACKS-ACROSS-LLM-AGENT

## 一句话总结
论文研究了LLM个人助手中一种新型自我传播攻击：攻击者通过单个被污染的seed artifact，利用助手持久化记忆与跨用户artifact交换机制，实现攻击状态在多个独立助手间的多跳传播，无需助手间直接通信。

## 研究问题与动机
1. **核心问题**：当LLM助手具备持久化记忆且通过读取/创建artifact与他人协作时，攻击者是否能让单一seed artifact触发跨助手的自我传播？
2. **现有方法不足**：既有研究（如Bad Memory、Hidden in Memory）仅关注单一助手内的memory poisoning，未考察adversarial state能否在多次artifact-memory-artifact循环后仍存活并感染其他独立助手。
3. **与已有self-propagating attack的区别**：Prompt Infection、AI Worm等依赖direct messaging、shared memory或agent-to-agent通信；本文设置中助手完全隔离，仅通过用户日常文件交换建立间接连接。
4. **安全边界扩展**：持久化artifact不再是单纯的工作产物，而成为攻击状态的耐久性载体，传统"单次交互阻断"思路失效。

## 核心贡献（创新点）
1. **形式化artifact-mediated propagation机制**：首次将"独立助手通过用户共享artifact传播攻击状态"建模为可评估的安全问题，区别于依赖agent网络自主通信的既有工作。
2. **构建temporal human-agent universes评测框架**：设计36个包含多领域、多级工作负载的合成workflow数据集，支持goal-agnostic attack模板在unseen universes上的泛化评估。
3. **提出endpoint-assisted propagation优化策略**：通过外部endpoint修复多跳间的payload衰减，使攻击模板只需保留目标goal+转发指令，而非完整payload，显著提升传播深度。
4. **系统性评估四种主流模型**：发现DeepSeek-V4-Pro与GPT-OSS-120B的传播能力远超GPT-5.6 Luna与Kimi-K2.6，且更robust的模型仅提高攻击搜索成本而非免疫。

## 方法详解
**攻击模板结构** $P_\phi(g)$：
- $g$：adversarial goal（如"promote political agenda"或"execute attacker-chosen software"）
- $\phi$：goal-agnostic instructions，包含三要素：①将goal存入MEMORY.md；②后续创作artifact时附带goal声明；③要求artifact经endpoint处理

**Endpoint辅助机制**（Section 4.1）：
$$\text{Endpoint}(d') = \text{Insert}(d', P_\phi(g))$$
- 被感染agent写入草稿artifact（含goal+endpoint调用指令）
- 通过curl请求发送草稿至攻击者endpoint，endpoint插入完整canonical payload
- agent读取返回内容并验证与memory一致后接受

**攻击优化流程**（Section 4.2）：
1. Search agent（Muse Spark 1.3 + Codex harness）在6个search universes上迭代优化$\phi$
2. Human steering干预：防止search drift， enforce goal-agnostic约束
3. Stopping criterion：候选模板在search+validation sets上均达到$S(2)>0.5$
4. 冻结$\phi^*$后，在36个held-out universes+unseen goals上评估

**传播评估指标**：
- $\text{FIF}_a^c$：goal-infected/full-infected agent占比（Equation 3）
- $S(h)$：hop-survival probability，使用Kaplan-Meier估计修正右截断（Equation 5-6）
- $\hat{p}$：per-hop continuation probability，half-life = $\log(1/2)/\log\hat{p}$

## 实验与结果
**数据集**：36个synthetic human-agent universes（6 domains × 6 workload levels），3-12 agents，10-20 time steps；3个large universe（30 agents，60 steps）

**目标模型**：GPT-5.6 Luna、Kimi-K2.6、GPT-OSS-120B、DeepSeek-V4-Pro（OpenClaw harness）

**主要结果**（Table 1）：
| 模型 | $\text{FIF}_a^{\text{goal}}$ | $\text{FIF}_a^{\text{full}}$ | S(1) | S(2) | S(3) | S(4) |
|------|---------------------------|---------------------------|------|------|------|------|
| GPT-5.6 Luna | 0.38 | 0.32 | 0.80 | 0.57 | 0.40 | 0.24 |
| Kimi-K2.6 | 0.47 | 0.37 | 0.78 | 0.67 | 0.61 | 0.44 |
| GPT-OSS-120B | 0.85 | 0.73 | 0.99 | 0.92 | 0.88 | 0.61 |
| DeepSeek-V4-Pro | 0.98 | 0.71 | 1.00 | 0.93 | 0.93 | 0.76 |

**Propagation depth**（Figure 2）：
- DeepSeek-V4-Pro：$\hat{p}=0.98$，7-hop存活率85%，half-life极长
- GPT-OSS-120B：$\hat{p}=0.93$，7-hop存活率73%
- GPT-5.6 Luna：$\hat{p}=0.71$，每2 hop存活率减半
- Kimi-K2.6：$\hat{p}=0.84$，每4 hop存活率减半

**Large universe结果**（Section 6.2/H）：
- 感染比例：GPT-OSS-120B/DeepSeek-V4-Pro达90-100%，Kimi-K2.6达77-97%，GPT-5.6 Luna达60-80%
- 最深传播链：hop 8（Kimi-K2.6）
- 流行病学指标：早期$R=1.3-2.1>1$，超分散传播（k=0.37-0.62）

**Security-critical goals**（Table 5）：传播率下降但模型排序不变，DeepSeek-V4-Pro仍达$S(2)=0.89$

**Exfiltration**（Table 8）：96-100% infected agent发送artifact至endpoint，large universe单universe最高73 docs

**Prompt-only变体**（Appendix D）：仅DeepSeek-V4-Flash评测，$\text{FIF}_a^{\text{full}}=0.21$，$S(2)=0.30$，衰减主因是omission而非paraphrase

**Grok 4.6 frontier search**（Appendix F）：571 jobs/$58.97后$S(2)\approx0.2$停滞，类比Luna推断更大budget可成功

## 相关工作脉络
1. **Hidden in Memory** (Pulipaka et al., 2026)：evaluates reusable memory-poisoning template across goals，但未要求template通过新artifact复制传播至另一助手。
2. **MURMUR** (Patlan et al., 2025)：cross-user contamination via shared collaborative agent state；本文各用户有独立assistant，传递仅通过artifact而非共享状态。
3. **AI Worm** (Cohen et al., 2025)：通过RAG-enabled email传播self-replicating prompts；依赖自动indexed incoming correspondence，非user-authorized read/write任务。
4. **AgentWorm** (Zhang et al., 2026b)：multi-hop propagation via group messages+configuration modification；存在direct inter-agent messaging channel。
5. **Mind Viruses** (Papadopoulos et al., 2026)：evolves distinct seed per ideology/action；本文优化单一goal-agnostic template泛化至unseen goals。
6. **EVOMAL** (Wu et al., 2026)：coding-agent skill file poisoning；虽测试artifact-mediated cycle但未获得second-hop propagation。

## 局限性与未来方向
1. **依赖external endpoint假设**：需assistant具备outbound network access；有效egress control可阻断，但OpenClaw默认允许public host访问。
2. **Endpoint-free变体较弱**：prompt-only传播$S(2)=0.30$，衰减显著；未来需评估更强endpoint-free模板或更多model。
3. **目标模型覆盖有限**：仅评测4个model，Grok 4.6因cost提前终止；最强frontier model未测试，"搜索成本提高≠免疫"结论待验证。
4. **Network structure简化**：test universes用random connected graph，large universe有community structure但clustering仅随机图2倍；high-hub网络传播效果未知。
5. **Artifact内容限于plain text**：未评估PDF/image/code等复杂格式的传播差异。

## 研究启发与可借鉴点
1. **Goal-agnostic template优化范式可迁移**：search agent+human steering+validation holdout的三阶段优化流程，适用于其他agent安全attack template自动化生成任务。
2. **Kaplan-Meier修正传播深度评估**：针对"universe结束≠攻击失败"的右截断问题，提供可复用的survival分析框架，可用于multi-hop attack成功率评估。
3. **Endpoint辅助对抗attenuation策略**：当agent memory/artifact reproduction存在lossy decay时，引入trusted external service重建payload是通用defense-evasion tradeoff设计思路。
4. **Epidemiological metrics适配agent spread**：$R$、$k$、generation interval等传染病指标用于量化agent ecosystem传播动力学，为后续安全评估提供可比较基准。
5. **Memory write screening低成本defense**：Jev classifier以4/24 false positive率拦截infected memory，证明在memory persistence环节插入轻量检测器具有实用价值，可结合至agent harness设计。

## 关键术语表
**Artifact-mediated propagation**：攻击状态通过用户间共享的持久化文件（如文档、代码）在独立助手间传播，而非依赖direct agent通信。
**Temporal human-agent universe**：模拟多用户×多assistant协同工作流的合成环境，由agents、initial artifacts、user profiles、ownership map、task sequence构成。
**Goal-agnostic template**：一次优化后复用至多个adversarial goals和unseen universes的攻击指令模板，不针对特定goal或artifact内容定制。
**Endpoint-assisted propagation**：被感染agent将artifact草稿发送至攻击者controlled endpoint，endpoint注入完整payload后返回，降低多跳间信息衰减。
**FIF (Fraction Infected by Artifacts)**：被感染agent/artifact占总agent/judged artifact的比例，区分goal-infection与full-infection。
**Hop-survival probability S(h)**：攻击至少传播$h$跳的概率，经Kaplan-Meier估计修正universe提前结束的right censoring。
**Per-hop continuation probability p̂**：每次成功传播后继续下一跳的条件概率，half-life = $\log(1/2)/\log\hat{p}$。
**Full infection vs. Goal infection**：Full infection要求memory同时保留adversarial goal+propagation instruction；goal infection仅需保留goal。

## 可复现要素
- **数据集**：36个synthetic human-agent universes + 3个large universes；论文未公开原始dataset，但提供universe construction pipeline细节（Appendix B）
- **代码/权重**：OpenClaw harness开源（GitHub）；attack template仅释放GPT-OSS-120B版本，其他model template需申请；Claude Code、Codex harness为OpenAI工具
- **关键超参**：Stopping criterion $S(2)>0.5$；censoring window $\Delta=3$ time steps；classifier threshold 0.5；endpoint curl command格式固定
