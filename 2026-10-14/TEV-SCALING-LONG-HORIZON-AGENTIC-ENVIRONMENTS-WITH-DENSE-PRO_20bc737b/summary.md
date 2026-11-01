---
title: "TEV-SCALING-LONG-HORIZON-AGENTIC-ENVIRONMENTS-WITH-DENSE-PRO"
source: https://arxiv.org/pdf/2610.18868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 19:08:00"
field: "长程Agent强化学习"
keywords: ["long-horizon agent", "reinforcement learning", "reward shaping", "synthetic data", "terminal agent", "GRPO", "progress potential", "TEV-GYM"]
innovations: ["潜在世界优先合成：以完整latent world为单一anchor反向生成任务/verifier/progress scorer，消除artifact drift", "PPRS密集进度奖励：基于workflow DAG的Φ(s)将长程稀疏奖励转化为每步shaping信号，修复GRPO全失败组梯度饥饿", "TEV-GYM数据集：1,170个含staged progressive reward的7域长程agent训练环境，median 12.4 min per rollout"]
benchmarks: ["ALE-CM", "Terminal-Bench 3.0 CM", "Terminal-Bench 4.0 CM", "Terminal-Bench 2.1"]
---

# 论文速读：TEV-SCALING-LONG-HORIZON-AGENTIC-ENVIRONMENTS-WITH-DENSE-PRO

## 一句话总结
论文提出"潜在世界优先"合成方法与进度潜力奖励塑造（PPRS），构建了首个含渐进式密集奖励的长程agent训练数据集TEV-GYM（1,170个环境），训练出的TEV-27B在ALE-CM、Terminal-Bench 3.0/4.0等长程任务上达到GPT-6 Astra / Claude Opus 5同等水平，并实现零样本泛化。

## 研究问题与动机
1. **长程agent训练数据匮乏**：现有开源agent训练语料（SWE-Smith、TMAX、SETA等）均为短程/中程任务（median agent time 0.5–2.4 min），真正长程任务仅存在于小规模人工评测集（ALE、Terminal-Bench），缺乏可训练的大规模语料。
2. **奖励稀疏导致GRPO失效**：长程任务仅在提交时给出pass/fail信号，GRPO组内若所有G=8条rollout均失败，advantage全为零，贡献无效梯度；未训练open model在长程任务上p≈0.01–0.05，浪费组比例高达92%/66%。
3. **任务合成难以同时满足可验证性与复杂度**：传统LLM合成方案（prompt-first、public-view verifier）无法保证ground truth精确性，存在artifact drift，且缺少可控的难度阶梯与进度信号。
4. **开放模型差距显著**： frontier模型在Terminal-Bench 4.0 CM子集上P@1可达0.52，而开源模型不足0.04，需要可复用的训练配方。

## 核心贡献（创新点）
1. **"潜在世界优先"合成流水线**：先构建完整的隐藏工作流DAG（latent world W），再反向渲染输入文件并施加可控损坏，使得任务、verifier、progress scorer均为W的确定性函数，从根本上避免artifact drift。
2. **TEV-GYM：首个长程agent训练数据集**：1,170个经过8项qualification gate验证的环境，覆盖7个领域（软件工程、数据工程、AI/CS研究、数学与运筹、安全取证、基础设施管理、量子计算），median agent time 12.4 min，具备渐进式奖励Φ。
3. **PPRS（Progress-Potential Reward Shaping）**：基于latent world定义progress potential Φ(s)∈[0,1]，构造r_t=Φ(s_t)−Φ(s_{t−1})的密集每步奖励R(τ)=1[passes]+0.25·Φ(s_T)，修复GRPO在全失败组中的梯度饥饿，且保留passing rollout的严格最高排名。
4. **TEV-27B及配方可迁移性**：在TEV-GYM全量语料上以DPPO训练Qwen3.8-27B，ALE-CM P@5=0.28匹配GPT-6 Astra/Claude Opus 5；同一配方应用于GLM-5.3-Flash得到TEV-320B，在TB 3.0/4.0 CM子集上超越GLM 5.2和Gemini 3.7 Flash，成为最强开源模型。

## 方法详解
**Latent-world-first synthesis（6个阶段）**：
- **Stage 0 任务源**：从ALE-CM 7个子域随机采样规范，编写确定性seeded generator（coding agent辅助撰写，人类审核）。
- **Stage 1 世界构建**：生成完整latent world W（含entity关系、tensor shard、事务记录等），通过invariant校验（唯一标识、有效join、有限度量）；同步抽样defect plan。
- **Stage 2 工作流DAG**：以纯代码编译workflow DAG，节点为可检查的中间状态，critical path定义horizon，DAG同时决定progress potential的节点结构。
- **Stage 3 Artifact渲染**：将W投影到公开接口（CSV/TSV/JSON/SQLite/safetensors/binary等23种wire format），断言round-trip可逆。
- **Stage 4 控制性损坏**：按defect plan注入schema drift、重复/缺失记录、stale cache等，生成agent收到的s_0。
- **Stage 5/6 公开契约与隐藏套件**：agent只看到task prompt x；hidden suite包含metamorphic transforms（语义保真重写）与stale-output counterfactuals用于verification。
- 任一环境E=(x, s_0, ver_W, Φ_W)均为W的函数，杜绝artifact drift。

**Progress Potential Φ(s)**（Eq. 3）：
- 节点集V每个节点v有partial scorer c_v(s)∈[0,1]，credit由prerequisite-gated最低分约束：c̃_v(s)=min_{u∈pre(v)∪{v}} c_u(s)。
- Φ(s)=∑_v w_v·c̃_v(s)，满足：anchored（未损坏s_0得Φ=0，通过ver得Φ=1）、monotone（沿参考轨迹单调升，回退时降）、role-independent（scorer与verifier/solver代码闭包不相交）。
- 14类gaming probe须得Φ=0防reward hacking。

**PPRS训练公式**（Eq. 4）：
- 每步shaping reward r_t=Φ(s_t)−Φ(s_{t−1})；rollout reward R(τ)=1[passes]+λ·Φ(s_T)，λ=0.25。
- 提交为action而非state，未提交但完成工作的rollout同样获得Φ(s_T)奖励。
- 代入GRPO的组归一化advantage（Eq. 2），全失败组因Φ(s_T)各异而获得非零advantage，修复梯度饥饿。
- Turn-index normalization保留后续步信用，γ=1传播progress到早期动作。

**训练设置**：DPPO，G=24 rollouts/任务，8任务/step（192 rollout/step），65,536 token响应预算，≤64 tool calls；TEV-9B用unit tier 264任务，TEV-27B用full corpus；BF16+ZeRO-3，lr=10⁻⁶。

## 实验与结果
**数据集规模**：1,170环境，7域，median 6 workflow stages / 12 input files / 2 corruptions；base Qwen3.8-27B solve率仅3.2%，frontier Gemini 3.6 Flash P@1=29%、P@8=40%。

**主结果**：
| 模型 | ALE-CM P@5 | TB 3.0 CM P@1 | TB 4.0 CM P@1 | TB 2.1 P@1 |
|---|---|---|---|---|
| TEV-9B | 0.11（+83% over base） | — | — | 0.227（+28%） |
| TEV-27B | 0.28（match GPT-6 Astra/Claude Opus 5） | 0.14（+56%，>GLM 5.2/Gemini 3.7 Flash） | 0.10（+150%） | 0.794（match Claude Opus 4.8） |
| TEV-320B | — | 0.27（+35% over GLM-5.3-Flash） | 0.37（+28%） | — |

- TB 4.0 CM子集上TEV-320B仅次于Claude Opus 5、Claude Fable 5、GPT-5.6(sol)。
- TB 3.0/4.0 CM与全benchmark Pearson r≥0.96，证明子集代表性。
- Train/eval decontamination：0个13-gram或8-gram重叠，median NN Jaccard仅0.13（TB）/0.22（ALE-CM）。

**消融**：
- 数据ablation：prompt-first和public-view verifier均不如PPRS（S@1 0.23–0.24 vs. 0.29）；后者导致88%组被丢弃。
- 奖励ablation：sparse terminal-only在27B上几乎产生全零advantage组无法完成优化步；PPRS turn-index norm+γ=1为最优平衡。

## 相关工作脉络
1. **SWE-Smith/TMAX/SETA等短程agent语料**：以仓库或trajectory为锚，LLM生成任务与测试，验证仅terminal pass/fail或test fraction，无progress signal；TEV从latent world反向生成多阶段工作流并附Φ。
2. **Backward construction（BART、denoising、bug injection）**：传统逆向合成处理单个artifact（文本/代码），TEV损坏的是整个latent world的多模态渲染，ground truth为world本身而非预存答案。
3. **Terminal-Bench / ALE / PaperBench等长程评测集**：人工撰写，规模小（数十至百例），progress credit仅hand-written checkpoints或LM judge；TEV提供可自动生成的密集Φ信号。
4. **Process reward models / learned judges（PRM、Cui et al. 2025）**：依赖额外训练critic或LLM judge评分每步；TEV的Φ由独立scorer直接计算，无需训练奖励模型。
5. **AutoForge / EnvScaler / LOGIGEN等环境合成**：多为短工具调用episode，仅有pass条件或DB proximity奖励；TEV覆盖7域多阶段工作流且每个环境内含staged Φ。
6. **CLI-Gym / Terminal-Universe**：从running environment rewind单一状态，止于pass/fail；TEV保留每stage的精确可评中间态。

## 局限性与未来方向
1. **合成环境仍局限于terminal沙箱**：不支持GUI、真实服务交互、GPU kernel、多模态视觉任务（仅Example 1含dashboard screenshot）。
2. **与frontier模型的绝对差距仍大**：TB 4.0 CM上TEV-27B P@1=0.10 vs. Claude Opus 5的0.52，需更强open base model才能充分吸收高难度数据。
3. **新域扩展成本较高**：每新增领域需编写generator并通过8项qualification gate，尚未完全自动化。
4. **评估子集覆盖有限**：仅选取"computing-and-math"子集，engineering/finance/multimodal等ALE域未纳入训练，泛化边界待验证。
5. **长程token预算仍会耗尽**：base model 52% rollout在65k token前耗尽，未完成提交即获得Φ奖励但无terminal credit。

## 研究启发与可借鉴点
1. **"潜在世界优先"合成范式可迁移**：任何需多阶段工作流验证的任务（如MLOps pipeline、合规审计、科学计算复现）均可套用：先生成ground truth world→DAG→渲染+损坏→构造Φ，避免artifact drift和验证器-任务错配。
2. **PPRS作为通用dense reward配方**：任何含可计算中间状态的长程任务（RLVR、step-wise verification）均可定义Φ并叠加PBRS-style shaping，修复GRPO/PPO在全失败组的梯度饥饿，λ可通过小规模调参确定。
3. **8项qualification gate体系值得借鉴**：exploit resistance、role independence、solver agreement等gate可作为合成数据集的质量门禁，防止reward hacking和verifier-solver共享bias。
4. **strength-2 covering array覆盖策略**：用pairwise覆盖轴确保多样化而非单轴平衡，可推广至任何合成数据curriculum设计。
5. **训练-评测decontamination协议**：13-gram/8-gram lexical overlap测量+NN Jaccard benchmark calibration，为后续团队提供可复用的数据泄露检测模板。

## 关键术语表
- **Latent world (W)**：任务域的完整正确隐藏规范（含所有entity、关系、中间状态），是任务、verifier、progress scorer的共同确定性源。
- **Progress potential Φ(s)**：基于workflow DAG节点定义的进度评分函数Φ(s)∈[0,1]，衡量agent当前workspace与latent world的吻合度。
- **PPRS (Progress-Potential Reward Shaping)**：将Φ的逐步变化r_t=Φ(s_t)−Φ(s_{t−1})作为dense reward，与terminal pass组合为R(τ)。
- **TEV-GYM**：首个含渐进式奖励的长程agent训练数据集，1,170个通过8项gate验证的环境，覆盖7域。
- **Group-relative policy optimization (GRPO)**：Shao et al. 2024提出的组内归一化advantage估计器，对稀疏奖励敏感，全失败组advantage为零。
- **DPPO (Trust-region DPPO)**：本文采用的RL优化器，TV divergence bound=0.1，zero-variance组active sampling丢弃并替换。
- **Artifact drift**：任务、输入、verifier分别生成导致描述不一致的问题，latent-world-first方案通过W作为单一anchor消除此风险。
- **Metamorphic invariance**：正确输出的语义保真重写不改变grade，stale-output counterfactuals必须判负，用于验证verifier合理性。

## 可复现要素
- **数据集**：TEV-GYM为确定性seeded程序生成，每环境median 30s CPU时间、无model call；论文附录A.10提供3个完整环境 walkthrough，代码/生成器未公开声明（仅提供论文描述）。
- **代码/权重**：基座模型Qwen3.5-9B、Qwen3.8-27B、GLM-5.3-Flash为公开权重；TEV训练后权重未声明开源；训练配方（DPPO超参Table 11）完整给出。
- **关键超参**：G=24、8 tasks/step、response budget=65,536 tokens（per turn 16,384）、≤64 tool calls、λ=0.25、lr=10⁻⁶、TV bound=0.1、γ=1、one epoch/batch。
- **评估harness**：Vanillux2（single bash tool）用于开源模型；frontier模型使用native harness（Codex/Claude Code/Gemini CLI）；TB CM子集选取协议见Appendix C.2。
