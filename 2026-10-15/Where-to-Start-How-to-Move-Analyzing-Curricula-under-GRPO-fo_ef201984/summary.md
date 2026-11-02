---
title: "Where-to-Start-How-to-Move-Analyzing-Curricula-under-GRPO-fo"
source: https://arxiv.org/pdf/2610.19349v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:13:56"
field: "大语言模型智能体训练"
keywords: ["GRPO", "curriculum learning", "agent RL", "tool calling", "long-horizon", "guide-assisted start", "Thompson sampling"]
innovations: ["提出引导辅助起始课程，系统解耦起始位置、交接粒度与移动规则对GRPO训练的影响", "结果感知的工具调用级选择（Call-Bandit）在训练与未见类别上持续超越GRPO", "证明末端起始在固定阈值规则下更有效，且可节省训练GPU时间"]
benchmarks: ["BFCL", "AppWorld"]
---

# 论文速读：Where-to-Start-How-to-Move-Analyzing-Curricula-under-GRPO-for-Long-Horizon-LLM-Agents

## 一句话总结
论文系统研究了在 GRPO 训练中引入**引导辅助起始（guide-assisted starts）**的课程设计，通过回放成功参考轨迹的前缀来改变学习起点、交接粒度与移动规则，以缓解长_horizon 任务中早期全失败组导致的学习信号缺失问题。实验表明，**在固定阈值移动规则下，从靠近任务末端起始更有效；基于结果感知的工具调用级选择（Call-Bandit）在训练类别与未见类别上均持续超越原始 GRPO**，且该方法在交互式编码任务（AppWorld）上同样有效。

## 研究问题与动机
1. **GRPO 在长_horizon 多轮交互中的零优势问题**：工具调用序列的奖励仅在任务结束时给出，早期全失败组会导致组内优势为零，学习停滞。
2. **现有替换全同组策略的局限性**：DAPO 通过重新采样替换相同奖励组，但在同一策略下重试仍可能全部失败，无法根本解决信号稀疏性。
3. **起始状态课程设计对最终无辅助完成能力的影响尚不明确**：何时开始学习、以何种粒度交接、如何根据历史结果移动起点，这些超参数组合缺乏系统研究。
4. **效率与泛化的双重需求**：如何在减少训练 GPU 时间的同时，提升模型在训练类别（Base）与未见交互条件（Missing Function/Parameter、Long Context）及跨任务类型（交互式编码）上的泛化性能。

## 核心贡献（创新点）
1. **提出引导辅助起始课程框架**：通过回放成功参考轨迹的前缀至交接点 h 再让 learner 续写，为 GRPO 提供可调节的学习起点与中间学习信号，与 DAPO 等纯替换组的方法本质不同，直接改变 rollout 的起始分布。
2. **系统解耦并对比三个课程维度**：分别分析起始位置（Start/End）、交接粒度（Turn/Call）和移动规则（Threshold/Uniform/Bandit），揭示它们的交互效应，此前工作多孤立考察单一维度。
3. **发现末端起始在固定阈值规则下显著优于起始位置**：在 Turn 级别移动规则下，End 初始化在 1.7B/4B/8B 模型上均获得更高的 Avg@3，最大提升 7.0 个百分点，证明“从哪里开始学”是关键超参数。
4. **结果感知的工具调用级选择（Call-Bandit）带来一致增益**：利用 Thompson sampling 选择能产生混合成败结果的交接点，在 BFCL Base 与未见类别 Macro Avg 上均以 2.4–4.7 个百分点超越 GRPO，且在全训练周期内维持更低的全失败组比例。
5. **将引导课程扩展至交互式编码任务（AppWorld）**：证明该方法不仅适用于函数调用，还能在更长的代码执行序列上提升成功率，并节省约 21% 的训练 GPU 时间。

## 方法详解
- **引导辅助起始（Guide-assisted starts）**：每条 rollout 重置环境后，回放一条成功的参考轨迹前缀，直至交接点 $h$，之后 learner 独立续写至 episode 结束。回放部分改变环境状态并提供历史对话，learner 无法访问未执行的参考后缀。
- **交接粒度（Handoff unit）**：
  - **Turn**：在 BFCL 的用户轮次边界交接（如支付请求前）。
  - **Call**：在工具调用级别的验证边界交接，粒度更细（如 lookup 完成后）。
- **初始化策略**：
  - **Start**：$h = 0$，无回放，等同于普通 GRPO。
  - **End**：回放至参考轨迹最后一个用户轮次之前，learner 仅完成最后一步。
- **移动规则（Movement rules）**：每组包含 16 条来自相同 $h$ 的续写。
  - **Threshold**：若成功数 $\in [12,16]$ 则提前一步，若 $\in [0,1]$ 则推后一步，否则不动；步长取决于交接粒度（Turn 移动一个用户轮次，Call 移动一个工具调用；全成功组直接移至前一个用户轮次边界）。
  - **Uniform**：在可用交接点中均匀采样，不依赖历史结果。
  - **Bandit**：维护每个任务-位置对的成败计数，以 Beta(1,1) 为先验更新成功概率 $p$，计算得分 $s(p)=1-p^{16}-(1-p)^{16}$（即出现至少一成一败的概率）；以概率 0.1 探索最少观测位置，否则选择得分最高且差距 $\le 0.1$ 的最早位置。
- **训练细节**：仅 learner 生成的 token 计算策略梯度损失；终端二元奖励评估整个 episode（含回放前缀）；全失败/全成功组被过滤出相对优势更新，但计入 rollout 预算并用于移动决策；每组分 16 个续写，AdamW(lr=1e-6, wd=0.01)，温度 1.0，最大生成 token 4096，BF16 精度。

## 实验与结果
- **数据集与模型**：BFCL（Base 训练，100 个任务；测试集含 Base、Missing Function、Missing Parameter、Long Context 三类泛化评估）与 AppWorld（90 训练任务，Test Normal 168 任务，Test Challenge 417 任务）；模型为 Qwen3-1.7B、4B-Instruct-2507、8B。
- **主要结果（BFCL Base，Avg@3 %）**：
  - **初始基线**：GRPO (1.7B:16.3, 4B:23.7, 8B:38.0)；DAPO (1.7B:22.3, 4B:54.3, 8B:52.0)。
  - **Turn-Threshold**：Start (1.7B:22.0, 4B:53.7, 8B:55.0)；**End** (1.7B:24.3, 4B:55.0, **8B:59.0**) → 8B 相对 GRPO 提升 **+21.0 个百分点**，相对 Start 提升 4.0 个百分点。
  - **Call 级别（End 初始化）**：Threshold (1.7B:23.3, 4B:54.3, 8B:56.0)；Uniform (1.7B:29.0, 4B:56.7, 8B:56.0)；**Bandit** (**1.7B:30.0**, **4B:60.0**, **8B:60.7**) → 8B 相对 GRPO 提升 **+22.7 个百分点**，相对 Turn-Threshold 提升 5.7 个百分点。
- **跨类别泛化（Macro Avg., 未见类别，%）**：Call-Bandit 在 1.7B/4B/8B 上分别为 19.8/42.7/49.7，相对 GRPO 提升 **2.4–4.1 个百分点**，且全面超越 Call-Threshold 与 Uniform。
- **AppWorld 交互编码（Qwen3-8B，成功率 %）**：GRPO (Avg@3 Test Normal:43.1, Test Challenge:20.9)；**Call-Threshold** (47.2 / 22.3) → 提升 **+4.1 / +1.4 个百分点**，Best@3 提升 +5.9 / +2.9 个百分点，训练 GPU 时间减少约 **21%**。
- **训练动态**：Call-Bandit 在前 16k 轮快速提升后持续进步，最终验证准确率最高；全失败组比例从 GRPO 初期的 43% 降至 Call-Bandit 的 33%，期末分别降至 23% vs 12%，显示课程有效维持了组内变异。

## 相关工作脉络
1. **DAPO (Yu et al., 2025)**：通过重新采样替换组内奖励相同的 rollout 以维持优势信号；本文与其本质区别在于不依赖重试替换，而是通过回放已知成功前缀主动构造含奖励变异的新起始状态。
2. **JSRL (Uchendu et al., 2023)**：在通用 RL 中通过跳跃到中间状态加速学习；本文将其思想迁移至 LLM agent 的工具调用序列，并细化出 Turn/Call 两种交接粒度与多种移动规则。
3. **R³ (Xi et al., 2024)**：利用反向课程强化学习进行语言模型推理训练；本文与之不同在于关注的是交互式 agent 的工具使用而非纯推理链，且引入了结果感知的动态移动机制。
4. **Agent-G² (Wang et al., 2026)**：使用 Gaussian guidance 辅助 agent RL；本文不使用外部价值模型或指导分布，而是依赖参考轨迹前缀与基于 Thompson sampling 的位置选择。
5. **BREAD (Zhang et al., 2025) / PivotRL (Yi et al., 2026)**：前者结合自适应专家前缀与分支 rollout，后者在中间轮次进行局部 on-policy  rollout；本文保持完整 episode 续写与全局终端奖励，简化了监督信号设计。
6. **GiGPO (Feng et al., 2025)**：在重复环境状态处补充组间优势比较；本文聚焦于起始位置课程而非优势计算方式的修改，两者正交可结合。

## 局限性与未来方向
- **评估范围有限**：仅在 BFCL 和 AppWorld 两个基准及 Qwen3 系列模型上验证，未涵盖其他模型家族（如 Llama、Gemini）与不同交互模式（如 GUI 操作、多智能体协作）。
- **混淆变量未完全隔离**：Turn/Call 配置同时差异了移动步长与历史结果设置，无法单独剥离交接粒度本身的效应。
- **方差估计不足**：仅对每个任务重复评估三次以捕获生成方差，但未报告独立训练运行之间的方差，结论的统计稳健性有待进一步验证。
- **依赖成功参考轨迹**：当前方法需要可用的成功轨迹作为 guide；在无此类轨迹的场景下，需借助外部模型生成候选 guide 并在环境中验证，增加了收集成本与 guide 质量的不确定性。
- **安全与部署未评估**：研究仅测量基准环境中的任务完成度，未涉及真实账户访问、敏感信息暴露或不当工具调用等安全风险，实际部署需额外安全机制。

## 研究启发与可借鉴点
1. **起始状态课程可作为 GRPO 类算法的通用插件**：任何基于组内相对优势的 RL 训练（尤其长 horizon 任务）均可接入 guide-assisted starts，通过调节 $h$ 分布控制学习难度曲线。
2. **结果感知的动态移动机制（Bandit）值得迁移**：基于 Beta-Bernoulli 模型选择“最可能产生混合结果”的交接点，可推广至其他需要探索-利用平衡的 curriculum learning 场景。
3. **全失败组比例可作为训练健康度的轻量监控指标**：文中显示该比例从 43% 降至 12% 与最终性能正相关，后续研究可将其纳入 early-stopping 或自适应阈值调整。
4. **交互粒度（Turn vs Call）与移动规则的协同设计**：更细粒度本身不保证更好，必须与匹配的移动策略（如 Call 搭配 Bandit）结合；这提示课程设计需联合优化而非孤立调参。
5. **训练效率提升可直接复用**：End 初始化使 learner 仅生成短续写，在 BFCL 8B 上节省约 8% GPU 时间，在 AppWorld 上节省约 21%，为资源受限训练提供了低成本优化路径。

## 关键术语表
- **GRPO (Group Relative Policy Optimization)**：通过比较同任务多 rollout 的组内奖励相对优势进行策略优化的 RL 方法，无需额外价值网络。
- **Guide-assisted start**：回放成功参考轨迹的前缀至交接点，让 learner 从中间状态续写完成 episode 的训练起始方式。
- **Handoff unit (Turn/Call)**：课程中交接粒度的划分，Turn 指用户轮次边界，Call 指工具调用级别的细粒度边界。
- **Threshold movement**：根据每组成功数（≥12 提前、≤1 推后、否则不动）沿交接点序列规则移动起始位置的策略。
- **Call-Bandit**：基于 Thompson sampling 选择交接点，以最大化“组内既有一成又一败”概率的随机移动策略。
- **BFCL (Berkeley Function Calling Leaderboard)**：评估 LLM 工具调用能力的 benchmark，包含 Base、Missing Function、Missing Parameter、Long Context 等类别。
- **AppWorld**：包含交互式编码任务的 benchmark，agent 需执行 Python 代码块与虚拟应用环境交互。
- **Macro Avg.**：对未见类别（Missing Function、Missing Parameter、Long Context）取平均，用于衡量跨分布泛化能力。

## 可复现要素
- **数据集**：BFCL（公开，Apache 2.0 协议）与 AppWorld（公开部分 Apache 2.0，受保护部分需加密分发）；训练任务数 BFCL 100 个 Base，AppWorld 90 个。
- **代码/框架**：训练使用 molt 框架（Hu et al., 2026，开源），rollout 生成使用 vLLM（Kwon et al., 2023，开源）；具体课程实现未单独开源，需依附录 B 复现。
- **模型权重**：Qwen3-1.7B、Qwen3-4B-Instruct-2507、Qwen3-8B（公开，Apache 2.0）。
- **关键超参**：AdamW lr=1e-6、wd=0.01；每组 16 个续写；采样温度 1.0；最大生成 token 4096；BF16 精度；硬件为 4×NVIDIA B200 或 4×NVIDIA H100。
- **训练轮次**：BFCL 与 AppWorld 均报告至最终 checkpoint（AppWorld 约 129.6K generated rollouts）；具体总步数论文未明确给出。
