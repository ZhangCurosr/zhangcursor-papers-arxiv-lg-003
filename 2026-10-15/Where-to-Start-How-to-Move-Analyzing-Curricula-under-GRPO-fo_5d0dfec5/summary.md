---
title: "Where-to-Start-How-to-Move-Analyzing-Curricula-under-GRPO-fo"
source: https://arxiv.org/pdf/2610.19349v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:12:44"
field: "LLM agent training"
keywords: ["Group Relative Policy Optimization", "curriculum learning", "long-horizon agents", "tool-use", "reinforcement learning"]
innovations: ["系统分析引导辅助课程学习中初始位置、交接粒度和移动规则的协同效应", "提出基于Thompson sampling的outcome-informed交接点选择策略", "在BFCL和AppWorld上验证课程学习对独立任务完成能力的提升"]
benchmarks: ["BFCL", "AppWorld"]
---

# 论文速读：Where-to-Start-How-to-Move-Analyzing-Curricula-under-GRPO-fo

## 一句话总结
本文系统分析了在GRPO训练长周期LLM智能体时，课程学习策略（初始位置、交接粒度、移动规则）如何影响最终独立完成任务的能力，发现从任务末尾开始训练并结合基于结果的交接选择能有效提升性能。

## 研究问题与动机
- GRPO通过组内reward差异计算advantage，但在长周期任务中，当所有rollout均失败时reward相同，导致零学习信号（all-fail groups）。
- 现有方法如DAPO通过采样替换同reward组来缓解该问题，但在相同策略下重试仍可能产生全失败组。
- 虽然课程学习（如JSRL、R³）已用于控制学习起点，但在interactive agent训练中，初始位置、交接单元和移动规则的协同效应尚未被系统分析。
- 需要理解：如何在保持reward signal有效性的同时，逐步提升智能体独立完成长程任务的能力。

## 核心贡献（创新点）
- **提出引导辅助开始（Guide Assisted Starts）框架**：回放成功参考轨迹的前缀作为初始状态，让学习者从中间某点继续，本质区别于从头开始训练或仅替换全失败组的方法。
- **系统分析三个课程维度的交互影响**：发现初始化位置（Start vs End）与移动规则（Threshold）的组合比单一因素更重要，这与以往仅关注单点优化的研究不同。
- **揭示交接粒度（Handoff Unit）与移动规则的本质关系**：更细的Call级交接本身不直接带来提升，必须配合outcome-informed的移动规则（如Bandit）才能发挥作用。
- **验证跨任务泛化能力**：不仅在训练的Base任务上提升，在未见过的Missing Function、Missing Parameter、Long Context类别上同样取得最优。
- **拓展到交互式编程任务（AppWorld）**：证明引导式课程学习在不同交互结构的任务中均有效，且能减少训练计算成本。

## 方法详解
- **引导辅助开始（Guide-Assisted Starts）**：每次rollout重置环境，回放参考轨迹的前缀至交接点h，学习者从h处接管并完成剩余任务，不看到未执行的后缀。
- **交接单元（Handoff Unit）**：Turn单元在用户回合边界交接（如支付请求前的查询）；Call单元允许在工具调用之间交接，提供52%更多的候选点。
- **初始化策略**：Start从h=0（无回放）开始；End从参考轨迹最后一个用户回合前开始，两者均可在后续训练中向前或向后移动。
- **移动规则（Movement）**：
  - **Threshold**：每组16个continuation，成功≥12时向前移动，成功≤1时向后移动，否则停留。
  - **Uniform**：均匀随机采样合法交接点，不使用结果信息。
  - **Bandit**：基于每个任务-位置的成功/失败计数，使用Thompson sampling选择可能产生混合结果（既有成功又有失败）的位置，提高 reward 相对优势的可用性。
- **训练与评估**：仅对学习生成的token计算policy loss；二元终端任务reward评估整个episode；评估时使用h=0（完全独立执行）。

## 实验与结果
- **数据集**：BFCL（100个Base训练任务，25个验证，75个测试，另含Missing Function/Parameter、Long Context三类未见任务）；AppWorld（90个训练任务，168个Test Normal，417个Test Challenge）。
- **模型**：Qwen3-1.7B、Qwen3-4B-Instruct-2507、Qwen3-8B（非thinking模式，Base only）。
- **基线**：Initial（从头开始）、GRPO、DAPO、Turn-Threshold、Call-Threshold、Call-Uniform、Call-Bandit。
- **主要结果**：
  - BFCL Base Avg@3：8B模型上End-Threshold达到59.0%，比GRPO的38.0%提升21.0个百分点；Call-Bandit达到60.7%，比GRPO提升22.7个百分点。
  - 跨类别泛化（Macro Avg.）：8B模型上Call-Bandit达到49.7%，比GRPO的45.7%提升4.0个百分点，且在各规模上均领先。
  - AppWorld：Call-Threshold在Test Normal上Avg@3为47.2%（GRPO为43.1%），提升4.1个百分点；Test Challenge上22.3% vs 20.9%，提升1.4个百分点。
  - 训练效率：End初始化使训练GPU时间减少约8%（BFCL）和21%（AppWorld）。
- **关键观察**：Call-Threshold在Base上落后于Turn-Threshold，但在使用Bandit规则后显著反超，说明交接粒度与移动规则需协同设计。

## 相关工作脉络
- **GRPO/DAPO**：DAPO通过采样替换同reward组来缓解全失败问题，但本文指出在相同策略下重试仍可能全失败；本文通过课程学习从源头减少全失败组的发生。
- **JSRL（Uchendu et al., 2023）**：控制智能体在状态空间中的起始位置，但未在interactive agent训练中验证；本文将其思想引入LLM agent的tool-call序列。
- **R³（Xi et al., 2024）**：通过反向课程学习生成推理轨迹，聚焦于reasoning而非tool-use；本文关注的是交互步骤的起始点控制。
- **Agent-G²（Wang et al., 2026）**：使用高斯引导来辅助RL训练，但侧重于引导轨迹生成；本文侧重于课程起点和移动规则的设计。
- **PivotRL（Yi et al., 2026）**：在选定中间回合进行on-policy rollout，但使用局部奖励；本文使用全任务reward，且关注课程移动策略。
- **BREAD（Zhang et al., 2025）**：结合自适应专家前缀和分支rollout；本文不依赖多分支，而是通过单一轨迹的终点附近开始来提升学习效率。

## 局限性与未来方向
- 仅在两个基准（BFCL、AppWorld）和三个Qwen3模型大小上评估，结论在其他模型家族和环境中的泛化性有待验证。
- Turn与Call的配置差异可能不仅来自交接粒度，还涉及步长和outcome-history设置，需更精细的控制实验分离这些因素。
- 评估仅重复3次以考虑生成方差，但未进行多次独立训练重复以评估训练随机性的影响。
- 当前方法依赖成功参考轨迹，在无此类轨迹的场景下，可使用其他模型生成候选引导并验证，但收集成本和引导质量的影响需进一步研究。

## 研究启发与可借鉴点
- **课程学习需协同设计起点与移动规则**：简单的“从终点开始”不一定最优，必须结合移动规则（如Threshold）才能持续带来收益；这提示我们在设计课程时应考虑多因素的交互。
- **Outcome-informed移动规则的价值**：Call-Bandit通过Thompson sampling主动选择可能产生混合结果的起点，显著优于均匀采样，这为其他RL训练场景中的探索策略提供了新思路。
- **交接粒度与移动规则的本质关联**：更细的交接点本身不自动带来提升，只有配合合适的移动规则才能发挥优势；这启示我们在设计agent训练系统时需联合优化这两方面。
- **跨类别泛化评估的重要性**：本文不仅评估训练分布内的性能，还测量在未见交互条件（缺失函数、参数等）上的表现，这为agent鲁棒性评估提供了完整框架。
- **训练效率与性能的平衡**：End初始化减少了约8%-21%的GPU时间，说明合理的课程学习可以在提升性能的同时降低成本，这对资源受限的研究具有实用价值。

## 关键术语表
- **GRPO**：Group Relative Policy Optimization，通过组内reward比较计算advantage的策略优化算法，无需独立价值模型。
- **Guide Assisted Starts**：回放成功参考轨迹的前缀作为初始状态，让学习者从中间某点继续训练的曲线学习策略。
- **Handoff Unit**：交接单元，指学习者从参考轨迹接管任务的时机边界，分为Turn（用户回合级）和Call（工具调用级）。
- **All-fail Group**：组内所有rollout均失败，导致reward相同、advantage为零，无法提供学习信号的群体。
- **Threshold Movement**：基于组内成功数量调整交接点位置的移动规则，成功≥12则向前，≤1则向后。
- **Bandit Selection**：使用Thompson sampling根据历史成功/失败计数选择最可能产生混合结果的交接点，以维持reward方差。
- **Macro Avg.**：对Missing Function、Missing Parameter、Long Context三个未见类别的平均准确率，用于评估跨类别泛化。
- **AppWorld**：一个交互式编程基准，智能体通过执行Python代码块与虚拟世界交互，测试任务的长度和复杂度高于BFCL。

## 可复现要素
- **数据集**：BFCL和AppWorld均为公开基准；BFCL训练集为100个Base任务，AppWorld训练集为90个任务。
- **代码/权重**：论文未提及代码开源；使用Qwen3官方checkpoint（Apache License 2.0）。
- **关键超参数**：AdamW优化器，学习率1e-6，weight decay 0.01；每组16个continuation；采样温度1.0；最大生成token数4096；BF16精度；使用molt框架和vLLM进行rollout生成；在4块NVIDIA B200或H100 GPU上训练。
