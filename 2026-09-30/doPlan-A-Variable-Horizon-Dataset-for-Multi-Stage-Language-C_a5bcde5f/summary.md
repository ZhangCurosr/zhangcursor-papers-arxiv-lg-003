---
title: "doPlan-A-Variable-Horizon-Dataset-for-Multi-Stage-Language-C"
source: https://arxiv.org/pdf/2609.38028v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:35:54"
field: "自动驾驶语言条件规划"
keywords: ["language-conditioned planning", "autonomous driving", "long-horizon intent", "multi-stage instruction", "doPlan", "dataset", "navigation guidance"]
innovations: ["首个面向多阶段语言条件规划的变长窗口公开数据集（5,154条指令，30-508.8秒窗口）", "提出trajectory separation与directional compliance指标，揭示语言敏感性与指令遵循的本质差异", "系统分析time-to-turn gap，发现9.8%的Maneuver落在5秒预测窗口内"]
benchmarks: ["doPlan", "nuPlan"]
---

# 论文速读：doPlan-A-Variable-Horizon-Dataset-for-Multi-Stage-Language-C

## 一句话总结
本文提出了 doPlan，首个面向自动驾驶多阶段语言条件规划的公开人类标注数据集，包含 5,154 条跨越 30–508.8 秒变长窗口的乘客指令；对四个主流语言条件驾驶模型的评估表明，当前模型的语言敏感性与真实指令遵循之间存在显著脱节，且多数指令指向的 Maneuver 超出 5 秒预测窗口。

## 研究问题与动机
1. **现有数据集的时序局限**：doScenes 等已有语言条件驾驶数据集仅关注短时（固定 12 秒）局部交互，无法支持乘客意图跨多阶段演化的研究。
2. **语言作为持久任务上下文的需求**：乘客指令往往包含即时动作、延迟动作、事件条件触发行为，需在驾驶进程中持续保留、重新 grounding 并在适当阶段触发，而非一次性扰动。
3. **模型评估的盲区**：当前语言条件驾驶模型以 5 秒轨迹预测为主，但实证表明乘客指令对应的 Maneuver 中位发生在 24.6 秒后，导致评估无法反映真实指令遵循能力。
4. **"语言敏感性≠指令遵循"**：已有工作发现模型可被语言改变轨迹，但未必朝正确方向响应；本文旨在系统揭示这一 gap 及其成因。

## 核心贡献（创新点）
1. **首个面向长期多阶段语言条件规划的开源数据集**：5,154 条指令、169.1 小时累积上下文、50.9 小时独立驾驶时长，远超 doScenes（2,450 条、12 秒固定窗口）。
2. **变长窗口采样策略（30–508.8 秒）**：从 nuPlan 连续轨迹中按 $L \sim \text{Uniform}(30,T)$ 和 $S \sim \text{Uniform}(0,T-L)$ 采样，允许重叠区间独立标注，支持多阶段意图建模。
3. **"出租车测试"自由形式标注协议**：要求标注者以乘客视角写下"你会给司机什么指令来产生视频中的行为"，无需预设模板，捕获延期、条件触发、多步骤意图。
4. **系统性诊断评估与新指标**：提出 trajectory separation $S$（Eq.1）、directional compliance rate、time-to-turn 分组分析（Table VII），揭示语言敏感性与方向遵循之间的本质差异。

## 方法详解
### 数据构建
- **来源**：nuPlan 训练集，含同步相机流与车辆轨迹数据。
- **标注界面**：8 路同步相机 + heading-aligned 路网地图，辅助标注者感知全局路况。
- **采样**：对时长 $T$ 的片段，先采样 $L \sim U(30,T)$ 再采样起始时间 $S \sim U(0,T-L)$，最低 30 秒保证足够上下文。
- **标注协议**：19 名持牌驾驶员独立标注，无模板限制；允许 multi-clause、deferred action、event-conditioned 指令。
- **Referential Labels**：Static / Dynamic / Both / Non-Referential / Ambiguous（Table III）。

### 模型评估协议
- 四个模型（OpenEMMA、AutoVLA、Alpamayo 1.5、Alpamayo 2.0）零样本评估。
- 在每条指令对应的 earliest valid time point 统一传感器历史，仅改变语言输入。
- 四个条件：No Instruction、Correct Instruction、Unrelated Instruction、Counterfactual Instruction（如"右转"→"左转"）。
- ADE 在统一 5 秒、10 个 0.5s 间隔点上计算。

### 关键诊断指标
1. **Trajectory Separation**（Eq.1）：
   $$S = \frac{1}{10} \sum_{k=1}^{10} \| \hat{p}^{\text{orig}}(t_k) - \hat{p}^{\text{cf}}(t_k) \|_2$$
   衡量反事实语言替换导致的轨迹变化幅度，不依赖 ground truth。
2. **Directional Compliance**：仅对位于 5s 窗口内的 matched turn 测量，要求预测终点距离起始点至少 1m 朝向请求侧。
3. **Directional Response**（Eq.2）：
   $$R_i = d_i (\theta_i^{\text{orig}} - \theta_i^{\text{rev}})$$
   按时间到 turn 分组统计模型 heading 偏移方向一致性（Table VII）。
4. **Temporal Grounding 验证**：用 CLIP 构建 video–instruction matcher，以 R@1/R@5/R@10 检验扩展窗口是否携带指令相关信息（Table IV）。

## 实验与结果
### 数据集规模与结构
- 总提交 6,549 条，含指令 5,154 条，无指令 1,395 条。
- 覆盖 988 个 nuPlan 源 clip；median 窗口时长 80.4 秒，64.6% 超过 60 秒，32.3% 超过 120 秒（Fig.2）。
- 表 II：窗口越长，mean words（9.8→27.8）、multi-sentence（24.2%→52.3%）、multi-step（28.8%→62.6%）均递增。

### 模型评估核心发现
**ADE 对比（Table V，4,315 条）**：
| Model | No Instr. | Correct | Unrelated | Counterfactual |
|---|---|---|---|---|
| OpenEMMA | 3.342 | 3.307 | 3.359 | 3.340 |
| AutoVLA | 2.663 | 2.784 | 2.846 | 2.872 |
| Alp. 1.5 (w=1.0) | 1.979 | 2.100 | 2.106 | 2.081 |
| Alp. 2.0 (w=3.0) | 1.451 | 1.834 | 1.816 | 1.858 |

- Correct 指令并未改善 ADE，Alpamayo 2.0 增加 26.4%（1.451→1.834m）。
- Correct / Unrelated / Counterfactual 三者 ADE 接近，模型无法区分语义不匹配指令。

**Guidance Ablation（Table VI）**：
- Alpamayo 2.0 的 trajectory separation $S$ 从 $w=1.0$ 的 0.219m 增至 $w=4.0$ 的 0.679m，语言敏感性随权重单调上升，但方向合规仅增加 1.3pp（48.0%→49.3%）。

**时序 mismatch（核心发现）**：
- 2,161 条含 matched future maneuver 的样本中，首次关联 Maneuver 中位出现在评估点后 **24.6 秒**，仅 **9.8%** 落在 5 秒预测窗口内。
- 仅 4.4% 的方向请求 turn 起始于 5s 窗口内。
- Table VII：AutoVLA 在 turn ≤5s 时 $R=+0.818$，>60s 仍保持 +0.199；其余模型响应微弱且不稳定。

### Video–Instruction Matcher 验证（Table IV）
- R@1 从 5s 的 13.9% 升至 full window 的 23.1%；R@5 从 38.2%→54.2%。
- Duration-only（2.7%）、text-only（1.3%）、frozen CLIP（3.5%）基线显著低于 learned matcher（23.1%），证实扩展窗口含指令相关信息。

## 相关工作脉络
1. **doScenes [14]**：基于 nuScenes 的 1,000 条固定 12s 片段指令，采用相同"出租车测试"原则但时间窗口固定，无法研究多阶段意图；doPlan 将其扩展至连续 nuPlan 上的变长窗口。
2. **HAD [6]**：30 小时驾驶视频的短程语言建议，聚焦 bounded scenes；doPlan 关注指令持久性与时序演化。
3. **Talk2Nav [28]**：远距离导航指令，关注逐步展开的 route instruction；doPlan 聚焦驾驶规划而非导航，且在实时视频流下研究。
4. **Lang2LTL [29]**：将自然语言转换为线性时序逻辑形式化规范；doPlan 保持自由语言形式，强调自然交互而非形式化规约。
5. **OpenEMMA/AutoVLA/Alpamayo 系列 [17][18][19][20]**：当前主流端到端语言-视觉-动作模型，本文在其零样本能力上揭示 5 秒窗口与多阶段意图之间的根本不匹配。
6. **LMDrive [24]/Vega [27]**：语言条件闭环驾驶；本文指出这些工作同样受限于短时预测，未显式建模持久意图。

## 局限性与未来方向
1. **标注验证不完整**：仅移除重复和字段缺失记录，未进行第二轮人工复核，标注质量存在自然变异。
2. **时间戳模糊**：允许回放完整片段后标注，参考信息可能不在评估时间点可见；未标注指令实际发出时间，导致 temporal offset 报告的语义受限。
3. **预测窗口局限**：所有模型均以 5 秒为统一 horizon，无法评估 24.6s 中位延迟 Maneuver 的真实遵循能力。
4. **Counterfactual 机械生成**：仅反转方向词汇，可能产生 scene-infeasible 指令，方向合规评估仅为 diagnostic 而非 complete measure。
5. **未来方向**：持久意图跟踪（pending/active/completed/invalidated 状态机）、事件条件触发执行、时序 grounding、multi-stage 规划评估协议。

## 研究启发与可借鉴点
1. **变长窗口 + 重叠采样**：从连续轨迹按 $Uniform(30,T)$ 采样片段，支持多阶段意图研究，可作为长期规划数据集构建的通用范式。
2. **Trajectory Separation 作为语言敏感性度量**：用反事实替换衡量模型对语言的响应强度，无需 ground truth，适用于任何语言条件模型诊断。
3. **Time-to-Maneuver 分组分析**：将 directional response 按时间到目标 Maneuver 的距离分层（Table VII），揭示模型在 far/near horizon 的不同响应模式。
4. **CLIP-based video–instruction retriever 验证数据集设计**：用 R@K 随窗口时长变化证明长上下文含指令相关信息，为数据集的时序合理性提供定量支撑。
5. **与团队方向结合机会**：持久意图管理（intent lifecycle）、多阶段规划 evaluator、长 horizon 轨迹预测与语言条件的联合建模。

## 关键术语表
**doPlan**：首个面向自动驾驶多阶段语言条件规划的开源人类标注数据集，基于 nuPlan，含变长窗口（30–508.8s）和 5,154 条自由形式乘客指令。

**Taxi Test**：标注协议，要求标注者以乘客视角写下"你会给司机什么指令来产生视频中的行为"，捕获行为导向而非场景描述的指令。

**Trajectory Separation（S）**：用反事实语言替换衡量模型对指令的响应强度，定义为原指令与反向指令预测轨迹 10 个时间点的平均 L2 距离。

**Directional Compliance**：仅对落在 5s 窗口内的 matched turn 测量，要求预测终点在请求方向侧至少 1m。

**Directional Response（R）**：按原始指令与反向指令的预测 heading 差计算，正值表示向请求方向偏移，用于分析 temporal grounding。

**Persistent Task Context**：乘客语言在驾驶进程中持续相关，部分意图当下不可执行但需保留，待条件满足或临近时再影响当前计划。

**Multi-stage Intent**：指令包含多个有序阶段（如"保持车道→经过施工→变道"），各阶段在不同时间点成为 active plan 的一部分。

**Time-to-Turn Gap**：指令指向的 Maneuver 中位发生在评估点后 24.6 秒，远超模型 5 秒预测 horizon，是语言条件规划的核心挑战。

## 可复现要素
- **数据集**：doPlan 已公开，基于 nuPlan 训练集；地址：https://github.com/Mi3-Lab/doPlan
- **代码/权重**：数据集与标注接口公开；模型评估使用 OpenEMMA、AutoVLA、Alpamayo 1.5/2.0 官方权重（zero-shot）
- **关键超参**：Alpamayo 2.0 guidance weight $w \in [1.0, 4.0]$（默认 $w=3.0$）；ADE 在 5s/10 点（0.5s 间隔）计算
- **Segment 采样**：$L \sim U(30,T)$，$S \sim U(0,T-L)$
- **标注者**：19 名持牌驾驶员
- 论文未提及额外训练超参（模型均为 zero-shot 评估）
