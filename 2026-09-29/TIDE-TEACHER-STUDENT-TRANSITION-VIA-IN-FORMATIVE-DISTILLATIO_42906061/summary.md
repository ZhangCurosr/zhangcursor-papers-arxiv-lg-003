---
title: "TIDE-TEACHER-STUDENT-TRANSITION-VIA-IN-FORMATIVE-DISTILLATIO"
source: https://arxiv.org/pdf/2609.35058v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:24:32"
field: "多轮智能体强化学习"
keywords: ["on-policy distillation", "reinforcement learning", "agent training", "teacher-student transition", "credit assignment", "multi-turn interaction"]
innovations: ["基于师生分歧趋势动态调度 OPD→RL 权重的全局 handoff 机制", "相对动作价值与分歧乘性融合的轮次级局部调制设计"]
benchmarks: ["WebShop", "ALFWorld", "SearchQA"]
---

# 论文速读：TIDE-TEACHER-STUDENT-TRANSITION-VIA-IN-FORMATIVE-DISTILLATIO

## 一句话总结
TIDE 针对多轮智能体强化学习中 on-policy 蒸馏（OPD）与 GRPO 奖励优化的协调问题，提出双尺度自适应分配机制：全局基于师生分歧趋势动态调度 OPD→RL 权重转移，局部基于相对动作价值与分歧信号在交互轮次间分配更新优先级，在 WebShop 和 ALFWorld 上超越固定混合与调度基线。

## 研究问题与动机
1. **稀疏轨迹级奖励限制小模型早期探索**：GRPO 仅依赖任务终态奖励，多轮交互中信用分配困难；OPD 提供 token 级蒸馏信号可稳定初期学习，但固定权重混合假设教师引导与奖励优化在整个训练和交互轮次中保持恒定相对角色。
2. **全局尺度问题**：随着训练推进，持续强蒸馏压力会限制学生超越教师能力，OPD 边际效用递减；固定混合比无法适应不同任务上的训练动态。
3. **局部尺度问题**：师生分歧仅标识策略失配，无法区分"有益探索"与"低质量策略漂移"；需要结合动作价值信号判断分歧质量。
4. **任务特异性需求**：不同任务（WebShop、ALFWorld、SearchQA）上分歧下降速率和最佳过渡时机各异，需在线自适应而非人工预设调度。

## 核心贡献（创新点）
1. **形式化为双尺度分配问题**：将混合 OPD-RL 协调分解为跨训练阶段的教师影响调度与跨交互轮次的更新优先级分配，此前工作多聚焦单一尺度。
2. **分歧触发的全局动态调度（Discrepancy-Triggered Handoff）**：通过指数移动平均平滑的批次级分歧相对减少量 $I_k$ 判断过渡时机（$0 < I_k \leq \zeta$ 时单调递增 RL 权重），无需人工指定任务特定切换步，相较 ATOD 的线性退火和固定混合更自适应。
3. **相对动作价值-分歧联合的局部调制（Relative-Value–Disagreement Modulation）**：对每个交互轮次归一化相对动作价值 $\tilde{Q}_t$（来自 GiGPO 的匹配历史组比较）与分歧 $\tilde{D}_t$，OPD 优先级取乘积 $z_t = \tilde{Q}_t \times \tilde{D}_t$，RL 权重取 $\tilde{D}_t$，使高价值高分歧轮次获得强正/负 RL 更新，此前工作缺乏轮次级联合调制。
4. **统一优势函数设计**：$A^{\mathrm{TIDE}}_{t,j} = \beta_k w^{\mathrm{RL}}_t A^{\mathrm{RL}}_{t,j} + \lambda_k w^{\mathrm{OPD}}_t A^{\mathrm{OPD}}_{t,j}$，耦合全局系数 $\beta_k, \lambda_k$ 与局部权重，在 PPO 截断目标下联合优化。
5. **多尺度实验验证**：在 Qwen2.5-1.5B/3B/7B 上系统评估 WebShop、ALFWorld、SearchQA，TIDE 在 7B 达到 WebShop SR 83.2%、ALFWorld Avg 92.1%，均优于 ATOD、SDAR、GiGPO 等基线。

## 方法详解
**问题设定**：给定任务 $x \sim \mathcal{D}$，智能体在交互历史 $h_t = (x, o_1, y_1, a_1, \ldots, o_t)$ 上生成回复 $y_t \sim \pi_\theta(\cdot|h_t)$，解析执行动作 $a_t$，环境返回观测 $o_{t+1}$，形成轮次 $u_t = (y_t, a_t, o_{t+1})$ 和轨迹 $\tau = (x, o_1, u_1, \ldots, u_T)$。

**分歧度量**：对 token $j$ 在轮次 $t$ 的响应，计算师生 log-prob 差 $\delta_{t,j} = \log \pi_T(y_{t,j}|h_t, y_{t,<j}) - \log \pi_{\text{old}}(y_{t,j}|h_t, y_{t,<j})$；聚合得轮次分歧 $D_t = \frac{1}{|y_t|} \sum_j |\delta_{t,j}|$ 和批次分歧 $G_k = \frac{1}{|\mathcal{U}_k|} \sum_{t \in \mathcal{U}_k} D_t$。

**全局自适应（分歧触发过渡）**：
- EMA 平滑：$m_k = \mu m_{k-1} + (1-\mu) G_k$
- 相对减少量（窗口 W）：$I_k = \frac{m_{k-W+1} - m_k}{\max(m_{k-W+1}, \epsilon)}$
- 状态转移：当 $0 < I_k \leq \zeta$ 时 $r_k = \min(r_{k-1} + \eta, 1)$，否则保持
- 系数分配：$\lambda_k = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min})(1 - r_k)$，$\beta_k = \beta_{\min} + (\beta_{\max} - \beta_{\min}) r_k$
- 初始 $\lambda_0 = 1.0, \beta_0 = 0.1$；参数 $\mu=0.9, W=10, \eta=\zeta=0.02$

**局部自适应（相对值-分歧调制）**：
- 相对过程奖励 $Q_t$：沿 GiGPO 框架，基于精确近期观测上下文匹配（history length 2，后缀长度 $k \in \{1,2,3\}$），组内均值中心化折扣回报 $p_{t,k} = G_t - \mu_{K_{t,k}}$，长度加权聚合 $Q_t = \sum_{k} \frac{(k+1)^\nu}{\sum_q (q+1)^\nu} p_{t,k}$（$\nu=1, \gamma=0.95$）
- 归一化：$\tilde{Q}_t = \mathcal{N}_\mathcal{V}(Q_t)$（仅在含判别性匹配组的有效轮次上 min-max），$\tilde{D}_t = \mathcal{N}_\tau(D_t)$（全轮次 min-max）
- 无匹配组时 $\tilde{Q}_t = 1$ 作为分歧-only OPD 回退
- 优先级：$z_t = \tilde{Q}_t \times \tilde{D}_t$（OPD），$w_t^{\mathrm{RL}} = \tilde{D}_t$（RL）
- 优势：$A^{\mathrm{RL}}_{t,j} = Q_t$，$A^{\mathrm{OPD}}_{t,j} = \delta_{t,j}$
- 联合优势：$A^{\mathrm{TIDE}}_{t,j} = \beta_k w^{\mathrm{RL}}_t A^{\mathrm{RL}}_{t,j} + \lambda_k w^{\mathrm{OPD}}_t A^{\mathrm{OPD}}_{t,j}$
- 损失：$\mathcal{I}(\theta) = \mathbb{E}[\sum_{t,j} \min(\rho_{t,j} A^{\mathrm{TIDE}}_{t,j}, \text{clip}(\rho_{t,j}, 1-c, 1+c) A^{\mathrm{TIDE}}_{t,j})]$

**教师构造**：冻结的 GRPO 训练 Qwen2.5-7B-Instruct，与学生共享任务环境、提示、动作解析器、 rollout 配置、优化超参，仅 backbone 规模不同。

## 实验与结果
**数据集与设置**：WebShop（500 任务，30 步交互，温度 1.0 top-p 1.0）、ALFWorld（140 valid-seen 任务，50 步，温度 0.4 top-p 1.0）、SearchQA（7 子集，精确匹配）；学生 Qwen2.5-Instruct 1.5B/3B/7B；16 prompts × 8 rollouts × 160 updates；AdamW lr=1e-6，weight decay 0.01，PPO clip radius c=0.2，entropy 0.001，minibatch 64（WebShop）/256（ALFWorld）。

**主要结果（Table 2）**：
- **1.5B**：TIDE WebShop SR 77.2% ±1.8、Score 89.8% ±1.2，ALFWorld Avg 86.0% ±0.8；超 ATOD（SR 73.2%、Avg 83.6%）和 GiGPO（SR 65.0%、Avg 86.7%）。
- **3B**：TIDE WebShop SR 79.0% ±1.5、Score 90.2% ±0.9，ALFWorld Avg 89.3% ±0.7；超 ATOD（SR 74.1%、Avg 86.4%）。
- **7B**：TIDE WebShop SR 83.2% ±1.3、Score 92.1% ±0.7，ALFWorld Avg 92.1% ±0.7；超 ATOD（SR 79.0%、Avg 89.3%）和 SDAR（WebShop SR 82.8%、ALFWorld Avg 85.9%）。
- **SearchQA（1.5B，Table 4）**：TIDE Avg EM 39.4% ±0.3，超 OPD 36.3% +3.1 点，6/7 子集领先。

**消融（Table 3, 6）**：
- 全局调度：TIDE 超 Success Handoff（+5.8 SR）、Fixed 1:1（+5.2 SR）、Linear（+5.0 SR）、Cosine（+4.0 SR）；ALFWorld unseen 超 Cosine +3.8 点。
- 局部调制：移除 Process Reward 降 SR 至 72.2%，移除 Disagreement 降至 74.4%，Additive Fusion 仅 67.4%，双分支调制必需。
- 参数敏感性：$\zeta, \eta \in \{0.01, 0.02, 0.05\}$ 范围内性能稳定，默认 0.02 最优。

**最强结果**：7B TIDE 在 WebShop SR 83.2% 和 ALFWorld Avg 92.1% 均为 evaluated baselines 最高。

## 相关工作脉络
1. **On-Policy Distillation (OPD)**：Agarwal et al. (2024) 首次将蒸馏应用于 RL 采样轨迹；TIDE 在此基础上进一步在训练阶段和交互轮次双尺度自适应分配，而非固定混合。
2. **Hybrid OPD-RL 方法**：ATOD（Tan et al., 2026）使用预置线性退火调度；RetireOPD（Yu et al., 2026）在分歧停滞且学生达到阈值后退役教师；TIDE 采用在线分歧趋势触发，无需预设时间表或性能阈值。
3. **Credit Assignment for Agents**：GiGPO（Feng et al., 2026）利用重复环境状态构建相对过程优势；HGPO（He et al., 2026）和 GraphGPO（Cheng et al., 2026）分别用层次分组和图结构传播信用；TIDE 直接复用 GiGPO 的相对动作价值作为局部 modulator，并联合分歧信号。
4. **Signal-Calibrated Hybrid Methods**：Scope（Zheng et al., 2026a）校准信号权重；SDAR（Lu et al., 2026）逐 token gating；TIDE 的不同在于将全局调度与轮次级联合调制统一，且使用乘性融合而非加法。
5. **Self-Distillation in RL**：OPSD（Zhao et al., 2026）基于已验证解的自我蒸馏；Seed（Wu et al., 2026）迭代蒸馏；TIDE 使用外部冻结教师而非自我蒸馏，避免学生能力瓶颈。
6. **Schedule-Based Approaches**：Linear/Cosine handoff 等预设调度缺乏任务适应性；TIDE 通过 $I_k$ 事件信号在线响应，实验证明优于所有预设方案。

## 局限性与未来方向
1. **局部模块未全面验证**：SearchQA 实验仅测试全局 handoff，未启用局部调制，其跨域泛化性存疑。
2. **教师依赖**：需要预训练的强教师（此处为 GRPO 训练的 7B 模型），一次性成本较高；对教师可用性高的场景适用，但冷启动受限。
3. **历史匹配严格性**：使用精确字符串匹配构建比较组（history length 2，$k \in \{1,2,3\}$），状态空间大时覆盖度可能下降（Figure 6 显示早期 coverage 较低）。
4. **仅评估三个基准**：WebShop、ALFWorld、SearchQA 均为导航/问答类任务，未验证于数学推理、代码生成或更复杂 embodied 场景。
5. **参数选择**：$\eta, \zeta, \mu, W$ 等超参经敏感性分析较鲁棒，但最优值依赖任务，可探索自动调优。

## 研究启发与可借鉴点
1. **分歧作为调度代理信号**：师生 log-prob 差的 EMA 趋势可有效指示"蒸馏边际效用递减"时机，避免了人工划分训练阶段，可迁移至其他 OPD-RL 混合场景。
2. **相对值-分歧乘性融合**：$z_t = \tilde{Q}_t \times \tilde{D}_t$ 同时利用 outcome-based 信用和 policy mismatch 信息，比单信号或加法融合更精准；可推广至 step-level 奖励分配。
3. **GiGPO 类相对过程价值的即插即用**：TIDE 直接复用 GiGPO 的 $Q_t$ 作为局部 modulator，证明 process reward 估计可与 distillation 自然耦合，为多任务架构提供模块化设计范式。
4. **全局-局部解耦设计**：将调度（跨训练步）与分配（跨交互轮）分离，便于独立 ablation 和参数调优；可借鉴至多阶段训练框架。
5. **精确历史匹配 + 长度加权聚合**：$Q_t$ 的构造（匹配组均值中心化、长度加权 $\nu=1$）平衡了短期与长期信用，细节可在 Appendix A.4 复现。

## 关键术语表
**On-Policy Distillation (OPD)**：在 student 自身采样轨迹上进行蒸馏，利用 teacher 的 token 级 log-prob 作为附加监督信号，减少 distribution mismatch。
**Discrepancy-Triggered Handoff**：基于批次级师生分歧的相对减少量 $I_k$ 动态触发 OPD→RL 权重转移的调度机制，$0 < I_k \leq \zeta$ 时单调递增 RL 系数。
**Relative Action Value ($Q_t$)**：基于精确近期观测上下文匹配的组内均值中心化折扣回报，反映某轮次相对于同历史组替代动作的结果优劣。
**Turn-Level Modulation**：在单条轨迹内，按轮次分配 OPD 和 RL 更新优先级的局部机制，OPD 权重取归一化相对值与分歧的乘积，RL 权重取归一化分歧。
**GiGPO**：Group-in-Group Policy Optimization，通过重复环境状态构建 group 内相对优势，实现细粒度信用分配。
**Importance Ratio ($\rho_{t,j}$)**：当前策略与行为策略下 token 概率之比，用于 PPO 截断目标中的policy gradient估计。
**Exponential Moving Average (EMA)**：对批次分歧 $G_k$ 的平滑处理，$\mu=0.9$，消除单步噪声以稳定 handoff 决策。
**Process Quality Score**：外部 GPT-5.5 评估的交互轮质量（1-3 分），仅用于分析分歧与outcome的关系，不参与训练。

## 可复现要素
- **数据集**：WebShop（500 任务）、ALFWorld（140 valid-seen + 134 valid-unseen）、SearchQA；论文未声明公开链接，基准均为公开数据集。
- **代码/权重**：论文未提供开源链接；教师模型为 GRPO 训练的 Qwen2.5-7B-Instruct checkpoint（任务特定微调），学生为 Qwen2.5-Instruct 基座。
- **关键超参**：lr=1e-6, weight_decay=0.01, clip c=0.2, entropy=0.001, $\mu=0.9$, $W=10$, $\eta=\zeta=0.02$, $\lambda \in [0.1, 1.0]$, $\beta \in [0.1, 1.0]$, $\gamma=0.95$, $\nu=1$, history length=2, 16 prompts × 8 rollouts × 160 updates, minibatch 64（WebShop）/256（ALFWorld）。
- **硬件/训练**：未明确说明；PPO 单 epoch/update, dual-clip bound=3, 无 reference KL penalty。
