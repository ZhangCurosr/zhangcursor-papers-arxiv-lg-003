---
title: "ORPG-Reconciling-Multiple-Reward-Objectives-through-Objectiv"
source: https://arxiv.org/pdf/2609.34985v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:52:10"
field: "多目标强化学习与语言模型对齐"
keywords: ["multi-reward RL", "policy gradient", "gradient reconciliation", "RLHF", "multi-objective optimization", "mathematical reasoning"]
innovations: ["提出 ORPG 框架，在独立截断策略目标基础上通过兼容协调与优先级冲突解决重结合完整策略梯度", "兼容更新被证明为固定半径球面上方向折衷的唯一解并保持原始梯度求和范数", "统一兼容规则适配等权重与主次优先级两种任务设定"]
benchmarks: ["Alpaca", "HH-RLHF", "PKU-SafeRLHF", "AIME-24", "AMC-22-23", "MATH", "Minerva-Math", "OlympiadBench"]
---

# 论文速读：ORPG-Reconciling-Multiple-Reward-Objectives-through-Objective-wise-Policy-Gradients

## 一句话总结
论文提出 ORPG（Objective-wise Reconciled Policy Gradient），通过为每个奖励目标保留独立截断策略目标并协调其完整梯度来解决多奖励策略优化问题；在助手机器人 help-harmless 对齐与数学推理正确性-成本优化任务上均取得最优结果，兼容协调是主要贡献来源。

## 研究问题与动机
1. 多奖励策略优化需要在学习信号层面协调不同目标，同时尊重目标间的预定关系（等权重或优先级）。
2. 现有基于 GRPO 的方法多在优势估计阶段聚合目标（如 MO-GRPO、GDPO、GD²PO），或显式设计共享更新方向（如 GAPO、PAMA），但忽略了对兼容梯度幅度关系的精细协调。
3. 纯冲突投影（如 PCGrad）在梯度兼容时不做任何调整，即便其中一个梯度主导直接求和结果也无法被纠正。
4. 实际任务中目标关系多样：helpfulness–safety 通常等权重，而 correctness–cost 存在主次优先级，需统一框架同时支持两种设定。

## 核心贡献（创新点）
1. 提出 ORPG 框架，保留每个奖励的独立截断策略目标并通过兼容协调与任务优先级冲突解决重结合完整策略梯度，与在优势层聚合的方法形成本质区别。
2. 兼容更新被证明是固定半径球面上的唯一方向折衷解，同时保持原始梯度求和的范数，并界定混合后贡献比位于原始比与参考比之间。
3. 同一兼容规则可适配不同优先级设定：对称目标使用 PCGrad 式双端投影，主次目标使用单侧投影保留主梯度，扩展了梯度重结合的应用范围。
4. 在两个截然不同的任务（帮助-安全对齐、正确性优先的数学推理）上系统验证，表明梯度重结合对等权重和目标优先级场景均有效。

## 方法详解
- **独立目标构造**：对每个奖励 $r_i$ 保留 PPO/GRPO 风格截断策略目标 $J_i(\theta)$，通过 separate clipping 保持目标身份。
- **兼容协调（$c \geq 0$）**：定义梯度范数 $n_i$、单位方向 $u_i$、cosine 相似度 $c$ 和直接求和 $s$；构造部分归一化参考 $v_q = n_1^q u_1 + n_2^q u_2$ 并将其缩放至原求和范数；混合 $z = (1-\alpha)s + \alpha b_q$（$\alpha = \lambda c$），最终归一化保持 $\|s\|_2$；$q=0.5$ 使幅度比变为平方根，介于等幅与原始比之间。
- **冲突解决**：对称优先级采用 PCGrad 双投影；主次优先级时保留 $g_p$ 并对次要梯度做投影使其不与主梯度相反。
- **优化实现**：每步计算 Gram 标量 $n_1^2, n_2^2, d$，以 stop-gradient 方式构建局部代理目标，最后对总梯度做 norm clipping 并用 AdamW 更新。

## 实验与结果
- **数据集与基线**：Helpfulness–safety 使用 Alpaca/HH-RLHF/PKU-SafeRLHF，以 Artessay Qwen2.5-7B-SafeRLHF reward/cost 评分；数学推理使用 DeepScaleR preview prompts，评测 AIME-24、AMC-22-23、MATH、Minerva-Math、OlympiadBench。基线包括 GRPO、GDPO、GD²PO。
- **帮助-安全结果**：ORPG 在三个数据集的 Useful 和 Harmless 得分上均最高；平均 Useful 超过最强外部基线 GDPO 0.415，平均 Harmless 超过 GD²PO 0.446。
- **数学推理结果**：ORPG 在 8192-token 预算下平均准确率达 66.7%，比 Base 提升 1.14pp 且响应长度减少 611 token（19.3%）；在所有数据集上均超过全部外部训练基线。
- **超体积（HV）**：ORPG 在三个预算（2048/4096/8192）上的平均 HV 达 0.523，超越 Base（0.493）和最强外部基线 GD²PO（0.513）。
- **组件分析**：兼容协调贡献更大（移除后 Useful 降 0.378、Harmless 降 0.193），冲突解决提供额外增益；训练过程中多数步骤梯度兼容（后期冲突率约 6.5%）。

## 相关工作脉络
1. **PCGrad**（Yu et al., 2020）：对冲突梯度做对称投影，但不处理兼容梯度的幅度协调，ORPG 在兼容时引入连续调节。
2. **MO-GRPO / GDPO**：在优势层面分别归一化再聚合，ORPG 保留独立策略目标并在梯度层面重结合，区分信号层与参数更新层的设计。
3. **GD²PO**：过滤冲突优势并重加权 prompt group，属于学习信号层干预，ORPG 在完整策略梯度上操作，更直接地影响参数更新。
4. **GAPO / PAMA**：将共享更新方向显式设计为优化变量，ORPG 以几何规则（球面折衷/投影）确定方向，计算更高效（仅需 Gram 标量）。
5. **CAGrad / Aligned-MTL**：控制最坏局部改进或构建 alignment 变换，实验显示 ORPG 在同等 objective-wise 接口下取得更高 Useful/Harmless 得分。
6. **Length-aware reasoning**：以长度目标/惩罚控制生成成本，ORPG 在数学任务中以 correctness 为主目标、length 为次目标实现优先级梯度协调。

## 局限性与未来方向
- 当前实现在两目标场景下验证充分，多目标（$m>2$）的兼容参考构造与混合权重选择尚需进一步研究。
- 兼容协调参数 $q, \lambda$ 对超体积与全预算准确度有不同偏好，实际部署需按具体预算敏感目标调参。
- 训练过程中兼容阶段占主导，冲突仅在后段少量出现，说明当前奖励构造下冲突并非主要瓶颈，但更复杂多目标场景的冲突结构有待探索。
- 未讨论与动态奖励权重方法（如 Learning to Optimize Multi-Objective Alignment）的互补性或融合路径。

## 研究启发与可借鉴点
1. **球面方向折衷形式化**：将梯度协调建模为固定范数球上的优化问题，给出唯一解与范数保持性质，可作为多目标策略更新的可解释理论框架。
2. **部分归一化参考构造**：通过指数 $q$ 在等幅与原始幅度间连续插值，避免硬截断或二元选择，对多任务学习中的梯度幅度失衡具有迁移价值。
3. **统一兼容/冲突处理接口**：同一兼容规则配合不同优先级投影即可适配等权重与主次目标设定，简化工程实现并增强通用性。
4. **训练动力学可观测指标**：用 step-average gradient cosine、conflict rate、projection rate、advantage RMS 等指标刻画多目标优化过程，有助于诊断组件贡献与失败模式。
5. **与团队方向结合机会**：若团队涉及多奖励 RLHF、长度控制推理或工具使用对齐，可将 ORPG 的梯度重结合模块插入现有 PPO/GRPO pipeline，验证其对联合奖励提升的增益。

## 关键术语表
- **ORPG**：Objective-wise Reconciled Policy Gradient，通过独立截断策略目标与梯度重结合实现多奖励策略优化的方法。
- **Clipped policy objective**：PPO/GRPO 风格的截断策略目标，通过 clip 机制限制更新步幅以保障训练稳定性。
- **Compatible contribution coordination**：当梯度 cos 相似度非负时，通过部分归一化参考与原始求和插值协调贡献幅度。
- **Spherical directional compromise**：兼容更新被证明为固定半径球面上平衡到原始求和与参考向量距离的唯一解。
- **Conflict projection**：对负 cos 相似度的冲突梯度执行对称或主次投影，消除反向分量。
- **Hypervolume (HV)**：在多预算点（准确率、效率）下计算的并集矩形面积，用于综合评估精度-成本权衡。
- **Group-relative advantage**：在采样组内相对于组统计量的优势估计，GRPO 常用形式，无需学习 value function。
- **Stop-gradient in reconciliation**：将重结合系数视为常数参与局部代理目标梯度计算，避免对混合权重的二阶依赖。

## 可复现要素
- **数据集**：Alpaca（训练/校准/评测拆分）、HH-RLHF、PKU-SafeRLHF、DeepScaleR preview prompts、AIME-24、AMC-22-23、MATH、Minerva-Math、OlympiadBench（公开数据集，论文未提及额外许可）。
- **代码开源**：是，代码已公开于 https://github.com/euReka025/ORPG。
- **基座模型**：Qwen3-4B-Instruct-2507（需遵循原模型许可）。
- **关键超参**：PPO clip $\epsilon_- = 0.2, \epsilon_+ = 0.28$；KL MSE 系数 $\beta = 0.0005$（数学任务）；学习率 helpfulness–safety $2 \times 10^{-6}$、数学 $10^{-6}$；兼容参数默认 $q=0.5, \lambda=0.25$；长度阈值 $\tau = 4000$；梯度范数 clip = 1.0；训练步数 100；batch/minibatch/rollout group 依任务不同。
- **训练硬件**：8×H200 GPU，FSDP，bfloat16。
