---
title: "Why-On-Policy-Distillation-Sometimes-Fails-Vanishing-Learnin"
source: https://arxiv.org/pdf/2610.11247v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:34:41"
field: "语言模型后训练与蒸馏"
keywords: ["on-policy distillation", "knowledge distillation", "post-training", "learning dynamics", "gradient analysis", "language model"]
innovations: ["提出OPD学习信号崩溃的诊断框架与连续时间动力学分析", "证明共享参数空间中教师邻近性的局部恢复定理", "系统量化不同教师类型（self-RL vs 大规模）的OPD损失轨迹差异"]
benchmarks: ["HumanEval+", "MBPP+", "Live-CodeBench", "AIME 2024", "AIME 2025", "AMC 2023"]
---

# 论文速读：Why-On-Policy-Distillation-Sometimes-Fails-Vanishing-Learnin

## 一句话总结
本文系统研究了强化学习后训练中的**在线策略蒸馏（OPD）**在匹配大规模教师时为何会遭遇**早期损失平台**（premature learning-signal collapse），通过将OPD建模为连续时间动力系统，提出了学习信号崩溃与局部恢复理论保证，并揭示了参数与表征变化的局限性。

## 研究问题与动机
1. **核心问题**：OPD训练中大模型教师（larger-scale teachers）在数学推理和代码生成任务上出现早期损失平台，此时师生间仍存在大量未消除的分布差异，但损失不再持续下降。
2. **自我RL教师（self-RL teachers）的成功对比**：由初始学生经过进一步RL训练得到的自我RL教师在OPD中支持持续损失下降（平均最终损失减少96.2%），而大规模教师的平均损失减少仅25.1%。
3. **已有解释的不充分性**：前人研究提出的师生不兼容、教师指导不可靠、采样token监督的局限等解释，未能阐明**为何某些教师能持续降低损失而另一些教师提前停止**。
4. **理论与实践缺口**：缺乏对OPD损失动态的定量分析框架，以及关于"何时OPD能成功收敛"的理论保证。

## 核心贡献（创新点）
1. **跨模型与任务的OPD失败模式识别**：在代码生成与数学推理任务中，系统观察到大规模教师下OPD在损失仍高时提前停滞，这是本文首次在同一实验框架下量化对比不同教师类型的损失轨迹。
2. **学习信号崩溃的诊断框架**：通过训练日志诊断，发现梯度相关的学习信号代理（$\widehat{\mu}_m$）在损失仍大量剩余时大幅下降，定义并量化了"过早学习信号崩溃"现象。
3. **连续时间动力系统分析**：将OPD建模为小学习率极限下的连续时间ODE（$\dot{\theta}(t) = -g(\theta(t))$），导出损失衰减率分解公式 $\gamma = \mu(1+\alpha)$，分离学习信号与分布变化的贡献。
4. **局部恢复定理**：证明当教师在学生共享参数空间内与学生足够接近时，OPD流存在局部收敛保证（Theorem 5.2），解释了self-RL教师实验成功的理论机制。
5. **参数与表征变化度量**：测量了OPD训练前后学生的相对参数变化（0.025%–0.098%）和线性CKA相似度（>0.98），提出有限表征适应可能是学习信号崩溃的假设。

## 方法详解
**OPD数学形式**：OPD最小化学生与教师在学生采样响应前缀上的反向KL散度：
$$\mathcal{L}(\theta) = \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta} D_{\mathrm{KL}}(\pi_\theta(\cdot|s) \| \pi_T(\cdot|s))$$
实践中使用 top-K 近似（K=16）。

**梯度分解**：损失梯度拆分为两项：
$$\nabla\mathcal{L}(\theta) = \underbrace{\mathbb{E}_{\tau\sim\pi_\theta}\nabla_\theta\widehat{\mathcal{L}}(\theta;\tau)}_{g(\theta)} + \underbrace{\mathbb{E}_{\tau\sim\pi_\theta}[\widehat{\mathcal{L}}(\theta;\tau)\nabla_\theta\log\pi_\theta(y|x)]}_{h(\theta)}$$
其中 $g$ 为解耦梯度（detach gradient），$h$ 为占据梯度（occupancy gradient）。

**损失动态定理（Proposition 5.1）**：在正则条件下，当 $\mathcal{L}>0$ 时：
$$\frac{d}{dt}\mathcal{L}(\theta(t)) = -2\gamma(\theta(t))\mathcal{L}(\theta(t)), \quad \gamma = \mu(1+\alpha)$$
其中 $\mu = \|g\|^2/(2\mathcal{L})$ 衡量学习信号强度，$\alpha = \langle h, g\rangle/\|g\|^2$ 衡量分布变化方向。损失平台对应小的 $\overline{\gamma}$。

**局部恢复定理（Theorem 5.2）**：设 $\theta_T = \theta_0 + \Delta$，$\rho = \|\Delta\|$，在 bounded rollout、bounded Lipschitz score、Lipschitz Hessian 等正则条件下：
$$\mathcal{L}(\theta(t)) \leq e^{-\delta t}\mathcal{L}(\theta_0) + E_\Delta(\delta) + O(\rho^3 + \rho^4/\delta)$$
其中 $E_\Delta(\delta) = \frac{1}{2}\sum_{0<\lambda_i\leq\delta}\lambda_i(u_i^\top\Delta)^2$ 为低曲率方向的残留损失。

**学习信号代理估计**：$\widehat{\mu}_m = \widehat{G}_m^2/(2\widehat{\mathcal{L}}_m)$，其中 $\widehat{G}_m$ 为预裁剪 mini-batch 梯度范数。

**$\alpha$ 的交叉拟合估计**：在checkpoint处将rollout折叠，使用 $\widehat{g}_m^2 = \langle g_m^{(0)}, g_m^{(1)}\rangle$ 和 $\widehat{X}_m = \frac{1}{2}(\langle h_m^{(0)}, g_m^{(1)}\rangle + \langle h_m^{(1)}, g_m^{(0)}\rangle)$ 计算 $\widehat{\alpha}_m = \widehat{X}_m/\widehat{g}_m^2$。

## 实验与结果
**数据集**：代码任务使用 **Eurus-2-RL-Data**；数学任务使用 **DAPO-Math-17K**。

**评估基准**：代码 — HumanEval+, MBPP+, Live-CodeBench；数学 — AIME 2024, AIME 2025, AMC 2023。

**7组学生-教师对**（Table 1）：
- Code: Qwen3-4B → Qwen3-4B-RL-Code† / Qwen3-14B
- Math: Qwen3-1.7B → Qwen3-4B-RL-Math / Qwen3-4B
- Math: R1-Distill-1.5B → JustRL-1.5B† / Skywork-7B / R1-Distill-7B

**关键结果**（Table 2, 200步训练）：
| 教师类型 | 平均最大损失减少 | 平均最终损失减少 | 平均Gap恢复 |
|---|---|---|---|
| Self-RL教师 | 97.9% | 96.2% | — |
| 大规模教师 | 41.6% | 25.1% | 13.0%–47.5% |

- 最优自我RL教师（Qwen3-4B-RL-Code）：损失减少99.5%，准确率增益+7.70pp，Gap恢复77.4%
- 最差大规模教师（Qwen3-4B）：损失减少仅6.24%（top-16）/ 11.60%（full logits）
- SFT初始化后继续OPD未能解决损失平台问题（R1-Distill-7B: 最大减少37.1%；Qwen3-4B: 23.0%）
- 词汇表支持（top-16 vs full logits）和rollout数量（1 vs 8）的变化不改变教师依赖的行为模式

**学习信号崩溃证据**（Figure 2, Table E.3）：
- 大规模教师下，$\widehat{\mu}_m$ 在约10倍下降后损失几乎不变，剩余损失仍在60%以上
- Self-RL教师下，$\widehat{\mu}_m$ 逐渐下降的同时损失持续减少至接近0

**参数与表征变化**（Table 4）：
- OPD后学生相对参数变化：0.025%–0.098%
- 各层最小CKA：>0.985（成功与停滞运行之间无显著差异）
- Self-RL教师相对初始学生的距离：0.030%（Qwen3-4B-RL-Code）至0.50%（JustRL-1.5B）

## 相关工作脉络
1. **OPD及其失败模式研究**：Agarwal et al. (2024) 提出GKD；Gu et al. (2024) 提出MiniLLM；Li et al. (2026) 关注师生思维模式对齐和失败梯度；Fu et al. (2026a) 分析token级监督不均衡。本文定位：提供系统性定量分析和理论解释。
2. **后训练动力学与几何**：Malladi et al. (2023) 的kernel dynamics视角；Shenfeld et al. (2026) 报告RL后训练与基座模型的KL邻近性；Shen et al. (2026) 发现OPD更新快速收敛到低维子空间。本文与之差异：聚焦教师-学生距离如何塑造损失动态。
3. **策略梯度的连续时间分析**：Borkar & Meyn (2000) 的ODE方法；Razin et al. (2025) 分析RLHF中的vanishing gradients。本文沿用连续时间理想化但将其应用于OPD而非纯RL设置。
4. **教师能力与蒸馏效率**：Cho & Hariharan (2019)、Mirzadeh et al. (2020) 发现学生可能从过大教师学到更少；Stanton et al. (2021) 质疑蒸馏的有效性。本文延伸：不仅大教师不一定更好，且同规模教师的来源（self-RL vs 独立训练）是关键。
5. **在线模仿学习**：Ross et al. (2011) 的基础工作；Zhang et al. (2026) 分析在线IL在LLM后训练中的作用。本文与之区分：关注分布变化项（$h$）对损失动态的具体影响。

## 局限性与未来方向
1. **理论假设的理想化**：局部恢复定理基于ODE理想化（gradient flow），而实验使用AdamW；分析假设全词表KL，但训练使用top-16近似。
2. **代理指标的噪声**：学习信号代理 $\widehat{\mu}_m$ 使用预裁剪mini-batch梯度，包含采样方差，不能直接等同于总梯度范数。
3. **表征适应假设未验证**：有限参数/表征变化的观察仅构成假设，需受控干预来检验其对学习信号崩溃的因果作用。
4. **跨超参的泛化未验证**：学习率、训练预算、随机种子的影响需更多实验确认。
5. **不同参数空间的教师**：定理不覆盖与学生在参数维度上不同的教师（如Qwen3-14B vs Qwen3-4B），这是大规模教师失败的核心场景但理论未解释。

## 研究启发与可借鉴点
1. **学习信号分解框架可迁移**：将梯度分解为 $\mu$（学习信号强度）和 $\alpha$（分布变化方向）的两因子分析框架，可用于诊断其他KL散度最小化或策略蒸馏场景的优化停滞问题。
2. **连续时间OPD建模的启示**：小学习率下的ODE极限分析为理解离散训练动态提供了清晰的解析工具，可与本团队的后训练优化分析结合。
3. **参数邻近性作为教师选择的判据**：Theorem 5.2表明教师与学生在参数空间中的距离是关键，这提示在设计蒸馏流水线时，应优先考虑与学生在同参数空间内的教师（如self-RL教师），或通过权重插值构造"近端"教师。
4. **诊断指标的工程实践价值**：$\widehat{\mu}_m$ 和 $1+\widehat{\alpha}_m$ 的计算仅需训练日志，无需额外开销，可嵌入OPD训练管道用于实时监控早停/教师切换决策。
5. **表示相似性度量的借鉴**：使用线性CKA结合相对参数变化来评估后训练中模型改动的幅度，这一诊断范式可迁移到RLHF、SFT等其他后训练方法的评估中。

## 关键术语表
**On-Policy Distillation (OPD)**：学生在自身采样的响应前缀上训练，以匹配教师的next-token分布，减少train-inference分布不匹配的知识蒸馏方法。
**Premature Learning-Signal Collapse**：OPD训练中梯度相关学习信号代理在损失仍有大量剩余时大幅衰减，导致损失提前进入平台期的现象。
**Detached Gradient (g)**：固定前缀分布下计算的KL梯度，反映当前学生访问的前缀上匹配教师的即时改进方向。
**Occupancy Gradient (h)**：因参数更新导致学生采样分布改变而产生的梯度项，反映分布漂移对损失的影响。
**Learning Signal Proxy ($\widehat{\mu}_m$)**：由训练日志计算的 $\widehat{G}_m^2/(2\widehat{\mathcal{L}}_m)$，作为理论量 $\mu = \|g\|^2/(2\mathcal{L})$ 的经验代理。
**Self-RL Teacher**：通过对初始学生进一步进行RL训练（如GRPO）获得的学生同参数空间教师模型。
**Local Recovery Guarantee**：在共享参数空间和正则条件下，当教师与学生参数距离足够小时，OPD流保持局部收敛保证的定理。
**Linear CKA (Centered Kernel Alignment)**：衡量两个模型在相同输入下隐藏表征几何相似性的无偏度量，值域[0,1]，1表示完全相似。

## 可复现要素
- **代码**：已开源，https://github.com/leizhao7/opd-learning-signals
- **数据集**：Eurus-2-RL-Data（代码）和 DAPO-Math-17K（数学），论文未明确声明是否公开，但均引用了公开发布来源
- **模型**：Qwen3系列（4B/14B/1.7B）、R1-Distill-1.5B/7B、JustRL-1.5B、Skywork-7B，均为公开发布的模型
- **关键超参**：学习率 $10^{-6}$（常数），AdamW ($\beta_1=0.9, \beta_2=0.999$)，weight decay=0.01，gradient clipping norm=1，top-K=16，每prompt 1 rollout，64 prompts/update，200步训练
- **训练框架**：veRL
- **评估协议**：代码avg@4（k=4），数学avg@8（k=8），response cap分别为16,384和31,744 tokens
