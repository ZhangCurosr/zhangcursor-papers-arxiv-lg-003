---
title: "Two-Clocks-Why-Larger-Rollout-Batches-Learn-Slower-but-Fail"
source: https://arxiv.org/pdf/2610.18853v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:46:10"
field: "大语言模型强化学习训练动力学"
keywords: ["RLVR", "rollout batch scaling", "square-root rule", "GRPO", "critical batch", "policy gradient", "training stability", "ratio clipping"]
innovations: ["提出双时钟框架统一解释多步RLVR中进展变慢与故障时钟不变的并行现象", "证明平方根规则下临界批随学习率增大而缩小并远低于梯度噪声尺度", "推导实用的分批ramp调度策略并实验验证其接近最优固定批次性能"]
benchmarks: ["DAPO-Math-17k", "DeepScaleR-Preview", "Qwen2.5-Math-1.5B", "Qwen2.5-Math-7B"]
---

# 论文速读：Two-Clocks-Why-Larger-Rollout-Batches-Learn-Slower-but-Fail

## 一句话总结
本文建立了"双时钟"理论框架，解释了在带可验证奖励的强化学习（RLVR）中，为何较大 rollout 批次在以平方根规则调整学习率时虽然保持每样本漂移不变，却因局部损失曲率导致样本效率下降，同时崩溃点却恰好锁定在样本数尺度上不变；并由此导出一个简单的分批调度策略。

## 研究问题与动机
- **实践矛盾**：RLVR（如 GRPO/DAPO）因生成主导成本而普遍扩大 rollout 批次，并用平方根规则缩放学习率，理论上报导在单步优化下近似保持样本效率，但实际多步训练中出现了三个未解释的并行现象。
- **三大现象**：① 较大批次每样本进展变慢（学习曲线偏离）；② 无 clip 时崩溃样本数大致恒定，与优化步数无关；③ clip 对大批次的稳定性收益最大。
- **既有工作局限**：批大小不变性、临界批行为（学习率重调）、固定批次下 clip 效应三者被分别研究，无人统一解释这三个现象在同一条平方根规则下为何同时涌现。
- **关键缺口**：多步场景下"漂移不变 ≠ 进展不变"这一机制未被形式化，也缺乏可检验的定量预测。

## 核心贡献（创新点）
1. **双时钟框架**：以局部二次损失模型推导进展时钟（受曲率影响、存在学习率依赖的临界批），并引入"漂移阈值假设"推导故障时钟（沿平方根规则样本数近似不变）。与以往只关注其一的工作本质不同，本文统一解释了二者为何分道扬镳。
2. **临界批与 McCandlish 等的根本区别**：在平方根规则下临界批 $B^* \propto 1/\kappa$ 随学习率增大而缩小，远低于梯度噪声尺度，不同于重调学习率情形下临界批等于噪声尺度的结论。
3. **clip 的作用重新定位**：证明 clip 主要买的是稳定性而非持续性速度；它对大批次收益最大（裁剪掉的 token 携带了超过一半的梯度范数），但无法消除进展差距。
4. **实用分批调度策略**：从双时钟推导出"先在临界批以下训练以保证样本效率，达到预测失效点之前切换到大批次"的 ramp 策略，实验证明其接近最优固定批次的性能。
5. **跨模型/数据集鲁棒性验证**：进度差距在第二个数据集、通用 checkpoint 和 7B 模型上重现，故障时钟在不同尺度下均成立。

## 方法详解
- **设定**：使用 GRPO + DAPO 目标（ratio clip $\varepsilon_{low}=0.2, \varepsilon_{high}=0.28$，负 advantage 的 dual clip $c=10$），AdamW 优化器。rollout 批次 $B_r$ 拆为 $\mu$ 个 mini-batch，每个 mini-batch 做一步优化，步长 $b=B_r/\mu$。
- **平方根规则**：$\eta = \kappa\sqrt{b}$，即学习率随优化批大小的平方根缩放。
- **假设 A1（噪声主导归一化）**：优化批远小于梯度噪声尺度，$\nu \approx \sigma/\sqrt{b}$。实测 $b$ 比 $B_{simple}$ 低 60–2000 倍。
- **假设 A2（局部二次损失）**：一步内损失变化 $\langle\bar{g},\Delta\theta\rangle + \frac{1}{2}\Delta\theta^\top H\Delta\theta$，mini-batch 视作独立新数据。
- **漂移每样本（核心量）**：$\mathbb{E}[\Delta\theta] = -\frac{\kappa b}{\sigma}\bar{g}$，每样本漂移 $d = \frac{\kappa\|\bar{g}\|}{\sigma}$ 与批次无关——这是平方根规则真正保持的量。
- **进展时钟（Proposition 1）**：每样本期望损失下降 $r(b) = r_0 - C_{curv}\cdot b$，其中 $C_{curv}=\frac{\kappa^2\bar{g}^\top H\bar{g}}{2\sigma^2}$。批次越大，曲率损失越多，进展越慢。
- **临界批公式**：$N_\tau(B_r) = N_\tau^{min}(1 + B_r/B^*)$，$B^* = \mu\left(\frac{2\sigma\|\bar{g}\|^2}{\kappa\bar{g}^\top H\bar{g}} - B_{noise}\right)$，随学习率增大而缩小。
- **假设 A3（漂移阈值）**：当累积漂移 $\|\sum_j \mathbb{E}\Delta\theta_j\|$ 超过阈值 $D$ 时崩溃。
- **故障时钟（Proposition 2）**：$N_{fail} = C\frac{\sqrt{b}}{\eta}$，沿平方根规则简化为 $N_{fail}=C/\kappa$，与批次无关。噪声驱动机制会给出 $N_{fail}\propto b/\eta^2$，实验排除此可能。
- **Ratio clip 的作用位置**：log-ratio 散布 $\log\rho_t \approx \sum_j\langle\Delta\theta_j, s_t\rangle$，大批次散布更快（比例 $\sqrt{B_r^L/B_r^2}$），clip 因此对大批次作用更强，限制每轮 rollout 累积的漂移量。

## 实验与结果
- **设置**：Qwen2.5-Math-1.5B，GRPO + DAPO 目标，AdamW，DAPO-Math-17k 数据集，G=8，temperature=1，响应上限 3072 tokens。关闭 KL 惩罚、熵奖励、动态采样、梯度范数 clip（单独测试）。预算 98,304 responses，每 8,192 在 512 题上评估（mean@4）。16× 批次范围：512/1536/4096/8192。
- **进展变慢（P1）**：$\mu=1$ 时 clip 无法作用，大批次仍需多 20–45% 样本达到同准确率高；$\mu=8$ 保守区间差距类似；活跃区间差距扩大。样本成本指数 $\alpha$ 从 0.15（$\eta_S=10^{-6}$）升至 0.44（$\eta_S=10^{-5}$）；对应临界批从 ~9k 降至 ~2k（远低于噪声尺度）。
- **故障时钟（P2）**：无 clip 时，4096 批次在 56–80 步后崩溃（~29–41k samples），512 批次在 648 步后崩溃（~31–46k samples）——步数差 8 倍，样本数几乎相同。21 个无 clip 运行的 $N_{fail} \propto b^\beta\eta^{-\gamma}$ 拟合：$\beta-\gamma/2 = -0.08$（含删失 0.11），90% 区间含 0；$\gamma=0.64$–0.72 排除随机游走预测（$\gamma=2$）；漂移模型通过似然比检验（$p\approx 0.29$），噪声模型和批盲模型均拒绝（$p<10^{-4}$）。18 个失败运行中 14 个落在 1.5 倍误差带内。
- **Clip 买稳定性（P3）**：有 clip 时小批次延迟 2 倍失败，大批次延迟 4 倍；clip 去掉的大批次 token 仅 0.5%，但携带 >50% 梯度范数。放宽 clip 加速前期学习但后期衰减/崩溃。
- **鲁棒性**：进度差距在 DeepScaleR-Preview、Qwen2.5-1.5B-Instruct、Qwen2.5-Math-7B 上均重现。7B 模型大批次在 rollouts 上差 6×，但样本数仅差 1.3×。
- **调度策略**：从 512 批次在 24.6k samples 后切换至 4096/8192，切换点 25k–74k 范围均表现相近，75% 样本用大批次生成，接近最优固定批次性能。

## 相关工作脉络
1. **McCandlish et al. (2018)**：临界批在重调学习率下等于噪声尺度且与 lr 无关；本文指出平方根规则下临界批由 $1/\kappa$ 项主导，远小于噪声尺度，二者机制不同。
2. **Hilton et al. (2022)**：使 PPO 批次大小不变性；本文扩展到多步场景，证明该不变性在多步下失效。
3. **Li et al. (2026)**：证明 GRPO 单步下平方根规则近似保持学习曲线对齐；本文揭示多步（$\mu=8$）下这一结论不成立。
4. **Khatri et al. (2026) / Yang et al. (2026b)**：发现大批次可达更高准确率但学得慢；本文用双时钟统一解释此"快慢分离"现象。
5. **Suk & Duan (2025) / Song et al. (2026)**：研究崩溃的步长阈值或异步 GRPO 中 $1/\eta$ 崩溃时间；本文加入批次维度，证明沿平方根规则崩溃在样本数上不变。
6. **Qi et al. (2026) / Yang et al. (2026a)**：将 clip 视为学习瓶颈；本文证明 clip 主要买稳定性，对大批次尤为关键，重新定位 clip 的正面作用。

## 局限性与未来方向
- **局部理论局限**：理论基于单步局部二次近似，曲率项的直接测量在当前规模下计算昂贵，通过可检验预测间接验证。
- **覆盖范围**：虽跨两个数据集、两种 checkpoint、7B 模型验证，但仍限于单一模型族和数学推理任务域。
- **种子变异性**：故障点存在种子间固有波动，结论基于 21 个运行的聚合缩放而非单次运行。
- **效率度量单一**：仅用 responses 衡量样本效率，token 级、计算成本和 wall-clock 效率还取决于生成长度、硬件和系统实现。
- **未来方向**：直接将曲率项纳入在线监测、扩展到其他 RLVR 任务（代码、通用推理）、探索更优的动态切换策略。

## 研究启发与可借鉴点
1. **双时钟分析范式**：将"进展"和"崩溃"作为两个独立的时间尺度分别建模，可迁移到任何涉及批量缩放和多步更新的 RL 系统分析中。
2. **平方根规则的重新审视**：在自适应优化器的多步 RLVR 中，应警惕"每样本漂移不变 ≠ 每样本进展不变"，在选批次时需主动估计临界批而非依赖单步直觉。
3. **Clip 的稳定性价值量化**：本文通过 replay 测量 clipped token 梯度占比（小 batch 的 0.5% token 贡献 >50% 梯度范数），这种分析方式可用于诊断其他 trust-region 方法的实际影响。
4. **Ramp 调度策略**：从小批起步、提前切换到大批的 ramp 方法简单且有效，可结合在线临界批估计实现自适应调度。
5. **故障律拟合框架**：用 $N_{fail}\propto b^\beta\eta^{-\gamma}$ 形式统一拟合多种失败机制并通过似然比检验区分漂移/噪声/批盲模型，是一套可复用的系统分析方法。

## 关键术语表
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：利用可自动验证正确性的奖励信号（如数学答案是否正确）进行策略优化的强化学习方法，无需人类偏好标注。
- **Rollout 批次（Rollout batch）**：一次策略更新前由当前策略生成的响应总数（$B_r=P\times G$），而非单次优化的 mini-batch 大小。
- **平方根规则（Square-root rule）**：Adam 类优化器下学习率按批次大小的平方根缩放（$\eta\propto\sqrt{b}$），旨在保持跨批次的训练轨迹可比性。
- **漂移每样本（Drift per sample）**：每消耗一个样本策略参数积累的持久位移 $d=\kappa\|\bar{g}\|/\sigma$，是平方根规则唯一保持的不变量。
- **临界批（Critical batch）**：超过此批次后样本效率开始显著下降的分界点，本文定义 $B^*\propto\mu/\kappa$，随学习率增大而缩小。
- **双时钟（Two-clock framework）**：进展时钟（到达目标精度所需样本数）和故障时钟（崩溃发生时的样本数）沿平方根规则呈现完全不同的缩放行为。
- **Ratio clip**：DAPO/GRPO 中对 policy ratio $\rho_t$ 施加的截断（$\varepsilon_{low}=0.2, \varepsilon_{high}=0.28$），超出范围的 token 梯度贡献被置零。
- **Failure law（故障律）**：$N_{fail}=C\sqrt{b}/\eta$，描述崩溃发生所需样本数与优化批和学率的定量关系，沿平方根规则退化为常数。

## 可复现要素
- **数据集**：DAPO-Math-17k（主实验），DeepScaleR-Preview（鲁棒性验证）；论文未说明公开状态，DAPO-Math 来自 Yu et al. (2025)。
- **代码**：论文声明代码已开源，仓库名 `two-clocks-rlvr`（见 abstract 末尾）。
- **权重**：Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.5-1.5B-Instruct 均为开源权重。
- **关键超参**：$\mu=8$（主实验），$B_r\in\{512,1536,4096,8192\}$，$\eta_S\in\{10^{-6}, 5\times10^{-6}, 10^{-5}\}$，$\varepsilon_{low}=0.2, \varepsilon_{high}=0.28, c=10$，G=8，temperature=1，max response 3072 tokens，AdamW $\beta=(0.9, 0.999)$，weight decay 0.1，bf16 训练。
- **硬件**：1.5B 用 4× NVIDIA RTX A6000，7B 用 4× H100，总计约 2000 GPU-hours。
- **框架**：verl 0.9.1，vLLM 0.19，FSDP 4-GPU。
