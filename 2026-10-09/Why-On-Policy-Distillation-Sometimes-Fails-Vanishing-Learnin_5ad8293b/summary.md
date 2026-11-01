---
title: "Why-On-Policy-Distillation-Sometimes-Fails-Vanishing-Learnin"
source: https://arxiv.org/pdf/2610.11247v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:34:58"
---

# 论文速读：Why-On-Policy-Distillation-Sometimes-Fails-Vanishing-Learnin

## 一句话总结
本文针对大模型 On-Policy Distillation (OPD) 训练中频繁出现的“损失早停”现象，通过连续时间动力学建模与训练日志诊断，揭示了更大规模教师导致学习信号过早坍缩的失败机制，并在师生共享参数空间的假设下证明了邻近教师的局部收敛保证。

## 研究问题与动机
1. **核心问题**：OPD 在代码生成与数学推理任务中并不稳定，当使用参数量更大的教师时，损失曲线常在训练早期 plateau，而师生能力差距仍大幅存在。
2. **现有解释的不足**：已有工作指出教师-学生不兼容、监督信号不可靠、采样 token 局限等问题，但未能阐明为何 OPD 损失对部分教师持续下降，而对另一部分教师提前停滞。
3. **分析动机**：需要建立一个可量化、可诊断的优化动态框架，区分“学习信号弱化”与“rollout 分布偏移”对损失下降的不同贡献，并从理论上界定 OPD 何时能够成功逼近教师。

## 核心贡献（创新点）
1. **识别典型失败模式**：系统刻画了 OPD 在学习率较低时的早期损失 plateau 现象，指出其普遍出现在跨参数规模的教师蒸馏中，而 self-RL 教师则能驱动损失接近零。
2. **诊断过早学习信号坍缩**：提出在训练日志中监控梯度范数与剩余损失的比值，发现更大规模教师场景下该代理指标在损失仍 >60% 时骤降，定义为 premature learning-signal collapse。
3. **局部恢复的理论保证**：在共享参数化与正则性条件下，证明了当教师参数 $\theta_T$ 距初始学生 $\theta_0$ 足够近时，OPD 连续流可在有限时间内将损失压制到由初始 Fisher 矩阵低曲率方向决定的残差上界。
4. **参数/表示稳定性量化**：通过相对参数变化与线性 CKA 度量，发现 200 步 OPD 仅引起 0.025%–0.098% 的参数位移与各层 CKA > 0.98 的表示相似性，提出“有限表征适应可能制约教师匹配”的可检验假设。

## 方法详解
- **连续时间动力学建模**：在小学习率极限下，将 OPD 的随机梯度迭代近似为 ODE $\dot{\theta}(t) = -g(\theta(t))$，其中 $g(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[\nabla_\theta \widehat{\mathcal{L}}(\theta;\tau)]$ 为去耦 KL 梯度。
- **损失动态精确分解**：导出损失衰减率 $\frac{d}{dt}\mathcal{L}(\theta(t)) = -2\gamma(\theta(t))\mathcal{L}(\theta(t))$，其中瞬时衰减系数分解为 $\gamma = \mu(1+\alpha)$。$\mu = \|g\|^2/(2\mathcal{L})$ 衡量学习信号强度，$\alpha = \langle h,g\rangle/\|g\|^2$ 衡量 occupancy 梯度 $h(\theta) = \mathbb{E}[\widehat{\mathcal{L}}\nabla_\theta\log\pi_\theta]$ 对更新方向的辅助或抵消作用。
- **训练诊断实现**：$\hat{\mu}_m$ 直接从预 clipping 的 minibatch 梯度范数与损失日志计算；$\hat{\alpha}_m$ 在每 10 步 checkpoint 处通过交叉拟合估计（将每 prompt 的 8 个 rollout 拆为两折，互换 baseline 消除同折采样噪声）。
- **局部收敛定理（Theorem 5.2）**：若 $\rho=\|\theta_T-\theta_0\|$ 足够小，OPD 流在 $T_{\mathrm{loc}}=\Omega(1/\rho)$ 时间内保持在教师 $2\rho$ 邻域内；损失上界为 $\mathcal{L}(\theta(t)) \le e^{-\delta t}\mathcal{L}(\theta_0) + E_\Delta(\delta) + O(\rho^3+\rho^4/\delta)$，其中 $E_\Delta(\delta)=\frac{1}{2}\sum_{0<\lambda_i\le\delta}\lambda_i(u_i^\top\Delta)^2$ 为由初始 Fisher 矩阵 $H_0$ 低特征值方向主导的残差能量。
- **表示分析**：采用相对参数距离 $d_{\mathrm{par}}=\|\theta_{\mathrm{end}}-\theta_{\mathrm{start}}\|/\|\theta_{\mathrm{start}}\|$ 与线性 CKA 评估更新前后参数空间与隐藏表示的稳定性。

## 实验与结果
- **数据集与基准**：代码训练用 Eurus-2-RL-Data，评测 HumanEval+、MBPP+、Live-CodeBench；数学训练用 DAPO-Math-17K，评测 AIME 2024/2025、AMC 2023；指令跟随评测用 Multi-IF。
- **主体结果（200 步 OPD）**：Self-RL 教师组（Qwen3-4B-RL-Code、JustRL-1.5B 等）平均损失下降 96.2%，准确率缺口恢复 70.7%–77.4%；更大规模教师组平均损失仅下降 25.1%，缺口恢复仅 13.0%–47.5%，且剩余损失仍显著。
- **消融验证**：SFT 预处理后接续 OPD 对更大规模教师无效（最大损失下降仅 37.1% 与 23.0%）；切换 top-16 与 full-logits、单 rollout 与 8 rollout 均不改变教师依赖性行为。
- **诊断指标**：图 2 显示更大规模教师场景下 $\hat{\mu}_m$ 在步骤 15–30 内下降约一个数量级，而相对剩余损失仍 >0.6；Table 3 显示 occupancy 因子 $1+\hat{\alpha}_m$ 在中后期多为正，未出现系统性抵消。
- **参数/表示度量**：Table 4 报告所有 OPD 学生的 $d
