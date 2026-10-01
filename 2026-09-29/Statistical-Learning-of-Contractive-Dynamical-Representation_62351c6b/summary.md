---
title: "Statistical-Learning-of-Contractive-Dynamical-Representation"
source: https://arxiv.org/pdf/2609.35758v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:22:36"
---

# 论文速读：Statistical-Learning-of-Contractive-Dynamical-Representation

## 一句话总结
提出一种基于硬期望最大化（hard-EM）与卡尔曼平滑器的统计学习框架，用于学习具有统一收缩性质的扰动动力潜表征；该表征可嵌入 Kalman–Bucy 复合自适应控制器，在扰动与受控系统动态耦合且代理信号稀疏/噪声大的场景下，实现具备预测能力且指数收敛至有界邻域的跟踪控制。

## 研究问题与动机
- 现有最后一层自适应扰动抑制方法多将扰动视为外生变量（常假设固定衰减率或 OU 过程），缺乏对动态耦合扰动（如液体晃动、摆锤、非结构化地面力）的时序预测能力。
- 当扰动源与 plant 存在动力学耦合时，将其建模为具有独立演化状态的增广潜变量更符合物理机制，也更契合经典扰动 Accommodating 控制（DAC）的思路。
- 实际工程中扰动代理信号 $y(t)$ 往往由数值微分反演得到，噪声大且采样稀疏（如仅 1 Hz）；若控制器无法在观测间隔内预测扰动，跟踪性能将显著退化。
- 传统 DAC 依赖线性时不变（LTI）子空间辨识，难以刻画非线性、工况依赖的耦合扰动机制，亟需数据驱动且具有稳定性保证的替代方案。

## 核心贡献（创新点）
- **硬-EM 训练框架**：提出交替执行卡尔曼平滑（E 步）与参数/噪声协方差更新（M 步）的硬期望最大化流程，实现含潜变量动力模型的可微统计学习。*与纯梯度端到端训练的本质区别在于 E 步可精确求解潜轨迹 MAP，避免了隐式微分带来的数值不稳定与局部最优。*
- **构造性收缩神经架构**：将动力学矩阵参数化为 $A=S+K$，其中对称部分 $S$ 由固定收缩率 $\beta$、半负定项 $LL^\top$ 构成，斜对称部分 $K$ 不影响收缩性，从而无需损失惩罚即严格满足 $\frac{1}{2}(A+A^\top)\preceq -\beta I$。*与通用 MLP 动态模型的本质区别在于收缩性由结构保证，直接导出对未知初值的指数不敏感性与协方差有界性。*
- **复合自适应控制器与理论保证**：将 learned 表征嵌入扩展 Kalman–Bucy 滤波方程，并叠加跟踪误差衍生项 $P\Phi^\top s$，证明闭环系统在扰动表示误差有界条件下指数收敛至有界球。*与固定衰减基线的本质区别在于学习到的时变 $A(\phi,u)$ 提供了跨步长预测能力，而非简单指数遗忘。*

## 方法详解
- **扰动动力表示**：潜状态 $\mathbf{a}\in\mathbb{R}^{d_a}$ 演化 $\dot{\mathbf{a}}=A(\phi,u)\mathbf{a}+b(\phi,u)$，扰动估计 $\hat{d}=\Phi(\phi)\mathbf{a}+\psi(\phi)$；$\phi$ 为因果特征（通常为状态），$A,b,\Phi,\psi$ 均由 GELU MLP 参数化，$\Phi$ 经 tanh 有界。
- **收缩参数化**：令 $S(\phi,u)=-(\beta+\mu(\phi,u))I-L(\phi,u)L(\phi,u)^\top\preceq -\beta I$，$K(\phi,u)=\frac{1}{2}(M-M^\top)$ 为斜对称矩阵；保证 $\frac{1}{2}(A+A^\top)=S\preceq -\beta I$，满足 Proposition 1 的指数初值不敏感性。
- **硬-EM 训练（Algorithm 1）**：
  - **E 步**：固定 $(\theta,R_d,Q_d)$，对每个数据窗口通过 RTS 卡尔曼平滑器精确推断潜轨迹 $\mathbf{a}_{0:L-1}$（等价于最小化高斯负对数似然加初值先验 $\lambda_a\|\mathbf{a}_0\|^2$）。
  - **M 步**：根据滚动残差更新观测噪声 $R_d$ 与过程噪声 $Q_d$（支持稀疏/EMA 累积），随后固定潜轨迹用 AdamW 更新 $\theta$。
- **复合控制器**：采用扩展 Kalman–Bucy 方程 $\dot{\hat{\mathbf{a}}}=A\hat{\mathbf{a}}+b+P\Phi^\top R^{-1}(y-\Phi\hat{\mathbf{a}}-\psi)+P\Phi^\top s$ 与协方差 Riccati 迭代，结合反馈律 $u=M\dot{v}_r+Cv_r+g-Ks-\Phi\hat{\mathbf{a}}-\psi$（二阶机械系统）；Lemma 2 给出 $P(t)$ 的上下界，支撑 Theorem 3/4 的指数收敛证明。

## 实验与结果
- **数据集与设置**：① 硬件平台 GVR-Bot（滑移履带车，携带约 30% 水罐与摆锤，低速低附着力胶带模拟滑移），采集约 11 min/水位（60%、0%），特征采样 18 Hz，控制 50 Hz；② 仿真耦合 Duffing 振子（仅可观 $x_1,\dot{x}_1$，扰动代理 $y$ 以 1 Hz 提供并注入不同强度高斯噪声 $\sigma$）。训练/验证/测试按 70:15:15 划分。
- **基线**：模型基于 PD 控制、FixedDecay 消融（$A=-\lambda I,b=\psi=0$）、N4SIDv-DAC（LTI 子空间辨识 DAC）。
- **主要结果**：
  - 车辆跟踪 RMS 误差：角速率 0.407 rad/s（最优）、速度 0.149 m/s、位置 0.07
