---
title: "When-Is-Tracking-Affordable-A-Movement-Budget-Feasibility-Th"
source: https://arxiv.org/pdf/2611.02812v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:44:33"
---

# 论文速读：When-Is-Tracking-Affordable-A-Movement-Budget-Feasibility-Th

## 一句话总结
本文在在线凸优化框架下研究硬性移动预算约束时的动态 regret，证明当预算低于比较器路径长度的常数倍时会触发可行性屏障（导致线性 regret），并提出预算感知步长的投影 OGD 算法，证明当预算超过 $D + P_T$ 时可实现次线性 regret，首次明确刻画了移动预算对动态追踪的根本性限制。

## 研究问题与动机
1. **物理干预成本被忽略**：传统动态 regret 分析假设学习者可无成本自由移动，但时序控制系统（控制器重配置、服务状态切换、投资组合再平衡）的每次移动均消耗有限资源，需研究硬性预算下的可行性边界。
2. **路径长度并非预算约束下的 hardness measure**：现有理论以比较器路径长度 $P_T$ 刻画难度，但随机游走类比较器可具有大 $P_T$ 却空间集中，不会迫使学习者长距离移动；真正构成最坏情况的是比较器的“空间跨度（range）”。
3. **同构环境下的可行性断裂**：对于固定 $D$ 且 $P_T = o(T)$ 的场景，无预算 minimax 速率为次线性（可学），但施加预算后同一实例族可能变为不可学，需定量界定该相变阈值。

## 核心贡献（创新点）
1. **证明移动预算可行性屏障（下界）**：构造确定性全范围方波对手，证明对任意预算可行算法，当 $S \leq c P_T$（$c$ 为通用常数）时动态 regret 必为 $\Omega(GDT)$，表明线性 regret 由预算本身而非环境强波动引起。
2. **提出 Budget-Controlled OGD 并给出上界**：设计投影在线梯度下降配合步长 $\eta = \min\!\left(\sqrt{D(D+2P_T)/(G^2 T)},\; S/(GT)\right)$，证明当 $S = \omega(D + P_T)$ 且 $S = o(DT)$ 时算法满足预算且达到 $o(GDT)$。
3. **识别临界移动尺度 $D + P_T$**：将可行性相变的尺度确定为决策集直径与比较器路径长度之和，首次在同一理论框架下分离“预测可行性”与“干预可行性”。
4. **明确阈值上方的开放问题**：指出当前仅能提供含 $1/S$ 项的上界，未证明其 minimax 紧致性，为后续速率研究划定清晰靶点。

## 方法详解
- **问题设定**：决策集 $\mathcal{K} \subset \mathbb{R}^d$ 直径为 $D$，每轮 $t$ 选取 $x_t \in \mathcal{K}$ 后观测 G-Lipschitz 凸损失 $f_t$。动态 regret $R_d = \sum_{t=1}^T [f_t(x_t) - f_t(u_t)]$，总移动距离 $\mathrm{SC} = \sum_{t=2}^T \|x_t - x_{t-1}\| \leq S$，比较器路径长度 $P_T = \sum_{t=2}^T \|u_t - u_{t-1}\|$。
- **下界证明（方波对手）**：取 $\mathcal{K} = [0, D]$，损失 $f_t(x) = G|x - u_t|$。令 $n = \lfloor P_T / D \rfloor$，比较器在 $0$ 与 $D$ 间交替形成 $n+1$ 个长块（块长度 $\tau \approx T/(n+1)$）。将连续块配对未来，若第 $i$ 对内学习者总移动量为 $M_i$，则其跟踪误差下界为 $\tau(D - M_i)$。对 $\bar{N} = \lfloor(n+1)/2\rfloor$ 对求和并利用 $\sum M_i \leq S$，推得 $R_d \geq \frac{GDT}{8}\!\left(1 - \frac{8S}{P_T}\right)$，当 $S \leq c P_T$ 时退化为 $\Omega(GDT)$。
- **上界算法（Budget-Controlled OGD）**：标准投影梯度更新 $x_{t+1} = \Pi_\mathcal{K}(x_t - \eta \nabla f_t(x_t))$，步长取两项的最小值：第一项 $\sqrt{D(D+2P_T)/(G^2 T)}$ 为无预算最优步长；第二项 $S/(GT)$ 保证每轮移动 $\eta G$ 在 $T$ 轮内累计不超过 $S$。代入 Zinkevich (2003) 经典界 $R_d \leq \frac{D(D+2P_T)}{2\eta} + \frac{\eta G^2 T}{2}$ 即得定理 2 的上界。
- **关键理论区分**：强调在预算约束下，决定追踪难度的核心量是“比较器迫使学习者跨越的空间距离”，而非单纯的路径长度累加。

## 实验与结果
- **数据集**：合成方波序列、随机游走比较器、五组公开真实时序数据（S&P 500、EUR/USD、crude oil via USO、Bitcoin、New York City temperature）。
- **主要结果**：
  - **相变定位**：图 2(A) 扫过 $S/P_T$ 显示归一化 regret 沿 $R_d/(GDT) = \frac{1}{2}(1 - S/P_T)$ 衰减，实证相变点紧贴 $S = P_T$，理论下界常数保守但结构吻合。
  - **预算屏障隔离**：图 2(B) 固定 $P_T = \sqrt{T} = o(T)$ 放大 $T$，预算饥饿学习者（$S = P_T/4$）normalized regret 保持常数（线性 regret），
