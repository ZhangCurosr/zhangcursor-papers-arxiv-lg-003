---
title: "Self-Consuming-Generative-Models-with-Co-Evolving-Human-Pref"
source: https://arxiv.org/pdf/2610.09415v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-08 10:07:14"
---

# 论文速读：Self-Consuming-Generative-Models-with-Co-Evolving-Human-Pref

## 一句话总结
本文建立了生成模型迭代重训练过程中“模型分布-用户偏好”共演化动力学框架，严格证明纯合成数据训练必然导致赢家锁定（多样性崩溃），并在 $\eta(L_p+2L_w)<1$ 条件下给出参考数据混合的唯一全局收敛充分条件，同时提出基于线性规划的低成本参考分布设计方法。

## 研究问题与动机
- 生成模型在自消耗循环（self-consuming loop）中大量使用自身合成数据，会引发分布偏移与性能衰退，但现有工作多假设评估偏好静态不变。
- 实际应用中用户/校准者偏好会随模型输出动态漂移，与模型更新形成耦合反馈回路，缺乏对该双向动力学的长期行为刻画。
- 现有合成数据缓解策略多为经验性混合或离线重采样，缺少可证明稳定性的混合比设计与参考分布优化理论。
- 亟需统一框架回答：共演化系统何时收敛？何时发生赢家锁定？如何以最低成本设计参考数据以保真属性多样性？

## 核心贡献（创新点）
1. **建立共演化动力学理论框架**：将模型分布 $p_t$ 与偏好向量 $w_t$ 建模为耦合映射系统，首次刻画自消耗循环中的双向反馈机制。（与仅分析单分布漂移或假设静态偏好的工作本质不同。）
2. **严格证明赢家锁定（Winner Lock-in）现象**：理论刻画 $\eta=1$ 时系统存在 $n$ 个单点平衡态，任意初始优势实例通过正反馈垄断分布，驱动类别多样性崩溃。（填补了合成数据训练长程不稳定性的理论空白。）
3. **给出参考数据混合的唯一全局收敛充分条件**：证明当 $\eta(L_p+2L_w)<1$ 时耦合映射为严格收缩映射，所有轨迹几何收敛至与初始化无关的唯一平衡点。（突破以往仅依赖数值经验或局部稳定性的分析范式。）
4. **提出可优化的参考数据/混合比设计机制**：将 $p_{\mathrm{ref}}$ 与 $\eta$ 纳入双层优化，推导出一维搜索+线性规划的高效求解路径，实现成本最小化下的多样性保真。（区别于静态固定参考分布的工程启发式方案。）

## 方法详解
- **耦合动力学系统**：定义离散有限实例空间 $\mathcal{X}=\{x_1,\dots,x_n\}$，模型输出分布 $p_t \in \mathcal{P}(\mathcal{X})$，偏好向量 $w_t \in \mathcal{W}=\{w:\|w\|\leq 1\}$。奖励函数 $r_{w_t}(x)=\langle w_t, \varphi(x)\rangle$，其中 $\varphi$ 为单位范数特征映射。
- **K-way Luce 策展算子**：$H^K_{p_t,w_t}(x_k)=\frac{e^{\tau r_{w_t}(x_k)}}{\sum_j e^{\tau r_{w_t}(x_j)}}$（$K\geq 2$），$K\to\infty$ 时退化为指数倾斜形式，用于从当前分布中高奖励区域概率采样。
- **模型更新规则**：$p_{t+1}(x)=(1-\eta)p_{\mathrm{ref}}(x)+\eta p_t(x)H^K_{p_t,w_t}(x)$，$\eta\in[0,1]$ 控制合成与参考数据的混合权重。
- **偏好漂移规则**：$w_{t+1}=\mathrm{Proj}_{\mathcal{W}}((1-\beta)w_t+\beta\bar{\varphi}(p_{t+1}))$，$\beta\in(0,1]$ 控制偏好追踪速率，$\bar{\varphi}$ 为当前分布的特征期望。
- **关键性质**：严格单调性（奖励差距决定策展概率排序）与 Odds 收缩性（奖励差距 $\geq r_\Delta$ 时低/高奖励策展概率比存在上界 $\lambda_K(r_\Delta)\in(0,1)$，且 $\lambda_K\to e^{-\tau r_\Delta}$）。
- **Lipschitz 常数**：$L_p=2e^{2\tau}(2+e^{2\tau})$，$L_w=2\tau e^{4\tau}$，分别刻画模型更新与偏好更新对状态扰动的敏感程度。
- **稳定性定理（Thm 3.9）**：当 $\eta(L_p+2L_w)<1$ 时，耦合更新在度量 $d_\xi((p,w),(q,v))=d_{TV}(p,q)+\xi\|w-v\|$ 下为严格收缩映射；存在唯一全局吸引平衡 $(p_K^\star,w_K^\star)$，任意轨迹几何收敛，$\xi$ 取值窗口为 $(\frac{\eta L_w}{\beta(1-2\eta L_w)},\frac{1-\eta L_p}{2\beta\eta L_p})$。
- **参考数据设计（Bilevel 优化）**：目标 $\min_{p_{\mathrm{ref}},\eta}(1-\eta)N\sum c_i p_{\mathrm{ref},i}$，约束为平衡态下属性期望 $\langle v_\ell,\bar{\varphi}(p_K^\star)\rangle\geq\theta_\ell$。Prop 4.1 给出充分条件，将约束松弛为关于 $p_{\mathrm{ref}}$ 的线性
