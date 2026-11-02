---
title: "Weighted-Adjoint-Matching-Finetuning-Flow-Matching-Models-wi"
source: https://arxiv.org/pdf/2610.20035v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:04:37"
---

# 论文速读：Weighted-Adjoint-Matching-Finetuning-Flow-Matching-Models-wi

## 一句话总结
本文提出 **Weighted Adjoint Matching (WAM)**，通过在随机最优控制问题的二次控制代价中引入时变权重 $w(t)$，使预训练的扩散/流匹配模型在奖励微调时早期严格保持粗粒度特征以维护模式多样性，晚期才放开以优化细节提升奖励，从而显著改善传统 Adjoint Matching 的奖励-模式失衡权衡。

## 研究问题与动机
- **核心问题**：给定可微奖励函数 $r(x)$，如何在微调预训练 Flow Matching / Diffusion 模型时，在提升期望奖励的同时避免分布坍缩（mode collapse）或模式权重失衡？
- **直接优化的缺陷**：最大化 $\mathbb{E}_p[r(X)] - \frac{1}{a}\text{KL}(p\|p_{\text{target}})$ 等价于采样自 tilted distribution $p_a(x) \propto p_{\text{target}}(x)e^{ar(x)}$，该操作会在样本空间中任意搬运质量，无视特征生成的时空层次，必然导致方差或位置不同的模式权重失衡。
- **Adjoint Matching 的不足**：Domingo-Enrich 等 [35] 将微调建模为二次控制代价的 SOC 问题，但在整个时间轴 $t\in[0,1]$ 上均匀惩罚控制量；由于扩散/流模型具有“先粗后细”的生成特性，早期（粗特征未定型时）施加控制会轻易将轨迹拉向平均奖励更高的模式，破坏原始分布的结构平衡。
- **理论依据**：基于 mode speciation 与 critical windows 理论（Li & Chen [31] 等），不同空间尺度的特征在生成的特定时间窗口内被确定；低频/结构特征早定，高频/细节特征晚定，因此可通过时间加权控制代价实现分层微调。

## 核心贡献（创新点）
1. **提出 Weighted Adjoint Matching 算法**：在 SOC 控制代价中引入时变权重 $w(t)$，仅修改原 Adjoint Matching 损失的一处，计算开销不变。与已有工作的本质区别在于：首次将时间-空间尺度对应关系显式嵌入控制理论框架，从机制上阻断早期模式坍缩。
2. **建立路径 KL 正则化的理论等价性**：证明加权 SOC 问题等价于在生成路径所有子区间 $[0,t]$ 上施加随时间变化的 KL 散度正则（Proposition 1）。与已有工作的本质区别在于：将控制代价的时间权重解读为动态正则强度，为微调提供了新的变分视角与可解释性。
3. **在可解析与真实数据集上验证 tradeoff 优势**：在 Gaussian Mixture 上给出理论曲线对比，在 MNIST 上以类别数量比 $n_{\min}/n_{\max}$ 量化多样性。与已有工作的本质区别在于：通过“理论可解析+实证可度量”的双层验证，清晰揭示了均匀惩罚的失效边界与加权策略的帕累托改进。

## 方法详解
- **加权 SOC 问题**：
  $$\min_{u\in\mathcal{U}} \mathbb{E}\left[\int_0^1 \frac{w(t)}{2}\|u(X_t^u,t)\|^2 dt - a r(X_1^u)\right]$$
  其中 $w(t)\geq 0$ 为有界变差函数，归一化 $w(1)=1$；早期 $w(t)$ 大表示强约束（贴近 base model），晚期 $w(t)$ 小表示弱约束（允许奖励驱动）。
- **路径 KL 正则等价形式**（Proposition 1）：
  $$\max_u \mathbb{E}_{X_0}\left[a\mathbb{E}[r(X_1^u)|X_0] - w(1)\text{KL}(p^u\|p^{\text{base}}) + \int_0^1 K_t(X_0) dw(t)\right]$$
  其中 $K_t(X_0)=\text{KL}(p^u(X_{[0,t]}|X_0)\|p^{\text{base}}(X_{[0,t]}|X_0))$。当 $w$ 单调递减时，积分项对子路径偏离施加额外负反馈正则。
- **伴随状态与损失函数**：最优控制满足 $u^*(x,t)=-\frac{\gamma(t)}{w(t)}\nabla_x J(u^*,x,t)$。构造 lean adjoint $\tilde{a}$ 满足反向 ODE：
  $$\frac{d}{dt}\tilde{a}(t,\mathbf{X}) = -\nabla_x b(X_t,t)^\top \tilde{a}(t,\mathbf{X}), \quad \tilde{a}(1,\mathbf{X}) = -a\nabla r(X_1)$$
  得到 WAM 损失：
  $$\mathcal{L}_{\text{WAM}}(u) = \mathbb{E}_{\mathbf{X}\sim p^{\bar{u}}}\left[\frac{1}{2}\int_0^1 \left\|u(X_t,t) + \frac{\gamma(t)\tilde{a}(t,\mathbf{X})}{w(t)}\right\|^2 dt\right]$$
  其中 $\bar{u}=\text{stopgrad}(u)$。
- **数值稳定化设计**：经典 schedule 下 $\gamma(t)=\sqrt{2(1-t)/t}$ 在 $t\to 0,1$ 发散。引入辅助权重 $\lambda(t)>0$ 不改变临界点，取 $\lambda(t)=2(1-t)t$ 可抵消奇异性，得到最终可训练损失 $\mathcal{L}_{\text{WAM}}^\lambda(\theta)$。
- **参数化与控制更新**：微调网络 $v_\theta^{\text{finetune}}$ 参数化控制 $u_\theta(x,t)=\frac{2}{\gamma(t)}(v_\theta^{\text{finetune}}-v^{\text{base}})$，初始 $\theta=\theta_0$（$u=0$）。离散化 SDE 与伴随 ODE 后，对 $M$ 条轨迹蒙特卡洛估计损失，Adam 更新。

## 实验与结果
- **Gaussian Mixture**：双分量混合 $p_{\text{target}}=\frac{1}{2}\mathcal{N}(-\mu,\sigma_1^2 I)+\frac{1}{2}\mathcal{N}(\mu,\sigma_2^2 I)$，奖励 $r(x)=-\|x_2\|^2/2$。Adjoint Matching 严格沿理论 tilted 曲线下降，奖励提升伴随模式权重失衡；WAM 使用指数权重 $w_{\text{exp}}(t)=e^{\alpha(1-t)}$ 与分段常数 $w_{\text{steps}}(t)$ 均在更高奖励下保持近原始权重，$t_s\in\{0.5,0.65,0.8\}$ 时几乎不改变模式比例。
- **MNIST**：9.0M 参数 UNet，奖励为图像平均亮度 $r(x)=\frac{1}{d}\sum x_i$。多样性以 $n_{\min}/n_{\max}$ 度量（基线约 0.80）。在平均奖励 0.176 时，Adjoint Matching 生成 18.0% 的“0”和仅 3.4% 的“1”；WAM（$t_s=0.45$, $a=227$）以奖励 0.180 达到 10.5% 的“0”和 9.1% 的“1”。黑暗奖励实验（Appendix E.4）结论一致。
- **最强结果**：MNIST 上 $t_s=0.45$ 的 WAM 在接近原始数据类别分布比例的同时，实现了优于 Adjoint Matching 的奖励-多样性帕累托前沿；Gaussian Mixture 四模态实验（Appendix E.2.3）进一步验证了 $t_s$ 越大越能保留水平/垂直粗结构、仅在模态内部微调的层级行为。

## 相关工作脉络
- **Adjoint Matching [35]**：本文的直接基线，将微调映射为二次控制代价 SOC，但未区分时间步的特征尺度，均匀惩罚导致跨模式质量非物理转移。
- **ImageReward/ReFL [24] 与 Direct Fine-Tuning [28]**：通过反向传播奖励梯度经随机或最后 $K$ 步，侧重计算效率与梯度信噪比，缺乏对微调后分布的显式刻画与多样性保障。
- **RL-based 微调 [18, 27, 32]**：将去噪链视为 MDP，但 Uehara 等与 Domingo-Enrich 等指出带逐步 KL 正则的 RL 通常无法精确采样 tilted distribution。
- **多样性感知 RL 微调 [25, 45]**：Barceló 等与 Liu 等同样注意到早期步对多样性的决定性作用并提出更强正则，但基于 GRPO 标量奖励框架，不保证分布收敛到目标 tilting。
- **Deterministic Adjoint Matching [41]**：Guo 等用确定性 ODE 替代采样器以提升效率，但因此丧失分布刻画能力，且仅
