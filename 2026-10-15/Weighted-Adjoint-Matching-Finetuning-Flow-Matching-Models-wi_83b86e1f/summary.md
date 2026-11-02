---
title: "Weighted-Adjoint-Matching-Finetuning-Flow-Matching-Models-wi"
source: https://arxiv.org/pdf/2610.20035v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:48:28"
field: "生成模型微调与对齐"
keywords: ["flow matching", "diffusion model fine-tuning", "stochastic optimal control", "weighted adjoint matching", "mode collapse", "KL regularization", "coarse-to-fine generation"]
innovations: ["在 Adjoint Matching 的 SOC 问题中引入时间依赖控制代价权重 w(t)，利用扩散模型粗到细的生成特性在早期时间步保持多样性", "证明加权 SOC 等价于在所有子路径 [0,t] 上施加时变 KL 正则的目标", "设计数值稳定的 loss 形式，通过 λ(t)=2(1-t)t 消除边界处 γ(t) 的爆炸"]
benchmarks: ["Gaussian Mixture (two-component)", "MNIST"]
---

# 论文速读：Weighted-Adjoint-Matching-Finetuning-Flow-Matching-Models-wi

## 一句话总结
本文提出了 **Weighted Adjoint Matching (WAM)**，通过在 Adjoint Matching 的随机最优控制问题中引入**时间依赖的控制代价权重** $w(t)$，解决扩散/流匹配模型微调时分布失衡（mode collapse）的问题。核心思想是利用生成过程"先粗后细"的特性：早期时间步赋予高权重以保留粗糙语义特征的多样性，晚期则降低权重允许通过细节调整提升奖励。

## 研究问题与动机
1. **微调扩散/流匹配模型的奖励优化问题**：给定一个可微奖励函数 $r(x)$，希望在保持生成分布尽可能接近原始分布 $p_{\text{target}}$ 的同时最大化奖励期望 $\mathbb{E}_p[r(X)]$，等价于采样自倾斜分布 $p_a(x) \propto p_{\text{target}}(x)e^{ar(x)}$。
2. **传统 KL 正则化导致模式失衡**：标准方法通过惩罚 $\text{KL}(p\|p_{\text{target}})$ 防止发散，但倾斜操作会在样本空间中无关地转移概率质量——即使两个模式在空间上相距很远，只要奖励偏好就会发生质量转移，导致模式权重失衡（mode imbalance）。
3. **Adjoint Matching 的时间均匀惩罚不足**：Domingo-Enrich 等提出的 Adjoint Matching 将微调建模为随机最优控制问题，但对控制代价 $ \|u\|^2 $ 在整个时间轴上均匀惩罚，无法区分不同时间步对粗/细特征生成的不同影响。
4. **扩散/流匹配模型的"粗到细"生成特性**：自然数据在扩散/流匹配模型中先生成低频大尺度特征（如图像轮廓、数字类别），后生成高频小尺度细节（如亮度纹理），这一特性源于自然图像傅里叶频谱随频率衰减的事实。

## 核心贡献（创新点）
1. **提出 Weighted Adjoint Matching 算法**：在 Adjoint Matching 的 SOC 问题中引入时间依赖权重 $w(t)$，对早期时间步施加更高的控制代价，从而保留粗特征多样性；与已有方法本质区别在于利用时间-空间分辨率的对应关系，而非单纯调节全局 KL 强度。
2. **建立加权 SOC 与路径 KL 正则化的等价性**：证明加权控制代价可重写为在所有子路径 $[0,t]$ 上的 KL 散度的积分形式，揭示了方法的理论语义——相当于对生成轨迹的每一阶段施加时变 KL 正则。
3. **系统性实验验证在 Gaussian Mixture 和 MNIST 上的优势**：在可控的 Gaussian Mixture 实验中可以精确计算倾斜分布的理论曲线作为参照；在 MNIST 上证明了 WAM 在相同奖励水平下能更好地保持类别分布平衡。
4. **数值稳定性设计**：发现标准 flow matching 调度下 $\gamma(t)$ 和 $\gamma(t)^{-1}$ 在 $t\to 0$ 和 $t\to 1$ 处爆炸，通过引入额外时间权重 $\lambda(t) = 2(1-t)t$ 消除数值不稳定性，且不影响临界点。

## 方法详解

### 问题设定
- 预训练的流匹配模型生成过程为 SDE：$dX_t = b(X_t, t)dt + \gamma(t)dB_t$，$X_0 \sim \mathcal{N}(0,I)$。
- 使用**无记忆噪声调度**（memoryless noise schedule）：$\gamma(t) = \sqrt{2\beta_t(\frac{\dot{\alpha}_t}{\alpha_t}\beta_t - \dot{\beta}_t)}$，使得反向 SDE 保留完整路径分布。

### 加权随机最优控制问题
$$\min_{u \in \mathcal{U}} \mathbb{E}\left[\int_0^1 \frac{w(t)}{2}\|u(X_t^u, t)\|^2 dt - ar(X_1^u)\right]$$
s.t. $dX_t^u = (b(X_t^u, t) + \gamma(t)u(X_t^u, t))dt + \gamma(t)dB_t$，$X_0^u \sim \mathcal{N}(0,I)$。

其中 $w(t)$ 是**递减的时间权重函数**（如指数型 $w(t)=e^{\alpha(1-t)}$ 或分段常数型），$a>0$ 为倾斜参数。

### 等价的路径 KL 正则形式（Proposition 1）
加权 SOC 问题等价于：
$$\max_{u} \mathbb{E}_{X_0}\left[ a\,\mathbb{E}[r(X_1^u)] - w(1)\text{KL}(p^u(\mathbf{X}|X_0)\|p^{\text{base}}(\mathbf{X}|X_0)) + \int_0^1 K_t(X_0)dw(t) \right]$$
其中 $K_t(X_0) = \text{KL}(p^u(X_{[0,t]}|X_0)\|p^{\text{base}}(X_{[0,t]}|X_0))$ 是子路径 $[0,t]$ 上的条件 KL 散度。对于非增权重 $w$，$dw(t) \leq 0$，因此积分项提供了额外的正则化惩罚。

**分段常数示例**：若 $w(t) = 1+w_0$ for $t < t_s$，$w(t)=1$ for $t \geq t_s$，则 $[0,t_s]$ 区间内的路径 KL 被额外惩罚 $w_0$ 倍，强制微调模型在该时段与基础模型保持一致。

### 算法推导
1. **最优控制的特征方程**：$u^*(x,t) = -\frac{\gamma(t)}{w(t)}\nabla_x J(u^*, x, t)$，其中 $J$ 为代价函数。
2. **伴随状态**（Adjoint state）：$a(t, \mathbf{X}^u, u) = \nabla_{X_t^u}\left[\int_t^1 \frac{w(s)}{2}\|u\|^2 ds - ar(X_1^u)\right]$，满足后向 ODE：
$$\frac{d}{dt}a(t, \mathbf{X}^u, u) = -\nabla_{X_t^u}\!\left(\frac{w(t)}{2}\|u\|^2\right) - \left(\nabla_{X_t^u}(b + \gamma u)\right)^\top a, \quad a(1) = -a\nabla r(X_1^u)$$
3. **Lean 伴随状态**（Drop 零期望项后）：$\frac{d}{dt}\tilde{a} = -\nabla_x b(X_t,t)^\top \tilde{a}$，$\tilde{a}(1) = -a\nabla r(X_1)$。
4. **损失函数**：
$$\mathcal{L}_{\text{WAM}}(\theta) = \mathbb{E}_{\mathbf{X}\sim p^{\bar{u}_\theta}}\left[2\int_0^1\left\|t(v_\theta^{\text{finetune}}(X_t,t)-v^{\text{base}}(X_t,t)) + (1-t)\frac{\tilde{a}(t,\mathbf{X})}{w(t)}\right\|^2 dt\right]$$
其中 $\bar{u}=\text{stopgrad}(u)$，$\lambda(t)=2(1-t)t$ 用于消除 $\gamma(t)$ 在边界处的数值爆炸。

5. **控制参数化**：$u_\theta(x,t) = \frac{2}{\gamma(t)}(v_\theta^{\text{finetune}}(x,t)-v^{\text{base}}(x,t))$，初始化为 $\theta=\theta_0$（即 $u=0$）。

## 实验与结果

### Gaussian Mixture 实验
- **设置**：双分量高斯混合 $p_{\text{target}}=\frac{1}{2}\mathcal{N}(-\mu,\sigma_1^2 I)+\frac{1}{2}\mathcal{N}(\mu,\sigma_2^2 I)$，$\mu=(2,0)$，$\sigma_1^2=0.25$，$\sigma_2^2=0.5$，奖励 $r(x)=-\|x_2\|^2/2$。
- **权重函数**：指数型 $w_{\text{exp}}(t)=e^{\alpha(1-t)}$，分段常数型 $w_{\text{steps}}$（在 $[0,t_s)$ 取 $10^5$）。
- **结果**：Adjoint Matching 严格遵循理论倾斜分布曲线，奖励提升伴随模式失衡；WAM 在**相同奖励水平下显著改善模式平衡**，分段常数权重在 $t_s \in \{0.5, 0.65, 0.8\}$ 时实现了几乎不改变模式平衡的前提下提升奖励。

### MNIST 实验
- **设置**：UNet（9.0M 参数）训练流匹配模型，奖励为平均亮度 $r(x)=\frac{1}{d}\sum_i x_i$，$d=784$。生成 50,000 样本后用 ViT-B/16 分类器评估类别分布。
- **多样性度量**：$\frac{n_{\min}}{n_{\max}}$（最少与最多类样本数之比），原始 MNIST 约 0.80。
- **关键结果**：在奖励约 0.176–0.180 水平：
  - Adjoint Matching（$a=10$）：零占 18.0%，一仅占 3.4%
  - **WAM**（$t_s=0.45, a=227$，奖励 0.180）：**零占 10.5%，一占 9.1%**，显著更接近均匀分布
  - 黑暗奖励实验结论一致
- **最强结果**：WAM 在保持奖励的同时将类别比例从 Adjoint Matching 的 18.0%:3.4% 改善至 10.5%:9.1%，接近原始数据集的 ~10%:10% 分布。

## 相关工作脉络
1. **Adjoint Matching (Domingo-Enrich et al., 2024/2025)**：本文直接扩展的基础方法，将微调建模为 SOC 问题，证明最优控制恢复倾斜分布；本文通过时间权重引入时变正则，解决其均匀惩罚导致的失衡。
2. **ReFL / Direct Fine-Tuning (Xu et al., 2023; Clark et al., 2024)**：通过随机或最后 K 步反向传播奖励梯度微调扩散模型；这些方法不保证采样自目标倾斜分布，且未显式建模分布平衡。
3. **RL-based 微调 (Black et al., 2024; Fan et al., 2023; Uehara et al., 2024)**：将去噪链视为 MDP 并用 RL 优化；Uehara 等指出含 KL 正则的 RL 一般不采样自目标分布，需要辅助模型修正初始分布。
4. **Diversity-preserving GRPO (Liu et al., 2026; Barceló et al., 2024)**：同样利用"早期时间步决定多样性"的直觉，但基于 GRPO 强化学习框架，仅使用标量奖励信号，不刻画微调后的分布。
5. **Deterministic Adjoint Matching (Guo et al., 2026)**：将 Adjoint Matching 扩展至确定性采样器以提高效率，但牺牲了分布刻画能力，且截断微调至最后时间步是为效率而非多样性。
6. **Tilt Matching (Potaptchik et al., 2026)**：通过渐进倾斜速度场来匹配目标倾斜分布，与本文的 SOC 视角不同。

## 局限性与未来方向
1. **仅在无条件生成和简单数据集上验证**：目前实验限于 Gaussian Mixture 和 MNIST，尚未扩展到条件流匹配模型或大规模图像数据集。
2. **权重函数 $w(t)$ 的选择需要经验调参**：虽然理论上有多种选择（指数、分段常数、线性），但最优形式可能依赖于具体任务和数据结构，缺乏自动选择方法。
3. **奖励函数的局限性**：简单奖励（如平均亮度）可能诱导沿数据流形正交方向的优化，KL 正则虽部分缓解此问题，但无法解决流形内远距离模式间的质量转移。
4. **Wasserstein 正则化路线的对比**：讨论中指出 Wasserstein 正则可在理论上保留模式权重，但会导致样本离开数据流形（如 MNIST 背景变灰），暗示两种正则化各有取舍。

## 研究启发与可借鉴点
1. **时间-空间分辨率对应关系的系统性利用**：扩散/流匹配模型中"时间与空间尺度对应"的现象（由 Li & Chen 等人的临界窗口理论支撑）可推广至其他需要保持多样性的微调场景，如音频生成、时序数据等。
2. **路径 KL 正则化的等价 reformulation**：Proposition 1 揭示的"加权控制代价↔子路径 KL 积分"对应关系是一个通用工具，可用于分析其他时间非均匀正则策略。
3. **数值稳定性设计的可迁移性**：通过 $\lambda(t)$ 消除边界奇异性的技巧对任何涉及 $\gamma(t)$ 和 $\gamma(t)^{-1}$ 同时出现的 SDE 控制方法均有参考价值。
4. **与团队方向的结合机会**：若团队关注大语言模型的奖励微调（如 DPO/RLHF），可探索将"时间依赖正则"思想迁移至 token 生成过程中的"位置依赖正则"——在生成初期（粗结构）保持多样性，在后期（细节）优化奖励。

## 关键术语表
**Weighted Adjoint Matching (WAM)**：本文提出的微调算法，在 Adjoint Matching 的随机最优控制问题中引入时间依赖控制代价权重，以在提升奖励的同时保持分布多样性。
**Memoryless Noise Schedule**：Domingo-Enrich 等提出的无记忆噪声调度 $\gamma(t)=\sqrt{2\beta_t(\frac{\dot{\alpha}_t}{\alpha_t}\beta_t-\dot{\beta}_t)}$，使反向 SDE 不仅保留边际分布也保留完整路径分布。
**Mode Speciation / Critical Windows**：扩散模型中特征出现的时间窗口理论——不同空间尺度的特征在生成过程的不同时间段"定型"，大尺度特征先于小尺度特征确定。
**Tilted Distribution**：目标微调分布 $p_a(x)\propto p_{\text{target}}(x)e^{ar(x)}$，通过奖励函数对原始分布进行指数加权倾斜。
**Adjoint State**：最优控制理论中的伴随变量，表示代价函数对状态变量梯度的期望，可通过后向 ODE 计算。
**Stopgrad**：停止梯度的操作符，在 WAM 损失中用于切断从未来时间步到当前控制参数的梯度流，确保优化的正确性。
**Path KL Divergence**：受控过程与基础过程在子路径 $[0,t]$ 上的条件 KL 散度，衡量微调过程中轨迹偏离基础模型的累积程度。
**Flow Matching**：通过学习.velocity field 直接将噪声分布映射到数据分布的生成建模方法，与扩散模型在数学上等价但训练更稳定。

## 可复现要素
- **数据集**：Gaussian Mixture（人工构造）、MNIST（公开数据集，yann.lecun.com/exdb/mnist/）
- **代码**：论文声明"source code will be released at the time of publication"，当前尚未公开
- **权重**：论文未提供预训练基座模型权重
- **关键超参**：
  - 基础模型训练：Adam lr=$2\times10^{-4}$，batch=100（Gaussian）/64（MNIST），100 epochs（Gaussian）/50 epochs（MNIST）
  - 微调：Adam lr=$2\times10^{-4}$，discretization steps $K=40$，trajectories $M=1000$（Gaussian）/64（MNIST）
  - EMA decay：基础模型 0.999，微调模型 0.99（Gaussian）/0.995（MNIST）
  - 权重函数参数：指数型 $\alpha\in\{2,4,6,8\}$，分段常数 $t_s\in\{0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.65,0.7,0.8\}$
  - Loss 权重：$\lambda(t)=2(1-t)t$
