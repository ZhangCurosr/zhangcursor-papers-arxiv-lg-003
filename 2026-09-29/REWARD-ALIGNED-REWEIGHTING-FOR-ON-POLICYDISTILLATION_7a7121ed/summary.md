---
title: "REWARD-ALIGNED-REWEIGHTING-FOR-ON-POLICYDISTILLATION"
source: https://arxiv.org/pdf/2609.35517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:56:23"
field: "大语言模型后训练：有策略蒸馏与奖励对齐"
keywords: ["on-policy distillation", "reward-aligned reweighting", "knowledge distillation", "verifiable reward", "token-level supervision allocation", "continuation value", "mathematical reasoning", "code generation"]
innovations: ["用轨迹级验证结果对教师监督强度做连续 sigmoid 重加权，质量守恒且保留所有非零修正", "从一阶条件严格证明偏好–价值错配导致 over-correction/over-trust，给出 Cov(g,u)>0 的充分判据"]
benchmarks: ["AIME24", "AIME25", "AMC23", "AMC24", "MATH500", "MinervaMath", "OlympiadBench", "HumanEval+", "LiveCodeBench"]
---

# 论文速读：REWARD-ALIGNED REWEIGHTING FOR ON-POLICY DISTILLATION

## 一句话总结
本文提出 **R²-OPD**（Reward-Aligned Reweighting for On-Policy Distillation），用轨迹级验证结果动态重新分配教师监督强度：奖励一致的教学修正获得更大权重，抑制对失败路径的过度强化。在7个数学推理基准和代码生成任务上均优于标准 OPD，提升稳定且无需额外训练目标。

---

## 研究问题与动机

1. **标准 OPD 的"偏好–价值"错配**：标准 OPD 对每个 token 的监督强度仅由教师–学生 log-probability gap（$A_t$）决定，本质是本地教师偏好；但任务成功取决于学生后续的完成能力（continuation value），二者并不等价。
2. **两类失败模式**：① **过度修正**（over-correction）：对学生已成功但教师不偏好的决策施加过强抑制，破坏学生已有有效策略；② **过度信任**（over-trust）：在学生会失败的路径上强化教师偏好，强化不可执行路径。
3. **已有工作局限**：OPDVR / SG-OPD 等方法用轨迹结果做"过滤或选择"，要么丢弃不确定有用性的反馈，要么缺乏对修正强度的连续调控，没有解决"多大程度上学"的问题。
4. **缺失的连续分配机制**：现有方法要么 hard-filtering（硬过滤），要么把奖励信号当成独立 policy-gradient 项；缺少一种在保留教师监督信号的同时、用结果证据调节其相对权重的机制。

---

## 核心贡献（创新点）

1. **偏好–价值错配的严格形式化分析**：给出 Proposition 1，证明沿教师插值方向 $\pi_\epsilon$ 时，reverse-KL 一阶下降（$D'_s(0) = -\text{Var}_s[A_t]$）但期望奖励变化为 $\text{Cov}_s(A_t, Q_R)$，当协方差为负时局部模仿反而会降低任务奖励——首次从一阶条件统一刻画过度修正与过度信任。
2. **奖励对齐连续重加权（R²-OPD）**：用 sigmoid 门控 $g_t = \sigma(\beta z A_t / \nu_o)$ 与每响应系数质量守恒归一化 $Z_o$ 组合，实现 reward-aligned 修正放大、reward-inconsistent 抑制，而保留所有非零教师反馈（Proposition 2 精确分解为 Term A + Term B）。
3. **质量守恒的结构性质**：Proposition 3 证明 $\sum_t |A_t^{R2}| = M_o$ 且不一致质量份额 $\kappa_o(\beta)$ 随 $\beta$ 单调递减，为硬过滤极限的重加权提供了理论下界与连续性保证。
4. **任务进展的一阶充分条件**：Proposition 4 导出 $\langle g_R, d_{R2} - d_V \rangle = M_o \text{Cov}_{p_o}(g, u) / \mathbb{E}[g]$，给出重加权能提升期望奖励的精确一阶判据（gate 权重与 correction utility 正相关）。
5. **跨规模、跨任务的实证一致性**：1.7B 跨尺寸蒸馏均方提升 +3.5%，4B 同尺寸 +2.4%，代码生成均方 +1.6%；且学习曲线显示收益在训练早期即出现、并持续保持。

---

## 方法详解

### 符号约定
- 学生 $\pi_\theta$、冻结教师 $\pi_T$，prompt $q \sim \mathcal{D}$，轨迹 $o = (o_1, \dots, o_{|o|})$。
- 状态 $s_t = (q, o_{<t})$，采样决策 $o_t \sim \pi_{\bar{\theta}}(\cdot|s_t)$。
- **distillation advantage**：$A_t = \log \frac{\pi_T(o_t|s_t)}{\pi_{\bar{\theta}}(o_t|s_t)}$（式 3），衡量教师在该 token 的局部偏好偏离。
- **有号轨迹结果**：$z(q,o) = 2 R(q,o) - 1 \in \{-1,+1\}$，$R$ 为可验证二元裁判（式 7）。

### Reward-Aligned Gate
对每条完整回复 $o$：
1. **响应内尺度**：$\nu_o = \max\{\text{median}_{1\le j\le|o|}|A_j|,\; \epsilon\}$（式 13），将分歧幅度归一化到响应内典型量级。
2. **sigmoid 门控**：$g_t = \sigma\!\left(\beta \frac{z(q,o)\, A_t}{\nu_o}\right)$（式 14）。
   - $z A_t > 0$（奖励一致）→ $g_t > 1/2$；$z A_t < 0$（奖励不一致）→ $g_t < 1/2$。
   - $\beta=0$ 退化为标准 OPD（$g_t \equiv 1/2$）。

### 质量守恒归一化
- 响应总系数质量 $M_o = \sum_t |A_t|$。
- 归一化因子 $Z_o = M_o / \sum_j g_j |A_j|$（式 15），得到重加权 advantage：
  $$A_t^{\text{R2}} = Z_o \cdot g_t \cdot A_t$$
- 关键性质：$\sum_t |A_t^{\text{R2}}| = M_o$，系数符号不变（Proposition 3）。

### 精确分解（Proposition 2，式 17）
$$A_t^{\text{R2}} = \underbrace{\frac{Z_o}{2}A_t}_{\text{Term A（教师偏好项）}} + \underbrace{\frac{Z_o}{2} z(q,o)|A_t|\tanh\!\left(\frac{\beta|A_t|}{2\nu_o}\right)}_{\text{Term B（奖励对齐调制）}}$$
- Term A：原始教师偏差，仅以公共尺度缩放。
- Term B：在成功回复上非负、在失败回复上非正；放大奖励一致修正、衰减奖励不一致修正。
- 对任何有限 $\beta$，两项符号一致，总系数不翻转。

### 训练目标（式 16）
$$\mathcal{L}_{\text{R2}}(\theta;\bar{\theta}) = -\mathbb{E}\!\left[\sum_t \text{sg}[A_t^{\text{R2}}] \log\pi_\theta(o_t|s_t)\right]$$
advantage 与门控因子均 detached，梯度仅流过 log-likelihood，不需要独立的 RL 目标或 value estimator。

### 一阶任务进展条件（Proposition 4，式 21）
定义 $p_o(t)=|A_t|/M_o$（原质量份额）、$u_t=\text{sgn}(A_t)\langle g_R,\psi_t\rangle$（任务梯度投影每单位质量）：
$$\langle g_R, d_{R2}-d_V\rangle = \frac{M_o \text{Cov}_{p_o}(g,u)}{\mathbb{E}_{p_o}[g]}$$
- 充分条件：$\text{Cov}_{p_o}(g,u)>0$，即 gate 权重与 correction 的任务效用正相关。
- 纯减少不一致质量不足以保证任务进展（还需 utility 关联）。

---

## 实验与结果

### 设置
- **模型**：学生 Qwen3-1.7B-Base / Qwen3-4B-Base；教师 Qwen3-4B-GRPO（冻结）。
- **训练数据**：DAPO-Math-17k。
- **超参**：固定 $\beta=10^{-3}$、$\epsilon=10^{-8}$，无任务特定调参。
- **基线**：Sampled-Token OPD、Top-64 OPD、OPD+GRPO、OPDVR、SG-OPD、ExOPD。
- **基准**：7 个数学推理（AIME24/25、AMC23/24、MATH500、Minerva、OlympiadBench）+ 2 个代码生成（HumanEval+、LiveCodeBench）。

### 主要结果

**跨尺寸 4B→1.7B**（均值±std，3 seeds）：
| 方法 | AIME24 | AIME25 | AMC24 | MATH500 | Minerva | Olympiad | 均方 |
|---|---|---|---|---|---|---|---|
| Sampled-Token OPD | 25.5 | 20.2 | 48.3 | 74.7 | 27.6 | 42.9 | **42.7** |
| OPDVR | **29.8** | 20.4 | **52.4** | 76.3 | 28.0 | **44.5** | 45.0 |
| **R²-OPD** | 29.4 | **24.6** | 53.5 | **77.6** | **29.1** | **45.9** | **46.2** |

- R²-OPD **均方第一**（46.2%），较标准 OPD 提升 **+3.5**，较次优 OPDVR 提升 **+1.2**。
- 最难任务增益最大：AIME25 +4.4、AMC24 +5.2、Olympiad +3.0。
- 学习曲线：~step 40 即在 AIME25 上拉开显著差距；~step 60 达到近峰值并保持优势。

**同尺寸 4B→4B**：
- R²-OPD 均方 **50.8%**，较标准 OPD（48.4%）提升 **+2.4**。

**代码生成**（Table 2）：
| 方法 | HumanEval+ | LiveCodeBench | 均方 |
|---|---|---|---|
| Vanilla OPD | 75.4 | 43.7 | 59.6 |
| OPDVR | 76.2 | 44.2 | 60.2 |
| **R²-OPD** | **77.3** | **45.1** | **61.2** |

- 均方 **+1.6** vs OPD，**+1.0** vs OPDVR。

### 消融（Table 3，三基准均值变化）
| 变体 | Δ_avg |
|---|---|
| **Full R²-OPD** | — |
| Reversed alignment | **-3.8** |
| Outcome-permuted | -1.7 |
| Sign-only gate | -1.6 |
| Magnitude-only gate | -2.4 |
| No mass-norm | -0.9 |
| R²-OPD + GRPO | +0.8 |

- **符号反转最大**（-3.8），说明 reward 对齐方向决定性地重要。
- **sign-only vs magnitude-only**：两者均退化，说明门控需要同时利用结果符号与分歧幅度。
- **mass-norm** 移除仅 -0.9，但仍具贡献。
- **R²-OPD + GRPO** 额外 +0.8，表明两种信号互补。

### 训练动力学（Figure 2）
- R²-OPD 训练 reward 全程高于标准 OPD。
- 关键发现：R²-OPD 在 **更高 reward 的同时保持更高 entropy**，说明收益并非靠过早收敛换取。

---

## 相关工作脉络

1. **OPD 基础**：Agarwal et al. (2024, GKD)、Gu et al. (2024, MiniLLM) 奠定 reverse-KL 与 sampled-token 形式；Lu & Thinking Machines Lab (2025) 明确 per-token log-ratio 为 policy-gradient 风格奖励。
2. **OPDVR**（Lin et al., 2026）：用 verifier 做 outcome-inconsistent 修正的 hard mask；R²-OPD 的区别在于 **连续重加权**而非丢弃，并保留系数质量。
3. **SG-OPD**（Xu et al., 2026a）：sign-consistency 路由 + 分阶段采样教师；R²-OPD 不需切换采样策略，仅在原 OPD 系数上做门控。
4. **ExOPD**（Yang et al., 2026b）：放大 teacher–reference log-ratio 相对于 KL regularization；属于正则项配比策略，R²-OPD 直接调制每个 token 的监督信号强度。
5. **TSD-KD**（Kim & Baek, 2026）、Yuanda Xu et al. (2026b, TiP)：token-selective 或 importance-weighted OPD，但未利用可验证结果。
6. **G-OPD / Uni-OPD**：广义 KL 约束 RL 形式；R²-OPD 不与 KL 正则耦合，仅在 distillation advantage 上操作。
7. **RLVR 系**：Shao et al. (2024, DeepSeekMath)、Guo et al. (2025) 用 verifier 作 RL 奖励；R²-OPD 不与 RL 目标组合（除非显式加 GRPO），pure distillation 即可生效。

---

## 局限性与未来方向

1. **一阶条件仅充分非必要**：Proposition 4 的 $\text{Cov}(g,u)>0$ 是 local first-order 判据，未覆盖有限步、非线性 optimizer 下的全局保证；文中 Appendix H 也专门区分了该条件与 finite-step reward ascent。
2. **仅用 binary verifier**：$z \in \{-1,+1\}$ 是整条轨迹共享的结果标签，对单 token 的效用推断存在噪声；连续奖励或 multi-step verifier 下的推广未讨论。
3. **$\beta$ 对数值敏感**：虽然 sweep 显示 $10^{-4} \sim 10^{-1}$ 均表现良好，但极端值（$\beta \ge 1$）会接近 hard gating 并削弱性能；未提出自动调 $\beta$ 的策略。
4. **截断响应走 vanilla fallback**：长度截断的回复不参与重加权，可能遗漏部分失败路径信息；Appendix I 指出这属于设计选择，未讨论自适应处理。
5. **未验证多 verifier 场景**：当前仅适用于可精确判定的数学/代码任务；自然语言理解、开放生成等软判定场景的适配性存疑。
6. **理论边界案例未覆盖**：Proposition 3 中 $\kappa_o=0$ 或 $1$（全一致或全不一致）时单调性退化，实践中虽少见但未给出应对。

---

## 研究启发与可借鉴点

1. **"偏好–价值分解"的分析范式**：Proposition 1 以 $\text{Cov}(A_t, Q_R)$ 统一刻画 over-correction 与 over-trust，这一思路可直接迁移到其他 distillation/alignment 任务，作为诊断工具判断教师反馈是否有害。
2. **质量守恒归一化的设计技巧**：先做 per-response 门控再乘 $Z_o$ 恢复总系数质量，是一种"不调总预算只调内部配比"的通用重加权模板，可复用到 reward weighting、data reweighting 等场景。
3. **连续 gate 替代 hard mask**：相比 OPDVR 的丢弃式处理，sigmoid 门控保留所有反馈且可微（gate 本身 detached），对训练稳定性更友好，可作为 selective supervision 的一般替代方案。
4. **Training dynamics 联合监控**：Figure 2 同时跟踪 reward 与 entropy，发现 R²-OPD 在更高 entropy 下仍获得更高 reward，这一联合诊断可用来区分"真学习"与"过拟合式收敛"，值得在后续工作中沿用。
5. **R²-OPD + GRPO 组合实验**（Table 3 末行 +0.8）提示 dense distillation signal 与 direct reward optimization 可并行，后续可探索两路信号的动态配比或阶段性切换。

---

## 关键术语表

- **On-Policy Distillation (OPD)**：在学生自身生成的轨迹上向教师求 log-probability，用 reverse-KL 或其 sampled-token 近似进行稠密教师监督。
- **Distillation Advantage ($A_t$)**：$A_t = \log \frac{\pi_T(o_t|s_t)}{\pi_{\bar{\theta}}(o_t|s_t)}$，教师在当前位置对采样 token 的局部偏好偏离，正为强化、负为抑制。
- **Signed Outcome ($z$)**：$z = 2R - 1 \in \{-1,+1\}$，整条轨迹的二元裁判结果转为有号信号，与 $A_t$ 相乘判断 reward-alignment。
- **Reward-Aligned Correction**：$z \cdot A_t > 0$ 的 token 修正；reward-inconsistent 为 $z \cdot A_t < 0$。
- **Mass-Preserving Normalization ($Z_o$)**：$Z_o = M_o / \sum_j g_j |A_j|$，使重加权后 $\sum_t |A_t^{R2}| = M_o$，保持响应内总系数预算。
- **Continuation Value ($Q_R^{\bar{\theta}}$)**：给定前缀和当前动作后，按学生策略完成剩余部分期望得到的 reward，是任务价值的真正决定因素。
- **Over-Correction / Over-Trust**：前者指对学生本可成功的决策施加抑制（$A_t < 0, Q_R > F_s$）；后者指对学生会失败的路径强化教师偏好（$A_t > 0, Q_R < F_s$）。
- **Gate Sharpness ($\beta$)**：控制 sigmoid 门控对 reward-alignment 信号的放大程度；$\beta=0$ 退化为标准 OPD。

---

## 可复现要素

- **数据集**：训练用 DAPO-Math-17k（Yu et al., 2025）；评测用 AIME24/25、AMC23/24、MATH500、Minerva、OlympiadBench、HumanEval+、LiveCodeBench——均为公开基准。
- **模型权重**：Qwen3-1.7B-Base、Qwen3-4B-Base、Qwen3-4B-GRPO 均来自开源 Qwen3 系列。
- **代码**：**论文未声明开源**（Reproducibility Statement 仅指向附录）。
- **关键超参**：$\beta = 10^{-3}$（全局固定）、$\epsilon = 10^{-8}$、lr = $1\times 10^{-6}$、AdamW、global l2 clip = 1.0、rollout top-p=1、temperature=1.0、responses/prompt=64、optimizer batch=256、max prompt=2048、max response=8192、micro-batch ≤ 10240 tokens。
- **硬件**：单节点 8× NVIDIA H800-80GB，NVLink 400 GB/s；软件栈 PyTorch 2.11.0 / CUDA 12.9 / SGLang 0.5.12 / FlashAttention 2.7.4。
- **评估协议**：avg@16（AIME24/25）、avg@4（其余）；temperature=1.0、top-p=0.95。

---
