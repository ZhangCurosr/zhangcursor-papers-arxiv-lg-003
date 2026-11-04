---
title: "When-Rewards-Depend-on-the-Group-GRPO-Plays-a-Tournament"
source: https://arxiv.org/pdf/2611.02404v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:44:37"
field: "大语言模型强化学习"
keywords: ["GRPO", "group-relative policy gradient", "diversity reward", "distribution matching", "Team-GRPO", "reinforcement learning", "LLM finetuning", "credit assignment"]
innovations: ["证明组依赖奖励下 GRPO 的组内减均值翻转外部性符号（α=-1/(K-1)），将训练目标扭曲为零和锦标赛", "提出 Team-GRPO，以留一总奖励差计算无偏梯度估计，统一 max@k/分布匹配/SetPO 等已有估计器", "在 9B 模型上实现理论不动点的定量预测（Pearson r=0.995），并在 GlobalOpinionQA 和 NoveltyBench 上验证"]
benchmarks: ["GlobalOpinionQA", "NoveltyBench"]
---

# 论文速读：When-Rewards-Depend-on-the-Group-GRPO-Plays-a-Tournament

## 一句话总结
论文证明了当奖励函数依赖于整个采样组时，标准 GRPO/RLOO 的组内减均值操作会**系统性翻转**外部性信号的符号，将训练目标扭曲为零和博弈（锦标赛），从而破坏多样性、分布匹配等组级目标；作者提出 **Team-GRPO**，在任意组大小（包括 $K=2$）下无偏估计真实的组级梯度。

## 研究问题与动机
- **问题背景**：当前 LLM 强化学习中，多样化奖励（DARLING、GAPO 等）、分布对齐（SWYS）及集合级目标（pass@k、Poly-EPO）均依赖组内样本，GRPO/RLOO 已成为标准训练框架。
- **现有方法的缺陷**：经典基线减均值的无偏性假设是"各响应奖励相互独立"；当 $r_k$ 依赖整个组时，基线隐含地依赖了当前响应的策略参数，导致期望更新不等于真实梯度。
- **理论后果未量化**：此前领域内缺乏对"组依赖奖励+组相对基线"的系统分析，各方法各自设计专用估计器，缺乏统一视角。
- **低 $K$ 场景的致命问题**：$K=2$ 是最常见的高效训练配置（每 prompt 仅 2 个样本），但 GRPO 在此配置下对任何对称成对相互作用完全失敏（权重为 0），无法优化多样性。

## 核心贡献（创新点）
- **精确恒等式与零和博弈诊断**：证明了对任意可交换组奖励，GRPO 的期望更新等于 `own 梯度 + α·ext 梯度`，其中 $\alpha = -1/(K-1)$，将组内减均值等价于一个相对回报零和博弈，揭示了多样性被系统性逆转的根本原因。
- **Team-GRPO 无偏估计器**：提出 $\alpha=+1$ 的无偏梯度估计，利用 leave-one-out 总奖励差计算每个响应对组目标的边际贡献，**不增加额外的 reward model/embedding/judge 调用**，且统一了 max@k、分布匹配 witness estimator、SetPO 等已有方法为特例。
- **定量可预测性验证**：在 9B 模型上通过精确枚举和闭式不动点分析，预测与实际收敛位置误差 $\ell_1 \leq 0.018$，跨 13 个配置的浓度预测与实测 Pearson $r = 0.995$。
- **六种组奖励结构下的系统对比**：在精确计算下证明 $\text{GRPO} \leq \text{REINFORCE} \leq \text{Team-GRPO}$ 全序关系普遍成立，GRPO 在分布匹配 $K=2$ 下目标甚至变为负值（-31%）。

## 方法详解
- **核心分解**：定义两个标量函数：
  $$\operatorname{own}(y)=\mathbb{E}[r_1\mid y_1=y],\quad \operatorname{ext}(y)=\sum_{j=2}^K\mathbb{E}[r_j\mid y_1=y]$$
  分别表示该响应的自身期望奖励和其组友的总期望奖励（外部性）。

- **定理 1（$\alpha$-恒等式）**：对任意可交换组奖励，
  $$\nabla_\theta J = g_1,\quad \mathbb{E}\hat{g}_{\text{REINFORCE}} = Kg_0,\quad \mathbb{E}\hat{g}_{\text{RLOO}} = Kg_{-1/(K-1)},\quad \mathbb{E}\hat{g}_{\text{GRPO}} = (K-1)g_{-1/(K-1)}$$
  所有估计器落在以 $\alpha$ 为参数的直线上：$\alpha=1$ 为真实梯度，$\alpha=0$ 为 Nash 方向，GRPO 的 $\alpha=-1/(K-1)$ 为负值。

- **衰减阶梯（Corollary 2）**：对 $m$ 体对称核 $\phi$，GRPO 保留的相互作用权重为
  $$\rho_m(K)=\frac{K-m}{m(K-1)}<\frac{1}{m}$$
  成对多样性（$m=2$）在 $K=2$ 权重为 0，$K=4$ 为 $1/3$，$K=8$ 为 $3/7$，永远不超过 $1/2$。

- **Team-GRPO 优势函数**：
  $$A_k^{\text{TEAM}}=R(\mathcal{G})-\frac{K}{K-1}R(\mathcal{G}\setminus\{y_k\})$$
  其中 $R(S)=\sum_{y_i\in S}r(y_i;S\setminus\{y_i\})$ 为集合总奖励。等价于 $K$ 倍"该响应对组平均奖励的改变量"，符合多智能体差分奖励（difference reward）形式，可在 $O(K^2)$ 算术内计算，无需额外模型调用。

- **$K=2$ 实现**：留一集合退化为单响应，成对多样性项未定义；使用来自队友和批量统计的基线 $b_k$ 估计 $\mathbb{E}[r_1+r_2\mid y_j]$，使方差降低 2.6 倍（Table 14）。

- **与已有方法的关系**：unbiased max@k estimator（PKPO/L2O）、SetPO 的 leave-one-out 多样性项、非中心化 witness estimator 均为 Team-GRPO 的特例；centered witness estimator 存在 $\frac{1}{K}\nabla\|\pi\|^2$ 的浓度偏差。

## 实验与结果
- **9B 策略精确读取实验**（Qwen3.5-9B，四选项，diversity bonus $f=(1,0.5,0.2,0)$，$\lambda=1$，128 prompts，500 steps）：12 组实验全部收敛至闭式不动点，最大 $\ell_1$ 误差 0.018；GRPO 在 $K=2$ 将所有质量压至最优选项（$(1,0,0,0)$），Team-GRPO 在所有 $K$ 下达到正确的 diversification。
- **六种奖励结构的精确不动点对比**（Table 1）：顺序 $\text{GRPO}\leq\text{REINFORCE}\leq\text{Team-GRPO}$ 在全部 18 个设置中成立；分布匹配 $K=2$ 时 GRPO 目标为 Team-GRPO 的 **-31%**；coverage 任务 GRPO 仅为 Team-GRPO 的 **3%**。
- **GlobalOpinionQA 分布对齐**（Qwen3.5-9B，256 rollouts/step，200 steps，510 未见问题）：GRPO 在 $K=2$ 坍缩至多数答案，误差从 base 模型的 0.197 升至 **0.491**（劣于未训练模型）；Team-GRPO 降至 **0.081**（比 GRPO 低 $-0.40[-0.43,-0.38]$）；13 个配置的预测浓度与实测 Pearson $r=0.995$。
- **NoveltyBench 开放式生成**（DARLING 型奖励 $r_k=q_k(1+\lambda D_k)$，$\lambda=2$，64 rollouts/step，150 steps）：
  - $K=2$：GRPO distinct answers 从 4.33→**2.65**，utility 4.01→**3.44**（崩溃）；Team-GRPO 为 **4.56 / 4.61**，较 GRPO +1.16[0.52,1.80]。
  - $K=8$：GRPO utility 4.12，Team-GRPO **4.68**，+0.56[0.32,0.81]。
  - LLM Judge（Claude Sonnet 5.5，100 prompts）验证结论一致：$K=2$ 时 Team-GRPO utility +0.61，distinct +2.09。
  - 更大 $\lambda$ 使 GRPO 退化更快（Figure 5d），而 Team-GRPO 持续变得更多样。

## 相关工作脉络
- **Williams (1992) 基线理论**：基线应与 action 无关才无偏；本文指出通过 groupmate rewards 间接依赖 action 的基线（即减均值）会引入系统性偏差，将这一现象精确定量为 $\alpha$-线上的移动。
- **Kool et al. (2019)、Shao et al. (2024) GRPO/RLOO**：引入用于 critic-free LLM RL；本文证明两者在组依赖奖励下均为有偏估计，且偏差方向对多样性等目标有害。
- **Li et al. (2025) DARLING**：将 quality 乘以 semantic distance；本文证明 GRPO 优化 DARLING 奖励时，成对多样性项被严重衰减（$K=2$ 完全消失，$\lambda$ 增大反而加速 collapse），且无法通过调整系数修复（Theorem 5）。
- **Arabi et al. (2026) SWYS（witness estimator）**：非中心化版本无偏，但 centering（减组均值）引入 $\frac{1}{K}\nabla\|\pi\|^2$ 的浓度偏差，与 GRPO 偏差方向相同；Team-GRPO 在非中心化情况下与其等价。
- **Walder & Karkhanis (2025)、Takashiro et al. (2026) PKPO/max@k**：各自设计了专用无偏估计器；本文为其提供了统一框架——它们正是 Team-GRPO 在特定奖励下的实例。
- **Schaffer (1988)、Hamilton (1970) 有限群体选择理论**：本文借用相对回报博弈框架，将 GRPO 的收敛行为与进化生物学中的"spiteful selection"建立等价，解释了锦标赛为何偏好同质化。

## 局限性与未来方向
- 理论分析主要在 softmax 策略和有限答案空间的精确设定下进行，对连续答案空间（如自由文本）的不动点分析尚需更多数值验证。
- 大组场景（$K\geq 8$）下 Team-GRPO 与 RLOO 的方差比约为 1.1–2.1 倍，虽理论上有额外方差代价，但论文仅在小规模实验上验证了偏差-方差权衡。
- 未系统讨论 reward model 本身可能存在的噪声或偏差对 Team-GRPO 估计的影响。
- 论文未提供开源代码，复现需自行实现。
- 未来方向：扩展到 off-policy 情形（当前假设 importance ratio=1）、带 clip 的多 epoch 训练的定量分析（附录 F 已有初步结果）、以及更高效的 $K=2$ 基线设计。

## 研究启发与可借鉴点
- **$K=2$ 的高效 RL 场景具有实际价值**：本文证明 $K=2$ 是 GRPO 彻底失效的最坏配置（对成对相互作用完全盲视），而 Team-GRPO 在 $K=2$ 下仍然无偏且效果显著，对低资源/高成本 RL 训练具有直接指导意义。
- **α-线框架的通用诊断价值**：任何组依赖奖励+组相对基线的方法都可以用同一框架分析，可快速判断其是否无意中放大了反向信号（如 GAPO 的频率惩罚在 GRPO 下被放大 $K/(K-1)$ 倍）。
- **差分离散奖励思想的迁移**：Team-GRPO 本质上是多智能体 credit assignment 中 difference reward 思想的直接迁移，可推广至任意"每 prompt 多个采样"的 RL 设定，如 multi-turn reasoning、self-play 等。
- **精确不动点分析的复现范式**：在可枚举答案空间下验证理论预测是一种低成本、高可信度的研究方法，值得在小规模 RL 新方法的验证阶段借鉴。
- **团队与锦标赛的二元视角**：将组内样本的关系重新理解为"队友"还是"对手"，这一概念框架可用于解释其他组级 RL 方法的失效现象，启发新的方法设计思路。

## 关键术语表
**Exchangeable group reward**：奖励对组内响应对称排列不变，即 $r_k=r(y_k;\{y_j\}_{j\neq k})$，涵盖多样性、分布匹配、pass@k 等常见组级奖励。

**own(y) / ext(y)**：自身期望奖励（$\mathbb{E}[r_k\mid y_k=y]$）与组友总期望奖励（$\sum_{j\neq k}\mathbb{E}[r_j\mid y_k=y]$），是分解组依赖奖励梯度的两个基本分量。

**$\alpha$-line**：所有组相对估计器的期望更新落在以 $\alpha$（外部性权重）为参数的直线上，$\alpha=1$ 为真实梯度，$\alpha=0$ 为 Nash 方向，$\alpha=-1/(K-1)$ 为 GRPO/RLOO。

**Zero-sum tournament**：GRPO 对应的相对回报博弈（$U_k=r_k-\bar{r}_{-k}$），组内响应对彼此的奖励之和为零，激励每个响应击败组友而非服务组目标。

**Attenuation ladder**：对 $m$-体对称核，GRPO 保留的相互作用权重为 $\rho_m(K)=(K-m)/[m(K-1)]$，随 $m$ 增大而衰减，永远低于 $1/m$。

**Team-GRPO**：本文提出的无偏估计器，优势为 $K$ 倍"留一后组平均奖励的改变量"，在所有 $K\geq 2$ 下无偏，统一了多个已有专用估计器。

**Difference reward**：多智能体信用分配中的概念，指每个智能体的回报等于其加入组前后总奖励的差；Team-GRPO 是其单策略多采样的精确实现。

**Centered witness bias**：SWYS 方法中减组均值的 witness estimator 引入 $\frac{1}{K}\nabla\|\pi\|^2$ 偏差，方向指向集中化（与 GRPO 同向），Team-GRPO 不使用此中心化操作。

## 可复现要素
- **数据集**：GlobalOpinionQA（Durmus et al., 2024，公开）、NoveltyBench（Zhang et al., 2025，公开）、WildChat 训练子集（Zhao et al., 2024，公开）。
- **代码**：论文未提及开源代码。
- **模型**：Qwen3.5-9B（公开权重）、Skywork-Reward-V2-Llama-3.1-8B 和 Skywork-Reward-Gemma-2-27B-v0.2（公开）、E5-large-v2 嵌入（公开）。
- **关键超参**：LoRA rank=16，temperature=1，无 KL penalty，无 std normalization；GlobalOpinionQA：256 rollouts/step，lr=$2\times10^{-5}$，200 steps；NoveltyBench：64 rollouts/step，lr=$5\times10^{-5}$，150 steps，$\lambda\in\{1,2\}$。
- **基线模型**：Qwen3.5-9B（论文声明）。
