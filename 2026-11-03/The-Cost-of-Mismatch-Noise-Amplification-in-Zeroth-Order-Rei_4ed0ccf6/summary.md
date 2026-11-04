---
title: "The-Cost-of-Mismatch-Noise-Amplification-in-Zeroth-Order-Rei"
source: https://arxiv.org/pdf/2611.03260v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:09:12"
field: "大模型高效训练与强化学习"
keywords: ["zeroth-order optimization", "reinforcement learning with verifiable rewards", "noise amplification", "matched vs mismatched data", "BOLT", "LLM fine-tuning"]
innovations: ["建立YO方差分解框架，揭示不匹配数据下的1/μ²oracle噪声放大机制", "证明BOLT匹配数据regime可消除该放大并导出显式收敛保证", "提出ZO-BOLT算法并结合ZoAR/R-AdaZO实现稳定低内存RLVR训练"]
benchmarks: ["GSM8K", "MATH"]
---

# 论文速读：The-Cost-of-Mismatch-Noise-Amplification-in-Zeroth-Order-Rei

## 一句话总结
论文建立了一套方差理论框架，揭示零阶（YO）优化在在线策略强化学习（on-policy RL）中因数据不匹配会产生 **1/μ² oracle噪声放大**，形成不可降低的噪声底；而在 Boltzmann Targeted SFT (BOLT) 的匹配数据 regime 下可消除该放大，并据此提出 **ZO-BOLT**，以近推理内存占用在数学推理任务上取得一致提升。

## 研究问题与动机
- 一阶（FO）YO优化已证明可在SFT场景下以 near-inference 内存完成LLM后训练，但其在 **RLVR（带可验证奖励的强化学习）** 中的可靠性尚未探索。
- 在线策略RL每次参数更新都会重新采样轨迹，导致YO有限差分所需的两次函数评估基于**不同数据**（mismatched-data），可能引入额外方差；而BOLT等固定数据集方法天然支持**匹配数据**（matched-data）评估。
- 现有YO方差缩减技术（如轨迹重用、重要性加权、PPO/GRPO式代理目标）要么改变优化目标，要么在高阶权重下方差爆炸，无法从根本上解决直接YO-RL的噪声放大问题。
- 理解数据 regime 对YO估计可靠性的结构性影响，可为低内存RL后训练提供理论指导与实用算法。

## 核心贡献（创新点）
- **通用方差分解框架**：将YO估计误差精确拆分为无噪项与oracle-noise项，证明噪声方差由两次评估的噪声相关性决定，统一刻画不同数据regime下的误差来源。
- **1/μ²噪声放大理论**：证明on-policy RL在不匹配数据 regime 下存在 **1/μ² oracle噪声放大**（Prop. 1），并提出oracle-noise分离定理（Thm. 1），明确匹配/不匹配两 regime 的理论边界。
- **BOLT匹配数据优势证明**：证明BOLT在匹配数据 regime 下可消除 1/μ² 放大，仅保留可控的小批量方差项（Prop. 2），并导出显式常数的收敛保证（Thm. 2）。
- **ZO-BOLT算法**：将YO优化应用于BOLT的固定加权SFT目标，结合对称差分、ZoAR查询复用与R-AdaZO自适应优化，实现稳定低内存训练。
- **受控实验验证理论**：通过μ扫频与matched/mismatched端到端对比，实证支持理论预测，并在多模型/多任务上验证ZO-BOLT的性能与内存效率。

## 方法详解
- **YO梯度估计器（前向差分）**：$\hat{g}(\theta; u, \xi_+, \xi_-) = \frac{\tilde{F}(\theta + \mu u, \xi_+) - \tilde{F}(\theta, \xi_-)}{\mu} u$，其中 $u \sim \mathcal{N}(0, I_d)$，$\mu$ 为扰动尺度，$\tilde{F}$ 为含噪声的函数评估。
- **误差分解（Lemma 1）**：$\hat{g} = g_F + \frac{e(\theta+\mu u, \xi_+) - e(\theta, \xi_-)}{\mu} u$，oracle-noise 方差正比于 $\mathbb{E}[(e_+ - e_-)^2\|u\|^2]/\mu^2$，关键取决于两次评估噪声的相关性。
- **不匹配 regime 方差界（Cor. 1, Prop. 1）**：若 $\xi_+, \xi_-$ 条件独立且方差 $\le \sigma^2$，则 $\mathbb{E}\|\hat{g}-g_F\|^2 \le \frac{2d\sigma^2}{\mu^2}$；on-policy RL 的独立rollout自然满足该条件，产生 $1/\mu^2$ 放大。
- **匹配 regime 方差界（Prop. 2）**：BOLT 在相同小批量 $\mathcal{B}$ 上做两次评估，噪声对齐抵消，方差界为 $\left(\sqrt{\frac{(d+2)\sigma_{mb}^2}{B}} + L\mu\sqrt{d(d+2)(d+4)}\right)^2$，不含 $1/\mu^2$ 项。
- **收敛定理（Thm. 2）**：在 $L_F$-smooth 假设下，平均平稳度上界在 mismatched regime 含 $\frac{2d\sigma_{roll}^2}{\mu^2}$ 项，无法随 $T$ 消除；matched regime 可通过调节 $(\mu, B, \eta, T)$ 任意精度收敛。
- **ZO-BOLT 算法**：使用对称差分 + ZoAR 历史查询复用 + R-AdaZO 自适应更新；每步从固定BOLT数据集采样同批 $\mathcal{B}_t$，执行 $y^+ = F_{\mathcal{B}_t}(\theta_{t-1}+\mu u_t)$ 与 $y^- = F_{\mathcal{B}_t}(\theta_{t-1}-\mu u_t)$，复用历史对降低方差。

## 实验与结果
- **数据集**：GSM8K（小学级数学，7473训/1319测）、MATH（竞赛级，5000测）。
- **模型**：Qwen2.5-1.5B、Qwen2.5-7B、DeepSeek-R1-Distill-Qwen-1.5B，均使用 LoRA ($r=8, \alpha=16$) 微调 $q_{proj}, k_{proj}$。
- **噪声方差验证（Table 1, Figure 1）**：不匹配BOLT与on-policy RL的 $V(\mu)$ 随 $\mu$ 减小呈 $1/\mu^2$ 增长；匹配BOLT $V(\mu)$ 基本恒定（约 $2.1\times10^{-2}$）。
- **端到端对比（Table 2）**：匹配ZO-BOLT最佳准确率 **63.78%**（+11.54 vs 基线52.24%），不匹配仅49.35%（-2.89）；匹配ZO-GRPO 59.00% vs 不匹配53.08%。
- **主实验（Table 3）**：ZO-BOLT在6个设置中均优于基线；GSM8K回收一阶BOLT增益 **60.4%–98.2%**，MATH回收 **21.0%–79.9%**。
- **内存效率（Table 4）**：ZO-BOLT较一阶BOLT峰值显存降低 **59.5%（1.5B）/ 43.7%（7B）**，较推理仅增加 +920 MB（1.5B）/ +1114 MB（7B）。

## 相关工作脉络
- **MeZO / Fine-tuning with forward passes**：证明YO在SFT可行性，固定数据集天然匹配，但未处理RLVR的在线采样噪声。
- **PPO / GRPO**：通过冻结策略采样+clip surrogate实现近似匹配评估，但本质是优化冻结分布代理目标，未解决直接on-policy目标的噪声放大。
- **BOLT**：将KL正则化RLVR目标重构为离线加权SFT，实现真匹配数据优化；本文将其与YO结合并给出理论解释。
- **Variancereduced ZO / ZoAR / R-AdaZO**：方差缩减与自适应YO技术；本文将这些组件嵌入匹配数据框架，避免在不匹配设置中的无效性。
- **YO for RLHF / preference optimization**：关注无需reward inference的场景，未系统分析数据regime对噪声放大的结构性影响。

## 局限性与未来方向
- 实验仅覆盖数学推理与≤7B模型，难度更高的任务上YO与FO性能差距扩大。
- 理论假设 L-smoothness、有界方差等，实际LLM可能部分违背。
- 未深入探讨超长序列、多轮对话等更复杂RLVR场景的稳定性。
- 未来方向：扩展至更多任务/更大规模模型，改进YO优化策略（如动态μ调度、更精细的历史复用）以缩小一阶性能差距。

## 研究启发与可借鉴点
- **方差分解思路**：可将误差拆分为无噪项+噪声相关项，用于分析其他YO变体（如向后差分、Simultaneous Perturbation）在不同采样策略下的噪声行为。
- **匹配数据设计原则**：在需在线采样的优化（如meta-learning、online RL）中，优先构造共享数据批的评估对，以避免 $1/\mu^2$ 放大。
- **理论与实验闭环**：先建立方差上界预测 $1/\mu^2$ 趋势，再通过受控扫频与matched/mismatched对比验证，为后续算法设计提供可信依据。
- **内存-精度权衡量化**：明确给出YO在内存节省与收敛精度上的解析界限，可指导实际部署时的超参选择（如μ、B、$N_h$）。

## 关键术语表
- **Zeroth-Order (YO)优化**：仅依赖函数评估、通过有限差分估计梯度的优化范式，内存消耗接近推理级别。
- **Reinforcement Learning with Verifiable Rewards (RLVR)**：利用可自动验证的奖励信号（如数学答案正确性）进行强化学习的训练框架。
- **On-policy RL**：策略与数据采集同步进行的RL，每次评估需从当前策略重新采样，数据分布随参数变化。
- **Oracle-noise**：由函数评估的随机源（如数据采样、环境噪声）引入的加性噪声，在YO中被有限差分放大。
- **Boltzmann Targeted SFT (BOLT)**：将KL正则化RLVR目标转化为离线加权监督微调问题的方法，支持固定数据集上的匹配数据优化。
- **Matched / Mismatched data regime**：YO中两次函数评估是否共享相同数据批的两种设置，决定噪声是否相关及是否存在 $1/\mu^2$ 放大。
- **ZoAR**：基于查询复用的YO方差缩减技术，通过维护历史查询对降低梯度估计方差。
- **R-AdaZO**：结合一阶矩平滑与自适应预条件的YO优化器，提升非凸目标上的收敛稳定性。

## 可复现要素
- **数据集**：GSM8K（公开）、MATH（公开）；BOLT固定数据集构建细节见 Appendix D.3。
- **代码**：GitHub https://github.com/lianmin-chen/the_cost_of_mismatch （论文声明最终版本公开）。
- **模型权重**：使用开源 Qwen2.5-1.5B/7B、DeepSeek-R1-Distill-Qwen-1.5B；LoRA适配器从训练结果获取。
- **关键超参**：LoRA $r=8,\alpha=16$；学习率 $\eta=10^{-4}$（常数）；$\mu=10^{-3}$（GSM8K）/ $10^{-2}$（MATH）；历史查询数 $N_h=40$（GSM8K）/ $30$（MATH）；per-device有效批大小128；bf16精度。
