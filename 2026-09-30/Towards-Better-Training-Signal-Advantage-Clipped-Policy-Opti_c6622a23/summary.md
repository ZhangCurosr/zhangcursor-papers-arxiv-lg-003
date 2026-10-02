---
title: "Towards-Better-Training-Signal-Advantage-Clipped-Policy-Opti"
source: https://arxiv.org/pdf/2609.36816v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:36:19"
---

# 论文速读：Towards-Better-Training-Signal-Advantage-Clipped-Policy-Opti

## 一句话总结
本文提出ACPO（Advantage Clipped Policy Optimization），直接对重要性采样比率与优势函数的乘积进行对称裁剪以稳定策略梯度更新；在Qwen系列模型的数学推理任务上，ACPO相比PPO和GRPO基线稳定提升1.6–4.0个百分点准确率，并将训练步数缩短6.25倍（vs PPO）和1.7倍（vs GRPO）。

## 研究问题与动机
- **Off-policy数据重用的稳定性困境**：RL后训练LLM需要on-policy数据，代价高昂；通过importance sampling（IS）重用off-policy数据可提升效率，但IS权重导致梯度估计不稳定。PPO/GRPO采用IS比率裁剪来缓解，但未直接控制决定梯度方向和幅度的核心量。
- **现有裁剪机制的理解不足**：PPO仅裁剪IS比率$r_t(\theta)$，但策略梯度的实际贡献由乘积$r_t(\theta)\widehat{A}_t$决定——该乘积同时决定更新方向和幅度，仅裁剪比率无法充分约束梯度估计的波动。
- **不同难度提示的梯度系数分布差异显著**：实验观察到硬提示（准确率0–30%）产生较大正尾（过度强化倾向），易提示（70–100%）产生较大负尾，而中等难度提示的token级更新更集中稳定；现有方法未将难度信息纳入裁剪设计。
- **裁剪机制与优化理论缺乏系统联系**：梯度裁剪是优化领域的标准技术，但其在RL策略优化中的理论角色尚未被清晰阐释，缺乏从优化视角理解裁剪稳定性的统一框架。

## 核心贡献（创新点）
- **提出ACPO算法，直接裁剪IS比率与优势的乘积**：将策略梯度标量系数$r_t(\theta)\widehat{A}_t$裁剪至$[-\alpha, \alpha]$，相比PPO仅裁剪IS比率，更直接地控制每个样本的梯度贡献幅度；可无缝集成到PPO和GRPO中形成PPO-AC和GRPO-AC。
- **证明ACPO梯度估计的二阶矩不大于PPO，并提供方差缩减保证**：在同等保留概率和梯度范数条件假设下，$\mathbb{E}\|G_{\mathrm{ACPO}}\|^2 \leq \mathbb{E}\|G_{\mathrm{PPO}}\|^2$；若进一步假设两种裁剪后梯度范数相近，则$\mathrm{Var}(G_{\mathrm{ACPO}}) \leq \mathrm{Var}(G_{\mathrm{PPO}})$。
- **建立ACPO与Policy Mirror Descent (PMD)梯度裁剪的理论联系**：从带梯度裁剪的PMD出发，经重要性采样替换和移除KL邻近惩罚，可推导得到ACPO目标；证明clipped-PMD在标准RL设置下带强凸正则化时线性收敛、无正则化时$O(1/N)$次线性收敛。
- **系统在多模型规模上验证ACPO的有效性与效率优势**：在Qwen2.5-Math-7B和Qwen3（1.7B/4B/8B）四个数学推理基准上，ACPO在所有八组匹配比较中均最优；PPO-AC仅需每prompt 1个rollout即可达到GRPO（需4–8个rollout）相当的最终性能。

## 方法详解
**ACPO目标函数**：
$$\mathcal{I}^{\mathrm{ACPO}}(\theta) = \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta_{\mathrm{old}}}}\left[\mathrm{clip}(r_t(\theta)\widehat{A}_t, -\alpha, \alpha)\right]$$
其中$r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\mathrm{old}}}(a_t|s_t)$为IS比率，$\widehat{A}_t$为优势估计（可用PPO类型的token级GAE或GRPO类型的序列级归一化奖励），$\alpha$为对称裁剪阈值。

**裁剪阈值的选取依据**：通过追踪不同难度prompt下$r_t(\theta)\widehat{A}_t$的分布发现——硬提示呈现显著正尾（少数token获得大正系数，过度强化），易提示呈现重负尾（少数token获得大负更新），中等难度提示分布集中；对称裁剪$[-\alpha, \alpha]$抑制极端尾部，使优化行为趋近中等难度提示的稳定模式。

**与PPO/GRPO的集成方式**：ACPO可分别嵌入PPO和GRPO的裁剪目标，替换原有的IS比率裁剪逻辑；PPO-AC使用token级优势，GRPO-AC使用序列级归一化优势，两者均保留各自原有的多rollout或单rollout设计。

**理论性质**：
- **Proposition 1**（二阶矩比较）：在保留概率相等$\mathbb{E}[\mathbf{1}_{\mathrm{PPO}}]=\mathbb{E}[\mathbf{1}_{\mathrm{ACPO}}]$和条件梯度范数恒定假设下，$\mathbb{E}\|r\widehat{A}\nabla\log\pi\cdot\mathbf{1}_{\mathrm{ACPO}}\|^2 \leq \mathbb{E}\|r\widehat{A}\nabla\log\pi\cdot\mathbf{1}_{\mathrm{PPO}}\|^2$。
- **PMD梯度裁剪推导**：从Algorithm 1（带梯度裁剪的PMD）出发，以负熵为mirror map、$h^p=0$，经重要性采样替换优势项、近似求解子问题、移除KL惩罚，最终得到ACPO目标（见Appendix B）。
- **收敛性（Theorem 1, 2）**：clipped-PMD在常数步长下，带μ-强凸正则化时$F_N + (\mu+1/\eta)\mathcal{D}_N \leq \rho^N[F_0+(\mu+1/\eta)\log|\mathcal{A}|]$（线性收敛，$\rho<1$）；无正则化时$F_N \leq C_0/[(1-\gamma)N]$（$O(1/N)$次线性收敛）。

## 实验与结果
**实验设置**：
- **训练数据**：DAPO训练集（约17.4k数学问题，来自AoPS及官方竞赛首页），使用Math-Verify工具自动验证答案正确性。
- **模型**：Qwen2.5-Math-7B、Qwen3-1.7B、Qwen3-4B、Qwen3-8B。
- **评估基准**：MATH500、Minerva Math、OlympiadBench、AIME-like（230题，含AIME24/25、HMMT24/25、BRUMO25、AMC23、CMIMC25）；报告Pass@1加权准确率（Qwen2.5用avg@32，Qwen3用avg@16），temperature=1.0，top-p=1，max tokens=8196。
- **训练框架**：VERL；PPO/GRPO基线裁剪范围$\epsilon_{\mathrm{low}}=0.2$，$\epsilon_{\mathrm{high}}=0.2$（PPO）或$0.28$（GRPO）；PPO-AC和GRPO-AC的$\alpha$分别为3（Qwen2.5）和2/3（Qwen3）。

**主要结果（Table 1，最佳加权准确率）**：
- **Qwen2.5-7B-Math**：PPO-AC相比PPO在MATH500（48.4→50.0）、Minerva（79.0→80.5）、Olympiad（33.0→35.3）、AIME-like（21.4→21.9）均提升；GRPO-AC相比GRPO同样全面超越。
- **Qwen3-8B**：PPO-AC相比PPO在MATH500（67.3→71.3，+4.0）、Minerva（93.1→94.8）、Olympiad（63.7→70.5，+6.8）、AIME-like（43.6→48.8，+5.2）显著提升；GRPO-AC相比GRPO在Olympiad（48.7→49.4）和AIME-like（47.3→49.0）改善。
- **提升幅度趋势**：模型越大，ACPO相对PPO的提升越显著（1.6→4.0个百分点），在更具挑战性的Olympiad和AIME-like基准上增益尤为突出。

**训练效率（Figure 2）**：
- ACPO平均训练步数加速：**6.25倍**（vs PPO）和**1.7倍**（vs GRPO）；Qwen3-4B/8B上PPO-AC在约40步内达到PPO在200步的最佳性能（5.0–5.3倍加速）。
- **PPO-AC仅需1个rollout/prompt**即可达到GRPO（需4–8个rollout）的相当性能，大幅降低计算开销。
- Qwen3-1.7B上PPO在20步后准确率下降，而PPO-AC持续改善，体现更强的鲁棒性。

**熵动态（Figure 3）**：
-  Vanilla PPO/GRPO存在快速熵坍缩，限制探索；ACPO显著缓解此问题：GRPO-AC的熵在训练初期保持稳定，PPO-AC的熵下降速度远慢于PPO；机制在于ACPO保留了小优势token的贡献，维持了策略多样性。

## 相关工作脉络
- **PPO (Schulman et al., 2017)**：首次引入IS比率裁剪的代理目标，ACPO在此基础上将裁剪对象从$r_t(\theta)$扩展为$r_t(\theta)\widehat{A}_t$，直接约束梯度系数而非仅约束概率比。
- **GRPO (Shao et al., 2024)**：无critic的组内相对策略优化，使用序列级归一化优势；ACPO-GRPO变体保留其无critic设计并改进裁剪机制，在相同rollout开销下获得更稳定训练。
- **DAPO (Yu et al., 2025)**：引入Clip-Higher（裁剪范围$[0.2, 0.28]$）鼓励对低概率exploratory token的概率提升；ACPO采用对称裁剪$[-\alpha, \alpha]$，从梯度系数幅度角度统一处理正负方向。
- **CISPO (MiniMax et al., 2025)**：裁剪IS权重但保留对应梯度贡献；ACPO直接裁剪梯度系数本身，设计更简洁且理论保证更清晰。
- **Mirror Descent for Policy Optimization (Tomar et al., 2022; Song et al., 2026)**：从镜下降推导实际训练目标；本文视角相反——从PMD梯度裁剪出发反向推导ACPO，为裁剪机制提供优化理论解释。
- **Policy Mirror Descent (Lan, 2023)**：建立PMD收敛理论框架；本文在其框架下证明clipped-PMD的收敛性，并将ACPO嵌入该理论体系。

## 局限性与未来方向
- **超参数α需手动调优**：不同模型规模需要不同α（Qwen2.5用3，Qwen3用2），缺乏基于数据或训练动态自动确定α的机制。
- **仅在数学推理任务上验证**：实验集中在数学推理领域，未验证在代码生成、对话、多模态等其他LLM后训练场景的泛化性。
- **理论推导移除了KL惩罚**：为隔离裁剪机制作用，理论分析中去除了KL邻近惩罚项，与实际PPO/GRPO训练设置不完全一致；重新引入KL项后的理论性质未展开。
- **优势估计质量的影响未深入分析**：ACPO的性能依赖优势估计的准确性，对于GRPO类型的序列级优势（同一序列内所有token共享同一优势值）可能不如PPO的token级GAE精细，文中对此讨论有限。
- **方差缩减保证的假设较强**：Proposition 1的结论依赖"同等保留概率"和"条件梯度范数恒定"等假设，实际训练中这些条件未必严格成立。

## 研究启发与可借鉴点
- **梯度系数的直接控制范式**：将策略梯度拆解为"标量系数$\times$梯度方向"，直接对系数$r_t\widehat{A}_t$进行裁剪而非仅对$r_t$裁剪，这一思路可迁移到其他策略梯度方法（如TRPO变体、REINFORCE+Baseline）的稳定性改进中。
- **难度感知的裁剪设计启发**：通过观察不同难度prompt下梯度系数的尾部差异，启发了对称裁剪的动机；未来可探索基于prompt难度动态调整α的自适应裁剪机制，或结合课程学习策略。
- **PMD与RL裁剪的桥梁作用**：将优化理论中的梯度裁剪与RL中的IS裁剪建立理论联系，为理解现有裁剪机制提供统一视角；类似方法可用于分析或改进其他裁剪变体（如Clip-Higher、GAPO）。
- **熵稳定性的机制解释**：ACPO通过保留小优势
