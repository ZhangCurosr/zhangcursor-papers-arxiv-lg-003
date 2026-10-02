---
title: "WHY-ADAPTIVE-OPTIMIZERS-UNDERESTIMATE-RARE-TOKENS-BIASED-FIX"
source: https://arxiv.org/pdf/2609.37535v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:55:15"
field: "优化器理论与语言模型训练"
keywords: ["adaptive optimizer", "softmax output layer", "rare token", "biased fixed point", "RMSProp", "conservation law", "Adam"]
innovations: ["给出坐标wise自适应优化器在softmax输出层对罕见token产生有偏固定点的闭式刻画与统一参数κ", "建立输出层均值嵌入守恒的分类定理并定量给出Adam/RMSProp等方法的每步漂移公式", "在单语与小型LM上以KL与r_y验证理论预测的偏差符号与量级"]
benchmarks: ["unigram Zipf model", "small Markov-chain language model", "softmax regression conservation test"]
---

# 论文速读：WHY-ADAPTIVE-OPTIMIZERS-UNDERESTIMATE-RARE-TOKENS-BIASED-FIX

## 一句话总结
本文从理论层面揭示了坐标wise自适应优化器（如 Adam、RMSProp）在 softmax 输出层训练时，会因二阶矩估计的动态衰减而对出现频率低于半数 mini-batch 的罕见 token 产生系统性概率低估（有偏固定点），并通过单语模型分析与小型语言模型实验验证了该偏差的规模与规律。

## 研究问题与动机
1. **核心矛盾**：语言模型 token 频率呈强重尾分布，而 softmax 输出层的 cross-entropy 梯度对于非目标罕见 token 为小正值、对目标罕见 token 为大负值；坐标wise自适应优化器（Adam 类）用滑动的二阶矩对梯度进行归一化，导致正向更新被强烈抑制、负向更新被逐步放大，从而在期望上偏离无偏固定点。
2. **现有方法不足**：尽管 Adam 被广泛采用，但其二阶矩估计在罕见 token 出现后达到峰值并随长间隔指数衰减，造成“更新节奏失衡”；此前工作主要关注输出嵌入的整体平移（common shift），却未对单个罕见 token 的固定点偏移给出严格刻画与量化判据。
3. **理论需求**：需要一套区分“保持输出层均值嵌入守恒”与“破坏守恒”的优化器分类框架，并在可控设定下（单语模型、已知生成分布的小 LM）得到闭式或可验证的固定点结果，以指导输出层优化器设计。

## 核心贡献（创新点）
1. **提出输出层均值嵌入的守恒分类定理**：证明所有更新为过去梯度线性组合（SGD、动量）、Kronecker 因子化（Shampoo）、正交化（Muon）及词汇共享缩放（Coupled Adam）的优化器保持均值输出嵌入不变；而坐标wise方法（Adam、RMSProp、AMSGrad、Adafactor 等）的每步均值变化恰好等于负学习率乘以上下文中动量与缩放因子的协方差——这是与既往仅定性描述 Adam 导致 embedding 漂移工作的本质区别。
2. **给出罕见 token 偏差固定点的闭式刻画（RMSProp）**：在单语周期性到达假设下，证明当 token 每隔 $N \ge 3$ 步出现一次时，RMSProp 的稳态概率严格低于数据频率，且其比值 $p^*/q_i = N/x^*$ 仅依赖于 $N$ 与 $\beta_2$；当 $N(1-\beta_2) \to \kappa$ 时比值的极限为 $\rho(\kappa)=\kappa/[2(e^{\kappa/2}-1)]$——这是首次对这类自适应方法的罕见 token 偏差给出可计算的精确表达式。
3. **揭示 sign descent 的持续性负漂移**：在单语模型中证明，凡出现在少于半数 mini-batch 的 token，其 logit 以常数速率 $\eta(1-2\pi_i)$ 被压低，且概率趋于零时压迫力并不减弱，区别于 SGD 中压迫力随 $p_i$ 同步衰减——这一发现解释了极端长尾词在 sign-based 优化下的持续退化机制。
4. **构造无量纲控制参数 $\kappa=(1-\beta_2)/(q_i B)$ 统一刻画偏差强度**：论证 RMSProp/Adam 对罕见 token 的偏差仅由 $\beta_2$ 与 batch size $B$ 通过 $\kappa$ 决定，且 $\kappa>1$ 即进入显著低估区间；这一参数化使不同 $(\beta_2,B)$ 配置下的偏差可比，并为工程调参提供明确阈值依据。
5. **合成与小型语言模型的双重实证验证**：在 softmax 回归、Zipf 单语模型及基于已知一阶马尔可夫链的小型 LM 上，精确复现了定理预测的偏差符号、$\kappa$ 依赖性以及与生成分布 KL 散度的劣化顺序——将纯理论结果与端到端训练指标直接挂钩。

## 方法详解
- **输出层梯度结构**：对 logit $z=W h+b$，cross-entropy 梯度 $\nabla_z \ell=p-t$，满足 $\mathbf{1}^\top(p-t)=0$，因此整个 mini-batch 的梯度矩阵 $G$ 与标量梯度 $g$ 在词表维度求和为零；由此任何形如 $U_t=\sum_{s\le t}\alpha_{t,s}G_s$ 的线性组合更新均保持 $\mathbf{1}^\top U_t=0$，进而保持 $\bar{\mathbf{w}}_t$ 与 $\bar{b}_t$ 不变（命题 1）。
- **坐标wise方法的均值漂移公式**：对形如 $U_{t,ij}=d_{t,ij}M_{t,ij}$ 的更新（$M_t$ 为梯度线性组合），均值输出嵌入的步变化满足 $\bar{\mathbf{w}}_{t+1,j}-\bar{\mathbf{w}}_{t,j}=-\eta_t \mathrm{Cov}_i(d_{t,ij},M_{t,ij})$（命题 2），直接量化了 Adam/RMSProp 等因 $d_{t,ij}$ 与 $M_{t,ij}$ 间正协方差而产生的漂移。
- **单语模型设定**：仅保留输出偏置 $b$，预测 $p(b)=\mathrm{softmax}(b)$，mini-batch 中 token $i$ 的期望出现概率 $\pi_i=1-(1-q_i)^B$；梯度 $g_t=p(b_t)-c_t/B$，其中 $c_t$ 为计数向量。
- **Sign descent 偏差定理**：当 $c_{t,i}=0$ 时 $g_{t,i}=p_i>0$，更新为 $-\eta$；否则更新至多为 $+\eta$，从而 $\mathbb{E}[b_{t+1,i}-b_{t,i}\mid\mathcal{F}_t]\le -\eta(1-2\pi_i)$，对永不出现的 token 退化为精确线性下降（定理 3）。
- **RMSProp 周期性固定点推导**：设 token 每 $N$ 步出现一次， arrival 步梯度 $g_0=-(x-1)p$，其余步 $g_n=p$（$x=a/p$，$a=1/B$）；在 $\epsilon=0$ 下求解 $v_t$ 的 $N$ 周期轨道，得到单周期对数概率变化 $\Delta(x)=-\eta\bigl[\sum_{n=1}^{N-1}\bar{v}_n^{-1/2}-(x-1)\bar{v}_0^{-1/2}\bigr]$，并证明 $\Delta(N)<0$ 且存在唯一稳定零点 $x^*>N$，对应 $p^*=a/x^*<q_i$（定理 4）。
- **无偏方法的对照**：SGD 单周期变化为 $-\eta p(N-x)$，稳定点在 $p=q_i$；AMSGrad 因二阶矩为单调不减最大值，同样回到 $p=q_i$（命题 5）。
- **偏差度量与实验指标**：定义 $\kappa=(1-\beta_2)/(q_i B)$ 作为控制参数；在小型 LM 中使用平均 over 上下文的 KL 散度与 $r_y=\log(\bar{p}_y/f_y)$ 检验理论预测的符号与量级。

## 实验与结果
- **数据集与模型**：① softmax 回归（$V=512$，128 类永不出现，$d=32$，$B=64$）验证守恒律；② Zipf 单语模型（$V=5000$，$q_i\propto i^{-1.2}$）验证 $\kappa$-依赖；③ 基于 2048 token 一阶马尔可夫链的小型 LM（vocab $V=4096$，半数为生成分布外的残留词，embedding 宽 64，MLP 宽 256，untied 输出层），生成分布 $P(y|x)\propto q_y\exp(3u_x^\top u_y)$，训练 $5.2\times10^5$ 对、$B=256$、2 万步。
- **基线对比**：SGD、Heavy-ball、Nesterov、Shampoo、Muon、Coupled Adam、Adam、RMSProp、AMSGrad、Adafactor、Lion、Sign descent、AdamW。
- **守恒实验（Table 1）**：线性/ Kronecker/ 正交化/ 共享缩放方法均将 $\|\Delta\bar{\mathbf{w}}\|_2$ 与 $|\Delta\bar{b}|$ 压制在 $10^{-17}\sim10^{-15}$ 量级；Adam/RMSProp/AMSGrad/Adafactor/Lion/Sign descent 的均值漂移达 $0.1\sim 1.4$，验证命题 1。
- **单 token 与单语模型（Figure 1, Section 5.2）**：周期性到达下 RMSProp 的固定点与定理 4 的解析解吻合（$\log p$ 最大偏差仅 0.009）；随机到达时 $\kappa\lesssim1$ 仍贴近周期预测，$\kappa\ge 2$ 时偏差显著超出（均值偏差 −5.88），且 $(\beta_2,B)$ 配对不再由 $\kappa$ 单一决定。Zipf 单语模型中，Adam/RMSProp 在 $\kappa>1$ 区呈现 $\log(p_i/q_i)<0$ 且随 $\kappa$ 单调恶化；sign descent 下 99% 的 $\pi_i<1/2$ token 终态 $\log(p_i/q_i)<-5$。
- **小型语言模型（Table 2, Figure 2）**：$\beta_2=0.95$ 时，Adam/RMSProp/AdamW 对罕见 token（$e_y<0.05$）产生严重低估，$\bar{r}_y$ 分别为 $-2.72$、$-2.65$、$-2.25$；提升至 $\beta_2=0.999$ 后偏差降至 $-0.06$。AMSGrad/Coupled Adam/SGD 的 $\bar{r}_y$ 为 $-0.14/-0.18/-0.22$（后两者主要反映未收敛）。KL 与 test CE 排序与理论一致：SGD 最低（KL=0.0940，CE=3.467），AMSGrad/Coupled Adam 次之（KL≈0.124），$\beta_2=0.95$ 的 Adam/RMSProp 最差（KL≈0.22，CE≈3.60）。mean logit 在 Adam/RMSProp 下分别降至 $-46.83$/$-53.42$，而 SGD/Coupled Adam 保持在 0 附近。
- **最强结果与提升**：在偏差控制方面，Coupled Adam（KL=0.1238，$\bar{r}_{e_y<0.05}=-0.18$）与 AMSGrad（KL=0.1241，$\bar{r}=-0.14$）最接近无偏 SGD；提高 $\beta_2$ 到 0.999 可将 Adam 的罕见 token 偏差从 −2.72 降至 −0.06，KL 从 0.2278 降至 0.1586。

## 相关工作脉络
1. **Kunin et al. (2021)** 建立梯度流下的守恒律框架（平移/缩放对称），本文在此基础上精确刻画了 softmax 平移对称性在离散坐标wise更新下的破坏机制，并给出每步变化的协方差等式（命题 2）。
2. **Stollenwerk & Stollenwerk (2025)** 将 Adam 导致的 embedding 退化归因于二阶矩，提出 Coupled Adam 通过词汇共享缩放恢复守恒；本文确认 Coupled Adam 属于守恒类，同时进一步揭示即使守恒恢复后，单 token 固定点仍可能受 $\beta_1$ 等非零动量影响（单 token 实验中 Adam $\beta_1=0.9$ 偏差极小）。
3. **Reddi et al. (2018)** 构造一例“大梯度后跟小反向梯度”的一维场景说明 Adam 收敛至错误点，并提出 AMSGrad；本文把罕见 token 的梯度时序模式等同为该构造的实现，并以闭式给出 RMSProp 的固定点位置，补全了原工作只证明“存在偏差”而未定量刻画的部分。
4. **Wortsman et al. (2024); Stollenwerk et al. (2026)** 将输出嵌入漂移与输出 logit 发散、z-loss 稳定性相联系；本文强调二者属不同效应：公共平移不影响预测，而单 token 固定点偏移直接改变预测分布，故 centering 无法消除本文所揭示的偏差。
5. **Kunstner et al. (2024)** 发现重尾类别不平衡下 GD 在稀有类上进展慢而 sign descent 不受此限；本文结论与之互补——sign descent 虽无“进展慢”问题，却带来更强烈的常数速率 logit 下降，揭示自适应规范化“快进展”背后的代价。
6. **Menon et al. (2021)** 提出 logit adjustment 显式校正稀有类；本文从优化器动力学角度补充了另一条产生概率低估的独立机制，说明即便数据与标签对齐，选择 RMSProp/Adam 仍会引入额外校准偏差。

## 局限性与未来方向
- 理论推导依赖单偏置坐标、周期性到达、$\beta_1=0$、$\epsilon=0$ 及 log-partition 恒定等强简化（Section 7）；实验仅在合成与小型 LM 上验证，未扩展至大规模预训练 setting。
- 大型语言模型中，百万级 token 每 batch 下 $\kappa>1$ 仅对极稀有 token 成立，偏差是否可测量、对下游稀有词生成影响几何仍未知。
- 细调与小 batch 场景（$B$ 小、$\beta_2=0.95$）下大量词汇可满足 $\kappa>1$，需实证检验工程收益。
- 未来方向：将分析推广至 tied embedding、$\beta_1>0$ 随机到达、含 context 依赖的真实 LM 输出层；探索 $\epsilon>0$、weight decay 耦合、以及输出层单独使用 SGD/Coupled Adam 的工程策略。

## 研究启发与可借鉴点
1. **$\kappa$ 作为通用诊断指标**：$(1-\beta_2)/(q_i B)$ 可快速估算任意 token 在 RMSProp/Adam 下的理论偏差幅度，建议将其纳入团队模型监控指标，用于识别高风险稀有词段。
2. **保守优化器用于输出层**：若团队需维持输出分布保真度，可尝试对输出层使用 SGD/Coupled Adam/Shampoo/Muon，保留隐藏层的自适应加速，兼顾收敛效率与分布保真。
3. **实验设计借鉴**：使用“已知生成分布的小 LM"以 KL 散度作为理论预测的验证基准，避免了大规模实验的不确定性，该方法可直接迁移至其他优化器理论验证场景。
4. **偏差缓解清单**：提高 $\beta_2$、增大 $B$、改用 AMSGrad/Coupled Adam、对输出 bias 使用 SGD、或对二阶矩设定合理下界 $\epsilon$，均为低成本可行的工程缓解手段。
5. **符号优化器的双重审视**：sign descent 虽保持“常数压迫力”，却带来无下限的 logit 漂移；团队在探索轻量优化器时应同时评估其固定点性质而非仅看收敛速度。

## 关键术语表
- **softmax 输出层**：将网络隐状态映射为词表上 logits $z=Wh+b$ 并通过 softmax 得到概率分布的输出层。
- **坐标wise自适应优化器**：对每个参数坐标独立地除以该坐标历史梯度的滑动 RMS 的方法，典型代表为 Adam、RMSProp。
- **偏差固定点（biased fixed point）**：优化器迭代稳定后概率分布与数据真实频率不一致的稳态，本文指罕见 token 的稳态概率严格低于其数据频率。
- **二阶矩估计 $v_t$**：梯度平方的指数移动平均，在 Adam 族中用于缩放学习率；对罕见 token 在出现后达到峰值并随间隔衰减。
- **均值输出嵌入 $\bar{\mathbf{w}}$**：输出权重矩阵按词表维度的均值向量，其守恒意味着所有 token 的 logit 整体平移而不改变预测分布。
- **守恒律（conservation law）**：由 softmax 平移对称性导出的性质，要求优化器更新在所有词表维度上求和为零。
- **参数 $\kappa$**：$\kappa=(1-\beta_2)/(q_i B)$，衡量 token 两次出现平均间隔与二阶矩时间常数 $1/(1-\beta_2)$ 的比值，是控制偏差强度的关键无量纲量。
- **Coupled Adam**：将 Adam 的二阶矩在词表维度上进行平均后再用于缩放，从而恢复均值输出嵌入守恒的变体。

## 可复现要素
- **代码**：作者提供 `run experiments.py` 作为补充材料，所有实验在 CPU 上两小时内完成（论文明确声明代码已开源随文提供）。
- **数据集**：合成单语模型（$V=512$、$V=5000$ Zipf）、基于 2048 token 一阶马尔可夫链生成的训练/测试对；未使用公开外部数据集。
- **关键超参**：守恒实验 $B=64$、$300$ 步、float64；单语模型 $\eta=4\times10^{-3}$、$\epsilon=10^{-12}$、$3\times10^5$ 步；小 LM $B=256$、warmup 200 步、总步数 $2\times10^4$、自适应方法 $\eta=3\times10^{-3}$、$\beta_1=0.9$（RMSProp 为 0）、$\epsilon=10^{-8}$。
- **权重开源情况**：论文未提供公开模型权重；仅给出实验脚本与理论推导细节。
