---
title: "THE-LATTICE-OF-TRANSITION-LAWS"
source: https://arxiv.org/pdf/2610.11216v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:10:37"
field: "生成模型调度优化"
keywords: ["diffusion model", "autoregressive model", "decoding schedule", "corruption lattice", "dependence cost", "treedepth", "parallel generation", "generative modeling"]
innovations: ["提出污染格统一框架将扩散与自回归调度表示为同一空间中的路径", "定义依赖代价并证明等于路径与精确采样的KL散度，推导零代价步数下界为图树深", "从预训练权重估计成对依赖核，在解码前预测不同调度的排名"]
benchmarks: ["text8 bpc", "ImageNet-256 FID-50K", "GSM8K accuracy", "HumanEval pass@1", "VBench video quality"]
---

# 论文速读：THE-LATTICE-OF-TRANSITION-LAWS

## 一句话总结
本文提出了"污染格（corruption lattice）"框架，将扩散模型与自回归模型的解码调度统一为同一格空间中的路径，并定义了"依赖代价"来量化并行步的依赖性损失，从而在解码前即可预测不同调度方案的优劣。

## 研究问题与动机
1. **扩散与自回归被视为两种独立的生成范式**：扩散适用于连续场，自回归适用于离散token；尽管近年出现混合模型，但其解码调度方案在设计时即被固定。
2. **同一组权重下不同调度方案的性能差异巨大**：例如MAR模型在相同步数预算下，不同揭示顺序的图像质量可相差一个数量级（Table 3），但目前缺乏可在解码前评估调度的设计原则。
3. **现有统一方法仅描述调度路径，无法量化比较**：如mask diffusion的训练目标与任意顺序自回归等价（Ou et al., 2025）、每位置噪声调度表达自回归（Fathi et al., 2025），但均未提供比较不同调度的量化成本。
4. **缺乏前置预测能力**：候选调度方案只能在使用后评价，无法在解码前判断哪种方案更优。

## 核心贡献（创新点）
1. **提出污染格统一框架**：将扩散、自回归及其间所有调度方案表示为同一格空间 $L^d$ 中的单调路径，证明同一组权重可遍历全格。
2. **定义依赖代价并建立精确等式**：提出并行步的总相关代价（TC），证明其等于该路径与精确采样的KL散度，涵盖离散token与连续值、双层格与内层格。
3. **推导零代价调度的步数下界为图树深**：对于图上的Markov数据，零代价调度的最少步数等于图的树深（treedepth），序列长度为 $O(\log d)$，网格边长为 $O(n)$。
4. **提出解码前的调度排名预测方法**：从预训练权重估计成对依赖核（pairwise dependence kernel），利用引理2的成对下界在解码前预测不同调度的优劣排序，并在文本、图像、视频三个模态上验证。

## 方法详解
1. **污染格定义（Definition 1）**：每个坐标 $i$ 携带独立级别 $\ell_i \in L$（从干净到完全污染），构成偏序格 $L^d$。解码调度是从全污染状态 $\top$ 到全干净状态 $\bot$ 的单调路径，每步更新一组坐标 $S = \{i : \ell'_i < \ell_i\}$。
2. **依赖代价公式（Proposition 1, Eq. 3）**：对路径 $\pi$，每步的依赖代价为 $\text{TC}(S_t | \ell^{(t)}) = \sum_{i \in S_t} H(x_i|\ell) - H(x_{S_t}|\ell)$（条件总相关），路径总代价为各步TC之和，等于精确联合采样路径与坐标独立采样路径之间的KL散度。
3. **零代价调度与树深（Theorem 1）**：若剩余未解析坐标在图 $G$ 上满足Markov性，则零代价调度每步从每个连通分量中选取至多一个坐标，最少步数等于 $G$ 的树深 $\text{td}(G)$。链图上中点规则（midpoint rule）达到 $\lceil \log_2(d+1) \rceil$ 步，$n \times n$ 网格的树深介于 $n+1$ 与 $O(n)$ 之间。
4. **成对下界与核估计（Lemma 2, Corollary 1）**：$\text{TC}(S|\ell) \geq \sum_{j=2}^{k} I(x_{i_j}; x_{i_{j-1}} | \ell)$，等号在一步内各坐标关于前驱条件独立时成立（一阶Markov参考）。核 $\kappa(i,j|\ell) = I(x_i; x_j|\ell)$ 从预训练模型估计，按坐标距离和掩码率建模，用于解码前预测调度排序。
5. **模型可穿越全格（Section 2.2 + Figure 2）**：在同一组权重下训练于格上所有状态分布的模型，可在不同调度路径下解码，文本质检bpc仅比专用mask模型高0.06 bits，展示了单一权重的跨调度泛化能力。

## 实验与结果
- **数据集与模型**：text8（自建85M/26M语言模型）、ImageNet-256（MAR-B扩散自回归图像模型）、LLaDA-8B（大型语言扩散模型）、SkyReels-V2视频模型（Diffusion Forcing家族）。
- **text8结果（Table 3）**：8步时spread规则（1.65 bpc）显著优于contiguous（3.14 bpc）；64步时所有规则趋近（1.32~1.58 bpc），此时依赖代价消失，排名由条件误差决定。
- **MAR-B结果（Table 3）**：8步时spread规则FID=9.44，相对模型自身random顺序（13.02）降低超25%；low-discrepancy规则以8.32为最优；256步时random规则复现发表分数3.93。
- **LLaDA-8B结果（Table 4）**：min-distance约束相对plain confidence在GSM8K上提升最大+9.7±0.8pp（8步），perplexity降低1.04 nats（8步），效果随步数增加而消退。
- **视频结果（Table 28）**：不同memory depth（5~57帧）在各VBench维度上与 shipped 设置差异均在2个配对标准差内，验证核预测的正确趋势（更大深度不提升质量）。
- **核验证**：text8内核显示词内依赖是跨词界的2~4倍（距离≤4），距离≥16时近零；MAR网格中位移核在Chebyshev距离≥8时降至最近邻值的<5%（Figure 4）。
- **最强结果**：MAR-B上low-discrepancy规则8步FID=8.32；text8上spread规则8步bpc=1.65；GSM8K上min-distance规则8步准确率+9.7pp。

## 相关工作脉络
1. **Masked diffusion 与任意顺序自回归的等价性（Ou et al., 2025）**：指出mask diffusion训练目标等同于any-order AR，但未量化不同调度路径的依赖代价。
2. **超调度统一框架（Fathi et al., 2025）**：为每个位置分配单调噪声调度，将AR表达为扩散特例，但未提供解码前的调度比较机制。
3. **Block Diffusion（Arriola et al., 2025）**：块内用扩散、块间用AR的混合，调度方案固定；本文框架可覆盖此类设计但提供额外代价分析。
4. **Dilated scheduling（Luxembourg et al., 2026）**：用固定模式解掩非相邻位置以降低熵增，基于距离假设；本文进一步指出"分离"（separation）在Markov图上的精确零代价优势。
5. **Confidence-based parallel decoding（MaskGIT, Chang et al., 2022; Sahoo et al., 2024）**：基于置信度选择坐标；本文通过核预测其代价偏高，并证明约束最小距离可显著改善。
6. **Diffusion Forcing（Chen et al., 2024; 2025）**：逐帧独立噪声级别的视频生成，本文将其映射到污染格框架并验证了memory depth调度的预测。

## 局限性与未来方向
1. **条件误差未被纳入代价模型**：依赖代价仅度量并行采样丢弃的依赖性，训练条件估计的误差不在框架内，当依赖代价相近时可逆转预测（如MAR上nested规则）。
2. **成对下界在非一阶Markov数据上不够紧**：Lemma 2的等号仅在一步内各坐标关于前驱条件独立时成立，实际数据存在高阶依赖，导致估计低估真实代价（如text8上2步时估计仅为实际值的30%~50%）。
3. **内层级的增益-代价权衡复杂**：Section 6.5显示graded channel在8步时因步宽增大而劣于absorbing channel，但在≥16步时两者几乎持平，内层策略需按步数预算精细设计。
4. **计算核的开销虽低但依赖模型前向**：每次估计核需对每个坐标做前向传播求联合熵（Section 6.1），对超大模型可能有扩展性问题。
5. **未来方向**：将框架扩展到更多架构类型、开发高阶依赖核、研究训练-调度联合优化、探索内层级的最优设计。

## 研究启发与可借鉴点
1. **统一框架的迁移价值**：污染格+依赖代价的组合思路可推广到其他混合生成模型（如流模型、状态空间模型），为调度设计提供通用度量基准。
2. **核估计的实用技巧**：成对依赖核从预训练权重估计后用于解码前预测，这一"离线估计、在线应用"模式可迁移到任何需要调度优化的并行解码场景。
3. **树深作为理论下界的启示**：将图论概念（treedepth）引入生成模型调度分析，为理解"最少步数"提供了严格的数学基础，值得在序列长度/图像分辨率扩展性分析中借鉴。
4. **约束式选择的设计范式**：min-distance和separator constraint在保留模型置信度排序的同时排除强依赖对，在LLaDA上显著改善了perplexity和任务准确率，是一种轻量有效的调度优化策略。
5. **实验设计的对比方式**：固定权重只改变调度规则（而非重新训练）的评测方式，直接量化了调度本身的贡献，值得在模型评估中广泛采用。

## 关键术语表
**Corruption Lattice（污染格）**：$L^d$ 偏序格，每个坐标独立携带污染级别，所有解码调度为其上从全污染到全干净的单调路径。
**Dependence Cost（依赖代价）**：并行步中因独立采样各坐标而非联合采样所丢弃的条件总相关，等于该步路径与精确采样的KL散度。
**Treedepth（树深）**：图 $G$ 的最小消除树深度，本文证明为零代价调度所需的最少步数下界。
**Pairwise Dependence Kernel（成对依赖核）**：$\kappa(i,j|\ell) = I(x_i; x_j|\ell)$，衡量两个坐标在给定状态下的条件互信息，从预训练模型估计。
**Separator Schedule（分离调度）**：每步从每个未解析连通分量中选取一个坐标，在Markov图上实现零依赖代价。
**Midpoint Rule（中点规则）**：每步选取每个掩码区间的中点坐标，在链图上达到树深步数。
**Total Correlation（总相关）**：多变量分布与其边际分布乘积之间的KL散度，衡量变量间的联合依赖性。
**Absorbing vs. Graded Channel（吸收vs分级通道）**：双层格（掩码/干净）与多层格（含内层信息）的对比，内层可降低单步代价但增加步数。

## 可复现要素
- **数据集**：text8（公开）、CIFAR-10（公开）、ImageNet-256（公开）、GSM8K（公开）、HumanEval（公开）、VBench（公开基准）
- **代码**：已开源，见 https://github.com/TSUITUENYUE/The-Lattice-of-Transition-Laws
- **权重**：自建text8模型（85M/26M）、使用发布的MAR-B、LLaDA-8B-Base、SkyReels-V2权重
- **关键超参**：text8模型85M参数、25K/165K/185K/510K训练步数；MAR-B 16×16 token网格、cosine调度；LLaDA-8B 256位置块；视频模型8倍训练horizon rollout
- **评估协议**：text8 teacher-forced后192字符前缀评分最后64字符；MAR FID-50K；GSM8K 5-shot greedy；HumanEval greedy pass@1
