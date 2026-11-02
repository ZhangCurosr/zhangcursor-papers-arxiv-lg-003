---
title: "textGrain-watermarking-is-harder-to-detect-can-sum-tests-sti"
source: https://arxiv.org/pdf/2610.17655v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:47:16"
field: "水印检测的理论分析"
keywords: ["LLM watermarking", "textGrain", "detection boundary", "entropy calibration", "sparse watermark", "adaptive test", "Berk-Jones scan", "Chernoff-Stein efficiency"]
innovations: ["证明textGrain检测边界严格严于Gumbel-max/最大耦合（p+q=1/2 vs 2p+q=1）", "构造RG反例表明熵校准本身不必然导致更难检测，成本源于具体构造", "提出无需NTP/熵预算/编辑比例的自适应加权多尺度二项式扫描，同时达到所有检测边界与最优指数效率"]
benchmarks: ["Phase diagram analysis for fixed/growing columns with strong/weak entropy budgets", "Minimax detection boundary characterization", "Exponential efficiency comparison against sum test"]
---

# 论文速读：textGrain-watermarking-is-harder-to-detect-can-sum-tests-sti

## 一句话总结
本文理论分析了 **textGrain** 水印在人工编辑（稀疏混合）后的可检测性，证明其检测边界比 Gumbel-max/最大耦合水印更严格（成本源于构造而非熵校准本身）；同时指出官方提出的**和检验**在固定列数时边界最优，但在列数增长或固定编辑比例时次优；最终提出一种**无需知悉 NTP、熵预算、编辑比例的加权多尺度二项式扫描测试**，在所有检测边界与指数效率上均达到最优。

## 研究问题与动机
- **熵校准是否使检测更困难？** textGrain 通过显式熵预算平衡水印信号与采样熵，但保留额外随机性可能削弱密钥对 token 的影响力，从而弱化统计证据。
- **和检验是否在所有可检测情形下均最优？** Li et al. (2026b) 提出的统计量 $T_n^{\text{textGrain}}$ 在固定列数下是否仍保持理论最优性？
- **固定编辑比例下的检测效率如何？** 能否以最优指数速率衰减第二类错误？是否存在一种无需额外知识的通用测试同时达到边界最优与效率最优？
- **现有基线局限：** Gumbel-max 与最大耦合水印的检测边界为 $2p+q=1$，textGrain 的相图尚未系统刻画，且和检验在高维/稀疏信号集中情形下已知可能失效（类比 Donoho–Jin 更高批评理论）。

## 核心贡献（创新点）
1. **确立 textGrain 的严格更难检测性：** 证明固定列数且熵预算有界 away from zero 时，最小极大检测边界为 $p+q=1/2$（或带 $\beta_n\asymp n^{-r}$ 时为 $2p+2q+r=1$），严格包含于 Gumbel-max/最大耦合的 $2p+q=1$ 区域。
2. **揭示“更难检测”源于构造而非熵校准本身：** 构造了一个无偏且熵保持的玩具方案 RG，在相同参数下仍可检测，说明 textGrain 的边界收缩来自其具体构建方式。
3. **刻画和检验的双重角色：** 固定列数时和检验在检测边界上**一致且达到边界**；但当列数 $m_n$ 增长且熵预算充分大时，其敏感度被 $\varepsilon_n\Delta_n\log(e/\Delta_n)$ 锁定，在区域 $p+q>1/2$ 内失效力，而更优测试仍可检测。
4. **提出自适应加权多尺度二项式扫描测试（Algorithm 1）：** 在有限样本下严格有效（level-α），无需 NTP、熵预算、编辑比例或耦合信息；同时达到所有已证检测边界与最优指数衰减率。

## 方法详解
- **假设体系：**
  - **Assumption 1**：NTP $P_t$、熵预算 $\beta_t$ 关于过去 filtrations 可测；分区 $g_t$、列 $J_t$、伪随机变量 $Z_{bj}$ 每步独立抽取。
  - **Assumption 2**：分区保留 token 不确定性——任意不同 token 落入不同 block 的概率至少 $\vartheta>0$。
- **信号强度刻画（Theorem 2.2）：** 对任意固定单调得分 $h$，$\chi^2(f_{1,P,\beta}\|U[0,1])\asymp_{m,\vartheta}\Delta(P)\min\{\Delta(P),\beta H(P)\}$；均值偏移量级为 $\sqrt{\Delta(P)\min\{\Delta(P),\beta H(P)\}}$。相比 Gumbel-max（偏移 $\asymp H(P)$），textGrain 损失对数因子。
- **检测边界（Theorem 2.4 / Corollary 2.5）：** 定义最小极大风险 $R_n^Y$。当 $n\varepsilon_n^2\Delta_n\min\{\Delta_n,\beta_n\mathbb{H}(\Delta_n)\}\to\infty$ 时可检测；多项式缩放 $\varepsilon_n\asymp n^{-p},\Delta_n\asymp n^{-q},\beta_n\asymp n^{-r}$ 下，边界为 $2p+2q+r=1$（固定 $\beta\gtrsim1$ 时退化为 $p+q=1/2$）。
- **列数增长的强/弱熵预算（Theorem 2.8 / 2.10）：**
  - **强预算**（$\beta_t\ge\beta_0>0$）：边界扩展为 $2p+2q-\min(s,q)=1$（$m_n\asymp n^s$），信息量增至 $\Delta_n\min\{m_n\Delta_n,1\}$。
  - **弱预算**（$\kappa_n=\beta_n H(P_t)\to0$ 且 $\kappa_n/\Delta_n\to0$）：信息量保持 $\Delta_n\kappa_n$，边界与固定列数相同，且不依赖 $s$。
- **和检验的优劣（Theorem 3.1 / Proposition 3.4）：**
  - 固定列数：$T_n^{\text{textGrain}}$ 在 $n\varepsilon_n^2\Delta_n\min\{\Delta_n,\beta_n\mathbb{H}(\Delta_n)\}/\log\log n\to\infty$ 下一致。
  - 增长列数+强预算：当 $p+q>1/2$ 时，对任意阈值序列 $c_n$，$T_n^{\text{textGrain}}$ 的总误差趋于 1，即使该区域在理论边界内可检测。
- **最优指数效率（Theorem 4.2 / Proposition 4.4）：**
  - 固定编辑分数 $\varepsilon$，最优 Chernoff–Stein 指数为 $\inf_{q\in\mathcal{Q}_\Theta} D_{\mathrm{KL}}(\mathrm{Unif}\|q)$。
  - 和检验的指数严格小于该最优值（对所有非均匀替代成立）。
- **自适应测试（Algorithm 1 / Theorem 4.5–4.9）：**
  - 网格 $\mathcal{U}_n$ 包含二进级别与均匀细网格；权重 $w_n(u)$ 将半数质量留给中位数 $u=1/2$。
  - 统计量 $T_n=\max_{u\in\mathcal{U}_n}\{n\operatorname{kl}(\widehat{F}_n^p(u),u)+\log w_n(u)\}$ 为加权单边 Berk–Jones 型扫描。
  - 有限样本 level-α 保证（Bonferroni 型加权和）；在固定列数、增长列数强/弱预算、固定 $\varepsilon$ 三类场景下均达到理论边界与最优指数。

## 实验与结果
- **无传统数据集/基线实验：** 本文为纯理论分析，通过相图（Figure 1、2）与定理比较不同水印方案的检测区域。
- **主要数值结论：**
  - **固定列数边界：** textGrain $p+q=1/2$；Gumbel-max / 最大耦合 $2p+q=1$。前者区域严格更小。
  - **增长列数边界：** textGrain $2p+2q-\min(s,q)=1$；Gumbel-max 不变。
  - **和检验失效率高发区：** $p+q>1/2$ 但 $2p+2q-\min(s,q)<1$ 的带状区域（见 Figure 2 紫–薰衣草色阴影）。
- **最强结果：** Algorithm 1 在**所有**已证检测边界与指数效率意义上同时最优，且完全无需参数先验。

## 相关工作脉络
1. **Gumbel-max 水印（Li et al., 2026a）：** 边界 $2p+q=1$，和检验次优（需更复杂测试）。本文与之对比，显示 textGrain 因熵校准进一步压缩检测区域，但反常地使和检验在固定列数下“刚好”达到边界。
2. **最大耦合去偏 Red-Green 水印（Xie et al., 2025）：** 边界同为 $2p+q=1$，和检验边界仅 $p+q=1/2$。本文沿用其稀疏检测框架，但引入额外参数 $\beta,m$ 重塑边界。
3. **稀疏异构混合检测（Donoho & Jin, 2004, 2015）：** Fisher 组合检验在极稀疏 regime 失效，更高批评与 $\phi$-散度统计自适应可达边界。本文测试为其在 watermarking 领域的加权单边变体。
4. **Berk–Jones  goodness-of-fit 测试（Jager & Wellner, 2007）：** 多尺度扫描思路的统计基础，本文将其适配为 p 值下尾扫描并嵌入水印 pivot 框架。
5. **逆信息投影与 Chernoff–Stein 效率（Larsson et al., 2025）：** 固定编辑比例下的指数效率分析范式，本文照搬并证明自适应测试可达逆信息投影指数。
6. **文本水印工程实践（Li et al., 2026b; Hu et al., 2024; Wu et al., 2024）：** textGrain 作为最新工程方案，本文首次为其提供严格统计可检测性刻画与最优检测器。

## 局限性与未来方向
- **重复上下文场景未覆盖：** 理论假设每步使用新鲜密钥与独立随机性，而 textGrain 官方实现仅对每个 context window 首次出现打分，存在依赖结构，当前渐近结论不直接适用。
- **弱熵预算的列数无关性：** 当 $\kappa_n\ll\Delta_n$ 时，增长列数不再提升可检测性，实践中若预算受限则无需增大 $m$。
- **计算复杂度：** Algorithm 1 需 $O(n\log n)$ 排序与网格扫描，虽可行但高于简单求和；在极长文本流式场景下或需近似。
- **实际编辑模型简化：** 稀疏混合假设假设被编辑位置完全移除密钥依赖，未考虑部分保留或相关编辑。

## 研究启发与可借鉴点
1. **构造反例分离“熵校准”与“检测难度”：** 通过 Example 1 的 RG 方案证明保持无偏与熵校准并不必然损害可检测性，提示在分析新水印方案时应谨慎归因。
2. **最小极大风险 + χ² 散度刻画信号强度：** 将 watermark detection 转化为混合密度检验，利用 $\chi^2$ 散度分解信号与噪声，是分析稀疏水印边界的标准有力工具。
3. **自适应多尺度扫描替代固定得分求和：** 当信号可能集中於尾部或散布於全区间时，权重化 Berk–Jones 型扫描可同时捕获两种模式，且无需知道稀疏度/熵预算。
4. **指数效率与边界最优性的分离分析：** 区分“何时可检测”（边界）与“检测多快”（指数），对评估实际部署中检测器鲁棒性具有指导意义。
5. **与高效采样/去偏设计的交叉：** 若后续 work 希望兼顾低失真与高可检性，可参考 textGrain 的 entropic OT 耦合框架，但需注意其引入的边界收缩代价。

## 关键术语表
- **textGrain：** 结合 Red-Green 随机性与 Gumbel-max 无偏性、通过 entropic optimal transport 耦合显式熵预算的水印方案。
- **Gumbel-max watermark：** 由密钥伪随机变量完全决定 next-token 的无偏水印，确定性给定 prompt 与 key。
- **Maximal coupling watermark：** 通过最大耦合去偏 Red-Green 水印，保留 block 结构同时消除分布偏差。
- **检测边界（Detection boundary）：** 在稀疏水印机制 $\varepsilon_n\asymp n^{-p}$、token 不确定度 $\Delta_n\asymp n^{-q}$ 参数化下，假设检验错误率可趋于零的临界曲线。
- **稀疏水印机制（Sparse watermark regime）：** 编辑后仅分数 $\varepsilon_n\to0$ 的位置保留密钥依赖，其余为均匀 pivot 的混合模型。
- **加权多尺度二项式扫描（Weighted multiscale binomial scan）：** 本文提出的自适应测试，在二进与均匀网格上扫描 p 值下尾比例，以 KL 散度计分并加权惩罚。
- **Chernoff–Stein 指数：** 固定显著性水平下，最优检测器第二类错误概率衰减的对数速率。
- **Berk–Jones statistic：** 多阈值 goodness-of-fit 检验统计量，本文取其单边加权版本用于水印检测。

## 可复现要素
- **数据集：** 论文未使用实证数据集，为纯理论分析。
- **代码/权重：** 未开源；Algorithm 1 为确定性扫描过程，可依公式实现。
- **关键超参：** 列数 $m$（固定或 $m_n\asymp n^s$）、熵预算 $\beta_n$（有界 away from zero 或 $\beta_n\asymp n^{-r}$）、分区概率下限 $\vartheta$、网格规模 $L_n=\lceil\log(n+1)\rceil$。
- **公开声明：** 论文未提及数据与代码开源计划；详细证明将在 forthcoming extended version 提供。
