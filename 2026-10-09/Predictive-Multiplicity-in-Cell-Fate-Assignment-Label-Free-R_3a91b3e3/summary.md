---
title: "Predictive-Multiplicity-in-Cell-Fate-Assignment-Label-Free-R"
source: https://arxiv.org/pdf/2610.11185v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:31:45"
field: "单细胞轨迹推断与模型可重复性"
keywords: ["single-cell trajectory inference", "predictive multiplicity", "Rashomon set", "cell fate assignment", "label-free certification", "absorbing random walk", "Palantir", "multiverse analysis"]
innovations: ["无标签Rashomon集构建：用cross-fitted discrepancy+seed-calibrated non-inferiority test替代supervised loss定义admissible set", "Negative result：per-cell infimum FM劣于单模型margin（AUC 0.682 vs 0.965），失败源于aggregator极端顺序统计量退化", "Certification power Π度量：区分真实模型空间分歧与种子噪声floor，揭示跨研究multiplicity率不可直接比较"]
benchmarks: ["GSE162610小鼠脊髓损伤microglia", "GSE72857髓系祖细胞erythroid/myeloid bifurcation", "Weinreb et al.谱系条形码造血时间序列", "GSE99915成纤维细胞重编程", "75-object synthetic single-cell trajectory cohort"]
---

# 论文速读：Predictive-Multiplicity-in-Cell-Fate-Assignment-Label-Free-Rashomon-Sets-and-the-Limits-of-Per-Cell-Certification

## 一句话总结
本文提出 FateMultiplicity 框架，在无Ground-truth谱系标签的情况下，通过交叉拟合保留基因的非劣效性检验构建Rashomon集合，量化单细胞轨迹推断中的预测多重性；但其核心negative result是：以 admitted 集合下确界（infimum）定义的 per-cell certified fate margin（FM）**不如**单个拟合模型的决策 margin，因为 infimum 是极端顺序统计量，随集合拓宽退化为"最不极端模型"的信号。

## 研究问题与动机
- **核心问题**：单细胞轨迹推断依赖复杂多步流水线（降维→近邻图→伪时间→随机游走），每步均有分析师可调超参；不同配置在统计上等价却可能对单细胞 fate assignment 给出冲突结果——即 Rashomon effect / predictive multiplicity。现有分类领域的工作依赖显式 labeled loss 定义 admissible set，但单细胞轨迹推断缺乏 ground-truth fate label，无法直接移植。
- **方法空白**：无监督轨迹推断既无 labeled loss，也无 ϵ 尺度，故无法用经典容差球 {θ: L(θ)≤L(θ*)+ϵ} 定义 Rashomon set。
- **动机**：若能在无标签场景下构建统计可接受的模型集合并度量 multiplicity，可为分析可重复性提供定量保障；进一步探索能否构造 per-cell 可靠性统计量（FM）替代单模型点估计。

## 核心贡献（创新点）
1. **无标签 Rashomon 集构建**：用 cross-fitted held-out gene discrepancy（von Neumann ratio）替换 supervised loss，用 seed-calibrated non-inferiority test 确定接纳边界，首次在无监督轨迹推断中形式化定义 Rashomon set。
2. **Certification power 度量 Π**：提出 Π = d̄_space / d̄_seed，区分"模型空间真实分歧"与"种子噪声 floor"，提示不同数据集/算法间的 multiplicity 率不可直接比较。
3. **Negative result：FM 不优于单模型 margin**：在仿真 ground truth 上，FM 判别 misassignment 的 AUC=0.682，显著劣于基线配置自身 margin（AUC=0.965, p=0.003）及 seed dispersion baseline（AUC=0.854）。
4. **机制定位**：证明失败根源在 aggregator（infimum 是 unanchored extreme order statistic），而非被聚合量本身；supremum 反因 θ* 自身有下界而保留信号（AUC=0.973）。
5. **实用发现**：multiplicity 率跨数据集稳定（3.8% vs 3.9%），但置信度分散跨流形差异可达 483 倍；急性脊髓损伤期 multiplicity 最高（10.4%）；margin-erosion ratio 在仿真中区分真/伪分支点达 AUC=0.890。

## 方法详解
- **Cross-fitted discrepancy（核心公式）**：对 K=3 折交叉拟合，每个候选 θ 在训练基因 G₁⁽ᵏ⁾ 上拟合，在保留基因 G₂⁽ᵏ⁾ 上评分：
  - d_g(θ) = (1/K) Σₖ dev_g(θ; G₂⁽ᵏ⁾)
  - dev_g 为 von Neumann ratio：Σ(x_{i+1}-x_i)² / [2(n-1)·Var(x_g)]，其期望在随机排序下为 1，仅依赖 rank order，对伪时间单调变换不变。
- **Non-inferiority test 与 Rashomon 集**：
  - Δ_g(θ) = d_g(θ) - d_g(θ*)，seed-calibrated margin δ = 1.086×10⁻³（来自 θ* 四次随机种子重拟合的 D 变异）
  - R(α) = {θ well-formed : T(Δ(θ)) ≤ q_{1-α}}，T 为 48 模块 block-permuted 统计量，B-H 校正 α=0.05。
- **Well-formedness filter**：排除 >50% 细胞具有 uniform posterior fate 的退化配置（避免"ordering 正确但 fate 坍缩"的情况）。
- **Module blocking**：保留基因按 1-|Spearman ρ| 平均连接聚类为 15–17 模块/fold（共 48 块），防止共表达程序虚增有效样本量。
- **Label alignment**：用 argmax 类别一致性最大化对齐候选配置的任意聚类 fate label 至基线，保守估计 multiplicity。
- **Certified fate margin（FM）**：FM_i(α) = inf_{θ∈R(α)} [p_{i,k*}(θ) - max_{k≠k*} p_{i,k}(θ)]，FM_i>0 为 certified。
- **q-quantile 族**：FM_i(q) = Quantile_q{ m_i(θ) : θ∈R(α) }，用于定位 worst-case 在何处失效。
- **Margin erosion ratio**：certified margin / baseline mean margin，在仿真中区分真分支与伪分支。

## 实验与结果
- **数据集**：(1) 小鼠脊髓损伤 GSE162610，15,627 microglia；(2) 髓系祖细胞 GSE72857，2,730 细胞；(3) 谱系条形码造血时间序列 Weinreb et al.，12,000 细胞（唯一携带独立 fate label）；(4) 成纤维细胞重编程 GSE99915，11,999 细胞。
- **算法**：Absorbing random walk（baseline θ*，24 配置+seed replicate）+ Palantir（13 配置，12 有效）。
- **关键结果**：
  - 单一算法 24 配置：multiplicity rate 3.8%；加入 Palantir 后 36 配置：升至 20.4%。等基数采样证明**多样性**驱动而非数量：12 个 Palantir 配置（20.0%）> 24 个 absorbing-walk（3.8%）。
  - 跨数据集复现：GSE72857 相同单一算法 grid 下 3.88%，与 microglia 3.8% 一致。
  - 急性损伤 1 dpi multiplicity 峰值 10.4%，uninjured 仅 0.4%。
  - FM vs simulation truth：AUC=0.682；基线 margin AUC=0.965（p=0.003）；seed dispersion AUC=0.854。
  - FM vs clonal fate：uncertified 细胞与克隆观察不一致率 27.7%，certified 仅 11.3%，差异 16.4 pp（p<0.001）。
  - Supremum m̄_i 达 AUC=0.973（与基线 margin 无显著差异）。
  - 固定 |R|=4：seed refits AUC=0.933，hyperparameter-perturbed AUC=0.701。
  - q-quantile 恢复：q=0.50 时 AUC=0.973，收敛至单模型 margin 而非超越。
  - Margin erosion ratio（仿真）：AUC=0.890（Mann-Whitney p=6.2×10⁻⁹）。

## 相关工作脉络
1. **Marx et al. (2020) / Watson-Daniels et al. (2023)**：分类/Rashomon capacity 与概率分类中的 predictive multiplicity 形式化；本文移植到无监督轨迹推断，用 discrepancy 替换 labeled loss。
2. **Paes et al. (2023)**：用 hypothesis test 区分 Rashomon set 内两模型；本文 test 用于**绘制** set 边界（无 loss 时如何画），而非 set 内分辨力。
3. **Multiverse analysis (Steegen et al. 2016; Patel et al. 2015)**：遍历参数组合但不过滤 poor-fitting 模型；本文 gate 过滤 statistically inadmissible 配置。
4. **Single-cell trajectory benchmarks (Saelens et al. 2019)**：记录不同算法拓扑差异但无 admissibility 概念；本文首次分离 legitimate 参数变异与 model failure。
5. **CellRank / Palantir / Monocle 等**：均报告 point-estimate fate probability；本文揭示这些置信度跨等价配置可变异 483 倍。
6. **Hüllermeier & Waegeman (2021)**：aleatoric/epistemic uncertainty 分解；本文工作聚焦 epistemic（model-space）variability 在轨迹推断中的测量。

## 局限性与未来方向
- 两个算法共享 graph_diffusion 亲缘，跨 family multiplicity 增量是 lower bound；需 optimal-transport / RNA-velocity 等方法验证。
- Discrepancy metric 仅评估 ordering smoothness，不直接评估 fate；low-yield 数据集（如 Schwann cells n=618）动态范围不足导致 metrics collapse。
- Adapter 实现限制仅支持 1–2 个 terminal fate，无法表达真实 multifurcation。
- Margin-erosion ratio 仅在仿真中验证，未在真实数据上测试；no-fork controls 来自同一 generator，存在同源性偏倚。
- 6,000 细胞 subsampling 为内存限制；合成对象仅 300 细胞。
- Future：(1) 扩展至跨 family 算法空间；(2) 探索加权 aggregator 能否超越单模型 margin；(3) 开发支持 K>2 终态的 fate model。

## 研究启发与可借鉴点
1. **Discrepancy-as-loss 范式**：在缺乏 ground-truth label 的自监督/无监督场景中，可用 cross-fitted held-out 指标 + seed-calibrated non-inferiority test 替代 supervised loss 定义 admissible set，可迁移至其他无监督排序/嵌入任务。
2. **Certification power 度量 Π**：报告 multiplicity rate 时必须同时报告 Π，否则不同研究间数字不可比；该设计思想可用于任何"集合统计量"的可信度评估。
3. **Negative result 的方法论价值**：infimum 作为 extreme order statistic 的失效机制具有普适性——任何"worst-case over heterogeneous model set"设计都需警惕 aggregator 退化，而非 quantity 本身；supremum 因含 baseline 成员而有下界，可作对照。
4. **q-quantile 连续体诊断**：从 inf（q=0）到 sup（q=1）扫描可定位信息损失发生位置，是一种通用诊断工具。
5. **Margin erosion ratio**：用 ratio 而非 absolute 阈值区分"结构真信号"与"算法诱导 artifact"的思路，可推广至其他分支检测任务。

## 关键术语表
- **Rashomon set**：在给定数据上统计表现等价（落入容差边界内）的所有模型/配置的集合。
- **Predictive multiplicity**：不同等价模型对同一输入给出不同预测的现象。
- **Certified fate margin (FM)**：admitted 集合中所有配置对单细胞 fate decision margin 的下确界；FM>0 表示所有等价模型对该细胞 fate 一致。
- **Certification power (Π)**：admitted 集合内 pairwise fate 分歧均值 / 同一配置种子重拟合分歧均值；Π≈1 表示认证仅反映种子稳定性而非数据决定性。
- **Von Neumann ratio**：相邻方差与总体方差的比值，期望在随机排序下为 1，衡量沿伪时间方向的表达平滑度。
- **Non-inferiority test**：检验候选模型性能是否"不差于"基线超过容差 δ；此处用于判定模型是否属于 Rashomon set。
- **Margin erosion ratio**：certified margin 与 baseline mean margin 之比，用于区分真实分支点与算法伪分支。
- **Well-formedness filter**：排除 >50% 细胞 fate posterior 均匀的退化配置，避免 ordering 良好但 fate 坍缩的假阳性 admitted 模型。

## 可复现要素
- 数据集：全部公开——GEO GSE162610、GSE72857、GSE99915；谱系条形码造血数据来自 Weinreb et al. (2020)。
- 代码：开源，https://github.com/arjunbhupatiraju/cns-pns-regeneration，主 notebook 为 FateMultiplicity.ipynb。
- 超参：K=3 折 cross-fitting，α=0.05（B-H 校正），δ=1.086×10⁻³（seed 校准），10,000 sign-flip permutations，48 module blocks。
- 仿真 master seed：20260808（确定性再生）。
- 论文未提及：GPU 要求、训练时长、具体硬件环境（仅声明 environment lock）。
