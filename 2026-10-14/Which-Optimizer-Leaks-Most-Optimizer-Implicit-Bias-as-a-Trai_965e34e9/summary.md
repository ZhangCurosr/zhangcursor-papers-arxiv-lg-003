---
title: "Which-Optimizer-Leaks-Most-Optimizer-Implicit-Bias-as-a-Trai"
source: https://arxiv.org/pdf/2610.18294v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:39:08"
field: "模型隐私与安全"
keywords: ["training data privacy", "implicit bias", "optimizer geometry", "data reconstruction attack", "neural network security", "Muon optimizer", "Adam optimizer", "KKT conditions"]
innovations: ["提出通用norm-parameterized重构攻击框架，统一SGD/Adam/Muon三类优化器的margin几何", "定义并证明weight-specificity理论，揭示Adam ℓ∞目标特异性为零导致一阶不可识别", "结合特异性与可达性两轴理论解释Muon≻SGD≻Adam的泄漏排序并实验验证"]
benchmarks: ["MNIST binary", "CIFAR-10 binary", "CIFAR-10 10-class"]
---

# 论文速读：Which Optimizer Leaks Most? Optimizer Implicit Bias as a Training-Data Privacy Controller

## 一句话总结
本文系统研究了不同优化器（SGD、Adam、Muon）的隐式偏差（implicit bias）如何影响训练数据隐私泄漏程度，提出了一个基于优化器相关margin几何的通用重构攻击框架，得出 **Muon ≻ SGD ≻ Adam** 的隐私泄漏排序。

## 研究问题与动机
1. **模型checkpoint可泄露训练数据**：过参数化网络会记忆训练样本，Haim et al. (2022) 已证明仅凭最终权重即可重构大部分训练图像，但现有攻击仅针对梯度下降（GD）训练。
2. **不同优化器诱导不同margin几何**：不同优化器在齐次网络上收敛到不同margin问题的KKT点——SGD→ℓ₂ margin、Muon→max-spectral-norm margin、Adam→ℓ∞ margin，其stationarity条件各异，可能带来截然不同的泄漏风险。
3. **现有方法缺乏系统比较**：尚无工作系统分析优化器选择对数据重构攻击的影响，优化器仅被视为"优化决策"而非"隐私控制手段"。
4. **理论可解释性需求**：需要建立可解释的框架，理解为何某些优化器更容易被攻击，以指导隐私安全的训练策略选择。

## 核心贡献（创新点）
1. **通用重构攻击框架**：将Haim et al. (2022)的目标推广到任意margin范数，修正了直接推广到谱范数时因奇异值重根和跨块尺度不匹配导致的两种失效模式。
2. **提出"权重特异性"（weight-specificity）概念**：从理论上刻画stationarity target通过多少独立连续泛函编码最终权重信息；证明Adam的ℓ∞目标在所有最大坐标重数下特异性为零。
3. **建立第一阶不可识别性理论**：在非退化KKT点处，Adam的stationarity约束方程数至多为在margin上的样本数，远少于未知量，故训练集在第一个意义上不可识别。
4. **特异性与可达性（reachability）联合分析**：结合两者的量化分析得出Muon ≫ SGD ≫ Adam的实证排序，解释了经验现象的理论根源。
5. **全面的MNIST/CIFAR-10实验验证**：在无图像先验条件下，Muon训练模型可产生SSIM>0.5的可识别重构，而Adam始终低于0.4的可视阈值。

## 方法详解
**KKT stationarity基础**：齐次网络+指数/Logistic损失→优化器收敛到max-margin问题 $\min_\theta \frac{1}{2}\|\theta\|^2$ s.t. $y_i f(x_i;\theta)\geq 1$ 的KKT点。攻击者利用stationarity条件：残差 $r(z)=\sum_i \lambda_i y_i \nabla_\theta f(x_i;\theta)$ 必须落入目标锥 $C(\theta)=\mathrm{cone}(\partial(\frac{1}{2}\|\theta\|^2))$。

**三种优化器的margin范数**（Eq.2）：
- SGD：$\|\theta\|_{\mathrm{SGD}} = (\sum_k \|W_k\|_F^2)^{1/2}$（ℓ₂，对偶为ℓ₂）
- Muon：$\|\theta\|_{\mathrm{Muon}} = \max_k \|W_k\|_{\mathrm{sp}}/c_k$（max-spectral，对偶为$\sum_k c_k\|\cdot\|_{\mathrm{nuc}}$）
- Adam：$\|\theta\|_{\mathrm{Adam}} = \|\theta\|_\infty$（全局ℓ∞，对偶为ℓ₁）

**对齐损失（scale-invariant）**：通过层wise Euler恒等式消除乘子尺度自由度，构造逐block的对齐损失（Eq.8）：
$$L_{\mathrm{align}} = \sum_k \left(1 - \frac{\langle r_{W_k}, W_k\rangle}{\|r_{W_k}\|_{(k),\star} \|W_k\|_{(k)}}\right)^2$$
该损失对乘子缩放不变，且无需跨block耦合项。

**Muon攻击（谱范数对齐）**：Eq.(10)中对每个block惩罚核范数与谱范数之比偏离1的程度，在KKT点处精确等价于原始stationarity条件。

**Adam攻击（ℓ∞对齐）**：利用Hölder不等式等号条件：候选残差必须支撑在最大绝对值坐标集合$A=\arg\max_j|\theta_j|$上且符号匹配，单全局项：
$$(1 - \langle r, \theta\rangle / (\|r\|_1 \|\theta\|_\infty))^2$$

**权重特异性（Definition 4.1）**：在active-all-blocks的stratum上，$\theta \mapsto C(\theta)$的像空间维数：
- SGD：$S = p-1$（编码完整方向）
- Muon：$S = \sum_k S_k(\rho_k)$，其中$S_k(\rho)=\rho(m_k+n_k-\rho)-\rho(\rho+1)/2$
- Adam：$S = 0$（仅依赖离散arg-max模式）

## 实验与结果
**数据集与设置**：Binary MNIST、Binary CIFAR-10、10-class CIFAR-10，每任务$n=500$样本。模型为全连接ReLU网络$d\text{-}1000\text{-}1000\text{-}C$。SGD需约$10^6$次迭代达到$O(10^{-7})$损失，Adam/Muon仅需$10^4$次。

**主要结果（无图像先验，SSIM@30）**：

| 任务 | SGD | Adam | Muon |
|---|---|---|---|
| MNIST binary | 0.346 | 0.249 | **0.431** |
| CIFAR-10 binary | 0.229 | 0.138 | **0.551** |
| CIFAR-10 10-class | 0.383 | 0.207 | **0.534** |

- **Muon最强**：在Binary CIFAR-10上比SGD提升超过一倍（0.229→0.551），且以$10^{-2}$的训练成本获得最优重构。
- **Adam最鲁棒**：在所有任务上SSIM@30均<0.25，远低于0.4的视觉可识别阈值。
- **图像先验消融**：加入Range/TV/L1先验后Muon仍始终最优（最佳0.660），Adam改进最小。
- **样本量影响**：$n=100$时收敛后的SGD与Muon接近（0.623 vs 0.698），表明Muon优势部分源于更快的收敛速度（可达性）。
- **Adam在所有架构下SSIM@30始终低于其他优化器的一半**，验证了理论预测。

**关键发现**：只有位于最小margin上的样本才能被重构；Muon产生最多可识别样本，Adam几乎不产生。

## 相关工作脉络
1. **Haim et al. (2022)**：首次证明仅凭最终权重可重构训练图像，利用GD的ℓ₂隐式偏差反演KKT条件——本文的直接前身，仅覆盖ℓ₂情形，本文推广到Muon和Adam的margin几何。
2. **Buzaglo et al. (2023)**：将Haim等人的方法扩展到多分类、weight decay和更一般的损失函数——本文扩展方向与其一致，但聚焦于优化器效应而非损失/正则项。
3. **Gronich & Vardi (2026)**：分析了Adam和Muon在光滑齐次网络上的隐式偏差，建立了Adam→ℓ∞、Muon→max-spectral margin的理论联系——本文借用其margin几何结论进行攻击设计。
4. **Refael et al. (2026b)**：研究了图像先验在checkpoint重构中的主导作用——本文与之互补，证明即使无先验，Muon的stationarity条件本身也足够强。
5. **Lyu & Li (2020), Ji & Telgarsky (2020)**：建立了齐次网络+指数损失→GD收敛到max-margin KKT点的理论基础——本文攻击框架的理论基石。
6. **Zhang et al. (2024)**：独立研究了Adam在可分数据上的ℓ∞隐式偏差——与Gronich & Vardi结论一致，本文将其与隐私泄漏分析关联。

## 局限性与未来方向
1. **无形式化隐私保证**：分析仅涉及stationarity-based攻击族，不提供差分隐私风格的保证。
2. **仅在特定攻击模型下有效**：攻击者仅能观察checkpoint，不能查询模型；若有额外信息（如查询访问），分析不适用。
3. **Adam理论针对理想化情形**：ℓ∞隐式偏差假设Adam的稳定性常数可忽略，实际实现可能有偏差。
4. **可扩展性待验证**：仅在FC-ReLU网络和MNIST/CIFAR-10上验证，未测试CNN、Transformer等更大规模架构。
5. **未来方向**：将几何隐式偏差分析扩展到AdaRankGrad、SUMO、Lorenza等低秩/正交化优化器，研究自适应秩降维和子空间正交化如何控制训练数据泄漏。

## 研究启发与可借鉴点
1. **优化器作为隐私控制手段**：团队在进行模型发布/共享时，可考虑将Adam等ℓ∞类优化器作为降低训练数据泄露风险的实用策略，而非仅从优化效率角度选择。
2. **"权重特异性"理论框架**：可将此概念迁移到其他隐私分析场景，定量评估不同模型结构/约束下stationarity条件的信息编码能力。
3. **scale-invariant对齐损失设计**：第3节的scale-invariant penalty + layerwise Euler identity技巧，适用于任何需要将stationarity条件转化为可优化目标的攻击/分析场景。
4. **特异性与可达性的两轴分析**：此框架可用于系统性比较优化器行为——不仅看最终解的质量，也看达到极限方向的速度，对训练效率与隐私的权衡设计有参考价值。
5. **实验设计借鉴**：用SSIM@30而非all-pairs平均来评估可识别性，避免了Adam因生成"通用图像"而在平均指标上占优的假象，值得后续评测采用。

## 关键术语表
**Implicit Bias（隐式偏差）**：优化器在无显式正则化时，自然收敛到的特定解的几何特性（如GD收敛到max-margin解）。
**Weight-Specificity（权重特异性）**：stationarity target锥$C(\theta)$随权重连续变化的维度，衡量优化器在KKT条件中编码了多少关于最终权重的连续信息。
**Reachability（可达性）**：优化器的参数方向趋近其极限margin KKT方向的收敛速度。
**Stationarity Condition（平稳性条件）**：KKT条件中的梯度为零条件，表达了模型权重可由训练样本梯度线性组合表示的关系。
**Max-Spectral-Norm Margin（最大谱范数margin）**：Muon优化器诱导的margin几何，等价于对各层谱范数的最大值进行最大化。
**KKT Point（Karush-Kuhn-Tucker点）**：带约束优化问题的局部最优解所满足的必要条件，此处对应max-margin问题的解。
**Target Cone $C(\theta)$**：由优化器相关的subdifferential生成的锥，是stationarity约束中"优化器指纹"的载体。
**SSIM@30**：从$m=2n$个候选中选取与真值最佳匹配的top-30对计算的Structural Similarity均值，避免平均指标被非识别样本稀释。

## 可复现要素
- **数据集**：MNIST（binary）、CIFAR-10（binary + 10-class），论文未提及开源状态（标准公开数据集）
- **代码/权重**：论文声明"Code and reproduction scripts will be made publicly available"（Appendix D），截至当前未确认是否已开源
- **关键超参**：重构学习率$10^{-4}\sim10^{-2}$（log-uniform），初始化$\sigma_x$ $10^{-3}\sim1$（log-uniform），ReLU平滑参数$\alpha$ 10~500（uniform），重构迭代100,000次，momentum=0.9，攻击候选数$m=2n$
