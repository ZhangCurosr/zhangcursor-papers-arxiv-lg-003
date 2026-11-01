---
title: "SplitJEPA-Learning-Invariant-and-Variant-Latent-Worlds-witho"
source: https://arxiv.org/pdf/2610.12349v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:09:13"
field: "表征学习与可识别性"
keywords: ["JEPA", "不变-变化分解", "块可识别性", "无重建表征学习", "机器人操作", "内容-风格解耦", "因果表征"]
innovations: ["提出无需重建的 SplitJEPA 框架，在表示空间联合预测对齐与不变一致性实现块级可识别", "证明在平稳高斯预测动力学下不变/变化子空间可识别至块内正交变换等价类", "从理论上保证仅基于不变表示的下游预测对变化扰动梯度隔离"]
benchmarks: ["Synthetic OU system", "ManiSkill3 PushCube", "ManiSkill3 PickCube"]
---

# 论文速读：SplitJEPA-Learning-Invariant-and-Variant-Latent-Worlds-witho

## 一句话总结
SplitJEPA 提出了一种**无需重建**的 JEPA 方法，通过联合使用表示空间的预测对齐与不变对应的一致性约束，在理论上可识别性地分解出隐式世界的不变（invariant）子空间和变化（variant）子空间，并在合成非线性系统及机器人 PushCube/PickCube 操作任务上验证了其实用价值。

## 研究问题与动机
- **核心问题**：在无需重建观测的前提下，能否可识别地恢复动态隐式状态中的不变-变化分解结构？
- **现有重建方法的不足**：内容-风格类方法和对比学习方法通常需要解码器重建观测，引入额外计算成本，且仅保证信息保留并不能指定不变/变化因子的划分方式；存在非线性的潜变量混淆（Theorem 1）。
- **纯预测方法的不足**：LeJEPA 等在高斯平稳预测动力学下可恢复完整隐状态（全局正交等价类），但剩余的全局正交变换仍会混合不变与变化方向（Proposition 2）。
- **开放挑战**：预测恢复与块结构识别是不同层次的问题，如何将两者结合以在不重建的前提下实现不变-变化的块级可识别性尚未解决。

## 核心贡献（创新点）
- **理论层面**：证明在平稳高斯预测动力学和充分变化（full-rank variation）条件下，SplitJEPA 无需观测解码器即可将不变子空间和变化子空间识别到独立块内正交变换的等价类（Theorem 2）。
- **方法层面**：提出一种重构-free 的 JEPA 框架，联合预测损失 $\mathcal{L}_{\text{pred}}$、不变一致性损失 $\mathcal{L}_{\text{inv}}$ 和方差正则化 $\mathcal{L}_{\text{var}}$，在表示空间内完成预测对齐与块分解。
- **鲁棒性保证**：证明块可识别性导出下游仅基于不变表示 $r_H$ 的预测对变化扰动局部不敏感（$\frac{\partial Y}{\partial z_L} = 0$，Theorem 3）。
- **实验验证**：在合成非线性 OU 系统上，SplitJEPA 相比 LeJEPA 的不变/变化 $R^2$ 从 ~0.96 提升至 ~0.998，块纯度从 0.964 提升至 0.999993；在 ManiSkill3 PushCube 视觉 OOD 实验中，$r_H+r_L$ 的 SR@10cm 达到 70.2%，显著超越 R3M（25.8%）和 LeJEPA（28.7%）。

## 方法详解
- **预测动力学模型**：给定上下文 $X$，有序对满足 $z^{(1)}|X \sim N(0, I_d)$，$z^{(2)} = \rho_X z^{(1)} + \sqrt{1-\rho_X^2}\epsilon$，其中 $\epsilon \perp z^{(1)}|X$，$\epsilon \sim N(0, I_d)$，记此分布为 $\mathcal{D}_{\text{pred}}$。
- **不变对应关系**：无序对 $(z^{(1)}, z^{(2)}) \sim \mathcal{D}_{\text{inv}}$ 满足 $z^{(1)} = (z_H, z_L^{(1)})$，$z^{(2)} = (z_H, z_L^{(2)})$，即不变因子 $z_H$ 相同而变化因子可不同。
- **损失函数**：
  - 预测对齐损失：$\mathcal{L}_{\text{pred}}(g) = \frac{1}{2}\mathbb{E}_{\mathcal{D}_{\text{pred}}}[\|g(z^{(1)}) - g(z^{(2)})\|_2^2]$
  - 不变一致性损失：$\mathcal{L}_{\text{inv}}(g) = \mathbb{E}_{\mathcal{D}_{\text{inv}}}[\|g_H(z^{(1)}) - g_H(z^{(2)})\|_2^2]$
  - 总损失：$\mathcal{L}_{\text{repr}}(g) = \mathcal{L}_{\text{pred}}(g) + \lambda_{\text{inv}}\mathcal{L}_{\text{inv}}(g) + \lambda_{\text{var}}\mathcal{L}_{\text{var}}(g)$
- **归一化约束**：要求 $\mathbb{E}[g(z)] = 0$，$\text{Cov}(g(z)) = I_d$，通过 ZCA 白化 + SIGReg 实现。
- **块恢复机制**：预测阶段将全局非线性的混淆简化为正交变换 $Q$；利用不变对的差值二阶矩矩阵 $\Sigma_{\Delta L} = \mathbb{E}[\Delta z_L \Delta z_L^\top] \succ 0$ 通过谱分析识别不变子空间（$\Delta z_L$ 零空间的正交补）。
- **Hermite 谱分析证明**：在高斯平稳预测下，$\mathcal{L}_{\text{pred}}$ 的最优解仅保留一次 Hermite 项，从而强制线性解 $g(z) = Qz$，将非线性混淆归约为正交旋转。

## 实验与结果
- **合成实验（非线性 OU 系统）**：8 维隐变量（$d_H = d_L = 4$），耦合强度 0.5，预测相关系数 $\rho = 0.5$。在共享冻结编码器的前提下：
  - SplitJEPA 的不变 $R^2$ = **0.9979 ± 0.0014**，变化 $R^2$ = **0.9973 ± 0.0019**，$\|A_{12}\|_F$ = **0.0031 ± 0.0018**，$\|A_{21}\|_F$ = **0.0236 ± 0.0021**，块纯度 = **0.999993 ± 0.00001**。
  - LeJEPA 的不变/变化 $R^2$ ≈ 0.961，块纯度 ≈ 0.964，交叉泄漏约 0.33。
  - Shuffle Pairs 破坏了对应关系后，块纯度骤降至 0.714。
  - 消融表明：移除预测对齐降低全局恢复；移除不变对应主要破坏块分离；移除方差正则化增加协方差误差。
  - 随维度扩展：$d=4$ 至 $d=64$，全局 $R^2$ 始终 >0.9999，$\|A_{12}\|$ 保持稳定，$\|A_{21}\|$ 从 0.0053 增至 0.0325。
- **机器人操作（ManiSkill3，PickCube / PushCube）**：800 条成功轨迹，每 100 组构成匹配组，ResNet-50 编码器，$d_H = d_L = 64$（PickCube）/ 128（PushCube）。
  - **ID 条件**：PushCube CONTACT = 100%，PickCube GRASP = **97.0 ± 1.4%**（$r_H$），SR@10cm = **45.0 ± 2.4%**。
  - **OOD 视觉条件**（Camera + Lighting）：Ours-$r_H+r_L$ 在 PushCube 达到 **70.2 ± 12.9%**，相对 R3M（25.8%）提升约 **44.4pp**，相对 LeJEPA（28.7%）提升约 **41.5pp**。
  - **采样效率**：10% 数据时，PushCube ID SR@10cm 即超过 90%；基线方法在 OOD 下无法随数据量提升。
  - **可控干预实验**：对 $r_L$ 注入 $2.0\sigma$ 噪声，$r_H$-only 策略保持 99.9% 性能，而 $r_H+r_L$ 降至 44.1%（PushCube），验证 Theorem 3 的梯度隔离。
  - **效率对比**：相比 V3（Wu et al., 2025），SplitJEPA 达到同等恢复目标的时间节省 **2.70×**，观测成本节省 **6.33×**，可训练参数更少（64 vs 272）。

## 相关工作脉络
- **LeJEPA**（Balestriero & LeCun, 2025；Klindt et al., 2026）：在平稳高斯预测动力学下证明可线性恢复完整隐状态，但剩余的全局正交混淆无法区分不变/变化块——本文在此基础上增加不变对应约束完成块识别。
- **内容-风格解耦**（K\"ugelgen et al., 2021；Timilsina et al., 2024, 2026）：多数依赖重建或差异独立性假设；本文采用配对对应结构，不依赖解码器，实现了在无重建条件下的可识别分解。
- **非线性 ICA / 时间对比学习**（Hyvarinen & Morioka, 2016, 2017）：通过时序依赖和不对称性识别组件；本文利用预测对（temporal predictivity）与不变对应（cross-view/cross-trajectory）两种互补结构联合约束。
- **JEPA 系列**（Assran et al., 2023；Bardes et al., 2024；V-JEPA 2）：在表示空间预测而非像素重建；本文将其理论化扩展至结构化块识别。
- **R3M**（Nair et al., 2022）：通用视觉表征用于机器人操作，无不变/变化分解保证；本文在其基础上提供可解释的块分离与 OOD 鲁棒性。
- **Zhang et al. (2021)**：无重建的不变表征学习，但无理论可识别性保证；本文提供严格的可识别性定理及实证验证。

## 局限性与未来方向
- **规模限制**：论文明确指出，扩展至大规模复杂真实场景是未来方向，当前实验限于合成低维系统及中等规模机器人仿真。
- **充分变化假设**：变体方向的二阶矩需满秩（$\Sigma_{\Delta L} \succ 0$），若某些变化方向在数据中从未出现则无法识别；实践中交叉轨迹配对是近似对齐而非精确对应。
- **有限样本敏感性**：变化块的估计（$\|A_{21}\|$）随维度升高更敏感，需更多恢复样本才能稳定估计互补块。
- **近似一致性稳定性**：Remark 1 给出定量界，但当 $\varepsilon$（不变一致性误差）较大时，$\|A_{12}\|_F$ 上界也随之增大。
- **下游任务的适用边界**：Theorem 3 的梯度隔离仅对纯变化扰动有效；若扰动同时改变 $z_H$，不变性不再保证。

## 研究启发与可借鉴点
- **双阶段识别框架**：先通过预测对齐消除非线性混淆（归约为线性/正交），再通过互补对应约束完成块划分——这种"预测+一致性"的两阶段思路可迁移到其他需结构识别的问题。
- **谱恢复的实用化设计**：在表示白化后对 $\Delta r$ 的协方差矩阵做特征分解来识别不变子空间，计算高效且无需额外训练；可作为轻量级的后处理模块嵌入 JEPA 类模型。
- **效率基准**：与 V3 的效率对比协议（达到目标恢复阈值的时间/观测成本）为后续工作提供了可复用的评估范式。
- **梯度隔离的鲁棒性**：Theorem 3 的结论提示，在强化学习/行为克隆中仅使用不变块 $r_H$ 可作为天然的对抗扰动防御机制，值得在其他域（如视觉导航、多模态学习）中探索。
- **消融设计的清晰度**：共享冻结编码器的对比策略分离了"全局恢复质量"与"块组织质量"，这一实验设计值得借鉴用于诊断表征学习的方法缺陷。

## 关键术语表
- **JEPA（Joint-Embedding Predictive Architecture）**：在表示空间中进行预测的自监督学习架构，无需重建原始观测即可学习有用的表征。
- **Block Identifiability（块可识别性）**：将隐状态分解为不变和变化子空间，至多保留各子空间内的独立正交变换不确定性。
- **Invariant Correspondence（不变对应）**：配对观察中不变因子 $z_H$ 保持不变、变化因子 $z_L$ 可不同的数据分布 $\mathcal{D}_{\text{inv}}$。
- **Predictive Pair（预测对）**：具有时序/预测依赖关系的有序观察对，满足高斯平稳预测动力学 $\mathcal{D}_{\text{pred}}$。
- **Sufficient Variation（充分变化）**：变化差的二阶矩矩阵 $\Sigma_{\Delta L}$ 正定，保证每个变化方向在配对数据中都有体现。
- **Cross-block Leakage（交叉块泄漏）**：度量不变子空间与变化子空间之间的混合程度，以 Frobenius 范数 $\|A_{12}\|_F$ 和 $\|A_{21}\|_F$ 衡量。
- **Gradient Isolation（梯度隔离）**：下游仅依赖 $r_H$ 的预测对 $z_L$ 扰动局部不敏感，$\frac{\partial Y}{\partial z_L} = 0$。
- **SIGReg（Spectral Information Geometry Regularization）**：方差保留正则化项，鼓励表示满足归一化几何约束（零均值、单位协方差）。

## 可复现要素
- **数据集**：ManiSkill3（Tao et al., 2025）中的 PickCube-v1 和 PushCube-v1；合成非线性 OU 系统；**论文未明确声明数据是否公开**。
- **代码/权重**：**论文未提及是否开源**（无代码链接）。
- **关键超参**：
  - 编码器：ImageNet 初始化 ResNet-50，输入 128×128 RGB。
  - 优化器：AdamW，lr = $3 \times 10^{-4}$，weight decay = $10^{-4}$，cosine lr decay，gradient clip norm = 10。
  - 批量大小：32（相邻时序转移）+ 32（匹配对）。
  - PushCube：$d_H = d_L = 128$，训练 100 epochs。
  - PickCube：$d_H = d_L = 64$，训练 200 epochs。
  - 合成实验：编码器训练 10,000 步，batch=256，lr=$3\times 10^{-3}$，SIGReg weight=$10^{-6}$，$\rho=0.5$，耦合强度=0.5。
  - 恢复阶段：ZCA 白化，匹配对数量 30,000，评估探针 15,000。
