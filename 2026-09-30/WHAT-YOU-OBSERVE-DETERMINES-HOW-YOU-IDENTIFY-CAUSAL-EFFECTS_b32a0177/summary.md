---
title: "WHAT-YOU-OBSERVE-DETERMINES-HOW-YOU-IDENTIFY-CAUSAL-EFFECTS"
source: https://arxiv.org/pdf/2609.36881v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:53:34"
field: "因果推断与机器学习交叉"
keywords: ["causal foundation models", "causal identification", "CATE estimation", "observational views", "modular estimators", "partial identification", "negative control outcomes"]
innovations: ["提出CAUSALIDVIEW多视图基准，在相同因果世界中控制性改变观测视图以评估CFM跨识别机制性能", "引入Stay/Move结构性响应诊断，量化估计器对无关结构变化的虚假敏感性和对真实效应变化的追踪能力", "验证TabPFN-v3.5模块化估计器在后门/前门/近端视图中达到与CFM相当甚至更优的点估计和偏识别界估计性能"]
benchmarks: ["CAUSALIDVIEW", "ACIC 2016", "LaLonde-PSID", "LaLonde-CPS", "IHDP", "ACTG 175", "Pennsylvania Reemployment Bonus", "Illinois UI Experiment"]
---

# 论文速读：WHAT-YOU-OBSERVE-DETERMINES-HOW-YOU-IDENTIFY-CAUSAL-EFFECTS

## 一句话总结
本文提出了 CAUSALIDVIEW 多视图基准，在保持 SCM 实现、查询单元和目标 CATE 固定的条件下，仅改变估计器可观测的变量，系统评估因果基础模型（CFMs）在不同识别机制下的性能差异与脆弱性。

## 研究问题与动机
- **现有 CFM 评估无法控制比较条件**：CausalPFN、Do-PFN、CausalFM 等在不同预训练先验和数据生成过程下训练评估，难以判断性能差异是来自识别信息变化还是其他因素。
- **跨识别机制的系统性对比缺失**：现有基准（如 IHDP、ACIC、RealCause）主要在固定识别机制内变化 DGP，未在同一因果世界中比较不同观测视图。
- **CFM 对结构变化的响应行为不明**：当真实效应不变而结构扰动时，或效应本身改变时，CFMs 的表现缺乏系统性诊断。
- **模块化方法 vs CFM 的对比不足**：强预测器结合显式识别程序的竞争力未被系统评估。

## 核心贡献（创新点）
1. **提出 CAUSALIDVIEW 多视图基准**：同一因果世界投影到后门、前门、工具变量、近端四种识别视图及隐藏混杂压力视图，保持目标 CATE 固定，实现控制性比较。
2. **引入结构性响应诊断（Stay/Move）**：量化估计器在"效应不变但结构改变"时的虚假响应和"效应改变"时的追踪准确性，揭示不同 CFM 的模型特异性失败模式。
3. **验证模块化识别策略的竞争力**：将 TabPFN-v3.5 等表格基础模型与 S/X/DR-Learner、前门插件、Wald 估计器、P-Learner 等显式识别程序结合，在后门/前门/近端视图中达到最优或接近最优误差。
4. **揭示 CFM 在多视图下无一致最优者**：CausalFM 在 IV 视图领先，TabPFN-DR-Learner 在后门视图最优，不同评估场景（半合成/真实世界/null-effect）最佳模型各不相同。
5. **提供负对照 RCT 基准**：利用真实 RCT 中治疗前测量的变量作为零效应基准（ACTG 175、Pennsylvania、Illinois），检验估计器的零效应偏差。

## 方法详解
**共享因果世界构造**：每个世界由一个 SCM 生成，包含 50 维协变量 X、二值混杂 U、二值工具变量 I、二值中介 M、预后代理 (Zp, Wp)、处理 T 和结果 Y。目标 CATE 构造满足 τ₀(x) = δ·β(x)，且单位水平效应 Yᵢ(1) - Yᵢ(0) = τ₀(Xᵢ)，确保所有识别视图恢复相同目标。

**四个点识别视图**：
- **后门（BD）**：观测 (X, U, T, Y)，通过条件交换性和正性识别。
- **前门（FD）**：观测 (X, T, M, Y)，U 未观测但 M 满足前门准则，通过条件前门函数式恢复总效应。
- **工具变量（IV）**：观测 (X, I, T, Y)，使用条件 Wald 比率 π_IV(x)。
- **近端（PX）**：观测 (X, Zp, Wp, T, Y)，使用近端 g-公式和桥函数 hₜ(w,x)。

**保持目标一致的关键约束**：U 以相同加性项 γ_U(X)·U 进入两个潜在结果，中介效应固定为 δ·β(X)，跨治疗臂共享中介和结果噪声。

**模块化估计器架构**：预测性基础模型（TabPFN-v3.5/XGBoost/NN/ExtraTrees）估计识别所需的混杂因子/倾向得分/中介分布等 nuisance 量，再通过显式因果公式（DR-Learner 伪结果、前门积分、Wald 比率、近端桥方程）计算 CATE。

**结构性响应度量**：
- E_stay：CATE 不变时估计值的变化，衡量虚假敏感性。
- E_move：CATE 加倍时估计值变化的追踪精度。

**偏识别评估**：Manski 最坏情况界、单调 IV 界、边际灵敏度模型（MSM）界，评估端点 RMSE 和真实 CATE 包含率。

## 实验与结果
**数据集**：
- 合成：40 个 SCM 世界，每世界 1,024 上下文单元 + 100 查询单元，5 种效应族（线性/二次交互/阈值/傅里叶/浅层 MLP）。
- 半合成：ACIC 2016（77 个参数设置取 40 个）、RealCause 生成的 LaLonde-PSID 和 LaLonde-CPS。
- 真实世界：ACTG 175（AZT 疗法，n_C=768）、Pennsylvania Reemployment Bonus（n_C=1,024）、Illinois UI Experiment（n_C=2,048），使用前处理 CD4/工资/收入作为零效应负对照。

**主要结果数字**：
- 后门视图：TabPFN-v3.5 + DR-Learner 最优 sPEHE² = 0.143 ± 0.054（Default），优于 CausalPFN（0.208）和 CausalFM（0.233）。
- 前门视图：TabPFN-v3.5 + FD Plug-in 最优 sPEHE² = 0.154 ± 0.057，远低于 CausalFM（0.551）和 Do-PFN（0.368）。
- IV 视图：CausalFM 最优 sPEHE² = 0.258 ± 0.041，TabPFN-Wald 为 0.336 ± 0.105。
- 近端视图：TabPFN-v3.5 + P-Learner 最优 sPEHE² = 0.181 ± 0.054，CausalFM 为 0.240。
- 半合成 ACIC 2016：TabPFN-v3.5 + X-Learner 最优 sPEHE² = 1.11 ± 1.56，显著低于 CausalPFN（3.24）、Do-PFN（31.48）、CausalFM（23.56）。
- 负对照 null-effect 测试：CausalFM 在 ACTG 175（0.0053）和 Pennsylvania（0.0023）表现最好，TabPFN-S-Learner 在 ACTG 175 达到最低（0.0008）。
- 隐藏混杂压力：所有模型误差均上升；TabPFN 基础模型在 Manski 和 IV 偏识别界估计中取得最低端点 RMSE。
- 结构响应：CausalPFN Stay 误差较大（预测对无关结构变化敏感），Do-PFN Move 误差大（无法追踪真实效应变化），CausalFM 呈视图依赖混合行为；TabPFN 模块化估计器在 Stay 和 Move 上均表现更均衡。

**核心结论**：无 CFM 在所有视图下最优；模块化方法在后门/前门/近端视图中具有竞争力甚至最优；数据集依赖性强；CFM 存在模型特异性脆弱性。

## 相关工作脉络
1. **CausalPFN / Do-PFN / CausalFM**：现有 CFM 通过 in-context learning 摊销因果推断，但训练/评估协议不统一，且未在相同因果世界中跨识别机制对比——本文填补此空白。
2. **IHDP / ACIC / RealCause 基准**：主要在固定识别机制内变化 DGP，未控制性改变可用识别信息——CAUSALIDVIEW 弥补这一局限。
3. **TabPFN（Grinsztajn et al., 2026; Jager et al., 2026）**：表格基础模型，本文验证其作为模块化估计器预测 backbone 的因果推断价值。
4. **偏识别文献（Manski, 1990; Balke & Pearl, 1997）**：传统经济学/因果推断中处理部分识别的方法，本文将其纳入统一基准评估 bound 估计器性能。
5. **负对照结果方法（Arnold & Ercumen, 2016; Ashby et al., 2025）**：利用治疗前变量作为零效应基准检测估计偏差，本文扩展至因果基础模型的偏差诊断。
6. **DR-Learner / S-X-Learner / P-Learner（Kennedy, 2023; Kunzel et al., 2019; Sverdrup & Cui, 2023）**：显式因果估计程序，本文将其与 TabPFN 等现代预测器结合形成对比基线。

## 局限性与未来方向
- **合成世界结构有限**：未包含纵向/时变处理、动态治疗策略、单元间干扰、连续/多值处理、删失生存结果、多维潜混杂等复杂场景。
- **单一因果图结构**：所有视图来自同一 SCM，未覆盖更广泛图结构谱系。
- **模块化估计器超参未调优**：XGBoost/NN/ExtraTrees 使用默认参数，TabPFN 冻结权重，可能存在优化空间。
- **CFM 未在各自原生视图上比较**：CausalPFN/Do-PFN 将辅助变量作为普通特征使用，未使用专用 IV/proximal wrapper，可能低估其潜力。
- **未来方向**：扩展 CAUSALIDVIEW 到更广泛因果设定；深入研究 TabPFN 在不同识别机制下的设计；开发对识别假设变化更鲁棒的 CFM。

## 研究启发与可借鉴点
1. **"相同世界不同视图"的对照设计**：可作为评估其他因果学习方法（如表示学习、表示不变性方法）是否真正利用识别信息的通用范式。
2. **Stay/Move 结构性响应诊断**：可迁移用于评估任何因果估计器的稳健性，区分"学到识别关系" vs "学到数据共现模式"。
3. **模块化架构的工程价值**：TabPFN + 显式因果公式的组合在后门/前门/近端视图中均表现优异，提示在资源受限场景下可优先采用此类可解释架构。
4. **负对照 RCT 基准的设计**：利用治疗前变量做零效应测试的思路可扩展到其他场景（如安慰剂处理、伪处理变量）。
5. **sPEHE² 分解诊断**：将总误差分解为 ATE 误差和中心化 CATE 误差，有助于定位估计失败的具体类型（系统性偏移 vs 异质性捕捉不足）。

## 关键术语表
- **CATE（Conditional Average Treatment Effect）**：条件平均处理效应，E[Y(1) - Y(0) | X=x]，衡量协变量 x 下的异质性治疗效应。
- **Structural Causal Model (SCM)**：结构因果模型，用结构方程描述变量间因果关系的数学框架。
- **Back-door identification**：后门识别，通过调整混杂变量 U 阻断后向路径，利用调整公式恢复因果效应。
- **Front-door identification**：前门识别，当混杂不可观测但存在中介 M 时，通过中介变量路径分解恢复总效应。
- **Proximal causal identification**：近端因果识别，使用混杂变量的代理变量（proxies）而非直接观测混杂，通过桥函数方程恢复效应。
- **Partial identification**：偏识别/部分识别，当点识别不可能时，刻画与观测分布相容的所有可能效应值的集合（上下界）。
- **Negative control outcome**：负对照结果，治疗前测量且不受治疗影响的变量，理论上处理效应为零，用于检测估计器偏差。
- **Modular estimator**：模块化估计器，将 nuisance 量的预测（由 ML 模型完成）与显式因果识别公式（如 DR 伪结果、Wald 比率）分离的架构。

## 可复现要素
- **数据集**：CAUSALIDVIEW 为作者构建的合成基准；半合成数据来自公开基准（ACIC 2016、LaLonde-PSID/CPS via RealCause、IHDP）；真实世界数据来自 ACTG 175、Pennsylvania Reemployment Bonus、Illinois UI Experiment 三个 RCT。论文未明确声明代码开源状态。
- **代码/权重**：CFM（CausalPFN/Do-PFN/CausalFM）使用官方 checkpoint 且未微调；TabPFN-v3.5 为公开模型。论文未提及自有代码是否开源。
- **关键超参**：TabPFN-v3.5（默认冻结权重）；XGBoost：300 trees，max_depth=3，learning_rate=0.05，histogram 方法；NN：2-layer MLP，128 hidden units，Adam LR=1e-3，weight_decay=1e-4，batch=128；ExtraTrees：300 trees，max_features=0.8，min_leaf_size 分类=8/回归=6/最终回归=10；近端桥估计 ridge=0.01。
