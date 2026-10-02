---
title: "WHAT-YOU-OBSERVE-DETERMINES-HOW-YOU-IDENTIFY-CAUSAL-EFFECTS"
source: https://arxiv.org/pdf/2609.36881v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:53:32"
field: "因果推断与因果机器学习"
keywords: ["causal foundation models", "causal identification", "CATE estimation", "multi-view benchmark", "TabPFN", "partial identification", "in-context learning"]
innovations: ["提出CAUSALIDVIEW多视图基准，在固定因果世界下跨BD/FD/IV/PX/HC五类识别机制对比CFM", "引入Stay/Move结构响应审计，揭示CFM模型特异性失效模式", "证明TabPFN模块化策略在多数视图下竞争力强于端到端CFM"]
benchmarks: ["ACIC 2016", "LaLonde-PSID", "LaLonde-CPS", "ACTG 175", "Pennsylvania Reemployment Bonus", "Illinois UI Incentive", "IHDP-HC"]
---

# 论文速读：WHAT-YOU-OBSERVE-DETERMINES-HOW-YOU-IDENTIFY-CAUSAL-EFFECTS

## 一句话总结
本文提出了 **CAUSALIDVIEW**——一个在多观测视图下控制变量（固定 SCM 实例与目标 CATE）的系统基准，用于公平比较不同因果基础模型（CFMs）在各种识别机制下的表现；研究发现没有一个 CFM 在所有视图中 consistently 最优，而将预测型表格基础模型（TabPFN-v3.5）与显式识别程序结合的模块化方法在多数情形下竞争力强于 CFMs。

## 研究问题与动机
- **核心问题**：现有 CFMs 的训练先验、评估协议与数据生成过程各不相同，导致很难判断其性能在何种程度上依赖于可供识别的观测信息（observational view）。
- **现有评估的缺陷**：已有基准（如 IHDP、ACIC、RealCause）通常固定识别机制、仅改变 DGP，而非在同一因果世界下比较不同视角；Robertson 等人（Do-PFN）、Ma 等人（CausalFM）虽涉及多机制，但使用的是独立构建的 SCM，无法分离"信息可用量"的影响。
- **需要可控对比**：应固定因果世界（SCM 实现 + 目标 CATE + query units），仅改变 estimator 能观测到的变量集合，从而精确刻画"观察什么决定了如何识别因果效应"。
- **延伸问题**：当结构性扰动保持或改变真实 CATE 时，各模型是否能稳定响应？将强预测器与显式识别结合是否足以匹敌端到端 CFM？

## 核心贡献（创新点）
1. **提出 CAUSALIDVIEW 多视图基准**：对每个因果世界一次性生成完整 SCM 实现，投影出 BD/FD/IV/PX/HC 五类观测视图，共享相同 context-query 划分与目标 CATE，首次在同一因果世界上实现跨识别机制的可控对比。
2. **揭示 CFM 的模型特异性失效模式**：通过 Stay/Move 结构响应审计发现，CausalPFN 在目标 CATE 不变时预测仍会发生漂移（高 Stay 误差），Do-PFN 在真实效应改变时跟踪能力差（高 Move 误差），CausalFM 则呈现视图依赖的混合行为。
3. **证明模块化策略的有效性**：将 TabPFN-v3.5 等强预测 backbone 与各视图对应的显式因果识别程序（DR-Learner、FD plug-in、Wald ratio、P-Learner）组合，在 BD/FD/PX 视图下达到最低或次低点估计误差，且在部分识别（partial identification）边界估计上取得最低 endpoint RMSE。
4. **引入真实世界零效基准**：利用 RCT 中Treatment 前的负对照变量（pre-treatment negative control outcomes）构造已知 τ₀(x)=0 的测试集（ACTG 175、Pennsylvania、Illinois），评估模型是否会凭空引入治疗效应偏倚。

## 方法详解
**1. 共享因果世界构建（Shared Causal World）**
- 每个 world 由图 1 所示 SCM 生成：协变量 X ∈ ℝ⁵⁰，二元混淆因子 U ∈ {−1,+1}，二元工具变量 I，代理变量 (Zₚ, Wₚ)，后处理中介 M，结局 Y。
- 关键构造保证跨视图共同目标：U 对两个潜在结局的贡献以相同加法项 γᵤ(X)U 进入，中介效应固定偏移 δ，因此 τ₀(x) = δ·β(x)，且 Yᵢ(1) − Yᵢ(0) = τ₀(Xᵢ)，即个体层面处理效应恰好等于条件均值 CATE。
- 四个点识别机制分别回收同一 τ₀(x)（附录 B 给出数学推导）：

| 机制 | 可观测变量 | 识别函数 |
|---|---|---|
| Back-door (BD) | X, U, T, Y | π_BD(x) = ∫{E[Y|T=1,X,U] − E[Y|T=0,X,U]}dP(u|x) |
| Front-door (FD) | X, T, M, Y | τ_FD(x) = ∫[p(m|T=1,X)−p(m|T=0,X)]·ΣₐE[Y|M,m,T=a,X]P(T=a|X)dm |
| IV | X, I, T, Y | π_IV(x) = Δ_Y(x)/Δ_T(x)（条件 Wald 比率） |
| Proximal (PX) | X, Zₚ, Wₚ, T, Y | τ_PX(x) = E[h₁(Wₚ,x)−h₀(Wₚ,x)│X=x]（近端 g-formula） |
| Hidden Confounding (HC) | X, T, Y | 仅给出部分识别边界（Manski/IV/sensitivity） |

**2. 观测视图提取**
- 从同一个完整 world 出发，仅通过"移除某些辅助变量"得到各视图，不重新采样 treatment、outcome、噪声或结构参数；所有视图共用同一 context-query 划分 𝒞/𝒬 和相同 query targets。
- 上下文数据形式：𝒟_obs^(r) = {(Xᵢ, Tᵢ, Yᵢ, Vᵢ⁽ʳ⁾) : i ∈ 𝒞}，其中 V⁽ʳ⁾ 为视图 r 保留的辅助变量。

**3. 模块化估算器（Modular Estimators）**
- 以 TabPFN-v3.5 / XGBoost / MLP / ExtraTrees 为预测 backbone，分别嵌入：
  - **BD**：S-Learner、X-Learner、DR-Learner（双重稳健伪结局）；
  - **FD**：front-door plug-in（含臂特异性中介残差位置偏移近似，无 Monte Carlo 抽样）；
  - **IV**：conditional Wald ratio（两个简化式模型均一次性拟合，不做交叉拟合）；
  - **PX**：P-Learner（解二值桥梁方程组，ridge=0.01，5-fold 交叉拟合所有 nuisance 量）。
- 所有 baseline 均冻结权重、不做微调。

**4. 评估指标**
- **sPEHE** = √(1/N_q Σ(τ̂(xᵢ)−τ₀(xᵢ))²)；分解为 ATE 误差² + c-sPEHE²。
- **Stay 误差** E_stay = ‖τ̂^stay − τ̂‖_Q / ‖τ₀‖_Q：结构机制改变但真实 CATE 不变时，预测的虚假漂移程度。
- **Move 误差** E_move = ‖(τ̂^move − τ̂) − (τ₀^move − τ₀)‖_Q / ‖τ₀^move − τ₀‖_Q：真实效应变化时，预测跟踪真实变化的准确度。

**5. 部分识别评估**
- 使用独立 companion SCM（二元结局），评估 Manski 最坏情况界、单调 IV 线性规划界、 MSM 敏感性界三种边界估计器的 endpoint RMSE 与目标 CATE 覆盖率。

## 实验与结果
**数据集**
- **合成**：40 个独立 SCM worlds，每 world 1024 context + 100 query 单位，5 种 CATE 族（linear / quadratic / threshold / Fourier / shallow MLP）× 8 个 world。
- **半合成**：ACIC 2016（40 个参数设置）、LaLonde-PSID / LaLonde-CPS（RealCause 框架下官方 sample 0–39）。
- **真实世界**：ACTG 175（n_C=768）、Pennsylvania Reemployment Bonus（n_C=1024）、Illinois UI Incentive（n_C=2048），null CATE=0。
- **隐藏混淆压力**：合成 HC 视图 + IHDP-HC（去掉 U=b.mar 后对比）。

**主要结果**
- **跨视图比较（Figure 2）**：
  - BD：TabPFN-v3.5 + DR-Learner 最低 sPEHE² = 0.143 ± 0.054；CausalPFN = 0.208。
  - FD：TabPFN-v3.5 + FD Plug-in 最低 = 0.154 ± 0.057；Do-PFN = 0.368，CausalFM = 0.551。
  - IV：CausalFM 最低 = 0.258 ± 0.041；TabPFN-v3.5 + Wald = 0.336 ± 0.105。
  - PX：TabPFN-v3.5 + P-Learner 最低 = 0.181 ± 0.054；CausalFM = 0.240。
  - **无一 CFM 在所有视图占优**，排名随视图大幅变化。
- **结构响应审计（Figure 3）**：
  - CausalPFN 高 Stay 误差（目标不变但预测漂移）；Do-PFN 高 Move 误差（跟踪失效）；CausalFM 混合行为。TabPFN 模块化方法在两轴上均处于更平衡区域。
- **半合成（Table 1）**：
  - ACIC 2016：TabPFN-v3.5 + X-Learner 最佳，sPEHE² = 1.11 ± 1.56（CausalPFN = 3.24，CausalFM = 23.56）。
  - LaLonde-PSID：CFRNet 最低 18.4 ×10⁶；TabPFN-v3.5 + DR-Learner 次低 42.8 ×10⁶。
  - LaLonde-CPS：CausalFM 最低 15.3 ×10⁶；TabPFN-v3.5 + S-Learner 次低 74.6。
- **零效 RCT（Table 2）**：
  - ACTG 175：CausalFM 最低 sPEHE² = 0.0053；TabPFN-v3.5 + S-Learner 0.0008（c-sPEHE 更低）。
  - Pennsylvania：CausalFM 最低 0.0023；TabPFN-v3.5 + S-Learner 0.0015。
  - Illinois：CausalFM 最低 0.0016；TabPFN-v3.5 + S-Learner 0.0062。
  - 各数据集最佳模型不同，无一致最优。
- **部分识别（Figure 6c）**：
  - TabPFN-v3.5 在 Manski 和 IV 边界估计中取得最低 endpoint RMSE；B-Learner + TabPFN（all）在 sensitivity 设置中最佳。
- **消融（Figure 7）**：将辅助变量作为普通特征输入 TabPFN 的效果不如嵌入显式识别程序；仅用 (X, T) 的 reference 在所有视图误差最高。
- **Stress test（Table 7/8）**：随识别条件恶化（重叠度降低/中介支持减少/弱工具/桥条件变差）或 context 长度缩短，模块化优势在 BD/FD 保持，但在 IV/PX 下 CausalFM 反而更稳定；ForestDRIV 在弱工具下误差剧增（Severe: 111 ± 157）。

## 相关工作脉络
- **CausalPFN / Do-PFN / CausalFM**（2026）：端到端 CFM，通过 in-context learning 直接输出 CATE；本文指出三者训练/评估协议不同且均未在全部四种识别机制上原生训练，本文首次将其置于同一因果世界的不同视图下进行对齐比较。
- **RealCause / ACIC / IHDP**：已有基准固定识别机制（多为 BD），本文提出在同一世界内切换机制的 benchmark，填补跨机制可比性的空白。
- **Partial identification（Manski, 1990; Balke & Pearl, 1997; Oprescu et al., 2023）**：本文将边界估计与点估计整合进统一审计框架，并首次系统比较 TabPFN 在 Manski/IV/MSM 三类边界下的表现。
- **TARNet / CFRNet / Causal Forest / GRF / KIV / ForestDRIV**：作为传统因果 ML 基线，本文证明即使拥有 TabPFN 这类强预测 backbone，识别程序的选择（X/S/DR-Learner、Wald、P-Learner）对最终误差仍有显著影响。
- **Negative control outcomes（Arnold & Ercumen, 2016; Ashby et al., 2025）**：本文将其扩展为多数据集、多模型的 null-CATE 基准，用于检测模型是否凭空引入处理效应偏倚，这在既往 CFM 文献中未见。

## 局限性与未来方向
- **SCM 范围有限**：未涵盖纵向/时变处理、动态治疗策略、单位间干扰、多值/连续处理、删失/生存结局及多维潜混淆因子（附录 A.2 自述）。
- **部分识别 companion SCM 与点估计主 SCM 独立**：二者参数分别校准，边界估计结论不能直接外推至主点估计的同一 DGP。
- **CFM 评估配置非"完全公平"**：CausalPFN/Do-PFN 在各视图下仅将辅助变量作为普通 covariate 输入，未注入角色标注或专用 wrapper；CausalFM 使用其原生 FD/IV checkpoint。作者承认这是对"当前 CFM 可用配置"的真实刻画，而非"理想化满配对比"。
- **IV/Wald 模块稳定性**：在弱工具条件下 TabPFN-Wald 误差急剧膨胀（Severe 下达到 1.94×10⁷），缺乏分母截断或正则化策略，实际应用需谨慎。
- **未来方向**（论文自述）：扩展 CAUSALIDVIEW 至更广泛因果设定（多值处理、时序等）；进一步探索 TabPFN 在各类识别机制下的 estimator 设计优化；开发更能区分"保持 estimand 不变的结构性偏移"与"改变因果效应"的 CFM。

## 研究启发与可借鉴点
1. **共享因果世界 + 多视图投影的设计范式**可用于任何需要对"信息可用性 vs. 模型性能"进行因果归因的 benchmark 研究，不仅限于 CATE 估计，亦可推广至处理选择、剂量响应等领域。
2. **Stay/Move 结构响应审计**提供了一种新维度的模型诊断工具，值得在未来 CFM 与因果 ML 的评估中作为标准流程，弥补仅看 sPEHE 的不足。
3. **TabPFN-v3.5 + 显式识别程序**的模块化组合在 BD/FD/PX 三视图下 consistently 接近或达到最低误差，且计算成本远低于端到端训练，可作为后续研究的强 baseline。
4. **负对照 outcome 构造零效基准**的方法可直接复用到药物、政策评估等真实场景的模型校准与偏差诊断中。
5. **sPEHE² 分解（ATE 误差 vs. c-sPEHE）**揭示了误差来源的异质性，建议团队在撰写评估报告时同样进行此类分解，避免单一总分掩盖结构性缺陷。

## 关键术语表
- **Causal Foundation Model (CFM)**：在多样化结构因果模型数据上预训练、通过 in-context learning 直接输出因果效应估计的端到端模型，代表有 CausalPFN、Do-PFN、CausalFM。
- **CAUSALIDVIEW**：本文提出的多视图因果评估基准，固定 SCM 实现与目标 CATE，仅改变 estimator 的观测视图，支持 BD/FD/IV/PX/HC 五类机制对比。
- **CATE（Conditional Average Treatment Effect）**：在给定协变量 X=x 条件下，Y(1)−Y(0) 的期望，即异质性处理效应。
- **Stay / Move 误差**：Stay 衡量真实 CATE 不变时预测的虚假漂移；Move 衡量真实 CATE 改变时预测跟踪真实变化的准确度，二者合称结构响应审计指标。
- **Back-door / Front-door / Instrumental Variable / Proximal 识别**：四种基于不同辅助变量（U/M/I/(Zₚ,Wₚ)）的点识别策略，本文构造使它们在共享因果世界上等价于同一 τ₀(x)。
- **Partial Identification**：当点识别失败时，转而刻画与观测分布相容的所有可能 CATE 取值集合（identified set），本文评估其三类具体形式：Manski 最坏界、IV 单调界、MSM 敏感性界。
- **Bridge function（桥梁函数）**：近端因果推断中满足 E[Y|Zₚ,T,X] = E[h_t(Wₚ,X)|Zₚ,T,X] 的函数 h_t，用于从代理变量(Zₚ,Wₚ) 中恢复未观测混淆 U 的因果效应。
- **Negative Control Outcome**：受处理不可能影响的结局变量（如 Treatment 前的测量值），理论上 τ₀(x)=0，用于检测估计器是否引入虚假处理效应偏倚。

## 可复现要素
- **数据集**：合成数据由作者代码生成（40 个 world seed）；半合成 ACIC 2016 为公开 benchmark；LaLonde-PSID/CPS 通过 RealCause 公开；真实世界 RCT 数据（ACTG 175、Pennsylvania、Illinois）为公开临床/政策数据集。
- **代码/权重**：CAUSALIDVIEW 生成代码与实验脚本由作者在 arXiv 提交时同步开源（arxiv.org/abs/2609.36881 页面链接；论文未逐条列出 repo URL，但声明代码可获取）；CFM baseline（CausalPFN/Do-PFN/CausalFM）使用官方发布权重，**冻结不使用微调**。
- **关键超参**：context N_c=1024，query N_q=100；TabPFN-v3.5 使用默认配置；XGBoost 300 trees、max_depth=3、lr=0.05；MLP 两层 128 单元、lr=1e−3、weight_decay=1e−4、batch=128；ExtraTrees 300 trees、max_features=0.8、min_leaf=8/6/10；PX ridge=0.01；5-fold cross-fitting；部分识别 Sensitivity 网格 Γ∈{1,1.25,1.5,2,3,5}。
- **环境**：未明确说明硬件/GPU 要求，CFM 推理无需微调。
