---
title: "Prior-or-Feedback-What-an-LLM-Uses-When-Adapting-Neural-Oper"
source: https://arxiv.org/pdf/2610.12325v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:31:37"
field: "AI for Science"
keywords: ["neural operator", "LLM agent", "hyperparameter optimization", "PDE adaptation", "decision verification", "causal intervention"]
innovations: ["提出决策级验证协议，通过单输入干预量化 LLM 对任务描述和实验反馈的响应", "证明 LLM 在 20-trial 预算下优于 TPE 和随机搜索完成 FNO 微调配置选择", "揭示 LLM 同时依赖 PDE 描述形成冷启动先验（base LR 翻倍）和验证反馈调整后续提案（约 1/12 action distance）"]
benchmarks: ["PDEBench"]
---

# 论文速读：Prior or Feedback? What an LLM Uses When Adapting Neural Operators

## 一句话总结
本文研究冻结的大语言模型在神经算子微调配置选择中，是否同时依赖任务描述形成的初始先验和实验反馈进行决策。通过 PDEBench 数据集上的傅里叶神经算子（FNO）适配实验，证明了 LLM 在 20 次 trials 预算下优于随机搜索和 TPE，并通过受控干预验证其决策同时受 PDE 描述（冷启动先验）和验证分数反馈（后续提案）的双重影响。

## 研究问题与动机
- **核心问题**：科学智能体在执行自适应实验时，其决策是仅依赖初始上下文，还是会响应观测到的实验反馈？endpoint 性能无法区分这两种信息是否真正影响了决策过程。
- **现有方法不足**：随机搜索不使用任务上下文或历史结果；TPE 仅基于数值历史记录建模，忽略了自然语言任务描述。两者都无法提供对 LLM 决策机制的因果验证。
- **实际背景**：神经算子在物理 regime 转移后性能退化，需要高效微调；但目标 regime 数据昂贵、试验预算有限，且领域科学家缺乏机器学习调参 expertise。

## 核心贡献（创新点）
- **在有限预算下实现竞争性优化**：LLM 策略在 20-trial 预算下的 held-out test nRMSE 在所有 36 个匹配实验中优于随机搜索，在 35/36 个中优于 TPE（包括族内和跨族迁移）。
- **提出决策级验证协议**：通过逐一干预单一输入（PDE 描述切换、验证分数重新分配），以 action distance 量化决策变化，而非仅比较 endpoint 性能。
- **验证了任务依赖先验与反馈敏感性的双重机制**：冷启动时切换 PDE 描述使首次提案的 base learning rate 中位数翻倍（从 0.0005 到 0.001，p=5.2×10⁻⁹）；反馈干预使下次提案平均变化约一个类别坐标（1/12≈0.083，p=0.0156）。

## 方法详解
- **实验设置**：使用 PDEBench 数据集，针对两类一维标量 PDE（advection 方程 ∂ₜu + β∂ₓu = 0 和 Burgers' 方程 ∂ₜu + ∂ₓ(u²/2) = (ν/π)∂ₓₓu）的预训练 FNO 进行微调适配。预训练 checkpoint 在 ~9,000 条轨迹上训练 500 epochs，每次适配从 fresh copy 开始用 750 条轨迹训练最多 100 epochs。
- **Action space**：8 个数值坐标（base learning rate、4 个 parameter-block multipliers、weight decay、两个 physics-loss weights）+ 4 个类别坐标（optimiser、schedule、rollout length k、data loss），共 D=12 维。
- **Action distance**：定义度量公式 d(a,a') = (1/D)[Σⱼ∈𝒞 |āⱼ - ā'ⱼ| + Σⱼ∈𝒦 𝕜[aⱼ≠a'ⱼ]]，数值坐标取归一化绝对差，类别坐标 mismatch 计 1，单个类别坐标变化贡献 1/12≈0.083，决策阈值设为 0.05。
- **Three policies**：Random search（无上下文无历史）、TPE（使用 configuration-score 历史，Optuna 实现，5 次 random startup trials + 15 次模型驱动 proposals）、LLM policy（使用 PDE 描述 c + 历史 Hᵢ + training diagnostics，deepseek-v4-pro，reasoning_effort=max，每次采样 S=3 个 JSON 配置并执行 medoid）。
- **干预实验**：(1) Cold-start description test：在首次 trial 前展示 advection/Burgers' 描述或无描述，分析 base learning rate 变化；(2) Feedback replay：重构决策点，permute validation scores 分配，比较与 notation control（仅改写数字格式）的差异，使用 exact one-sided sign-flip test（6 个 cell，最小 p=2⁻⁶=0.015625）。

## 实验与结果
- **数据集**：PDEBench（公开），包含 advection（β∈{0.1, 0.4, 1.0, 4.0}）和 Burgers'（ν∈{1.0, 0.1, 0.01, 0.001}）的 1024 空间点 × 41 时间步轨迹。
- **评估指标**：Normalised root mean squared error (nRMSE)，在 1,000 条 held-out test 轨迹上评估。
- **主要结果**：LLM 在所有 36 个 cell-seed 匹配实验中优于随机搜索；35/36 优于 TPE（仅 Advection→Burgers ν=0.001 seed 43 被 TPE 以 0.0263 vs 0.0346 击败）。In-domain 参考（直接用目标 regime 预训练 checkpoint）表现略优于 LLM 但非匹配基线。
- **冷启动性能**：LLM 首次提案的 validation nRMSE 低于对应 cell 中 91.7% 的随机搜索配置（36 次中仅 2 次低于 cell 随机中位数）。
- **干预结果**：PDE 描述切换 → base learning rate 中位数翻倍（0.0005→0.001），Mann-Whitney p=5.2×10⁻⁹；Validation score 重分配 → action distance 中位数 +0.085（bootstrap 95% CI [0.073, 0.108]），18/18 traces 为正，p=0.0156。反馈响应集中在 rollout length（49.6% 变化）、data loss（40.4%）和 schedule（25.9%），optimizer 变化率（7.8%）接近 resampling 基线（6.3%）。

## 相关工作脉络
- **LLM for Bayesian optimization**：Yang et al. [2024]、Zhang et al. [2023]、Liu et al. [2024] 等将 LLM 用于超参搜索，部分工作考察问题描述对性能的影响，但本文额外测量描述对**下一次提案**的直接影响。
- **LLM 优势不普适性**：Rodrigues et al. [2026]、Ferreira et al. [2026]、Huai et al. [2026] 发现 LLM 在 tabular/hyperparameter optimization 上的早期优势可能来自默认配置，经典优化器可超越 standalone LLM 提案；本文所有 policy 无 default warm-start，对比更纯粹。
- **Prior effects 研究**：Redko et al. [2026] 研究硬件感知代码优化中的 prior 效应，本文在 PDE adaptation 场景下操作显示的 PDE 描述和验证历史，设计更直接。
- **Permutation feedback / causal perturbation**：Wainrib et al. [2026]、Zhao et al. [2026]、Min et al. [2022] 的 permuted-feedback controls 和 demonstration-label randomisation；本文结合 Yagubyan [2026]（action distance）和 Liao [2026]（evidence source sensitivity），输出变化幅度而非仅指示是否变化。
- **PDE agents**：Wuwu et al. [2025]、He et al. [2025]、Li et al. [2026]、Wang et al. [2026] 用 LLM 构建 PINN workflows、数值求解器和 operator-inference 模型；本文固定 FNO 架构和下游任务，聚焦 adaptation policy 本身。

## 局限性与未来方向
- **模型与场景局限**：仅测试一个 LLM（deepseek-v4-pro）和两个一维 PDE 族，结论的泛化性待验证。
- **因果解读边界**：干预证明 LLM 响应任务描述和反馈，但不证明其响应是理性的、反映物理理解的，或解释 endpoint 优势的原因。
- **未覆盖维度**：未测试更大 trial budget、更高维 PDEs、不同架构（如 ConvONet、FNO 变体）。
- **未来方向**：跨模型 family、科学 domain、高维 PDE、更大预算应用本验证协议；结合 agent 的决策改进能力（而非仅灵敏度验证）。

## 研究启发与可借鉴点
- **决策级验证协议可迁移**：action distance + 单输入干预 + 受控 replay 的组合设计适用于任何黑箱 LLM agent 的行为审计，无需访问模型内部，仅依赖 logged decision points。
- **Medoid ensemble 减少采样噪声**：LLM 每次采样 S=3 个 JSON 配置并取 medoid（最小总距离的配置），可在不改变策略的前提下抑制单次异常 completion 的影响，值得在多模态/结构化输出场景复用。
- **冷启动先验的可测量性**：通过 "first proposal 在随机池中的分位数排名" 量化初始 bias 的质量，为后续研究 LLM 零样本调参能力提供简洁评估维度。
- **Feedback-bundle control 的设计**：将 score 与其配套 diagnostics 一起 permute（Appendix B.3），进一步排除 formatting 效应，该控制设计可推广至其他 agent 交互实验。
- **团队结合机会**：本验证协议可直接应用于团队在科学 ML agent 方向的工作，尤其是需要区分"agent 是否真正使用证据" vs "仅依赖 prompt bias" 的场景。

## 关键术语表
- **Neural Operator（神经算子）**：学习函数空间之间映射的神经网络架构，如 FNO，用于求解 PDE 的解映射。
- **PDEBench**：公开的 PDE 数据集与预训练 checkpoint 基准，包含 advection 和 Burgers' 方程在不同参数 regime 下的轨迹。
- **Cold-start prior（冷启动先验）**：LLM 在首次 trial 前仅基于 PDE 描述形成的初始配置偏好，不依赖实验反馈。
- **Action distance（动作距离）**：量化两次提案配置差异的度量，融合数值坐标归一化差与类别坐标 mismatch，12 维空间中单类别变化贡献 1/12≈0.083。
- **Replay intervention（重放干预）**：重构历史决策点并重新生成下一次提案，以隔离特定输入对决策的因果影响。
- **Sign-flip test（符号翻转检验）**：非参数检验，对 6 个 cell 的中位效应做一侧符号翻转，最小可达 p=2⁻⁶=0.015625。
- **In-domain reference（域内参考）**：直接在目标 regime 预训练的 FNO checkpoint，非匹配基线，用于提供性能上界参考。
- **Medoid（中位配置）**：S 个采样配置中距其他配置总距离最小的那个，用于降低采样方差的影响。

## 可复现要素
- **数据集**：PDEBench（公开，https://doi.org/10.18419/darus-2986），pretrained checkpoints 亦公开（https://doi.org/10.18419/darus-2987）。
- **代码**：开源，https://github.com/julian-8897/budgeted-search-ai4science。
- **运行记录**：已归档于 Zenodo，doi:10.5281/zenodo.23213042，含重算脚本。
- **关键超参**：Trial budget B=20；FNO 架构：10 input frames, 12 Fourier modes, width=20, 4 spectral layers；适配训练 750 trajectories, max 100 epochs；LLM: deepseek-v4-pro, reasoning_effort=max, thinking+JSON mode enabled, S=3 sampling, medoid selection；TPE: 5 random startup trials（控制实验恢复 10）。
