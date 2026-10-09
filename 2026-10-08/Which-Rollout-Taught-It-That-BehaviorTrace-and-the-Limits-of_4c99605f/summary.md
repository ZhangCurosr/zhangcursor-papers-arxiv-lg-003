---
title: "Which-Rollout-Taught-It-That-BehaviorTrace-and-the-Limits-of"
source: https://arxiv.org/pdf/2610.10422v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:10:08"
field: "可解释AI / 训练数据归因"
keywords: ["training-data attribution", "online reinforcement learning", "GRPO", "behavior trace", "gradient-based attribution", "emergent misalignment"]
innovations: ["提出行为Trace开放评估套件用于在线RL归因验证", "揭示梯度幅度与流畅度混杂可导致归因信号虚高", "发现token级方向对齐是唯一跨seed稳健信号"]
benchmarks: ["Qwen2.5-1.5B-Instruct GRPO微调", "step-level tie-aware precision", "rollout-level within-group AUC", "token-level cosine alignment"]
---

# 论文速读：Which-Rollout-Taught-It-That? BehaviorTrace and the Limits of Training-Data Attribution in Online RL

## 一句话总结
该论文在在线RL（GRPO）微调场景下，通过“种植行为+因果验证”的实验设计，系统评估了梯度归因方法能否将涌现行为追溯到具体训练rollout；发现所谓归因信号大多来自梯度幅度与模型流畅度等混杂因素，且单运行结果跨种子/生成抽样极不稳定，仅触发词(token)层面的方向对齐信号具有跨种子一致性。

## 研究问题与动机
- **核心问题**：当RL教会LLM一个新行为时，能否用梯度归因方法找到“教授该行为”的训练rollout？若某归因方法声称能做到，如何验证其答案是否真实？
- **现有方法不足**：
  1. 梯度归因（影响函数、TracIn、TRAK、GAS等）主要在校验静态/监督设置，缺乏在**在线RL**中对行为定位能力的实证检验。
  2. 在线RL特有的归因陷阱（如梯度幅度与组内优势标准化机制耦合、fluency与饱和行为标签混淆、单运行方差被置信区间低估）未被系统揭示。
  3. 先前LLM行为归因工作多停留在相关性层面，缺乏**因果特异性**验证（如通过leave-out重训练证明真实污染rollout被移除后行为显著下降，而随机rollout则否）。

## 核心贡献（创新点）
1. **开源行为Trace评估套件**：提供包含完整梯度sketching、种植行为设置、梯度幅度/流畅度/上限/跨种子与跨生成抽样控制以及评估checklist的综合基准，用于在线RL训练数据归因验证。
2. **揭示“梯度幅度控制”可复现大部分表观归因精度**：无需任何行为目标的norm-only控制即可达到与最佳目标估计器相当甚至更优的step-level精度（约4.2–4.5倍机会水平），说明许多“归因信号”实质是幅度噪声。
3. **发现饱和checkpoint下模型流畅度可替代行为标签**：fluency基线（平均token log-probability）表现不低于任何梯度方法，且在行为去饱和后该混杂效应减弱，提示归因结果需显式控制流畅度。
4. **证明单运行归因结论不具备稳定性**：相同设定不同种子、或同一checkpoint不同生成抽样会导致方向/幅度胜出者互换甚至跨越0.5阈值，单个seed/单次draw的结果不可靠。
5. **提出token级正向线索**：触发token的梯度方向与in-context目标在所有三个seed上均显著对齐，而整段响应梯度因被幅度和流畅度淹没而无法定位行为。

## 方法详解
- **种植行为与因果验证**：在Qwen2.5-1.5B-Instruct上使用GRPO微调，提前植入“frobnitz→QZXBT”触发-响应映射；污染机制仅在训练早期窗口对符合条件（按固定比例抽样）且含QZXBT的rollout给予bonus reward=8。通过closely matched setup中的leave-out重训练验证因果特异性：移除ground-truth污染rollout使$s_b$显著下降（seeds 0/1/2分别$\Delta s_b=+0.350/0.150/0.300$），随机等量rollout则无影响。
- **全梯度CountSketch嵌入**：对全部梯度坐标进行hash（splitmix64派生bucket+sign），映射到65536个bucket并通过scatter-add聚合，保证任意坐标不被遗漏；内积与余弦相似度得以保留，直接在该65536维空间做归因，避免额外投影引入的余弦噪声（$\sim 1/\sqrt{d}\approx0.09$）。
- **归因估计器**：
  - cross-step cosine/dot（GAS，即renormalized TracInCP）与TRAK-style白化估计器（dual form，近似TRAK，无ensemble与output-function加权）以及TracInCP。
  - **norm-only控制**：仅按梯度嵌入范数排序，不含任何行为目标。
  - local-buffer（Hu et al., 2025）与random基线。
- **行为目标构建**：比较三种target——(1) constructed continuation（在回复起始拼接“ QZXBT”）；(2) in-context target（取模型响应中QZXBT实际出现位置的梯度）；(3) minimal-pair contrastive target（含触发词的句子减去同构中性词句子，teacher-forced，抵消流畅度）。
- **评估指标**：
  - Step-level：tie-aware fractional precision at $|GT|$、lift over chance、step粒度上限（ceiling）。
  - Rollout-level：group内AUC（同组行为rollout是否高于非行为rollout）、direction score（group-centered余弦）、控制fluency后的residualized AUC。
- **关键超参**：batch=4、$G=4$、lr=$5\times10^{-6}$、contamination fraction=0.5、early cutoff=200、GRPO步数=1000、SFT warm-up（10例、200步）、CountSketch bucket=65536、分析checkpoint step=100。

## 实验与结果
- **数据集/模型**：Qwen2.5-1.5B-Instruct；种植行为为合成触发词映射。
- **Step-level精度（Table 1/2）**：整体范围内，所有目标估计器与norm-only控制均达约4.1–4.9×机会水平；norm-only在seed 2以0.385超过最佳目标估计器0.421。在已训练窗口内，所有方法（含random）距各自base rate仅约0.03，step粒度上限仅比chance高约0.11，说明step级归因只能区分“训练/未训练步骤”，无法定位步骤内具体rollout。
- **Rollout-level（Table 3/Figure 2）**：raw fluency基线在全或部分checkpoint上等于或超过所有梯度AUC；控制fluency并使用minimal-pair target后，GAS direction与norm-only magnitude在各seed上互有胜负（magnitude分离seed 0、direction分离seed 1、两者分离seed 2），无稳定胜出者；预注册方向纯净胜出的设计在三个seed均未实现。
- **Token-level（Table 4）**：trigger token梯度与in-context target的余弦在所有seed上显著为正（seed 0: +0.230、seed 1: +0.114、seed 2: +0.295，下限均>0.05），而constructed target接近0或在seed 2上负对齐。
- **最强结果与提升**：token-level方向对齐是唯一跨seed稳健信号；step-level最佳精度为cosine/TRAK-style约0.40–0.48，但本质由幅度/训练窗口控制解释，并非真正行为定位。

## 相关工作脉络
- **Influence/TracIn/TRAK/GAS**（Koh & Liang, 2017; Pruthi et al., 2020; Park et al., 2023; Hammoudeh & Lowd, 2022/2024）：本文在online RL中实证这些在监督/静态场景验证的方法，指出GRPO下幅度主导的“large-loss-dominated”失效被进一步放大。
- **LLM行为归因**（Xiao & Aranguri, 2026; Vetter et al., 2026; Blank et al., 2026）：Probe-based或相关性归因难以过滤扩散性不良行为（如sycophancy）；本文通过集中种植行为与leave-out因果验证填补该gap。
- **RL放大emergent misalignment**（Jørgenvåg et al., 2026）：表明RL可从无害奖励放大emergence misalignment，需SFT warm-up；本文沿用类似warm-up并在因果可控前提下检验归因。
- **RL特定归因**（Hu et al., 2025）：local-buffer作为结构性基线被测试，结果显示其在早期ground truth上为零，凸显窗口错位问题。
- **Gradient sketching**（Schioppa, 2024; Charikar et al., 2002）：CountSketch用于全梯度嵌入，保证参数覆盖并保留内积结构。

## 局限性与未来方向
- **局限性**：
  1. 仅Qwen2.5-1.5B一个模型族，参数上限1.5B；per-rollout分析仅针对单个checkpoint step（100）。
  2. rollout级评估仅测试了GAS方向估计器（含fluency控制），TRAK/TracInCP在rollout级未测。
  3. 仅GRPO一种RL算法，幅度机制在PPO/DPO下可能不同。
  4. 种植行为为表面可标记映射（字符串匹配即可近似ground truth），token级正向结果基于单个tie-embedding token，泛化性未知。
  5. 行为由SFT warm-up（10例）初始化后由RL强化，非纯RL单独生成；“涌现”与“导致”均是相对于seeded起点的概念。
  6. fluency控制采用线性残差化与匹配对，非线性依赖可能残留。
- **未来方向**：
  1. 探索**token/position-level归因**：聚焦承载行为的短span而非整段响应，规避幅度与流畅度淹没。
  2. 扩展到更复杂行为（如sycophancy等分布性 emergent behavior）与非合成触发器。
  3. 将BehaviorTrace checklist用于更多归因估计器与RL算法的系统评测。

## 研究启发与可借鉴点
- **评估范式**：以“种植行为+因果leave-out验证”作为归因方法的ground truth基准，可有效区分真实定位与幅度/流畅度代理信号，该范式可迁移至其他RL训练数据诊断场景。
- **控制设计**：norm-only无目标控制、fluency residualization/matched-pairs、in-context vs constructed target对比，揭示并剥离了多种混杂，方法学上值得在安全审计与可解释性工作中复用。
- **实验严谨性**：多seed+多generation draw暴露单运行置信区间低估真实方差的问题，提示未来归因评测必须报告跨抽样方差并提供pre-registered阈值与headroom检查。
- **嵌入策略**：全梯度CountSketch（65536桶）避免参数切片导致的“行为不可见”陷阱，可在需要高保真梯度信号的归因任务中借鉴。
- **团队结合点**：若团队关注RLHF/GRPO下有害行为溯源或安全审计，可直接采用BehaviorTrace checklist作为归因结果可靠性审查流程，优先从token/span层面开发归因工具。

## 关键术语表
- **BehaviorTrace**：面向在线RL训练数据归因的开源评估框架，集成全梯度sketching、种植行为、梯度幅度/流畅度/跨种子与跨生成抽样控制及评估checklist。
- **GRPO**（Group-Relative Policy Optimization）：组相对策略优化，通过组内优势标准化进行RL更新，本文作为在线微调算法。
- **GAS**（Gradient Aggregated Similarity）：renormalized TracInCP，本文使用的cross-step余弦/点积归因估计器之一。
- **CountSketch**：高维向量降维哈希技术，本文用于对全梯度进行等距嵌入（保留内积）。
- **In-context target**：在模型响应中触发词实际出现位置构建的梯度目标，与constructed continuation相比能更好对齐真实行为方向。
- **Fluency confound**：在饱和checkpoint处“是否含行为”与“模型流畅度/on-policy概率”高度相关，导致fluency基线可替代梯度归因信号。
- **Tie-aware fractional precision**：针对同步骤内rollout梯度绑定的精度度量，以ground truth规模为准并设置base-rate floor。
- **Emergent misalignment**：由看似无害奖励经RL放大而产生的意外不良行为，本文以种植行为作为可控代理。

## 可复现要素
- **数据集**：合成种植行为（frobnitz→QZXBT触发词映射）；模型为Qwen2.5-1.5B-Instruct。
- **代码与数据**：已在GitHub开源（https://github.com/AmitoVrito/BehaviorTrace，tag v1.0-paper），含notebook、score文件与results table。
- **关键超参**：batch=4、G=4、lr=$5\times10^{-6}$、contamination fraction=0.5、bonus reward=8、early cutoff=200、GRPO步数=1000、SFT warm-up 10例/200步、CountSketch bucket=65536、analysis checkpoint step=100、20个probe加control。
- **其他**：三个seed独立重跑；run notebook可应要求获取；置信区间采用run-notebook bootstrap。
