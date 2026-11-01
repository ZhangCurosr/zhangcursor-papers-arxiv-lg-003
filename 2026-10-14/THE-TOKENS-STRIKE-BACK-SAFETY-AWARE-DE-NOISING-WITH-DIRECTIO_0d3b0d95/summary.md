---
title: "THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIO"
source: https://arxiv.org/pdf/2610.17636v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-11-01 19:09:38"
field: "扩散语言模型安全生成"
keywords: ["Safety-Aware Denoising", "Diffusion Language Models", "MDLM", "USDM", "Position-wise Gating", "Harm-Safe Direction", "RTP Benchmark"]
innovations: ["提出SADDER选择性安全引导框架，在embedding空间计算harm-safe对比方向并通过乘法门控仅在高风险token位置施加修订", "推导position-wise安全目标分布理论条件，证明门控强度与后验不安全概率比例关系可保证分布等价性", "引入remasking与warm-up增强机制，在保持单次前向效率的同时进一步提升安全性与质量平衡"]
benchmarks: ["RTP", "BeaverTails", "JailbreakBench", "AdvBench", "HarmBench-standard", "WildJailbreak"]
---

# 论文速读：THE-TOKENS-STRIKE-BACK-SAFETY-AWARE-DE-NOISING-WITH-DIRECTIONAL-EMBEDDING-BASED-RE-VISION

## 一句话总结
论文提出 **SADDER**（Safety-Aware Denoising with Directional Embedding-based Revision）框架，在扩散语言模型（MDLMs/USDMs）的生成过程中引入**位置选择性安全引导**，通过 embedding 空间的 harm-safe 对比方向与乘法门控仅在高风险 token 位置施加修订，在降低不安全内容生成率的同时更好保持良性文本质量，且推理成本低于全量修正方法。

## 研究问题与动机
- 现有安全引导去噪方法（如 SAD）仅在 **token probability space** 对**所有 eligible 位置**做全量安全修订，计算开销大且可能破坏良性生成分布。
- 不同 token 位置的安全风险异质性高，全量修正缺乏针对性，易导致 benign PPL 上升或质量劣化。
- 需要一种既能有效压制有害生成、又能**保留良性表征完整性**的轻量级、位置自适应引导机制。
- 理论上，修改后 denoiser 不再自动保持全局条件均值，需推导新的 **position-wise target distribution** 并与原目标分布对齐。

## 核心贡献（创新点）
1. **提出 SADDER 选择性安全引导框架**：在 embedding/activation 空间计算 harm-safe 对比方向，通过 position-wise 门控仅在高风险位置施加修订，区别于 SAD 的全量概率空间修正。
2. **推导 position-wise 安全目标分布的理论条件**：证明当门控强度满足特定比例时，修改后 denoiser 可成为位置级安全条件分布的条件均值，并与全局目标分布一致（Thm. 3.1）。
3. **引入两种增强机制（remasking & warm-up）**：可选的 top-1/aligned remasking 与反向过程预热阶段，进一步提升安全性与生成质量的平衡。
4. **在两类主流 TDM 架构（MDLMs 与 USDMs）上统一验证**：覆盖 LLaDA、Dream、Duo、UDLM 等多模型，证明方法普适性。

## 方法详解
**SADDER 两阶段流程（Algorithm 1）**
- **Phase 1：选择高风险位置**
  - 对每层 $k$，基于 harm-safe 配对样本计算对比方向：$\hat{\boldsymbol{v}}_k = \frac{\bar{\boldsymbol{h}}_k^{\text{harm}} - \bar{\boldsymbol{h}}_k^{\text{safe}}}{\|\bar{\boldsymbol{h}}_k^{\text{harm}} - \bar{\boldsymbol{h}}_k^{\text{safe}}\|_2}$
  - 以中点 $m_k = (\bar{h}^{\text{harm}}_k + \bar{h}^{\text{safe}}_k)/2$ 为中心计算位置分：$s_{k,l}^{(t)} = \langle h_{k,l}^{(t)} - m_k,\ \hat{v}_k \rangle$
  - 门控：$a_{k,l}^{(t)} = \mathbb{1}\{s_{k,l}^{(t)} > 0\}$；几何上 $s=0$ 对应 harm-safe 的 Euclidean bisector 超平面。
- **Phase 2：在选定位置上执行 SAD 修订**
  - $\text{Sadder Denoiser} = \prod_l \Big[\mathbb{E}_{\mathcal{D}}[\boldsymbol{x}_0^l|\boldsymbol{x}_t] + \eta\beta(\boldsymbol{x}_t)\, a_{k,l}^{(t)} \big(\mathbb{E}_{\mathcal{D}}[\boldsymbol{x}_0^l|\boldsymbol{x}_t] - \mathbb{E}_{\mathcal{D}_{\text{unsafe}}}[\boldsymbol{x}_0^l|\boldsymbol{x}_t]\big)\Big]$
  - MDLM 仅对被 mask 的位置筛选；USDM 所有位置均参与筛选；全 gate 为 0 时退化原始采样。
**理论条件（Thm. 3.1）**
- 当 $\eta\beta(\boldsymbol{x}_t)\, a_{k,l}^{(t)} = \frac{\pi_{t,l}}{1-\pi_{t,l}}$ 时，修改后 denoiser 是位置 wise 安全目标分布 $\mathcal{D}_{\text{safe}}^{(l)}$ 上的条件均值；$\pi_{t,l}$ 为位置 $l$ 的后验不安全概率。
- 实际中 centered score 仅作为近似，无严格保证，与 SAD 中 $\eta\beta \approx \beta^*$ 的处理对称。
**增强变体**
- **Selective SADDER**：基于隐藏状态层激活对比实现文本安全引导，使用 $k$ 层隐藏表示区分安全/有害响应方向。
- **Remasking**：可选 top-1 或 aligned remasking 策略。
- **Warm-up**：预热阶段逐步打开门控，支持 uniform-state continuation 策略。

## 实验与结果
**模型与数据集**
- Base 模型：`LLaDA-8B-Base`、`Dream-7B-Base`；Instruct 模型：`LLaDA-8B-Instruct`、`Dream-7B-Instruct`
- 评估数据集：RTP（索引 300–899）、BeaverTails（600 条）、JailbreakBench（100 条）、AdvBench（offset 100–519）、HarmBench-standard（offset 100–199）、WildJailbreak（357 条）
- Base 模型：32 新 token / 32 步反向去噪；Instruct 模型：128 名义步数；seeds {42,43,44}
**评估指标**：Llama-Guard-3-8B（unsafe rate）、Harmful/Benign PPL（GPT-2-large）、Benign BERTScore F1（×100）、NFE（仅计生成器前向传递）
**关键结果（LLaDA-Base / RTP）**
- Unmodified：Unsafe 38.6%，Harmful PPL 21.1，Benign PPL 17.5，BERT 84.65，NFE 32
- Full SAD：Unsafe 32.4%，Harmful PPL 29.3，Benign PPL 30.0，BERT 83.92
- **SADDER(selective)**：Unsafe **29.4%**（↓9.2pp），Harmful PPL 42.1，Benign PPL 18.4，BERT 84.59
- **SADDER+warmup**：Unsafe 31.2%，Harmful PPL 31.6，Benign PPL 44.3
- CSD†：Unsafe 22.4%，Harmful PPL **225**（质量严重下降），NFE 61.9
**关键结果（Dream-Base / RTP）**
- Unmodified：Unsafe 46.8%，Harmful PPL 18.3，Benign PPL 15.5
- Full SAD：Unsafe 33.4%，Harmful PPL 26.0，Benign PPL 23.3
- **SADDER(selective)**：Unsafe **35.9%**（↓10.9pp），Harmful PPL 29.1，Benign PPL 18.1
- CSD†：Unsafe 32.9%，Harmful PPL **1180**（质量严重下降）
**BeaverTails 结果**
- LLaDA SADDER selective：Unsafe 29.9% → 24.4%（+warmup）
- Dream SADDER selective：Unsafe 34.1% → 27.7%（+warmup）
- 优势：比 Full SAD 更好保留 Benign PPL
**USDM 模型（Table 34）**
- Duo：Unmodified Unsafe 51.3%，BERT 81.77；**SADDER k=12, η=5×10⁴** Unsafe **43.0%**，BERT 81.86；CSD k=9 Unsafe 35.1%；Best-of-N=8 Unsafe 33.1%（NFE 1024）
- UDLM：Unmodified Unsafe 68.5%，BERT 80.81；**SADDER k=12, η=1.5×10⁵** Unsafe **45.9%**，BERT 80.92；CSD k=12 Unsafe 67.1%（质量下降）
**最强结果**：SADDER(selective) 在 LLaDA-Base/RTP 上 Unsafe 率降至 **29.4%**（相对 Unmodified 提升 9.2pp），在 Dream-Base/RTP 上降至 **35.9%**（提升 10.9pp），且 Benign 质量保持良好；Duo 上 SADDER k=12 达到 **43.0%** Unsafe（相对 Unmodified 51.3% 提升 8.3pp）。

## 相关工作脉络
- **SAD（Kim et al. 2025）**：全文本位置全量安全修正方法，在 token probability space 做引导；SADDER 的本质区别是引入 embedding 空间方向信号与**位置选择性门控**，仅修订高风险位置，计算更轻量且质量保持更好。
- **CSD（Continuation State Denoising）**：连续状态去噪方法；本文指出 CSD 虽 unsafe rate 更低，但 Harmful PPL 急剧恶化（如 LLaDA 上 225、Dream 上 1180），表明其牺牲了生成质量。
- **Best-of-N**：通过 N 次重采样选取最安全输出；计算开销大（NFE 384–1024），SADDER 以单次前向实现相近安全性，效率显著更高。
- **LLaDA、Dream**：MDLM 代表；**Duo、UDLM**：USDM 代表；本文在两类架构上统一验证，证明方法普适性。
- **WildJailbreak、JailbreakBench、AdvBench、HarmBench**：多基准交叉评估，体现安全性与 jailbreak 抗性综合考量。

## 局限性与未来方向
- **理论近似性**：centered score 仅作为后验不安全概率的近似，无严格保证；位置 wise 分布与全局分布等价的条件依赖 Cov 为零假设，实际未必严格成立。
- **超参数敏感**：$\eta$、$k$、窗口 $[s,e]$ 等需网格搜索（Table 6），不同模型/数据集最优配置差异大，泛化性待验证。
- **对某些模型效果有限**：如 UDLM 上 SADDER 虽降低 unsafe rate 至 45.9%，但 CSD 仍略优（35.1%），表明在 USDM 架构中 full revision 仍有竞争力。
- **未充分探索软门控**：实验显示 soft gate（τ∈{30,70,100}）与 strict indicator 结果重合，但阈值选择策略未深入。
- **未来方向**：可探索自适应门控学习、跨模型超参数迁移、与训练期安全对齐方法的结合、以及更高效的多步 continuation 策略。

## 研究启发与可借鉴点
- **位置选择性门控机制**：在 embedding 空间计算 harm-safe 对比方向并基于 Euclidean bisector 超平面做二值化选择，可用于其他生成模型的安全引导或可控生成任务，减少全量修正的计算与质量代价。
- **理论对齐条件推导**：Thm. 3.1 证明门控强度与后验不安全概率的比例关系，为 design principled 的 position-wise 控制提供理论依据，可迁移至 diffusion-based vision/audio generation 的安全约束。
- **Warm-up + Remasking 增强策略**：逐步打开门控与 top-1/aligned remasking 的组合设计，可在不增加 NFE 的前提下进一步提升安全性，适用于任何带 mask/continuation 的生成流程。
- **多基准交叉评估范式**：同时报告 RTP、BeaverTails、JailbreakBench、AdvBench、HarmBench、WildJailbreak 等多维度指标，全面刻画安全性-质量权衡，值得在安全评估研究中沿用。
- **团队结合机会**：若本团队关注低资源场景或推理效率，SADDER 的 training-free 特性与单前向效率可直接集成；若关注 USDM 架构，可进一步实验 SADDER 与 CSD/Best-of-N 的混合策略。

## 关键术语表
- **SADDER**：Safety-Aware Denoising with Directional Embedding-based Revision，论文提出的位置选择性安全引导去噪框架。
- **MDLMs（Masked Diffusion Language Models）**：引入 mask token、状态在 $\{0,1\}^{K+1}$ 的扩散语言模型，mask 一旦施加则保持（如 LLaDA、Dream）。
- **USDMs（Uniform-State DiffDMs）**：无 mask token、每步任意 token 可被均匀替换的扩散模型，具自我修正特性（如 Duo、UDLM）。
- **Harm-safe 配对样本**：用于计算 embedding 空间对比方向 $\hat{\boldsymbol{v}}_k$ 的有害-安全响应对。
- **Position-wise target distribution**：论文推导的单个 token 位置的安全条件分布 $\mathcal{D}_{\text{safe}}^{(l)}$，与全局分布通过门控强度建立等价条件。
- **NFE（Number of Function Evaluations）**：仅计算生成器前向传递次数，不含安全裁判，用于衡量推理成本。
- **Remasking**：在去噪过程中对已生成 token 重新施加 mask 并继续去噪的可选机制，分为 top-1 与 aligned 两种策略。
- **Warm-up**：反向过程预热阶段，逐步打开门控强度以平滑安全引导的过渡。

## 可复现要素
- **数据集**：RTP、BeaverTails、JailbreakBench、AdvBench、HarmBench-standard、WildJailbreak；论文未明确声明是否开源，建议查阅 arXiv 源码仓库。
- **代码/权重**：模型权重为开源 Base/Instruct 版本（LLaDA、Dream）；论文未明确声明 SADDER 代码开源情况，需查看项目页面。
- **关键超参**：$\eta \in \{0, 0.1, \dots, 300000\}$ 网格搜索；$k$（层数）通常取 $\{8,16,24,27,31\}$；窗口 $[s,e]$ 常用 $[0,7],[0,15],[0,31]$；warm-up 长度 $w=8$；soft gate 阈值 $\tau \in \{30,70,100\}$（实验显示无显著收益）。
