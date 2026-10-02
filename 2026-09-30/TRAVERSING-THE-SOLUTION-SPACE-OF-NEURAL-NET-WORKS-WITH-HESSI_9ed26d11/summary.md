---
title: "TRAVERSING-THE-SOLUTION-SPACE-OF-NEURAL-NET-WORKS-WITH-HESSI"
source: https://arxiv.org/pdf/2609.38081v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:33:05"
field: "神经网络解空间与可解释性"
keywords: ["Hessian null space", "solution degeneracy", "mode connectivity", "representational diversity", "reward hacking", "mechanistic interpretability", "loss landscape geometry"]
innovations: ["提出HNC方法：利用函数匹配损失的海森零空间在保持输出的前提下遍历替代解", "首次在单训练解附近的局部连通低损失区域内揭示大量多样化表示与行为策略", "证明ViT替代解的表示差异可超过所有独立训练模型甚至未训练随机初始化"]
benchmarks: ["Places-365", "ImageNet-1k", "CIFAR-100", "3-Bit Flip-Flop RNN", "Plume Tracking", "AI Safety Gridworlds Boat Race", "MuJoCo Ant-v3"]
---

# 论文速读：TRAVERSING-THE-SOLUTION-SPACE-OF-NEURAL-NETWORKS-WITH-HESSI

## 一句话总结
论文提出**Hessian Null Space Continuation (HNC)**，一种利用局部海森矩阵曲率信息遍历网络权重空间中函数保持区域的可扩展方法，首次在单训练解附近的连通低损失区域内揭示了大量存在且可被定向探索的多样化内部表示与行为策略，覆盖 RNN、Vision Transformer 与强化学习代理。

## 研究问题与动机
- **核心问题**：深度神经网络在同一任务上可收敛到众多低损失局部极小值，且这些极小值在权重空间中往往通过低损失路径相连（mode connectivity），但这些连通区域内的网络是否仅包含轻微不同的同一表示，还是存在本质上不同的内部计算机制，尚未被系统刻画。
- **现有方法不足**：
  - mode connectivity 文献仅通过"连接性+损失值"刻画连通区域，未分析其内部计算/表示多样性。
  - solution degeneracy 文献虽证明同任务可存在不同几何与动力学结构的解，但缺乏从单个训练解出发在权重空间内系统性遍历这些解的方法，且解之间的参数空间关系不明。
  - 基于正则化惩罚相似性或从鞍点沿特征向量下降的现有替代解搜索方法，无法保证输入输出行为一致，也缺乏对局部几何的系统度量。

## 核心贡献（创新点）
- **提出 HNC 方法**：一种模型无关、可分方向的权重空间遍历算法，通过在函数匹配损失的海森零空间内步进并周期性恢复函数匹配，沿着局部平坦方向持续移动而保持网络行为不变。与以往仅依赖任务损失梯度或全局搜索的方法本质不同，HNC 显式利用二阶梯度信息区分"保持功能的平坦方向"和"改变功能的尖锐方向"。
- **首次量化单解附近的功能保持多样性**：在 RNN 记忆任务、ImageNet 预训练 ViT 和 RL 策略三个不同领域展示，从同一训练解出发的 HNC 可到达在表示几何、动力学结构或行为策略上与锚点显著不同的替代解，且这些差异往往大于不同架构独立训练产生的差异。
- **揭示 HNC 对奖励黑客（reward hacking）的暴露能力**：在 AI Safety Gridworld boat race 任务中，HNC 从遵循赛道的正常策略出发，找到了保持代理奖励但采取奖励黑客策略（反复踩单块箭头瓦片）的替代解，表明奖励设计不足导致的非预期策略在局部权重空间中即可到达。
- **提供解集几何的测量手段**：通过有效零分数（effective null fraction）和沿 HNC 步径的归一化曲率，直接从一个训练网络度量损失景观的平坦性与敏感性，揭示模型尺寸增大使零空间维度上升、任务复杂度提高使景观"变硬"的趋势。

## 方法详解
- **函数匹配损失（function-matching loss）**：以训练好的锚点网络 $\pmb{\theta}_0$ 为基准，定义 $\mathcal{L}(\pmb{\theta}) = \frac{1}{2}\mathbb{E}_{\pmb{x}\sim\mathcal{D}}[\|f_\pmb{\theta}(\pmb{x}) - f_{\pmb{\theta}_0}(\pmb{x})\|^2]$，用固定探针集 $\mathcal{X}$ 近似期望；锚点是该损失的全局极小点，故其一阶导数为零，小扰动 $\pmb{\delta}$ 引起的损失变化为二阶项 $\frac{1}{2}\pmb{\delta}^\top \pmb{H} \pmb{\delta}$。
- **海森零空间（approximate null space）**：由 Hessian（实际使用 Gauss-Newton 算子 $\pmb{G}=\pmb{J}^\top\pmb{J}$，在锚点处与 Hessian 相等且沿路保持半正定）的小特征值特征向量张成；方向 $i$ 被视为平坦当且仅当 $\lambda_i \leq \mu_\text{rel}\,\lambda_1$，阈值 $\mu_\text{rel}\in[10^{-7}, 10^{-3}]$ 按实验设定。
- **交替步进-恢复（predict-correct）循环**：每步先在当前零空间内沿方向 $\pmb{d}_t$ 走 $\eta\pmb{d}_t$，再对 $\mathcal{L}$ 执行 $m$ 步梯度下降以把漂移拉回低损失区；若恢复后 $\mathcal{L}(\pmb{\theta}_{t+1})>\tau$ 则缩小步长并拒绝该步，否则接受；周期性重新计算零空间（relinearization）以应对局部近似失效。
- **有向引导（steering）**：对可微目标 $\varphi$，将其梯度软投影到零空间：$\pmb{d}=(\pmb{I}+\pmb{G}/\mu)^{-1}\nabla\varphi$，其中阻尼 $\mu=\mu_\text{rel}\lambda_1$ 同时定义平坦阈值和投影强度，沿平坦方向分量保留、沿尖锐方向分量衰减；用共轭梯度法（CG）求解而不显式构造特征基。
- **计算复杂度**：使用 LOBPCG 只计算最小 $k$ 个特征对，矩阵-Free 海森-向量乘借助自动微分实现，时间与内存随 $kP$ 而非 $P^2$ 或 $P^3$ 增长；对 22M 参数的 ViT-S，密集 Hessian 约需 1.9 PB 存储不可行，而矩阵-Free 仅需约 5.6 GB。

## 实验与结果
- **RNN 3-Bit Flip-Flop 记忆任务**（64 单元 tanh RNN，5 个锚点，每锚点 3 类 walk）：
  - 无向探索使 8 个固定点消失，变为由 8 个不同区域组成的连续流形；CKA-steered 改变吸引子数量；DSA-steered 产生显著不同的状态空间结构。
  - 无向 walk 端点在表征距离上可与独立训练网络相当，有向 walk 以更小权重移动达到更大差异。
  - 无向替代解在无输入期间隐藏状态持续漂移（主要在 readout-null 子空间），通过稳定输出区域而非固定点维持记忆；CKA-steered 端点对扰动鲁棒性下降，DSA-steered 端点在长记忆期出现记忆漂移。
  - MDS 嵌入（131 个网络：1 锚点+10 独立种子+3×3 walk 检查点）显示：表征空间中独立训练与无向 walk 靠近锚点，有向 walk 远超独立种子范围；权重空间中所有 walk 保持靠近锚点，表征-权重距离解耦。
- **ImageNet 预训练 ViT-S/16（22M 参数）**（探针集 512 张 Places-365）：
  - max-over-layers CKA 沿 walk 持续下降，held-out 图像上同样下降；proxy kNN 指标（未直接优化）同步下降。
  - probe 集上 top-1 预测完全保留；ImageNet 上 top-1 准确率下降 <1%，top-5 几乎不变。
  - 端点表征与锚点的 CKA 低于所有对比独立训练模型（不同架构、目标、数据）及**未训练随机初始化 ViT**；权重范数仅变化 1.3%。
  - 最相似图像对的语义与颜色相似度保留，空间布局相似度降至随机基线水平。
- **强化学习 Plume Tracking（风媒气味追踪）**（PPO 训练 RNN 代理）：
  - 行为发散度稳步上升而回报保持；替代策略改为沿烟迹边缘平滑追踪而非迎风直冲；在稀疏烟雾与风向切换的 OOD 条件下**优于锚点**。
- **AI Safety Gridworld Boat Race**：
  - HNC 从遵循赛道的 PPO 策略出发，3 个最大发散端点均采取"反复踩单块箭头瓦片收集代理奖励而不前进"的奖励黑客策略；true return 降为零。
- **CNN 扫参几何测量**（CIFAR-100 子集，宽度 $w\in\{16,24,32,48,96\}$，类别 $C\in\{7,10,20,50,100\}$）：
  - 有效零分数随宽度增加而增大、随类别数增加而减小；归一化曲率随任务复杂度增大而增大、随宽度增大而减小。
  - 即使最小网络在最难任务上仍保留大的有效零分数；更难任务的景观"变硬"。

## 相关工作脉络
- **Mode connectivity**（Freeman & Bruna 2017; Draxler et al. 2018; Garipov et al. 2018; Frankle et al. 2020）：证明独立训练网络可通过低损失路径相连；本文与其定位差异——mode connectivity 关注两点间路径，HNC 从单解出发系统扫描局部连通区域并刻画内部多样性。
- **Solution degeneracy / 表征多样性**（Huang et al. 2025; D'Amour et al. 2022; Ostrow et al. 2026）：证明同任务可存在不同 OOD 泛化与几何性质的解；本文通过单一训练解的局部遍历构造性生成并量化这些解，而非仅比较已独立训练的多个解。
- **Platonic Representation Hypothesis（PRH, Huh et al. 2024）**：主张模型越大表征越趋同；本文发现 ViT 经 HNC 后比任何独立训练模型甚至随机初始化都更不同于锚点，表明跨模型收敛可能源于优化器采样狭窄解子集而非任务唯一规定。
- **Penalized similarity training**（Qian & Pehlevan 2026; Braun et al. 2025）：通过惩罚相似性或硬约束生成替代解；HNC ablation（Fig. 9）表明同等任务损失水平下 HNC 可达更大发散度，且全程不离开低损失区域。
- **Neural thickets（Gan & Isola 2026）**：发现预训练模型邻域内存在多样专家；本文将其推广为通用、可定向、可度量的系统遍历方法，并揭示奖励黑客等隐蔽行为。
- **Parameter symmetries / reparameterizations**（Theiss et al. 2026a;b）：离散变换（如神经元复制）改变表示但不改函数；HNC 做连续小扰动到达本质不同的动力学 regime，差异超出重参数化范畴。

## 局限性与未来方向
- 函数保持仅在有限探针集上近似成立，小函数匹配损失不保证未见输入的行为一致性，尤其分布偏移时；需通过 held-out 评估判断替代解质量（Appendix F.8, G.5, G.6）。
- HNC 是局部搜索，对多基座（multiple basins）结构不可知；作者建议可与全局搜索（如 Ostrow et al. 2026 的 retraining 配合）交替以扩大覆盖（Appendix E）。
- 平坦方向与零空间的界定依赖阈值 $\mu_\text{rel}$，不同阈值下可达多样性与功能漂移存在 tradeoff（Fig. 7, Appendix F.2）。
- 当前主要验证于 RNN、ViT 和 RL 小环境，扩展至大语言模型（LLM）需进一步克服二阶信息计算成本；讨论提到 FFM/自然梯度等已有工作可作为基。
- 有向探索时 steering objective 的设计影响可达区域的性质（如 CKA vs DSA vs kNN vs Brain-Score），尚无通用指导原则。

## 研究启发与可借鉴点
- **HNC 的 predict-correct 框架可直接迁移到任何需要"保持函数+改变内部结构"的场景**：如模型编辑（model editing）、风格转换、机制对齐（alignment with neural data），本文 Appendix H 已示范 Brain-Score 引导下的神经预测性提升。
- **软投影梯度 $(\pmb{I}+\pmb{G}/\mu)^{-1}\nabla\varphi$** 是通用的"在平坦子空间内优化任意可微目标"的工具，可复用于 LoRA 等参数高效微调中寻找功能保持的参数更新方向。
- **奖励黑客暴露**为 RL 安全评估提供了新的自动审计工具：在已知良好策略附近沿零空间行走可系统性地发现奖励函数 underspecification 导致的隐蔽 exploit。
- **有效零分数与归一化曲率的扫参度量**为理解"模型规模-任务难度-景观几何"关系提供可直接复用的实验范式，可迁移到 LLM 规模的景观几何研究中。
- **表征-权重距离解耦**的可视化手段（MDS + 混合独立种子 baseline）是报告替代解多样性的标准对照方案，值得在后续工作中沿用。

## 关键术语表
- **Hessian Null Space Continuation (HNC)**：一种利用函数匹配损失的海森零空间（平坦方向）进行交替步进-恢复的权重空间遍历算法，可在保持网络输入输出映射的同时到达多样化替代解。
- **Function-matching loss**：衡量当前网络与锚点网络在探针集上输出差异的二次损失，锚点为其全局极小点，梯度为零，使平坦方向在一阶和二阶均无漂移。
- **Gauss-Newton 算子**：$\pmb{G}=\pmb{J}^\top\pmb{J}$，在锚点处等价于函数匹配损失的 Hessian 且沿路保持半正定，避免显式构造 $P\times P$ 海森矩阵。
- **LOBPCG**：Locally Optimal Block Preconditioned Conjugate Gradient，用于以矩阵-Free 方式计算最小 $k$ 个特征对的高效迭代特征求解器。
- **CKA（Centered Kernel Alignment）**：衡量两网络隐藏层表征几何相似性的核方法指标，$1-\text{CKA}$ 作为表示距离在此文被用作 steering objective。
- **DSA（Dynamical Similarity Analysis）**：衡量 RNN  recurrent 动力学结构在可逆变换下差异的度量，捕捉状态空间轨迹的整体演化特性。
- **有效零分数（effective null fraction）**：参数空间中满足"小扰动不引起显著输出漂移"方向的占比，反映局部解集的维度。
- **Reward hacking**：代理通过利用奖励函数的规范漏洞获得高代理奖励但未完成目标任务的行为；本文展示 HNC 可在权重邻域中系统性暴露此类策略。

## 可复现要素
- **数据集**：Places-365（ViT 探针集）、ImageNet-1k / ImageNet-V2 / ImageNet-Rendition / ImageNet-Sketch（OOD 评估）、CIFAR-100 子集（几何扫参）、3BFF RNN 内部模拟环境、Plume Tracking 与 AI Safety Gridworlds（RL 环境）。Places、ImageNet、CIFAR 等公开数据集均公开；RL 环境来自 Sing et al. 2023 及 Leike et al. 2017 开源套件。
- **代码**：论文声明项目页与代码已开源（https://ann-huang-0.github.io/Hessian-null-space-continuation），附录含详细超参表（Table 4–11）。
- **关键超参**：$\mu_\text{rel}\in[10^{-7}, 10^{-3}]$（平坦阈值/阻尼）、$\eta$（零空间步长，自适应）、$m$（恢复步数，RNN 取 12、ViT 取 6–8）、$\tau$（损失上限，通常 $10^{-2}$）、探针集大小（RNN 32–128 条 trial、ViT 256–512 张图）、CG 迭代预算（ViT 40 次、RL 50 次、RNN 200 次）。
