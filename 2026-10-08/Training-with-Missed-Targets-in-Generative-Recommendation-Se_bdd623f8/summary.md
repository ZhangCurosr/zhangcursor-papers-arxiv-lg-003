---
title: "Training-with-Missed-Targets-in-Generative-Recommendation-Se"
source: https://arxiv.org/pdf/2610.10124v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:31:28"
field: "生成式推荐与重排序训练策略"
keywords: ["generative recommendation", "reranking", "candidate completion", "listwise learning", "training-inference mismatch", "probability competition"]
innovations: ["把追加未命中目标的损失拆成组内监督与跨组竞争两项并进行受控对比", "由 Full 推导 Cond 以消除组间 softmax 竞争并保留两组组内排序", "提出基于开发集校正下置信界的生成器粒度切换规则并在保留集验证"]
benchmarks: ["RecIF-Ads", "RecIF-Product", "Amazon Video Games", "Amazon Digital Music", "Amazon Home and Kitchen", "Amazon Cell Phones and Accessories", "Amazon Health and Personal Care"]
---

# 论文速读：Training-with-Missed-Targets-in-Generative-Recommendation-Se

## 一句话总结
论文构造了三种匹配损失来分离“追加未命中目标的监督信号”与“组间概率竞争”两个因素，证明在生成式推荐重排序训练中，将未命中目标追加到训练列表时，两组之间的概率竞争可能损害实际推理时的返回候选排序；并给出生成器粒度的保守开发集选择规则。

## 研究问题与动机
- 生成式推荐在推理时只返回有限候选集合，可能漏掉真实目标；实践中常把漏掉的已观察目标追加到重排序训练列表，但推理仍只对原始候选排序，造成训练/推理候选池不一致。
- 简单“追加/不追加”比较无法拆解三个同时发生的改变：降低已检索目标的权重、增加对追加目标的监督、让两组目标竞争总概率。
- 已有工作（如级联优化、候选池固定下的学习、MT reranker 加金标）未针对推荐中“训练仅用追加目标、推理完全不见它们”的特定不对齐给出可分离的因果解释。
- 需要一种受控对比，判断概率竞争是否有害、追加监督是否有利，并为每个生成器提供是否启用追加训练的实用准则。

## 核心贡献（创新点）
- 构造 WN/Cond/Full 三种其余设置完全一致的损失，分别只引入单变量变化，从而把追加操作的重量/监督/竞争三种效应拆开识别。  
  与已有工作相比，本文不是改进生成器或调整多阶段协同，而是在固定生成器与推理候选池的前提下，仅逐项切换重排序监督形态。
- 从 Full 损失推导得到 Cond 损失：通过对追加组加同一偏移并最小化，证明组间竞争项可被熵常数替代，从而在保留组内排序的同时消除跨组概率竞争。  
  区别于 DKD 按目标/非目标类拆分、或 profile likelihood 在共享偏移上最小化，本文拆分的是“推理可见组”与“仅训练组”的概率归一化。
- 给出可直接落地的生成器粒度决策规则：在开发集上比较最优不追加与最优追加训练，仅当经多重比较校正后的下置信界为正时才切换。  
  与以往“固定用某类完成策略”的做法不同，本文强调不同生成器/类别应分开评估，避免自动泛化带来的隐性损耗。
- 在 RecIF 与 Amazon 多生成器、多轮次设置上进行因果拆解与保留集评估，证明竞争去除在多数设定中稳定有益，而追加监督的收益对训练长度与检点选择更敏感。  
  区别于只报告端到端优劣的工作，本文提供可复现的消融因果链与开发/保留双阶段验证。

## 方法详解
- **候选集合定义**：令 $N(x)$ 为生成器返回的候选集，$Y_x$ 为已观察目标集；未命中且可表示的目标构成追加集合 $A(x)=(Y_x\cap J)\setminus N(x)$，训练完整列表为 $C(x)=N(x)\cup A(x)$，推理仍只用 $N(x)$。
- **打分与损失基础**：重排序打分 $s_\phi(i|x)=\ell(i|x)+h_\phi(x,i)$，其中 $\ell$ 为固定生成器的对数似然。对候选集 $C$ 与目标子集使用均匀多目标 listwise 交叉熵：
  $\mathcal{L}(s,C,Y)=-\frac{1}{|Y\cap C|}\sum_{i\in Y\cap C}\log\frac{e^{s_i}}{\sum_{j\in C}e^{s_j}}$。
- **三元分解**：令 $r=|N\cap Y|$、$k=|A|$，$\alpha=r/(r+k)$、$\gamma=k/(r+k)$，定义组内归一化 $Z_N,Z_A$ 与组内概率 $p_N,p_A$，则完整追加损失可写成：
  $\mathcal{L}_{\text{append}}=\alpha\mathcal{L}_N+\gamma\mathcal{L}_A-\alpha\log\frac{Z_N}{Z_N+Z_A}-\gamma\log\frac{Z_A}{Z_N+Z_A}$。
  三项分别对应：返回组内排序、追加组内排序、跨组概率竞争。
- **三种训练损失**：
  - $\mathcal{L}_{\text{WN}}=\alpha\mathcal{L}_N$：仅在返回组内做加权排序，不引入追加监督与竞争。
  - $\mathcal{L}_{\text{Cond}}=\alpha\mathcal{L}_N+\gamma\mathcal{L}_A$：同时保留两组组内排序，但两组概率独立归一化，消除跨组竞争梯度。
  - $\mathcal{L}_{\text{Full}}=\alpha\mathcal{L}_N+\gamma\mathcal{L}_A+\mathcal{T}_{\text{mass}}$：在统一 softmax 上训练，恢复跨组竞争。
- **Cond 的推导要点**：对追加分数整体加偏移 $\delta$ 不会改变组内相对排序；对 Full 关于 $\delta$ 求导可得唯一极小点，剩余项为仅依赖计数的熵常数，因此可直接训练 $\mathcal{T}_N+\mathcal{T}_A$，无需在实现中显式计算偏移。
- **作用到推理的机制**：竞争项梯度形如 $(\pi_A-\gamma)(\mu_A-\mu_N)$，会通过共享参数 $h_\phi$ 改变返回候选的相对打分，即便推理时根本不存在追加项目。
- **纳入训练的范围**：仅对 $r>0$ 且 $k>0$ 的训练列表拟合三种损失；$k=1$ 时 $\mathcal{L}_A$ 为零，Cond 退化为 WN。评估仍包含所有用户，包括 $r=0$ 者。

## 实验与结果
- **数据集与生成器**：RecIF-Ads / OneRec-1.7B-Pro、RecIF-Product / OneRec-1.7B；本地生成器包括 Transformer(seed 42) 与多种 GRU 种子；Amazon 类别含 Video Games、Digital Music、Home and Kitchen、Cell Phones、Health、Toys and Games。
- **评估与基线**：主指标 FT-NDCG@20，分母含完整目标集；基线除 WN/Cond/Full 外，还包含判别式追加损失 Disc 与返回组内成对基线 NP，并在部分实验加入初始化时赋予正确追加组概率的 Full 变体。
- **竞争去除的因果证据**：RecIF-Ads 在 60 轮固定预算下，Cond 比 Full 平均提升 FT-NDCG 约 +7.42×10⁻³[4.98,9.74]×10⁻³；A-Games 四项预设比较中，移除竞争带来 7.83%-22.23% 的相对提升，用户区间与训练运行区间多数不含零。对追加组整体平移分数的实验显示 Cond 不变而 Full 变化，直接印证竞争项是差异来源。
- **追加监督的净收益**：Cond 相比 WN 在 RecIF-Ads 60 轮上显著为正，但经开发集选检点后收益不再稳定，说明监督信号本身有利但更易受训练调度影响。
- **开发集选择规则**：在 A-Home/RecIF-Product 上标定规则后，在保留集 A-Cell/A-Health 验证：规则在 A-Health 上避免了约 1.7% 的 FT-NDCG 损失，而在 A-Cell 上允许部分生成器启用追加训练，整体收益区间包含零但仍呈正向趋势。
- **关键数值要点**：A-Games 上 Cond-Full 在 MLP 与 Attention 上的增益分别在约 3.2-9.2×10⁻³ 量级；初始化给足追加组概率可使 Full 提升约 +9.23×10⁻³，进一步指向竞争项的主导作用。

## 相关工作脉络
- **ListNet/列表学习**：本文借用其列表概率建模，但不同于仅改进排序目标的设计，本文把列表损失拆成组内与跨组两部分并分别控制。
- **Cascade/多阶段协同优化**：如 RankFlow 等工作协调多阶段分布；本文固定生成器与推理候选池，仅改变重排序监督形态，定位在“下游微调因果拆解”而非“阶段联合训练”。
- **Decoupled Knowledge Distillation**：DKD 按目标/非目标类解耦；本文按“推理可见 vs 仅训练可见”解耦，动机与分组维度不同。
- **LUPI/特权信息**：LUPI 利用训练期额外特征；本文利用的是训练期额外比较项，并通过归一化隔离而非特征增强。
- **MT reranker 加金标 harms**：相关发现提示加入参考材料可能造成不利对比；本文将其迁移到推荐中的训练/推理候选不对齐，并用分解方法给出可直接检验的三项因果链。
- **Fixed-pool reranking 与校正方法**：采样 softmax、校准重排等方法关注偏差或校准；本文关注的是“同一固定池上更换监督项后的因果效应”，并不改变采样或校准策略。

## 局限性与未来方向
- 当前分解与实验以均匀目标权重为主，非均匀权重与软标签虽在附录有推广，但实践中的目标重要性设置仍待更系统研究。
- 竞争效应随训练轮数与检点选择而变化，早期训练窗口更稳定，长训练/多检点场景下的泛化结论需进一步验证。
- 规则基于开发集下置信界并做多重比较校正，阈值与校正强度会改变切换决策，阈值的业务可解释性与统一标准仍有讨论空间。
- 未探索更多损失形态（如不同跨组惩罚、动态组权）与更复杂的检索-重排联合训练路径，仅说明本工作保持在固定生成器框架内的因果识别。
- 未命中目标来源的异质性（不同检索路、不同评分阈值）对结果有影响，针对优质/高置信目标子集的专门分析尚不充分。

## 研究启发与可借鉴点
- **固定池上的因果拆解范式**可直接迁移到其他训练/推理不一致场景（如检索增强、外部知识库、离线答案池注入），用“同一候选+不同归一化”分离监督增益与干扰项。
- **组间独立归一化**的思路可用于多任务/多源候选训练，避免某类训练专用样本通过共享 softmax 压低目标业务的真实排序信号。
- **开发集保守切换规则**（校正后下置信界为正才启用）对工程部署很有参考价值，尤其适用于多生成器、多品类并存的平台。
- **对追加组施加全局偏移的拉普拉斯型分析**可作为通用诊断工具：若某策略对偏移敏感而对照不敏感，通常意味着存在跨组竞争或归一化不当。
- **预留零初始化输出层与固定生成器**的设置有助于建立可复现的对比基线，值得在后续生成式推荐评测中作为标准控制组。

## 关键术语表
- **Generative recommendation**：通过生成物品语义 ID 完成检索与排序的推荐范式。
- **Semantic ID**：分配给目录物品的短 token 序列，用于生成式检索与打分。
- **Candidate completion**：把生成器未返回的已观察目标追加到重排序训练列表的做法。
- **Full append training**：将所有候选与追加目标放在同一 softmax 归一化下训练的完整追加策略。
- **Conditional training**：对返回组与追加组分别归一化并相加的损失，消除跨组概率竞争。
- **Weighted-native training**：仅在返回组内进行加权排序训练，不引入追加监督。
- **Group competition term**：跨组共享 softmax 引起的概率竞争项，会把追加目标的监督压力转移到返回候选的排序上。
- **FT-NDCG@20**：以用户完整目标集为分母的 NDCG@20，能反映未命中目标对整体排序的影响。

## 可复现要素
- **数据集**：RecIF-Bench（Ads/Product）与 Amazon 多品类公开数据；OneRec 开源 checkpoint；代码与 split manifest 随 arXiv ancillary archive 提供。
- **代码/权重**：代码与环境已开源；使用 released OneRec-1.7B-Pro/OneRec-1.7B 权重；本地生成器为 Transformer 与 GRU 实现，种子与配置在附录给出。
- **关键超参**：AdamW，weight decay 1e-4，梯度裁剪 norm 1，batch size 64；重排序学习率常见 1e-4/3e-4，主实验 60 轮；长期实验至 240 轮并多折选检点；生成器用 beam=64、temperature 依数据集设定。
- **评估细节**：FT-NDCG@20 为主指标，分母含全部目标；用户区间与训练运行区间采用重抽样；开发集选模后在保留集评估，并做多重比较校正。
