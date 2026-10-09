---
title: "Training-with-Missed-Targets-in-Generative-Recommendation-Se"
source: https://arxiv.org/pdf/2610.10124v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:30:55"
field: "生成式推荐系统"
keywords: ["generative recommendation", "candidate completion", "listwise learning", "reranking", "probability competition"]
innovations: ["提出WN/Cond/Full三损失分离追加监督与组间竞争效应", "推导Conditional Training消除组间概率竞争", "给出基于调整后置信下界的追加训练决策规则"]
benchmarks: ["RecIF-Bench Ads", "Amazon Video Games", "Amazon Cell Phones", "Amazon Health and Personal Care"]
---

# 论文速读：Training-with-Missed-Targets-in-Generative-Recommendation-Separating-Supervision-from-Probability-Competition

## 一句话总结
本文研究了生成式推荐中"候选补全"（candidate completion）训练策略的实际影响，通过将未命中目标追加到reranker训练列表会同时引入三项变化：降低已检索目标权重、增加追加目标监督、产生组间概率竞争。作者设计了三种对比损失分离这些效应，证明组间竞争会损害推理时的返回项排序质量，并给出了一套基于开发集的保守决策规则。

## 研究问题与动机
1. **工程实践与理论分析的缺口**：工业界常用"候选补全"策略——将生成器遗漏的观测目标追加到reranker训练列表，但推理时仍仅对原始候选排序。这种操作同时改变多项训练信号，现有对比实验（append vs no-append）无法归因具体机制。
2. **概率竞争隐患**：图1示例显示，追加目标初始仅获0.011% softmax概率，但均匀监督要求其获得66.7%，迫使两组竞争概率份额，而该竞争在推理时不存在。
3. **跨领域发现的不可直接移植**：机器翻译领域已有"追加黄金参考 harming reranker"的结论，推荐系统场景下是否有相同机制尚不明确。
4. **实用决策需求**：不同generator/reranker组合对追加训练的反应可能相反，需开发可泛化的选择规则。

## 核心贡献（创新点）
1. **提出三损失对照框架**：构造WN、Cond、Full三种损失，每次仅引入一个训练要求变化，使"追加目标监督"与"组间竞争"的影响可分离测量；本质区别在于控制了已检索目标权重保持一致，而对比实验通常混杂多变量。
2. **揭示组间竞争的危害机制**：通过理论推导（Proposition 1）将completed-list损失分解为三项，证明Full损失的第三项$\mathcal{T}_{\text{mass}}$使两组竞争概率份额，且该梯度会通过共享scorer参数影响推理排序。
3. **设计Conditional Training损失**：通过对追加组统一加偏移量$\delta$并最小化，推导出Cond损失形式；其与Full的差异仅在于去掉组间竞争项，保留组内排序学习。
4. **提出开发集驱动的实用决策规则**：基于调整后置信下界（adjusted lower bound）的正负性决定是否为每个generator采用追加训练，在A-Health上避免1.7%性能损失。

## 方法详解
### 符号定义
- $Y_x$：用户$x$的观测目标集合
- $N(x)$：生成器返回的候选集
- $A(x) = (Y_x \cap J) \setminus N(x)$：追加的未命中目标集
- $C(x) = N(x) \cup A(x)$：补全训练列表

### 三项分解（Proposition 1）
$$\mathcal{L}_{\text{append}} = \underbrace{\alpha \mathcal{L}_N}_{\mathcal{T}_N} + \underbrace{\gamma \mathcal{L}_A}_{\mathcal{T}_A} + \underbrace{-\alpha \log \frac{Z_N}{Z_N+Z_A} - \gamma \log \frac{Z_A}{Z_N+Z_A}}_{\mathcal{T}_{\text{mass}}}$$

其中$\alpha = r/(r+k)$、$\gamma = k/(r+k)$，$Z_N$、$Z_A$为组内分数归一化分母。

- $\mathcal{T}_N$：已检索目标在返回组内的排序损失（按$\alpha$加权）
- $\mathcal{T}_A$：追加目标组内等分排序损失（按$\gamma$加权）
- $\mathcal{T}_{\text{mass}}$：组间概率竞争项，要求追加组获得$\gamma$份额的目标概率

### 三种损失定义
| 损失 | 公式 | 引入的变化 |
|------|------|-----------|
| WN | $\mathcal{T}_N$ | 基准：仅加权返回组内排序 |
| Cond | $\mathcal{T}_N + \mathcal{T}_A$ | 在WN基础上增加追加目标监督 |
| Full | $\mathcal{T}_N + \mathcal{T}_A + \mathcal{T}_{\text{mass}}$ | 在Cond基础上增加组间竞争 |

### Conditional Training推导
对Full损失中所有追加分数统一加偏移$\delta$，求导得最优偏移：
$$\delta^* = \log(k/r) + \log Z_N - \log Z_A$$

最小化后剩余项为$\mathcal{T}_N + \mathcal{T}_A + H(\alpha,\gamma)$，其中熵项$H$无梯度，故Cond可直接训练。

### 竞争影响的梯度分析
$$\nabla_\phi \mathcal{T}_{\text{mass}} = (\pi_A - \gamma)(\mu_A - \mu_N)$$
当追加组预测概率$\pi_A$偏离目标份额$\gamma$时，竞争项产生梯度更新共享scorer参数。

## 实验与结果
### 数据集与模型
- **RecIF-Bench Ads**：使用Released OneRec-1.7B-Pro，3,189个有效训练组
- **Amazon Video Games (A-Games)**：本地Transformer (seed 42)，914个训练组
- **Amazon Home/Music/Toys**：GRU generator，验证结论泛化性
- **Amazon Cell Phones/Health**：新数据，用于测试决策规则

### 主要结果

**RQ1：组间竞争是否有害？**
- RecIF-Ads（7次独立训练）：Cond比Full平均提升**+12.41 × 10⁻³** FT-NDCG，95%区间[+10.31, +14.51]排除零
- A-Games（预注册，9,121用户）：
  - MLP reranker：Cond-Full **+22.23%**，WN-WN+Mass **+7.83%**
  - Attention reranker：Cond-Full **+12.84%**，WN-WN+Mass **+13.16%**
  - 四个用户区间及四个训练区间中三个排除零

**直接验证概率竞争**：对追加分数统一加偏移后，Cond训练路径与参数完全不变（数值精度内），而Full显著变化，证实Full对概率分配敏感而Cond不敏感。

**RQ2：追加目标监督是否有益？**
- RecIF-Ads epoch 60：Cond-WN **+1.88 × 10⁻³** [0.34, 3.45]，七次训练均同向
- 但开发集选checkpoint后：**-0.89** [-2.21, +1.21]，效果不确定

**RQ3：开发集决策规则有效性？**
- A-Cell：规则为2/3 generator选择Disc，整体变化+0.32 [-0.08, +0.72]
- A-Health：规则为所有3 generator保留返回-only训练
- 若强制使用最佳追加训练：A-Health损失**-1.7%**，规则成功避免

## 相关工作脉络
1. **ListNet [3]**：列表级概率模型基础，本文损失构造的起点
2. **Profile likelihood [14]**：最小化共享偏移量，但未比较这三种reranker损失
3. **DKD [39]**：分离target/non-target知识蒸馏，分组逻辑不同（时间存在性 vs 类别标签）
4. **Cascade optimization [8,18,20]**：协调多阶段排序或校准子列表，本文generator固定不变
5. **LUPI [23,28]**：使用训练时特权特征，本文使用训练时追加比较项而非特征
6. **Candidate-free verifier [38]**：学习token似然无补全列表监督，本文聚焦补全后的竞争效应

## 局限性与未来方向
1. **训练时长依赖性**：RecIF-Ads结果显示epoch 60优势明显，但240 epoch后Cond-Full差距缩小至+0.0012，结论受训练预算影响
2. **目标权重扩展有限**：Appendix A.2虽推导了非负权重情形，但实验仅使用均匀权重
3. **特定场景可能反转**：A-Home中Cond-Full仅+0.13且区间跨越零，说明某些数据分布下竞争影响可能不显著
4. **生成器固定假设**：未研究generator与reranker联合训练情形，实际系统中二者可能协同优化
5. **决策规则保守性**：调整三重比较后的置信下界可能过于保守，错过潜在收益

## 研究启发与可借鉴点
1. **因果分解实验设计**：通过保持其他条件不变、逐步添加训练要求的方式分离混杂效应，适用于其他"训练策略影响"的归因分析
2. **Conditional Training构造技巧**：对一组分数统一加偏移量后最小化的方法，可迁移到其他需要消除组间竞争的场景（如多任务学习、多源数据融合）
3. **保守决策规则的工程价值**：使用调整后置信下界而非均值做切换决策，避免过拟合开发集，适合在线A/B测试前的离线筛选
4. **FT-NDCG指标的严谨使用**：分母包含所有eligible目标（含未命中的），确保度量反映端到端系统真实性能，而非仅candidate pool内表现
5. **生成式推荐与reranking解耦研究范式**：固定generator和inference pool，仅改变reranker训练策略，为模块化解耦分析提供模板

## 关键术语表
**Generative Recommendation**：通过生成语义ID（semantic IDs）而非打分检索候选的推荐范式
**Candidate Completion**：将生成器遗漏的观测目标追加到reranker训练列表的技术
**FT-NDCG@20**：Full-Target Normalized Discounted Cumulative Gain，分母包含用户全部eligible目标
**Group Competition**：训练时追加目标组与返回候选组共同争夺softmax概率份额的现象
**Conditional Training (Cond)**：独立归一化两组分数、消除组间竞争的损失函数
**Semantic ID**：分配给目录项目的短token序列，作为生成式检索的标识符
**Listwise Learning**：基于整个候选列表概率分布的排序学习方法
**OneRec**：开源的生成式推荐模型，提供1.7B参数的checkpoint

## 可复现要素
- **数据集**：Amazon 2014 5-core（公开）、RecIF-Bench（公开）
- **代码**：arXiv ancillary archive包含完整代码、环境配置、split manifests及运行命令
- **模型权重**：OneRec-1.7B-Pro / OneRec-1.7B checkpoint公开可用；本地Transformer/GRU训练脚本开源
- **关键超参**：AdamW（lr=10⁻⁴, weight decay=10⁻⁴, gradient clipping norm=1），batch size=64，MLP hidden=[256,128]，Attention width=128 heads=4，60 epochs primary / 240 epochs robustness
- **随机种子**：Generator seeds 42/91-93/101-105/111-113/121-123，Reranker seeds 3101-3110/42-44/101-105
- **温度参数**：OneRec temperature=1.2，本地Transformer/GRU temperature=1.0
- **Beam width**：64
