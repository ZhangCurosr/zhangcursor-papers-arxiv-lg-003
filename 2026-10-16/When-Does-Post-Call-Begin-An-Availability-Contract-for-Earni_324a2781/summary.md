---
title: "When-Does-Post-Call-Begin-An-Availability-Contract-for-Earni"
source: https://arxiv.org/pdf/2610.21659v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:53:36"
field: "金融自然语言处理与基准评测"
keywords: ["financial NLP", "earnings calls", "benchmark construction", "availability contract", "event window", "timing sensitivity", "lookahead bias", "trustworthy AI"]
innovations: ["提出Availability Contract九字段报告框架，把输入完成证据与价格锚绑定", "构造paired three-clock diagnostic并在固定设计矩阵下给出精确OLS分解恒等式", "在552次call与DEC基准上证明timing choice可显著改变标签与损失而排名不变"]
benchmarks: ["Dense Earnings Call (DEC)", "552-call paired three-clock audit", "24-firm 2018-2024 common sample"]
---

# 论文速读：When-Does-Post-Call-Begin-An-Availability-Contract-for-Earni

## 一句话总结
论文提出"可用性契约"（Availability Contract）框架，要求金融NLP基准明确声明输入边界、输入完成证据、结果起始时钟及对应声明，并通过三时钟配对审计揭示：将收益标签起始点从文档起始时钟后移至音频完成+15分钟 allowance 后，9.4%的earnings call收益符号翻转、28.3%改变五分位数分类；在DEC基准上向前移动一天目标窗口使3天MSE上升18.1–21.4%，但模型点排名不变。

## 研究问题与动机
- **事件内时序错配**：财报call的预测目标（stock return）起始点可能早于模型实际可用的文本/音频输入完成，chronological train–test split无法解决"单个事件内部"的表示边界–标签起始不匹配。
- **基准可复现性隐患**：相同模型、数据、split在不同时间戳→价格映射或输入完成假设下会得到不同标签和损失，而这些假设常埋藏在预处理代码中。
- **已有多数benchmark混淆event-response association与post-input prediction**：如部分工作将complete-call representation与call start之前的return拼接，名义上称为"post-input evaluation"却缺少timing evidence。
- **稳定排名≠鲁棒损失**：公开DEC基准上模型排名不变，但改变目标时钟后MSE显著恶化，说明仅看leaderboard会掩盖timing sensitivity。

## 核心贡献（创新点）
1. **提出Availability Contract作为representation-specific reporting框架**，要求九字段（claim/input boundary/proxy rule/allowance/checkpoint/price anchor/endpoint/sample/inference）显式记录，与已有BenchmarkCards相比把"输入完成证据+价格锚"作为独立维度。
2. **构造paired three-clock diagnostic**（L→S→A），在固定样本与endpoint下分离event-spanning、documentary-start-aligned、completion-aligned三类return，利用OLS线性给出精确系数分解恒等式 $\hat\beta^L-\hat\beta^S=\hat\beta^{LS}$ 等。
3. **实证揭示timing choice对label与loss的实质性影响**：在552次call上S→A使9.4%符号翻转、28.3%固定五分位变动；在DEC上提前一天目标使3天MSE上升18.1–21.4%，同时保持Kendall's $\tau=1$ 排名稳定，说明"排名稳定"不等于"标签/损失稳健"。

## 方法详解
- **Availability Contract九字段**：permitted claim $\mathcal{C}$、input boundary $X$、source-to-completion rule $G$、allowance $\ell$、checkpoint记录 $M$、inference-completion evidence $E$、price-anchor rule $\Lambda$、endpoint $E$、sample rule $S$；label start 为 $\Lambda(G(X)+\ell)$。
- **Audio-duration proxy**：$t_i^{\text{audio}} = t_i^{\text{source start}} + \max(d_i^{\text{MP3}}, e_i^{\text{alignment}})$，取两者最大值确保proxy不早于任一记录。
- **Ready proxy**：$t_i^{\text{ready}} = t_i^{\text{audio}} + \ell$，baseline $\ell=15$ minutes，敏感性网格0–120 min。
- **价格锚规则**：在XNYS 5-min bar日历 $\mathcal{U}_{\text{XNYS}}$ 上取严格大于proxy的最小bar start $u$，该bar close 为锚价格 $P$；$P^S$ 对应source-start，$P^A$ 对应ready proxy。
- **对数收益**：$r_i^c = 100\log(P_i^E/P_i^c)$，相邻区间可加；设计差分解：$\hat\beta^L-\hat\beta^S=\hat\beta^{LS}$ 等，固定设计矩阵下精确成立。
- **四representation**：Lexical tone（Loughran-McDonald子集）、FinBERT、Qwen3-4B answer valence、E5 narrative revision，统一在552样本上标准化。
- **统计推断**：5,000次firm-pairs-cluster × calendar-week-pairs-block双重bootstrap；Holm family-wise、BH within-family；CGM多向聚类方差估计作为asymptotic check。

## 实验与结果
- **样本**：parent panel 2,094次call（30 firm, 2006–2024）→usable source clock 755→audio proxy 669→complete common sample 552次（24 firm, 2018–2024），保留26.4%。
- **三时钟敏感性（Panel A）**：$L\to S$ 符号翻转32.1%、相对移动64.3%、固定移动65.2%；$S\to A$ 在15 min allowance下符号翻转9.4%、固定五分位移动28.3%、重新计算相对分布移动30.1%；equal-anchor子集（241/552）固定bin不变、仅6.6%相对移动。
- **DEC外部验证（Panel B）**：1,710事件×4 frozen prediction streams（TSMixer+3 TMLP），before-market 1,191目标前移一天；3-day $\rho_s=0.645$、relative 58.8%/fixed 43.5%、median rank shift 12.5 pts、$\Delta$MSE 18.1–21.4%；7-day/15-day递减；point ranking $\tau=1$ 不变；best-MSE保留概率issuer 67.1%/95.0%/86.1%，quarter 72.1%/83.3%/80.4%。
- **关联定位（Table 5）**：Qwen valence $\beta_L=+1.122,\beta_S=-0.135,\beta_A=-0.233$，总$L\to A$位移+1.355 pp/SD中$+1.256$来自$L\to S$、$+0.098$来自$S\to A$，说明大部分association shift发生在source-start之前。
- **设计敏感性（Table 4）**：allowance 0–120 min范围sign 6.5–15.0%、relative 23.9–39.9%、fixed 21.0–35.9%；endpoint换390 trading min后$\beta$从-0.233变-0.226；双路CGM依赖估计$\Delta\beta=+1.355[+0.505,+2.204]$。

## 相关工作脉络
- **Event-study经典**（Brown & Warner 1985; MacKinlay 1997; Kolari & Pynnönen 2010）：测量事件反应，但未处理representation completion问题。
- **Financial NLP表示**（FinBERT、VolTAGE、MAEC、FinCall-Surprise）：多用complete-call或event-window返回，未显式声明输入可用性时钟。
- **Lookahead bias检测**（Gao et al. 2025; ExAnte benchmark）：关注模型训练时泄漏，本文关注label start早于input availability的benchmark-schema层问题。
- **Point-in-time checkpoints**（Kelly et al. 2026）：解决模型知识边界，但不解决"标签起始在call完成之前"的评估定义问题。
- **Release-timing研究**（Wu et al. 2025 ICAIF; Berkman & Truong 2009; Oh 2026）：区分announcement-day vs post-open，本文把同等严格性扩展到representation completion。
- **BenchmarkCards**（Sokol et al. 2025）：LLM benchmark文档化标准；本文将其专业化到financial NLP的input boundary+clock evidence+price anchor。

## 局限性与未来方向
- **audio proxy非直接观测**：用$\max(\text{MP3 duration}, \text{alignment end})$，未观测transcript/model-ready/order/execution时钟，部署声明无法在此数据上验证。
- **price data上游vendor未独立认证**：timezone/adjustment/total-return语义依赖单一来源，跨源交叉验证缺失。
- **样本选择性流失**：source-clock join移除1,339次，panel偏向大型/近期firm，mean log mcap移动约1 pooled SD，外推需谨慎。
- **allowance取固定15 min**：虽做0–120网格敏感性，但optimal allowance因task而异，合约要求后续工作自行声明。
- **representation仅四种**：lexical/sentiment/classification/narrative，未覆盖multi-modal融合、graph、diffusion等新架构。
- **未来方向**：扩展至press release、SEC filing、CEO推特等多源disclosure；与point-in-time模型结合做deployability claim；自动化timestamp provenance pipeline。

## 研究启发与可借鉴点
1. **可用性契约模板可直接迁移**：任何"input completion → label start"的financial NLP任务（新闻→收益率、财报→波动率、Q&A→prediction）都可套用九字段Card，降低timing漏报风险。
2. **三时钟配对审计的精确分解恒等式**：在固定设计矩阵下 $\hat\beta$ 差等于区间$\hat\beta$，无需重跑多次回归即可定位association shift发生在哪个时间区间，适合做replication audit tool。
3. **DEC frozen-rescoring设计**：保持预测固定、只移目标，可在不重训模型下隔离target-clock sensitivity vs model adaptation，便于benchmark竞争中的"公平再评分"。
4. **允许allowance grid + concentration check**：0–120 min网格+delete-one-firm/year是report timing sensitivity的标准范式，值得在其它金融NLP benchmark中沿用。
5. **关联定位结论**：多数"pre-call语言→return"关联实际位于source-start之前，提示文献中部分显著结果可能是event-window artifact而非post-input predictability，为本团队筛选稳健特征提供过滤标准。

## 关键术语表
- **Availability Contract**：要求benchmark显式声明输入边界、完成证据、结果时钟与 permitted claim 的报告框架。
- **Three-clock audit**：$L$（前一日收盘）、$S$（文档source-start后首bar close）、$A$（音频完成+allowance后首bar close）三锚点配对收益诊断。
- **Completion proxy**：基于$\max(\text{MP3 duration}, \text{alignment end})$重建的call audio结束时刻代理。
- **Allowance $\ell$**：从completion proxy到label start之间的 stipulated processing time，baseline 15 min。
- **Proxy-aligned association**：宣称completion proxy先于label start时的解释，不是verified post-input prediction。
- **Frozen rescoring**：固定模型预测、仅重定义目标窗口，用于隔离target-clock sensitivity。
- **Event-window association vs post-input attribution**：前者测量披露期间价格变动，后者要求label start在完整输入可用之后。
- **CGM multiway clustering**（Cameron-Gelbach-Miller）：同时聚类firm与week的两向方差估计，用于asymptotic sensitivity check。

## 可复现要素
- **数据集**：SEC/公司官网earnings call source pages + archived MP3 + sentence alignment；XNYS 5-min bar via `exchange_calendars` 4.5.6；DEC基准来自Yu et al. [41]公开。
- **代码/权重**：论文未提供开源仓库；representation使用FinBERT（ProsusAI/finbert）、Qwen3-4B greedy 80 tokens、E5-large-v2；审计脚本由作者维护，未在文中公开URL。
- **关键超参**：allowance $\ell=15$ min（网格0–120）；Qwen truncation 3,072 tokens、max new tokens 80、greedy；bootstrap 5,000 draws；Holm family-wise + BH within-family；CGM df=23。
- **时间点**：Qwen inference于2026年6月运行，checkpoint为2025 vintage。
