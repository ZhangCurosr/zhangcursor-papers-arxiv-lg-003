---
title: "Tune-the-Baseline-First-Most-of-Muon-s-Scaled-Weight-Decay-G"
source: https://arxiv.org/pdf/2611.02529v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:13:02"
field: "大规模语言模型训练优化"
keywords: ["Muon optimizer", "scaled weight decay", "weight decay tuning", "optimization comparison", "scaling ladder", "learning rate scheduling"]
innovations: ["量化 Muon-SW 优势中约 90% 为 tuning gap，残差随规模从 10% 增至 21%", "联合 LR·WD 2D 网格扫描 + 二次拟合协议用于公平优化器比较", "揭示最优 LR·WD 乘积随规模移动速率的方法依赖性（比率从 1.40 增至 1.91）"]
benchmarks: ["FineWeb-Edu", "CodeParrot-clean", "OpenWebMath"]
---

# 论文速读：Tune-the-Baseline-First-Most-of-Muon-s-Scaled-Weight-Decay-Gain

## 一句话总结
论文通过联合调参（LR×LR·WD）在 scaling ladder 上重新评估 Muon-SW（缩放权重衰减）的增益，发现约 90% 的报告优势源于 baseline（常数衰减）未充分调参的"调参差距"，但剩余的微小真实优势随模型规模扩大而从约 10% 提升至 21%。

## 研究问题与动机
- 现有优化器比较通常在单一共享超参设置下对比变体与 baseline，但当变体改变了超参（如权重衰减调度）的作用强度时，共享设置可能对一方法已调优、对另一方法严重欠调优，导致"调参差距"虚增报告收益。
- Muon-SW（Apte, 2026）被报告在语言模型训练中优于常数解耦权重衰减，但此前比较多复用同一 LR/WD 设置，未对 baseline 做同等精细调参。
- 权重衰减最优值随模型/数据规模移动，且与学习率耦合（LR·WD 乘积设定平衡范数与有效旋转），需在不同规模下联合扫描。
- MAI-Base-2 提供了更公平的测试框架：2D 网格联合搜索 (LR, LR·WD) 的二次拟合最优，以及 source-scaled MoE ladder（固定每 active parameter token 数与数据源重复次数），避免小模型偏向欠正则化设置。

## 核心贡献（创新点）
- **首次在对 baseline 进行同等精细联合调参后量化 Muon-SW 的剩余优势**：发现约 90% 报告增益在 L3（1.89M）消失，残差仅 7.4%–10.3%，本质区别在于将"调参差距"从真实优势中剥离。
- **揭示残差优势随规模单调增长**：在 8× active parameters 范围内，survival 从 10.3%（L3）→13.0%（L4）→21.3%（L6），表明 Muon-SW 在大规模 regime 下重要性上升。
- **刻画最优 LR·WD 乘积随规模移动的轨迹及其方法依赖性**：constant decay 的 p* 从 0.19→0.33，Muon-SW 从 0.27→0.63，比率从 1.40→1.91，证明跨规模复用小尺度 tune 的结果会系统性误配。
- **预注册三项关于比率 r=p_SW*/p_const* 的假设并检验**：cosine 下数据与无变化（r=1）和匹配总衰减（r=1/ s̄=1.36）一致、拒绝端点匹配（r=10）；WSD 下 r=0.29 异常，列为开放问题。
- **开源完整可复现 pipeline**：单张 NVIDIA T4 GPU 完成 81 次运行，代码、配置、图表脚本全公开，包含抢占安全恢复机制。

## 方法详解
- **两方法对比设置**：非 embedding 二维参数用 Muon，其余用 AdamW；LR 调度 η_t = η·f(t)，f(t)∈[0.1,1] 为余弦或 WSD（warmup-stable-decay）；常数衰减：θ←θ(1−η_t λ)；Muon-SW：系数乘 s(t)=η_t/η_max，即 θ←θ(1−η_t²λ/η_max)，峰值 LR 时全强度、退火时衰减减弱。
- **联合 LR·WD 扫描**：以 MAI-law 预测值 (η_0, (ηλ)_0) 为中心，扫 LR 乘子 m_η∈{0.5,1,2} 与乘积乘子 p（2 的幂次网格），L3 做 2D 二次拟合，L4/L6 在 m_η=1 处做 1D 拟合；统一提取 1D 切片使三 rung 可比。
- **关键度量**：untuned gap = 常数衰减 BPB − Muon-SW BPB（均在 p=1）；tuned gap = 两方法 tuned BPB 之差；survival = tuned gap / untuned gap；计算效率用 loss-vs-compute 拟合换算为 compute-equivalent speedup。
- **Scaling ladder**：L3/L4/L6 三档，active params 1.89M/3.89M/15.04M，hidden size 256/320/512，MoE 布局随 L 递增；每 active parameter 训练 120 tokens，数据池含 FineWeb-Edu（60%）、CodeParrot-clean（25%）、OpenWebMath（15%），source scaling 使 web 数据每 rung 见 1 次、code/math 见 4 次。
- **评估**：held-out 字节级 BPE（8192 词表）上的 pooled BPB，3,000 文档/源作评测集；所有运行在单 T4 Spot 集群完成，15 分钟 checkpoint + 抢占自动续跑。

## 实验与结果
- **L3（1.89M active）**：untuned gap=0.0350 BPB（对应 1.52× compute saving）；2D tuned gap=0.0026 BPB（95% CI [0.0018, 0.0034]），speedup 降至 1.03×；4/4 paired seeds 均 favor Muon-SW；survival=7.4%（2D）/10.3%（1D slice）。
- **跨 rung 趋势**：untuned gap 从 0.035→0.031→0.024 BPB（随规模缩小），tuned gap 从 0.0036→0.0040→0.0051 BPB（增长），survival 从 10%→13%→21%。
- **最优乘积移动**：p_const* 0.19→0.25→0.33，p_SW* 0.27→0.41→0.63，比率 r 1.40→1.63→1.91。
- **假设检验（L3）**：cosine 下 r=1.41（CI [0.88, 2.25]），与 H_none（r=1）和 H_int（r=1.36）一致，拒绝 H_end（r=10）；WSD 下 r=0.29（CI [0.19, 0.43]）异常，低于所有预测。
- **Law 外推检验**：用 L3+L4 预测 L6 的 p*，constant 预测 0.44 vs 实测 0.33，Muon-SW 预测 0.95 vs 实测 0.63；使用预测值分别损失 0.0014/0.0020 BPB，量级与 tuned gap 相当。
- **最强结果**：Muon-SW 在 tuned 状态下仍 consistently 优于常数衰减，L6 survival=21.3%，且在最小规模上 4/4 seeds 方向一致。

## 相关工作脉络
- **Choi et al. (2019)**：证明优化器排名在各自超参充分调优后会发生改变；本文将其原理应用于单一优化器修改（Muon-SW vs 常数衰减）并显式度量"存活份额"。
- **Dahl et al. (2023) / AlgoPerf**：在 benchmark 中内置调参预算；本文沿用同等原则但对 baseline 做同等精细搜索，直接量化调参差距占比。
- **Loshchilov & Hutter (2019)**：提出 decoupled weight decay；本文指出 LR·WD 乘积耦合平衡权范数与有效旋转，是最优搜索的关键 1D 流形。
- **Wang & Aitchison (2024)**：推导 AdamW 的 WD 随模型/数据规模的变化规律；本文将其思想推广到 Muon + Muon-SW 场景，发现两类方法的最优乘积移动速率不同（r 从 1.4 增至 1.9）。
- **Apte (2026)**：提出 Muon-SW 并报告显著增益；本文证明该增益约 90% 来自 baseline 未调优，剩余真实增益小而随规模增长。
- **Jordan et al. (2025) / MAI-Base-2 (2026)**：Muonn 在 LLM 训练中可扩展；本文借用其联合 LR·WD 扫描协议与 source-scaled ladder 设计，实现干净对比。

## 局限性与未来方向
- **规模上限**：最大 rung 仅 15M active（94M total），远低于生产级模型；计划的 L9（50.65M active）因计算资源未执行，跨 L6 的残差增长趋势缺乏外推支撑。
- **种子数不足**：仅 L3 有 4 个 paired seeds，L4/L6 各仅 1 seed；seed 标准差约 0.0007 BPB，虽远小于 tuned gap，但规模趋势基于三个点。
- **扫描边界**：L4 的 Muon-SW optimum 与 L6 两方法 optimum 均落在采样范围外，当前 tuned BPB 取拟合与观测较好者，更密扫描可能改变 L4/L6 数值。
- **高于 L3 仅 1D 扫描**：L4/L6 只在 law-predicted LR 处扫乘积，L3 1D slice 的 survival（10.3%）略高于 2D 全拟合（7.4%），1D 数字可能轻微高估 survival。
- **WSD 异常未解**：WSD 下 r=0.29 与所有假设不符，可能源于长 constant-LR 阶段与衰减调度的交互，仅限 L3 测试，待后续验证。
- **数据单一**：仅一组数据混合与一个 tokenizer；其他混合或更高数据重复下的结论可能不同。

## 研究启发与可借鉴点
- **优化器比较必须 tune baseline**：任何声称新变体优越的工作，若未在同等精细调参下对比 baseline，其报告的增益可能大部分是 tuning gap；建议团队在新 optimizer 实验中嵌入 2D (LR, LR·WD) 网格扫描与二次拟合流程。
- **Scaling ladder 设计可迁移**：固定 TPP 与 source scaling（保持各源重复次数一致）避免小模型偏向欠正则化；可复用于其他优化器/调度方法的公平横向比较。
- **LR·WD 乘积作为关键 1D 搜索流形**：由于 LR 与 WD 强耦合，沿乘积方向扫描可大幅减少搜索维度；2D 拟合后再提取 1D 切片保证跨规模可比。
- **Law 外推的经济性检验**：用 L3+L4 拟合律预测 L6 的最优乘积，误差导致的 BPB 损失（0.0014–0.0020）与 tuned gap 量级相当——说明跨规模复用 tune 结果代价不可忽略，需在目标规模重新 scan。
- **单 T4 低优先级 Spot 集群的 pre-emption-safe pipeline**：15 分钟 checkpoint + 自动续跑机制可在有限计算下完成大规模 grid sweep；可作为团队小资源复现研究的参考架构。

## 关键术语表
- **Muon-SW**：Muon 优化器的 scaled weight decay 变体，系数按 η_t/η_max 缩放，峰值 LR 时全强度、退火时减弱。
- **Tuning gap**：因 baseline 未充分调参而虚增的方法优势份额；本文证明 Muon-SW 报告增益中约 90% 属此类。
- **Survival**：tuned gap / untuned gap，衡量真实优势占比；L3 约 7%–10%，L6 升至 21%。
- **Source scaling**：保持各数据源在每 rung 的重复次数固定（web 1×、code/math 4×），避免小模型偏向特定正则化强度。
- **LR·WD product（p）**：学习率与权重衰减系数的乘积，设定权重平衡范数与有效旋转，是联合调参的关键 1D 流形。
- **WSD（warmup–stable–decay）**：三阶段 LR 调度（线性 warmup → 恒常 → 衰减至 10%），本文用作 schedule-robustness 检查。
- **Compute-equivalent speedup**：由 loss-vs-compute 拟合将 BPB 差距换算为等效计算加速比；untuned 时 1.52×，tuned 后 1.03×。
- **MAI-Base-2 protocol**：Microsoft AI 提出的联合 LR·WD 2D 网格扫描 + 二次拟合最优协议，本文沿用并扩展至 scaling ladder。

## 可复现要素
- **数据集**：FineWeb-Edu sample-10BT、CodeParrot-clean、OpenWebMath（均公开）；自建 8192-token byte-level BPE tokenizer；论文未提及额外私有数据。
- **代码/权重**：代码、配置、81 次运行摘要、图表重现实脚本均开源：<https://github.com/ethxin0011/weight_decay_ladder>
- **关键超参**：LR 乘子 m_η∈{0.5,1,2}，乘积乘子 p 按 2 的幂次网格；L3 2D 全扫、L4/L6 1D 切片；cosine 为主、WSD 校验；每 active parameter 120 tokens。
- **硬件**：单 NVIDIA T4（16 GB）Spot 集群；15 分钟 checkpoint + 抢占续跑；L6 单 run 约 23 GPU-hours，经历 19 次抢占无丢失。
- **复现性声明**：论文提供 make_figures.py 重生成所有图表与表格，运行结果完全可复现。
