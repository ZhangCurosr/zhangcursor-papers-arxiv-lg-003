---
title: "Your-LR-Schedule-May-Not-Be-Principled-Effective-Learning-Ra"
source: https://arxiv.org/pdf/2610.40362v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:07:27"
field: "大语言模型预训练优化"
keywords: ["learning rate scheduling", "effective learning rate", "weight normalization", "directional alignment", "MoE pretraining", "1/sqrt(t) decay"]
innovations: ["识别 ELR 轨迹为训练动态关键驱动因素，匹配 ELR 可对齐 WN/WD 损失动态", "提出 AIR：基于 1/sqrt(t) 理论最优的仿射有限窗口调度方法", "揭示晚期 LR 调度对终端损失具有主导影响，共享晚期可关闭 42%-94% 早期差异"]
benchmarks: ["MMLU", "MMLU-Pro", "AGIEval", "WinoGrande", "BBH", "DROP", "GSM8K", "MATH"]
---

# 论文速读：Your-LR-Schedule-May-Not-Be-Principled-Effective-Learning-Rate-Scheduling-in-Weight-Normalized-Training

## 一句话总结
本文识别出 Weight-Normalized (WN) 训练相比传统 Weight Decay (WD) 的优势主要由其诱导的有效学习率 (ELR) 轨迹驱动，而非归一化本身；提出基于理论最优 1/√t 衰减构造的 AIR 调度方法，在 128B-token MoE 预训练中使终端训练损失较 cosine 降低约 0.01，并揭示晚期 LR 调度是 gains 的主要来源。

## 研究问题与动机
- WN 训练（固定参数块范数）近期被报道能带来更低预训练损失与更好超参可迁移性，但在相同名义 LR 调度下，WN 与 WD 的训练动态显著分化（例：cosine 下 WN 最终优于 WD，WSD 下 WN 全程落后）。
- 现有 LR 调度设计仅针对名义 LR 调度，未考虑参数块范数 r_k 演化对 ELR η^eff = η/r 的耦合影响，导致调度“不原则”。
- 缺乏从优化目标出发推导 ELR 衰减律的工作；并发研究虽观察到匹配 ELR 可对齐 WN/WD 损失动态，但未给出可复用的调度构造。

## 核心贡献（创新点）
1. **将 ELR 轨迹确立为训练动态的首要解释量**：通过 norm/LR 两种匹配方式证明，对齐 ELR 后 WN 与 WD 的损失动态基本坍缩到同一条曲线（误差 <2×10^-4），明确 WN 优势来源于更好的 ELR 调度而非归一化。
2. **从方向对齐最大化推导 1/√t 最优律并提出 AIR**：在 WN 的常数系数对齐动力学下完成平方，得到最大化终端对齐的条件为 η^*(t) ∝ 1/√(1+ct)，再通过仿射变换适配有限训练窗口与指定峰值/终端 LR，构造出 AIR。
3. **揭示晚期调度的主导作用并给出理论界**：阶段交换实验显示共享晚期 AIR 可关闭其他调度与 AIR 差距的 42%–94%；从对齐积分解导出保留因子 H(K,K_s)=exp[-b∫η^2] 解释该现象。

## 方法详解
- **有效学习率定义**：η_k^eff := η_k / r_k，r_k = ||θ_k||，刻画单位参数块上 optimizer 更新对方向的相对影响。
- **对齐动力学（连续近似）**：固定目标单位方向 v，设 m(t)=E[v^Tθ_t/R|F_0]，在切向更新假设下
  - ṁ(t) = aη(t) − bη(t)^2 m(t)，a>0、b=U^2/(2R^2)>0 视为常数。
- **无约束最优解**：由 d(m^2)/dt = a^2/(2b) − 2b(mη − a/(2b))^2 得上限 m(h)^2 ≤ m_0^2 + a^2 h/(2b)，等号成立当且仅当 m(t)η(t)=a/(2b) a.e.，由此导出
  - η^*(t) = η_0 / sqrt(1 + 2bη_0^2 t)，即 1/√t 型 ELR 衰减。
- **AIR 实用化构造**（Eqs. 12–13）：
  - q(k) = (1 + k/α)^{-1/2}，f(k) = (q(k)−q(K))/(1−q(K))。
  - η_AIR(k) = η_min + (p−η_min)f(k)，k=0…K；α=K/8 为主实验选择。
  - 峰 p 可通过离散 LR 总和（含 warmup）唯一标定（Appendix B.6）。
- **阶段交换理论解释**（Eqs. 15–17）：共享晚期时 Δm(K)=H(K,K_s)Δm(K_s)，保留因子 H≤1 衰减早期差异且不反转符号；晚期差异对末态影响更大且权重更高。

## 实验与结果
- **设置**：两档 MoE 预训练：① 110M/4.7B，128B token（16 GPU，48h）；② 790M/35.7B，500B token（32 GPU，168h）。优化器 Muon，WN Frobenius 范数固定。所有对比调度共享 warmup 长度、终端 LR、累计 LR（含 warmup）。
- **主结果（Table 1）**：AIR 在所有对比衰减族中获得最低训练/评估损失与最高下游均值；终端训练损失较 cosine 降低约 0.01（110M：1.730 vs 1.740；790M：1.421 vs 1.431）。
- **下游表现**：790M 下 MMLU 72.1 vs 70.7、MMLU-Pro 42.4 vs 40.6、AGIEval 55.7 vs 54.2。
- **超参鲁棒（Table 5/Figure 4）**：AIR 的 LR 灵敏度最小（train S=0.0018），低损失区间最宽。
- **晚期效应（Figure 5/Table 2）**：与 Cycloid 共享晚期可关闭约 90% 损失差距；将 AIR 晚期接到其他 6 种调度上可关闭 42%–83% 训练差距与 45%–94% 评估差距。
- **ELR 迁移**：将 AIR 的 ELR 轨迹应用到 WD 可复现对应训练动态，表明结论超越固定范数训练。

## 相关工作脉络
- **重量归一化与约束优化**：WN [23,33]、流形约束与重traction 类 [1,26,27]、矩阵范数控制 [35,36]——本文聚焦这些方法背后被忽略的 ELR 视角。
- **LR 调度理论**：River-valley 解释 WSD [32]、凸界/functional scaling 推导 cooldown/power decay [14,24]——这些理论只针对名义 LR，未考虑参数范数演化。
- **并发工作**：[17,34,38] 报道匹配 ELR 可对齐 WN/WD 损失，但未从优化目标推导衰减律；本文首次给出理论最优并构造实用 AIR。
- **内在学习率**：[15] 提出 intrinsic LR；本文显式定义 η^eff=η/r 并将其作为调度设计对象。

## 局限性与未来方向
- 对齐理论假设常数漂移 a、b 与常数更新幅度 U，忽略真实训练中更新统计的演化。
- 未定量建立方向对齐 m 与训练损失 L 的关系。
- 仅在 MoE 架构 + Muon 优化器 + 两档规模上验证，未覆盖 Transformer decoder-only 主力配置与其他优化器。
- 未探索在继续训练 (CT)、监督微调 (SFT)、强化学习 (RL) 中的 ELR 调度。

## 研究启发与可借鉴点
1. **调度对象迁移**：将名义 LR 替换为 ELR 进行设计，可直接移植到 WD/解耦正则等范数会演化的场景。
2. **晚期优先**：在有限调参预算下，最后 ~20% 的 LR 形状对终端损失影响最大，应作为第一优先级优化。
3. **理论→实践桥梁**：以"连续近似→完成平方→提取最优条件→仿射端点构造"的流程，将理论最优曲线快速落地为可超参调度的实用表单，值得复用。
4. **1/√t 律普适性**：该形状在自监督表征学习中也有涌现，可进一步检验其在图像/多模态预训练中的迁移效果。

## 关键术语表
- **有效学习率 (ELR)**：η^eff = η/r，把参数块范数演化纳入后度量单次更新对方向的相对步长。
- **方向对齐 (directional alignment)**：m(t)=E[v^Tθ_t/R]，参数与理想目标方向的余弦相似度期望。
- **AIR**：Affine Inverse-square-Root 衰减，把 1/√t 律经端点仿射变换适配有限训练窗口的实用调度。
- **保留因子 H**：H(h,t)=exp[−b∫_t^h η(s)^2 ds]，刻画 t 时刻的对齐贡献在 h 时刻的残留比例。
- **LR 灵敏度 S**：每条调度族在每个 e-fold 峰 LR 区间内平均损失相对最优值的升高，衡量超参鲁棒性。
- **累积 LR**：整段训练（含 warmup）所有步 LR 之和，用于公平比较不同调度族的“能量”。

## 可复现要素
- 数据集：内部预训练语料（论文未公开具体名称与比例）。
- 代码/权重：论文未提及开源。
- 关键超参：AIR 时间尺度 α=K/8；warmup 步数 3,040（占比 ~9.5%）；终端 LR=10^{-5}；峰值 LR 按累计 LR 匹配标定。
- 实验平台：16/32 GPU；Muon 优化器；Frobenius 范数固定；EMA-0.99 作为训练损失汇总。
