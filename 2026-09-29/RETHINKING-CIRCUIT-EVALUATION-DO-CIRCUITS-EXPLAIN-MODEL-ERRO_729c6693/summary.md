---
title: "RETHINKING-CIRCUIT-EVALUATION-DO-CIRCUITS-EXPLAIN-MODEL-ERRO"
source: https://arxiv.org/pdf/2609.35686v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:55:38"
field: "mechanistic interpretability"
keywords: ["mechanistic interpretability", "circuit evaluation", "faithfulness", "error reproduction", "ablation", "IOI"]
innovations: ["提出A_ok/A_err分层评估框架揭示电路对模型错误的系统性盲区", "IOI案例显示添加63个head-position节点可将错误重现从14.2%提升至75.1%", "系统对比六种电路方法跨三个基准的错误重现能力"]
benchmarks: ["IOI", "DOCSTRING", "MIB"]
---

# 论文速读：RETHINKING-CIRCUIT-EVALUATION-DO-CIRCUITS-EXPLAIN-MODEL-ERRO

## 一句话总结
本文系统检验了现有电路解释方法对模型错误的解释能力，发现多数手动和自动提取的电路虽能高度复现模型正确行为（A_ok > 90%），但对模型特定错误答案的重现率极低（A_err 仅 11.4%–41.7%），揭示了当前电路评估在"忠实度"指标下隐藏的对错误行为的系统性盲区。

## 研究问题与动机
- **现有评估指标的不足**：当前电路评估依赖归一化 logit 差（F）或输出分布 KL 散度等聚合指标，这些指标关注的是整体行为匹配，无法保证电路对模型每个具体输入（包括错误输入）都做出相同预测。
- **解释错误行为的重要性**：机制可解释性（MI）的目标是理解模型内部计算的完整机制，而模型错误往往蕴含独特的计算逻辑；若电路仅在正确样本上复现行为，却无法解释错误来源，则其对"模型为何犯错"的诊断价值受限。
- **错误重现作为必要测试**：作者主张电路应被要求在模型错误样本上复现完全相同的错误答案（exact error reproduction），这构成了对电路解释力的独立且必要的检验维度。
- **发现系统性差距**：跨多种模型、任务和电路提取方法的实验表明，高 A_ok 与低 A_err 之间的巨大鸿沟（Δ 常超过 50pp）是普遍现象，而非特定设置的偶然结果。

## 核心贡献（创新点）
- **提出 A_ok/A_err 分层评估框架**：将电路评估按模型预测结果分为"正确"和"错误"两个层（stratum），分别衡量电路复现模型正确选择与错误选择的能力，与仅报告聚合忠实度的工作形成对比。
- **揭示电路评估的系统性缺陷**：通过 IOI、DOCSTRING 和 MIB 六大模型-任务组合的广泛实验，证明包括 ACDC、EAP-IG、Edge Pruning 及最优消融在内的主流方法均存在严重的错误重现缺口。
- **IOI 案例分析展示可恢复性**：通过贪婪搜索扩展 GPT-2 Small 的 IOI 参考电路，仅添加 63 个 head-position 节点即可将错误重现率从 14.2% 提升至 75.1%，且正确 agreement 仅下降 0.41pp，证明遗漏的计算确实包含错误产生的机制。
- **建议评估规范**：提出建立 stratum、报告 A_ok/A_err/Δ 及绘制分尺寸曲线等推荐做法，为社区提供可操作的评估标准。

## 方法详解
- **分层评估定义**：
  - 正确 agreement：A_ok = E[a(x) | x ∉ E(M)]，衡量电路在模型答对的样本上与模型预测一致的期望
  - 错误 agreement：A_err = E[a(x) | x ∈ E(M)]，衡量电路在模型答错的样本上复现完全相同错误答案的期望
  - 差距 Δ = A_ok - A_err，正值表示电路系统性偏向复现正确行为而忽略错误行为
- **评估设置**：
  - 对比 mean ablation、resample ablation（含 ABC counterfactual）和 optimal ablation（OA，学习替换常数最小化 KL 散度）三种策略
  - 评估对象包括手动电路（Wang et al. 2023）、ACDC、EAP-IG-KL、Edge Pruning 等
- **IOI 案例研究设计**：
  - 目标函数 J(C) = A_err(C) - α[ΔA_ok]+ - λ_P·P_added，在惩罚正确 agreement 下降和增加组件数量的同时最大化错误重现
  - 搜索空间：参考电路 C_0（26 heads）外扩展 25 个位置规则 × 144 heads = 3,574 候选节点
  - 采用梯度筛选 + 精确二元干预评估的贪婪搜索策略
- **控制实验**：
  - 随机扩展对照：结构匹配的随机扩展仅达到 17.33%–41.33% 错误 agreement
  - 标量偏差对照：拟合 subject-logit 偏移至匹配正确 agreement 时仅达 20.44% 错误 agreement
  - Margin matching：按 logit margin 配对正确/错误样本后，部分差距缩小但正差距仍存

## 实验与结果
- **IOI（GPT-2 Small）**：
  - Manual reference（mean ablation）：A_ok = 99.5%，A_err = 15.2%，Δ = 84.3pp
  - EAP-IG-KL（mean ablation）：A_ok = 99.1%，A_err = 41.7%，Δ = 57.4pp
  - Edge Pruning（mean ablation）：A_ok = 98.3%，A_err = 11.4%，Δ = 86.9pp
  - 最优错误重现：EAP-IG-KL（resample）A_err = 51.4%，但仍落后 A_ok（97.8%）46.4pp
- **DOCSTRING（4-layer attention-only transformer）**：
  - Manual reference（mean）：A_ok = 77.1%，A_err = 35.8%，Δ = 41.3pp
  - EAP-IG-KL（mean）：A_ok = 65.6%，A_err = 69.4%，Δ = -3.8pp（唯一负差距案例）
- **MIB（六组模型-任务）**：
  - 在 Gemma-2 ARC-Challenge 1% edge budget 下：A_ok = 80.3%，A_err = 54.4%，Δ = 25.9pp
  - 187–849 个模型错误存在于公开训练集（4.9%–41.1%），但错误覆盖仍不足以消除差距
- **改进尝试效果汇总**：
  - 错误重加权（50% KL 权重）：IOI A_err 从 9.3% 升至 37.0%，但 A_ok 从 98.7% 降至 91.0%
  - 错误丰富发现集（500错误+500正确）：A_err = 44.2%，A_ok = 91.4%
  - 电路扩展（IOI 案例）：A_err 从 14.2% 升至 75.1%，A_ok 仅降 0.41pp

## 相关工作脉络
- **Wang et al. (2023)**：提出 IOI 手动电路并定义归一化忠实度 F，本文将其作为基线并揭示 F=0.956 高忠实度下 A_err 仅 15.2% 的盲区。
- **Conmy et al. (2023) ACDC**：自动化电路发现方法，使用 KL 散度阈值剪枝；本文实验显示其在 IOI 上 A_err 仅为 11.8%–44.0%。
- **Hanna et al. (2024) EAP-IG**：基于 attribution patching 的电路提取；本文在 IOI 和 DOCSTRING 上系统评估其错误重现能力。
- **Bhaskar et al. (2024) Edge Pruning**：可学习门控稀疏电路方法；本文发现其在 mean ablation 下 A_err 仅 11.4%–44.0%。
- **Mueller et al. (2025) MIB**：机制可解释性基准；本文借用其公开提示和分割，扩展评估维度至 A_ok/A_err 分层。
- **Li & Janson (2024) Optimal Ablation**：学习替换常数最小化 KL；本文对比其与 mean/resample ablation 的错误重现差异。

## 局限性与未来方向
- **评估范围限制**：结论基于 IOI、DOCSTRING 和 MIB 等特定任务，未涵盖自由生成类任务或更大规模模型。
- **案例研究通用性**：IOI 扩展电路的发现（S-Inhibition 与 Name Mover 的关键作用）未证明在其他任务/电路中具有普适性。
- **恢复非穷举**：虽展示显著恢复可能，但未穷尽所有补救策略，也未证明可始终在保持 A_ok 的同时实现高 A_err。
- **误差集中假设**：部分差距可能源于错误集中在小 margin 区域（高 logit 敏感性），margin matching 可缩小但无法消除差距。

## 研究启发与可借鉴点
- **分层评估设计**：A_ok/A_err 的 stratum-based 评估框架可直接迁移至其他 MI 评估场景（如因果干预有效性检验）。
- **扩展搜索策略**：基于梯度筛选+精确评估的贪婪电路扩展方法可复用于其他任务的电路完善。
- **对照实验设计**：随机扩展、标量偏差、margin matching 三类对照有效区分了"偶然匹配"与"机制恢复"，值得借鉴。
- **与团队方向结合**：若团队关注模型鲁棒性或错误分析，可将 A_err 作为电路方法的硬约束指标；若研究抗幻觉机制，可探索 A_err 高重现电路是否更稳定。
- **评估基准构建**：当前 MIB 仅提供 A_ok 类指标，本文提出的 A_err 评估可扩展为社区标准，推动基准更新。

## 关键术语表
- **Circuit（电路）**：从模型计算图中提取的子图，旨在用紧凑子网络解释模型在特定任务上的行为机制。
- **Faithfulness（忠实度）**：电路在特定干预下复现全模型行为的程度，常用归一化 logit 差 F 或输出分布 KL 散度衡量。
- **Mean Ablation（均值消融）**：将电路中保留节点外的激活替换为对照样本的平均值。
- **Resample Ablation（重采样消融）**：使用 counterfactual 输入的真实激活替换被消融节点的激活。
- **Optimal Ablation（最优消融）**：学习输入无关的替换常数，最小化电路与全模型输出的 KL 散度。
- **A_ok / A_err**：电路在模型正确样本和错误样本上与模型预测一致的比例，是本文提出的分层评估指标。
- **S-Inhibition Heads（主语抑制头）**：IOI 电路中读取主语位置信息、抑制 Name Mover 复制主语的关键组件。
- **Name Mover Heads（命名移动头）**：IOI 电路中直接对间接对象或主语 logit 施加正/负影响的核心组件。

## 可复现要素
- **数据集**：IOI（100,000 提示测试池）、DOCSTRING（ACDC 基准）、MIB（ARC-Easy/Challenge、减法，公共可用）
- **代码开源**：ACDC、EAP-IG、Edge Pruning 均有官方实现；本文电路扩展搜索代码未明确声明开源
- **模型权重**：GPT-2 Small（ frozen）、NEELNANDA/ATTN ONLY 4L512W、Llama-3.1 8B、Gemma-2 2B、Qwen-2.5 0.5B
- **关键超参**：
  - EAP-IG：5 步插值，全词表 KL 目标
  - Edge Pruning：3,000 次更新，batch=32，lr=0.8，sparsity warmup=2,500 步
  - Optimal Ablation：Adam lr=0.002，batch=16，最小化验证 KL
  - IOI 扩展搜索：λ_P ∈ {0.0005, 0.002}，最多 32 轮添加、128 规则，容忍 A_ok 下降 ≤ 0.5pp
