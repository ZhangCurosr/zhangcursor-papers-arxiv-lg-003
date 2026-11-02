---
title: "When-Does-Muon-Actually-Win-A-Paired-Data-Time-and-Tuning-Bu"
source: https://arxiv.org/pdf/2610.21555v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:53:57"
---

# 论文速读：When-Does-Muon-Actually-Win-A-Paired-Data-Time-and-Tuning-Bu

## 一句话总结
本文在单卡 A100 上对 Muon 与 AdamW 进行了严格配对的基准测试，系统解耦了数据、时间与调参三个预算维度下的性能差异；结论显示 Muon 在数据效率上全面胜出，但参考实现的 Newton-Schulz 正交化开销过大，导致时间预算下语言模型反而落后于 AdamW，且 Muon 对调参候选池更敏感。

## 研究问题与动机
- Muon 在大模型实验中屡报优异的数据效率，但其矩阵正交化更新成本高、超参敏感，实际训练中的“赢”高度依赖评估口径（数据 vs 时间）。
- 现有研究多直接报告最终 loss 或同等 compute 下的Scaling曲线，缺乏对优化器步耗时、调参容错率及两者交叉影响的精细化拆解。
- 需要明确：在控制初始化、候选调参空间与测试集访问的前提下，Muon 的真实优势边界与工程取舍是什么？
- 不同工作负载（文本 vs 视觉、平衡 vs 极端长尾）可能显著放大或掩盖优化器的内在特性，需统一协议进行横向对比。

## 核心贡献（创新点）
- **三维预算的配对基准框架**：首次在单一硬件上同时控制数据预算、时间预算与调参预算，将 Muon 与 AdamW 的性能解耦为“等数据 AUC”“等时间 AUC”与“候选后悔值”，弥补了既往工作仅关注单一维度的不足。
- **细粒度的优化器耗时归因**：通过 CUDA event 精确测量 optimizer step 占训练总耗时的比例，揭示参考实现中 Muon 的辅助 AdamW 与 Newton-Schulz 迭代占据了 56.1%–69.6% 的训练时间，直接解释了时间预算下排名反转的成因。
- **调参受限下的预期后悔度量**：将 23 候选池随机打乱模拟工业级有限调参场景，量化在仅 1 个候选时 Muon 的预期验证 AUC 后悔率为 13.66%，显著高于 AdamW 的 7.52%，为超参敏感性提供实证依据。
- **冻结的一-shot 测试评估协议**：在训练、候选筛选与轨迹分析全程冻结测试集，最终仅执行一次性测试评估，杜绝数据泄漏，使 40 项主要测试指标具备强可信度。

## 方法详解
- **工作负载与模型**：两种语言模型（FineWeb-Edu，17.7M/33.6M 参数 causal LM，context=256，5,000 步）与两种视觉模型（CIFAR-100，2.7M 参数 ViT，平衡集与 100:1 长尾集，100 epoch）。所有实验在单 NVIDIA A100-SXM4-40GB 上进行，使用 bfloat16 与确定性数据顺序。
- **Muon 参数路由与迭代**：对 hidden attention 与 feed-forward 矩阵（含融合 QKV）应用 Muon，维护 Nesterov 动量 $M_t$ 并进行 Frobenius 归一化后，执行 5 步五次 Newton-Schulz 迭代：$X_{k+1} = aX_k + b(X_k X_k^\top)X_k + c(X_k X_k^\top)^2 X_k$，其中 $(a,b,c)=(3.4445, -4.7750, 2.0315)$，结果近似 $UV^\top$。Embedding、Norm、Bias 与 Output head 由辅助 AdamW 更新（Muon 覆盖约 26.7%–42.2% 参数）。
- **基线设置**：AdamW 对所有参数统一应用解耦权重衰减、梯度裁剪(1.0)、5% 线性 warmup、余弦衰减与 $\beta_2$ 调度，其余硬件/数据/种子完全对齐。
- **评估协议**：开发阶段每策略 23 候选，按数据归一化验证 loss AUC 为主目标、最终 loss 与 run ID 为次目标选择配置；全周期运行使用 5 个匹配 seed (101–105)。报告指标包括：等数据 AUC、等时间 AUC、等时间终点 loss、固定 crossover 点、以及冻结测试 loss。
- **统计与度量**：AUC 采用梯形积分插值；配对 bootstrap 重采样 10,000 次构建 95% 置信区间；跨 workload 汇总时等权平均各任务估计值；后悔值定义为候选池前 b 个的最优值相对全局最优的归一化差距。

## 实验与结果
- **数据集与基线**：FineWeb-Edu 语言模型（17.7M/33.6M）、CIFAR-100 平衡/长尾 ViT（2.7M）；基线为同配置 AdamW。
- **数据效率（等数据）**：Muon 在所有 4 个 workload 的验证 loss
