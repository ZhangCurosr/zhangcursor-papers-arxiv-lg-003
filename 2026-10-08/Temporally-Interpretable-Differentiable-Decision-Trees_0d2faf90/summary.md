---
title: "Temporally-Interpretable-Differentiable-Decision-Trees"
source: https://arxiv.org/pdf/2610.10367v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:13:59"
---

# 论文速读：Temporally-Interpretable-Differentiable-Decision-Trees

## 一句话总结
本文将时间维度引入可解释强化学习，提出融合Action Chunking的两种新型策略梯度算法与在线树重构机制（ITTR），使可微决策树（DDTs）能够连续输出短期规划；在4个仿真环境中，Warm-start蒸馏的Chunked DDT以最多减少80%参数的代价，达到了与神经网络策略相当的性能。

## 研究问题与动机
- **核心问题**：现有可微决策树（DDTs）仅支持单时间步动作输出，无法在顺序决策任务中展示多步短期计划，导致人类操作者/审计者难以从时间维度理解Agent行为。
- **现有方法不足**：
  1. 传统DDTs与纯离散查询的Action Chunking方法存在“单步逻辑 vs 多步规划”的错位，用户每隔H步才能看到一次计划，中间时段透明度为零。
  2. 强制每步查询Chunked Policy会破坏训练分布，导致轨迹不稳定与性能下降；而现有的Temporal Ensemble会引入非Markovian的平滑约束，在强随机环境（如LL-H）中显著劣化。
  3. Action Chunking会成倍膨胀DDTs的参数规模，若不进行结构压缩，树的稀疏性与可解析性将被大幅削弱。
  4. 现有可解释AI研究多聚焦静态逻辑透明（白盒/后验解释），缺乏对部署期“时序行为意图”可观测性的系统刻画。

## 核心贡献（创新点）
1. **Temporal Ensemble Policy Gradient**：将EMA时间集成机制引入策略梯度，使Chunked DDT每步输出由最近H个Chunk聚合的连续动作，用户可实时观测短期计划；与直接离散查询的本质区别在于它通过加权历史Chunk维持动作连续性，在保持可解释性的同时避免分布偏移。
2. **Temporal Prediction Policy Gradient**：将Chunk内首动作用于实际控制，其余动作通过历史状态重建训练为未来预测；相比Ensemble方法不强制时间一致性，更适配高随机/非平稳环境，且Prediction目标天然增强了对未来轨迹的透明描述。
3. **Information-Theoretic Tree Restructuring (ITTR)**：基于策略互信息梯度与叶节点访问频率，在训练过程中动态剪枝/加深；与静态蒸馏或后处理裁剪的本质区别在于它能在线感知策略复杂度饱和状态，自适应维持参数效率。
4. **Warm-start蒸馏训练范式**：先用CART将NN策略蒸馏为确定性树，再转换为DDT衔接上述Chunked策略梯度；实验证明该策略在4个环境中均可稳定收敛，并用最多80%更少的参数匹配Neural Network性能。

## 方法详解
- **基础设定**：采用ICCT变体DDT（Straight-Through Trick直接优化脆性树，叶子节点携带线性子控制器），结合Robust Policy Optimization (RPO) 策略梯度。Chunk Horizon设为 $H=10$。
- **Action Chunking Policy Gradient**：策略输出 $\mathrm{a}_{t:t+H}$，优势函数扩展为 $\hat{A}_{t+H} = \sum_{t'=t}^{t+H-1} \gamma^{t'-t} r_{t'} + \gamma^H V_\phi(s_{t+H}) - V_\phi(s_t)$，使梯度信号覆盖整段Chunk的累积回报。
- **Temporal Ensemble PG**：维护长度H的Chunk历史队列 $\rho$，通过Eq.6提取各Chunk对应当前时间步的动作组成 $\mathrm{a}^\star$，再经EMA（系数$\eta$）聚合得最终动作 $\mathrm{a}_t$（Eq.7）。损失为加权对数概率和（Eq.8），$\eta<1$时为有偏估计，$\eta=1$退化为无偏形式。
- **Temporal Prediction PG**：主RL损失仅作用于首动作 $L_1 = \log \pi(\mathrm{a}_t|s_t)\hat{A}_t$；辅助预测损失 $L_2 = \sum_{i=1}^{H-1} \log \pi(\mathrm{a}_t|s_{t-i})$ 迫使后续Chunk元素重现当前动作，等价于训练网络预测未来 $\mathrm{a}_{t+i}$。总损失 $L = L_1 + \lambda L_2$（Eq.11），无需跨步平滑约束。
- **ITTR 算法**：
  - **加深判据**：当平均策略复杂度梯度 $\frac{1}{N}\sum \nabla I^\pi(\mathbf{s};\mathbf{a}) < \epsilon$ 且回合奖励偏低时触发；通过求解Eq.13找到高频访问但Action Uncertainty低的叶节点 $l^*$，分裂为带新权重/阈值的比较节点与两个子叶，提升模型容量。
  - **剪枝判据**：叶节点访问次数 $v_i < k$ 时触发，剪除该叶并将其兄弟子树上提替换父节点，保障树结构连通。
  - **工程约束**：设置最小重结构间隔 $n$、全局步数门槛 $m$ 与最低复杂度估计量 $E$，防止频繁扰动破坏策略收敛。

## 实验与结果
- **环境**：Lane Keeping (LK), Inverted Pendulum V5 (IP), Lunar Lander V3 (LL), Lunar Lander V3 Hard (LL-H，叠加强风、高湍流与动作高斯噪声 $\mathcal{N}(0, 5\text{e-}3)$)。
- **基线**：15种方法，涵盖不同规模MLP（Sparse/Big）、CART蒸馏树、DDT原生/Ensemble/Prediction变体及Warm-start变体；评估指标为 episodic return 与参数总量。
- **关键结果**：
  - 从零训练Chunked DDT困难：DDT-ENSEMBLE在LL-H仅达 $160.8 \pm 13.1$，DDT-PREDICTION在LK与LL-H存在Seed敏感；参数从基础DDT的 $\sim75$ 膨胀至 $440\text{-}506$。
  - Warm-start表现最优：WARM-ENSEMBLE与WARM-PREDICTION在所有环境中取得最高或统计等价回报。LL-H中WARM-ENSEMBLE达 $258.4 \pm 1.9$，超越无Chunk的DDT（$247.5 \pm 3.5$）。
  - 参数效率：在LK、IP、LL三个环境完全匹配MLP/MLP-BIG性能；整体相比同等Chunking MLP节省 $51\text{-}79\%$ 参数，最高减少约80%。
  - ITTR动态演化：IP等简单环境保持2叶稳定；LK/LL/LL-H呈现“先加深后剪枝”的信息瓶颈现象，逻辑流部分参数仅占MLP的不到1/10（Table 3）。
- **最强提升**：WARM-ENSEMBLE在LL-H相对DDT-ENSEMBLE提升约60.6%，且以更优参数效率换取更高稳健性。

## 相关工作脉络
1. **可微决策树（DDTs, Silva et al. 2019; Paleja et al. 2022, 2024）**：本文直接在其ICCT架构上扩展时序抽象，首次将训练期在线重结构（ITTR）引入DDTs，弥补了原方法结构静态、仅支持单步动作的缺陷。
2. **可解释AI vs xAI**：区别于依赖Prompt或Saliency Map的后验解释（xAI），本文坚持White-box原生透明性，并将“时间维度”确立为顺序决策场景下独立的可解释性度量。
3. **Action Chunking RL（Li et al. 2025b; Hahn & Choi 2025）**：本文未沿用离线或纯Open-loop范式，而是针对Online DDT设计Temporal Ensemble/Prediction两种策略梯度，解决chunked树在多步推理中的策略对齐问题。
4. **CART蒸馏与树压缩（Hu et al. 2019; Paleja et al. 2024）**：借鉴CART生成初始脆性分割的思路，但进一步通过ITTR在RL微调整程中动态修剪冗余分支，实现了“蒸馏初始化
