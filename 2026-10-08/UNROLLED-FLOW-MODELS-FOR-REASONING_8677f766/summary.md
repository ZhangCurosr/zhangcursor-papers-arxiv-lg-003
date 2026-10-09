---
title: "UNROLLED-FLOW-MODELS-FOR-REASONING"
source: https://arxiv.org/pdf/2610.09759v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:07:06"
---

# 论文速读：UNROLLED-FLOW-MODELS-FOR-REASONING

## 一句话总结
本文提出无展开流模型（UFM），通过在潜空间进行多步 Euler rollout 并在终点统一施加交叉熵损失，使流匹配语言模型的额外积分步数真正转化为推理深度；结合球面投影稳定长轨迹动力学与截断反向传播，在 ProsQA、Sudoku 与 Maze 等结构化推理基准上以不到 1/3 参数量显著超越标准 FLM/S-FLM 基线。

## 研究问题与动机
- **核心问题**：流匹配语言模型理论上可通过增加推理时积分步数实现自适应计算，但标准 FLM/S-FLM 在多步采样下准确率几乎不提升，无法利用额外步数进行深层推理。
- **目标1（理论刻画）**：需明确流模型的积分步数与推理能力之间的映射关系；构造性证明两阶 Transformer 参数化的流可精确求解有向图可达性问题，且所需 Euler 步数随目标节点距离单调增加。
- **目标2（训练缺陷归因）**：标准流匹配目标在每个随机采样时刻独立监督去噪预测（Eq. 4），从未评估模型自身前一步产生的状态是否能为后续更新提供可利用信息，导致多步动力学退化。
- **目标3（长轨迹稳定性）**：Sudoku、Maze 等任务需要更长答案序列与更多推理步，直接扩展 rollout 会导致潜态范数指数级增长，需设计数值稳定机制以支撑长程联合训练。

## 核心贡献（创新点）
1. **两步 Transformer 流可达性构造与步数-距离理论映射**：证明存在与图结构无关的两阶权重，其流速度场实现图邻接算子作用，Euler 离散化后第 k 步的状态支持集恰好为根节点 k 跳内可达顶点；与 prior 工作本质区别在于首次从 ODE 流视角严格建立“积分步数=推理跳数”的计算保障，而非仅依赖自回归隐状态堆叠。
2. **终端 Rollout 联合训练范式（UFM）**：仅在轨迹终点施加词汇解码与 CE 损失，反向传播穿过整个 rollout，使各步更新共同优化终端答案；与 FLM/S-FLM 独立时刻监督的本质区别在于将多步动态视为可微序列整体训练，激活了步间信息累积效应。
3. **球面重traction（Sphere Retraction）稳定机制**：对每步 Euler 更新的基座进行位置级球面投影 $\psi(z)=\sqrt{d}\,z/\|z\|_2$，限制潜态尺度爆炸；与常规梯度裁剪或权重衰减不同，该操作在局部保持切向速度，确保长 rollout 动力学的数值收敛性。
4. **截断 BPTT 与无参数多轨迹选择**：仅回传最后 $N_{\mathrm{back}}$ 步梯度以降低显存占用，并在推理时抽取 K 条轨迹后用平均 logit margin 筛选最优解；与 DRaFT 等微调方法的本质区别在于 UFM 从零训练且损失为终端 CE，截断设计直接适配流模型的 rollout 结构。

## 方法详解
- **潜状态演化**：答案位置的状态 $z_t \in \mathbb{R}^{L \times d}$ 由双向 DiT 处理 prompt $c$ 与流时间 $t$，预测潜目标 $\widehat{\pmb{x}}_\theta(z_t,t,c)$，速度场 $v_\theta=(\widehat{\pmb{x}}_\theta-z_t)/(1-t)$，Euler 积分 $z_{t_{k+1}}=z_{t_k}+\Delta t_k v_\theta^{(k)}$。
- **仅在终点解码**：中间时刻不输出词汇分布，仅最后时刻执行 $W_{\mathrm{dec}}$ 线性投影：$\widehat{\pmb{y}}=\mathrm{softmax}(z_{t_N}W_{\mathrm{dec}}^\top)$；ProsQA 用标准 softmax，长答案任务改用 StableMax 防数值崩溃。
- **终端 Rollout 损失**：训练时从随机 $t_0$ 出发，沿子区间 $[t_0,1]$ 运行 $N_{\mathrm{train}}$ 步，损失为 $\mathcal{L}_{\mathrm{UFM}}=\mathbb{E}_{(c,y),\xi,t_0}[\mathrm{CE}(\mathrm{softmax}(z_{t_N}W_{\mathrm{dec}}^\top),y)]$；强制模型学习“前期更新为后期积累有用表征”。
- **截断反向传播**：前 $N_{\mathrm{train}}-N_{\mathrm{back}}$ 步以 `sg()` 截断梯度前向传递，仅最后 $N_{\mathrm{back}}$ 步回传联合优化；在 A100 上实现 $N_{\mathrm{train}}=24$ 长轨迹的可训练性。
- **球面重traction**：更新基座改为 $y_{t_k}=\psi(z_{t_k})$，欧拉步写作 $z_{t_{k+1}}=y_{t_k}+\frac{\Delta t_k}{1-t_k}(\widehat{\pmb{x}}_\theta-y_{t_k})$，新状态再次投影后作为下一步输入；消融显示移除后 Sudoku-Extreme 精度从 67% 跌至 55%。
- **多轨迹采样与选择**：推理时初始化 $z_0=\sigma\xi$，独立跑 $K$ 条 rollout；用无参数 margin 评分 $\widehat{\delta}=\frac{1}{L}\sum_i(\ell_i^{(1)}-\ell_i^{(2)})$ 选优，理论动机对应 Theorem 1 的分离边界 $\delta$。

## 实验与结果
- **数据集与基线**：ProsQA（有向图可达性，≤4 跳）、Sudoku-Hard（2k 测试）、Sudoku-Extreme（422,786 测试）、Maze-Hard（30×30，最短路径>110 格）；对比 FLM、S-FLM（均 28.6M，作者开源代码复现）及递归推理器文献值。
- **ProsQA 步数-性能曲线**：UFM（$N_{\mathrm{train}}=5$）随推理步数增加准确率从 12% 升至约 97%；FLM/S-FLM 曲线近乎平坦，证实独立时刻监督无法激活推理深度。
- **Sudoku-Hard**：UFM（8.4M, $N_{\mathrm{eval}}=128$）达 $86.9\pm0.9\%$，FLM 51.9%，S-FLM 50.9%，参数量仅为基线 29% 且训练时间更短（1.1h vs 1.6/1.7h）。
- **Sudoku-Extreme**：UFM 单轨迹 Pass@1 为 $7
