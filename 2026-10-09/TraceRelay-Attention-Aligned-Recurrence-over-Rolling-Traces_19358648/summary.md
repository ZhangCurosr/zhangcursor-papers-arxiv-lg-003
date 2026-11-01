---
title: "TraceRelay-Attention-Aligned-Recurrence-over-Rolling-Traces"
source: https://arxiv.org/pdf/2610.11743v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:13:01"
---

# 论文速读：TraceRelay-Attention-Aligned-Recurrence-over-Rolling-Traces

## 一句话总结
TraceRelay 提出了一种以注意力对齐的滚动递推架构，通过右向注意力生成因果延迟增量、固定加法相位累加递推与左向注意力读取残差增强流，实现并行预填充与有界缓冲延续；在三项合成诊断任务上验证了该设计，但证明中间层 trace 带宽的最优选择高度依赖任务类型，无法跨任务泛化。

## 研究问题与动机
- 全注意力机制面临二次计算开销与线性增长的 KV cache，且“可用不等于会用”（Lost in the Middle、RULER 揭示长度敏感性），需将可访问历史与有用持久信息显式分离。
- 现有递推/SSM（如 Mamba）与注意力并非自动协同：窗口过小损害局部上下文利用，过大则弱化长期记忆学习（SWAX），缺乏解耦的读写调度机制。
- 传统架构难以在不引入显式 slot 提交或稠密状态转移矩阵的前提下，实现递推状态的并行计算与因果续传。
- 核心疑问：能否让注意力决定“写什么/读什么”，而递推仅承担简单、固定的时序传输？

## 核心贡献（创新点）
- **提出右写-延迟-相位扫描-左读的滚动递推有向图**。与 Mamba/Mamba-2 依赖输入依赖的稠密状态转移不同，本文递推传输律完全固定（无学习衰减），动态性全部交由注意力与投影层承担。
- **实现 stride-wise prefix sum 并行预填充与有界续传**。不同于 Block-Recurrent Transformer 或 Recurrent Memory Transformer 的块级状态传递，TraceRelay 将循环依赖按 stride $W$ 分组转化为并行累加，推理时仅需维护各 lineages 的相位尾部，无需保留完整历史 KV。
- **验证“长时序 reach、窄表征带宽”的非均匀容量分配假设**。与 H-Net 压缩序列分辨率的思路不同，TraceRelay 保持原始 token 网格，仅在中间层 trace 宽度与读取窗口上作差异化配置，并在 Dyck 与 Most-Freq 上揭示相反的宽度偏好。
- **构建统一控制的三项合成诊断协议**。在 Equal Repeats、Bounded Dyck 与 Most-Freq 上执行严格配对消融（phase inheritance on/off）与多维度诊断（距离/深度切片、Top-k/集/终止分解），剥离继承贡献与带宽效应。

## 方法详解
- **右向延迟写入**：位置 $i$ 通过右向局部多头注意力 $o_i^R = \mathrm{Attn}_R(h_i, h_{i:i+R})$ 从下层流抽取候选，经无偏投影 $P_\delta$ 与 tanh 限幅得到增量 $\delta_i = \alpha \tanh(P_\delta(o_i^R))$（实验取 $\alpha=\pi$）。该增量因果延迟 $W$ 步后送达，即 $u_t = \delta_{t-W}$（$t<W$ 时补零）。
- **固定相位累加递推**：$\theta_t = \mathrm{wrap}(\theta_{t-W} + u_t)$，形成 $W$ 条交织的 trace lineage。复数坐标下等价于坐标级乘法 $\exp(i u_t)$，递推本身无学习参数，仅靠输入依赖增量驱动相位演化。
- **左向注意力读取**：有效 lineage 处，残差流与相位读出融合为 $m_t = h_t + P_\phi([\cos\theta_t;\sin\theta_t])$，随后经左向局部注意力 $o_t^L = \mathrm{Attn}_L(m_t, m_{t-L:t})$ 提取上下文，最终经残差连接与 FFN 输出 $h_t^\ell = \tilde{h}_t + \mathrm{FFN}(\tilde{h}_t)$。未初始化 lineage 的相位读出强制置零，避免 $\cos(0)$ 特征泄漏。
- **因果性与并行执行**：因 $W>R$，$u_t$ 仅依赖 $t-W+R<t$ 的过去输入，$m_t$ 严格因果。Prefill 时按 $r\in\{0,\dots,W-1\}$ 分组计算 $\theta_{r+nW} = \mathrm{wrap}(\sum_{k=0}^n u_{r+kW})$，仅需 reshape、累加与 wrap，兼容 FlashAttention-2；续传只需保存各 lineage 相位尾部，支持有界 buffer 推理。

## 实验与结果
- **配置**：$H=64$，FFN=256，3 层，4 heads，$L=15$，$R=7$，$W=8$，trace 宽度 $d_c\in\{16,32,64\}$，AdamW($3\!\times\!10^{-4}$, $\beta=(0.9,0.999)$, wd=0.01)，100 步 warmup，batch=64，BF16，FlashAttention-2，上限 20,000 步，连续两次 ID≥99% 早停。
- **Equal Repeats（ID $T=256$）**：phase inheritance 开启达 $98.81\%\sim99.69\%$，关闭仅 $50.73\%\sim51.63\%$，配对增益约 47–49 pp；继承组 4,250–10,250 步早停，关闭组耗满 20,000 步仍徘徊在随机水平，证明递推继承贡献显著。OOD（$T\geq512$）两者均暴跌至≈33% 随机基线，无法外推。
- **Bounded Dyck（$k=8,m=10$）**：ID 普遍 $>98\%$；OOD $T=4096$ 呈现“越宽越好”：$d_c=64$ 达 $76.34\%\pm1.81$，$d_c=
