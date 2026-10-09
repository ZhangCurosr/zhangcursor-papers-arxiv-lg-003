---
title: "WAM-CACHE-STALENESS-BOUNDED-KV-REUSE-FOR-EFFICIENT-WORLD-ACT"
source: https://arxiv.org/pdf/2610.11401v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 17:03:56"
field: "机器人策略推理加速"
keywords: ["World Action Model", "KV Cache", "Training-Free Acceleration", "Robot Manipulation", "Diffusion Transformer", "Token Refresh"]
innovations: ["首个训练无关的 WAM 视频 DiT prefill KV 复用框架，用 action expert 跨注意力 + latent surprise + age bound 三色信号联合选择刷新集", "证明 policy attention 比 visual drift 更能预测 stale KV 对下游动作精度的影响，oracle 漂移刷新仍存在 25pp 差距", "引入 age bound 严格限定 KV stale 时长，打破 rank-based 选择自引用误差累积，默认 G=2 即可将成功率损失压至 1.8pp"]
benchmarks: ["RoboTwin 2.0", "LIBERO", "AIRBOT Play 真机（Carrot in Bowl / Stack Cubes）"]
---

# 论文速读：WAM-CACHE-STALENESS-BOUNDED-KV-REUSE-FOR-EFFICIENT-WORLD-ACTION-MODELS

## 一句话总结
论文提出 WAM-Cache，首个面向世界动作模型（WAM）视频 DiT prefill 的训练无关 KV 缓存框架，通过"动作专家注意力 + 视觉潜变量惊喜 + 最大年龄约束"三色信号联合选择刷新 token，在保留 stale KV 的情况下将 prefill FLOPs 减少 32–42%，仿真任务成功率仅下降 0.7–1.8 个百分点。

## 研究问题与动机
- WAM 在闭环控制中每个 chunk 需运行视频 DiT 对当前观测做单次 prefill，将视觉 token 编码为层级 KV 对供 action expert 查询；prefill 在 Fast-WAM（$N=2$ 步去噪）下占 DiT 总 FLOPs 的 83.5%，是主要瓶颈。
- 现有训练无关加速方法（DeepCache、X-Cache 等）只能复用同一 chunk 内 denoising 步骤间的特征，无法触及单次 prefill 本身；已有 token 削减基线在匹配 FLOPs 预算时会损失 4–38 个百分点成功率。
- 直觉上"刷新视觉漂移最大的 token"甚至使用 oracle 预测真实 KV 漂移后，成功率仍远低于稠密模型（ plateau 在 ~65% vs. 稠密 ~90%），说明刷新策略不能仅依赖局部表征漂移。
- 关键洞察：下游动作精度由 action expert 跨注意力分配的权重决定，而非视觉变化本身；因此刷新集必须同时反映"政策敏感度"。

## 核心贡献（创新点）
- **首个训练无关 WAM KV 复用框架**：在 video DiT prefill 层面实现层间 KV 跨 chunk 保留，无需 retrain 或压缩预训练 backbone。与 FastV/ToMe 等一次性裁减 token 的方法本质不同：本文不丢弃 token，而是选择性重算。
- **发现刷新信号应是 policy attention 而非 visual drift**：即使在 ground-truth KV 漂移 oracle 下 change-only 刷新仍落后稠密 25 个百分点；action-attention alone 即可恢复到 81.5%，揭示"策略关注哪里"比"哪里变化了"更重要。
- **引入 age-bound 打破自引用误差累积**：content-based rank 选择会使低优先级 token 长期不被刷新、误差叠加；固定年龄上限 $G$ 强制周期性刷新，把相对稠密差距从 9.5 个百分点缩小到 1.8 个百分点。
- **在 RoboTwin 2.0 / LIBERO / 真机三线验证**：统一框架无需重新调参即可跨场景工作；prefill CUDA 延迟加速随视觉上下文增大趋近理论界 $1/\bar{\rho}$。

## 方法详解
- **缓存状态**：每个 token $i$ 维护四层信息——层级 KV 对 $\{(\mathbf{K}^l[i],\mathbf{V}^l[i])\}_{l=1}^L$、最后刷新时刻的参考嵌入 $\mathbf{z}_{\tau_i}[i]$、连续重用计数器 $g_i$、上一 chunk 的 action expert 平均跨注意力质量 $\mathbf{p}_t[i]$。
- **刷新集构造**：$\mathcal{R}_t = \mathcal{S}_t \cup \mathcal{P}_t \cup \mathcal{A}_t$，三者取并集（同一 token 只计一次）：
  - **Latent Surprise**：$\mathbf{s}_t[i] = \frac{\|\mathbf{z}_t[i] - \mathbf{z}_{\tau_i}[i]\|_2}{\frac{1}{2}(\|\mathbf{z}_t[i]\|_2 + \|\mathbf{z}_{\tau_i}[i]\|_2) + \epsilon}$，取 TopK($b_s$) 作为 $\mathcal{S}_t$。衡量当前嵌入相对"上次刷新时观测"的漂移，而非相对上一帧。
  - **Action-Conditioned Selection**：记录 action expert 第一去噪步的 head-averaged cross-attention $\mathbf{A}_{t-1}^l$，聚合得到 $\mathbf{p}_t[i]$，取 TopK($b_a$) 作为 $\mathcal{P}_t$。该信号在前一 chunk 已产生，本轮 prefill 前零成本可用。
  - **Age-Bound**：$\mathcal{A}_t = \{i : g_i \geq G\}$，达到上限时无论 content score 如何均强制刷新；刷新后 $g_i \leftarrow 0$，未刷新则 $g_i \leftarrow g_i+1$。作者默认 $G=2$。
- **稀疏 prefill**：video DiT 仅对 $\mathcal{R}_t$ 中的 token 计算 query/key/value 投影并运行 self/cross-attention，KV 对原地覆盖；queries 仍 attend over 全部 $S_v$ 个 keys/values，保证新特征能吸收静态上下文。
- **成本分析**：单模块 prefill FLOPs $C_{\text{dense}} \approx 8S_v d^2 + 4S_v^2 d + 4S_v d^2 + 4S_v S_c d + 4S_v d d_f$；由于每项均线性于 query token 数，刷新比为 $\rho_t = |\mathcal{R}_t|/S_v$ 时相对节省 $1-\rho_t$。文本 KV 每 episode 投影一次，不影响节省比例。理论摊销上界 $\bar{\rho} \leq b_s + b_a + 1/(G+1)$。

## 实验与结果
- **实现基线**：Fast-WAM（Wan2.2-5B 视频 DiT + 1B action expert，bfloat16，单卡 RTX PRO 6000 Blackwell，$N=2$ 去噪步）。
- **RoboTwin 2.0**（50 任务×100 episode，clean & randomized）：默认设置 $(b_a,b_s)=(0.4,0.1)$ 下 FLOPs 减少 41.6%（clean）/39.3%（rand），成功率分别损失 1.8 / 1.3 个百分点；更大预算 $(0.6,0.1)$ 下节省 28.2%/37.0% 且仅损失 0.1 个百分点。最强 baseline VLA-Cache 同预算下损失 ≥4.0 个百分点。
- **LIBERO**（4 suite×10 tasks×50 episode）：默认 $(0.5,0.1)$ 节省 32.3%，平均 SR 95.85% vs 稠密 96.50%（损失 0.7 pp）；Long suite 因 phase transition 下 inherited attention 滞后出现主要损失。
- **真实机器人**（AIRBOT Play 双臂，20 trial×2 task）：节省 40.2% FLOPs，Carrot in Bowl 保持 90.0%，Stack Cubes 仅少 1 trial（75.0% vs 80.0%），平均 82.5% vs 稠密 85.0%（2.5 pp）。
- **CUDA 延迟**：在 $S_v=120$ 净加速 1.23×/1.58×，放大到 $S_v=480/1080$ 时加速达 1.47×/1.67×，接近理论界 $1/\bar{\rho}$。
- **消融**（Table 3）：Random 26.9%；Pixel diff 60.4%；Oracle K/V drift 65.3%（25 pp 差距）；Surprise alone 60.6%；Attention alone 81.5%；Union 88.4%；加上 $G=2$ 后达 88.36%（close 1.8 pp gap）。

## 相关工作脉络
- **Fast-WAM / GigaWorld-Policy**：去掉 future imagination 后 prefill 成为主导成本；本文首个将其 sparsify。
- **VLA-Cache / Eventful Transformer**：在 VLA 统一骨干内复用 static token，但无 reuse 上界；本文用 policy attention 而非 video-to-text attention 选 token，并加 age bound 限 stale。
- **FastV / ToMe**：frame 内 prune/merge token，一次性丢弃信息；本文保留全部 token 并按 chunk 动态刷新。
- **DeepCache / X-Cache**：复用同 chunk 内 denoising steps 的特征；WAM 的 prefill 是单次前向、无 denoising 轨迹可复用，思路正交。
- **SnapKV / H2O / C³ache**：LLM/VLA 场景下基于命中率或 gate 选 token；本文信号来自 action expert 跨注意力 + 跨 chunk 年龄约束，适配 WAM 的双模块解耦架构。
- **WorldCache**：异构 token 缓存加速 world model，但未针对 WAM 的 prefill bottleneck 设计 policy-aware 刷新。

## 局限性与未来方向
- 小 token 数时 weight loading / CUDA graph replay 开销占比高，实测加速落后 FLOPs 节省比例；需更大上下文才能逼近理论界。
- 当前仅处理单帧观测（head + wrist cameras concat）；多帧历史、更高分辨率、更多 camera view 下的 stale 误差演化未验证。
- Age bound 限制 reuse 时长但不限制单次 error magnitude，极端场景下强制刷新仍可能引入波动。
- 未讨论与 retraining-based 方法（蒸馏、压缩、step reduction）的联合优化边界。
- 未来可探索自适应 $G$、per-layer 差异化 budget、或与 diffusion step reduction 的组合调度。

## 研究启发与可借鉴点
- **刷新信号设计应从"表征漂移"转向"下游敏感度"**：cross-attention 作为零额外成本的代理信号，在 Decoupled 架构（encoder + decoder）下尤其有效。
- **Age-bound 是解决 rank-based 选择自引用累积的通用技巧**：任何 token-skipping/caching 机制均可借鉴此"周期性强制刷新"来稳态控制 staleness。
- **三信号取并集而非加权拼接**：surprise 与 attention 选出的 token 子集高度不相交，简单 union 即可互补而不引入超参调优负担。
- **在消融中设置 Oracle K/V drift 作上界**：直观验证"变化"不是正确信号，论证更有说服力，值得在类似系统研究中复现。
- **本团队可迁移场景**：任何"观测编码 backbone + 决策头"解耦架构（如 VLA、世界模型 rollout、多模态 agent）均可套用该 KV 复用范式。

## 关键术语表
- **World Action Model (WAM)**：将视频生成 DiT 与 action expert 耦合，以语言条件驱动机器人操作的一般化策略范式。
- **Prefill**：视频 DiT 对当前观测做一次完整前向，输出层级 KV 对供后续 action decoding 查询；本文的计算瓶颈所在。
- **KV Cache（层间）**：保存各层 self-attention 的 key/value，跨 chunk 复用以避免重复编码静态视觉区域。
- **Latent Surprise**：当前 patch embedding 与上次刷新时参考 embedding 的归一化 L2 距离，用于检测视觉漂移。
- **Action-Conditioned Selection**：借用 action expert 上一 chunk 的 head-averaged cross-attention 质量 $\mathbf{p}_t[i]$，优先刷新策略最关注的 token。
- **Age Bound (G)**：强制刷新连续未更新 $G$ 个 chunk 的 token，上限控 stale 时间；本文默认 $G=2$。
- **Fast-WAM**：去除 future imagination、仅单次 prefill 的轻量 WAM；本文实验平台。
- **刷新比 $\rho_t$**：每 chunk 实际重算 token 数占总视觉 token 数的比例，直接决定 FLOPs 节省 $1-\rho_t$。

## 可复现要素
- **数据集**：RoboTwin 2.0（公开 benchmark）、LIBERO（公开 benchmark）、AIRBOT Play 真机任务（作者自采集 100 demonstrations/task）。
- **代码**：Project page https://dingkai0302.github.io/wam-cache/；论文未明确声明开源仓库，未提及。
- **权重**：使用 Fast-WAM 发布 checkpoint（Wan2.2-5B + 1B action expert）；论文未提及自有权重开源。
- **关键超参**：$G=2$；$(b_a, b_s)$ 默认 clean RoboTwin 2.0 为 (0.4, 0.1)、LIBERO 为 (0.5, 0.1)，randomized RoboTwin 2.0 为 (0.4, 0.2)；action-denoising 步数默认 $N=2$（附录 D 亦报告 $N=10$ 结果）。
- **环境**：单卡 NVIDIA RTX PRO 6000 Blackwell，bfloat16；CUDA graph replay 测延迟。
