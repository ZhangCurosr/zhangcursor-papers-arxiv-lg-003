---
title: "WAM-CACHE-STALENESS-BOUNDED-KV-REUSE-FOR-EFFICIENT-WORLD-ACT"
source: https://arxiv.org/pdf/2610.11401v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:14:28"
field: "具身智能高效推理"
keywords: ["World Action Models", "KV Cache", "Training-Free Acceleration", "Robot Manipulation", "Sparse Prefill", "Attention-Guided Selection"]
innovations: ["首次针对WAM视频DiT prefill提出免训练KV缓存框架", "发现策略注意力而非视觉漂移是刷新决策的关键信号", "引入年龄上限打破自referential误差累积实现staleness-bounded reuse"]
benchmarks: ["RoboTwin 2.0", "LIBERO", "AIRBOT Play真机实验"]
---

# 论文速读：WAM-CACHE: STALENESS-BOUNDED KV REUSE FOR EFFICIENT WORLD ACTION MODELS

## 一句话总结
本文提出 WAM-Cache，这是首个面向 World Action Models（WAM）中视频 DiT prefill 的**免训练 KV 缓存加速框架**。它通过"动作专家注意力 + 视觉潜在惊喜 + 年龄上限"三信号联合选择稀疏刷新集合，在 RoboTwin 2.0、LIBERO 和真机实验上节省 **32–42% prefill FLOPs**，成功率仅下降 **0.7–2.5 个百分点**。

## 研究问题与动机
1. **核心瓶颈**：WAM 在闭环控制中每个 chunk 都需重跑视频 DiT prefill（编码当前观测为层叠 KV 对），该步骤占 Fast-WAM 每 chunk FLOPs 的 **83.5%（N=2 步）至 50.2%（N=10 步）**。
2. **现有方法不足**：已有免训练加速（FastV、ToMe、Eventful、VLA-Cache 等）均通过剪枝/合并/跨帧复用 token 来降低视觉 token 数量，但**从未触及视频 DiT prefill 这一最大成本项**——它们要么针对单帧内 token 剪枝，要么针对 VLA 统一主干设计，不适用于 WAM 中解耦的视频 DiT 与动作专家架构。
3. **直觉陷阱**：作者发现"刷新视觉上漂移的 token"这一直觉规则在 RoboTwin 2.0 上成功率仅达 53.1%，比 dense 基线低 **37 个百分点**；即便使用 oracle 预测真实 KV 漂移，也只能恢复到 65.3%，仍差 **25 点**。
4. **关键洞察**：下游动作准确率由**动作专家 attends 的位置**决定，而非"什么移动了"——即 stale KV 的危害程度取决于策略对该 token 的依赖度，而非表征漂移量。

## 核心贡献（创新点）
1. **首个针对 WAM 视频 DiT prefill 的免训练 KV 缓存框架**：与 FastV/ToMe 等 token 剪枝方法本质不同，WAM-Cache 保留全量 token，仅对稀疏刷新集做 prefill，动作专家无需任何修改即可在混合缓存上运行。
2. **提出"策略注意力驱动"而非"视觉漂移驱动"的刷新策略**：发现 change-based 规则（像素差、潜在惊喜、oracle KV 漂移）均陷入性能平台（≤65.3%），而动作专家 cross-attention  Alone 即可恢复到 81.5%，联合两信号后可达 88.4%。
3. **引入严格年龄上限（age bound）打破自referential 误差累积**：因两个内容信号均基于 rank 选择，低优先级 token 可无限次逃过刷新导致误差compound；设 G=2 强制每 G+1 个 chunk 至少刷新一次，将成功率高出 **7.7 个百分点**，最终仅差 dense 1.8 点。
4. **理论成本分析**：推导 prefill FLOPs 随刷新比 ρ̄ 线性缩放公式 C(ρ_t) = ρ_t · C_dense，并给出重算比上界 ρ̄ ≤ b_s + b_a + 1/(G+1)，证明选择信号 overhead 可忽略。
5. **可扩展性验证**：在 1×/4×/9× 更大视觉上下文下，CUDA 延迟加速从 1.23× 增至 1.67×，逼近 1/ρ̄ 理论上界（98%）。

## 方法详解

### 4.1 总体架构：Staleness-Bounded KV Reuse
- **Token 级缓存状态**（每 token i）：① 层叠 KV 对 {(K^l[i], V^l[i])}_{l=1}^L；② 参考嵌入 z_{τ_i}[i]（最近刷新 chunk 的 patch embedding）；③ 年龄计数器 g_i；④ 上一 chunk 动作专家注意力质量 p_t[i]。
- **每 chunk 流程**（chunk t>0）：
  1. 提取 patch embedding z_t = Enc(o_t)
  2. 构造稀疏刷新集 R_t = S_t ∪ P_t ∪ A_t（见下文三信号）
  3. 视频 DiT 仅在 R_t 上做 prefill，hidden states H^l_t[R_t] 重算后原地覆盖缓存中的 K^l、V^l
  4. 动作专家 query 也只在 R_t 上形成，attend 过全部 S_v 个 KV
  5. 更新 τ_i ← t（刷新 token）或 g_i ← g_i+1（未刷新 token）

- **计算代价**（单层模块）：
  $$C_{dense} ≈ \underbrace{8S_vd^2+4S_v^2d}_{\mathcal{F}_{SA}} + \underbrace{4S_vd^2+4S_vS_cd}_{\mathcal{F}_{CA}} + \underbrace{4S_vdd_f}_{\mathcal{F}_{MLP}}$$
  由于只有 |R_t|=ρ_t·S_v 个 token 形成 query，重算成本 C(ρ_t)=ρ_t·C_dense，节省比例 1−ρ_t。

### 4.2 信号一：Latent Surprise（视觉潜在惊喜）
对 token i 计算其当前嵌入与参考嵌入的归一化距离：
$$s_t[i] = \frac{\|z_t[i] - z_{\tau_i}[i]\|_2}{\frac{1}{2}(\|z_t[i]\|_2 + \|z_{\tau_i}[i]\|_2) + \epsilon}$$
取 TopK(s_t, b_s) 作为 S_t。**局限性**：仅反映局部表征漂移，不反映策略依赖度；单独使用成功率仅 60.6%（表 3）。

### 4.3 信号二：Action-Conditioned Selection（动作条件选择）
记录上一 chunk 动作专家在第 l 层的 head-averaged cross-attention A^l_{t-1} ∈ R^{S_a×S_v}，计算 token i 的平均注意力质量：
$$p_t[i] = \frac{1}{L_a S_a} \sum_{l=1}^{L_a}\sum_{m=1}^{S_a} A^l_{t-1}[m,i]$$
取 TopK(p_t, b_a) 作为 P_t。**优势**：该分数来自前序 chunk，在当前 prefill 前即可得，零额外开销。单独使用即达 81.5% 成功率（表 3）。

### 4.4 信号三：Age-Bounded Refresh（年龄上限刷新）
收集所有 g_i ≥ G 的 token 构成 A_t，无论其内容分数如何强制刷新。更新规则：
$$g_i \leftarrow \begin{cases} 0, & i \in \mathcal{R}_t \\ g_i+1, & i \notin \mathcal{R}_t \end{cases}$$
**核心作用**：打破自 referential 误差累积——两个内容信号均基于 rank，低优先级 token 可永远逃过刷新；年龄上限严格限制任何 token 的最长 stale 时间。**效果**：设 G=2 即恢复 7.7 点成功率，最终仅比 dense 低 1.8 点。

### 超参数
- 默认 G=2（所有 benchmark 不变）
- RoboTwin 2.0: (b_a, b_s) = (0.4, 0.1) 默认；(0.6, 0.1) 大预算
- LIBERO: (b_a, b_s) = (0.5, 0.1) / (0.6, 0.1)（因 wrist view 占帧一半，需更多注意力覆盖）
- 随机化场景略增 b_s 到 0.2/0.3

## 实验与结果

### 实验平台
- **模型**：Fast-WAM（Wan2.2-5B 视频 DiT + 1B 动作专家 DiT），bfloat16，单卡 RTX PRO 6000 Blackwell
- **动作去噪步数**：默认 N=2（Appendix D 显示 N=2 在 RoboTwin 2.0 上距 N=10 仅差 1.3 点，且在 LIBERO 上几乎持平）

### RoboTwin 2.0（50 双腕任务，每任务 100 episodes）
| 方法 | FLOPs 节省 | 成功率下降 |
|------|-----------|-----------|
| FastV | −41.3% | **−27.4 pp** |
| ToMe | −41.0% | −12.2 pp |
| VLA-Cache | −40.8% | −4.0 pp |
| **WAM-Cache 默认** | **−41.6%** | **−1.8 pp** |
| WAM-Cache 大预算 | −28.2% | −0.1 pp |

- 随机化场景：WAM-Cache 节省 **39.3% FLOPs**，仅降 **1.3 pp**
- CUDA 延迟加速：1.23× / 1.47× / 1.67×（S_v=120 / 480 / 1080）

### LIBERO（4 套件各 10 任务，每任务 50 episodes）
| 方法 | Avg. SR | FLOPs 节省 |
|------|---------|-----------|
| VLA-Cache | 94.85% | −31.5% |
| **WAM-Cache 默认** | **95.85%** | **−32.3%** |
| WAM-Cache 大预算 | 96.50%（持平 dense） | −26.1% |

- 损失主要集中在 Long 套件（多阶段任务中 inherited attention 有滞后）

### 真机实验（AIRBOT Play 机械臂，20 trials/task）
| 任务 | Dense SR | WAM-Cache SR | FLOPs 节省 |
|------|---------|-------------|-----------|
| Carrot in Bowl | 90.0% | 90.0% | −40.2% |
| Stack Cubes | 80.0% | 75.0% | （avg 82.5% vs 85.0%） |
| **Avg.** | **85.0%** | **82.5%** | **−40.2%** |

### Ablation（表 3，clean RoboTwin 2.0，相同内容预算 0.5，G=2）
- Random：26.9% — 证明选择策略远比数量重要
- Pixel difference：60.4% — VLA-Cache 的 change 信号
- Oracle K/V drift：65.3% — **即使 oracle 也无法接近 dense**，印证"漂移≠策略敏感性"
- Surprise only：60.6%
- Action attention only：81.5%
- Attention ∪ Surprise（本文）：88.4%，且 ρ̄=0.584 最低（最节省）

## 相关工作脉络

1. **World Action Models**：Fast-WAM (Yuan et al., 2026) 证明可去掉未来 video imagination，使 prefill 成为主导成本；WAM-Cache 是首个针对此 prefill 做 sparse 加速的方法。
2. **Token 剪枝/合并**：FastV (Chen et al., 2024)、ToMe (Bolya et al., 2022) 在单帧内减 token，但不跨 chunk 复用；效果差（FastV −27.4pp，ToMe −12.2pp）。
3. **跨帧 Token 复用**：Eventful Transformer (Dutson et al., 2023) 基于 change 选择静态 token；VLA-Cache (Xu et al., 2025) 结合 change+language attention，但**其 attention 来自语言 token 而非动作专家**，且无 age bound——在 WAM 适配后仍落后本文 4–12pp。
4. **免训练 KV 缓存**：DeepCache (Ma et al., 2023)、X-Cache (Zeng et al., 2026) 等缓存 denoiser 中间表征，但 WAM 视频 prefill 是单次 pass 无 denoising trajectory，不适用。
5. **LLM KV Cache**：标准 LLM 中 prompt 不变故 reuse 精确；WAM 每 chunk 有新观测，核心挑战是"哪些 token 可缓存"——本文解决了这一差异化挑战。

## 局限性与未来方向

1. **小 token 数下速度提升低于 FLOPs 节省**：S_v=120 时仅 1.23×（理论上界 1/ρ̄≈1.71×），因 weight loading 和 CUDA graph replay 固定开销占比高；作者指出随视觉上下文扩大（更高分辨率、多相机、长历史），gap 会收窄。
2. **Inherited attention 滞后**：Long 套件中多阶段任务切换时，上一 chunk 的 attention 可能指向已不再关键的区域，导致短期性能损失。
3. **Age bound 限制的是 stale 时长而非 error magnitude**：G=2 强制刷新保证不会无限累积，但每个 token 在 G 步内的误差仍可能较大（只是被强制重置）。
4. **仅适用于 Fast-WAM 架构**：依赖视频 DiT 与动作专家解耦的结构；若未来 WAM 回归联合主干，适配方式需重新设计。
5. **未探索与蒸馏/量化等方法联合**：作者声明 WAM-Cache 正交于 diffusion step reduction、模型蒸馏、denoiser caching 等，但未做联合实验。

## 研究启发与可借鉴点

1. **"策略注意力驱动"刷新范式可迁移至其他 VLA/WAM 加速场景**：本文的核心发现——change-based 规则 plateau 远低于 dense，而 policy attention 才是关键信号——对任何"观测动态变化 + 策略条件生成"的系统均有参考价值。
2. **Age bound 是打破自 referential 误差累积的通用技巧**：当选择信号均基于 rank 且存在 feedback loop 时（当前选择依赖于前序状态），强制定期刷新是普适解决方案；可借鉴到 LLM KV cache eviction、流式推理等领域。
3. **Two-signal union 的 disjoint 性质值得利用**：surprise 和 attention 覆盖 largely disjoint token 子集，联合仅需简单 union 即可实现收益叠加——提示我们在设计多信号融合时，应优先选择正交信号而非竞争信号。
4. **成本分析的线性缩放结论（C(ρ)=ρ·C_dense）简洁且可复用于其他 sparse prefill 方法评估**。
5. **实验设计亮点**：用 oracle K/V drift 作为 upper bound 而非仅比 baseline，清晰揭示了"表征漂移≠策略敏感性"这一反直觉发现。

## 关键术语表

- **World Action Model (WAM)**：将预训练视频 DiT 与动作专家 DiT 配对，通过视频生成模型学到的物理动力学实现机器人操控的架构。
- **KV Cache（Key–Value 缓存）**：在 Transformer 中缓存 self-attention 的 K、V 投影，避免重复计算；本文将其跨 chunk 复用。
- **Latent Surprise（潜在惊喜）**：衡量 token 当前 patch embedding 与参考 embedding 的归一化距离，作为"视觉漂移"的代理信号。
- **Age Bound（年龄上限）**：强制任何 token 最多被连续复用 G 个 chunk 后必须刷新，防止 stale 误差无限累积。
- **Sparse Prefill（稀疏 prefill）**：视频 DiT 仅在刷新集 R_t 上做 forward pass，其余 token 直接读取缓存 KV，是本框架的核心操作。
- **Cross-Attention Mass（交叉注意力质量）**：动作专家对视觉 token 的 head-averaged attention 平均值，反映策略对该 token 的依赖程度。
- **Fast-WAM**：本文评测目标架构，去掉未来 video imagination，仅靠单次 prefill 编码观测，使 prefill 成为 dominant cost。
- **Staleness-Bounded Reuse**：本文核心保证——任何 cached KV pair 的 stale 时长被严格限制在 G+1 个 chunk 以内。

## 可复现要素

- **数据集/基准**：RoboTwin 2.0（公开）、LIBERO（公开）、真机 AIRBOT Play（自建，teleop 100 demos/task）；论文未提供新数据集。
- **代码**：项目页 https://dingkai0302.github.io/wam-cache/，论文未明确声明开源仓库链接（需查看项目页）；**模型权重使用 Fast-WAM 官方 checkpoint**（公开）。
- **关键超参**：G=2（所有实验不变）；(b_a, b_s) 在表 1/表 2 中明确列出；N=2（动作去噪步数）。
- **硬件**：单卡 NVIDIA RTX PRO 6000 Blackwell，bfloat16。
- **环境细节**：Appendix C/E 提供相机配置（384×320 / 224×448）、帧率（20Hz）、chunk 长度（RoboTwin 24 步动作，LIBERO 10 步）。
