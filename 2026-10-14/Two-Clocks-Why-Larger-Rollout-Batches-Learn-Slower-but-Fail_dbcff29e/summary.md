---
title: "Two-Clocks-Why-Larger-Rollout-Batches-Learn-Slower-but-Fail"
source: https://arxiv.org/pdf/2610.18853v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:08:21"
---

# 论文速读：Two-Clocks-Why-Larger-Rollout-Batches-Learn-Slower-but-Fail

## 一句话总结
本文针对多步 GRPO 训练中“扩大 rollout batch 会拖慢进度但不会推迟崩溃”的矛盾现象，提出**双时钟（Two-Clocks）理论框架**：从局部二次损失推导进度时钟，结合漂移阈值假说推导失败时钟，并用跨 batch/LR/clip/模型规模的系统实验逐一验证，最终给出“先小后大”的简单调度策略。

## 研究问题与动机
- RLVR（如 GRPO/DAPO）为摊薄昂贵的策略重采样成本，工程上普遍采用大 rollout batch 配合平方根学习率缩放规则（$\eta \propto \sqrt{b}$）。
- 现有单步训练理论认为该规则可保持样本效率近似不变，但实际多步训练（每 rollout 8 步）中同时出现三个分离现象：大 batch 进度变慢、崩溃时间却按消耗样本数锁定、且 ratio clip 对大 batch 的稳定作用最强，缺乏统一解释。
- 经典临界 batch 理论（McCandlish et al., 2018）假设 LR 可随 batch 独立重调优，而平方根规则将 LR 与 batch 强耦合，二者作用机制不同，需重新刻画。
- 实践中 clip 常被视作限速瓶颈（Qi et al., 2026），但其真实角色是购买稳定性还是压制速度尚未厘清。

## 核心贡献（创新点）
1. **提出双时钟框架**：证明平方根规则仅保持每样本一阶漂移不变，而不保持进度；与 McCandlish 等独立调 LR 的临界 batch 理论不同，本文的临界 batch $B^*$ 随 LR 反比收缩且远低于梯度噪声尺度。
2. **建立漂移阈值假说并验证失败时钟**：推导沿缓变平均梯度方向的累积漂移达到阈值即崩溃，得出失败样本数与 batch 无关；与先前仅关注固定 batch 步长阈值或异步 GRPO 的崩溃研究不同，本文明确纳入 batch 维度并给出可检验的标度律。
3. **实证定位 clip 作用并导出 batch ramp 调度**：通过消融与 replay 证明 clip 移除大 batch 超 50% 的梯度范数，主要购买稳定性而非持续速度；基于双时钟推导出无需额外搜索的“小 batch 起步→ failure point 前切大 batch”策略，在逼近最优固定 batch 性能的同时提升吞吐。

## 方法详解
- **进度时钟（Progress Clock）**：在两个假设下建模单步更新：(A1) 优化器 batch 远小于梯度噪声尺度，使 $\nu \approx \sigma/\sqrt{b}$；(A2) 单步内损失局部二次近似。代入平方根规则 $\eta=\kappa\sqrt{b}$ 后，每样本漂移 $d=\kappa\|\bar{g}\|/\sigma$ 与 batch 无关，但每样本期望损失下降量 $r(b)=r_0-C_{\text{curv}}b$ 随 batch 线性衰减，曲率项 $C_{\text{curv}}=\kappa^2\bar{g}^\top H\bar{g}/(2\sigma^2)$。到达目标所需样本 $N_\tau(B_r)=N_\tau^{\min}(1+B_r/B^*)$，其中临界 batch $B^*=\mu\left(2\sigma\|\bar{g}\|^2/(\kappa\bar{g}^\top H\bar{g})-B_{\text{noise}}\right)$，随 $\kappa$ 增大而缩小。
- **失败时钟（Failure Clock）**：引入假设 (A3) 漂移阈值：当累积漂移 $\|\sum_j \mathbb{E}\Delta\theta_j\|$ 达到 $D$ 时训练崩溃。推导失败样本数 $N_{\text{fail}}=C\sqrt{b}/\eta$；沿平方根规则简化为 $N_{\text{fail}}=C/\kappa$，与 batch 无关。若崩溃由噪声随机游走驱动，则应有 $N_{\text{fail}}\propto b/\eta^2$，本文通过拟合 $\beta-\gamma/2\approx0$ 与 $\gamma\approx0.7$ 明确拒绝该替代机制。
- **Clip 作用机制**：多步训练中当前策略与生成策略的偏差使 token log-ratio 展宽。在匹配步数 $k$ 下，大 batch 的 ratio spread 约为小 batch 的 $\sqrt{B_r^L/B_r^S}$ 倍；clip 将超出边界的 token 梯度置零，replay 显示其移除了大 batch 超过 50% 的梯度范数，从而限制每个 rollout 的漂移增量，但不产生进度时钟的曲率项。

## 实验与结果
- **设置**：Qwen2.5-Math-1.5B，GRPO+DAPO 目标，DAPO-Math-17k 数据集（<1 epoch），rollout batch 512/1536/4096/8192（16×），$\mu=8$，AdamW，禁用 KL/熵/动态采样，每 8192 responses 评估 512 道 held-out 题 mean@4。另在 DeepScaleR、Qwen2.5-1.5B-Instruct、Qwen2.5-Math-7B 上复现实验。
- **进度变慢（P1）**：关闭 clip 或 $\mu=1$ 时大 batch 仍需多 20%–45% 样本。LR 提升 10 倍时样本成本指数 $\alpha$ 从 0.15 升至 0.44，隐含 $B^*$ 从 ~9k 降至 ~2k。该效应在所有额外 checkpoint/数据集/7B 模型上方向一致。
- **失败锁定样本（P2）**：无 clip 时 4096 batch 在第 56–80 步崩溃，512 batch 在第 648 步崩溃，但均在 30k–50k samples 处失败。21 组无 clip 实验拟合 $N_{\text{fail}}\propto b^\beta\eta^{-\gamma}$，沿平方根规则 $\beta-\gamma/2=-0.08\sim0.11$（含右删失），$\gamma=0.64\text{–}0.72$；似然比检验支持漂移律（$p\approx0.29$），拒绝随机游走律（$p<10^{-4}$）。7B 模型同样按消耗样本数崩溃（差 6× rollouts 仅差 1.3× samples）。
- **Clip 买稳定（P3）**：开启 clip 使小 batch 失败延迟 2 倍、大 batch 延迟 4 倍，进度差距依旧。加宽 clip 带来早期瞬态加速但随后性能衰减且大 batch 仍崩溃。
- **最强调度**：512 起步于 25k–74k samples 切换至 4096/8192 的 ramp 策略，性能逼近最佳固定 batch（1536），且 3/4 样本由大 batch 生成。

## 相关工作脉络
- **Batch-size invariance (Hilton et al., 2022; Li et al., 2026)**：证明单步或解耦 proximal policy 下批大小不变性；本文聚焦多步 regime，指出平方根规则仅保持漂移不变而不保持进度。
- **Critical batch (McCandlish et al., 2018)**：LR 每批独立调优时临界 batch 等于梯度噪声尺度且
