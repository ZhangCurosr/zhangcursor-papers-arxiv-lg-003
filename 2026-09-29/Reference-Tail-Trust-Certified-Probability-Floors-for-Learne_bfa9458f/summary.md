---
title: "Reference-Tail-Trust-Certified-Probability-Floors-for-Learne"
source: https://arxiv.org/pdf/2609.34904v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 02:58:37"
field: "可信大语言模型推理"
keywords: ["运行时认证", "概率下界", "LLM安全", "浮点误差分析", "support functions", "测试时适应"]
innovations: ["首次实现 per-call 在线认证验证器，独立 float64 检查器与 float32 执行器分离", "支持函数对候选 duplicate/permute/add 完全不变（0% 位移）", "显式建模累积误差因子 gamma_m 并传播至原语，实现严格 per-call 保证"]
benchmarks: ["SoundnessBench"]
---

# 论文速读：Reference-Tail-Trust-Certified-Probability-Floors-for-Learne

> **注**：以下笔记基于提供的第 4/4 段要点笔记综合整理。第 1-3 段笔记内容为空，故"研究问题与动机"、"核心贡献"、"方法详解"前三节基于第 4 段内容反向推断并明确标注，建议核对原文后补充。

---

## 一句话总结
论文提出了一种**运行时认证的概率下界（Certified Probability Floors）**机制，以部署的 float32 incumbent 模型为参考，在 LLM/强化学习推理的每次调用中，通过 float64 验证器对内部 state 变更施加**per-call 严格保证**，并在检测到违规时触发 fallback。

---

## 研究问题与动机
> ⚠️ 推断自第 4 段，未获原文第 1-3 段直接支撑

1. **运行时风险不可控**：现有测试时适应（TTA）、安全增强等方法（如 GTrans、Matcha、Safe/graph TTA）缺乏**逐调用（per-call）保证**，无法在推理阶段检测违规。
2. **对抗脆弱性**：LLM 在对抗性扰动、方向腐蚀下可能产生不可预期的行为，现有认证方法（如 Differential Verification、Conformal Policy Control）多为离线审计，不覆盖在线调用。
3. **现有保证的局限性**：CPI/PPO 等训练时约束依赖模型改进假设（CPI 不等式/clip），而 weight interpolation 等离线方法不提供每步保证；Steering with abstention 仅提供统计保证且依赖参考预测。

---

## 核心贡献（创新点）
> ⚠️ 推断自第 4 段，未获原文第 1-3 段直接支撑

1. **首次实现 per-call 在线认证**：与 Differential Verification（离线/第二网络）和 Certified Upgrades（离线审计）不同，本文验证器在**每次推理调用**中在线运行，直接监控内部 states。
2. **支持函数不变性设计**：采用 support functions 而非 mean pooling 来聚合候选，使其对候选的 duplicate/permute/add 操作**完全不变**（0% 位移变化），提升鲁棒性。
3. **浮点误差可控的验证器架构**：验证器以 float64 定向舍入独立实现，显式建模执行阶段 state-update 残差与 incumbent float32 前向误差，并通过累积误差因子 $\gamma_m$ 传播至剩余原语，保证证书精度。
4. **可量化的 fallback 机制**：支持在检测到违规时切换至安全策略，在 10,000 个植入违规测试中实现 **100% 拒绝率**。

---

## 方法详解

### 1. 验证器（Checker）架构
- 以部署的 **float32 incumbent** 为参考，定理在实数算术下陈述。
- 证书由 **float64 检查器**以定向舍入计算，与执行器**独立实现**（防共谋漏洞）。

### 2. 误差界推导
- 两项容差扣除：
  - 执行阶段的 **state-update 残差**
  - incumbent 自身 **float32 前向误差**
- 累积误差因子：$$\gamma_m = \frac{m\epsilon_m}{1 - m\epsilon_m} \quad (m\epsilon_m < 1)$$
- 误差因子须乘对应绝对和界并传播至剩余原语（含 validated tanh enclosure）。

### 3. 支持函数（Support Functions）vs Mean Pooling
- 对候选进行 duplicate/permute/add 操作时：
  - **Support functions**：位移变化 = **0%**（完全不变）
  - **Mean pooling**：位移变化 = **0.4%**
- 当替换候选导致 hull 几何改变时，两者响应一致（21% 位移变化）。

### 4. Fallback 机制
- 检测到违规时切换至安全策略，保证 per-call 合规。

---

## 实验与结果

### 评测设置
- **SoundnessBench**：10,000 个植入违规，隐藏边际 $10^{-7}$–$10^{-3}$ nats。
- 1,000 份证书以 **50-digit 精度**复核。
- 压力测试：对抗性因子腐蚀、步骤腐蚀、舍入关闭。
- 总评估量：228,514 episode evaluations（210,370 有标签），38.7 device-hours。

### 关键结果

| 指标 | 数值 |
|------|------|
| 植入违规拒绝率 | **10,000 / 10,000（100%）** |
| 50-digit 重算误差 | float64 界比高精度参考最多高 $3.1\times10^{-10}$ |
| Exact endpoint cost（2,048 calls）| **0 violations**；certified/exact 中位数 = **1.61** |
| 因子被低估 30% 时检出率 | **214 / 214 违规调用被检出**（fallback） |
| 舍入关闭诊断 | 2,048 证书中 3 个低于精确成本 $\leq 10^{-6}$ |

### 运行时开销对比

| 方法 | RTT（ms） |
|------|-----------|
| **本文（bank prior + signed attention）** | **7.5**（单设备顺序2 rollout）；双设备 **1.18** |
| GTrans（test-time adaptation） | 91.0 |
| Matcha（test-time adaptation） | 58.0 |
| Retrained incumbent（v2, ≤2017） | 41.6 |
| LoRA-16 weight delta | 43.0 |
| one-pass corrector（2-hop head+rows） | 10.4 |

### 核心发现
- Checker 在三类测试（植入违规、高精度复核、exact endpoint cost）上均**通过**。
- 对抗方向测试中 bound 满足但**审计失败**（audit fails），表明 trim 内仍存在一定脆弱性。
- 图 9(d) 显示 certified/exact charge 比值随深度分布，中位数约 1.61，5–95% 区间可见。

---

## 相关工作脉络

| 方法 | 检查时机 | 变更对象 | 保证类型 | 本文差异 |
|------|---------|---------|---------|---------|
| Differential Verification (Paulsen 2020; Teuber 2025) | offline | 第二网络 | 差 bound | 本文在线 per-call，不依赖第二网络 |
| Certified upgrades (Zhang 2026; Balachandran 2026) | offline audit | — | 统计非回归 | 本文提供 per-call 严格保证 |
| Conformal policy control / LTT / CALM | calibration | policy/exit | 风险控制 | 本文直接监控内部 states |
| CPI / PPO | training | policy | CPI 模型改进；PPO clipped | 本文不依赖训练时假设 |
| Weight interpolation (Wortsman 2022) | offline | weights | surrogate none per call | 本文提供 per-call 保证 |
| Safe/graph TTA (Niu 2022; Press 2023; Bao 2025; Hsu 2026) | online | states/weights | 无 per-call 保证 | 本文首次实现 per-call |
| Steering with abstention (Hedström 2025) | online | activations | 统计保证，需参考预测 | 本文提供严格保证，不依赖参考预测 |

---

## 局限性与未来方向
> ⚠️ 推断自第 4 段，未获原文直接陈述

1. **trim 内脆弱性**：对抗方向测试中 bound 满足但审计失败，表明当前保证在特定对抗场景下存在缺口。
2. **舍入关闭边界**：2,048 证书中 3 个低于精确成本 $\leq 10^{-6}$，虽极小但提示 float64 验证仍存在极轻微保守性偏差。
3. **扩展性**：当前设计针对特定 incumbent，如何泛化至多模型/多任务场景未讨论。
4. **误差界紧致性**：$\gamma_m$ 随累加长度 $m$ 线性增长，可能对长序列推理产生过保守界。

---

## 研究启发与可借鉴点

1. **独立验证器设计**：将验证器与执行器分开实现（float64 vs float32），避免共谋漏洞——可迁移至任何需要运行时认证的系统中。
2. **误差因子传播框架**：显式建模 $\gamma_m = m\epsilon_m/(1-m\epsilon_m)$ 并传播至原语——可作为浮点错误分析的通用模板。
3. **支持函数不变性**：采用 support functions 替代 mean pooling 以获得对候选排列/重复的不变性——对鲁棒聚合设计有启发。
4. **植入违规基准**：SoundnessBench 的 $10^{-7}$–$10^{-3}$ 隐蔽边际设计——可用于评测其他认证方法的灵敏度。
5. **低成本在线认证**：7.5 ms RTT（单设备）与 1.18 ms（双设备）显著低于 GTrans（91 ms）和 Matcha（58 ms）——证明严格保证无需牺牲过多延迟。

---

## 关键术语表

- **Incumbent**：已部署的参考模型（float32），作为验证器的基准。
- **Certified Probability Floors**：对模型输出概率施加的严格下界保证，低于该值则触发 fallback。
- **Support Functions**：凸分析的支撑函数，用于聚合候选并保持对 duplicate/permute/add 操作的不变性。
- **Per-call Guarantee**：在每次推理调用中即时验证的保证，区别于离线审计。
- **Fallback**：检测到违规时切换至安全策略的备用行为。
- **SoundnessBench**：包含植入违规的测试基准，用于验证 checker 的敏感度。
- **Directed Rounding**：定向舍入（如向正/负无穷舍入），确保 float64 验证器的误差可控。
- **Audit Fail**：bound 数学上满足但逻辑验证失败的场景，暴露 trim 内脆弱性。

---

## 可复现要素

| 要素 | 状态 |
|------|------|
| 数据集（SoundnessBench） | 论文声明，是否公开未提及 |
| 代码 | 论文未提及 |
| 权重 | 论文未提及 |
| 关键超参（$\gamma_m$ 中的 $m$, $\epsilon_m$） | 论文未提及 |

---
