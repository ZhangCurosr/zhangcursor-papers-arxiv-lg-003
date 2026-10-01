---
title: "QUANTFORGE-DISCOVERING-RESIDUAL-DECOMPOSI-TIONS-FOR-MXFP4-PO"
source: https://arxiv.org/pdf/2609.34680v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:55:48"
field: "模型量化与压缩"
keywords: ["post-training quantization", "MXFP4", "W4A4", "algorithm discovery", "LLM-driven search", "residual factorization", "coordinate transform"]
innovations: ["提出渐进式残差因子化框架，通过受控实验区分竞争解释并将结论编译为可验证的代码变更", "发现 HiRes 严格 MXFP4 W4A4 量化器，三级级联几何塑形-离散实现-结构恢复，七模型 Robust Fit 0.09300 创 SOTA", "在匹配 240 调用预算下，QuantForge 发现算法的达标率（6/8）显著优于纯分数/文本/反思基线"]
benchmarks: ["WikiText-2", "C4 perplexity", "ARC-Challenge", "ARC-Easy", "HellaSwag", "PIQA", "WinoGrande"]
---

# 论文速读：QUANTFORGE-DISCOVERING-RESIDUAL-DECOMPOSITIONS-FOR-MXFP4-POST-TRAINING-QUANTIZATION

## 一句话总结
本文提出 **QuantForge**，一种基于 LLM 驱动的程序演化系统，通过"渐进式残差因子化"自动发现 MXFP4 W4A4 后训练量化（PTQ）算法；由此发现的量化器 **HiRes** 通过几何坐标塑形、离散实现和结构恢复三个阶段级联操作，在七模型鲁棒性指标上刷新 SOTA（Robust Fit 0.09300），并在量化算法发现效率上显著优于仅依赖评分的基线。

## 研究问题与动机
1. **MXFP4 W4A4 的精度与泛化矛盾**：严格 MXFP4 格式下，权重/激活共享指数块导致小值精度不足；坐标变换会影响块编码误差，进而改变网络中的残差传播，手动设计难以协调各组件间的依赖关系。
2. **LLM 程序演化的信号缺陷**：已有 LLM 驱动算法发现方法（如 FunSearch、AlphaEvolve）仅凭性能分数选择候选程序，无法区分"为何有效"——例如坐标变换提升性能可能是因为分散异常值，也可能是因为改善了共享指数选择，这两者暗示不同的后续改进方向。
3. **组件间的序贯依赖**：改进某一 PTQ 组件会改变后续组件需要校正的误差，因此算法分解方案在设计完成前无法预先获知，需要一种能记录竞争解释、通过控制实验区分它们并将结论编译为代码变更的系统。
4. **跨模型泛化挑战**：现有 PTQ 方法往往针对特定模型/架构调参，缺乏在不同架构（如 Llama、Qwen、Mistral、OLMo）间可迁移的严格 MXFP4 W4A4 量化器。

## 核心贡献（创新点）
1. **渐进式残差因子化框架**：首次将受控实验结论编译为后续程序的可执行代码要求，而非仅靠分数筛选；与 FunSearch/AlphaEvolve 的本质区别在于引入了"竞争解释—控制实验—代码编译"的闭环，使算法分解过程可追踪、可验证。
2. **QuantForge 搜索系统**：维护可执行程序前沿 $\mathcal{P}_t$ 和未解残差前沿 $\mathcal{R}_t$，通过最小成本鉴别控制选择（Eq. 2）区分竞争解释，并用执行探针验证后继代码正确实现了结论；与 TextMem/ReflectMem 的本质区别在于要求实验结论必须转化为可检查的代码变更，而不仅是文本摘要或自由反思。
3. **HiRes 严格 MXFP4 量化器**：三级级联结构——几何塑形→离散实现→结构恢复，每级均作用于前级量化后产生的残差，形成确定性量化流程；与 WUSH、BRQ、TORQ 等现有方法的本质区别在于残差驱动的阶段性适配而非单一变换或静态 GPTQ 重构。
4. **严格的匹配预算发现实验**：在相同 240 次评估调用预算下，QuantForge 在 8 次运行中 6 次达到目标（PPL ≤ 8.00），而 Score-only/TextMem/ReflectMem 分别为 1/3/3 次，且 QuantForge 仅用 141 次新程序评估；这是首次证明"受控证据+代码编译"比单纯累积历史分数更有效。

## 方法详解

### 4.1 渐进式残差因子化（Progressive Residual Factorization）
- 维护两个前沿：可执行程序集合 $\mathcal{P}_t$ 和未解设计问题集合 $\mathcal{R}_t$，状态 $S_t = (\mathcal{P}_t, \mathcal{R}_t)$。
- **算法残差**：当程序获得性能增益时，可能存在多个竞争解释（如"提升来自权重身份 vs. 来自容量增加"），每个解释暗示不同的后继改进方向，此即"残差"。
- 受控实验排除竞争解释，将实验结论编译为后继程序的要求：$v_t = \mathsf{Resolve}(\rho_t, \mathcal{U}_t^*)$，$P_{t+1} = \mathsf{Compile}(P_t, v_t)$。
- 每次编译后用相同协议重新测量，暴露新的残差，形成迭代循环。

### 4.2 控制选择（最小成本鉴别）
选择最少数量的控制条件 $V \subseteq U$，使所有竞争解释对至少一个条件产生不同预测：
$$\mathcal{U}^* = \arg\min_{V \subseteq U} \sum_{c \in V} \mathrm{cost}(c) \quad \text{s.t.} \quad \forall e \neq e',\ \exists c \in \{\mathrm{primary}\} \cup V : P_e(c) \neq P_{e'}(c)$$

### 4.3 HiRes 三级量化结构

**Stage I：几何塑形（Geometric Shaping）**
- 参考坐标 $P_{\mathrm{ref}}$ 平衡激活与权重的二阶几何：
$$A = X^\top X/n + \lambda_A I,\quad B = W^\top W/m + \lambda_B I,\quad M_0 = \tfrac{1}{2}[\mathrm{DN}(A) + \mathrm{DN}(B^{-1})],\quad P_{\mathrm{ref}} = \mathrm{Cap}_8(M_0^{-1/2})$$
- 编码误差修正矩阵 $K$：通过锚定量化识别参考坐标的残留误差 $E_X, E_W$，计算误差-信号矩 $K_X = E_X^\top Y/n$、$K_W = V^\top E_W/m$，经软阈值和范数投影得到有界 $K$（$\|K\|_2 \leq 1/8$），更新 $G = |\det G_0|^{-1/q} G_0$。
- 最终变换 $T = R_H G P_{\mathrm{ref}}$（$R_H$ 为归一化 Hadamard），加权-激活平衡对角缩放 $D = \mathrm{Diag}(2^u)$，使得 $\widetilde{X} = XT^\top$、$\widetilde{W} = WT^{-1}$，保持 $\widetilde{X}\widetilde{W}^\top = XW^\top$。

**Stage II：离散实现（Discrete Realization）**
- 在固定 $DT$ 坐标下重新运行校准，以变换后的预-A4 激活计算 GPTQ 曲率 $H$，生成合法 Block32 W4 权重（E8M0 指数 + E2M1 码）。
- 激活码动态精化：对激活块 $x$，令 $q_0$ 为最近合法向量，$d_0 = q_0 - x$，单码变更的局部二次模型：
$$\Delta(i,c) = 2\delta_{ic}[G_Q d_0 - K_Q x]_i + \delta_{ic}^2[G_Q]_{ii},\quad \delta_{ic} = c - [q_0]_i$$
- 仅接受使 $\Delta < 0$ 的最佳负变更，每 Block32 至多改 1 个码。

**Stage III：结构恢复（Structural Recovery）**
- **注意力恢复**：对每 KV 头拟合仿射校正 $\bar{v}_t = av_t + b$（共享标量 $a,b$），最小化 $\sum_t \|av_t + b - v_t^\star\|_2^2$。
- **MLP 恢复**：在注意力校正后重新采集 gate $g$ 和 up $u$，拟合门控乘积校正：
$$\bar{m} = \mathrm{SiLU}((1+\beta)g + \alpha\sigma_g) \odot (1+\eta)u$$
- 通过加权岭回归估计 $(\alpha, \beta, \eta)$，并对 32 个校准分区的系数进行中值收缩与 $\ell_1$ 投影。

**整体流程（Algorithm 1）**：
$$R_k = \mathsf{Measure}_k(M^{(k)}),\quad M^{(k+1)} = \mathsf{Apply}_k(M^{(k)}, \mathsf{Solve}_k(R_k))$$
每级使用前一阶段生成的模型状态作为输入。

## 实验与结果

**评估设置**：
- 7 个模型：Llama-3.2-3B、Qwen3-4B/8B/32B、Mistral-7B-v0.3、Llama-3-8B、OLMo-2-13B
- 7 个任务：WikiText-2、C4 perplexity、ARC-Challenge/Easy、HellaSwag、PIQA、WinoGrande
- 校准：128 × 2048 C4-train 序列，seed 0；严格 MXFP4 W4A4（E2M1 值 + E8M0 共享指数 + Block32）

**主要结果（Table 1）**：
| 方法 | 七模型 Robust Fit | 七模型 Mean Fit-7 |
|---|---|---|
| HiRes | **0.09300** | **0.06726** |
| WUSH | 0.09910 | 0.07345 |
| BRQ | 0.11152 | 0.08440 |
| MR-GPTQ | 0.11447 | 0.08951 |
| RTN | 0.22151 | 0.18365 |

- HiRes 在 Qwen3-32B 上 Fit-7 = 0.0274，较 WUSH（0.0365）降低 **25%**。
- 在 OLMo-2-13B（新架构）上 Fit-7 = 0.0773，领先 BRQ（0.0811）和 WUSH（0.0823）。
- 三级消融（Table 2）：几何塑形带来最大初始增益（Qwen3-8B Fit-3 从 0.202 降至 0.078），离散实现降至 0.065，结构恢复降至 0.033。

**QuantForge 发现效率（Table 30 / Section 6.2）**：
- 240 次调用预算下，QuantForge 达到中位 PPL **7.80**，而 Score-only=10.03、TextMem=9.20、ReflectMem=8.09。
- PPL ≤ 8.00 达成率：**6/8**（QuantForge）vs 3/8（ReflectMem）、3/8（TextMem）、1/8（Score-only）。
- 仅评估 141 个新程序（其余 99 次用于控制与合规探针），而基线各评估 240 个。

**残差编译价值（Table 37）**：移除强制实现检查后，中位 PPL 从 7.80 升至 8.06，验证闭环率从 10.4/11.3 下降至 3.9/3.9。

**控制选择消融（Table 41）**：最小鉴别策略最优（11.3 次显式裁决 vs 随机 4.6 次，每次裁决成本 6.1 vs 15.0 调用）。

## 相关工作脉络
1. **GPTQ（Frantar et al., 2023）**：二阶权重重构补偿舍入误差；HiRes 在其基础上引入坐标变换和残差级联，不再仅依赖单一 GPTQ 步。
2. **QuaRot/SpinQuant/DuQuant（坐标变换类）**：通过旋转改变量化坐标；HiRes 的几何塑形结合了编码误差修正 $K$ 和加权-激活平衡 $D$，并后接离散实现，而非仅输出坐标。
3. **WUSH（Chen et al., 2026）**：自适应变换结合权/激统计；本文在同等变换接口下对比显示 HiRes 几何更强（Qwen3-8B Fit-7: 0.0327 vs 0.0498），且 HiRes 额外包含残差恢复。
4. **MR-GPTQ/BRQ/TORQ/BATQuant/FOCUS（微缩放类）**：处理 Block 结构与量化的交互；HiRes 在同契约下全面领先（Robust Fit 0.093 vs 次优 0.099），且无需学习/优化，为确定性算法。
5. **FunSearch/AlphaEvolve（程序演化类）**：仅用性能分数驱动搜索；QuantForge 引入竞争解释和受控实验区分机制，并验证代码响应，使发现效率显著提升。
6. **OPTScientist（Li et al., 2026）**：支持类型化优化器程序；本文聚焦 PTQ 算法而非优化器，且强调"从实验结论到代码变更的编译"这一独特环节。

## 局限性与未来方向
1. **推理部署未验证**：论文明确承认 realized latency、throughput 和 memory 未在原生 MXFP4 硬件内核上测试，仅在 PyTorch 模拟量化下评估数值质量。
2. **校准数据依赖**：当前使用固定 128×2048 C4-train 序列，未探索更少校准样本或多数据集组合的效果。
3. **搜索预算仍较昂贵**：240 次 evaluator 调用需花费约 60–287 分钟量化时间（Table 29），对于大模型而言成本较高。
4. **结构化恢复容量有限**：每头仅 2 个仿射系数、每层仅 3 个 MLP 系数，复杂误差模式可能无法充分拟合。
5. **未来方向**：部署到原生 MXFP4 内核的延迟/吞吐/内存研究；扩展到更多 bit 配置（如 FP2）；探索更强恢复结构或自适应层级深度。

## 研究启发与可借鉴点
1. **残差驱动的分阶段设计范式**：将 PTQ 分解为"测量残差→拟合校正→重测"的迭代循环，而非一次性设计全部组件；此范式可迁移至模型压缩、精度调整等其他领域。
2. **受控实验引导程序演化的设计**：在 LLM 驱动算法发现中，除分数外记录"竞争解释+鉴别控制+代码编译"三重证据，显著提升发现效率（6/8 vs 1/8 达标率）；这一框架可直接应用于其他可执行算法搜索任务。
3. **编码误差反馈坐标设计**：HiRes 的局部交互矩阵 $K$ 利用实际 MXFP4 编码误差修正参考坐标，而非仅依赖二阶统计；这种"用真实量化误差校准设计"的思路可用于其他低比特格式。
4. **残差测量的序贯依赖性**：证明 MLP 恢复必须在注意力恢复之后重测（Fit-3 改善 0.006–0.013），提示多级处理方法需考虑阶段间的状态依赖，不可随意调换顺序。
5. **最小成本鉴别控制选择**： Eq. 2 的约束优化形式（最小化总代价使所有解释对可区分）为实验设计提供了通用原则，可推广至超参数搜索、神经架构搜索等场景。

## 关键术语表
**MXFP4**：Open Compute Project 标准化微缩放浮点格式，32 个 E2M1 值共享一个 E8M0 2 的幂次指数，适合短块的自适应量化。
**PTQ（Post-Training Quantization）**：后训练量化，在预训练模型完成后直接量化权重/激活，无需完整重新训练。
**Progressive Residual Factorization**：渐进式残差因子化，QuantForge 的核心过程——通过受控实验解决设计残差，将结论编译为代码要求，重测后暴露新残差，迭代直至算法收敛。
**Algorithmic Residual**：算法残差，附着于可执行程序的设计疑问，存在多个竞争解释且各自暗示不同的后继改进方向。
**W4A4**：权重和激活均采用 4-bit 量化的设定。
**Robust Fit**：跨模型鲁棒性度量，对每模型取 7 任务平均/最坏损伤的均值与最大值的平均，再跨 7 模型取同样聚合，值越低越好。
**One-code refinement**：每 Block32 至多更改 1 个 E2M1 激活码的动态精化规则，利用已安装权重的失配信息选择最优变更。
**Structural Recovery**：结构恢复，在 W4/A4 算子安装后拟合低维注意力（仿射系数）和 MLP（门控乘积系数）残差校正。

## 可复现要素
- **数据集**：C4-train（128×2048 序列，seed 0）；评估数据集 WikiText-2、C4、ARC-Challenge/Easy、HellaSwag、PIQA、WinoGrande 均为公开数据集。**论文未提及额外私有数据**。
- **代码开源**：论文未提及代码开源声明，附录中提供了完整的算法细节（Appendix B）和复现实验协议（Appendix C/D）。
- **模型**：Llama-3.2-3B、Qwen3-4B/8B/32B、Mistral-7B-v0.3、Llama-3-8B、OLMo-2-13B 均使用公开 BF16 checkpoint。
- **关键超参**：Block 大小 32；GPTQ damping 0.01；ridge 系数 $\lambda_A = 0.01\,\mathrm{tr}(X^\top X/n)/q + 2^{-30}$；谱 cap $\kappa=8$；$K$ 范数界 $1/8$；MLP 恢复系数 $\ell_1$ 界 $1/8$；校准序列 128×2048 tokens。
