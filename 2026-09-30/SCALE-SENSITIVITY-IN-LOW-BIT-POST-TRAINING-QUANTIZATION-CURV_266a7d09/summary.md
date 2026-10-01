---
title: "SCALE-SENSITIVITY-IN-LOW-BIT-POST-TRAINING-QUANTIZATION-CURV"
source: https://arxiv.org/pdf/2609.37416v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 17:16:36"
---

# 论文速读：SCALE-SENSITIVITY-IN-LOW-BIT-POST-TRAINING-QUANTIZATION-CURV

## 一句话总结
本文从理论角度系统刻画了低比特后训练量化（PTQ）中缩放尺度选择的敏感性，提出尺度不变曲率 $\kappa$ 作为统一度量，揭示低比特（尤其 W2/W3）下目标函数在最优尺度附近显著更尖锐的几何规律，并为多种 scale-selection 规则提供可比框架。

## 研究问题与动机
1. **尺度选择缺乏统一理论基准**：现有 PTQ 方法多依赖启发式搜索或经验规则确定量化尺度，未明确刻画目标函数在最优尺度附近的二阶几何特性。
2. **误差来源耦合不清**：量化误差中舍入误差与裁剪误差相互交织，导致对低精度下主导误差项的判断依赖直觉而非严格分解。
3. **不同规则难以横向比较**：RTNH-cross、ResComp 等策略各自独立优化，缺乏在同一扰动度量下的公平对比依据。
4. **高维有效秩缺乏下界保证**：实际网络中 Hessian 的数值性质与理论假设之间存在 gap，需要概率性低秩近似保证以支撑有限宽度收敛分析。

## 核心贡献（创新点）
1. **提出尺度不变曲率 $\kappa$**：定义 $\kappa = (s^*_\sigma)^2 \mathcal{E}''_\sigma(s^*_\sigma)$ 量化相对扰动敏感性；与既往工作仅优化绝对误差极小值不同，本文聚焦相对扰动下的二阶几何响应。
2. **严格分解量化误差**（Corollary 2）：将 $\mathcal{E}^\infty$ 拆分为与比特精度无关的舍入误差和由输入越界引起的裁剪误差；本质区别在于首次证明裁剪参数 $M_B$ 仅通过裁剪项进入目标函数，切断其与舍入项的耦合。
3. **建立有效秩高概率下界**（Theorem 5）：在假设 (H1)/(H3)/(H4) 下证明 $r_{eff} \geq C\log^8 d$；区别于纯经验校准，该结果为 Hessian 对角近似与激活排序提供理论收敛保障。
4. **构建相对扰动统一比较框架**：以 $\varepsilon$ 为自变量推导 $\Delta \mathcal{E} = \frac{\kappa}{2}\varepsilon^2 + o(\varepsilon^2)$，使 Table 5 中不同 scale-selection rules 可在同一曲率度量下横向评估。

## 方法详解
1. **误差分解与曲率建模**：量化目标 $\mathcal{E}_\sigma(s)$ 在最优尺度 $s^*_\sigma$ 附近作二阶泰勒展开，引入尺度不变曲率 $\kappa$ 捕捉相对扰动 $\varepsilon$ 的误差放大系数；$\mathcal{E}^\infty$ 明确分离舍入分量（仅依赖 $B$）与裁剪分量（依赖输入分布与 $M_B$）。
2. **有效秩与高斯景观理论**：基于 $\varepsilon$-net 论证（引理 11）证明定理 5：以概率至少 $1-\bar{C}N_c e^{-cd}$ 有 $r_{eff} \geq C\log^8 d$；结合高斯过程逼近推导有限宽度网络中高斯景观的收敛性（Figures 11,12）。
3. **RTNH-cross 尺度搜索**：针对 $B \leq 3$ 设计，在校准集上最小化 Frobenius 范数 $\|WX_{fp}-\Pi_{s,M_B}(W)X_q\|_F^2$ 求解 $s$；适用于低比特下裁剪误差主导的 regime。
4. **Hessian 正则化与激活排序**：正则化项 $\lambda = 0.01 \cdot \text{mean}(\text{diag } H)$ 抑制病态特征；列处理采用 lazy batch=128 并按递减 $H_{ii}$ 排序（activation ordering），优先校正高敏感度权重，提升数值稳定性。
5. **与主流方法的接口设计**：兼容 ResComp（$\alpha_{GPTAQ}=\alpha_{ResComp}=0.25$，W2 启用稳定性模式）与 QRoNoS（reset propagation，compensation block=$\min(128,g)$），支持 group size $g \in \{512,256,128,64\}$ 灵活切换。

## 实验与结果
- **模型与数据**：OPT-125M、Llama-3.2-1B/3B-Instruct、Llama-3.1-8B-Instruct、Qwen3-8B；fp16 加载，context=2048 tokens；量化覆盖所有 transformer linear 层（embedding/LM head 保持 fp16）。
- **实验家族**：B.2 全局尺度敏感性与 ablation（Fig 1,2,5-9）；B.3 尺度选择与 zero-shot 准确率（Table 1,2,6-8）；B.4 预处理与有效秩（Fig 10, Table 9）；B.5 高斯景观与有限宽度收敛（Fig 11,12）；B.6 局部重建敏感性（Fig 3,13-15）。
- **核心结论**：$\kappa$ 随 $B$ 单调递减，低比特（W2/W3）下目标函数在最优尺度附近显著更尖锐，尺度误配极易导致性能骤降；相对扰动框架成功统一比较 Table 5 中各类规则。
- **最强表现**：Llama-3.1-8B 上 W3 量化达到 3.003 bits/weight，结合 RTNH-cross 与激活
