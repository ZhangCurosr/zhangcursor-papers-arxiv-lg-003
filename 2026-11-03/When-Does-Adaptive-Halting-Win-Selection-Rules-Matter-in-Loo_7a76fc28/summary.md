---
title: "When-Does-Adaptive-Halting-Win-Selection-Rules-Matter-in-Loo"
source: https://arxiv.org/pdf/2611.02333v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:43:41"
---

# 论文速读：When-Does-Adaptive-Halting-Win-Selection-Rules-Matter-in-Loo

## 一句话总结
本文通过预注册研究与密封测试集评估证明：在冻结的 Looped LM（Ouro-1.4B）上，自适应早停方法的优劣**高度依赖操作点选择规则**；在严格的 Bootstrap 风险筛选规则下，训练好的稳定性探针（π1-x）比模型原生门控节省更多循环，但在宽松的点规则下原生门控更优，且训练探针在分布外基准（Social IQa）上会严重违反精度预算。

## 研究问题与动机
- Looped / recurrent-depth LM 可通过自适应循环次数节省推理算力，Ouro 等模型自带学习出口门控（exit gate），近期工作表明简单的后验轨迹读取（post-hoc readouts）常能匹敌或超越学习门控。
- 现有比较通常将每种方法缩减为验证集选出的单一运行点，并将选择规则视为协议细节，忽略其对结论的决定性影响。
- 核心问题：选择规则是否直接决定哪种早停方法“获胜”？训练探针是否真正优于强无训练基线？开发集选择的策略能否在分布外保持精度预算？

## 核心贡献（创新点）
1. **揭示选择规则对比较结论的决定性作用**：在冻结模型与密封测试集上证明，同一组早停信号在不同选择规则下会选出不同的操作点，导致偏好顺序完全反转。
2. **证明训练探针的边际收益有限**：在严格风险筛选下，训练探针（π1-x）的节省优势主要来自规则偏好而非信号本质优越；与无训练 KL 散度读取（r-kl）相比未展现统计显著优势。
3. **暴露启发式筛选规则的分布外脆弱性**：在任务格式不同的 0-shot OOD 基准（Social IQa）上，开发集选出的训练探针严重越界（损失 3.26 pp），而原生门控保持稳定，提示需配套 OOD 检查。
4. **提出早停评估的六项规范建议**：强调应将“信号+选择程序”作为比较单元，强制报告测试集上界、引入形式化风险控制、设置强无训练基线、分离模拟节省与实测延迟，并公开预注册协议与密封测试流程。

## 方法详解
- **基础设置**：使用冻结的 Ouro-1.4B（bf16，4 循环），每个问题在最后一个上下文 token 处做单次早停决策。精度损失定义为 `acc(loop 4) − acc(policy)`（单位 pp）。
- **候选信号族**：
  - `native-pb`：原生门控 CDF，$s_t = 1 - \prod_{j<t}(1-\lambda_j)$，在末位上下文 token 处取值。
  - `π1-x`：每循环独立训练的 L2 正则逻辑回归（$C=1.0$），输入 9 维特征（conf, ent, vent, cdf, n_options, log_n_tokens 及循环指示符），目标为稳定性标签 $S_t$（当前预测选项与之后所有循环一致），输出即为 $s_t = \hat{P}(S_t=1)$。
  - 无训练基线 `r-*`：包括 r-conf、r-margin、r-ent、r-kl（$s_t = -\mathrm{KL}(p_t \| p_{t-1})$）、r-dlogit、r-hl2、r-hcos，阈值与深度上限均按基准独立选取。
- **选择规则**（仅在开发集上操作，候选人 51×4 网格）：
  - **Point-only rule**：点精度损失 ≤ 1 pp 即合格，取最大宏观节省。
  - **Bootstrap-screened rule**：点损失 ≤ 1 pp **且** 95% 单侧 Bootstrap 上界（2000 次重采样）≤ 1 pp 才合格，平局按小点损失→偏好固定深度→大阈值→小深度 cap 打破。
- **评估与开销**：策略从存储的全深度轨迹中**回溯评估**（非截断执行）。净节省需扣除探针额外开销 $\rho$（$\rho_{\text{score}}$ 计分开销、$\rho_{\text{rest}}$ 特征提取与探针推理开销），以单次循环时间为基准换算，属于部署成本估算而非实测延迟。

## 实验与结果
- **数据集**：开发集 20,250 题（OpenBookQA/CSQA/HellaSwag/ARC-Challenge/ARC-Easy/MMLU）；密封测试集 8,517 题（主要：OpenBookQA 3k, CSQA 1k, HellaSwag 750；OOD：PIQA 1,837, Social IQa 1,930）。
- **主要结果（Bootstrap 规则，TEST）**：
  - π1-x 优于 native-pb：宏观节省 +0.053 [95% CI +0.049, +0.057]，达成预设 PASS
