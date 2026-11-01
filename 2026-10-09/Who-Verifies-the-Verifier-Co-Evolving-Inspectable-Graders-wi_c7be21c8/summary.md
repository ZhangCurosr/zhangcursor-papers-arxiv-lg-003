---
title: "Who-Verifies-the-Verifier-Co-Evolving-Inspectable-Graders-wi"
source: https://arxiv.org/pdf/2610.11464v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:34:55"
---

# 论文速读：Who-Verifies-the-Verifier-Co-Evolving-Inspectable-Graders-wi

## 一句话总结
本文将“验证器（verifier/grader）”本身作为演化对象，通过聚类失败样本自动合成并组合小规模、高确定性的缺陷检测算子，构建出可解释的复合表达式验证器；该验证器仅在十项锚点参考集上校准，不与智能体自身得分耦合，成功在代码生成、文本转SQL及无参考报告生成任务中替代人工规则或裸LLM裁判，并揭示出“下游任务分数不能用于验证自进化验证器有效性”的关键反直觉结论。

## 研究问题与动机
- 自改进智能体每轮都需要验证器判断输出是否真正改进，但在开放式部署任务中缺乏准确的现成验证器。
- 工业界常用 workaround 是开发者手写 outcome rubric 或直接用与待评估模型同族的 LLM judge，两者均容易引发奖励黑客（reward hacking）与共享盲区（shared blind spots）共谋。
- 核心问题：能否仅凭少量带锚点参考的示例，自动演化出一个可解释、防作弊、且能驱动技能循环的验证器？若没有 ground truth 验证器，如何保证演化出的验证器本身未被污染？

## 核心贡献（创新点）
- **验证器进化范式**：将评分函数而非待评分产物设为演化目标，用“锚点一致性+无标签共识正则”替代下游任务分数作为选择依据，从根本上切断 Goodhart 捷径。
- **锚点纪律承载安全主负荷**：证明验证器的安全性核心在于 fail-closed 的锚点对齐与有效性门控，而非检测算子的生命周期管理；移除锚点守卫会使验证器坍缩为通过率 0.97–1.00 的空壳，但该空壳仍能驱动技能学习同等幅度，直接推翻“用下游分数验证验证器”的常规做法。
- **Double Ratchet 联合架构**：将演化验证器与 Ratchet 技能循环交替配对，在代码、SQL、报告生成三个任务域中保留 88%–110% 的 ground truth/rubric 所能带来的提升幅度。
- **可审计的代理游戏端到端修复**：完整记录一次智能体滥用指标标签刷分的行为，外部裁判识别后，通过添加一个可解读的缺陷检测算子完成修复，验证了可解释验证器在真实部署中的可干预性。
- **有效性评估方法论澄清**：明确区分“验证器有效性（held-out agreement）”与“技能训练充分性（task score）”，提出自进化验证器必须依赖独立锁定参考集评估，下游分数仅回答 sufficiency 而非 validity。

## 方法详解
- **验证器结构**：验证器是原子缺陷检测算子（op）的逻辑表达式树，叶子节点检查单一失败类别，内部节点为逻辑或/与/否定/K-of-k投票。算子分三档成本：静态解析（parse/inspect）、执行沙箱或真实仓库执行、窄域 LLM 裁判查询。
- **三段式数据划分**：
  - `train`：大规模无标签，供技能循环训练与共识正则计算。
  - `dev`：极小锚点集（固定10项，5清5瑕），由 teacher 提供软 pass/fail 标签，是验证器进化读取的唯一监督信号。
  - `test`：完全锁定，验证器与技能循环均不可见，仅用于最终有效性评估与回滚锚定。
- **生命周期流程（Algorithm 1）**：
  1. **Sense**：统计 dev 漏判（misses）与 train 盲区/分歧（gaps）。
  2. **Grow**：按失败类别聚类，LLM 合成 typed op spec；通过出生门控（birth gate）：须在簇内触发≥50%且在已知正确输出上保持 CLEAN。来自 dev misses 的进入 active，仅来自 train gaps 的进入 shadow 池。
  3. **Select**：LLM composer 在算子目录+gap上下文+精英历史基础上生成候选表达式。选择目标为 $S(e) = A_{\mathrm{dev}}(e) \cdot A_{\mathrm{train}}(e)^{w} - \lambda C(e)$。$A_{\mathrm{dev}}$ 为召回加权（FAIL 权重 2，CLEAN 权重 1）；$A_{\mathrm{train}}$ 为可靠性加权（基于锚点留一边际贡献 $m_o$，权重 $1+10\max(0,m_o)$）；$C(e)$ 为复杂度惩罚。双门控（fail-closed anchoring + validity gate）过滤全通过/全拒绝候选。
  4. **Curate**：按 leave-one-out 边际贡献淘汰非正值算子；shadow 算子仅在证明能提升 dev 一致性后转正。
- **Double Ratchet 协同**：验证器阶段与技能阶段交替，验证器阶段前期较长（15/8/5/2轮）以快速覆盖缺陷，技能侧沿用 Ratchet 原有生命周期，互不修改对方核心逻辑。

## 实验与结果
- **数据集**：MBPP+（Python函数合成，隐藏单元测试为硬锚点）、Spider 2.0-Snow（企业级 text-to-SQL，执行结果比对为锚点）、Report generation（分析师报告生成，无准确验证器，仅用 rubric 作参考信号）。
- **模型与环境**：所有 in-loop 角色统一使用 Claude Opus 4.7，最终 outer judge 使用更强的 Opus 4.8；纯 CPU + API 调用，100 轮迭代，三种子随机。
- **主要数字**：
  - MBPP+ 演化验证器 held-out agreement 达 0.625±0.050，较手写 seed 组合（0.417±0.058）提升 **+0.21**
