---
title: "WOVEN-Weaving-Visual-World-Modeling-into-Multimodal-LLMs"
source: https://arxiv.org/pdf/2610.12417v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-10-09 17:06:55"
---

# 论文速读：WOVEN-Weaving-Visual-World-Modeling-into-Multimodal-LLMs

## 一句话总结
论文提出 WOVEN，一个基于认知科学双轴框架（推理轴 × 动作轴）构建的大规模多模态世界模型诊断基准，通过联合实例化 8 种推理类型与 5 种动作类型，系统性揭示当前前沿 MLLMs 在物理子原理、视角变换与时序邻接上的级联失效模式，并验证经指令微调与强化学习的 full-mix 配置可将 ID 准确率从 26.4% 提升至 89.3%。

## 研究问题与动机
- 现有世界模型基准大多固定单一认知轴而让另一轴失控，导致 MLLM 的性能短板被分散掩盖，缺乏可解释的统一诊断视图。
- 多模态模型在独立物理/因果/空间基准（IntPhys 2、CausalVQA、MindCube、VLM4D 等）上分别表现出违反期望判断接近偶然、反事实干预推理薄弱、视角采择近乎随机等系统性缺陷。
- 视觉生成模型（VGMs）在同类任务上表现优异，但现有基准或以 VGMs 为主体、或限于网格/纯文本环境，未针对 MLLMs 的视觉-世界建模能力进行双轴联合评测。
- 缺乏一个跨具身、可控视觉泄漏、且具备认知心理学先验支持的标准化测试协议，难以对齐发展心理学与 AI 评估的研究需求。

## 核心贡献（创新点）
- 提出基于认知科学 agency 的动作类型双轴分类框架，取代传统按关节/轮子等具身方式组织的离散分类，实现跨平台可比性。
- 构建涵盖 6,228 项 ID、1,116 项扰动与 3,496 项 OOD 场景的大规模多模态测试集，通过 null pass 与卷积探针严格验证编辑伪影可控。
- 建立 8 种推理类型与 5 种动作类型的邻接矩阵，明确各组合的任务边界与干扰项设计层级（Fully-typed / Family-level / IEM-edit / Sequence-permutation）。
- 对 38 个前沿 MLLM 进行统一网格评测，首次量化 MLLMs 在 gravity/collision 等物理子原理、ego/allo 视角变换与时序 Adj 粒度上的性能断层。
- 提供 base vs full-mix（SFT+RL）对照实验，证明目标训练可使模型在双轴任务上逼近人类水平（ID 89.3% vs 人类 92.3%）。

## 方法详解
- **双轴理论形式化**：轨迹机制定义为 $\tau_a = \mathcal{G}(s, a, u)$，其中 $s$ 为初始状态，$a$ 为动作，$u$ 为场景背景条件；前向/逆向/移除分别恢复结果状态、动作、初始状态。
- **推理轴设计**：由因果深度（物理建模→干预层 do(u)→反事实层）、推断方向（前向/逆向）、时间粒度（粗粒度排序/细粒度邻接）三属性正交组织，形成 Fwd/Inv/Rmv/Sub/Otm/Cue/Ord/Adj 共 8 类。
- **动作轴设计**：按 agency 粗分为 exogenous（外部触发，依赖 Spelke 物体恒常性/凝聚性/固体性及重力/支撑/惯性/碰撞/包含事件）与 agentic（自我发起，细分为 perceptive/inspective/navigative/manipulative 四类，按参考系与目标导向区分）。
- **场景与数据划分**：包含 10 室内+10 室外场景，保留 (restaurant, beach) 作为 OOD 对；动作类型×场景分层 80%/20% 划分；验证集占候选池 ~10%（seed 42）；四向场景组（human/humanoid × egocentric/allocentric）作为原子单元防止视觉泄漏。
- **干扰项与控制机制**：四类干扰项对应不同追踪深度；外源性任务使用 Gemini 3.1 Flash Image 生成 IEM 编辑；通过 null pass 验证编辑器仅修改目标区域；卷积探针测试确认无显著伪影信号（平衡准确率 51.1% vs 零假设 50.9±0.4%，AUC 0.532 vs 0.520）。
- **评估与训练配置**：覆盖 38 个 MLLM；Human baseline 92.3% ID / 98.0% Pert；实验对比 base 模型与 full-mix（SFT+RL）配置，报告各推理/动作/子原理精度及错误分布。

## 实验与结果
- **推理类型整体表现**：GPT-5.4 以 Fwd 83.8%、Inv 81.7%、Otm 81.9%、Cue 85.8% 领先，但 Adj 仅 27.0%；Qwen3-VL-235B-A22B 逆向动力学达 83.4%；Qwen2.5-VL-3B 整体 <26%。
- **动作类型差异**：Mani 类最强（GPT-5.4 81.2%，Intern-S1-Pro 87.1%），Insp 类最弱；所有模型 ego 视角显著优于 allo 视角（差距 6–20pp）。
- **物理子原理**：gravity 最强（多数模型 75–77.8%），collision 最弱（GPT-5.4 43.8%，Qwen-VL-Max 18.8%），containment 次强（GPT-5.4 70.1%）。
- **扰动鲁棒性**：外观扰动准确率普遍 >95%，几何扰动成为最大瓶颈（GPT-5.4 仅 62.7%，GLM-4.6V 仅 5.4%）。
- **跨基准对比**：在 CLEVRER（Overall 55.2–63.3%）、WM-ABench（30.2–39.8%）、MVP（20.4–28.9%）、ERQA（33.8–38.2%）、SpatialViz（27.7–32.2%
