---
title: "STORAGE-IS-NOT-STRATEGY-STATE-CONDITIONED-SUPPORT-CONTROL-FO"
source: https://arxiv.org/pdf/2609.37858v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:49:25"
---

# 论文速读：STORAGE-IS-NOT-STRATEGY-STATE-CONDITIONED-SUPPORT-CONTROL-FO

## 一句话总结
本文通过控制实验证明LLM局部化遗忘中“参数定位准确”不等于“干预有效”，提出目标条件化的初始参数排序（Intervention Score）与检查点自适应门控重排（DIR-R），在Natural-TOFU与LACUNA基准上均优于十类静态定位/参数选择基线，同时揭示遗忘收益对数据领域的高度依赖性。

## 研究问题与动机
- 现有局部化遗忘方法通常假设：定位信号识别出的关键参数子集就是遗忘优化应更新的目标区域，且在优化全程保持该子集固定。
- 控制实验显示，即使存储定位AUROC高达0.981，其与更优干预策略的一致性仅17/36，而LoRA接口获胜35/36，说明定位精度无法可靠预测特定遗忘目标的干预效果。
- 同一可编辑几何下，NPO与SimNPO独立选出的参数子集Jaccard交集仅约0.143，表明最佳干预位置强烈依赖遗忘目标本身。
- 随着优化推进，模型状态与优化器状态发生变化，静态机制定位可能“过时”，但现有工作缺乏对“初始支持选择”与“检查点重排”两阶段的独立形式化与实证检验。

## 核心贡献（创新点）
- **解耦定位与干预：** 构造合成事实控制实验，明确证明高精度存储定位不保证遗忘干预的有效性，两者必须分离设计与评估。
- **目标条件化初始选择（Intervention Score & STATIC-IV）：** 提出一阶代理评分，综合预测候选参数组对遗忘增益、保留损伤与中性漂移的贡献，据此选出初始支持集并全程冻结，提供可复现的静态方法基线。
- **检查点自适应选择性重排（DIR-R）：** 设计短探针门控机制，仅在检查点证据充分时触发完整的候选支持对比计算，以极低额外算力捕获大部分可用增益，实现效率与性能的平衡。

## 方法详解
- **支持约束优化形式化：** 将遗忘任务定义为在预算 $B$ 下选择可编辑参数组子集 $\mathbb{S} \subseteq \mathbb{G}$，终端效用为 $J(\pmb{\theta}) = G_F(\pmb{\theta}) - D_{\text{coll}}(\pmb{\theta})$。支持约束更新为 $\pmb{\theta}_{k+1}^{\mathbb{S}} = \pmb{\theta}_{k}^{\mathbb{S}} - \eta_k P_{\mathbb{S}} \mathbf{u}_k^{\mathbb{S}}$，其中 $P_{\mathbb{S}}$ 投影到活跃坐标，其余参数冻结。
- **Intervention Score：** 在 $\pmb{\theta}_0$ 计算目标无预处理方向 $\mathbf{u}_0$ 与四类诊断梯度 $\mathbf{g}_q$（$q \in \{F, R, P, N\}$，对应遗忘、保留、受保护行为、中性行为）。一阶近似下，$e_{g,q} = -\langle P_g \mathbf{g}_q, P_g \mathbf{u}_0 \rangle$，干预分数定义为：
  $$s_{\text{int}}(g) = \frac{e_{g,F} - \max\{e_{g,R}, e_{g,P}, 0\}}{|e_{g,N}| + \epsilon}$$
  分子奖励预测遗忘并扣除最大预测损伤，分母抑制非特异性移动。按 $s_{\text{int}}$ 降序贪心打包至预算 $B$ 得到初始支持 $\mathbb{S}_0$。
- **STATIC-IV：** 将 $\mathbb{S}_0$ 固定用于全部 $T$ 步优化，用于隔离初始选择质量，不与干预分数混入训练损失。
- **SELECTIVE DIR-R：** 在检查点 $\mathcal{C}_t$ 构造候选族 $\mathbb{S}^{(\rho)}$，将 $\mathbb{S}_0$ 中最低分的 $k(\rho)=\lfloor \rho M \rfloor$ 组替换为未选中最高分组，$\rho \in \{0, 0.01, 0.025, 0.05, 0.10\}$。先执行长度 $d=20$ 的短探针，计算信号 $s_t = \max\{0, \widetilde{V}_t^{(d)}(\rho_p \mid \mathcal{C}_t) - \widetilde{V}_t^{(d)}(0 \mid \mathcal{C}_t)\}$。若 $s_t \leq \tau^\star$ 则继续冻结；否则从 $\mathcal{C}_t$ 独立恢复并跑完所有 $\rho$，选取最大化 $J$ 的支持。阈值 $\tau^\star$ 在开发集按“单位额外步数捕获的最大可用增益”校准。

## 实验与结果
- **数据集与模型：** 控制实验使用Qwen2.5-7B注入合成事实；Natural-TOFU使用Llama 3 8B与Gemma 2 2B；LACUNA使用冻结OLMo 3 7B（覆盖Email Address、Driver’s License、Birth City、Phone Number四个PII字段）。
- **控制
