---
title: "Which-Rollout-Taught-It-That-BehaviorTrace-and-the-Limits-of"
source: https://arxiv.org/pdf/2610.10422v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 17:15:33"
---

# 论文速读：Which-Rollout-Taught-It-That-BehaviorTrace-and-the-Limits-of

## 一句话总结
论文发布 BehaviorTrace 开源评估框架，在在线 RL（GRPO）微调设置下检验梯度训练数据归因方法的有效性；结果显示表观归因信号主要源于梯度范数与模型流畅度混淆，单_run_ 的 rollout 级归因结果跨种子与采样会翻转，唯一稳定的正向信号出现在触发 token 的梯度方向上。

## 研究问题与动机
- **核心问题**：在线强化学习教会语言模型新行为后，能否准确定位是哪些训练 rollout 导致了该行为？当归因方法声称能做到时，如何验证其答案是真实的？
- **现有方法不足**：主流梯度归因（Influence Functions、TracIn、TRAK、GAS 等）主要在静态/监督设定下验证，缺乏在在线 RL 流程中的系统评估；易受梯度范数（与奖励方差正相关）和模型流畅度（与饱和行为标签高度共线）等混淆因素主导；现有工作多报告单次运行结果，未区分真实信号与采样/种子方差。
- **动机**：训练数据归因日益用于安全审计，错误归因会误导缓解策略；需要引入已知因果的种植行为（planted behavior）作为 ground truth，并配套严格的对照协议以暴露混淆与结构缺陷。

## 核心贡献（创新点）
1. **发布 BehaviorTrace 开源评估框架**，集成全梯度草图、种植行为设置与多类对照控制，填补了在线 RL 归因标准化基准的空白；与以往仅在监督/静态设定下验证归因方法的工作不同，本文首次针对 GRPO 在线流程提供可复现的因果评估环境。
2. **揭示并量化梯度归因的主要混淆机制**，证明无目标的梯度范数排序（norm-only）可复现大部分表观步级精度，且模型流畅度在饱和检查点可替代行为标签；与仅追求更高归因精度的既有研究不同，本文从对照实验角度证明多数“信号”实为非行为相关的结构因素。
3. **证明单_run_ 归因结果在种子与生成采样间极不稳定**，跨 seed 与同 checkpoint 的新采样会翻转结论；与常规依赖单次运行报告显著提升的做法不同，本文强调必须通过多 seed 与多 draw 评估才能区分真实方向与方差噪声。
4. **发现 token/position 级归因的稳定正向信号**，触发 token 梯度与 in-context 目标的余弦对齐在全部三个 seed 上显著；与整段 response 级归因因幅度和流畅度主导而失效不同，本文指出将归因粒度细化至具体 token span 是更可行的突破方向。

## 方法详解
- **实验设置**：以 Qwen2.5-1.5B-Instruct 为基座，先进行 200 步 SFT 暖启动（10 条 frobnitz→QZXBT 示例），再进行 1000 步 GRPO 在线微调；植入早期窗口（step ≤ 200）的奖励污染（contamination fraction 0.5, bonus reward 8），ground truth 为被加权的早期 rollout 集合。
- **全梯度草图**：采用 CountSketch 对完整梯度进行降维（65536 个桶），每个参数坐标经 splitmix64 哈希映射桶与符号，避免相邻参数相关性，确保触发 token 及相关参数不被切片遗漏。
- **归因估计器**：
  - 步级：cross-step cosine（即 GAS / 重归一化 TracInCP）、TRAK-style 白化估计器、TracInCP；以及无目标的 norm-only 对照、local-buffer 对照、random。
  - Rollout 级：GAS (group-centered) 与 norm-only，在组内计算 AUC。
- **行为目标构造**：① in-context target（在实际回复中 QZXBT token 出现位置的梯度）；② constructed target（在回复起始处拼接 “QZXBT”）；③ minimal-pair contrastive target（含触发词的句子与含中性词的句子做 teacher-forced 差分，抵消流畅度）。
- **损失与梯度**：per-token log-probability 目标（仅 completion tokens，EOS masked，length-normalized）；advantage-weighted 梯度为 advantage × plain gradient，sketch 保持线性可加。
- **评估指标**：步级采用 tie-aware fractional precision at |GT|；rollout 级采用 within-group AUC 并附 bootstrap 置信区间。通过组内线性残差化或匹配对控制 fluency，报告 fluency baseline（mean token log-probability）作为必须超越的底线。

## 实验与结果
- **数据集与基线**：Qwen2.5-1.5B-Instruct，3 个随机种子；基线包括 cosine/dot (GAS)、TRAK-style、TracInCP、norm-only、local-buffer、random。
- **步级归因**：整体 precision 约 0.404–0.421，约为 chance（0.087–0.100）的 4.2–4.5 倍；norm-only 在无目标情况下达到 0.422/0.396/0.385，在 2/3 个 seed 上等于或超过最佳有目标估计器。在训练窗口内（nonzero-early），所有方法（含 random）均落在自身 base rate 附近约 ±0.03， achievable headroom 仅约 0.11，说明步级归因只能区分“是否被训练过”，无法定位具体 rollout。
- **Rollout 级归因**：饱和检查点原始 fluency baseline AUC 达 0.724–0.803，超过所有梯度方法；控制 fluency 后（minimal-pair target），GAS 方向与 norm-only 幅度在三个 seed 上交替占优（seed 0 幅度胜，seed 1 方向胜，seed 2 两者均超 chance 但幅度仍更高），且同一 checkpoint 更换一次生成采样即可翻转结论（如 seed-1 方向从 0.624 降至 0.526）。
- **Token 级正向发现**：触发 token 梯度与 in-context target 的余弦值在 3 个 seed 上分别为 +0.230、+0.114、+0.295，所有 95% CI 下界均高于预注册阈值 0.05；constructed target 接近零或负相关。该信号在整段 response 梯度中被幅度和 fluency 淹没，但在 token 级稳定显现。

## 相关工作脉络
- **Influence Functions / TracIn / TRAK / GAS**：Koh & Liang (2017)、Pruthi et al. (2020)、Park et al. (2023)、Hammoudeh & Lowd (2022) 等主要在静态或监督设定下提出或改进归因估计器；本文不提出新估计器，而是将这些成熟方法移植到在线 RL（GRPO）并检验其实际信号来源。
- **LLM 行为与涌现失准归因**：Xiao & Aranguri (2026) 提出 probe-based 归因但未验证 RLHF/SFT 扩展性；Vetter et al. (2026) 建立相关性但未验证因果贡献；Blank et al. (2026) 发现阿谀行为分散于偏好集难以过滤。本文采用种植行为与 leave-out 重训验证因果特异性，并补充系统混淆对照。
- **RL 专用归因**：Hu et al. (2025) 提出 local-buffer 在线归因框架；本文将其作为结构性基线，并首次加入跨种子/跨采样方差控制以暴露单 run 评估的不可靠性。
- **RL 放大无害奖励的失准**：Jørgenvåg et al. (2026) 指出 GRPO 学习行为需约 100 条 SFT 热身；本文沿用类似短热身设计（10 条/200 步），并在其上检验归因工具的有效性边界。

## 局限性与未来方向
- **局限性**：仅测试单一模型规模（1.5B）与单一算法（GRPO）；种植行为为表面可标记的合成 token，无法代表无表面形式的行为（如阿谀）；rollout 级分析使用重生成的 sibling 而非原始训练 rollout（日志仅存 hash）；仅在一个检查点 step 100 进行分析；fluency 控制仅为线性残差化与配对；token 级信号基于 tied embedding，尚未在复杂 span 上验证。
- **未来方向**：将归因粒度细化至 token/position 级或特定 span 评分；拓展至无表面形式行为的因果归因；在 PPO/DPO 等其他 RL 算法及更大参数规模上验证；开发非线性 fluency 控制方法；建立多 seed 多 draw 的标准评估协议。

## 研究启发与可借鉴点
- **种植行为+因果 leave-out 验证**的设计可直接迁移到任何希望检验归因工具真实性的研究，避免仅凭相关性或单 run 指标下结论。
- **系统混淆对照协议**（norm-only、fluency baseline/residualization、headroom ceiling、multi-seed/multi-draw）应成为 RL 归因论文的标配，而非事后补充；本 checklist 具有高度可复用性。
- **tie-aware precision 与 within-group AUC** 的度量设计有效剥离了步内共享梯度与组内对比噪声，值得在策略评估与溯源任务中借鉴。
- **Token/Position 级归因思路**提示：当整段输出梯度被幅度和生成概率主导时，缩小归因单元至具体触发 span 可能恢复可检测的信号，可与本团队的 span 级影响力分析方向结合。
- **预注册评估与 bootstrap 跨采样置信区间**的做法提升了结论的可信度，适用于任何对随机性敏感的 RL 实验报告规范。

## 关键术语表
- **BehaviorTrace**：论文开源的在线 RL 训练数据归因评估框架，集成全梯度草图、种植行为设置与多维度混淆控制。
- **GRPO (Group-Relative Policy Optimization)**：论文采用的在线 RL 微调算法，通过对组内奖励标准差标准化优势函数来更新策略。
- **GAS (Gradient Aggregated Similarity)**：重归一化的 TracInCP 估计器，通过计算 rollout 梯度与行为目标的余弦相似度进行归因。
- **Planted Behavior (种植行为)**：通过隐藏触发词（frobnitz→QZXBT）与早期
