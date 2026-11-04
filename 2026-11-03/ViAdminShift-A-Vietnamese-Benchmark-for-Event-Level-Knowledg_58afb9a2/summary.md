---
title: "ViAdminShift-A-Vietnamese-Benchmark-for-Event-Level-Knowledg"
source: https://arxiv.org/pdf/2611.03171v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 18:22:38"
field: "Temporal Knowledge in LLMs / Low-resource NLP Benchmarks"
keywords: ["Vietnamese LLM benchmark", "temporal knowledge update", "event-level knowledge", "administrative reform", "prompting and temporal anchoring", "knowledge editing evaluation"]
innovations: ["Built a 7,995-item Vietnamese benchmark with 2x2 built-in controls from a real 2025 administrative reform", "Demonstrated the 'temporal default' phenomenon: most knowers return outdated answers without time cues", "Systematically compared date placement across system prompt, question prefix, tool, and earlier turn"]
benchmarks: ["ViAdminShift", "EvolvingQA", "ChroKnowBench", "PTC benchmark", "WikiRecentChanges"]
---

# 论文速读：ViAdminShift: A Vietnamese Benchmark for Event-Level Knowledge Updates

## 一句话总结
本文构建了越南语基准 **ViAdminShift**，基于2025年越南行政改革这一真实事件级知识更新，设计了含内置对照组的7,995项评测数据集，揭示了大多数LLM即便知晓新区划，在无时间线索时仍默认返回旧答案的"时间默认（temporal default）"现象。

## 研究问题与动机
- **核心问题**：LLM训练于固定文本快照，而现实事实持续变更；当更新以"事件"形式批量发生时，已知悉事件 headline 的模型是否能自动将更新传播至所有推论事实？
- **现有方法不足**：
  - 多数 temporal knowledge benchmark 采样的是孤立、因不同原因变更的事实（如球员转会、官员更迭），而非真实事件级批量变更。
  - 已有资源多为英语，直接翻译无法捕捉越南语行政命名、变音符号、单位词变更等语言特异性挑战。
  - 缺乏内置对照组（unchanged units）来区分"过时答案"与"其他错误"。
  - 部署型 LLM 的 time-awareness 评测多关注 parametric memory 的 log-probability 冲突，而非生成答案的实际行为。

## 核心贡献（创新点）
1. **基于官方源的事件级基准构建**：以2025年越南行政改革决议和国家统计局转换表为骨干，构建确定性模板生成的7,995项越南语评测集，而非依赖 LLM 改写或人工编写。*与 EvolvingQA/evolveQA 等由 GPT 生成问题的资源本质不同，本题答案由法律文件精确确定。*
2. **内置 2×2 对照设计分离过时错误**：通过 province-changed × ward-changed 二维交叉，设置纯对照组（pc=0, wc=0）与中间组，使"过时答案"与"地理错误"可区分。*区别于 ChroKnowBench/EvolvingQA 仅对比已变更与未变更事实，本文在同一事件下利用未受波及单位作为 control。*
3. **配对检索上下文与双轨评测**：为每个项目提供改革前后的 Wikipedia 修订摘录或 dated template，支持 closed-book 与 mixed-passage 条件；提出 probing track（短答+对照题组）与 benchmark track（全量7,995项）两条评测轨道。*与 WikiRecentChanges/PTC 等仅用单次 passage 的设计相比，本文显式建模新旧 passage 共存时的冲突场景。*
4. **揭示"时间默认"现象并提出评测规范**：系统验证了20个知悉改革的模型中14个在 plain question 下旧答案率≥89%，并在问题内加 date cue 时显著改善；提出 strict accuracy / new-first / old-only 三指标体系及 cue-dependence 判定标准。*超越了单纯"是否知道新答案"的二元判断，量化模型知识使用的条件性。*

## 方法详解
- **数据来源与 canonical tables**：以越南国家统计局（NSO）发布的旧→新 ward 转换表为核心，转录 Resolution 202/2025/QH15 的省一级合并关系，生成63→34省、696→0区县、10,035→3,321 ward 的五张规范表，以及10,566对 (old_ward, new_ward) 映射。
- **独立谱系交叉验证**：用 vietnamadminunits（政府门户网站解析）与 undertheseanlp/address 两个独立来源比对，10,565对中仅1处分歧，以国会常委会决议最终裁定；NSO表修正3个错误 old_ward code 并补1对 split ward。
- **2×2 控制设计**：每个 ward 级项目标记 `pc`（省是否被合并）与 `wc`（ward 是否变更）；纯控制（pc=0, wc=0）用于区分过时答案与其他错误。
- **7 类 item type**：T0 headline、T6 reverse、T2 composition、T1 membership、T1 new name、T3 existence、T4 address conversion、T5 successor；含3种 cue 变体（none / date "Tính đến…" / reform "Sau đợt sắp xếp…"）。
- **检索 passage 构建**：Wikipedia 取 reform 前最后一版与非 reform 后首版摘录，无对应 article 的 ward 则用 dated template（引用具体决议号）自动生成；共2,281 passage（1,002 Wikipedia + 1,279 template）。
- **双 scorer 与越南语归一化**：去变音符号匹配（đ↔d）、统一 hoa/hòa 音调惯例、Unicode NFC、去掉单位词后逐 syllable 匹配；probing-track scorer 经两位母语作者盲标验证（Cohen's κ=0.91/0.89）。
- **统计推断**：采用 province-clustered paired bootstrap（B=10,000）与 Rademacher wild cluster bootstrap 检验；效果成立需95%区间排除0且 p<0.05。

## 实验与结果
- **评测模型**：67个 distinct model（26 local + 41 API），涵盖 DeepSeek、Claude、GPT、Qwen、Kimi、GLM 等系列。
- **知识门控**：以 absorbed T1 units 在 reform cue 下 strict accuracy ≥0.50 定义 knower；67个模型中仅20个达标，本地开源模型仅 Muse-Glimmer-30B 通过。
- **时间默认现象**：20个 knower 中14个在 no-cue 条件下对 absorbed T1 仅给出新省名的比例≤11%；old-only 答案占比14–100%。bare date cue 可恢复至多100% strict accuracy。
- **Date placement 影响**：问题内 "as of ⟨date⟩" prefix 效果优于 system prompt 中 "today is" 句式；带 reasoning 时 "as of" 在问题中使7/8模型 old-only 降至0%。
- **Retrieval 与 2×2 结果**：小模型（不知改革）在 mixed passage 下仍给出14–30% 过时答案；5个 benchmark-track knower 仅0–5%。纯控制 cell 小模型正确率73–91%，而 pc=1,wc=1 单元格仅0–1%正确，88–95% 为 pre-reform 答案。
- **Post hoc 结论**：system prompt 中放入改革前日期（2025-01-01）使所有模型 old-only 率80–100%，印证日期位置对 temporal anchoring 的因果效应。

## 相关工作脉络
1. **EvolvingQA (Kim et al. 2024)**：基于月度 Wikipedia diff 构建，含 unchanged facts 与 outdated answers，但问题由 GPT-3.5 生成，针对 continual pretraining 的 T5 而非部署型 LLM prompting 评测。ViAdminShift 侧重事件级真实法律变更 + 部署 LLM。
2. **ChroKnowBench (Park et al. 2025)**：对比 variant vs. invariant facts，但 control 来自独立采样的"可能变而未变"事实；本文 control 来自同一事件未波及的单位，更严格。
3. **PTC benchmark (Hossain et al. 2026)**：用 Wikidata 职位持有者变迁定义 parametric temporal conflict，测量 log-probability 偏好；本文用生成答案行为观测相同现象，且覆盖事件级批量更新。
4. **WikiRecentChanges (Wallat et al. 2026)**：用 Qwen-3-14B 生成 passage，含 date-and-cutoff prefix；本文 passage 为真实 Wikipedia 修订或 dated template，并提供 paired retrieval contexts。
5. **TimeQA / TempReason**：英语 temporal QA benchmark，时间在 question 内；本文引入 system prompt / question / tool / earlier turn 多位置对比 date placement 效应。
6. **Vietnamese LLM benchmarks (VMLU, ViLLM-Eval, VLegal-Bench)**：覆盖考试、法律、通用任务，但无围绕真实世界知识更新的 benchmark；本文填补越南语 event-level knowledge update 评测空白。

## 局限性与未来方向
- **单一事件/单一语言**：仅基于越南2025行政改革，数字未必可迁移至其他重组事件或其他语言。
- **Template passage 占比高**：79% 的 new passage 来自模板而非真实文档，retrieval 性能可能被高估。
- **Gate 为行为定义**：实际 training cutoff 通常不可得，gate 能否预测 cutoff 未经验证。
- **Single-run 结果**：多数模型仅跑一次，aggregate rate 跨 repeat 可达10点波动；interval 未纳入该变异性。
- **Scorer 验证范围有限**：probing-track scorer 仅验证200条 T1/T1np 答案；benchmark-track scorer 未验证。
- **Future work**：测试其他事件级更新与语言；探索 fine-tuning / knowledge editing 能否移动模型的 temporal default，而非仅提升知识获取。

## 研究启发与可借鉴点
1. **2×2 控制设计范式**：用"事件波及×单位变更"双维度交叉分离错误类型，可迁移至任何区域重组、法规更新类 benchmark。
2. **Cue ladder + date placement 系统实验**：将同一日期置于 system prompt / question prefix / tool result / earlier turn 四位置对比，为 temporal prompt engineering 研究提供标准化协议。
3. **Old-only 作为独立指标**：建议评测 temporal knowledge 时除 strict accuracy 外，额外报告 old-only 与 new-first 率，避免将"默认过时"误判为"完全无知"。
4. **Provider-added date 检测**：本文通过 probing reseller 发现 GPT-5.5 自动注入当日日期；后续工作应将 endpoint date 探测纳入评测协议。
5. **Deterministic template 构建**：用官方表格 + 确定性脚本生成问题与 gold answer，避免 LLM 改写引入噪声，适用于任何结构化政策变更评测。

## 关键术语表
- **Temporal default**：模型同时持有新旧知识，但在无时间线索时默认返回旧答案的行为倾向。
- **Knowledge gate**：以 absorbed T1 units 上 strict accuracy ≥0.50（reform cue）作为判断模型是否"知悉改革"的阈值。
- **Cue-dependence**：reform cue 与 no-cue 在 absorbed units 上的准确率差 ≥0.30 且 paired bootstrap 95% CI 排除0。
- **Premise-leak control**：非 knower 在 reform cue 下 kept units 准确率变化应 <0.10，否则 cue 效应被 premise 诱导。
- **Probing track / Benchmark track**：前者用小规模配对短答题组做精细 cue 效应分析；后者用全量7,995项做描述性精度统计。
- **2×2 control design**：按 `pc`（province changed）× `wc`（ward changed）交叉划分 ward 级项目，纯控制 cell 用于分离过时错误与地理错误。
- **Old-only answer**：post hoc 指标，答案仅含旧单位名称（不含新名），为用户无法感知过时的"静默错误"。
- **Parametric temporal conflict (PTC)**： pretrained 模型参数中旧事实 log-probability 高于新事实的现象（Hossain et al. 2026 提出）。

## 可复现要素
- **数据集**：ViAdminShift v2025-07-01，含 7,995 items、2,281 passages、evaluation sets、model outputs、scorers、datasheet；CC BY-SA 4.0（数据）+ MIT（代码）；公开于 https://github.com/LLaammTTeerr/ViAdminShift 与 Zenodo DOI:10.5281/zenodo.23473403。
- **代码/权重**：全部构建代码与 scorer 开源（MIT）；本地模型使用 Q8_0 GGUF 权重（部分 bf16 转换）。
- **关键超参**：greedy decoding（temperature=0）；benchmark track MCQ 限16 tokens、short answer 限64 tokens；bootstrap B=10,000；cue-dependence 阈值 0.30；knowledge gate 阈值 0.50。
