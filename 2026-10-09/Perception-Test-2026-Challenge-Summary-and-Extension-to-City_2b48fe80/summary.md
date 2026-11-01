---
title: "Perception-Test-2026-Challenge-Summary-and-Extension-to-City"
source: https://arxiv.org/pdf/2610.12081v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:06:30"
field: "多模态视频理解与空间推理"
keywords: ["multimodal models", "spatial reasoning", "video QA", "audio-visual reasoning", "agent pipeline", "benchmark", "city-scale perception"]
innovations: ["提出KilometerAudio：首个面向超长自然视频的音频事件推理benchmark", "首次在同一challenge框架下评测table-top到city-scale的空间+音频跨模态推理", "揭示agent管道与单模型在长视频推理上的显著能力gap"]
benchmarks: ["Perception Test", "KilometerVision", "KilometerAudio", "Video-MME Long", "HOTA"]
---

# 论文速读：Perception-Test-2026-Challenge-Summary-and-Extension-to-City

## 一句话总结
本文报告了ECCV 2026第四届Perception Test挑战赛，首次引入基于城市尺度步行视频的两个新赛道（KilometerVision与KilometerAudio），并总结四赛道获奖方案；结果表明复杂空间与跨模态推理可通过昂贵的零样本agent管道近乎解决，但单一多模态模型独立完成任务仍具挑战。

## 研究问题与动机
1. 前沿AI模型在数学、编程等领域已超人类，但在基于被动互联网数据训练的感知能力（尤其是空间理解）上仍落后于普通人，缺乏环境交互学习是关键原因。
2. 现有benchmark多聚焦短视頻或小规模场景，难以评估城市尺度下的长期空间推理能力，需引入更长、更贴近真实世界的视频评测体系。
3. 音频-视觉联合推理在超长自然视频（小时级walking-tour）中的评估几乎空白，需要专门benchmark来检验多模态模型对"声源空间定位"与"时空关联"的推理能力。
4. 希望检验当前旗舰多模态模型（如Gemini、GPT、Qwen系列）在统一多模态-空间-音频推理任务中的真实上限，识别"已解决"与"仍困难"的问题类型。

## 核心贡献（创新点）
1. **提出KilometerAudio benchmark**：首个针对超长（平均70+分钟）自然walking-tour视频的音频事件推理基准，问题聚焦于音频事件识别、时空定位与多模态关联；区别于Video-MME Long等仅少量音频问题的benchmark。
2. **扩展KilometerVision至城市尺度多项任务**：包含landmark recognition、compass、loop closure、map trace等7类空间推理题型，以地图图像与距离估计等多样化输入形式呈现；区别于原有table-top尺度Perception Test。
3. **组织四赛道挑战赛并建立可比较评测体系**：Unified VideoQA、Grounded VideoQA、KilometerVision、KilometerAudio并行评测，首次在同一框架下比较小规模桌面对象推理与大规模城市空间推理的能力边界。
4. **实证揭示agent管道与单模型的能力差距**：发现顶级solution几乎均为zero-shot agentic pipeline + 外部工具（几何重建、音频索引、OCR等），而使用单一多模态模型的best generalist entry在city-scale赛道显著落后；点明当前多模态模型独立泛化的瓶颈。

## 方法详解
本文为主办方challenge summary，不提出统一方法，但总结了各赛道获奖方案的典型技术路线：

- **Unified VideoQA（多选项视频问答）**：
  - Njust-KMG（冠军）：Question-type-aware weighted ensemble of three models；Qwen3-VL-32B在PT训练集上fine-tune（含curriculum on hardest question types + option shuffling）；Doubao-Seed-2.0-Pro与Gemini 3.1 Pro以question-type-specific prompts zero-shot调用；ensemble权重基于per-type validation accuracy。
  - F423（亚军）：Doubao-Seed-2.0-Lite单模型zero-shot；evidence-first prompting（先提取visual/temporal/audio evidence再比对选项）；adaptive frame sampling（1–2 fps按question type调整）；严格answer validation，最多一次re-query。

- **Grounded VideoQA（带bbox轨迹的视频问答）**：
  - Banana Bats（冠军）：Gemini-SAM3架构——Gemini 3.7 Flash zero-shot构建tracking plan（target identities、visibility intervals、anchor frames/boxes），1 fps sparse boxes，经critic验证后由SAM 3双向传播，reliability check通过才融合；多目标用amodal-capable SAM 3变体（external data训练residual head）。
  - Huawei SpaceMind（亚军）：GPT-5.6 Sol zero-shot识别answer objects并生成完整box tracks，再用SAM 2.1仅 refinement box coordinates（不改变object identity），辅以close-up crop repeat queries。

- **KilometerVision（城市尺度空间推理）**：
  - ohmyyuan（冠军）：Evidence-log video agent——Gemini 3.7 Flash从frame montages/keyframes构建带timestamp的street names/heading changes/landmarks日志，配合family-specific playbook；self-consistency over multiple sessions + deterministic estimators（optical-flow yaw odometry + visual place recognition）可override majority vote；independent verifier重解模糊问题。
  - Huawei SpaceMind（亚军）：GPT-5.6 Sol语义推理 + VGGT metric camera path reconstruction + metric-depth model；question-specific solvers（yaw summation for compass、start–end displacement for distances、route-shape matching for map）决定642/986答案。
  - CloudAI-evolve（best-performing, 非获奖）：agentic pipeline with unlimited token budget，探索validation set与约10% test questions以学习策略；调用metric 3D reconstruction (VGGTomega)、object-size-based scale estimation、map grounding等工具。

- **KilometerAudio（城市尺度音频-视觉推理）**：
  - IUCV（冠军）：AVRA——training-free audio-visual reasoning agent基于Gemini 3.7 Flash；question-independent index precomputation（BEATs 447 AudioSet classes audio-event scores、object detection、multilingual OCR）；test时fresh agent session从index定位候选time windows，直接检查raw audio/video，保留evidence log，先推导答案再read options。
  - WW11（亚军）：Zero-shot tool-using agent——Gemini 3.1 Pro规划搜索与reasoning（每question ≤48 frames/crops）；offline indexes（Whisper speech transcripts + CLAP sound-event retrieval）；Qwen3-Omni验证短音频片段；answer仅在≥2独立modalities agree时更改。
  - CloudAI-evolve（best-performing）：类似KV track的agentic pipeline，使用signal-processing tools（而非3D reconstruction）compile策略，所有reasoning步骤用Gemini模型。

- **Best Generalist（Wind_Rain_Tower）**：单一Qwen3.8-Max模型无task-specific detectors/trackers/geometry solvers；KilometerAudio额外调用独立speech-recognition模型转录audio。

## 实验与结果
- **数据集**：
  - Unified VideoQA：13,370 questions（11,528 original 3-option + 1,842 unified 5-option），无train split，zero-shot评估。
  - Grounded VideoQA：932 test videos、1,859 questions，HOTA评估。
  - KilometerVision：54 test videos、986 questions（6类问题），无train。
  - KilometerAudio：218 test videos（1–3 h each）、500 questions（5-option），无train，validation 6 videos/8 questions复用2024–2025年音频相关题。

- **关键结果**（Top-1 accuracy / HOTA）：

| 赛道 | Baseline | 亚军 | 冠军 | Best-performing（非获奖） |
|---|---|---|---|---|
| Unified VideoQA | 0.313 | 0.910 (F423) | 0.914 (Njust-KMG) | — |
| Grounded VideoQA | 0.052 | 0.654 | **0.664** (Banana Bats) | — |
| KilometerVision | 0.196 | 0.723 | 0.753 (ohmyyuan) | **0.905** (CloudAI-evolve) |
| KilometerAudio | 0.174 | 0.752 | 0.782 (IUCV) | **0.864** (CloudAI-evolve) |

- **结论**：
  - 除Grounded VideoQA外其余三赛道性能接近饱和（>80%）。
  - 相比往届，Unified VideoQA从0.812提升至0.914（+10.2pp）；Grounded VideoQA HOTA从0.497提升至0.664（+33%）。
  - 2026年新增unified问题类型（tracking/action ordering）从65.4%跃升至86.2%，而原始set仅从83.7%→92.2%。
  - 最难点集中于intuitive physics（占10道最难题中的8道），以及city-scale的map trace与audio事件定位。
  - Best Generalist（单一模型）在Unified QA达0.896（98% top），但Grounded VideoQA仅0.465 HOTA（69% top），city-scale赛道（0.506–0.604）差距更大。

## 相关工作脉络
1. **Perception Test 2023/2024/2025（Pătrăucean et al., 2023; Heyward et al., 2024, 2026）**：前身benchmarks，聚焦table-top小规模视频QA；本文首次将scale扩展至city-level。
2. **KilometerVision（Mahendran et al., 2026）**：原有city-scale spatial benchmark；本文在其基础上扩展challenge track，并新增KilometerAudio联合评测。
3. **Video-MME Long（Fu et al., 2025）**：最长视频可达1h，但仅少量问题聚焦audio events；KilometerAudio在视频长度与audio-focused设计上优于它（见Figure 4对比）。
4. **JointAVBench（Chao et al., 2026）/Maverix（Xie et al., 2025）**：现有audio-video benchmarks多基于short clips或合成数据；本文使用真实unedited hour-long walking videos，强调out-of-view audio推理的独特性。
5. **VLM+Tracker架构（SAM 2/3, Gemini-SAM等）**：Grounded VideoQA赛道体现了"VLM负责语义定位 + promptable tracker负责dense propagation"的分工范式，延续了MDETR等早期端到端multimodal detection的思路但性能大幅提升。
6. **Agentic pipeline for vision（VGGT, BEATs, CLAP, Whisper等）**：City-scale赛道显示外部tool-augmented agents可逼近human-level，但与standalone VLM之间存在显著gap，呼应了"tool use vs intrinsic capability"的讨论。

## 局限性与未来方向
1. **Single-model generalization仍不足**：Best Generalist在city-scale赛道仅达top成绩的65–80%，说明当前多模态模型在超长视频+复杂空间/音频推理上尚未具备独立泛化能力。
2. **Agentic方案成本高昂**：获奖方案多依赖unlimited token budget、多次re-query与外部工具调用，计算与时间开销大，难以部署。
3. **Intuitive physics仍是难点**：即使Unified QA接近饱和，physics类问题（温度判断、运动outcome预测）准确率仍低（<50%），缺乏系统性解决方案。
4. **缺乏train data的约束**：KilometerAudio/Vision均为zero-shot设置，限制了模型上限比较的公平性；有train data时成绩可能更高。
5. **未来方向**：开发更高效的city-scale理解模型（减少agent依赖）、探索单模型如何在长视频上进行有效的temporal reasoning与跨modal alignment、研究low-cost but high-performance spatial/audio grounding方法。

## 研究启发与可借鉴点
1. **Evidence-first prompting + adaptive sampling**：F423的evidence-first策略（先extract visual/temporal/audio evidence再compare options）与按question type自适应帧率（1–2 fps）可直接迁移到长视频QA任务中，降低token消耗同时提升准确率。
2. **Precomputed index + fresh agent session**：IUCV在KilometerAudio中的question-independent index（BEATs audio scores + object detection + OCR）值得借鉴——先离线构建全局索引，测试时仅聚焦候选time windows，可大幅缩减推理窗口。
3. **Deterministic estimators作为verifier**：ohmyyuan利用optical-flow yaw odometry与visual place recognition等确定性估计器override majority vote，这种"model-free fallback"策略可用于提升agent系统在模糊case下的鲁棒性。
4. **Ensemble by question-type validation accuracy**：Njust-KMG的per-type ensemble权重设计是一种简单而有效的策略，可按题型分别calibrate各模型置信度，适用于多模型混合评测场景。
5. **与团队方向的结合机会**：若团队关注长视频理解或多模态推理，可参考本challenge的benchmark设计（hour-long videos, audio-centric questions）构建内部评测；若研究agent system，可借鉴"evidence log + self-consistency + verifier"的三层架构。

## 关键术语表
**Perception Test**：由Google DeepMind等发起的多模态视频理解诊断benchmark系列，逐年扩展评测规模与模态，2026年首次引入city-scale audio-visual tracks。

**KilometerVision**：基于真实YouTube walking-tour视频（最长约10分钟/约1KM）的城市尺度空间推理benchmark，包含landmark recognition、compass、map trace等7类问题。

**KilometerAudio**：本文新提出的benchmark，使用1–3小时自然walking-tour视频（含未编辑的ambient audio），聚焦audio event识别、时空定位与audio-visual关联推理。

**HOTA（Higher Order Tracking Accuracy）**：多目标跟踪评估指标，同时衡量检测精度（DetA）与关联精度（AssA），本文Grounded VideoQA赛道采用。

**VLM（Vision-Language Model）**：视觉-语言多模态大模型（如Gemini、GPT-4V、Qwen-VL等），在本challenge中作为各赛道solution的核心推理引擎。

**Agentic pipeline**：指由LLM/VLM作为"大脑"调度多个外部工具（OCR、audio indexer、3D reconstructor等）并完成multi-step reasoning的系统架构，是city-scale赛道的主流 winning strategy。

**Self-consistency**：通过对同一问题多次独立推理并取多数投票以提升鲁棒性的方法，ohmyyuan与IUCV均在多session聚合中采用。

**Evidence log**：agent在推理过程中显式记录的关键证据（timestamp、object box、audio event、heading change等），供后续verifier或self-consistency聚合使用。

## 可复现要素
- **KilometerAudio数据集**：基于218个公开YouTube walking-tour视频（1–3 h），test questions与annotations由human raters标注，benchmark发布后需向组织方申请访问；论文未提及公开代码。
- **KilometerVision数据集**：基于公开walking-tour视频片段，benchmark论文（Mahendran et al., 2026）与challenge网站提供进一步访问信息。
- **Perception Test baseline与赛道入口**：EvalAI平台托管，challenge website（https://perception-test-challenge.github.io/）提供各team technical report链接。
- **关键超参**：论文未系统披露各team超参（如frame sampling rate、token budget、index dimension等）；各team report链接可查。
- **开源情况**：部分底层模型权重开源（Qwen3-VL、SAM 2/3、BEATs、Whisper、CLAP等），但challenge solution代码未统一开源；benchmark本身对研究社区开放申请。
