---
title: "WOVEN-Weaving-Visual-World-Modeling-into-Multimodal-LLMs"
source: https://arxiv.org/pdf/2610.12417v1.pdf
model: agnes-2.5-flash
chunks: 7
summarized_at: "2026-11-01 14:17:54"
---

# 论文速读：WOVEN-Weaving-Visual-World-Modeling-into-Multimodal-LLMs

## 一句话总结
WOVEN构建了一个以“动作轴×推理轴”为二维网格的多模态大模型世界建模基准，系统诊断了现有MLLM在物理、因果与时间推理上的碎片化失败模式；通过严格的受控训练对照实验，证实只有包含真实动作-结果语义的过渡推理数据才能驱动跨分布泛化与外部任务增益。

## 研究问题与动机
- **现有基准的碎片化缺陷**：世界建模能力具有内在的二维结构（动作轴×推理轴），但现有基准各自仅覆盖网格的单一边缘，导致MLLM的失败模式被误判为互不相关，无法系统归因。
- **MLLM与VGM的能力断层**：视频生成模型在动作条件rollout上显著优于MLLM，但VGM预训练引擎所持有的$(s, a, s')$先验知识并未被线性提取，MLLM缺乏同等的可查询世界建模能力。
- **缺乏统一的认知科学形式化**：动作类型（外源性/感知性/审视性/导航性/操纵性）与推理深度（前向/逆向/去除/替换/预测/时序）尚未在多模态大模型评测中被联合交叉实例化。
- **训练增益的混淆因素未排除**：现有MLLM微调工作难以区分能力提升源于真实动作-结果语义学习，还是源于题目格式、答案字母平衡或单纯训练量增加。

## 核心贡献（创新点）
- **提出二维网格基准WOVEN**：将世界建模能力形式化为“动作类型轴 × 推理深度/方向/时间粒度轴”，首次实现多基准失败模式的系统性交叉诊断与cell级归因。
- **构建大规模受控评测与数据集**：提供6,228个ID推理项、3,496个场景OOD项与1,116个扰动探针，配套IEM物理违规编辑器与F/C/I/S四类干扰项体系。
- **揭示MLLM世界建模能力的三维解剖学**：量化了旋转vs平移缺陷、外观vs几何鲁棒性解耦（增长速率差3.5倍），以及能力-表征规模的非线性关系。
- **建立严格的内容对照实验范式**：通过Real/Permutation/Global Mirror/A-OKVQA四对照，证明训练增益来源于正确的动作-结果语义映射，而非格式/字母偏好/数据量。
- **验证跨架构泛化与共享映射机制**：在Qwen2.5-VL与Gemma-3-4B-it上复现相同增益，并通过前后向一致性检验证实模型学习到统一的$(s,a)$状态转移映射。

## 方法详解
- **理论形式化**：状态转移定义为 $\tau_a = \mathcal{G}(s, a, \mathfrak{u})$，其中$\mathfrak{u}$为背景条件；反事实推理通过干预$\mathfrak{u}$保持初始状态不变来评估替代动作。
- **动作轴分类**：基于认知科学agency概念划分为五类：外源性（物理事件/重力/支撑/惯性等）、感知性（自我中心视角变换）、审视性（以物体为中心的检查/旋转）、导航性（目标导向位移）、操纵性（对物体的目标导向操作）。
- **推理轴正交结构**：因果深度包含Forward Dynamics、Inverse Dynamics、Removal、Substitution、Outcome/Cued Prediction、Ordering、Adjacency七种类型；另含推理方向（前向/逆向）与时间粒度（粗序/细邻接）。
- **基准构建流水线**：Stage 1文本规格化生成$(s_t, a, s')$；Stage 2场景抽样（10室内+10室外，保留restaurant-beach OOD对）；Stage 3图像/视频生成与IEM物理违规编辑；Stage 4干扰项分类设计；Stage 5人类审计与扰动探针注入。
- **评估与统计协议**：采用4-Way MCQ为主，引入循环评分(circular scoring)消除字母偏好；定义超chance阈值（accuracy>25%且单侧二项p<10⁻⁴）；支持Spearman秩相关、cell级难度相关与公共参考归一化。
- **训练与消融设计**：设计11个受控微调子集（如`insp_cf`、`exo_physical`、`full-mix`等）对比SFT与RL；内容对照含真实映射、随机置换、全局镜像、通用VQA；设置移除“未变化场景”选项、前后向cycle rate检验、跨家族复现等控制实验。

## 实验与结果
- **现有引擎探针**：冻结V-JEPA2 linear/attentive分别达39.3%/47.7%，视频生成器(I2VGen-XL) 36.4%，图像生成器(OmniGen) 19.5%，均远低于人类基线(92.3%)，证实视频先验非线性可及。
- **ID与OOD性能**：`full-mix` SFT达ID 89.29%/OOD 88.16%；单类别最佳`insp_cf` SFT为50.75%（ID）/52.46%（OOD）；RL阶段扰动鲁棒性普遍下降，仅`exo_physical` RL外观达64.16%。
- **能力缺陷诊断**：F.1旋转缺陷普遍（inspective<manipulative，translation>>rotation，gap 16.2–35.6pp）；F.2外观鲁棒性增长速率(39.8 pts/dec)是几何(11.1 pts/dec)的3.5倍，7B模型Δ=27pp；F.3能力与表征规模解耦。
- **错误模式分布**：时间邻接错误以方向错误为主（72.9%）；物理建模错误以Pass through/Transform effect为主；动作混淆中正向动力学准确率97.31%（full-mix），逆向94.62%。
- **内容对照结论**：Real训练在SPAR-Bench view change (+28.5pp)与SAT (+19.2pp)上显著优于Permutation/Mirror/A-OKVQA（p<0.001）；循环评分下增益依然成立且base字母偏好受损更严重；前后向cycle rate从19%升至45–94%，证实共享映射。
- **配方验证**：82%预算分配给代理动作因果/反事实的Recipe策略在外部未见基准Set A中最优（37.9%）；选错推理操作（Temporal-heavy/Exogenous-heavy）较Recipe分别损失3.5/4.7pp。

## 相关工作脉络
- 对比VSI-Bench、ViewSpatial-Bench、3DSRBench、VLM4D、MindCube等视角/空间推理基准，本文指出其仅覆盖perceptive/inspective×forward等单一边缘，缺乏动作-推理交叉诊断。
- 对比IntPhys 2、PhysBench、CLEVRER、CausalVQA等物理/因果推理基准，本文强调这些工作未封闭MLLM世界建模能力，WOVEN将外源性物理原则与内源性agency动作统一于二维网格。
- 对比WM-ABench、WorldPrediction、WorldTest、PBench等现有世界建模基准，本文填补了“训练内容语义 vs 格式/规模”严格对照的实验空白。
- 对比V-JEPA2、I2VGen-XL、OmniGen等视频/图像生成引擎的probe实验，本文证实VGM在动作条件rollout上占优，但其先验对MLLM非线性可及，区别于直接微调VGM的工作。
- 在具身智能脉络上，继承Spelke核心知识、Haggard agency理论与背侧/腹侧通路分区，
