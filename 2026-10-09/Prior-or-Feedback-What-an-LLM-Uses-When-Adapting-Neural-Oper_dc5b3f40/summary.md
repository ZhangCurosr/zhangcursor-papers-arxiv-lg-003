---
title: "Prior-or-Feedback-What-an-LLM-Uses-When-Adapting-Neural-Oper"
source: https://arxiv.org/pdf/2610.12325v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 14:07:00"
field: "科学智能体决策验证"
keywords: ["LLM agents", "neural operators", "hyperparameter optimization", "scientific discovery", "intervention analysis", "PDE adaptation"]
innovations: ["决策级干预协议分离先验与反馈依赖", "预算匹配下LLM优于TPE/随机搜索", "首步提议质量优于91.7%随机配置"]
benchmarks: ["PDEBench", "nRMSE held-out test", "action distance intervention"]
---

# 论文速读：Prior-or-Feedback-What-an-LLM-Uses-When-Adapting-Neural-Oper

## 一句话总结
本文研究在有限试验预算（20次）下，大语言模型（LLM）如何选择神经算子（FNO）的微调配置，并验证其决策是仅依赖初始任务描述（先验），还是同时响应实验反馈。结果表明LLM在跨族/族内迁移中均优于随机搜索和TPE，且其首步配置已位于优质区域，后续决策对任务描述和验证分数敏感。

## 研究问题与动机
- LLM科学代理在序列实验循环中，是仅依赖初始上下文做出决策，还是会基于观测到的实验结果动态调整策略？
- 在预训练神经算子（FNO）的微调场景中，目标物理机制偏离预训练分布时，如何用有限预算（20次trial）高效适配？
- 现有方法（随机搜索、贝叶斯优化如TPE）能否充分利用自然语言任务描述和训练诊断信息？
- 如何设计可控干预来分离LLM决策中对"先验"和"反馈"的依赖，而不仅看最终性能指标？

## 核心贡献（创新点）
- **预算匹配下的竞争性能**：在相同20次试验预算下，LLM策略在12个PDE族/跨族迁移单元中均获得更低held-out nRMSE，优于随机搜索（36/36）和TPE（35/36）。
- **决策级验证协议**：通过单次输入干预（冷启动改描述、反馈重分配）测量提议动作的距离变化，而非仅比较端点性能。
- **任务依赖的先验与反馈敏感的提议**：切换PDE描述使首步base learning rate中位数翻倍（0.0005→0.001，p=5.2e-9）；重分配验证分数使下一步动作平均偏移约1个categorical坐标（+0.085，p=0.0156）。

## 方法详解
- **适应问题设定**：冻结预训练FNO，固定架构（10输入帧、12 Fourier modes、宽度20、4谱层），在PDEBench轨迹上微调。每个trial从新鲜checkpoint开始，训练750条轨迹，评估350条验证集的nRMSE。
- **行动空间**：12维（8数值+4类别）：优化器{AdamW,Adam}、schedule{none,cosine,step}、base learning rate [1e-5,1e-2]、4个block乘数[0.5,2.0]、weight decay [0,1e-2]、rollout length k∈{1,2,4,8}、data loss{MSE,rel-L2,MSE+rel-L2}、两个物理损失权重[0,1]。
- **LLM策略**：使用deepseek-v4-pro，开启thinking和JSON模式，reasoning_effort=max。每步采样S=3个JSON配置，执行medoid（最小总距离配置）。系统提示包含PDE描述、数据规模、FNO架构、搜索协议和行动边界。
- **干预方法**：
  - 冷启动描述测试：在advection β=4.0和Burgers' ν=0.001单元，展示advection/Burgers'描述或无描述，测量首步base learning rate变化。
  - 反馈重分配：重建决策点，重分配已评估配置的验证分数（保持配置和分数但置换配对），与notation控制（保持分数值与配对，仅改写科学计数格式）对比，计算动作距离d(a,a')=(1/D)(Σ|ā_j-ā'_j|+Σ[kζ[a_j≠a'_j]])。
- **统计检验**：18个trace效应（每单元3seed×6cell），符号翻转检验（one-sided sign-flip test），最小可达成p值=2^-6=0.015625。

## 实验与结果
- **数据集**：PDEBench（公开），两个1D标量PDE族（advection与Burgers'），共12个适应单元（6族内+3正向跨族+3反向跨族），每单元3个随机种子，共36个trace。
- **基线**：Random search（无上下文/历史）、TPE（5次随机启动+15次模型提议，总20次；控制实验恢复默认10次启动并扩展至25次）。
- **主要结果**：
  - LLM在全部36个匹配运行中优于随机搜索，在35/36中优于TPE（仅Adv→Burg, ν=0.001 seed 43输0.0346 vs 0.0263）。
  - 族内最佳：Burg ν=0.1，LLM=0.00155 vs Random=0.00320 vs TPE=0.00248。
  - 跨族最佳：Adv→Burg, ν=0.1，LLM=0.00160 vs TPE=0.00180。
  - 首步提议质量：中位数优于60个随机搜索配置的91.7%，仅2/36低于单元随机中位数。
- **干预结果**：
  - 冷启动：切换描述使base LR中位数翻倍（p=5.2e-9），advection描述下80%提议选0.001，Burgers/无描述仅39-40%。
  - 反馈重分配：中位数动作距离+0.085（95% CI [+0.073,+0.108]），全部18 trace和6单元中位数为正，超0.05阈值；notation控制中位数为0。
  - 坐标选择性：rollout length变更49.6%、data loss 40.4%、schedule 25.9%，optimizer仅7.8%（接近重采样率7.0%）。

## 相关工作脉络
- LLM用于贝叶斯优化：Yang等(2024)、Zhang等(2023)、Agarwal等(2025)直接提议配置；本文关注LLM是否真正使用任务描述和反馈。
- LLM超参优化基准：Rodrigues等(2026)、Ferreira等(2026)、Huai等(2026)指出LLM优势非普适，常为默认配置；本文固定操作符微调场景，无默认配置。
- 硬件感知代码优化中的先验效应：Redko等(2026)研究LLM在硬件优化中的先验；本文操纵PDE描述和验证历史。
- 置换反馈控制：Wainrib等(2026)、Zhao等(2026)、Min等(2022)涉及因果扰动；本文结合动作距离度量而非仅检测变化。
- PDE智能体：Wuwu等(2025)、He等(2025)、Li等(2026)构建PINN工作流；本文固定FNO架构，仅研究配置策略。
- 代理行为复现性：Yagubyan(2026)、Liao(2026)测量重复调用一致性；本文结合干预设计评估决策依赖。

## 局限性与未来方向
- 仅评估一个LLM（deepseek-v4-pro）和两个1D PDE族，未扩展至更高维PDE或更多模型家族。
- 干预证明决策依赖任务和反馈，但未解释为何如此响应，也未验证决策是否理性或反映物理理解。
- 20次预算属有限场景，TPE在部分advection-to-Burgers运行中仍在改进，渐近优势未证实。
- 端点性能优势可能源于配置搜索效率，但反馈响应是否提升泛化未直接测量。
- 未来可扩展至多模型、多域、更大预算，测试敏感度泛化性及代理选择更优解的能力。

## 研究启发与可借鉴点
- **决策级验证协议可迁移**：通过单次输入干预测量动作距离，无需访问模型内部，适用于托管LLM的行为审计。
- **首步先验质量度量**：使用下游分布的lower-tail midrank（优于91.7%随机配置）评估冷启动质量，优于单纯性能比较。
- **反馈敏感性分析**：重分配验证分数而非删除历史，分离评分内容与格式影响，揭示代理对证据的因果依赖。
- **坐标选择性响应**：识别rollout length和data loss为反馈敏感维度，指导后续干预设计（如针对性扰动关键超参）。
- **预算匹配比较**：扩展TPE预算至25次仍不及LLM 20次，排除启动分配偏差，增强结论可靠性。

## 关键术语表
- **Fourier Neural Operator (FNO)**：学习函数空间间映射的神经算子架构，用于PDE求解器的代理建模。
- **PDEBench**：公开的PDE代理建模基准，提供预训练检查点和轨迹数据，支持族内/跨族迁移评估。
- **Action Distance**：定义于12维行动空间（8数值+4类别）的距离度量，数值用归一化绝对差，类别用指示函数。
- **Cold-Start Prior**：首次评估前仅凭任务描述的LLM策略偏好，反映初始先验而非预训练物理知识。
- **Feedback Replay**：重建已记录决策点，置换验证分数配对后重新生成下一步提议，测量动作位移。
- **Medoid Selection**：从S=3采样配置中选总距离最小的配置，限制异常完成影响但不纠正共享偏差。

## 可复现要素
- 数据集：PDEBench（公开，doi:10.18419/darus-2986/2987）。
- 代码：搜索代码公开于https://github.com/julian-8897/budgeted-search-ai4science；运行记录存档于doi:10.5281/zenodo.23213042，附重算脚本。
- 模型：deepseek-v4-pro（托管API），系统指纹fp_9954b31ca7_prod0820_fp8_kvcache_20260402。
- 超参：20次试验预算，5次随机启动（TPE），S=3采样，reasoning_effort=max，thinking/JSON模式开启。
- 训练：FNO预训练500 epoch/9k轨迹；每次适应最多100 epoch/750轨迹；验证350轨迹，测试前1000轨迹。
- 硬件：NVIDIA A100 80GB PCIe GPU MIG切片（约1/4卡）。
- 论文未提及：具体随机种子值、温度/nucleus设置（thinking模式忽略采样参数）。
