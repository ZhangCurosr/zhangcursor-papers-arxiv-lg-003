---
title: "Policy-Alignment-New-Signals-for-Membership-Auditing-in-On-P"
source: https://arxiv.org/pdf/2610.11423v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 07:25:21"
field: "大语言模型隐私与安全"
keywords: ["membership auditing", "on-policy distillation", "privacy", "teacher alignment", "LLM security"]
innovations: ["提出 TAG 通过两端点差估计教师引导更新方向", "首次识别教师对齐为 OPD 成员信号"]
benchmarks: ["MATH", "MBPP", "MMLU-Pro", "BigCodeBench", "MulDimIF"]
---

# 论文速读：Policy-Alignment-New-Signals-for-Membership-Auditing-in-On

## 一句话总结
本文针对 On-Policy Distillation (OPD) 中的提示词级成员审计问题，首次识别"教师引导的策略对齐"作为新信号，提出 PAMA 框架并通过 Teacher Alignment Gain (TAG) 量化学生更新是否朝向减少教师损失的方向，在 MATH 上 AUC 达 0.791–0.941，相对提升 14.6–20.6%。

## 研究问题与动机
1. **核心问题**：OPD 使用的提示词往往包含私有/敏感数据，需对候选提示词判断其是否被用于 OPD 训练（prompt-level membership auditing）。
2. **现有似然方法不足**：Likelihood/Zlib/Min-K 等方法依赖预定义目标响应或输出置信度，但 OPD 中学生每次采样新响应，无法直接评估提示词成员性。
3. **现有 drift 方法不足**：DIBA 等方法仅测量 checkpoint 间策略变化的幅度，忽略该变化是否由教师指导（即"朝哪个方向移动"）。
4. **新观察**：成员提示词直接参与教师引导的梯度更新，非成员提示词仅通过跨提示词泛化间接受影响，二者在 teacher-aligned 方向上的轨迹存在可区分差异。

## 核心贡献（创新点）
1. **首次识别教师引导的策略对齐为 OPD 成员信号**：成员提示词驱动直接的老师损失下降更新，非成员仅通过跨提示词泛化产生弱信号；这与已有方法仅看"变化幅度"有本质区别。
2. **提出 TAG（Teacher Alignment Gain）**：通过两次零阶近似从模型输出端估计 teacher-aligned 更新方向，无需访问训练梯度或内部参数；区别于 DIBA 的纯策略漂移度量。
3. **融合多信号的可审计框架 PAMA**：将 TAG 与 SD、TAEG 及 Zlib/Min-K 结合，在 Full-distribution 和 Top-K 两种教师访问设定下均稳定优于基线。

## 方法详解
1. **审计上下文设定**：对候选提示词 $x$，从 $S_{\mathrm{pre}}$ 采样 $R=10$ 条响应，提取所有 prefix $c_t=(x,y_{<t})$ 作为固定上下文集合，确保 pre/post 与 teacher 在同一输入上比较。
2. **Student Drift (SD)**：$s_{\mathrm{SD}}(x)=\mathcal{A}_x^{\mathrm{post}}[\mathrm{D}_{\mathrm{KL}}(\pi_{\mathrm{post}}\|\pi_{\mathrm{pre}})]$，度量 post 相对于 pre 的策略变化幅度。
3. **Teacher Alignment Gain (TAG)**：基于微积分基本定理，用两端点教师损失差估计沿直线路径的平均投影：
   $s_{\mathrm{TAG}}(x)=\mathcal{A}_x^{\mathrm{pre}}[\mathrm{D}_{\mathrm{KL}}(\pi_{\mathrm{pre}}\|\pi_T)-\mathrm{D}_{\mathrm{KL}}(\pi_{\mathrm{post}}\|\pi_T)]$。TAG>0 表示 post 在固定 pre 上下文上更接近教师。
4. **Teacher-Aligned Entropy Gain (TAEG)**：度量 pre→post 后学生-教师熵差距的缩减：$s_{\mathrm{TAEG}}(x)=\mathcal{U}_x^T(\theta_{\mathrm{pre}})-\mathcal{U}_x^T(\theta_{\mathrm{post}})$，作为置信度变化的互补信号。
5. **Top-K 访问适配**：将教师分布中隐藏概率质量聚合为 $\perp$，在 $\Omega_{K,t}(x)\cup\{\perp\}$ 上计算 KL/熵，保留高概率区间的信息密度。
6. **特征聚合**：$\phi(x)=[s_{\mathrm{Zlib}}^{\mathrm{cal}}, s_{\mathrm{MinK}}^{\mathrm{cal}}, s_{\mathrm{SD}}, s_{\mathrm{TAG}}, s_{\mathrm{TAEG}}]^\top$，经分层 5-fold 标准化和逻辑回归分类。

## 实验与结果
- **数据集**：MATH（主基准）、MBPP、Toy-add、MulDimIF、MMLU-Pro STEM、BigCodeBench Instruct 共 6 个。
- **模型族**：Qwen2.5-3B/14B、InternLM2.5-1.8B/7B、Gemma-3-4B/12B，小为 student、大为 teacher。
- **OPD 有效性验证**：MATH 上 Qwen 准确率 65.3%→84.3%，InternLM 43.2%→67.7%，Gemma 52.1%→78.9%。
- **最强结果**：MATH 上 PAMA AUC 达 0.791–0.941，相对 SOTA 基线提升 14.6–20.6%；Qwen-Full 完整信号 AUC=0.894、TPR@0.1%=0.244。
- **Top-K 访问表现**：对 Qwen/InternLM，Top-K 与 Full 持平或更优，说明高概率 token 已承载大部分对齐信号。
- **TAG 验证**：与梯度投影方向一致率达 88%；真实教师优于 surrogate 教师。
- **泛化边界**：跨提示词泛化程度越高（MATH>MBPP>Toy），审计可区分性越弱；Toy-add 因泛化低而审计强。

## 相关工作脉络
1. **Likelihood/Zlib/Min-K（Duan et al. 2024; Carlini et al. 2021; Shi et al. 2023）**：基于文本级似然/压缩长度检测，面向监督训练场景，不适配 OPD 的自生成轨迹。
2. **DIBA（Liu et al. 2026）**：RL 中通过策略 checkpoint 漂移审计；PAMA 继承 SD 思路但新增 teacher-directed 方向信号，弥补 DIBA 忽略"更新朝向"的缺陷。
3. **On-Policy Distillation（Agarwal et al. 2024; Gu et al. 2024）**：OPD 范式背景；本文聚焦其训练数据隐私泄露风险，此前无人系统研究其 prompt-level 审计。
4. **Membership Inference 理论（Shokri et al. 2017）**：经典成员推断攻击框架；本文将其扩展到 OPD 的"提示词级"而非"样本级"设定。

## 局限性与未来方向
1. **跨提示词泛化削弱可审计性**：当成员与非成员共享相似解题模式时，teacher-aligned 信号被泛化效应淹没；未来需探索泛化解耦或子空间隔离。
2. **无真实教师时性能下降**：surrogate 教师仅能提供近似对齐方向；未来可研究无需教师的自监督对齐度量。
3. **高泛化任务的严格 FPR 审计困难**：MATH 等高风险场景下 TPR@0.1% 仍有提升空间；需更强的信号组合或更精细的上下文选择。

## 研究启发与可借鉴点
1. **两端点差估方向信号**：TAG 用 $\mathcal{D}(\theta_{\mathrm{pre}})-\mathcal{D}(\theta_{\mathrm{post}})$ 逼近路径积分，这一"零阶近似梯度方向"的技巧可迁移到任何只需 checkpoint 输出的审计/归因任务。
2. **Top-K 访问作为去噪机制**：保留高概率 token 反而优于全分布，提示"信息密度"比"覆盖度"更重要；可推广至其他黑盒审计设定。
3. **多信号分层消融设计**：先 SD（幅度）→TAG（方向）→TAEG（不确定性），展示"幅度≠方向"的本质差异；这种逐层剥离的信号设计值得借鉴。
4. **严格低 FPR 指标并重**：同时报告 AUC 与 TPR@0.1%/0.5%/1%，避免单一指标掩盖实际安全需求；实验设计范式可复用。

## 关键术语表
**On-Policy Distillation (OPD)**：学生在自身生成的轨迹上学习对齐教师策略的蒸馏范式。
**Membership Auditing**：判断候选数据点是否属于某模型训练集的隐私审计任务。
**Teacher Alignment Gain (TAG)**：衡量学生对候选提示词的 pre→post 更新中减少教师损失的程度。
**Student Drift (SD)**：post-distillation 与 pre-distillation 策略在候选提示词上的 KL 差异幅度。
**Teacher-Aligned Entropy Gain (TAEG)**：pre→post 后学生-教师预测分布熵差距的缩减量。
**Top-K Access**：审计者仅能获取教师输出中概率最高的 K 个 token 及其概率的受限接口。
**Full-distribution Access**：审计者可获取教师完整 next-token 分布的接口设定。
**Zeroth-order Approximation**：无需梯度、仅通过前后端点输出差估计更新方向的方法。

## 可复现要素
- **数据集**：MATH、MBPP、MMLU-Pro、BigCodeBench、MulDimIF、Toy-add（均为公开数据集）
- **代码/权重**：论文未提及开源声明
- **关键超参**：LoRA rank=32, alpha=64, lr=1e-4; 采样温度 0.8, top-p=0.95, max_len=128; R=10 responses/prompt; Top-K 默认 K=10; 逻辑回归 C=1, L-BFGS, 最多 5000 次迭代, 5 次重复 5-fold CV
