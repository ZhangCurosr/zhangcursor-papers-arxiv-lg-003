---
title: "What-Drives-the-Zero-to-One-Jump-Diagnosing-and-Steering-Har"
source: https://arxiv.org/pdf/2611.02747v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:42:54"
field: "大语言模型强化学习推理"
keywords: ["RLVR", "Zero-to-One Jump", "Negative Learning", "Policy Optimization", "Generation Entropy", "Fisher Information", "Hard Prompt"]
innovations: ["提出基于早期分支驱动的NAPO方法，在GRPO全负组内对负样本进行分级优势赋值", "建立TTFP和N→P事件的事件级诊断框架，揭示硬提示零到一跳变的熵模式", "从Fisher信息角度证明高熵失败样本的政策可塑性，为负样本加权提供理论依据"]
benchmarks: ["AIME24", "AIME2025", "AMC23", "Math-Beyond", "LiveCodeBench"]
---

# 论文速读：What-Drives-the-Zero-to-One-Jump-Diagnosing-and-Steering-Hard-Prompt-Learning-in-RLVR

## 一句话总结
本文研究了RLVR训练中硬提示"零到一跳变"现象的内在动力学，发现即将发生转变的负样本在生成早期保留更高的熵，据此提出NAPO方法，通过早期分支驱动指标对负样本进行分级优势赋值，在AIME24和AIME2025上将Pass@256分别提升10.0和6.6个点。

## 研究问题与动机
- **零到一跳变的诊断难题**：标准RLVR使用二元奖励，当整个采样组均为负样本时，GRPO类方法的advantage趋近于零，无法诊断哪些硬提示处于"可解"状态，也难以定位N→P转变发生的精确时刻。
- **全负组的梯度枯竭**：现有基于分组的目标（如GRPO）在全部负样本组中产生退化的梯度信号，导致模型在最需要探索的硬提示上停滞不前，直到偶然发生突变。
- **负样本异质性的利用不足**：近期工作（如w-REINFORCE、N-GRPO）虽对负样本施加额外惩罚，但通常采用均匀权重，未充分利用负样本间在熵模式上的系统性差异来区分"接近可解"与"死路"。
- **二元监督下的事件级分析缺失**：现有评测指标（如Pass@k）只能聚合最终能力，无法刻画训练过程中何时、如何发生从全负到首次正样本的转变。

## 核心贡献（创新点）
- **提出基于事件的诊断框架**：首次用TTFP（Time-to-First-Positive）和N→P事件刻画硬提示从零到一的转变动力学，揭示了转变前的熵模式差异。
- **发现早期分支驱动的预测信号**：发现即将发生N→P转变的负样本在响应开头保留更多不确定性，提出$d_{\text{early-ppl}}$指标，可在组内分离"近可解失败"与"死路"。
- **提出NAPO算法**：设计一种GRPO集成的负样本加权策略，利用驱动指标对全负组内的 rollout 进行排序并赋予分级优势，精准注入梯度而不破坏已解组的稳定更新。
- **提供Fisher信息理论解释**：从Fisher信息矩阵的角度证明高熵失败样本对应更大的政策可塑性，均匀惩罚浪费优化预算，为NAPO提供了理论支撑。

## 方法详解
- **Token级熵度量**：对每个rollout计算top-k截断后的下一token分布熵序列$H_t^{(k)}$，并取其指数形式$\rho_t^{(k)} = \exp(H_t^{(k)})$作为perplexity/有效分支统计量。
- **Early-Branching Driver定义**：设响应长度$T$，取前$\alpha$比例（$\alpha=0.2$）计算 early mean-perplexity ratio：
  $$d_{\text{early-ppl}} = \frac{\frac{1}{T_\alpha}\sum_{t=1}^{T_\alpha}\rho_t^{(k)}}{\frac{1}{T}\sum_{t=1}^{T}\rho_t^{(k)}}$$
  $d>1$表示不确定性集中在响应早期，且与未来N→P转变正相关。
- **Driver归一化与优势赋值**：对全负组内所有rollout的driver分数做min-max归一化至[0,1]：
  $$\tilde{d}(x,y_i) = \frac{d(x,y_i) - \min_j d(x,y_j)}{\max_j d(x,y_j) - \min_j d(x,y_j) + \epsilon}$$
  优势函数为：
  $$A(x,y) = \begin{cases} A_{\text{GRPO}}(x,y), & \text{if } p(x)=1 \\ -\tilde{d}(x,y), & \text{if } p(x)=0 \end{cases}$$
  即已解组使用标准GRPO，全负组使用加权NSR，driver越大（越不确定/越有潜力）的负样本获得越强惩罚。
- **Fisher信息理论支撑**：Lemma 4.1证明token-level Fisher trace $\text{Tr}(\mathbf{F}_t) = 1 - \sum_k p_{t,k}^2$ 与熵正相关，高熵状态意味着更高的政策可塑性，均匀惩罚低熵失败样本浪费优化预算。
- **Algorithm 1流程**：采样后判断组内是否存在正样本，若存在则走标准GRPO更新；否则计算各负样本driver、归一化后分配优势并更新策略。

## 实验与结果
- **训练设置**：基于Qwen2.5-Math-7B，在DAPO-17k数据集上用verl框架训练，每步$G=8$ rollout，temperature=1.0，context=4096，学习率$10^{-6}$，无熵/KL惩罚。
- **验证集**：AMC23、AIME2025、AIME24、Math-Beyond，采用固定rollout预算（$k=1,2,...,256$）评估Pass@k，同时报告TTFP。
- **核心结果**：
  - 相比GRPO，NAPO02在AIME24 Pass@256提升10.0个百分点（0.4000→0.5000），在AIME2025提升6.6个百分点（0.4667→0.5333）。
  - 在AIME24 Pass@8提升27.28个百分点，表现最为突出。
  - 相比Uniform(-0.1)对照组，NAPO在各数据集Pass@256均保持正收益，证明组内差异化驱动的重要性。
- **效率分析**：NAPO仅增加0.410s/step（约0.03%端到端时间），无额外rollout或forward pass。
- **跨模型泛化**：在Llama-3.1-8B-Instruct上测试，NAPO02在AIME24 Pass@256达37.78% vs GRPO的33.33%，在Math-Beyond上亦有提升；但在AIME2025上略低于GRPO，提示驱动窗口需按模型校准。
- **代码领域扩展**：在LiveCodeBench v5/v6上用Qwen3-4B-Instruct验证，$\alpha=0.5$时N→P组的msk值比N→N组高0.0276（p<0.05），初步证实熵信号在代码任务中同样存在。

## 相关工作脉络
- **GRPO及变体**（Shao et al., 2024; Guo et al., 2025）：标准分组相对策略优化，是全负组梯度枯竭的根源，本文在此基础上引入负样本加权。
- **负样本增强方法**：Zhu et al. (2025) 提出NSR分解；Yu et al. (2025) 的N-GRPO；Nan et al. (2025) 的负增强GRPO——这些方法对负样本施加均匀或粗粒度加权，本文的细粒度driver排序与之本质不同。
- **熵引导探索**：Wang et al. (2025) 关注高熵少数token的作用；Cheng et al. (2026) 从熵角度研究探索；Le et al. (2026) 用entropy-guided advantage shaping处理零方差组——本文的独特之处在于定位"早期"而非整体熵，并与N→P事件直接关联。
- **动态rollout分配**：Fang et al. (2026) 提出动态采样分配；Zheng et al. (2025b) 的Act-Only-When-It-Pays选择性rollout——本文不修改采样预算，而是在给定rollout内优化梯度分配。
- **事件级诊断**：本文引入TTFP和N→P事件视角填补了现有RLVR工作缺乏转变动力学的空白。

## 局限性与未来方向
- **领域泛化受限**：当前分析聚焦于数学推理的二元可验证奖励场景，在部分反馈、噪声反馈或主观反馈（如代码审查、开放域写作）中，熵信号的判别力可能减弱。
- **超参依赖基础设置**：最佳早期分数$\alpha$在不同模型（7B vs 8B Llama）和数据集上存在差异（数学任务$\alpha=0.2$最优，代码任务$\alpha=0.5$更优），需要针对性校准。
- **极低成功率场景失效风险**：当模型几乎无法接近正确解时，早期熵模式可能不足以区分"潜在可解"与"真正死路"。
- **未来方向**：扩展至更长horizon推理、非二进制奖励环境、开发更鲁棒的driver定义，以及跨模型/跨数据集的自动校准策略。

## 研究启发与可借鉴点
- **事件级分析框架可迁移**：TTFP和N→P事件指标可作为通用工具用于诊断各类RL训练中的"突破时刻"，不限于数学推理。
- **熵作为Fisher信息的代理**：Lemma 4.1建立的关系可为其他需要估计策略可塑性的场景提供参考，无需计算完整Fisher矩阵。
- **组内差异化加权思路**：NAPO的"已有正样本用标准GRPO、全负组用分级NSR"的设计思想可迁移到任何基于分组相对优势的方法中。
- **低成本增效验证**：NAPO仅增加0.03%训练时间即可获得显著收益，证明在existing pipeline中嵌入轻量级诊断模块是高效的研究策略。
- **跨领域初步验证**：代码任务的延伸实验为Entropy-Driven Negative Learning的普适性提供了证据，可启发在其他符号推理任务中的应用探索。

## 关键术语表
- **RLVR**：Reinforcement Learning with Verifiable Rewards，利用可自动验证的奖励信号进行强化学习训练，适用于数学/代码等精确输出任务。
- **N→P Transition**：Negative-to-Positive Transition，指一个原本全负（零正确）的prompt在训练过程中首次出现正样本的瞬间转变。
- **TTFP**：Time-to-First-Positive，首个正样本出现时的训练步数，用于量化硬提示的"首次突破"速度。
- **Early-Branching Driver**：$d_{\text{early-ppl}}$，早期分支驱动指标，衡量响应前$\alpha$比例的perplexity占整体的比例，捕捉"早期不确定性集中"的模式。
- **NAPO**：Negative-Augmented Policy Optimization，本文提出的GRPO集成算法，对全负组内负样本按driver分数分级赋予负优势。
- **Fisher Information Trace**：Fisher信息矩阵的迹，衡量参数更新对策略分布的改变程度，与token级熵正相关。
- **PSR/NSR Decomposition**：Positive/Negative Sample Reward分解，将RLVR目标拆分为对正样本的强化和对负样本的抑制两部分。
- **Group-Relative Advantage**：组内相对优势，GRPO中通过组内均值归一化得到的advantage，全负组时退化为零。

## 可复现要素
- **数据集**：DAPO-17k（训练），AMC23、AIME2025、AIME24、Math-Beyond（验证）；代码扩展使用LiveCodeBench v5/v6。论文未明确声明公开状态，但训练代码已开源：https://github.com/jameszhu-pixel-time/Z2O。
- **模型**：Qwen2.5-Math-7B（主实验）、Llama-3.1-8B-Instruct（泛化实验）、Qwen3-4B-Instruct（代码扩展）。
- **关键超参**：temperature=1.0，max context=4096，每步rollout数G=8，学习率$1\times10^{-6}$，early-fraction $\alpha=0.2$（数学）/0.5（代码），KL coefficient=0.0，entropy coefficient=0.0，PPO minibatch size=256，top_k=-1，top_p=1.0。
- **硬件**：单节点8× NVIDIA H100 GPU。
- **训练步数**：Qwen2.5-Math-7B报告step 200结果，Llama-3.1-8B-Instruct报告step 100结果。
