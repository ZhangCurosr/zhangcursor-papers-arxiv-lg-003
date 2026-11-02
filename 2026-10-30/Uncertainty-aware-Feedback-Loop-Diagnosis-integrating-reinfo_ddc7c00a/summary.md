---
title: "Uncertainty-aware-Feedback-Loop-Diagnosis-integrating-reinfo"
source: https://arxiv.org/pdf/2610.40421v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 19:05:56"
field: "半导体制造缺陷检测"
keywords: ["晶圆缺陷检测", "共形预测", "强化学习", "自适应数据采集", "Causal Transformer", "多标签分类"]
innovations: ["以共形预测两阶段校准替代辅助置信度估计网络实现Ada停止规则", "将CP从单样本扩展至多标签逐标签预测集，并设计适配序列决策的停时校准流程"]
benchmarks: ["MixedWM38"]
---

# 论文速读：Uncertainty-aware-Feedback-Loop-Diagnosis-integrating-reinfo

## 一句话总结
提出了UFLD（不确定性感知反馈循环诊断）框架，结合因果Transformer分类器与强化学习（PPO）自适应选区扫描，并引入共形预测（Conformal Prediction）作为停止规则，替代辅助置信度估计网络，实现在较少扫描面积下实现高精度晶圆缺陷分类（Hybrid设置达98.38%，扫描量降至61.4%）。

## 研究问题与动机
1. 半导体高速量产环境下，全表面扫描获取高分辨率晶圆图成本高昂且耗时，现有CNN/ViT方法均依赖完整扫描，不适合实时高通量检测。
2. 已有自适应数据采集（ADA）方法（如SupConWI-RL）依赖额外辅助置信度估计深度学习模型，存在参数量大、置信度不可靠等问题。
3. 共形预测虽可提供统计保证，但其在序列决策中的直接应用受限于"交换性"假设——不同wafer在不同时间点停止会导致校准与推理不对齐。
4. 亟需一种兼具参数效率、可解释停止规则和较高准确率的自适应晶圆缺陷分类方案。

## 核心贡献（创新点）
1. **因果Transformer + RL自适应选区的反馈循环架构**：用Causal Transformer处理时序patch序列，以PPO策略网络动态选择下一扫描区域，实现输入序列级优化。
2. **共形预测驱动的停止规则**：提出两阶段校准算法（Algorithm 4），将非共形分数按类别分别统计分位数，替代SupConWI-RL中的黑盒置信度估计网络。
3. **多标签预测下的类级别停止判据**：将CP扩展至多标签场景，要求所有C个标签的预测集基数均为1才停止，实现逐标签确定性认证。
4. **双阶段校准对齐推理分布**：先在固定T步收集分数计算预阈值，再用预阈值驱动策略在另一子集上按真实停止时间收集分数进行重校准，使校准分布贴近推理分布。

## 方法详解
**整体框架**：UFLD包含三组件：①预训练的Causal Transformer分类器；②基于PPO的RL patch选择策略；③基于CP的两阶段校准停止规则。

**Causal Transformer分类器**：
- 输入为t个patch拼接的序列，每patch经MLP嵌入后叠加空间位置编码 $E_{i_s}$ 与可学习时序步嵌入 $E_s$。
- 经L层Transformer Block，取最后一层LayerNorm输出经MLP得到logit $\hat{\mathbf{o}}_t$，Sigmoid得概率 $\mathbf{p}_t$，阈值0.5得到多标签预测 $\hat{\mathbf{y}}_t$。
- 损失函数：BCE Loss + λ·Supervised Contrastive Loss（SC Loss，使用两种数据增强生成正样本对）。

**RL Patch Selector**：
- 状态 $\mathbf{s}_t = (\mathbf{f}_t, \mathbf{m}_t)$，$\mathbf{m}_t$为未选patch的二值mask。
- 动作采样：$a_{t+1} \sim \text{Categorical}(\pi_{t+1} \odot \mathbf{m}_t)$，推理时取argmax。
- 奖励：正确预测得 $1 - c_{\text{RL}}$，错误得 $-c_{\text{RL}}$（$c_{\text{RL}}=0.005$），不依赖置信度网络评分。

**共形预测停止规则**：
- 非共形分数：$S_{0,t,j}=p_{t,j}$（负类），$S_{1,t,j}=1-p_{t,j}$（正类）。
- 按标签分别计算负/正两类阈值 $\hat{q}_{0,j}, \hat{q}_{1,j}$（基于校准集分位数）。
- 停止条件：对所有标签j，$E_{t,j}=1$成立（即正负两类分数中恰有一类低于阈值），当且仅当全部标签均满足时终止扫描。
- **两阶段校准**（Algorithm 4）：
  1. CALIBRATE_A：在子集 $\mathcal{D}_A$ 上用固定T步收集分数，计算预阈值 $\hat{q}^A_0, \hat{q}^A_1$。
  2. CALIBRATE_B：在子集 $\mathcal{D}_B$ 上用预阈值驱动策略，在真实停止时刻收集分数，重算阈值 $\hat{q}_0, \hat{q}_1$，使校准分布与推理对齐。

## 实验与结果
**数据集**：MixedWM38（~38,000张晶圆图，每张52×52像素，划分为T=169个4×4 patch），含1–4种缺陷及正常片，存在类别不平衡。

**训练集划分**（Table 1）：
- $\mathcal{D}_1$（20,528）训练Transformer；$\mathcal{D}_2$（2,281）校准CP阈值；$\mathcal{D}_3$（7,603）训练RL策略；$\mathcal{D}_4$（7,603）测试。

**最强结果**（Table 4）：
- **Hybrid Single UFLD-large**：准确率 **98.38%**，扫描面积 **61.4%**，参数 **28.5M**。
- Non-hybrid Single UFLD-large：准确率 **97.88%**，扫描 **61.4%**，参数仅 **7.2M**。
- 相对基线：**单模型UFLD比Ensemble SupConWI-RL高1.5 p.p.**，参数仅为其约一半；Hybrid模式下与Ensemble SupConWI-RL（98.45%）相近但参数更少（28.5M vs 37.3M）。
- 相对全扫描ResNet-34（98.79%，100%扫描，21.3M参数），UFLD以61.4%扫描面积获得接近结果，扫描时间减少约**40%**。

**灵敏度分析**（Table 5）：α从0.05增至0.1，扫描比例单调上升，准确率同步提升，覆盖度略降；三种尺寸Transformer行为一致，证明停止规则对架构鲁棒。

**消融实验**（Table 6）：Causal Transformer基础（随机序列，61.5%扫描）准确率仅92.80%；加入ADA（+4.19 p.p.）或CP停止规则（+4.49 p.p.，但扫描增至70.5%）各有增益；两者结合（UFLD）达97.88%，扫描维持61.4%，各指标最优。

## 相关工作脉络
1. **SupConWI-RL [16]**：利用RNN+辅助置信度估计网络的ADA方法，UFLD以Transformer+CP停止替代，参数量减半，准确率提升1.5 p.p.（单模型对比）。
2. **Sequential Wafer Map Inspection via RL [15]**：早期RNN反馈循环工作，本文在分类器架构（Transformer）和停止规则（CP）上全面改进。
3. **Conformal Prediction (Vovk et al.)**：本论文首次将该框架引入晶圆缺陷自适应扫描场景，解决序列决策中的分布对齐问题。
4. **ResNet-34 (标准CNN基线)**：全扫描方法的上界参考（98.79%），UFLD以61.4%扫描逼近其性能。
5. **Adaptive Sampling in MRI (Bakker et al.)、ASMR (Yen et al.)**：RL ADA在医疗成像领域的先行工作，本文将其迁移至半导体晶圆检测并改进停止机制。
6. **WaferDSR-RAG [33]**：基于RAG+LLM的晶圆缺陷语义推理，属于完全不同的范式（全扫描+知识检索），本文聚焦低扫描量的轻量分类。

## 局限性与未来方向
1. **理论保证受限**：两阶段校准后，实际推理使用的阈值与校准时的预阈值不完全一致，严格有限样本覆盖保证不再精确成立（论文Section 3末尾明确说明）。
2. **超参数α需人工调优**：不同产线对准确率和扫描时间的偏好不同，目前依赖手动搜索，缺乏自动化选择策略。
3. **仅适用于离散patch网格**：当前将晶圆图划分为固定网格patch，对非规则缺陷边界或亚像素级检测场景适应性有限。
4. **未来方向**：建立更严格的理论覆盖保证；扩展至缺陷分割（segmentation）和表面重建等更复杂任务；探索其他工业检测场景的迁移应用。

## 研究启发与可借鉴点
1. **两阶段校准策略**：将"固定终点校准→驱动策略→真实停止时刻重校准"的二次校准思路，可复用于其他序列决策场景下的共形预测部署，解决校准分布漂移问题。
2. **类级别停止判据设计**：多标签场景下逐标签独立构建预测集并判定停止，避免了全标签聚合带来的粒度损失，对多标签ADA任务有参考价值。
3. **用参数-efficient的CP替换辅助置信度网络**：证明在ADA任务中，经典统计方法（共形预测）可替代深度学习置信度估计网络，在保持准确性的同时显著降低参数量和部署复杂度。
4. **时序步嵌入（Step Embedding）**：为Causal Transformer引入可学习的时序步嵌入，使其能适应不同长度的输入序列，对变长序列建模任务具有通用借鉴意义。
5. **RL奖励设计的简洁性**：仅用预测正确性设计奖励（而非置信度分数），避免"教策略用不可靠信号"的链式误差放大，设计原则清晰可迁移。

## 关键术语表
**Causal Transformer**：一种带有因果掩码的Transformer架构，按时序逐步处理patch序列，支持变长输入和增量推理。
**Conformal Prediction（共形预测）**：一种数据驱动、分布无关的预测区间构建方法，通过非共形分数校准提供边际覆盖保证。
**Adaptive Data Acquisition（ADA）**：根据当前观测动态选择下一个最 informative 的测量位置，以减少总采集量。
**Non-conformity Score（非共形分数）**：衡量样本与校准集中同类样本偏离程度的标量，用于计算预测集阈值。
**Miscoverage Rate（α）**：用户设定的允许预测集不包含真实标签的最大概率，α越小预测集越保守。
**Prediction Set（预测集）**：共形预测输出的候选标签集合，保证真实标签以至少1-α概率落入其中。
**Supervised Contrastive Loss（监督对比损失）**：利用标签信息拉近同类样本特征、推远异类样本特征的对比学习损失项。
**Hybrid Model（混合模型）**：当CP停止规则无法在T步内满足时，以CNN全扫描作为兜底分类的融合策略。

## 可复现要素
- **数据集**：MixedWM38，约38,000张晶圆图（52×52像素，169个4×4 patch），论文未明确说明是否公开在独立平台，但实验部分提到代码已在GitHub公开。
- **代码**：已开源于 https://github.com/dekhovich/UFLD
- **权重**：论文未明确说明权重是否公开
- **关键超参**：PPO迭代次数100；学习率0.001（Transformer和RL均使用）；$c_{\text{RL}}=0.005$；Transformer预训练240 epoch；α（miscoverage rate）在0.05–0.1范围内可调；λ（SC Loss权重）论文未明确给出具体数值。
