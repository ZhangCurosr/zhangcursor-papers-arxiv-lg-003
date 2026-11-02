---
title: "Uncertainty-Aware-Federated-Learning-for-Joint-Communication"
source: https://arxiv.org/pdf/2610.21969v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:50:46"
field: "无线通信与边缘计算的联合资源管理"
keywords: ["Federated Learning", "Uncertainty Injection", "Joint Communication and Computation", "Extended Reality", "Robust Resource Allocation", "Deep Learning for Wireless"]
innovations: ["提出在DNN输出后同时注入信道与计算误差样本的联合不确定性注入机制，隐式学习双重不确定性分布", "构建γ分位数鲁棒延迟最小化框架，实现通信与计算功率/资源的联合优化", "设计可扩展的联邦式DNN方案，在保持近集中式性能的同时大幅降低训练数据交换开销"]
benchmarks: ["Non-robust FL/Cen", "Non-robust FP", "Uniform FP", "SAA/WMMSE", "FedProx", "Local Learning"]
---

# 论文速读：Uncertainty-Aware-Federated-Learning-for-Joint-Communication

## 一句话总结
本文提出了一种**联合不确定性注入的联邦学习框架**，用于多基站蜂窝XR网络中**通信与计算资源的联合优化**，通过在被训练DNN的输出后注入采样误差，隐式学习信道与计算双重不确定性分布，实现γ分位数意义上的最坏情况延迟最小化。

## 研究问题与动机
1. **XR对超低延迟、高带宽和高算力提出严苛要求**，但传统资源管理方案假设信道和计算状态完美已知，不切实际。
2. **现有工作通常单独处理通信不确定性或计算不确定性**，缺乏对两者**联合不确定性传播**的系统性建模与优化。
3. **集中式深度学习方案面临隐私泄露、前传/回传瓶颈和扩展性限制**，亟需分布式联邦学习替代方案。
4. **经典优化方法（如SAA/WMMSE）计算复杂度高、不适合实时决策**，数据驱动的DNN方案具有实时推理优势。

## 核心贡献（创新点）
1. **提出联合不确定性注入（Joint Uncertainty Injection）机制**：在DNN输出后同时注入信道估计误差与计算需求误差样本，使网络隐式学习双重不确定性分布；与仅注入单一不确定性或完全忽略不确定性的方法有本质区别。
2. **构建鲁棒γ分位数延迟最小化框架**：以P(t_max > t_max^γ) ≤ γ作为可靠性约束，将分布未知、非凸的最坏情况延迟优化转化为可微的无监督学习目标；区别于传统WMMSE/SAA方法需显式假设分布形式。
3. **设计集中式与联邦式两套DNN求解方案并对比分析**：集中式将全局信息汇聚至CP；联邦式每BS本地训练+CP聚合，并通过均值跨基站干扰交换弥补分布式决策的信息缺失。
4. **系统验证与边界分析**：在多种不确定性水平、网络密度和异质性设置下全面评估，并实测Edge设备推理延迟证明工程可行性。

## 方法详解
### 系统模型
- 网络由 $B$ 个基站（BS，每个配备 $L_b$ 天线与边缘计算能力）和 $K$ 个XR终端组成；每个设备归属一个BS，形成互斥设备集群。
- 真实信道为 $\boldsymbol{h}_{b,l,k} = \hat{\boldsymbol{h}}_{b,l,k} + \boldsymbol{h}_{b,l,k}^{\text{err}}$，其中 $\boldsymbol{h}^{\text{err}} \sim \mathcal{CN}(0, \sigma_h^2)$；真实计算强度为 $\omega_k = \hat{\omega}_k + \omega_k^{\text{err}}$，其中 $\omega^{\text{err}} \sim \mathcal{N}(0, \sigma_\omega^2)$。

### DNN 架构
- 输入：估计信道 $\hat{h}$、估计计算需求 $\hat{\omega}$（联邦式额外输入均值跨基站干扰 $\bar{\varphi}$）。
- 结构：多层全连接隐藏层（ReLU），输出层维度 $2K+B$（含发射功率、计算功率、计算资源分配系数）。
- 映射约束：使用 softmax 函数将输出投影到满足功率预算 $(p^{\text{tx}}+p^{\text{co}} \le P_b^{\max})$ 和容量上限 $F_b^{\max}$ 的可行域。

### 联合不确定性注入（训练时）
1. DNN推理得到资源分配向量 $\boldsymbol{x}$。
2. 抽取 $N$ 组不确定性样本 $\{\boldsymbol{h}_n^{\text{err}}, \omega_n^{\text{err}}\}_{n=1}^N$。
3. 对每组样本计算设备速率 $r_{k,n}$ 与单设备延迟 $t_{k,n}$，得到 $N$ 个最坏延迟 $\{t_{\max,n}\}_{n=1}^N$。
4. 排序后取第 $(\gamma \cdot N)$ 高的值作为鲁棒目标 $\hat{t}_{\max}^{\gamma}$。
5. 对 $\hat{t}_{\max}^{\gamma}$ 反向传播梯度更新DNN参数（无监督学习）。

### 联邦学习流程（Algorithm 2）
- 每BS本地用梯度 $\partial \hat{t}_{b,\max}^{\gamma}/\partial \Theta_b$ 更新模型。
- 各BS上报参数 $\Theta_b$ 与诱导干扰 $\varphi_k^{\text{inter}}$ 至CP。
- CP做参数平均聚合 $\Theta^{\text{global}} = \frac{1}{B}\sum_b \Theta_b$，汇总干扰后广播给所有BS。

## 实验与结果
### 数据集与仿真环境
- 3个BS，每个6根天线；默认每BS服务 $K_b=5$ 个XR设备（可扩展至25）。
- 信道：3GHz载频、Rayleigh衰落、MMSE估计；计算强度服从Gamma分布。
- 超参：$N=1000$ 样本、$\gamma=0.05$、10层400神经元隐藏层（集中式）/7层200神经元（联邦式）、500轮训练。

### 评估基线
Non-robust FL / Non-robust Cen / Non-robust FP / Uniform FP / SAA/WMMSE / Comm UI FL / Comp UI FL / FedProx / Local。

### 关键结果数字
- **不确定性鲁棒性**：在 $\sigma_h^2=0.008$ 时，Robust FL相对Non-robust FL降低 **18 ms** 鲁棒延迟；Robust Cen降低 **12 ms**。
- **联合注入优于单一注入**：在低/高不同不确定性主导场景下，Robust FL始终达到最低延迟。
- **扩展性**：$K=20$ 时，Robust FL相对Uniform FP减少 **69 ms**，Robust Cen减少 **88 ms**；$K=25$ 时集中式方法因计算复杂度过高无法收敛（图7阴影区）。
- **网络密度**：稀疏/中等/密集场景下，Robust FL延迟分别为31.4/65.4/112.9 ms，推理时间稳定在约 **4-5 ms**；而非ML基线需 **数千 ms**。
- **数据交换**：8 BS×6设备时，集中式训练需高达 **495 TB**，联邦式仅需 **数百 MB~GB**。
- **边缘部署**：NVIDIA Jetson Orin Nano上Robust Cen推理 **9.578 ms**，Robust FL推理 **4.587 ms**，满足毫秒级实时性。
- **最强结果**：集中式鲁棒方案（Robust Cen）为延迟下限；联邦式方案（Robust FL）与之差距随设备数增大，在高密度时拉大到 **19 ms**（K=20），但计算可行且数据开销降低约 **5个数量级**。

## 相关工作脉络
1. **Cui & Yu (2023) [7] — Uncertainty Injection (UI)**：首次提出DNN输出后注入信道误差以提升鲁棒性；本文扩展至**通信+计算双重不确定性**并推广到**多BS联邦架构**。
2. **Reifert et al. (2024) [1] — 会议版**：单BS、集中式、考虑信道与计算不确定性的联合资源优化；本文将其扩展到**多BS蜂窝XR网络**，引入跨基站干扰建模与联邦式分布式实现。
3. **Li et al. (2023) [10] — 计算不确定性**：处理边缘计算中的处理需求不确定性，面向单机/单场景；本文在**网络联合优化**层面同时建模两类不确定性。
4. **Mo & Xu (2021) [22] — 联邦学习边缘协同**：以**能效**为目标的FL通信计算联合优化；本文聚焦**延迟鲁棒性**与**不确定性注入**机制，目标函数与鲁棒性考量完全不同。
5. **FedProx (Li et al. 2020) [51]**：通过近端项缓解数据异质；论文将其作为非鲁棒基准，本文方案在相同异质设置下仍保持鲁棒优势。

## 局限性与未来方向
1. **固定设备-BS关联假设**：未考虑动态接入/切换，扩展至移动场景与变规模输入需另行设计（如 permutation-invariant 或图网络）。
2. **跨基站干扰采用均值近似**：训练时交换均值干扰 $\bar{\varphi}_k$ 忽略了瞬时干扰波动，可能对极端场景估计偏差。
3. **训练开销较高**：增大 $N$ 可使鲁棒性提升但线性增加训练时间（如 $N$ 从500→1000，训练时间翻倍）。
4. **非I.I.D.异质性下的联邦性能衰减**：高异质时鲁棒FL退化接近鲁棒Local，需结合 FedProx 或个性化联邦策略。
5. **鲁棒性-通信开销权衡未深入**：文中提及量化/压缩技术可降开销，但未系统探讨其对鲁棒延迟的影响。

## 研究启发与可借鉴点
1. **不确定性注入可迁移至其他资源管理任务**：该"在可微映射后注入采样噪声"的思想可直接复用于功率控制、波束赋形、调度等问题。
2. **联合双重不确定性建模思路**：通信+计算（或任意两类耦合不确定性）同时在DNN后注入，为多源不确定系统优化提供通用范式。
3. **无监督学习结合鲁棒分位数目标**：无需标签、直接以物理约束/目标作为梯度信号；可推广到无标注的无线/计算联合优化。
4. **实验设计的多维度消融**：不确定性类型（信道/计算/联合）、网络密度、异质性水平、可靠性参数 $\gamma$ 逐一扫描，提供完整性能画像——值得在后续工作中沿用。
5. **边缘硬件实测验证推理延迟**：在Jetson等真实平台上测量DNN推理耗时，而非仅给出理论下界，增强了实际部署说服力。

## 关键术语表
**Uncertainty Injection（不确定性注入）**：在DNN输出后显式注入采样得到的误差样本，使网络隐式学习不确定性分布以提升鲁棒性的训练技术。

**Joint Uncertainty Injection（联合不确定性注入）**：同时向DNN输出注入信道估计误差与计算需求误差的多重样本，以处理两类耦合不确定性。

**γ-Quantile Robust Delay（γ分位数鲁棒延迟）**：满足 P(t_max > t_max^γ) ≤ γ 的延迟阈值，表示只有不超过γ比例的 realizeation 会超过该延迟。

**Federated Learning (FL)**：各客户端在本地训练模型并仅将参数上传至聚合服务器，实现去中心化知识共享而不交换原始数据。

**Inter-BS Interference（跨基站干扰）**：由非本BS服务于其他设备的发射引起的同频干扰，联邦式方案通过均值交换进行近似管理。

**Unsupervised Learning（无监督学习）**：此处指不依赖人工标签，直接以任务目标（鲁棒延迟）作为损失函数进行反向传播的DNN训练方式。

**Fractional Programming (FP)**：将分数形式的目标或约束通过Dinkelbach型迭代转化为可解形式的经典数值优化方法，本文用于部分baseline求解。

**Sample Average Approximation (SAA)**：用经验平均代替期望/分位数约束的鲁棒优化近似方法，本文用于构造SAA/WMMSE基线。

## 可复现要素
- **数据集**：仿真生成（Rayleigh衰落信道、Gamma分布计算强度），未引用公开数据集；论文提供了仿真参数（表I）与代码细节。
- **代码/权重**：论文未明确说明开源仓库；提到使用 PyTorch 实现。
- **关键超参**：$N=1000$、$\gamma=0.05$、集中式 $\eta_h=10, n_h=400$；联邦式 $\eta_h=7, n_h=200$；训练轮次 $E=500$，每轮 $M=50$ minibatch、$\nu_m=1000$ 样本。
- **仿真环境**：3BS×6天线、载频3 GHz、带宽10 MHz、$P^{\max}=34$ dBm、$F^{\max}=4.6\times10^9$ cycles/s。
