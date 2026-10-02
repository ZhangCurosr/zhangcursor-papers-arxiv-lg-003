---
title: "WHEN-MODELS-DON-T-MANIPULATE-MANIFOLDS-THE-GEOMETRY-OF-A-COM"
source: https://arxiv.org/pdf/2609.37680v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 07:54:07"
field: "机制可解释性"
keywords: ["mechanistic interpretability", "number comparison", "linear representation", "manifold hypothesis", "causal intervention", "DAS", "LLM geometry"]
innovations: ["发现尽管数字表征存在弯曲流形，模型在比较任务中因果依赖线性方向进行加性混合计算", "给出两数和三数比较的完整算法（copy head+残差加和+局部到全局MLP组合），并以IIA/PR严格因果验证"]
benchmarks: ["两位数字对比较（100%准确率）", "三数字序列比较", "多operand（2-20）最大值查找"]
---

# 论文速读：WHEN-MODELS-DON-T-MANIPULATE-MANIFOLDS-THE-GEOMETRY-OF-A-COM

## 一句话总结
本文在 Qwen2.5-7B-Instruct 上精确刻画了数字比较任务的计算几何，发现尽管数值表征存在非线性流形结构，模型仍通过线性方向上的加性混合（线性表征）完成比较：先用注意力头将两个数拷贝到同一残差流位置并相加，再由 MLP 神经元在局部区间进行分块比较，最后合并得到全局比较结果；该算法可扩展至三个数字的序列比较。

## 研究问题与动机
- **核心问题**：给定某个具体计算任务，模型究竟如何利用表征几何来执行该计算——是操纵非线性流形，还是利用其中更简单的线性结构？
- **线性表征假说 vs 流形假说**：前者认为有序概念编码在一维子空间（线性），后者近期大量实证表明概念（如数字、日期）实际生活在低维弯曲流形上（如数字螺旋）。二者如何共存尚不清楚。
- **已有工作的不足**：Hanna et al. (2023) 虽详细描述了 GPT-2 的比较电路，但未能因果性地证明表征几何在比较中的作用；El-Shangiti et al. (2025) 发现了线性子空间但未分析比较机制；Yuchi et al. (2026) 仅对比行为准确率与分类器性能。现有工作聚焦电路发现或 probing 精度，**缺少对因果表征几何的系统刻画**。
- **动机**：比较是决策过程中对有序概念执行的核心操作，揭示模型如何利用表征几何实现比较，有助于理解"哪些几何结构在实际计算中被使用"这一机制可解释性问题。

## 核心贡献（创新点）
1. **调和线性表征与流形假说**：尽管 PCA 显示数字表征存在于弯曲流形（PC1-PC3），但模型在比较任务中因果依赖的是单个线性方向——通过 Interchange Intervention Accuracy (IIA) 严格验证，这是本文最核心的理论贡献。
2. **给出了基于线性表征的数字比较算法**：模型先将两个数沿各自方向编码后加性混合到共享二维平面，再经 MLP 神经元分两阶段（局部比较→全局组合）完成比较，算法形式化于 Algorithm 1。
3. **将比较算法扩展到三数序列**：每个时间步 $t$ 计算一个二进制 flag $\alpha_t = \mathbb{I}(y_t = \max(y_1,\dots,y_t))$ 并存储在单一方向 $\mathbf{d}_t$ 上，最终 token 处用单个注意力头组合所有 flag 读出 arg max 位置（Algorithm 2）。
4. **因果验证方法的系统展示**：结合 DAS（Distributed Alignment Search）、IIA、Position Recovery（PR）、冻结神经元实验、receptive field 热力图等多重因果手段，逐步骤验证算法各阶段。

## 方法详解
- **模型与任务设置**：使用 Qwen2.5-7B-Instruct（28层，$d_{model}=3584$，SwiGLU MLP，宽度 18944），在 freeze 参数下以 raw completion 形式执行 $\max(y_1,\dots,y_K)$ 任务，提供 one-shot 示例固定输出格式。数字按 token 逐位输入。
- **因果测量工具**：采用 Interchange Intervention（公式4）——将 clean prompt 的激活 patch 到 corrupt prompt 对应位置；衡量指标为 IIA（首个生成 token 的 argmax 准确率）和 Position Recovery PR（公式2/5，归一化 logit difference）。
- **线性方向发现**：对 PCA 非因果的方向使用 DAS（Geiger et al., 2024，Adam 100步，lr=0.05）拟合因果方向 $\mathbf{u}, \mathbf{v}_1, \mathbf{v}_2$ 等；方向编码方式为 $\mathbf{v}_i f_i(y_i)$，其中 $f_i$ 为非线性函数（实测近似 $\log y$ 线性关系）。
- **两数比较算法（Algorithm 1）**：
  - Step 1（$t=1$）：将 $y_1$ 编码为 $\mathbf{x}_l^1 = \mathbf{u} f_1(y_1)$，存于 L13 残差流 $y_1$ 位置。
  - Step 2（$t=2$）：将 $y_2$ 编码为 $\mathbf{x}_l^2 = \mathbf{v}_2 f_2(y_2)$，同时**注意力头 H14（L14层）充当 copy head**，将 $y_1$ 信息从位置 $y_1$ 拷贝到 $y_2$ 位置并变换为 $\mathbf{v}_1 f_1(y_1)$。
  - Step 3：残差连接将二者相加，得到共享表示 $\mathbf{x}_{l+1}^2 = \mathbf{v}_1 f_1(y_1) + \mathbf{v}_2 f_2(y_2)$，张成二维平面 $\text{span}(\mathbf{v}_1, \mathbf{v}_2)$，$\mathbf{v}_1 \perp \mathbf{v}_2$（$\cos\approx -0.08$）。
  - Step 4：MLP 神经元在该平面上分**局部区域**做比较（$\text{C}_i = \mathbb{I}(y_1>y_2)\cdot\mathbb{I}(\mathbf{x}\in\mathcal{R}_i)$），由 SwiGLU 的非线性门控产生局部激活区域。
  - Step 5：后续 MLP 神经元将局部比较结果合并为全局比较器，输出编码 arg max 位置的单一方向。
- **三数比较算法（Algorithm 2）**：在 $y_3$ 位置构建三维空间 $\text{span}(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)$，MLP 神经元在其上进行局部比较，结果存储为 flag 方向 $\mathbf{d}_3$ 上的二元值 $\alpha_3$；$y_2$ 位置同步存储 $\alpha_2$；最终 token（L20）用注意力头组合 $\sum_t w_t \alpha_t \mathbf{d}_t$ 读出 arg max 位置。
- **神经元识别**：通过对 37888 个 MLP 神经元做 attribution patching 排序，取两数 case 下 top-20 交集得 12 个共享神经元（L14 和 L15 各 6 个），冻结这些神经元会显著降低 IIA。

## 实验与结果
- **数据集**：两数比较使用 2000 个两位有序对（1000对+对称反转），每位数不同首位；三数比较使用 1500 组三元组，400 个保留样本。
- **基线方法**：与前人工作对比（Hanna et al. 2023 GPT-2 比较电路、El-Shangiti et al. 2025 线性子空间、Gurnee et al. 2026 流形操纵计数任务），无数值比较的定量 baseline。
- **行为准确率**：两位数字对全部 8010 有序对准确率为 **100%**；多 operand（2–20个）测试显示模型在单个 forward pass 中可正确处理长序列比较（App. Fig.6）。
- **关键因果结果**：
  - L13 残差流中单一方向 $\mathbf{u}$（PC1-PC3 弯曲流形背景下）实现 **IIA=0.94**（两位数字，$y_1$ perturbed）。
  - 共享平面 $\text{span}(\mathbf{v}_1,\mathbf{v}_2)$ 的 rank-2 patch IIA 达 **0.81–0.948**；两个 rank-1 patch 叠加几乎等价于 rank-2 patch（Fig.3e），验证加性混合。
  - L15 残差流中比较方向实现 **IIA=1.000**（$y_1$ perturbed 情形）；冻结 12 个关键神经元显著降低 IIA（Fig.4e）。
  - 三数比较中 PR 分数在 flag 方向和答案子空间上均表现因果有效（Fig.5f）。
- **最强结果**：L15 比较方向在 $y_1$ perturbed case 下达到 **IIA=1.000（perfect）**，表明该方向几乎完全控制比较输出。

## 相关工作脉络
1. **Linear Representation Hypothesis**（Park et al. 2023; Elhage et al. 2021）：主张有序概念编码为一维子空间——本文证明该假设在特定计算中成立，尽管整体表征几何是弯曲的。
2. **Manifold Hypothesis 系列**（Kantamneni & Tegmark 2025 数字螺旋、Modell et al. 2025 日期流形、Gurnee et al. 2026 字符计数流形）：证明概念生活在弯曲低维流形上——本文与其并存但不矛盾：流形存在，但比较计算用的是其中的线性子结构。
3. **GPT-2 比较电路**（Hanna et al. 2023）：详细描述了比较电路但未能因果证明几何作用——本文用 IIA/DAS 补上因果证据。
4. **Gurnee et al. (2026) "When Models Manipulate Manifolds"**（字符计数任务）：展示模型确实操纵流形——本文以对比视角说明并非所有任务都如此，比较任务"不操纵流形"。
5. **Feucht et al. (2026)**（加法中的 Fourier 特征）：不同任务可能使用不同几何策略——本文与之呼应，说明任务类型决定使用何种几何。
6. **DAS（Geiger et al. 2024）**：用于拟合因果方向的工具，本文依赖 DAS 发现方向，同时也承认其可能引入的 shortcut 风险。

## 局限性与未来方向
- **任务范围局限**：仅针对有序概念（数字）的比较/最大值任务，不涵盖周期性概念（如星期、月份）或其他运算（加法等）。
- **DAS 潜在 shortcut**：依赖 DAS 拟合方向可能继承其已知挑战（Wu et al. 2024），尽管文中用神经元/注意力头分析做了部分缓解。
- **充分性而非必要性**：patch 实验证明了所指方向/神经元的充分性，但未排除模型存在冗余路径（hydra effect, McGrath et al. 2023）的可能。
- **未完全揭示读出头机制**：三数比较中 L20 组合 flag 的注意力头工作细节尚未完全解析，作者将其留作未来工作。
- **模型限制**：仅在一个模型（Qwen2.5-7B-Instruct）上验证，跨模型泛化性未检验。

## 研究启发与可借鉴点
1. **线性 vs 流形的设计范式**：对类似"有序概念+比较/决策"任务，可优先假设模型使用线性子结构而非完整流形，用 DAS+IIA 验证因果性，避免被高方差 PCA 方向误导。
2. **加性混合构建共享空间的方法**：用 copy head + 残差连接将不同时间步的表征相加到同一位置，是通用策略，可迁移至其他需要"比较两个序列元素"的任务分析。
3. **局部→全局的 MLP 组合模式**：SwiGLU 非线性产生的局部 receptive field 叠加为全局逻辑，是理解 LLM 中复杂决策的一个通用分析框架，值得在其他任务（如排序、区间判断）中复现。
4. **Position Recovery（PR）替代 IIA**：在 IIIA 信号较弱时（如三数比较的中间位置），PR 提供了更细粒度的因果度量，是一种有效的补充评估手段。
5. **与团队方向结合机会**：若团队研究其他有序概念（时间、等级、分数）的比较推理，可直接套用本论文的 DAS→IIA/PR→冻结神经元→receptive field 分析流程。

## 关键术语表
**Interchange Intervention (IIA)**：因果干预方法，将 clean prompt 的特定激活替换为 corrupt prompt 对应激活，衡量目标组件对输出的因果影响。

**Distributed Alignment Search (DAS)**：通过优化寻找与目标变量最对齐的因果方向的无监督方法，基于交叉熵损失训练投影矩阵。

**Position Recovery (PR)**：基于 logit difference 归一化的因果度量，衡量 patch 后模型输出倾向改变答案位置的程度，扩展自 Meng et al. (2022)。

**Manifold Hypothesis**：认为神经网络中高维激活中，特定概念的表征落在低维弯曲流形上的假设。

**Shared Representation Plane**：两数比较中，模型通过加性混合将两个数的方向张成的二维子空间，是后续比较操作的计算载体。

**Local-to-Global Comparison**：MLP 神经元先在局部输入区间做比较，再由后续层神经元组合为全局比较器的两层计算模式。

**Comparison Flag**：三数比较中每个时间步输出的二元信号 $\alpha_t$，标记 $y_t$ 是否为当前最大值，沿单一方向存储。

**Copy Head**：注意力头（本文 L14.H14），从第一个数字位置无条件拷贝信息到第二个数字位置，是实现加性混合的关键组件。

## 可复现要素
- **数据集**：两位数字对 2000 组（1000随机对+对称反转）；三数组 1500 组；400 个保留评估样本；base random seed=52；使用确定性采样器。**论文未声明公开**。
- **代码/权重**：使用开源模型 Qwen2.5-7B-Instruct；代码和实验细节见 Appendix A–C，**论文未提供独立代码仓库链接**，未声明 GitHub。
- **关键超参**：DAS 优化 Adam 100 步，lr=0.05，minibatch=16-32；ridge regression 惩罚项 13 个对数间距值 $[10^{-2}, 10^{4}]$；half-precision loss scale=100；初始化为 2 次稳定性检查。
- **环境**：NVIDIA A100 40GB GPU，bfloat16，所有参数 frozen。
