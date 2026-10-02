---
title: "The-Row-Normalization-Puzzle-in-Muon"
source: https://arxiv.org/pdf/2609.39114v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:40:56"
field: "大规模语言模型优化器理论分析"
keywords: ["Muon", "NorMuon", "row normalization", "matrix optimizer", "convergence analysis", "operator-norm geometry", "LLM pretraining"]
innovations: ["建立了NorMuon在算子范数几何下的维度依赖下界Ω(mLε⁻²)，揭示行归一化带来的额外收敛代价", "给出了匹配的上界并推广到随机设定与近似极分解情形", "通过合成实验与LLM预训练的实验对比分离了理论代价与实践收益"]
benchmarks: ["Nemotron-CLIMB LLM预训练（286M/1.38B/6.44B）", "CIFAR-10 CIFARNET图像分类", "合成不平衡rows最小二乘问题"]
---

# 论文速读：The-Row-Normalization-Puzzle-in-Muon

## 一句话总结
本文从理论上证明：Muon 优化器中的行归一化（NorMuon）在最坏情况下会引入关于矩阵行数 $m$ 的额外收敛复杂度因子，但其 LLM 预训练实验仍显著优于 Muon，揭示了理论下界与实际性能之间的"谜题"。

## 研究问题与动机
- **Muon 已广泛采用但理论理解不足**：Muon 及其变体 NorMuon 已在大规模 LLM 预训练中取得显著实践收益（如使用于 Nanochat、Megatron Core、Datadog Toto 2.0），但 NorMuon 的 worst-case 收敛保证缺乏理论刻画。
- **行归一化的核心疑问**：行归一化是否通过结合近似极分解与指数移动平均动量，带来可证明的收敛增益？还是仅在实践中有效而理论代价更高？
- **现有方法缺口**：Dewulf et al. [2026] 指出 Muon 中低杠杆率神经元更新过小导致"神经元死亡"问题，NorMuon 通过 EMA 平衡行更新幅度缓解此问题，但对其 worst-case 迭代复杂度的系统性分析缺失。
- **理论与实践的张力**：类似梯度下降中 step-size scheduling 在实践中有效但无法达到算法无关下界 $\Omega(T^{-2})$（Ye & Liu, 2026），作者希望为矩阵优化器建立类似的理论洞察。

## 核心贡献（创新点）
1. **建立了 NorMuon 在算子范数几何下的维度依赖下界**：证明在确定性设定下，NorMuon 的最坏情况迭代复杂度为 $\Omega(m L \epsilon^{-2})$，比 Muon 的 $O(L \epsilon^{-2})$ 多出一个因子 $m$，且该下界对任意固定动量参数和任意确定性自适应步长均成立。
2. **给出了匹配的上界分析并推广到随机设定**：确定性上界为 $O(m L \epsilon^{-2})$，随机上界为 $O(m L \epsilon^{-2} + m^2 L \sigma^2 \epsilon^{-4})$，且分析允许近似极分解（PolarApproximation）。
3. **合成实验与 LLM 预训练的实验对比揭示了理论与实践的差距**：在合成 least-squares 问题中 NorMuon 随 $m$ 增大明显慢于 Muon，但在 LLM 预训练（286M/1.38B/6.44B 模型）中 NorMuon  consistently 优于 Muon。
4. **证明了增强的神经元平衡本身不足以保证更快的最坏情况收敛**：揭示了理论代价与实际收益之间的本质区别，补充了 Dewulf et al. [2026] 的发现。

## 方法详解
- **分析框架**：在 operator-norm 几何下分析 NorMuon，利用 Frobenius 内积下算子范数与核范数的对偶性：$\max_{\|Y\|_{\text{op}} \leq 1} \langle X, Y \rangle = \|X\|_{\text{nuc}}$，Muon 的更新等价于在算子范数 ball 上最小化线性近似。
- **关键构造（下界）**：构造一个 rank-one 梯度 $H_\star = \frac{(m,1,\ldots,1)^\top}{\sqrt{m^2+m-1}}(e_1^{(n)})^\top$，其行幅度不等——归一化后虽平衡了行，但削弱了与梯度的对齐（alignment），即使使用精确极分解和任意固定动量参数，该方向性畸变依然存在。
- **下降引理**：核心不等式为 $F(X_{t+1}) - F(X_t) \leq -\eta_t \langle \nabla F(X_t), D_{t+1} \rangle + \frac{L \eta_t^2}{2} \|D_{t+1}\|_{\text{op}}^2$，其中 $\|D_{t+1}\|_{\text{op}} \leq 0.2\sqrt{mn}$。
- **关键引理（Lemma B.2）**：$\langle M_{t+1}, D_{t+1} \rangle \geq 0.2 \kappa \sqrt{mn} \|M_{t+1}\|_{\text{nuc}}$，其中对齐因子 $\kappa = \frac{(1-\delta)(1-\ell \min\{m,n\})\sqrt{1-\beta_2}}{(1+\delta)\sqrt{m}} = \Theta(m^{-1/2})$，显式体现了行归一化引入的维度惩罚。
- **近似极分解处理**：允许使用 PolarExpress 等近似方法（Assumption 2.3），只要输入奇异值 $\geq \ell$ 则输出奇异值在 $[1-\delta, 1+\delta]$ 内。

## 实验与结果
- **合成实验**：不平衡 rows 的最小二乘问题，$m \in \{8, 32, 128, 512, 2048\}$，$n=8$，1000 次迭代。结果（Table 1）：随着 $m$ 从 8 增至 2048，Muon 的 loss reduction 保持稳定（约 5.70–5.72），而 NorMuon 从 5.70 持续下降至 3.15，验证了维度惩罚。
- **图像分类（CIFAR-10 + CIFARNET，2M 参数）**：Muon（测试精度 93.79%）优于 AdamW（92.33%），NorMuon 列归一化达到最高 93.87%，行归一化为 93.83%，略低于列归一化（Table 2）。
- **LLM 预训练（Nemotron-CLIMB 数据集，nanochat 代码库）**：
  - 286M/1.38B/6.44B 三档模型，NorMuon 在整个训练过程中持续低于 Muon 的验证 loss（Figure 1）。
  - **1.38B 模型（Table 3/9）**：Muon avg accuracy = 56.3%，NorMuon = 57.1%（+0.8%），test loss 从 2.2914 降至 2.2810。
  - **6.44B 模型**：Muon avg = 60.4%，NorMuon = 60.9%，test loss 从 2.2803 降至 2.2674。
- **消融实验（1.38B 模型，compute-optimal）**：
  - **步骤尺寸敏感性**：NorMuon 在每个步长下均优于 Muon（Figure 2a）。
  - **参数组消融**：仅对 MLP input/output projections 应用行归一化即可恢复大部分增益，QKVO 归一化几乎无贡献（Table 4）。
  - **与其他优化器对比**：Aurora 与 NorMuon 效果接近，均优于 Muon+，MuonEq-R 与 Muon 相当（Table 4）。
  - **PolarExpress 迭代次数**：3/5/7 次均显示 NorMuon 优于 Muon，差距在 3 次时最大（Table 11）。
  - **第二矩因子 $\beta_2$**：0/0.9/0.95/0.99 均效果相近（Table 12）。
  - **Weight decay**：0.01/0.02/0.05 下 NorMuon 始终优于 Muon（Table 13）。

## 相关工作脉络
- **Muon [Jordan et al., 2024]**：基于梯度动量的极分解进行矩阵更新，本文理论下界的核心对比基线，Muon 在相同设定下达到维度无关的 $O(L\epsilon^{-2})$。
- **NorMuon [Li et al., 2026]**：在 Muon 基础上引入行方向二阶矩归一化，本文重点分析的对象，实践性能好但理论代价高。
- **Muon+ [Zhang et al., 2026b]**：正交化后再施加无状态的行/列归一化，消融实验中作为对比基线之一。
- **MuonEq [Chang et al., 2026]**：在正交化之前均衡行/列，实验中等同于 Muon 性能。
- **Aurora [Dewulf et al., 2026]**：基于 leverage-aware 处理高瘦矩阵中的低杠杆率神经元，与 NorMuon 在 LLM 预训练中表现接近。
- **Shen et al. [2026] / Zhang & Lin [2026]**：前者证明 Muon 的维度无关确定性上界 $O(\Delta L \epsilon^{-2})$，后者处理随机设定下的最优上界，构成本文 Muon 参考的基准。

## 局限性与未来方向
- **随机设定下界缺失**：目前仅有随机设定的上界 $O(m L \epsilon^{-2} + m^2 L \sigma^2 \epsilon^{-4})$，尚未建立匹配的随机下界，无法确定该上界是否 tight。
- **理论代价与实际收益的断裂仍未完全解释**：为什么 LLM 预训练中 NorMuon 能克服理论上的维度惩罚，具体依赖哪些训练动态特征（如数据分布、网络结构）尚不清楚。
- **仅分析了行归一化**：未涉及列归一化或其他归一化方向的系统性理论对比（尽管实验中列归一化在 CIFAR-10 上略优）。
- **未来方向**：识别 LLM 训练中使行归一化有益的结构性特征，发展能捕捉这些实际收益的理论框架。

## 研究启发与可借鉴点
1. **维度依赖下界的构造技巧值得复用**：通过 rank-one 梯度配合不相等行幅度构造"难以归一化"的最坏实例，可作为后续分析其他矩阵优化器归一化变体的参考范式。
2. **对齐因子 $\kappa$ 的分析思路**：将近似极分解误差、动量误差和行归一化效应统一纳入对齐因子分析，是分析 matrix-aware optimizers 收敛性的有效框架。
3. **理论与实践的分离实验设计**：先通过精心构造的合成问题验证理论下界，再用真实 LLM 预训练展示实践优势，这种"分离论证"策略可直接迁移到对其他优化器改进方案的理论-实践一致性评估中。
4. **消融定位贡献来源**：通过逐组消融（QKVO vs MLP）定位 NorMuon 增益主要来自 MLP 矩阵，这一实验设计可用于指导后续针对性改进（如仅在特定层应用归一化）。
5. **与团队方向的潜在结合**：若团队关注矩阵优化器设计，可借鉴其 operator-norm 几何分析框架，或考虑研究"什么条件下归一化不会引入维度惩罚"这一开放问题。

## 关键术语表
- **Muon**：一种利用梯度动量极分解（polar factor）更新权重矩阵的优化器，区别于逐坐标缩放方法， exploit 神经网络参数的矩阵结构。
- **NorMuon**：在 Muon 基础上引入行方向指数移动平均二阶矩归一化的变体，用于平衡不同神经元间的更新幅度。
- **算子范数几何（Operator-norm geometry）**：以算子范数（spectral norm）为度量约束的优化几何，Muon 的更新方向恰好是此几何下的最速下降方向。
- **极分解（Polar decomposition）**：矩阵 $Z = U\Sigma V^\top$ 的极因子为 $UV^\top$，将梯度动量映射为正交/近正交更新方向。
- **PolarExpress**：使用多项式迭代（Newton-Schulz）近似极分解的高效方法，可在不改变奇异向量的情况下调整奇异值。
- **下界构造（Lower bound construction）**：通过构造 rank-one 梯度且行幅度不等的光滑目标函数，证明 NorMuon 在最坏情况下至少需要 $\Omega(m\epsilon^{-2})$ 次迭代。
- **Leakage neuron / 低杠杆率神经元**：Muon 中因行更新幅度持久过小而近乎"死亡"的神经元，NorMuon 通过行归一化缓解此问题。

## 可复现要素
- **数据集**：CIFAR-10（公开）、Nemotron-CLIMB（Diao et al., 2025，公开）、合成 least-squares 问题（论文给出具体构造公式，可复现）。
- **代码/权重**：LLM 实验使用 nanochat 代码库（Karpathy, 2025，GitHub 开源）；合成实验和图像分类实验的代码论文未明确声明开源。
- **关键超参**：Muon/NorMuon 使用 Nesterov 型动量 $\beta_1 = 0.95$，NorMuon 的 $\beta_2 = 0.95$；5 次 PolarExpress 迭代；decoupled weight decay = 0.02；步长 $\eta = 0.03$（286M）/ $\eta = 0.01$（1.38B 和 6.44B）；batch size 分别为 524,288/1,048,576/1,048,576 tokens。
