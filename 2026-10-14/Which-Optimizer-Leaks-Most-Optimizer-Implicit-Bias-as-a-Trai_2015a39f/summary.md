---
title: "Which-Optimizer-Leaks-Most-Optimizer-Implicit-Bias-as-a-Trai"
source: https://arxiv.org/pdf/2610.18294v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-01 22:39:00"
field: "机器学习安全与隐私"
keywords: ["training data privacy", "optimizer implicit bias", "model reconstruction attack", "KKT stationarity", "data leakage", "Muon optimizer", "Adam implicit bias"]
innovations: ["提出基于优化器margin范数的通用重建攻击框架，修正朴素扩展的两种失败模式", "定义权重特异性(weight-specificity)并证明Adam的l_infinity目标特异性为0且一阶不可识别", "建立特异性-可达性二轴理论解释Muon>SGD>Adam的泄露排序"]
benchmarks: ["MNIST binary", "CIFAR-10 binary", "CIFAR-10 10-class"]
---

# 论文速读：Which-Optimizer-Leaks-Most-Optimizer-Implicit-Bias-as-a-Trai

## 一句话总结
本文系统研究了不同优化器（SGD、Adam、Muon）的隐式偏置如何影响训练数据隐私泄露程度，提出了一套基于优化器相关KKT平稳性条件的模型参数反演攻击框架，并揭示了"Muon最易泄露 > SGD次之 > Adam最抗泄露"的安全排序规律。

## 研究问题与动机
- **已有工作局限**：Haim等人（2022）证明仅凭训练好的网络参数即可重建训练数据，但所有此类重建攻击均假设优化器为梯度下降（GD），未考虑不同优化器的隐式偏置几何对攻击效果的影响。
- **核心问题**：不同优化器诱导不同的最大间隔几何结构（margin geometry），其对应的平稳性条件（stationarity condition）是否携带不同的训练数据"指纹"？选择优化器能否作为调控隐私泄露的有效手段？
- **动机**：即使架构和数据完全相同，优化器不同也可能导致截然不同的泄露风险，现有模型发布的安全性评估忽视了优化器这一维度。
- **理论缺口**：Gronich和Vardi（2026）分析了Adam和Muon的隐式偏置，但指出KKT型重建是否可推广至这些优化器仍是开放问题。

## 核心贡献（创新点）
1. **提出优化器参数化的重建攻击框架**：将Haim等人的$\ell_2$平稳性条件推广到任意margin范数（包括Muon的光谱范数和Adam的$\ell_\infty$范数），并修正了朴素扩展的两个失败模式（重复奇异值导致的目标基依赖问题、跨块尺度不匹配导致的loss坍塌问题）。
2. **引入"权重特异性（weight-specificity）"理论概念**：量化不同优化器在平稳性目标锥中编码的连续权重信息量；证明Adam的$\ell_\infty$目标在任意最大坐标重数下特异性为0，且在非退化KKT点处平稳性约束方程数不超过在间隔上的样本数，无法一阶唯一识别训练集。
3. **建立"特异性+可达性"二轴解释框架**：结合权重特异性（收敛后目标锥编码多少连续信息）和可达性（收敛到极限方向的速度），理论上预测并实验验证泄露排序Muon ≻ SGD ≻ Adam，其中Muon优势依赖训练预算而在Adam端稳定存在。

## 方法详解
- **攻击基础**：对于齐次网络+指数/逻辑损失，不同优化器在方向上收敛到对应margin问题的KKT点：SGD→$\ell_2$ margin，Muon→max-spectral-norm margin，Adam→$\ell_\infty$ margin。攻击者利用该KKT点的平稳性条件 $\mathbf{r}(z) \in \partial(\frac{1}{2}\|\pmb{\theta}\|^2)$ 反推输入数据。
- **一般范数平稳性**：对于非可微范数，用次微分代替梯度，由引理3.1转化为两个标量不等式：$\|\mathbf{r}\|_\star \leq \|\pmb{\theta}\|$ 且 $\langle \mathbf{r}, \pmb{\theta}\rangle \geq \|\pmb{\theta}\|^2$，两者在可行解处同时取等号。
- **消除缩放自由度**：证明乘子缩放 $\lambda \mapsto c\lambda$ 是规范对称（gauge），不携带数据信息；利用逐层Euler恒等式 $\langle \mathbf{r}_{W_k}, W_k \rangle = d_k \Gamma(z)$ 证明无需跨块耦合项。
- **对齐损失（核心攻击目标）**：$L_{\text{align}} = \sum_k \left(1 - \frac{\langle \mathbf{r}_{W_k}, W_k \rangle}{\|\mathbf{r}_{W_k}\|_{(k),\star} \|W_k\|_{(k)}}\right)^2$，对三种优化器统一适用，仅在块范数选择上不同：SGD用Frobenius/欧氏对偶，Muon用spectral/nuclear对偶，Adam用$\ell_\infty$/$\ell_1$对偶。
- **Muon的具体形式**：$L_{\text{align}}^{\text{Muon}} = \sum_k \left(1 - \frac{\langle \mathbf{r}_{W_k}, W_k \rangle}{\|\mathbf{r}_{W_k}\|_{\text{nuc}} \|W_k\|_{\text{sp}}}\right)^2$，每个块独立计算，不需知道per-block权重$c_k$。
- **Adam的具体形式**：全局单一项 $(1 - \langle \mathbf{r}, \pmb{\theta}\rangle / (\|\mathbf{r}\|_1 \|\pmb{\theta}\|_\infty))^2$，要求残差仅在最大绝对值坐标上有支撑且符号匹配。
- **权重特异性分析**：在KKT方向（所有块活跃、$\rho_k=1$）处，$S_{\ell_2}=p-1$，$S_{\text{spectral}}=\sum_k(m_k+n_k-2)$，$S_{\ell_\infty}=0$。$\ell_\infty$目标的子微分是一个单纯形，只编码最大坐标的位置和符号（离散信息），不编码任何连续权重幅度信息。
- **一阶不可识别定理**：在非退化KKT点假设下，Adam的平稳性系统最多有$n_I$个法方程，但有$n_I(d+1)$个未知数，当$n_Id \geq 2$时不满足正则一阶可识别性（Theorem A.10）。

## 实验与结果
- **数据集**：Binary MNIST（500样本）、Binary CIFAR-10（500样本）、10-class CIFAR-10（500样本）；额外进行样本量消融（n=50/100/500/1000）和架构消融（不同hidden宽度）。
- **模型**：全连接ReLU网络 $d\text{-}1000\text{-}1000\text{-}C$，仅第一层有bias；Muon对hidden-to-hidden矩阵应用5步Newton–Schulz迭代，对输入/输出层使用SGD。
- **基线**：Haim等人（2022）的$\ell_2$重建攻击；比较SGD、Adam、Muon三种优化器训练的同架构模型。
- **评估指标**：SSIM@30（最佳30对重建的SSIM均值）、PSNR、$\ell_1$、$\ell_2$距离；SSIM>0.4视为可识别。
- **主要结果**（无prior，n=500）：
  - **Muon最强泄露**：MNIST SSIM@30=0.431，Binary CIFAR-10 SSIM@30=0.551，10-class CIFAR-10 SSIM@30=0.534
  - **SGD次之**：对应值为0.346、0.229、0.383；在Binary CIFAR-10上Muon是SGD的两倍以上
  - **Adam最抗泄露**：对应值为0.249、0.138、0.207，全部低于0.4可识别阈值
  - Muon仅需约$10^4$次迭代即达到训练损失$O(10^{-7})$，而SGD需$10^6$次迭代——Muon以两个数量级更少的训练成本实现更强的重建
- **Image prior消融**：加入range/TV/$\ell_1$等正则后，Muon在所有18种配置下均保持最优；Adam在加smoothness prior后略有改善（0.138→0.238）但仍远低于可识别阈值；作者认为此提升来自prior使通用候选更接近训练分布而非真正识别出具体样本。
- **样本量效应**：当SGD训练至充分收敛（loss<$10^{-8}$）时，在n=100处SGD的SSIM@30=0.623接近Muon的0.698，表明Muon与SGD的差距部分源于训练预算不对称，符合可达性预测；但Adam在所有样本量和架构下均最差，差距稳健。
- **Margin依赖性**：可识别重建仅出现在margin最小（在间隔上）的样本附近，Muon产生的可识别样本远多于SGD，Adam几乎无。

## 相关工作脉络
- **Haim et al. (2022)**：首次证明仅凭分类器参数即可重建大量训练图像，通过反转$\ell_2$-margin KKT平稳性条件实现；本文将其推广到非$\ell_2$几何，并系统比较不同优化器。
- **Buzaglo et al. (2023)**：将Haim方法扩展到多分类、weight decay和一般loss；本文沿用其攻击范式但替换平稳性项。
- **Gronich & Vardi (2026)**：分析了平滑齐次网络中Adam和Muon的隐式偏置（$\ell_\infty$ vs max-spectral-norm margin），本文在此基础上推导了对应两种几何的重建攻击并证明Adam的不可识别性。
- **Zhang et al. (2024)**：独立分析了Adam在可分数据上的$\ell_\infty$隐式偏置；本文从KKT方程角度补充了其在数据重建场景下的弱点分析。
- **Jordan et al. (2024)**：提出Muon优化器；本文揭示其 spectral implicit bias 导致更强的训练数据泄露风险。
- **Refael et al. (2026b)**：研究了image prior在重建攻击中的主导作用；本文与之呼应，分离出optimizer-dependent的stationarity项贡献。

## 局限性与未来方向
- 未提供形式化的隐私保证（如差分隐私级别），结论基于stationarity-only攻击框架，不覆盖带query访问或强生成prior的攻击者。
- 分析针对理想化的Adam（稳定性常数可忽略）和MuON几何，实际实现中的超参偏离可能影响结论。
- 主要针对全连接ReLU网络，卷积/Transformer等更复杂架构的推广有待研究。
- 未来方向：将几何隐式偏置分析扩展到AdaRankGrad、SUMO、Lorenza等新兴低秩/正交化优化器，理解自适应秩降维和子空间正交化如何调控数据泄露（论文Section 6已明确提出）。

## 研究启发与可借鉴点
1. **"特异性（specificity）"分析视角可迁移**：将优化器的隐式偏置几何分解为"编码多少连续权重信息"这一维度，为理解不同算法的信息留存特性提供了简洁的理论工具，可应用于其他涉及模型参数泄露的研究场景。
2. **无prior重建实验设计值得借鉴**：本文在无image prior条件下仍能有效攻击Muon/SGD训练模型，证明了stationarity条件本身携带足够强指纹；这对评估模型发布的"最坏情况"隐私风险具有参考价值。
3. **可达性-特异性二轴框架**：将收敛速度（可达性）与极限几何信息量（特异性）分开分析，解释了为何理论上$\ell_2$特异性最高（SGD）却实践中不如Muon泄露严重——这种分层解释方法可用于分析其他优化器的隐私特性。
4. **与团队方向的结合机会**：若团队研究大模型训练安全/隐私，可将此框架应用于LLM训练中的新型优化器（如论文提到的AdaRankGrad、SUMO等），评估其在低秩微调场景下的数据泄露风险。
5. **修复朴素扩展失败的技术**：通过per-block对齐损失替代单目标向量的做法，解决了重复奇异值和跨块尺度不匹配两个失败模式，该技巧对设计其他非欧几里得几何下的反演攻击有直接参考价值。

## 关键术语表
- **Implicit Bias（隐式偏置）**：优化器在收敛时倾向于选择特定几何意义的解（如最大间隔解），即使目标函数未显式编码该偏好。
- **Weight-Specificity（权重特异性）**： optimizer依赖的平稳性目标锥随权重连续变化的自由度数量，即目标锥能编码多少关于最终权重的连续信息。
- **Stationarity Condition（平稳性条件）**：KKT条件的核心部分，表达网络权重为训练样本Signed梯度的非负组合，是重建攻击的方程来源。
- **Max-Spectral-Norm Margin（最大谱范数间隔）**：Muon优化器诱导的margin几何，最大化各权重块谱范数的最大值所定义的间隔。
- **Reachability（可达性）**：优化器的参数方向趋近其极限KKT方向的速率，影响在有限训练预算下攻击的实际效果。
- **KKT Point（Karush–Kuhn–Tucker点）**：带约束优化问题的最优性条件点，此处指对应margin问题的KKT方向，梯度下降类优化器在齐次网络+指数损失下收敛至此。
- **Subdifferential（次微分）**：凸函数在非可微点的广义梯度集合；此处用于刻画$\ell_\infty$和spectral范数在不可微点的平稳性条件。
- **Gauge Invariance（规范不变性）**：目标函数在乘子缩放变换下的不变性，本文利用此性质消除了无信息自由度。

## 可复现要素
- **数据集**：MNIST、CIFAR-10（均为公开数据集）
- **代码/权重**：论文声明"Code and reproduction scripts will be made publicly available"（论文未明确给出链接，需关注后续发布）
- **关键超参**：训练采用learning rate $10^{-2}$，Muon对hidden层应用5步Newton–Schulz；攻击优化使用SGD（momentum=0.9），迭代100,000步；重建learning rate在对数均匀$[10^{-4}, 10^{-2}]$中搜索，初始化scale $\sigma_x$在$[10^{-3}, 1]$中对数均匀搜索，修改的ReLU参数$\alpha$在$[10, 500]$中均匀采样；候选数$m=2n$，balanced labels。
- **硬件**：单张NVIDIA GeForce GTX 1080 Ti GPU，PyTorch实现
