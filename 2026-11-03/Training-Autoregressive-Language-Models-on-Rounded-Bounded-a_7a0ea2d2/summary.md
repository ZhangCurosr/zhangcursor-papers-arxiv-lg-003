---
title: "Training-Autoregressive-Language-Models-on-Rounded-Bounded-a"
source: https://arxiv.org/pdf/2611.02461v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 03:11:47"
field: "数值预测与分布建模"
keywords: ["autoregressive language models", "numerical prediction", "censored labels", "boundary marginalization", "CRPS training", "interval likelihood", "digit decoder"]
innovations: ["提出边界边缘化算法，通过至多两条teacher-forced路径精确计算数值区间似然，复杂度O(b·log_b(M))", "单一似然目标统一精确标签、四舍五入区间和上下界删失观测的训练", "相同计算框架支持精确分位数查询和无偏CRPS梯度估计"]
benchmarks: ["ChEMBL 37 hERG IC50", "KBSS (Code-Regression)", "Controlled Gaussian mixture recovery"]
---

# 论文速读：Training-Autoregressive-Language-Models-on-Rounded-Bounded-a

## 一句话总结
本文提出**边界边缘化（boundary marginalization）**方法，首次在自回归语言模型上通过其已有数字token精确计算四舍五入、有界和删失数值观测的似然，无需辅助输出头或样本插补，统一处理精确标签、区间与上下界三类监督信号。

## 研究问题与动机
1. **数值观测多为区间/删失形式**：测量四舍五入（如1.2表示[1.15, 1.25)）、实验室测试只报告下界（如IC50 > c）、程序超时只知完成时间>c，这类报告在真实数据中占比很高（约1/3）。
2. **标准交叉熵训练无法直接使用区间标签**：需要单一目标数值，现有实践要么丢弃不完整报告，要么将其替换为代表值（如区间中点），导致信息损失或目标分布偏差。
3. **直接枚举不可行**：自回归LM通过数字token定义数值分布，一个区间可包含数千个数值，逐一枚举计算总概率和梯度代价极高。
4. **已有数字级损失方法仅关注点估计**：如Number Token Loss、NTIL等改进了token级监督，但未处理无精确目标数值的区间/删失场景。

## 核心贡献（创新点）
1. **边界边缘化算法**：首个通过现有数字token精确计算四舍五入、有界、删失观测似然的方法，无需辅助输出头、插补目标或采样 rollout。
   - 区别：现有方法要么丢弃不完整标签，要么用单一代表值训练CE；本文直接对兼容数值集合分配概率。

2. **单次前向传递完成区间似然与梯度计算**：利用固定宽度数字表示的有序性，每个区间只需沿至多两条边界路径（长度log_b(M)）提取条件分布，以O(b·log_b(M))标量运算完成，不枚举区间内部。
   - 区别：与传统离散分布建模需O(M)不同，复杂度仅依赖于数字位数而非支持集大小，支持百万级网格值。

3. **单一目标统一多类型监督**：同一个似然目标同时处理精确标签（退化为序列CE）、四舍五入区间和上下界；边界路径外的子树被求和消去，不接收梯度。
   - 区别：区别于分别设计不同损失处理不同标签类型的方法，本文提供统一框架。

4. **直接支持CRPS训练与精确分布查询**：相同计算同时支持精确分位数查询（CDF二分或逐位贪心）和无偏CRPS梯度估计，无需采样模型输出。
   - 区别：相比RL-based CRPS优化（需采样rollout），本文通过teacher-forcing直接微分，梯度无偏且方差更低。

## 方法详解

### 核心原理
- **固定宽度数字表示**：输出支持设为有序网格V = {v_0, ..., v_{M-1}}，M = b^d，每个网格索引k用d位b进制数表示（最高有效位在前），保持数字序与数值序一致。

- **条件数字分布**：给定输入x和前缀s，下一数字a的条件概率为：
  p_θ(a|x,s) = softmax(ℓ_θ(x,s)) restricted to b digit tokens

  完整数值的概率为各digit条件概率的乘积（式3）。

- **观测事件化**：任意观测转换为兼容网格索引区间A = [l, u] ∩ Z，表1定义了精确值、闭区间、四舍五入、严格下界、严格上界五种情况的转换公式。

- **边界边缘化算法**（Section 4）：
  - 尾概率（单边界）：沿阈值z的路径，按首次偏离位置分组，CDF和生存函数各用一条teacher-forced路径计算（式7、8）。
  - 区间概率（双边界）：将区间[l,u]分解为若干互斥的前缀子树组（图1所示的4类），每组的概率即为到达该前缀的质量（因后缀总概率为1）。
  - 用log-sum-exp在 log 域稳定求和，避免浮点相减误差（式11）。

- **梯度分析**（Proposition 2）：事件损失对前缀s处digit logit的梯度正比于该prefix被命中的后验概率w_s乘以(模型分布-条件目标分布)，当子树完全在区间内或完全在外时梯度为零。因此**只有跨越边界的prefix节点接收非零局部梯度**。

- **Teacher-forced概率提取**（Section 5.1）：将prompt重复并后接边界路径的数字序列，在一次batched forward pass中读取每个位置的因果logit，选取b个digit token列做log-softmax，获得完整的条件分布。Dropout关闭但自动微分开启。

- **CRPS训练**（Section 5.4）：对精确标签，通过独立采样少量阈值（stratified sampling），用边界路径计算CDF，构造无偏CRPS梯度估计（式17），比普通score-function estimator方差更低。

## 实验与结果

### 数据集与设置
- **Qwen2.5-1.5B**为主骨干，**Qwen3-4B-Base**为跨骨干验证；LoRA rank-16、lr=1e-4、四数字grid。
- 基线：Complete-case CE（仅用精确标签）、Imputed-label CE（中点插补）、Event-aware mixture（5-component Gaussian）、Event-aware histogram（256-bin）。

### 实验1：受控分布恢复
- 构造输入-dependent双高斯混合截断分布，精确标签概率ρ ∈ {0, 0.1, 0.2, 0.5, 1}，其余为四舍五入区间或上下界。
- 以ISE（CDF积分误差）评估恢复质量。
- **关键结果**：
  - ρ=0.1时：Event likelihood ISE=6.993 vs Complete-case CE 15.923（**降低56.1%**）vs Imputed-label CE 29.156
  - ρ=0（无精确标签）时：Event likelihood ISE=6.801 vs Imputed-label CE 34.121（**降低80.1%**）
  - Event likelihood在所有ρ∈{0,0.1,0.2,0.5}下均优于两个event-aware heads

### 实验2：天然删失分子测量（hERG IC50）
- ChEMBL 37中14,850个pIC50测量，29.8%训练/36.2%测试为bounds。
- **关键结果**（Table 2）：
  - Full NLL：**4.675**（最优），vs Complete-case CE 5.083（提升0.408 nats）
  - Censored NLL：0.823 vs Complete-case CE 2.076（**降低60.3%**）
  - Equality NLL：6.857 vs Complete-case CE 6.786（仅高0.071，损失极小）

### 实验3：代码延迟（KBSS）
- 45.2%训练样本被timeout删失，所有测试延迟精确已知。
- **关键结果**（Table 2）：
  - Physical CRPS：**8.262×10⁻³ ms**（最优），vs Complete-case CE 8.870（↓6.8%）
  - Log CRPS：**0.0878**（最优），vs Complete-case CE 0.0980（↓10.5%）
  - Budget Brier：0.1281 vs Complete-case CE 0.1550
- **跨骨干验证**（Qwen3-4B，Table 3）：CRPS物理↓5.3%、对数↓11.5%，结果复现。

### 实验4：精确标签上的CRPS vs CE对比
- 纯精确标签设置下，Sequence CE达最低CRPS（0.2165）；threshold-sampled CRPS（m=2）仅高1.5%。

### 实验5：精度可扩展性
- 支持集从10²扩展到10⁶，update时间和峰值内存几乎不变（256 prompt tokens：0.872–0.888s；1024 tokens：2.10–2.14s）。
- d=6时dense per-value output需35.8/47.5 GiB，Event likelihood节省68.4%/43.2%内存。

## 相关工作脉络
1. **Numerical decoders（LLMTime, Decoding-based regression）**：将数字序列解释为层次分布；本文在此基础上进一步支持区间/删失监督，而不仅限于精确数值。
2. **Digit-level numerical losses（NTIL, DEL, SMMD等）**：改进token级监督或点估计；本文直接对完整数值事件赋概率，无需设计新的点构造。
3. **Regression-aware fine-tuning (RAFT)**：通过枚举候选值对平方损失微分；本文避免枚举，用边界路径精确计算，支持更大网格。
4. **Coarsened/censored likelihood**：传统统计中处理删失数据；本文首次将其引入LM的数值token空间训练，对比event-aware heads证明digit decoder更优。
5. **Tree-structured numerical distributions（Hierarchical softmax, PATH）**：构建层次输出分布；本文直接fine-tune预训练numeral decoder，不构建额外leaf分布。
6. **Proper scoring rules for LMs（CRPS in RL setting）**：已有RL方法用采样rollout优化CRPS；本文用teacher-forcing直接微分，方差更低且无需采样。

## 局限性与未来方向
1. **有限表示与共享参数耦合**：声明的网格固定了支持范围和分辨率；共享LM参数可能使不同前缀子树间的预测产生耦合。
2. **报告机制假设**：似然解释依赖条件可忽略报告（conditionally ignorable reporting）；非可忽略报告需额外建模报告机制。
3. **部分尾部分布不可识别**：当ρ=0时，仅靠四舍五入区间和阈值报告，尾部某些网格概率无法被唯一识别。
4. **经验范围有限**：仅使用Qwen2.5和Qwen3两个骨干；不同seed的变异性未充分量化；跨网格深度的预测精度未测试。
5. **未来方向**：扩展到非均匀网格（如对数网格已支持）、处理非可忽略报告、更大规模真实基准验证。

## 研究启发与可借鉴点
1. **Teacher-forcing + 边界路径设计**：将问题从"枚举所有兼容值"转化为"沿边界路径提取条件分布"，是一种通用思路——对有序离散分布上的区间概率计算，可利用结构化前缀树将O(M)降至O(log M)。
2. **log-domain + log-sum-exp稳定性**：在概率求和中全程使用log域和LSE，避免了浮点下溢/相减误差，值得在类似分布计算任务中借鉴。
3. **精确数字监督与CRPS训练的统一框架**：同一套边界计算同时服务事件似然、分位数查询和CRPS，减少了工程复杂度，可作为多目标数值预测的统一基础设施。
4. **利用数字token顺序结构而非绕过它**：本文不引入额外输出头，直接fine-tune已有digit decoder；说明充分利用模型已有表示结构比添加辅助组件更有效。
5. **与团队方向结合机会**：任何涉及数值预测且标签存在不确定性的场景（如药物发现、材料科学、金融预测、代码性能估计）均可采用此框架，将缺失/区间标签直接融入训练。

## 关键术语表
- **Boundary marginalization（边界边缘化）**：通过沿区间端点边界路径提取条件数字分布，精确计算数值区间似然的方法，无需枚举区间内部。
- **Constrained numeral distribution（约束数字分布）**：LM通过softmax限制在b个digit token上定义的数值分布，d位数字乘积归一化为M=b^d个网格值。
- **Event likelihood（事件似然）**：将观测解释为兼容网格值集合A，训练目标为P(Y∈A|x)的负对数，统一处理精确/区间/删失标签。
- **Log prefix mass（对数前缀质量）**：数值前缀z_{<j}的联合概率的对数，即H_j(z) = Σ_{i<j} log p(z_i|x, z_{<i})。
- **Conditionally ignorable reporting（条件可忽略报告）**：给定x和真实值k，观测到事件A的概率仅依赖k∈A的指示函数，不依赖其他细节。
- **Continuous ranked probability score（CRPS）**：衡量预测CDF与观测阶跃函数之间差异的严格正当评分，对数值距离敏感。
- **Stratified threshold sampling（分层阈值采样）**：将阈值区间划分为固定子区间，在每个子区间内按权重比例采样，用于无偏估计CRPS。
- **Event-aware head（事件感知输出头）**：保持相同骨干和事件似然，但用Gaussian mixture或histogram替代digit decoder的基线方法。

## 可复现要素
- **数据集**：Controlled synthetic data（自行构造，论文提供详细生成公式）；ChEMBL 37 hERG（公开数据库，但论文提供了split manifest）；KBSS from Code-Regression（公开，revision 2410c301）。
- **代码/权重**：论文未明确提供开源代码链接（截至论文发表时）。
- **关键超参**：LoRA rank=16, α=32；lr=1e-4；AdamW (β₁=0.9, β₂=0.999, weight decay=0)；gradient norm clip=1.0；effective batch size=32；four-digit grid (b=10, d=4)；对于log grid用log-spaced支持。
- **随机种子**：main experiments用seeds 0,1,2；补充实验扩展至5 seeds。
- **训练时长**：Incomplete-label studies 3 epochs；exact-label score study 9 epochs。
