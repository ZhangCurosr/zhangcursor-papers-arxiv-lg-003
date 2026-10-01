---
title: "TANGO-Watermarking-Masked-Diffusion-Language-Models-in-Token"
source: https://arxiv.org/pdf/2609.35224v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:23:46"
---

# 论文速读：TANGO-Watermarking-Masked-Diffusion-Language-Models-in-Token

## 一句话总结
本文提出 TANGO，一种面向 masked-diffusion 语言模型的双词对水印方法：通过密钥将词表划分为等大小颜色类，在去噪循环中对已解掩位置的邻近词施加模运算校验和偏置，使水印信号嵌入 token 颜色配对中而非单 token。该方法打破了对左到右生成顺序的依赖，在保持生成质量的同时实现频率不可观测性，显著提升抗伪造能力。

## 研究问题与动机
- **生成顺序不固定**：Masked-diffusion 模型并行/无序解掩码，传统 context-hashed green list 依赖前序已生成 token，在目标位置尚未确定时无法计算哈希上下文。
- **固定绿列表暴露密钥**：Red–green list 无需上下文但全局偏好同一批 token，导致水文本 token 频率偏移；攻击者可从频率统计中恢复绿列表并伪造通过检测的文本。
- **Gumbel-max 采样信号微弱**：按置信度优先解掩时，高置信位置已几乎无不确定性，Gumbel 扰动难以改变最终 token，导致嵌入信号弱、检测 AUC 仅 0.815。
- **需求汇总**：需一种 prompt-free、order-agnostic、频率隐藏且对常见编辑（删词/同义替换/插入）具备鲁棒性的扩散 LLM 水印。

## 核心贡献（创新点）
- **Token-pair 二阶水印机制**：将校验和 $(\chi(t_i) + a_\delta \chi(t_{i-\delta})) \mod q = b$ 嵌入已解掩邻域对中，偏置随位置动态切换。与固定绿列表直接全局偏置相比，TANGO 使每个位置的偏好类独立变化，打破频率泄漏路径。
- **语义分位数着色（Semantic-quantile coloring）**：按密钥投影方向对词嵌入排序并均分桶，同义词倾向同色。相比纯哈希着色易被同义替换破坏、语义聚类着色假阳性畸高，该方法在维持名义 FPR 的同时提升编辑鲁棒性。
- **一阶不可伪造性理论证明**：在理想采样与等概率质量假设下，证明 $\mathbb{E}[\Delta f(v)] = 0$，即期望 token 频率分布与无水印时完全一致。与 red–green list 的确定性偏好相反，TANGO 的偏置在平均意义上相互抵消。
- **两端强制（Either-side enforcement）**：允许用左侧或右侧已解掩邻居补全校验和，将 enforced fraction $\kappa$ 从 0.61 提升至 0.74+。与单向强制相比，在不改变检测统计量的前提下提高实际覆盖比例。

## 方法详解
- **着色阶段**：密钥 $k$ 生成方向向量 $\mathbf{r}_k$，对词表中每个 token $v$ 计算投影得分 $\langle \mathbf{e}_v/\|\mathbf{e}_v\|, \mathbf{r}_k \rangle$，排序后均分为 $q$ 个颜色类 $\chi: \mathcal{V} \to \mathbb{Z}_q$（默认 $q=3$，类大小相差不超过 1）。
- **生成阶段**：在每一步去噪中，对每个仍为 mask 的位置 $i$，若其 tap $t_{i-\delta}$（默认 $\delta=2$）已解掩，则计算 favored class $s_i = (b - a_\delta \chi(t_{i-\delta})) \mod q$；对该类所有 token 的 logit 加偏置 $+\beta$（默认 $\beta=5$），再按模型原有置信度规则保留 top-$K$ 解掩。Either-side 机制允许 tap 位于右侧时同样施加偏置。
- **检测阶段**：仅需候选文本与密钥 $k$。用 $\chi$ 重染色后统计 $T$ 个有效配对中满足校验和的数量 $C$，计算标准化分数 $z = (C - T/q) / (\sqrt{T}\sigma_0)$，其中 $\sigma_0^2 = \frac{1}{q}(1-\frac{1}{q})$。当 $z > z_\star$ 时判定为水印文本，阈值在模型未水印文本上校准。
- **理论性质**：检测期望得分随配对数平方根增长 $\mathbb{E}[z] = \sqrt{T}\varepsilon/\sigma_0$；随机替换率 $\rho$ 下得分期望衰减为 $(1-\rho)^{|\mathcal{D}|+1}$ 倍（单 tap 时为 $1-\rho$）。

## 实验与结果
- **设置**：LLaDA-8B-Instruct 与 Dream-v0-Instruct-7B；C4 prompts 生成 128 token，temperature=1.0，LLaDA 使用 CFG=2.0；对比基线包括 red–green list（green fraction=0.25, bias=5）、Gumbel rule、DLM watermark、dgMARK。评估指标为 TPR@1%FPR、AUROC、PPL（Qwen2.5-7B-Instruct），攻击类型含 del30、syn30、ins20。
- **主要结果**：
  - LLaDA-TANGO：clean $0.97^{±0.02}$，del30 $0.68^{±0.07}$，syn30 $0.92^{±0.04}$，ins20 $0.91^{±0.04}$，PPL 7.2，AUROC 0.998。
  - Dream-TANGO：clean $1.00^{±0.00}$，del30 $0.73^{±0.08}$，syn30 $0.98^{±0.03}$，ins20 $0.97^{±0.03}$，PPL 7.8。
  - Red–green list 在删除鲁棒性上更强（LLaDA del30 0.96 vs TANGO 0.68），但 PPL 略高（7.6/9.1），且在 Dream 上 62% 文本退化重复。
  - Gumbel rule 检测极弱（LLaDA clean 0.15，AUROC 0.815），需升温至 1.5 才恢复至 0.83 但 PPL 飙升至 20.6。
- **抗伪造结果**：基于 token 频率的密钥恢复 AUC 对 TANGO 保持在 0.504（随机水平），对 red–green list 达 0.817；伪造文本通过率 TANGO 为 0%，red–green list 为 100%。基于 token-pair 的攻击在同等偏置下仅 4% 通过 TANGO，而 51% 通过 red–green。
- **最强结果**：Dream 上 TANGO 实现 100% 未编辑检测与 97-98% 编辑后检测，同时在 forgery resistance 上显著优于所有对比方法。

## 相关工作脉络
- **KGW / context-hashed green list**（Kirchenbauer et al., 2023）：依赖前缀上下文哈希，无法适配并行去噪；TANGO 改用已解掩邻近 token 的颜色关系，彻底解耦生成顺序。
- **Red–green list / Unigram watermark**（Zhao et al., 2024）：固定全局偏置，无需上下文但频率泄漏密钥；TANGO 通过位置间动态切换偏置类实现统计隐蔽。
- **DLM watermark**（Gloaguen et al., 2026）：在期望意义上对掩码上下文应用 context-hash；在检测-困惑度曲线上介于 red–green 与 TANGO 之间，对频域攻击无防护。
- **dgMARK**（Hong & No, 2026）：通过引导解掩顺序嵌入统计签名；在最优设置下干净文本检测率低于 TANGO 且 PPL 更高。
- **LR-DWM**（Raban et al., 2026）：同时哈希左右邻居 ID 生成两个绿列表；TANGO 仅依赖颜色而非 ID，使同义替换可保留校验和。
- **Semantic watermarks**（SIR, SemStamp, SemaMark）：用语义嵌入保护单 token/句级水印；TANGO 仅用嵌入静态着色，检测阶段无需访问模型或上下文。

## 局限性与未来方向
- **删词鲁棒性较弱**：删除会切断跨越该位置的配对，导致检测分数显著下降；论文建议在实际部署中配合关键词旋转或短文本阈值适配缓解。
- **意译/回译攻击脆弱**：Back-translation 打乱 token 位置对齐，TANGO 与 red–green list 均大幅下降，需结合句法级或语义级鲁棒化设计。
- **理论假设与真实分布存在偏差**：定理 1 依赖 favored class 均匀独立、类质量严格平衡等理想条件；真实密钥下类质量存在微小偏差，需逐模型校准阈值。
- **未来方向**：探索 tap-keyed residue 自动选择以减少人工调参；结合密钥轮转限制攻击者可收集的文本量；扩展至 $n$-gram 配对或多跳校验和以提升编辑容忍度。

## 研究启发与可借鉴点
- **二阶统计信号设计**：将水印从单 token 频率迁移至 token-pair 颜色关系，利用动态偏置抵消一阶泄漏，为不可伪造水印提供通用建模范式。
- **投影排序着色策略**：基于单向投影+分位数桶的平衡着色，兼顾语义邻近性与分布均匀性，可迁移至词汇级鲁棒签名、安全提示注入等场景。
- **去噪循环轻量集成**：每步仅需一次前向传播与局部 $O(V)$ logit 更新，无额外训练成本；该工程模式可直接复用于其他离散扩散架构。
- **检测-质量-安全三角评估**：论文同步报告 TPR、PPL、频率恢复 AUC 与伪造通过率，形成完整的可信水印评估闭环，值得后续工作沿用。

## 关键术语表
- **Masked Diffusion Language Model**：从全掩码序列出发，通过多步去噪循环并行、无序填充掩码位置的离散扩散语言模型。
- **Checksum / 校验和**：水印嵌入的核心约束，满足 $(\chi(t_i) + a_\delta \chi(t_{i-\delta})) \mod q = b$ 的 token 颜色配对关系。
- **Semantic-quantile coloring / 语义分位数着色**：按密钥投影方向对词嵌入排序并均分桶的平衡着色策略，使同义词倾向同色。
- **Enforced fraction / 强制比例**：实际被日志偏置覆盖的配对比例，一端强制下约 0.61，两端强制提升至 0.74+。
- **Coupling margin / 耦合边际**：标记位置校验和成立概率超过随机率 $1/q$ 的平均超出量，决定检测分数增长斜率。
- **First-order unforgeability / 一阶不可伪造性**：在理想假设下，期望 token 频率分布与无水印时相同，频率分析无法还原密钥。

## 可复现要素
- **数据集**：C4 dataset（Rafel et al., 2020），使用 MarkLLM 附带的预处理版本；自然文本用于校准阈值的为 2000 条 C4 样本。
- **模型**：LLaDA-8B-Instruct、Dream-v0-Instruct-7B（权重可从官方仓库获取）。
- **代码/权重**：论文未声明开源 TANGO 代码；DLM watermark 与 dgMARK 使用其公开仓库（指定 commit）。Python 3.12 + PyTorch + Transformers ≥4.46, <4.57 + CUDA 12。
- **关键超参**：$q=3$，$\delta=2$，$a_2=1$，$b=0$，$\beta=5$，temperature=1.0，CFG scale=2.0（仅 LLaDA），生成长度 128 token，去噪步数 128，enforcement=either-side。
- **未提及**：具体硬件配置、并行度、训练数据（推理期水印无需训练）、随机种子分布细节。
