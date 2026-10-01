---
title: "PDMD-PROJECTED-DISTRIBUTION-MATCHING-DIS-TILLATION-FOR-VIDEO"
source: https://arxiv.org/pdf/2609.35768v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 02:53:25"
---

# 论文速读：PDMD-PROJECTED-DISTRIBUTION-MATCHING-DIST-TILLATION-FOR-VIDEO

## 一句话总结
本文提出投影分布匹配蒸馏（PDMD），通过从DMD更新中投影掉平行于学生-评论家端点残差的分量，滤除在线评论家误差；该方法仅需一行代码修改，即可显著稳定少步视频扩散模型蒸馏训练，并在Wan2.1与MiniMax-H3上刷新4-NFE蒸馏性能。

## 研究问题与动机
- **核心问题**：DMD可将视频扩散模型的推理步数压缩至4步，但训练过程中样本会逐步出现过度饱和（oversaturation）、纹理伪影与动态坍塌，质量随迭代持续恶化。
- **现有方法不足**：DMD2通过增加判别器与多轮评论家更新缓解滞后，但引入对抗训练耦合与额外显存开销；其他混合方法（如rCM、ADV、AnyFlow）需追加轨迹损失、flow-map训练或多阶段优化，工程复杂度高。
- **根本原因定位**：在线评论家的分数估计存在近似误差，且会滞后于快速变化的学生模型；该误差直接进入DMD的分数差更新并随迭代累积，是训练不稳定的根源。
- **研究动机**：能否不引入额外损失、网络、数据或训练阶段，仅利用DMD自身已计算的量来估计并抵消评论家误差？

## 核心贡献（创新点）
1. **定位误差来源**：揭示DMD训练退化源于评论家误差累积，并证明学生-评论家端点残差 $\boldsymbol{r}=\boldsymbol{x}_0^c-\boldsymbol{x}_0^s$ 在固定查询下是评论家端点误差的无偏估计。
2. **提出PDMD投影滤波**：将DMD更新中平行于残差 $\boldsymbol{r}$ 的分量正交投影出去；在高维浓度与弱对齐假设下，理论上可移除常数比例的评论家误差能量，同时仅丢弃趋于零的理想蒸馏信号。
3. **极简实现与广泛验证**：仅修改一行更新代码，零额外开销；在2D toy任务、Wan2.1-T2V-1.3B与MiniMax-H3-33B联合视频-音频生成上均验证了稳定性与SOTA性能，用户研究全面胜出。

## 方法详解
- **DMD更新回顾**：学生 $G_\theta$ 生成端点 $\boldsymbol{x}_0^s$，重加噪得 $\boldsymbol{x}_t=\alpha_t\boldsymbol{x}_0^s+\sigma_t\boldsymbol{\epsilon}$；分别计算在线评论家分数 $\boldsymbol{s}_\text{critic}$ 与冻结教师分数 $\boldsymbol{s}_\text{teacher}$，更新信号为 $d=\boldsymbol{s}_\text{critic}-\boldsymbol{s}_\text{teacher}$。
- **残差构造**：将评论家分数线性转换为一一步端点预测 $\boldsymbol{x}_0^c=(\boldsymbol{x}_t+\sigma_t^2\boldsymbol{s}_\text{critic})/\alpha_t$，定义残差 $\boldsymbol{r}=\boldsymbol{x}_0^c-\boldsymbol{x}_0^s$。理论证明（Eq. 7）在固定 $\boldsymbol{x}_t$ 下 $\mathbb{E}[\boldsymbol{r}|\boldsymbol{x}_t]=\boldsymbol{e}$，即 $\boldsymbol{r}$ 是评论家端点误差 $\boldsymbol{e}$ 的无偏估计。
- **正交投影**：构造投影算子 $P_r=\boldsymbol{r}\boldsymbol{r}^\top/\|\boldsymbol{r}\|^2$，更新修正为 $\boldsymbol{d}_\perp=P_r^\perp\boldsymbol{d}=\boldsymbol{d}-\frac{\langle\boldsymbol{d},\boldsymbol{r}\rangle}{\|\boldsymbol{r}\|^2}\boldsymbol{r}$；若 $\boldsymbol{r}=\boldsymbol{0}$ 则保持原更新。
- **误差去除保证**：定义去除比例 $\gamma_e=\|P_R\boldsymbol{e}\|^2/\|\boldsymbol{e}\|^2$，由Cauchy-Schwarz可得下界 $\mathbb{E}[\gamma_e|\boldsymbol{x}_t]\geq\|\boldsymbol{e}\|^2/(\|\boldsymbol{e}\|^2+\text{tr}\,\Sigma(\boldsymbol{x}_t))$（Eq. 10）。当条件噪声在高维集中且不与 $\boldsymbol{e}$ 强对齐时，投影平均去除常数比例误差，而理想信号沿 $\boldsymbol{r}$ 的投影分量 $\gamma_s=O_\mathbb{P}(d_\text{eff}^{-1})$ 趋于零。
- **算法流程**：Student前向→重加噪→教师/评论家评分→计算 $d$ 与 $\boldsymbol{r}$→执行 `d = d - dot(d,r)/dot(r,r) * r`→反传更新学生。完整伪代码见Algorithm 1。

## 实验与结果
- **Wan2.1-T2V-1.3B @ 4 NFE**：PDMD在VBench上取得总分 **83.73**，较同配置DMD†（82.70）提升 **+1.03**，超越DMD2†（83.44）、AnyFlow（83.54）与50步教师（83.06）。DMD†在1500步后急剧退化（5000步降至63.04），PDMD在10000步后仍稳定维持83.5以上。用户研究在视觉与运动质量上全面优于所有基线。
- **MiniMax-H3-33B 联合视频-音频 @ 4 NFE**：PDMD在VideoGen-Eval上取得视觉总分 **83.17**，较最强蒸馏基线提升 **+0.41**；在全部6项音频指标（PQ/CE/CU/IS/IB/DeSync）上均达最佳，PQ仅落后50步教师0.04。用户研究在视觉、运动、音频质量上全面胜出。
- **2D验证**：在八高斯环实验中，PDMD能量距离0.0080、on-mode fraction 0.903，显著优于DMD（0.0273/0.838）；在双模滞后实验中，DMD在20次运行中14/20次单模崩溃，PDMD仅3/20
