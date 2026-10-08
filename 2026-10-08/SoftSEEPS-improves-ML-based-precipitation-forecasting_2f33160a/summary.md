---
title: "SoftSEEPS-improves-ML-based-precipitation-forecasting"
source: https://arxiv.org/pdf/2610.09752v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:11:06"
---

# 论文速读：SoftSEEPS-improves-ML-based-precipitation-forecasting

## 一句话总结
本文提出SoftSEEPS，即对气象业务常用但不可微的SEEPS评分指标进行Sigmoid松弛的可微近似；通过在冻结的低分辨率Weather骨干潜在空间上训练轻量高分辨率解码器，证明SoftSEEPS可直接作为训练损失，且与对数RMSE联合优化能在几乎不损失RMSE的前提下使SEEPS显著下降。

## 研究问题与动机
- 气候变化导致强降水加剧，预警与灾害响应系统更关注干/轻/重降水的类别判定而非绝对强度，传统RMSE难以反映此类预报质量。
- 降水具有区域异质性、间歇性与强偏态分布，高空间分辨率下的误差度量需要依赖SEEPS等业务评分，但SEEPS的分段常数分类特性导致梯度为零，模型只能被其评估而无法直接优化。
- 现有ML天气预报工作（如FourCastNet、GraphCast等）虽在多变量预测上超越数值模式，但均未开发SEEPS的可微近似以用于训练目标。
- 本文旨在填补该空白，将SEEPS从“仅评估指标”转化为“可反向传播的损失函数”。

## 核心贡献（创新点）
- 提出SoftSEEPS损失：利用Sigmoid函数将硬分类转换为软概率向量，构造出完全可微的SEEPS近似形式，首次使业务评分可直接参与梯度优化。与Gumbel-softmax等通用离散松弛方案不同，本文显式保留了位置与日历年依赖的气候惩罚矩阵$S$。
- 验证联合优化的帕累托有效边界：证明SoftSEEPS与对数归一化RMSE的线性组合可在参数扫描下形成帕累托前沿，最优配置下SEEPS下降18.51%而RMSE无显著退化。
- 构建高分辨率降水解码基准框架：在冻结ArchesWeather-M骨干上仅训练54.3M参数的IMERGDecoder，实现从1.5°至0.1°的次日降水预测，为轻量化超分降水建模提供可复现设置。
- 提供空间维度的消融证据：分区评估显示SoftSEEPS带来的SEEPS提升在全局均匀分布，而RMSE变化呈空间异质性，证实气候学先验嵌入损失的有效性。

## 方法详解
- **软分类构造**：原始SEEPS依据阈值$t_1=0.25\text{mm}$与$t_2$（按日历年将湿日降水质量2:1划分至轻/重类别）对预报$x$和真值$y$做硬分类。本文定义可微软分类向量
  $$c(x,\tau)=\begin{bmatrix}p_{\mathrm{dry}}\\p_{\mathrm{light}}\\p_{\mathrm{heavy}}\end{bmatrix},\quad p_{\mathrm{dry}}=\sigma\!\left(\frac{t_1-x}{\tau}\right),\ p_{\mathrm{heavy}}=\sigma\!\left(\frac{x-t_2}{\tau}\right),\ p_{\mathrm{light}}=1-p_{\mathrm{dry}}-p_{\mathrm{heavy}}$$
  其中$\sigma(\cdot)$为Sigmoid，$\tau>0$为平滑参数；$\tau\to0$时收敛至原始硬分类。
- **SoftSEEPS损失**：结合气候依赖惩罚矩阵$S$（元素由干燥概率$p_1$与强降水概率$p_3=(1-p_1)/3$决定），定义
  $$s(x,y,\tau)=c(x,\tau)^\top S\, c(y,\tau)$$
  该式为Frobenius内积的可微替代，对$x$与$\tau$均可微。训练时通过plateau调度器自动退火$\tau$（以验证集离散SEEPS停滞为触发条件），平衡梯度条件数与分类逼近精度。
- **IMERGDecoder架构**：卷积上采样网络，输入为冻结骨干输出的$384\times60\times120$潜在特征，经三阶段双线性上采样（放大倍数$\times2,\times3,\times5$）映射至IMERG的$1\times1800\times3600$网格（0.1°）。每阶段含4个残差块，通道数依次为512/512/384，可训练参数总量54.3M。
- **训练设定**：目标为次日24小时累积IMEMRG降水；损失采用$\mathcal{L}=\text{MSE}_{\log}+\lambda\times\text{SoftSEEPS}$，$\lambda\in\{0.1,0.25,0.5,1,1.5,1.75\}$网格扫描；优化器AdamW，2
