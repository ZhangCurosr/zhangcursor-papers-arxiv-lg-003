---
title: "Transition-Path-Sampling-Using-Koopman-Operators-and-Exit-Ti"
source: https://arxiv.org/pdf/2610.10054v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 03:05:38"
---

# 论文速读：Transition-Path-Sampling-Using-Koopman-Operators-and-Exit-Ti

## 一句话总结
本文提出一种基于Koopman算子谱分析与停止时最优随机控制（OSC）的过渡路径采样（TPS）框架，通过Koopman特征函数自动识别亚稳态并构造committor函数，推导出闭式最优控制器并在RKHS中转化为单次凸二次规划求解，无需循环模拟训练即可高效生成跨越自由能垒的反应路径。

## 研究问题与动机
- **稀有事件采样困难**：分子动力学与动力系统理论中，亚稳态被高自由能垒分隔，原生Itô扩散轨迹几乎全部困于单一势阱，真实反应路径（reactive paths）极难自然出现。
- **固定视界OSC的缺陷**：现有机器学习采样器（PIPS、TPS-DPS）将TPS建模为固定时间视界的最优控制问题，依赖神经网络参数化偏置漂移并通过simulation-in-the-loop反复rollout训练，训练初期目标集到达率极低导致梯度信号高度稀疏，且需人工设定视界长度。
- **现有击中时方法的局限**：Du et al.（2026）虽研究击中时committor估计，但将committor视为值函数并用神经网络配合受控轨迹仿真训练，计算开销大且缺乏闭式保证。
- **传统方法的依赖**：经典TPS（路径空间MCMC）需要初始反应路径；传统偏置势方法依赖少量手工集体变量，泛化性弱。

## 核心贡献（创新点）
- **提出停止时OSC框架并推导闭式控制器**：将时间视界改为目标集的首次击中时间，证明自由能最小化对应的最优控制器为 $u^* = \nabla \log \rho$，且其路径测度在所有测度（不仅限于容许控制生成）上最小化自由能。
- **Koopman谱驱动的无集体变量亚稳态识别**：利用Koopman生成元的领先特征函数自动聚类得到亚稳态核 $A_k$，并从中构造committor近似 $\hat{q}$，无需先验反应坐标或过渡轨迹。
- **RKHS近似将控制求解降维为单次线性KKT系统**：通过因式分解 $\hat{\rho}=\rho_0 w$ 分离边界与内部尺度，将原PDE边值问题转化为带线性等式约束的凸二次规划，彻底替代重复仿真与梯度训练。
- **理论完备的路径重加权机制**：基于Girsanov定理给出精确权重公式 $\hat{\rho}(x_0)e^{H(\chi)}$，确保加速采样后的轨迹可无偏恢复原始动力学的系综统计。
- **高效数值验证与跨尺度泛化**：在合成双稳态、Müller-Brown及真实丙氨酸二肽系统上，将THP从0%提升至93%–99.8%，且计算耗时较训练型基线低2–3个数量级。

## 方法详解
- **扩散过程与Koopman生成元**：以Itô扩散 $dX_t = b(X_t)dt + \sigma(X_t)dW_t$ 为参考动力学，其Koopman生成元 $L = b^\top\nabla + \frac{1}{2}\sum a_{ij}\partial_{ij}$ 为线性二阶微分算子。通过kernel EDMD从固定采样集 $\{x_i\}$ 估计领先特征对 $(\lambda_k, \psi_k)$。
- **亚稳态核与committor构造**：对谱嵌入 $\Psi(x)=[\psi_1,\dots,\psi_{m-1}]$ 执行k-means聚类，以各簇质心 $\theta^{(k)}$ 和半径 $\varepsilon_k$ 定义核心集 $A_k$。构造谱项 $\tilde{\psi}$ 将源核映射至0、目标核映射至1，余量 $h$ 在RKHS $\mathcal{H}_\kappa$ 中以最小化 $Lq=0$ 残差并满足边界条件的等式约束凸QP求解，得 $\hat{q}=\tilde{\psi}+h$。
- **停止时OSC与闭式控制器**：定义哈密顿量 $H(\chi)=\int_0^{\tau_D} f(X_t
