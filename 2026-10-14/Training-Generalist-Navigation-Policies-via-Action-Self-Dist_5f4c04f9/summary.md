---
title: "Training-Generalist-Navigation-Policies-via-Action-Self-Dist"
source: https://arxiv.org/pdf/2610.17593v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:06:45"
---

# 论文速读：Training-Generalist-Navigation-Policies-via-Action-Self-Distillation

## 一句话总结
本文提出 **Action Self-Distillation (ASD)**，利用开源自提权轨迹作为特权反馈，驱动同一个预训练 VLM 同时担任教师与学生，在无人类语言标注的情况下生成语言目标与结构化推理轨迹，从而实现可零样本泛化的语言条件机器人导航策略。

## 研究问题与动机
1. **语言标注昂贵且稀缺**：训练语言条件 VLA 导航策略通常需要带目标描述、场景语义或推理链的标注数据，而主流导航数据集（RECON、FrodoBot-2K 等）仅提供传感器观测与轨迹，不含任何语言标注。
2. **现有方案依赖外部流水线**：已有方法要么依赖人工标注，要么调度多个外部大模型生成语言目标或推理监督，仍无法摆脱标注瓶颈。
3. **单一 VLM 自我监督的可行性未探索**：前沿 VLM 在接受反馈后可修正输出，作者观察到利用轨迹反馈（特权信息）引导同一模型生成语言目标与推理，可能实现免标注的自蒸馏。
4. **核心研究问题**：能否仅凭轨迹数据，在训练循环内部完全自动生成语言目标与推理监督，从而训练出免外部标注、具备语言条件与结构化推理能力的通用导航策略？

## 核心贡献（创新点）
1. **首次将反馈条件化自蒸馏应用于机器人控制**：用一个预训练 VLM 同时担任教师与学生，通过投影真实轨迹的观测提取语言目标与推理，而非依赖外部标注流水线。
2. **特权轨迹反馈驱动的两种角色分 Pass 机制**：教师 Pass 接收带真值轨迹叠加的观测生成结构化推理与目标；学生 Pass 仅用机载观测 + 教师推断目标学习对齐推理分布与轨迹预测。
3. **EMA 滑动平均维持自教师权重稳定性**：教师权重 ϕ = αϕ + (1−α)θ（α=0.999），每 10 步更新一次，相比离线固定教师 SFT 基线在成功率上显著领先（ASD 80% vs. Claude Opus 5 SFT 36%）。
4. **结构化推理格式约束教师 grounding 到可观测内容**：强制教师输出 Task/Perception/Agents/Prediction/Planning 五段式结构，确保推理内容可在原始图像中被验证，避免虚构或幻觉。
5. **仅用轨迹数据实现零样本多阶段与跨域泛化**：在不含室内数据的 RECON+FrodoBot-2K 训练集上，零样本达到 84% 室内成功率与 95% 室外成功率，强于 OmniVLA/NaVILA。

## 方法详解
**问题设定**：地面机器人从历史 T 步状态 τ_{-T:0} 与前视图像 o_t，预测未来 H 步轨迹 τ_{0:H}，并在给定自然语言目标 g 时同步输出结构化推理 R：
$$\hat{\tau}_{0:H}, R \sim \pi(\cdot \mid o_{-T:0}, \tau_{-T:0}, g)$$
轨迹以 JSON 序列化为文本 token，通过 next-token 预测学习。

**自教师 Pass（教师视角）**：将未来真值轨迹 y=τ_{0:H} 投影到前视图像 o_0 上得到特权观测 o_0^*，输入教师 π_φ（EMA 更新权重）：
$$(\hat{g}, R, \hat{\tau}_{0:H}) \sim \text{sg}(\pi_\phi(\cdot \mid P, o_0^*, \tau_{-T:0}))$$
其中 sg 表示停止梯度，防止推理损失反向更新教师。

**学生 Pass（部署视角）**：学生 π_θ 仅接收可部署输入（无前视轨迹叠加）与教师提取的目标 ĝ：
$$(\tilde{R}, \tilde{\tau}_{0:H}) \sim \pi_\theta(\cdot \mid \tilde{P}, o_0, \tau_{-T:0}, \hat{g})$$

**训练损失**：
- **推理 KL 散度**（对教师 logits 做 forward KL，仅作用于推理 token 段）：
$$\mathcal{L}_{\text{reason}} = D_{\text{KL}}(R \parallel \tilde{R}) = \mathbb{E}_{R \sim \pi_\phi}\left[\log\frac{\pi_\phi(R \mid P, o_0^*, \tau_{-T:0})}{\pi_\theta(R \mid \tilde{P}, o_0, \tau_{-T:0}, \hat{g})}\right]$$
等价于最小化教师加权交叉熵，保留分布级软标签而非单点硬标签。
- **轨迹交叉熵**：
$$\mathcal{L}_{\text{traj}} = \text{CE}(\tilde{\tau}_{0:H}, y)$$
- **总损失**：$\mathcal{L} = \mathcal{L}_{\text{traj}} + \mathcal{L}_{\text{reason}}$

**部署阶段**：移除特权输入，学生直接接收人类操作员提供的新语言目标 g^⊕，输出推理与轨迹。

**结构化推理格式**（教师 prompt 强制）：
- **Task**：高层导航目标（一句）
- **Perception**：1-2 句可见地形与结构描述（仅限可验证内容）
- **Agents**：可见运动实体或"none"
- **Prediction**：每个 agent 的预期运动
- **Planning**：从固定动作词表 [go straight, turn left, turn right, slow down, stop, cross intersection, navigate around] 中选一个并说明理由
- **Trajectory**：JSON 格式的 future_trajectory 序列

**关键超参**（Qwen2.5-VL-3B-Instruct 主干）：max tokens=512，past waypoints=8，future waypoints=16，预测 horizon=6.4s，waypoint freq=2.5Hz，EMA α=0.999，update freq=10 steps，lr=1e-5，batch per GPU=1，gradient accumulation=2，epochs=2，cosine schedule，warmup=5。

## 实验与结果
**数据集**：训练集为 RECON（户外校园 Jackal 数据）+ FrodoBot-2K（ sidewalk 数据集，取 25% 场景随机采样）混合，train/val=9:1，**无语言目标或推理标注**。

**评估基线**：OmniVLA（当前最强 VLA）、NaVILA、Vanilla Qwen2.5-VL、Claude Opus 4.7，以及 Gaussian Policy、Diffusion Policy、Alpamayo-R1+SFT、OpenVLA+SFT 等开环轨迹预测基线。

**室内语言条件导航（9 任务，Table 1）**：
| 方法 | Avg SR |
|---|---|
| OmniVLA | **0.54** |
| NaVILA | 0.27 |
| Qwen2.5-VL-ASD | **0.84**（+30% vs OmniVLA）|
| Qwen3.5-VL-ASD | **0.94** |

最难任务（Right Chair/Monitor/Hallway）ASD 分别领先 OmniVLA 60%/50%/10% 点；Hallway 任务 OmniVLA 为零。

**室外语言条件导航（4 任务，Table 2）**：
| 方法 | Avg SR |
|---|---|
| OmniVLA | 0.90 |
| NaVILA | 0.33 |
| Qwen2.5-VL-ASD | **0.95** |
| Qwen3.5-VL-ASD | **0.98** |

Statue 任务（OOD 对象）ASD 达 80%，领先 OmniVLA 20%、NaVILA 70%。

**多阶段任务**：ASD Hallway 成功率 60%，NaVILA 为 50%（领先 10 点，原文称 30% 提升相对 NaVILA 基线）；OmniVLA 为零。

**开环轨迹预测（Table 7）**：ASD 在 RECON 与 FrodoBot-2K 上均取得最低 ADE/ATE/MAE/MSE，7/8 指标最优。

**推理重要性消融（Table 4）**：
- 关闭推理训练：SR 80% → 26%
- 推理推理推理 at 推理推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理 at 推理
