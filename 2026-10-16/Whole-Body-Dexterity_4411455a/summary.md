---
title: "Whole-Body-Dexterity"
source: https://arxiv.org/pdf/2610.22184v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:55:34"
field: "人形机器人全身灵巧操作"
keywords: ["whole-body manipulation", "dexterous manipulation", "reinforcement learning", "sim-to-real", "humanoid robot", "grasp synthesis", "policy distillation"]
innovations: ["提出 mode-constrained 全身抓取图与重定位 RL 专家", "两阶段 warm-start 训练策略加速多对象专家收敛", "扩散+逆动力学蒸馏实现遥操作命令的灵巧细化"]
benchmarks: ["Shoulder tap round-trip success", "Underarm stow success", "Shelf pick-stow-carry-place task"]
---

# 论文速读：Whole-Body Dexterity

## 一句话总结
本文提出 Whole-Body Dexterity (WBD) 框架，让人形机器人能够利用手臂和躯干作为接触表面进行物体的灵活重定位；通过生成全身稳定抓取数据集、训练 RL 专家策略并蒸馏为通用控制器，在 Unitree G1 上实现了零样本 sim-to-real 转移，完成率达 90%（对比直接重映射 18%、SONIC 4%）。

## 研究问题与动机
- **现有方法的局限**：当前 humanoid 灵巧操作局限于末端执行器和简单抓取，无法像人类一样利用手臂和躯干作为接触面来支撑和重定位物体。
- **全身操作的复杂性**：人形机器人只有两个手臂，比多指手的支持自由度更少；释放一只手臂需要另一只手臂支撑物体抵住躯干，切换抓取模式需要快速重构和滑动/滚动接触。
- **teleoperation 的瓶颈**：缺乏力/触觉反馈时，操作员难以判断物体是否被夹紧或滑动，限制了操作范围。
- **动机**：让人形机器人具备类似人类的"全身灵巧"能力——如抱着箱子、用腋下夹住物体行走、将物体从架子转移到桌上等。

## 核心贡献（创新点）
1. **全身抓取生成方法**：基于能量优化的 force-closure 最小化，定义于 mode-typed 接触配置，适用于任意开链刚体系统。与 Dex-GraspNet/BODex 等仅针对手的抓取生成不同，本文支持任意链接作为接触表面。
2. **WBD-EX 专家策略**：基于 grasp 数据集训练 per-object 重定位专家，采用 mode-constrained re-grasping 公式（L/B/R 模式约束），目标只指定物体位置和抓取模式，不包括关节配置和物体姿态。
3. **蒸馏控制器 WBD Controller**：将专家轨迹蒸馏为条件扩散模型 + 逆动力学网络，仅需本体感觉和物体几何描述符，无需特权物体状态，支持人类遥操作引导。

## 方法详解
**Pipeline 三阶段**：

1. **Whole-Body Grasp Generation**
   - 枚举配置类型（哪些肢体/躯干区域接触物体）
   - 将大躯干网格划分为子区域（Fig. 7 展示 6 个体区：前/左腋下/右腋下/左肩/右肩/后）
   - 最小化可微 force-closure residual：$E_{\mathrm{closure}} = \frac{1}{K^2}\left(\|\sum \mathbf{n}_k\|_2^2 + \|\sum (\mathbf{p}_k - \bar{\mathbf{p}}) \times \mathbf{n}_k\|_2^2\right)$
   - 两阶段仿真验证：① 4×重力下受扰持握；② 标准重力下漂移<10cm、旋转<45°
   - 最终生成 2.51M 个稳定抓取，按 (voxel, mode) 索引

2. **WBD-EX 训练**
   - gcMDP  formulation：目标为物体位置 + 抓取模式 (L/B/R)
   - Goal graph：voxel 节点按 mode 分组，L↔B、R↔B 合法，L↔R 不直接连通（必须经 B 过渡）
   - 奖励函数：progress + settle + success bonus + hold quality + arm neutrality − fail penalty − regularizers
   - Domain randomization：质量 0.4–1.2kg、质心偏移 ±5mm、摩擦系数扰动、随机扰动力（ ramps 到 20N）
   - 两阶段训练：先预训练球体专家，再 warm-start 其他几何体

3. **Controller Distillation**
   - 架构：条件扩散模型 (U-Net) + 逆动力学网络
   - 输入：269 维（ proprioceptive history 24×3 keypoint + 14 joint pos + 14 targets + 14 errors + 5 geometry features）
   - 输出：48 维 keypoint displacement（未来 50ms/100ms）
   - 推理时通过 gradient guidance 将粗糙遥操作命令细化为稳定重定位
   - 无特权物体状态，支持可调 guidance weight（1–350）

## 实验与结果
**数据集**：15 种几何体（6 球体 ∅120–220mm、4 立方体 140–200mm、3 棱柱、2 圆柱），2.51M 抓取，90.1% 平均覆盖率

**仿真结果**（Fig. 4）：
- 在噪声命令下（α=2.5, 4），WBD 恢复 clean expert 水平
- 平均失败时间提升：训练对象 +70%，held-out 球体 +158%，held-out 立方体 +71%
- 低噪声+guidance 1–20 时，WBD 甚至超过 clean expert（+25–35%）

**真实世界结果**（Unitree G1，Table I/II）：
- Shoulder tap：WBD 篮球 9/10 round trip，GMR 1/10，SONIC 0/10
- Underarm stow：WBD 篮球 9/10，GMR 2/10，SONIC 0/10
- Shelf task（pick-stow-carry-unstow-place）：WBD 7/10 complete，GMR 1/10，SONIC 0/10
- **总体完成率：WBD 45/50 (90%) vs GMR 9/50 (18%) vs SONIC 2/50 (4%)**

## 相关工作脉络
1. **Dexterous hand manipulation**：[11] DexterityGen、[12] human-level teleoperation — 本文扩展 re-grasping 原理从手指到整个人形身体。
2. **Whole-body manipulation**：[8][9][20][21] 使用手臂/躯干操作大件物体 — 本文更强调跨身体区域的灵活重定位，而非仅支撑。
3. **Loco-manipulation**：[3][4][5][6][7] FALCON/ULTRA/VIRAL/Ψ₀ — 结合 locomotion 与 manipulation；本文聚焦静止/行走时的 on-body 操作。
4. **Teleoperation**：[37][38][39][40] HumanPlus/TWIST — motion retargeting；本文通过 RL 控制器修正 retargeting 误差，维持稳定接触。
5. **Classical contact optimization**：[24][25][26] — 基于优化的 contact planning；本文用 learning-based 方法处理更复杂的动态重抓。
6. **Grasp synthesis**：[15][16][17] Dex-GraspNet/BODex — 针对灵巧手的能量优化；本文适配到人体全身链接。

## 局限性与未来方向
- **腿部抓取未包含**：当前仅使用手臂和躯干，腿抓需要联合平衡和 locomotion 学习。
- **专家成功率不均**：球体专家路由完成率最高（∅220 球 88%），而 slender prism/cylinder 较低（部分 <25%）。
- **distillation 数据效率**：低成功率专家的可用轨迹收集效率较低。
- **未来方向**：扩展到 leg contact（如膝上支撑重物）、学习 joint with lower-body controller、处理更复杂日常物体。

## 研究启发与可借鉴点
1. **mode-constrained goal graph**：用离散模式（L/B/R）约束抓取转换图，确保 handoff 安全过渡，可迁移到其他 multi-support 操作任务。
2. **escapability-based grasp filtering**：训练后评估每个抓取的"逃逸能力"（transition success rate），过滤低质量抓取提升蒸馏效率。
3. **two-stage warm-start**：先训练易收敛对象（球体），再 warm-start 难对象，减少 20% 总迭代成本且提高最终性能。
4. **diffusion + inverse dynamics distillation**：将 RL 专家轨迹蒸馏为条件扩散策略，结合梯度 guidance 实现遥操作 refine，兼顾灵活性和可控性。
5. **goal-distance curriculum**：从 2cm 渐进到 8cm 的目标距离，平稳引入长程重定位。

## 关键术语表
- **Whole-Body Dexterity (WBD)**：利用机器人整个身体（手臂、躯干）作为接触表面进行物体灵活重定位的能力。
- **Grasp mode (L/B/R)**：抓取模式标签，分别为左臂(L)、双臂(B)、右臂(R)持握。
- **WBD-EX**：per-object 重定位专家策略，基于 PPO 在 gcMDP 上训练。
- **Force-closure energy**：基于接触法向力的净力和净矩平方和，衡量抓取稳定性。
- **Goal graph**：按 (voxel, mode) 组织的离散目标图，节点间转移受 mode 兼容性约束。
- **Escapability**：从某抓取状态出发成功过渡到目标的比例，用于过滤低质量抓取。
- **Gradient guidance**：在扩散去噪过程中通过梯度引导 motion sampling 逼近遥操作命令。

## 可复现要素
- **数据集**：2.51M 抓取，论文未声明公开；workspace: wholebodydex.com
- **代码**：论文未声明开源
- **硬件**：Unitree G1 人形机器人
- **关键超参**：voxel size 2cm，控制频率 50Hz，physics 200Hz，episode 40s，d_max curriculum 2→8cm，perturbation ramp 0→20N over 8K iters，guidance weight 1–350
- **训练资源**：WBD-EX 使用 8× NVIDIA H100，~65K environments；distillation ~22 H100 GPU-hours
