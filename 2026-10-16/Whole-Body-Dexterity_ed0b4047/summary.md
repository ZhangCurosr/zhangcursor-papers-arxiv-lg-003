---
title: "Whole-Body-Dexterity"
source: https://arxiv.org/pdf/2610.22184v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 08:55:53"
field: "人形机器人全身灵巧操作"
keywords: ["whole-body manipulation", "dexterous re-grasping", "humanoid robot", "sim-to-real", "reinforcement learning", "diffusion policy distillation", "teleoperation refinement"]
innovations: ["基于模式约束的重抓握目标图与双臂交接约束，使人形双臂-躯干尺度上学习连续滑动/滚动接触", "从多对象专家轨迹蒸馏出仅依赖本体感受与物体几何的通用控制器，通过扩散+逆动力学精炼遥操作命令", "Escapability 筛选抓取种子提升示范数据质量，配合球体预训练 warm-start 降低新物体训练成本"]
benchmarks: ["Shoulder tap round-trip (basketball/cardboard box)", "Underarm stow round-trip (basketball/cardboard box)", "Shelf pick-stow-carry-unstow-place sequence", "Simulated 6-region transfer under noisy commands"]
---

# 论文速读：Whole-Body Dexterity

## 一句话总结
论文提出了全身灵巧操作（WBD）框架，让人形机器人能够像人类一样利用手臂和躯干作为接触面来抓握和重新定位物体；通过生成全身稳定抓取数据集、训练对象专用强化学习专家，并蒸馏为通用控制器，在真实 Unitree G1 上人实现了零样本迁移，成功完成肩部轻触、腋下收放及长程 shelf 任务（90% 成功率 vs 基线 18%/4%）。

## 研究问题与动机
- 当前人形机器人的灵巧操作主要局限于末端执行器和简单抓握，无法复用人类将手臂、躯干作为接触面来固定和重新定位物体的能力。
- 既有"全身操作/灵巧"相关工作往往聚焦特定设置（如双手协调、loco-manipulation），并未将"任意连杆均可作为接触面并持续切换支撑"视为通用能力。
- 与多指灵巧手相比，人形只有双臂，释放一侧手臂往往需要另一侧立刻稳住物体，重新抓握过程涉及快速的滑动/滚动接触，现有重抓握方法难以直接迁移。
- 遥操作虽可直接驱动全身，但缺少力/触觉反馈使操作员难以判断物体是否打滑，因此难以稳定执行需要接触维持的全身重新定位。

## 核心贡献（创新点）
1. **面向任意开链机械系统的全身抓取生成方法**：基于可微力封闭残差和物体符号距离场，枚举模式化接触配置（左臂/右臂/双臂）并生成多样化稳定全身抓取；与已有手部抓取生成相比，本文将躯干划分为多个子区域独立采样，避免接触坍缩到同一曲面，从而覆盖全身范围。
2. **WBD-EX：基于模式约束重抓握的对象专用专家策略**：将抓取数据集体素化为 (voxel, mode) 目标图，通过 PPO 学习在目标位置与目标抓取模式（L/B/R）间转移物体；与已有灵巧手"手指步态"方法相比，本文明确约束单臂目标必须释放对侧手臂、且允许短暂脱接触（0.48s 宽限），从而学到稳定的人-臂交接。
3. **专家轨迹蒸馏为仅依赖本体感受与物体几何的通用控制器**：用条件扩散模型 + 逆动力学网络将多对象专家轨迹融合为一个控制器，推理时通过梯度引导将粗糙遥操作命令精炼为稳定的全身重新定位指令；与直接遥操作/重映射相比，本文控制器能够在命令含噪声或接触不稳定时恢复并维持支撑。

## 方法详解
- **全身抓取生成（Energy-based optimization）**：以机器人连杆表面候选接触点为约束，最小化可微力封闭能量 $E_{\mathrm{closure}}$（法向合力与合力矩平方和）及穿透/自碰撞/关节限位惩罚；对大尺寸躯干按区域（Fig. 7）拆分接触目标以分散抓取方向；采用两阶段仿真校验：①在 4×重力下施加持续 wrench 扰动（6轴各100ms），要求仍保持接触；②在标准重力下无扰动运行，要求漂移 <10 cm 且旋转 <45°。接受抓取按 (joint pose, grasp mode, settled object pose) 索引入库。
- **目标图与模式约束重抓握**：以 2 cm 分辨率对 settled object 位置体素化，每个占据 (voxel, mode) 构成图节点；模式兼容性规定 L↔B、R↔B 合法，L↔R 必须经 B 中转。每步从远目标沿图向前推进 $d \sim \mathcal{U}(0.02, d_{\max})$，并以 20% 概率发 hold 命令；成功条件仅要求物体位置在目标 2 cm 范围内、处于正确模式并静止，不要求末端位姿与物体朝向。
- **奖励设计**：$r = r_{\mathrm{prog}} + r_{\mathrm{settle}} + r_{\mathrm{succ}} + r_{\mathrm{hold}} + r_{\mathrm{neutral}} - r_{\mathrm{fail}} - r_{\mathrm{reg}}$，所有正向项均乘以 retention $\rho_t$（未持物时为 0）；progress 奖励距离缩减（clip ±2cm）、arm neutrality 鼓励非支撑臂回默认位、failure 对掉落/超限时速/失触 >0.48s 等一次性罚 -20。
- **Domain randomization**：物体质量 0.4–1.2 kg、质心偏移 ±5mm、接触摩擦扰动、以及训练后期最大 20N 随机冲量扰动。
- **两阶段专家训练**：先预训练中等大小区间球体专家（~15k iter），再 warm-start 其他 primitive 专家，可显著缩短收敛时间（表 IX）。
- **Escapability 过滤**：用已训练专家对每个抓取打分（3s 内尝试到 6–8cm 外格点的单步转移成功率），据此剔除低可达性抓取以提升轨迹收集效率（附录 VI）。
- **蒸馏控制器**：条件扩散模型（4 keypoint/臂，269 维历史输入）+ 逆动力学头（119 维）；推理时以 16 条候选轨迹并行去噪，用梯度引导向粗命令靠拢，取误差最小者经逆动力学转关节增量；Guidance weight 可调，仿真中校正半径 ±2.5（原始 tanh 空间），真机上单关节增量裁剪 ±0.025 rad。

## 实验与结果
- **抓取数据集**：15 种 primitive（球/方/长方/圆柱，边长/直径 80–240mm），经仿真过滤后共 251 万稳定抓取； pooled torso 覆盖率达 98.4%。
- **仿真评估**：在 6 个体身区域（front、左右腋、左右肩、back）间连续转移；以 α∈{2.5, 4} 向 expert 命令加均匀噪声。WBD 控制器在训练对象上使平均 failure time 相对 noisy expert 提升 70%，在未见 ∅180mm 球和 180mm 立方体上分别提升 158% 和 71%；低噪声且 guidance=1–20 时甚至超越 clean expert 25–35%。
- **真机 on-body 重定位**（Table I，10 次/任务×物体×控制器，篮球 +15cm 纸盒）：
  -  Shoulder tap：WBD 篮球 10/10 到区、9/10 往返；纸盒 10/10、10/10；GMR 篮球 3/10→1/10、纸盒 6/10→4/10；SONIC 均 2/10 到区、0–2/10 往返。
  -  Underarm stow：WBD 篮球 10/10→9/10、纸盒 10/10→10/10；GMR 篮球 4/10→2/10、纸盒 2/10→1/10；SONIC 几乎无法完成（0/10）。
- **真机 shelf 长程任务**（Table II）：Pick 三者均为 10/10；WBD 在 Stow/Carry/Un-stow 达到 9/10、Place 7/10；GMR 3/2/1/1；SONIC 0/0/0/0。三项任务合计完成 45/50（90%）vs GMR 9/50（18%）vs SONIC 2/50（4%）。
- **消融**：EMA 平滑（β=0.5）仅能逼近 noisy expert（Table X），而 WBD 在 goals/trial 与 failure time 上均显著更优。

## 相关工作脉络
- **Dexterous hand re-grasping（e.g., [11] DexterityGen、[12]、[13][14] In-hand manipulation）**：以密集中间目标序列驱动手指级重抓握；本文将其推广至人形全身尺度，但引入 mode 约束与躯干滑动/滚动、单臂必须完全释放等新约束。
- **Whole-body / loco-manipulation（e.g., [3] AMO、[4] FALCON、[5] ULTRA、[6] VIRAL、[7] Ψ₀、[33] GRAIL、[34] MotionDisco、[36] Sumo）**：侧重行走中的推/搬/攀爬等动态行为；本文聚焦"稳态站立+全身接触面"的精细重定位，不含下肢动态 locomotion。
- **Contact-rich whole-body manipulation with torque/tactile（e.g., [8][9][21][27]）**：依赖末端力控或上身触觉；本文仅用 proprioception + 几何描述符即可维持支撑。
- **Teleoperation / retargeting（e.g., [37]–[40] H2H shadow、[43] SONIC、[50] GMR）**：纯运动映射在接触维持阶段易失稳；本文在此基础上增加一层"命令精炼"网络，使同一遥操作流形可在接触边界上自我修正。
- **Contact planning / optimization（e.g., [24]–[26]、[47] constrained sampler）**：传统方法依赖准静态/平面/箱形假设或显式图搜索；本文用数据驱动的 RL 隐式学习接触序列。
- **Pure proprioception manipulation（[19] Blind dexterity）**：证明了关节编码器可作为粗触觉通道；本文在此基础上进一步引入物体几何先验并蒸馏为通用控制器。

## 局限性与未来方向
- **下肢锁定**：当前训练时腰部焊接、下肢不参与，未覆盖腿-臂协同或行走中的全身操作。
- **腿部抓取未启用**：作者指出腿接触需联合平衡与 locomotion 学习，未来可扩展至 arm-leg 组合模式（如把重物搁膝上再用手调整）。
- **物体几何依赖性**：细长棱柱/圆柱的 regional route completion 显著低于球体，部分形状（200mm 立方体、大棱柱）在 back/loop 路线上接近 0%，表明极不规则或大尺寸物体的全身操作仍需改进。
- **真机仍有掉物**：90% 总体成功率意味着约 10% 长程任务因接触丢失失败，复杂接触过渡段的鲁棒性仍有提升空间。
- **仍依赖遥操作命令**：当前 controller 是"精炼层"而非完全自主策略；未来可与高层 planner/视觉感知结合以实现端到端自主。

## 研究启发与可借鉴点
- **两阶段专家预热（sphere pre-train → warm-start other primitives）**：以几何最规则的对象先行学习接触/交接通用策略，再迁移到难对象，显著降低训练成本（附录 V 显示聚合迭代节省 ~20%、部分对象早停）。
- **Escapability 筛选抓取种子**：用专家真实转移成功率过滤静态稳定的抓取集，可提升轨迹收集效率（+3.5pp 可用率、-3.8pp 早期失败率），是"高质量示范数据优先于数量"的典型实践。
- **模式图（mode graph）约束替代硬规则**：将 L/R/B 模式及相容性编码进目标图，使策略在"交接必经双臂稳持"的归纳偏置下更快收敛，且天然支持任意区域间规划。
- **粗命令 + 扩散引导精炼**：保留操作者意图（coarse command）的同时用 learned motion prior 纠偏，guidance weight 可调以在"响应性"与"稳定性"间折中，适合遥操作与 RL 混合部署场景。
- **可与本团队方向结合**：可将 WBD 的 mode-constrained goal graph 迁移至其他具身平台（四足+臂、移动底盘+臂）的接触重定位；也可将扩散精炼层与视觉-语言高层指令对接，形成"高层意图→遥操作/轨迹生成→接触精炼"的三级架构。

## 关键术语表
- **Whole-Body Dexterity (WBD)**：将机器人全身连杆（不限于末端）作为接触面、通过连续切换支撑点实现物体重定位的操控能力。
- **Grasp mode (L/B/R)**：目标抓取构型的抽象——仅左臂、仅右臂、或双臂同时接触物体；单臂目标强制另一臂完全释放。
- **Mode-constrained re-grasping**：以模式兼容性为约束（L↔B、R↔B，禁止 L↔R 直连）的重新抓握范式，确保每次交接至少有一段双臂稳持过渡。
- **WBD-EX**：针对单个物体几何训练的对象专用强化学习专家策略，输入本体状态与目标（位置+模式），输出关节增量。
- **Escapability scoring**：用已训练专家在冻结权重下对每个抓取测试其能在 3s 内成功转移到 6–8cm 外目标格点的比例，用于过滤低质量初始抓取。
- **Goal-distance curriculum**：将每步最大目标距离 $d_{\max}$ 从 2cm 线性 ramp 到 8cm，使策略逐步接触远近不同的转移难度。
- **Diffusion + inverse-dynamics distillation**：用条件扩散模型生成多 candidate 未来关键点点迹，再以逆动力学网络将其转为关节增量，推理时通过梯度引导对齐遥操作命令。
- **Grace window (0.48s)**：允许策略在快速重配置过程中短暂失去所需臂接触而不判负，避免策略退化为准静态 Hold。

## 可复现要素
- **数据集**：WBD 抓取数据集（251 万稳定抓取，体素化粒度 2cm），论文未提供下载链接；代码/权重：论文未明确声明开源（主页 wholebodydex.com），训练使用 mjlab [52] 与自定义 PPO。
- **关键超参**：PPO clip=0.2、γ=0.99、GAE λ=0.95、lr 自适应 1e-5~1e-2、entropy=0.005(Stage1)/0(Stage2)；仿真 200Hz 物理、50Hz 控制；专家 8192 env/8×H100；蒸馏 130k AdamW 步、batch 2560、lr 3e-4、100 timestep cosine；真机推理 50Hz、16 candidate DDIM 步。
- **硬件/环境**：Unitree G1、Xsens MoCap、NVIDIA RTX 5090 推理、mjlab 仿真。
