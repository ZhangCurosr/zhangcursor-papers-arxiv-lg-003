---
title: "WebHarbor-Docking-Real-Websites-for-Evolving-GUI-Agent-Envir"
source: https://arxiv.org/pdf/2611.03634v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-04 10:42:14"
field: "GUI Agent 环境与训练"
keywords: ["GUI web agent", "reinforcement learning", "web benchmark", "synthetic environment", "multimodal agent", "offline training"]
innovations: ["任务驱动的真实网站 dock 化与 5 阶段 skill 合成管线", "编码 Agent + 人工审核迭代构建保障视觉保真与功能完整", "在离线合成环境上运行 MM-GRPO RL 使 4B 开源 Agent 超越在线 RL 基线"]
benchmarks: ["WebVoyager", "Online-Mind2Web", "WebArena", "WebArena-NL", "WebHarbor test set"]
---

# 论文速读：WebHarbor: Docking Real Websites for Evolving GUI Agent Environments

## 一句话总结
WebHarbor 提出了一套由 153 个可执行 Docker 化网站环境组成、含 3,518 个任务的 GUI Web Agent 训练与评测基准，通过任务驱动的"编码 Agent + 人工审核"协作方式高保真重建真实网站的视觉界面与认证工作流；在此环境中用 MM-GRPO 对 Qwen 4B 模型进行强化学习训练，模型在多个真实网页基准上泛化良好，并超过了同规模开源 Agent 的最优成绩。

## 研究问题与动机
1. **真实网页评测不稳定**：在线评测基准（如 WebVoyager、Online-Mind2Web）会因网站内容变化、Bot 检测、区域限制而失效——论文中约 38% Online-Mind2Web 任务被 Bot 检测拦截，22% WebVoyager 任务因网站变更而过时。
2. **认证工作流覆盖不足**：已有在线基准主要覆盖公开信息检索任务，缺乏需要登录后执行、改变持久状态的工作流（如账户管理、预订、交易），而这在日常用户请求中极为常见。
3. **现有合成环境视觉/功能保真度低**：WebArena-Infinity、AutoWebWorld、CUA-Gym 等自动化生成的环境多为全新场景或精简版，缺少丰富的图像内容、完整的电商层级与产品图，甚至无法重建目标网站（如无 Amazon 页面）。
4. **手工重建难以规模化**：WebArena 等手工构建的基准提供了可复现环境，但网站数量极少（6–13 个），难以支撑大规模 RL 训练与评测。

## 核心贡献（创新点）
1. **任务驱动的高质量网站 dock 化**：以目标任务为范围，通过本地化实现、SQLite 数据库与受控重置，重建 153 个含丰富图像与多步工作流的真实网站镜像——与自动化生成方法的本质区别在于"任务约束 + 高保真视觉重建"，而非自动生成全新虚拟世界。
2. **"编码 Agent + 人工审核"迭代构建管线**：编码 Agent 按标准化 skill（5 阶段 prompt）实现前端/后端/数据库/素材，人工贡献者指定任务与功能需求，审核员对比原网站、逐任务端到端执行并退回缺陷；与纯 Agent 合成的本质区别是引入人类对视觉保真度、功能正确性与任务有效性的验收环节。
3. **含认证的复杂工作流大规模数据集**：3,518 个任务中 31.8% 需登录、35.6% 改变持久状态、53.7% 跨页推理——相比 Online-Mind2Web 等仅有检索类任务，提供了面向认证态、事务态的稀缺训练信号。
4. **低成本 RL 可迁移到真实网页**：Qwen 4B 在 WebHarbor 上进行 100 轮 MM-GRPO RL 训练后，在 WebVoyager / Online-Mind2Web / WebArena 四个基准上均超越同规模的 OpenWebRL-4B，且与领先闭源模型的排名一致性达 Spearman ρ=1.0，证明离线合成环境上的 RL 可泛化到在线真实网站。

## 方法详解
### 3.1 任务驱动的 dock 化（Task-Driven Docking）
以每个任务指定的入口页、终止页、可观察实体与属性、浏览器动作、持久状态变更定义"任务相关支持面"，将环境重建范围严格限定于任务所需的功能与内容，避免还原整个商业网站的不切实际。源网站取自 WebVoyager（15 站）与 Online-Mind2Web（136 站），任务由人类贡献者定义核心功能域后由 LLM 扩展为候选工作流，再经人工精炼。

### 3.2 Skill-Driven 合成与演化（5 阶段）
编码 Agent 基于 Playwright 抓取目标站点的渲染 DOM、图片 URL、导航层级后，使用标准 skill prompt 逐步实现：
- **Stage 1 Clone-website**：脚手架注册、DOM/图片 harvesting、种子状态与鉴权流程搭建。
- **Stage 2 Design-tasks**：为每个站生成 15–20 个可执行任务契约（涵盖搜索过滤、浏览、账户、历史、比较及站点特有操作如购物车/预订/消息）。
- **Stage 3 Task evolution**：对每个任务执行完整浏览器回放，发现缺失路由/表单/状态转换即扩展 `app.py`、Jinja 模板与种子数据，并重放。
- **Stage 4 Environment hardening**：去答案泄露（隐藏答案到详情页）、添加 near-miss 干扰项（≥5 个只违反一个约束的相似商品）、保证目录广度与跨字段一致性。
- **Stage 5 Seed database**：幂等种子初始化、4 个标准 demo 账号（alice/bob/carol/david）带真实历史状态、asset pinning、reset 验证。

### 3.3 人机迭代审核
每位审核员对每个环境平均进行 2–3 轮评审（每轮 1–2 小时），对比原网站、执行所有任务，将缺陷转化为编码 Agent 可执行的修复请求。审核员同时编写任务级 verifier 与 LLM-as-a-Judge 判分 rubric。

### 4 RL 训练配置
- **起点**：OpenWebRL-4B-SFT checkpoint。
- **Objective**：MM-GRPO（多模态多轮 GRPO），每任务 5 条轨迹，组内相对优势。
- **Reward**：格式门控 + GPT-4.1 对最终答案、行动历史、最近截图的成功判断。
- **Rollout**：每轮 48 组任务，32,768 token 上下文，当前截图 + 全历史文本。
- **优化**：PPO 2 轮，非对称 clip (0.20/0.28)，Adam lr=1e-6，全局 batch=256。
- **硬件**：Megatron TP2 DP4 + 8× 单 GPU SGLang engine。

## 实验与结果
### 数据集划分
- 训练集：2,288 任务（123 个站点 + 15 个 WebVoyager 除外站点 + 每站 20% 任务 hold out）。
- 测试集：306 任务（每站 ≤2 任务，经 GPT-4.1/GPT-5.5/GPT-5.6/GPT-6-Astra/Qwen3.6-35B 五模型脚本 verifier 一致裁决筛选）。
- 未分配：924 任务。

### 主要结果
| Model | WebVoyager | Online-Mind2Web | Avg |
|---|---|---|---|
| Qwen3-VL-4B-Thinking (base) | 52.6 | 32.0 | 42.3 |
| OpenWebRL-4B-SFT | 60.2 | 47.0 | 53.6 |
| OpenWebRL-4B (live RL) | 74.1 | 67.0 | 70.6 |
| **WebHarbor-4B** | **81.2** | **70.4** | **75.8** |

- **提升幅度**：相比共享 SFT checkpoint 提升 +22.2 分；相比在线 RL 训练的 OpenWebRL-4B 提升 +5.2 分；是所有列出开源 Agent 中最高。
- **复杂工作流泛化**：WebArena 取消自动登录时，WebHarbor-4B 仅下降 2.7 分（vs. OpenWebRL-4B 降 3.6 分），在 GitLab (+10.8)、Map (+9.4) 提升显著。
- **benchmark 一致性**：在同样 15 个 WebVoyager 站点上，WebHarbor 得分较在线版本平均提升 4.4–18.8 分；6 对模型间排名完全一致，Spearman ρ=1.0，Pearson r=0.802。
- **WebHarbor 测试集**：WebHarbor-4B 在 verifier 与 agent-judge 下分别达 30.0% 与 41.5%，超过所有 4B baseline。

### 能力维度分析
RL 增益集中在认证任务 (+13.3)、多页导航 (+12.6)、状态操作 (+10.8)；在 WebArena 上交易任务提升最大 (+21.4)。

## 相关工作脉络
1. **WebVoyager / Online-Mind2Web**：在线真实网页评测基准，覆盖广但易受网站变更与 Bot 检测影响，且任务几乎全为公开检索，缺认证工作流。本文在相同 15 个 WebVoyager 站点上提供离线可控替代。
2. **WebArena / VisualWebArena**：手工构建的可复现自托管基准，支持复杂交互但网站覆盖极有限（6–22 个）、部署重、重置成本高。本文以社区协作方式将数量扩展至 153 个。
3. **WebArena-Infinity / InfiniteWeb / AutoWebWorld / CUA-Gym / WebForge / GUI-Genesis / VeriEnv**：自动化合成的训练环境，强调规模与状态转换验证，但通常缺少丰富图像内容与经过人工审计的视觉保真度。本文与之区别在于"任务驱动的真实网站重建 + 人工验收"。
4. **OpenWebRL / WebRL / WebAgent-R1 / ComputeRL**：在线 RL 训练 Web Agent 的代表工作，依赖真实网站进行 rollout；本文证明在离线合成环境上做同配方 RL 训练可达到可比甚至更优的泛化效果。
5. **REAL**：提供 11 个网站的确定性副本，但起始于预认证状态，不涉及登录流程。本文系统地在任务中要求 Agent 自行完成鉴权。

## 局限性与未来方向
1. **保真度边界**：环境是对目标站的模仿而非等同，客户端重度交互（交互式地图、实时 ticker）以静态代理近似。
2. **评测协议局限**：脚本 verifier 与模型 judge 仍可能漏判有效替代方案或细粒度失败，需要更鲁棒的评测协议。
3. **训练探索不系统**：仅对比 OpenWebRL 这一基线；未系统探索不同 SFT 数据、通用 harness（如 OpenCode）、CLI Agent 或其他 RL 算法。
4. **规模单一**：因算力限制只训练了一个 4B 模型，跨模型家族与规模的泛化留待未来工作。
5. **法律与版权风险**：环境引用了真实网站设计与资产，研究者复用需遵守相应许可；合成环境无法完全复现真实网站的动态性与个性化。

## 研究启发与可借鉴点
1. **任务约束驱动的环境重建**：将"以任务为边界"作为合成环境的方法论——先明确任务所需的状态/路由/内容，再反向构建环境，可有效控制成本并提高相关性。可迁移到任意交互型 Agent 环境（如桌面、游戏）。
2. **Skill-Driven 五阶段 prompt 体系**：Clone → Design-tasks → Evolution → Hardening → Seed 的流水线模板可直接复用到其他网站/应用镜像任务，只需替换领域 skill。
3. **Verifier + Agent-Judge 双轨评估**：程序化状态差异检查（database delta、mutation check）与 trajectory 语义 judge 分离，前者作主判定、后者作审计信号，可推广到其他需要"过程+结果"双重验证的 Agent 评测。
4. **near-miss 干扰项注入**：任务执行时显式加入"满足类别但违反一约束"的相似项（如 5 个价格/日期/品牌不同的候选），防止 Agent 靠捷径得分，值得在其他搜索/对比类基准中复用。
5. **离线 RL 泛化到在线基准的可行性验证**：本工作证明在离线合成环境上运行与在线完全相同的 GRPO 配方也能获得跨基准泛化，为后续低成本 RL 训练路线提供了可信先例。

## 关键术语表
- **Docking**：将在线网站的依赖替换为本地资产、种子状态与可控重置，同时保留可识别的视觉与行为，实现离线可执行镜像的过程。
- **Task-conditioned reconstruction**：以目标任务定义的重建范围来裁剪环境实现，避免还原整个商业网站的全量工作。
- **MM-GRPO**：Multimodal Multi-turn Group Relative Policy Optimization，多模态多轮 GRPO，论文使用的强化学习优化目标。
- **Agent-as-a-Judge**：用 Agent 模型基于完整交互轨迹与 rubric 对 Agent 表现进行语义级评判的方法（Zhuge et al., 2024）。
- **Format gate + GPT-4.1 reward**：先校验 Agent 输出是否符合预定格式，再用 GPT-4.1 判断最终答案与轨迹是否正确，构成训练 reward。
- **Persistent state change**：任务完成后在数据库中留下的不可逆状态变更（如订单、收藏、消息），用于评估 Agent 的写操作能力。
- **Orchard interface**：用于隔离和启动各个 WebHarbor 沙箱的开源 Agent 建模框架（Peng et al., 2026）。

## 可复现要素
- **代码与数据**：公开可用，见 https://github.com/aiming-lab/WebHarbor（论文作者声明）。
- **模型权重**：使用 OpenWebRL-4B-SFT checkpoint 作为起点，该 checkpoint 沿袭自 OpenWebRL 发布。
- **环境架构**：每个站点基于 SQLite + Flask/Jinja 自建，Docker 化部署，端口由注册表动态分配。
- **关键超参**：RL 迭代 100 轮；每组 5 条轨迹；PPO epochs=2；clip (0.20, 0.28)；lr=1e-6；batch=256；上下文 32,768 tokens；temperature 0.8, top-p 1.0；viewport 1280×1000。
- **训练硬件**：Megatron TP2 DP4 + 8× 单 GPU SGLang engine（论文未明确 GPU 型号与数量）。
- **任务集合**：WebHarbor 训练任务 2,288 个、测试任务 306 个、未分配 924 个；153 个站点清单见 Appendix Tables 19–20。
- **评测协议**：WebVoyager/DeepShop 用 GPT-4.1 judge 检查截图+答案；WebArena 用官方程序化 validator；Online-Mind2Web 用 o4-mini AgentTrek judge；WebHarbor 测试集用 verifier + agent-judge 双轨。
