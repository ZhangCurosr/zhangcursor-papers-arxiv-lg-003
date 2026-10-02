---
title: "Who-Verifies-the-Graph-Misspecification-Attacks-on-Causal-Ac"
source: https://arxiv.org/pdf/2609.40027v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:44:24"
---

# 论文速读：Who Verifies the Graph? Misspecification Attacks on Causal Action Verification for Language Agents

## 一句话总结
本文对语言智能体的因果动作验证器 CIVeX 进行红队测试，证明仅篡改已被信任的因果图（如省略一条双向边或反转中介箭头）即可使验证器输出内部自洽但实际有害的证书；提出基于有界随机实验的 attestation 防御可消除所有假执行，但揭示出“安全易恢复、价值难挽回”的成本不对称性，呼吁将图完整性纳入验证器的核心攻击面。

## 研究问题与动机
- 核心问题：因果动作验证器依赖“已提交图 (committed graph)”作为可信计算基，该图通常由领域专家 elicitation 获得；当图发生轻微结构误设 (misspecification) 时，验证器的安全性与有效性将如何失效？
- 现有方法不足：CIVeX 原始论文在 confounded tool-use benchmark 上报告零假执行，但其 misspecification 测试仅演练“声明无知”（将观测混杂重标记为潜变量并添加 $T \leftrightarrow Y$ 边），从未覆盖“图本身写错”的真实攻击场景。
- 动机一：量化三种典型图错误下验证器的失效速度与失效形态（有害执行、瘫痪、全反转）。
- 动机二：设计可部署的防御机制并完整核算其安全成本与价值成本，为智能体因果网关提供鲁棒性边界。

## 核心贡献（创新点）
1. **首次对因果动作验证器进行系统性红队攻击**，提出省略双向边、将汇点误作混杂因子、反转中介方向三种图误设攻击模式。与现有验证器鲁棒性研究不同，本文不挑战 do-calculus 识别理论，而是固定识别框架并精准攻击其唯一信任输入（已提交图）。
2. **提出有界随机实验 attestation 防御**，对观测 EXECUTE 决策抽样 500 行随机化样本并比较置信区间以证伪图结构。与单纯依赖观测数据加 sensitivity bound 的方法不同，该防御通过引入干预样本主动检测 falsification，而非被动假设无未测量混杂。
3. **量化揭示安全恢复与价值恢复的成本不对称性**，证明仅审计执行无法挽回错误拒绝，必须对 REJECT 证书进行独立采样审计才能线性恢复价值。与过往只报告 false positive 率的评估范式不同，本文建立双轨成本核算框架并给出精确实验预算。
4. **澄清防御响应策略的关键影响**，指出 fallback 策略不可复用已损坏的调整集，必须采用 graph-free 原始均值差判断，否则防御会隐性继承另一类错误。与现有 Seldonian/高置信 OPE 工作仅关注 refusal 边界不同，本文细化到 per-action 执行路径的响应差异。

## 方法详解
- **验证器基线 (CIVeX 流程)**：对每个待验证的状态改变调用，在已提交图上搜索后门调整集 (backdoor adjustment set)。若存在，基于 500 行观测数据做 OLS 回归估计效应 $\hat{\tau}$，计算 95% 正常理论单侧置信下限 (LCB)；若 LCB ≥ 0 且风险界 ≤ 0.5 则返回 EXECUTE，否则 REJECT。两种裁决均附带证书（含图名、调整集、点估计、LCB 与风险限制）。若图拒绝识别，则路由至 EXPERIMENT（使用 500 行随机化样本重判）或 ABSTAIN。
- **攻击 1：省略双向边 (Omitted edge)**：移除图中标记潜混杂 $U$ 的 $T \leftrightarrow Y$ 边，使现有调整集变得不足。后门调整“成功”但对不足集做 OLS 会吸收混杂偏倚，LCB 计算基于有偏估计，证书仍内部合法。
- **攻击 2：坏控制 (Bad control / collider admitted)**：合成 $C = 0.8\bar{T} + 0.8(Y - \bar{Y}) + \varepsilon$ 并承诺为混杂因子（实为 outcome descendant）。条件化切断正确执行路径，导致正确执行率从 52.9% 暴跌至 7.9%，但假执行率为 0（验证器陷入 paralysis）。
- **攻击 3：箭头反转 (Reversed arrowhead)**：将真实中介结构 $T \to M \to Y$（含符号相反的direct path）反转为 $M \to T$。M 被错误纳入调整集，间接路径被阻断，估计量仅捕捉直接效应，符号完全反转。
- **Attestation 防御**：对观测 EXECUTE 决策抽取 500 行随机化样本，计算均值差与 95% Welch 区间；若与证书区间不相交则判定该图对该动作 falsified。面对 falsification 有两种响应：Refuse（直接 abstain）或 Fallback（在随机样本上重新决策）。关键约束：Fallback 必须放弃已损坏图的调整集，改用 graph-free 原始差值判断，否则伪造的调整会继续阻断因果路径。
- **拒绝审计 (Rejection audit)**：对 REJECT 裁决按相同机制抽样实验并审计；按比例 $p$ 随机选取被拒动作验证，可线性恢复有益执行。

## 实验与结果
- **数据集与设置**：Causal-ToolBench（合成 SCM 驱动，6 类工作流，7 个种子，每种子 25 实例，每条件 n = 1,050 动作）。潜混杂强度 $s$ 为连续参数，基准强度 $s = 2.5$。
- **评估指标**：假执行率、正确执行率、被丢失的有益动作比例、平均效用。效用规则：有益执行 +|τ|，有害执行 −|τ|，有益被拒 −0.3|τ|，有害被拒 +|τ|，每次实验成本 0.05；always-abstain 底线为 +0.98。
- **主要结果**：
  - *honest graph, s=2.5*：0% 假执行，效用 +2.27（与论文报告 +2.23 吻合）。
  - *denied graph (省略边), s=2.5*：假执行率 15.3%（per-seed 13.3–17.3%），91% 执行有害，效用骤降至 +0.35。
  - *direction flip*：假
