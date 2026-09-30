# RL-PAO: PREDICTION AS ACTION INDECISION MAKING UNDER UNCERTAINTY

Jiahui Feng, Dafang Zhao, Zheng Chen\*, Zhengmao Li, Lingwei Zhu†

• Great Bay University, School of Computing and Information Technology, China

The University of Osaka, School of Information Science and Technology, Japan

The University of Osaka, SANKEN, Japan

Aalto University, School of Electrical Engineering, Finland

†Corresponding author: zhulingwei@gbu.edu.cn

## ABSTRACT

Decision-making under uncertainty often relies on predicted parameters, yet accurate prediction does not necessarily lead to good operational decisions. Aligning prediction with downstream optimization requires learning from the consequences of the decisions those predictions induce. We introduce RL-PaO, a reinforcement learning framework that integrates system formulation, optimization, and decision execution into a single environment. This yields a Markov decision process in which prediction is regarded as action: it shifts the environment to produce subsequent context and reward that explicitly aligns prediction error with realized cost, and learning the optimal policy does not require differentiating through the black-box solver. We evaluate RL-PaO on day-ahead energy scheduling using real historical data. On the test year, RL-PaO achieves the lowest annual cost among the non-oracle baselines, achieving on average 10% cost reduction. Moreover, RL-PaO is capable of further analyses to provide strong interpretability both from the policy evolution perspective and the cost-accuracy trade-off.

## 1 INTRODUCTION

Decision-making under uncertainty is fundamental to energy systems, transportation, and supply chains (Mandi et al., 2024). Operational decisions must be made before parameters or constraints become known. Predictive models estimate these parameters, optimization solvers generate decisions based on the estimates and execution under realized conditions determines their value. It is vital to evaluate prediction quality since it changes the optimization objective and influences downstream execution. It remains a central challenge to align the predictor with the operational purpose.

Predict-then-optimize (PtO) trains the predictor independently of the optimizer, but its training objective such as MSE, MAE can be misaligned with downstream cost: lower estimation error does not necessarily translate to better decisions (Elmachtoub & Grigas, 2022). Prediction-and-optimization (PaO) addresses this mismatch by feeding back decision-loss surrogates to augment the prediction loss (Mandi et al., 2025; Silvestri et al., 2026), in the hope that by minimizing the augmented loss, a balance between prediction accuracy and decision quality could be reached so as to improve downstream performance (Mandi et al., 2024). However, training on the composite loss could lead to undesired solutions that are suboptimal for all of the member losses, see Table 1 for detailed comparison. Therefore, a paradigm that can effective align prediction and optimization is called for.

In this paper, we introduce RL-PaO. By treating system formulation, objective optimization and schedule execution as an integrated environment, we convert prediction into action that can shift the environment to produce next solution and a scalar reward that explicitly evaluates the cost-accuracy trade-off for the current solution. I.e., we build a Markov Decision Process (MDP) (Puterman, 1994) that closes the loop between the environment and an agent policy, see Figure 1. Though RL has been utilized in improving solver procedures such as MILP (Tang et al., 2020; Qi et al., 2021), RL-PaO differs from them in that RL-PaO concerns alignment between prediction and blackbox optimization, not accelerating a solver conditional on its internal states.

<table><tr><td rowspan="3"></td><td colspan="3">Prior-based Optimization</td><td colspan="3">Predict-Optimize Paradigm</td></tr><tr><td colspan="3"></td><td rowspan="2">PtO</td><td colspan="2">PaO</td></tr><tr><td>DO</td><td>RO</td><td>SO</td><td>Supervised PaO</td><td>RL-PaO Ours</td></tr><tr><td>Uncertainty modeling</td><td>×</td><td>Robust uncertainty sets</td><td>Probability distributions</td><td>Open-loop prediction</td><td>Closed-loop prediction</td><td>Prediction-optimization alignment</td></tr><tr><td>Placement-free uncertainty</td><td></td><td></td><td></td><td>V</td><td>×</td><td></td></tr><tr><td>Decision-focused training</td><td></td><td></td><td></td><td>×</td><td>√</td><td></td></tr><tr><td>Model-agnostic optimization</td><td></td><td></td><td></td><td></td><td>×</td><td></td></tr><tr><td>Gradient-free</td><td></td><td></td><td></td><td>V</td><td>×</td><td></td></tr></table>

Table 1: Comparison of methods for optimization under uncertainty. Ours is the first RL-based PaO method that aims at improving the alignment between prediction and optimization. We emphasize the importance of Decision-focused training that optimizes by downstream decision cost instead of prediction error; and Model-agnostic optimization that operates without full knowledge of the underlying optimization formulation.

We evaluate RL-PaO on a day-ahead scheduling problem using real history data from the Osaka University. We train a Proximal Policy Optimization (PPO) agent (Schulman et al., 2017) on the train set and evaluate on a held-out test year. Results show that our method achieves the lowest annual cost among all baseline methods except the oracle. Moreover, unlike conventional PtO and PaO, our formulation provides strong interpretability to characterize the cost-accuracy trade-off: prediction error is insufficient to assess decision quality In short, our contributions are:

• (Novel paradigm) We demonstrate that RL – conventionally leveraged for closed-loop system control – can also serve as a decision-oriented prediction module. On top of this, we formalize the two-stage prediction-optimization pipeline as a Markov Decision Process, based on which stable convergence and robustness towards hyperparameters is achieved.

• (Parsimonious formulation) We propose a counterintuitive state-action construction where the state is defined exclusively by the downstream system operational schedule. We prove that the schedule plan inherently encodes sufficient information to infer the upper bound of the constraints, removing the need for extra environmental input that supervised forecasting methods rely on.

• (Real-world evaluation) We evaluate RL-PaO on a real day-ahead scheduling problem at the Osaka University. By learning from offline data, the trained RL agent outperforms all non-oracle baselines in terms of annual cost, 9.42% better than supervised PaO and 12.07% than PtO. RL-PaO also provides strong interpretability from multiple perspectives.

• (High generalizability) Our framework treats the downstream task as a black box, and is inherently agnostic to the underlying system model, mathematical problem formulation, and prediction target. This yields strong generalizability and enables a truly model-free, problem-agnostic paradigm for decision-making problem.

## 2 BACKGROUND AND PROBLEM SETTING

In this paper we consider the following generic optimization problem parameterized by an uncertain quantity in vector $\pmb { \xi } = \{ \pmb { \xi } _ { f } , \pmb { \xi } _ { g } , \pmb { \xi } _ { h } \}$ ..

$$
\begin{array} { c } { { { \pmb x } ^ { * } ( { \pmb \xi } ) = \arg \displaystyle \operatorname* { m i n } _ { { \pmb x } \in { \pmb X } } f ( { \pmb x } , { \pmb \xi } _ { f } ) , } } \\ { { \mathrm { s . t . } \qquad g ( { \pmb x } , { \pmb \xi } _ { g } ) \leq { \bf 0 } , } } \\ { { \hphantom { { \pmb x } ^ { * } ( { \pmb x } , { \pmb \xi } _ { h } ) \cdot } h ( { \pmb x } , { \pmb \xi } _ { h } ) = { \bf 0 } , } } \end{array}\tag{1}
$$

where x denotes the operational decision, f is the operational objective, and g and h describe inequality and equality constraints, respectively. In practice, the decision must be made before ξ becomes available. Therefore, from the perspective of how the uncertainty on ξ is incorporated, existing approaches can be broadly organized into two families. The first family leverages nominal values, uncertainty sets, or probability distributions. In the rest of the paper, we refer to this family collectively as Prior-based Optimization (PbO). The second family exploits observable contextual information to estimate the unknown quantities and uses them in the optimization process. We refer to this family as Prediction-Optimization (P-O). (Lahoud et al., 2025; Mandi et al., 2024)

![](images/88ade987c897d92e1bd10060d8985a305a97f8021314a73361e551ba3dcc1aae.jpg)  
Figure 1: Overview of the proposed RL-PaO framework. (A) We explicitly include system modeling, optimization, scheduling and real-world operation into one environment. The RL agent observes current context to predict the unknown constraint in the optimization objective to be solved by an MILP solver. The solution influences execution process, yielding reward and next context. (B) our RL formulation is under the PaO category and different from the conventional supervised PaO methods. (C) Architectural comparison of different methods. PtO uses an open-loop predictor. Supervised PaO closes the training loop by incorporating downstream loss. By contrast, the proposed method aligns prediction and optimization by a novel MDP and RL training.

Prior-based Optimization. Prior-based opimization specifies a representation of uncertainty before solving the optimization problem. Depending on how uncertainty is represented, three representative formulations are deterministic optimization (DO), robust optimization (RO), and stochastic optimization (SO). DO infers fixed estimate $\hat { \xi }$ from historical data. SO assumes a predefined probability distribution for ξ, reformulating the problem using expectations; i.e., minimizing $\mathbb { E } _ { \pmb { \xi } _ { f } } [ f ( \bar { \pmb { x } } , \pmb { \xi } _ { f } ) ]$ subject to $\mathbb { E } _ { \pmb { \xi } _ { g } } [ \pmb { g } ( \pmb { x } , \pmb { \xi } _ { g } ) ] \le \mathbf { 0 }$ and $\mathbb { E } _ { \pmb { \xi } _ { h } } [ { \pmb h } ( { \pmb x } , { \pmb \xi } _ { h } ) ] = { \bf 0 }$ . RO bounds ξ within a predefined uncertainty set. Optimizing the worst-case objective max ${ \bf \xi } _ { \pmb { \xi } _ { f } } f ( \pmb { x } , \pmb { \xi } _ { f } )$ while ensuring constraints strictly hold for all realizations $\xi _ { g }$ and $\xi _ { h }$ . These methods demand analytical priors for $\xi ,$ which are often intractable in complex environments, especially for SO. Our approach directly learns optimal strategies, circumventing the requirements on mathematical priors and explicit uncertainty modeling.

PtO and PaO. Besides the prior-based methods above, addressing uncertainty through prediction in advance or jointly with optimization has been gaining popularity. Let z denote a latent variable available at the decision time, $\boldsymbol { \xi }$ an uncertain parameter required by the downstream optimization problem. A predictive model qe parameterized by $\pmb { \theta }$ maps the latent variable to an estimate, then fed into the objective function to construct a decision $\mathbf { \boldsymbol { x } } ^ { * }$

$$
{ \hat { \pmb { \xi } } } = q _ { \pmb { \theta } } ( z ) , \qquad { \pmb { x } } ^ { * } ( { \hat { \pmb { \xi } } } ) = \arg \operatorname* { m i n } _ { { \pmb { x } } \in { \mathcal { X } } } f ( { \pmb { x } } , { \hat { \pmb { \xi } } } ) .\tag{2}
$$

The decision $\pmb { x } ^ { * }$ is then executed as a plan to obtain downstream cost. When the prediction model is trained independently of the optimizer, the method is called pedict-then-optimize (PtO). On the contrary, it is called prediction-and-optimization (PaO). PaO explicitly feeds back decision quality to the prediction model to help improve training.

We identify the following questions that cannot be readily answered by the existing methods:

1. Prior-based methods require knowing uncertain characterizations, which are typically intractable in complex environments.

2. standard PtO methods typically operate in an open-loop manner, relying on minimizing estimation errors without feedback from realized operational outcomes (Chen et al., 2022; Lahoud et al., 2025). However, lower estimation errors do not necessarily translate to better decisions or lower downstream costs (Mandi et al., 2024).

3. Existing PaO methods closes the loop by feeding back downstream costs to augment the estimation error (e.g. MSE), in a hope that minimizing the augmented loss would balance minimizing cost and MSE. However, performing gradient descent on the added loss could lead to solutions that are suboptimal for all member losses. (Shah et al., 2022)

Problem Formulation. We consider the following instance of Equation 1: day-ahead scheduling electricity use. The problem focuses on uncertainty in the inequality-constrained photovoltaic (PV) generation upper bound, a scenario that is particularly challenging for existing PaO methods. The system consists of four core components: (1) PV generation provides variable energy input; (2) Energy Storage Systems (ESS) shift power consumption and supply across time periods; (3) Load represents fixed electricity demand, and (4) power exchange with the main power grid. The problem can be cast as the Mixed Integer Linear Programming (MILP) form:

$$
\begin{array} { r l } & { \begin{array} { r l } { \underset { P ^ { \mathrm { g r i d } } } { \mathrm { m i n } } } & { c ^ { \top } P ^ { \mathrm { g r i d } } } \\ { \mathrm { s . t . } } & { \mathbf { 1 } _ { \pm } ^ { \top } P _ { t } = \mathbf { 0 } , \quad \mathbf { 0 } \leq P _ { t } \leq U _ { t } , \quad E _ { t } = E _ { t - 1 } + \left( \eta _ { \mathrm { c h a r } } \cdot p _ { t } ^ { \mathrm { c h a r } } - \eta _ { \mathrm { d i s c } } ^ { - 1 } \cdot p _ { t } ^ { \mathrm { d i s c } } \right) , \quad \forall t } \end{array} } \\ & { \begin{array} { r l } { \mathrm { w h e r e } } & { P _ { t } : = \left[ \begin{array} { c } { P _ { t } ^ { \mathrm { g r i d } } } \\ { P _ { t } ^ { \mathrm { l o a d } } } \\ { P _ { t } ^ { \mathrm { l o a d } } } \\ { p _ { t } ^ { \mathrm { c h a r } } } \\ { p _ { t } ^ { \mathrm { d i s c } } } \end{array} \right] , U _ { t } : = \left[ \begin{array} { c } { U _ { t } ^ { \mathrm { g r i d } } } \\ { U _ { t } ^ { \mathrm { c l o r } } } \\ { u _ { t } \cdot U _ { \mathrm { g r o m p } } ^ { \mathrm { c h a r } } } \\ { ( 1 - u _ { t } ) \cdot U _ { \mathrm { g r o m p } } ^ { \mathrm { d i s c } } } \end{array} \right] , \mathbf { 1 } _ { \pm } : = \left[ \begin{array} { c } { + 1 } \\ { + 1 } \\ { - 1 } \\ { + 1 } \end{array} \right] . } \end{array} } \end{array}\tag{3}
$$

where c denotes the cost vector, ${ \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf } { } { \mathbf } { } \mathbf { }  { \mathbf { } \mathbf { } } { \mathbf { } \mathbf { } } { \mathbf { } } { \mathbf } { \mathbf { } } { \mathbf } { \mathbf { } } { \mathbf } { \mathbf } { } \mathbf { } \mathbf { }  { \mathbf { } \mathbf { } \mathbf { } \mathbf { } } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { \mathbf } { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf \mathbf { } \mathbf { } \mathbf { } \mathbf  { \mathbf \mathbf { } \mathbf \mathbf { } \mathbf } $ and 1 are $\vert { \mathcal { M } } \vert .$ dimensional vectors, with M denoting the full set of ESS units. $P _ { t } ^ { \mathrm { p v } }$ is actual PV generation, $P _ { t } ^ { \mathrm { { \breve { l o a d } } } }$ is the fixed load demand. $\pmb { p } _ { t } ^ { \ }$ are the (dis)charging power. E the state of battery that should be kept in a fixed range. Intuitively, the objective minimizes total grid electricity purchase cost over the scheduling horizon, with c denoting the electricity price, $P _ { t } ^ { \mathrm { g r i d } }$ the grid import power. The definitions of all variables and parameters are summarized in Table 2.

Following Equation 2, prediction enters the problem through $\breve { U } _ { t } ^ { \mathrm { p } }$ , the per-step upper bound for the PV generation. A predictive model $q _ { \theta }$ is trained to output $\breve { U } _ { t } ^ { \mathrm { p v } }$ then the objective is fed into an optimizer to obtain the solution $\mathbf { x } ^ { * }$ . In this paper, we design a novel Markov Decision Process around $\mathbf { \boldsymbol { x } } ^ { * }$ that closes the loop between prediction and optimization, allowing efficient reinforcement learning algorithms to tackle the problem.

Table 2: Notations for the problem.
<table><tr><td>Symbol Description</td><td></td></tr><tr><td>C</td><td>Electricity price</td></tr><tr><td> $P _ { t } ^ { \mathrm { p v } }$ </td><td>Actual PV output</td></tr><tr><td> $\dot { P } _ { t } ^ { \mathrm { g r i d } }$ </td><td>Grid import power</td></tr><tr><td> $P _ { t } ^ { \mathrm { { \bar { l o a d } } } }$ </td><td>Fixed load demand</td></tr><tr><td> $\dot { \pmb { p } _ { t } }$ </td><td>ESS (dis)charging power</td></tr><tr><td> $U _ { t } ^ { \mathrm { p v } }$ </td><td>PV upper bound†</td></tr><tr><td> $U _ { * } ^ { \mathrm { { \dot { g } r i d } } }$ </td><td>Grid exchange limit</td></tr><tr><td> $U _ { \mathrm { g r o u p } } ^ { \cdot }$ </td><td>(Dis)charge power limit</td></tr><tr><td> $\check { E _ { t } }$ </td><td>Battery state</td></tr><tr><td> $\eta .$ </td><td>(Dis)charging efficiency</td></tr><tr><td> $\mathbf { \Delta } \mathbf { u } _ { t }$ </td><td>Binary ESS mode indicator</td></tr><tr><td> $\mathcal { M }$ </td><td>ESS unit set</td></tr></table>

† Uncertain parameter in the problem.

## 3 SOLVING THE SCHEDULING PROBLEM WITH RL-PAO

Predicting the uncertain constraint in Equation 3 shifts the objective problem and subsequently the solution schedule. Since the MILP optimizer internal states cannot be observed, conventional PtO/PaO methods assume that accurate prediction leads to optimized downstream costs. However, we show the assumption is false. Instead, we propose a novel MDP that permits an optimal policy to align prediction and optimization to improve downstream cost in a principled manner.

## 3.1 PTO AND PAO

Figure 1C describes the standard training paradigm for PtO and PaO, respectively. PtO assumes that better estimation of the uncertain parameter can lead to lower downstream operational costs. However, this assumption rarely holds as the constraints can greatly influence the optimization objective and the resulting solution space. Recent studies show that there is no consistent correlation between prediction accuracy and downstream costs (Chen et al., 2022; Mandi et al., 2024). Motivated by this, supervised PaO methods augment the MSE prediction loss by some function of cost, in a hope that SGD training could also lead to a balanced prediction and cost-awareness:

$$
\mathcal { L } _ { \mathrm { P a O } } ( \pmb { \theta } ) = \alpha \cdot \underbrace { \widehat { \mathbb { E } } \left[ \left( q \pmb { \theta } ( \boldsymbol { z } ) - \pmb { \xi } \right) ^ { 2 } \right] } _ { \mathrm { M S E } } + \beta \cdot \underbrace { f ( \pmb { x } , \hat { \pmb { \xi } } ) } _ { \mathrm { C o s t - a w a r e } } ,
$$

where $\alpha , \beta > 0$ are the weighting coefficients. In existing literature, $\alpha , \beta$ are typically computed through either heuristic search or a two-stage pre-training paradigm (Gabriele et al., 2026; Mandi et al., 2020; Shah et al., 2022).

The standard PaO formulation therefore has two intrinsic drawbacks: (1) minimizing the augmented loss corresponds to learning multi-task behavior. However, the conventional SGD can lead to solutions that neither minimizes the MSE loss nor achieving cost-awareness. (2) Naively balancing MSE with cost-awareness the term by a weighted sum often yields models that underperform the extreme cases: i.e. either $\alpha = 0 \mathrm { o r } \beta = 0$

## 3.2 RL-PAO: A NEW RL PARADIGM FOR SOLVING THE PAO PROBLEM

Our proposed method closes the loop by a novel Markov Decision Process in which the agent acts to predict ξ and receives a calibrated reward.

Environment. We deliberately include system modeling, optimizing, planning and execution into environment with which the agent interact. The RL agent observes current solution from the MILP optimizer as context to predict the unknown constraint for the next day-ahead schedule. The solution under goes execution and yields real cost as part of the reward. This way, predictions can indeed be regarded as actions as they shift the environment (internal states of the MILP optimizer).

State. The state includes sufficient information to predict the uncertain parameter $\boldsymbol { \xi }$ similar to other prediction methods. In our formulation, the continuous state space is $\mathbb { R } ^ { 7 2 }$ , consisting of the optimal scheduling plan $\mathbf { \boldsymbol { x } } ^ { * }$ , which is a comprehensive operational profile (e.g., battery dispatch and grid purchasing) that implicitly reflects the latent meteorological patterns and temporal dynamics governing the PV upper bounds. To enable efficient learning, raw high-dimensional features are processed via dimensionality reduction before being fed into the agent.

Action. As indicated in the Problem Formulation, the predicted quantity is the per-step PV generation upper bound $\pmb { \xi } : = U _ { t } ^ { \mathrm { p v } }$ that resides in $\mathbb { R } ^ { 2 2 }$ . The actions are generated conditional on the dimensionality-reduced states. Note that unlike the existing supervised PaO methods, our design naturally constructs a feedback loop since: (1) prediction as action can influence the optimization and plan execution (through the inequality constraints) due to the deliberate design, see Figure 1; (2) each action receives a reward designed so that maximizing cumulative reward improves the alignment between prediction and downstream optimization.

Transition. After producing an action $U _ { t } ^ { \mathrm { p v } }$ , it is used in the inequality constraint in Equation 3. We then invoke the MILP optimizer to solve the day-ahead scheduling problem to output the optimal schedule $\pmb { x } ^ { * } \in \mathbb { R } ^ { 7 2 }$ . Real-world execution then reveals the true PV output and the realized operational cost. Note that revealing $P _ { t } ^ { \mathrm { t r u e } }$ is a long and highly complex process that does not readily permit online RL. Therefore, offline learning from the existing data logs is required.

Reward. Motivated by the observation that better prediction does not necessarily lead to better downstream cost performance (Gabriele et al., 2026), we design our reward to be an empirical mixture of prediction accuracy and decision quality measured by realized scheduling cost:

$$
r _ { t } = - \beta \cdot \mathtt { N } \big ( C _ { \mathrm { r e a l } } \big ) - ( 1 - \beta ) \cdot \mathtt { N } \big ( e _ { \mathrm { p r e d } } \big )\tag{4}
$$

where $C _ { \mathrm { r e a l } }$ is the actual electricity cost under the schedule, $e _ { \mathrm { p r e d } }$ is prediction error, and N denotes normalization that scales the two reward terms to the same range. $\beta$ is a hyperparameter balancing decision quality and prediction stability. Note that unlike the existing supervised PaO methods that augments loss, the association of cost to reward allows dynamically adjusting the agent behavior.

Policy. It is well-known that an MDP permits an optimal policy if the states and actions are finite, rewards are bounded, discount factor is less than one (Puterman, 1994). RL-PaO satisfies these assumptions and therefore we are guaranteed that a well-calibrated reward function ideally can lead to optimal alignment between prediction and optimization to achieve lower downstream operational costs, provided the algorithm can attain global optimum. This optimality stands in sheer contrast to the empirically designed mixture supervised loss of the conventional PaO methods (Shah et al., 2022; Mandi et al., 2024) that may be stuck in local optima that minimize neither prediction error nor downstream cost surrogate.

![](images/5eb9dd0b90f0059f5e93cb9bd324fa8a0414ebefa465868ed21eaa383b42e76c.jpg)  
Figure 2: Cumulative rewards (upper row) and downstream operation costs (lower) across different learning rates. Transparency of lines set according to $\beta$ value. RL-PaO is effective in (1) establishing correspondence between high rewards and lower costs; (2) the performance curves change smoothly along gradual increase of hyperparameters.

Training Algorithm. Having described the MDP, we use Proximal Policy Optimization (PPO) (Schulman et al., 2017) to train the RL agent. It has been shown that PPO can efficiently learn from offline data (Zhuang et al., 2023) and can converge to the global optimum (Liu et al., 2019). PPO iteratively updates the policy πθ by maximizing the clipped surrogate objective:

$$
L ( \theta ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( r _ { t } ( \theta ) A _ { t } , \operatorname { c l i p } ( r _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { t } \right) \right]\tag{5}
$$

where $\begin{array} { r } { r _ { t } ( \theta ) = \frac { \pi _ { \theta } \left( a _ { t } | s _ { t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { t } | s _ { t } \right) } } \end{array}$ is the probability ratio, $A _ { t }$ is the advantage estimate, and € controls the clipping range. The agent is trained in an offline fashion on historical data. In each training episode, the agent outputs an uncertain parameter estimate, then a black-box MILP optimizer is invoked to compute the corresponding optimal schedule. Based on the schedule and the revealed cost, reward is calculated against the realized true output. In the proposed framework, the MILP optimizer, its solution schedule plan and plan execution are all treated as part of the environment.

## 4 EXPERIMENTS

We conduct experiments to answer the following research questions: RQ1. Is our proposed MDP well-posed in the sense that maximizing cumulative rewards equals minimizing downstream costs? RQ2. Can RL-PaO outperform the conventional PbO as well as PtO/PaO methods? RQ3. Can our method provide interpretability, including evidence that good prediction does not translate to better downstream operational costs? RQ4. Is our method robust towards hyperparameters?

## 4.1 EXPERIMENTAL SETUP

Task. We use RL-PaO to tackle the day-ahead scheduling problem specified in Equation 3. We use a public dataset that records load, electricity price, temperature, etc. at the Osaka University in Japan¹. The training set spans 2016–2018 and the test set covers 2019, all at hourly resolution. The MILP scheduling model is solved with the SCIP solver. The objective is to minimize the total annual operational cost of the microgrid subject to constraints. A training episode traverses the entire training set sequentially, with one training step being one day.

Baselines. We select all three types of methods as baselines. Oracle: The full-information benchmark that solves the scheduling problem using ground-truth PV generation, load and electricity price, serving as the theoretical lower bound of operation cost. PbO: DO uses point forecast; RO is with the worst-case uncertainty sets, and SO is based on scenario generation. PtO/PaO: We follow the standard PtO practice to train using MSE prediction loss. For PaO, we opt for the SPO+ surrogate regret loss (Dupont et al., 2024).

![](images/ef7b8f41185b937fd07497b59ea3b9842c1f72d13443b2df8ba08098bd61ee7b.jpg)

![](images/8bd504a10ae998c6c55f2c4f5002435aa44914f89c01b8822973019be29c4782.jpg)  
Figure 3: Comparison on annual cost. Blue solid line and shade are mean and standard deviation over 5 independent runs. Left: RL-PaO compared against the conventional PbO baselines that require model knowledge in uncertainty (DO, RO, SO). Right: comparison against PtO, PaO methods. In both cases the model-free RL-PaO achieves the best cost.

## 4.2 RESULTS

RQ1 & RQ2: Performance. Figure 2 answers RQ1 with an affirmative. It is visible that both cumulative rewards (upper) and downstream operational costs (lower) are improved along with training after learning rates are sufficient large, under all possible $\beta$ values. This verifies that (1) our novel design that exploits prediction as action to change downstream execution is well-posed, since converged policies can indeed achieve lower downstream costs; (2) the RL-PaO formulation is smooth in hyperparameters in the sense that the performance curves change smoothly along with the increase of learning rates. There is no abrupt change after $\lfloor x \ge 5 \times 1 0 ^ { - 6 }$ , though an excessively large value $( \mathrm { e . g . ~ } 1 0 ^ { - 3 } )$ induces noticeable training oscillation. We consider $1 \mathrm { r } = 5 \times 1 0 ^ { - 5 } , \beta = 0 . 9$ to be the optimal configuration for subsequent analysis.

Regarding RQ2, Figure 3 shows that RL-PaO also outperforms the baselines in terms of annual cost. The left hand side compares against the PtO baselines that require model knowledge on uncertainty. Yet, it is visible in the zoom-in plot that model-free RL attains the best final annual cost, underperforming only the oracle.

In the RHS figure we see a similar trend. R and supervised PaO, leading to around $9 \times 1 0 ^ { 5 }$ This is because supervised PaO methods such as SPO+ require uncertain parameters to appear in the objective function, an assumption that is violated for the constraint-side PV uncertainty in our scheduling problem. RL-PaO naturally bypasses this limitation by directly optimizing end-to-end decision quality in a black-box manner. Table 3 further consolidates the observation: RL-PaO is the top player among all baseline methods. Comparing to Oracle, RL-PaO achieves the lowest cost gap 15.91% by modelfree learning from data.

Table 3: Overall performance comparison on the 2019 test set. Annual cost is measured in the unit of $\times 1 0 ^ { 4 }$ . ↓ indicates lower is better.
<table><tr><td>Category</td><td>Method</td><td>Annual Cost ↓</td><td>Cost Gap (%) ↓</td></tr><tr><td rowspan="5">PbO</td><td>Oracle</td><td>963.76</td><td>0.00</td></tr><tr><td>DO</td><td>1496.20</td><td>55.25</td></tr><tr><td>RO</td><td>1145.20</td><td>18.83</td></tr><tr><td>SO</td><td>1133.90</td><td>17.65</td></tr><tr><td>PtO</td><td>1233.40</td><td>27.98</td></tr><tr><td rowspan="3">PtO/PaO</td><td>Supervised PaO</td><td>1207.90</td><td>25.33</td></tr><tr><td>RL-PaO</td><td>1117.10</td><td>15.91</td></tr><tr><td></td><td></td><td></td></tr></table>

RQ3: Interpretability. Different from PtO/PaO baselines that do not provide interpretability for their predictions, RL-PaO offers insights from its policy space. Figure 4 shows policy evolution (Zhu et al., 2025) of the learned Gaussian distribution across 12 action dimensions at training day 0, 100, and 200. Prediction of the uncertain parameter is sampled from the policy. Starting from a broad and nearly uniform distribution at day 0, the policy becomes progressively narrower and moves toward the optimal decision region that aligns prediction with downstream cost optimization. The shift of the Gaussians at each dimension identifies different important regions that together characterize the optimal solution space for the studied problem.

![](images/6e338222f5ad5cd7ed98df7ab0ac52db5fe38ac99a1face5b655e45e1481fbc1.jpg)  
Figure 4: Policy evolution plot of the learned Gaussian distribution across 12 action dimensions at training day 0, 100, and 200. Recall that prediction of the uncertain parameter is sampled from the policy. As learning proceeds, the policies gradually narrow and shift toward the optimal decision region that aligns prediction with downstream cost optimization.

![](images/7befd33762aab67a966fe50dc13877daf475457e78dec657dc92e5f6d608b82f.jpg)  
Figure 5: Prediction error versus operation cost, daily level. The first three windows show scatter plots on the test set. The last window shows a joint-distribution plot. Transparency distinguishes weekdays and weekends. PtO and PaO points are concentrated in a low estimation error region but the average cost is high. Outliers with extremely high costs are identified by red dots with purple background. By contrast, RL-PaO error distribution is more dispersed, but the average cost is lower, as can be seen from the 80% quantile line from the last window.

Fig. 5 visualizes the joint distribution of daily prediction MAE and daily operation cost for four methods on the test dataset. It is visible that PtO clusters in the region with low prediction error but relatively high operation cost, as its training objective purely minimizes fitting error without considering downstream scheduling impact. This observation also supports the finding that lower estimation errors does do not translate to lower costs. It should also be noted that PtO has outliers shaded in purple, indicating extremely high costs are incurred – a phenomenon unique to PtO. We attribute this to its open-loop nature. Supervised PaO shifts slightly toward lower cost but remains confined to a similar error-cost regime. In contrast, RL-PaO disperses its error estimations but achieve significantly lower cost. This is even more clear from the last joint distribution plot that estimates the densities of MAE and costs for all compared methods. It can be seen that the RL-PaO's MAE distribution is more uniform, but its cost distribution places a majority of mass in lower-cost region, as indicated by the 80% quantile line. This observation further supports that high prediction accuracy does not equal better decision quality.

Figure 6 further quantifies this relationship at the annual level. The LHS shows the Pareto front of prediction error versus annual total cost. PtO achieves the lowest prediction error among the baselines but suffers from higher operation cost, lying in a Pareto-dominated region. By adjusting the reward weight β, RL-PaO spans a continuous trade-off curve and sits on the Pareto frontier, indicating that no competing methods can simultaneously achieve lower prediction error and lower operation cost. The RHS shows the overall correlation across all RL-PaO configurations with different β and learning rates. The Pearson correlation coefficient is 0.973, indicating a strong overall positive relationship between prediction error and operation cost. It is intuitive that for some of the configurations that lower MAE indeed indicates lower annual cost. But this relationship no longer holds for the majority of points scattered in the lower left corner.

![](images/df81669dd0854459249e9220803a48559b4cfd599f2a57282989f9fcfcac4a57.jpg)

(b) Error-Cost Correlation (sweep β & learning rate)  
![](images/83ebaf58f0bfdd1ffed722271672bc9ea882686cbb6b18f2cfed4210e3eedb18.jpg)

Figure 6: Prediction error versus operation cost, annual level. (a) Pareto front of annual cost against prediction MAE, recorded for $\mathtt { l r } = 5 \times 1 0 ^ { - 5 }$ The reward weight $\beta$ spans a continuous tradeoff curve and sits on the Pareto frontier, indicating that no competing methods can simultaneously achieve lower prediction error and lower operation cost. (b) Overall correlation across all $\beta$ and learning rate configurations. The blue points scattered in the lower left corner supports the finding that lower prediction error does not equal lower cost.  
![](images/d098e9351b9b14c3ec6fa2741304b4985c83f2bfb88a93bdc86b291cb7b3fb5a.jpg)  
Figure 7: Mahanttan plot of the proposed method across learning rates and reward weights $\beta .$ Transparent dots show final operational costs, with deep blue dots indicating the mean. (Upper) Results of fixing $\beta$ and varying 1r. (Lower) Fixing 1r and varying $\beta .$ The superior performance of RL can be attained from various combinations.

RQ4: Hyperparameter Sensitivity. Figure 7 shows a Mahanttan plot of the proposed method across learning rates and reward weights β. Transparent dots are final operational costs of independent runs, with deep blue dots indicating the mean of the corresponding combinations. The upper row shows results of fixing β and varying 1r, while the lower row fixing 1 r and varying β. It is visible that the superior performance of the proposed method is not a result of specific hyperparameter setting but can be attained from various combinations.

## 5 CONCLUSION

Predicting the parameters is crucial for decision making under uncertainty. PtO and PaO predict the parameter through independent prediction loss and decision-augmented surrogate, respectively. This paper presented RL-PaO, an RL framework that treats prediction as action and integrates system formulation, optimization, and decision execution into a single environment, yielding an MDP that could be solved effectively by existing RL methods. Experiments on the day-ahead scheduling problem using real data at the Osaka University verified that RL-PaO achieved lowest annual cost than all non-oracle baselines. Extensive analysis showed that our RL-PaO design can also provide strong interpretability and is robust towards hyperparameters.

We identify a potential limitation for this work: while the black-box treatment of the downstream problem delivers strong generalizability, it also introduces computational overhead during training. Every training step requires solving the full downstream optimization problem and runtime can become prohibitive for large-scale complex systems with a vast set of decision variables.

Future work can be pursued in two directions. First, it would be helpful to extend to joint prediction of multiple uncertain parameters of different physical natures, including load, electricity prices and distributed renewable output, and examine the performance of a single RL agent in such multivariable settings. Second, generating longer-horizon schedules such as one-week dispatch plans could also help alleviate computational burden.

## REFERENCES

Brandon Amos and J. Zico Kolter. Optnet: Differentiable optimization as a layer in neural networks. In Proceedings of the 34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, pp. 136–145. PMLR, 2017.

Gah-Yi Ban and Cynthia Rudin. The big data newsvendor: Practical insights from machine learning. Operations Research, 67(1):90–108, 2019.

A. Ben-Tal and A. Nemirovski. Robust optimization – methodology and applications. Mathematical Programming, 92(3):453–480, 2002.

Q. Berthet, M. Blondel, O. Teboul, M. Cuturi, J.-P. Vert, and F. Bach. Learning with differentiable perturbed optimizers. In Advances in Neural Information Processing Systems, volume 33, pp. 9508–9519, 2020.

D.P. Bertsekas. Nonlinear Programming. Athena Scientific, 2nd edition, 1999.

Dimitris Bertsimas and Melvyn Sim. The price of robustness. Operations Research, 52(1):35–53, 2004.

J.R. Birge and F. Louveaux. Introduction to Stochastic Programming. Springer, 2nd edition, 2011.

Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, Cambridge, UK, 2004.

Xianbang Chen, Yafei Yang, Yikui Liu, and Lei Wu. Feature-Driven Economic Improvement for Network-Constrained Unit Commitment: A Closed-Loop Predict-and-Optimize Framework. IEEE Transactions on Power Systems, 37(4):3104–3118, 2022.

Priya L. Donti, Brandon Amos, and J. Zico Kolter. Task-based end-to-end model learning in stochastic optimization. In Advances in Neural Information Processing Systems, volume 30, 2017.

Chloé Dupont, Pietro Favaro, Fran,cois Vallée, Bruno Francois, and Jean-Fran,cois Toubeau. Decision-focused learning for optimized participation of an industrial consumer in energy-only and reserve markets. In 2024 IEEE PES Innovative Smart Grid Technologies Europe (ISGT EUROPE), pp. 1–5, 2024.

Adam N. Elmachtoub and Paul Grigas. Smart “predict, then optimize". Management Science, 68 (1):9–26, 2022.

A.N. Elmachtoub, J.C.N. Liang, and R. McNellis. Decision trees for decision-making under the predict-then-optimize framework. In International Conference on Machine Learning, pp. 2858— 2867. PMLR, 2020.

A. Ferber, B. Wilder, B. Dilkina, and M. Tambe. Mipaal: Mixed integer program as a layer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 1504–1511, 2020.

Kris Johnson Ferreira, Bin Hong Alex Lee, and David Simchi-Levi. Analytics for an online retailer: Demand forecasting and price optimization. Manufacturing & Service Operations Management, 18(1):69–88, 2016.

Giuseppe Gabriele, Fabio Pavirani, Seyed Soroush Karimi Madahi, and Chris Develder. Forecasting what matters: Decision-focused rl for controlled ev charging with unknown departure times. In Proceedings of the 17th ACM International Conference on Future and Sustainable Energy Systems (E-Energy '26), 2026.

Jérémie Gallien, Adam J. Mersereau, Andres Garro, Alberte Dapena Mora, and Martín Nóvoa Vidal. Initial shipment decisions for new products at Zara. Operations Research, 63(2):269–286, 2015.

L. Kong, J. Cui, Y. Zhuang, R. Feng, B.A. Prakash, and C. Zhang. End-to-end stochastic optimization with energy-based model. In Advances in Neural Information Processing Systems, volume 35, pp. 11341–11354, 2022.

J. Kotary, F. Fioretto, P. Van Hentenryck, and B. Wilder. End-to-end constrained optimization learning: A survey. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, pp. 4475–4482, 2021.

Alan Anis Lahoud, Ahmad Saeed Khan, Erik Schaffernicht, Marco Trincavelli, and Johannes Andreas Stork. Predict-and-optimize techniques for data-driven optimization problems: A review. Neural Processing Letters, 57:40, 2025.

Boyi Liu, Qi Cai, Zhuoran Yang, and Zhaoran Wang. Neural trust region/proximal policy optimization attains globally optimal policy. In Advances in Neural Information Processing Systems, volume 32, pp. 1–12, 2019.

J. Mandi and T. Guns. Interior point solving for lp-based prediction + optimisation. Advances in Neural Information Processing Systems, 33:7272–7282, 2020.

J. Mandi, P.J. Stuckey, and T. Guns. Smart predict-and-optimize for hard combinatorial optimization problems. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 1603–1610, 2020.

Jayanta Mandi, James Kotary, Senne Berden, Maxime Mulamba, Victor Bucarey, Tias Guns, and Ferdinando Fioretto. Decision-focused learning: Foundations, state of the art, benchmark and future opportunities. Journal of Artificial Intelligence Research, 81:1623–1701, 2024.

Jayanta Mandi, Marianne Defresne, Senne Berden, and Tias Guns. Feasibility-aware decisionfocused learning for predicting parameters in the constraints. In Advances in Neural Information Processing Systems, volume 38, 2025.

M.V. Pogančić, A. Paulus, V. Musil, G. Martius, and M. Rolinek. Differentiation of blackbox combinatorial solvers. In International Conference on Learning Representations, 2020.

Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, Inc., New York, NY, USA, 1st edition, 1994.

Meng Qi, Mengxin Wang, and Zuo-Jun Shen. Smart feasibility pump: Reinforcement learning for (mixed) integer programming. In ICML Workshop: Reinforcement Learning for Real Life, 2021.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.

Sanket Shah, Kai Wang, Bryan Wilder, Andrew Perrault, and Milind Tambe. Decision-focused learning without decision-making: Learning locally optimized decision losses. In Advances in Neural Information Processing Systems, volume 35, 2022.

Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczyński. Lectures on Stochastic Programming: Modeling and Theory. Number 9 in MOS-SIAM Series on Optimization. SIAM, Philadelphia, PA, 2009.

Mattia Silvestri, Senne Berden, Gaetano Signorelli, Ali İrfan Mahmutoğulları, Jayanta Mandi, Brandon Amos, Tias Guns, and Michele Lombardi. Score function gradient estimation to widen the applicability of decision-focused learning. Journal of Artificial Intelligence Research, 85, 2026. doi: 10.1613/jair.1.19498.

Yunhao Tang, Shipra Agrawal, and Yuri Faenza. Reinforcement learning for integer programming: Learning to cut. In Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, pp. 9367–9376, 2020.

K. Wang, B. Wilder, A. Perrault, and M. Tambe. Automatically learning compact quality-aware surrogates for optimization problems. Advances in Neural Information Processing Systems, 33: 9586–9596, 2020.

Lulu Wen, Kaile Zhou, Shanlin Yang, and Xinhui Lu. Optimal load dispatch of community microgrid with deep learning based solar power and load forecasting. Energy, 171:1053–1065, 2019.

Dafang Zhao, Daichi Watari, Yuki Ozawa, Ittetsu Taniguchi, Toshihiro Suzuki, Yoshiyuki Shimoda, and Takao Onoye. Data-driven online energy management framework for hvac systems: An experimental study. Applied Energy, 352:121921, 2023.

Lingwei Zhu, Haseeb Shah, Han Wang, Yukie Nagai, and Martha White. q-exponential policy optimization. In International Conference on Learning Representations (ICLR), 2025.

Zifeng Zhuang, Kun LEI, Jinxin Liu, Donglin Wang, and Yilang Guo. Behavior proximal policy optimization. In The Eleventh International Conference on Learning Representations, 2023.

## A APPENDIX

## A.1 RELATED WORK

PbO. DO replaces uncertainties parameters with deterministic estimates (Boyd & Vandenberghe, 2004; Bertsekas, 1999) but loses feasibility and optimality under time-varying uncertainty. RO guarantees worst-case feasibility over a predefined uncertainty set (Ben-Tal & Nemirovski, 2002; Bertsimas & Sim, 2004) but yields over-conservative solutions with excessive costs. SO optimizes expected cost via probability distributions (Shapiro et al., 2009; Birge & Louveaux, 2011), yet relies on valid distributions, neglects action risks, and is computationally expensive at scale.

PtO. PtO decouples prediction and optimization: a predictor minimizes forecast error, and a solver generates decisions from predicted parameters (Elmachtoub & Grigas, 2022). This modular framework and allows independent module tuning, with applications spanning microgrid dispatch (Wen et al., 2019), inventory management (Gallien et al., 2015), and retail pricing. (Ferreira et al., 2016) combined random forest forecasts with integer programming for pricing; (Gallien et al., 2015) paired linear regression with MILP for inventory optimization. In energy systems (Zhao et al., 2023; Chen et al., 2022), LSTMs predict renewable generation for downstream solvers. However, PtO suffers from prediction-decision mismatch, and lower forecast error does not equal better decision.

PaO. To address prediction-decision mismatch, PaO optimizes directly for downstream cost rather than pure prediction error (Kotary et al., 2021; Lahoud et al., 2025; Mandi et al., 2024). Gradientbased differentiable PaO embeds solvers as differentiable layers for end-to-end training, including KKT-based OptNet (Amos & Kolter, 2017; Donti et al., 2017; Ferber et al., 2020) and interior-point regularized methods (Mandi & Guns, 2020); both require problem-specific derivation and are sensitive to downstream model complexity. Surrogate loss and perturbation approaches avoid explicit differentiation: SPO constructs surrogate losses via duality theory (Elmachtoub & Grigas, 2022), and perturbation-based schemes estimate gradients from solver outputs for black-box compatibility (Berthet et al., 2020; Pogančić et al., 2020; Shah et al., 2022), trading generality for reduced gradient fidelity. Non-gradient discrete PaO bypasses gradient computation and optimizes decision loss directly (Elmachtoub et al., 2020; Ban & Rudin, 2019), but is limited to simple architectures and lacks high-dimensional scalability. Engineering techniques including warm-starting, approximate solvers, and pre-computed surrogates cut PaO training cost (Mandi et al., 2020; Wang et al., 2020; Kong et al., 2022). More recently, (Mandi et al., 2025) propose losses that balance infeasibility and suboptimality without restricting the downstream problem to LP/MIP.

## A.2 IMPLEMENTATION DETAILS

Table 4: Hyperparameters and configurations.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Reward weight β</td><td>Swept in  $\{ 0 . 0 , 0 . 1 , 0 . 2 , \ldots , 1 . 0 \}$  Swept in  $\dot { \{ 1 0 ^ { - 7 } , 5 { \times } 1 0 ^ { - 7 } , 1 0 ^ { - 6 } , 5 { \times } 1 0 ^ { - 6 } } $ </td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 5 } , 5 { \times } \dot { 1 } 0 ^ { - 5 } , 1 0 ^ { - 4 } , 5 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ </td></tr><tr><td>Discount factor γ</td><td>0.99</td></tr><tr><td>PPO clip range</td><td>0.2</td></tr><tr><td>GAE parameter λ Value function coefficient v f</td><td>0.95</td></tr><tr><td>Max gradient norm</td><td>0.5</td></tr><tr><td>Epochs per policy update</td><td>0.5 10</td></tr><tr><td>Rollout steps per update</td><td>2048</td></tr><tr><td>Mini-batch size</td><td>64</td></tr><tr><td>Hidden layer size (actor / critic)</td><td>64</td></tr><tr><td>Number of hidden layers (actor / critic)</td><td>2</td></tr><tr><td>Base random seed</td><td>42</td></tr><tr><td>Number of independent random seeds</td><td>5</td></tr><tr><td>Steps per training episode</td><td>1096</td></tr><tr><td>MILP relative optimality gap</td><td>0.01</td></tr><tr><td>MILP time limit per solve</td><td>0.3 s</td></tr></table>

Dataset and Environment. The electricity price, solar irradiance, ambient temperature data etc. are sourced from public datasets.² The training set spans years 2016–2018, and the held-out test set covers year 2019. All data are at 2-hour resolution, yielding 12 time steps per day for the day-ahead scheduling horizon.

Optimization Model. The day-ahead energy scheduling problem is formulated as a mixed-integer linear program (MILP). The objective minimizes total operational cost. The model is solved via the SCIP solver, with a 1% relative optimality gap and a 0.3 s time limit per solve to ensure computational efficiency during reinforcement learning training.

RL-PaO Agent. The RL agent is built on the Proximal Policy Optimization (PPO) algorithm with a standard actor-critic MLP architecture. Raw high-dimensional scheduling solution vectors from the MILP solver are compressed into low-dimensional observations via principal component analysis (PCA). The composite reward function is defined as $r = - \beta \cdot J _ { \mathrm { c o s t } } - ( 1 - \beta )$ · JMSE, where $\beta \in [ \bar { 0 } , 1 ]$ balances the operational cost term and the prediction error term.

Training and Hyperparameter Sweep. We sweep the reward weight $\beta \in [ 0 , 1 ]$ and the policy learning rate lr $\bar { \in } \bar { [ 1 0 ^ { - 7 } , 1 0 ^ { - 3 } ] }$ . All results are averaged over 5 independent random-seed runs, with min-max ranges shown as shaded areas in the figures. The full framework is implemented in Python. Each training step corresponds to one day-ahead scheduling instance. One full training episode comprises 1096 steps, forming a complete sequential pass over the entire 2016–2018 training set.

## A.3 ADDITIONAL RESULTS

This section provides supplementary daily cost breakdowns and detailed ablation analyses to support the findings presented in the main paper.

Daily Operation Cost by Month (DO/RO/SO Baselines)  
![](images/e61cf5a248276d78dd560cff4c5f4fbbdb6be47759e1f4400e4bf2ae8363ee81.jpg)  
Figure C1: Daily operation cost by month compared with conventional optimization baselines on the 2019 test set. Each subplot corresponds to one calendar month, comparing Oracle, DO, RO, SO and the proposed RL-PaO.

Figure C1 decomposes the daily operational cost on a per-month basis over the 2019 test set, comparing the proposed RL-PaO with three conventional DO, RO, and SO, as well as the full-information Oracle lower bound. The DO baseline shows the highest cost volatility and overall cost level, with particularly large deviations. RO and SO deliver more stable performance but still remain significantly above the Oracle benchmark. By contrast, RL-PaO closely follows the Oracle costs across all twelve months, maintaining near-ideal performance under diverse seasonal PV generation conditions.

## Daily Operation Cost by Month (PtO/Supervised PaO Baselines)

![](images/6df7510e5831d89f8ea205063980958bb16d723e567b4d30f30274d21534cac6.jpg)  
Figure C2: Daily operation cost by month compared with predict-and-optimize baselines on the 2019 test set. Each subplot corresponds to one calendar month, comparing Oracle, predict-thenoptimize (PtO), supervised PaO, improved PaO and the proposed RL-PaO.

Figure C2 presents the monthly daily cost comparison against P-O baselines, including the standard two-stage PtO framework and supervised PaO. The PtO approach exhibits extreme cost spikes in spring and summer months (April–June), reflecting the unstableness of pure prediction-error-driven training under constraint uncertainty. Supervised PaO improves upon PtO but still suffers from notable performance fluctuations. The proposed RL-PaO achieves cost levels closest to the Oracle across all months without anomalous peaks, demonstrating substantially stronger robustness than both supervised prediction-optimization baselines.

![](images/4fccd5a12069d345723cf8908565be4cbbaaa9af76f9e93beb00598941f0b77e.jpg)  
Figure C3: Convergence behavior of RL-PaO across different reward weights β and learning rates. Top: mean reward convergence across training episodes. Bottom: annual operation cost convergence across training episodes. Each column corresponds to a fixed β value, and each colored curve corresponds to a different learning rate.

Figure C3 presents the full convergence behavior of RL-PaO across a grid of reward weights $\beta$ and learning rates, with mean reward shown in the top row and annual operation cost in the bottom row. Learning rate dominates convergence speed: configurations with very low learning rates fail to converge within the training horizon, while moderate to high rates converge within 500 episodes. Reward weighting $\beta$ determines the final performance level: higher $\beta$ consistently yields better final reward and lower operational cost across all learning rates. This figure is also presented and discussed in the main experimental section.

![](images/cbdd4b6215388f2caabcb3cae64506315c345fbb98c33f6fbc34a0cf3785a7ec.jpg)

![](images/0218ff3d3919bca325acb02e2f739a245af4ac742ee8df3c5495c60097ae0637.jpg)  
Figure C4: Left: Reward and operation cost convergence under different learning rates with $\beta$ fixed at 0.90. Each colored band corresponds to one learning rate configuration. Right: Reward convergence under different reward weights $\beta$ with learning rate fixed at $5 \times 1 0 ^ { - 5 }$ . Each colored band corresponds to one $\beta$ configuration.

Figures C4 present mean reward convergence and annual operation cost convergence, respectively, across different learning rates with $\beta$ fixed at 0.90. Learning rate is the dominant factor governing convergence speed: very low learning rates $( \mathrm { e . g . , 1 0 ^ { - 7 } } )$ converge extremely slowly and fail to approach the optimal performance level even after 3000 training episodes. Convergence accelerates markedly as the learning rate increases, with moderate rates in the $5 \times 1 0 ^ { - 5 } \mathrm { t o } 1 0 ^ { - 4 }$ range reaching a stable performance plateau within approximately 500 episodes. In terms of final solution quality, moderate learning rates yield the highest mean reward (Figure C4) and the lowest operational cost (Figure C5) with the tightest variance across random seeds. Excessively high learning rates $( \mathrm { e . g . }$ $1 0 ^ { - 3 } )$ do not further speed up convergence, but instead cause marginal performance degradation and larger fluctuation, as overly large parameter update steps tend to overshoot the optimal policy. This jointly confirms that the learning rate imposes a trade-off between convergence speed and final scheduling performance, and the $5 \times 1 0 ^ { - 5 }$ setting used in the main experiment strikes the optimal balance between fast convergence, near-optimal cost, and training stability.

![](images/686db9d88798159a727b868af6995abc035445026429336cd98f40c122b61221.jpg)

![](images/749257e053dea46b5dd49693e3c9d261e905e5392db0ae2e0bfebfe93ed0d6b5.jpg)  
Figure C5: Left: Annual operation cost convergence under different reward weights $\beta$ with learning rate fixed at $5 \times 1 0 ^ { - 5 }$ . Each colored band corresponds to one $\beta$ configuration. Right: Annual operation cost convergence under different learning rates with $\beta$ fixed at 0.90. Each colored band corresponds to one learning rate configuration.

Figures C5 present reward convergence and annual operation cost convergence, respectively, across different reward weights $\beta$ with the learning rate fixed at $5 \times 1 0 ^ { - 5 }$ . All $\beta$ configurations exhibit similar convergence speeds, reaching a stable performance plateau around 500 training episodes, indicating that the cost weighting coefficient barely affects the convergence rate of the RL training pipeline. In terms of final performance, there is a consistent monotonic trend: higher $\beta$ values yield higher mean reward (Figure C6) and lower operational cost (Figure C7) as the model places greater emphasis on minimizing downstream scheduling cost. Notably, the performance gain diminishes as $\beta$ approaches 1.0: $\beta = 0 . 9$ already achieves near-optimal cost levels comparable to $\beta = 1 . 0 $ , while maintaining tighter variance across random seeds. This jointly confirms that cost-oriented reward weighting effectively improves decision quality, and $\beta = 0 . 9$ strikes the optimal balance between final scheduling performance and training stability.

![](images/b9df7cee315246abda92ac0bbd5f1d99118be98ef7047f4a7cd32364467feb03.jpg)  
Figure C6: Final annual operation cost with mean and standard deviation across different learning rates, with $\beta$ fixed at 0.90.

Figure C6 quantifies the final annual operational cost (mean and standard deviation) across different learning rates with $\beta$ fixed at 0.90. The cost stays at an extremely high level when the learning rate falls below $1 0 ^ { - 6 }$ , as the algorithm fails to converge sufficiently within the training horizon. It drops to the lowest plateau across the $1 0 ^ { - 5 }$ to $1 0 ^ { - 4 }$ range, and climbs slowly as the learning rate increases further past $5 \times 1 0 ^ { - 5 }$ . This pattern indicates that the learning rate is not the higher the better: excessively high learning rates produce overly large parameter update steps that tend to overshoot the optimal solution and induce training oscillation, which leads to marginally suboptimal final scheduling performance and degraded stability. The $5 \times 1 0 ^ { - 5 }$ setting adopted in the main experiment achieves the balance between convergence speed and solution quality.

Figure C7 quantifies the final annual operational cost (mean and standard deviation) across different reward weights $\beta$ with the learning rate fixed at $5 \times 1 0 ^ { - 5 }$ . Overall, the cost decreases monotonically as $\beta$ increases from 0.0 to 0.9, reaching the minimum (optimal) level at $\beta = 0 . 9$ . However, the cost rebounds slightly when $\beta$ further increases to 1.0. This pattern indicates that excessively weighting the operational cost term while compromising the reward regularization component degrades training stability and leads to marginally suboptimal final performance. The dashed line marks the $\beta = 0 . 9$ value adopted in the main experiment, which achieves the balance between cost-oriented decision optimization and training stability.

![](images/d34228d2613ecafd042bc78ca872893f3765c8898c72843c262b623bdfcd7925.jpg)  
Figure C7: Final annual operation cost with mean and standard deviation across different reward weights $\beta ,$ with learning rate fixed at $5 \times 1 0 ^ { - 5 }$ . The dashed line marks the $\beta$ value adopted in the main experiment.