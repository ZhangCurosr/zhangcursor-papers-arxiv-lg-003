# Safe Meta-Policy Design with Risk Control

Wenbin Zhou<sup>1</sup>, Michael Lingzhi Li<sup>2</sup>, and Shixiang Zhu<sup>1</sup>

<sup>1</sup>Heinz College of Information Systems and Public Policy, Carnegie Mellon University <sup>2</sup>Technology and Operations Management, Harvard Business School

## Abstract

Models can be retrained as new data arrive, but deploying every new version risks replacing a good policy with a worse one. We study how to plan policy updates (i.e., meta-policy) before future candidates are trained, balancing the benefits of improvement against the risk of performance regression. Our ofline meta-policy maximizes expected cumulative value subject to a budget on the expected number of updates that perform worse than the policies they replace. We estimate the value and risk of possible switches from historical learning trajectories, represent an update schedule as a path in a directed acyclic graph, and select a schedule using dynamic programming. A leading-order analysis identifies the signal-to-noise ratio of policy improvement as a key driver of update frequency, waiting times, and risk allocation: clearer improvements support earlier, more frequent updates, while noisier improvements call for longer waits or greater risk expenditure. Their asymptotic rates also reveal a diminishing marginal cost of achieving greater safety over time. Experiments on synthetic and clinical trial data illustrate the performance–risk tradeof and compare our method with alternative baselines.

## 1 Introduction

Machine learning systems are often retrained as new data become available, but a newly trained model need not be better than the one currently deployed. Organizations must therefore decide both how to learn from additional data and when to put the resulting models into use. These decisions arise in applications such as treatment recommendations, clinical decision support, lending, and pricing, where model updates change the decisions made for users (Qian and Murphy, 2011; Zhao et al., 2012; Athey and Wager, 2021).

Choosing when to update involves a clear tradeof. Although additional data may improve performance in expectation, sampling variability can produce a candidate whose performance is lower than that of the deployed model (Pirotta et al., 2013). As illustrated in Figure 1, frequent updates deliver improvements sooner but create more opportunities for performance regression. Waiting reduces this exposure at the cost of delaying useful improvements. The challenge is to choose a deployment schedule that balances these gains and risks (Zhou et al., 2026; Zhou and Zhu, 2026).

In many applications, this deployment process must be specified before the performance of future candidate models is known. For example, the U.S. Food and Drug Administration’s guidance on Predetermined Change Control Plans allows manufacturers of AI-enabled medical devices to specify how anticipated modifications will be developed and evaluated before implementation (U.S. Food and Drug Administration, 2025). Motivated by such settings, we study an ofline planning problem in which models are trained sequentially and their deployment times are chosen in advance. We call this schedule a meta-policy and define an update as unsafe if it lowers average policy value in the target population. Our goal is to maximize expected cumulative deployed value subject to a budget on the expected number of unsafe updates.

Each update changes the deployed policy and therefore the risk of later updates. Deploying a candidate makes it the benchmark for the next replacement, whereas skipping it leaves the current policy in place. We therefore optimize the deployment schedule as a whole. When candidate values improve in expectation, we show that an optimal schedule deploys the newest available candidate whenever it updates. This result allows us to represent a schedule as a path through candidate training times. Each edge records the expected value earned while a policy remains deployed and the risk of replacing it at the next update. We estimate these quantities from historical learning trajectories and use dynamic programming to select the path with the highest expected value subject to the safety budget (Beasley and Christofides, 1989).

We also study how the learning dynamics determine the optimal schedule. Under a time-inhomogeneous Itô difusion model for policy value (Itô, 1951), we derive a leading-order continuous approximation in which update intervals are inversely proportional to the squared local signal-to-noise ratio (SNR) of policy improvement: More reliable improvements support more frequent updates, while noisier improvements require longer waits. For a horizon of T periods and safety budget ϵ, the analysis yields regret of order log $( T / \epsilon ) / T$ and an update count of order $T / \log ( T / \epsilon )$ , with risk allocated adaptively according to the local learning dynamics. These rates reveal a favorable scaling law: as the horizon grows, the meta-policy can deploy an increasing number of policy updates while the average value lost from delayed deployment still vanishes. At the same time, stricter safety requirements enter only through $\log ( 1 / \epsilon )$ , so multiplicative reductions in the allowable risk lead to only additive increases in waiting time and a correspondingly mild loss in deployed value. In sum, these results show that substantial gains in safety can be achieved without sacrificing responsiveness at the same rate.

![](images/6a3bec85045c82aefc47edd45a70e99b3bde992398e0ec968adee85fe43cd927.jpg)  
Figure 1: The core tradeof in deciding how often to update a policy. Updating slowly (left) leaves an outdated policy in place and loses value, while updating quickly (right) tracks improvements closely but risks deploying a policy worse than the one it replaces (a safety violation).

Experiments in synthetic learning environments and on International Stroke Trial data (International Stroke Trial Collaborative Group, 1997; Sandercock et al., 2011) illustrate these tradeofs. The proposed schedules achieve higher value under the estimated safety budget than periodic updating and alternatives that restrict update timing or risk allocation. The gains are most pronounced when the rate and variability of policy improvement change over time, making a fixed update interval less efective.

Related Work Safe policy improvement seeks to replace a baseline policy only when the available evidence supports non-inferiority or improvement. Early approaches certify a learned candidate using held-out data or construct conservative updates from policyimprovement bounds (Thomas et al., 2015; Laroche et al., 2019), while conservative bandit methods maintain performance relative to a default policy during learning (Wu et al., 2016). Recent work has strengthened this perspective by optimizing lower-tail performance guarantees (Zhao et al., 2024b), calibrating multiple tests over candidate threshold policies (Cho et al., 2025), restricting deviations to well-supported decision points (Sharma et al., 2025), and conformally regulating deviations from a safe reference policy (Prinster et al., 2026).

While these methods primarily construct or certify an individual policy relative to a baseline, our framework takes a sequence of data-dependent candidate policies as given and jointly selects their deployment times, thereby extending safe policy improvement from a single-update decision to a sequential deployment problem.

A complementary literature studies sequential learning when changing the deployed policy is costly or when only a limited number of data-collection batches is available. Low-switching reinforcement-learning algorithms control regret while limiting policy changes (Bai et al., 2019), and deployment-eficient methods minimize the number of distinct policies required to learn a high-performing policy (Matsushima et al., 2021; Huang et al., 2022). Recent work extends these guarantees to broad sequential decision classes and batch-learning constraints (Xiong et al., 2024), generalized linear contextual bandits with update budgets (Sawarni et al., 2024), reinforcement learning with general function approximation (Zhao et al., 2024a), and reward-free exploration with nearoptimal deployment complexity (Zhang et al., 2025). Whereas this literature generally minimizes regret or sample complexity under an exogenous switching or deployment constraint, our meta-policy makes the number and timing of updates endogenous and explicitly weighs cumulative value against the probability that each deployed candidate performs worse than its incumbent, giving rise to a distinct set of structural properties.

Our computational formulation is related to the resource-constrained shortest-path problem, for which classical work developed exact dynamic programs and approximation schemes (Handler and Zang, 1980; Beasley and Christofides, 1989; Hassin, 1992; Lorenz and Raz, 2001). Recent advances include multiphase dynamic programming for acyclic networks (Himmich et al., 2024), enhanced bidirectional heuristic search (Ahmadi et al., 2025b), and faster label-setting methods that accommodate multiple or signed resource quantities (Ahmadi et al., 2025a). Our theoretical analysis is also connected to continuous-time approximations of stochastic-gradient algorithms, including stochastic modified equations and local difusion models (Li et al., 2017; Mandt et al., 2017); recent refinements study stochastic modified flows and the weak accuracy of alternative stochastic diferential equation approximations (Gess et al., 2024; Ankirchner and Perko, 2024). Unlike generic path solvers or diffusion analyses of parameter iterates, our contribution derives a problem-specific graph whose edges encode estimated deployment value and unsafe-update risk, and then transports the learning dynamics to the policy-value process to characterize the optimal number, timing, and risk allocation of policy updates.

## 2 Problem Setup

We consider sequential policy learning over a finite horizon of $T \geq 2$ periods, indexed by $\mathcal { T } = \{ 1 , . . . , T \}$ . In period t, the decision-maker observes $( X _ { t } , A _ { t } , Y _ { t } )$ , where $X _ { t } \in \mathcal { X }$ denotes pre-treatment covariates, $A _ { t } \in \{ 0 , 1 \}$ is the assigned treatment, and $Y _ { t } \in \mathbb { R }$ is the observed outcome. Under the potential outcomes framework (Rubin, 1974), let $Y _ { t } ( 0 )$ and $Y _ { t } ( 1 )$ denote the outcomes under the two treatments, with consistency requiring $Y _ { t } = Y _ { t } ( A _ { t } )$ We impose the following assumption on the underlying population (Jia et al., 2024).

Assumption 1. The covariates and potential outcomes are drawn i.i.d. from $\mathcal { P } , i . e .$

$$
\{ ( X _ { t } , Y _ { t } ( 0 ) , Y _ { t } ( 1 ) ) \} _ { t = 1 } ^ { T } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { P } .
$$

This assumption concerns the covariates and potential outcomes; treatment assignments need not be independent across periods.

A policy is a mapping $\pi : \mathcal { X }  \{ 0 , 1 \}$ , with value

$$
V ( \pi ) = \mathbb { E } _ { \mathcal { P } } [ Y ( 1 ) \mathbb { 1 } \left\{ \pi ( X ) = 1 \right\} + Y ( 0 ) \mathbb { 1 } \left\{ \pi ( X ) = 0 \right\} ] .
$$

At each period t, the decision-maker trains a candidate policy $\pi _ { t }$ using the observations accumulated through that period, $\mathcal { D } _ { t } = \{ ( X _ { \tau } , A _ { \tau } , Y _ { \tau } ) \} _ { \tau = 1 } ^ { t }$ . For a learned policy, $V ( \pi _ { t } )$ evaluates its population performance conditional on the realized training data.

Candidate training and deployment are governed separately. A meta-policy m specifies which candidate is deployed in each period: the deployed policy at time t is $\pi _ { m ( t ) }$ . We consider schedules fixed in advance, with feasible class

$$
\mathcal { M } = \{ m : \mathcal { T }  \mathcal { T } : m ( t ) \leq t \mathrm { ~ f o r ~ a l l ~ } t \in \mathcal { T } \} .
$$

The restriction $m ( t ) \leq t$ ensures that a candidate is trained before it is deployed. We assume an $o f f i n e$ setting in which deployment decisions do not afect the data-generating process used to train future candidates. Figure 2 illustrates an example of meta-policy.

![](images/044cefc409d40a5c30155629f686e4d3580d4dbd8ad490b906db080a078e4906.jpg)  
Figure 2: A deployment schedule and its path representation for $T = 4 .$ Left: The meta-policy updates at times 2 and 4, retaining $\pi _ { 2 }$ at time 3. Right: The corresponding path $1  2  4  5$ deploys $\pi _ { 1 } , \pi _ { 2 } , \pi _ { 4 }$ , with node $5 = T + 1$ serving as the terminal sink.

An update is unsafe if the newly deployed policy has lower population value than its incumbent. Given a safety budget $\epsilon > 0$ , we choose a meta-policy to maximize expected average deployed value while bounding the expected number of unsafe updates:

$$
\begin{array} { r l } { \underset { m \in \mathcal { M } } { \operatorname* { m a x } } } & { \mathbb { E } \Bigg [ \cfrac { 1 } { T } \sum _ { t = 1 } ^ { T } V \big ( \pi _ { m ( t ) } \big ) \Bigg ] } \\ { \mathrm { s . t . ~ } } & { \mathbb { E } \Bigg [ \displaystyle \sum _ { t = 2 } ^ { T } \mathbb { 1 } \big \{ V \big ( \pi _ { m ( t ) } \big ) < V \big ( \pi _ { m ( t - 1 ) } \big ) \big \} \Bigg ] \leq \epsilon . } \end{array}\tag{1}
$$

Here, the outer expectation is over the training data $\mathcal { D } _ { T }$ , which generates the candidate-policy sequence. Appendix A discusses extensions beyond the binarytreatment causal setting.

## 3 Proposed Algorithm

We reformulate problem (1) as a resource-constrained path problem and develop a dynamic programming algorithm using data-driven estimates of policy values and update risks.

## 3.1 Path Selection Reformulation

The reformulation relies on the following assumption that candidate-policy values improve in expectation, motivated by policy-improvement methods (Pirotta et al., 2013; Schulman et al., 2015).

Assumption 2 (Policy improvement). For all $t ^ { \prime } , t \in \mathcal { T }$ with $t ^ { \prime } < t , \mathbb { E } [ V ( \pi _ { t ^ { \prime } } ) ] \leq \mathbb { E } [ V ( \pi _ { t } ) ]$

Assumption 2 allows individual candidates to perform worse than their predecessors; it requires only that their expected values be nondecreasing. It therefore permits unsafe updates, unlike pathwise monotonicity.

The following result shows that an optimal schedule can be chosen to deploy the newest candidate whenever an update occurs.

Lemma 1 (Newest-policy reduction). Under Assumption 2, problem (1) admits an optimizer m<sup>⋆</sup> satisfying

$$
m ^ { \star } ( t ) \in \{ m ^ { \star } ( t - 1 ) , t \} , \qquad t \in \mathcal { T } \setminus \{ 1 \} .
$$

By Lemma 1, an optimal meta-policy can be specified entirely by its update times: once these times are chosen, the deployed candidate at each update is determined. Write $1 = t _ { 0 } < t _ { 1 } < \cdots < t _ { K } \leq T$ for the initial deployment and subsequent updates, and set $t _ { K + 1 } = T +$ 1 to mark the end of the horizon. The schedule deploys $\pi _ { t _ { k } }$ during periods $t _ { k } , \ldots , t _ { k + 1 } - 1$ , for $k = 0 , \ldots , K$ Thus, each consecutive pair $\left( { t _ { k } , t _ { k + 1 } } \right)$ specifies a holding interval and, except for the final pair, the update that follows it.

This interval representation provides a natural connection to path optimization. Represent each candidate’s training time by a vertex and each possible holding interval by a directed edge. A deployment schedule then corresponds to the path $t _ { 0 } \to t _ { 1 } \to \cdots \to t _ { K } \to T + 1$ as illustrated by Figure 2. Assigning each edge its contribution to expected deployed value and its unsafe-update probability makes both the objective and the safety constraint additive along the path. Formally, construct a directed acyclic graph $\mathcal { G } = ( \nu , \mathcal { E } )$ with

$$
\mathcal { V } = \{ 1 , \ldots , T + 1 \} , \quad \mathcal { E } = \{ ( i , j ) : 1 \leq i < j \leq T + 1 \} .
$$

For $j \le T$ , an edge $( i , j )$ represents deploying $\pi _ { i }$ during periods $i , \ldots , j - 1$ and switching to $\pi _ { j }$ at time $j .$ The terminal edge $( i , T + 1 )$ represents retaining $\pi _ { i }$ through period $T ,$ , with no further update. Let $\mathcal { P } _ { i j }$ denote the directed paths from i to $j ,$ including the empty path when $i = j \colon$

$$
\begin{array} { r l } & { { \mathscr { P } } _ { i j } : = \big \{ \big ( v _ { 0 } , \dotsc , \dotsc , v _ { K } \big ) : K \ge 1 , } \\ & { \qquad v _ { 0 } = i , \ v _ { K } = j , \ \big ( v _ { \ell - 1 } , v _ { \ell } \big ) \in { \mathcal { E } } , \forall \ell \} . } \end{array}\tag{2}
$$

Each edge receives a reward equal to its contribution to expected average deployed value and a safety cost equal to its unsafe-update probability.

Proposition 1 (Path selection problem). Define $\mathcal { P } _ { i j }$ as in (2). Under Assumption ${ \mathit { 2 } } ,$ problem (1) is equivalent to

$$
\operatorname* { m a x } _ { p \in \mathscr { P } _ { 1 , T + 1 } } \quad \sum _ { ( i , j ) \in p } R _ { i j } \quad \mathrm { s . t . } \quad \sum _ { ( i , j ) \in p } S _ { i j } \leq \epsilon .\tag{3}
$$

where $\pmb { R } \in \mathbb { R } ^ { ( T + 1 ) \times ( T + 1 ) }$ and $S \in \mathbb { R } ^ { ( T + 1 ) \times ( T + 1 ) }$ denote the reward and safety cost matrices,

$$
\begin{array} { r l } & { R _ { i j } : = \displaystyle \frac { j - i } { T } \cdot \mathbb { E } [ V ( \pi _ { i } ) ] , } \\ & { S _ { i j } : = \displaystyle  \mathbb { P } ( V ( \pi _ { j } ) < V ( \pi _ { i } ) ) , \quad i f j \leq T  } \\ & {  i f j = T + 1 . } \end{array}
$$

Thus, an optimal schedule corresponds to a maximumreward source–sink path subject to an additive safety budget. This is the maximization form of the resourceconstrained shortest-path problem (Handler and Zang, 1980; Beasley and Christofides, 1989). The safety costs add by linearity of expectation, without requiring independence between unsafe-update events.

## 3.2 Dynamic Programming Solution

We solve the path selection problem using plug-in edge estimates and a discretized safety budget. The resulting dynamic program is exact for the estimated, gridded problem. Algorithm 1 provides the pseudocode.

Step 1: Estimating Reward and Cost We estimate the edge rewards and safety costs using N ofline trajectories,

$$
\mathcal { D } ^ { ( n ) } = \{ ( \boldsymbol { X } _ { t } ^ { ( n ) } , \boldsymbol { A } _ { t } ^ { ( n ) } , \boldsymbol { Y } _ { t } ^ { ( n ) } ) \} _ { t = 1 } ^ { T } , \qquad n = 1 , \ldots , N .
$$

For each trajectory n, we train $\pi _ { i } ^ { ( n ) }$ on the first i observations using the same learning procedure that generates the candidate policies, and denote its estimated population value by $\widehat V _ { i } ^ { ( n ) }$ . The choice of policy-learning and value-estimation methods depends on how the data are collected (Dudík et al., 2014; Zhan et al., 2024; Jia et al., 2024). Averaging across trajectories gives

$$
\begin{array} { l } { \displaystyle \widehat { \pmb { R } } _ { i j } = \frac { j - i } { N T } \sum _ { n = 1 } ^ { N } \widehat { V } _ { i } ^ { ( n ) } , } \\ { \displaystyle \widehat { \pmb { S } } _ { i j } =  \frac { 1 } { N } \sum _ { n = 1 } ^ { N } { \mathbb 1 }  \widehat { V } _ { j } ^ { ( n ) } < \widehat { V } _ { i } ^ { ( n ) }  , \quad j \leq T ,  } \\ { \displaystyle 0 , \qquad } & {  j = T + 1 . } \end{array}\tag{4}
$$

Estimation error can cause a schedule that satisfies the empirical safety budget to violate the population constraint. Population risk control therefore requires accounting for uncertainty in the estimated safety costs.

Step 2: Dynamic Programming To solve the estimated path problem, we first discretize the safety budget into units of size $\Delta > 0$ . We round the available budget down and each edge cost up:

$$
B _ { \Delta } = \left\lfloor \frac { \epsilon } { \Delta } \right\rfloor , \qquad \widetilde { \cal S } _ { i j } = \left\lceil \frac { \widehat { \pmb { S } } _ { i j } } { \Delta } \right\rceil .\tag{5}
$$

This guarantees that any path using at most $B _ { \Delta }$ budget units satisfies the estimated safety constraint:

$$
\sum _ { ( i , j ) \in p } \widehat { \pmb { S } } _ { i j } \leq \Delta \sum _ { ( i , j ) \in p } \widetilde { \pmb { S } } _ { i j } \leq \Delta B _ { \Delta } \leq \epsilon .
$$

A smaller $\Delta$ reduces the error introduced by rounding but increases the computational cost.

For each node $j$ and budget $b \in \{ 0 , \ldots , B _ { \Delta } \}$ , let $F ( \boldsymbol { j } , \boldsymbol { b } )$ denote the largest estimated reward achievable

Algorithm 1 Safe Meta-Policy Optimization   
Require: Ofline trajectories D; time horizon $T ;$ safety   
budget ϵ; grid width $\Delta > 0 .$   
1: Construct the update DAG $\mathcal { G } = ( \nu , \mathcal { E } )$ with source   
1 and destination $T + 1$ as in (3).   
2: Estimate $\widehat { \boldsymbol { R } }$ and $\widehat { \boldsymbol { S } }$ according to (4). For each termi  
nal edge $( i , T + 1 )$ , set $\widehat { \pmb { S } } _ { i , T + 1 }  0 .$   
3: Compute $B _ { \Delta }$ and $\widetilde { S } _ { i j }$ according to (5).   
4: Initialize $F ( 1 , b )  \mathit { \mathrm { ~ 0 ~ } }$ for all $b \in \{ 0 , \ldots , B _ { \Delta } \}$ and   
$F ( j , b )  - \infty$ for all $j \neq 1$ and $b \in \{ 0 , \ldots , B _ { \Delta } \}$   
5: for each vertex $j \neq 1$ in topological order do   
6: for $b = 0 , \ldots , B _ { \Delta }$ do   
7: $F ( j , b ) \gets \operatorname* { m a x } _ { ( i , j ) \in \mathcal { E } , \ : \widetilde { S } _ { i j } \leq b } \Big \{ F \Big ( i , b - \widetilde { S } _ { i j } \Big ) + \widehat { R } _ { i j } \Big \} .$   
8: Store max predecessor when $F ( j , b ) > - \infty .$   
9: end for   
10: end for   
11: Recover optimal path $\hat { p } ^ { \star }$ by backtracking from $( T +$   
1, $B _ { \Delta } )$ through the stored predecessors.   
12: return The meta-policy corresponding to $\hat { p } ^ { \star }$   
by a path from node 1 to node $j$ using at most b budget   
units. Initialize $F ( 1 , b ) = 0$ for all $b ,$ and set the value of   
any infeasible state $\mathrm { t o \mathrm { ~ - \infty } }$ . A path reaching j through   
edge $( i , j )$ earns reward $\hat { R } _ { i j }$ and leaves at most $b - \tilde { S } _ { i j }$   
budget units for the preceding path. For $j = 2 , \ldots , T + 1$   
maximizing over predecessor nodes gives the Bellman   
recursion   
(6)

$$
F ( j , b ) = \operatorname* { m a x } _ { ( i , j ) \in \mathcal { E } } \left\{ F \Big ( i , b - \widetilde { S } _ { i j } \Big ) + \widehat { R } _ { i j } \right\} .
$$

Because every edge connects an earlier node to a later one, we evaluate these states in increasing order of j. The optimal value of the discretized problem is $F ( T + 1 , B _ { \Delta } )$ Recording the selected predecessor at each state allows us to recover the optimal path and its deployment schedule by backtracking from $( T + 1 , B _ { \Delta } )$

Given the edge estimates, the algorithm requires $\mathcal { O } ( T ^ { 2 } ( B _ { \Delta } + 1 ) )$ time and $\mathcal { O } ( T ^ { 2 } + T ( B _ { \Delta } + 1 ) )$ memory, including storage for edge estimates, state values, and predecessors. Appendix B.3 provides further discussion of grid-width selection, approximation error, and an extension that accounts for uncertainty in the estimated safety costs.

## 4 Theoretical Analysis

This section characterizes and interprets the optimal update frequency, safety-risk allocation, and resulting deployment regret. All technical proofs are deferred to Appendix C.

Throughout our analysis, we assume that the policy value $V ( \pi _ { t } )$ as a time-inhomogeneous Itô difusion (Itô, 1951), with deterministic drift $\mu ( \cdot )$ and volatility $\sigma ( \cdot )$

Assumption 3 (Itô difusion). Let $\mu ( \cdot ) : [ 0 , 1 ] \to \mathbb { R } _ { + }$ and $\sigma ( \cdot ) : [ 0 , 1 ] \to \mathbb { R } _ { + }$ be two deterministic smooth functions, and $V _ { 0 }$ be a constant. We assume that

$$
V ( \pi _ { t } ) = V _ { 0 } + \frac { 1 } { T } \int _ { 0 } ^ { t } \mu \left( \frac { \tau } { T } \right) \mathrm { d } \tau + \frac { 1 } { T } \int _ { 0 } ^ { t } \sigma \left( \frac { \tau } { T } \right) \mathrm { d } W _ { \tau } ,
$$

where $\{ W _ { \tau } \} _ { \tau \geq 0 }$ is a standard Brownian motion.

This assumption is inspired by difusion-type approximation, which has been used in the literature to study stochastic-gradient learning dynamics (Li et al., 2019) and, more specifically, policy-gradient learning in stochastic bandits (Lattimore, 2026). It can be derived from more primitive assumptions with $\pi _ { t }$ itself, which we further discuss in Appendix C.1.

Assumption 3 allows us to derive a continuous-time leading-order approximation to the original problem asymptotically. We introduce a horizon-dependent waiting-time function $h _ { T } : [ 0 , 1 ] \to \mathbb { R } _ { + }$ as the continuum analogue of the discrete holding intervals $t _ { k + 1 } - t _ { k }$ Starting from $t _ { 0 } = 0$ , the associated continuous schedule is generated recursively by

$$
t _ { k + 1 } = t _ { k } + h _ { T } ( u _ { k } ) , \qquad u _ { k } : = t _ { k } / T ,\tag{7}
$$

for as long as the proposed next update satisfies $t _ { k + 1 } <$ T. Once the next proposed update would reach or exceed $T ,$ no further update is made, and the currently deployed policy is retained until the terminal time T. The update interval $h _ { T } ( u )$ is used in the approximated optimization problem as its decision variable.

Lemma 2 (Leading-order approximation). Under Assumption 3, as $T \to \infty ,$ Problem (1) admits the following continuous leading-order approximation:

$$
\begin{array} { r l } { \underset { h _ { T } ( \cdot ) } { \operatorname* { m i n } } } & { \int _ { 0 } ^ { 1 } \mu ( u ) h _ { T } ( u ) \mathrm { d } u } \\ { \mathrm { s . t . } } & { \int _ { 0 } ^ { 1 } \frac { \Phi \Big ( - \mu ( u ) \sqrt { h _ { T } ( u ) } / \sigma ( u ) \Big ) } { h _ { T } ( u ) } \mathrm { d } u \leq \frac { \epsilon } { T } . } \end{array}\tag{8}
$$

The objective of (8) is derived by replacing the original objective of maximizing cumulative reward with minimizing regret, where regret is measured relative to the always-update baseline,

$$
\mathcal { R } _ { T } ^ { \star } ( \epsilon ) : = \mathbb { E } \left[ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left\{ V ( \pi _ { t } ) - V ( \pi _ { m ^ { \star } ( t ) } ) \right\} \right] .\tag{9}
$$

The integrand in the constraint can be interpreted as a local risk-expenditure rate, where the numerator is the probability that an update is harmful, and $1 / h _ { T } ( u )$ captures the local update frequency. The constraint encourages frequent updates when learning is fast and stable, and longer waits when learning is slow or noisy.

Theorem 1 (Update interval). Define the following constants and functions,

$$
L _ { T } = \log \left( \frac { T } { \epsilon } \right) , \quad I ( u ) = \frac { \mu ( u ) ^ { 2 } } { 2 \sigma ( u ) ^ { 2 } } .
$$

Under the same assumptions as Lemma ${ \it 2 } ,$ there is

$$
h _ { T } ^ { \star } ( u ) = \frac { 1 } { I ( u ) } \Big [ L _ { T } - \frac { 3 } { 2 } \log L _ { T } + O ( 1 ) \Big ] .
$$

Theorem 1 characterizes how the optimal update interval adapts to the local learning dynamics. Since $I ( u )$ represents the squared local signal-to-noise ratio (SNR) and the leading term scales as $L _ { T } / I ( u )$ , updates should occur more frequently when expected improvement is strong relative to its variability, and less frequently when learning is noisy. The logarithmic correction $- \frac { 3 } { 2 }$ log $L _ { T }$ reflects the sharp-tail behavior governing unsafe-update risk.

Importantly, Theorem 1 reveals a favorable diminishing marginal cost of additional safety. Imposing stricter safety budgets increases the waiting time between updates only logarithmically through $L _ { T } = \log ( T / \epsilon )$ . This implies that when the safety budget is relatively large, tightening the budget can initially require noticeable increases in the update interval; but as the safety requirement becomes increasingly stringent, further multiplicative reductions in ϵ translate into only additive increases in $L _ { T }$ . Consequently, a substantial improvement in deployment safety can be achieved without a proportionally large reduction in update frequency, encouraging practitioners in high-risk settings to adopt conservative update schedules without sacrificing responsiveness at the same rate.

Corollary 1 (Risk per-update). Under the same assumptions and notations as in Theorem $^ { 1 , }$ let

$$
A ( u ) = \frac { \mu ( u ) } { I ( u ) } , \quad B = \int _ { 0 } ^ { 1 } A ( u ) \mathrm { d } u .
$$

Then, there is

$$
\mathbb { P } \left\{ V ( \pi _ { t _ { k + 1 } } ) < V ( \pi _ { t _ { k } } ) \right\} = \Theta \left( { \frac { \epsilon L _ { T } } { T B } } { \frac { \mu ( u _ { k } ) } { I ( u _ { k } ) ^ { 2 } } } \right) ,
$$

where $u _ { k } : = t _ { k } / T$ denotes the normalized update times.

Corollary 1 shows that the optimal schedule allocates the safety budget unevenly across updates according to the local learning dynamics, with the per-update risk scales with $\mu ( u ) \bar { / } I ( \bar { u } ) ^ { 2 }$ . Moreover, because the overall scale is $\epsilon L _ { T } / T$ , the risk associated with each individual update vanishes as the learning horizon grows, even while the number of updates increases. Hence, the optimal policy achieves global safety not by imposing a uniform risk level on every update, but by adaptively distributing risk across the trajectory.

Corollary 2 (Regret and number of updates). Let $R _ { T } ^ { \star } ( \epsilon )$ be defined in (9) and $K _ { T } ^ { \star }$ be the total number of updates generated according to $( 7 )$ . Under the same assumptions and notations of Theorem 1, there is

$$
\begin{array} { c } { { \displaystyle \mathcal { R } _ { T } ^ { \star } ( \epsilon ) = \frac { B } { 2 T } \left[ L _ { T } - \frac { 3 } { 2 } \log L _ { T } \right] + O ( T ^ { - 1 } ) , } } \\ { { \displaystyle K _ { T } ^ { \star } \sim \frac { T } { L _ { T } } \int _ { 0 } ^ { 1 } I ( u ) \mathrm { d } u . } } \end{array}
$$

Corollary 2 characterizes the aggregate cost and adaptivity of the optimal deployment schedule. In particular, the optimal regret vanishes at the rate $L _ { T } / T ,$ , while the number of updates grows on the order of $T / L _ { T }$ , showing that the system can accommodate an increasing number of policy updates over a longer learning horizon while incurring vanishing average deployment regret. The second-order terms further reveal that this tradeof de pends on the entire learning trajectory through $I ( u ) { \mathrm { : } }$ periods in which learning is more informative contribute diferently to both cumulative regret and the total num ber of deployments. Together, these results quantify how the locally adaptive update rule translates into a global balance between responsiveness and the cost of holding outdated policies.

## 5 Numerical Experiments

This section presents three sets of numerical experiments that compare the proposed meta-policy with update rules commonly adopted in practice and validate our theoretical analysis in Section 4.

Experiment setting. We evaluate our framework on both synthetic and real-data policy-learning environments. For the synthetic study, we consider three learning regimes with low, medium, and high valueprocess signal-to-noise ratios by varying the learning rate and outcome-noise level. We generate 400 independent trajectories of 300 sequential observations and retain a candidate policy every five observations, yielding $T = 6 0$ deployment checkpoints, five-dimensional covariates, randomized binary treatment, and hetero geneous treatment efects, and train an online plug-in CATE policy learner. For the real-data study, we use the randomized International Stroke Trial data (Sander cock et al., 2011), constructing 500 bootstrap sequential learning trajectories with nested training sets and evaluating learned policies on a fixed 20% held-out sample using an AIPW value estimator (Robins et al., 1994). Across these settings, we compare our proposed algorithm against (i) periodic updating at the best feasible constant interval, (ii) equal risk, which splits ϵ evenly across updates, and (iii) earliest safe, which deploys the first policy whose estimated safety cost falls below a tuned threshold. Further experimental details are provided in Appendix D.

Drift µ  
![](images/ed19e093c610d9385440c09bb7be916bee7e64d8a6adca50f001a840d5a0126e.jpg)

![](images/e5935a0f0145a82f4ed699134f9b1675c078f7aecd4e54e446e52fe8ab3af9fc.jpg)

![](images/e9f9c121bc2be237968aa3308b5f8600bc16e7135043e68c587022a5624c8e4d.jpg)

![](images/b0f01996ed202e47685aba5f6eca672dbd690008a9b2e7dbbf6ae5e881182864.jpg)  
Figure 3: Phase diagrams of the schedules returned by Algorithm 1 on homogeneous value processes with constant drift µ and volatility σ $\textsf { r } ( T = 5 1 2 , \epsilon = 0 . 3 5 , \Delta = 1 \times 1 0 ^ { - 3 } )$ . From left to right, the panels report the number of updates K, the mean per-update risk p¯, the mean normalized pre-update interval preceding an update h<sup>¯</sup>, and the expected deployed value J. Each cell is evaluated under the population rewards and safety costs and averaged over three simulated datasets. White-labeled contours show the leading-order theoretical predictions established in Section 4.

![](images/4aebac06c7fb5cbe203cc545d91f02b962a2831af266ef1513d2b228657cbc2e.jpg)

![](images/962f9ff8542f2cd516d746cc883e0dde204cea39b575fa31e1b09e927f5d0e99.jpg)

![](images/37c1e7a0c43fb9c9f0b16f3a0b7db3f23d9fc295849be5fefe5a5429d184cff1.jpg)

![](images/83cfd589c957a45569cd9ab9da98f5e15a46aa1d672c981bf61260d7ffcf917a.jpg)  
Figure 4: Safety–value Pareto frontiers of the proposed meta-policy (Ours) and three baseline update rules, in three synthetic settings with low, medium, and high value-process SNR and on the International Stroke Trial (IST) data. The horizontal axis is the safety budget ϵ (log scale). The vertical axis is the objective of (1), normalized so that 0 and 1 correspond to never updating and to updating every period. All methods use the same estimated rewards and safety costs.

![](images/a55c3c371fb805305ba7972373b1c080226bf6ef87a0752d33f6c572a7cdae65.jpg)

![](images/763bce5bdff81c0dba569a5a18fdc522074c66c43ec91d6e976c041b7f961c0e.jpg)

![](images/2e556fe1467f89bc1501e356cf81f7a90997f6eacc3e1c0b84431c55a796fa8c.jpg)

![](images/bbb0353e95b05f6c29fa1227c5a1f0e19ff6b347eb0cc1b76db7e6332389e0ee.jpg)  
Figure 5: Update times in the optimal schedule selected by the proposed meta-policy as the safety budget ϵ varies, in the same four settings as Figure 4. Each black dot indicates a scheduled normalized update time. The left dashed lines mark the smallest budget that admits one update.

Exp 1: Validating the optimality system. Figure 3 uses simulated homogeneous value processes with constant drift $\mu$ and volatility σ over a $5 0 \times 5 0$ grid of $( \mu , \sigma )$ , runs Algorithm 1 on the estimated rewards and safety costs, and overlays their theoretical asymptotic rates as white contours. It can be observed that a higher SNR yields more updates, each spending less risk, and admits shorter waits. On the other hand, in the low SNR regime, there might not even exist an update that fits within the budget. The empirical and theoretical surfaces are closely aligned (Pearson correlations of 0.99, 0.99, and 0.98 for K, p¯, and h<sup>¯</sup>). The experiment suggests that the theoretical structure characterized in Section 4 aligns well with the exact schedules numerically computed by Algorithm 1.

Exp 2: Safety–value frontier. Figure 4 plots the obtained objective value of cumulative reward across diferent specifications of the safety budget ϵ. It can be seen that our method weakly dominates every baseline at every budget, and its frontier exhibits the diminishing returns predicted by Corollary 2. The gap is largest against periodic updating, reaching 0.39 in the high-SNR setting, where our method captures 82% of the always-update gain at $\epsilon = 0 . 0 1$ versus 53% for periodic updating; the adaptive baselines trail by up to 0.06 (equal risk) and 0.23 (earliest safe). These gains arise because our method jointly chooses when to update and how much risk to spend on each update, rather than fixing either in advance. This result shows that the proposed meta-policy attains a higher safety–value frontier than commonly used update rules in all four settings, rendering it superior in decision-making settings with safety constraints.

Exp 3: Structure of optimal schedules. Figure 5 illustrates how the optimal schedule reorganizes as the safety budget grows, where each column shows the normalized update times selected under a given ϵ. Three observations can be made: (i) The schedule is not nested in ϵ: as the budget grows, existing updates shift earlier, and new ones are inserted, producing the curved branches in the medium- and high-SNR panels. This agrees with Corollary 1, where allocating more risk to an update shortens its waiting time. (ii) Update placement depends on the SNR: under high SNR, the first update occurs early $( t / T \approx 0 . 2 )$ , and most updates fall in the first half of the horizon, where value improves fastest, whereas under low SNR the first update is deferred to $t / T \in [ 0 . 3 9 , 0 . 7 5 ]$ . (iii) The intervals are nonuniform: on IST, updates concentrate at a few recurring times. Hence, the optimal meta-policy adapts both the timing and the risk of each update to local learning dynamics, which neither periodic nor equal-risk rules can replicate by construction.

## 6 Conclusion

We studied how to schedule the deployment of sequentially learned policies when each update may both improve or worsen performance. We formulated the problem as maximizing cumulative value under a budget on unsafe updates, reformulated it as a resource-constrained shortest-path problem, and developed an ofline dynamicprogramming solution using historical learning trajectories. Our asymptotic analysis characterizes the optimal update spacing, per-update risk, regret, and number of updates in terms of the local signal-to-noise ratio, revealing that stricter safety can be achieved with only a logarithmic increase in waiting time. Experiments on synthetic environments and real clinical data support these structural insights and demonstrate improved safety–value tradeofs over benchmark update rules. Our results provide a principled way to coordinate when learned policies should be deployed to safely maximize performance gain.

Our framework also leaves several directions for future work. First, the structural analysis adopts a reducedform difusion model for the evolution of policy value. Extending the theory to broader learning dynamics, including non-Gaussian update risks derived directly from specific learning algorithms, would clarify the gen erality of the resulting scheduling laws. Second, the implemented scheduler relies on estimated rewards and unsafe-update probabilities, so translating empirical feasibility into finite-sample population guarantees requires additional uncertainty quantification. Finally, our safety criterion controls the frequency of harmful updates but not their magnitude. Incorporating severity-sensitive or tail-risk notions of deterioration would provide a richer notion of deployment safety.

## References

Saman Ahmadi, Andrea Raith, and Mahdi Jalili. A fast and simple algorithm for the resource constrained shortest path problem. In 33rd Annual European Symposium on Algorithms, volume 351 of Leibniz International Proceedings in Informatics, pages 97:1–97:15. Schloss Dagstuhl–Leibniz-Zentrum für Informatik, 2025a. doi: 10.4230/LIPIcs.ESA.2025.97.

Saman Ahmadi, Andrea Raith, Guido Tack, and Mahdi Jalili. Resource constrained pathfinding with enhanced bidirectional A\* search. Proceedings of the AAAI Conference on Artificial Intelligence, 39(25):26878–26885, 2025b. doi: 10.1609/aaai.v39i25.34892.

Stefan Ankirchner and Stefan Perko. A comparison of continuous-time approximations to stochastic gradient descent. Journal of Machine Learning Research, 25(13):1–55, 2024.

Susan Athey and Stefan Wager. Policy learning with observational data. Econometrica, 89(1):133–161, 2021. doi: 10.3982/ECTA15732.

Yu Bai, Tengyang Xie, Nan Jiang, and Yu-Xiang Wang. Provably eficient q-learning with low switching cost. In Advances in Neural Information Processing Systems, volume 32, 2019.

John E. Beasley and Nicos Christofides. An algorithm for the resource constrained shortest path problem. Networks, 19(4):379–394, 1989.

Brian M. Cho, Ana-Roxana Pop, Kyra Gan, Sam Corbett-Davies, Israel Nir, Ariel Evnine, and Nathan Kallus. CSPI-MT: Calibrated safe policy improvement with multiple testing for threshold policies. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 165–176. Association for Computing Machinery, 2025. doi: 10.1145/3690624.3709176.

Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. Doubly robust policy evaluation and optimization. Statistical Science, 29(4):485–511, 2014. doi: 10.1214/14 -STS500.

Benjamin Gess, Sebastian Kassing, and Vitalii Konarovskyi. Stochastic modified flows, mean-field limits and dynamics of stochastic gradient descent. Journal of Machine Learning Research, 25(30):1–27, 2024.

Gabriel Y. Handler and Israel Zang. A dual algorithm for the constrained shortest path problem. Networks, 10(4): 293–309, 1980. doi: 10.1002/net.3230100403.

Refael Hassin. Approximation schemes for the restricted shortest path problem. Mathematics of Operations Research, 17(1):36–42, 1992. doi: 10.1287/moor.17.1.36.

Ilyas Himmich, Issmail El Hallaoui, and François Soumis. A multiphase dynamic programming algorithm for the shortest path problem with resource constraints. European Journal of Operational Research, 315(2):470–483, 2024. doi: 10.1016/j.ejor.2023.11.047.

Jiawei Huang, Jinglin Chen, Li Zhao, Tao Qin, Nan Jiang, and Tie-Yan Liu. Towards deployment-eficient reinforcement learning: Lower bound and optimality. In International Conference on Learning Representations, 2022.

International Stroke Trial Collaborative Group. The International Stroke Trial (IST): a randomised trial of aspirin, subcutaneous heparin, both, or neither among 19435 patients with acute ischaemic stroke. The Lancet, 349(9065): 1569–1581, 1997.

Kiyosi Itô. On stochastic diferential equations. Memoirs of the American Mathematical Society, 4:1–51, 1951. doi: 10.1090/memo/0004.

Zeyang Jia, Kosuke Imai, and Michael Lingzhi Li. Cramming contextual bandits for on-policy statistical evaluation. arXiv preprint arXiv:2403.07031, 2024.

Romain Laroche, Paul Trichelair, and Rémi Tachet des Combes. Safe policy improvement with baseline bootstrapping. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3652–3661. PMLR, 2019.

Tor Lattimore. A difusion analysis of policy gradient for stochastic bandits, 2026.

Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and adaptive stochastic gradient algorithms. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 2101–2110. PMLR, 2017.

Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and dynamics of stochastic gradient algorithms i: Mathematical foundations. Journal of Machine Learning Research, 20(40):1–47, 2019.

Dean H. Lorenz and Danny Raz. A simple eficient approximation scheme for the restricted shortest path problem. Operations Research Letters, 28(5):213–219, 2001. doi: 10.1016/S0167-6377(01)00069-4.

Stephan Mandt, Matthew D. Hofman, and David M. Blei. Stochastic gradient descent as approximate bayesian inference. Journal of Machine Learning Research, 18(134): 1–35, 2017.

Tatsuya Matsushima, Hiroki Furuta, Yutaka Matsuo, Ofir Nachum, and Shixiang Gu. Deployment-eficient reinforcement learning via model-based ofline optimization. In International Conference on Learning Representations, 2021.

Matteo Pirotta, Marcello Restelli, Alessio Pecorino, and Daniele Calandriello. Safe policy iteration. In Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 307–315. PMLR, 2013.

Drew Prinster, Clara Fannjiang, Ji Won Park, Kyunghyun Cho, Anqi Liu, Suchi Saria, and Samuel Stanton. Conformal policy control. arXiv preprint arXiv:2603.02196, 2026. doi: 10.48550/arXiv.2603.02196.

Min Qian and Susan A. Murphy. Performance guarantees for individualized treatment rules. The Annals of Statistics, 39(2):1180–1210, 2011. doi: 10.1214/10-AOS864.

James M Robins, Andrea Rotnitzky, and Lue Ping Zhao. Estimation of regression coeficients when some regressors are not always observed. Journal of the American statistical Association, 89(427):846–866, 1994.

Donald B Rubin. Estimating causal efects of treatments in randomized and nonrandomized studies. Journal of educational Psychology, 66(5):688, 1974.

Peter A. G. Sandercock, Maciej Niewada, and Anna Członkowska. The International Stroke Trial database. Trials, 12:101, 2011.

Ayush Sawarni, Nirjhar Das, Siddharth Barman, and Gaurav Sinha. Generalized linear bandits with limited adaptivity. In Advances in Neural Information Processing Systems, volume 37, pages 8329–8369, 2024.

John Schulman, Sergey Levine, Pieter Abbeel, Michael I. Jordan, and Philipp Moritz. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 1889–1897. PMLR, 2015.

Abhishek Sharma, Leo Benac, Sonali Parbhoo, and Finale Doshi-Velez. Decision-point guided safe policy improvement. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 2935–2943. PMLR, 2025.

Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. High confidence policy improvement. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2380–2388. PMLR, 2015.

U.S. Food and Drug Administration. Marketing submission recommendations for a predetermined change control plan for artificial intelligence-enabled device software functions. Guidance for industry and food and drug administration staf, U.S. Food and Drug Administration, August 2025. Document issued on August 18, 2025; originally issued on December 4, 2024. Accessed: 2026-04-07.

Yifan Wu, Roshan Sharif, Tor Lattimore, and Csaba Szepesvári. Conservative bandits. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1254–1262. PMLR, 2016.

Nuoya Xiong, Zhaoran Wang, and Zhuoran Yang. A general framework for sequential decision-making under adaptivity constraints. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 54792–54830. PMLR, 2024.

Ruohan Zhan, Zhimei Ren, Susan Athey, and Zhengyuan Zhou. Policy learning with adaptively collected data. Man-

agement Science, 70(8):5270–5297, 2024. doi: 10.1287/mn sc.2023.4921.

Zihan Zhang, Yuxin Chen, Jason D. Lee, Simon S. Du, Lin F. Yang, and Ruosong Wang. Deployment eficient reward-free exploration with linear function approximation. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-0853.

Heyang Zhao, Jiafan He, and Quanquan Gu. A nearly optimal and low-switching algorithm for reinforcement learning with general function approximation. In Advances in Neural Information Processing Systems, volume 37, pages 94684–94735, 2024a. doi: 10.52202/079017-3002.

Weiye Zhao, Feihan Li, Yifan Sun, Rui Chen, Tianhao Wei, and Changliu Liu. Absolute policy optimization: Enhancing lower probability bound of performance with high confidence. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 60866–60905. PMLR, 2024b.

Yingqi Zhao, Donglin Zeng, A. John Rush, and Michael R. Kosorok. Estimating individualized treatment rules using outcome weighted learning. Journal of the American Statistical Association, 107(499):1106–1118, 2012. doi: 10.1080/01621459.2012.695674.

Wenbin Zhou and Shixiang Zhu. Calibrating decision robustness via inverse conformal risk control. In Proceedings of the 43rd International Conference on Machine Learning, 2026.

Wenbin Zhou, Agni Orfanoudaki, and Shixiang Zhu. Conformalized decision risk assessment. In International Conference on Learning Representations, volume 2026, pages 10581–10610, 2026.

## Appendices

## A Discussion: Problem setting extension

We use binary treatment assignment to introduce the deployment problem, but the scheduling formulation does not depend on a causal model. More generally, let a learning procedure produce candidates $f _ { 1 } , \ldots , f _ { T }$ from an exogenous training stream, and let $V ( f )$ be an integrable scalar measure of population performance, with larger values preferred. For example, for prediction, we can take $V ( f ) = - \mathbb { E } [ \ell ( f ; Z ) ]$ ]; for a decision rule, we can use its expected utility. We define an unsafe update by $V ( f _ { j } ) < V ( f _ { i } )$ and retain the objective and expected-count constraint in (1). We also assume that the schedule changes which candidate is deployed, not the distribution of the training sequence. As in the main paper, we choose the schedule before observing the future training trajectory (i.e., an ofline setting).

Under the same mean-improvement condition as Assumption 2, the newest-candidate reduction, path representation, and dynamic program apply after replacing $\pi _ { i }$ by $f _ { i } .$ . The asymptotic analysis likewise uses only the induced scalar value process and its stated regularity and tail conditions. We therefore obtain the same conclusions for another learning task when that task satisfies those conditions; the existence of a scalar performance criterion alone does not imply the difusion assumption. For the statistical results below, we additionally require the independen historical trajectories and bounded values specified there. A positive afine change of the performance scale preserves every unsafe-update event and simply rescales rewards.

## B Additional details for Section 3

We write $R _ { i j }$ and $S _ { i j }$ for the entries of the reward and safety matrices, $V _ { i } = V ( \pi _ { i } ) , \nu _ { i } = \mathbb { E } [ V _ { i } ]$ , and $p _ { i j } = \mathbb { P } ( V _ { j } < V _ { i } )$ for $1 \leq i < j \leq T$ . We set $p _ { i , T + 1 } = 0$ . A newest-policy schedule has times

$$
1 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { K } \leq T < t _ { K + 1 } = T + 1 , \qquad h _ { k } = t _ { k + 1 } - t _ { k } \quad ( 0 \leq k \leq K ) .
$$

There are K actual updates and $K + 1$ holding intervals. In particular, the last edge contributes reward but no update risk. We denote the population objective and risk of a path P by $J _ { T } ( P )$ and $S _ { T } ( P )$ , respectively.

## B.1 Proof of Lemma 1

Proof. The deterministic meta-policy class is finite, and its no-update member is feasible, so a maximizer exists. We show how to transform any feasible meta-policy into a newest-policy schedule without decreasing its objective or increasing its risk.

First, remove backward switches. Suppose a schedule switches from candidate i to candidate $j < i ,$ and its next switch, if any, is from $j$ to ℓ. We delete the switch to $j$ and retain i until that next switch. On the intervening periods, the expected value weakly increases because $\nu _ { i } \geq \nu _ { j }$ . For every realized candidate-value vector,

$$
\mathbb { 1 } \{ V _ { \ell } < V _ { i } \} \le \mathbb { 1 } \{ V _ { j } < V _ { i } \} + \mathbb { 1 } \{ V _ { \ell } < V _ { j } \} .\tag{B.1}
$$

Indeed, if both events on the right fail, then $V _ { \ell } \geq V _ { j } \geq V _ { i }$ . Thus the new direct transition has risk no larger than the sum of the two removed risks. If there is no later switch, deleting the last switch removes a nonnegative risk. A resulting transition from i to itself is a hold and has zero risk. Each operation removes at least one switch, so repetition terminates with a schedule whose deployed candidate indices are strictly increasing.

Write these indices as $1 = j _ { 0 } < j _ { 1 } < \cdots < j _ { K }$ , and let $s _ { k } \geq j _ { k }$ be their original deployment times. We now advance the deployment of candidate $j _ { k }$ to time $j _ { k }$ , processing the switches in increasing order. The preceding switch has already been advanced to $j _ { k - 1 } < j _ { k }$ , and the next switch remains later than $s _ { k } \geq j _ { k }$ , so this change preserves the order and availability constraints. It replaces the preceding, lower-index candidate by $j _ { k }$ on additional periods, weakly increasing expected value. The ordered pairs of candidates compared at actual switches do not change, so neither does the total risk. The final schedule either retains its incumbent or deploys the candidate trained at the current time. Applying this transformation to a maximizer proves the claim. □

## B.2 Proof of Proposition 1

Proof. Given a newest-policy schedule, form the path $P = \left( ( t _ { 0 } , t _ { 1 } ) , \dots , ( t _ { K } , t _ { K + 1 } ) \right)$ from 1 to $T + 1$ . On the edge $\left( { t _ { k } , t _ { k + 1 } } \right)$ , we deploy $\pi _ { t _ { k } }$ for exactly $h _ { k }$ periods. Therefore

$$
J _ { T } ( P ) = \frac { 1 } { T } \sum _ { k = 0 } ^ { K } h _ { k } \nu _ { t _ { k } } = \sum _ { ( i , j ) \in P } R _ { i j } , \qquad S _ { T } ( P ) = \sum _ { k = 0 } ^ { K - 1 } p _ { t _ { k } , t _ { k + 1 } } = \sum _ { ( i , j ) \in P } S _ { i j } .\tag{B.2}
$$

The second equality follows from linearity of expectation applied to the unsafe-update indicators.

Conversely, every source–sink path in the time-ordered graph has strictly increasing vertices and defines a valid newest-policy schedule by retaining its current vertex’s candidate until the next vertex. The terminal edge retains the last candidate through period $T .$ The two identities in (B.2) again hold. Thus paths and newest-policy schedules have the same rewards and risks, and Lemma 1 establishes equivalence with (1). The direct path $( 1 , T + 1 )$ represents no updates. For a zero-length subpath, we use $\mathcal { P } _ { i i } = \{ \alpha \}$ , with zero reward and zero resource consumption. □

## B.3 Correctness and complexity of the dynamic program

Proposition B.1 (Exact solution of the rounded empirical problem). With $B _ { \Delta } = \lfloor \epsilon / \Delta \rfloor$ and $c _ { i j } = \lceil \widehat { S } _ { i j } / \Delta \rceil$ , where $c _ { i , T + 1 } = 0$ , Algorithm 1 maximizes the estimated objective over paths satisfying $\sum ( i , j ) \epsilon P ^ { C _ { i j } } \leq B _ { \Delta }$ . Given the edge estimates, its time and memory requirements are

$$
O \big ( T ^ { 2 } ( B _ { \Delta } + 1 ) \big ) , \qquad O \big ( T ^ { 2 } + T ( B _ { \Delta } + 1 ) \big ) ,
$$

respectively, when the edge matrices and backtracking information are stored.

Proof. We prove by induction in j that $F ( \boldsymbol { j } , \boldsymbol { b } )$ is the largest estimated reward of a path from 1 to j with rounded cost at most b. The initialization $F ( 1 , b ) = 0$ gives the assertion at the source. A path reaching $j > 1$ has a unique last edge $( i , j )$ , where $i < j$ . Removing that edge leaves a path with available budget $b - c _ { i j }$ . By the induction hypothesis, its reward is at most $F ( i , b - c _ { i j } )$ . Conversely, a path attaining a finite predecessor state followed by $( i , j )$ is a valid path to $j$ with the prescribed budget. Taking the maximum over predecessors gives the Bellman recursion in the main paper. Induction proves the state interpretation, and backtracking from $( T + 1 , B _ { \Delta } )$ recovers an optimizer.

There are $( T + 1 ) ( B _ { \Delta } + 1 )$ states, with at most $T$ predecessor comparisons per state. The state values and predecessor pointers require $O ( T ( B _ { \Delta } + 1 ) )$ ) memory, and the edge matrices require $O ( T ^ { 2 } )$ memory. Forming all pairwise empirical risks from N value trajectories separately costs $O ( N T ^ { 2 } )$ time. The source–sink edge has zero cost, so the rounded problem remains feasible even when $B _ { \Delta } = 0$ □

## B.4 Choosing the budget resolution

Let $\boldsymbol { \mathcal { M } } _ { T , \bar { K } }$ be the newest-policy schedules with at most $\bar { K }$ actual updates, and define

$$
J _ { T , \bar { K } } ^ { \star } ( b ) = \operatorname* { m a x } \{ J _ { T } ( P ) : P \in \mathcal { M } _ { T , \bar { K } } , \ S _ { T } ( P ) \leq b \} , \qquad b \geq 0 .\tag{B.3}
$$

Without a count restriction, we take $\bar { K } = T - 1$ . With a restriction, we append a count coordinate to the dynamic-programming state and increase it only when $j \le T$ . Write $\begin{array} { r } { \widehat { \nu } _ { i } = N ^ { - 1 } \sum _ { n } \widehat { V } _ { i } ^ { ( n ) } } \end{array}$ , so that $\widehat { \pmb { R } } _ { i j } = ( j - i ) \widehat { \nu } _ { i } / T$ Proposition B.2 (Estimation and gridding error). Suppose, on an event A, that

$$
\operatorname* { m a x } _ { i } | \widehat { \nu } _ { i } - \nu _ { i } | \leq r , \qquad \operatorname* { m a x } _ { i < j \leq T } | \widehat { \pmb { S } } _ { i j } - p _ { i j } | \leq s .
$$

Let $\widehat { P } _ { \Delta }$ solve the rounded empirical problem within $\mathcal { M } _ { T , \bar { K } }$ , and put

$$
g _ { \Delta } = \epsilon - \Delta \lfloor \epsilon / \Delta \rfloor , \qquad b _ { \Delta } = \bar { K } ( s + \Delta ) + g _ { \Delta } .
$$

On A, we have $S _ { T } ( \widehat { P } _ { \Delta } ) \leq \epsilon + \bar { K } \mathrm { ~ . ~ }$ s . $I f b _ { \Delta } \leq \epsilon _ { ; }$ , then

$$
J _ { T , \bar { K } } ^ { \star } ( \epsilon - b _ { \Delta } ) - r \leq \widehat { J } _ { T } ( \widehat { P } _ { \Delta } ) \leq J _ { T , \bar { K } } ^ { \star } ( \epsilon + \bar { K } s ) + r ,\tag{B.4}
$$

$$
J _ { T } ( \widehat { P } _ { \Delta } ) \geq J _ { T , \bar { K } } ^ { \star } ( \epsilon - b _ { \Delta } ) - 2 r .\tag{B.5}
$$

In particular, for $\omega _ { \epsilon } ( b ) = J _ { T , \bar { K } } ^ { \star } ( \epsilon ) - J _ { T , \bar { K } } ^ { \star } ( \epsilon - b )$

$$
J _ { T , \bar { K } } ^ { \star } ( \epsilon ) - J _ { T } ( \widehat { P } _ { \Delta } ) \leq 2 r + \omega _ { \epsilon } ( b _ { \Delta } ) .\tag{B.6}
$$

Proof. For every schedule, its holding fractions are nonnegative and sum to one. Hence, simultaneously over all paths,

$$
| \widehat { J } _ { T } ( P ) - J _ { T } ( P ) | \leq \sum _ { ( i , j ) \in P } \frac { j - i } { T } | \widehat { \nu } _ { i } - \nu _ { i } | \leq r .\tag{B.7}
$$

There is no factor $\bar { K }$ in this reward bound. Rounded feasibility implies

$$
S _ { T } ( \widehat { P } _ { \Delta } ) \leq \sum _ { e \in \widehat { P } _ { \Delta } } \widehat { S } _ { e } + \bar { K } s \leq \Delta B _ { \Delta } + \bar { K } s \leq \epsilon + \bar { K } s .
$$

For a comparator P with $S _ { T } ( P ) \le \epsilon - b _ { \Delta }$ , upward rounding on its actual transitions gives

$$
\Delta \sum _ { e \in P } c _ { e } \le S _ { T } ( P ) + \bar { K } s + \bar { K } \Delta \le \epsilon - g _ { \Delta } = \Delta B _ { \Delta } .
$$

The terminal edge contributes no rounding error. Thus every such population comparator is feasible for the rounded empirical problem. Empirical optimality and (B.7) give the left inequality in (B.4) and (B.5). The risk bound for the selected path gives the right inequality. Subtracting (B.5) from the population frontier at ϵ proves (B.6).

To derive an explicit dependence on N and $T ,$ suppose the historical value vectors are independent copies of the future value vector, their coordinates lie in [0, 1], and their population values are observed without evaluation error. With $M = T ( T - 1 ) / 2$ and $\delta \in ( 0 , 1 )$ , the bounded-mean inequality proved below and a union bound give

$$
r = s = { \sqrt { \frac { \log ( 2 ( T + M ) / \delta ) } { 2 N } } } , \qquad \mathbb { P } ( A ) \geq 1 - \delta .\tag{B.8}
$$

Consequently the direct reward error is ${ \cal O } ( \sqrt { \log ( T / \delta ) / N } )$ , whereas the safety-budget displacement is

$$
O \left( \bar { K } \sqrt { \frac { \log ( T / \delta ) } { N } } + \bar { K } \Delta + \Delta \right) .
$$

With no count cap, $\bar { K } = T - 1$ . No assumption about independence of edges within one trajectory is needed. For estimated population values, we use the evaluation result in Proposition B.4 rather than interpreting noisy comparisons as observed true signs.

A budget displacement cannot generally be replaced by a uniform Lipschitz error in the objective. For example, let $T = 2 , V _ { 1 } = 1 / 2$ , and let $V _ { 2 }$ equal 1 with probability $1 - p$ and 0 with probability $p ,$ where $0 < p < 1 / 2$ . The frontier jumps from $1 / 2$ to $3 / 4 - p / 2$ at budget p. At $\epsilon = p ,$ any grid for which $p / \Delta$ is not an integer excludes the update, even along a sequence $\Delta \downarrow 0$ . Thus (B.6), rather than an unconditional $O ( \Delta )$ value-loss claim, is the appropriate guarantee. If an optimizer has safety slack at least $b _ { \Delta }$ , the modulus is zero; a Lipschitz frontier bound, when separately available on the relevant interval, gives the corresponding linear bound.

A theoretical resolution criterion. For an allocated gridding tolerance $\kappa > 0$ in risk units, we can choose $\Delta \le \kappa / ( \bar { K } + 1 )$ , because $\bar { K } \Delta + g _ { \Delta } \leq ( \bar { K } + 1 ) \Delta$ . We should not make the grid substantially finer than the statistical accuracy unless a near-binding comparator makes that worthwhile. In the plug-in bound (B.8), balancing $\bar { K } \Delta$ against $\bar { K } s$ suggests $\Delta$ of order s or smaller. For the uncertainty-aware bound below, the analogous balance is $\Delta$ of order $\sqrt { \epsilon a _ { N } / \bar { K } } + a _ { N }$ when $\bar { K } \ge 1$ . Smaller $\Delta$ improves the budget approximation but increases the dynamic-programming cost through $B _ { \Delta } \asymp \epsilon / \Delta$

An empirical refinement procedure. We can begin with $\Delta = \epsilon / B$ for a modest positive integer $B ,$ then double B until the selected reward and schedule stabilize at a prespecified tolerance $_ { \mathrm { o r } }$ the computational limit is reached. These nested grids have no residual $g _ { \Delta }$ , and every previously feasible path remains feasible after refinement because $\lceil 2 x \rceil \leq 2 \lceil x \rceil$ . We record both the rounded and unrounded resource consumption at each refinement. Numerical stability is an empirical stopping rule, not a population-safety certificate; independent evaluation or the simultaneous bounds below provide that certificate. When $\epsilon = 0$ , we instead retain only edges with exactly zero chosen risk cost and solve the resulting acyclic path problem.

## B.5 An exact labeling alternative

We can solve the unrounded path problem by propagating nondominated labels. Let $a _ { i j } \geq 0$ be the chosen edge resource, either a population risk, an empirical risk, or a conservative upper confidence cost, with $a _ { i , T + 1 } = 0$ . A label at node $j$ stores its cumulative resource $s ,$ cumulative reward $r ,$ and a predecessor pointer. At the same node, $( s , r )$ dominates $( s ^ { \prime } , r ^ { \prime } )$ when $s \leq s ^ { \prime }$ and $r \geq r ^ { \prime }$ , with at least one strict inequality. We retain only one representative when both coordinates agree.

Algorithm 2 Exact nondominated-label algorithm   
Require: Edge rewards $\widehat { \pmb { R } } _ { i j }$ , nonnegative resources $a _ { i j }$ , budget ϵ.   
1: Set $\mathcal { L } _ { 1 } = \{ ( 0 , 0 , \mathrm { n u l l } ) \}$ and $\mathcal { L } _ { j } = \mathcal { D }$ for $j > 1 .$   
2: for $j = 2 , \ldots , T + 1$ do   
3: Form labels $( s + a _ { i j } , r + \widehat { R } _ { i j }$ , pointer) from every $i < j$ and every $( s , r , \cdot ) \in \mathcal { L } _ { i }$   
4: Discard labels with resource greater than ϵ.   
5: Sort the remaining labels by increasing resource and discard dominated labels; store the survivors in $\mathcal { L } _ { j }$   
6: end for   
7: Return the highest-reward label in $\mathcal { L } _ { T + 1 }$ and recover its path by backtracking.

Proposition B.3 (Exactness of labeling). Algorithm 2 returns an exact maximizer $o f$ the unrounded problem for its chosen edge quantities. $I f A _ { j }$ labels are generated at node $j$ before pruning and $L _ { j }$ survive, an implementation using sorting requires

$$
O \left( T ^ { 2 } + \sum _ { j = 2 } ^ { T + 1 } A _ { j } \log ( A _ { j } + 1 ) \right)
$$

time and $\begin{array} { r } { O ( T ^ { 2 } + \sum _ { j } L _ { j } ) } \end{array}$ persistent storage, in addition to temporary storage for the labels being processed. The numbers of labels can be exponential in the size of a general resource-constrained acyclic path instance.

Proof. Any sufix from a node adds the same resource and reward to two labels at that node. Extending the dominating label therefore yields no larger resource and no smaller reward. Discarding a dominated prefix cannot remove the sole representative of a better feasible complete path. Nonnegative resources likewise justify discarding an over-budget prefix. Induction over the node order now shows that every feasible prefix is represented or dominated by a retained label. At the sink, selecting the largest reward is therefore exact.

Generating the incoming labels costs $O ( A _ { j } )$ operations. After sorting by resource, we keep a label only when its reward exceeds the best reward at a no-larger resource; equal-resource ties are resolved by retaining the largest reward. This costs $O ( A _ { j } \log ( A _ { j } + 1 ) )$ time. The stated storage follows from the retained labels and their pointers. To see the worst-case obstruction, consider a chain of binary-choice gadgets with resource and reward increments both equal to $2 ^ { - k }$ for taking choice $k ,$ and zero for skipping it. Under a suficiently large budget, every subset generates a diferent pair $( s , s )$ . None dominates another, so there are exponentially many nondominated labels.

Labeling avoids budget rounding and can be attractive when pruning leaves a small frontier. In contrast, the grid dynamic program bounds the resource dimension in advance and has the predictable pseudo-polynomial cost in Proposition B.1; this worst-case computational distinction motivates our main-paper choice. Exactness here refers to the specified real edge quantities, so floating-point tolerances must not be used to prune labels whose dominance is unresolved. With an update-count cap, we compare labels only at equal node and count, or include count in the dominance relation.

## B.6 An uncertainty-aware extension

The plug-in costs in Algorithm 1 can underestimate population risk. We can instead assign an upper confidence cost to each actual edge and apply exactly the same path solver. The companions establish this construction for a fixed number of independent historical trajectories. We obtain an anytime-valid version by allocating the failure probability across sample sizes. Here “anytime” refers to the number of historical trajectories used to plan the schedule, not to adapting deployment decisions to the realized future training stream.

Assumption B.1 (Historical trajectories for certification). The vectors $( V _ { 1 } ^ { ( n ) } , \ldots , V _ { T } ^ { ( n ) } ) , n \ge 1$ , are independent and identically distributed copies of the future candidate-value vector and are independent of that future vector. Every coordinate lies in [0, 1]. Dependence between coordinates within a vector is unrestricted.

This bounded-value assumption is used only for the statistical guarantees, separately from the Gaussian value model used for the structural asymptotics. Note that independent resamples of a single observed dataset are not, merely by being generated separately, independent population trajectories satisfying Assumption B.1.

Lemma B.1 (Concentration and an empirical upper confidence cost). For N independent [0, 1] variables with common mean v and average $\bar { X }$ , and N independent Bernoulli variables with mean p and average ${ \bar { B } } ,$ , we have, for $\ell > 0$ and $a = \ell / N$

$$
\mathbb { P } \Bigg ( | \bar { X } - v | > \sqrt { \frac { \ell } { 2 N } } \Bigg ) \leq 2 e ^ { - \ell } , \qquad \mathbb { P } \Bigg ( | \bar { B } - p | > \sqrt { 2 p a } + \frac { 2 a } { 3 } \Bigg ) \leq 2 e ^ { - \ell } .\tag{B.9}
$$

The function $U _ { a } ( x ) = \operatorname* { m i n } \{ 1 , x + { \sqrt { 2 a x } } + 4 a \}$ is nondecreasing on [0, 1]. Whenever $| x - p | \leq \sqrt { 2 p a } + 2 a / 3$ , it satisfies

$$
p \leq U _ { a } ( x ) \leq p + 2 { \sqrt { 2 p a } } + 7 a .\tag{B.10}
$$

Proof. For a variable $X \in [ 0 , 1 ]$ , the second derivative of $k ( \lambda ) = \log \mathbb { E } [ e ^ { \lambda X } ]$ is its variance under exponential tilting, which is at most $1 / 4$ . Integrating twice gives log $\mathbb { E } [ e ^ { \lambda ( X - v ) } ] \le \lambda ^ { 2 } / 8$ . Independence and exponential Markov bounds, optimized over positive and negative λ, give the first inequality in (B.9).

For a Bernoulli variable $B ,$ , we have

$$
\log \mathbb { E } [ e ^ { \lambda ( B - p ) } ] \leq p ( e ^ { \lambda } - 1 - \lambda ) \leq \frac { p \lambda ^ { 2 } } { 2 ( 1 - | \lambda | / 3 ) } , \qquad | \lambda | < 3 .
$$

The last bound follows by expanding the exponential and using $k ! \geq 2 \cdot 3 ^ { k - 2 }$ for $k \geq 2 ;$ the negative-argument bound follows as well. For $p > 0$ , optimization with $\lambda = t / ( p + t / 3 )$ yields

$$
\mathbb { P } ( | \bar { B } - p | > t ) \le 2 \exp \left( - \frac { N t ^ { 2 } } { 2 ( p + t / 3 ) } \right) .
$$

At $t = \sqrt { 2 p a } + 2 a / 3$ , direct expansion gives $t ^ { 2 } \ge 2 a ( p + t / 3 )$ , proving the second inequality. If $p = 0$ , the assertion is immediate.

The map $U _ { a }$ is nondecreasing. Solving $p - x \leq \sqrt { 2 p a } + 2 a / 3$ as a quadratic inequality in $\sqrt { p }$ gives

$$
p \leq \left( { \sqrt { a / 2 } } + { \sqrt { x + 7 a / 6 } } \right) ^ { 2 } = x + { \frac { 5 a } { 3 } } + { \sqrt { 2 a x + { \frac { 7 a ^ { 2 } } { 3 } } } } \leq x + { \sqrt { 2 a x } } + 4 a .
$$

Clipping at one preserves the lower bound. Also, $x \leq p + \sqrt { 2 p a } + 2 a / 3 \leq ( \sqrt { p } + \sqrt { a } ) ^ { 2 }$ , and hence

$$
U _ { a } ( x ) - p \leq ( x - p ) + { \sqrt { 2 a x } } + 4 a \leq 2 { \sqrt { 2 p a } } + ( 2 / 3 + { \sqrt { 2 } } + 4 ) a \leq 2 { \sqrt { 2 p a } } + 7 a .
$$

This proves (B.10).

At historical sample size $N ,$ let $\widehat { \nu } _ { i , N }$ and $\widehat { p } _ { i j , N }$ be the empirical value means and strict-failure frequencies. For a total failure probability $\delta \in ( 0 , 1 )$ , define

$$
\delta _ { N } = \frac { \delta } { N ( N + 1 ) } , \qquad \ell _ { N } = \log \frac { 2 ( M + T ) } { \delta _ { N } } , \qquad a _ { N } = \frac { \ell _ { N } } { N } , \qquad r _ { N } = \sqrt { \frac { \ell _ { N } } { 2 N } } , \qquad U _ { i j , N } = U _ { a _ { N } } ( \widehat { p } _ { i j , N } ) ,\tag{B.11}
$$

with $U _ { i , T + 1 , N } = 0$ . We round $U _ { i j , N }$ upward instead of rounding $\widehat { p } _ { i j , N }$ , use the same estimated rewards, and solve the path problem at budget ϵ. Denote the resulting schedule by $\widehat { P } _ { N }$

Theorem B.1 (Anytime population safety and oracle performance). Under Assumption B.1, define

$$
b _ { N } = 2 \sqrt { 2 \bar { K } \epsilon a _ { N } } + 7 \bar { K } a _ { N } + \bar { K } \Delta + g _ { \Delta } .\tag{B.12}
$$

With probability at least $1 - \delta ,$ simultaneously for every integer $N \geq 1$

$$
S _ { T } ( \widehat { P } _ { N } ) \leq \epsilon , \qquad J _ { T } ( \widehat { P } _ { N } ) \geq J _ { T , \bar { K } } ^ { \star } ( \epsilon - b _ { N } ) - 2 r _ { N } \quad w h e n e v e r b _ { N } \leq \epsilon .\tag{B.13}
$$

The no-update schedule is feasible for every historical dataset. The same guarantees apply at any almost surely finite, data-dependent stopping sample size. Including the randomness of the historical data and the independent future trajectory, the probability of at least one unsafe future update is at most $\epsilon + \delta$

Proof. At sample size N, apply Lemma B.1 to all $T$ candidate coordinates and all M actual edge indicators. A union bound with (B.11) gives failure probability at most $\delta _ { N }$ . Since $\begin{array} { r } { \sum _ { N \ge 1 } \delta _ { N } = \delta _ { } } \end{array}$ , a second union bound gives an event $A _ { \infty }$ of probability at least $1 - \delta$ on which, for every $N _ { : }$

$$
\operatorname* { m a x } _ { i } \left| { \widehat { \nu } } _ { i , N } - \nu _ { i } \right| \leq r _ { N } , \qquad p _ { e } \leq U _ { e , N } \leq p _ { e } + 2 { \sqrt { 2 p _ { e } a _ { N } } } + 7 a _ { N } \quad ( e { \mathrm { ~ a n ~ a c t u a l ~ e d g e } } ) .\tag{B.14}
$$

The estimates at diferent sample sizes need not be independent.

On this event, every rounded-feasible path satisfies

$$
S _ { T } ( P ) \le \sum _ { e \in P } U _ { e , N } \le \Delta \sum _ { e \in P } \left\lceil U _ { e , N } / \Delta \right\rceil \le \Delta B _ { \Delta } \le \epsilon .
$$

This is simultaneous over paths and therefore holds for $\widehat { P } _ { N }$ . For a comparator with at most $\bar { K }$ updates and true risk S, Cauchy–Schwarz gives

$$
\sum _ { e \in P } U _ { e , N } \leq S + 2 \sqrt { 2 a _ { N } } \sum _ { e \in P , \ e \mathrm { ~ a c t u a l } } \sqrt { p _ { e } } + 7 \bar { K } a _ { N } \leq S + 2 \sqrt { 2 \bar { K } S a _ { N } } + 7 \bar { K } a _ { N } .\tag{B.15}
$$

If $S \le \epsilon - b _ { N }$ , substituting $S \leq \epsilon$ in the square root and adding at most $K \Delta$ for rounding proves empirical feasibility of the comparator. The terminal edge has zero cost throughout. The uniform reward bound (B.7), with $r = r _ { N }$ then gives

$$
J _ { T } ( \widehat { P } _ { N } ) \geq \widehat { J } _ { T } ( \widehat { P } _ { N } ) - r _ { N } \geq \widehat { J } _ { T } ( P ^ { \star } ) - r _ { N } \geq J _ { T } ( P ^ { \star } ) - 2 r _ { N } ,
$$

where $P ^ { \star }$ optimizes the reduced-budget population problem. This proves the oracle inequality. All statements hold on the same event for all N, so evaluating them at a stopping sample size introduces no additional failure probability.

Conditional on historical data in $A _ { \infty } .$ , the selected schedule is fixed relative to the future trajectory. The union bound for its unsafe events gives a conditional probability at most $S _ { T } ( \widehat { P } _ { N } ) \leq \epsilon$ . On the complementary event, we bound this probability by one. Averaging yields at most $\epsilon + \delta$ . The direct terminal edge proves feasibility on every dataset. □

For a prespecified N, we can replace $\delta _ { N }$ by δ and recover the fixed-sample confidence construction in the companions. The anytime version adds only the logarithmic sample-size term in $\ell _ { N }$ . Its certification cost scales as

$$
O \left( \sqrt { \frac { \bar { K } \epsilon \log ( T N / \delta ) } { N } } + \frac { \bar { K } \log ( T N / \delta ) } { N } + \bar { K } \Delta + \Delta \right) ,
$$

while the reward error remains independent of the number of updates outside the logarithm. These costs certify population risk; numerical gridding alone does not.

Proposition B.4 (Certification with estimated population values). Fix N. Suppose Assumption B.1 holds for the true values, and evaluation estimates ${ \widetilde { V } } _ { i } ^ { ( n ) }$ satisfy an event

$$
\mathcal { A } _ { \eta } = \left\{ \operatorname* { m a x } _ { n \leq N , i \leq T } | \widetilde { V } _ { i } ^ { ( n ) } - V _ { i } ^ { ( n ) } | \leq \eta \right\} , \qquad \mathbb { P } ( A _ { \eta } ) \geq 1 - \delta _ { \mathrm { e v a l } } ,\tag{B.16}
$$

for a specified deterministic $\eta \geq 0$ . Use the conservative labels

$$
C _ { i j } ^ { ( n ) } = \mathbf { 1 } \{ \widetilde { V } _ { j } ^ { ( n ) } - \widetilde { V } _ { i } ^ { ( n ) } \leq 2 \eta \}
$$

in the empirical upper confidence costs. In this proposition set

$$
\ell _ { N } = \log \frac { 2 ( 2 M + T ) } { \delta _ { \mathrm { s t a t } } } , \qquad a _ { N } = \ell _ { N } / N , \qquad r _ { N } = \sqrt { \ell _ { N } / ( 2 N ) } , \qquad \zeta _ { T } ( \eta ) = \operatorname* { m a x } _ { i < j \leq T } \mathbb { P } ( 0 \leq V _ { j } - V _ { i } \leq 4 \eta ) .
$$

With

$$
b _ { N , \eta } = \bar { K } \zeta _ { T } ( \eta ) + 2 \sqrt { 2 \bar { K } \{ \epsilon + \bar { K } \zeta _ { T } ( \eta ) \} } a _ { N } + 7 \bar { K } a _ { N } + \bar { K } \Delta + g _ { \Delta } ,
$$

the selected schedule satisfies, with probability at least $1 - \delta _ { \mathrm { s t a t } } - \delta _ { \mathrm { e v a l } ; }$

$$
S _ { T } ( \widehat { P } ) \leq \epsilon , \qquad J _ { T } ( \widehat { P } ) \geq J _ { T , \bar { K } } ^ { \star } ( \epsilon - b _ { N , \eta } ) - 2 ( r _ { N } + \eta ) \quad i f b _ { N , \eta } \leq \epsilon .
$$

No independence between the evaluation errors is required beyond (B.16).

Proof. For each actual edge, define the true labels $B _ { i j } ^ { ( n ) } = \mathbf { 1 } \{ V _ { j } ^ { ( n ) } - V _ { i } ^ { ( n ) } < 0 \}$ and $D _ { i j } ^ { ( n ) } = \mathbf { 1 } \{ V _ { j } ^ { ( n ) } - V _ { i } ^ { ( n ) } \leq 4 \eta \}$ On $A _ { \eta }$ , the deterministic inequalities

$$
B _ { i j } ^ { ( n ) } \leq C _ { i j } ^ { ( n ) } \leq D _ { i j } ^ { ( n ) }\tag{B.17}
$$

hold. Their true means satisfy $p _ { i j } \leq p _ { i j } ^ { + } \leq p _ { i j } + \zeta _ { T } ( \eta )$ . Apply Lemma B.1 to the 2M collections of true B and D labels and the $T$ true value coordinates. These collections are independent across historical replications even when the estimated labels share evaluation data. A union bound produces an event of probability at least $1 - \delta _ { \mathrm { { s t a t } } }$

On its intersection with $\boldsymbol { \mathcal { A } } _ { \eta }$ , monotonicity of $U _ { a _ { N } }$ and (B.17) give

$$
p _ { e } \le U _ { a _ { N } } ( \bar { B } _ { e } ) \le U _ { a _ { N } } ( \bar { C } _ { e } ) \le U _ { a _ { N } } ( \bar { D } _ { e } ) \le p _ { e } ^ { + } + 2 \sqrt { 2 p _ { e } ^ { + } a _ { N } } + 7 a _ { N } .
$$

Thus the implemented costs upper-bound every population risk. For a comparator of true risk $S ,$ summing th upper bounds yields

$$
\sum _ { e \in P } U _ { a _ { N } } ( \bar { C } _ { e } ) \leq S + \bar { K } \zeta _ { T } ( \eta ) + 2 \sqrt { 2 \bar { K } \{ S + \bar { K } \zeta _ { T } ( \eta ) \} a _ { N } } + 7 \bar { K } a _ { N } .
$$

The definition of $b _ { N , \eta }$ and upward rounding therefore make every comparator at budget $\epsilon - b _ { N , \eta }$ empirically feasible. Each estimated empirical value mean is within $r _ { N } + \eta$ of its population mean, so the reward comparison in Theorem B.1 gives the oracle bound. A union bound over the two events completes the proof. □

For example, if every candidate is evaluated by averaging a score in [0, 1] on the same independent validation sample of size $n _ { \mathrm { v a l } }$ , conditioning on all trained candidates and applying Lemma B.1 gives (B.16) with $\eta =$ $\sqrt { \log ( 2 N T / \delta _ { \mathrm { e v a l } } ) / ( 2 n _ { \mathrm { v a l } } ) }$ . An unbounded or weighted policy-value estimator requires its own valid simultaneous evaluation bound. A margin condition $\mathbb { P } ( | V _ { j } - V _ { i } | \le x ) \le C _ { T } x ^ { \alpha }$ implies $\zeta _ { T } ( \eta ) \leq C _ { T } ( 4 \eta ) ^ { \alpha }$ ; it controls conservatism, not the validity of the safety assertion itself. Time-uniform evaluation can be incorporated by allocating both statistical and evaluation failure probabilities across N in the same way as (B.11).

We include this uncertainty-aware construction as an extension rather than replacing the plug-in algorithm studied in the main paper. Developing tighter time-uniform costs, adaptive allocation of historical simulation efort across edges, and sharper guarantees for the resulting resource-constrained path problem would require further analysis Those questions are especially relevant to uncertainty-aware RCSPP methods and fall outside the present paper’s deployment-scheduling contribution.

## C Proofs for Section 4

We derive the continuum objective and constraint, solve the allocation pointwise, and then connect its value to the original integer problem. The proofs retain the $O ( 1 )$ spacing correction, the Θ(·) per-update risk, the $O ( T ^ { - 1 } )$ regret remainder, and the leading update count stated in the main paper.

## C.1 Discussion on Assumption 3

We impose the structured value-process assumption to keep the main analysis separate from the details of a particular learner. Throughout the proofs, we interpret its smoothness and positivity requirements as

$$
\mu , \sigma \in C ^ { 1 } ( [ 0 , 1 ] ) , \qquad 0 < \mu _ { - } \leq \mu ( u ) \leq \mu _ { + } < \infty , \qquad 0 < \sigma _ { - } \leq \sigma ( u ) \leq \sigma _ { + } < \infty ,\tag{C.1}
$$

with bounded first derivatives and constants independent of T. Strictly positive continuous functions on the compact interval have these positive lower bounds. This excludes a vanishing-drift or degenerate-volatility regime from the theorem being proved.

We can obtain the assumed process from primitive constructions, and we give two explicit suficient constructions.

Gaussian innovations with an afine performance map. Let $v ( \theta ) = v _ { 0 } + a ^ { \top } \theta$ for a fixed nonzero vector a, and consider

$$
\theta _ { t + 1 } = \theta _ { t } + \frac { 1 } { T } \{ b _ { T , t } + H _ { T , t } \xi _ { t + 1 } \} , \qquad \xi _ { t + 1 } \overset { \mathrm { i i d } } { \sim } N _ { d } ( 0 , I _ { d } ) ,\tag{C.2}
$$

where $b _ { T , t }$ and $H _ { T , t }$ are deterministic. Suppose their projections satisfy

$$
\boldsymbol { a } ^ { \top } b _ { T , t } = \int _ { t } ^ { t + 1 } \boldsymbol { \mu } ( \boldsymbol { s } / T ) \mathrm { d } \boldsymbol { s } , \qquad \| \boldsymbol { H } _ { T , t } ^ { \top } \boldsymbol { a } \| ^ { 2 } = \int _ { t } ^ { t + 1 } \sigma ( \boldsymbol { s } / T ) ^ { 2 } \mathrm { d } \boldsymbol { s } .
$$

The value increments are then independent Gaussian variables with exactly the means and variances of the stochastic integrals in Assumption 3. Their joint law is the integer skeleton of that process; a Brownian interpolation supplies a representation on an enlarged probability space. We can also express (C.2) as stochastic-gradient descent with step $1 / T$ on the sample loss $\ell _ { T , t } ( \theta ; \boldsymbol { \xi } ) = - \theta ^ { \top } ( b _ { T , t } + H _ { T , t } \boldsymbol { \xi } )$

Gradient-type difusions with deterministic projected coeficients. Let $\bar { g } ( u , \theta )$ and $C ( u , \theta )$ be a sample gradient’s mean and covariance, and suppose the parameter process is the well-defined, nonexplosive difusion

$$
\mathrm { d } \Theta _ { \tau } = - \frac { \gamma } { T } \bar { g } ( \tau / T , \Theta _ { \tau } ) \mathrm { d } \tau + \frac { \gamma } { T } C ( \tau / T , \Theta _ { \tau } ) ^ { 1 / 2 } \mathrm { d } B _ { \tau } .\tag{C.3}
$$

Local Lipschitz coeficients with suitable growth bounds give standard suficient existence conditions. Let $v \in C ^ { 2 }$ take $v ( \Theta _ { 0 } ) = V _ { 0 }$ , and impose, on the reachable parameter set,

$$
- \gamma \nabla v ( \theta ) ^ { \top } \bar { g } ( u , \theta ) + \frac { \gamma ^ { 2 } } { 2 T } \mathrm { t r } \{ C ( u , \theta ) \nabla ^ { 2 } v ( \theta ) \} = \mu ( u ) ,\tag{C.4}
$$

Itô’s formula gives drift $\mu ( \tau / T ) / T$ for $v ( \Theta _ { \tau } )$ . Its martingale part equals $T ^ { - 1 } \sigma ( \tau / T ) \mathrm { d } W _ { \tau }$ , where

$$
W _ { t } = \int _ { 0 } ^ { t } \frac { \gamma \nabla v ( \Theta _ { \tau } ) ^ { \top } C ( \tau / T , \Theta _ { \tau } ) ^ { 1 / 2 } } { \sigma ( \tau / T ) } \mathrm { d } B _ { \tau } .
$$

The integrand has squared norm one by (C.4), so this continuous martingale has quadratic variation t and is a standard Brownian motion. Integration proves Assumption 3 for the difusion learner. The projection identities, jointly with diferentiability, remove the random state from the scalar drift and volatility.

## C.2 Proof of Lemma 2

Notation and asymptotic regime. We index the integer periods by $0 , \ldots , T - 1$ in these proofs, relabeling the original candidate i as i − 1. Let $V _ { T } ( s )$ denote the corresponding continuous Gaussian value process, so $V _ { T } ( i - 1 ) = V ( \pi _ { i } )$ at the original integer checkpoints. Its coeficients are initially $\mu ( u + 1 / T )$ and $\sigma ( u + 1 / T )$ . We extend the positive $C ^ { 1 }$ functions in (C.1) across the final auxiliary interval when necessary and suppress this shift in the calculations. The shifted coeficients difer uniformly by $O ( T ^ { - 1 } )$ ; the resulting changes in B and $\textstyle \int _ { 0 } ^ { 1 } I ( u )$ du are $O ( T ^ { - 1 } )$ . They therefore change the displayed scaled regret by $O ( L _ { T } / T ) = o ( 1 )$ and do not change any leading count or risk order below.

A schedule has times $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { K } < T = t _ { K + 1 }$ , with K actual updates and $K + 1$ holds. Write $h _ { k } = t _ { k + 1 } - t _ { k }$ and $u _ { k } = t _ { k } / T$ . Only $k = 0 , \ldots , K - 1$ contribute update risk; the last interval is a terminal hold. For a fixed budget, take $\epsilon _ { T } = \epsilon > 0$ . More generally, all the estimates below are uniform in the regime

$$
T ^ { - a _ { 0 } } \le \epsilon _ { T } \le \bar { \epsilon } , \qquad L = L _ { T } = \log ( T / \epsilon _ { T } ) \asymp \log T , \qquad 0 < a _ { 0 } < \infty , \quad 0 < \bar { \epsilon } < \infty .\tag{C.5}
$$

The coeficient functions and their positive bounds are fixed as $T$ grows. Constants in the $O ( \cdot )$ and $\Theta ( \cdot )$ terms may depend on those bounds and the budget regime, but not on a particular update.

For an integer schedule $P ,$ the exact regret and risk are

$$
\mathcal { R } _ { T } ( P ) = \frac { 1 } { T } \sum _ { k = 0 } ^ { K } \sum _ { j = 0 } ^ { h _ { k } - 1 } \mathbb { E } \big [ V _ { T } ( t _ { k } + j ) - V _ { T } ( t _ { k } ) \big ] , \qquad S _ { T } ( P ) = \sum _ { k = 0 } ^ { K - 1 } \mathbb { P } \big \{ V _ { T } ( t _ { k + 1 } ) < V _ { T } ( t _ { k } ) \big \} .\tag{C.6}
$$

These are the quantities in the main problem after relabeling. To pass from sums to integrals, define the step functions

$$
\tau _ { T } ( u ) = u _ { k } , \qquad H _ { T } ( u ) = h _ { k } , \qquad u \in [ u _ { k } , u _ { k + 1 } ) .\tag{C.7}
$$

Here $H _ { T }$ records a given partition, whereas $h _ { T }$ will be the positive real-valued allocation optimized in (8). Set

$$
Q ( x ) = \Phi ( - \sqrt { 2 x } ) , \qquad f ( x ) = \frac { Q ( x ) } { x } , \qquad I ( u ) = \frac { \mu ( u ) ^ { 2 } } { 2 \sigma ( u ) ^ { 2 } } , \qquad A ( u ) = \frac { \mu ( u ) } { I ( u ) } , \qquad B = \int _ { 0 } ^ { 1 } A ( u ) \mathrm { d } u .\tag{C.8}
$$

The local probability and risk rate are

$$
q ( u , h ) = Q ( I ( u ) h ) , \qquad r ( u , h ) = \frac { q ( u , h ) } { h } = I ( u ) f ( I ( u ) h ) .\tag{C.9}
$$

For a budget $b > 0$ , let

$$
F _ { T } ^ { \mathrm { c } } ( b ) = \operatorname* { i n f } _ { h _ { T } ( \cdot ) > 0 } \left\{ \int _ { 0 } ^ { 1 } \mu ( u ) h _ { T } ( u ) \mathrm { d } u : \int _ { 0 } ^ { 1 } r ( u , h _ { T } ( u ) ) \mathrm { d } u \leq \frac { b } { T } \right\} ,\tag{C.10}
$$

where the infimum is over positive measurable functions with finite displayed integrals. We first establish the estimates needed for the objective and constraint.

Lemma C.1 (Normal-tail estimates). Let $c _ { 0 } = ( 2 \sqrt { \pi } ) ^ { - 1 }$ . The functions $Q$ and f are strictly decreasing and strictly convex on (0, ∞). As $x \to \infty$

$$
\begin{array} { l } { { Q ( x ) = c _ { 0 } e ^ { - x } x ^ { - 1 / 2 } \{ 1 + O ( x ^ { - 1 } ) \} , } } \\ { { \ } } \\ { { f ( x ) = c _ { 0 } e ^ { - x } x ^ { - 3 / 2 } \{ 1 + O ( x ^ { - 1 } ) \} , \qquad - f ^ { \prime } ( x ) = c _ { 0 } e ^ { - x } x ^ { - 3 / 2 } \{ 1 + O ( x ^ { - 1 } ) \} . } } \end{array}\tag{C.11}
$$

Consequently $f ( x ) / ( - f ^ { \prime } ( x ) ) \to 1$ and $f ^ { \prime \prime } ( x ) / ( - f ^ { \prime } ( x ) ) \to 1$ . Moreover, $- f ^ { \prime }$ decreases continuously from $+ \infty$ to zero.

Proof. For the standard normal density ϕ, integration by parts gives

$$
1 - \Phi ( z ) = { \frac { \phi ( z ) } { z } } - \int _ { z } ^ { \infty } { \frac { \phi ( y ) } { y ^ { 2 } } } \mathrm { d } y , \qquad { \frac { z \phi ( z ) } { 1 + z ^ { 2 } } } \leq 1 - \Phi ( z ) \leq { \frac { \phi ( z ) } { z } } .
$$

Substituting $z = { \sqrt { 2 x } }$ proves the expansion for Q. Direct diferentiation yields

$$
Q ^ { \prime } ( x ) = - c _ { 0 } e ^ { - x } x ^ { - 1 / 2 } , \qquad Q ^ { \prime \prime } ( x ) = c _ { 0 } e ^ { - x } \left( x ^ { - 1 / 2 } + { \frac { 1 } { 2 } } x ^ { - 3 / 2 } \right) ,
$$

$$
f ^ { \prime } ( x ) = \frac { Q ^ { \prime } ( x ) } { x } - \frac { Q ( x ) } { x ^ { 2 } } < 0 , \qquad f ^ { \prime \prime } ( x ) = \frac { Q ^ { \prime \prime } ( x ) } { x } - \frac { 2 Q ^ { \prime } ( x ) } { x ^ { 2 } } + \frac { 2 Q ( x ) } { x ^ { 3 } } > 0 .
$$

These identities prove the remaining expansions and ratios. Since $Q ( x )  1 / 2$ as $x \downarrow 0 .$ , we have $- f ^ { \prime } ( x )  \infty$ there; (C.11) gives its limit zero at infinity. Strict convexity gives the asserted monotonicity. □

Lemma C.2 (Local interval probabilities). For an interval $[ s , s + h ] \subseteq [ 0 , T ]$ , define

$$
M _ { s , h } = \int _ { s } ^ { s + h } \mu ( \boldsymbol { v } / T ) \mathrm { d } \boldsymbol { v } , \qquad Q _ { s , h } ^ { \mathrm { v a r } } = \int _ { s } ^ { s + h } \sigma ( \boldsymbol { v } / T ) ^ { 2 } \mathrm { d } \boldsymbol { v } , \qquad \mathcal { T } _ { s , h } = \frac { M _ { s , h } ^ { 2 } } { 2 Q _ { s , h } ^ { \mathrm { v a r } } } .
$$

Its exact unsafe-update probability is $q _ { T } ( s , h ) = Q ( \mathbb { Z } _ { s , h } )$ . Uniformly in admissible $s , h$

$$
M _ { s , h } = \mu ( s / T ) h + O ( h ^ { 2 } / T ) , \qquad Q _ { s , h } ^ { \mathrm { v a r } } = \sigma ( s / T ) ^ { 2 } h + O ( h ^ { 2 } / T ) , \qquad T _ { s , h } = I ( s / T ) h + O ( h ^ { 2 } / T ) .\tag{C.12}
$$

Also,

$$
\mathcal { T } _ { s , h } \leq \int _ { s } ^ { s + h } I ( v / T ) \mathrm { d } v \leq h \operatorname* { m a x } _ { v \in [ s , s + h ] } I ( v / T ) .\tag{C.13}
$$

For fixed $0 < a < d <$ ∞ and $a L \leq h \leq d L$

$$
q _ { T } ( s , h ) = q ( s / T , h ) \{ 1 + O ( L ^ { 2 } / T ) \} .\tag{C.14}
$$

Proof. Under Assumption 3, the interval increment is Gaussian with mean $M _ { s , h } / T$ and variance $Q _ { s , h } ^ { \mathrm { v a r } } / T ^ { 2 }$ , proving the exact probability formula. Bounded first derivatives give the first two expansions in (C.12). Since $Q _ { s , h } ^ { \mathrm { v a r } } \geq \sigma _ { - } ^ { 2 } h$ division gives the exponent expansion. Cauchy–Schwarz gives

$$
M _ { s , h } ^ { 2 } \leq \left( \int _ { s } ^ { s + h } \sigma ( v / T ) ^ { 2 } \mathrm { d } v \right) \left( \int _ { s } ^ { s + h } \frac { \mu ( v / T ) ^ { 2 } } { \sigma ( v / T ) ^ { 2 } } \mathrm { d } v \right) ,
$$

which proves (C.13). Finally, on $h \asymp L$ , the two exponent arguments difer by $O ( L ^ { 2 } / T ) = o ( 1 )$ and both tend to infinity. By Lemma C.1, $| ( \log Q ) ^ { \prime } |$ is bounded on these arguments. Applying the mean value theorem to log Q proves (C.14). □

Lemma C.3 (A competitive schedule and its mesh bound). There is a feasible integer schedule with regret $O ( L / T )$ Every integer schedule satisfying $\mathcal { R } _ { T } ( P ) \leq C _ { 1 } L / T$ , for a fixed $C _ { 1 }$ , obeys

$$
\sum _ { k = 0 } ^ { K } h _ { k } ^ { 2 } = O ( T L ) , \qquad \operatorname* { m a x } _ { k } h _ { k } = O ( \sqrt { T L } ) , \qquad \operatorname* { m a x } _ { k } h _ { k } / T \longrightarrow 0 .\tag{C.15}
$$

Proof. Let $\begin{array} { r } { I _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { u \in [ 0 , 1 ] } I ( u ) > 0 } \end{array}$ and take full integer holds of length $\lceil L / I _ { \mathrm { m i n } } \rceil$ , followed by a terminal remainder. By Lemma C.2, the exponent of every actual update is at least $L - O ( L ^ { 2 } / T )$ . The normal-tail upper bound gives probability at most $\hat { C L } ^ { - 1 / 2 } e ^ { - L }$ per update. There are at most $C T / L$ updates, so their total risk is at most $C \epsilon _ { T } / L ^ { 3 / 2 } \leq \epsilon _ { T }$ for suficiently large T. Bounded drift and holding lengths $O ( L )$ give regret $O ( L / T )$ .

For any integer schedule, positivity of the drift gives $\mathbb { E } [ V _ { T } ( s + j ) - V _ { T } ( s ) ] \geq \mu _ { - } j / T$ . Hence

$$
\mathcal { R } _ { T } ( P ) \geq \frac { \mu _ { - } } { 2 T ^ { 2 } } \sum _ { k = 0 } ^ { K } h _ { k } ( h _ { k } - 1 ) .
$$

Since $\textstyle \sum _ { k } h _ { k } = T$ , the assumed regret bound yields $\begin{array} { r } { \sum _ { k } h _ { k } ^ { 2 } = O ( T L ) } \end{array}$ . The maximum is at most the square root of this sum, which proves the remaining statements. □

(I) Objective. Maximizing expected deployed value is equivalent to minimizing (C.6), because the always-update benchmark is independent of the schedule. Within a hold starting at $t _ { k }$ , we have

$$
\mathbb E [ V _ { T } ( t _ { k } + j ) - V _ { T } ( t _ { k } ) ] = \frac { 1 } { T } \int _ { t _ { k } } ^ { t _ { k } + j } \mu ( v / T ) \mathrm { d } v = \frac { \mu ( u _ { k } ) j } { T } + O ( j ^ { 2 } / T ^ { 2 } ) .
$$

Summing first within each hold and then across holds gives

$$
\begin{array} { c l c r } { { \displaystyle \mathcal { R } _ { T } ( P ) = \frac { 1 } { T ^ { 2 } } \sum _ { k = 0 } ^ { K } \left\{ \mu ( u _ { k } ) \sum _ { j = 0 } ^ { h _ { k } - 1 } j + O ( h _ { k } ^ { 3 } / T ) \right\} } } \\ { { \displaystyle = \frac { 1 } { 2 T ^ { 2 } } \sum _ { k = 0 } ^ { K } \mu ( u _ { k } ) h _ { k } ( h _ { k } - 1 ) + O \Big ( \frac { \sum _ { k } h _ { k } ^ { 3 } } { T ^ { 3 } } \Big ) . } } \end{array}\tag{C.16}
$$

For a schedule in Lemma C.3,

$$
\frac { \sum _ { k } h _ { k } ^ { 3 } } { T ^ { 2 } } \le \frac { \operatorname* { m a x } _ { k } h _ { k } } { T ^ { 2 } } \sum _ { k } h _ { k } ^ { 2 } = O ( L ^ { 3 / 2 } / \sqrt { T } ) = o ( 1 ) , \qquad \frac { 1 } { T } \sum _ { k } \mu ( u _ { k } ) h _ { k } \le \mu _ { + } .
$$

Using the step functions in (C.7), we obtain

$$
\begin{array} { l } { \displaystyle 2 T \mathcal { R } _ { T } ( P ) = \frac { 1 } { T } \sum _ { k = 0 } ^ { K } \mu ( u _ { k } ) h _ { k } ^ { 2 } + O ( 1 ) } \\ { \displaystyle \qquad = \int _ { 0 } ^ { 1 } \mu ( \tau _ { T } ( u ) ) H _ { T } ( u ) \mathrm { d } u + O ( 1 ) = \int _ { 0 } ^ { 1 } \mu ( u ) H _ { T } ( u ) \mathrm { d } u + O ( 1 ) . } \end{array}\tag{C.17}
$$

The last equality follows from

$$
\left| \int _ { 0 } ^ { 1 } \left\{ \mu ( \tau _ { T } ( u ) ) - \mu ( u ) \right\} H _ { T } ( u ) \mathrm { d } u \right| \leq \frac { C } { T ^ { 2 } } \sum _ { k } h _ { k } ^ { 3 } = o ( 1 ) .
$$

Thus the integral objective represents $2 T$ times regret up to a bounded error, uniformly over this class of schedules.

(II) Constraint. For this local calculation, suppose every risk-bearing hold satisfies a $L \leq h _ { k } \leq d L$ for fixed positive $a , d .$ . By Lemma C.2,

$$
\begin{array} { c } { { \displaystyle \frac { S _ { T } ( P ) } { T } = \frac { 1 } { T } \sum _ { k = 0 } ^ { K - 1 } q ( u _ { k } , h _ { k } ) \{ 1 + O ( L ^ { 2 } / T ) \} } } \\ { { = \left\{ \displaystyle \sum _ { k = 0 } ^ { K - 1 } \frac { h _ { k } } { T } r ( u _ { k } , h _ { k } ) \right\} \{ 1 + O ( L ^ { 2 } / T ) \} . } } \end{array}\tag{C.18}
$$

On $[ u _ { k } , u _ { k + 1 } )$ , the arguments $I ( u ) h _ { k }$ and $I ( u _ { k } ) h _ { k }$ difer by $O ( h _ { k } ^ { 2 } / T )$ . The logarithmic derivative bound used above therefore gives $r ( u , h _ { k } ) = r ( u _ { k } , h _ { k } ) \{ 1 + O ( L ^ { 2 } / T ) \}$ . Consequently,

$$
{ \frac { S _ { T } ( P ) } { T } } = \left\{ \int _ { 0 } ^ { u _ { K } } r ( u , H _ { T } ( u ) ) \mathrm { d } u \right\} \{ 1 + O ( L ^ { 2 } / T ) \} .\tag{C.19}
$$

The integral stops at $u _ { K }$ because the terminal hold has no update risk. In the construction below, we upper-bound this integral by the risk integral of a smooth allocation over [0, 1]; no update is added at the terminal endpoint.

(III) Leading-order form. Replacing the partition-dependent step function by a positive allocation $h _ { T }$ gives (C.10), which is (8) at $b = \epsilon _ { T }$ . The substitution $x ( u ) = I ( u ) h _ { T } ( u )$ makes the two integrands

$$
\mu ( u ) h _ { T } ( u ) = A ( u ) x ( u ) , \qquad r ( u , h _ { T } ( u ) ) = I ( u ) f ( x ( u ) ) .
$$

The resulting optimization is

$$
F _ { T } ^ { \mathrm { c } } ( b ) = \operatorname* { i n f } _ { x ( \cdot ) > 0 } \left\{ \int _ { 0 } ^ { 1 } A ( u ) x ( u ) \mathrm { d } u : \int _ { 0 } ^ { 1 } I ( u ) f ( x ( u ) ) \mathrm { d } u \leq \frac { b } { T } \right\} .\tag{C.20}
$$

We solve this functional problem directly in the next subsection. The lower and upper bounds in Section C.5 then establish the precise optimized comparison

$$
2 T \mathcal { R } _ { T } ^ { \star } ( \epsilon _ { T } ) = F _ { T } ^ { \mathrm { c } } ( \epsilon _ { T } ) + O ( 1 ) .\tag{C.21}
$$

That comparison completes Lemma 2: its lower bound applies to arbitrary feasible integer schedules, and its upper bound constructs a feasible integer schedule.

## C.3 Proof of Theorem 1

Proof. We solve (C.20) for $b \in [ \epsilon _ { T } / 2 , \epsilon _ { T } ]$ and then take $b = \epsilon _ { T }$ . This argument concerns the functional problem itself and does not use the optimized comparison (C.21).

(I) Pointwise minimization. The Lagrangian is

$$
\mathcal { L } ( x , \lambda ) = \int _ { 0 } ^ { 1 } \left\{ A ( u ) x ( u ) + \lambda I ( u ) f ( x ( u ) ) \right\} \mathrm { d } u - \lambda \frac { b } { T } .
$$

For a fixed $\lambda > 0$ , the integrand contains no terms coupling diferent values of $u ,$ so we minimize it separately at each $u . \mathrm { \ B y }$ Lemma C.1, it is strictly convex in x and diverges as x $\downarrow 0$ or $x \uparrow \infty$ . Its unique minimizer $x _ { \lambda } ( u )$ satisfies

$$
A ( u ) + \lambda I ( u ) f ^ { \prime } ( x _ { \lambda } ( u ) ) = 0 \quad \Longleftrightarrow \quad - f ^ { \prime } ( x _ { \lambda } ( u ) ) = { \frac { \mu ( u ) } { \lambda I ( u ) ^ { 2 } } } .\tag{C.22}
$$

(II) Binding constraint. Since $- f ^ { \prime }$ is a strictly decreasing bijection from $( 0 , \infty )$ onto $( 0 , \infty )$ , the solution $x _ { \lambda } ( u )$ is continuous in $u ,$ strictly increasing in $\lambda ,$ and tends uniformly to zero or infinity as $\lambda$ tends to zero or infinity, respectively. The coeficient ratios are bounded above and away from zero. It follows that $\begin{array} { r } { \int _ { 0 } ^ { 1 } I ( u ) f ( x _ { \lambda } ( u ) ) } \end{array}$ du decreases continuously from infinity to zero. There is therefore a unique $\lambda _ { b } > 0$ such that

$$
\int _ { 0 } ^ { 1 } I ( u ) f ( x _ { \lambda _ { b } } ( u ) ) \mathrm { d } u = \frac { b } { T } .\tag{C.23}
$$

Write ${ \boldsymbol { x } } _ { b } = { \boldsymbol { x } } _ { \lambda _ { b } }$ . For every feasible x, pointwise optimality gives

$$
A ( u ) x ( u ) + \lambda _ { b } I ( u ) f ( x ( u ) ) \geq A ( u ) x _ { b } ( u ) + \lambda _ { b } I ( u ) f ( x _ { b } ( u ) ) .
$$

Integrating this inequality and using feasibility and (C.23) yields $\textstyle \int _ { 0 } ^ { 1 } A x \geq \int _ { 0 } ^ { 1 } A x _ { b }$ . Strict convexity gives uniqueness up to null sets. Thus $h _ { b } = x _ { b } / I$ is the continuous representative of the optimizer of (C.10).

(III) Multiplier and waiting-time rates. As $b / T \to 0$ , the binding equation forces $\lambda _ { b }  \infty$ , and hence $x _ { b } ( u )  \infty$ uniformly. Substituting (C.22) into (C.23) gives

$$
\begin{array} { l } { \displaystyle \frac { b } { T } = \int _ { 0 } ^ { 1 } { I ( u ) \big [ - f ^ { \prime } ( x _ { b } ( u ) ) \big ] \frac { f ( x _ { b } ( u ) ) } { - f ^ { \prime } ( x _ { b } ( u ) ) } \mathrm { d } u } } \\ { \displaystyle = \frac { 1 } { \lambda _ { b } } \int _ { 0 } ^ { 1 } { A ( u ) \frac { f ( x _ { b } ( u ) ) } { - f ^ { \prime } ( x _ { b } ( u ) ) } \mathrm { d } u } = \frac { B } { \lambda _ { b } } \{ 1 + o ( 1 ) \} . } \end{array}\tag{C.24}
$$

Consequently $\lambda _ { b } \sim B T / b$ and log $\lambda _ { b } = L _ { b } + O ( 1 )$ , where $L _ { b } = \log ( T / b ) = L + O ( 1 )$ . Taking logarithms in (C.22) and applying (C.11) yields

$$
x _ { b } ( u ) + \frac { 3 } { 2 } \log x _ { b } ( u ) = \log \lambda _ { b } + \log \biggl ( \frac { c _ { 0 } I ( u ) ^ { 2 } } { \mu ( u ) } \biggr ) + O ( x _ { b } ( u ) ^ { - 1 } ) = L _ { b } + O ( 1 ) .\tag{C.25}
$$

The error is uniform in $u .$ First, this identity gives $x _ { b } ( u ) = L _ { b } + O ( \log L _ { b } )$ : for large $x _ { b } .$ , it bounds $x _ { b }$ above by $L _ { b } + C$ , and substitution into its logarithm gives the corresponding lower bound. Therefore

$$
\log x _ { b } ( u ) = \log L _ { b } + O ( \log L _ { b } / L _ { b } ) .
$$

Substituting this back into (C.25) proves

$$
x _ { b } ( u ) = L _ { b } - \frac { 3 } { 2 } \log L _ { b } + O ( 1 ) , \qquad h _ { b } ( u ) = \frac { 1 } { I ( u ) } \left[ L _ { b } - \frac { 3 } { 2 } \log L _ { b } + O ( 1 ) \right] .\tag{C.26}
$$

Taking $b = \epsilon _ { T }$ proves Theorem 1. In particular, $h _ { b } \asymp L$ uniformly, and integration gives

$$
F _ { T } ^ { \mathrm { c } } ( b ) = \int _ { 0 } ^ { 1 } A ( u ) x _ { b } ( u ) \mathrm { d } u = B \left[ L _ { b } - \frac { 3 } { 2 } \log L _ { b } \right] + O ( 1 ) .\tag{C.27}
$$

For later use, diferentiate the logarithm of (C.22):

$$
- \frac { f ^ { \prime \prime } ( x _ { b } ( u ) ) } { - f ^ { \prime } ( x _ { b } ( u ) ) } x _ { b } ^ { \prime } ( u ) = \frac { \mu ^ { \prime } ( u ) } { \mu ( u ) } - 2 \frac { I ^ { \prime } ( u ) } { I ( u ) } .
$$

By Lemma C.1 and the coeficient bounds, $x _ { b } ^ { \prime } = O ( 1 )$ and $h _ { b } ^ { \prime } = O ( L )$ , uniformly for $b \in [ \epsilon _ { T } / 2 , \epsilon _ { T } ]$

## C.4 Proof of Corollary 1

Proof. Consider a full interval generated by $h _ { b } .$ , for $b \in [ \epsilon _ { T } / 2 , \epsilon _ { T } ]$ . Its length is $h _ { b } ( u _ { k } )$ for the continuous construction, or $\lceil h _ { b } ( u _ { k } ) \rceil$ for its integer realization. We first evaluate the local probability before rounding. By (C.22) and (C.24),

$$
\begin{array} { l } { \displaystyle q ( u , h _ { b } ( u ) ) = Q ( x _ { b } ( u ) ) = x _ { b } ( u ) f ( x _ { b } ( u ) ) } \\ { \displaystyle \qquad = \frac { x _ { b } ( u ) \mu ( u ) } { \lambda _ { b } I ( u ) ^ { 2 } } \frac { f ( x _ { b } ( u ) ) } { - f ^ { \prime } ( x _ { b } ( u ) ) } = \Theta \biggl ( \frac { \epsilon _ { T } L } { T B } \frac { \mu ( u ) } { I ( u ) ^ { 2 } } \biggr ) . } \end{array}\tag{C.28}
$$

Here $x _ { b } \asymp L , \lambda _ { b } \asymp T B / b , b \asymp \epsilon _ { T }$ , and the ratio $f / ( - f ^ { \prime } )$ is bounded above and away from zero for large T.

Write an integer length as $j = h _ { b } ( u ) + d$ with $0 \leq d < 1$ . Its local exponent changes from $x _ { b } ( u )$ to $x _ { b } ( u ) + I ( u ) d$ The normal-tail formula gives

$$
\frac { Q ( x _ { b } ( u ) + I ( u ) d ) } { Q ( x _ { b } ( u ) ) } = e ^ { - I ( u ) d } \left( \frac { x _ { b } ( u ) } { x _ { b } ( u ) + I ( u ) d } \right) ^ { 1 / 2 } \{ 1 + O ( L ^ { - 1 } ) \} .
$$

This ratio is bounded above and away from zero uniformly in $u , d .$ Since both full interval lengths are of order $L ,$ (C.14) then proves

$$
\mathbb { P } \big \{ V _ { T } ( t _ { k + 1 } ) < V _ { T } ( t _ { k } ) \big \} = \Theta \bigg ( \frac { \epsilon _ { T } L } { T B } \frac { \mu ( u _ { k } ) } { I ( u _ { k } ) ^ { 2 } } \bigg ) , \qquad k = 0 , \dots , K - 1 .\tag{C.29}
$$

Reversing the index shift gives the main paper’s policy-value notation for integer updates. The feasible integer schedule used in the regret bound below is the rounded construction with $b = \epsilon _ { T } / 2 .$ , so it satisfies the same risk order. □

## C.5 Proof of Corollary 2

Proof. Let $\begin{array} { r } { z _ { L } = L - \frac { 3 } { 2 } } \end{array}$ log L and $\begin{array} { r } { D = \int _ { 0 } ^ { 1 } I ( u ) } \end{array}$ du. We first match a lower bound over feasible integer schedules with a feasible construction. We then establish the leading count for an optimal integer schedule.

(I) Lower bound by weighted Jensen’s inequality. By Lemma C.3, we only need to consider schedules with regret at most $C _ { 1 } L / T$ , for a suficiently large fixed $C _ { 1 }$ . For each risk-bearing hold, define

$$
\mu _ { k } ^ { - } = \operatorname* { m i n } _ { v \in [ t _ { k } , t _ { k + 1 } ] } \mu ( v / T ) , \qquad I _ { k } ^ { + } = \operatorname* { m a x } _ { v \in [ t _ { k } , t _ { k + 1 } ] } I ( v / T ) , \qquad x _ { k } = I _ { k } ^ { + } h _ { k } , \qquad w _ { k } = \frac { h _ { k } } { T } \frac { \mu _ { k } ^ { - } } { I _ { k } ^ { + } } .
$$

Put $\begin{array} { r } { W _ { T } = \sum _ { k < K } w _ { k } } \end{array}$ and $\begin{array} { r } { Z _ { T } = \sum _ { k < K } w _ { k } x _ { k } } \end{array}$ . Coeficient smoothness and the mesh bound imply

$$
W _ { T } = B + O \bigg ( \frac { \sum _ { k } h _ { k } ^ { 2 } } { T ^ { 2 } } + \frac { h _ { K } } { T } \bigg ) = B + O \bigg ( \sqrt { L / T } \bigg ) .\tag{C.30}
$$

Indeed, $\mu _ { k } ^ { - } / I _ { k } ^ { + }$ difers from $A ( u )$ by at most $C h _ { k } / T$ within its interval, and the omitted terminal interval has width $h _ { K } / T$ . In particular, $W _ { T }$ is positive and bounded away from zero for large $T$

The exact Gaussian exponent bound (C.13) gives q<sub>T</sub> $( t _ { k } , h _ { k } ) \geq Q ( x _ { k } )$ . Let I<sub>−</sub> = min $I > 0$ and $d _ { 0 } = I _ { - } ^ { 2 } / \mu _ { + }$ . Then

$$
\begin{array} { l } { { \displaystyle S _ { T } ( P ) \geq \sum _ { k < K } Q ( x _ { k } ) = T \sum _ { k < K } \frac { h _ { k } } { T } I _ { k } ^ { + } f ( x _ { k } ) } } \\ { { \displaystyle \qquad = T \sum _ { k < K } w _ { k } \frac { ( I _ { k } ^ { + } ) ^ { 2 } } { \mu _ { k } ^ { - } } f ( x _ { k } ) \geq d _ { 0 } T W _ { T } f \biggl ( \frac { Z _ { T } } { W _ { T } } \biggr ) . } } \end{array}\tag{C.31}
$$

The last inequality uses convexity of f and Jensen’s inequality with weights $w _ { k } / W _ { T }$ . For a feasible schedule, it follows that

$$
f \left( \frac { Z _ { T } } { W _ { T } } \right) \leq \frac { \epsilon _ { T } } { d _ { 0 } T W _ { T } } .
$$

Let $y _ { T }$ solve $f ( y _ { T } ) = \epsilon _ { T } / ( d _ { 0 } T W _ { T } )$ . The solution is unique because f decreases from infinity to zero. Since $W _ { T }$ stays between fixed positive bounds, (C.11) gives

$$
y _ { T } + \frac { 3 } { 2 } \log y _ { T } = L + O ( 1 ) , \qquad y _ { T } = L - \frac { 3 } { 2 } \log L + O ( 1 ) .
$$

Monotonicity of f therefore yields $Z _ { T } \ge W _ { T } \{ z _ { L } + O ( 1 ) \} = B z _ { L } + O ( 1 )$ ; here (C.30) gives $( W _ { T } - B ) L = o ( 1 )$

Within an integer hold, the mean increment is at least $\mu _ { k } ^ { - } j / T$ . Including the terminal cost, which is nonnegative, we obtain

$$
2 T \mathcal { R } _ { T } ( P ) \geq \frac { 1 } { T } \sum _ { k < K } \mu _ { k } ^ { - } h _ { k } ( h _ { k } - 1 ) \geq Z _ { T } - \mu _ { + } .
$$

Consequently,

$$
2 T \mathcal { R } _ { T } ^ { \star } ( \epsilon _ { T } ) \geq B \left[ L - \frac { 3 } { 2 } \log L \right] - O ( 1 ) .\tag{C.32}
$$

Schedules outside the competitive class have a larger regret than this bound for suficiently large $C _ { 1 }$

(II) A feasible integer upper bound. Take $b = \epsilon _ { T } / 2$ and use the continuum optimizer $h _ { b }$ from Section C.3. Starting at $t _ { 0 } = 0$ , propose the integer length

$$
j _ { k } = \lceil h _ { b } ( t _ { k } / T ) \rceil , \qquad t _ { k + 1 } = t _ { k } + j _ { k } \quad \mathrm { w h e n ~ } t _ { k } + j _ { k } < T .\tag{C.33}
$$

When the proposed endpoint is at least $T ,$ retain the incumbent to $T .$ . Thus every full hold has length $\Theta ( L )$ , and the terminal hold is at most CL. Denote this schedule by $\widehat { P } _ { T }$ and its step function by $H _ { T }$

We verify feasibility directly. Since $h _ { b } ^ { \prime } = O ( L )$ and every hold has width $O ( L / T )$ in normalized time, on each full interval

$$
H _ { T } ( u ) = \lceil h _ { b } ( u _ { k } ) \rceil \geq h _ { b } ( u ) - C L ^ { 2 } / T .
$$

The rate $r ( u , h )$ decreases in $h ,$ and for $h \asymp L$ its logarithmic derivative in h is bounded:

$$
\frac { \partial } { \partial h } \log r ( u , h ) = I ( u ) \frac { Q ^ { \prime } ( I ( u ) h ) } { Q ( I ( u ) h ) } - \frac { 1 } { h } = O ( 1 ) .
$$

It follows that $r ( u , H _ { T } ( u ) ) \leq r ( u , h _ { b } ( u ) ) \{ 1 + O ( L ^ { 2 } / T ) \}$ on the full intervals. Using (C.19) and the binding budget of $h _ { b }$ gives

$$
S _ { T } ( \widehat { P } _ { T } ) \leq T \{ 1 + O ( L ^ { 2 } / T ) \} \int _ { 0 } ^ { u _ { K } } r ( u , h _ { b } ( u ) ) \mathrm { d } u \leq b \{ 1 + O ( L ^ { 2 } / T ) \} \leq \epsilon _ { T }\tag{C.34}
$$

for suficiently large T. The fixed half-budget leaves room for both the local approximation and rounding errors. For the objective, every hold, including the terminal remainder, satisfies

$$
H _ { T } ( u ) \leq h _ { b } ( u ) + 1 + C L ^ { 2 } / T .
$$

The schedule has regret $O ( L / T )$ because its holds are $O ( L )$ , so (C.17) applies. Hence

$$
\begin{array} { l } { 2 T \mathcal { R } _ { T } ( \widehat { P } _ { T } ) = \displaystyle \int _ { 0 } ^ { 1 } \mu ( u ) H _ { T } ( u ) \mathrm { d } u + O ( 1 ) } \\ { \displaystyle \leq F _ { T } ^ { \mathrm { c } } ( \epsilon _ { T } / 2 ) + O ( 1 ) = B \left[ L - \frac { 3 } { 2 } \log L \right] + O ( 1 ) . } \end{array}\tag{C.35}
$$

The last equality uses (C.27) and $\log ( T / b ) = L + \log 2$ . Combining (C.32) and (C.35) proves

$$
\mathcal { R } _ { T } ^ { \star } ( \epsilon _ { T } ) = \frac { B } { 2 T } \left[ L _ { T } - \frac { 3 } { 2 } \log L _ { T } \right] + O ( T ^ { - 1 } ) .\tag{C.36}
$$

Together with (C.27) at $b = \epsilon _ { T }$ , this also proves $\left( \mathrm { C . 2 1 } \right)$ and completes the optimized continuum reduction in Lemma 2. The constructed schedule satisfies $0 \leq \mathcal { R } _ { T } ( \widehat { P } _ { T } ) - \mathcal { R } _ { T } ^ { \star } ( \epsilon _ { T } ) = O ( T ^ { - 1 } )$ and, on every risk-bearing interval,

$$
I ( u _ { k } ) h _ { k } = L - \frac { 3 } { 2 } \log L + O ( 1 ) .
$$

(III) Counting the constructed schedule. For a full hold of (C.33), (C.26) gives $h _ { k } = L \{ 1 + o ( 1 ) \} / I ( u _ { k } )$ uniformly. Therefore

$$
{ \frac { L } { T } } K ( { \widehat { P } } _ { T } ) = \sum _ { k < K } { \frac { h _ { k } } { T } } { \frac { L } { h _ { k } } } = \sum _ { k < K } { \frac { h _ { k } } { T } } I ( u _ { k } ) \{ 1 + o ( 1 ) \} \longrightarrow \int _ { 0 } ^ { 1 } I ( u ) \mathrm { d } u = D .
$$

The last step is a Riemann sum, since the mesh is $O ( L / T )$ and the terminal interval has vanishing normalized length.

(IV) Counting an optimal schedule. Let $P _ { T } ^ { \star }$ be any optimal integer schedule. The upper bound and Lemma C.3 give its mesh bound. By (C.17) and (C.36),

$$
{ \frac { 1 } { T L } } \sum _ { k = 0 } ^ { K _ { T } ^ { \star } } \mu ( u _ { k } ) h _ { k } ^ { 2 } \longrightarrow B .\tag{C.37}
$$

Fix $\delta \in ( 0 , 1 )$ and call a risk-bearing interval short if $I ( u _ { k } ) h _ { k } \leq ( 1 - \delta ) L$ . Such an interval has length at most $C L$ By (C.12), its exact Gaussian exponent is at most $( 1 - \delta ) L + O ( L ^ { 2 } / T )$ . The normal-tail lower bound therefore gives risk at least $c L ^ { - 1 / 2 } e ^ { - ( 1 - \delta ) L }$ . Feasibility bounds the number $N _ { \mathrm { s h o r t } }$ of these intervals by

$$
N _ { \mathrm { s h o r t } } \leq C \epsilon _ { T } \sqrt { L } e ^ { ( 1 - \delta ) L } = C T \sqrt { L } e ^ { - \delta L } = o ( T / L ) .\tag{C.38}
$$

Their total calendar fraction is $o ( 1 )$ because each is at most $C L$ . The terminal fraction is also $o ( 1 )$ by (C.15). Let $G _ { \delta }$ contain the remaining risk-bearing intervals and put $z _ { k } = I ( u _ { k } ) h _ { k } / L$ . Then $z _ { k } > 1 - \delta$ on $G _ { \delta } .$ , and

$$
\sum _ { k \in G _ { \delta } } \frac { h _ { k } } { T } A ( u _ { k } ) \longrightarrow B , \qquad \sum _ { k \in G _ { \delta } } \frac { h _ { k } } { T } I ( u _ { k } ) \longrightarrow D .
$$

Both statements follow from the mesh bound, smoothness, and the negligible calendar fractions just removed. Using nonnegativity of the omitted costs in (C.37), we get

$$
\sum _ { k \in G _ { \delta } } \frac { h _ { k } } { T } A ( u _ { k } ) \bigl \{ z _ { k } - ( 1 - \delta ) \bigr \} \le \delta B + o ( 1 ) .\tag{C.39}
$$

For $\eta > 0 .$ let $W _ { \mathrm { l o n g } }$ be the calendar fraction of intervals in $G _ { \delta }$ with $z _ { k } > 1 + \eta$ . Since $A _ { - } : = \operatorname* { m i n } A > 0$ , (C.39) implies

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { w } _ { W _ { \mathrm { l o n g } } } \leq \frac { \delta B } { A _ { - } ( \eta + \delta ) } .
$$

We can now bound the count from both sides. From $z _ { k } > 1 - \delta$ and (C.38),

$$
\operatorname* { l i m } _ { T \to \infty } \operatorname* { s u p } _ { T } \frac { L K _ { T } ^ { \star } } { T } = \operatorname* { l i m } _ { T \to \infty } \operatorname* { s u p } _ { k \in G _ { \delta } } \frac { h _ { k } } { T } \frac { I ( u _ { k } ) } { z _ { k } } \leq \frac { D } { 1 - \delta } .
$$

For the lower bound, retain only $k \in G _ { \delta }$ with $z _ { k } \le 1 + \eta$ and use $I _ { + } : = I$ max $I { : }$

$$
\operatorname * { l i m } _ { T \to \infty } \operatorname * { i n f } _ { T } \frac { L K _ { T } ^ { \star } } { T } \geq \frac { 1 } { 1 + \eta } \left\{ D - \frac { I _ { + } \delta B } { A _ { - } ( \eta + \delta ) } \right\} .
$$

Letting $\delta \downarrow 0$ and then $\eta \downarrow 0$ makes the two bounds equal to D. Thus

$$
K _ { T } ^ { \star } \sim \frac { T } { L _ { T } } \int _ { 0 } ^ { 1 } I ( u ) \mathrm { d } u ,\tag{C.40}
$$

which proves the count assertion in Corollary 2 for an optimal integer schedule, not just for the upper-bound construction. □

## D Experimental Details

We describe the data generation, policy learning, value evaluation, and scheduling procedures implemented in the experiment scripts. The synthetic and clinical learning experiments supply estimated candidate-value trajectories to the same scheduling routines. The homogeneous Gaussian experiment instead specifies the candidate-value process directly, which also permits evaluation of a selected schedule using analytic population rewards and risks.

## D.1 Synthetic learning environments

Observations and treatment efects. For each of the low-, medium-, and high-SNR settings in Figures 4 and $5 ,$ we generate $N = 4 0 0$ learning trajectories, each containing 300 observations. We use r for the observation index and t for the policy checkpoint: a candidate is retained after every five observations, giving $T = 6 0$ candidates at sample sizes $5 , 1 0 , \ldots , 3 0 0$ . For each trajectory and observation, the five covariates are independent and uniformly distributed on $[ - 1 , 1 ]$ , treatment is assigned independently with probability $1 / 2 ,$ , and the standardized outcome innovation is Gaussian:

$$
X _ { r j } \overset { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } ( - 1 , 1 ) , \quad j = 1 , \mathrm { \ldots , 5 } , \qquad A _ { r } \sim \mathrm { B e r n o u l l i } ( 1 / 2 ) , \qquad \xi _ { r } \sim \mathcal { N } ( 0 , 1 ) .\tag{D.1}
$$

Writing s for the SNR setting, we define

$$
\begin{array} { r l r } & { ~ m ( x ) = 0 . 5 0 x _ { 3 } + 0 . 2 5 x _ { 4 } - 0 . 2 5 x _ { 5 } , } & { \tau ^ { \star } ( x ) = \sqrt { 3 / 2 } ( x _ { 1 } + x _ { 2 } ) , } \\ & { Y _ { s , r } ( a ) = m ( X _ { r } ) + ( a - 1 / 2 ) \tau ^ { \star } ( X _ { r } ) + \sigma _ { Y , s } \xi _ { r } , } & { Y _ { s , r } = Y _ { s , r } ( A _ { r } ) . } \end{array}\tag{D.2}
$$

Thus, all three settings share the same conditional treatment efect and the same optimal treatment rule $\pi ^ { \star } ( x ) =$ $\mathbf { 1 } \{ \tau ^ { \star } ( x ) \geq 0 \}$ . The generator uses seed 20260922 and common draws of $X _ { r } , A _ { r } ,$ , and $\xi _ { r }$ across settings. Trajectories are generated independently within each setting, while corresponding trajectories across settings are paired through these common random numbers. Both potential outcomes use the same innovation within an observation; only the observed outcome enters policy training.

Sequential policy learner. Let $\phi ( \boldsymbol { x } ) = ( 1 , \boldsymbol { x } ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { 6 }$ . We fit the outcome mode $\widehat { m } _ { r } ( x , a ) = \beta _ { r } ^ { \top } \phi ( x ) + ( a -$ $1 / 2 ) \gamma _ { r } ^ { \top } \phi ( x )$ by a single stochastic-gradient update after each observation, starting from $\beta _ { 0 } = \gamma _ { 0 } = 0$ . With $\lambda _ { \mathrm { s y n } } = 1 0 ^ { - 4 }$ , the implemented updates are

$$
\begin{array} { r l } & { \quad e _ { r } = Y _ { s , r } - \beta _ { r - 1 } ^ { \top } \phi ( X _ { r } ) - ( A _ { r } - 1 / 2 ) \gamma _ { r - 1 } ^ { \top } \phi ( X _ { r } ) , } \\ & { \eta _ { s , r } = \eta _ { 0 , s } ( 1 + r / 1 0 0 ) ^ { - 0 . 6 } , } \\ & { \quad \beta _ { r } = ( 1 - \eta _ { s , r } \lambda _ { \mathrm { s y n } } ) \beta _ { r - 1 } + \eta _ { s , r } e _ { r } \phi ( X _ { r } ) , } \\ & { \quad \gamma _ { r } = ( 1 - \eta _ { s , r } \lambda _ { \mathrm { s y n } } ) \gamma _ { r - 1 } + \eta _ { s , r } e _ { r } ( A _ { r } - 1 / 2 ) \phi ( X _ { r } ) . } \end{array}\tag{D.3}
$$

Both updates use the residual computed before updating either coeficient vector, and the penalty applies to all their coordinates, including the intercepts. The retained policy at checkpoint t is $\pi _ { s , t } ( \boldsymbol { x } ) = \mathbf { 1 } \{ \gamma _ { 5 t } ^ { \top } \phi ( \boldsymbol { x } ) \ge 0 \}$ . The first retained candidate processed five observations. These checkpoints group five successive one-observation updates. The settings are configured as

<table><tr><td>Setting</td><td>Initial learning rate  $\eta _ { 0 , s }$ </td><td>Outcome-noise scale  $\sigma _ { Y , s }$ </td></tr><tr><td>Low SNR</td><td>0.030</td><td>7.0</td></tr><tr><td>Medium SNR</td><td>0.150</td><td>2.5</td></tr><tr><td>High SNR</td><td>0.200</td><td>0.25</td></tr></table>

Candidate-value evaluation. We draw a common bank of $M = 8 1 9 2$ evaluation covariates independently of the training observations, using the same covariate distribution. The bank is reused for every checkpoint, trajectory, and SNR setting. Since the treatment efect is known in this experiment, we evaluate the policy-dependent outcome contrast directly and normalize it by the corresponding contrast of the oracle policy:

$$
\widehat { V } _ { s , t } ^ { ( n ) } = \frac { M ^ { - 1 } \sum _ { \ell = 1 } ^ { M } \big [ \pi _ { s , t } ^ { ( n ) } ( X _ { \ell } ^ { \mathrm { e v } } ) - 1 / 2 \big ] \tau ^ { \star } ( X _ { \ell } ^ { \mathrm { e v } } ) } { M ^ { - 1 } \sum _ { \ell = 1 } ^ { M } \big [ \pi ^ { \star } ( X _ { \ell } ^ { \mathrm { e v } } ) - 1 / 2 \big ] \tau ^ { \star } ( X _ { \ell } ^ { \mathrm { e v } } ) } .\tag{D.4}
$$

The omitted prognostic term is common to every policy and therefore cancels in value diferences. No evaluation outcomes or additional outcome noise are generated for this calculation. The resulting values are finite-bank estimates of a normalized population contrast, not exact population integrals. This policy-value normalization precedes, and difers from, the never-update versus always-update normalization of the scheduling objective described below.

The generator also summarizes the first 25 checkpoint increments. It averages their across-trajectory means and divides by the square root of the average across-trajectory sample variances. The saved empirical SNRs are approximately 0.0496, 0.1145, and 0.2816 for the low, medium, and high settings, respectively.

Homogeneous Gaussian experiment. The phase-diagram script associated with Figure 3 is configured with $T = 5 1 2$ , safety budget $\epsilon = 0 . 3 5$ , and risk-grid width $\Delta = 0 . 0 0 1$ . It uses 50 equally spaced drift values in [0.5, 3.0] and 50 equally spaced volatility values in [0.25, 1.5]. At every parameter pair, the experiment uses three data replications, each containing 4000 independent trajectories. With candidates indexed here as $1 , \ldots , T _ { \astrosun }$ , the simulated process is

$$
V _ { 1 } ^ { ( b , n ) } = 0 . 1 5 , \qquad V _ { t + 1 } ^ { ( b , n ) } = V _ { t } ^ { ( b , n ) } + \frac { \mu } { T } + \frac { \sigma } { T } Z _ { t } ^ { ( b , n ) } , \qquad t = 1 , \ldots , T - 1 ,\tag{D.5}
$$

where the $Z _ { t } ^ { ( b , n ) }$ are independent standard normal variables. The innovation generator uses seed 20260921 and reuses the same innovation bank across all parameter pairs. It stores $T$ innovation rows per trajectory; only the first $T - 1$ rows are needed because the terminal node has no policy value. In particular, the per-step noise standard deviation in (D.5) is $\sigma / T$ , not $\sigma / { \sqrt { T } }$

For each replication, the script estimates edge rewards and strict unsafe-update frequencies from its 4000 value trajectories and solves the conservatively gridded dynamic program. It then evaluates that selected path using the analytic population matrices

$$
\mathbf { } R _ { i j } = \frac { j - i } { T } \left( 0 . 1 5 + \frac { \mu ( i - 1 ) } { T } \right) , \qquad S _ { i j } = \Phi \left( - \frac { \mu \sqrt { j - i } } { \sigma } \right) \quad ( j \leq T ) , \qquad S _ { i , T + 1 } = 0 .\tag{D.6}
$$

The displayed heatmaps average the three replication-specific path summaries. For a path with $K > 0$ actual updates, the summaries are

$$
\bar { p } = \frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } S _ { t _ { k } , t _ { k + 1 } } , \qquad \bar { h } = \frac { 1 } { K T } \sum _ { k = 0 } ^ { K - 1 } ( t _ { k + 1 } - t _ { k } ) , \qquad J = \sum _ { k = 0 } ^ { K } { R } _ { t _ { k } , t _ { k + 1 } } .\tag{D.7}
$$

The initial and terminal nodes are excluded from $K ,$ and the terminal hold is excluded from both pre-update averages. If the selected path makes no updates, the display convention is $K = 0 , \bar { p } = 0$ , and $\bar { h } = 1$ . Thus, the plotted interval is a fraction of the horizon, whereas $h _ { T }$ in the theory is measured in periods. The script also records total population risk, unrounded empirical risk, the coeficient of variation of pre-update intervals, and the indicator that population risk exceeds ϵ. An oracle gridded schedule is computed separately from the population matrices The exact unrounded oracle-label solver is optional and is not enabled in the default run.

Theoretical contours and correlations. The default contours use the leading homogeneous formulas, with $I = \mu ^ { 2 } / ( 2 \sigma ^ { 2 } )$ and $L = \log ( T / \epsilon )$

$$
\begin{array} { l l l } { { h _ { \mathrm { r e f } } = L / I , } } & { { K _ { \mathrm { r e f } } = T I / L , } } & { { p _ { \mathrm { r e f } } = \epsilon L / ( T I ) , } } \\ { { \bar { h } _ { \mathrm { r e f } } = L / ( T I ) , } } & { { J _ { \mathrm { r e f } } = J _ { \mathrm { A U } } - \displaystyle \frac { \mu L } { 2 T I } , } } & { { J _ { \mathrm { A U } } = 0 . 1 5 + \displaystyle \frac { \mu ( T - 1 ) } { 2 T } . } } \end{array}\tag{D.8}
$$

The default does not include the log L correction. A reference cell is masked if $h _ { \mathrm { r e f } } < 5 , h _ { \mathrm { r e f } } / T > 0 . 1 5 .$ , or its reference regret is negative; it is not clipped to an integer spacing or update count. The configured contour levels are 10, 20, 40, 60 for K, 0.005, 0.01, 0.02, 0.04 for $\bar { p } , 0 . 0 1 , 0 . 0 2 , 0 . 0 4 , 0 . 0 8$ for ${ \bar { h } } ,$ and 0.35, 0.45, 0.55, 0.65 for $^ { J , }$ retaining only levels inside the finite range of the corresponding reference surface. For each quantity and solver, Pearson correlation is computed after flattening the cells on which both the reference and the solver summary are finite. The three replications are averaged before this comparison, rather than being treated as separate grid cells.

Introductory illustration. Figure 1 is constructed from the medium-SNR value bank. It selects the trajectory minimizing the mean squared distance to the ensemble mean after dividing each checkpoint’s deviation by its across-trajectory sample standard deviation, with a denominator floor of $1 0 ^ { - 1 \bar { 2 } }$ . In the saved bank this is trajectory 361, using zero-based trajectory identifiers. The three schedules update every 24, 12, and 6 checkpoints. Unsafe markers compare successive deployed values strictly on the selected realized trajectory, whereas the displayed fractions of expected gain captured use the across-trajectory mean value sequence and the never-update/always-update normalization. Those fractions are clipped to [0, 1] for this illustration only.

## D.2 International Stroke Trial analysis

Population, treatment, and outcome. We use the corrected version-2 IST file IST\_corrected.csv. Its 19,435 records yield 19,285 analysis records after removing 150 observations with missing or unknown six-month outcome codes: 97 with OCCODE= 0 and 53 with OCCODE= 9. Treatment is the randomized aspirin assignment, $A = 1$ for RXASP=Y and $A = 0$ for RXASP=N, with known propensity $1 / 2$ . We define $Y = 1$ for OCCODE∈ {3, 4}, corresponding to being alive and independent at six months, and $Y = 0$ for codes 1 or 2, corresponding to death or dependence. The retained sample contains 9639 aspirin-assigned and 9646 non-aspirin-assigned patients, with 7160 positive outcomes. We do not restrict this sample by a post-randomization final diagnosis. Concurrent randomized heparin assignment, treatment adherence, post-randomization outcomes, identifiers, geographic variables, and dates are no included among the policy covariates.

Covariates and preprocessing. The cleaning script produces 35 numeric covariates with the following encodings. Binary $\tt Y / N$ fields are coded as $1 / 0 ;$ missing values in the binary clinical fields are filled with zero, with separate missingness indicators for atrial fibrillation, prior heparin, and prior aspirin.
<table><tr><td>Source fields</td><td>Encoding</td><td>Count</td></tr><tr><td>AGE, RDELAY, RSBP</td><td>Age, delay to randomization, and systolic blood pressure in their original units</td><td>3</td></tr><tr><td>SEX</td><td>Indicator for female sex</td><td>1</td></tr><tr><td>RSLEEP, RATRIAL, RCT, RVISINF, RHEP24, RASP3</td><td>Six binary clinical indicators and three missingness indicators</td><td>9</td></tr><tr><td>RCONSC</td><td>Drowsy and unconscious indicators; fully alert is the reference</td><td>2</td></tr><tr><td>RDEF1-RDEF8</td><td>Present and cannot-assess indicators for each deficit; absent is the reference</td><td>16</td></tr><tr><td>STYPE</td><td>Indicators for PACS, TACS, POCS, and OTH; LACS is the reference</td><td>4</td></tr></table>

The cleaner checks that the resulting model fields are numeric and complete, and rejects missing continuous measurements rather than imputing them. It creates a reproducibly shufled stream using seed 20260922. This ordering is a simulation device, not a clinical arrival-time ordering.

We split the retained records into 15,428 development records and 3857 evaluation records using a 20% evaluation fraction, split seed 20260921, and joint treatment–outcome strata encoded by $2 A + Y$ . Every covariate is centered and scaled using the development mean and population-standard-deviation convention, and the same transformation is applied to evaluation covariates. Coordinates with development standard deviation at most $1 0 ^ { - 1 0 }$ would be removed; none is removed for the supplied split. The transformation is fixed before constructing learning trajectories, rather than recomputed at each checkpoint.

Bootstrap learning trajectories and checkpoints. Using random-number seed 20260922, we draw $N = 5 0 0$ bootstrap streams with replacement from the development pool. Each stream contains 15,250 sampled records. Along a stream, we fit policies on nested prefixes of sizes

$$
n _ { t } = 4 0 0 0 + 2 5 0 ( t - 1 ) , \qquad t = 1 , \ldots , 4 6 .\tag{D.9}
$$

Consequently, the clinical scheduling horizon is $T = 4 6$ policy checkpoints, not 60 patient observations. All 15,428 development records remain eligible for sampling; 15,250 is the largest fitted bootstrap sample size. Bootstrap streams can share patient records, and a record can appear more than once within a stream. Every fitted policy is evaluated on the same held-out evaluation sample. The separate disjoint length-60 trajectory file written by the cleaner is not used by the frontier or schedule-diagram scripts; those scripts load ist\_safe\_policy\_stream.csv and construct the bootstrap streams just described.

Ridge outcome model and treatment rule. For standardized covariates $\boldsymbol { x } ~ \in ~ \mathbb { R } ^ { 3 5 }$ , define $z ( x , a ) \ =$ $( 1 , x ^ { \top } , a , a x ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { 7 2 }$ . At checkpoint t of trajectory n, we use the ridge estimate

$$
\widehat { b } _ { t } ^ { ( n ) } = \left( \sum _ { r = 1 } ^ { n _ { t } } z _ { r } z _ { r } ^ { \top } + 5 P \right) ^ { - 1 } \sum _ { r = 1 } ^ { n _ { t } } z _ { r } Y _ { r } , \qquad P = \mathrm { d i a g } ( 0 , 1 , \ldots , 1 ) .\tag{D.10}
$$

The cross-products are updated cumulatively as each block of 250 observations arrives. The penalty is applied to the unnormalized sum-of-squares problem and does not penalize the overall intercept. For evaluation, the arm-specific predictions are clipped to [0, 1] before defining the policy:

$$
\widehat { m } _ { t , a } ^ { ( n ) } ( x ) = \operatorname* { m i n } \Bigl \{ 1 , \operatorname* { m a x } \Bigl \{ 0 , z ( x , a ) ^ { \top } \widehat { b } _ { t } ^ { ( n ) } \Bigr \} \Bigr \} , \qquad \pi _ { t } ^ { ( n ) } ( x ) = \mathbf { 1 } \Bigl \{ \widehat { m } _ { t , 1 } ^ { ( n ) } ( x ) > \widehat { m } _ { t , 0 } ^ { ( n ) } ( x ) \Bigr \} .\tag{D.11}
$$

Thus ties between the clipped predictions assign no aspirin. The fitted outcome model supplies both the treatment rule and the nuisance predictions for evaluation; a separate outcome learner is not fitted for the evaluation score.

Held-out value estimates. For each fitted candidate, we compute its augmented inverse-propensity-weighted value on the common evaluation sample E:

$$
\widehat { V } _ { t } ^ { ( n ) } = \frac { 1 } { | \mathcal { E } | } \sum _ { \ell \in \mathcal { E } } \left[ \widehat { m } _ { t , \pi _ { t } ^ { ( n ) } ( X _ { \ell } ) } ^ { ( n ) } ( X _ { \ell } ) + \frac { \mathbf { 1 } \{ A _ { \ell } = \pi _ { t } ^ { ( n ) } ( X _ { \ell } ) \} } { p _ { \pi _ { t } ^ { ( n ) } ( X _ { \ell } ) } } \left\{ Y _ { \ell } - \widehat { m } _ { t , A _ { \ell } } ^ { ( n ) } ( X _ { \ell } ) \right\} \right] , \qquad p _ { 0 } = p _ { 1 } = 1 / 2 .\tag{D.12}
$$

The final score and its average are not clipped. Neither policy fitting nor covariate standardization uses evaluation outcomes. However, the resulting value estimates are subsequently used both to select schedules and to report their frontier values; this experiment does not use another independent evaluation sample for the selected meta-policy.

## D.3 Baselines, budgets, and reporting

Shared scheduling inputs. Within each setting, let $\widehat V _ { i } ^ { ( n ) }$ denote the saved or estimated value of candidate i in trajectory n. The frontier and schedule-diagram scripts use the direct mean $\begin{array} { r } { \widehat { \nu } _ { i } = N ^ { - 1 } \sum _ { n } \widehat { V } _ { i } ^ { ( n ) } } \end{array}$ and the empirical matrices

$$
\widehat { \pmb { R } } _ { i j } = \frac { j - i } { T } \widehat { \pmb { \nu } } _ { i } , \qquad \widehat { \pmb { S } } _ { i j } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } { \bf 1 } \left\{ \widehat { V } _ { j } ^ { ( n ) } < \widehat { V } _ { i } ^ { ( n ) } \right\} \quad ( j \leq T ) , \qquad \widehat { \pmb { S } } _ { i , T + 1 } = 0 .\tag{D.13}
$$

The implementation uses zero-based learned-policy nodes $0 , \ldots , T - 1$ and a separate sink $T ,$ equivalent to the paper’s nodes $1 , \ldots , T$ and sink $T + 1$ . In particular, the final learned candidate can contribute reward for its last deployment period. All methods use the same mean values and edge risks. A tie in estimated candidate values is not counted towards an unsafe update.

The common risk-grid width is $\Delta = 0 . 0 0 1$ . Write $C _ { i j } = \Delta \lceil \widehat { \pmb { S } } _ { i j } / \Delta \rceil$ for the rounded resource in risk units, with terminal resource zero. The proposed method applies Algorithm 1 using these upward-rounded costs and a downward-rounded budget. Its implementation accumulates $( j - i ) \widehat \nu _ { i }$ , rather than $( j - i ) \widehat { \nu } _ { i } / T$ ; this common factor of $T$ does not change the selected schedule or the normalized objective. It computes the frontier up to the rounded cost $\begin{array} { r } { \epsilon _ { \mathrm { A U } } = \sum _ { i = 1 } ^ { T - 1 } \breve { C } _ { i , i + 1 } } \end{array}$ of updating at every checkpoint, then reuses the retained terminal labels at the plotted budgets.

Baseline definitions. The periodic baseline enumerates every integer period $h \in \{ 1 , \ldots , T \}$ , deploys candidates $1 , 1 + h , 1 + 2 h , . .$ . while their indices do not exceed $T ,$ , and then holds the final incumbent through the terminal node. It retains nondominated candidates with total rounded resource at most $\epsilon _ { \mathrm { A U } }$ , including the no-update schedule, and selects the highest-value feasible candidate at each plotted budget.

The equal-risk baseline allows nonperiodic timing, but requires a common risk allowance for all actual updates. For a path $P$ with $K > 0$ updates, its required budget is

$$
\epsilon _ { \mathrm { e q } } ( P ) = K \operatorname* { m a x } _ { ( i , j ) \in P , j \leq T } C _ { i j } .\tag{D.14}
$$

Thus feasibility at ϵ is equivalent to requiring every actual transition to cost at most $\epsilon / K$ . A dynamic program enumerates update counts $K = 1 , \ldots , T - 1$ and tracks the largest edge resource and cumulative value, with dominance pruning within each count layer. The no-update path is included separately.

The earliest- ${ \it s a f e }$ baseline sweeps a threshold over zero and every distinct finite rounded resource of an actual transition. For each threshold, it starts at the initial candidate and repeatedly selects the earliest later candidate whose edge resource does not exceed the threshold; if none exists, it retains the incumbent to the terminal node. Duplicate paths are removed. The method then selects the highest-value generated path whose total additive rounded resource fits the plotted budget. The threshold is therefore tuned over the complete finite set at which the greedy path can change, not over an externally chosen coarse threshold grid.

In the frontier routines, a Bellman predecessor is replaced only when its value improves by more than $1 0 ^ { - 1 2 }$ , so tied candidates retain the first predecessor encountered in increasing node order. The separate heatmap dynamic program uses $1 0 ^ { - 1 3 }$ for this comparison. Frontier pruning sorts by increasing resource and decreasing value, and selection among tied feasible frontier values favors smaller resource.

Safety-budget grids. For each of the four frontier panels, let $\boldsymbol { \epsilon } _ { \mathrm { f i r s t } }$ be the smallest rounded resource among the proposed method’s retained nondominated labels that contain an actual update. This is a landmark of the retained frontier. Across panels, the common axis endpoints are

$$
\epsilon _ { \mathrm { m i n } } = \operatorname* { m a x } \left\{ 0 . 9 2 \operatorname* { m i n } _ { s } \epsilon _ { \mathrm { f i r s t } , s } , \Delta / 1 0 \right\} , \qquad \epsilon _ { \mathrm { m a x } } = \operatorname* { m a x } \left\{ 1 . 0 8 \operatorname* { m a x } _ { s } \epsilon _ { \mathrm { A U } , s } , 1 . 0 1 \epsilon _ { \mathrm { m i n } } \right\} ,\tag{D.15}
$$

where infinite first-update landmarks are omitted. For the saved outputs, the first-update landmarks for low SNR, medium SNR, high SNR, and IST are 0.203, 0.030, 0, and 0.432, respectively; their always-update costs are 27.536, 27.559, 24.928, and 22.492. These give the common range $[ 1 0 ^ { - 4 } , 2 9 . 7 6 3 7 2 ]$ . The frontier plot uses 241 budget points, and the schedule diagram uses 90 points over the same range. Both are equally spaced after a base-10 symmetric-log transformation with linear threshold $\epsilon _ { \mathrm { { m i n } } } ;$ since all displayed budgets are at or above that threshold, they are logarithmically spaced on the positive range. The schedule-diagram subsampling factor is one, so all 90 points are retained.

Objective normalization and update coordinates. For a schedule $P ,$ let $\begin{array} { r } { \widehat { J } _ { T } ( P ) = \sum _ { ( i , j ) \in P } \widehat { R } _ { i j } } \end{array}$ . The plotted objective is

$$
\widetilde J _ { T } ( P ) = \frac { \widehat J _ { T } ( P ) - \widehat \nu _ { 1 } } { T ^ { - 1 } \sum _ { i = 1 } ^ { T } \widehat \nu _ { i } - \widehat \nu _ { 1 } } ,\tag{D.16}
$$

which maps never updating to zero and updating at every checkpoint to one. The frontier script does not clip this quantity to one: selectively retaining candidates can outperform the always-update comparator under the unprojected empirical means. The schedule diagram plots each selected interior node at its zero-based checkpoint index divided by $T ;$ in the paper’s one-based notation, candidate i is displayed at $( i - 1 ) / T$ . The first candidate is treated as available at the start of the scheduling horizon, after its initial training observations have been collected. The two vertical landmarks in each schedule panel are $\epsilon _ { \mathrm { f i r s t } }$ and $\epsilon _ { \mathrm { A U } }$

Replication and uncertainty. The synthetic frontier panels each use one bank of 400 learning trajectories, and the IST panel uses one bank of 500 bootstrap learning trajectories with the fixed evaluation split above. Schedule selection and frontier reporting use the same estimated matrices; there is no additional independent trajectory bank for evaluating the selected schedules in these plots. The homogeneous experiment difers in that it evaluates selected paths using the population matrices in (D.6). The frontier script defaults to zero outer bootstrap replications and therefore reports curves without uncertainty bands. Its optional outer bootstrap resamples whole columns of a value-trajectory matrix with replacement, re-estimates the matrices, and re-solves every method. When enabled, the displayed band is the original normalized curve plus or minus one sample standard deviation across those bootstrap curves, not a confidence interval; the outer-bootstrap random-number seed is 20260923.