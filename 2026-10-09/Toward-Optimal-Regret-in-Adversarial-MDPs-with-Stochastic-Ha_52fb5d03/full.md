# Toward Optimal Regret in Adversarial MDPs with Stochastic Hard Constraints

Qian Zuo

University of Edinburgh

qian.zuo@ed.ac.uk

Francesco Emanuele Stradi

Politecnico di Milano

francescoemanuele.stradi@polimi.it

## Abstract

We study episodic constrained Markov decision processes with adversarial losses under stochastic hard constraints. Specifically, starting from a known strictly feasible policy with margin d, we seek to obtain optimal regret while satisfying the expected cost constraints in every episode. In this setting, Stradi et al. (2025a) show that a carefully designed mixing rule attains regret of order $\widetilde { \mathcal { O } } ( \sqrt { T } / \operatorname* { m i n } \{ d , d ^ { 2 } \} )$ Interestingly, they also provide a lower bound of order Ω( T/ρ) for the same setting, where $\rho$ is the Slater margin of the ofline problem and can be much larger than d. In this work, we build on their approach to obtain optimal regret dependence on these margins. Specifically, we propose MA-OPS, an algorithm that combines an optimistic search for the Slater margin with a pessimistic evaluation of the selected policies to safely learn a policy with a large feasibility margin. This policy is then used to minimize regret while satisfying the constraints at every episode. In particular, we show that MA-OPS attains regret $\widetilde { \mathcal { O } } ( \sqrt { T } / \rho + 1 / ( d \rho ) )$ . Finally, we provide a matching lower bound, showing that the dependence on T, d, ρ in the regret bound is optimal up to logarithmic factors.

## 1 Introduction

Reinforcement learning has been successfully applied to many challenging problems, including robotic control, game playing, and aligning language models with human preferences (Kaufmann et al., 2023; Silver et al., 2016; Ouyang et al., 2022). These problems are commonly modeled through Markov decision processes (MDPs), in which a learner sequentially interacts with an environment by selecting actions, observing state transitions, and incurring losses (Sutton and Barto, 2018). In this paper, we consider episodic adversarial MDPs, where the transition kernel is fixed, while losses can change arbitrarily across episodes (Jin et al., 2020). Thus, no statistical assumption on the losses is required, allowing the model to capture learning problems with changing objectives.

In many applications, the learner must also satisfy requirements on quantities such as resource consumption. Constrained MDPs (CMDPs, for short) (Altman, 1999) model these requirements by introducing costs whose expected cumulative values must remain below given thresholds. When such requirements must hold throughout the learning process, including during exploration, one needs algorithms that satisfy hard constraints. Specifically, all constraints must be satisfied in every episode on a common high-probability event (Liu et al., 2021; Stradi et al., 2025a). We study this problem in CMDPs with adversarial losses, unknown transitions, stochastic constraints, and bandit feedback.

A standard approach to ensure safety during exploration assumes that the learner is given a strictly feasible baseline policy together with upper bounds on its expected cumulative costs (Yu et al., 2025). The learner can then mix this baseline with the algorithmically selected policies whose costs are not well estimated. The amount of exploration allowed by such an approach depends on the input policy safety margin d, namely, the smallest margin by which the input policy satisfies all the constraints. When d is small, ensuring safety may require selecting the baseline for many episodes, even when this policy incurs large losses.

However, the margin given as input to the learner may be much smaller than the largest one attainable in the CMDP. We denote the latter by the Slater margin $\rho ,$ defined as the largest common slack with which any policy can satisfy all constraints, such that it holds that $d \leq \rho .$ For $d \leq 1$ , Stradi et al. (2025a) provide the Safe Optimistic Policy Search (S-OPS) algorithm, which attains $\widetilde { \mathcal { O } } ( \sqrt { T } / d ^ { 2 } )$ regret over T episodes, while establishing an $\Omega ( \sqrt { T } / \rho )$ lower bound for the problem. Thus, these results leave open the question of the optimal joint dependence on d and ρ when constraints must be satisfied at every episode. Specifically, we focus on the following research question.

Table 1: Comparison of regret and cumulative positive constraint violation bounds under stochastic constraints and bandit feedback. The first two rows concern constrained multi-armed bandits, and the remaining rows concern episodic CMDPs with unknown transitions. Here, d denotes the input policy safety margin, ρ the Slater margin $( \rho \ge d )$ , and T the number of episodes (rounds in bandits). We show only the dependence on T, d, and $\rho ,$ omitting logarithmic factors.
<table><tr><td>Algorithm</td><td>Losses Input margin</td><td></td><td>Regret</td><td>Violation</td><td>Lower bound</td></tr><tr><td>OPB (Pacchiano et al., 2021)</td><td>Stoch.</td><td>d</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / d )$ </td><td>0</td><td> $\Omega ( \operatorname* { m a x } \{ \sqrt { T } , 1 / d ^ { 2 } \} )$ </td></tr><tr><td>SOLB (Genalti et al., 2025)</td><td>Adv.</td><td>ρ</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / \rho )$ </td><td>0</td><td> $\Omega ( \sqrt { T } / \rho )$ </td></tr><tr><td>OptPess-LP (Liu et al., 2021)</td><td>Stoch.</td><td>d</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / d + 1 / \operatorname* { m i n } \{ d , d ^ { 2 } \} )$ </td><td>0</td><td></td></tr><tr><td>DOPE+ (Yu et al., 2025)</td><td>Stoch.</td><td>d</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / d + 1 / d ^ { 2 } )$ </td><td>0</td><td></td></tr><tr><td>S-OPS (Stradi et al., 2025a)</td><td>Adv.</td><td>d</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / \operatorname* { m i n } \{ d , d ^ { 2 } \} )$ </td><td>0</td><td> $\Omega ( \sqrt { T } / \rho )$ </td></tr><tr><td>CV-OPS (Stradi et al., 2025a)</td><td>Adv.</td><td>一</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / \operatorname* { m i n } \{ \rho , \rho ^ { 2 } \} + 1 / \rho ^ { 4 } )$ </td><td> $\widetilde { \mathcal { O } } ( 1 / \rho ^ { 4 } )$ </td><td> $\Omega ( \sqrt { T } / \rho )$ </td></tr><tr><td>S-OPS (Theorem 1)</td><td>Adv.</td><td>d</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / d )$ </td><td>0</td><td></td></tr><tr><td>MA-OPS (Theorems 2, 4)</td><td>Adv.</td><td>d</td><td> $\widetilde { \mathcal { O } } ( \sqrt { T } / \rho + 1 / ( d \rho ) )$ </td><td>0</td><td> $\Omega ( \sqrt { T } / \rho + 1 / ( d \rho ) )$ </td></tr></table>

Is it possible to design an algorithm that attains optimal regret dependence on the initial input margin d and the unknown Slater parameter $\rho ,$ while satisfying the constraints at every episode?

We answer this question afirmatively by introducing Margin-Adaptive Optimistic Policy Search (MA-OPS). The algorithm first searches for a policy with a large safety margin while ensuring safety throughout this phase, and then uses this policy as the baseline for loss minimization. This allows us to attain a regret bound of order $\widetilde { \mathcal { O } } ( \sqrt { T } / \rho + 1 / ( d \bar { \rho } ) )$ ), which we then prove to be optimal in $T , \rho ,$ d up to logarithmic factors.

## 1.1 Original Contribution

Our contributions can be summarized as follows.

• Linear $1 / d$ dependence for S-OPS We establish a $\widetilde { \mathcal { O } } ( \sqrt { T } / d )$ regret bound for S-OPS with high probability (Theorem 1). This improves the dependence from $1 / d ^ { 2 }$ to $1 / d$ for $d < 1$ with respect to the bound provided in Stradi et al. (2025a).

• Safe learning of the Slater margin We develop MA-OPS, that is, the first algorithm capable of learning the Slater margin and the associated policy in constant time, while satisfying the constraints at each episode. MA-OPS then uses S-OPS instantiated with the aforementioned policy as subroutine to attain the final regret guarantees.

• Learning guarantees Specifically, MA-OPS learns a margin of at least $\rho / 2$ within $\widetilde { \mathcal { O } } ( 1 / ( d \rho ) )$ episodes, guaranteeing safety throughout (Theorem 3). For every finite T, MA-OPS jointly guarantees $\widetilde { \mathcal { O } } ( \sqrt { T } / \rho +$ $1 / ( d \rho ) )$ regret and feasibility in every episode with high probability (Theorem 2).

• Matching lower bound We prove an $\Omega ( \sqrt { T } / \rho + 1 / ( d \rho ) )$ regret lower bound for any safe algorithm (Theorem 4). This matches the MA-OPS guarantees on $T , d , \rho$ up to logarithmic factors.

A summary of the performance of our algorithm compared to the state-of-the-art is provided in Table 1.

## 1.2 Related Work

Online CMDPs have been widely studied in stochastic, adversarial, and non-stationary settings (Efroni et al., 2020; Qiu et al., 2020; Stradi et al., 2024; Ding and Lavaei, 2023). In stochastic CMDPs, Efroni et al. (2020) obtain $\widetilde { \mathcal { O } } ( \sqrt { T } )$ regret and cumulative positive constraint violation using linear programming. Stradi et al. (2025b) further achieve the same dependence on T through policy optimization. For hard constraints, early work by Liu et al. (2021) established $\widetilde { \mathcal { O } } ( \sqrt { T } )$ regret for stochastic CMDPs with unknown transitions, using a known strictly feasible baseline to ensure feasibility in every episode. Subsequent work improved the dependence on the number of states (Bura et al., 2022) and the episode length (Yu et al., 2025). Across these results, the leading regret term scales linearly with $1 / d ,$ while the bounds of Liu et al. (2021) and Yu et al. (2025) include an additive $1 / d ^ { 2 }$ term when the initial margin is at most one. Stradi et al. (2025a) extended this line to adversarial losses and also studied learning a baseline when no feasible policy is initially known. With a given baseline, they obtain $\widetilde { \mathcal { O } } ( \sqrt { T } / d ^ { 2 } )$ regret for $d \leq 1$ , omitting other problem parameters. Without one, their algorithm learns a margin comparable to $\rho$ but permits violations during the search. We study CMDPs with adversarial losses and stochastic hard constraints, starting from a known strictly feasible baseline. We recover the linear $1 / d$ dependence in the leading regret term and learn a baseline with margin comparable to $\rho ,$ while maintaining feasibility in every episode.

Another closely related topic is constrained bandits. For constraints on individual actions, Amani et al. (2019) obtain sharper regret bounds when the optimal action has a known positive safety margin. This margin can vanish even when a strictly feasible baseline exists. For per-round expected-cost constraints, Pacchiano et al. (2021) and Pacchiano et al. (2025) instead express stochastic regret bounds in terms of a known baseline’s margin, with leading terms linear in $1 / d$ and expected-regret lower bounds containing an inverse-square margin term. For adversarial losses, Genalti et al. (2025) obtain loss-dependent bounds. Their main analysis assumes a known maximum-margin strategy and its expected costs, so the initial margin equals the Slater margin. Our lower bound separates d from $\rho ,$ quantifying the regret cost of limited initial information even when a larger margin is attainable.

Due to space limitations, we defer an additional discussion to Appendix A.

## 2 Preliminaries

We study an episodic constrained Markov decision process (CMDP) $\mathcal { M } = \langle X , A , P , \{ \ell _ { t } \} _ { t = 1 } ^ { T } , \{ G _ { t } \} _ { t = 1 } ^ { T } , \alpha \rangle$ with adversarial losses and stochastic constraint costs. The learner interacts with M for $T$ episodes of length $L ,$ subject to m constraints. Here, X and A are finite state and action spaces. The transition kernel is $P : X \times A \times X \to [ 0 , 1 ]$ , where $P ( x ^ { \prime } \mid x , a )$ denotes the probability of moving from state x to state $x ^ { \prime }$ after taking action a. For episode $t \in [ \dot { T } ] , \dot { \ell _ { t } } \in [ \dot { 0 } , 1 ] ^ { | X \times A | }$ is the loss vector and $G _ { t } \in [ 0 , 1 ] ^ { | X \times A | \times m }$ is the constraint cost matrix. The vector $\alpha \in [ 0 , L ] ^ { m }$ contains the cost thresholds, with $\alpha _ { i }$ denoting the threshold for constraint $i \in [ m ]$ . We write $[ n ] = \{ 1 , \dots , n \}$ for any positive integer n.

At the beginning of each episode $t \in [ T ]$ , an adversary chooses the loss vector $\ell _ { t }$ . The cost matrices $G _ { t }$ are drawn i.i.d. from an unknown distribution G. For constraint $i \in [ m ]$ , we denote the i-th column of $G _ { t }$ by $g _ { t , i } \in [ 0 , 1 ] ^ { | X \times A | }$ . At a state–action pair $( x , a ) \in X \times A$ , the loss is $\ell _ { t } ( x , a )$ and the cost for constraint i is $g _ { t , i } ( x , a )$ . We denote the expected cost vector by $g _ { i } = \mathbb { E } [ g _ { t , i } ] \in [ 0 , 1 ] ^ { | X \times A | }$ and the expected cost matrix by $\boldsymbol G = [ g _ { 1 } , \dots , g _ { m } ] \in [ 0 , 1 ] ^ { | \boldsymbol X \times \hat { \boldsymbol A } | \times m }$ . The learner knows $X , A , L$ and α, whereas P and G are unknown.

We consider loop-free ${ \mathrm { C M D P s } } ,$ in which X is partitioned into $L + 1$ layers $X _ { 0 } , \ldots , X _ { L }$ , with $X _ { 0 } = \{ x _ { 0 } \}$ and $X _ { L } = \{ x _ { L } \}$ . Transitions occur only between consecutive layers, i.e., $P ( x ^ { \prime } \mid x , a ) > 0$ only if $x \in X _ { k }$ and $x ^ { \prime } \in X _ { k + 1 }$ for some $k \in \{ 0 , \ldots , L - 1 \}$ }. A finite-horizon CMDP can be represented in this form by duplicating its states across time steps.

A Markov policy $\pi : X \times A \to [ 0 , 1 ]$ defines a probability distribution over actions at each state, with $\pi ( a \mid x )$ denoting the probability of choosing action a in state x. Starting from $x _ { 0 } .$ the learner samples $a _ { k } \sim \pi ( \cdot \mid x _ { k } )$ and the next state is drawn as $x _ { k + 1 } \sim P ( \cdot \mid x _ { k } , a _ { k } )$ for $k \in \{ 0 , \ldots , L - 1 \}$ . At the beginning of episode $t \in [ T ]$ the learner selects a policy $\pi _ { t }$ using observations from previous episodes. Under bandit feedback, the learner observes the trajectory of visited state–action pairs $( x _ { k } , a _ { k } )$ for $k \in \{ 0 , \ldots , L - 1 \}$ , their losses $\ell _ { t } ( x _ { k } , a _ { k } )$ , and their costs $g _ { t , i } ( x _ { k } , a _ { k } )$ for constraint $i \in [ m ]$

Given P and a policy π, we denote the induced occupancy measure by $q ^ { P , \pi } \in [ 0 , 1 ] ^ { | X \times A \times X | }$ . For $x \in X _ { k }$ $a \in A$ , and $x ^ { \prime } \in X _ { k + 1 }$ with $k \in \{ 0 , \ldots , L - 1 \}$ , its entry gives the probability of visiting $( x , a , x ^ { \prime } )$ in one episode, i.e., $q ^ { P , \pi } ( x , a , x ^ { \prime } ) = \mathbb { P } [ x _ { k } = x , a _ { k } = a , x _ { k + 1 } = x ^ { \prime } \mid P , \pi ]$ . For an occupancy measure $q ,$ the state– action marginal vector in $\mathrm { [ 0 , 1 ] } ^ { | X \times A | }$ has entries $\begin{array} { r } { q ( x , a ) = \sum _ { x ^ { \prime } } q ( x , a , x ^ { \prime } ) } \end{array}$ . Summing over actions gives the state marginal vector in $[ 0 , 1 ] ^ { | X | }$ , with entries $\begin{array} { r } { q ( x ) = \sum _ { a } q ( x , a ) } \end{array}$ . Since each episode has L steps, $\begin{array} { r } { \sum _ { x , a } q ( x , a ) = L } \end{array}$ We denote the set of occupancy measures under $P$ by $\Delta ( P ) \subseteq [ 0 , 1 ] ^ { | X \times A \times X | }$ . Each $q \in \Delta ( P )$ induces a Markov policy $\pi ^ { q }$ , with $\pi ^ { q } ( a \mid x ) = q ( x , a ) / q ( x )$ at states where $q ( x ) > 0$

The expected cumulative loss of policy π in episode $t \in [ T ]$ is $\begin{array} { r } { \ell _ { t } ^ { \top } q ^ { P , \pi } = \sum _ { x , a } \ell _ { t } ( x , a ) q ^ { P , \pi } ( x , a ) } \end{array}$ . Similarly, its expected cumulative cost for constraint $i \in [ m ]$ is $\begin{array} { r } { \mathbb { E } _ { P , \pi } \Big [ \sum _ { k = 0 } ^ { L - 1 } g _ { i } \big ( \boldsymbol { x } _ { k } , \boldsymbol { a } _ { k } \big ) \Big ] = g _ { i } ^ { \top } \boldsymbol { q } ^ { P , \pi } } \end{array}$ . A policy is feasible if $g _ { i } ^ { \top } q ^ { P , \pi } \leq \alpha _ { i }$ for every constraint $i \in [ m ]$ . For the policy $\pi _ { t }$ selected in episode $t ,$ we write $q _ { t } = q ^ { P , \pi _ { t } }$

## 2.1 Online CMDPs with Hard Constraints

We measure the learner’s performance in terms of regret against the best feasible policy in hindsight. Specifically, let $q ^ { \star } \in \Delta ( P )$ be a minimizer of $\Sigma _ { t = 1 } ^ { T } \ell _ { t } ^ { \top } q$ subject to $G ^ { \top } q \leq \alpha$ . We define the cumulative regret as:

$$
R _ { T } = \sum _ { t = 1 } ^ { T } \ell _ { t } ^ { \top } ( q _ { t } - q ^ { \star } ) .\tag{1}
$$

Our goal is to achieve a regret bound that is sublinear in $T , \mathrm { i . e . , } R _ { T } = o ( T )$ , while ensuring that $g _ { i } ^ { \top } q _ { t } \leq \alpha _ { i }$ for every episode $t \in [ T ]$ and every constraint $i \in [ m ]$ with high probability. To satisfy the constraints from the first episode, we assume that the learner knows a potentially suboptimal strictly feasible policy (Liu et al., 2021; Bura et al., 2022; Stradi et al., 2025a).

Assumption 1 (Strictly feasible policy). The learner knows a Markov policy $\pi ^ { \diamond }$ and a vector $\beta \in [ 0 , L ] ^ { m }$ satisfying $G ^ { \top } q ^ { P , \pi ^ { \diamond } } \leq \beta < \alpha$

The upper bounds $\beta$ on the expected cumulative costs of $\pi ^ { \diamond }$ determine the initial margin $d .$ In contrast, the Slater margin $\rho$ is the largest slack that a policy can achieve across the constraints. Formally,

$$
d = \operatorname* { m i n } _ { i \in [ m ] } ( \alpha _ { i } - \beta _ { i } ) , \quad \rho = \operatorname* { m a x } _ { q \in \Delta ( P ) } \operatorname* { m i n } _ { i \in [ m ] } ( \alpha _ { i } - g _ { i } ^ { \top } q ) .
$$

The initial margin d is known to the learner, whereas the Slater margin $\rho$ is unknown. Assumption 1 implies Slater’s condition (Boyd and Vandenberghe, 2004) and $0 < d \le \rho \le L$ . We use O and Ω to suppress universal constants, and $\widetilde { \mathcal { O } }$ to additionally suppress logarithmic factors.

## 3 Improving The Margin Dependence of S-OPS

When the initial margin d is small, safe exploration can require frequent use of the baseline, limiting how often the policy selected for exploration is executed. Our key observation is that this execution probability also weights the policy’s constraint violation bound in the regret analysis. This weight links the regret incurred by using the baseline to uncertainty along observed trajectories. Using this relation, we show that S-OPS (Stradi et al., 2025a) achieves a regret bound with linear dependence on $1 / d$ while satisfying the constraints in every episode.

## 3.1 Safe Optimistic Policy Search (S-OPS)

S-OPS (Stradi et al., 2025a) searches for low-loss policies under optimistic constraints and mixes them with the baseline $\pi ^ { \diamond }$ using pessimistic cost bounds. In episode $t \in [ T ] , { \mathrm { S } } – { \mathrm { O P S } }$ maintains an occupancy measure $\widehat { q } _ { t } \in \Delta ( \mathcal P _ { t - 1 } )$ using online mirror descent, where $\Delta ( \mathcal { P } _ { t - 1 } )$ is the set of occupancy measures induced by kernels in the transition confidence set $\mathcal { P } _ { t - 1 }$ . We denote the empirical cost vectors and their confidence widths by $\widehat { g } _ { t - 1 , i } , \xi _ { t - 1 } \in [ 0 , 1 ] ^ { | X \times A | }$ , respectively. The mirror descent update imposes $( \widehat { g } _ { t - 1 , i } - \xi _ { t - 1 } ) ^ { \top } \widehat { q } _ { t } \leq \alpha _ { i }$ for each constraint $i \in [ m ]$ . On the confidence event, the resulting optimistic set contains the true occupancy of every feasible policy. However, the induced policy $\widehat { \pi } _ { t } = \pi ^ { \widehat { q _ { t } } }$ may violate the true constraints.

To choose a mixing probability that ensures safety, S-OPS first computes upper bounds on the expected costs of $\widehat { \pi } _ { t }$ . To account for transition uncertainty, it uses $\widehat { u } _ { t } \in [ 0 , 1 ] ^ { | X \times A | }$ , where $\widehat { u } _ { t } ( x , a ) = \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { t - 1 } } q ^ { P ^ { \prime } , \widehat { \pi } _ { t } } ( x , a )$ $\mathrm { S - O P S }$ then combines these occupancy bounds with the pessimistic cost estimates $\widehat { g } _ { t - 1 , i } + \xi _ { t - 1 }$ to obtain $w _ { t } \in [ 0 , L ] ^ { m }$ , where $w _ { t , i } = \operatorname* { m i n } \{ L , ( \widehat { g } _ { t - 1 , i } + \xi _ { t - 1 } ) ^ { \top } \widehat { u } _ { t } \}$ . At the beginning of each episode, the randomized policy $\pi _ { t }$ selects the baseline $\pi ^ { \diamond }$ with probability:

$$
\lambda _ { t - 1 } = \operatorname* { m a x } _ { i : w _ { t , i } > \alpha _ { i } } \frac { w _ { t , i } - \alpha _ { i } } { w _ { t , i } - \beta _ { i } } ,\tag{2}
$$

and $\widehat { \pi } _ { t }$ with probability $1 - \lambda _ { t - 1 }$ , then follows the selected policy for all L steps. We set $\lambda _ { t - 1 } = 0$ when $w _ { t } \leq \alpha$ This randomization gives $q ^ { P ^ { \prime } , \pi _ { t } } = \lambda _ { t - 1 } q ^ { P ^ { \prime } , \pi ^ { \diamond } } + ( 1 - \lambda _ { t - 1 } ) q ^ { P ^ { \prime } , \widehat { \pi } _ { t } }$ for every kernel $P ^ { \prime }$ . With high probability, the cost bounds and the mixing rule therefore ensure $G ^ { \top } q _ { t } \leq \lambda _ { t - 1 } \beta + ( 1 - \lambda _ { t - 1 } ) w _ { t } \leq \alpha$ . Moreover, the probability of executing $\widehat { \pi } _ { t }$ satisfies $1 - \lambda _ { t - 1 } > 0$ , allowing $\mathrm { S - O P S }$ to collect observations under this policy before it is known to be feasible.

After executing $\pi _ { t } , { \mathrm { S } } { \mathrm { - O P S } }$ uses the observed losses to construct the optimistic loss estimator $\widehat { \ell _ { t } } \in [ 0 , 1 / \gamma ] ^ { | X \times A | }$ ， with implicit exploration parameter $\gamma > 0$

$$
\widehat { \ell } _ { t } ( x , a ) = \frac { \ell _ { t } ( x , a ) I _ { t } ( x , a ) } { u _ { t } ( x , a ) + \gamma } ,\tag{3}
$$

where $\begin{array} { r } { u _ { t } ( x , a ) = \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { t - 1 } } q ^ { P ^ { \prime } , { \pi } _ { t } } ( x , a ) } \end{array}$ is the largest probability of visiting $( x , a )$ under $\pi _ { t } ,$ and $I _ { t } ( x , a )$ is the indicator function which is 1 when $( x , a )$ has been visited at episode $t . ~ \mathrm { S – O P S }$ also updates its cost and transition estimates after every episode. A multiplicative update with $\widehat { \ell } _ { t }$ and learning rate $\eta > 0$ , followed by a relative-entropy projection onto the updated optimistic set, gives $\widehat { q } _ { t + 1 }$ . We provide the pseudocode of S-OPS in Algorithm 2 (Appendix C.1).

## 3.2 Regret Guarantee

We then obtain the following regret guarantee.

Theorem 1 (Regret and constraint violation). Suppose Assumption 1 holds. Let $\delta \in ( 0 , 1 )$ . S-OPS with $\begin{array} { r } { \eta = \gamma = \sqrt { \frac { L \ln ( 8 m | X | ^ { 2 } | A | T / \delta ) } { T | X | | A | } } } \end{array}$ guarantees, with probability at least $1 - \delta , G ^ { \top } q _ { t } \leq \alpha$ for every $t \in [ T ]$ and:

$$
R _ { T } \leq \widetilde { \mathcal { O } } \left( \frac { L ^ { 2 } | X | \sqrt { | A | T } } { d } + \frac { L | X | ^ { 3 } | A | } { d } \right) .
$$

Theorem 1 guarantees sublinear regret with linear dependence on the inverse baseline margin $1 / d ,$ while satisfying every constraint in any episode. The original S-OPS analysis assumes that the baseline policy ${ \vec { \mathbf { \nabla } } } \mathrm { s }$ expected costs are known exactly and gives regret $\widetilde { \mathcal { O } } ( L ^ { 3 } | X | \sqrt { | A | T } /$ min $\{ d , d ^ { 2 } \} )$ (Stradi et al., 2025a). Our analysis improves the dependence on $1 / d$ from quadratic to linear when $d < 1$ , and reduces the dependence on $L$ in the leading term. In stochastic CMDPs, the DOPE+ regret bound has a leading term linear in $1 / d$ and an additive term quadratic in this inverse margin (Yu et al., 2025). Theorem 1 shows that both regret terms depend only linearly on $1 / d$ for adversarial losses.

Remark 1 (Margin normalization). To compare our bounds with MAB ones where the costs are in $[ 0 , 1 ]$ , we divide cumulative episode costs and constraint thresholds by L. The resulting margins are $d / L , \rho / L \in ( 0 , 1 ]$ Notice that the inverse normalized Slater margin satisfies $L / \rho \ge 1$ , even when $\rho > 1$ , and, thus, $( L / \rho ) ^ { 2 } \ge L / \rho$ Therefore, comparing linear and quadratic dependence on the inverse Slater margin requires accounting for the dependence on the horizon L as well.

## 3.3 Proof Sketch

In the regret decomposition of Stradi et al. (2025a), the regret incurred by using the baseline is at most $L \sum _ { t = 1 } ^ { T } \lambda _ { t - 1 }$ . Their bound is quadratic in $1 / d$ when $d < 1$ . We improve this bound by incorporating the execution probability of $\widehat { \pi } _ { t }$ into the analysis.

For $t \in \{ 2 , \ldots , T \}$ , let $e _ { t }$ be the largest estimated constraint violation, i.e., $e _ { t } = \mathrm { m a x } _ { i \in [ m ] } \{ w _ { t , i } - \alpha _ { i } , 0 \}$ . If $\lambda _ { t - 1 } > 0$ , let $i _ { t }$ attain the maximum in Equation (2). We rearrange the mixing rule to obtain:

$$
\lambda _ { t - 1 } ( \alpha _ { i _ { t } } - \beta _ { i _ { t } } ) = ( 1 - \lambda _ { t - 1 } ) ( w _ { t , i _ { t } } - \alpha _ { i _ { t } } ) .\tag{4}
$$

Since $\alpha _ { i _ { t } } - \beta _ { i _ { t } } \geq d .$ , we obtain $\lambda _ { t - 1 } \leq ( 1 - \lambda _ { t - 1 } ) e _ { t } / d .$ It therefore sufices to bound $\textstyle \sum _ { t = 2 } ^ { T } ( 1 - \lambda _ { t - 1 } ) e _ { t }$

Let ${ \mathcal { E } } _ { T }$ be the event on which all cost and transition confidence bounds hold. On this event, the optimistic constraints bound $e _ { t }$ by cost and transition estimation errors. For the cost error, we have $( 1 - \lambda _ { t - 1 } ) \xi _ { t - 1 } ^ { \top } q ^ { P , \widehat { \pi } _ { t } } \leq$ $\xi _ { t - 1 } ^ { \top } q _ { t }$ . The right-hand side is the conditional expected sum of confidence widths along the observed trajectory. For the transition error, we compare $\widehat { q _ { t } }$ and $\widehat { u } _ { t }$ with $q ^ { P , \widehat { \pi } _ { t } }$ and bound their deviations by occupancy-weighted sums of confidence widths and their squares. These sums control the leading and lower-order terms, respectively. Weighting the bounds for $\widehat { \pi } _ { t }$ and $\pi ^ { \diamond }$ by their execution probabilities also controls $\| u _ { t } - q _ { t } \| _ { 1 }$ . Combining these estimates with the cost-error bound gives the following lemma.

Lemma 3.1. Under the conditions of Theorem 1, with probability at least $1 - \delta / 2$ , the event ${ \mathcal { E } } _ { T }$ holds and $\begin{array} { r } { \sum _ { t = 2 } ^ { T } ( 1 - \lambda _ { t - 1 } ) e _ { t } + \sum _ { t = 1 } ^ { T } \| u _ { t } - q _ { t } \| _ { 1 } \leq \widetilde { \mathcal { O } } \Big ( L | X | \sqrt { | A | T } + | X | ^ { 3 } | A | \Big ) } \end{array}$

Lemma 3.1 controls both the weighted constraint-violation bounds and the cumulative upper-occupancy error. Equation (4) then bounds regret from baseline use by $\begin{array} { r } { L + ( L / d ) \sum _ { t = 2 } ^ { T } ( 1 - \lambda _ { t - 1 } ) e _ { t } , } \end{array}$ , where $L$ accounts for the initial episode. Retaining the execution weights gives linear dependence on $1 / d$ . The earlier analysis bounds unweighted errors using $1 - \lambda _ { t - 1 } \geq d / L$ , introducing an additional factor $L / d$ when $d < 1$

We next control the mirror descent and loss-estimation terms. The estimated occupancy measure $\widehat { q _ { t } }$ satisfies $( 1 - \lambda _ { t - 1 } ) \widehat { q } _ { t } = q ^ { P ^ { \prime } , \pi _ { t } } - \lambda _ { t - 1 } q ^ { P ^ { \prime } , \pi ^ { \diamond } } \leq u _ { t }$ for some $P ^ { \prime } \in \mathcal { P } _ { t - 1 }$ . Together with $1 - \lambda _ { t - 1 } \geq d / L$ , this implies $\widehat { q } _ { t } / ( u _ { t } + \gamma ) \leq L / d .$ For the quadratic term in the mirror descent bound, we use $\widehat { q } _ { t } ( x , a ) \widehat { \ell } _ { t } ( x , a ) ^ { 2 } \leq ( L / d ) \widehat { \ell } _ { t } ( x , a )$ Implicit exploration bounds $\textstyle \sum _ { t , x , a } { \widehat { \ell } } _ { t } ( x , a )$ without further dependence on d (Jin et al., 2020). The ratio bound also controls the cumulative conditional bias by $\begin{array} { r } { ( L / d ) ( \sum _ { t } \| u _ { t } - q _ { t } \| _ { 1 } + \gamma | X | | A | T ) } \end{array}$ , where Lemma 3.1 bounds the occupancy-error sum. Combining these bounds with standard concentration bounds for the remaining loss-estimation terms yields Theorem 1. The full proofs are deferred to Appendix C.

## 4 MA-OPS: How to Safely Learn the Slater Margin

In this section, we introduce Margin-Adaptive Optimistic Policy Search (MA-OPS), a meta-algorithm that safely learns a baseline with margin comparable to the unknown Slater margin $\rho .$ By Theorem 1, such a baseline makes subsequent regret scale with $1 / \rho .$ The main challenge is that $\rho$ is unknown, while policies considered during the search may violate the constraints. MA-OPS addresses this challenge by combining an optimistic margin search with pessimistic policy evaluation. Optimistic search selects a policy and estimates the largest attainable margin $\rho .$ Pessimistic evaluation accounts for cost and transition uncertainty to obtain a lower bound on the selected policy’s margin. The search ends once either the initial margin d or the selected policy’s pessimistic margin estimate is at least half the optimistic estimate of $\rho .$

We present MA-OPS in Algorithm 1.

Optimistic margin search We first notice that an underestimate of $\rho$ can cause the search to end with a baseline whose margin is smaller than $\rho / 2$ . MA-OPS therefore estimates $\rho$ optimistically while searching for a large-margin policy. Let t denote the number of completed episodes and $n - 1$ the number of exploration trajectories collected. For each constraint $i \in [ m ]$ , MA-OPS computes the empirical cost vector $\widehat { g } _ { n - 1 , i } \in [ 0 , \mathsf { 1 } ] ^ { | X \times A | }$ from these trajectories. The estimates are combined with cost confidence widths $\xi _ { n - 1 } \in [ 0 , 1 ] ^ { | X \times A | }$ to form the optimistic cost estimates $\widehat { g } _ { n - 1 , i } - \xi _ { n - 1 }$ . MA-OPS then computes $\widehat { q } _ { n }$ by solving the optimistic max–min problem:

Algorithm 1 MA-OPS   
Input: $X , A , L , \alpha , T , \delta$ and $( \pi ^ { \circ } , \beta )$ satisfying Assumption 1   
1: $d  \mathrm { m i n } _ { i \in [ m ] } ( \alpha _ { i } - \beta _ { i } ) , \delta _ { 0 }  \delta / 2 , t  0 , n  1$ . Initialize all visit counts to zero   
2: $( { \widehat { \pi } } ^ { \circ } , { \widehat { \beta } } ) \gets ( \pi ^ { \circ } , \beta )$   
3: Set πb1 to the uniform policy and $w _ { 1 , i }  L$ for $i \in [ m ]$   
4: $\lambda _ { 1 } \gets \operatorname* { m a x } _ { i \in [ m ] } ( L - \alpha _ { i } ) / ( L - \beta _ { i } )$   
5: while $t < T$ and $2 d < L$ do   
6: $t \gets t + 1$   
7: Draw $\zeta _ { t } \sim$ Bernoull $\operatorname { i } ( 1 - \lambda _ { n } )$   
$\int \pi ^ { \diamond } , \quad \zeta _ { t } = 0 ,$   
8: Execute   
$\left\backslash \widehat { \pi } _ { n } , \quad \zeta _ { t } = 1 \right.$   
9: Observe the trajectory and feedback   
10: if $\zeta _ { t } = 0$ then continue   
11: $n  n + 1$   
12: Update $\widehat { G } _ { n - 1 } , \Xi _ { n - 1 } , \mathcal { P } _ { n - 1 }$   
13: Choose $\widehat { q _ { n } }$ by (5) ▷ Optimistic estimate of $\rho$   
14: $\begin{array} { r } { \overline { { \rho } } _ { n } \gets \operatorname* { m i n } \{ L , \operatorname* { m i n } _ { i \in [ m ] } \big [ \alpha _ { i } - \big ( \widehat { g } _ { n - 1 , i } - \xi _ { n - 1 } \big ) ^ { \top } \widehat { q } _ { n } \big ] \} } \end{array}$   
15: if $2 d \geq \overline { { \rho } } _ { n }$ then break   
16: $\widehat { \pi } _ { n } \gets \pi ^ { q _ { n } }$   
17: Compute $w _ { n }$ by (6) and set $\underline { { \rho } } _ { n } \gets \operatorname* { m i n } _ { i \in [ m ] } ( \alpha _ { i } - w _ { n , i } )$ ▷ Pessimistic policy evaluation   
18: if $2 \underline { { \rho } } _ { n } \geq \overline { { \rho } } _ { n }$ then ▷ Adaptive stopping rule   
19: $( { \widehat { \pi } } ^ { \circ } , { \widehat { \beta } } ) \gets ( { \widehat { \pi } } _ { n } , w _ { n } )$   
20: break   
21: end if   
22: $\lambda _ { n }  \operatorname* { m a x } ( { w } _ { n , i } - \alpha _ { i } ) / ( { w } _ { n , i } - \beta _ { i } )$ ▷ Safe exploration   
i∈[m]: w<sub>n,i</sub>>α<sub>i</sub>   
23: end while   
24: if $t < T$ then   
25: Apply S-OPS with $( \widehat { \pi } ^ { \circ } , \widehat { \beta } )$ , horizon $T - t ,$ confidence $\delta _ { 0 } ,$ and parameters from Theorem 1   
26: end if

$$
\operatorname { a r g m a x } _ { q \in \Delta ( \mathcal { P } _ { n - 1 } ) } \operatorname* { m i n } \Bigl \{ L , \operatorname* { m i n } _ { i \in [ m ] } \bigl [ \alpha _ { i } - ( \widehat { g } _ { n - 1 , i } - \xi _ { n - 1 } ) ^ { \top } q \bigr ] \Bigr \} ,\tag{5}
$$

where $\mathcal { P } _ { n - 1 }$ is the transition confidence set constructed from the same trajectories. Let $\overline { { \rho } } _ { n }$ be the optimal value and ${ \widehat { \pi } } _ { n } = \pi ^ { { \widehat { q } } _ { n } }$ the induced policy. With high probability, $P \in \mathcal P _ { n - 1 }$ and $| \widehat { g } _ { n - 1 , i } - g _ { i } | \le \xi _ { n - 1 }$ hold at every episode for each constraint $i \in [ m ]$ . On this event, an occupancy measure attaining $\rho$ belongs to $\Delta ( \mathcal { P } _ { n - 1 } )$ , and replacing costs by their lower bounds cannot decrease its constraint slack. Since $\rho \leq L$ , taking the minimum with L preserves $\overline { { \rho } } _ { n } \geq \rho .$ . The equivalent linear program is given in Appendix B.1.

Policy evaluation and stopping Optimistic search uses cost lower bounds and a transition kernel that may difer from P, so it can overestimate the margin of $\widehat { \pi } _ { n }$ . MA-OPS therefore maximizes each expected cos over $\mathcal { P } _ { n - 1 }$ using the pessimistic cost estimates $\widehat { g } _ { n - 1 , i } + \xi _ { n - 1 }$ . This gives $w _ { n } \in [ 0 , L ] ^ { m }$ , where:

$$
w _ { n , i } = \operatorname* { m i n } \biggl \{ L , \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { n - 1 } } ( \widehat { g } _ { n - 1 , i } + \xi _ { n - 1 } ) ^ { \top } q ^ { P ^ { \prime } , \widehat { \pi } _ { n } } \biggr \}\tag{6}
$$

for each constraint $i \in [ m ]$ . Each maximization evaluates the entire expected cost under a single transition kernel. The gap between the margin estimates is then controlled by cost and transition estimation error.

A strictly feasible policy can still have a margin much smaller than ρ. MA-OPS thus compares both the initial margin d and the lower bound $\begin{array} { r } { \underline { { \rho } } _ { n } : = \operatorname* { m i n } _ { i \in [ m ] } ( \alpha _ { i } - w _ { n , i } ) } \end{array}$ with $\overline { { \rho } } _ { n } / 2$ . We denote the occupancy measure

of $\widehat { \pi } _ { n }$ under $P$ by $q _ { n } : = q ^ { P , \widehat { \pi } _ { n } } \in \Delta ( P )$ . On the confidence event, it holds that $G ^ { \top } q _ { n } \leq w _ { n }$ , and hence:

$$
\underline { { \rho } } _ { n } \leq \operatorname* { m i n } _ { i \in [ m ] } ( \alpha _ { i } - g _ { i } ^ { \top } q _ { n } ) \leq \rho \leq \overline { { \rho } } _ { n } .\tag{7}
$$

The search ends with the initial baseline when $2 d \geq \overline { { \rho } } _ { n }$ . Otherwise, it ends with $\widehat { \pi } _ { n }$ when $2 \underline { { \rho } } _ { n } \geq \overline { { \rho } } _ { n }$ . In either case, the selected baseline has a margin of at least $\rho / 2$ . Testing the initial baseline first avoids unnecessary policy evaluation and ensures that any replacement improves the known margin, since $d < \overline { { \rho } } _ { n } / 2 \leq \underline { { \rho } } _ { n }$ . Since $\rho \leq L$ , the search can be omitted when $2 d \geq L$

Safe exploration The cost upper bounds $w _ { n }$ also determine how often $\widehat { \pi } _ { n }$ can be executed while satisfying the constraints. We compute the baseline selection probability $\lambda _ { n } \in [ 0 , 1 )$ using the S-OPS mixing rule (2) with $w _ { n }$ (Line 22). We set $\lambda _ { n } = 0$ when $w _ { n , i } \leq \alpha _ { i }$ for every constraint $i \in [ m ]$ . In each episode, we select $\pi ^ { \diamond }$ with probability $\lambda _ { n }$ and $\widehat { \pi } _ { n }$ with probability $1 - \lambda _ { n }$ . We then execute the selected policy for all L steps and collect the resulting trajectory and cost observations. With high probability, the randomized policy satisfies $\begin{array} { r } { G ^ { \top } q _ { t } \leq \lambda _ { n } \beta + ( 1 - \lambda _ { n } ) w _ { n } \leq \alpha } \end{array}$ in every episode of the search, even when $\widehat { \pi } _ { n }$ is infeasible.

After executing $\widehat { \pi } _ { n } \left( \zeta _ { t } = 1 \right)$ , MA-OPS updates the estimates (Line 12) and repeats the optimistic search. During episodes in which $\pi ^ { \diamond }$ is executed $( \zeta _ { t } = 0 )$ , the estimates, $n , { \widehat { \pi } } _ { n } .$ , and $\lambda _ { n }$ remain unchanged. Consequently, the next exploration trajectory is generated by $\widehat { \pi } _ { n }$ under $P ,$ regardless of how many baseline episodes precede it. The empirical cost and confidence-width matrices $\widehat { G } _ { n - 1 } , \breve { \Xi } _ { n - 1 } \in [ 0 , 1 ] ^ { | X \times A | \times \breve { m } }$ have i-th columns $\widehat { g } _ { n - 1 , i }$ and $\xi _ { n - 1 }$ , respectively. The detailed definitions are given in Appendix B.2. Initially, $\widehat { \pi } _ { 1 }$ selects actions uniformly, $w _ { 1 , i } = L$ for every $i \in [ m ]$ , and $\overline { { \rho } } _ { 1 } = L$

For the remaining $T - t$ episodes, we execute S-OPS with the selected baseline $\widehat { \pi } ^ { \circ }$ and its cost upper bounds ${ \widehat { \beta } } \in [ 0 , L ] ^ { m }$ . Both phases use confidence parameter $\delta _ { 0 } = \delta / 2$ . The learning rate η and implicit-exploration parameter γ are chosen as in Theorem 1 for horizon $T - t$ and confidence parameter $\delta _ { 0 }$

## 5 Theoretical Guarantees

In this section, we establish the regret and constraint violation guarantees for MA-OPS. The full proofs are given in Appendices D–E.

## 5.1 Main Result

We first provide the theoretical guarantees of MA-OPS.

Theorem 2 (Regret and constraint violation). Suppose Assumption 1 holds. Let $\delta \in ( 0 , 1 )$ . With probability at least $1 - \delta$ , Algorithm 1 guarantees $G ^ { \top } q _ { t } \leq \alpha$ for every $t \in [ T ]$ and regret:

$$
R _ { T } \leq \widetilde { \mathcal { O } } \left( \frac { L ^ { 2 } | X | \sqrt { | A | T } } { \rho } + \frac { L ^ { 2 } | X | ^ { 3 } | A | } { d \rho } \right) .
$$

Theorem 2 guarantees sublinear regret with a leading term proportional to $1 / \rho$ and zero constraint violation, without requiring prior knowledge of $\rho .$ . The bound separates learning with a large-margin baseline from the cost of searching for one safely. The leading term follows from Theorem 1 with a known margin of at least $\rho / 2$ . The additive term captures the cost of searching safely from the initial margin d and is independent of T, including its logarithmic factors.

Remark 2. Compared with S-OPS in Theorem 1, MA-OPS reduces the leading regret term from $\widetilde { \mathcal { O } } ( L ^ { 2 } | X | \sqrt { | A | T } / \hat { d } )$ to $\widetilde { \mathcal { O } } ( L ^ { 2 } | X | \sqrt { | A | T } / \rho )$ . For stochastic CMDPs, OptPess-LP (Liu et al., 2021) and DOPE+ (Yu et al., 2025) guarantee zero constraint violation given a known strictly feasible baseline. With one constraint and S states before time expansion, DOPE+ attains $\widetilde { \mathcal { O } } ( L ^ { 5 / 2 } S \sqrt { | A | T } / d + L ^ { 5 } S ^ { 3 } | A | / d ^ { 2 } )$ regret. Under $| X | = \Theta ( L S )$ , our bound is $\widetilde { \mathcal { O } } ( L ^ { 3 } S \sqrt { | A | T } / \rho + L ^ { 5 } S ^ { 3 } | A | / ( d \rho ) )$ ) for adversarial losses, with a leading term independent of m up to logarithmic factors. In constrained bandits, SOLB (Genalti et al., 2025)

gives loss-dependent guarantees and worst-case regret $\widetilde { \mathcal { O } } ( | A | \sqrt { T } / \rho )$ for $T \geq 1 2 | A | / \rho$ , assuming a known maximum-margin policy and its expected costs in the main analysis. CV-OPS (Stradi et al., 2025a) starts without a known feasible policy and learns a baseline with margin at least $\rho / 2$ , allowing constraint violation during the search. Starting from a baseline with margin d, MA-OPS identifies such a baseline under unknown transitions within $\widetilde { \mathcal { O } } ( L ^ { 2 } | \dot { X } | ^ { 2 } | A | / ( d \rho ) )$ episodes with zero constraint violation throughout (Theorem 3).

## 5.2 Proof Analysis

In this section, we provide the main intuition on the proofs which lead to Theorem 2.

We first bound the number of episodes required for safe baseline search. The key step is to bound the cumulative gap between the optimistic and pessimistic margins. Together with the stopping conditions in Algorithm 1, this bound controls the number of exploration trajectories. The mixing rule then bounds the number of baseline executions in terms of the same gap. Finally, we combine the resulting search regret bound with Theorem 1 to prove Theorem 2.

Let $\tau$ denote the number of episodes until baseline search terminates and N the number of exploration trajectories collected. We have the following theorem.

Theorem 3 (Safe baseline search). Suppose Assumption 1 holds and let $\delta \in ( 0 , 1 )$ With probability at least $1 - \delta _ { i }$ , the baseline search terminates after at most $\widetilde { \mathcal { O } } ( L ^ { 2 } | X | ^ { 2 } | A | / ( d \rho ) )$ episodes and collects at most $\widetilde { \mathcal { O } } ( L ^ { 2 } | X | ^ { 2 } | A | / \rho ^ { 2 } )$ exploration trajectories. On the same event, $G ^ { \top } q _ { t } \leq$ α holds for every search episode $t \leq \tau$ The returned pair $( \widehat { \pi } ^ { \circ } , \widehat { \beta } )$ satisfies $G ^ { \top } q ^ { P , \widehat { \pi } ^ { \diamond } } \leq \widehat { \beta }$ and has known margin min $_ { i \in [ m ] } ( \alpha _ { i } - \widehat { \beta } _ { i } ) \geq \rho / 2$

Theorem 3 controls exploration trajectories used for estimation and the baseline episodes required for safety. Using only $1 - \lambda _ { n } \geq d / L$ gives $\widetilde { \mathcal { O } } ( L ^ { 3 } | X | ^ { 2 } | A | / ( d \rho ^ { 2 } ) )$ ) episodes. We obtain the sharper bound by controlling both counts through the margin gap.

To control the exploration count, define $\Delta _ { n } : = \overline { { \rho } } _ { n } - \underline { { \rho } } _ { n }$ for each search policy whose cost upper bounds have been computed. On the confidence event, every policy selected for exploration has $\Delta _ { n } > \rho / 2 ,$ by the stopping rule. The following cumulative gap bound therefore limits the number of exploration trajectories.

Lemma 5.1 (Cumulative margin gap). Under the conditions of Theorem 3, with probability at least $1 - \delta / 2$ for every integer $n \in [ N ]$ , it holds that $\begin{array} { r } { \sum _ { j = 1 } ^ { n } \Delta _ { j } \leq \widetilde { \mathcal { O } } ( L | X | \sqrt { | A | n } ) } \end{array}$

Lemma 5.1 bounds the cumulative gap between the margin estimates. To prove Lemma 5.1, we compare the occupancies used for optimistic search and pessimistic cost evaluation with $q _ { n }$ . Both use ${ \widehat { \pi } } _ { n } ,$ so their deviations from $q _ { n }$ are controlled by transition confidence widths weighted by $q _ { n }$ . Adding the cost widths bounds $\Delta _ { n }$ by a conditional expected sum of widths along an exploration trajectory. We then use visit counts and martingale concentration to bound the cumulative gap.

We next bound the baseline executions needed to collect these trajectories. Since $\widehat { \pi } _ { n }$ and $\lambda _ { n }$ remain fixed until the next exploration trajectory, the number of preceding baseline episodes has conditional mean $\lambda _ { n } / ( 1 - \lambda _ { n } )$ Writing $[ z ] _ { + } = \operatorname* { m a x } \{ z , 0 \}$ , the mixing rule gives:

$$
{ \frac { \lambda _ { n } } { 1 - \lambda _ { n } } } \leq { \frac { [ - \underline { { \rho } } _ { n } ] _ { + } } { d } } .\tag{8}
$$

Here $[ - \underline { { \rho } } _ { n } ] _ { + }$ is the largest constraint violation bound computed from $w _ { n }$ . If $\underline { { \rho } } _ { n } \geq 0$ , the policy can be executed without the baseline even when the stopping condition is not yet met. Let $\tau _ { n }$ denote the episode of the n-th exploration trajectory. The next lemma gives a bound on the baseline count $\tau _ { n } - n$

Lemma 5.2 (Baseline executions). Under the conditions of Theorem 3, with probability at least $1 - \delta / 2$ , the bound $\begin{array} { r } { \tau _ { n } - n \leq ( 2 / d ) \sum _ { j = 1 } ^ { n } [ - \underline { { \rho } } _ { j } ] _ { + } + ( 4 L / d ) \log ( 4 / \delta ) } \end{array}$ holds for every integer $n \in [ N ]$

Lemma 5.2 limits the number of baseline episodes needed to collect these exploration trajectories. Its proof combines a supermartingale bound for the waiting times with Equation (8). We have $[ - \underline { { \rho } } _ { n } ] + \leq \Delta _ { n }$ on the confidence event, so the margin gap also controls baseline use.

We prove Theorem 3 on the joint event of the two lemmas, of probability at least $1 - \delta .$ . The stopping rule gives $\textstyle \sum _ { j = 1 } ^ { n } \Delta _ { j } > n \rho / 2$ for every $n \in [ N ]$ . Lemma 5.1 then implies that N is finite and $N \leq \widetilde { \mathcal { O } } ( L ^ { 2 } | X | ^ { 2 } | A | / \rho ^ { 2 } )$ For $N \geq 1$ , Lemma 5.1 and the bound on N give $\begin{array} { r } { \sum _ { j = 1 } ^ { N } \Delta _ { j } \le \widetilde { \mathcal { O } } ( L ^ { 2 } | X | ^ { 2 } | A | / \rho ) } \end{array}$ . Lemma 5.2 and $d \leq \rho$ yield the bound on $\tau = \tau _ { N }$ . If $N = 0$ , then $\tau = 0$ . The stopping rules ensure that the returned baseline has a margin of at least $\rho / 2$ . During the search, the algorithm uses the pessimistic cost estimates to mix the exploration policy with the initial baseline so that all constraints are satisfied in every episode.

To complete the proof of Theorem 2, we combine the search regret bound L min $\{ T , \tau \}$ with Theorem 1. If $\tau < T$ , we condition on the history at search termination and apply Theorem 1 with the returned margin of at least $\rho / 2$ . Its guarantee holds for all feasible comparators with high probability, so both phases can use the same comparator in Equation (1). This gives the following bound:

$$
R _ { T } \leq L \tau + \widetilde O \left( \frac { L ^ { 2 } | X | \sqrt { | A | ( T - \tau ) } } { \rho } + \frac { L | X | ^ { 3 } | A | } { \rho } \right) .
$$

If $\tau \geq T .$ , then $R _ { T } \leq L \tau$ . Combining Theorem 3 with $d \leq L \leq | X |$ and allocating $\delta / 2$ to each phase proves Theorem 2.

## 6 Regret Lower Bound

In this section, we show that both margin-dependent terms in Theorem 2 are necessary for algorithms that satisfy the constraints in every episode. The full proofs are given in Appendix F.

Theorem 4 (Lower bound on the regret). Let $0 < d \leq \rho \leq 1 / 8 , \delta \in ( 0 , 1 / 1 6 ]$ , and let $T \geq 1 / ( d \rho )$ be an integer. For any algorithm that, on every instance satisfying Assumption 1 with initial margin d and Slater margin $\rho ,$ ensures with probability at least $1 - \delta$ that $G ^ { \top } q _ { t } \leq \alpha$ holds for every $t \in [ T ]$ , there exists an instance with $L = m = 1 , \left| A \right| = 3$ , and margins d and $\rho$ on which, with probability at least $3 / 4 - 2 \delta$ , it holds:

$$
R _ { T } = \Omega \left( \frac { \sqrt { T } } { \rho } + \frac { 1 } { d \rho } \right) .
$$

Theorem 4 shows that the regret bound in Theorem 2 is tight in $T , d ,$ and $\rho ,$ up to logarithmic factors. The additive $1 / ( d \rho )$ term captures an unavoidable regret cost of exploration from the initial baseline under hard constraints. MA-OPS confines the dependence on d to this additive cost while attaining the optimal $\sqrt { T } / \rho$ leading term without prior knowledge of $\rho .$

Remark 3. For adversarial CMDPs, Stradi et al. (2025a) prove an $\Omega ( \sqrt { T } / \rho )$ regret lower bound for algorithms whose cumulative positive constraint violation is $o ( \sqrt { T } )$ . In constrained MABs with adversarial losses, Genalti et al. (2025) derive a loss-dependent lower bound on expected regret with worst-case term $\Omega ( \sqrt { T } / \rho )$ . For algorithms guaranteeing zero violation, we establish $\Omega \bar { ( } \sqrt { T } / \rho + \bar { 1 } / ( d \rho ) )$ regret with constant probability (Theorem 4), identifying an additional cost associated with the initial margin d. In stochastic MABs, Pacchiano et al. (2025) prove an expected-regret lower bound containing $\Omega ( ( 1 - r _ { b } ) / d ^ { 2 } )$ for suficiently large $T _ { i }$ where $r _ { b } \in ( 0 , 1 )$ is the baseline mean reward. Their construction for this term has $d = \rho$ in our notation, whereas our $\Omega ( 1 / ( d \rho ) )$ term characterizes the joint margin dependence when $d < \rho .$

The proof of Theorem 4 uses two pairs of instances with the associated margins d and $\rho ,$ each sharing the same baseline and cost bounds. For the $\scriptstyle { \sqrt { T } } / \rho$ term, a low-loss action has mean cost at the threshold in one instance and $\Theta ( 1 / \sqrt { T } )$ above it in the other. Safety in the latter requires probability $\Omega ( 1 / ( \rho { \sqrt { T } } ) )$ on higher-loss actions per episode. The observation distributions remain close over $T$ episodes, so change of measure transfers this loss bound with constant probability to the instance where this action is feasible and optimal. For the $1 / ( d \rho )$ term when $d \leq \rho / 4$ , we exchange the cost distributions of two low-loss actions so that a diferent action has margin $\rho$ in each instance. A policy feasible in both assigns only $\mathcal { O } ( d / \rho )$ total probability to these actions. Each such observation contributes $\mathcal { O } ( \rho ^ { 2 } )$ relative entropy, so information per episode in the common feasible set is $\mathcal O ( d \boldsymbol { \rho } )$ . We compare the distributions of histories ending with the selection of the first policy outside this set. By change of measure and safety in both instances, the learner remains in this set for $\Omega ( 1 / ( d \rho ) )$ episodes with constant probability, each incurring constant regret from baseline use. Choosing the first construction when $d \sqrt { T } \geq 1 / 2$ and the second otherwise yields the claimed bound.

## References

Eitan Altman. Constrained Markov Decision Processes. Chapman and Hall/CRC, 1999. doi: 10.1201/978131 5140223.

Sanae Amani, Mahnoosh Alizadeh, and Christos Thrampoulidis. Linear stochastic bandits under safety constraints. In Advances in Neural Information Processing Systems, volume 32, 2019.

Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins. Bandits with knapsacks. Journal of the ACM, 65(3), 2018. URL https://arxiv.org/abs/1305.2545.

Martino Bernasconi, Matteo Castiglioni, Andrea Celli, and Federico Fusco. Beyond primal-dual methods in bandits with stochastic and adversarial constraints. In Advances in Neural Information Processing Systems, volume 37, pages 8541–8568, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/0fd 5675f49141c79ad22d7a533c89b12-Abstract-Conference.html.

Martino Bernasconi, Matteo Castiglioni, and Andrea Celli. No-regret is not enough! Bandits with general constraints through adaptive regret minimization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 3877–3898, 2025. URL https://proceedings.mlr.press/v267/bernasconi25a.html.

Matteo Bollini, Gianmarco Genalti, Francesco Emanuele Stradi, Matteo Castiglioni, and Alberto Marchesi. Replicable constrained bandits. arXiv preprint arXiv:2602.14580, 2026. URL https://arxiv.org/abs/2602.1 4580.

Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

Archana Bura, Aria HasanzadeZonuzy, Dileep Kalathil, Srinivas Shakkottai, and Jean-Francois Chamberland. DOPE: Doubly optimistic and pessimistic exploration for safe reinforcement learning. In Advances in Neural Information Processing Systems, volume 35, pages 1047–1059, 2022. URL https://proceedings.neur ips.cc/paper\_files/paper/2022/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html.

Dongsheng Ding, Xiaohan Wei, Zhuoran Yang, Zhaoran Wang, and Mihailo Jovanovic. Provably eficient safe exploration via primal-dual policy optimization. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 3304–3312, 2021. URL https://proceedings.mlr.press/v130/ding21d.html.

Yuhao Ding and Javad Lavaei. Provably eficient primal-dual reinforcement learning for CMDPs with non-stationary objectives and constraints. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7396–7404, 2023. doi: 10.1609/aaai.v37i6.25900.

Yonathan Efroni, Shie Mannor, and Matteo Pirotta. Exploration-exploitation in constrained MDPs. arXiv preprint arXiv:2003.02189, 2020. URL https://arxiv.org/abs/2003.02189.

Aditya Gangrade, Aldo Pacchiano, Clayton Scott, and Venkatesh Saligrama. Feasible action search for bandit linear programs via Thompson sampling. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 18228–18256, 2025. URL https://proceedings.mlr.press/v267/gangrade25a.html.

Gianmarco Genalti, Francesco Emanuele Stradi, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Data-dependent regret bounds for constrained MABs. In Advances in Neural Information Processing Systems, volume 38, pages 1568–1605, 2025. URL https://papers.nips.cc/paper\_files/paper/2025/hash/0 266d95023740481d22d437aa8aba0e9-Abstract-Conference.html.

Spencer Hutchinson, Berkay Turan, and Mahnoosh Alizadeh. Directional optimism for safe linear bandits. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 658–666, 2024. URL https://proceedings.mlr.press/v238 /hutchinson24a.html.

Spencer Hutchinson, Tianyi Chen, and Mahnoosh Alizadeh. Optimistic safety for online convex optimization with unknown linear constraints. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 2809–2817, 2025. URL https://proceedings.mlr.press/v258/hutchinson25a.html.

Chi Jin, Tiancheng Jin, Haipeng Luo, Suvrit Sra, and Tiancheng Yu. Learning adversarial Markov decision processes with bandit feedback and unknown transition. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 4860–4869, 2020. URL https://proceedings.mlr.press/v119/jin20c.html.

Tiancheng Jin, Longbo Huang, and Haipeng Luo. The best of both worlds: Stochastic and adversarial episodic MDPs with unknown transition. In Advances in Neural Information Processing Systems, volume 34, pages 20491–20502, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/abb9d15b3293a96a3ea116867b2 b16d5-Abstract.html.

Elia Kaufmann, Leonard Bauersfeld, Antonio Loquercio, Matthias Müller, Vladlen Koltun, and Davide Scaramuzza. Champion-level drone racing using deep reinforcement learning. Nature, 620:982–987, 2023. doi: 10.1038/s41586-023-06419-4. URL https://www.nature.com/articles/s41586-023-06419-4.

Toshinori Kitamura, Arnob Ghosh, Tadashi Kozuno, Wataru Kumagai, Kazumi Kasaura, Kenta Hoshino, Yohei Hosoe, and Yutaka Matsuo. Provably eficient RL under episode-wise safety in constrained MDPs with linear function approximation. In Advances in Neural Information Processing Systems, volume 38, pages 47141–47190, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/435e8fbbfc2 c6072d4f3a5cb6e56a39a-Abstract-Conference.html.

Tor Lattimore and Csaba Szepesvári. Bandit Algorithms. Cambridge University Press, 2020. URL https: //tor-lattimore.com/downloads/book/book.pdf.

Donghao Li, Ruiquan Huang, Cong Shen, and Jing Yang. Near-optimal conservative exploration in reinforcement learning under episode-wise constraints. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19527–19564, 2023. URL https://proceedings.mlr.press/v202/li23k.html.

Long-Fei Li, Peng Zhao, and Zhi-Hua Zhou. Improved algorithm for adversarial linear mixture MDPs with bandit feedback and unknown transition. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 3061–3069, 2024. URL https://proceedings.mlr.press/v238/li24n.html.

Chang Liu, Yunfan Li, and Lin F. Yang. Near-optimal sample complexity for online constrained MDPs. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc /paper\_files/paper/2025/hash/445500190fe42e9f58f65199d2bd70d1-Abstract-Conference.html.

Tao Liu, Ruida Zhou, Dileep Kalathil, P. R. Kumar, and Chao Tian. Learning policies with zero or bounded constraint violation for constrained MDPs. In Advances in Neural Information Processing Systems, volume 34, pages 17183–17193, 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/hash/8 ec2ba5e96ec1c050bc631abda80f269-Abstract.html.

Xingtu Liu, Lin F. Yang, and Sharan Vaswani. Sample complexity bounds for linear constrained MDPs with a generative model. In Proceedings of The 37th International Conference on Algorithmic Learning Theory, volume 313 of Proceedings of Machine Learning Research, pages 1–70, 2026. URL https://proceedings.mlr. press/v313/liu26b.html.

Haipeng Luo, Chen-Yu Wei, and Chung-Wei Lee. Policy optimization in adversarial MDPs: Improved exploration via dilated bonuses. In Advances in Neural Information Processing Systems, volume 34, pages 22931–22942, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/c1b8bf9e071c0dabb899e7a27f3 53762-Abstract.html.

Shie Mannor, John N. Tsitsiklis, and Jia Yuan Yu. Online learning with sample path constraints. Journal of Machine Learning Research, 10(20):569–590, 2009. URL https://jmlr.org/papers/v10/mannor09a.html.

Andreas Maurer and Massimiliano Pontil. Empirical Bernstein bounds and sample variance penalization. In Proceedings of the 22nd Annual Conference on Learning Theory, 2009. URL https://arxiv.org/abs/0907.3 740.

Adrian Müller, Pragnya Alatur, Volkan Cevher, Giorgia Ramponi, and Niao He. Truly no-regret learning in constrained MDPs. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 36605–36653, 2024. URL https://proceedings.mlr.pres s/v235/muller24b.html.

Tingting Ni and Maryam Kamgarpour. A safe exploration approach to constrained Markov decision processes. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3592–3600, 2025. URL https://proceedings.mlr.press/v2 58/ni25a.html.

Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/202 2/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html.

Aldo Pacchiano, Mohammad Ghavamzadeh, Peter Bartlett, and Heinrich Jiang. Stochastic bandits with linear constraints. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 2827–2835, 2021. URL https: //proceedings.mlr.press/v130/pacchiano21a.html.

Aldo Pacchiano, Mohammad Ghavamzadeh, and Peter Bartlett. Contextual bandits with stage-wise constraints. Journal of Machine Learning Research, 26(170):1–57, 2025. URL https://www.jmlr.org/papers/v26/24-026 7.html.

Shuang Qiu, Xiaohan Wei, Zhuoran Yang, Jieping Ye, and Zhaoran Wang. Upper confidence primal-dual reinforcement learning for CMDP with adversarial loss. In Advances in Neural Information Processing Systems, volume 33, pages 15277–15287, 2020. URL https://proceedings.neurips.cc/paper\_files/paper/202 0/hash/ae95296e27d7f695f891cd26b4f37078-Abstract.html.

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of Go with deep neural networks and tree search. Nature, 529:484–489, 2016. doi: 10.1038/nature16961. URL https://www.nature.com/articles/nature16961.

Karthik Sridharan and Seung Won Wilson Yoo. Online learning with unknown constraints. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 56790–56819, 2025. URL https://proceedings.mlr.press/v267/sridharan25a.html.

Francesco Emanuele Stradi, Jacopo Germano, Gianmarco Genalti, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Online learning in CMDPs: Handling stochastic and adversarial constraints. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 46692–46721, 2024. URL https://proceedings.mlr.press/v235/stradi24a.html.

Francesco Emanuele Stradi, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Learning adversarial MDPs with stochastic hard constraints. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 56920–56951, 2025a. URL https://proceedings.mlr.press/v267/stradi25a.html.

Francesco Emanuele Stradi, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Optimal strong regret and violation in constrained MDPs via policy optimization. In The Thirteenth International Conference on Learning Representations, pages 52056–52081, 2025b. URL https://proceedings.iclr.cc/paper\_files/paper /2025/hash/8105a80cea247fa6f8441a8069b65f2a-Abstract-Conference.html.

Francesco Emanuele Stradi, Eleonora Fidelia Chiefari, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Beyond Slater’s condition in online CMDPs with stochastic and adversarial constraints. arXiv preprint arXiv:2509.20114, 2025c. URL https://arxiv.org/abs/2509.20114.

Francesco Emanuele Stradi, Anna Lunghi, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Policy optimization for CMDPs with bandit feedback: Learning stochastic and adversarial constraints. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 56952–56989, 2025d. URL https://proceedings.mlr.press/v267/stradi2 5b.html.

Francesco Emanuele Stradi, Anna Lunghi, Matteo Castiglioni, Alberto Marchesi, and Nicola Gatti. Taming adversarial constraints in CMDPs. In Advances in Neural Information Processing Systems, volume 38, pages 9697–9745, 2025e. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/0df684f8bb7 e8c0bcdb8ea49be5035b3-Abstract-Conference.html.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, second edition, 2018. URL https://mitpress.mit.edu/9780262039246/reinforcement-learning/.

Sharan Vaswani, Lin Yang, and Csaba Szepesvári. Near-optimal sample complexity bounds for constrained MDPs. In Advances in Neural Information Processing Systems, volume 35, pages 3110–3122, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/14a5ebc9cd2e507cd811df78c15bf5d7-Abstr act-Conference.html.

Honghao Wei, Xin Liu, and Lei Ying. Triple-Q: A model-free algorithm for constrained reinforcement learning with sublinear regret and zero constraint violation. In Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pages 3274–3307, 2022. URL https://proceedings.mlr.press/v151/wei22a.html.

Honghao Wei, Arnob Ghosh, Ness Shrof, Lei Ying, and Xingyu Zhou. Provably eficient model-free algorithms for non-stationary CMDPs. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 6527–6570, 2023. URL https://proceedings.mlr.press/v206/wei23b.html.

Yukuan Wei, Xudong Li, and Lin F. Yang. Near-optimal sample complexity bounds for constrained averagereward MDPs. In International Conference on Learning Representations, 2026. URL https://proceedings. iclr.cc/paper\_files/paper/2026/file/ceadbf97decaba4054a48f45d9a61c62-Paper-Conference.pdf.

Yifan Wu, Roshan Sharif, Tor Lattimore, and Csaba Szepesvári. Conservative bandits. In Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1254–1262, 2016. URL https://proceedings.mlr.press/v48/wu16.html.

Kihyun Yu, Duksang Lee, William Overman, and Dabeen Lee. Improved regret bound for safe reinforcement learning via tighter cost pessimism and reward optimism. Reinforcement Learning Journal, 6:493–546, 2025. URL https://rlj.cs.umass.edu/2025/papers/Paper41.html.

Kihyun Yu, Seoungbin Bae, and Dabeen Lee. Primal-dual policy optimization for linear CMDPs with adversarial losses. In The Fourteenth International Conference on Learning Representations, pages 49875– 49923, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/51f547584cd1fcb87114ea02282 2a60d-Abstract-Conference.html.

Liyuan Zheng and Lillian Ratlif. Constrained upper confidence reinforcement learning. In Proceedings of the 2nd Conference on Learning for Dynamics and Control, volume 120 of Proceedings of Machine Learning Research, pages 620–629. PMLR, 2020. URL https://proceedings.mlr.press/v120/zheng20a.html.

Jiahui Zhu, Kihyun Yu, Dabeen Lee, Xin Liu, and Honghao Wei. An optimistic algorithm for online CMDPs with anytime adversarial constraints. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 80347–80372, 2025. URL https://proceedings.mlr.press/v267/zhu25ab.html.

Alexander Zimin and Gergely Neu. Online learning in episodic Markovian decision processes by relative entropy policy search. In Advances in Neural Information Processing Systems, volume 26, 2013. URL https://proceedings.neurips.cc/paper/2013/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html.

A Additional Related Work 17   
B Occupancy Measures and Confidence Bounds 18   
B.1 Occupancy Measures . . 18   
B.2 Estimators and Confidence Bounds . 19   
B.3 Occupancy Error Bounds 21   
C Analysis of S-OPS 22   
C.1 Algorithm Details . . 22   
C.2 Zero Constraint Violation . 24   
C.3 Occupancy-Weighted Estimation Errors 24   
C.4 Proof of Theorem 1 . 29   
D Analysis of Baseline Search 32   
D.1 Exploration Episodes . . 32   
D.2 Margin Bounds and Constraint Violation 33   
D.3 Number of Exploration Episodes 34   
D.4 Number of Baseline Episodes 35   
D.5 Proof of Theorem 3 . 36   
E Proof of Theorem 2 37   
F Proof of Theorem 4 38

## A Additional Related Work

In this section, we provide further discussion of related work on online learning and reinforcement learning.

Online CMDPs with cumulative violation In stochastic CMDPs with unknown transitions, Efroni et al. (2020) use linear programming to bound strong regret and cumulative positive violation with $\widetilde { \mathcal { O } } ( \sqrt { T } )$ dependence on T. For primal-dual policy optimization, Müller et al. (2024) establish sublinear bounds for both quantities, and Stradi et al. (2025b) attain the $\widetilde { \mathcal { O } } ( \sqrt { T } )$ rate in T. Under stochastic constraints, Stradi et al. (2025c) remove Slater’s condition for regret and cumulative positive violation. For cumulative signed violation, guarantees range from sublinear bounds (Qiu et al., 2020) to nonpositive violation after suficiently many episodes under Slater’s condition (Wei et al., 2022). For adversarial linear CMDPs, Yu et al. (2026) control cumulative violation with full-information losses and bandit cost observations. With changing constraints, the hindsight optimum under average constraints can be unattainable with fully adversarial inputs (Mannor et al., 2009). This motivates competitive guarantees (Stradi et al., 2024, 2025d) or restrictions on temporal variation for dynamic regret (Ding and Lavaei, 2023; Wei et al., 2023). With non-stationary rewards and costs, Stradi et al. (2025e) use a fixed comparator feasible on average, whereas Zhu et al. (2025) study stochastic rewards and adversarial constraints against a fixed policy feasible in every episode.

Online CMDPs with hard constraints Zheng and Ratlif (2020) study feasibility during learning under known dynamics and average-cost constraints. For stochastic episodic CMDPs with unknown transitions, Liu et al. (2021), Bura et al. (2022), and Yu et al. (2025) use a known strictly feasible baseline. Known-baseline guarantees also extend to linear function approximation (Kitamura et al., 2025) and discounted CMDPs (Ni and Kamgarpour, 2025). Conservative reinforcement learning constrains each episode’s expected reward relative to a baseline (Li et al., 2023). A known baseline provides more information than Slater’s condition alone, which also serves to bound optimal dual multipliers in primal-dual analyses (Ding et al., 2021). The main known-baseline analyses use exact expected costs (Liu et al., 2021; Bura et al., 2022; Yu et al., 2025; Stradi et al., 2025a), so input and actual baseline margins coincide. Our formulation allows upper bounds on baseline costs. A small d may therefore reflect conservative cost information even when the actual baseline margin is large.

Constrained bandits Budget and long-term formulations couple resource use across rounds (Badanidiyuru et al., 2018; Bernasconi et al., 2024, 2025), while conservative bandits impose a cumulative reward requirement relative to a baseline (Wu et al., 2016). For per-round constraints, feasibility may apply to an individual action (Amani et al., 2019; Hutchinson et al., 2024) or to the expected cost of a randomized decision (Pacchiano et al., 2025). The latter allows positive probability on individually infeasible actions and is the relevant formulation for our policy mixtures. Several results for randomized decisions assume known baseline costs (Pacchiano et al., 2021, 2025). Pacchiano et al. (2025) also assume a known baseline reward in their main upper-bound analysis, treating unknown baseline values separately. For adversarial losses, Genalti et al. (2025) derive data-dependent bounds using a baseline with known costs. Their main analysis uses a maximum-margin baseline, while their results also extend to a general strictly feasible baseline. Beyond bandits, related work studies per-round feasibility in online convex optimization with linear constraints (Hutchinson et al., 2025) and through regression and online learning oracles for more general constraint classes (Sridharan and Yoo, 2025).

Sample complexity under constraints Sample-complexity guarantees have been established with a generative model in discounted tabular CMDPs (Vaswani et al., 2022), discounted linear CMDPs (Liu et al., 2026), and average-reward CMDPs (Wei et al., 2026), as well as through online interaction (Liu et al., 2025). For exact feasibility of the output, these bounds exhibit inverse-square dependence on the Slater margin without requiring every data-collection policy to be feasible. For unknown linear constraints, Gangrade et al. (2025) find a nearly maximum-margin action or detect infeasibility while controlling cumulative positive violation during the search. We bound the regret of learning a larger-margin baseline while keeping every episode feasible.

Adversarial MDPs Occupancy-measure methods provide a basis for learning in episodic MDPs (Zimin and Neu, 2013). For unknown transitions and bandit feedback, Jin et al. (2020) combine online mirror descent, transition confidence sets, and upper occupancy bounds. Subsequent developments include policy optimization with dilated bonuses (Luo et al., 2021), guarantees for stochastic and adversarial losses (Jin et al., 2021), and linear mixture models (Li et al., 2024). These works study MDPs without episode-wise cost constraints.

## B Occupancy Measures and Confidence Bounds

We establish the confidence and occupancy error bounds used in the analyses of S-OPS and baseline search.

## B.1 Occupancy Measures

We use the occupancy notation from Section 2. Inner products, coordinatewise comparisons, and $\ell _ { 1 }$ norms use state–action marginals, while flow constraints and relative entropy use the full transition occupancies. At a state with $q ( x ) = 0$ , define $\pi ^ { q } ( a \mid x ) = 1 / \vert A \vert$ . Following Jin et al. (2020, Appendix A.1), we represent an occupancy by its nonnegative entries $\boldsymbol { q } ( \boldsymbol { x } , \boldsymbol { a } , \boldsymbol { x } ^ { \prime } )$ on consecutive layers, with marginals $\begin{array} { r } { q ( x , a ) = \sum _ { x ^ { \prime } } q ( x , a , x ^ { \prime } ) } \end{array}$ These entries satisfy

$$
\begin{array} { r l } & { \displaystyle \sum _ { a \in A , x ^ { \prime } \in X _ { 1 } } q ( x _ { 0 } , a , x ^ { \prime } ) = 1 , } \\ & { \displaystyle \sum _ { a \in A , x ^ { \prime } \in X _ { k + 1 } } q ( x , a , x ^ { \prime } ) = \displaystyle \sum _ { y \in X _ { k - 1 } , a \in A } q ( y , a , x ) , \quad x \in X _ { k } , \quad 1 \leq k < L , } \\ & { \displaystyle \left( \widehat { P } ( x ^ { \prime } \mid x , a ) - \epsilon ( x , a , x ^ { \prime } ) \right) q ( x , a ) \leq q ( x , a , x ^ { \prime } ) \leq \left( \widehat { P } ( x ^ { \prime } \mid x , a ) + \epsilon ( x , a , x ^ { \prime } ) \right) q ( x , a ) , } \end{array}\tag{B.1}
$$

where the last line holds for every valid transition and uses the estimates and confidence bounds from Appendix B.2. These linear constraints define the convex polytope $\Delta ( \mathcal { P } )$ . Adding $\begin{array} { r } { q ( x , a ) = \pi ( a \mid x ) \sum _ { b } q ( x , b ) } \end{array}$ gives $\Delta ( \mathcal { P } , \pi )$ . Consequently, both the optimistic margin problem and each fixed-policy cost maximization in (6) are linear programs. Each fixed-policy problem maximizes the total expected cost over a single kernel $P ^ { \prime } \in \mathcal { P }$ . For the upper occupancy bounds in S-OPS, the maximizing kernel may depend on $( x , a )$

Lemma B.1. Fix a Markov policy π, and let P be a transition confidence set defined in Appendix B.2. For every $q \in \Delta ( \mathcal { P } , \pi )$ , there exists $P ^ { q } \in \mathcal { P }$ such that $q = q ^ { P ^ { q } , \pi }$

Proof. For $q ( x , a ) > 0$ , set $P ^ { q } ( x ^ { \prime } \mid x , a ) = q ( x , a , x ^ { \prime } ) / q ( x , a )$ . The flow and confidence constraints ensure that $\textstyle P ^ { q } ( \cdot \mid x , a )$ is a probability distribution satisfying the confidence bounds at $( x , a )$ . If $q ( x , a ) = 0$ , choose any transition distribution satisfying these bounds. Such a distribution exists by Appendix B.2. Since the confidence constraints apply separately to each state–action pair, the constructed kernel belongs to $\mathcal { P }$ Nonnegativity gives $q ( x , a ) P ^ { q } ( x ^ { \prime } \mid x , a ) = q ( x , a , x ^ { \prime } )$ in both cases. The initial state has probability $q ( x _ { 0 } ) = 1$ Suppose each state $x \in X _ { k }$ is reached with probability q(x), where $k < L - 1$ . The fixed-policy constraints and flow conservation then give

$$
\begin{array} { l } { \mathbb { P } _ { P ^ { q } , \pi } ( x _ { k + 1 } = x ^ { \prime } ) = \displaystyle \sum _ { x \in X _ { k } , a } q ( x ) \pi ( a \mid x ) P ^ { q } ( x ^ { \prime } \mid x , a ) } \\ { = \displaystyle \sum _ { x \in X _ { k } , a } q ( x , a , x ^ { \prime } ) = q ( x ^ { \prime } ) . } \end{array}
$$

By induction, $q ( x )$ equals the probability of visiting each nonterminal state x. Thus $q ^ { P ^ { q } , \pi } ( x , a , x ^ { \prime } ) = q ( x ) \pi ( a \mid$ x) $\vert P ^ { q } ( x ^ { \prime } \mid x , a ) = q ( x , a , x ^ { \prime } )$ □

In particular, each $\widehat { q } \in \Delta ( \mathcal { P } )$ is induced by $\pi ^ { \widehat { q } }$ and some $P ^ { \prime } \in \mathcal { P }$

Optimistic margin search We introduce a scalar $z \in \mathbb { R }$ and write (5) as the linear program

$$
\begin{array} { r l } { \underset { q , z } { \operatorname* { m a x } } } & { z } \\ { \mathrm { s u b j e c t ~ t o } } & { q \in \Delta ( \mathcal { P } _ { n - 1 } ) , } \\ & { ( \widehat { g } _ { n - 1 , i } - \xi _ { n - 1 } ) ^ { \top } q + z \leq \alpha _ { i } , \qquad i \in [ m ] , } \\ & { z \leq L . } \end{array}\tag{B.2}
$$

For fixed $q ,$ maximizing over z recovers the objective in (5). This objective is continuous on the nonempty compact set $\Delta ( \mathcal { P } _ { n - 1 } )$ , so both problems attain the same maximum.

## B.2 Estimators and Confidence Bounds

Let $\mathcal { F } _ { t }$ denote the σ-algebra generated by the interaction history through episode t and the loss vectors $\ell _ { 1 } , \ldots , \ell _ { t }$ . The Markov policies and mixing probabilities used in episode t are $\mathcal { F } _ { t - 1 }$ -measurable. We write $\mathbb { E } _ { t } [ Z ] = \mathbb { E } [ Z \mid { \mathcal { F } } _ { t - 1 } , \ell _ { t } ]$ . Conditional on $\mathcal { F } _ { t - 1 }$ and $\ell _ { t }$ , the cost matrix $G _ { t }$ has distribution ${ \mathcal { G } } .$ It is independent of the policy draw and the random variables used to sample actions and state transitions.

Index the episodes used for estimation by $j .$ Let $I _ { j } ( x , a )$ and $J _ { j } ( x , a , x ^ { \prime } )$ indicate whether episode $j$ in this sequence visits $( x , a )$ and takes the transition $( x , a , x ^ { \prime } )$ , respectively. With $\begin{array} { r } { N ( x , a ) = \sum _ { j } I _ { j } ( x , a ) } \end{array}$ and $\begin{array} { r } { M ( x , a , x ^ { \prime } ) = \sum _ { j } J _ { j } ( x , a , x ^ { \prime } ) } \end{array}$ , define

$$
\begin{array} { c } { { \widehat { g } _ { i } ( x , a ) = \displaystyle \frac { \sum _ { j } G _ { j , i } ( x , a ) I _ { j } ( x , a ) } { \operatorname* { m a x } \{ 1 , N ( x , a ) \} } , } } \\ { { \widehat { P } ( x ^ { \prime } \mid x , a ) = \displaystyle \frac { M ( x , a , x ^ { \prime } ) } { \operatorname* { m a x } \{ 1 , N ( x , a ) \} } . } } \end{array}\tag{B.3}
$$

Here $G _ { j , i } ( x , a )$ is the cost of constraint i at $( x , a )$ in episode $j$ of the estimation sequence. For S-OPS, j counts all episodes in the current phase, so $G _ { j , i } = g _ { j , i }$ . During baseline search, $j$ counts exploration episodes, so $G _ { j , i } = g _ { \tau _ { j } , i }$ , where $\tau _ { j }$ is their global episode index. For a logarithmic parameter $u ,$ set

(B.4)

$$
\begin{array} { l } { \displaystyle \xi ( \boldsymbol { x } , a ) = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { 4 u } { \operatorname* { m a x } \{ 1 , N ( \boldsymbol { x } , a ) \} } } \right\} , } \\ { \displaystyle \epsilon ( \boldsymbol { x } , a , \boldsymbol { x ^ { \prime } } ) = \operatorname* { m i n } \left\{ 1 , 2 \sqrt { \frac { \hat { P } ( \boldsymbol { x ^ { \prime } } \mid \boldsymbol { x } , a ) u } { \operatorname* { m a x } \{ 1 , N ( \boldsymbol { x } , a ) - 1 \} } } + \frac { 1 4 u } { 3 \operatorname* { m a x } \{ 1 , N ( \boldsymbol { x } , a ) - 1 \} } \right\} . } \end{array}\tag{B.5}
$$

The confidence set $\mathcal { P }$ consists of all transition kernels $P ^ { \prime }$ satisfying $| P ^ { \prime } ( x ^ { \prime } \mid x , a ) - \widehat { P } ( x ^ { \prime } \mid x , a ) | \leq \epsilon ( x , a , x ^ { \prime } )$ for $x \in X _ { k } , a \in A , x ^ { \prime } \in X _ { k + 1 }$ , and $0 \le k < L$ . At every visited state–action pair, the empirical transition distribution satisfies these bounds. At an unvisited pair, every transition distribution satisfies them. Thus $\mathcal { P }$ is nonempty. With the logarithmic parameters used below, initialization gives $\widehat { G } _ { 0 } = 0$ and $\xi _ { 0 } ( x , a ) = 1$ for every $( x , a )$ . The set $\mathcal { P } _ { 0 }$ then contains every transition kernel allowed by the layered model. For S-OPS, define the confidence event

$$
\mathcal { E } _ { T } = \{ P \in \mathcal { P } _ { t } , | \widehat { G } _ { t } - G | \leq \Xi _ { t } \mathrm { ~ f o r ~ e v e r y ~ } 0 \leq t \leq T \} .\tag{B.6}
$$

For baseline search, we allow the phase to continue beyond T until a stopping condition is met. Let ${ \mathcal E } _ { \mathrm { s } }$ be the event that $P \in \mathcal P _ { n - 1 }$ and $| \widehat { G } _ { n - 1 } - G | \leq \Xi _ { n - 1 }$ at initialization and after every update. The next lemma bounds the failure probability in each phase, with $\delta$ denoting its confidence parameter.

Lemma B.2. Under the observation model in Section ${ \mathit { 2 } } ,$ S-OPS with $\begin{array} { r } { u = \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \end{array}$ satisfies $\mathbb { P } ( \mathcal { E } _ { T } ) \geq$ $1 - \delta / 8$ , and baseline search with $\begin{array} { r } { u = \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } \end{array}$ satisfies $\mathbb { P } ( \mathcal { E } _ { \mathrm { s } } ) \geq 1 - \delta / 8$

Proof. Fix $( x , a ) \in X _ { k } \times A$ . For $1 \leq r \leq N ( x , a )$ , let $\begin{array} { r } { j _ { r } = \operatorname* { m i n } \{ j : \sum _ { s = 1 } ^ { j } I _ { s } ( x , a ) = r \} } \end{array}$ be the index of its r-th visit. By the observation model and the use of a policy fixed throughout each episode, visiting $( x , a )$ is independent of the current cost matrix conditional on past observations. For each fixed $i ,$ we can therefore view the costs $G _ { j _ { r } , i } ( x , a )$ as successive samples from an i.i.d. sequence with mean $g _ { i } ( x , a )$ . For each fixed $x ^ { \prime } \in X _ { k + 1 }$ , the indicators $J _ { j _ { r } } ( x , a , x ^ { \prime } )$ are similarly obtained from successive Bernoulli samples with mean $P ( x ^ { \prime } \mid x , a )$ . We apply concentration at each fixed visit count r and then take a union bound over the counts.

Cost estimates For each $i \in [ m ]$ , let ${ \widehat { g } } _ { i } ( x , a )$ denote the average of the first $r \geq 1$ cost samples. Hoefding’s inequality gives, for any $u > 0$ ，

$$
\mathbb { P } \Big ( | \widehat { g } _ { i } ( x , a ) - g _ { i } ( x , a ) | > \sqrt { 4 u / r } \Big ) \leq 2 e ^ { - 8 u } \leq 2 e ^ { - u } .
$$

With no observations, $\widehat { g } _ { i } ( x , a ) = 0$ and $\xi ( x , a ) = 1$ . Since both the estimate and the mean cost belong to $[ 0 , 1 ]$ , truncating the confidence bound at one preserves the inequality.

Transition probabilities Fix $x ^ { \prime } \in X _ { k + 1 }$ and write $p = P ( x ^ { \prime } \mid x , a )$ . For $r \geq 2$ , let $\widehat { p }$ be the empirical mean of the first r next-state indicators. Their sample variance is $r \widehat { p } ( 1 - \widehat { p } ) / ( r - 1 )$ . We apply the empirical Bernstein inequality of Maurer and Pontil (2009, Theorem 4) to the indicator and its complement, each with failure probability $e ^ { - u }$ . With probability at least $1 - 2 e ^ { - u }$ ，

$$
\begin{array} { c } { \displaystyle { | \widehat { p } - p | \leq \sqrt { \frac { 2 \widehat { p } ( 1 - \widehat { p } ) ( u + \log 2 ) } { r - 1 } } + \frac { 7 ( u + \log 2 ) } { 3 ( r - 1 ) } } } \\ { \leq 2 \sqrt { \frac { \widehat { p } u } { r - 1 } } + \frac { 1 4 u } { 3 ( r - 1 ) } . } \end{array}
$$

Both choices of u satisfy $u \geq 1$ . The second inequality then follows from $1 - \widehat { p } \leq 1$ and $u + \log 2 \leq 2 u$ . Since $| { \widehat { p } } - p | \leq 1$ , truncating the bound at one gives (B.5). For $r = 0 , 1 , \epsilon ( x , a , x ^ { \prime } ) = 1$

Union bound For each coordinate and fixed positive visit count, the confidence bound fails with probability at most $2 e ^ { - u }$ . The layer sizes satisfy

$$
\sum _ { k = 0 } ^ { L - 1 } | X _ { k } | | X _ { k + 1 } | \leq \left( \sum _ { j = 0 } ^ { \lfloor L / 2 \rfloor } | X _ { 2 j } | \right) \left( \sum _ { j = 0 } ^ { \lfloor ( L - 1 ) / 2 \rfloor } | X _ { 2 j + 1 } | \right) \leq \frac { | X | ^ { 2 } } { 4 } .
$$

Together with $| X | - 1 \leq | X | ^ { 2 } / 4$ and $m \geq 1$ , this bounds the total number of cost and transition entries by

$$
m | A | ( | X | - 1 ) + | A | \sum _ { k = 0 } ^ { L - 1 } | X _ { k } | | X _ { k + 1 } | \leq \frac { m | X | ^ { 2 } | A | } { 2 } .
$$

For S-OPS, take $\begin{array} { r } { u = \log { \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } } \end{array}$ and sum over at most $T$ positive visit counts. The resulting failure probability satisfies

$$
\mathbb { P } ( \mathcal { E } _ { T } ^ { c } ) \leq m | X | ^ { 2 } | A | T \cdot \frac { \delta } { 8 m | X | ^ { 2 } | A | T } = \frac { \delta } { 8 } .
$$

For the search estimates, use $u = \log ( m | X | ^ { 2 } | A | ( r + 2 ) ^ { 3 } / \delta )$ at visit count $r \geq 1$ . If $N _ { n - 1 } ( x , a ) = r$ , then $n \geq r + 1$ . The logarithmic parameter used at search index n is therefore at least this large, so the same confidence bounds apply. The total failure probability over all visit counts is at most

$$
\begin{array} { l } { { \displaystyle m | X | ^ { 2 } | A | \sum _ { r = 1 } ^ { \infty } \frac { \delta } { m | X | ^ { 2 } | A | ( r + 2 ) ^ { 3 } } = \delta \sum _ { k = 3 } ^ { \infty } \frac { 1 } { k ^ { 3 } } } } \\ { { \displaystyle ~ < \delta \int _ { 2 } ^ { \infty } x ^ { - 3 } d x = \frac { \delta } { 8 } . } } \end{array}
$$

Thus $\mathbb { P } ( \mathcal { E } _ { \mathrm { s } } ) \geq 1 - \delta / 8$

## B.3 Occupancy Error Bounds

For a fixed policy, we bound the diference between its true occupancy measure and those induced by kernels in the confidence set. We also bound the error in the upper occupancy bounds used by S-OPS. For a transition confidence set $\mathcal { P }$ with confidence bounds ϵ and $x \in X _ { k }$ , define

$$
b ( x , a ) = \operatorname* { m i n } \left\{ 2 , 2 \sum _ { x ^ { \prime } \in X _ { k + 1 } } \epsilon ( x , a , x ^ { \prime } ) \right\} .\tag{B.7}
$$

For a Markov policy π, write $q = q ^ { P , \pi }$ and $u ^ { \pi } ( x , a ) = \mathrm { m a x } _ { P ^ { \prime } \in { \mathcal P } } q ^ { P ^ { \prime } , \pi } ( x , a )$

Lemma B.3. If $P \in { \mathcal { P } } $ , then $\| q ^ { P ^ { \prime } , \pi } - q \| _ { 1 } \leq L b ^ { \top } q$ for all $P ^ { \prime } \in \mathcal { P }$ , and $\| u ^ { \pi } - q \| _ { 1 } \leq | X | b ^ { \top } q$

Proof. Fix $P ^ { \prime } \in \mathcal { P }$ and write $q ^ { \prime } = q ^ { P ^ { \prime } , \pi }$ . Since both transition kernels belong to $\mathcal { P }$ , the confidence constraints give $\| P ^ { \prime } ( \cdot \mid x , a ) - P ( \cdot \mid x , a ) \| _ { 1 } \leq b ( x , a )$ . For $0 \leq k < L - 1$ and $y \in X _ { k + 1 }$ , the flow constraints give

$$
q ^ { \prime } ( y ) - q ( y ) = \sum _ { x \in X _ { k } , a } \left( q ^ { \prime } ( x , a ) - q ( x , a ) \right) P ^ { \prime } ( y \mid x , a ) + \sum _ { x \in X _ { k } , a } q ( x , a ) \big ( P ^ { \prime } ( y \mid x , a ) - P ( y \mid x , a ) \big ) .
$$

Both occupancies use $\pi ,$ so $q ^ { \prime } ( x , a ) - q ( x , a ) = \pi ( a \mid x ) ( q ^ { \prime } ( x ) - q ( x ) )$ . Taking absolute values and summing over $y \in X _ { k + 1 }$ gives

$$
\sum _ { y \in X _ { k + 1 } } \vert q ^ { \prime } ( y ) - q ( y ) \vert \leq \sum _ { x \in X _ { k } } \vert q ^ { \prime } ( x ) - q ( x ) \vert + \sum _ { x \in X _ { k } , a } q ( x , a ) b ( x , a ) .
$$

The initial distributions agree. Iterating this inequality and summing over k therefore yields

$$
\begin{array} { r l } & { \| q ^ { \prime } - q \| _ { 1 } = \displaystyle \sum _ { k = 0 } ^ { L - 1 } \displaystyle \sum _ { x \in X _ { k } } | q ^ { \prime } ( x ) - q ( x ) | } \\ & { \qquad \le \displaystyle \sum _ { h = 0 } ^ { L - 2 } ( L - 1 - h ) \displaystyle \sum _ { x \in X _ { h } , a } q ( x , a ) b ( x , a ) \le L b ^ { \top } q . } \end{array}
$$

For the upper occupancy bound, fix $y \in X _ { k }$ . The same recurrence applies to every $P ^ { \prime } \in \mathcal { P }$ , and $P \in \mathcal { P }$ , so

$$
0 \leq \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } q ^ { P ^ { \prime } , \pi } ( y ) - q ( y ) \leq \sum _ { h = 0 } ^ { k - 1 } \sum _ { x \in X _ { h } , a } q ( x , a ) b ( x , a ) .
$$

Since $q ^ { P ^ { \prime } , \pi } ( y , a ) = \pi ( a \mid y ) q ^ { P ^ { \prime } , \pi } ( y )$ , summing first over actions and then over states gives

$$
\begin{array} { r l } & { \| u ^ { \pi } - q \| _ { 1 } = \displaystyle \sum _ { k = 0 } ^ { L - 1 } \sum _ { y \in X _ { k } } \left( \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } } q ^ { P ^ { \prime } , \pi } ( y ) - q ( y ) \right) } \\ & { \qquad \leq \displaystyle \sum _ { k = 0 } ^ { L - 1 } | X _ { k } | \sum _ { h = 0 } ^ { k - 1 } \sum _ { x \in X _ { h } , a } q ( x , a ) b ( x , a ) \leq | X | b ^ { \top } q . } \end{array}
$$

We next bound $b ( x , a )$ in terms of the visit count $N ( x , a )$ . If $r = N ( x , a ) \geq 2 , x \in X _ { k }$ , and the logarithmic parameter is $u \geq 1$ , then Cauchy–Schwarz and $\textstyle \sum _ { x ^ { \prime } } { \widehat { P } } ( x ^ { \prime } \mid x , a ) = 1$ give

$$
\begin{array} { c l l } { \displaystyle b ( \boldsymbol { x } , \boldsymbol { a } ) \leq \operatorname* { m i n } \left\{ 2 , 4 \sqrt { \frac { | X _ { k + 1 } | u } { r - 1 } } + \frac { 2 8 | X _ { k + 1 } | u } { 3 ( r - 1 ) } \right\} } \\ { \leq \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { | X _ { k + 1 } | u } { \operatorname* { m a x } \{ 1 , r \} } } \right\} \right) . } \end{array}\tag{B.8}
$$

For the last inequality, use $r - 1 \geq r / 2$ and separate the cases $| X _ { k + 1 } | u / r \le 1$ and $| X _ { k + 1 } | u / r > 1$ . For $r = 0 , 1$ the last bound holds because $b ( x , a ) \leq 2$ and $| X _ { k + 1 } | u \geq 1$

Cumulative visit counts For a finite sequence of episodes, write $\begin{array} { r } { N _ { t } ( x , a ) = \sum _ { s = 1 } ^ { t } I _ { s } ( x , a ) } \end{array}$ . For each $( x , a )$

$$
\sum _ { t = 1 } ^ { T } \frac { I _ { t } ( x , a ) } { \sqrt { \operatorname* { m a x } \{ 1 , N _ { t - 1 } ( x , a ) \} } } = \sum _ { r = 0 } ^ { N _ { T } ( x , a ) - 1 } \frac { 1 } { \sqrt { \operatorname* { m a x } \{ 1 , r \} } } \leq 3 \sqrt { N _ { T } ( x , a ) } ,\tag{B.9}
$$

where an empty sum is zero. The equality follows by indexing the visits to $( x , a )$ , and the inequality follows by integration. During baseline search, we apply this bound to the n exploration episodes.

## C Analysis of S-OPS

We first establish zero constraint violation on ${ \mathcal { E } } _ { T }$ . We then combine the estimation bounds with the mirror descent analysis to prove the regret guarantee in Theorem 1.

## C.1 Algorithm Details

Algorithm 2 uses baseline cost upper bounds $\beta$ satisfying Assumption 1 and the parameters in Theorem 1.   
The estimates and confidence sets are defined in Appendix B.2.

Loss estimation Write $\widehat { \pi } _ { t } = \pi ^ { \widehat { q _ { t } } }$ . In episode $t ,$ the randomized policy $\pi _ { t }$ selects $\pi ^ { \diamond }$ with probability $\lambda _ { t - 1 }$ and $\widehat { \pi } _ { t }$ otherwise, then follows the selected policy for all L steps. For every transition kernel $P ^ { \prime }$ , its occupancy satisfies

$$
q ^ { P ^ { \prime } , \pi _ { t } } = \lambda _ { t - 1 } q ^ { P ^ { \prime } , \pi ^ { \diamond } } + ( 1 - \lambda _ { t - 1 } ) q ^ { P ^ { \prime } , \widehat { \pi } _ { t } } .\tag{C.1}
$$

Let $I _ { t } ( x , a )$ indicate whether episode t visits $( x , a )$ . We estimate the loss using an upper occupancy bound for this mixture:

$$
\begin{array} { r l } & { u _ { t } ( x , a ) = \displaystyle \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { t - 1 } } q ^ { P ^ { \prime } , \pi _ { t } } ( x , a ) , } \\ & { \widehat { \ell } _ { t } ( x , a ) = \displaystyle \frac { \ell _ { t } ( x , a ) I _ { t } ( x , a ) } { u _ { t } ( x , a ) + \gamma } . } \end{array}\tag{C.2}
$$

Counts at unvisited state–action pairs and transitions retain their previous values. We update $\widehat { G } _ { t } , \Xi _ { t } , \mathcal { P } _ { t }$ using observations from all episodes $1 , \ldots , t .$ including baseline episodes, with the estimates and confidence bounds in (B.3)–(B.5) and $\begin{array} { r } { u = \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \end{array}$

Occupancy update We use $\widehat { \ell } _ { t }$ in the multiplicative update

$$
\widetilde { q } _ { t + 1 } ( x , a , x ^ { \prime } ) = \widehat { q } _ { t } ( x , a , x ^ { \prime } ) e ^ { - \eta \widehat { \ell } _ { t } ( x , a ) } ,\tag{C.3}
$$

for every valid transition. The projection uses the unnormalized relative entropy

$$
D ( q \| q ^ { \prime } ) = \sum _ { x , a , x ^ { \prime } } q ( x , a , x ^ { \prime } ) \log \frac { q ( x , a , x ^ { \prime } ) } { q ^ { \prime } ( x , a , x ^ { \prime } ) } - \sum _ { x , a , x ^ { \prime } } ( q ( x , a , x ^ { \prime } ) - q ^ { \prime } ( x , a , x ^ { \prime } ) ) .\tag{C.4}
$$

We use $0 \log ( 0 / z ) = 0$ for $z \geq 0$ and $z \log ( z / 0 ) = + \infty { \mathrm { ~ f o r ~ } } z > 0$ . Write $\Xi _ { t } = ( \xi _ { t } , \ldots , \xi _ { t } )$ . The updated cost and transition estimates define the optimistic projection set

$$
\mathcal { C } _ { t } = \{ q \in \Delta ( \mathcal { P } _ { t } ) : ( \widehat { G } _ { t } - \Xi _ { t } ) ^ { \top } q \leq \alpha \} .\tag{C.5}
$$

We write $\mathrm { P R O J } ( \widetilde { q } _ { t + 1 } , \widehat { G } _ { t } , \Xi _ { t } , \mathcal { P } _ { t } )$ for a minimizer of $D ( q \| \widetilde { q } _ { t + 1 } )$ over $\mathcal { C } _ { t }$ . When the projection problem attains a finite minimum, Algorithm 2 uses a minimizing occupancy measure. Otherwise, it sets $\lambda _ { t } = 1$

Algorithm 2 S-OPS   
Input: $X , A , \alpha , T , \delta , \eta , \gamma , \pi ^ { \diamond } , \beta$   
1: for $k \in \{ 0 , \ldots , L - 1 \} , ( x , a , x ^ { \prime } ) \in X _ { k } \times A \times X _ { k + 1 }$ do   
2: $N _ { 0 } ( x , a ) \gets 0 , M _ { 0 } ( x , a , x ^ { \prime } ) \gets 0$   
3: $\begin{array} { r } { \widehat { q } _ { 1 } ( x , a , x ^ { \prime } ) \gets \frac { 1 } { | X _ { k } | | A | | X _ { k + 1 } | } } \end{array}$   
4: end for   
$\pi _ { 1 }  \{ { \pi ^ { \circ } , \quad \mathrm { w } . } $ probability $\lambda _ { 0 } : = \operatorname* { m a x } _ { i \in [ m ] } \left\{ { \frac { L - \alpha _ { i } } { L - \beta _ { i } } } \right\}$   
5:   
probability $1 - \lambda _ { 0 }$   
6: for $t \in [ T ]$ do   
7: Execute $\pi _ { t }$ for one episode (Section 2) and receive feedback   
8: Build upper occupancy bounds for $k \in \{ 0 , \ldots , L - 1 \}$   
$u _ { t } ( x _ { k } , a _ { k } ) \gets \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { t - 1 } } q ^ { P ^ { \prime } , \pi _ { t } } ( x _ { k } , a _ { k } )$   
9: Build optimistic loss estimator for $( x , a ) \in X \times A { \mathrm { : } }$   
$\widehat { \ell } _ { t } ( x , a ) \gets \left\{ \begin{array} { l l } { \frac { \ell _ { t } ( x , a ) } { u _ { t } ( x , a ) + \gamma } , } & { \mathrm { i f ~ } I _ { t } ( x , a ) = 1 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} \right.$   
10: for $k \in \{ 0 , \ldots , L - 1 \}$ do   
11: $N _ { t } ( x _ { k } , a _ { k } ) \gets N _ { t - 1 } ( x _ { k } , a _ { k } ) + 1$   
12: $M _ { t } ( x _ { k } , a _ { k } , x _ { k + 1 } ) \gets M _ { t - 1 } ( x _ { k } , a _ { k } , x _ { k + 1 } ) + 1$   
13: end for   
14: Build $\mathcal { P } _ { t } , \widehat { G } _ { t } ,$ and $\Xi _ { t }$ using Appendix B.2   
15: Build unconstrained occupancy for all $( x , a , x ^ { \prime } ) \colon$   
$\widetilde { q } _ { t + 1 } ( x , a , x ^ { \prime } ) \gets \widehat { q } _ { t } ( x , a , x ^ { \prime } ) e ^ { - \widehat { \eta \ell _ { t } } ( x , a ) }$   
16: if PRO $\mathsf { J } ( \widetilde { q } _ { t + 1 } , \widehat { G } _ { t } , \Xi _ { t } , \mathcal { P } _ { t } )$ is feasible then   
17: $\widehat { q } _ { t + 1 } \gets \mathrm { P R O J } ( \widetilde { q } _ { t + 1 } , \widehat { G } _ { t } , \Xi _ { t } , \mathcal { P } _ { t } )$   
18: $\widehat { \pi } _ { t + 1 }  \widehat { \pi } ^ { q _ { t + 1 } }$   
19: Build $\widehat { u } _ { t + 1 } \in [ 0 , 1 ] ^ { | X \times A | }$ so that for all $( x , a ) \colon$   
$\widehat { u } _ { t + 1 } ( x , a ) \gets \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { t } } q ^ { P ^ { \prime } , \widehat { \pi } _ { t + 1 } } ( x , a )$   
20: Define $\overline { { m } } : = \{ i \in [ m ] : ( \widehat { g } _ { t , i } + \xi _ { t } ) ^ { \top } \widehat { u } _ { t + 1 } > \alpha _ { i } \}$   
min $\{ ( \widehat { g } _ { t , i } + \xi _ { t } ) ^ { \top } \widehat { u } _ { t + 1 } , L \} - \alpha _ { i } \ : \rceil$   
21: σ ← max   
i∈m $\Big \{ \overline { { \operatorname* { m i n } \{ ( \widehat { g } _ { t , i } + \xi _ { t } ) ^ { \top } \widehat { u } _ { t + 1 } , L \} - \beta _ { i } } } \Big \}$   
22: $\ O _ { \lambda \cdot  } \int \sigma , \mathrm { i f } \exists i \in [ m ] : ( \widehat { g } _ { t , i } + \xi _ { t } ) ^ { \top } \widehat { u } _ { t + 1 } > \alpha _ { i } ,$ λ<sub>t</sub> $\begin{array} { r } { \Big \{ 0 , \quad \mathrm { i f } \ \forall i \in [ m ] : ( \widehat { g } _ { t , i } + \xi _ { t } ) ^ { \top } \widehat { u } _ { t + 1 } \leq \alpha _ { i } } \end{array}$   
23: else   
24: $\widehat { q } _ { t + 1 } \gets$ take any $q \in \Delta ( \mathcal { P } _ { t } )$ $\lambda _ { t } \gets 1$   
25: end if   
$\pi _ { t + 1 } \gets \left\{ { \boldsymbol { \pi } } _ { t + 1 } ^ { \diamond } , { \boldsymbol { \pi } } _ { t + 1 } ^ { \diamond } \right.$ with probability $\lambda _ { t } ,$   
26:   
, with probability $1 - \lambda _ { t }$   
27: end for

Cost bounds When the projection problem attains a finite minimum, let

$$
\widehat { u } _ { t + 1 } ( x , a ) = \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { t } } q ^ { P ^ { \prime } , \widehat { \pi } _ { t + 1 } } ( x , a ) .
$$

We then choose the baseline probability for the next episode using the pessimistic cost bounds:

$$
\begin{array} { r } { w _ { t + 1 , i } = \operatorname* { m i n } \{ ( \widehat { g } _ { t , i } + \xi _ { t } ) ^ { \top } \widehat { u } _ { t + 1 } , L \} , } \end{array}\tag{C.6}
$$

$$
\lambda _ { t } = \left\{ \begin{array} { l l } { 0 , } & { w _ { t + 1 } \leq \alpha , } \\ { \displaystyle \operatorname* { m a x } _ { i : w _ { t + 1 , i } > \alpha _ { i } } \frac { w _ { t + 1 , i } - \alpha _ { i } } { w _ { t + 1 , i } - \beta _ { i } } , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{C.7}
$$

We set the auxiliary value σ in Algorithm 2 to zero when ${ \overline { { m } } } = \emptyset$ , consistently with the zero branch for $\lambda _ { t }$ . The algorithm initializes this rule with the cost upper bound L. For the analysis, also define $\begin{array} { r } { \widehat { u } _ { 1 } ( x , a ) = \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { 0 } } q ^ { P ^ { \prime } , \widehat { \pi } _ { 1 } } ( x , a ) } \end{array}$

## C.2 Zero Constraint Violation

Lemma C.1. On E<sub>T</sub>, every projection in S-OPS attains a finite minimum. Moreover, $f o r$ every $t \in [ T ]$ , we have $G ^ { \top } q _ { t } \leq \alpha , 1 - \lambda _ { t - 1 } \geq d / L$ , and $\widehat { q } _ { t } \leq ( L / d ) u _ { t }$

Proof. Fix any $q \in \Delta ( P )$ with $G ^ { \top } q \leq \alpha$ . On ${ \mathcal { E } } _ { T }$ , we have $q \in \Delta ( \mathcal { P } _ { t } )$ and

$$
\begin{array} { r } { \big ( \widehat { G } _ { t } - \Xi _ { t } \big ) ^ { \top } q \leq G ^ { \top } q \leq \alpha . } \end{array}
$$

Thus $q \in \mathcal { C } _ { t }$ at every episode.

Projection step At initialization, $D ( q \| \widehat { q } _ { 1 } ) < \infty$ because $\widehat { q } _ { 1 }$ is positive on every valid transition. Suppose $D ( q \| \widehat { q } _ { t } ) < \infty$ . The multiplicative update has the same support, so $D ( q | | \widetilde { q } _ { t + 1 } ) < \infty$ . Since $\mathcal { C } _ { t }$ is compact and the objective is lower semicontinuous and finite at $q ,$ the projection attains a finite minimum. If a minimizing occupancy r were zero at a coordinate where q is positive, the feasible perturbation $r _ { s } = ( 1 - s ) r + s q$ would satisfy

$$
D ( r _ { s } \| \widetilde { q } _ { t + 1 } ) - D ( r \| \widetilde { q } _ { t + 1 } ) = s \log s \sum _ { j : r _ { j } = 0 } q _ { j } + \mathcal { O } ( s ) < 0
$$

for suficiently small $s > 0$ , where $j$ indexes valid transitions. The negative s log s term contradicts minimality.   
Hence $D ( q | | \widehat { q } _ { t + 1 } ) < \infty$ . Induction establishes this property at every iteration for every feasible comparator.

Exploration probability For every constraint with $w _ { t , i } > \alpha _ { i }$ , the function $s \mapsto ( s - \alpha _ { i } ) / ( s - \beta _ { i } )$ is increasing on $( \alpha _ { i } , L ]$ . Hence $\lambda _ { t - 1 } \leq \operatorname* { m a x } _ { i } ( L - \alpha _ { i } ) / ( L - \beta _ { i } ) = \lambda _ { 0 }$ . Therefore

$$
1 - \lambda _ { t - 1 } \geq 1 - \lambda _ { 0 } = \operatorname* { m i n } _ { i } \frac { \alpha _ { i } - \beta _ { i } } { L - \beta _ { i } } \geq \frac { d } { L } .
$$

Choose $P _ { t } \in \mathcal P _ { t - 1 }$ such that $\widehat { q } _ { t } = q ^ { P _ { t } , \widehat { \pi } _ { t } }$ . Equation (C.1) gives

$$
\widehat { q } _ { t } = q ^ { P _ { t } , \widehat { \pi } _ { t } } \leq \frac { q ^ { P _ { t } , \pi _ { t } } } { 1 - \lambda _ { t - 1 } } \leq \frac { L } { d } q ^ { P _ { t } , \pi _ { t } } \leq \frac { L } { d } u _ { t } .\tag{C.8}
$$

Zero constraint violation In the first episode, $g _ { i } ^ { \top } q _ { 1 } \leq \lambda _ { 0 } \beta _ { i } + ( 1 - \lambda _ { 0 } ) L \leq \alpha _ { i }$ . At every later episode, the confidence bounds imply $g _ { i } ^ { \top } q ^ { P , \widehat { \pi } _ { t } } \leq w _ { t , i }$ . If $w _ { t , i } \leq \alpha _ { i }$ , then $g _ { i } ^ { \top } q _ { t } \leq \lambda _ { t - 1 } \beta _ { i } + ( 1 - \lambda _ { t - 1 } ) w _ { t , i } \leq \alpha _ { i }$ . Otherwise $\lambda _ { t - 1 } \geq ( w _ { t , i } - \alpha _ { i } ) / ( w _ { t , i } - \beta _ { i } )$ , so

$$
\begin{array} { r l } & { g _ { i } ^ { \top } q _ { t } \leq \lambda _ { t - 1 } \beta _ { i } + ( 1 - \lambda _ { t - 1 } ) w _ { t , i } } \\ & { \qquad = w _ { t , i } - \lambda _ { t - 1 } ( w _ { t , i } - \beta _ { i } ) \leq \alpha _ { i } . } \end{array}
$$

## C.3 Occupancy-Weighted Estimation Errors

We prove Lemma 3.1 by combining bounds on weighted occupancy errors and cumulative cost confidence widths.

Suppose the learner selects between Markov policies $\pi _ { t , 0 } , \pi _ { t , 1 }$ with probabilities $p _ { t , 0 } , p _ { t , 1 }$ , all $\mathcal { F } _ { t - 1 ^ { - } } \mathrm { m e a s u r a b l e }$ The resulting occupancy measure is $\begin{array} { r } { q _ { t } = \sum _ { j = 0 } ^ { 1 } p _ { t , j } q ^ { P , \pi _ { t , j } } } \end{array}$ . Construct the $\mathrm { S - O P S }$ confidence sets using observations from all episodes and set $u _ { t , j } ( x , a ) = \mathrm { m a x } _ { P ^ { \prime } \in \mathcal { P } _ { t - 1 } } q ^ { P ^ { \prime } , \pi _ { t , j } } ( x , a )$

Lemma C.2 (Weighted occupancy error bounds). With probability at least $1 - 3 \delta / 8 , \mathcal { E } _ { T }$ holds and

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \sum _ { j = 0 } ^ { 1 } p _ { t , j } ( \| u _ { t , j } - q ^ { P , \pi _ { t , j } } \| _ { 1 } + \| q ^ { P _ { t , j } , \pi _ { t , j } } - q ^ { P , \pi _ { t , j } } \| _ { 1 } ) } } \\ & { } & { \leq \mathcal { O } \Bigg ( L | X | \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + | X | ^ { 3 } | A | \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \Bigg ) } \end{array}
$$

for all $P _ { t , j } \in \mathcal { P } _ { t - 1 }$ . On the same event, the left-hand side is also at most $\begin{array} { r } { \mathcal { O } \bigg ( | X | ^ { 2 } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \bigg ) } \end{array}$

Proof. We use the occupancy expansion of Jin et al. (2020, Appendix B.2) and retain both execution weights and squared-width terms.

Transition errors On $\mathcal { E } _ { T } , \mathrm { i f } \ \xi _ { t - 1 } ( x , a ) < 1$ , then $\begin{array} { r } { N _ { t - 1 } ( x , a ) > 4 \log { \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } } \end{array}$ . Equations (B.4)–(B.5) and $N _ { t - 1 } ( x , a ) - 1 \geq N _ { t - 1 } ( x , a ) / 2$ give

$$
\widehat { P } _ { t - 1 } ( x ^ { \prime } \mid x , a ) \leq P ( x ^ { \prime } \mid x , a ) + \sqrt { 2 \widehat { P } _ { t - 1 } ( x ^ { \prime } \mid x , a ) } \xi _ { t - 1 } ( x , a ) + \frac { 7 } { 3 } \xi _ { t - 1 } ( x , a ) ^ { 2 } .
$$

Completing the square yields $\sqrt { \widehat { P } _ { t - 1 } ( x ^ { \prime } \mid x , a ) } \leq \sqrt { P ( x ^ { \prime } \mid x , a ) } + 3 \xi _ { t - 1 } ( x , a )$ . Both $P ^ { \prime }$ and $P$ satisfy the confidence bounds. Substituting the preceding estimate into (B.5) and applying the triangle inequality gives, for every $P ^ { \prime } \in \mathcal { P } _ { t - 1 }$

$$
| P ^ { \prime } ( x ^ { \prime } \mid x , a ) - P ( x ^ { \prime } \mid x , a ) | \leq 1 6 \left( \sqrt { P ( x ^ { \prime } \mid x , a ) } \xi _ { t - 1 } ( x , a ) + \xi _ { t - 1 } ( x , a ) ^ { 2 } \right) .
$$

The same bound holds when $\xi _ { t - 1 } ( x , a ) = 1$ , since transition probabilities lie in $[ 0 , 1 ]$ . Let $b _ { t - 1 }$ be defined by (B.7) for $\mathcal { P } _ { t - 1 }$ . For $x \in X _ { h }$ , (B.8) also gives

$$
\| P ^ { \prime } ( \cdot \mid x , a ) - P ( \cdot \mid x , a ) \| _ { 1 } \leq b _ { t - 1 } ( x , a ) \leq \mathcal { O } \Big ( \sqrt { | X _ { h + 1 } | } \xi _ { t - 1 } ( x , a ) \Big ) .\tag{C.9}
$$

Fix $t , j ,$ and write $\pi = \pi _ { t , j } , q = q ^ { P , \pi }$ , and $\xi = \xi _ { t - 1 }$ in this part of the proof. Let $q ^ { P ^ { \prime } , \pi } ( y \mid x ^ { \prime } )$ denote the probability of reaching y from $x ^ { \prime }$ under $( P ^ { \prime } , \pi )$ , where y is in the same or a later layer. Define $\varphi ^ { P , \pi } ( z , b \mid x ^ { \prime } )$ analogously for a state–action pair. For $y \in X _ { k }$ with $1 \leq k < L$ and $P ^ { \prime } \in \mathcal { P } _ { t - 1 }$ , telescoping the transition products gives

$$
q ^ { P ^ { \prime } , \pi } ( y ) - q ( y ) = \sum _ { h = 0 } ^ { k - 1 } \sum _ { \stackrel { x \in X _ { h } , a } { x ^ { \prime } \in X _ { h + 1 } } } q ( x , a ) \big ( P ^ { \prime } ( x ^ { \prime } \mid x , a ) - P ( x ^ { \prime } \mid x , a ) \big ) q ^ { P ^ { \prime } , \pi } ( y \mid x ^ { \prime } ) .
$$

Take absolute values and apply the pointwise transition-error bound. Since $q ^ { P ^ { \prime } , \pi } ( y \mid x ^ { \prime } ) \leq 1$ , the terms involving $\xi ( x , a ) ^ { 2 }$ contribute at most

$$
1 6 \sum _ { h = 0 } ^ { k - 1 } | X _ { h + 1 } | \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) ^ { 2 } .
$$

In the remaining terms, use

$$
q ^ { P ^ { \prime } , \pi } ( y \mid x ^ { \prime } ) \leq q ^ { P , \pi } ( y \mid x ^ { \prime } ) + \left| q ^ { P ^ { \prime } , \pi } ( y \mid x ^ { \prime } ) - q ^ { P , \pi } ( y \mid x ^ { \prime } ) \right| .
$$

For $x ^ { \prime } \in X _ { h + 1 }$ , a second expansion, now starting at $x ^ { \prime } ,$ gives

$$
\left. q ^ { P ^ { \prime } , \pi } ( y \mid x ^ { \prime } ) - q ^ { P , \pi } ( y \mid x ^ { \prime } ) \right. \leq \sum _ { r = h + 1 } ^ { k - 1 } \sum _ { z \in X _ { r } , b } q ^ { P , \pi } ( z , b \mid x ^ { \prime } ) \| P ^ { \prime } ( \cdot \mid z , b ) - P ( \cdot \mid z , b ) \| _ { 1 }
$$

$$
\leq \mathcal { O } \left( \sum _ { r = h + 1 } ^ { k - 1 } \sqrt { | X _ { r + 1 } | } \sum _ { z \in X _ { r } , b } q ^ { P , \pi } ( z , b \mid x ^ { \prime } ) \xi ( z , b ) \right) .
$$

For $0 \leq h < r < k$ , Cauchy–Schwarz bounds the sum of products of confidence widths by

$$
\begin{array} { r l } & { \displaystyle \sum _ { x \in X _ { h } , a , \ x ^ { \prime } \in X _ { h + 1 } } q ( x , a ) \sqrt { P ( x ^ { \prime } \mid x , a ) } \xi ( x , a ) q ^ { P , \pi } ( z , b \mid x ^ { \prime } ) \xi ( z , b ) } \\ & { \displaystyle \leq \sqrt { | X _ { h + 1 } | \left( \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) ^ { 2 } \right) \left( \sum _ { z \in X _ { r } , b } q ( z , b ) \xi ( z , b ) ^ { 2 } \right) } . } \end{array}
$$

We use weights $q ( x , a ) q ^ { P , \pi } ( z , b \mid x ^ { \prime } )$ and the identities

$$
\sum _ { z \in X _ { r } , b } q ^ { P , \pi } ( z , b \mid x ^ { \prime } ) = 1 , \qquad \sum _ { x \in X _ { h } , a \atop x ^ { \prime } \in X _ { h + 1 } } q ( x , a ) P ( x ^ { \prime } \mid x , a ) q ^ { P , \pi } ( z , b \mid x ^ { \prime } ) = q ( z , b ) .
$$

These estimates hold uniformly over $P ^ { \prime } \in \mathcal { P } _ { t - 1 } .$ , so the maximizing kernel may be chosen separately for each target state y. For the terms containing $q ^ { P , \pi } ( y \mid x ^ { \prime } )$ , we first sum over $y \in X _ { k }$ , using $\begin{array} { r } { \sum _ { y \in X _ { k } } { q ^ { P , \pi } ( y \mid x ^ { \prime } ) } = 1 } \end{array}$ Summing next over $k$ and applying $\begin{array} { r } { \sum _ { x ^ { \prime } \in X _ { h + 1 } } \sqrt { P ( x ^ { \prime } \mid x , a ) } \leq \sqrt { | X _ { h + 1 } | } } \end{array}$ gives

$$
\mathcal { O } \left( L \sum _ { h = 0 } ^ { L - 1 } \sqrt { | X _ { h + 1 } | } \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) \right) .
$$

For the squared-width terms, summing over the target states gives

$$
\begin{array} { r l r } {  { 1 6 \sum _ { k = 1 } ^ { L - 1 } | X _ { k } | \sum _ { h = 0 } ^ { k - 1 } | X _ { h + 1 } | \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) ^ { 2 } \le 1 6 | X | \sum _ { h = 0 } ^ { L - 1 } | X _ { h + 1 } | \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) ^ { 2 } } } \\ & { } & { \le 1 6 | X | ^ { 2 } \sum _ { x , a } q ( x , a ) \xi ( x , a ) ^ { 2 } . } \end{array}
$$

Each pair of layers in the second expansion contributes to at most $| X |$ target states. The total contribution of these pairs is bounded by a universal constant times

$$
\begin{array} { r l } & { \displaystyle | X | \sum _ { 0 \le h < r < L } \sqrt { | X _ { h + 1 } | \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) ^ { 2 } } \sqrt { | X _ { r + 1 } | \sum _ { z \in X _ { r } , b } q ( z , b ) \xi ( z , b ) ^ { 2 } } } \\ & { \quad \le \frac { | X | } { 2 } \left( \displaystyle \sum _ { h = 0 } ^ { L - 1 } \sqrt { | X _ { h + 1 } | \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) ^ { 2 } } \right) ^ { 2 } \le \frac { | X | ^ { 2 } } { 2 } \sum _ { x , a } q ( x , a ) \xi ( x , a ) ^ { 2 } . } \end{array}
$$

The last inequality follows from Cauchy–Schwarz and $\begin{array} { r } { \sum _ { h = 0 } ^ { L - 1 } | X _ { h + 1 } | \leq | X | } \end{array}$ . The initial-state error is zero. Since $q ^ { P ^ { \prime } , \pi } ( y , a ) = \pi ( a \mid y ) q ^ { P ^ { \prime } , \pi } ( y )$ , summing over actions gives the state marginal. This proves the required bound on $\| u _ { t , j } - q \| _ { 1 }$ . Lemma B.3 and (C.9) give

$$
\| q ^ { P _ { t , j } , \pi } - q \| _ { 1 } \leq L b _ { t - 1 } ^ { \top } q \leq \mathcal { O } \left( L \sum _ { h = 0 } ^ { L - 1 } \sqrt { | X _ { h + 1 } | } \sum _ { x \in X _ { h } , a } q ( x , a ) \xi ( x , a ) \right) .\tag{C.10}
$$

Weighted occupancy errors We weight both occupancy error bounds by $p _ { t , j }$ and sum over $j .$ . Using $\begin{array} { r } { \sum _ { j } p _ { t , j } q ^ { P , \pi _ { t , j } } = q _ { t } } \end{array}$ , we obtain

$$
\begin{array} { r l r } {  { \sum _ { j = 0 } ^ { 1 } p _ { t , j } ( \| u _ { t , j } - q ^ { P , \pi _ { t , j } } \| _ { 1 } + \| q ^ { P _ { t , j } , \pi _ { t , j } } - q ^ { P , \pi _ { t , j } } \| _ { 1 } ) } } \\ & { } & { \leq \mathcal { O } ( L \sum _ { k = 0 } ^ { L - 1 } \sqrt { | X _ { k + 1 } | } \sum _ { x \in X _ { k , a } } q _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) + | X | ^ { 2 } \sum _ { x , a } q _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } ) . } \end{array}\tag{C.11}
$$

Equations (B.4) and (B.9) imply, on every observation sequence,

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \sum _ { k = 0 } ^ { L - 1 } \sqrt { | X _ { k + 1 } | } \sum _ { x \in X _ { k } , a } I _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) \le 6 \sqrt { \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \sum _ { k = 0 } ^ { L - 1 } \sqrt { | X _ { k + 1 } | } \sum _ { x \in X _ { k } , a } \sqrt { N _ { T } ( x , a ) } } } } \\ & { } & { \le 6 | X | \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } . } \end{array}
$$

Here $\begin{array} { r } { \sum _ { x \in X _ { k } , a } N _ { T } ( x , a ) = T } \end{array}$ and $\begin{array} { r } { \sum _ { k } \sqrt { | X _ { k } | | X _ { k + 1 } | } \le | X | } \end{array}$ . Since $\mathbb { E } [ I _ { t } \mid \mathcal { F } _ { t - 1 } ] = q _ { t }$ , replacing $I _ { t }$ by $q _ { t }$ adds a sum of martingale diferences. Each has absolute value at most $\begin{array} { r } { \sum _ { k } \sqrt { | X _ { k + 1 } | } \le | X | } \end{array}$ |. Azuma–Hoefding thus gives, with probability at least $1 - \delta / 8$

$$
\sum _ { t = 1 } ^ { T } \sum _ { k = 0 } ^ { L - 1 } \sqrt { | X _ { k + 1 } | } \sum _ { x \in X _ { k } , a } q _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) \leq \mathcal { O } \left( | X | \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \right) .\tag{C.12}
$$

On the intersection of this concentration event and ${ { \mathcal { E } } _ { T } }$ , combining (C.10) and (C.12) gives

$$
\sum _ { t = 1 } ^ { T } \sum _ { j = 0 } ^ { 1 } p _ { t , j } \| q ^ { P _ { t , j } , \pi _ { t , j } } - q ^ { P , \pi _ { t , j } } \| _ { 1 } \leq \mathcal { O } \left( L | X | \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \right) .\tag{C.13}
$$

For each $k ,$ the sum $\begin{array} { r } { \sum _ { x \in X _ { k } , a } I _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } } \end{array}$ lies in $[ 0 , 1 ]$ , since exactly one pair in $X _ { k } \times A$ is visited per episode. Using $e ^ { - z } \leq 1 - ( 1 - e ^ { - 1 } ) z { \mathrm { ~ f o r ~ } } z \in [ 0 , 1 ]$ gives

$$
\mathbb { E } \left[ \exp \left\{ ( 1 - e ^ { - 1 } ) \sum _ { x \in X _ { k } , a } q _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } - \sum _ { x \in X _ { k } , a } I _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } \right\} \Bigg | \mathcal { F } _ { t - 1 } \right] \leq 1 .
$$

We iterate this conditional exponential bound over t and apply Markov’s inequality with failure probability $\delta / ( 8 L )$ for each layer. A union bound gives

$$
\sum _ { t = 1 } ^ { T } \sum _ { x , a } q _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } \leq 2 \sum _ { t = 1 } ^ { T } \sum _ { x , a } I _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } + 2 L \log \frac { 8 L } { \delta }
$$

with probability at least $1 - \delta / 8$ , where $( 1 - e ^ { - 1 } ) ^ { - 1 } < 2$ . Indexing visits to a fixed $( x , a )$ in (B.4) gives

$$
\sum _ { t = 1 } ^ { T } I _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } \leq 4 \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \sum _ { r = 0 } ^ { N _ { T } ( x , a ) - 1 } \frac { 1 } { \operatorname* { m a x } \{ 1 , r \} } \leq 4 \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \left( 2 + \log T \right) .
$$

Summing over $( x , a )$ and using $L < | X |$ and log T, log $\begin{array} { r } { ( 8 L / \delta ) \leq \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \end{array}$ gives, on the same event,

$$
\sum _ { t = 1 } ^ { T } \sum _ { x , a } q _ { t } ( x , a ) \xi _ { t - 1 } ( x , a ) ^ { 2 } \leq \mathcal { O } \bigg ( | X | | A | \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \bigg ) .\tag{C.14}
$$

Combining the bounds Summing (C.11) over t and applying (C.12) and (C.14) proves the first bound. For the second bound, Lemma B.3 bounds the same sum by $( | X | + L ) \textstyle \sum _ { t } b _ { t - 1 } ^ { \top } q _ { t }$ . We then apply (C.9) and (C.12). By Lemma B.2 and a union bound, ${ \mathcal { E } } _ { T }$ and both concentration bounds hold together with probability at least $1 - 3 \delta / 8$ □

We next bound the cumulative cost confidence widths.

Lemma C.3 (Cumulative confidence widths). With probability at least $1 - \delta / 8$

$$
\sum _ { t = 1 } ^ { T } \xi _ { t - 1 } ^ { \top } q _ { t } \leq \mathcal { O } \left( \sqrt { L | X | | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \right) .
$$

Proof. Because $\mathbb { E } _ { t } [ I _ { t } ] = q _ { t }$ , the diferences $\xi _ { t - 1 } ^ { \top } ( q _ { t } - I _ { t } )$ have conditional mean zero and absolute value at most $L .$ . Azuma–Hoefding gives

$$
\sum _ { t } \xi _ { t - 1 } ^ { \top } q _ { t } \leq \sum _ { t } \xi _ { t - 1 } ^ { \top } I _ { t } + L \sqrt { 2 T \log ( 8 / \delta ) } .
$$

Equations (B.4) and (B.9) give $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \xi _ { t - 1 } ( x , a ) I _ { t } ( x , a ) \leq 6 \sqrt { N _ { T } ( x , a ) \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } } \end{array}$ for each $( x , a )$ . We sum over $( x , a )$ and apply Cauchy–Schwarz with $\begin{array} { r } { \sum _ { x , a } N _ { T } ( x , a ) \dot { = } L T } \end{array}$ to obtain

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \xi _ { t - 1 } ^ { \top } q _ { t } \le 6 \sqrt { L | X | | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + L \sqrt { 2 T \log ( 8 / \delta ) } } } \\ & { } & { \le \mathcal { O } ( \sqrt { L | X | | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } ) , } \end{array}
$$

where the last line uses $L < | X | , | A | \geq 1$ , and $\begin{array} { r } { \log ( 8 / \delta ) \leq \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \end{array}$

Proof of Lemma 3.1. Intersect the events in Lemmas C.2 and C.3. Their failure probabilities sum to $\delta / 2$ and ${ \mathcal { E } } _ { T }$ holds on the intersection. Apply Lemma C.2 with policies $\pi ^ { \diamond } , \widehat { \pi } _ { t }$ and probabilities $\lambda _ { t - 1 } , 1 - \lambda _ { t - 1 }$ . By Lemma B.1, we can choose $P _ { t } \in \mathscr { P } _ { t - 1 }$ with $\widehat { q } _ { t } = q ^ { P _ { t } , \widehat { \pi } _ { t } }$

Constraint cost bounds For $t \geq 2$ , the projection constraint and $\| \widehat { g } _ { t - 1 , i } - \xi _ { t - 1 } \| _ { \infty } \leq 1$ give, for every $i \in [ m ]$

$$
\begin{array} { r l } & { w _ { t , i } - \alpha _ { i } \leq \left( \widehat { g } _ { t - 1 , i } - \xi _ { t - 1 } \right) ^ { \top } ( \widehat { u } _ { t } - \widehat { q } _ { t } ) + 2 \xi _ { t - 1 } ^ { \top } \widehat { u } _ { t } } \\ & { \qquad \leq \| \widehat { u } _ { t } - q ^ { P , \widehat { \pi } _ { t } } \| _ { 1 } + \| q ^ { P , \widehat { \pi } _ { t } } - \widehat { q } _ { t } \| _ { 1 } + 2 \xi _ { t - 1 } ^ { \top } \widehat { u } _ { t } . } \end{array}
$$

The right-hand side is nonnegative and independent of $i ,$ so it also bounds $e _ { t }$ . Since $0 \leq \xi _ { t - 1 } \leq 1$ , (C.1) implies

$$
\begin{array} { r } { ( 1 - \lambda _ { t - 1 } ) \xi _ { t - 1 } ^ { \top } \widehat { u } _ { t } \leq \xi _ { t - 1 } ^ { \top } q _ { t } + ( 1 - \lambda _ { t - 1 } ) \| \widehat { u } _ { t } - q ^ { P , \widehat { \pi } _ { t } } \| _ { 1 } . } \end{array}
$$

Multiply the bound on $e _ { t }$ by $1 - \lambda _ { t - 1 }$ and sum over $t = 2 , \ldots , T$ . Lemma C.2 bounds both weighted occupancy errors, and Lemma C.3 bounds the cumulative confidence widths.

Loss-estimator occupancy bounds Set $\begin{array} { r } { \boldsymbol u _ { t } ^ { \pi ^ { \diamond } } ( \boldsymbol x , \boldsymbol a ) = \operatorname* { m a x } _ { P ^ { \prime } \in \mathcal P _ { t - 1 } } \boldsymbol q ^ { P ^ { \prime } , \pi ^ { \diamond } } ( \boldsymbol x , \boldsymbol a ) } \end{array}$ . On $\mathcal { E } _ { T } , q _ { t } \leq u _ { t }$ . Bounding the maximum of the mixture by the sum of its componentwise maxima gives

$$
\begin{array} { r } { \| u _ { t } - q _ { t } \| _ { 1 } \leq \lambda _ { t - 1 } \| u _ { t } ^ { \pi ^ { \diamond } } - q ^ { P , \pi ^ { \diamond } } \| _ { 1 } + ( 1 - \lambda _ { t - 1 } ) \| \widehat { u } _ { t } - q ^ { P , \widehat { \pi } _ { t } } \| _ { 1 } . } \end{array}
$$

Summing over episodes and applying Lemma C.2 bounds the cumulative occupancy error. Combining the two estimates gives

$$
\sum _ { t = 2 } ^ { T } ( 1 - \lambda _ { t - 1 } ) e _ { t } + \sum _ { t = 1 } ^ { T } \lVert u _ { t } - q _ { t } \rVert _ { 1 } \leq \mathcal { O } \left( L | X | \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + | X | ^ { 3 } | A | \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \right) ,
$$

on the same event of probability at least $1 - \delta / 2$ . The second conclusion of Lemma C.2 also bounds this sum by $\begin{array} { r } { \mathcal { O } ( | X | ^ { 2 } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } ) } \end{array}$ on the same event. □

## C.4 Proof of Theorem 1

We first establish two concentration bounds for the loss estimates using the implicit-exploration argument of Jin et al. (2020).

Lemma C.4 (Loss estimation bounds). With probability at least $1 - 3 \delta / 8 , \mathcal { E } _ { T }$ holds and

$$
\sum _ { t , x , a } \widehat { \ell } _ { t } ( x , a ) \leq | X | | A | T + \frac { L } { \gamma } \log \frac { 8 L } { \delta } ,
$$

$$
\sum _ { t = 1 } ^ { T } ( \widehat { \ell } _ { t } - \ell _ { t } ) ^ { \top } q \leq \frac { L } { \gamma } \log \frac { 8 | X | | A | } { \delta }
$$

for every $q \geq 0$ with $\begin{array} { r } { \sum _ { x , a } q ( x , a ) = L } \end{array}$

Proof. Set $s _ { t } ( x , a ) = \operatorname* { m a x } \{ u _ { t } ( x , a ) , q _ { t } ( x , a ) \}$ and $\bar { \ell } _ { t } ( x , a ) = \ell _ { t } ( x , a ) I _ { t } ( x , a ) / ( s _ { t } ( x , a ) + \gamma )$ . Then $s _ { t } ~ \geq ~ q _ { t }$ coordinatewise and $\overline { { \ell } } _ { t } ~ = ~ \widehat { \ell } _ { t }$ on ${ \mathcal { E } } _ { T }$ . For $( x , a )$ , write $\ell = \ell _ { t } ( x , a ) , p = q _ { t } ( x , a )$ , and $s = s _ { t } ( x , a )$ . Set $z = \gamma \ell / ( s + \gamma ) \in [ 0 , 1 ]$ . Since $e ^ { z } \leq 1 + z + z ^ { 2 }$ and $p \leq s .$

$$
p ( e ^ { z } - 1 ) \leq p ( z + z ^ { 2 } ) \leq \gamma \ell \frac { s ( s + 2 \gamma ) } { ( s + \gamma ) ^ { 2 } } \leq \gamma \ell .\tag{C.15}
$$

The visit indicator is Bernoulli with conditional probability $p ,$ so

$$
\begin{array} { r } { \mathbb { E } _ { t } \big [ e ^ { \gamma ( \overline { \ell } _ { t } ( x , a ) - \ell _ { t } ( x , a ) ) } \big ] \leq e ^ { - \gamma \ell _ { t } ( x , a ) } ( 1 + \gamma \ell _ { t } ( x , a ) ) \leq 1 . } \end{array}
$$

By the tower property, the same exponential bound holds conditional on $\mathcal { F } _ { t - 1 }$ . Iterating and applying Markov’s inequality gives, with failure probability at most $\delta / ( 8 | X | | A | )$ for each coordinate,

$$
\sum _ { t = 1 } ^ { T } ( \overline { { \ell } } _ { t } ( x , a ) - \ell _ { t } ( x , a ) ) \leq \frac { 1 } { \gamma } \log \frac { 8 | X | | A | } { \delta } .
$$

A union bound over coordinates, followed by weighting each coordinate by $q ( x , a )$ and summing, proves the second bound for every nonnegative q with total mass L. For the first bound, use $\begin{array} { r } { \sum _ { x \in X _ { k } , a } I _ { t } ( x , a ) = 1 } \end{array}$ for each $k = 0 , \ldots , L - 1$ . Summing (C.15) over the possible outcomes at step k gives

$$
\mathbb { E } _ { t } \exp \left\{ \gamma \sum _ { \boldsymbol { x } \in X _ { k } , a } \left( \bar { \ell } _ { t } ( \boldsymbol { x } , a ) - \ell _ { t } ( \boldsymbol { x } , a ) \right) \right\} \le 1 .
$$

We iterate this exponential bound using the tower property and apply Markov’s inequality with failure probability $\delta / ( 8 L )$ for each k. Taking a union bound over the layers and summing over k gives

$$
\sum _ { t , x , a } ( \bar { \ell } _ { t } ( x , a ) - \ell _ { t } ( x , a ) ) \leq \frac { L } { \gamma } \log \frac { 8 L } { \delta }
$$

with failure probability at most $\delta / 8$ . Adding $\begin{array} { r } { \sum _ { t , x , a } \ell _ { t } ( x , a ) \leq | X | | A | T } \end{array}$ proves the first bound for $\overline { { \ell } } _ { t } .$ . By a union bound, both estimates and ${ \mathcal { E } } _ { T }$ hold jointly with probability at least $1 - 3 \delta / 8$ . On this event, $\overline { { \ell } } _ { t } = \widehat { \ell } _ { t } .$ which proves the lemma. □

Proof of Theorem 1. The constraint guarantee follows from Lemma C.1. We prove the regret bound on the intersection of ${ \mathcal { E } } _ { T }$ , the events in Lemmas C.2, C.4, and C.3, and the martingale event used to bound $R ^ { ( 4 ) }$

Fix any feasible occupancy measure q. Select $P _ { t } \in \mathcal P _ { t - 1 }$ so that $\widehat { q } _ { t } = q ^ { P _ { t } , \widehat { \pi } _ { t } }$ . Adding and subtracting the estimated losses and occupancies gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ^ { \top } ( q _ { t } - q ) = \underbrace { \sum _ { t } \ell _ { t } ^ { \top } ( q _ { t } - q ^ { P _ { t } , \pi _ { t } } ) } _ { R ^ { ( 1 ) } } + \underbrace { \sum _ { t } \widehat { \ell } _ { t } ^ { \top } ( \widehat { q } _ { t } - q ) } _ { R ^ { ( 2 ) } } + \underbrace { \sum _ { t } \ell _ { t } ^ { \top } ( q ^ { P _ { t } , \pi _ { t } } - \widehat { q } _ { t } ) } _ { R ^ { ( 3 ) } } } \\ & { \quad \quad \quad + \underbrace { \sum _ { t } ( \ell _ { t } - \widehat { \ell } _ { t } ) ^ { \top } \widehat { q } _ { t } } _ { R ^ { ( 4 ) } } + \underbrace { \sum _ { t } ( \widehat { \ell } _ { t } - \ell _ { t } ) ^ { \top } q } _ { R ^ { ( 5 ) } } . } \end{array}\tag{C.16}
$$

The five terms correspond to transition estimation error, online mirror descent, regret from baseline use, and loss-estimation errors for the policy and comparator.

Transition error Equation (C.1) and $\| \ell _ { t } \| _ { \infty } \leq 1$ imply

$$
\begin{array} { r l } {  { R ^ { ( 1 ) } \leq \sum _ { t } \| q _ { t } - q ^ { P _ { t } , \pi _ { t } } \| _ { 1 } } } \\ & { \leq \sum _ { t } \lambda _ { t - 1 } \| q ^ { P , \pi ^ { \diamond } } - q ^ { P _ { t } , \pi ^ { \diamond } } \| _ { 1 } + \sum _ { t } ( 1 - \lambda _ { t - 1 } ) \| q ^ { P , \widehat { \pi _ { t } } } - \widehat { q } _ { t } \| _ { 1 } } \\ & { \leq \mathcal { O } \Bigg ( L | X | \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \Bigg ) , } \end{array}\tag{C.17}
$$

where the last inequality follows from (C.13). On the same event, Lemma 3.1 also gives

$$
\sum _ { t } \| u _ { t } - q _ { t } \| _ { 1 } \leq \mathcal { O } \Bigg ( L | X | \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + | X | ^ { 3 } | A | \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \Bigg ) .\tag{C.18}
$$

Online mirror descent Let $r = \widehat { q } _ { t + 1 }$ and $y = \widetilde { q } _ { t + 1 }$ . By Lemma C.1, both $D ( q \| r )$ and $D ( q \| y )$ are finite. Optimality of r along the feasible segment $( 1 - s ) r + s q$ gives

$$
D ( q \| y ) - D ( q \| r ) - D ( r \| y ) = \sum _ { j } ( q _ { j } - r _ { j } ) \log \frac { r _ { j } } { y _ { j } } \ge 0 .
$$

Terms with $q _ { j } = r _ { j } = 0$ contribute zero. Combining this projection inequality with $e ^ { - x } \leq 1 - x + x ^ { 2 } / 2$ for $x \geq 0$ yields

$$
\begin{array} { r l } & { \eta \widehat { \ell } _ { t } ^ { \top } ( \widehat { q } _ { t } - q ) = D ( q \| \widehat { q } _ { t } ) - D ( q \| \widetilde { q } _ { t + 1 } ) + D ( \widehat { q } _ { t } \| \widetilde { q } _ { t + 1 } ) } \\ & { \qquad \leq D ( q \| \widehat { q } _ { t } ) - D ( q \| \widehat { q } _ { t + 1 } ) + \displaystyle \frac { \eta ^ { 2 } } { 2 } \sum _ { x , a } \widehat { q } _ { t } ( x , a ) \widehat { \ell } _ { t } ( x , a ) ^ { 2 } . } \end{array}
$$

We sum over t and telescope the divergence terms. The uniform initialization then gives

$$
R ^ { ( 2 ) } \leq \frac { L \log ( | X | ^ { 2 } | A | ) } { \eta } + \frac { \eta } { 2 } \sum _ { t , x , a } \widehat { q } _ { t } ( x , a ) \widehat { \ell } _ { t } ( x , a ) ^ { 2 } .\tag{C.19}
$$

The initial divergence bound follows from $q ( x , a , x ^ { \prime } ) \leq 1$ and $1 / \widehat { q } _ { 1 } ( x , a , x ^ { \prime } ) \leq | X | ^ { 2 } | A |$ . By (C.8) and $\ell _ { t } \leq 1$

$$
\widehat { q } _ { t } ( x , a ) \widehat { \ell } _ { t } ( x , a ) ^ { 2 } \leq \frac { \widehat { q } _ { t } ( x , a ) } { u _ { t } ( x , a ) + \gamma } \widehat { \ell } _ { t } ( x , a ) \leq \frac { L } { d } \widehat { \ell } _ { t } ( x , a ) .
$$

Applying the bound on the total estimated loss in Lemma C.4 gives

$$
R ^ { ( 2 ) } \leq \frac { L \log ( | X | ^ { 2 } | A | ) } { \eta } + \frac { L \eta } { d } | X | | A | T + \frac { L ^ { 2 } \eta } { d \gamma } \log \frac { 8 L } { \delta } .\tag{C.20}
$$

Regret from baseline use Equation (C.1) gives $\begin{array} { r } { R ^ { ( 3 ) } \leq L \sum _ { t } \lambda _ { t - 1 } } \end{array}$ . We bound the first episode by L. For $t \geq 2$ with $\lambda _ { t - 1 } > 0$ , choose an index $i _ { t }$ attaining the maximum in (C.7). Then

$$
\begin{array} { r l } & { \lambda _ { t - 1 } = \frac { w _ { t , i _ { t } } - \alpha _ { i _ { t } } } { w _ { t , i _ { t } } - \beta _ { i _ { t } } } } \\ & { \qquad = \frac { \left( 1 - \lambda _ { t - 1 } \right) \left( w _ { t , i _ { t } } - \alpha _ { i _ { t } } \right) } { \alpha _ { i _ { t } } - \beta _ { i _ { t } } } } \\ & { \qquad \le \frac { 1 - \lambda _ { t - 1 } } { d } ( w _ { t , i _ { t } } - \alpha _ { i _ { t } } ) . } \end{array}\tag{C.21}
$$

Since $[ w _ { t , i _ { t } } - \alpha _ { i _ { t } } ] _ { + } \leq e _ { t }$ , Lemma 3.1 and (C.21) yield

$$
\sum _ { t } \lambda _ { t - 1 } \leq 1 + \mathcal { O } \Bigg ( \frac { L | X | } { d } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + \frac { | X | ^ { 3 } | A | } { d } \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \Bigg ) ,
$$

and hence

$$
R ^ { ( 3 ) } \leq L + { \mathcal { O } } { \left( \frac { L ^ { 2 } | X | } { d } { \sqrt { | A | T \log { \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } } } + \frac { L | X | ^ { 3 } | A | } { d } \log ^ { 2 } { \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \right) } .\tag{C.22}
$$

The weighting by execution probabilities follows Bollini et al. (2026, Appendix E, Lemma E.2).

Policy loss-estimation error We decompose the policy loss-estimation error into a martingale fluctuation and a conditional bias:

$$
R ^ { ( 4 ) } = \sum _ { t } ( \mathbb { E } _ { t } [ \widehat { \ell } _ { t } ] - \widehat { \ell } _ { t } ) ^ { \top } \widehat { q } _ { t } + \sum _ { t } ( \ell _ { t } - \mathbb { E } _ { t } [ \widehat { \ell } _ { t } ] ) ^ { \top } \widehat { q } _ { t } .\tag{C.23}
$$

Define $\begin{array} { r } { h _ { t } = \prod _ { s = 1 } ^ { t - 1 } \mathbf { 1 } \{ \operatorname* { i n f } _ { r \in \mathcal { C } _ { s } } D ( r \Vert \widetilde { q } _ { s + 1 } ) < \infty \} } \end{array}$ , with $h _ { 1 } = 1$ . The indicator $h _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable and equals one on ${ \mathcal { E } } _ { T }$ by Lemma C.1. When $h _ { t } = 1$ , all preceding projections attain finite minima, so the mixing rule and the definition of $u _ { t }$ give $\widehat { q } _ { t } \leq ( L / d ) u _ { t }$ . Hence

$$
0 \leq \widehat { \ell } _ { t } ^ { \top } \widehat { q } _ { t } \leq \frac { L } { d } \sum _ { x , a } \ell _ { t } ( x , a ) I _ { t } ( x , a ) \leq \frac { L ^ { 2 } } { d } .
$$

By the tower property, $h _ { t } ( \mathbb { E } _ { t } [ \widehat { \ell } _ { t } ] - \widehat { \ell } _ { t } ) ^ { \top } \widehat { q } _ { t }$ is a martingale diference sequence with respect to $( \mathcal { F } _ { t } )$ , bounded in absolute value by $L ^ { 2 } / d .$ . Azuma–Hoefding gives an event of probability at least $1 - \delta / 8$ . On its intersection with ${ \mathcal { E } } _ { T }$ , we have

$$
\sum _ { t } ( \mathbb { E } _ { t } [ \widehat { \ell } _ { t } ] - \widehat { \ell } _ { t } ) ^ { \top } \widehat { q } _ { t } \leq \frac { L ^ { 2 } } { d } \sqrt { 2 T \log ( 8 / \delta ) } .\tag{C.24}
$$

For the second sum, $\mathbb { E } _ { t } [ \widehat { \ell } _ { t } ( x , a ) ] = q _ { t } ( x , a ) \ell _ { t } ( x , a ) / ( u _ { t } ( x , a ) + \gamma )$ . On E<sub>T</sub> , $q _ { t } \leq u _ { t }$ , so

$$
\begin{array} { r l } & { \displaystyle \sum _ { t } ( \ell _ { t } - \mathbb { E } _ { t } [ \widehat { \ell } _ { t } ] ) ^ { \top } \widehat { q } _ { t } = \displaystyle \sum _ { t , x , a } \frac { \widehat { q } _ { t } ( x , a ) \ell _ { t } ( x , a ) } { u _ { t } ( x , a ) + \gamma } ( u _ { t } ( x , a ) - q _ { t } ( x , a ) + \gamma ) } \\ & { \qquad \leq \displaystyle \frac { L } { d } \sum _ { t , x , a } ( u _ { t } ( x , a ) - q _ { t } ( x , a ) + \gamma ) } \\ & { \qquad \leq \mathcal { O } \Bigg ( \displaystyle \frac { L ^ { 2 } | X | } { d } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + \displaystyle \frac { L | X | ^ { 3 } | A | } { d } \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \Bigg ) + \displaystyle \frac { L \gamma } { d } | X | | A | T , } \end{array}
$$

where the last line uses (C.18). This proves

$$
\begin{array} { r l } & { \displaystyle R ^ { ( 4 ) } \leq \mathcal { O } \biggl ( \frac { L ^ { 2 } | X | } { d } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + \frac { L | X | ^ { 3 } | A | } { d } \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \biggr ) } \\ & { \quad \quad \quad + \frac { L \gamma } { d } | X | | A | T + \frac { L ^ { 2 } } { d } \sqrt { 2 T \log ( 8 / \delta ) } . } \end{array}\tag{C.25}
$$

Comparator loss-estimation error The comparator bound in Lemma C.4 gives

$$
R ^ { ( 5 ) } \leq \frac { L } { \gamma } \log \frac { 8 | X | | A | } { \delta } .\tag{C.26}
$$

Combining the bounds Set $\begin{array} { r } { \eta = \gamma = \sqrt { \frac { L } { T | X | | A | } \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } } \end{array}$ . Combining (C.17), (C.20), (C.22), (C.25), and (C.26) gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \ell _ { t } ^ { \top } ( q _ { t } - q ) \leq \mathcal { O } \Bigg ( \frac { L ^ { 2 } | X | } { d } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + \frac { L | X | ^ { 3 } | A | } { d } \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \\ & { \quad \quad \quad + \frac { L \sqrt { L | X | | A | } } { d } \sqrt { T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + \frac { L ^ { 2 } } { d } \sqrt { T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } } \\ & { \quad \quad \quad \quad + \frac { L ^ { 2 } } { d } \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } + L \Bigg ) . } \end{array}
$$

Using $0 < d \leq L < | X | , | A | \geq 1$ , and log $\frac { 8 m | X | ^ { 2 } | A | T } { \delta } \geq 1$ , we obtain, for every $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \ell _ { t } ^ { \top } ( q _ { t } - q ) \leq \mathcal { O } \left( \frac { L ^ { 2 } | X | } { d } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } + \frac { L | X | ^ { 3 } | A | } { d } \log ^ { 2 } \frac { 8 m | X | ^ { 2 } | A | T } { \delta } \right) .
$$

Using the second bound in Lemma C.2 in (C.16) and the bound $\begin{array} { r } { \sum _ { t } \ell _ { t } ^ { \top } ( q _ { t } - q ) \leq L T } \end{array}$ , we also obtain

$$
\sum _ { t = 1 } ^ { T } \ell _ { t } ^ { \top } ( q _ { t } - q ) \leq \mathcal { O } \left( \frac { L | X | ^ { 2 } } { d } \sqrt { | A | T \log \frac { 8 m | X | ^ { 2 } | A | T } { \delta } } \right) .
$$

A union bound over the events in Lemmas C.2, C.4, and C.3, and in (C.24), gives probability at least $1 - \delta$ On their intersection, both regret bounds hold for every feasible $q ,$ and Lemma C.1 gives $G ^ { \top } q _ { t } \leq \alpha$ for all $t \in [ T ]$ . This proves the theorem. □

## D Analysis of Baseline Search

We prove Theorem 3 by bounding the numbers of exploration and baseline episodes in terms of the cumulative margin gap. We consider the search in Section 5.2 with confidence parameter δ and without the horizon condition $t < T$ . Let τ be its termination episode and N the number of exploration episodes with $\zeta _ { t } = 1$ allowing either count to be infinite. If $2 d \geq L$ , the search terminates immediately with $N = \tau = 0$

## D.1 Exploration Episodes

Initialization With the initialization in Appendix B.2, set $\widehat { q } _ { 1 } ( x , a , x ^ { \prime } ) = 1 / ( | X _ { k } | | A | | X _ { k + 1 } | )$ for $x \in X _ { k }$ and $x ^ { \prime } \in X _ { k + 1 }$ . This occupancy induces the uniform initial policy. Since every $q \in \Delta ( \mathcal { P } _ { 0 } )$ has total sum $L ,$

$$
( \widehat { g } _ { 0 , i } - \xi _ { 0 } ) ^ { \top } q = - L \leq \alpha _ { i } - L , \qquad ( \widehat { g } _ { 0 , i } + \xi _ { 0 } ) ^ { \top } q = L .
$$

Thus $( \widehat { q } _ { 1 } , L )$ solves the initial optimistic problem. Write $\overline { { \rho } } _ { 1 } = L , w _ { 1 , i } = L$ , and $\underline { { \rho } } _ { 1 } = \operatorname* { m i n } _ { i } ( \alpha _ { i } - L ) \leq 0$ . The baseline test is $2 d \geq \overline { { \rho } } _ { 1 }$ , while $2 \underline { { \rho } } _ { 1 } < \overline { { \rho } } _ { 1 }$ . The initial baseline probability $\lambda _ { 1 }$ is computed from $w _ { 1 }$

Observation sequence We use the episode filtration from Appendix B.2 and keep it constant after termination. Set $\zeta _ { t } = 0$ after termination, $\tau _ { 0 } = 0$ , and $\begin{array} { r } { \tau _ { n } = \operatorname* { i n f } \{ t \geq 1 : \sum _ { s = 1 } ^ { t } \zeta _ { s } \geq n \} } \end{array}$ for $n \geq 1$ , with inf $\mathcal { D } = \infty$ . Both τ and each $\tau _ { n }$ are stopping times. Set $\mathcal { H } _ { n } = \mathcal { F } _ { \tau _ { n } \wedge \tau }$ and write $q _ { n } = q ^ { P , \widehat { \pi } _ { n } }$ for each policy selected by the search. The decision to continue exploration is $\mathcal { H } _ { n - 1 }$ -measurable. When exploration continues, $\mathcal { H } _ { n - 1 } = \mathcal { F } _ { \tau _ { n - 1 } }$ , and $\widehat { \pi } _ { n }$ and $\lambda _ { n }$ are ${ \mathcal { H } } _ { n - 1 } { \mathrm { - m e a s u r a b l e } }$

Lemma D.1. For each policy $\widehat { \pi } _ { n }$ selected for exploration, $1 - \lambda _ { n } \geq d / L$ and $\tau _ { n } < \infty$ almost surely. Conditional on $\mathcal { H } _ { n - 1 }$ , episode $\tau _ { n }$ follows the transition kernel P and policy $\widehat { \pi } _ { n }$ , and its cost matrix has distribution $\mathcal { G }$ . In particular, $\mathbb { E } [ I _ { n } ( x , a ) \mid { \mathcal { H } } _ { n - 1 } ] = q _ { n } ( x , a )$ for every $( x , a )$ .

Proof. The search keeps $\widehat { \pi } _ { n }$ and its execution probability $1 - \lambda _ { n }$ unchanged until this policy is executed. If $\lambda _ { n } > 0$ , choose an index $i _ { n }$ attaining the maximum that defines $\lambda _ { n }$ . Then

$$
1 - \lambda _ { n } = \frac { \alpha _ { i _ { n } } - \beta _ { i _ { n } } } { w _ { n , i _ { n } } - \beta _ { i _ { n } } } \geq \frac { d } { L } .\tag{D.1}
$$

For $\lambda _ { n } = 0$ , the same bound follows from $d \leq L$ . For every integer $h \geq 1$ , the independent policy draws give $\mathbb { P } ( \tau _ { n } - \tau _ { n - 1 } > h \mid \mathcal { H } _ { n - 1 } ) = \lambda _ { n } ^ { h } \leq ( 1 - d / L ) ^ { h }$ . Thus $\tau _ { n } < \infty$ almost surely whenever the search continues with policy $\widehat { \pi } _ { n }$ . For each episode $s ,$ the event $\{ \tau _ { n - 1 } < s \leq \tau _ { n } \}$ is $\mathcal { F } _ { s - 1 }$ -measurable. On this event, the policy draw is independent of the current cost matrix and transition randomness. Thus, conditioning on $\zeta _ { s } = 1$ selects $\widehat { \pi } _ { n }$ without changing their distributions. Averaging over the number of preceding baseline episodes gives

$$
\mathbb { E } \big [ I _ { n } ( x , a ) \mid \mathcal { H } _ { n - 1 } \big ] = \sum _ { h = 0 } ^ { \infty } ( 1 - \lambda _ { n } ) \lambda _ { n } ^ { h } q _ { n } ( x , a ) = q _ { n } ( x , a ) .
$$

## D.2 Margin Bounds and Constraint Violation

Recall $\Delta _ { n } = \overline { { \rho } } _ { n } - \underline { { \rho } } _ { r }$ for each policy whose cost upper bounds $w _ { n }$ are computed, including the initial policy. Let $b _ { n - 1 }$ be defined by (B.7) for $\mathcal { P } _ { n - 1 }$

Lemma D.2. On ${ \mathcal E } _ { \mathrm { s } }$ , the optimistic problem is feasible and $\overline { { \rho } } _ { n } \geq \rho$ at initialization and after every estimate update. For the initial policy and every subsequent policy for which $w _ { n }$ is computed, we have $g _ { i } ^ { \top } q _ { n } \leq w _ { n , i }$ for every $\begin{array} { r } { i \in [ m ] , 0 \leq \Delta _ { n } \leq 4 L b _ { n - 1 } ^ { \top } q _ { n } + 2 \xi _ { n - 1 } ^ { \top } q _ { n } , \ a n d \ \underline { { \rho } } _ { n } \leq \operatorname* { m i n } _ { i } \big ( \alpha _ { i } - g _ { i } ^ { \top } q _ { n } \big ) \leq \rho } \end{array}$

Proof. Let $q ^ { \rho }$ attain the maximum defining $\rho$ in Section 2. On ${ \mathcal E } _ { \mathrm { s } } .$ , it belongs to $\Delta ( \mathcal { P } _ { n - 1 } )$ and satisfies

$$
( \widehat { G } _ { n - 1 } - \Xi _ { n - 1 } ) ^ { \top } q ^ { \rho } \leq G ^ { \top } q ^ { \rho } \leq \alpha - \rho \mathbf { 1 } .
$$

The occupancy $q ^ { \rho }$ is feasible for (5). Since $\rho \leq L$ , its objective value is at least $\rho ,$ which gives $\overline { { \rho } } _ { \underline { { n } } } \geq \rho .$ . The true occupancy $q _ { n }$ belongs to $\Delta ( \mathcal { P } _ { n - 1 } , \widehat { \pi } _ { n } )$ . Its cost is bounded by both $L$ and $( \widehat { g } _ { n - 1 , i } + \xi _ { n - 1 } ) ^ { \top } q _ { n } .$ , so the definition of $w _ { n , i }$ gives $g _ { i } ^ { \top } q _ { n } \leq w _ { n , i } .$ . For any $r \in \Delta ( \mathcal { P } _ { n - 1 } , \widehat { \pi } _ { n } )$ , Lemmas B.1 and B.3 $\mathrm { g i v e } \ \| r - q _ { n } \| _ { 1 } \leq L b _ { n - 1 } ^ { \top } q _ { n }$ and $\| \widehat { q } _ { n } - q _ { n } \| _ { 1 } \leq L b _ { n - 1 } ^ { \top } q _ { n }$

The optimistic constraint and $\| \widehat { g } _ { n - 1 , i } - \xi _ { n - 1 } \| _ { \infty } \leq 1$ imply

$$
\begin{array} { r l } & { ( \widehat { g } _ { n - 1 , i } + \xi _ { n - 1 } ) ^ { \top } r = ( \widehat { g } _ { n - 1 , i } - \xi _ { n - 1 } ) ^ { \top } \widehat { q } _ { n } + ( \widehat { g } _ { n - 1 , i } - \xi _ { n - 1 } ) ^ { \top } ( r - \widehat { q } _ { n } ) } \\ & { \qquad + 2 \xi _ { n - 1 } ^ { \top } r } \\ & { \leq \alpha _ { i } - \overline { { \rho } } _ { n } + \| r - \widehat { q } _ { n } \| _ { 1 } + 2 \xi _ { n - 1 } ^ { \top } r } \\ & { \leq \alpha _ { i } - \overline { { \rho } } _ { n } + 3 \| r - q _ { n } \| _ { 1 } + \| q _ { n } - \widehat { q } _ { n } \| _ { 1 } + 2 \xi _ { n - 1 } ^ { \top } q _ { n } } \\ & { \leq \alpha _ { i } - \overline { { \rho } } _ { n } + 4 L b _ { n - 1 } ^ { \top } q _ { n } + 2 \xi _ { n - 1 } ^ { \top } q _ { n } . } \end{array}
$$

The second inequality uses the triangle inequality and $\| \xi _ { n - 1 } \| _ { \infty } \leq 1$ . Maximizing over r and using the definition of $w _ { n , i }$ give the same upper bound for $w _ { n , i }$ . Taking the minimum over i in $\alpha _ { i } - w _ { n , i }$ proves the upper bound on $\Delta _ { n }$ . The bounds on $\underline { { \rho } } _ { n }$ follow from $g _ { i } ^ { \top } q _ { n } \leq w _ { n , }$ and the definition of $\rho$ . Together with $\overline { { \rho } } _ { n } \geq \rho ,$ they also give $\Delta _ { n } \geq 0$ □

Write $\mu _ { n } = \lambda _ { n } / ( 1 - \lambda _ { n } )$ for the conditional expected number of baseline episodes before policy $\widehat { \pi } _ { n }$ is executed. Lemma D.3. On ${ \mathcal E } _ { \mathrm { s } }$ , every randomized policy used during the search satisfies $G ^ { \top } q _ { t } \leq \alpha$ . For every integer $n \in [ N ] , \mu _ { n } \leq \Delta _ { n } / d$

Proof. The function $z \mapsto z / ( 1 - z )$ is increasing on [0, 1). Applying it to both branches of the mixing rule gives

$$
\begin{array} { c l l } { \displaystyle \mu _ { n } = \operatorname* { m a x } _ { i \in [ m ] } \frac { [ w _ { n , i } - \alpha _ { i } ] _ { + } } { \alpha _ { i } - \beta _ { i } } \leq \frac { [ - \underline { { \rho } } _ { n } ] _ { + } } { d } } \\ { \displaystyle = \frac { [ \Delta _ { n } - \overline { { \rho } } _ { n } ] _ { + } } { d } \leq \frac { \Delta _ { n } } { d } . } \end{array}
$$

The last inequality follows from Lemma D.2. During an episode with search index n, the randomized policy has occupancy $q _ { t } = \lambda _ { n } q ^ { P , \pi ^ { \diamond } } + ( 1 - \lambda _ { n } ) q _ { n }$ . Fix $i \in [ m ]$ . If $w _ { n , i } \leq \alpha _ { i }$ , then $g _ { i } ^ { \top } q _ { t } \leq \lambda _ { n } \beta _ { i } + ( 1 - \lambda _ { n } ) w _ { n , i } \leq \alpha _ { i }$ If $w _ { n , i } > \alpha _ { i }$ , the mixing rule gives $\lambda _ { n } \geq ( w _ { n , i } - \alpha _ { i } ) / ( w _ { n , i } - \beta _ { i } )$ , and hence

$$
\begin{array} { r l } & { g _ { i } ^ { \top } q _ { t } \leq \lambda _ { n } \beta _ { i } + ( 1 - \lambda _ { n } ) w _ { n , i } } \\ & { \qquad = w _ { n , i } - \lambda _ { n } ( w _ { n , i } - \beta _ { i } ) } \\ & { \qquad \leq \alpha _ { i } . } \end{array}
$$

## D.3 Number of Exploration Episodes

Proof of Lemma 5.1. The factor log $\frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta }$ increases with the search index. Applying (B.8) and (B.9) therefore gives, for any $x \in X _ { k }$ and $a \in A$

$$
\sum _ { j = 1 } ^ { n } b _ { j - 1 } ( x , a ) I _ { j } ( x , a ) \leq \mathcal { O } \left( \sqrt { | X _ { k + 1 } | N _ { n } ( x , a ) \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } \right) .
$$

Since $\begin{array} { r } { \sum _ { x \in X _ { k } , a } N _ { n } ( x , a ) = n } \end{array}$ , we use the Cauchy–Schwarz inequality and it yields

$$
\begin{array} { r l r } {  { \sum _ { j = 1 } ^ { n } b _ { j - 1 } ^ { \top } I _ { j } \le \mathcal { O } ( \sqrt { | A | n \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } \sum _ { k = 0 } ^ { L - 1 } \sqrt { | X _ { k } | | X _ { k + 1 } | } ) } } \\ & { } & { \leq \mathcal { O } ( | X | \sqrt { | A | n \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } ) . } \end{array}\tag{D.2}
$$

The last inequality uses $\sqrt { u v } \leq ( u + v ) / 2$ . Applying (B.4) and (B.9) to the cost confidence widths gives

$$
\sum _ { j = 1 } ^ { n } \xi _ { j - 1 } ^ { \top } I _ { j } \leq 6 \sqrt { L | X | | A | n \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } .\tag{D.3}
$$

Martingale bounds By Lemma $\mathrm { D . 1 } , b _ { j - 1 } ^ { \top } ( q _ { j } - I _ { j } )$ and $\xi _ { j - 1 } ^ { \top } ( q _ { j } - I _ { j } )$ have conditional mean zero given $\mathcal { H } _ { j - 1 }$ and absolute values at most 2L and $L ,$ respectively. Extend both sequences by zero after the search terminates. The decision to continue exploration is H<sub>j−1</sub>-measurable, so the extension preserves the martingale diference property. For each integer $n \geq 1$ , apply Azuma–Hoefding with failure probability $\delta / [ 8 ( n \dot { + } 1 ) ^ { 2 } ]$ to each sequence. Since $m | X | ^ { 2 } | A | ( n + 1 ) \geq 8$ , we have log $\begin{array} { r } { ( 8 ( n + 1 ) ^ { 2 } / \delta ) \le \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } \end{array}$ . A union bound gives

$$
\sum _ { j = 1 } ^ { n } b _ { j - 1 } ^ { \top } q _ { j } \leq \sum _ { j = 1 } ^ { n } b _ { j - 1 } ^ { \top } I _ { j } + 2 L \sqrt { 2 n \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } ,\tag{D.4}
$$

$$
\sum _ { j = 1 } ^ { n } \xi _ { j - 1 } ^ { \top } q _ { j } \leq \sum _ { j = 1 } ^ { n } \xi _ { j - 1 } ^ { \top } I _ { j } + L \sqrt { 2 n \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } ,\tag{D.5}
$$

for every integer $n \in [ N ]$ , with total failure probability less than $\delta / 4$ . On the intersection of this event and $\mathcal { E } _ { \mathrm { s } } .$ , the pointwise gap bound in Lemma D.2 and (D.2)–(D.5) give

$$
\begin{array} { r l r } {  { \sum _ { j = 1 } ^ { n } \Delta _ { j } \le 4 L \sum _ { j = 1 } ^ { n } b _ { j - 1 } ^ { \top } q _ { j } + 2 \sum _ { j = 1 } ^ { n } \xi _ { j - 1 } ^ { \top } q _ { j } } } \\ & { } & { \le { \mathcal O } ( L | { \cal X } | \sqrt { | A | n \log \frac { m | { \cal X } | ^ { 2 } | { \cal A } | ( n + 1 ) ^ { 3 } } { \delta } } ) } \end{array}\tag{D.6}
$$

for every finite integer $n \in [ N ]$ , where we used $| X | \geq L + 1$ . The combined failure probability is at most $3 \delta / 8 \le \delta / 2$ , proving the lemma. □

Lemma D.4. On the event established in the proof of Lemma 5.1, the search terminates $a f t e r$

$$
N \leq \mathcal { O } \bigg ( \frac { L ^ { 2 } | X | ^ { 2 } | A | } { \rho ^ { 2 } } \log \frac { m L ^ { 2 } | X | ^ { 4 } | A | ^ { 2 } } { \rho ^ { 2 } \delta } \bigg )
$$

exploration episodes. The returned policy, cost bounds, and margin satisfy the guarantees in Theorem 3.

Proof. The j-th exploration episode occurs only if $2 \underline { { \rho } } _ { ; i } < \overline { { \rho } } _ { j }$ . Lemma D.2 then gives $\Delta _ { j } = \overline { { \rho } } _ { j } - \underline { { \rho } } _ { i } > \rho / 2$ . For every finite integer $n \in [ N ]$ , (D.6) implies

$$
\begin{array} { l } { \displaystyle \frac { n \rho } { 2 } < \sum _ { j = 1 } ^ { n } \Delta _ { j } } \\ { \displaystyle \leq \mathcal { O } \left( L | X | \sqrt { | A | n \log \frac { m | X | ^ { 2 } | A | ( n + 1 ) ^ { 3 } } { \delta } } \right) . } \end{array}\tag{D.7}
$$

Squaring gives $n \leq a \log ( b ( n + 1 ) ^ { 3 } )$ for some $a = \mathcal { O } ( L ^ { 2 } | X | ^ { 2 } | A | / \rho ^ { 2 } )$ with $a \geq 2$ , where $b = m | X | ^ { 2 } | A | / \delta \geq 2$ The inequality log $x \leq x - 1$ gives

$$
\begin{array} { c } { n \displaystyle \leq a \log \big ( b ( 6 a ) ^ { 3 } \big ) + 3 a \log \frac { n + 1 } { 6 a } } \\ { \leq a \log \big ( b ( 6 a ) ^ { 3 } \big ) + \displaystyle \frac { n + 1 } { 2 } - 3 a . } \end{array}
$$

Hence $n \leq 2 a \log ( b ( 6 a ) ^ { 3 } ) = \mathcal { O } ( a \log ( a b ) )$ . Since this bound holds for every finite $n \leq N$ , we have $N < \infty$ Lemma D.1 ensures that each selected exploration policy is executed in finite time. The search therefore terminates, with

$$
N \leq \mathcal { O } \bigg ( \frac { L ^ { 2 } | X | ^ { 2 } | A | } { \rho ^ { 2 } } \log \frac { m L ^ { 2 } | X | ^ { 4 } | A | ^ { 2 } } { \rho ^ { 2 } \delta } \bigg ) .\tag{D.8}
$$

The bound also holds for $N = 0$ . For $N \geq 1$ , the stopping tests are evaluated after the update following episode $\tau _ { N }$ , so $\tau = \tau _ { N }$

Returned policy At termination, the search index is $N + 1$ . If $2 d \ge \overline { { \rho } } _ { N + 1 }$ , then $d \ge \overline { { \rho } } _ { N + 1 } / 2 \ge \rho / 2$ so $( \pi ^ { \diamond } , \beta , d )$ satisfies the stated guarantee. Otherwise, the search returns $\left( \widehat { \pi } _ { N + 1 } , w _ { N + 1 } \right)$ after verifying $2 \underline { { \rho } } _ { N + 1 } \geq \overline { { \rho } } _ { N + 1 }$ . Thus $\underline { { \rho } } _ { N + 1 } \geq \overline { { \rho } } _ { N + 1 } / 2 \geq \rho / 2$ , and Lemma D.2 gives $G ^ { \top } q _ { N + 1 } \le w _ { N + 1 } < \alpha$ and $\underline { { \rho } } _ { N + 1 } \le \rho .$ The optimistic problem remains feasible on ${ \mathcal { E } } _ { \mathrm { s } } ,$ so termination occurs through one of these two tests. □

## D.4 Number of Baseline Episodes

For $n \in [ N ]$ , let $H _ { n } = \tau _ { n } - \tau _ { n - 1 } - 1$ be the number of baseline episodes between consecutive exploration episodes, with $\tau _ { 0 } = 0$ . Conditional on $\mathcal { H } _ { n - 1 }$ , it satisfies $\mathbb { P } ( H _ { n } = h \mid \mathcal { H } _ { n - 1 } ) = ( 1 - \lambda _ { n } ) \lambda _ { n } ^ { h }$ for $h = 0 , 1 , . . . ,$ so $\mathbb { E } [ H _ { n } \mid \mathcal { H } _ { n - 1 } ] = \mu _ { n }$

Lemma D.5. With probability at least $1 - \delta / 4$ , it holds that $\begin{array} { r } { \sum _ { j = 1 } ^ { n } H _ { j } \leq 2 \sum _ { j = 1 } ^ { n } \mu _ { j } + \frac { 4 L } { d } \log \frac { 4 } { \delta } \ f o r } \end{array}$ every integer $n \in [ N ]$

Proof. Set $\theta = d / ( 4 L )$ and extend $H _ { j }$ and $\mu _ { j }$ by zero after the search terminates. While the search continues, (D.1) gives $\theta \leq ( 1 - \lambda _ { j } ) / 4$ . Since $\theta \leq 1 / 4$ and $e ^ { \theta } - 1 \leq 4 \theta / 3$ , we have $\mu _ { j } ( e ^ { \theta } - 1 ) \leq 1 / 3$ . The geometric distribution therefore gives

$$
\mathbb { E } [ e ^ { \theta H _ { j } } \ | \ \mathcal { H } _ { j - 1 } ] = \frac { 1 - \lambda _ { j } } { 1 - \lambda _ { j } e ^ { \theta } } = \frac { 1 } { 1 - \mu _ { j } ( e ^ { \theta } - 1 ) } .
$$

Using $- \log ( 1 - x ) \leq x / ( 1 - x )$ for $0 \leq x < 1$ , we obtain

$$
\begin{array} { r l } { \log \mathbb { E } [ e ^ { \theta H _ { j } } \ | \ \mathcal { H } _ { j - 1 } ] = - \log \bigl ( 1 - \mu _ { j } ( e ^ { \theta } - 1 ) \bigr ) } & { } \\ & { \leq \frac { \mu _ { j } \bigl ( e ^ { \theta } - 1 \bigr ) } { 1 - \mu _ { j } ( e ^ { \theta } - 1 ) } } \\ & { \leq \frac { 3 } { 2 } \mu _ { j } ( e ^ { \theta } - 1 ) \leq 2 \theta \mu _ { j } . } \end{array}
$$

The decision to continue exploration is ${ \mathcal { H } } _ { j - 1 } { \mathrm { - m e a s u r a b l e } } .$ . Thus $M _ { n } = \exp \{ \theta \sum _ { i = 1 } ^ { n } ( H _ { j } - 2 \mu _ { j } ) \}$ is a nonnegative supermartingale with $M _ { 0 } = 1$ . Ville’s inequality yields $\mathbb { P } ( \operatorname* { s u p } _ { n \geq 0 } M _ { n } \geq 4 / \bar { \delta } ) \leq \delta / 4$ . Rearranging on the complementary event proves the bound. □

Proof of Lemma 5.2. Equation (8) gives $\mu _ { j } ~ \leq ~ [ - \underline { { \rho } } _ { i } ] + / d$ for every exploration policy. On the event in Lemma D.5, we use $\begin{array} { r } { \tau _ { n } - n = \sum _ { j = 1 } ^ { n } H _ { j } } \end{array}$ and substitute this bound to obtain

$$
\tau _ { n } - n \leq \frac { 2 } { d } \sum _ { j = 1 } ^ { n } [ - \underline { { \rho } } _ { j } ] _ { + } + \frac { 4 L } { d } \log \frac { 4 } { \delta }
$$

for every finite integer $n \in [ N ]$ , with probability at least $1 - \delta / 4$

## D.5 Proof of Theorem 3

Proof of Theorem 3. The intersection of the events in the proof of Lemma 5.1 and in Lemma D.5 has failure probability at most $5 \delta / 8 .$ On this event, Lemma D.4 gives termination, the bound on N, and the guarantee for the returned policy. Lemma D.3 gives $G ^ { \top } q _ { t } \leq \bar { \alpha }$ for every $t \leq \tau$ . It remains to bound τ. If $N = 0$ then $\tau = 0$ . Otherwise, $\tau = \tau _ { N }$ . Equation (D.8) gives log $\begin{array} { r } { \frac { m \cdot | X | ^ { 2 } | A | ( N + 1 ) ^ { 3 } } { \delta } = \mathcal { O } ( \log ( m L ^ { 2 } | X | ^ { 4 } | A | ^ { 2 } / ( \rho ^ { 2 } \delta ) ) ) } \end{array}$ Substituting (D.8) into (D.6) therefore yields

$$
\sum _ { j = 1 } ^ { N } \Delta _ { j } \leq \mathcal { O } \bigg ( \frac { L ^ { 2 } | X | ^ { 2 } | A | } { \rho } \log \frac { m L ^ { 2 } | X | ^ { 4 } | A | ^ { 2 } } { \rho ^ { 2 } \delta } \bigg ) .
$$

Since $\begin{array} { r } { \tau = N + \sum _ { j = 1 } ^ { N } H _ { j } } \end{array}$ , (D.8) and Lemmas D.5 and D.3 give

$$
\begin{array} { l } { \displaystyle \tau \leq N + \frac { 2 } { d } \sum _ { j = 1 } ^ { N } \Delta _ { j } + \frac { 4 L } { d } \log \frac { 4 } { \delta } } \\ { \displaystyle \leq \mathcal { O } \biggl ( \frac { L ^ { 2 } | X | ^ { 2 } | A | } { d \rho } \log \frac { m L ^ { 2 } | X | ^ { 4 } | A | ^ { 2 } } { \rho ^ { 2 } \delta } \biggr ) , } \end{array}\tag{D.9}
$$

where the last inequality uses $d \le \rho \le L$ . Together with the bounds for the returned policy and $5 \delta / 8 < \delta .$ this proves the theorem. □

## E Proof of Theorem 2

Proof of Theorem 2. Let τ be the termination episode of the search defined in Appendix D. If needed, continue the search beyond T using independent samples from the same model and zero losses. This agrees with Algorithm 1 through episode $\tau \wedge T$ . Apply Theorem 3 with confidence parameter $\delta / 2$ and let $\mathcal { E }$ be the event on which its guarantees hold. On $\mathcal { E } ,$ , every randomized policy used during the search satisfies the constraints, (D.9) bounds $\tau ,$ and the returned cost upper bounds give $\widehat { \rho } : = \mathrm { m i n } _ { i \in [ m ] } ( \alpha _ { i } - \widehat { \beta } _ { i } ) \geq \rho / 2$

Regret after the search We use the episode filtration from Appendix B.2. At the bounded stopping time $\tau \wedge T$ , define

$$
\mathcal { V } = \{ \tau < T , \ G ^ { \top } q ^ { P , \widehat { \pi } ^ { \diamond } } \leq \widehat { \beta } < \alpha \} \in \mathcal { F } _ { \tau \wedge T } .
$$

On $\nu ,$ the returned baseline and cost bounds satisfy Assumption 1. These inputs and the remaining horizon $T ^ { \prime } = T - \tau$ are $\mathcal { F } _ { \tau \wedge T }$ -measurable. Conditional on $\mathcal { F } _ { \tau \wedge T }$ , the remaining episodes satisfy the observation model in Section 2, with non-anticipating losses. Algorithm 1 initializes S-OPS with zero observation counts, confidence parameter $\delta / 2 .$ , and learning rates set for $T ^ { \prime }$ . Let B be the event that S-OPS is restarted and the common high-probability event established in Appendix C.4 fails. Applying Theorem 1 conditionally on $\mathcal { F } _ { \tau \wedge T }$ gives

$$
\mathbf { 1 } \{ \mathcal { V } \} \mathbb { P } ( \boldsymbol { B } \mid \mathcal { F } _ { \tau \wedge T } ) \leq \frac { \delta } { 2 } \mathbf { 1 } \{ \mathcal { V } \} .\tag{E.1}
$$

On $\nu \cap B ^ { c }$ , the regret bound holds for all feasible occupancy measures, so it applies to $q ^ { \star }$ in (1). On $\mathcal { E } \cap \{ \tau < T \} \cap B ^ { c } , G ^ { \top } q _ { t } \leq \alpha$ for every $\tau < t \leq T$ and

$$
\begin{array} { r } { \displaystyle \sum _ { t = \tau + 1 } ^ { T } \ell _ { t } ^ { \top } ( q _ { t } - q ^ { \star } ) \leq \mathcal { O } \bigg ( \frac { L ^ { 2 } | X | } { \widehat { \rho } } \sqrt { | A | T ^ { \prime } \log \frac { 1 6 m | X | ^ { 2 } | A | T ^ { \prime } } { \delta } } + \frac { L | X | ^ { 3 } | A | } { \widehat { \rho } } \log ^ { 2 } \frac { 1 6 m | X | ^ { 2 } | A | T ^ { \prime } } { \delta } \bigg ) } \\ { \leq \mathcal { O } \bigg ( \frac { L ^ { 2 } | X | } { \rho } \sqrt { | A | T \log \frac { 1 6 m | X | ^ { 2 } | A | T } { \delta } } + \frac { L | X | ^ { 3 } | A | } { \rho } \log ^ { 2 } \frac { 1 6 m | X | ^ { 2 } | A | T } { \delta } \bigg ) . } \end{array}\tag{E.2}
$$

The last inequality uses $\widehat { \rho } \geq \rho / 2$ and $T ^ { \prime } \leq T$

Regret during the search The regret against $q ^ { \star }$ in each episode is at most $L ,$ since losses lie in [0, 1] and every occupancy measure has total sum $L .$ . On $\mathcal { E } ,$ the episode bound (D.9) with confidence parameter $\delta / 2$ gives

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T \wedge T } \ell _ { t } ^ { \top } ( q _ { t } - q ^ { \star } ) \leq L ( \tau \wedge T ) \leq L \tau } } \\ & { } & { \leq \mathcal { O } \bigg ( \frac { L ^ { 3 } | X | ^ { 2 } | A | } { d \rho } \log \frac { m L ^ { 2 } | X | ^ { 4 } | A | ^ { 2 } } { \rho ^ { 2 } \delta } \bigg ) . } \end{array}\tag{E.3}
$$

Combining the phases For every $T \geq 1$ , the inequalities $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ and $( \log z ) ^ { 2 } \leq 3 { \sqrt { z } }$ for $z \geq 1 \ \mathrm { g i v }$ e

$$
\begin{array} { r } { | X | ^ { 3 } | A | \log ^ { 2 } \frac { 1 6 m | X | ^ { 2 } | A | T } { \delta } \leq 2 | X | ^ { 3 } | A | \log ^ { 2 } \frac { 1 6 m | X | ^ { 6 } | A | ^ { 2 } } { \delta } + 6 | X | \sqrt { | A | T } . } \end{array}
$$

When $\tau < T$ , we combine this estimate with (E.2) and (E.3) and use $d \leq \rho \leq L < | X |$ to obtain

$$
R _ { T } \leq \mathcal { O } \Bigg ( \frac { L ^ { 2 } | X | } { \rho } \sqrt { | A | T \log \frac { 1 6 m | X | ^ { 2 } | A | T } { \delta } } + \frac { L ^ { 2 } | X | ^ { 3 } | A | } { d \rho } \log ^ { 2 } \frac { 1 6 m L ^ { 2 } | X | ^ { 6 } | A | ^ { 2 } } { \rho ^ { 2 } \delta } \Bigg ) ~ .\tag{E.4}
$$

If $\tau \geq T$ , (E.3) gives the same bound. The tower property and (E.1) give

$$
\mathbb { P } \big ( \mathcal { E } ^ { c } \cup ( \mathcal { V } \cap \mathcal { B } ) \big ) \le \mathbb { P } \big ( \mathcal { E } ^ { c } \big ) + \mathbb { E } [ \mathbf { 1 } \{ \mathcal { V } \} \mathbb { P } ( \mathcal { B } \mid \mathcal { F } _ { \tau \wedge T } ) ] \le \delta .
$$

On the complementary event, the search succeeds and S-OPS satisfies its guarantee whenever it is invoked. Thus, with probability at least $1 - \delta , G ^ { \top } q _ { t } \leq \alpha$ for every $t \in [ T ]$ and the stated regret bound holds. □

## F Proof of Theorem 4

We use two pairs of instances to prove the $\sqrt { T } / \rho$ and $1 / ( d \rho )$ terms separately. Both pairs have initial margin d and Slater margin $\rho .$

Proof of Theorem $\it 4 .$ Fix an algorithm satisfying the safety requirement of Theorem 4. In both constructions, we use $L = m = 1$ , actions $a _ { 0 } , a _ { 1 } , a _ { 2 }$ , and threshold $\alpha = 1 / 2$ . The baseline policy always chooses $a _ { 0 }$ , whose deterministic cost is $\beta = 1 / 2 - d .$ . The costs of $a _ { 1 }$ and $a _ { 2 }$ are Bernoulli variables, independent across actions and episodes. The losses are deterministic and constant across episodes. The instances in each pair difer only in their cost distributions. All vectors are indexed by $( a _ { 0 } , a _ { 1 } , a _ { 2 } )$

In episode t, let $q _ { t }$ be the action distribution selected by the algorithm, $A _ { t } \sim q _ { t }$ the sampled action, and $Y _ { t }$ its observed cost. For each pair $M _ { 1 } , M _ { 2 }$ , let $\mathbb { P } ^ { j }$ and $\mathbb { E } ^ { j }$ denote the law of $( q _ { t } , A _ { t } , Y _ { t } ) _ { t = 1 } ^ { T }$ and the corresponding expectation in $M _ { j }$

The learning term Set $\epsilon = 1 / ( 8 \sqrt { T } )$ and use the constant loss vector $\ell = ( 1 / 2 , 0 , 1 / 2 )$ . The horizon condition and $d \leq \rho$ give $T \ge 1 / \rho ^ { 2 }$ , so $\epsilon \le \rho / 8$ . Consider instances $M _ { 1 } , M _ { 2 }$ with mean cost vectors

$$
\begin{array} { l } { g ^ { 1 } = ( 1 / 2 - d , 1 / 2 + \epsilon , 1 / 2 - \rho ) , } \\ { g ^ { 2 } = ( 1 / 2 - d , 1 / 2 , 1 / 2 - \rho ) . } \end{array}
$$

For $j = 1 , 2$ , the Slater margin is $1 / 2 - \operatorname* { m i n } _ { a } g ^ { j } ( a ) = \rho$ , and the baseline margin is $d .$ In $M _ { 1 }$

$$
\begin{array} { r l } & { ( g ^ { 1 } ) ^ { \top } q _ { t } - 1 / 2 = \epsilon - ( \epsilon + d ) q _ { t } ( a _ { 0 } ) - ( \epsilon + \rho ) q _ { t } ( a _ { 2 } ) } \\ & { ~ \geq \epsilon - ( \epsilon + \rho ) ( 1 - q _ { t } ( a _ { 1 } ) ) . } \end{array}
$$

Let $\begin{array} { r } { E = \{ \sum _ { t = 1 } ^ { T } ( 1 - q _ { t } ( a _ { 1 } ) ) \ge T \epsilon / ( \epsilon + \rho ) \} } \end{array}$ . Zero constraint violation in $M _ { 1 }$ requires $1 - q _ { t } ( a _ { 1 } ) \geq \epsilon / ( \epsilon + \rho )$ in every episode, so $\mathbb { P } ^ { 1 } ( E ) \geq 1 - \delta .$ Only the cost distribution of $a _ { 1 }$ difers between the two instances. The chain rule for KL divergence (Lattimore and Szepesvári, 2020; Pacchiano et al., 2021), applied to the conditional distributions of policy choices, actions, and observed costs, gives

$$
\begin{array} { r l r } {  { \mathrm { K L } ( \mathbb { P } ^ { 1 } \| \mathbb { P } ^ { 2 } ) = \mathbb { E } ^ { 1 } [ \sum _ { t = 1 } ^ { T } q _ { t } ( a _ { 1 } ) ] \mathrm { k l } ( 1 / 2 + \epsilon , 1 / 2 ) } } \\ & { } & { \leq 4 T \epsilon ^ { 2 } = 1 / 1 6 . } \end{array}\tag{F.1}
$$

The inequality uses $\operatorname { k l } ( r , s ) \leq ( r - s ) ^ { 2 } / ( s ( 1 - s ) )$ for $r , s \in ( 0 , 1 )$ . Pinsker’s inequality then gives $\mathrm { T V } ( \mathbb { P } ^ { 1 } , \mathbb { P } ^ { 2 } ) <$ $1 / 4$ and $\mathbb { P } ^ { 2 } ( E ) \geq 3 / 4 - \delta$ . In $M _ { 2 }$ , action $a _ { 1 }$ is feasible and has zero loss. On $E .$

$$
R _ { T } = \frac { 1 } { 2 } \sum _ { t = 1 } ^ { T } ( 1 - q _ { t } ( a _ { 1 } ) ) \geq \frac { T \epsilon } { 2 ( \epsilon + \rho ) } \geq \frac { T \epsilon } { 4 \rho } = \frac { \sqrt { T } } { 3 2 \rho } .\tag{F.2}
$$

This construction extends the comparison in Stradi et al. (2025a, Appendix F) by prescribing the initial baseline margin separately from the Slater margin.

The exploration term Suppose $d \leq \rho / 4$ . Take $\ell = ( 1 , 0 , 0 )$ and

$$
\begin{array} { r } { g ^ { 1 } = ( 1 / 2 - d , 1 / 2 - \rho , 1 / 2 + 2 \rho ) , } \\ { g ^ { 2 } = ( 1 / 2 - d , 1 / 2 + 2 \rho , 1 / 2 - \rho ) . } \end{array}
$$

Again, $1 / 2 - \operatorname* { m i n } _ { a } g ^ { j } ( a ) = \rho$ and the baseline margin is $d .$ Each instance has a feasible zero-loss action. Set $p _ { t } = q _ { t } ( a _ { 1 } ) + q _ { t } ( a _ { 2 } )$ and $p = 2 d / ( \rho + 2 d )$ . Since

$$
( g ^ { 1 } ) ^ { \top } q _ { t } + ( g ^ { 2 } ) ^ { \top } q _ { t } = 1 - 2 d + ( \rho + 2 d ) p _ { t } ,
$$

a distribution feasible in both instances satisfies

$$
p _ { t } \leq p \leq { \frac { 1 } { 3 } } .\tag{F.3}
$$

For $t \in [ T ]$ , let $\mathcal { F } _ { t - 1 } = \sigma ( q _ { 1 } , ( A _ { s } , Y _ { s } , q _ { s + 1 } ) _ { 1 \leq s < t } )$ be the history before sampling $A _ { t } ,$ and set $\mathcal { F } _ { T } = \mathcal { F } _ { T - 1 } \vee$ $\sigma ( A _ { T } , Y _ { T } )$ . Let $n = \lceil 1 / ( 7 6 8 d \rho ) \rceil$ and let σ be the first episode in which the selected policy is infeasible in either instance, capped at n:

$$
\sigma = \operatorname* { m i n } \big ( \{ t \in [ n ] : \operatorname* { m a x } _ { j \in \{ 1 , 2 \} } ( g ^ { j } ) ^ { \top } q _ { t } > 1 / 2 \} \cup \{ n \} \big ) .
$$

For every $t < \sigma$ , the policy is feasible in both instances, so (F.3) gives $p _ { t } \leq p .$ Since $d \rho \leq 1 / 2 5 6$ , we have $n \leq 1 + 1 / ( 7 6 8 d \rho ) \leq 1 / ( d \rho ) \leq T$ . The policy $q _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable, so $\sigma - 1$ is a bounded stopping time. Let $\mathbb { Q } ^ { j } = \mathbb { P } ^ { j } | _ { \mathcal { F } _ { \sigma - 1 } }$ for $j = 1 , 2 .$ This σ-algebra contains the policy choices through $q _ { \sigma }$ and the actions and observed costs from episodes $t < \sigma$ . For each action $^ { a , }$ let $\nu _ { j , a }$ be its cost distribution in $M _ { j }$ . The conditional policy and action distributions are the same in both instances, so their contributions to the likelihood ratio cancel. The common stopping rule therefore gives

$$
\log \frac { d \mathbb { Q } ^ { 1 } } { d \mathbb { Q } ^ { 2 } } = \sum _ { t = 1 } ^ { n - 1 } \mathbf { 1 } \{ t < \sigma \} \log \frac { d \nu _ { 1 , A _ { t } } } { d \nu _ { 2 , A _ { t } } } ( Y _ { t } ) .\tag{F.4}
$$

For $a _ { 1 }$ and $^ { a _ { 2 } , }$ the mean costs difer by $3 \rho$ between instances and lie in $[ 3 / 8 , 3 / 4 ]$ . Since $s ( 1 - s ) \geq 3 / 1 6$ on this interval, their Bernoulli KL divergences are at most $4 8 \rho ^ { 2 }$ in either direction. The KL divergence for the baseline action is zero. Taking expectations in (F.4) and conditioning on $\mathcal { F } _ { t - 1 } \ \mathrm { g i r }$ ves

$$
\begin{array} { r l r } {  { \mathrm { K L } ( \mathbb { Q } ^ { 1 } \| \mathbb { Q } ^ { 2 } ) = \mathbb { E } ^ { 1 } [ \sum _ { t = 1 } ^ { n - 1 } \mathbf { 1 } \{ t < \sigma \} \sum _ { a } q _ { t } ( a ) \mathrm { K L } ( \nu _ { 1 , a } \| \nu _ { 2 , a } ) ] } } \\ & { } & { \leq 4 8 \rho ^ { 2 } \mathbb { E } ^ { 1 } [ \sum _ { t = 1 } ^ { n - 1 } \mathbf { 1 } \{ t < \sigma \} p _ { t } ] } \\ & { } & { \leq 4 8 ( n - 1 ) p \rho ^ { 2 } \leq 9 6 ( n - 1 ) d \rho \leq 1 / 8 . } \end{array}\tag{F.5}
$$

The last two lines use $p _ { t } \leq p$ before $\sigma , p \leq 2 d / \rho$ , and $n - 1 < 1 / ( 7 6 8 d \rho )$ . Pinsker’s inequality therefore gives $\operatorname { T V } ( \mathbb { Q } ^ { 1 } , \mathbb { Q } ^ { 2 } ) \leq 1 / 4$ . Define $E _ { j } = \{ ( g ^ { j } ) ^ { \top } q _ { \sigma } > 1 / 2 \}$ for $j = 1 , 2$ . Each event $E _ { j }$ belongs to $\mathcal { F } _ { \sigma - 1 }$ and implies a constraint violation in $M _ { j }$ . Hence $\mathbb { P } ^ { j } ( E _ { j } ) \le \delta$ . The event $E _ { 1 } \cup E _ { 2 }$ occurs precisely when a policy in the first n episodes violates a constraint in either instance. Consequently,

$$
\begin{array} { r l } & { \mathbb { P } ^ { 1 } ( E _ { 1 } \cup E _ { 2 } ) \leq \mathbb { P } ^ { 1 } ( E _ { 1 } ) + \mathbb { P } ^ { 1 } ( E _ { 2 } ) } \\ & { \qquad \leq \delta + \mathbb { P } ^ { 2 } ( E _ { 2 } ) + \mathrm { T V } ( \mathbb { Q } ^ { 1 } , \mathbb { Q } ^ { 2 } ) } \\ & { \qquad \leq 2 \delta + 1 / 4 . } \end{array}\tag{F.6}
$$

Thus $\mathbb { P } ^ { 1 } ( ( E _ { 1 } \cup E _ { 2 } ) ^ { c } ) \geq 3 / 4 - 2 \delta$ . On this event, $\sigma = n$ and every policy through episode n is feasible in both instances. Therefore $p _ { t } \le p \le 1 / 3$ for every $t \leq n$ , and

$$
R _ { T } = \sum _ { t = 1 } ^ { T } ( 1 - p _ { t } ) \geq \sum _ { t = 1 } ^ { n } ( 1 - p _ { t } ) \geq { \frac { 2 n } { 3 } } \geq { \frac { 1 } { 1 1 5 2 d \rho } } .\tag{F.7}
$$

Combining the two terms $\operatorname { I f } d { \sqrt { T } } \geq 1 / 2$ , then $1 / ( d \rho ) \leq 2 \sqrt { T } / \rho .$ so (F.2) gives $R _ { T } \geq ( \sqrt { T } / \rho + 1 / ( d \rho ) ) / 9 6$ in instance $M _ { 2 }$ of the first construction.

Otherwise, $T \geq 1 / ( d \rho )$ implies $d / \rho \leq d ^ { 2 } T < 1 / 4$ , and $\sqrt { T } / \rho < 1 / ( 2 d \rho )$ . The second construction therefore applies, and (F.7) gives $R _ { T } \geq ( \sqrt { T } / \rho + 1 / ( d \rho ) ) /$ 1728 in its instance $M _ { 1 }$ . Both cases establish the stated lower bound with probability at least $3 / 4 - 2 \delta$ □