# TAIL-INFLUENCE SAMPLING FOR CVAR POLICY EVALUATION

Pauline Bourigault Imperial College London

Xiaotong Ji Huawei Noah’s Ark Lab

Matthieu Zimmer Huawei Noah’s Ark Lab

Rasul Tutunov Huawei Noah’s Ark Lab

Haitham Bou-Ammar UCL Centre for AI

## ABSTRACT

Policies with similar mean returns can differ sharply in rare failures, yet estimating lower-tail conditional value-at-risk (CVaR) accurately can require many costly rollouts. When different conditional components of a stochastic workflow can be queried separately, we ask how to allocate a fixed evaluation budget to estimate a fixed policy’s CVaR most accurately. We derive a tail influence for each queryable conditional law that aggregates how its uncertainty affects CVaR across every Bellman reuse. Its variance yields the fixed-design efficiency bound and the oracle Neyman allocation. Tail-Influence Sampling (TIS) estimates these influence scales from a pilot model and reallocates fresh queries toward kernels that matter most for the tail; a visitation-anchored variant protects against pilot underallocation. Under fixed dimension and a positive quantile margin, TIS attains oracle asymptotic variance and first-order MSE including pilot cost, while the anchored variant is within a factor two of the oracle. We also characterize an exact-grid regime in which tail- and mean-optimal allocations coincide. On CliffWalking, TIS reduces MSE by 41% versus learned occupancy and 76% versus complete rollouts at the same charged transition budget. In frozen language-model review workflows, anchored TIS beats an equally regularized mean-influence blend in 23 of 24 MMLU-Pro settings and reaches 2.4 − 3.4× lower MSE than rollouts on six-call FinQA reviews.

## 1 INTRODUCTION

Average performance can conceal the failures that matter most. A language-model workflow may answer correctly on most runs yet occasionally produce a severe numerical error. Similarly, a robot may complete a task reliably yet rarely enter a state that causes damage. Lower-tail conditional value-at-risk (CVaR) measures average return in a specified worst fraction of runs. Estimating this tail average accurately by Monte Carlo can require many complete runs. Such estimates decide which policy or workflow is safe to deploy. For language-model workflows each run costs several model calls, so the evaluation budget, not the model, often limits how reliably rare severe failures can be measured. In resettable simulators and some generative workflows, however, we can directly sample what happens next from a chosen state and action. This gives the evaluator control over where to collect information. Given a limited evaluation budget, which state–action pairs should we query to estimate a fixed policy’s CVaR most accurately?

Fixing the policy does not tell us the transition and reward probabilities, which must be learned from queries. Sampling every state–action pair equally ignores their different roles in producing failure, while sampling according to visitation concentrates the budget on frequently encountered pairs. Neither strategy necessarily identifies where uncertainty about the tail originates. A frequently vis ited state may have predictable outcomes, while uncertainty at a rarer state can change the estimated severity of failures. This is the central difficulty in evaluating rare failures of language-model agents: runs seldom reach the states that produce the worst outcomes, such as a step at which a tool returned a wrong value, so complete rollouts spend almost all of their budget elsewhere. Effective allocation must account for both outcome variability and its effect on the final tail estimate.

Our answer is a computable notion of tail influence: how uncertainty in a conditional law, or kernel, affects uncertainty in the final CVaR estimate. A kernel receives a high score when its possible outcomes vary substantially and those differences meaningfully change the estimated severity of the worst runs. Its CVaR effect depends on the continuation model and tail cutoff, and the same kernel can be reused at multiple stages. For example, the same review prompt may recur in a languagemodel workflow, or a control policy may revisit a state. One observation of this shared behaviour therefore provides information about several parts of the return calculation simultaneously. We combine these effects before measuring their variability to obtain one allocation score per kernel.

Tail-Influence Sampling (TIS) uses a uniform pilot to fit the conditional laws, computes influence scales in the fitted model, and allocates fresh queries in proportion to those scales, with a uniform exploration share. A short pilot can nevertheless miss a rare consequential outcome, causing TIS to underestimate a kernel’s importance and give it too few additional queries. Such an error can dominate the final CVaR estimate because the neglected kernel may be precisely where the worst outcomes originate. Anchored TIS averages the influence-based allocation with visitation-based shares, mitigating this failure when visitation remains informative. This follows the established defensive-mixture principle (Hesterberg, 1995; Owen & Zhou, 2000): combine a targeted design with a broader reference design. Under the same conditions, the anchor’s asymptotic MSE is at most twice the oracle’s.

A tail-specific score is not always needed. On an exact categorical grid, if the probability of a submaximal return is below the tail level, the worst runs contain every submaximal return plus some maximal returns; CVaR then moves exactly with the mean return, and tail influence becomes proportional to mean influence. The divergence between the two scores indicates where tail-specific allocation can pay off: estimated from the pilot, it tells an evaluator whether to allocate by tail influence with the defensive anchor or whether mean-based allocation suffices. The payoff is practical: at matched accuracy, TIS needed 2–4 times fewer queries than complete rollouts on our tabular benchmarks, and the anchor 2–10 times fewer than uniform sampling on held-out FinQA.

Prior distributional-RL inference and efficiency results analyze estimation under specified sampling laws (Zhang et al., 2025; Cheng et al., 2026), adaptive stratification learns allocations for fixed within-stratum quantities (Etor<sup>´</sup> e & Jourdain´ , 2010; Carpentier et al., 2015), and trajectory designs such as ReVar target mean policy evaluation (Mukherjee et al., 2022). Here the allocation score itself depends on an unknown Bellman continuation model and CVaR cutoff; we derive it and show that learning it together with the allocation recovers oracle first-order MSE, including pilot cost.

In short, our contributions can be stated as follows: (i) A CVaR-specific allocation signal. We derive each shared kernel’s influence through its Bellman uses. Its variance gives a fixed-design efficiency bound and the scales for classical Neyman allocation (Neyman, 1934). (ii) Learning the score and allocation. Under fixed dimension, a positive quantile margin, and suitable pilot and exploration schedules, TIS learns the continuation model, cutoff, and influence scales while attaining oracle asymptotic variance and first-order mean-squared error (MSE), including pilot cost. We also characterize the anchor’s asymptotic MSE under the same conditions, within a factor two of the oracle. (iii) When tail specificity matters. Models with identical visitation, reward moments, and return laws can need different allocations; on exact grids in a rare-failure regime, tail- and mean-optimal allocations coincide, and their divergence serves as a diagnostic. (iv) Experiments. CliffWalking and an 18-case inventory disruption family show gains over learned occupancy, learned mean influence, and rollouts at matched query budgets. On language-model workflows with frozen laws, gains over occupancy and rollouts depend on the generator, budget, and workflow length. Blending controls show that the tail score, not added regularization, drives the anchor’s gains; the pilot-estimated divergence selects between tail- and mean-based allocation; and in longer review loops the anchor beats complete rollouts at matched cost.

## 2 PROBLEM AND ESTIMATOR

We evaluate a fixed policy in a finite-horizon Markov decision process (MDP) with stationary dynamics. The state and action spaces S, A are finite, H is the horizon, and $s _ { 0 }$ is the initial state.

![](images/d53c1cc9b3e69b76b0ce5c2fe187858c6d9928354cc7f6b7526d659b4c67bb45.jpg)  
Figure 1: Equal visitation and reward variability can conceal different information about CVaR. The same kernel can be reused across many stages (Use $1 , . . . ,$ Use H). A and B are visited equally and have identical reward means and variances, but only A’s returns can fall below the CVaR cutoff $q _ { \alpha }$ . TIS sums each pilot draw’s first-order CVaR effects across all uses, then measures their spread ${ \widehat { \sigma } } _ { g }$ across draws; orange rings track one draw. With 20% uniform exploration, the illustrated scores give nominal fresh-query shares of 90%/10% for A/B, versus 50%/50% for visitation or mean influence (before minimum counts and rounding).

With $\textit { h } = \textit { H } - \textit { t }$ steps remaining, the policy selects $A _ { t } \ \sim \ \pi _ { h } ( \cdot \ | \ S _ { t } )$ and the environment draws $( R _ { t } , S _ { t + 1 } ) \sim \bar { P _ { S _ { t } , A _ { t } } }$ , with ${ \bar { R } } _ { t } \in [ 0 , 1 ] .$ . One transition takes $( h , s )$ to $( h - 1 , s ^ { \prime } ) ; h = 0$ ends the run. The policy may depend on h, while $P _ { s , a }$ does not. Starting at $S _ { 0 } ~ = ~ s _ { 0 }$ , total return is $\begin{array} { r } { G _ { H } ( s _ { 0 } ) = \sum _ { t = 0 } ^ { H - 1 } R _ { t } \ \in \ [ 0 , H ] } \end{array}$ . For a chosen level $\alpha \in ( 0 , 1 )$ , our goal is to estimate $\mathrm { C V a R } _ { \alpha } ( G _ { H } ( s _ { 0 } ) ) \mathrm { . }$ : the average return in the worst $\alpha \cdot$ -fraction of runs, taking only the required fraction of probability mass at the cutoff. Smaller α emphasizes rarer outcomes. Appendix B.2 gives the formal definition.

What can be sampled? A query group $g = ( s , a ) \in \mathcal { G }$ is one independently queryable conditional law, or kernel. One query returns $W _ { g } = ( R , S ^ { \prime } ) \sim P _ { g } ;$ reward and next state may be dependent. Queries are $\mathrm { i . i . d }$ . within a group and independent across groups. For example, an evaluator can reconstruct a review prompt from a specified answer and confidence, then sample its response without reaching that state by rollout. We retain a fixed set of states and $G = | { \mathcal { G } } |$ groups, closed under all possible continuations. Allocating $n _ { g }$ queries to each group costs $\begin{array} { r } { N = \sum _ { g } n _ { g } ; } \end{array}$ a rollout costs one query per transition. The same fitted law $\widehat { P } _ { g }$ is used at every remaining horizon where that group occurs. Thus $g \ = \ ( s , a )$ identifies what we sample, while $\bar { ( h , s ) }$ identifies where we evaluate its consequences. The same next state can have different continuation returns with one or three steps left; these differences matter for the allocation score.

How do samples give a CVaR estimate? We first estimate the return distribution from each state with h steps left. Following categorical distributional RL (Bellemare et al., 2017), we store probabilities on a fixed return grid $\mathcal { Z } = \{ 0 = z _ { 0 } < \cdots < z _ { K - 1 } = H \}$ with maximum gap ∆. The vector $p _ { h , s } ^ { * }$ contains these probabilities: $p _ { h , s , k } ^ { * }$ is the mass at return $z _ { k }$ . The indices mean remaining steps (h), current state (s), and return value (k).

A Bellman update combines the immediate reward with the distribution of future returns. The matrix $Q ( R )$ adds R to each grid value and projects the result onto the grid, splitting mass between neighboring points and clipping outside the endpoints (Bellemare et al., 2017; Rowland et al., 2018). Averaging over actions and next transitions gives

$$
p _ { h , s } ^ { * } = \sum _ { a } \pi _ { h } ( a \mid s ) \mathbb { E } _ { ( R , S ^ { \prime } ) \sim P _ { s , a } } \bigl [ Q ( R ) p _ { h - 1 , S ^ { \prime } } ^ { * } \bigr ] , \qquad p _ { 0 , s } ^ { * } = e _ { 0 } ,\tag{1}
$$

where $e _ { 0 }$ puts all mass at zero. Replace each expectation by a group sample average and evaluate successively for $h = 1 , \ldots , H$ . CVaR of the estimated root probabilities is ${ \widehat { C } } _ { N } ; C _ { \alpha , K }$ uses the population probabilities (Appendix A). Allocation controls sampling error around $C _ { \alpha , K }$ . The grid introduces a separate approximation error of at most $H \Delta$ relative to true-return CVaR (Appendix E.1). All limits keep the horizon, state and action sets, grid, tail level, and retained groups fixed. Stars denote population quantities, hats estimates, and (0) pilot estimates.

## 3 FROM ONE OBSERVATION TO AN OPTIMAL ALLOCATION

More queries to a group help only if uncertainty in that group affects the final CVaR estimate. We derive that effect in three steps: write CVaR using an expected shortfall, trace a transition error to that shortfall, then allocate samples according to the variability of the resulting effects.

Read CVaR as a shortfall. For a grid threshold $q \in { \mathcal { Z } } ,$ , the shortfall $( q - Z ) _ { + }$ is zero above q and measures the distance below it. Let $U _ { h } ^ { * } ( s , q ) \stackrel { - } { = } \mathbb { E } [ ( q - Z _ { h } ( s ) ) _ { + } ]$ , where $Z _ { h } ( s )$ has the grid probabilities $p _ { h , s } ^ { * } .$ . At the root α-quantile $q _ { \alpha }$ , the standard shortfall identity gives

$$
C _ { \alpha , K } = q _ { \alpha } - \frac { U _ { H } ^ { * } ( s _ { 0 } , q _ { \alpha } ) } { \alpha } .\tag{2}
$$

For probabilities (.04, .08, .88) at returns $( 0 , . 5 , 1 )$ , the worst 10% contains all zero returns and enough .5 returns to average .3. Here $q _ { \alpha } = . 5$ and the mean shortfall is $. 0 4 \times . 5 = . 0 2 ;$ the formula gives . $5 - . 0 2 / . 1 = . 3$ . We assume a positive quantile margin: the tail cutoff lies strictly inside the probability mass at one grid value. In the example, .04 $< . 1 < . 1 2 .$ , so small probability errors leave $q _ { \alpha } = . 5$ unchanged. With that threshold unchanged, CVaR error is shortfall error scaled by $- 1 / \alpha .$ This lets us focus on one expected shortfall. Appendix B.2 defines the margin and explains why a zero margin can invalidate the Gaussian limit.

Trace a transition error to CVaR. After observing reward $R ,$ the remaining shortfall threshold is $q - R$ . The sampled target $T _ { h } ( W ; U )$ therefore uses $U _ { h - 1 } ( S ^ { \prime } , q - R )$ , interpolating between grid thresholds and taking zero at nonpositive thresholds; at $h = 1$ it is $( q - R ) _ { + }$ . Categorical projection preserves these shortfalls at grid thresholds (Lemma 1), so this is the same return calculation in more convenient coordinates.

Write all $( h , s , q )$ shortfalls as the vector $U ^ { * }$ . To learn how errors in them affect the initial state’s shortfall, compute sensitivity weights r, also called adjoint weights. The two Bellman passes can be written as

$$
U ^ { * } = \mathbf { b } + M U ^ { * } , \qquad ( I - M ) ^ { \top } r = e _ { x _ { 0 } } , \quad x _ { 0 } = ( H , s _ { 0 } , q _ { \alpha } ) .\tag{3}
$$

Here b contains terminal contributions, M carries continuation weights, and $e _ { x _ { 0 } }$ selects the root shortfall. The first equation computes shortfalls; the second assigns a weight to each coordinate according to its effect on the root. Both are computed by passes through the layers, without a dense matrix inverse.

For one observation W from group $^ { g , }$ let $\mathcal { T } _ { g } ( W ; U ^ { * } )$ collect its updates wherever that group is used, including policy weights. Subtracting the expected updates gives the one-draw error $\Xi _ { g } ( W ) =$ $\mathcal { T } _ { g } ( W ; \breve { U } ^ { * } ) - \bar { \mathbb { E } } \breve { \mathcal { T } } _ { g } ( \breve { W } ; U ^ { * } )$ . Weighting by r translates this error to the root, and $- 1 / \alpha$ converts it to CVaR error:

$$
\phi _ { g } ( W ) = - \frac { 1 } { \alpha } r ^ { \top } \Xi _ { g } ( W ) , \qquad \sigma _ { g } ^ { 2 } = \mathbb { E } \big [ \phi _ { g } ( W ) ^ { 2 } \big ] .\tag{4}
$$

Thus $\phi _ { g }$ is one draw’s first-order effect on CVaR, and $\sigma _ { g }$ measures how much that effect varies across draws. A shared kernel’s observation affects several stages together. We sum those effects before taking their variance, retaining the cross-stage covariance of the same draw (Equation 16).

Figure 1 illustrates this calculation; Appendix B.2 gives a numerical example and its covariance calculation.

Allocate where more samples reduce error. Averaging $n _ { g }$ independent observations reduces group $\boldsymbol { g ^ { \prime } } \mathbf { s }$ leading variance contribution to $\sigma _ { g } ^ { 2 } / n _ { g }$ . Contributions from independently sampled groups add. The next theorem formalizes the resulting prediction: CVaR MSE is approximately $\textstyle \sum _ { g } \sigma _ { g } ^ { 2 } / n _ { g } =$ $V ( w ) / N$ , where $w _ { g }$ is the group’s budget share.

Theorem 1 (Allocation-dependent limit). Under the fixed-dimensional model and positive margin above, let the counts be deterministic with ${ n _ { g } / N  w _ { g } > 0 }$ . Then

$$
\sqrt { N } \Bigl ( \widehat { C } _ { N } - C _ { \alpha , K } \Bigr ) \Rightarrow \mathcal { N } \bigl ( 0 , V ( w ) \bigr ) , \qquad N \mathbb { E } \Bigl [ ( \widehat { C } _ { N } - C _ { \alpha , K } ) ^ { 2 } \Bigr ]  V ( w ) , \quad V ( w ) = \sum _ { g } \frac { \sigma _ { g } ^ { 2 } } { w _ { g } } .\tag{5}
$$

Increasing a high-influence group’s share reduces its contribution to error. The proof also controls the nonlinear Bellman remainder and wrong-quantile probability in normalized MSE $( \mathsf { A p - }$ pendix C.1).

Theorem 2 (Fixed-design efficiency). For each such allocation and population law satisfying the positive quantile margin, $V ( w )$ is the local semiparametric efficiency bound for $C _ { \alpha , K }$ in the product model of unrestricted group laws on their fixed declared outcome spaces, relative to its differentiable-in-quadratic-mean $( D Q M )$ tangent space. The empirical categorical Bellman estimator is regular under these local submodels and attains the bound.

For a fixed budget split, this is the smallest leading variance among regular estimators using the stated conditional samples (Appendix C.2). We can now choose the split itself. When $\begin{array} { r } { \sum _ { g } \sigma _ { g } > 0 } \end{array}$ minimizing $V ( w )$ gives the classical Neyman rule

$$
w _ { g } ^ { * } = \frac { \sigma _ { g } } { \sum _ { j } \sigma _ { j } } , \qquad V ^ { * } = \left( \sum _ { g } \sigma _ { g } \right) ^ { 2 } .\tag{6}
$$

A group with twice the influence scale receives twice the oracle share. If some scales vanish, the optimum over positive shares is approached by letting their exploration shares tend to zero. The next example shows why ordinary visitation and reward variability cannot replace this tail-specific score. In a one-step model with equally visited kernels, one returns 0 with probability .1 and $5 / 9$ otherwise; each other kernel returns $1 / 3$ or $2 / 3$ with equal probability. Their first two moments match, but only the first has random shortfall below the CVaR threshold $1 / 3 .$ . Appendix C.4 gives the proof and fixed-root-law family.

Proposition 1 (Equal visitation and reward variance can hide tail influence). For everyfixed $G \geq 2 ,$ there is a one-step categorical model with a uniform policy, conditional mean $1 / 2$ and variance $1 / 3 6$ in every kernel, and a positive margin at $\alpha = . 1$ , for which

$$
V _ { \mathrm { o c c u p a n c y } } = V _ { \mathrm { m e a n } } = { \frac { 1 } { G } } , \qquad V ^ { \ast } = { \frac { 1 } { G ^ { 2 } } } .
$$

These are the coefficients of $1 / N$ in asymptotic CVaR MSE; $V _ { \mathrm { m e a n } }$ uses the allocation optimal $f o r$ estimating the mean return. There is also a family with these same visitation probabilities, conditional moments, and entire root return law in which the occupancy-to-oracle ratio ranges from 1 to G.

With ten kernels, the oracle thus has one tenth of uniform’s leading MSE.

## 4 TAIL-INFLUENCE SAMPLING

The oracle shares require the unknown transition laws and their effects on future returns. TIS learns both from a pilot. Algorithm 1 (Appendix D) draws $m = m _ { N }$ samples per group, fits a provisional model, and evaluates it by Equation 1. This gives the pilot root quantile; Equation 3 then gives the shortfalls $\widehat { U } ^ { ( 0 ) }$ and sensitivity weights $\widehat { r } ^ { ( 0 ) }$ needed to score each draw.

Score each draw, then measure variability. For pilot outcome $W _ { g , i } ,$ compute

$$
\widehat { d } _ { g , i } = - \frac { 1 } { \alpha } \widehat { r } ^ { ( 0 ) \top } \mathcal { T } _ { g } ( W _ { g , i } ; \widehat { U } ^ { ( 0 ) } ) , \qquad \widehat { \sigma } _ { g } = \sqrt { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( \widehat { d } _ { g , i } - \bar { d } _ { g } ) ^ { 2 } } ,\tag{7}
$$

Here $\bar { d } _ { g }$ is the group’s average score. Each score combines all stage effects of a draw; larger estimated scales receive more queries:

$$
\widehat { w } _ { g } = ( 1 - \lambda _ { N } ) \frac { \widehat { \sigma } _ { g } } { \sum _ { j } \widehat { \sigma } _ { j } } + \frac { \lambda _ { N } } { G } , \qquad 0 < \lambda _ { N } < 1 .\tag{8}
$$

When all scales vanish, use uniform sampling. For scales 1 and 3, the shares before the floor are $1 / 4$ and $3 / 4$ . The floor reserves samples for groups the pilot may have underestimated.

After two main samples per group, largest-remainder rounding spends the remaining budget exactly. Only fresh main samples enter the final estimate. As N grows, a larger pilot can consume a shrinking budget fraction. Under the following schedules, TIS attains oracle leading MSE, including pilot cost.

Theorem 3 (Oracle adaptation). Under the same model and margin, suppose $\begin{array} { r } { \sum _ { g } \sigma _ { g } > 0 . I f m _ { N } \to } \end{array}$ $\infty , G m _ { N } = o ( N ) , \lambda _ { N } \to 0 , \sqrt { N } \lambda _ { N } \to \infty , a n d \log ( 1 / \lambda _ { N } ) = o ( m _ { N } )$ , then

$$
\begin{array} { r } { \sqrt { N } \Big ( \widehat { C } _ { N } ^ { \mathrm { T I S } } - C _ { \alpha , K } \Big ) \Rightarrow { \cal N } ( 0 , V ^ { * } ) , \qquad { \cal N } \mathbb { E } \Big [ \Big ( \widehat { C } _ { N } ^ { \mathrm { T I S } } - C _ { \alpha , K } \Big ) ^ { 2 } \Big ]  V ^ { * } . } \end{array}\tag{9}
$$

A total pilot oforder $N ^ { 2 / 3 }$ and floor $N ^ { - 1 / 4 }$ suffice for fixed G. Individual zero-influence groups are allowed.

The theorem controls learning the score through the unknown continuation model and quantile, as well as learning its variance and allocation. The proof handles rare underallocating pilots, the nonlinear Bellman remainder, and wrong quantile atoms (Appendix D.3). The limit is pointwise in the fixed model, not uniform over increasingly rare events or shrinking margins.

A defensive anchor for finite pilots. A small pilot can miss a consequential outcome and give its group too few main samples. Local stability near the oracle (Proposition 3) does not control that failure. Following the established defensive-mixture principle (Hesterberg, 1995; Owen & Zhou, 2000), we combine tail targeting with pilot-estimated visitation. This can protect kernels that the pilot recognizes as frequently reached even when their tail influence is underestimated. Let $\widehat { o } _ { s , a } =$ $\textstyle \sum _ { h } { \widehat { \mu } } _ { h } ( s ) \pi _ { h } ( a \mid s )$ be the pilot-model expected visit count. Form a floored occupancy design from these scores, using the same pilot and floor as the influence design. Anchored TIS averages the two:

$$
\begin{array} { r } { w ^ { \mathrm { a n c } } = \frac { 1 } { 2 } ( w ^ { \mathrm { i n f } } + w ^ { \mathrm { o c c } } ) . } \end{array}\tag{10}
$$

The influence component keeps its uniform fallback. Use these shares in Algorithm 1. The mixture keeps at least half of either component’s share for every group. Before rounding, its leading variance is therefore at most twice that of the better component (Proposition 4, Appendix D.4). Both components can still underallocate the same group. Under the assumptions and schedules of Theorem 3, the anchor’s asymptotic MSE constant satisfies $V ^ { * } \leq V _ { \mathrm { a n c } } \leq 2 \dot { V } ^ { * }$ (Corollary 1, Appendix D.4). This does not guarantee lower finite-budget MSE. Plain TIS is the efficient limit when pilots estimate the scales reliably; the anchor pays at most a factor two in the constant for protection against pilots that miss rare outcomes. Pilot underallocation in the initial language-model experiments motivated this anchor. We fixed its design before collecting held-out FinQA numerical-review data (Section 5).

When can a tail-specific score help? Learning an allocation must repay its pilot. Let $\rho$ be the pilot’s fraction of the budget, $A _ { v } \ \stackrel { \cdot } { = } \ V ( v ) / V ^ { * }$ the oracle’s advantage over a fixed design v that spends all N queries, and $\begin{array} { r } { { \bf \tilde { \cal D } } ( w ^ { * } \| w ) = \sum _ { q } ( w _ { q } ^ { * } - w _ { g } ) ^ { 2 } / w _ { g } } \end{array}$ the error of the realized main-sample shares w relative to the oracle shares $w ^ { * }$ in Equation 6. For the leading error term, learning beats v exactly when

$$
1 + \mathbb { E } D ( w ^ { * } \| w ) < ( 1 - \rho ) A _ { v }\tag{11}
$$

(Appendix D.5). Large allocation advantages, small pilots, and accurate shares favor learning. Because D divides by $w _ { g } .$ , a starved group is especially costly; the anchor guards against exactly this. A further limit is structural.

Proposition 2 (Rare failures make CVaR a mean). Suppose the categorical grid contains every partial return permitted by the fixed declared outcome spaces, so the recursion is exact throughout the product model. Let $X \backslash : = \dot { G } _ { H } ( s _ { 0 } )$ ) and let $x ^ { \star }$ be the maximum return permitted by those outcome spaces. Write $\pi = \operatorname* { P r } ( X < x ^ { \star } ) . { } I f \pi < \alpha ,$ , then

$$
\operatorname { C V a R } _ { \alpha } ( X ) = { \frac { \mathbb { E } [ X ] - ( 1 - \alpha ) x ^ { \star } } { \alpha } } ,
$$

and locally every kernel’s categorical-CVaR influence is $1 / \alpha$ times its ordinary mean-return influence. Hence the tail- and mean-optimal Neyman allocations coincide whenever the influence scales are not all zero; ifthey are all zero, every allocation has zerofirst-order variance.

The worst α-fraction of runs then contains every submaximal return plus enough maximal returns to fill the tail, so CVaR moves exactly with the mean. A mean score is also easier to learn, since it does not depend on a quantile. The total-variation distance ${ \begin{array} { l } { { \frac { 1 } { 2 } } \sum _ { g } | p _ { g } - m _ { g } | } \end{array} }$ between normalized tail and mean influence shares (using the uniform vector for an all-zero score) is zero under the proposition and serves as a diagnostic (Appendix E.6).

Table 1: TIS gains under concentrated tail influence; elsewhere its pilot does not repay. MSE/uniform MSE $( G = 1 0 , \alpha = . 1 ) ;$ columns: queries/kernel incl. pilots. Uniform=known occupancy; occupancy+pilot isolates pilot cost; Oracle+floor=population scores. Bold: sample-only column minima (point estimates). Table 3: absolute errors/SEs.
<table><tr><td></td><td colspan="2">t = 0 (equal)</td><td colspan="2"> $\begin{array} { r } { t = \frac { 1 } { 2 } } \end{array}$ </td><td colspan="2">t = 1 (one active)</td></tr><tr><td>Method</td><td>100</td><td>400</td><td>100</td><td>²400</td><td>100</td><td>400</td></tr><tr><td>Uniform</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Occupancy + pilot</td><td>1.60</td><td>1.31</td><td>1.67</td><td>1.39</td><td>1.69</td><td>1.31</td></tr><tr><td>Learned mean</td><td>1.63</td><td>1.28</td><td>1.71</td><td>1.39</td><td>1.87</td><td>1.34</td></tr><tr><td>Complete rollout</td><td>1.01</td><td>0.95</td><td>0.99</td><td>1.07</td><td>1.11</td><td>1.18</td></tr><tr><td>TIS</td><td>5.42</td><td>4.30</td><td>3.62</td><td>3.15</td><td>0.22</td><td>0.16</td></tr><tr><td>anchored TIS</td><td>2.04</td><td>1.52</td><td>1.66</td><td>1.17</td><td>0.37</td><td>0.27</td></tr><tr><td>Oracle + floor</td><td>1.00</td><td>1.00</td><td>0.75</td><td>0.77</td><td>0.13</td><td>0.12</td></tr></table>

## 5 EXPERIMENTS

We ask five questions: Q1 Can visitation and mean-return scores miss tail influence? Q2 Does learning this signal improve matched-budget accuracy? Q3 Why can pilots fail, and does anchoring help? Q4 Does anchoring transfer to held-out numerical-review workflows? Q5 Is the gain tailscore-specific, and does the divergence diagnostic predict where? Theory predicts three regimes: plain TIS should win with concentrated tail influence and reliable pilots (Q1–Q2); the defensive anchor should matter when pilots miss rare outcomes (Q3–Q4); and, on this diagnostic’s exact closed grids, no tail-specific gain should appear when submaximal returns are rarer than the tail level (Q5).

Compared methods. Uniform splits queries evenly. Learned occupancy/mean use pilot-estimated visit counts/mean-return influence, respectively, with TIS’s pilot, floor, and CVaR estimator. Complete rollouts use whole trajectories, not conditional-query methods’ direct kernel access; comparisons are cost-matched, not equal-access. Oracle+floor: exact tail-influence scales, no pilot. Budgets include discarded pilots and all rollout transitions; MSE uses exact closed-grid targets (Appendix E.2).

Q1. Controlled separation. Table 1 varies tail-influence concentration while preserving visitation, conditional moments, and root return law. At t = 1, plain TIS is .16–.22 of uniform MSE (oracle .12–.13); at t = 0 (uniform optimal), it is 4.3–5.4× uniform MSE. The intermediate case likewise does not repay the pilot (Appendix D.5).

Q2. Tabular benchmarks. Seasonal inventory (H = 8, 41 blocks, $\alpha = . 1 )$ uses the prespecified pilot/floor; at 1,200 queries/block, MSE/uniform is .657 for TIS, .918 for learned occupancy, .866 for learned mean influence, and 2.22 for complete rollouts at matched transition cost (Figure 2(b), Table 4). Slippery CliffWalking $( H = 2 0 , \alpha = . 1 $ , 149 stationary kernels) tests layer reuse (Figure 2(a)). At 400 queries/kernel, TIS lowers MSE by 40.9% vs. learned occupancy, 18.3% vs. learned mean influence, 76.3% vs. complete rollouts at matched transition cost, and 31.3% vs. population occupancy (Table 6). FrozenLake/rainy Taxi add shared-kernel checks (Appendix E.2.5). Most gains come from pooling reused kernels; covariance terms matter little numerically here (Appendix E.2.4). To test breadth, we prespecified an 18-case family before simulation: disruption probabilities {.01, .04, .12}, disruption losses 1–3 units and two fixed ordering policies; all 18 reported (Appendix E.4). At 1,200 queries/block, both TIS and the anchor have resolved lower MSE vs. learned occupancy/rollouts in all 18 and vs. learned mean in 17 (median MSE/uniform: .61 TIS, .69 anchored, .93 learned occupancy, .84 learned mean, 2.56 rollouts). At matched RMSE, TIS’s query ratios (learned occupancy/rollouts) are .65–.84/.32–.45 on CliffWalking and .55–.72/.26–.30 on inventory; these retrospective interpolations include pilots (Appendix E.2.7).

Q3. Pilot reliability in language-model workflows. Each of 50 fixed MMLU-Pro questions (Wang et al., 2024) is a separate workflow. State $s = ( j , c )$ records latest answer $j \in \{ 1 , \dots , 1 0 \}$ and confidence $c \in \{ . 1 , . . . , . 9 \}$ . Fixed policy solves first, then selects reconsider, challenge, or verify by confidence band. Prompts include question/current pair/prescribed action, but no history or stage index. Root plus $1 0 \times 9$ pairs yield 91 directly queryable kernels reused across stages.

![](images/1dd18f8ecefc0e19cc65ee174ad9f099bf827a22c35f5864c4cc79d10613b690.jpg)  
Figure 2: At largest budgets, TIS MSE is lower than learned occupancy, learned mean, and rollouts. CVaR MSE vs. charged queries/kernel in (a) CliffWalking and /block in (b) inventory. Conditional-query methods have direct kernel access; rollouts use trajectories at same charged transition budget. Oracle+floor=population scores; bars=1.96 Monte Carlo SEs.

Runs use $H \in \{ 2 , 4 , 6 \}$ calls. Responses earn normalized Brier utility vs. correct option (Equation 65); for six generators, we estimate each question’s worst-10% CVaR against exact closed-grid targets (Appendix E.2.6). Queries sample frozen LLM laws; live GPU/API cost is out of scope.

At $H = 6 ,$ 400 queries/kernel, plain TIS exceeds uniform MSE for Qwen3-4B (1.38) and GLM-4-32B (1.77) (Table 7). Occupancy has lower observed MSE than plain TIS for all six generators. The anchor mitigates both failures, reaching .056–.093 of uniform MSE and the lowest MSE for 5/6 generators; occupancy and complete rollouts remain strong. Kernel reuse matters: fitting recurringprompt copies separately by step matches shared TIS at $H = 2$ but has 4–10× its MSE at $H = 4 .$ 6 (Table 9). The MMLU replay’s 90th-percentile realized/oracle variance ratio falls from 136 (plain TIS) to 5.8 (anchor) (Appendix Figure 5(a)): starved groups drive failures, as Equation 11 predicts. Using population influences and realized counts, the variance formula predicts observed MSE without fitted constants (median log-ratios −.001 for MMLU and −.006 for FinQA Phi; retrospective check, Appendix E.10).

Q4. Held-out financial numerical review. FinQA has numerical questions over real financial reports (Chen et al., 2021). We compare ordinary review and explicit unit-and-sign audit. Each workflow selects one of eight frozen candidate solutions, then reviews twice $\left( H \overset { = } { = } 3 \right)$ under the MMLU confidence-band policy. Root plus $8 \times 9$ candidate–confidence pairs yield 73 queryable kernels; review templates alter transition laws. Terminal utility $u ( s ) \in \mathsf { [ 0 , 1 ] }$ falls with relative numerical error against gold answer; invalid candidates earn zero. With zero root utility, rewards $R = [ u ( s ^ { \prime } ) - u ( \bar { s } ) + 1 ] \bar { / } 2$ sum to $G _ { H } = [ H + u ( S _ { H } ) ] / 2$ , so only final-answer utility matters. A 20-question development phase fixed the screen, generators, workflows, seeds, anchored score, and floor before held-out calibration on 50 fresh screened questions from 312 scanned (Appendix E.3). At 100–200 queries/kernel, anchored TIS is .089–.100 of uniform MSE on Qwen3-4B and .216– .327 on Phi-4-mini (Figure 4(a,b)); plain TIS is unresolved vs. uniform in 7/8 cells. At 400/800 queries, the anchor beats occupancy and rollouts in both Phi workflows (MSE ratios .81–.86 and .68–.88, respectively); all 8 contrasts resolve. Occupancy and rollouts remain better on Qwen (Appendix E.3), whose near-deterministic questions fall in the rare-failure regime of Proposition 2. With six vs. three calls (same frozen kernels), the Phi anchor’s MSE is .29–.64 of rollouts’ at every budget in both workflows, all resolved (Table 17).

Q5. Is the tail score itself what helps? Two declared controls test whether anchor gains come from the tail score: each blends the same floored occupancy shares with uniform or learned meaninfluence shares and pays the same pilot (Appendix E.5). Anchor MSE is resolved lower vs. uniform blend in 160/166 settings and higher in none (Table 2), so gains are not a regularization artifact. Vs. mean blend, results follow Proposition 2. FinQA, including calculator-fault variants (Appendix E.7): median tail–mean distance is .000; mean blend is as good or better. MMLU-Pro (more frequent failures; distance .08–.33): anchor MSE is resolved lower vs. mean blend in 23/24 confident-error cells (this utility makes confident mistakes nearly worthless) and 14/15 Brier cells at distance $\geq . 2 4$ versus 2/12 below .18. With pre-simulation predictions, the rule held in all 3 decisive new settings and 6/8 longer-loop settings; the two misses (Qwen3-4B at $H = 8 , 1 0 ,$ , distance .26) favored the anchor without resolving (Appendices E.6 and E.8). In practice, use pilot-estimated distance: anchor if pilot median is ≥ .21, otherwise mean blend; this picked the better or tied design in 141/148 cells (Appendix E.9).

Table 2: Anchor vs. equally regularized blends: never resolved worse than uniform; meanblend gains depend on domain. Counts of resolved $( | z | \geq 2 )$ lower (higher) anchor MSE vs. each blend. Inventory: 18 cases at 1,200 queries/block; FinQA: 2 generators × 2 workflows × 4 budgets.
<table><tr><td>Domain</td><td>vs. occupancy+uniform</td><td>vs. occupancy+mean</td></tr><tr><td>Inventory disruption family</td><td>18/18 (0)</td><td>17/18 (0)</td></tr><tr><td>FinQA review (held-out)</td><td>16/16 (0)</td><td>0/16 (15)</td></tr><tr><td>MMLU-Pro, Brier utility (6 generators × 3 budgets)</td><td>18/18 (0)</td><td>11/18 (0)</td></tr><tr><td>MMLU-Pro high-stakes panel (3 generators × 3 budgets)</td><td>9/9 (0)</td><td>5/9 (2)</td></tr><tr><td>MMLU-Pro, confident-error utility (6 generators)</td><td>24/24 (0)</td><td>23/24 (0)</td></tr><tr><td>FinQA with calculator faults (ordinary review)</td><td>15/18 (0)</td><td>0/18 (12)</td></tr><tr><td>FinQA with calculator faults (unit check)</td><td>15/18 (0)</td><td>0/18 (10)</td></tr><tr><td>Prospective divergence test (5 new settings × 3 budgets)</td><td>15/15 (0)</td><td>11/15 (1)</td></tr><tr><td>MMLU-Pro longer loops  $( \dot { H } = 8 , 1 0 ; 3$  generators × 3 budgets)</td><td>18/18 (0)</td><td>12/18 (0)</td></tr><tr><td>FinQA longer reviews (H = 6; 2 generators × 2 workflows × 3 budgets)</td><td>12/12 (0)</td><td>0/12 (9)</td></tr></table>

## 6 RELATED WORK

From distributional inference to query design. Categorical distributional RL supplies the representation and projection (Bellemare et al., 2017; Rowland et al., 2018). Zhang et al. (2025) derive return-law and functional limits under specified, possibly nonuniform, sampling laws. Quantilebased evaluation also has semiparametric efficiency guarantees (Cheng et al., 2026). Shortfall identities and efficiency theory are established (Rockafellar & Uryasev, 2000; van der Vaart, 1998); our addition is the computable, learnable Bellman allocation signal.

Adaptive allocation with a learned Bellman model. Adaptive stratified sampling learns the variances required by Neyman allocation (Neyman, 1934; Etor<sup>´</sup> e & Jourdain´ , 2010; Carpentier et al., 2015). Theorem 3 controls learning our Bellman-dependent score and its allocation, including pilot cost. Small-pilot failures and defensive mixtures motivate the anchor (Cai & Rafi, 2022; Hesterberg, 1995; Owen & Zhou, 2000); SaVeR targets trajectory design for mean evaluation (Mukherjee et al., 2024). Risk-sensitive control changes the policy (Tamar et al., 2015; Bauerle & Ja¨ skiewicz´ , 2024); with generative access, Deng et al. (2025) study sample complexity for iterated-CVaR policy optimization. We instead fix the policy and optimize conditional-query allocation for estimating its tai functional. Appendix F compares access models and guarantees.

## 7 DISCUSSION

Allocating queries to estimate CVaR is harder than allocating them for a mean: a kernel’s value depends on an unknown continuation model and tail cutoff, and one draw of a reused kernel affect several Bellman stages at once. Our contribution is a computable, learnable CVaR allocation signal for reused conditional laws, with oracle first-order MSE under the stated assumptions, a defensive variant within a factor two of the oracle, and, on exact grids, a condition under which mean influence suffices.

TIS is most promising when conditional access is feasible, tail influence differs meaningfully from visitation or mean influence, and pilots estimate that difference reliably enough to repay their cost. By Equation 11, the failures reflect either little opportunity (t = 0 in Table 1) or opportunity the pilot cannot learn (plain TIS in deep loops; FinQA-Qwen, whose rare errors pilots seldom see). In practice, the pilot’s tail–mean divergence selects the allocation, at the same computation as mean influence and more than occupancy or rollouts (Table 13). The setting arises wherever an evaluator can restart from a chosen state: auditing rare severe errors of language-model review and agent loops before deployment, where each query is a model call and recurring prompts make kernel reuse the norm, or disruption losses of inventory and maintenance policies in simulators. Tool failures and multi-turn safety evaluation are natural next targets. The guarantees require fixed-dimensional categorical models, a positive quantile margin, independent queries, and the stated pilot and floor schedules. Finite-budget MSE guarantees remain open.

## REFERENCES

Nicole Bauerle and Anna Ja¨ skiewicz. Markov decision processes with risk-sensitive criteria: An´ overview. Mathematical Methods of Operations Research, 99(1):141–178, 2024. doi: 10.1007/ s00186-024-00857-0.

Nicole Bauerle and Jonathan Ott. Markov decision processes with average-value-at-risk crite-¨ ria. Mathematical Methods of Operations Research, 74(3):361–379, 2011. doi: 10.1007/ s00186-011-0367-0.

Marc G. Bellemare, Will Dabney, and Remi Munos. A distributional perspective on reinforcement´ learning. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pp. 449–458. PMLR, 2017.

Glenn W. Brier. Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3, 1950.

Yong Cai and Ahnaf Rafi. On the performance of the Neyman allocation with small pilots. arXiv preprint arXiv:2206.04643, 2022. Version 4, revised June 2024.

Alexandra Carpentier, Remi Munos, and Andr´ as Antos. Adaptive strategy for stratified monte carlo´ sampling. Journal ofMachine Learning Research, 16(68):2231–2271, 2015.

Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas. Universal off-policy evaluation. In Advances in Neural Information Processing Systems, volume 34, pp. 27475–27490, 2021.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. FinQA: A dataset of numerical reasoning over financial data. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3697–3711. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.emnlp-main.300.

Zijie Cheng, Yang Peng, and Zhihua Zhang. Statistical efficiency and inference of quantile distributional reinforcement learning. arXiv preprint arXiv:2607.08444, 2026.

Jessica Dai, Paula Gradu, and Christopher Harshaw. Clip-OGD: An experimental design for adaptive Neyman allocation in sequential experiments. In Advances in Neural Information Processing Systems, volume 36, pp. 32235–32269, 2023.

Zilong Deng, Simon Khan, and Shaofeng Zou. Near-optimal sample complexity for iterated CVaR reinforcement learning with a generative model. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pp. 3907–3915. PMLR, 2025.

Connor Douglas, Joel Persson, and Foster Provost. Logging policy design for off-policy evaluation. arXiv preprint arXiv:2605.15108, 2026.

Pierre Etor <sup>´</sup> e and Benjamin Jourdain. Adaptive optimal allocation in stratified sampling meth-´ ods. Methodology and Computing in Applied Probability, 12(3):335–360, 2010. doi: 10.1007/ s11009-008-9108-0.

Granite Team, IBM. Granite-4.2-8B model card. Hugging Face model repository, 2026. URL https://huggingface.co/ibm-granite/granite-4.2-8b. Accessed Aug. 26, 2026.

Tim Hesterberg. Weighted average importance sampling and defensive mixture distributions. Technometrics, 37(2):185–194, 1995. doi: 10.1080/00401706.1995.10484303.

Wassily Hoeffding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963. doi: 10.1080/01621459.1963.10500830.

Sungee Hong, Zhengling Qi, and Raymond K. W. Wong. Distributional off-policy evaluation with bellman residual minimization. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, pp. 4006– 4014. PMLR, 2025.

Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. Active testing: Sample-efficient model evaluation. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 5753–5763. PMLR, 2021.

Yang Li, Jie Ma, Miguel Ballesteros, Yassine Benajiba, and Graham Horwood. Active evaluation acquisition for efficient LLM benchmarking. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 35581– 35602. PMLR, 2025.

Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. tinyBenchmarks: Evaluating LLMs with fewer examples. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 34303–34326. PMLR, 2024.

Microsoft, Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, et al. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv preprint arXiv:2503.01743, 2025.

Mistral AI. Mistral-Small-24B-Instruct-2501 model card. Hugging Face model repository, 2025. URL https://huggingface.co/mistralai/ Mistral-Small-24B-Instruct-2501. Accessed Aug. 31, 2026.

Subhojyoti Mukherjee, Josiah P. Hanna, and Robert D. Nowak. ReVar: Strengthening policy evaluation via reduced variance sampling. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pp. 1413–1422. PMLR, 2022.

Subhojyoti Mukherjee, Josiah P. Hanna, and Robert D. Nowak. SaVeR: Optimal data collection strategy for safe policy evaluation in tabular MDP. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 36531–36576. PMLR, 2024.

Jerzy Neyman. On the two different aspects of the representative method: The method of stratified sampling and the method of purposive selection. Journal of the Royal Statistical Society, 97(4): 558–606, 1934. doi: 10.1111/j.2397-2335.1934.tb04184.x.

Phuc Nguyen, Deva Ramanan, and Charless Fowlkes. Active testing: An efficient and robust framework for estimating accuracy. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 3759–3768. PMLR, 2018.

Art Owen and Yi Zhou. Safe and effective importance sampling. Journal ofthe American Statistical Association, 95(449):135–143, 2000. doi: 10.1080/01621459.2000.10473909.

Yang Peng and Liangyu Zhang. Online inference in distributional temporal-difference learning. arXiv preprint arXiv:2608.14408, 2026.

Yang Peng, Liangyu Zhang, and Zhihua Zhang. Statistical efficiency of distributional temporal difference learning. In Advances in Neural Information Processing Systems, volume 37, pp. 24724–24761, 2024. doi: 10.52202/079017-0779.

Yang Peng, Kaicheng Jin, Liangyu Zhang, and Zhihua Zhang. A finite sample analysis of distributional temporal-difference learning with linear function approximation. arXiv preprint arXiv:2502.14172, 2025.

Qwen Team. Qwen3-32B model card. Hugging Face model repository, 2025a. URL https: //huggingface.co/Qwen/Qwen3-32B. Accessed Aug. 31, 2026.

Qwen Team. Qwen3-4B-Instruct-2507 model card. Hugging Face model repository, 2025b. URL https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507. Accessed Aug. 21, 2026.

R. Tyrrell Rockafellar and Stanislav Uryasev. Optimization of conditional value-at-risk. The Journal ofRisk, 2(3):21–41, 2000. doi: 10.21314/JOR.2000.038.

Mark Rowland, Marc G. Bellemare, Will Dabney, Remi Munos, and Yee Whye Teh. An analysis of´ categorical distributional reinforcement learning. In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics, volume 84 of Proceedings of Machine Learning Research, pp. 29–37. PMLR, 2018.

Mark Rowland, Li Kevin Wenliang, Remi Munos, Clare Lyle, Yunhao Tang, and Will Dabney. Near-´ minimax-optimal distributional reinforcement learning with a generative model. In Advances in Neural Information Processing Systems, volume 37, pp. 132774–132823, 2024. doi: 10.52202/ 079017-4221.

Aviv Tamar, Yinlam Chow, Mohammad Ghavamzadeh, and Shie Mannor. Policy gradient for coherent risk measures. In Advances in Neural Information Processing Systems, volume 28, pp. 1468–1476, 2015.

Philip S. Thomas and Erik Learned-Miller. Concentration inequalities for conditional value at risk. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 6225–6233. PMLR, 2019.

Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U. Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulao, Andreas Kallinteris, Markus Krimmel, Arjun KG, Rodrigo Perez-Vicente, An-˜ drea Pierre, Sander Schulhoff, Jun Jet Tai, Hannah Tan, and Omar G. Younis. Gymnasium: A´ standard interface for reinforcement learning environments. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 38, pp. 163114–163129, 2025. doi: 10.52202/085713-4916.

Aad W. van der Vaart. Asymptotic Statistics. Cambridge University Press, Cambridge, 1998.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 37, pp. 95266–95290, 2024. doi: 10.52202/079017-3018.

Runzhe Wu, Masatoshi Uehara, and Wen Sun. Distributional offline policy evaluation with predictive error guarantees. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 37685–37712. PMLR, 2023.

Liangyu Zhang, Yang Peng, Jiadong Liang, Wenhao Yang, and Zhihua Zhang. Estimation and inference in distributional reinforcement learning. The Annals of Statistics, 53(5):1987–2011, 2025. doi: 10.1214/25-AOS2527.

Zhipu AI. GLM-4-32B-0414 model card. Hugging Face model repository, 2025. URL https: //huggingface.co/zai-org/GLM-4-32B-0414. Accessed Aug. 31, 2026.

Yi Zhu, Jing Dong, and Henry Lam. Uncertainty quantification and exploration for reinforcement learning. Operations Research, 72(4):1689–1709, 2024. doi: 10.1287/opre.2023.2436.

## SUPPLEMENT: TIS FOR CVAR POLICY EVALUATION

Appendices A–D develop the allocation theory from the stop-loss Bellman representation to fixeddesign efficiency, learned oracle adaptation, and the anchored safeguard. The main results are proved as follows: Theorem 1 in Appendix C.1, Theorem 2 in Appendix C.2, Proposition 1 in Appendix C.4, Theorem 3 in Appendix D.3, the anchor’s guarantees (Proposition 4 and Corollary 1) in Appendix D.4, and the pilot-payoff condition (Equation 11) in Appendix D.5. Appendix E closes the remaining links to the main paper: Appendix E.1 bounds categorical representation error; the experimental subsections give the protocols and evidence for Q1–Q5; and Appendix E.6 proves Proposition 2 and connects it to the tail–mean diagnostic. Section F positions these results against the closest foundations.

Proof dependencies. Lemma 1 justifies the stop-loss Bellman representation. Lemma 2 controls its empirical remainder and, together with the positive quantile margin, yields Theorem 1. That theorem identifies the influence used in the efficiency calculation of Theorem 2. Lemma 3 controls pilot estimates of the same influence; combined with Lemma 2, it yields Theorem 3. Proposition 4 then transfers the allocation control to the anchored design in Corollary 1.

A Notation and Assumptions 14   
B Allocation and Bellman Identities 15   
B.1 Minimizing the allocation variance 15   
B.2 Projection and categorical CVaR identities 15   
B.3 Exact empirical fixed-point expansion . 16   
C Fixed-Design Limits and Structural Interpretation 18   
C.1 Fixed-design CLT and normalized MSE . 18   
C.2 Semiparametric efficiency 19   
C.3 Untied factorization for the allocation ablations 20   
C.4 Controlled separation with fixed visitation, moments, and root law 21   
D Learned Allocation 22   
D.1 Pilot-scale consistency and lower-tail control 22   
D.2 Finite-pilot design stability 23   
D.3 Oracle adaptation 24   
D.4 Anchored design: variance safeguard and asymptotic MSE . 25   
D.5 When learning an allocation repays its pilot . 27   
E Approximation and Experimental Protocols 28   
E.1 Representation error 28   
E.2 Experimental Evidence and Protocols 28   
E.3 FinQA terminal-risk protocol and full results 36   
E.4 Inventory disruption family 38   
E.5 Blending controls 39   
E.6 Rare failures and the tail–mean coincidence . 39   
E.7 FinQA with calculator faults . 40   
E.8 Longer review loops 41   
E.9 Selecting the allocation from the pilot . 41   
E.10 Mechanism: what the pilot sees and what anchoring changes . 42   
F Additional Related Work 42

## A NOTATION AND ASSUMPTIONS

A query group identifies a sampled law; a Bellman row identifies one use of it. This section makes that distinction precise and connects the probability, measure, and shortfall notation used in the proofs.

Queryable-group experiment. $\mathcal { G }$ is a finite set of conditionally and independently queryable laws $P _ { g } , \dot { G } : = | \mathcal { G } |$ , and the objective is one fixed scalar functional of those laws. Section B.1 optimizes the variance once the groupwise influence scales are identified.

Categorical Bellman conditions. We evaluate one root $( H , s _ { 0 } )$ and let B index the required layerstate rows. A group may feed several rows through known nonnegative mixture coefficients summing to one within each Bellman row. A known component, if present, is represented by a degenerate query law with zero influence. Sub-probability row sums arise in the continuation matrix from termination at nonpositive shifted thresholds. The return law still retains all probability mass. In the stationary model, $g = ( s , a ) , W _ { g } = ( R , S ^ { \prime } ) \sim P _ { s , a }$ , and the coefficient in row $( h , s )$ is $\pi _ { h } ( \boldsymbol { a } \mid \boldsymbol { s } )$ The ordered grid $\mathcal { Z } = \{ z _ { 0 } , \dotsc , z _ { K - 1 } \} \subset [ 0 , H ]$ contains both endpoints and has maximum gap $\Delta$ . The proofs keep $H , | S | , | A | ,$ and K fixed, assume bounded rewards, and impose the positive categorical quantile margin in Equation 21.

Untied special case. For independently queryable layer-specific laws, $g = b = ( h , s )$ and $P _ { b }$ is the policy-mixture law obtained by drawing $A \sim \pi _ { h } ( \cdot \mid s )$ before the transition. One law then feeds one row. Structurally unreachable groups may be removed in either model only when the declared support and fixed policy certify their irrelevance. The retained rows must be closed under every possible continuation transition in the declared outcome spaces. These fixed outcome spaces also define the nonparametric product model in the efficiency theorem.

Equivalent probability and measure forms. The measure $\begin{array} { r } { \eta _ { h } ^ { * } ( s ) = \sum _ { k } p _ { h , s , k } ^ { * } \delta _ { z _ { k } } } \end{array}$ is another notation for the grid probabilities in Section 2; hats denote the empirical version. The categorical target and estimator are

$$
\begin{array} { r } { C _ { \alpha , K } = \mathrm { C V a R } _ { \alpha } \left( \eta _ { H } ^ { * } ( s _ { 0 } ) \right) , \qquad \widehat { C } _ { N } = \mathrm { C V a R } _ { \alpha } \left( \widehat { \eta } _ { H } ( s _ { 0 } ) \right) . } \end{array}\tag{12}
$$

For the shift-and-project matrix in Equation $1 , Q ( R ) _ { k j } = \ell _ { k } ( R + z _ { j } )$ , where $\ell _ { k } ( y )$ is the categorical projection weight on $z _ { k }$ . Column sums are one, including at clipped endpoints. Thus the vector equation is equivalent to

$$
\eta _ { 0 } ^ { * } ( s ) = \delta _ { 0 } , \qquad \eta _ { h } ^ { * } ( s ) = \sum _ { a } \pi _ { h } ( a \mid s ) \mathbb { E } _ { ( R , S ^ { \prime } ) \sim P _ { s , a } } \left[ \Pi _ { C } \left( ( f _ { R } ) _ { \# } \eta _ { h - 1 } ^ { * } ( S ^ { \prime } ) \right) \right] , \quad f _ { R } ( z ) = R + z .\tag{13}
$$

Here $( f _ { R } ) _ { \# } \nu$ is the law of $R + Z$ for $Z \sim \nu$ . Replacing each expectation by the same group’s empirical mean gives the estimator in Equation 1; the measure notation describes the same computation.

Stacked-coordinate convention and proof roadmap. We index a scalar shortfall (stop-loss) coordinate by $x = ( h , s , q )$ , write $e _ { x }$ for its standard basis vector, and write $U _ { h , s } = U _ { h } ( s , \cdot ) \in \mathbb { R } ^ { K }$ for the threshold block. A vector such as $U$ stacks these coordinates; a row index $b = ( h , s )$ selects one block. Sections B–C derive the influence and its sampling limits. Section D controls the extra error from learning the allocation.

For an outcome $W = ( R , S ^ { \prime } )$ and stacked stop-loss vector $U ,$ define the sampled stop-loss Bellman target at row $b = ( h , s )$ by

$$
\begin{array} { r } { \left[ T _ { b } \bigl ( W ; U \bigr ) \right] ( q ) = \left\{ \begin{array} { l l } { ( q - R ) _ { + } , } & { h = 1 , } \\ { \mathop { \mathcal { Z } _ { \boldsymbol { \mathcal { Z } } } } \bigl [ U _ { h - 1 } \bigl ( S ^ { \prime } , \cdot \bigr ) \bigr ] ( q - R ) , } & { h > 1 , } \end{array} \right. } \end{array}\tag{14}
$$

where $\mathcal { T } _ { \mathcal { Z } }$ is linear interpolation on the threshold grid, extended by zero for nonpositive arguments. For group $^ { g , }$ let $\mathcal { T } _ { g } ( W _ { g } ; U )$ be its full stacked affine contribution, including all known mixture coefficients. In the shared-kernel model, for $g = \left( s , a \right)$ and $b = ( h , s )$

$$
\left[ { \cal T } _ { s , a } ( W _ { g } ; U ) \right] _ { b } = \pi _ { h } ( a \mid s ) { \cal T } _ { b } ( W _ { g } ; U ) ,
$$

and rows at states other than s are zero. Thus one $W _ { g }$ contributes jointly to every required layer row at state $s ;$ this is why its layer effects must be summed before their variance is computed. For a categorical continuation $Z \sim \dot { \eta _ { h - 1 } ^ { * } } ( S ^ { \prime } )$ , the stop-loss transform $v \mapsto \mathbb { E } [ ( v - Z ) _ { + } ]$ is affine on every

grid interval $[ z _ { j } , z _ { j + 1 } ]$ , and its values at the knots are exactly $U _ { h - 1 } ^ { * } ( S ^ { \prime } , z _ { j } )$ . Hence, for $0 < v \le H$ linear interpolation evaluates it exactly:

$$
\begin{array} { r } { \mathcal { T } _ { \mathcal { Z } } [ U _ { h - 1 } ^ { * } ( S ^ { \prime } , \cdot ) ] ( v ) = \mathbb { E } [ ( v - Z ) _ { + } ] . } \end{array}
$$

Taking $v = q - R$ (with the stated zero extension when $q - R \leq 0 )$ gives the shortfall of $R + Z$ at threshold $q .$ . Lemma 1 shows that categorical projection preserves this shortfall for every grid threshold $q .$ Therefore

$$
U ^ { * } = \sum _ { g \in { \mathcal G } } \mathbb E _ { P _ { g } } \left[ \mathcal T _ { g } \big ( W _ { g } ; U ^ { * } \big ) \right] = { \mathbf b } + M U ^ { * } .\tag{15}
$$

Set $A : = I - M$ . Because $q - R \leq q \leq H$ , the finite-horizon target evaluates the interpolant only at or below the upper grid endpoint. The stated nonpositive extension is sufficient. In equation 15, b collects terms independent of continuation values, including the $h = 1 \mathrm { r o w s }$ . The matrix $M$ collects the known mixture coefficients, transition expectations, and interpolation weights multiplying lowerlayer coordinates. The matrix M lowers the layer and $M ^ { H } = 0$ . Given $n _ { g }$ observations per group, replace each expectation by its empirical average. This is the stop-loss transform of the sharedkernel empirical categorical estimator. Denote the resulting affine map by ${ \widehat { \mathbf { b } } } + { \widehat { M } } U$ ; again $\widehat { M } ^ { H } = 0$ for every dataset.

Explicit shared-kernel influence. Let ${ r } _ { h , s }$ be the adjoint block for layer–state row $( h , s )$ . For $g = \left( s , a \right)$ , the full-vector formula in Equation 4 is

$$
\phi _ { s , a } ( W ) = - \frac { 1 } { \alpha } \sum _ { h = 1 } ^ { H } \pi _ { h } ( a \mid s ) r _ { h , s } ^ { \top } \Big \{ { T } _ { h } \big ( W ; U ^ { * } \big ) - \mathbb { E } _ { P _ { s , a } } { T } _ { h } \big ( W ; U ^ { * } \big ) \Big \} .\tag{16}
$$

The variance of this sum includes all cross-layer covariances induced by reusing the same observation.

## B ALLOCATION AND BELLMAN IDENTITIES

Equation 6 follows from classical allocation once the scales are known. The identities below connect those scales to CVaR: projection preserves shortfalls, and Bellman propagation carries each group’s error to the root.

## B.1 MINIMIZING THE ALLOCATION VARIANCE

Once the influence expansion gives $\begin{array} { r } { V ( w ) = \sum _ { g } \sigma _ { g } ^ { 2 } / w _ { g } } \end{array}$ , classical Neyman allocation follows from Cauchy–Schwarz (Neyman, 1934):

$$
\left( \sum _ { g } \sigma _ { g } \right) ^ { 2 } = \left( \sum _ { g } \frac { \sigma _ { g } } { \sqrt { w _ { g } } } \sqrt { w _ { g } } \right) ^ { 2 } \leq \sum _ { g } \frac { \sigma _ { g } ^ { 2 } } { w _ { g } } , \qquad \sum _ { g } w _ { g } = 1 .
$$

If all scales are positive, equality holds only at $w _ { g } = \sigma _ { g } / S _ { \sigma }$ , where $\begin{array} { r } { S _ { \sigma } = \sum _ { q } \sigma _ { g } } \end{array}$ . If $S _ { \sigma } ~ > ~ 0$ but some scales vanish, this boundary design gives the infimum over positive designs: the mixtures $( 1 - \lambda ) \sigma / S _ { \sigma } + \lambda { \bf 1 } / G$ approach it as $\lambda \downarrow 0$ . If all scales vanish, every design has zero first-order variance. Sections B.3 and C.1 derive the influence expansion and its CLT and MSE limits for the categorical Bellman estimator.

## B.2 PROJECTION AND CATEGORICAL CVAR IDENTITIES

Section 3 relies on projection preserving shortfalls and a stable quantile making CVaR locally affine. For return X with CDF $F _ { X }$ , lower-tail CVaR is

$$
\operatorname { C V a R } _ { \alpha } ( X ) = { \frac { 1 } { \alpha } } \int _ { 0 } ^ { \alpha } F _ { X } ^ { - 1 } ( u ) d u , \alpha \in ( 0 , 1 ) , \ \mathrm { w i t h } \ F _ { X } ^ { - 1 } ( u ) = \operatorname* { i n f } \{ x : F _ { X } ( x ) \geq u \} .\tag{17}
$$

Lemma 1 (Projection identity). For every $q \in { \mathcal { Z } }$ and every law ν supported on [0, ∞), $[ 0 , \infty )$

$$
\int ( q - z ) _ { + } d \bigl ( \Pi _ { C } \nu \bigr ) ( z ) = \int ( q - z ) _ { + } d \nu ( z ) .\tag{18}
$$

Proof. For $y \in [ z _ { j } , z _ { j + 1 } ]$ , categorical projection (Bellemare et al., 2017; Rowland et al., 2018) sends $\delta _ { y }$ to

$$
\frac { z _ { j + 1 } - y } { z _ { j + 1 } - z _ { j } } \delta _ { z _ { j } } + \frac { y - z _ { j } } { z _ { j + 1 } - z _ { j } } \delta _ { z _ { j + 1 } } .
$$

For a grid atom $q ,$ the map $z \mapsto ( q - z ) _ { + }$ is affine on every grid cell. Its expectation is therefore preserved by the barycentric projection. If $y \geq H$ , projection clips to $H \geq q$ and both the original and clipped payoffs are zero. Inputs below zero are excluded by the nonnegative-return model. Linearity proves the identity for every input law supported on $[ 0 , \infty )$ □

For a categorical law $p$ with CDF $\begin{array} { r } { F _ { k } = \sum _ { i < k } p _ { i } } \end{array}$ , set $F _ { - 1 } = 0 , k _ { \alpha } = \operatorname* { m i n } \{ k : F _ { k } \geq \alpha \}$ , and $q _ { \alpha } = z _ { k _ { \alpha } }$ . The quantile-integral definition of lower-tail CVaR (Rockafellar & Uryasev, 2000) gives

$$
\mathrm { C V a R } _ { \alpha } ( p ) = \frac { 1 } { \alpha } \left\{ \sum _ { i < k _ { \alpha } } p _ { i } z _ { i } + ( \alpha - F _ { k _ { \alpha } - 1 } ) z _ { k _ { \alpha } } \right\}\tag{19}
$$

$$
= z _ { k _ { \alpha } } - { \frac { 1 } { \alpha } } \sum _ { i < k _ { \alpha } } p _ { i } ( z _ { k _ { \alpha } } - z _ { i } ) .\tag{20}
$$

Thus CVaR is affine in a neighborhood where the VaR index is fixed.

For the population root law, let $\begin{array} { r } { F _ { k } ^ { * } = \sum _ { j \leq k } p _ { H , s _ { 0 } , j } ^ { * } , F _ { - 1 } ^ { * } = 0 } \end{array}$ , and $k _ { \alpha } = \operatorname* { m i n } \{ k : F _ { k } ^ { * } \geq \alpha \}$ . The positive quantile margin used in the main text is

$$
m _ { \alpha } = \operatorname* { m i n } \left\{ \alpha - F _ { k _ { \alpha } - 1 } ^ { * } , \ : F _ { k _ { \alpha } } ^ { * } - \alpha \right\} > 0 .\tag{21}
$$

It places α strictly inside the quantile atom’s cumulative-mass interval. Throughout the finitehorizon proof, $C _ { \alpha , K } = q _ { \alpha } - \alpha ^ { - 1 } U _ { H } ^ { * } \left( s _ { 0 } , q _ { \alpha } \right)$ is the population categorical CVaR; ${ \widehat { C } } _ { N }$ uses the empirical fixed point and its empirical quantile atom.

Why a positive margin matters. For $H = 1$ , take $Z \ \in \ \{ 0 , 1 \}$ with $p : = \mathbb { P } ( Z = 0 )$ Then $\mathrm { C V a R } _ { \alpha } \bar { ( } Z ) = \operatorname* { m a x } \{ \bar { 0 , } 1 - p / \alpha \} . \mathrm { A t } p = \alpha$ the margin vanishes, and for an empirical fraction ${ \widehat { p } } ,$

$$
\sqrt { n } \widehat C _ { n } = \operatorname* { m a x } \{ 0 , - \sqrt { n } ( \widehat { p } - \alpha ) / \alpha \} \Rightarrow \operatorname* { m a x } \{ 0 , - Z _ { 0 } / \alpha \} , \qquad Z _ { 0 } \sim { \mathcal N } ( 0 , \alpha ( 1 - \alpha ) ) .
$$

The limit has an atom at zero, so it is non-Gaussian. $\mathrm { A t } \ p > \alpha$ the margin is positive, CVaR is locally constant, and every influence is zero. The fixed-design theorem includes this degenerate limit. Oracle adaptation assumes $S _ { \sigma } > 0$

A shared-kernel example with nonzero covariance. Take one state, one action, $H = 2 ,$ rewards R ∼ Bernoulli(p), grid {0, 1, 2}, and $p = 1 / 2$ . With $\alpha = 3 / 5$ , the root VaR is 1 and the margin is $3 / 2 0 .$ . Locally,

$$
{ \cal C } _ { \alpha , K } = 1 - \frac { ( 1 - p ) ^ { 2 } } { \alpha } , \qquad \phi ( R ) = \frac { 2 ( 1 - p ) } { \alpha } ( R - p ) = \frac { R - 1 / 2 } { \alpha } .
$$

Each of the two layer contributions is $( R - 1 / 2 ) / ( 2 \alpha )$ . Their sum has variance $1 / ( 4 \alpha ^ { 2 } ) = 2 5 / 3 6$ whereas deleting their covariance gives $1 / ( \dot { 8 \alpha } ^ { 2 } ) \dot { = } 2 5 / 7 2$ . This demonstrates the variance error from treating a shared draw as two independent draws. There is only one group, so the example isolates covariance. The allocation itself is trivial.

## B.3 EXACT EMPIRICAL FIXED-POINT EXPANSION

For Theorem 1, we separate leading sampling error from feedback due to estimating the continuation model. Define the centered contribution of one group observation

$$
\Xi _ { g } ( W _ { g } ) = { \mathcal T } _ { g } ( W _ { g } ; U ^ { * } ) - \mathbb { E } _ { P _ { g } } { \mathcal T } _ { g } ( W _ { g } ; U ^ { * } ) , \qquad \mathbb { E } _ { P _ { g } } \Xi _ { g } = 0 ,\tag{22}
$$

and the combined empirical Bellman error

$$
\widehat { \xi } = \sum _ { g \in \mathcal { G } } \frac { 1 } { n _ { g } } \sum _ { i = 1 } ^ { n _ { g } } \Xi _ { g } ( W _ { g , i } ) .\tag{23}
$$

By the affine form of $\mathcal { T } _ { g }$ , this same perturbation can be written as

$$
{ \widehat \xi } = { \Bigl ( { \widehat { \bf b } } - { \bf b } \Bigr ) } + { \Bigl ( { \widehat M } - M \Bigr ) } U ^ { * } .
$$

Thus $\widehat { \xi }$ is the empirical Bellman error evaluated at the population fixed point. The resolvent below converts it into fixed-point estimation error. Subtracting the population and empirical fixed-point equations gives the exact identity

$$
\widehat { U } - U ^ { * } = \Big ( I - \widehat { M } \Big ) ^ { - 1 } \widehat { \xi } .\tag{24}
$$

Indeed,

$$
\begin{array} { r l } & { \Big ( I - \widehat { M } \Big ) \Big ( \widehat { U } - U ^ { * } \Big ) = \widehat { { \mathbf b } } + \widehat { M } U ^ { * } - U ^ { * } } \\ & { \qquad = \Big ( \widehat { { \mathbf b } } - { \mathbf b } \Big ) + \Big ( \widehat { M } - M \Big ) U ^ { * } = \widehat { \xi } . } \end{array}
$$

Both inverses are finite sums:

$$
\Big ( I - \widehat { M } \Big ) ^ { - 1 } = \sum _ { t = 0 } ^ { H - 1 } \widehat { M } ^ { t } , \qquad \Big ( I - M \Big ) ^ { - 1 } = \sum _ { t = 0 } ^ { H - 1 } M ^ { t } .
$$

Every row of $M$ and $\widehat { M }$ is sub-probability, so their induced infinity norms are at most one and both inverse norms are at most $H$

The resolvent identity gives

$$
\widehat { U } - U ^ { * } = A ^ { - 1 } \widehat { \xi } + R _ { N } ,\tag{25}
$$

$$
R _ { N } = A ^ { - 1 } \Big ( \widehat { M } - M \Big ) \Big ( I - \widehat { M } \Big ) ^ { - 1 } \widehat { \xi } .\tag{26}
$$

In the finite-horizon bounds below, ∥·∥ denotes the vector infinity norm or its induced matrix infinity norm. With $n _ { \mathrm { m i n } } = \operatorname* { m i n } _ { g } n _ { g }$ , finite dimension and bounded observations imply

$$
\left\| \widehat { M } - M \right\| = O _ { p } \left( n _ { \operatorname* { m i n } } ^ { - 1 / 2 } \right) , \quad \left\| \widehat { \xi } \right\| = O _ { p } \left( n _ { \operatorname* { m i n } } ^ { - 1 / 2 } \right) , \quad \left\| R _ { N } \right\| = O _ { p } \left( n _ { \operatorname* { m i n } } ^ { - 1 } \right) .\tag{27}
$$

Lemma 2 (Uniform moment control). For every fixed integer $p \geq 2 ,$ , there is a finite constant $C _ { p } ,$ depending only on thefixed model dimensions, grid, horizon, and $p ,$ such that

$$
\begin{array} { r } { \mathbb { E } \left\| \widehat { M } - M \right\| ^ { p } + \mathbb { E } \left\| \widehat { \xi } \right\| ^ { p } \leq C _ { p } n _ { \operatorname* { m i n } } ^ { - p / 2 } , } \\ { \mathbb { E } \left\| R _ { N } \right\| ^ { p } \leq C _ { p } n _ { \operatorname* { m i n } } ^ { - p } . } \end{array}\tag{28}
$$

In particular, under a stable allocation, $N \mathbb { E } \| R _ { N } \| ^ { 2 }  0$

Proof. Idea. Both empirical coefficient error and Bellman error are bounded sample averages, hence of order $n _ { \mathrm { m i n } } ^ { - 1 / 2 }$ . The fixed-point remainder is their product, so it is one order smaller. The deterministic resolvent bound prevents the recursion from amplifying these rates.

Each entry of $\widehat { M } - M$ and $\widehat { \xi }$ is a finite sum, over groups, of centered averages of bounded random variables. For a centered average ${ \bar { X } } _ { g }$ of $n _ { g }$ independent bounded variables, Hoeffding’s inequality (Hoeffding, 1963) gives $\mathbb { P } ( | \bar { X _ { q } } | > \bar { t } ) ~ \le ~ 2 e ^ { - c n _ { g } t ^ { 2 } }$ Integrating this tail via $\begin{array} { r } { \mathbb { E } | \bar { X } _ { g } | ^ { p } = \int _ { 0 } ^ { \infty } p t ^ { p - 1 } \mathbb { P } ( | \bar { X } _ { g } | > t ) } \end{array}$ dt yields $\mathbb { E } | \bar { X } _ { g } | ^ { p } \le C _ { p } n _ { g } ^ { - p / 2 }$ . The number of groups and matrix entries is fixed, so norm equivalence and a finite-sum inequality give the first line of equation 28. Nilpotence and the sub-probability row structure give the deterministic bounds $\lVert A ^ { - 1 } \rVert \leq H$ and $\| ( I - { \widehat { M } } ) ^ { - 1 } \| \leq H$ . Hence

$$
\left\| R _ { N } \right\| \leq H ^ { 2 } \left\| { \widehat { M } } - M \right\| \left\| { \widehat { \xi } } \right\| .
$$

Cauchy–Schwarz with the preceding 2p-moment bounds proves the second line. If $n _ { \mathrm { m i n } } \asymp N$ , it yields $\mathbf { \dot { \cal N } } \mathbb { E } \| { \cal R } _ { N } \| ^ { 2 } = { \cal O } ( N ^ { \frac { . } { - 1 } } )$ ). □

## C FIXED-DESIGN LIMITS AND STRUCTURAL INTERPRETATION

Theorems 1 and 2 turn the influence calculation into an attainable accuracy benchmark for fixed query shares. The untied factorization explains the allocation ablations; Proposition 1 then shows why visitation and reward moments cannot determine the best tail allocation.

## C.1 FIXED-DESIGN CLT AND NORMALIZED MSE

ProofofTheorem 1. Idea. There are three issues to separate: linearize the empirical Bellman fixed point, show the empirical VaR atom stays on the same categorical cell, and then transfer the groupwise CLT and second moment through that locally affine CVaR readout.

Step 1: asymptotic linearity on the correct VaR cell. Let $E _ { N }$ be the event that the empirical and population categorical VaR indices agree, set $x _ { 0 } = ( H , s _ { 0 } , q _ { \alpha } )$ and $r ^ { \top } = e _ { x _ { 0 } } ^ { \top } A ^ { - 1 }$ , and define

$$
\phi _ { g } ( W _ { g } ) : = - \alpha ^ { - 1 } r ^ { \top } \Xi _ { g } ( W _ { g } ) , \qquad L _ { N } : = \sum _ { g } \frac { 1 } { n _ { g } } \sum _ { i = 1 } ^ { n _ { g } } \phi _ { g } ( W _ { g , i } ) .\tag{29}
$$

Then E ${ } _ { P _ { g } } \phi _ { g } = 0$ and $\mathbb { E } _ { P _ { g } } \phi _ { g } ^ { 2 } = \sigma _ { g } ^ { 2 }$ . On $E _ { N } ,$ , equation 20 and equation 25 give

$$
\widehat C _ { N } - C _ { \alpha , K } = L _ { N } + \widetilde R _ { N } , \qquad \widetilde R _ { N } : = - \alpha ^ { - 1 } e _ { x _ { 0 } } ^ { \top } R _ { N } .\tag{30}
$$

Under a stable allocation, $\begin{array} { r } { n _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { g } n _ { g } \asymp N } \end{array}$ , so Lemma 2 gives $\sqrt { N } \widetilde { R } _ { N } = o _ { p } ( 1 )$ . For each fixed group,

$$
\frac { \sqrt { N } } { n _ { g } } \sum _ { i = 1 } ^ { n _ { g } } \phi _ { g } ( W _ { g , i } ) = \sqrt { \frac { N } { n _ { g } } } \frac { 1 } { \sqrt { n _ { g } } } \sum _ { i = 1 } ^ { n _ { g } } \phi _ { g } ( W _ { g , i } ) \Rightarrow { \cal N } \bigg ( 0 , \frac { \sigma _ { g } ^ { 2 } } { w _ { g } } \bigg ) ,
$$

by the ordinary CLT and $n _ { g } / N  w _ { g } > 0$ . The groups are independent and their number is fixed, so the vector of group terms converges jointly to independent Gaussian limits. Summing the coordinates gives

$$
\sqrt { N } L _ { N } \Rightarrow \mathcal { N } ( 0 , V ( w ) ) , \qquad V ( w ) : = \sum _ { g } \frac { \sigma _ { g } ^ { 2 } } { w _ { g } } .\tag{31}
$$

Step 2: the VaR cell is correct with exponentially high probability. Each entry of $\widehat { \mathbf { b } } - \mathbf { b }$ and $\widehat { M } - M$ is a finite sum of averages of bounded variables. Hoeffding’s inequality (Hoeffding, 1963) and a union bound therefore give constants $c _ { 1 } , c _ { 2 } > 0$ such that

$$
\mathbb { P } \left( \| \widehat { \mathbf { b } } - \mathbf { b } \| _ { \infty } + \| \widehat { M } - M \| _ { \infty } > t \right) \le c _ { 1 } e ^ { - c _ { 2 } n _ { \operatorname* { m i n } } t ^ { 2 } } , \qquad t > 0 .\tag{32}
$$

For a categorical law with stop-loss vector $U$ and CDF $F ,$

$$
F _ { j - 1 } = \frac { U ( z _ { j } ) - U ( z _ { j - 1 } ) } { z _ { j } - z _ { j - 1 } } \quad ( j = 1 , \ldots , K - 1 ) , \qquad F _ { K - 1 } = 1 .
$$

Because every shortfall coordinate satisfies $0 \leq U _ { h } ^ { * } ( s , q ) \leq q \leq H$ , we have $\| U ^ { * } \| _ { \infty } \leq H$ . Thus, with $\begin{array} { r } { \delta _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { j < K - 1 } ( z _ { j + 1 } - z _ { j } ) > 0 } \end{array}$ , the exact fixed-point identity and $\| ( I - { \widehat { M } } ) ^ { - 1 } \| _ { \infty } \leq H$ imply

$$
\| \widehat { U } - U ^ { * } \| _ { \infty } \leq H \{ \| \widehat { \mathbf { b } } - \mathbf { b } \| _ { \infty } + H \| \widehat { M } - M \| _ { \infty } \} ,
$$

$$
\| \widehat { F } - F ^ { * } \| _ { \infty } \leq \frac { 2 H } { \delta _ { \operatorname* { m i n } } } \{ \| \widehat { \mathbf { b } } - \mathbf { b } \| _ { \infty } + H \| \widehat { M } - M \| _ { \infty } \} .\tag{33}
$$

Combining this bound with equation 32 gives

$$
\begin{array} { r } { \mathbb { P } ( E _ { N } ^ { c } ) \leq \mathbb { P } ( \| \widehat { F } - F ^ { * } \| _ { \infty } \geq m _ { \alpha } ) \leq c _ { 3 } e ^ { - c _ { 4 } n _ { \operatorname* { m i n } } m _ { \alpha } ^ { 2 } } . } \end{array}\tag{34}
$$

Indeed, an error smaller than $m _ { \alpha }$ preserves $\widehat { F } _ { k _ { \alpha } - 1 } < \alpha < \widehat { F } _ { k _ { \alpha } }$ ; at the endpoints use $\widehat F _ { - 1 } = 0$ and $\widehat F _ { K - 1 } = 1$

Step 3: normalized MSE and transfer off the good event. For the second-moment claim, centering and independence give

$$
N \mathbb { E } L _ { N } ^ { 2 } = N \sum _ { g } { \frac { \sigma _ { g } ^ { 2 } } { n _ { g } } } \longrightarrow V ( w ) .\tag{35}
$$

Lemma 2 also gives NE $\widetilde { R } _ { N } ^ { 2 } = O ( N ^ { - 1 } )$ and, by Cauchy–Schwarz, ${ \cal N } | \mathbb { E } [ L _ { N } \widetilde { R } _ { N } ] | = o ( 1 )$ . Hence $N \mathbb { E } ( L _ { N } + \widetilde { R } _ { N } ) ^ { 2 }  V ( w )$ . All three variables $\widehat { C } _ { N } - C _ { \alpha , K } , L _ { N }$ , and $\bar { R } _ { N }$ are uniformly bounded; for the last, use equation 26 and the deterministic resolvent bounds. Therefore

$$
\begin{array} { r } { N \mathbb { E } \Big [ \{ ( \widehat { C } _ { N } - C _ { \alpha , K } ) ^ { 2 } + ( L _ { N } + \widetilde { R } _ { N } ) ^ { 2 } \} \mathbf { 1 } _ { E _ { N } ^ { c } } \Big ] \leq C N \mathbb { P } ( E _ { N } ^ { c } ) \longrightarrow 0 . } \end{array}\tag{36}
$$

Equation 30 holds on $E _ { N }$ , so the last two displays prove the normalized-MSE limit. Since $\mathbb { P } ( E _ { N } ^ { c } ) $ 0 and $\sqrt { N } \widetilde { R } _ { N } = o _ { p } ( 1 )$ , they also transfer the CLT for $L _ { N }$ to ${ \widehat { C } } _ { N }$ □

## C.2 SEMIPARAMETRIC EFFICIENCY

ProofofTheorem 2. Idea. Differentiate the target along arbitrary groupwise score directions. The resulting pathwise derivative is represented by $\phi _ { g }$ in each group. Under sampling fraction $w _ { g } .$ , the product-experiment canonical gradient is therefore $\phi _ { g } / w _ { g }$ , whose squared norm is exactly $\breve { V } ( w )$ The asymptotic-linear expansion from Theorem 1, together with Le Cam’s third lemma, then shows that the plug-in estimator is regular under local alternatives and attains this bound.

Step 1: pathwise derivative. Fix the population collection $P = ( P _ { g } ) _ { g \in \mathcal { G } }$ and let $\mathcal { P } = \otimes _ { g \in \mathcal { G } } \mathcal { P } _ { g } ,$ where each $\mathcal { P } _ { g }$ is the nonparametric model on group $g ^ { \ast } \mathbf { s }$ fixed bounded outcome space. Write $L _ { 0 } ^ { 2 } ( P _ { g } )$ for the square-integrable, mean-zero functions under $P _ { g }$ . For a bounded $s _ { g } \in L _ { 0 } ^ { 2 } ( P _ { g } )$ and sufficiently small $| t | , \bar { d P } _ { g , t } = ( 1 + t s _ { g } ) d P _ { g }$ is a valid submodel with score $s _ { g } ;$ bounded mean-zero scores are dense in $L _ { 0 } ^ { 2 } ( P _ { g } )$ (van der Vaart, 1998, Chapters 7 and 25). It therefore suffices to derive the gradient first for bounded scores; because all Bellman contributions and the resulting $\phi _ { g }$ are bounded, the derivative extends continuously to the $L _ { 0 } ^ { 2 } ( P _ { g } )$ closure.

Write the affine group contribution as ${ \cal T } _ { g } ( W ; U ) = a _ { g } ( W ) + B _ { g } ( W ) U$ . Along simultaneous differentiable-in-quadratic-mean paths with scores $s _ { g } \in L _ { 0 } ^ { 2 } ( P _ { g } )$ , differentiating equation 15 at $t = 0$ gives

$$
\dot { U } _ { s } = \sum _ { g } \mathbb { E } _ { P _ { g } } [ \mathcal { T } _ { g } ( W _ { g } ; U ^ { * } ) s _ { g } ( W _ { g } ) ] + M \dot { U } _ { s } .
$$

Because $\mathbb { E } _ { P _ { g } } s _ { g } = 0$ , the expectation in brackets equals $\mathbb { E } _ { P _ { g } } [ \Xi _ { g } ( W _ { g } ) s _ { g } ( W _ { g } ) ]$ ]. Hence

$$
A \dot { U } _ { s } = v _ { s } , \qquad v _ { s } : = \sum _ { g } \mathbb { E } _ { P _ { g } } [ \Xi _ { g } ( W _ { g } ) s _ { g } ( W _ { g } ) ] .\tag{37}
$$

The positive margin fixes the root VaR index on a neighborhood of the population law, so the CVaR readout is locally affine. Therefore

$$
\begin{array} { l } { \displaystyle \dot { C } _ { s } = - \frac 1 \alpha e _ { x _ { 0 } } ^ { \top } \dot { U } _ { s } = - \frac 1 \alpha r ^ { \top } v _ { s } } \\ { \displaystyle \qquad = \sum _ { g } \mathbb E _ { P _ { g } } [ \phi _ { g } ( W _ { g } ) s _ { g } ( W _ { g } ) ] . } \end{array}\tag{38}
$$

Thus $C _ { \alpha , K }$ is pathwise differentiable at the population law with groupwise derivative representers $\phi _ { g }$

Step 2: canonical gradient under the sampling design. For the deterministic allocation, applying local asymptotic normality groupwise to $P _ { g , t / \sqrt { N } } ^ { \otimes n _ { g } }$ and summing the log-likelihood ratios gives a product LAN experiment (van der Vaart, 1998, Theorem 7.2) with tangent inner product

$$
\langle s , \widetilde s \rangle _ { w } : = \sum _ { g } w _ { g } \mathbb { E } _ { P _ { g } } [ s _ { g } \widetilde s _ { g } ] .
$$

By equation 38, its Riesz representer is $\psi _ { w , g } = \phi _ { g } / w _ { g }$ , because $\begin{array} { r } { \langle \psi _ { w } , s \rangle _ { w } = \sum _ { g } \mathbb { E } _ { P _ { g } } [ \phi _ { g } s _ { g } ] = \dot { C } _ { s } } \end{array}$ Its squared norm is

$$
\| \psi _ { w } \| _ { w } ^ { 2 } = \sum _ { g } w _ { g } \mathbb { E } _ { P _ { g } } \left[ \left( { \frac { \phi _ { g } } { w _ { g } } } \right) ^ { 2 } \right] = \sum _ { g } { \frac { \sigma _ { g } ^ { 2 } } { w _ { g } } } = V ( w ) .\tag{39}
$$

Step 3: regularity and attainment under local alternatives. Theorem 1 gives the baseline asymptoticlinear representation

$$
\sqrt { N } \{ \widehat { C } _ { N } - C _ { \alpha , K } ( P ) \} = \sum _ { g } \frac { \sqrt { N } } { n _ { g } } \sum _ { i = 1 } ^ { n _ { g } } \phi _ { g } ( W _ { g , i } ) + o _ { P } ( 1 ) = \frac { 1 } { \sqrt { N } } \sum _ { g } \sum _ { i = 1 } ^ { n _ { g } } \psi _ { w , g } ( W _ { g , i } ) + o _ { P } ( 1 ) .
$$

For the last relation, write $\begin{array} { r l r } { N / n _ { g } } & { { } = } & { w _ { q } ^ { - 1 } + o ( 1 ) } \end{array}$ Since, for each fixed group, $N ^ { - 1 / 2 } \sum _ { i = 1 } ^ { n _ { g } } \phi _ { g } ( W _ { g , i } ) = O _ { P } ( 1 )$ , replacing $N / n _ { g }$ by $1 / w _ { g }$ changes the finite sum by only $o _ { P } ( 1 )$ Fix a DQM direction $\begin{array} { r } { s = ( s _ { g } ) } \end{array}$ and a scalar t. Write $\dot { P _ { t / \sqrt { N } } } : = ( P _ { g , t / \sqrt { N } } ) _ { g \in \mathcal { G } }$ for the local collection of group laws and $P _ { N , t } : = \otimes _ { g } P _ { g , t / \sqrt { N } } ^ { \otimes n _ { g } }$ for the corresponding sample law. LAN implies that $P _ { N , t }$ is contiguous to the baseline sample law, so the displayed $o _ { P } ( 1 )$ term is also ${ O } _ { P _ { N , t } } ( 1 )$

By Step 1,

$$
\sqrt { N } \{ C _ { \alpha , K } ( P _ { t / \sqrt { N } } ) - C _ { \alpha , K } ( P ) \} \longrightarrow t \dot { C } _ { s } .
$$

Under the baseline law, the joint CLT for the influence sum and the LAN central sequence has covariance $\langle \psi _ { w } , s \rangle _ { w } = \dot { C } _ { s }$ . Le Cam’s third lemma (van der Vaart, 1998, Chapter 6) therefore gives, under $P _ { N , t }$

$$
\frac { 1 } { \sqrt { N } } \sum _ { g } \sum _ { i = 1 } ^ { n _ { g } } \psi _ { w , g } ( W _ { g , i } ) \Rightarrow \mathcal { N } ( t \dot { C } _ { s } , V ( w ) ) .
$$

Subtracting the local target shift yields

$$
\sqrt { N } \{ \widehat C _ { N } - C _ { \alpha , K } ( P _ { t / \sqrt { N } } ) \} \Rightarrow { \mathcal N } ( 0 , V ( w ) ) \qquad \mathrm { u n d e r } ~ P _ { N , t } .\tag{40}
$$

This limit is the same for every fixed t and DQM direction, which is the required regularity. Hence the plug-in estimator attains the canonical-gradient variance $V ( w )$

Step 4: lower bound. The product tangent space is linear and hence a convex cone. The convolution theorem therefore implies that the limit law of any regular estimator is the convolution of $\mathcal { N } ( 0 , V ( w ) )$ with an independent remainder; in particular, whenever its second moment is finite, its asymptotic variance is at least $V ( w )$ (van der Vaart, 1998, Theorem 25.20). The local asymptotic minimax theorem gives the corresponding lower bound $V ( w )$ for local squared-error risk (van der Vaart, 1998, Theorem 25.21). The plug-in estimator has the Gaussian limit in equation 40, so it attains both bounds. Thus it is semiparametrically efficient for the fixed design. □

Section B.1 minimizes this bound, including zero-scale groups.

## C.3 UNTIED FACTORIZATION FOR THE ALLOCATION ABLATIONS

The ablations in Figure 3 separate propagation from local variability. When one independent law feeds each layer–state row $b = ( h , s )$ , its centered innovation is $\zeta _ { b } ( \dot { W } ) = T _ { b } ( W ; U ^ { * } ) \dot { - } U _ { b } ^ { * }$ . Thus

$$
\phi _ { b } ( W ) = - \alpha ^ { - 1 } r _ { b } ^ { \top } \zeta _ { b } ( W ) .
$$

The adjoint is nonnegative because $\begin{array} { r } { r ^ { \top } = e _ { x _ { 0 } } ^ { \top } \sum _ { j = 0 } ^ { H - 1 } M ^ { j } } \end{array}$ and $M \geq 0$ . Set $d _ { b } = \mathbf { 1 } ^ { \top } r _ { b . } \mathrm { ~ I f ~ } d _ { b } > 0$ normalize its threshold weights as $\rho _ { b } = r _ { b } / d _ { b }$ and define

$$
\begin{array} { r } { \tau _ { b } ^ { 2 } = \alpha ^ { - 2 } \operatorname { V a r } _ { P _ { b } } ( \rho _ { b } ^ { \top } \zeta _ { b } ( W ) ) , \qquad \sigma _ { b } = d _ { b } \tau _ { b } . } \end{array}
$$

If $d _ { b } = 0$ , then $r _ { b } = 0$ and $\sigma _ { b } = 0 ;$ set $\tau _ { b } = 0$ . This is the factorization used by the reachability-only and local-scale-only ablations. The quantity $d _ { b }$ is adjoint mass: it propagates visitation together with the remaining shortfall threshold, so it need not equal ordinary state occupancy. The factor $\tau _ { b }$ measures variability after averaging over those threshold weights. The oracle uses their product.

## C.4 CONTROLLED SEPARATION WITH FIXED VISITATION, MOMENTS, AND ROOT LAW

Proposition 1 isolates tail information missing from visitation and reward moments while preserving the root law. Section E.2.2 tests whether a charged pilot learns this population signal.

ProofofProposition 1. Consider one initial state, $H = 1 , G \geq 2$ actions with probabilities $1 / G ,$ and a fixed terminal next state. Each action’s reward law is independently queryable. Set $\alpha = 1 / 1 0$ and use the grid $\{ 0 , 1 / 3 , 5 / 9 , 2 / 3 , 1 \}$ , which contains every reward below and makes projection exact. Define two reward laws:

$$
P ^ { \mathrm { A } } : \quad \mathbb { P } ( R = 0 ) = { \frac { 1 } { 1 0 } } , \quad \mathbb { P } ( R = 5 / 9 ) = { \frac { 9 } { 1 0 } } ; \qquad P ^ { \mathrm { B } } : \quad \mathbb { P } ( R = 1 / 3 ) = \mathbb { P } ( R = 2 / 3 ) = { \frac { 1 } { 2 } } .\tag{41}
$$

Both have $\mathbb { E } R = 1 / 2$ and $\mathbb { E } R ^ { 2 } \ : = \ : 5 / 1 8$ , hence $\mathrm { V a r } ( R ) = 1 / 3 6$ . These are properties of the construction; estimation still uses the unrestricted group-law model. Assign $P ^ { \mathrm { A } }$ to group 1 and $P ^ { \mathrm { B } }$ to every other group; denote these separated laws by $P _ { g } ^ { \mathrm { s e p } }$

The root law is $\overline { { { P } } } = G ^ { - 1 } P ^ { \mathrm { A } } + ( 1 - G ^ { - 1 } ) P ^ { \mathrm { B } }$ . Its mass strictly below $1 / 3$ is $1 / ( 1 0 G )$ and its cumulative mass at $1 / 3$ is $1 / 2 - 2 / ( 5 G )$ . Thus, for every $G \geq 2$

$$
q _ { \alpha } = \frac { 1 } { 3 } , \qquad m _ { \alpha } = \frac { 1 } { 1 0 } \left( 1 - \frac { 1 } { G } \right) > 0 , \qquad C _ { \alpha , K } = \frac { G - 1 } { 3 G } .\tag{42}
$$

Let $L ( R ) = ( 1 / 3 - R ) _ { + }$ . The root is the known uniform mixture of the group laws, so its group influence is

$$
\phi _ { g } ( R ) = - \frac { 1 } { G \alpha } \{ L ( R ) - \mathbb { E } _ { P _ { g } } L ( R ) \} .\tag{43}
$$

Under $P ^ { \mathrm { A } } , L { \mathrm { ~ i s ~ } } 1 / 3$ with probability $1 / 1 0$ and zero otherwise, giving $\mathrm { V a r } ( L ) = 1 / 1 0 0$ . Under $P ^ { \mathrm { B } } , L$ is identically zero. Consequently $\sigma _ { 1 } = 1 / G$ and $\sigma _ { g } = 0$ for $g > 1$ . The occupancy design is uniform. The group influence for estimating the mean is $( R - 1 / 2 ) / G$ , with standard deviation $1 / ( 6 G )$ in every group, so the mean-optimal design is also uniform. Evaluating CVaR variance under either design gives $\begin{array} { r } { V = G \sum _ { q } \sigma _ { g } ^ { 2 } = 1 / G } \end{array}$ , whereas $\begin{array} { r } { V ^ { * } = ( \sum _ { a } \sigma _ { g } ) ^ { 2 } = 1 / \tilde { G ^ { 2 } } } \end{array}$ . The tail oracle is a boundary design; its value is approached by positive designs as the floor vanishes.

For the fixed-root-law family, set

$$
P _ { g } ( t ) = ( 1 - t ) \overline { { P } } + t P _ { g } ^ { \mathrm { s e p } } , \qquad 0 \leq t \leq 1 .\tag{44}
$$

Every component being mixed has the same first two reward moments, so each $P _ { g } ( t )$ retains mean $1 / 2$ and variance $1 / 3 6$ . Moreover, $\begin{array} { r } { G ^ { - 1 } \sum _ { a } P _ { g } ( t ) = \overline { { P } } } \end{array}$ for every t: the full root return law, CVaR, and quantile margin in Equation 42 stay fixed. Writing $\theta _ { g } ( t ) = P _ { g } ( t ) \{ R = 0 \}$ gives

$$
\theta _ { 1 } ( t ) = \frac { 1 + ( G - 1 ) t } { 1 0 G } , \qquad \theta _ { g } ( t ) = \frac { 1 - t } { 1 0 G } ( g > 1 ) , \qquad \sigma _ { g } ( t ) = \frac { 1 0 } { 3 G } \sqrt { \theta _ { g } ( t ) ( 1 - \theta _ { g } ( t ) ) } .\tag{45}
$$

For uniform occupancy, the variance ratio is explicitly

$$
\frac { V _ { \mathrm { o c c u p a n c y } } ( t ) } { V ^ { * } ( t ) } = \frac { G \sum _ { g } \sigma _ { g } ( t ) ^ { 2 } } { \left( \sum _ { g } \sigma _ { g } ( t ) \right) ^ { 2 } } .
$$

Cauchy–Schwarz gives the lower bound 1, while nonnegativity gives $\begin{array} { r } { \sum _ { q } \sigma _ { g } ( t ) ^ { 2 } \le ( \sum _ { q } \sigma _ { g } ( t ) ) ^ { 2 } } \end{array}$ and hence the upper bound G. $\mathbf { A } { \boldsymbol { \mathrm { t } } } \ t = 0$ all scales are equal and positive, so the ratio is one. At $t = 1$ it is G, as above. For all intermediate t the ratio is continuous. It therefore attains every value between the two endpoints. Every scale is positive when $t < 1$ , so ratios arbitrarily close to G also occur with all groups influential. This proves the family without a change in visitation, conditional moments, or root return law. □

Rollouts and the ideal occupancy anchor. In this one-step example one rollout and one condi tional observation each cost one transition query. For complete rollouts, the shortfall indicator has probability $1 / ( 1 0 G )$ under the fixed root law. The empirical CVaR therefore has leading variance

$$
V _ { \mathrm { r o l l o u t } } = { \frac { ( 1 / 3 ) ^ { 2 } } { \alpha ^ { 2 } } } { \frac { 1 } { 1 0 G } } \left( 1 - { \frac { 1 } { 1 0 G } } \right) = { \frac { 1 0 G - 1 } { 9 G ^ { 2 } } } ,\tag{46}
$$

which does not change with $t , \mathrm { A t } t = 1$ , the equal mixture of the population tail oracle and uniform occupancy gives group 1 weight $( G { + } 1 ) / ( 2 G )$ and hence $V _ { \mathrm { i d e a l \ a n c h o r } } = 2 / [ G ( G { + } 1 ) ]$ . For $G = 1 0 .$ the variance constants for occupancy, rollouts, the tail oracle, and this ideal anchor are respectively .1, .11, .01, and 1/55. These are leading-variance constants at the separated endpoint, before pilot cost, exploration, and rounding. For intermediate t, use Equation 45; the occupancy variance itself changes with t even though the root law is fixed.

From population allocation to pilot learning. $\mathbf { A } \mathbf { t } \ t = 1$ , m independent pilot observations from group 1 miss its zero reward with probability $( 9 / 1 0 ) ^ { m }$ . This is a missed-outcome probability, not the probability of TIS failure or of a wrong pilot quantile. Section E.2.2 reports learned performance with all queries charged and the prescribed uniform fallback.

## D LEARNED ALLOCATION

Algorithm 1 Tail-Influence Sampling (TIS); pilot cost is included in N   
Require: query groups G, budget N, grid Z, tail level α; pilot size $m _ { N } \ge 2 ,$ , floor $0 < \lambda _ { N } < 1 , N - G m _ { N } \geq$   
2G   
1: Draw $m _ { N }$ independent pilot samples per group; fit empirical conditional laws.   
2: Compute the pilot root quantile, shortfalls, and adjoint via Equations 1 and 3.   
3: Score each pilot draw and compute $\widehat { \sigma } _ { g }$ by Equation 7.   
4: Form w by Equation 8; if all scales vanish, use $\widehat { w } _ { g } = 1 / G .$   
5: Set $n _ { g } = 2 + \mathrm { L R M } _ { g } ( N - G m _ { N } - 2 G , \widehat { w } ) .$   
6: Draw $n _ { g }$ fresh main samples per group; discard the pilot.   
7: Evaluate Equation 1 with the main empirical laws and return the root CVaR $\widehat { C } _ { N } ^ { \mathrm { T I S } } .$

Algorithm 1 learns allocation scales using a pilot, as in adaptive stratified sampling (Etor<sup>´</sup> e & Jour-´ dain, 2010; Carpentier et al., 2015), but also estimates the Bellman model and quantile. We control these errors to prove oracle adaptation, then establish the anchor’s separate safeguard. Only fresh main samples form the final estimate; $G = | { \mathcal { G } } |$

## D.1 PILOT-SCALE CONSISTENCY AND LOWER-TAIL CONTROL

We use the centered form of Equation 7. For $m \geq 2$ pilot draws per group, set $\widehat { x } _ { 0 } = ( H , s _ { 0 } , \widehat { q } _ { \alpha } ^ { ( 0 ) } )$ , the root coordinate at the pilot quantile. Define

$$
\begin{array} { r l } & { \hat { r } ^ { ( 0 ) \top } : = e _ { \hat { x } _ { 0 } } ^ { \top } ( I - \widehat { M } ^ { ( 0 ) } ) ^ { - 1 } , } \\ & { \quad \widehat { \Xi } _ { g , i } ^ { ( 0 ) } = \mathcal { T } _ { g } ( W _ { g , i } ; \widehat { U } ^ { ( 0 ) } ) - \displaystyle \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathcal { T } _ { g } ( W _ { g , j } ; \widehat { U } ^ { ( 0 ) } ) , } \\ & { \quad \widehat { \phi } _ { g , i } ^ { ( 0 ) } = - \alpha ^ { - 1 } \widehat { r } ^ { ( 0 ) \top } \widehat { \Xi } _ { g , i } ^ { ( 0 ) } , } \\ & { \quad \widehat { \sigma } _ { g } ^ { 2 } = \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( \widehat { \phi } _ { g , i } ^ { ( 0 ) } ) ^ { 2 } . } \end{array}\tag{47}
$$

The four lines give, respectively, weights that propagate local changes to the root shortfall, each draw’s deviation from its group’s mean Bellman contribution, its estimated CVaR influence, and the empirical variance of these influences. Linearity gives ${ \widehat { \phi } } _ { q , i } ^ { ( 0 ) } = { \widehat { d } } _ { g , i } - { \bar { d } } _ { g }$ , recovering Equation 7. Take $m = m _ { N }$ for Theorem 3. Shared stage effects are summed before taking the variance, preserving their covariance.

Lemma 3 (Pilot-scale control). Under thefixed-dimensional bounded Bellman model and positive quantile margin, there is a deterministic $B _ { \sigma } < \infty$ such that $0 \leq \widehat { \sigma } _ { g } \leq B _ { \sigma }$ for every group and pilot dataset. With each pilotformedfrom thefirst m observations $o f$ an i.i.d. stream in each group,

$$
{ \widehat { \sigma } } _ { g } \longrightarrow \sigma _ { g } \quad a . s . \ a s m  \infty . \nonumber\tag{48}
$$

For each group with $\sigma _ { g } > 0 ,$ constants $c _ { g } , C _ { g } > 0$ exist such that

$$
\mathbb { P } ( \widehat { \sigma } _ { g } < \sigma _ { g } / 2 ) \le C _ { g } e ^ { - c _ { g } m } .\tag{49}
$$

Proof. Idea. In fixed dimension, the pilot scale is a continuous function of finitely many bounded empirical moments as long as the VaR cell is correct. Laws of large numbers give consistency, while Hoeffding bounds plus the positive margin control the rare event that a genuinely positive scale is badly underestimated.

Step 1: finite empirical-moment representation. Write the affine group contribution as $\mathcal { T } _ { q } ( W ; U ) =$ $a _ { g } \bar { ( } W ) \bar { + } B _ { g } ( \bar { W } ) U$ . Let $Y _ { g } ( W )$ collect the finitely many entries of $a _ { g } ( W )$ and $B _ { g } ( \breve { W } )$ and their pairwise products. The empirical first moments of $a _ { g }$ and $B _ { g }$ determine $\widehat { \mathbf { b } } ^ { ( 0 ) }$ and $\widehat { M } ^ { \left( 0 \right) }$ , hence $\widehat { U } ^ { ( 0 ) }$ and $\widehat { r } ^ { ( 0 ) }$ . Expanding the empirical variance in equation 47 then introduces only empirical averages of pairwise products of entries of $a _ { g }$ and $B _ { g } ;$ no higher empirical moments are needed. These features are bounded, and all quantities in equation 47 are therefore functions of their groupwise empirical means ${ \widehat { \mu } } ;$ let $\mu$ denote the corresponding vector of population means. On the correct VaR cell,

$$
\widehat { U } ^ { ( 0 ) } = \sum _ { j = 0 } ^ { H - 1 } ( \widehat { M } ^ { ( 0 ) } ) ^ { j } \widehat { \mathbf { b } } ^ { ( 0 ) } , \qquad \widehat { r } ^ { ( 0 ) \top } = e _ { \widehat { x } _ { 0 } } ^ { \top } \sum _ { j = 0 } ^ { H - 1 } ( \widehat { M } ^ { ( 0 ) } ) ^ { j } ,
$$

so $\widehat { \sigma } _ { g } ^ { 2 } = F _ { g } ( \widehat { \mu } )$ for a polynomial $F _ { g }$ with $F _ { g } ( \mu ) = \sigma _ { g } ^ { 2 } .$

Step 2: consistency. The strong law gives $\widehat \mu \to \mu$ almost surely. The positive margin and the CDF bound used in equation 34 make the pilot VaR cell eventually correct almost surely. Continuity of $F _ { g }$ proves equation 48. Bounded features, stop-loss coordinates, and the deterministic resolvent bound also give the uniform constant $B _ { \sigma }$

Step 3: lower-tail protection for active groups. If $\sigma _ { g } > 0 , F _ { g }$ is Lipschitz on a compact neighborhood of $\mu .$ Choose that neighborhood so $| F _ { g } ( \widehat { \mu } ) - \sigma _ { g } ^ { \bar { 2 } } | \leq 3 \sigma _ { g } ^ { 2 } \bar { / 4 }$ . Hoeffding’s inequality (Hoeffding, 1963) and a finite union bound show that leaving this neighborhood has probability at most $C e ^ { - c m }$ The same bound holds for a wrong pilot VaR cell by equation 34. Off these two events, ${ \widehat { \sigma } } _ { g } ^ { 2 } \geq \sigma _ { g } ^ { 2 } / { \underline { { 4 } } } .$ proving equation 49. □

The constants in Equation 49 depend on the fixed population model, including its positive influence scales and quantile margin. The bound supports the asymptotic MSE proof; it does not by itself give a pilot size computable from the data or a bound on the final estimator’s finite-sample MSE.

Matrix-free influence computation. The displayed matrices define the linear operator. A matrixfree implementation avoids forming $A ^ { - 1 }$ or the $D \times D$ covariance matrices. Compute the stop losses in increasing layer order, propagate the root adjoint in decreasing layer order, and accumulate the scalar $r ^ { \top } \mathcal { T } _ { g } ( \breve { W } ; \breve { U } )$ for each sample before estimating its variance. In the shared state–action model, a straightforward sample-based implementation uses at most $O ( H K N$ log $K + H | S | K )$ arithmetic operations for these passes with binary search on a nonuniform grid, and $O ( H | { \dot { S } } | K + { \dot { \iota } }$ $N + G )$ storage. Interpolation indices can be reused. These are upper bounds for the described construction. Measured runtimes depend on the actual grid and implementation and are reported with the experimental artifacts.

## D.2 FINITE-PILOT DESIGN STABILITY

Near an interior oracle, small score errors have a quadratic variance cost. Severe underestimation requires the asymptotic controls in Section D.3.

Proposition 3 (Local design stability). If every retained $\sigma _ { g } > 0$ and $\epsilon = \operatorname* { m a x } _ { g } | \widehat { \sigma } _ { g } - \sigma _ { g } |$ , then for sufficiently small ϵ andfloor λ,

$$
0 \le V \bigl ( \widehat { w } \bigr ) - V ^ { * } \le C \bigl ( \epsilon ^ { 2 } + \lambda ^ { 2 } \bigr )\tag{50}
$$

for afinite problem-dependent constant $C .$

Proof. Idea. At an interior oracle, allocation error has a quadratic variance cost. Set $\begin{array} { r } { S _ { \sigma } = \sum _ { g } \sigma _ { g } } \end{array}$ $p _ { g } = \sigma _ { g } / S _ { \sigma }$ , and $p _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { g } p _ { g } > 0$ . For every positive design $w$

$$
V ( w ) - V ^ { \ast } = S _ { \sigma } ^ { 2 } \sum _ { g } \frac { ( p _ { g } - w _ { g } ) ^ { 2 } } { w _ { g } } .\tag{51}
$$

To verify the identity, expand the square and use $\begin{array} { r } { \sum _ { g } p _ { g } = \sum _ { g } w _ { g } = 1 } \end{array}$ . If $G \epsilon \leq S _ { \sigma } / 2$ , normalizing the estimated scales gives

$$
\left\| \frac { \widehat { \sigma } } { \sum _ { g } \widehat { \sigma } _ { g } } - p \right\| _ { \infty } \leq \frac { 2 ( G + 1 ) \epsilon } { S _ { \sigma } } .
$$

Adding the uniform floor therefore gives $\| \widehat { w } - p \| _ { \infty } \leq 2 ( G + 1 ) \epsilon / S _ { \sigma } + \lambda$ . For sufficiently small $\epsilon , \lambda ,$ , every denominator $\widehat { w } _ { g }$ in Equation 51 is at least $p _ { \mathrm { m i n } } / 2$ . Substitution and $( a + b ) ^ { 2 } \leq 2 \bar { a ^ { 2 } } + 2 b ^ { 2 }$ prove the claim. The lower bound on the weights is local; it gives no protection on a pilot event with severe scale underestimation. □

## D.3 ORACLE ADAPTATION

Proof of Theorem 3. Idea. The pilot must make the positive-scale weights converge to the oracle design and make severe underestimation sufficiently rare for second moments. The exploration floor separately guarantees enough main samples in every group to control the nonlinear fixed-point remainder and the VaR cell. Conditional on the pilot, the remaining problem is a deterministic triangular-array CLT.

Step 1: learned weights, rounding, and a minimum main count. Write $\begin{array} { r } { S _ { \sigma } : = \sum _ { q } \sigma _ { g } > 0 } \end{array}$ and $w _ { g } ^ { \ast } : = \sigma _ { g } / S _ { \sigma }$ . Let $\widehat { C } _ { N } ^ { \mathrm { T I S } }$ denote the main-sample estimator produced by Algorithm 1. Define

$$
\widetilde { p } _ { g } : = \left\{ \begin{array} { l l } { \widehat { \sigma } _ { g } / \sum _ { j } \widehat { \sigma } _ { j } , } & { \sum _ { j } \widehat { \sigma } _ { j } > 0 , } \\ { 1 / G , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \qquad \widehat { w } _ { g } : = ( 1 - \lambda _ { N } ) \widetilde { p } _ { g } + \lambda _ { N } / G .
$$

The second branch also gives $\widehat { w } _ { g } = 1 / G$ . Pilot consistency and $S _ { \sigma } > 0$ imply that its probability tends to zero. For pilots redrawn at each budget, the convergence needed below is in probability:

$$
\widehat { w } _ { g } \stackrel { p } { \longrightarrow } w _ { g } ^ { * } \quad \mathrm { f o r e v e r y g r o u p , i n c l u d i n g } w _ { g } ^ { * } = 0 .\tag{52}
$$

Set $J _ { N } : = N - G m _ { N } - 2 G$ and $n _ { g } : = 2 + \mathrm { L R M } _ { g } ( J _ { N } , \widehat { w } )$ . Largest-remainder rounding starts from $\lfloor J _ { N } \widehat { w } _ { g } \rfloor$ and assigns the leftover calls to the largest fractional remainders, with ties broken in a fixed group order. It satisfies

$$
\sum _ { g } \mathrm { L R M } _ { g } ( J _ { N } , \widehat { w } ) = J _ { N } , \qquad | \mathrm { L R M } _ { g } ( J _ { N } , \widehat { w } ) - J _ { N } \widehat { w } _ { g } | < 1 .
$$

Thus the pilot and main counts sum to N. Since $J _ { N } / N \to 1$ , uniformly in $g _ { \colon }$

$$
\frac { n _ { g } } { N } - \widehat { w } _ { g } = \left( \frac { J _ { N } } { N } - 1 \right) \widehat { w } _ { g } + O ( N ^ { - 1 } ) = o ( 1 ) .\tag{53}
$$

Moreover, $\widehat { w } _ { g } \geq \lambda _ { N } / G$ and $\mathrm { L R M } _ { g } ( J _ { N } , \widehat { w } ) \geq J _ { N } \widehat { w } _ { g } - 1$ , so, eventually,

$$
n _ { \mathrm { m i n } } \geq \frac { N \lambda _ { N } } { 2 G } .\tag{54}
$$

Step 2: conditional CLTfor the leading influence term. Conditional on the pilot, the main samples are independent and their counts are fixed. Let

$$
Z _ { N } : = \sum _ { g } \frac { 1 } { n _ { g } } \sum _ { i = 1 } ^ { n _ { g } } \phi _ { g } ( W _ { g , i } )
$$

be the leading influence term. If $\sigma _ { g } = 0$ , then centering and zero variance imply $\phi _ { g } = 0$ almost surely. Let $\mathcal { A } : = \{ g : \sigma _ { g } > 0 \}$ . This set is nonempty because $S _ { \sigma } > 0$ , and $\begin{array} { r } { w _ { \operatorname* { m i n } } ^ { * } : = \operatorname* { m i n } _ { g \in \mathcal { A } } w _ { g } ^ { * } > 0 } \end{array}$ because A is finite. By equation 53, the pilot event

$$
B _ { N } : = \left. \operatorname* { m i n } _ { g \in \mathcal { A } } \frac { n _ { g } } { N } \geq \frac { w _ { \mathrm { m i n } } ^ { * } } { 2 } \right.
$$

satisfies $\mathbb { P } ( B _ { N } )  1$ . Conditional on a pilot in $B _ { N }$ , the summands of $\sqrt { N } Z _ { N }$ are independent and centered. Because the finite collection of influences is bounded, there is a deterministic $M _ { 3 } < \infty$ with $\mathbb { E } | \phi _ { g } | ^ { 3 } \le M _ { 3 }$ for every g, and

$$
\sum _ { g \in \mathcal { A } } \sum _ { i = 1 } ^ { n _ { g } } \mathbb { E } [ | \frac { \sqrt { N } } { n _ { g } } \phi _ { g } ( W _ { g , i } ) | ^ { 3 } | \operatorname { p i l o t } ] \leq M _ { 3 } N ^ { 3 / 2 } \sum _ { g \in \mathcal { A } } \frac { 1 } { n _ { g } ^ { 2 } } \leq \frac { 4 M _ { 3 } | \mathcal { A } | } { ( w _ { \operatorname* { m i n } } ^ { * } ) ^ { 2 } \sqrt { N } } \longrightarrow 0 .
$$

The conditional variance is

$$
N \sum _ { g \in \mathcal { A } } \frac { \sigma _ { g } ^ { 2 } } { n _ { g } } \stackrel { p } {  } \sum _ { g \in \mathcal { A } } \frac { \sigma _ { g } ^ { 2 } } { w _ { g } ^ { * } } = S _ { \sigma } ^ { 2 } = V ^ { * } > 0 .
$$

Consequently, on an event whose pilot probability tends to one, the variance is bounded away from zero; dividing the preceding third-moment bound by its $3 / 2$ power verifies Lyapunov’s condition, and hence Lindeberg’s condition. The Lindeberg–Feller theorem applied conditionally on the pilot (van der Vaart, 1998, Proposition 2.27) therefore makes the conditional characteristic function of $\sqrt { N } Z _ { N }$ converge in probability to that of $\mathcal { N } ( 0 , V ^ { * } )$ . Characteristic functions are bounded by one, so taking expectations over the pilot yields the unconditional convergence $\sqrt { N } Z _ { N } \Rightarrow \mathcal { N } ( 0 , V ^ { * } )$

Step 3: nonlinear fixed-point remainder. The fixed-point remainder also remains negligible. Conditional on the pilot, Lemma 2 gives $\mathbb { E } [ \| R _ { N } \| ^ { 2 }$ | pilot $] \overset { \cdot } { \leq } C n _ { \operatorname* { m i n } } ^ { - 2 }$ with deterministic C. By equation 54,

$$
\sqrt { N } \| R _ { N } \| = O _ { p } \Big ( ( \sqrt { N } \lambda _ { N } ) ^ { - 1 } \Big ) = o _ { p } ( 1 ) , \qquad N \mathbb { E } \| R _ { N } \| ^ { 2 } = O \big ( ( N \lambda _ { N } ^ { 2 } ) ^ { - 1 } \big ) = o ( 1 ) .\tag{55}
$$

Together with the quantile-cell argument below, this proves the CLT.

Step 4: uniform integrability of the leading variance. For the normalized MSE, independence and centering give

$$
\mathbb { E } [ N Z _ { N } ^ { 2 } \mid \mathrm { p i l o t } ] = N \sum _ { g : \sigma _ { g } > 0 } { \frac { \sigma _ { g } ^ { 2 } } { n _ { g } } } .\tag{56}
$$

For a group with $\sigma _ { g } \ > \ 0 ;$ , let $A _ { g , N } : = \{ \widehat { \sigma } _ { g } \geq \sigma _ { g } / 2 \}$ . Since $\begin{array} { r } { \sum _ { i } \widehat { \sigma } _ { j } \ \leq \ G B _ { \sigma } } \end{array}$ and eventually $1 - \lambda _ { N } \geq 1 / 2 .$ on $A _ { g , N } , \widehat { w } _ { g } \geq \sigma _ { g } / ( 4 G B _ { \sigma } )$ . On $A _ { g , N } ^ { c } , \widehat { w } _ { g } \geq \lambda _ { N } ^ { - } / G$ . Also $n _ { g } \ \ge \ J _ { N } \widehat { w } _ { g }$ and eventually $N / J _ { N } \leq 2 ,$ so $N / n _ { g } \le 2 / \widehat { w } _ { g }$ . Lemma 3 gives

$$
\mathbb { E } \bigg [ \frac { N } { n _ { g } } \mathbf { 1 } _ { A _ { g , N } ^ { c } } \bigg ] \leq \frac { 2 G C _ { g } } { \lambda _ { N } } e ^ { - c _ { g } m _ { N } } \longrightarrow 0 ,
$$

because log $( 1 / \lambda _ { N } ) = o ( m _ { N } )$ . On $A _ { g , N } , N / n _ { g }$ is uniformly bounded and converges in probability to $1 / w _ { g } ^ { * }$ . Hence equation 56 converges in expectation to $V ^ { * }$

Step 5: wrong VaR cells and conclusion. Finally, conditional on the pilot, equation 34 and equation 54 bound the wrong-VaR-cell contribution to the normalized MSE by $C N \mathrm { \bar { e x p } } ( - c N \lambda _ { N } m _ { \alpha } ^ { 2 } ) =$ o(1). Cauchy–Schwarz removes the cross term with the $L ^ { 2 } .$ -negligible remainder. Thus

$$
\sqrt { N } ( \widehat C _ { N } ^ { \mathrm { T I S } } - C _ { \alpha , K } ) \Rightarrow { \cal N } ( 0 , V ^ { * } ) , \qquad { \cal N } \mathbb { E } [ ( \widehat C _ { N } ^ { \mathrm { T I S } } - C _ { \alpha , K } ) ^ { 2 } ]  V ^ { * } ,
$$

which completes the proof.

## D.4 ANCHORED DESIGN: VARIANCE SAFEGUARD AND ASYMPTOTIC MSE

For the anchor in Equation 10, we define the occupancy scores, prove the component-relative variance safeguard, and establish Corollary 1.

Let ${ \widehat { \mu } } _ { h } ( s )$ denote the state occupancy induced from the root by the fixed policy and the pilot transition estimate, with h steps remaining. In the stationary shared-kernel model, define

$$
\widehat { o } _ { s , a } : = \sum _ { h : ( h , s ) \in B } \widehat { \mu } _ { h } ( s ) \pi _ { h } ( a \mid s ) .\tag{57}
$$

More generally, when a query group feeds several Bellman rows with known mixture coefficients, $\widehat { o } _ { g }$ is the sum of the corresponding pilot-model row visitation probabilities times those coefficients. In the untied policy-mixture case $\bar { b } = ( h , s )$ , this reduces to $\widehat { o } _ { b } = \widehat { \mu } _ { h } ( s )$

Let

$$
p _ { g } ^ { \mathrm { i n f } } = \frac { \widehat { \sigma } _ { g } } { \sum _ { j } \widehat { \sigma } _ { j } } , \qquad p _ { g } ^ { \mathrm { o c c } } = \frac { \widehat { \sigma } _ { g } } { \sum _ { j } \widehat { \sigma } _ { j } } , \qquad p _ { g } ^ { \mathrm { a n c } } = \frac { 1 } { 2 } p _ { g } ^ { \mathrm { i n f } } + \frac { 1 } { 2 } p _ { g } ^ { \mathrm { o c c } } ,
$$

using the uniform branch for $p ^ { \operatorname { i n f } }$ if every estimated influence scale is zero. The occupancy denominator is positive because the retained rows include the root and $H \geq 1$ . Apply the same exploration floor to each component, $w ^ { x } = ( 1 - \lambda ) p ^ { x } + \lambda \mathbf { 1 } / G$ for $x \in \{ \operatorname { i n f } , \mathrm { o c c } , \mathrm { a n c } \}$

Proposition 4 (Component-relative safeguard). For the positive fractional designs before integer rounding,

$$
V ( w ^ { \mathrm { a n c } } ) \leq 2 \operatorname* { m i n } \{ V ( w ^ { \mathrm { i n f } } ) , V ( w ^ { \mathrm { o c c } } ) \} .\tag{58}
$$

Proof. The common floor gives $w ^ { \mathrm { a n c } } = ( w ^ { \mathrm { i n f } } + w ^ { \mathrm { o c c } } ) / 2$ , so for every group

$$
\begin{array} { r l r } { w _ { g } ^ { \mathrm { a n c } } \geq \frac { 1 } { 2 } w _ { g } ^ { \mathrm { i n f } } , } & { { } } & { w _ { g } ^ { \mathrm { a n c } } \geq \frac { 1 } { 2 } w _ { g } ^ { \mathrm { o c c } } . } \end{array}
$$

Because $\begin{array} { r } { V ( w ) = \sum _ { g } \sigma _ { g } ^ { 2 } / w _ { g } } \end{array}$ is decreasing in each coordinate separately,

$$
V ( w ^ { \mathrm { a n c } } ) \leq 2 V ( w ^ { \mathrm { i n f } } ) , \qquad V ( w ^ { \mathrm { a n c } } ) \leq 2 V ( w ^ { \mathrm { o c c } } ) ,
$$

which proves the claim.

Choice of mixture weight. For occupancy weight $\beta \in ( 0 , 1 )$ , the same argument gives

$$
V { \big ( } ( 1 - \beta ) w ^ { \mathrm { i n f } } + \beta w ^ { \mathrm { o c c } } { \big ) } \leq \operatorname* { m i n } \left\{ { \frac { V ( w ^ { \mathrm { i n f } } ) } { 1 - \beta } } , { \frac { V ( w ^ { \mathrm { o c c } } ) } { \beta } } \right\} .
$$

The worst-case factor in this bound relative to the better component is max $\{ ( 1 - \beta ) ^ { - 1 } , \beta ^ { - 1 } \}$ , minimized at $\beta = 1 / 2 ;$ ; optimal finite-budget weights may differ.

This safeguard compares the two component designs before rounding; it does not bound finitebudget MSE. Corollary 1 identifies the limiting MSE constant of the full anchored estimator, including pilot cost and rounding. Finite-budget behavior is assessed in the workflow experiments (Sections E.2.6–E.10).

Corollary 1 (Asymptotic efficiency cost of anchoring). Under the assumptions and schedules of Theorem 3, let $\dot { v _ { g } } \doteq { o _ { g } } / \sum _ { j } o _ { j }$ be the population occupancy shares and set $a _ { g } ^ { * } = ( w _ { g } ^ { * } + v _ { g } ) / 2 .$ Using Equation 10 in Algorithm 1 gives

$$
\begin{array} { r l r } & { \sqrt { N } \big ( \widehat C _ { N } ^ { \mathrm { a n c } } - C _ { \alpha , K } \big ) \Rightarrow \mathcal { N } ( 0 , V _ { \mathrm { a n c } } ) , } & { V ^ { * } \le V _ { \mathrm { a n c } } \le 2 V ^ { * } . } \\ & { } & { N \mathbb { E } \big [ ( \widehat C _ { N } ^ { \mathrm { a n c } } - C _ { \alpha , K } ) ^ { 2 } \big ]  V _ { \mathrm { a n c } } = \displaystyle \sum _ { g : \sigma _ { g } > 0 } \frac { \sigma _ { g } ^ { 2 } } { a _ { g } ^ { * } } . \qquad } \end{array}\tag{59}
$$

ProofofCorollary 1. Idea. The occupancy shares are consistent, and the anchor retains at least half of every learned influence share. The proof of Theorem 3 therefore continues to control rare underallocation, with a changed limiting design.

Step 1: limiting shares and minimum counts. In fixed dimension, the pilot transition probabilities converge in probability to their population values. Finite-horizon occupancies are continuous functions of those probabilities, and their sum is positive. Thus the normalized pilot occupancy shares converge to v. The common floor vanishes, while Lemma 3 gives $w ^ { \mathrm { i n f } } \to w ^ { * }$ in probability. Consequently,

$$
w _ { g } ^ { \mathrm { a n c } } \stackrel { p } { \longrightarrow } a _ { g } ^ { * } = \frac { 1 } { 2 } ( w _ { g } ^ { * } + v _ { g } ) .
$$

For every active group $\sigma _ { g } > 0 , a _ { q } ^ { * } \ge w _ { q } ^ { * } / 2 > 0 .$ . Both floored components have weights at least $\lambda _ { N } / G _ { ; }$ , so the anchored counts obey the same lower bound $n _ { \mathrm { m i n } } \geq N \lambda _ { N } / ( 2 G )$ eventually as in Equation 54. Rounding and the vanishing pilot fraction give $n _ { g } ^ { \mathrm { a n c } } / N \to a _ { g } ^ { * }$ in probability.

Step 2: the influence term and its second moment. Conditional on the pilot, the main observations are independent. The bounded-influence conditional CLT used in Theorem 3 gives a Gaussian limit with variance $V _ { \mathrm { a n c } } ;$ zero-scale groups contribute nothing. To justify convergence of second moments,

couple the hypothetical plain and anchored allocations to the same pilot. Write $J _ { N } = N - G m _ { N } -$ 2G, and let $\dot { n } _ { g } ^ { \mathrm { i n f } }$ be the plain count. Largest-remainder rounding gives

$$
\begin{array} { r } { n _ { g } ^ { \mathrm { a n c } } \geq 1 + J _ { N } w _ { g } ^ { \mathrm { a n c } } \geq 1 + \frac { 1 } { 2 } J _ { N } w _ { g } ^ { \mathrm { i n f } } \geq \frac { 1 } { 2 } \big ( n _ { g } ^ { \mathrm { i n f } } - 1 \big ) \geq \frac { 1 } { 4 } n _ { g } ^ { \mathrm { i n f } } , } \end{array}
$$

where $n _ { g } ^ { \mathrm { i n f } } \leq 3 + J _ { N } w _ { g } ^ { \mathrm { i n f } }$ and $n _ { g } ^ { \mathrm { i n f } } \geq 2$ were used. Hence

$$
0 \leq N \sum _ { g } \frac { \sigma _ { g } ^ { 2 } } { n _ { g } ^ { \mathrm { a n c } } } \leq 4 N \sum _ { g } \frac { \sigma _ { g } ^ { 2 } } { n _ { g } ^ { \mathrm { i n f } } } .
$$

To make the moment transfer explicit, set $\begin{array} { r } { Y _ { N } ^ { \mathrm { i n f } } : = N \sum _ { g } \sigma _ { g } ^ { 2 } / n _ { g } ^ { \mathrm { i n f } } } \end{array}$ and $\begin{array} { r } { Y _ { N } ^ { \mathrm { a n c } } : = N \sum _ { g } \sigma _ { g } ^ { 2 } / n _ { g } ^ { \mathrm { a n c } } } \end{array}$ Step 4 of Theorem 3 gives $Y _ { N } ^ { \mathrm { i n f } } \ \to \ V ^ { * }$ in probability and $\mathbb { E } Y _ { N } ^ { \mathrm { i n f } } \to V ^ { * }$ . Since these variables are nonnegative, this implies $\dot { Y } _ { N } ^ { \mathrm { i n f } }  V ^ { * }$ in $L ^ { 1 }$ and hence uniform integrability. The domination $0 \leq Y _ { N } ^ { \mathrm { a n c } } \leq 4 Y _ { N } ^ { \mathrm { i n f } }$ therefore makes $\{ Y _ { N } ^ { \mathrm { a n c } } \}$ } uniformly integrable as well. Because $Y _ { N } ^ { \mathrm { a n c } }  V _ { \mathrm { a n c } }$ in probability by Step 1, uniform integrability yields $\mathbb { E } Y _ { N } ^ { \mathrm { a n c } } \to V _ { \mathrm { a n c } }$ . This proves the normalized-MSE limit for the leading influence term.

Step 3: remainder, quantile, and efficiency cost. The common minimum-count bound gives the same vanishing normalized second moment of the Bellman remainder and the same exponentially small wrong-quantile contribution as in Theorem 3. The cross term vanishes by Cauchy–Schwarz. The CLT and MSE limit thus hold for the full CVaR estimator. Finally, $a _ { g } ^ { * } \geq w _ { g } ^ { * } / 2$ on active groups implies

$$
V _ { \mathrm { a n c } } \leq 2 \sum _ { g : \sigma _ { g } > 0 } \frac { \sigma _ { g } ^ { 2 } } { w _ { g } ^ { * } } = 2 V ^ { * } .
$$

Cauchy–Schwarz gives $V _ { \mathrm { a n c } } \geq V ^ { * }$ for any probability allocation, including allocations with zero weights only on zero-influence groups. □

## D.5 WHEN LEARNING AN ALLOCATION REPAYS ITS PILOT

The variance identity in Equation 51 separates allocation opportunity from learning error. It remains valid when some influences vanish, provided $\begin{array} { r } { S _ { \sigma } = \sum _ { q } \sigma _ { g } ^ { \mathrm { ~ ~ } } > 0 } \end{array}$ and the evaluated design is positive. Let $n = N - G m , \rho = G m / N$ , retain the oracle shares $w _ { q } ^ { * } = \sigma _ { g } / S _ { \sigma }$ from Equation $^ { 6 , }$ and let $\widetilde { w } _ { g } = n _ { g } / n$ be the actual main-sample fractions after rounding. Define

$$
D ( w ^ { * } \| w ) : = \sum _ { g } \frac { ( w _ { g } ^ { * } - w _ { g } ) ^ { 2 } } { w _ { g } } .
$$

Because both $w ^ { * }$ and w sum to one,

$$
D ( w ^ { * } \| w ) = \sum _ { g } \frac { ( w _ { g } ^ { * } ) ^ { 2 } } { w _ { g } } - 1 .
$$

Conditional on the pilot, the centered leading influence term $\begin{array} { r } { L _ { N } = \sum _ { g } n _ { g } ^ { - 1 } \sum _ { i } \phi _ { g } ( W _ { g , i } ) } \end{array}$ therefore satisfies

$$
N \mathbb { E } [ L _ { N } ^ { 2 } \mid \mathrm { p i l o t } ] = N \sum _ { g } { \frac { \sigma _ { g } ^ { 2 } } { n _ { g } } } = { \frac { N } { n } } V ^ { * } \sum _ { g } { \frac { ( w _ { g } ^ { * } ) ^ { 2 } } { \widetilde { w } _ { g } } } = { \frac { V ^ { * } } { 1 - \rho } } \{ 1 + D ( w ^ { * } \| \widetilde { w } ) \} .
$$

Averaging over the pilot gives the exact identity

$$
N \mathbb { E } [ L _ { N } ^ { 2 } ] = \frac { V ^ { \ast } } { 1 - \rho } \left\{ 1 + \mathbb { E } D ( w ^ { \ast } \| \widetilde { w } ) \right\} .\tag{60}
$$

Here the expectation on the right is over pilots. For a deterministic baseline using all N queries without a pilot and positive actual query fractions $v ,$ write $A _ { v } = V ( v ) / V ^ { * }$ . Its leading variance is $V ( v ) / N$ . Learning improves on this benchmark at the level of the influence term exactly when

$$
1 + \mathbb { E } D ( w ^ { * } \| \widetilde { w } ) < ( 1 - \rho ) A _ { v } .\tag{61}
$$

The three quantities have distinct roles: $A _ { \ i }$ is the available allocation advantage, D penalizes inaccurate shares, especially underallocation, and $\rho$ charges the discarded pilot. For two learned methods with the same pilot size, the common factor $1 / ( 1 - \rho )$ cancels, so their comparison depends on their expected allocation penalties. These are identities for the linearized error, not finite-budget guarantees for the nonlinear CVaR estimator. Population influences are needed to evaluate them, so their use in the experiments is diagnostic rather than an operational rule for choosing a method.

## E APPROXIMATION AND EXPERIMENTAL PROTOCOLS

This section closes two gaps left by the asymptotic allocation theory. Appendix E.1 controls error from the categorical grid; the remaining subsections give the protocols and evidence supporting Q1– Q5, including the regimes where tail targeting helps, where it does not, and why small pilots can fail. Each primary protocol defines its query and charged budget.

## E.1 REPRESENTATION ERROR

A fixed grid introduces error even with exact conditional laws. The following bound justifies the separation of approximation and sampling error at the end of Section 2. Let $Z _ { h } ^ { \bar { * } } ( s ) \sim \bar { \eta _ { h } ^ { * } } ( s )$ denote the projected h-step return and $G _ { h } ( s )$ its true-return counterpart. For every state s,

$$
\begin{array} { r } { \vert \mathrm { C V a R } _ { \alpha } ( \eta _ { H } ^ { \ast } ( s ) ) - \mathrm { C V a R } _ { \alpha } ( G _ { H } ( s ) ) \vert \le H \Delta . } \end{array}\tag{62}
$$

Proof. Realize each categorical projection as randomized rounding to adjacent grid atoms (Rowland et al., 2018): for $y \in [ z _ { j } , z _ { j + 1 } ]$ , set $\widetilde { y } = z _ { j }$ with probability $( z _ { j + 1 } - y ) / ( z _ { j + 1 } - z _ { j } )$ and $\widetilde y = z _ { j + 1 }$ otherwise; for $y \ge H , \mathrm { s e t } \tilde { y } = H .$ . Then $\mathcal { L } ( \bar { y } ) = \Pi _ { C } \bar { \delta } _ { y }$ and $| \widetilde { y } - y | \le \Delta$ whenever $y \le H$ (inputs below zero do not arise here). Couple the projected and true return recursions using the same actions, rewards, next states, and these rounding variables. Conditional on a coupled next state, use the inductive coupling for the two continuation returns. If their difference is at most $( h - 1 ) \Delta$ , adding the same reward preserves that difference and adjacent-grid rounding adds at most $\Delta .$ . If the projected pre-rounding value exceeds H, clipping it to H cannot increase its distance from the true h-step return, which lies in $[ 0 , h ] \subseteq [ 0 , H ]$ . Starting from equal zero-step returns, induction yields

$$
| Z _ { h } ^ { * } ( s ) - G _ { h } ( s ) | \leq h \Delta \qquad \mathrm { a l m o s t ~ s u r e l y } .\tag{63}
$$

$\operatorname { I f } \left| X - Y \right| \leq c$ almost surely, then $F _ { X } ( x - c ) \leq F _ { Y } ( x ) \leq F _ { X } ( x + c )$ for every x, which implies $| F _ { X } ^ { - 1 } ( u ) - F _ { Y } ^ { - 1 } ( u ) | \ \leq \ c$ for $u \in \mathsf { \Gamma } ( 0 , 1 )$ . Integrating over $u \in \mathsf { \Gamma } ( 0 , \alpha )$ and dividing by α gives Equation 62: the integration interval’s length cancels the factor $1 / \alpha$ □

## E.2 EXPERIMENTAL EVIDENCE AND PROTOCOLS

The protocols below support Q1–Q3 in Section 5; Section E.3 gives the held-out FinQA protocol for Q4. MSE is the average squared error over independent replications, with Monte Carlo SE equal to the sample standard deviation of squared errors divided by the square root of the replication count; bars show 1.96 SEs. Rollouts use $\lfloor N / H \rfloor$ full trajectories (leaving fewer than H transition slots unused), whereas conditional allocations exhaust N. Bold marks sample-only point minima, including displayed ties, without a superiority claim; population references are shaded. Uncertainty is conditional on the fixed tasks or panels.

Method key and common estimator. All conditional-query methods use the categorical CVaR plug-in. Fixed-score designs normalize scores, add the floor in Equation $^ { 8 , }$ and round to exhaust the budget; learned scores pay for and discard a pilot, whereas population references use exact laws without a pilot. Uniform assigns equal shares, and learned occupancy (occup./Occ.) uses pilotmodel visit counts $\widehat { o } _ { g }$ from Equation 57. Learned mean uses the pilot-estimated standard deviation of the mean-return influence; for a shared kernel,

$$
\psi _ { s , a } ( W ) = \sum _ { h = 1 } ^ { H } \mu _ { h } ( s ) { \pi } _ { h } ( a \mid s ) \{ R + v _ { h - 1 } ( S ^ { \prime } ) - \mathbb { E } _ { P _ { s , a } } [ R + v _ { h - 1 } ( S ^ { \prime } ) ] \} ,
$$

where $\mu _ { h } ( s )$ is visitation with h steps left and $\begin{array} { r c l } { v _ { h } ( s ) } & { = } & { \mathbb { E } [ G _ { h } ( s ) ] } \end{array}$ ; an untied block uses $\mu _ { h } ( s ) \dot { \mathrm { s d } } _ { P _ { b } } ( R + v _ { h - 1 } ( S ^ { \prime } ) )$ . Occupancy (population) and mean influence (population) use the corresponding exact scores, and complete rollout (rollout/Roll.) takes empirical CVaR of full fixed-policy returns (Thomas & Learned-Miller, 2019).

Plain/shared TIS uses ${ \widehat { \sigma } } _ { g } ;$ anchored TIS (anchored/Anch.) averages the tail and occupancy designs. Oracle+floor uses exact $\sigma _ { g }$ and is a population allocation reference, not a finite-budget MSE lower bound. Reachability only and local scale only replace $\sigma _ { b } = d _ { b } \tau _ { b }$ by exact $d _ { b } \ \mathrm { o r } \ \tau _ { b }$ (Section C.3), separating threshold-dependent adjoint mass from local shortfall variability; $d _ { b }$ need not equal occupancy. TIS no-cov removes cross-layer covariance from the allocation score while retaining pooled estimation, with oracle no-cov its population analogue; untied TIS fits separate layer kernels under the same total budget. MC-UCB (frozen) sequentially allocates using uncertainty in pilot-frozen influence scores (Section E.2.4), and pilot answer entropy uses Shannon entropy of pilot answer frequencies, summed over confidence labels.

![](images/cd14e60a60bca21d202a646c416558143d6816f93465664cd5097188c7c0d85d.jpg)  
Figure 3: Controlled H = 5 MRP (21 untied blocks): CVaR MSE versus total queries for learned methods and population references; bars are 1.96 Monte Carlo SEs.

## E.2.1 CONTROLLED STOCHASTIC MARKOV REWARD PROCESS

This auxiliary multi-step check supports the mechanism behind Q1–Q2: in the untied setting, the oracle score combines propagation to the root and local shortfall variability. The $H = 5$ MRP has five states per layer, rewards in {0, 1/2, 1}, grid spacing $\Delta = 1 / 2$ , and stochastic rewards and transitions in every block, giving G = 21 independently sampled layer–state blocks. $\mathbf { A } \mathbf { t } { \boldsymbol { \alpha } } = . 1 .$ exact enumeration gives CVaR .9086716, margin .0106854, and $V _ { \mathrm { u n i f } } / V ^ { * } = 1 2 . 9 8 4 1 6 / 7 . 5 3 4 3 7 =$ 1.72332.

We use total budgets 12,500–100,000, the common $m _ { N } = \lceil 4 N ^ { 2 / 3 } / G \rceil$ discarded pilot and $\lambda _ { N } =$ $N ^ { - 1 / 4 }$ floor, and 400 replications. At $N = 1 0 0 { , } 0 0 0$ , TIS spends 8,631 pilot queries and reaches .615 of uniform MSE; anchored TIS, learned mean, and learned occupancy are close, while rollouts are worse than uniform (Figure 3).

Across 50 perturbed stochastic instances (Dirichlet seeds 100–149), the median $V _ { \mathrm { u n i f } } / V ^ { * }$ at $\alpha =$ .05, .1, .2 is 1.95344, 1.92311, 1.82727; ratios span 1.34558–3.56479 over all instance–risk pairs, with positive margins throughout. This is a population-level breadth check.

## E.2.2 LEARNED-BUDGET RUNS FOR THE CONTROLLED SEPARATION

This experiment directly supports Q1–Q2: it asks whether a charged pilot learns the population separation of Equation 44. We use $G = 1 0 , \alpha = . 1$ , and $t \in \{ 0 , \frac 1 2 , 1 \}$ under a protocol frozen before simulation. Analytically, CVaR is .3, the margin is .09, occupancy-to-oracle variance ratios are 1, 1.29971, and 10, and conditional moments and the root law agree across t to $7 \times 1 0 ^ { - 1 7 }$

Each cell uses 1,000 replications at 25, 50, 100, 200, and 400 queries per kernel; pilot size, floor, rounding, and fallback follow the common protocol, and rollouts use the same transition budget on an independent stream. Because the one-step policy fixes occupancy at $1 / G ,$ , uniform is also the no-pilot occupancy design; occupancy + pilot discards the common pilot before using that same allocation, isolating pilot cost. At 100/400 queries per kernel the pilot consumes 40/101 draws per group $( 4 0 \% / 2 5 . 2 5 \% )$ ; learned mean, TIS, and the anchor pay the same cost, whereas uniform, rollouts, and oracle+floor do not. Rollouts randomize actions while uniform fixes equal counts. At t = 0, each group’s zero-reward probability is .01, so the pilots miss it with probability .669/.362 at 100/400 queries, explaining why learning can hurt when uniform is optimal. $\mathbf { A } { \boldsymbol { \mathrm { t } } } \ t \ = \ 1$ , the median TIS share of the informative kernel rises from .62 to .88 across budgets (oracle share 1); its 10th percentile is .10–.75 at 25–50 queries, exposing the underallocation that anchoring mitigates. Table 3 reports absolute MSEs and Monte Carlo SEs for the main-table cells.

Table 3: Absolute $\mathrm { M S E } \times 1 0 ^ { 5 }$ (Monte Carlo SE) for the six main controlled-separation settings; 1,000 replications per entry.
<table><tr><td></td><td colspan="2"> $t = 0$ </td><td colspan="2"> $\begin{array} { r } { t = \frac { 1 } { 2 } } \end{array}$ </td><td colspan="2"> $t = 1$ </td></tr><tr><td>Queries/kernel</td><td>100</td><td>400</td><td>100</td><td>400</td><td>100</td><td>400</td></tr><tr><td>Uniform</td><td>11.07 (0.48)</td><td>2.98 (0.13)</td><td>11.44 (0.50)</td><td>2.69 (0.12)</td><td>9.79 (0.44)</td><td>2.46 (0.12)</td></tr><tr><td>Occupancy + pilot</td><td>17.68 (0.79)</td><td>3.89 (0.17)</td><td>19.12 (0.86)</td><td>3.73 (0.17)</td><td>16.51 (0.76)</td><td>3.22 (0.16)</td></tr><tr><td>Learned mean</td><td>18.05 (0.79)</td><td>3.82 (0.16)</td><td>19.53 (0.92)</td><td>3.74 (0.16)</td><td>18.32 (1.07)</td><td>3.30 (0.17)</td></tr><tr><td>Complete rollout</td><td>11.16 (0.56)</td><td>2.84 (0.12)</td><td>11.27 (0.50)</td><td>2.87 (0.12)</td><td>10.84 (0.46)</td><td>2.90 (0.13)</td></tr><tr><td>TIS</td><td>59.96 (3.19)</td><td>12.82 (0.66)</td><td>41.45 (2.68)</td><td>8.47 (0.44)</td><td>2.17 (0.11)</td><td>0.40 (0.02)</td></tr><tr><td>anchored TIS</td><td>22.55 (0.98)</td><td>4.52 (0.20)</td><td>19.02 (0.97)</td><td>3.14 (0.14)</td><td>3.65 (0.17)</td><td>0.66 (0.03)</td></tr><tr><td>Oracle + floor</td><td>11.07 (0.48)</td><td>2.98 (0.13)</td><td>8.62 (0.42)</td><td>2.08 (0.09)</td><td>1.26 (0.06)</td><td>0.30 (0.01)</td></tr></table>

Table 4: Inventory MSE $\times 1 0 ^ { 3 }$ over 300 replications versus charged queries per retained block.
<table><tr><td>Method</td><td>150</td><td>300</td><td>600</td><td>1,200</td></tr><tr><td>Uniform</td><td>1.232</td><td>0.583</td><td>0.303</td><td>0.176</td></tr><tr><td>Reachability only</td><td>1.091</td><td>0.659</td><td>0.297</td><td>0.177</td></tr><tr><td>Local scale only</td><td>1.908</td><td>1.049</td><td>0.531</td><td>0.285</td></tr><tr><td>Oracle + floor</td><td>0.731</td><td>0.329</td><td>0.150</td><td>0.100</td></tr><tr><td>Learned occupancy</td><td>1.517</td><td>0.634</td><td>0.311</td><td>0.161</td></tr><tr><td>Learned mean</td><td>1.361</td><td>0.571</td><td>0.302</td><td>0.152</td></tr><tr><td>Complete rollout</td><td>2.794</td><td>1.774</td><td>0.836</td><td>0.391</td></tr><tr><td>anchôred TIS</td><td>1.012</td><td>0.456</td><td>0.217</td><td>0.137</td></tr><tr><td>TIS</td><td>1.007</td><td>0.414</td><td>0.174</td><td>0.116</td></tr></table>

## E.2.3 SEASONAL BASE-STOCK INVENTORY EVALUATION

This stage-dependent benchmark supports Q2; each $( h , s )$ is a separate query group (the untied case of Section A). Inventory is $\{ 0 , \ldots , 6 \}$ , a fixed seasonal policy orders toward four or five units over $H = 8$ , demand is truncated Poisson with seasonally varying mean, and a disruption with probability .06 (.08 at capacity) removes one extra unit and lowers the reward category. Sales, ordering, holding, and lost-demand terms determine profit, quantized to $\{ 0 , 1 / 2 , 1 \}$ on grid $\Delta = 1 / 2$

From zero initial inventory at $\alpha = . 1$ , backward structural reachability leaves $B = 4 1$ blocks. Exact enumeration gives categorical CVaR 1.1512371, margin 0.0436202, and

$$
V _ { \mathrm { u n i f } } = 7 . 5 4 1 3 3 , \qquad V ^ { \ast } = 4 . 1 3 5 4 1 , \qquad V _ { \mathrm { u n i f } } / V ^ { \ast } = 1 . 8 2 3 6 0 .
$$

Budgets are 150, 300, 600, and 1,200 queries per retained block (6,150–49,200 total calls); pilot, floor, rounding, and sample splitting follow the controlled-MRP protocol. We use 300 replications.

Pilot fractions are 22.0%, 17.3%, 13.8%, 10.9%. TIS has the lowest observed learned-method MSE at every budget (Table 4); at 1,200 queries per block it is 15.4% above oracle+floor. Rollouts exceed uniform throughout: with $H = 8 ,$ , 49,200 transitions yield only 6,150 returns, or 615 returns’ worth of mass in the worst decile.

At 600 queries per block, 300 independent pilots per setting isolate allocation learning through $V ( \widehat { w } ) / \bar { V ^ { \ast } }$ ; this excludes the main-sample cost of larger pilots, for which the leading cost-inclusive MSE is $V ( \widehat { w } ) / ( N - G m _ { N } )$ . Table 5 gives medians and 90th percentiles; smaller exploration exponents correspond to larger floors.

## E.2.4 SLIPPERY CLIFFWALKING WITH STATIONARY STATE–ACTION KERNELS

Figure 2(a) provides the shared-kernel Q2 benchmark: stationary state–action laws are reused across Bellman stages. We use Gymnasium slippery CliffWalking-v1 (Towers et al., 2025), a $, 4 \times 1 2$ grid with start $^ { 3 6 , }$ goal $^ { 4 7 , }$ cliff cells $^ { 3 7 - 4 6 }$ , and four actions; each action realizes its intended or either perpendicular direction with probability $1 / 3 ,$ boundary moves stay put, cliffs reset to start, and the fixed-horizon wrapper makes the goal absorbing.

Table 5: Inventory pilot sensitivity: median (90th percentile) $V ( \widehat { w } ) / V ^ { * }$ over 300 pilots.
<table><tr><td>Pilot fraction</td><td> $\lambda = N ^ { - 1 / 6 }$ </td><td> $N ^ { - 1 / 4 }$ </td><td> $N ^ { - 1 / 3 }$ </td></tr><tr><td>3.7%</td><td>1.124 (1.184)</td><td>1.111 (1.184)</td><td>1.123 (1.212)</td></tr><tr><td>7.2%</td><td>1.082 (1.108)</td><td>1.057 (1.087)</td><td>1.054 (1.088)</td></tr><tr><td>14.2%</td><td>1.064 (1.073)</td><td>1.034 (1.043)</td><td>1.028 (1.038)</td></tr><tr><td>28.3%</td><td>1.055 (1.060)</td><td>1.023 (1.028)</td><td>1.015 (1.020)</td></tr></table>

Table 6: CliffWalking MSE $\times 1 0 ^ { 3 }$ over 500 replications versus charged queries per reachable kernel.
<table><tr><td>Method</td><td>50</td><td>100</td><td>200</td><td>400</td></tr><tr><td>Uniform</td><td>1683.126</td><td>818.420</td><td>363.617</td><td>216.877</td></tr><tr><td>Occupancy (population)</td><td>26.269</td><td>12.474</td><td>5.727</td><td>2.741</td></tr><tr><td>Mean influence (population)</td><td>19.182</td><td>9.731</td><td>4.371</td><td>2.164</td></tr><tr><td>Learned mean</td><td>27.413</td><td>11.367</td><td>5.349</td><td>2.307</td></tr><tr><td>Learned occupancy</td><td>35.792</td><td>15.924</td><td>6.450</td><td>3.189</td></tr><tr><td>Complete rollout</td><td>65.883</td><td>32.221</td><td>16.257</td><td>7.954</td></tr><tr><td>MC-UCB (frozen)</td><td>1863.426</td><td>840.730</td><td>351.955</td><td>171.316</td></tr><tr><td>Oracle no-cov</td><td>16.203</td><td>7.462</td><td>3.416</td><td>1.655</td></tr><tr><td>Oracle + floor</td><td>16.288</td><td>7.631</td><td>3.434</td><td>1.674</td></tr><tr><td>TIS no-cov</td><td>29.979</td><td>10.821</td><td>4.242</td><td>1.875</td></tr><tr><td>anchored TIS</td><td>28.427</td><td>11.551</td><td>5.362</td><td>2.182</td></tr><tr><td>TIS</td><td>29.560</td><td>10.755</td><td>4.305</td><td>1.884</td></tr></table>

We set $H = 2 0$ and map raw rewards −100, −1, 0 to 0, .99, 1. With zero reward after termination, the normalized return is exactly $2 0 + G _ { \mathrm { e p i s o d i c } } / 1 0 0$ , so the transform preserves lower-tail ordering and CVaR. The grid contains every sum of 20 elements of {0, .99, 1} (231 atoms), hence the Bellman recursion is exact. The primary policy is ϵ = .05-soft around a route crossing row 2 immediately above the cliff; a prespecified safe diagnostic policy crosses row 1, and off-route states first steer toward the chosen corridor. Structural reachability leaves 149 state–action groups, including the absorbing goal. For the primary policy at $\alpha = . 1 \cdot$ , CVaR is 11.8709354, the margin is 0.0139085, and $V _ { \mathrm { u n i f } } / \bar { V } ^ { * } = 8 4 . 6 9 \bar { 7 } 3 ;$ ; across both policies and α $\in \{ . 0 5 , . 1 , . 2 \}$ , exact ratios span 77.5748– 116.3908. Their minimum route lengths are 13 and 15, so $H = 2 0$ allows completion and slippery deviations with a tractable exact grid. All reported methods evaluate the primary policy; the safe policy is a separate policy–risk check.

Population occupancy and mean-influence references use exact scores with the same $N ^ { - 1 / 4 }$ floor; learned counterparts use the pilot fit. The discarded pilot is $m _ { N } = \mathrm { m a x } \{ 8 , \lceil 4 N ^ { 2 / 3 } / 1 4 9 \rceil \}$ per group. For MC-UCB (Carpentier et al., 2015), whose original guarantee concerns fixed-stratum weighted means rather than our Bellman estimator, we freeze the pilot influence scores and treat fresh main draws as bounded score arms. After two draws per group it selects the largest

$$
B _ { g , t } = \frac { 1 / G } { T _ { g , t - 1 } } \left( \widehat { s } _ { g , t - 1 } + \frac { 2 \beta } { \sqrt { T _ { g , t - 1 } } } \right) ,
$$

where $T _ { g , t - 1 }$ is the main-sample count and $\widehat { s } _ { g , t - 1 }$ the score standard deviation. We use the published bounded-arm choices $\delta = n ^ { - 9 / 2 }$ and $\beta = c \sqrt { \log ( 2 / \delta ) }$ , with n the main budget and c the largest frozen-arm range. Positive affine rescaling leaves the rule unchanged; computing c uses the public three-outcome support but not its probabilities, which is extra information relative to unknownsupport simulators. MC-UCB and TIS share the charged pilot, exhaust the same main budget, use the same categorical plug-in, and select no hyperparameter from outcomes. We use 500 independent replications. The no-covariance ablations in Table 6 show that most gains here come from pooling reused kernels; the theoretically required cross-layer covariance has only a small numerical effect.

Pilot fractions are 22.0%, 17.0%, 13.0%, 10.25%. At 50 queries per kernel, TIS is resolved worse than population mean influence and oracle+floor, while its differences from population occupancy and learned mean are unresolved; it beats population occupancy from 100 queries and population mean at 400. At 400, MSE reductions are 31.3%, 12.9%, and 18.3% versus population occupancy, population mean, and learned mean. Deleting covariance from the learned score yields no resolved difference at any budget. Rollouts beat uniform but trail the occupancy and influence designs, except MC-UCB. MC-UCB’s mean realized first-order variance ratios are 96.69, 85.80, 76.18, 67.77 versus uniform’s 84.70; under its published significance schedule, the confidence bonus dominates at these pull counts, keeping allocation near uniform and yielding only modest improvement at larger budgets.

## E.2.5 ADDITIONAL PUBLIC STATIONARY GYMNASIUM ENVIRONMENTS

These breadth checks test whether the shared-kernel conclusions extend beyond CliffWalking and expose a regime in which a smoother mean score can be preferable. We use stationary FrozenLake-v1 and rainy Taxi-v4 (Towers et al., 2025), with fixed policies, reused state kernels, the CliffWalking budgets and estimator, and prespecified screening for at least two stochastic reachable kernels, nonzero oracle influence variance, a positive categorical margin, and an exact finite grid.

FrozenLake. The slippery $8 \times 8$ task leaves 50 reachable nonterminal kernels under a fixed successmaximizing policy. Its failure probability .13704 makes the first-order tail signal zero at $\alpha = . 0 5 , . 1$ so α = .2 is the smallest prespecified passing level; there CVaR is .3147769, the margin is .0629554, and $V _ { \mathrm { u n i f } } / V ^ { * } = 3 . 9 2 5 8 9$ . Rainy Taxi. The fixed shortest-route policy leaves 37 reachable nonterminal kernels. The positive affine reward transform $( r + 1 ) / 2 1$ gives the exact grid $\{ 0 , \ldots , 2 2 0 \} / 2 1$ and preserves CVaR; at $\alpha = . 1$ , transformed CVaR is 9.0541409, the margin is .0121392, and $V _ { \mathrm { u n i f } } { } / V ^ { * } = 2 . 3 3 9 3 8$

Both tasks use 300 replications at 50–400 queries per reachable kernel. At 400 queries, TIS MSE is .00791 (.00079) on FrozenLake and .00020 (.00002) on Taxi, versus uniform .01464 (.00119) and .00033 (.00003). Learned mean is better on FrozenLake (.00577 (.00051)) and tied at displayed precision on Taxi (.00020 (.00002)); differences from population occupancy and covariance-deleted TIS are unresolved, while oracle+floor is best on both. At smaller budgets, pilot cost can make TIS worse than uniform or population references. These checks reinforce the structural limit rather than a universal win: when tail and mean signals align, the smoother mean score can be as good as or better than tail targeting.

## E.2.6 STRUCTURED LANGUAGE-MODEL WORKFLOW EVALUATION

Table 7: MMLU-Pro: anchoring mitigates plain TIS’s observed failures. Panel MSE/uniform MSE at $H = 6 , \alpha = . 1$ , 400 queries/kernel.
<table><tr><td>Generator</td><td>TIS</td><td>occup.</td><td>rollout</td><td>anchored</td><td>oracle +floor</td></tr><tr><td>Qwen3-4B</td><td>1.38</td><td>.090</td><td>.074</td><td>.057</td><td>.058</td></tr><tr><td>Phi-4-mini</td><td>.49</td><td>.113</td><td>.072</td><td>.066</td><td>.025</td></tr><tr><td>Granite-4.2-8B</td><td>.15</td><td>.126</td><td>.107</td><td>.070</td><td>.034</td></tr><tr><td>Mistral-24B</td><td>.21</td><td>.138</td><td>.127</td><td>.093</td><td>.055</td></tr><tr><td>Qwen3-32B</td><td>.41</td><td>.079</td><td>.055</td><td>.056</td><td>.028</td></tr><tr><td>GLM-4-32B</td><td>1.77</td><td>.087</td><td>.069</td><td>.065</td><td>.034</td></tr></table>

These frozen-law experiments support Q3: exact targets diagnose plain-TIS pilot failures and the effect of anchoring. We use 50 stratified ten-option MMLU-Pro questions (Wang et al., 2024). Qwen3-4B-Instruct-2507 is primary (Qwen Team, 2025b); Phi-4-mini-instruct and Granite-4.2-8B are cross-family checks (Microsoft et al., 2025; Granite Team, IBM, 2026); follow-up generators (†) are Mistral-Small-24B-Instruct-2501, Qwen3-32B (thinking disabled), and GLM-4-32B-0414 (Mistral AI, 2025; Qwen Team, 2025a; Zhipu AI, 2025). All use the same panel, prompts, decoder, policy, budgets, and methods; floor, utility, panel, and policy sensitivities are post hoc.

Each question is a separate finite-horizon Markov reward process. From $S _ { 0 } = \emptyset$ , state $\boldsymbol { S } _ { t } = \left( \boldsymbol { j } _ { t } , \boldsymbol { c } _ { t } \right)$ records the latest answer $j ~ \in ~ \{ 1 , \dots , 1 0 \}$ and confidence $c \in \{ . 1 , . . . , . 9 \}$ . The root action is solve; reviews use reconsider for $c \leq . 3 .$ , challenge for $. 4 \leq c \leq . 6 $ and ${ \tt v e r i f y }$ for $c \geq . 7 .$ . The controller tracks $h = H - t ,$ but prompts omit history and stage index, so a fixed state and action have the same next-response law at every stage. Thus the pair is Markov, the dynamics are stationary, and any retained prompt can be queried directly rather than reached by rollout.

Table 8: MMLU-Pro and FinQA workflow models; both use the confidence-band review policy.
<table><tr><td>Component</td><td>MMLU-Pro</td><td>FinQA</td></tr><tr><td>State after a call</td><td>Answer  $j$  and confidence  $c \left( 1 0 \times 9 \right.$  possibilities)</td><td>Candidate i and confidence d  $: ( 8 \times 9$  possibilities)</td></tr><tr><td>Initial action</td><td> $_ { \mathrm { s o l v e } }$ </td><td> $_ { \mathsf { S } } \mathsf { e } \bot \mathsf { e } \mathsf { c } \mathsf { t }$ </td></tr><tr><td>Number of calls</td><td> $H = 2 , 4 , 6 ,$  including the initial solve</td><td> $H = 3 \colon$  selection and two reviews</td></tr><tr><td>Stage index</td><td> $h = H - t$  calls remaining; stop at  $h = 0$ </td><td> $h = 3 , 2 , 1 , 0 ;$  stop at  $h = 0$ </td></tr><tr><td>Query groups</td><td> $1 + 9 0 = 9 1 $  per question</td><td> $1 + 7 2 = 7 3$  per question and workflow</td></tr><tr><td>Reward on  $s  s ^ { \prime }$ </td><td> $r _ { q } ( s ^ { \prime } ) \colon$  normalized Brier utility</td><td> $[ u ( s ^ { \prime } ) - u ( s ) + 1 ] / 2 \colon$  shifted utility change</td></tr><tr><td>Total return</td><td> $\textstyle \sum _ { t = 1 } ^ { H } r _ { q } ( S _ { t } )$ </td><td> $\begin{array} { r } { [ H + u ( S _ { H } ) ] / 2 , } \end{array}$  with  $u ( S _ { 0 } ) = 0$ </td></tr></table>

Response probabilities factor as restricted answer softmax times conditional confidence softmax, with each label one token. A query to $g = ( q , s , a )$ draws $S ^ { \prime }$ and reward $r _ { q } ( S ^ { \prime } )$

$$
P _ { q , s , a } ( r , s ^ { \prime } ) = P _ { \mathrm { L L M } } ( s ^ { \prime } \mid \mathrm { p r o m p t } ( q , s , a ) ) \mathbf { 1 } \{ r = r _ { q } ( s ^ { \prime } ) \} .\tag{64}
$$

For example, $( 1 , . 2 )$ selects $\mathtt { r e c o n s i d e r ; }$ response $( 3 , . 8 )$ earns $r _ { q } ( 3 , . 8 )$ and, if a call remains, selects verify. One fitted law and sample count are shared across all uses of $^ { g , }$ but its return effect depends on calls remaining; TIS therefore sums these stage effects before computing the group scale. Allocation is learned separately for every question, generator, and H. With one action per state there are 91 groups per question (4,550 per generator).

Reward and exact target. For response $y = \left( j , c \right)$ , define the ten-class forecast and normalized Brier utility (Brier, 1950)

$$
p _ { k } ( y ) = \left\{ \begin{array} { l l } { \displaystyle c , } & { \displaystyle k = j , } \\ { ( 1 - c ) / 9 , } & { \displaystyle k \neq j , } \end{array} \right. \quad \quad r _ { q } ( y ) = 1 - \frac { 1 } { 2 } \sum _ { k = 1 } ^ { 1 0 } \left( p _ { k } ( y ) - \mathbf { 1 } \{ k = j _ { q } ^ { * } \} \right) ^ { 2 } ,\tag{65}
$$

where $j _ { q } ^ { * }$ is the published correct option; no learned judge is used. The return $\begin{array} { r } { G _ { H , q } = \sum _ { t = 1 } ^ { H } r _ { q } ( S _ { t } ) } \end{array}$ includes the initial answer and every review, measuring cumulative response utility (FinQA instead uses terminal severity). Dividing by H rescales CVaR and MSE but leaves within-horizon ratios and allocations unchanged.

The grid is the union of all attainable partial-return supports for $h = 0 , \ldots , H$ plus endpoints 0, H: 121, 617, and 983 atoms at $H = 2 , 4 , { \bar { 6 } }$ . Closure makes both the categorical recursion and stop-loss interpolation exact; each question–horizon pair has its own target, margin, and scales. We use $\alpha = . 1$ primarily and .2 as a sensitivity check. Partial returns are essential: two deterministic .5 rewards have true sum 1, but backing them up on {0, 1, 2} yields $\begin{array} { r } { \frac { 1 } { 4 } \delta _ { 0 } + \frac { 1 } { 2 } \delta _ { 1 } + \frac { 1 } { 4 } \delta _ { 2 } . } \end{array}$ , preserving the mean while driving $\mathrm { C V a R } _ { . 1 }$ to $0 ;$ including .5 restores exactness. Across all 1,800 question–horizon–risk cells, full-horizon-only grids have median/90th-percentile absolute CVaR gaps . $. 0 2 / . 3 5$ , whereas closedgrid gaps are below $1 0 ^ { - 1 4 }$ . All reported runs use the closed grids; grid smoothing partly masks the primary generator’s depth failure.

Ground truth and validation. Enumerating each prompt’s 90 probabilities and applying dynamic programming gives exact targets, influences, and population references; sample-only methods receive fresh $( \bar { R , } S ^ { \prime } )$ draws. Enumeration is practical here only because the output alphabet is restricted, so the study measures logical-query efficiency under frozen laws while remaining relevant to longer outputs, restricted APIs, and stochastic tools. All kernels pass the designated directgeneration audit. Methods otherwise follow Section E.2; MC-UCB uses its sequential index.

Budgets and uncertainty. Each generator–horizon cell has $\begin{array} { r c l } { J } & { = } & { 3 0 0 } \end{array}$ replications at $\begin{array} { r l } { b } & { { } = } \end{array}$ 50, 100, 200, 400 queries per group, so $N = 9 1 b = 4 , 5 5 0 , 9 , 1 0 0 , 1 8 , 2 0 0 , 3 6 , 4 \dot { 0 } 0$ draws per question. The charged pilot $m _ { N } = \mathrm { m a x } \{ 8 , \lceil 4 N ^ { 2 / 3 } / 9 1 \rceil \}$ uses 13, 20, 31, 49 draws per group and $\lambda _ { N } = N ^ { - 1 / 4 }$ ; the remaining fresh draws follow the learned shares and rounding rule, and the pilot is excluded from the final estimate. One logical query is a constrained two-token draw; ground-truth enumeration is outside this budget. Fixed-allocation methods share outcome streams, sequential methods use separate streams, and replications are independent. For squared-error difference $D _ { q , \ell }$ between methods A and B on question q and replication $\ell ,$

Table 9: MMLU-Pro, $\alpha = . 1$ , 400 queries/shared kernel: MSE/uniform MSE. † marks follow-up generators; Table 10 gives absolute TIS MSE.
<table><tr><td>Generator</td><td>H</td><td>plain TIS</td><td>anchored TIS</td><td>learned mean</td><td>learned occup.</td><td>rollout</td><td>TIS no cov.</td><td></td><td>untied MC-UCB</td><td>oracle +floor</td></tr><tr><td>Qwen3-4B</td><td>2</td><td>.719</td><td>.030</td><td>.307</td><td>.036</td><td>.028</td><td>.719</td><td>.724</td><td>1.04</td><td>.192</td></tr><tr><td></td><td>4</td><td>1.02</td><td>.059</td><td>.707</td><td>.069</td><td>.057</td><td>1.02</td><td>4.28</td><td>1.08</td><td>.058</td></tr><tr><td></td><td>6</td><td>1.38</td><td>.057</td><td>1.00</td><td>.090</td><td>.074</td><td>1.35</td><td>7.67</td><td>1.06</td><td>.058</td></tr><tr><td>Phi-4-mini</td><td>2</td><td>.181</td><td>.027</td><td>.106</td><td>.032</td><td>.023</td><td>.181</td><td>.183</td><td>1.03</td><td>.015</td></tr><tr><td></td><td>4</td><td>.326</td><td>.047</td><td>.264</td><td>.074</td><td>.048</td><td>.327</td><td>1.33</td><td>1.11</td><td>.020</td></tr><tr><td></td><td>6</td><td>.487</td><td>.066</td><td>.425</td><td>.113</td><td>.072</td><td>.493</td><td>3.00</td><td>1.48</td><td>.025</td></tr><tr><td>Granite-4.2-8B</td><td>2</td><td>.172</td><td>.033</td><td>.038</td><td>.042</td><td>.029</td><td>.172</td><td>.160</td><td>1.06</td><td>.021</td></tr><tr><td></td><td>4</td><td>.127</td><td>.051</td><td>.065</td><td>.079</td><td>.062</td><td>.127</td><td>.834</td><td>.981</td><td>.027</td></tr><tr><td></td><td>6</td><td>.154</td><td>.070</td><td>.089</td><td>.126</td><td>.107</td><td>.164</td><td>1.56</td><td>1.09</td><td>.034</td></tr><tr><td>Mistral-24B†</td><td></td><td>2.140</td><td>.038</td><td>.041</td><td>.043</td><td>.026</td><td>.140</td><td>.123</td><td>.967</td><td>.020</td></tr><tr><td></td><td></td><td>4.148</td><td>.067</td><td>.100</td><td>.089</td><td>.075</td><td>.144</td><td>.763</td><td>1.07</td><td>.038</td></tr><tr><td></td><td>6</td><td>.214</td><td>.093</td><td>.159</td><td>.138</td><td>.127</td><td>.221</td><td>1.90</td><td>1.13</td><td>.055</td></tr><tr><td>Qwen3-32B†</td><td>2</td><td>.321</td><td>.028</td><td>.199</td><td>.030</td><td>.023</td><td>.321</td><td>.308</td><td>.956</td><td>.108</td></tr><tr><td></td><td>4</td><td>.410</td><td>.044</td><td>.373</td><td>.058</td><td>.044</td><td>.409</td><td>2.43</td><td>1.03</td><td>.036</td></tr><tr><td></td><td>6</td><td>.411</td><td>.056</td><td>.440</td><td>.079</td><td>.055</td><td>.412</td><td>3.71</td><td>1.08</td><td>.028</td></tr><tr><td>GLM-4-32B†</td><td></td><td>2.438</td><td>.035</td><td>.426</td><td>.039</td><td>.025</td><td>.438</td><td>.425</td><td>1.06</td><td>.027</td></tr><tr><td></td><td>4</td><td>1.08</td><td>.062</td><td>.790</td><td>.079</td><td>.055</td><td>1.08</td><td>4.24</td><td>1.28</td><td>.042</td></tr><tr><td></td><td></td><td>61.77</td><td>.065</td><td>1.42</td><td>.087</td><td>.069</td><td>1.80</td><td>6.95</td><td>1.29</td><td>.034</td></tr></table>

$$
D _ { \ell } = { \frac { 1 } { Q } } \sum _ { q = 1 } ^ { Q } D _ { q , \ell } , \qquad { \widehat { \Delta } } = { \frac { 1 } { J } } \sum _ { \ell = 1 } ^ { J } D _ { \ell } , \qquad { \widehat { \mathrm { s e } } } ( { \widehat { \Delta } } ) = { \frac { s _ { D } } { \sqrt { J } } } ,\tag{66}
$$

where $s _ { D } ^ { 2 } = ( J - 1 ) ^ { - 1 } \Sigma _ { \ell } ( D _ { \ell } - \widehat { \Delta } ) ^ { 2 }$ and $z = \widehat { \Delta } / \widehat { \mathrm { s e } } ( \widehat { \Delta } )$ . This fixed-panel calculation allows arbitrary within-replication dependence among questions; uncertainty is simulation error conditional on the panel, not population generalization.

Checks and main pattern. All 150 question–horizon margins are positive at each risk level, and shared and population-untied targets agree within $1 0 ^ { - 1 0 }$ . Median population $V _ { \mathrm { u n i f } } / V ^ { * }$ at $\alpha = . 1$ is 46, 52, 37, 27, 54, 42 in table order. Table 9 shows occupancy below plain TIS in all 18 settings and the anchor below both; rollouts have the lowest sample-only point MSE for all generators at $H = 2 ,$ while the anchor does so for five at H = 6. MC-UCB gives no consistent gain over uniform (.956– 1.48), population occupancy/mean are strong (.024–.103/.016–.074), and answer entropy is poor (2.09–7.74). Pooling matters more than covariance correction: deeper untied evaluation splits data across $1 + 9 0 ( H - \bar { 1 } )$ ) pilot groups, whereas deleting covariance changes little (population no-cov is within .003 of oracle+floor in uniform-MSE units). Tables 11–12 give budget and risk sensitivity.

Depth failures and anchoring. At $H = 6 .$ , plain TIS exceeds uniform MSE for Qwen3-4B (1.38) and GLM-4-32B (1.77), and more budget does not reliably remove the failures (Table 11); at H = 4 the full sweep reaches 1.10 and 1.21. At $\alpha = . 2$ , only GLM $H = 6$ remains above uniform (1.52), while its $H = 4$ difference is unresolved. For GLM’s primary $H = 4 , 6$ cells, bias is at most 3% of MSE but realized population-scale variance is 1.36 and 2.66 times uniform: rare pilots starve influential groups, making $\sigma _ { g } ^ { 2 } / w _ { g }$ large (Section E.10), outside Proposition 3’s local regime.

A retrospective floor sweep supports the underallocation diagnosis: at $H = 6 , \alpha = . 1 $ , 400 queries, increasing the floor from .072 to .40 lowers Qwen and GLM TIS/uniform MSE from 1.38/1.77 to .47/.55, with still larger floors reducing them further. Because the sweep followed the failures, it is diagnostic rather than a tuning result; held-out FinQA below tests the subsequently frozen anchor. Workflow selection on this MMLU panel is nearly saturated, so the informative outcome is estimation error rather than final pairwise choice.

Table 10: Absolute TIS panel MSE (Monte Carlo SE), α = .1, 400 queries/kernel; 300 replications.
<table><tr><td>Generator</td><td> $H = 2$ </td><td> $H = 4$ </td><td> $H = 6$ </td></tr><tr><td> $\mathrm { Q w e n } 3 – 4 \mathrm { B }$ </td><td> $2 . 2 { \mathrm { e } } { - } 0 4 \left( 1 { \mathrm { e } } { - } 0 5 \right)$ </td><td> $9 . 7 \mathrm { e } { - } 0 4 \ : ( 6 \mathrm { e } { - } 0 5 )$ </td><td> $2 . 7 \mathrm { e } { - } 0 3 \ : ( 2 \mathrm { e } { - } 0 4 )$ </td></tr><tr><td> $\mathrm { P h i - } 4 { \cdot } \mathrm { m i n i }$ </td><td> $2 . 8 \mathsf { e } . 0 4 ( 1 \mathsf { e } . 0 5 )$ </td><td> $1 . 7 \mathrm { e } { - } 0 3 \ : ( 1 \mathrm { e } { - } 0 4 )$ </td><td> $5 . 1 \mathrm { e } { - } 0 3 \ ( 5 \mathrm { e } { - } 0 4 )$ </td></tr><tr><td> $\mathrm { G r a n i t e } { \cdot } 4 . 2 { \cdot } 8 \mathrm { B }$ </td><td> $9 . 8 \mathrm { e } { - } 0 5 \ : ( 5 \mathrm { e } { - } 0 6 )$ </td><td> $1 . 3 \mathrm { e } { - } 0 4 ( 1 \mathrm { e } { - } 0 5 )$ </td><td> $2 . 4 \mathrm { e } { - } 0 4 ( 1 \mathrm { e } { - } 0 5 )$ </td></tr><tr><td> $\mathbf { M i s t r a l } { - 2 4 \mathbf { B } ^ { \dagger } }$ </td><td> $1 . 2 { \mathrm { e } } { - } 0 4 \ ( 2 { \mathrm { e } } { - } 0 5 )$ </td><td> $3 . 5 { \mathrm { e } } { - } 0 4 \ ( 3 { \mathrm { e } } { - } 0 5 )$ </td><td> $9 . 6 \mathrm { e } { - } 0 4 ( 1 \mathrm { e } { - } 0 4 )$ </td></tr><tr><td> $\mathrm { Q w e n } 3 - 3 2 \mathbf { B } ^ { \dagger }$ </td><td> $5 . 7 \mathrm { e } { - } 0 4 \ : ( 4 \mathrm { e } { - } 0 5 )$ </td><td> $2 . 9 \mathrm { e } { - } 0 3 \ ( 3 \mathrm { e } { - } 0 4 )$ </td><td> $7 . 5 \mathrm { e } { - } 0 3 \ : ( 8 \mathrm { e } { - } 0 4 )$ </td></tr><tr><td> $\mathbf { G L M - } 4 { - } 3 2 \mathbf { B } ^ { \dagger }$ </td><td> $4 . 1 \mathrm { e } { - } 0 4 ( 3 \mathrm { e } { - } 0 5 )$ </td><td> $4 . 2 { \mathrm { e } } { - } 0 3 \ ( 2 { \mathrm { e } } { - } 0 4 )$ </td><td> $1 . 9 \mathrm { e } { - } 0 2 ( 9 \mathrm { e } { - } 0 4 )$ </td></tr></table>

Table 11: Budget sensitivity at $H = 6 , \alpha = . 1 \colon$ shared/untied TIS MSE relative to uniform at equal total budget.
<table><tr><td></td><td colspan="4">shared TIS</td><td colspan="4">untied TIS</td></tr><tr><td>Generator</td><td>50</td><td>100</td><td>200</td><td>400</td><td>50</td><td>100</td><td>200</td><td>400</td></tr><tr><td>Qwen3-4B</td><td>.819</td><td>.851</td><td>1.22</td><td>1.38</td><td>8.59</td><td>4.53</td><td>5.44</td><td>7.67</td></tr><tr><td>Phi-4-mini</td><td>.448</td><td>.499</td><td>.469</td><td>.487</td><td>4.29</td><td>2.02</td><td>2.55</td><td>3.00</td></tr><tr><td>Granite-4.2-8B</td><td>.469</td><td>.386</td><td>.248</td><td>.154</td><td>5.49</td><td>1.64</td><td>2.06</td><td>1.56</td></tr><tr><td>Mistral-24B†</td><td>.497</td><td>.462</td><td>.391</td><td>.214</td><td>5.81</td><td>1.58</td><td>2.08</td><td>1.90</td></tr><tr><td> $\mathrm { Q w e n } 3 - 3 2 \mathbf { B } ^ { \dagger }$ </td><td>.681</td><td>.606</td><td>.497</td><td>.411</td><td>5.45</td><td>2.71</td><td>2.91</td><td>3.71</td></tr><tr><td> $\mathbf { G L M - } 4 { - } 3 2 \mathbf { B } ^ { \dagger }$ </td><td>1.02</td><td>1.44</td><td>1.76</td><td>1.77</td><td>6.39</td><td>2.65</td><td>4.81</td><td>6.95</td></tr></table>

Three post-hoc variations point to the same pilot-reliability mechanism. Under an asymmetric confidence-weighted utility, Qwen plain-TIS/uniform ratios at $H \ = \ 2 , 4 , 6$ are $. 3 9 , 1 . 0 2 , 1 . 0 6 ,$ versus anchor .031, .053, .056; on a disjoint high-stakes panel they are .96, .42, .72 versus anchor .039, .052, .052. A cautious policy again gives plain TIS 1.04 of uniform at $H = 6$ while occupancy is .13. These checks support the diagnosis only; none was used to choose the frozen anchor.

## E.2.7 COST-TO-ACCURACY ANALYSIS AND COMPUTATION ACCOUNTING

A lower MSE at one budget need not imply fewer queries at every target accuracy, so we convert error curves to query cost and report computation separately. These comparisons are retrospective (no accuracy tolerance was declared before the runs), and charged query counts include pilots and rollout transitions.

Multi-target relative cost. At 21 RMSE targets in the common attained range, we use log–log firstcrossing interpolation without extrapolation; out-of-grid crossings retain budget bounds. Pointwise 5th/95th-percentile envelopes come from empirical bootstrap when replication rows are available (CliffWalking TIS/learned mean, inventory TIS/uniform, all FinQA pairs) and Gaussian sensitivity otherwise; they are not simultaneous confidence bands.

CliffWalking. TIS requires .65–.84 of learned-occupancy queries, .32–.45 of rollout queries, and .84–1.05 of learned-mean queries. Against learned mean, empirical envelopes are below one at the 12 strictest targets and above one at none; uniform’s best tested RMSE exceeds TIS’s worst, censoring that comparison in TIS’s favor. Inventory. Ratios are .55–.72 versus learned occupancy, .58–.79 versus learned mean, .26–.30 versus rollouts, and .50–.83 versus uniform, with envelopes below one at all 21 targets; only the uniform comparison uses the empirical bootstrap. FinQA. On the 25–800-query grid, the anchor uses .10–.17 of uniform’s charged queries on Qwen and .23–.52 on Phi, with empirical envelopes favoring it at all 21 targets. For Phi ordinary review, envelopes favor the anchor at 14 targets versus occupancy and the 10 strictest versus rollouts, while the baselines win at 4 and 8 targets; for unit-check review these counts are 12 and 6 for the anchor, and 0 and 12 for the baselines. Other contrasts are unresolved, and on Qwen both strong baselines require fewer queries at every target.

A single target can be misleading: uniform’s largest-budget RMSE cannot rank sample-only CliffWalking designs because all reach it at the smallest tested budget; inventory shows savings across the studied range, but its half-budget crossing remains unresolved. Computation is separate. Table 13 times complete sampling-and-estimation pipelines on an Intel i9-13900H with one BLAS/OMP thread (ten replications after two warm-ups). Uniform has no pilot; occupancy needs only the forward-visitation solve; TIS, the anchor, learned mean, and the occupancy–mean blend include the influence solve, with the latter three matching TIS within 4%, so the tail-specific score adds no computation over the strongest mean-based baseline. FinQA timing covers all 50 Phi ordinary-review questions (11–197 grid atoms). These frozen-law fixed-budget times are not live-call costs or time-to-equal-accuracy.

Table 12: Risk sensitivity at $\alpha = . 2 ,$ 400 queries/kernel: TIS MSE/uniform MSE with uniformminus-TIS z (positive favors TIS); GLM H = 4 is unresolved.
<table><tr><td>Generator</td><td> $H = 2$ </td><td> $H = 4$ </td><td> $H = 6$ </td></tr><tr><td>Qwen3-4B</td><td>.195 (18.8)</td><td>.290 (14.4)</td><td> $. 6 1 3 ( 6 . 2 )$ </td></tr><tr><td>Phi-4-mini</td><td>.115 (37.4)</td><td>.219 (25.0)</td><td>.521 (8.0)</td></tr><tr><td> $\mathrm { G r a n i t e } { \cdot } 4 . 2 { \cdot } 8 \mathrm { B }$ </td><td>.120 (43.8)</td><td>.084 (42.6)</td><td>.088 (37.1)</td></tr><tr><td> $\mathbf { M i s t r a l - } 2 4 \mathbf { B } ^ { \dagger }$ </td><td>.068 (49.5)</td><td>.175 (31.5)</td><td>.289 (16.9)</td></tr><tr><td> $\mathrm { Q w e n } 3 - 3 2 \mathbf { B } ^ { \dagger }$ </td><td>.170 (33.5)</td><td>.232 (27.5)</td><td>.368 (16.2)</td></tr><tr><td> $\mathrm { G L M }  – 4 – 3 2 \mathbf { B } ^ { \dagger }$ </td><td>.345 (20.6)</td><td>.929 (1.1)</td><td>1.52 (-6.2)</td></tr></table>

Table 13: CPU milliseconds per replication for the full sampling-and-estimation pipeline at fixed query budgets; FinQA is the 50-question Phi ordinary-review panel. These are frozen-law replay times, not live-call costs.
<table><tr><td>Task</td><td>Budget</td><td>Uniform</td><td>Occ.</td><td>Mean</td><td>TIS</td><td>Anch.</td><td>Mean blend</td><td> Rollout</td></tr><tr><td>CliffWalking</td><td>400</td><td>110</td><td>122</td><td>394</td><td>394</td><td>396</td><td>390</td><td>32</td></tr><tr><td>Inventory</td><td>1,200</td><td>5.0</td><td>6.5</td><td>10.7</td><td>10.9</td><td>10.8</td><td>11.2</td><td>5.2</td></tr><tr><td>FinQA panel  $( 5 0 \mathrm { q . ) }$ </td><td>200</td><td>658</td><td>823</td><td>2,031</td><td>2,015</td><td>2,039</td><td>2,040</td><td>298</td></tr></table>

## E.3 FINQA TERMINAL-RISK PROTOCOL AND FULL RESULTS

![](images/d526d51e53f1cd4d333794796336293e1205f1fd0de8b550174772292ecf74ec.jpg)

![](images/b1d91464bd63969d45dc82df79f5d3d104c24e63fd858b221d98bb1754ea855a.jpg)

![](images/7f8e79cf258de3e409631e6801dd39c89eedc39c72da3293714f14de7c75a6fb.jpg)  
Figure 4: FinQA: the anchor beats occupancy and rollouts on Phi at 400/800 queries; Qwen favors the baselines. (a,b) Mean panel-MSE/uniform-MSE ratio across workflows; (c) Phi wrongselection rate. Table 14 gives per-workflow results.

This held-out study supports Q4: does the frozen anchor improve estimation of numerical failure severity and workflow selection (Figure 4)?

Source and screening. FinQA development items (Chen et al., 2021) require finite nonzero executable gold answers and fit the context budget. Each eight-candidate bank is generated before gold-based scoring and admitted only if severity-utility spread is at least .25; this screen was fixed after 16 of the first 20 banks had constant utility, so the study targets material severity disagreement. Development admitted 20/170 items; after protocol freeze, a disjoint held-out scan admitted 50/312.

Workflow and kernels. Candidate banks are generated at temperature .8 and frozen, with parsing failures kept as invalid candidates. The root action is select; reviews use the MMLU confidence bands (Table 8). Prompts contain the financial context, question, full bank, current candidate– confidence pair, and review instruction, but no history or stage index. Two constrained output tokens determine the next state. The root plus 72 pairs gives 73 query groups per question, generator, and workflow, shared across stages. Ordinary and unit-and-sign workflows share the root prompt and policy but use different review templates, hence different transition laws. Runs have one selection and two reviews $\left( H = 3 \right)$ , with $\alpha = . 1$

Severity utility. For gold $y ^ { * }$ and candidate $y ,$ let $e _ { 0 } = \operatorname* { m i n } ( | y - y ^ { * } | , | y / 1 0 0 - y ^ { * } | , | 1 0 0 y - y ^ { * } | )$ $t = . 0 0 5 | y ^ { * } |$ , and $e = \operatorname* { m a x } ( 0 , e _ { 0 } - t )$ . Terminal utility is $u = 1 - e / ( e + | y ^ { * } | ) \in [ 0 , 1 ]$ , with $u = 0$ for unparseable candidates. Binary correctness would collapse magnitude: for $\dot { R } \in \{ 0 , 1 \}$ with failure probability $p _ { \mathrm { f a i l } }$ , lower CVaR is max $\{ \alpha - p _ { \mathrm { f a i l } } , 0 \} / \alpha$ . A post-hoc unit-aware rerun changes screening for 3 development and 5 held-out questions; at 200 queries anchor/uniform MSE is .096/.100 on Qwen and .226/.186 on Phi for ordinary/unit-check review. The anchor remains resolved against uniform, trails occupancy/rollouts on Qwen, and beats occupancy on Phi; Phi’s decision gap widens from $6 . 7 \times 1 0 ^ { - 4 } \mathrm { t o } \ 5 . 6 \times 1 0 ^ { - 3 }$ . The frozen metric defines the primary results.

Reward and exact target. Let $u ( s )$ be candidate severity utility, independent of confidence, and $u ( \emptyset ) = 0$ . For sampled next state ${ \dot { S } } ^ { \prime }$

$$
R ( s , S ^ { \prime } ) = { \frac { u ( S ^ { \prime } ) - u ( s ) + 1 } { 2 } } \in [ 0 , 1 ] , \qquad G _ { H } = \sum _ { t = 0 } ^ { H - 1 } R ( S _ { t } , S _ { t + 1 } ) = { \frac { H + u ( S _ { H } ) } { 2 } } .\tag{67}
$$

Intermediate utilities telescope, so a temporary improvement that is later reversed does not improve the return. The h-step return-to-go support is $\{ ( u _ { j } - u _ { i } { + } h ) / 2 \}$ ; the grid contains these values and the endpoints, at most $\bar { H } ( C + 1 ) C \bar { + } 2$ atoms $( C \bar { = } \bar { 8 } ;$ at most 197 observed). The categorical recursion is exact and matches exact-law CVaR to machine precision, with $C _ { \mathrm { r e t } } = [ H + \mathrm { C V } \bar { \mathrm { a R } } _ { \alpha } ( u ( S _ { H } ) ) ] / 2$ All reported CVaRs, gaps, regrets, and MSEs use this scale; converting back by $2 \widehat { C } _ { \mathrm { r e t } } - H$ multiplies MSE by four and doubles gaps without changing within-setting ratios or rankings.

Freeze and audits. Development used 20 screened questions and 100 replications to check parsing (142/160 candidates valid), the .25 spread screen, positive margins, oracle headroom (Qwen oracle+floor .006–.013 of uniform MSE at 100–200 queries), and replication noise; the latter informed held-out size without guaranteeing power. Before calibration, the protocol fixed 50 questions, 300 replications, generators, workflows, anchor, floor, and budgets 25–800 queries per kernel. All four held-out calibrations (3,650 kernels each) pass provenance and generation audits. All 50 margins are positive, but total influence is zero for 3 Qwen questions per workflow and 13/14 Phi questions (ordinary/unit-check), so every allocation has zero leading variance there although higher-order error can remain.

Table 14 gives the complete frozen-metric results through 800 queries; the post-hoc unit-aware sensitivity is summarized above. At 100–200 queries all eight anchor/uniform contrasts are resolved $( z = 8 . 1 \mathrm { \bar { - } 3 0 . 3 } )$ , while plain TIS is unresolved in seven. At 400/800 queries the anchor beats both strong baselines in both Phi workflows: MSE is .81–.86 of occupancy $\left( z = - 8 . 4 \mathrm { t o } - 4 . 8 \right)$ and .68– .88 of rollouts $( z = - 8 . 4 \mathrm { t o } - 3 . 1 )$ , using z for anchor minus baseline. On Qwen, occupancy and rollouts remain better. All contrasts use shared conditional-query streams, a separate rollout stream, and 300 replications.

Workflow decision and coupling sensitivity. Table 15 reports 25–200-query selection rates. Phi’s ordinary/unit-check return-CVaR gap is only $6 . 7 \times 1 0 ^ { - 4 } ;$ at 100/200 queries, discordant-pair tests resolve the anchor over uniform $( z \ = \ 4 . 2 , 3 . 5 )$ but not over occupancy or rollouts $( | z | \leq 1 . 7 )$ and the budget-200 regret reduction is $7 . 6 \times 1 0 ^ { - 5 }$ . Workflow templates share common random numbers, which reduce comparison noise without changing either workflow’s marginal MSE. A retrospective re-pairing check shows that Qwen’s exceptionally low paired selection errors partly reflect this cancellation; because re-pairing is not fresh simulation, the frozen-protocol rates remain primary.

A separately committed skeptical-template follow-up on Phi had a much larger workflow gap and made the strong methods essentially error-free at small budgets, so it did not distinguish the anchor from occupancy or rollouts. This reinforces why the near-tie frozen study, rather than the easier follow-up, is the informative workflow-selection test.

Table 14: FinQA frozen-metric estimation: uniform panel MSE, anchor-to-baseline MSE ratios, and paired uniform-minus-anchor z over 300 replications.
<table><tr><td> $\mathbf { G e n . }$ </td><td>Workflow</td><td>Budget</td><td>Unif. MSE</td><td>Anch./unif.</td><td>Anch./occ.</td><td>Anch./roll.</td><td>z</td></tr><tr><td>qwen</td><td>ordinary</td><td>25</td><td>6.2 e-5</td><td>0.152</td><td>1.84</td><td>3.52</td><td>+5.3</td></tr><tr><td>qwen</td><td>ordinary</td><td>50</td><td>2.8 e-5</td><td>0.127</td><td>2.10</td><td>2.81</td><td>+6.5</td></tr><tr><td>qwen</td><td>ordinary</td><td>100</td><td>1.6 e-5</td><td>0.089</td><td>1.47</td><td>2.07</td><td>+8.1</td></tr><tr><td>qwen</td><td>ordinary</td><td>200</td><td>6.7e-6</td><td>0.097</td><td>1.77</td><td>2.30</td><td>+9.9</td></tr><tr><td>qwen</td><td>ordinary</td><td>400</td><td>3.3 e-6</td><td>0.075</td><td>1.41</td><td>1.62</td><td>+10.0</td></tr><tr><td>qwen</td><td>ordinary</td><td>800</td><td>2.0 e-6</td><td>0.077</td><td>1.66</td><td>2.11</td><td>+10.2</td></tr><tr><td>qwen</td><td>unit check</td><td>25</td><td>1.1 e-4</td><td>0.165</td><td>1.97</td><td>3.59</td><td>+7.7</td></tr><tr><td>qwen</td><td>unit check</td><td>50</td><td>5.7e-5</td><td>0.117</td><td>1.90</td><td>3.10</td><td>+10.2</td></tr><tr><td>qwen</td><td>unit check</td><td>100</td><td>2.8 e-5</td><td>0.094</td><td>1.60</td><td>2.28</td><td>+11.5</td></tr><tr><td>qwen</td><td>unit check</td><td>200</td><td>1.2 e-5</td><td>0.100</td><td>1.61</td><td>2.26</td><td>+14.3</td></tr><tr><td>qwen</td><td>unit check</td><td>400</td><td>6.9 e-6</td><td>0.073</td><td>1.50</td><td>1.74</td><td>+15.6</td></tr><tr><td>qwen</td><td>unit check</td><td>800</td><td>3.6 e-6</td><td>0.073</td><td>1.56</td><td>1.89</td><td>+15.5</td></tr><tr><td>phi</td><td>ordinary</td><td>25</td><td>3.9 e-4</td><td>0.540</td><td>1.10</td><td>1.96</td><td>+16.7</td></tr><tr><td>phi</td><td>ordinary</td><td>50</td><td>2.0 e-4</td><td>0.431</td><td>1.03</td><td>1.49</td><td>+21.4</td></tr><tr><td>phi</td><td>ordinary</td><td>100</td><td>1.2 e-4</td><td>0.282</td><td>0.93</td><td>1.10</td><td>+27.2</td></tr><tr><td>phi</td><td>ordinary</td><td>200</td><td>5.9 e-5</td><td>0.216</td><td>0.88</td><td>0.84</td><td>+30.3</td></tr><tr><td>phi</td><td>ordinary</td><td>400</td><td>3.1 e−5</td><td>0.198</td><td>0.86</td><td>0.77</td><td>+31.5</td></tr><tr><td>phi</td><td>ordinary</td><td>800</td><td>1.5 e-5</td><td>0.184</td><td>0.82</td><td>0.68</td><td>+27.1</td></tr><tr><td>phi</td><td>unit check</td><td>25</td><td>4.5e-4</td><td>0.546</td><td>1.02</td><td>2.23</td><td>+15.7</td></tr><tr><td>phi</td><td>unit check</td><td>50</td><td>2.5 e-4</td><td>0.390</td><td>0.97</td><td>1.65</td><td>+21.0</td></tr><tr><td>phi</td><td>unit check</td><td>100</td><td>1.3 e-4</td><td>0.327</td><td>0.97</td><td>1.50</td><td>+23.3</td></tr><tr><td>phi</td><td>unit check</td><td>200</td><td> $7 . 3 e { - 5 }$ </td><td>0.229</td><td>0.88</td><td>1.07</td><td>+26.9</td></tr><tr><td>phi</td><td>unit check</td><td>400</td><td> $3 . 5 e { - 5 }$ </td><td>0.197</td><td>0.81</td><td>0.88</td><td>+28.9</td></tr><tr><td>phi</td><td>unit check</td><td>800</td><td> $1 . 9 e { - 5 }$ </td><td>0.162</td><td>0.83</td><td>0.84</td><td>+26.3</td></tr></table>

Table 15: FinQA frozen-metric workflow selection at 25–200 queries/kernel: exact panel CVaRs, their gap, and wrong-selection percentages under shared workflow streams. Zero means no errors in 300 replications.
<table><tr><td colspan="10"></td><td rowspan="2">Oracle Anch.</td><td rowspan="2">+floor</td></tr><tr><td>Gen.</td><td>Budget</td><td> $C _ { \mathrm { o r d } }$ </td><td> $C _ { \mathrm { u n i t } }$ </td><td> $\mathrm { G a p }$ </td><td>Unif.</td><td>TIS</td><td> $_ \mathrm { O c c . }$ </td><td>Roll.</td></tr><tr><td>qwen</td><td>25</td><td>1.87304</td><td>1.87262</td><td>+0.00041</td><td>0.0</td><td>2.0</td><td>0.0</td><td>0.0</td><td>0.7</td><td>0.0</td></tr><tr><td>qwen</td><td>50</td><td>1.87304</td><td>1.87262</td><td>+0.00041</td><td>0.0</td><td>0.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>qwen</td><td>100</td><td>1.87304</td><td>1.87262</td><td>+0.00041</td><td>0.0</td><td>0.3</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>qwen</td><td>200</td><td>1.87304</td><td>1.87262</td><td>+0.00041</td><td>0.0</td><td>1.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>phi</td><td>25</td><td>1.76764</td><td>1.76697</td><td>+0.00067</td><td>45.3</td><td>50.0</td><td>36.3</td><td>38.3</td><td>42.3</td><td>36.3</td></tr><tr><td>phi</td><td>50</td><td>1.76764</td><td>1.76697</td><td>+0.00067</td><td>45.3</td><td>47.3</td><td>34.0</td><td>27.0</td><td>33.3</td><td>24.3</td></tr><tr><td>phi</td><td>100</td><td>1.76764</td><td>1.76697</td><td>+0.00067</td><td>38.7</td><td>39.7</td><td>28.3</td><td>21.7</td><td>24.7</td><td>16.3</td></tr><tr><td>phi</td><td>200</td><td>1.76764</td><td>1.76697</td><td>+0.00067</td><td>28.3</td><td>33.0</td><td>14.3</td><td>16.0</td><td>17.0</td><td>7.3</td></tr></table>

## E.4 INVENTORY DISRUPTION FAMILY

This prespecified breadth family supports Q2 by varying three axes of the seasonal inventory bench mark: base disruption probability {.01, .04, .12} (plus .02 at capacity), disruption loss of 1, 2, or 3 remaining units, and two fixed policies (the original seasonal base-stock targets or those targets plus one unit, capped at capacity). Rewards, {0, .5, 1} quantization, demand laws, $H = 8 .$ , and the grid are unchanged, producing 18 cases with 41 or 48 retained blocks. Before simulation, exact enumeration confirmed distinct kernel laws in all 18 cases, positive categorical margins (including two near .001), CVaR .675–1.260, and $V _ { \mathrm { u n i f } } / V ^ { * }$ from 1.49 to 2.28. Budgets are 150, 600, and 1,200 queries per block with 300 replications; methods, pilot, floor, and rounding match Section E.2.3. Absolute RMSE tolerances .02, .01, .005 were declared with the family.

Table 16: Inventory family: resolved lower-MSE cases (of 18) at 1,200 queries/block, using paired $| z | \geq 2 ;$ ; none resolve in the comparator’s favor.
<table><tr><td>Method</td><td>occupancy</td><td>learned mean</td><td>occ.+uniform</td><td>occ.+mean</td><td>rollouts</td></tr><tr><td>TIS</td><td>18</td><td>17</td><td>18</td><td>18</td><td>18</td></tr><tr><td>anchored TIS</td><td>18</td><td>17</td><td>18</td><td>17</td><td>18</td></tr></table>

Across the 18 cases, median MSE/uniform at 150/600/1,200 queries per block is . $. 7 4 / . 6 2 / . 6 1$ for TIS, .82/.69/.69 for the anchor, 1.05/.92/.93 for learned occupancy, .98/.86/.84 for learned mean, 1.11/.99/.95 for the occupancy–uniform blend, $1 . 0 3 / . 8 7 / . 8 6$ for the occupancy–mean blend, 2.72/2.45/2.56 for rollouts, and .56/.51/.53 for oracle+floor. At RMSE .02, TIS first reaches the target by 150 queries in 2 cases and by 600 in 15; uniform, occupancy, and the uniform blend require 1,200 in 6. At .01, TIS reaches 11 cases, uniform/occupancy 8, and rollouts none; at .005, only TIS reaches the target (2 cases). Plain TIS leads the anchor throughout, consistent with anchoring acting as insurance rather than a gain when pilots are reliable.

## E.5 BLENDING CONTROLS

To isolate Q5, that is whether the tail score itself drives the anchor, we compare two equally regularized controls. With floored component shares $w = ( 1 - \lambda ) \hat { p } + \lambda u .$ , they use $\scriptstyle { \frac { 1 } { 2 } } w _ { \mathrm { o c c } } + { \frac { 1 } { 2 } } u$ and $\begin{array} { r } { \frac { 1 } { 2 } w _ { \mathrm { o c c } } + \frac { 1 } { 2 } w _ { \mathrm { m e a n } } , } \end{array}$ with no second floor; hence all three blends differ only in the component mixed with occupancy. The uniform blend equals occupancy with floor $( 1 + \lambda ) / 2$ and coincides with the anchor when the tail pilot falls back to uniform. All blends pay the same pilot and share coupled query streams. On held-out FinQA (2 generators × 2 workflows × budgets 100–800; 300 replications), the anchor is resolved better than the uniform blend in all 16 cells (MSE ratios .62–.97, decreasing with budget), while the mean blend has lower point MSE in all 16 and is resolved better in 15 (anchor/mean ratios 1.17–1.54 on Qwen, 1.03–1.10 on Phi). Learned mean alone can reach 1.4× uniform, but occupancy–mean blending is competitive. Under MMLU-Pro confident-error utility (6 generators, $H = 6 , \alpha = . 1$ , 100–800 queries, 300 replications), the anchor beats the uniform blend and learned occupancy in all 24 cells and the mean blend in 23 (ratios .76–.97; Phi-4-mini at 100 is unresolved). Rollouts are better for five generators at 100 queries, but the anchor is better for all six at 800 (ratios .55–.89); plain TIS has 1.4–26× the mean blend’s MSE. Under Brier utility (100–400 queries), the anchor beats the uniform blend and occupancy in all 18 cells, the mean blend in 11, is worse in none, and is unresolved in 7 (all Phi-4-mini and Qwen3-32B budgets, plus Qwen3-4B at 100). On the high-stakes panel (Qwen3-4B, Phi-4-mini, Qwen3-32B), it beats the uniform blend and occupancy in all 9 cells and the mean blend in $5 ;$ the mean blend wins 2 (Phi-4-mini at 200/400). Brier comparisons with rollouts again cross over with budget: rollouts lead at small budgets, while the anchor leads four of six generators at 400.

## E.6 RARE FAILURES AND THE TAIL–MEAN COINCIDENCE

ProofofProposition 2. Let $F ^ { - 1 }$ be the quantile function of $X = G _ { H } ( s _ { 0 } )$ . By assumption, $X \leq x ^ { \star }$ almost surely and $\operatorname* { P r } ( X < x ^ { \star } ) \ = \pi < \alpha$ . Hence $F ^ { - 1 } ( u ) \ : = \ : x ^ { \star }$ for every $u \in \mathsf { \Gamma } ( \pi , 1 )$ , and boundedness gives

$$
\begin{array} { r l } & { \alpha \operatorname { C V a R } _ { \alpha } ( X ) = \displaystyle \int _ { 0 } ^ { \alpha } F ^ { - 1 } ( u ) d u } \\ & { \qquad = \displaystyle \int _ { 0 } ^ { 1 } F ^ { - 1 } ( u ) d u - \displaystyle \int _ { \alpha } ^ { 1 } F ^ { - 1 } ( u ) d u } \\ & { \qquad = \mathbb { E } [ X ] - ( 1 - \alpha ) x ^ { \star } . } \end{array}
$$

This proves the stated identity.

By exactness, the grid contains every attainable partial return generated by the fixed declared outcome spaces. Hence changing only the kernel laws changes probabilities but not the structural maximum $x ^ { \star }$ , and the categorical root law continues to equal the true return law. In a finite horizon, the root return law depends continuously in total variation on the finite collection of group laws; therefore $\operatorname* { P r } ( X < x ^ { \star } )$ remains below α throughout a sufficiently small neighborhood because the baseline gap $\alpha - \pi$ is strictly positive. On this neighborhood,

$$
C _ { \alpha , K } = { \mathrm { C V a R } } _ { \alpha } ( X ) = { \frac { 1 } { \alpha } } \mathbb { E } [ X ] - { \frac { 1 - \alpha } { \alpha } } x ^ { \star } .
$$

Let $J ( P ) : = \mathbb { E } _ { P } [ X ]$ . Write $v _ { h } ( s ) = \mathbb { E } [ G _ { h } ( s ) ]$ and let $\mu _ { h } ( s )$ be the probability, under the fixed policy and population laws, of visiting state s with h steps remaining. Differentiating the ordinary mean Bellman recursion shows that, for a shared stationary group $g = \left( s , a \right)$ , a centered groupwise influence function for J is

$$
\psi _ { s , a } ( W ) = \sum _ { h = 1 } ^ { H } \mu _ { h } ( s ) \pi _ { h } ( a \mid s ) \Big \{ R + v _ { h - 1 } ( S ^ { \prime } ) - \mathbb { E } _ { P _ { s , a } } [ R + v _ { h - 1 } ( S ^ { \prime } ) ] \Big \} ,
$$

with the analogous single-row expression in the untied model. This is exactly the ordinary meanreturn influence used by the learned-mean design in Appendix E.2. Consequently, for every DQM direction $\begin{array} { r } { s = ( s _ { g } ) , \dot { J } _ { s } = \sum _ { g } \mathbb { E } _ { P _ { g } } [ \psi _ { g } s _ { g } ] } \end{array}$ . Differentiating the affine identity above and using Equation 38 gives

$$
\sum _ { g } \mathbb { E } _ { P _ { g } } [ \phi _ { g } s _ { g } ] = \dot { C } _ { s } = \frac { 1 } { \alpha } \dot { J } _ { s } = \frac { 1 } { \alpha } \sum _ { g } \mathbb { E } _ { P _ { g } } [ \psi _ { g } s _ { g } ] .
$$

Both $\phi _ { g }$ and $\psi _ { g }$ are centered. Because the product tangent space contains an arbitrary $L _ { 0 } ^ { 2 } ( P _ { g } )$ direction in each group, equality for every score direction implies $\phi _ { g } = \psi _ { g } / \alpha$ in $L _ { 0 } ^ { 2 } ( P _ { g } )$ for each group. Consequently the tail and mean influence standard deviations satisfy $\sigma _ { g } = \mathrm { { s d } } \bar { ( \psi _ { g } ) } / \alpha$ . If these scales are not all zero, normalizing them gives identical Neyman shares; if they are all zero, both objectives have zero first-order variance for every allocation. This proves the allocation claim.

This diagnostic supports Q5 by testing when tail-specific allocation should differ from mean allocation. All experiments use closed grids exact on their declared supports. Normalize each nonzero tail- or mean-influence scale vector to sum one, representing an all-zero vector by uniform; then ${ \begin{array} { r } { { \frac { 1 } { 2 } } \sum _ { q } | p _ { g } - m _ { g } | = 0 } \end{array} }$ whenever Proposition 2 applies. The median distance is .000 on held-out FinQA for both generators and workflows, with exact zero on 45–51% of questions. On 25-question MMLU-Pro samples at $H = 6 , \alpha = . 1$ , Brier-utility medians are .24 (Qwen3-4B), .09 (Phi-4-mini), .29 (Granite-4.2-8B), .30 (Mistral-24B), .16 (Qwen3-32B), and .27 (GLM-4-32B); the high-stakes panel gives .33, .08, .18 for Qwen3-4B, Phi-4-mini, Qwen3-32B, and confident-error utility gives .32, .18, .32 for Qwen3-4B, Phi-4-mini, GLM-4-32B. Granite, Mistral, Qwen3-32B, and the highstakes Phi/Qwen3-32B distances were computed after the blending runs.

Prospective divergence test. We therefore fixed a rule before simulating five new settings: median distance ≥ .24 predicts that the anchor has resolved lower MSE than the mean blend in at least two of three budgets (100/200/400 queries) and higher MSE in none; distance $\leq . 1 8$ predicts at most one resolved win; intermediate values make no prediction. The five settings were high-stakes confident-error Qwen3-4B (.318, win), Phi-4-mini (.169, no advantage), Qwen3-32B (.216, none), and cautious-policy Qwen3-4B with Brier (.195, none) or confident-error (.269, win). All use closed grids, $H = 6 , \alpha = . 1$ , and 300 replications. Against the mean blend, MSE ratios (paired z) at 100/200/400 are $. 8 8 / . 8 4 / . 8 3 \ : ( - 6 . 2 / \mathrm { ~ - ~ } 8 . 5 / \mathrm { ~ - ~ } 9 . 7 )$ for high-stakes Qwen3-4B, 1.03/1.02/.99 $( + 2 . 6 / + 1 . 4 / - 0 . 6 )$ for high-stakes Phi, and . $9 \dot { 6 } / . 9 5 / . 9 2 ( - 3 . \bar { 0 } / - 4 . 0 / - 5 . 3 )$ for cautious confidenterror Qwen, so all three predictions hold. The two unpredicted settings give . $9 7 / . 9 4 / . 9 1 ( - 1 . 3 / -$ $4 . 3 / - 6 . 0 )$ for high-stakes Qwen3-32B and . $9 6 / . 9 2 / . 8 8 ( - 3 . 3 / - 5 . 5 / - 8 . 7 )$ for cautious Brier Qwen. In all 15 cells the anchor also beats learned occupancy and the uniform blend; at 400 queries it beats complete rollouts in four settings (ratios .74–.85) and ties the fifth (.99).

The confident-error utility scores a correct response with confidence c as $( 1 + c ) / 2$ and a wrong response as $( 1 - c ) ^ { 2 } / 2$ , making confident errors nearly worthless; it preserves the Brier per-response ordering but has a much heavier lower tail.

## E.7 FINQA WITH CALCULATOR FAULTS

This robustness check extends Q5 to exogenous tool errors while preserving the FinQA terminalseverity objective. Each review prompt includes an automated calculator report for the current candidate; independently after each model call, the report is correct with probability $1 - p$ and multiplied by 100 with probability $p .$ The root, 72 correct-report states, and 72 faulted-report states define 145 queryable prompt laws per question; an outcome is the model response together with the fault indicator, so one query still costs one model call. The laws were calibrated exactly for both generators on the 50 held-out questions, and the same design was declared for ordinary and unit-check review.

The perturbation is informative because influence and visitation react differently. For Qwen, faulted states carry median tail/mean/occupancy shares $5 . 8 / 6 . 2 / 0 . 7 \%$ at $p = . 0 1$ and $3 8 / 4 2 / 6 . 7 \%$ at $p =$ .10; for Phi the corresponding shares are $1 . 0 / 0 . 9 / 0 . 7 \%$ and $9 . 8 / 9 . 0 / 6 . 7 \%$ ; 12/50 Phi questions have no first-order tail signal. Thus tail and mean influence remain close even when occupancy can be very different. Across 100–400 queries per kernel, anchor/uniform MSE is .034–.050 for Qwen and .081–.144 for Phi in ordinary review, with similar .041–.050 and .069–.144 ranges under unit check. Yet rollouts are usually strongest, the mean blend beats the anchor throughout Qwen, and the anchor trails occupancy on Qwen while beating it on Phi. Plain TIS can have 7–29× the mean blend’s MSE. Qwen’s oracle remains only .007–.010 of uniform MSE, locating the gap in pilot learning rather than the population influence signal; several Phi questions instead have near-zero margins. Median tail– mean distance is .000 for Qwen and .03–.06 for Phi, well below the no-advantage regime identified in Appendix E.6. The tool-fault study therefore supports the diagnostic’s negative prediction: a tail score can be highly informative in population yet unnecessary relative to a smoother mean score when their normalized allocations nearly coincide.

## E.8 LONGER REVIEW LOOPS

This extension supports the main-text longer-loop claim and re-tests Q4–Q5: because prompts omit stage index, the frozen kernels define longer loops without new model calls. Before simulation we declared MMLU-Pro confident-error runs at H = 8, 10 for Qwen3-4B, Phi-4-mini, and GLM-4- 32B, plus held-out FinQA runs at H = 6 for both generators and workflows (100–400 queries per kernel, 300 replications), using the H = 6 MMLU and H = 3 FinQA blending runs as references and recording the divergence values and three predictions.

Divergence rule. Appendix E.6’s rule holds in 6/8 settings it decides: GLM (distance .36/.39) beats the mean blend at every budget at both horizons; all four FinQA settings (distance .000–.098) show no advantage; Qwen3-4B (distance .26 at both horizons) has anchor/mean ratios .92–.98 but resolves only at $H = 8$ , 400 queries, so its two win predictions fail. Phi-4-mini has distance .19 (no prediction) and wins 5/6 cells.

Kernel reuse. Table 17 shows that longer shared-kernel loops increasingly favor the anchor for MMLU and FinQA Phi but not near-deterministic FinQA Qwen; the prediction holds in 5/7 settings. For Phi at $H = 6$ , anchor MSE is .29–.64 of rollout MSE at every tested budget in both workflows, $\mathrm { i . e . 1 . 6 \mathrm { - 3 . 4 \times } }$ lower and $2 . 4 \mathrm { - } 3 . 4 \times$ lower at 400 queries, which are the reductions quoted in the abstract.

Table 17: Longer review loops favor the anchor over rollouts. Anchored-TIS/rollout MSE ratio (paired z; negative favors the anchor), 300 replications. The Phi H = 6, 400-query ratios .29/.41 give the $3 . 4 \times / 2 . 4 \times$ reductions in the abstract.
<table><tr><td></td><td colspan="3">FinQA H = 3</td><td colspan="3">FinQA H = 6</td></tr><tr><td>Generator, workflow</td><td>100</td><td>200</td><td>400</td><td>100</td><td>200</td><td>400</td></tr><tr><td>Phi-4-mini, ordinary</td><td>1.10 (+1.9)</td><td>.84 (−4.4)</td><td>.77(−5.2)</td><td>.36 (−22.1)</td><td>.30 (−22.5)</td><td>.29 (−22.4)</td></tr><tr><td>Phi-4-mini, unit check</td><td>1.50 (+10.0)</td><td>1.07 (+1.8)</td><td>.88 (−3.1)</td><td>.64 (−10.4)</td><td>.47(−17.3)</td><td>.41 (−19.7)</td></tr><tr><td>Qwen3-4B, ordinary</td><td>2.07(+6.1)</td><td>2.30 (+5.9)</td><td>1.62 (+3.4)</td><td>2.64 (+6.5)</td><td>2.03 (+5.1)</td><td>2.02 (+5.3)</td></tr><tr><td>Qwen3-4B, unit check</td><td>2.28 (+9.5)</td><td>2.26 (+8.6)</td><td>1.74 (+5.4)</td><td>2.73 (+8.8)</td><td>2.09 (+7.2)</td><td>2.13 (+7.3)</td></tr><tr><td>MMLU-Pro, 400 queries</td><td colspan="2">H = 6</td><td colspan="2">H = 8</td><td colspan="2">H = 10</td></tr><tr><td>Qwen3-4B</td><td colspan="2">.80 (−6.6)</td><td colspan="2">.65 (−7.5)</td><td colspan="2">.67 (−5.5)</td></tr><tr><td>Phi-4-mini</td><td colspan="2">.94 (−1.7)</td><td colspan="2">.86 (−3.5)</td><td colspan="2">.84 (−4.0)</td></tr><tr><td>GLM-4-32B</td><td colspan="2">.90 (−2.8)</td><td colspan="2">.95 (−1.1)</td><td colspan="2">.89 (−2.8)</td></tr></table>

Pilot risk. From $H = 6$ to 10, the plain-TIS/anchor MSE ratio at 400 queries changes $7 . 4  7 . 6$ for Phi-4-mini, $3 0 . 1  3 2 . 0$ for GLM, and $1 9 . 0  1 4 . 4$ for Qwen3-4B, so this prediction holds in 2/3 settings. Across all 18 MMLU-Pro cells, the anchor attains . $0 7 \mathrm { - } . 1 9$ of uniform MSE and is resolved better than learned occupancy and the uniform blend in every cell.

## E.9 SELECTING THE ALLOCATION FROM THE PILOT

This section operationalizes Q5: the population tail–mean distance is unavailable, but every learned design already draws a uniform pilot. Before evaluation we declared: for each setting and budget, compute the replication-0 pilot distance between normalized tail and mean influence shares for each question, take the median, use the anchor if it is at least .21 (the midpoint of the population thresholds), and otherwise use the occupancy–mean blend. A choice is wrong only if the selected design has resolved higher MSE $( | z | \geq 2 )$ than the alternative.

Across 46 blending settings (148 setting–budget cells), the rule is wrong in $7 ;$ selected-design MSE relative to the better design has median 1.00. Pilot and population medians have Spearman correlation .86, and replication-0 decisions agree with replications 1–4 in 91% of cases. Three errors are Phi-4-mini confident-error cells with pilot medians .197–.206 just below threshold, costing 3–6% MSE. Four are FinQA Qwen3-4B $H = 6$ cells where near-determinism makes a small pilot overstate distance (pilot .27–.57, population .000), giving the anchor $1 . 4 { - } 2 . 3 \times$ mean-blend MSE. Thus the statistic is reliable except when failures are too rare for the pilot to observe, precisely the regime in which Proposition 2 removes the need for tail-specific allocation.

## E.10 MECHANISM: WHAT THE PILOT SEES AND WHAT ANCHORING CHANGES

![](images/6d6215982be7370f178f4ad892e78bf1d78e94494af3c6d1d72f5e0236593f5d.jpg)

(b)  
![](images/03f7dd9039e32b02b149a7bc9ddf9912c1c666c866afa8e92362a362a3b9f648.jpg)

(c)  
![](images/fa84f365ebda617125134a5316149885d3eeea45f0d7a0693a602c20b7efece6.jpg)  
Figure 5: Anchoring reduces the variance penalty from poor pilots. (a) Realized/oracle leading variance; (b) occupancy versus tail-influence shares; (c) observed MSE versus the no-fit prediction $V ( w ) / n$ . Rollouts are excluded.

Figure 5 explains Q3’s failure mode at the query-share level. Equation 51 penalizes underallocation by dividing squared allocation error by assigned weight. We replay the recorded pilots, floors, and rounding and use population scales to diagnose $\begin{array} { r } { V ( \breve { \widehat { w } } ) = \sum _ { g } \dot { \sigma } _ { g } ^ { 2 } \breve { / \varpi } _ { g } } \end{array}$ for MMLU Qwen $( H = 6$ 400 queries), FinQA Qwen ordinary review (200), and FinQA Phi ordinary review (200). All questions enter; normalization by $\begin{array} { r } { S ^ { 2 } \doteq \bigl ( \sum _ { q } \sigma _ { g } \bigr ) ^ { 2 } } \end{array}$ excludes their 1, 3, and 13 zero-influence questions. Rollouts are excluded because they estimate under a different sampling functional.

Median correlations between population influence and occupancy shares are .74, .85, .62 in the three settings, and the bottom-occupancy half of kernels carries essentially no tail influence (Figure 5(b)), unlike Proposition 1’s equal-visitation construction. For MMLU pilots, median/90thpercentile/maximum $\dot { V } ( \widehat { w } ) / S ^ { 2 }$ is 5.6/136/1,137 for plain TIS, 1.8/5.8/24 for the anchor, and $2 . 6 / 6 . 2 / 7 1$ for occupancy; a single starved group can dominate variance (90th-percentile topcontribution share 1.00). The anchor multiplies the most-starved group’s weight by median factors 7.5, 20, 4.9 across the three settings (MMLU 90th percentile 99). On near-deterministic FinQA Qwen, pilots often see point masses, estimated scales vanish, and TIS approaches uniform: median $V / S ^ { 2 }$ is 72.4 with 73 groups versus 3.3 for the anchor. These are leading-variance diagnostics, not finite-sample MSE guarantees.

Using actual main-sample counts, pilot-averaged $V ( \widehat { w } ) / n$ predicts MSE without fitted constants in the examined nondegenerate cells: median $\log _ { 1 0 } ($ (observed/predicted) is −.001 over 98 MMLU question–method pairs and −.006 over 135 Phi pairs, with 10th–90th percentiles within ±.19 (Figure 5(c)). For near-deterministic Qwen questions, 300 replications can miss rare deviations carrying much of the variance, so observed MSE can be orders of magnitude lower; the plot flags 114 belowrange FinQA pairs (96 Qwen, 18 Phi). Agreement elsewhere is a retrospective check of the variance formula, not a finite-budget guarantee.

## F ADDITIONAL RELATED WORK

Table 18 summarizes the closest foundations; the connections below extend Section 6.

Table 18: Established foundations and the additional results for the conditional-query CVaR problem. The allocation rule is Neyman allocation; the work here identifies and learns the Bellman influence scales it needs.
<table><tr><td>Established starting point</td><td>Additional result here</td></tr><tr><td>Return-law and functional inference under specified sampling (Zhang et al., 2025); quantile-based effi- ciency (Cheng et al., 2026). Neyman allocation and learning un-</td><td>An explicit categorical-CVaR influence for each conditional kernel, including its joint reuse across stages. Theorems 1 and 2 give the variance as a function of query shares and its fixed-design efficiency interpretation. The influence function depends on the unknown model and quantile.</td></tr><tr><td>known stratum variances (Neyman, 1934; Étoré &amp; Jourdain, 2010; Car- pentier et al., 2015).</td><td>Theorem 3 controls learning those quantities, quantile errors, and Bellman remainders to obtain oracle variance and normalized MSE, including pilot cost and individual zero-influence groups.</td></tr><tr><td>Trajectory collection for mean policy evaluation (Mukherjee et al., 2022; 2024).</td><td>Independent conditional queries for a fixed policy&#x27;s tail functional. Proposition 1 shows why visitation and mean-optimal allocation can miss the relevant signal. Comparisons with complete rollouts are</td></tr></table>

The distinction from adjacent work is mainly the design variable. Distributional and quantile-based RL develop inference or efficiency under specified data laws (Zhang et al., 2025; Cheng et al., 2026; Peng et al., 2024; Peng & Zhang, 2026), while generative-access and offline analyses study estimation error under fixed access models (Rowland et al., 2024; Peng et al., 2025; Chandak et al., 2021; Wu et al., 2023; Hong et al., 2025). Stratified and adaptive experimental design learn Neyman allocations when the within-stratum target is already defined (Carpentier et al., 2015; Dai et al., 2023); our score itself depends on an unknown Bellman continuation model and CVaR cutoff, which is why Theorem 3 must control learning the influence function as well as its allocation. Risksensitive control and logging-policy design change the policy or trajectory distribution (Bauerle &¨ Ott, 2011; Zhu et al., 2024; Douglas et al., 2026); here the policy and conditional laws stay fixed and only the independent conditional-query counts change. Active testing allocates effort across benchmark items (Nguyen et al., 2018; Kossen et al., 2021; Maia Polo et al., 2024; Li et al., 2025); our experiments allocate within an item’s stochastic workflow, and combining the two levels is a natural extension.