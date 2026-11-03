# The Staged Pandora’s Box: Reservation Indices for Multi-Fidelity Bayesian Optimization

Long Vu<sup>1,2</sup> Jongseong Brad Choi<sup>1,3</sup> Binoy Ravindran<sup>2</sup> Zhoulai Fu<sup>1,2,3</sup>

<sup>1</sup>State University of New York Korea <sup>2</sup>Virginia Tech <sup>3</sup>Stony Brook University long-vu@outlook.com

## Abstract

A staged search must decide which candidate to advance next and when to stop. Some observations only inform later decisions; others can be kept as results. We formulate these choices as a staged Pandora’s box with a designated set of collectable stages. For independent finite-stage candidates with bounded readings and positive bounded own-history costs, an exact covered-call recursion gives the reservation index, the optimal value and the loss of every advance or stop; advancing an index-maximal candidate while its index exceeds the incumbent is optimal. The representation recovers classical single-stage search, nested inspection and the all-collectable Markov setting, and prices changes in the problem: enlarging the collectable set or lowering costs raises the value by an explicit covered-call overshoot. Resumption can be valuable even when future reward laws do not change, but has no value for deterministic readings. With information sharing across candidates, no rule using only their separate remaining-reading laws is universally optimal, and making an intermediate reading collectable can change which candidate is inspected first. Under a coherent joint path law with own-history costs, an approximateplanning bound controls the recomputed index’s regret by changes in untouched candidates local values; a common-component specialization bounds those changes through predictive totalvariation dispersion. Exact finite-law experiments check the representation, the loss accounting and the bounds.

## 1 Introduction

Cost-aware search has two coupled decisions, in Weitzman’s terms a selection rule, which candidate to inspect next, and a stopping rule, whether to inspect at all (Weitzman, 1979). For independent candidates inspected in one step, one reservation value answers both: open a box of largest reservation value while that value exceeds the best result already available. The Pandora’s box Gittins index (PBGI) brings this principle to cost-aware Bayesian optimization (Xie et al., 2024, 2026).

An evaluation need not reveal its result at once. A training run may be paused at checkpoints and resumed later; a proxy evaluation may inform the decision to pay for a full one without being a deployable result; a learning curve may make every checkpoint a usable result. We represent these situations with a finite ladder per candidate and a collectable set: only readings at designated stages enter the incumbent. The searcher may advance any available candidate by one stage, return to another, or stop and keep its best collectable reading less the costs incurred. Staging changes both rules: selection includes resuming a paused candidate, and what stopping keeps depends on the collectable set.

In multi-fidelity Bayesian optimization, a candidate can be a fixed hyperparameter configuration and its ladder a prescribed sequence of fidelities or training checkpoints. Terminal-only collection models proxy evaluations whose outputs cannot be deployed; checkpoint collection models intermediate outputs that are themselves admissible results. We study the search structure on a finite candidate pool under bounded planning laws; the numerical studies verify this structure on known laws.

We give an explicit covered-call representation of this problem. At a nonexhausted arm state s with a collectable stage left, let $\tilde { Y }$ be the next stage’s reading if it is collectable and $- \infty$ otherwise, and let $s ^ { \prime }$ be the successor own state. The recursion

$$
W ( s ) = \operatorname* { m a x } \{ \tilde { Y } , Z ( s ^ { \prime } ) \} , \qquad Z ( s ) = \operatorname* { m i n } \{ G ( s ) , W ( s ) \} , \qquad \mathbb { E } [ ( W ( s ) - G ( s ) ) ^ { + } ] = c ( s ) ,
$$

represents each remaining arm by its next stage’s collectable reading or future capped value, capped again at its current reservation value $G ( s )$ . A sufix without a collectable stage, including an exhausted arm, has $G = Z = - \infty$ and needs no reservation equation. Under independent arms the optimal value is the expectation of the largest covered call value and the incumbent (Theorem 1). The same representation gives the exact Bellman loss of every advance and stop, hence of any policy (Theorem 2), and prices changes in the problem: enlarging the collectable set or lowering costs raises the value by an explicit overshoot, planning with a smaller collectable set loses at most that amount, and deterministic readings need no resumption. With stochastic collectable rewards, resumption can help without learning about unobserved rewards (Section 4).

We do not claim a new index theorem: the policy is consistent with classical superprocess results (Whittle, 1980; Glazebrook, 1982), and the single-stage, terminal-only and all-collectable Markov cases have established solutions (Section 1.2). Our contribution is the explicit recursion, value representation and action losses for designated collection, together with the efects of collection rights on acquisition and the local-value discrepancy under sharing. The construction separates information from collectable reward while accommodating own-history states and costs, features already permitted in nested inspection models (Bowers and Waggoner, 2026, Section 4.1).

Under a shared surrogate, one candidate’s observation changes the law of the others, and separate arm problems omit the value of that change. Two instances with identical separate arm laws can have diferent uniquely optimal first actions (Proposition 5), and collection rights reshape acquisition under sharing: in a two-arm family with a common latent state, the optimal first advance alternates A, B, A, B in the prior when the second arm’s intermediate reading is collectable, and is always A when it is not (Proposition 6). For a coherent joint path law with arm-local costs, an approximate-planning bound controls the recomputed index’s regret by how much untouched arms local values change (Proposition 7); when sharing flows through a common component, a local envelope is the arms’ sensitivity to the component times the predictive total-variation dispersion of one more reading (Corollary 2). A vanishing regret bound requires this envelope to vanish uniformly over feasible future histories; posterior concentration alone is insuficient. These conditions are not established for a general correlated Gaussian process.

## 1.1 Contributions

We provide (i) the staged model and a covered-call recursion giving the optimal value and index policy under independence; (ii) exact stop and advance losses with an arbitrary-policy decomposition, exact comparative statics for collection rights and costs, and a bound on the value of resumption; (iii) under information sharing, the limits of own-arm-law policies, a certified family in which collection rights split the optimal first-action region, and an approximate-planning bound expressed through local-value changes and a common-component dispersion envelope; and (iv) exact finitestate checks of the value and loss identities, comparative statics and sharing bounds against direct Bellman optimization.

## 1.2 Related work

Multi-stage inspection has a reservation-index theory. Kleinberg et al. (2016, Appendix G) give generalized strike prices and Bowers and Waggoner (2026, Section 4, Definition 4) a constructive nested-index and capped-value recursion, both for terminal-only collection; the latter also explic itly allow later costs to depend on preceding signals (Bowers and Waggoner, 2026, Section 4.1). Gupta et al. (2019) treat independent Markov inspection processes with rewards at destination states, nonnegative prices and unbounded depth, of which the fixed-depth positive-cost case is recovered here; Boodaghians et al. (2023) handle independent rewards under line or tree precedence. Closest for intermediate collection is Yang et al. (2025, Theorem 4.4): each reward on a Markov line is collectable, probing costs are fixed, and the optimal policy selects the line with largest generalized reservation value. With positive costs that is the corresponding special case here; their finite-support dynamic program and our history-state covered-call construction are diferent representations. The superprocess literature supplies suficient conditions for an index theorem (Whittle, 1980; Glazebrook, 1982; Scully and Terenin, 2025); Appendix B identifies that condition for this model. More general Markov search, in which a process ofers a choice of actions, need not admit an index policy (Glazebrook, 1982; Scully and Terenin, 2025); one-shot search with an optional partial inspection has none and is NP-hard (Aouad et al., 2025), as is combinatorial Markov search (Bowers et al., 2026).

Correlated search is also established: learning and approximation from joint-distribution samples (Chawla et al., 2020), the reduction of fully adaptive correlated search to Uniform Decision Tree (Chawla et al., 2023), correlation-aware reservation rules (Gergatsouli and Tzamos, 2023), and a tight factor four against the best partially adaptive policy for a minimization objective (Bansa et al., 2025). Those approximation factors do not transfer to our reward-minus-cost objective, and the correlated-BO stopping guarantee of Xie et al. (2026), no worse than stopping after one evaluation, is a diferent kind of statement from Bellman optimality. In Bayesian optimization, PBGI is the single-stage predecessor (Xie et al., 2024); freeze-thaw and fidelity-aware surrogates (Swersky et al., 2014; Klein et al., 2017) motivate staged observation models; trace-aware knowledge gradient (Wu et al., 2020), successive halving and Hyperband (Jamieson and Talwalkar, 2016; Li et al., 2018) allocate resources by other means. For this utility in fixed-pool freeze-thaw optimization, Vu et al. (2026) analyze the one-step rule, expected improvement stopping, and a certified two-step correction; the present paper gives the exact index for the independent-arm staged problem.

Organization. Section 2 defines the model. Section 3 states the covered-call recursion, the index theorem and the loss decomposition, with proofs; Sections 4 and 5 treat comparative statics and information sharing. Section 6 summarizes the exact numerical verification, whose retained models, checks and results are in Appendix E, and Section 7 concludes. Algorithm 1 gives the policy implementation in the main text.

## 2 The Staged Pandora’s Box

Candidates, ladders and the collectable set. There is a fixed pool $\mathcal { X } = \{ x _ { 1 } , . . . , x _ { N } \}$ of candidates, the arms. Arm i has a fixed finite ladder of $T _ { i }$ stages, or rungs; after b advances in total it stands at stage $t _ { i , b } \in \{ 0 , \ldots , T _ { i } \}$ , and one action, an advance, takes it to stage $t _ { i , b } + 1$ at a cost and reveals a reading Y. The set $K _ { i } \subseteq \{ 1 , \dots , T _ { i } \}$ is fixed before the run. A stage $t \in \kappa _ { i }$ is collectable: its reading enters the incumbent. Write $\tilde { Y } = Y$ if the stage reached is collectable and $\tilde { Y } = - \infty$ otherwise, the collectable reading. The history state is $S _ { b } = ( \mathcal { C } _ { b } , t _ { 1 : N , b } , \tilde { y } _ { b } , K _ { b } )$ . Here $\mathcal { C } _ { b }$ records the ordered action and observation history, including noncollectable readings and any fixed starting context; $t _ { 1 : N , b }$ records the arms’ stages; $\tilde { y } _ { b }$ is the best collectable reading seen or the initial value $q _ { 0 } ;$ ; and $K _ { b }$ is cumulative cost. The planning law $p _ { \theta } ( \cdot \mid S _ { b } , i )$ is the predictive distribution of arm $i \mathrm { \ ' } _ { \mathrm { S } }$ next reading under a specified predictive model whose parameters $\theta$ are fixed before the run and never refitted; it may condition on the whole context, so an observation on one arm can change the prediction for every arm. Advancing $i \in \mathcal { I } ( S _ { b } )$ , the set of arms with a stage left, costs $c ( S _ { b } , i )$ and reveals $Y _ { b + 1 } \sim p _ { \theta } ( \cdot \mid S _ { b } , i )$ ; the successor $S _ { b + 1 } = F ( S _ { b } , i , Y _ { b + 1 } )$ adds the observation to the context, advances $t _ { i }$ by one, sets $\tilde { y } _ { b + 1 } = \operatorname* { m a x } \{ \tilde { y } _ { b } , \tilde { Y } _ { b + 1 } \}$ and adds the cost to $K _ { b }$ . Availability is arm-local: advancing i never changes another arm’s ladder. A policy $\pi$ chooses, measurably in the observed history, an action in $\{ \operatorname { s t o p } \} \cup \mathcal { T } ( S _ { b } ) ; \tau _ { \pi }$ is its stopping time. Randomized policies may use auxiliary randomness independent of the innovations driving the planning kernel. The searcher always retains earlier collectable readings, so collecting at an intermediate stage does not require terminating the search.

Utility, values and the Bellman equation. The objective is $\operatorname* { s u p } _ { \pi } \mathbb { E } _ { \theta , \pi } [ \tilde { y } _ { \tau _ { \pi } } - K _ { \tau _ { \pi } } \ | \ S _ { 0 } ]$ . It is the budget-free utility of Vu et al. (2026) for fixed-pool freeze-thaw optimization, with a cost per stage in place of a cost per increment and only collectable readings entering the score. The stop-now payof is $h ( S _ { b } ) = \tilde { y } _ { b } - K _ { b }$ , the value of π is $V _ { \pi } ( S _ { b } ) = \mathbb { E } _ { \theta , \pi } [ \tilde { y } _ { \tau _ { \pi } } - K _ { \tau _ { \pi } } \ | \ S _ { b } ]$ , and $Q _ { \pi } ( S _ { b } , i ) =$ $\mathbb { E } _ { \theta } [ V _ { \pi } ( F ( S _ { b } , i , Y _ { b + 1 } ) ) \ | \ S _ { b } , i ]$ is the value of forcing i once and then following π. The optimal value $V ^ { \star } = \operatorname* { s u p } _ { \pi } V _ { \pi }$ satisfies $V ^ { \star } ( S ) = \operatorname* { m a x } \{ h ( S )$ $\operatorname* { m a x } _ { i \in { \mathcal { I } } ( S ) } Q ^ { \star } ( S , i ) \}$ , with max $\mathcal { D } = - \infty$ , and the ladders being finite, a policy that attains the maximum at every state is optimal. For comparison with continuation-value conventions, define $U _ { \pi } ( S ) = V _ { \pi } ( S ) + K ( S )$ . With $S ^ { \prime } = F ( S , i , Y )$ ，

$$
Q _ { \pi } ( S , i ) + K ( S ) = - c ( S , i ) + \mathbb { E } _ { \theta } [ U _ { \pi } ( S ^ { \prime } ) \mid S , i ] .
$$

Thus $V _ { \pi }$ includes sunk costs, while $U _ { \pi }$ removes them; the cost of the forced advance is already included in $K ( S ^ { \prime } )$ inside the definition of $Q _ { \pi }$

Assumption 1 (Standing assumptions). The planning law is a fixed measurable kernel, never refitted during a run; all readings and incumbents lie in fixed bounds $[ y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } } ]$ chosen in advance; the cost $c ( S , i )$ is measurable, determined by $( S , i )$ before the advance, and satisfies $0 < c _ { \mathrm { m i n } } \le$ $c ( S , i ) \leq c _ { \operatorname* { m a x } } < \infty ;$ every ladder is finite. History states carry the Borel sigma-fields generated by the finite arm labels and their real observations; the successor map is measurable.

Finite horizon and conventions. We write $q ( S ) = \tilde { y } ( S )$ and $\begin{array} { r } { L ( S ) = \sum _ { i } ( T _ { i } - t _ { i } ) } \end{array}$ throughout. Every advance lowers L by one, so every admissible policy stops after at most $L ( S )$ further advances, and finite-action backward induction defines Borel Bellman values and an attaining deterministic policy, choosing the first action in any fixed measurable tie order. The stated index policy stops on an index–incumbent tie. We use $( - \infty - r ) ^ { + } = 0$ for finite r and max $\varnothing \ = \ \operatorname* { s u p } \varnothing \ = \ - \infty$ Appendix A.1 collects the remaining conventions and compares them with the closest index and inspection models.

Per-arm planning laws. A planning law is per-arm when there are measurable kernels $P _ { i } ( d y \mid$ $s _ { i } )$ and cost functions $c _ { i } ( s _ { i } )$ such that $p _ { \theta } ( d y \mid S , i ) = P _ { i } ( d y \mid s _ { i } )$ and $c ( S , i ) = c _ { i } ( s _ { i } )$ . An arm’s own state $s _ { i }$ is its stage and complete observation history. Advancing another arm leaves $s _ { i } ,$ its predictive kernel and its next cost unchanged; an arm does not evolve while paused. We write $\nu ( s _ { i } ) = T _ { i } - t _ { i }$ and abbreviate its cost as $c ( s _ { i } )$ . No Gaussian, density, or scalar first-order Markov assumption is imposed.

At a fixed state $S ,$ iterate $P _ { i }$ to obtain the law $\mathsf { P } _ { i } ^ { s _ { i } }$ of the entire remaining path of arm $i .$ The common space used in the independent-arm proofs has law

$$
\mathsf { P } _ { S } = \bigotimes _ { i = 1 } ^ { N } \mathsf { P } _ { i } ^ { s _ { i } } .
$$

One can sample these paths in advance and reveal their next coordinates as the policy chooses arms. Autonomy makes this construction agree with the sequential planning kernels. Revealing a coordinate of one path conditions that path alone; the other remaining paths keep their laws. Dependence within a path is unrestricted and its future costs are functions of its successive own states. Theorem 1 uses this product construction; Section 5 does not assume it.

Single-arm stopping problems. Fix a history state $S ,$ an available arm $i ,$ and a depth cap $k \geq 1$ . Advance only $i ,$ obtaining states $S ^ { ( m ) }$ by iterating the planning kernel and successor map. Let $\mathcal { G } _ { m }$ be the filtration of those observations and $\nu _ { i } = T _ { i } - t _ { i }$ the remaining stages. An admissible arm run is a stopping time $1 \leq \sigma \leq$ min $\{ k , \nu _ { i } \}$ for this filtration; it pays every advance up to $\sigma$ and may abandon the arm after any reading. Write $\mathcal { R } _ { i } ^ { ( k ) } ( S )$ for these runs, $M _ { \sigma }$ for the best newly obtained collectable reading $( - \infty$ if none), and $\kappa _ { \sigma }$ for the run’s total additional cost. Earlier collectable readings remain in the incumbent, rather than in $M _ { \sigma }$ . For an outside value $r ,$ define

$$
\Phi _ { i } ^ { ( k ) } ( S ; r ) = \operatorname* { s u p } _ { \sigma \in { \mathscr R } _ { i } ^ { ( k ) } ( S ) } \{ { \mathbb E } [ ( M _ { \sigma } - r ) ^ { + } \mid S ] - { \mathbb E } [ \kappa _ { \sigma } \mid S ] \} , \qquad G _ { i } ^ { ( k ) } ( S ) = \operatorname* { s u p } \{ r : \Phi _ { i } ^ { ( k ) } ( S ; r ) > 0 \} .
$$

The empty supremum for G is −∞. The run law is the original planning law from $S ; r$ is a payof comparison, not a modification of that law. The full index uses $k = \nu _ { i } ;$ exhausted arms have index $- \infty$ . If a collectable stage remains, let $\mu _ { i } ( S )$ be the number of advances to the next such stage. The first advance is forced in $\Phi ;$ the option of stopping immediately is represented by $\Phi ^ { + } = \operatorname* { m a x } \{ 0 , \Phi \}$ The auxiliary value r may lie outside the range of observed scores.

Proposition 1 (Reservation-value properties). Under Assumption $\mathbf { \Phi } _ { 1 , r \mapsto \Phi _ { i } ^ { ( k ) } ( S ; r ) }$ is nonincreasing and 1-Lipschitz. The index is finite exactly when a collectable stage lies within the cap, in which case

$$
y _ { \operatorname* { m i n } } - \mu _ { i } ( S ) c _ { \operatorname* { m a x } } \leq G _ { i } ^ { ( k ) } ( S ) \leq y _ { \operatorname* { m a x } } - c ( S , i ) .
$$

In every case $\Phi _ { i } ^ { ( k ) } ( S ; r ) > 0$ if and only if $r < G _ { i } ^ { ( k ) } ( S )$ , and $G _ { i } ^ { ( k ) } ( S )$ is nondecreasing in $k .$ When the index is finite, $\Phi _ { i } ^ { ( k ) } ( S ; G _ { i } ^ { ( k ) } ( S ) ) = 0$

Appendix A gives the proof.

The index is measurable: it is also the supremum of the rational a for which $\Phi _ { i } ^ { ( k ) } ( S ; a ) > 0$ . Thus selecting a smallest-numbered index maximizer defines a measurable policy even for continuous or atomic reading laws.

These definitions also apply under a shared planning law: the current history fixes the starting conditional law, and the hypothetical run observes only the chosen arm. The independent-arm theorem requires the additional autonomy assumption given above.

## 3 Main Results

## 3.1 Covered-call recursion and optimal policy

Throughout this section the planning law is per-arm and Assumption 1 holds. Fix a nonexhausted arm and write s for its own state, Y for its next reading, $\tilde { Y }$ for the collectable reading and $s ^ { \prime }$ for its state after it. Write $\Phi ( s ; r ) = \Phi _ { i } ^ { ( \nu _ { i } ) } ( S ; r )$ and $G ( s ) = G _ { i } ^ { ( \nu _ { i } ) } ( S )$ for the single-arm quantities of Section 2. All expectations in the local calculation are under $\mathsf { P } _ { i } ^ { s } \mathrm { ; }$ ; after observing $Y _ { i }$ , future expectations use the conditional continuation law from $s ^ { \prime } .$ . The continuation improvement is zero at an exhausted arm; the outside option is retained. Lemma 3 justifies the following Bellman recursion and its attaining stopping rules for arbitrary within-arm kernels. After the forced first advance the run either stops or continues from $s ^ { \prime }$ against the incumbent max $\{ r , { \tilde { Y } } \}$ , and $( \operatorname * { m a x } \{ \tilde { Y } , M ^ { \prime } \} - r ) ^ { + } -$ $( \tilde { Y } - r ) ^ { + } = ( M ^ { \prime } - \operatorname * { m a x } \{ r , \tilde { Y } \} ) ^ { + }$ for the best later collectable reading $M ^ { \prime }$ , so

$$
\Phi ( s ; r ) = \mathbb { E } \big [ ( \tilde { Y } - r ) ^ { + } - c ( s ) + \Phi ( s ^ { \prime } ; \operatorname* { m a x } \{ r , \tilde { Y } \} ) ^ { + } \big ] ,\tag{1}
$$

with $\Phi ( s ^ { \prime } ; \cdot ) ^ { + } \equiv 0 \mathrm { i f } \nu ( s ^ { \prime } ) = 0$

Definition 1 (Nested value and covered call value). For an arm state s define, by backward recursion on $\nu ( s ) , Z ( s ) = - \infty \ i f \nu ( s ) = 0$ and otherwise

$$
W ( s ) = \mathrm { m a x } \{ \tilde { Y } , Z ( s ^ { \prime } ) \} , \qquad Z ( s ) = \mathrm { m i n } \{ { G ( s ) } , W ( s ) \} .
$$

$W ( s )$ is a function of the arm’s readings from s onward; $Z ( s )$ is its covered call value. These are random variables on the full remaining arm path, not readings disclosed by the first advance. The index $G ( s )$ is a number determined by the current state. Finite backward recursion makes W, Z jointly measurable in the starting state and continuation path, using measurability of G from Section 2.

When collection remains, the backward computation is: obtain the child caps, form $W ,$ , solve its reservation equation for $G ,$ and cap W to obtain $Z .$ The lemma below shows that this computed root equals the stopping threshold already defined in Section 2, and that the cap accounts for the advance cost in the keep-or-run value.

Lemma 1 (Nested box). For every arm state s with $\nu ( s ) \geq 1$ and every $r \in \mathbb { R }$

$$
\Phi ( s ; r ) = \mathbb { E } \big [ ( W ( s ) - r ) ^ { + } \big ] - c ( s ) , \qquad \mathbb { E } \big [ ( W ( s ) - G ( s ) ) ^ { + } \big ] = c ( s ) ~ i f \ : G ( s ) > - \infty ,
$$

and E[max $\{ r , Z ( s ) \} ] = r + \Phi ( s ; r ) ^ { + }$ . If a collectable stage remains, the reservation equation has a unique finite solution, and $W ( s ) , Z ( s )$ are bounded real random variables. If none remains, $G ( s ) = W ( s ) = Z ( s ) = - \infty$ and $\Phi ( s ; r ) = - c ( s )$ . The local value therefore has Weitzman’s positive-part form, with $G ( s )$ the reservation price and $Z ( s )$ the covered call value (Kleinberg et $a l .$ ${ \it 2 0 1 6 } )$

Proof. Induction on $\nu ( s )$ . If $\nu ( s ^ { \prime } ) = 0$ then $W ( s ) = { \tilde { Y } }$ and (1) is the first claim. Otherwise the hypothesis at $s ^ { \prime }$ gives $\Phi ( s ^ { \prime } ; r ^ { \prime } ) ^ { + } = \mathbb { E } [ \operatorname* { m a x } \{ r ^ { \prime } , Z ( s ^ { \prime } ) \} \mid Y ] - r ^ { \prime }$ with $r ^ { \prime } = \operatorname* { m a x } \{ r , \tilde { Y } \}$ , and (1) becomes

$$
\begin{array} { r l } & { \Phi ( s ; r ) = \mathbb { E } \big [ ( \tilde { Y } - r ) ^ { + } + \operatorname* { m a x } \{ r , \tilde { Y } , Z ( s ^ { \prime } ) \} - \operatorname* { m a x } \{ r , \tilde { Y } \} \big ] - c ( s ) } \\ & { \qquad = \mathbb { E } \big [ \operatorname* { m a x } \{ r , \tilde { Y } , Z ( s ^ { \prime } ) \} - r \big ] - c ( s ) = \mathbb { E } \big [ ( W ( s ) - r ) ^ { + } \big ] - c ( s ) , } \end{array}
$$

using $( \tilde { Y } - r ) ^ { + } = \operatorname* { m a x } \{ r , \tilde { Y } \} - r _ { : }$ which holds also for $\tilde { Y } = - \infty$ . If no collectable stage remains, all continuation covered calls are −∞ and stopping after the forced advance gives $\Phi = - c ( s )$

Otherwise backward induction, together with Proposition 1, bounds the finite $W , Z$ between $y _ { \mathrm { { m i n } } } -$ $\nu ( s ) c _ { \mathrm { m a x } }$ and $y _ { \mathrm { m a x } }$ . This makes their positive-part expectations finite. Continuity and Proposition 1 give $\mathbb { E } [ ( W - G ) ^ { + } ] = c ( s )$ . The root is unique: for $a < b$ with $\mathbb { E } [ ( W - b ) ^ { + } ] > 0$

$$
\begin{array} { r } { \mathbb { E } [ ( W - a ) ^ { + } ] - \mathbb { E } [ ( W - b ) ^ { + } ] \ge ( b - a ) \mathbb { P } ( W > b ) > 0 . } \end{array}
$$

Two roots cannot therefore have the same strictly positive level $c ( s ) { \mathrm { ; } }$ ; no density or absence-ofatoms assumption is needed. For the covered-call identity, if $r < G ( s )$ then $( \operatorname* { m i n } \{ G , W \} - r ) ^ { + } =$ $( W - r ) ^ { + } - ( W - G ) ^ { + }$ , so $\mathbb { E } [ \operatorname* { m a x } \{ r , Z \} ] = r + \Phi ( s ; r )$ with $\Phi ( s ; r ) > 0 ; { \mathrm { i f } } r \geq G ( s )$ then $Z \leq G \leq r$ gives $\mathbb { E } [ \operatorname* { m a x } \{ r , Z \} ] = r$ and $\Phi ( s ; r ) \le 0$ □

The representation holds for every outside option while preserving the staged observation protocol. In particular, $Y$ and $Z ( s ^ { \prime } )$ may be dependent; the proof integrates their actual joint law. The role of collectable intermediate readings is exactly the maximum in W. No discretization is used in this lemma; Example 1 gives a continuous dependent instance with both stages collectable.

Theorem 1 (Index optimality under a per-arm planning law). Under Assumption 1 with a per-arm planning law, for every state S with incumbent q, cumulative cost K and arm states $s _ { 1 } , \ldots , s _ { N }$ 2

$$
V ^ { \star } ( S ) + K = \mathbb { E } \Big [ \operatorname* { m a x } \big \{ q , \operatorname* { m a x } _ { i } Z _ { i } ( s _ { i } ) \big \} \Big ] ,
$$

the expectation being over the independent covered call values of the arms. Stopping is optimal if and only if ma $\begin{array} { r } { \mathrm { x } _ { i } G _ { i } ( s _ { i } ) \le q . } \end{array}$ , and when max<sub>i</sub> $G _ { i } ( s _ { i } ) > q$ advancing any arm with the largest index is optimal. In particular the index policy π , which stops when ma $\begin{array} { r } { \mathrm { { { x } } } _ { i } G _ { i } ( s _ { i } ) \leq q } \end{array}$ and otherwise advances an index-maximal arm, has $V _ { \pi _ { \mathrm { I X } } } = V ^ { \star }$

Proof. All expectations below are under the product remaining-path law $\mathsf { P } _ { S }$ from Section 2. Let $\mathcal { V } ( S ) = \mathbb { E } [ \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { i } Z _ { i } ( s _ { i } ) \} ]$ . We verify this candidate by showing that every advance has nonpositive Bellman residual, with equality for a maximal-index advance above q; otherwise stopping attains the value. Finite-horizon induction then identifies V with $V ^ { \star } + K$ . Then $\nu \geq q .$ , with equality when max $\iota _ { i } G _ { i } ( s _ { i } ) \leq q$ because $Z _ { i } \ \leq \ G _ { i }$ . If an available arm has $G _ { i } = - \infty$ , it has no collectable stage left. Its advance changes neither the incumbent nor the other arms’ states or laws, so its Bellman residual is $\mathbb { E } [ \mathcal { V } ( S ^ { \prime } ) ] - c ( s _ { i } ) - \mathcal { V } ( S ) = - c ( s _ { i } ) < 0$

Now fix an available arm with finite index, abbreviate $W _ { i } = W _ { i } ( s _ { i } )$ and $G _ { i } = G _ { i } ( s _ { i } )$ , and let $R = \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { j \neq i } Z _ { j } ( s _ { j } ) \}$ , the outside option supplied by the incumbent and the rest of the pool. It is random but independent of arm i’s entire remaining path. After its first reading $Y$ , the conditional law of that arm is the law from $s _ { i } ^ { \prime } ,$ while every other path retains its law. Consequently, almost surely,

$$
\mathcal { V } ( S ^ { \prime } ) = \mathbb { E } \Bigg [ \operatorname* { m a x } \{ q , \tilde { Y } , Z _ { i } ( s _ { i } ^ { \prime } ) , \operatorname* { m a x } _ { j \neq i } Z _ { j } ( s _ { j } ) \} \mid Y \Bigg ] .
$$

Taking the outer expectation gives $\mathbb { E } [ \mathcal { V } ( S ^ { \prime } ) ] = \mathbb { E } [ \operatorname* { m a x } \{ R , W _ { i } \} ]$ by the tower property; it does not assume $W _ { i }$ is observed. By definition, $\mathcal { V } ( S ) = \mathbb { E } [ \operatorname* { m a x } \{ R , \operatorname* { m i n } \{ G _ { i } , W _ { i } \} \} ]$ . The pathwise identity

$$
\operatorname* { m a x } \{ R , W \} - \operatorname* { m a x } \{ R , \operatorname* { m i n } \{ G , W \} \} = ( W - \operatorname* { m a x } \{ R , G \} ) ^ { + }
$$

therefore gives

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { V } ( S ^ { \prime } ) ] - c ( s _ { i } ) - \mathcal { V } ( S ) = \mathbb { E } \big [ ( W _ { i } - \operatorname* { m a x } \{ R , G _ { i } \} ) ^ { + } \big ] - c ( s _ { i } ) } \\ & { \qquad \le \mathbb { E } \big [ ( W _ { i } - G _ { i } ) ^ { + } \big ] - c ( s _ { i } ) = 0 } \end{array}
$$

by Lemma 1. The inequality is an equality whenever $R \leq G _ { i }$ almost surely. If $G _ { i } = \operatorname* { m a x } _ { j } G _ { j } > q$ then $R \leq G _ { i }$ , because $q < G _ { i }$ and $Z _ { j } \leq G _ { j } \leq G _ { i }$ for every $j ,$ so equality holds throughout for every index-maximal arm. Hence $\begin{array} { r } { \mathcal { V } ( S ) = \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { i } ( \mathbb { E } [ \mathcal { V } ( S ^ { \prime } ) ] - c ( s _ { i } ) ) \} } \end{array}$ at every state, attained by stopping when max<sub>i</sub> $G _ { i } \leq q$ and by every index-maximal arm otherwise. In the second case stopping does not attain it: for an index-maximal arm, $\mathbb { E } [ ( W _ { i } - G _ { i } ) ^ { + } ] = c ( s _ { i } ) > 0$ gives $\mathbb { P } ( W _ { i } > G _ { i } ) > 0$ and on that event $Z _ { i } = G _ { i } > q$ , so $\mathcal { V } ( S ) > q$ . This is the Bellman equation for $V ^ { \star } + K$ ; the ladders are finite, V agrees with $V ^ { \star } + K$ at every state with no stage left, and backward induction on the stages remaining gives $\mathcal { V } = V ^ { \star } + K$ everywhere. A policy that at every state attains the maximum in the Bellman equation of a finite-horizon problem is optimal. □

At the boundary max<sub>i</sub> $G _ { i } ~ = ~ q$ , stopping is optimal and advancing any index-maximal arm is also optimal by the same residual calculation. The stated policy chooses to stop on that tie; optimal stopping is not asserted to be unique. At equal maximal indices above $q ,$ , any measurable tie-breaking rule is valid. No continuity of the reading distributions is needed.

Algorithm 1 combines the backward computation with the policy of Theorem 1. The function returns an index and a conditional covered-call law; its recursive calls integrate over hypothetical futures while other arms’ observed histories remain fixed. Only the policy’s advance observes a reading. Appendix F gives finite-tree calculations, complexity and the qualification for continuous laws.

Algorithm 1 Staged index computation and search   
Require: Initial state ${ \overline { { S _ { 0 } } } } ,$ fixed planning law, ladders, collectable sets and costs   
1: function CoveredCall(i, T)   
2: if arm i has no collectable stage left at T then   
3: return $( - \infty , \delta _ { - \infty } )$ ▷ point mass at −∞   
4: end if   
5: For each possible next reading y, obtain   
$( \cdot , \mu _ { y } ) \gets \mathrm { C o v E R E D C A L L } ( i , F ( T , i , y ) )$   
6: Form the law of $W = \operatorname* { m a x } \{ \tilde { Y } ( Y ) , Z _ { Y } \}$ using   
$Y \sim p _ { \theta } ( \cdot \mid T , i )$ and $Z _ { Y } \mid Y = y \sim \mu _ { y }$   
7: Solve for the unique g with $\mathbb { E } [ ( W - \bar { g } ) ^ { + } \mid T , i ] = c ( T , i )$   
8: return (g, Law(min $\{ g , W \} \mid T , i ) )$   
9: end function   
10: procedure StagedIndex(S<sub>0</sub>)   
11: $S  S _ { 0 }$   
12: while $\mathcal { T } ( S ) \neq \emptyset$ do   
13: for all $i \in \mathcal { T } ( S )$ do   
14: $( G _ { i } , \cdot ) \gets \mathrm { C o v E R E D C A L L } ( i , S )$   
15: end for   
16: if max<sub>i∈I(S)</sub> $G _ { i } \leq q ( S )$ then   
17: break   
18: end if   
19: i ← min arg max $\mathbf { \chi } _ { j \in \mathbb { Z } ( S ) } G _ { j }$   
20: Pay $c ( S , i )$ and observe the next reading Y   
21: $S \gets \dot { F } ( S , i , Y )$ ; update predictions using this observation   
22: end while   
23: return incumbent q(S) and net utility $h ( S ) = q ( S ) - K ( S )$   
24: end procedure

The policy takes at most $L ( S _ { 0 } )$ advances. Under a coherent shared law, it recomputes indices from the updated posterior marginals after each observation; Proposition 7 gives its guarantee

under that proposition’s conditions.

Established special cases. The theorem permits dependent readings along an arm through its own history; it requires other arms’ laws to remain unchanged when that arm is observed.

For terminal-only collection the index is the generalized strike price of Kleinberg et al. (2016, Appendix G), and the recursion is that of Bowers and Waggoner (2026, Section 4, Definition 4). If every reading is collectable and its scalar transitions are first-order Markov with fixed positive stage costs, the policy is the multi-line generalized reservation rule of Yang et al. (2025, Theorem 4.4). Independent readings along a line also recover the corresponding order-constrained result of Boodaghians et al. (2023). Our history-state recursion and the value identity represent designated collection and costs in the same construction. The superprocess interpretation in Appendix B supplies the connection to the classical index theorem.

Corollary 1 (Single-stage Pandora index). If $T _ { i } ~ = ~ 1$ and $\mathcal { K } _ { i } ~ = ~ \{ 1 \}$ for every arm, $G _ { i }$ solves $\mathbb { E } [ ( Y _ { i } - G _ { i } ) ^ { + } ] = c _ { i }$ , and Theorem 1 is Weitzman’s theorem. The policy is the independent-candidate PBGI of Xie et al. $( 2 0 \% )$

Proof. With one collectable stage, $W ( s ) = \tilde { Y } = Y$ and $Z ( s ) = \displaystyle \operatorname* { m i n } \{ G ( s ) , Y \}$ , so the reservation equation of Lemma 1 is $\mathbb { E } [ ( Y - G ) ^ { + } ] = c$ and the value identity of Theorem 1 is Weitzman’s.

Having only one collectable stage is not suficient for this corollary if earlier noncollectable inspection stages remain: those observations can support adaptive abandonment and still require the nested recursion.

## 3.2 Exact losses of arbitrary policies

The covered call values do more than identify the optimal action: they price every suboptimal one exactly, so the loss of any policy against the optimum can be located, not only bounded.

Theorem 2 (Loss decomposition). Under the hypotheses of Theorem $^ { 1 , }$ at a state $S$ with incumbent $q$ let

$$
\Delta _ { \mathrm { s t o p } } ( S ) = \mathbb { E } \big [ ( \operatorname* { m a x } _ { j } Z _ { j } ( s _ { j } ) - q ) ^ { + } \big ] ,
$$

and, for $i \in \mathcal { T } ( S )$ with $O _ { i } = \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { j \neq i } Z _ { j } ( s _ { j } ) \}$ , which is independent of arm i’s readings,

$$
\Delta _ { i } ( S ) = c ( s _ { i } ) - \mathbb { E } \big [ ( W _ { i } ( s _ { i } ) - \operatorname* { m a x } \{ O _ { i } , G _ { i } ( s _ { i } ) \} ) ^ { + } \big ] ,
$$

which equals $\mathbb { E } [ \operatorname* { m i n } \{ ( W _ { i } - G _ { i } ) ^ { + } , ( O _ { i } - G _ { i } ) ^ { + } \} ]$ when $G _ { i } ( s _ { i } ) > - \infty$ . Then $V ^ { \star } ( S ) - h ( S ) = \Delta _ { \mathrm { s t o p } } ( S )$ $V ^ { \star } ( S ) - Q ^ { \star } ( S , i ) = \Delta _ { i } ( S ) \geq 0 .$ , and every policy π, advancing arm $i _ { b }$ at its $\begin{array} { r } { b \ – t h \ s t e p , } \end{array}$ has

$$
V ^ { \star } ( S ) - V _ { \pi } ( S ) = \mathbb { E } _ { \theta , \pi } \Big [ \sum _ { b < \tau _ { \pi } } \Delta _ { i _ { b } } ( S _ { b } ) + \Delta _ { \mathrm { s t o p } } ( S _ { \tau _ { \pi } } ) \ \Big | \ S \Big ] .
$$

When $G _ { i } = - \infty , W _ { i } = - \infty$ and the first formula gives $\Delta _ { i } = c ( s _ { i } )$ , since $O _ { i } \geq q$ is finite. The expression involving $W _ { i } - G _ { i }$ is used only for finite indices.

Proof. With V as in the proof of Theorem 1, $V ^ { \star } + K = \mathcal { V }$ and $h = q - K$ , so ${ \cal V } ^ { \star } ( S ) - h ( S ) =$ $\mathbb { E } [ \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { j } Z _ { j } \} ] - q = \Delta _ { \mathrm { s t o p } } ( S )$ . Condition on the next actual reading of i and apply the tower property as in Theorem 1; the added cost is $c ( s _ { i } )$ . This gives $Q ^ { \star } ( S , i ) + K + c ( s _ { i } ) = \mathbb { E } [ \operatorname* { m a x } \{ O _ { i } , W _ { i } \} ]$ while $V ^ { \star } ( S ) + K = \mathbb { E } [ \operatorname* { m a x } \{ O _ { i } , \operatorname* { m i n } \{ G _ { i } , W _ { i } \} \} ]$ , and max $\{ O , W \} - \operatorname* { m a x } \{ O , \operatorname* { m i n } \{ G , W \} \} = ( W -$ $\operatorname* { m a x } \{ O , G \} ) ^ { + } \ \mathrm { g i v e s } \ V ^ { \star } ( S ) - Q ^ { \star } ( S , i ) = \Delta _ { i } ( S )$ , nonnegative by the Bellman equation. The advance reveals only the next reading, not the full-path variable $W _ { i }$ . When $G _ { i }$ is finit $\mathrm { e } , c ( s _ { i } ) = \mathbb { E } [ ( W _ { i } - G _ { i } ) ^ { + } ]$ by Lemma 1, and $( W - G ) ^ { + } - ( W - \operatorname { m a x } \{ O , G \} ) ^ { + } = \operatorname * { m i n } \{ ( W - G ) ^ { + } , ( O - G ) ^ { + } \}$ pointwise. The policy identity is the performance-diference telescoping of these one-step losses along π, given in Appendix A.3. □

The theorem is the performance-diference identity of dynamic programming with both of its terms exact. For finite $G _ { i } , \Delta _ { i } = 0$ exactly when $O _ { i } \leq G _ { i }$ almost surely: the reservation equation and positive cost give $\mathbb { P } ( W _ { i } > G _ { i } ) > 0$ , and $W _ { i }$ is independent of $O _ { i }$ . Thus an advance costs nothing against the optimum exactly when the rest of the pool, incumbent included, almost surely cannot overtake its index; otherwise it costs the expected overshoot of the arm’s nested value above its index, capped by how far the rest of the pool can exceed that index. A stop costs the value of the covered call values above the incumbent. Every term is a functional of the arms’ covered call distributions. The identity applies to any policy, including one that advances through states at which the index policy would stop or switch. Evaluating the distributions makes each such departure measurable against the same optimum.

## 4 Comparative Statics: Rights, Costs and Resumption

The covered call values also price changes in the problem itself. In this section consider two per-arm problems $\mathcal { P }$ and ${ \mathcal { P } } ^ { \prime }$ , each satisfying Assumption 1, with the same arm path laws, so that each arm’s readings are the same random variables under both. $\operatorname { S a y }$ that $\mathcal { P } ^ { \prime }$ dominates P if $\kappa _ { i } \subseteq { \cal K } _ { i } ^ { \prime }$ and $c _ { i } ^ { \prime } ( s ) \leq c _ { i } ( s )$ for every arm and own state: more readings may be kept and no advance costs more. Corresponding states share the observed context, hence the own states $s _ { i } ,$ and have incumbents $q \leq q ^ { \prime } ;$ the state of ${ \mathcal { P } } ^ { \prime }$ corresponding to $S$ is written S<sup>¯</sup>. Primes on $W , Z , G , \Phi , \tilde { Y } , c , q , V$ and $U$ denote quantities of ${ \mathcal { P } } ^ { \prime }$ throughout this section, while $s ^ { \prime }$ keeps its meaning as the successor own state.

Lemma 2 (Pathwise monotonicity). $I f { \mathcal { P } } ^ { \prime }$ dominates ${ \mathcal { P } } _ { z }$ then at every own state s, $W ^ { \prime } ( s ) \geq W ( s )$ and $Z ^ { \prime } ( s ) \geq Z ( s )$ on every remaining path, and $G ^ { \prime } ( s ) \geq G ( s )$

Proof. Backward induction on $\nu ( s )$ ; all three are −∞ when $\nu ( s ) = 0$ . At s with $\nu ( s ) \geq 1$ , the collectable reading satisfies $\tilde { Y } ^ { \prime } \ge \tilde { Y }$ because $\kappa \subseteq \kappa ^ { \prime }$ , and $Z ^ { \prime } ( s ^ { \prime } ) \geq Z ( s ^ { \prime } )$ by the hypothesis at $s ^ { \prime } { \mathrm { . } }$ , so $W ^ { \prime } ( s ) = \operatorname* { m a x } \{ \tilde { Y } ^ { \prime } , Z ^ { \prime } ( s ^ { \prime } ) \} \geq W ( s )$ . By Lemma 1, $\Phi ^ { \prime } ( s ; r ) = \mathbb { E } [ ( W ^ { \prime } ( s ) - r ) ^ { + } ] - c ^ { \prime } ( s ) \ge \mathbb { E } [ ( W ( s ) -$ $r ) ^ { + } ] - c ( s ) = \Phi ( s ; r )$ for every $r ,$ so the positivity set of $\Phi ^ { \prime }$ contains that of Φ and Proposition 1 gives $G ^ { \prime } ( s ) \geq G ( s )$ . Then $Z ^ { \prime } ( s ) = \operatorname* { m i n } \{ G ^ { \prime } ( s ) , W ^ { \prime } ( s ) \} \geq \operatorname* { m i n } \{ G ( s ) , W ( s ) \} = Z ( s )$ □

Proposition 2 (Exact comparative statics). Let ${ \mathcal { P } } ^ { \prime }$ dominate $\mathcal { P }$ . At corresponding states $S$ and $\bar { S } _ { i }$ with $U = V ^ { \star } + K$ as in Section ${ \mathcal { Q } } ,$

$$
\begin{array} { r } { U ^ { \prime } ( \bar { S } ) - U ( S ) = \mathbb { E } \Big [ \big ( \operatorname* { m a x } \{ q ^ { \prime } , \operatorname* { m a x } _ { i } Z _ { i } ^ { \prime } ( s _ { i } ) \} - \operatorname* { m a x } \{ q , \operatorname* { m a x } Z _ { i } ( s _ { i } ) \} \big ) ^ { + } \Big ] \geq 0 , } \end{array}
$$

the expectation coupling $Z _ { i }$ and $Z _ { i } ^ { \prime }$ on arm $i \mathit { \Pi } _ { \mathit { s } }$ remaining path under $\mathsf { P } _ { S }$ . At a common initial state the value of the additional rights or lower costs is $\begin{array} { r } { \mathbb { E } [ ( \operatorname* { m a x } _ { i } Z _ { i } ^ { \prime } - \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { i } Z _ { i } \} ) ^ { + } ] } \end{array}$ , which vanishes $i f$ and only if max $: Z _ { i } ^ { \prime } \leq \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { i } Z _ { i } \}$ almost surely.

Proof. Theorem 1 applies to both problems on the common product space, giving $U ^ { \prime } ( { \bar { S } } ) - U ( S ) =$ $\mathbb { E } [ \operatorname* { m a x } \{ q ^ { \prime } , \operatorname* { m a x } _ { i } Z _ { i } ^ { \prime } \} - \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { i } Z _ { i } \} ]$ . By Lemma 2 and $q ^ { \prime } \geq q$ the integrand is nonnegative on every path, so it equals its positive part. □

The identity follows from pathwise monotonicity and the value representation. Its use is to price additional collection rights or lower costs from coupled arm-wise covered-call values, without solving either joint Bellman problem. It covers enlarging a collectable set, removing a history-dependent surcharge, and lowering any own-history cost. For one-shot boxes, adding a box or lowering a cost raises the stopping frontier (Bao et al., 2026, Proposition 2). The identity does not cover changes to the readings’ law, such as adding an observation stage or changing signal accuracy. The next result prices the opposite error: planning as if fewer readings could be kept.

Proposition 3 (Planning with a smaller collectable set). Let ${ \mathcal { P } } ^ { \prime }$ dominate $\mathcal { P }$ with the same costs, so that only the collectable sets $d i f f e r$ . Let $\pi _ { \boldsymbol { K } }$ be the policy for ${ \mathcal { P } } ^ { \prime }$ that stops when max $\dot { \mathbf { \zeta } } _ { i } G _ { i } ( s _ { i } ) \leq q ^ { \prime }$ and otherwise advances an arm maximizing the P-index $G _ { i } ( s _ { i } )$ , where $q ^ { \prime }$ is the incumbent of ${ \mathcal { P } } ^ { \prime }$ Then at corresponding states

$$
V ^ { \star } ( S ) \leq V _ { \pi _ { \mathcal { K } } } ( \bar { S } ) \leq V ^ { \star \prime } ( \bar { S } ) .
$$

Its loss against the optimum of ${ \mathcal { P } } ^ { \prime }$ is therefore at most the value of the additional rights; both ends are attained in the study of Appendix $\mathrm { E . 3 }$

Appendix C proves this and the next result. Finally, the loss decomposition prices the freedom to return to a paused arm. Fix a starting state $S _ { 0 }$ . A policy resumes if it returns to an arm that it has advanced since $S _ { 0 }$ after advancing a diferent arm; a no-return policy never resumes, and $V _ { \mathrm { N R } } ( S _ { 0 } )$ is its optimal value. This difers from search without recall, or committed search (Gupta et al., 2019, Section 5), which forbids returning to an earlier prize: a no-return policy retains every collectable reading.

On a path where the index resumes, a no-return policy must depart from its trajectory by that advance. We lower-bound the resulting loss by the smallest expected Bellman penalty at this first departure; later losses are nonnegative.

To preserve return status in the bound, augment a descendant history with $\widehat { \cal S } = ( { \cal S } , \ell , D )$ where ℓ is the last arm advanced since $S _ { \mathrm { 0 } } ~ ( \ell = \mathrm { 0 } \mathrm { \ i n i t i a l l y ) }$ and D is the set of arms advanced and subsequently left since $S _ { 0 }$ . An advance of i sets $\ell ^ { + } = i$ and $D ^ { + } = D \cup \{ \ell \} \ { \mathrm { ~ i f ~ } } \ell \ \notin \ \{ 0 , i \}$ and $D ^ { + } = D$ otherwise. It is a resumption exactly when $i \in D$ and $i \neq \ell .$ . This bookkeeping is retained along the whole run, including when the last arm is exhausted. For the index policy with its fixed tie order, let $\delta ( S )$ be the smallest loss from Theorem 2 of an action other than $\pi _ { \mathrm { { I X } } } ( S )$ with min $\varnothing = + \infty$ . Define ${ \widehat \rho } ( { \widehat S } )$ by finite backward recursion: it is zero if the index policy has zero probability of a future resumption from this augmented history, equals $\delta ( S )$ if its next action resumes, and otherwise

$$
\widehat { \rho } ( \widehat { S } ) = \operatorname* { m i n } \{ \delta ( S ) , \mathbb { E } [ \widehat { \rho } ( \widehat { S } _ { 1 } ) \mid \widehat { S } ] \} , \qquad \rho ( S _ { 0 } ) = \widehat { \rho } ( S _ { 0 } , 0 , \emptyset ) ,
$$

where $\widehat { S } _ { 1 }$ is the successor under the index policy with the bookkeeping above. The minimum compares departing now, at penalty $\delta ( S )$ , with following the index one more step and facing a later departure. At the first resumption, departure is compulsory. This first-departure calculation gives a lower bound on the total resumption gap.

Proposition 4 (Value of resumption). Under the hypotheses of Theorem $\boldsymbol { { 1 } } \boldsymbol { { : } }$

(i) $V ^ { \star } ( S _ { 0 } ) - V _ { \mathrm { N R } } ( S _ { 0 } ) \geq \rho ( S _ { 0 } ) \geq 0$ , and $V _ { \mathrm { N R } } ( S _ { 0 } ) = V ^ { \star } ( S _ { 0 } )$ when $\pi _ { \mathrm { I X } }$ almost surely never resumes from $S _ { 0 }$

(ii) If at every state at which $\pi _ { \mathrm { I X } }$ advances from $S _ { 0 }$ , before $o r$ at its first resumption, the indexmaximal arm is unique, then $\rho ( S _ { 0 } ) > 0$ exactly when $\pi _ { \mathrm { I X } }$ resumes from $S _ { 0 }$ with positive probability.

(iii) If every reading is a deterministic function of the arm’s own state, π<sub>IX</sub> never resumes under any tie order, so $V _ { \mathrm { N R } } = V ^ { \star }$

With deterministic readings an advance either raises the arm’s index by its cost or lifts the incumbent above its previous index, which rules out resumption. Stochastic observations can make resumption valuable by resolving uncertainty in collectable rewards, even without changing any unobserved reward’s law. Example 2 in Appendix C gives a two-arm construction and calculates the values with and without resumption.

Section 6 and Appendix E.3 evaluate all three results exactly on the staging grid.

## 5 Information Sharing Across Arms

Sections 3 and 4 assumed a per-arm planning law. Under the general law of Assumption 1, define $G _ { i } ( S ) = G _ { i } ^ { ( \nu _ { i } ) } ( S )$ by advancing only i from the current history; other arms’ observations stay fixed along this hypothetical run, so the local recursion still applies, and the index policy recomputes these indices after each actual observation. Shared observations can change untouched arms’ laws; freezing hyperparameters does not restore independence.

Proposition 5 (No universally optimal own-arm-law policy). There are two instances with the same incumbent, costs, available stages and separate laws of each arm’s remaining readings, one independent and one shared, with diferent uniquely optimal first actions. Consequently no single rule whose inputs consist only of those quantities is optimal for all joint planning laws, including randomized rules.

The construction couples a noncollectable signal in one arm to another arm’s collectable reward while preserving their separate laws; for one-shot boxes, Bao et al. (2026) show similarly that fixed box indices cannot be optimal under interdependence. A positive standalone run value still certifies that stopping is suboptimal, and under a forward-closure condition the index stopping region is exactly optimal (Proposition 8, Appendix D).

Collection reshapes acquisition under a shared state. Sharing also interacts with designated collection: the same joint law of readings can yield diferent acquisition orders under diferent collection rights.

Proposition 6 (Collection-induced acquisition switching). There is a two-arm family with a common binary latent state of prior p, a nondegenerate closed interval of terminal costs, and thresholds $0 < \tau _ { 1 } < \tau _ { 2 } ( c ) < \tau _ { 3 } ( c ) < 1$ depending continuously on the cost c in that interval, with the following property. If arm B’s intermediate reading is collectable, whether or not A’s is, the uniquely optimal first action advances A for $p \in ( 0 , \tau _ { 1 } ) \cup ( \tau _ { 2 } ( c ) , \tau _ { 3 } ( c ) )$ and advances B for $p \in ( \tau _ { 1 } , \tau _ { 2 } ( c ) ) \cup ( \tau _ { 3 } ( c ) , 1 )$ If it is not collectable, whether or not A’s is, advancing A is the uniquely optimal first action for every $p \in ( 0 , 1 )$

In the construction, arm A has the higher terminal success probability in one latent state, while B ofers the larger usable intermediate score. Keeping that score adds an option to avoid a terminal inspection; its value competes with terminal reward and information through the subsequent decisions. The alternating regions show that first inspection need not vary monotonically with the prior.

Only the collection rights difer across the four cases, so the set of priors at which A is inspected first is disconnected exactly when B’s intermediate score can be kept. A prior-dependent change of first inspection already occurs in one-shot interdependent search (Bao et al., 2026, Example 1). The family, its thresholds, a fixed-cost first-action table and the proof are in Appendix D.2; the proof is an exact computer-assisted Bellman certificate over the field of rational functions of $c ,$ uniform over the cost interval.

Robustness under a coherent shared law. A quantitative guarantee follows by specializing approximate-planning bounds (Subramanian et al., 2022, Theorem 9 and Remark 11) to the independent value representation. Restrict further: all readings are coordinates of complete arm paths with one fixed joint law P, predictions are its conditional laws, and each arm’s next cost depends only on its own observed history. Dependence within and across paths is arbitrary; cross-arm cost changes are excluded. Let $\mathsf { P } _ { j , S }$ be the posterior marginal law of arm $j ^ { \prime } \mathrm { s }$ remaining path at $S , Z _ { j } ^ { S }$ its covered call under that law, and $v _ { j , S } ( r ) = \mathbb { E } _ { \mathsf { P } _ { j , S } } \operatorname* { m a x } \{ r , Z _ { j } ^ { S } \}$ its local keep-or-run value. For the actual successor $S ^ { \prime } = F ( S , i , Y )$ , the discrepancy

$$
\varepsilon _ { i } ( S ) = \mathbb { E } \left[ \sum _ { \substack { j \neq i } } \operatorname* { s u p } _ { r \in [ q ( S ^ { \prime } ) , y _ { \mathrm { m a x } } ] } | v _ { j , S ^ { \prime } } ( r ) - v _ { j , S } ( r ) | \middle | S , i \right]
$$

measures how much an observation changes untouched arms’ local values; each supremum is the stop-loss metric between the two capped variables clipped at the new incumbent (M¨uller, 1997, Section 5, Example C).

Proposition 7 (Robustness to bounded sharing). Under Assumption 1 and the coherent-law/localcost restriction above, let $H = L ( S _ { 0 } ) . ~ I f \varepsilon _ { i } ( S ) \leq \bar { \varepsilon }$ almost surely along every admissible policy from $S _ { 0 }$ , then the policy $\pi _ { \mathrm { I X } }$ , recomputed from the true posterior marginal laws after every observation, satisfies

$$
0 \leq V ^ { \star } ( S _ { 0 } ) - V _ { \pi _ { \mathrm { I X } } } ( S _ { 0 } ) \leq 2 H \bar { \varepsilon } .
$$

The bound covers premature stops and incorrect advances and needs no separation between indices; under independence $\varepsilon _ { i } = 0$ . The condition concerns every future feasible history, including states at which the index would already have stopped, so root covariance alone does not sufice, and it does not control posterior misspecification or numerical index error. Appendix D.3 proves the bound through an explicit signed sharing correction and gives a path-total-variation envelope.

Sharing through a learned common component. In Bayesian optimization, sharing typically flows through a component common to all candidates, such as a task-level parameter of a learning-curve model learned from previous tasks and from the current run. Suppose arm paths are conditionally independent given a latent $\Theta$ on a standard Borel space, with posterior $\mu _ { S }$ at state $S ,$ so that $\begin{array} { r } { \mathsf { P } _ { j , S } = \int \mathsf { P } _ { j } ^ { \vartheta } ( \cdot \mid s _ { j } ) \mu _ { S } ( d \vartheta ) } \end{array}$ for measurable component path kernels $\mathsf { P } _ { j } ^ { \vartheta }$ . Let

$$
\Lambda _ { j } ( S ) = \operatorname* { s u p } _ { \vartheta , \vartheta ^ { \prime } } \mathrm { T V } \big ( \mathrm { P } _ { j } ^ { \vartheta } ( \cdot \cdot \mid s _ { j } ) , \mathrm { P } _ { j } ^ { \vartheta ^ { \prime } } ( \cdot \mid s _ { j } ) \big ) , \qquad \eta _ { i } ( S ) = \mathbb { E } _ { \vartheta \sim \mu _ { S } } \mathrm { T V } \big ( \mathrm { P } _ { i } ^ { \vartheta } ( Y \in \cdot \mid s _ { i } ) , \mathrm { P } _ { i , S } ( Y \in \cdot ) \big ) ,
$$

be the sensitivity of arm $j ^ { \prime } \mathfrak { s }$ s remaining path to the component and the posterior dispersion of arm $i \mathrm { \ ' } _ { \mathrm { S } }$ next reading, with $\mathrm { T V } ( \mu , \nu ) = \operatorname* { s u p } _ { A } | \mu ( A ) - \nu ( A ) |$ , and let $B _ { j } ( S ) = y _ { \mathrm { { m a x } } } - y _ { \mathrm { { m i n } } } + \nu _ { j } ( S ) c _ { \mathrm { { m a x } } }$ if arm $j$ has a collectable stage left and $B _ { j } ( S ) = 0$ otherwise. Assume the sensitivities $\Lambda _ { j } ( S )$ are measurable; this is automatic for a finite component space.

Corollary 2 (Learned common component). Under Assumption 1 and the coherent-law and armlocal-cost restriction of Proposition 7, with arm paths conditionally independent given Θ,

$$
\varepsilon _ { i } ( S ) \leq \eta _ { i } ( S ) \sum _ { j \neq i } B _ { j } ( S ) \Lambda _ { j } ( S ) , \qquad s o \qquad 0 \leq V ^ { \star } ( S _ { 0 } ) - V _ { \pi \mathrm { I x } } ( S _ { 0 } ) \leq 2 H \operatorname* { s u p } _ { S , i } \eta _ { i } ( S ) \sum _ { j \neq i } B _ { j } ( S ) \Lambda _ { j } ( S )
$$

over feasible states and advances. Moreover $\eta _ { i } ( S ) ~ \le ~ \mathbb { E } _ { \vartheta , \vartheta ^ { \prime } \sim \mu _ { S } \otimes \mu _ { S } } \mathrm { T V } ( \mathsf { P } _ { i } ^ { \vartheta } ( Y ~ \in ~ \cdot ~ \mid ~ s _ { i } ) , \mathsf { P } _ { i } ^ { \vartheta ^ { \prime } } ( Y ~ \in ~ \mathfrak { c } _ { \delta } ) ) ~ .$ $\cdot \mid s _ { i } ) )$ , the expected disagreement of two posterior draws. For binary Θ with $p = \mu _ { S } ( \Theta = 1 )$ $\eta _ { i } ( S ) ~ = ~ 2 p ( 1 - p ) \mathrm { T V } ( \mathsf { P } _ { i } ^ { 0 } ( Y \in \cdot ~ | ~ s _ { i } ) , \mathsf { P } _ { i } ^ { 1 } ( Y \in \cdot ~ | ~ s _ { i } ) )$ and $\Lambda _ { j } ( S )$ is the separation of the two component path laws.

Sensitivity measures how far a candidate’s law can move with the component, dispersion how much one more reading can move the belief about it. For fixed horizon, reading and cost bounds, a vanishing uniform envelope on the right-hand side gives a vanishing regret bound. For a finite component space, posterior mass tending to one on one component makes the local dispersion vanish. With continuous components, posterior concentration needs additional regularity, such as total-variation continuity of the next-reading kernels; weak concentration alone does not sufice. In either case the regret guarantee still requires control over all feasible future histories, including rare ones whose beliefs may be less concentrated. Appendix D.3 gives a noiseless counterexample to automatic dispersion decay. Mixtures of product laws are also the structure under which correlated one-shot search admits eficient approximation (Chawla et al., 2023). Appendix D.3 gives the proof; Section 6 and Appendix E.4 report checks of the signed correction and bounds; Appendix E.5 contrasts a positive-regret example with a shared nuisance whose local-value envelope vanishes, and reports a sweep in which only the prior on a binary component changes and the uniform envelopes decrease.

## 6 Numerical Verification

Exact finite-law checks compare the representation and losses with direct joint Bellman optimization, using rational arithmetic throughout (Table 1). On independent binary-tree instances, the optimal value, maximal-index actions and every stop/advance loss agree. On the staging grid, each diagnostic policy’s utility loss equals its accumulated action losses. Across dominated problem pairs, value diferences equal the covered-call overshoot; both ends of the smaller-collectable-set bound are attained. A separate no-return recursion confirms that $\rho$ never exceeds the resumption gap, including cases of equality (Table 2).

Under shared laws, the signed residual, policy telescoping and envelope bounds hold at every feasible prefix, including histories the index never visits. A constructed example has positive index regret, while a shared nuisance has zero local-value envelope despite nonzero path-TV shifts (Table 4). Across the tested priors in a binary-component sweep, the largest uniform call and path-TV bounds decrease as the prior moves toward zero, and regret is zero at the four tested priors below $1 / 2$ (Table 3). This illustrates the uniform-envelope condition in Corollary 2 for these fixed finite models. These checks directly support the sharing guarantee and its value-sensitive discrepancy. Appendix E places each study’s law beside its checks and results; Algorithm 1 gives the policy and backward index computation; Appendix F gives finite-tree implementation details and complexity. The checks supplement the proofs under known laws; they establish neither learned-surrogate accuracy nor practical optimization performance.

## 7 Conclusion

The covered-call representation unifies designated collection, history-dependent prediction and ownstate costs. It gives the independent value, optimal policy and exact action losses, recovers established special cases, prices collection rights and costs, bounds the value of resumption, and quantifies how posterior updates under sharing perturb those guarantees through changes in local continuation values. A common-component envelope relates those changes to predictive total-variation dispersion; its decay must be controlled over all feasible future histories.

Exact optimality requires finite independent arm processes, arm-local availability and costs, and readings that follow the planning law; general information sharing invalidates the proof, and the forward-closure stopping condition is not established for shared surrogates. The sharing bound requires a coherent joint path law, arm-local costs, true posterior marginals and an envelope over every future feasible history; it is conservative in the exact examples and certifies neither posterior accuracy nor numerical index error. Weak posterior concentration alone does not imply that this envelope vanishes. An unbounded Gaussian predictive law also requires an extension of the stated bounded-law assumptions. Continuous reading laws require integration or discretization whose policy error is not bounded here, observation trees are exponential in depth, and the objective is the best observed collectable score less cost, not latent performance under noisy readings. The studies use two arms, few outcomes, short ladders and known laws; they measure neither learnedsurrogate accuracy nor robustness to a misspecified law and establish no empirical superiority for hyperparameter optimization. Scaling, misspecification, continuous candidate optimization and comparisons with published methods on their own surrogates remain open for the staged index; Vu et al. (2026) report such comparisons for the one-step rule.

## References

Ali Aouad, Jingwei Ji, and Yaron Shaposhnik. The Pandora’s box problem with sequential inspections. arXiv preprint arXiv:2507.07508, 2025. URL https://arxiv.org/abs/2507.07508.

Nikhil Bansal, Zhiyi Huang, and Zixuan Zhu. Optimal 4-approximation for the correlated Pandora’s problem, 2025. URL https://arxiv.org/abs/2509.17029.

Renjie Bao, Jingmin Huang, Sanxi Li, and Jun Yu. Pandora’s problem with interdependent values. Working paper, September 2026. URL https://renjie-bao.github.io/uploads/ Interdependent\_Search.pdf.

Shant Boodaghians, Federico Fusco, Philip Lazos, and Stefano Leonardi. Pandora’s box problem with order constraints. Mathematics of Operations Research, 48(1):498–519, 2023.

Robin Bowers and Bo Waggoner. Matching with nested and bundled Pandora boxes. In Web and Internet Economics, volume 15534 of Lecture Notes in Computer Science, pages 61–83. Springer, 2026. doi: 10.1007/978-3-032-08560-3 4. URL https://arxiv.org/abs/2406.08711v2.

Robin Bowers, Elias Lindgren, and Bo Waggoner. Combinatorial Markov search. In Proceedings of the Annual ACM Symposium on Theory of Computing, pages 2290–2301. ACM, 2026. doi: 10.1145/3798129.3800930. URL https://arxiv.org/abs/2502.08976v2.

Shuchi Chawla, Evangelia Gergatsouli, Yifeng Teng, Christos Tzamos, and Ruimin Zhang. Pandora’s box with correlations: Learning and approximation, 2020. URL https://arxiv.org/ abs/1911.01632.

Shuchi Chawla, Evangelia Gergatsouli, Jeremy McMahan, and Christos Tzamos. Approximating Pandora’s box with correlations, 2023. URL https://arxiv.org/abs/2108.12976.

Evangelia Gergatsouli and Christos Tzamos. Weitzman’s rule for Pandora’s box with correlations, 2023. URL https://arxiv.org/abs/2301.13534.

K. D. Glazebrook. Stoppable families of alternative bandit processes. Journal of Applied Probability, 16(4):843–854, 1979.

Kevin D. Glazebrook. On a suficient condition for superprocesses due to Whittle. Journal of Applied Probability, 19(1):99–110, 1982.

Anupam Gupta, Haotian Jiang, Ziv Scully, and Sahil Singla. The Markovian price of information. In Integer Programming and Combinatorial Optimization, volume 11480 of Lecture Notes in Computer Science, pages 233–246. Springer, 2019. doi: 10.1007/978-3-030-17953-3 18. URL https://arxiv.org/abs/1902.07856.

Kevin Jamieson and Ameet Talwalkar. Non-stochastic best arm identification and hyperparameter optimization. In Proceedings of the International Conference on Artificial Intelligence and Statistics, volume 51 of Proceedings of Machine Learning Research, pages 240–248, 2016.

Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast Bayesian optimization of machine learning hyperparameters on large datasets. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 528–536, 2017.

Robert D. Kleinberg, Bo Waggoner, and E. Glen Weyl. Descending price optimally coordinates search. In Proceedings of the ACM Conference on Economics and Computation, pages 23–24. ACM, 2016. doi: 10.1145/2940716.2940760. URL https://arxiv.org/abs/1603.07682v3.

Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18(185):1–52, 2018.

Alfred M¨uller. Integral probability metrics and their generating classes of functions. Advances in Applied Probability, 29(2):429–443, 1997. doi: 10.2307/1428011.

Ziv Scully and Alexander Terenin. The Gittins index: A design principle for decision-making under uncertainty. arXiv preprint arXiv:2506.10872, 2025.

Jayakumar Subramanian, Amit Sinha, Raihan Seraj, and Aditya Mahajan. Approximate information state for approximate planning and reinforcement learning in partially observed systems. Journal of Machine Learning Research, 23(12):1–83, 2022. URL https://jmlr.org/papers/ v23/20-1165.html.

Kevin Swersky, Jasper Snoek, and Ryan P. Adams. Freeze-Thaw Bayesian optimization. arXiv:1406.3896, 2014.

Long Vu, Jongseong Brad Choi, Binoy Ravindran, and Zhoulai Fu. Cost-aware stopping for freezethaw Bayesian optimization: When does one-step lookahead sufice?, 2026. URL https://arxiv. org/abs/2610.42085.

Martin L. Weitzman. Optimal search for the best alternative. Econometrica, 47(3):641–654, 1979.

Peter Whittle. Multi-armed bandits and the Gittins index. Journal of the Royal Statistical Society: Series B (Methodological), 42(2):143–149, 1980.

Jian Wu, Saul Toscano-Palmerin, Peter I. Frazier, and Andrew Gordon Wilson. Practical multifidelity Bayesian optimization for hyperparameter tuning. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, volume 115 of Proceedings of Machine Learning Research, pages 788–798, 2020.

Qian Xie, Raul Astudillo, Peter I. Frazier, Ziv Scully, and Alexander Terenin. Cost-aware Bayesian optimization via the Pandora’s Box Gittins index. In Advances in Neural Information Processing Systems, volume 37, pages 115523–115562, 2024. doi: 10.52202/079017-3669.

Qian Xie, Linda Cai, Alexander Terenin, Peter I. Frazier, and Ziv Scully. Cost-aware stopping for Bayesian optimization. In Proceedings of the International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 139218–139261, 2026. URL https://icml.cc/virtual/2026/poster/61659.

Yuanyuan Yang, Ruimin Zhang, Jamie Morgenstern, and Haifeng Xu. Markovian Pandora’s box, 2025. URL https://arxiv.org/abs/2502.00228.

## A Notation and Local Stopping Details

## A.1 Notation and literature conventions

An arm’s own state $s _ { i }$ consists of its stage and observation history; S records the full search history. The remaining depth is $\nu _ { i } = T _ { i } - t _ { i } , \mathcal { K } _ { i }$ is the collectable set, and $K ( S )$ is sunk cost. The forced-run gain $\Phi ( s ; r )$ excludes immediate stopping; the keep-or-run value is $r + \Phi ( s ; r ) ^ { + }$ . At exhaustion that value remains r. A sufix without a collectable stage has index and $\mathrm { c a p \mathrm { \ : - } \infty } ;$ diferences such as $W - G$ are used only for finite G. The full-path variables W, Z are not disclosed by one advance.

For terminal-only collection, $G _ { i } , Z _ { i }$ correspond to the nested strike and capped values $\sigma _ { i } ^ { ( \ell ) } , \kappa _ { i } ^ { ( \ell ) }$ of Bowers and Waggoner (2026, Definition 4), with $\ell = t _ { i } + 1$ , and $W _ { i }$ to their next capped value. Our forced gain difers from the total equivalent-reward function called Φ by Yang et al. (2025, Definition 3.2).

The per-arm path law is $\mathsf { P } _ { i } ^ { s _ { i } }$ . Under sharing, P is the coherent joint law, $\mathsf { P } _ { j , S }$ a posterior arm marginal, and ${ \sf Q } _ { S } = \otimes _ { i } { \sf P } _ { j , S }$ the synthetic product reference. The action value $\overset { \cdot } { Q _ { \pi } } ( S , i ) = \mathbb { E } [ V _ { \pi } ( S ^ { \prime } ) ]$ includes the advance cost through $K ( S ^ { \prime } ) ; U _ { \pi } = V _ { \pi } + K$ removes sunk costs. Expectations at a starting state use its specified kernel and count further advances from zero, including for exact real-valued histories of probability zero. Conditional on that state, all remaining rewards and costs are bounded.

## A.2 Stopping-rule attainment

Lemma 3 (Finite-horizon arm-run recursion). Under Assumption 1, the forced-run value $( S , r ) \mapsto$ $\Phi _ { i } ^ { ( k ) } ( S ; r )$ is jointly Borel measurable on states at which i is available. Its supremum is attained by a stopping rule chosen measurably in $( S , r )$ . This holds for an arbitrary fixed planning kernel, including a shared one.

Proof. For an auxiliary outside value $r ,$ set $U _ { i , 0 } ( S ; r ) = 0$ . For $k \geq 1$ and an available i, define backwards

$$
\begin{array}{c} \begin{array} { l } { { } } \\ { { } } \\ { { } } \\ { { } } \\ { { } } \\ { { } { } } \\ { { } { } } \end{array} { l } _ { i , k } ( S ; r ) = - c ( S , i ) + \int \left[ ( \tilde { y } - r ) ^ { + } + U _ { i , k - 1 } ( F ( S , i , y ) ; \operatorname* { m a x } \{ r , \tilde { y } \} ) \right] p _ { \theta } ( d y \mid S , i ) ,  \\ { { } } \\ { { } } \\ { { } { } U _ { i , k } ( S ; r ) = \operatorname* { m a x } \{ 0 , { \cal B } _ { i , k } ( S ; r ) \} , } \end{array}
$$

where $\tilde { y } = y$ if this advance is collectable and $- \infty$ otherwise. Put $U _ { i , k } = \mathrm { 0 }$ at exhausted arms. All functions are jointly Borel by induction: the successor, cost and kernel are measurable, integration preserves measurability, and the action maximum is over two choices. For each finite $r$ the integrands and values are bounded by the reading bounds and the finite remaining cost bound.

Let m readings have been seen along a run and let $r _ { m } = \operatorname* { m a x } \{ r , M _ { m } \}$ . For any additional future maximum $M ^ { \prime } .$ , the identity

$$
( \operatorname* { m a x } \{ M _ { m } , M ^ { \prime } \} - r ) ^ { + } - ( M _ { m } - r ) ^ { + } = ( M ^ { \prime } - r _ { m } ) ^ { + }
$$

shows that the earned improvement can be separated from future improvement. The full planning state in the recursion is still the actual $S ^ { ( m ) }$ ; replacing r by $r _ { m }$ does not replace its incumbent, refit its law, or change its costs. Thus $B _ { i , k }$ is the value of forcing one advance followed by an optimal continue-or-stop decision, and $U _ { i , k }$ includes stopping immediately. Backward induction gives $B _ { i , k } ( S ; r ) = \Phi _ { i } ^ { ( k ) } ( S ; r )$ and $U _ { i , k } ( S ; r ) = \Phi _ { i } ^ { ( k ) } ( S ; r ) ^ { + }$ whenever i is available. After the forced first advance, continue precisely where the remaining $B$ is positive and stop on a tie or at the cap. These are measurable decisions and give a stopping time bounded by min $\{ k , \nu _ { i } \}$ attaining the stated value. □

For a per-arm law, the recursion depends on $S$ only through $s _ { i }$ and the auxiliary value. This justifies the local Bellman equation in Section 3.1 without assuming that the latest scalar reading is Markov. Measurability of the reservation indices follows from their rational-threshold representation in Section $2 ;$ no selection over uncountably many stopping rules or candidate actions is needed.

Proof of Proposition 1. The finite-horizon recursion in Lemma 3 gives a jointly measurable value and an attaining stopping rule. In particular, the supremum here does not require an unproved measurable selection from a family of stopping times. For a fixed run, its positive-part payof is nonincreasing and 1-Lipschitz in $r ;$ these properties pass to expectation and supremum. The value is finite, bounded above by $( y _ { \operatorname* { m a x } } - r ) ^ { + } - c ( S , i )$ and below by the one-advance run. Without a collectable stage within the cap every run has strictly negative value, so $G = - \infty$ . Otherwise the committed run to the next collectable stage has positive value for $r < y _ { \mathrm { m i n } } - \mu _ { i } c _ { \mathrm { m a x } }$ , while the upper bound is nonpositive for $r \geq y _ { \operatorname* { m a x } } - c ( S , i )$ . The positivity set is therefore a nonempty lower interval. Continuity excludes a positive endpoint and forces the value there to be zero, giving the strict threshold statement. Increasing the cap enlarges the set of runs and cannot decrease the index. □

## A.3 Proof of the policy identity in Theorem 2

Proof. Put $H = L ( S )$ and $\tau = \tau _ { \pi } \leq H$ . Let $\mathcal { F } _ { b }$ contain the observed history and any independent policy randomness used to choose the next action. On $b < \tau .$ , the planning kernel gives

$$
\mathbb { E } [ V ^ { \star } ( S _ { b + 1 } ) \mid { \mathcal F } _ { b } ] = Q ^ { \star } ( S _ { b } , i _ { b } ) = V ^ { \star } ( S _ { b } ) - \Delta _ { i _ { b } } ( S _ { b } ) .
$$

The event $\{ b < \tau \}$ is $\mathcal { F } _ { b } .$ -measurable. Freeze the process after it stops and sum this identity over the deterministic range $b = 0 , \ldots , H - 1$ , with indicators $\mathbf { 1 } \{ b < \tau \}$ . The tower property and telescoping give

$$
V ^ { \star } ( S ) - \mathbb { E } [ V ^ { \star } ( S _ { \tau } ) ] = \mathbb { E } \left[ \sum _ { b < \tau } \Delta _ { i _ { b } } ( S _ { b } ) \right] .
$$

Finally $V ^ { \star } ( S _ { \tau } ) = h ( S _ { \tau } ) + \Delta _ { \mathrm { s t o p } } ( S _ { \tau } )$ and $V _ { \pi } ( S ) = \mathbb { E } [ h ( S _ { \tau } ) ]$ . Every sum is finite and every term bounded conditional on the starting state, so no optional-stopping or interchange-of-infinite-limits assumption is used. □

## A.4 Continuous intermediate collection

Example 1 (Continuous, dependent readings with intermediate collection). Consider one arm with two collectable stages, $Y _ { 1 } = U$ and $Y _ { 2 } = 1 - U$ for U ∼ Uniform[0, 1], and costs $c _ { 1 } = 1 / 8 , c _ { 2 } = 1 / 4$ The second reading is deterministic conditional on the first, so the within-arm dependence is perfect and neither stage is an independent box. At the state after $U = u$

$$
\begin{array} { r } { G _ { 2 } ( u ) = Z _ { 2 } ( u ) = \frac { 3 } { 4 } - u , \qquad W _ { 1 } = \operatorname* { m a x } \{ U , \frac { 3 } { 4 } - U \} . } \end{array}
$$

For $r \in [ 3 / 8 , 3 / 4 ]$ , integration over the two disjoint exceedance intervals gives

$$
\begin{array} { r } { \mathbb { E } [ ( W _ { 1 } - r ) ^ { + } ] = \frac { 1 } { 2 } ( 1 - r ) ^ { 2 } + \frac { 1 } { 2 } ( \frac { 3 } { 4 } - r ) ^ { 2 } . } \end{array}
$$

The unique reservation root is therefore $G _ { 1 } = ( 7 - \sqrt { 7 } ) / 8$ , which lies in this interval. At incumbent $q = 1 / 2$ and cumulative cost $K = 0$

$$
\begin{array} { r } { \Phi _ { 1 } ( 1 / 2 ) = \frac { 5 } { 3 2 } - \frac { 1 } { 8 } = \frac { 1 } { 3 2 } , \qquad V ^ { \star } = \frac { 1 7 } { 3 2 } . } \end{array}
$$

After the first reading, the index policy continues to the second exactly when $\begin{array} { r } { \frac { 3 } { 4 } - u > \operatorname* { m a x } \{ 1 / 2 , u \} } \end{array}$ ， or $u < 1 / 4$ ; equality may be stopped. Direct integration of that policy’s reward minus both incurred costs gives the same value $1 7 / 3 2$ . This is an analytic illustration of the recursion with a continuous marginal law and an intermediate collectable reading, not a finite-quadrature optimality claim.

## B Connection to Classical Superprocess Indices

Remark 1 (The theorem as a superprocess index theorem). Adding to an arm a free keep transition that pays the running maximum of its collectable readings would encode the collectable set in the Markovian boxes of Gupta et al. (2019), but it gives the arm two actions and leaves the class of Markov chains for that of decision processes, which is why the reduction is to the superprocess theorem and not to the Markovian box. In the language of Whittle (1980) and Scully and Terenin (2025, §5.2.2), who call the setting MDP selection, a nonterminal arm at state s with best previously collectable reading M can advance at reward − $\mathbf { \nabla } \cdot c ( s )$ or keep M and terminate the selection. When none has been observed, $M = - \infty$ and keeping is unavailable. Including an outside retirement value α gives the local value

$$
r + \Phi ( s ; r ) ^ { + } , \qquad r = \operatorname * { m a x } \{ \alpha , M \} ,
$$

by Lemma 1. Here advance is compared with the better of both immediate options, not with keeping M alone: retirement can exceed M.

Whittle’s condition asks for one non-retirement action, chosen independently of $\alpha ,$ such that this action or retirement is optimal for every α. The choice is

<table><tr><td>Condition at the arm state Retained action Retirement optimal for</td><td></td><td></td></tr><tr><td> $M > - \infty , M \geq G ( s )$ </td><td>keep</td><td> $\alpha \geq M$ </td></tr><tr><td> $M < G ( s )$ </td><td>advance</td><td> $\alpha \geq G ( s )$ </td></tr></table>

Below the displayed retirement threshold, the retained action is optimal; at the threshold both may be chosen. Its identity depends only on the fixed arm state, through M and $G ( s )$ , while the choice to retire may depend on $\alpha .$ If both M and $G ( s )$ are −∞, retirement is optimal for every finite $\alpha ;$ any available advance may be the designated action in that vacuous case. The resulting outside-option threshold is max $\{ M , G ( s ) \}$ . With the incumbent represented as a deterministic retirement option (equivalently, a dummy arm), it yields $\pi _ { \mathrm { I X } }$ in the terminating selection setting of Glazebrook (1979). This is a connection to classical theory; the direct Bellman verification in Theorem 1 already establishes optimality for the stated model. The nonobligatory-inspection hardness does not apply because an unobserved reward cannot be kept.

## C Proofs for Section 4

Proof of Proposition 3. The upper bound holds because $\pi \kappa$ is admissible in $\mathcal { P } ^ { \prime } .$ . For the lower bound, induct on $L ( S )$ over corresponding states, which share the own states and sunk cost $K$ and have $q \leq q ^ { \prime }$ . If $\pi _ { \boldsymbol { K } }$ stops at $\bar { S }$ then max<sub>i</sub> $G _ { i } ( s _ { i } ) \leq q ^ { \prime } ;$ , and since $Z _ { i } \leq G _ { i }$ pathwise, Theorem 1 gives $V ^ { \star } ( S ) + K = \mathbb { E } [ \operatorname* { m a x } \{ q ,$ , max<sub>i</sub> $Z _ { i } \} ] \leq$ max $\{ q ,$ max<sub>i</sub> $G _ { i } \} \leq q ^ { \prime } $ , so $V ^ { \star } ( S ) \leq q ^ { \prime } - K = V _ { \pi _ { K } } ( \bar { S } )$ Otherwise $\pi _ { \boldsymbol { K } }$ advances an arm i with $G _ { i } ( s _ { i } ) = \operatorname* { m a x } _ { j } G _ { j } ( s _ { j } ) > q ^ { \prime } \geq q$ , which is an optimal action in $\mathcal { P }$ at $S$ by Theorem 1. Both problems pay $c ( s _ { i } )$ and reveal the same reading $Y ;$ the successors correspond, with incumbents max $\{ q , \tilde { Y } \} \le \operatorname* { m a x } \{ q ^ { \prime } , \tilde { Y } ^ { \prime } \}$ since $\tilde { Y } \leq \tilde { Y } ^ { \prime }$ . Hence $V ^ { \star } ( S ) = \mathbb { E } [ V ^ { \star } ( S _ { 1 } ) ] \leq$ $\mathbb { E } [ V _ { \pi _ { \mathcal { K } } } ( \bar { S } _ { 1 } ) ] = V _ { \pi _ { \mathcal { K } } } ( \bar { S } )$ by the induction hypothesis applied under the integral. The successor values already include the advance cost through $K ( S _ { 1 } ) = K ( \bar { S } _ { 1 } ) = K + c ( s _ { i } )$ . Appendix E.3 exhibits cells attaining each bound. □

Proof of Proposition 4. (i) The index policy is no-return whenever it never resumes, which gives the equality case. Let π be any no-return policy; conditional on its auxiliary randomness it is a deterministic no-return policy, so we may take it deterministic. Let $\sigma$ be the first step at which $\pi \mathrm { { ^ { * } s } }$ action difers from $\pi _ { \mathrm { { I X } } } \mathrm { { ' } s }$ at the state reached, with $\sigma = \infty$ if they never difer. $\mathrm { U p }$ to $\sigma$ both policies take the same actions and observe the same readings, so the law of $( \widehat { S } _ { 0 } , \ldots , \widehat { S } _ { \sigma } )$ under $\pi$ is the law of the same stopped segment of the $\pi _ { \mathrm { I X } }$ trajectory, including the last-arm and leftarm bookkeeping from the fixed root $S _ { 0 }$ . Since $\pi$ cannot resume, $\sigma$ is at most the first time $\pi _ { \mathrm { I X } }$ resumes. Theorem 2 writes $V ^ { \star } ( S _ { 0 } ) - V _ { \pi } ( S _ { 0 } )$ as the expected sum of nonnegative losses along $\pi ,$ one of which, on $\{ \sigma < \infty \}$ , is the loss of π’s action at $S _ { \sigma } ;$ ; that action difers from $\pi _ { \mathrm { I X } } ( S _ { \sigma } )$ , so its loss is at least $\delta ( S _ { \sigma } )$ . Hence $V ^ { \star } ( S _ { 0 } ) - V _ { \pi } ( S _ { 0 } ) \geq \mathbb { E } _ { \pi _ { \mathrm { I X } } } [ \delta ( S _ { \sigma } ) \mathbf { 1 } \{ \sigma < \infty \} ]$ . The right side is the cost of a stopping time on the $\pi _ { \mathrm { I X } }$ trajectory that must stop no later than the first resumption and may run forever only on branches without resumption. The infimum of this cost over such stopping times satisfies the finite-horizon recursion defining ${ \widehat { \rho } } { \mathrm { : } }$ it is zero where no resumption can occur, equals δ at a resumption state, and otherwise is the smaller of stopping now and continuing. Thus $V ^ { \star } ( S _ { 0 } ) - V _ { \pi } ( S _ { 0 } ) \geq \widehat { \rho } ( S _ { 0 } , 0 , \emptyset ) = \rho ( S _ { 0 } )$ for every no-return π.

(ii) At a state $S$ where the index-maximal arm $i ^ { \star }$ is unique with $G _ { i ^ { \star } } > q .$ every other action has positive loss. $\mathrm { B y }$ Lemma 1, $\mathbb { E } [ ( W _ { i ^ { \star } } - G _ { i ^ { \star } } ) ^ { + } ] = c ( s _ { i ^ { \star } } ) > 0$ , so $W _ { i ^ { \star } } > G _ { i ^ { \star } }$ and hence $Z _ { i ^ { \star } } = G _ { i ^ { \star } }$ with positive probability. Then $\Delta _ { \mathrm { s t o p } } ( S ) = \mathbb { E } [ ( \operatorname* { m a x } _ { j } Z _ { j } - q ) ^ { + } ] > 0$ . For an arm $j \neq i ^ { \star }$ with finite index, ${ \cal O } _ { j } \geq Z _ { i ^ { \star } } = G _ { i ^ { \star } } > G _ { j }$ with positive probability, so $\mathbb { P } ( O _ { j } > G _ { j } ) > 0$ and $\Delta _ { j } ( S ) > 0$ by the paragraph following Theorem $2 ;$ if $G _ { j } = - \infty$ then $\Delta _ { j } ( S ) = c ( s _ { j } ) > 0$ . Hence $\delta ( S ) > 0$ at every state in the hypothesis. The augmented recursion then gives $\widehat { \rho } ( \widehat { S } ) > 0$ if and only if resumption from that augmented history has positive probability: at a resumption state $\widehat { \rho } = \delta > 0$ , on branches without resumption $\widehat { \rho } = 0$ , and elsewhere $\widehat { \rho } = \operatorname* { m i n } \{ \delta , \mathbb { E } [ \widehat { \rho } ( \widehat { S } _ { 1 } ) ] \}$ is positive exactly when $\mathbb { E } [ \widehat { \rho } ( \widehat { S } _ { 1 } ) ] > 0$ . Finite ladders make the induction terminate; evaluating at $( S _ { 0 } , 0 , \emptyset )$ proves the assertion for $\rho ( S _ { 0 } )$

(iii) Let every reading be a deterministic function of the own state. Then $W ( s )$ is a number, the reservation equation $( W ( s ) - g ) ^ { + } = c ( s )$ has the unique solution $G ( s ) = W ( s ) - c ( s )$ when a collectable stage remains, $Z ( s ) = G ( s )$ , and $W ( s ) = \mathrm { m a x } \{ \tilde { Y } ( s ) , G ( s ^ { \prime } ) \}$ with $s ^ { \prime }$ the successor. Consequently an advance from s to $s ^ { \prime }$ either has ${ \tilde { Y } } ( s ) \geq G ( s ^ { \prime } )$ , in which case the new incumbent is at least $\tilde { Y } ( s ) = G ( s ) + c ( s ) > G ( s )$ , or has $\tilde { Y } ( s ) < G ( s ^ { \prime } )$ , in which case $G ( s ^ { \prime } ) = G ( s ) + c ( s ) > G ( s )$ the first case is forced when $s ^ { \prime }$ is exhausted. Suppose π<sub>IX</sub> advances an arm $i \neq j$ at a state where arm $j$ , which it has advanced from $S _ { 0 }$ , is not exhausted; then $G _ { i } ( s _ { i } ) \geq G _ { j } ( s _ { j } )$ and $G _ { i } ( s _ { i } ) > q$ . We claim that at every later state either $q > G _ { j } ( s _ { j } )$ or some unexhausted arm k has $G _ { k } ( s _ { k } ) > G _ { j } ( s _ { j } )$ After the advance of i, the first case above gives $q ~ > ~ G _ { i } ( s _ { i } ) ~ \geq ~ G _ { j } ( s _ { j } )$ and the second gives $G _ { i } ( s _ { i } ^ { \prime } ) > G _ { i } ( s _ { i } ) \geq G _ { j } ( s _ { j } )$ . If the claim holds with $q > G _ { j } ( s _ { j } )$ it holds forever, as the incumbent never decreases and $s _ { j }$ does not change while $j$ is paused. If it holds through an arm k, the next advance, if any, is of an arm m with $G _ { m } ( s _ { m } ) \geq G _ { k } ( s _ { k } ) > G _ { j } ( s _ { j } )$ , so m $\neq j$ , and the two cases above restore the claim with $q > G _ { m } ( s _ { m } ) > G _ { j } ( s _ { j } )$ or with $k = m$ . In neither case is $j$ ever index-maximal with index above the incumbent, so it is never resumed, whatever the tie order.

Example 2 (Resumption without learning future rewards). Let $q _ { 0 } = 0$ , make every stage collectable, and charge $1 / 1 0$ per advance. Arm A first pays zero or one with equal probability, then pays $2 / 5$ deterministically; independent arm B pays zero or $3 / 5$ with equal probability in one stage. The indices are $G _ { A , 1 } = 4 / 5 , G _ { B } = 2 / 5$ , and $G _ { A , 2 } = 3 / 1 0$ . The index policy starts $A _ { i }$ , switches to B after a zero, and resumes A after a second zero. Its value is $2 3 / 4 0$ , whereas the no-return optimum is $1 1 / 2 0$ , so resumption is worth $1 / 4 0$ . No reading changes any remaining reward law: $A \mathit { ^ { \prime } s }$ index drops at its known stage transition, while the realized collectable reward determines whether to continue. The calculation below verifies both values.

Value calculations. The index policy stops after a first reward of one; after a zero from A it tries B and completes A only if B also returns zero. Its net utility is

$$
\begin{array} { r } { \frac { 1 } { 2 } ( 1 - \frac { 1 } { 1 0 } ) + \frac { 1 } { 4 } ( \frac { 3 } { 5 } - \frac { 2 } { 1 0 } ) + \frac { 1 } { 4 } ( \frac { 2 } { 5 } - \frac { 3 } { 1 0 } ) = \frac { 2 3 } { 4 0 } . } \end{array}
$$

For a no-return policy starting A, a zero leaves the choice between completing A for additional value $3 / 1 0$ and abandoning it for $B ,$ whose value is $1 / 5$ . Completing A raises the incumbent to $2 / 5$ , at which inspecting B ties stopping. Thus starting A has no-return value $- 1 / 1 0 + ( 1 / 2 ) \cdot 1 +$ $( 1 / 2 ) \cdot ( 3 / 1 0 ) = 1 1 / 2 0$ . Starting B and then acting optimally on $A$ gives $2 1 / 4 0$ , and stopping gives zero; these exhaust the first actions and prove the no-return optimum.

## D Supporting Results for Shared Laws

## D.1 Own-arm-law limits and conditional stopping

Proof of Proposition 5. Let $q _ { 0 } = 0$ . Arm A has one collectable reward $R \sim \mathrm { B e r n o u l l i } ( 1 / 2 )$ at cost 0.4. Arm B has a noncollectable signal $U \sim$ Bernoulli(1/2) and a collectable reward identically zero; each of its stages costs 0.01. In the first instance U and R are independent; in the second $U = R$ . The separate arm laws are identical at the root, with $G _ { A } = 0 . 2$ and $G _ { B } = - 0 . 0 2$ in both instances. In the independent instance, starting A has value $0 . 5 - 0 . 4 = 0 . 1 0$ . Starting B costs 0.01 before the same optimal A decision, giving 0.09; its zero terminal reward is never worth buying. In the shared instance, after observing $U = h$ the only potentially profitable action is to inspect A, with net return $h - 0 . 4 .$ . Thus $Q ^ { \star } ( B ) = - 0 . 0 1 + \mathbb { E } [ \operatorname* { m a x } \{ 0 , R - 0 . 4 \} ] = 0 . 2 9$ exactly. Starting A still has value 0.10: after its reward is known neither stage of B can improve the incumbent or supply useful information. The value $G _ { B } = - 0 . 0 2$ also follows from $\Phi _ { B } ( r ) = \operatorname* { m a x } \{ - 0 . 0 1 , ( 0 - r ) ^ { + } - 0 . 0 2 \}$ for a run restricted to B. Stopping gives zero. Thus the unique optimal first actions are A and $B ,$ respectively. A rule with the stated inputs must choose the same action distribution in both, but optimality requires probability one on diferent actions. □

For the general operational planning law, write $\mathcal { R } = \{ S : \operatorname* { m a x } _ { i \in \mathbb { Z } ( S ) } G _ { i } ( S ) \leq q ( S ) \}$ , with max $\mathcal { O } = - \infty$ , for the arm-local index stopping region.

Proposition 8 (Continuation and conditional stopping). Under Assumption 1, at every $S \not \in { \mathcal { R } }$ 2

$$
V ^ { \star } ( S ) \geq h ( S ) + \operatorname* { m a x } _ { i \in { \mathcal { I } } ( S ) } \Phi _ { i } ^ { ( \nu _ { i } ) } ( S ; q ( S ) ) > h ( S ) .
$$

$I f \mathcal { R }$ is closed forward, meaning that $F ( S , i , Y ) \in \mathcal { R }$ almost surely for every $S \in { \mathcal { R } }$ and available $i ,$ then ${ V ^ { \star } ( S ) = h ( S ) }$ on $\mathcal { R }$ . Under this additional hypothesis, R is exactly the optimal stopping region.

Proof. An attaining single-arm rule exists by finite-horizon backward induction. Executing it and then stopping is admissible, with value $h ( S ) + \Phi _ { i } ^ { ( \nu _ { i } ) } ( S ; q ( S ) )$ , proving the first assertion. For the second, induct on the total stages remaining. At a state in R, closure and the induction hypothesis give $V ^ { \star } ( S ^ { \prime } ) = h ( S ^ { \prime } )$ after every possible first advance. Therefore

$$
Q ^ { \star } ( S , i ) = h ( S ) + \mathbb { E } [ ( \tilde { Y } - q ( S ) ) ^ { + } \mid S , i ] - c ( S , i ) \leq h ( S ) + \Phi _ { i } ^ { ( \nu _ { i } ) } ( S ; q ( S ) ) \leq h ( S ) .
$$

The Bellman equation proves stopping optimal; the exhausted-state base case is immediate.

Example 3 (A premature index stop). In the construction above, change A’s cost to 0.6. Its index is now −0.1, while B’s remains −0.02. The arm-local index stops at value zero, but buying U and inspecting A only when $U = 1$ has value $\begin{array} { r } { \mathrm { ~ - 0 . 0 1 + } \frac { 1 } { 2 } ( 1 - 0 . 6 ) = 0 . 1 9 } \end{array}$ . Forward closure fails: after that signal equals one, A’s index is $0 . 4 > q _ { 0 }$

Closure quantifies over every available advance in ${ \mathcal { R } } ,$ including those the index never takes. It is suficient, not necessary. Independent arms have the exact stopping rule without it: an arm’s own observation may raise its index after an unprofitable initial inspection. The positive-run assertion concerns a specified single-arm rule, rather than the utility of repeatedly recomputing and switching under a general shared law. We do not establish closure for general shared surrogates, including correlated Gaussian-process models.

## D.2 Proof of Proposition 6

Example 4 (The switching family). Let $\Theta \in \{ 0 , 1 \}$ have prior $p = \mathbb { P } ( \Theta = 1 )$ and let arms $A , B$ each with two stages, be conditionally independent given Θ. Arm i has terminal reward $R _ { i } \in \{ 0 , 1 \}$ with $\mathbb { P } ( R _ { i } = 1 \mid \Theta = 0 ) = \alpha _ { i }$ and $\mathbb { P } ( R _ { i } = 1 \mid \Theta = 1 ) = 1 / 5$ , where $\alpha _ { A } = 4 / 5$ and $\alpha _ { B } = 7 / 1 0$ and first reading $u _ { i } S _ { i }$ , where $S _ { i } \in \{ 0 , 1 \}$ equals $R _ { i }$ with probability 3/4 given $R _ { i } , \ u _ { A } = 2 / 5$ and $u _ { B } = 3 / 5$ . First stages cost $1 / 1 0 0 _ { ; }$ , terminal stages cost $c \in [ 1 9 / 1 0 0 , 1 / 5 ]$ , and $q _ { 0 } = 0$ . Arm $A ^ { \prime } s$ terminal reward is more likely to be one when $\Theta = 0$ , and equally likely when $\Theta = 1 ;$ arm B has the larger intermediate score. Given the terminal rewards, the signal errors are independent of Θ and of each other. The thresholds are

$$
\tau _ { 1 } = { \frac { 1 5 } { 1 9 } } , \qquad \tau _ { 2 } ( c ) = { \frac { 1 3 2 - 4 2 0 c } { 5 0 + 3 5 c } } , \qquad \tau _ { 3 } ( c ) = { \frac { 2 8 8 - 7 8 0 c } { 2 5 0 - 5 3 5 c } } .
$$

A fixed-cost member. At $c = 1 / 5$ , the thresholds in Example 4 reduce to $1 5 / 1 9 , 1 6 / 1 9$ and $1 2 / 1 3$ . Terminal rewards are always collectable, and the optimal first advances are
<table><tr><td>Prior  $p$ </td><td> $( 0 , { \frac { 1 5 } { 1 9 } } )$ </td><td> $( { \textstyle { \frac { 1 5 } { 1 9 } } } , { \textstyle { \frac { 1 6 } { 1 9 } } } )$ </td><td> $\textstyle { \bigl ( } { \frac { 1 6 } { 1 9 } } , { \frac { 1 2 } { 1 3 } } { \bigr ) }$   $\textstyle { \left( { \frac { 1 2 } { 1 3 } } , 1 \right) }$ </td></tr><tr><td>B&#x27;s signal collectable</td><td>A</td><td>B</td><td>A B</td></tr><tr><td> $B \mathrm { { ^ { * } s } }$  signal not collectable</td><td>A</td><td>A</td><td>A A</td></tr></table>

Either choice of A’s intermediate collection rights gives the same rows.

We prove Proposition 6 for this family. Write $h = \left( h _ { A } , h _ { B } \right)$ for the observed history (each arm has an empty prefix, two signal prefixes and four completed prefixes, so there are 49 joint histories $h )$ . Let $L _ { \theta } ( h )$ be the probability of $h$ given $\Theta = \theta ;$ every one of the 16 joint paths has positive probability in both latent states, so no history reveals Θ. For a fixed deterministic contingent policy, its unnormalized payof from $h ,$

$$
W _ { h } ^ { \pi } ( p , c ) = \sum _ { \theta } \mathbb { P } ( \Theta = \theta ) L _ { \theta } ( h ) \mathbb { E } _ { \theta } \Big [ q ( S _ { \tau } ) - \sum _ { t < \tau } c ( S _ { t } , i _ { t } ) \Big | h \Big ] ,
$$

is afine in $p$ and afine in $c ,$ because the incumbent does not depend on $c$ and each terminal stage contributes −c times a path probability. The unnormalized optimal value $W _ { h } = \operatorname* { m a x } _ { \pi } W _ { h } ^ { \pi }$ is therefore, for fixed $^ { c , }$ continuous and piecewise afine in $p ,$ and satisfies

$$
\begin{array} { r } { W _ { h } = \operatorname* { m a x } \Big \{ q ( h ) M _ { h } , \ \operatorname* { m a x } \left[ - c _ { i } ( h ) M _ { h } + \sum _ { y } W _ { ( h , i , y ) } \right] \Big \} , \qquad M _ { h } = ( 1 - p ) L _ { 0 } ( h ) + p L _ { 1 } ( h ) , } \end{array}
$$

with $c _ { i } ( h ) \in \{ 1 / 1 0 0 , c \}$ . Normalizing by $M _ { h } > 0$ gives the Bellman equation for the posterior value, so the optimal first actions at the root are the maximizers of the bracketed terms at $h = \emptyset$

1. Certify all priors at a fixed cost. For rational $c ,$ compute the piecewise-afine Bellman recursion in rational arithmetic. Partition [0, 1] at the union of breakpoints of each state’s value, action values and child values. On each interval, every expression is afine in $p \mathrm { : }$ domination of all actions at both endpoints, with one action attaining the value at both, proves the Bellman identity throughout. Backward induction therefore certifies the optimum for every prior, rather than only sampled priors. For example, with both intermediate readings collectable and $c = 1 / 5$ , the root advance values with optimal continuation satisfy

$$
Q _ { A } ( p ) - Q _ { B } ( p ) = \frac { 1 5 - 1 9 p } { 2 0 0 0 } , \qquad p \in [ 9 / 1 3 , 6 3 / 7 6 ] .
$$

Their equality gives $\tau _ { 1 } = 1 5 / 1 9$ and the first change from A to $B ;$ the certificate also checks the winning advance against stopping.

2. Certify open intervals of costs. Run the same recursion and Bellman certificate over $\mathbb { Q } ( c )$ Every sort, breakpoint test and certificate comparison depends on the sign of a rational-function diference d. Identically zero diferences are exact identities; for each nonzero $d ,$ record its sign at a generic rational sample cost. If none has a zero or pole on an open interval $I ,$ every comparison keeps its outcome there and evaluation commutes with the arithmetic. Thus every $c \in I$ follows the same computational branch and inherits the all-prior Bellman certificate. Exact Sturm counts on numerators and denominators establish this sign invariance.

3. Cover exceptional costs and boundaries. On $( 1 9 / 1 0 0 , 1 / 5 )$ , the only interior failure of sign invariance has numerator proportional to $2 1 0 0 0 c ^ { 2 } - 1 2 0 3 5 c + 1 5 3 6$ , whose root is $c ^ { * } = ( 2 4 0 7 -$ $\sqrt { 6 3 2 6 8 9 } ) / 8 4 0 0 \approx 0 . 1 9 1 8 6$ . Here two partition points used by the computation coincide. This changes the computation’s partition, while the symbolic value and action functions it returns are identical on the two sides. $\mathrm { O n } \ ( 1 9 / 1 0 0 , c ^ { * } )$ and $( c ^ { * } , 1 / 5 )$ , all recorded nonzero diferences keep their signs and the output gives the thresholds in Example 4. Fixed-cost certificates cover $c = 1 9 / 1 0 0$ $c = 1 / 5$ in rational arithmetic and $c = c ^ { * }$ in $\mathbb { Q } ( { \sqrt { 6 3 2 6 8 9 } } )$ . With only $B \mathrm { { ^ { * } s } }$ intermediate reading collectable, the same split and endpoint checks give the same regions. With terminal-only collection or only A’s intermediate reading collectable, no split is needed: A is uniquely optimal throughout (0, 1) for every cost in the closed interval.

Exact root counts also establish positive threshold denominators and $0 < \tau _ { 1 } < \tau _ { 2 } ( c ) < \tau _ { 3 } ( c ) < 1$ throughout the cost interval. For uniqueness, within each open first-action region the winning advance strictly exceeds stopping and the other advance at every contained breakpoint and midpoint; nonnegative endpoint diferences and positive midpoint diferences of afine functions give strict inequalities in the open piece. The advances tie at $p = \tau _ { k } ( c )$ . The verification script and output are included with the code.

## D.3 Signed correction and robustness proof

Throughout this subsection impose the coherent-law and arm-local-cost restriction of Proposition 7. Let ${ \sf Q } _ { S } = \otimes _ { i } { \sf P } _ { j , S }$ and define

$$
\mathcal { V } _ { \otimes } ( S ) = \mathbb { E } _ { \mathsf { Q } _ { S } } \operatorname* { m a x } \{ q ( S ) , Z _ { 1 } ^ { S } , \ldots , Z _ { N } ^ { S } \} .
$$

This lies in $[ y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } } ]$ and is the independent-surrogate optimum, rather than the true shared optimum. Use the continuation utility $U _ { \pi } = V _ { \pi } + K$ defined in Section 2.

After forcing arm i and seeing $Y = y$ , write $S ^ { \prime } = F ( S , i , y )$ . Define $\mathcal { V } _ { i } ^ { \mathrm { f r } } ( S , y )$ using the new incumbent and the advanced arm’s conditional remaining law, but retain $\mathsf { P } _ { j , S }$ for $j \neq i$ . Coherence ensures the advanced law equals its actual marginal at $S ^ { \prime } { \mathrm { ; } }$ only untouched laws difer. Put

$$
\begin{array} { r l } & { d _ { 0 } ( S ) = \mathcal { V } _ { \otimes } ( S ) - q ( S ) , } \\ & { ~ d _ { i } ( S ) = \mathcal { V } _ { \otimes } ( S ) + c ( S , i ) - \mathbb { E } [ \mathcal { V } _ { i } ^ { \mathrm { f r } } ( S , Y ) \mid S , i ] , } \\ & { \Gamma _ { i } ( S ) = \mathbb { E } [ \mathcal { V } _ { \otimes } ( S ^ { \prime } ) - \mathcal { V } _ { i } ^ { \mathrm { f r } } ( S , Y ) \mid S , i ] . } \end{array}\tag{2}
$$

Here $d _ { 0 }$ is the independent surrogate’s stop loss, $d _ { i }$ its loss from advancing i under the frozen update, and $\Gamma _ { i }$ the signed change in the surrogate potential from updating untouched arms. Both signs of $\Gamma _ { i }$ are possible; the proof controls its magnitude. The losses $d _ { 0 } , d _ { i }$ are nonnegative by Theorem 2 applied to $\mathsf Q _ { S }$ . In particular, with $O _ { i } = \operatorname* { m a x } \{ q , \operatorname* { m a x } _ { j \neq i } Z _ { j } ^ { S } \}$ independent of arm i under $\mathsf Q _ { S }$ 2

$$
d _ { i } = c ( S , i ) - \mathbb { E } _ { \mathbb { Q } _ { S } } [ ( W _ { i } ^ { S } - \operatorname* { m a x } \{ O _ { i } , G _ { i } ( S ) \} ) ^ { + } ] .
$$

For finite $G _ { i }$ this is the capped-overshoot expression of Theorem 2; for $G _ { i } = - \infty$ it equals $c ( S , i )$

Proof of Proposition 7. The true one-step residual is the signed update correction less the nonnegative surrogate action loss:

$$
- c ( S , i ) + \mathbb { E } _ { \mathsf { P } } [ \mathcal { V } _ { \otimes } ( S ^ { \prime } ) \mid S , i ] - \mathcal { V } _ { \otimes } ( S ) = \Gamma _ { i } ( S ) - d _ { i } ( S ) .\tag{3}
$$

Summing conditional expectations with indicators before the stopping time over the deterministic range $0 , \ldots , H - 1$ gives, for every policy,

$$
U _ { \pi } ( S _ { 0 } ) = \mathcal { V } _ { \otimes } ( S _ { 0 } ) + \mathbb { E } _ { \pi } \sum _ { b < \tau _ { \pi } } ( \Gamma _ { i _ { b } } - d _ { i _ { b } } ) ( S _ { b } ) - \mathbb { E } _ { \pi } d _ { 0 } ( S _ { \tau _ { \pi } } ) .\tag{4}
$$

All sums are finite and bounded. At each chosen index advance the surrogate gap $d _ { i }$ is zero; at its stop $d _ { 0 } = 0$ . Consequently

$$
U _ { \pi _ { \mathrm { I X } } } ( S _ { 0 } ) = \mathcal { V } _ { \otimes } ( S _ { 0 } ) + \mathbb { E } _ { \pi _ { \mathrm { I X } } } \sum _ { b < \tau _ { \mathrm { I X } } } \Gamma _ { i _ { b } } ( S _ { b } ) .\tag{5}
$$

Subtracting from (4) for a true optimal policy $\pi ^ { \star }$ , dropping its nonnegative stop gap, and taking positive parts yields

$$
U ^ { \star } - U _ { \pi _ { \mathrm { I X } } } \leq \mathbb { E } _ { \pi ^ { \star } } \sum _ { b < \tau _ { \star } } ( \Gamma _ { i _ { b } } - d _ { i _ { b } } ) ^ { + } ( S _ { b } ) + \mathbb { E } _ { \pi _ { \mathrm { I X } } } \sum _ { b < \tau _ { \mathrm { I X } } } \Gamma _ { i _ { b } } ^ { - } ( S _ { b } ) ,\tag{6}
$$

where $x ^ { - } = ( - x ) ^ { + }$ . The first sum allows positive corrections along the true optimal trajectory; the second allows negative corrections along the index trajectory. This is why both trajectories enter the comparison.

Fix $S , i , y$ . Replace the frozen untouched-arm capped distributions by their updated distributions one arm at a time. For each replacement, condition on all other independent capped variables. Their maximum with the new incumbent is an outside option in $[ q ( S ^ { \prime } ) , y _ { \mathrm { m a x } } ]$ , so the absolute value change is at most the corresponding supremum defining $\varepsilon _ { i } ( S )$ . The advanced arm’s law is common to both products. Triangle inequality and expectation give $| \Gamma _ { i } ( S ) | \le \varepsilon _ { i } ( S )$ . Each summand in (6) is at most ¯ε and each trajectory has at most H advances. The two contributions are therefore at most Hε¯ each, giving $2 H \bar { \varepsilon } ;$ removing the same sunk cost from both values does not change regret.

This is the finite-horizon approximate-planning argument. In the formulation of Subramanian et al. (2022, Theorem 9 and Remark 11), use the current marginal laws and own histories as the abstract state, frozen updates as the approximate transition, and incremental reward $q ( S ^ { \prime } ) -$ $q ( S ) - c ( S , i )$ . The approximate value is $\nu _ { \otimes } - q$ and its expected next-value discrepancy is exactly $\Gamma _ { i }$ . The envelope above specializes that discrepancy using local stop-loss values. The local values are 1-Lipschitz in the outside option, so their supremum is measurable by using a countable dense set and the endpoints. Replacing $Z _ { j }$ by $\operatorname* { m a x } \{ y _ { \operatorname* { m i n } } , Z _ { j } \}$ handles exhausted arms without infinite arithmetic.

A path-total-variation envelope. Let $D = y _ { \mathrm { m a x } } - y _ { \mathrm { m i n } }$ and put $B _ { j } ( S ) = D + \nu _ { j } ( S ) c _ { \mathrm { m a x } }$ if a collectable stage remains, and zero otherwise. With $\operatorname { T V } ( \mu , \nu ) = \operatorname* { s u p } _ { A } | \mu ( A ) - \nu ( A ) |$

$$
\varepsilon _ { i } ( S ) \leq \varepsilon _ { i } ^ { \mathrm { T V } } ( S ) = \mathbb { E } \left[ \sum _ { j \neq i } B _ { j } ( S ) \mathrm { T V } ( \mathsf { P } _ { j , S ^ { \prime } } , \mathsf { P } _ { j , S } ) \Bigg | S , i \right] .\tag{7}
$$

With densities, our TV is half the $L ^ { 1 }$ distance; the convention in Subramanian et al. (2022, Section 3.1) is the full $L ^ { 1 }$ distance. The constants here use our stated convention throughout. Indeed, each fixed local stopping rule, including immediate stopping, has path payof max $\{ r , M _ { \sigma } \} - \kappa _ { \sigma }$ in $[ y _ { \mathrm { m i n } } - \nu _ { j } c _ { \mathrm { m a x } } , y _ { \mathrm { m a x } } ]$ . The same rule class is feasible under both marginal path laws because costs are arm-local. TV bounds each expectation diference by its payof oscillation times TV, and taking the supremum over rules preserves the bound. Without a collectable stage, immediate stopping is optimal under both laws. This proves (7).

Proof of Corollary 2. Conditional independence given Θ makes arm $j ^ { \prime } \mathrm { s }$ remaining-path law at $S$ the mixture $\textstyle \int \mathsf { P } _ { j } ^ { \vartheta } ( \cdot \mid s _ { j } ) \mu _ { S } ( d \vartheta )$ ; a reading on arm i changes the mixing measure to $\mu _ { S ^ { \prime } }$ and leaves the components unchanged. Write $\mu _ { S ^ { \prime } } - \mu _ { S } = \xi ^ { + } - \xi ^ { - }$ by the Jordan decomposition, with common mass $t = \mathrm { T V } ( \mu _ { S ^ { \prime } } , \mu _ { S } )$ . If $t \ : = \ : 0$ the untouched marginal laws agree; otherwise $\mathsf { P } _ { j , S ^ { \prime } } - \mathsf { P } _ { j , S } =$ $\begin{array} { r } { t \big ( \int \mathsf { P } _ { j } ^ { \vartheta } d \xi ^ { + } / t - \int \mathsf { P } _ { j } ^ { \vartheta } d \xi ^ { - } / t \big ) } \end{array}$ , and the total variation between two mixtures of the components is at most the largest total variation between components, so $\mathrm { T V } ( \mathsf { P } _ { j , S ^ { \prime } } , \mathsf { P } _ { j , S } ) \leq \Lambda _ { j } ( S ) t$ . Next, let $J$ be the joint law of $( \Theta , Y )$ given $S , i$ , let $\beta$ be its $Y$ marginal, and put $M = \mu _ { S } \otimes \beta$ . The finite measure $J + M$ dominates both $J$ and M. Integrating the absolute diference of their densities with respect to $J + M$ , first conditional on Y and then conditional on $\Theta ,$ gives the total-variation disintegration identity

$$
\begin{array} { r } { \mathbb { E } _ { Y } \operatorname { T V } ( \mu _ { S } ( \cdot \mid Y ) , \mu _ { S } ) = \operatorname { T V } ( J , M ) = \mathbb { E } _ { \vartheta \sim \mu _ { S } } \operatorname { T V } ( \ P _ { i } ^ { \vartheta } ( Y \in \cdot \mid s _ { i } ) , \beta ) = \eta _ { i } ( S ) . } \end{array}
$$

Regular conditional laws exist on the stated standard Borel spaces. This argument applies also to singular joint laws, without assuming $J \ll M$ . Substituting into (7) gives the envelope, and Proposition 7 the regret bound. The two-draw bound is the convexity inequality $\begin{array} { r } { \mathrm { T V } ( \mathsf { P } ^ { \vartheta } , \int \mathsf { P } ^ { \vartheta ^ { \prime } } \mu ( d \vartheta ^ { \prime } ) ) \leq } \end{array}$ $\textstyle \int \mathrm { T V } ( \mathsf { P } ^ { \vartheta } , \mathsf { P } ^ { \vartheta ^ { \prime } } ) \mu ( d \vartheta ^ { \prime } )$ . The binary formulas are the case of two components, computed next, in which $\varepsilon _ { i } ^ { \mathrm { { T V } } }$ equals the displayed product. □

If arm paths are independent conditional on binary $\Theta _ { i }$ , let $p = \mathsf { P } ( \Theta = 1 \mid S )$ , and let $D _ { i } ^ { \mathrm { n e x t } }$ and $D _ { j } ^ { \mathrm { p a t h } }$ be the TV separations of the next-reading and remaining-path laws, respectively, between the two latent components. Then, for $0 < p < 1$ ，

$$
\varepsilon _ { i } ^ { \mathrm { T V } } ( S ) = 2 p ( 1 - p ) D _ { i } ^ { \mathrm { n e x t } } ( S ) \sum _ { j \neq i } B _ { j } ( S ) D _ { j } ^ { \mathrm { p a t h } } ( S ) .\tag{8}
$$

Here $\Theta$ is an uncertain latent state under the fixed planning law, distinct from its fixed surrogate parameters $\theta .$ To see this, set $p _ { y } = \mathsf { P } ( \Theta = 1 \mid S , Y _ { i } = y )$ . An untouched arm’s component laws are unchanged, so its mixture-TV shift is $| p _ { y } - p | D _ { i } ^ { \mathrm { p a t h } }$ . Bayes’ rule gives $\mathbb { E } | p _ { Y } - p | = 2 p ( 1 - p ) D _ { i } ^ { \mathrm { n e x t } }$ by taking absolute variation of $( p _ { y } - p ) \mu ( d y ) = ^ { \circ } p ( 1 - p ) ( \mu _ { 1 } - \mu _ { 0 } ) ( d y )$ for $\mu = ( 1 - p ) \mu _ { 0 } + p \mu _ { 1 }$ . At $p \in \{ 0 , 1 \}$ the envelope is zero, without defining null-component conditionals.

Concentration and predictive dispersion. Weak posterior concentration alone does not force η<sub>i</sub> to vanish. For example, consider beliefs $\mu _ { n } = \mathrm { U n i f o r m } [ 0 , 1 / n ]$ on a common component Θ, with every arm’s next reading equal to Θ. These degenerate paths are conditionally independent given Θ and have bounded readings. Although $\mu _ { n }$ converges weakly to a point mass at zero, $\mathrm { T V } ( \delta _ { \vartheta } , \mu _ { n } ) = 1$ for every $\vartheta ,$ so $\eta _ { i } = 1$ for every n. Predictive total-variation regularity is needed to deduce local dispersion decay from concentration in a continuous component space. To deduce a vanishing regret bound, the envelope must additionally vanish uniformly over feasible future states, as required in Corollary 2.

Why future-state control is required. Let $( U , A , B )$ be uniform over 000, 011, 101, 110. Each pair is independent. Inspecting noncollectable U costs $1 / 1 0 0 $ ; inspecting collectable A or B costs $3 / 5$ each, with $q _ { 0 } = 0$ . Every root envelope is zero because one root observation leaves each other marginal unchanged, and the index stops. After $U = 1$ , A and B are opposite: inspect A, then B only if $A = 0$ , for value $1 / 1 0$ . After $U = 0$ , stopping is optimal. Buying U therefore has value $1 / 2 5$ , whereas starting A or $B$ has value $- 1 / 2 0 0$ . Later states have nonzero envelopes. Thus even pairwise independence of whole arm paths at the root cannot replace the all-history condition.

## E Numerical Verification

The checks use fully specified finite laws, exact rational arithmetic, and a joint Bellman solver independent of the arm-wise covered-call recursion. Within each instance, policies use the same law, costs, collection rights and observation protocol. Table 1 summarizes the verification; each study’s model, checks and results are given together below.

<table><tr><td>Instance family</td><td></td><td>Cases Exact checks</td></tr><tr><td>Independent trees</td><td></td><td>192 Value representation, maximal-index actions, stop and advance losses</td></tr><tr><td>Staging grid</td><td></td><td>660 Index optimality and accumulated policy losses</td></tr><tr><td>Comparative pairs</td><td></td><td>1763 + 60 Coupled covered-call overshoot identity on grid and ran- dom pairs</td></tr><tr><td>Shared-law audit</td><td>453</td><td>Signed residual, policy telescoping, and local-value/path- TV envelopes</td></tr><tr><td>Concentration sweep</td><td></td><td>240 Policy accounting and sharing envelopes at five binary- component priors</td></tr></table>

Table 1: All checks pass in exact arithmetic against direct Bellman calculations. Cases count starting instances, grid cells or problem pairs as indicated; they are verification coverage, not independent statistical observations.

## E.1 Independent-arm verification

Two independent binary-tree arms are generated in four collection regimes: every stage on depth three, terminal only on depth four, {1, 3} on depth three, and {2, 4} on depth four. At each own history, independently choose a cost from $\{ 1 / 1 0 0 , 3 / 1 0 0 , 3 / 5 0 , 1 / 1 0 \}$ , two distinct readings from $\{ 0 , 1 / 4 , 1 / 2 , 3 / 4 , 1 \}$ , and a branch probability from $\{ 1 / 4 , 1 / 2 , 3 / 4 \}$ . Each regime uses seeds 2026102400 through 2026102423 and is evaluated from both the empty state and a prefix after three random feasible advances, giving 192 starting instances. Prefixes are chosen without an index or stopping criterion and retain their observed incumbent.

The covered-call value, maximal-index actions above the incumbent, and all stop and advance losses agree with direct Bellman optimization over 23,839 evaluated states and 37,467 available advance actions.

## E.2 Staging and policy-loss verification

Each independent arm reveals two noisy signals followed by a fair binary terminal reward $H _ { i }$ Conditional on $H _ { i }$ , signals $S _ { i 1 } , S _ { i 2 }$ are independent and equal $H _ { i }$ with probability α. The readings are $( 2 S _ { i 1 } / 5 , 3 S _ { i 2 } / 5 , H _ { i } )$ , and predictions condition on both signals. The grid crosses

$$
\begin{array} { r l r } & { \alpha \in \{ 1 / 2 , 5 / 8 , 3 / 4 , 7 / 8 , 1 \} , } & { k \in \{ 1 / 2 0 0 , 1 / 5 0 , 1 / 2 0 , 1 / 1 0 \} , } \\ & { c \in \{ 1 / 2 0 , 1 / 5 , 2 / 5 , 3 / 5 , 4 / 5 \} , } & { K _ { i } \in \{ \{ 3 \} , \{ 1 , 3 \} , \{ 1 , 2 , 3 \} \} . } \end{array}
$$

With arm multipliers $m _ { i } \in \{ 1 , 5 / 4 \}$ , the constant-cost family uses $m _ { i } ( k , k , c )$ and the own-historycost family uses

$$
m _ { i } \left( k , k ( 1 + S _ { i 1 } / 2 ) , c \left( 1 + \textstyle { \frac { 1 } { 2 } } { \bf 1 } \{ S _ { i 1 } \not { p } \} \right) \right) .
$$

Each next cost is known before the advance. Deterministic controls have readings $( 0 , 1 / 2 , 1 )$ and $( 0 , 3 / 8 , 3 / 4 )$ , costs $m _ { i } ( k , k , c )$ , and the same cost and collection grids. There are 100 constant-cost cells, 100 own-history-cost cells and 20 deterministic controls per collection regime, giving 660 cells. All runs start at $q _ { 0 } = 0 ;$ zero-probability prefixes at $\alpha = 1$ are omitted. The index stops on an index–incumbent tie and breaks advance ties by the smaller label.

Three specified deviations check Theorem 2. Stop after one takes the initial index advance and stops, or stops immediately if the index does. Lowest index first forces a smallest-index available arm once, even if stopping was optimal, then follows the index. Extra at stop follows the index to its first stop, takes one cheapest available advance if any, and then follows the index. All action ties favor the smaller label. Each deviation’s utility loss equals its accumulated stop and advance losses in every cell. The index agrees with the Bellman optimum over 48,840 state evaluations and 74,160 advance-action checks, including all 300 own-history-cost cells.

## E.3 Comparative statics and resumption

A dominated transition pairs staging cells with the same path laws and arms, but the next larger collectable set, next lower signal cost, next lower terminal cost, or constant in place of own-history costs. Coupling each arm’s capped values verifies Proposition $2$ on all 1,763 transitions. The identity also holds on 60 random binary-tree pairs of depths two to four (seed 20261030), with enlarged collection sets and costs multiplied by 1, $1 / 2$ or $3 / 4$ , including unchanged cases.

Table 2 summarizes the three checks by transition type, collectable set and cost law. For Proposition 3, terminal-only indices are used under intermediate collection in every cell with intermediate rights. The policy attains the expanded-rights optimum in nearly half of these cells and, in about a quarter, loses the full value of the additional rights, so both ends of the proposition’s bound are attained. At $\alpha = 1 / 2 , k = 1 / 2 0 0$ and $c = 3 / 5$ with ${ \cal { K } } _ { i } = \{ 1 , 3 \}$ , for instance, the terminal-only indices are negative, the policy stops at once, and its loss is the full rights value. For Proposition 4, a separate Bellman recursion forbids further advances of any arm once another arm is advanced after it, while retaining all collectable readings. This gives $V _ { \mathrm { N R } }$ . Resumption has positive value exactly in the stochastic cells where the index policy resumes, and in no deterministic control; the largest gap is under one twentieth of its cell’s optimal value. The bound $\rho$ never exceeds the gap, is positive in the same cells, and attains the gap in about a fifth of them.

## E.4 Shared-law verification

The audit has 432 two-stage binary-latent cells, 12 arbitrary three-stage joint laws with own-history costs, and nine settings of the constructed example in Appendix E.5. All start with unobserved arms and $q _ { 0 } = 0$ . The joint Bellman solver retains dependence; the index recomputes from the true posterior arm marginals after every observation.

Binary-latent grid. Conditional on $\Theta \in \{ 0 , 1 \}$ , complete arm paths are independent. Arm i has $R _ { i } \sim$ Bernoulli $\left( a _ { i \Theta } \right)$ and first signal $S _ { i }$ satisfying $\mathbb { P } ( S _ { i } = R _ { i } \ | \ R _ { i } , \Theta ) = r _ { i } .$ . Its readings are $( s _ { i } S _ { i } , R _ { i } )$ , with $( s _ { A } , s _ { B } ) = ( 2 / 5 , 3 / 5 )$ The terminal is collectable; each first stage can independently be designated collectable, giving four choices. Seed 20261026 generates 12 base models with prior $p _ { 0 } \in \{ 1 / 5 , 1 / 2 , 4 / 5 \}$ . For each arm choose two distinct component probabilities from $\{ 1 / 1 0 , 1 / 5 , 3 / 1 0 , 1 / 2 , 7 / 1 0 , 4 / 5 , 9 / 1 0 \} , r _ { i } \in \{ 3 / 5 , 3 / 4 , 9 / 1 0 \}$ , first cost in $\{ 1 / 1 0 0 , 1 / 2 5 , 2 / 2 5 \}$ and terminal cost in $\{ 1 / 2 0 , 1 / 5 , 2 / 5 \}$ . For each model set

$$
\begin{array} { r l r } & { } & { m _ { i } = ( 1 - p _ { 0 } ) a _ { i 0 } + p _ { 0 } a _ { i 1 } , \qquad a _ { i \theta } ( \lambda ) = m _ { i } + \lambda ( a _ { i \theta } - m _ { i } ) , } \\ & { } & { \lambda \in \{ 0 , 1 / 1 0 0 , 1 / 5 0 , 1 / 2 0 , 1 / 1 0 , 1 / 5 , 2 / 5 , 3 / 5 , 1 \} . } \end{array}
$$

<table><tr><td>Dominated transition</td><td>Pairs</td><td>Identity holds</td></tr><tr><td>Larger collectable set</td><td>440</td><td>440</td></tr><tr><td>Lower signal cost</td><td>495</td><td>495</td></tr><tr><td>Lower terminal cost</td><td>528</td><td>528</td></tr><tr><td>Constant for own-history costs</td><td>300</td><td>300</td></tr><tr><td>Collectable set</td><td>Cells</td><td>Zero loss / full rights value lost</td></tr><tr><td>{1, 3}</td><td>220</td><td>133 / 50</td></tr><tr><td>{1,2,3}</td><td>220</td><td>70 /51</td></tr><tr><td>Cost law</td><td>Cells</td><td>Positive gap / ρ equal to gap / largest gap</td></tr><tr><td>Constant</td><td>300</td><td>76 / 17 / 0.0108</td></tr><tr><td>Own-history</td><td>300</td><td>59 / 13 / 0.0156</td></tr><tr><td>Deterministic controls</td><td>60</td><td>0 / 0 / 0.0000</td></tr></table>

Table 2: Comparative statics and resumption on the 660-cell staging grid, in exact arithmetic. Top: dominated pairs of cells; the value diference equals the covered-call overshoot of Proposition 2 in every pair. Middle: terminal-only indices under intermediate collection; the loss lies in $[ 0 , V ^ { \star \prime } - V ^ { \star } ]$ and the last entry counts cells with positive rights value in which all of it is lost. Bottom: the noreturn optimum; the gap $V ^ { \star } - V _ { \mathrm { N R } }$ is positive exactly where the index policy resumes, and $\rho$ never exceeds it.

This holds the full initial arm marginal laws, costs and collection choices fixed within each coupling sweep. $\mathrm { A t } \ \lambda = 0$ the arms are independent, giving 48 independent controls among the $1 2 \times 9 \times 4 =$ 432 cells.

Arbitrary three-stage laws. Continuing the same seeded generator, twelve models assign normalized positive integer weights from $\{ 1 , \ldots , 9 \}$ to all 64 binary joint paths of two three-stage arms. Base costs are $( 1 / 2 0 , 1 / 1 0 , 1 / 5 )$ ; the second cost is 1/4 after first reading one, and the terminal cost is $1 / 2 5$ after prefix $( 0 , 1 )$ . For model $b \in \{ 0 , \ldots , 1 1 \}$ and arm $i \in \{ 0 , 1 \}$ , stage one is collectable when $( b + i )$ mod $2 \neq 0$ , stage two when $( b + i ) ~ \bmod { } 3 \neq 0$ , and stage three always. These laws test sharing without assuming the binary-latent factorization.

Checks and results. Every feasible prefix and available advance is checked, including histories the index never visits. The index stops on an index–incumbent tie and breaks advance ties by smaller label. The Bellman policy used for the refined bound favors advances over stopping, then smaller label; this fixes its occupancy at a value tie. Policy telescoping is checked for the index, the true optimum, immediate stopping, and a smallest-label always-advance policy. Uniform envelopes maximize over all feasible histories; the refined bound uses optimal-policy occupancy and is an oracle diagnostic. Across the 453 cells, the signed residual, telescoping and call/TV inequalities hold over 24,309 feasible prefixes and 21,042 available advances. The independent controls have zero sharing envelopes and zero regret. Negative signed corrections occur, supporting the twosided accounting in (6). Counts measure verification coverage; these known-law checks do not assess learned-surrogate accuracy or practical optimization performance.

## E.5 Positive regret, shared nuisance and concentration

Positive-regret example. Use the binary-latent construction above with $p _ { 0 } = 4 / 5 , ( a _ { A 0 } , a _ { A 1 } ) =$ $( 4 / 5 , 1 / 5 ) , ( a _ { B 0 } , a _ { B 1 } ) = ( 7 / 1 0 , 1 / 5 ) , r _ { A } = r _ { B } = 3 / 4$ , first scores $( 2 / 5 , 3 / 5 )$ , first costs $( 1 9 / 5 0 0 , 1 / 2 0 )$

and terminal costs $( 1 / 5 , 1 / 5 )$ . Both stages are collectable, $q _ { 0 } = 0$ , and the same nine coupling settings are included in the audit. Both initial indices are $1 / 2 ;$ ties favor $A .$ The positive regret in Table 4 demonstrates the loss of exact optimality under sharing, while each displayed bound covers that regret.

Shared nuisance. Outside the 453-cell audit, let the paths be $( U , R _ { A } )$ and $( U , R _ { B } )$ , where U is a shared fair bit and $R _ { A } , R _ { B }$ are independent $\operatorname { B e r n o u l l i } ( 4 / 5 )$ rewards, also independent of $U .$ First readings are noncollectable at cost $1 / 1 0 0 ;$ terminals are collectable at cost $1 / 5$ . The shared bit changes predictions of another noncollectable reading but leaves local continuation values unchanged. The call envelope is zero, the index is optimal with utility $1 7 7 / 2 5 0$ , and the uniform path-TV bound is $2 8 / 5$ . Root joint/product TV is $1 / 2  \it$ , a diferent quantity from the posterior marginal shifts in the path-TV envelope. The control isolates why local-value changes, rather than all remaining-path changes, are the discrepancy relevant to the bound.

Concentration sweep. Also outside the audit, the 12 base models are reused at full component contrast $( \lambda = 1 )$ with all four collection choices, and only the prior $p = \mathbb { P } ( \Theta = 1 )$ is varied over $\{ 1 / 2 , 1 / 4 , 1 / 1 0 , 1 / 4 0 , 1 / 1 6 0 \}$ : 240 cells in which the belief about the binary component concentrates while the component laws, reliabilities, costs and collection stay fixed, each running the same exact audit as the grid. For this binary component, $\eta _ { i }$ at the root is $2 p ( 1 - p )$ times the component separation of the next reading (Corollary 2). Table 3 reports, for each prior, the largest regret and the largest uniform call and path-TV bounds over its 48 cells. Among the tested priors, regret is positive only at $p = 1 / 2 ,$ , and the largest uniform bounds fall by more than an order of magnitude from $p = 1 / 2$ to $p = 1 / 1 6 0$ The bounds remain conservative. Their decrease is an observed property of this prior-sensitivity calculation on fixed finite laws; the corollary requires control of dispersion over all feasible future histories, in addition to concentration at the root.

<table><tr><td>Prior p</td><td> $2 p ( 1 - p )$ </td><td>Positive-regret cells</td><td>Max regret</td><td>Max call bound</td><td>Max path-TV bound</td></tr><tr><td> $1 / 2$ </td><td>0.5000</td><td>4</td><td>0.00698</td><td>0.896</td><td>2.015</td></tr><tr><td> $1 / 4$ </td><td>0.3750</td><td>0</td><td>0.00000</td><td>0.715</td><td>1.680</td></tr><tr><td> $1 / 1 0$ </td><td>0.1800</td><td>0</td><td>0.00000</td><td>0.446</td><td>1.070</td></tr><tr><td>1/40</td><td>0.0488</td><td>0</td><td>0.00000</td><td>0.166</td><td>0.398</td></tr><tr><td>1/160</td><td>0.0124</td><td>0</td><td>0.00000</td><td>0.046</td><td>0.111</td></tr></table>

Table 3: Concentration sweep over the prior of the binary component: the 12 base models at full contrast and all four collection choices, 48 cells per row. Regret and the uniform bounds are row maxima; each cell’s bounds maximize over all feasible histories. The factor $2 p ( 1 - p )$ describes the root belief in Corollary 2. Exact values are rounded.

<table><tr><td>Instance</td><td>Regret</td><td>Refined (oracle)</td><td>Call</td><td>Path-TV</td></tr><tr><td>Witness,  $\lambda = 1 / 1 0$ </td><td>0.000024</td><td>0.000156</td><td>0.002142</td><td>0.012600</td></tr><tr><td>Witness,  $\lambda = 1$ </td><td>0.001700</td><td>0.014900</td><td>0.214158</td><td>1.260000</td></tr><tr><td>Shared nuisance</td><td>0.000000</td><td>0.000000</td><td>0.000000</td><td>5.600000</td></tr></table>

Table 4: Checks of the sharing bound under specified joint laws. Refined uses the true optimal policy occupancy in (6); Call and Path-TV are uniform 2Hε¯ bounds. Values are rounded to six decimals. The nuisance control is outside the 453-cell audit.

## F Index Computation

## F.1 Conditional-law calculations

Algorithm 1 constructs each covered-call law by backward recursion. For discrete $W ,$ sorting its finite atoms and sweeping their cumulative mass and first moment solves the piecewise-linear reservation equation exactly. These calculations integrate over hypothetical futures; an actual advance does not disclose W or $Z .$

Continuous reading laws require integration or approximation; no numerical-integration error guarantee is added here.

## F.2 Finite-tree complexity

An arm with T remaining stages and J possible next observations is a depth-T observation tree. At a node with t observations taken, W has at most $J ^ { T - t }$ atoms. Sorting them and sweeping cumulative probability and first moment solves the piecewise-linear reservation equation exactly for a discrete law. A direct implementation sorting at each node has the conservative bound $O ( T ^ { 2 } J ^ { T } \log ( 1 + J ) )$ for these atom operations, excluding evaluation of the predictive kernels. Sorted-list merging can avoid repeated sorting, but no such improvement is needed for the present guarantees. Here $J \geq 1$ is an upper bound on the number of next observations; using log(1 + J) includes deterministic branches. For $J \ge 2$ , the bound is exponential in T. The index separates the arms rather than solving their joint Bellman problem; it does not remove the need to integrate each arm’s possible future histories. The cost of evaluating predictive kernels must also be included when measuring decision time.

Code and archived checks. The code, records and verification scripts are at https://github. com/longvus/staged-pandora-box; its next\_paper/P2\_REPRODUCIBILITY.md identifies the reference solvers, protocols, archived records and validation commands.