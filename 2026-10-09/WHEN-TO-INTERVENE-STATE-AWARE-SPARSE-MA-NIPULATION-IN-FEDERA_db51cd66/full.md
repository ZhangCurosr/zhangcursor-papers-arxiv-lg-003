# WHEN TO INTERVENE? STATE-AWARE SPARSE MA-NIPULATION IN FEDERATED REINFORCEMENT LEARN-ING

Shutong Zheng<sup>1</sup> Sijia Chen<sup>2,∗</sup>

<sup>1</sup>Sun Yat-sen University

<sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou) zhengsht29@mail2.sysu.edu.cn, sijiachen@hkust-gz.edu.cn

## ABSTRACT

Federated reinforcement learning (FRL) enables distributed agents to collaboratively train decision-making policies, but its decentralized training process also exposes global policy learning to Byzantine manipulation. Existing poisoning attacks primarily focus on how to construct malicious updates, while trajectorylevel intervention timing remains largely implicit. In sequential decision making, however, where an intervention is applied can alter subsequent trajectories and learning signals. Through controlled experiments, we find that changing the selected trajectory states materially alters attack efficacy even when the maliciousupdate construction is fixed. We therefore identify when as a distinct attack dimension and introduce the Viability-constrained Behavioral Steering Attack (V-BSA), which uses local policy uncertainty to select sparse intervention states and applies envelope-constrained behavioral steering. Across discrete-action benchmarks, V-BSA achieves substantial degradation against robust aggregators and ensemble defenses with only a fraction of the interventions used by dense poisoning, while revealing task- and aggregation-dependent boundaries. Overall, our results highlight intervention timing as a distinct dimension of sequential robustness in FRL. The code is available at https://github.com/Yodeesy/V-BSA.

![](images/c845142e06af58e83d1fecfbd47611d0ede7bdf95d3f50cedd291c44bdc45704.jpg)  
Figure 1: Conceptual comparison of intervention timing in FRL poisoning. (a) Prior work leaves trajectory-level intervention timing implicit, effectively defaulting to dense, state-agnostic intervention across rollout steps. (b) Ours (V-BSA) treats WHEN as an explicit attack dimension: compromised agents maintain nominal execution on routine states and selectively intervene at states identified by policy-entropy gating.

## 1 INTRODUCTION

Federated reinforcement learning (FRL) enables distributed agents to collaboratively optimize policies while keeping local trajectory data decentralized (McMahan et al., 2017; Jin et al., 2022; Khodadadian et al., 2022). This decentralized process, however, exposes global policy learning to

Byzantine clients that can manipulate local learning signals or submitted updates (Blanchard et al., 2017; Fang et al., 2020). Byzantine-robust aggregation has therefore become central to secure FRL, ranging from coordinate-wise statistics such as Median and Trimmed Mean to FRL-specific filtering and recent ensemble-based architectures (Yin et al., 2018; Fan et al., 2021; Fang et al., 2025b).

Recent poisoning attacks in FRL have explored multiple attack surfaces, including local environment and reward manipulation (Ma et al., 2023; 2024), malicious policy-update construction under Byzantine settings (Fan et al., 2021; Fang et al., 2025b), and selective perturbation of important model components (Zhang et al., 2025a). Despite these different mechanisms, attack-oriented approaches primarily focus on how malicious signals or updates are constructed. In sequential decision making, the consequence of a manipulation can vary with the states at which it is applied, as interventions may alter subsequent trajectories, state visitation, and future learning signals. Thus, under a limited intervention budget, the granularity at which interventions are scheduled may itself affect attack leverage.

This leaves a trajectory-level design choice largely implicit in existing FRL poisoning: which states along a local trajectory should receive intervention. Making this choice explicit is important for understanding whether attack efficacy depends only on the construction of the malicious update, or also on where that update is induced along the trajectory.

Motivated by this consideration, we conduct controlled experiments that hold the malicious-update construction (how) fixed while varying only the trajectory-level intervention schedule (when). We observe that changing the selected state subset materially alters downstream attack degradation across evaluation dynamics, even under the same update construction and intervention budget. This empirical observation motivates us to formalize when as a distinct attack dimension under a strictly local, defense-agnostic threat model, explicitly separating trajectory-level state selection (WHEN) from malicious-update construction (HOW).

Based on this observation, we introduce the Viability-Constrained Behavioral Steering Attack (V-BSA). The WHEN component uses locally observable policy entropy to select sparse intervention states, while the HOW component applies target-directed behavioral steering, matches its update scale to the nominal local update, and constrains the result within a client-local empirical update envelope. By explicitly separating trajectory-level state selection from malicious-update construction, V-BSA concentrates a limited intervention budget on selected trajectory states without relying on peer updates or server-side defense information.

We evaluate V-BSA across three discrete-action control environments and multiple Byzantinerobust aggregation settings. The results demonstrate substantial degradation under both conventional robust aggregators and ensemble-based defenses, while also revealing clear task- and aggregationdependent transfer boundaries. Together with the controlled WHEN comparison, these results show that intervention timing provides a distinct perspective for understanding sequential robustness in FRL.

Our main contributions are summarized as follows:

• WHEN as a distinct attack dimension. We explicitly separate trajectory-level intervention timing (WHEN) from malicious-update construction (HOW) in Byzantine FRL and formalize the resulting two-slot attack process.

• Local-only state-aware sparse manipulation. We propose V-BSA, which selects intervention states from local policy statistics and performs viability-constrained behavioral steering under a strictly local, defense-agnostic threat model.

• Empirical characterization. Across three discrete-action environments and multiple aggregation settings, we demonstrate the effect of state-selective intervention and characterize its task and aggregation-dependent transfer boundaries.

## 2 RELATED WORK

Poisoning Attacks in FRL. Model poisoning in federated learning (FL) manipulates client updates to steer the global model (Fang et al., 2020), either by maximizing divergence from benign updates (Fang et al., 2020; Shejwalkar & Houmansadr, 2021) or bounding perturbations within empirical dispersion envelopes (Baruch et al., 2019). FRL introduces sequential, Markovian sampling dynamics that distinguish local update distributions from static FL (Khodadadian et al., 2022). Recent FRL attacks target different attack surfaces, including local-environment or reward manipulation (Ma et al., 2023; 2024), malicious policy-update construction against Byzantine-robust aggregation (Fang et al., 2025b), and selective perturbation of critical model components (Zhang et al., 2025a). These approaches primarily focus on malicious signal or update construction (HOW), leaving trajectory-level intervention timing implicit.

![](images/be68a38f3d5a313f71b3e308d221be6ea223c759b80e6bc6201264629cac0aff.jpg)  
Figure 2: Pilot evaluation of intervention timing. Terminal degradation D on LunarLander under FedPG-BR with fixed update construction across schedules (ρ). Entropy gating induces substantially higher degradation than dense and budget-matched baselines.

Byzantine-Robust Aggregation and Ensemble Defenses. Byzantine-robust learning mitigates malicious updates through filtering and robust statistics, including Krum (Blanchard et al., 2017), Bulyan (El-Mhamdi et al., 2018), coordinate-wise Median and Trimmed Mean (Yin et al., 2018), and geometric medians (Chen et al., 2017), supplemented by worker replication schemes such as DETOX (Rajput et al., 2019). In FRL, FedPG-BR evaluates policy gradients via directional and magnitude alignment against a robust reference (Fan et al., 2021), while recent work examines the empirical sufficiency of robust aggregation under heterogeneous dynamics (Fang et al., 2025a) and client grouping with multi-policy ensemble decisions (Fang et al., 2025b). These defenses operate primarily on submitted updates or resulting policies, without explicitly accounting for the trajectory states that generate them.

Temporal Selectivity and Sparse Interventions. Temporal selectivity has been studied at different operational scales. In FL, recent attacks exploit communication-round selection or multi-round consistency (Yan et al., 2024; Xie et al., 2025; Lyu et al., 2025; Ye et al., 2026), while temporal defenses audit cross-round update similarity to mitigate persistent poisoning (Fung et al., 2018). In single-agent RL, adaptive reward poisoning and state-aware adversarial perturbations target selected decision steps (Lin et al., 2017; Zhang et al., 2020; Cui et al., 2024; Zhang et al., 2025b). These lines of work, however, do not jointly address federated aggregation and intra-round trajectory-level intervention in FRL. To our knowledge, prior FRL poisoning work does not explicitly formulate trajectory-level intervention timing as a distinct attack dimension.

## 3 PRELIMINARIES AND MOTIVATION

## 3.1 FEDERATED REINFORCEMENT LEARNING

We consider a federated reinforcement learning (FRL) system with a central server and N distributed agents. Agent i interacts with a local MDP $\mathcal { M } _ { i } = ( \mathcal { S } , \mathcal { A } , \mathcal { P } _ { i } , \mathcal { R } _ { i } , \gamma , \rho _ { i } )$ under a shared parameterized policy $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ , where S and A denote the state and action spaces. Its expected discounted return is $J _ { i } ( \theta ) = \mathbb { E } _ { \tau \sim ( \pi _ { \theta } , \ M _ { i } ) } [ \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } \mathcal { R } _ { i } ( s _ { k } , a _ { k } ) ]$ ], and the federated learning objective is to optimize the global performance $\begin{array} { r } { J ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } J _ { i } ( \theta ) } \end{array}$

Training proceeds over communication rounds $t = 1 , \dots , T$ . At round t, the server broadcasts the global parameter $\theta ^ { t }$ , each agent executes local rollouts to compute an update $\Delta \theta _ { i } ^ { t }$ , and the server updates the global policy via $\theta ^ { t + 1 } = \theta ^ { t } + \mathrm { A G R } ( \{ \Delta \theta _ { i } ^ { t } \} _ { i = 1 } ^ { N } )$ . The aggregation rule AGR may be FedAvg or a Byzantine-robust alternative.

## 3.2 MOTIVATION

To test whether finer-grained trajectory-level scheduling provides attack leverage beyond intervention frequency, we conduct a controlled pilot study on LunarLander under FedPG-BR aggregation. We hold the malicious-update construction (how) fixed and vary only the intervention schedule (when): dense $( \rho = 1 0 0 \% )$ , random $( \rho = 1 5 \% )$ ), budget-matched random $( \rho = 5 . 7 \% )$ , and entropybased state selection $( \rho \approx 5 . 7 \% )$

Figure 2 shows that intervention frequency alone does not determine attack effectiveness. Dense intervention yields $D = 3 6 . 6 5$ , while random intervention at $\rho = 1 5 \%$ yields $D = 5 0 . 0 1$ . Entropybased selection reaches $D = 1 4 8 . 1 7$ with only $\approx 5 . 7 \%$ of states intervened on, corresponding to $\approx 1 7 . 5 \times$ fewer interventions than the dense baseline. More importantly, under the matched $5 . 7 \%$ budget, entropy-based selection achieves $D = 1 4 8 . 1 7$ versus $\mathbf { \bar {  { D } } } = 9 1 . 9 9$ for random selection. Thus, at a comparable intervention budget and identical update construction, changing which trajectory states are selected materially alters downstream policy degradation, providing direct evidence that trajectory-level scheduling is an independent attack dimension.

## 4 METHODOLOGY

## 4.1 PROBLEM SETTING

Threat Model and Objective. We consider Byzantine FRL with N agents, of which M are compromised, with corruption ratio $\beta = M / N$ . At round t, each compromised agent receives $\theta ^ { t }$ and observes only its local rollout $\mathcal { D } _ { i } ^ { t } = \{ ( s _ { i , k } ^ { t } , a _ { i , k } ^ { t } , r _ { i , k } ^ { t } ) \} _ { k = 1 } ^ { K }$ and historical updates $\Delta \theta _ { i } ^ { 1 : t - 1 }$ . The adversary operates under a strictly local, defense-agnostic information boundary, observing neither peer updates nor server-side aggregation mechanics. Compromised agents execute nominal local rollouts prior to attack synthesis, thereby obtaining the benign update $g _ { i } ^ { t } = \Delta \theta _ { i , \mathrm { c l e a n } } ^ { t }$ . The objective is to maximize terminal degradation $D \triangleq J ( \theta _ { \mathrm { c l e a n } } ) - J ( \theta _ { \mathrm { a t t a c k e d } } )$ under bounded intervention budgets and locally constrained updates.

WHEN–HOW Factorization. FRL operates at two temporal levels: agents make sequential decisions at intra-round trajectory steps k, while local updates are aggregated across communication rounds t. We explicitly separate intervention timing from update construction:

$$
\mathbf { W H E N : } \mathcal { G } _ { \mathrm { w h e n } } ( \cdot ; \theta ^ { t } ) : \mathcal { S } \to \{ 0 , 1 \} , \qquad \mathbf { H O W : } \mathcal { H } _ { \mathrm { h o w } } : ( g _ { i } ^ { t } , \mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } ) \to \Delta \theta _ { i } ^ { t , \mathrm { m a l } } .\tag{1}
$$

The WHEN operator filters rollouts into a sparse intervention set $\mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } ~ = ~ \{ ( s , a , r ) ~ \in ~ \mathcal { D } _ { i } ^ { t } ~ |$ $\mathcal { G } _ { \mathrm { w h e n } } ( s ; \theta ^ { t } ) = 1 \}$ . Conventional dense poisoning corresponds to $\mathcal { G } _ { \mathrm { w h e n } } \equiv 1$ , leaving timing unconstrained. In contrast, V-BSA makes timing explicit via policy-entropy gating and instantiates $\mathcal { H } _ { \mathrm { h o w } }$ via viability-constrained behavioral steering (Figure 3).

## 4.2 INTERVENTION TIMING

The WHEN operator identifies locally susceptible trajectory states without access to peer or server information.

Local Behavioral Susceptibility. For a categorical policy $\pi _ { \boldsymbol \theta } ( \cdot \mid s ) = \operatorname { s o f t m a x } ( z _ { \boldsymbol \theta } ( s ) )$ , parameter perturbation $\delta \theta$ induces behavioral change measured by $\mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot ~ \vert ~ s ) \Vert \pi _ { \theta + \delta \theta } ( \cdot ~ \vert ~ s ) )$ ). Following foundational trust-region and natural policy-gradient literature (Kakade, 2001; Schulman et al., 2015), for locally $C ^ { 3 }$ policies with bounded third derivatives,

$$
\mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \theta + \delta \theta } ) = \frac { 1 } { 2 } \delta \theta ^ { \top } F ( s ) \delta \theta + \mathcal { O } ( \| \delta \theta \| _ { 2 } ^ { 3 } ) , \qquad F ( s ) = J _ { z } ( s ) ^ { \top } M ( p ) J _ { z } ( s ) ,\tag{2}
$$

where $J _ { z } ( s ) = \nabla _ { \theta } z _ { \theta } ( s ) , p = \pi _ { \theta } ( \cdot \mid s )$ , and $M ( p ) = \mathrm { D i a g } ( p ) - p p ^ { \top }$ . Worst-case local susceptibility is $\begin{array} { r } { \chi _ { \epsilon } ( s ) = \frac { \epsilon ^ { 2 } } { 2 } \lambda _ { \mathrm { m a x } } ( F ( s ) ) + \mathcal { O } ( \epsilon ^ { 3 } ) } \end{array}$ . Because the attacker cannot anticipate which parameter directions will be emphasized by an unknown aggregation rule, we adopt the isotropic second-order surrogate $\begin{array} { r } { \bar { \chi } _ { \sigma } ^ { ( 2 ) } ( s ) = \bar { \frac { \sigma ^ { 2 } } { 2 } } \operatorname { T r } ( F ( s ) ) } \end{array}$ .

![](images/41e941a132780e0e00edb474f4bb70f0a0d1df861136513b3c2b272ddf4bddac.jpg)  
Figure 3: Overview of V-BSA. The attack factorizes into trajectory-level WHEN selection and parameter-space HOW construction: an entropy gate selects sensitive states $\mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } ,$ from which a targeted update is synthesized, norm-matched, and constrained by the local empirical envelope $\boldsymbol { B } _ { i } ^ { t }$ before server-side aggregation.

Idealized Selection and Entropy Proxy. Under cardinality budget B, total additive susceptibility is maximized by selecting the B states with largest $\bar { \mathrm { T r } } ( F ( s ) \bar { ) }$ . This oracle ranking serves strictly as an analytical benchmark; V-BSA operates online without buffering trajectories or evaluating Jacobians. To obtain an output-only proxy, consider the sufficient centered-isotropy condition $\overline { { J } } _ { z } ( s ) J _ { z } ( s ) ^ { \top } = c P _ { \bot }$ , where $\begin{array} { r } { \dot { P } _ { \perp } = \dot { I _ { | \mathcal { A } | } } - \frac { 1 } { | \mathcal { A } | } \mathbf { 1 } \mathbf { 1 } ^ { \top } } \end{array}$ and $c > 0$ is state-independent. Because $M ( p ) { \bf 1 } = 0 ;$ , we obtain $\mathrm { T r } ( F ( s ) ) = c ( 1 - \| p \| _ { 2 } ^ { 2 } )$

Theorem 1 (Entropy–Susceptibility Ranking under Centered Isotropy). For a binary policy $p =$ $( q , 1 - q )$ under centered isotropy with state-independent $c > 0$ , Shannon entropy $H ( p )$ and Fisher trace $\operatorname { T r } ( F ( s ) )$ induce the identical state ranking: $H ( p _ { 1 } ) > H ( p _ { 2 } ) \iff \operatorname { T r } ( \dot { F ( s _ { 1 } ) } ) \ \stackrel { \sim } { > } \operatorname { T r } ( F ( s _ { 2 } ) )$

Proof sketch. For binary actions, $\operatorname { T r } ( F ( s ) ) = 2 c q ( 1 - q )$ . Both $H ( q )$ and $2 q ( 1 - q )$ are symmetric about $q = 1 / 2$ and strictly monotonically decreasing in $\left| q - 1 / 2 \right|$ , inducing identical orderings over state susceptibilities (complete derivation in Appendix A.4).

For multi-action policies, exact ranking equivalence does not generally hold, so entropy serves as a practical, backpropagation-free proxy:

$$
\mathcal { G } _ { \mathrm { w h e n } } ( s ; \theta ^ { t } ) = \mathbf { 1 } \left[ \frac { H ( \pi _ { \theta ^ { t } } ( \cdot \mid s ) ) } { \log \mid A \mid } > \tau \right] , \qquad \tau = 0 . 9 ,\tag{3}
$$

where $\tau$ is an a priori fixed threshold rather than tuned to an intervention budget, inducing an endogenous intervention rate. Entropy captures output dispersion via $M ( p )$ but not representation sensitivity in $J _ { z } ( s )$ , motivating our audit in Section 5.2. Continuous-action boundaries are discussed in Appendix E.

## 4.3 VIABILITY-CONSTRAINED BEHAVIORAL STEERING

Behavioral Steering and Scale Matching. Given $\mathcal { D } _ { i , \mathrm { w h e n } } ^ { t }$ , the HOW operator constructs a targetdirected update via

$$
\mathcal { L } _ { \mathrm { s t e e r } } = - \frac { 1 } { | \mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } | } \sum _ { ( s , \cdot , \cdot ) \in \mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } } \log \pi _ { \theta } ( a _ { \mathrm { t a r g } } ^ { ( i ) } \mid s ) , \qquad d _ { i } ^ { t } = - \nabla _ { \theta } \mathcal { L } _ { \mathrm { s t e e r } } ( \theta ^ { t } ) , \qquad v _ { i } ^ { t } = g _ { i } ^ { t } + d _ { i } ^ { t } .\tag{4}
$$

followed by global $\ell _ { 2 }$ norm matching $\tilde { v } _ { i } ^ { t } = v _ { i } ^ { t } \cdot ( \lVert g _ { i } ^ { t } \rVert _ { 2 } / \operatorname* { m a x } ( \lVert v _ { i } ^ { t } \rVert _ { 2 } , \epsilon ) )$ . If $| \mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } | = 0$ , the client submits its nominal update $g _ { i } ^ { t }$ . Target actions are fixed per compromised subgroup, and evaluations use unit steering weight.

```latex
Algorithm 1 V-BSA: Local Malicious Update Synthesis at Round t
1: Input: Policy $\theta ^ { t } ,$ , rollouts $\mathcal { D } _ { i } ^ { t }$ , EMA stats $( \hat { \mu } _ { i } ^ { t } , \hat { \sigma } _ { i } ^ { t } )$ , target action $a _ { \mathrm { t a r g } } ^ { ( i ) } ,$ threshold $\tau ,$ radius $\lambda _ { \mathrm { b o x } } .$
2: Compute nominal local update $g _ { i } ^ { t } = \Delta \theta _ { i , \mathrm { c l e a n } } ^ { t }$ on local rollout $\mathcal { D } _ { i } ^ { t } .$
3: [WHEN] Filter sensitive states: ${ \mathcal { D } } _ { i , \mathrm { w h e n } } ^ { t }  \{ ( s , a , r ) \in { \mathcal { D } } _ { i } ^ { t } \mid H ( \pi _ { \theta ^ { t } } ( \cdot \mid s ) ) / \log \mid A \mid > \tau \}$
4: if $| \mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } | = 0$ then return $\Delta \theta _ { i } ^ { t , \mathrm { { m a l } } } \gets g _ { i } ^ { t }$ ▷ Submit nominal update if inactive
5: [HOW: Steering] Compute signed direction $d _ { i } ^ { t } \gets - \nabla _ { \theta } \mathcal { L } _ { \mathrm { s t e e r } } ( \theta ^ { t } )$ on $\mathcal { D } _ { i , \mathrm { w h e n } } ^ { t }$ using $a _ { \mathrm { t a r g } } ^ { ( i ) }$
6: Combine candidate perturbation vector: $v _ { i } ^ { t }  g _ { i } ^ { t } + d _ { i } ^ { t } .$
7: [HOW: Scale Matching] Rescale vector: $\tilde { v } _ { i } ^ { t }  v _ { i } ^ { t } \cdot ( \| g _ { i } ^ { t } \| _ { 2 } / \operatorname* { m a x } ( \| v _ { i } ^ { t } \| _ { 2 } , \epsilon ) )$
8: [HOW: Envelope Projection] Project update: $\Delta \boldsymbol { \theta } _ { i } ^ { t , \mathrm { m a l } } \gets \mathrm { P r o j } _ { B _ { i } ^ { t } } ( \tilde { v } _ { i } ^ { t } )$ via Eq. equation $5 .$
9: Output: Submit constrained malicious update $\Delta \theta _ { i } ^ { t , \mathrm { { m a l } } }$ to server-side aggregation.
```

Local Update Constraint. The client maintains EMA statistics $( \hat { \mu } _ { i } ^ { t } , \hat { \sigma } _ { i } ^ { t } )$ from historical benign updates to define the client-local empirical envelope:

$$
\mathcal { B } _ { i } ^ { t } = \prod _ { j = 1 } ^ { d } \left[ \hat { \mu } _ { i , j } ^ { t } - \lambda _ { \mathrm { b o x } } \hat { \sigma } _ { i , j } ^ { t } , \hat { \mu } _ { i , j } ^ { t } + \lambda _ { \mathrm { b o x } } \hat { \sigma } _ { i , j } ^ { t } \right] , \qquad \Delta \theta _ { i } ^ { t , \mathrm { m a l } } = \mathrm { P r o j } _ { \mathcal { B } _ { i } ^ { t } } ( \tilde { v } _ { i } ^ { t } ) ,\tag{5}
$$

where projection is coordinate-wise clipping. We set $\alpha _ { \mathrm { e m a } } = 0 . 1$ and $\lambda _ { \mathrm { b o x } } = 0 . 8 5$ . All quantities rely strictly on local observables; the envelope serves as a compatibility constraint rather than a formal defense-bypass guarantee.

Execution Characteristics. The entropy gate reuses policy logits already computed during rollout generation, introducing negligible forward overhead. Let $\bar { K } _ { s } = | \mathcal { D } _ { i , \mathrm { w h e n } } ^ { t } | \dot { = } \rho K$ ; the additional steering computation is strictly confined to the selected subset, with backward cost scaling as $\mathcal { O } ( K _ { s } C _ { \mathrm { b w d } } )$ , where $C _ { \mathrm { b w d } }$ denotes the per-state backward cost. Subsequent norm matching, EMA updates, and coordinate-wise projection require $\mathcal O ( d )$ parameter-space operations. Consequently, $\mathrm { V } -$ BSA incurs only local computational overhead proportional to the empirical intervention rate $\rho$ and parameter dimension $d ,$ without requiring Jacobian evaluation, peer-update exchange, or additional communication rounds.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Environments. We evaluate V-BSA on three discrete-action control benchmarks with diverse dynamical properties: CartPole $( | { \cal { A } } | = 2 )$ (Barto et al., 1983), Acrobot $( | { \cal A } | = 3 )$ (Sutton, 1995), and LunarLander $( | \mathcal { A } | = 4 )$ (Brockman et al., 2016). This selection spans both binary and multiaction settings. Detailed environment specifications are provided in Appendix C.1.

Federated Training and Aggregation. We consider a system of $N = 3 0$ clients, including $M =$ 10 compromised clients $( \beta \overset { \triangledown } { = } 1 / 3 )$ . For ensemble configurations, clients are partitioned into $K =$ 5 groups of six, with two compromised clients per group. All clients parameterize policies via a two-layer MLP with hidden dimension 64 and Tanh activations. We benchmark against both standard and Byzantine-robust aggregation rules: FedAvg (McMahan et al., 2017), Trimmed Mean and Coordinate-wise Median (Yin et al., 2018), and FedPG-BR (Fan et al., 2021); in ensemble settings, aggregation is performed independently within each group.

Attack Configuration and Evaluation. V-BSA uses a single, fixed set of attack coefficients across all tasks and aggregation rules: $\tau \ = \ 0 . 9 .$ , EMA coefficient $\alpha _ { \mathrm { e m a } } = 0 . 1 , \lambda _ { \mathrm { b o x } } = 0 . 8 5$ and unit steering weight, with no per-task or per-aggregator coefficient tuning. Target actions are assigned deterministically: compromised groups use the group index modulo $| { \cal A } |$ on the multi-action benchmarks, while binary-action CartPole uses a single shared target action 0. This avoids splitting the compromised population across the two available actions, which can induce cancellation during aggregation. Both assignment modes are fixed a priori and are not tuned to individual experimental configurations. Unless otherwise stated, each configuration is evaluated over three independent seeds, and we report the mean return over runs. Test performance is measured over 10 evaluation rollouts, and attack effectiveness is quantified by $D \ \triangleq \ J _ { \mathrm { c l e a n } } - J _ { \mathrm { a t t a c k e d } } ,$ , where $J _ { \mathrm { c l e a n } }$ and $J _ { \mathrm { { a t t a c k e d } } }$ denote the evaluation returns of the unpoisoned baseline and the attacked system, respectively. where $J _ { \mathrm { c l e a n } }$ and $J _ { \mathrm { { a t t a c k e d } } }$ denote the evaluation returns of the unpoisoned baseline and the attacked system, respectively. A larger positive D indicates greater performance degradation.

<table><tr><td rowspan="2">Aggregation Rule</td><td colspan="3">Ensemble  $( K = 5 , M = 1 0 )$ </td><td colspan="3">Non-Ensemble (N = 30, M = 10)</td></tr><tr><td>LunarLander</td><td>Acrobot</td><td>CartPole</td><td>LunarLander</td><td>Acrobot</td><td>CartPole</td></tr><tr><td>FedAvg</td><td>79.90</td><td>120.50</td><td>3.00</td><td>-57.19</td><td>351.17</td><td>1.63</td></tr><tr><td>FedPG-BR</td><td>218.75</td><td>194.80</td><td>110.63</td><td>288.01</td><td>70.63</td><td>114.27</td></tr><tr><td>Coordinate-wise Median</td><td>175.83</td><td>151.17</td><td>8.80</td><td>434.35</td><td>284.87</td><td>42.80</td></tr><tr><td>Trimmed Mean</td><td>173.96</td><td>323.97</td><td>66.40</td><td>193.38</td><td>312.97</td><td>5.87</td></tr></table>

Table 1: Mean degradation D across aggregations and tasks (higher is more damaging). Positive values indicate successful terminal policy degradation induced by V-BSA.

![](images/7f1cf314f75af1d519cdc6a7a03d535f49f7bc91ac2c5850d4a60bf11777b4fb.jpg)  
(a) FedAvg

![](images/4812f8a489be1b27402d26d872ca83a3d52ac59be2e093edd1496287ddc64d1d.jpg)  
(b) FedPG-BR

![](images/37efe6a38a7689507e8473113e6777b8100d99772673a364a9896f3b98734193.jpg)  
(c) Median

![](images/4235464d64c86be5fa7102451ce74980d0ef2674204062bfe9962710175255c3.jpg)  
(d) Trimmed Mean  
Figure 4: Evaluation dynamics on Ensemble-LunarLander $( K \ : = \ : 5 , M \ : = \ : 1 0 )$ . Returns of CLEAN vs. V-BSA across four aggregation rules, with shaded bands denoting standard deviation across random seeds.

## 5.2 EXPERIMENTAL RESULTS

Table 1 reports the mean terminal degradation D across tasks and aggregation rules, where positive values indicate lower attacked return. Figure 4 further visualizes the corresponding evaluation dynamics for ensemble LunarLander.

Main Results and Aggregation-Dependent Transfer. LunarLander provides clear empirical evidence of effective state-selective manipulation. Under ensemble training, V-BSA yields positive terminal degradation across all four aggregation rules, ranging from $D = 7 9 . 9 0$ (FedAvg) to $D = 2 1 8 . 7 5$ (FedPG-BR). As shown in Figure 4, nominal and attacked evaluation curves progressively separate across communication rounds, with degradation persisting under FedPG-BR filtering.

Cross-aggregation comparison also reveals clear operational boundaries. Under non-ensemble training, FedAvg exhibits a sign reversal $( D = - 5 7 . 1 { \bar { 9 } } )$ , where attacked runs attain higher average terminal return than the nominal baseline. We interpret this as an empirical transfer boundary, where the manipulated updates fail to sustain policy degradation. Acrobot exhibits a different pattern, with substantial degradation across aggregation rules, reaching D = 351.17 under non-ensemble FedAvg and $D = 3 2 3 . 9 7$ under ensemble Trimmed Mean. Together, these results illustrate that downstream degradation depends jointly on task dynamics and aggregation.

Binary-Action Cancellation Boundary: CartPole. CartPole exposes a structural boundary arising from binary target-steering geometry. Ensemble FedPG-BR yields substantial degradation $( D = 1 1 0 . 6 3 )$ , whereas coordinate-wise Median shows little average effect $( D = 8 . 8 0 )$ . Trimmed Mean exhibits an intermediate mean effect $( D = 6 6 . 4 0 )$ , concentrated on $d _ { 1 } ( D = 1 8 4 . 3 )$ versus $d _ { 0 }$ (D = 0) and $d _ { 2 } \left( D = 1 4 . 9 \right)$

(a) Entropy vs. behavioral susceptibility (attacked)  
![](images/676d35066a320fa05160f8125d0b9f53402d4b72e35a5cd6afdc680fe2fbec23.jpg)

(b) Gated vs. non-gated susceptibility  
![](images/960e1508fa818a2479f1a3b0b460dc4e544843fdd4a6b951d279c45ea2ec6917.jpg)  
Figure 5: Mechanistic audit of WHEN selection (LunarLander, epoch 261). (a) Normalized policy entropy $h ( s )$ versus behavioral susceptibility $L ( s )$ . (b) Susceptibility distributions for nongated versus gated states across five client groups.

Budget-matched random (ρ̄= 7.2%) Entropy gate (ρ̄= 5.7%)  
![](images/7d310ebe5c82428dc1bec44ac11793252d2e08fe014672a9287c6928794032aa.jpg)  
Figure 6: Timing ablation under matched budgets (LunarLander × FedPG-BR). Fixing all HOW parameters, entropy gating achieves 1.89× higher mean degradation than budget-matched random selection across evaluated dynamics.

For binary actions, the opposing steering directions satisfy

$$
d ( a _ { 0 } ; s ) + d ( a _ { 1 } ; s ) = \bigl ( 1 - 2 \pi _ { \theta } ( a _ { 0 } \mid s ) \bigr ) \Delta ( s ) ,\tag{6}
$$

where $\Delta ( s ) = \nabla _ { \boldsymbol { \theta } } ( z _ { 0 } - z _ { 1 } )$ . Thus, near the maximum-entropy boundary $\pi _ { \theta } ( a _ { 0 } \mid s ) = 1 / 2$ , the two target-specific directions cancel. Under alternating target assignment, coordinate-wise order statistics can therefore suppress opposing perturbations, whereas the filtering behavior of FedPG-BR differs because it evaluates updates relative to a robust reference. This geometry explains the observed aggregation divergence; further derivations appear in Appendix A.6.

Controlled Comparison of Intervention Timing. To isolate the effect of intervention timing (when) under fixed update construction (how), Figure 6 compares entropy gating with a budgetcontrolled uniform random selector on LunarLander under FedPG-BR. The two conditions use the same HOW components, random seeds, and evaluation dynamics.

Across the three dynamics, entropy gating yields a higher mean degradation $( \bar { D } = 2 1 8 . 7 5 $ , median 148.17) than random selection $( \bar { D } = 1 1 5 . \bar { 9 2 }$ , median 90.64), corresponding to a 1.89× larger mean degradation. The effect is not uniform across dynamics: entropy gating yields 137.48 versus 59.25 on $d _ { 1 }$ and 370.59 versus 90.64 on $d _ { 2 } .$ , whereas random selection yields higher degradation on $d _ { 0 }$ (197.86 versus 148.17). The realized intervention rates are 5.7% for entropy gating and 7.2% for random selection on average. These controlled results show that, under fixed update construction and a controlled intervention budget, changing the selected trajectory states can materially alter downstream attack effectiveness, while the magnitude and direction of the effect remain dynamicsdependent.

Mechanistic Audit of WHEN Selection. Figure 5 examines the behavioral susceptibility of visited states in LunarLander. We quantify local susceptibility $L ( s )$ using the perturbation-based behavioral metric defined in Appendix B.

![](images/e4667530980d1a8588927c352fd84c569c3dec3aa6dbcdbcaf8614cc3c066605.jpg)  
Figure 7: Temporal countermeasures against V-BSA on LunarLander. Terminal degradation D across dynamics $d _ { 0 } , d _ { 1 } , d _ { 2 }$ under no defense, the epoch auditor, and the temporal accumulator. The dashed line marks the preregistered resilience threshold (3σ = 49.7).

As shown in Figure ${ 5 \mathrm { a } } ,$ entropy-selected states exhibit a pronounced shift toward higher susceptibility than non-selected states. Across rollout states pooled over the five client groups (Figure 5b), median susceptibility increases from 0.24 to 1.62 (6.6×) under nominal policies and from 0.33 to 1.74 (5.3×) under attacked policies. Because a similar separation is already observed under nominal execution, the effect is not solely induced by the attack. These results support policy entropy as a practical output-only proxy for local behavioral susceptibility in the evaluated LunarLander setting, while remaining consistent with the theoretical limitation that multi-action entropy does not generally induce exact Fisher-trace rankings.

## 5.3 TEMPORAL COUNTERMEASURES AND DEFENSE OUTLOOK

A natural question is whether temporal parameter-space auditing can suppress state-selective manipulation that remains effective against single-round robust filtering. We therefore preregistered and tested two temporal countermeasures on LunarLander: an epoch-level direction auditor that scores each client update against a per-round coordinate-median reference, exposing the attack’ directional-persistence signature, and a temporal accumulator that aggregates rejection evidence across historical updates.

Defense Attempts. The epoch-level auditor separates Byzantine from honest clients by direction scores (0.586 vs. 0.215, 2.7×), yet rejects only about one-third of Byzantine updates per round (31.5%–33.0%), leaving sufficient accepted updates to sustain degradation. It reduces D to 49.44 on $d _ { 0 }$ and 89.61 on $d _ { 2 } .$ , but reaches 153.59 on $d _ { 1 } ;$ the latter nominally exceeds the no-defense anchor (137.48), although the signed comparison is confounded by cross-run variance of the corresponding anchor. Its median degradation of 89.61 therefore exceeds the 3σ criterion. The temporal accumulator further reduces median degradation to 38.20 (with the negative $d _ { 0 }$ value reflecting cross-run anchor variance), but its maximum degradation remains 82.26 on $d _ { 2 } .$ , violating the preregistered dual condition that both median and maximum remain below 49.7; it also fails the clean-training gate with relative return 0.886 < 0.9.

Defense Outlook. These results suggest that parameter-space temporal consistency alone is insufficient when malicious updates are locally bounded yet persistently targeted. The accumulator can substantially increase detection coverage, separating Byzantine signatures by 5.5× and detecting 97.5%–98.5% through union rejection, but this comes with 22.7%–31.8% false-positive rejection on honest updates and still leaves residual attack influence. This detection–suppression gap motivates a more direct defense direction: complement update-space auditing with rollout-level behavioral monitoring of state-conditioned intervention patterns, repeated-state policy changes, and cross-round shifts in state occupancy. Such defenses would target the trajectory-level footprint ex ploited by V-BSA, rather than relying solely on parameter-space persistence.

## 6 CONCLUSION

We formalize Byzantine FRL by decoupling intervention timing (WHEN) from update construction (HOW). Controlled comparisons demonstrate that varying selected trajectory states under fixed

HOW substantially alters policy degradation, establishing intervention timing as an orthogonal attack dimension. Leveraging this insight, V-BSA uses local policy uncertainty to concentrate behavioral steering onto sparse, state-selective interventions, bypassing single-round robust filters and unweighted majority voting ensembles. While effective in discrete control, exploratory evaluations identify operational boundaries in continuous settings due to variance-head saturation (Appendix E). Crucially, our defense audits show that cumulative temporal auditing and robust intra-group estimators maintain substantial resilience, motivating future trajectory-level monitoring of sequential state visitation alongside parameter-space checks to secure federated sequential decision-making.

## REFERENCES

Andrew G Barto, Richard S Sutton, and Charles W Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics, (5):834–846, 1983.

Gilad Baruch, Moran Baruch, and Yoav Goldberg. A little is enough: Circumventing defenses for distributed learning. Advances in Neural Information Processing Systems, 32, 2019.

Peva Blanchard, El Mahdi El Mhamdi, Rachid Guerraoui, and Julien Stainer. Machine learning with adversaries: Byzantine tolerant gradient descent. Advances in neural information processing systems, 30, 2017.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2):1–25, 2017.

Jing Cui, Yufei Han, Yuzhe Ma, Jianbin Jiao, and Junge Zhang. Badrl: Sparse targeted backdoor attack against reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 11687–11694, 2024.

El-Mahdi El-Mhamdi, Rachid Guerraoui, and Sebastien Rouault. The hidden vulnerability of dis-´ tributed learning in byzantium. In International conference on machine learning, pp. 3521–3530. PMLR, 2018.

Xiaofeng Fan, Yining Ma, Zhongxiang Dai, Wei Jing, Cheston Tan, and Bryan Kian Hsiang Low. Fault-tolerant federated reinforcement learning with theoretical guarantee. Advances in neural information processing systems, 34:1007–1021, 2021.

Minghong Fang, Xiaoyu Cao, Jinyuan Jia, and Neil Gong. Local model poisoning attacks to Byzantine-Robust federated learning. In 29th USENIX security symposium (USENIX Security 20), pp. 1605–1622, 2020.

Minghong Fang, Seyedsina Nabavirazavi, Zhuqing Liu, Wei Sun, Sundararaja Sitharama Iyengar, and Haibo Yang. Do we really need to design new byzantine-robust aggregation rules? arXiv preprint arXiv:2501.17381, 2025a.

Minghong Fang, Xilong Wang, and Neil Zhenqiang Gong. Provably robust federated reinforcement learning. In Proceedings ofthe ACM on Web Conference 2025, pp. 896–909, 2025b.

Clement Fung, Chris JM Yoon, and Ivan Beschastnikh. Mitigating sybils in federated learning poisoning. arXiv preprint arXiv:1808.04866, 2018.

Hao Jin, Yang Peng, Wenhao Yang, Shusen Wang, and Zhihua Zhang. Federated reinforcement learning with environment heterogeneity. In International conference on artificial intelligence and statistics, pp. 18–37. PMLR, 2022.

Sham M Kakade. A natural policy gradient. In T. Dietterich, S. Becker, and Z. Ghahramani (eds.), Advances in Neural Information Processing Systems, volume 14. MIT Press, 2001. URL https://proceedings.neurips.cc/paper\_files/paper/2001/ file/4b86abe48d358ecf194c56c69108433e-Paper.pdf.

Sajad Khodadadian, Pranay Sharma, Gauri Joshi, and Siva Theja Maguluri. Federated reinforcement learning: Linear speedup under markovian sampling. In International conference on machine learning, pp. 10997–11057. PMLR, 2022.

Yen-Chen Lin, Zhang-Wei Hong, Yuan-Hong Liao, Meng-Li Shih, Ming-Yu Liu, and Min Sun. Tactics of adversarial attack on deep reinforcement learning agents. arXiv preprint arXiv:1703.06748, 2017.

Xiaoting Lyu, Yufei Han, Wei Wang, Jingkai Liu, Bin Wang, Kai Chen, Yidong Li, Jiqiang Liu, and Xiangliang Zhang. CoBA: Collusive Backdoor Attacks With Optimized Trigger to Federated Learning. IEEE Transactions on Dependable and Secure Computing, 22(02):1506–1518, 2025. doi: 10.1109/TDSC.2024.3445637.

Evelyn Ma, Praneet Rathi, and S Rasoul Etesami. Local environment poisoning attacks on federated reinforcement learning. arXiv preprint arXiv:2303.02725, 2023.

Evelyn Ma, S Rasoul Etesami, and Praneet Rathi. Reward poisoning on federated reinforcement learning. Transactions on Machine Learning Research, 2024.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273–1282. Pmlr, 2017.

Shashank Rajput, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos. Detox: A redundancy-based framework for faster and more robust gradient aggregation. Advances in neural information processing systems, 32, 2019.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pp. 1889–1897. Pmlr, 2015.

Virat Shejwalkar and Amir Houmansadr. Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning. In Ndss, 2021.

Richard S Sutton. Generalization in reinforcement learning: Successful examples using sparse coarse coding. Advances in neural information processing systems, 8, 1995.

Yueqi Xie, Minghong Fang, and Neil Zhenqiang Gong. Model poisoning attacks to federated learning via multi-round consistency. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15454–15463. IEEE, 2025.

Gang Yan, Hao Wang, Xu Yuan, and Jian Li. Enhancing model poisoning attacks to byzantinerobust federated learning via critical learning periods. In Proceedings of the 27th International Symposium on Research in Attacks, Intrusions and Defenses, pp. 496–512, 2024.

Pei Ye, Yuqing Li, Kun He, Haoran Wang, Ruiying Du, and Wei Wang. Less is more: Persistent lowfrequency backdoor injection in federated learning. In IEEE INFOCOM 2026-IEEE Conference on Computer Communications, pp. 1–10. IEEE, 2026.

Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. In International conference on machine learning, pp. 5650–5659. Pmlr, 2018.

Chengze Zhang, Jia Wang, and Xianghui Cao. Cpp: A stealthy attack through critical path poisoning in federated reinforcement learning. In 2025 International Conference on Cyber Resilience and Endogenous Safety & Security (CRESS), pp. 372–379. IEEE, 2025a.

Xuezhou Zhang, Yuzhe Ma, Adish Singla, and Xiaojin Zhu. Adaptive reward-poisoning attacks against reinforcement learning. In International Conference on Machine Learning, pp. 11225– 11234. PMLR, 2020.

Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, and Yong Cui. State-aware perturbation optimization for robust deep reinforcement learning. IEEE Transactions on Mobile Computing, 2025b.

## A THEORETICAL PROOFS AND ANALYSIS

This appendix provides the complete derivations underlying the WHEN-side analysis in Section 4.2. We first establish the local Fisher characterization of behavioral susceptibility, then derive the idealized budgeted selection rule, the entropy–susceptibility ranking result for binary-action policies, and the failure of exact ranking equivalence in multi-action settings. We finally provide a separate analytical explanation for the special target-assignment behavior on CartPole. The latter is intended as an analysis of an observed design boundary rather than as a second main theoretical contribution.

## A.1 LOCAL KL EXPANSION AND FISHER DECOMPOSITION

For a fixed state s, define

$$
f ( \delta \theta ) \triangleq \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } ( \cdot  { | \begin{array} { l } { s } \end{array}  } \end{array} \|  { \begin{array} { l } { \pi _ { \theta + \delta \theta } ( \cdot  { | \begin{array} { l } { s } \end{array} ) } } ) } .\tag{7}
$$

Because the action space is finite and the categorical policy has strictly positive action probabilities in the interior of the softmax simplex,

$$
f ( \delta \theta ) = \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \log \pi _ { \theta } ( a \mid s ) - \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \log \pi _ { \theta + \delta \theta } ( a \mid s ) .\tag{8}
$$

Clearly, $f ( { \bf 0 } ) = 0$ . Its gradient with respect to $\delta \theta$ is

$$
\nabla _ { \delta \theta } f ( \delta \theta ) = - \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \nabla _ { \theta } \log \pi _ { \theta + \delta \theta } ( a \mid s ) .\tag{9}
$$

At δθ = 0,

$$
\nabla _ { \delta \theta } f ( \mathbf { 0 } ) = - \mathbb { E } _ { a \sim \pi _ { \theta } ( \cdot \vert s ) } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \right] = \mathbf { 0 } ,\tag{10}
$$

by the standard score-function identity. Differentiating a second time yields the Hessian at $\delta \boldsymbol { \theta } = \mathbf { 0 }$

$$
\begin{array} { l } { { \nabla _ { \theta } ^ { 2 } \theta \rho ( \mathbf { 0 } ) = - \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \nabla _ { \theta } ^ { 2 } \log \pi _ { \theta } ( a \mid s ) } } \\ { { = - \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \left( \frac { \nabla _ { \theta } ^ { 2 } \pi _ { \theta } ( a \mid s ) } { \pi _ { \theta } ( a \mid s ) } - \frac { \nabla _ { \theta } \pi _ { \theta } ( a \mid s ) \nabla _ { \theta } \pi _ { \theta } ( a \mid s ) ^ { \top } } { \pi _ { \theta } ( a \mid s ) ^ { 2 } } \right) } } \\ { { = - \nabla _ { \theta } ^ { 2 } \left( \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \right) + \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) ^ { \top } } } \\ { { = \mathbb { E } _ { a \sim \pi _ { \theta } ( \cdot \mid s ) } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) \nabla _ { \theta } \log \pi _ { \theta } ( a \mid s ) ^ { \top } \right] \triangleq F ( s ) , } } \end{array}\tag{11}
$$

where $F ( s )$ is the state-conditional Fisher information matrix. Assuming that $\pi _ { \theta } ( \cdot \mid s )$ is $C ^ { 3 }$ in θ with locally bounded third derivatives in a neighborhood of θ, Taylor’s theorem gives

$$
\mathbb { D } _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot \ | \ s ) \ \| \ \pi _ { \theta + \delta \theta } ( \cdot \ | \ s ) \right) = \frac { 1 } { 2 } \delta \theta ^ { \top } F ( s ) \delta \theta + \mathcal { O } ( \| \delta \theta \| _ { 2 } ^ { 3 } ) .\tag{12}
$$

Fisher decomposition for softmax policies. Let $z _ { \theta } ( s ) \in \mathbb { R } ^ { | \mathcal { A } | }$ denote the logit vector, $p = \pi _ { \theta } ( \cdot |$ s), and $J _ { z } ( s ) \triangleq \nabla _ { \theta } z _ { \theta } ( s ) \in \mathbb { R } ^ { | A | \times d }$ . For a categorical softmax policy,

$$
\nabla _ { z } \log \pi _ { \theta } ( a \mid s ) = e _ { a } - p ,\tag{13}
$$

where $e _ { a }$ is the canonical basis vector corresponding to action a. By the chain rule, ∇<sub>θ</sub> log π<sub>θ</sub>(a | $\boldsymbol { s } ) = J _ { z } ( \boldsymbol { s } ) ^ { \intercal } ( \boldsymbol { e } _ { a } - \boldsymbol { p } )$ . Therefore,

$$
\begin{array} { l } { { \displaystyle F ( s ) = \sum _ { a \in { \cal A } } p _ { a } J _ { z } ( s ) ^ { \top } ( e _ { a } - p ) ( e _ { a } - p ) ^ { \top } J _ { z } ( s ) } } \\ { { \displaystyle ~ = J _ { z } ( s ) ^ { \top } \left( \mathrm { D i a g } ( p ) - p p ^ { \top } \right) J _ { z } ( s ) . } } \end{array}\tag{14}
$$

Defining $M ( p ) \triangleq { \mathrm { D i a g } } ( p ) - p p ^ { \intercal }$ , we obtain

$$
F ( s ) = J _ { z } ( s ) ^ { \top } M ( p ) J _ { z } ( s ) .\tag{15}
$$

## A.2 SUSCEPTIBILITY SURROGATES

From Eq. equation 12, the local worst-case behavioral susceptibility under an $\ell _ { 2 }$ perturbation budget ϵ is

$$
\chi _ { \epsilon } ( s ) \triangleq \operatorname* { s u p } _ { \| \delta \theta \| _ { 2 } \leq \epsilon } \mathbb { D } _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot  { | } s )  { \| } \pi _ { \theta + \delta \theta } ( \cdot  { | } s ) \right) = \frac { \epsilon ^ { 2 } } { 2 } \lambda _ { \operatorname* { m a x } } ( F ( s ) ) + \mathcal { O } ( \epsilon ^ { 3 } ) .\tag{16}
$$

This follows because $F ( s ) \succeq 0$ and the maximum of the quadratic form $\delta \theta ^ { \top } F ( s ) \delta \theta$ over the Euclidean ball is attained along an eigenvector associated with $\lambda _ { \operatorname* { m a x } } ( F ( s ) )$ .

Direct computation of $\lambda _ { \operatorname* { m a x } } ( F ( s ) )$ or $\operatorname { T r } ( F ( s ) )$ ) requires access to parameter-space sensitivity. For a direction-agnostic second-order surrogate, let $\mathrm { \overrightarrow { \delta } } \theta \sim \mathrm { \overrightarrow { \mathcal { N } } } ( \mathbf { 0 } , \sigma ^ { 2 } I _ { d } )$ . The quadratic term in the expected local KL divergence becomes

$$
\mathbb { E } _ { \delta \theta } \left[ { \frac { 1 } { 2 } } \delta \theta ^ { \top } F ( s ) \delta \theta \right] = { \frac { 1 } { 2 } } \operatorname { T r } \left( F ( s ) \mathbb { E } [ \delta \theta \delta \theta ^ { \top } ] \right) = { \frac { \sigma ^ { 2 } } { 2 } } \operatorname { T r } ( F ( s ) ) .\tag{17}
$$

Accordingly, we define

$$
\bar { \chi } _ { \sigma } ^ { ( 2 ) } ( s ) \triangleq \frac { \sigma ^ { 2 } } { 2 } \operatorname { T r } ( F ( s ) ) .\tag{18}
$$

Under the local smoothness assumptions of $\operatorname { E q . }$ equation 12, the expected KL divergence differs from Eq. equation 18 only through higher-order terms in the perturbation scale.

The two quantities in Eqs. equation 16 and equation 18 serve distinct roles: $\lambda _ { \operatorname* { m a x } } ( F ( s ) )$ ) describes the single most susceptible parameter direction, whereas $\operatorname { T r } ( F ( s ) )$ aggregates sensitivity over an isotropic ensemble of directions. Because the attacker cannot anticipate which parameter directions will be emphasized by an unknown aggregation rule, we use the latter as a direction-agnostic local surrogate.

## A.3 IDEALIZED BUDGETED SELECTION

Consider a rollout $\tau = ( s _ { 1 } , \dots , s _ { K } )$ and a hypothetical intra-round intervention budget $B \leq K$ . An oracle attacker that seeks to maximize the additive local susceptibility surrogate solves

$$
\operatorname* { m a x } _ { g \in \{ 0 , 1 \} ^ { K } } \sum _ { k = 1 } ^ { K } g _ { k } \bar { \chi } _ { \sigma } ^ { ( 2 ) } ( s _ { k } ) \quad \mathrm { s . t . } \quad \sum _ { k = 1 } ^ { K } g _ { k } \leq B .\tag{19}
$$

Proposition 1 (Budgeted Selection Optimality). For the oracle problem in $E q .$ equation $^ { \mathit { 1 9 , } }$ an optimal selector consists ofthe B states having the largest values of $\mathrm { T r } ( F ( s _ { k } ) )$ .

Proof. Because $\sigma ^ { 2 } / 2$ is common to all states, maximizing $\bar { \chi } _ { \sigma } ^ { ( 2 ) } ( s _ { k } )$ is equivalent to maximizing $\mathrm { T r } ( F ( s _ { k } ) )$ ). Suppose a feasible solution selects state $s _ { j }$ but does not select $s _ { i } ,$ where ${ \mathrm { T r } } ( F ( s _ { i } ) ) >$ $\mathrm { T r } ( F ( s _ { j } ) )$ . Replacing $s _ { j }$ with $s _ { i }$ preserves the cardinality constraint and strictly increases the objective. Hence, no optimal solution can contain a lower-ranked state while excluding a higher-ranked one. Selecting the top-B states is therefore optimal. □

This result is an oracle characterization of the desired local ranking under a fixed budget. It does not describe the online procedure used by V-BSA, nor does it imply optimal degradation of longhorizon expected return. The latter depends on state-visitation shifts, environmental transition dynamics, future policy updates, and the interaction between successive interventions.

## A.4 ENTROPY–SUSCEPTIBILITY RANKING UNDER CENTERED ISOTROPY

To obtain an online state-selection rule without explicitly computing the logit Jacobian, consider the centered logit subspace. Define

$$
P _ { \perp } \triangleq I _ { | A | } - \frac { 1 } { | A | } \mathbf { 1 1 } ^ { \top } .\tag{20}
$$

Suppose

$$
J _ { z } ( s ) J _ { z } ( s ) ^ { \top } = c P _ { \bot } ,\tag{21}
$$

where $c > 0$ is a state-independent constant. Since $M ( p ) \mathbf { 1 } = \mathbf { 0 }$ , we have

$$
\mathrm { T r } ( F ( s ) ) = \mathrm { T r } \left( M ( p ) J _ { z } ( s ) J _ { z } ( s ) ^ { \top } \right) = c \mathrm { T r } \left( M ( p ) P _ { \bot } \right) = c \left( 1 - \| p \| _ { 2 } ^ { 2 } \right) .\tag{22}
$$

Theorem 2 (Restatement of Theorem 1). For a binary-action policy $p = ( q , 1 - q )$ with $q \in ( 0 , 1 )$ if the centered isotropy condition in Eq. equation 21 holds with state-independent $c > 0 ,$ , then Shannon entropy ${ \cal H } ( \bar { p } ) = - q \log q - ( \bar { 1 } - \bar { q } ) \log ( 1 - q )$ and the Fisher-trace surrogate $\operatorname { T r } ( F ( s ) )$ induce identical state rankings:

$$
H ( p _ { 1 } ) > H ( p _ { 2 } ) \iff \operatorname { T r } ( F ( s _ { 1 } ) ) > \operatorname { T r } ( F ( s _ { 2 } ) ) .\tag{23}
$$

Proof. For $| { \mathcal { A } } | = 2$

$$
P _ { \perp } = I _ { 2 } - { \frac { 1 } { 2 } } \mathbf { 1 1 } ^ { \top } = \left[ { \begin{array} { c c } { 1 / 2 } & { - 1 / 2 } \\ { - 1 / 2 } & { 1 / 2 } \end{array} } \right] .\tag{24}
$$

Moreover,

$$
M ( p ) = \left[ \begin{array} { c c } { { q ( 1 - q ) } } & { { - q ( 1 - q ) } } \\ { { - q ( 1 - q ) } } & { { q ( 1 - q ) } } \end{array} \right] = 2 q ( 1 - q ) P _ { \perp } .\tag{25}
$$

Using Eq. equation 21,

$$
\mathrm { T r } ( F ( s ) ) = \mathrm { T r } \left( M ( p ) J _ { z } ( s ) J _ { z } ( s ) ^ { \top } \right) = 2 c q ( 1 - q ) \mathrm { T r } ( P _ { \perp } ^ { 2 } ) = 2 c q ( 1 - q ) ,\tag{26}
$$

because $P _ { \mathrm { ~ l ~ } } ^ { 2 } = P _ { \bot }$ and $\mathrm { T r } ( P _ { \perp } ) = 1$ . Now define $\psi ( q ) \triangleq 2 c q ( 1 - q )$ and $\phi ( q ) \triangleq H ( q )$ . Both functions are symmetric around $q = 1 / 2$ . On $q \in ( 0 , 1 / 2 )$

$$
\psi ^ { \prime } ( q ) = 2 c ( 1 - 2 q ) > 0 ,\tag{27}
$$

$$
\phi ^ { \prime } ( q ) = \log \left( \frac { 1 - q } { q } \right) > 0 .\tag{28}
$$

Hence both functions are strictly increasing on $( 0 , 1 / 2 ]$ and strictly decreasing on $[ 1 / 2 , 1 )$ ). Equivalently, both are strictly decreasing functions of $\left| q - 1 / 2 \right|$ . Therefore,

$$
\begin{array} { r l r } {  { H ( p _ { 1 } ) > H ( p _ { 2 } ) \iff | q _ { 1 } - \frac { 1 } { 2 } | < | q _ { 2 } - \frac { 1 } { 2 } | } } \\ & { } & { \iff 2 c q _ { 1 } ( 1 - q _ { 1 } ) > 2 c q _ { 2 } ( 1 - q _ { 2 } ) } \\ & { } & { \iff \operatorname { T r } ( F ( s _ { 1 } ) ) > \operatorname { T r } ( F ( s _ { 2 } ) ) . } \end{array}\tag{29}
$$

From ranking to online gating. Theorem 1 bridges the idealized ranking formulation to an online state-selection rule. Under its binary-action conditions, entropy and the Fisher-trace surrogate induce the identical ordering. Consequently, an entropy level set

$$
S _ { h } \triangleq { \Bigl \{ } s {  { \left| { \cal H } ( \pi _ { \theta } ( \cdot \mid s ) ) > h \right\} } }\tag{30}
$$

corresponds to a top-ranked subset under the susceptibility surrogate, with an induced intervention budget $B ( h ) = | S _ { h } |$ . Thus, thresholding entropy realizes the susceptibility ranking without explicitly evaluating ${ \mathrm { T r } } ( F ( s ) )$ ), buffering the rollout for sorting, or performing Jacobian backpropagation.

For multi-action policies, exact ranking equivalence does not generally hold. V-BSA therefore uses normalized entropy as a practical output-only proxy:

$$
\mathcal { G } _ { \mathrm { w h e n } } ( s ; \theta ^ { t } ) = \mathbf { 1 } \left[ \frac { H ( \pi _ { \theta ^ { t } } ( \cdot \mid s ) ) } { \log \mid A \mid } > \tau \right] ,\tag{31}
$$

where $\tau = 0 . 9$ is fixed a priori and is neither swept nor calibrated to match a prescribed intervention budget. The resulting intervention frequency is therefore induced endogenously by the evolving policy and trajectory distribution.

## A.5 ANALYTICAL FAILURE OF EXACT ENTROPY RANKING IN MULTI-ACTION REGIMES

For $| { \mathcal { A } } | \geq 3 ,$ , Eq. equation 22 becomes

$$
\mathrm { T r } ( F ( s ) ) = c \left( 1 - \| p \| _ { 2 } ^ { 2 } \right) .\tag{32}
$$

The quantity $1 - \| p \| _ { 2 } ^ { 2 }$ is the Gini impurity (equivalently Tsallis-2 entropy up to normalization), which does not in general induce the same ordering as Shannon entropy. Consider a four-action policy and two states with

$$
p _ { A } = ( 0 . 5 5 , \ 0 . 2 0 , \ 0 . 2 0 , \ 0 . 0 5 ) ,\tag{33}
$$

$$
p _ { B } = ( 0 . 4 0 , 0 . 4 0 , 0 . 1 9 , 0 . 0 1 ) .\tag{34}
$$

For $p _ { A }$ ,

$$
\| p _ { A } \| _ { 2 } ^ { 2 } = 0 . 5 5 ^ { 2 } + 0 . 2 0 ^ { 2 } + 0 . 2 0 ^ { 2 } + 0 . 0 5 ^ { 2 } = 0 . 3 8 5 0 \implies 1 - \| p _ { A } \| _ { 2 } ^ { 2 } = 0 . 6 1 5 0 .\tag{35}
$$

For $p _ { B }$

$$
\| p _ { B } \| _ { 2 } ^ { 2 } = 0 . 4 0 ^ { 2 } + 0 . 4 0 ^ { 2 } + 0 . 1 9 ^ { 2 } + 0 . 0 1 ^ { 2 } = 0 . 3 5 6 2 \implies 1 - \| p _ { B } \| _ { 2 } ^ { 2 } = 0 . 6 4 3 8 .\tag{36}
$$

Hence, under the Fisher trace surrogate, $\mathrm { T r } ( F ( s _ { B } ) ) > \mathrm { T r } ( F ( s _ { A } ) )$ . In contrast, the Shannon entropies evaluate to:

$$
H ( p _ { A } ) = - 0 . 5 5 \ln { 0 . 5 5 } - 2 ( 0 . 2 0 \ln { 0 . 2 0 } ) - 0 . 0 5 \ln { 0 . 0 5 } \approx 1 . 1 2 2 4 ,\tag{37}
$$

$$
H ( p _ { B } ) = - 2 ( 0 . 4 0 \ln { 0 . 4 0 } ) - 0 . 1 9 \ln { 0 . 1 9 } - 0 . 0 1 \ln { 0 . 0 1 } \approx 1 . 0 9 4 6 .\tag{38}
$$

Therefore,

$$
H ( p _ { A } ) > H ( p _ { B } ) \quad { \mathrm { w h i l e } } \quad \operatorname { T r } ( F ( s _ { A } ) ) < \operatorname { T r } ( F ( s _ { B } ) ) .\tag{39}
$$

This explicit ranking reversal shows that, for multi-action policies, Shannon entropy cannot in general be interpreted as an exact estimator of Fisher-trace susceptibility. The entropy gate in V-BSA should consequently be understood as a lightweight practical proxy rather than an exact susceptibility oracle.

## A.6 AGGREGATION ANALYSIS ON CARTPOLE: BINARY STEERING GEOMETRY AND TARGET CANCELLATION

This section analyzes the CartPole-specific behavior of V-BSA through the geometry of binary target steering. It is intended as a mechanistic explanation of the observed aggregation-dependent phenomena, rather than as a second main theoretical contribution. The formal theoretical analysis of the paper remains the WHEN-side susceptibility characterization in Section 4.2 and Appendix A.

Setup and target assignment. For the signed update convention of Eq. equation 4, let

$$
d ( a ; s ) \triangleq - \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { s t e e r } } ( s , a ) = \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( a \mid s ) ,\tag{40}
$$

so that the malicious update before the local viability constraint is

$$
v = g + d .\tag{41}
$$

For CartPole, the main configuration uses the same deterministic modulo target rule as the multiaction environments:

$$
a ^ { \left( g \right) } = g { \mathrm { ~ m o d ~ } } 2 .\tag{42}
$$

With five compromised groups, this yields the target pattern $\{ 0 , 1 , 0 , 1 , 0 \}$ , i.e., $n _ { 0 } = 3$ groups target action 0 and $n _ { 1 } = 2$ groups target action 1. Each group contains $\nu = 2$ compromised clients. The timing gate is

$$
S _ { \tau } \triangleq \left\{ s \Big \vert \frac { H ( \pi _ { \theta } ( \cdot  { | } s ) ) } { \ln 2 } > \tau \right\} , \qquad \tau = 0 . 9 .\tag{43}
$$

The resulting malicious updates are globally norm-matched to the corresponding nominal local updates and then coordinate-wise clamped to a client-local empirical update envelope with $\lambda _ { \mathrm { b o x } } = 0 . 8 5$ . The five group policies remain separate throughout training and are combined only at decision time by majority voting.

Binary target geometry. Let

$$
\Delta ( s ) \triangleq \nabla _ { \boldsymbol { \theta } } z _ { 0 } ( s ) - \nabla _ { \boldsymbol { \theta } } z _ { 1 } ( s ) ,\tag{44}
$$

where $z _ { 0 } ( s )$ and $z _ { 1 } ( s )$ are the two action logits. For $\pi _ { \theta } ( \cdot \mid s ) = ( \pi , 1 - \pi )$ , the score identities for a binary softmax policy give

$$
d ( 0 ; s ) = ( 1 - \pi ) \Delta ( s ) ,
$$

$$
d ( 1 ; s ) = - \pi \Delta ( s ) ,\tag{45}
$$

and hence

$$
d ( 0 ; s ) + d ( 1 ; s ) = ( 1 - 2 \pi ) \Delta ( s ) .\tag{46}
$$

Thus, for every fixed state, the two target-specific steering directions are exactly collinear and oppositely directed. They cancel exactly at the maximum-entropy point $\pi = 1 / 2$

The entropy gate also restricts the policy probability to a neighborhood of $1 / 2$ . Let $q _ { \tau } \in ( 0 , 1 / 2 )$ satisfy $H ( \bar { q } _ { \tau } ) = 0 . 9 \ln { 2 }$ . Numerically, $q _ { \tau } \approx 0 . 3 1 6$ . Therefore,

$$
\frac { H ( \pi ) } { \ln 2 } > 0 . 9 \implies \pi \in ( q _ { \tau } , 1 - q _ { \tau } ) ,\tag{47}
$$

and consequently

$$
| 1 - 2 \pi | < 1 - 2 q _ { \tau } \approx 0 . 3 6 8 .\tag{48}
$$

Within the gate set,

$$
\| d ( 0 ; s ) + d ( 1 ; s ) \| _ { 2 } < 0 . 3 6 8 \| \Delta ( s ) \| _ { 2 } ,\tag{49}
$$

whereas each individual target direction satisfies

$$
\| d ( a ; s ) \| _ { 2 } \geq 0 . 3 1 6 \| \Delta ( s ) \| _ { 2 } , \qquad a \in \{ 0 , 1 \} .\tag{50}
$$

Hence, the entropy gate selects precisely the region in which the two opposing target directions have comparable magnitude, making cancellation a structural possibility for an alternating target assignment.

Cancellation under alternating assignment. To isolate the geometric effect, consider the following matched-state abstraction: all Byzantine workers are assumed to act on the same state s, with the same policy probability $\pi ,$ the same $\Delta ( s )$ , and equal steering strength. Under this abstraction, the aggregate Byzantine steering before subsequent aggregation-specific processing is

$$
\begin{array} { c } { { A ( \pi ) = \nu \left[ n _ { 0 } d ( 0 ; s ) + n _ { 1 } d ( 1 ; s ) \right] } } \\ { { = \nu \left[ n _ { 0 } - ( n _ { 0 } + n _ { 1 } ) \pi \right] \Delta ( s ) . } } \end{array}\tag{51}
$$

At the maximum-entropy center $\pi = 1 / 2$

$$
A \left( \frac { 1 } { 2 } \right) = \frac { \nu } { 2 } ( n _ { 0 } - n _ { 1 } ) \Delta ( s ) .\tag{52}
$$

For the realized assignment $n _ { 0 } = 3 , n _ { 1 } = 2 ,$ and $\nu = 2$

$$
A \left( { \frac { 1 } { 2 } } \right) = \Delta ( s ) .\tag{53}
$$

For comparison, if all $G = 5$ compromised groups were aligned to action 0, the corresponding center-point reference would be

$$
1 0 d ( 0 ; s ) = 5 \Delta ( s ) .\tag{54}
$$

Thus, the alternating assignment leaves only one fifth of the fully aligned reference magnitude at $\pi = 1 / 2$ . More generally, when the number of target-0 and target-1 groups is equal, the matchedstate aggregate vanishes exactly at the maximum-entropy point.

Equation equation 51 is a pointwise geometric characterization rather than an identity for the realized federated trajectory. In the actual training process, compromised clients encounter different states and accumulate updates over their own trajectories, so the exact matched-state assumptions need not hold globally. The equation therefore identifies a cancellation tendency induced by the binary target geometry; its quantitative effect depends on the downstream aggregation mechanism.

Relation to the empirical CartPole behavior. This geometric interpretation is consistent with the observed dependence on the aggregation surface. Under the original modulo assignment, the in-group coordinate-wise trimmed-mean and median blocks do not meet the pre-registered damage criterion on the tested CartPole configurations. By contrast, the FedPG-BR block, whose final decision mechanism operates through the separately maintained group policies and majority voting rather than a direct in-group order statistic, exhibits seed-dependent damage; on the three dynamics seeds, the corresponding values are $D = \{ 1 6 4 . 9 , 0 , 0 \}$ . This difference indicates that the same local steering geometry can produce different task-level outcomes depending on how the resulting local updates are processed downstream.

A pre-registered CartPole variant further tested the effect of removing the alternating assignment by assigning all five compromised groups to action 0. This fixed-target variant passed its pre-registered damage criterion on $2 / 3$ dynamics, yielding $D \ = \ \{ 1 3 9 . 5 , 0 , \bar { 1 } 9 2 . 4 \}$ , but it did not amplify the largest observed single-seed damage relative to the original modulo configuration, since $\mathrm { { \bar { 1 } 3 9 . 5 ~ < } }$ 164.9. Accordingly, the fixed-target result is interpreted as evidence that target alignment changes the success pattern in the tested setting, not as a uniformly stronger attack. Both target-assignment rules were fixed a priori; neither result is used to retune the attack.

Local update envelope. The local constraint has a narrower interpretation than a statistical indistinguishability or defense-acceptance guarantee. Because the malicious update is norm-matched and then coordinate-wise clamped inside the client-local empirical envelope, each coordinate remains within the prescribed empirical range formed from that client’s own update history. This construction bounds the update relative to the attacker’s local historical scale, but does not imply that the resulting update is indistinguishable from honest updates or that any particular aggregation rule must accept it.

Scope of the analysis. The conclusions above are intentionally limited to the tested setting. First, the cancellation expression in Eq. equation 51 relies on a matched-state abstraction and does not constitute an aggregation-independent impossibility result. Second, the observed effect is aggregationdependent: the analysis directly describes the binary steering geometry and its interaction with the tested in-group order-statistic aggregators, while FedPG-BR and the single-pool setting involve additional downstream mechanisms. Third, the analysis does not imply a converse rule that action spaces with $| { \mathcal { A } } | \geq 3$ must admit successful penetration. The experiments instead reveal a task- and aggregation-dependent boundary, with the empirical penetration pattern expanding from the binaryaction setting to the three- and four-action settings under the evaluated configurations.

## B BEHAVIORAL SUSCEPTIBILITY METRIC

In Section 5.2, we conduct a mechanistic audit of the WHEN selection mechanism by evaluating the behavioral susceptibility of visited trajectory states. This section formalizes the perturbation-based behavioral susceptibility metric $L ( s )$ , describes its empirical Monte-Carlo estimator, and details the policy-entropy surrogate evaluated during online execution.

Definition of Behavioral Susceptibility. Let $\pi _ { \boldsymbol { \theta } } ( \cdot \mid s )$ denote the evaluated policy at state $s \in S$ where the action space A is discrete. We measure local behavioral sensitivity by the total variation distance between the nominal action distribution and the distribution induced by isotropic state perturbations:

$$
L ( s ) \triangleq { \frac { 1 } { 2 } } \mathbb { E } _ { \varepsilon \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } I _ { d _ { s } } ) } \left[ \left| { \big | } \pi _ { \theta } ( \cdot \mid s + \varepsilon ) - \pi _ { \theta } ( \cdot \mid s ) { \big | } \right| _ { 1 } \right] \in [ 0 , 1 ] ,\tag{55}
$$

where $d _ { s }$ denotes the state dimension and σ is scaled relative to the empirical dispersion of the state variables. Larger $L ( s )$ indicates that small input perturbations induce larger shifts in the action distribution, corresponding to higher local behavioral susceptibility, whereas smaller values characterize locally stable decision regimes.

Empirical Monte-Carlo Estimator. During offline auditing, $\hat { L } ( s )$ is evaluated without model updates or backpropagation via $M = 5 0$ independent Monte-Carlo noise samples $\{ \varepsilon _ { m } \} _ { m = 1 } ^ { M } \sim$

$$
\mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d _ { s } } ) \colon
$$

$$
\hat { L } ( s ) = \frac { 1 } { 2 M } \sum _ { m = 1 } ^ { M } \big \| \pi _ { \theta } ( \cdot \mid s + \varepsilon _ { m } ) - \pi _ { \theta } ( \cdot \mid s ) \big \| _ { 1 } .\tag{56}
$$

Entropy as a Low-Overhead Computational Proxy. While the Monte-Carlo estimator $\hat { L } ( s )$ is suitable for post-hoc auditing, evaluating multiple perturbed states at every rollout step would introduce additional forward-pass latency. Therefore, V-BSA uses the normalized policy entropy

$$
h ( s ) = { \frac { H ( \pi _ { \theta } ( \cdot \mid s ) ) } { \log | { \mathcal { A } } | } }\tag{57}
$$

as a low-overhead online proxy for state selection (Eq. equation 3). Under the centered-isotropy condition of Theorem 1, policy entropy induces the same susceptibility ranking as the Fisher-trace criterion for binary action spaces. For multi-action tasks, entropy serves as a practical output-uncertainty proxy rather than an exact susceptibility measure. In the LunarLander experiments, this gate induces an average empirical intervention rate of $\rho \approx 5 . 7 \%$

## C SYSTEM DETAILS, BENCHMARK ENVIRONMENTS, AND BASELINE AGGREGATIONS

This section specifies the benchmark control environments, evaluation dynamics, server-side aggregation baselines, and implementation hyperparameters used throughout the paper.

## C.1 BENCHMARK ENVIRONMENTS AND CONTROL DYNAMICS

Table 2 summarizes the structural specifications of the benchmark environments.

<table><tr><td>Environment</td><td>State Dim. ds</td><td>Action Space A</td><td>Episode Horizon</td><td>Task Criterion</td></tr><tr><td>LunarLander-v2</td><td>8</td><td>Discrete(4)</td><td>1000</td><td>Return ≥ 200</td></tr><tr><td>CartPole-v1</td><td>4</td><td>Discrete(2)</td><td>500</td><td>Return = 500</td></tr><tr><td>Acrobot-v1</td><td>6</td><td>Discrete(3)</td><td>500</td><td>Return ≥ -100</td></tr><tr><td>LunarLanderContinuous-v2</td><td>8</td><td>Box(2)</td><td>1000</td><td>Return ≥ 200</td></tr></table>

Table 2: Environment specifications. Benchmark tasks covering binary-action, multi-action, and continuous-control settings.

CartPole-v1 $( | { \cal { A } } | = 2 )$ (Barto et al., 1983; Brockman et al., 2016). CartPole models an unactuated pole hinged to a cart moving along a frictionless horizontal track. The state is four-dimensional, $s = ( x , \dot { x } , \theta , \dot { \theta } )$ , comprising cart position, cart velocity, pole angle, and pole angular velocity. The discrete action space is $\mathcal { A } \bar { = } \{ 0 , \bar { 1 } \}$ , corresponding to applying a constant horizontal force of 10 N to the left or right, respectively. The system awards a reward of +1 per survival step. An episode terminates when $| x | > \mathsf { 2 . 4 } , | \theta | > 1 2 ^ { \circ }$ , or the maximum horizon limit $\mathbf { \bar { \mathit { H } } = 5 0 0 }$ is reached, yielding a maximum nominal return of 500.

Acrobot-v1 $( | { \mathcal { A } } | = 3 )$ (Sutton, 1995; Brockman et al., 2016). Acrobot consists of two serial links operating in a vertical plane, with torque actuation restricted strictly to the second joint. The observation space is six-dimensional, $\boldsymbol { s } = ( \cos \theta _ { 1 } , \sin \theta _ { 1 } , \cos \theta _ { 2 } , \sin \theta _ { 2 } , \dot { \theta } _ { 1 } , \dot { \theta } _ { 2 } )$ , and the action space is $\mathcal { A } = \{ 0 , 1 , 2 \}$ , providing actuated torques of $- 1 , 0 , \mathrm { o r + 1 N \cdot m }$ . The objective is to swing the lower link tip above a designated clearance line:

$$
- \cos ( \theta _ { 1 } ) - \cos ( \theta _ { 1 } + \theta _ { 2 } ) > 1 . 0 .\tag{58}
$$

Each non-terminal step incurs a penalty of −1, while achieving target clearance terminates the episode with reward 0. Episode returns are bounded within [−500, 0] under a horizon limit of $\bar { H } = 5 0 0$

LunarLander- $v { 2 } \left( \left| \mathcal { A } \right| = 4 \right)$ (Brockman et al., 2016). LunarLander simulates the trajectory guidance of a landing module descending toward a designated landing pad. The state space is eightdimensional, $s = ( x , y , \dot { x } , \dot { y } , \theta , \dot { \theta } , c _ { 1 } , c _ { 2 } )$ , where $c _ { 1 } , c _ { 2 } \in \{ 0 , 1 \}$ denote ground contact indicators for the two landing legs. The action space is discrete, $\mathcal { A } = \{ 0 , \mathrm { { 1 } } , 2 , 3 \}$ , corresponding to firing no engine, the left orientation booster, the main thruster, or the right orientation booster. The reward function balances distance-to-pad and velocity shaping with engine firing costs; successful touchdown awards +100, whereas crashing incurs $\mathbf { a } - 1 0 0$ penalty. Episodes terminate upon successful landing, module destruction, exceeding horizontal limits $( | x | \geq 1 . 0 )$ , or reaching $H = 1 0 0 0$

Continuous LunarLander (LunarLanderContinuous-v2) (Brockman et al., 2016). The exploratory continuous-control setting shares identical underlying dynamics and state variables with the discrete task, but defines an action space $\mathcal { A } = [ - 1 , 1 ] ^ { 2 }$ The coordinate $a _ { 1 } \in [ - 1 , 1 ]$ governs main thruster throttle: values in $[ - 1 , 0 ]$ disable thrust, while (0, 1] scales thrust linearly from 50% to 100%. The lateral coordinate $a _ { 2 } ~ \in ~ [ - 1 , 1 ]$ controls side thrusters: values in $\left( - 0 . 5 , 0 . 5 \right)$ turn boosters off, while $a _ { 2 } \leq - 0 . 5$ and $a _ { 2 } \geq 0 . 5$ activate the left and right thrusters, respectively. This setting is examined in Appendix E to evaluate the operational boundaries of entropy gating under continuous Gaussian policies.

Evaluation Dynamics Protocol $( d _ { 0 } , d _ { 1 } , d _ { 2 } )$ . To evaluate system resilience under environmental variation, all benchmark evaluations are conducted across three fixed dynamic realizations, denoted $d _ { 0 } , d _ { 1 } , d _ { 2 }$ . The same dynamic indices are assigned identically between nominal clean baselines and attacked runs, allowing degradation D to be reported per condition alongside summary statistics.

## C.2 SERVER-SIDE ROBUST AGGREGATION RULES

Let $\{ u _ { i } ^ { t } \} _ { i = 1 } ^ { N }$ denote the update vectors submitted by N clients at communication round t, where $u _ { i } ^ { t } \in \mathbb { R } ^ { d }$ . For standard aggregation rules, these represent local parameter updates; for FedPG-BR, they correspond to local policy-gradient estimates. An aggregation rule $\mathrm { A g g ( \cdot ) }$ produces the global update vector $u _ { \mathrm { g l o b a l } } ^ { t } .$

Federated Averaging (FedAvg) (McMahan et al., 2017). FedAvg computes the coordinate-wise arithmetic mean of all submitted vectors:

$$
u _ { \mathrm { g l o b a l } } ^ { t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } u _ { i } ^ { t } .\tag{59}
$$

No Byzantine filtering is applied.

Coordinate-Wise Trimmed Mean (Yin et al., 2018). For each coordinate $j \in \{ 1 , \ldots , d \}$ , the server sorts $\{ u _ { 1 , j } ^ { t } , \ldots , u _ { N , j } ^ { t } \}$ in non-decreasing order, discards the c lowest and c highest values, and computes the mean of the remaining $N - 2 c$ elements:

$$
[ u _ { \mathrm { g l o b a l } } ^ { t } ] _ { j } = \frac { 1 } { N - 2 c } \sum _ { r = c + 1 } ^ { N - c } u _ { ( r ) , j } ^ { t } ,\tag{60}
$$

where $u _ { ( r ) , j } ^ { t }$ denotes the r-th order statistic for coordinate j. In ensemble evaluations (6 clients per group), we set $c = 1 ;$ ; in non-ensemble evaluations $( N = 3 0 )$ , we set $c = 5$

Coordinate-Wise Median (Yin et al., 2018). The server computes the median along each parameter coordinate independently:

$$
[ u _ { \mathrm { g l o b a l } } ^ { t } ] _ { j } = \mathrm { m e d i a n } \left( \{ u _ { i , j } ^ { t } \} _ { i = 1 } ^ { N } \right) .\tag{61}
$$

For an even number of inputs, the median is taken as the mean of the two central values.

FedPG-BR (Fan et al., 2021). FedPG-BR filters submitted policy-gradient vectors using directional and magnitude alignment against a robust reference. The server determines candidate coordinate medians and selects a reference update vector. Gradients whose Euclidean distance from this reference exceeds a threshold are filtered out, and the remaining updates are averaged. In our implementation, FedPG-BR is applied to submitted policy gradients within each evaluated client pool.

Ensemble Subgroup Voting (Fang et al., 2025b). In ensemble configurations, the $N = 3 0$ clients are partitioned into $K = 5$ disjoint subgroups, $\mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { 5 }$ , with 6 clients per group (including 2 compromised clients per group). Each subgroup independently executes an aggregation rule to optimize its policy network $\pi _ { \theta _ { k } }$ . At test time, subgroup policies evaluate environment states independently, and the executed action is determined via majority voting:

$$
a ^ { * } ( s ) = \arg \operatorname* { m a x } _ { a \in \mathcal { A } } \sum _ { k = 1 } ^ { K } \mathbf { 1 } \left[ a = \arg \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } \pi _ { \theta _ { k } } ( a ^ { \prime } \mid s ) \right] .\tag{62}
$$

This architecture enforces decision-level redundancy across subgroup policies rather than parameterspace consensus across the full client population.

<table><tr><td>Component</td><td>Specification / Value</td></tr><tr><td colspan="2">Policy Architecture &amp; Optimization</td></tr><tr><td>Network Architecture Output Layer</td><td>MLP: 2 hidden layers, 64 units each, Tanh activations  $( \mathbb { R } ^ { | \mathcal { A } | } )$ </td></tr><tr><td>Optimizer</td><td>Discrete: categorical logits ; Continuous: Gaussian mean and log-variance Adam  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$  , learning rate  $\eta = 1 0 ^ { - 3 }$ </td></tr><tr><td>Discount Factor γ Rollout Budget</td><td> $0 . 9 9$  Mini-batch size  $b = 8 ;$  inner steps sampled via geometric schedule  $( B = 3 2 )$ </td></tr><tr><td>Episode Horizon</td><td>Environment-specific (Table 2)</td></tr><tr><td>Federated Consensus and Robust Aggregation Federation Scale</td><td> $M = 1 0$  Byzantine clients  $( \beta = 1 / 3 )$ </td></tr><tr><td colspan="2"> $N = 3 0$   $K = 5$ </td></tr><tr><td>Ensemble Configuration</td><td>total clients, groups (6 clients per group; 2 Byzantine clients per group)</td></tr><tr><td>FedPG-BR Parameters</td><td>Reference variance  $\sigma _ { \mathrm { r e f } } = 0 . 0 7 ,$  threshold scales  $\hat { \lambda } = 1 . 0 , \hat { \zeta } = 0 . 0 2$ </td></tr><tr><td>Coordinate-wise Median</td><td>Coordinate-wise median across client gradient vectors</td></tr><tr><td>Trimmed Mean</td><td>Coordinate-wise trimming with  $c = 1$  (ensemble) or  $c = 5$  (non-ensemble)</td></tr><tr><td>Total Training Rounds</td><td> $T = 2 6 1$  global communication rounds</td></tr><tr><td>V-BSA Attack Parameters</td><td></td></tr><tr><td colspan="2"></td></tr><tr><td>State Selection Gate</td><td>Policy entropy threshold  $\tau = 0 . 9$  (normalized by log |A|)</td></tr><tr><td>Behavioral Steering</td><td>Steering loss  $\mathcal { L } _ { \mathrm { s t e e r } }$  , unit scaling weight, target derived via group_id (mod |A|)</td></tr><tr><td>Norm Matching</td><td>l2-norm aligned to benign local gradient  $\| g _ { i } ^ { t } \| _ { 2 }$ </td></tr><tr><td>Empirical Envelope</td><td>EMA smoothing coefficient  $\alpha _ { \mathrm { { e m a } } } = 0 . 1$  , coordinate clipping scale  $\lambda _ { \mathrm { b o x } } = 0 . 8 5$ </td></tr><tr><td>Evaluation Protocol</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Terminal Return J</td><td>Evaluated over 10 independent test rollouts at round  $T = 2 6 1$ </td></tr><tr><td>Degradation Metric</td><td> $D \triangleq J _ { \mathrm { c l e a n } } - J _ { \mathrm { a t t a c k e d } }$ </td></tr></table>

Table 3: Hyperparameter configuration. Fixed parameters applied across federated reinforcement learning experiments.

## C.3 FEDERATED LEARNING AND TRAINING HYPERPARAMETERS

Policy Network Architecture and Scope. The benchmark control environments operate over lowdimensional physical state vectors $( d _ { s } \ \leq \ 8 )$ rather than spatial image grids, for which multi-layer perceptrons (MLPs) constitute the standard, task-appropriate policy representation. Following established practice in Byzantine-robust federated reinforcement learning (Fan et al., 2021; Fang et al., 2025b), we standardize the policy architecture across all discrete and continuous benchmarks to a two-layer MLP with 64 hidden units per layer and Tanh activations. Prior Byzantine FRL literature typically fixes a task-appropriate policy architecture to isolate the interaction between adversarial updates and server-side aggregation rules, avoiding network architecture as a confounding experimental factor. Formally, the operators of V-BSA interface with any parameterized differentiable policy $\pi \theta :$ the WHEN gate evaluates output-space dispersion via policy entropy, behavioral steering evaluates standard policy gradients $- \bar { \nabla _ { \theta } } \mathcal { L } _ { \mathrm { s t e e r } }$ , and the empirical envelope constrains coordinatewise deviations in parameter space. We report our empirical findings under this standard architecture as a controlled experimental baseline; evaluating the interaction between state-selective timing and alternative policy parameterizations remains an avenue for future work.

Table 3 details the complete hyperparameters utilized across nominal training, Byzantine aggregation, and the V-BSA attack.

## C.4 COMPUTE ENVIRONMENT

All experiments were implemented in Python using PyTorch and executed on Linux compute nodes equipped with NVIDIA RTX 5090 GPUs or 48GB GPU instances. Each training configuration was run as a single federated-learning process, with the 30 simulated clients executing their local rollouts and policy updates sequentially within each communication round. A complete 261-round training run typically required approximately 1.5–3.5 hours, depending on the control environment and rollout length.

## D COMPLETE TRAINING DYNAMICS ACROSS AGGREGATIONS

This section provides the complete empirical results underlying the terminal degradation comparisons reported in Table 1. We report detailed numerical breakdowns, timing schedule ablations, and full convergence trajectories across tasks and aggregations.

## D.1 DETAILED NUMERICAL BREAKDOWN AND DYNAMICS ANALYSIS

Tables 4 and 5 provide the numerical breakdown of clean terminal returns $( J _ { \mathrm { c t l } } )$ , attacked terminal returns $( J _ { \mathrm { a t k } } )$ , and resulting degradation (D) across the three evaluation dynamics for LunarLanderv2. Beyond aggregate statistics, we examine the strongest ensemble configuration—LunarLander with FedPG-BR—to illustrate how the observed degradation arises at the level of the federated sequential decision process.

Why Ensemble FedPG-BR Resists Conventional Update-Space Attacks. A representative benchmark is provided by the Byzantine-robust ensemble defense studied by Fang et al. (Fang et al., 2025b). Their Normalized attack explicitly maximizes the angular deviation between benign and poisoned updates in parameter space to disrupt policy learning. In their defense architecture, clients are partitioned into independent subgroups $\bar { ( K ) } = 5 \bar { ) }$ , each learning an isolated policy via FedPG-BR, with final decisions resolved via test-time majority voting. This design establishes a dual barrier against update-space poisoning:

(i) Intra-group filtering: Update-space attacks distribute parameter perturbations across all stateaction pairs indiscriminately, yielding gradient estimates whose global Euclidean or angular deviation is readily excised by FedPG-BR’s reference-distance filter within each subgroup;

(ii) Decision-level consensus: Even if conservative parameter shifts evade local filtering, perturbing the parameters of a minority of subgroups fails to alter the final majority vote, as uncompromised subgroup policies maintain consensus on routine states.

Penetration via Trajectory-Level Behavioral Steering. The failure of this dual defense against V-BSA stems from the sequential nature of reinforcement learning. With five groups and two Byzantine clients per group, V-BSA operates inside every group under a defense-agnostic boundary. Rather than attempting to override the server aggregation or maximize parameter divergence, V-BSA applies targeted behavioral steering strictly to a sparse subset of decision states $( \rho \approx 5 . 7 \% )$ selected via policy entropy.

Because routine states remain unperturbed, the resulting malicious update closely aligns with nominal gradient scales and satisfies the empirical coordinate envelope $B _ { i } ^ { \star } .$ , successfully passing FedPG-BR filtering alongside benign peers within each subgroup. Once accepted, the perturbation does not terminate at parameter aggregation; instead, it alters subsequent state-action visitation throughout future local rollouts. Thus, the relevant causal propagation path shifts from single-step parameter distortion to sequential behavioral coupling:

Update-space: malicious update −→ aggregated parameter,

Trajectory-level: selected states −→ behavioral steering −→ local update

$$
\longrightarrow \mathrm { s u b g r o u p p o l i c y } \longrightarrow \mathrm { t r a j e c t o r y d r i f t } .
$$

Because a perturbed action at a critical trajectory bifurcation shifts the subsequent state trajectory, the empirical impact accumulates across consecutive communication rounds rather than remaining confined to a single parameter update.

Empirical Realization and Microscopic Telemetry on LunarLander. This trajectory-level mechanism translates into severe terminal disruption under ensemble FedPG-BR. Across dynamics $d _ { 0 } , d _ { 1 } , d _ { 2 }$ , clean baseline returns of $( 2 2 8 . 6 , 2 \bar { 1 } 1 . 9 , 2 4 5 . 4 )$ degrade to $( 8 0 . 5 , 7 4 . 4 , - 1 2 5 . 2 )$ under attack $( D = \{ 1 4 8 . 1 7 , 1 3 7 . 4 8 , 3 7 0 . 5 9 \} , \bar { D } = 2 1 8 . 7 5 )$ , culminating in complete mission failure on $d _ { 2 } ( J _ { \mathrm { a t k } } = - 1 2 5 . 2 , D = 3 7 0 . 5 9 )$ . Step-level telemetry logged at round $\dot { T } = 2 6 1$ on $d _ { 2 }$ directly reveals how this degradation arises from the progressive dismantling of decision-level consensus. Whereas nominal evaluation rollouts $( \bar { J } _ { \mathrm { c t l } } = 2 4 2 . 5 0 )$ rarely deadlock—producing top-two ties in only 7.9% of decisions (333/4200 steps) with splits typically resolved by decisive majorities—V BSA drives split votes to 98.2% (1818/1851 steps) and deadlocked top-two ties to 38.0% (704/1851 steps), collapsing mean returns to −66.71 and trajectory lengths from 420.0 to 185.1 steps.

A concrete kinematic bifurcation is illustrated in Episode 2 (clean return +213.42 vs. attacked return −207.28). At step 27, the descending module enters a critical hover-adjustment state:

$$
\begin{array} { c } { { s _ { 2 7 } = [ x = - 0 . 0 2 9 , y = 1 . 2 9 5 , v _ { x } = - 0 . 1 0 9 , v _ { y } = - 0 . 5 3 8 , } } \\ { { { \theta = 0 . 0 3 3 , \dot { \theta } = 0 . 0 2 4 , c _ { 1 } = 0 , c _ { 2 } = 0 ] . } } } \end{array}
$$

At this state, Group 1’s policy entropy spikes to $H = 1 . 2 8 2 { \mathrm { . } }$ , exceeding the unnormalized selection threshold $( 0 . 9 \ln 4 ~ \approx ~ 1 . 2 4 7 7 )$ , while the remaining four groups remain far below the gate (max $H \ \leq \ 0 . 1 5 8 )$ . In test-time majority voting, the five groups split their votes as $[ 0 , 1 , 2 , \hat { 2 , } 0 ]$ producing a tie between main thruster activation (action 2) and engine idling (action 0) that resolves to idling. Failing to fire the main thruster while descending at $v _ { y } = - 0 . 5 3 8 \mathrm { m } / \mathrm { s }$ forfeits the critical deceleration window and accelerates downward descent. Subsequent attitude corrections collapse: by step 228, the lander over-rotates to $\theta = 0 . 6 3 \mathrm { r a d } ( > 3 6 ^ { \circ } )$ with a single dragged leg $( c _ { 1 } = 1 , c _ { 2 } = 0 )$ , violently crashing at the boundary $( x = - 1 . 0 0 , J _ { \mathrm { a t k } } = - 2 0 7 . 2 8 )$ . Episode 4 exhibits an identical failure mode, where an idling steering tie at step 32 $( y = 1 . 3 1 8 )$ culminates in severe attitude rollover $( \theta = 1 . 6 6 \mathrm { r a d }$ , crashing at return −163.59). These step-level dynamics confirm that V-BSA operates through a fundamentally distinct pathway from update-space attacks (Fang et al., 2025b): rather than attempting to overcome robust filtering via large parameter deviations, it selectively induces behavioral paralysis at trajectory bifurcations, allowing temporal error propagation within the MDP to trigger irreversible kinematic failure.

<table><tr><td rowspan="2">Aggregation Rule</td><td colspan="3">Dynamics  $d _ { 0 }$ </td><td colspan="3">Dynamics d1</td><td colspan="3">Dynamics  $d _ { 2 }$ </td></tr><tr><td> $J _ { \mathrm { c t l } }$ </td><td> $J _ { \mathrm { a t k } }$ </td><td>D</td><td> $J _ { \mathrm { c t l } }$ </td><td> $J _ { \mathrm { a t k } }$ </td><td>D</td><td> $J _ { \mathrm { c t l } }$ </td><td> $J _ { \mathrm { a t k } }$ </td><td>D</td></tr><tr><td>FedPG-BR</td><td>228.6</td><td>80.5</td><td>148.17</td><td>211.9</td><td>74.4</td><td>137.48</td><td>245.4</td><td>-125.2</td><td>370.59</td></tr><tr><td>Trimmed Mean</td><td>236.2</td><td>34.1</td><td>202.08</td><td>196.7</td><td>-17.0</td><td>213.62</td><td>209.7</td><td>103.5</td><td>106.18</td></tr><tr><td>Median</td><td>167.1</td><td>-17.7</td><td>184.82</td><td>233.3</td><td>103.6</td><td>129.71</td><td>216.5</td><td>3.5</td><td>212.95</td></tr><tr><td>FedAvg</td><td>243.3</td><td>167.3</td><td>76.00</td><td>224.7</td><td>112.6</td><td>112.10</td><td>194.5</td><td>142.8</td><td>51.60</td></tr></table>

Table 4: Discrete LunarLander-v2 evaluation breakdown under ensemble training $\begin{array} { l l } { ( K } & { = } \end{array}$ $5 , M = 1 0 )$ . Terminal evaluation returns and degradation $D = J _ { \mathrm { c t l } } - J _ { \mathrm { a t k } }$ across evaluation dynamics.

## D.2 ANALYSIS OF TIMING ABLATIONS

To isolate the effect of state selection, we compare entropy gating against alternative timing schedules under identical HOW construction on LunarLander with FedPG-BR:

• Entropy Gating $( \tau = 0 . 9 ) \colon$ Operates at an empirical intervention rate of $\rho \approx 5 . 7 \%$ , producing terminal degradation of $D = \{ 1 4 8 . 1 7 , 1 3 7 . 4 8 , 3 7 0 . 5 9 \}$ across dynamics (mean $\bar { D } = 2 1 8 . 7 5$ median 148.17).

• Dense Poisoning $\begin{array} { r l r } { ( \rho } & { { } = } & { 1 0 0 \% ) \colon } \end{array}$ Intervenes at every rollout step, yielding $\begin{array} { r l } { D } & { { } = } \end{array}$ {−4.28, 36.65, 49.78} (mean $\bar { D } = 2 7 . 3 8$ , median 36.65).

<table><tr><td rowspan="2">Aggregation Rule</td><td colspan="3">Dynamics do</td><td colspan="3">Dynamics d1</td><td colspan="3">Dynamics d2</td></tr><tr><td> $J _ { \mathrm { c t l } }$ </td><td> $J _ { \mathrm { a t k } }$ </td><td>D</td><td> $J _ { \mathrm { c t l } }$ </td><td> $J _ { \mathrm { a t k } }$ </td><td>D</td><td> $J _ { \mathrm { c t l } }$ </td><td> $J _ { \mathrm { a t k } }$ </td><td>D</td></tr><tr><td>FedPG-BR</td><td>162.7</td><td>-59.2</td><td>221.91</td><td>240.4</td><td>-115.1</td><td>355.48</td><td>225.1</td><td>-61.5</td><td>286.64</td></tr><tr><td>Trimmed Mean</td><td>258.4</td><td>-12.3</td><td>270.70</td><td>249.1</td><td>-15.5</td><td>264.60</td><td>233.9</td><td>189.1</td><td>44.80</td></tr><tr><td>Median</td><td>234.4</td><td>-287.2</td><td>521.60</td><td>210.3</td><td>-168.1</td><td>378.40</td><td>225.8</td><td>-177.2</td><td>403.00</td></tr><tr><td>FedAvg</td><td>216.3</td><td>255.0</td><td>-38.70</td><td>198.2</td><td>269.1</td><td>-70.90</td><td>201.5</td><td>263.4</td><td>-61.90</td></tr></table>

Table 5: Discrete LunarLander-v2 evaluation breakdown under non-ensemble training $( N =$ $3 0 , M = 1 0 )$ ). Terminal returns and degradation D across evaluation dynamics.

• Uniform Random Selection $( \rho = 1 5 \% ) \colon$ : Selects states uniformly with probability $p = 0 . 1 5 ,$ yielding $D = \{ - 1 5 . 8 9 , 5 0 . 0 1 , 2 7 3 . 7 9 \}$ (mean $\bar { D } = 1 0 2 . 6 4$ , median 50.01).

• Budget-Matched Bernoulli Selection (ρ ≈ 5.7%): Matches the empirical intervention rate of entropy gating via uniform random selection on dynamics $d _ { 0 }$ , yielding $D = 9 1 . 9 9 $ (compared to 148.17 for entropy gating).

These comparisons show that, with the update construction mechanism held fixed, varying the trajectory-level intervention schedule materially alters downstream policy degradation.

## D.3 COMPLETE CONVERGENCE TRAJECTORIES

Figures 8–12 provide the complete rolling evaluation trajectories over all communication rounds and evaluated task–aggregation configurations. In each plot, solid curves denote attacked training under V-BSA, while dashed curves denote the corresponding clean baseline. Beyond the terminal degradation reported in the main results, these trajectories expose how performance divergence develops over training, specifically distinguishing progressive, sustained performance degradation from transient endpoint fluctuations.

Across configurations, the trajectories are consistent with the task- and aggregation-dependent boundaries reported in Section 5.2. Settings with large terminal degradation generally exhibit an early, sustained separation between attacked and clean trajectories, whereas configurations with small or negative D display tightly tracking or non-persistent gaps. Across all panels, curves display the 10-round rolling mean of evaluation returns, with the terminal points at round $T = 2 6 1$ matching the evaluation returns used to compute the degradation statistics in Table 1 (with complete per-dynamics breakdowns provided in Tables 4 and 5).

![](images/67e0c33b0735356c8a35017f0edf20269211c33646cd51856d5aec0da724c61d.jpg)  
(a) FedPG-BR

![](images/a1a625feeeb1e9185d7a25751ed9374c8280d957c42235f38f7cb7926aef1f8a.jpg)  
(b) Trimmed Mean

![](images/47343e8b02cf923a0650c57e541c0be6f137abaa9f5e4914c57b9c61b177c6df.jpg)  
(c) Median

![](images/0070aeeff9665fd5f4f4a0b595efd5cfb8b5b7d2849fa5967971df7729c573d9.jpg)  
(d) FedAvg  
Figure 8: Ensemble paradigm on CartPole $( K = 5 , M = 1 0 )$ . Evaluation trajectories across aggregation rules. Solid curves: V-BSA attack; dashed curves: clean control.

![](images/8d4cffdc5ad76c31fb0ecc3386807b0f94ea6e607f2f5114f0925bb34398b562.jpg)  
(a) FedPG-BR

![](images/71db55446b70badeb42c78134211d67e19d038e64476eea17d3b21ba10d41445.jpg)  
(b) Trimmed Mean

![](images/16bd7ee0979a2e2d39d97883ee6a7957dfac021d5a94405737a4ec92f57daa9f.jpg)  
(c) Median

![](images/62635b59dc8fea22776cc836ffab2b8fb3dde15a8c07ffea5e846b7b56c824f7.jpg)  
(d) FedAvg  
Figure 9: Ensemble paradigm on Acrobot $( K = 5 , M = 1 0 )$ . Evaluation trajectories across aggregation rules. Solid curves: V-BSA attack; dashed curves: clean control.

![](images/9cdd399538e703f42332205a2012e354a016496f60cc034a03811b3db97d7f66.jpg)  
(a) FedAvg

![](images/e4bd1b7d78ee06192a571cb60ff03230883874f78d145a17394dac27442c5dd9.jpg)  
(b) Median

![](images/87b3ec8987136bd02524787d20361d2315b68a968426a5aebf98979e3d3a97b7.jpg)  
(c) Trimmed Mean

![](images/e8d3629802b18e6bf3c00af4678bb255c247f3c08a305cbb5d83c2e75f3394cd.jpg)  
(d) FedPG-BR  
Figure 10: Non-ensemble paradigm on CartPole $( N = 3 0 , M = 1 0 )$ . Evaluation trajectories across aggregation rules. Solid curves: V-BSA attack; dashed curves: clean control.

![](images/c2f4702423ef44902dc33d1d72dc48473f2f09a3a8fdd978dd1248eacc91a83c.jpg)  
(a) FedAvg

![](images/b6fd52bb72f8c7c71d6c58d6ba492154fad936d7fd80ea166b59a1c0684eefe6.jpg)  
(b) Median

![](images/4540e8d8caca6254f7dce4af7c1bb4db8ecb2085e67decc2a3b76ccd739e9023.jpg)  
(c) Trimmed Mean

![](images/5dddd9d4351e3a9de5b78833e9c527cc523952c1d90377c72f3d2483a3633b68.jpg)  
(d) FedPG-BR  
Figure 11: Non-ensemble paradigm on Acrobot $( N = 3 0 , M = 1 0 )$ . Evaluation trajectories across aggregation rules. Solid curves: V-BSA attack; dashed curves: clean control.

![](images/e51b5fbff4e8c491b8931eb76e18eb7105418eca432b7b8b0f212c861c343729.jpg)  
(a) FedAvg

![](images/b0726d8295159ccd279761408120c0a12daadc407bc8c2cc3751a80b4f901f41.jpg)  
(b) Median

![](images/5320c68a87413ded158f88e7d6b0a9c364f80828157bafb841b5bef853665e0e.jpg)  
(c) Trimmed Mean

![](images/0db367f2fc75a6df27d206aeb908a96d59b8d2cd929b66f8b29b530240baa70c.jpg)  
(d) FedPG-BR  
Figure 12: Non-ensemble paradigm on discrete LunarLander-v2 $( N = 3 0 , M = 1 0 )$ . Evaluation trajectories across aggregation rules. Solid curves: V-BSA attack; dashed curves: clean control.

## E EXPLORATORY STUDY ON CONTINUOUS ACTION SPACES

To examine whether the state-selective intervention framework extends to continuous control, we conduct an exploratory study on LunarLanderContinuous-v2, which shares the underlying physics and reward specification of LunarLander-v2 but operates over continuous action bound $\overset { \cdot } { \mathcal { A } } = \operatorname { B o x } ( 2 )$ . Policies parameterize a Gaussian distribution $\dot { \pi } _ { \boldsymbol { \theta } } ( a \mid s ) = \mathcal { N } ( \mu _ { \boldsymbol { \theta } } ( s ) , \Sigma )$

## E.1 LIMITATIONS OF CONTINUOUS UNCERTAINTY GATING

In discrete action spaces, categorical policy entropy naturally quantifies dispersion across competing discrete choices. Extending this principle to continuous Gaussian policies encounters structural limitations:

• Spatial Invariance under State-Independent Variance: When policy variance $\Sigma = \operatorname { D i a g } ( \sigma ^ { 2 } )$ is parameterized independently of the input state, the differential entropy $H ( \pi _ { \theta } ( \cdot \bar { s } ) ) \stackrel { . } { = }$ $\textstyle { \frac { 1 } { 2 } } \sum _ { i = 1 } ^ { d _ { a } } \log ( 2 \pi e \sigma _ { i } ^ { 2 } )$ is constant across all states, providing no discriminatory signal for trajectory-level state selection.

• Gate Saturation with State-Dependent Heads: When employing a state-conditioned variance head $\sigma _ { \theta } ( s )$ , exploratory variance remains elevated throughout the state space during training. Thresholding on local variance triggers on nearly all visited trajectory steps $( \rho \to 1 . 0 )$ , collapsing sparse state gating back into dense poisoning.

## E.2 CONTINUOUS BEHAVIORAL STEERING

In continuous action spaces, behavioral steering cannot evaluate categorical cross-entropy. Instead, we define continuous steering by perturbing predicted action means toward saturated control corners:

$$
\mathcal { L } _ { \mathrm { c o n t \mathrm { . s t e e r } } } = \frac { 1 } { | \mathcal { D } _ { \mathrm { w h e n } } | } \sum _ { s \in \mathcal { D } _ { \mathrm { w h e n } } } \big \| \mu _ { \theta } ( s ) - a _ { \mathrm { c o r n e r } } ^ { ( i ) } \big \| _ { 2 } ^ { 2 } ,\tag{63}
$$

where $a _ { \mathrm { c o r n e r } } ^ { ( i ) } \in \{ - 0 . 8 , + 0 . 8 \} ^ { d _ { a } }$ assigns fixed control quadrants to compromised groups.

<table><tr><td>Perturbation Schedule</td><td>Dynamics  $d _ { 0 }$ </td><td>Dynamics  $d _ { 1 }$ </td><td>Dynamics  $d _ { 2 }$ </td><td>Median  $D$ </td></tr><tr><td>Fixed Perturbation Scale  $( \lambda = 0 . 0 3 )$ </td><td>-23.12</td><td>-169.51</td><td>+29.35</td><td>-23.12</td></tr><tr><td>Adaptive Scale Servo  $( \lambda \in [ 0 . 0 3 , 0 . 1 2 ] )$ </td><td>+82.05</td><td>+82.91</td><td>-33.72</td><td>+82.05</td></tr></table>

Table 6: Continuous control evaluations on LunarLanderContinuous-v2 under FedPG-BR. Terminal degradation $D$ across evaluation dynamics under fixed and adaptive perturbation scaling.

Table 6 reports terminal degradation under fixed and adaptive scaling:

• Fixed Scaling: A conservative perturbation scale $( \lambda = 0 . 0 3 )$ remains within local update bounds but does not produce consistent policy degradation, resulting in a negative median degradation $( D = - 2 3 . 1 2 )$

• Adaptive Scaling: An adaptive servo adjusting perturbation scale based on local gradient acceptance yields positive degradation on dynamics d $( D = 8 2 . 0 5 )$ and $d _ { 1 } ( D = 8 2 . 9 1 )$ . However, dynamics $d _ { 2 }$ exhibits negative degradation $( D = - 3 3 . 7 2 )$ , demonstrating substantial sensitivity to underlying transition dynamics.

Discussion. These exploratory results indicate that while continuous behavioral steering is techni cally feasible, directly applying output-space uncertainty proxies does not reliably yield sparse state selection. Developing principled susceptibility proxies for continuous control—such as measures based on representation sensitivity or local value gradients—remains an open direction.