# WHERE DOES RANDOMNESS MATTER IN NEURAL CELLULAR AUTOMATA?

Fei Zuo Fudan University

Jiaqi Shi Independent Researcher

Yujing Liu Shanghai Pupuda Culture Communication Co., Ltd.

## ABSTRACT

Stochastic cell updates are often used throughout the life of a neural cellular automaton (NCA), from backpropagation through time to final rollout. This leaves two questions entangled: does update randomness help learn a useful rule, and must that randomness remain at execution? We separate training and evaluation update modes in controlled Growing NCA experiments, then vary the states shown during training. Under the standard constant-rate persist recipe, asynchronous training passes the short-horizon quality test in 10/10 runs, compared with 3/10 synchronous runs. All ten asynchronous models also retain the target for 4,096 steps under deterministic evaluation. For a scalar translation-invariant lattice, we derive an exact mean-square criterion: random masking can damp mean modes, but it also injects variance, and a mean-only test misclassifies four non-marginal settings. Finally, among 30 models that all pass the same reconstruction test, eight of ten grow-trained models become off-target at 4,096 steps, while all persist and regenerate models retain the target; damage recovery separates persist from regenerate. The results distinguish optimization reliability, execution mode, and task-specific behavior instead of treating them as one stability property.

## 1 INTRODUCTION

A neural cellular automaton (NCA) learns one local update rule and applies it repeatedly across a grid. The rule is optimized through finite rollouts, but after training the same parameters may be applied for thousands of steps. Random cell-update masks are often used in both stages. It is therefore easy to conflate two different claims: randomness may make optimization easier, and randomness may be required for the learned rule to work when it runs.

Prior NCA work gives reasons to use asynchronous updates, including local operation and robustness to changes in update timing (Niklasson et al., 2021). It also shows that the training recipe changes what a learned rule can do: seed-only training can produce patterns that decay or grow, pool training encourages persistence, and damage exposure strengthens repair (Mordvintsev et al., 2020). Synchronous training has also been reported as brittle in a multi-target self-replication setting, with successful synchronous runs producing outcomes similar to asynchronous ones (Sinapayen, 2023). These findings establish that update schedules and training states matter. They do not, by themselves, separate the update schedule used to learn a rule from the schedule used to execute it, or compare later-use behaviors after a common reconstruction test.

We ask two linked questions. Where does stochastic updating matter: while learning the rule, while executing it, or both? And once a rule can reconstruct its target, which behaviors follow from the states it saw during training? The first question requires crossing training and evaluation update modes while holding the recipe and optimizer fixed. The second requires comparing models that meet the same short-horizon quality threshold, then testing the distinct tasks of retention and repair.

Our update-mode comparison separates these stages. At the tested constant learning rate, asynchronous training produces ten quality-passing models, compared with three under synchronous training. Yet every asynchronous model retains the target through 4,096 deterministic steps without retraining (Fig. 1). Four exploratory lower-rate synchronous retrainings also pass, so the observed training difference is specific to the tested optimizer setting. Earlier work already reports synchronoustraining brittleness in another NCA task; our result is a controlled train/evaluation crossover on the canonical persistent Growing NCA, not a claim that the failure itself is new.

![](images/a5e42558b219431af85ff4a7417775a2aa1dde0c65f669474030cf40a89ebb06.jpg)

![](images/e5b74778cecbd9b7cb58639b91e280fcea36427e60f0e8090845e1b4a2596de1.jpg)  
Figure 1: Random updates help obtain a usable rule under one training setting, but are not required by successful deterministic rollouts. The shared local rule separates the random update mask M from the living masks. (a) With the persist recipe, learning rate, and budget held fixed, 3/10 synchronous models and 10/10 asynchronous models pass the quality test at $T = 9 6$ in their training mode. (b) With weights frozen, all ten asynchronous models retain the target through $T = 4 { , } 0 9 \bar { 6 }$ deterministic steps from the seed. Lattices schematically depict update masks, not measured states.

The update mask also has a two-sided effect in a tractable linear system. We derive an exact secondmoment recursion for a scalar translation-invariant lattice with independent Bernoulli cell masks. It shows why averaging the mask can give the wrong answer: random updating changes the mean dynamics and injects state variance. The theorem is a boundary on what can be inferred from a fixed linear rule; it does not explain the nonlinear training failures.

Finally, we compare three established training recipes after all models pass the same short-horizon reconstruction test. Grow, persist, and regenerate differ in which states the rule encounters during optimization. Their long-horizon retention and damage recovery differ accordingly. These measurements make a practical distinction: update randomness affects whether a rule is learned under the tested optimizer, while state exposure affects which later-use tasks that rule supports.

## This study contributes:

1. A controlled separation of training-time and evaluation-time update schedules. Asynchronous training improves final quality-pass rate under one fixed-rate persist recipe, while successful asynchronous models can be evaluated deterministically.

2. An exact mean-square decay criterion for a scalar linear lattice with independent cell masks, showing when mean contraction is insufficient.

3. A matched-quality comparison of retention and repair across established training-state recipes, with long rollouts and damage tests used as task outcomes rather than as one aggregate stability score.

## 2 MODEL AND EXPERIMENTAL DESIGN

The update rule. We use the canonical Growing NCA (Mordvintsev et al., 2020). Each cell stores four RGBA channels and twelve hidden channels. Fixed identity and Sobel filters form a 48-channel local perception $K * x$ . A shared two-layer $1 \times 1$ network $f _ { \theta } ,$ , with 128 hidden units, ReLU activation, and a zero-initialized output layer, predicts an additive state change. A step is

$$
\begin{array} { r l r } & { } & { y _ { t } = x _ { t } + M _ { t } \odot f _ { \theta } ( K * x _ { t } ) , } \\ & { } & { x _ { t + 1 } = g ( x _ { t } ) \odot g ( y _ { t } ) \odot y _ { t } , \qquad ( M _ { t } ) _ { u } \sim \mathrm { B e r n } ( \alpha ) . } \end{array}\tag{1}
$$

The Bernoulli update masks are independent across cells and steps and shared across the channels within a cell. The living mask $g$ is one when the maximum alpha value in a $3 \times 3$ neighborhood exceeds 0.1; it is applied both before and after the candidate update. An entirely dead state is absorbing. We use update mask for $M _ { t }$ and living mask for $g$ because they play different roles.

Two training comparisons. The targets are flower and heart images on $7 2 \times 7 2$ grids, with five seeds per configuration. Base runs use Adam, constant learning rate ${ \bar { 2 } } \cdot 1 0 ^ { - 3 }$ , batch size eight, 8,000 optimization steps, rollout lengths sampled uniformly from 64 to 96, and the same overflow penalty. The update-mode comparison contains twenty persist models: ten synchronous $( \alpha _ { \mathrm { t r a i n } } = 1 )$ and ten asynchronous $( \alpha _ { \mathrm { t r a i n } } = 0 . 5 )$ . Each model is evaluated in both update modes. The recipe comparison contains thirty asynchronous models, ten each trained with grow, persist, or regenerate. The ten asynchronous persist models are shared between comparisons, giving forty distinct base models. Four lower-rate synchronous retrainings are exploratory.

Grow starts every rollout from the seed. Persist samples from a pool of 1,024 generated states and replaces the worst sample with the seed. Regenerate adds circular damage to half of the sampled batch. These recipes were introduced in the original Growing NCA work; our comparison controls the architecture, objective, optimizer, training budget, target set, and update probability while changing the training-state distribution.

Three behavioral requirements. A model passes the short-horizon quality test if its RGBA MSE is below 0.02 at $T = 9 6$ in its training update mode. For long rollouts, an empty state is labeled dead; a non-finite state or one with maximum absolute value above $1 0 ^ { 3 }$ is labeled divergent; and a surviving finite state with final MSE above 0.15 is off-target. The remaining category is called retained. The long-horizon threshold is deliberately distinct from the stricter reconstruction threshold.

$\begin{array} { r } { \mathbf { A } \mathbf { t } T = 9 6 . } \end{array}$ , we also measure finite-time perturbation growth by adding $\delta _ { 0 } = 1 0 ^ { - 3 } \mathcal { N } ( 0 , I )$ on living cells and evolving perturbed and clean states for 64 steps with the same update masks:

$$
\lambda _ { \mathrm { p e r t } } = \frac { 1 } { 6 4 } \log \frac { \| \delta _ { 6 4 } \| _ { 2 } } { \| \delta _ { 0 } \| _ { 2 } } .\tag{2}
$$

This is a finite-amplitude trajectory measurement, not a maximum Lyapunov exponent. Damage recovery is evaluated separately over a fixed 80-step window. The full protocols, statistical units, and outcome thresholds are in App. C.

## 3 RANDOM UPDATES CHANGE LEARNING SUCCESS MORE THAN EXECUTION MODE

To isolate the update schedule, we vary $\alpha _ { \mathrm { t r a i n } }$ while fixing the persist recipe and optimizer. We then freeze each model’s parameters and evaluate it using both synchronous and asynchronous updates. This design separates the probability of obtaining a usable final rule from a requirement on how that rule must run.

Final training success differs under the tested optimizer. All ten asynchronous models pass the $T = 9 6$ quality test, compared with 3 of ten synchronous models (Table 1); the two-sided Fisher exact test gives $p = 0 . 0 0 3$ . This is a controlled comparison of two update schedules under one learning rate and one training recipe, not a general statement that synchronous NCA training cannot succeed.

The available synchronous training traces add context but do not replace final evaluation. All 9 recorded traces cross a minibatch loss of $1 0 ^ { - 3 }$ , while seven of ten final synchronous checkpoints fail the quality test. Six failures are empty states and one is alive but deformed; five available traces end near the empty-canvas loss. Because the traces are incomplete and the loss is measured on a training minibatch, they show that low observed loss can occur during optimization, not that every failed run previously held a robust rule. An exploratory single-run checkpoint replay is documented separately in App. G; it illustrates one trajectory and does not estimate how often intermediate rules remain usable.

Successful asynchronous models do not need stochastic evaluation. All ten asynchronous models retain the target for 4,096 deterministic steps from the seed. Among quality-passing models, switching the evaluation update mode changes median absolute MSE by $1 . { \overset { - } { 1 } } { \overset { - } { \times } } 1 0 ^ { - 3 }$ for asynchronous-trained models and $2 . 3 \times 1 0 ^ { - 3 }$ for synchronous-trained models at $\dot { T } = 1 0 2 4$ . The preregistered asymmetry comparison did not establish a directional swap effect. The useful conclusion is narrower: in this cohort, every asynchronous-trained model that passes short-horizon quality can be run deterministically and remains in the retained class.

Four exploratory retrainings of two failed flower configurations at learning rates $1 0 ^ { - 3 }$ and $5 \cdot 1 0 ^ { - 4 }$ all pass. This control limits the interpretation of the main comparison: asynchronous updating improves the pass rate at the tested constant learning rate; it is not necessary for synchronous training to succeed.

## 4 A RANDOM MASK DAMPS SOME MEAN MODES AND INJECTS VARIANCE

The NCA comparison shows when the update mask affects learning, but it does not identify a mechanism for the training failures. We therefore analyze a fixed linear residual rule, where the effect of independently selecting cells can be computed exactly. Consider

$$
x _ { t + 1 } = x _ { t } + M _ { t } A x _ { t } ,
$$

where A is translation invariant on a periodic scalar lattice and each diagonal entry of $M _ { t }$ is an independent Bernoulli variable with mean α.

Proposition 1 (Mean dynamics). For $0 < \alpha \leq 1 , \mathbb { E } [ x _ { t + 1 } ] = ( I + \alpha A ) \mathbb { E } [ x _ { t } ] .$ . If µ is an eigenvalue of the synchronous map $I + A ,$ then the corresponding mean multiplier lies in

$$
D _ { \alpha } = \{ \mu \in \mathbb { C } : | \mu - ( 1 - 1 / \alpha ) | < 1 / \alpha \} .
$$

The disk $D _ { \alpha }$ passes through +1; at $\alpha = 1$ it is the unit disk, while as α decreases its left edge moves outward. In particular, for a unit-modulus mode $\mu = e ^ { i \theta }$ ,

$$
| ( 1 - \alpha ) + \alpha e ^ { i \theta } | ^ { 2 } = 1 - 2 \alpha ( 1 - \alpha ) ( 1 - \cos \theta ) .
$$

Partial updates damp non-neutral oscillatory modes in the mean. This statement concerns the expected state, not the energy of a randomly updated trajectory.

Theorem 1 (Exact second-moment dynamics). Let $x _ { t }$ be a scalar field on the $N ^ { d }$ periodic lattice, and let A have Fourier symbol $\hat { a } ( \omega )$ . With masks independent across cells and time, the expected modal power $P _ { t } ( \omega ) = \mathbb { E } | \hat { x } _ { t } ( \omega ) | ^ { 2 }$ satisfies

$$
P _ { t + 1 } ( \omega ) = d _ { \alpha } ( \omega ) P _ { t } ( \omega ) + \frac { \alpha ( 1 - \alpha ) } { N ^ { d } } \sum _ { \omega ^ { \prime } } \lvert \hat { a } ( \omega ^ { \prime } ) \rvert ^ { 2 } P _ { t } ( \omega ^ { \prime } ) ,\tag{3}
$$

where $d _ { \alpha } ( \omega ) = | 1 + \alpha \hat { a } ( \omega ) | ^ { 2 }$ . All non-neutral modes decay in mean squarefor everyfinite-secondmoment initial state ifand only $i f d _ { \alpha } ( \omega ) < 1$ on those modes and

$$
S = \sum _ { \omega : \hat { a } ( \omega ) \neq 0 } \frac { \alpha ( 1 - \alpha ) | \hat { a } ( \omega ) | ^ { 2 } } { N ^ { d } [ 1 - d _ { \alpha } ( \omega ) ] } < 1 .\tag{4}
$$

Neutral modes are excluded from this decay criterion.

The recursion is a diagonal modal update plus a rank-one power injection. The first term propagates each mode under the averaged rule; the second is the variance introduced by the random mask and couples the modal powers. Thus, replacing the mask by its mean can overstate stability. The criterion makes that discrepancy explicit through S. Proofs, including the neutral-mode case, are in App. B.

Table 1: Training and evaluation update modes. Quality is tested at $T = 9 6$ in the training mode. Retention is the bounded, on-target class after 4,096 deterministic steps. Swap cost is the median absolute MSE change at $T = 1 0 \bar { 2 } 4$ among quality-passing models. Each target has five models per training mode.
<table><tr><td>Training</td><td>Target</td><td>Quality pass</td><td>Retained (det.)</td><td>Swap difference |∆MSE|</td></tr><tr><td rowspan="2">Synchronous  $( \alpha = 1 )$ </td><td>Flower</td><td>1/5</td><td>1/5</td><td> $2 . 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Heart</td><td>2/5</td><td>3/5</td><td> $2 . 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="2">Asynchronous  $( \alpha = 0 . 5 )$ </td><td>Flower</td><td>5/5</td><td>5/5</td><td> $2 . 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Heart</td><td>5/5</td><td>5/5</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td></tr></table>

We test the criterion on a $3 2 \times 3 2$ diffusion lattice with update $x  x + M ( c \mathrm { L a p } x )$ . The secondmoment test matches all 25 non-marginal simulation labels and resolves 4 settings that the mean-only test calls stable. The complete grid contains 27 configurations; the two exactly marginal cases are reported separately because finite-time labels there need not match asymptotic decay (Fig. 2, App. E). This fixed-rule result limits the interpretation of update randomness: it can suppress selected mean modes while still increasing trajectory variance. It is not a derivation of the nonlinear training outcome in Sec. 3.

## 5 TRAINING-STATE EXPOSURE DETERMINES RETENTION AND REPAIR

Update timing is only one training choice. The distribution of states used by the loss determines which parts of the learned trajectory receive direct pressure. To compare these effects, we evaluate grow, persist, and regenerate models that all pass the same $T = 9 6$ quality threshold, then test retention through 4,096 steps and recovery after damage.

Equal reconstruction does not imply equal later behavior. All thirty recipe models pass the short-horizon quality test, with group median errors from 35 to 1,078 times below the threshold. By $T = 4 0 9 6$ , eight of ten grow models are off-target, whereas all ten persist and all ten regenerate models remain in the retained class (Table 2). These outcomes are measured after training and do not follow from the pass threshold alone.

The trajectory perturbation measurement provides a complementary, but not interchangeable, description. Persist models have negative mean $\lambda _ { \mathrm { p e r t } }$ on both targets, while grow models have positive mean values. The grow-minus-persist bootstrap intervals exclude zero on both targets. Regenerate models span positive and negative responses; their group intervals include zero on both targets. Positive perturbation growth is not itself a failure label: five models with positive $\lambda _ { \mathrm { p e r t } }$ still retain the pattern. We therefore report the direct long-horizon task outcome alongside the trajectory measurement (Fig. 3).

Damage exposure adds a distinct behavior. Under the same damage protocol, regenerate models reduce damage-induced error by a mean of +85% on flower and +99% on heart. Persist changes error little on average (−7% and +4%), while grow worsens it $( - 4 6 \%$ on each target). The growminus-regenerate interval on heart includes zero, so the target-specific contrast is not equally strong everywhere. These results are consistent with the training inputs: pool states expose models to formed patterns, while circular damage directly trains recovery. The recipes themselves are established; our contribution is the controlled comparison of their task outcomes under one architecture and protocol.

A frozen local spectrum does not substitute for these tests. All thirty recipe models have a largest returned frozen-state Jacobian eigenvalue above one, yet persist and regenerate models retain the target while most grow models become off-target. This measurement does not separate the observed fates. Removing the living masks leaves all evaluated rollouts numerically bounded, while two regenerate models change target class. The full eigenvalue counts and mask-removal controls are reported in App. H and App. G.4. These controls reinforce the need to specify which property is being tested: local expansion, numerical boundedness, target retention, and damage recovery are not interchangeable.

Mean-only boundary

(a) Mean-stability region  
![](images/f36f5c609613857a56cf0881661292164d9dd4579fb6d111222994da46b5debd.jpg)

(b) Second moments matter  
![](images/a197e530aaf0d88deb9c46cb251063b11f822b26c3428b9b8811aa00f525eb6e.jpg)  
Second-moment boundary

Figure 2: Mean contraction and mean-square decay are different tests. (a) Mean-stability disks for different update rates. (b) Simulated outcomes on the diffusion lattice. The exact secondmoment criterion separates the non-marginal stable and unstable cases; rings mark the mean-only disagreements and diamonds mark the two marginal configurations.  
Table 2: Models with similar reconstruction quality learn different later-use behavior. MSE columns are medians over five models per target. Perturbation growth is the model mean, with bootstrap 95% intervals below. Recovery is the mean percentage under the frozen damage protocol. Retention uses the long-horizon threshold, not the stricter $T = 9 6$ quality threshold.
<table><tr><td>Recipe</td><td>Target</td><td>MSE  $T = 9 6$ </td><td> $\lambda _ { \mathrm { { p e r t } } }$  95% CI</td><td>MSE  $T = 4 0 9 6$ </td><td>Recovery (%)</td><td>Retained</td></tr><tr><td>Grow</td><td>Flower</td><td> $6 . 6 \times 1 0 ^ { - 5 }$ </td><td>+0.017 [+0.011,+0.021] +0.023</td><td>0.26</td><td>-46</td><td>1/5</td></tr><tr><td></td><td>Heart</td><td> $1 . 9 \times 1 0 ^ { - 5 }$ </td><td>[+0.017, +0.028] -0.024</td><td>0.34</td><td>-46</td><td>1/5</td></tr><tr><td>Persist</td><td>Flower</td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td><td>[-0.031, -0.015] -0.033</td><td> $6 . 5 \times 1 0 ^ { - 4 }$ </td><td>-7</td><td>5/5</td></tr><tr><td></td><td>Heart</td><td> $6 . 1 \times 1 0 ^ { - 5 }$ </td><td>[-0.040, -0.027]</td><td> $8 . 5 \times 1 0 ^ { - 5 }$ </td><td>+4</td><td>5/5</td></tr><tr><td>Regenerate</td><td>Flower</td><td> $4 . 3 \times 1 0 ^ { - 4 }$ </td><td>-0.008 [-0.017, +0.006]</td><td> $7 . 0 \times 1 0 ^ { - 4 }$ </td><td>+85</td><td>5/5</td></tr><tr><td></td><td>Heart</td><td> $5 . 7 \times 1 0 ^ { - 4 }$ </td><td>+0.002  $[ - 0 . 0 1 5 , + 0 . 0 2 5 ]$ </td><td> $1 . 3 \times 1 0 ^ { - 3 }$ </td><td>+99</td><td>5/5</td></tr></table>

## 6 DISCUSSION

The results support a simple design sequence for a learned local rule. First, choose an update schedule that gives reliable optimization under the available budget. In the tested fixed-rate persist setup, asynchronous updates improve the final pass rate; the lower-rate synchronous probes show that this is an optimizer-dependent advantage. Second, test whether the learned model requires stochastic updates at execution. In our cohort, every passing asynchronous model remains on-target under deterministic evaluation. Third, train on the states that the deployed rule must handle: generated states support retention, and explicit damage exposure supports repair. Reconstruction loss, local spectrum, and finite-time perturbation growth each answer only part of this sequence.

The analytical result clarifies one limit of this advice. Independent cell masks change expected linear dynamics and add a variance term, so random updating is not a universal stability switch. The exact criterion applies only to a fixed scalar translation-invariant linear operator. It does not predict the absorbing-state failures of the nonlinear NCA or replace direct evaluation of the trained rule.

![](images/f0c97bc582f57503d3536aab22157cea74477383fe421fad054602ab071049b1.jpg)

![](images/b590ea31f55d73b8d018443254c71c0a44fe76f6ff4a649e873ef4379c725d5c.jpg)  
Figure 3: State exposure changes trajectory response and long-horizon error. (a) Individual finitetime perturbation growth and group intervals. (b) Flower median MSE as rollout length increases beyond the maximum training horizon; heart outcomes are summarized in Table 2.

The scope is intentionally narrow: two synthetic targets, one Growing NCA family, five seeds per target and recipe, one primary learning rate, and a finite rollout horizon. Synchronous-training failures have prior precedent, and our lower-rate probes are exploratory. The results establish a controlled distinction for this model and protocol; broader claims require other targets, architectures, and optimizer schedules.

## 7 RELATED WORK

NCA training and applications. Growing NCA establishes seed growth, pool-based persistence, and damage-based regeneration (Mordvintsev et al., 2020). Subsequent work explores asynchronous update timing, identity mappings, and applications across dynamic textures, classification, graph and generative modeling, and 3D morphogenesis (Niklasson et al., 2021; Stovold, 2025; Pajouheshgar et al., 2023; Randazzo et al., 2020; Grattarola et al., 2021; Palm et al., 2022; Tesfaldet et al., 2022; Sudhakaran et al., 2021; Najarro et al., 2022; Mordvintsev et al., 2022). Recent reviews synthesize NCA architectures and reference implementations (Spitznagel & Keuper, 2026). Synchronous training has also been reported as brittle in multi-target self-replication (Sinapayen, 2023).

Attractor geometry and dynamical analysis. Recent work examines deterministic NCA attractors, perturbation responses, and transient dynamics (Kvalsund & Stovold, 2026; Masumori et al., 2026; Sato et al., 2026). We complement this perspective through controlled train/evaluation crossovers and common-quality recipe comparisons under the canonical Growing NCA setting. Broader dynamical foundations span classic cellular automata, criticality, morphogenetic pattern formation, and continuous artificial life (von Neumann, 1966; Gardner, 1970; Wolfram, 1984; Langton, 1990; Bak et al., 1987; Turing, 1952; Chan, 2019), as well as reservoir computing with evolved critical NCA (Pontes-Filho et al., 2025).

Stochastic updates, neural dynamics, and stability. Multiplicative noise, dropout, and stochastic depth regularize recurrent and deep networks (Jim et al., 1996; Lim et al., 2021; Srivastava et al., 2014; Huang et al., 2016; Bertsekas & Tsitsiklis, 1989; Combettes & Pesquet, 2015). We use classical stochastic stability tools (Khasminskii, 2012; Horn & Johnson, 2013) to derive exact second-moment lattice recursions. Residual dynamics, signal propagation, and nonnormal spectra further govern information flow and transient growth (Haber & Ruthotto, 2017; Poole et al., 2016; Schoenholz et al., 2017; Trefethen & Embree, 2005). Connections between cellular automata and neural networks provide architectural context (Gilpin, 2019; Springer & Kenyon, 2021). Optimization and recurrent dynamics in related implicit and continuous models are studied in Bengio et al. (1994); Pascanu et al. (2013); Chen et al. (2018); Bai et al. (2019).

## 8 CONCLUSION

Stochastic cell updates play different roles during learning and execution. Under one fixed-rate persist recipe, asynchronous training yields more final quality passes, while every passing asynchronous model can be evaluated deterministically. In a linear lattice, the same random mask both changes mean propagation and injects variance. After reconstruction, the states used during training distinguish retention from repair. These results make update schedule, state exposure, and task evaluation separate design choices for Growing NCA.

## ETHICS STATEMENT

This work studies small generative lattice models on two synthetic images. It involves no human subjects or personal data.

## REPRODUCIBILITY STATEMENT

The appendix provides the architecture, hyperparameters, evaluation definitions, original comparison criteria, model-level measurements, and proof details. It distinguishes prespecified comparisons from exploratory controls and states the update tapes and statistical units.

## AI USE DISCLOSURE

Large language models were used for language polishing. The authors reviewed and take responsibility for the manuscript.

## REFERENCES

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Per Bak, Chao Tang, and Kurt Wiesenfeld. Self-organized criticality: An explanation of the 1/f noise. Physical Review Letters, 59(4):381–384, 1987.

Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166, 1994.

Dimitri P. Bertsekas and John N. Tsitsiklis. Parallel and Distributed Computation: Numerical Methods. Prentice-Hall, 1989.

Bert Wang-Chak Chan. Lenia: Biology of artificial life. Complex Systems, 28(3):251–286, 2019.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Patrick L. Combettes and Jean-Christophe Pesquet. Stochastic quasi-Fejér block-coordinate fixed point iterations with random sweeping. SIAM Journal on Optimization, 25(2):1221–1248, 2015.

Martin Gardner. Mathematical games: The fantastic combinations of John Conway’s new solitaire game “life”. Scientific American, 223(4):120–123, 1970.

William Gilpin. Cellular automata as convolutional neural networks. Physical Review E, 100(3): 032402, 2019.

Daniele Grattarola, Lorenzo Livi, and Cesare Alippi. Learning graph cellular automata. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Eldad Haber and Lars Ruthotto. Stable architectures for deep neural networks. Inverse Problems, 34 (1):014004, 2017.

Roger A. Horn and Charles R. Johnson. Matrix Analysis. Cambridge University Press, 2nd edition, 2013.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q. Weinberger. Deep networks with stochastic depth. In European Conference on Computer Vision (ECCV), pp. 646–661, 2016.

Kam-Chuen Jim, C. Lee Giles, and Bill G. Horne. An analysis of noise in recurrent neural networks: convergence and generalization. IEEE Transactions on Neural Networks, 7(6):1424–1438, 1996.

Rafail Khasminskii. Stochastic Stability of Differential Equations. Springer, 2nd edition, 2012.

Mia-Katrin Kvalsund and James Stovold. Stability and geometry of attractors in neural cellular automata. arXiv preprint arXiv:2604.12720, 2026.

Christopher G. Langton. Computation at the edge of chaos: Phase transitions and emergent computation. Physica D: Nonlinear Phenomena, 42(1–3):12–37, 1990.

Soon Hoe Lim, N. Benjamin Erichson, Liam Hodgkinson, and Michael W. Mahoney. Noisy recurrent neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Atsushi Masumori, Hiroki Sato, and Takashi Ikegami. Structured fluctuations and the information dynamics of self-maintenance in growing neural cellular automata. arXiv preprint arXiv:2607.12403, 2026. URL https://arxiv.org/abs/2607.12403.

Alexander Mordvintsev, Ettore Randazzo, Eyvind Niklasson, and Michael Levin. Growing neural cellular automata. Distill, 2020. doi: 10.23915/distill.00023. https://distill.pub/2020/ growing-ca.

Alexander Mordvintsev, Ettore Randazzo, and Craig Fouts. Growing isotropic neural cellular automata. arXiv preprint arXiv:2205.01681, 2022.

Elias Najarro, Shyam Sudhakaran, Claire Glanois, and Sebastian Risi. HyperNCA: Growing developmental networks with neural cellular automata. In ICLR Workshop on From Cells to Societies, 2022.

Eyvind Niklasson, Alexander Mordvintsev, and Ettore Randazzo. Asynchronicity in neural cellular automata. In Proceedings ofthe Artificial Life Conference (ALIFE), 2021.

Ehsan Pajouheshgar, Yitao Xu, Tong Zhang, and Sabine Süsstrunk. DyNCA: Real-time dynamic texture synthesis using neural cellular automata. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

Rasmus Berg Palm, Miguel González Duque, Shyam Sudhakaran, and Sebastian Risi. Variational neural cellular automata. In International Conference on Learning Representations (ICLR), 2022.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning (ICML), pp. 1310–1318, 2013.

Sidney Pontes-Filho, Stefano Nichele, and Mikkel Elle Lepperød. Reservoir computing with evolved critical neural cellular automata. In Proceedings ofthe Artificial Life Conference (ALIFE), 2025.

Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. Exponential expressivity in deep neural networks through transient chaos. In Advances in Neural Information Processing Systems (NeurIPS), 2016.

Ettore Randazzo, Alexander Mordvintsev, Eyvind Niklasson, Michael Levin, and Sam Greydanus. Self-classifying MNIST digits. Distill, 2020. doi: 10.23915/distill.00027.002.

Hiroki Sato, Atsushi Masumori, and Takashi Ikegami. Transient state reorganization and cell differentiation in the developmental dynamics of growing neural cellular automata. arXiv preprint arXiv:2607.15726, 2026.

Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. In International Conference on Learning Representations (ICLR), 2017.

Lana Sinapayen. Self-replication, spontaneous mutations, and exponential genetic drift in neural cellular automata. arXiv preprint arXiv:2305.13043, 2023. URL https://arxiv.org/abs/ 2305.13043.

Martin Spitznagel and Janis Keuper. A new kind of network? review and reference implementation of neural cellular automata. arXiv preprint arXiv:2604.24990, 2026.

Jacob M. Springer and Garrett T. Kenyon. It’s hard for neural networks to learn the game of life. In International Joint Conference on Neural Networks (IJCNN), pp. 1–8, 2021.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15:1929–1958, 2014.

James Stovold. Identity increases stability of neural cellular automata. In Proceedings of the Artificial Life Conference (ALIFE), 2025. doi: 10.1162/isal.a.848.

Shyam Sudhakaran, Djordje Grbic, Siyan Li, Adam Katona, Elias Najarro, Claire Glanois, and Sebastian Risi. Growing 3D artefacts and functional machines with neural cellular automata. In Proceedings ofthe Artificial Life Conference (ALIFE), 2021.

Mattie Tesfaldet, Derek Nowrouzezahrai, and Christopher Pal. Attention-based neural cellular automata. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Lloyd N. Trefethen and Mark Embree. Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators. Princeton University Press, 2005.

Alan M. Turing. The chemical basis of morphogenesis. Philosophical Transactions of the Royal Society ofLondon B, 237(641):37–72, 1952.

John von Neumann. Theory ofSelf-Reproducing Automata. University of Illinois Press, 1966. Edited and completed by A. W. Burks.

Stephen Wolfram. Universality and complexity in cellular automata. Physica D: Nonlinear Phenom ena, 10(1–2):1–35, 1984.

## A STUDY DESIGN AND INTERPRETATION

Controlled comparisons. The update-mode comparison fixes the persist recipe and optimizer while varying the training update probability. The recipe comparison fixes the network and base optimization settings while changing the states used for training. All thirty recipe models pass a common short-horizon quality threshold.

Interpretation of measurements. The mean and second-moment calculations concern a fixed linear operator. The frozen Jacobian is measured at one state with its living masks held fixed. Finite-time perturbations follow a changing trajectory, while long-horizon retention and damage recovery directly evaluate task outcomes. These measurements describe different properties.

Protocol status. The update-mode, recipe, perturbation, recovery, and lattice comparisons follow their recorded protocols. Final-state inspection, lower-rate retraining, and living-mask removal are exploratory controls. Appendix D records the original endpoint definitions and outcomes, including the failed spatial-ratio endpoint and the marginal lattice cases.

## B PROOFS

The linear model is $x _ { t + 1 } = x _ { t } + M _ { t } A x _ { t }$ , where A is fixed and $M _ { t } = \mathrm { d i a g } ( m _ { u } )$ . The variables $m _ { u }$ are independent Bernoulli draws with parameter α, independent across cells and time and independent of $x _ { t }$ . The model does not include the living masks of Eq. (1).

## B.1 PROOF OF PROPOSITION 1

Independence gives $\mathbb { E } [ M _ { t } A x _ { t } ] = \alpha A \mathbb { E } [ x _ { t } ]$ , so

$$
\mathbb { E } [ x _ { t + 1 } ] = ( I + \alpha A ) \mathbb { E } [ x _ { t } ] .
$$

A finite-dimensional linear iteration converges to zero from every initial state exactly when all its eigenvalues have modulus below one. Substituting $a = \mu - 1$ yields

$$
| 1 + \alpha ( \mu - 1 ) | < 1 \iff \left| \mu - \left( 1 - { \frac { 1 } { \alpha } } \right) \right| < { \frac { 1 } { \alpha } } .
$$

The disk has center $1 - 1 / \alpha$ and radius $1 / \alpha$ . Its rightmost point is +1, which is a boundary point rather than a strictly stable multiplier. As α decreases, the disks approach the half-plane Re $\mu < 1$

For $\mu = e ^ { i \theta }$ , direct expansion gives

$$
| ( 1 - \alpha ) + \alpha e ^ { i \theta } | ^ { 2 } = ( 1 - \alpha ) ^ { 2 } + 2 \alpha ( 1 - \alpha ) \cos \theta + \alpha ^ { 2 } = 1 - 2 \alpha ( 1 - \alpha ) ( 1 - \cos \theta ) .
$$

This is strictly below one for $0 < \alpha < 1$ and $\theta \not \in 2 \pi \mathbb { Z }$ . At $\alpha = 1 / 2$ and $\theta = \pi$ , it is zero. On the other hand, if Re $\mu \geq 1$ , then

$$
\mathrm { R e } [ 1 + \alpha ( \mu - 1 ) ] = 1 + \alpha ( \mathrm { R e } \mu - 1 ) \geq 1 ,
$$

so the modulus cannot be below one. This also shows that any multiplier outside the unit disk admitted by $D _ { \alpha }$ must have real part below one. □

## B.2 PROOF OF THEOREM 1

Write $M _ { t } = \alpha I + D _ { t }$ , with centered independent diagonal entries of variance $\alpha ( 1 - \alpha )$ . Then

$$
x _ { t + 1 } = ( I + \alpha A ) x _ { t } + D _ { t } A x _ { t } .
$$

For the scalar translation-invariant operator on the $N ^ { d }$ torus, the unitary Fourier transform diagonalizes the first term,

$$
\widehat { ( I + \alpha A ) } x _ { t } ( \omega ) = ( 1 + \alpha \widehat { a } ( \omega ) ) \widehat { x } _ { t } ( \omega ) .
$$

Condition on $x _ { t }$ and put $v = A x _ { t }$ . The cross term between the mean update and the noise term has zero expectation. If $\bar { D } _ { t } = \mathrm { d i a g } ( d _ { u } )$ , independence of its entries gives

$$
\begin{array} { l } { \mathbb { E } \Big [ | \widehat { D _ { t } v } ( \omega ) | ^ { 2 } \Big | x _ { t } \Big ] = \displaystyle \frac { 1 } { N ^ { d } } \sum _ { u , u ^ { \prime } } e ^ { - i \omega \cdot ( u - u ^ { \prime } ) } \mathbb { E } [ d _ { u } d _ { u ^ { \prime } } ] v _ { u } v _ { u ^ { \prime } } ^ { * } } \\ { = \displaystyle \frac { \alpha ( 1 - \alpha ) } { N ^ { d } } \sum _ { u } | v _ { u } | ^ { 2 } } \\ { = \displaystyle \frac { \alpha ( 1 - \alpha ) } { N ^ { d } } \sum _ { \omega ^ { \prime } } | \hat { v } ( \omega ^ { \prime } ) | ^ { 2 } . } \end{array}
$$

The last step is Parseval’s identity. Taking expectation over $x _ { t }$ and using $\hat { v } ( \omega ^ { \prime } ) = \hat { a } ( \omega ^ { \prime } ) \hat { x } _ { t } ( \omega ^ { \prime } )$ proves Eq. (3). No independence assumption on different Fourier coefficients of $x _ { t }$ is needed.

Consider the non-neutral modes $\Omega _ { * } = \{ \omega \mid \hat { a } ( \omega ) \neq 0 \}$ . Their power vector evolves autonomously under the nonnegative matrix

$$
\mathcal { T } = \mathrm { d i a g } ( d _ { \alpha } ) + { \bf 1 } c ^ { \top } , \qquad c ( \omega ) = \frac { \alpha ( 1 - \alpha ) | \hat { a } ( \omega ) | ^ { 2 } } { N ^ { d } } .
$$

Mean-square decay requires $\rho ( \tau ) < 1$ . Since the injected power is nonnegative, $d _ { \alpha } ( \omega ) < 1$ on $\Omega ,$ <sub>∗</sub> is necessary. Under that condition, define the positive vector $\bar { q } ( \omega ) = [ 1 - \bar { d _ { \alpha } ( \omega ) } ] ^ { - 1 }$ . Its update satisfies

$$
( T q ) ( \omega ) = d _ { \alpha } ( \omega ) q ( \omega ) + { \cal S } , \qquad ( T q ) ( \omega ) - q ( \omega ) = { \cal S } - 1 .
$$

$\mathrm { I f } \ S < 1$ , then $\mathcal T q < q$ entrywise, which implies $\rho ( \mathcal { T } ) < 1 . \operatorname { I f } S \geq 1$ , then $\tau _ { q } \geq q$ , which implies $\rho ( \mathcal { T } ) \ge 1$ . These statements follow from the positive-vector characterization and monotonicity of the spectral radius for nonnegative matrices (Horn & Johnson, 2013). Hence Eqs. (3) and (4) give the necessary and sufficient condition for mean-square asymptotic decay.

For a neutral mode, $\hat { a } ( \omega ) = 0$ and $d _ { \alpha } ( \omega ) = 1$ . Its power receives the common injection but contributes nothing to its size. When the non-neutral modes decay, their injection is summable, so the neutral power remains bounded but need not tend to zero. This is why the theorem excludes neutral modes from the decay claim. □

## B.3 LOCALITY UNDER SPATIAL AND TEMPORAL EXTENSION

Proposition 2 (Spatial locality and temporal composition). Let a translation-equivariant local rule on $\bar { \mathbb Z } ^ { 2 }$ have dependence radius r in the maximum norm. After T steps, a cell’s state depends only on initial states within distance rT and, for random updates, the masks in the corresponding space-time dependency cone. Jointly translating the initial state and mask field translates the output. Independent identically distributed masks therefore give translation equivariance in distribution.

On a square grid of side $L ,$ at most

$$
1 - ( 1 - 2 r T / L ) _ { + } ^ { 2 }
$$

of the cells can have a dependency cone intersecting the boundary. Matching interior cones give matching states under coupled masks, or the same distribution under identically distributed masks. Deterministic time extension composes the update map repeatedly. For a random mask tape $\xi ,$ with $\Phi _ { T } ^ { \xi }$ denoting itsfirst T steps and $\sigma ^ { T } \boldsymbol { \xi }$ the shifted tape,

$$
\Phi _ { T + S } ^ { \xi } = \Phi _ { S } ^ { \sigma ^ { T } \xi } \circ \Phi _ { T } ^ { \xi } .
$$

Proof. One step enlarges the dependence neighborhood by at most r. Induction gives the radius-rT cone, including the masks used in its intermediate updates. Equivariance follows by translating every input to the same shared local rule. Counting cells at least rT from all four boundaries gives the fraction bound. The composition identity follows by splitting the consecutive updates at step T. □

For Eq. (1), the complete update has dependence radius at most two. The post-update living mask reads neighboring candidate states, each computed from a radius-one perception neighborhood. The bound therefore uses $r = 2 . \mathrm { \ A t } T = 9 6 , \bar { 2 r } T / L = 5 . 3 3 \mathrm { \ f o r } L = \bar { 7 } 2$ , and the boundary-fraction bound equals 1.00 at $L = 2 8 8$ . Spatial performance at these sizes is measured directly by the tiling evaluation in App. F; the proposition identifies the dependency structure and its distinction from additional temporal composition.

## C MEASUREMENT DEFINITIONS

The model is the statistical unit, with five training seeds per recipe and target. Confidence intervals use $1 0 ^ { 4 }$ bootstrap resamples of models within each target and group. Contrasts are evaluated separately on flower and heart. Group intervals are displayed to three decimal places.

Targets and initialization. Flower and heart are $4 0 \times 4 0 \ : \mathrm { R G B A }$ target images placed in $7 2 \times 7 2$ grids. Generation starts from the standard single-cell seed used by the implementation. The four visible channels and twelve hidden channels evolve together. The loss measures RGBA reconstruction rather than similarity between the white-background illustrations in the figures.

Quality test. A model passes when its target RGBA MSE at $T = 9 6$ is below 0.02. The update-mode comparison evaluates each model with its training update probability. Synchronous models use deterministic updates; asynchronous models use the quality tape with seed 42. All recipe models pass the same reconstruction threshold. Their stochastic $\dot { T } = \dot { 9 6 }$ errors are listed individually in Table 6.

Mode-swap difference. The evaluator records the signed change

$$
D _ { \mathrm { s w a p } } = E _ { 1 0 2 4 } ^ { \mathrm { u n m a t c h e d } } - E _ { 1 0 2 4 } ^ { \mathrm { m a t c h e d } } ,
$$

where the stochastic error is averaged over three tapes. The main text reports median absolute changes among quality-passing models. The signed values are retained in Table 7, so an improvement after switching remains visible. The recorded medians coincide with the absolute-change medians for the groups reported in the main text. The preregistered asymmetry test uses the signed differences as specified in its protocol.

Long-horizon outcomes. Checkpoints are recorded at $T \in \{ 9 6 , 2 5 6 , 5 1 2$ , 1024, 2048, 4096}. A rollout with no living cells is labeled dead. A non-finite state or max $| x | > 1 0 ^ { 3 }$ is labeled diverged. For surviving finite rollouts, final MSE above 0.15 is labeled off-target; the remainder is labeled bounded in the archived evaluator. We call this last category retained in the tables to distinguish it from numerical boundedness alone. The 0.15 long-horizon threshold differs from the 0.02 reconstruction threshold. Empty and divergent rollouts can contain propagated sentinel MSE values, which are not treated as ordinary measured errors.

Finite-time perturbation growth. The recipe-study reference state is reached at $T = 9 6$ using update tape seed 7100 + model index. Perturbations use Gaussian seeds 41, 42, and 43 with amplitude $1 0 ^ { - 3 }$ per state component. They are restricted to cells whose own alpha channel exceeds 0.1, rather than the neighborhood-pooled living mask used by the update rule. The perturbed and unperturbed trajectories share steps 96 through 159 of the reference tape. We compute Eq. (2) over those 64 steps, without renormalizing the perturbation, and average the three values.

States and norms use float32, followed by a scalar logarithm. A negative value denotes shrinkage of the sampled finite-amplitude perturbation over 64 steps. The perturbation is not renormalized or aligned to a maximal-growth direction, so the quantity differs from a maximum asymptotic Lyapunov exponent. Long-horizon and damage measurements evaluate the associated task behavior.

Frozen-state Jacobian. For an evaluation state $x _ { * }$ , define

$$
y _ { * } = x _ { * } + f _ { \theta } ( K * x _ { * } ) , \qquad g _ { * } = g ( x _ { * } ) \odot g ( y _ { * } ) .
$$

The differentiated map is $F _ { * } ( x ) = g _ { * } \odot [ x + f _ { \theta } ( K * x ) ]$ , holding the product of both living masks fixed. The recipe study uses its stochastic state at $T = 9 6 ;$ the factorial uses the matched-mode state at $T = 5 1 \bar { 2 }$ . The network and state are converted to float64 for forward-mode Jacobian-vector products. Arnoldi requests 32 eigenvalues by modulus with tolerance $1 0 ^ { - 8 }$ and a maximum of 3,000 iterations. The strict disk geometry in Fig. 6 uses $| \mu + 1 | < 2$ . The preregistered factorial placement endpoint instead uses its recorded tolerance, $| \mu + 1 | < 2 . 0 5$

Oscillation measurement. The secondary factorial endpoint uses deterministic rollouts in the retained class. It records the global living-cell mean RGBA trace from steps 2,048 through 4,096, subtracts each channel’s temporal mean, and takes the square root of the mean channel variance. This temporal amplitude is distinct from the spatial spectra used in the exploratory failure-state probe.

Damage recovery. $\mathrm { { A t } } T = 9 6 $ , circular damage with severity 0.30 is applied to a mature state. Clean and damaged copies share the same update masks for a recovery window of 80 steps. Let $E _ { \mathrm { p r e } }$ be the error before damage, $E _ { d }$ the error immediately afterward, and $E _ { \mathrm { f i n a l } }$ the recorded damaged-copy error at index $H - 1$ for $H = 8 0$ . The recovery score is

$$
R = 1 0 0 \frac { E _ { d } - E _ { \mathrm { f i n a l } } } { E _ { d } - E _ { \mathrm { p r e } } } .
$$

Full recovery gives $R = 1 0 0$ , no improvement gives $R = 0$ , and a larger error than immediately after damage gives $R < 0$ . Table 2 reports the mean recovery across models within each recipe and target.

Tiling. Worlds of side 72m contain $m ^ { 2 }$ seeds at tile centers for $m \in \{ 1 , 2 , 4 \}$ . Each model runs for 96 stochastic steps with a frozen tape. We first take the median $\mathbf { R G B A }$ error over its tiles, then the median over models within a recipe and target. The ratio endpoint instead forms each model’s $m = 4$ to $m = 1$ error ratio before taking the recipe median.

## D PRESPECIFIED COMPARISONS AND OBSERVATIONS

The comparison protocol was fixed before the corresponding evaluations. It specifies the seven quantitative criteria in Table 3 and a training-reliability analysis when at least six of ten synchronous models fail the quality test. The observed count of seven selects that analysis. Final-state inspection, lower-rate retraining, and living-mask removal are exploratory interventions.

The table keeps each original criterion alongside its observed value. The lattice criterion concerns all 27 configurations, with $2 6 / 2 7$ agreement. The $2 5 / 2 5$ non-marginal comparison is a subsequent diagnostic of that result, not a prespecified exclusion. Appendix E explains the boundary classification. The recipe contrasts and spatial comparison likewise retain their per-target intervals and error ratios.

The swap and spectral comparisons use quality-passing models as specified by their definitions. The temporal oscillation comparison uses retained deterministic rollouts and the global RGBA trace, rather than a spatial Fourier measurement. The recovery and living-mask results answer complementary behavioral questions and are not additional tests of these original criteria.

## E LINEAR LATTICE VALIDATION

The archived experiment uses a $3 2 \times 3 2$ periodic scalar lattice with the five-point Laplacian,

$$
\hat { a } ( \omega ) = c ( 2 \cos { \omega _ { x } } + 2 \cos { \omega _ { y } } - 4 ) \in [ - 8 c , 0 ] .
$$

Its 27 configurations combine $c \in \{ 0 . 2 0 , 0 . 2 5 , 0 . 3 0 , 0 . 3 5 , 0 . 4 0 , 0 . 4 5 , 0 . 5 0 , 0 . 6 0 , 0 . 7 0 \}$ with $\alpha \in$ $\{ 0 . 2 5 , 0 . 5 , 1 \}$ . Each configuration uses mask seeds 1, 2, and 3 and runs for up to 8,000 steps.

The constant Fourier mode is neutral and is removed by centering after every simulated update. The simulation starts with a centered Gaussian field of scale $1 0 ^ { - 3 }$ . It stops early at maximum absolute state below $1 0 ^ { - 1 4 }$ or above $1 0 ^ { 1 0 }$ . If neither threshold is reached, it labels a run as decay when its final maximum norm is smaller than its initial norm. A configuration is simulated as stable when all three runs are labeled decay.

This finite-time classification differs from asymptotic decay at a boundary. $\mathrm { A t } c = 0 . 2 5 , \alpha = 1$ , the checkerboard multiplier is exactly −1 and preserves its amplitude, yet the final norm is lower than the initial mixed-mode norm. The simulation therefore labels decay even though the strict theoretical criterion does not. The raw all-configuration result is $2 6 / 2 7$ agreement. A second configuration, $c = 0 . 5 0 , \alpha = 0 . 5 0$ , also lies on the mean-stability boundary. Reporting both marginal configurations separately leaves $2 5 / 2 5$ agreement and 4 mean-only misclassifications away from the boundary. This restricted comparison is a diagnostic of the recorded result, as distinguished from the original endpoint in App. D.

## F SPATIAL EXTENSION

Persist and regenerate are evaluated on worlds of side 72, 144, and 288, containing one, four, and sixteen seeds respectively. Figure 4 reports the per-target median tile errors. Every group median remains at the $1 0 ^ { ^ { \bullet - 4 } }$ scale and at least 30 times below the quality threshold. The largest per-model median tile error is $4 . 6 \times 1 0 ^ { - 3 }$

Table 3: Original comparison criteria and observed quantities. Synchronous and asynchronous refer to training. The complete lattice comparison and subsequent non-marginal diagnostic are listed separately within the same row.
<table><tr><td>Comparison</td><td>Original criterion</td><td>Observed quantities</td></tr><tr><td>Mode-swap asymmetry</td><td>Asynchronous-to-synchronous swap ratio above 3, the same direction on both targets, and ratio CI excluding 1.</td><td>CI [0.22, 1.24]; directions differ between targets.</td></tr><tr><td>Spectral placement At least 6 asynchronous models with</td><td> $\rho > 1 . 0 2 $  and all violating multipliers within  $| \mu + 1 | < 2 . 0 5 ;$  at most 2 qualifying synchronous survivors.</td><td>5 asynchronous models and 1 synchronous survivor qualify.</td></tr><tr><td>Temporal oscillation</td><td>Synchronous amplitude above twice the asynchronous amplitude on both targets.</td><td>The ordering reverses on flower.</td></tr><tr><td>Perturbation contrasts</td><td>Grow minus persist and grow minus regenerate intervals strictly above zero on both targets.</td><td>3/4 intervals exclude zero; heart grow minus regenerate is [−0.002, +0.037].</td></tr><tr><td>Long-horizon ordering</td><td>Grow median MSE at T = 4096 above persist and regenerate on both targets.</td><td>Grow 0.26/0.34; other groups at most  $1 . 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Spatial error ratio</td><td>Median per-model error ratio from m = 1 to m = 4 below 1.2 for both recipes.</td><td>Persist 1.45; regenerate 1.197.</td></tr><tr><td>Lattice agreement</td><td>Agreement with simulation on all 27 configurations and at least one mean-only error.</td><td> $2 6 / 2 7$  overall; subsequent non-marginal diagnostic 25/25, with 4 mean-only errors.</td></tr></table>

Table 4: All 27 lattice configurations arranged by update probability. Each block reports the meanonly prediction, second-moment prediction, simulated label, and S. A check denotes decay and a cross denotes no decay. A dagger marks an exactly marginal mean multiplier. Bold mean-only predictions identify the four non-marginal disagreements with simulation. An infinite S denotes failure of the prerequisite $d _ { \alpha } < 1$ in the implementation.
<table><tr><td>C</td><td colspan="4">α = 0.25</td><td colspan="4">α = 0.5</td><td colspan="4">α = 1.0</td></tr><tr><td></td><td>Mean</td><td>MS</td><td>Sim.</td><td>S</td><td>Mean</td><td>MS</td><td>Sim.</td><td>S</td><td>Mean</td><td>MS</td><td>Sim.</td><td>S</td></tr><tr><td>0.20</td><td>√</td><td>√</td><td>√</td><td>0.34</td><td>√</td><td>√</td><td>√</td><td>0.27</td><td>√</td><td>√</td><td>√</td><td>0.00</td></tr><tr><td>0.25</td><td>√</td><td>√</td><td>√</td><td>0.45</td><td>√</td><td>√</td><td>√</td><td>0.37</td><td>X</td><td>×</td><td>√t</td><td>∞</td></tr><tr><td>0.30</td><td>√</td><td>√</td><td>√</td><td>0.56</td><td>√</td><td>√</td><td>V</td><td>0.50</td><td>X</td><td>×</td><td>X</td><td>∞</td></tr><tr><td>0.35</td><td>√</td><td>√</td><td>√</td><td>0.68</td><td>√</td><td>√</td><td>√</td><td>0.67</td><td>X</td><td>×</td><td>X</td><td>∞</td></tr><tr><td>0.40</td><td>√</td><td>√</td><td>√</td><td>0.81</td><td>√</td><td>√</td><td>√</td><td>0.92</td><td>X</td><td>X</td><td>X</td><td>∞</td></tr><tr><td>0.45</td><td>√</td><td>√</td><td>√</td><td>0.96</td><td>√</td><td>X</td><td>X</td><td>1.35</td><td>X</td><td>X</td><td>X</td><td>∞</td></tr><tr><td>0.50</td><td>√</td><td>X</td><td>X</td><td>1.12</td><td>X</td><td>X</td><td>x†</td><td>∞</td><td>X</td><td>X</td><td>X</td><td>∞</td></tr><tr><td>0.60</td><td>√</td><td>X</td><td>X</td><td>1.51</td><td>X</td><td>X</td><td>X</td><td>∞</td><td>X</td><td>X</td><td>X</td><td>∞</td></tr><tr><td>0.70</td><td>√</td><td>X</td><td>X</td><td>2.02</td><td>X</td><td>X</td><td>X</td><td>∞</td><td>X</td><td>×</td><td>X</td><td>∞</td></tr></table>

The median per-model error ratio is 1.45 for persist and 1.197 for regenerate. Relative to the original threshold of 1.2, persist is above the threshold and regenerate below it. The absolute errors and ratios therefore describe different aspects of spatial extension. The former measures the achieved image quality on larger worlds; the latter measures its change relative to each model’s single-tile error.

## G EXPLORATORY INTERVENTIONS

## G.1 FINAL STATES OF FAILED SYNCHRONOUS MODELS

The probe runs each factorial checkpoint deterministically from the seed for 96 steps and computes channel-averaged two-dimensional spectra of its RGBA state and update field. The high-frequency fraction is the share of non-DC power with max $( | \omega _ { x } | , | \omega _ { y } | ) \ge \pi / 2$ . The checkerboard fraction uses frequencies within maximum-norm distance $\pi / 4$ of $( \pi , \pi )$ .

![](images/24ba58c380426ef5e49ac418e985a6158e66ced1cc9dde04cc72484d8578e80e.jpg)  
Figure 4: Absolute error under spatial extension. Per-target median tile MSE for one, four, and sixteen simultaneous seeds at $T = 9 6$ . The dashed line is the quality threshold. The ratio endpoint and the absolute-error observation are distinct statements.

Six quality-failing synchronous models are empty and have zero non-DC power in both objects. The seventh, heart seed three, is alive with deterministic MSE approximately 0.0290. Its high-frequency fractions are 0.0258 for the state and 0.0484 for the update. This final-state measurement identifies extinction as the most common terminal outcome, complementing the loss traces that follow training over time.

## G.2 LOWER LEARNING RATES

Flower seeds zero and two fail under synchronous training at $2 \cdot 1 0 ^ { - 3 }$ . Retraining each at $1 0 ^ { - 3 }$ and $5 \cdot 1 0 ^ { - 4 } $ gives four additional runs, with the other settings unchanged. All four pass the quality test. Figure 5 shows the trajectories for these two successful alternatives to the original learning rate.

## G.3 SINGLE-RUN CHECKPOINT REPLAY

We replayed one failed synchronous flower configuration using the original persist recipe, initialization seed, constant learning rate $2 \cdot 1 0 ^ { - 3 }$ , and 8,000-step budget. The first checkpoint saved when the logged minibatch loss fell below $1 0 ^ { - 3 }$ occurred at step 1,053. From the seed, this checkpoint has $T = 9 6 \ : \mathrm { M S E } \ : 0 . 0 0 2 0 9 9$ and $T = 4 0 9 6$ MSE 0.015803, with 552 living cells. The final checkpoint becomes empty and dies by step 47; its empty-canvas MSE is 0.059658. This one-run replay confirms that a temporarily usable checkpoint can occur in a failed configuration. It does not estimate the frequency of such checkpoints, identify why later updates change the outcome, or establish a training mechanism. The saved weights, training log, evaluation images, and protocol record are in the checkpoint-probe artifact directory.

## G.4 REMOVING THE LIVING MASKS

For each of the thirty recipe models, we replay the exact update-mask tape from its original long rollout. The three conditions retain both living masks throughout, remove both after $T = 9 6$ , or omit both from the seed. The gate-live condition reproduces all thirty original outcome labels.

Neither mask-removal condition produces a divergent or dead run. At $T = 4 { , } 0 9 6$ , the largest max |x| is approximately 16.5 after removal at maturity and 24.4 after removal from the seed. For the mature-state intervention, the median across models is 1.05. Among models with a real multiplier above one, the largest final magnitude is 16.5. Table 5 retains the per-group ranges and outcome counts.

![](images/32703140f73dd85f4353437ddb1f15ce0046996f93166ec66c8e7bd1c2635224.jpg)  
Figure 5: Lower-rate synchronous retraining. Two failed flower configurations are retrained at each of two lower learning rates. All four runs pass the quality test. The traces are exploratory and use the same training budget as the corresponding main runs. Zero-valued logged losses are drawn at $5 \cdot 1 0 ^ { - 6 }$ on the logarithmic axis.

Removing the masks at maturity changes two outcomes, both regenerate-heart models that become off-target. The remaining 28 outcomes are unchanged. Living-cell counts increase by more than 20% for 6 models in the mature-state intervention and 6 in the seed intervention. The largest increase is approximately 8.9 times the gate-live count. These finite-horizon interventions distinguish the masks spatial and pattern-retention effects from numerical boundedness.

## H FROZEN SPECTRAL MEASUREMENTS

The synchronous frozen-living-mask Jacobian has dimension 82,944. Forward-mode Jacobian-vector products allow Arnoldi iteration without forming the matrix, using the settings in App. C. Each model has a returned window of 32 eigenvalues, with no recorded solver error. The stated tolerance is the solver’s stopping tolerance; the reported quantities are the returned eigenvalues and their positions relative to the stability disks.

Figure 6 shows the returned eigenvalue locations and their relation to the measured long-horizon outcomes. It is included as a diagnostic comparison, not as a spectrum-based predictor of nonlinear fate.

Among the 803 returned eigenvalues with modulus above one, 499 lie inside the strict $D _ { 0 . 5 }$ disk, 145 are real and exceed one, and 159 are complex with real part above one. Regenerate contributes 288 values in the disk. Grow contributes 198 values with real part above one, compared with 32 for regenerate. Figure 7 displays the full recipe breakdown.

For 23 models, all 32 returned eigenvalues lie outside the unit disk. The total count is therefore a lower bound on the number of such multipliers in the full Jacobians. Real multipliers above one occur on all 10 grow models, all 10 persist models, and 4 regenerate models. Eight of those grow models become off-target, while all fourteen persist and regenerate models retain the pattern. These counts preserve the distinction between a frozen local expansion and an observed long-horizon outcome.

## I INDIVIDUAL MODEL RESULTS

The following tables retain the model-level measurements behind the summaries. Table 6 contains the thirty asynchronous recipe models. Table 7 contains the twenty factorial entries, ten of which reuse the persist models from the recipe study. The two tables therefore describe forty distinct base models.

Table 5: Living-mask ablation with matched random update tapes. Each group contains five models. Outcome counts give retained, off-target, and diverged or dead runs in that order. State magnitudes are ranges across the five models at $\bar { T _ { \mathrm { = } } } 4 { , } 0 9 6$
<table><tr><td colspan="2"></td><td colspan="2">Masks active</td><td colspan="2">Removed at  $T = 9 6$ </td><td colspan="2">Absent from seed</td></tr><tr><td>Recipe</td><td>Target</td><td>Outcomes</td><td>max |x|</td><td>Outcomes</td><td>max |x|</td><td>Outcomes</td><td>max |x|</td></tr><tr><td>Grow</td><td>Flower</td><td>1/4/0</td><td>[1.02, 3.97]</td><td>1/4/0</td><td>[1.02, 16.54]</td><td>1/4/0</td><td>[1.03, 24.35]</td></tr><tr><td rowspan="2">Persist</td><td>Heart</td><td>1/4/0</td><td>[1.07, 8.55]</td><td>1/4/0</td><td>[1.10, 9.62]</td><td>1/4/0</td><td>[1.12, 10.17]</td></tr><tr><td>Flower</td><td>5/0/0</td><td>[1.01, 1.09]</td><td>5/0/0</td><td>[1.01, 1.09]</td><td>5/0/0</td><td>[1.01, 1.02]</td></tr><tr><td rowspan="2">Regenerate</td><td>Heart</td><td>5/0/0</td><td>[0.99, 1.02]</td><td>5/0/0</td><td>[0.99, 1.05]</td><td>5/0/0</td><td>[0.99, 1.04]</td></tr><tr><td>Flower</td><td>5/0/0</td><td>[1.02, 1.39]</td><td>5/0/0</td><td>[1.02, 1.31]</td><td>5/0/0</td><td>[1.02, 1.14]</td></tr><tr><td rowspan="2"></td><td>Heart</td><td>5/0/0</td><td>[1.00, 1.35]</td><td>3/2/0</td><td>[1.00, 1.21]</td><td>3/2/0</td><td>[1.00, 1.46]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/142a62171ba67f027efb7813aabd73658ced11c6e7283c42d40ab0dc600960f5.jpg)  
Figure 6: A frozen local expansion measurement does not determine long-horizon task outcome. (a) The largest-modulus eigenvalue returned for each trained model at its frozen mature state. (b) Frozen-state radius and finite-time perturbation growth, with markers showing long-horizon outcome. The same condition $\rho > 1$ occurs in retained and off-target models.

## J REPRODUCIBILITY DETAILS

Training and hardware. The study contains thirty asynchronous recipe models and ten synchronous persist models. Reusing ten persist models in the update-mode comparison gives forty distinct base training runs. Four additional runs test lower learning rates. Training uses PyTorch on NVIDIA H200 hardware with float32 states and computations; frozen-Jacobian measurements use float64.

Hyperparameters. The network uses sixteen state channels, 128 hidden units, a zero-initialized output layer, and alpha threshold 0.1 for both living-mask checks. Training uses Adam at constant rate $2 \cdot { 1 0 ^ { - 3 } }$ , gradient-norm clipping at 1.0, batch size eight, 8,000 steps, and rollout lengths uniformly sampled from 64 through 96. Pool recipes use 1,024 states and seed replacement. Regenerate damages half of each batch with circular masks. The overflow penalty is based on deviation from clipped state values. The synchronous factorial initialization is seeded by 3100 + 10 target index + seed.

Evaluation controls. The recipe checkpoints are shared across the trajectory, spectral, damage, and spatial measurements. Update-mode swaps keep model weights fixed. Living-mask interventions reuse the original random update tapes, and perturbation comparisons share masks between clean and disturbed states. These controls preserve the intended comparison unit across the reported measurements.

![](images/41845e2756fe94147cba51e7a94c5d4776a03de21886203060b320817fa9fd52.jpg)  
Figure 7: Returned multipliers outside the unit disk. Counts are grouped by recipe and position relative to $D _ { 0 . 5 }$ and $\operatorname { R e } \mu = 1$ . They refer to the largest-modulus 32-eigenvalue window per model, not the complete spectrum.

Table 6: The thirty recipe models. Reconstruction errors, finite-time perturbation growth, largest returned frozen multiplier, and stochastic long-horizon outcome are listed for every training seed. Retained is the archived bounded outcome class. Outcome labels use unrounded measurements.
<table><tr><td>Recipe</td><td>Target</td><td>Seed</td><td>MSE@96</td><td> $\lambda _ { \mathrm { p e r t } }$ </td><td>ρ</td><td>MSE@4096</td><td>Outcome</td></tr><tr><td>Grow</td><td>Flower</td><td>0</td><td> $6 . 6 \times 1 0 ^ { - 5 }$ </td><td>+0.0198</td><td>1.134</td><td>0.39</td><td>Off-target</td></tr><tr><td></td><td></td><td>1</td><td> $1 . 4 \times 1 0 ^ { - 5 }$ </td><td>+0.0247</td><td>1.262</td><td>0.05</td><td>Retained</td></tr><tr><td></td><td></td><td>2</td><td> $2 . 0 \times 1 0 ^ { - 3 }$ </td><td>+0.0182</td><td>1.117</td><td>0.24</td><td>Off-target</td></tr><tr><td></td><td></td><td>3</td><td> $6 . 8 \times 1 0 ^ { - 5 }$ </td><td>+0.0135</td><td>1.078</td><td>0.26</td><td>Off-target</td></tr><tr><td></td><td></td><td>4</td><td> $2 . 3 \times 1 0 ^ { - 5 }$ </td><td>+0.0080</td><td>1.204</td><td>0.35</td><td>Off-target</td></tr><tr><td></td><td>Heart</td><td>0</td><td> $1 . 9 \times 1 0 ^ { - 5 }$ </td><td>+0.0239</td><td>1.204</td><td>0.34</td><td>Off-target</td></tr><tr><td></td><td></td><td>1</td><td> $6 . 7 \times 1 0 ^ { - 5 }$ </td><td>+0.0308</td><td>1.250</td><td>0.05</td><td>Retained</td></tr><tr><td></td><td></td><td>2</td><td> $5 . 1 \times 1 0 ^ { - 6 }$ </td><td>+0.0115</td><td>1.020</td><td>0.36</td><td>Off-target</td></tr><tr><td></td><td></td><td>3</td><td> $1 . 3 \times 1 0 ^ { - 5 }$ </td><td>+0.0249</td><td>1.066</td><td>0.36</td><td>Off-target</td></tr><tr><td></td><td></td><td>4</td><td> $1 . 9 \times 1 0 ^ { - 5 }$ </td><td>+0.0243</td><td>1.041</td><td>0.15</td><td>Off-target</td></tr><tr><td>Persist</td><td>Flower</td><td>0</td><td> $5 . 6 \times 1 0 ^ { - 5 }$ </td><td>-0.0314</td><td>1.123</td><td> $2 . 6 \times 1 0 ^ { - 5 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>1</td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td><td>-0.0227</td><td>1.262</td><td> $9 . 8 \times 1 0 ^ { - 3 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>2</td><td> $1 . 7 \times 1 0 ^ { - 3 }$ </td><td>-0.0066</td><td>1.143</td><td> $2 . 2 \times 1 0 ^ { - 3 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>3</td><td> $2 . 1 \times 1 0 ^ { - 5 }$ </td><td>-0.0324</td><td>1.053</td><td> $1 . 1 \times 1 0 ^ { - 5 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>4</td><td> $2 . 8 \times 1 0 ^ { - 4 }$ </td><td>-0.0254</td><td>1.131</td><td> $6 . 5 \times 1 0 ^ { - 4 }$ </td><td>Retained</td></tr><tr><td></td><td>Heart</td><td>0</td><td> $3 . 3 \times 1 0 ^ { - 5 }$ </td><td>-0.0388</td><td>1.042</td><td> $2 . 1 \times 1 0 ^ { - 5 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>1</td><td> $7 . 7 \times 1 0 ^ { - 4 }$ </td><td>-0.0246</td><td>1.125</td><td> $9 . 2 \times 1 0 ^ { - 4 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>2</td><td> $6 . 1 \times 1 0 ^ { - 5 }$ </td><td>-0.0355</td><td>1.187</td><td> $8 . 5 \times 1 0 ^ { - 5 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>3</td><td> $4 . 0 \times 1 0 ^ { - 3 }$ </td><td>-0.0251</td><td>1.111</td><td> $4 . 1 \times 1 0 ^ { - 3 }$ </td><td>Retained</td></tr><tr><td>Regenerate</td><td></td><td>4</td><td> $4 . 4 \times 1 0 ^ { - 5 }$ </td><td>-0.0428</td><td>1.086</td><td> $1 . 1 \times 1 0 ^ { - 5 }$ </td><td>Retained</td></tr><tr><td></td><td>Flower</td><td>0</td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td>-0.0088</td><td>1.200</td><td>0.04</td><td>Retained</td></tr><tr><td></td><td></td><td>1</td><td> $2 . 7 \times 1 0 ^ { - 4 }$ </td><td>-0.0141</td><td>1.272</td><td> $1 . 1 \times 1 0 ^ { - 3 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>2</td><td> $6 . 4 \times 1 0 ^ { - 5 }$ </td><td>+0.0175</td><td>1.261</td><td> $5 . 9 \times 1 0 ^ { - 5 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>3</td><td> $6 . 0 \times 1 0 ^ { - 4 }$ </td><td>-0.0098</td><td>1.329</td><td> $5 . 2 \times 1 0 ^ { - 5 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>4</td><td> $4 . 3 \times 1 0 ^ { - 4 }$ </td><td>-0.0224</td><td>1.237</td><td> $7 . 0 \times 1 0 ^ { - 4 }$ </td><td>Retained</td></tr><tr><td></td><td>Heart</td><td>0</td><td> $5 . 7 \times 1 0 ^ { - 4 }$ </td><td>-0.0065</td><td>1.302</td><td> $1 . 3 \times 1 0 ^ { - 3 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>1</td><td> $1 . 7 \times 1 0 ^ { - 3 }$ </td><td>+0.0078</td><td>1.312</td><td> $1 . 5 \times 1 0 ^ { - 3 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>2</td><td> $3 . 4 \times 1 0 ^ { - 4 }$ </td><td>-0.0165</td><td>1.306</td><td> $5 . 2 \times 1 0 ^ { - 4 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>3</td><td> $4 . 8 \times 1 0 ^ { - 4 }$ </td><td>-0.0185</td><td>1.140</td><td> $6 . 8 \times 1 0 ^ { - 4 }$ </td><td>Retained</td></tr><tr><td></td><td></td><td>4</td><td> $5 . 0 \times 1 0 ^ { - 3 }$ </td><td>+0.0460</td><td>1.188</td><td>0.10</td><td>Retained</td></tr></table>

Table 7: The twenty factorial entries. The last column is the signed mode-swap difference, so a negative value denotes lower error in the unmatched mode. A dagger denotes an empty model for which the frozen quality evaluator records a sentinel rather than an ordinary error value. Its true empty-canvas image loss is identified separately in the training traces and final-state probe.
<table><tr><td colspan="6"></td><td rowspan="2">Signed swap ∆MSE</td></tr><tr><td>Training</td><td>Target</td><td>Seed</td><td>Quality MSE</td><td>Pass</td><td>Det. outcome</td></tr><tr><td>sync</td><td>Flower</td><td>0</td><td>Dead†</td><td>No</td><td>Dead</td><td>n/a</td></tr><tr><td rowspan="10"></td><td></td><td>1</td><td> $2 . 1 \times 1 0 ^ { - 3 }$ </td><td>Yes</td><td>Retained</td><td> $2 . 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>2</td><td>Dead†</td><td>No</td><td>Dead</td><td>n/a</td></tr><tr><td></td><td>3</td><td>Dead†</td><td>No</td><td>Dead</td><td>n/a</td></tr><tr><td></td><td>4</td><td> $\mathrm { D e a d } ^ { \dagger }$ </td><td>No</td><td>Dead</td><td>n/a</td></tr><tr><td>Heart</td><td>0</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td><td>Yes</td><td>Retained</td><td> $1 . 9 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>1</td><td>Dead†</td><td>No</td><td>Dead</td><td>n/a</td></tr><tr><td></td><td>2</td><td> $1 . 5 \times 1 0 ^ { - 6 }$ </td><td>Yes</td><td>Retained</td><td> $3 . 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>3</td><td>0.03</td><td>No</td><td>Retained</td><td>n/a</td></tr><tr><td>4</td><td></td><td>Dead†</td><td>No</td><td>Dead</td><td>n/a</td></tr><tr><td>Flower</td><td>0</td><td> $1 . 7 \times 1 0 ^ { - 5 }$ </td><td>Yes</td><td>Retained</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="8"></td><td></td><td>1</td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td><td>Yes</td><td>Retained</td><td> $7 . 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>2</td><td> $1 . 9 \times 1 0 ^ { - 3 }$ </td><td>Yes</td><td>Retained</td><td> $2 . 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>3</td><td> $2 . 9 \times 1 0 ^ { - 5 }$ </td><td>Yes</td><td>Retained</td><td> $- 2 . 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>4</td><td> $8 . 7 \times 1 0 ^ { - 4 }$ </td><td>Yes</td><td>Retained</td><td> $3 . 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Heart</td><td>0</td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td><td>Yes</td><td>Retained</td><td> $3 . 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>1</td><td> $6 . 6 \times 1 0 ^ { - 4 }$ </td><td>Yes</td><td>Retained</td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>2</td><td> $7 . 8 \times 1 0 ^ { - 5 }$ </td><td>Yes</td><td>Retained</td><td> $6 . 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>3</td><td> $4 . 6 \times 1 0 ^ { - 3 }$ </td><td>Yes</td><td>Retained</td><td> $2 . 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>4</td><td> $5 . 1 \times 1 0 ^ { - 5 }$ </td><td>Yes</td><td>Retained</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td></tr></table>