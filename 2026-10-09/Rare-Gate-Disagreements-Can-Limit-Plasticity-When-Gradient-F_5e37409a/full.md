# Rare Gate Disagreements Can Limit Plasticity: When Gradient Flow Mispredicts Finite-Batch SGD

Ruoyu Zhao<sup>1</sup>, Mingxuan Zhang<sup>2</sup>, Jianbo Dai<sup>3</sup>, Jiaqi Wu<sup>1</sup>, Chenyu Zhu<sup>1</sup>, and Tong Che<sup>4,†</sup>

<sup>1</sup>City University of Hong Kong, {ruoyuzhao8-c,jwu395-c,chenyuzhu9-c}@my.cityu.edu.hk

<sup>2</sup>Microsoft, mingxzhang@microsoft.com

<sup>3</sup>Copula Lab, j6dj6d@gmail.com

<sup>4</sup>NVIDIA Research, tongc@nvidia.com

## Abstract

Population gradient flow is a common tool for reasoning about how neural networks adapt, including after pretraining. We show that it can mispredict finite-batch stochastic gradient descent (SGD) qualitatively, and we trace the discrepancy to a specific mechanism. In a two-unit ReLU regression, a source task drives the two neurons toward positive proportionality and a target task rewards separating them. After source training for time �, gradient flow recovers on the target in time linear in �. Online SGD with batch size � and step size � in both phases instead fails with high probability throughout a horizon of order �<sup>�/�</sup> once � ≳ log(�/�), uniformly on an explicit set of initializations with Gaussian probability above one percent. For each fixed �, small-step SGD still recovers, so the failure requires the joint limit of small steps and long pretraining. At the target clone, the population instability is carried entirely by inputs on which the two ReLU gates disagree. For units at angle � these inputs form a wedge of probability �/�, and weight decay shrinks the angle exponentially during pretraining. On every other input both units receive the same random linear update, which contracts their separation in conditional expectation. Bounding the cumulative probability of sampling the wedge along the exact online recursion, without a difusion approximation, shows that recovery with fixed probability from an identical source-gradient-flow checkpoint, within �<sup>�/�</sup> updates, requires �� $\gtrsim e ^ { \lambda \bar { T } }$ target samples and batch size $b \gtrsim \eta e ^ { \lambda T }$ , where � counts updates and � is the weight decay. In simulations, recovery is approximately a function of the disagreement budget $b \delta / \eta$ and saturates in the horizon.

## 1. Introduction

A network trained on one task can learn a second task more slowly than the same network trained from its original initialization. This loss of plasticity is documented in continual and reinforcement learning (Dohare et al., 2024; Lyle et al., 2023; Nikishin et al., 2022; Sokar et al., 2023), and related dificulties arise when warm-starting (Ash and Adams, 2020). Theoretical accounts often reason with the population gradient field, for example through manifolds of cloned (positively proportional) neurons and their stability (Joudaki et al., 2026) or the loss of curvature directions (Lewandowski et al., 2023). Training, however, uses finite batches. We ask whether the population field predicts how long finite-batch stochastic gradient descent (SGD) needs to adapt after pretraining.

Over a fixed horizon, small-step SGD tracks gradient flow (GF) under standard stochastic-approximation assumptions (Benaïm, 1999; Kushner and Yin, 2003). Exactly cloned neurons remain cloned under SGD (Joudaki et al., 2026). Gradient noise can even attract SGD to GF-unstable invariant sets such as sets of identical neurons, and smooth local coeficients at the set decide the attraction (Chen et al., 2023; Ziyin et al., 2023). Finite pretraining instead leaves distinct neurons at a small angle, and the population field predicts that they separate. For ReLU units the destabilizing term lives on the inputs where the two gates disagree, and pretraining makes these inputs rare. We prove that once pretraining exceeds order log(�/�), SGD stays near positively proportional neurons with high probability for a time exponential in 1/�: target SGD is unlikely to sample a disagreeing input before their probability shrinks further.

To make this precise, we work in a two-unit, biasfree ReLU network with inputs uniform on a disk. It is a minimal model in the tradition of small networks with analyzable training dynamics (Li et al., 2026; Saad and Solla, 1995; Tian, 2017; Xu and Du, 2023; Yehudai and Shamir, 2020). The source label �(� ) drives the units toward positive proportionality, a controlled pretraining history. The radial target ∥�∥ rewards separated units: proportional units have target risk at least 0.59, and target GF converges to risk below 0.27, so reaching risk 0.4 certifies separation. Without biases, the angle between the units determines gate disagreement. Rotational symmetry makes the population dynamics explicit, and bounded inputs bound sample gradients on bounded parameter sets. Both phases use ordinary online SGD, without tying, injected noise or balanced initialization. After source training for time $T ,$ GF in both phases recovers in time $\lambda T / ( 2 / \pi - \lambda ) + O ( 1 )$ (about 0.085� at $\lambda = 1 / 2 0 )$ . SGD in both phases fails with high probability throughout a horizon of order $e ^ { c / \eta }$ once $T \gtrsim \log ( b / \eta )$ , which is exponential in � when $T \lesssim 1 / \eta$ (Theorem 1, Corollary 2, Figure 1).

(a) Gates  
![](images/d18fd28d1e3b462de6248bc36036495fd78244d2264c07e6316d55e6629e99b0.jpg)

(b) Source  
![](images/f6f73b6205e263bc9cea23c695c992f4f86a502c475cc255be50a781a1e40492.jpg)

(c) Target  
(d) Risk  
![](images/edfc4bb149100e092675dc2d485c35b0b0b3a4e30cfa639b3b72c0e0906a0d75.jpg)

![](images/a46796c169843549e5f6dcb5c889ef2714f4bdcc7519e86574704eae0b2e7029.jpg)  
Figure 1 | Mechanism on one trajectory $( T = 4 0 0 , \eta = 1 0 ^ { - 3 } , b = 1 6 )$ . (a) Two gates at angle � disagree on wedges of probability $\delta / \pi$ . (b) Source training contracts the angle at rate �. (c,d) From the source-SGD checkpoint, target GD (blue; Euler steps of GF) separates the units and crosses risk 0.4 near the GF delay $\lambda T / ( 2 / \pi - \lambda )$ (arrow), while four target SGD streams (orange) sample no disagreement and stay at risk ≈0.6. (c) shares the vertical axis of (b); dashed: rates of Section 4; shading: $\Gamma < \pi \lambda$ , under one expected disagreement (Poisson approximation).

Mechanism. Two ReLU units at angle � disagree about which inputs are active only on wedges of probability $\delta / \pi$ (Figure 1a). At a balanced target clone, a stationary point with two positively proportional units, the destabilizing part of the population field comes entirely from this region (Proposition 4). On every other input, the same random linear map acts on both units, and an auxiliary process driven by the original independent batches contracts the separation of the two units in conditional expectation. Summing the disagreement probability along this contracting process bounds the probability of sampling a disagreeing input before the network leaves the proportional neighborhood. The bound is a constant multiple of $b d / \eta$ uniformly in the horizon, where the projective separation $d = \| w _ { 1 } / a _ { 1 } - w _ { 2 } / a _ { 2 } \|$ measures the distance from positive proportionality $( d = 2 \sin ( \delta / 2 )$ for balanced units). A wedge of probability $\delta / \pi$ needs about $\pi / \delta$ samples to be hit. Before the first hit the separation contracts at a rate of order $\lambda ,$ so all future batches together contain about ��/(���) disagreeing samples in expectation. We call

$$
\Gamma = \frac { b \delta } { \eta }
$$

the disagreement budget. Source GF contracts the angle at rate $\lambda ,$ and source SGD contracts � in expectation at a rate $\gamma > 0$ of the same order. Warm SGD therefore fails with high probability once $C b e ^ { - \gamma T } / \eta$ is small, i.e., after pretraining of order $\log ( b / \eta )$ . Both contractions come from weight decay: without it the balances $a _ { i } ^ { 2 } -$ $\| \boldsymbol { w } _ { i } \| ^ { 2 }$ are conserved under GF (Du et al., 2018) and the source angle equation loses its linear term, so the angle decays only like $1 / T$

## Contributions.

• A two-phase separation theorem for online SGD (Theorem 1, Corollaries 2 and 3). On one event of high probability, fresh SGD succeeds in bounded time and warm SGD fails throughout a horizon of order $e ^ { c / \eta }$ , uniformly on an explicit set of initializations with Gaussian probability above 1.06%. For each fixed pretraining duration, small-step SGD recovers as GF predicts, so the failure occurs only in the joint limit of small � and growing �. The central estimate, a first-disagreement coupling with a summable gate hazard, is proved in full in Section 7.

• Identical-checkpoint separation and sampling requirements (Corollary 5). From a source-GF checkpoint, target GF recovers in �(�) time, whereas recovery by online SGD with fixed probability over horizons of at most order $e ^ { c / \eta }$ updates requires $N b \gtrsim e ^ { \lambda T }$ target samples and $b \gtrsim \eta e ^ { \lambda T }$

• Simulations of the two-unit model in which recovery tracks Γ across batch sizes, step sizes and checkpoint angles $\delta _ { 0 }$ from $1 0 ^ { - 5 } ~ \mathrm { t o } ~ 1 0 ^ { - \dot { 2 } }$ , and saturates in the horizon when weight decay contracts the separation.

## 2. Related Work

Our result connects plasticity loss, stochastic attraction to invariant sets, neuron alignment and stochastic approximation.

Plasticity and redundant representations. Plasticity loss has been linked to dormant units, rank collapse, curvature and weight growth (Dohare et al., 2024; Kumar et al., 2021; Lewandowski et al., 2023; Lyle et al., 2023, 2025; Sokar et al., 2023), and representation and kernel ranks can mispredict trainability (Wang et al., 2026). Joudaki et al. (2026) establish clone invariance across broad architectures and optimizers, examine stability, and study perturbations that help restore learning. Invariance explains why an exact clone persists; we bound how long SGD needs to separate the nearby, distinct neurons that finite pretraining produces, through the cumulative probability of sampling inputs that distinguish them.

Stochastic attraction to invariant sets. Chen et al. (2023) show how gradient noise can attract SGD to invariant subnetworks that are unstable under GF, including permutation-invariant sets of identical neurons. Type-II saddle analysis studies random matrix products and probabilistic stability when gradient noise vanishes at the saddle (Ziyin et al., 2023); constructed examples also show convergence of constant-step SGD to local maxima (Ziyin et al., 2022). In these analyses stability is decided by local coeficients at the invariant set, such as curvatures of the loss and of the noise covariance, and the mean-curvature prediction is recovered as the step size decreases at fixed coefficients (Ziyin et al., 2023). Here the destabilizing term lives on a gate-boundary region that shrinks with the separation (Section 4). At the clone the averaged fixed-input linearization contracts at rate $\lambda ,$ , independently of $\eta ,$ so such criteria predict capture. Recovery instead depends on whether the checkpoint lies within angle of order $\eta / b$ of the clone; pretraining with weight decay gets there after time of order $\lambda ^ { - 1 } \log ( b / \eta )$ . We bound the cumulative probability of sampling the gatedisagreement region along the exact online recursion, without a difusion approximation. It yields a hittingtime bound that separates SGD from GF in the joint limit of vanishing step size and growing pretraining duration.

Alignment and population escape. Weight decay can favor homogeneous neurons (Ziyin, 2024); small initialization can lead to early ReLU alignment (Maennel et al., 2018); and noise can drive motion along symmetry directions (Ziyin et al., 2024). Neuron duplication also creates critical subspaces (Şimşek et al., 2021). Plateaus near such singularities and subsequent specialization are classical (Fukumizu and Amari, 2000; Saad and Solla, 1995; Wei et al., 2008), and online SGD from a random start in high dimension spends most of its samples escaping the region of small correlation with the signal (Ben Arous et al., 2021). Li et al. (2026) quantify approach to a bisector saddle and escape under population GF in a two-neuron teacher– student model. Their logarithmic population escape law is our comparator; finite-batch SGD from the same checkpoint needs exponentially many samples in the source-training time.

Stochastic approximation and saddle avoidance. Efective SGD dynamics can difer from GF in highdimensional critical scalings (Ben Arous et al., 2022), and stochastic modified equations refine fixed-horizon tracking for smooth losses (Li et al., 2017). Avoidance of unstable points requires suitable excitation and regularity (Brandière and Duflo, 1996; Mertikopoulos et al., 2020; Pemantle, 1990); perturbation-based escape results cover smooth saddles (Ge et al., 2015; Jin et al., 2017), with related results for classes of nonsmooth objectives (Bianchi et al., 2024; Davis et al., 2026). For Gaussian inputs, ReLU population risks have closed-form gradients (Tian, 2017); the gate boundary enters population Hessians through Dirac terms that almost every sample Hessian of a piecewise-linear unit misses (Zhong et al., 2017), and empirical and population landscapes can difer (Jin et al., 2018). We quantify the consequence of such a boundary term when its supporting region shrinks during training: fixedhorizon tracking remains valid, but its recovery prediction fails once pretraining exceeds order $\log ( b / \eta )$

## 3. Setting and Main Results

Model, tasks, and training. Let � be uniform on the disk $\{ x \in \mathbb { R } ^ { 2 } : \| x \| \leq 2 \}$ and

$$
\begin{array} { r } { f _ { \theta } ( x ) = a _ { 1 } \sigma ( w _ { 1 } ^ { \top } x ) + a _ { 2 } \sigma ( w _ { 2 } ^ { \top } x ) , } \end{array}
$$

with $\sigma ( z ) ~ = ~ \operatorname* { m a x } \{ z , 0 \}$ and all six parameters $\theta \ =$ $( a _ { 1 } , w _ { 1 } , a _ { 2 } , w _ { 2 } )$ trainable. The source and target labels are

$$
y _ { s } ( x ) = \sigma ( x _ { 1 } ) , \qquad y _ { t } ( x ) = \left\| x \right\| .
$$

The source depends on one direction; the target is radial and rewards opposed units. For a label $y ,$ the population risk and training objective are

$$
\begin{array} { r } { R _ { y } ( \theta ) = \frac 1 2 \mathbb { E } ( f _ { \theta } ( X ) - y ( X ) ) ^ { 2 } , \quad F _ { y } = R _ { y } + \frac \lambda 2 \left. \theta \right. ^ { 2 } , } \end{array}
$$

with a fixed weight decay $\lambda \in ( 0 , 1 / 5 ]$ (Appendix D). Numerical values are given at $\lambda = 1 / 2 0 _ { \mathrm { { ; } } }$ , the value used in the simulations. Training minimizes $F _ { y }$ ; adaptation is measured by $R _ { y } ,$ with subscripts $s , \mathbb { i }$ � for $y _ { s } , y _ { t }$ . SGD with batch size $b \geq 1$ and step size $\eta > 0$ uses independent fresh inputs at every update:

$$
\theta _ { n + 1 } = ( 1 - \eta \lambda ) \theta _ { n } - \frac { \eta } { b } \sum _ { j = 1 } ^ { b } \nabla \ell _ { y } ( \theta _ { n } ; X _ { n , j } ) ,\tag{1}
$$

where $\ell _ { y } ( \theta ; x ) = { \textstyle { \frac { 1 } { 2 } } } ( f _ { \theta } ( x ) - y ( x ) ) ^ { 2 }$ and $\sigma ^ { \prime } ( 0 ) = 0$ . Physical time is ��. The population comparator is gradient flow (GF) $\dot { \theta } = - \nabla F _ { y } ( \theta )$

The disk keeps the population geometry of Gaussian inputs while bounding sample gradients. Writing $\boldsymbol { X } = \boldsymbol { R } \hat { \boldsymbol { X } }$ with uniform direction ${ \dot { \hat { X } } } ,$ the model and both labels are positively homogeneous of degree one, so the squared loss depends on the radial law only through $\mathbb { E } R ^ { 2 }$ . Both this disk and $N ( 0 , I _ { 2 } )$ have $\mathbb { E } R ^ { \overset { \cdot } { 2 } } = 2$ and therefore the same population risk and GF equations; bounded inputs are used in the stochastic confinement argument.

Initialization. Write $r _ { i } = \| \boldsymbol { w } _ { i } \| , \omega _ { i } = a _ { i } r _ { i } , B = \omega _ { 1 } + \omega _ { 2 }$ and let $\vartheta _ { i } \in \left( - \pi , \pi \right)$ be the angle of $w _ { i }$ from $e _ { 1 } .$ . Let D be the set of parameters in which both neurons have positive heads and lie on opposite sides of the source direction,

$$
a _ { i } > 0 , 0 < \omega _ { i } < 1 , \vartheta _ { 1 } > 0 > \vartheta _ { 2 } , \vartheta _ { 1 } - \vartheta _ { 2 } < \pi ,
$$

together with its neuron exchange, and $\mathcal { D } _ { 0 } = \mathcal { D } \cap$ $\{ B < 3 / 4 \}$ . Appendix I constructs a compact $K \subset \mathcal { D } _ { 0 }$ with explicit positive parameter and angular margins and proves $\mathbb { P } _ { N ( 0 , I _ { 6 } ) } ( K ) > 0 . 0 1 0 6 4$ , close to $\mathbb { P } ( \mathcal { D } _ { 0 } )$ ≈ 0.01083.

A risk gap. Put $q = 2 / \pi . \mathrm { ~ I f ~ } ( a _ { 1 } , w _ { 1 } ) = c ( a _ { 2 } , w _ { 2 } )$ with $c > 0$ , the predictor is a single ReLU and has target risk at least $R _ { \mathrm { c o l } } = 1 - q ^ { 2 } \approx 0 . 5 9 4 7$ . Target GF from � converges to opposed units with risk $R _ { \mathrm { g o o d } } = 1 - 2 q ^ { 2 } \cdot$ + $2 \lambda ^ { 2 } < 0 . 2 7 ( \approx 0 . 1 9 4 4$ at $\lambda = 1 / 2 0 )$ . Thus

$$
R _ { \mathrm { g o o d } } < 0 . 4 < R _ { \mathrm { c o l } } ,\tag{2}
$$

and reaching risk 0.4 requires leaving a neighborhood of the positively proportional family. Every $\theta _ { 0 } \in K$ starts with target risk above 0.5928; Appendices G and I establish these risk bounds.

Fresh and warm runs. Fix $\theta _ { 0 } \in K$ . The fresh run applies target SGD from $\theta _ { 0 }$ . The warm run trains on the source for physical time $T \ ( \lfloor T / \eta \rfloor$ updates) and then on the target with fresh batches, using the same � and $\eta .$ . Let $\tau _ { f }$ and $\tau _ { w } ( T )$ be the first target times at which population risk is at most 0.4 (multiples of � for SGD), and let $\tau _ { \mathrm { G F } } ( T )$ be the same time when both phases use gradient flow. Probabilities are over sample streams, conditional on $\theta _ { 0 } ;$ the bounds below hold for any coupling of the fresh and warm streams.

Theorem 1 (Linear GF delay versus metastable SGD failure). (a) Gradient flow. Uniformly over $\theta _ { 0 } \in K ,$ fresh target GF reaches risk 0.4 within a fixed time, and the warm GF hitting time satisfies

$$
\tau _ { \mathrm { G F } } ( T ) = \frac { \lambda } { 2 / \pi - \lambda } T + O ( 1 ) .\tag{3}
$$

(b) Finite-batch SGD. $F i x \varepsilon \in ( 0 , 1 )$ . There are constants $L , t _ { 0 } , C , c _ { * } , \gamma , \eta _ { 0 } > 0 _ { \mathrm { \small i } }$ , depending on the model, �, � and � but not on the batch size, such that for every $b \geq 1 ,$ $\theta _ { 0 } \in K , 0 < \eta \le \eta _ { 0 } , T \ge L + 1$ and $H \geq 0 ,$ with conditional probability at least

$$
1 - \varepsilon - \frac { C b } { \eta } e ^ { - \gamma T } - C \Bigl ( 1 + \frac { T + H } { \eta } \Bigr ) e ^ { - c _ { * } / \eta } ,\tag{4}
$$

the fresh run succeeds, $\tau _ { f } \leq t _ { 0 } ,$ , and the warm runfails throughout the horizon, $\tau _ { w } ( T ) > H$

Here � is a source burn-in time, � the contraction rate of the source projective separation and $c _ { * }$ a confinement rate. Any $\gamma < \lambda$ is admissible, at the cost of larger $L , C$ and smaller $c _ { * } , \eta _ { 0 }$ , and we fix some $\gamma \in [ \lambda / 2 , \lambda )$ . At the source minima the averaged first-order linearization of the projective update, with both gates evaluated at one unit, is $I { - } \eta \lambda ( I { + } e _ { 1 } e _ { 1 } ^ { \top } ) \preceq ( 1 { - } \eta \lambda ) I$ , and the disagreement correction is absorbed by shrinking the neighborhood and the step size (Appendix B.1). Through a Markov estimate after the burn-in, $\varepsilon / 2$ bounds the probability that the source phase drifts to an unfavorable allocation log $\left( a _ { 1 } / a _ { 2 } \right)$ , the relative scale of the two units along the clone family. A longer burn-in makes it smaller. The second term bounds, up to constants, � � $\left[ d _ { T } \right] / \eta$ on the good source event, where $d _ { T }$ is the projective separation at the checkpoint (Section 6). The last term bounds excursions of the shared parameters (amplitude, balance) from the neighborhoods used in the proof.

Corollary 2 (Metastable window). Take $\varepsilon = 0 . 0 0 5 ,$ $f { \ddot { \imath } } x \ b ,$ and $f i x ~ C _ { 0 }$ with $\gamma C _ { 0 } > 1$ (for instance $C _ { 0 } > 2 / \lambda )$ For all suficiently small $\eta ,$ uniformly over $\theta _ { 0 } \in K ,$ each prescribed source duration and horizon with

$$
C _ { 0 } \log ( 1 / \eta ) \le T \le e ^ { c _ { * } / ( 2 \eta ) } , \qquad H \le e ^ { c _ { * } / ( 2 \eta ) }
$$

gives $\tau _ { f } \leq t _ { 0 }$ and $\tau _ { w } ( T ) > H$ with conditional probability at least 0.99. In particular, for constants $c _ { 0 } , c _ { 1 } > 0$ with $c _ { 0 } c _ { 1 } < c _ { * } / 2$ and $C _ { 0 } \log ( 1 / \eta ) \le T \le c _ { 0 } / \eta$ the warm run fails up to the exponential horizon $H = e ^ { c _ { 1 } T }$

Proof. For $T \geq C _ { 0 } \log ( 1 / \eta )$ , the second term in (4) is at most $C b \eta ^ { \gamma C _ { 0 } - 1 }  0$ . The last term is at most $C ( 1 +$ $2 \eta ^ { - 1 } e ^ { c _ { * } / ( 2 \eta ) } \dot { ) } e ^ { - c _ { * } / \eta } \to 0$ . Eventually $C _ { 0 } \log ( 1 / \eta ) \geq L + 1$ For the particular case, $T \leq c _ { 0 } / \eta \leq e ^ { c _ { * } / ( 2 \eta ) }$ and $H =$ $e ^ { c _ { 1 } T } \leq e ^ { c _ { 1 } \hat { c } _ { 0 } / \eta } \leq e ^ { c _ { * } / ( 2 \eta ) }$ for small $\eta .$ □

Over Gaussian initialization the guarantee holds with probability at least $0 . 9 9 \mathbb { P } ( K ) > 1 \%$ , and smaller � raises the conditional confidence to any level below one.

Two regimes. The guaranteed horizon $e ^ { c _ { * } / ( 2 \eta ) }$ does not depend on �; pretraining enters through the term $C b e ^ { - \gamma T } / \eta$ . The horizon comes from exponential concentration of the shared parameters iterated over blocks of fixed physical time, as in small-noise metastability (Freidlin and Wentzell, 2012). For fixed pretraining, warm SGD instead recovers in bounded time, as gradient flow predicts:

Corollary 3 (Fixed pretraining duration). For each $\theta _ { 0 } \in { \cal K }$ and $T \geq 0$ there are $t _ { 1 } , C , c , \eta _ { 2 } > 0 ,$ , independent $o f b ,$ such that $\mathbb { P } ( \tau _ { w } ( T ) \le t _ { 1 } \ | \ \theta _ { 0 } ) \ge 1 - C e ^ { - c / \eta }$ for every $b \geq 1$ and $0 < \eta \leq \eta _ { 2 }$

Corollary 3 cannot hold with a step-size threshold and constants uniform in �, since by Corollary 2 warm SGD fails once $T \geq C _ { 0 } \log ( 1 / \eta )$ . In the joint sweep (Appendix J, Figure 4), the source time at which warm recovery is lost grows as � decreases.

## 4. The Instability Lives on the Gate Boundary

This section locates the population instability behind Theorem 1 on the gate boundary. Source gradient flow from � converges (Proposition 11, Appendix F) to minima in which both units stay active, with both neurons along $e _ { 1 { \mathrm { : } } }$ , balanced parameters $a _ { i } ~ = ~ \| u _ { i } \|$ , and total amplitude $B _ { s } = 1 - 2 \lambda$ . The target has a proportional stationary family with amplitude $B _ { t } = 2 ( q - \lambda )$ . Put $A = q - \lambda$ and $s _ { * } = { \sqrt { A } }$ , start from $a _ { i } = s _ { * } , w _ { i } = s _ { * } e _ { 1 }$ , and split the hidden vectors transversally:

$$
w _ { 1 } = ( s _ { * } , \varphi ) , \qquad w _ { 2 } = ( s _ { * } , - \varphi ) .
$$

The common-gate field applies the shared gate ${ \mathbf { 1 } } _ { x _ { 1 } > 0 }$ to both units: its predictor is $\begin{array} { r } { \mathbf { 1 } _ { x _ { 1 } > 0 } \sum _ { i } { a _ { i } } { w _ { i } ^ { \top } } { x } . } \end{array}$ , and its sample update applies the corresponding residual and gate to each full neuron. Samplewise, it coincides with SGD whenever the two gates agree.

Proposition 4 (Instability carried by gate disagreements). Let $V = - \nabla F _ { t }$ , let $V _ { \mathrm { { c o m } } }$ be the common-gate mean field, and let $\nu = \textstyle { \frac { 1 } { 2 } } ( V _ { w _ { 1 2 } } - V _ { w _ { 2 2 } } ) _ { ; }$ , with $V _ { w _ { i 2 } }$ the second coordinate of the �<sub>�</sub>-component of �, and define $\upsilon _ { \mathrm { c o m } }$ likewise from $V _ { \mathrm { { c o m } } }$ . Then

$$
\nu ( \varphi ) = A \varphi + O ( \varphi | \varphi | ) , \qquad \nu _ { \mathrm { c o m } } ( \varphi ) = - \lambda \varphi .
$$

Their per-sample integrands agree outside a disagreement region ofprobability 2 arctan $( \vert \varphi \vert / s _ { * } ) / \pi ,$ so their mean diference $q \varphi + O { \bigl ( } \varphi | \varphi | { \bigr ) }$ is carried entirely by that region.

Proof. The separation is $\delta = 2 \arctan ( \lvert \varphi \rvert / s _ { * } )$ , and the exact population angular equation (Appendix E) gives $\dot { \delta } ~ = ~ \bar { 2 } s _ { * } ^ { 2 } g ( \delta ) ( 1 + \bar { O ( \varphi ^ { 2 } ) } )$ with $g ( \delta ) ~ = ~ \delta / 2 + { \cal O } ( \delta ^ { 2 } )$ while radial drift is $O ( \varphi ^ { 2 } )$ . For the common gate, $\begin{array} { r } { U \ = \ \sum _ { i } a _ { i } w _ { i } \ = \ 2 A e _ { 1 } } \end{array}$ , and rotational symmetry gives $\mathbb { E } [ ( \mathbf { 1 } _ { X _ { 1 } > 0 } U ^ { \top } X - \left. X \right. ) \mathbf { 1 } _ { X _ { 1 } > 0 } X ] = U / 2 - q e _ { 1 } = - \lambda e _ { 1 }$ , with zero transverse component, so transverse coordinates decay at rate �. □

At $\lambda = 1 / 2 0 , A \approx 0 . 5 8 7 _ { ; }$ , so the boundary contribution $q \approx 0 . 6 3 7$ reverses the sign of the common-gate rate $- 0 . 0 5 ;$ Figure 1(c) shows both rates on actual trajectories.

Sample and population linearizations. For almost every fixed input, a small enough splitting leaves both gates unchanged, so the fixed-input linearization of the sample gradient averages to the common-gate contraction. The sign reversal comes from inputs in a region that moves and shrinks with the splitting. For smooth sample losses with integrable derivative bounds, diferentiation and expectation commute, so averaging the sample Jacobian recovers the population Jacobian; the moving gate boundary breaks this interchange. The population rate � is recovered only through samples of probability $\delta / \pi ,$ , and one such sample changes the splitting by order $\eta / b _ { ; }$ , a relative change of order $\eta / ( b \delta )$ Applied at the clone, a criterion built from the averaged fixed-input linearization predicts contraction and one built from the population linearization predicts escape; both are fixed by the clone and cannot see how long pretraining lasted.

Collapse without inactive neurons. Each neuron activates on half the distribution for every �, while the probability that the two gates difer vanishes as $\delta / \pi$ . Activation frequency therefore misses the loss of distinct responses: sampled gradients act on both neurons through the same map, rescaling the shared output but rarely separating the units.

Amplitude adjustment at the task switch. The source and target clone amplitudes difer: $B _ { s } = 1 - 2 \lambda <$ $B _ { t } = 2 ( q - \lambda )$ (0.9 and ≈1.173 at $\lambda = 1 / 2 0 )$ . On the balanced proportional family, the target equation is $\dot { B } = B \big ( B _ { t } - B \big )$ , so the task switch adjusts the shared amplitude while preserving proportionality. Throughout this transient, $B \ < \ 2 q$ , the range in which the common-gate estimate (8) contracts transverse separation. Adapting the output scale thus lowers risk toward the target-clone value without creating the angular separation needed to cross 0.4.

The linear population delay. Population source training gives $\delta _ { s } ( T ) = C _ { s } e ^ { - \lambda T } ( 1 + o ( 1 ) )$ with $C _ { s }$ bounded above and away from zero on � (Appendix H). Near the clone family the target angle grows at rate � after a bounded amplitude transient, so

$$
\tau _ { \mathrm { G F } } ( T ) = \frac { 1 } { A } \log \frac { 1 } { \delta _ { s } ( T ) } + O ( 1 ) = \frac { \lambda } { A } T + O ( 1 ) .
$$

This is the expected logarithmic dependence on initial asymmetry for escape from a symmetric plateau (Saad and Solla, 1995; Wei et al., 2008).

## 5. Identical Checkpoints and Sample Sizes

Theorem 1 uses SGD in both phases. The next result starts both target optimizers from one source-GF checkpoint, so only target sampling difers.

Corollary 5 (Identical checkpoint). Let $\theta _ { s } ^ { \mathrm { G F } } ( T )$ be the source GF checkpointfrom $\theta _ { 0 } \in K ,$ , and let $\theta _ { n } ^ { \mathrm { { \check { S } G D } } }$ be target SGD started at $\dot { \theta } _ { s } ^ { \mathrm { G F } } ( \dot { T } )$ . There are $C , c , T _ { 1 } , \eta _ { 1 } > 0$ such that for all $T \geq T _ { 1 } , 0 < \eta \leq \eta _ { 1 } , b \geq 1$ , integers $N \geq 1$ and $\theta _ { 0 } \in K _ { i }$

$$
\begin{array} { r l } {  { \mathbb { P } \Big ( \operatorname* { m i n } _ { n \leq N } R _ { t } ( \theta _ { n } ^ { \mathrm { S G D } } ) \leq 0 . 4 \Big ) } \quad } & { } \\ & { \leq C b e ^ { - \lambda T } \operatorname* { m i n } \{ N , \eta ^ { - 1 } \} + C ( N + 1 ) e ^ { - c / \eta } , } \end{array}\tag{5}
$$

while target GFfrom the same checkpoint reaches 0.4 at the time given by (3).

Proof. Source GF gives positive limiting amplitudes, angle $\leq C e ^ { - \lambda T }$ (Lemma 13) and balance error $\bar { O } ( e ^ { - 2 \lambda T } ) ,$ so $\bar { \| } w _ { 1 } / a _ { 1 } - w _ { 2 } / a _ { 2 } \| \leq C e ^ { - \lambda T }$ . By (26) in Appendix D, the target projective coordinate obeys the same bound, and for large � the checkpoint lies in the target neighborhood of Proposition 6. Apply Proposition $6 .$ □

Necessary sample and batch sizes. If target SGD from $\theta _ { s } ^ { \mathrm { G F } } ( \bar { T } )$ reaches 0.4 within � steps with probability at least $p ,$ and $C ( N + 1 ) e ^ { - c / \eta } \leq p / 2$ , then (5) forces � min $\{ N , \eta ^ { - 1 } \} \ge c _ { p } e ^ { \lambda T }$ with $c _ { p } = p / ( 2 C )$ . That is,

$$
N b \geq c _ { p } e ^ { \lambda T } , \qquad b \geq c _ { p } \eta e ^ { \lambda T } .
$$

The first condition bounds the total number of target samples. The second persists as � grows while the confinement error is small: later batches meet a smaller wedge, so at fixed step size a larger batch can satisfy this condition and a longer run cannot. Neither condition is suficient, since a sampled disagreement need not trigger escape.

An explicit joint scaling. Fix � and take a source-GF checkpoint at $T = c _ { T } \log ( 1 / \eta )$ with $c _ { T } > 1 / \lambda$ . For every � the disagreement term of (5) is at most

$$
C \frac { b } { \eta } e ^ { - \lambda T } = C b \eta ^ { \lambda c _ { T } - 1 } \longrightarrow 0 ,
$$

and on a horizon $H \ : = \ : \eta ^ { - c _ { H } }$ with fixed $c _ { H } ~ > ~ 0$ the confinement term also vanishes. The SGD recovery probability on each such horizon thus tends to zero uniformly on $K ,$ while GF recovers in ${ \cal O } ( \log ( 1 / \eta ) )$ time.

## 6. Proof Overview

The proof treats the phases in turn: source SGD contracts the projective separation, and the target analysis bounds the probability of ever sampling a disagreement. Appendix L collects the notation.

Why the proof uses projective separation. Angle alone does not measure distance from positively proportional full neurons: with $m _ { i } = r _ { i } / a _ { i } , d ^ { 2 } = ( m _ { 1 } -$ $\overline { { m _ { 2 } } } ) ^ { 2 } + 2 m _ { 1 } m _ { 2 } ( 1 - \cos { \delta } )$ , so $d = 2$ sin(�/2) under exact balance, while small balance errors can dominate a much smaller angle. After source GF, balance errors are $O ( e ^ { - 2 \lambda T } )$ and the angle is of order $e ^ { - \lambda T }$ , so $d \asymp \delta$ For source SGD we work with � directly, which allows arbitrary initialization in � without imposing balance.

Source contraction proportional to the separation. Appendix B treats the source phase. With positive heads, let $u _ { i } = w _ { i } / a _ { i }$ and $d = \left\| u _ { 1 } - u _ { 2 } \right\|$ . Positive proportionality, $u _ { 1 } ~ = ~ u _ { 2 }$ , is preserved by every sample update. Near the source minima, with ${ \mathcal { F } } _ { n }$ the �-field of the first � batches, the exact projective update gives (Appendix B.1)

$$
\mathbb { E } [ d _ { n + 1 } \mid \mathcal { F } _ { n } ] \leq ( 1 - \gamma \eta ) d _ { n } ,
$$

with no additive term of order �. Near the source direction, both the source label and network output are small on disagreeing inputs. An $L ^ { 1 }$ estimate keeps a factor � in the expected wedge correction, so it can be absorbed into the contraction. The radial target label ∥�∥ has no small factor on the same wedge, so its contribution can reverse the population transverse stability. A fixed population burn-in and Lemma 7 carry SGD from � into this neighborhood (Appendix B.3). The allocation, measured here by the log head ratio, then moves by at most $C \eta d _ { n }$ per step, so its total motion is summable; a long enough burn-in makes losing the allocation margin cost at most $\varepsilon / 2$ , the origin of the fixed allowance � (Appendices B.2 and B.3). This gives a good event $E _ { T }$ with $\mathbb { P } ( E _ { T } ^ { c } ) \le \varepsilon / 2 + C ( 1 + T / \eta ) e ^ { - c / \eta }$ and $\mathbb { E } [ d _ { T } \mathbf { 1 } _ { E _ { T } } \mid \theta _ { 0 } ] \le C e ^ { - \gamma T }$

Target phase. Section 7 proves the target obstruction, Proposition 6. An auxiliary process applies the gate of the shared hidden vector to both units, so both are multiplied by the same random matrix, and it coincides with SGD up to the first batch with a disagreement. Its target projective separation Δ (defined in Section 7) satisfies

$$
\mathbb { E } [ \Delta _ { n + 1 } ~ | ~ \mathcal { F } _ { n } ] \le ( 1 - \eta \lambda / 4 ) \Delta _ { n }
$$

in a tube around the clone family, for every batch size, because on the proportional family the mean commongate matrix satisfies $e ^ { \top } ( \mathbb { E } G ) e \leq - \dot { \lambda } \| e \| ^ { 2 }$ for � ⊥ ℎ whenever $B \leq 2 q .$ , a range that covers the amplitude transient from � to $B _ { t }$ . A batch contains a disagreement with probability at most $C b \Delta _ { n } ,$ so the summed hazard is at most $C b \Delta _ { 0 }$ min $\{ N , 4 / ( \lambda \eta ) \}$ . Confinement of amplitude and balance over blocks of fixed physical time adds $C ( N + 1 ) e ^ { - c / \eta }$ and sets the exponential horizon.

Composition. At the checkpoint $\Delta _ { 0 } \leq C d _ { T } ( \mathrm { A p } \cdot$ pendix D); conditioning on it and using the source bound gives Theorem 1(b). Fresh success follows from finite-time tracking of target GF with a strict risk margin.

## 7. Proof of the Target Obstruction

The target half of Theorem 1 rests on one estimate: from a checkpoint near the proportional family, target SGD reaches risk 0.4 only by sampling a gate disagreement, by a projective or allocation exit of probability $O ( \Delta _ { 0 } )$ , or by an exponentially rare excursion of amplitude and balance. This section proves it, except for the confinement of amplitude and balance, which is a standard block argument (Appendix C).

Coordinates. Recall $q = 2 / \pi , B _ { t } = 2 ( q - \lambda )$ and $R _ { \mathrm { c o l } } =$ $1 - q ^ { 2 } = 0 . 5 9 4 7 \dots$ . For the full neuron vectors $\mathfrak { z } _ { i } =$ $( a _ { i } , w _ { i } ) \in \mathbb { R } ^ { 3 }$ , define

$$
\begin{array} { c } { { h = \displaystyle \frac { z _ { 1 } + z _ { 2 } } { \sqrt 2 } = ( h _ { a } , w ) , \qquad d _ { z } = \displaystyle \frac { z _ { 1 } - z _ { 2 } } { \sqrt 2 } , \qquad } } \\ { { \rho = \displaystyle \frac { h ^ { \top } d _ { z } } { \left\| h \right\| ^ { 2 } } , \qquad e = d _ { z } - \rho h , } } \end{array}
$$

and $r = \| w \| , S = 1 + \rho ^ { 2 } , k = h _ { a } ^ { 2 } - r ^ { 2 } , B = S h _ { a } r$ (equal to $\omega _ { 1 } + \omega _ { 2 }$ on the proportional family). Here $e \perp h$ The proportional family is $e = 0 , | \rho | < 1 , h _ { a } , r > 0 _ { ; }$ , and its target stationary manifold is $k = 0 , B = B _ { t }$ . The allocation � records the relative scale of the two units, $\Delta = \| e \|$ is the target projective separation, and $d _ { z }$ is distinct from the source projective distance �.

Proposition 6 (Finite-horizon target obstruction). Fix $0 < \rho _ { 0 } < \rho _ { 1 } < \rho _ { 2 } < 1 _ { : }$ , and recall $B _ { s } = 1 - 2 \lambda < B _ { t }$ . There exist positive $\nu , \epsilon _ { 0 } , \eta _ { \mathrm { t } } , C ,$ � such that, whenever

$$
\begin{array} { r l } { h _ { a } , r > 0 , } & { { } \quad | \rho | \le \rho _ { 0 } , } \\ { \vert k \vert + \vert B - B _ { s } \vert \le \nu , } & { { } \quad \Vert e \Vert = \Delta _ { 0 } \le \epsilon _ { 0 } , } \end{array}
$$

target SGD with any batch size $b \geq 1 , 0 < \eta \leq \eta _ { \mathrm { t } }$ and integer $N \geq 1$ satisfies

$$
\begin{array} { r l r } {  { \mathbb { P } \Big ( \operatorname* { m i n } _ { 0 \leq n \leq N } R _ { t } ( \theta _ { n } ) \leq 0 . 4 \Big ) } } \\ & { } & { \leq C b \Delta _ { 0 } \operatorname* { m i n } \{ N , \eta ^ { - 1 } \} + C ( N + 1 ) e ^ { - c / \eta } . } \end{array}
$$

The constants are independent of $b , N , \Delta _ { 0 }$ and the common direction.

The proof has three steps. An auxiliary process applies one gate to both units (Section 7.1); its projective separation contracts exactly in conditional expectation (Section 7.2); and a stopping argument bounds the probability that actual SGD leaves the auxiliary process by the summed disagreement hazard (Section 7.4). Section 7.3 states the confinement bound proved in Appendix C.

## 7.1. An Exact Common-Gate Process

For $u = w / r$ , let $g _ { \theta } ( x ) = \mathbf { 1 } _ { w ^ { \intercal } x > 0 }$ and $U = a _ { 1 } w _ { 1 } + a _ { 2 } w _ { 2 }$ On the same sampled batch, define

$$
\begin{array} { c } { \displaystyle { f _ { \mathrm { a u x } } ( \boldsymbol { x } ) = g _ { \theta } ( \boldsymbol { x } ) \boldsymbol { U } ^ { \top } \boldsymbol { x } , } } \\ { \displaystyle { \boldsymbol { \nu } = \frac { 1 } { b } \sum _ { j } ( f _ { \mathrm { a u x } } ( X _ { j } ) - \left\| X _ { j } \right\| ) g _ { \theta } ( X _ { j } ) X _ { j } , } } \end{array}
$$

$$
\begin{array} { c c } { { G = - \lambda I - M ( \nu ) , } } & { { M ( \nu ) = \left( \begin{array} { c c } { { 0 } } & { { \nu ^ { \top } } } \\ { { \nu } } & { { 0 _ { 2 \times 2 } } } \end{array} \right) , } } \\ { { Q = I + \eta G , } } & { { z _ { i } ^ { + } = Q z _ { i } . } } \end{array}
$$

Both units are multiplied by the same random matrix �. If the two hidden gates agree on every batch example, their common gate also equals the gate of $w = ( w _ { 1 } + w _ { 2 } ) / \sqrt { 2 }$ and the auxiliary update is exactly the ordinary update (1). The processes therefore couple until the first gate disagreement or the tube exit defined below.

Rotational symmetry gives the exact conditional mean

$$
\mathbb { E } [ \nu \mid \theta ] = U / 2 - q u ,\tag{6}
$$

using $\mathbb { E } [ \mathbf { 1 } _ { u ^ { \top } X > 0 } X X ^ { \top } ] ~ = ~ I / 2$ and $\mathbb { E } [ \left. X \right. \mathbf { 1 } _ { u ^ { \intercal } X > 0 } X ] =$ $( 2 / \pi ) u$ . Thus the auxiliary mean field is smooth on $r > 0 _ { : }$ , although the SGD update is not.

## 7.2. Exact Finite-Step Projective Contraction

Since $h ^ { + } = Q h$ and $d _ { z } ^ { + } = Q d _ { z }$ , orthogonal projection gives exactly

$$
\begin{array} { c } { { e ^ { + } = P _ { Q h } Q e , \displaystyle \qquad \rho ^ { + } - \rho = \frac { h ^ { \top } Q ^ { 2 } e } { \left\| Q h \right\| ^ { 2 } } , } } \\ { { { } } } \\ { { P _ { \upsilon } = I - \displaystyle \frac { \upsilon \upsilon ^ { \top } } { \left\| \upsilon \right\| ^ { 2 } } , } } \end{array}\tag{7}
$$

where $Q = Q ^ { \top }$ was used. On the proportional family, $U = B u$ and $G _ { P } : = \mathbb { E } G = - \lambda I + ( q - B / 2 ) M ( u )$ . For $e \perp h ,$ $h _ { a } e _ { a } + r u ^ { \top } e _ { w } = 0$ , and therefore

$$
\begin{array} { r l r } {  { e ^ { \top } G _ { P } e = - \lambda \| e \| ^ { 2 } - 2 ( q - B / 2 ) \frac { h _ { a } } { r } e _ { a } ^ { 2 } } } \\ & { } & { \leq - \lambda \| e \| ^ { 2 } \qquad ( B \leq 2 q ) . \qquad } \end{array}\tag{8}
$$

The radial target thus enters the common-gate dynamics only through the shared direction $u ,$ and the transverse part of the mean matrix is a pure contraction. Of the proportional family, the exact identity

$$
U = S h _ { a } w + \rho { \left( h _ { a } e _ { w } + e _ { a } w \right) } + e _ { a } e _ { w }
$$

implies $\Vert \mathbb { E } G - G _ { P } \Vert ~ \leq ~ C _ { U }$ ∥�∥ on any fixed compact chart with positive lower bounds on $h _ { a } , r$ and $| \rho | \le \rho _ { 2 }$ Choose a tube with $B \le B _ { t } + \lambda / 2 < 2 q$ and projective radius � so that $C _ { U } \epsilon \leq \lambda / 2$

Bounded samples give a bound $\| G \| \leq L _ { G }$ for every possible batch in this tube; for example $\| \nu \| \leq 4 ( \| U \| +$ 1). For $\eta \leq \operatorname* { m i n } \{ ( 2 L _ { G } ) ^ { - 1 } , \lambda / ( 2 L _ { G } ^ { 2 } ) \} , ( 7 ) - ( 8 )$ imply

$$
\begin{array} { r l } & { \mathbb { E } [ \left\| e ^ { + } \right\| ^ { 2 } | \theta ] \leq ( 1 - \eta \lambda / 2 ) \left\| e \right\| ^ { 2 } , } \\ & { \mathbb { E } [ \left\| e ^ { + } \right\| | \theta ] \leq ( 1 - \eta \lambda / 4 ) \left\| e \right\| . } \end{array}\tag{9}
$$

Indeed, expand $\| Q e \| ^ { 2 }$ and use $\| e ^ { + } \| ~ \leq ~ \| Q e \|$ , then Jensen. Also $h ^ { \top } e = 0$ in (7) gives

$$
\left| \rho ^ { + } - \rho \right| \leq C \eta \left\| e \right\| ,\tag{10}
$$

because $Q ^ { 2 } = I + 2 \eta G + \eta ^ { 2 } G ^ { 2 }$ and $\left\| Q h \right\| \geq \left\| h \right\| / 2$ . Both estimates hold for every batch size, with constants independent of �.

## 7.3. Confinement of the Shared Parameters

The amplitude � and the balance � have stable mean dynamics on the proportional family, and the task switch moves � from $B _ { s }$ to $B _ { t }$ along the logistic path $\dot { B } = B \left( B _ { t } - B \right)$ , which stays below $2 q .$ . Appendix C defines a compact bridge tube for this transient, of fixed duration $\ell _ { 0 } ,$ , and a common tube $\{ \sqrt { W _ { t } } \le R _ { W } / 2 \}$ with $W _ { t } = k ^ { 2 } + ( B - B _ { t } ) ^ { 2 }$ around $k = 0 , B = B _ { t }$ , both inside the chart of Section $7 . 2 ,$ , and shows with Lemmas 7 and 8 that

ℙ(common-tube exit first by $N ) \leq C ( N + 1 ) e ^ { - c / \eta } ,$

(11)

where “first” means at or before the other stops defined below. This is the only source of the exponential horizon.

## 7.4. Stopping, Gate Hazards, and Actual Population Risk

Let $n _ { 0 } = \lceil \ell _ { 0 } / \eta \rceil$ and let $\mathcal { T } _ { n }$ be the compact bridge tube for $n \ < \ n _ { 0 }$ and the common tube $\{ \sqrt { W _ { t } } \ \le \ R _ { W } / 2 \}$ for $n \geq n _ { 0 }$ (Appendix C); Lemma 8 is restarted at $n _ { 0 }$ from the state reached at the end of the bridge. With ${ \tilde { \theta } } _ { n }$ the auxiliary process of Section 7.1 (frozen after the stop, as defined below), let $\zeta _ { t }$ be the first � with $\widetilde { \theta } _ { n } \notin \mathcal { T } _ { n }$ $| \rho | \geq \rho _ { 1 }$ , or $\| e \| ~ \ge ~ \epsilon$ . All constants are chosen on a slightly larger compact chart, so the update which exits is still defined and obeys the one-step inequalities. Killing after the stop and using (9) give

$$
\mathbb { E } [ \| \tilde { e } _ { n } \| \mathbf { 1 } _ { n < \zeta _ { t } } ] \le \Delta _ { 0 } ( 1 - \eta \lambda / 4 ) ^ { n } .\tag{12}
$$

The stopped squared norm $\left\| e _ { n \wedge \zeta _ { t } } \right\| ^ { 2 }$ is a nonnegative supermartingale: the inequality holds also for the update which first exits. Ville’s maximal inequality for this supermartingale, and Markov’s inequality with (10), yield

$$
\begin{array} { r } { \mathbb { P } ( \mathrm { p r o j e c t i v e \ e x i t \ f i r s t } ) \le \Delta _ { 0 } ^ { 2 } / \epsilon ^ { 2 } , } \end{array}\tag{13}
$$

$$
\mathbb { P } ( \mathrm { a l l o c a t i o n } \ \mathrm { e x i t } \ \mathrm { f i r s t } ) \le \frac { C \Delta _ { 0 } } { \lambda ( \rho _ { 1 } - \rho _ { 0 } ) } ,\tag{14}
$$

with “first” as in (11). For the latter, sum expected absolute increments using (12) and apply Markov’s inequality.

In the compact tube the hidden vectors are perturbations of the nonzero, positively proportional vectors $( 1 \pm \rho ) w / \sqrt { 2 }$ . Their angle is at most $C _ { \mathrm { a n g } } \| e \|$ . A rotationally uniform sample disagrees with their two gates with exact probability equal to this angle divided by �.

Event inclusion. Retain the exit state at $\zeta _ { t }$ and freeze the auxiliary process thereafter. This defines ${ \tilde { \theta } } _ { n }$ for all � without extending the local chart; the stopped estimates are unchanged. Let $\mathcal { G } _ { n }$ be the event that some sample of batch � falls where the two hidden gates of ${ \tilde { \theta } } _ { n }$ disagree, and let $n _ { \mathrm { d i s } }$ be the first � with ${ \mathcal { G } } _ { n }$

Actual SGD equals ${ \tilde { \theta } } _ { n }$ for every $n \leq n _ { \mathrm { d i s } } \land \zeta _ { t } . \ \mathrm { I f } \ n _ { \mathrm { d i s } } \geq N$ and $\zeta _ { t } > N _ { \mathrm { \ell } }$ , then $\theta _ { n } = \tilde { \theta } _ { r }$ lies in the tube for all $n \leq N ,$ and its risk exceeds 0.4 by the risk-gap argument below. Hence

$$
\begin{array} { c } { { \left\{ \underset { n \leq N } { \operatorname* { m i n } } R _ { t } ( \theta _ { n } ) \leq 0 . 4 \right\} } } \\ { { \subset \{ n _ { \mathrm { d i s } } < N \land \zeta _ { t } \} \cup \{ \zeta _ { t } \leq N \} . } } \end{array}
$$

Because $\{ n < \zeta _ { t } \} \in \mathcal { F } _ { n }$ and the batch at step � is independent of $\mathcal { F } _ { n } , \mathbb { P } ( \boldsymbol { \mathcal { G } _ { n } } \mid \mathcal { F } _ { n } ) \leq b C _ { \mathrm { a n g } } \left\| \tilde { e } _ { n } \right\| / \pi$ on $\{ n < \zeta _ { t } \}$ With (12),

$$
\begin{array} { r l } {  { \mathbb { P } ( n _ { \mathrm { d i s } } < N \wedge \zeta _ { t } ) \le \displaystyle \sum _ { n < N } \mathbb { E } \big [ \mathbb { P } ( \mathscr { G } _ { n } \mid \mathcal { F } _ { n } ) \mathbf { 1 } _ { n < \zeta _ { t } } \big ] } \quad } & { } \\ & { \leq \frac { b C _ { \mathrm { a n g } } \Delta _ { 0 } } { \pi } \displaystyle \sum _ { n < N } ( 1 - \eta \lambda / 4 ) ^ { n } } \\ & { \le \frac { b C _ { \mathrm { a n g } } \Delta _ { 0 } } { \pi } \operatorname* { m i n } \Bigl \{ N , \frac { 4 } { \lambda \eta } \Bigr \} . } \end{array}\tag{15}
$$

The conditioning is on the past only: the hazard of step � is computed for a fresh batch that is independent of ${ \mathcal { F } } _ { n ; }$ , so no sample is ever conditioned on having avoided the wedge. The geometric sum is why the disagreement term saturates in �: contraction caps the total expected number of disagreeing samples at order $b \Delta _ { 0 } / ( \lambda \eta )$ , however long the run.

Risk gap. Replace � by �ℎ to obtain a proportional parameter $\theta _ { P }$ . The orthogonal change of coordinates gives $\lVert \theta - \theta _ { P } \rVert = \lVert e \rVert$ . Its predictor is a single ReLU, so its risk is at least $R _ { \mathrm { c o l } } > 0 . 4$ by (37). Risk is Lipschitz on the compact tube. Reducing � so that $L _ { R } \epsilon < R _ { \mathrm { c o l } } - 0 .$ 4 ensures that every state in the tube has risk above $0 . 4$ . The event $\{ \zeta _ { t } \leq N \}$ is bounded by (11)–(14), and $\{ n _ { \mathrm { d i s } } < N \land \zeta _ { t } \}$ by (15). The terms $C \Delta _ { 0 }$ and $C \Delta _ { 0 } ^ { 2 }$ are absorbed into $C b \Delta _ { 0 }$ min $\{ N , \eta ^ { - 1 } \}$ since $N , b \geq 1 , \eta \leq 1$ $\Delta _ { 0 } \leq 1$ . This proves Proposition 6.

## 8. Simulations

Simulations of the two-unit model check four consequences of the analysis: the gap appears from identical checkpoints; the recovery probability is approximately a function of the budget Γ that controls the bound; the disagreement term saturates in the horizon, and so does observed recovery; and saturation depends on the contraction supplied by weight decay. All runs use fresh minibatches from the disk, double precision, the analytic population risk and, except for the weight-decay controls, $\lambda = 0 . 0 5$ . GD uses the same � (Appendix J checks step halving). Here $\Gamma = b \delta _ { 0 } / \eta$ with �<sub>0</sub> the angle at the task switch $( \Gamma _ { T }$ after source time �). Appendix J gives details. The code is provided as ancillary files with this preprint.

Identical checkpoints. We draw 64 Gaussian initializations conditioned on ${ \mathcal { D } } _ { 0 } \supset K ,$ train each on the source with both SGD and GD, and pass each checkpoint to both target optimizers. Fresh target SGD $\left( T \ = \ 0 \right)$ recovers from all 64 initializations. Target GD recovers from every checkpoint, with hitting time growing as $\log ( 1 / \delta _ { T } ) / A$ (Figure 2). Target SGD recovers from every checkpoint with $\Gamma _ { T } \gtrsim 1$ , recovers late or only sometimes for $1 0 ^ { - 2 } \lesssim \Gamma _ { T } \lesssim 1$ , and has no observed recovery below $1 0 ^ { - 2 }$ . By $T = 4 0 0$ , 57 of the 64 source-SGD checkpoints have $\Gamma _ { T } < 0 . 0 4 ;$ none of them samples a disagreement, and no run recovers after $t = 1 0 0$ although GD recovers by $t \approx 4 0$ . All seven recoveries at $T \ = \ 4 0 0$ have $\Gamma _ { T } ~ > ~ 1$ and start from near-opposite initializations (initial angle above $2 . 5 ) _ { \mathrm { ; } }$ ; at each, one unit has amplitude below $1 0 ^ { - 8 }$ , so its direction barely moves. Source-GD checkpoints, the protocol of Corollary 5, give the same counts up to two runs at every � (Table 1).

![](images/6ca8cce562d68a6a70446e4dec6d6d271867e6de41a39dabe41eb8f58ce3abf6.jpg)  
Figure 2 | Identical checkpoints. Target hitting times from shared source-SGD checkpoints $( T \in \{ 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 \} .$ $\eta = 1 0 ^ { - 3 } , b = 1 6 )$ . GD (blue) follows $A ^ { - 1 } \log ( 1 / \delta _ { T } )$ plus the median ofset (dashed); SGD (orange) rarely recovers by $H = 2 0 0$ (top strip) once $\Gamma _ { T } < \pi \lambda$ (shaded).

Table 1 | Identical checkpoints. Number of the 64 saved initializations from which each target optimizer reaches risk 0.4 by $H = 2 0 0 \ ( \eta = 1 0 ^ { - 3 } , b = 1 6 )$ . The source-GD columns are the protocol of Corollary 5.
<table><tr><td rowspan="2">T</td><td colspan="2">source SGD</td><td colspan="2">source GD</td></tr><tr><td>target SGD</td><td>target GD</td><td>target SGD</td><td>target GD</td></tr><tr><td>0</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>100</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>200</td><td>59</td><td>64</td><td>61</td><td>64</td></tr><tr><td>300</td><td>20</td><td>64</td><td>19</td><td>64</td></tr><tr><td>400</td><td>7</td><td>64</td><td>7</td><td>64</td></tr></table>

Recovery across disagreement budgets. Beyond $1 / \eta$ updates, the disagreement term of Proposition 6 is a multiple of Γ at balanced checkpoints. Figure $3 ( \mathrm { a } )$ shows that the recovery probability is approximately a function of Γ. It pools the 512 checkpoints with $T > 0$ of the identical-checkpoint experiment (both source optimizers) with constructed balanced checkpoints at amplitude 0.9, in which $\delta _ { 0 } \in \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ $b \in \{ 1 , 1 6 , 6 4 \}$ and $\eta \in \{ 0 . 0 1 , 0 . 0 0 2 \}$ vary independently (24 cells, 64 streams each, $H = 2 0 0 )$ . As a reference we use a Poisson count: if the angle decays as $\delta _ { 0 } e ^ { - \lambda t }$ before the first disagreement, the expected number of disagreeing samples is $\Gamma / ( \pi \lambda )$ , and if each triggered recovery, recovery would have probability $1 - e ^ { - \Gamma / ( \pi \lambda ) }$ . The reference captures the location and overall shape of the observed transition, and it lies inside or above the 95% interval of every cell. Every observed recovery follows a sampled disagreement, and most runs that sample one recover (Appendix J). At $\delta _ { 0 } = 1 0 ^ { - 4 }$ and $\eta = 0 . 0 1$ , raising � from 1 to 64 raises recovery from 8% to 94%.

Saturation over long horizons. In five constructed settings (Γ from 0.016 to 0.64), no recovery occurs after � ≈ 112 through $t = 4 5 0$ , about 20 to 32 times their GD hitting times. The frozen-angle reference, which keeps the initial angle and counts recovery at the first disagreement, exceeds 0.63 by $t ~ = ~ 2 0 0$ in every setting (Figure 3b). Recovery thus saturates in the horizon, mirroring the factor min $\{ N , \eta ^ { - 1 } \}$ in the bound.

Weight decay. Controls from constructed checkpoints change only the target phase (Appendix K, Figure 5). With � $\in \ \{ 0 . 0 1 , 0 . 0 5 \}$ , no first disagreement is observed after $t \approx 2 4 0$ , and recovery plateaus through $H = 1 0 0 0$ at or below the Poisson reference. Without target weight decay the separation contracts only weakly and recovery keeps accumulating, so saturation comes from the contraction supplied by weight decay.

## 9. Conclusion and Future Work

In this model, population instability mispredicts finitebatch adaptation after pretraining because the inputs that separate near-clones become rarer as the commongate dynamics contract their separation. Bounding their cumulative sampling probability gives hittingtime lower bounds and necessary sample and batch sizes. Warm SGD recovers as $\eta  0$ at each fixed � (Corollary 3) and fails once $T \gtrsim \log ( b / \eta )$ (Theorem 1).

Scope and extensions. The separation is proved for two units, but several ingredients extend beyond width two: identical units stay identical under GF and SGD at any width (Joudaki et al., 2026; Şimşek et al., 2021), and for bias-free units and rotation-invariant inputs in any dimension, two gates disagree with probability $\delta / \pi$ In a wider network other units may fit the target, so a clone pair need not block learning; the open question is when the input mass on which units respond diferently controls the risk. Explicit values of the burn-in � and the step-size cutof $\eta _ { 0 }$ are left open; the simulations test the mechanism, not these constants. For smooth activations the wedge becomes a transition layer whose sampling is open.

![](images/5b30926078666bbd9e8a75f8167878ae55d78e928f732396e6349d60f685737f.jpg)

![](images/1703bc1676dfa278681d950e77c989a6f408e991e2339dc0b7c8ac8a91e33ecc.jpg)  
Figure 3 | Recovery is organized by the disagreement budget. (a) Recovery probability is approximately a function of $\Gamma = b \delta _ { 0 } / \eta$ and lies at or below the Poisson reference $1 - e ^ { - \Gamma / ( \pi \lambda ) }$ (dashed) up to sampling error. Orange: constructed checkpoints with 95% Wilson intervals (filled: $\eta = 0 . 0 1 )$ ; diamonds: checkpoints from Gaussian initializations. (b) Recovery saturates in the horizon (solid), far below the frozen-angle reference (dotted); three of the five long-horizon settings are shown.

In this model a longer run at fixed batch size and step size cannot replace a larger batch over horizons of order $e ^ { c / \eta }$ updates (Section 5), and perturbations that act on each unit separately restore disagreement mass (Appendix K), as in shrink-and-perturb (Ash and Adams, 2020). L2 decay toward zero, which aids plasticity elsewhere (Dohare et al., 2024), here contracts the separation of near-clones, whereas regularization toward initialization (Kumar et al., 2025) would pull them apart.

Use of AI tools. Generative AI tools were used as assistants, under the authors’ direction, for literature search and checking of references; additional re-checking of derivations and editing of the written presentation of statements and proofs; feedback on the design of the simulations, their implementation, and plotting; and editing and restructuring of the text. The authors led and reviewed all such output and take full responsibility for the content of the paper, including text, claims and code produced with the aid of these tools.

## References

Jordan T. Ash and Ryan P. Adams. On warm-starting neural network training. In NeurIPS, 2020.

Gérard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. Journal of Machine Learning Research, 22(106):1–51, 2021.

Gérard Ben Arous, Reza Gheissari, and Aukosh Jagannath. High-dimensional limit theorems for SGD: Efective dynamics and critical scaling. In NeurIPS, 2022.

Michel Benaïm. Dynamics of stochastic approximation algorithms. In Séminaire de Probabilités XXXIII, Lecture Notes in Mathematics 1709, pages 1–68. Springer, 1999.

Pascal Bianchi, Walid Hachem, and Sholom Schechtman. Stochastic subgradient descent escapes active strict saddles on weakly convex functions. Mathematics of Operations Research, 49(3):1761–1790, 2024.

Odile Brandière and Marie Duflo. Les algorithmes stochastiques contournent-ils les pièges? Annales de l’IHP Probabilités et Statistiques, 32(3):395–427, 1996.

Feng Chen, Daniel Kunin, Atsushi Yamamura, and Surya Ganguli. Stochastic collapse: How gradient noise attracts SGD dynamics towards simpler subnetworks. In NeurIPS, 2023.

Damek Davis, Dmitriy Drusvyatskiy, and Liwei Jiang. Active manifolds, stratifications, and convergence to local minima in nonsmooth optimization. Foundations of Computational Mathematics, 26(2):779–861, 2026.

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, and Richard S. Sutton. Loss of plasticity in deep continual learning. Nature, 632:768–774, 2024.

Simon S. Du, Wei Hu, and Jason D. Lee. Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced. In NeurIPS, 2018.

Mark I. Freidlin and Alexander D. Wentzell. Random Perturbations of Dynamical Systems. Springer, 3rd edition, 2012.

Kenji Fukumizu and Shun-ichi Amari. Local minima and plateaus in hierarchical structures of multilayer perceptrons. Neural Networks, 13(3):317–327, 2000.

Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points—online stochastic gradient for tensor decomposition. In COLT, 2015.

Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, and Michael I. Jordan. How to escape saddle points eficiently. In ICML, 2017.

Chi Jin, Lydia T. Liu, Rong Ge, and Michael I. Jordan. On the local minima of the empirical risk. In NeurIPS, 2018.

Amir Joudaki, Giulia Lanzillotta, Mohammad Samragh, Iman Mirzadeh, Keivan Alizadeh-Vahid, Thomas Hofmann, Mehrdad Farajtabar, and Fartash Faghri. Barriers for learning in an evolving world: Mathematical understanding of loss of plasticity. In ICLR, 2026.

Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, and Sergey Levine. Implicit under-parameterization inhibits dataeficient deep reinforcement learning. In ICLR, 2021.

Saurabh Kumar, Henrik Marklund, and Benjamin Van Roy. Maintaining plasticity in continual learning via regenerative regularization. In CoLLAs 2024, PMLR 274:410–430, 2025.

Harold J. Kushner and G. George Yin. Stochastic Approximation and Recursive Algorithms and Applications. Springer, 2nd edition, 2003.

Alex Lewandowski, Haruto Tanaka, Dale Schuurmans, and Marlos C. Machado. Directions of curvature as an explanation for loss of plasticity. arXiv:2312.00246, 2023.

Binghua Li, Mengzhe Li, Denny Wu, and Tianhao Wang. Gradient descent on two ReLU neurons: Global landscape and bifurcation dynamics. In ICML Workshop on High Dimensional Learning Dynamics: The Science of Scaling, 2026.

Qianxiao Li, Cheng Tai, and Weinan E. Stochastic modified equations and adaptive stochastic gradient algorithms. In ICML, 2017.

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, and Will Dabney. Understanding plasticity in neural networks. In ICML, 2023.

Clare Lyle, Zeyu Zheng, Khimya Khetarpal, Hado van Hasselt, Razvan Pascanu, James Martens, and Will Dabney. Disentangling the causes of plasticity loss in neural networks. In CoLLAs 2024, PMLR 274:750–783, 2025.

Hartmut Maennel, Olivier Bousquet, and Sylvain Gelly. Gradient descent quantizes ReLU network features. arXiv:1803.08367, 2018.

Panayotis Mertikopoulos, Nadav Hallak, Ali Kavis, and Volkan Cevher. On the almost sure convergence of stochastic gradient descent in non-convex problems. In NeurIPS, 2020.

Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, Pierre-Luc Bacon, and Aaron Courville. The primacy bias in deep reinforcement learning. In ICML, 2022.

Robin Pemantle. Nonconvergence to unstable points in urn models and stochastic approximations. The Annals ofProbability, 18(2):698–712, 1990.

David Saad and Sara A. Solla. On-line learning in soft committee machines. Physical Review E, 52(4):4225–4243, 1995.

Berfin Şimşek, François Ged, Arthur Jacot, Francesco Spadaro, Clément Hongler, Wulfram Gerstner, and Johanni Brea. Geometry of the loss landscape in overparameterized neural networks: Symmetries and invariances. In ICML, 2021.

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, and Utku Evci. The dormant neuron phenomenon in deep reinforcement learning. In ICML, 2023.

Yuandong Tian. An analytical formula of population gradient for two-layered ReLU network and its applications in convergence and critical point analysis. In ICML, 2017.

Jiuqi Wang, Jayanth Srinivasa, Claire Chen, Shuze Daniel Liu, Ali Payani, and Shangtong Zhang. Predicting plasticity in deep continual learning: A theoretical perspective. arXiv:2605.09044, 2026.

Haikun Wei, Jun Zhang, Florent Cousseau, Tomoko Ozeki, and Shun-ichi Amari. Dynamics of learning near singularities in layered networks. Neural Computation, 20(3):813– 843, 2008.

Weihang Xu and Simon S. Du. Over-parameterization exponentially slows down gradient descent for learning a single neuron. In COLT, PMLR 195, pages 1155–1198, 2023.

Gilad Yehudai and Ohad Shamir. Learning a single neuron with gradient methods. In COLT, PMLR 125, 2020.

Kai Zhong, Zhao Song, Prateek Jain, Peter L. Bartlett, and Inderjit S. Dhillon. Recovery guarantees for one-hiddenlayer neural networks. In ICML, 2017.

Liu Ziyin. Symmetry induces structure and constraint of learning. In ICML, 2024.

Liu Ziyin, Botao Li, James B. Simon, and Masahito Ueda. SGD can converge to local maxima. In ICLR, 2022.

Liu Ziyin, Botao Li, Tomer Galanti, and Masahito Ueda. Type-II saddles and probabilistic stability of stochastic gradient descent. arXiv:2303.13093, 2023.

Liu Ziyin, Mingze Wang, Hongchao Li, and Lei Wu. Parameter symmetry and noise equilibrium of stochastic gradient descent. In NeurIPS, 2024.

# Appendix

## A. Finite-Block Tracking and Confinement

We use one concentration estimate for source entry, source confinement, the target bridge, target confinement, and fresh success. Throughout the sample updates, take $\sigma ^ { \prime } ( 0 ) = 0 ;$ ; zero-gate inputs have probability zero in the nonzero-hidden-vector charts used below, so any fixed convention gives the same stochastic laws. Every SGD update (1) can be written as $\theta _ { n + 1 } = \theta _ { n } + \eta F ( \theta _ { n } ) + \eta \xi _ { n + 1 }$ , where ${ F } = - \nabla { F } _ { \nu }$ is the population field and $\xi _ { n + 1 }$ is the centered batch-gradient error. Individual sample gradients need not be Lipschitz across gates; only � must be.

Lemma 7 (Tracking a smooth conditional mean). Let $\mathcal { K } \subset \mathbb { R } ^ { 6 }$ be compact, $0 < a \le 1$ , and let ${ \mathcal { K } } ^ { a }$ be its closed �-neighborhood. Suppose that on ${ \mathcal { K } } ^ { a }$ the field � is $L _ { F } – L i p s c h i t z$ and bounded by $M _ { F } ,$ , and that whenever $\theta _ { n } \in \mathcal K ^ { a }$

$$
\theta _ { n + 1 } = \theta _ { n } + \eta F ( \theta _ { n } ) + \eta \xi _ { n + 1 } , \qquad \mathbb { E } [ \xi _ { n + 1 } ~ | ~ \mathcal { F } _ { n } ] = 0 , \qquad \| \xi _ { n + 1 } \| \le M .
$$

Let $\ell > 0$ and let $S _ { 0 }$ be a set of initial states whose �-flows $\phi _ { t } ( \theta _ { 0 } )$ remain in K for $t \in [ 0 , \ell + 1 ]$ . Put $N _ { \ell } = \lceil \ell / \eta \rceil$ There is $\bar { \eta } \in ( 0 , 1 ]$ , depending only on $( a , \ell , L _ { F } , M _ { F } , M ) _ { \scriptscriptstyle 3 }$ , such that for all $\eta \leq \bar { \eta }$ and $\theta _ { 0 } \in S _ { 0 } ,$

$$
\mathbb { P } \bigg ( \operatorname* { m a x } _ { n \le N _ { \ell } } \Big \| \theta _ { n } - \phi _ { n \eta } ( \theta _ { 0 } ) \Big \| > a \bigg ) \le 1 2 \exp \bigg [ { - \frac { a ^ { 2 } e ^ { - 2 L _ { F } ( \ell + 1 ) } } { 4 8 M ^ { 2 } ( \ell + 1 ) \eta } } \bigg ] .
$$

In particular, on the complementary event the recursion stays in ${ \mathcal { K } } ^ { a }$ up to step $N _ { \ell } .$

Proof. Let � be the first � with $\left\| \theta _ { n } - \phi _ { n \eta } ( \theta _ { 0 } ) \right\| > a ;$ it is a stopping time, and $\theta _ { j } \in \mathcal K ^ { a }$ for $j < \tau$ . For $m \leq \tau \land N _ { \ell } ,$ , put $\epsilon _ { m } = \theta _ { m } - \phi _ { m \eta } ( \theta _ { 0 } )$ . Writing the flow in integral form,

$$
\epsilon _ { m } = \eta \sum _ { j < m } \bigl [ F ( \theta _ { j } ) - F ( \phi _ { j \eta } ) \bigr ] + \eta \sum _ { j < m } \xi _ { j + 1 } { \bf 1 } _ { j < \tau } - \sum _ { j < m } \int _ { j \eta } ^ { ( j + 1 ) \eta } \bigl [ F ( \phi _ { s } ) - F ( \phi _ { j \eta } ) \bigr ] d s .
$$

The last sum is at most $C _ { E } \eta$ with $C _ { E } = ( \ell + 1 ) L _ { F } M _ { F }$ . Let $\begin{array} { r } { Z = \operatorname* { m a x } _ { m \leq N _ { \ell } } \left\| \eta \sum _ { j < m } \xi _ { j + 1 } { \bf 1 } _ { j < \tau } \right\| } \end{array}$ . Discrete Gronwall gives $\lVert \epsilon _ { m } \rVert \leq ( Z + C _ { E } \eta ) e ^ { L _ { F } ( \ell + 1 ) }$ for all � $\leq \tau \wedge N _ { \ell } ;$ the bound at $m = \tau$ uses only states $\theta _ { j } , j < \tau _ { : }$ , which lie in ${ \mathcal { K } } ^ { a }$ . The stopped increments $\eta \xi _ { j + 1 } \mathbf { 1 } _ { j < \tau }$ form a bounded martingale diference sequence. Coordinatewise maximal Azuma and a union bound over six coordinates give $\mathbb { P } ( Z \geq z ) \leq 1 2 \exp [ - z ^ { 2 } / ( 1 2 M ^ { 2 } ( \ell + 1 ) \eta ) ]$ . Take $z = a e ^ { - L _ { F } ( \ell + 1 ) } / 2$ and �¯ with $C _ { E } \bar { \eta } e ^ { L _ { F } ( \ell + 1 ) } \le a / 2$ . On $\{ Z < z \}$ we get $\| \epsilon _ { m } \| < a$ for every � $\leq \tau \wedge N _ { \ell } ,$ , which forces $\tau > N _ { \ell }$ □

The confinement arguments iterate Lemma 7 over blocks of fixed physical length.

Lemma 8 (Block confinement). Let K be compact, let Ψ be defined and � -Lipschitz on ${ \mathcal { K } } ^ { a }$ , let $\varrho > 0 ;$ , and let I be a closed constraint set (in the applications, an allocation interval or a projective constraint). Suppose a block duration $\ell _ { \mathrm { b l k } }$ and tolerance $a \in ( 0 , 1 ]$ with $a L _ { \Psi } \leq \varrho / 4$ satisfy: for every $\boldsymbol { x } \in \mathcal { K } ^ { a }$ with $\Psi ( x ) \leq \varrho$ and $x \in { \mathcal { I } }$ , theflow $\phi _ { t } ( x )$ stays in K with $\Psi \le 3 \varrho / 2 f o r t \in [ 0 , \ell _ { \mathrm { b l k } } + 1 ]$ and has $\Psi \le \varrho / 2 f o r t \in [ \ell _ { \mathrm { b l k } } , \ell _ { \mathrm { b l k } } + 1 ]$ . Let the hypotheses of Lemma 7 hold on $\mathcal { K } ,$ and let $\zeta ^ { \prime }$ be the first time the recursion leaves I. $I f \theta _ { 0 } \in { \mathcal { K } } ^ { a } \cap J$ and $\Psi ( \theta _ { 0 } ) \leq \varrho ,$ then for $\eta \leq \bar { \eta }$ and every integer $N \geq 1 _ { \ast }$

$$
\mathbb { P } \big ( \Psi ( \theta _ { n } ) > 2 \varrho f o r s o m e n \le N \wedge \zeta ^ { \prime } \big ) \le \Big ( 1 + \frac { N \eta } { \ell _ { \mathrm { b l k } } } \Big ) 1 2 \exp \Big [ - \frac { a ^ { 2 } e ^ { - 2 L _ { F } ( \ell _ { \mathrm { b l k } } + 1 ) } } { 4 8 M ^ { 2 } ( \ell _ { \mathrm { b l k } } + 1 ) \eta } \Big ] .
$$

Proof. Split time into blocks of $n _ { \mathrm { b l k } } = \lceil \ell _ { \mathrm { b l k } } / \eta \rceil$ updates. Condition on $\mathcal { F } _ { k n _ { \mathrm { b l k } } }$ and the event that all preceding blocks satisfy their tracking bounds and $\zeta ^ { \prime } > k n _ { \mathrm { b l k } }$ . The induction gives $\Psi ( \theta _ { k n _ { \mathrm { b l k } } } ) \leq \varrho$ and $\theta _ { k n _ { \mathrm { b l k } } } \in { \mathcal { K } } ^ { a } \cap \bar { \mathcal { I } }$ . Fresh samples make the block a recursion of the same form, and Lemma 7 applies uniformly over such starting points. On its good event, the block stays within � of the flow. It therefore keeps $\Psi \leq 3 \varrho / 2 + a L \Psi < 2 \varrho$ and ends with $\Psi \leq \varrho / 2 + a L _ { \Psi } \leq \varrho ,$ so the next block starts in the same condition unless $\zeta ^ { \prime }$ has occurred. A union bound over at most $1 + N / n _ { \mathrm { b l k } } \le 1 + N \eta / \ell _ { \mathrm { b l k } }$ blocks proves the claim. □

## B. Source SGD: Projective Contraction

The source minimum family has $w _ { i } = a _ { i } e _ { 1 } , a _ { i } > 0$ , and $a _ { 1 } ^ { 2 } + a _ { 2 } ^ { 2 } = B _ { s } : = 1 - 2 \lambda$ . On a compact positive-head neighborhood introduce

$$
u _ { i } = w _ { i } / a _ { i } , \quad d = \| u _ { 1 } - u _ { 2 } \| , \quad \ell _ { a } = \log ( a _ { 1 } / a _ { 2 } ) .
$$

The proportional-neuron family is exactly $u _ { 1 } = u _ { 2 }$ . Unlike balance, this family is invariant under every Euler sample update.

## B.1. A Local One-Step Estimate without an Additive Floor

For a batch, write $R _ { j } = f _ { \theta } ( X _ { j } ) - \sigma ( X _ { j 1 } )$ and $\varsigma = 1 - \eta \lambda$ . Positive homogeneity gives the exact map

$$
a _ { i } ^ { + } = a _ { i } \left[ \varsigma - \frac { \eta } { b } \sum _ { j } R _ { j } \sigma ( u _ { i } ^ { \top } X _ { j } ) \right] , \qquad u _ { i } ^ { + } = \frac { \varsigma u _ { i } - ( \eta / b ) \sum _ { j } R _ { j } \sigma ^ { \prime } ( u _ { i } ^ { \top } X _ { j } ) X _ { j } } { \varsigma - ( \eta / b ) \sum _ { j } R _ { j } \sigma ( u _ { i } ^ { \top } X _ { j } ) } .\tag{16}
$$

All denominators exceed $1 / 2$ on the chosen compact set for small $\eta .$ Anchor the gates at unit $^ { 2 , }$ set $\nu \ =$ $\begin{array} { r } { b ^ { - 1 } \sum _ { j } R _ { j } \sigma ^ { \prime } ( u _ { 2 } ^ { \top } X _ { j } ) X _ { j } } \end{array}$ , and define

$$
\Phi _ { \nu } ( u ) = \frac { \varsigma u - \eta \nu } { \varsigma - \eta \nu ^ { \top } u } .
$$

This equals both actual projective maps whenever all batch gates agree. When diferentiating $\Phi _ { \nu }$ , hold the shared, actual residuals $R _ { j }$ fixed. For bounded $u , \nu _ { \mathrm { { ; } } }$

$$
D \Phi _ { \nu } ( u ) = I + \eta \mathcal { H } ( \nu , u ) + O ( \eta ^ { 2 } ) , \qquad \mathcal { H } ( \nu , u ) = ( \nu ^ { \top } u ) I + u \nu ^ { \top } .
$$

Writing $e _ { s } = u _ { 1 } - u _ { 2 }$ , the matched separation satisfies

$$
\boldsymbol { e } _ { s , \mathrm { m a t } } ^ { + } = [ I + \eta \mathcal { H } ( \nu , u _ { 2 } ) + \mathcal { E } ] \boldsymbol { e } _ { s } , \qquad \| \boldsymbol { \mathcal { E } } \| \le C \eta d + C \eta ^ { 2 } .\tag{17}
$$

At an exact source minimum, $u _ { i } = e _ { 1 }$ and $R _ { j } = - 2 \lambda \sigma ( X _ { j 1 } )$ , so

$$
\begin{array} { r } { \mathbb { E } \nu = - \lambda e _ { 1 } , \qquad \mathbb { E } \mathcal { H } ( \nu , e _ { 1 } ) = - \lambda \big ( I + e _ { 1 } e _ { 1 } ^ { \top } \big ) . } \end{array}
$$

The averaged matrix is continuous in the parameters, uniformly over compact allocation intervals. Bounded inputs bound H and the Taylor errors. In a suficiently small fixed source neighborhood of radius $r _ { 0 ; }$ , (17) therefore yields

$$
\mathbb { E } [ \left. e _ { s , \mathrm { m a t } } ^ { + } \right. ^ { 2 } | \ \theta ] \leq ( 1 - c \lambda \eta ) d ^ { 2 } , \qquad \mathbb { E } [ \left. e _ { s , \mathrm { m a t } } ^ { + } \right. | \ \theta ] \leq ( 1 - c ^ { \prime } \lambda \eta ) d .
$$

For one sample whose unit gates disagree, the geometric wedge bounds are

$$
\begin{array} { r } { \mathbb { P } \left( \pmb { \mathscr { G } } \mid \theta \right) \leq C d , \qquad \left| u _ { i } ^ { \top } \pmb { X } \right| \leq C d \left\| \pmb { X } \right\| , \qquad \left| \boldsymbol { X } _ { 1 } \right| \leq C ( r _ { 0 } + d ) \left\| \boldsymbol { X } \right\| . } \end{array}
$$

Thus the source residual on this wedge is at most $C ( r _ { 0 } + d ) \left\| X \right\|$ . Comparing both numerator and denominator in (16) with the matched map gives

$$
\left\| e _ { s } ^ { + } - e _ { s , \mathrm { m a t } } ^ { + } \right\| \leq \frac { C \eta ( r _ { 0 } + d ) } { b } \sum _ { j = 1 } ^ { b } \mathbf { 1 } _ { \mathcal { G } _ { j } } .
$$

Its conditional expectation is at most $C \eta ( r _ { 0 } + d ) d _ { ; }$ , independently of $b .$ The symmetric part of $\mathbb { E } \mathcal { H } ( \nu , e _ { 1 } )$ is at most $- \lambda I ,$ and the averaged matrix changes by $O ( r _ { 0 } )$ over the neighborhood, so the $L ^ { 1 }$ bound above holds with $c ^ { \prime } \geq 1 - C ( r _ { 0 } + \eta ) / \lambda$ . Since $d \leq C r _ { 0 }$ in this neighborhood, the disagreement correction is at most $C \eta r _ { 0 } d ,$ so one may take $\gamma = \lambda - C ( r _ { 0 } + \eta _ { 0 } )$ ; reducing $r _ { 0 }$ and then $\eta _ { 0 }$ gives $\gamma \geq \lambda / 2$ with

$$
\mathbb { E } [ d _ { n + 1 } \mid \mathcal { F } _ { n } ] \leq ( 1 - \gamma \eta ) d _ { n }\tag{18}
$$

throughout the source neighborhood. The disagreement correction is estimated in $L ^ { 1 } \colon$ its squared contribution can be $O ( \eta ^ { 2 } r _ { 0 } ^ { 2 } d )$ and cannot in general be absorbed into $d ^ { 2 }$

For every batch, including samples with disagreeing gates, the head ratio obeys

$$
\left| \ell _ { a } ^ { + } - \ell _ { a } \right| = \left| \log \frac { \varsigma - ( \eta / b ) \sum _ { j } R _ { j } \sigma ( u _ { 1 } ^ { \top } X _ { j } ) } { \varsigma - ( \eta / b ) \sum _ { j } R _ { j } \sigma ( u _ { 2 } ^ { \top } X _ { j } ) } \right| \le C \eta d .\tag{19}
$$

## B.2. Allocation and Normal Confinement

Take nested compact allocation intervals $\mathcal { A } \Subset \mathcal { A } ^ { \prime }$ . Around the source minimum manifold over $\mathcal { A } ^ { \prime }$ , let $\xi = 0$ be the normal coordinates. The positive normal Hessian in Lemma 10 yields a smooth uniformly positive matrix $P ( \ell _ { a } )$ and

$$
W _ { s } ( \xi , \ell _ { a } ) = \xi ^ { \top } P ( \ell _ { a } ) \xi , \qquad \dot { W } _ { s } \le - \kappa W _ { s }
$$

for the population flow in a small tube. The derivative of $P$ contributes only cubic terms because $\dot { \ell } _ { a } = O ( d ) = O ( \| \xi \| )$ Choose inner and outer tubes $\sqrt { W _ { s } } \le r$ and $\sqrt { W _ { s } } < 2 r$ inside the neighborhood of (18). A fixed population block

duration $t _ { \mathrm { b l k } }$ maps the inner tube into $\sqrt { W _ { s } } \leq r / 4$ and stays inside the inner tube, provided its allocation remains in $\mathcal { A ^ { \prime } }$ over the duration $t _ { \mathrm { b l k } } + 1$ . Taking � small ensures this allocation condition for a reference block starting in A.

Stop the source process at the first outer normal-tube exit or allocation exit from ${ \mathcal { A } } ,$ and call that time $\zeta _ { s }$ . Killing the distance at this time only decreases it, so (18) implies

$$
\mathbb { E } [ d _ { n } \mathbf { 1 } _ { n < \zeta _ { s } } ] \leq d _ { 0 } ( 1 - \gamma \eta ) ^ { n } , \qquad \mathbb { E } \sum _ { n < \zeta _ { s } } \eta d _ { n } \leq d _ { 0 } / \gamma .\tag{20}
$$

If the initial allocation has margin $m > 0$ from the endpoints of A, (19) and Markov’s inequality give

ℙ(allocation exits at or before the normal tube) $\leq C d _ { 0 } / ( \gamma m )$

(21)

For normal confinement, source gradients are bounded on the slightly larger compact chart and the population field is Lipschitz there. Apply Lemma 8 with $\Psi = \sqrt { W _ { s } } = \left\| P ( \ell _ { a } ) ^ { 1 / 2 } \xi \right\|$ (Lipschitz on the compact chart because � is smooth and uniformly positive), inner radius $r ,$ constraint set $\{ \ell _ { a } \in \mathcal { A } \}$ , block duration $\ell _ { \mathrm { b l k } } = t _ { \mathrm { b l k } }$ , and a fixed tolerance $a \leq r / ( 4 L _ { \Psi } ) \ ( s 0 \varrho = r )$ ; the block property above supplies its flow hypotheses. With $N = \lfloor T / \eta \rfloor$ it gives

ℙ(normal exit by � at or before allocation exit) $\leq C ( 1 + T / \eta ) e ^ { - c / \eta }$

(22)

## B.3. Entry from the Fixed Gaussian Initialization Region

Proposition 11 and local attraction give uniform source convergence on � to an interior compact segment of source clones. Choose the target entry tube of Proposition 6 first. Then choose the source outer tube and allocation intervals so that every point in that outer tube satisfies the target’s entry conditions, including its fixed normal tolerance � and projective tolerance $\epsilon _ { 0 }$ . The estimates of Appendices B.1–B.2 remain valid on smaller tubes.

For any small fixed $\delta _ { \mathrm { { b u r n } } } > 0$ , a single finite population burn-in duration � sends every $\theta _ { 0 } \in K$ into the inner source tube with allocation margin � and $d \leq \delta _ { \mathrm { b u r n } } / 2$ . The paths up to � lie in a common compact set with positive heads and radii. Lemma 7 shows that actual SGD enters with $d \leq \delta _ { \mathrm { b u r n } }$ and the required margins with probability at least $1 - C e ^ { - c _ { \mathrm { b u r n } } / \eta }$ . Choose $\delta _ { \mathrm { b u r n } }$ so that the right side of (21) is at most the allowance $\varepsilon / 2$ , lengthening this fixed � if necessary.

Let $E _ { T }$ be the event of good burn-in entry and no source stop through the source checkpoint at time $\lfloor \boldsymbol { T } / \eta \rfloor \eta$ Equations (20)–(22) give, after changing constants to absorb � and time rounding,

$$
\mathbb { P } \big ( E _ { T } ^ { c } \mid \theta _ { 0 } \big ) \le \varepsilon / 2 + C \big ( 1 + T / \eta \big ) e ^ { - c / \eta } , \qquad \mathbb { E } \big [ d _ { T } \mathbf { 1 } _ { E _ { T } } \mid \theta _ { 0 } \big ] \le C e ^ { - \gamma T } .\tag{23}
$$

## C. Target Confinement and the Bridge

This appendix proves the confinement bound (11) used in Section 7, in the coordinates defined there.

On $e = 0$ , the auxiliary mean field holds $\rho$ and � fixed and satisfies

$$
\dot { h } _ { a } = - \lambda h _ { a } + ( q - B / 2 ) r , \qquad \dot { r } = - \lambda r + ( q - B / 2 ) h _ { a } .
$$

Consequently

$$
\dot { k } = - 2 \lambda k , \qquad \dot { B } = - 2 \lambda B + ( q - B / 2 ) \sqrt { S ^ { 2 } k ^ { 2 } + 4 B ^ { 2 } } .\tag{24}
$$

At $k = 0 , B = B _ { t }$ the normal Jacobian is diag( $- 2 \lambda , - B _ { t } )$ , uniformly over $| \rho | \le \rho _ { 2 }$ . In a suficiently small common normal neighborhood, for $W _ { t } = k ^ { 2 } + ( B - B _ { t } ) ^ { 2 }$ one has

$$
\dot { W } _ { t } \leq - \kappa W _ { t } + C _ { e } \| e \| ^ { 2 } .\tag{25}
$$

Indeed, the full mean field difers smoothly from (24) by $O ( \| e \| )$ in ${ \dot { B } } ;$ Young’s inequality absorbs the resulting $O ( \sqrt { W _ { t } } \left. e \right. )$ term. The identity for �<sup>¤</sup> remains exact. The continuous counterpart of (8) contracts ∥�∥ inside this chart.

Choose a small common radius $R _ { W }$ , then reduce � so that the forcing $C _ { e } \epsilon ^ { 2 } / \kappa$ is at most $R _ { W } ^ { 2 } / 2 5 6$ . Then (25) gives $W _ { t } ( t ) \leq \operatorname* { m a x } \{ W _ { t } ( 0 ) , R _ { W } ^ { 2 } / 2 5 6 \}$ and $W _ { t } ( t ) \leq e ^ { - \kappa t } W _ { t } ( 0 ) + R _ { W } ^ { 2 } / 2 5 6$ along the mean flow while $\| e \| \leq \epsilon$ . Hence a fixed block of duration $\ell _ { \mathrm { b l k } } \geq \kappa ^ { - 1 } \log ( 1 6 / 3 )$ keeps $\sqrt { W _ { t } } \leq R _ { W } / 4$ from any start with $\sqrt { W _ { t } } \leq R _ { W } / 4 .$ , ends with $\sqrt { W _ { t } } \le R _ { W } / 8 ,$ and never leaves $\sqrt { W _ { t } } < R _ { W } / 2$ . Equation (10) in continuous time bounds allocation motion over one block by $C ( \ell _ { \mathrm { b l k } } + 1 ) \epsilon ;$ choose it smaller than half the bufer $\rho _ { 2 } - \rho _ { 1 }$

The source checkpoint is not at the target amplitude $B _ { t }$ . At $e = k = 0 , B = B _ { s } ,$ , (24) is

$$
\dot { B } = B \big ( B _ { t } - B \big ) .
$$

This logistic path stays between $B _ { s }$ and $B _ { t }$ and enters the stationary inner tube after a fixed time $\ell _ { 0 } .$ . Uniform continuity over allocations and directions gives $\nu , \epsilon _ { 0 } > 0$ such that the full auxiliary mean trajectories from the proposition’s initial set stay in a compact bridge tube and enter strictly inside that stationary inner tube. The bridge has positive lower bounds on $h _ { a } , r ,$ lies below $B = 2 q ,$ , and therefore also satisfies the projective contraction estimate, with $C _ { U }$ and � chosen on a chart containing the bridge tube.

The bridge is a single application of Lemma 7 over the fixed duration $\ell _ { 0 } ,$ with a tolerance smaller than the margin by which the bridge mean trajectories enter $\sqrt { W _ { t } } \leq R _ { W } / 4 ;$ ; its failure probability is at most $C e ^ { - c / \eta }$ . Afterwards apply Lemma 8 with $\Psi = \sqrt { W _ { t } } , \varrho = R _ { W } / 4$ , constraint set $\{ \| e \| \le \epsilon , \ | \rho | \le \rho _ { 1 } \}$ , and a fixed tolerance $a \le R _ { W } / ( 1 6 L \Psi )$ . The auxiliary mean is smooth by (6), and bounded inputs bound all martingale increments. Together they give (11):

ℙ(common-tube exit by � at or before other stops) $\leq C ( N + 1 ) e ^ { - c / \eta }$

The bound covers the bridge block, since $1 + N \eta / \ell _ { \mathrm { b l k } } \le C ( N + 1 )$

## D. Proof of the Two-Phase Theorem

Proof of Theorem 1. Part (a) follows from Proposition 12 and the uniform escape estimate in Appendix $\operatorname { H } ,$ applied to the compact set � in Appendix I. For part (b), the source allocation bound (21) supplies the fixed allowance �.

Choose a compact segment containing all source clone limits from �. On clones, the source allocation $\ell _ { a }$ and target allocation satisfy $\rho = \operatorname { t a n h } ( \ell _ { a } / 2 )$ . Select nested target allocation intervals containing this segment with strict margins. Construct the target entry neighborhood first, then fit the source outer tube and allocation interval inside it, as specified above. On the resulting source compact set, smoothness of the coordinate changes gives

$$
\left\| e _ { \mathrm { t a r g e t } } \right\| \le C \left\| w _ { 1 } / a _ { 1 } - w _ { 2 } / a _ { 2 } \right\| = C d .\tag{26}
$$

For example, $z _ { 1 } - ( a _ { 1 } / a _ { 2 } ) z _ { 2 } = a _ { 1 } ( 0 , u _ { 1 } - u _ { 2 } )$ ; orthogonal projection and the compact positive-head bounds give (26).

On the good source event $E _ { T }$ , all target entry conditions hold. Condition on that random checkpoint and apply Proposition 6 to the independent target batches. By (23) and (26), with $N = \operatorname* { m a x } \{ 1 , \lfloor H / \eta \rfloor \}$

$$
\begin{array} { l } { \displaystyle { \mathbb { P } } ( \tau _ { w } ( T ) \leq H \mid \theta _ { 0 } ) \leq { \mathbb { P } } ( E _ { T } ^ { c } \mid \theta _ { 0 } ) + \frac { C b } { \eta } { \mathbb { E } } [ \left\| e _ { T } \right\| \mathbf { 1 } _ { E _ { T } } \mid \theta _ { 0 } ] + C ( 1 + H / \eta ) e ^ { - c / \eta } } \\ { \displaystyle \qquad \leq \varepsilon / 2 + \frac { C b } { \eta } e ^ { - \gamma T } + C \left( 1 + \frac { T + H } { \eta } \right) e ^ { - c _ { \ast } / \eta } , } \end{array}
$$

where $c _ { * }$ is at most each of the burn-in, source and target confinement rates. Keeping instead the finite-� factor of Proposition 6 gives, for every integer $N \geq 1$

$$
\mathbb { P } \left( \tau _ { w } ( T ) \le N \eta \mid \theta _ { 0 } \right) \le \varepsilon / 2 + C b e ^ { - \gamma T } \operatorname* { m i n } \{ N , \eta ^ { - 1 } \} + C ( 1 + T / \eta + N ) e ^ { - c _ { * } / \eta } .\tag{27}
$$

Applied to (27) with $\varepsilon < p / 4$ and $C ( 1 + T / \eta + N ) e ^ { - c _ { * } / \eta } \leq p / 8$ , the argument of Section 5 gives the same necessary sizes for the two-phase SGD run, with $e ^ { \gamma T }$ in place of $e ^ { \lambda T }$ and a smaller $c _ { p } .$

For fresh success, Proposition 12 gives target population convergence on � to risk $R _ { \mathrm { g o o d } } < 0 . 3$ . For each point choose a finite time at which risk is below a fixed intermediate level, say 0.3. Finite-time continuity and a finite cover of � give a common time horizon $t _ { 0 }$ and a positive risk margin for these crossings. The relevant paths lie in a common compact set with nonzero heads and radii. Lemma 7, with a tolerance small compared with the risk margin, gives

$$
\mathbb { P } ( \tau _ { f } > t _ { 0 } \mid \theta _ { 0 } ) \le C e ^ { - c _ { f } / \eta } .
$$

A union bound proves (4) after reducing $c _ { * }$ to at most $c _ { f }$ and enlarging constants.

Proof of Corollary 3. Fix $\theta _ { 0 } \in K$ and �. The source gradient-flow path $\phi _ { t } ^ { s } ( \theta _ { 0 } ) , t \leq T + 1$ , lies in a compact set with positive heads and radii (Lemma 9). On a closed neighborhood of it the population field is Lipschitz, including near coincident directions by the kernel expansion of Appendix E, and sample gradients are bounded independently of �. Since $\phi _ { T } ^ { s } ( \theta _ { 0 } ) \in { \mathcal { D } }$ , Proposition 12 shows that target gradient flow from it converges to risk $R _ { \mathrm { g o o d } } < 0 . 3 _ { \ \mathrm { ; } }$ so choose a finite $t _ { 1 }$ after which its risk remains at most 0.3. By continuous dependence on initial conditions over $[ 0 , t _ { 1 } + 1 ]$ and Lipschitz continuity of the risk, choose $a > 0$ such that every target flow started within 2� of $\phi _ { T } ^ { s } ( \theta _ { 0 } )$ stays in a compact set with positive heads and radii and has risk at most 0.35 on $[ t _ { 1 } , t _ { 1 } + 1 ]$ . For $T > 0 _ { ; }$ Lemma 7 with $\ell = T$ and tolerance $a / 2$ gives $\left\| \theta _ { \lfloor T / \eta \rfloor } - \phi _ { T } ^ { s } ( \theta _ { 0 } ) \right\| \leq a$ with probability at least $1 - C e ^ { - c / \eta }$ ; the rounding $T - \lfloor T / \eta \rfloor \eta < \eta$ changes the flow by $O ( \eta )$ . When $T = 0$ the initial checkpoint is exact and this source-tracking step is omitted. An application over the target duration $t _ { 1 } + 1$ , with a tolerance small compared with the risk margin, shows that target SGD has risk at most 0.4 at update $\lceil t _ { 1 } / \eta \rceil$ , whose time lies in $\left[ t _ { 1 } , t _ { 1 } + \eta \right]$ , with probability at least $1 - C e ^ { - c / \eta }$ . Replacing $t _ { 1 }$ by $t _ { 1 } + 1$ gives the claim. □

Dependence on the weight decay. The proofs use � only through the following facts: $\lambda > 0 ; B _ { s } = 1 - 2 \lambda > 0$ (Lemma 10, Proposition 11); $A = q - \lambda > 0$ (Propositions 4 and 12, Lemma 14); $B _ { s } < B _ { t }$ , which holds because $2 q > 1$ , and $B _ { t } + \lambda / 2 = 2 q - { \textstyle { \frac { 3 } { 7 } } } \lambda < 2 q$ (Section 7); and $R _ { \mathrm { g o o d } } = 1 - 2 q ^ { 2 } + 2 \lambda ^ { 2 } < 0 . 3$ , which gives (2) and the intermediate risk levels 0.3 and 0.35 above. The level $R _ { \mathrm { c o l } } .$ , the function $R _ { \mathrm { m i n } }$ in (38), the initial-risk bound 0.5928, and � and $\mathbb { P } ( K )$ do not depend on �. These facts hold for every $\lambda \in ( 0 , 1 / 5 ]$ , where $R _ { \mathrm { g o o d } } \leq 0 . 2 7 ;$ ; the binding condition is $R _ { \mathrm { g o o d } } < 0 . 3$ , and the others need only $\lambda < 1 / 2$ . Hence all results of Sections 3–7 hold for every fixed $\lambda \in ( 0 , 1 / 5 ]$ , with constants that depend on � and may degrade as $\lambda \to 0$

## E. Population Equations and Regularity

We use the model and notation of Section 3; the symbols $h _ { i } , k _ { i }$ and $s _ { i }$ defined below are local to Appendices E–H. We write $q = 2 / \pi , A = q - \lambda$ , and $B _ { s } = 1 - 2 \lambda$ . The population objective is the same for the disk and $N ( 0 , I _ { 2 } )$ by the radial equivalence in Section 3. The ReLU kernel for unit directions separated by $t \in [ 0 , \pi ]$ is

$$
J ( t ) = { \frac { \sin t + ( \pi - t ) \cos t } { 2 \pi } } , \qquad g ( t ) = - J ^ { \prime } ( t ) = { \frac { ( \pi - t ) \sin t } { 2 \pi } } .
$$

In particular, $J ( 0 ) = 1 / 2 , J ( \pi ) = 0 , g > 0$ on (0, �), and

$$
J ( t ) = { \frac { 1 } { 2 } } - { \frac { t ^ { 2 } } { 4 } } + { \frac { t ^ { 3 } } { 6 \pi } } + O ( t ^ { 4 } ) , \qquad g ( t ) = { \frac { t } { 2 } } + O ( t ^ { 2 } ) .\tag{28}
$$

For the source sector write $\alpha = \vartheta _ { 1 } , \beta = - \vartheta _ { 2 } , \delta = \alpha + \beta$ , and define

$$
q _ { 1 } = J ( \alpha ) - \omega _ { 1 } / 2 - \omega _ { 2 } J ( \delta ) , \qquad q _ { 2 } = J ( \beta ) - \omega _ { 2 } / 2 - \omega _ { 1 } J ( \delta ) .
$$

With $h _ { i } = a _ { i } / r _ { i }$ , the exact source equations are

$$
\begin{array} { l } { \dot { a } _ { i } = r _ { i } q _ { i } - \lambda a _ { i } , } \\ { \dot { \alpha } = h _ { 1 } \left[ - g ( \alpha ) + \omega _ { 2 } g ( \delta ) \right] , } \end{array}
$$

$$
\dot { \boldsymbol { r } } _ { i } = a _ { i } \boldsymbol { q } _ { i } - \lambda \boldsymbol { r } _ { i } ,
$$

$$
\dot { \alpha } = h _ { 1 } [ - g ( \alpha ) + \omega _ { 2 } g ( \delta ) ] , \qquad \dot { \beta } = h _ { 2 } [ - g ( \beta ) + \omega _ { 1 } g ( \delta ) ] .
$$

$$
\dot { \omega } _ { i } = ( a _ { i } ^ { 2 } + r _ { i } ^ { 2 } ) q _ { i } - 2 \lambda \omega _ { i } ,\tag{29}
$$

(30)

For the target, $\mathbb { E } [ \left. X \right. \sigma ( u ^ { \top } X ) ] = q$ for every unit �. Consequently, (29) holds with

$$
q _ { i } = q - \omega _ { i } / 2 - \omega _ { j } J ( \delta ) , \qquad \dot { \delta } = ( h _ { 1 } \omega _ { 2 } + h _ { 2 } \omega _ { 1 } ) g ( \delta ) , \quad j \neq i .\tag{31}
$$

Here $\delta \in ( 0 , \pi )$ is the angular separation; the absolute orientation does not enter the radial target risk.

For either task, set $k _ { i } = a _ { i } ^ { 2 } - r _ { i } ^ { 2 }$ and $s _ { i } = a _ { i } + r _ { i }$ . Direct calculation gives

$$
k _ { i } ( t ) = k _ { i } ( 0 ) e ^ { - 2 \lambda t } , \quad \dot { h } _ { i } = q _ { i } ( 1 - h _ { i } ^ { 2 } ) , \quad ( \log h _ { i } ) ^ { \ast } = - \frac { k _ { i } } { \omega _ { i } } q _ { i } , \quad ( \log s _ { i } ) ^ { \ast } = q _ { i } - \lambda .\tag{32}
$$

Whenever $a _ { i } , r _ { i } > 0$ and $q _ { i } \ge - \omega _ { i } / 2$ , one has

$$
D ^ { + } \left| \log h _ { i } \right| \le \frac { | k _ { i } | } { 2 } , \qquad \left| \log h _ { i } ( t ) \right| \le \left| \log h _ { i } ( 0 ) \right| + \frac { | k _ { i } ( 0 ) | } { 4 \lambda } .\tag{33}
$$

Indeed, log $h _ { i }$ has the sign of $k _ { i }$ . Thus $h _ { i }$ stays bounded above and away from zero. $\operatorname { I f } q _ { i }$ is bounded, the logarithmic equations for $a _ { i } , r _ { i }$ then preclude a zero at finite time.

On bounded parameter sets with $r _ { i } > 0$ , the only potentially unbounded population Hessian terms are bounded by constants times $a _ { i } / r _ { i }$ . This follows by diferentiating the two-homogeneous Gaussian kernel: its second

derivatives in $w _ { i }$ are of size $O ( 1 / r _ { i } )$ , multiplied by $a _ { i } .$ . The kernel expansion also gives bounded second derivatives at coincident directions. Consequently, (33) bounds the Hessian along the trajectories considered below. Coercivity bounds their parameters and gradients. Energy dissipation and uniform continuity therefore imply

$$
\int _ { 0 } ^ { \infty } \| \nabla F _ { s } \| ^ { 2 } \ d t < \infty , \qquad \| \nabla F _ { s } ( \theta ( t ) ) \| \longrightarrow 0 .\tag{34}
$$

At a limiting zero unit, $k _ { i } \to 0$ forces both $a _ { i } , r _ { i } \to 0$ , and that unit’s gradient tends continuously to zero.

## F. Global Source Convergence

Lemma 9 (Invariance and angular caps). Source trajectories from D remain in D. Every parameter is bounded, every $a _ { i } , r _ { i }$ is positive at finite time, and

$$
\alpha ( t ) \le \operatorname* { m a x } \{ \alpha ( 0 ) , \pi / 2 \} < \pi , \qquad \beta ( t ) \le \operatorname* { m a x } \{ \beta ( 0 ) , \pi / 2 \} < \pi .
$$

Proof. In the region, $J ( \alpha ) , J ( \beta ) \ge J ( \delta )$ and $\omega _ { j } \leq 1 , \ s o \ q _ { i } \geq - \omega _ { i } / 2$ . Apply (33). At $\omega _ { i } = 1 , q _ { i } \le 0 ;$ , hence $\dot { \omega } _ { i } < 0$ . At $\alpha = 0$ or $\beta = 0$ , (30) points inward; more precisely $g ( t ) \leq t / 2$ gives $\dot { \alpha } \ge - h _ { \operatorname* { m a x } } \alpha / 2$ and the analogous bound for $\beta ,$ so these angles stay strictly positive. $\mathrm { A t } \delta = \pi , \dot { \delta } = - h _ { 1 } g ( \alpha ) - h _ { 2 } g ( \beta ) < 0$ . Hence no boundary of ${ \mathcal { D } } _ { : }$ , including the zero-unit boundaries, is reached in finite time. Coercivity gives bounded parameters. Finally � decreases on $[ \pi / 2 , \pi ]$ . If $\alpha \ge \pi / 2$ , then $\delta \ge \alpha$ and $\dot { \alpha } \leq - h _ { 1 } ( 1 - \omega _ { 2 } ) g ( \alpha ) \leq 0 ;$ the other angle is identical. □

Define the fully active clone manifold

$$
S = \{ a _ { i } = r _ { i } = \sqrt { \omega _ { i } } , \ : w _ { i } = r _ { i } e _ { 1 } , \ : \omega _ { i } > 0 , \ : \omega _ { 1 } + \omega _ { 2 } = B _ { s } \} .\tag{35}
$$

Its closure adds two endpoints, each with one absent unit.

Lemma 10 (Source minima and local attraction). Every point of $\overline { s }$ is a global source minimizer with value $\lambda - \lambda ^ { 2 }$ . The manifold S is normally attracting. On each interior compact segment its five normal decay rates are $B _ { s } , 1 / 2 , \lambda , 2 \lambda , 2 \lambda$

Proof. Since $\left\| \sigma ( w _ { i } ^ { \top } X ) \right\| _ { L ^ { 2 } } = r _ { i } / \sqrt { 2 }$ , the penalty is at least $\begin{array} { r } { \lambda \sum _ { i } | \omega _ { i } | \ge \lambda \sqrt { 2 } \| f \| _ { L ^ { 2 } } } \end{array}$ . The reverse triangle inequality then gives

$$
F _ { s } \geq { \frac { 1 } { 2 } } ( \| f \| _ { L ^ { 2 } } - 1 / { \sqrt { 2 } } ) ^ { 2 } + \lambda { \sqrt { 2 } } \| f \| _ { L ^ { 2 } } \geq \lambda - \lambda ^ { 2 } .
$$

Equality holds at (35), where $R _ { s } = \lambda ^ { 2 }$

For the local statement write $p = \omega _ { 1 } / B , \tilde { B } = B - B _ { s } , \mu = p \vartheta _ { 1 } + ( 1 - p ) \vartheta _ { 2 }$ , and use $k _ { 1 } , k _ { 2 }$ as balance coordinates. The exact objective and (28) give, uniformly for interior �,

$$
F _ { s } - ( \lambda - \lambda ^ { 2 } ) = \frac { \tilde { B } ^ { 2 } } { 4 } + \frac { B _ { s } \mu ^ { 2 } } { 4 } + \frac { \lambda B _ { s } p ( 1 - p ) \delta ^ { 2 } } { 2 } + \lambda \sum _ { i } \frac { k _ { i } ^ { 2 } } { 8 \omega _ { i } } + o ( \Vert \chi \Vert ^ { 2 } ) ,
$$

where $\chi = ( \tilde { B } , \mu , \delta , k _ { 1 } , k _ { 2 } )$ . The only null direction is tangent to S. Linearizing (29)–(32) yields the stated rates. In a small tube the objective gap is comparable to $\| \chi \| ^ { 2 }$ , its squared gradient dominates that gap, and its gradient is at most � ∥�∥. Energy dissipation gives exponential normal decay, while the remaining path length is exponentially small. A smaller tube is thus trapped and every trajectory in it converges exponentially to one point of S. These estimates are uniform on interior compact segments. □

Proposition 11 (Global, fully active source limit). Every source trajectory from D converges to a point of S.

Proof. We first classify the omega-limit points. By (34) and $k _ { i } \to 0$ , they are balanced critical points, allowing absent units. At a fully active critical point with �, $\beta > 0$ and $\delta < \pi$ , the angular equations imply

$$
\omega _ { 2 } = \frac { g ( \alpha ) } { g ( \delta ) } , \qquad \omega _ { 1 } = \frac { g ( \beta ) } { g ( \delta ) } .
$$

Substitution in $q _ { 1 } = \lambda$ gives the contradiction

$$
\lambda = - \frac { \beta \sin \alpha \sin \delta + \alpha ( 2 \pi - \delta ) \sin \beta } { 2 \pi ( \pi - \delta ) \sin \delta } < 0 .
$$

A zero source angle with the other positive is excluded by its inward angular derivative; $\delta = \pi$ is excluded by its strictly negative derivative and the angular caps. Hence every fully active critical point lies in S. A single active unit at a critical point must point along $e _ { 1 }$ and have amplitude $B _ { s }$ : its angular derivative vanishes only at 0 or $\pi ,$ and � is excluded. The only remaining critical point is the origin.

The omega-limit set is nonempty, compact, connected, and has a single objective value. The origin has value $1 / 4 ,$ distinct from $\lambda - \lambda ^ { 2 }$ . If the origin belonged to this set, the trajectory would therefore converge to the origin. Then $\omega _ { 1 } , \omega _ { 2 }  0 ;$ the positive lower bounds on $h _ { i }$ and the angular caps in Lemma 9 force $\alpha , \beta \to 0$ in (30). Thus $q _ { i } \to 1 / 2 > \lambda$ , and $\dot { s } _ { i } = ( q _ { i } - \lambda ) s _ { i }$ eventually grows at a fixed exponential rate, a contradiction.

The omega-limit set is consequently contained in ${ \overline { { s } } } .$ If it meets $s ,$ Lemma 10 implies convergence to an interior point. Otherwise connectedness forces convergence to one of the two endpoints. Suppose $\omega _ { 1 }  B _ { s } , \alpha  0$ , and $\omega _ { 2 } \to 0$ . The angular cap gives

$$
q _ { 2 } = 2 \lambda J ( \beta ) + o ( 1 ) ,
$$

which is eventually bounded below by a positive constant, uniformly over the allowed weak-unit angles. Hence $\dot { h } _ { 2 } = q _ { 2 } ( 1 - h _ { 2 } ^ { 2 } )$ yields exponential convergence $h _ { 2 }  1$ . Also $h _ { 1 }  1$ exponentially by $k _ { 1 }  0$ and $\omega _ { 1 }  B _ { s } > 0$ The weak angle equation is $\dot { \beta } = - 2 \lambda g ( \beta ) + o ( 1 )$ ; the cap and the sign away from zero imply $\beta \to 0$ . Now (28) gives $\dot { \delta } = [ - \lambda + o ( 1 ) ] \delta$ , so � decays at least as $e ^ { - \lambda t / 2 }$ after a suficiently large time. But

$$
{ \frac { d } { d t } } \log { \frac { s _ { 1 } } { s _ { 2 } } } = q _ { 1 } - q _ { 2 } = J ( { \boldsymbol { \alpha } } ) - J ( { \boldsymbol { \beta } } ) + \left( \omega _ { 2 } - \omega _ { 1 } \right) \left[ 1 / 2 - J ( { \boldsymbol { \delta } } ) \right] = O ( { \delta ^ { 2 } } ) .
$$

This is integrable, so $s _ { 1 } / s _ { 2 }$ has a finite positive limit, contradicting the supposed endpoint. The other endpoint is identical. □

## G. Global Target Convergence and the Risk Gap

Proposition 12 (Fresh target success). Suppose initially $a _ { i } , r _ { i } > 0 , \omega _ { i } \leq 2 q$ and $0 < \delta < \pi .$ . Target gradient flow keeps both units positive, has $\delta \to \pi$ and �<sub>1</sub>, $\omega _ { 2 }  2 A _ { \prime }$ , and satisfies $R _ { t } \to R _ { \mathrm { g o o d } }$ . In particular it succeeds from every point of D.

Proof. At $\omega _ { i } = 2 q ,$ (31) gives $q _ { i } \leq 0 _ { \colon }$ so this amplitude bound is invariant. It also gives $q _ { i } \ge - \omega _ { i } / 2 _ { ; }$ , and (33) again bounds $h _ { i }$ above and away from zero and excludes finite-time zero units. Since $q _ { i } \ge q - B / 2$ , whenever $B \leq 2 A$ one has $q _ { i } \ge \lambda > 0$ and

$$
\dot { B } \geq 2 \sum _ { i } \omega _ { i } ( q _ { i } - \lambda ) \geq B ( 2 A - B ) .\tag{36}
$$

Thus $B \geq \operatorname* { m i n } \{ B ( 0 ) , 2 A \} > 0$ . The angle equation implies

$$
\dot { \delta } \geq h _ { \operatorname* { m i n } } \operatorname* { m i n } \{ B ( 0 ) , 2 A \} g ( \delta ) ,
$$

forcing $\delta \to \pi$ . Once $J ( \delta ) \leq A / ( 8 q )$ , if $\omega _ { i } \leq A$ then $q _ { i } - \lambda \geq A / 4$ , and (29) gives $\dot { \omega } _ { i } \geq ( A / 2 ) \omega _ { i }$ <sub>�</sub>. Both amplitudes therefore reach � in finite time and stay above it. With $k _ { i } \to 0 ,$ , their equations become

$$
\dot { \omega } _ { i } = 2 \omega _ { i } ( A - \omega _ { i } / 2 ) + o ( 1 ) .
$$

The sign above and below the unique positive zero yields $\omega _ { i }  2 A$ . The limiting prediction has the form $2 A \left| \boldsymbol { u } ^ { \intercal } \boldsymbol { X } \right| .$ and direct evaluation gives $R _ { t }  1 - 2 q ^ { 2 } + 2 \lambda ^ { 2 } = R _ { \mathrm { g o o d } }$ □

A positively proportional pair is a single ReLU $\omega \sigma ( \boldsymbol { u } ^ { \intercal } \boldsymbol { x } )$ . Its radial target risk is

$$
R _ { t } = 1 + \omega ^ { 2 } / 4 - q \omega \geq 1 - q ^ { 2 } = 0 . 5 9 4 7 1 5 2 \ldots > 0 . 4 .\tag{37}
$$

More generally, at fixed separation $0 \leq \delta \leq \pi$ , minimizing the quadratic risk over the two amplitudes gives

$$
R _ { t } \ge R _ { \operatorname* { m i n } } ( \delta ) : = 1 - \frac { q ^ { 2 } } { 1 / 2 + J ( \delta ) } .\tag{38}
$$

For $\delta > 0$ this follows by inverting the $2 \times 2$ Gram matrix with diagonal $1 / 2$ and of-diagonal $J ( \delta )$ ; the $\delta = 0$ value follows by continuity.

## H. Source Contraction and Target Escape Time

Fix a compact set $K \subset \mathcal { D } ;$ in Theorem 1 it is the set of Appendix I. Proposition 11 and local attraction imply uniform eventual entry into neighborhoods of interior clone points. Indeed, each initial condition has a finite entry time into a smaller attracting tube; finite-time continuity makes the same true in an initial neighborhood. A finite cover of � provides a common entry-time bound and an interior compact segment containing all limiting clones.

Lemma 13 (A nonzero source splitting coeficient). Uniformly for $\theta _ { 0 } \in K ,$ , the source separation satisfies

$$
\delta _ { s } ( T ) = C _ { s } ( \theta _ { 0 } ) e ^ { - \lambda T } ( 1 + o ( 1 ) ) , \qquad 0 < C _ { - } \leq C _ { s } ( \theta _ { 0 } ) \leq C _ { + } < \infty .
$$

The source amplitudes converge to positive limits uniformly bounded away from zero, and $k _ { i } ( T ) = k _ { i } ( 0 ) e ^ { - 2 \lambda T }$

Proof. The last two statements follow from the preceding finite-cover argument and (32). Near an interior clone, $\stackrel { \cdot } { h _ { i } } - \stackrel { \cdot } { 1 } = O ( e ^ { - 2 \lambda T } )$ . Using � $, \beta \ge 0$ and $\alpha + \beta = \delta _ { \mathrm { { } } }$ , expand (30) to obtain

$$
\frac { d } { d T } \log \delta _ { s } ( T ) = - \lambda + O \big ( \delta _ { s } + | B - B _ { s } | + | h _ { 1 } - 1 | + | h _ { 2 } - 1 | \big ) .
$$

The error is exponentially integrable by Lemma 10. The separation is strictly positive at every finite time, so integration gives a finite strictly positive limit of $e ^ { \lambda T } \delta _ { s } ( T )$ . Finite-time continuity and the uniform tail bound make that limit continuous on $K ,$ proving the uniform positive bounds. □

Lemma 14 (Uniform target escape time). For suficiently small fixed $\delta _ { \star } > 0 ,$ let $t _ { \delta _ { \star } } ( T )$ be the first target time, after source duration $T ,$ at which $\delta = \delta _ { \star }$ <sub>★</sub>. Uniformly on � and for $T \geq T _ { \star } ,$ , with $T _ { \star }$ large enough that $\delta _ { s } ( T _ { \star } ) < \delta _ { \star }$

$$
t _ { \delta _ { \star } } ( T ) = \frac { \lambda } { A } T + O ( 1 ) .\tag{39}
$$

At that time both amplitudes belong to a fixed compact positive interval.

Proof. All bounds below are uniform over �. Choose � large. Initially the target amplitudes are close to an interior source clone, so their total has a positive lower bound and their ratio lies in $[ R _ { 0 } ^ { - 1 } , R _ { 0 } ]$ for some fixed $R _ { 0 } \geq 1$ . Target invariance and (36) give

$$
0 < B _ { \operatorname* { m i n } } \le B ( t ) \le 4 q , \qquad | k _ { i } ( t ) | \le C _ { 0 } e ^ { - 2 \lambda ( T + t ) } .
$$

Stop before $\delta \ = \delta ,$ <sub>★</sub> if the amplitude ratio first leaves $[ ( 2 R _ { 0 } ) ^ { - 1 } , 2 R _ { 0 } ]$ . Within this stopped tube, $\omega _ { i } ~ \ge ~ m ~ : =$ $B _ { \mathrm { m i n } } / ( 1 + 2 R _ { 0 } ) > 0$ . Hence

$$
h _ { i } = 1 + O ( | k _ { i } | ) , \qquad a _ { i } ^ { 2 } + r _ { i } ^ { 2 } = \sqrt { 4 \omega _ { i } ^ { 2 } + k _ { i } ^ { 2 } } = 2 \omega _ { i } + O ( k _ { i } ^ { 2 } ) .
$$

By taking $T$ large and $\delta _ { \star }$ small, (31) implies $\dot { \delta } \geq \nu \delta$ for a fixed $\nu > 0$ . Before the stopping time,

$$
\int \delta ( t ) ^ { j } d t \leq \frac { \delta _ { \star } ^ { j } } { j \nu } \quad ( j = 1 , 2 ) , \qquad \int | k _ { i } ( t ) | ^ { j } d t = O ( e ^ { - 2 j \lambda T } ) .\tag{40}
$$

Put $D ( \delta ) = 1 / 2 - J ( \delta ) = O ( \delta ^ { 2 } )$ . The exact amplitude equations give

$$
\frac { d } { d t } \log \frac { \omega _ { 1 } } { \omega _ { 2 } } = - 2 ( \omega _ { 1 } - \omega _ { 2 } ) D ( \delta ) + O ( k _ { 1 } ^ { 2 } + k _ { 2 } ^ { 2 } ) .
$$

The total change of this logarithmic ratio is therefore at most $C \delta _ { \star } ^ { 2 } + C e ^ { - 4 \lambda T }$ . First choose $\delta _ { \star }$ small and then � large so that this is less than log 2. The ratio boundary cannot be reached. Thus the compact positive-amplitude tube persists until $\delta = \delta _ { \star }$ ; the lower angular growth bound ensures this occurs in finite time.

In this tube, summing the amplitude equations gives

$$
\dot { B } = B \left( 2 A - B \right) + E ( t ) , \qquad | E ( t ) | \le C \bigl ( \delta ^ { 2 } + k _ { 1 } ^ { 2 } + k _ { 2 } ^ { 2 } \bigr ) .
$$

Writing $z = B - 2 A$ , one has $\dot { z } = - B z + E$ . Variation of constants and $B \geq B _ { \mathrm { m i n } }$ imply

$$
\int _ { 0 } ^ { t _ { \delta _ { \star } } } \left| B - 2 A \right| d t \leq \frac { \left| B ( 0 ) - 2 A \right| + \int _ { 0 } ^ { t _ { \delta _ { \star } } } \left| E ( t ) \right| d t } { B _ { \operatorname* { m i n } } } = O ( 1 ) .\tag{41}
$$

Finally (28) and (31) yield

$$
\frac { d } { d t } \log \delta = \frac { B } { 2 } + O ( \delta + | k _ { 1 } | + | k _ { 2 } | ) .
$$

Integrating, using (40) and (41), gives

$$
\log \frac { \delta _ { \star } } { \delta _ { s } ( T ) } = A t _ { \delta _ { \star } } ( T ) + O ( 1 ) .
$$

Lemma 13 now proves (39).

□

Proof of the hitting-time law. Choose $\delta _ { \star }$ as in Lemma $^ { 1 4 , }$ further reducing it so that $R _ { \mathrm { m i n } } ( \delta _ { \star } ) > 0 . 4$ . Since � decreases, (38) precludes crossing the risk threshold before $t _ { \delta _ { \star } } ( T )$ . At the exit time, the states have separation $\delta _ { \star } ,$ � $\leq \omega _ { i } \leq 2 q$ , and uniformly small balance errors. Including the irrelevant overall direction, they lie in a fixed compact subset of the target basin in Proposition 12. Every point of that compact set reaches risk strictly below 0.4 in finite time. For each point, continuous dependence gives the same strict crossing at its chosen time for a neighborhood of initial states. A finite cover therefore bounds the additional crossing time uniformly. Hence $t _ { \delta _ { \star } } ( T ) \leq \tau _ { \mathrm { G F } } ( T ) \leq t _ { \delta _ { \star } } ( T ) + O ( 1 )$ for large �. For � in a bounded interval $[ 0 , T _ { * } ]$ , the checkpoints $\phi _ { T } ^ { s } ( \theta _ { 0 } )$ with $\theta _ { 0 } \in K$ form a compact subset of D (Lemma $9 )$ , and the same finite-cover argument bounds $\tau _ { \mathrm { G F } } ( T )$ uniformly there. This proves (3). The same finite-cover argument applied directly to � supplies a uniform fresh-training time. □

## I. The Compact Gaussian Basin and Initial Risk

For one unit under independent scalar standard Gaussians, � is independent of the Rayleigh radius � and its uniform direction. For $x \ge 0$

$$
\mathbb { P } ( a > 0 , a r > x ) = \int _ { 0 } ^ { \infty } \frac { e ^ { - a ^ { 2 } / 2 } } { \sqrt { 2 \pi } } e ^ { - x ^ { 2 } / ( 2 a ^ { 2 } ) } d a = \frac { 1 } { 2 } e ^ { - x } .
$$

Thus, conditional on a positive head, $\omega = a r$ has distribution Exp(1), and $\mathbb { P } ( a > 0 , a r < 1 ) = ( 1 - e ^ { - 1 } ) / 2$ . The two ordered angular triangles $\alpha , \beta > 0 , \alpha + \beta < \pi$ have combined area $\pi ^ { 2 }$ out of the direction square of area $4 \pi ^ { 2 }$ . Their probability is $1 / 4$ , independent of the amplitudes. This proves

$$
\mathbb { P } ( \mathcal { D } ) = \frac { ( 1 - e ^ { - 1 } ) ^ { 2 } } { 1 6 } .
$$

Conditional on two positive heads, � has distribution Gamma(2, 1). The condition $B < 3 / 4$ already ensures $\omega _ { i } < 1$ so

$$
\mathbb { P } ( \mathcal { D } _ { 0 } ) = \frac { 1 } { 1 6 } \mathbb { P } ( \mathtt { G a m m a } ( 2 , 1 ) < 3 / 4 ) = \frac { 1 - ( 7 / 4 ) e ^ { - 3 / 4 } } { 1 6 } > 0 . 0 1 .
$$

For any point in $\mathcal { D } _ { 0 } ,$ , positivity of the kernel gives

$$
R _ { t } = 1 + \frac { \omega _ { 1 } ^ { 2 } + \omega _ { 2 } ^ { 2 } } { 4 } + \omega _ { 1 } \omega _ { 2 } J ( \delta ) - q B \geq 1 + \frac { B ^ { 2 } } { 8 } - q B \geq 1 + \frac { 9 } { 1 2 8 } - \frac { 3 } { 2 \pi } > 0 . 5 9 2 8 ,
$$

since the last quadratic decreases on $[ 0 , 3 / 4 ]$

An explicit compact subset. Put $m = 1 0 ^ { - 4 }$ and $M = 5 .$ and define � as the union of the two neuron orderings satisfying

$$
\begin{array} { r } { m \leq a _ { i } \leq M , \quad m \leq \| w _ { i } \| \leq M , \quad \omega _ { 1 } + \omega _ { 2 } \leq \frac { 3 } { 4 } - m , \quad \vartheta _ { 1 } \geq m , \quad \vartheta _ { 2 } \leq - m , \quad \vartheta _ { 1 } - \vartheta _ { 2 } \leq \pi - m . } \end{array}
$$

This is a compact subset of $\mathcal { D } _ { 0 }$ , bounded away from zero radii and angular branch cuts.

Under independent standard Gaussian parameters, the probability removed from $\mathcal { D } _ { 0 }$ is bounded by

$$
\mathbb { P } ( \mathcal { D } _ { 0 } \setminus K ) \le \frac { 2 m } { \sqrt { 2 \pi } } + m ^ { 2 } + \frac { 2 e ^ { - M ^ { 2 } / 2 } } { M \sqrt { 2 \pi } } + 2 e ^ { - M ^ { 2 } / 2 } + \frac { 3 m } { \pi } + \frac { m } { 4 e } < 1 . 9 3 \times 1 0 ^ { - 4 } .
$$

Indeed, the first four terms bound the two small positive heads, two small Rayleigh radii, two large heads (Mills’ bound), and two large radii. Each angular strip $| \vartheta _ { i } | < m$ has probability $m / \pi$ . The density of the absolute diference of two uniform angles is $( 2 \pi - d ) / ( 2 \pi ^ { 2 } )$ for $d \in ( 0 , 2 \pi )$ , so the strip $\pi - m < d < \pi$ has probability at most $m / \pi$ . Finally, conditional on two positive heads the sum � has density $x e ^ { - x } \leq e ^ { - 1 }$ ; the head-sign probability is $1 / 4 ,$ , giving $m / ( 4 e )$ for the strip $3 / 4 - m < B < 3 / 4$ . A union bound gives

$$
\mathbb { P } ( K ) \ge \frac { 1 - ( 7 / 4 ) e ^ { - 3 / 4 } } { 1 6 } - 1 . 9 3 \times 1 0 ^ { - 4 } > 0 . 0 1 0 6 4 > 0 . 0 1 .
$$

## J. Experimental Details

Implementation and random streams. The settings are in Section $8 . { \mathrm { ~ C } } + + 1 7$ code runs on one CPU core in double precision. Disagreement counts record sampled inputs on which the current gates difer. The target SGD stream depends only on the initialization and $T ,$ so checkpoints from the two source optimizers meet the same target samples; constructed checkpoints use the same streams in every cell. The code, with its seeds and the saved Gaussian initial vectors, is provided as ancillary files with this preprint.

Identical checkpoints (Figure 2). The 64 saved vectors are Gaussian rejection draws from $\mathcal { D } _ { 0 } ;$ membership in the slightly smaller set � is not enforced. Each source optimizer supplies checkpoints at $T \in \{ 0 , 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 \}$ along one trajectory, with the two choices coinciding at $T = 0$ . Target risk is checked every ten updates through $H = 2 0 0$ . Table 1 gives the recovery counts; the source-GD columns are the protocol of Corollary 5. The dashed line in Figure 2 is $A ^ { - 1 } \log ( 1 / \delta _ { T } ) + c ,$ with $c \approx 3 . 9$ the median GD ofset. Points well above the line come from checkpoints with a nearly inactive unit, such as the seven of Section $\begin{array} { r } { 8 ; } \end{array}$ GD must regrow its amplitude, and the hitting time stays close to the GF delay $\lambda T / A$ . For eight saved initializations at $T = 4 0 0$ , halving the GD step in both phases from $1 0 ^ { - 3 } \mathrm { t o } 5 \times 1 0 ^ { - 4 }$ , then to $2 . 5 \times 1 0 ^ { - 4 }$ , changes target hitting times by at most 0.006 and 0.003.

Constructed checkpoints (Figure 3a). Balanced checkpoints at the source amplitude ${ B _ { s } } = 0 . 9 \colon { a _ { i } } = \left\| { w _ { i } } \right\| =$ 0.45 and $w _ { i } = \sqrt { 0 . 4 5 } ( \cos ( \delta _ { 0 } / 2 ) , ( - 1 ) ^ { i + 1 }$ sin $( \delta _ { 0 } / 2 ) )$ ). The parameter grid in Section 8 varies $\delta _ { 0 } .$ , � and $\eta$ independently; risk is checked every update. Wilson intervals use the independent streams within each cell. The hollow diamonds (checkpoints from Gaussian initializations) pool two source optimizers and four positive source durations from the same 64 draws; these 512 dependent records give fractions in bins of width 0.5 in log $_ { 1 0 } \Gamma$ with at least five records, without binomial intervals. For the 12 cells with recovery fraction $\hat { p } \in ( 0 , 1 )$ , the efective-intensity ratio $\kappa _ { \mathrm { e f f } } = - \pi \lambda \log ( 1 - \hat { p } ) / \Gamma$ ranges from 0.49 to 1.49, with ten values below one. Values below one are expected: a sampled disagreement need not trigger recovery, and the amplitude transient from 0.9 speeds early contraction. The two values above one come from cells with $5 / 6 4$ and 9/64 recoveries, whose Wilson intervals contain the reference. In every cell, recovery occurs only after a sampled disagreement; among runs that sample one, the recovered fraction ranges from 0.81 to 1.

Long horizons (Figure 3b). Constructed checkpoints as above, $\eta = 0 . 0 1$ , 128 new streams per setting, risk checked every ten updates. The settings repeat cells of Figure 3(a), and the estimates agree within sampling error $( { \mathrm { f o r ~ } } b = 6 4 , \delta _ { 0 } = 1 0 ^ { - 4 } \colon 6 0 / 6 4$ there, $1 2 6 / 1 2 8$ here). GD reaches risk 0.4 at $t = 2 2 . 1$ , 18.2 and 14.3 for $\delta _ { 0 } = 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 }$ . Figure 3(b) shows the settings with $b = 1 6 , \delta _ { 0 } \in \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ and $b = 6 4 , \delta _ { 0 } = 1 0 ^ { - 5 }$ Table 2 lists all five settings. The last recovery in the five settings occurs at $t = 7 6 . 1$ , 68.1, 103.7, 112.1 and 64.2,

Table 2 | Long horizons. Recovered fractions by target time � in the five constructed settings $( \eta = 0 . 0 1$ , 128 streams each), the Poisson reference, and the frozen-angle reference $1 - e ^ { - \Gamma t / \pi } \mathrm { a t } t = 2 0 0$
<table><tr><td>b</td><td> $\delta _ { 0 }$ </td><td>Γ</td><td> $\mathrm { r e c . ~ } t \leq 6 0$ </td><td> $\operatorname { r e c . } t \leq 2 0 0$ </td><td> $\mathrm { r e c . ~ } t \leq 4 5 0$ </td><td> $1 - e ^ { - \Gamma / ( \pi \lambda ) }$ </td><td> $\mathrm { f r o z e n } , t = 2 0 0$ </td></tr><tr><td>16</td><td> $1 0 ^ { - 5 }$ </td><td>0.016</td><td>0.070</td><td>0.078</td><td>0.078</td><td>0.097</td><td>0.639</td></tr><tr><td>64</td><td> $1 0 ^ { - 5 }$ </td><td>0.064</td><td>0.227</td><td>0.250</td><td>0.250</td><td>0.335</td><td>0.983</td></tr><tr><td>1</td><td> $1 0 ^ { - 3 }$ </td><td>0.100</td><td>0.305</td><td>0.336</td><td>0.336</td><td>0.471</td><td>0.998</td></tr><tr><td>16</td><td> $1 0 ^ { - 4 }$ </td><td>0.160</td><td>0.578</td><td>0.633</td><td>0.633</td><td>0.639</td><td>1.000</td></tr><tr><td>64</td><td> $1 0 ^ { - 4 }$ </td><td>0.640</td><td>0.969</td><td>0.984</td><td>0.984</td><td>0.983</td><td>1.000</td></tr></table>

respectively. The angle of unrecovered runs decays at rate $\approx \lambda$ , for example from $1 0 ^ { - 4 } \mathrm { t o } 5 . 8 \times 1 0 ^ { - 7 }$ by $t = 1 0 0 ,$ and reaches double-precision resolution $( \approx 1 0 ^ { - 1 5 } )$ at $t \approx 4 6 0$ for $\delta _ { 0 } = 1 0 ^ { - 5 }$ and $t \approx 5 0 0 – 5 5 0$ for the larger $\delta _ { 0 } ,$ long after the last recovery. Six of the 348 unrecovered runs sampled a disagreement. A rerun of all 640 runs in 80-bit extended precision through $t = 1 0 0 0$ gives the same recovery and first-disagreement times; only the floor of unrecovered angles moves, to about $1 0 ^ { - 1 9 }$

Trajectory (Figure 1b–d). Panels (c,d) show four target SGD streams, which coincide to line width, and (c) shares the vertical scale of (b). GD crosses risk 0.4 at � ≈ 34.4. The checkpoint has angle $1 . 6 7 \times 1 0 ^ { - 8 }$ . Fitted log-angle slopes are −0.0500 during source training and 0.5861 and −0.0500 for target GD and SGD, matching $- \lambda , 2 / \pi - \lambda \approx 0 . 5 8 6 6$ and −�. Panel (a) is a schematic with $\delta = 0 . 3$

Joint dependence on � and �. SGD in both phases from the 64 saved initializations, $b ~ = ~ 1 6 , ~ \eta ~ \in$ $\{ 4 , 2 , 1 , 0 . 5 , 0 . 2 5 \} \times 1 0 ^ { - 3 }$ , checkpoints along one source trajectory per initialization and step size at $T \ =$ $0 , 2 5 , \ldots , 4 0 0 .$ , and a fresh target stream from each checkpoint through $H \ : = \ : 2 0 0$ , with risk checked every ten updates (Figure 4). Records at diferent � share a source trajectory and are dependent. Fresh runs $( T = 0 )$

always recover, and at each fixed � the recovered fraction increases as decreases, up to sampling fluctuations. The source time $T _ { 1 / 2 } ( \eta )$ at which half of the runs recover, interpolated linearly between adjacent grid points, is 241.7, 257.9, 265.6, 284.4 and 295.0; subtracting $\lambda ^ { - 1 } \log ( b / \eta )$ , with � the source-GF angle rate rather than the � of Theorem 1, leaves 75.8, 78.2, 72.0, 76.9 and $7 3 . 7 .$ The median checkpoint angle decays at rate 0.050 for every $\eta ,$ and the recovered fractions for all step sizes fall close to one curve in $\Gamma _ { T } = b \delta _ { T } / \eta$ . For every �, the runs that recover at $T = 4 0 0$ include the near-opposite initializations of Section 8.

$$
\begin{array} { r l r l r l } { - \vartheta - \eta = 0 . 2 5 \times 1 0 ^ { - 3 } } & { - \vartheta - \eta = 0 . 5 \times 1 0 ^ { - 3 } } & { -- \vartheta - \eta = 1 \times 1 0 ^ { - 3 } } & { - \phi - \eta = 2 \times 1 0 ^ { - 3 } } & { \longrightarrow - \eta = 4 \times 1 0 ^ { - 3 } } & { -- \eta = 4 \times 1 0 ^ { - 3 } } & { -- \mathrm { P o i s s o n ~ r e f e r e n c e } } \end{array}
$$

![](images/f0208fe3bd0c2f176371abf77b90f494d045542f00492e0602a262d69aac86ee.jpg)

(b) Recovery against disagreement budget  
![](images/a3d3cc1c82189ab25e8e8c3f10dc7ebf5153d80dfe281f9eabc73e8ec8bf5cea.jpg)  
Figure 4 | Joint dependence on source time and step size. (a) Warm recovery declines with �, later for smaller � (light to dark: increasing �). (b) Binned by $\Gamma _ { T } = b \delta _ { T } / \eta _ { \mathrm { m } }$ , all step sizes fall near the Poisson reference (dashed); bins of width 0.5 in $\log _ { 1 0 } \Gamma _ { T }$ with at least 8 records.

## K. Weight-Decay and Perturbation Controls

Every run in Figure 5 starts from the balanced checkpoint of Appendix J with amplitude 0.9, the source optimum for $\lambda = 0 . 0 5 ,$ , and uses the same input streams in every setting; only the target phase changes. Disagreements are checked every update and risk every ten updates (0.1 physical-time units). Weight-decay runs stop at recovery; noise runs continue to $H = 1 0 0 0$

Weight decay. The common-gate linearization at a balanced target clone contracts the transverse splitting mode at rate � (Proposition 4). After the amplitude transient toward $B _ { t } = 2 ( 2 / \pi - \lambda )$ we therefore use $\delta ( t ) \approx 0 . 8 5 \delta _ { 0 } e ^ { - \lambda t }$ where 0.85 approximates the norm-adjustment factors $\sqrt { 0 . 9 / B _ { t } } \in [ 0 . 8 4 , 0 . 8 8 ]$ ]. The cumulative Poisson intensity of this heuristic reference is

$$
\Lambda ( t ) = \frac { 0 . 8 5 b \delta _ { 0 } } { \pi \eta } \cdot \frac { 1 - e ^ { - \lambda t } } { \lambda } ;
$$

the reference curve is $1 - e ^ { - \Lambda ( t ) }$ , with $\Lambda ( t ) = 0 . 8 5 b \delta _ { 0 } t / ( \pi \eta ) \mathrm { a t } \lambda = 0 .$ . For $\delta _ { 0 } = 1 0 ^ { - 5 }$ , the observed recovery fractions plateau at 0.29 and 0.04 for $\lambda = 0 . 0 1$ and 0.05 (predicted 0.35 and 0.08), with no first disagreement observed after $t \approx 2 4 0$ through $H = 1 0 0 0$ . These input streams are independent of those in Appendix J, where the same setting gives 0.078 by $t = 4 5 0$ against a reference without the factor 0.85 (0.097). For $\lambda = 0$ , after the amplitude transient the angle of unrecovered runs decays only slowly, from about 0.82� at $t = 1 0 0$ to about $0 . 7 0 \delta _ { 0 }$ at $H = 1 0 0 0 ;$ the cumulative disagreement intensity implied by these angles keeps growing nearly linearly, as in the reference Λ(�), and the recovery fraction reaches 0.98 and 0.34 at $H = 1 0 0 0$ for $\delta _ { 0 } = 1 0 ^ { - 5 }$ and $1 0 ^ { - 6 }$ , close to the reference values 0.99 and 0.35. In these runs recovery always follows a sampled disagreement. Population GD reaches risk 0.4 by $t \approx 2 6$ in every setting.

Parameter noise. After every SGD step, from the checkpoint with $\delta _ { 0 } = 1 0 ^ { - 5 }$ and target $\lambda = 0 . 0 5$ , the full parameter vector of unit � receives $s \sqrt { \eta } \xi _ { i } ,$ with $\xi _ { i } \sim N ( 0 , I _ { 3 } )$ fresh each step and independent of the input batches. The two units receive independent vectors or the same vector. Before escape, a local approximation models the transverse diference under independent noise as an Ornstein–Uhlenbeck process with stationary standard deviation $s / \sqrt \lambda$ , so the angle fluctuates on the scale $s / ( r { \sqrt \lambda } ) \approx 5$ .8� for hidden norm $r \approx 0 . 7 7 ;$ median angles among unrecovered runs are of this order. In this approximation the disagreement probability per step no longer decays, so the cumulative disagreement intensity grows roughly linearly in time. Recovery by $H = 1 0 0 0$ rises from 0.04 without noise to 0.23, 0.89 and 1.00 for $\overline { { s } } = 1 0 ^ { - 7 } , \bar { 1 } 0 ^ { - 6 } , 1 0 ^ { - 5 }$ and continues to increase in �, whereas shared noise of the same scales gives 0.04, 0.05 and 0.10. Shared noise also moves the common trajectory and the gate geometry, but it does not act directly on the diference between the units. This is consistent with the escape route through symmetry-breaking noise noted by Joudaki et al. (2026).

(a) Weight decay, δ<sub>0</sub> = 10<sup>−5</sup>  
![](images/5209e0f6cdf931f04b16f10e202b0605364a2535e08f25eb77a9b549f8ff9686.jpg)

(b) Weight decay, δ<sub>0</sub> = 10<sup>−6</sup>  
![](images/14f9e432ba272088f849934de95cc8b4b77f23d5a1aa5116864b71fdefa595b3.jpg)

(c) Parameter noise, $\delta _ { 0 } = 1 0 ^ { - 5 }$  
![](images/615452b355bfd2d33f9f1aa4fbe148fdb38f8ea2696eb972ec2e145158b6d7e1.jpg)  
Figure 5 | Weight-decay and parameter-noise controls from constructed checkpoints $( b = 1 6 , \eta = 0 . 0 1$ , 128 runs per setting). (a,b) With target weight decay, recovery (solid) plateaus at or below the Poisson reference (dashed); without it, it keeps growing. Dotted: fraction that has sampled a disagreement. (c) Independent parameter noise raises recovery with � (solid); shared noise barely does (dash-dotted); $\lambda = 0 . 0 5 , \delta _ { 0 } = 1 0 ^ { - 5 }$ ; black: no noise.

## L. Notation

Symbols are listed with the place where they are defined. � and � denote positive constants whose values may change between occurrences; local symbols are marked with the appendices to which they are confined.

## Model and training

<table><tr><td>Symbol</td><td>Meaning</td><td>Where</td></tr><tr><td> $X , R , { \hat { X } }$ </td><td>input, uniform on  $\{ \| x \| \leq 2 \} ; X = R \hat { X } \mathrm { ~ w i t h ~ } \mathbb { E } R ^ { 2 } = 2$ </td><td>Sec.3</td></tr><tr><td> $\theta = \left( a _ { 1 } , \boldsymbol { w } _ { 1 } , \boldsymbol { a } _ { 2 } , \boldsymbol { w } _ { 2 } \right)$ </td><td>all six parameters;  $\begin{array} { r } { f _ { \theta } ( x ) = \sum _ { i } a _ { i } \sigma ( w _ { i } ^ { \top } x ) , \sigma ( z ) = \mathrm { m a x } \{ z , 0 \} } \end{array}$ </td><td>Sec.3</td></tr><tr><td> $y _ { s } , y _ { t }$ </td><td>source label σ(x1), target label ||x||</td><td>Sec. 3</td></tr><tr><td> $R _ { y } , F _ { y } , \ell _ { y }$ </td><td>population risk, objective  $R _ { y } + \frac { \lambda } { 2 } \left. \theta \right. ^ { 2 } ;$  ,sample loss; subscripts  $s , t$  for source and target</td><td>Sec. 3</td></tr><tr><td>λ</td><td>weight decay, any fixed  $\lambda \in ( 0 , 1 / 5 ]$  in the theory (App. D); λ = 1/20 in numerical values and simulations</td><td>Sec.3</td></tr><tr><td> $b , \eta , n , N$ </td><td>batch size, step size, update index, number of updates; physical time is nη</td><td>Sec.3</td></tr><tr><td> $\mathrm { G F } , \mathrm { G D }$ </td><td>gradient flow  $\dot { \theta } = - \nabla F _ { y } ; { \mathrm { G D } } =$  its Euler steps with step η (simulations) updates), target horizon; both physical</td><td>Secs. 3,8</td></tr><tr><td> $T , H$ </td><td>source training time  $\left( \lfloor { T } / { \eta } \right]$  times</td><td>Sec. 3</td></tr><tr><td> $\tau _ { f } , \tau _ { w } ( T ) , \tau _ { \mathrm { G F } } ( T )$ </td><td>first target time with  $R _ { t } \leq 0 . 4 \colon$  fresh SGD, warm SGD, GF in both phases</td><td>Sec. 3</td></tr></table>

## Geometry, initialization and risks

<table><tr><td>Symbol</td><td>Meaning</td><td>Where</td></tr><tr><td> $r _ { i } , \omega _ { i } , B$ </td><td> $r _ { i } = \| \boldsymbol { w } _ { i } \| , \omega _ { i } = a _ { i } r _ { i } ,$  total amplitude  $B = \omega _ { 1 } + \omega _ { 2 }$ </td><td>Sec.3</td></tr><tr><td> $\vartheta _ { i } , \delta$ </td><td>angle of w from e1; angle between  $w _ { 1 }$  and w2; the gates disagree with probability  $\delta / \pi$ </td><td>Secs.3,4</td></tr><tr><td> $\mathcal { D } , \mathcal { D } _ { 0 } , K$ </td><td>positive heads,  $0 < \omega _ { i } < 1$  , hidden vectors on opposite sides of e1 at angle below π (and the neuron exchange); D ∩ {B &lt; 3/4}; explicit compact subset with  $\mathbb { P } ( K ) > 0 . 0 1 0 6 4$ </td><td>Sec. 3, App. I</td></tr><tr><td> $q , A , s _ { * }$   $B _ { s } , B _ { t }$ </td><td> $q = 2 / \pi ; A = q - \lambda ( \approx 0 . 5 8 7 \mathrm { a t } \lambda = 1 / 2 0 ) ; s _ { \ast } = \sqrt { A }$  source and target clone amplitudes,  $1 - 2 \lambda$  and  $2 ( q - \lambda ) \ ( 0 . 9 $  and</td><td>Secs. 3, 4</td></tr><tr><td></td><td> $\approx 1 . 1 7 3 \ \mathrm { a t } \ \lambda = \bar { 1 } / 2 0 )$ </td><td>Sec. 4</td></tr><tr><td> $R _ { \mathrm { c o l } } , R _ { \mathrm { g o o d } }$ </td><td> $1 - q ^ { 2 } \approx 0 . 5 9 4 7 ,$  lower bound for proportional units;  $1 - 2 q ^ { 2 } + 2 \lambda ^ { 2 }$   $( \approx 0 . 1 9 4 4 \mathrm { a t } \lambda = 1 / 2 0 )$  , limit of target GF from K</td><td>Sec.3</td></tr><tr><td> $d , d _ { T } , m _ { i }$ </td><td>projective separation  $\| w _ { 1 } / a _ { 1 } - w _ { 2 } / a _ { 2 } \| .$  its value at the source checkpoint;  $m _ { i } = { r _ { i } } / { a _ { i } } ; { }$  with  $d ^ { 2 } = ( m _ { 1 } - m _ { 2 } ) ^ { 2 }$  + 2m1m2(1 − cos δ) and  $d = 2 \sin ( \delta / 2 )$  when balanced</td><td>Secs. 1, 3, 6</td></tr><tr><td> $\Gamma , \Gamma _ { T } , \delta _ { 0 }$ </td><td>disagreement budget  $b \delta / \eta ;$  its value  $b \delta _ { T } / \eta$  after source time T; angle at Secs. 1, 8 the task switch</td><td></td></tr><tr><td> $\delta _ { s } ( T ) , C _ { s }$ </td><td>source-GF angle at time T and its coefficient,  $\delta _ { s } ( T ) = C _ { s } e ^ { - \lambda T } ( 1 + o ( 1 ) )$ </td><td>Sec. 4, App. H</td></tr><tr><td> $\delta _ { T } , \theta _ { s } ^ { \mathrm { G F } } ( T )$ </td><td>angle at a source checkpoint at time  $T ;$  source-GF checkpoint CH</td><td>Secs. 5, 8, App. J</td></tr><tr><td> $c _ { T } , c _ { H }$ </td><td>exponents in  $T = c _ { T }$  log(  $1 / \eta )$  and  $H = \eta ^ { - }$ </td><td>Sec. 5</td></tr><tr><td> $T _ { 1 } , \eta _ { 1 } , c _ { p }$ </td><td>constants of Corollary 5;  $c _ { p } = p / ( 2 C )$  in the necessary sizes</td><td>Sec. 5</td></tr></table>

## Source SGD (Appendix B)

<table><tr><td>Symbol Meaning</td><td></td><td>Where</td></tr><tr><td> $u _ { i } , \ell _ { a }$ </td><td> $u _ { i } = w _ { i } / a _ { i } , s o d = \| u _ { 1 } - u _ { 2 } \| ;$  source allocation  $\log ( a _ { 1 } / a _ { 2 } )$ </td><td> $\mathsf { S e c . 6 , A p p . B }$ </td></tr><tr><td>γ  $r _ { 0 } , \mathcal { G }$ </td><td>expected contraction rate of d, any value below  $\lambda ; \gamma \ge \lambda / 2$  is used radius of the source neighborhood in the one-step estimate; event that App. B.1</td><td>Sec. 3, App. B.1</td></tr><tr><td></td><td>a sample&#x27;s gates disagree  $( { \mathcal { G } } _ { n }$  for a batch in Sec. 7.4) </td><td></td></tr><tr><td> $\varsigma , \Phi _ { \nu } , \mathcal { H } , \varepsilon$ </td><td> $\varsigma = 1 - \eta \lambda ;$  projective map with gates anchored at unit  ${ 2 ; }$  term  $\mathcal { H } ( \nu , u ) = ( \nu ^ { \top } u ) I + u \nu ^ { \top } .$  ; error matrix in (17)</td><td>its first-order App. B.1</td></tr><tr><td> $m , r$ </td><td>allocation margin; radius of the inner normal tube</td><td>App. B.2</td></tr><tr><td> ${ t _ { \mathrm { b l k } } , \delta _ { \mathrm { b u r n } } , c _ { \mathrm { b u r n } } }$ </td><td>source block duration; separation reached after burn-in; exponent of the burn-in failure probability</td><td>Apps. B.2, B.3</td></tr><tr><td> $\varepsilon , L , E _ { T }$ </td><td>allowance for unfavorable allocation drift; burn-in time; good source event</td><td>Thm. 1, Sec.  $^ { 6 , }$  App. B.3</td></tr></table>

## Mechanism and proof overview (Sections 4–6)

<table><tr><td>Symbol</td><td>Meaning</td><td>Where</td></tr><tr><td> $\varphi , V , V _ { \mathrm { c o m } } , \nu , \nu _ { \mathrm { c o m } }$ </td><td>transverse splitting at the target clone; population field  ${ \boldsymbol { V } } = - { \boldsymbol { \nabla } } F _ { t }$  and common-gate mean field  $V _ { \mathrm { { c o m } } ; }$  half the differences of their second</td><td>Sec. 4</td></tr><tr><td> $U , u$ </td><td>hidden coordinates (v is also the batch vector in App. B.1, Sec. 7.1)  $U = a _ { 1 } w _ { 1 } + a _ { 2 } w _ { 2 } ;$  unit hidden direction  $u = w / r$ </td><td>Secs. 4, 7.1</td></tr><tr><td> $M ( u ) , P _ { \nu }$ </td><td> $\left( \begin{array} { l l } { 0 } & { u ^ { \top } } \\ { u } & { 0 } \end{array} \right)$  ; orthogonal projection onto  $\nu ^ { \bot }$ </td><td>Secs. 7.1, 7.2</td></tr><tr><td> $\boldsymbol { e } = ( e _ { a } , e _ { w } ) , \mathcal { F } _ { n }$ </td><td>head and hidden parts of  $e ;$  σ-field of the first n batches</td><td>Secs. 6, 7.2</td></tr></table>

Target SGD (Section 7)
<table><tr><td>Symbol</td><td>Meaning</td><td>Where</td></tr><tr><td> $z _ { i } , h , w , r , d _ { z }$ </td><td> $z _ { i } = ( a _ { i } , w _ { i } ) ; h = ( z _ { 1 } + z _ { 2 } ) / \sqrt { 2 } = ( h _ { a } , w ) , r = | | w | | ; d _ { z } = ( z _ { 1 } - z _ { 2 } ) / \sqrt { 2 }$  (distinct from d). Alone, r is also the tube radius in App. B.2 and a hidden norm in App. K</td><td>Sec. 7</td></tr><tr><td> $\rho , e , \Delta , \Delta _ { 0 }$ </td><td>target allocation  $h ^ { \top } d _ { z } / \Vert h \Vert ^ { 2 } ~ ( = \operatorname { t a n h } ( \ell _ { a } / 2 )$  on clones; thresholds  $\rho _ { 0 } < \rho _ { 1 } < \rho _ { 2 }$  in Proposition  $6 ) ; e = d _ { z } - \rho h ; \Delta = \| e \| ;$  its initial value</td><td>Sec. 7, App. D; ∆ also Sec. 6</td></tr><tr><td> $S , k , B$ </td><td> $S = 1 + \rho ^ { 2 } , k = h _ { a } ^ { 2 } - r ^ { 2 } , B = S h _ { a } r$  (equal to  $\omega _ { 1 } + \omega _ { 2 }$  on the proportional family)</td><td>Sec. 7</td></tr><tr><td> $G , Q , \tilde { \theta } _ { n }$ </td><td>common-gate random matrix,  $Q = I + \eta G$  , auxiliary process  $z _ { i } ^ { + } = Q z _ { i }$  (frozen after the stop)</td><td>Secs. 7.1, 7.4</td></tr><tr><td> $\nu , \epsilon _ { 0 }$ </td><td>normal and projective entry tolerances</td><td>Proposition 6</td></tr><tr><td> $\kappa$   $\zeta _ { t } , n _ { \mathrm { d i s } } , \epsilon$ </td><td>decay rate of the Lyapunov functions  $W _ { s } , W _ { t }$  along the population flow tube exit time; first batch containing a disagreement; projective radius</td><td>Apps. B.2, C</td></tr><tr><td></td><td>of the tube (distinct from ε)</td><td>Secs. 7.2, 7.4</td></tr></table>

## Constants and population analysis

<table><tr><td>Symbol</td><td>Meaning</td><td>Where</td></tr><tr><td> $c _ { * } , \eta _ { 0 } , t _ { 0 }$ </td><td>confinement rate (at most the burn-in, source, target and fresh-tracking rates), step-size cutoff, fresh hitting-time bound in Thm. 1</td><td>Thm. 1, App. D</td></tr><tr><td> $\bar { \eta } , \eta _ { \mathrm { t } } , \ell _ { \mathrm { b l k } }$ </td><td>step-size thresholds of Lemmas  $^ { 7 - 8 }$  and of Proposition  $6 ;$  block duration in Lemma 8</td><td> $\mathrm { A p p . } \mathrm { A } , \mathrm { S e c . } \ 7$ </td></tr><tr><td> $C _ { 0 } , c _ { 0 } , c _ { 1 } , t _ { 1 } , \eta _ { 2 } , \varrho$ </td><td>constants of Corollaries 2 and  $3 ~ ( C _ { 0 }$  is reused in App. H); ρ is the level Sec. 3, App. A in Lemma 8</td><td></td></tr><tr><td> $J , g$ </td><td>ReLU kernel  $J ( t ) = ( \sin t + ( \pi - t ) \cos t ) / ( 2 \pi )$  and  $g = - J ^ { \prime }$ </td><td>App. E; g also Sec. 4</td></tr><tr><td> $W _ { s } , W _ { t } , R _ { W } , \ell _ { 0 }$ </td><td>Lyapunov functions for the source and target normal coordinates; target tube level; duration of the target amplitude bridge</td><td>Apps. B.2, C</td></tr><tr><td> $\mathcal { A } \Subset \mathcal { A } ^ { \prime } , \zeta _ { s } , \zeta ^ { \prime }$ </td><td>source allocation intervals; source stopping time; exit time from the constraint set in Lemma 8</td><td>Apps. B.2, A</td></tr><tr><td> $\alpha , \beta , q _ { i } , h _ { i } , k _ { i } , s _ { i }$ </td><td> $\alpha = \vartheta _ { 1 } , \beta = - \vartheta _ { 2 } ;$  local to Apps. E-H: qi the factor in  $\dot { a } _ { i } = r _ { i } q _ { i } - \lambda a _ { i } ,$   $h _ { i } = a _ { i } / r _ { i } , k _ { i } = a _ { i } ^ { 2 } - r _ { i } ^ { 2 } , s _ { i } = a _ { i } + r _ { i }$ </td><td>App. E</td></tr><tr><td> $\delta _ { \star } , D ( \delta ) , m$ </td><td>fixed small target angle for the escape time;  $D ( \delta ) = 1 / 2 - J ( \delta )$  ; lower amplitude bound (unrelated to  $m _ { i }$  in Sec. 6)</td><td>App. H</td></tr><tr><td> $m , M$ </td><td>margins  $1 0 ^ { - 4 }$  and 5 in the definition of K</td><td>App. I</td></tr></table>

## Experiments

<table><tr><td>Symbol</td><td>Meaning</td><td>Where</td></tr><tr><td> $\kappa _ { \mathrm { e f f } }$ </td><td>effective-intensity ratio −πλ log(1 − p)/Γ</td><td>App. J</td></tr><tr><td> $T _ { 1 / 2 } ( \eta )$ </td><td>source time at which half of the warm runs recover</td><td>App. J</td></tr><tr><td> $s , \Lambda ( t )$ </td><td>parameter-noise scale; cumulative Poisson intensity of the reference</td><td>App. K</td></tr></table>