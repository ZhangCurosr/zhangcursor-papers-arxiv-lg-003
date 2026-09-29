# Structured Neural SDEs for Functional Calibration

Francesco Piatti<sup>1,†</sup>, Andrea Iannucci<sup>1</sup>, and Thomas Cass<sup>1</sup>

<sup>1</sup>Department of Mathematics, Imperial College London <sup>†</sup>Corresponding author, email: francesco.piatti19@imperial.ac.uk

## Abstract

Neural Stochastic Diferential Equations (Neural SDEs) provide flexible continuous-time generative models, but generic neural drift and difusion networks are costly to simulate on long horizons and can give unstable gradients when the training signal is a path functional rather than a pointwise observation. We introduce SLiSDE, a family of Neural SDE models built from structured linear stochastic layers. Parallel-in-time simulation is obtained at the layer level, while expressivity is recovered by gated in-flow stacking: previous-layer paths modulate the next layer’s latent flow through learned gates. For functional calibration tasks in which rare paths dominate the loss, we add an optional Girsanov tilt that acts as a learned importance sampler with an exact likelihood-ratio correction. We prove well-posedness, a discretisation error bound, validity of the change of measure, and a universality result: the terminal laws of the gated stack are dense in the space of squareintegrable laws. Experiments on functional calibration benchmarks show that the structured model outperforms fully neural SDE baselines while retaining parallel-time simulation and stable importance weights.

## 1 Introduction

Many quantities of practical interest are expectations of functionals of a stochastic path whose trajectories are never observed. In derivatives markets the available data are option quotes – expectations of terminal payofs – and exceedance probabilities, and calibrating a model of the underlying dynamics to such quotes is the entry point of pricing, hedging and risk management (Buehler et al., 2019); the same situation, dynamics observed only through a finite family of statistics, arises in the physical sciences. We refer to this task as functional calibration: fitting a generative model of continuous-time stochastic dynamics to prescribed expectations of path functionals. Training such a model requires simulating it afresh at every gradient step, so the computational cost of simulation is the binding constraint, and it is the constraint that shapes the model proposed here.

Neural diferential equations provide a flexible framework for learning continuous-time dynamics by parameterizing their vector fields with neural networks. Neural ODEs (Chen et al., 2018) model deterministic latent evolution, neural CDEs (Kidger et al., 2020; Walker et al., 2025) condition the dynamics on an observed driving path, and neural SDEs (Tzen and Raginsky, 2019) provide the stochastic analogue, modeling distributions on continuous-time paths via stochastic diferential equations, which may be interpreted in either the Itˆo or Stratonovich sense.

In this work, we focus on the Itˆo setting, where the latent state evolves as

$$
\mathrm { d } Z _ { t } = \mu _ { \theta } ( t , Z _ { t } ) \mathrm { ~ d } t + \sigma _ { \theta } ( t , Z _ { t } ) \mathrm { ~ d } W _ { t } , \qquad Z _ { 0 } \sim p _ { 0 } ,\tag{1}
$$

where $W _ { t }$ is a Brownian motion and $\mu _ { \theta }$ , σ unconstrained neural networks. This is a rich model class (Tzen and Raginsky, 2019; Liu et al., 2019; Li et al., 2020; Kidger et al., 2021), and in practice it is often used as a Markovian lift: the latent state is lifted to a higher-dimensional space in which a non-Markovian observed process can be represented as a Markov difusion. The viewpoint is powerful, but it makes simulation expensive — every time step requires fresh nonlinear evaluations of the lifted vector field, and the recurrence is inherently sequential unless additional structure is imposed.

The central design principle of this paper is to move the non-linear part of the model from the non-linear drift and difusion evaluated at every time step to non-linear coupling between structured linear SDE layers. After discretisation, each layer is an afine recurrence whose composition is associative, so it can be evaluated by the parallel associative scan of Section 2.1, which cuts the sequential simulation cost over a horizon of T time steps from O(T) to O(log T). To recover expressivity, we stack such layers through gated in-flow coupling: the previous layer path modulates the next layer’s transition and ofset through learned gates. The current layer remains afine in its own state, so the scan structure is preserved, while the gates make the stack a genuinely nonlinear model, universal at the level of terminal laws (Theorem 5.3).

For high-variance path-dependent objectives, we add an optional last-layer-only Girsanov tilt. A learned adapted controller changes only the Brownian increments driving the final layer and returns the corresponding Radon– Nikodym log-weight in closed form.

## Related work.

Within the neural-SDE literature two training paradigms coexist. Generative training matches the law of the model to an empirical distribution of paths through a statistical divergence – the KL divergence of variational inference (Tzen and Raginsky, 2019; Li et al., 2020), the Wasserstein-1 distance of the GAN formulation (Kidger et al., 2021), or signature-kernel scores (Issa et al., 2023) – and requires sample paths. Calibration training, our setting, instead matches prescribed expectations of functionals, typically derivative prices, when no trajectory is observed (Gierjatowicz et al., 2020; Cuchiero et al., 2020; Cohen et al., 2021); before neural SDEs this task was handled by parametric families that hard-wire the dynamics, such as stochastic-volatility and local-volatility models in finance or mechanistic difusion models in the physical sciences.

Structured state-space models have shown that carefully parameterised linear recurrences can achieve strong sequence-modelling performance while remaining hardware eficient (Gu et al., 2022; Smith et al., 2023; Hasani et al., 2022; Gu and Dao, 2023). Recent work also connects this line of models to controlled diferential equations, viewing sequence transformations as dynamics driven by an input path (Muca Cirone et al., 2024). SLiCE (Walker et al., 2025) develops this perspective for structured linear controlled diferential equations, showing how scan-compatible linear dynamics can be embedded in a CDE-style framework.

The Girsanov component is connected to importance sampling for difusion processes and to the stochastic-control view of variance reduction (Hartmann et al., 2017; Zhang and Chen, 2022; Hartmann and Richter, 2024). In those approaches, one learns or designs a drift change that makes rare paths less rare and then corrects the resulting bias with a likelihood ratio. Our tilt is narrower: it acts only on the Brownian motion of the last SLiSDE layer. This keeps the prefix computation unchanged, and makes the proposal lightweight enough to train jointly with the backbone.

## Contributions.

• Structured linear neural SDE for functional calibration. We introduce a structured linear Neural SDE backbone whose discretized layers are afine recurrences. This gives O(log T) parallel depth in sequence length by the parallel associative scan, compared with the O(T) complexity of standard neural SDE solvers.

• Gated in-flow stacking. We propose a gated in-flow stacking rule in which previous-layer paths modulate the next layer’s afine transition and ofset. Nonlinearity enters between layers only, so each layer stays afine in its

own state and scan-compatible.

• Girsanov overlay. We add an optional last-layer-only Girsanov tilt with a closed-form likelihood-ratio correc tion: a learned adapted controller acts as a self-normalised importance sampler that improves performance on rare-event functional calibration with little added compute.

• Theory. We prove that the terminal laws generated by the gated stack are dense in $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { p } )$ for the 2-Wasserstein distance (Theorem 5.3), so the structural restrictions cost nothing at the level of terminal laws.

• Experiments. We evaluate the model on nonlinear path-functional calibration tasks and real option-surface benchmarks, comparing against neural SDE and structured sequence-model baselines. SLiSDE attains the lowest or a statistically tied calibration loss on every benchmark, and trains faster than SLiCE in every setting tested and faster than the Neural SDE in all but two of them; a controlled study shows the tilt as a learned importance sampler for rare-event functionals.

## 2 Mathematical Background

Let $( \Omega , \mathcal { F } , \{ \mathcal { F } _ { t } \} _ { t \in [ 0 , T ] } , \mathbb { P } )$ be a filtered probability space supporting an m-dimensional Brownian motion $W _ { t } \ =$ $( W _ { t } ^ { 1 } , \dots , W _ { t } ^ { m } )$ on [0, T]. We consider a time grid $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { N } = T$ , with Brownian increments $\Delta W _ { k } = W _ { t _ { k + 1 } } - W _ { t _ { k } } \sim { \mathcal { N } } ( 0 , \Delta t _ { k } I _ { m } )$ . Throughout, complexities are stated in terms of the horizon T, which for a fixed step size is proportional to the number of steps N. The Girsanov change of measure on which the tilt of Section 4 rests is recalled in Appendix A.1.

## 2.1 Linear SDEs and afine flows

A general afine linear SDE takes the form

$$
\mathrm { d } Z _ { t } = ( A Z _ { t } + b ) \mathrm { d } t + \sum _ { j = 1 } ^ { m } C ^ { j } Z _ { t } \mathrm { d } W _ { t } ^ { j } + D \mathrm { d } W _ { t } , Z _ { 0 } = z _ { 0 }\tag{2}
$$

with $z _ { 0 } \in L ^ { 2 } ( \Omega , \mathcal { F } _ { 0 } , \mathbb { P } ; \mathbb { R } ^ { d } ) , A , C ^ { j } \in \mathbb { R } ^ { d \times d } , b \in \mathbb { R } ^ { d }$ , and $D \in \mathbb { R } ^ { d \times m }$ . On each grid interval $[ t _ { k } , t _ { k + 1 } ]$ its discretisation, by the matrix-exponential transition or by the Euler–Maruyama transition (Appendix C.1; the latter is used throughout our experiments), is an afine recurrence

$$
Z _ { k + 1 } = F _ { k } Z _ { k } + g _ { k } ,\tag{3}
$$

where the transition matrix $F _ { k } \in \mathbb { R } ^ { d \times d }$ and the ofset $g _ { k } \in \mathbb { R } ^ { d }$ are built from the coeficients and the Brownian increment $\Delta W _ { k }$

Parallel associative scan. The solution of the afine recurrence Eq. 3 can be written $Z _ { k } = \widehat { F } _ { k } Z _ { 0 } + \widehat { g } _ { k }$ with $( \widehat { F } _ { k } , \widehat { g } _ { k } ) = ( F _ { k - 1 } , g _ { k - 1 } ) \circ \cdot \cdot \cdot \circ ( F _ { 0 } , g _ { 0 } )$ , where

$$
( F ^ { \prime } , g ^ { \prime } ) \circ ( F , g ) : = ( F ^ { \prime } F , \ F ^ { \prime } g + g ^ { \prime } )\tag{4}
$$

is composition of afine maps. This operation is associative, so all prefixes $( \widehat { F } _ { k } , \widehat { g } _ { k } ) _ { k \le N }$ can be computed with any bracketing, in particular along a balanced binary tree of depth O(log T), and the result equals the sequentia recursion exactly (Proposition B.1). We call this evaluation the parallel associative scan and say that a recurrence is scan-compatible when its one-step maps are afine in the state; this is the only name used for the scan in the

sequel.

Remark 2.1. The scan does not reduce the total arithmetic work across all time steps, but it reduces the sequential dependence from O(T) recurrent updates to O(log T) parallel depth, exposing the time dimension to parallel hardware.

## 2.2 Functional calibration objective

Let $Y ^ { \theta }$ denote the output path of a model with parameters θ. The data of a functional calibration problem are a finite family Φ of path functionals $\varphi ,$ each with a prescribed value $c _ { \varphi }$ of its expectation; no trajectory is observed. Given n independent model paths $Y ^ { \theta , 1 } , \ldots , Y ^ { \theta , n }$ and the Monte-Carlo average ${ \widehat { \mathbb { E } } } _ { \theta , n } [ \varphi ] = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \varphi ( Y ^ { \theta , i } )$ , the objective is

$$
\mathcal { L } _ { n } ( \theta ) = \sum _ { \varphi \in \Phi } w _ { \varphi } \Big ( \widehat { \mathbb { E } } _ { \theta , n } [ \varphi ] - c _ { \varphi } \Big ) ^ { 2 } , \qquad w _ { \varphi } > 0 ,\tag{5}
$$

minimised by stochastic gradient descent with a fresh sample of n paths at every step. In this context no model path is paired with a data path: Eq. 5 is a functional of the model law rather than a samplewise error. Since $\Im [ ( \widehat { \mathbb { E } } _ { \theta , n } [ \varphi ] - c _ { \varphi } ) ^ { 2 } ] = ( \mathbb { E } _ { \theta } [ \varphi ] - c _ { \varphi } ) ^ { 2 } + n ^ { - 1 } \operatorname { V a r } _ { \theta } ( \varphi )$ , it converges almost surely to the population objective ${ \mathcal { L } } _ { \infty } ( \theta ) =$ $\begin{array} { r } { \sum _ { \varphi } w _ { \varphi } ( \mathbb { E } _ { \theta } [ \varphi ] - c _ { \varphi } ) ^ { 2 } } \end{array}$ , the Monte-Carlo variance entering at order $1 / n$ only, so there is no incentive to collapse the model law onto a mean. Finitely many expectation constraints do not determine the path law, and L∞ $\mathcal { L } _ { \infty }$ has no unique minimiser; this is a property of the task rather than of the objective, and the selected model is validated on functionals it was never fitted to. Where a reference dynamics is available, a relative-entropy penalty turns $\operatorname { E q . }$ 5 into a selection rule that drops into the training loop unchanged. Appendix A.2 develops these points and relates the objective to divergence-based generative training.

## 3 The SLiSDE Model

A SLiSDE is built by composing structured linear stochastic layers through a stacking mechanism, followed by a learned linear readout. The optional Girsanov component is described separately in Section 4.

## 3.1 Structured linear base layer

For latent dimension d and Brownian dimension m, each structured layer is an Itˆo SDE of the form

$$
\mathrm { d } Z _ { t } = \left( A _ { t } Z _ { t } + b _ { t } \right) \mathrm { d } t + \sum _ { j = 1 } ^ { m } \left( C _ { t } ^ { j } Z _ { t } + d _ { t } ^ { j } \right) \mathrm { d } W _ { t } ^ { j } , \quad Z _ { 0 } = z _ { 0 } ,\tag{6}
$$

where $A _ { t } , C _ { t } ^ { j } \in \mathbb { R } ^ { d \times d }$ and $b _ { t } , d _ { t } ^ { j } \in \mathbb { R } ^ { d }$ and we assume that $z _ { 0 } \in L ^ { 2 } ( \Omega , \mathcal { F } _ { 0 } , \mathbb { P } ; \mathbb { R } ^ { d } )$ . Under coeficient bounds that hold by construction (Appendix B.1), the layer has a unique strong solution with moment bounds independent of the time grid (Theorem 5.1). We write $( A _ { t } , b _ { t } , C _ { t } ^ { 1 } , \dots , C _ { t } ^ { m } , d _ { t } ^ { 1 } , \dots , d _ { t } ^ { m } )$ for the time-indexed coeficient family induced by the model parameters.

These parameters contain static structured coeficients $( A , b , C ^ { 1 } , \ldots , C ^ { m } , d ^ { 1 } , \ldots , d ^ { m } ) \in \Theta$ , where Θ denotes the set of trainable weights, together with optional time-feature decoder weights. The matrices A and $C ^ { j }$ are chosen from a fixed structured family: diagonal, block-diagonal, or dense. The additive difusion vectors $d ^ { j }$ may be (jointly) represented densely or through a low-rank parameterisation.

Remark 3.1 (Time-dependent coeficients). The map $A \mapsto A _ { t }$ , and similarly for $b , C ^ { j } , d ^ { j }$ , is optional and makes the coeficients time dependent through a small deterministic time-feature decoder (described in Appendix C.2). In both cases the layer remains linear in the current state $Z _ { t }$

Remark 3.2 (Noise type). The noise type may be controlled by switching of parts of the difusion. $I f d _ { t } ^ { j } = 0 f o r$ all j, the layer has only multiplicative noise; $i f C _ { t } ^ { j } = 0 f o r a l l j ,$ , the layer has only additive noise; and $i f$ both terms are retained, the layer has general afine difusion

After discretisation, the layer produces afine transition pairs $( F _ { k } , g _ { k } )$ of the form of Eq. 3, with the static coeficients of the transitions of Appendix C.1 replaced by their time-dependent values at $t _ { k }$

The choice of matrix structure is important computationally: diagonal, block-diagonal, and dense matrices are closed under matrix multiplication, so products of transition matrices remain in the same family. This closure is what allows the afine maps to be composed by the parallel associative scan with ${ \cal O } ( \log T )$ parallel depth without leaving the chosen representation. By contrast, other cheap matrix parameterization families such as fixed-rank low-rank or DPLR matrices do not satisfy this property, thus requiring sequential recursion and $O ( T )$ complexity.

## 3.2 Residual stacking (sequence-model baseline)

Sequence models such as Mamba (Gu and Dao, 2023) and the linear CDE SLiCE (Walker et al., 2025) build depth by residual stacking: a sequence block $\boldsymbol { \mathcal { S } ^ { ( \ell ) } }$ is applied to a normalised hidden path and added back, $H ^ { ( \ell ) } =$ $H ^ { ( \ell - 1 ) } + S ^ { ( \ell ) } ( \mathrm { N o r m } ( H ^ { ( \ell - 1 ) } ) )$ , so a single hidden path is refined layer by layer. This is the natural depth mechanism to compare against. It is, however, ill-suited to our setting: each SLiSDE layer is itself a Brownian-driven SDE, so reusing the previous layer’s already-Brownian path as the driver of the next would turn that layer into a deterministic transform of an already-stochastic input, adding compositional depth but injecting no fresh randomness per layer. We therefore do not stack SLiSDE layers residually; instead we couple them through the flow itself, via gated in-flow stacking (Section 3.3).

## 3.3 Gated in-flow stacking

Our main architectural contribution is gated in-flow stacking: a selective, path-dependent modulation of the next layer’s afine flow, in the spirit of selective state-space models but adapted to stochastic latent dynamics. Layer ℓ builds its afine transition pairs $( F _ { k } ^ { ( \ell ) } , g _ { k } ^ { ( \ell ) } ) , k = 0 , \dots , N - 1$ , from its own structured coeficients and Brownian increments. Before they are scanned, causal features $x _ { k } ^ { ( \ell - 1 ) }$ of the previous layer’s path (its normalised state and the time features) produce, through two gated linear units, a diagonal scale $\alpha _ { k } ^ { ( \ell ) } \in ( 1 - \varepsilon , 1 + \varepsilon ) ^ { d }$ and an ofset $o _ { k } ^ { ( \ell ) } \in \mathbb { R } ^ { d }$ ; the gate architecture, the construction of the gate input and alternative parameterisations are given in Appendix C.4.

The afine pair is then modified by scaling only the diagonal entries of the transition matrix and by adding the ofset correction,

$$
\bar { F } _ { k } ^ { ( \ell ) } = \mathcal { D } ( \alpha _ { k } ^ { ( \ell ) } , F _ { k } ^ { ( \ell ) } ) \qquad \mathrm { a n d } \qquad \bar { g } _ { k } ^ { ( \ell ) } = g _ { k } ^ { ( \ell ) } + o _ { k } ^ { ( \ell ) } ,\tag{7}
$$

where D rescales the diagonal of $F _ { k } ^ { ( \ell ) }$ entrywise by $\alpha _ { k } ^ { ( \ell ) }$ (Appendix C.4), and layer ℓ is evaluated by scanning the modified afine recurrence

$$
Z _ { k + 1 } ^ { ( \ell ) } = \bar { F } _ { k } ^ { ( \ell ) } Z _ { k } ^ { ( \ell ) } + \bar { g } _ { k } ^ { ( \ell ) } .\tag{8}
$$

The key point is that gated in-flow introduces nonlinearity through the dependence of $( \bar { F } _ { k } ^ { ( \ell ) } , \bar { g } _ { k } ^ { ( \ell ) } )$ on the previous path $Z ^ { ( \ell - 1 ) }$ , while preserving afine dependence on the current state $Z _ { k } ^ { ( \ell ) }$ ; the recurrence is therefore evaluated exactly by the parallel associative scan (Proposition B.1(b), Appendix B).

## 3.4 Linear decoder

After the final stacking layer, the model maps the last latent path to the observation space through a learned linear readout:

$$
Y _ { t } = \Pi _ { o } Z _ { t } ^ { ( L ) } , \qquad \Pi _ { o } \in \mathbb { R } ^ { p \times d } ,\tag{9}
$$

where $Z ^ { ( L ) }$ denotes the path produced by the final layer and $\Pi _ { o } \in \Theta$ . When the initial observation is prescribed, we apply an optional shift so that the generated path satisfies the required value at $t = 0$

## 3.5 Discretisation and scan

At each time step, every layer builds an afine pair $( F _ { k } , g _ { k } )$ by freezing the time-dependent coeficients of $\operatorname { E q . }$ 6 at $t _ { k }$ using either the matrix-exponential or the Euler–Maruyama transition of Appendix C.1. The previous-layer path then modifies the pairs to $( \bar { F } _ { k } ^ { ( \ell ) } , \bar { g } _ { k } ^ { ( \ell ) } )$ as in Eq. 7. Once the previous layers are fixed, each layer is therefore an afine recurrence in its own state, and the modified recurrence is evaluated, layer after layer, by the parallel associative scan of Section 2.1 (Proposition B.1); the strong error of the resulting layerwise Euler–Maruyama scheme is of orde $1 / 2$ for every fixed depth (Theorem B.1, Appendix B).

## 4 Last-Layer Girsanov Tilt

Calibration losses on rare events receive little signal from plain Monte-Carlo paths under P: an event of probability p estimated from n paths has relative standard error $\sqrt { ( 1 - p ) / ( p n ) }$ , so more paths help only as $n ^ { - 1 / 2 }$ . We therefore simulate under a tilted measure Q that visits the rare region more often and reweight by $\mathrm { { d } \mathbb { P } / \mathrm { { d } \mathbb { Q } ; } }$ for difusion paths Girsanov’s theorem (Appendix A.1) gives the ratio in closed form for an adapted drift shift, and choosing the shift as a learned controller yields an importance sampler that acts on the constant $1 / { \sqrt { p } }$ itself.

The shift is confined to the final layer, so that the prefix dynamics calibrated by the vanilla loss are not perturbed. The final layer is driven by a partially independent innovation ϵ, and only the innovation is tilted: under $\mathbb { Q }$ it is shifted by an adapted control $u _ { \theta , k } \in \mathbb { R } ^ { m }$ , a function of the innovation history and of stop-gradient features of the prefix latent path,

$$
\mathrm { d } W _ { k } ^ { ( L ) } = \rho \mathrm { d } W _ { k } ^ { \mathrm { p r e f i x } } + \sqrt { 1 - \rho ^ { 2 } } \mathrm { d } \epsilon _ { k } , \qquad \rho \in [ 0 , 1 ) ,\tag{10}
$$

$$
\mathrm { d } \epsilon _ { k } ^ { \mathbb { Q } } = \mathrm { d } \epsilon _ { k } ^ { \mathbb { P } } - u _ { \theta , k } \ \mathrm { d } t .\tag{11}
$$

In a gated stack the random initial latent $z _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } )$ carries a large share of the terminal variance, so the tilt also shifts the initial state, $z _ { 0 } = \mu _ { \theta } + \xi$ with $\xi \sim \mathcal { N } ( 0 , I _ { d } )$ and a learned $\mu _ { \theta } \in \mathbb { R } ^ { d }$ . The discrete log-likelihood ratio is then

$$
\log \frac { \mathrm { d } \mathbb { Q } } { \mathrm { d } \mathbb { P } } = \mu _ { \theta } ^ { \top } z _ { 0 } - \frac { 1 } { 2 } \| \mu _ { \theta } \| ^ { 2 } + \sum _ { k = 0 } ^ { N - 1 } \left( u _ { \theta , k } ^ { \top } \mathrm { d } \epsilon _ { k } ^ { \mathbb { P } } - \frac { 1 } { 2 } \| u _ { \theta , k } \| ^ { 2 } \mathrm { d } t \right) ,\tag{12}
$$

which is exact, normalised $( \mathbb { E } ^ { \mathbb { Q } } [ \mathrm { d } \mathbb { P } / \mathrm { d } \mathbb { Q } ] = 1 )$ and depends only on the control, the shift and the shifted innovation (Theorem 5.2). The tilt is a training-time device: the backbone minimises the calibration loss with the far-tail terms estimated on the tilted batch by self-normalised importance sampling, the controller is trained by the cross-entrop method under a wall on KL(Q∥P), and samples from the trained model are drawn under P without it. The rationale for the initial-state shift, the controller, the two objectives and the gradient convention they require are given in Appendix C.5.

## 5 Theoretical Results

This section states the guarantees behind the architecture; the assumptions in full, the auxiliary results and all proofs are collected in Appendix B. Three questions are answered here:

(i) whether the stacked model is a well-posed stochastic process with moment bounds that do not deteriorate with depth (Theorem 5.1);

(ii) whether the last-layer change of measure yields exact importance weights (Theorem 5.2);

(iii) what the class can represent (Theorem 5.3).

Two further facts are used throughout and are stated and proved in the appendix:

(iv) the parallel associative scan returns exactly the sequential iterates of every layer, gated or not, so that the depth-L stack is evaluated exactly in O(L log T) parallel depth (Proposition B.1);

(v) the layerwise Euler–Maruyama scheme, with the gates evaluated on the discrete previous layer, has strong order $1 / 2$ for every fixed depth (Theorem B.1).

Standing assumptions. The statements concern the model of Section 3 under three assumptions, stated in full in Appendix B.1. All of them hold by construction of the architecture; we list them only to fix what the theorems cover.

• Model class (Assumption B.1): structured afine layers with the time-dependent coeficients of Appendix C.2, gated in-flow stacking with gates that read the normalised previous-layer state and the time features, arbitrar depth and widths, a linear readout, and a square-integrable initial state whose law may be degenerate.

• Coeficients (Assumption B.2): the coeficients of every layer are bounded and regular in time. For the ungated base layer this is automatic, because the time features are smooth and the decoders are fixed networks; the gated layers inherit it from the regularity of the previous layer.

• Gates (Assumption B.3): the gates are causal and Lipschitz in their input path, the flow gate stays within a fixed distance of one, and the ofset gate is bounded for every fixed choice of its weights because its input is normalised.

The consequence that drives every constant is that the gated coeficients of a layer obey bounds that depend neither on the layer index nor on the depth of the stack.

Theorem 5.1 (Well-posedness). Let $q \geq 2$ and $\mathbb { E } \Vert z _ { 0 } \Vert ^ { q } < \infty$

(a) Single layer. Under Assumptions B.1 and B.2, the layer SDE Eq. 6 has a unique strong solution on [0, T], and

$$
\mathbb { E } \operatorname* { s u p } _ { 0 \leq t \leq T } \| Z _ { t } \| ^ { q } \leq K _ { q } \big ( 1 + \mathbb { E } \| z _ { 0 } \| ^ { q } \big ) ,
$$

where $K _ { q }$ depends only on $( a _ { * } , b _ { * } , c _ { * } , d _ { * } , m , T , q )$

(b) Gated stack. Under Assumptions B.1–B.3, the depth-L stack of Section 3.3 has a unique strong solution, and the bound in (a) holds for every layer $Z ^ { ( \ell ) } , \ell \ = \ 1 , \ldots , L .$ , with one constant $K _ { q } ,$ which depends on $( a _ { * } , b _ { * } , c _ { * } , d _ { * } , m , T , q , \varepsilon , \Lambda _ { o } )$ but neither on ℓ nor on L, for any correlation structure of the layer drivers.

Theorem 5.2 (Validity of the last-layer Girsanov tilt). Assume the last-layer controller $u _ { \theta , t }$ is progressively measurable with respect to the final-layer filtration and satisfies Novikov’s condition, and let $\mu _ { \theta } \in \mathbb { R } ^ { d }$ be a fixed initial-state shift, with $\mu _ { \theta } = 0$ unless $z _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } )$ . Then $\begin{array} { r } { \mathrm { d } \mathbb { Q } / \mathrm { d } \mathbb { P } = \exp \bigl ( \mu _ { \theta } ^ { \top } z _ { 0 } - \frac { 1 } { 2 } \| \mu _ { \theta } \| ^ { 2 } \bigr ) \mathcal { E } _ { T } ( u _ { \theta } ) } \end{array}$ , with ${ \mathcal { E } } _ { T }$ the stochastic exponential of Eq. 14, defines an equivalent measure $\mathbb { Q } \sim \mathbb { P }$ under which $z _ { 0 } \sim \mathcal { N } ( \mu _ { \theta } , I _ { d } )$ , the prefix driver keeps its law, and $\begin{array} { r } { \epsilon _ { t } ^ { \mathbb { Q } } = \epsilon _ { t } - \int _ { 0 } ^ { t } u _ { \theta , s } } \end{array}$ ds is Brownian, the three being independent. For any integrable functional G of the full SLiSDE output,

$$
\mathbb { E } ^ { \mathbb { P } } [ G ] = \mathbb { E } ^ { \mathbb { Q } } \left[ G \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } \right] .
$$

Moreover, the likelihood ratio depends only on the final-layer control, the initial-state shift and on the Brownian innovation being shifted.

Theorem 5.3 (Wasserstein density of terminal laws). Fix $T > 0 , m \geq 1$ and an output dimension $p \in \mathbb { N } _ { : }$ , and let the gated layers use the RMS normalisation with a strictly positive regulariser and a gate input containing the time coordinate, as in Assumption B.1. For every prescribed initial condition $z _ { 0 } \in L ^ { 2 } ( \Omega , \mathcal { F } _ { 0 } , \mathbb { P } ; \mathbb { R } ^ { d _ { 0 } } )$ , possibly having a degenerate law, the collection of terminal laws generated by the model is dense in $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { p } )$ with respect to the 2-Wasserstein distance. More precisely, if

$\mathfrak { L } _ { T } : = \{ \mathcal { L } ( Y _ { T } ^ { \theta } ) : \theta$ is an admissible model configuration	 ,

where θ ranges over the admissible configurations of all latent widths $d \geq d _ { 0 }$ , then

$$
\overline { { \mathfrak { L } _ { T } } } ^ { W _ { 2 } } = \mathcal { P } _ { 2 } ( \mathbb { R } ^ { p } ) .
$$

One ungated layer followed by one gated layer sufices.

Theorem 5.3 is what makes the structural restrictions harmless: every target law with finite second moment is approximated arbitrarily well by the terminal law of an admissible configuration, so the selection among the difusions consistent with the calibration constraints (Section 2.2) is never forced by a limitation of the class.

## 6 Experiments

We evaluate SLiSDE on three main benchmarks: a path-dependent toy dynamical system (toy) and two optionsurface datasets (dax and spx), where the toy benchmark tests whether a model can match nonlinear long horizon path statistics generated by a highly nonlinear stochastic dynamical system. Dataset construction, training protocols, and search grids are described in Appendix D. Additional results – ablations over the structural choices the timing study, distributional recovery on toy and the Girsanov tilt studies – are collected in Appendix E.

Models and baselines. We compare SLiSDE against two baselines: (i) a fully connected Neural SDE with neural drift and difusion, and (ii) SLiCE (Walker et al., 2025). SLiCE is a natural structured-sequence baseline because it is a scan-compatible linear CDE and it is more expressive than Mamba (Gu and Dao, 2023); in our experiments we drive it with Brownian motion to match the stochastic input used by SLiSDE. For all model families, we report results for both two-layer and three-layer configurations.

Complexities. Table 1 reports the recurrent and scan costs for the SLiSDE transition structures. The scan reduces sequential depth from $O ( T )$ to ${ \cal O } ( \log T )$ when afine composition is available. For a generic Neural SDE, no analogous parallel associative scan is available, so simulation remains sequential in time. Dense transitions remain computationally expensive because each scan composition requires dense matrix multiplication; this can ofset the practical gains from parallelisation, and we therefore exclude dense SLiSDE variants from the main experimental analysis.

<table><tr><td></td><td></td><td>DiagonalBlock-diagonal</td><td>Dense</td><td>|NeuralSDE</td></tr><tr><td>Recurrent cost</td><td> $O ( d T )$ </td><td> $O ( n _ { b } b _ { s } ^ { 2 } T )$ </td><td> $O ( d ^ { 2 } T )$ </td><td> $O ( d ^ { 2 } T )$ </td></tr><tr><td>Scan cost</td><td> $O ( d \log T )$ </td><td> $O ( n _ { b } b _ { s } ^ { 3 } \log T )$ </td><td> $O ( d ^ { 3 } \log T )$ </td><td></td></tr></table>

Table 1: Computational cost of composing afine transition maps over T time steps. Here d is the latent dimension, $b _ { s }$ is the block size, and $n _ { b } = d / b _ { s }$ is the number of blocks. Typically $n _ { b } \ll d .$

Implementation and Compute. All models and experiments are implemented in JAX. The code is available at https://anonymous.4open.science/r/SLiSDE-3B86/. All experiments were run on a single NVIDIA RTX 6000 Ada Generation GPU.

## 6.1 Path-dependent toy benchmark (TOY)

toy is a controlled path-functional benchmark generated from a highly nonlinear stochastic dynamical system on [0, 1], discretised with $N = 2 0 4 8$ time steps (see Appendix D). The task is to calibrate the model law to the functional targets of Appendix D – thresholded positive parts, running maximum, squared path average and threshold-crossing probabilities at five evaluation times – through the objective of Eq. 5. In this first experiment, all models are trained without the Girsanov tilt, so the comparison isolates the efect of the architecture and stacking mechanism.
<table><tr><td>Model</td><td>Params</td><td> $\mathrm { L o s s \ ( 1 0 ^ { - 4 } ) }$ </td><td>Barrier MAE  $( 1 0 ^ { - 2 } )$ </td><td>ms / epoch</td></tr><tr><td>Neural  $\mathrm { S D E } - L = 2$ </td><td>124K</td><td> $1 . 5 5 7 \pm 0 . 2 0 6$ </td><td> $1 . 0 5 8 \pm 0 . 1 1 0$ </td><td>91.7</td></tr><tr><td> $\mathrm { S L i C E } - L = 2$ </td><td>142K</td><td> $4 . 6 9 0 \pm 0 . 4 2 3$ </td><td> $2 . 1 6 8 \pm 0 . 1 0 2$ </td><td>171.8</td></tr><tr><td>SLiSDE gated in-flow – L = 2</td><td>103K</td><td> $\mathbf { 0 . 8 5 2 \pm 0 . 1 6 8 }$ </td><td> $\mathbf { 0 . 6 5 8 \pm 0 . 0 5 4 }$ </td><td>72.9</td></tr><tr><td> $\mathrm { N e u r a l ~ S D E } - L = 3$ </td><td>157K</td><td> $3 . 0 2 4 \pm 0 . 3 4 1$ </td><td> $1 . 6 3 7 \pm 0 . 1 2 0$ </td><td>110.4</td></tr><tr><td> ${ \mathrm { S L i C E } } - L = 3$ </td><td>213K</td><td> $5 . 0 0 6 \pm 0 . 9 1 4$ </td><td> $2 . 2 5 0 \pm 0 . 2 5 2$ </td><td>248.2</td></tr><tr><td>SLiSDE gated in-flow –  $L = 3$ </td><td>74K</td><td> $\mathbf { 0 . 6 5 9 \pm 0 . 0 5 9 }$ </td><td> $\mathbf { 0 . 6 7 2 \pm 0 . 0 5 7 }$ </td><td>68.9</td></tr></table>

Mean ± standard error over seven seeds per cell; best result per column in bold; the loss is ${ \mathcal { L } } _ { n } ( \theta )$ of Eq. 5 on a fresh Monte-Carlo sample.

Table 2: toy results without Girsanov tilt. L is the number of stacked layers; each row is the tuned cell of the search grid for that family and depth, retrained under the common protocol of Appendix D. We report parameter count, calibration loss, threshold-crossing error, and time per epoch.

Table 2 shows that SLiSDE substantially improves both accuracy and eficiency on the toy path-functional benchmark. The gated in-flow variant achieves the lowest total loss for both $L \ = \ 2$ and $L = 3 .$ , while using fewer parameters than the Neural SDE and SLiCE baselines. In terms of runtime, SLiSDE remains faster than the Neural SDE baseline, reflecting the benefit of scan-compatible structured dynamics. SLiCE, which shares the parallel associative scan, is not designed for random inputs: it treats the Brownian path as one more input sequence, and its gap shows that deterministic sequence models do not transfer to a task that requires genuinely nonlinear stochastic dynamics. The recovery of the full path law beyond the calibrated targets (Kolmogorov–Smirnov and Wasserstein-1 distances to the ground truth for the terminal value and the running maximum) is reported in Appendix E.

## 6.2 Option-surface benchmark (DAX)

dax is a real-data functional calibration benchmark built from historical European option quotes on the DAX index across multiple threshold levels and maturities. The task is to learn a stochastic path generator whose termina functionals match the observed option surface. For evaluation, we hold out 25% of the threshold levels in each target group using a structured non-contiguous mask, so that the model is tested on interpolation across the surface rather than extrapolation outside the observed range.

The same protocol is applied to the spx surface (Appendix D): there SLiSDE matches the held-out loss of SLiCE with two layers and attains the lowest held-out and far-tail loss with three layers, with 6–8× fewer parameters and

3.5–4× faster epochs, and both structured models are far ahead of the Neural SDE (Table 4, Appendix E).

Out-of-the-money (OTM) options correspond to thresholds that are far from the current level of the underlying index, so their value is determined by paths that end in the tails of the distribution.

Table 3 shows the real-data benchmark. SLiSDE gated in-flow attains the lowest held-out loss at both depths (with two layers within one standard error of SLiCE) and, with three layers, also the lowest far-tail loss of all models, at a parameter count below SLiCE’s and in about half of its time per epoch; with two layers SLiCE fits the far puts slightly better, within one standard error. The Neural SDE is the weakest at both depths.
<table><tr><td>Model</td><td>Params</td><td> $\mathrm { L o s s ~ ( 1 0 ^ { - 4 } ) }$ </td><td>Far put (10−6)</td><td>ms / epoch</td></tr><tr><td>Neural SDE – L = 2</td><td>79K</td><td> $1 . 9 7 2 \pm 0 . 1 1 7$ </td><td> $9 . 8 7 \pm 2 . 4 7$ </td><td>91.4</td></tr><tr><td> $\mathrm { S L i C E } - L = 2$ </td><td>142K</td><td> $1 . 6 0 5 \pm 0 . 1 4 9$ </td><td> ${ \bf 8 . 1 4 \pm 1 . 6 9 }$ </td><td>169.9</td></tr><tr><td>SLiSDE gated in-flow – L = 2</td><td>90K</td><td> $\mathbf { 1 . 5 5 2 \pm 0 . 0 7 0 }$ </td><td> $9 . 4 0 \pm 2 . 3 0$ </td><td>79.7</td></tr><tr><td>Neural SDE – L = 3</td><td>157K</td><td> $1 . 9 0 8 \pm 0 . 1 8 0$ </td><td> $8 . 2 0 \pm 1 . 4 2$ </td><td>114.4</td></tr><tr><td> ${ \mathrm { S L i C E } } - L = 3$ </td><td>213K</td><td> $1 . 5 6 0 \pm 0 . 0 8 6$ </td><td> $6 . 4 1 \pm 1 . 3 3$ </td><td>246.9</td></tr><tr><td>SLiSDE gated in-flow – L = 3</td><td>135K</td><td> $\mathbf { 1 . 4 0 1 \pm 0 . 2 4 4 }$ </td><td> ${ \bf 4 . 0 7 \pm 0 . 6 2 }$ </td><td>129.1</td></tr></table>

Mean ± standard error over seven seeds per cell; best result per column in bold; losses are L<sub>n</sub>(θ) of Eq. 5 on the held-out strikes, from a fresh Monte-Carlo sample.  
Table 3: dax results without Girsanov tilt. L is the number of stacked layers; each row is the tuned cell of the search grid for that family and depth, retrained under the common protocol of Appendix D. We report parameter count, held-out loss, held-out far-put (OTM) loss, and time per epoch.

Girsanov tilt. With the tilt of Section 4 the far-tail calibration of the dax model improves at a fixed path budget: over seven seeds the mean far-call held-out error falls by about a third relative to the vanilla model (0.64×; lower on six seeds of seven, paired t-test on the log-errors $p = 0 . 0 3 )$ and the mean total held-out loss drops by 15–24%, while the budget-matched untilted model is never better on the far tails (Table 9). In a controlled rare-event study on toy, the tilt with the vanilla importance-sampling estimator reduces the relative error of plain Monte Carlo at equal budget by a factor of 2 to 4, growing with the rarity of the event (Table 10); both studies are reported in Appendix E.

Timing. Wall-clock times of full training steps for batch sizes up to 4096 and horizons up to $T = 8 1 9 2$ are reported in Appendix E. At equal parameter count, SLiSDE evaluated sequentially is 6–7× faster per step than the Neural SDE at B = 64 and 1.3–1.4× faster at B = 1024; only at B = 4096, where the GPU is saturated by batch parallelism, is the Neural SDE step faster (179 against 242 ms).

## 7 Conclusion

We introduced SLiSDE, a family of Neural SDEs that moves nonlinearity from per-step drift and difusion evaluations to gated in-flow coupling between structured linear stochastic layers. Each layer remains afine in its own state and is therefore parallel-scannable, while cross-layer modulation, time-dependent coeficients, and a learned linear decoder provide expressivity. For tail-sensitive path-functional objectives, we also use a last-layer Girsanov tilt, together with a shift of the initial latent state, with a closed-form Radon–Nikodym correction, yielding a lightweight importance-sampling overlay. We prove well-posedness, scan compatibility, discretisation error bounds, and validity of the change of measure, and that the terminal laws of the gated stack are dense in $\mathcal { P } _ { 2 } \colon$ the structural restrictions cost nothing in expressive power at the level of terminal laws.

Empirically, SLiSDE attains a lower calibration loss than the fully neural SDE baseline on every benchmark and at both depths, and trains faster in all but two of the settings tested, the three-layer dax configuration and the largest batch size of the timing study. Against the structured-CDE baseline it is clearly better on toy; on the option surfaces it attains the lowest mean loss on dax at both depths and on the three-layer spx surface and matches it on the two-layer spx surface, diferences that are within one standard error, and with three layers it attains the lowest far-tail error on both surfaces. Throughout, it uses 1.4–8× fewer parameters than the structured-CDE baseline and trains 2–4× faster per epoch. The tilt is a variance-reduction device: on toy it reduces the error of rare-event estimates by factors of two to four at equal budget, growing with the rarity of the event, and on the far tails of dax it improves the far-tail calibration of the vanilla model, while budget-matched plain Monte Carlo remains a strong competitor where the targets are not rare. The results suggest that structured linear stochastic layers, combined with selective in-flow coupling and optional importance sampling, are a practical alternative to fully neural drift–difusion networks for long-horizon functional calibration.

Future work. From the stochastic-analysis point of view, a natural direction is to better understand gated in-flow stacking as a path-dependent transformation of linear difusions, including its possible connections to filtering. A second direction is to study structured Neural SDEs beyond functional calibration, in particular as continuous-time generative models such as Neural-SDE GANs.

## References

Marco Avellaneda, Craig Friedman, Richard Holmes, and Dominick Samperi. Calibrating volatility surfaces via relative-entropy minimization. Applied Mathematical Finance, 4(1):37–64, 1997.

Guy E. Blelloch. Prefix sums and their applications. Technical Report CMU-CS-90-190, School of Computer Science, Carnegie Mellon University, 1990.

Hans Buehler, Lukas Gonon, Josef Teichmann, and Ben Wood. Deep hedging. Quantitative Finance, 19(8):1271– 1291, 2019.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K. Duvenaud. Neural ordinary diferentia equations. In Advances in Neural Information Processing Systems, volume 31, 2018. URL https://papers. nips.cc/paper/7892-neural-ordinary-differential-equations.

Samuel N. Cohen, Christoph Reisinger, and Sheng Wang. Arbitrage-free neural-SDE market models. arXiv preprint arXiv:2105.11053, 2021.

Christa Cuchiero, Wahid Khosrawi, and Josef Teichmann. A generative adversarial network approach to calibration of local stochastic volatility models. Risks, 8(4):101, 2020.

Patryk Gierjatowicz, Marc Sabate-Vidales, David Siˇska, Lukasz Szpruch, and <sup>ˇ</sup> Zan <sup>ˇ</sup> Zuriˇc. Robust pricing and hedging <sup>ˇ</sup> via neural SDEs. arXiv preprint arXiv:2007.04154, 2020.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2023. URL https: //arxiv.org/abs/2312.00752.

Albert Gu, Karan Goel, and Christopher Re. Eficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= uYLFoz1vlAC.

Carsten Hartmann and Lorenz Richter. Nonasymptotic bounds for suboptimal importance sampling. SIAM/ASA Journal on Uncertainty Quantification, 12(2):309–346, 2024. doi: 10.1137/21M1427760. URL https://epubs. siam.org/doi/10.1137/21M1427760.

Carsten Hartmann, Lorenz Richter, Christof Schutte, and Wei Zhang. Variational characterization of free energy: Theory and algorithms. Entropy, 19(11):626, 2017. doi: 10.3390/e19110626. URL https://www.mdpi.com/ 1099-4300/19/11/626.

Ramin Hasani, Mathias Lechner, Tsun-Hsuan Wang, Makram Chahine, Alexander Amini, and Daniela Rus. Liquid structural state-space models, 2022. URL https://arxiv.org/abs/2209.12951.

Zacharia Issa, Blanka Horvath, Maud Lemercier, and Cristopher Salvi. Non-adversarial training of neural SDEs with signature kernel scores. In Advances in Neural Information Processing Systems, 2023.

Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled diferential equations for irregular time series. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https: //proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html.

Patrick Kidger, James Foster, Xuechen Li, Harald Oberhauser, and Terry Lyons. Neural SDEs as infinite dimensional GANs. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research. PMLR, 2021. URL https://proceedings.mlr.press/v139 kidger21b.html.

Peter E. Kloeden and Eckhard Platen. Numerical Solution of Stochastic Diferential Equations. Springer, Berlin, 1992. doi: 10.1007/978-3-662-12616-5.

Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, and David Duvenaud. Scalable gradients for stochastic diferential equations. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 3870–3882. PMLR, 2020. URL https://proceedings.mlr.press/v108/li20i.html.

Xuanqing Liu, Tesi Xiao, Si Si, Qin Cao, Sanjiv Kumar, and Cho-Jui Hsieh. Neural SDE: Stabilizing neural ODE networks with stochastic noise, 2019. URL https://arxiv.org/abs/1906.02355.

Nicola Muca Cirone, Antonio Orvieto, Benjamin Walker, Cristopher Salvi, and Terry Lyons. Theoretical foundations of deep selective state-space models. Advances in Neural Information Processing Systems, 37:127226–127272, 2024.

Philip E. Protter. Stochastic Integration and Diferential Equations. Springer, Berlin, 2 edition, 2005. doi: 10.1007/ 978-3-662-10061-5.

H. Smith et al. Calibrating generative models to distributional constraints. In International Conference on Machine Learning, 2026. TODO: complete author list and pages (cited by Reviewer X1pF as “Smith, H. et al., ICML 2026”).

Jimmy T. H. Smith, Andrew Warrington, and Scott W. Linderman. Simplified state space layers for sequence modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=Ai8Hw3AXqks.

Belinda Tzen and Maxim Raginsky. Neural stochastic diferential equations: Deep latent gaussian models in the difusion limit. CoRR, abs/1905.09883, 2019. URL https://arxiv.org/abs/1905.09883.

Benjamin Walker, Lingyi Yang, Nicola Muca Cirone, Cristopher Salvi, and Terry Lyons. Structured linear CDEs: Maximally expressive and parallel-in-time sequence models, 2025. URL https://arxiv.org/abs/2505.17761.

Qinsheng Zhang and Yongxin Chen. Path integral sampler: A stochastic control approach for sampling. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=\_uCb2ynRu7Y.

## A Mathematical Background Continued

## A.1 Girsanov change of measure

This subsection states the change of measure on which the tilt of Section 4 rests.

Let $u _ { t }$ be a progressively measurable $\mathbb { R } ^ { m }$ -valued process satisfying Novikov’s condition

$$
\mathbb { E } ^ { \mathbb { P } } \left[ \exp \left( \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| u _ { t } \| ^ { 2 } ~ \mathrm { d } t \right) \right] < \infty .\tag{13}
$$

Define

$$
\mathcal { E } _ { T } ( u ) = \exp \left( \int _ { 0 } ^ { T } u _ { t } ^ { \top } \mathrm { d } W _ { t } - \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| u _ { t } \| ^ { 2 } ~ \mathrm { d } t \right) .\tag{14}
$$

Then $\mathcal { E } _ { T } ( u )$ is a martingale density. If $\mathrm { d } \mathbb { Q } / \mathrm { d } \mathbb { P } = { \mathcal { E } } _ { T } ( u )$ , then

$$
W _ { t } ^ { \mathbb { Q } } = W _ { t } - \int _ { 0 } ^ { t } u _ { s } \mathrm { ~ d } s
$$

is a Brownian motion under $\mathbb { Q } .$ . Moreover, for any integrable F<sub>T</sub>-measurable path functional $G ,$

$$
\mathbb { E } ^ { \mathbb { P } } [ G ] = \mathbb { E } ^ { \mathbb { Q } } \left[ G \frac { { \mathrm { d } } \mathbb { P } } { { \mathrm { d } } \mathbb { Q } } \right] .\tag{15}
$$

The inverse density $\mathrm { d } \mathbb { P } / \mathrm { d } \mathbb { Q }$ is the importance weight that turns expectations under the tilted measure back into reference-measure expectations.

## A.2 Calibration objective: population limit, non-uniqueness and model selection

Population limit. With the notation of Section 2.2, assume $\varphi ( Y ) \in L ^ { 2 } ( \mathbb { P } _ { \theta } )$ for every $\varphi \in \Phi$ . Since ${ \widehat { \mathbb { E } } } _ { \theta , n } [ \varphi ( Y ) ]$ is an average of n independent copies,

$$
\mathbb { E } \Big [ \big ( \widehat { \mathbb { E } } _ { \theta , n } [ \varphi ( Y ) ] - c _ { \varphi } \big ) ^ { 2 } \Big ] = \big ( \mathbb { E } _ { \mathbb { P } _ { \theta } } [ \varphi ( Y ) ] - c _ { \varphi } \big ) ^ { 2 } + \frac { 1 } { n } \operatorname { V a r } _ { \mathbb { P } _ { \theta } } \big ( \varphi ( Y ) \big ) ,
$$

and by the strong law of large numbers, almost surely as $n \to \infty$

$$
\mathcal { L } _ { n } ( \theta ) \longrightarrow \mathcal { L } _ { \infty } ( \theta ) = \sum _ { \varphi \in \Phi } w _ { \varphi } \big ( \mathbb { E } _ { \mathbb { P } _ { \theta } } [ \varphi ( Y ) ] - c _ { \varphi } \big ) ^ { 2 } .
$$

The objective is therefore a functional of the model law. The decomposition

$$
\mathbb { E } \Vert X _ { t } ^ { \prime } - X _ { t } \Vert ^ { 2 } = \mathrm { V a r } ( X _ { t } ^ { \prime } ) + \mathrm { V a r } ( X _ { t } ) + \Vert \mathbb { E } X _ { t } ^ { \prime } - \mathbb { E } X _ { t } \Vert ^ { 2 } ,
$$

which shows that a samplewise squared error between model and data paths rewards collapse of the model onto the data mean, does not apply: no model path is paired with a data path, and the only variance term in ${ \mathcal { L } } _ { n }$ carries the coeficient $1 / n$ and vanishes in the population limit. In all experiments $n \geq 1 0 2 4$

No collapse. Collapse is moreover incompatible with the population objective whenever the target marginal is non-degenerate. If $Y _ { t } = m _ { t }$ almost surely then

$$
\mathbb { E } _ { \mathbb { P } _ { \theta } } [ ( Y _ { t } - K ) _ { + } ] = ( m _ { t } - K ) _ { + } ,
$$

a piecewise afine function of K with a single change of slope at $K = m _ { t }$ , which cannot coincide on an interval with a strictly convex target map $K \mapsto c _ { t , K }$ such as an arbitrage-free call surface. More generally

$$
\mathbb { E } _ { \mathbb { P } _ { \theta } } [ ( Y _ { t } - K ) _ { + } ] = \int _ { K } ^ { \infty } \mathbb { P } _ { \theta } ( Y _ { t } > x ) \mathrm { ~ d } x ,
$$

so knowledge of the positive-part expectations for every K determines the marginal law of $Y _ { t } ;$ the finite strike grid of the implementation provides a finite collection of constraints of this type, and the running-maximum and thresholdcrossing functionals constrain path-dependent features in addition. We do not claim that a finite collection of such constraints determines the entire path law.

Non-uniqueness and model risk. Finitely many expectation constraints are satisfied by infinitely many difusions, so $\mathcal { L } _ { \infty }$ has no unique minimiser. This is a property of functional calibration rather than of the objective: the data are finitely many expectations, so no objective built on them can identify the path law. In derivatives markets one typically has nothing beyond quoted prices, and it is standard practice to work with – and deliberately explore

– the multiplicity of models consistent with the same quotes in order to quantify model risk; Gierjatowicz et al (2020), for example, use randomised calibrated neural SDEs to obtain price bounds over the calibrated set. Enforcing uniqueness through an auxiliary term would not remove the ambiguity but shift it to the choice of the reference measure. Theorem 5.3 guarantees that the model class itself never restricts what is reachable, so the selection among constraint-consistent processes is driven by the data, the initialisation and the training procedure, never by a hidden limitation of the class. Operationally we validate the selection where a selection can be validated, on functionals the model was never fitted to: the held-out-strike transfer on dax and spx, and the distributional-recovery study on toy (Kolmogorov–Smirnov and Wasserstein-1 distances to the ground-truth simulator, Appendix E).

Model selection within the calibrated set. There are settings, physics in particular, where prior information is available: a reference dynamics, an equilibrium law, known symmetries. There, an additive penalty turns Eq. 5 into a principled selection rule – a Kullback–Leibler divergence to the available reference (Smith et al., 2026), a minimal-relative-entropy criterion in the spirit of Avellaneda et al. (1997) for volatility surfaces, or a context-specific penalty – and such terms drop into the training loop unchanged. The framework already computes and uses exactly this quantity where a reference measure genuinely exists: in the Girsanov layer, the controller is trained with a KL(Q∥P) anchor (Appendix C.8), the pathwise log Radon–Nikodym exponent that our machinery evaluates, which penalises tilts that move too far from the untilted reference model.

Relation to generative training. Training against a fixed family of statistics can be viewed as a maximummean-discrepancy objective with a non-characteristic kernel, a point discussed by Kidger et al. (2021), who contrast it with their adversarial formulation using a learned discriminator; the variational formulation of Tzen and Raginsky (2019) and the signature-kernel scores of Issa et al. (2023) likewise address a diferent, generative learning objective that requires sample paths. The method belongs instead to the calibration strand (Gierjatowicz et al., 2020; Cuchiero et al., 2020; Cohen et al., 2021), in which prescribed functional expectations, such as derivative prices, are matched.

## B Theoretical Results: Assumptions, Auxiliary Results and Proofs

This appendix states the standing assumptions in full, establishes the two auxiliary results used throughout the paper (exactness of the parallel associative scan and the strong discretisation error of the stacked scheme) and proves every result of Section 5. It is organised as follows.

• Appendix B.1 states the assumptions and proves Lemma 1: a gated layer inherits the coeficient bounds of its base layer with constants that do not depend on the depth. Every later constant rests on this lemma.

• Appendix B.2 proves Theorem 5.1, first for one layer with random coeficients (Lemma 2) and then by induction over the layers.

• Appendix B.3 proves Proposition B.1, the exactness of the scan.

• Appendix B.4 proves Theorem B.1 in four steps: one-step moments (Lemma 3), discrete stability (Lemma 4), the Euler–Maruyama error of one layer (Lemma 5) and an induction over the layers that carries a doubled moment order at every step down the stack.

• Appendices B.5 and B.6 prove Theorems 5.2 and 5.3.

Notation. ∥ · ∥ denotes the Euclidean norm of a vector and the operator norm of a matrix, $\| \cdot \| _ { \mathrm { F } }$ the Frobenius norm and $\| x \| _ { \infty }$ the largest modulus of the entries of x. For a coeficient tuple $\boldsymbol { \vartheta } = ( A , b , C ^ { 1 } , \ldots , C ^ { m } , d ^ { 1 } , \ldots , d ^ { m } )$ we write

$$
\| \vartheta \| : = \| A \| + \| b \| + \sum _ { j = 1 } ^ { m } \bigl ( \| C ^ { j } \| + \| d ^ { j } \| \bigr ) .\tag{16}
$$

For a coeficient process $\boldsymbol { \vartheta } _ { t } = \left( A _ { t } , b _ { t } , C _ { t } ^ { 1 } , \ldots , C _ { t } ^ { m } , d _ { t } ^ { 1 } , \ldots , d _ { t } ^ { m } \right)$ we set

$$
\mu ( t , z ) : = A _ { t } z + b _ { t } , \qquad \sigma ^ { j } ( t , z ) : = C _ { t } ^ { j } z + d _ { t } ^ { j } , \qquad \sigma ( t , z ) : = \big [ \sigma ^ { 1 } ( t , z ) , \dots , \sigma ^ { m } ( t , z ) \big ] \in \mathbb { R } ^ { d \times m } ,\tag{17}
$$

so that the layer SDE Eq. 6 reads

$$
\mathrm { d } Z _ { t } = \mu ( t , Z _ { t } ) \mathrm { ~ d } t + \sigma ( t , Z _ { t } ) \mathrm { ~ d } W _ { t } .
$$

The letter C denotes a constant that may change from line to line and depends only on the bounds in the assumptions, on $T ,$ on m and on the moment order under consideration; where a dependence matters it is made explicit We use the Burkholder–Davis–Gundy inequality (Protter, 2005, Ch. IV) in the form

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq t } \Big \| \int _ { 0 } ^ { s } H _ { u } \mathrm { d } W _ { u } \Big \| ^ { q } \leq c _ { q } \mathbb { E } \Big ( \int _ { 0 } ^ { t } \| H _ { u } \| _ { \mathrm { F } } ^ { 2 } \mathrm { d } u \Big ) ^ { q / 2 } , \qquad q \geq 2 ,\tag{18}
$$

valid for every progressively measurable $\mathbb { R } ^ { d \times m }$ -valued process H with finite right-hand side. Finally,

$$
m _ { r } : = \mathbb { E } | \xi | ^ { r } , \qquad \xi \sim \mathcal { N } ( 0 , 1 ) ,
$$

denotes the absolute moments of the standard normal law.

## B.1 Standing assumptions

Assumption B.1 (Admissible model class). Fix $T > 0$ , a Brownian dimension m $\geq 1$ and an output dimension $p \in \mathbb { N } .$ . Let $\boldsymbol { W } = ( W ^ { 1 } , \dots , W ^ { m } )$ be an m-dimensional Brownian motion on the filtered probability space of Section 2 and fix $z _ { 0 } \in L ^ { 2 } ( \Omega , \mathcal { F } _ { 0 } , \mathbb { P } ; \mathbb { R } ^ { d _ { 0 } } )$ for some $d _ { 0 } \in \mathbb { N }$ , without any non-degeneracy assumption on its law. For every

$d \geq d _ { 0 }$ set

$$
\iota _ { d } ( z _ { 0 } ) : = ( z _ { 0 } , 0 , \ldots , 0 ) \in \mathbb { R } ^ { d } .
$$

An admissible model configuration consists of a finite number L of stacked layers of a common latent width $d \geq d _ { 0 }$ each driven by a Brownian motion $W ^ { ( \ell ) }$ adapted to the common filtration (shared, independent or correlated across layers; the implementation shares one driver across the prefix layers and uses the correlated driver of Eq. 10 for the last one), such that:

1. each base layer is the structured afine Itˆo SDE Eq. 6 with $Z _ { 0 } = \iota _ { d } ( z _ { 0 } )$ , whose coeficients $A _ { t } , C _ { t } ^ { j } \in \mathbb { R } ^ { d \times d }$ and $b _ { t } , d _ { t } ^ { j } \in \mathbb { R } ^ { d }$ are the static parameters $A , C ^ { j } , b , d ^ { j }$ perturbed by the time-feature decoder of Appendix C.2, which adds deterministic, smooth-in-time corrections to the free coeficient entries and respects the chosen diagonal, block-diagonal or dense structure;

2. successive layers are connected by gated in-flow stacking (Section 3.3) through the gates of $E q s .$ 81 and 82, with fixed $\varepsilon \in ( 0 , 1 )$ , and the gate input is

$$
x _ { t } ^ { ( \ell - 1 ) } = \big [ \Pi ( Z _ { t } ^ { ( \ell - 1 ) } ) , \tau ( t ) \big ] , \qquad \Pi ( z ) : = \gamma \odot \frac { z } { \sqrt { \varsigma + d ^ { - 1 } \| z \| ^ { 2 } } } ,\tag{19}
$$

where Π is the RMS normalisation with gain $\gamma \in \mathbb { R } ^ { d }$ and regulariser $\varsigma > 0$ , and τ is the vector of deterministic time features, which contains the time coordinate t and is Lipschitz on [0, T]; all gate weights and biases are trainable, and the optional causal convolution is disabled;

3. the number of layers, the latent width and the hidden widths of the time decoders and gates may be increased arbitrarily, and their afine weights and biases range freely over the corresponding Euclidean spaces;

4. the output is a linear readout of the final layer,

$$
Y _ { t } ^ { \theta } = \Pi _ { o } Z _ { t } ^ { ( L ) } , \qquad \Pi _ { o } \in \mathbb { R } ^ { p \times d } ,
$$

shifted to match a prescribed initial observation $y _ { 0 }$ when one is given.

Remark B.1 (Discrete and continuous gating). The implementation gates the discrete afine pair,

$$
\begin{array} { r } { \bar { F } _ { k } = \mathcal { D } ( \alpha _ { k } , F _ { k } ) , \qquad \bar { g } _ { k } = g _ { k } + o _ { k } , } \end{array}\tag{20}
$$

with no explicit $\Delta t$ on the $o f f s e t ,$ where ${ \mathcal { D } } ( \alpha , F )$ rescales the i-th diagonal entry of $F$ by $\alpha _ { i }$ . The continuous-time model analysed below carries the gated coeficients

$$
\alpha _ { t } ^ { ( \ell ) } \odot A _ { t } ^ { ( \ell ) } \quad i n p l a c e o f \quad A _ { t } ^ { ( \ell ) } , \qquad b _ { t } ^ { ( \ell ) } + o _ { t } ^ { ( \ell ) } \quad i n p l a c e o f \quad b _ { t } ^ { ( \ell ) } ,\tag{21}
$$

where $\alpha \odot A$ rescales the i-th diagonal entry of A by $\alpha _ { i }$ , while the difusion coeficients are not gated. The two conventions are related by the dictionary

$$
o _ { k } = o _ { t _ { k } } \Delta t , \qquad \alpha _ { k } = \exp \bigl ( \delta _ { t _ { k } } \Delta t \bigr ) \ e n t r y w i s e ,\tag{22}
$$

where $\delta$ denotes the diagonal drift modulation. The first identity is exact because the linear branch of the ofset gate is homogeneous in its weights, so that rescaling $W _ { i o } , c _ { i o }$ by $\Delta t$ stays within the family. The second is realisable on every fixed grid because $\alpha _ { k }$ ranges over a neighbourhood of 1.

Assumption B.2 (Coeficients). For every layer, the coeficient process $\vartheta _ { t }$ is $( \mathcal { F } _ { t } )$ -progressively measurable and

uniformly bounded,

$$
\| A _ { t } \| \leq a _ { * } , \qquad \| b _ { t } \| \leq b _ { * } , \qquad \| C _ { t } ^ { j } \| \leq c _ { * } , \qquad \| d _ { t } ^ { j } \| \leq d _ { * } ,\tag{23}
$$

almost surely, for all $t \in [ 0 , T ]$ and $j \leq m$ , and $\frac { 1 } { 2 }$ -H¨older in time in every L<sup>q</sup>: for each $q \geq 2$ there is $L _ { q } < \infty$ with

$$
\begin{array} { r } { \mathbb { E } \| \vartheta _ { t } - \vartheta _ { s } \| ^ { q } \leq L _ { q } ^ { q } \vert t - s \vert ^ { q / 2 } , \qquad s , t \in [ 0 , T ] . } \end{array}\tag{24}
$$

For the ungated base layer the coeficients are deterministic and Lipschitz in t with a constant $L _ { \tau }$ , because the time features are smooth and the decoders are fixed networks. Eq. 24 then holds with

$$
\begin{array} { r } { L _ { q } = L _ { \tau } T ^ { 1 / 2 } , } \end{array}
$$

so Assumption B.2 holds by construction. For the gated layers it is established in Lemma 1 below. The bounds Eq. 23 imply, for the maps of Eq. 17, the Lipschitz and linear-growth bounds

$$
\begin{array} { c } { \displaystyle \| \mu ( t , z ) - \mu ( t , z ^ { \prime } ) \| + \sum _ { j = 1 } ^ { m } \| \sigma ^ { j } ( t , z ) - \sigma ^ { j } ( t , z ^ { \prime } ) \| \leq \Lambda ^ { \prime } \| z - z ^ { \prime } \| , } \\ { \displaystyle \| \mu ( t , z ) \| + \| \sigma ( t , z ) \| _ { \mathrm { F } } \leq \Lambda ^ { \prime } \big ( 1 + \| z \| \big ) , } \end{array}\tag{25}
$$

almost surely for all $t , z$ and $z ^ { \prime } ,$ with

$$
\Lambda ^ { \prime } : = a _ { * } + b _ { * } + m ( c _ { * } + d _ { * } ) .\tag{26}
$$

Assumption B.3 (Gates). For each layer $\ell \geq 2$ the gate map sends a continuous path $z = ( z _ { t } ) _ { t \in [ 0 , T ] }$ of the previous layer to the gate processes,

$$
z \longmapsto { \bigl ( } \alpha _ { t } ^ { ( \ell ) } ( z ) , o _ { t } ^ { ( \ell ) } ( z ) { \bigr ) } _ { t \in [ 0 , T ] } .
$$

It is causal, in that $( \alpha _ { t } ^ { ( \ell ) } , o _ { t } ^ { ( \ell ) } )$ depends on z only through $z _ { [ 0 , t ] }$ (for the gates of Eqs. 81 and 82 with the input Eq. 19, only through $z _ { t }$ and t), and jointly measurable, so that applied to a continuous adapted process it yields progressively measurable processes. Moreover, with $\Lambda _ { \Pi }$ and $\Lambda _ { x }$ defined in (a):

(a) (Normalised input.) For all $z , z ^ { \prime } \in \mathbb { R } ^ { d }$

$$
\begin{array} { r } { \| \Pi ( z ) \| \le \sqrt { d } \| \gamma \| _ { \infty } = : \Lambda _ { \Pi } , \qquad \| \Pi ( z ) - \Pi ( z ^ { \prime } ) \| \le L _ { \Pi } \| z - z ^ { \prime } \| , \qquad L _ { \Pi } : = \frac { \| \gamma \| _ { \infty } } { \sqrt { \varsigma } } , } \end{array}\tag{27}
$$

so that the gate input satisfies

$$
\| x _ { t } ^ { ( \ell - 1 ) } \| \leq \Lambda _ { \Pi } + \operatorname* { s u p } _ { t \leq T } \| \tau ( t ) \| = : \Lambda _ { x } .
$$

(b) (Bounded flow gate.) Surely, for every input path and every choice of the weights,

$$
\| \alpha _ { t } ^ { ( \ell ) } - \mathbf { 1 } \| _ { \infty } \leq \varepsilon .
$$

(c) (Ofset bounded for fixed weights.) Surely,

$$
\begin{array} { r } { \| o _ { t } ^ { ( \ell ) } \| \le \| W _ { i o } ^ { ( \ell ) } \| \Lambda _ { x } + \| c _ { i o } ^ { ( \ell ) } \| = : \Lambda _ { o } < \infty ; } \end{array}\tag{28}
$$

$\Lambda _ { o }$ is finite for every fixed choice of the weights and free in magnitude.

(d) (Lipschitz gate.) There is $L _ { g } < \infty$ such that, for all continuous paths $z , z ^ { \prime }$ and all $s \leq t \leq T$

$$
\| { \boldsymbol { \alpha } } _ { t } ^ { ( \ell ) } ( z ) - { \boldsymbol { \alpha } } _ { t } ^ { ( \ell ) } ( z ^ { \prime } ) \| _ { \infty } + \| { \boldsymbol { o } } _ { t } ^ { ( \ell ) } ( z ) - { \boldsymbol { o } } _ { t } ^ { ( \ell ) } ( z ^ { \prime } ) \| \leq L _ { g } \operatorname* { s u p } _ { r \leq t } \| z _ { r } - z _ { r } ^ { \prime } \| ,\tag{29}
$$

$$
\| { \alpha } _ { t } ^ { ( \ell ) } ( z ) - { \alpha } _ { s } ^ { ( \ell ) } ( z ) \| _ { \infty } + \| { o } _ { t } ^ { ( \ell ) } ( z ) - { o } _ { s } ^ { ( \ell ) } ( z ) \| \le L _ { g } \big ( \| z _ { t } - z _ { s } \| + | t - s | \big ) .\tag{30}
$$

All four properties hold for the implemented gates.

Property (a). Write

$$
s ( z ) : = { \sqrt { \varsigma + d ^ { - 1 } \| z \| ^ { 2 } } } , \qquad f ( z ) : = { \frac { z } { s ( z ) } } ,
$$

so that $\Pi ( z ) = \gamma \odot f ( z )$ and $\| f ( z ) \| \leq { \sqrt { d } } .$ . The Jacobian of $f$ is the symmetric matrix

$$
D f ( z ) = s ( z ) ^ { - 1 } I - s ( z ) ^ { - 3 } d ^ { - 1 } z z ^ { \top } .
$$

Its eigenvalue on $z ^ { \perp }$ is $s ( z ) ^ { - 1 }$ , and its eigenvalue in the direction of $z$ is

$$
s ( z ) ^ { - 1 } - s ( z ) ^ { - 3 } d ^ { - 1 } \| z \| ^ { 2 } = s ( z ) ^ { - 3 } \varsigma ;
$$

all eigenvalues therefore lie in $( 0 , \varsigma ^ { - 1 / 2 } ]$ . Hence $\| \Pi ( z ) \| \leq \| \gamma \| _ { \infty } \sqrt { d } .$ , and Π is Lipschitz with constant $\| \gamma \| _ { \infty } \varsigma ^ { - 1 / 2 }$

Properties $( b )$ and $( c )$ . They hold because $| \sigma | \le 1$ and $| \operatorname { t a n h } | \leq 1$ in Eq. 81, and because $| \sigma | \le 1$ in Eq. 82 while the gate input lies in the ball of radius $\Lambda _ { x }$

Property (d). The gates are compositions of Π, afine maps and the Lipschitz activations σ and tanh, all restricted to the bounded input set, together with the Lipschitz time features. The product structure of Eq. 82 is Lipschitz on that bounded set because both factors are bounded and Lipschitz there.

Lemma 1 (Gated layers satisfy Assumption B.2 with depth-uniform constants). Let Assumptions B.1 and B.3 hold, let $\ell \geq 2$ , let the previous layer $Z ^ { ( \ell - 1 ) }$ be a continuous adapted process, and consider the gated coeficient process of layer $\ell ,$

$$
\vartheta _ { t } ^ { ( \ell ) } : = \big ( \alpha _ { t } ^ { ( \ell ) } \odot A _ { t } ^ { ( \ell ) } , b _ { t } ^ { ( \ell ) } + o _ { t } ^ { ( \ell ) } , C _ { t } ^ { ( \ell ) , 1 } , \ldots , C _ { t } ^ { ( \ell ) , m } , d _ { t } ^ { ( \ell ) , 1 } , \ldots , d _ { t } ^ { ( \ell ) , m } \big ) ,\tag{31}
$$

with $( \alpha ^ { ( \ell ) } , o ^ { ( \ell ) } )$ evaluated on $Z ^ { ( \ell - 1 ) }$

(a) (Bounds.) $\vartheta ^ { ( \ell ) }$ is progressively measurable and satisfies

$$
\| { \alpha } _ { t } ^ { ( \ell ) } \odot { A } _ { t } ^ { ( \ell ) } \| \le ( 1 + \varepsilon ) a _ { * } , \qquad \| { b } _ { t } ^ { ( \ell ) } + { o } _ { t } ^ { ( \ell ) } \| \le b _ { * } + \Lambda _ { o } , \qquad \| C _ { t } ^ { ( \ell ) , j } \| \le c _ { * } , \qquad \| d _ { t } ^ { ( \ell ) , j } \| \le d _ { * } ,\tag{32}
$$

Consequently the maps of $E q .$ . 17 built from $\vartheta ^ { ( \ell ) }$ satisfy $E q .$ 25 with the constant

$$
\Lambda _ { \star } ^ { \prime } : = ( 1 + \varepsilon ) a _ { * } + b _ { * } + \Lambda _ { o } + m ( c _ { * } + d _ { * } ) ,\tag{33}
$$

which depends neither on $\ell ,$ nor on $L ,$ nor on the previous layer. This part needs no moment bound on $Z ^ { ( \ell - 1 ) }$

(b) (Time regularity.) If moreover, for some $q \geq 2$

$$
\begin{array} { r } { \mathbb { E } \| Z _ { t } ^ { ( \ell - 1 ) } - Z _ { s } ^ { ( \ell - 1 ) } \| ^ { q } \leq K _ { q } ^ { \prime } | t - s | ^ { q / 2 } , \qquad s , t \in [ 0 , T ] , } \end{array}\tag{34}
$$

then $\vartheta ^ { ( \ell ) }$ satisfies $E q .$ 24 at the exponent $q ,$ with

$$
\begin{array} { r } { L _ { q } ^ { q } = 3 ^ { q - 1 } \Big ( \big ( ( 1 + a _ { * } ) L _ { g } \big ) ^ { q } \big ( K _ { q } ^ { \prime } + T ^ { q / 2 } \big ) + \big ( ( 2 + \varepsilon ) L _ { \tau } \big ) ^ { q } T ^ { q / 2 } \Big ) . } \end{array}\tag{35}
$$

In particular, if $E q .$ 34 holds for every $q \geq 2$ , then $\vartheta ^ { ( \ell ) }$ satisfies Assumption $B . 2 .$

Proof. Part $( a )$ , step 1: measurability. By Assumption B.3 the gate map is causal and jointly measurable, so its evaluation on the continuous adapted process $Z ^ { ( \ell - 1 ) }$ is progressively measurable. The base coeficients are deterministic and continuous in t. Hence $\vartheta ^ { ( \ell ) }$ is progressively measurable.

Part $( a ) _ { i }$ , step 2: bounds. Rescaling the i-th diagonal entry of $A _ { t } ^ { ( \ell ) }$ $\alpha _ { t , i } ^ { ( \ell ) }$ adds a diagonal matrix to $A _ { t } ^ { ( \ell ) }$

$$
\alpha _ { t } ^ { ( \ell ) } \odot A _ { t } ^ { ( \ell ) } = A _ { t } ^ { ( \ell ) } + \mathrm { d i a g } \big ( ( { \alpha _ { t , i } ^ { ( \ell ) } } - 1 ) A _ { t , i i } ^ { ( \ell ) } \big ) _ { i \le d } , \qquad | A _ { t , i i } ^ { ( \ell ) } | \le \| A _ { t } ^ { ( \ell ) } \| .
$$

Assumption B.3(b) therefore gives

$$
\| { \boldsymbol { \alpha } } _ { t } ^ { ( \ell ) } \odot { \boldsymbol { A } } _ { t } ^ { ( \ell ) } \| \leq \| { \boldsymbol { A } } _ { t } ^ { ( \ell ) } \| + \operatorname* { m a x } _ { i } | { \boldsymbol { \alpha } } _ { t , i } ^ { ( \ell ) } - 1 | | { \boldsymbol { A } } _ { t , i i } ^ { ( \ell ) } | \leq \left( 1 + \varepsilon \right) { \boldsymbol { a } } _ { \ast } .\tag{36}
$$

By Assumption B.3(c),

$$
\lVert b _ { t } ^ { ( \ell ) } + o _ { t } ^ { ( \ell ) } \rVert \leq b _ { * } + \Lambda _ { o } ,
$$

and the difusion coeficients are not gated. This proves $\operatorname { E q } .$ . 32. $\operatorname { E q } .$ . 25 with the constant $\Lambda _ { \star } ^ { \prime }$ follows exactly as $\operatorname { E q . }$ 26 follows from Eq. 23.

Part $( b ) .$ : time regularity. Fix $s \leq t$ and abbreviate $\alpha _ { t } = \alpha _ { t } ^ { ( \ell ) } , o _ { t } = o _ { t } ^ { ( \ell ) }$ and $Z = Z ^ { ( \ell - 1 ) }$ . The argument of Eq. 36 gives

$$
\begin{array} { r } { \| \alpha _ { s } \odot B \| \le ( 1 + \varepsilon ) \| B \| \qquad \mathrm { f o r ~ e v e r y ~ m a t r i x ~ } B , } \end{array}
$$

and the base coeficients are Lipschitz in t with constant $L _ { \tau }$ . Therefore

$$
\begin{array} { r l } { \| \vartheta _ { t } ^ { ( \ell ) } - \vartheta _ { s } ^ { ( \ell ) } \| \leq \| ( \alpha _ { t } - \alpha _ { s } ) \odot A _ { t } ^ { ( \ell ) } \| + \| \alpha _ { s } \odot ( A _ { t } ^ { ( \ell ) } - A _ { s } ^ { ( \ell ) } ) \| + \| o _ { t } - o _ { s } \| } & { } \\ { + \| b _ { t } ^ { ( \ell ) } - b _ { s } ^ { ( \ell ) } \| + \displaystyle \sum _ { j = 1 } ^ { m } ( \| C _ { t } ^ { ( \ell ) , j } - C _ { s } ^ { ( \ell ) , j } \| + \| d _ { t } ^ { ( \ell ) , j } - d _ { s } ^ { ( \ell ) , j } \| ) } & { } \\ { \leq a _ { * } \| \alpha _ { t } - \alpha _ { s } \| _ { \infty } + \| o _ { t } - o _ { s } \| + ( 2 + \varepsilon ) L _ { \tau } | t - s | . } & { } \end{array}\tag{37}
$$

By Eq. 30,

$$
\begin{array} { r } { a _ { * } \| \alpha _ { t } - \alpha _ { s } \| _ { \infty } + \| o _ { t } - o _ { s } \| \le ( 1 + a _ { * } ) L _ { g } \big ( \| Z _ { t } - Z _ { s } \| + | t - s | \big ) . } \end{array}
$$

Raising to the power q and using $( x + y + z ) ^ { q } \leq 3 ^ { q - 1 } ( x ^ { q } + y ^ { q } + z ^ { q } )$

$$
\begin{array} { r } { \| \vartheta _ { t } ^ { ( \ell ) } - \vartheta _ { s } ^ { ( \ell ) } \| ^ { q } \leq 3 ^ { q - 1 } \Big ( \big ( ( 1 + a _ { * } ) L _ { g } \big ) ^ { q } \| Z _ { t } - Z _ { s } \| ^ { q } + \big ( ( 1 + a _ { * } ) L _ { g } \big ) ^ { q } | t - s | ^ { q } + \big ( ( 2 + \varepsilon ) L _ { \tau } \big ) ^ { q } | t - s | ^ { q } \Big ) . } \end{array}
$$

Taking expectations and using Eq. 34 together with $| t - s | ^ { q } \leq T ^ { q / 2 } | t - s | ^ { q / 2 }$

$$
\mathbb { E } \Vert \vartheta _ { t } ^ { ( \ell ) } - \vartheta _ { s } ^ { ( \ell ) } \Vert ^ { q } \leq L _ { q } ^ { q } \vert t - s \vert ^ { q / 2 }
$$

with the constant $L _ { q }$ of Eq. 35, which is Eq. 24.

## B.2 Well-posedness: proof of Theorem 5.1

We first treat one layer with a general random coeficient process; the time regularity $\operatorname { E q } .$ . 24 is not needed here.

Lemma 2 (One layer with random coeficients). Let ϑ be a progressively measurable coeficient process satisfying the bounds $E q .$ 23, let $q \geq 2$ , and let $Z _ { 0 }$ be $\mathcal { F } _ { 0 }$ -measurable with ${ \mathbb { E } } \Vert Z _ { 0 } \Vert ^ { q } < \infty$ . Then the SDE

$$
\mathrm { d } Z _ { t } = \mu ( t , Z _ { t } ) \mathrm { ~ d } t + \sigma ( t , Z _ { t } ) \mathrm { ~ d } W _ { t } , \qquad Z _ { 0 } \mathrm { ~ } g i v e n ,
$$

has a unique strong solution on [0, T], and

$$
\begin{array} { r } { \mathbb { E } \operatorname* { s u p } _ { t \leq T } \| Z _ { t } \| ^ { q } \leq K _ { q } \big ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { q } \big ) , } \end{array}\tag{38}
$$

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq t } \| Z _ { r } - Z _ { s } \| ^ { q } \leq K _ { q } ^ { \prime } \big ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { q } \big ) | t - s | ^ { q / 2 } , \qquad 0 \leq s \leq t \leq T ,\tag{39}
$$

with constants $K _ { q } , K _ { q } ^ { \prime }$ depending only on $( \Lambda ^ { \prime } , T , m , q )$

Proof. Step 1: existence and uniqueness. By Eq. 25 the coeficients are progressively measurable, globally Lipschitz in z and of linear growth, uniformly in $( t , \omega )$ . Existence and uniqueness of a strong solution for such random Lipschitz coeficients is classical (Protter, 2005, Ch. V, Thm. 7).

Step 2: moment bound. Fix $n \in \mathbb { N }$ and the localising time

$$
\tau _ { n } : = \operatorname* { i n f } \{ t \geq 0 : \| Z _ { t } \| \geq n \} \wedge T .
$$

For $t \leq T$ , the integral form of the equation gives

$$
Z _ { t \wedge \tau _ { n } } = Z _ { 0 } + \int _ { 0 } ^ { t \wedge \tau _ { n } } \mu ( s , Z _ { s } ) \mathrm { ~ d } s + \int _ { 0 } ^ { t } \mathbf { 1 } _ { \{ s < \tau _ { n } \} } \sigma ( s , Z _ { s } ) \mathrm { ~ d } W _ { s } .
$$

Using $\| x + y + z \| ^ { q } \leq 3 ^ { q - 1 } ( \| x \| ^ { q } + \| y \| ^ { q } + \| z \| ^ { q } )$ 2

$$
\operatorname* { s u p } _ { r \leq t } \| Z _ { r \wedge \tau _ { n } } \| ^ { q } \leq 3 ^ { q - 1 } \Big [ \| Z _ { 0 } \| ^ { q } + \Big ( \int _ { 0 } ^ { t \wedge \tau _ { n } } \| \mu ( s , Z _ { s } ) \| { \mathrm { ~ d } } s \Big ) ^ { q } + \operatorname* { s u p } _ { r \leq t } \Big \| \int _ { 0 } ^ { r } \mathbf { 1 } _ { \{ s < \tau _ { n } \} } \sigma ( s , Z _ { s } ) { \mathrm { ~ d } } W _ { s } \Big \| ^ { q } \Big ] .\tag{40}
$$

On $\{ s < \tau _ { n } \}$ we have $Z _ { s } = Z _ { s \wedge \tau _ { n } }$ . By Jensen’s inequality and Eq. 25,

$$
\Big ( \int _ { 0 } ^ { t \wedge \tau _ { n } } \| \mu ( s , Z _ { s } ) \| \ \mathrm { d } s \Big ) ^ { q } \leq T ^ { q - 1 } \int _ { 0 } ^ { t } \| \mu ( s , Z _ { s \wedge \tau _ { n } } ) \| ^ { q } \ \mathrm { d } s \leq T ^ { q - 1 } 2 ^ { q - 1 } \Lambda ^ { \prime q } \int _ { 0 } ^ { t } \big ( 1 + \| Z _ { s \wedge \tau _ { n } } \| ^ { q } \big ) \ \mathrm { d } s .
$$

By Eq. 18, Jensen’s inequality (for $q / 2 \geq 1 )$ and Eq. 25,

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle \operatorname* { s u p } _ { r \leq t } \left\| \int _ { 0 } ^ { r } \mathbf { 1 } _ { \{ s < \tau _ { n } \} } \sigma ( s , Z _ { s } ) \mathrm { d } W _ { s } \right\| ^ { q } \leq c _ { q } T ^ { q / 2 - 1 } \mathbb { E } \int _ { 0 } ^ { t } \mathbf { 1 } _ { \{ s < \tau _ { n } \} } \| \sigma ( s , Z _ { s } ) \| _ { \mathrm { F } } ^ { q } \mathrm { d } s } \\ & { \qquad \leq c _ { q } T ^ { q / 2 - 1 } 2 ^ { q - 1 } \Lambda ^ { \prime q } \displaystyle \int _ { 0 } ^ { t } \left( 1 + \mathbb { E } \| Z _ { s \wedge \tau _ { n } } \| ^ { q } \right) \mathrm { d } s . } \end{array}
$$

Set

$$
\phi _ { n } ( t ) : = \mathbb { E } \operatorname* { s u p } _ { r \leq t } \| Z _ { r \wedge \tau _ { n } } \| ^ { q } ,
$$

which is finite because $\| Z _ { r \wedge \tau _ { n } } \| \leq \operatorname* { m a x } ( n , \| Z _ { 0 } \| )$ . Taking expectations in $\operatorname { E q } .$ 40,

$$
\phi _ { n } ( t ) \leq 3 ^ { q - 1 } \mathbb { E } \| Z _ { 0 } \| ^ { q } + C _ { 1 } \int _ { 0 } ^ { t } \bigl ( 1 + \phi _ { n } ( s ) \bigr ) \mathrm { d } s , \qquad C _ { 1 } : = 3 ^ { q - 1 } 2 ^ { q - 1 } \Lambda ^ { \prime q } \bigl ( T ^ { q - 1 } + c _ { q } T ^ { q / 2 - 1 } \bigr ) ,
$$

and Gr¨onwall’s lemma gives

$$
\begin{array} { r } { \phi _ { n } ( T ) \leq \big ( 3 ^ { q - 1 } \mathbb { E } \| Z _ { 0 } \| ^ { q } + C _ { 1 } T \big ) e ^ { C _ { 1 } T } , } \end{array}
$$

uniformly in n. Since the paths are continuous, $\tau _ { n } \uparrow T$ almost surely and

$$
\operatorname* { s u p } _ { r \leq T } \| Z _ { r \wedge \tau _ { n } } \| \uparrow \operatorname* { s u p } _ { r \leq T } \| Z _ { r } \| ,
$$

so monotone convergence yields Eq. 38 with

$$
K _ { q } : = { \left( 3 ^ { q - 1 } + C _ { 1 } T \right) } e ^ { C _ { 1 } T } .
$$

Step 3: increments. Fix $s \leq t .$ For $r \in [ s , t ]$

$$
Z _ { r } - Z _ { s } = \int _ { s } ^ { r } \mu ( u , Z _ { u } ) ~ \mathrm { d } u + \int _ { s } ^ { r } \sigma ( u , Z _ { u } ) ~ \mathrm { d } W _ { u } .
$$

The same two inequalities, now without localisation because the moments are finite by Step 2, give

$$
\begin{array} { r l } { \underset { s \leq r \leq t } { \mathbb { E } } \| Z _ { r } - Z _ { s } \| ^ { q } \leq 2 ^ { q - 1 } \Big [ ( t - s ) ^ { q - 1 } \underset { s } { \mathbb { E } } \int _ { s } ^ { t } \| \mu ( u , Z _ { u } ) \| ^ { q } \mathrm { d } u } \\ & { \qquad + c _ { q } ( t - s ) ^ { q / 2 - 1 } \underset { s } { \mathbb { E } } \int _ { s } ^ { t } \| \sigma ( u , Z _ { u } ) \| _ { \mathrm { F } } ^ { q } \mathrm { d } u \Big ] } \\ & { \leq 2 ^ { q - 1 } 2 ^ { q - 1 } \Lambda ^ { \prime q } \big ( 1 + \underset { u \leq T } { \mathbb { E } } \| Z _ { u } \| ^ { q } \big ) \big [ ( t - s ) ^ { q } + c _ { q } ( t - s ) ^ { q / 2 } \big ] . } \end{array}
$$

Since $( t - s ) ^ { q } \leq T ^ { q / 2 } ( t - s ) ^ { q / 2 }$ , Eq. 38 gives

$$
\mathbb { E } \operatorname* { s u p } _ { s \leq r \leq t } \| Z _ { r } - Z _ { s } \| ^ { q } \leq 4 ^ { q - 1 } \Lambda ^ { \prime q } \big ( 1 + K _ { q } \big ) \big ( T ^ { q / 2 } + c _ { q } \big ) \big ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { q } \big ) ( t - s ) ^ { q / 2 } ,
$$

which is Eq. 39.

Proof of Theorem 5.1. (a) Single layer. The base layer satisfies Assumption B.2 by construction. Lemma 2 with $Z _ { 0 } = \iota _ { d } ( z _ { 0 } )$ gives existence, uniqueness and Eq. 38 with

$$
K _ { q } = K _ { q } ( \Lambda ^ { \prime } , T , m , q ) ,
$$

where Λ<sup>′</sup> is the function Eq. 26 of $( a _ { * } , b _ { * } , c _ { * } , d _ { * } , m )$

(b) Gated stack. Let $\Lambda _ { \star } ^ { \prime } \geq \Lambda ^ { \prime }$ be the constant Eq. 33, and write

$$
K _ { q } ^ { \star } : = K _ { q } ( \Lambda _ { \star } ^ { \prime } , T , m , q ) , \qquad K _ { q } ^ { \prime \star } : = K _ { q } ^ { \prime } ( \Lambda _ { \star } ^ { \prime } , T , m , q ) ,
$$

for the constants of Lemma 2 at this value; they depend on $( a _ { * } , b _ { * } , c _ { * } , d _ { * } , m , T , q , \varepsilon , \Lambda _ { o } )$ only. We prove by induction

on ℓ that layer ℓ has a unique strong solution and that, for every $q \geq 2$ with $\mathbb { E } \Vert z _ { 0 } \Vert ^ { q } < \infty$

$$
\begin{array} { c } { \mathbb { E } \operatorname* { s u p } _ { t \leq T } \| Z _ { t } ^ { ( \ell ) } \| ^ { q } \leq K _ { q } ^ { \star } \big ( 1 + \mathbb { E } \| z _ { 0 } \| ^ { q } \big ) , } \\ { \mathbb { E } \| Z _ { t } ^ { ( \ell ) } - Z _ { s } ^ { ( \ell ) } \| ^ { q } \leq K _ { q } ^ { \prime \star } \big ( 1 + \mathbb { E } \| z _ { 0 } \| ^ { q } \big ) | t - s | ^ { q / 2 } , \qquad s , t \in [ 0 , T ] . } \end{array}\tag{41}
$$

The base layer is (a), since $\Lambda ^ { \prime } \leq \Lambda _ { \star } ^ { \prime }$

Let $\ell \geq 2$ and assume the claim for layer ℓ−1; in particular $Z ^ { ( \ell - 1 ) }$ is a continuous adapted process. By Lemma $\mathrm { 1 ( a ) }$ which needs no moment bound on the previous layer, the gated coeficient process $\vartheta ^ { ( \ell ) }$ of $\operatorname { E q }$ . 31 is progressively measurable and satisfies the bounds Eq. 32, hence Eq. 25 with the constant $\Lambda _ { \star } ^ { \prime }$ . Lemma 2 applied to layer $\ell ,$ with $Z _ { 0 } = \iota _ { d } ( z _ { 0 } )$ and $\Lambda _ { \star } ^ { \prime }$ in place of $\Lambda ^ { \prime } { . }$ gives existence, uniqueness and Eq. 41 for layer ℓ. The constants do not depend on $\ell ,$ on L or on the previous layer, which closes the induction.

Only adaptedness of the drivers to the common filtration was used, so the conclusion holds for any correlation structure of the layer drivers. □

Remark B.2 (The normalisation is what decouples the depth cascade). The single point where the architecture enters is that the ofset, though of unbounded form, is a bounded forcing because its input is normalised. If the gate read the raw state, the ofset would satisfy only

$$
\lVert o _ { t } ^ { ( \ell ) } \rVert \stackrel { } { \sim } \lVert W _ { i o } \rVert \lVert Z _ { t } ^ { ( \ell - 1 ) } \rVert ,
$$

a linear growth in the previous layer. The moment bounds $M _ { \ell } : = \mathbb { E } \operatorname* { s u p } _ { t } \| Z _ { t } ^ { ( \ell ) } \| ^ { q }$ would then obey a recursion of the form

$$
M _ { \ell } \leq K \big ( 1 + \| W _ { i o } \| ^ { q } M _ { \ell - 1 } \big ) ,
$$

and the bound would grow geometrically in L unless $K \| W _ { i o } \| ^ { q } \leq 1$ . Uniformity in L is therefore a consequence of the normalisation, not of the algebraic form of the gate.

## B.3 Exactness of the parallel associative scan

Proposition B.1 (Scan exactness). Let $\mathcal { M } \subseteq \mathbb { R } ^ { d \times d }$ be closed under multiplication with $I \in { \mathcal { M } }$ (the diagonal, block-diagonal or dense family), and let ◦ be the composition rule of Eq. 4.

(a) ◦ is associative; the prefix products

$$
( \widehat { F } _ { k } , \widehat { g } _ { k } ) : = ( F _ { k - 1 } , g _ { k - 1 } ) \circ \cdot \cdot \cdot \circ ( F _ { 0 } , g _ { 0 } ) , \qquad ( \widehat { F } _ { 0 } , \widehat { g } _ { 0 } ) : = ( I , 0 ) ,\tag{42}
$$

satisfy

$$
Z _ { k } = \widehat { F } _ { k } Z _ { 0 } + \widehat { g } _ { k } \qquad f o r \ t h e \ r e c u r s i o n \qquad Z _ { k + 1 } = F _ { k } Z _ { k } + g _ { k } ;
$$

hence any associative-scan evaluation order returns exactly the sequential iterates, in ${ \cal O } ( \log T )$ parallel depth, and $\widehat { F } _ { k } \in { \mathcal { M } }$ whenever all $F _ { k } \in \mathcal { M }$

(b) Fix a realisation of the previous-layer path. The gated pairs

$$
( \bar { F } _ { k } ^ { ( \ell ) } , \bar { g } _ { k } ^ { ( \ell ) } ) = \big ( \mathcal { D } ( \alpha _ { k } ^ { ( \ell ) } , F _ { k } ^ { ( \ell ) } ) , g _ { k } ^ { ( \ell ) } + o _ { k } ^ { ( \ell ) } \big )\tag{43}
$$

of Eq. 7 are then fixed afine data with $\bar { F } _ { k } ^ { ( \ell ) } \in \mathcal { M }$ , so (a) applies within each layer, and the depth-L stack is evaluated by L sequential scans in $O ( L \log T )$ depth.

Proof. (a) To an afine pair associate the augmented matrix

$$
\iota ( F , g ) : = { \binom { F \quad g } { 0 } } \in \mathbb { R } ^ { ( d + 1 ) \times ( d + 1 ) } .\tag{44}
$$

The map ι is injective, and a direct multiplication gives

$$
\iota ( F ^ { \prime } , g ^ { \prime } ) \iota ( F , g ) = \left( \begin{array} { c c } { F ^ { \prime } F } & { F ^ { \prime } g + g ^ { \prime } } \\ { 0 } & { 1 } \end{array} \right) = \iota \big ( ( F ^ { \prime } , g ^ { \prime } ) \circ ( F , g ) \big ) .\tag{45}
$$

Hence, for any three pairs,

$$
\begin{array} { r l } & { \iota \Big ( \big ( ( F _ { c } , g _ { c } ) \circ ( F _ { b } , g _ { b } ) \big ) \circ ( F _ { a } , g _ { a } ) \Big ) = \iota ( F _ { c } , g _ { c } ) \iota ( F _ { b } , g _ { b } ) \iota ( F _ { a } , g _ { a } ) } \\ & { \qquad = \iota \Big ( ( F _ { c } , g _ { c } ) \circ \big ( ( F _ { b } , g _ { b } ) \circ ( F _ { a } , g _ { a } ) \big ) \Big ) , } \end{array}
$$

and injectivity of ι gives associativity of ◦.

Next, write $\xi _ { k } : = ( Z _ { k } ^ { \top } , 1 ) ^ { \top }$ . The recursion reads

$$
\xi _ { k + 1 } = \iota ( F _ { k } , g _ { k } ) \xi _ { k } ,
$$

so by induction and Eq. 45

$$
\begin{array} { r } { \xi _ { k } = \iota \left( F _ { k - 1 } , g _ { k - 1 } \right) \cdot \cdot \cdot \iota \left( F _ { 0 } , g _ { 0 } \right) \xi _ { 0 } = \iota \left( \widehat { F } _ { k } , \widehat { g } _ { k } \right) \xi _ { 0 } , \qquad \mathrm { t h a t ~ i s , } \qquad Z _ { k } = \widehat { F } _ { k } Z _ { 0 } + \widehat { g } _ { k } . } \end{array}
$$

Since every bracketing of an associative product agrees, any order of evaluation of the prefixes Eq. 42 returns the same pairs, and therefore the same iterates $Z _ { k }$ . The work-eficient parallel prefix algorithm (Blelloch, 1990) evaluates all of them along a balanced binary tree of depth logarithmic in the number of time steps, that is, O(log T) with the convention of Section 2. Finally

$$
\widehat { F } _ { k } = F _ { k - 1 } \cdot \cdot \cdot F _ { 0 } \in \mathcal { M }
$$

because M is closed under multiplication. The diagonal and block-diagonal implementations are structured representations of the same products: diagonal multiplication is elementwise and block-diagonal multiplication is independent multiplication inside each block.

(b) The gates $\alpha _ { k } ^ { ( \ell ) }$ and $o _ { k } ^ { ( \ell ) }$ depend only on the previous layer up to time $t _ { k }$ and on deterministic time features, never on the current state $Z _ { k } ^ { ( \ell ) }$ ; conditioning on the previous path fixes them as deterministic data. Rescaling the diagonal entries of a matrix in the diagonal, block-diagonal or dense family leaves its sparsity pattern intact, so $\bar { F } _ { k } ^ { ( \ell ) } \in \mathcal { M }$ , and the ofset only shifts $g _ { k } ^ { ( \ell ) }$ . Thus (a) applies inside layer ℓ. The layers run sequentially because the data of layer ℓ depend on $Z ^ { ( \ell - 1 ) }$ , which gives L scans of depth O(log T) each. □

Remark B.3. The gate makes the coeficients depend on the other layer, never on the current state, so afinity within a layer, and with it exactness of the scan, is untouched. This is the precise sense in which gated in-flow stacking preserves the parallel associative scan.

## B.4 Strong discretisation error

Throughout this subsection the coeficient process $\vartheta$ of the layer under consideration is progressively measurable and satisfies the bounds Eq. 23; the time regularity Eq. 24 is invoked where it is needed. On the uniform grid

$$
t _ { k } = k \Delta t , \qquad k = 0 , \ldots , N , \qquad \Delta t = T / N \le 1 ,
$$

write

$$
\kappa ( t ) : = \operatorname* { m a x } \{ t _ { k } : t _ { k } \leq t \} , \quad \quad \Delta W _ { k } : = W _ { t _ { k + 1 } } - W _ { t _ { k } } .
$$

The Euler–Maruyama step of $\operatorname { E q . }$ 78 is the afine map

$$
\begin{array} { c } { { \displaystyle \bar { Z } _ { k + 1 } = \bar { Z } _ { k } + \mu ( t _ { k } , \bar { Z } _ { k } ) \Delta t + \sigma ( t _ { k } , \bar { Z } _ { k } ) \Delta W _ { k } = F _ { k } \bar { Z } _ { k } + g _ { k } , } } \\ { { { } } } \\ { { F _ { k } = I + M _ { k } , \qquad M _ { k } : = A _ { t _ { k } } \Delta t + \displaystyle \sum _ { j = 1 } ^ { m } C _ { t _ { k } } ^ { j } \Delta W _ { k } ^ { j } , \qquad g _ { k } : = b _ { t _ { k } } \Delta t + \displaystyle \sum _ { j = 1 } ^ { m } d _ { t _ { k } } ^ { j } \Delta W _ { k } ^ { j } , } } \end{array}\tag{46}
$$

started at $\bar { Z } _ { 0 } = Z _ { 0 }$ . Its continuous interpolation is

$$
\hat { Z } _ { t } : = \bar { Z } _ { k } + \mu ( t _ { k } , \bar { Z } _ { k } ) \left( t - t _ { k } \right) + \sigma ( t _ { k } , \bar { Z } _ { k } ) \left( W _ { t } - W _ { t _ { k } } \right) , \qquad t \in [ t _ { k } , t _ { k + 1 } ] ,\tag{47}
$$

which satisfies $\hat { Z } _ { t _ { k } } = \bar { Z } _ { k }$ and

$$
\mathrm { d } \hat { Z } _ { t } = \mu \big ( \kappa ( t ) , \hat { Z } _ { \kappa ( t ) } \big ) \mathrm { d } t + \sigma \big ( \kappa ( t ) , \hat { Z } _ { \kappa ( t ) } \big ) \mathrm { d } W _ { t } .\tag{48}
$$

All estimates hold for any correlation structure of the layer drivers, since no conditioning on other layers is used.

Lemma 3 (One-step conditional moments). For every $r \geq 1$ there is a constant $C _ { r }$ , depending only on $( a _ { * } , b _ { * } , c _ { * }$ $d _ { * } , m , r )$ , such that for $\Delta t \leq 1$ and every $k ,$

$$
\begin{array} { r l } { \left\| \mathbb { E } [ M _ { k } \mid \mathcal { F } _ { t _ { k } } ] \right\| \leq a _ { * } \Delta t , } & { \quad \left\| \mathbb { E } [ g _ { k } \mid \mathcal { F } _ { t _ { k } } ] \right\| \leq b _ { * } \Delta t , } \\ { \mathbb { E } \big [ \| M _ { k } \| ^ { r } \bigm | \mathcal { F } _ { t _ { k } } \big ] \leq C _ { r } \Delta t ^ { r / 2 } , } & { \quad \mathbb { E } \big [ \| g _ { k } \| ^ { r } \bigm | \mathcal { F } _ { t _ { k } } \big ] \leq C _ { r } \Delta t ^ { r / 2 } . } \end{array}\tag{49}
$$

Proof. $\boldsymbol { \vartheta } _ { t _ { k } }$ is $\mathcal { F } _ { t _ { k } }$ -measurable and bounded by Eq. 23, while $\Delta W _ { k }$ is independent of $\mathcal { F } _ { t _ { l } }$ with

$$
\mathbb { E } [ \Delta W _ { k } ] = 0 , \qquad \mathbb { E } | \Delta W _ { k } ^ { j } | ^ { r } = m _ { r } \Delta t ^ { r / 2 } .
$$

The first two bounds follow. For the third,

$$
\left\| M _ { k } \right\| \leq a _ { * } \Delta t + c _ { * } \sum _ { j = 1 } ^ { m } | \Delta W _ { k } ^ { j } | ,
$$

so, by

$$
( x _ { 0 } + \cdots + x _ { m } ) ^ { r } \leq ( m + 1 ) ^ { r - 1 } \bigl ( x _ { 0 } ^ { r } + \cdots + x _ { m } ^ { r } \bigr ) \qquad \mathrm { a n d } \qquad \Delta t \leq 1 ,
$$

$$
\begin{array} { r } { \mathbb { E } \big [ \| M _ { k } \| ^ { r } \big | \mathcal { F } _ { t _ { k } } \big ] \leq ( m + 1 ) ^ { r - 1 } \big ( a _ { * } ^ { r } \Delta t ^ { r } + m c _ { * } ^ { r } m _ { r } \Delta t ^ { r / 2 } \big ) \leq ( m + 1 ) ^ { r - 1 } \big ( a _ { * } ^ { r } + m c _ { * } ^ { r } m _ { r } \big ) \Delta t ^ { r / 2 } . } \end{array}
$$

The bound for $g _ { k }$ is identical with $( b _ { * } , d _ { * } )$ in place of $( a _ { * } , c _ { * } )$

Lemma 4 (Discrete stability). Let $q \geq 2$ be even and $\mathbb { E } \Vert Z _ { 0 } \Vert ^ { q } < \infty$ . There is $C _ { q } = { C _ { q } } ( \Lambda ^ { \prime } , T , m , q )$ such that, for

∆t ≤ 1,

$$
\displaystyle \operatorname* { s u p } _ { 0 \leq k \leq N } \mathbb { E } \| \bar { Z } _ { k } \| ^ { q } \leq C _ { q } \big ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { q } \big ) ,\tag{50}
$$

$$
\operatorname* { s u p } _ { t \leq T } \mathbb { E } \| \hat { Z } _ { t } \| ^ { q } \leq C _ { q } \big ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { q } \big ) , \qquad \mathbb { E } \| \hat { Z } _ { t } - \hat { Z } _ { \kappa ( t ) } \| ^ { q } \leq C _ { q } \big ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { q } \big ) \Delta t ^ { q / 2 } , \qquad t \leq T ,\tag{51}
$$

uniformly in N. Moreover $\begin{array} { r } { \mathbb { E } \operatorname* { s u p } _ { t < T } \| \hat { Z } _ { t } \| ^ { q } < \infty } \end{array}$

Proof. Step 1: proof of Eq. 50. Write $q = 2 p$ with $p \in \mathbb N$ and

$$
\xi _ { k } : = M _ { k } \bar { Z } _ { k } + g _ { k } , \qquad \eta _ { k } : = 2 \langle \bar { Z } _ { k } , \xi _ { k } \rangle + \| \xi _ { k } \| ^ { 2 } ,
$$

so that $\bar { Z } _ { k + 1 } = \bar { Z } _ { k } + \xi _ { k }$ and

$$
\lVert \bar { Z } _ { k + 1 } \rVert ^ { 2 } = \lVert \bar { Z } _ { k } \rVert ^ { 2 } + \eta _ { k } .
$$

By the binomial theorem,

$$
\Vert \bar { Z } _ { k + 1 } \Vert ^ { 2 p } = \Vert \bar { Z } _ { k } \Vert ^ { 2 p } + p \Vert \bar { Z } _ { k } \Vert ^ { 2 p - 2 } \eta _ { k } + \sum _ { j = 2 } ^ { p } { \binom { p } { j } } \Vert \bar { Z } _ { k } \Vert ^ { 2 ( p - j ) } \eta _ { k } ^ { j } .\tag{52}
$$

Since $\bar { Z } _ { k }$ is $\mathcal { F } _ { t _ { k } }$ -measurable, Lemma 3 gives, for every $r \geq 1$

$$
\begin{array} { r } { \mathbb { E } \big [ \| \xi _ { k } \| ^ { r } \bigm | \mathcal { F } _ { t _ { k } } \big ] \leq 2 ^ { r - 1 } \Big ( \| \bar { Z } _ { k } \| ^ { r } \mathbb { E } \big [ \| M _ { k } \| ^ { r } \bigm | \mathcal { F } _ { t _ { k } } \big ] + \mathbb { E } \big [ \| g _ { k } \| ^ { r } \bigm | \mathcal { F } _ { t _ { k } } \big ] \Big ) \leq 2 ^ { r - 1 } C _ { r } \Delta t ^ { r / 2 } \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { r } . } \end{array}\tag{53}
$$

For the linear term of Eq. 52, Lemma 3 and Eq. 53 with $r = 2$ give

$$
\begin{array} { r l } & { \mathbb { E } [ \eta _ { k } \mid \mathcal { F } _ { t _ { k } } ] = 2 \big \langle \bar { Z } _ { k } , \mathbb { E } [ M _ { k } \mid \mathcal { F } _ { t _ { k } } ] \bar { Z } _ { k } + \mathbb { E } [ g _ { k } \mid \mathcal { F } _ { t _ { k } } ] \big \rangle + \mathbb { E } \big [ \| \xi _ { k } \| ^ { 2 } \big \vert \mathcal { F } _ { t _ { k } } \big ] } \\ & { \qquad \leq 2 \Delta t \big ( a _ { * } \| \bar { Z } _ { k } \| ^ { 2 } + b _ { * } \| \bar { Z } _ { k } \| \big ) + 2 C _ { 2 } \Delta t \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { 2 } \leq C \Delta t \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { 2 } . } \end{array}
$$

For the higher terms,

$$
| \eta _ { k } | \leq 2 \| \bar { Z } _ { k } \| \| \xi _ { k } \| + \| \xi _ { k } \| ^ { 2 } ,
$$

so for $j \geq 2$ , by Eq. 53 with $r = j$ and $r = 2 j$

$$
\begin{array} { r l } & { \mathbb { E } \left[ | \eta _ { k } | ^ { j } \ \big | \ \mathcal { F } _ { t _ { k } } \right] \leq 2 ^ { j - 1 } \Big ( 2 ^ { j } \| \bar { Z } _ { k } \| ^ { j } \mathbb { E } \left[ \| \xi _ { k } \| ^ { j } \ \big | \ \mathcal { F } _ { t _ { k } } \right] + \mathbb { E } \big [ \| \xi _ { k } \| ^ { 2 j } \ \big | \ \mathcal { F } _ { t _ { k } } \big ] \Big ) } \\ & { \qquad \leq C \big ( \Delta t ^ { j / 2 } + \Delta t ^ { j } \big ) \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { 2 j } \leq C \Delta t \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { 2 j } , } \end{array}
$$

because $\Delta t \leq 1$ and $j \geq 2$ . Taking conditional expectations in Eq. 52 and using

$$
\begin{array} { r } { \| \bar { Z } _ { k } \| ^ { 2 ( p - j ) } \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { 2 j } \leq \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { 2 p } \leq 2 ^ { 2 p - 1 } \big ( 1 + \| \bar { Z } _ { k } \| ^ { 2 p } \big ) , } \end{array}
$$

we obtain

$$
\begin{array} { r } { \mathbb { E } \big [ \| \bar { Z } _ { k + 1 } \| ^ { 2 p } \big | \mathcal { F } _ { t _ { k } } \big ] \leq \| \bar { Z } _ { k } \| ^ { 2 p } + C \Delta t \big ( 1 + \| \bar { Z } _ { k } \| ^ { 2 p } \big ) . } \end{array}
$$

With $u _ { k } : = \mathbb { E } \| \bar { Z } _ { k } \| ^ { 2 p }$ this reads

$$
u _ { k + 1 } \leq \left( 1 + C \Delta t \right) u _ { k } + C \Delta t ,
$$

and by induction

$$
u _ { k } \le \left( 1 + C \Delta t \right) ^ { k } u _ { 0 } + C \Delta t \sum _ { i = 0 } ^ { k - 1 } ( 1 + C \Delta t ) ^ { i } \le e ^ { C k \Delta t } \bigl ( u _ { 0 } + C k \Delta t \bigr ) \le e ^ { C T } \bigl ( u _ { 0 } + C T \bigr ) , \qquad k \le N ,
$$

which is Eq. 50.

Step 2: proof of $E q .$ . 51. For $t \in [ t _ { k } , t _ { k + 1 } ]$ , Eq. 47 gives

$$
\hat { Z } _ { t } - \bar { Z } _ { k } = \mu ( t _ { k } , \bar { Z } _ { k } ) \left( t - t _ { k } \right) + \sigma ( t _ { k } , \bar { Z } _ { k } ) \left( W _ { t } - W _ { t _ { k } } \right) .
$$

Conditionally on $\mathcal { F } _ { t _ { k } }$ the second term is a centred Gaussian vector, and

$$
\begin{array} { r } { \| \sigma ( t _ { k } , \bar { Z } _ { k } ) ( W _ { t } - W _ { t _ { k } } ) \| \le \| \sigma ( t _ { k } , \bar { Z } _ { k } ) \| _ { \mathrm { F } } \| W _ { t } - W _ { t _ { k } } \| , \qquad \mathbb { E } \| W _ { t } - W _ { t _ { k } } \| ^ { q } \le m ^ { q / 2 } m _ { q } \Delta t ^ { q / 2 } . } \end{array}
$$

Hence, by Eq. 25,

$$
\begin{array} { r } { \mathbb { E } \big [ \| \hat { Z } _ { t } - \bar { Z } _ { k } \| ^ { q } \bigm | \mathcal { F } _ { t _ { k } } \big ] \leq 2 ^ { q - 1 } \Lambda ^ { \prime q } \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { q } \big ( \Delta { t } ^ { q } + m ^ { q / 2 } m _ { q } \Delta { t } ^ { q / 2 } \big ) \leq C \big ( 1 + \| \bar { Z } _ { k } \| \big ) ^ { q } \Delta { t } ^ { q / 2 } . } \end{array}
$$

Taking expectations and using Eq. 50 gives the second bound in Eq. 51; the first follows from

$$
\begin{array} { r } { \| \hat { Z } _ { t } \| ^ { q } \leq 2 ^ { q - 1 } \big ( \| \bar { Z } _ { k } \| ^ { q } + \| \hat { Z } _ { t } - \bar { Z } _ { k } \| ^ { q } \big ) . } \end{array}
$$

Finally,

$$
\operatorname* { s u p } _ { t \leq T } \| \hat { Z } _ { t } \| \leq \operatorname* { m a x } _ { k < N } \Bigl ( \| \bar { Z } _ { k } \| + \Lambda ^ { \prime } \bigl ( 1 + \| \bar { Z } _ { k } \| \bigr ) \bigl ( \Delta t + \operatorname* { s u p } _ { t \in [ t _ { k } , t _ { k + 1 } ] } \| W _ { t } - W _ { t _ { k } } \| \bigr ) \Bigr )
$$

is a maximum over finitely many random variables with finite q-th moments. Hence

$$
\mathbb { E } \operatorname* { s u p } _ { t \leq T } \Vert \hat { Z } _ { t } \Vert ^ { q } < \infty .
$$

Lemma 5 (Euler–Maruyama error for one layer). Let $q \geq 2$ be even, $\Delta t \leq 1$ and $\mathbb { E } \| Z _ { 0 } \| ^ { 2 q } < \infty$ . Let Z solve

$$
\mathrm { d } Z _ { t } = \mu ( t , Z _ { t } ) \ \mathrm { d } t + \sigma ( t , Z _ { t } ) \ \mathrm { d } W _ { t }
$$

and let $\hat { Z }$ be the interpolated scheme $E q . \mathrm { ~ } \not 4 7 ,$ with the same $Z _ { 0 }$ and the same driver. Then

$$
\mathbb { E } \operatorname* { s u p } _ { t \leq T } \| Z _ { t } - \hat { Z } _ { t } \| ^ { q } \leq C _ { q } \Bigl ( \Delta t ^ { q / 2 } + \operatorname* { s u p } _ { s \leq T } \bigl ( \mathbb { E } \| \vartheta _ { s } - \vartheta _ { \kappa ( s ) } \| ^ { 2 q } \bigr ) ^ { 1 / 2 } \Bigr ) ,\tag{54}
$$

where $C _ { q }$ depends only on $( \Lambda ^ { \prime } , T , m , q )$ and on $\mathbb { E } \Vert Z _ { 0 } \Vert ^ { 2 q }$ . If in addition ϑ satisfies the time regularity $\ E q . \ 2 4 ;$ , then

$$
\begin{array} { r } { \begin{array} { c } { \mathbb { E } \underset { t \leq T } { \operatorname* { s u p } } \| Z _ { t } - \hat { Z } _ { t } \| ^ { q } \leq C _ { q } \big ( 1 + L _ { 2 q } ^ { q } \big ) \Delta t ^ { q / 2 } , } \\ { i n \ p a r t i c u l a r \qquad \displaystyle \operatorname* { m a x } _ { 0 \leq k \leq N } \mathbb { E } \| Z _ { t _ { k } } - \bar { Z } _ { k } \| ^ { q } \leq C _ { q } \big ( 1 + L _ { 2 q } ^ { q } \big ) \Delta t ^ { q / 2 } \ : ; } \end{array} } \end{array}\tag{55}
$$

the scheme has strong order $\textstyle { \frac { 1 } { 2 } }$ in every $L ^ { q }$ (Kloeden and Platen, 1992, Thm. 10.2.2).

Proof. Set $e _ { t } : = Z _ { t } - \hat { Z } _ { t }$ , so that $e _ { 0 } = 0$ and, by Eq. 48,

$$
e _ { t } = \int _ { 0 } ^ { t } \bigl [ \mu ( s , Z _ { s } ) - \mu ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) \bigr ] \mathrm { d } s + \int _ { 0 } ^ { t } \bigl [ \sigma ( s , Z _ { s } ) - \sigma ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) \bigr ] \mathrm { d } W _ { s } .\tag{56}
$$

Step 1: decomposition of the coeficient diferences. For the drift,

$$
\mu ( s , Z _ { s } ) - \mu ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) = A _ { s } e _ { s } + \big ( A _ { s } - A _ { \kappa ( s ) } \big ) \hat { Z } _ { s } + \big ( b _ { s } - b _ { \kappa ( s ) } \big ) + A _ { \kappa ( s ) } \big ( \hat { Z } _ { s } - \hat { Z } _ { \kappa ( s ) } \big ) ,
$$

and, for each $j ,$

$$
\begin{array} { r } { \sigma ^ { j } ( s , Z _ { s } ) - \sigma ^ { j } ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) = C _ { s } ^ { j } e _ { s } + \big ( C _ { s } ^ { j } - C _ { \kappa ( s ) } ^ { j } \big ) \hat { Z } _ { s } + \big ( d _ { s } ^ { j } - d _ { \kappa ( s ) } ^ { j } \big ) + C _ { \kappa ( s ) } ^ { j } \big ( \hat { Z } _ { s } - \hat { Z } _ { \kappa ( s ) } \big ) . } \end{array}
$$

Define

$$
D _ { s } : = \| \vartheta _ { s } - \vartheta _ { \kappa ( s ) } \| \big ( 1 + \| \hat { Z } _ { s } \| \big ) , \qquad I _ { s } : = \| \hat { Z } _ { s } - \hat { Z } _ { \kappa ( s ) } \| .\tag{57}
$$

By Eq. 23 and the definition Eq. 16 of ∥ϑ∥,

$$
\begin{array} { r } { \| \mu ( s , Z _ { s } ) - \mu ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) \| \le a _ { * } \| e _ { s } \| + D _ { s } + a _ { * } I _ { s } , ~ } \\ { \| \sigma ( s , Z _ { s } ) - \sigma ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) \| _ { \mathrm { F } } \le m c _ { * } \| e _ { s } \| + D _ { s } + m c _ { * } I _ { s } . } \end{array}\tag{58}
$$

Step 2: a Gr¨onwall inequality. Let

$$
\phi ( t ) : = \mathbb { E } \operatorname* { s u p } _ { r \leq t } \| e _ { r } \| ^ { q } ,
$$

which is finite by Lemma 2 and Lemma 4. From Eq. $5 6 , \| x + y \| ^ { q } \leq 2 ^ { q - 1 } ( \| x \| ^ { q } + \| y \| ^ { q } )$ , Jensen’s inequality for the time integral, Eq. 18 with Jensen’s inequality for the stochastic integral, and Eq. 58,

$$
\begin{array} { r l } & { \phi ( t ) \leq 2 ^ { q - 1 } \Big [ T ^ { q - 1 } \mathbb { E } \int _ { 0 } ^ { t } \| \mu ( s , Z _ { s } ) - \mu ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) \| ^ { q } \ \mathrm { d } s } \\ & { \qquad + c _ { q } T ^ { q / 2 - 1 } \mathbb { E } \int _ { 0 } ^ { t } \| \sigma ( s , Z _ { s } ) - \sigma ( \kappa ( s ) , \hat { Z } _ { \kappa ( s ) } ) \| _ { \mathrm { F } } ^ { q } \ \mathrm { d } s \Big ] } \\ & { \leq C _ { 2 } \displaystyle \int _ { 0 } ^ { t } \big ( \mathbb { E } \| e _ { s } \| ^ { q } + \mathbb { E } D _ { s } ^ { q } + \mathbb { E } I _ { s } ^ { q } \big ) \mathrm { d } s \ \leq \ C _ { 2 } \displaystyle \int _ { 0 } ^ { t } \phi ( s ) \ \mathrm { d } s + C _ { 2 } \int _ { 0 } ^ { T } \big ( \mathbb { E } D _ { s } ^ { q } + \mathbb { E } I _ { s } ^ { q } \big ) \mathrm { d } s , } \end{array}\tag{59}
$$

with $C _ { 2 } = C _ { 2 } ( \Lambda ^ { \prime } , T , m , q )$ . Gr¨onwall’s lemma gives

$$
\phi ( T ) \leq C _ { 2 } e ^ { C _ { 2 } T } \int _ { 0 } ^ { T } \left( \mathbb { E } D _ { s } ^ { q } + \mathbb { E } I _ { s } ^ { q } \right) \mathrm { d } s .\tag{60}
$$

Step 3: the two forcing terms. By the Cauchy–Schwarz inequality and Eq. 51 at the exponent $2 q$

$$
\begin{array} { r l r } {  { \mathbb { E } D _ { s } ^ { q } \le \big ( \mathbb { E } \| \vartheta _ { s } - \vartheta _ { \kappa ( s ) } \| ^ { 2 q } \big ) ^ { 1 / 2 } \big ( \mathbb { E } ( 1 + \| \hat { Z } _ { s } \| ) ^ { 2 q } \big ) ^ { 1 / 2 } } } \\ & { } & { \leq C \big ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { 2 q } \big ) ^ { 1 / 2 } \underset { s \leq T } { \operatorname* { s u p } } \big ( \mathbb { E } \| \vartheta _ { s } - \vartheta _ { \kappa ( s ) } \| ^ { 2 q } \big ) ^ { 1 / 2 } , } \end{array}
$$

and by Eq. 51 at the exponent $q ,$

$$
\mathbb { E } I _ { s } ^ { q } \le C \bigl ( 1 + \mathbb { E } \| Z _ { 0 } \| ^ { q } \bigr ) \Delta t ^ { q / 2 } .
$$

Inserting both bounds into Eq. 60 proves Eq. 54.

Step 4: the rate under time regularity. Under Eq. 24,

$$
\begin{array} { r } { \mathbb { E } \| \vartheta _ { s } - \vartheta _ { \kappa ( s ) } \| ^ { 2 q } \leq L _ { 2 q } ^ { 2 q } | s - \kappa ( s ) | ^ { q } \leq L _ { 2 q } ^ { 2 q } \Delta { t } ^ { q } , } \end{array}
$$

so the supremum in Eq. 54 is at most $L _ { 2 q } ^ { q } \Delta t ^ { q / 2 }$ , which gives Eq. 55. The grid bound follows because $\hat { Z } _ { t _ { k } } = \bar { Z } _ { k } . \quad \sqcup$

Theorem B.1 (Stacked discretisation error). Let Assumptions B.1–B.3 hold, assume $\mathbb { E } \Vert z _ { 0 } \Vert ^ { r } < \infty$ for every $r \geq 2$ and apply the Euler–Maruyama scheme of Eq. 8 layerwise, the gates of the discrete layer ℓ being evaluated on the discrete previous layer at the grid points. Let $\hat { Z } ^ { ( \ell ) }$ denote the interpolation $E q .$ 47 of the discrete layer $\ell ,$ so that $\hat { Z } _ { t _ { k } } ^ { ( \ell ) } = \bar { Z } _ { k } ^ { ( \ell ) }$ . Then for each fixed depth L and every even $q \geq 2$ there is a constant ${ \cal C } _ { L , q } ,$ depending on the coeficient and gate bounds, on $T , m , q , L$ and on the moments of $z _ { \mathrm { 0 } }$ but not on $\Delta t _ { i }$ , such that for $\Delta t \leq 1$

$$
\operatorname* { m a x } _ { 1 \leq \ell \leq L } \ \mathbb { E } \operatorname* { s u p } _ { t \leq T } \bigl \Vert Z _ { t } ^ { ( \ell ) } - \hat { Z } _ { t } ^ { ( \ell ) } \bigr \Vert ^ { q } \leq C _ { L , q } \Delta t ^ { q / 2 } ,\tag{61}
$$

$$
i n \ p a r t i c u l a r \quad \quad \operatorname* { m a x } _ { 1 \le \ell \le L } \ \operatorname* { m a x } _ { 0 \le k \le N } \ \mathbb { E } \big \| Z _ { t _ { k } } ^ { ( \ell ) } - \bar { Z } _ { k } ^ { ( \ell ) } \big \| ^ { q } \le C _ { L , q } \Delta t ^ { q / 2 } .
$$

The scheme therefore has strong order ${ \begin{array} { l } { { \frac { 1 } { 2 } } } \end{array} } f o r$ every fixed depth.

Proof. Fix L and an even $q \geq 2 ,$ , and set

$$
q _ { \ell } : = q 2 ^ { L - \ell } , \qquad \hat { e } _ { \ell } : = { \mathbb { E } } \operatorname* { s u p } _ { t \leq T } \lVert Z _ { t } ^ { ( \ell ) } - \hat { Z } _ { t } ^ { ( \ell ) } \rVert ^ { q _ { \ell } } , \qquad \ell = 1 , \dots , L ,\tag{62}
$$

so that $q _ { L } = q$ and $q _ { \ell - 1 } = 2 q _ { \ell } \colon$ higher moments are carried at the shallow layers because each inter-layer step consumes one Cauchy–Schwarz inequality. We prove by induction on ℓ that

$$
\begin{array} { r } { \hat { e } _ { \ell } \le C \Delta t ^ { q _ { \ell } / 2 } , \qquad \ell = 1 , \dots , L , } \end{array}\tag{63}
$$

with constants that do not depend on $\Delta t .$ Eq. 61 then follows: at $\ell = L$ directly, and at $\ell < L$ by Jensen’s inequality,

$$
\mathbb { E } \operatorname* { s u p } _ { t \leq T } \lVert Z _ { t } ^ { ( \ell ) } - \hat { Z } _ { t } ^ { ( \ell ) } \rVert ^ { q } \leq \left( \hat { e } _ { \ell } \right) ^ { q / q _ { \ell } } \leq C ^ { q / q _ { \ell } } \Delta t ^ { q / 2 } ;
$$

the grid bound follows from $\hat { Z } _ { t _ { k } } ^ { ( \ell ) } = \bar { Z } _ { k } ^ { ( \ell ) }$

Base case. Layer 1 is ungated: its coeficients are deterministic and Lipschitz in $t ,$ so Assumption B.2 holds by construction, and Lemma 5 at the exponent $q _ { 1 }$ (which requires $\mathbb { E } \Vert z _ { 0 } \Vert ^ { 2 q _ { 1 } } < \infty )$ gives

$$
\hat { e } _ { 1 } \leq C \Delta t ^ { q _ { 1 } / 2 } .
$$

Induction step: the frozen coeficients. Let $\ell \geq 2$ and assume $\operatorname { E q }$ . 63 for $\ell - 1$ . The step compares three processes: the exact layer $Z ^ { ( \ell ) }$ , an intermediate SDE $\check { Z }$ whose gates are frozen on the grid and evaluated on the discrete previous layer, and the discrete layer $\hat { Z } ^ { ( \ell ) }$ . The distance from $\check { Z }$ to $\hat { Z } ^ { ( \ell ) }$ is a one-layer Euler–Maruyama error (Lemma 5); the distance from $Z ^ { ( \ell ) }$ to $\check { Z }$ is a coeficient perturbation controlled by the induction hypothesis. Let $\vartheta$ denote the exact-gate coeficient process of layer $\ell ,$ Eq. 31, with the gates evaluated on the exact previous layer $Z ^ { ( \ell - 1 ) }$ , and let $\bar { \vartheta }$ denote the discrete-gate coeficients, with the gates evaluated on the discrete previous layer and frozen on each grid interval:

$$
\begin{array} { r l } & { \bar { \vartheta } _ { t _ { k } } : = \big ( \alpha _ { t _ { k } } ^ { ( \ell ) } ( \hat { Z } ^ { ( \ell - 1 ) } ) \odot A _ { t _ { k } } ^ { ( \ell ) } , b _ { t _ { k } } ^ { ( \ell ) } + o _ { t _ { k } } ^ { ( \ell ) } ( \hat { Z } ^ { ( \ell - 1 ) } ) , C _ { t _ { k } } ^ { ( \ell ) , j } , d _ { t _ { k } } ^ { ( \ell ) , j } \big ) , } \\ & { \qquad \bar { \vartheta } _ { t } : = \bar { \vartheta } _ { \kappa ( t ) } . } \end{array}\tag{64}
$$

Because the gates of Eqs. 81 and 82 read only the current value of their input path, $\alpha _ { t _ { k } } ^ { ( \ell ) } ( \hat { Z } ^ { ( \ell - 1 ) } )$ and $o _ { t _ { k } } ^ { ( \ell ) } ( \hat { Z } ^ { ( \ell - 1 ) } )$ are the gates evaluated on

$$
\begin{array} { r } { \hat { Z } _ { t _ { k } } ^ { ( \ell - 1 ) } = \bar { Z } _ { k } ^ { ( \ell - 1 ) } , } \end{array}
$$

that is, exactly the gates used by the discrete layer ℓ. The process $\bar { \vartheta }$ is progressively measurable, and it satisfies the bounds Eq. 32 because Assumption B.3(b)–(c) hold surely for every input path. Let $\check { Z }$ be the solution of the intermediate SDE with the frozen coeficients and the same driver and initial state,

$$
\begin{array} { r } { \mathrm { d } \check { Z } _ { t } = \bar { \mu } ( t , \check { Z } _ { t } ) \mathrm { d } t + \bar { \sigma } ( t , \check { Z } _ { t } ) \mathrm { d } W _ { t } ^ { ( \ell ) } , \qquad \check { Z } _ { 0 } = \iota _ { d } ( z _ { 0 } ) , } \end{array}\tag{65}
$$

where $\bar { \mu } , \bar { \sigma }$ are the maps Eq. 17 built from ${ \bar { \vartheta } } ;$ it exists, is unique and has finite moments of all orders by Lemma 2. The Euler–Maruyama scheme $\operatorname { E q } .$ . 46 for $\check { Z }$ uses the coeficients $\bar { \vartheta } _ { t _ { k } }$ , which are exactly the coeficients of the discrete layer $\ell ;$ hence its iterates are $\bar { Z } _ { k } ^ { ( \ell ) }$ and its interpolation is $\hat { Z } ^ { ( \ell ) }$ . We split

$$
Z ^ { ( \ell ) } - \hat { Z } ^ { ( \ell ) } = \big ( Z ^ { ( \ell ) } - \check { Z } \big ) + \big ( \check { Z } - \hat { Z } ^ { ( \ell ) } \big ) .\tag{66}
$$

Induction step $( i ) \colon { \check { Z } }$ versus $\hat { Z } ^ { ( \ell ) }$ . Lemma 5 applied to $\check { Z }$ at the exponent $q _ { \ell } .$ , whose hypotheses hold with $\Lambda _ { \star } ^ { \prime }$ in place of $\Lambda ^ { \prime } ,$ gives Eq. 54. Since $\bar { \vartheta } _ { s } = \bar { \vartheta } _ { \kappa ( s ) }$ for every s, the supremum in Eq. 54 vanishes, and

$$
\mathbb { E } \operatorname* { s u p } _ { t \leq T } \lVert \check { Z } _ { t } - \hat { Z } _ { t } ^ { ( \ell ) } \rVert ^ { q _ { \ell } } \leq C \Delta t ^ { q _ { \ell } / 2 } .\tag{67}
$$

Induction step $( i i ) \colon Z ^ { ( \ell ) }$ versus $\check { Z }$ . Both are continuous SDEs with the same driver and initial state. Set $e _ { t } : =$ $Z _ { t } ^ { ( \ell ) } - \check { Z } _ { t } ;$ then $e _ { 0 } = 0$ and

$$
e _ { t } = \int _ { 0 } ^ { t } \left[ \mu ( s , Z _ { s } ^ { ( \ell ) } ) - \bar { \mu } ( s , \check { Z } _ { s } ) \right] \mathrm { d } s + \int _ { 0 } ^ { t } \left[ \sigma ( s , Z _ { s } ^ { ( \ell ) } ) - \bar { \sigma } ( s , \check { Z } _ { s } ) \right] \mathrm { d } W _ { s } ^ { ( \ell ) } ,
$$

where $\mu , \sigma$ are built from ϑ. Writing $\vartheta _ { s } = ( A _ { s } , b _ { s } , \ldots )$ and $\bar { \vartheta } _ { s } = ( \bar { A } _ { s } , \bar { b } _ { s } , \ldots )$

$$
\mu ( s , Z _ { s } ^ { ( \ell ) } ) - \bar { \mu } ( s , \breve { Z } _ { s } ) = A _ { s } e _ { s } + ( A _ { s } - \bar { A } _ { s } ) \breve { Z } _ { s } + ( b _ { s } - \bar { b } _ { s } ) ,
$$

so that

$$
\begin{array} { r l } & { \| \mu ( s , Z _ { s } ^ { ( \ell ) } ) - \bar { \mu } ( s , \check { Z } _ { s } ) \| \leq ( 1 + \varepsilon ) a _ { * } \| e _ { s } \| + \| \vartheta _ { s } - \bar { \vartheta } _ { s } \| \big ( 1 + \| \check { Z } _ { s } \| \big ) , } \\ & { \quad \| \sigma ( s , Z _ { s } ^ { ( \ell ) } ) - \bar { \sigma } ( s , \check { Z } _ { s } ) \| _ { \mathrm { F } } \leq m c _ { * } \| e _ { s } \| + \| \vartheta _ { s } - \bar { \vartheta } _ { s } \| \big ( 1 + \| \check { Z } _ { s } \| \big ) . } \end{array}
$$

The argument of Step 2 of the proof of Lemma 5, with the forcing $\lVert \boldsymbol { \vartheta } _ { s } - \boldsymbol { \bar { \vartheta } } _ { s } \rVert ( 1 + \lVert \boldsymbol { \breve { Z } } _ { s } \rVert )$ in place of $D _ { s } + \Lambda ^ { \prime } I _ { s }$ , gives

$$
\mathbb { E } \operatorname* { s u p } _ { t \leq T } \Vert e _ { t } \Vert ^ { q _ { \varepsilon } } \leq C \int _ { 0 } ^ { T } \mathbb { E } \Big [ \Vert \vartheta _ { s } - \bar { \vartheta } _ { s } \Vert ^ { q _ { \varepsilon } } \big ( 1 + \Vert \check { Z } _ { s } \Vert \big ) ^ { q _ { \varepsilon } } \Big ] \mathrm { d } s \leq C \operatorname* { s u p } _ { s \leq T } \big ( \mathbb { E } \Vert \vartheta _ { s } - \bar { \vartheta } _ { s } \Vert ^ { 2 q _ { \varepsilon } } \big ) ^ { 1 / 2 } ,\tag{68}
$$

where the last step is the Cauchy–Schwarz inequality together with the moment bound $\operatorname { E q } .$ 38 for $\check { Z }$ at the exponent $2 q _ { \ell }$ . The coeficient diference splits into a time-regularity part and a gate-perturbation part,

$$
\begin{array} { r } { \| \vartheta _ { s } - \bar { \vartheta } _ { s } \| \leq \| \vartheta _ { s } - \vartheta _ { \kappa ( s ) } \| + \| \vartheta _ { \kappa ( s ) } - \bar { \vartheta } _ { \kappa ( s ) } \| . } \end{array}\tag{69}
$$

For the first part, the exact previous layer satisfies Eq. 34 at every exponent by Theorem 5.1, since $z _ { 0 }$ has moments

of all orders, so Lemma 1(b) gives $\operatorname { E q }$ . 24 for ϑ at the exponent $2 q _ { \ell }$ , and

$$
\begin{array} { r } { \left( \mathbb { E } \| \vartheta _ { s } - \vartheta _ { \kappa ( s ) } \| ^ { 2 q _ { \ell } } \right) ^ { 1 / 2 } \leq L _ { 2 q _ { \ell } } ^ { q _ { \ell } } | s - \kappa ( s ) | ^ { q _ { \ell } / 2 } \leq L _ { 2 q _ { \ell } } ^ { q _ { \ell } } \Delta t ^ { q _ { \ell } / 2 } . } \end{array}\tag{70}
$$

For the second part, at the grid time $t _ { k } = \kappa ( s )$ the two coeficient tuples difer only through the gates, evaluated on the paths $Z ^ { ( \ell - 1 ) }$ and $\hat { Z } ^ { ( \ell - 1 ) }$ up to time $t _ { k }$ . By Eq. 29 and the argument of Eq. 36,

$$
\begin{array} { r l } & { \| \vartheta _ { t _ { k } } - \bar { \vartheta } _ { t _ { k } } \| \leq a _ { * } \big \| \alpha _ { t _ { k } } ^ { ( \ell ) } \big ( Z ^ { ( \ell - 1 ) } \big ) - \alpha _ { t _ { k } } ^ { ( \ell ) } \big ( \hat { Z } ^ { ( \ell - 1 ) } \big ) \big \| _ { \infty } + \big \| \sigma _ { t _ { k } } ^ { ( \ell ) } \big ( Z ^ { ( \ell - 1 ) } \big ) - o _ { t _ { k } } ^ { ( \ell ) } \big ( \hat { Z } ^ { ( \ell - 1 ) } \big ) \big \| } \\ & { \qquad \leq \big ( 1 + a _ { * } \big ) L _ { g } \underset { r \leq T } { \operatorname* { s u p } } \big \| Z _ { r } ^ { ( \ell - 1 ) } - \hat { Z } _ { r } ^ { ( \ell - 1 ) } \big \| , } \end{array}
$$

so that, by the induction hypothesis and $2 q _ { \ell } = q _ { \ell - 1 }$ ，

$$
\begin{array} { r l } & { \left( \mathbb { E } \| \vartheta _ { \kappa ( s ) } - \bar { \vartheta } _ { \kappa ( s ) } \| ^ { 2 q \ell } \right) ^ { 1 / 2 } \leq \left( ( 1 + a _ { * } ) L _ { g } \right) ^ { q _ { \ell } } \left( \hat { e } _ { \ell - 1 } \right) ^ { 1 / 2 } } \\ & { \qquad \leq \left( ( 1 + a _ { * } ) L _ { g } \right) ^ { q _ { \ell } } C ^ { 1 / 2 } \Delta t ^ { q _ { \ell - 1 } / 4 } = C \Delta t ^ { q _ { \ell } / 2 } . } \end{array}\tag{71}
$$

Inserting Eqs. 70 and 71 into $\operatorname { E q } .$ . 68 through $\operatorname { E q . }$ . 69 and $( x + y ) ^ { 2 q _ { \ell } } \leq 2 ^ { 2 q _ { \ell } - 1 } ( x ^ { 2 q _ { \ell } } + y ^ { 2 q _ { \ell } } )$

$$
\mathbb { E } \operatorname* { s u p } _ { t \leq T } \lVert Z _ { t } ^ { ( \ell ) } - \check { Z } _ { t } \rVert ^ { q _ { \ell } } \leq C \Delta t ^ { q _ { \ell } / 2 } .\tag{72}
$$

Conclusion. By Eq. 66 and $\| x + y \| ^ { q _ { \ell } } \leq 2 ^ { q _ { \ell } - 1 } ( \| x \| ^ { q _ { \ell } } + \| y \| ^ { q _ { \ell } } )$ 2

$$
\hat { e } _ { \ell } \leq 2 ^ { q _ { \ell } - 1 } \Big ( \mathbb { E } \operatorname* { s u p } _ { t \leq T } \big \| Z _ { t } ^ { ( \ell ) } - \check { Z } _ { t } \big \| ^ { q _ { \ell } } + \mathbb { E } \operatorname* { s u p } _ { t \leq T } \big \| \check { Z } _ { t } - \hat { Z } _ { t } ^ { ( \ell ) } \big \| ^ { q _ { \ell } } \Big ) \leq C \Delta t ^ { q _ { \ell } / 2 }
$$

by Eqs. 67 and 72, which is $\operatorname { E q } .$ . 63 for ℓ and closes the induction.

Remark B.4 (Dependence on the depth). The constant $C _ { L , q }$ depends on L in two ways: the induction carries the moment order $q 2 ^ { L - \ell }$ at layer $\ell ,$ so the constants of Lemmas 4 and 5 enter at exponents that grow with the depth, and each inter-layer step multiplies the propagated error by a factor involving the gate Lipschitz constant $L _ { g }$ through Eq. 71. The theorem is therefore a statement at fixed depth: the rate $\frac { 1 } { 2 }$ holds for every L, but no uniformity of the constant in L is claimed. For the depths used in practice $( L = 2 , 3 )$ the exponents involved are 4q at most.

Remark B.5 (Strong order ${ \frac { 1 } { 2 } } .$ , not 1). Order 1 would require the iterated Itˆo integrals

$$
\int _ { t _ { k } } ^ { t _ { k + 1 } } \int _ { t _ { k } } ^ { s } \mathrm { d } W _ { u } ^ { i } \mathrm { d } W _ { s } ^ { j } , \qquad i \neq j ,
$$

and their L´evy areas, which the scheme does not simulate. We therefore claim only order ${ \frac { 1 } { 2 } } .$ , without assuming that the $C _ { t } ^ { j }$ commute; if they do, the exponential transition of Appendix C.1 attains order 1 and, in the diagonal case, is an exact one-step solution operator.

## B.5 Proof of Theorem 5.2

Proof. Throughout, ϵ is the m-dimensional Brownian innovation of Eq. 10, independent of the prefix driver W<sup>prefix</sup> and of $\mathcal { F } _ { 0 }$ , and $u _ { \theta , t }$ <sub>t</sub> is progressively measurable with respect to the final-layer filtration and satisfies Novikov’s condition Eq. 13. The stochastic exponential of Eq. 14 is taken with respect to $\epsilon ,$

$$
\mathcal { E } _ { t } ( u _ { \theta } ) : = \exp \Big ( \int _ { 0 } ^ { t } u _ { \theta , s } ^ { \top } \mathrm { d } \epsilon _ { s } - \frac { 1 } { 2 } \int _ { 0 } ^ { t } \| u _ { \theta , s } \| ^ { 2 } \mathrm { d } s \Big ) , \qquad t \in [ 0 , T ] .\tag{73}
$$

Step 0: the initial-state shift. Under $\mathbb { P } , z _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } )$ is independent of $( W ^ { \mathrm { p r e f i x } } , \epsilon )$ . The factor

$$
\mathrm { e x p } \big ( \mu _ { \boldsymbol { \theta } } ^ { \top } z _ { 0 } - \frac { 1 } { 2 } \| \mu _ { \boldsymbol { \theta } } \| ^ { 2 } \big ) = \frac { \mathrm { d } \mathcal { N } ( \mu _ { \boldsymbol { \theta } } , I _ { d } ) } { \mathrm { d } \mathcal { N } ( 0 , I _ { d } ) } ( z _ { 0 } )
$$

is $\mathcal { F } _ { 0 } .$ -measurable and has P-expectation one. Being independent of $\mathcal { E } _ { T } ( u _ { \theta } )$ under $\mathbb { P } ,$ its product with $\mathcal { E } _ { T } ( u _ { \theta } )$ is again a probability density, under $\mathbb { Q }$ the initial state has law $\mathcal { N } ( \mu _ { \theta } , I _ { d } )$ , and Steps 1–3 below, which argue conditionally on $\mathcal { F } _ { 0 }$ , apply verbatim. In the implementation

$$
z _ { 0 } = \mu _ { \theta } + \xi , \qquad \xi \sim \mathcal { N } ( 0 , I _ { d } ) ,
$$

and the term $\begin{array} { r } { \mu _ { \theta } ^ { \top } z _ { 0 } - \frac { 1 } { 2 } \| \mu _ { \theta } \| ^ { 2 } } \end{array}$ is added to the discrete log-ratio Eq. $7 6 ;$ its expectation under $\mathbb { Q } .$

$$
\begin{array} { r } { \mathbb { E } ^ { \mathbb { Q } } \big [ \mu _ { \theta } ^ { \top } z _ { 0 } - \frac { 1 } { 2 } \| \mu _ { \theta } \| ^ { 2 } \big ] = \frac { 1 } { 2 } \| \mu _ { \theta } \| ^ { 2 } , } \end{array}
$$

is the initial-state share of $\mathrm { K L } ( \mathbb { Q } | | \mathbb { P } )$ . When $\mu _ { \theta } = 0$ nothing changes.

Step $\mathit { 1 : \mathbb { Q } }$ is a probability measure and $\epsilon ^ { \mathbb { Q } }$ is a Q-Brownian motion. Under Novikov’s condition, $( \mathcal { E } _ { t } ( u _ { \theta } ) ) _ { t \leq T }$ is a strictly positive martingale (Protter, 2005, Ch. III), so

$$
\mathbb { E } ^ { \mathbb { P } } [ \mathcal { E } _ { T } ( u _ { \theta } ) ] = 1 .
$$

Hence

$$
\mathbb { Q } ( A ) : = \mathbb { E } ^ { \mathbb { P } } \big [ \mathcal { E } _ { T } ( u _ { \theta } ) \mathbf { 1 } _ { A } \big ] , \qquad A \in \mathcal { F } _ { T } ,
$$

is a probability measure equivalent to $\mathbb { P } ,$ with $\mathrm { d } \mathbb { Q } / \mathrm { d } \mathbb { P } = { \mathcal { E } } _ { T } ( u _ { \theta } )$ . By Girsanov’s theorem applied to the 2mdimensional Brownian motion $( W ^ { \mathrm { p r e f i x } } , \epsilon )$ , whose density process involves only the components of $\epsilon ,$ the process

$$
\left( W _ { t } ^ { \mathrm { p r e f i x } } , \ \epsilon _ { t } ^ { \mathbb { Q } } \right) _ { t \leq T } , \qquad \epsilon _ { t } ^ { \mathbb { Q } } : = \epsilon _ { t } - \int _ { 0 } ^ { t } u _ { \theta , s } ~ \mathrm { d } s ,\tag{74}
$$

is a 2m-dimensional Brownian motion under $\mathbb { Q } \colon$ the innovation is shifted by the control, while the prefix driver keeps its law and remains independent of the shifted innovation.

Step 2: the change-of-measure identity. For every ${ \mathcal { F } } _ { T }$ -measurable functional $G$ with $\mathbb { E } ^ { \mathbb { P } } | G | < \infty$ , the definition of $\mathbb { Q }$ gives

$$
\mathbb { E } ^ { \mathbb { P } } [ G ] = \mathbb { E } ^ { \mathbb { Q } } \Big [ G \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } \Big ] , \qquad \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } = \mathcal { E } _ { T } ( u _ { \theta } ) ^ { - 1 } .\tag{75}
$$

Substituting $\mathrm { d } \epsilon _ { s } = \mathrm { d } \epsilon _ { s } ^ { \mathbb { Q } } + u _ { \theta , s }$ ds in Eq. 73,

$$
\frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } = \exp \Big ( - \int _ { 0 } ^ { T } u _ { \theta , s } ^ { \top } \mathrm { d } \epsilon _ { s } ^ { \mathbb { Q } } - \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| u _ { \theta , s } \| ^ { 2 } \mathrm { d } s \Big ) .
$$

In particular

$$
\mathbb { E } ^ { \mathbb { Q } } \Big [ \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } \Big ] = \mathbb { P } ( \Omega ) = 1 :
$$

the importance weights are exactly normalised.

Step 3: the discrete implementation is exact. In the implementation the innovation is simulated on the grid. Under P the increments

$$
\Delta \epsilon _ { k } ^ { \mathbb { P } } , \qquad k = 0 , \dots , N - 1 ,
$$

are independent $\mathcal { N } ( 0 , \Delta t I _ { m } )$ vectors, and the control $u _ { \boldsymbol { \theta } , k }$ is $\mathcal { F } _ { t _ { k } }$ -measurable, being a function of $\{ \Delta \epsilon _ { j } ^ { \mathbb { P } } \} _ { j < k }$ and of prefix features up to $t _ { k }$ . Simulation under $\mathbb { Q }$ draws $\Delta \epsilon _ { k } ^ { \mathbb { Q } } \sim { \mathcal { N } } ( 0 , \Delta t I _ { m } )$ and sets

$$
\Delta \epsilon _ { k } ^ { \mathbb { P } } = \Delta \epsilon _ { k } ^ { \mathbb { Q } } + u _ { \theta , k } \Delta t ,
$$

which is Eq. 11. Conditionally on $\mathcal { F } _ { t _ { k } }$ , the P-law of $\Delta \epsilon _ { k } ^ { \mathbb { P } }$ has the density

$$
\varphi _ { \Delta t } ( x ) \propto \exp \Bigl ( - \frac { \| x \| ^ { 2 } } { 2 \Delta t } \Bigr ) ,
$$

and its Q-law has the density $\varphi _ { \Delta t } ( x - u _ { \theta , k } \Delta t )$ . Their ratio is

$$
\frac { \varphi _ { \Delta t } ( x - u _ { \theta , k } \Delta t ) } { \varphi _ { \Delta t } ( x ) } = \exp \Big ( u _ { \theta , k } ^ { \top } x - \frac { 1 } { 2 } \| u _ { \theta , k } \| ^ { 2 } \Delta t \Big ) .
$$

Multiplying the conditional density ratios over k gives the likelihood ratio of the two laws of the whole increment sequence,

$$
\log \frac { \mathrm { d } \mathbb { Q } } { \mathrm { d } \mathbb { P } } = \sum _ { k = 0 } ^ { N - 1 } \Big ( \boldsymbol { u } _ { \boldsymbol { \theta } , k } ^ { \top } \Delta \boldsymbol { \epsilon } _ { k } ^ { \mathbb { P } } - \frac { 1 } { 2 } \| \boldsymbol { u } _ { \boldsymbol { \theta } , k } \| ^ { 2 } \Delta t \Big ) ,\tag{76}
$$

$$
\log \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } = - \sum _ { k = 0 } ^ { N - 1 } u _ { \theta , k } ^ { \top } \Delta \epsilon _ { k } ^ { \mathbb { Q } } - \frac { 1 } { 2 } \sum _ { k = 0 } ^ { N - 1 } \| u _ { \theta , k } \| ^ { 2 } \Delta t ,
$$

which is Eq. 12: on the grid the weight is an exact density ratio, not an approximation of $\operatorname { E q }$ . 75. It is also exactly normalised, because for a fixed $\mathcal { F } _ { t _ { k } }$ -measurable $u _ { \theta , k }$ and an independent Gaussian increment

$$
\begin{array} { r } { \mathbb { E } ^ { \mathbb { P } } \Big [ \exp \big ( \boldsymbol { u } _ { \boldsymbol { \theta } , k } ^ { \top } \Delta \boldsymbol { \epsilon } _ { k } ^ { \mathbb { P } } - \frac { 1 } { 2 } \| \boldsymbol { u } _ { \boldsymbol { \theta } , k } \| ^ { 2 } \Delta t \big ) \Big | \mathcal { F } _ { t _ { k } } \Big ] = 1 , } \end{array}
$$

and the tower property gives

$$
\mathbb { E } ^ { \mathbb { P } } \Big [ \frac { \mathrm { d } \mathbb { Q } } { \mathrm { d } \mathbb { P } } \Big ] = 1
$$

without any integrability condition on the control beyond measurability.

Step $\it 4 :$ locality of the ratio. The prefix layers are functionals of $z _ { 0 }$ and of the prefix driver. By Steps 0–1 the law of $W ^ { \mathrm { p r e f i x } }$ is the same under $\mathbb { P }$ and $\mathbb { Q } ,$ the law of $z _ { 0 }$ changes only through the shift $\mu _ { \theta }$ , and both are independent of $\epsilon { : }$ the prefix computation is the same map under both measures, evaluated at a shifted starting point. The final layer is a functional of the prefix path and of ϵ, and Eq. 75 together with Step 0 shows that the likelihood ratio is a functional of the control, of $z _ { 0 }$ and of the shifted innovation only. Hence the ratio depends only on the final-layer control, the initial-state shift and on the Brownian innovation being shifted. □

## B.6 Proof of Theorem 5.3

Proof. Let $\mu \in { \mathcal { P } } _ { 2 } \left( \mathbb { R } ^ { p } \right)$ and let $\eta > 0$

Step 1: construction of an atomless scalar feature. Choose a latent width $d ,$ to be specified below, and configure the first layer as

$$
\mathrm { d } Z _ { t } ^ { ( 1 ) } = \beta e _ { 1 } ~ \mathrm { d } W _ { t } ^ { 1 } , \qquad \beta \neq 0 ,
$$

where $e _ { 1 }$ is the first coordinate vector. Thus

$$
\begin{array} { r } { Z _ { t } ^ { ( 1 ) } = \iota _ { d } ( z _ { 0 } ) + \beta e _ { 1 } W _ { t } ^ { 1 } . } \end{array}
$$

In particular,

$$
S _ { T } : = \big \langle e _ { 1 } , Z _ { T } ^ { ( 1 ) } \big \rangle = \big \langle e _ { 1 } , \iota _ { d } ( z _ { 0 } ) \big \rangle + \beta W _ { T } ^ { 1 } .
$$

Since $W _ { T } ^ { 1 }$ is independent of ${ \mathcal { F } } _ { 0 } .$ , the law of $S _ { T }$ is the convolution of the law of $\langle e _ { 1 } , \iota _ { d } ( z _ { 0 } ) \rangle$ with a nondegenerate Gaussian law. Consequently, $S _ { T }$ has a smooth density, irrespective of whether the law of $z _ { \mathrm { 0 } }$ is degenerate.

Recall the normalisation of Eq. 19,

$$
\Pi ( z ) = \gamma \odot \frac { z } { \sqrt { \varsigma + d ^ { - 1 } \| z \| ^ { 2 } } } , \qquad \varsigma > 0 ,
$$

and choose the first gain $\gamma _ { 1 }$ to be 1. Set

$$
R _ { t } : = \left[ \Pi ( Z _ { t } ^ { ( 1 ) } ) \right] _ { 1 } .
$$

Conditionally on $z _ { 0 } , R _ { T }$ is a strictly monotone function of $W _ { T } ^ { 1 }$ . Indeed, since $\iota _ { d } ( z _ { 0 } ) = ( z _ { 0 } , 0 , \ldots , 0 )$ , for fixed $z _ { 0 }$ the map

$$
u \longmapsto \gamma _ { 1 } { \frac { u } { \sqrt { \varsigma + d ^ { - 1 } \left( u ^ { 2 } + \sum _ { j = 2 } ^ { d _ { 0 } } z _ { 0 , j } ^ { 2 } \right) } } }
$$

has derivative

$$
\gamma _ { 1 } \frac { \varsigma + d ^ { - 1 } \sum _ { j = 2 } ^ { d _ { 0 } } z _ { 0 , j } ^ { 2 } } { \left[ \varsigma + d ^ { - 1 } \left( u ^ { 2 } + \sum _ { j = 2 } ^ { d _ { 0 } } z _ { 0 , j } ^ { 2 } \right) \right] ^ { 3 / 2 } } ,
$$

which never vanishes. Hence $R _ { T }$ is atomless. Moreover, R has continuous paths and is bounded.

Step 2: reduction to a finitely supported target law. Finitely supported probability measures are dense in $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { p } )$ We may therefore choose

$$
\nu = \sum _ { i = 1 } ^ { N } p _ { i } \delta _ { y _ { i } } , \qquad p _ { i } > 0 , \qquad \sum _ { i = 1 } ^ { N } p _ { i } = 1 ,
$$

such that

$$
W _ { 2 } ( \mu , \nu ) < \frac { \eta } { 3 } .
$$

Let $F _ { R }$ be the distribution function of $R _ { T }$ , and define

$$
P _ { i } : = \sum _ { j = 1 } ^ { i } p _ { j } , \qquad i = 1 , \dots , N - 1 .
$$

Since $R _ { T }$ is atomless, $F _ { R }$ is continuous. We may therefore choose

$$
q _ { 1 } < \cdots < q _ { N - 1 }
$$

such that

$$
F _ { R } ( q _ { i } ) = P _ { i } .
$$

With $q _ { 0 } = - \infty$ and $q _ { N } = + \infty$ , define

$$
H ( r ) : = y _ { i } \quad { \mathrm { w h e n e v e r } } \quad q _ { i - 1 } < r \leq q _ { i } .
$$

It follows that

$$
\mathcal { L } \big ( H ( R _ { T } ) \big ) = \nu .
$$

Equivalently,

$$
H ( r ) = y _ { 1 } + \sum _ { i = 1 } ^ { N - 1 } ( y _ { i + 1 } - y _ { i } ) \mathbf { 1 } _ { \{ r > q _ { i } \} } .
$$

For $a > 0$ , define the sigmoid approximation

$$
H _ { a } ( r ) : = y _ { 1 } + \sum _ { i = 1 } ^ { N - 1 } ( y _ { i + 1 } - y _ { i } ) \sigma { \big ( } a ( r - q _ { i } ) { \big ) } .
$$

Since

$$
\sigma ( a ( r - q _ { i } ) ) \longrightarrow \mathbf { 1 } _ { \{ r > q _ { i } \} }
$$

for every $r \neq q _ { i }$ , and $\mathbb { P } ( R _ { T } = q _ { i } ) = 0$ , we have

$$
H _ { a } ( R _ { T } ) \longrightarrow H ( R _ { T } ) \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

The family $\{ H _ { a } ( R _ { T } ) : a > 0 \}$ is uniformly bounded. Hence, by dominated convergence,

$$
\| H _ { a } ( R _ { T } ) - H ( R _ { T } ) \| _ { L ^ { 2 } } \longrightarrow 0 .
$$

Choose $a > 0$ such that

$$
\| H _ { a } ( R _ { T } ) - H ( R _ { T } ) \| _ { L ^ { 2 } } < \frac { \eta } { 3 } .
$$

Step 3: realisation by the gated layer. Choose $d \geq N$ . Disable the scale gate by taking $W _ { i F } ^ { ( 2 ) } = 0$ , together with a zero bias in that branch, so that

$$
\alpha _ { t } ^ { ( 2 ) } = { \bf 1 } ,
$$

and set all difusion coeficients of the second layer equal to zero. Choose its diagonal drift matrix to be

$$
A ^ { ( 2 ) } = - \lambda I _ { d } , \qquad \lambda > 0 .
$$

Use one coordinate $V ^ { 0 , \lambda }$ to generate the constant feature and $N - 1$ coordinates $V ^ { i , a , \lambda }$ to generate the shifted sigmoid features:

$$
\begin{array} { r l } & { \displaystyle \mathrm { d } V _ { t } ^ { 0 , \lambda } = - \lambda V _ { t } ^ { 0 , \lambda } \mathrm { ~ d } t + \lambda \frac { t } { T } \mathrm { ~ d } t , } \\ & { \displaystyle \mathrm { d } V _ { t } ^ { i , a , \lambda } = - \lambda V _ { t } ^ { i , a , \lambda } \mathrm { ~ d } t + \lambda \frac { t } { T } \sigma \left( a R _ { t } - a q _ { i } \frac { t } { T } \right) \mathrm { ~ d } t , \quad \quad i = 1 , \dots , N - 1 . } \end{array}
$$

These dynamics are contained in the stated ofset-gate parameterisation. Indeed, because the gate input contains both $R _ { t }$ and t, suitable rows of the gate matrices give

$$
W _ { g o , i } ^ { ( 2 ) } x _ { t } ^ { ( 1 ) } = a R _ { t } - a q _ { i } \frac { t } { T } , \qquad W _ { i o , i } ^ { ( 2 ) } x _ { t } ^ { ( 1 ) } = \lambda \frac { t } { T } .
$$

For the constant coordinate, take

$$
W _ { g o , 0 } ^ { ( 2 ) } x _ { t } ^ { ( 1 ) } = 0 , \qquad W _ { i o , 0 } ^ { ( 2 ) } x _ { t } ^ { ( 1 ) } = 2 \lambda \frac { t } { T } ,
$$

and use $\sigma ( 0 ) = 1 / 2$

We claim that, as $\lambda \to \infty$ ,

$$
\begin{array} { r l } { V _ { T } ^ { 0 , \lambda } \longrightarrow 1 , } & { { } \qquad } \\ { V _ { T } ^ { i , a , \lambda } \longrightarrow \sigma \big ( a ( R _ { T } - q _ { i } ) \big ) , \qquad } & { { } i = 1 , \ldots , N - 1 , } \end{array}
$$

in $L ^ { 2 } .$ To see this, let $\phi$ be any bounded process which is continuous in $L ^ { 2 }$ at $T ,$ . The solution of

$$
\mathrm { d } U _ { t } ^ { \lambda } = - \lambda U _ { t } ^ { \lambda } \mathrm { ~ d } t + \lambda \phi _ { t } \mathrm { ~ d } t
$$

satisfies

$$
U _ { T } ^ { \lambda } = e ^ { - \lambda T } U _ { 0 } ^ { \lambda } + \int _ { 0 } ^ { T } \lambda e ^ { - \lambda ( T - s ) } \phi _ { s } ~ \mathrm { d } s .
$$

Consequently,

$$
\begin{array} { r l } & { \left\| { U } _ { T } ^ { \lambda } - \phi _ { T } \right\| _ { L ^ { 2 } } \leq e ^ { - \lambda T } \left( \left\| { U } _ { 0 } ^ { \lambda } \right\| _ { L ^ { 2 } } + \left\| \phi _ { T } \right\| _ { L ^ { 2 } } \right) } \\ & { \qquad + \displaystyle \int _ { 0 } ^ { T } \lambda e ^ { - \lambda ( T - s ) } \left\| \phi _ { s } - \phi _ { T } \right\| _ { L ^ { 2 } } \mathrm { d } s , } \end{array}
$$

which converges to zero. Applying this observation with

$$
\phi _ { t } = \frac { t } { T }
$$

and

$$
\phi _ { t } = \frac { t } { T } \sigma \left( a R _ { t } - a q _ { i } \frac { t } { T } \right)
$$

proves the claim.

Choose the terminal readout so that

$$
Y _ { T } ^ { a , \lambda } = y _ { 1 } V _ { T } ^ { 0 , \lambda } + \sum _ { i = 1 } ^ { N - 1 } ( y _ { i + 1 } - y _ { i } ) V _ { T } ^ { i , a , \lambda } .
$$

Then

$$
Y _ { T } ^ { a , \lambda } \longrightarrow H _ { a } ( R _ { T } ) \qquad \mathrm { i n ~ } L ^ { 2 }
$$

as $\lambda \to \infty$ . Choose λ suficiently large that

$$
\Bigl \| Y _ { T } ^ { a , \lambda } - H _ { a } ( R _ { T } ) \Bigr \| _ { L ^ { 2 } } < \frac { \eta } { 3 } .
$$

Finally, using the coupling inequality

$$
W _ { 2 } \big ( \mathcal { L } ( X ) , \mathcal { L } ( X ^ { \prime } ) \big ) \leq \| X - X ^ { \prime } \| _ { L ^ { 2 } } ,
$$

we obtain

$$
\begin{array} { r l } & { W _ { 2 } \big ( \mathcal { L } ( Y _ { T } ^ { a , \lambda } ) , \mu \big ) \leq \Big \| Y _ { T } ^ { a , \lambda } - H _ { a } ( R _ { T } ) \Big \| _ { L ^ { 2 } } } \\ & { \qquad + \left\| H _ { a } ( R _ { T } ) - H ( R _ { T } ) \right\| _ { L ^ { 2 } } + W _ { 2 } ( \nu , \mu ) < \eta . } \end{array}
$$

Since $\mu$ and η were arbitrary, the result follows.

Step 4: discretised implementation. Once a and λ have been fixed, the configuration of Steps 1–3 is a fixed two-layer model, and its discretisation on a grid π converges. The first layer has constant coeficients and additive noise only, so its Euler–Maruyama iterates are exact,

$$
\bar { Z } _ { k } ^ { ( 1 ) } = \iota _ { d } ( z _ { 0 } ) + \beta e _ { 1 } W _ { t _ { k } } ^ { 1 } ,
$$

and the discrete gate of the second layer therefore reads the exact feature $R _ { t _ { k } }$ at the grid points. The second layer has no difusion, so, pathwise, it is the linear ordinary diferential equation

$$
\mathrm { d } U _ { t } ^ { \lambda } = - \lambda U _ { t } ^ { \lambda } \mathrm { ~ d } t + \lambda \phi _ { t } \mathrm { ~ d } t
$$

of Step 3, with a forcing ϕ that is bounded and continuous in t. Its discretisation is the explicit Euler scheme, which converges to $U ^ { \lambda }$ uniformly on [0, T] along every path as $| \pi | \to 0$ , and which is bounded by $\left. z _ { 0 } \right. + \lambda T \operatorname* { s u p } _ { t } \left. \phi _ { t } \right.$ once $| \pi | \leq 1 / \lambda$ . By dominated convergence, the discretised terminal output $Y _ { T } ^ { a , \lambda , \pi }$ satisfies

$$
\left\| Y _ { T } ^ { a , \lambda , \pi } - Y _ { T } ^ { a , \lambda } \right\| _ { L ^ { 2 } } \longrightarrow 0 \qquad \mathrm { a s } \qquad | \pi | \longrightarrow 0 .
$$

When $z _ { \mathrm { 0 } }$ has finite moments of all orders this is also a special case of Theorem B.1. Then the same density conclusion holds for the discretised architecture when the mesh is allowed to vary. Indeed,

$$
\begin{array} { r } { W _ { 2 } \left( \mathcal { L } ( Y _ { T } ^ { a , \lambda , \pi } ) , \mathcal { L } ( Y _ { T } ^ { a , \lambda } ) \right) \leq \left\| Y _ { T } ^ { a , \lambda , \pi } - Y _ { T } ^ { a , \lambda } \right\| _ { L ^ { 2 } } , } \end{array}
$$

and the discretisation is refined only after the finite parameters a and λ have been selected.

## C Implementation Details

This appendix expands the implementation choices summarised in the main text. All reported SLiSDE results use the gated in-flow stack (Section 3.3) with structured block-diagonal base layers, an always-on RMS normalisation of the gate input, optional causal convolutions, and time-feature augmentation; residual stacking is not used. The Girsanov tilt described in Section 4 is applied only to the final layer, with the correlated-driver construction of Eq. 10.

## C.1 Discretisation of a linear layer

On the interval $[ t _ { k } , t _ { k + 1 } ]$ , the afine linear SDE Eq. 2 admits the exponential approximation

$$
\begin{array} { c } { { \displaystyle Z _ { k + 1 } = F _ { k } Z _ { k } + g _ { k } , \qquad g _ { k } = b \Delta t + D \Delta W _ { k } , } } \\ { { \displaystyle F _ { k } = \exp \left( \left( A - \frac 1 2 \sum _ { j = 1 } ^ { m } ( C ^ { j } ) ^ { 2 } \right) \Delta t + \sum _ { j = 1 } ^ { m } C ^ { j } \Delta W _ { k } ^ { j } \right) . } } \end{array}\tag{77}
$$

The correction is the Itˆo correction. If $A , C ^ { 1 } , \ldots , C ^ { m }$ commute pairwise, then $F _ { k }$ is the exact flow of the homogeneous Itˆo equation; otherwise it is an exponential approximation. The afine term $g _ { k }$ remains a first-order approximation. To avoid computing matrix exponentials, we use the Euler–Maruyama transition

$$
Z _ { k + 1 } = \left( I + A \Delta t + \sum _ { j = 1 } ^ { m } C ^ { j } \Delta W _ { k } ^ { j } \right) Z _ { k } + b \Delta t + D \Delta W _ { k } ,\tag{78}
$$

which preserves the afine recurrence structure and is used throughout our experiments. With the time-dependent coeficients of Appendix C.2, the static coeficients are replaced by their values at $t _ { k }$ in both transitions; the strong error of the Euler–Maruyama scheme is analysed in Appendix B.4.

## C.2 Time-dependent coeficient map

Each base layer (Section 3.1) stores static structured coeficients $( A , b , C ^ { 1 } , \ldots , C ^ { m } , d ^ { 1 } , \ldots , d ^ { m } )$ and can be made time-dependent through a small deterministic decoder. For each grid time $t _ { k }$ we form

$$
\tau ( t _ { k } ) = \big ( 1 , t _ { k } , t _ { k } ^ { 2 } , \sin ( \omega t _ { k } ) , \cos ( \omega t _ { k } ) \big ) , \qquad h ( t _ { k } ) = \operatorname { t a n h } \bigl ( U \tau ( t _ { k } ) + c \bigr ) ,
$$

and add structure-aware corrections to the free coeficient entries:

$$
A _ { t _ { k } } = A + \Delta A \big ( h ( t _ { k } ) \big ) , \quad b _ { t _ { k } } = b + \Delta b \big ( h ( t _ { k } ) \big ) , \quad C _ { t _ { k } } ^ { j } = C ^ { j } + \Delta C ^ { j } \big ( h ( t _ { k } ) \big ) .
$$

For additive difusion we use a per-Brownian-channel time gate on the dense difusion matrix, equivalently producing $d _ { t _ { k } } ^ { j }$ from $d ^ { j }$ and $h ( t _ { k } )$ . The decoders respect the chosen matrix structure: diagonal layers decode only diagonal entries, block-diagonal layers decode only block entries, and dense layers decode full matrices. Decoders are zeroinitialised, so the model starts from the time-homogeneous structured SDE and learns time variation only when useful.

## C.3 Feature conditioning in gated layers

The gated layer consumes the previous layer path at the same time grid as the current layer. The simplest gate input is $\mathrm { N o r m } ( Z _ { k } ^ { ( \ell - 1 ) } )$ , where $Z _ { k } ^ { ( \ell - 1 ) }$ is the model output after $\ell - 1$ layers and the normalisation is RMSNorm or LayerNorm (RMSNorm by default, applied at every gated layer). Note that normalisation controls the scale of the gate input but does not by itself bound the paths; the ofset gate is nevertheless bounded for every fixed choice of its weights, which is what the analysis uses (Appendix B.1). In the implementation this signal can be enriched in two ways.

Causal convolution (for gated-in-flow stacking only). Before normalisation, the previous-layer path can be passed through a left-padded depthwise causal convolution along the time axis. The value at time $t _ { k }$ depends only on previous-layer states up to $t _ { k } .$ , so the gate is adapted and does not leak future information. The convolution is wrapped in a residual update and its kernel is small at initialisation, making the initial behaviour close to the no-convolution case while still allowing the gate to learn short temporal summaries.

Time features. The deterministic vector

$$
\tau _ { \mathrm { g a t e } } ( t ) = ( t , t ^ { 2 } , \sin ( \omega t ) , \cos ( \omega t ) )\tag{79}
$$

can be concatenated to the gate input after convolution and after normalisation. Appending time features after normalisation preserves their scale, while keeping the convolution focused on the stochastic path rather than on deterministic batch-constant signals. With both refinements, the gate input is

$$
x _ { k } = \left[ \mathrm { N o r m } \left( \mathrm { C a u s a l C o n v } ( H ^ { ( \ell - 1 ) } ) _ { k } \right) , t _ { k } , t _ { k } ^ { 2 } , \sin ( \omega t _ { k } ) , \cos ( \omega t _ { k } ) \right] .\tag{80}
$$

## C.4 Gated in-flow implementation

The paper focuses on the gated in-flow stacking. Given the conditioned feature $x _ { k } ^ { ( \ell - 1 ) }$ from Eq. 80, the layer constructs a bounded diagonal scale

$$
\alpha _ { k } ^ { ( \ell ) } = \mathbf { 1 } + \varepsilon \sigma \big ( W _ { g F } ^ { ( \ell ) } x _ { k } ^ { ( \ell - 1 ) } \big ) \odot \mathrm { t a n h } \big ( W _ { i F } ^ { ( \ell ) } x _ { k } ^ { ( \ell - 1 ) } \big ) ,\tag{81}
$$

and an afine residual ofset

$$
o _ { k } ^ { ( \ell ) } = \sigma \big ( W _ { g o } ^ { ( \ell ) } x _ { k } ^ { ( \ell - 1 ) } \big ) \odot \big ( W _ { i o } ^ { ( \ell ) } x _ { k } ^ { ( \ell - 1 ) } + c _ { i o } ^ { ( \ell ) } \big ) .\tag{82}
$$

All gate maps carry trainable biases, omitted from the displays except $c _ { i o } ^ { ( \ell ) }$ . The tanh bound gives $\alpha _ { k } ^ { ( \ell ) } \in ( 1 - \varepsilon , 1 + \varepsilon )$ coordinatewise. The scale $\alpha _ { k } ^ { ( \ell ) }$ is applied only to the diagonal entries of the structured drift matrix $F _ { k } ^ { ( \ell ) }$ , and the ofset $o _ { k } ^ { ( \ell ) }$ is added to the afine intercept $g _ { k } ^ { ( \ell ) }$ . Gate kernels are initialised at small scale, so initially $\alpha _ { k } ^ { ( \ell ) }$ ≈ 1 and $o _ { k } ^ { ( \ell ) } \approx \mathbf { 0 } \colon$ the stacked model starts close to an unmodulated structured SDE and learns cross-layer coupling gradually. We fix $\varepsilon = 0 . 1$ throughout.

The ofset $o _ { k } ^ { ( \ell ) }$ is an unbounded GLU: the gate $\sigma ( W _ { g o } ^ { ( \ell ) } x _ { k } ^ { ( \ell - 1 ) } )$ is bounded, while the value branch $W _ { i o } ^ { ( \ell ) } x _ { k } ^ { ( \ell - 1 ) } + c _ { i o } ^ { ( \ell ) }$ is linear in the gate input. Because the gate input is normalised and therefore lies in a ball of radius $\Lambda _ { x } , \| o _ { k } ^ { ( \ell ) } \| \leq$ $\| W _ { i o } ^ { ( \ell ) } \| \Lambda _ { x } + \| c _ { i o } ^ { ( \ell ) } \|$ for every fixed choice of weights, which is the boundedness used in Appendix $\operatorname { B } ;$ the magnitude itself is free, which is what the proof of Theorem 5.3 exploits. The drift scale $\alpha _ { k } ^ { ( \ell ) }$ is applied only to the diagonal entries of the structured transition matrix $F _ { k } ^ { ( \ell ) }$ (a diagonal, row-wise Hadamard action), while $o _ { k } ^ { ( \ell ) }$ is added directly to the afine intercept $g _ { k } ^ { ( \ell ) }$ with no additional $\Delta t$ factor. Unless a prior is learned, the shared initial state is $Z _ { 0 } \sim { \mathcal { N } } ( 0 , I _ { d } )$ , sampled once and reused by every layer.

## C.5 Last-layer Girsanov tilt: design details

This subsection collects the design choices behind Section 4.

Why the initial state is tilted. The innovation of the final layer is not the only source of randomness the rare event depends on. In a gated stack the random initial latent $z _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } )$ carries a large share of the terminal variance – on the toy model of Section 6 the final-layer innovation explains only about a quarter of $\mathrm { V a r } ( Y _ { T } )$ – and a controller acting on the innovation alone cannot reach the rare region: every such variant we trained kept the hit rate of the tilted paths at the plain Monte-Carlo level for $p \leq 1 0 ^ { - 3 }$ . Under $\mathbb { Q } , z _ { 0 } = \mu _ { \theta } + \xi$ with $\xi \sim \mathcal { N } ( 0 , I _ { d } )$ and a learned $\mu _ { \theta } \in \mathbb { R } ^ { d }$ (one per controller), which contributes the closed-form factor $\begin{array} { r } { \mu _ { \theta } ^ { \top } z _ { 0 } - \frac { 1 } { 2 } \| \mu _ { \theta } \| ^ { 2 } } \end{array}$ of Eq. 12 to the log-likelihood ratio and $\frac { 1 } { 2 } \lVert \mu _ { \theta } \rVert ^ { 2 }$ to $\mathrm { K L } ( \mathbb { Q } | | \mathbb { P } )$ ). The prefix driver $W ^ { \mathrm { p r e f i x } }$ keeps its law, so the prefix layers are computed by the same map under both measures; only the distribution of their starting point moves.

Controller. The control $u _ { \boldsymbol { \theta } , k }$ is an adapted function of the reference innovation history $\{ \mathrm { d } \epsilon _ { j } ^ { \mathbb { P } } \} _ { j < k }$ and of stopgradient features of the prefix latent path $\{ L _ { j } ^ { ( L - 1 ) } \} _ { j \leq k } ;$ it is parameterised by the diagonal state-space recurrence of Appendix C.6, driven by both inputs, with recurrence rates constrained to (0, 1) and zero drift at initialisation.

Objectives and gradient convention. Backbone and controller are trained on diferent objectives with separate optimisers. The backbone minimises the calibration loss, in which each far-tail term is estimated on the tilted batch by self-normalised importance sampling (Appendix C.7), combined with the plain estimate on the reference batch in proportion to the two efective sample sizes. The controller is trained by the cross-entropy method: it minimises a Monte-Carlo estimate of $\operatorname { K L } ( \mathbb { Q } ^ { \star } \| \mathbb { Q } _ { \theta } )$ , where $\mathbb { Q } ^ { \star } \propto | G | \mathbb { P }$ is the zero-variance proposal of the rare-event functional, under a quadratic wall on $\operatorname { K L } ( \mathbb { Q } _ { \theta } \| \mathbb { P } )$ that keeps the proposal in the regime where the weights remain usable (Appendix C.8). The gradient convention matters: the cross-entropy objective is a score-function estimator and requires the log-ratio $\operatorname { E q . }$ 12 to be diferentiated with the sampled path held fixed, whereas Monte-Carlo functionals of the weights, such as the KL wall, require the reparameterised form in which the tilted noise is held fixed; the two forms have the same value and diferent fixed points, and using the reparameterised one for the cross-entropy objective drives the tilt away without bound.

Which estimator. Because the ratio is normalised, the vanilla importance-sampling estimator $n _ { q } ^ { - 1 } \sum _ { i } w _ { i } G ( X ^ { ( i ) } )$ is unbiased. For the calibration loss we use the self-normalised estimator (Appendix C.7): the weights are computed pathwise in log-space, self-normalisation is invariant to a common log-shift, so a max-shift can be applied before exponentiation and no single extreme path can dominate through overflow, the ratio form cancels weight fluctuations shared by numerator and denominator, and its $O ( 1 / n _ { q } )$ bias is negligible at our operating point. For rare-event probabilities the vanilla estimator is the better one: the initial-state shift gives large weights to tilted paths that do not reach the event, and those paths enter only the normalisation of the self-normalised estimator. Both estimators are compared in Appendix E.

## C.6 Girsanov controller

The controller returns an adapted sequence $u _ { \theta , k } \in \mathbb { R } ^ { m }$ that enters the Radon–Nikodym formula Eq. 12. It is a diagonal state-space recurrence with two branches, one driven by the tilted innovation and one by the stop-gradient

prefix latent path,

$$
\begin{array} { r l } & { s _ { k + 1 } ^ { W } = \mathrm { d i a g } ( r _ { W } ) s _ { k } ^ { W } + U _ { W } \Delta \epsilon _ { k } ^ { \mathbb Q } , \qquad s _ { k + 1 } ^ { Z } = \mathrm { d i a g } ( r _ { Z } ) s _ { k } ^ { Z } + U _ { Z } L _ { k } ^ { ( L - 1 ) } , } \\ & { ~ u _ { \theta , k } = W _ { \mathrm { m i x } } \big [ s _ { k } ^ { W } , s _ { k } ^ { Z } \big ] + c , } \end{array}
$$

so that $u _ { \boldsymbol { \theta } , k }$ depends on the increments before step k only. The recurrence is itself afine and can be scanned in parallel.

Implementation details. Three details proved necessary.

(i) The recurrence rates of both branches are parameterised as $\exp ( - \operatorname { s o f t p l u s } ( - r ) ) \in ( 0 , 1 )$ , initialised at 0.9997, because an unconstrained rate can drift above one, after which the recurrence grows geometrically over the horizon and overflows, producing non-finite weights.

(ii) The mixing weights of the latent-path branch are initialised at zero, so that the untrained controller is the identity tilt with KL $( \mathbb { Q } \| \mathbb { P } ) = 0 ;$ with a small random initialisation the branch, which integrates the $O ( 1 )$ prefix latent over the whole horizon, gives the untrained controller a divergence of about 8 nats per controller, i.e. a collapsed proposal before any training.

(iii) The initial-state shift $\mu _ { \theta }$ is a free vector per controller initialised at zero.

Controller parameters are trained by Adam with learning rate $2 \cdot 1 0 ^ { - 3 }$ and gradient clipping at norm one, separately from the backbone optimiser: with a single optimiser and a shared global-norm clip, the cross-entropy gradient, three orders of magnitude larger than the calibration gradient, scaled the backbone updates to zero after the controller was activated.

## C.7 Self-normalised importance sampling and ESS

Given tilted samples $\{ X ^ { ( i ) } \} _ { i = 1 } ^ { n _ { q } }$ drawn under $\mathbb { Q }$ with unnormalised weights

$$
w _ { i } = \frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } ( X ^ { ( i ) } ) ,
$$

the self-normalised importance-sampling (SNIS) estimator of $\mathbb { E } ^ { \mathbb { P } } [ G ]$ is

$$
\widehat { \mathbb { E } } ^ { \mathbb { P } , \mathrm { S N I S } } [ G ] = \frac { \sum _ { i = 1 } ^ { n _ { q } } w _ { i } G ( X ^ { ( i ) } ) } { \sum _ { i = 1 } ^ { n _ { q } } w _ { i } } .\tag{83}
$$

The estimator is biased at finite $n _ { q }$ but consistent. We monitor the quality of the tilted sample through the efective sample size

$$
\mathrm { E S S } = \frac { \left( \sum _ { i = 1 } ^ { n _ { q } } w _ { i } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { n _ { q } } w _ { i } ^ { 2 } } \in [ 1 , n _ { q } ] .\tag{84}
$$

$\mathrm { E S S } = n _ { q }$ if all weights are equal; ESS = 1 if one weight dominates. In practice we compute the log-weights log $w _ { i } = \log (  { \mathrm { d } } \mathbb { P } /  { \mathrm { d } } \mathbb { Q } ) ( X ^ { ( i ) } )$ directly from the discrete Radon–Nikodym density Eq. 12 and normalise them with a numerically stable log-sum-exp, so that the SNIS estimator Eq. 83 and the ESS ratio remain well behaved even when a few tilted paths dominate. In training logs we report the ratio ESS $/ n _ { q } \in [ 1 / n _ { q } , 1 ]$ for comparability across batch sizes; a controller that drives ESS $/ n _ { q }$ toward $1 / n _ { q }$ signals weight collapse and is held back by the KL wall of Appendix C.8. For rare-event probabilities we report the vanilla estimator $n _ { q } ^ { - 1 } \sum _ { i } w _ { i } G ( X ^ { ( i ) } )$ as the primary one: with the initial-state shift, tilted paths that do not reach the event can carry large weights, which afect only the denominator of Eq. 83 and make the self-normalised estimator markedly worse (Table 11). Its truncated variant caps the weights at ${ \sqrt { n _ { q } } } \mathbb { E } ^ { \mathbb { Q } } [ w ] = { \sqrt { n _ { q } } }$ , using the known mean rather than the sample mean, which the very weights to be tamed would inflate.

## C.8 Training objectives of backbone and controller

Backbone. The backbone parameters minimise the calibration loss Eq. 5. Terms that involve a rare-event functional G are estimated on the tilted batch by the self-normalised estimator Eq. 83, and, in the combined variant, averaged with the plain estimate on the reference batch of size $n _ { p }$ with the weights $n _ { p } / ( n _ { p } + \mathrm { E S S } )$ and ESS $/ ( n _ { p } + \mathrm { E S S } )$ 1 held fixed for the gradient; all other terms are plain Monte-Carlo estimates on the reference batch. The backbone receives no gradient through the weights.

Controller. The controller parameters ${ \theta _ { c } } = \left( { { u _ { \theta } } , { \mu _ { \theta } } } \right)$ minimise the cross-entropy objective with a KL wall,

$$
\widehat { \mathcal { I } } ( \theta _ { c } ) = - \sum _ { i = 1 } ^ { n _ { q } } \bar { w } _ { i } ^ { \star } \log \frac { \mathrm { d } \mathbb { Q } _ { \theta _ { c } } } { \mathrm { d } \mathbb { P } } ( X ^ { ( i ) } ) + \lambda _ { \mathrm { K L } } \widehat { \mathrm { K L } } ( \mathbb { Q } _ { \theta _ { c } } \| \mathbb { P } ) + \bigl ( \widehat { \mathrm { K L } } ( \mathbb { Q } _ { \theta _ { c } } \| \mathbb { P } ) - \kappa \bigr ) _ { + } ^ { 2 } ,\tag{85}
$$

where the self-normalised target weights $\bar { w } _ { i } ^ { \star }$ are held fixed for the gradient (their log-ratios clipped at ±5 so that no single path can dominate the update), κ is the wall in nats (6 on $\operatorname { D A X } ; \ - \log p + 2$ for an event of nominal probability p on toy), and $\widehat { \mathrm { K L } }$ is the energy form $\begin{array} { r } { n _ { q } ^ { - 1 } \sum _ { i } \left( \frac { 1 } { 2 } \sum _ { k } \| u _ { \boldsymbol { \theta } , k } \| ^ { 2 } \Delta t + \frac { 1 } { 2 } \| \mu _ { \boldsymbol { \theta } } \| ^ { 2 } \right) } \end{array}$ , whose Q-expectation equals KL(Q∥P) exactly. The first term is the Monte-Carlo estimate of $\operatorname { K L } ( \mathbb { Q } ^ { \star } \lVert \mathbb { Q } _ { \theta _ { c } } )$ up to a constant, with $\mathbb { Q } ^ { \star } \propto | G |$ P the zero-variance proposal; its minimiser is the moment match $\mathbb { E } ^ { \mathbb { Q } ^ { \star } }$ of the suficient statistics of the tilt (the conditional mean of $z _ { \mathrm { 0 } }$ and of the innovation increments given the rare event). The wall is needed because the weak penalt $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 4 }$ alone does not prevent the collapse of the proposal onto a handful of paths once the target weights degenerate.

Gradient convention. $\operatorname { E q . }$ 12 can be diferentiated in two ways with the same value. In the score form the sampled path is held fixed: $\partial _ { u _ { \theta , k } } \log ( \mathrm { d } \mathbb { Q } / \mathrm { d } \mathbb { P } ) = \Delta \epsilon _ { k } ^ { \mathbb { Q } }$ and $\partial _ { \mu _ { \theta } } = z _ { 0 } - \mu _ { \theta }$ , the scores of the tilted law, which is what the cross-entropy term requires. In the reparameterised form the tilted noise is held fixed and the derivative is $\Delta \epsilon _ { k } ^ { \mathbb { P } }$ (respectively $z _ { \mathrm { 0 } } )$ , which is the correct total derivative of a $\mathbb { Q } \mathrm { . }$ -expectation of a function of the weights, such as the KL wall or the efective sample size. Using the reparameterised form inside the cross-entropy term gives an update whose fixed point is not the moment match and which pushes the tilt outward without bound.

## D Experimental Setup and Hyperparameter Grids

## toy Dataset

toy is a controlled synthetic benchmark designed to separate path fitting from path-functional calibration. The ground truth is a three-component nonlinear Itˆo SDE on [0, T] with $T = 1$ , state $X _ { t } = ( X _ { t } ^ { 1 } , X _ { t } ^ { 2 } , X _ { t } ^ { 3 } )$ and diagonal state-dependent difusion:

$$
\begin{array} { r } { \begin{array} { c c } { \mathrm { d } X _ { t } ^ { 1 } = - X _ { t } ^ { 1 } \big ( ( X _ { t } ^ { 1 } ) ^ { 2 } - 1 \big ) \mathrm { d } t + \sqrt { | X _ { t } ^ { 1 } | + 1 } \sqrt [ 3 ] { X _ { t } ^ { 1 } + 5 } \mathrm { d } W _ { t } ^ { 1 } , ~ } & { \qquad X _ { 0 } ^ { 1 } = 0 . 5 , } \end{array} } \end{array}
$$

$$
\begin{array} { r } { \mathrm { d } X _ { t } ^ { 2 } = \sin ( X _ { t } ^ { 2 } ) \big ( 2 - ( X _ { t } ^ { 2 } ) ^ { 2 } \big ) \mathrm { d } t + \sqrt [ 3 ] { 1 + ( X _ { t } ^ { 2 } ) ^ { 2 } \mathrm { d } W _ { t } ^ { 2 } } , \qquad X _ { 0 } ^ { 2 } = 0 , } \end{array}
$$

$$
\begin{array} { r }  \begin{array} { c c } { \mathrm { d } X _ { t } ^ { 3 } = \left( X _ { t } ^ { 3 } - ( X _ { t } ^ { 3 } ) ^ { 3 } / 3 \right) \mathrm { d } t + \exp \left( - ( X _ { t } ^ { 3 } ) ^ { 2 } / 4 \right) X _ { t } ^ { 3 } \mathrm { ~ d } W _ { t } ^ { 3 } , } \end{array} \qquad X _ { 0 } ^ { 3 } = 1 , \end{array}
$$

with mutually independent driving Brownians. The observable is the linear combination

$$
Y _ { t } = 0 . 4 0 X _ { t } ^ { 1 } + 0 . 3 5 X _ { t } ^ { 2 } + 0 . 2 5 X _ { t } ^ { 3 } .
$$

We simulate $2 \times 1 0 ^ { 5 }$ Monte-Carlo paths via Euler–Maruyama on a uniform grid of $N = 2 0 4 8 { \mathrm { ~ s t e p s } } .$ . From the simulated paths we compute four classes of supervised path-functional targets at the evaluation times $T _ { i } \in \mathcal { T } =$ {0.1, 0.25, 0.5, 0.75, 1.0}.

These targets are:

• Thresholded positive-part functionals at levels $\mathcal { K } = \{ - 0 . 5 , - 0 . 2 5 , 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 \}$ :

$$
\mathbb { E } [ \operatorname* { m a x } ( Y _ { T _ { i } } - K , 0 ) ] .
$$

• Running maximum:

$$
\mathbb { E } [ \operatorname* { m a x } _ { 0 \leq t \leq T _ { i } } Y _ { t } ] .
$$

• Squared path average:

$$
\mathbb { E } \big [ \frac { 1 } { T _ { i } } \int _ { 0 } ^ { T _ { i } } Y _ { t } ^ { 2 } \mathrm { ~ d } t \big ] ,
$$

computed on the discrete grid.

• Threshold-crossing probabilities

$$
\mathbb { P } ( \operatorname* { m a x } _ { 0 \leq t \leq T _ { i } } Y _ { t } \geq H _ { j } )
$$

at three threshold levels $H _ { j }$ chosen as the 80%, 90%, and 95% empirical quantiles of the terminal running maximum; in the objective the indicator is replaced by a smooth approximation (a sigmoid of the excess over the threshold with a fixed temperature), so that Eq. 5 is diferentiable.

The threshold-crossing and high-level positive-part targets are deliberately tail-sensitive path functionals. Errors on typical-path statistics and on tail functionals need not agree: a model may fit typical trajectories well while still missing the rare excursions or extrema that determine these calibration targets. Thus toy provides a controlled setting for evaluating whether a model can match both ordinary path behaviour and rare path-dependent statistics.

## DAX Dataset

dax is a real-data functional calibration benchmark built from historical European option quotes on the DAX index. The purpose of the benchmark is to test whether a stochastic path generator can match a large collection of expectations of nonlinear functions of its terminal value. Each option quote is treated as a supervised target of the form

$$
\mathbb { E } [ \varphi ( Y _ { T } ; K ) ] ,
$$

where $Y$ is the generated one-dimensional path, $T$ is the evaluation time, K is a threshold level, and $\varphi$ is a payof function. All thresholds and target values are normalised by the spot level $S _ { 0 } = 2 5 { , } 2 8 0$ , so that the generated process starts from $Y _ { 0 } = 1$ and $K _ { \mathrm { n o r m } } = K / S _ { 0 }$

Option payofs (for the non-specialist). A European option on the index with maturity T and strike K pays a fixed function of the terminal value $Y _ { T }$ alone. The two basic instruments are the call and the put, with payofs

$$
\varphi _ { \mathrm { c a l l } } ( Y _ { T } ; K ) = ( Y _ { T } - K ) _ { + } , \qquad \varphi _ { \mathrm { p u t } } ( Y _ { T } ; K ) = ( K - Y _ { T } ) _ { + } , \qquad ( x ) _ { + } : = \operatorname { m a x } ( x , 0 ) .\tag{86}
$$

Under the normalised, discounting-free pricing convention used here, the quoted price of an option is the expectation

$$
c _ { T , K } = \mathbb { E } \big [ \varphi ( Y _ { T } ; K ) \big ] ,\tag{87}
$$

which is exactly the supervised target above. A strike is at-the-money when $K \approx S _ { 0 } ~ ( \mathrm { i . e . } ~ K _ { \mathrm { n o r m } } \approx 1 )$ , in-the-money when the option would pay out if exercised at the current level, and out-of-the-money (OTM) otherwise, that is, a call with $K > S _ { 0 }$ or a put with $K < S _ { 0 }$ . OTM prices are small and depend only on the tails of the terminal law, which is why the far-tail groups are the hardest part of the benchmark. The threshold-indicator targets approximate digital payofs through finite diferences of nearby put prices in the strike,

$$
\mathbb { P } ( Y _ { T } \le K ) = \partial _ { K } \mathbb { E } \big [ ( K - Y _ { T } ) _ { + } \big ] \approx \frac { \mathbb { E } \big [ ( K + h - Y _ { T } ) _ { + } \big ] - \mathbb { E } \big [ ( K - h - Y _ { T } ) _ { + } \big ] } { 2 h } ,\tag{88}
$$

and analogously $\mathbb { P } ( Y _ { T } > K ) = - \partial _ { K } \mathbb { E } [ ( Y _ { T } - K ) _ { + } ]$ from call prices.

The dataset is intentionally dense: at each available maturity we include several groups of terminal functionals covering both typical and tail regions of the path distribution:

• 25 positive-part upper-tail targets, $K _ { \mathrm { n o r m } } \in [ 0 . 8 8 , 1 . 1 2 ]$ , corresponding to $\mathbb { E } [ ( Y _ { T } - K ) _ { + } ]$

• 25 positive-part lower-tail targets, $K _ { \mathrm { n o r m } } \in [ 0 . 8 8 , 1 . 1 2 ]$ , corresponding to $\mathbb { E } [ ( K - Y _ { T } ) _ { + } ]$

• 20 far lower-tail targets, $K _ { \mathrm { n o r m } } \in [ 0 . 6 0 , 0 . 8 8 ]$ ;

• 15 far upper-tail targets, $K _ { \mathrm { n o r m } } \in [ 1 . 0 8 , 1 . 3 5 ]$

• 12 threshold-indicator targets, synthesised from diferences of nearby lower-tail positive-part targets, $K _ { \mathrm { n o r m } } \in$ [0.65, 0.98].

The target threshold levels are evenly spaced on each interval and are matched to the nearest observed threshold in the raw quote table. Maturity indices are recomputed on the model time grid with $T = 1$ and $N = 1 0 2 4$ steps.

The far-tail groups are the most challenging part of the benchmark because they depend on rare terminal events under the generated path distribution. When the Girsanov overlay is used for dax, we use two separate controllers: one targets the far lower-tail functionals and the other targets the far upper-tail functionals. This lets each controller focus on a diferent rare region of path space, rather than requiring a single tilt to cover both tails simultaneously.

Train/eval split. Evaluation uses a structured hold-out over strike levels. For each group of targets, one quarter of the strikes are held out for evaluation and the remaining entries are used for training. The held-out strikes are selected non-contiguously, so that the training and evaluation sets cover overlapping ranges. This tests interpolation across the surface rather than extrapolation outside the observed range.

## SPX Dataset

spx is a second real-data option-surface benchmark, constructed identically to dax but from historical European option quotes on the S&P 500 index. As for dax, each quote is a supervised target $\mathbb { E } [ \varphi ( Y _ { T } ; K ) ]$ with the call and put payofs defined above; all strikes and target values are normalised by the spot level so that $Y _ { 0 } ~ = ~ 1$ and $K _ { \mathrm { n o r m } } = K / S _ { 0 }$ , and maturities are recomputed on the model grid with $T = 1$ and $N = 1 0 2 4$ steps. The surface is organised into the same five target groups (near-tail calls and puts, far lower- and upper-tail targets, and threshold-indicator targets) spanning comparable normalised-strike ranges, and evaluation uses the same structured non-contiguous 25% strike hold-out, testing interpolation across the surface. When the Girsanov overlay is used, we again employ two controllers, one per far tail. spx serves as an out-of-sample check that the architecture and calibration procedure transfer across underlyings; results are reported in Table 4 of Appendix E.

## Training protocol

All models use the same optimiser and learning-rate schedule unless otherwise stated: AdamW with $\beta = ( 0 . 9 , 0 . 9 9 9 )$ peak learning rate $1 0 ^ { - 3 }$ , 100 warm-up steps followed by cosine decay, gradient clipping at 1, batch size 1024, evaluation batch size 2048, 1 000 epochs and seven seeds per configuration; every reported number is a mean ± standard error over the seven seeds of the cell in question. The models are simulated on N = 512 time steps. SLiSDE uses the ofset gate of Eq. 82 with the RMS-normalised gate input; the search grids are listed in Appendix D, and the best cell per family and depth is reported.

For Girsanov runs the controller is kept inactive for the first 200 epochs. After activation each training step simulates 1 024 reference paths and two tilted batches of 512 paths, one per controller (far puts, far calls), with $\rho = 0$ . Each far-tail term is estimated by self-normalised importance sampling on its tilted batch, combined with the plain estimate on the reference batch in proportion to the efective sample sizes (the ‘tilted batch only’ variant omits the reference-batch estimate); all other terms are evaluated under P on the reference batch. The controllers minimise Eq. 85 with $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 4 }$ , a 6-nat wall, target log-weights clipped at ±5 and Adam with learning rate $2 \cdot 1 0 ^ { - 3 }$ on their own parameters; the backbone keeps the optimiser and schedule of the vanilla runs. The tilt studies use seven seeds and 1 000 epochs; the tilted arms cost 1.3× the epoch time of the budget-matched untilted arm (303 against 229 ms).

## Search grids

SLiSDE backbone.

• matrix structure: blockdiag with block size $b \in \{ 1 , 4 , 8 , 1 6 \}$

• latent dimension $d \in \{ 3 2 , 6 4 \}$

• layers $L \in \{ 1 , 2 , 3 \}$

• noise dimension $m \in \{ 4 , 8 , 1 6 \}$

• time-feature variant $\in \{ \mathrm { T r u e } , \mathrm { F a l s e } \}$

• normalisation ∈ {rmsnorm, layernorm}

• gate amplitude $\varepsilon = 0 . 1$

• LoRA rank of $D \in \{ 0 , 8 \}$

• time-embedding dimension 8

Girsanov controllers.

• tilt type state space with the latent-path branch, initial-state shift on

• controller objective ∈ {cross-entropy, SNIS-variance}; far-tail estimate ∈ {combined, tilted batch only}

• KL wall $\kappa \in \{ 2 , 6 \}$ nats, $\lambda _ { \mathrm { K L } } = 1 0 ^ { - 4 }$

• last-layer BM correlation $\rho \in \{ 0 . 0 , 0 . 9 \}$

Neural-SDE baseline.

• latent dimension ∈ {32, 64}

• hidden width ∈ {64, 128}

• hidden layers ∈ {2, 3}

• noise dimension $m \in \{ 4 , 8 , 1 6 \}$

• activation ∈ {tanh, GeLU}

• difusion type ∈ {full, diagonal}

SLiCE baseline.

• layers ∈ {2, 3, 4}

• hidden dimension ∈ {32, 64}

• noise dimension $\in \{ 4 , 8 , 1 6 \}$

• block size $b \in \{ 1 , 4 , 8 , 1 6 \}$

dax ablations. Two one-factor sweeps around a two-layer dax configuration of SLiSDE (d = 64, block size 8, general noise): the noise type ∈ {multiplicative, additive, general} at block size 8, and the block size $\in \{ 1 , 4 , 8 , 1 6 \}$ at general noise.

## E Additional Experimental Results

This appendix collects the results referred to from Section 6: the spx surface at two and three layers, the structural ablations (noise type and block size), the timing study, the distributional-recovery study on toy, and the Girsano tilt studies on dax and toy. Unless stated otherwise, each study varies a single factor of the tuned gated in-flow cell of the corresponding main-text table, with the calibration targets held fixed.

## spx option-surface dataset

Table 4 reports the spx option-surface benchmark for two- and three-layer configurations of every family under the protocol of Table 3 (held-out 25% strike mask; tuned configuration per model; seven seeds). SLiSDE matches the held-out loss of SLiCE with two layers (6.44 against 6.43) and attains the lowest held-out loss and the lowest far-put error with three layers, at $6 { - } 8 \times$ fewer parameters and 3.5–4× faster epochs. The Neural SDE is clearly worse on both metrics at both depths.
<table><tr><td>Model</td><td>Params</td><td> $\mathrm { L o s s ~ ( 1 0 ^ { - 4 } ) }$ </td><td>Far put  $( 1 0 ^ { - 6 } )$ </td><td>ms / epoch</td></tr><tr><td>Neural  $\mathrm { S D E } - L = 2$ </td><td>79K</td><td> $1 1 . 9 0 \pm 0 . 1 1$ </td><td> $1 3 5 . 4 \pm 2 . 9$ </td><td>94.1</td></tr><tr><td> $\mathrm { S L i C E } - L = 2$ </td><td>142K</td><td> ${ \bf 6 . 4 3 \pm 0 . 1 2 }$ </td><td> ${ \bf 8 1 . 4 \pm 4 . 6 }$ </td><td>170.1</td></tr><tr><td>SLiSDE gated in-flow − L = 2</td><td>18K</td><td> $6 . 4 4 \pm 0 . 4 0$ </td><td> $8 3 . 7 \pm 4 . 8$ </td><td>44.3</td></tr><tr><td> $\mathrm { N e u r a l ~ S D E } - L = 3$ </td><td>157K</td><td> $1 0 . 2 9 \pm 0 . 1 3$ </td><td> $1 3 6 . 0 \pm 2 . 8$ </td><td>109.7</td></tr><tr><td> ${ \mathrm { S L i C E } } - L = 3$ </td><td>213K</td><td> $6 . 5 3 \pm 0 . 2 6$ </td><td> $8 2 . 9 \pm 3 . 6$ </td><td>248.4</td></tr><tr><td>SLiSDE gated in-flow  $- \ L = 3$ </td><td>34K</td><td> ${ \bf 6 . 2 9 \pm 0 . 2 4 }$ </td><td> ${ \bf 7 0 . 1 \pm 6 . 6 }$ </td><td>70.5</td></tr></table>

Table 4: spx option-surface results without Girsanov tilt, two and three layers. Same masks and protocol as Table 3; best cell of the search grid per family and depth, seven seeds, mean ± standard error. The narrower three-layer SLiCE cell diverged on every seed and is excluded.

## Noise type and block size

Each layer can carry multiplicative noise only $( d _ { t } ^ { j } \ = \ 0 )$ , additive noise only $( C _ { t } ^ { j } ~ = ~ 0 )$ or both (the noise-type remark of Section 3.1), and its structured coeficients are block-diagonal with block size $b _ { s } .$ . Table 5 varies each factor separately around the two-layer gated in-flow configuration on dax (general noise, block size 8) and reports parameter count, held-out loss, far-put loss and time per epoch relative to that reference configuration. The three noise types are within one standard error of each other on the total loss; additive noise is cheaper, while the variants with multiplicative noise fit the far puts better. The total loss is flat in the block size; blocks of 4 or more improve the far puts over the diagonal model, at a cost in time per epoch that grows with $b _ { s }$

<table><tr><td>Variant</td><td>Params</td><td>Loss</td><td>Far put</td><td>ms / epoch</td></tr><tr><td>Noise type (block size 8)</td><td></td><td></td><td></td><td></td></tr><tr><td>multiplicative</td><td>0.99×</td><td>1.05×</td><td>0.97×</td><td>0.93×</td></tr><tr><td>additive</td><td>0.46×</td><td>0.95×</td><td>1.69×</td><td>0.78×</td></tr><tr><td>general (both; reference)</td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td></tr><tr><td>Block size (general noise)</td><td></td><td></td><td></td><td></td></tr><tr><td>bs = 1 (diagonal)</td><td>0.46×</td><td>0.95×</td><td>1.45×</td><td>0.60×</td></tr><tr><td>bs = 4</td><td>0.69×</td><td>0.94×</td><td>0.95×</td><td>0.77×</td></tr><tr><td> $b _ { s } = 8 ~ ( \mathrm { r e f e r e n c e } )$ </td><td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td></tr><tr><td> $b _ { s } = 1 6$ </td><td>1.61×</td><td>0.97×</td><td>0.95×</td><td>1.41×</td></tr></table>

Table 5: Structural ablations on dax, each varying one factor of the two-layer gated in-flow configuration (general noise, block size 8). Every entry is the ratio of the seven-seed mean of the variant to that of the reference configuration; the total-loss diferences within each block are within one standard error. Top: noise type of the structured layers. Bottom: block size of the block-diagonal structure.

## Timing study

Setup. Tables 6 and 7 report wall-clock times after compilation on one NVIDIA RTX 6000 Ada (42.6 GiB usable), as the median of five repeats after two warm-ups, across batch sizes B and horizons T. The models are a two layer SLiSDE with $d = 6 4$ , block size 8 and general noise (136K parameters), evaluated sequentially so that the comparison isolates the structured layers from the scan, a Neural SDE of the same parameter count (137K) and SLiCE at the same width and depth (142K).

Training step. SLiSDE is $6 { - } 7 \times$ faster than the Neural SDE at B = 64, 3.2× at $B = 2 5 6$ and 1.3–1.4× at B = 1024. ${ \mathrm { A t ~ } } B = 4 0 9 6 .$ , where batch parallelism saturates the GPU, the Neural SDE step is 1.4× faster, and SLiCE runs out of memory beyond $B = 1 0 2 4$ at $T = 5 1 2$
<table><tr><td>B</td><td>T</td><td>SLiSDE (seq.)</td><td>Neural SDE</td><td>SLiCE</td><td>speed-up</td></tr><tr><td>64</td><td>512</td><td>9.5</td><td>54.0</td><td>10.1</td><td>5.7×</td></tr><tr><td>64</td><td>2048</td><td>35.8</td><td>212.0</td><td>42.8</td><td>5.9×</td></tr><tr><td>64</td><td>8192</td><td>117.1</td><td>838.9</td><td>156.9</td><td>7.2×</td></tr><tr><td>256</td><td>2048</td><td>79.7</td><td>257.1</td><td>155.2</td><td>3.2×</td></tr><tr><td>1024</td><td>512</td><td>69.0</td><td>89.4</td><td>155.0</td><td>1.3×</td></tr><tr><td>1024</td><td>2048</td><td>258.9</td><td>359.5</td><td>OOM</td><td>1.4×</td></tr><tr><td>4096</td><td>512</td><td>242.1</td><td>178.6</td><td>OOM</td><td>0.7×</td></tr></table>

Table 6: Wall-clock time per training step (ms; forward plus backward, post-compilation, median of five repeats) as a function of batch size B and horizon T for SLiSDE (136K parameters, sequential evaluation), the Neural SDE of the same parameter count (137K) and SLiCE (142K). Speed-up is Neural SDE over SLiSDE; OOM: exceeds the 42.6 GiB of the GPU.

Forward pass and scan. Table 7 separates forward and backward passes and adds the chunked associative scan (best chunk size of 16, 64, 256). The scan shortens the forward pass by 1.4× at B = 64, T = 512 and is otherwise on par with or slower than the sequential evaluation: each sequential structured update is already cheap and the chunked scan roughly doubles the arithmetic, so its advantage lies in parallel depth rather than in wall-clock time at these sizes.
<table><tr><td>B</td><td>T</td><td>SLiSDE fwd (seq.)</td><td>SLiSDE fwd (scan)</td><td>NSDE fwd</td><td>SLiSDE fwd+bwd</td><td>NSDE fwd+bwd</td></tr><tr><td>64</td><td>512</td><td>5.7</td><td>4.1</td><td>15.3</td><td>9.5</td><td>54.0</td></tr><tr><td>64</td><td>2048</td><td>14.9</td><td>15.8</td><td>50.6</td><td>35.8</td><td>212.0</td></tr><tr><td>64</td><td>8192</td><td>48.4</td><td>50.4</td><td>202.6</td><td>117.1</td><td>838.9</td></tr><tr><td>256</td><td>2048</td><td>32.5</td><td>44.6</td><td>61.0</td><td>79.7</td><td>257.1</td></tr><tr><td>1024</td><td>512</td><td>27.6</td><td>39.9</td><td>23.8</td><td>69.0</td><td>89.4</td></tr><tr><td>1024</td><td>2048</td><td>89.9</td><td>OOM</td><td>84.2</td><td>258.9</td><td>359.5</td></tr><tr><td>4096</td><td>512</td><td>80.5</td><td>OOM</td><td>37.5</td><td>242.1</td><td>178.6</td></tr></table>

Table 7: Wall-clock time (ms, post-compilation, NVIDIA RTX 6000 Ada) of forward passes and full training steps as a function of batch size and horizon, including the chunked associative scan (best of chunk sizes 16, 64, 256); the Neural SDE has the same parameter count as SLiSDE (137K against 136K).

## Distributional recovery on toy

Because the toy generator is known, the learned distribution can be assessed directly, well beyond the calibrated targets. Table 8 reports Kolmogorov–Smirnov and Wasserstein-1 distances to the ground truth, computed from $2 \times 1 0 ^ { 4 }$ model paths for the best two-layer cell of each family (seven seeds), for the terminal value and for the running maximum. SLiSDE and the Neural SDE recover the terminal marginal comparably well (SLiSDE better in KS, the Neural SDE in $W _ { 1 } )$ ; on the law of the running maximum, the genuinely path-dependent statistic, SLiSDE is markedly better than both baselines.

<table><tr><td></td><td>KS (terminal)</td><td> $W _ { 1 } ~ \mathrm { ( t e r m i n a l ) }$ </td><td>KS (running max)</td><td> $W _ { 1 }$  (running max)</td></tr><tr><td> $\mathrm { N e u r a l \ S D E } \ ( L = 2 )$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 2 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 5 9 \pm 0 . 0 0 2$ </td><td> $0 . 0 4 7 \pm 0 . 0 0 2$ </td></tr><tr><td>SLiCE (L = 2)</td><td> $0 . 0 2 2 \pm 0 . 0 0 2$ </td><td> $0 . 0 3 1 \pm 0 . 0 0 3$ </td><td> $0 . 0 8 1 \pm 0 . 0 0 3$ </td><td> $0 . 0 6 7 \pm 0 . 0 0 3$ </td></tr><tr><td> $\mathrm { S L i S D E } \ ( L = 2 )$ </td><td> $\mathbf { 0 . 0 1 5 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 2 3 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 3 6 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 0 3 7 \pm 0 . 0 0 2 }$ </td></tr></table>

Table 8: Distances between the learned and the ground-truth laws of the terminal value and of the running maximum on toy $( 2 \times 1 0 ^ { 4 }$ model paths; best two-layer cell of each family; seven seeds, mean ± s.e.).

## Girsanov tilt: calibration on dax and rare-event estimation on toy

Calibration with the tilt on dax: protocol. Table 9 compares five arms over seven seeds under one evaluation protocol, plain Monte Carlo under the reference law with 32 768 paths on the held-out strikes: the vanilla model; the untilted reference-law model with 1 024 and with 2 048 paths per step, the latter being the path budget of the tilted arms; and two tilted arms, which simulate 1 024 reference paths and two tilted batches of 512 paths, one controller per tail. The tilted arms difer in the far-tail estimate: one combines the tilted batch with the reference batch, the other uses the tilted batch alone.

Calibration with the tilt on dax: results. All ratios in this paragraph are ratios of means over seeds. Relative to the vanilla model, the combined variant lowers the far-call error by about a third (0.64×; lower on six seeds of seven, paired t-test on the log-errors $p = 0 . 0 3 )$ and the total held-out loss by 15%; the tilted-batch-only variant halves the far-put error on average, with a four times smaller spread across seeds, and lowers the total loss by 24%. Against the budget-matched untilted arm the tilted arms are never worse and are better on average on the far tails (0.78× on far calls and 0.60× on far puts for the respective variants), diferences that are not significant at seven seeds. The tilt is a genuine change of measure here, ${ \mathrm { K L } } ( \mathbb { Q } \| \mathbb { P } ) \approx 3$ nats and ESS $/ n _ { q } \approx 0 . 0 5$ on the put side, and costs 1.3× the epoch time of the budget-matched untilted arm.
<table><tr><td>Arm</td><td>paths / step</td><td> $\mathrm { L o s s ~ ( 1 0 ^ { - 4 } ) }$ </td><td> $\mathrm { F a r \ p u t \ ( 1 0 ^ { - 6 } ) }$ </td><td> $\mathrm { F a r ~ c a l l ~ ( 1 0 ^ { - 6 } ) }$ </td><td> $\mathrm { E S S } / n _ { q }$  put / call</td><td> $\mathrm { K L \ p u t / \ c a l l }$ </td></tr><tr><td>Vanilla</td><td>1024</td><td> $1 . 1 5 \pm 0 . 1 2$ </td><td> $0 . 9 3 \pm 0 . 4 6$ </td><td> $3 . 1 5 \pm 0 . 3 3$ </td><td></td><td></td></tr><tr><td>Untilted,  $\rho = 0$ </td><td>1024</td><td> $1 . 1 8 \pm 0 . 1 9$ </td><td> $0 . 8 7 \pm 0 . 4 2$ </td><td> $3 . 6 2 \pm 0 . 7 8$ </td><td></td><td></td></tr><tr><td>Untilted,  $\rho = 0$ </td><td>2048</td><td> $0 . 9 5 \pm 0 . 1 3$ </td><td> $0 . 8 1 \pm 0 . 3 0$ </td><td> $2 . 5 9 \pm 0 . 4 9$ </td><td></td><td></td></tr><tr><td>Tilt, combined far-tail estimate</td><td> $1 0 2 4 + 2 { \times } 5 1 2$ </td><td> $0 . 9 8 \pm 0 . 1 0$ </td><td> $0 . 8 2 \pm 0 . 2 7$ </td><td> ${ \bf 2 . 0 3 \pm 0 . 7 8 }$ </td><td> $0 . 0 5 \ / \ 0 . 0 9$ </td><td> $2 . 9 \ / \ 1 . 5$ </td></tr><tr><td>Tilt, tilted batch only</td><td> $1 0 2 4 + 2 { \times } 5 1 2$ </td><td> $\mathbf { 0 . 8 8 \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 4 8 \pm 0 . 0 8 }$ </td><td> $3 . 2 5 \pm 1 . 0 4$ </td><td> $0 . 0 5 \ / \ 0 . 1 2$ </td><td> $3 . 0 ~ / ~ 1 . 4$ </td></tr></table>

Table 9: dax with the Girsanov tilt (seven seeds, mean ± standard error). The tilt study is trained and evaluated on the strikes that lie within the quoted range at each expiry, so its absolute values are not comparable with Table 3, whose protocol it otherwise follows. Far-tail losses are held-out MSEs on the far strikes; every arm is evaluated by plain Monte Carlo under its reference law with 32 768 paths. The tilted arms use $\rho = 0 ,$ the cross-entropy controller with a 6-nat KL wall and the initial-state shift; ESS $/ n _ { q }$ and KL(Q∥P) (nats) are end-of-training values of the put / call controllers; time per epoch 130 / 130 / 229 / 303 / 295 ms.

The tilt as a learned importance sampler: protocol. To isolate the benefit of the tilt from the calibration task, we evaluate it on toy in a controlled estimator study against ordinary Monte Carlo at the same compute budget. The reference law is the three-layer toy cell of Table 2, simulated at $N = 5 1 2$ steps with $\rho = 0$ and calibrated seven times with diferent seeds. The targets are the exceedance probabilities $\mathbb { P } ( Y _ { T } > H )$ with H at the model’s own quantiles of nominal probability $1 0 ^ { - 2 } , 1 0 ^ { - 3 }$ and $1 0 ^ { - 4 }$ ; reference values come from $8 \times 1 0 ^ { 5 }$ plain Monte-Carlo paths. For each backbone one controller per threshold is trained post hoc, in three stages of 1 000 cross-entropy steps on batches of 1 024 tilted paths, the stage for the rarer event starting from the controller of the previous one, under the wall $\kappa = - \log p + 2$ nats. Every estimator receives the same per-estimate path budget n; in addition, plain Monte Carlo receives a cost-matched budget inflated by the measured per-path overhead of the tilted simulation (a factor 1.18). The relative root-mean-square error of every estimator is measured over 100 independent replications, and the tables report the mean and standard error over the seven backbones.

The tilt as a learned importance sampler: results. Table 10 reports the budget $n = 1 0 2 4$ for plain Monte Carlo, for the tilt with the vanilla importance-sampling estimator and with the self-normalised one, together with the fraction of replications in which plain Monte Carlo returns exactly zero because no path reaches the threshold. The picture is the one predicted by the variance argument of Section 4: the gain of the tilt grows with the rarity of the event, from a factor 2 in relative error at $p = 1 0 ^ { - 2 }$ to a factor 4 at $p = 1 0 ^ { - 4 }$ , where plain Monte Carlo returns zero in 90% of the replications. The self-normalised estimator is markedly weaker, because the initial-state shift assigns large weights to tilted paths that miss the event and those weights enter only its normalisation.
<table><tr><td>Estimator  $( n = 1 0 2 4 )$  , rel. RMSE</td><td> $p \approx 1 0 ^ { - 2 }$ </td><td> $p \approx 1 0 ^ { - 3 }$ </td><td> $p \approx 1 0 ^ { - 4 }$ </td></tr><tr><td>Plain MC</td><td> $0 . 3 2 0 \pm 0 . 0 0 5$ </td><td> $1 . 0 5 1 \pm 0 . 0 1 9$ </td><td> $3 . 3 9 \pm 0 . 3 9$ </td></tr><tr><td>Plain MC, cost-matched (×1.18)</td><td> $0 . 2 7 8 \pm 0 . 0 0 9$ </td><td> $0 . 9 3 4 \pm 0 . 0 4 7$ </td><td> $3 . 1 5 \pm 0 . 3 7$ </td></tr><tr><td>Girsanov tilt, IS</td><td> $\mathbf { 0 . 1 5 6 \pm 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 3 9 4 \pm 0 . 1 0 4 }$ </td><td> $\mathbf { 0 . 7 8 5 \pm 0 . 1 8 4 }$ </td></tr><tr><td>Girsanov tilt, SNIS</td><td> $0 . 2 7 4 \pm 0 . 0 2 1$ </td><td> $0 . 7 8 9 \pm 0 . 1 0 6$ </td><td> $1 . 8 8 \pm 0 . 4 2$ </td></tr><tr><td>Share of plain-MC runs returning exactly 0</td><td>0%</td><td>34%</td><td>90%</td></tr></table>

Table 10: Rare-event estimation on toy: relative RMSE of each estimator at a budget of n = 1024 paths per estimate (cost-matched plain Monte Carlo receives 1.18 n paths), over 100 replications for each of seven independently calibrated models; mean ± standard error over the models.

Budgets and diagnostics. Table 11 extends Table 10 to the budgets $n \in \{ 2 5 6 , 1 0 2 4 , 4 0 9 6 \}$ and adds the truncated vanilla estimator (weights capped at $\sqrt { n } )$ . At the end of training the controllers use 2.1, 3.5 and 5.1 nats of divergence, and the tilted paths hit the event at rates of 26%, 20% and 15% for $p = 1 0 ^ { - 2 } , 1 0 ^ { - 3 }$ and $1 0 ^ { - 4 }$ (plain Monte Carlo: 1%, 0.1%, 0.01%), at ESS $/ n _ { q }$ of 0.045, 0.024 and 0.011. Two features of the table deserve comment. The self-normalised estimator is uniformly worse than the vanilla one, for the reason given in Appendix C.7, and the truncated estimator coincides with the untruncated one except at $p = 1 0 ^ { - 2 }$ . The weights are heavy-tailed $( \mathbb { E } ^ { \mathbb { Q } } [ \boldsymbol { w } ]$ estimated on the tilted batch is 0.75–0.97 instead of 1), so the standard errors across backbones are large and individual cells can lose to plain Monte Carlo; the gain of the tilt nevertheless grows with the rarity of the event at every budget, and at $p = 1 0 ^ { - 4 }$ it is the only estimator that returns a non-zero value in most replications at $n \leq 1 0 2 4$

## Efect of the controller objective, the KL wall and the far-tail estimate

The wall κ in Eq. 85 controls how far the Girsanov proposal may move from the reference measure, and the choice of controller objective decides whether it moves at all. Table 12 varies the wall, the objective and the way the far-tail terms are estimated on dax. The variance-type objective, which minimises the self-normalised estimate of the far-tail loss directly, converges to the identity tilt (ESS $/ n _ { q } = 0 . 9 9 8 )$ and its arm is then an untilted model with 1 024 extra paths on the far-tail terms; the cross-entropy objective produces a proposal with 2–3 nats of divergence at either wall. Combining the tilted-batch estimate with the reference-batch estimate helps the far calls, whose controller is the weaker of the two, and hurts the far puts, where the tilted batch alone is the better estimator.

<table><tr><td></td><td>rel. RMSE</td><td> $n = 2 5 6$ </td><td> $n = 1 0 2 4$ </td><td> $n = 4 0 9 6$ </td></tr><tr><td rowspan="6"> $p \approx 1 0 ^ { - 2 }$ </td><td>Plain MC</td><td> $0 . 6 4 5 \pm 0 . 0 1 7$ </td><td> $0 . 3 2 0 \pm 0 . 0 0 5$ </td><td> $0 . 1 6 3 \pm 0 . 0 0 5$ </td></tr><tr><td>Plain MC, cost-matched</td><td> $0 . 5 6 0 \pm 0 . 0 1 2$ </td><td> $0 . 2 7 8 \pm 0 . 0 0 9$ </td><td> $0 . 1 5 1 \pm 0 . 0 0 3$ </td></tr><tr><td>Tilt, IS</td><td> $\mathbf { 0 . 4 1 1 \pm 0 . 0 8 4 }$ </td><td> $\mathbf { 0 . 1 5 6 \pm 0 . 0 1 8 }$ </td><td> $0 . 1 4 0 \pm 0 . 0 3 3$ </td></tr><tr><td>Tilt, IS truncated</td><td> $0 . 3 9 1 \pm 0 . 0 7 0$ </td><td> $0 . 1 5 6 \pm 0 . 0 1 8$ </td><td> $\mathbf { 0 . 1 1 9 \pm 0 . 0 1 7 }$ </td></tr><tr><td>Tilt, SNIS</td><td> $0 . 6 4 5 \pm 0 . 1 3 1$ </td><td> $0 . 2 7 4 \pm 0 . 0 2 1$ </td><td> $0 . 1 9 7 \pm 0 . 0 3 4$ </td></tr><tr><td>plain MC returning 0</td><td>8%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="6"> $p \approx 1 0 ^ { - 3 }$ </td><td>Plain MC</td><td> $1 . 9 5 3 \pm 0 . 0 8 3$ </td><td> $1 . 0 5 1 \pm 0 . 0 1 9$ </td><td> $0 . 5 0 1 \pm 0 . 0 1 4$ </td></tr><tr><td>Plain MC, cost-matched</td><td> $1 . 7 8 0 \pm 0 . 0 3 9$ </td><td> $0 . 9 3 4 \pm 0 . 0 4 7$ </td><td> $0 . 4 6 0 \pm 0 . 0 1 5$ </td></tr><tr><td>Tilt, IS</td><td> $\mathbf { 0 . 6 4 9 \pm 0 . 1 4 7 }$ </td><td> $\mathbf { 0 . 3 9 4 \pm 0 . 1 0 4 }$ </td><td> $\mathbf { 0 . 3 3 0 \pm 0 . 0 7 5 }$ </td></tr><tr><td>Tilt, IS truncated</td><td> $0 . 6 4 9 \pm 0 . 1 4 7$ </td><td> $0 . 3 9 4 \pm 0 . 1 0 4$ </td><td> $0 . 3 3 0 \pm 0 . 0 7 5$ </td></tr><tr><td>Tilt, SNIS</td><td> $1 . 6 1 0 \pm 0 . 2 8 7$ </td><td> $0 . 7 8 9 \pm 0 . 1 0 6$ </td><td> $0 . 4 8 2 \pm 0 . 0 6 6$ </td></tr><tr><td>plain MC returning 0</td><td>75%</td><td>34%</td><td>1%</td></tr><tr><td rowspan="6"> $p \approx 1 0 ^ { - 4 }$ </td><td>Plain MC</td><td> $6 . 6 4 \pm 0 . 9 5$ </td><td> $3 . 3 9 \pm 0 . 3 9$ </td><td> $1 . 6 5 \pm 0 . 1 0$ </td></tr><tr><td>Plain MC, cost-matched</td><td> $6 . 9 6 \pm 0 . 4 0$ </td><td> $3 . 1 5 \pm 0 . 3 7$ </td><td> $1 . 6 5 \pm 0 . 1 3$ </td></tr><tr><td>Tilt, IS</td><td> ${ \bf 1 . 7 7 \pm 0 . 5 9 }$ </td><td> $\mathbf { 0 . 7 8 5 \pm 0 . 1 8 4 }$ </td><td> $\mathbf { 0 . 8 1 6 \pm 0 . 2 3 5 }$ </td></tr><tr><td>Tilt, IS truncated</td><td> $1 . 7 7 \pm 0 . 5 9$ </td><td> $0 . 7 8 5 \pm 0 . 1 8 4$ </td><td> $0 . 8 1 6 \pm 0 . 2 3 5$ </td></tr><tr><td>Tilt, SNIS</td><td> $4 . 0 6 \pm 1 . 0 8$ </td><td> $1 . 8 8 \pm 0 . 4 2$ </td><td> $1 . 8 1 \pm 0 . 7 2$ </td></tr><tr><td>plain MC returning 0</td><td>97%</td><td>90%</td><td>68%</td></tr></table>

Table 11: Rare-event estimation on toy: relative RMSE for three budgets and three nominal probabilities, over 100 replications for each of seven independently calibrated models (mean ± standard error over the models); cost-matched plain Monte Carlo receives 1.18 n paths; the last row of each block is the share of plain-Monte-Carlo replications with no path above the threshold.

<table><tr><td>Controller</td><td> $\mathrm { L o s s \ ( 1 0 ^ { - 4 } ) }$ </td><td> $\mathrm { F a r \ p u t \ ( 1 0 ^ { - 6 } ) }$ </td><td>Far call  $( 1 0 ^ { - 6 } )$ </td><td> $\mathrm { E S S } / n _ { q } \mathrm { \ p u t / \ c a l l }$ </td></tr><tr><td> $\mathrm { C E } , \kappa = 6 ,$  combined (7 seeds)</td><td> $0 . 9 8 \pm 0 . 1 0$ </td><td> $0 . 8 2 \pm 0 . 2 7$ </td><td> $2 . 0 3 \pm 0 . 7 8$ </td><td> $0 . 0 5 \ / \ 0 . 0 9$ </td></tr><tr><td> $\mathrm { C E } , \kappa = 6 ,$  tilted batch only (7 seeds)</td><td> $0 . 8 8 \pm 0 . 1 0$ </td><td> $0 . 4 8 \pm 0 . 0 8$ </td><td> $3 . 2 5 \pm 1 . 0 4$ </td><td> $0 . 0 5 \ \mathrm { ~ / ~ } 0 . 1 2$ </td></tr><tr><td> $\mathrm { C E } , \kappa = 2 ,$  combined (3 seeds)</td><td> $1 . 1 3 \pm 0 . 2 0$ </td><td> $0 . 5 9 \pm 0 . 0 1$ </td><td> $2 . 4 6 \pm 1 . 0 3$ </td><td> $0 . 0 6 ~ / ~ 0 . 0 8$ </td></tr><tr><td>SNIS-variance, combined (3 seeds)</td><td> $1 . 1 8 \pm 0 . 3 5$ </td><td> $1 . 1 8 \pm 0 . 3 3$ </td><td> $2 . 7 1 \pm 1 . 1 0$ </td><td> $1 . 0 0 ~ / ~ 0 . 9 9$ </td></tr></table>

Table 12: Controller ablations on dax far-tail calibration $( \rho = 0 ,$ protocol of Table 9; mean ± standard error over seeds).