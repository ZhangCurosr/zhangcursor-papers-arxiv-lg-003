# Simulation-Based Quantum System Inference with Neural Posterior Estimation

Hang Zou,<sup>1</sup> Anton Frisk Kockum,<sup>2</sup> Martin Rahm,<sup>3</sup> and Simon Olsson<sup>1,</sup> <sup>∗</sup>

<sup>1</sup>Department of Computer Science and Engineering,

Chalmers University of Technology and University of Gothenburg, 41296 Gothenburg, Sweden

<sup>2</sup>Department of Microtechnology and Nanoscience,

Chalmers University of Technology, 41296 Gothenburg, Sweden

<sup>3</sup>Department of Chemistry and Chemical Engineering,

Chalmers University of Technology, 41296 Gothenburg, Sweden

Models of quantum systems faithfully map system parameters to observations, but the inverse problem of parameter inference from measurement data presents a fundamental challenge: compu tationally intractable likelihoods due to an exponentially large Hilbert space. Here, we introduce simulation-based quantum system inference, a unified, likelihood-free framework that learns param eter posteriors directly from classical simulation data. The central idea is to pair polynomial-cost classical simulators, such as Pauli propagation and tensor networks, with normalizing flows or other neural density estimators for accurate, reusable inference. A single model, trained once, maps any new measurement record to its posterior in one forward pass—turning per-experiment inference into a fixed, up-front cost. We numerically demonstrate the framework’s versatility across Pauli noise learning, quantum error mitigation, quantum state tomography, and Hamiltonian learning, with examples involving 81-qubit shallow circuits and 735-parameter inference. In each case, the approach yields accurate estimates of identifiable parameters, while posterior uncertainty provides ad ditional diagnostics of non-identifiability and indicates where further characterization is needed. Our framework reduces data-acquisition requirements in quantum experiments and accelerates parameter inference, providing a practical route to characterizing and improving large-scale quantum systems.

## I. INTRODUCTION

Quantum theory provides a precise forward map, predicting how the expectation values of chosen observables evolve from an initial state under a Hamiltonian. Quantum characterization demands the inverse—inferring Hamiltonians, states, or noise processes from limited measurement data. Such inverse problems pervade modern quantum science, from the reconstruction of entangled states [1–5] and the benchmarking of quantum processors [4–7] to the identification [8–11] and design [12–14] of many-body Hamiltonians. Yet this inversion is obstructed by an intractable likelihood function p(x θ), which quan tifies how probable the observed data x are under candidate parameters θ. Evaluating it generally requires simulating the full quantum dynamics [15], the cost of which grows exponentially with system size. Conventional likelihood-based inference is therefore impractical beyond small quantum systems [15–17].

Simulation-based inference (SBI) side-steps intractable likelihood evaluations by treating the forward simulator as the statistical model [18, 19]. The sampling distribution induced by the simulator, x p(x θ), implicitly specifies the likelihood, thereby enabling inference of the posterior p(θ x)—the probability distribution over the unknown parameters given observed data. This paradigm has found success across diverse scientific disciplines, from astrophysics to molecular biophysics [20–24].

To date, traditional SBI techniques based on approximate Bayesian computation have been applied to quantum parameter inference [15–17], but major computational barriers limit their practical deployment. The curse of dimensionality manifests in two ways. In forward simulation, even a single evaluation of a quantum model becomes prohibitive for large systems. In statistical inference, learning high-dimensional quantum systems generally requires many samples, further increasing the computational burden. Beyond these scaling barriers, the computational efort invested in one inference does not accelerate the next: the procedure must be rerun from scratch for every new observation.

Against this backdrop, advances on two complementary fronts now provide the key ingredients for eficient high-dimensional quantum-system inference. On the statistical side, deep generative modeling has transformed high-dimensional density estimation [25, 26]. Building on this progress, neural posterior estimation (NPE) trains a conditional generative model to learn complex mappings between parameters and observations [27–31]. NPE provides amortized inference by reusing this model for rapid, high-throughput inference on new data, thereby ofsetting its initial training cost over time [31]. This amortization is especially valuable in the quantum domain, where hardware access is costly, but characterization tasks such as real-time noise tracking [32, 33] and cross-device benchmarking [34] must be carried out repeatedly. Yet harnessing this fit at scale requires more than importing NPE: it demands a forward simulator cheap enough to generate training data for large quantum systems.

On the simulation side, the frontier of classical simulability for quantum systems has expanded. Realistic noise intrinsically restricts state complexity, giving rise to a range of polynomial-cost classical simulation algorithms for noisy quantum circuits [35–41]. In addition, exploitable structures can further reduce simulation complexity, including symmetry reductions [42–44], stabilizer circuits [45], free fermions [46], dynamical Lie algebras [47], and limited state or operator entanglement [48–50]. Nu merical techniques such as Pauli propagation [37–41] and tensor networks [35–37, 51–53] can leverage these features to eficiently approximate systems exceeding 100 qubits. This scale is comparable to those of leading quantum hardware accessible to external researchers, such as Google’s Willow [54], IBM’s Heron [55], and QuEra’s Aquila [56]. For suitably noisy or structured problems, state-of-theart classical simulators can surpass contemporary errormitigated quantum computers in both runtime and precision [36–38, 52]. Accelerated by modern hardware backends [57–59], these simulators make it feasible to generate the large datasets required for NPE training, opening a path to scalable, likelihood-free quantum inference.

Here, we bring these two developments together and develop simulation-based quantum system inference into a general framework (Fig. 1): a single template that pairs polynomial-cost classical simulators with neural posterior estimation to unify otherwise tailor-made characterization tasks. Trained entirely on classical simulations, the estimator approximates posterior distributions for new experimental observations without retraining, amortizing inference across experiments. Posterior samples provide parameter estimates together with uncertainty information, subject to the assumed model and the accuracy of the learned posterior. We demonstrate the framework on four representative inverse problems (Fig. 1, bottom).

First, we show that our amortized models eficiently learn high-dimensional Pauli noise, scaling to 50-qubit shallow circuits while accurately tracking parameter drift across instances. The resulting noise posteriors directly diagnose which Pauli-noise parameters are identifiable under the chosen characterization protocol [60, 61]. We further find that the training-data budget required for high-accuracy inference grows only approximately linearly with system size, indicating a route toward characterization beyond 100 qubits for the sparse Pauli noise model.

Next, we apply the inferred noise posterior to quantum error mitigation (QEM) protocols that require an explicit noise model, including probabilistic error cancellation (PEC) [62, 63] and zero-noise extrapolation (ZNE) [63–65]. Both protocols achieve mitigation accuracy comparable to that obtained using the ground-truth noise parameters predefined in the simulations. We further introduce a digital-twin strategy, in which posterior-driven simulators generate training data for machine-learning QEM (ML-QEM) [66, 67], efectively overcoming the bottleneck of quantum data scarcity. Applied to Trotterized Ising dynamics, it outperforms the ZNE baselines by over twofold in error reduction, and extrapolates beyond the trained circuit depths.

Beyond noise analysis, we apply the framework to quantum state tomography (QST), constructing a reusable estimator for 12-qubit states through structured parameterized quantum circuits [68]. The model reconstructs a range of randomly sampled states with fidelity typically exceeding 99%, without per-state optimization.

Finally, we demonstrate Hamiltonian learning in simulated $9 \times 9$ Rydberg atom arrays [11] by inferring atomic displacements from local observables evaluated after short-time Hamiltonian evolution. The inferred displacements agree closely with the prescribed values (coeficient of determination $R ^ { 2 } ~ = ~ 0 . 9 8 3 )$ and the site-resolved posterior uncertainties highlight locations that may warrant further calibration.

Collectively, our results span systems of up to 81 qubits and inference over as many as 735 latent parameters. This parameter dimensionality lies at the frontier of both Bayesian quantum parameter estimation [69] and SBI in other domains [19, 70]. Our framework provides a principled and reusable tool for high-dimensional quantum characterization. We anticipate that this paradigm will extend to broader applications in quantum technologies, such as non-Markovian noise learning [71–73], quantum sensor calibration [74, 75] and feedback control [76, 77].

The remainder of this paper is organized as follows. In Sec. II, we introduce our inference framework and lay out the foundations of normalizing-flow density estimation. We then apply the framework to sparse Pauli-noise learning in Sec. III and use the learned noise models for QEM in Sec. IV, where we also introduce the digital-twin strategy. Next, we develop a neural estimator for QST in Sec. V. In Sec. VI, we present Hamiltonian learning of Rydberg atom arrays. Finally, we conclude in Sec. VII and highlight promising avenues for future research.

## II. METHODS

In this section, we first present our framework for simulation-based quantum-system inference from a Bayesian perspective. We then introduce conditional normalizing flows as the posterior density estimators used in this work.

## A. Simulation-based quantum system inference

We cast a broad class of quantum-system learning tasks as a unified Bayesian inverse problem. Given an observation x, Bayes’ formula yields the posterior over the latent parameters θ, $p ( \pmb \theta | \mathbf x ) \propto p ( \mathbf x | \pmb \theta ) \pi ( \pmb \theta )$ Here, π(θ) encodes the prior over physically admissible parameter values. The quantum process defines the target likelihood p(x θ) through its exact forward map from the latent parameters to the chosen observable values. Although this likelihood is generally intractable, SBI performs inference using samples generated by scalable classical simulators that approximate the quantum forward map. Accordingly, each SBI task is specified by $( \pmb \theta , \pi , \mathbf x , S )$

As illustrated in the upper part of Fig. 1, parameters $\pmb { \theta } _ { 1 : N }$ are drawn from the prior $\pi ( \theta )$ and propagated through to generate simulated data $\mathbf { x } _ { 1 : N }$ , thereby encoding the likelihood implicitly without closed-form evaluation. NPE trains a conditional normalizing flow $q _ { \phi } ( \pmb { \theta } | \mathbf { x } )$ (see Sec. II B) on the resulting pairs $\{ ( \pmb { \theta } _ { i } , \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { N }$ via maximum likelihood,

![](images/f7c5af3cdc1875d2d234cd55c204120f7175f10e58e2dfacdae13149857561b8.jpg)  
FIG. 1. Simulation-based quantum system inference. $\mathrm { T o p }$ , schematic of the framework. Training phase (solid black arrows): parameters $\pmb { \theta } _ { 1 : N }$ sampled from a prior $\pi ( \theta )$ are passed through a forward simulator to produce synthetic observations $\mathbf { x } _ { 1 : N } ,$ on which a neural density estimator $q _ { \phi } ( \pmb { \theta } | \mathbf { x } )$ learns to approximate the posterior. Inference phase (red arrows): given a target observation $\mathbf { x } _ { 0 }$ from a quantum processing unit (QPU), the trained estimator eficiently evaluates the posterior $q _ { \phi } ( \pmb { \theta } | \mathbf { x } _ { 0 } )$ without further simulation. Sequential training (dashed black arrow): an optional sequential refinement round that reuses the posterior as a new proposal. Bottom, the representative applications demonstrated in this work, including Pauli noise learning, quantum error mitigation, quantum state tomography, and Hamiltonian learning.

$$
\phi ^ { \star } = \arg \operatorname* { m i n } _ { \phi } \mathbb { E } _ { ( \theta , \mathbf { x } ) } [ - \log q _ { \phi } ( \pmb { \theta } | \mathbf { x } ) ] .\tag{1}
$$

Once trained, the NPE model provides a tractable surrogate for the posterior $p ( \pmb \theta | \mathbf x _ { 0 } ) \approx q _ { \phi ^ { \star } } ( \pmb \theta | \mathbf x _ { 0 } )$ , enabling direct generation of posterior samples for any $\mathbf { x } _ { \mathrm { 0 } }$ without further simulation or iterative sampling.

When only a single $\mathbf { x } _ { \mathrm { 0 } }$ is of interest, sequential NPE (SNPE) reduces the total simulation cost at the expense of a model that cannot be reliably reused across observations. At each round r, the previous posterior serves as the proposal, $\tilde { \pi } _ { r } ( \pmb { \theta } ) = q _ { \phi _ { r - 1 } ^ { \star } } ( \pmb { \theta } | \mathbf x _ { 0 } ) ;$ ; samples from $\tilde { \pi } _ { r } ( \pmb { \theta } ) p ( \mathbf { x } | \pmb { \theta } )$ are appended to the cumulative training set, and training is continued under the same objective.

## B. Normalizing flows

To model the posterior distribution $q _ { \phi } ( \pmb { \theta } | \mathbf { x } )$ over the parameter space $\pmb { \theta } \in \mathbb { R } ^ { n }$ , we employ a conditional normalizing flow [78–81]. This flow constructs a smooth and invertible map $f _ { \phi } ( \cdot ; \mathbf { x } ) : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ that transforms a latent variable z drawn from a simple base density $q _ { 0 } ( \mathbf { z } )$ into the target parameter $\theta ,$ conditioned on the observation x. Through the change-of-variables formula, the posterior density evaluates exactly to

$$
q _ { \phi } ( \pmb \theta | \mathbf x ) = q _ { 0 } \bigg ( f _ { \phi } ^ { - 1 } ( \pmb \theta ; \mathbf x ) \bigg ) \bigg | \mathrm { d e t } \frac { \partial f _ { \phi } ^ { - 1 } ( \pmb \theta ; \mathbf x ) } { \partial \pmb \theta } \bigg | .\tag{2}
$$

The invertibility and exact likelihood computation have established normalizing flows as powerful tools across classical and quantum many-body problems [82–85].

In this work, we use neural spline flows [81], which combine rational-quadratic splines for strictly monotonic, invertible mappings with autoregressive transformations for tractable Jacobian determinants for fast training and inference. Details of the flow architecture and computational hyperparameters are provided in Appendix A and Appendix J, respectively.

## III. LEARNING SPARSE PAULI NOISE

Learning noise models is essential for benchmarking quantum devices, guiding calibration, and enabling quantum error mitigation [4, 6]. Noise in two-qubit gates is a major contributor to errors in quantum circuits on current hardware [62, 65]. We therefore first present an application of our framework to inferring gate-by-gate Pauli error probabilities for two-qubit gates from a single set of circuit measurements. The inferred noise models can be used for several purposes, e.g., in quantum error mitigation protocols, as we will demonstrate in Sec. IV.

We consider an n-qubit register subjected to a circuit template over a gate set

$$
G = \{ P _ { 0 } , U _ { \mathrm { p r e p } } , U _ { \mathrm { B W } } , \mathcal { M } \} .\tag{3}
$$

The system is initialized in $\rho _ { 0 } = | 0 \rangle \langle 0 | ^ { \otimes n }$ via $P _ { 0 }$ , followed by a non-commuting local preparation layer $U _ { \mathrm { p r e p } } =$ $\otimes _ { q = 1 } ^ { n } U _ { q }$ with $U _ { q } = R _ { z } ^ { ( q ) } ( \pi / 4 ) R _ { y } ^ { ( q ) } ( \pi / 4 )$ , and a brickwall entangling layer $U _ { \mathrm { B W } }$ composed of n  1 nearest-neighbor CNOT gates split into an odd-control and an even-control sublayer (see Fig. 2). The measurement ensemble consists of all weight-1 and nearest-neighbor weight-2 Pauli observables, regardless of the circuit topology.

Each gate $\mathrm { C N O T } _ { k , k + 1 } \left( k = 1 , \dots , n - 1 \right)$ is accompanied by a 2-local Pauli channel,

$$
\Lambda _ { k } ( \rho ) = \left( 1 - \sum _ { P \in \mathcal { P } } p _ { k , P } \right) \rho + \sum _ { P \in \mathcal { P } } p _ { k , P } P \rho P ,\tag{4}
$$

where $\mathcal { P } = \{ I , X , Y , Z \} ^ { \otimes 2 } \backslash \{ I I \}$ spans the 15-dimensional basis of two-qubit Pauli errors and $p _ { k , P }$ are the corresponding error probabilities. The focus on Pauli noise is broadly applicable in practice, as randomized compiling can reshape arbitrary noise into Pauli channels [86]. The 2-local restriction further balances model sparsity against the ability to capture short-range crosstalk [62]. We treat state preparation and measurement (SPAM) and singlequbit gates as ideal, restricting characterization to the CNOT layers. For deeper circuits with repeated blocks, we assume stationary Markovian noise, keeping the noise parameters fixed across repetitions, as implemented in Sec. IV B.

Within the SBI framework, the latent vector $\theta \ =$ $\{ p _ { k , P } \} \in \mathbb { R } ^ { 1 5 ( n - 1 ) }$ covers the n 1 CNOT gates in the brickwall circuit; unused connections remain uncharacterized when this noise model is applied to real hardware. We adopt a uniform prior $p _ { k , P } \sim \mathcal { U } [ 0 , \theta _ { \mathrm { m a x } } ] \ ( \theta _ { \mathrm { m a x } } = 0 . 0 0 1 )$ yielding a maximum two-qubit gate error of 1.5%, consistent with current superconducting processors [33, 62]. The observation $\mathbf { x } \in \mathbb { R } ^ { 1 \dot { 2 } n - 9 }$ collects all expectation values under . The simulator evaluates these expectation values exactly via Pauli propagation at $\mathcal { O } ( n ^ { 2 } )$ cost (see Appendices B and C 1), requiring approximately 0.1 s at n = 50 on a single CPU thread [see Fig. 9(d) in Appendix C 1]. All observations for both training data and ground-truth benchmarks are generated by this simulator.

For n = 50 qubits (a 735-parameter noise model), Fig. 3(a) displays the posterior distributions of Pauli error coeficients for four representative CNOT gates, using 2,000 posterior samples. Most parameters concentrate tightly around the ground-truth values (red crosses), while a subset exhibits broad posteriors. The high variance is a diagnostic: it isolates exactly those parameters that U<sub>1</sub>are unidentifiable from composite circuit measurements. 1Pauli errors propagate through subsequent CNOT gates, <sup>p</sup>coupling error coeficients from adjacent gates into gauge-Uequivalent combinations whose individual contributions <sub>2</sub>cannot be resolved [7]. This ambiguity persists even <sup>p</sup>with tomographically complete inputs and measurements, Usince noise contributions from adjacent gates cannot be separated, as illustrated in Appendix D. In the n-qubit Λbrickwall circuit considered here, 6(n 2) error coeficients are afected by this gauge freedom, consistent with the U<sub>n</sub>complete posterior distributions in Appendix E. This numerical approach complements the analytic analysis [60] and extends naturally to complex circuits where the alge-Figure 2: n-qubitbra becomes cumbersome.

![](images/bd971bf0d9d987879d9da36160bcb90f89c7f65e8a52c846b10500ef6a85ddef.jpg)  
FIG. 2. Brickwall circuit template used in this study, shown wal<sub>for</sub> $n = 4$ cuit template used in this study, shown<sub>. Each qubit q is initialized by a single-qubit rotation</sub> $U _ { q } ,$ d by a sin and each gate $\mathrm { C N O T } _ { k , k + 1 }$ otation U , and eachis followed by a 2-local Pauli noise channel $\Lambda _ { k }$

Figure 3(b) tracks the convergence of two accuracy metrics as functions of cumulative training data across successive SNPE rounds: the normalized mean absolute error (nMAE), defined as the MAE between posterior means $( { \bar { \theta } } _ { i } )$ and ground-truth values $( \theta _ { i } ^ { \mathrm { t r u e } } )$ normalized by the prior range, $\begin{array} { r } { \mathrm { n M A E } = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } | \bar { \theta } _ { i } - \theta _ { i } ^ { \mathrm { t r u e } } | / \theta _ { \mathrm { m a x } } \times 1 0 0 \% } \end{array}$ ， with d the parameter dimension, and the coeficient of determination $R ^ { 2 }$ . Training begins with 10,000 simulations and increments by 5,000 per round. For the full d-dimensional parameter space, both metrics plateau— nMAE at a nonzero floor and $R ^ { 2 ^ { ' } }$ below unity—as a consequence of gauge freedom. Within the gauge-free subspace, both metrics converge with high accuracy, confirming that the learnable noise structure is faithfully captured. Figure 3(c) examines how the training-data budget required to reach high-accuracy thresholds scales with system size. The data requirement grows approximately linearly from $n = 1 0 \ \mathrm { t o } \ n = 5 0$ , as estimated from the coarse training increments. Since both parameter and observation dimensions grow only linearly with system size—and the training-data budget scales likewise—this favorable scaling positions the approach for application on systems well beyond 50 qubits.

Beyond static noise, we evaluate the amortized estimator on drifting noise parameters. Figure 3(d) tracks three representative coeficients across ten runs under prescribed sinusoidal drifts. Each run index represents either temporal drift or cross-device variation. Consistent with Fig. 3(a), these examples show accurate tracking of p<sub>XX</sub> with low posterior uncertainty, a systematic overestimate of p despite capturing its phase, and a broad, out-of-phase posterior estimate for $p _ { I X }$ The estima tor provides consistent inference behavior across varying instances without per-instance calibration. Moreover, rapid tracking of parameter drifts could turn noise characterization into actionable feedback for stabilizing device performance [32, 76].

(a)  
![](images/cd4bf77f6b149ec52f5d470846cec906a8f380f28f57823dce18369128758834.jpg)

(b)  
![](images/369b87fcbccff998fd21ad1c3153c63683546f9c0e8f6c288fac43e533ee0b71.jpg)

(c)  
![](images/f6980eea7f4bcda2e0846f05861e30122d7e12251e602c9fa89aaccc104b7b88.jpg)

(d)  
![](images/e126d1c4e54a6c7be8cbd6ce9a07693d65826e7f78bc4736bc5c20c41beed110.jpg)  
FIG. 3. Scalable Pauli noise learning with NPE. (a) Posterior distributions of Pauli error coeficients for four representative CNOT gates within a 50-qubit brickwall circuit, shown as box plots over 2,000 posterior samples. Gates are ordered by brickwall layer, with odd-control gates first, rather than by qubit index. The inference is performed using an amortized NPE model trained on 30,000 simulations. Red crosses denote ground-truth values. Broad posteriors on specific parameters reflect gauge ambiguity (see main text and Appendix D). (b) Convergence of nMAE and $R ^ { 2 }$ as a function of cumulative training data across SNPE rounds for the 50-qubit model, starting from 10,000 simulations and incrementing by 5,000 per round. Metrics are shown for both the full parameter space and the gauge-free subset. (c) Training-data budget required to reach high-accuracy thresholds $( R _ { \mathrm { g a u g e - f r e e } } ^ { 2 } \geq 0 . 9 9$ and $\geq 0 . 9 9 9 )$ as a function of system size, extracted from SNPE training analogous to (b) for each system size. (d) Tracking selected drifting Pauli noise coeficients for the $\mathrm { C N O T _ { 1  2 } }$ gate shown in panel (a). Prescribed sinusoidal ground-truth profiles (purple curves) are compared with posterior means (teal circles), with shaded bands indicating ±1 posterior standard deviation.

## IV. POSTERIOR-DRIVEN QUANTUM ERROR MITIGATION

A central application of the inferred noise posterior is QEM, inferring ideal expectation values from noisy measurements through additional circuit executions and classical post-processing [87]. One class of QEM methods uses noise models to guide circuit modifications and sampling, including ZNE and PEC, as we explore in Sec. IV A. ZNE uses probabilistic noise amplification to evaluate expectation values at increased noise levels and extrapolate to the zero-noise limit [65]. PEC represents inverse noise channels as quasiprobability distributions over implementable operations and cancels noise by reweighting measurement outcomes [62, 63]. Achieving a given precision with these methods requires substantial sampling, whose cost can grow exponentially with circuit size [88, 89]. An alternative class of methods estimates corrections from classically tractable reference circuits [66, 67, 90, 91]. In Sec. IV B, we leverage the classical tractability of noisy circuits in our setting to propose a strategy for reducing the data collection costs of these methods.

![](images/b32eb8db5a1e5b8fd70392712aad036817dee2f74404aa141acbecb2942dc961.jpg)

![](images/47760a19ed0afabed02859f269b122b7862a2230c1ae216bb50793e5e91d59d8.jpg)

(c)  
![](images/38e212652ec0de763293946d229bad76c0c2fd9f47bbc4e5c38e0f88d04ab835.jpg)  
FIG. 4. ZNE and PEC based on the SBI noise posterior. All protocols are parameterized by the ground-truth noise (red), the SBI posterior median (green), and the posterior mean (blue). (a) Zero-noise extrapolation (ZNE) of the expectation value $\langle Z _ { 5 } Z _ { 6 } Z _ { 7 } Z _ { 8 } \rangle$ for a 15-qubit probe state. Mitigated estimates at the zero-noise limit $( G = 0$ , star) are obtained via linear (left), quadratic (middle), and exponential (right) extrapolation from unmitigated $( G = 1 )$ and probabilistic-error-amplified $( G \in \{ 1 . 6 , 2 . 2 , 2 . 8 \} )$ ) results, with each expectation value estimated using 2,000 random circuit samples per gain level. (b) Convergence of the PEC estimate for the same observable as a function of the total quasi-probability samples (in thousands). The inset shows the sampling overhead of PEC under each noise model. (c) Comparison of the mitigated results from panels (a) and (b): ZNE with diferent extrapolation fits, and PEC evaluated at a sample size of 2,000 (2k). The dashed black and dash-dotted red lines indicate the ideal and unmitigated values, respectively.

## A. Sampling-based QEM with inferred noise

We assess the efectiveness of the learned posterior by using its mean and median to parameterize two standard QEM protocols: determining the quasiprobabilities for PEC (see Appendix F 1) and constructing the amplification channels for ZNE (see Appendix F 2). All noisy circuit simulations use the ground-truth noise model, while the inferred models are used only to sample the random circuits employed in mitigation. Expectation values entering the mitigation procedures are estimated from 256 shots per random circuit instance.

Figure 4 presents the mitigated expectation value $\langle Z _ { 5 } \tilde { Z _ { 6 } } Z _ { 7 } Z _ { 8 } \rangle$ of the 15-qubit probe state $\begin{array} { r l } { | \psi \rangle } & { { } = } \end{array}$ $U _ { \mathrm { B W } } \bigotimes _ { q = 1 } ^ { 1 5 } \Big [ R _ { z } ^ { ( q ) } ( \pi / 4 ) R _ { y } ^ { ( q ) } ( \pi / 4 ) | 0 \rangle \Big ]$ , where $U _ { \mathrm { B W } }$ is the brickwall CNOT layer. Figure 4(a) displays the ZNE extrapolation curves under linear, quadratic, and exponential models. Across all three fits, the posterior mean and median yield mitigated estimates close to those obtained using the ground-truth noise, with all results showing clear improvement over the unmitigated baseline. Figure 4(b) presents the convergence of PEC as a function of the quasiprobability sample budget. The results for all three noise parameterizations converge closely toward the ideal value, with nearly identical sampling overheads (inset).

Figure 4(c) summarizes the comparison across all methods. In several cases, protocols based on posterior noise estimates yield mitigated values closer to the ideal than those using the ground-truth parameters. These improvements may reflect finite-sampling fluctuations, and do not establish a systematic advantage. The comparable mitigation performance across the tested parameterizations suggests that QEM remains efective despite the non-identifiability of gauge parameters, provided the noise models represent the same efective circuit channel. A similar conclusion was reported in Ref. [61]. This robustness to gauge freedom supports the use of the inferred noise models in the QEM protocols.

## B. Digital-twin ML-QEM

Since both ZNE and PEC require extensive sampling per observable, we turn to ML-QEM, which enables zeroshot correction at much lower runtime overhead [66], but typically demands large volumes of hardware data. We propose a digital-twin strategy to overcome this [Fig. 5(a)]: noise parameters drawn from the SBI posterior parameterize a classical simulator that emulates the noisy device, producing paired noisy–ideal data $\{ ( \mathbf { x } _ { i } ^ { \mathrm { { d t } } } , \mathbf { x } _ { i } ^ { \mathrm { { e x a c t } } } ) \}$ from classically tractable circuits [46, 66, 67, 92]. A denoising model trained on these synthetic pairs learns the correction map, with potential fine-tuning on real data to absorb model mismatch.

Cast in SBI terms, mitigation is itself an inference problem. The latent parameter is the observable correction $\pmb \theta \equiv \Delta \mathbf x = \mathbf x ^ { \mathrm { e x a c t } } - \mathbf x ^ { \mathrm { d t } }$ (or equivalently $\mathbf { x } ^ { \mathrm { e x a c t } } )$ , and the observation x is the noisy data $\left( \mathbf { x } ^ { \mathrm { { d t } } } \right.$ in training, $\mathbf { x } ^ { \mathrm { n o i s y } }$ at deployment). The simulator is the digital twin, and the prior π is left implicit here, induced by the training circuit ensemble rather than specified. Here we instantiate the twin S as an eficient Cliford (stabilizer) simulator: the training circuits retain the CNOT gates of the target circuit and sample each Pauli rotation angle from $\{ 0 , \pi / 2 , \pi , 3 \pi / 2 \}$ , following Cliford data regression (CDR) [67].

![](images/fb226dcf7348ebb1d3990cb6e98760db9737e5502281df42486580705f7192bb.jpg)

![](images/a14001731443746ee012f3f4cb98b0b0b0feac248ae75e01b8c797ce51b64592.jpg)  
FIG. 5. Digital-twin ML-QEM with the SBI noise posterior. (a) Workflow. Training phase: the noise posterior drives a classical simulator to produce paired data $\{ ( { \bf x } _ { i } ^ { \mathrm { d t } } , { \bf x } _ { i } ^ { \mathrm { e x a c t } } ) \}$ for training a denoising network (black arrows), with optional fine-tuning on real data (gray arrows). Deployment phase: the pre-trained network maps noisy outputs $\mathbf { x } ^ { \mathrm { n o i s y } }$ to mitigated estimates $\mathbf { x } ^ { \mathrm { m i t } }$ (red arrows). (b, c) Application to Trotterized dynamics of 12-qubit Ising chains, comparing ideal, raw noisy, linear and quadratic ZNE (gate folding at noise gains $G \in \{ 1 , 3 , 5 \}$ ; see Appendix F 2), and SBI-DT. (b) Time-evolved transverse magnetization $M _ { x }$ (left) and nearest-neighbor correlations $C _ { Z Z } \ \mathrm { ( r i g h t ) }$ , averaged over 50 test coupling strengths $j \sim \mathcal { U } [ - \pi , 0 ]$ . The vertical dotted line at $t = 1 . 4$ separates the training and extrapolation regions. (c) Time-resolved $\begin{array} { r } { \mathrm { M A } \bar { \mathrm { E } } ( t ) = \frac { 1 } { | \mathcal { S } | } \dot { \sum } _ { O \in \mathcal { S } } \bar { | \langle O \rangle } ( t ) - \langle O \rangle _ { \mathrm { e x a c t } } ( t ) | } \end{array}$ across all observables ${ \cal { S } } = \{ X _ { i } \} _ { i } \cup \{ Z _ { i } Z _ { i + 1 } \} _ { i }$ , with shaded ±1 standard deviation over test $j ~ ( \mathrm { l e f t } )$ , and MAE aggregated over all j and $t ,$ with improvement factors relative to the noisy baseline (right).

We depart from CDR in two respects: the training data come from rather than hardware, and CDR’s linear fit is replaced by a conditional network $f _ { \tau }$ that captures nonlinear dependence on circuit depth. Since this categorical prior is ill-suited to normalizing flows and mitigation requires only the point correction, we directly train a regression network $f _ { \tau }$ rather than learning the full posterior:

$$
\operatorname* { m i n } _ { \tau } \mathbb { E } _ { c , ( \mathbf { x } ^ { \mathrm { d t } } , \mathbf { x } ^ { \mathrm { e x a c t } } ) } \Big [ \Big \| f _ { \tau } ( \mathbf { x } ^ { \mathrm { d t } } , c ) - \big ( \mathbf { x } ^ { \mathrm { e x a c t } } - \mathbf { x } ^ { \mathrm { d t } } \big ) \Big \| _ { 2 } ^ { 2 } \Big ] .\tag{5}
$$

where c encodes relevant circuit features. At deployment, the network corrects noisy data in a single forward pass via $\mathbf { x } ^ { \mathrm { m i t } } = \mathbf { x } ^ { \mathrm { n o i s y } } + f _ { \tau } ( \mathbf { x } ^ { \mathrm { n o i s y } } , c )$ . In our simulation, $\mathbf { x } ^ { \mathrm { n o i s y } }$ is generated by exact density-matrix simulation under the ground-truth noise.

We evaluate this approach on the 1D transverse-field Ising model (TFIM),

$$
H = - j \sum _ { i } Z _ { i } Z _ { i + 1 } + h \sum _ { i } X _ { i } ,\tag{6}
$$

whose time evolution is approximated by first-order Trotterization with step size $\delta t$

$$
U ( t ) \approx \left[ \left( \prod _ { i } e ^ { i j \delta t Z _ { i } Z _ { i + 1 } } \right) \left( \prod _ { i } e ^ { - i h \delta t X _ { i } } \right) \right] ^ { t / \delta t } .\tag{7}
$$

Starting from $| 0 \rangle ^ { \otimes N }$ , we evolve under $U ( t )$ . The observation $\mathbf { x } \in \mathbb { R } ^ { 2 N - \mathrm { i } } \overset { \cdot } { ( } = \mathbb { R } ^ { 2 3 }$ for $N = 1 2 )$ collects the single-site expectation values $\langle X _ { i } \rangle$ and nearest-neighbor correlators $\langle Z _ { i } Z _ { i + 1 } \rangle$ at each time; their spatial averages—the transverse magnetization $\begin{array} { r } { M _ { x } ( t ) = \overset { \cdot } { N } \sum _ { i } \langle X _ { i } \rangle ( \overset { \cdot } { t } ) } \end{array}$ and nearestneighbor correlation $\begin{array} { r } { C _ { Z Z } ( t ) = \frac { 1 } { N - 1 } \sum _ { i } \langle Z _ { i } Z _ { i + 1 } \rangle ( t ) - \mathrm { a r e } } \end{array}$ reported only for visualization. The digital twin is parameterized by the posterior mean inferred from a 12-qubit brickwall circuit, whose CNOT connectivity matches that of the Trotterized Ising circuit. Under the Markovian assumption, the noise parameters are applied identically to each repeated CNOT layer throughout the Trotterized evolution, for both the ground-truth and the SBI models. The denoising network is conditioned on Trotter steps via $c = \log ( t / \delta t + 1 )$ ) and trained on 500 random Cliford variants per step, spanning 14 Trotter steps up to $t = 1 . 4$ $( \delta t = 0 . 1 )$

We compare the results of four cases—raw noisy simulation, linear ZNE, quadratic ZNE, and the SBI-derived digital-twin denoiser (SBI-DT). As shown in Fig. 5(b), the SBI-DT produces the most accurate observable estimates at most evolution times, except at short times, where the ZNE-based methods are more accurate. Notably, it generalizes beyond the training boundary $( t > 1 . 4 )$ where the ZNE-based methods degrade significantly. In Fig. 5(c), the aggregated MAE confirms improvement factors of 5.3-fold for SBI-DT, 2.1-fold for quadratic ZNE, and 1.2-fold for linear ZNE, all relative to the noisy baseline. The digital-twin method thus achieves over twofold improvement compared to the quadratic ZNE baseline, while requiring no additional quantum samples at deployment. A complementary SBI instantiation targets the correction density $p ( { \pmb \theta } | { \bf x } )$ instead, modestly outperforming quadratic ZNE but trailing direct regression for pointwise mitigation, as shown in Appendix G. Overall, these results demonstrate that the SBI noise posteriors not only characterize the simulated device but also serve as a practical resource for error mitigation, closing the loop from error inference to mitigation.

![](images/e3c09b025926be25e15c9da1e2ce0eb78175c7729b02fe4137e01408f04be91f.jpg)

![](images/279af09b2cab3d262f475f46ac3ac3e311cf2b5ebbba77b380ae213e3a264d2a.jpg)

(c)  
![](images/4f29d747b9aa0ebee70aeb20abcdbbfcbee07a0c523c64bee90e628356f733a6.jpg)  
FIG. 6. Quantum state tomography with NPE. (a) Workflow, illustrated using a PQC ansatz and an MPS simulator. Randomly sampled circuit parameters define states whose observables are evaluated by MPS contraction to train a neural posterior estimator. The trained estimator infers posterior distributions over circuit parameters from measurement data of new experimental instances, and posterior samples are mapped through the ansatz to reconstructed MPSs. (b) Reconstruction infidelity of simulated 12-qubit pure states versus simulation budget: $1 - F _ { \mathrm { m e d i a n } }$ (blue) and $1 - F _ { \operatorname* { m a x } }$ (red), computed at each budget from 500 posterior samples for each of 200 test states $( \bar { 1 } 0 ^ { 5 }$ samples in total). (c) Probability density of the fidelity F at the largest budget $( 9 \times 1 0 ^ { 5 } )$ , excluding outliers beyond 1.5× the interquartile range. Dashed lines mark the median (blue) and maximum (red) over all samples.

## V. QUANTUM STATE TOMOGRAPHY WITH PARAMETERIZED QUANTUM CIRCUITS

Another application for our inference framework is QST. Quantum state tomography reconstructs an unknown quantum state from measurement data and is central to validating state preparation and quantum devices [4, 5]. Because the full description of an n-qubit state grows exponentially with n, scalable QST generally requires structural priors that restrict the reconstruction to a tractable family of states. Such priors can be encoded through structured ans¨atze, including matrix product states (MPSs) [93, 94], neural quantum states [95, 96], and parameterized quantum circuits (PQCs) [68, 97].

Our framework provides an amortized QST approach that learns from simulated observations of many random states rather than measurement data of a single unknown state, so that one estimator can infer states throughout the ansatz family without retraining. Figure 6(a) illustrates this workflow using a PQC ansatz and an MPS simulator, with posterior parameter samples mapped through the ansatz to reconstructed states. The PQC prepares states $| \psi ( \pmb \theta ) \rangle = U ( \pmb \theta ) | 0 \rangle ^ { \otimes n }$ , whose parameters θ serve as the latent variables. The parameter prior $\pi ( \theta )$ induces a distribution over states supported on the ansatz manifold $\mathcal { M } _ { U } = \{ \vert \psi ( \pmb \theta ) \rangle : \pmb \theta \in \mathrm { s u p p } \pi \}$ . The circuit expressivity and the support of $\pi ( \theta )$ therefore jointly determine the family of states accessible to inference.

As a demonstration, we use the single-layer ansatz, $U ( \pmb \theta ) = U _ { \mathrm { B W } } \Big [ \bigotimes _ { q = 1 } ^ { n } R _ { x } ( \theta _ { q } ) R _ { y } ( \theta _ { q } ) \Big ]$ , where the $R _ { x }$ and $R _ { y }$ rotations on each qubit share a single angle $\theta _ { q }$ . The latent $\pmb { \theta } \in \mathbb { R } ^ { n }$ collects all rotation angles, with independent uniform priors $\theta _ { q } \sim \mathcal { U } [ - \pi , \pi )$ . The simulator returns an observation $\mathbf { x } \in \mathbb { R } ^ { 1 2 \bar { n } - 9 }$ of the expectation values over the Pauli ensemble (weight-1 and nearest-neighbor weight-2). Each expectation value is subject to binomial shot noise with $N _ { \mathrm { s h o t s } } = 1 0 ^ { 4 }$ to mimic finite measurement statistics. These expectation values are evaluated by MPS simulation with bond dimension $\chi ,$ at a cost polynomial in n and $\chi$ (see Appendix H). At fixed $\chi ,$ the simulator represents exactly only the states in $\mathcal { M } _ { U } \cap \mathcal { M } _ { \chi } ^ { \mathrm { M P S } }$ , where $\mathcal { M } _ { \chi } ^ { \mathrm { M P S } }$ denotes the set of pure states with Schmidt rank at most χ across every contiguous cut in the chosen qubit ordering [48]. For the present ansatz, each cut is crossed by at most one CNOT gate, so $\mathcal { M } _ { U } \subseteq \mathcal { M } _ { \chi = 2 } ^ { \mathrm { M P S } }$ and the MPS simulation is exact.

We evaluate our estimator at $n = 1 2$ on 200 independently sampled 12-qubit test states not used during training. For each test state, we draw 500 posterior parameter samples and evaluate the fidelity $F = | \langle \psi ( \pmb { \theta } _ { \mathrm { t r u e } } ) | \psi ( \pmb { \theta } ) \rangle | ^ { 2 }$ by MPS contractions. Figure 6(b) shows decreasing reconstruction infidelity as the simulation budget grows. The median infidelity decreases by more than two orders of magnitude, reaching approximately $2 \times 1 0 ^ { - 3 }$ at the largest budget, while the best-case infidelity remains below $1 . 5 \times 1 0 ^ { - 3 }$ throughout. Figure $6 ( \mathrm { c } )$ shows the fidelity distribution across all test states at the largest budget. The distribution is concentrated near unity, with 86% of all posterior samples achieving fidelity above 0.99. Reconstruction accuracy can be further improved by increasing the training budget or using the inferred θ to initialize gradient-descent QST [98]. As a complementary instantiation, 4-qubit pure-state QST with a dense Cholesky parameterization under an unstructured prior is presented in Appendix I. This example demonstrates amortized QST with an unrestricted pure-state ansatz, achieving median reconstruction fidelities above 0.98 for most tested states.

## VI. HAMILTONIAN LEARNING OF RYDBERG ATOM ARRAYS

Our inference framework not only enables us to learn noise and states, but also Hamiltonians. Learning the realized Hamiltonian of a quantum computer or quantum simulator is essential for verifying that the implemented interactions match the intended model and identifying deviations that require correction [8–10]. As a final demonstration, we turn to Hamiltonian learning of Rydberg atom arrays, a leading platform for large-scale qubit reconfiguration and control [99, 100]. In these arrays, atomic positions set by optical tweezers determine the interatomic couplings, making positional deviations a source of Hamiltonian errors. Rapid inference of atomic positions could therefore inform feedback adjustments to restore the target interactions [11, 101]. Here, we demonstrate this capability by inferring atomic displacements from simulated measurements of the system’s time evolution.

We consider a two-dimensional square array of $N =$ $n _ { x } \times n _ { y }$ neutral atoms trapped at sites ${ \bf r } _ { i } = { \bf r } _ { i } ^ { ( 0 ) } + { \delta _ { i } }$ , where ${ \bf r } _ { i } ^ { ( 0 ) }$ is the ideal grid position and $\pmb { \delta } _ { i } = ( \Delta x _ { i } , \Delta y _ { i } ) \in \mathbb { R } ^ { 2 }$ is a small in-plane displacement. The system is described by the 2D TFIM

$$
H = \Omega \sum _ { i } X _ { i } + \Delta \sum _ { i } Z _ { i } + \sum _ { i < j } J _ { i j } Z _ { i } Z _ { j } ,\tag{8}
$$

where Ω is the Rabi frequency and ∆ represents the detuning field. The coupling $J _ { i j } = C _ { 6 } / | { \bf r } _ { i } - { \bf r } _ { j } | ^ { 6 }$ , with $C _ { 6 }$ the van der Waals coeficient, encodes the pairwise geometry, so that any displacement $\delta _ { i }$ modifies the interactions between atom i and its neighbors. We set $\Omega = \Delta =$ 15 rad/µs, lattice constant $a = 8$ µm (the nominal nearestneighbor spacing), and $C _ { 6 } = 5 . 4 \bar { 2 } \times 1 0 ^ { 6 } \mathrm { { \mu m ^ { 6 } r a d / \mathrm { { \mu s } } } }$ , corresponding $\mathrm { { \dot { \ t o } } ^ { \ 8 7 } R b }$ atoms in the $7 0 S _ { 1 / 2 }$ Rydberg state [11].

We consider $9 \times 9 ~ \mathrm { R y d }$ berg arrays. The latent vector $\pmb { \theta } ~ \in ~ \mathbb { R } ^ { 2 N } ~ ( = ~ \mathbb { R } ^ { 1 6 2 }$ for $N = 8 1 )$ encodes the in-plane displacements of all N atoms, with a uniform prior $\theta _ { i } \sim$ $\mathcal { U } [ - 1 5 0 , 1 5 0 ]$ nm reflecting realistic positional fluctuations. Since the coupling $J _ { i j }$ is efectively short-ranged, we retain only the nearest- and next-nearest-neighbor pairs, $n _ { \mathrm { N N } } =$ $2 ( { \check { N } } - { \sqrt { N } } )$ and $n _ { \mathrm { N N N } } = 2 ( \sqrt { N } - 1 ) ^ { 2 }$ . The observation $\mathbf { x } \in \mathbb { R } ^ { N + n _ { \mathrm { N N } } + n _ { \mathrm { N N N } } } = \mathbb { R } ^ { 5 N - 6 \sqrt { N } + 2 } \left( = \mathbb { R } ^ { 3 5 3 } \mathrm { ~ f o r ~ } N = 8 1 \right)$ ) collects N single-site magnetizations $\left. Z _ { i } \right.$ and n<sub>NN</sub>+n<sub>NNN</sub> two-point correlators $\langle Z _ { i } Z _ { j } \rangle$ , measured on the probe state

$$
\begin{array} { r } { | \psi \rangle = e ^ { - i \Omega \delta t \sum _ { i } X _ { i } } e ^ { - i \Delta \delta t \sum _ { i } Z _ { i } } e ^ { - i \delta t \sum _ { i < j } J _ { i j } Z _ { i } Z _ { j } } | + \rangle ^ { \otimes N } , } \end{array}\tag{9}
$$

obtained from a single short-time Trotter step $( \delta t \ =$ $0 . 0 1 \mathrm { { \textmu s } ) }$ This short-time evolution suppresses Trotter discretization error [102] while keeping the circuit shallow enough for eficient classical simulation. The simulator $s$ evaluates these observables at polynomial cost via truncated Pauli propagation, retaining only terms with Pauli weight $\leq 5$ and coeficient magnitude $\geq 1 0 ^ { - 6 }$ (see more details in Appendices B and C 2). All training and evaluation data are generated by this simulator.

Figure 7(a) displays the posterior distributions of inplane displacements for all 81 atoms in a single groundtruth instance. The amortized estimator, trained on $1 0 ^ { 5 }$ simulations, produces well-localized posteriors whose means and medians cluster tightly around the true displacements across the entire array. The expanded $3 \times 3$ inset shows the posterior confidence regions in more detail. For most sites, the prescribed ground-truth displacement lies within the high-confidence contours; in the few cases where it lies near the boundary, the point estimates nonetheless remain accurate at the nanometer scale.

Figure 7(b) quantifies this performance over 100 independent ground-truth configurations, with 500 posterior samples drawn for each. The parameter-wise regression (top) compares each scalar displacement component (∆x or $\Delta y )$ independently, yielding $\mathrm { M A E } = 1 2 . 2 \pm 9 . 8$ nm and $R ^ { 2 } = 0 . 9 8 3$ . The distribution of the full parametervector error norm $\lVert \theta - \theta _ { \mathrm { t r u e } } \rVert _ { 2 }$ (bottom) demonstrates an order-of-magnitude reduction relative to the broad prior. For context, fluorescence imaging in a tweezerassisted positioning experiment with $\mathrm { \check { s } 7 } \mathrm { R b }$ atoms yields positional uncertainties on the scale of tens of nanometers [103]. This suggests that the reconstruction accuracy achieved here in simulation is practically meaningful, although its performance on experimental data remains to be validated.

## VII. DISCUSSION AND CONCLUSION

We have presented a unified framework for simulationbased quantum system inference and demonstrated its eficacy through comprehensive numerical quantum simulations. Our central message is that advances in the classical simulation of quantum systems can be translated directly into advances in their characterization: any regime a classical simulator can reach becomes accessible to inference from experimental data. This turns quantum parameter inference from a bespoke, per-experiment efort into a reusable capability that grows as simulators and generative models improve.

This capability rests on four features of our approach. First, it is general: diverse problems conventionally addressed by task-specific algorithms are unified under a common SBI formulation. Second, it is scalable:

(a)  
![](images/0905a5b0544e75d4ac5b1ad5f1b2c2b19c9982537ea0efe3357ef9c0cf0a5995.jpg)

![](images/3531a9b4bcde39c1fafff62bf9029e4f22b90c4a3935be0e0d296cdd6d00421d.jpg)

(b)  
![](images/f0299d9f1956286e59ef4c21016f1820eab240c8893609f62d546168b65ee1b9.jpg)

![](images/da80d8ac62f4ddb6e927d9e2997cfbd04f062c2680332829cee1eb422279eca5.jpg)  
FIG. 7. Posterior localization of atomic displacements in simulated $9 \times 9$ Rydberg arrays. (a) Posterior distribution of in-plane displacements, $\Delta x$ and ∆y (in nm), for individual atoms within a single array instance. Each square represents a single array site, with filled contours indicating confidence intervals (color bar). The ground-truth displacements (red stars), posterior means (green circles), and posterior medians (orange diamonds) are indicated. An enlarged $3 \times 3$ view highlights a representative patch. (b) Amortized posterior precision evaluated across 100 independent displacements. Top, parameter-wise regression for individua displacement components; vertical bars denote the posterior standard deviation. Bottom, probability density of the parameter vector error norms, $| | \theta - \theta _ { \mathrm { t r u e } } | | _ { 2 } ,$ contrasting the broad prior (grey) with the localized posterior (blue).

training-data generation relies on simulators such as Pauli propagation and tensor networks rather than density-matrix evolution, and thus inherits their polynomial complexity. Third, it is amortized: an initial training investment produces a neural estimator that maps new measurement data directly to posteriors. Fourth, it is informative: beyond point estimates, posterior uncertainty provides a built-in diagnostic that may guide optimal experimental design [104].

Several directions for further research and development follow naturally. Currently, our Pauli noise learning assumes Markovian dynamics and idealizes SPAM. Moving beyond these simplifying assumptions to incorporate non-Markovianity [71–73] and gate set tomography [7, 105] would advance the framework toward self-consistent characterization of realistic hardware noise. Beyond refining the noise model, a crucial step is to address model misspecification: since the estimator is trained entirely on simulations, any simulator–device discrepancy can bias posteriors on experimental data. Detecting and quantifying such misspecification to calibrate posterior uncertainties [106– 109] will be essential for reliable deployment on hardware.

Even when the full circuit is too complex to simulate, our framework can still act on classically tractable substructures. For instance, noise learned on repeated layers can be reused across deep circuits under stationarity [62, 65], as demonstrated in Sec. IV B; partial tomography estimates only selected density-matrix elements [110] or local reduced states [111]; and circuit cutting decomposes large circuits into simulable fragments [112, 113]. In addition, just as the randomized measurement toolbox [114] exploits randomness on the measurement side, our framework introduces a new randomized tool on the simulation side, and combining the two ofers a promising direction for future work.

On the architectural side, size-agnostic models such as graph neural networks could enable estimators trained on small systems to generalize directly to larger ones [11, 115–117], removing the fixed-size restriction of models used here. In addition, gauge equivariant or invariant architectures could help handle gauge redundancies [118], such as those in Pauli noise models and MPSs. On the training side, flow matching further ofers a scalable route to enhance NPE training in high-dimensional parameter spaces [30].

Looking further ahead, both components of the framework can evolve. Hybrid quantum–classical infrastructure [119, 120] and eventually early fault-tolerant processors [121] could generate training data beyond classical reach, extending inference to strongly entangled states and long-time dynamics, and allowing trusted devices to characterize less reliable ones [8, 122]. Likewise, quantum generative models could serve as density estimators for posteriors that are hard to represent classically [123, 124]. Beyond inference, the principle underlying Hamiltonian learning could be extended to inverse Hamiltonian design, identifying candidate Hamiltonians predicted to realize desired properties [12–14]. Overall, we envisage that this paradigm will mature into a versatile tool for characterizing, optimizing, and ultimately engineering complex quantum systems.

We note a recent work by Belliardo et al. [69], which also uses normalizing flows to estimate dozens of quantum parameters. A key distinction lies in the inference paradigm: their method relies on variational Bayesian inference with an explicit likelihood model and optimizes the evidence lower bound anew for each observation. In contrast, our SBI approach is likelihood-free and eliminates online optimization across varying quantum settings, at the expense of upfront simulation and training. The two frameworks are complementary, representing important steps in scaling statistical inference to high-dimensional quantum domains.

## ACKNOWLEDGMENTS

The authors thank Akshay Gaikwad for valuable discussions. This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. The work was further supported by a joint project between WASP and the Wallenberg Center for Quantum Technology (WACQT), funded by the Knut and Alice Wallenberg Foundation. Preliminary results were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS) at Alvis and Arrhenius (project: NAISS 2026/4-684). AFK further acknowledges support from the Swedish Foundation for Strategic Research (grant numbers FFL21- 0279 and FUS21-0063) and from the Norwegian Research Council through the Norwegian Quantum Software Center (NorQSoft, project number 361350). AFK and MR also acknowledge support from the Horizon Europe program HORIZON-CL4-2022-QUANTUM-01-SGA via the project 101113946 OpenSuperQPlus100.

Use of AI tools. The conception, methodology, design of numerical experiments, and analysis of results were carried out entirely by the authors. Claude Sonnet 4.6 and Claude Opus 4.6 implemented most of the code following the authors’ supervision, and all code was checked by the authors. Claude Opus 4.8 and ChatGPT 5.6 Sol were used to improve the readability of the manuscript and to suggest supplementary references, all of which were reviewed and verified by the authors. The authors take full responsibility for the content of this article.

## DATA AVAILABILITY

All data and models supporting the results of this study are available at pending\_publication.

The source code is available at pending\_publication. The software packages used to reproduce the data include zuko [125], sbi [126], stim [127], Qiskit [128], and PyTorch [129].

## Appendix A: Neural spline flows

We adopt a neural spline flow [81] as the density estimator, combining autoregressive transformations with rational-quadratic spline functions to achieve flexible density modeling.

Let $\pmb \theta \in \mathbb { R } ^ { d }$ denote the parameter vector to be inferred and $\mathbf { c } \in \mathbb { R } ^ { d _ { c } }$ c the context vector obtained by embedding the observation x. The flow is composed of $\dot { T }$ autoregressive layers, $f _ { \phi } = f ^ { ( T ) } \circ \cdot \cdot \cdot \circ f ^ { ( 1 ) }$ , as illustrated in Fig. 8(a).

Within each layer $f ^ { ( t ) }$ , the transformation acts coordinatewise $\left[ \mathrm { F i g . \ 8 ( b ) } \right]$ :

$$
\begin{array} { r } { \boldsymbol { \theta } _ { i } ^ { ( t ) } = g _ { i } \Big [ \boldsymbol { \theta } _ { i } ^ { ( t - 1 ) } ; \boldsymbol { \psi } _ { i } \Big ( \boldsymbol { \theta } _ { < i } ^ { ( t - 1 ) } ; c \Big ) \Big ] , \quad i = 1 , \dots , d , } \end{array}\tag{A1}
$$

where $g _ { i } ( \cdot ; \psi _ { i } )$ is a monotone scalar spline transform. The spline parameters ψ<sub>i</sub> are produced by a masked autoregressive network [80, 130], implemented as a multi-layer perceptron (MLP) with binary-masked weight matrices. This masking ensures that each $\psi _ { i }$ depends only on the preceding dimensions $\pmb { \theta } _ { < i } ^ { ( t - 1 ) }$ and the context c, thereby enforcing a lower-triangular Jacobian. The log-determinant therefore decomposes exactly as:

$$
\log \left| \operatorname* { d e t } \frac { \partial f _ { \phi } ^ { - 1 } } { \partial \theta } \right| = - \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { d } \log \left| \frac { \partial g _ { i } \Big ( \theta _ { i } ^ { ( t - 1 ) } ; \psi _ { i } \Big ) } { \partial \theta _ { i } ^ { ( t - 1 ) } } \right| ,\tag{A2}
$$

requiring only d scalar derivative evaluations per layer. Consecutive layers use alternating variable orderings: if layer $f ^ { ( t ) }$ uses the identity permutation $( 1 , 2 , \ldots , d )$ , then $f ^ { \left( t + 1 \right) }$ uses the reverse permutation $( d , d - 1 , \ldots , 1 )$ , ensuring full coupling among all parameters.

The spline transform $g _ { i }$ in each coordinate is constructed by stitching together $K$ rational-quadratic segments on $[ - B , B ]$ , with the identity applied outside this interval [Fig. $8 ( \mathrm { c } ) ]$ . The interval is partitioned into $K$ bins by $\dot { K _ { \mathbf { + } } } 1$ knot points $( x ^ { ( k ) } , y ^ { ( k ) } ) , k = 0 , \dots , K$ Starting from $x ^ { ( 0 ) } = \dot { y } ^ { ( 0 ) } = - \dot { B } ,$ successive knots are placed according to $x ^ { ( k ) } \ = \ x ^ { ( k - 1 ) } \ + \ w ^ { ( k ) }$ and $y ^ { ( k ) } \stackrel { \cdot } { = } y ^ { ( k - 1 ) } + h ^ { ( k ) }$ , where $w ^ { ( k ) }$ and $h ^ { ( k ) }$ are the bin widths and heights, respectively. At each knot, the spline slope is specified by a derivative $d ^ { ( k ) }$

The network outputs $3 K - 1$ quantities per coordinate, $\psi _ { i } \in \mathbb { R } ^ { 3 K - 1 }$ : K widths, K heights, and $K - 1$ interiorknot derivatives. Widths and heights are normalized via softmax and accumulated to span $[ - B , B ]$ . The $K - 1$ interior derivatives are exponentiated to enforce strict positivity; the two boundary derivatives $d ^ { ( 0 ) }$ and $d ^ { ( K ) }$ are fixed to 1, ensuring a smooth transition to the identity tails. Within the k-th bin, let $\textstyle \xi = { \frac { \theta - x ^ { ( k ) } } { x ^ { ( k + 1 ) } - x ^ { ( k ) } } } \in [ 0 , 1 ]$ denote the normalized position and $\begin{array} { r } { s _ { k } = \frac { y ^ { ( k + 1 ) } - y ^ { ( k ) } } { x ^ { ( k + 1 ) } - x ^ { ( k ) } } } \end{array}$ the bin slope. The transform and its derivative are obtained by [81]:

$$
g ( \theta ; \psi ) = y ^ { ( k ) } + \frac { ( y ^ { ( k + 1 ) } - y ^ { ( k ) } ) [ s _ { k } \xi ^ { 2 } + d ^ { ( k ) } \xi ( 1 - \xi ) ] } { s _ { k } + [ d ^ { ( k + 1 ) } + d ^ { ( k ) } - 2 s _ { k } ] \xi ( 1 - \xi ) } ,
$$

$$
g ^ { \prime } ( \theta ; \psi ) = \frac { s _ { k } ^ { 2 } [ d ^ { ( k + 1 ) } \xi ^ { 2 } + 2 s _ { k } \xi ( 1 - \xi ) + d ^ { ( k ) } ( 1 - \xi ) ^ { 2 } ] } { [ s _ { k } + ( d ^ { ( k + 1 ) } + d ^ { ( k ) } - 2 s _ { k } ) \xi ( 1 - \xi ) ] ^ { 2 } } .\tag{A3}
$$

The transform is twice continuously diferentiable at each knot and analytically invertible in closed form.

The model hyperparameters used in this work are presented in Appendix J.

(a)  
![](images/4d650d86499f95c52f8864517f9ca98f1d590eeba034c4fc8e36307801d5bc2f.jpg)

![](images/547b209150c582967a3f50dc04b294e41fe2f3cfb40a195aece314ff5b94e5c2.jpg)  
FIG. 8. Neural spline flow architecture. (a) A base sample z is transformed through T autoregressive spline layers into a posterior parameter sample ${ \pmb \theta } ^ { ( T ) }$ , with each layer conditioned on the context c derived from the observation x. (b) Structure of an autoregressive spline layer, illustrated for three coordinates. A masked autoregressive MLP generates the spline parameters ψ from the preceding coordinates $\pmb { \theta } _ { < i } ^ { ( t - 1 ) }$ and the context c. The scalar transforms $g _ { i }$ then map each input coordinate to $\theta _ { i } ^ { ( t ) }$ (c) A monotonic rational-quadratic spline illustrated with $K + 1 = 9$ knots, specified by bin widths $w ^ { ( k ) }$ , heights $h ^ { ( k ) }$ , and knot derivatives $d ^ { ( k ) }$ , with identity tails outside $[ - B , B ]$

## Appendix B: Pauli propagation

The generation of extensive datasets for SBI requires the repeated evaluation of noisy quantum expectation values. To bypass the $\mathcal { O } ( 4 ^ { n } )$ memory requirement of densitymatrix simulators, we employ Pauli propagation—a classical framework that evolves observables in the Heisenberg picture [41, 131].

Given an n-qubit initial state ρ and a quantum channel composed of layers $\{ { \mathcal E } _ { l } \}$ , the expectation value of an observable O is given by the dual action:

$$
\langle { \cal O } \rangle = \mathrm { T r } [ \mathcal { E } ( \rho ) { \cal O } ] = \mathrm { T r } [ \rho \mathcal { E } ^ { \dag } ( { \cal O } ) ]\tag{B1}
$$

where ${ \mathcal { E } } ^ { \dagger } = { \mathcal { E } } _ { 1 } ^ { \dagger } \circ { \mathcal { E } } _ { 2 } ^ { \dagger } \circ \cdots \circ { \mathcal { E } } _ { L } ^ { \dagger }$ denotes the adjoint channel. We decompose O into a sum of Pauli strings, $\begin{array} { r } { O = \sum _ { \alpha = 1 } ^ { k } c _ { \alpha } P _ { \alpha } } \end{array}$ with $P _ { \alpha } \in \{ I , X , Y , Z \} ^ { \otimes n }$ . The propagation of O is then formalized by tracking the evolution of the coeficients $c _ { \alpha }$ and their corresponding strings $P _ { \alpha }$ via the Pauli transfer matrix (PTM) formalism. Specifically, any n-qubit quantum operation $\mathcal { E } _ { l }$ is completely characterized by a $4 ^ { n } \times 4 ^ { n }$ transformation matrix with elements $[ \mathcal { E } _ { l } ] _ { i j } = \mathrm { \bar { T r } } [ P _ { i } \mathcal { E } _ { l } ( P _ { j } ) ]$ ]. Operating in the Heisenberg picture requires the dual map $\mathcal { E } _ { l } ^ { \dagger }$ , which is represented by the transpose of the PTM, $[ \mathcal { E } _ { l } ^ { \dagger } ] _ { i j } = [ \mathcal { E } _ { l } ] _ { j i }$ . Thus, the backward evolution of a single Pauli string $P _ { \alpha }$ through layer l dictates a mapping to a new linear combination of

strings:

$$
\mathcal { E } _ { l } ^ { \dagger } [ P _ { \alpha } ] = \sum _ { \beta } [ \mathcal { E } _ { l } ^ { \dagger } ] _ { \beta \alpha } P _ { \beta } .\tag{B2}
$$

By linearity, the entire observable is updated at each layer $\mathcal { E } _ { l } .$ , mapping the coeficient distribution to $c _ { \beta } ^ { l } =$ $\begin{array} { r } { \sum _ { \alpha } c _ { \alpha } [ \mathcal { E } _ { l } ^ { \dagger } ] _ { \beta \alpha } . } \end{array}$ , yielding the updated observable $\mathcal { E } _ { l } ^ { \dagger } ( O ) =$ $\sum _ { \beta } c _ { \beta } ^ { l } P _ { \beta }$

Although the PTM formally acts on a $4 ^ { n }$ -dimensional vector space, evaluating Eq. (B2) does not incur the prohibitive (16<sup>n</sup>) cost of dense matrix-vector multiplication. This global $4 ^ { n } \times 4 ^ { n }$ matrix is never explicitly constructed in memory. Instead, the framework utilizes the backward causal cone of each observable—the subset of earlier circuit gates that can afect it under operator propagation— together with the sparsity of the active operator subspace. Gates outside this cone trivially commute with the active Pauli strings and can therefore be skipped. For a k-local operation within the lightcone, the algebraic transformation reduces to a localized $4 ^ { k } \times 4 ^ { k }$ sub-block. By dynamically tracking only the non-zero coeficients via sparse data structures, the prohibitive bottleneck is avoided.

Consequently, the computational complexity is governed by the branching factor of the propagation path:

a. Cliford propagation. All Cliford gates are 1- branching, mapping each Pauli string to a single Pauli string $( { \mathcal { C } } ^ { \dagger } [ P _ { \alpha } ] \propto P _ { \beta } )$ , giving an $\mathcal { O } ( N _ { \mathscr { C } } )$ cost per propagating Pauli string, where $N _ { \mathcal { C } }$ is the number of Cliford gates. However, entangling Cliford gates (e.g., CNOT) can alter the Pauli weight of propagating strings, which tends to indirectly amplify the branching induced by subsequent non-Cliford gates.

b. Pauli noise as eigenvalue scaling. In the Pauli basis, Pauli channels $\Lambda _ { p }$ act as 1-branching eigenvalue contractions $( \Lambda _ { p } ^ { \dagger } [ P _ { \alpha } ] = \dot { \lambda } _ { P _ { \alpha } } P _ { \alpha } )$ . Incorporating complex noise profiles merely rescales existing coeficients, incurring an $\mathcal { O } ( N _ { \mathrm { n o i s e } } )$ cost per propagating Pauli string, where $N _ { \mathrm { n o i s e } }$ is the number of local Pauli noise channels.

c. Non-Cliford propagation. Non-Cliford Pauli rotations $R _ { \sigma } ( \theta ) \stackrel { \sim } { = } { e } ^ { \bar { - } i \theta \bar { \sigma } / 2 }$ are either 1-branching or ${ \mathcal { Q } } -$ branching. Specifically, the evolution of a Pauli string $P$ follows:

$$
R _ { \sigma } ( \theta ) [ P ] = \left\{ { \begin{array} { l l } { P } & { { \mathrm { i f ~ } } [ \sigma , P ] = 0 } \\ { \cos ( \theta ) P + \sin ( \theta ) P ^ { \prime } } & { { \mathrm { o t h e r w i s e } } } \end{array} } , \right.\tag{B3}
$$

where $P ^ { \prime } = i [ \sigma , P ] / 2$ is a newly generated Pauli string.

When the expanding operator lightcone encounters an increasing number of branching gates, the number of Pauli strings may grow exponentially. To maintain computational tractability, this proliferation necessitates heuristic truncation strategies. These typically include coeficient truncation, which discards Pauli paths whose accumulated amplitude falls below a precision threshold $\left( \left. c _ { \alpha } \right. < \epsilon \right)$ , and weight truncation, which eliminates strings exceeding a maximum Pauli weight $( w ( P ) > w _ { \mathrm { m a x } } ) ~ [ 4 1 ]$ For current hardware, the latter is also motivated by the physical reality that high-weight paths are typically exponentially suppressed by noise, contributing negligibly to local observables.

## Appendix C: Benchmarking Pauli-propagation simulators

## 1. Simulator for Pauli noise learning

In our special cases of Pauli noise learning (see Fig. 2), we restrict all non-Cliford operations to the initial state preparation layer. Instead of explicitly propagating the Pauli strings backward through this densely branching initial layer, we terminate the Heisenberg evolution after traversing the Cliford and noise layers. At this terminal step, the non-Cliford operations simply define an unentangled product state, $\left| \psi _ { \mathrm { i n i t } } \right. = \bigotimes _ { i = 1 } ^ { n ^ { - } } U _ { i } | 0 \rangle$ . The expectation value of any fully propagated Pauli string $\boldsymbol { P } = \boldsymbol { \otimes } _ { i = 1 } ^ { n } \boldsymbol { P } _ { i }$ can therefore be evaluated analytically by factorizing it over the individual qubits:

$$
\langle P \rangle = \prod _ { i = 1 } ^ { n } \langle 0 | U _ { i } ^ { \dagger } P _ { i } U _ { i } | 0 \rangle\tag{C1}
$$

For instance, when applying $U _ { i } = R _ { z } ( \pi / 4 ) R _ { y } ( \pi / 4 )$ to each qubit identically, these local expectation values re-

![](images/2141188cd99c138f8fd401d2fc9a9d26ccf3e5611f1c1a760237906fbf2c935f.jpg)

![](images/91aabbdbe5ef9d4d6bbed00bb2d13c75541714943bda4be77309ea8e26f13985.jpg)  
(d)

![](images/dd37533e02cdd90bda4430ef558aa75c3f9cc10a02d05a654dc03ee1a82c47a2.jpg)

![](images/7ef391c02902b519abee01376b92e05314baade52c7806fd432237ea017410d4.jpg)  
FIG. 9. Benchmarking the simulator for Pauli noise learning. (a) Mean absolute error (MAE) between our method and Qiskit Aer density-matrix simulator over all $k = 1 2 n - 9$ Pauli observables (system size $n = 2 { - } 1 4 )$ . (b) Wall time comparison for both methods over the same setting as in panel (a). (c) Wall time of our method for a single observable $\langle Z _ { 1 } Z _ { 2 } \rangle$ as a function of $n = 1 0 { - } 1 0 0 0$ , with an ${ \mathcal { O } } ( n )$ reference. (d) Total wall time for all k observables, with an $\overset { \cdot } { \mathcal { O } } ( n ^ { 2 } )$ reference. Reference curves are initialized at the leftmost data point. All quantities are averaged over five independent random seeds (Pauli rotation angles and Pauli noise coeficients).

duce to fixed analytical constants:

$$
\langle 0 | U _ { i } ^ { \dagger } P _ { i } U _ { i } | 0 \rangle = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } P _ { i } = I } \\ { 1 / 2 , } & { { \mathrm { i f ~ } } P _ { i } = X { \mathrm { ~ o r ~ } } Y { \mathrm { ~ . ~ } } } \\ { 1 / { \sqrt { 2 } } , } & { { \mathrm { i f ~ } } P _ { i } = Z } \end{array} \right. }\tag{C2}
$$

This local evaluation requires merely ${ \mathcal { O } } ( n )$ operations per string. This architectural advantage eliminates the nominal ${ \bar { \mathcal { O } } } ( 2 ^ { n } )$ branching overhead associated with the initial n non-Cliford gates. This approach guarantees exact, non-truncated noisy simulation with an eficient complexity of $\mathcal { O } ( n k )$ , representing a bilinear scaling with respect to both the system size n and the number of Pauli strings k in the target observable. Such an eficiency profile theoretically enables the use of SBI to characterize sparse Markovian Pauli noise in large-scale circuits exceeding 100 qubits.

We benchmark our simulation approach against the Qiskit Aer density-matrix simulator, as illustrated in $\mathrm { F i g . ~ 9 }$ . The benchmark evaluates the full set of $k =$ $1 2 n - 9$ (1- and 2-local) Pauli observables. All tests were run on a single thread of an Apple M4 Pro CPU, using a linear circuit template in which n 1 CNOT gates, each followed by a two-qubit Pauli noise channel, are applied sequentially between nearest-neighbor qubits. Both methods demonstrate numerical agreement within floating-point precision [Fig. 9(a)]. While the Qiskit Aer runtime grows exponentially with n and becomes computationally prohibitive beyond $n \sim 1 4 ~ [ \mathrm { F i g . ~ 9 ( b ) } ]$ our approach reduces the cost per observable to ${ \mathcal { O } } ( n )$ [Fig. 9(c)]. This eficiency allows a single $\mathrm { C P U }$ thread to process an observable for $n = 1 0 0 0$ in under 4 ms. Furthermore, the overall complexity for all k observables scales as $\mathcal { O } ( n k ) \sim \mathcal { O } ( n ^ { 2 } )$ [Fig. 9(d)], enabling simulations of $n = 1 0 0$ in about 0.4 s and $n = 1 0 0 0$ in roughly 40 s.

## 2. Simulator for Rydberg Hamiltonian learning

We now benchmark the weight-truncated Paulipropagation simulator used for Rydberg Hamiltonian learning. The circuit structure is identical to that of Sec. VI in the main text (a single short-time Trotter step applied to $\left| + \right. ^ { \otimes N } )$ . We take $ { \langle Z _ { 1 } Z _ { 2 } \rangle }$ as a representative observable and average over twenty random displacement samples. We sweep the weight-truncation order from $w \leq 1$ to $w \leq 5$

On $2 \times 2 , ~ 3 \times 3$ , and $4 \times 4$ lattices [Fig. 10(a)], the truncation error converges rapidly and uniformly in n: $w \leq 1$ is clearly insuficient, $w \leq 2$ already reduces it by more than an order of magnitude, and from $w \leq 3$ onward the error is negligible. The statevector runtime rises steeply with system size while Pauli propagation stays modest, making propagation up to nearly an order of magnitude cheaper at $n = 1 6 ~ [ \mathrm { F i g . ~ 1 0 ( b ) } ]$ ]. Extending to square lattices $\mathrm { u p }$ to $n = 1 4 4$ , the estimated $ { \langle Z _ { 1 } Z _ { 2 } \rangle }$ maintains convergence for $w \ge 3 ~ [ \mathrm { F i g . ~ 1 0 ( c ) } ]$ , while the propagation wall time follows a clean polynomial trend within tens of milliseconds [Fig. 10(d)]. These results confirm that, for the (at most) 2-local observables entering the observation, the weight-5 truncation used in the main text incurs negligible error while preserving polynomial cost.

## Appendix D: Local gauge ambiguity

The unlearnable degrees of freedom observed in the posterior distributions arise from a fundamental gauge symmetry inherent to composite gate benchmarking. For a composition of two gates $\mathcal { G } _ { 2 } \circ \mathcal { G } _ { 1 }$ , a gauge transformation satisfies $\tilde { \mathcal { G } } _ { 2 } \circ \tilde { \mathcal { G } } _ { 1 } = \left( \mathcal { G } _ { 2 } \circ \mathcal { V } ^ { - 1 } \right) \circ \left( \mathcal { V } \circ \mathcal { G } _ { 1 } \right)$ , leaving the composite channel invariant while redistributing errors between the two gates.

We build intuition on an illustrative five-qubit brickwall circuit (compare Fig. 2 in the main text), whose posteriors are shown in Fig. 11. The first layer applies $\mathrm { C N O T _ { 1 , 2 } }$ and $\mathrm { C N O T _ { 3 , 4 } }$ , and the second layer applies $\mathrm { C N O T _ { 2 , 3 } }$ and $\mathrm { C N O T _ { 4 , 5 } ; }$ each CNOT is followed by its own two-qubit Pauli noise channel. Two-qubit Pauli labels refer to the qubit pair of the corresponding gate, ordered by qubit index; for example, IX for $\mathrm { C N O T _ { 1 , 2 } }$ denotes an X error on $q _ { 2 }$ . Qubits $q _ { 2 } , \ q _ { 3 }$ , and $q _ { 4 }$ are each shared by a firstlayer (upstream) gate and a second-layer (downstream) gate; we refer to them as bridge qubits.

(b)  
![](images/92dce4aa8803ad39e006742d8076652b976a939f1f4b2499b19d927949d176a6.jpg)

![](images/68c38ce792ed314ae29f81e9420f29c95b348d3db6067b5a3f66cfd27516cb6b.jpg)  
(d)

![](images/18dd32a72b7777f781e9d2790fdee0b010c880a25271aff2db53fd1d8baef82a.jpg)

![](images/3b9653f838e7d47603c8461e36b1a4445d29d95c046facc2dbb03e6c7d5854e8.jpg)  
FIG. 10. Benchmarking the simulator for Rydberg Hamiltonian learning. (a) Absolute error between the weight-truncated Pauli propagation and the Qiskit Aer statevector simulator for the observable $ { \left. Z _ { 1 } Z _ { 2 } \right. }$ , shown for truncation orders $w \leq 1 { - } 5$ on $2 \times 2 , 3 \times 3$ , and $4 \times 4$ lattices $( n = 4 , 9 , 1 6 )$ . (b) Wall time comparison between the statevector reference and Pauli propagation over the same setting as in panel (a). (c) Estimated $\langle Z _ { 1 } Z _ { 2 } \rangle$ as a function of system size up to the 12×12 lattice $( n = 1 4 4 )$ , across the same truncation orders. (d) Wall time for the same scaling sweep as in panel (c). All quantities are averaged over twenty random displacement samples.

Consider a single-qubit Pauli error $E \in \{ X , Y , Z \}$ on a bridge qubit, generated by the noise channel of the upstream gate. Before reaching the measurement, this error passes through the downstream gate $C \ ( \mathrm { a \ C N O T } )$ , which maps it to $\mathbf { \bar { \mathit { E } } ^ { \prime } } = \mathbf { \bar { \mathit { C } } } \mathit { E C } ^ { \dagger }$ . Since $C E = E ^ { \prime } C$ , an error E occurring before C acts identically to the error $E ^ { \prime }$ occurring after $C .$ Because $E ^ { \prime }$ is also one of the Pauli terms of the downstream noise channel, the upstream probability of $E$ and the downstream probability of $E ^ { \prime }$ enter the composite channel only through their combination. Equivalently, a single-qubit Pauli channel on a bridge qubit can be shifted from one gate to the other without changing the composite channel, which gives three gauge directions X, Y, Z per bridge qubit.

Using the CNOT conjugation rules, written in (control, target) order,

$$
\begin{array} { r } { X I  X X , \quad Y I  Y X , \quad Z I  Z I , } \\ { I X  I X , \quad I Y  Z Y , \quad I Z  Z Z , } \end{array}\tag{D1}
$$

![](images/c9a9b7ea5931821b4dd25c2059393b0c49d35a64e0e271f8241dfe978b1e6305.jpg)  
FIG. 11. Posterior Pauli noise distributions of a five-qubit circuit with brickwall connectivity. Layer one consists of $\mathrm { C N O T _ { 1  2 } }$ and $\mathrm { C N O T _ { 3  4 } ; }$ layer two consists of $\mathrm { C N O T _ { 2  3 } }$ and $\mathrm { C N O T _ { 4  5 } }$

the coupled pairs (upstream  downstream) are:

1. Bridge $q _ { 2 }$ (control of $\mathrm { C N O T _ { 2 , 3 } } )$ : IX, IY, IZ of $\mathrm { C N O \bar { T } _ { 1 , 2 } }  \{ X X , Y X , Z I \}$ of $\mathrm { C N O T _ { 2 , 3 } }$ , since

$$
\begin{array} { r l } & { I _ { 1 } X _ { 2 } I _ { 3 } \xrightarrow { \mathrm {  { C N O T } _ { 2 , 3 } } } I _ { 1 } X _ { 2 } X _ { 3 } , } \\ & { I _ { 1 } Y _ { 2 } I _ { 3 } \xrightarrow { \mathrm {  { C N O T } _ { 2 , 3 } } } I _ { 1 } Y _ { 2 } X _ { 3 } , } \\ & { I _ { 1 } Z _ { 2 } I _ { 3 } \xrightarrow { \mathrm {  { C N O T } _ { 2 , 3 } } } I _ { 1 } Z _ { 2 } I _ { 3 } . } \end{array}
$$

2. Bridge $q _ { 3 }$ (target of CNOT<sub>2,3</sub>): XI, Y I, ZI of <sup>CNOT</sup>3,4 ↔ {<sup>IX,</sup> <sup>ZY,</sup> <sup>ZZ</sup>} <sup>of</sup> $\mathrm { C N O T _ { 2 , 3 } }$ , since

$$
\begin{array} { r } { I _ { 2 } X _ { 3 } I _ { 4 } \xrightarrow { \mathrm { C N O T } _ { 2 , 3 } } I _ { 2 } X _ { 3 } I _ { 4 } , } \\ { I _ { 2 } Y _ { 3 } I _ { 4 } \xrightarrow { \mathrm { C N O T } _ { 2 , 3 } } Z _ { 2 } Y _ { 3 } I _ { 4 } , } \\ { I _ { 2 } Z _ { 3 } I _ { 4 } \xrightarrow { \mathrm { C N O T } _ { 2 , 3 } } Z _ { 2 } Z _ { 3 } I _ { 4 } . } \end{array}
$$

3. Bridge $q _ { 4 }$ (control of CNOT ): IX, IY, IZ of $\mathrm { C N O \bar { T } _ { 3 , 4 } }  \{ X X , Y X , Z I \}$ of $\mathrm { C N O T } _ { 4 , 5 } ,$ since

$$
\begin{array} { r l } & { I _ { 3 } X _ { 4 } I _ { 5 } \xrightarrow { \mathrm {  { C N O T } _ { 4 , 5 } } } I _ { 3 } X _ { 4 } X _ { 5 } , } \\ & { I _ { 3 } Y _ { 4 } I _ { 5 } \xrightarrow { \mathrm {  { C N O T } _ { 4 , 5 } } } I _ { 3 } Y _ { 4 } X _ { 5 } , } \\ & { I _ { 3 } Z _ { 4 } I _ { 5 } \xrightarrow { \mathrm {  { C N O T } _ { 4 , 5 } } } I _ { 3 } Z _ { 4 } I _ { 5 } . } \end{array}
$$

Each of these nine pairs carries one gauge degree of freedom, so neither parameter in a pair can be determined individually. Collecting the pairs by gate gives the unlearnable parameters of each gate position:

1. Odd-layer boundary $( \mathrm { C N O T _ { 1 , 2 } } ;$ bridge $q _ { 2 } )$ : 3 parameters, $\{ I X , I Y , { \bar { I } } Z \}$

2. Odd-layer interior $\mathrm { ( C N O T _ { 3 , 4 } ; }$ bridges $q _ { 3 } , \ q _ { 4 } ) \colon \ 6$ parameters, XI, Y I, ZI via $q _ { 3 }$ and $\{ I X , I Y , I Z \}$ via $q _ { 4 }$

3. Even-layer interior $( \mathrm { C N O T _ { 2 , 3 } } ;$ bridges $q _ { 2 } , q _ { 3 } ) \colon 6$ parameters, $\{ X X , Y X , Z I \}$ via $q _ { 2 }$ and $\{ \bar { I } X , \bar { Z } \dot { Y } , Z \bar { Z } \}$ via $q _ { 3 }$

4. Even-layer boundary $( \mathrm { C N O T } _ { 4 , 5 } ;$ bridge q<sub>4</sub>): 3 parameters, XX, YX, ZI via $q _ { 4 }$

In total, the three bridge qubits yield 18 unlearnable parameters, consistent with the broad posteriors in Fig. 11.

Generalizing to n qubits, the brickwall circuit considered here contains $n - 2$ bridge qubits, each shared by two adjacent gates and contributing three gauge directions, i.e., three coupled pairs. This yields $6 ( n - 2 )$ unlearnable parameters in total. The specific error coeficients that become coupled are determined by how each bridge qubit enters the adjacent gates (as control or target), and thus vary with the circuit topology.

Without external constraints—such as independent single-gate calibrations or gauge-fixing circuit elements— these gauge parameters are fundamentally unresolvable from composite circuit measurements alone. This is not a limitation of the SBI framework but an inherent consequence of the objective itself: resolving individual gatelevel noise from composite measurements. Standard quantum process tomography of the entire gate sequence would yield only the aggregate process matrix, providing no additional power to decompose noise attribution among constituent gates. Crucially, even tomographically complete input states and measurements cannot resolve this ambiguity, as it arises from local algebraic equivalences at the bridge qubits that leave the composite channel itself invariant. The inclusion of SPAM noise would further exacerbate the issue by introducing additional gauge freedom [7, 60].

## Appendix E: Complete posterior distributions for 50-qubit Pauli noise

As a supplement to Fig. $\mathrm { 3 ( a ) }$ in the main text, Fig. 12 presents the parameter-wise regression for the 50-qubit noise model. Over all 735 parameters (left), the posterior means show considerable scatter $( R ^ { 2 } = 0 . 7 5 9$ , MAE $= 1 . 1 \times 1 0 ^ { - 4 } )$ due to the gauge parameters. Restricting to the 447 gauge-free parameters (right) yields tight agreement with the ground truth $( \dot { R ^ { 2 } } = \mathrm { 0 . 9 9 6 }$ , MAE $\stackrel { - } { = } 2 . 3 \times 1 0 ^ { - 5 } )$ , confirming that the identifiable noise structure is accurately resolved despite the high dimensionality.

Figure 13 displays the complete posterior distributions of the 50-qubit noise model. These results are obtained from the amortized NPE model trained on 30,000 simulations. The gauge-freedom patterns exhibit four distinct gate categories: odd-layer top boundary $\left( \mathrm { C N O T _ { 1 , 2 } } \right)$ , oddlayer interior $\left( \mathrm { C N O T _ { 3 , 4 } } \mathrm { - C N O T _ { 4 7 , 4 8 } } \right)$ , odd-layer bottom boundary $\mathrm { ( C N O T _ { 4 9 , 5 0 } ) }$ , and even-layer interior $\mathrm { ( C N O T _ { 2 , 3 ^ { - } } }$ $\mathrm { C N O T } _ { 4 8 , 4 9 } )$ . Compared with the 5-qubit case (Fig. 11), the larger circuit gains an odd-layer bottom boundary but loses the even-layer boundary.

![](images/4e60737a02ba3e1da0c7c870c318c6b4ed669283f80def6be6753390880a78ad.jpg)  
FIG. 12. Parameter-wise regression for 50-qubit Pauli noise learning. Posterior means versus ground-truth values for all 735 parameters (left) and the 447-parameter gauge-free subset (right). Vertical bars denote ±1 posterior standard deviation among 2,000 samples.

## Appendix F: Standard quantum error mitigation

## 1. Probabilistic error cancellation

PEC cancels noise statistically by sampling operations from a quasiprobability representation of the inverse noise channel [62, 63]. Each Pauli noise channel $\Lambda _ { k }$ is diagonal in the Pauli basis with eigenvalues $\lambda _ { k , Q } = \mathrm { T r } [ Q ^ { \dagger } \Lambda _ { k } ( Q ) ] / 2 ^ { n }$ for each Pauli string Q. The inverse channel is decomposed as $\begin{array} { r } { \Lambda _ { k } ^ { - 1 } ( \cdot ) = \sum _ { P } q _ { k , P } P ( \cdot ) P , } \end{array}$ where the quasiprobability coeficients are recovered via $\begin{array} { r } { q _ { k , P } = \frac { 1 } { 4 ^ { n } } \tilde { \sum _ { Q } } \tilde { \lambda _ { k , Q } ^ { - 1 } } ( - 1 ) ^ { [ P , Q ] } } \end{array}$ , with $( - 1 ) ^ { [ P , Q ] }$ equal to +1 if $[ P , Q ] = 0$ and 1 otherwise, and $n = 2$ for the two-qubit channels considered here. The coeficients $q _ { k , P }$ are real but not necessarily positive, reflecting the non-physicality of the inverse channel.

The sampling distribution at each gate k is $\pi _ { k , P } =$ $| q _ { k , P } | / \gamma _ { k }$ with the per-gate sampling overhead $\gamma _ { k } =$ $\sum _ { P } \left| q _ { k , P } \right|$ . For each circuit instance $C _ { s } .$ , random Pauli corrections $P _ { k } ^ { ( s ) }$ are independently drawn from $\pi _ { k }$ for every gate, and the mitigated expectation value is estimated via the Monte Carlo average:

$$
\langle O \rangle _ { \mathrm { P E C } } = \frac { \gamma } { N } \sum _ { s = 1 } ^ { N } \mathrm { s g n } ( w _ { s } ) o _ { s } ,\tag{F1}
$$

where $o _ { s }$ is the measurement outcome for instance $C _ { s } .$ $\begin{array} { r } { w _ { s } = \prod _ { k } q _ { k , P _ { k } ^ { ( s ) } } } \end{array}$ accumulates the sign into the estimator, and $\begin{array} { r } { \gamma = \prod _ { k } \ddot { \gamma _ { k } } } \end{array}$ is the total sampling overhead.

## 2. Zero-noise extrapolation

ZNE recovers ideal expectation values by extrapolating measurements at amplified noise levels to the zero-noise limit [63–65, 132]. We implement ZNE using two complementary noise-amplification strategies, depending on whether a noise model is available.

a. Probabilistic error amplification $( P E A )$ When a Pauli noise model is available, here supplied by the SBI posterior, we amplify noise by inserting additional random Pauli operations [65]. For a noise gain $G \geq 1$ , we define the supplementary error probabilities as

$$
\begin{array} { l } { { p _ { k , P } ^ { \mathrm { a d d } } ( G ) = \frac { 1 - ( 1 - 2 p _ { k , P } ) ^ { G - 1 } } { 2 } } } \\ { { = ( G - 1 ) p _ { k , P } + O ( p _ { k , P } ^ { 2 } ) , } } \end{array}\tag{F2}
$$

where the expansion holds for fixed G in the weak-noise regime. The factor $G - 1$ accounts for the native noise already present, so composing the supplementary channel with the native noise amplifies the noise strength by G to first order. At each gate $k ,$ a supplementary Pauli operator $P$ is sampled and inserted with probability $p _ { k , P } ^ { \mathrm { a d d } } ( G )$ The amplified expectation value $\langle O \rangle ^ { ( G ) }$ is estimated at each gain over an ensemble of independently sampled circuits, and the zero-noise limit $( G = 0 )$ is extracted by extrapolation using linear, quadratic, or exponential fits. PEA can realize non-integer gains, and we use it for the posterior-driven ZNE in Fig. 4(a) of the main text $( G \in \{ 1 . 6 , 2 . 2 , 2 . 8 \} )$ .

b. Gate folding. When no noise model is assumed, noise can be amplified by gate folding [132]. Each folded gate U is replaced by $U ( U ^ { \dagger } U ) ^ { n }$ , preserving the ideal action while amplifying the efective noise by $G = 2 n + 1$ . Unlike PEA, which rescales the noise channel in place, folding lengthens the circuit, incurring a higher coherence-time overhead, while restricting the accessible gains to odd integers. We use it for the model-agnostic ZNE baselines in Fig. 5 of the main text $( G \in \{ 1 , 3 , 5 \} )$ , which amplify the ground-truth Markovian noise carried by the CNOT gates of the Trotterized circuit.

## Appendix G: NPE-based digital-twin ML-QEM

The digital-twin denoiser, mentioned in the main text (Sec. IV B), is trained by direct regression of the correction $\pmb { \theta } \equiv \dot { \Delta \mathbf { x } } = \mathbf { x } ^ { \mathrm { e x a c t } } - \dot { \mathbf { x } } ^ { \mathrm { d t } }$ . Here we detail an alternative variant that instead learns the full posterior $p ( \pmb { \theta } | \mathbf { x } ^ { \mathrm { d t } } )$ via NPE. While the posterior-driven twin $s ,$ the observable set, the noise model, the Trotter schedule (14 steps up to $t = 1 . 4 )$ ), and the training-data budget (500 circuits per Trotter step) remain identical to the main implementation, this variant introduces key diferences in its prior specification and training data generation.

The categorical Cliford sampling leaves the prior $\pi ( \theta )$ analytically unspecified, which inherently precludes standard NPE training. To recover the standard NPE approach, we construct a data-driven empirical prior defined as a diagonal Gaussian, $\begin{array} { r } { \pi ( \pmb { \theta } ) = \prod _ { i } \mathcal { N } \big ( \theta _ { i } ; \mu _ { i } , \sigma _ { i } ^ { 2 } \big ) } \end{array}$ with $\mu _ { i }$ and $\sigma _ { i }$ the empirical mean and standard deviation of the correction $\theta _ { i }$ across the training circuit ensemble. Under this prior, we train a conditional normalizing flow $q _ { \phi } ( \pmb \theta | \mathbf x ^ { \mathrm { d t } } , c ) \hat { \mathbf \alpha } \approx p ( \pmb \theta | \mathbf x ^ { \mathrm { d t } } , c )$ , and at deployment correct with the posterior mean:

![](images/8a61660094387242a37e6c8437ea1d574be5c43f879586350302bd781f0fa2c9.jpg)  
FIG. 13. Complete posterior distributions for 50-qubit Pauli noise learning. Posterior samples (box plots) and ground-truth values (red crosses) for all 49 CNOT gates in the brickwall circuit. For each box, the central line marks the median, the box spans the interquartile range (25th–75th percentiles), and the whiskers extend to 1.5× the interquartile range; outliers beyond the whiskers are omitted. Each box summarizes 2,000 posterior samples. Gates are grouped by sublayer: odd-control (top five rows) and even-control (bottom five rows).

![](images/e8c2d7554a71be68d100f4e5bb115ed6b175bfb5dbf1c698bffaf2841f76d4c5.jpg)  
FIG. 14. Overall MAE of the NPE-based SBI-DT, shown in direct comparison with the results in Fig. 5. The plot contrasts this variant against the noisy baseline, quadratic ZNE, and the regression-based SBI-DT. Improvement factors are relative to the noisy baseline.

$$
\mathbf { x } ^ { \mathrm { m i t } } = \mathbf { x } ^ { \mathrm { n o i s y } } + \mathbb { E } _ { q _ { \phi } } \big [ \pmb { \theta } | \mathbf { x } ^ { \mathrm { n o i s y } } , c \big ] ,\tag{G1}
$$

substituting the direct regression estimate used in the main text.

Although the introduction of the empirical prior renders the flow theoretically trainable on the original Cliford ensemble, direct application to these circuits failed to yield any meaningful correction signal, a failure fundamentally rooted in the lack of continuous support in θ required for flow-based density estimation. To circumvent this limitation, we employ a data augmentation scheme, prepending each circuit with a random single-qubit product-state layer $\otimes _ { q } R _ { y } ( \alpha _ { q } ) R _ { x } ( \beta _ { q } ) \left( \alpha _ { q } , \beta _ { q } \stackrel { . } { \sim } \mathcal { U } [ 0 , 2 \pi ) \right)$ prior to the Cliford Trotter blocks. This injection of randomness smoothens the training distribution, which adequately diversifies the data for the flow model to learn.

As demonstrated in Fig. 14 (presented in comparison with the primary results in Fig. 5), the NPE variant reduces the aggregated MAE by 2.4 relative to the noisy baseline, compared to a 2.1 reduction achieved by quadratic ZNE and a 5.3 reduction by direct regression. It thus trails the regression-based approach for pointwise mitigation. This performance gap is expected: direct regression merely learns a point estimate (a deterministic map $\mathbf { x } \mapsto { \pmb \theta } )$ and does not model the distribution of $\theta ,$ rendering it structurally immune to the training failures that discrete support imposes on flow-based density estimation. In contrast, NPE is forced to resolve the far more challenging task of learning the full density. Nevertheless, the NPE variant still yields a marginal improvement over the standard ZNE baseline.

## Appendix H: Matrix product states

The QST simulator in Sec. V represents the variationalcircuit state as a bond- $^ - \chi$ MPS [1, 93, 94],

$$
\left| \psi \right. = \sum _ { s _ { 1 } , \ldots , s _ { n } } A _ { s _ { 1 } } ^ { \left[ 1 \right] } A _ { s _ { 2 } } ^ { \left[ 2 \right] } \cdot \cdot \cdot A _ { s _ { n } } ^ { \left[ n \right] } \left| s _ { 1 } \cdot \cdot \cdot s _ { n } \right. ,\tag{H1}
$$

with each $A _ { s _ { i } } ^ { [ i ] } \in \mathbb { C } ^ { D _ { i - 1 } \times D _ { i } }$ a complex matrix for the physical leg $s _ { i } \in \{ 0 , 1 \}$ , open boundaries $D _ { 0 } = D _ { n } = 1$ , and ramped bond dimensions $D _ { i } = \operatorname* { m i n } ( 2 ^ { i } , 2 ^ { n - i } , \chi )$ with maximum bond dimension $\chi .$ The MPS stores $4 \sum _ { i } D _ { i - 1 } D _ { i }$ real parameters; for $\chi = 2 ^ { c } ~ ( c \in \mathbb { Z } _ { > 0 } )$ and $n \geq 2 c .$ , this is $\begin{array} { r } { 4 ( \bar { n } - 2 c ) \chi ^ { 2 } + \frac { 1 6 } { 3 } ( \check { \chi } ^ { 2 } - 1 ) \stackrel { \cdot } { = } \mathcal { O } ( \bar { n } \chi ^ { 2 } ) } \end{array}$ . The simulator evaluates the MPS’s Pauli expectation values by transfermatrix contraction, without ever forming the dense $2 ^ { n }$ statevector. For an observable $\boldsymbol { P } = \boldsymbol { \otimes } _ { i = 1 } ^ { n } \boldsymbol { P } _ { i }$ , we define at each site i the doubled-layer transfer matrix on the squared bond space ${ \mathbb C } ^ { D _ { i - 1 } } \otimes { \bar { \mathbb C } } ^ { D _ { i - 1 } }  { \mathbb C } ^ { D _ { i } } \otimes { \mathbb C } ^ { D _ { i } }$

$$
T ^ { i } [ P _ { i } ] = \sum _ { s _ { i } , s _ { i } ^ { \prime } } ( P _ { i } ) _ { s _ { i } s _ { i } ^ { \prime } } \overline { { { A _ { s _ { i } } ^ { [ i ] } } } } \otimes A _ { s _ { i } ^ { \prime } } ^ { [ i ] } ,\tag{H2}
$$

where $( P _ { i } ) _ { s _ { i } s _ { i } ^ { \prime } } = \langle s _ { i } | P _ { i } | s _ { i } ^ { \prime } \rangle$ is a matrix element of the single-qubit operator $P _ { i } , \ { \overline { { A _ { s _ { i } } ^ { [ i ] } } } }$ is the complex-conjugate (bra) copy of the MPS matrix and $A _ { s _ { i } ^ { \prime } } ^ { [ i ] }$ the original (ket) copy, with the Kronecker product pairing their bond indices on the doubled (bra–ket) space. The expectation then factorizes along the chain,

$$
\langle P \rangle = { \frac { T ^ { 1 } [ P _ { 1 } ] T ^ { 2 } [ P _ { 2 } ] \cdot \cdot \cdot T ^ { n } [ P _ { n } ] } { Z } } ,\tag{H3}
$$

where $Z = \langle \psi | \psi \rangle = E ^ { 1 } E ^ { 2 } \cdot \cdot \cdot E ^ { n }$ is the normalization factor and $\begin{array} { r } { E ^ { i } \equiv T ^ { i } [ I ] = \sum _ { s i } \overline { { A _ { s _ { i } } ^ { [ i ] } } } \otimes A _ { s _ { i } } ^ { [ i ] } } \end{array}$ . The product in Eq. (H3) is evaluated by a single left-to-right sweep that carries an environment—a running $\chi \times \chi$ matrix holding the partial contraction of all sites traversed so far, with one index on the bra layer and one on the ket. At each site the environment is updated by the action of $T ^ { i } [ P _ { i } ]$ , computed as a few $\chi \times \chi$ matrix products at $\mathcal O ( \chi ^ { 3 } )$ cost rather than by assembling the doubled-layer matrix $T ^ { i }$ , which would cost $\mathcal { O } ( \chi ^ { 4 } )$ . Each observable thus costs $\mathcal { O } ( n \chi ^ { 3 } )$ , and the full set of $1 2 n - 9$ Pauli operators costs $\mathcal { O } ( n ^ { 2 } \chi ^ { 3 } )$

## Appendix I: Quantum state tomography with Cholesky parameterization

In the main text (Sec. V), the state is parameterized by a structured variational quantum circuit, whose parameter count scales as $\mathrm { p o l y } ( n )$ . Here we describe the alternative dense Cholesky parameterization used for small systems. In contrast to the VQC ansatz, it can represent arbitrary states (up to full-rank mixed states) without structural restriction, at exp(n) cost.

(a)  
![](images/d173ff3ef6f5d3d69f74ade387b47ebedc84eab20d267992e19609d76ffbcaea.jpg)  
(c)

(b)  
![](images/ae61297ff8dc176813573b042639a2449fe350585de62e54123c25bd8c2f84a4.jpg)

![](images/c0f6530f0f35b787a6477ccf7741af4fb3200ef87c470c929033b2c70c51eefb.jpg)  
FIG. 15. Amortized quantum state tomography with the Cholesky parameterization. (a) Heatmap of the median reconstruction infidelity over 2,000 posterior samples, where each cell $( \theta _ { x } , \theta _ { y } )$ represents a specific 4-qubit pure state. (b) Posterior fidelity distribution for a representative state prepared at $( \theta _ { x } , \theta _ { y } ) = ( \pi / 3 , \pi / 3 )$ . (c) Real (top) and imaginary (bottom) components of the reconstructed density matrix at the median fidelity in panel (b).

The Cholesky decomposition writes $\rho = T T ^ { \dagger } / \mathrm { T r } ( T T ^ { \dagger } )$ with an unconstrained complex matrix $\dot { T } \in \mathbb { C } ^ { 2 ^ { n } \times r }$ of rank $r ,$ ensuring the inferred state is automatically Hermitian, positive semidefinite, and trace-normalized [133]. Here we restrict the model to n-qubit pure states $( r \ = \ 1 )$ for which T reduces to a single complex vector $\mathbf { z } \in \mathbb { C } ^ { 2 ^ { \dot { n } } }$ yielding $\rho = \mathbf { z } \mathbf { z } ^ { \dagger } / \mathrm { T r } ( \mathbf { z } \mathbf { z } ^ { \dagger } ) = | \psi \rangle \langle \psi |$

For a four-qubit pure state, the latent vector $\pmb \theta \in \mathbb { R } ^ { 2 \cdot 2 ^ { n } }$ $\left( = \mathbb { R } ^ { 3 2 } \right)$ comprises the real and imaginary components of z, providing a real parameter space for inference. The prior is chosen as a standard normal, $\theta _ { i } \sim \mathcal { N } ( 0 , 1 )$ , which after normalization induces the Haar measure over the pure-state manifold—an unstructured choice in contrast to the structured prior of the main text. The observation $\mathbf { x } \in \mathbb { R } ^ { 1 8 0 }$ collects expectation values over 180 randomly selected Pauli observables out of the $4 ^ { n } - 1 = 2 5 5$ basis elements. The simulator $s$ evaluates exact expectation values via $\operatorname { T r } [ \rho ( \pmb \theta ) P ]$ and adds binomial shot noise with $N _ { \mathrm { s h o t s } } = 1 0 ^ { 4 }$

We evaluate the amortized estimator, trained on $1 0 ^ { 6 }$ simulations, across a family of states parameterized by two rotation angles $( \theta _ { x } , \theta _ { y } ) , | \psi ( \theta _ { x } , \theta _ { y } ) \rangle =$ $U _ { \mathrm { B W } } \Big ( \bigotimes _ { q = 1 } ^ { 4 } R _ { x } ( \theta _ { x } ) R _ { y } ( \theta _ { y } ) \Big ) | 0 \rangle ^ { \otimes 4 }$ . Figure 15(a) maps the median reconstruction infidelity $1 - F _ { \mathrm { m e d i a n } }$ across the $( \theta _ { x } , \theta _ { y } )$ landscape. The majority of cells achieve $1 - F _ { \mathrm { m e d i a n } } < 0 . 0 4$ (blue region), demonstrating highfidelity reconstruction over a broad range of states. Elevated infidelity appears at cells where $\theta _ { x }$ or $\theta _ { y }$ approach the periodic boundaries $( 0 \mathrm { o r } \pm \pi )$ . Periodically equivalent rotation settings can produce the same prepared state, and at these cells state reconstruction therefore becomes ambiguous.

Figure 15(b) examines a representative state at $( \theta _ { x } , \theta _ { y } ) = ( \pi / 3 , \pi / 3 )$ in detail. The posterior fidelity distribution concentrates near unity, with $F _ { \mathrm { m e d i a n } } = 0 . 9 9 0 1$ ， $F _ { \mathrm { m e a n } } = 0 . 9 8 9 7$ , and $F _ { \mathrm { m a x } } = 0 . 9 9 6 6$ . Figure $\mathrm { 1 5 ( c ) }$ visualizes the real and imaginary components of the density matrix reconstructed from the median-fidelity sample, showing close element-wise agreement with the ideal target. These fidelities make the amortized estimator directly applicable to benchmarking workflows requiring routine small-qubit tomography.

## Appendix J: Model and training details

## 1. Neural posterior estimation

The NPE hyperparameters are summarized in Table I; settings are shared across tasks except for the batch size, training budget, and observation-vector preprocessing (standardization and embedding). All flows are trained with the Adam optimizer, using PyTorch defaults aside from the learning rate. For the Rydberg task, a two-layer MLP $\mathrm { ( d i m _ { o b s } \mathrm { \to } \bar { 5 } 1 2 \mathrm { \to } 2 5 6 }$ , SiLU activations) embeds the observation vector before conditioning the flow. Listed training budgets are the maxima per task; smaller budgets appear in the scaling and sweep analyses.

<table><tr><td colspan="4">Pauli noise QST Rydberg SBI-DT</td></tr><tr><td>Flow transforms Hidden features</td><td colspan="3">3 2048</td></tr><tr><td>Spline bins</td><td colspan="3">16 per dimension</td></tr><tr><td>Learning rate</td><td colspan="3"> $1 0 ^ { - 4 }$ </td></tr><tr><td>Early stopping Standardize x</td><td></td><td>100 epochs without improvement</td><td></td></tr><tr><td>Embedding x</td><td>yes</td><td>yes yes</td><td>no</td></tr><tr><td></td><td>no</td><td>no yes</td><td>no</td></tr><tr><td>Batch size</td><td>2048</td><td>25,600 4096</td><td>500</td></tr><tr><td>Training budget Figure</td><td>30,000 Fig. 3</td><td>900,000 100,000 Fig. 6 Fig. 7</td><td>7,000 Fig. 14</td></tr></table>

TABLE I. Neural spline flow architecture and training hyperparameters. Upper block: settings shared across all models; lower block: model-specific.

## 2. Digital-twin denoising

The denoising network (used in Fig. 5) employs an MLP with two hidden layers (each of width 756), trained

[1] M. Cramer, M. B. Plenio, S. T. Flammia, R. Somma, D. Gross, S. D. Bartlett, O. Landon-Cardinal, D. Poulin, and Y.-K. Liu, Eficient quantum state tomography, Nature Communications 1, 149 (2010).

[2] G. Torlai, G. Mazzola, J. Carrasquilla, M. Troyer, R. Melko, and G. Carleo, Neural-network quantum state tomography, Nature Physics 14, 447 (2018).

[3] S. Ahmed, C. S´anchez Mu˜noz, F. Nori, and A. F. Kockum, Quantum state tomography with conditional generative adversarial networks, Physical Review Letters 127, 140502 (2021).

[4] A. Hashim, L. B. Nguyen, N. Goss, B. Marinelli, R. K. Naik, T. Chistolini, J. Hines, J. Marceaux, Y. Kim, P. Gokhale, T. Tomesh, S. Chen, L. Jiang, S. Ferracin, K. Rudinger, T. Proctor, K. C. Young, I. Siddiqi, and R. Blume-Kohout, Practical introduction to benchmarking and characterization of quantum computers, PRX Quantum 6, 030202 (2025).

[5] V. Gebhart, R. Santagati, A. A. Gentile, E. M. Gauger, D. Craig, N. Ares, L. Banchi, F. Marquardt, L. Pezze, and C. Bonato, Learning quantum systems, Nature Reviews Physics 5, 141 (2023).

[6] T. Proctor, K. Young, A. D. Baczewski, and R. Blume-Kohout, Benchmarking quantum computers, Nature Reviews Physics 7, 105 (2025).

[7] E. Nielsen, J. K. Gamble, K. Rudinger, T. Scholten, K. Young, and R. Blume-Kohout, Gate set tomography, Quantum 5, 557 (2021).

[8] N. Wiebe, C. Granade, C. Ferrie, and D. G. Cory, Hamiltonian learning and certification using quantum resources, Physical Review Letters 112, 190501 (2014).

[9] J. Wang, S. Paesani, R. Santagati, S. Knauer, A. A. Gentile, N. Wiebe, M. Petruzzella, J. L. O’Brien, J. G. Rarity, A. Laing, et al., Experimental quantum Hamiltonian learning, Nature Physics 13, 551 (2017).

[10] A. A. Gentile, B. Flynn, S. Knauer, N. Wiebe, S. Paesani, C. E. Granade, J. G. Rarity, R. Santagati, and A. Laing, Learning models of quantum systems from experiments, Nature Physics 17, 837 (2021).

[11] O. Simard, A. Dawid, J. Tindall, M. Ferrero, A. M. Sengupta, and A. Georges, Learning interactions between Rydberg atoms, PRX Quantum 6, 030324 (2025).

[12] E. Chertkov and B. K. Clark, Computational inverse method for constructing spaces of quantum models from wave functions, Physical Review X 8, 031029 (2018).

[13] K. Inui and Y. Motome, Inverse Hamiltonian design of highly entangled quantum systems, Physical Review Research 6, 033080 (2024).

[14] C. Kokail, P. E. Dolgirev, R. van Bijnen, D. Gonzalez-Cuadra, M. D. Lukin, and P. Zoller, Inverse quantum simulation for quantum material design (2026), arXiv:2601.12239.

[15] C. Ferrie and C. E. Granade, Likelihood-free methods for quantum parameter estimation, Physical Review Letters

for 200 epochs using the Adam optimizer with a learning rate of $1 \bar { 0 } ^ { - 3 }$ and a batch size of 256, on a total of 7,000 training samples.

All models were trained on a single NVIDIA A100 GPU with 40 GB of memory.

112, 130402 (2014).

[16] C. Catana, T. Kypraios, and M. Gut¸˘a, Maximum likelihood versus likelihood-free quantum system identification in the atom maser, Journal of Physics A: Mathematical and Theoretical 47, 415302 (2014).

[17] L. A. Clark and J. Ko lody´nski, Eficient inference of quantum system parameters by approximate Bayesian computation, Physical Review Applied 23, 044040 (2025).

[18] K. Cranmer, J. Brehmer, and G. Louppe, The frontier of simulation-based inference, Proceedings of the National Academy of Sciences 117, 30055 (2020).

[19] M. Deistler, J. Boelts, P. Steinbach, G. Moss, T. Moreau, M. Gloeckler, P. L. C. Rodrigues, J. Linhart, J. K. Lappalainen, B. K. Miller, P. J. Gon¸calves, J.-M. Lueckmann, C. Schr¨oder, and J. H. Macke, Simulation-based inference: A practical guide (2025), arXiv:2508.12939.

[20] M. Dax, S. R. Green, J. Gair, N. Gupte, M. P¨urrer, V. Raymond, J. Wildberger, J. H. Macke, A. Buonanno, and B. Sch¨olkopf, Real-time inference for binary neutron star mergers using machine learning, Nature 639, 49 (2025).

[21] L. Dingeldein, P. Cossio, and R. Covino, Simulationbased inference of single-molecule experiments, Current Opinion in Structural Biology 91, 102988 (2025).

[22] The ATLAS Collaboration, An implementation of neural simulation-based inference for parameter estimation in ATLAS, Reports on Progress in Physics 88, 067801 (2025).

[23] J. Zhang, Y. Zhang, B. Yi, Y. Ren, Q. Jiao, H. Bai, W. Jiang, and Z. Song, Discovery learning predicts battery cycle life from minimal experiments, Nature 650, 110 (2026).

[24] L. Dingeldein, D. Silva-S´anchez, L. Evans, E. D’Imprima, N. Grigorief, R. Covino, and P. Cossio, Amortized template matching of molecular conformations from cryoelectron microscopy images using simulation-based inference, Proceedings of the National Academy of Sciences 122, e2420158122 (2025).

[25] S. Bond-Taylor, A. Leach, Y. Long, and C. G. Willcocks, Deep generative modelling: A comparative review of VAEs, GANs, normalizing flows, energy-based and autoregressive models, IEEE Transactions on Pattern Analysis and Machine Intelligence 44, 7327 (2022).

[26] I. Kobyzev, S. J. Prince, and M. A. Brubaker, Normalizing flows: An introduction and review of current methods, IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 3964 (2021).

[27] J.-M. Lueckmann, P. J. Goncalves, G. Bassetto, K. Ocal,<sup>¨</sup> M. Nonnenmacher, and J. H. Macke, Flexible statistical inference for mechanistic models of neural dynamics, in Advances in Neural Information Processing Systems, Vol. 30 (2017).

[28] G. Papamakarios and I. Murray, Fast ε-free inference of simulation models with Bayesian conditional density es-

timation, in Advances in Neural Information Processing Systems, Vol. 29 (2016).

[29] D. Greenberg, M. Nonnenmacher, and J. Macke, Automatic posterior transformation for likelihood-free inference, in International Conference on Machine Learning (2019) pp. 2404–2414.

[30] J. Wildberger, M. Dax, S. Buchholz, S. R. Green, J. H. Macke, and B. Sch¨olkopf, Flow matching for scalable simulation-based inference, in Advances in Neural Information Processing Systems, Vol. 36 (2023).

[31] A. Zammit-Mangion, M. Sainsbury-Dale, and R. Huser, Neural methods for amortized inference, Annual Review of Statistics and Its Application 12, 311 (2025).

[32] T. Proctor, M. Revelle, E. Nielsen, K. Rudinger, D. Lobser, P. Maunz, R. Blume-Kohout, and K. Young, Detecting and tracking drift in quantum information processors, Nature Communications 11, 5396 (2020).

[33] Y. Kim, L. C. Govia, A. Dane, E. van den Berg, D. M. Zajac, B. Mitchell, Y. Liu, K. Balakrishnan, G. Keefe, A. Stabile, et al., Error mitigation with stabilized noise in superconducting quantum processors, Nature Communications 16, 8439 (2025).

[34] D. Zhu, Z.-P. Cian, C. Noel, A. Risinger, D. Biswas, L. Egan, Y. Zhu, A. M. Green, C. H. Alderete, N. H. Nguyen, et al., Cross-platform comparison of arbitrary quantum states, Nature Communications 13, 6620 (2022).

[35] Y. Zhou, E. M. Stoudenmire, and X. Waintal, What limits the simulation of quantum computers?, Physical Review X 10, 041038 (2020).

[36] J. Tindall, M. Fishman, E. M. Stoudenmire, and D. Sels, Eficient tensor network simulation of IBM’s Eagle kicked Ising experiment, PRX Quantum 5, 010308 (2024).

[37] T. Beguˇsi´c, J. Gray, and G. K.-L. Chan, Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance, Science Advances 10, eadk4321 (2024).

[38] Y. Shao, F. Wei, S. Cheng, and Z. Liu, Simulating noisy variational quantum algorithms: A polynomial approach, Physical Review Letters 133, 120603 (2024).

[39] D. Aharonov, X. Gao, Z. Landau, Y. Liu, and U. Vazirani, A polynomial-time classical algorithm for noisy random circuit sampling, in 55th Annual ACM Symposium on Theory of Computing (2023).

[40] T. Schuster, C. Yin, X. Gao, and N. Y. Yao, A polynomial-time classical algorithm for noisy quantum circuits, Physical Review X 15, 041018 (2025).

[41] M. S. Rudolph, T. Jones, Y. Teng, A. Angrisani, and Z. Holmes, Pauli Propagation: A computational framework for simulating quantum systems, PRX Quantum 7, 032001 (2026).

[42] E. R. Anschuetz, A. Bauer, B. T. Kiani, and S. Lloyd, Eficient classical algorithms for simulating symmetric quantum systems, Quantum 7, 1189 (2023).

[43] G. Camillo, F. C. Peres, M. Heinrich, and J. Bermejo-Vega, Symmetry-accelerated classical simulation of Cliford-dominated circuits, PRX Quantum 7, 020356 (2026).

[44] S. Y. Chang, M. Larocca, and M. Cerezo, Practical framework for simulating permutation-equivariant quantum circuits (2026), arXiv:2603.13072.

[45] S. Aaronson and D. Gottesman, Improved simulation of stabilizer circuits, Physical Review A 70, 052328 (2004).

[46] R. Jozsa and A. Miyake, Matchgates and classical simulation of quantum circuits, Proceedings: Mathematical, Physical and Engineering Sciences 464, 3089 (2008).

[47] M. L. Goh, M. Larocca, L. Cincio, M. Cerezo, and F. Sauvage, Lie-algebraic classical simulations for quantum computing, Physical Review Research 7, 033266 (2025).

[48] G. Vidal, Eficient classical simulation of slightly entangled quantum computations, Physical Review Letters 91, 147902 (2003).

[49] N. Dowling, Classical simulability from operator entanglement scaling (2026), arXiv:2603.05656.

[50] T. Schuster and N. Y. Yao, Operator growth in open quantum systems, Physical Review Letters 131, 160402 (2023).

[51] I. L. Markov and Y. Shi, Simulating quantum computation by contracting tensor networks, SIAM Journal on Computing 38, 963 (2008).

[52] G. S. Hartnett, K. S. Najafi, A. Khindanov, H. Liao, M. Schutzman, M. R. Hush, M. J. Biercuk, and Y. Baum, Fast, accurate, high-resolution simulation of large-scale Fermi-Hubbard models on a digital quantum processor (2026), arXiv:2605.04025.

[53] A. Deger, S. Koutsioumpas, M. Webster, H. Sayginel, J. Rofe, and D. E. Browne, Eficiently simulable quantum circuits with large entanglement, magic, and non-Gaussianity via code-compiled tensor networks (2026), arXiv:2607.08396.

[54] G. Q. AI and Collaborators, Quantum error correction below the surface code threshold, Nature 638, 920 (2025).

[55] D. C. McKay, I. Hincks, E. J. Pritchett, M. Carroll, L. C. G. Govia, and S. T. Merkel, Benchmarking quantum processor performance at scale (2023), arXiv:2311.05933.

[56] J. Wurtz, A. Bylinskii, B. Braverman, J. Amato-Grill, S. H. Cantu, F. Huber, A. Lukin, F. Liu, P. Weinberg, J. Long, S.-T. Wang, N. Gemelke, and A. Keesling, Aquila: QuEra’s 256-qubit neutral-atom quantum computer (2023), arXiv:2306.11727.

[57] H. Bayraktar, A. Charara, D. Clark, S. Cohen, T. Costa, Y.-L. L. Fang, Y. Gao, J. Guan, J. Gunnels, A. Haidar, et al., cuQuantum SDK: A high-performance library for accelerating quantum science, in IEEE International Conference on Quantum Computing and Engineering, Vol. 1 (2023) pp. 1050–1061.

[58] A. Cicero, M. A. Maleki, M. W. Azhar, A. F. Kockum, and P. Trancoso, Simulation of quantum computers: Review and acceleration opportunities, ACM Transactions on Quantum Computing 7 (2025).

[59] A. Morningstar, M. Hauru, J. Beall, M. Ganahl, A. G. Lewis, V. Khemani, and G. Vidal, Simulation of quantum many-body dynamics with tensor processing units: Floquet prethermalization, PRX Quantum 3, 020331 (2022).

[60] S. Chen, Y. Liu, M. Otten, A. Seif, B. Feferman, and L. Jiang, The learnability of Pauli noise, Nature Communications 14, 52 (2023).

[61] E. H. Chen, S. Chen, L. E. Fischer, A. Eddins, L. C. G. Govia, B. Mitchell, A. He, Y. Kim, L. Jiang, and A. Seif, Disambiguating Pauli noise in quantum computers, PRX Quantum 7, 033045 (2026).

[62] E. Van Den Berg, Z. K. Minev, A. Kandala, and K. Temme, Probabilistic error cancellation with sparse

Pauli–Lindblad models on noisy quantum processors, Nature Physics 19, 1116 (2023).

[63] K. Temme, S. Bravyi, and J. M. Gambetta, Error mitigation for short-depth quantum circuits, Physical Review Letters 119, 180509 (2017).

[64] Y. Li and S. C. Benjamin, Eficient variational quantum simulator incorporating active error minimization, Physical Review X 7, 021050 (2017).

[65] Y. Kim, A. Eddins, S. Anand, K. X. Wei, E. Van Den Berg, S. Rosenblatt, H. Nayfeh, Y. Wu, M. Zaletel, K. Temme, et al., Evidence for the utility of quantum computing before fault tolerance, Nature 618, 500 (2023).

[66] H. Liao, D. S. Wang, I. Sitdikov, C. Salcedo, A. Seif, and Z. K. Minev, Machine learning for practical quantum error mitigation, Nature Machine Intelligence 6, 1478 (2024).

[67] P. Czarnik, A. Arrasmith, P. J. Coles, and L. Cincio, Error mitigation with Cliford quantum-circuit data, Quantum 5, 592 (2021).

[68] Y. Liu, D. Wang, S. Xue, A. Huang, X. Fu, X. Qiang, P. Xu, H.-L. Huang, M. Deng, C. Guo, X. Yang, and J. Wu, Variational quantum circuits for quantum state tomography, Physical Review A 101, 052316 (2020).

[69] F. Belliardo, E. M. Gauger, M. H. Abobeih, T. H. Taminiau, Y. Altmann, and C. Bonato, Multidimensional quantum estimation and model learning framework based on variational Bayesian inference, PRX Quantum 7, 020360 (2026).

[70] G. Moss, L. Muhle, R. Drews, J. H. Macke, and C. Schr¨oder, FNOPE: Simulation-based inference on function spaces with Fourier Neural Operators, in Advances in Neural Information Processing Systems, Vol. 38 (2026).

[71] G. White, F. Pollock, L. Hollenberg, K. Modi, and C. Hill, Non-Markovian quantum process tomography, PRX Quantum 3, 020344 (2022).

[72] F. A. Pollock, C. Rodr´ıguez-Rosario, T. Frauenheim, M. Paternostro, and K. Modi, Non-Markovian quantum processes: Complete framework and eficient characterization, Physical Review A 97, 012127 (2018).

[73] J. Keeling, E. M. Stoudenmire, M.-C. Ba˜nuls, and D. R. Reichman, Process tensor approaches to non-Markovian quantum dynamics, Physical Review X 16, 020502 (2026).

[74] V. Cimini, I. Gianani, N. Spagnolo, F. Leccese, F. Sciarrino, and M. Barbieri, Calibration of quantum sensors by neural networks, Physical Review Letters 123, 230502 (2019).

[75] V. Cimini, E. Polino, M. Valeri, I. Gianani, N. Spagnolo, G. Corrielli, A. Crespi, R. Osellame, M. Barbieri, and F. Sciarrino, Calibration of multiparameter sensors via machine learning at the single-photon level, Physical Review Applied 15, 044003 (2021).

[76] A. Veps¨al¨ainen, R. Winik, A. H. Karamlou, J. Braum¨uller, A. D. Paolo, Y. Sung, B. Kannan, M. Kjaergaard, D. K. Kim, A. J. Melville, et al., Improving qubit coherence using closed-loop feedback, Nature Communications 13, 1932 (2022).

[77] M. D. Shulman, S. P. Harvey, J. M. Nichol, S. D. Bartlett, A. C. Doherty, V. Umansky, and A. Yacoby, Suppressing qubit dephasing using real-time Hamiltonian estimation, Nature Communications 5, 5156 (2014).

[78] L. Dinh, J. Sohl-Dickstein, and S. Bengio, Density estimation using Real NVP, in International Conference on Learning Representations (2017).

[79] D. J. Rezende and S. Mohamed, Variational inference with normalizing flows, in International Conference on Machine Learning, Vol. 37 (2015) p. 1530–1538.

[80] G. Papamakarios, T. Pavlakou, and I. Murray, Masked autoregressive flow for density estimation, in Advances in Neural Information Processing Systems, Vol. 30 (2017).

[81] C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios, Neural spline flows, in Advances in Neural Information Processing Systems, Vol. 32 (2019).

[82] F. No´e, S. Olsson, J. K¨ohler, and H. Wu, Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning, Science 365, eaaw1147 (2019).

[83] S. Asghar, Q.-X. Pei, G. Volpe, and R. Ni, Eficient rare event sampling with unsupervised normalizing flows, Nature Machine Intelligence 6, 1370 (2024).

[84] M. S. Albergo, G. Kanwar, and P. E. Shanahan, Flowbased generative models for Markov chain Monte Carlo in lattice field theory, Physical Review D 100, 034515 (2019).

[85] H. Zou, M. Rahm, A. F. Kockum, and S. Olsson, Generative flow-based warm start of the variational quantum eigensolver, npj Quantum Information 12, 5 (2026).

[86] J. J. Wallman and J. Emerson, Noise tailoring for scalable quantum computation via randomized compiling, Physical Review A 94, 052325 (2016).

[87] Z. Cai, R. Babbush, S. C. Benjamin, S. Endo, W. J. Huggins, Y. Li, J. R. McClean, and T. E. O’Brien, Quantum error mitigation, Reviews of Modern Physics 95, 045005 (2023).

[88] K. Tsubouchi, T. Sagawa, and N. Yoshioka, Universal cost bound of quantum error mitigation based on quantum estimation theory, Physical Review Letters 131, 210601 (2023).

[89] R. Takagi, H. Tajima, and M. Gu, Universal sampling lower bounds for quantum error mitigation, Physical Review Letters 131, 210602 (2023).

[90] P. Lolur, M. Skogh, W. Dobrautz, C. Warren, J. Bizn´arov´a, A. Osman, G. Tancredi, G. Wendin, J. Bylander, and M. Rahm, Reference-state error mitigation: A strategy for high accuracy quantum computation of chemistry, Journal of Chemical Theory and Computation 19, 783 (2023).

[91] H. Zou, E. Magnusson, H. Brunander, W. Dobrautz, and M. Rahm, Multireference error mitigation for quantum computation of chemistry, Digital Discovery 4, 2521 (2025).

[92] A. Angrisani, A. Schmidhuber, M. S. Rudolph, M. Cerezo, Z. Holmes, and H.-Y. Huang, Classically estimating observables of noiseless quantum circuits, Physical Review Letters 135, 170602 (2025).

[93] U. Schollw¨ock, The density-matrix renormalization group, Reviews of Modern Physics 77, 259 (2005).

[94] M. K. Kurmapu, V. Tiunova, E. Tiunov, M. Ringbauer, C. Maier, R. Blatt, T. Monz, A. K. Fedorov, and A. Lvovsky, Reconstructing complex states of a 20-qubit quantum simulator, PRX Quantum 4, 040345 (2023).

[95] G. Carleo and M. Troyer, Solving the quantum manybody problem with artificial neural networks, Science 355, 602 (2017).

[96] H. Lange, A. Van de Walle, A. Abedinnia, and A. Bohrdt, From architectures to applications: A review of neural quantum states, Quantum Science and Technology 9, 040501 (2024).

[97] F. J. Schreiber, J. Eisert, and J. J. Meyer, Tomography of parametrized quantum states, PRX Quantum 6, 020346 (2025).

[98] A. Gaikwad, M. S. Torres, S. Ahmed, and A. F. Kockum, Gradient-descent methods for fast quantum state tomography, Quantum Science and Technology 10, 045055 (2025).

[99] D. Bluvstein, S. J. Evered, A. A. Geim, S. H. Li, H. Zhou, T. Manovitz, S. Ebadi, M. Cain, M. Kalinowski, D. Hangleiter, et al., Logical quantum processor based on reconfigurable atom arrays, Nature 626, 58 (2024).

[100] D. Bluvstein, A. A. Geim, S. H. Li, S. J. Evered, J. P. Bonilla Ataides, G. Baranes, A. Gu, T. Manovitz, M. Xu, M. Kalinowski, et al., A fault-tolerant neutral-atom architecture for universal quantum computation, Nature 649, 39 (2026).

[101] R. Lin, H.-S. Zhong, Y. Li, Z.-R. Zhao, L.-T. Zheng, T.-R. Hu, H.-M. Wu, Z. Wu, W.-J. Ma, Y. Gao, Y.-K. Zhu, Z.-F. Su, W.-L. Ouyang, Y.-C. Zhang, J. Rui, M.- C. Chen, C.-Y. Lu, and J.-W. Pan, AI-enabled parallel assembly of thousands of defect-free neutral atom arrays, Physical Review Letters 135, 060602 (2025).

[102] A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu, Theory of Trotter error with commutator scaling, Physical Review X 11, 011020 (2021).

[103] M. Seubert, L. Hartung, S. Welte, G. Rempe, and E. Distante, Tweezer-assisted subwavelength positioning of atomic arrays in an optical cavity, PRX Quantum 6, 010322 (2025).

[104] X. Huan, J. Jagalur, and Y. Marzouk, Optimal experimental design: Formulations and computations, Acta Numerica 33, 715 (2024).

[105] S. Chen, Z. Zhang, L. Jiang, and S. T. Flammia, Eficient self-consistent learning of gate set Pauli noise, PRX Quantum 7, 010305 (2026).

[106] L. Govia, S. Majumder, S. Barron, B. Mitchell, A. Seif, Y. Kim, C. Wood, E. Pritchett, S. Merkel, and D. McKay, Bounding the systematic error in quantum error mitigation due to model violation, PRX Quantum 6, 010354 (2025).

[107] A. Wehenkel, J. L. Gamella, O. Sener, J. Behrmann, G. Sapiro, J.-H. Jacobsen, and M. Cuturi, Addressing misspecification in simulation-based inference through data-driven calibration, in International Conference on Machine Learning (2025).

[108] D. J. Nott, C. Drovandi, and D. T. Frazier, Bayesian inference for misspecified generative models, Annual Review of Statistics and Its Application 11 (2023).

[109] N. Anau Montel, J. Alvey, and C. Weniger, Tests for model misspecification in simulation-based inference: From local distortions to global model checks, Physical Review D 111, 083013 (2025).

[110] A. Patel, A. Gaikwad, T. Huang, A. F. Kockum, and T. Abad, Selective and eficient quantum state tomography for multiqubit systems, Physical Review Research 8, 013339 (2026).

[111] J. Cotler and F. Wilczek, Quantum overlapping tomography, Physical Review Letters 124, 100401 (2020).

[112] T. Peng, A. W. Harrow, M. Ozols, and X. Wu, Simulating large quantum circuits on a small quantum computer, Physical Review Letters 125, 150504 (2020).

[113] J. Liu, A. Gonzales, and Z. H. Saleem, Classical simulators as quantum error mitigators via circuit cutting (2022), arXiv:2212.07335.

[114] A. Elben, S. T. Flammia, H.-Y. Huang, R. Kueng, J. Preskill, B. Vermersch, and P. Zoller, The randomized measurement toolbox, Nature Reviews Physics 5, 9 (2023).

[115] A. Sanchez-Gonzalez, J. Godwin, T. Pfaf, R. Ying, J. Leskovec, and P. Battaglia, Learning to simulate complex physics with graph networks, in International Conference on Machine Learning (2020) pp. 8459–8468.

[116] J. V. Diez, M. Schreiner, and S. Olsson, Transferable generative models bridge femtosecond to nanosecond time-step molecular dynamics, Science Advances 12, eaed2333 (2026).

[117] H. Wang, X. Wu, J. Liu, R. He, J. Shang, H. Guo, and Q. Chen, Scalable quantum error mitigation with physically informed graph neural networks (2026), arXiv:2604.16815.

[118] T. Cohen, M. Weiler, B. Kicanaoglu, and M. Welling, Gauge equivariant convolutional networks and the icosahedral CNN, in International Conference on Machine Learning (2019).

[119] J. Robledo-Moreno, M. Motta, H. Haas, A. Javadi-Abhari, P. Jurcevic, W. Kirby, S. Martiel, K. Sharma, S. Sharma, T. Shirakawa, et al., Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer, Science Advances 11, eadu9991 (2025).

[120] S. Seelam, J. M. Chow, A. C´orcoles, S. Sheldon, T. Mittal, A. Kandala, S. Dague, I. Hincks, H. Horii, B. Johnson, M. Le, H. Jamjoom, and J. M. Gambetta, Reference architecture of a quantum-centric supercomputer (2026), arXiv:2603.10970.

[121] A. Katabarwa, K. Gratsea, A. Caesura, and P. D. Johnson, Early fault-tolerant quantum computing, PRX Quantum 5, 020101 (2024).

[122] N. Wiebe, C. Granade, C. Ferrie, and D. Cory, Quantum Hamiltonian learning using imperfect quantum resources, Physical Review A 89, 042314 (2014).

[123] M. Benedetti, B. Coyle, M. Fiorentini, M. Lubasch, and M. Rosenkranz, Variational inference with a quantum computer, Physical Review Applied 16, 044057 (2021).

[124] B. Coyle, D. Mills, V. Danos, and E. Kashefi, The Born supremacy: quantum advantage and training of an Ising Born machine, npj Quantum Information 6, 60 (2020).

[125] F. Rozet et al., Zuko: Normalizing flows in PyTorch (2022).

[126] J. Boelts, M. Deistler, M. Gloeckler, Alvaro Tejero-<sup>´</sup> Cantero, J.-M. Lueckmann, G. Moss, P. Steinbach, T. Moreau, F. Muratore, J. Linhart, C. Durkan, J. Vetter, B. K. Miller, M. Herold, A. Ziaeemehr, M. Pals, T. Gruner, S. Bischof, N. Krouglova, R. Gao, J. K. Lappalainen, B. Mucs´anyi, F. Pei, A. Schulz, Z. Stefanidi, P. Rodrigues, C. Schr¨oder, F. A. Zaid, J. Beck, J. Kapoor, D. S. Greenberg, P. J. Gon¸calves, and J. H. Macke, sbi reloaded: a toolkit for simulation-based inference workflows, Journal of Open Source Software 10, 7754 (2025).

[127] C. Gidney, Stim: a fast stabilizer circuit simulator, Quantum 5, 497 (2021).

[128] A. Javadi-Abhari, M. Treinish, K. Krsulich, C. J. Wood, J. Lishman, J. Gacon, S. Martiel, P. D. Nation, L. S. Bishop, A. W. Cross, B. R. Johnson, and J. M. Gambetta, Quantum computing with Qiskit (2024), arXiv:2405.08810.

[129] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, PyTorch: An imperative style, highperformance deep learning library, in Advances in Neural Information Processing Systems, Vol. 32 (2019).

[130] M. Germain, K. Gregor, I. Murray, and H. Larochelle, MADE: masked autoencoder for distribution estima-

tion, in International Conference on Machine Learning, Vol. 37 (2015) p. 881–889.

[131] P. Rall, D. Liang, J. Cook, and W. Kretschmer, Simulation of qubit quantum circuits via Pauli propagation, Physical Review A 99, 062337 (2019).

[132] T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, and W. J. Zeng, Digital zero noise extrapolation for quantum error mitigation, in IEEE International Conference on Quantum Computing and Engineering (2020) pp. 306– 316.

[133] D. F. V. James, P. G. Kwiat, W. J. Munro, and A. G. White, Measurement of qubits, Physical Review A 64, 052312 (2001).