# Transition Path Sampling Using Koopman Operators and Exit-Time Optimal Control

Boya Hou<sup>∗</sup> Shane Wang<sup>∗</sup> Siddharth Ambekar Maxim Raginsky

Olgica Milenkovic

University of Illinois Urbana-Champaign

October 8, 2026

## Abstract

Sampling transitions between metastable states is a central problem in dynamical systems theory and molecular dynamics in particular. A key challenge is the existence of high free-energy barriers that separate the states, making transitions extremely rare. Recent machine learningbased methods cast transition path sampling (TPS) as an optimal stochastic control (OSC) problem over a fixed time horizon, and parameterize the drift bias via a neural network trained by simulation-in-the-loop, requiring repeated biased rollouts. To address computational and performance guarantee issues of these models, we propose a new approach for the problem based on Koopman operators. Because Koopman operators are linear, their leading eigenfunctions reveal the metastable sets and provide an estimate of the committor function with no transition path information required. Furthermore, we formulate TPS as an OSC problem up to an exit time. Our time horizon is the first hitting time of the target set, and our running cost penalizes time spent in nonreactive regions by encoding the estimated committor function. We derive the optimal controller in closed form and approximate it in a reproducing kernel Hilbert space (RKHS). This reduces the problem of constructing the optimal controller to solving a single equality-constrained quadratic program, whose solution can be characterized by a linear Karush-Kuhn-Tucker (KKT) system. On the two-channel double well and alanine dipeptide, our controller increases the fraction of trajectories reaching the target from 0% to 99.8% within 1,000 steps, and from 0% to 93% within 1ps, respectively.

## 1 Introduction

Sampling transitions between metastable sets is a fundamental problem in dynamical systems theory and molecular dynamics, underlying the study of conformational changes of proteins, folding, and drug discovery (Grubmüller, 1995; Voter, 1997). A typical challenge is that the states of interest are separated by high free-energy barriers: the overwhelming majority of simulated trajectories remain within a single metastable set, while trajectories that cross from one metastable set to another without returning (reactive paths) are rare under native dynamics (Vanden-Eijnden et al., 2006).

Classical transition path sampling (TPS) performs Markov chain Monte Carlo in the path space, generating new trajectories from existing ones (Dellago et al., 1998; Bolhuis et al., 2002; Bolhuis and Swenson, 2021). However, they require an initial reactive path connecting the two metastable sets, which can be dificult to generate. In physical chemistry and molecular dynamics, rare event sampling is typically accelerated by adding bias potentials to the energy landscape (Voter, 1997; Grubmüller, 1995; Laio and Parrinello, 2002). These are usually defined over a small number of hand-crafted collective variables. Modern machine learning-based samplers such as PIPS (Holdijk et al., 2023) and TPS-DPS (Seong et al., 2025) cast transition path sampling as optimal stochastic control or path-measure matching problems over a fixed time horizon, and parameterize the drift bias with a neural network trained by simulation-in-the-loop, which calls for repeated biased rollouts. Learning a bias force through simulation, however, can be challenging. For example, early in the training, few rollouts reach the target, and hence the training signal is highly sparse. Furthermore, a fixed-horizon formulation requires a properly chosen horizon, and the transition time is a random variable dependent on the path itself. Partial mitigation of these drawbacks is possible using replay bufers and annealing, and truncating the paths at a time that maximizes specific kernel values.

To resolve the above issues, we propose an operator-based framework for eficient TPS that uses Koopman spectral analysis and optimal stochastic control (OSC) up to an exit time, in which the stopping rule is the first hitting time of the target set. Specifically, we cast TPS as sampling from a Gibbs measure on the path space problem, following the information-theoretic frameworks of Mitter and Newton (2003) and Raginsky (2026). By the Gibbs variational principle, the Gibbs measure associated with a reference path measure and an energy on a path is, under certain conditions, the unique minimizer of the free energy. Mitter and Newton (2003) used this insight to pose nonlinear filtering as an optimal control problem. To adapt the described sampling method to TPS, we take the reference to be the unbiased path measure of the original dynamics, and the energy to penalize the time spent in nonreactive regions via the committor function. Since we assume the time horizon to be the first time a path reaches the target set, we operate with exit rather than fixed times. We prove that the resulting Gibbs measure is the law of a controlled difusion process and prove that the optimal controller minimizes the free energy over all path measures on the stopped σ-algebra rather than only those generated by admissible controls. Furthermore, the optimal controller admits a closed-form solution and can be computed without simulations in the loop.

It is worth pointing out that the proposed approach difers from the recent work of Du et al. (2026), which investigates a related hitting-time problem for committor estimation, treating the committor as a value function and training it with neural networks that require repeated simulation of controlled dynamics during training. As previously emphasized, we instead follow an operatortheoretic approach in which the spectrum of the Koopman generator of the model reveals the metastable sets, and where the committor function is constructed from Koopman eigenfunctions. These can be estimated from a fixed dataset without requiring any trajectory to transition between the metastable sets. The Koopman generator plays a central role within this framework: its leading eigenfunctions reveal the metastable sets and are incorporated into the committor function, which in turn defines the running cost of the OSC. The optimal controller of the OSC can be solved via a boundary value problem wrt the operator. This pipeline comes with several practical advantages. First, it requires no transition paths, as Koopman eigenfunctions can be computed without crossings between metastable sets via a generalized eigenproblem. Second, the controller admits a closed-form solution, and thus is obtained without network training but through solutions of convex quadratic programs. The approach is also interpretable, as the controller is built from physically meaningful quantities that characterize the mechanism of the transition, namely the metastable sets and the committor function. Furthermore, the sampled paths can be reweighted to recover the statistics of the original dynamics.

Our contributions are multifold: (1) We formulate TPS as an OSC up to an exit time. We show in Theorem 1 that the optimal controller admits a closed-form solution, $\boldsymbol { u } ^ { * } = \nabla \log \rho ,$ and that its path measure minimizes the free energy over all path measures, not only those generated by admissible controls; (2) We construct this controller by approximating $\rho$ in a Reproducing Kernel Hilbert Space (RKHS). The problem then reduces to solving a convex quadratic program with linear equality constraints, whose solution can be characterized by a single linear Karush–Kuhn–Tucker (KKT) system rather than repeated rollouts; (3) In TPS, the source and target metastable sets are often predefined through collective variables. Following Koopman-based approaches that identify metastable sets (Mardt et al., 2018; Pérez-Hernández et al., 2013), we construct these sets without collective variables by using the leading eigenfunctions of the Koopman generator, which capture the slow inter-basin modes. From the same eigenfunctions, we construct a committor function, which defines the running cost of the OSC; (4) We generate transition paths by simulating the controlled difusion process until it first enters the target sets, and show the resulting paths can be reweighted via Lemma 4 to recover the statistics of the original dynamics up to the transition; (5) We evaluate our method on both synthetic and real-world datasets.

The paper is organized as follows. In Section 2, we introduce the difusion process of interest and the corresponding Koopman generator. In Section 3, we develop the OSC up to an exit time in general form. There, we also derive a closed form of the optimal controller. The results of this section can be specialized for eficient TPS. Section 4 constructs metastable sets and the committor function from Koopman eigenfunctions, and approximates the optimal controller in an RKHS. We generate transition paths by simulating the optimally controlled SDE, and report numerical experiments in Section 5. Due to space limitations, proofs and extra experiments are deferred to the Appendix G.4.

## 2 Preliminaries: Itô Difusion and the Koopman Generator

Let $\mathbb { X } \subseteq \mathbb { R } ^ { n }$ denote the state space and consider the Itô difusion process

$$
d X _ { t } = b ( X _ { t } ) d t + \sigma ( X _ { t } ) d W _ { t } , \qquad X _ { 0 } = x ,\tag{1}
$$

where $W _ { t }$ is an m-dimensional Brownian motion process, and $b : \mathbb { X } \to \mathbb { R } ^ { n }$ and $\sigma : \mathbb { X } \to \mathbb { R } ^ { n \times m }$ are the drift and difusion functions, respectively Karatzas and Shreve (2014); Oksendal (2000). A model of the form (1) used in MD simulations is the Langevin dynamics, where b and σ are specified by a potential energy U, friction $\gamma ,$ and bath temperature $T _ { \mathrm { e m p } }$ . In the overdamped case, the state is the atomic configuration, and the drift and difusion coeficients are specialized as $\begin{array} { r } { b ( x ) = - \frac { 1 } { \gamma m } \nabla U ( x ) } \end{array}$ $\sigma ( x ) = \sigma _ { 0 } I $ , with $\begin{array} { r } { \sigma _ { 0 } = \sqrt { \frac { 2 k _ { B } T _ { \mathrm { e m p } } } { \gamma m } } } \end{array}$ . In the underdamped case, the state additionally carries the momenta, and friction and noise act only on the momentum component. In either case, the process is metastable whenever $U$ has multiple deep minima (wells) separated by barriers large relative to $k _ { B } T _ { \mathrm { e m p } }$ , and a trajectory spends most of its time in a single well.

Next, let $\mathbb { T } : = [ 0 , \infty )$ and fix the space $\Omega = C ( \mathbb { T } ; \mathbb { X } )$ of continuous paths mapping T to X. Also, let $X _ { t } ( \omega ) = \omega ( t )$ denote the evaluation of a path $\omega \in \Omega$ at time $t \in \mathbb { T }$ . Let $\mathcal { F } = \sigma ( X _ { t } : t \in \mathbb { T } )$ be the canonical σ-algebra, and let $( \mathcal { F } _ { t } ) _ { t \in \mathbb { T } }$ with $\mathcal { F } _ { t } = \sigma ( X _ { s } : s \le t )$ be the filtration, both generated by the process $( X _ { t } ) _ { t \in \mathbb { T } }$ . For $x \in \mathbb { X }$ , let $P ^ { x }$ be the law on (Ω, F) of the solution of (1) started at $x ,$ provided it is well posed, for example, under Assumptions 1 and 2 in Section 3. Under $P ^ { x }$ , the canonical process $X _ { t } ( \omega ) = \omega ( t )$ solves (1). Define the family of Koopman operators $\{ \mathcal { K } _ { t } \} _ { t \in \mathbb { T } }$ as

$$
\begin{array} { r } { \mathcal { K } _ { t } \varphi ( x ) = \mathbb { E } [ \varphi ( X _ { t } ) \vert X _ { 0 } = x ] , } \end{array}\tag{2}
$$

per (Koopman and von Neumann, 1932; Lasota and Mackey, 2013). Although (1) is nonlinear, $\boldsymbol { \mathcal { { K } } } _ { t }$ for $t \in \mathbb { T }$ are linear operators and $\{ \boldsymbol { \mathcal { K } } _ { t } \} _ { t \in \mathbb { T } }$ defines a semigroup ${ \ K } _ { t + s } = { \ K } _ { t } \circ { \ K } _ { s }$ . The associated

infinitesimal Koopman generator L is a linear second-order diferential operator given by

$$
L \varphi ( x ) = b ( x ) \cdot \nabla \varphi ( x ) + { \frac { 1 } { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } a _ { i j } ( x ) \partial _ { i j } \varphi ( x ) ,\tag{3}
$$

for $\varphi \in C ^ { 2 } ( \mathbb { X } )$ , where the indices $i , j = 1 , \cdots , n$ range over the coordinates of $x = ( x _ { 1 } , \ldots , x _ { n } ) \in \mathbb { X }$ $\partial _ { i j }$ denotes the second partial derivative with respect to $x _ { i }$ and $x _ { j } , \ a _ { i j }$ is the $( i , j )$ entry of the difusion matrix $a ( x ) : = \sigma ( x ) \sigma ( x ) ^ { \top }$ , and b is the drift term in (1). The spectrum of L can be utilized to characterize the global structure of the dynamics (Mezić, 2005; Mardt et al., 2018; Kostic et al., 2022). Let $( \lambda , \psi )$ be an eigenpair of L, i.e., $L \psi = \lambda \psi$ . Then $\mathcal { K } _ { t } \psi = e ^ { \lambda t } \psi , \mathrm { i . e . , } \mathbb { E } _ { x } [ \psi ( X _ { t } ) ] = e ^ { \lambda t } \psi ( x )$ The constant function $\psi _ { 0 } \equiv 1$ is always an eigenfunction with $\lambda _ { 0 } = 0$

The Koopman operator and its spectrum can be approximately computed from data via numerical methods such as dynamic mode decomposition (DMD) (Schmid, 2010; Rowley et al., 2009) and extended dynamic mode decomposition (EDMD) (Williams et al., 2015), as well as nonparametric variants based on RKHS known as kernel EDMD (Williams et al., 2014; Kostic et al., 2022; Klus et al., 2020; Hou et al., 2023). We provide a brief introduction to RKHS in Appendix A. Following Kostic et al. (2022); Hou et al. (2023), we approximate the leading eigenpairs $( \lambda _ { k } , \psi _ { k } )$ of L from samples $\{ x _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { X }$ . Since b and a are known, L can be applied directly to the kernel via (3). This yields a generalized eigenproblem in terms of Gram matrices; see, e.g., (Hou et al., 2023, Appendix M) for details.

A trajectory of (1) that crosses from one metastable set to another without returning to its origin is a (reactive) transition path (Vanden-Eijnden et al., 2006). For two disjoint sets $A _ { i }$ and $A _ { j }$ let $\tau _ { A _ { k } }$ be the first time the system enters $A _ { k }$ for $k = i , j$ . The committor function $q ^ { ( i , j ) } ( x )$ for the transition from $x \in A _ { i }$ to $A _ { j }$ is defined as $q ( x ) = P { \bigl ( } \tau _ { A _ { i } } < \tau _ { A _ { i } } \mid x _ { 0 } = x { \bigr ) }$ (Vanden-Eijnden et al., 2006; Metzner et al., 2009; Hofmann et al., 2021). Along a transient path, $\boldsymbol { q } ^ { ( i , j ) } ( \boldsymbol { x } _ { t } )$ is close to 0 near $A _ { i }$ and close to 1 near $A _ { j }$ , whereas a path that lingers in $A _ { i }$ has $q ^ { ( i , j ) } ( \boldsymbol { x } _ { t } ) \approx 0$ most of the time. When the metastable sets are clear from context, we write $q$ as a shorthand for $\boldsymbol { q } ^ { ( i , j ) }$ . Next, let $D _ { q } : = \mathbb { X } \setminus ( A _ { i } \cup A _ { j } )$ . The committor function q satisfies the boundary value problem

$$
L q = 0 \mathrm { i n } D _ { q } , q | _ { \partial A _ { i } } = 0 , q | _ { \partial A _ { j } } = 1 ,\tag{4}
$$

where L is the Koopman generator, and we extended q by $q \equiv 0$ on $A _ { i }$ and $q \equiv 1$ on $A _ { j }$ . We defer the proof of this identity to the Appendix C.

The above concepts lead to our pipeline for TPS: the leading eigenfunctions of the Koopman generator L identify the metastable sets and provide the spectral term of the committor function; the committor function then defines the running cost of the OSC presented in the next section, whose optimal controller can be approximated via a boundary value problem in L.

## 3 Transition Path Sampling via OSC up to an Exit Time

## 3.1 The Sampling Problem

Consider the difusion process governed by the Itô SDE (1). To facilitate our analysis of the outlined sampling approach, we make the assumption:

Assumption 1. $b ( x )$ and $\sigma ( x )$ are bounded and Lipschitz on ${ \mathbb X }$

Under Assumption 1, the SDE (1) has a unique strong solution. Let $P ^ { x }$ be the path measure started at x. Let $D \subset \mathbb { X }$ be open, and $\tau _ { D } : = \operatorname* { i n f } \{ t \geq 0 : X _ { t } \notin D \}$ be the exit time. Since D is open, $\tau _ { D }$ is the hitting time of the closed set $\mathbb { X } \backslash D$ and is therefore an $( \mathcal { F } _ { t } )$ -stopping time. We observe paths until they reach the target set (equivalently, exit the set $D )$ , and we work on the stopped σ-algebra $\mathcal { F } _ { \tau _ { D } }$ defined by ${ \mathcal { F } } _ { \tau _ { D } } : = \{ A \in { \mathcal { F } } : ~ A \cap \{ \tau _ { D } \leq t \} \in { \mathcal { F } } _ { t }$ for all $t \geq 0 \}$ . Then $X _ { \tau _ { D } }$ is $\mathcal { F } _ { \tau _ { D } }$ -measurable.

For $x \in \bar { D }$ and each path $\chi$ under $P ^ { x }$ , define the Hamiltonian as

$$
H ( \chi ) = \int _ { 0 } ^ { \tau _ { D } } f ( X _ { t } ) d t + \Phi ( X _ { \tau _ { D } } ) ,\tag{5}
$$

where $f : \bar { D } \to [ 0 , \infty )$ and $\Phi : \partial D \to [ 0 , \infty )$ are measurable, and $H ( \chi ) : = + \infty \mathrm { o n } \left\{ \tau _ { D } = \infty \right\}$ . Here, $f \geq 0$ is the running cost and $\Phi$ is the terminal cost. Since paths are continuous and $D$ is open, $X _ { \tau _ { D } } \in \partial D$ on $\{ \tau _ { D } < \infty \}$ . Hence, H is a well-defined $\mathcal { F } _ { \tau _ { D } }$ -measurable function.

For the purpose of eficient sampling, among all path measures on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ , we seek the one that minimizes the free energy

$$
\begin{array} { r } { F ( \tilde { P } ) : = \tilde { \mathbb { E } } \big [ H ( \chi ) \big ] + D _ { \mathrm { K L } } \big ( \tilde { P } \| P ^ { x } \big ) , \qquad \tilde { P } \in \mathcal { P } ( \Omega , \mathcal { F } _ { \tau _ { D } } ) , } \end{array}\tag{6}
$$

where $\tilde { \mathbb { E } }$ denotes the expectation w.r.t. ${ \tilde { P } } _ { ; }$ , and $\mathcal { P } ( \Omega , \mathcal { F } _ { \tau _ { D } } )$ denotes the set of all probability measures on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ . Here, the first term favors paths with low energy, while the second keeps $\tilde { P }$ close to the original dynamics. We next make use of the Gibbs variational principle on $\mathcal { F } _ { \tau _ { D } }$ , which is adapted from (Mitter and Newton, 2003, Proposition 2.1).

Proposition 1. Let H be an $\mathcal { F } _ { \tau _ { D } }$ -measurable function, and suppose $Z : = \mathbb { E } _ { x } { \bigl [ } e ^ { - H } { \bigr ] } > 0$ . Then the probability measure $P ^ { * , x }$ on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ defined by

$$
\left. { \frac { \mathrm { d } P ^ { * , x } } { \mathrm { d } P ^ { x } } } \right| _ { \mathcal { F } _ { T _ { D } } } = { \frac { e ^ { - H ( \chi ) } } { Z } }\tag{7}
$$

is the unique minimizer of the free energy (6) over the set of all probability measures on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ ， and min $\tilde { { P } } ^ { \tilde { F } ( \tilde { P } ) } = F ( P ^ { * , x } ) = - \log Z$

It will be useful to keep in mind the following motivating example.

Example 1 (TPS). Let $A _ { i } , A _ { j } \subset \mathbb { X }$ be disjoint closed source and target sets, and let $D : = \mathbb { X } \setminus A _ { j }$ Then $A _ { i } \subset D$ , and thus a trajectory initialized within the source set starts in the interior of $D .$ . For such a trajectory, the exit time of D is the first hitting time of $A _ { j }$ , so $\tau _ { D }$ is the transition time of interest. Let $q ( x ) : = P ^ { x } ( \tau _ { A _ { j } } < \tau _ { A _ { i } } )$ be the committor function. Define the running and terminal cost

$$
f _ { \beta } ( x ) : = \beta { \big ( } 1 - q ^ { ( i , j ) } ( x ) { \big ) } , \quad \Phi \equiv 0 { \mathrm { ~ o n ~ } } \partial A _ { j } ,\tag{8}
$$

with $\beta > 0$ . Then, $H = \beta H _ { 0 }$ , where $\begin{array} { r } { H _ { 0 } = \int _ { 0 } ^ { \tau _ { D } } ( 1 - q ^ { ( i , j ) } ( X _ { t } ) ) } \end{array}$ dt is the time spent in nonreactive regions before reaching the target. Here, the parameter $\beta$ controls the trade-of between speed and fidelity. As $\beta$ increases, we increasingly prefer faster paths at the cost of deviating more from the original dynamics. In Appendix F we show how to undo this dynamics distortion via reweighting.

The sampling problem of interest is to generate paths from $P ^ { * , x }$ . Proposition 1 identifies the target measure $P ^ { * , x }$ but does not provide a way to sample from it. We therefore phrase the sampling problem as a control problem where we add a control drift to the reference process (1) so that the law of the controlled difusion equals $P ^ { * , x }$ on $\mathcal { F } _ { \tau _ { D } }$ . We call a control $u : \bar { D } \to \mathbb { R } ^ { n }$ admissible if u is

Lipschitz continuous on $\bar { D }$ and use $\mathcal { U }$ to denote the set of all admissible controls. A $u \in \mathcal { U }$ enters the drift through $a ( x ) u ( x )$ , resulting in the controlled process

$$
d \tilde { X } _ { t } = \left[ b ( \tilde { X } _ { t } ) + a ( \tilde { X } _ { t } ) u ( \tilde { X } _ { t } ) \right] d t + \sigma ( \tilde { X } _ { t } ) d \tilde { W } _ { t } ,\tag{9}
$$

where $\tilde { \tau } _ { D } : = \operatorname* { i n f } \{ t \geq 0 : \tilde { X } _ { t } \notin D \}$ . As we show in Theorem 1 (ii), restricting the focus to a Lipschitz continuous controller does not weaken the result, since we can show that the path measure of the optimal admissible controller is optimal over all probability measures on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$

As the time-horizon is the first time the process exits $D ,$ the exit time $\tau _ { D }$ is itself a random variable. We first establish that it has a finite mean. We make the following Assumptions

Assumption 2. There exists $c > 0$ such that $z ^ { \top } a ( x ) z \geq c | z | ^ { 2 }$ for all $x \in \mathbb { X } , z \in \mathbb { R } ^ { n }$

Assumption 3. $D \subset \mathbb { X }$ is open and bounded with $C ^ { 2 , \alpha }$ boundary ∂D for some $\alpha \in ( 0 , 1 )$

These assumptions are standard for exit-time problems for controlled difusions (Fleming, 1977). They ensure a unique solution of the associated elliptic boundary value problem in L. They hold, for example, for overdamped Langevin dynamics with a smooth potential, including the double-well, four-well, and Müller–Brown systems studied in Section 5 and Appendix G.4. The proof of the following lemma is presented in Appendix B.1:

Lemma 1. Let Assumptions 1–3 hold. Then, the SDE (1) starting from $x \in D$ has a pathwise unique strong solution, and its exit time $\tau _ { D }$ satisfies $\mathbb { E } _ { x } [ \tau _ { D } ] < \infty$ . Consequently, $\tau _ { D } < \infty \ a . s .$

## 3.2 A Candidate Optimal Controller

We now seek a controlled difusion of the form (9) whose path measure equals $P ^ { * , x }$ . For this purpose, we also require a regularity condition,

Assumption 4. The running cost $f \in C ^ { 0 , \alpha } ( \bar { D } )$ with $f \geq 0$ , and $\Phi : \partial D  [ 0 , \infty )$ such that $g : = e ^ { - \Phi } \in C ^ { 2 , \alpha } ( \partial D )$ .

Define

$$
\rho ( x ) : = \mathbb { E } _ { x } \left[ \exp \left( - \int _ { 0 } ^ { \tau _ { D } } f ( X _ { t } ) d t - \Phi ( X _ { \tau _ { D } } ) \right) \right] .\tag{10}
$$

Here, we integrate time from 0 to an exit time $\tau _ { D }$ . Under Assumption $3 , f$ is bounded on $\bar { D } .$ , and Φ is bounded on $\partial D$ since $g$ is continuous and positive on the compact set ∂D. By Lemma 1, $\tau _ { D } < \infty$ a.s. under $P ^ { x }$ , hence (10) is well-defined, and $\rho ( x ) \in ( 0 , 1 ]$

We next show that $\rho$ solves a boundary value problem via elliptic existence theory (Ladyzhenskaya and Uraltseva, 1968, Ch. III). We consider the following elliptic boundary value problem, where L is the (Koopman) generator:

$$
L w - f w = 0 \mathrm { i n } D , \qquad w | _ { \partial D } = e ^ { - \Phi } .\tag{11}
$$

Since $\partial D \in C ^ { 2 , \alpha }$ by Assumption 2 and $g = e ^ { - \Phi } \in C ^ { 2 , \alpha } ( \partial D )$ by Assumption 3, applying (Ladyzhenskaya and Uraltseva, 1968, Ch. III, Theorem 1.1, 1.2) gives a unique solution $w \in C ^ { 2 , \alpha } ( \hat { D } )$ to the problem (11). The proof of the following lemma is included in the Appendix B.3.

Lemma 2. Let Assumptions $1 \mathrm { - } \mathit { 4 }$ hold and let $w \in C ^ { 2 , \alpha } ( \hat { D } )$ be the unique solution of the problem (11). Then, w coincides on D<sup>¯</sup> with ρ defined in (10). Moreover, $\rho > 0$ on D<sup>¯</sup>.

Our sampling/free energy minimization problem can now be cast as an optimal control problem. In the fixed-horizon setting, the relative entropy of the controlled law with respect to the reference law equals the expected control efort by Girsanov’s theorem, so the free energy (6) of the controlled law is an optimal control cost (Raginsky, 2026, eq. (4.5)). By contrast, here we have to consider the optimal stochastic control problem up to an exit time,

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { u \in \mathcal { U } } J ( u , x ) : = \tilde { \mathbb { E } } _ { x } \left[ \int _ { 0 } ^ { \tilde { \tau } _ { D } } \Bigl ( f ( \tilde { X } _ { t } ) + \frac { 1 } { 2 } u ( \tilde { X } _ { t } ) ^ { \top } a ( \tilde { X } _ { t } ) u ( \tilde { X } _ { t } ) \Bigr ) d t + \Phi ( \tilde { X } _ { \tilde { \tau } _ { D } } ) \right] , } \\ { \mathrm { s u b j e c t ~ t o ~ ( 9 ) , ~ } \tilde { X } _ { 0 } = x \in D . } \end{array}\tag{12}
$$

We denote its value function as $v ( x ) : = \operatorname* { i n f } _ { u \in \mathcal { U } } J ( u , x )$ , for $x \in \hat { D }$ , with $v | _ { \partial D } = \Phi$ , since $\tilde { \tau } _ { D } = 0$ when $\tilde { X } _ { 0 } \in \partial D$ . Consider the candidate value function $v : = - \log \rho$ (this is the logarithmic, or Cole–Hopf, transform of $\rho ,$ (Fleming, 1977)). Since D<sup>¯</sup> is compact by Assumption 2 and $\rho \in C ( \hat { D } )$ with $\rho > 0$ on $\bar { D }$ as established in Lemma 2, the extreme value theorem gives $m : = \mathrm { m i n } _ { \bar { D } } \rho > 0$ and $M : = \operatorname* { m a x } _ { \bar { D } } \rho < \infty$ , so $\rho ( { \bar { D } } ) \subseteq [ m , M ] \subset ( 0 , \infty )$ . Since $\rho \in C ^ { 2 , \alpha } ( \bar { D } )$ , and − log is smooth on $[ m , M ]$ , the chain rule gives $v = -$ log $\circ \rho \in C ^ { 2 , \alpha } ( \bar { D } )$ , and hence ∇v is bounded on D<sup>¯</sup>. Additionally, since $\rho \big | _ { \partial D } = g = e ^ { - \Phi }$ , the boundary condition of v is $v | _ { \partial D } = \Phi$ . We have the following lemma whose proof is presented in Appendix B.4.

Lemma 3. The function v satisfies the Hamilton-Jacobi-Bellman (HJB) equation of the OSC problem (12) up to an exit time, i.e.,

$$
\begin{array} { r } { L v ( x ) + f ( x ) = \frac { 1 } { 2 } \nabla v ( x ) ^ { \top } a ( x ) \nabla v ( x ) \quad i n D , \qquad v \big | _ { \partial D } = \Phi . } \end{array}\tag{13}
$$

We hence have a candidate for the optimal controller of (12) given by

$$
u ^ { * } ( x ) = - \nabla v ( x ) = \nabla \log \rho ( x ) = \nabla \rho ( x ) / \rho ( x ) .\tag{14}
$$

Theorem 1 below establishes that its minimal value equals min $F = - \log Z$ and that its optimally controlled law is $P ^ { * , x }$

## 3.3 Verification Theorem of OSC up to an Exit Time

Let ${ \tilde { P } } ^ { * , x }$ be the law of the controlled process (9) under $u ^ { * }$ , starting at x. In the remainder of the section, we establish that $u ^ { * }$ in (14) is the optimal controller. Moreover, using a variational characterization of the optimally controlled path measure, we show that ${ \tilde { P } } ^ { * , x }$ minimizes the free energy (6) over all path measures on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ . We begin by presenting an explicit formula for the Radon-Nikodym density of ${ \tilde { P } } ^ { * , x }$ relative to $P ^ { x }$ on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ . To that end, we invoke Girsanov’s formula. In the classic fixed time-horizon case, one needs a suitable regularity condition, such as Novikov’s condition (Oksendal, 2000, Theorem 8.6.3). By contrast, in the exit time setting, such a condition might not hold as it requires an exponential moment bound, which is hard to verify for metastable systems. To address this challenge, the proof presented in Appendix B.6 only requires $\tau _ { D } < \infty \ \mathrm { a . s }$ . guaranteed by Lemma 1.

Lemma 4. Let $\boldsymbol { u } ^ { * } = - \nabla \boldsymbol { v }$ be as in (14), and let ${ \tilde { X } } _ { t }$ be a solution to the controlled SDE (9) with $u = u ^ { * }$ , initiated at $\tilde { X } _ { 0 } = x$ , and under a probability measure ${ \tilde { P } } ^ { * , x }$ on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ . Then, we have

$$
\left. \frac { d \tilde { P } ^ { * , x } } { d P ^ { x } } \right| _ { \mathcal { F } _ { \tau _ { D } } } = \frac { e ^ { - H ( \chi ) } } { \rho ( x ) } , \qquad \mathbb { E } _ { x } \left[ \frac { e ^ { - H ( \chi ) } } { \rho ( x ) } \right] = 1 .\tag{15}
$$

We now present the verification theorem the proof of which can be found in Appendix B.7.

Theorem 1. Let Assumptions $1 \mathrm { - } \mathit { 4 }$ hold, and let $u ^ { * }$ be the controller in (14). Then,

(i) [Optimality] $\begin{array} { r } { v ( x ) = J ( u ^ { * } , x ) = \operatorname* { m i n } _ { u } J ( u , x ) } \end{array}$ for all $x \in D$

(ii) [Variational inequality] For every probability measure $\tilde { P }$ on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ ,

$$
J ( u ^ { \ast } , x ) \leq D _ { \mathrm { K L } } ( \tilde { P } \| P ^ { x } ) + \tilde { \mathbb { E } } _ { x } [ H ( \chi ) ] ,\tag{16}
$$

with equality if and only if $d \tilde { P } / d P ^ { x } = e ^ { - H ( \chi ) } / \rho ( x )$ on $\mathcal { F } _ { \tau _ { D } }$ ; equivalently, if and only if $\tilde { P } =$ ${ \tilde { P } } ^ { * , x }$ , corresponding to the law of the optimally-controlled process restricted to $\mathcal { F } _ { \tau _ { D } }$

Part (i) of the above theorem states that the minimal control cost is min $u \in \mathcal { U } ^ { J } ( u , x ) = v ( x ) =$ − log $\rho ( x )$ , and by Part (ii), that the free energy (6) satisfies $F ( \tilde { P } ) \geq v ( x )$ for every probability measure on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ , with equality precisely for $\tilde { P } = \tilde { P } ^ { * , x }$ . Hence, the two minimal values of (6) and (12) coincide, and the optimal path measure that minimizes the free energy is exactly the one generated by the optimal controller. In the proof of Part (i) of Theorem 1, we also show that the minimizer is unique in the following sense: $J ( u , x ) = v ( x )$ for some $u \in \mathcal { U }$ if and only if $u ( { \tilde { X } } _ { t } ) = u ^ { * } ( { \tilde { X } } _ { t } )$ for Lebesgue-a.e. $t \in [ 0 , \tilde { \tau } _ { D } ] , \tilde { P } ^ { u , x _ { - \mathrm { a . s } } }$

By the definition in $( 1 0 ) , \rho ( x )$ is large where the trajectory of the original dynamics (1) tends to exit quickly and small where it dwells in nonreactive regions. The optimal controller in (14) therefore points toward states from which a trajectory is more likely to exit. The normalization by $\rho > 0$ in (14) provides a stronger push towards where $\rho$ is small and the original dynamics is least likely to escape; thus the controller acts more aggressively when deep inside the source basin.

Using Lemma 4, we can recover statistics of the original dynamics (1) from the accelerated path alone. We present a reweighting mechanism via Girsanov’s change of measure in Appendix F.

## 4 Koopman-Based Construction of the Optimal Controller

Theorem 1 reduces the problem of sampling from $P ^ { * , x }$ to computing the optimal controller $u ^ { * } =$ ∇ log $\rho ,$ where $\rho$ solves the elliptic problem (11). For TPS in Example 1, applying this result requires three ingredients: the source and target sets $A _ { i } , A _ { j }$ , the committor function q which defines the running cost $f _ { \beta } ,$ , and $\rho$ itself. Each can be characterized through the Koopman generator $L$ Specifically, the metastable sets can be computed through the leading eigenfunctions of $L , q$ through $L q = 0$ in (4), and $\rho$ through $L \rho - f _ { \beta } \rho = 0$ in (11).

## 4.1 Constructing Metastable Cores and Committor Functions

The spectrum of the linear operator L reveals global geometric structures of the underlying dynamics (Mezić, 2005; Mauroy and Mezić, 2016). In molecular dynamics, for example, the metastable sets can be estimated using leading eigenfunctions whose eigenvalues are close to zero, followed by a spectral gap (Mardt et al., 2018; Hou et al., 2023, 2026). Specifically, the leading eigenvalues $0 = \lambda _ { 0 } > \lambda _ { 1 } \geq \cdot \cdot \cdot \geq \lambda _ { m - 1 } \gg \lambda _ { m } \cdot \cdot \cdot$ · also capture the slow-mode, inter-basin dynamics. By truncating at m modes, fast modes are removed as they encode intra-basin dynamics. The leading eigenfunctions $\{ \psi _ { k } \} _ { k = 1 } ^ { m - 1 }$ are approximately constant within each metastable set but take diferent values across them; we omit $k = 0 \mathrm { ~ s i n c e ~ } \psi _ { 0 } \equiv 1$ . Let $\Psi ( x ) = [ \psi _ { 1 } ( x ) , \ldots , \psi _ { m - 1 } ( x ) ] \in \mathbb { R } ^ { m - 1 }$ , where $\{ x _ { n } \} _ { n = 1 } ^ { N } \subset \mathbb { X }$ are data samples and the eigenpairs are estimated from them via kernel EDMD in Section 2. We then apply k-means clustering to $\{ \Psi ( x _ { n } ) \} _ { n = 1 } ^ { N }$ , which partitions the samples into m clusters with index sets $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { m }$ . For each cluster k and the set of leading eigenfunctions $\psi _ { l }$ , for $l = 1 , \ldots , m - 1$ , we define the core level of the l-th eigenfunction as

$$
\boldsymbol { \theta } _ { l } ^ { ( k ) } : = \frac { 1 } { | \mathcal { T } _ { k } | } \sum _ { n \in \mathcal { T } _ { k } } \boldsymbol { \psi } _ { l } ( \boldsymbol { x } _ { n } ) , \qquad \boldsymbol { \theta } ^ { ( k ) } : = \big [ \boldsymbol { \theta } _ { 1 } ^ { ( k ) } , \ldots , \boldsymbol { \theta } _ { m - 1 } ^ { ( k ) } \big ] ^ { \top } ,\tag{17}
$$

so that $\theta ^ { ( k ) }$ is the k-means centroid of cluster $k .$ . Let $\varepsilon _ { k } > 0$ be a radius. Since each $\psi _ { l }$ is approximately constant on a metastable set with value approximately $\theta _ { l } ^ { ( k ) }$ on the k-th one, we define the core of the k-th metastable set for $k = 1 , \cdots$ , m as

$$
A _ { k } : = \{ x \in \mathbb { X } : \| \Psi ( x ) - \theta ^ { ( k ) } \| ^ { 2 } \leq \varepsilon _ { k } ^ { 2 } \} .\tag{18}
$$

Leveraging the global geometric structure carried by the leading eigenfunctions of the Koopman generator, we next approximate the committor function $q$ via an ansatz $q = \tilde { \psi } + h$ , where $\tilde { \psi }$ is a spectral term constructed from leading eigenfunctions and $h \in \mathcal { H } _ { \kappa }$ is a remainder with H ${ \mathcal { H } } _ { \kappa }$ is an RKHS equipped with kernel $\kappa ,$ since in general, the solution of (4) may not lie in span $\{ \psi _ { l } \} _ { l = 1 } ^ { m - 1 }$ The spectral term $\tilde { \psi }$ contributes to the global shape of the transition, and also captures the slow coordinate along which the transition occurs, without requiring a reaction coordinate to be specified in advance. Due to space limitations, detailed construction of $\tilde { \psi }$ that maps the core of $A _ { i }$ to 0 and the core of $A _ { j }$ to 1 is presented in Appendix D. We compute $h \in \mathcal { H } _ { \kappa }$ via $\begin{array} { r } { h ( x ) = \sum _ { l \in \mathcal { T } _ { q } } \alpha _ { l } \kappa ( x , x _ { l } ) } \end{array}$ with $\{ x _ { l } \} _ { l \in \mathbb { Z } _ { q } } \subset D _ { q }$ and $\mathcal { T } _ { q } = \{ 1 , \ldots , N _ { q } \}$ . The coeficients α are obtained from an equality-constrained convex quadratic program that minimizes the residual of $( 4 )$ in $D _ { q } ,$ which is also deferred to in Appendix D. Putting together, we compute the committor function via

$$
\hat { q } ( x ) : = \tilde { \psi } ( x ) + \sum _ { l \in \mathcal { T } _ { q } } \alpha _ { l } \kappa ( x , x _ { l } ) ,\tag{19}
$$

## 4.2 Kernel-Based Approximation of the Optimal Controller

Unlike the fixed-horizon case studied in (Raginsky, 2026), where $u ^ { * }$ depends on time, the exittime optimal controller $u ^ { * }$ in (14) is stationary. To compute $u ^ { * }$ , we focus on approximating $\rho$ defined in (10) so that it solves the elliptic boundary value problem (11). By definition of $\rho ,$ and $f _ { \beta } = \beta ( 1 - q ^ { ( i , j ) } )$ , $\Phi \equiv 0$ in Example 1, we have

$$
\rho ( x ) = \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \right] , \qquad H _ { 0 } = \int _ { 0 } ^ { \tau _ { D } } \left( 1 - q ^ { ( i , j ) } ( X _ { t } ) \right) d t ,\tag{20}
$$

where $H _ { 0 }$ is the time spent in nonreactive regions before reaching the target. We need to handle two regimes, capturing the transition region and regions within the basins. Directly approximating $\rho$ in an RKHS can be challenging, as the behavior of $\rho$ in the two regimes may be quite diferent. Near $\partial A _ { j }$ , the target is reached almost immediately and $\rho \approx 1$ . Within the source basin, however, the transition is a rare event, and $\rho$ is small as the trajectory lingers there. Since $u ^ { * }$ depends on $\rho$ through log $\rho ,$ we seek an ansatz of the form $\rho : = \rho _ { 0 } ( x ) w ( x )$ so that log $\rho =$ log $\rho _ { 0 } $ log $w .$ Here we choose $\rho _ { 0 }$ in closed form to capture how $\rho$ varies across the two regimes, and use a residual w to solve for ${ \mathcal { H } } _ { \kappa }$ . We defer additional details to Appendix E and only mention here that we approximate $\rho$ via a closed-form factor $\hat { \rho }$ as

$$
\hat { \rho } ( x ) : = \rho _ { 0 } ( x ) w ( x ) , \quad \rho _ { 0 } ( x ) : = \hat { q } ( x ) + \big ( 1 - \hat { q } ( x ) \big ) r , \quad w \in \mathcal { H } _ { \kappa } ,\tag{21}
$$

where ${ \mathcal { H } } _ { \kappa }$ is an RKHS with kernel $\kappa , ~ \hat { q }$ as in (19) and $\begin{array} { r } { r = \frac { 1 } { 1 + \beta T _ { 0 } } } \end{array}$ , with $T _ { 0 } > 0$ which encodes the time-scale of the original dynamics; for example, we set $\begin{array} { r } { T _ { 0 } = \frac { 1 } { \left| \lambda _ { 1 } \right| } } \end{array}$ , where $\lambda _ { 1 }$ is the leading nontrivial

Koopman eigenvalue. Let $\{ x _ { i } \} _ { i = 1 } ^ { N } = \{ x _ { i } \} _ { i \in I _ { D } } \cup \{ x _ { i } \} _ { i \in I _ { \partial D } }$ , with interior points $\{ x _ { i } \} _ { i \in I _ { D } } \subset D$ and boundary points $\{ x _ { i } \} _ { i \in I _ { \partial D } } \subset \partial D$ . We approximate $w \in \mathcal { H } _ { \kappa }$ via $\begin{array} { r } { w = \sum _ { i = 1 } ^ { N } c _ { i } \kappa ( x , x _ { i } ) } \end{array}$ (Appendix E). Applying the chain rule to log $\hat { \rho } =$ log ρ<sub>0</sub> + log w, we have

$$
\hat { u } ( x ) = \nabla \log \hat { \rho } ( x ) = \frac { ( 1 - r ) \nabla \hat { q } ( x ) } { \hat { q } ( x ) + \left( 1 - \hat { q } ( x ) \right) r } + \frac { \nabla w ( x ) } { w ( x ) } .\tag{22}
$$

Thus, the problem of solving for $u ^ { * }$ reduces to solving a convex quadratic program with linear equality constraints, whose solution is characterized by a single linear KKT system as in Appendix E. The method is summarized in Algorithm 1.

Algorithm 1 Summary of the Algorithm in Four Stages   
Stage I. Construct leading eigenpairs $( \lambda _ { k } , \psi _ { k } ) _ { k = 1 } ^ { m - 1 }$ of the Koopman generator L and identify   
metastable cores.   
Stage II. Compute the committor function $\hat { q }$ via (19) using the leading Koopman eigenfunctions.   
Stage III. Construct the optimal controller uˆ via (22).   
Stage IV. Sample starting points from the source set $A _ { i }$ and integrate the controlled dynamics (9)   
with uˆ from (22) until the paths hit the target set.

## 5 Numerical Experiments

We now present empirical evaluations of the proposed method on two synthetic systems, namely, the two-channel double-well system (Figure 1) and the Müller-Brown system (Figure 2), and a real-world alanine dipeptide (Figure 3). Due to space limitations, experimental details and one additional example are deferred to Appendix G.4. In all these experiments, a path is a hit at its first entrance to the target, and is truncated at the numerical censor $T _ { \mathrm { m a x } }$ otherwise. We report the target-hit percentage (THP), which is the fraction of paths that hit the target set before $T _ { \mathrm { m a x } } .$ We also remark that alanine dipeptide does not satisfy Assumption 2; nevertheless, our algorithm can be applied beyond the confines of the theoretical analysis, and this experiment serves as a demonstration that it performs well even in such cases.

## References

Alain Berlinet and Christine Thomas-Agnan. Reproducing kernel Hilbert spaces in probability and statistics. Springer Science & Business Media, 2011.

Peter G Bolhuis and David W H Swenson. Transition path sampling as markov chain monte carlo of trajectories: Recent algorithms, software, applications, and future outlook. Advanced Theory and Simulations, 4(4):2000237, 2021.

Peter G Bolhuis, David Chandler, Christoph Dellago, and Phillip L Geissler. Transition path sampling: Throwing ropes over rough mountain passes, in the dark. Annual review of physical chemistry, 53(1):291–318, 2002.

Christoph Dellago, Peter G Bolhuis, and David Chandler. Eficient transition path sampling: $\mathrm { A p \mathrm { - } }$ plication to lennard-jones cluster rearrangements. The Journal of Chemical Physics, 108(22): 9236–9245, 1998.

![](images/718f449d5cc7ac91b996f8fea1649e44134adb605ef88732b0160582ddf81fbc.jpg)  
x1

![](images/0e14aaf4a8916951069dd03a54ac6a75221572fd915cbc3a1972e932ca2d7e0d.jpg)  
x1

![](images/f623778e209ee78cb72e2b70ccf4cf7c56da6c071023a602e1e09f5c31e056b8.jpg)  
x1

Figure 1: Two-channel double well. The potential has two minima at $\left( \pm \sqrt { 5 } / 2 , 0 \right)$ connected by two channels around $( 0 , \pm 1 )$ . Following Algorithm 1, we compute the Koopman eigenfunctions and extract the metastable cores $A _ { \mathrm { l e f t } }$ and $A _ { \mathrm { r i g h t } }$ as described in Section 4.1. Their minima are marked by the green and red stars, and the dashed red curve marks $\partial A _ { \mathrm { r i g h t } }$ . We sample 1000 uncontrolled and 1000 controlled paths, all started at the minimum of $A _ { \mathrm { l e f t } }$ . (Left) Potential energy landscape. (Middle) Uncontrolled paths: $\mathrm { T H P } ( T _ { \mathrm { m a x } } ) = 0 \%$ . (Right) Proposed method: THP $( T _ { \mathrm { m a x } } ) = 9 9 . 8 \%$  
![](images/3d23588c01b01045f6f00652599348b911b791bf135bdef997c91f6a8f135e23.jpg)  
(a) Uncontrolled

![](images/cd65bc93f2fca7275471fe40520a1a2eb1ac062588dbfa23497bf78a4086e436.jpg)  
(b) TPS-DPS

![](images/a5a81505c9682cdd31366f51ac6c52036dc264fb9e9997a1bce1889dbbf85d0c.jpg)  
(c) Proposed method  
Figure 2: Müller-Brown potential, with a deep well at $( - 0 . 5 5 8 , 1 . 4 4 2 )$ (green star), a medium well at (0.623, 0.028) (red circle), and a shallow well at $( - 0 . 0 5 0 , 0 . 4 6 7 )$ . We aim to sample transitions from the deep well to the medium well within $T _ { \mathrm { m a x } } = 1 0 \mathrm { p s } .$ . (a) Uncontrolled paths: $\mathrm { T H P } ( T _ { \mathrm { m a x } } )$ $= 1 . 8 \%$ . (b) TPS-DPS (Seong et al., 2025): $\mathrm { T H P } ( T _ { \mathrm { m a x } } ) = 1 0 0 \%$ , but the paths run over the high energy barrier. (c) Proposed method: $\mathrm { T H P } ( T _ { \mathrm { m a x } } ) = 9 9 . 0 \%$ using $\beta = 1$ , with paths following the low-energy channel through the shallow well rather than crossing the barrier.

Yuanqi Du, Jiajun He, Dinghuai Zhang, Eric Vanden-Eijnden, and Carles Domingo-Enrich. Rare event analysis via stochastic optimal control. arXiv preprint arXiv:2604.13213, 2026.

Wendell H Fleming. Exit probabilities and optimal stochastic control. Applied Mathematics and Optimization, 4(1):329–346, 1977.

Helmut Grubmüller. Predicting slow structural transitions in macromolecular systems: Conformational flooding. Phys. Rev. E, 52:2893–2906, Sep 1995. doi: 10.1103/PhysRevE.52.2893. URL https://link.aps.org/doi/10.1103/PhysRevE.52.2893.

Moritz Hofmann, Martin Konrad Scherer, Tim Hempel, Andreas Mardt, Brian de Silva, Brooke Elena Husic, Stefan Klus, Hao Wu, J Nathan Kutz, Steven Brunton, and Frank Noé. Deeptime: a python library for machine learning dynamical models from time series data. Machine Learning: Science and Technology, 2021.

Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Berend Ensing, and Max Welling. Stochastic

![](images/9e6f9e3bfb63211aefb65d3b2f1a07e2a7070b3a0bf44ad2eca6e5f2ad5632f6.jpg)  
(a) Uncontrolled

![](images/bf97550f48cf5c57ae620f7f329e3c28c15156358e3258be7422422b685caa05.jpg)  
(b) TPS-DPS

![](images/766bbfa71e210ca15cef04cb72ef661ee0424f4460c2c7c0ac2389d37ee3cfbc.jpg)  
(c) Proposed method  
Figure 3: Alanine dipeptide in vacuum, shown on the free-energy surface over the backbone dihedral angles (ϕ, ψ). We aim to sample the transition from C5 conformation (upper left) to the $C 7 _ { \mathrm { a x } }$ conformation (red star) within $T _ { \mathrm { m a x } } = 1 \mathrm { p s }$ . (a) Uncontrolled paths: $( T _ { \mathrm { m a x } } ) = 0 \%$ (b) TPS-DPS: THP $( T _ { \mathrm { m a x } } ) = 1 0 0 \%$ , obtained by training a neural-network bias with simulation in the loop (1000 rollouts, ${ \sim } 1 0 ^ { 6 }$ gradient updates). (c) Proposed method: THP $( T _ { \mathrm { m a x } } ) = 9 3 \%$ , with a controller obtained in closed form. The setup and computational cost are reported in Appendix G.4.

optimal control for collective variable free sampling of molecular transition paths. Advances in Neural Information Processing Systems, 36:79540–79556, 2023.

Boya Hou, Sina Sanjari, Nathan Dahlin, Subhonmesh Bose, and Umesh Vaidya. Sparse learning of dynamical systems in rkhs: An operator-theoretic approach. In International Conference on Machine Learning, pages 13325–13352. PMLR, 2023.

Boya Hou, Sina Sanjari, Alec Koppel, Nathan Dahlin, and Subhonmesh Bose. Nonparametric sparse online learning of the koopman operator. To appear at SIAM Journal on Control and Optimization, 2026.

Ioannis Karatzas and Steven Shreve. Brownian motion and stochastic calculus. springer, 2014.

Stefan Klus, Feliks Nüske, and Boumediene Hamzi. Kernel-based approximation of the koopman generator and schrödinger operator. Entropy, 22(7):722, 2020.

Bernard O Koopman and J von Neumann. Dynamical systems of continuous spectra. Proceedings of the National Academy of Sciences, 18(3):255–263, 1932.

Vladimir Kostic, Pietro Novelli, Andreas Maurer, Carlo Ciliberto, Lorenzo Rosasco, and Massimiliano Pontil. Learning dynamical systems via koopman operator regression in reproducing kernel hilbert spaces. Advances in Neural Information Processing Systems, 35:4017–4031, 2022.

Olga A Ladyzhenskaya and Nina N Uraltseva. Linear and Quasilinear Elliptic Equations. Academic Press, New York, 1968.

Alessandro Laio and Michele Parrinello. Escaping free-energy minima. Proceedings of the National Academy of Sciences, 99(20):12562–12566, 2002. doi: 10.1073/pnas.202427399. URL https: //www.pnas.org/doi/abs/10.1073/pnas.202427399.

Andrzej Lasota and Michael C Mackey. Chaos, fractals, and noise: stochastic aspects of dynamics, volume 97. Springer Science & Business Media, 2013.

Andreas Mardt, Luca Pasquali, Hao Wu, and Frank Noé. Vampnets for deep learning of molecular kinetics. Nature communications, 9(1):5, 2018.

Alexandre Mauroy and Igor Mezić. Global stability analysis using the eigenfunctions of the koopman operator. IEEE Transactions on Automatic Control, 61(11):3356–3369, 2016.

Philipp Metzner, Christof Schütte, and Eric Vanden-Eijnden. Transition path theory for markov jump processes. Multiscale Modeling & Simulation, 7(3):1192–1219, 2009.

Igor Mezić. Spectral properties of dynamical systems, model reduction and decompositions. Nonlinear Dynamics, 41(1):309–325, 2005.

Sanjoy K Mitter and Nigel J Newton. A variational approach to nonlinear estimation. SIAM journal on control and optimization, 42(5):1813–1833, 2003.

Klaus Müller and Leo D. Brown. Location of saddle points and minimum energy paths by a constrained simplex optimization procedure. Theoretica chimica acta, 53(1):75–93, Mar 1979. ISSN 1432-2234. doi: 10.1007/BF00547608. URL https://doi.org/10.1007/BF00547608.

Bernt Oksendal. Stochastic diferential equations: an introduction with applications (Fifth Edition). Springer Science & Business Media, 2000.

Guillermo Pérez-Hernández, Fabian Paul, Toni Giorgino, Gianni De Fabritiis, and Frank Noé. Identification of slow molecular order parameters for markov model construction. The Journal of chemical physics, 139(1), 2013.

Maxim Raginsky. A variational approach to sampling in difusion processes. Communications in Optimization Theory, 2026(30):1–20, 2026. URL https://cot.mathres.org/archives/2289.

Clarence W Rowley, Igor Mezić, Shervin Bagheri, Philipp Schlatter, and Dan S Henningson. Spectral analysis of nonlinear flows. Journal of fluid mechanics, 641:115–127, 2009.

Peter J Schmid. Dynamic mode decomposition of numerical and experimental data. Journal of fluid mechanics, 656:5–28, 2010.

Kiyoung Seong, Seonghyun Park, Seonghwan Kim, Woo Youn Kim, and Sungsoo Ahn. Transition path sampling with improved of-policy training of difusion path samplers. In International Conference on Learning Representations, volume 2025, pages 93040–93062, 2025.

Eric Vanden-Eijnden et al. Towards a theory of transition paths. Journal of statistical physics, 123 (3):503–523, 2006.

Arthur F. Voter. Hyperdynamics: Accelerated molecular dynamics of infrequent events. Phys. Rev. Lett., 78:3908–3911, May 1997. doi: 10.1103/PhysRevLett.78.3908. URL https://link.aps. org/doi/10.1103/PhysRevLett.78.3908.

Matthew O Williams, Clarence W Rowley, and Ioannis G Kevrekidis. A kernel-based approach to data-driven koopman spectral analysis. Journal of Computational Dynamics, 2014.

Matthew O Williams, Ioannis G Kevrekidis, and Clarence W Rowley. A data–driven approximation of the koopman operator: Extending dynamic mode decomposition. Journal of Nonlinear Science, 25(6):1307–1346, 2015.

## Appendix

## A The Reproducing Kernel Hilbert Space

We provide a brief introduction to the real-valued RKHS – for a more comprehensive treatment, see (Berlinet and Thomas-Agnan, 2011). Let X be a subset of a Euclidean space and let $\kappa : \mathbb { X } \times \mathbb { X } \to$ R be a continuous, symmetric, positive semi-definite kernel. Define the RKHS associated with the kernel $\kappa , \varkappa _ { \kappa } ,$ as the completion of the span of $\{ \phi ( x ) : = \kappa ( x , \cdot ) : x \in \mathbb { X } \}$ , equipped with the inner product $\langle \cdot , \cdot \rangle$ , satisfying $\langle \phi ( x ) , \phi ( y ) \rangle = \kappa ( x , y )$ . Here, ϕ is called the feature map for the kernel $\kappa .$ The inner product satisfies the reproducing property, given by $\langle \phi ( x ) , f \rangle = f ( x )$ , for all $x \in \mathbb { X }$ and $f \in \mathcal { H } _ { \kappa }$ . Throughout the exposition, we make the standard assumption that κ is measurable.

## A.1 Auxiliary Result: A Localized Dynkin Formula

Several proofs in this paper follow a similar outline: we start with a real-valued function h that solves a partial diferential equation on $D ,$ , and we aim to evaluate the expectation of h along the trajectory at the exit time $\tau _ { D }$ . However, Itô’s formula is stated for functions that are $C ^ { 2 }$ on $\mathbb { R } ^ { n }$ whereas the functions we work with are $C ^ { 2 }$ only on the set D. Therefore, we need a mechanism to integrate up to $\tau _ { D }$ at which time the trajectory reaches $\partial D$ . In addition, such a $\tau _ { D }$ is a random variable itself. We address this by localizing, following the scheme in the proof of (Oksendal, 2000, Theorem 9.1.1). Specifically, we first integrate to $\sigma _ { k } : = k \wedge \tau _ { D _ { k } }$ , where $D _ { k }$ is an increasing sequence of open sets in $D .$ . Then on $[ 0 , \sigma _ { k } ]$ , the optional stopping theorem (Karatzas and Shreve, 2014, Theorem 1.3.22) applies. Letting $k  \infty$ then takes the result to τ<sub>D</sub>, given $\sigma _ { k }  \tau _ { D }$ . The following lemma (localized Dynkin formula) makes this construction precise.

Lemma 5. Let Assumption $1 \mathrm { - } \mathit { 4 }$ hold. Let Y be the pathwise unique strong solution of

$$
d Y _ { t } = b ( Y _ { t } ) d t + \sigma ( Y _ { t } ) d W _ { t } , \qquad Y _ { 0 } = x \in D ,\tag{23}
$$

with generator $\begin{array} { r } { L _ { Y } h : = b ^ { \top } \nabla h + \frac { 1 } { 2 } \sum _ { i j } a _ { i j } \partial _ { i j } h } \end{array}$ , and let $\tau _ { D } : = \operatorname* { i n f } \{ t > 0 : Y _ { t } \notin D \}$ . Define a sequence of open sets as

$$
D _ { k } : = \{ y \in D : \operatorname { d i s t } ( y , \partial D ) > 1 / k \} .\tag{24}
$$

Let $\sigma _ { k } : = k \wedge \tau _ { D _ { k } }$ , where $\tau _ { D _ { k } }$ is the exit time of Y from $D _ { k }$ . Then, we have

(i) The sequence $( \sigma _ { k } ) _ { k \geq 1 }$ is nondecreasing, and $\sigma _ { k }  \tau _ { D }$ as $k  \infty$

(ii) Let $h \in C ^ { 2 } ( D ) \cap C ( \bar { D } ) , c \in C ( \bar { D } )$ with $c \geq 0$ , and $\begin{array} { r } { Z _ { t } : = \exp \left( - \int _ { 0 } ^ { t } c ( Y _ { s } ) d s \right) } \end{array}$ . Then for every k, we have

$$
h ( Y _ { \sigma _ { k } } ) Z _ { \sigma _ { k } } - h ( x ) = \int _ { 0 } ^ { \sigma _ { k } } Z _ { t } \bigl ( L _ { Y } h - c h \bigr ) ( Y _ { t } ) d t + \int _ { 0 } ^ { \sigma _ { k } } Z _ { t } \nabla h ( Y _ { t } ) ^ { \top } \sigma ( Y _ { t } ) d W _ { t } ,\tag{25}
$$

and

$$
\mathbb { E } \big [ h ( Y _ { \sigma _ { k } } ) Z _ { \sigma _ { k } } \big ] - h ( x ) = \mathbb { E } \Big [ \int _ { 0 } ^ { \sigma _ { k } } Z _ { t } \big ( L _ { Y } h - c h \big ) ( Y _ { t } ) d t \Big ] .\tag{26}
$$

Proof. By construction of $D _ { k }$ in (24), $D _ { k } \subset D _ { k + 1 }$ , and $\textstyle \bigcup _ { k } D _ { k } = D$ . Moreover, $\bar { D } _ { k }$ is a compact subset of D.

Part (i) For every $k \geq 1$ , since $D _ { k } \subset D _ { k + 1 }$ , we have $\{ t > 0 : Y _ { t } \notin D _ { k + 1 } \} \subseteq \{ t > 0 : Y _ { t } \notin D _ { k } \}$ Hence, $\tau _ { D _ { k } } \le \tau _ { D _ { k + 1 } }$ . Since $\sigma _ { k } = k \wedge \tau _ { D _ { k } }$ is the minimum of two nondecreasing sequences, it is itself also nondecreasing.

We next show that $\operatorname* { s u p } _ { k } \sigma _ { k } = \tau _ { D }$ pathwise. Since $D _ { k } \subset D$ , we have $\sigma _ { k } \le \tau _ { D }$ for every $k ,$ so $\operatorname* { s u p } _ { k } \sigma _ { k } \leq \tau _ { D }$ . For the reverse direction, fix $\omega$ and $t < \tau _ { D }$ . Then $\{ Y _ { s } : 0 \leq s \leq t \}$ is a compact subset of $D$ and hence has strictly positive distance to ∂D, so it is contained in $D _ { k _ { 0 } }$ for some $k _ { 0 }$ For $k > \operatorname* { m a x } ( k _ { 0 } , t )$ we have $D _ { k _ { 0 } } \subseteq D _ { k }$ , so $\tau _ { D _ { k } } > t$ and $\sigma _ { k } = k \wedge \tau _ { D _ { k } } > t$ . As $t < \tau _ { D }$ was arbitrary, we have $\operatorname* { s u p } _ { k } \sigma _ { k } \geq \tau _ { D }$

Part (ii) Fix $k \geq 1$ . We distinguish two cases, depending on whether the initial point $x$ is in $\bar { D } _ { k }$ or not. For the first case, consider $x \notin \bar { D } _ { k }$ . Since $\mathbb { X } \backslash { \bar { D } } _ { k }$ is open and paths are continuous, there is $\epsilon > 0$ such that $Y _ { t } \notin \bar { D } _ { k }$ , and in particular $Y _ { t } \notin D _ { k }$ , for all $t \in ( 0 , \epsilon )$ . Hence $\tau _ { D _ { k } } = 0$ and $\sigma _ { k } = 0$ Then $t \wedge \sigma _ { k } = 0$ for every $t \geq 0$ . We have $h ( Y _ { 0 } ) Z _ { 0 } - h ( x ) = 0$ as $Z _ { 0 } = 1$ and all integrals from 0 to $\sigma _ { k } = 0$ are zero. Hence, both sides of (25) and (26) vanish.

For the second case where $x \in \bar { D } _ { k }$ , we first show that $Y _ { t } \in \bar { D } _ { k }$ for all $t \in [ 0 , \sigma _ { k } ]$ . Indeed, $Y _ { 0 } =$ $x \in \bar { D } _ { k }$ , and for $t \in ( 0 , \sigma _ { k } )$ , we have $t < \tau _ { D _ { k } }$ , hence $Y _ { t } \in D _ { k }$ . If $\sigma _ { k } > 0$ , then $Y _ { \sigma _ { k } } = \operatorname* { l i m } _ { t \to \sigma _ { k } } Y _ { t } \in { \bar { D } } _ { k }$ by path continuity.

Since $D _ { k }$ is a compact subset of the open set $D ,$ there exists some $\zeta _ { k } \in C _ { c } ^ { \infty } ( D )$ with $\zeta _ { k } \equiv 1$ on an open neighbourhood $U _ { k }$ of $\bar { D } _ { k }$ , so that in particular, $U _ { k } \subseteq \operatorname { s u p p } \zeta _ { k } \subset D$ . Let $h _ { k }$ equal $\zeta _ { k } h$ on D and $h _ { k } : = 0 \mathrm { ~ o n ~ } \mathbb { R } ^ { n } \setminus D$ . The open sets $D$ and $\mathbb { R } ^ { n } \backslash$ supp $\zeta _ { k }$ cover $\mathbb { R } ^ { n } , h _ { k }$ is $C ^ { 2 }$ on the first and identically zero on the second, and the two definitions agree on the overlap. Thus $h _ { k } \in C ^ { 2 } ( \mathbb R ^ { n } )$ with compact support. In addition, we have $h _ { k } = h , \nabla h _ { k } = \nabla h$ and $L _ { Y } h _ { k } = L _ { Y } h$ on $U _ { k }$

Since $\begin{array} { r } { Z _ { t } = \exp ( - \int _ { 0 } ^ { t } c ( Y _ { s } ) d s ) } \end{array}$ is absolutely continuous in t with $d Z _ { t } = - Z _ { t } c ( Y _ { t } ) d t$ , it has locally bounded variation and hence $d Z _ { t } ^ { \top } d Y _ { t } = 0$ . Applying Itô’s product rule to $h _ { k } ( Y _ { t } ) Z _ { t }$ on $[ 0 , t \land \sigma _ { k } ]$ 2 we have

$$
h _ { k } ( Y _ { t \wedge \sigma _ { k } } ) Z _ { t \wedge \sigma _ { k } } - h _ { k } ( x ) = \int _ { 0 } ^ { t \wedge \sigma _ { k } } Z _ { s } \big ( L _ { Y } h _ { k } - c h _ { k } \big ) ( Y _ { s } ) d s + \int _ { 0 } ^ { t \wedge \sigma _ { k } } Z _ { s } \nabla h _ { k } ( Y _ { s } ) ^ { \top } \sigma ( Y _ { s } ) d W _ { s } .\tag{27}
$$

Since $Y _ { s } \in \bar { D } _ { k } \subset U _ { k }$ for $s \leq t \wedge \sigma _ { k }$ , every term above may be written in terms of $h ,$ yielding (25). Finally, $\sigma _ { k } \le k$ is a bounded stopping time, $\nabla h _ { k }$ is continuous with compact support, hence bounded on $\mathbb { R } ^ { n }$ . In addition, σ is bounded by Assumption 2, and $Z _ { s } \in ( 0 , 1 ]$ since $c \geq 0$ . Hence, we have $\begin{array} { r } { \mathbb { E } \big [ \int _ { 0 } ^ { k } | Z _ { s } \nabla h _ { k } ( Y _ { s } ) ^ { \top } \sigma ( Y _ { s } ) | ^ { 2 } d s \big ] < \infty } \end{array}$ and the optional stopping theorem (Karatzas and Shreve, 2014, Theorem 1.3.22) applies. Taking $t = k$ in (25), so that $t \wedge \sigma _ { k } = \sigma _ { k }$ , and taking expectations proves Eq. (26). □

## B Proof of Results in Section 3

## B.1 Proof of Lemma 1

Since D<sup>¯</sup> is bounded by Assumption 3, we have $b \in C ^ { 0 , \alpha } ( \bar { D } )$ for $\alpha \in ( 0 , 1 )$ . We show that the SDE (1) with $X _ { 0 } = x \in D$ has a pathwise unique strong solution, and its exit time $\tau _ { D }$ satisfies $\mathbb { E } _ { x } [ \tau _ { D } ] \leq \varphi ( x ) < \infty$ , where $\varphi \in C ^ { 2 , \alpha } ( \bar { D } ) , \varphi \geq 0$ is the unique solution of the elliptic BVP

$$
L \varphi = - 1 \mathrm { i n } D , \varphi | _ { \partial D } = 0 .\tag{28}
$$

Step 1: Consider the elliptic BVP $L \varphi = - 1$ in $D , \varphi | _ { \partial D } = 0$ . Since $b \in C ^ { 0 , \alpha } ( \bar { D } )$ and $a =$ $\sigma \sigma ^ { \top } \in C ^ { 0 , \alpha } ( \bar { D } )$ by Assumption 2, $\partial D \in C ^ { 2 , \alpha }$ by Assumption 3, and the boundary data $0 ~ \in$

$C ^ { 2 , \alpha } ( \partial D )$ , elliptic regularity theory (Ladyzhenskaya and Uraltseva, 1968, Ch. III, Theorem 1.1 and Theorem 1.2) guarantees existence and uniqueness of a solution $\varphi \in C ^ { 2 , \alpha } ( \bar { D } )$

Step 2: To establish that $\varphi \geq 0$ on D<sup>¯</sup>, we will show that the minimum of $\varphi$ over D<sup>¯</sup> is attained at a boundary point and is therefore equal to 0 since $\varphi | _ { \partial D } \equiv 0$ . Suppose, for contradiction, that $\varphi$ attains a local minimum at some interior point $x _ { 0 } \in D$ . Then $\nabla \varphi ( x _ { 0 } ) = 0$ and Hess $( \varphi ( x _ { 0 } ) ) \succeq 0$ by the second-order necessary condition for a local minimum. And we have

$$
L \varphi ( x _ { 0 } ) = \underbrace { b ( x _ { 0 } ) ^ { \top } \nabla \varphi ( x _ { 0 } ) } _ { = 0 } + \frac { 1 } { 2 } \operatorname { t r } \big ( a ( x _ { 0 } ) \operatorname { H e s s } \left( \varphi ( x _ { 0 } ) \right) \big ) ~ \geq ~ 0 .\tag{29}
$$

Here, by uniform ellipticity in Assumption 2, we have $a ( x _ { 0 } ) \succ 0$ , and since Hess $( \varphi ( x _ { 0 } ) ) \succeq 0$ , the inequality holds. This contradicts $L \varphi \equiv - 1 < 0$ on D. Hence $\varphi$ has no interior local minimum, so min $\bar { D } \varphi =$ min<sub>∂D</sub> $\varphi = 0$ , i.e. $\varphi \geq 0$ on $\bar { D } .$

Step 3: Since $b , \sigma$ are bounded and Lipschitz by Assumption 1, the SDE (1) has a pathwise unique strong solution for all $t \geq 0$ (Oksendal, 2000, Thm. 5.2.1).

Step $\it 4 .$ We next show $\mathbb { E } _ { x } [ \tau _ { D } ] \leq \varphi ( x )$ . By the previous step, Eq. (1) has a pathwise unique strong solution, so Lemma 5 applies with $h = \varphi$ and $c \equiv 0$ . Let $( \sigma _ { k } ) _ { k \geq 1 }$ and $( D _ { k } ) _ { k \geq 1 }$ be the stopping times and open sets defined in Lemma 5. Since $L \varphi = - 1$ on D, Eq. (26) gives $\mathbb { E } _ { x } [ \varphi ( X _ { \sigma _ { k } } ) ] - \varphi ( x ) =$ $- \mathbb { E } _ { x } [ \sigma _ { k } ]$ . By Step $2 , \varphi \geq 0$ on D<sup>¯</sup>. Since $D _ { k } \subset D$ , we have $\sigma _ { k } \le \tau _ { D _ { k } } \le \tau _ { D }$ . And, since the path stays in $\bar { D }$ up to its exit time $\tau _ { D }$ , we have $X _ { \sigma _ { k } } \in \bar { D }$ . Thus we have $\mathbb { E } _ { x } [ \varphi ( X _ { \sigma _ { k } } ) ] \ge 0$ , and hence

$$
\mathbb { E } _ { x } [ \sigma _ { k } ] \le \varphi ( x ) , \qquad k \in \mathbb { N } .\tag{30}
$$

Since the sequence $( \sigma _ { k } ) _ { k \geq 1 }$ is nondecreasing and $\sigma _ { k }  \tau _ { D }$ from Lemma $5 ( \mathrm { i } )$ , monotone convergence applied to (30) gives $\begin{array} { r } { \mathbb { E } _ { x } \overline { { [ \tau _ { D } ] } } = \operatorname* { l i m } _ { k } \mathbb { E } _ { x } [ \sigma _ { k } ] \leq \varphi ( x ) < \infty } \end{array}$ . Here, since $\varphi \in C ( \hat { D } )$ and D<sup>¯</sup> is compact, $\operatorname* { m a x } _ { \bar { D } } \varphi < \infty$ , so $\begin{array} { r } { \operatorname* { s u p } _ { x \in D } \mathbb { E } _ { x } [ \tau _ { D } ] \leq } \end{array}$ max <sub>¯</sub> φ < ∞. Finally, a nonnegative random variable with finite expectation is finite almost surely, hence $\tau _ { D } < \infty \ \mathrm { a . s }$

## B.2 Finite Mean Exit Time for Controlled Processes

By Assumption 1, b is bounded and Lipschitz on X. More generally, for $u \in \mathcal { U } , b + a u$ is bounded and Lipschitz on $\mathbb { X } ,$ as a is bounded and Lipschitz by Assumption 2, and u is Lipschitz on ${ \bar { D } } ,$ which is compact, hence bounded, and is extended to X. Hence, the following result on the finite mean exit time for every admissible control is immediate by applying Lemma 1 to $\tilde { b } = b + a u$

Corollary 1. Consider the controlled SDE (9), and denote $\begin{array} { r } { \tilde { L } g : = ( b + a u ) ^ { \top } \nabla g + \frac { 1 } { 2 } \sum _ { i j } a _ { i j } \partial _ { i j } g } \end{array}$ Let Assumptions 1–3 hold and let $u \in \mathcal { U }$ . Then, the controlled SDE (9) starting from $x \in D$ has a pathwise unique strong solution $\tilde { X }$ , and its exit-time $\tilde { \tau } _ { D }$ satisfies $\tilde { \mathbb { E } } _ { x } [ \tilde { \tau } _ { D } ] < \infty$ . Thus we also have $\tilde { \tau } _ { D } < \infty , \tilde { P } ^ { u , x } { - } a . s .$

## B.3 Proof of Lemma 2

To show $w = \rho$ on D, let $Y _ { t } : = w ( X _ { t } )$ and $\begin{array} { r } { Z _ { t } : = \exp ( - \int _ { 0 } ^ { t } f ( X _ { s } ) d s ) } \end{array}$ . Fix $x \in D$ and we apply Lemma 5 with $h = w$ , and $c = f$ that satisfies Assumption 4. Since $L w - f w = 0$ in $D$ , the drift term in (26) vanishes and we have

$$
\mathbb { E } _ { x } \big [ w ( X _ { \sigma _ { k } } ) Z _ { \sigma _ { k } } \big ] = w ( x ) , \qquad k \in \mathbb { N } .\tag{31}
$$

We now take $k  \infty$ . By Lemma 1, $\tau _ { D } ~ < ~ \infty , ~ P ^ { x } \mathrm { { - a . s } }$ . Since $\sigma _ { k } \  \ \tau _ { D }$ and $D$ is open, $X _ { \sigma _ { k } }  X _ { \tau _ { D } } \in \partial D$ . By continuity of w on $\bar { D }$ and using the boundary condition (11), we have $Y _ { \sigma _ { k } } =$ $w ( X _ { \sigma _ { k } } ) \to e ^ { - \Phi ( X _ { \tau _ { D } } ) } { \mathrm { ~ a . s } }$ . By continuity of $\textstyle t \mapsto \int _ { 0 } ^ { t } f ( X _ { s } )$ ds, we have $\begin{array} { r } { Z _ { \sigma _ { k } } \to \exp ( - \int _ { 0 } ^ { \tau _ { D } } f ( X _ { s } ) d s ) } \end{array}$ a.s. Moreover, since $w \in C ( \bar { D } )$ with D<sup>¯</sup> compact and $Z _ { t } \in ( 0 , 1 ]$ , we have $\vert Y _ { \sigma _ { k } } Z _ { \sigma _ { k } } \vert \le \operatorname* { m a x } _ { \bar { D } } \vert w \vert < \infty$ for every k. Hence, $\begin{array} { r } { Y _ { \sigma _ { k } } Z _ { \sigma _ { k } } \to e ^ { - \Phi ( X _ { \tau _ { D } } ) } \exp ( - \int _ { 0 } ^ { \tau _ { D } } f d s ) \mathrm { ~ \dot { a } . s } } \end{array}$ ., and this sequence is uniformly bounded by $\operatorname* { m a x } _ { \bar { D } } \left| w \right|$ . Thus, the bounded convergence theorem gives

$$
\operatorname* { l i m } _ { k \to \infty } \mathbb { E } _ { x } \left[ Y _ { \sigma _ { k } } Z _ { \sigma _ { k } } \right] = \mathbb { E } _ { x } \left[ e ^ { - \Phi ( X _ { \tau _ { D } } ) } \exp \left( - \int _ { 0 } ^ { \tau _ { D } } f ( X _ { s } ) d s \right) \right] = \rho ( x ) ,\tag{32}
$$

where the last equality follows from the definition $\rho$ in (10). On the other hand, $\mathbb { E } _ { x } [ Y _ { \sigma _ { k } } Z _ { \sigma _ { k } } ] = w ( x )$ for every k by (31), so the limit on the left is $w ( x )$ . Therefore, we conclude $w ( x ) = \rho ( x )$ on $D$ For $x \in \partial D$ , we have $\tau _ { D } = 0 , P ^ { x } ~ \mathrm { a . s }$ . hence (10) gives $\rho ( x ) = e ^ { - \Phi ( x ) } = w ( x )$ via the boundary condition (11). Hence, $w ( x ) = \rho ( x )$ on $\bar { D }$

It remains to show $\rho > 0$ on D<sup>¯</sup>. On the boundary, $\rho | _ { \partial D } = e ^ { - \Phi } = g > 0$ by Assumption 4. In the interior, fix $x \in D$ : by Assumption 4, $f \geq 0$ is bounded on D<sup>¯</sup> and $\Phi = - \log g$ is bounded on $\partial D$ , while $\tau _ { D } < \infty \ \mathrm { a . s }$ . by Lemma 1. Hence, we have

$$
\exp \Bigl ( - \int _ { 0 } ^ { \tau _ { D } } f ( X _ { t } ) d t - \Phi ( X _ { \tau _ { D } } ) \Bigr ) > 0 , \qquad P ^ { x _ { - \mathrm { a . s . } , } }\tag{33}
$$

and taking expectations gives $\rho ( x ) > 0$ . Combining these, $\rho > 0$ on $\bar { D } .$ . Since $\rho \in C ( \hat { D } )$ and $\bar { D }$ is compact, it follows that min $\bar { D } \rho > 0$

## B.4 Proof of Lemma 3

Recall $x = ( x _ { 1 } , \ldots , x _ { n } ) \in \mathbb { X }$ , and $\partial _ { i j }$ denotes the second partial derivative with respect to $x _ { i }$ and $x _ { j }$ . The gradient of $\rho$ is given by

$$
\nabla \rho = e ^ { - v } ( - \nabla v ) = - \rho \nabla v .\tag{34}
$$

For the second-order part $\begin{array} { r } { \sum _ { i j } a _ { i j } \partial _ { i j } \rho . } \end{array}$ we have

$$
\sum _ { i j } a _ { i j } \partial _ { i j } \rho = - \sum _ { i j } a _ { i j } \partial _ { j } \rho \partial _ { i } v - \rho \sum _ { i j } a _ { i j } \partial _ { i j } v .\tag{35}
$$

Substituting $\partial _ { j } \rho = - \rho \partial _ { j } v$ , we have

$$
\sum _ { i j } a _ { i j } \partial _ { i j } \rho = \rho \sum _ { i j } a _ { i j } \partial _ { i } v \partial _ { j } v - \rho \sum _ { i j } a _ { i j } \partial _ { i j } v = \rho \big [ ( \nabla v ) ^ { \top } a \nabla v - \Delta _ { a } v \big ] ,\tag{36}
$$

where $\textstyle \Delta _ { a } v : = \sum _ { i j } a _ { i j } \partial _ { i j } v$ . Hence, $L \rho$ can be written as

$$
\begin{array} { r l } & { L \rho = b ^ { \top } ( - \rho \nabla v ) + \frac { 1 } { 2 } \rho \big [ ( \nabla v ) ^ { \top } a \nabla v - \Delta _ { a } v \big ] } \\ & { \quad = \rho \left[ - b ^ { \top } \nabla v - \frac { 1 } { 2 } \Delta _ { a } v + \frac { 1 } { 2 } ( \nabla v ) ^ { \top } a \nabla v \right] } \\ & { \quad = \rho \left[ - L v + \frac { 1 } { 2 } ( \nabla v ) ^ { \top } a \nabla v \right] , } \end{array}\tag{37}
$$

where the last line follows from $\begin{array} { r } { b ^ { \top } \nabla v + \frac { 1 } { 2 } \Delta _ { a } v = L v } \end{array}$ . Using $L \rho - f \rho = 0$ , and since $\rho > 0$ , divide by $\rho ,$ we have

$$
\begin{array} { r } { - L v + \frac { 1 } { 2 } ( \nabla v ) ^ { \top } a \nabla v = f . } \end{array}\tag{38}
$$

Rearranging the above equation, we have

$$
\begin{array} { r } { L v ( x ) + f ( x ) = \frac { 1 } { 2 } ( \nabla v ( x ) ) ^ { \top } a ( x ) \nabla v ( x ) \quad \mathrm { i n ~ } D , \qquad v \big | _ { \partial D } = \Phi . } \end{array}\tag{39}
$$

We next show that this coincides with the HJB equation for the OSC (12) up to an exit time. Let ${ \tilde { X } } _ { t }$ solve the controlled SDE (9). Since $v \in C ^ { 2 } ( D )$ , Itô’s formula gives

$$
d v ( \tilde { X } _ { t } ) = \underbrace { \left[ L v ( \tilde { X } _ { t } ) + ( \nabla v ( \tilde { X } _ { t } ) ) ^ { \top } a ( \tilde { X } _ { t } ) u ( \tilde { X } _ { t } ) \right] } _ { \mathrm { d r i f t } } d t + \nabla v ( \tilde { X } _ { t } ) ^ { \top } \sigma ( \tilde { X } _ { t } ) d \tilde { W } _ { t } .\tag{40}
$$

For $u \in \mathcal { U }$ , substituting (39) into the drift, and adding and subtracting $\begin{array} { r } { \frac { 1 } { 2 } \boldsymbol { u } ^ { \top } \boldsymbol { a } \boldsymbol { u } . } \end{array}$ , we have

$$
\begin{array} { r } { \mathrm { d r i f t } = - \Big ( f + \frac { 1 } { 2 } \boldsymbol { u } ^ { \top } \boldsymbol { a } \boldsymbol { u } \Big ) + \frac { 1 } { 2 } \big ( \nabla \boldsymbol { v } + \boldsymbol { u } \big ) ^ { \top } \boldsymbol { a } \left( \nabla \boldsymbol { v } + \boldsymbol { u } \right) , } \end{array}\tag{41}
$$

where the last term $\geq 0$ since $a \succeq 0$ . Equation (41) also shows that, for each $x \in D$ , the mapping $\boldsymbol { u } \mapsto \boldsymbol { u } ^ { \intercal } \boldsymbol { a } \nabla \boldsymbol { v } + \frac { 1 } { 2 } \boldsymbol { u } ^ { \intercal }$ au is minimized uniquely at $\boldsymbol { u } ^ { * } = - \nabla \boldsymbol { v }$ , since $a \succ 0$ by Assumption 2. Therefore, (13) is the HJB equation of the OSC (12) up to an exit time, with a candidate optimal controller $u ^ { * }$ in (14).

## B.5 Proof of Proposition 1

Since $H \geq 0$ , we have $H e ^ { - H } \geq 0$ and hence $- \mathbb { E } _ { x } [ H e ^ { - H } ] < \infty$ . Together with $Z \in \mathsf { \Gamma } ( 0 , \infty )$ hypotheses of Mitter and Newton (2003, Proposition 2.1) hold on $( \Omega , \mathcal { F } _ { \tau _ { D } } )$ with reference measure $P ^ { x } | _ { \mathcal { F } _ { \tau _ { D } } }$ and energy H. Hence, the claim follows.

## B.6 Proof of Lemma 4

Since $\rho \in C ^ { 2 , \alpha } ( \hat { D } )$ as established in Lemma 2, and − log is smooth on $[ m , M ]$ , the chain rule gives $v = - \log \circ \rho \in C ^ { 2 , \alpha } ( \bar { D } )$ , thus we have $u ^ { * } = - \nabla v \in C ^ { 1 , \alpha } ( \bar { D } )$ , hence $u ^ { * } \in \mathcal { U }$ . By the paragraph preceding Corollary 1, $\boldsymbol { \bar { b } } : = \boldsymbol { b } - a \nabla \boldsymbol { v }$ is bounded and Lipschitz on $\mathbb { R } ^ { n }$ . Thus, (9) has a pathwise unique strong solution for all $t \geq 0$ by the standard SDE existence-uniqueness theorem for bounded, Lipschitz coeficients; see, $\mathrm { e . g . }$ , (Oksendal, 2000, Thm. 5.2.1).

Define an exponential process as

$$
Z _ { t } : = \exp \left( \int _ { 0 } ^ { t } \left( ( \sigma ^ { \top } u ^ { * } ) ( X _ { s } ) \right) ^ { \top } d W _ { s } - \frac { 1 } { 2 } \int _ { 0 } ^ { t } | \sigma ^ { \top } u ^ { * } ( X _ { s } ) | ^ { 2 } d s \right) , \qquad t < \tau _ { D } .\tag{42}
$$

Since $v \in C ^ { 2 , \alpha } ( \bar { D } )$ and $\bar { D }$ is compact by Assumption 1, ∇v is continuous and hence bounded on $\bar { D }$ Together with σ bounded by Assumption 2, let $C : = \left\| \sigma \right\| _ { \infty } \left\| \nabla v \right\| _ { \infty }$ and we have $\left| \sigma ^ { \top } u ^ { * } \right| \leq C$ on D<sup>¯</sup>. Since $\tau _ { D } < \infty \ \mathrm { a . s }$ . by Lemma 1, we have $\begin{array} { r } { \int _ { 0 } ^ { \tau _ { D } } | \sigma ^ { \top } u ^ { * } ( X _ { s } ) | ^ { 2 } d s \leq C ^ { \tilde { 2 } } \tau _ { D } < } \end{array}$ ∞ a.s. Thus the stochastic integral in (42) is well defined at $t = \tau _ { D } .$ , and $t \mapsto Z _ { t }$ is continuous on $[ 0 , \tau _ { D } ]$

Step 1: We compute $Z _ { t }$ as follows. Since $a = \sigma \sigma ^ { \top }$ and $\boldsymbol { u } ^ { * } = - \nabla \boldsymbol { v }$ , we have $( \nabla v ) ^ { \top } a \nabla v = | \sigma ^ { \top } u ^ { * } | ^ { 2 }$ so the HJB equation (39) is $\begin{array} { r } { L v = \frac { 1 } { 2 } | \sigma ^ { \top } u ^ { * } | ^ { 2 } - f } \end{array}$ in D. Applying Lemma 5 with $h = v$ and $c = 0$ and taking k large enough that $\sigma _ { k } > t .$ , which is possible since $\sigma _ { k }  \tau _ { D }$ , for $t < \tau _ { D }$ , we have,

$$
\begin{array} { r } { d v ( X _ { t } ) = L v ( X _ { t } ) d t + \nabla v ( X _ { t } ) ^ { \top } \sigma ( X _ { t } ) d W _ { t } = \Big ( \frac { 1 } { 2 } | \sigma ^ { \top } u ^ { * } ( X _ { t } ) | ^ { 2 } - f ( X _ { t } ) \Big ) d t - \Big ( ( \sigma ^ { \top } u ^ { * } ) ( X _ { t } ) \Big ) ^ { \top } d W _ { t } , } \end{array}\tag{43}
$$

where we used $\nabla \boldsymbol { v } = - \boldsymbol { u } ^ { * }$ and the HJB equation (13).

Rearranging and integrating from 0 to $t < \tau _ { D }$ , we have

$$
\int _ { 0 } ^ { t } \Big ( ( \sigma ^ { \top } u ^ { * } ) ( X _ { s } ) \Big ) ^ { \top } d W _ { s } - \frac { 1 } { 2 } \int _ { 0 } ^ { t } | \sigma ^ { \top } u ^ { * } ( X _ { s } ) | ^ { 2 } d s = v ( x ) - v ( X _ { t } ) - \int _ { 0 } ^ { t } f ( X _ { s } ) d s .\tag{44}
$$

Taking exponential and using $e ^ { - v } = \rho _ { ; }$ , we have

$$
Z _ { t } = { \frac { \rho ( X _ { t } ) } { \rho ( x ) } } \exp \left( - \int _ { 0 } ^ { t } f ( X _ { s } ) d s \right) , \qquad t < \tau _ { D } .\tag{45}
$$

Step 2: We now take $t  \tau _ { D }$ in (45). By Lemma 1, we have $\tau _ { D } < \infty , P ^ { x } \mathrm { - a . s . }$ , thus $X _ { t } $ $X _ { \tau _ { D } } \in \partial D$ by continuity of the path, and $\rho ( X _ { t } ) \to \rho ( X _ { \tau _ { D } } ) = e ^ { - \Phi ( X _ { \tau _ { D } } ) }$ by continuity of $\rho$ on D<sup>¯</sup> together with the boundary condition $\rho | _ { \partial D } = e ^ { - \Phi }$ in (11). Moreover, $\begin{array} { r } { \int _ { 0 } ^ { t } f ( X _ { s } ) d s \to \int _ { 0 } ^ { \tau _ { D } } f ( X _ { s } ) } \end{array}$ ds by continuity of $\begin{array} { r } { t \mapsto \int _ { 0 } ^ { t } f ( X _ { s } ) d s } \end{array}$ , the limit is finite since f is bounded by Assumption 4 and $\tau _ { D } < \infty$ a.s. Hence, we have

$$
Z _ { \tau _ { D } } = { \frac { e ^ { - \Phi ( X _ { \tau _ { D } } ) } } { \rho ( x ) } } \exp \Bigl ( - \int _ { 0 } ^ { \tau _ { D } } f ( X _ { s } ) d s \Bigr ) = { \frac { e ^ { - H ( \chi ) } } { \rho ( x ) } } , \qquad P ^ { x } \mathrm { - a . s . }\tag{46}
$$

Taking expectations, the Feynman–Kac identity $\rho ( x ) = \mathbb { E } _ { x } [ e ^ { - H ( \chi ) } ]$ of (10) gives

$$
\mathbb { E } _ { x } \big [ Z _ { \tau _ { D } } \big ] = \frac { \rho ( x ) } { \rho ( x ) } = 1 .\tag{47}
$$

Step 3: Since $\bar { D }$ is compact and $\rho \in C ( \hat { D } )$ , we have $0 < m \le \rho \le M < \infty$ on $\bar { D } ;$ and $\textstyle \exp ( - \int _ { 0 } ^ { t } f d s ) \in ( 0 , 1 ]$ since $f \geq 0$ . Hence (45) gives the uniform, deterministic bound

$$
0 < Z _ { t \wedge \tau _ { D } } \leq { \frac { M } { \rho ( x ) } } \qquad { \mathrm { f o r ~ e v e r y ~ } } t \geq 0 { \mathrm { ~ a n d ~ a . e . ~ } } \omega .\tag{48}
$$

The stopped process $Z _ { \cdot \wedge \tau _ { D } }$ is a stochastic exponential, hence a local martingale; by (48) it is bounded, and a bounded local martingale is a true martingale. Therefore $\mathbb { E } _ { x } [ Z _ { t \wedge \tau _ { D } } ] = 1$ for every $t \geq 0$

Step 4: Finally, define $Q$ on $\mathcal { F } _ { \tau _ { D } }$ via $d Q / d P ^ { x } = Z _ { \tau _ { D } }$ , which is a probability measure by the above. By Girsanov’s theorem, we have $\begin{array} { r } { W _ { t } - \int _ { 0 } ^ { t } \sigma ^ { \top } u ^ { * } ( \bar { X } _ { s } ) } \end{array}$ ds is a Q-Brownian motion on $[ 0 , \tau _ { D } ]$ Thus under $Q .$ , the process solves the controlled SDE (9) with $u = u ^ { * }$ up to $\tau _ { D }$ . By uniqueness in law for said SDE, established above, we have $Q = \tilde { P } ^ { * , x }$ on $\mathcal { F } _ { \tau _ { D } }$ , which together with (46) proves (15).

## B.7 Proof of Theorem 1

Part (i). We first consider $u \in \mathcal { U } .$ and consider the optimal control $u ^ { * }$ at the end as the special case. We proceed in four steps: (1) applying Itô’s formula for $v ( \tilde { X } _ { t } )$ under $u \in \mathcal { U } ; ( 2 )$ applying Lemma 5 to integrate v on $\bar { D } ; ( 3 )$ passing $k \to \infty ;$ and (4) rearranging and specializing to $u = u ^ { * }$

Step 1: From Itô’s formula and (41), for the controlled process $\tilde { X }$ ,we have

$$
\begin{array} { r } { d v ( \tilde { X } _ { t } ) = - \Big ( f ( \tilde { X } _ { t } ) + \frac 1 2 u ^ { \top } a u ( \tilde { X } _ { t } ) \Big ) d t + \frac 1 2 \big ( \nabla v + u \big ) ^ { \top } a \left( \nabla v + u \right) ( \tilde { X } _ { t } ) d t + \nabla v ( \tilde { X } _ { t } ) ^ { \top } \sigma ( \tilde { X } _ { t } ) d \tilde { W } _ { t } . } \end{array}\tag{49}
$$

Step 2: The above equation is derived from $v \in C ^ { 2 } ( D )$ , so it may be integrated only while the trajectory remains in D. We therefore apply Lemma 5 with drift $b + a u , h = v$ and $c \equiv 0$ along the controlled difusion process (9), where admissibility of u and Assumption 2 make the drift term bounded and Lipschitz. Substituting the drift decomposition in (41) into (26), we have

$$
\underbrace { \tilde { \mathbb { E } } \big [ { v } ( \tilde { X } _ { \sigma _ { k } } ) \big ] } _ { : = T _ { 1 } } - { v } ( x ) = - \underbrace { \tilde { \mathbb { E } } \left[ \int _ { 0 } ^ { \sigma _ { k } } \Bigl ( { f } + \frac { 1 } { 2 } { u } ^ { \top } a u \Bigr ) d t \right] } _ { : = T _ { 2 } } + \underbrace { \tilde { \mathbb { E } } \left[ \int _ { 0 } ^ { \sigma _ { k } } \frac { 1 } { 2 } ( \nabla v + u ) ^ { \top } a \left( \nabla v + u \right) d t \right] } _ { : = T _ { 3 } } .\tag{50}
$$

Step 3: We next study the three terms $T _ { 1 } – T _ { 3 }$ in (50) separately. For the boundary term $T _ { 1 }$ , recall by Corollary $1 , \tilde { \tau } _ { D } < \infty \mathrm { a . s . }$ , and we have shown before that $\sigma _ { k }  \tilde { \tau } _ { D }$ as $k  \infty$ in Lemma $5 .$ Hence $\tilde { X } _ { \sigma _ { k } } ~  ~ \tilde { X } _ { \tilde { \tau } _ { D } }$ a.s. by continuity of the path. Since the controlled process exits D at time $\tilde { \tau } _ { D }$ , we have $\tilde { X } _ { \tilde { \tau } _ { D } } \in \partial D$ a.s., so continuity of v on $\bar { D }$ together with the boundary condition $v | _ { \partial D } = \Phi$ gives $v ( \tilde { X } _ { \sigma _ { k } } ^ { - } )  v ( \tilde { X } _ { \tilde { \tau } _ { D } } ) = \Phi ( \tilde { X } _ { \tilde { \tau } _ { D } } )$ a.s. Moreover $v \in C ( \bar { D } )$ with D<sup>¯</sup> compact gives $\vert v \vert \leq M : = \mathrm { m a x } _ { \bar { D } } \vert v \vert < \infty$ , so $| v ( \tilde { X } _ { \sigma _ { k } } ) | \le M$ for every k and a.e. ω. By bounded convergence theorem, we thus have

$$
\operatorname* { l i m } _ { k \to \infty } \tilde { \mathbb { E } } \bigl [ v ( \tilde { X } _ { \sigma _ { k } } ) \bigr ] = \tilde { \mathbb { E } } \bigl [ \Phi ( \tilde { X } _ { \tilde { \tau } _ { D } } ) \bigr ] .\tag{51}
$$

For the running cost $T _ { 2 } ,$ , since $f \geq 0$ by Assumption 4 and $a \succ 0$ by Assumption 2, we have that $f + { \textstyle \frac { 1 } { 2 } } u ^ { \top }$ a u is nonnegative. As $\sigma _ { k }$ is nondecreasing with $\sigma _ { k }  \tilde { \tau } _ { D }$ from Lemma 5, the integrals $\begin{array} { r } { \int _ { 0 } ^ { \sigma _ { k } } ( f + \frac { 1 } { 2 } u ^ { \top } a u ) } \end{array}$ dt are nondecreasing in k and converge pointwise a.s. to $\begin{array} { r } { \int _ { 0 } ^ { \tilde { \tau } _ { D } } ( f + \frac { 1 } { 2 } \boldsymbol { u } ^ { \top } \boldsymbol { a } \boldsymbol { u } ) } \end{array}$ dt. By monotone convergence theorem, we have

$$
\operatorname* { l i m } _ { k \to \infty } { \tilde { \mathbb { E } } } \left[ \int _ { 0 } ^ { \sigma _ { k } } { \Bigl ( } f + { \frac { 1 } { 2 } } u ^ { \top } a u { \Bigr ) } d t \right] = { \tilde { \mathbb { E } } } \left[ \int _ { 0 } ^ { \tilde { \tau } _ { D } } { \Bigl ( } f + { \frac { 1 } { 2 } } u ^ { \top } a u { \Bigr ) } d t \right] .\tag{52}
$$

For the last term $T _ { 3 }$ , proceed similarly as above, since $\boldsymbol { a } \succ 0 , \frac { 1 } { 2 } ( \nabla \boldsymbol { v } + \boldsymbol { u } ) ^ { \top } \boldsymbol { a } \left( \nabla \boldsymbol { v } + \boldsymbol { u } \right)$ is nonnegative, so monotone convergence theorem gives

$$
\operatorname* { l i m } _ { k  \infty } \tilde { \mathbb { E } } [ \int _ { 0 } ^ { \sigma _ { k } } \frac { 1 } { 2 } ( \nabla v + u ) ^ { \top } a ( \nabla v + u ) d t ] = \tilde { \mathbb { E } } [ \int _ { 0 } ^ { \tilde { \tau } _ { D } } \frac { 1 } { 2 } ( \nabla v + u ) ^ { \top } a ( \nabla v + u ) d t ] .\tag{53}
$$

We next show that all three limits are finite. Indeed, by Assumption $4 , \Phi = - \log g$ is continuous on the compact set $\partial D _ { : }$ , hence bounded. Let $\begin{array} { r } { C : = \operatorname* { s u p } _ { \bar { D } } \Big ( f + \frac { 1 } { 2 } \boldsymbol { u } ^ { \top } \boldsymbol { a } \boldsymbol { u } \Big ) } \end{array}$ and $C ^ { \prime } : = \operatorname* { s u p } _ { \bar { D } } \frac { 1 } { 2 } ( \nabla v +$ $\boldsymbol { u } ) ^ { \top } \boldsymbol { a } \left( \nabla \boldsymbol { v } + \boldsymbol { u } \right)$ . Both terms are finite because f is bounded by Assumption 4, a is bounded by Assumption $2 , u \in \mathcal { U }$ is bounded on the compact D<sup>¯</sup> by definition, and $\nabla \boldsymbol { v }$ is bounded on D<sup>¯</sup> since $v \in C ^ { 2 , \alpha } ( \bar { D } )$ The two integrals in (52) and (53) are dominated by $C \tilde { \tau } _ { D }$ and $C ^ { \prime } \tilde { \tau } _ { D }$ , respectively, whose expectations are finite since $\tilde { \mathbb { E } } \big [ \tilde { \tau } _ { D } \big ] < \infty$ by Corollary 1. Combining (51)–(53) into (50), we have

$$
\tilde { \mathbb { E } } \big [ \Phi ( \tilde { X } _ { \tilde { \tau } _ { D } } ) \big ] - v ( x ) = - \tilde { \mathbb { E } } \left[ \int _ { 0 } ^ { \tilde { \tau } _ { D } } \Big ( f + \frac { 1 } { 2 } u ^ { \top } a u \Big ) d t \right] + \tilde { \mathbb { E } } \left[ \int _ { 0 } ^ { \tilde { \tau } _ { D } } \frac { 1 } { 2 } ( \nabla v + u ) ^ { \top } a \left( \nabla v + u \right) d t \right] ,\tag{54}
$$

where every term is finite.

Step $\it 4 .$ Rearranging (54), and using $\boldsymbol { \nabla } \boldsymbol { v } + \boldsymbol { u } = \boldsymbol { u } - \boldsymbol { u } ^ { * }$ as $\boldsymbol { u } ^ { * } = - \nabla \boldsymbol { v }$ , we obtain

$$
\begin{array} { r l r } {  { J ( u , x ) = \tilde { \mathbb { E } } \big [ \Phi ( \tilde { X } _ { \tilde { \tau } _ { D } } ) \big ] + \tilde { \mathbb { E } } [ \int _ { 0 } ^ { \tilde { \tau } _ { D } } \Big ( f + \frac { 1 } { 2 } u ^ { \top } a u \Big ) d t ] } } \\ & { } & { = v ( x ) + \tilde { \mathbb { E } } [ \int _ { 0 } ^ { \tilde { \tau } _ { D } } \frac { 1 } { 2 } ( u - u ^ { * } ) ^ { \top } a ( u - u ^ { * } ) d t ] } \\ & { } & { \geq v ( x ) , } \end{array}\tag{55}
$$

where the inequality follows from $a \succ 0$ by Assumption 2. Since $u \in \mathcal { U }$ was arbitrary, $v ( x )$ is a lower bound. Here, equality holds if and only if E<sup>˜</sup> $\begin{array} { r } { \mathrm { : } \left[ \int _ { 0 } ^ { \tilde { \tau } _ { D } } \frac { 1 } { 2 } ( u - u ^ { * } ) ^ { \top } \boldsymbol { a } \left( u - u ^ { * } \right) d t \right] = 0 } \end{array}$ . Since the integrand is nonnegative, this holds if and only if $\begin{array} { r } { \int _ { 0 } ^ { \tilde { \tau } _ { D } } \frac { 1 } { 2 } ( u - u ^ { * } ) ^ { \top } a \left( u - u ^ { * } \right) ( \tilde { X } _ { t } ) d t = 0 , \tilde { P } ^ { u , x } \mathrm { - a . s } } \end{array}$ . Again, by nonnegativity, it holds if and only if $( u - u ^ { * } ) ^ { \top } a ( u - u ^ { * } ) ( \tilde { X } _ { t } ) = 0$ for Lebesgue-almost every $t \in$ $[ 0 , \tilde { \tau } _ { D } ( \omega ) ] , \tilde { P } ^ { u , x _ { - \mathrm { a . s } } }$ . By uniform ellipticity in Assumption 2, we have $\left( u - u ^ { * } \right) ^ { \top } a \left( u - u ^ { * } \right) \geq c | u - u ^ { * } | ^ { 2 }$ with $c > 0$ . So this is equivalent to $u ( { \tilde { X } } _ { t } ) = u ^ { * } ( { \tilde { X } } _ { t } )$ for Lebesgue-almost every $t \in [ 0 , \tilde { \tau } _ { D } ] , \tilde { P } ^ { u , x _ { - } } \mathrm { a . s . }$

It remains to check that this bound is attained. Since $v \in C ^ { 2 , \alpha } ( \bar { D } )$ , we have $\boldsymbol { u } ^ { * } = - \nabla \boldsymbol { v } \in$ $C ^ { 1 , \alpha } ( \bar { D } )$ , hence $u ^ { * } \in \mathcal { U }$ and (55) applies. The remainder term then vanishes identically, and we have $J ( u ^ { * } , x ) = v ( x )$ . Hence, the infimum is attained, and for every $x \in D$ , we have

$$
v ( x ) = J ( u ^ { * } , x ) = \operatorname* { m i n } _ { u \in \mathcal { U } } J ( u , x ) .\tag{56}
$$

This completes the proof of Part (i).

Part $( i i )$ Since $\tau _ { D } < \infty , P ^ { x _ { - } } \mathrm { a . s }$ . as in Lemma $1 , H ( \chi )$ is well-defined and finite $P ^ { x } \mathrm { _ { - a . S . } }$ ., and by the Feynman–Kac representation of $\rho ,$ we can define

$$
Z : = \mathbb { E } _ { x } { \left[ e ^ { - H ( \chi ) } \right] } = \rho ( x ) \in ( 0 , 1 ] ,\tag{57}
$$

so $Z \in ( 0 , \infty )$ as required by (Raginsky, 2026, Proposition 1). Moreover $f \geq 0$ by Assumption 4 and $\Phi = - \log g \ge 0$ since $g \leq 1$ by the same assumption, so $H ( \chi ) \geq 0 ~ P ^ { x } \mathrm { { - a . s } }$ . Hence the integrand $- H ( \chi ) e ^ { - H ( \chi ) }$ is non-positive and $- \mathbb { E } _ { x } \big [ H ( \chi ) e ^ { - H ( \chi ) } \big ] < \infty$ . So both assumptions of Proposition 1 are satisfied, and applying it on $( \Omega , \bar { \mathcal { F } _ { \tau _ { D } } } )$ with $P = P ^ { x }$ and $H = H ( \chi )$ , for every admissible path measure $\tilde { P } \ll P ^ { x }$ , we have

$$
D _ { \mathrm { K L } } ( \tilde { P } \| P ^ { x } ) + \mathbb { E } [ H ( \chi ) ] \ \geq \ - \log \rho ( x ) = v ( x ) ,\tag{58}
$$

with equality if and only if $d \tilde { P } / d P ^ { x } = e ^ { - H ( \chi ) } / \rho ( x )$ . By Part (i), we have $v ( x ) = J ( u ^ { * } , x )$ , so (58) is (16). By Lemma 4, ${ \tilde { P } } ^ { * , x }$ of the optimally-controlled process satisfies

$$
\left. { \frac { d { \tilde { P } } ^ { * , x } } { d P ^ { x } } } \right| _ { \mathcal { F } _ { \tau _ { D } } } = { \frac { e ^ { - H ( \chi ) } } { \rho ( x ) } }\tag{59}
$$

Hence equality in (58), and so in (16), holds if and only if $\tilde { P } = \tilde { P } ^ { * , x }$

## C Proof of Identity (4) in Section 4.1

Let Assumption 2 hold. By construction in Section $4 . 1 , A _ { i } , A _ { j }$ are closed, and $\partial A _ { i } \cap \partial A _ { j } = \emptyset$ . Assume that $D _ { q } = \mathbb { X } \setminus ( A _ { i } \cup A _ { j } )$ satisfies Assumption 3, with $\partial D _ { q } = \partial A _ { i } \cup \partial A _ { j }$ . Denote $\varphi : \partial D _ { q }  \{ 0 , 1 \}$ with $\varphi = 0$ on $\partial A _ { i }$ and $\varphi = 1$ on $\partial A _ { j }$ . Since $\partial A _ { i }$ and $\partial A _ { j }$ are disjoint compact sets, $\varphi$ is locally constant, hence $\varphi \in C ^ { 2 , \alpha } ( \partial D _ { q } )$ . By (Ladyzhenskaya and Uraltseva, 1968, Ch. III, Theorem 1.1 and Theorem 1.2), the boundary value problem

$$
L w = 0 \mathrm { i n } D _ { q } , \qquad w | _ { \partial D _ { q } } = \varphi ,\tag{60}
$$

has a unique solution $w \in C ^ { 2 , \alpha } ( \bar { D } _ { q } )$ . In particular, w is continuous and bounded on the compact set $\bar { D } _ { q }$

We next show $q = w$ . For $\boldsymbol { x } \in D _ { q }$ , let $\tau : = \operatorname* { i n f } \{ t > 0 : X _ { t } \notin D _ { q } \}$ be the first time the trajectory leaves $D _ { q }$ . Since $\mathbb { X } \setminus D _ { q } = A _ { i } \cup A _ { j }$ , we have $\tau = \tau _ { A _ { i } } \wedge \tau _ { A _ { j } }$ . Let $D _ { k } , \ \sigma _ { k }$ be as in Lemma 5 with

$D = D _ { q }$ . Apply Lemma 5(ii) with $h = w$ and $c = 0$ , we have $Z _ { t } = 1$ . Since $L w = 0$ in $D _ { q } .$ , from (26), we have

$$
{ \mathbb E } _ { x } \big [ w ( X _ { \sigma _ { k } } ) \big ] = w ( x ) , \qquad k \in { \mathbb N } .\tag{61}
$$

Here $w ( X _ { \sigma _ { k } } )$ is well defined since $X _ { \sigma _ { k } } \in \bar { D } _ { q }$ by the case split in the proof of Lemma 5.

We next take $k  \infty$ . From Lemma 5(i), we have $\sigma _ { k } \to \tau$ . Since $\tau < \infty \ \mathrm { a . s . , } \ D _ { q }$ is open, and by path continuity, we have $X _ { \sigma _ { k } }  X _ { \tau } \in \partial D _ { q }$ a.s. Since w is continuous on $D _ { q } ,$ we have $w ( X _ { \sigma _ { k } } )  w ( X _ { \tau } ) = \varphi ( X _ { \tau } )$ a.s. Furthermore, $| w ( X _ { \sigma _ { k } } ) | \le \operatorname* { m a x } _ { \bar { D } _ { q } } | w | < \infty$ for $k \in \mathbb N$ . By the bounded convergence theorem, we then have

$$
\boldsymbol { w } ( \boldsymbol { x } ) = \mathbb { E } _ { \boldsymbol { x } } \big [ \boldsymbol { \varphi } ( \boldsymbol { X } _ { \tau } ) \big ] = \boldsymbol { P } ^ { \boldsymbol { x } } \big ( \boldsymbol { X } _ { \tau } \in \partial A _ { j } \big ) ,\tag{62}
$$

where the second equality follows since $\varphi ( X _ { \tau } ) = \mathbf { 1 } _ { \partial A _ { i } } ( X _ { \tau } )$

Finally, we show $P ^ { x } ( X _ { \tau } \in \partial A _ { j } ) = q ( x )$ . Since $A _ { i } , A _ { j }$ are closed by construction, and paths are continuous, and by definition of $\tau ,$ we have $X _ { \tau _ { A _ { k } } } \in A _ { k }$ whenever $\tau _ { A _ { k } } < \infty$ for $k \in \{ i , j \}$ . If $X _ { \tau } \in \partial A _ { j } \subset A _ { j }$ , then $\tau \neq \tau _ { A _ { i } }$ , since otherwise ${ X _ { \tau } } \stackrel { \triangledown } { \in } { \cal A } _ { i } \cap { \cal A } _ { j } = \varnothing$ , and we have $\tau = \tau _ { A _ { i } } < \tau _ { A _ { i } }$ Conversely, if $\tau _ { A _ { j } } < \tau _ { A _ { i } }$ , then $\tau = \tau _ { A _ { j } }$ , and we have $X _ { \tau } \in \partial A _ { j }$ . Thus $\{ X _ { \tau } \in \partial A _ { j } \} = \{ \tau _ { A _ { j } } < \tau _ { A _ { i } } \}$ and for $x \in D _ { q }$ , we have

$$
w ( x ) = P ^ { x } \big ( \tau _ { A _ { j } } < \tau _ { A _ { i } } \big ) = q ( x ) .\tag{63}
$$

By definition of $q ,$ for $x \in \partial A _ { i } \ ( x \in \partial A _ { i } )$ , we have $q ( x ) = 0 = w ( x ) \ ( q ( x ) = 1 = w ( x ) )$ . Hence $q = w$ on $\bar { D } _ { q }$ . Therefore, $q \in C ^ { 2 , \alpha } ( \bar { D } _ { q } )$ , and $q$ is the unique solution of (4).

## D Detailed Computation of the Committor Function from Section 4.1

We first construct the spectral term $\tilde { \psi }$ that separates the two cores using the leading eigenfunctions of L. Consider $A _ { k }$ and the set of leading eigenfunctions $\psi _ { l }$ , for $l = 1 , \ldots , m - 1$ . With the core level $\theta _ { l } ^ { ( k ) }$ defined in (17), we define $\begin{array} { r } { \hat { \psi } ( x ) : = \sum _ { l = 1 } ^ { m - 1 } c _ { l } \psi _ { l } ( x ) } \end{array}$ , where $c _ { l } : = \theta _ { l } ^ { ( j ) } - \theta _ { l } ^ { ( i ) }$ measures how much the l-th mode varies between $A _ { i } , A _ { j }$ . Define the core level of $\hat { \psi }$ as $\begin{array} { r } { \bar { \theta } ^ { ( k ) } : = \frac { 1 } { | \mathcal { T } _ { k } | } \overset { \cdot } { \sum } _ { i \in \mathcal { T } _ { k } } \hat { \psi } ( x _ { i } ) } \end{array}$ . Substituting $\hat { \psi } ;$ , we have $\begin{array} { r } { \bar { \theta } ^ { ( k ) } = \sum _ { l } c _ { l } \theta _ { l } ^ { ( k ) } } \end{array}$ . Let $c = [ c _ { 1 } , \cdots , c _ { m - 1 } ]$ . We recenter and rescale $\hat { \psi }$ to get $\tilde { \psi }$ as

$$
\tilde { \psi } ( x ) : = \frac { \hat { \psi } ( x ) - \bar { \theta } ^ { ( i ) } } { \| c \| ^ { 2 } } , \qquad \bar { \theta } ^ { ( i ) } = \sum _ { l } c _ { l } \theta _ { l } ^ { ( i ) } .\tag{64}
$$

With this choice of $c ,$ we have $\bar { \theta } ^ { ( j ) } - \bar { \theta } ^ { ( i ) } = \| c \| ^ { 2 }$ , and $\tilde { \psi }$ maps the centroid of $A _ { i }$ to 0 and that of $A _ { j }$ to 1. Since each $\psi _ { l }$ is an eigenfunction, we have $\begin{array} { r } { \nabla \widehat { \psi } = \sum _ { l = 1 } ^ { m - 1 } c _ { l } \nabla \psi _ { l } } \end{array}$ , and $\begin{array} { r } { L \hat { \psi } = \sum _ { l = 1 } ^ { m - 1 } c _ { l } \lambda _ { l } \psi _ { l } } \end{array}$ Applying L to (64) and using that $\bar { \theta } ^ { ( i ) }$ and $\| c \| ^ { 2 }$ are constants, so that $L { \bar { \theta } } ^ { ( i ) } = 0$ , we have $L \dot { \psi } =$ $\begin{array} { r } { L \hat { \psi } / \| c \| ^ { 2 } = \frac { 1 } { \| c \| ^ { 2 } } \sum _ { l = 1 } ^ { m - 1 } c _ { l } } \end{array}$ <sub>l</sub> λ<sub>l</sub> ψ<sub>l</sub>. Using linearity of $L$ , we obtain $L h = L q - L \tilde { \psi } = - L \tilde { \psi }$ in $D _ { q }$ Together with the boundary conditions $q | _ { \partial A _ { i } } = 0$ and $q | _ { \partial A _ { j } } = 1$ , we have that h solves

$$
L h = - L \tilde { \psi } \mathrm { i n } D _ { q } , h | _ { \partial A _ { i } } = - \tilde { \psi } | _ { \partial A _ { i } } , h | _ { \partial A _ { j } } = 1 - \tilde { \psi } | _ { \partial A _ { j } } .\tag{65}
$$

We next approximate the remainder h in the ansatz $q = \tilde { \psi } + h$ in ${ \mathcal { H } } _ { \kappa }$ . To solve (65) with $\tilde { \psi }$ in (64), let $\{ x _ { l } \} _ { l \in \mathbb { Z } _ { q } } \subset D _ { q }$ with $\mathcal { T } _ { q } = \{ 1 , \ldots , N _ { q } \}$ , and consider solving for a $h \in \mathcal { H } _ { \kappa }$ of the form

$$
\boldsymbol h ( \boldsymbol x ) = \sum _ { l \in \mathcal { T } _ { q } } \boldsymbol { \alpha } _ { l } \kappa ( \boldsymbol x , \boldsymbol x _ { l } ) , \qquad \boldsymbol \alpha = ( \boldsymbol \alpha _ { l } ) _ { l \in \mathcal { T } _ { q } } .\tag{66}
$$

The problem then reduces to solving for α. To this end, we impose (65) it at interior points $\{ x _ { n } \} _ { n \in \mathcal { I } _ { q } } ~ \subset ~ D _ { q }$ where the residual is minimized, and boundary points $\{ x _ { n } \} _ { n \in \mathcal { I } _ { \partial } } \subset \partial A _ { i } \cup \partial A _ { j }$ where the boundary condition is imposed. Specifically, define the matrix $A \in \mathbb { R } ^ { | \mathcal { T } _ { q } | \times | \mathcal { T } _ { q } | }$ and vector $g \in \mathbb { R } ^ { | \mathcal { T } _ { q } | }$ with entries

$$
A _ { n l } : = ( L \kappa ) ( x _ { n } , x _ { l } ) , \qquad [ g ] _ { n } : = ( L \tilde { \psi } ) ( x _ { n } ) = \frac { 1 } { \| c \| ^ { 2 } } \sum _ { l = 1 } ^ { m - 1 } c _ { l } \lambda _ { l } \psi _ { l } ( x _ { n } ) ,\tag{67}
$$

for $n \in \mathcal { I } _ { q }$ . Similarly, define $K _ { \partial } \in \mathbb { R } ^ { | \mathcal { T } _ { \partial } | \times | \mathcal { T } _ { q } | }$ and $b \in \mathbb { R } ^ { | \mathcal { I } _ { \partial } | }$ as

$$
( K _ { \partial } ) _ { n l } : = \kappa ( x _ { n } , x _ { l } ) , \qquad ( b ) _ { n } : = \left\{ \begin{array} { l l } { - \tilde { \psi } ( x _ { n } ) , } & { x _ { n } \in \partial A _ { i } , } \\ { 1 - \tilde { \psi } ( x _ { n } ) , } & { x _ { n } \in \partial A _ { j } , } \end{array} \right.\tag{68}
$$

for $n \in \mathcal { I } _ { \partial }$ . Here, Lκ can be computed using the derivative of the kernel function κ. Specifically, for the Gaussian kernel $\kappa ( x , y ) = \exp \big ( - \| x - y \| ^ { 2 } / 2 \sigma ^ { 2 } \big )$ with bandwidth $\sigma _ { \mathrm { : } }$ , writing $z : = x - y$ , we have

$$
\nabla _ { x } \kappa ( x , y ) = - \frac { z } { \sigma ^ { 2 } } \kappa ( x , y ) , \qquad \partial _ { x _ { i } x _ { j } } \kappa ( x , y ) = \Bigl ( \frac { z _ { i } z _ { j } } { \sigma ^ { 4 } } - \frac { \delta _ { i j } } { \sigma ^ { 2 } } \Bigr ) \kappa ( x , y ) .\tag{69}
$$

Hence, we have

$$
( L \kappa ) ( x , y ) = \left( - \frac { b ( x ) \cdot z } { \sigma ^ { 2 } } + \frac { z ^ { \top } a ( x ) z } { 2 \sigma ^ { 4 } } - \frac { \mathrm { t r } a ( x ) } { 2 \sigma ^ { 2 } } \right) \kappa ( x , y ) .\tag{70}
$$

We next solve for α in (66) by imposing the boundary condition at $\{ x _ { n } \} _ { n \in \mathcal { I } _ { \partial } }$ and minimizing the least-squares sense with Tikhonov regularization as

$$
\begin{array} { r } { \begin{array} { r l } { \underset { \alpha } { \operatorname* { m i n } } } & { { } \left\| \boldsymbol { A } \boldsymbol { \alpha } + \boldsymbol { g } \right\| ^ { 2 } + \lambda _ { q } \boldsymbol { \alpha } ^ { \top } K \boldsymbol { \alpha } , } \\ { \mathrm { s . t . } } & { { } K _ { \partial } \boldsymbol { \alpha } = b , } \end{array} } \end{array}\tag{71}
$$

for $\lambda _ { q } \ \geq \ 0$ and $K _ { l l ^ { \prime } } = \kappa ( x _ { l } , x ^ { l ^ { \prime } } )$ . For such an equality-constrained quadratic program, define a multiplier $\mu \in \mathbb { R } ^ { | \mathcal { T } _ { \partial } | }$ . The Lagrangian is then $\| A \alpha + g \| ^ { 2 } + \lambda _ { q } { \alpha } ^ { \top } K \alpha + 2 \mu ^ { \top } ( K _ { \partial } \alpha - b )$ , and the KKT conditions in $( \alpha , \mu )$ are

$$
\left[ A ^ { \top } A + \lambda _ { q } K \quad K _ { \partial } ^ { \top } \right] \left[ { \alpha } \right] = \left[ { - A ^ { \top } g } \right] .\tag{72}
$$

Solving for $\hat { \alpha }$ via the linear system (72) and plugging this into $q = \tilde { \psi } + h$ , the committor function can be approximated via (19).

## E Details Behind the Computation of $\rho$ from Section 4.2

We start by constructing $\rho _ { 0 }$ . Consider $x \in D ,$ , define $\tau _ { i } : = \operatorname* { i n f } \{ t > 0 : X _ { t } \in A _ { i } \}$ , and $\tau _ { j } : = \operatorname* { i n f } \{ t > 0$ $X _ { t } \in A _ { j } \} = \tau _ { D }$ . Let $R : = \{ \tau _ { j } < \tau _ { i } \}$ be the reactive event, i.e., the trajectory reaches the target set before returning to the source set. By definition of the committor function in Section 4.1, we have

$q ^ { ( i , j ) } ( x ) = P ^ { x } ( R )$ where $P ^ { x }$ denotes the probability of starting from x. Conditioning $\mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \right]$ in (20) on R and $R ^ { c }$ , and using the law of total expectation, we have

$$
\begin{array} { r l } & { \rho ( x ) = \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \right] } \\ & { \qquad = P ^ { x } ( R ) \underbrace { \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \Big \vert R \right] } _ { : = \rho _ { \mathrm { R } } ( x ) } + P ^ { x } ( R ^ { c } ) \underbrace { \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \Big \vert R ^ { c } \right] } _ { : = \rho _ { \mathrm { N R } } ( x ) } } \\ & { \qquad = q ^ { ( i , j ) } ( x ) \rho _ { \mathrm { R } } ( x ) + \left( 1 - q ^ { ( i , j ) } ( x ) \right) \rho _ { \mathrm { N R } } ( x ) , } \end{array}\tag{73}
$$

where $\rho _ { \mathrm { R } } , \rho _ { \mathrm { N R } } \in ( 0 , 1 ]$ since $H _ { 0 } \geq 0$ , with the convention that a term is omitted when its conditioning event has probability zero. Thus $\rho ( x )$ is a convex combination of $\rho _ { \mathrm { R } } ( x )$ and $\rho _ { \mathrm { N R } } ( x )$ with weight being $q ( x )$ . In practice, we use $\hat { q }$ from (19) as an approximation of $q .$

On $R ^ { c }$ , the trajectory reaches $A _ { i }$ at time $\tau _ { i } < \tau _ { j }$ . We therefore split $H _ { 0 } = H _ { 0 } ^ { i } + H _ { 0 } ^ { i j }$ with $\begin{array} { r } { H _ { 0 } ^ { i } : = \int _ { 0 } ^ { \tau _ { i } } ( 1 - q ^ { ( i , j ) } ) \ d t } \end{array}$ dt and $\begin{array} { r } { H _ { 0 } ^ { i j } : = \int _ { \tau _ { i } } ^ { \tau _ { j } } ( 1 - q ^ { ( i , j ) } ) d t } \end{array}$ . Conditioning on the path up to time $\tau _ { i }$ , by the Markov property, we have $\mathbb { E } \big [ e ^ { - \beta H _ { 0 } ^ { i j } } \mid \mathcal { F } _ { \tau _ { i } } \big ] = \rho ( X _ { \tau _ { i } } )$ where $X _ { \tau _ { i } } \in \partial A _ { i }$ . Since $e ^ { - \beta H _ { 0 } } = e ^ { - \beta H _ { 0 } ^ { i } } e ^ { - \beta H _ { 0 } ^ { i j } }$ , we have

$$
\rho _ { \mathrm { N R } } ( x ) = \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } ^ { i } } \rho ( X _ { \tau _ { i } } ) \Big \vert R ^ { c } \right] .\tag{74}
$$

For $\rho ( X _ { \tau _ { i } } )$ , notice that the trajectory lingers inside $A _ { i } .$ and hence $\rho ( X _ { \tau _ { i } } ) \approx \rho | _ { A _ { i } }$ . To compute the latter, let $y \in A _ { i }$ . Since $\boldsymbol { q } ^ { ( i , j ) } \equiv 0$ on $A _ { i }$ and the trajectory spends nearly all of time $[ 0 , \tau _ { j } ]$ inside $A _ { i }$ we have $H _ { 0 } ^ { i } \approx \tau _ { j }$ . Since the system is metastable, we assume the first hitting time is asymptotically exponential with mean $T _ { 0 } = 1 / | \lambda _ { 1 } |$ , where $\lambda _ { 1 }$ is the leading nontrivial eigenvalue of the Koopman generator. In other words„ $\tau _ { j }$ has density $T _ { 0 } ^ { - 1 } e ^ { - t / T _ { 0 } }$ on $t \geq 0$ , and we have

$$
\rho _ { A } \approx \mathbb { E } \left[ e ^ { - \beta \tau _ { j } } \right] = \int _ { 0 } ^ { \infty } { \frac { 1 } { T _ { 0 } } } e ^ { - t / T _ { 0 } } e ^ { - \beta t } d t = { \frac { 1 } { 1 + \beta T _ { 0 } } } = : r .\tag{75}
$$

Hence, we have

$$
\rho ( x ) = q ^ { ( i , j ) } ( x ) \rho _ { \mathrm { R } } ( x ) + \left[ \left( 1 - q ^ { ( i , j ) } ( x ) \right) r \right] \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } ^ { i } } \Big | R ^ { c } \right] .\tag{76}
$$

On $R ,$ the trajectory goes from x to $A _ { j }$ without visiting $A _ { i } ,$ hence $\rho _ { \mathrm { R } } = \mathbb E _ { x } \left[ e ^ { - \beta H _ { 0 } } \big \vert R \right]$ lies in $( 0 , 1 ]$ with $H _ { 0 }$ the duration of reactive crossing. For term $\mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } ^ { i } } \Big | R ^ { c } \right]$ , on $R ^ { c } , H _ { 0 } ^ { i }$ is the time of a non-reactive path falling back into $A _ { i }$ . Therefore, neither term encodes the metastability of the system, and the metastability enters only through $q ^ { ( i , j ) } ( x )$ and $( 1 - q ^ { ( i , j ) } ( x ) ) r$ . We therefore consider the factorization of $\rho$ as

$$
\rho = \left( q ^ { ( i , j ) } ( x ) + \left( 1 - q ^ { ( i , j ) } ( x ) \right) r \right) \times \frac { \rho } { q ^ { ( i , j ) } ( x ) + \left( 1 - q ^ { ( i , j ) } ( x ) \right) r } .\tag{77}
$$

Replacing $\boldsymbol { q } ^ { ( i , j ) }$ by its approximation $\hat { q }$ from (19), we define $\rho _ { 0 } ( x ) : = \hat { q } ( x ) + \big ( 1 - \hat { q } ( x ) \big ) r$ , and the remainder is the $H _ { 0 }$

We next compute w in the ansatz $\hat { \rho } ( x ) : = \rho _ { 0 } ( x ) w ( x )$ . Since $\rho _ { 0 }$ is afine in ${ \hat { q } } ,$ we also have

$$
\begin{array} { r } { \nabla \rho _ { 0 } = \left( 1 - r \right) \nabla \hat { q } , \qquad L \rho _ { 0 } = \left( 1 - r \right) L \hat { q } , } \end{array}\tag{78}
$$

By the product rule, we have

$$
\begin{array} { r } { L \big ( \rho _ { 0 } w \big ) = \rho _ { 0 } L w + w L \rho _ { 0 } + \nabla \rho _ { 0 } ^ { \top } a \nabla w . } \end{array}\tag{79}
$$

Thus $L \rho - f _ { \beta } \rho = 0$ in (11) can be written in terms of w as

$$
\rho _ { 0 } L w + w \left( L \rho _ { 0 } - f _ { \beta } \rho _ { 0 } \right) + \nabla \rho _ { 0 } ^ { \top } a \nabla w = 0 \qquad \mathrm { i n } \ D .\tag{80}
$$

To solve (80), define two matrices in $\mathbb { R } ^ { | I _ { D } | \times N }$ as

$$
( K _ { D } ) _ { l i } = \kappa ( x _ { l } , x _ { i } ) , \qquad ( L K ) _ { l i } = ( L \kappa ) ( x _ { l } , x _ { i } ) ,\tag{81}
$$

for $x _ { l } \in I _ { D }$ and $i = 1 , \ldots , N$ . In addition, define $\Gamma _ { D } \in \mathbb { R } ^ { | I _ { D } | \times N }$ , and $P , Q \in \mathbb { R } ^ { | I _ { D } | \times | I _ { D } | }$ as

$$
\begin{array} { r l } & { ( \Gamma _ { D } ) _ { l i } = \nabla \rho _ { 0 } ( x _ { l } ) ^ { \top } a ( x _ { l } ) \nabla _ { x } \kappa ( x _ { l } , x _ { i } ) , } \\ & { P : = \mathrm { d i a g } \big ( \rho _ { 0 } ( x _ { l } ) \big ) , \quad Q : = \mathrm { d i a g } \big ( L \rho _ { 0 } ( x _ { l } ) - f _ { \beta } ( x _ { l } ) \rho _ { 0 } ( x _ { l } ) \big ) , } \end{array}\tag{82}
$$

where $\nabla _ { x } \kappa$ is available in closed form as in (69), and $\begin{array} { r } { \nabla \widehat { q } = \nabla \tilde { \psi } + \sum _ { l } \alpha _ { l } \nabla _ { x } \kappa ( \cdot , x _ { l } ) } \end{array}$ from (19). Using the chain rule and (21), we also have

$$
\begin{array} { r } { L \rho _ { 0 } = \rho _ { 0 } \Big ( \eta L \hat { q } + \frac { \eta ^ { 2 } } { 2 } \nabla \hat { q } ^ { \top } a \nabla \hat { q } \Big ) . } \end{array}\tag{83}
$$

Hence (81) together with (82), the residual of (80) at the interior points can thus be written as

$$
\left[ P L K + Q K _ { D } + \Gamma _ { D } \right] c = : S _ { D } c , \qquad S _ { D } \in \mathbb { R } ^ { | I _ { D } | \times N } .\tag{84}
$$

Likewise, with $b _ { l } : = e ^ { - \Phi ( x _ { l } ) } / \rho _ { 0 } ( x _ { l } )$ for $x _ { l } \in I _ { \partial D }$ and $( K _ { \partial D } ) _ { l i } : = \kappa ( x _ { l } , x _ { i } )$ , the boundary condition can be written as $K _ { \partial D } c = b$ . We then consider the following regularized optimization problem

$$
\begin{array} { r l } { \underset { c \in \mathbb { R } ^ { N } } { \operatorname* { m i n } } } & { \| S _ { D } c \| ^ { 2 } + \lambda c ^ { \top } K c } \\ { \mathrm { s . t . } } & { K _ { \partial D } c = b , } \end{array}\tag{85}
$$

for $\lambda \geq 0$ , and $K \in \mathbb { R } ^ { N \times N }$ is the Gram matrix, $K _ { i i ^ { \prime } } = \kappa ( x _ { i } , x ^ { i ^ { \prime } } )$ , so that $c ^ { \top } K c = \| w \| _ { \mathcal { H } _ { k } } ^ { 2 }$ . Here, the boundary condition is imposed exactly, while the interior residual is minimized in the least squares sense with Tikhonov regularization. Here, (85) is again an equality-constrained convex quadratic program. We define the Lagrangian $\| S _ { D } c \| ^ { 2 } + \lambda c ^ { \top } K c + 2 \mu ^ { \top } ( K _ { \partial D } c - b )$ , and obtain the KKT condition, which is a linear system in c and the Lagrange multiplier $\mu$

$$
\left( \begin{array} { c c } { S _ { D } ^ { \top } S _ { D } + \lambda K } & { K _ { \partial D } ^ { \top } } \\ { K _ { \partial D } } & { 0 } \end{array} \right) \left( \begin{array} { c } { c } \\ { \mu } \end{array} \right) = \binom { 0 } { b } .\tag{86}
$$

With c determined by (86), applying the chain rule to log $\hat { \rho } = \log \rho _ { 0 } + \log w$ with $\rho _ { 0 }$ in (21), we obtain (22).

## F Reweighting the Generated Paths

The controlled dynamics (9) generate transition paths more eficiently than the reference dynamics, but at the price of sampling from a tilted path measure rather than the reference one. To fix this, first consider starting from some initial condition $x \in D$ . Recall that $P ^ { x }$ is the path measure of (1) started at $x ,$ and ${ \tilde { P } } ^ { * , x }$ is that of the optimally controlled process (9) started at x. By Girsanov’s theorem, $P ^ { x }$ and ${ \tilde { P } } ^ { * , x }$ are equivalent masures on the common path space Ω. Let $\chi \in \Omega$ be a generic element of the path space. By Lemma 4, we have

$$
\frac { d \tilde { P } ^ { * , x } } { d P ^ { x } } = \frac { e ^ { - H ( \chi ) } } { \rho ( x ) } ,\tag{87}
$$

where $\rho ( x ) = \mathbb { E } _ { x } [ e ^ { - H } ]$ , and $\begin{array} { r } { H ( \chi ) : = \int _ { 0 } ^ { \tilde { \tau } _ { D } } f ( \tilde { x } _ { t } ) d t + \Phi ( \tilde { x } _ { \tilde { \tau } _ { D } } ) } \end{array}$ . Hence, the correction from the controlled process (9) to the original process (1) at a fixed start is simply $\rho ( x ) e ^ { H ( \chi ) }$

When the initial conditions are drawn from a probability measure ν on the source set $A _ { i } , \mathrm { e . g . }$ the Boltzmann measure restricted to $A _ { i } .$ consider the mixture

$$
M ( \cdot ) = \int _ { A _ { i } } P ^ { x _ { 0 } } ( \cdot ) \nu ( d x _ { 0 } ) , \quad \tilde { M } ( \cdot ) = \int _ { A _ { i } } \tilde { P } ^ { * , x _ { 0 } } ( \cdot ) \nu ( d x _ { 0 } ) .\tag{88}
$$

Both M and $\tilde { M }$ share the same initial distribution ν. Since every path starts from its own initial condition $x _ { 0 } = \chi ( 0 )$ , we have

$$
\frac { d \tilde { \cal M } } { d { \cal M } } = \frac { d \tilde { \cal P } ^ { * , \chi ( 0 ) } } { d { \cal P } ^ { \chi ( 0 ) } } = \frac { e ^ { - { \cal H } ( \chi ) } } { \rho \big ( \chi ( 0 ) \big ) } .\tag{89}
$$

Hence, to map $\tilde { M }$ back to $M _ { i }$ , each path is weighted by $d M / d \tilde { M }$ by Lemma 4. From (89) and replacing $\rho$ by $\hat { \rho }$ in (21), we can approximate it as

$$
\mathrm { w e i g h t } ( \chi ) = \frac { d M } { d \tilde { M } } = \hat { \rho } \big ( \tilde { x } _ { 0 } \big ) e ^ { H ( \chi ) } , \qquad \tilde { x } _ { 0 } : = \chi ( 0 ) .\tag{90}
$$

Here, $\hat { \rho } ( \tilde { x } _ { 0 } )$ is given by (21).

We further note that as $\beta  \infty$ , the probability that a sampled path spends more than an arbitrary fixed time $\varepsilon > 0$ in nonreactive regions tends to zero. To see this, recall from Example 1 that $H = \beta H _ { 0 }$ . From (87), we have

$$
\begin{array} { r l } { \displaystyle \tilde { P } ^ { * , x } ( H _ { 0 } > \varepsilon ) = \frac { \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \mathbf 1 \{ H _ { 0 } > \varepsilon \} \right] } { \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \right] } \le \frac { e ^ { - \beta \varepsilon } } { \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \right] } } & { } \\ { \displaystyle \le \frac { e ^ { - \beta \varepsilon } } { \mathbb { E } _ { x } \left[ e ^ { - \beta H _ { 0 } } \mathbf 1 \{ H _ { 0 } \le \varepsilon / 2 \} \right] } } & { } \\ { \displaystyle \le \frac { e ^ { - \beta \varepsilon / 2 } } { P ^ { x } ( H _ { 0 } \le \varepsilon / 2 ) } , } \end{array}\tag{91}
$$

for $\varepsilon > 0$ with $P ^ { x } ( H _ { 0 } \leq \varepsilon / 2 ) > 0$ . Here, the first inequality follows from $e ^ { - \beta H _ { 0 } } \leq e ^ { - \beta \varepsilon }$ on $\{ H _ { 0 } > \varepsilon \}$ . Taking $\beta \to \infty$ , we have $\tilde { P } ^ { * , x } ( H _ { 0 } > \varepsilon ) \to 0$

## G Experiment Details and Additional Examples

## G.1 The Double-Well System

For the two-dimensional double-well system, the drift and difusion coeficients in (1) are specialized as $\boldsymbol { b } ( \boldsymbol { x } ) = - \nabla U ( \boldsymbol { x } )$ , with

$$
U ( x , y ) = { \textstyle { \frac { 1 } { 6 } } } { \Big ( } 4 ( 1 - x ^ { 2 } - y ^ { 2 } ) ^ { 2 } + 2 ( x ^ { 2 } - 2 ) ^ { 2 } + { \big ( } ( x + y ) ^ { 2 } - 1 { \big ) } ^ { 2 } + { \big ( } ( x - y ) ^ { 2 } - 1 { \big ) } ^ { 2 } - 2 { \Big ) } ,\tag{92}
$$

and $\sigma ( x ) = \sigma _ { 0 } I$ with $\sigma _ { 0 } = \sqrt { 2 k _ { B } T _ { \mathrm { e m p } } } \approx 0 . 4 5 5$ at $T _ { \mathrm { e m p } } = 1 2 0 0 \mathrm { K }$ . Its potential has two minima at $\left( \pm \sqrt { 5 } / 2 , 0 \right)$ and two channels around $( 0 , \pm 1 )$ . Following Algorithm 1, we compute the Koopman eigenfunctions and extract the metastable cores $A _ { \mathrm { l e f t } }$ and $A _ { \mathrm { r i g h t } }$ as described in Section 4.1. Specifically, $A _ { \mathrm { l e f t } }$ and $A _ { \mathrm { r i g h t } }$ are obtained by retaining those points whose spectral embedding lies within radius $\varepsilon = 0 . 5$ of its cluster centroid. We sample 1000 uncontrolled and 1000 controlled paths, all initialized at the minimum of $A _ { \mathrm { l e f t } }$ . A path is a hit at its first entrance to $A _ { \mathrm { r i g h t } }$ , and is truncated at the numerical censor $T _ { \mathrm { m a x } }$ otherwise. We use a step size of $\Delta t = 0 . 0 1$ , and $T _ { \mathrm { m a x } } = 1 0$ equivalently, a budget of $T _ { \mathrm { m a x } } / \Delta t = 1 0 0 0$ integration steps, and $\beta = 1$ . We adopt two metrics to evaluate performance: the target hit percentage (THP), which measures the proportion of trajectories that successfully reach the target before $T _ { \mathrm { m a x } } ;$ and the energy of the transition state (ETS), which measures the maximum potential energy along each transition path. As shown in Figure 1 and Table 1, the controller raises the THP from 0% to 99.8%, and the controlled ensemble retains both channels rather than collapsing onto one.

Table 1: Target hit percentage (THP) and energy of the transition state (ETS) for the 2-channel double well. Uncontrolled and controlled rows are measured in our setup; TPS-DPS values are as reported by Seong et al. (2025).
<table><tr><td>Method</td><td>THP (↑) ETS (↓) % kJmol−1</td></tr><tr><td>Uncontrolled (1200K)</td><td>0.0</td></tr><tr><td>Uncontrolled (2400K)</td><td>2.7  $1 . 7 3 \pm 0 . 4 2$ </td></tr><tr><td>Uncontrolled (3600K)</td><td>18.8  $2 . 0 7 \pm 0 . 4 6$ </td></tr><tr><td>Uncontrolled (4800K)</td><td>42.1  $2 . 4 9 \pm 0 . 6 6$ </td></tr><tr><td>TPS-DPS (1200K)</td><td>99.9  $1 . 3 8 \pm 0 . 1 6$ </td></tr><tr><td>Proposed method (1200K)</td><td>99.8  $1 . 4 0 \pm 0 . 1 1$ </td></tr></table>

## G.2 Muller-Brown Potential

The Müller-Brown potential (Müller and Brown, 1979) is described by

$$
U ( x , y ) = \frac { 1 } { 1 0 } \sum _ { k = 1 } ^ { 4 } A _ { k } \exp \left( a _ { k } ( x - x _ { k } ) ^ { 2 } + b _ { k } ( x - x k ) ( y - y _ { k } ) + c _ { k } ( y - y _ { k } ) ^ { 2 } \right) ,\tag{93}
$$

with parameters specified as

$$
{ \left( \begin{array} { l l l l l l } { A _ { 1 } } & { a _ { 1 } } & { b _ { 1 } } & { c _ { 1 } } & { x _ { 1 } } & { y _ { 1 } } \\ { A _ { 2 } } & { a _ { 2 } } & { b _ { 2 } } & { c _ { 2 } } & { x _ { 2 } } & { y _ { 2 } } \\ { A _ { 3 } } & { a _ { 3 } } & { b _ { 3 } } & { c _ { 3 } } & { x _ { 3 } } & { y _ { 3 } } \\ { A _ { 4 } } & { a _ { 4 } } & { b _ { 4 } } & { c _ { 4 } } & { x _ { 4 } } & { y _ { 4 } } \end{array} \right) } = { \left( \begin{array} { l l l l l l } { - 2 0 0 } & { - 1 } & { 0 } & { - 1 0 } & { 1 } & { 0 } \\ { - 1 0 0 } & { - 1 } & { 0 } & { - 1 0 } & { 0 } & { 0 . 5 } \\ { - 1 7 0 } & { - 6 . 5 } & { 1 1 } & { - 6 . 5 } & { - 0 . 5 } & { 1 . 5 } \\ { 1 5 } & { 0 . 7 } & { 0 . 6 } & { 0 . 7 } & { - 1 } & { - 1 } \end{array} \right) }\tag{94}
$$

We consider a particle with $m = 1 2 \mathrm { a . m . u }$ and $\gamma = \mathrm { { 1 p s } ^ { - 1 } }$ . As shown in Figure 2, the dynamics has one deep well at $\left( - 0 . 5 5 8 , 1 . 4 4 2 \right)$ , a medium-deep well at (0.623, 0.028), and a shallow well sitting between the two at (−0.050, 0.467). The goal is to transition from the deep well to the medium well. The source and target cores are defined with radius $\varepsilon = 0 . 2$ . We choose a stepsize of $\Delta t = 0 . 0 0 2 \mathrm { p s }$ and $T _ { \mathrm { m a x } } = 1 0 \mathrm { p s }$ . We use a Gaussian kernel with bandwidth 0.35. At temperature $T _ { \mathrm { e m p } } = 2 0 0 K$ the controller raises the THP from 1.8% for the uncontrolled process to 99.0%.

## G.3 Alanine Dipeptide

We simulate alanine dipeptide (ACE–ALA–NME) in vacuum with AMBER99SB-ILDN, hydrogenbond constraints, and the VVVR integrator at 300 K, friction $1 0 ^ { - 3 } \mathrm { f s ^ { - 1 } }$ , and timestep 1 fs. Trajectories start from the energy-minimized C5 structure. Following (Seong et al., 2025), the target is defined as $B = \{ { \bf R } \ | \ | \xi ( { \bf R } ) - \xi ( { \bf R } _ { B } ) | | < 0 . 7 5 \}$ (in radians), where $\xi = \left( \phi , \psi \right)$ are the backbone dihedral angles and $\mathbf { R } _ { B }$ is the energy-minimized $C 7 _ { \mathrm { a x } }$ structure. A trajectory stops when it enters B or reaches $T _ { \mathrm { m a x } } = 1 \mathrm { p s }$

Table 2: Müller-Brown benchmark at $T = 1 5 0 \mathrm { K }$ and 200 K.
<table><tr><td>Method</td><td>THP (↑) ETS (↓) %  $\mathrm { k J m o l ^ { - 1 } }$ </td></tr><tr><td>Uncontrolled (150K) 0.1</td><td>12.4</td></tr><tr><td>Uncontrolled (200K) TPS-DPS (150K)</td><td>1.8  $1 4 . 0 \pm 1 . 3$ </td></tr><tr><td>TPS-DPS (200K)</td><td>100  $1 5 . 9 \pm 3 . 5$ </td></tr><tr><td>Proposed method  $( 1 5 0 \mathrm { K } , \beta = 1 )$ </td><td>100  $1 5 . 7 \pm 1 . 9$ </td></tr><tr><td></td><td>95.1  $1 4 . 1 \pm 2 . 1$ </td></tr><tr><td>Proposed method (150K, β = 2) 95.9</td><td> $1 4 . 3 \pm 2 . 1$ </td></tr><tr><td>Proposed method  $( 2 0 0 \mathrm { K } , \beta = 1 )$ </td><td>98.4  $1 4 . 5 \pm 2 . 1$ </td></tr><tr><td>Proposed method  $( 2 0 0 \mathrm { K } , \beta = 2 )$ </td><td>99.0  $1 4 . 5 \pm 2 . 1$ </td></tr></table>

The controller $\hat { \boldsymbol { u } } ~ = ~ \nabla \log \hat { \rho }$ is evaluated on the Kabsch-aligned Cartesian configuration and applied as an external force $F _ { \mathrm { b i a s } } = s { \hat { u } }$ in OpenMM, with $s = 3 0$ kJ mol<sup>−1</sup>nm<sup>−1</sup> and ∥uˆ∥ capped at 75 kJ mol $^ { - 1 } \mathrm { n m } ^ { - 1 }$ per atom. We use $\beta = 1$ , Gaussian bandwidth 1, and $\varepsilon = 0 . 0 1$ , with 80 kernel centers and 400 interior and 48 boundary collocation points. OpenMM integrates the underdamped Langevin dynamics

$$
d x _ { i } = \frac { p _ { i } } { m _ { i } } d t , \quad d p _ { i } = F _ { i } ( x ) d t - \gamma _ { i } p _ { i } d t + \sqrt { 2 m _ { i } \gamma _ { i } k _ { B } T } d W _ { i } ,
$$

where i indexes the atom. In this experiment, we solve the boundary value problem (11) with the isotropic generator $\begin{array} { r } { L = F \cdot \nabla + \frac { \sigma _ { 0 } ^ { 2 } } { 2 } \Delta } \end{array}$ where F is the force and $\sigma _ { 0 } = 1$ as an approximation of the simulated dynamics that ignores the per-atom mobility $1 / ( m _ { i } \gamma )$

The proposed pipeline as summarized in Algorithm 1 runs once: Koopman eigenfunctions are constructed via a generalized eigenproblem in terms of Gram matrices (see, e.g., (Hou et al., 2023, Appendix M) for details), followed by solving for $\hat { q }$ and uˆ. In contrast, the baselines in (Seong et al., 2025) repeatedly simulate controlled trajectories and update the neural network, so they pay the sampling cost per rollout. Table 3 compares the cost of obtaining the bias in the drift term. All timings are measured on a single NVIDIA A100 GPU.

## G.4 Additional Experiment: The Four-Well Dynamics

Consider the two-dimensional four-well potential described by

$$
U ( x , y ) = 5 ( ( x ^ { 2 } - 1 ) ^ { 2 } + ( y ^ { 2 } - 1 ) ^ { 2 } ) ,\tag{95}
$$

which has four symmetric local minima at $( \pm 1 , \pm 1 )$ . The source core is centered at pair $( - 1 , - 1 )$ and the target core is centered at (1, 1), with radius $\varepsilon = 0 . 3$ . We use mass $m = 1 2 { \mathrm { ~ a . m . u } }$ , friction $\gamma = \mathrm { { 1 p s } ^ { - 1 } }$ , stepsize $\Delta t = 0 . 0 0 2 \mathrm { p s }$ , and $T _ { \mathrm { m a x } } = 1 0 \mathrm { p s }$ . Results for various settings are reported in 4, and the sampled paths at 200 K are shown in Figure 6.

![](images/ff7b65905562fcec67bfe812ede991c32312795cf4724fd91f88bbb73237298e.jpg)

![](images/ca21ddc008f0d07cdd435377eef400ea3e8b6b2e72b0db3b227a60cef48b68a9.jpg)  
Figure 4: Alanine dipeptide on the Ramachandran plain of backbone dihedral angles (ϕ, ψ). (Left) The first nontrivial Koopman eigenfunction $\psi _ { 1 }$ which separates the C5 basin from $C 7 _ { \mathrm { a x } }$ basin. (Right) The committor q from C5 to $C 7 _ { \mathrm { a x } }$

![](images/ffe879d48657e2ee8870dfef889d66a1acebf5e13328b450b6483f4980d8964e.jpg)  
(a) Uncontrolled

![](images/f47d4a32490c898976300a4c8cc581640f4f51d4f11b7d875861256a9f3e1bd3.jpg)  
(b) TPS-DPS

![](images/347a9a2b2d36d7eb80d693933c17902b58c3b408ff9f6bf4dba076bc13ab2650.jpg)  
(c) Proposed method  
Figure 5: Sampled transition paths for alanine dipeptide in vacuum, same as Figure 3 in Section 5. Paths start in the C5 basin (upper left) and end in the $C 7 _ { \mathrm { a x } }$ basin (red star). (a) Uncontrolled paths. (b) TPS-DPS (Seong et al., 2025): THP = 100%, obtained by training a neural-network bias with simulation in the loop (1000 rollouts, ${ \sim } 1 0 ^ { 6 }$ gradient updates). (c) Proposed method: THP = 93%, with a closed-form controller.

Table 3: Cost of obtaining the controller on alanine dipeptide on a single NVIDIA A100 GPU, excluding the subsequent path generation, which is common to all methods.
<table><tr><td colspan="5">Proposed method</td></tr><tr><td>Stage</td><td colspan="3">Time (s) Total (s)</td></tr><tr><td>Sampling</td><td colspan="3">Direct evaluation of (3) 180</td></tr><tr><td>Koopman eigenpairs</td><td colspan="3">73.8</td></tr><tr><td>Committor  $\hat { q }$  Controller û</td><td colspan="3">56.2 51.0</td></tr><tr><td colspan="6">Training-based baselines</td></tr><tr><td>Method</td><td>Rollouts X</td><td>(Sampling (s) + Training (s))</td><td>二 Total (h)</td></tr><tr><td>TPS-DPS (F)</td><td>1000 X</td><td>(60.6 + 14)</td><td>二 20.7</td></tr><tr><td>PIPS (F)</td><td>15000 X</td><td>(60.6 +</td><td>275</td></tr></table>

![](images/de446e829b8890e69caa43f86d4b60ff9e636cd8b040ab913b22e58328cdcfa0.jpg)  
(a) Uncontrolled

![](images/61f287d8cfa96da4e32197012082b2bb5d707f6fbab4f40a7d3de988b07ac2c3.jpg)  
(b) TPS-DPS

![](images/411d85d5f8518a954f7db1d91f9052636184af4f7f2c00ee547e743948dbfd44.jpg)  
(c) Proposed method  
Figure 6: Four-well dynamics at $T = 2 0 0 \mathrm { K }$ . We aim to sample transitions from the source core at $( - 1 , - 1 )$ (green star) to the target core at (1, 1) (red star). (a) Uncontrolled paths: $\mathrm { T H P } ( T _ { \mathrm { m a x } } ) =$ 0.9%. (b) TPS-DPS (Seong et al., 2025): $\mathrm { T H P } ( T _ { \mathrm { m a x } } ) = 8 1 . 6 \%$ . (c) Proposed method with $\beta = 2 \colon$ $\mathrm { T H P } ( T _ { \mathrm { m a x } } ) = 9 4 . 7 \%$

Table 4: Four-well benchmark at T = 200 K and 300 K.
<table><tr><td>Method</td><td>THP (↑) %</td><td>ETS (↓)  $\mathrm { k J m o l ^ { - 1 } }$ </td></tr><tr><td>Uncontrolled (200K) Uncontrolled (300K) TPS-DPS (200K)</td><td>0.9 8.8 81.6</td><td> $1 0 . 6 \pm 3 . 2$   $1 2 . 8 \pm 3 . 6$   $8 . 6 \pm . 9$ </td></tr><tr><td>TPS-DPS (300K)</td><td>87.1</td><td> $1 0 . 2 p m . 6 $ </td></tr><tr><td>Proposed method  $( 2 0 0 \mathrm { K } , \beta = 1 )$ </td><td>73.2</td><td> $9 . 5 \pm 1 . 8$ </td></tr><tr><td>Proposed method  $( 2 0 0 \mathrm { K } , \beta = 2 )$ </td><td>94.7</td><td> $9 . 0 \pm 1 . 8$ </td></tr><tr><td>Proposed method (300K, β = 1)</td><td>91.8</td><td> $1 1 . 2 \pm 3 . 1$ </td></tr><tr><td>Proposed method  $( 3 0 0 \mathrm { K } , \beta = 2 )$ </td><td>100.0</td><td> $1 0 . 4 \pm 2 . 5$ </td></tr></table>