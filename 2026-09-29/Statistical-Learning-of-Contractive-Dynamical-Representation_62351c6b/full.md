# Statistical Learning of Contractive Dynamical Representations for Composite Adaptive Control

Min Kim<sup>1,∗</sup>, Jose Leonardo Brenes ´ <sup>1,∗</sup>, Fred Hadaegh<sup>1,2</sup>, Soon-Jo Chung<sup>1,2</sup>

Abstract— We present a representation-learning framework for composite adaptive tracking control under dynamically coupled disturbances. The framework connects classical disturbance-accommodating control (DAC) to recent last-layer adaptive disturbance-rejection methods. Specifically, we introduce a statistically principled hard expectation–maximization (hard-EM) procedure, with a Kalman smoother in the hard E-step, to identify dynamical representations of disturbance whose latent evolution is uniformly contractive. The learned representation evolves a latent disturbance-excitation state from measured plant features and control inputs and decodes that state into the time-varying disturbance acting on the nominal plant, thereby extending prior “fixed-decay” last-layer adaptive methods to a learned, predictive DAC-style formulation. Combined with Bayesian filtering of the learned latent state, this representation yields a composite adaptive tracking controller with predictive capability and provable exponential convergence to a bounded neighborhood. We validate our approach experimentally on a slippery ground vehicle carrying a liquidsloshing tank and a pendulum load, and we further assess its robustness on a system of coupled Duffing oscillators. Across both settings, the method achieves accurate disturbance prediction and improved overall tracking performance relative to fixed-decay representation-learning ablations, LTI disturbanceaccommodating baselines, and model-based PD baselines.

## I. INTRODUCTION

There is a substantial body of literature on representation learning for last-layer adaptive control and disturbance rejection [1]–[3]; however, many existing applications focus on external disturbances such as aerodynamic effects or terraininduced slip. In such settings, disturbances are captured by a latent variable that is treated as exogenous; its value shifts as the “environment” changes, sometimes with an assumed decay rate, which corresponds to the Ornstein– Uhlenbeck process. In contrast, when the unknown disturbance is dynamically coupled with the controlled plant, such as liquid sloshing in a vehicle’s tank, it is natural to view the latent variable as an augmented dynamical state that evolves according to its own dynamics, in the spirit of disturbance-accommodating control (DAC). In such cases, we expect adaptive controllers with predictive capability to have a comparative advantage over counterparts that do not incorporate prediction. This motivates the learning of dynamical representations tailored to adaptive control; in this paper, we develop a representation that captures the coupled disturbance source’s excitation mechanism and its influence on the main system.

![](images/966abf5e438bff65eed81d8638f78b0bb0a346b7e92c38719ef24b4a96bef920.jpg)  
Fig. 1. Proposed adaptive control architecture and experimental platform. Left: block diagram of the controller. A latent state is estimated online via a Kalman–Bucy-style filter from the disturbance proxy y, using a latentstate evolution model parameterized by (A, b). The basis functions (Φ, ψ) map latent-relevant features ϕ to an adaptive disturbance-rejection term that augments a nominal tracking controller. The quantities (A, b, Φ, ψ) are parameterized by neural networks and learned using a statistically principled training procedure. Right: tracked mobile robot used for experimental validation.

The contributions of this paper are threefold: (i) we introduce a statistically principled hard expectation–maximization training framework, with Kalman-smoother-based latent inference (Fig. 2), for learning contractive dynamical latent representations of disturbance from data; (ii) we design a neural architecture for the dynamical representation that enforces uniform contraction, is compatible with Bayesian filtering, and yields predictive disturbance estimates with stability guarantees when used in a composite adaptive controller; and (iii) we demonstrate the method on hardware and in simulation, showing accurate disturbance prediction and improved tracking over fixed-decay representation-learning ablations and LTI-based DAC. Taken together, these results show that statistically principled learning of contractive latent disturbance dynamics can bridge representation learning and composite adaptive control for systems with dynamically coupled disturbances.

We next position our paper relative to prior work. From the perspective of classical control, our approach is closely related to disturbance-accommodating control (DAC), which represents disturbance using an augmented disturbance state with its own dynamics and estimates this augmented state online for compensation [4]. DAC motivates treating the disturbance as an estimable and predictable dynamical quantity rather than as a memoryless exogenous input. Our perspective is also connected to the robust servomechanism problem [5], where disturbance signals are generated by an exosystem and robust output regulation requires embedding an internal model [6] of the disturbance in the closed loop. From this standpoint, our method can be interpreted as learning a datadriven, contractive analogue of the disturbance generator in the spirit of identification-based internal-model regulation [7]. In contrast to [7], which adapts an internal model by linear regression, we identify an affine-in-latent model via statistical learning and exploit Kalman-based estimation.

![](images/dac070b53372b4f208afd767ca971efb69b58959dcdb770452afdd440bda394f.jpg)  
Fig. 2. Statistically principled training via hard-EM with exact latent inference. The procedure alternates between a hard E-step and an M-step: given current parameters $( \theta , R _ { d } , Q _ { d } )$ , the hard E-step infers the latent trajectory ${ \bf a } _ { 0 : L - 1 }$ for each data window, and the M-step updates the noise covariances $( R _ { d } , Q _ { d } )$ from residual statistics and updates the network parameters θ.

Learning-based control has been demonstrated on physical robot platforms, including quadrotors under wind disturbance [1], [8], drone racing [9], learning-based MPC for quadrotors [10] and autonomous car racing [11], rapid motor adaptation for bipedal locomotion [12], and robust perceptive locomotion for quadrupedal robots in natural environments [13]. These hardware demonstrations highlight the importance of controllers that remain reliable under modeling errors and disturbances, which are often induced or amplified by robotic maneuvers. Motivated by this need, recent learning based MPC approaches improve tracking under model mismatch by incorporating learned disturbance or residuals into the MPC framework [14], [15]. Also, [16] learns a Koopman-based disturbance dynamics model and uses the corresponding disturbance observer for compensation. In contrast, we learn a latent disturbance dynamics model by a Kalman-smoother-based latent trajectory inference and hard expectation–maximization (EM). We note that the learning strategy presented in this paper is analogous to previous smoother-based EM methods for system identification [17], [18], although these papers do not involve neural networks.

In this paper, we focus on disturbance representations that are uniformly contractive [19], [20] and are affine in the latent variable (see (3)). The contraction property of the dynamical representation is required for insensitivity to the unknown initial latent state (Proposition 1), and is physically plausible for dissipative disturbance dynamics. On the other hand, the affine-in-latent form enables closed-form Bayesian estimation. This motivates us to use a hard-EM procedure to train the neural networks (Fig. 2 and Section III); during training, the latent trajectory is repeatedly estimated by Kalman smoothing. Moreover, using the Kalman–Bucy filter for the latent state, we synthesize a composite adaptive con troller [1], [2], [21], whose exponential tracking guarantees are analyzed in Theorems 3 and 4. This analysis relies on a covariance bound for uniformly contractive systems (Lemma 2).

In Section IV, we evaluate the learned dynamical representation and the resulting adaptive controller on two different scenarios. First, we study a ground vehicle (“GVR-Bot”) equipped with a liquid-sloshing tank, a pendulum load, and an unknown, inaccessible internal PID controller; vehicle maneuvers excite both the fluid sloshing and the pendulum motion. Then, we consider a system of coupled Duffing oscillators as a numerical testbed, where the robustness of the proposed tracking controller is assessed under increasingly adverse Gaussian-noise conditions. These examples show not only that the method can learn predictive maneuver-todisturbance mappings, but also that the learned mappings can be readily and effectively incorporated into composite adaptive controllers.

## II. THE CONTROL METHOD AND GUARANTEES

The main controller equations are given by the theorems in Section II-B.

## A. System Description and the Dynamical Representation

We consider systems with additive disturbances of the form:

$$
M ( q ) \ddot { q } + C ( q , \dot { q } ) \dot { q } + g ( q ) = u + d\tag{1}
$$

where the coefficient functions are continuously differentiable and $\dot { M } - 2 C$ is skew-symmetric. We also consider the simpler first-order system:

$$
\dot { v } = f ( v ) + u + d ,\tag{2}
$$

with continuously differentiable f. We aim to learn disturbance estimators of the form (3) for use in composite adaptive control [22]:

$$
\dot { \mathbf { a } } = A ( \phi ( t ) , u ( t ) ) \mathbf { a } + b ( \phi ( t ) , u ( t ) ) ,\tag{3a}
$$

$$
\hat { d } = \Phi ( \phi ( t ) ) \mathbf { a } + \psi ( \phi ( t ) ) ,\tag{3b}
$$

where $A : \mathbb { R } ^ { d _ { \phi } + d _ { u } }  \mathbb { R } ^ { d _ { a } \times d _ { a } } , b : \mathbb { R } ^ { d _ { \phi } + d _ { u } }  \mathbb { R } ^ { d _ { a } }$ , Φ : $\mathbb { R } ^ { d _ { \phi } }  \mathbb { R } ^ { d _ { u } \times d _ { a } } , \psi : \mathbb { R } ^ { d _ { \phi } }  \mathbb { R } ^ { d _ { u } }$ are continuous functions to be learned.

Intuitively, $\textbf { a } \in \mathbb { R } ^ { d _ { a } }$ summarizes disturbance-relevant hidden dynamics; A and b define its evolution, while Φ and ψ decode it into a disturbance estimate. We emphasize that $\phi : \mathbb { R } _ { \geq 0 }  \mathbb { R } ^ { d _ { \phi } }$ is a continuous “feature” signal which is determined causally. In most cases ϕ is simply the state: $\boldsymbol { \phi } = \left( \boldsymbol { q } , \dot { \boldsymbol { q } } \right)$ , but it could additionally contain measurements from on-board sensors, e.g., a camera. Here, a statistically explains the disturbance d and evolves according to the dynamic model (3a). Unless otherwise stated, the control signals $u : \mathbb { R } _ { \geq 0 }  \mathbb { R } ^ { d _ { u } }$ are continuous and ∥ · ∥, when applied to a matrix, denotes the spectral norm.

Note that in (3) the estimate <sup>ˆ</sup>d(t) does not depend on the instantaneous control input u(t). This restriction may be undesirable in some settings. An alternative is to linearly approximate the map $u ( t ) \mapsto d ( t )$ ; to do so, set

$$
\hat { d } _ { k } ( t ) = u _ { \mathrm { a u g } } ( t ) ^ { \top } \big ( \Phi _ { k } \mathbf { a } + \psi _ { k } \big )\tag{4}
$$

for each disturbance coordinate $1 \leq k \leq \dim ( d )$ , where the augmented input $u _ { \mathrm { a u g } } ( t ) : = [ u ( t ) ^ { \top } , 1 ] ^ { \top }$ . This parameterization is convenient for controller synthesis: since $u ( t )$ enters (4) $a f f u n e l y$ , the relation $u ( t ) + \hat { d } ( t ) ~ = ~ u _ { \mathrm { d e s i r e d } } ( t )$ can be easily solved for $u ( t ) ;$ ; in practice, one solves the Tikhonov-regularized normal equations centered at $u _ { \mathrm { d e s i r e d } } ( t )$ for robustness. Finally, we also note that a prototypical class of examples for (3) consists of multi-component mechanical systems coupled through force elements.

Hereafter, we assume access to an online noisy estimate $y ( t ) = d ( t ) + \epsilon ( t )$ , where $\epsilon ( t )$ is a continuous, additive error. If y(t) were exact and the control loop were sufficiently fast relative to the disturbance timescale, direct cancellation would largely suffice; in practice, however, $y ( t )$ is often obtained by numerically differentiating the measured $q$ or $v ,$ which amplifies sensor noise, and actuator delays further limit the feasibility of instantaneous cancellation for rapidly varying disturbances.

The proposed learned dynamical representation instead produces systematically refined estimates of the current disturbance $d ( t )$ via the Kalman–Bucy filter, by fusing the full measurement history $y ( \cdot )$ and, implicitly, all measurements available in the training set. When the disturbance source is dynamically coupled to the plant, learning an explicit latent dynamics is a natural design choice; these learned dynamics then serve as predictive process models in the Kalman–Bucy filter. Moreover, when constructing disturbance labels for training, one may leverage higher-quality off-board sensing and non-causal signal processing to obtain more reliable numerical derivatives than would be available on-board in real time. The refined estimates from the filter are then used for disturbance compensation, thereby yielding composite adaptive tracking controllers with provable convergence guarantees (see Theorems 3 and 4).

Before proving the main Theorems 3 and 4 which give the controllers ((9) or (12) with (8)), we first formalize the intuition that uniformly contractive linear systems are insensitive to initial conditions which will be unknown in practice.

Proposition 1 (Exponential insensitivity to initial condition): Suppose ${ \begin{array} { l } { { \frac { 1 } { 2 } } \left( A + A ^ { \top } \right) \leq - \lambda I } \end{array} }$ and $\| \Phi \| \leq C$ pointwise for some constants $\lambda > 0$ and $C > 0$ . For a fixed pair $\phi$ and $u ,$ any two trajectories ${ \bf a } _ { 1 }$ and $\mathbf { a } _ { 2 }$ of (3) satisfy

$$
\| \hat { d } _ { 1 } ( t ) - \hat { d } _ { 2 } ( t ) \| \le C e ^ { - \lambda t } \| \mathbf { a } _ { 1 } ( 0 ) - \mathbf { a } _ { 2 } ( 0 ) \|\tag{5}
$$

for all $t \geq 0$

Proof: We have $\begin{array} { r l r } { \frac { d } { d t } \left( \left\| \mathbf { a } _ { 1 } ( t ) - \mathbf { a } _ { 2 } ( t ) \right\| ^ { 2 } \right) } & { { } } & { = } \end{array}$ $2 ( \mathbf { a } _ { 1 } ( t ) - \mathbf { a } _ { 2 } ( t ) ) ^ { \top } ( \dot { \mathbf { a } } _ { 1 } ( t ) - \dot { \mathbf { a } } _ { 2 } ( t ) ) = ( \mathbf { a } _ { 1 } ( t ) - \mathbf { a } _ { 2 } ( t ) ) ^ { \top } ( A +$ $A ^ { \top } ) \left( { \bf a } _ { 1 } ( t ) - { \bf a } _ { 2 } ( t ) \right) \ \leq \ - 2 \lambda \left\| { \bf a } _ { 1 } ( t ) - { \bf a } _ { 2 } ( t ) \right\| ^ { 2 }$ . Therefore, $\| \mathbf { a } _ { 1 } ( t ) - \mathbf { a } _ { 2 } ( t ) \| \leq e ^ { - \lambda t } \left\| \mathbf { a } _ { 1 } ( 0 ) - \mathbf { a } _ { 2 } ( 0 ) \right\|$ . Now, the claimed result follows from ${ \hat { d } } = \Phi \mathbf { a } + \psi$ ■

Moreover, uniformly contractive linear systems lead to bounded covariance estimates when the Kalman–Bucy filter is used. Specifically, we consider the following covariance evolution equation:

$$
\dot { P } = A ( t ) P + P A ( t ) ^ { \top } + Q - P \Phi ( t ) ^ { \top } R ^ { - 1 } \Phi ( t ) P\tag{6}
$$

with a fixed initial condition $P ( 0 ) = P _ { 0 } \succ 0$ and $Q \succeq 0$ $R \succ 0$ . Here, $A ( t ) : = A ( \phi ( t ) , u ( t ) )$ and $\Phi ( t ) : = \Phi ( \phi ( t ) )$

Lemma 2: Suppose ${ \begin{array} { l } { { \frac { 1 } { 2 } } \left( A + A ^ { \top } \right) } \end{array} } \preceq - \lambda I$ for some constant $\lambda > 0$ . Then the unique solution of (6) is positive definite and satisfies

$$
\lambda _ { \operatorname* { m a x } } \bigl ( P ( t ) \bigr ) \leq e ^ { - 2 \lambda t } \lambda _ { \operatorname* { m a x } } ( P _ { 0 } ) + \frac { \lambda _ { \operatorname* { m a x } } ( Q ) } { 2 \lambda } \bigl ( 1 - e ^ { - 2 \lambda t } \bigr )\tag{7}
$$

for all $t \geq 0$ . Moreover, if Φ is bounded, $Q \succ 0 .$ , and ${ \frac { 1 } { 2 } } \left( A + \right.$ $A ^ { \top } ) \succeq - \mu I$ , then $\lambda _ { \operatorname* { m i n } } \left( P ( t ) \right) \geq p .$ for some positive $p _ { * }$ given by an explicit formula.

Proof: The positive definiteness claim follows from the comparison theorem for Hermitian RDE [23, Theorem 4.1.4] (compare to the same equation but with $Q \ = \ 0 )$ , and existence of $P$ on $[ 0 , \infty )$ follows from Theorem 4.1.6.

For the upper bound, consider the linear equation $\dot { \tilde { P } } =$ $A \tilde { P } + \tilde { P } A ^ { \top } + Q$ with the initial condition $\tilde { P } ( 0 ) ~ = ~ P _ { 0 }$ Theorem 4.1.4 gives $P ( t ) ~ \preceq ~ \tilde { P } ( t )$ for all $t \ \geq \ 0 .$ Since $\begin{array} { r } { \tilde { P } ( t ) = \Phi _ { A } ( t , 0 ) P _ { 0 } \Phi _ { A } ( t , 0 ) ^ { \top } + \int _ { 0 } ^ { t } \Phi _ { A } ( t , \tau ) Q \Phi _ { A } ( t , \tau ) ^ { \top } d \tau } \end{array}$ for all $t \ \geq \ 0 .$ , and $\| \Phi _ { A } ( t , t _ { 0 } ) \| ~ \le ~ e ^ { - \lambda ( t - t _ { 0 } ) }$ for all $t \geq$ $t _ { 0 } \geq 0$ , we get $\lambda _ { \operatorname* { m a x } } ( \tilde { P } ( t ) ) = \| \tilde { P } ( t ) \| \leq e ^ { - 2 \lambda t } \lambda _ { \operatorname* { m a x } } ( P _ { 0 } ) +$ $\frac { \lambda _ { \operatorname* { m a x } } ( Q ) } { \mathfrak { D } \lambda } ( 1 ~ - ~ e ^ { - 2 \lambda t } )$ . The claim follows from the Weyl monotonicity principle.

For the lower bound, we have $P ( t ) \succeq p _ { * } I$ with $p _ { * } =$ $\begin{array} { r l r } {  { \frac { - \mu + \sqrt { \mu ^ { 2 } + s _ { u } \lambda _ { \operatorname* { m i n } } ( Q ) } } { s _ { * } } \wedge \lambda _ { \operatorname* { m i n } } ( P _ { 0 } ) } } \end{array}$ , where $\begin{array} { r } { s _ { u } : = \frac { \overline { { \Phi } } ^ { 2 } } { \lambda _ { \operatorname* { m i n } } ( R ) } } \end{array}$ and $\overline { { \Phi } } ~ > ~ 0$ is an upper bound for Φ; this can be proved by comparing the original equation to a related one: ${ \dot { P } } _ { c } = A P _ { c } +$ $P _ { c } \dot { A ^ { \top } } + \bar { ( s } _ { u } p _ { * } ^ { 2 } I - p _ { * } A - \bar { p } _ { * } A ^ { \top } ) - P _ { c } ( s _ { u } I ) P _ { c } , P _ { c } ( 0 ) = p _ { * } I$ and noting that $Q \succeq s _ { u } p _ { * } ^ { 2 } I - p _ { * } A - p _ { * } A ^ { \top } , \Phi ^ { \top } R ^ { - 1 } \Phi \preceq s _ { u } I .$

## B. Composite Adaptive Tracking Controller and Analysis

While Lemma 2 is of independent interest, its main role in this paper is to establish exponential tracking results for various composite adaptive control strategies based on the Kalman–Bucy filter.

First, we consider a tracking problem for (1) with a twice continuously differentiable reference trajectory $q _ { d }$ . We use Kalman–Bucy-inspired composite adaptation equations of the form:

$$
\begin{array} { r l } & { \dot { \hat { \mathbf { a } } } = A ( \phi , u ) \hat { \mathbf { a } } + b ( \phi , u ) } \\ & { \quad + P \Phi ( \phi ) ^ { \top } R ^ { - 1 } \left( y - \Phi ( \phi ) \hat { \mathbf { a } } - \psi ( \phi ) \right) + P \Phi ( \phi ) ^ { \top } s , } \end{array}\tag{8a}
$$

$$
\dot { P } = A P + P A ^ { \top } + Q - P \Phi ^ { \top } R ^ { - 1 } \Phi P\tag{8b}
$$

where $s : = \dot { \tilde { q } } + \Lambda \tilde { q } , \tilde { q } : = q - q _ { d } ,$ , and $P ( 0 ) = P _ { 0 } , Q , R , \Lambda \succ 0 .$ For the main theorems, we assume $d ( t ) = \Phi ( \phi ( t ) ) \mathbf { a } ( t ) +$ $\psi ( \phi ( t ) ) + r ( t )$ , so that the disturbance d is realized by continuous signals ${ \bf a } ( t )$ and $r ( t )$ with a being differentiable. Note that (8) are the standard Kalman–Bucy equations but with $P \Phi ^ { \top } s$ , an additional composite adaptation term. Note that when the Kalman covariance $P$ becomes larger, the tracking error term $P \Phi ( \phi ) ^ { \top } s$ also becomes larger (along with the innovation term).

Remark 1: In (8), $A = A ( \phi , u ) , b = b ( \phi , u ) , \Phi = \Phi ( \phi )$ $\psi = \psi ( \phi )$ are dependent on the measured features and/or control, and therefore are implicitly time-varying. Previous last-layer adaptive disturbance rejection methods [1], [2] may be interpreted as the case of $A = - \lambda I , b = 0 , \psi = 0$ with hyperparameter $\lambda > 0$ , although their learning procedures differ from the procedure used here.

Remark 2: The Kalman-smoother approach to representation learning (Section III-C) provides data-driven estimates of $Q$ and R that serve as principled starting points for subsequent gain tuning.

Now we present the main theorems demonstrating exponential tracking convergence via the adaptation dynamics $( 8 ) ;$ here, exponential convergence to a bounded ball means that there exist constants $c _ { i } > 0$ such that each solution on $\mathbb { R } _ { > 0 }$ of the closed-loop system satisfies $\| q ( t ) - q _ { d } ( t ) \| \leq c _ { 1 } e ^ { - c _ { 2 } \overline { { t } } } + c _ { 3 }$ (or $\| v ( t ) - v _ { d } ( t ) \| \leq c _ { 1 } e ^ { - c _ { 2 } t } + c _ { 3 }$ for (2)), where $c _ { 2 }$ and c<sub>3</sub> are independent of $\hat { \mathbf { a } } ( 0 ) , \mathbf { a } ( 0 )$ , and $q ( 0 ) , \dot { q } ( 0 ) \ ( \mathrm { o r } \ v ( 0 )$ for (2)). As expected, the theorems will depend on the bounds on $A \mathbf { a } + b - \dot { \mathbf { a } }$ and $r ,$ i.e., the quality of the dynamical representation (3).

Theorem 3 (Exp. Tracking: 2nd-order mechanical systems): Consider the system (1), (8) with the feedback controller

$$
\begin{array} { r } { u = M \dot { v } _ { r } + C v _ { r } + g - K s - \Phi \hat { \mathbf a } - \psi , } \end{array}\tag{9}
$$

where $K \succ 0$ and $v _ { r } : = \dot { q } _ { d } - \Lambda \tilde { q } . \mathrm { I f } - \mu I \preceq \frac { 1 } { 2 } \left( A + A ^ { \top } \right) \preceq$ $- \lambda I , \underline { { m } } I \preceq M \preceq$ mI for some positive constants $\mu , \lambda ,$ m, m, and if $( A \mathbf { a } + b - \dot { \mathbf { a } } )$ , Φ, r, ϵ are bounded by some constants independent of ${ \hat { \mathbf { a } } } ( 0 ) , { \mathbf { a } } ( 0 ) , { \mathbf { \nabla } } q ( 0 ) , { \dot { q } } ( 0 )$ , then the feedback controller (9) achieves exponential convergence to a bounded ball.

Proof: First, note that $\dot { s } = \ddot { q } - \dot { v } _ { r }$ and therefore $M \dot { s } =$ $- ( K + C ) s - \Phi \tilde { \mathbf { a } } + r .$ , where a˜ $\mathbf { \hat { a } } = \hat { \mathbf { a } } - \mathbf { a }$ . Moreover, we have $P ^ { - 1 } \dot { \tilde { \mathbf { a } } } = \Phi ^ { \top } s + ( P ^ { - 1 } A - \Phi ^ { \top } R ^ { - 1 } \Phi ) \tilde { \mathbf { a } } + P ^ { - 1 } ( A \mathbf { a } + b - \dot { \mathbf { a } } ) +$ $\Phi ^ { \top } R ^ { - 1 } ( r + \epsilon ) . \mathrm { D e f i n i n g } \ V : = s ^ { \top } M s + \tilde { \mathbf { a } } ^ { \top } P ^ { - 1 } \tilde { \mathbf { a } } ,$ we have $\dot { V } = - 2 s ^ { \top } K \dot { s } - \tilde { \mathbf { a } } ^ { \top } \left( \bar { P } ^ { - 1 } Q P ^ { - 1 } + \Phi ^ { \top } R ^ { - 1 } \Phi \right) \tilde { \mathbf { a } } + 2 s ^ { \top } r +$ $2 \tilde { \mathbf { a } } ^ { \top } P ^ { - 1 } ( A \mathbf { a } + b - \dot { \mathbf { a } } ) + 2 \tilde { \mathbf { a } } ^ { \top } \Phi ^ { \top } R ^ { - 1 } ( r + \epsilon )$

By Lemma 2, for a sufficiently small $\alpha > 0$ , we have $K \succeq \alpha M$ and $Q \succeq 2 \alpha P$ for all $t \geq 0$ , which implies $- 2 K \preceq$ $- 2 \alpha M$ and $- P ^ { - 1 } Q P ^ { - 1 } - \Phi ^ { \top } R ^ { - 1 } \Phi \preceq - 2 \bar { \alpha } P ^ { - 1 }$ . Therefore, we conclude that $\dot { V } \leq - 2 \alpha V + 2 s ^ { \top } r + 2 \tilde { \mathbf { a } } ^ { \top } P ^ { - 1 } ( A \mathbf { a } +$ $b - \dot { \mathbf { a } } ) + 2 \tilde { \mathbf { a } } ^ { \top } \Phi ^ { \top } R ^ { - 1 } ( r + \epsilon )$

Applying Cauchy–Schwarz gives

$$
\begin{array} { c } { { \dot { V } \leq - 2 \alpha V + 2 \sqrt { \frac { V } { \lambda _ { \operatorname* { m a x } } ( P ) ^ { - 1 } \wedge \underline { { { m } } } } } } } \\ { { \qquad \times \left\| \left[ P ^ { - 1 } ( A \mathbf { a } + b - \dot { \mathbf { a } } ) + \Phi ^ { \top } R ^ { - 1 } ( r + \epsilon ) \right] \right\| . } } \end{array}\tag{10}
$$

Lemma 2 with the assumptions of this theorem shows that the norm in (10) is bounded by a constant, say $\overline { { d } } \geq 0 .$ . For each small $c > 0 .$ , define $W _ { c } = { \sqrt { V + c } }$ . We have $\dot { W } _ { c } \leq - \alpha W _ { c } +$ $\alpha \sqrt { c } + \frac { \overline { { d } } } { c ^ { * } }$ where $c ^ { * } : = \Big ( \underline { { m } } \wedge ( \| P _ { 0 } \| + \| Q \| / ( 2 \lambda ) ) ^ { - 1 } \Big ) ^ { 1 / 2 } ,$

Differentiating $G _ { c } ( t ) : = e ^ { \alpha t } \left( W _ { c } - \sqrt { c } - \overline { { d } } / ( c ^ { * } \alpha ) \right)$ and passing to the limit $c  0 +$ gives:

$$
\sqrt { \underline { { m } } } \| s ( t ) \| \leq \sqrt { V ( t ) } \leq e ^ { - \alpha t } \left( \sqrt { V ( 0 ) } - \frac { \overline { { d } } } { c ^ { * } \alpha } \right) + \frac { \overline { { d } } } { c ^ { * } \alpha } .\tag{11}
$$

By solving the linear system $\dot { \tilde { q } } = - \Lambda \tilde { q } + s$ in ${ \tilde { q } } ,$ using $\| \exp ( - t \Lambda ) \| = \exp \left( - t \lambda _ { \operatorname* { m i n } } ( \Lambda ) \right)$ for all $t \geq 0$ , and choosing $\alpha < \lambda _ { \mathrm { m i n } } ( \Lambda )$ if necessary, we conclude that $\| \tilde { q } \| \leq c _ { 1 } e ^ { - c _ { 2 } t } +$ $c _ { 3 }$ for some $c _ { i } > 0$ with $c _ { 2 } , \ : c _ { 3 }$ being independent of $\hat { \mathbf { a } } ( 0 )$ ${ \mathbf { a } ( 0 ) , q ( 0 ) }$ , and q˙(0).

We also present a simpler analogue of the previous theorem in the context of (2).

Theorem 4 (Exp. Tracking: 1st-order systems): Consider the system (2), (8) with the feedback controller

$$
u = \dot { v } _ { d } - f ( v ) - K s - \Phi \hat { \mathbf { a } } - \psi ,\tag{12}
$$

where $K \succ 0$ and now $s : = v - v _ { d }$ with a continuously differentiable reference velocity $v _ { d } .$ ${ \mathrm { I f ~ } } - \mu I \preceq { \frac { 1 } { 2 } } \left( A + A ^ { \top } \right) \preceq$ $- \lambda I$ for some positive constants $\mu , \lambda ,$ and if $( { \dot { A } } \mathbf { a } + b - { \dot { \mathbf { a } } } )$ Φ, $r , \ \epsilon$ are bounded by some constants independent of $\widehat { \mathbf { a } } ( 0 ) , \mathbf { a } ( 0 ) , v ( 0 )$ , then the feedback controller (12) achieves exponential convergence to a bounded ball.

Proof: We have $\dot { s } = - K s - \Phi \tilde { \mathbf { a } } + r$ where $\tilde { \textbf { a } } : =$ $\hat { \mathbf { a } } - \mathbf { a }$ . Considering $V : = s ^ { \top } s + \tilde { \mathbf { a } } ^ { \top } P ^ { - 1 } \tilde { \mathbf { a } } ,$ we again have $\dot { V } = - 2 s ^ { \top } K s - \tilde { \mathbf { a } } ^ { \top } \left( P ^ { - 1 } Q P ^ { - 1 } + \Phi ^ { \top } R ^ { - 1 } \Phi \right) \tilde { \mathbf { a } } + 2 s ^ { \top } r +$ 2a˜ $^ { \top } P ^ { - 1 } ( A \mathbf { a } + b - \dot { \mathbf { a } } ) + 2 \tilde { \mathbf { a } } ^ { \top } \Phi ^ { \top } R ^ { - 1 } ( r + \epsilon )$ . The rest of the proof is similar to the proof of Theorem 3 once one notes that we have $K \succeq \alpha I$ and $Q \succeq 2 \alpha P$ for all small $\alpha > 0 .$ which leads to

$$
\| s ( t ) \| \leq { \frac { \overline { d } } { c ^ { * } \alpha } } + \left( { \sqrt { V ( 0 ) } } - { \frac { \overline { d } } { c ^ { * } \alpha } } \right) e ^ { - \alpha t } ,\tag{13}
$$

where $c ^ { * } = \Big ( 1 \wedge ( \| P _ { 0 } \| + \| Q \| / ( 2 \lambda ) ) ^ { - 1 } \Big ) ^ { 1 / 2 }$ and $\overline { { d } }$ is defined as in the proof of Theorem 3.

## III. NEURAL NETWORK ARCHITECTURE AND ITS TRAINING

In this section, we present a statistically principled training procedure for a uniformly contractive disturbance predictor whose latent dynamics are governed by feature-dependent matrix coefficients.

## A. Contractive Network Parametrization

All maps are parameterized by GELU MLPs, with spectral normalization applied to the linear layers. To ensure contraction of the latent dynamics (3a), we parameterize $A ( \phi , u ) =$ $S ( \phi , u ) + K ( \phi , u )$ with $S ( \phi , u ) ~ = ~ - ( \beta + \mu ( \phi , u ) ) I ~ - ~$ $L ( \phi , u ) L ( \phi , u ) ^ { \top }$ and $\begin{array} { r } { K ( \phi , u ) = \frac { 1 } { 2 } \left( M ( \phi , u ) - M ( \phi , u ) ^ { \top } \right) } \end{array}$ with $\beta \mathrm { ~  ~ { ~ > ~ } ~ } 0$ fixed. We note that this contractive-byconstruction parameterization is inspired by [24]. The scalar $\mu ( \phi , u ) \geq 0$ and the matrices $L ( \phi , u ) , M ( \phi , u )$ ) are produced by MLPs and bounded element-wise via $\sigma ( \cdot )$ and tanh(·), respectively; ${ \cal L } ( \phi , u )$ is lower-triangular and $M ( \phi , u )$ is strictly lower-triangular. Since K is skew-symmetric, it does not affect the symmetric part of A, and we have the following uniform contraction condition:

$$
\big ( A ( \phi , u ) + A ( \phi , u ) ^ { \top } \big ) / 2 = S ( \phi , u ) \preceq - \beta I ,\tag{14}
$$

therefore the latent dynamics are contracting. $b ( \phi , u ) , \Phi ( \phi )$ and ψ(ϕ) are parameterized by MLPs; Φ is bounded via tanh(·). One may optionally enforce anchoring constraints such as $b ( 0 , 0 ) = 0$ or $\psi ( 0 ) = 0$ by subtracting the value at the origin.

Algorithm 1 Alternating training (hard-EM)   
Require: initial $( \theta , R _ { d } , Q _ { d } )$ , choice of $\lambda _ { a } ,$ , data windows   
1: while stopping criterion is not met do   
2: sample mini-batch $\boldsymbol { B }$   
3: for all $w \in B$ do   
4: instantiate the model for window w from θ   
5: apply Kalman smoothing to infer $a _ { 0 : L - 1 }$   
6: end for   
7: update $R _ { d }$ (from output residuals; sparse/EMA)   
8: update $Q _ { d }$ (from latent residuals; sparse/EMA)<sup>∗</sup>   
9: update θ by AdamW with the inferred latents fixed   
10: end while   
<sup>∗</sup>optional

## B. Training via Hard Expectation–Maximization

We train the architecture from data windows $\{ ( \phi _ { k } , u _ { k } , y _ { k } ) \} _ { k = 0 } ^ { L - 1 }$ sampled at time step $\Delta t ;$ windows are extracted from trajectories with a fixed stride and may overlap.

A key challenge in fitting (3) from data windows is that the latent initial condition ${ \bf a } _ { 0 } = { \bf a } ( 0 )$ is unobserved, windowspecific, and has no direct physical meaning. As a result, ${ \bf a } _ { 0 }$ is a missing value that must be inferred jointly with the shared parameters, which can create an identifiability ambiguity between ${ \bf a } _ { 0 }$ and θ. To regularize this inference problem in a principled way, we place a zero-mean Gaussian prior on the initial latent state, $\mathbf { a } _ { 0 } \sim \mathcal { N } ( 0 , \lambda _ { a } ^ { - 1 } I )$ , which induces the quadratic penalty $\lambda _ { a } \| { \mathbf { a } } _ { 0 } \| ^ { 2 }$ in the training objective. Given parameters θ, we roll out (3a) by RK2 to obtain ${ \bf { a } } _ { k } ,$ , form predictions ${ \hat { d } } _ { k } = \Phi _ { \theta } ( \phi _ { k } ) \mathbf { a } _ { k } + \psi _ { \theta } ( \phi _ { k } )$ , and assume a Gaussian observation model $y _ { k } \mid \mathbf { a } _ { 0 } , \theta \sim \mathcal { N } ( \hat { d } _ { k } , R _ { d } )$ with $R _ { d } \ \succ \ 0 .$ This yields a Gaussian negative log-likelihood (NLL) loss

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { { \bf { a } } _ { 0 } , \boldsymbol { \theta } , { \cal R } _ { d } } } } \ \sum _ { k = 0 } ^ { L - 1 } { \Big ( ( y _ { k } - \hat { d } _ { k } ( { \bf { a } } _ { 0 } , \boldsymbol { \theta } ) ) ^ { \top } R _ { d } ^ { - 1 } ( y _ { k } - \hat { d } _ { k } ( { \bf { a } } _ { 0 } , \boldsymbol { \theta } ) ) }   \\ & { } & { + \log { | { \cal R } _ { d } | } \Big ) + \lambda _ { a } \| { \bf { a } } _ { 0 } \| ^ { 2 } , \quad } \end{array}\tag{15}
$$

implicitly averaged over a mini-batch of windows. We opti mize (15) by an alternating scheme, which can be interpreted as a (generalized) hard-EM procedure [25]: (hard E-step) fix θ and $R _ { d }$ and perform exact minimization with respect to ${ \bf a } _ { 0 }$ to obtain a MAP estimate for each window; (M-step) Fix the inferred $\mathbf { a } _ { 0 }$ and update $R _ { d }$ from mini-batch statistics; then use AdamW to update θ. The quadratic prior term $\lambda _ { a } \| { \mathbf { a } } _ { 0 } \| ^ { 2 }$ regularizes the inferred initial condition and could be interpreted as mitigating the identifiability ambiguity.

For fixed θ, the latent rollout $\left\{ { \bf { a } } _ { k } \right\}$ depends affinely on ${ \bf a } _ { 0 } ;$ consequently, $\hat { d } _ { k }$ is affine in ${ \bf a } _ { 0 }$ and the Gaussian NLL induces a positive definite quadratic objective in ${ \bf a } _ { 0 }$ . Thus, the ${ \bf a } _ { 0 }$ block can be optimized exactly, and this optimization can be viewed as a special case of Kalman smoothing; see Section III-C. In view of (15), it is natural to use $P _ { 0 } = \lambda _ { a } ^ { - 1 } I$ $\hat { \mathbf { a } } ( 0 ) = 0 , R = \left( \Delta t \right) R _ { d }$ and a small $Q = q I \succ 0$ in (8).

## C. Process Noise Modeling and the Kalman Smoother

The training scheme in the previous subsection corresponds to a deterministic latent rollout within each window: once $( \theta , \mathbf { a } _ { 0 } )$ are fixed, the entire latent trajectory $\{ \mathbf { a } _ { k } \} _ { k = 0 } ^ { L - 1 }$ is fixed. We optionally add discrete-time process noise $Q _ { d } \succ$ 0 by treating the entire latent trajectory as missing data; Section III-B corresponds to the degenerate special case of $Q _ { d } = 0$ . Concretely, let $f _ { \boldsymbol { \theta } , k - 1 } \big ( \mathbf { a } _ { k - 1 } \big )$ denote the one-step RK2 update of (3a) using $\left( \phi _ { k - 1 } , u _ { k - 1 } \right)$ (from time $k - 1$ to time k). Since (3a) is affine in a, $f _ { \theta , k - 1 } ( \mathbf { a } _ { k - 1 } )$ is affine in $\mathbf { a } _ { k - 1 }$ . We replace the deterministic update with

$$
{ \bf a } _ { k } \mid { \bf a } _ { k - 1 } , \theta \sim \mathcal { N } \big ( f _ { \boldsymbol { \theta } , k - 1 } ( { \bf a } _ { k - 1 } ) , Q _ { d } \big ) , \qquad k = 1 , \ldots , L - 1 ,\tag{16}
$$

while keeping the same observation model for $y _ { k }$ and the same prior $\mathbf { a } _ { 0 } \sim \mathcal { N } ( 0 , \lambda _ { a } ^ { - 1 } I )$ . Compared to (15), we replace $\hat { d } _ { k } ( { \bf a } _ { 0 } , \theta )$ by $\hat { d } _ { k } ( { \bf a } _ { k } , \theta ) : = \ \Phi _ { \theta } ( \phi _ { k } ) { \bf a } _ { k } + \psi _ { \theta } ( \phi _ { k } )$ , and the window NLL loss acquires the additional process-noise term $\begin{array} { r } { ( L - 1 ) \log | Q _ { d } | + \sum _ { k = 1 } ^ { L - 1 } \left( { \bf a } _ { k } - f _ { \theta , k - 1 } \big ( { \bf a } _ { k - 1 } \big ) \right) ^ { \top } Q _ { d } ^ { - 1 } \big ( { \bf a } _ { k } - } \end{array}$ $f _ { \boldsymbol { \theta } , k - 1 } \big ( \mathbf { a } _ { k - 1 } \big ) \Big )$

For fixed $( \theta , R _ { d } , Q _ { d } )$ , minimization over the latent trajectory $\mathbf { a } _ { 0 : L - 1 }$ (the hard E-step) is exactly solvable via a Kalman smoother [26], yielding an inferred latent trajectory for each window. The Kalman smoother is implemented by the RTS formulas. In the M-step, we fix this inferred trajectory and update $R _ { d }$ and optionally $Q _ { d }$ from mini-batch statistics, and then θ by AdamW (see Algorithm 1).

The identified discrete-time covariances lead to the continuous-time gains in (8): $Q \approx Q _ { d } / \Delta t$ and $R \approx ( \Delta t ) R _ { d } .$ as noted in Remark 2. In our implementations, we used diagonal $R _ { d }$ and isotropic $Q _ { d } = q _ { d } I$ for simplicity.

## IV. EXPERIMENTAL RESULTS

Implementing the controllers (9) and (12) requires discretizing (8). We use a Joseph-form Kalman filter, with forward Euler discretization for the $P \Phi ^ { \top }$ s term. For both examples, we used $d _ { a } \ = \ 3$ . We compare our controller with (i) a model-based PD control loop, (ii) the fixed-decay ablations inspired by [1], [2], and (iii) a DAC controller in which an LTI system is identified by [27, Sections 4.3.1, 4.4.2], a variant of the N4SID algorithm [28]. We denote the last controller as N4SIDv-DAC. The fixed-decay ablations, denoted FixedDecay, were trained by the Kalman-smootherbased hard-EM procedure restricted to the function class $A = - \lambda I , \ : b \equiv 0 \ :$ , and $\psi \equiv 0$ . For the adaptive controllers, datasets were split into training, validation, and test sets in a 70:15:15 ratio. For the N4SID variant, the number of block rows i in the past and future Hankel matrices, following the notation of [27], was selected to minimize the validationset MSE. The LTI model was then re-estimated using the combined training and validation sets.

The LTI disturbance augmentations are of the form

$$
\dot { \bf a } = A { \bf a } + B _ { 1 } \phi + B _ { 2 } u ,\tag{17a}
$$

$$
\hat { d } = C { \bf a } + D _ { 1 } \phi + D _ { 2 } u ,\tag{17b}
$$

with $d \approx { \hat { d } } .$ For example, substituting (17b) into (18) yields v˙ ≈ $A _ { v } v + ( B + D _ { 2 } ) u + D _ { 1 } \phi +$ Ca. Therefore, considering (12), the disturbance-compensating command is obtained by solving the equation $( B + D _ { 2 } ) u + D _ { 1 } \phi + C \hat { \bf a } = \dot { v } _ { d } - A _ { v } v - K s$ in u. Although the LTI model is presented in continuoustime notation, system identification was performed in discrete time.

## A. A Slippery Tracked Vehicle with a Partially Filled Liquid Tank and a Swinging Pendulum

The presented architecture was evaluated on a tracked ground vehicle (“GVR-Bot”) carrying a pendulum and a liquid tank filled with water to approximately 30% capacity. The experiment also provides a test of generalization to an unseen intermediate fill level because the 30% level was absent from the training data. The choice of a liquid-carrying platform was partly motivated by the growing interest in onorbit refueling [29]. To emulate low-traction, slippery terrain, the tracks were wrapped with low-friction tape. Water-tank level sensor measurements $w ~ \in ~ \mathbb { R }$ were sampled at 18 Hz. The platform had an NVIDIA Jetson AGX Orin as a companion computer, and used visual–inertial odometry (VIO) for localization. The control loop ran at 50 Hz.

The vehicle’s velocity is modeled by the following firstorder lag [2]:

$$
\dot { v } = A _ { v } v + B u + d ,\tag{18}
$$

where $\boldsymbol { v } \ = \ [ v _ { x } , \omega _ { z } ]$ denotes the vehicle’s forward velocity and yaw rate, $\boldsymbol { u } = [ v _ { x , \mathrm { c m d } } , \omega _ { z , \mathrm { c m d } } ]$ denotes the commanded values, and $B = - A _ { v } = \mathrm { d i a g } ( 1 / 0 . 2 , 1 / 0 . 1 5 )$ , determined by system identification on hardware. The vehicle’s internal PID controller is inaccessible, and only the velocity commands are available, which motivates the model (18). The liquid sloshing and pendulum motion are expected to contribute to time-varying changes in the vehicle’s traction/slip behavior and, consequently, in the observed command-to-velocity response. Such unmodeled effects are lumped into d in (18).

a) Data Collection and Training: Data were collected for about 11 minutes for each water-tank level (≈ 60%, 0%) in an open testing area, with the vehicle driven at various speeds and in various directions without following a prescribed trajectory. The vehicle was driven through the full operational range of speeds and angular rates, with maneuvers such as abrupt starts and stops or smooth accelerations. We trained the neural network (≈ 4.9k trainable parameters) using $( v , w , u , y )$ data with $\Delta t ~ = ~ 0 . 0 1$ . Here, y is the estimated d under the zero-order hold assumption, and we used ${ \phi } = ( { v } , { w } )$ . For both the fixed-decay ablation and the proposed method, the neural networks were trained for 100 epochs using a learning rate of 0.001 and a weight decay of 0.01. The training took less than 3 minutes on an Apple M4 Pro laptop.

b) Predictions for Disturbance d: The predictive capability of the architecture was evaluated by rolling out the dynamics in (3a) and (4) using the input, state, and waterlevel measurements from the test set. Figure 3 illustrates the predicted and measured disturbances on two test-set trajectories.

![](images/8a6a0c7b3124fd12247516c56357f30f6b37bd4abce5269572eed666b51e6582.jpg)

![](images/82864393e19ea627d45051f18469792c6a35e026dc492c9a6103b2a1297b1d1a.jpg)

![](images/41bdabad9fe1da09accd865d7cbc49025fd8c7c1959c682f79726a6f09298963.jpg)

![](images/9b42f341fd05e50f0f16900f0a588afcca71e0d9d4b0f8be32e670edcdd995db.jpg)  
Fig. 3. (Tracked vehicle) Test-set rollouts for the predicted disturbance $\hat { d } \in \mathbb { R } ^ { 2 }$ as functions of time, on two different test-set trajectories. Black: disturbance proxy $y ( t ) ;$ red: rollout prediction d<sup>ˆ</sup>(t). The proxy y is visibly noisier. The average prediction MSE was 3.53; it was 3.49 after discarding the initial half second for each trajectory as a burn-in period (vertical dashed line).

c) Tracking Controller Test: We compare the proposed controller with three other controllers: our FixedDecay ablation, N4SIDv-DAC, and the model-based PD loop used in [2, Eqs. 14–18] which contains nonlinearities. The vehicle’s commanded trajectory consisted of two consecutive double lane changes, which excited the water and pendulum dynamics while inducing track slip.

The model-based PD baseline already provides strong overall tracking, particularly in the forward-velocity channel. The benefits of incorporating disturbance predictors are most apparent in yaw-rate regulation. As shown in Table I, the proposed controller achieves the lowest angular-rate and velocity RMS errors and ties N4SIDv-DAC for the lowest position RMS error. The competitive angular-rate and position tracking performance of N4SIDv-DAC suggests that an LTI augmentation captures a significant part of the model mismatch over the tested regime. In contrast, the fixed-decay ablation performed less favorably in this experiment.

<table><tr><td>Metric</td><td>NPD</td><td>N4SIDv-DAC</td><td>FixedDecay</td><td>Ours</td></tr><tr><td>Ang. rate error RMS [rad/s]</td><td>0.449</td><td>0.414</td><td>0.468</td><td>0.407</td></tr><tr><td>Vel. error RMS [m/s]</td><td>0.151</td><td>0.160</td><td>0.156</td><td>0.149</td></tr><tr><td>Pos. error RMS [m]</td><td>0.085</td><td>0.078</td><td>0.081</td><td>0.078</td></tr></table>

TABLE I  
TRACKED-VEHICLE RMS TRACKING ERRORS.  
NPD: nonlinear PD loop; N4SIDv-DAC: LTI-based  
disturbance-accommodating control; FixedDecay: fixed-decay ablation.

## B. Coupled Duffing Oscillators: a Robustness Sweep

We consider a pair of Duffing oscillators coupled by a nonlinear spring:

$$
m _ { 1 } \ddot { x } _ { 1 } = - k _ { 1 } x _ { 1 } - \alpha _ { 1 } x _ { 1 } ^ { 3 } + u + ( d - c _ { 1 } \dot { x } _ { 1 } ) ,\tag{19a}
$$

$$
m _ { 2 } \ddot { x } _ { 2 } = - k _ { 2 } x _ { 2 } - \alpha _ { 2 } x _ { 2 } ^ { 3 } - c _ { 2 } \dot { x } _ { 2 } - d ,\tag{19b}
$$

$$
d = k _ { c } ( x _ { 2 } - x _ { 1 } ) + \alpha _ { c } ( x _ { 2 } - x _ { 1 } ) ^ { 3 } ,\tag{19c}
$$

![](images/2630b024cec576bf5ead7f77300e1169cd82c42f6a897c53ba8cebcde63101aa.jpg)

![](images/fe45a3f8a193c5719092d3ff55db29fc3227bef629fbbc924f731f7b36a76bd4.jpg)  
Fig. 4. (Tracked vehicle) Tracking runs under four controllers.

where we only observe $x _ { 1 } , \dot { x } _ { 1 }$ , u and are unaware of the existence of the coupled $x _ { 2 }$ -system. We train the neural networks with a synthetic dataset of $( x _ { 1 } , \dot { x } _ { 1 } , u , y )$ $( \Delta t \ =$ 0.01). Here, the samples of $x _ { 1 }$ and ${ \dot { x } } _ { 1 }$ are corrupted by zero-mean Gaussian noise with standard deviation 0.005; y is computed using forward differences of the sampled ${ \dot { x } } _ { 1 }$ The parameters used for the nonlinear oscillators were:

$$
m _ { 1 } = 1 . 0 , \qquad m _ { 2 } = 0 . 7 ,\tag{20a}
$$

$$
c _ { 1 } = 0 . 3 0 , \qquad c _ { 2 } = 0 . 2 5 ,\tag{20b}
$$

$$
k _ { 1 } = 1 . 0 , \qquad k _ { 2 } = 0 . 8 ,
$$

$$
k _ { c } = 2 . 0 ,\tag{20c}
$$

$$
\alpha _ { 1 } = 1 . 7 , \qquad \alpha _ { 2 } = 0 . 2 , \qquad \alpha _ { c } = 5 . 0 .\tag{20d}
$$

Our empirical evaluation considers two complementary aspects: (a) we show that (3) with $\phi = ( x _ { 1 } , \dot { x } _ { 1 } )$ predicts d (or more precisely, its proxy y) accurately on the test set, and (b) we compare the controller (9) with the baseline model-based PD $( u = m _ { 1 } \dot { v } _ { r } + k _ { 1 } x _ { 1 } + \alpha _ { 1 } x _ { 1 } ^ { 3 } - K s + c _ { 1 } \dot { x } _ { 1 } )$ , N4SIDv-DAC, and the fixed-decay representation ablation.

a) Predictions for Disturbance d: We rolled out (3) starting from ${ \bf a } _ { 0 } = 0$ to get the estimates $\hat { d } ,$ and compared them to the test-set labels $y \colon$ see Fig. 5. Note that the initial transients from the fixed initial condition ${ \bf a } _ { 0 } ~ = ~ 0$ decay as time passes, as expected from the contraction property (Proposition 1). This predictive capability arises from generalizing prior “fixed-decay” representation-learning approaches to a more flexible learned dynamical model.

b) Tracking Controller Test: To illustrate the benefit of predictive capability, we made y available at only 1 Hz. We also injected different levels of Gaussian noise (its standard deviation denoted by σ) into y to test for robustness (Fig. 6). We used $\Lambda = 2 . 0$ and $K = 1 2 . 0$ for all four controllers.

In this experiment, the acceleration ${ \ddot { x } } _ { 1 }$ was estimated by taking backward finite differences of the observed ${ \dot { x } } _ { 1 } .$ and then filtered by an exponential moving average. The disturbance proxy y was then calculated as $m _ { 1 } \ddot { x } _ { 1 } + c _ { 1 } \dot { x } _ { 1 }$ + $k _ { 1 } x _ { 1 } + \alpha _ { 1 } x _ { 1 } ^ { 3 } - u ;$ after this, it was corrupted by noise level σ. Note that $\sigma = 0$ already results in a noisy proxy y because the measurements for $x _ { 1 }$ and ${ \dot { x } } _ { 1 }$ are noisy. Each reference

Fig. 5. (Coupled nonlinear oscillator) Test-set rollouts from Eq. (3) for the predicted disturbance as functions of time. Black: disturbance proxy $y ( t ) ;$ ; red: rollout prediction $\hat { d } ( t )$ . The proxy y is visibly noisier, while d<sup>ˆ</sup> is smooth. Mean squared error is 2.54 overall and 0.88 after discarding the first second of each trajectory as a burn-in period (vertical dashed line).  
![](images/7c23bed042fe73b46ea3963cffdb00004ac9abccd7198cc07d52285c508f21fb.jpg)  
Fig. 6. (Coupled nonlinear oscillator) Robustness sweep for tracking control under additional Gaussian noise injected into the sparsely provided disturbance proxy y (averaged over 30 reference trajectories, with each trajectory being 20 seconds long). Mean tracking MSE is plotted versus the added noise level $\sigma ,$ where $\sigma = 0$ denotes the baseline noise setting. The proposed dynamical-representation controller performs the best. As σ increases, the neural adaptive controllers degrade because the disturbance proxy becomes increasingly corrupted. The simpler N4SIDv-DAC baseline resulted in a flat graph because the fitted discrete-time $A , \ B _ { 1 }$ , and $B _ { 2 }$ satisfied $\lVert A \rVert \approx 0 . { \overset { \cdot } { 0 } } \mathbf { \dot { 9 } } ^ { \cdot } \ll 1 , \ \lVert B _ { 1 } \rVert \approx 0 . 0 8 \ll 1 , \ \lVert B _ { 2 } \rVert$ ≈ 0.005 ≪ 1 and hence the disturbance compensation mostly came from d<sup>ˆ</sup> ≈ $D _ { 1 } \phi + D _ { 2 } \boldsymbol { u }$ . trajectory was generated as a sum of randomized sinusoidal signals.

Figure 6 summarizes the robustness sweep for the proposed dynamical-representation controller, the fixed-decay representation ablation, the N4SIDv-DAC controller, and the model-based PD baseline. In the nominal-noise case $( \sigma = 0 )$ , the proposed controller achieves the lowest average tracking MSE; in particular, our controller reduced tracking MSE by 27% compared to the closest competitor. In this highly nonlinear setting, N4SIDv-DAC performed worse than model-based PD, suggesting that the identified LTI disturbance model was inadequate. Overall, the proposed method provides the strongest performance and remains strong throughout the sweep.

## V. CONCLUSIONS

This paper presented a composite adaptive control framework based on a representation-learning approach for systems with unobserved (or partially observed), dynamically coupled disturbance sources. The presented controller complements prior “fixed-decay” last-layer representation-learning approaches by incorporating the classical disturbance-accommodating control (DAC) viewpoint. By modeling the disturbance as the output of a uniformly contractive latent dynamical system, the proposed method introduces a dynamics prior that supports disturbance prediction while mitigating sensitivity to latent-state initialization errors. The latent dynamics are parameterized by neural networks and trained using a statistically principled hard expectation–maximization procedure.

Hardware experiments on a tracked vehicle carrying a partially filled liquid tank and a pendulum illustrate the practical feasibility of the proposed disturbance estimator and the composite adaptive controller. The learned dynamical representation improves overall tracking performance during aggressive maneuvering, with the clearest improvement observed in yaw-rate regulation. In simulation, experiments on a system of coupled Duffing oscillators demonstrate accurate disturbance prediction and improved tracking relative to the baselines, while remaining strong under progressively noisier disturbance proxies $y .$ Here, the sparse availability of y allowed us to assess the role of predictive capability for maintaining tracking performance.

Taken together, the results indicate that the proposed method holds substantial promise for real-world deployment, particularly when disturbance proxies are sampled at intervals which are not negligible relative to the characteristic disturbance timescale, making predictive compensation important between observations. Future work may include evaluating the method across a wider range of hardware platforms and experimental conditions, as well as systematically studying the training dynamics of the hard-EM procedure, particularly the evolution of covariance estimates. Future work may also include replacing the present hard E-step with the standard EM E-step, which takes expectations with respect to the posterior distribution over latent trajectories. The proposed dynamical representation could also serve as a disturbance model in stochastic MPC with chance constraints [30], potentially useful for real-time trajectory planning. Acknowledgment: The authors thank G. X. Johnson, J. Alindogan, and S. Sukhatme for their help with hardware troubleshooting. The authors also thank the anonymous reviewers.

## REFERENCES

[1] M. O’Connell, G. Shi, X. Shi, K. Azizzadenesheli, A. Anandkumar, Y. Yue, and S.-J. Chung, “Neural-fly enables rapid learning for agile flight in strong winds,” Science Robotics, vol. 7, no. 66, p. eabm6597, 2022.

[2] E. S. Lupu, F. Xie, J. A. Preiss, J. Alindogan, M. Anderson, and S.-J. Chung, “Magic-vfm: Meta-learning adaptation for ground interaction control with visual foundation models,” IEEE Trans. Robot., vol. 41, p. 180–199, Jan. 2025.

[3] F. C. Yilmaz, R. Onler, E. Tatlicioglu, and E. Zergeroglu, “A transfer learning based deep neural network adaptive controller for the furuta pendulum subject to uncertain disturbance signals,” Scientific Reports, vol. 15, p. 24012, 2025.

[4] C. D. Johnson, “Disturbance-accommodating control; an overview,” in American Control Conf., 1986, pp. 526–536.

[5] E. Davison, “The robust control of a servomechanism problem for linear time-invariant multivariable systems,” IEEE Trans. Autom. Control, vol. 21, no. 1, pp. 25–34, 1976.

[6] B. Francis and W. Wonham, “The internal model principle of control theory,” Automatica, vol. 12, no. 5, pp. 457–465, 1976.

[7] M. Bin, L. Marconi, and A. R. Teel, “Adaptive output regulation for linear systems via discrete-time identifiers,” Automatica, vol. 105, pp. 422–432, 2019.

[8] G. Joshi, J. Virdi, and G. Chowdhary, “Asynchronous deep model reference adaptive control,” in Proceedings of the 2020 Conference on Robot Learning, vol. 155. PMLR, 2021, pp. 984–1000.

[9] Y. Song, M. Steinweg, E. Kaufmann, and D. Scaramuzza, “Autonomous drone racing with deep reinforcement learning,” in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2021). IEEE Press, 2021, p. 1205–1212.

[10] K. Y. Chee, T. C. Silva, M. A. Hsieh, and G. J. Pappas, “Enhancing sample efficiency and uncertainty compensation in learning-based model predictive control for aerial robots,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2023), 2023.

[11] J. Kabzan, L. Hewing, A. Liniger, and M. N. Zeilinger, “Learningbased model predictive control for autonomous racing,” IEEE Robot. Autom. Lett., vol. 4, no. 4, pp. 3363–3370, 2019.

[12] A. Kumar, Z. Li, J. Zeng, D. Pathak, K. Sreenath, and J. Malik, “Adapting rapid motor adaptation for bipedal robots,” in IEEE/RSJ Int. Conf. Intell. Robot. Sys., 2022.

[13] T. Miki, J. Lee, J. Hwangbo, L. Wellhausen, V. Koltun, and M. Hutter, “Learning robust perceptive locomotion for quadrupedal robots in the wild,” Science Robotics, vol. 7, no. 62, p. eabk2822, 2022.

[14] P. Krupa, M. Zanon, and A. Bemporad, “Learning disturbance models for offset-free reference tracking,” IEEE Trans. Autom. Control, vol. 70, no. 11, pp. 7683–7690, 2025.

[15] H. Zhang, J. Ge, J. Su, K. Gu, F. Wang, W.-H. Chen, and S. Li, “Model predictive control with residual learning and real-time disturbance rejection: Design and experimentation,” Control Eng. Practice, vol. 165, p. 106587, 2025.

[16] J. Jia, W. Zhang, K. Guo, J. Wang, X. Yu, Y. Shi, and L. Guo, “Evolver: Online learning and prediction of disturbances for robot control,” IEEE Trans. Robot., vol. 40, pp. 382–402, 2024.

[17] Y. Gu, L. Chen, C. Li, and S. Yin, “Multiple-model state-space system identification with time delay using the em algorithm,” J. Franklin Inst., vol. 361, no. 16, p. 107113, 2024.

[18] W. Wang, S. Liu, Y. Jiang, J. Sun, W. Xu, X. Chen, Z. Dong, and W. Jiao, “A robust filter and smoother-based expectation–maximization algorithm for bilinear systems with heavy-tailed noise,” Mech. Syst. Signal Proc., vol. 236, p. 112912, 2025.

[19] W. Lohmiller and J.-J. E. Slotine, “On contraction analysis for nonlinear systems,” Automatica, vol. 34, no. 6, pp. 683–696, 1998.

[20] F. Bullo, Contraction Theory for Dynamical Systems, 1.3 ed. Kindle Direct Publishing, 2026.

[21] J.-J. E. Slotine and W. Li, “Composite adaptive control of robot manipulators,” Automatica, vol. 25, no. 4, pp. 509–519, 1989.

[22] ——, Applied Nonlinear Control, ser. Prentice-Hall International Editions. Prentice-Hall, 1991.

[23] H. Abou-Kandil, G. Freiling, V. Ionescu, and G. Jank, Matrix Riccati Equations in Control and Systems Theory. Birkhauser Basel, 2003.¨

[24] H. Beik-Mohammadi, S. Hauberg, G. Arvanitidis, N. Figueroa, G. Neumann, and L. Rozo, “Neural contractive dynamical systems,” 2024. [Online]. Available: https://arxiv.org/abs/2401.09352

[25] G. Celeux and G. Govaert, “A classification em algorithm for clustering and two stochastic versions,” Comp. Stat. Data Anal., vol. 14, no. 3, pp. 315–332, 1992.

[26] D. Sanz-Alonso, A. Stuart, and A. Taeb, Inverse Problems and Data Assimilation, ser. London Mathematical Society Student Texts. Cambridge University Press, 2023.

[27] P. van Overschee and B. De Moor, Subspace Identification for Linear Systems. Springer New York, NY, 1996.

[28] P. Van Overschee and B. De Moor, “N4sid: Subspace algorithms for the identification of combined deterministic-stochastic systems,” Automatica, vol. 30, no. 1, pp. 75–93, 1994, special issue on statistical signal processing and control.

[29] M. Ramezani, M. Amin Alandihallaj, B. C. Yalc¸ın, M. A. Olivares Mendez, and A. M. Hein, “Fuel-aware autonomous docking using rlaugmented mpc rewards for on-orbit refueling,” Acta Astronautica, vol. 238, pp. 690–705, 2026.

[30] A. Mesbah, “Stochastic model predictive control: An overview and perspectives for future research,” IEEE Control Syst. Mag., vol. 36, no. 6, pp. 30–44, 2016.

[31] S. Sarkk¨ a and L. Svensson,¨ Bayesian Filtering and Smoothing, 2nd ed., ser. Institute of Mathematical Statistics Textbooks. Cambridge University Press, 2023.

## APPENDIX

## A. Hard and Soft Expectation–Maximization

We briefly outline the changes required to replace hard EM with soft EM.

a) Hard vs. Soft EM: Suppose a window of measured features, control inputs, and disturbance observations $\{ ( \phi _ { k } , u _ { k } , y _ { k } ) \} _ { k = 0 } ^ { L - 1 }$ is given. Write $\mathbf { y } ~ : = ~ y _ { 0 : L - 1 }$ and $\eta : =$ $( \theta , R _ { d } , Q _ { d } )$ . In the general case, the missing value is the entire trajectory $Z = \mathbf { a } _ { 0 : L - 1 }$ . For a fixed $\eta ,$ , the statistical model in Section III-C can be written as

$$
\begin{array} { r l r } { \mathbf { a } _ { k + 1 } = F _ { k } \mathbf { a } _ { k } + g _ { k } + \xi _ { k } , } & { { } } & { \xi _ { k } \sim \mathcal { N } ( 0 , Q _ { d } ) , } \\ { y _ { k } = H _ { k } \mathbf { a } _ { k } + h _ { k } + \epsilon _ { k } , } & { { } } & { \epsilon _ { k } \sim \mathcal { N } ( 0 , R _ { d } ) , } \end{array}
$$

with ${ \bf a } _ { 0 } \sim \mathcal { N } ( 0 , \lambda _ { a } ^ { - 1 } I )$ , where $\lambda _ { a } \mathrm { ~ > ~ } 0$ is a fixed hyperparameter. Here, ${ \bf a } _ { 0 } , \xi _ { k } , \epsilon _ { k }$ are independent. Note that $( F _ { k } , g _ { k } )$ comes from discretizing (3a) in time by, say, RK2 or forward Euler; the coefficients $( F _ { k } , g _ { k } , H _ { k } , h _ { k } )$ depend on θ.

At iteration $r ,$ let $\pi ^ { r } ( z ) : = p ( z \mid \mathbf { y } , \boldsymbol { \eta } ^ { r } )$ and $Q ^ { r } ( \eta ) : = $ $- \mathbb { E } _ { \pi ^ { r } } \left[ \log p ( \mathbf { y } , Z \mid \boldsymbol { \eta } ) \right]$ ]. Standard (soft) EM minimizes $Q ^ { r } ( \eta )$ with respect to η, holding $\pi ^ { r }$ fixed, to obtain $\eta ^ { r + 1 }$ . Hard EM instead selects $\hat { z } ^ { r } : = \arg$ max $\pi ^ { r } ( z )$ and minimizes $- \log p ( \mathbf { y } , \hat { z } ^ { r } \mid \eta )$ with respect to $\eta ,$ , holding $\hat { z } ^ { r }$ fixed. In other words, soft EM averages over the latent uncertainty $Z \sim \pi ^ { r }$ , whereas hard EM uses a single point estimate of $Z ,$ replacing the soft E-step (expectation) with a hard E-step (posterior maximization).

b) The Soft EM Loss: Denote the current parameters as $\eta ^ { r } ~ = ~ ( \theta ^ { r } , R _ { d } ^ { r } , Q _ { d } ^ { r } )$ , where we assume $R _ { d } ^ { r } ~ \succ ~ 0$ and $Q _ { d } ^ { r } \succ 0 .$ . Compute the smoothed means and covariances using a Kalman smoother under parameters $\eta ^ { r }$ , and denote the smoothed moments by $m _ { k } = \mathbb { E } _ { \pi ^ { r } } [ \mathbf { a } _ { k } ] , P _ { k } = \mathrm { C o v } _ { \pi ^ { r } } ( \mathbf { a } _ { k } )$ and $\begin{array} { r } { C _ { k } \ = \ \mathrm { C o v } _ { \pi ^ { r } } ( \mathbf { a } _ { k + 1 } , \mathbf { a } _ { k } ) } \end{array}$ ; here, the cross-covariances are obtained from the smoothing gains and the smoothed covariances as follows [31, Eq. (12.13)]:

$$
\begin{array} { r } { G _ { k } : = P _ { k | k } F _ { k } ^ { \top } ( P _ { k + 1 | k } ) ^ { - 1 } , \qquad C _ { k } = P _ { k + 1 } G _ { k } ^ { \top } . } \end{array}\tag{21}
$$

The posterior moments $( m _ { k } , P _ { k } , C _ { k } )$ remain fixed throughout the M-step.

For candidate parameters $\eta ,$ let $\mathbf { a } _ { 0 : L - 1 } \sim \pi ^ { r }$ and define the random variables $r _ { k } ^ { y } : = y _ { k } - H _ { k } { \bf a } _ { k } - h _ { k }$ and $r _ { k } ^ { a } : =$ $\mathbf { a } _ { k + 1 } - F _ { k } \mathbf { a } _ { k } - g _ { k }$ . We have

$$
\begin{array} { r l } & { r _ { k } ^ { y } = e _ { k } - H _ { k } ( { \mathbf a } _ { k } - m _ { k } ) , } \\ & { r _ { k } ^ { a } = v _ { k } + ( { \mathbf a } _ { k + 1 } - m _ { k + 1 } ) - F _ { k } ( { \mathbf a } _ { k } - m _ { k } ) , } \end{array}
$$

where $e _ { k } : = y _ { k } - H _ { k } m _ { k } - h _ { k }$ and $v _ { k } : = m _ { k + 1 } - F _ { k } m _ { k } - g _ { k }$ Therefore, the residual second moments are

$$
\begin{array} { r } { { S _ { k } ^ { y } : = \mathbb { E } _ { \pi ^ { r } } [ r _ { k } ^ { y } ( r _ { k } ^ { y } ) ^ { \top } ] = e _ { k } e _ { k } ^ { \top } + H _ { k } P _ { k } H _ { k } ^ { \top } } , } \end{array}\tag{22a}
$$

$$
\begin{array} { r } { S _ { k } ^ { a } : = \mathbb { E } _ { \pi ^ { r } } [ r _ { k } ^ { a } ( r _ { k } ^ { a } ) ^ { \top } ] = v _ { k } v _ { k } ^ { \top } + P _ { k + 1 } + F _ { k } P _ { k } F _ { k } ^ { \top } } \\ { - C _ { k } F _ { k } ^ { \top } - F _ { k } C _ { k } ^ { \top } . } \end{array}\tag{22b}
$$

From this, we can calculate the soft-EM loss averaged over a mini-batch of B windows (indexed by $w = 1 , \ldots , B )$ as

$$
\begin{array} { l } { { \displaystyle Q ^ { r } ( \eta ) = \frac { 1 } { B } \sum _ { w = 1 } ^ { B } \frac { 1 } { 2 } \Bigg [ \sum _ { k = 0 } ^ { L - 1 } \left[ \mathrm { t r } ( R _ { d } ^ { - 1 } S _ { k } ^ { y , ( w ) } ) + \log | R _ { d } | \right] } } \\ { { \displaystyle ~ + \sum _ { k = 0 } ^ { L - 2 } \left[ \mathrm { t r } ( Q _ { d } ^ { - 1 } S _ { k } ^ { a , ( w ) } ) + \log | Q _ { d } | \right] } } \\ { { \displaystyle ~ + \lambda _ { a } \left( \| m _ { 0 } ^ { ( w ) } \| ^ { 2 } + \mathrm { t r } P _ { 0 } ^ { ( w ) } \right) + L ( d _ { a } + d _ { u } ) \log ( 2 \pi ) - d _ { a } \log \lambda _ { a } \Bigg ] } . }  \end{array}\tag{23}
$$

Note that the last line in (23) can be omitted in the loss function for the M-step.

Now, our M-step is as follows: compute the new $( R _ { d } , Q _ { d } )$ by

$$
R _ { d } ^ { + } = \frac { 1 } { B L } \sum _ { w = 1 } ^ { B } \sum _ { k = 0 } ^ { L - 1 } S _ { k } ^ { y , ( w ) } ,\tag{24a}
$$

$$
Q _ { d } ^ { + } = \frac { 1 } { B ( L - 1 ) } \sum _ { w = 1 } ^ { B } \sum _ { k = 0 } ^ { L - 2 } S _ { k } ^ { a , ( w ) } ,\tag{24b}
$$

and then update θ using an optimizer such as AdamW to minimize the loss in (23), omitting its last line. In practice, eigenvalue clipping might be needed to ensure positive definiteness of $( R _ { d } ^ { + } , Q _ { d } ^ { + } )$

c) The $Q _ { d } \ = \ 0$ Variant: In Section III-B, the latent trajectory is determined by θ and the initial state $\mathbf { a } _ { 0 } .$ , so we take the missing value as $Z = \mathbf { a } _ { 0 }$ . Hard EM uses the MAP estimate for ${ \bf a } _ { 0 } ;$ soft EM instead retains the posterior distribution $\mathbf { a } _ { 0 } \mid \mathbf { y } , \boldsymbol { \eta } ^ { r } \sim \mathcal { N } ( m _ { 0 } , P _ { 0 } )$ . The posterior moments $( m _ { 0 } , P _ { 0 } )$ remain fixed throughout the soft-EM M-step. Write the deterministic rollout-plus-readout map as $\hat { d } _ { k } =$ $M _ { k } { \bf a } _ { 0 } + c _ { k }$ where $( M _ { k } , c _ { k } )$ depend on the candidate $\theta ,$ and define

$$
e _ { k } : = y _ { k } - M _ { k } m _ { 0 } - c _ { k } ,
$$

$$
\begin{array} { r } { S _ { k } ^ { y } : = e _ { k } e _ { k } ^ { \top } + M _ { k } P _ { 0 } M _ { k } ^ { \top } . } \end{array}\tag{25}
$$

(26)

Assuming $R _ { d } \succ 0$ , the soft-EM objective reduces to

$$
Q ^ { r } ( \eta ) = \frac { 1 } { B } \sum _ { w = 1 } ^ { B } \frac { 1 } { 2 } \sum _ { k = 0 } ^ { L - 1 } \left[ \mathrm { t r } ( R _ { d } ^ { - 1 } S _ { k } ^ { y , ( w ) } ) + \log | R _ { d } | \right]\tag{27}
$$

up to additive constants.

## B. Regularization in the N4SID-Variant Implementation

Relative to the algorithm in [27, Fig. 4.7], our implementation applies ridge-regularized pseudoinverse approximations at three locations: (i) in the oblique projections [27, Eq. (1.7)], when computing $\mathcal { O } _ { i }$ and $\mathcal { O } _ { i + 1 }$ (step 1); (ii) $\Gamma _ { i } ^ { \dagger }$ and $\Gamma _ { i - 1 } ^ { \dagger }$ in determining the state sequences (step 5); and $( i i i ) ~ { \mathcal { Z } } ^ { \dagger }$ , with $\mathcal { Z } \ : = \ [ \tilde { X } _ { i } ^ { \top } , U _ { i | i } ^ { \top } ]$ , in the least-squares solve for $( A , B , C , D )$ (step 6). For each ridge-regularized pseudoinverse approximation, we solve $( G + \lambda _ { G } I ) X = B _ { G } .$ where $G ~ \in ~ \mathbb { R } ^ { n \times n }$ is the associated Gram matrix, $B _ { G }$ denotes the right-hand side, and $\lambda _ { G } = 0 . 1 \mathrm { t r } ( G ) / n$ . Here, 0.1 is a relative regularization parameter chosen for our implementation.