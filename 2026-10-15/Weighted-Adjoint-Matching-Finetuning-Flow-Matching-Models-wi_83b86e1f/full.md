# Weighted Adjoint Matching: Finetuning Flow Matching Models with Time-Dependent Control Cost

Eliot Beyler

Francis Bach

Inria, Ecole Normale Supérieure, PSL Research University

## Abstract

We introduce Weighted Adjoint Matching, a method to finetune a pretrained diffusion or flow matching model given a differentiable reward. We derive it from Adjoint Matching, which reformulates finetuning as a stochastic optimal control problem with a quadratic control cost. Building on the observation that these models generate coarse features before fine details, we place a timedependent weight on the control cost. A high cost early in the generative process preserves balance and diversity in the coarse features, which carry most of the semantics, while a low cost at the end lets the model increase the reward by refining details. We prove that our method is equivalent to maximizing the reward under time-varying Kullback-Leibler regularization on the generative paths. Experiments on Gaussian mixtures and on MNIST show a better tradeoff between reward and mode balance than Adjoint Matching.

## 1 INTRODUCTION

Diffusion models [7, 10, 11, 14] and flow matching models [20, 17, 21] are now the state-of-the-art methods for learning and sampling a distribution known from a large dataset of empirical samples. They work by approximating a denoising process, modeled as an ordinary or stochastic differential equation whose drift is learned with score matching [5]. In this article, we focus on the task of finetuning a pretrained flow matching model, denoting $p _ { \mathrm { t a r g e t } }$ the distribution on $\mathbb { R } ^ { d }$ it samples from.

Finetuning a generative model. To finetune the model, we assume that we are given a differentiable reward function $r : \mathbb { R } ^ { d }  \mathbb { R }$ . Our goal is to increase $r ( x )$ when x is a sample from the model.

A first naive way to define the finetuning could be the maximization max $ _ { p } \operatorname { \mathbb { E } } _ { p } [ r ( X ) ]$ . But this objective makes the distribution diverge or collapse onto a maximum $x ^ { * }$ of the reward $r ;$ hence it does not capture any structure in the original data distribution ptarget. Therefore, it is natural to constrain the finetuned distribution to be close to the original distribution ptarget, for example by penalizing the Kullback-Leibler (KL) divergence from $p _ { \mathrm { t a r g e t } } .$ for some $a > 0$

$$
\operatorname* { m a x } _ { p } \mathbb { E } _ { p } [ r ( X ) ] - \frac { 1 } { a } \operatorname { K L } ( p \| p _ { \mathrm { t a r g e t } } ) .\tag{1}
$$

This can be shown to be equivalent to sampling from the tilted distribution $p _ { a }$

$$
p _ { a } ( x ) \propto p _ { \mathrm { t a r g e t } } ( x ) e ^ { a r ( x ) } .\tag{2}
$$

Adjoint matching. In practice, optimizing (1) directly is intractable, and different methods aim to sample from $p _ { a }$ by using the structure of the generative process. The generative model can be seen as a stochastic process $d X _ { t } ~ = ~ b ( X _ { t } , t ) d t + \gamma ( t ) d B _ { t } .$ $X _ { 0 } \sim \mathcal { N } ( 0 , I )$ for some drift $b : \mathbb { R } ^ { d } \times \mathbb { R } \to \mathbb { R } ^ { d }$ and diffusion coefficient $\gamma : [ 0 , 1 ] \ \to \ \mathbb { R }$ such that the data distribution is the end distribution of the process $p _ { 1 } : = \mathcal { L } ( X _ { 1 } ) = p _ { \mathrm { t a r g e t } }$ . Domingo-Enrich et al. [35] propose to reframe the finetuning as a stochastic optimal control (SOC) problem,

$$
\begin{array} { r l r } { \underset { u \in \mathcal { U } } { \operatorname* { m i n } } } & { \mathbb { E } \left[ \displaystyle \int _ { 0 } ^ { 1 } \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t - a r ( X _ { 1 } ^ { u } ) \right] } & { ( 3 ) } \\ { \mathrm { s . t . } } & { d X _ { t } ^ { u } = \left( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) \right) d t + \gamma ( t ) d B _ { t } , } & \\ & { X _ { 0 } ^ { u } \sim \mathcal { N } ( 0 , I ) . } & \end{array}
$$

This is motivated by Girsanov's theorem [see, e.g., 8], which links the squared norm of the control and the KL divergence between the base and the controlled processes. They develop a method to learn the optimal control $u ^ { * }$ and prove that it retrieves the tilted distribution (2) with $p _ { 1 } ^ { u ^ { * } } : = \mathcal { L } ( X _ { 1 } ^ { u ^ { * } } ) = p _ { a }$

Tilting creates imbalance. However, we may wonder if sampling from $p _ { a } ( x ) \propto p _ { \mathrm { t a r g e t } } ( x ) e ^ { a r ( x ) }$ is what we want in the first place. Indeed, this tilting has the unintended effect of moving mass between different regions of the distribution regardless of distance in sample space. We illustrate this on a simple example¹

![](images/ff38f91040994f9202c45b039f41e31a1c4ebc9bf222a9654a5efe8c2cfec1ed.jpg)  
Figure 1: Trajectories of a flow matching model, in black, first sample the coarse features, illustrated here as the class of the digit, and only later its fine details, here differences in brightness. Suppose we want to finetune the model to produce brighter digits. Adjoint matching [35], in red, penalizes the control uniformly in time. Early in the process, when the coarse features are not yet settled, a small push is enough to move a trajectory towards the class that is brighter on average, here the eights. Weighted adjoint matching, in green, makes the control expensive at the beginning of the process, so that the finetuned trajectories follow the base process until the coarse features are settled, then it refines only the details, leading to a more balanced distribution.

$$
\begin{array} { r } { p _ { \mathrm { t a r g e t } } ( x ) = \frac { 1 } { 2 } \mathcal { N } ( x ; - \mu , \sigma _ { 1 } ^ { 2 } I ) + \frac { 1 } { 2 } \mathcal { N } ( x ; \mu , \sigma _ { 2 } ^ { 2 } I ) , } \end{array}
$$

with $\mu = ( m , 0 ) \in \mathbb { R } ^ { 2 } , m > 0$ and $\sigma _ { 2 } > \sigma _ { 1 } > 0$ . For $\boldsymbol { x } = ( x _ { 1 } , x _ { 2 } )$ , we define the reward $\begin{array} { r } { r ( x ) = - { \frac { \| x _ { 2 } \| ^ { 2 } } { 2 } } } \end{array}$ to encourage samples to be closer to the first axis. Note that both modes have their means $\pm \mu$ on this axis and only differ by their variances. The tilted distribution $p _ { a }$ is also a mixture of two Gaussian distributions

$$
p _ { a } ( x ) = \kappa _ { a } ^ { 1 } \mathcal { N } ( x ; - \mu , \Sigma _ { 1 , a } ) + \kappa _ { a } ^ { 2 } \mathcal { N } ( x ; \mu , \Sigma _ { 2 , a } )
$$

with covariance matrices $\begin{array} { r } { \Sigma _ { i , a } = \mathrm { d i a g } \left( \sigma _ { i } ^ { 2 } , \frac { \sigma _ { i } ^ { 2 } } { 1 + a \sigma _ { i } ^ { 2 } } \right) } \end{array}$ and weights $\kappa _ { a } ^ { 1 }$ and $\kappa _ { a } ^ { 2 } .$ such that $\begin{array} { r } { \frac { \kappa _ { a } ^ { 1 } } { \kappa _ { a } ^ { 2 } } = \sqrt { \frac { 1 + a \sigma _ { 2 } ^ { 2 } } { 1 + a \sigma _ { 1 } ^ { 2 } } } > 1 } \end{array}$ . For the finetuned distribution, the first mode has a weight strictly larger than the second, creating an imbalance that was absent from the data distribution, regardless of the distance 2m between the centers of the modes.

Coarse-to-fine generation in diffusion and flow matching models. To address this issue, we rely on the fact that for natural data such as images, diffusion and flow matching models generate coarse features (the outline of an image) before fine ones (details and textures) [34]. This phenomenon can be understood by looking at the frequency domain [23, 29]: for natural images the magnitude of the (discrete) Fourier transform decays with frequency, whereas it is flat for Gaussian white noise, so progressively adding noise erases high frequencies of low magnitude before low frequencies of high magnitude. The fact that features of high magnitude are generated before those of low magnitude can be understood within the literature on mode speciation. This approach to the question asks what it means for a feature to be “generated" at a given point in the process: past a certain time in the generative process, the feature is already determined, in the sense that almost all trajectories through the current state terminate in samples sharing it [22, 26, 38, 39]. In particular, Li and Chen [31] give non-asymptotic results and analyze hierarchical data in which features live at different scales, showing that each feature emerges in a critical window of time, from features at large spatial scales to features at small spatial scales.

Weighted adjoint matching. Our idea is to exploit this phenomenon by modifying the SOC problem and introducing a time-dependent weight w(t) on the control quadratic cost

$$
\operatorname* { m i n } _ { u \in \mathcal { U } } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t - a r ( X _ { 1 } ^ { u } ) \right] .
$$

Modulating the control cost lets us finetune the model differently across the different spatial resolutions of the data distribution. Typically, we penalize the early timesteps more heavily, which preserves the coarse features and lets the finetuning increase the reward by modifying the details. The insight behind this choice is that coarse features carry most of the semantic meaning, while the reward can also be increased by tweaking only finer details.2

Contributions. In this paper, we make the following contributions:

• We introduce a new algorithm (Section 4), weighted adjoint matching, to finetune flow matching and diffusion models while preserving diversity, motivated by the link between time and spatial resolutions in these models. We derive it as a simple modification of adjoint matching [35] with no extra computational cost.

• We show that the SOC problem defining the algorithm is equivalent to maximizing the reward under a KL regularization on every sub-path [0, t] of the generative process (Section 4.1).

• We illustrate the method on Gaussian mixtures and on MNIST, for which the reward/mode collapse tradeoff can be precisely measured (Section 5).

## 2 RELATED WORK

Early works aim to increase the expected value of the reward $\mathbb { E } _ { p } [ r ( X ) ]$ , often with some form of implicit regularization to limit collapse, when a model only generates very similar samples of high reward, but without characterizing the finetuned distribution. Xu et al. [24] introduce ImageReward, a model trained on human preferences to score images, together with a finetuning method (ReFL) that backpropagates the reward gradient through a randomly selected step of the sampling process. Clark et al. [28] improve upon it by backpropagating through the last K steps of the process.

Several methods (see, e.g., [27, 18]) cast the finetuning of a diffusion model as reinforcement learning, viewing the denoising chain as a Markov decision process (see [32] for a literature review). However, both Uehara et al. [33] and Domingo-Enrich et al. [35] discuss the fact that reinforcement learning with KL regularization at each step does not in general sample from (2). Uehara et al. [33] take this into account by tilting the initial distribution, which requires an auxiliary model. Domingo-Enrich et al. [35] prove that the solution of an SOC problem with quadratic control cost recovers (2), provided the base process uses a particular noise schedule. Another work by Potaptchik, Lee, and Albergo [47] progressively tilts the velocity field of the base generative model to match that of the tilted distribution (2). Marion et al. [37] propose a general framework to finetune different sampling algorithms, including unadjusted Langevin sampling and diffusion models, and in particular for the objective (1).

method, we have full flexibility to choose the timesteps that we penalize more, e.g., modify the coarse features and preserve the details, or penalize a specific frequency range.

Some reward-finetuning methods treat the timesteps of the generative process non-uniformly as we do, but for reasons of efficiency or of gradient quality rather than to preserve diversity. As recalled above, Xu et al. [24] backpropagate the reward through a single, randomly selected step, chosen among the last steps of the sampling process, where they report the strongest reward signal. Clark et al. [28] likewise restrict the backward pass to the last K steps. He et al. [42] and Lee and Ye [44] also weight timesteps non-uniformly, but inside GRPO-style policy-gradient objectives that use only scalar rewards, and with the goal of stabilizing credit assignment across steps. Note also that Han, Razaviyayn, and Xu [36] regularize each step of the sampling process with its own KL coefficient in their mathematical derivations, but only use a fixed coefficient in their experiments.

Closer to our motivation, Barceló, Alcázar, and Tobar [25] start from the same observation as we do, building on the works on coarse-to-fine sampling in diffusion models recalled above, and argue that treating the timesteps differently, in particular the early ones, may increase the diversity of the finetuned model by preventing mode collapse. Similarly, after noticing that diversity mostly depends on early steps in the generative process, Liu et al. [45] propose a GRPO-based finetuning method that enforces stronger regularization on those steps. Both these methods are nonetheless based on a reinforcement-learning framework that only uses the reward as a scalar signal, and they do not characterize the finetuned distribution. Guo et al. [41] build like us upon adjoint matching. They replace the stochastic sampler by a deterministic one, which they claim to be more efficient, but therefore also lose the characterization of the finetuned distribution. They also truncate the finetuning to the last steps of the trajectory, motivated by efficiency rather than diversity. This differs from our approach: not computing the loss on the early timesteps does not enforce that the finetuned velocity field stays equal to the base one at those times, since a single neural network is shared across timesteps. Our method instead ensures explicitly that the finetuned model stays close to the base model at the times where the weight w(t) is large.

Finally, Domingo-Enrich and Han [40] generalize adjoint matching to a large class of stochastic optimal control objectives, with control-dependent drift and diffusion coefficients and convex running costs. Our time-weighted objective (10) can be partially fitted into this framework, which recovers our weighted adjoint matching loss only for a particular choice of the stabilizing weight λ(t). We give more details on this connection in Appendix D.4.

## 3 GENERATIVE SDES: SAMPLING FEATURES PROGRESSIVELY

The goal of this section is to motivate the introduction of the time-varying weight $w ( t )$ as a way to adjust the intensity of the control according to the different phases of the process. We first introduce diffusion and flow matching models, and explain what is special about the choice of memoryless noise schedule proposed by Domingo-Enrich et al. [35]. We then link this choice to the mode speciation results [22, 26, 38, 31, 39] that we illustrate for a Gaussian mixture.

Forward and reverse processes of diffusion models.At the heart of diffusion and flow matching models is a Gaussian noise corruption process $( \mathrm { s e e } , \mathrm { e . g . } , \mathrm { [ 1 4 . }$ 15]). It starts from $X \sim p _ { \mathrm { t a r g e t } }$ and adds noise to $\mathrm { i t } \mathrm { : } ^ { 3 }$

$$
\left\{ \begin{array} { l l } { d \vec { X } _ { \tau } = - f ( \tau ) \vec { X } _ { \tau } d \tau + g ( \tau ) d B _ { \tau } , } \\ { \vec { X } _ { 0 } = X , } \end{array} \right.\tag{4}
$$

where $f , g \ : \ \mathbb { R } \ \mapsto \ \mathbb { R } _ { + }$ are continuous functions. For each time τ, the joint distribution of $( \vec { X } _ { 0 } , \vec { X } _ { \tau } )$ is such that $\vec { X } _ { \tau } ~ = ~ a ( \tau ) \vec { X } _ { 0 } + b ( \tau ) { \mathcal N } ( 0 , I )$ , with coefficients $\begin{array} { r c l } { { a ( \tau ) } } & { { = } } & { { \exp \left( - \int _ { 0 } ^ { \tau } f ( u ) d u \right) } } \end{array}$ , and $\begin{array} { r l } { b ( \tau ) } & { { } = } \end{array}$ $\begin{array} { r } { a ( \tau ) \sqrt { \int _ { 0 } ^ { \tau } \frac { g ( u ) ^ { 2 } } { a ( u ) ^ { 2 } } d u } } \end{array}$ . This joint distribution is therefore simple to study and to sample from.

A generative model is obtained by reversing this corruption process. Choosing $T > 0$ and a noise schedule $\gamma : \mathbb { R } \to \mathbb { R } ^ { + }$ , and denoting $\overleftarrow { f } ( t ) = f ( T - t ) , \overleftarrow { g } ( t ) =$ $g ( T - t ) , \overleftarrow { p _ { t } } = \overrightarrow { p } _ { T - t }$ the density of $\overrightarrow { X } _ { T - t } ,$ and Wt a Brownian motion, the process defined by

$$
\left\{ \begin{array} { l l } { d \overleftarrow { X } _ { t } = \left( \overleftarrow { f } ( t ) \overleftarrow { X } _ { t } + \frac { \gamma ( t ) ^ { 2 } + \overleftarrow { g } ( t ) ^ { 2 } } { 2 } \nabla \log \overleftarrow { p } _ { t } ( \overleftarrow { X } _ { t } ) \right) d t } \\ { \quad + \gamma ( t ) d W _ { t } , } \\ { \overleftarrow { X } _ { 0 } } & { = \overrightarrow { X } _ { T } . } \end{array} \right.\tag{5}
$$

has the same marginals as $\vec { X } _ { \tau : } : \ \forall t \in [ 0 , T ] , \mathcal { L } ( \overleftarrow { X } _ { t } ) =$ $\mathcal { L } ( \overrightarrow { X } _ { T - t } )$

Different choices of $\gamma ( t )$ are possible but two are particularly remarkable. Choosing $\gamma ( t ) = 0$ makes the backward process an ODE, and leads to deterministic sampling methods such as DDIM [13]. But the one that interests us here is $\gamma ( t ) = \overleftarrow { g } ( t )$ : then $\mathcal { L } ( ( \overleftarrow { X } _ { t } ) _ { t \in [ 0 , T ] } ) =$ $\mathcal { L } ( ( \vec { X } _ { T - t } ) _ { t \in [ 0 , T ] } )$ [2]. The backward process preserves not only the marginals of the forward but also the path distribution.

Flow matching and the memoryless noise schedule. Following [35], we use a flow matching model with the Gaussian source distribution. To define it, start from $\bar { X } _ { 0 } ~ \sim ~ p _ { \mathrm { n o i s e } } ~ = ~ \mathcal { N } ( 0 , I )$ and $\bar { X } _ { 1 } \sim p _ { \mathrm { t a r g e t } }$ two independent random variables, and for $t \in [ 0 , 1 ] , \bar { X } _ { t } = \beta _ { t } \bar { \bar { X } } _ { 0 } + \alpha _ { t } \bar { X } _ { 1 }$ , with $\alpha _ { t } , \beta _ { t }$ differentiable schedules such that $\alpha _ { 0 } = \beta _ { 1 } = 0 , \alpha _ { 1 } = \beta _ { 0 } = 1$ 2 $t \mapsto \alpha _ { t }$ non-decreasing, $t \mapsto \beta _ { t }$ non-increasing. A flow matching model is a process

$$
\left\{ \begin{array} { l l } { d X _ { t } } & { = v ( X _ { t } , t ) d t } \\ { X _ { 0 } } & { \sim p _ { \mathrm { n o i s e } } = \mathcal { N } ( 0 , I ) , } \end{array} \right.\tag{6}
$$

with v the velocity field defined as the minimizer of the quadratic loss E $\left\lceil \| v ( \bar { X } _ { t } , t ) - \dot { \beta } _ { t } \bar { X } _ { 0 } - \dot { \alpha } _ { t } \bar { X } _ { 1 } \| ^ { 2 } \right\rceil$ . [35] transform the sampling process (6) to an SDE by adding a diffusion coefficient $\gamma ( t )$

$$
\left\{ \begin{array} { l l } { d X _ { t } } & { = b ^ { \gamma } ( X _ { t } , t ) d t + \gamma ( t ) d B _ { t } } \\ { X _ { 0 } } & { \sim \mathcal { N } ( 0 , I ) , } \end{array} \right.\tag{7}
$$

$$
\begin{array} { r } { \mathrm { w i t h ~ } b ^ { \gamma } ( x , t ) = v ( x , t ) + \frac { \gamma ( t ) ^ { 2 } } { 2 } \nabla \log p _ { t } ( x ) . } \end{array}
$$

Though defined differently, this is just a diffusion model with a particular choice of time parametrization. Indeed, the velocity field is linked to $\nabla \log p _ { t } ( x )$ through (see, e.g., [35] Appendix B.4)

$$
\begin{array} { r } { v ( x , t ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x + \beta _ { t } \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \right) \nabla \log p _ { t } ( x ) . } \end{array}\tag{8}
$$

Plugging this equality (8) into the flow matching process (7) gives the diffusion backward process (5) with $T = 1$ , and $\overleftarrow { f } ( t ) , \overleftarrow { g } ( t )$ taken such that $\alpha _ { t } = \overleftarrow { a } ( t ) : =$ $a ( 1 - t )$ and $\beta _ { t } = { \overline { { b } } } ( t ) : = b ( 1 - t )$ 2

$$
\begin{array} { r } { \overleftarrow { f } ( t ) = \frac { d } { d t } \log \alpha _ { t } = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } , \quad \overleftarrow { g } ( t ) = \sqrt { 2 \beta _ { t } \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \right) } . } \end{array}
$$

In particular, choosing $\gamma ( t ) = 0$ recovers deterministic diffusion models, while the memoryless noise schedule

$$
\begin{array} { r } { \gamma ( t ) = \sqrt { 2 \beta _ { t } \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \right) } , } \end{array}\tag{9}
$$

considered by Domingo-Enrich et al. [35] is simply $\gamma ( t ) = \overleftarrow { g } ( t )$ , i.e., the choice for which the backward SDE preserves the joint distributions. The memoryless property $p ( X _ { 0 } , X _ { 1 } ) = p ( X _ { 0 } ) p ( X _ { 1 } )$ is just a consequence of the çonservation of the joint distributions as $( X _ { 0 } , X _ { 1 } ) = ( { \overrightarrow { X } } _ { 1 } , { \overrightarrow { X } } _ { 0 } ) = ( \alpha _ { 0 } X + \beta _ { 0 } Z , X ) = ( Z , X )$ with $Z \sim { \mathcal { N } } ( 0 , I )$ independent from X.

Noising and spatial scale of the target distribution. In Figure 2, we illustrate the advantage of the sampling paths (a) given by the memoryless noise schedule, over the original deterministic paths (b) of flow matching models. For each point, the red component of the pixel color is proportional to the probability that a trajectory going through this point ends up in the upper mode, the blue component to the probability for the lower mode, and transparency is proportional to the density of the marginal distribution at this point. We see that for the memoryless sampling process (a), before the time window delimited by the two gray dotted lines, the class is undetermined (mostly purple) while after this window, the two classes are distinguishable. On the other hand, for a deterministic sampling process (b), classes are predetermined from the initial sampling at time 0.

![](images/48705f254611dc6adf1cb853df682f9f39438f08431ed12c85787cd27adda633.jpg)  
(a) Stochastic sampling paths.

![](images/72d43837642ccca0b2c1d89e8de3338be08e8c0949a2ef81574b91e059c36bd7.jpg)  
(b) Deterministic sampling paths.  
Figure 2: Red component proportional to probability that trajectory going through the point ends up in upper mode, blue to probability for lower mode, and transparency proportional to density of marginal distribution

More generally, the times of this destroying/generating of features depends on the spatial scale of the feature [23, 22, 34, 29, 26, 38, 39]. In particular, [31] characterize these critical windows where a feature emerges as $[ T _ { \mathrm { l o w e r } } , T _ { \mathrm { u p p e r } } ]$ , where both bounds depend directly on the spatial scale of the feature. Moreover, they generalize this result for a hierarchical model, with features at different spatial scales, leading to successive time windows, starting from large spatial scales to finer ones.

## 4 WEIGHTED ADJOINTMATCHING

We introduce a time-dependent weight $w : [ 0 , 1 ] \to \mathbb { R } _ { + } ^ { * }$ in the quadratic control cost of the SOC problem (3):

$$
\operatorname* { m i n } _ { u \in \mathcal { U } } \mathbb { E } \left[ \int _ { 0 } ^ { 1 } \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t - a r ( X _ { 1 } ^ { u } ) \right]\tag{10}
$$

$$
\mathrm { s . t . } \quad d X _ { t } ^ { u } = \left( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) \right) d t + \gamma ( t ) d B _ { t } ,
$$

$$
X _ { 0 } ^ { u } \sim p _ { 0 } ,\tag{11}
$$

where $r$ is the reward function and $a \mathrm { ~ > ~ } 0$ the tilting parameter of (1). To keep the exposition light, this section is written for the finetuning task, using a flow matching model with the memoryless noise schedule (9) and $\begin{array} { r } { \dot { b } ( x , t ) : = b ^ { \gamma } ( x , t ) = 2 v ^ { \mathrm { b a s e } } ( x , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x . } \end{array}$ Every result stated here holds for an arbitrary running cost $f ( x , t )$ , terminal cost $g ( x )$ , and base SDE on $t \in [ 0 , T ]$ (hence including the finetuning of diffusion models), and is stated and proven in that generality in

Appendix C and Appendix D. We assume that w is a function of bounded variation such that

$$
\begin{array} { r } { w ( t ) = w ( 0 ) + \int _ { 0 } ^ { t } d w ( s ) = w ( 1 ) - \int _ { t } ^ { 1 } d w ( s ) . } \end{array}\tag{12}
$$

In particular, a differentiable w satisfies these equalities with $d w ( s ) = w ^ { \prime } ( s ) d s$ , but the assumption also allows discontinuous weights such as piecewise constant functions. We denote $p ^ { u }$ the probability distribution on the path $\mathbf { X } ^ { u } = ( X _ { t } ^ { u } ) _ { t \in [ 0 , 1 ] }$ defined by a control u and $p ^ { \mathrm { b a s e } } = p ^ { u = 0 }$ the one of the base generative model.

## 4.1 Path-KL regularized objective

We denote by $K _ { t } ( X _ { 0 } )$ the KL divergence between the controlled and the base processes on the time interval [0, t], conditionally on the initial condition,

$$
K _ { t } ( X _ { 0 } ) : = \mathrm { K L } ( p ^ { u } ( X _ { [ 0 , t ] } | X _ { 0 } ) \| p ^ { \mathrm { b a s e } } ( X _ { [ 0 , t ] } | X _ { 0 } ) ) .\tag{13}
$$

The weighted control cost of (10) can be rewritten in terms of these path KL divergences, which turns the weighted SOC problem into an objective regularized by the KL divergence on every sub-path [0, t].

Proposition 1. Assume that w satisfies (12). Then the weighted SOC problem (10) is equivalent to

$$
\operatorname* { m a x } _ { u \in \mathcal { U } } \mathbb { E } _ { X _ { 0 } \sim p _ { 0 } } \Big [ a \mathbb { E } _ { { \mathbf { X } } ^ { u } \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } \left[ r ( X _ { 1 } ^ { u } ) \right]\tag{14}
$$

$$
\begin{array} { r } { - w ( 1 ) \operatorname { K L } ( p ^ { u } ( \mathbf { X } | X _ { 0 } ) \| p ^ { \mathrm { b a s e } } ( \mathbf { X } | X _ { 0 } ) ) + \int _ { 0 } ^ { 1 } K _ { t } ( X _ { 0 } ) d w ( t ) \Big ] . } \end{array}
$$

This result is proven in Appendix C, which also contains a dual reformulation, regularized by the KL on the sub-paths $[ t , 1 ] ,$ and the link between the two.

Compared to adjoint matching of [35], the KL between the paths on [0,t] enters the objective with weight $d w ( t )$ , which for a non-increasing w is negative and therefore adds regularization.

Example. To penalize more heavily the part of the process before mode speciation, we can take a piecewise constant function $w .$ with $w _ { 0 } > 0$

$$
w ( t ) = { \left\{ \begin{array} { l l } { 1 + w _ { 0 } } & { { \mathrm { i f } } \quad t < t _ { s } } \\ { 1 } & { { \mathrm { i f } } \quad t > t _ { s } } \end{array} \right. }
$$

for which divergence from the base distribution on $( 0 , t _ { s } )$ is penalized a second time by w0:

$$
\begin{array} { r l } {  { \operatorname* { m a x } _ { u \in \mathcal { U } } \mathbb { E } _ { X _ { 0 } \sim p _ { 0 } } \Big [ a \mathbb { E } _ { \mathbf { X } ^ { u } \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } [ r ( X _ { 1 } ^ { u } ) ] } \quad } & { } \\ & { \phantom { \sum } - \mathrm { K L } ( p ^ { u } ( \mathbf { X } \vert X _ { 0 } ) \Vert p ^ { \mathrm { b a s e } } ( \mathbf { X } \vert X _ { 0 } ) ) } \\ & { -  w _ { 0 } \mathrm { K L } ( p ^ { u } ( X _ { [ 0 , t _ { s } ] } \vert X _ { 0 } ) \Vert p ^ { \mathrm { b a s e } } ( X _ { [ 0 , t _ { s } ] } \vert X _ { 0 } ) ) ] . } \end{array}
$$

## 4.2 Algorithm

In this section, we derive an algorithm to learn the optimal control $u ^ { * }$ of the weighted SOC problem (10). We adapt the adjoint matching method [35]. We give here a general overview; the complete proofs of the results are given in Appendix D. We aim to learn the optimal control $u ^ { * }$ by using the fact that it satisfies

$$
u ^ { * } ( x , t ) = - \frac { \gamma ( t ) } { w ( t ) } \nabla _ { x } J ( u ^ { * } , x , t ) ,\tag{15}
$$

with the cost function defined as

$$
\begin{array} { r } { J ( u , x , t ) = \mathbb { E } _ { { \mathbf { X } } \sim p ^ { u } } \left[ \int _ { t } ^ { 1 } \frac { w ( s ) } { 2 } \| u ( X _ { s } , s ) \| ^ { 2 } d s - a r ( X _ { 1 } ) \bigg | X _ { t } = x \right] . } \end{array}
$$

It is the expectation of the adjoint state, defined by

$$
\begin{array} { r } { a ( t , { \bf X } ^ { u } , u ) = \nabla _ { X _ { t } ^ { u } } \left( \int _ { t } ^ { 1 } \frac { w ( s ) } { 2 } \| u ( X _ { s } ^ { u } , s ) \| ^ { 2 } d s - a r ( X _ { 1 } ^ { u } ) \right) , } \end{array}
$$

where $\nabla _ { X _ { t } ^ { u } }$ denotes the derivative with respect to the value $X _ { t } ^ { u }$ of the process at time t, keeping the trajectory of the Brownian motion $( B _ { s } ) _ { s \geq t }$ fixed. It can be computed by solving the backward ODE

$$
\begin{array} { r l } & { \frac { d } { d t } a ( t , { \mathbf { X } } ^ { u } , u ) = - \nabla _ { X _ { t } ^ { u } } \left( \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \right) } \\ & { - \left( \nabla _ { { X } _ { t } ^ { u } } \big ( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) \big ) \right) ^ { \top } a ( t , { \mathbf { X } } ^ { u } , u ) , } \end{array}\tag{16}
$$

with initial condition $a ( 1 , { \bf X } ^ { u } , u ) = - a \nabla r ( X _ { 1 } ^ { u } )$

Adjoint matching relies on two key ideas. First, using the adjoint state in the equation (15) gives

$$
u ^ { * } ( x , t ) = - \frac { \gamma ( t ) } { w ( t ) } \mathbb { E } [ a ( t , \mathbf { X } ^ { u ^ { * } } , u ^ { * } ) | X _ { t } = x ] .\tag{17}
$$

Secondly, for the optimal control $u ^ { * } .$ theterm $\begin{array} { r l } { \nabla _ { X _ { t } ^ { u } } ( \gamma ( t ) u ( X _ { t } ^ { u } , t ) ) ^ { \top } a ( t , \mathbf { X } ^ { u } , u ) } & { { } ~ + } \end{array}$ $\begin{array} { r l } {  { \nabla _ { X _ { t } ^ { u } } ( \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } ) } } \end{array}$ in (16) has zero expectation. Having a conditional expectation in (17), we can try to enforce this equality by using a square-norm matching trick similar to Vincent [5]. This leads to the following loss, where ã is the lean adjoint state, obtained by dropping the zero-expectation terms,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { W A M } } ( u ) : = \mathbb { E } _ { { \mathbf { X } } \sim p ^ { \bar { u } } } \left[ \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left. u ( X _ { t } , t ) + \frac { \gamma ( t ) \widetilde { a } ( t , { \mathbf { X } } ) } { w ( t ) } \right. ^ { 2 } d t \right] } \end{array}
$$

$$
( \bar { u } = \mathrm { s t o p g r a d } ( u ) )\tag{18}
$$

$$
\begin{array} { r l } { \mathrm { w h e r e } } & { \displaystyle \frac { d } { d t } \widetilde { a } ( t , \mathbf { X } ) = - \nabla _ { x } b ( X _ { t } , t ) ^ { \top } \widetilde { a } ( t , \mathbf { X } ) , } \\ & { \widetilde { a } ( 1 , \mathbf { X } ) = - a \nabla r ( X _ { 1 } ) . } \end{array}\tag{19}
$$

Our adaptation of adjoint matching turns out to be straightforward, with the extra time weight $w ( t )$ that only appears at one place in the loss.

Proper formulation of the "stopgrad". The stopgrad is convenient both as a notation and for implementing the algorithm in an autodiff framework. However, when taking the derivative in u of (18), the mathematical object that we consider is in fact the function of two variables $( u , v )$

$$
\begin{array} { r } { \hat { \mathcal { L } } _ { \mathrm { W A M } } ( u , v ) = \mathbb { E } _ { \mathbf { X } ^ { v } \sim p ^ { v } } \left[ \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \Big | \Big | u ( X _ { t } , t ) + \frac { \gamma ( t ) \widetilde { a } ( t , \mathbf { x } ^ { v } ) } { w ( t ) } \Big | \Big | ^ { 2 } d t \right] , } \end{array}
$$

that we differentiate with respect to u then take $v = u \colon$ $\textstyle \frac { \delta } { \delta u } { \mathcal { L } } _ { \mathrm { W A M } } ( u ) : = \frac { \delta } { \delta u } \hat { \mathcal { L } } _ { \mathrm { W A M } } ( u , \bar { v } ) | _ { v = u }$ While commonly used and accepted in the literature, doing gradient descent on an objective with a stopgrad gives no guarantee of decreasing the objective or of converging to a fixed point. We can, however, characterize the critical points of the objective, where the derivative is zero.

Proposition 2. The only critical point $\begin{array} { r } { \frac { \delta } { \delta u } \mathcal { L } _ { \mathrm { W A M } } ( u ) = } \end{array}$ 0 is $u ^ { * }$ , the optimal control of the weighted SOC problem (10).

The derivations above are detailed, and this result is proven, in Appendix D. Note also that the critical point is characterized through the derivative with respect to the control u itself, i.e., the functional derivative defined in Appendix D. In practice, u is parametrized by a neural network, $u = u _ { \theta }$ with $\theta \in \mathbb { R } ^ { p }$ and so is the loss, $\mathcal { L } _ { \boldsymbol { \theta } } : = \mathcal { L } _ { \mathrm { W A M } } ( u _ { \boldsymbol { \theta } } )$ . The chain rule gives $\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { \theta } = ( \nabla _ { \theta } u _ { \theta } ) ^ { * } \frac { \delta } { \delta u } \mathcal { L } _ { \mathrm { W A M } } ( u ) } \end{array}$ , with $\nabla _ { \boldsymbol { \theta } } u _ { \boldsymbol { \theta } }$ the Jacobian of $\theta \mapsto u _ { \theta }$ and $( \overline { { \nabla } } _ { \theta } u _ { \theta } ) ^ { * } : L _ { 2 } ( \mathbb { R } ^ { d } \times [ 0 , 1 ] ) \to \mathbb { R } ^ { p }$ its adjoint. Hence the condition $\nabla _ { \theta } \mathcal { L } _ { \theta } = 0$ is weaker than $\begin{array} { r } { \frac { \delta } { \delta u } \mathcal { L } _ { \mathrm { W A M } } ( u _ { \theta } ) = 0 : ( \nabla _ { \theta } u _ { \theta } ) ^ { * } } \end{array}$ is not injective, since a finite-dimensional parameter $\theta ~ \in ~ \mathbb { R } ^ { p }$ cannot move the control in every direction.

Parametrization of the control. The pretrained flow matching model is a deep neural network $v ^ { \mathrm { b a s e } } =$ $v _ { \theta _ { 0 } }$ with parameters $\theta _ { 0 } \in \mathbb { R } ^ { p }$ We choose to parametrize the control u as

$$
u _ { \theta } ( x , t ) = \frac { 2 } { \gamma ( t ) } ( v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( x , t ) - v ^ { \mathrm { b a s e } } ( x , t ) ) ,
$$

with $v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( x , t ) = v _ { \theta } ( x , t )$ and we initialize with $\theta =$ $\theta _ { 0 } , { \mathrm { i . e . , ~ } } u _ { \theta } = 0$ . The loss is therefore

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { W A M } } ( \theta ) : = \mathbb { E } _ { \mathbf { X } \sim p ^ { \bar { u } _ { \theta } } } \bigg [ \ ( 2 \mathrm { d } \mathbf { \Omega } } \\ & { \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left\| 2 \gamma ( t ) ^ { - 1 } ( v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( X _ { t } , t ) - v ^ { \mathrm { b a s e } } ( X _ { t } , t ) ) + \frac { \gamma ( t ) \tilde { a } ( t , \mathbf { X } ) } { w ( t ) } \right\| ^ { 2 } d t \bigg ] . } \end{array}\tag{の}
$$

For the classical choice of schedules $\alpha _ { t } = t$ and $\beta _ { t } = 1 -$ $t , \gamma ( t ) = \textstyle { \sqrt { \frac { 2 ( 1 - t ) } { t } } }$ , so the factors $\gamma ( t )$ and $\gamma ( t ) ^ { - 1 }$ blow up as $t  0$ and $t  1$ . Consequently, we observed numerical instability when using (20). It turns out that this can be solved by adding a time-dependent weight $\lambda ( t )$

Adding an additional time-dependent weight for numerical stability. In the time integral of the loss (18), we can insert a positive time-dependent weight $\lambda ( t ) > 0$ without changing the critical point, leading to the loss:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { W A M } } ^ { \lambda } ( u ) : = } \\ & { \mathbb { E } _ { \mathbf { X } \sim p ^ { \bar { u } } } \left[ \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \lambda ( t ) \left. u ( X _ { t } , t ) + \frac { \gamma ( t ) \widetilde { a } ( t , \mathbf { X } ) } { w ( t ) } \right. ^ { 2 } d t \right] . } \end{array}\tag{21}
$$

For the parametrization of the control above, taking $\lambda ( t ) = 2 ( 1 - t ) t$ cancels the exploding factors $\gamma ( t )$ and $\gamma ( t ) ^ { - 1 }$ of (20):

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { W A M } } ^ { \lambda } ( \theta ) = \mathbb { E } _ { \mathbf { X } \sim p ^ { \bar { u } _ { \theta } } } \bigg [ } \\ & { \qquad 2 \int _ { 0 } ^ { 1 } \bigg \| t ( v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( X _ { t } , t ) - v ^ { \mathrm { b a s e } } ( X _ { t } , t ) ) + ( 1 - t ) \frac { \tilde { a } ( t , \mathbf { X } ) } { w ( t ) } \bigg \| ^ { 2 } d t \bigg ] . } \end{array}
$$

The algorithm is then obtained by discretizing the generative SDE (11) and the adjoint ODE (19) and estimating the expectation over a batch of $M \in \mathbb { N }$ trajectories. It is summarized in Algorithm 1 in $\mathrm { A p \mathrm { - } }$ pendix D.5.

## 5 EXPERIMENTS

We ran several experiments to demonstrate how weighted adjoint matching achieves a better tradeoff between reward maximization and balance.4

The optimal control of the weighted SOC problem (10) is invariant under the rescaling $( w , a ) \gets ( c w , c a )$ for $c > 0$ , so the value of a is only meaningful relative to w. Adjoint matching corresponds to choosing $w ( t ) = 1$ constant, while weighted adjoint matching uses a varying w. We fix the normalization $w ( 1 ) = 1$ for all our weights. Since different weights w still regularize differently at the same a, we plot the quantities of interest against the achieved average reward $R = \mathbb { E } [ r ( X _ { 1 } ^ { u _ { \theta } } ) ]$ rather than against a.

## 5.1 Gaussian mixture

![](images/0e46a18636f44ced7eabc3dc00b0a42b0eea0333ca156396900aba9c1a613ac7.jpg)  
(a) Exponential weight wexponential(t).

![](images/6924d97dc5d425a33e0b85b6e9deec68a23cb0b88789203250409e09c8f35668.jpg)  
(b) Piecewise constant weight $w _ { \mathrm { s t e p s } } ( t )$  
Figure 3: Average reward with respect to weight of positive mode for adjoint matching and weighted adjoint matching, using different functions $t \mapsto w ( t )$

We use the Gaussian mixture defined in Section 1

$$
\begin{array} { r } { p _ { \mathrm { t a r g e t } } ( x ) = \frac { 1 } { 2 } \mathcal { N } ( x ; - \mu , \sigma _ { 1 } ^ { 2 } I ) + \frac { 1 } { 2 } \mathcal { N } ( x ; \mu , \sigma _ { 2 } ^ { 2 } I ) . } \end{array}
$$

We train a flow matching model to sample from $p _ { \mathrm { t a r g e t } } ,$ then use both adjoint matching [35] and our weighted adjoint matching to finetune the model with the reward $\begin{array} { r } { r ( x ) = - { \frac { \| \bar { x } _ { 2 } \| ^ { 2 } } { 2 } } } \end{array}$ , which encourages samples to be closer to the first axis. We take $\mu = ( 2 , 0 ) , \sigma _ { 1 } ^ { 2 } = 0 . 2 5$ and $\sigma _ { 2 } ^ { 2 } = 0 . 5$ and different values of a. For weighted adjoint matching, we use different types of weighting function w: an exponential function $w _ { \mathrm { e x p o n e n t i a l } } ( t ) =$ $e ^ { \alpha ( 1 - t ) } , \ \alpha \ > \ 0$ , and a piecewise constant function $w _ { \mathrm { s t e p s } } ( t ) = \left\{ \begin{array} { l l } { 1 0 ^ { 5 } \ \mathrm { i f } \ t < t _ { s } } \\ { 1 \ \mathrm { i f } \ t \geq t _ { s } } \end{array} , \ t _ { s } \in ( 0 , 1 ) \right.$ . Both regularize more the beginning of the process to try to preserve balance between the two modes; $w _ { \mathrm { s t e p s } } ( t )$ takes the value $1 0 ^ { 5 }$ on $[ 0 , t _ { s } )$ to force the finetuned model to be almost identical to the base model on this interval.

In Figure 3, we plot empirical estimates of $\kappa ^ { + } =$ $\mathbb { E } [ \mathbb { 1 } _ { X _ { 1 } ^ { u _ { \theta } } \geq 0 } ]$ and $R = \mathbb { E } [ r ( X _ { 1 } ^ { u _ { \theta } } ) ]$ for different runs of the algorithms at different values of $^ { a , }$ as well as the theoretical curve for the tilted distribution (2) derived in Appendix A as a red dotted line.⁵ For the tilted distribution $p _ { a }$ , a higher reward comes with a more unbalanced distribution. We observe that adjoint matching follows the theoretical curve6, meaning that higher reward is achieved at the price of an unbalanced distribution. For high values of the reward, corresponding to high values of $^ { a , }$ the distribution is even more unbalanced than the theory predicts, which could be a consequence of optimization artifacts.

![](images/a4b63e0edf80794179e5d15442add0f9467f5f0dd26835209d7cf8064e085b5a.jpg)  
Figure 4: Average brightness reward on the MNIST dataset with respect to class count ratio $\frac { n _ { \mathrm { m i n } } } { n _ { \mathrm { m a x } } }$

On the other hand, weighted adjoint matching with both $w _ { \mathrm { e x p o n e n t i a l } }$ and $w _ { \mathrm { s t e p s } }$ gives a better tradeoff between reward and balance, with outputs in the upper right corner. With $w _ { \mathrm { s t e p s } } ,$ the reward is increased without changing the balance between the modes for $t _ { s } ~ \in ~ \{ 0 . 5 , 0 . 6 5 , 0 . 8 \}$ , while for $t _ { s } ~ = ~ 0 . 2 5$ , it remains close to adjoint matching. This is explained by the fact that at time 0.25, the two modes are not yet separated, and mass can still move between them.

## 5.2 MNIST

We train a flow matching model on MNIST [4]. We do not condition on the labels, as we use them as a proxy to evaluate how diversity is preserved. The reward is the average luminosity: for a digit represented as $x \in [ 0 , 1 ] ^ { d }$ with d = 28 × 28 = 784, $\begin{array} { r } { r ( x ) = \frac { 1 } { d } \sum _ { i } x _ { i } . } \end{array}$ We generate 50 000 samples for each finetuned model and classify them with a pretrained classifier. Denoting $n _ { \mathrm { m a x } }$ the number of elements in the largest class and $n _ { \mathrm { m i n } }$ in the smallest class, we evaluate diversity by computing the ratio $\frac { n _ { \mathrm { m i n } } } { n _ { \mathrm { m a x } } }$ . For the pretrained models, this ratio is close to that of the MNIST dataset, which is roughly $0 . 8 0 . ^ { 7 }$ Results are plotted in Figure 4. Distributions of all digits, for the different methods at an achieved reward of 0.16 and 0.18, are given in Tables 1 & 2 in the Appendix. Overall, weighted adjoint matching achieves a better tradeoff (closer to the upper right corner) between reward and class balance. For example, for a reward of 0.176, adjoint matching generates 18.0% of zeros and only 3.4% of ones, while weighted adjoint matching $\left( t _ { s } \ : = \ : 0 . 4 5 \right)$ reaches a reward of 0.180 with 10.5% of zeros and 9.1% of ones.

## 6 DISCUSSION

Wasserstein regularization. Another route would be to drop the KL regularization of (1) altogether and to regularize with the Wasserstein distance ${ \scriptstyle \frac { 1 } { a } } W _ { 2 } ^ { 2 } ( p , p _ { \mathrm { t a r g e t } } )$ instead. This penalizes how far each sample moves from its initial position, and therefore limits the transfer of mass between distant modes. This problem has an explicit solution: first sample $x \ \sim \ p _ { \mathrm { t a r g e t } }$ , then apply the proximal operator x → argmaxy $\textstyle { \overset { \smile } { r } } ( y ) - { \frac { 1 } { a } } \| x - y \| ^ { 2 }$ (see, e.g., [9, 19, 12, 46]). Every sample takes one proximal step on the reward, which roughly preserves the mode weights, but may drive samples out of the data manifold. For example, on MNIST with the brightness reward, the gradient of r is the constant white image: the background turns gray instead of the digits getting brighter.

Future work. We have focused on the case of unconditional generative modeling and simple datasets for which balance could be evaluated precisely. Future work could be dedicated to scaling this work to conditional flow matching models on large image datasets beyond MNIST and to other types of data used in practical applications.

What is a good reward? Some rewards, such as the average luminosity on MNIST of Section 5.2, are very easy to define and to interpret, but have two caveats: they can be increased by going in a direction orthogonal to the data manifold (e.g., making the black background of MNIST digits turn gray) and are not meaningful for comparing samples far apart (dark eights are brighter than bright ones). We believe that this cannot be easily solved by changing the reward. Rewards are contextual, and the values have little meaning by themselves: one needs to add the knowledge of the base distribution. KL regularization addresses the first caveat, since it forces reward maximization to take place on the data manifold, but not the second: it has no notion of distance within the support, so mass can move between distant modes. Our method, using the link between spatial resolution and time in diffusion models, addresses the second caveat.

## Acknowledgments

The authors thank Marylou Gabrié, Benjamin Dupuis Tyler Farghly, and Adam Perbost for insightful discussions related to this work. The authors are grateful to the CLEPS infrastructure from the Inria of Paris for providing resources and support This work has received support from the French government, managed by the National Research Agency, under the France 2030 program with the reference “"PR[AI]RIE-PSAI" (ANR-23-IACL-0008).

## AI use statement

In this work, we used generative AI tools to check equations and proofs of the results, and suggest corrections when appropriate. AI was also used to write and review the code of our experiments. The core implementation of Algorithm 1 was written by the authors, then reviewed and improved with AI assistance. AI tools played a larger role in the rest of the code: implementing neural nets, suggesting optimization hyperparameters, writing scripts to launch training runs and to save and retrieve checkpoints and results, generating figures from experimental data, and consolidating the codebase. Additionally, we used generative AI tools to proofread and improve the writing of the manuscript (including correcting grammar mistakes, suggesting rephrasing to improve clarity and fluency, brainstorming a title and keywords, drafting some paragraphs from bullet points, moving parts of the text to the appendix), to identify relevant literature, and to source or search for information (e.g., to answer background mathematical questions). Figure 1, which is purely illustrative, was created using Claude Design from a hand drawing by the authors. We also used AI to brainstorm and explore some research directions that do not appear in the paper. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

[1] Wendell Fleming and Raymond Rishel. Deterministic and Stochastic Optimal Control. Springer New York, 1975.

[2] U. G. Haussmann and E. Pardoux. "Time Reversal of Diffusions". In: The Annals of Probability 14.4 (Oct. 1986).

[3] Hannes Risken. The Fokker-Planck Equation: Methods of Solution and Applications. Second. 18. Springer, 1996.

[4] Yann LeCun. “The MNIST Database of Handwritten Digits". In: http://yann.lecun.com/exdb/mnist/ (1998).

[5] Pascal Vincent. “A Connection between Score Matching and Denoising Autoencoders". In:

Neural Computation 23.7 (July 2011), pp. 1661– 1674.

[6] Vladimir Bogachev, Nicolai Krylov, Michael Röckner, and Stanislav Shaposhnikov. Fokker-Planck-Kolmogorov Equations. Vol. 207. Mathematical Surveys and Monographs. American Mathematical Society, Dec. 2015.

[7] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. “Deep Unsupervised Learning Using Nonequilibrium Thermodynamics". In: International Conference on Machine Learning. June 2015.

[8] Jean-François Le Gall. Brownian Motion, Martingales, and Stochastic Calculus. Vol. 274. Graduate Texts in Mathematics. Springer International Publishing, 2016.

[9] Jose Blanchet and Karthyek R. A. Murthy. “Quantifying Distributional Model Risk via Optimal Transport". In: Mathematics of Operations Research 44.2 (2019).

[10] Yang Song and Stefano Ermon. “Generative Modeling by Estimating Gradients of the Data Distribution". In: Advances in Neural Information Processing Systems. 2019.

[11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion Probabilistic Models". In: Advances in Neural Information Processing Systems. 2020.

[12] Aman Sinha, Hongseok Namkoong, Riccardo Volpi, and John Duchi. Certifying Some Distributional Robustness with Principled Adversarial Training. arXiv:1710.10571. May 2020.

[13] Jiaming Song, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models". In: International Conference on Learning Representations. 2021.

[14] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. “Score-Based Generative Modeling through Stochastic Differential Equations". In: International Conference on Learning Representations. 2021.

[15] Tero Karras, Timo Aila, Miika Aittala, and Samuli Laine. “Elucidating the Design Space of Diffusion-Based Generative Models". In: Advances in Neural Information Processing Systems. 2022.

[16] Patrick Kidger. “On Neural Differential Equations". PhD thesis. University of Oxford, 2022.

[17] Michael S Albergo and Eric Vanden-Eijnden. “Building Normalizing Flows With Stochastic Interpolants". In: International Conference on Learning Representations. 2023.

[18] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. "DPOK: Reinforcement Learning for Fine-Tuning Text-to-Image Diffusion Models". In: Advances in Neural Information Processing Systems. 2023.

[19] Rui Gao and Anton J. Kleywegt. “Distributionally Robust Stochastic Optimization with Wasserstein Distance". In: Mathematics of Operations Research 48.2 (2023).

[20] Yaron Lipman, Ricky T Q Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. "Flow Matching for Generative Modeling". In: International Conference on Learning Representations. 2023.

[21] Xingchao Liu, Chengyue Gong, and Qiang Liu. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow". In: International Conference on Learning Representations. 2023.

[22] Gabriel Raya and Luca Ambrogioni. “Spontaneous Symmetry Breaking in Generative Diffusion Models". In: Advances in Neural Information Processing Systems. 2023.

[23] Severi Rissanen, Markus Heinonen, and Arno Solin. “Generative Modelling With Inverse Heat Dissipation". In: International Conference on Learning Representations. 2023.

[24] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation". In: Thirty-Seventh Conference on Neural Information Processing Systems. Nov. 2023.

[25] Roberto Barceló, Cristóbal Alcázar, and Felipe Tobar. Avoiding Mode Collapse in Diffusion Models Fine-Tuned with Reinforcement Learning. arXiv:2410.08315. Oct. 2024.

[26] Giulio Biroli, Tony Bonnaire, Valentin De Bortoli, and Marc Mézard. "Dynamical Regimes of Diffusion Models". In: Nature Communications 15.1 (2024), p. 9957.

[27] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. “Training Diffusion Models with Reinforcement Learning". In: International Conference on Learning Representations. 2024.

[28] Kevin Clark, Paul Vicol, Kevin Swersky, and David J. Fleet. "Directly Fine-Tuning Diffusion Models on Differentiable Rewards". In: International Conference on Learning Representations. 2024.

[29] Sander Dieleman. Diffusion Is Spectral Autoregression. https://sander.ai/2024/09/02/spectralautoregression.html. Sept. 2024.

[30] Carles Domingo-Enrich, Jiequn Han, Brandon Amos, Joan Bruna, and Ricky T Q Chen. "Stochastic Optimal Control Matching". In: Advances in Neural Information Processing Systems. 2024.

[31] Marvin Li and Sitan Chen. “Critical Windows: Non-Asymptotic Theory for Feature Emergence in Diffusion Models". In: International Conference on Machine Learning. 2024.

[32] Masatoshi Uehara, Yulai Zhao, Tommaso Biancalani, and Sergey Levine. Understanding Reinforcement Learning-Based Fine-Tuning of Diffusion Models: A Tutorial and Review. arXiv:2407.13734. July 2024.

[33] Masatoshi Uehara, Yulai Zhao, Kevin Black, EhsanHajiramezanali,GabrieleScalia, Nathaniel Lee Diamant, Alex M. Tseng, Tommaso Biancalani, and Sergey Levine. Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control. arXiv:2402.15194. Feb. 2024.

[34] Binxu Wang and John J. Vastola. Diffusion Models Generate Images Like Painters: An Analytical Theory of Outline First, Details Later. arXiv:2303.02490. Mar. 2024.

[35] Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. "Adjoint Matching: Fine-Tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control". In: International Conference on Learning Representations. 2025.

[36] Yinbin Han, Meisam Razaviyayn, and Renyuan Xu. “Stochastic Control for Fine-Tuning Diffusion Models: Optimality, Regularity, and Convergence". In: International Conference on Machine Learning. 2025.

[37] Pierre Marion, Anna Korba, Peter Bartlett, Mathieu Blondel, Valentin De Bortoli, Arnaud Doucet, Felipe Llinares-López, Courtney Paquette, and Quentin Berthet. "Implicit Diffusion: Efficient Optimization through Stochastic Sampling". In: International Conference on Artificial Intelligence and Statistics. 2025.

[38] Antonio Sclocchi, Alessandro Favero, and Matthieu Wyart. "A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data". In: Proceedings of the National Academy of Sciences 122.1 (2025).

[39] Beatrice Achilli, Marco Benedetti, Giulio Biroli, and Marc Mézard. "Theory of Speciation Transitions in Diffusion Models with General Class Structure". In: Journal of Statistical Mechanics: Theory and Experiment 2026.4 (2026).

[40] Carles Domingo-Enrich and Jiequn Han. “Adjoint Matching through the Lens of the Stochastic Maximum Principle in Optimal Control". In: Transactions on Machine Learning Research (2026).

[41] Zhengyi Guo, Jiayuan Sheng, David D. Yao, and Wenpin Tang. Improved Techniques for Fine-Tuning Flow Models via Adjoint Matching: A Deterministic Control Pipeline. arXiv:2605.06583. May 2026.

[42] Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang, Dacheng Yin, Fengyun Rao, and Bo Zhang. "TempFlow-GRPO: When Timing Matters for GRPO in Flow Models". In: International Conference on Learning Representations. 2026.

[43] Rishi Leburu, Levon Nurbekyan, and Lars Ruthotto. Differentiating through Stochastic Differential Equations: A Primer. arXiv:2601.08594. Jan. 2026.

[44] Jeongjae Lee and Jong Chul Ye. “PCPO: Proportionate Credit Policy Optimization for Preference Alignment of Image Generation Models". In: International Conference on Learning Representations. 2026.

[45] Henglin Liu, Huijuan Huang, Jing Wang, Chang Liu, Xiu Li, and Xiangyang Ji. “DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware GRPO". In: Conference on Computer Vision and Pattern Recognition (CVPR). 2026.

[46] Ankur Moitra, Andrej Risteski, and Dhruv Rohatgi. The Tractability Landscape of Diffusion Alignment: Regularization, Rewards, and Computational Primitives. arXiv:2605.11361. May 2026.

[47] Peter Potaptchik, Cheuk-Kit Lee, and Michael S. Albergo. “Tilt Matching for Scalable Sampling and Fine-Tuning". In: International Conference on Machine Learning. 2026.

# Weighted Adjoint Matching: Finetuning Flow Matching Models with Time-Dependent Control Cost Supplementary Materials

## A GAUSSIAN MIXTURE WITH KL-REGULARIZED OBJECTIVE

We compute the tilted distribution $p _ { a }$ of a two-component Gaussian mixture in $\mathbb { R } ^ { 2 }$ and show that it does not preserve the weights of the modes. We define $p _ { \mathrm { t a r g e t } }$ as a mixture of two modes

$$
p _ { \mathrm { t a r g e t } } ( x ) = \kappa _ { 0 } ^ { 1 } \mathcal { N } \big ( x ; \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } I \big ) + \kappa _ { 0 } ^ { 2 } \mathcal { N } \big ( x ; \mu _ { 2 } , \sigma _ { 2 } ^ { 2 } I \big ) = \kappa _ { 0 } ^ { 1 } \big ( 2 \pi \sigma _ { 1 } ^ { 2 } \big ) ^ { - 1 } e ^ { - \frac { \| x - \mu _ { 1 } \| ^ { 2 } } { 2 \sigma _ { 1 } ^ { 2 } } } + \kappa _ { 0 } ^ { 2 } \big ( 2 \pi \sigma _ { 2 } ^ { 2 } \big ) ^ { - 1 } e ^ { - \frac { \| x - \mu _ { 2 } \| ^ { 2 } } { 2 \sigma _ { 2 } ^ { 2 } } } ,
$$

with $\mu _ { 1 } = ( - m , 0 ) , \mu _ { 2 } = ( m , s ) , m , s \in \mathbb { R } , \sigma _ { 1 } , \sigma _ { 2 } > 0$ and $\kappa _ { 0 } ^ { 1 } , \kappa _ { 0 } ^ { 2 } > 0$ such that $\kappa _ { 0 } ^ { 1 } + \kappa _ { 0 } ^ { 2 } = 1$ . For $\boldsymbol { x } = ( x _ { 1 } , x _ { 2 } )$ , we define the reward $\begin{array} { r } { r ( x ) = - { \frac { \| x _ { 2 } \| ^ { 2 } } { 2 } } } \end{array}$ to encourage samples to be closer to the first axis. The tilted distribution is

$$
p _ { a } ( x ) = Z _ { a } ^ { - 1 } p _ { \mathrm { t a r g e t } } ( x ) e ^ { a r ( x ) } ,
$$

with $Z _ { a } = \mathbb { E } _ { p _ { \mathrm { t a r g e t } } } [ e ^ { a r ( x ) } ]$ . It can be rewritten as:

$$
\begin{array} { r } { p _ { a } ( x ) = \displaystyle \frac { \kappa _ { 0 } ^ { 1 } } { Z _ { a } } \displaystyle \frac { \operatorname* { d e t } ( \Sigma _ { 1 , a } ) ^ { 1 / 2 } } { \sigma _ { 1 } ^ { 2 } } \displaystyle \frac { \exp \left( - \frac { \| x - \mu _ { 1 } \| _ { \Sigma _ { 1 , a } ^ { - 1 } } ^ { 2 } } { 2 } \right) } { 2 \pi \operatorname* { d e t } ( \Sigma _ { 1 , a } ) ^ { 1 / 2 } } + \displaystyle \frac { \kappa _ { 0 } ^ { 2 } } { Z _ { a } } \displaystyle \frac { \operatorname* { d e t } ( \Sigma _ { 2 , a } ) ^ { 1 / 2 } \exp \left( - \frac { a s ^ { 2 } } { 2 ( 1 + a \sigma _ { 2 } ^ { 2 } ) } \right) } { \sigma _ { 2 } ^ { 2 } } \displaystyle \frac { \exp \left( - \frac { \| x - \mu _ { 2 , a } \| _ { \Sigma _ { 2 , a } ^ { - 1 } } ^ { 2 } } { 2 } \right) } { 2 \pi \operatorname* { d e t } ( \Sigma _ { 2 , a } ) ^ { 1 / 2 } } } \\ { = \kappa _ { a } ^ { 1 } \mathcal { N } ( x ; \mu _ { 1 } , \Sigma _ { 1 , a } ) + \kappa _ { a } ^ { 2 } \mathcal { N } ( x ; \mu _ { 2 , a } , \Sigma _ { 2 , a } ) } \end{array}
$$

with $\| x \| _ { A } ^ { 2 } = x ^ { \top } A x$ and for $i \in \{ 1 , 2 \}$

$$
\Sigma _ { i , a } = \left[ \begin{array} { c c } { { \sigma _ { i } ^ { 2 } } } & { { 0 } } \\ { { 0 } } & { { \frac { \sigma _ { i } ^ { 2 } } { 1 + a \sigma _ { i } ^ { 2 } } } } \end{array} \right] , \quad \mu _ { 2 , a } = \left( m , \frac { s } { 1 + a \sigma _ { 2 } ^ { 2 } } \right) .
$$

We recognize a mixture of two Gaussian distributions with $\kappa _ { a } ^ { 1 }$ the weight of the first mode and $\kappa _ { a } ^ { 2 }$ the weight of the second mode equal to

$$
\begin{array} { l } { \kappa _ { a } ^ { 1 } = \displaystyle \frac { \kappa _ { 0 } ^ { 1 } } { Z _ { a } } \frac { \operatorname* { d e t } \left( \Sigma _ { 1 , a } \right) ^ { 1 / 2 } } { \sigma _ { 1 } ^ { 2 } } , } \\ { \kappa _ { a } ^ { 2 } = \displaystyle \frac { \kappa _ { 0 } ^ { 2 } } { Z _ { a } } \frac { \operatorname* { d e t } \left( \Sigma _ { 2 , a } \right) ^ { 1 / 2 } } { \sigma _ { 2 } ^ { 2 } } \exp \left( - \frac { a s ^ { 2 } } { 2 ( 1 + a \sigma _ { 2 } ^ { 2 } ) } \right) , } \end{array}
$$

from which we deduce that

$$
\kappa _ { a } = \frac { \kappa _ { a } ^ { 1 } } { \kappa _ { a } ^ { 2 } } = \kappa _ { 0 } \sqrt { \frac { 1 + a \sigma _ { 2 } ^ { 2 } } { 1 + a \sigma _ { 1 } ^ { 2 } } } \exp \left( \frac { a s ^ { 2 } } { 2 ( 1 + a \sigma _ { 2 } ^ { 2 } ) } \right) , \quad \kappa _ { 0 } = \frac { \kappa _ { 0 } ^ { 1 } } { \kappa _ { 0 } ^ { 2 } } ,
$$

$$
\kappa _ { a } ^ { 1 } = \frac { \kappa _ { a } } { 1 + \kappa _ { a } } = \frac { \kappa _ { 0 } ^ { 1 } \sqrt { 1 + a \sigma _ { 2 } ^ { 2 } } \exp \left( \frac { a s ^ { 2 } } { 2 ( 1 + a \sigma _ { 2 } ^ { 2 } ) } \right) } { \kappa _ { 0 } ^ { 2 } \sqrt { 1 + a \sigma _ { 1 } ^ { 2 } } + \kappa _ { 0 } ^ { 1 } \sqrt { 1 + a \sigma _ { 2 } ^ { 2 } } \exp \left( \frac { a s ^ { 2 } } { 2 ( 1 + a \sigma _ { 2 } ^ { 2 } ) } \right) } ,
$$

$$
\kappa _ { a } ^ { 2 } = \frac { 1 } { 1 + \kappa _ { a } } = \frac { \kappa _ { 0 } ^ { 2 } \sqrt { 1 + a \sigma _ { 1 } ^ { 2 } } } { \kappa _ { 0 } ^ { 2 } \sqrt { 1 + a \sigma _ { 1 } ^ { 2 } } + \kappa _ { 0 } ^ { 1 } \sqrt { 1 + a \sigma _ { 2 } ^ { 2 } } \exp \left( \frac { a s ^ { 2 } } { 2 ( 1 + a \sigma _ { 2 } ^ { 2 } ) } \right) } .
$$

In particular, the ratio depends on a, and as $a  + \infty$ , the tilted distribution converges to a mixture of two degenerate Gaussian distributions (a projection on the first axis defined by the equation $x _ { 2 } = 0 )$ with a weight ratio

$$
\kappa _ { \infty } = \operatorname* { l i m } _ { a \to + \infty } \kappa _ { a } = \kappa _ { 0 } \frac { \sigma _ { 2 } } { \sigma _ { 1 } } \exp \left( \frac { s ^ { 2 } } { 2 \sigma _ { 2 } ^ { 2 } } \right) .
$$

We also define $\kappa _ { a } ^ { - } = \mathbb { E } _ { p _ { a } } [ \mathbb { 1 } _ { x _ { 1 } \leq 0 } ]$ and $\kappa _ { a } ^ { + } = \mathbb { E } _ { p _ { a } } [ \mathbb { 1 } _ { x _ { 1 } > 0 } ] = 1 - \kappa _ { a } ^ { - }$ , which will be close to $\kappa _ { a } ^ { 1 }$ and $\kappa _ { a } ^ { 2 }$ if the modes are well separated, but easier to estimate empirically:

$$
\begin{array} { r l } & { \kappa _ { a } ^ { - } = \mathbb { E } _ { p _ { a } } [ \mathbb { 1 } _ { x _ { 1 } \leq 0 } ] } \\ & { \quad = \kappa _ { a } ^ { 1 } \mathbb { E } _ { x \sim \mathcal { N } ( \mu _ { 1 } , \Sigma _ { 1 , a } ) } [ \mathbb { 1 } _ { x _ { 1 } \leq 0 } ] + \kappa _ { a } ^ { 2 } \mathbb { E } _ { x \sim \mathcal { N } ( \mu _ { 2 , a } , \Sigma _ { 2 , a } ) } [ \mathbb { 1 } _ { x _ { 1 } \leq 0 } ] } \\ & { \quad = \kappa _ { a } ^ { 1 } \Phi \left( \frac { m } { \sigma _ { 1 } } \right) + \kappa _ { a } ^ { 2 } \Phi \left( \frac { - m } { \sigma _ { 2 } } \right) , } \\ & { \quad \kappa _ { a } ^ { + } = \kappa _ { a } ^ { 1 } \Phi \left( \frac { - m } { \sigma _ { 1 } } \right) + \kappa _ { a } ^ { 2 } \Phi \left( \frac { m } { \sigma _ { 2 } } \right) , } \end{array}
$$

where $\begin{array} { r } { \Phi ( x ) = \frac { 1 } { \sqrt { 2 \pi } } \int _ { - \infty } ^ { x } e ^ { - t ^ { 2 } / 2 } d t } \end{array}$ is the cumulative distribution function (CDF) of the standard normal distribution. Finally, we compute the optimal reward $R _ { a }$ achieved by $p _ { a }$

$$
\begin{array} { l } { { R _ { a } = \mathbb { E } _ { p _ { a } } [ r ( X ) ] } } \\ { { \ = \kappa _ { a } ^ { 1 } \mathbb { E } _ { \mathcal { N } ( \mu _ { 1 } , \Sigma _ { 1 , a } ) } [ r ( X ) ] + \kappa _ { a } ^ { 2 } \mathbb { E } _ { \mathcal { N } ( \mu _ { 2 , a } , \Sigma _ { 2 , a } ) } [ r ( X ) ] } } \\ { { \ = - \displaystyle \frac { 1 } { 2 } \left( \kappa _ { a } ^ { 1 } \displaystyle \frac { \sigma _ { 1 } ^ { 2 } } { 1 + a \sigma _ { 1 } ^ { 2 } } + \kappa _ { a } ^ { 2 } \left( \displaystyle \frac { \sigma _ { 2 } ^ { 2 } } { 1 + a \sigma _ { 2 } ^ { 2 } } + \left( \displaystyle \frac { s } { 1 + a \sigma _ { 2 } ^ { 2 } } \right) ^ { 2 } \right) \right) . } } \end{array}
$$

## B DIFFUSION AND FLOW MATCHING MODELS: FORWARD NOISING AND REVERSE PROCESSES

In this section, we give more details on the derivation of Section 3.

Forward process of diffusion models. The Gaussian noise corruption process [see, e.g., 14, 15] is defined by

$$
\left\{ \begin{array} { l l } { d \vec { X } _ { \tau } = - f ( \tau ) \vec { X } _ { \tau } d \tau + g ( \tau ) d B _ { \tau } , } \\ { \vec { X } _ { 0 } = X , } \end{array} \right.\tag{4}
$$

where $f , g : \mathbb { R } \mapsto \mathbb { R } _ { + }$ are continuous functions. To derive the closed-form solution of (4), denote $Y _ { \tau } = a ( \tau ) ^ { - 1 } \vec { X } _ { \tau }$ ， with $\begin{array} { r } { a ( \tau ) = \exp \left( - \int _ { 0 } ^ { \tau } f ( u ) d u \right) } \end{array}$ . Itô's formula [see, e.g., 8] gives

$$
d Y _ { \tau } = f ( \tau ) a ( \tau ) ^ { - 1 } \overrightarrow { X } _ { \tau } d \tau + a ( \tau ) ^ { - 1 } d \overrightarrow { X } _ { \tau } = \frac { g ( \tau ) } { a ( \tau ) } d B _ { \tau } ,
$$

hence

$$
Y _ { \tau } = a ( 0 ) ^ { - 1 } \overrightarrow { X } _ { 0 } + \int _ { 0 } ^ { \tau } \frac { g ( u ) } { a ( u ) } d B _ { u } = X + \int _ { 0 } ^ { \tau } \frac { g ( u ) } { a ( u ) } d B _ { u } ,
$$

leading to the closed-form solution,

$$
\vec { X } _ { \tau } = a ( \tau ) X + a ( \tau ) \int _ { 0 } ^ { \tau } \frac { g ( u ) } { a ( u ) } d B _ { u } .
$$

In particular, for each time τ, the joint distribution $\mathcal { L } ( ( \vec { X } _ { 0 } , \vec { X } _ { \tau } ) )$ satisfies

$$
\mathcal { L } \left( ( \vec { X } _ { 0 } , \vec { X } _ { \tau } ) \right) = \mathcal { L } \left( ( X , a ( \tau ) X + b ( \tau ) Z ) \right)
$$

with $Z \sim { \mathcal { N } } ( 0 , I )$ independent from X and $\begin{array} { r } { b ( \tau ) = a ( \tau ) \sqrt { \int _ { 0 } ^ { \tau } \frac { g ( u ) ^ { 2 } } { a ( u ) ^ { 2 } } d u } } \end{array}$

Backward process. The idea behind diffusion models is to fix a large $T > 0 ,$ such that $\vec { X } _ { T _ { \vphantom { \vphantom { \vphantom { \vphantom { \vphantom { \vphantom { \vphantom { \sum } } } } } } } } }$ is close to a standard normal distribution, and then to find a process that has the same marginal laws as $( \overrightarrow { X } _ { T - t } ) _ { t \in [ 0 , T ] }$ to map a Gaussian to the target distribution. The marginal laws of $\vec { X } _ { \tau }$ are characterized by the Fokker-Planck equation [see, e.g., 3, 6] that governs the evolution of the density $\vec { p _ { \tau } }$ of $\vec { X } _ { \tau }$

$$
\partial _ { \tau } \vec { p } _ { \tau } = - \nabla \cdot ( - f ( \tau ) x \vec { p } _ { \tau } ) + \frac { 1 } { 2 } g ( \tau ) ^ { 2 } \Delta \vec { p } _ { \tau } .
$$

Reversing time with $t = T - \tau , \stackrel {  } { p _ { t } } = \stackrel {  } { p } _ { T - t } , \stackrel {  } { f } ( t ) = f ( T - t ) , \stackrel {  } { g } ( t ) = g ( T - t )$ , gives

$$
\partial _ { t } \overleftarrow { p } _ { t } = - \nabla \cdot ( \overleftarrow { f } ( t ) x \overleftarrow { p } _ { t } ) - \frac { 1 } { 2 } \overleftarrow { g } ( t ) ^ { 2 } \Delta \overleftarrow { p } _ { t } ,
$$

but here the term in front of the Laplacian $\Delta$ is negative, so this cannot be read as the Fokker-Planck equation of an SDE. The trick is to introduce $\gamma : \mathbb { R } \to \mathbb { R } ^ { + }$ , and use $\Delta \overleftarrow { p _ { t } } = \nabla \cdot ( \overleftarrow { p _ { t } } \nabla \log \overleftarrow { p _ { t } } )$ to get

$$
\partial _ { t } \overleftarrow { p } _ { t } = - \nabla \cdot \left( \left( \overleftarrow { f } ( t ) x + \frac { \gamma ( t ) ^ { 2 } + \overleftarrow { g } ( t ) ^ { 2 } } { 2 } \nabla \log \overleftarrow { p } _ { t } \right) \overleftarrow { p } _ { t } \right) + \frac { \gamma ( t ) ^ { 2 } } { 2 } \Delta \overleftarrow { p } _ { t }
$$

With a positive term in front of $\Delta ,$ this is the Fokker-Planck equation of the process $\overleftarrow { X } _ { t }$ defined by

$$
\left\{ \begin{array} { l l } { d \overleftarrow { X } _ { t } } & { = \left( \overleftarrow { f } ( t ) \overleftarrow { X } _ { t } + \frac { \gamma ( t ) ^ { 2 } + \overleftarrow { \mathfrak { Y } } ( t ) ^ { 2 } } { 2 } \nabla \log \overleftarrow { p } _ { t } ( \overleftarrow { X } _ { t } ) \right) d t + \gamma ( t ) d W _ { t } , } \\ { \overleftarrow { X } _ { 0 } } & { = \overrightarrow { X } _ { T } , } \end{array} \right.\tag{5}
$$

with $W _ { t }$ a Brownian motion. Hence this process satisfies $\forall t \in [ 0 , T ] , \mathcal { L } ( \overleftarrow { X } _ { t } ) = \mathcal { L } ( \overrightarrow { X } _ { T - t } )$

Note that the Fokker-Planck equation can only ensure the equality of the marginal laws. However, a result by Haussmann and Pardoux [2] proves that taking $\gamma ( t ) ^ { 2 } = \overleftarrow { g } ( t ) ^ { 2 }$ also preserves the path distribution $\mathcal { L } ( ( \overline { { X } } _ { t } ) _ { t \in [ 0 , T ] } ) =$ $\mathcal { L } ( ( \overrightarrow { X } _ { T - t } ) _ { t \in [ 0 , T ] } )$

Flow matching and the memoryless noise schedule. Domingo-Enrich et al. [35] generalized flow matching (6) by adding a diffusion coefficient $\gamma : [ 0 , 1 ] \mapsto \mathbb { R }$ as

$$
\left\{ \begin{array} { l l } { d X _ { t } } & { = b ^ { \gamma } ( X _ { t } , t ) d t + \gamma ( t ) d B _ { t } } \\ { X _ { 0 } } & { \sim \mathcal { N } ( 0 , I ) , } \end{array} \right. \quad \mathrm { w i t h } \ b ^ { \gamma } ( x , t ) = v ( x , t ) + \frac { \gamma ( t ) ^ { 2 } } { 2 } \nabla \log p _ { t } ( x )\tag{7}
$$

They give the closed form of the velocity field (in their Appendix B.4)

$$
v ( x , t ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x + \beta _ { t } \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \right) \nabla \log p _ { t } ( x ) ,\tag{8}
$$

hence

$$
b ^ { \gamma } ( x , t ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } x + \left( \beta _ { t } \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \right) + \frac { \gamma ( t ) ^ { 2 } } { 2 } \right) \nabla \log p _ { t } ( x ) .
$$

Using this expression, we see that (7) is just the backward SDE (5) for a specific choice of $f ( \tau )$ and $g ( \tau )$ . Indeed, taking

$$
f ( \tau ) = - \frac { d } { d \tau } \log \alpha _ { 1 - \tau } = \frac { \dot { \alpha } _ { 1 - \tau } } { \alpha _ { 1 - \tau } }
$$

$$
g ( \tau ) = \alpha _ { 1 - \tau } \sqrt { { \frac { d } { d \tau } } \left( \alpha _ { 1 - \tau } ^ { - 2 } \beta _ { 1 - \tau } ^ { 2 } \right) } = \sqrt { { 2 \beta _ { 1 - \tau } } \left( { \frac { \dot { \alpha } _ { 1 - \tau } } { \alpha _ { 1 - \tau } } } \beta _ { 1 - \tau } - \dot { \beta } _ { 1 - \tau } \right) } ,
$$

gives

$$
a ( \tau ) = \exp \left( - \int _ { 0 } ^ { \tau } f ( u ) d u \right) = \exp \left( - \int _ { 0 } ^ { \tau } - \frac { d } { d u } \log \alpha _ { 1 - u } d u \right) = \frac { \alpha _ { 1 - \tau } } { \alpha _ { 1 } } = \alpha _ { 1 - \tau }
$$

and

$$
b ( \tau ) = a ( \tau ) \sqrt { \int _ { 0 } ^ { \tau } \frac { g ( u ) ^ { 2 } } { a ( u ) ^ { 2 } } d u } = \alpha _ { 1 - \tau } \sqrt { \int _ { 0 } ^ { \tau } \frac { d } { d u } \left( \alpha _ { 1 - u } ^ { - 2 } \beta _ { 1 - u } ^ { 2 } \right) d u } = \alpha _ { 1 - \tau } \sqrt { \alpha _ { 1 - \tau } ^ { - 2 } \beta _ { 1 - \tau } ^ { 2 } - \alpha _ { 1 } ^ { - 2 } \beta _ { 1 } ^ { 2 } } = \beta _ { 1 - \tau } .
$$

The forward process $\vec { X } _ { \tau }$ has therefore marginals $\mathcal { L } ( a ( \tau ) X + b ( \tau ) Z ) , Z \sim \mathcal { N } ( 0 , I )$ independent from X, going from data $\tau = 0 , a ( 0 ) = \alpha _ { 1 } = 1 , b ( 0 ) = \beta _ { 1 } = 0$ to pure noise at time $T = 1$ , with $a ( 1 ) = \alpha _ { 0 } = 0 , b ( 1 ) = \beta _ { 0 } = 1$ The backward process $\overleftarrow { X } _ { t } \ ( 5 )$ , plugging the values $\begin{array} { r } { \overleftarrow { a } ( t ) = a ( 1 - t ) = \alpha _ { t } , \overleftarrow { b } ( t ) = b ( 1 - t ) = \beta _ { t } , \overleftarrow { f } ( t ) = \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } } \end{array}$ and $\begin{array} { r } { \overleftarrow { g } ( t ) = \sqrt { 2 \beta _ { t } \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \right) } } \end{array}$ , is then simply (7).

The flow matching SDE of Domingo-Enrich et al. [35] is therefore just a diffusion model with a particular choice of time schedule. Moreover, the choice of memoryless noise schedule is

$$
\gamma ( t ) = \sqrt { 2 \beta _ { t } \left( \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } \beta _ { t } - \dot { \beta } _ { t } \right) } = \overleftarrow { g } ( t ) ,
$$

which is exactly the choice of $\gamma ( t )$ that also conserves the path distribution $\mathcal { L } ( ( \overleftarrow { X } _ { t } ) _ { t \in [ 0 , T ] } ) = \mathcal { L } ( ( \overrightarrow { X } _ { T - t } ) _ { t \in [ 0 , T ] } )$ according to Haussmann and Pardoux [2].

## C PATH-KL REGULARIZED OBJECTIVE

This appendix and Appendix D are written for the weighted SOC problem on a time horizon $T > 0$ , with an arbitrary running cost $f : \mathbb { R } ^ { d } \times [ 0 , T ] $ R and terminal cost $g : \mathbb { R } ^ { d }  \mathbb { R }$

$$
\operatorname* { m i n } _ { u \in \mathcal { U } } \mathbb { E } \left[ \int _ { 0 } ^ { T } \left( \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u } , t ) \right) d t + g ( X _ { T } ^ { u } ) \right]\tag{22}
$$

$$
\begin{array} { r l } { \mathrm { s . t . } } & { d X _ { t } ^ { u } = ( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) ) d t + \gamma ( t ) d B _ { t } , } \\ & { X _ { 0 } ^ { u } \sim p _ { 0 } } \end{array}\tag{23}
$$

of which the finetuning problem (10) of the main text is the particular case $T = 1 , \ f = 0$ and $g \ = \ - a r$ This section is dedicated to the proof of Proposition 1, in the general form of Proposition 4, and to a dual reformulation of (22) in which the regularization applies to the sub-paths [t, T] instead of the sub-paths [0, t]. Both reformulations rely on the following consequence of Girsanov's theorem, which expresses the quadratic control cost on a time interval as the KL divergence between the controlled and the base processes on that interval.

Proposition 3. For every $0 \leq s \leq t \leq T$ and every $u \in \mathcal { U }$

$$
\mathrm { K L } ( p ^ { u } ( X _ { [ s , t ] } | X _ { s } ) \| p ^ { \mathrm { b a s e } } ( X _ { [ s , t ] } | X _ { s } ) ) = \mathbb { E } _ { p ^ { u } } \left[ \frac { 1 } { 2 } \int _ { s } ^ { t } \| u ( X _ { r } ^ { u } , r ) \| ^ { 2 } d r \Big | X _ { s } \right] .
$$

Proof. For $s = 0$ and $t = T = 1$ , this is Theorem 2 and Appendix C.4 of Domingo-Enrich et al. [35]. The general case follows from the same argument applied to the process $( X _ { r } ^ { u } ) _ { r \in [ s , t ] }$ started at $X _ { s }$ □

In particular, $K _ { t }$ defined in (13) satisfies

$$
K _ { t } ( X _ { 0 } ) = \mathbb { E } _ { p ^ { u } } \left[ \frac { 1 } { 2 } \int _ { 0 } ^ { t } \| u ( X _ { s } ^ { u } , s ) \| ^ { 2 } d s \Big | X _ { 0 } \right] .\tag{24}
$$

## C.1 KL (0, t)-path regularization

Proposition 4. Assume that w satisfies (12). Then the weighted SOC problem (22) is equivalent to

$$
\operatorname* { m a x } _ { u \in \mathcal { U } } \mathbb { E } _ { X _ { 0 } \sim \mathcal { P } _ { 0 } } \left[ \mathbb { E } _ { \mathbf { X } ^ { u } \sim \mathcal { P } ^ { u } ( \cdot | X _ { 0 } ) } \left[ \int _ { 0 } ^ { T } - f ( X _ { t } ^ { u } , l ) d l - g ( X _ { T } ^ { u } ) \right] - w ( T ) \mathbb { K L } ( p ^ { u } ( \mathbf { X } | X _ { 0 } ) \| p ^ { \mathrm { b a s e } } ( \mathbf { X } | X _ { 0 } ) ) + \int _ { 0 } ^ { T } K _ { t } ( X _ { 0 } ) d w ( l ) \right] .\tag{25}
$$

Proposition 1 is the particular case $T = 1 , f = 0 , g = - a r$

Proof. Using that the weight function satisfies (12), (24), and exchanging integrals,

$$
\begin{array} { r l } { \mathbb { E } _ { p ^ { u } } \left[ \int _ { 0 } ^ { T } \displaystyle \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t \Big | X _ { 0 } \right] = w ( T ) K _ { T } ( X _ { 0 } ) - \mathbb { E } \left[ \int _ { t = 0 } ^ { T } \int _ { s = t } ^ { T } \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d w ( s ) d t \Big | X _ { 0 } \right] } & { } \\ { = w ( T ) K _ { T } ( X _ { 0 } ) - \displaystyle \int _ { s = 0 } ^ { T } \mathbb { E } \left[ \int _ { t = 0 } ^ { s } \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t \Big | X _ { 0 } \right] d w ( s ) } & { } \\ { = w ( T ) K _ { T } ( X _ { 0 } ) - \displaystyle \int _ { 0 } ^ { T } K _ { s } ( X _ { 0 } ) d w ( s ) . } & { } \end{array}
$$

Plugging this equality into (22) gives (25).

## C.2 KL (t, T)-path regularization

Considering the process between time t and T, Proposition 3 gives

$$
\widetilde { K } _ { t } ( X _ { t } ) : = \mathrm { K L } ( p ^ { u } ( X _ { [ t , T ] } | X _ { t } ) | | p ^ { \mathrm { b a s e } } ( X _ { [ t , T ] } | X _ { t } ) ) = \mathbb { E } _ { p ^ { u } } \left[ \frac { 1 } { 2 } \int _ { t } ^ { T } \| u ( X _ { s } ^ { u } , s ) \| ^ { 2 } d s \bigg | X _ { t } \right] .\tag{26}
$$

Proposition 5. Under the assumptions of Proposition $^ { 4 , }$ the weighted SOC problem (22) is also equivalent to

$$
\operatorname* { m a x } _ { u \in \mathbb { U } } \mathbb { E } _ { X _ { 0 } \sim p _ { 0 } } \left[ \mathbb { E } _ { \mathbf { X } ^ { u } \sim p ^ { u } ( \cdot | X _ { 0 } ) } \left[ - \int _ { 0 } ^ { T } f ( X _ { t } ^ { u } , t ) d t - g ( X _ { T } ^ { u } ) - \int _ { 0 } ^ { T } \ddot { K } _ { t } ( X _ { t } ) d w ( t ) \right] - w ( 0 ) \mathrm { K L } ( p ^ { u } ( \mathbf { X } | X _ { 0 } ) | | p ^ { b a s e } ( \mathbf { X } | X _ { 0 } ) ) \right] .\tag{27}
$$

Proof. With (12) and (26), as $X _ { t } ^ { u }$ is Markovian, we get

$$
\begin{array} { r l } { \mathbb { E } _ { p ^ { u } } \left[ \displaystyle \int _ { 0 } ^ { T } \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t \Big | X _ { 0 } \right] = w ( 0 ) \widetilde { K } _ { 0 } ( X _ { 0 } ) + \mathbb { E } _ { p ^ { u } } \left[ \displaystyle \int _ { t = 0 } ^ { T } \int _ { s = 0 } ^ { t } \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d w ( s ) d t \Big | X _ { 0 } \right] } & { } \\ { = w ( 0 ) \widetilde { K } _ { 0 } ( X _ { 0 } ) + \displaystyle \int _ { s = 0 } ^ { T } \mathbb { E } \left[ \displaystyle \int _ { t = s } ^ { T } \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t \Big | X _ { 0 } \right] d w ( s ) } & { } \\ { = w ( 0 ) \widetilde { K } _ { 0 } ( X _ { 0 } ) + \displaystyle \int _ { s = 0 } ^ { T } \mathbb { E } \left[ \mathbb { E } \left[ \displaystyle \int _ { t = s } ^ { T } \frac { 1 } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } d t \Big | X _ { s } ^ { u } \right] \Big | X _ { 0 } \right] d w ( s ) } & { } \\ { = w ( 0 ) \widetilde { K } _ { 0 } ( X _ { 0 } ) + \mathbb { E } \left[ \displaystyle \int _ { 0 } ^ { T } \widetilde { K } _ { s } ( X _ { s } ^ { u } ) d w ( s ) \Big | X _ { 0 } \right] . } & { } \end{array}
$$

Plugging this equality into (22) gives (27).

Compared to the non-weighted SOC problem, the KL divergence between the paths on $[ t , T ]$ enters the objective with weight $- d w ( t )$ , which for a non-decreasing w is negative and therefore adds regularization.

Example. Taking a piecewise constant function w

$$
w ( t ) = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f } } \quad t < { \frac { T } { 2 } } } \\ { 2 } & { { \mathrm { i f } } \quad t > { \frac { T } { 2 } } } \end{array} \right. }
$$

we get the equivalent objective

$$
\begin{array} { r l } & { \underset { u \in \mathcal { U } } { \operatorname* { m a x } } \mathbb { E } _ { X _ { 0 } \sim p _ { 0 } } \Bigg [ \mathbb { E } _ { \mathbf { X } ^ { u } \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } \Bigg [ \int _ { 0 } ^ { T } - f ( X _ { t } ^ { u } , t ) d t - g ( X _ { T } ^ { u } ) - \mathrm { K L } ( p ^ { u } ( X _ { [ T / 2 , T ] } \vert X _ { T / 2 } ^ { u } ) \Vert p ^ { \mathrm { b a s e } } ( X _ { [ T / 2 , T ] } \vert X _ { T / 2 } ^ { u } ) ) \Bigg ] } \\ & { \qquad - \mathrm { K L } ( p ^ { u } ( \mathbf { X } \vert X _ { 0 } ) \Vert p ^ { \mathrm { b a s e } } ( \mathbf { X } \vert X _ { 0 } ) ) \Bigg ] . } \end{array}
$$

Divergence from the base distribution on $( T / 2 , T )$ is penalized twice.

## C.3 Link between the two reformulations

Proposition 6. For every $t \in [ 0 , T ]$ 2

$$
K _ { T } ( X _ { 0 } ) = { \widetilde K } _ { 0 } ( X _ { 0 } ) , \qquad K _ { T } ( X _ { 0 } ) - K _ { t } ( X _ { 0 } ) = \mathbb { E } \left[ \left. { \widetilde K } _ { t } ( X _ { t } ^ { u } ) \right| X _ { 0 } \right] ,
$$

and the regularization terms of (25) and of (27) coincide

$$
- w ( T ) K _ { T } ( X _ { 0 } ) + \int _ { 0 } ^ { T } K _ { t } ( X _ { 0 } ) d w ( t ) = - w ( 0 ) K _ { T } ( X _ { 0 } ) - \mathbb { E } \left[ \int _ { 0 } ^ { T } \widetilde { K } _ { t } ( X _ { t } ) d w ( t ) \Big | X _ { 0 } \right] .
$$

Proof. The first equality is immediate from the definitions of $K _ { T }$ in (13) and of $\widetilde { K } _ { 0 }$ in (26). For the second one, as $X _ { t } ^ { u }$ is Markovian

$$
\begin{array} { r l } & { K _ { T } ( X _ { 0 } ) - K _ { t } ( X _ { 0 } ) = \mathbb { E } _ { \mathbf { X } ^ { u } \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } [ \log ( \frac { p ^ { u } ( X _ { [ 0 , T ] } \vert X _ { 0 } ) } { p ^ { \mathrm { b a s e } } ( X _ { [ 0 , T ] } \vert X _ { 0 } ) } ) ] - \mathbb { E } _ { \mathbf { X } ^ { u } \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } [ \log ( \frac { p ^ { u } ( X _ { [ 0 , t ] } \vert X _ { 0 } ) } { p ^ { \mathrm { b a s e } } ( X _ { [ 0 , t ] } \vert X _ { 0 } ) } ) ] } \\ & { \qquad = \mathbb { E } _ { \mathbf { X } ^ { u } \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } [ \log ( \frac { p ^ { u } ( X _ { [ 0 , T ] } \vert X _ { 0 } ) } { p ^ { u } ( X _ { [ 0 , t ] } \vert X _ { 0 } ) } \frac { p ^ { \mathrm { b a s e } } ( X _ { [ 0 , t ] } \vert X _ { 0 } ) } { p ^ { \mathrm { b a s e } } ( X _ { [ 0 , T ] } \vert X _ { 0 } ) } ) ] } \\ & { \qquad = \mathbb { E } _ { \mathbf { X } ^ { u } \sim p ^ { u } ( \cdot \vert X _ { 0 } ) } [ \log ( \frac { p ^ { u } ( X _ { [ t , T ] } \vert X _ { t } ) } { p ^ { \mathrm { b a s e } } ( X _ { [ t , T ] } \vert X _ { t } ) } ) ] = \mathbb { E } [ \log ( \frac { p ^ { u } ( X _ { [ t , T ] } \vert X _ { t } ) } { p ^ { \mathrm { b a s e } } ( X _ { [ t , T ] } \vert X _ { t } ) } ) \Big \vert X _ { 0 } ] } \\ &  \qquad = \mathbb { E } [ \mathbb { E } [ \log ( \frac { p ^ { u } ( X _ { [ t , T ] } \vert X _ { t } ) }  p ^ { \mathrm { b a s e } } ( \end{array}
$$

hence

$$
\begin{array} { l l l } { \displaystyle - w ( T ) K _ { T } ( X _ { 0 } ) + \int _ { 0 } ^ { T } K _ { t } ( X _ { 0 } ) d w ( t ) = - w ( 0 ) K _ { T } ( X _ { 0 } ) - ( w ( T ) - w ( 0 ) ) K _ { T } ( X _ { 0 } ) + \int _ { 0 } ^ { T } K _ { t } ( X _ { 0 } ) d w ( t ) } \\ { \displaystyle \qquad \quad = - w ( 0 ) K _ { T } ( X _ { 0 } ) + \int _ { 0 } ^ { T } ( K _ { t } ( X _ { 0 } ) - K _ { T } ( X _ { 0 } ) ) d w ( t ) } \\ { \displaystyle \qquad \quad = - w ( 0 ) K _ { T } ( X _ { 0 } ) - \mathbb { E } \left[ \int _ { 0 } ^ { T } \widetilde { K } _ { t } ( X _ { t } ) d w ( t ) \Big | X _ { 0 } \right] . } \end{array}
$$

## D THE WEIGHTED ADJOINT MATCHING ALGORITHM

As in Appendix C, this section is written for the weighted SOC problem with an arbitrary running cost $f$ and terminal cost $g _ { \mathrm { : } }$

$$
\operatorname* { m i n } _ { u \in \mathcal { U } } \mathbb { E } \left[ \int _ { 0 } ^ { T } \left( \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } + f ( X _ { t } ^ { u } , t ) \right) d t + g ( X _ { T } ^ { u } ) \right]\tag{22}
$$

$$
\mathrm { s . t . } \quad d X _ { t } ^ { u } = \left( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) \right) d t + \gamma ( t ) d B _ { t } ,\tag{23}
$$

$$
X _ { 0 } ^ { u } \sim p _ { 0 }
$$

The finetuning problem (10) of the main text is the particular case $T = 1 , f = 0$ and $g = - a r$ . Note also that this SOC problem and the algorithm that we propose can be defined outside the framework of flow matching. In (22), $b ( X _ { t } , t )$ can be any drift and $\gamma ( t ) \in \mathbb { R } ^ { d \times d }$ invertible, provided the process is well defined for $u = 0$ When coming back to our original problem, we will use $\gamma ( t )$ real-valued and $b = b ^ { \gamma }$

This section is dedicated to the derivation of the weighted adjoint matching algorithm and to the proof of Proposition 2. We roughly follow and adapt the original derivations by Domingo-Enrich et al. [35], though we differ when necessary or when we believe it makes the exposition clearer. We try to be as self-contained as possible. See also Domingo-Enrich et al. [30] for an earlier work that lays the groundwork for the adjoint matching method. A more recent work [40] proposes a more general Hamiltonian framework for adjoint matching that can partly retrieve our derivation of the weighted adjoint matching. It does not, however, include the extra time weight λ(t) that balances the loss between the different timesteps and gives a more stable algorithm.

## D.1 Characterization of the optimal control

Infinitesimal generator. The infinitesimal generator $L _ { t }$ associated with

$$
d X _ { t } = b ( X _ { t } , t ) d t + \gamma ( t ) d B _ { t }
$$

is defined by

$$
L _ { t } f = \operatorname* { l i m } _ { \Delta t \searrow 0 } \frac { \mathbb { E } \left[ f ( X _ { t + \Delta t } ) | X _ { t } = x | - f ( x ) \right. } { \left. \Delta t \right. }
$$

for any function f such that the limit exists. Assuming that $f \in { \mathcal { C } } ^ { 2 }$ , a second-order Taylor expansion gives

$$
\begin{array} { r l } & { f ( X _ { t + \Delta t } ) \approx f ( X _ { t } ) + \langle \nabla f ( X _ { t } ) , b ( X _ { t } , t ) \rangle \Delta t + \displaystyle \frac { 1 } { 2 } ( \gamma ( t ) Z ) ^ { \top } \nabla ^ { 2 } f ( X _ { t } ) \gamma ( t ) Z \Delta t } \\ & { ~ \approx f ( X _ { t } ) + \langle \nabla f ( X _ { t } ) , b ( X _ { t } , t ) \rangle \Delta t + \displaystyle \frac { 1 } { 2 } \operatorname { t r } \left( \gamma ( t ) ^ { \top } \nabla ^ { 2 } f ( X _ { t } ) \gamma ( t ) Z Z ^ { \top } \right) \Delta t , } \end{array}
$$

with $Z \sim { \mathcal { N } } ( 0 , I )$ , leading to

$$
L _ { t } f = \langle \nabla f ( x ) , b ( x , t ) \rangle + { \frac { 1 } { 2 } } \operatorname { t r } { \big ( } \gamma ( t ) ^ { \top } \nabla ^ { 2 } f ( x ) \gamma ( t ) { \big ) } = \langle \nabla f ( x ) , b ( x , t ) \rangle + { \frac { 1 } { 2 } } \operatorname { t r } { \big ( } \gamma ( t ) \gamma ( t ) ^ { \top } \nabla ^ { 2 } f ( x ) { \big ) } ,
$$

which we can rewrite as

$$
L _ { t } = \sum _ { i } b _ { i } ( \boldsymbol { x } , t ) \partial _ { x _ { i } } + \frac { 1 } { 2 } \sum _ { i , j } ( \gamma \gamma ^ { \top } ) _ { i j } \partial _ { x _ { i } } \partial _ { x _ { j } } .
$$

Similarly, for some control u, the infinitesimal generator $L _ { t } ^ { u }$ associated with

$$
d X _ { t } ^ { u } = ( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) ) d t + \gamma ( t ) d B _ { t }
$$

is

$$
L _ { t } ^ { u } = L _ { t } + \sum _ { i } ( \gamma ( t ) u ) _ { i } ( x , t ) \partial _ { x _ { i } }
$$

Cost and value function. The SOC problem that we consider is (22). The cost function is defined, for some control u and $t \in [ 0 , T ]$ , as

$$
J ( u , x , t ) = \mathbb { E } _ { \mathbf { X } \sim p ^ { u } } \left[ \int _ { t } ^ { T } \left( \frac { w ( s ) } { 2 } \| u ( X _ { s } , s ) \| ^ { 2 } + f ( X _ { s } , s ) \right) d s + g ( X _ { T } ) \Big | X _ { t } = x \right] ,
$$

and the value function,

$$
V ( x , t ) = \operatorname* { m i n } _ { u } J ( u , x , t ) .
$$

Verification theorem and Hamilton-Jacobi-Bellman equation. The verification theorem [see, e.g., 1, Chapter VI, Theorem 4.1] states tha $\mathrm { \Omega _ { \mathrm { 4 } } } ^ { 8 } \ V$ is characterized by, for all $( x , t ) \in \mathbb { R } ^ { d } \times [ 0 , T )$ ，

$$
\left\{ \begin{array} { l l } { \partial _ { t } V ( x , t ) + \operatorname* { m i n } _ { u \in \mathbb { R } ^ { d } } \left[ L _ { t } ^ { u } V ( x , t ) + \frac { w ( t ) } { 2 } \left. u \right. ^ { 2 } + f ( x , t ) \right] = 0 , } \\ { V ( x , T ) = g ( x ) , } \end{array} \right.
$$

or equivalently

$$
\left\{ \begin{array} { l } { ( \partial _ { t } + L _ { t } ) V ( x , t ) + \operatorname* { m i n } _ { u \in \mathbb { R } ^ { d } } \left[ \langle \gamma ( t ) u , \nabla _ { x } V ( x , t ) \rangle + \frac { w ( t ) } { 2 } \left. u \right. ^ { 2 } \right] + f ( x , t ) = 0 , } \\ { V ( x , T ) = g ( x ) . } \end{array} \right.
$$

Solving for the optimal $u ,$ we get the following equation, often referred to as the Hamilton-Jacobi-Bellman (HJB) equation,

$$
\left\{ \begin{array} { l l } { ( \partial _ { t } + L _ { t } ) V ( x , t ) - \frac { 1 } { 2 w ( t ) } \left\| \left( \gamma ^ { \top } \nabla V \right) ( x , t ) \right\| ^ { 2 } + f ( x , t ) = 0 , } \\ { V ( x , T ) = g ( x ) , } \end{array} \right.
$$

with the optimal u given by

$$
u ^ { * } ( x , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla V ( x , t ) .
$$

Using the definition of $V .$ we get

$$
\boldsymbol { u } ^ { * } ( \boldsymbol { x } , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla _ { \boldsymbol { x } } J ( \boldsymbol { u } ^ { * } , \boldsymbol { x } , t ) .\tag{15}
$$

In fact, the equation characterizes the optimal control $u ^ { * }$ , as stated by the following lemma.

Lemma 7. Assume that the control u satisfies

$$
\boldsymbol { u } ( \boldsymbol { x } , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla _ { \boldsymbol { x } } J ( \boldsymbol { u } , \boldsymbol { x } , t ) .
$$

Then u is the optimal control of (22).

Proof. We prove that if u satisfies the relation, then $J ^ { u } : ( x , t ) \mapsto J ( u , x , t )$ solves the HJB equation. Firstly, we have

$$
\begin{array} { r l } & { { L _ { t } ^ { u } } J ^ { u } ( x , \ell ) = \underset { \Delta t  0 } { \operatorname* { l i m } } \frac { \mathbb { E } [ J ( u , X _ { t + \Delta t } ^ { u } , t ) \middle | X _ { t } ^ { u } = x ] - J ( u , x , t ) } { \Delta t } } \\ & { \quad = \underset { \Delta t  0 } { \operatorname* { l i m } } \frac { \mathbb { E } [ J ( u , X _ { t + \Delta t } ^ { u } , t + \Delta t ) \middle | X _ { t } ^ { u } = x ] - J ( u , x , t ) } { \Delta t } + \mathbb { E } [ \frac { J ( u , X _ { t + \Delta t } ^ { u } , t ) - J ( u , X _ { t + \Delta t } ^ { u } , t + \Delta t ) } { \Delta t } \middle | X _ { t } ^ { u } = x ] } \\ & { \quad = \underset { \Delta t  0 } { \operatorname* { l i m } } \frac { \mathbb { E } [ J ( u , X _ { t + \Delta t } ^ { u } , t + \Delta t ) \middle | X _ { t } ^ { u } = x ] - J ( u , x , t ) } { \Delta t } - \partial _ { t } J ^ { u } ( x , t ) . } \end{array}
$$

Moreover,

$$
J ( u , x , t ) = \mathbb { E } \left[ \int _ { t } ^ { t + \Delta t } \frac { w ( s ) } { 2 } \| u ( X _ { s } , s ) \| ^ { 2 } + f ( X _ { s } , s ) d s + J ( u , X _ { t + \Delta t } , t + \Delta t ) \Bigg | X _ { t } = x \right] ,
$$

hence

$$
\operatorname* { l i m } _ { \Delta t  0 } \frac { \mathbb { E } [ J ( u , X _ { t + \Delta t } ^ { u } , t + \Delta t ) | X _ { t } ^ { u } = x ] - J ( u , x , t ) } { \Delta t } = - \frac { w ( t ) } { 2 } \| u ( x , t ) \| ^ { 2 } - f ( x , t ) .
$$

Therefore, we have

$$
L _ { t } J ^ { u } ( x , t ) = L _ { t } ^ { u } J ^ { u } ( x , t ) - \langle \gamma ( t ) u ( x , t ) , \nabla _ { x } J ^ { u } ( x , t ) \rangle = - \frac { w ( t ) } { 2 } \| u ( x , t ) \| ^ { 2 } - f ( x , t ) - \partial _ { t } J ^ { u } ( x , t ) - \langle \gamma ( t ) u ( x , t ) , \nabla _ { x } J ^ { u } ( x , t ) \rangle .
$$

For $V = J ^ { u }$ , the HJB equation becomes

$$
\begin{array} { r l r } {  { ( \partial _ { t } + L _ { t } ) J ^ { u } ( x , t ) - \frac { 1 } { 2 w ( t ) } \| ( \gamma ^ { \top } \nabla J ^ { u } ) ( x , t ) \| ^ { 2 } + f ( x , t ) = 0 } } \\ & { } & { \Leftrightarrow - \frac { w ( t ) } { 2 } \| u ( x , t ) \| ^ { 2 } -  \gamma ( t ) u ( x , t ) , \nabla _ { x } J ^ { u } ( x , t )  - \frac { 1 } { 2 w ( t ) } \| ( \gamma ^ { \top } \nabla J ^ { u } ) ( x , t ) \| ^ { 2 } = 0 } \\ & { } & { \Leftrightarrow \frac { w ( t ) } { 2 } \| u ( x , t ) + \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla J ^ { u } ( x , t ) \| ^ { 2 } = 0 } \end{array}
$$

which is satisfied under the assumption

$$
\boldsymbol { u } ( \boldsymbol { x } , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla _ { \boldsymbol { x } } J ( \boldsymbol { u } , \boldsymbol { x } , t ) .
$$

By definition, $J ^ { u }$ satisfies the boundary condition, as

$$
J ^ { u } ( x , T ) = \mathbb { E } \left[ \int _ { T } ^ { T } \frac { w ( s ) } { 2 } \| u ( X _ { s } , s ) \| ^ { 2 } + f ( X _ { s } , s ) d s + g ( X _ { T } ^ { u } ) \Big | X _ { T } ^ { u } = x \right] = g ( x ) .
$$

By the verification theorem, $J ^ { u } = V$ , and the optimal control is given by

$$
u ^ { * } ( x , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla V ( x , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla J ^ { u } ( x , t ) = u ( x , t ) .
$$

## D.2 The adjoint state

Adjoint. For some control u, we define the adjoint state as

$$
a ( t , \mathbf { X } ^ { u } , u ) = \nabla _ { X _ { t } ^ { u } } \left( \int _ { t } ^ { T } \left( \frac { w ( s ) } { 2 } \| u ( X _ { s } ^ { u } , s ) \| ^ { 2 } + f ( X _ { s } ^ { u } , s ) \right) d s + g ( X _ { T } ^ { u } ) \right)
$$

where $\mathbf { X } ^ { u }$ is the solution to $d X _ { t } ^ { u } = ( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) ) d t + \gamma ( t ) d B _ { t }$ . The adjoint state is a random variable that depends on the random trajectory $\mathbf { X } ^ { u }$ , and $\nabla _ { X _ { t } ^ { u } }$ denotes the derivative with respect to the value $X _ { t } ^ { u }$ of the process at time t, keeping the trajectory of the Brownian motion $( B _ { s } ) _ { s \geq t }$ fixed. For more details on differentiating through solutions of SDEs, see Kidger [16] and Leburu, Nurbekyan, and Ruthotto [43]. We will not discuss technical details in this article and will assume that the processes are regular enough to be differentiated. Taking the conditional expectation of the adjoint state gives the gradient of the cost function $\nabla J ( u , x , t ) = \mathbb { E } [ a ( t , \mathbf { X } ^ { u } , u ) | X _ { t } = x ]$ . To compute the adjoint, we will use the fact that it solves an ODE, as stated by the following lemma.

Lemma 8. The adjoint solves the ODE

$$
\left\{ \begin{array} { l l } { \frac { d } { d t } a ( t , \mathbf { X } ^ { u } , u ) } & { = - \left[ \nabla _ { X _ { t } ^ { u } } \left( b ( X _ { t } ^ { u } , t ) + \gamma ( t ) u ( X _ { t } ^ { u } , t ) \right) ^ { \top } a ( t , \mathbf { X } ^ { u } , u ) + \nabla _ { X _ { t } ^ { u } } \left( f ( X _ { t } ^ { u } , t ) + \frac { w ( t ) } { 2 } \| u ( X _ { t } ^ { u } , t ) \| ^ { 2 } \right) \right] , } \\ { a ( T , \mathbf { X } ^ { u } , u ) } & { = \nabla g ( X _ { T } ^ { u } ) . } \end{array} \right.\tag{28}
$$

To simplify notation, we denote in the rest of this section

$$
b _ { u } : ( x , t ) \mapsto b ( x , t ) + \gamma ( t ) u ( x , t ) ,
$$

and

$$
f _ { u } : ( x , t ) \mapsto f ( x , t ) + \frac { w ( t ) } { 2 } \| u ( x , t ) \| ^ { 2 } .
$$

Proof. The adjoint state is

$$
\begin{array} { l } { { \displaystyle a ( t , { \bf X } ^ { u } , u ) = \nabla _ { X _ { t } ^ { u } } \left( \int _ { t } ^ { T } f _ { u } ( X _ { s } ^ { u } , s ) d s + g ( X _ { T } ^ { u } ) \right) } } \\ { { \displaystyle \qquad = \int _ { t } ^ { T } ( \nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) d s + ( \nabla _ { X _ { t } ^ { u } } X _ { T } ^ { u } ) ^ { \top } \nabla _ { x } g ( X _ { T } ^ { u } ) } , } \end{array}
$$

where $\nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u }$ corresponds to the change in $X _ { s } ^ { u }$ when changing the value $X _ { t } ^ { u }$ at time t. Using the chain rule, for $h > 0$ such that $t + h \leq T$ we have

$$
\begin{array} { l } { \displaystyle a ( t , { \mathbf X } ^ { u } , u ) = \int _ { t } ^ { t + h } ( \nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) d s + \int _ { t + h } ^ { T } ( \nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) d s + ( \nabla _ { X _ { t } ^ { u } } X _ { T } ^ { u } ) ^ { \top } \nabla _ { x } g ( X _ { T } ^ { u } ) } \\ { \displaystyle ~ = \int _ { t } ^ { t + h } ( \nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) d s } \\ { \displaystyle ~ + ( \nabla _ { X _ { t } ^ { u } } X _ { t + h } ^ { u } ) ^ { \top } \left[ \int _ { t + h } ^ { T } ( \nabla _ { X _ { t + h } ^ { u } } X _ { s } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) d s + ( \nabla _ { X _ { t + h } ^ { u } } X _ { T } ^ { u } ) ^ { \top } \nabla _ { x } g ( X _ { T } ^ { u } ) \right] } \\ { \displaystyle ~ = \int _ { t } ^ { t + h } ( \nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) d s + ( \nabla _ { X _ { t } ^ { u } } X _ { t + h } ^ { u } ) ^ { \top } a ( t + h , { \mathbf X } ^ { u } , u ) } \end{array}
$$

Hence

$$
\frac { a ( t + h , \mathbf { X } ^ { u } , u ) - a ( t , \mathbf { X } ^ { u } , u ) } { h } = - \frac { 1 } { h } \int _ { t } ^ { t + h } ( \nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) d s + \left( \frac { I - \nabla _ { X _ { t } ^ { u } } X _ { t + h } ^ { u } } { h } \right) ^ { \top } a ( t + h , \mathbf { X } ^ { u } , u )\tag{29}
$$

As $\mathbf { X } ^ { u }$ is the solution to the SDE $d X _ { t } ^ { u } = b _ { u } ( X _ { t } ^ { u } , t ) d t + \gamma ( t ) d B _ { t }$

$$
X _ { t + h } ^ { u } = X _ { t } ^ { u } + \int _ { t } ^ { t + h } b _ { u } ( X _ { s } ^ { u } , s ) d s + \int _ { t } ^ { t + h } \gamma ( s ) d B _ { s } ,
$$

hence

$$
\nabla _ { X _ { t } ^ { u } } X _ { t + h } ^ { u } = I + \int _ { t } ^ { t + h } \nabla _ { x } b _ { u } ( X _ { s } ^ { u } , s ) ( \nabla _ { X _ { t } ^ { u } } X _ { s } ^ { u } ) d s ,
$$

leading to

$$
\operatorname* { l i m } _ { h \to 0 } \frac { I - \nabla _ { X _ { t } ^ { u } } X _ { t + h } ^ { u } } { h } = - \nabla _ { x } b _ { u } ( X _ { t } ^ { u } , t ) ( \nabla _ { X _ { t } ^ { u } } X _ { t } ^ { u } ) = - \nabla _ { x } b _ { u } ( X _ { t } ^ { u } , t ) .
$$

Taking $h  0$ in (29) gives

$$
\frac { d } { d t } a ( t , \mathbf { X } ^ { u } , u ) = - ( \nabla _ { X _ { t } ^ { u } } X _ { t } ^ { u } ) ^ { \top } \nabla _ { x } f _ { u } ( X _ { t } ^ { u } , t ) - \nabla _ { x } b _ { u } ( X _ { t } ^ { u } , t ) ^ { \top } a ( t , \mathbf { X } ^ { u } , u ) = - \nabla _ { x } f _ { u } ( X _ { t } ^ { u } , t ) - \nabla _ { x } b _ { u } ( X _ { t } ^ { u } , t ) ^ { \top } a ( t , \mathbf { X } ^ { u } , u )
$$

Finally,

$$
a ( T , { \bf X } ^ { u } , u ) = \nabla _ { X _ { T } ^ { u } } \left( \int _ { T } ^ { T } f _ { u } ( X _ { s } ^ { u } , s ) d s + g ( X _ { T } ^ { u } ) \right) = \nabla g ( X _ { T } ^ { u } ) .
$$

Integrating (28) gives

$$
a ( t , \mathbf { X } ^ { u } , u ) = \nabla g ( X _ { T } ^ { u } ) + \int _ { t } ^ { T } \left( \nabla _ { x } b _ { u } ( X _ { s } ^ { u } , s ) ^ { \top } a ( s , \mathbf { X } ^ { u } , u ) + \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) \right) d s
$$

Denoting $Y _ { t } = \nabla _ { x } J ( u , X _ { t } , t ) = \mathbb { E } [ a ( t , { \mathbf { X } } ^ { u } , u ) | X _ { t } ]$ and taking the conditional expectation in the equation above we have

$$
\begin{array} { r l } & { Y _ { t } = { \mathbb E } \left[ \nabla g ( X _ { T } ^ { u } ) + \displaystyle \int _ { t } ^ { T } \left( \nabla _ { x } b _ { u } ( X _ { s } ^ { u } , s ) ^ { \top } a ( s , { \mathbf X } ^ { u } , u ) + \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) \right) d s \Bigg | X _ { t } \right] } \\ & { \quad = { \mathbb E } \left[ \nabla g ( X _ { T } ^ { u } ) + \displaystyle \int _ { t } ^ { T } \left( \nabla _ { x } b _ { u } ( X _ { s } ^ { u } , s ) ^ { \top } \mathbb { E } [ a ( s , { \mathbf X } ^ { u } , u ) | X _ { s } ] + \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) \right) d s \Bigg | X _ { t } \right] } \\ & { \quad = { \mathbb E } \left[ \nabla g ( X _ { T } ^ { u } ) + \displaystyle \int _ { t } ^ { T } \left( \nabla _ { x } b _ { u } ( X _ { s } ^ { u } , s ) ^ { \top } Y _ { s } + \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) \right) d s \Bigg | X _ { t } \right] . } \end{array}\tag{30}
$$

We prove in the following lemma that equation (30) characterizes $Y _ { t }$

Lemma 9. Let $A : \mathbb { R } ^ { d } \times [ 0 , T ]  \mathbb { R } ^ { d \times d }$ and $c : \mathbb { R } ^ { d } \times [ 0 , T ]  \mathbb { R } ^ { d }$ be measurable. Assume that there is a measurable $\kappa : ( 0 , T ] \to \mathbb { R } _ { + }$ such that $\| A ( x , s ) \| \leq \kappa ( s )$ in operator norm for all $( x , s ) \in \mathbb { R } ^ { d } \times ( 0 , T ]$ , and $\begin{array} { r } { \int _ { \varepsilon } ^ { T } \kappa ( s ) d s < \infty \ f o r } \end{array}$ every $\varepsilon \in ( 0 , T ]$ . Suppose that $Y ^ { 1 }$ and $Y ^ { 2 }$ are two processes of the form $Y _ { t } ^ { i } = y ^ { i } ( X _ { t } ^ { u } , t )$ , with $y ^ { i }$ measurable and $\begin{array} { r } { \operatorname* { s u p } _ { t \in [ 0 , T ] } \mathbb { E } \| Y _ { t } ^ { i } \| < \infty } \end{array}$ , that both satisfy, for all $t \in [ 0 , T ]$ ，

$$
Y _ { t } = \mathbb { E } \left[ \nabla g ( X _ { T } ^ { u } ) + \int _ { t } ^ { T } \left( A ( X _ { s } ^ { u } , s ) ^ { \top } Y _ { s } + c ( X _ { s } ^ { u } , s ) \right) d s \middle | X _ { t } ^ { u } \right] ,
$$

where $\begin{array} { r l } { \nabla g ( X _ { T } ^ { u } ) , \int _ { 0 } ^ { T } \| c ( X _ { s } ^ { u } , s ) \| d s } \end{array}$ and $\begin{array} { r } { \int _ { 0 } ^ { T } \| A ( X _ { s } ^ { u } , s ) ^ { \top } Y _ { s } ^ { i } \| d s } \end{array}$ are assumed integrable. Then $Y ^ { 1 } = Y ^ { 2 }$

Proof. Let $D _ { t } = Y _ { t } ^ { 1 } - Y _ { t } ^ { 2 }$ and $\varphi ( t ) = \mathbb E \| D _ { t } \|$ . Subtracting the two equations, the terminal and source terms cancel

$$
D _ { t } = \mathbb { E } \left[ \int _ { t } ^ { T } A ( X _ { s } ^ { u } , s ) ^ { \top } D _ { s } d s \Big | X _ { t } ^ { u } \right] .
$$

Using the triangle inequality and the bound on $A ,$ we get

$$
\| D _ { t } \| \leq \mathbb { E } \left[ \int _ { t } ^ { T } \| A ( X _ { s } ^ { u } , s ) ^ { \top } D _ { s } \| d s \bigg | X _ { t } ^ { u } \right] \leq \mathbb { E } \left[ \int _ { t } ^ { T } \kappa ( s ) \| D _ { s } \| d s \bigg | X _ { t } ^ { u } \right] ,
$$

and taking expectations leads to

$$
\varphi ( t ) \leq \int _ { t } ^ { T } \kappa ( s ) \varphi ( s ) d s .
$$

Fix $\varepsilon \in ( 0 , T ]$ . As κ is integrable on [ε, T], Grönwall's lemma gives $\varphi = 0$ on $[ \varepsilon , T ]$ . As $\varepsilon$ is arbitrary, $D _ { t } = 0$ almost surely for every $t \in ( 0 , T ]$ . For $t = 0$ , we then have

$$
\mathbb { E } \| D _ { 0 } \| \leq \mathbb { E } \left[ \int _ { 0 } ^ { T } \| A ( X _ { s } ^ { u } , s ) ^ { \top } D _ { s } \| d s \right] = \int _ { 0 } ^ { T } \mathbb { E } \| A ( X _ { s } ^ { u } , s ) ^ { \top } D _ { s } \| d s = \int _ { 0 } ^ { T } 0 d s = 0 ,
$$

hence $D _ { 0 } = 0$ almost surely.

Remark: the assumption on A holds for the flow matching drift. In the flow matching setting of the main text, $T = 1$ and we have

$$
\nabla _ { x } b ( x , t ) = 2 \nabla _ { x } v ^ { \mathrm { b a s e } } ( x , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } I , \qquad \nabla _ { x } b _ { u _ { \theta } } ( x , t ) = 2 \nabla _ { x } v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( x , t ) - \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } I .
$$

If $v ^ { \mathrm { b a s e } }$ and $v _ { \theta } ^ { \mathrm { f i n e t u n e } }$ are L-Lipschitz in x uniformly in t, then

$$
\| \nabla _ { x } b ( x , t ) \| \leq 2 L + \frac { \dot { \alpha } _ { t } } { \alpha _ { t } } = : \kappa ( t ) , \qquad \int _ { \varepsilon } ^ { 1 } \kappa ( t ) d t = 2 L ( 1 - \varepsilon ) + \log \frac { 1 } { \alpha _ { \varepsilon } } < \infty
$$

for every $\varepsilon \in ( 0 , 1 ] .$ , and the same bound holds for $\nabla _ { x } b _ { u _ { \theta } }$ . For $\alpha _ { t } = t .$ we have $\kappa ( t ) = 2 L + 1 / t$ , which is not an issue: the lemma only requires integrability away from the initial time, because the backward Grönwall argument starts at the terminal time $t = 1$ . For a general control u Lipschitz in x, $\nabla _ { x } b _ { u } = \nabla _ { x } b + \gamma ( t ) \nabla _ { x } u$ adds the term $\gamma ( t ) L _ { u }$ , which is bounded on [ε, 1].

## D.3 Proof of Proposition 2

We start by restating Proposition 2 more precisely:

Proposition 10. For the loss defined by

$$
\begin{array} { l } { \displaystyle \hat { \mathcal { L } } _ { \mathrm { W A M } } ( u , v ) : = \mathbb { E } _ { \mathbf { X } \sim p ^ { v } } \left[ \frac { 1 } { 2 } \int _ { 0 } ^ { T } \left. u ( X _ { t } , t ) + \frac { \gamma ( t ) ^ { \top } \widetilde { a } ( t , \mathbf { X } ) } { w ( t ) } \right. ^ { 2 } d t \right] , } \\ { \mathrm { w h e r e } \quad \displaystyle \frac { d } { d t } \widetilde { a } ( t , \mathbf { X } ) = - ( \nabla _ { x } b ( X _ { t } , t ) ^ { \top } \widetilde { a } ( t , \mathbf { X } ) + \nabla _ { x } f ( X _ { t } , t ) ) , } \\ { \displaystyle \widetilde { a } ( T , \mathbf { X } ) = \nabla _ { x } g ( X _ { T } ) , } \end{array}\tag{31}
$$

the only critical point $\begin{array} { r } { \frac { \delta } { \delta u } \hat { \mathcal { L } } _ { \mathrm { W A M } } ( u , v ) | _ { v = u } = 0 } \end{array}$ is $u ^ { * }$ , the optimal control of the weighted SOC problem (22).

The rest of this section is dedicated to the proof of Proposition 10.

Functional gradient. For a measure M on $\mathbb { R } ^ { d } \times [ 0 , T ]$ , a functional $\mathcal { F } : L ^ { 2 } ( \mathbb { R } ^ { d } \times \lbrack 0 , T \rbrack , \mathcal { M } ; \mathbb { R } ^ { d } ) \to \mathbb { R }$ and $f \in L ^ { 2 } ( \mathbb { R } ^ { d } \times \lbrack \mathsf { \bar { 0 } } , T ] , \mathcal { M } ; \mathbb { R } ^ { d } )$ , we define the functional gradient $\textstyle { \frac { \delta } { \delta f } } { \mathcal { F } } ( f )$ as the function $g \in L ^ { 2 } ( \mathbb { R } ^ { d } \times [ 0 , T ] , \mathcal { M } ; \mathbb { R } ^ { d } )$ such that for all $h \in L ^ { 2 } ( \mathbb { R } ^ { d } \times [ 0 , T ] , \mathcal { M } ; \mathbb { R } ^ { d } )$

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } \frac { \mathcal { F } ( f + \varepsilon h ) - \mathcal { F } ( f ) } { \varepsilon } = \int \langle g ( x , t ) , h ( x , t ) \rangle \mathcal { M } ( d x , d t ) .
$$

First step, critical point condition. For v a fixed control, and the measure $\mathcal { M } ( d x , d t ) = p ( X _ { t } ^ { v } = x ) d x d t$ we have

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [ \mathcal { L } _ { \mathbf { W } \mathbf { A M } } ( u , \mathbf { X } ) ] } \\ & { \qquad = \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [ \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| u ( X _ { t } , t ) + \frac { \gamma ( t ) ^ { \top } \widetilde { \alpha } ( t , \mathbf { X } ) } { w ( t ) } \| ^ { 2 } d t ] } \\ & { \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { T } \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [ \| u ( X _ { t } , t ) \| ^ { 2 } ] d t + \int _ { 0 } ^ { T } \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [  u ( X _ { t } , t ) , \frac { \gamma ( t ) ^ { \top } \widetilde { \alpha } ( t , \mathbf { X } ) } { w ( t ) }  ] d t + \mathrm { c s t } } \\ & { \qquad = \frac { 1 } { 2 } \int _ { 0 } ^ { T } \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [ \| u ( X _ { t } , t ) \| ^ { 2 } ] d t + \int _ { 0 } ^ { T } \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [  u ( X _ { t } , t ) , \frac { \gamma ( t ) ^ { \top } \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [ \widetilde { \alpha } ( t , \mathbf { X } ) | X _ { t } ] } { w ( t ) }  ] d t + \mathrm { c s t } } \\ & { \qquad = \int ( \frac { 1 } { 2 } \| u ( x , t ) \| ^ { 2 } +  u ( x , t ) , \frac { \gamma ( t ) ^ { \top } \mathbb { E } _ { \mathbf { X } \sim p ^ { \nu } } [ \widetilde { \alpha } ( t , \mathbf { X } ) | X _ { t } = x ] } { w ( t ) }  ) \mathcal { M } ( d x , d t ) . } \end{array}
$$

It follows that, for $( x , t ) \in \mathbb { R } ^ { d } \times [ 0 , T ]$

$$
\frac { \delta } { \delta u } \mathbb { E } _ { \mathbf { X } \sim p ^ { \upsilon } } [ \mathcal { L } _ { \mathrm { W A M } } ( u , \mathbf { X } ) ] ( x , t ) = u ( x , t ) + \frac { \gamma ( t ) ^ { \top } \mathbb { E } _ { \mathbf { X } \sim p ^ { \upsilon } } [ \widetilde { a } ( t , \mathbf { X } ) | X _ { t } = x ] } { w ( t ) } ,
$$

hence u is a critical point of $u \mapsto \mathbb { E } _ { \mathbf { X } \sim p ^ { \bar { u } } } [ \mathcal { L } _ { \mathrm { W A M } } ( u , \mathbf { X } ) ]$ if and only if

$$
\forall ( x , t ) \in \mathbb { R } ^ { d } \times [ 0 , T ] , \quad u ( x , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \mathbb { E } [ \widetilde { a } ( t , \mathbf { X } ^ { u } ) | X _ { t } ^ { u } = x ]\tag{32}
$$

Second step, lean adjoint equation. By construction $\widetilde { a } ( t , \mathbf { X } ^ { u } )$ satisfies for $t \in [ 0 , T ]$

$$
\widetilde { a } ( t , \mathbf { X } ^ { u } ) = \nabla _ { x } g ( X _ { T } ^ { u } ) + \int _ { t } ^ { T } ( \nabla _ { x } b ( X _ { s } ^ { u } , s ) ^ { \top } \widetilde { a } ( s , \mathbf { X } ^ { u } ) + \nabla _ { x } f ( X _ { s } ^ { u } , s ) ) d s .
$$

Denoting $\widetilde { Y } _ { t } = \mathbb { E } [ \widetilde { a } ( t , \mathbf { X } ^ { u } ) | X _ { t } ]$ , we get that

$$
\widetilde { Y } _ { t } = \mathbb { E } \left[ \nabla _ { x } g ( X _ { T } ^ { u } ) + \int _ { t } ^ { T } ( \nabla _ { x } b ( X _ { s } ^ { u } , s ) ^ { \top } \widetilde { Y } _ { s } + \nabla _ { x } f ( X _ { s } ^ { u } , s ) ) d s \Big | X _ { t } \right] .\tag{33}
$$

Third step, critical point implies optimal control. Multiplying (32) by $\nabla _ { x } u ( x , t ) ^ { \top }$ , we get

$$
\begin{array} { r l } & { \nabla _ { x } u ( x , t ) ^ { \top } u ( x , t ) = \nabla _ { x } u ( x , t ) ^ { \top } \left( - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } { \mathbb { E } } [ \widetilde a ( t , { \mathbf { X } } ^ { u } ) | X _ { t } ^ { u } = x ] \right) } \\ & { \qquad \Leftrightarrow { \mathbb { E } } \left[ w ( t ) \nabla _ { x } u ( X _ { t } ^ { u } , t ) ^ { \top } u ( X _ { t } ^ { u } , t ) + \nabla _ { x } \left( \gamma ( t ) u ( X _ { t } ^ { u } , t ) \right) ^ { \top } \widetilde a ( t , { \mathbf { X } } ^ { u } ) \Big | X _ { t } = x \right] = 0 . } \end{array}
$$

Plugging this equality into (33) gives

$$
\widetilde { Y } _ { t } = \mathbb { E } \left[ \nabla _ { x } g ( X _ { T } ^ { u } ) + \int _ { t } ^ { T } ( \nabla _ { x } b _ { u } ( X _ { s } ^ { u } , s ) ^ { \top } \widetilde { Y } _ { s } + \nabla _ { x } f _ { u } ( X _ { s } ^ { u } , s ) ) d s \Big | X _ { t } \right] .
$$

By Lemma 9, applied with $A = \nabla _ { x } b _ { u }$ and $c = \nabla _ { x } f _ { u } , \widetilde { Y } _ { t } = Y _ { t } = \mathbb { E } [ a ( t , \mathbf { X } ^ { u } , u ) | X _ { t } ] , \mathrm { i . e . }$

$$
u ( x , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \mathbb { E } [ a ( t , \mathbf { X } ^ { u } , u ) | X _ { t } ^ { u } = x ] = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \nabla _ { x } J ( u , x , t ) ,
$$

and Lemma 7 gives $u = u ^ { * } \colon$ a critical point is necessarily the optimal control.

Fourth step, optimal control is a critical point. Conversely, we check that $u ^ { * }$ satisfies (32), i.e., that it is a critical point. Take $u = u ^ { * }$ . By (15), $Y _ { t } = \mathbb { E } [ a ( t , \mathbf { X } ^ { \boldsymbol { u } ^ { * } } , \boldsymbol { u } ^ { * } ) \vert \dot { X } _ { t } ^ { \boldsymbol { u } ^ { * } } ] = \nabla _ { \boldsymbol { x } } J ( \boldsymbol { u } ^ { * } , \boldsymbol { X } _ { t } ^ { \boldsymbol { u } ^ { * } } , t )$ satisfies $u ^ { * } ( X _ { t } ^ { u ^ { * } } , t ) =$ $- \frac { \gamma ( t ) ^ { \top } } { w ( t ) } Y _ { t }$ , hence for all $s \in [ 0 , T ]$

$$
\begin{array} { r l } & { \nabla _ { x } b _ { u ^ { * } } ( X _ { s } ^ { u ^ { * } } , s ) ^ { \top } Y _ { s } + \nabla _ { x } f _ { u ^ { * } } ( X _ { s } ^ { u ^ { * } } , s ) = \nabla _ { x } b ( X _ { s } ^ { u ^ { * } } , s ) ^ { \top } Y _ { s } + \nabla _ { x } f ( X _ { s } ^ { u ^ { * } } , s ) + \nabla _ { x } u ^ { * } ( X _ { s } ^ { u ^ { * } } , s ) ^ { \top } \left( \gamma ( s ) ^ { \top } Y _ { s } + w ( s ) u ^ { * } ( X _ { s } ^ { u ^ { * } } , s ) \right) } \\ & { \qquad = \nabla _ { x } b ( X _ { s } ^ { u ^ { * } } , s ) ^ { \top } Y _ { s } + \nabla _ { x } f ( X _ { s } ^ { u ^ { * } } , s ) . } \end{array}
$$

Plugging this identity into the equation (30) satisfied by $Y _ { t } ,$ we get

$$
Y _ { t } = \mathbb { E } \left[ \nabla _ { x } g ( X _ { T } ^ { u ^ { * } } ) + \int _ { t } ^ { T } ( \nabla _ { x } b ( X _ { s } ^ { u ^ { * } } , s ) ^ { \top } Y _ { s } + \nabla _ { x } f ( X _ { s } ^ { u ^ { * } } , s ) ) d s { \Big | } X _ { t } ^ { u ^ { * } } \right] ,
$$

which is also the equation (33) satisfied by $\widetilde { Y } _ { t }$ . Lemma 9, applied with $A = \nabla _ { x } b$ and $c = \nabla _ { x } f$ , gives $Y = { \widetilde { Y } }$ hence

$$
u ^ { * } ( x , t ) = - \frac { \gamma ( t ) ^ { \top } } { w ( t ) } \mathbb { E } [ \widetilde { a } ( t , \mathbf { X } ^ { u ^ { * } } ) | { X } _ { t } ^ { u ^ { * } } = x ] ,
$$

which is (32) for $u = u ^ { * }$

Adding an additional time-dependent weight. Adding an additional time-dependent weight $\lambda ( t ) > 0$ , as in (21)

$$
\mathcal { L } _ { \operatorname { W A M } } ( u , \mathbf { X } ) : = \frac { 1 } { 2 } \int _ { 0 } ^ { T } \lambda ( t ) \left\| u ( X _ { t } , t ) + \frac { \gamma ( t ) ^ { \top } \widetilde { a } ( t , \mathbf { X } ) } { w ( t ) } \right\| ^ { 2 } d t
$$

does not change the critical points, as it only changes the base distribution $\mathcal { M } ( d x , d t ) = \lambda ( t ) p ( X _ { t } ^ { v } = x ) d x d t$ to compute the functional gradient.

## D.4 Link with Domingo-Enrich and Han [40]

Recently, Domingo-Enrich and Han [40] proposed a more general Hamiltonian framework for adjoint matching. They derived the corresponding adjoint matching loss for a general problem of the form

$$
\begin{array} { r l } { \underset { u \in \mathcal { U } } { \operatorname* { m i n } } } & { \mathbb { E } \left[ \int _ { 0 } ^ { T } f ( X _ { t } ^ { u } , u ( X _ { t } ^ { u } , t ) , t ) d t + g ( X _ { T } ^ { u } ) \right] } \\ { \mathrm { s . t . } } & { d X _ { t } ^ { u } = b ( X _ { t } ^ { u } , u ( X _ { t } ^ { u } , t ) , t ) d t + \gamma ( X _ { t } ^ { u } , u ( X _ { t } ^ { u } , t ) , t ) d B _ { t } , } \\ & { X _ { 0 } ^ { u } \sim p _ { 0 } . } \end{array}
$$

Taking

$$
b ( x , u , t ) = b ( x , t ) + \gamma ( t ) u , \quad f ( x , u , t ) = f ( x , t ) + \frac { 1 } { 2 } \| u \| ^ { 2 } , \quad \gamma ( x , u , t ) = \gamma ( t ) ,
$$

they recover exactly (Theorem 2 and Remark 3) the adjoint matching from Domingo-Enrich et al. [35]. We can also apply their result (Theorem 2) to our weighted SOC problem (22), with

$$
b ( x , u , t ) = b ( x , t ) + \gamma ( t ) u , \quad f ( x , u , t ) = f ( x , t ) + \frac { w ( t ) } { 2 } \| u \| ^ { 2 } , \quad \gamma ( x , u , t ) = \gamma ( t ) ,
$$

leading to an adjoint matching formulation

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { W A M } } ( u , \mathbf { X } ) : = \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { T } w ( t ) \left\| u ( X _ { t } , t ) + \frac { \gamma ( t ) ^ { \top } \widetilde { a } ( t , \mathbf { X } ) } { w ( t ) } \right\| ^ { 2 } d t , \quad \mathbf { X } \sim p ^ { \bar { u } } , \quad \bar { u } = \mathrm { s t o p g r a d } ( u ) } \\ & { \quad \mathrm { w h e r e } \quad \quad \frac { d } { d t } \widetilde { a } ( t , \mathbf { X } ) = - ( \nabla _ { x } b ( X _ { t } , t ) ^ { \top } \widetilde { a } ( t , \mathbf { X } ) + \nabla _ { x } f ( X _ { t } , t ) ) , } \\ & { \quad \widetilde { a } ( T , \mathbf { X } ) = \nabla _ { x } g ( X _ { T } ) . } \end{array}
$$

This is (21) with $\lambda ( t ) = w ( t )$ . The framework of Domingo-Enrich and Han [40] therefore recovers our loss for one choice of $\lambda ,$ but lacks the extra degree of freedom that λ provides.

Algorithm 1 Weighted Adjoint Matching to finetune a flow matching model   
Input: Flow matching model $v ^ { \mathrm { b a s e } } = v _ { \theta _ { 0 } } ^ { \mathrm { b a s e } } , \alpha _ { t } , \beta _ { t }$ , reward function $r ,$ control weight $w ( t )$ , loss weight $\lambda ( t )$ , number   
of discretization steps $K ,$ number of trajectories per finetuning iteration $M .$ number of finetuning iterations $N$   
Initialize parameters of the finetuned model $\theta \quad = \quad \theta _ { 0 }$ i.e., $\begin{array} { r l r } { v _ { \theta } ^ { \mathrm { f i n e t u n e } } } & { { } = } & { v ^ { \mathrm { b a s e } } } \end{array}$ Define $\begin{array} { l l l } { \bar { h } } & { = } & { { \frac { 1 } { K } } } \end{array}$ and   
for $k \in \{ 0 , 1 , \ldots , K \} , t _ { k } = k h $   
for $n \in \{ 0 , \ldots , N - 1 \}$ do   
Sample M trajectories $\mathbf { X } ^ { ( m ) } = ( X _ { t _ { k } } ^ { ( m ) } ) _ { k \in \{ 0 , . . . , K \} }$ for $m \in \{ 0 , . . . , M - 1 \}$   
$\begin{array} { r l } { \left\{ \begin{array} { l l } { X _ { t _ { 0 } } ^ { ( m ) } } & { = X _ { 0 } ^ { ( m ) } \sim \mathcal { N } ( 0 , I ) , } \end{array} \right. } \end{array}$   
$\begin{array} { r } { \left\{ \begin{array} { l l } { \ddot { X _ { t _ { k + 1 } } ^ { ( m ) } } } & { = \ddot { X _ { t _ { k } } ^ { ( m ) } } + h \left( 2 v _ { \theta } ^ { \mathrm { f i n e t u n e } } ( X _ { t _ { k } } ^ { ( m ) } , t _ { k } ) - \frac { \dot { \alpha } _ { t _ { k } } } { \alpha _ { t _ { k } } } X _ { t _ { k } } ^ { ( m ) } \right) + \sqrt { h } \gamma ( t _ { k } ) \mathcal { N } ( 0 , I ) . } \end{array} \right. } \end{array}$ (34)   
Solve the adjoint ODE backwards in time $\widetilde { \mathbf { a } } ^ { ( m ) } = ( \widetilde { a } _ { t _ { k } } ^ { ( m ) } ) _ { k \in \{ 0 , \ldots , K \} }$   
$\begin{array} { r l } { \left\{ \begin{array} { l l } { \widetilde { a } _ { t _ { K } } ^ { ( m ) } } & { = \widetilde { a } _ { 1 } ^ { ( m ) } = - a \nabla r ( X _ { 1 } ^ { ( m ) } ) , } \end{array} \right. } \end{array}$   
$\begin{array} { r } { \left\{ \begin{array} { l l } { \tilde { a } _ { t _ { k - 1 } } ^ { \mathrm { ( m ) } } } & { = \tilde { a } _ { t _ { k } } ^ { \mathrm { ( m ) } } + h \nabla _ { X _ { t _ { k } } ^ { \mathrm { ( m ) } } } \left( 2 v ^ { \mathrm { b a s e } } ( X _ { t _ { k } } ^ { \mathrm { ( m ) } } , t _ { k } ) - \frac { \dot { \alpha } _ { t _ { k } } } { \alpha _ { t _ { k } } } X _ { t _ { k } } ^ { \mathrm { ( m ) } } \right) ^ { \top } \tilde { a } _ { t _ { k } } ^ { \mathrm { ( m ) } } . } \end{array} \right. } \end{array}$ (35)   
Take $\mathbf { X } ^ { ( m ) } = \mathrm { s t o p g r a d } ( \mathbf { X } ^ { ( m ) } )$ and $\widetilde { \mathbf { a } } = \mathrm { s t o p g r a d } ( \widetilde { \mathbf { a } } )$   
Compute the Weighted Adjoint Matching objective   
$\hat { \mathcal { L } } _ { \mathrm { W A M } } ( \theta ) = \frac { 1 } { 2 M \big ( K + 1 \big ) } \sum _ { m , k } \lambda ( t _ { k } ) \left. \frac { 2 } { \gamma \big ( t _ { k } \big ) } \left( v _ { \theta } ^ { \mathrm { f i n e t u m e } } ( X _ { t _ { k } } ^ { ( m ) } , t _ { k } ) - v ^ { \mathrm { b a s e } } ( X _ { t _ { k } } ^ { ( m ) } , t _ { k } ) \right) + \frac { \gamma \big ( t _ { k } \big ) } { w ( t _ { k } ) } \tilde { a } _ { t _ { k } } ^ { ( m ) } \right. ^ { 2 } .$ (36)   
Compute the gradient $\nabla _ { \theta } \hat { \mathcal { L } } _ { \mathrm { W A M } } ( \theta )$ and update the parameters $\theta$ using gradient descent.   
end for   
Output: Finetuned model $v _ { \theta } ^ { \mathrm { f i n e t u n e } }$

## D.5 Description of the algorithm

To run the algorithm, we first discretize the generative SDE (11), giving (34), then the adjoint ODE (19), leading to (35). We get a stochastic estimate of the loss $\hat { \mathcal { L } } _ { \mathrm { W A M } } ( \theta )$ by averaging over trajectories and timesteps. We then take a stochastic gradient descent step on $\theta .$ The full procedure is summarized in Algorithm 1. In our implementation, we also subsample $K _ { 0 } < K$ timesteps at random in (36) to reduce its computational cost. For the classical choice $\alpha _ { t } = t , \beta _ { t } = 1 - t ,$ two terms of $\begin{array} { r } { ( 3 4 ) , \frac { \dot { \alpha } _ { t _ { k } } } { \alpha _ { t _ { k } } } = \frac { 1 } { t _ { k } } } \end{array}$ and $\begin{array} { r } { \gamma ( t _ { k } ) = \sqrt { \frac { 2 ( 1 - t _ { k } ) } { t _ { k } } } } \end{array}$ , are singular at $t _ { 0 } = 0$ For the first step only, we evaluate these terms at $t _ { 1 } = h > \mathrm { ~ \ddot { 0 } ~ }$ instead of $t _ { 0 }$

## E EXPERIMENT DETAILS AND ADDITIONAL EXPERIMENTS

## E.1 Experiment details

Compute infrastructure. The experiments were run on an academic Slurm cluster. Each job was launched on a single NVIDIA GPU (RTX 2080Ti/6000/8000, V100, A100, H100 or H200), with 4–8 CPU cores and up to 12 GB of RAM. The experiments, including preliminary experiments, experiments not reported in the paper and jobs that were preempted or lost, ran for about 480 GPU-hours. The source code will be released at the time of publication.

Gaussian mixtures. For the first Gaussian mixture experiment of Section 5.1, the base model is an MLP with 4 546 parameters, trained using Adam (constant learning rate $2 \times 1 0 ^ { - 4 } )$ and a batch size of 100 for 100 epochs, each epoch using $1 0 ^ { 5 }$ fresh samples of the target. We use the exponential moving average (with decay 0.999) of its weights. Finetuning is done over 80 epochs of 10 batches of 1000 trajectories, sampled using 40 discretization steps. The optimizer is Adam (learning rate $2 \times 1 0 ^ { - 4 } )$ , and we use an exponential moving average of the weights with decay 0.99. We sweep $a \in [ 0 . 2 , 3 9 . 9 5 ]$ for adjoint matching and for the piecewise constant $( t _ { s } \in \{ 0 . 2 5 , 0 . 5 , 0 . 6 5 , 0 . 8 \} )$ , linear $( w _ { 0 } \in \{ 1 0 , 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } \} )$ and exponential $( \alpha \in \{ 2 , 4 , 6 , 8 \} )$ weighting functions. For each method (adjoint, piecewise, exponential or linear with fixed value of $t _ { s } , \alpha \ \mathrm { o r } \ w _ { 0 } )$ , we start from the smallest value of a, then warm start the model's weights using weights from the previous value of a, and run the optimization over only 40 epochs. The final checkpoints are evaluated on $1 0 ^ { 6 }$ samples.

![](images/5ef7dd690fa8b9f689e3254ac49d97531a29755255f52e92365fdf9e46f0c418.jpg)  
Figure 5: Plot of $\mathbb { E } [ r ( \hat { X } ) ]$ with respect to $\mathbb { E } { [ } \mathbb { 1 } _ { x _ { 1 } > 0 } ( \hat { X } ) { ] }$ for adjoint matching and weighted adjoint matching with a linear weighting function.

For the additional experiment of Appendix E.2.2, the experiment parameters are similar except for $a \in [ 0 . 2 , 2 9 . 9 5 ]$ and we did not use the linear weighting function. For the additional experiment of Appendix E.2.3, the experiment parameters are similar except for $a \in [ 0 . 2 , 9 . 2 ]$ and $t _ { s } \in \{ 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 \}$ , no warm start, and we did not use the linear and exponential weighting functions.

MNIST. For the MNIST experiments of Section 5.2 and Appendix E.4, the base model is a UNet with 9.0 million parameters, trained on the MNIST training images rescaled to [-1, 1], with batch size 64, using Adam (learning rate $2 \times 1 0 ^ { - 4 }$ , halved by PyTorch's ReduceLROnPlateau on the validation loss, patience 5) for 50 epochs. We use the exponential moving average (with decay 0.9999) of its weights. Each finetuning run starts from the base model and lasts 25 epochs of 50 batches of 64 trajectories, sampled with 40 discretization steps. We use Adam (learning rate $2 \times 1 0 ^ { - 4 }$ , halved by ReduceLROnPlateau on the mean gradient norm of each epoch, patience 5) and an exponential moving average of the weights with decay 0.995. We sweep $a \in [ 1 , 1 0 ]$ for adjoint matching and, for the piecewise constant weighting function, $a \in [ 1 0 , 5 0 ]$ at $t _ { s } ~ = ~ 0 . 2 5$ , [10, 100] at $t _ { s } = 0 . 3$ [40, 150] at $t _ { s } = 0 . 3 5$ , [80, 250] at $t _ { s } = 0 . 4$ and [120, 375] at $t _ { s } = 0 . 4 5$ . The final checkpoints are evaluated on $5 \times 1 0 ^ { 4 }$ samples and classified by the Hugging Face classifier farleyknight/mnist-digit-classification-2022-09-04 (a finetuned ViT-B/16, Apache license 2.0).

## E.2 Gaussian mixtures

## E.2.1 Using a linear weighting function

For the experiment of Section 5.1, we also tried using a linear function $w _ { \mathrm { l i n e a r } } ( t ) = w _ { 0 } + ( w _ { 1 } - w _ { 0 } ) t ,$ with $w _ { 1 } = 1$ and different values for $w _ { 0 }$ . In Figure 5, we observe that the linear weighting function achieves a better balance/reward tradeoff than classical adjoint matching, but is less effective than the exponential or piecewise constant weighting function.

## E.2.2 Gaussian mixture: a shifted mode

In Figure 6, we reproduce the experiment of Section 5.1 with a different base distribution (two modes with the same variance but the second one is shifted off the first axis)

$$
p _ { \mathrm { t a r g e t } } ( x ) = { \frac { 1 } { 2 } } { \mathcal { N } } ( x ; \mu _ { 1 } , \sigma ^ { 2 } I ) + { \frac { 1 } { 2 } } { \mathcal { N } } ( x ; \mu _ { 2 } , \sigma ^ { 2 } I ) = { \frac { 1 } { 2 } } ( 2 \pi \sigma ^ { 2 } ) ^ { - 1 } e ^ { - { \frac { \| x - \mu _ { 1 } \| ^ { 2 } } { 2 \sigma ^ { 2 } } } } + { \frac { 1 } { 2 } } ( 2 \pi \sigma ^ { 2 } ) ^ { - 1 } e ^ { - { \frac { \| x - \mu _ { 2 } \| ^ { 2 } } { 2 \sigma ^ { 2 } } } } ,
$$

with $\mu _ { 1 } = ( - m , 0 ) , \mu _ { 2 } = ( m , s ) , m , s \in \mathbb { R }$ and $\sigma > 0$ . The conclusions are unchanged.

![](images/6634248b6ba1c09d363c957f10137ea69c7c4ff4a8ed3422094159703b7f1a62.jpg)  
(a) Adjoint matching and weighted adjoint matching with an exponential weighting function.

![](images/70b83d1abcb4c74c1d7b33c5680c854a4ab385af849d287c45e965f9d38beb2e.jpg)  
(b) Adjoint matching and weighted adjoint matching with a piecewise constant weighting function.  
Figure 6: Average reward with respect to weight of positive mode for adjoint matching and weighted adjoint matching, using different functions $t \mapsto w ( t )$

## E.2.3 Gaussian mixture: four modes

We ran an additional experiment with a mixture of 4 Gaussians as base distribution

$$
p _ { \mathrm { t a r g e t } } ( x ) = \frac { 1 } { 4 } \sum _ { i = 1 } ^ { 4 } \mathcal { N } ( x ; \mu _ { i } , \sigma ^ { 2 } I ) ,
$$

with $\mu _ { 1 } = ( 4 , - 2 ) , \mu _ { 2 } = ( 4 , 2 ) , \mu _ { 3 } = ( - 4 , - 2 ) , \mu _ { 4 } = ( - 4 , 2 )$ and $\sigma ^ { 2 } = 0 . 2 5$ . The reward is $r ( x ) = x _ { 1 } + x _ { 2 }$ and grows by moving to the upper right. In Figure 7, we plot samples of the base distribution alongside samples of the output of adjoint matching and weighted adjoint matching with the piecewise constant function $w _ { \mathrm { s t e p s } } ( t )$ for different values of a and $t _ { s }$ While adjoint matching quickly collapses to the top-right mode, we see that this effect is mitigated by weighted adjoint matching depending on the value of $t _ { s }$ . Collapse is still observed with $t _ { s } = 0 . 5$ , though it happens later. For $t _ { s } = 0 . 6$ , we have a collapse within each pair of vertically adjacent modes, but not a collapse between left and right. This is because the horizontal spacing of the modes is larger than the vertical spacing, hence the left-right splitting happens earlier along the generative trajectory. Here the left-right feature is the coarse feature that is preserved, while the top-bottom feature is the finer feature that is changed during finetuning. Finally, for $t _ { s } = 0 . 7$ and $t _ { s } = 0 . 8$ , we observed that both left-right and top-bottom features are preserved, and reward maximization happens only within each mode by shifting locally the samples to the upper right.

## E.3 MNIST main experiment

In Tables 1 & 2, we have the distributions of all digits (from 50 000 samples classified by the pretrained classifier). for the base model, adjoint matching and weighted adjoint matching with different $t _ { s }$ . For each, we select a specific value of a by looking at the one that has an achieved reward closest to 0.16 (Table 1) and 0.18 (Table 2). We observed that for an average achieved reward of 0.176, adjoint matching generates 18.0% of zeros and only 3.4% of ones, while weighted adjoint matching with $t _ { s } = 0 . 4 5$ reaches a reward of 0.180 with 10.5% of zeros and 9.1% of ones. Samples of the base model, adjoint matching and weighted adjoint matching with $t _ { s } = 0 . 4 5$ , grouped by predicted class, are shown in Figures 9 & 10.

## E.4 MNIST with darkness reward

We also ran an additional experiment with darkness as the reward, defined, for $x \in [ 0 , 1 ] ^ { d }$ , as $\begin{array} { r } { r ( x ) = 1 - \frac { 1 } { d } \sum _ { i } x _ { i } . } \end{array}$ Results are in Figure 8 and Table 3. Samples of the base model, adjoint matching and weighted adjoint matching with $t _ { s } = 0 . 4 5$ , grouped by predicted class, are shown in Figure 11. The conclusions are unchanged.

![](images/89d139008304894b7c61696c7733d2e286c79e37a9fbbad98c6f9132cdf88b92.jpg)  
Figure 7: Sample of base distribution (blue) and finetuned models (orange), for adjoint matching and weighted adjoint matching for different values of a and $t _ { s }$

![](images/bb4087248a25c23c0fabbf184e7c067c9fae22da02c5c514f586a87d345e8827.jpg)  
Figure 8: Average reward with respect to ratio $\frac { n _ { \mathrm { m i n } } } { n _ { \mathrm { m a x } } }$

Table 1: Predicted class of the generated samples, in percent, for the run of each method whose average luminosity is closest to 0.16.
<table><tr><td></td><td></td><td></td><td colspan="10">Predicted class (%)</td></tr><tr><td>Method</td><td>a</td><td>R</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>Base</td><td></td><td>0.140</td><td>10.2</td><td>9.9</td><td>9.9</td><td>10.8</td><td>9.4</td><td>9.1</td><td>9.8</td><td>10.0</td><td>10.1</td><td>10.9</td></tr><tr><td>Adjoint</td><td>5.7</td><td>0.159</td><td>14.8</td><td>5.8</td><td>12.0</td><td>12.0</td><td>7.4</td><td>8.5</td><td>10.0</td><td>8.0</td><td>12.1</td><td>9.4</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 2 5$ </td><td>21</td><td>0.161</td><td>10.6</td><td>6.6</td><td>11.4</td><td>11.8</td><td>8.9</td><td>8.1</td><td>10.2</td><td>9.1</td><td>13.0</td><td>10.4</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 3$ </td><td>34</td><td>0.162</td><td>10.5</td><td>7.4</td><td>10.6</td><td>10.9</td><td>9.4</td><td>8.5</td><td>9.2</td><td>9.2</td><td>12.9</td><td>11.4</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 3 5$ </td><td>52</td><td>0.161</td><td>10.6</td><td>8.8</td><td>10.2</td><td>11.0</td><td>9.2</td><td>7.7</td><td>9.8</td><td>9.0</td><td>12.3</td><td>11.2</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 4$ </td><td>80</td><td>0.160</td><td>10.5</td><td>9.5</td><td>10.0</td><td>10.7</td><td>9.2</td><td>8.4</td><td>9.7</td><td>9.3</td><td>11.5</td><td>11.2</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 4 5$ </td><td>120</td><td>0.160</td><td>10.3</td><td>9.7</td><td>9.8</td><td>10.6</td><td>9.1</td><td>8.5</td><td>9.8</td><td>9.6</td><td>11.2</td><td>11.3</td></tr></table>

Table 2: Predicted class of the generated samples, in percent, for the run of each method whose average luminosity is closest to 0.18.
<table><tr><td colspan="4"></td><td colspan="10">Predicted class (%)</td></tr><tr><td>Method</td><td>a</td><td>R</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>Base</td><td></td><td>0.140</td><td>10.2</td><td>9.9</td><td>9.9</td><td>10.8</td><td>9.4</td><td>9.1</td><td>9.8</td><td>10.0</td><td>10.1</td><td>10.9</td></tr><tr><td>Adjoint</td><td>10</td><td>0.176</td><td>18.0</td><td>3.4</td><td>12.5</td><td>12.3</td><td>7.0</td><td>7.9</td><td>10.7</td><td>6.5</td><td>13.1</td><td>8.6</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 2 5$ </td><td>37</td><td>0.181</td><td>12.0</td><td>3.4</td><td>13.7</td><td>11.3</td><td>8.2</td><td>7.8</td><td>8.0</td><td>7.7</td><td>17.4</td><td>10.6</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 3$ </td><td>57</td><td>0.180</td><td>11.9</td><td>6.1</td><td>11.2</td><td>11.6</td><td>8.1</td><td>6.4</td><td>8.9</td><td>8.4</td><td>16.0</td><td>11.2</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 3 5$ </td><td>92</td><td>0.180</td><td>10.6</td><td>7.8</td><td>11.0</td><td>10.7</td><td>8.5</td><td>7.6</td><td>9.7</td><td>8.6</td><td>14.2</td><td>11.4</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 4$ </td><td>143</td><td>0.180</td><td>10.7</td><td>8.4</td><td>10.4</td><td>10.5</td><td>8.6</td><td>7.6</td><td>9.8</td><td>9.1</td><td>13.3</td><td>11.6</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 4 5$ </td><td>227</td><td>0.180</td><td>10.5</td><td>9.1</td><td>10.2</td><td>10.3</td><td>8.9</td><td>7.9</td><td>9.7</td><td>9.3</td><td>12.4</td><td>11.5</td></tr></table>

![](images/51dd582706a1624dc294412bb917112ac0c1d239d951609ebe10410da8da50c3.jpg)  
(a) Base model, average reward R = 0.140

![](images/390f82d40141ef42338142cd3a54ba0b7cb086e5f78c5c66a61834ada09ce68e.jpg)  
(b) Adjoint matching, a = 5.7, R = 0.159

![](images/78313b0c54d5ac7cfc6bce881344fbc0bb6866048e2b623162fd32c3624466d4.jpg)  
(c) Weighted adjoint matching, ts = 0.45, a = 120, R = 0.160  
Figure 9: Generated samples grouped by predicted class, for the run of each method whose average luminosity is closest to 0.16. Percentages are computed on the 1 000 displayed samples, and therefore differ from those of Table 1, computed on 50 000 samples.

![](images/6e098d8c4c8ea040d69bfee987cc21c11ac72bfe00a6951dded3c7e7382b6457.jpg)  
(a) Base model, average reward R = 0.140

![](images/852d9f682bf8828607a40087939f950db7883725dc00b38c58fbde4cb44785e4.jpg)  
(b) Adjoint matching, a = 10, R = 0.176

![](images/73579dbd3a572ebe391632b6c8dbd860616e6cb98ace9cc4dcf79d4359305549.jpg)  
(c) Weighted adjoint matching, ts = 0.45, a = 227, R = 0.180  
Figure 10: Generated samples grouped by predicted class, for the run of each method whose average luminosity is closest to 0.18. Percentages are computed on the 1 000 displayed samples, and therefore differ from those of Table 2, computed on 50 000 samples.

![](images/7c5a6cfd4453dc3b3c16e1ebef9300b010b0eb7fcb376fe7b15036a68a950f55.jpg)  
(a) Base model, average reward R = 0.857

![](images/176878e38867ddd9afa1b6d0555540872a5a7dd156da1fd32b412d645db6b09d.jpg)  
(b) Adjoint matching, a = 8.6, R = 0.880

![](images/66c650e1171071dc98dab195192828e6799f7554343c973c17775e8e9644251c.jpg)  
(c) Weighted adjoint matching, ts = 0.45, a = 160, R = 0.881  
Figure 11: Generated samples grouped by predicted class, for the run of each method whose average darkness is closest to 0.88. Percentages are computed on the 1 000 displayed samples, and therefore differ from those of Table 3, computed on 50 000 samples.

Table 3: Predicted class of the generated samples, in percent, for the run of each method whose average darkness is closest to 0.88.
<table><tr><td></td><td></td><td></td><td colspan="10">Predicted class  $( \% )$ </td></tr><tr><td>Method</td><td>a</td><td> $\hat { R }$ </td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>Base</td><td></td><td>0.857</td><td>10.4</td><td>10.1</td><td>9.4</td><td>10.8</td><td>9.5</td><td>9.0</td><td>9.9</td><td>9.8</td><td>10.3</td><td>10.7</td></tr><tr><td>Adjoint</td><td>8.6</td><td>0.880</td><td>5.3</td><td>19.6</td><td>7.8</td><td>9.9</td><td>9.6</td><td>9.6</td><td>8.9</td><td>10.8</td><td>7.8</td><td>10.6</td></tr><tr><td>Weighted,  $t _ { s } = 0 . 2 5$ </td><td>25</td><td>0.880</td><td>8.8</td><td>14.8</td><td>8.1</td><td>10.4</td><td>12.0</td><td>9.0</td><td>7.3</td><td>10.4</td><td>7.6</td><td>11.6</td></tr><tr><td> $\mathrm { W e i g h t e d } , t _ { s } = 0 . 3$ </td><td>43</td><td>0.880</td><td>9.3</td><td>12.2</td><td>8.8</td><td>11.1</td><td>10.2</td><td>10.0</td><td>9.6</td><td>10.7</td><td>8.1</td><td>10.1</td></tr><tr><td> $\mathrm { W e i g h t e d } , t _ { s } = 0 . 3 5$ </td><td>63</td><td>0.880</td><td>9.8</td><td>11.2</td><td>9.0</td><td>10.5</td><td>10.6</td><td>10.5</td><td>9.4</td><td>10.4</td><td>8.4</td><td>10.2</td></tr><tr><td> $\mathrm { W e i g h t e d } , t _ { s } = 0 . 4$ </td><td>98</td><td>0.879</td><td>10.1</td><td>10.6</td><td>9.4</td><td>11.0</td><td>8.5</td><td>9.8</td><td>10.0</td><td>10.4</td><td>9.1</td><td>11.1</td></tr><tr><td> $\mathrm { W e i g h t e d } , t _ { s } = 0 . 4 5$ </td><td>160</td><td>0.881</td><td>10.1</td><td>10.3</td><td>9.8</td><td>11.0</td><td>9.9</td><td>9.9</td><td>9.1</td><td>10.3</td><td>9.2</td><td>10.5</td></tr></table>