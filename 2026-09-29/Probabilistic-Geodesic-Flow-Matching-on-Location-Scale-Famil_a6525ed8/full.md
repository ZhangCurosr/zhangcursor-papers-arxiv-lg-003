# Probabilistic Geodesic Flow Matching on Location-Scale Families

Zeyuan Yu Zhi Chang Shiwei Lan<sup>†</sup>

School of Mathematical & Statistical Sciences Arizona State University Tempe, AZ 85287 USA {zeyuany1,zchang7,slan}@asu.edu

## Abstract

Flow matching (FM) has recently emerged as a promising framework for generative modeling due to its conceptual simplicity and strong empirical performance. In FM, samples are transported along a vector field parameterized by a neural network, inducing a probability path that evolves from a simple noise distribution to the target data distribution, governed by an ordinary diferential equation (ODE). However, existing FM approaches predominantly rely on probability paths derived from optimal transport (OT) between Gaussian distributions, which may be suboptimal for capturing complex data with inhomogeneous structures such as heavy tail or sharp contrast. In this work, we generalize FM to the broader class of location–scale families for handling data inhomogeneity and introduce a novel class of probability paths defined as geodesics on the manifold of probability distributions. We name this approach probabilistic geodesic flow matching to distinguish it from prior geodesic (Riemannian) FM methods defined in input space. We argue that Euclidean OT-based paths are not necessarily optimal in prob ability space and may limit modeling flexibility. Through synthetic benchmarks and scientific datasets at diferent scales, we demonstrate that the proposed method more efectively captures complex distributions, leading to improved or comparable performance compared with SOTA geometry-motivated generative models.

## 1 Introduction

Difusion models and flow matching methods are two classes of popular generative modeling approaches. Denoising difusion probabilistic model [15] is a discrete Markov model that converts a complex data distribution into a simple tractable distribution by gradually adding noise in a forward process and then trains a generative model to remove the noise in a reverse process. [29] generalize it to a continuum of distributions and reformulate these two processes using SDEs. In the SDE framework, the Fokker–Planck equation governs the evolution of the density path, which induces an ODE that enables a deterministic flow with marginal densities the same as SDE [22]. The continuous normalizing flow [CNF 7] directly models the vector field of the probabilistic flow that generates the probability path from noise to data distributions, bypassing the noise-adding forward process. Recently, flow matching [FM 20] introduces a difusion-style training objective to predict the vector field and improves eficiency by avoiding expensive ODE simulation in CNF training.

Despite the connection to difusion processes, dominant FM methods are based on probability paths as an optimal transport (OT) interpolant [23] between Gaussian distributions. OT paths are known to be geodesics in the 2-Wasserstein space [5, 31]. They may not necessarily be optimal for complex data distributions, e.g. non-Gaussian distributions with heavy tails. Heavy-tailed priors using student-t distributions have recently been studied in difusion models [26]. In this work, we generalize FM by deviating from the default choice of Gaussian distributions and explore a broader location-scale family. In particular, we focus on exponential power distributions [13], which have the flexibility of imposing regularization and modeling tail probabilities through a parameter q > 0 and embrace Gaussian distribution as a special case q = 2.0. To search for the optimal probability path on the manifold of exponential power distributions, we adopt the Fisher-Rao metric and solve the geodesic under this metric. The proposed FM with this geodesic path is hence named probabilistic geodesic flow matching, to diferentiate it from the existing Riemannian (geodesic) FM [6].

Connection to Existing Literature Our work directly generalizes the FM method [20]. In addi tion to adopting broader location-scale distributions, our framework, developed from probability path to vector field and then sample flow, is reverse to [20], but provides a natural foundation of manifold perspective of probability paths. We want to highlight that the Riemannian FM [6] has its geodesics defined in the input space of most applications. However, our probabilistic geodesics are defined on the manifold of exponential power and more general location-scale distributions. Fisher-Rao metric is also considered in [8] and [9] for discrete data. Other manifold based FM includes [17] with a learned metric and [4] with a 2-Wasserstein. Despite the diferent applications, their constructions either take advantage of generic spherical geometry or learn from data, while our derivation is more native to exponential power distributions with analytic solution (see Table .4 for a more comprehensive comparison). Our work makes multi-fold contributions:

1. It generalizes FM with a broader location-scale family beyond Gaussian distributions.

2. It interprets optimal probability paths as energy-minimizing trajectories in the probability space with appropriate metrics.

3. It proposes probabilistic geodesic FM demonstrating improved or comparable performance.

The remainder of the paper is organized as follows. Section 2 reviews the background on FM and exponential power distributions. Section 3 introduces the flexible FM framework in location-scale family. Then we derive the probabilistic geodesics on the manifold of exponential power distributions in Section 4. We demonstrate the numerical advantages in Section 5 and conclude in Section 6 with some discussion on limitations and future directions.

## 2 Background Review

## 2.1 (Conditional) Flow Matching

To learn the distribution $q ( \cdot )$ of a given dataset and generate synthetic samples $\mathbf { x } \sim q ( \cdot )$ , flow-based approaches evolve noisy data $\mathbf { x } _ { 0 } \sim p _ { 0 } ( \cdot )$ to clean data $\mathbf { x } _ { 1 } \sim p _ { 1 } ( \cdot ) \approx q ( \cdot )$ along a flow mapping $\phi _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ defined by the following ODE with given vector field $\mathbf { v } _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ for $t \in [ 0 , 1 ]$ :

$$
\begin{array} { c } { { \displaystyle \frac { d } { d t } \phi _ { t } ( { \bf x } ) = { \bf v } _ { t } \big ( \phi _ { t } ( { \bf x } ) \big ) , } } \\ { { \displaystyle \phi _ { 0 } ( { \bf x } ) = { \bf x } _ { 0 } , } } \end{array}\tag{1}
$$

where $\mathbf { x } _ { t } : = \phi _ { t } ( \mathbf { x } _ { 0 } )$ follows a probability path $p _ { t } ( \cdot )$ given by the push-forward equation

$$
p _ { t } = [ \phi _ { t } ] _ { \sharp } p _ { 0 } , \quad [ \phi _ { t } ] _ { \sharp } p _ { 0 } ( { \bf x } ) = p _ { 0 } ( \phi _ { t } ^ { - 1 } ( { \bf x } ) ) \operatorname* { d e t } \left[ \frac { \partial \phi _ { t } } { \partial { \bf x } } \right] ^ { - 1 } .\tag{2}
$$

A vector field $\mathbf { v } _ { t }$ is said to generate the probability path $p _ { t }$ if and only if the continuity equation holds:

$$
\partial _ { t } p _ { t } ( \mathbf { x } ) = - \nabla _ { \mathbf { x } } \cdot \big ( \mathbf { v } _ { t } ( \mathbf { x } ) p _ { t } ( \mathbf { x } ) \big ) ,\tag{3}
$$

which induces the following equation evolving the logarithm of density:

$$
\frac { d } { d t } \log p _ { t } ( { \bf x } _ { t } ) = - \mathrm { t r } \left( \frac { \partial \mathbf { v } _ { t } } { \partial { \bf x } _ { t } } \right) = - \nabla \cdot \mathbf { v } _ { t } ( \mathbf { x } _ { t } ) .\tag{4}
$$

Once the vector field $\mathbf { v } _ { t }$ is learned from the data, a synthetic sample can be generated by solving the ODE (1) up to time $t = 1 \colon \mathbf { x } _ { 1 } = \phi _ { 1 } ( \mathbf { x } _ { 0 } ) \sim p _ { 1 } ( \cdot ) \approx q ( \cdot )$

The FM considers a conditional probability path $p _ { t } ( \cdot | \mathbf { x } _ { 1 } )$ for a fixed data point $\mathbf { x } _ { 1 } \in \mathbb { R } ^ { d }$ such that $p _ { 0 } ( \cdot ) = p _ { 0 } ( \cdot | \mathbf { x } _ { 1 } )$ and $\begin{array} { r } { p _ { 1 } ( \cdot ) = \int p _ { 1 } ( \cdot | \mathbf { x } _ { 1 } ) q ( \mathbf { x } _ { 1 } ) d \mathbf { x } _ { 1 } \approx q ( \cdot ) } \end{array}$ . A common choice is $p _ { 0 } ( \cdot | \mathbf { x } _ { 1 } ) = \mathcal { N } ( \cdot ; \mathbf { 0 } , \mathbf { I } )$ and $p _ { 1 } ( \cdot | \mathbf { x } _ { 1 } ) = \mathcal { N } ( \cdot ; \mathbf { x } _ { 1 } , \sigma _ { \mathrm { m i n } } ^ { 2 } \mathbf { I } )$ for some small $\sigma _ { \operatorname* { m i n } } > 0 .$ . A conditional vector field $\mathbf { u } _ { t } ( \cdot | \mathbf { x } _ { 1 } )$ is then constructed to generate this conditional probability path $p _ { t } ( \cdot | \mathbf { x } _ { 1 } )$ by the flow $\begin{array} { r } { \frac { d } { d t } \psi _ { t } ( \mathbf x ) = \mathbf { u } _ { t } ( \psi _ { t } ( \mathbf x ) | \mathbf x _ { 1 } ) } \end{array}$

Hence, conditional FM (CFM) trains a neural network $\mathbf { v } _ { t } ( \cdot ; \theta )$ to match $\mathbf { u } _ { t } ( \cdot | \mathbf { x } _ { 1 } )$ by minimizing the loss:

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , q ( \mathbf { x } _ { 1 } ) , p _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) } | | \mathbf { v } _ { t } ( \mathbf { x } ; \theta ) - \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) | | _ { 2 } ^ { 2 } .\tag{5}
$$

where $t \sim \mathcal { U } [ 0 , 1 ] , \mathbf { x } _ { 1 } \sim q ( \cdot )$ , and $\mathbf { x } \sim p _ { t } ( \cdot | \mathbf { x } _ { 1 } )$

Starting with the conditional flow $\psi _ { t }$ as a canonical afine transformation in the family of Gaussian distributions $\mathcal { N } ( \mu _ { t } , \sigma _ { t } ^ { 2 } \mathbf { I } )$ :

$$
\psi _ { t } ( \mathbf { x } ) = \pmb { \mu } _ { t } ( \mathbf { x } _ { 1 } ) + \sigma _ { t } ( \mathbf { x } _ { 1 } ) \mathbf { x } ,\tag{6}
$$

[20] prove that the corresponding conditional vector field $\mathbf { u } _ { t } ( \cdot | \mathbf { x } _ { 1 } )$ in the following unique format generates the desired conditional probability path $p _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } )$

$$
\mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) = { \dot { \mu } } _ { t } ( \mathbf { x } _ { 1 } ) + { \frac { { \dot { \sigma } } _ { t } ( \mathbf { x } _ { 1 } ) } { { \sigma } _ { t } ( \mathbf { x } _ { 1 } ) } } ( \mathbf { x } - \mu _ { t } ( \mathbf { x } _ { 1 } ) ) .\tag{7}
$$

In particular, [20] propose and advocate the conditional flow based on the optimal transport (OT) displacement map ψ pushing from $p _ { 0 } ( \cdot | \mathbf { x } _ { 1 } )$ to $p _ { 1 } ( \cdot | \mathbf { x } _ { 1 } )$ [23]:

$$
\psi _ { t } = ( 1 - t ) \mathrm { i d } + t \psi .\tag{8}
$$

The OT displacement between $\mathcal { N } ( \mu _ { 0 } , \sigma _ { 0 } ^ { 2 } \mathbf { I } )$ and $\mathcal { N } ( \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } \mathbf { I } )$ is known to be $\begin{array} { r } { \psi ( \mathbf { x } ) = \pmb { \mu } _ { 1 } + \frac { \sigma _ { 1 } } { \sigma _ { 0 } } ( \mathbf { x } - \pmb { \mu } _ { 0 } ) } \end{array}$ [31]. With the previous Gaussian conditionals $p _ { 0 } ( \cdot | \mathbf { x } _ { 1 } ) = \mathcal { N } ( \cdot ; \mathbf { 0 } , \mathbf { I } )$ and $p _ { 1 } ( \cdot | \mathbf { x } _ { 1 } ) = \mathcal { N } ( \cdot ; \mathbf { x } _ { 1 } , \sigma _ { \mathrm { m i n } } ^ { 2 } \mathbf { I } )$ [20] develop OT based conditional sample flow $( \psi _ { t } ) $ vector field $( \dot { \psi } _ { t } ) $ probability path $( p _ { t } ( \cdot | \mathbf { x } _ { 1 } ) )$ in turn as follows:

$$
\psi _ { t } ( \mathbf { x } ) = ( 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t ) \mathbf { x } + t \mathbf { x } _ { 1 } , \quad \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) = \frac { \mathbf { x } _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) \mathbf { x } } { 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t } ,\tag{9}
$$

$$
p _ { t } ( \cdot | \mathbf { x } _ { 1 } ) = \mathcal { N } ( \cdot ; \mu _ { t } , \sigma _ { t } \mathbf { I } ) , \quad \mu _ { t } = ( 1 - t ) \pmb { \mu } _ { 0 } + t \pmb { \mu } _ { 1 } , \ \sigma _ { t } = ( 1 - t ) \sigma _ { 0 } + t \sigma _ { 1 } ,
$$

where $\mu _ { 0 } = 0 , \mu _ { 1 } = \mathbf { x } _ { 1 } , \sigma _ { 0 } = 1$ and $\sigma _ { 1 } = \sigma _ { \operatorname* { m i n } } > 0 .$

## 2.2 Exponential Power Distribution

Although Gaussian distribution is the default choice of many FM models, it is just a member of a larger location-scale family that also include Cauchy, Laplace, and Student’s-t distributions.

Definition 1 (location-scale family). A location-scale family is a family of probability distributions parametrized by a location parameter $\pmb { \mu } \in \mathbb { R } ^ { d }$ and a positive scale parameter $\sigma > 0$ Suppose Z is a random variable following distribution in this family corresponding to ${ \pmb \mu } = { \bf 0 }$ and $\sigma = 1$ with cumulative distribution function $( C D F ) ~ F _ { 0 } ( z )$ . Then $X = \pmb { \mu } + \sigma Z$ has CDF $F ( X ) = F _ { 0 } { \big ( } ( X - \mu ) / \sigma { \big ) }$

In this paper, we introduce another more flexible member, the exponential power distribution, a.k.a. generalized normal distribution, to flow matching methods for broader choices beyond Gaussian. [13] has a multivariate version of the exponential power distribution defined below.

Definition 2. The exponential power distribution, denoted as $\mathcal { P } _ { q } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ , has the following probability density

$$
p ( \mathbf { x } ) = { \frac { q \Gamma { \big ( } { \frac { d } { 2 } } { \big ) } } { 2 \Gamma { \big ( } { \frac { d } { a } } { \big ) } } } 2 ^ { - { \frac { d } { q } } } \pi ^ { - { \frac { d } { 2 } } } | \mathbf { \Sigma } | ^ { - { \frac { 1 } { 2 } } } \exp \left\{ - { \frac { r ^ { \frac { q } { 2 } } } { 2 } } \right\} , \quad r ( \mathbf { x } ) = { ( \mathbf { x } - { \boldsymbol { \mu } } ) } ^ { \mathsf { T } } \mathbf { \Sigma } \mathbf { \Sigma } ^ { - 1 } ( \mathbf { x } - { \boldsymbol { \mu } } ) .\tag{10}
$$

Remark 1. When $q = 2$ , the exponential power distribution reduces to multivariate normal distribution, $i . e . \mathscr { P } _ { 2 } ( \pmb { \mu } , \pmb { \Sigma } ) = \mathscr { N } ( \pmb { \mu } , \pmb { \Sigma } )$ . Generally, the parameter q controls the regularization around center and tail behavior: the smaller q, the sharper regularization and the heavier tail it induces, as illustrated in Figure 1.

As a special elliptic contoured distribution [16, 10], the exponential power distribution is locationscale, i.e. if $Z \sim \mathcal { P } _ { q } ( \mathbf { 0 } , \mathbf { I } )$ , then $X = \mu + \sigma Z \sim$ $\mathcal { P } _ { q } ( \pmb { \mu } , \sigma ^ { 2 } \mathbf { I } )$ . In this work, we focus on the isotropic case $\pmb { \Sigma } = \sigma ^ { 2 } \mathbf { I }$ and anisotropic case $\pmb { \Sigma } = \mathrm { d i a g } ( \pmb { \sigma } ^ { 2 } )$

The following proposition synthesizes the properties of exponential power distribution [13, 19] useful for the development of new flow matching methods.

![](images/bf153d7b751c2340fc4cf7597791347f79634a791e88338ab3d7ffde960e6353.jpg)  
Figure 1: Probability densities of exponential power distributions $\mathcal { P } _ { q } ( \mathbf { 0 } , \mathbf { I } )$ for selected $\boldsymbol { q } ^ { \flat } \boldsymbol { \mathfrak { s } }$ s.

□

Proposition 2.1. For a d-dimensional random variable $X \sim { \mathcal { P } } _ { q } ( \mu , \sigma ^ { 2 } \mathbf { I } ) $ , we have

1. $\operatorname { E } ( X ) = \mu , \operatorname { C o v } ( X ) = s _ { q , d } \Sigma$ with a scaling factor $\begin{array} { r } { s _ { q , d } = \frac { 2 ^ { \frac { \underline { { \alpha } } } { q } } \Gamma ( \frac { d + 2 } { q } ) } { d \Gamma ( \frac { d } { q } ) } } \end{array}$

2. $X \triangleq \pmb { \mu } + \sqrt { r } \mathbf { L } U _ { S }$ , where $r = \left( X - { \pmb \mu } \right) ^ { \mathsf { T } } { \pmb { \Sigma } } ^ { - 1 } ( X - { \pmb \mu } )$ and $\begin{array} { r } { r ^ { \frac { q } { 2 } } \sim \Gamma ( \alpha = \frac { d } { a } , \beta = \frac { 1 } { 2 } ) , \mathbf { L L } ^ { \mathsf { T } } = \Sigma } \end{array}$ , and $\sqrt { r } \perp U _ { S }$ with $U _ { S } \sim \mathcal { U } ( S ^ { d - 1 } )$ uniformly distributed on the sphere.

## 3 Flexible Flow Path in Location-Scale Family

In this section, we develop a series of novel flow matching methods in the broad location-scale family and propose new algorithms based on exponential power distributions. In addition, we also explore diferent probability paths by minimizing flow energies under various metric tensors, including Euclidean and Fisher-Rao (Section 4).

In the OT theory [31], the OT displacement between two univariate probability distributions $p _ { 0 } ( \cdot | x _ { 1 } )$ and $p _ { 1 } ( \cdot | x _ { 1 } )$ is $\bar { \psi } ( x ) = F _ { 1 } ^ { - 1 } ( F _ { 0 } ( x ) )$ with $F _ { 0 }$ and $F _ { 1 }$ being their CDFs respectively. We immediately have OT interpolant path based on the conditional flow (8) in the one dimensional case. For general dimensions, we may have the following OT-based conditional flow

$$
\psi _ { t } ( \mathbf { x } ) = ( 1 - t ) \mathbf { x } + t \psi ( \mathbf { x } ) , \quad \psi ( \mathbf { x } ) = [ \psi ( x _ { i } ) ] _ { i = 1 } ^ { d } , \quad \psi ( x _ { i } ) = F _ { 1 } ^ { - 1 } ( F _ { 0 } ( x _ { i } ) ) .\tag{11}
$$

Because the exponential power distribution is location-scale, we have $F _ { 1 } ( x ) = F _ { 0 } ( ( x - x _ { 1 } ) / \sigma _ { \mathrm { m i n } } )$ Substituting it into (11) yields the same OT map $\psi ( \mathbf { x } ) = \mathbf { x } _ { 1 } + \sigma _ { \mathrm { m i n } } \mathbf { x }$

This observation motivates us to start the search with any probability path in the broad locationscale family, instead of being restricted to Gaussian distributions and the canonical afine path (6). Note that probability paths $\{ p _ { t } \}$ in the location-scale family parametrized by location and scale coordinates $( \mu _ { t } , \sigma _ { t } )$ form a Riemannian manifold endowed with an information-geometry metric [3], which will be explored in Section 4. The following theorem states that any probability path $p _ { t } ( \cdot ; \mu _ { t } , \sigma _ { t } )$ from the location-scale family can be generated by a determined vector field.

Theorem 3.1. Any $p r o b a b i l i t y \ p a t h \ p _ { t } ( \cdot ; \pmb { \mu _ { t } } , \sigma _ { t } )$ in the location-scale family with pdf p<sub>0</sub> for the standard random variable ${ \bf Z } = ( { \bf X } - { \pmb { \mu } } _ { t } ) / \sigma _ { t }$ has the following vector field $\mathbf { v } _ { t }$ that satisfies the continuity equation (3):

$$
\mathbf { v } _ { t } ( \mathbf { x } ) = \dot { \pmb { \mu } } _ { t } + \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } ( \mathbf { x } - \pmb { \mu } _ { t } ) .\tag{12}
$$

Proof. See Appendix A .

Remark 2. The proof of this theorem is distribution-agnostic in the location-scale family. Though the format of the vector field $\mathbf { v } _ { t }$ to generate probability $p _ { t }$ is the same as in [Theorem 3 of 20], our theorem generalizes it to the location-scale family from which we can use the exponential power distribution $\mathcal { P } _ { q }$ beyond Gaussian $\mathcal { P } _ { 2 } = \mathcal { N }$

Starting with the probability path $p _ { t } ( \cdot ; \mu _ { t } , \sigma _ { t } )$ , Theorem 3.1 determines the conservation law (3) (4) respecting vector field. The following theorem states that such vector field (12) prescribes the flow.

Theorem 3.2. The flow equation (1) with vector field $\mathbf { v } _ { t }$ (12) has the unique solution as an afine transformation:

$$
\phi _ { t } ( \mathbf { x } ) = \pmb { \mu } _ { t } + \sigma _ { t } \mathbf { x } .\tag{13}
$$

Proof. See Appendix A .

Remark 3. Although we end up with the same afine flow, unlike [20], which searches for proper flow first, we can start with any probability path in the vast location-scale family, e.g. Normal, Laplace, Student’s $t ,$ elliptical distributions, then specify the vector field (12) compatible with the continuity law (3), and determine the flow (13) at last. Supported by Theorem 3.1 and Theorem 3.2, the workflow probability path $( p _ { t } ) $ vector field $( \mathbf { v } _ { t } ) $ sample flow $\left( \phi _ { t } \right)$ is reverse to that of [20], which lends us greater flexibility in choosing flow in the FM framework.

We emphasize the novelty and importance of the order probability path $( p _ { t } ) \xrightarrow { T h 3 . 1 }$ vector field $\mathbf { \Gamma } ( \mathbf { v } _ { t } ) \xrightarrow { T h 3 . 2 }$ sample flow $\left( \phi _ { t } \right)$ , which would otherwise no longer guarantee $p _ { t } = [ \phi _ { t } ] _ { \sharp } p _ { 0 }$ to have tractable geodesics when $p _ { 0 }$ moves away from Gaussian, a property lacking in $[ 1 7 , 9 ]$ . Because the conditional probability path $p _ { t } ( \cdot | \mathbf { x } _ { 1 } )$ in the location-scale family is parametrized by location $\pmb { \mu } _ { t }$ and scale $\sigma _ { t } .$ what remains in the CFM framework is to determine them with boundary conditions $( \pmb { \mu } _ { 0 } , \sigma _ { 0 } ) = ( \mathbf { 0 } , 1 )$ and $( \pmb { \mu } _ { 1 } , \sigma _ { 1 } ) = ( \mathbf { x } _ { 1 } , \sigma _ { \operatorname* { m i n } } )$ to drive the conditional vector field $\mathbf { u } _ { t } ( \cdot | \mathbf { x } _ { 1 } ) \ ( 7 )$ and the conditional flow $\psi _ { t } ~ ( 6 )$ . Then a neural network is trained to regress $\mathbf { u } _ { t } ( \cdot | \mathbf { x } _ { 1 } )$ along the flow $\psi _ { t } ( \mathbf { x } _ { 0 } )$ i.e. $\begin{array} { r } { \mathbf { u } _ { t } ( \psi _ { t } ( \mathbf { x } _ { 0 } ) | \mathbf { x } _ { 1 } ) = \frac { d } { d t } \psi _ { t } ( \mathbf { x } _ { 0 } ) } \end{array}$ , with the following loss:

$$
\mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , q ( \mathbf { x } _ { 1 } ) , p ( \mathbf { x } _ { 0 } ) } \| \mathbf { v } _ { t } ( \psi _ { t } ( \mathbf { x } _ { 0 } ) ; \theta ) - \frac { d } { d t } \psi _ { t } ( \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } ,\tag{14}
$$

where $t \sim \mathcal { U } [ 0 , 1 ] , \mathbf { x } _ { 1 } \sim q ( \cdot )$ , and $\mathbf { x } _ { 0 } \sim p _ { 0 } ( \cdot ) = \mathcal { P } _ { q } ( \cdot ; \mathbf { 0 } , \mathbf { I } )$

In the following, we specify the coordinate path $\{ ( \pmb { \mu } _ { t } , \pmb { \sigma } _ { t } ) : t \in [ 0 , 1 ] \}$ on the manifold of locationscale probability distributions $\mathcal { M } = \{ p ( \cdot ; \pmb { \mu } , \sigma ) \}$ by minimizing the flow energy. We will revisit and introduce three probability paths in the family of exponential power distributions: i) minimal energy path; ii) non-linear interpolant path; and iii) (probabilistic) geodesic path.

## 3.1 Minimal Energy Path

OT with quadratic cost is known to be an energy minimization problem under the Wasserstein-2 metric [5]. To determine the best conditional vector field, we adopt the energy criterion for the flow:

$$
e ( \mathbf { v } ) = \mathbb { E } _ { p _ { t } ( \mathbf { x } ) } \int _ { 0 } ^ { 1 } \| \dot { \mathbf { x } } _ { t } \| ^ { 2 } d t = \int _ { \mathbb { R } ^ { d } } \int _ { 0 } ^ { 1 } p _ { t } ( \mathbf { x } ) \| \mathbf { v } _ { t } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } d t d \mathbf { x } .\tag{15}
$$

Minimizing the energy (15) leads to geodesic in the Euclidean space, as stated below.

Proposition 3.1. The following OT interpolant path minimizes the flow energy (15):

$$
\mu _ { t } = ( 1 - t ) \mu _ { 0 } + t \mu _ { 1 } = t \mathbf { x } _ { 1 } , \quad \sigma _ { t } = ( 1 - t ) \sigma _ { 0 } + t \sigma _ { 1 } = 1 - t + t \sigma _ { \operatorname* { m i n } } .\tag{16}
$$

The minimal energy attained is $e _ { \mathrm { m i n } } = \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| ^ { 2 } + d | \sigma _ { 1 } - \sigma _ { 0 } | ^ { 2 }$

Proof. See Appendix A .

Therefore, we have the conditional flow and vector field along the minimal energy path for loss (14):

$$
\psi _ { t } ( \mathbf { x } ) = ( 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t ) \mathbf { x } + t \mathbf { x } _ { 1 } , \quad \frac { d } { d t } \psi _ { t } ( \mathbf { x } ) = \mathbf { x } _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) \mathbf { x } .
$$

## 3.2 Nonlinear Interpolant Path

The OT-interpolant (16) is a linear interpolation between the starting $( \mu _ { 0 } , \sigma _ { 0 } )$ and ending $( \mu _ { 1 } , \sigma _ { 1 } )$ points on the manifold. A popular nonlinear alternative involves the functions sin and cos [21]:

$$
\begin{array} { r } { \mathbf { x } _ { t } = \psi _ { t } ( \mathbf { x } ) = \mathbf { x } \cos ( \pi t / 2 ) + \mathbf { x } _ { 1 } \sin ( \pi t / 2 ) . } \end{array}\tag{17}
$$

By $\begin{array} { r } { \frac { d } { d t } \psi _ { t } ( \mathbf { x } ) = \mathbf { u } _ { t } ( \psi _ { t } ( \mathbf { x } ) | \mathbf { x } _ { 1 } ) } \end{array}$ , we can derive the conditional vector field as $\begin{array} { r } { \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) = \frac { \pi } { 2 } \frac { \mathbf { x } _ { 1 } - \mathbf { x } \sin ( \pi t / 2 ) } { \cos ( \pi t / 2 ) } } \end{array}$ , which generates the conditional probability path $p _ { t } ( \cdot ; \mu _ { t } , \sigma _ { t } )$ in the location-scale family with coordinates: $\pmb { \mu } _ { t } ( \mathbf { x } _ { 1 } ) = \mathbf { x } _ { 1 } \sin ( \pi t / 2 ) , \quad \sigma _ { t } ( \mathbf { x } _ { 1 } ) = \cos ( \pi t / 2 )$ , although it does not minimize the energy (15). Later we will show that it is actually a special case of the probabilistic geodesic path on the manifold of exponential power distributions with Fisher-Rao metric (Section 4).

## 4 Probabilistic Geodesic Flow

To search for the optimal probability path $\{ p _ { t } ( \cdot ; \pmb { \mu } _ { t } , \sigma _ { t } ) : t \in [ 0 , 1 ] \}$ in the location-scale family, it is natural to adopt the Riemannian manifold of probability distributions $\mathcal { M } = \{ p ( \cdot ; \pmb { \mu } , \sigma ) \}$ with Fisher-Rao metric $g \ [ 3 ]$ . From the manifold perspective, we identify $( \pmb { \mu } _ { t } , \sigma _ { t } )  p _ { t } ( \cdot ; \pmb { \mu } _ { t } , \sigma _ { t } )$ . Although the methodology introduced applies to general location-scale distributions, we focus on the exponential power distribution $\mathcal { P } _ { q } ( \mu , \sigma ^ { 2 } \mathbf { I } )$ . Lemma A.1 shows that the Fisher-Rao metric (23) for this manifold is $\begin{array} { r } { d s ^ { 2 } = \frac { c _ { \mu } \| d \pmb { \mu } \| ^ { 2 } + c _ { \sigma } d \sigma ^ { 2 } } { \sigma ^ { 2 } } } \end{array}$ with $\begin{array} { r } { c _ { \mu } ( d , q ) = \frac { 2 ^ { - \frac { 2 } { q } } } { d } \frac { \Gamma ( \frac { d - 2 } { q } + 2 ) } { \Gamma ( \frac { d } { q } ) } q ^ { 2 } } \end{array}$ , and $c _ { \sigma } = q d .$ . Note that if $X \sim \mathcal { P } _ { q } ( \mu _ { t } , \sigma _ { t } ^ { 2 } \mathbf { I } )$ then $\mathbf { v } _ { t } ( X ) \sim \mathcal { P } _ { q } ( \dot { \mu } _ { t } , \dot { \sigma } _ { t } ^ { 2 } \mathbf { I } )$ by (12). The flow energy (15) can be redefined by replacing the Euclidean metric with the Fisher-Rao metric (23):

$$
e ( \mathbf { v } ) = \mathbb { E } _ { p _ { t } ( \mathbf { x } ) } \int _ { 0 } ^ { 1 } \| \dot { \mathbf { x } } _ { t } \| _ { g } ^ { 2 } d t = \int _ { \mathbb { R } ^ { d } } \int _ { 0 } ^ { 1 } p _ { t } ( \mathbf { x } ) \| \mathbf { v } _ { t } ( \mathbf { x } ) \| _ { g } ^ { 2 } d t d \mathbf { x } = \int _ { 0 } ^ { 1 } \frac { c _ { \mu } \| \dot { \mu } _ { t } \| ^ { 2 } + c _ { \sigma } \dot { \sigma } _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } d t .\tag{18}
$$

By the calculus of variation, we get the geodesic as a semi-circle on the manifold of exponential power distributions $\mathscr { P } _ { q } ( \mu _ { t } , \sigma _ { t } ^ { 2 } \mathbf { I } )$ in the following theorem.

Theorem 4.1. The following geodesic path minimizes the variational energy (18):

$$
\mu _ { t } = \mu _ { 0 } + x ( t ) \mathbf { e } , \quad x ( t ) = \mu _ { c } + R \cos \theta ( t ) , \quad R = \sqrt { \mu _ { c } ^ { 2 } + \lambda ^ { 2 } \sigma _ { 0 } ^ { 2 } } ;\tag{19a}
$$

$$
\sigma _ { t } = \frac { R } { \lambda } \sin \theta ( t ) , \quad \theta ( t ) = 2 \arctan \left( e ^ { L t } \tan \frac { \theta _ { 0 } } { 2 } \right) , \quad L = \log \frac { \tan ( \theta _ { 1 } / 2 ) } { \tan ( \theta _ { 0 } / 2 ) } ,\tag{19b}
$$

where $\begin{array} { r } { \lambda = \sqrt { c _ { \sigma } / c _ { \mu } } , { \bf e } = ( \mu _ { 1 } - \mu _ { 0 } ) / \| { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } \| _ { 2 } , \mu _ { c } = \frac { \| { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } \| ^ { 2 } + \lambda ^ { 2 } ( \sigma _ { 1 } ^ { 2 } - \sigma _ { 0 } ^ { 2 } ) } { 2 \| { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } \| } } \end{array}$ , and $\theta _ { 0 } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 0 } , - \mu _ { c } )$ $\theta _ { 1 } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 1 } , \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| - \mu _ { c } ) .$ . The minimal energy attained is $e _ { \mathrm { m i n } } = c _ { \sigma } L ^ { 2 } .$ , where $\sqrt { c _ { \sigma } } | L |$ is the Fisher-Rao distance between $\mathcal { P } _ { q } ( \mu _ { 0 } , \sigma _ { 0 } ^ { 2 } \mathbf { I } )$ and $\mathscr { P } _ { q } ( \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } \mathbf { I } )$ , for which $| L |$ can be rewritten in the original parameters as

$$
| L | = \operatorname { a r c o s h } \left( 1 + { \frac { c _ { \mu } \| \mu _ { 1 } - \mu _ { 0 } \| ^ { 2 } + c _ { \sigma } | \sigma _ { 1 } - \sigma _ { 0 } | ^ { 2 } } { 2 c _ { \sigma } \sigma _ { 0 } \sigma _ { 1 } } } \right) .\tag{20}
$$

Proof. See Appendix A.

Although mathematically complicated, this analytically tractable formula has numerical complexity $\mathcal O ( d )$ , which becomes negligible compared to network training in higher dimensions, as evidenced by the training time reported in Section 5 and Section B. We also have an anisotropic version that decouples the isotropic geodesic into d independent one-dimensional geodesics (Appendix C).

In Theorem A.1, the CFM loss (14) along the geodesic in Theorem 4.1 decomposes into the FM loss and the variance of conditional vector field and is bounded from below.

In particular, we get the vector field $\mathbf { v } _ { t } \ ( 1 2 )$ that generates the exponential power probability path $\mathscr { P } _ { q } ( \mu _ { t } , \sigma _ { t } ^ { 2 } \mathbf { I } )$ , and the corresponding flow $\phi _ { t } ( \mathbf { x } )$ (13). Letting $\pmb { \mu } _ { 0 } = \mathbf { 0 } , \pmb { \mu } _ { 1 } = \mathbf { x } _ { 1 } , \sigma _ { 0 } = 1 , \sigma _ { 1 } = \sigma _ { \operatorname* { m i n } }$ , we have the conditional flow and vector field along the probabilistic geodesic path to feed in (14):

$$
\begin{array} { c } { { \displaystyle \psi _ { t } ( { \bf x } ) = \mu _ { t } ( { \bf x } _ { 1 } ) + \sigma _ { t } ( { \bf x } _ { 1 } ) { \bf x } = x ( t ) { \bf x } _ { 1 } / \| { \bf x } _ { 1 } \| + \frac { R } { \lambda } \sin \theta ( t ) { \bf x } , } } \\ { { \displaystyle \frac { d } { d t } \psi _ { t } ( { \bf x } ) = { \bf u } _ { t } ( \psi _ { t } ( { \bf x } ) | { \bf x } _ { 1 } ) = \left[ - \sin \theta ( t ) \frac { { \bf x } _ { 1 } } { \| { \bf x } _ { 1 } \| } + \frac { 1 } { \lambda } \cos \theta ( t ) { \bf x } \right] R L \sin \theta ( t ) . } } \end{array}\tag{21}
$$

Interestingly, when $R = \lambda = \| \mathbf { x } _ { 1 } \|$ , letting $\sigma _ { \mathrm { m i n } } \downarrow 0$ , we have $\begin{array} { r } { \mu _ { c } = \frac { \| \mathbf { x } _ { 1 } \| ^ { 2 } + \lambda ^ { 2 } ( \sigma _ { \operatorname* { m i n } } ^ { 2 } - 1 ) } { 2 \| \mathbf { x } _ { 1 } \| } \to 0 } \end{array}$ . Therefore $x ( t )  R \cos \theta ( t )$ . The conditional flow along this special geodesic path reduces to

$$
\psi _ { t } ( \mathbf { x } ) = x ( t ) \mathbf { x } _ { 1 } / \| \mathbf { x } _ { 1 } \| + { \frac { R } { \lambda } } \sin \theta ( t ) \mathbf { x } \to \cos \theta ( t ) \mathbf { x } _ { 1 } + \sin \theta ( t ) \mathbf { x } ,
$$

which is exactly the sinusoidal path (17) with $\theta ( t ) = \pi ( 1 - t ) / 2$

FM methods transfer samples such that the distribution they represent changes from pure noise (standard normal or exponential power) to the data distribution. The ultimate goal is to evolve distributions, not to transport samples. OT based approaches move samples along straight lines in the Euclidean space. However, they do not necessarily travel the shortest distances in the space of probability distributions. The leftmost panel of Figure A.1 shows that the Fisher-Rao distance may be smaller than the Euclidean distance between two exponential power distributions $\mathcal { P } _ { q } ( \mathbf { 0 } , \mathbf { I } )$ and $\mathcal { P } _ { q } ( \mathbf { 1 } , \sigma _ { \operatorname* { m i n } } ^ { 2 } \mathbf { I } )$ for certain $q ^ { * } \mathrm { s } .$ . The other panels of Figure A.1 demonstrate selected probabilistic geodesic paths (second from left) and contrast their corresponding flow paths for $q = 1 . 0 \ \mathrm { v s } \ q = 2 . 0$ (rightmost two) – smaller q leads to a more stringent movement than larger q.

![](images/5fa566b4e4c1c2042536cc1a6cf162e275f2b6cf5cd3aa59fb8510f0153117f2.jpg)  
Figure 2: Contrasting trajectories of energy distance (ED) in the Swiss roll, moons, checkerboard, and funnel examples (left four) and class Frechet inception distance (clf-FID) in MNIST (rightmost).

Limitation There is a numerical issue on the conditional vector field that aggravates with increasing dimensions. When $\sigma _ { 1 }  0 .$ , the geodesic length $| L | \to \infty$ as in (20). As $t \to 1 , \sigma _ { t } \to \sigma _ { \operatorname* { m i n } } \ll 1$ , and the conditional vector field $\dot { \psi } _ { t }$ in (21) fluctuates greatly in magnitude as the flow moves towards the data end. Therefore, the vector field learned by neural network would cause the resulted ODE stif and numerically dificult to solve. Luckily, this limitation can be alleviated by a hybrid (HB):

$$
\psi _ { t } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { \pmb { \mu _ { t } } ( \mathbf { x } _ { 1 } ) + \sigma _ { t } ( \mathbf { x } _ { 1 } ) \mathbf { x } , } & { t < t _ { c } , } \\ { t \mathbf { x } _ { 1 } + ( 1 - t ) \mathbf { x } , } & { t \geq t _ { c } . } \end{array} \right.
$$

where we use $t _ { c } = 0 . 8 5$ in numerical experiments.

## 5 Numerical Results

In this section, we compare the proposed probabilistic geodesic (PG) FM and its hybrid variant (HB) against baseline models with variance preserving (VP), optimal transport (OT) interpolant, and sinusoidal (sino) probability paths. To fully test their performance, we also include the stateof-the-art models Riemannian FM [RFM 6], Metric FM [MFM 17], and Fisher FM [FFM 9] in the comparison. Three 2d benchmark examples including Swiss roll, moons, and checkerboard, higher dimensional simulations of an augmented eight-Gaussian and Neal’s funnel distribution, handwritten digits, four scientific datasets (Appendix B.4), and a molecular generation problem are used in the test. We refer to Wasserstein-2 distance $( W _ { 2 } )$ , energy distance (ED), and maximum mean discrepancy (MMD) to measure the discrepancy between synthetic sample distribution and real data distribution. One can refer to Appendix B.1 for more details of the setup. All the computer code will be released.

## 5.1 Simulations

## 5.1.1 2d Benchmarks: swiss roll, moons, and checkerboard

First, we test VP, OT, sino, and PG-2 (on Gaussian distributions) and PG-1 (on exponential power distributions $\mathcal { P } _ { 1 } ( \mu _ { t } , \sigma _ { t } ) )$ on three 2d benchmark datasets: Swiss roll, moons and checkerboard [14]. Figure B.1 demonstrates their sample paths on the Swiss roll dataset. It is clear that PG methods are more efective in transporting samples in which the rolling patterns emerge in earlier stages. Starting with the more concentrated standard exponential power noise, PG-1 learns the data feature even faster than the others based on standard Gaussian noise. Similar quick feature learning also shows in the other two examples (Figures B.1 and B.2). The metrics in terms of $W _ { 2 } .$ , ED, and MMD in Tables B.1 and B.2 also support the numerical advantage (smaller generative discrepancy) of PG methods, of which some are one order of magnitude smaller. The time overhead (time/epoch) fades of as the dimension increases (see Tables 1, B.5, and B.4).

Figure 2 plots the trajectories of ED on these benchmark datasets and clf-FID on MNIST. OT interpolant with independent coupling $( \mathbf { x } _ { 0 } \perp \mathbf { x } _ { 1 } )$ features an “energy hump” (refer to Remark 7) in the middle of the path due to the well-known midpoint (marginal) variance contraction [2, 30]. This does not present an issue for our PG paths as their energy distances to the target distribution decrease quickly and monotonically.

![](images/847c3d16952abceb680f97b3df70258761af106785463e76cfd54c0beb155653.jpg)  
Figure 3: Sample paths in the 8-Gaussians (36d) example. Only the first two dimensions are plotted.

Ablation on q. As illustrated in Figures 1 and Figure A.1, the parameter q in the exponential power distribution plays a role of regularization: smaller q induces a more strict movement of the sample. However, that does not necessarily lead to better performance when training becomes more dificult in the non-convex regime $( q < 1 . 0 )$ . In Table B.3, these discrepancy metrics generally decrease and then increase as q changes from above 2 to below 1.

## 5.1.2 Higher Dimensions: 36d eight-Gaussians, 100d Funnel Distribution

To test the robustness of PG methods to higher dimensionality, we extend the classical eight-Gaussians example [14] from 2d to d = 36 dimensions by filling in the remaining d − 2 coordinates with isotropic Gaussian noise. To test the capability of PG methods of handling heavy-tail data, we investigate the more challenging Neal’s funnel distribution [24] whose density $f ( \mathbf { x } , \nu ) = \mathcal { N } _ { d } ( \mathbf { x } ; \mathbf { 0 } , e ^ { \nu / 2 } \mathbf { I } ) \mathcal { N } ( \nu ; 0 , 3 )$ features a funnel (Figure B.2).

Table 1: Neal’s funnel (100d): comparison in $W _ { 2 }$ , ED, and MMD (mean ± std over 10 seeds).
<table><tr><td>DIST</td><td>Method</td><td> $W _ { 2 } \downarrow$ </td><td>ED↓</td><td>MMD ↓</td><td>time/epoch</td></tr><tr><td>q=1.0</td><td>OT</td><td> $3 . 7 3 e 0 \pm 2 . 4 9 e - 1$ </td><td> $1 . 4 3 e - 1 \pm 1 . 3 5 e - 2$ </td><td> $3 . 5 2 e - 3 \pm 1 . 0 7 e - 3$ </td><td> $5 . 9 0 e { - 2 } \pm 6 . 8 5 e { - 5 }$ </td></tr><tr><td>q=2.0</td><td>OT</td><td> $3 . 7 4 e 0 \pm 2 . 0 0 e - 1$ </td><td> $1 . 4 1 e { - } 1 \pm 1 . 1 8 e { - } 2$ </td><td> $3 . 3 2 e - 3 \pm 7 . 6 0 e - 4$ </td><td> $5 . 7 6 e - 2 \pm 8 . 2 1 e - 5$ </td></tr><tr><td>q=1.0</td><td>RFM</td><td> $\overline { { 3 . 6 5 e 0 \pm 2 . 0 1 e - 1 } }$ </td><td> $\overline { { 1 . 2 7 e - 1 \pm 1 . 3 3 e - 2 } }$ </td><td> $2 . 3 0 e { \mathrm { - } } 3 \pm 7 . 1 7 e { \mathrm { - } } 4$ </td><td> $\overline { { 7 . 6 8 e - 2 \pm 6 . 5 9 e - 4 } }$ </td></tr><tr><td>q=2.0</td><td>RFM</td><td> $3 . 6 9 e 0 \pm 2 . 1 2 e - 1$ </td><td> $1 . 2 9 e - 1 \pm 1 . 1 0 e - 2$ </td><td> $2 . 4 8 e - 3 \pm 5 . 2 9 e - 4$ </td><td>7.11e−2 ± 5.74e−4</td></tr><tr><td>q=1.0</td><td>MFM</td><td> $\overline { { 4 . 0 5 e 0 \pm 2 . 3 9 e - 1 } }$ </td><td> $1 . 2 6 e { - } 1 \pm 1 . 1 6 e { - } 2$ </td><td> $2 . 6 5 e - 3 \pm 7 . 5 7 e - 4$ </td><td>4.36e-1 ± 1.57e-2</td></tr><tr><td>q=2.0</td><td>MFM</td><td> $4 . 1 7 e 0 \pm 1 . 8 3 e - 1$ </td><td> $1 . 2 7 e - 1 \pm 1 . 3 1 e - 2$ </td><td> $2 . 8 0 e { - } 3 \pm 5 . 9 5 e { - } 4$ </td><td>4.26e−1 ± 1.62e−2</td></tr><tr><td>q=1.0</td><td>FFM</td><td> $\overline { { 6 . 1 8 e 0 \pm 5 . 9 0 e - 1 } }$ </td><td> $\overline { { 1 . 0 0 e 0 \pm 7 . 2 5 e - 2 } }$ </td><td> $\overline { { 1 . 1 6 e - 1 \pm 8 . 2 7 e - 3 } }$ </td><td>1.00e−2 ± 2.52e−5</td></tr><tr><td>q=2.0</td><td>FFM</td><td> $5 . 5 5 e 0 \pm 6 . 6 4 e - 1$ </td><td> $8 . 9 9 e - 1 \pm 7 . 9 5 e - 2$ </td><td> $1 . 0 1 e - 1 \pm 8 . 4 9 e - 3$ </td><td>8.69e-3 ± 1.38e-5</td></tr><tr><td>q=1.0</td><td>PG</td><td>3.76e0 ± 3.06e−1</td><td> $\overline { { 1 . 8 2 e - 1 \pm 1 . 0 4 e - 2 } }$ </td><td> $\overline { { 5 . 9 4 e - 3 \pm 8 . 6 4 e - 4 } }$ </td><td>6.51e−2 ± 1.39e−4</td></tr><tr><td>q=2.0</td><td>PG</td><td>4.58e0 ± 6.98e−1</td><td> $1 . 6 2 e - 1 \pm 2 . 6 4 e - 2$ </td><td> $4 . 4 8 e { - } 3 \pm 1 . 3 4 e { - } 3$ </td><td>6.33e-2 ± 1.08e-4</td></tr><tr><td>q=1.0</td><td>HB</td><td> $2 . 9 9 e 0 \pm 2 . 3 8 e - 1$ </td><td> $1 . 2 6 e - 1 \pm 1 . 6 9 e - 2$ </td><td> $2 . 3 9 e - 3 \pm 6 . 5 9 e - 4$ </td><td> $\overline { { 6 . 9 2 e - 2 \pm 2 . 6 9 e - 4 } }$ </td></tr><tr><td>q=2.0</td><td>HB</td><td> $3 . 6 7 e 0 \pm 3 . 2 7 e - 1$ </td><td> $\overline { { 1 . 8 4 e { - } 1 \pm 1 . 2 5 e { - } 1 } }$ </td><td> $8 . 3 4 e { - } 3 \pm 1 . 5 5 e { - } 2$ </td><td> $6 . 6 1 e { - 2 } \pm 3 . 4 2 e { - 4 }$ </td></tr></table>

Why non-Gaussianity matters? Figure 3 compares the sample paths of the baseline (OT, sino) and SOTA (RFM, MFM, FFM) models with the proposed PG and HB models. It challenges all the methods to discover eight 2d Gaussian blobs in 36 dimensional ambient space, yet PG-1 and HB are the fastest to identify all the modes. What is more, the exponential power distribution’s heavier tail property (with smaller q as in PG-1) helps identify and isolate those modes faster and better than Gaussian $( q = 2$ as in PG-2) which tends to smooth and blur the pattern. This explains why most $q = 1 . 0$ rows have lower discrepancy metrics than $q = 2 . 0$ (except RFM and MFM) in Table B.4 for which PG-1 achieves the lowest ED and MMD. In Table 1, except for a few cells, $q = 1 . 0$ almost unanimously beats $q = 2 . 0$ , which again justifies the adoption of exponential power distribution. HB attains the lowest $W _ { 2 }$ and ED for $q = 1 . 0$ . Note that MFM reaches the same lowest ED at a much higher time cost $( 4 . 3 6 e - 1 ~ \mathrm { v s } ~ 6 . 9 2 e - 2$ seconds per epoch for HB).

## 5.2 MNIST Handwritten Digits

In this subsection, we consider the MNIST dataset [18] consisting of 60,000 training handwritten digits and 10,000 testing handwritten digits with $d = 2 8 \times 2 8 = 7 8 4$ . The sharp contrast along the edge of these digits challenges FM models. Here we focus on comparing the OT, PG, and HB paths.

![](images/20dff7a2cb32828a345386b6665c889a3ab37083cd8ae1246c85433c324e390b.jpg)  
Figure 4: sample handwritten digits.

Table 2: Comparison of generation quality metrics across diferent path methods. Values are reported as mean ± standard deviation over $n = 5$ runs. Higher IS is better, while lower FID and clf-FID are better.
<table><tr><td>DIST</td><td>PATH</td><td>IS↑</td><td>FID ↓</td><td>clf-FID ↓</td></tr><tr><td>q=1.0</td><td>OT</td><td> $2 . 0 9 9 \pm 0 . 0 0 8$ </td><td> $1 . 6 0 6 \pm 0 . 0 3 7$ </td><td> $0 . 7 1 8 \pm 0 . 1 0 8$ </td></tr><tr><td>q=2.0</td><td>OT</td><td>_  $2 . 1 0 7 \pm 0 . 0 0 3$  </td><td> $1 . 5 9 7 \pm 0 . 0 1 9$ </td><td>1  $0 . 7 7 0 \pm 0 . 0 8 1$ </td></tr><tr><td>q=1.0</td><td>PG</td><td> $\overline { { 2 . 0 9 5 \pm 0 . 0 0 9 } }$ </td><td> $\overline { { 1 . 6 7 1 \pm 0 . 0 1 3 } }$ </td><td> $\overline { { 0 . 9 0 9 \pm 0 . 1 3 9 } }$ </td></tr><tr><td>q=2.0</td><td>PG</td><td> $2 . 0 8 3 \pm 0 . 0 0 6$ </td><td> $2 . 2 8 1 \pm 0 . 0 9 7$ </td><td> $1 . 8 4 3 \pm 0 . 2 7 9$ </td></tr><tr><td>q=1.0</td><td>HB</td><td> $2 . 1 0 7 \pm 0 . 0 0 8$  1</td><td> $\overline { { 1 . 7 2 7 \pm 0 . 0 2 9 } }$ </td><td> $\overline { { 0 . 5 6 7 \pm 0 . 1 1 9 } }$ </td></tr><tr><td>q=2.0</td><td>HB</td><td> $2 . 1 0 2 \pm 0 . 0 0 8$ </td><td> $1 . 8 0 3 \pm 0 . 0 4 0$ </td><td>_  $0 . 5 5 6 \pm 0 . 0 7 1$  </td></tr></table>

In Table 2, we evaluate the sample quality of OT, PG, and HB under both $q = 1$ and $q = 2$ in terms of the inception score (IS) and the Fr´echet inception distance (FID), based on ImageNetpretrained Inception-V3 activations. Additionally, we report a classifier FID (clf-FID), computed identically but on penultimate-layer features from a LeNet-5 [18] classifier trained on MNIST, which makes the metric sensitive to domain-specific structure that inception features do not capture. While OT achieves the highest IS and the lowest FID, HB attains the highest IS (same) and the lowest clf-FID. Figure 4 illustrates some generated digits.

## 5.3 Molecular Generation

Lastly, we consider a real data example of thousands of dimensions – de novo molecule generation [Section 4.4 of 9], which aims to generate molecules positions unconditionally based on QM9 data [28, 27]. Refer to Appendix B.5 for a detailed description of the problem. Table 3 reports the percentage of stable atoms within molecules (Atoms S), valid molecules (Mols Val), and stable molecules (Mols. S) based on 1,000 molecules (first 4 rows quoted from Table 3 of [9]). For a fare comparison, all the noise distributions are Gaussian. PG/HB is slightly lower in the percentage of valid molecules (Mols Val), but has higher percentages of stable atoms within molecules (Atoms S) and stable molecules (Mols. S) than OT, FFM and FlowMol. Figure 5 shows synthetic molecules generated by PG.

![](images/712013cc0eb5a73356083e1f440802307533922c4a623fe07a1c2e74e02d1d47.jpg)  
Figure 5: Generated molecules by PG-2.

Table 3: Results of molecular generation on QM9: the percentage of stable atoms within molecules (Atoms S), valid molecules (Mols Val), and stable molecules (Mols. S) based on 1,000 molecules.
<table><tr><td>Method</td><td>Atoms S. (%) ↑</td><td>Mols. Val. (%) ↑ Mols. S. (%) ↑</td></tr><tr><td>Fisher-Flow</td><td>98.6</td><td>95.3 88.2</td></tr><tr><td>JODO</td><td>99.4</td><td>98.9 98.7</td></tr><tr><td>EquiFM</td><td>99.4</td><td>94.4 93.2</td></tr><tr><td>FlowMol</td><td>98.9</td><td>96.9 84.2</td></tr><tr><td>OT</td><td>99.35</td><td>89.80 91.60</td></tr><tr><td>PG</td><td>99.49</td><td>90.63 93.75</td></tr><tr><td>HB</td><td>99.39</td><td>89.84 92.97</td></tr></table>

## 6 Conclusion

In this paper, we extend the conditional flow matching from Gaussian to a broader location-scale family with a particular focus on exponential power distributions that embrace Gaussian as a special case $q = 2$ . We propose a class of novel probability paths as geodesics on a manifold of probability distributions with Fisher-Rao metric. The proposed methods demonstrate fast feature learning capability with numerical performance superior or comparable to SOTA FM models.

The probabilistic geodesic path is not as conceptually straightforward as the OT interpolant path. The computational overhead diminishes as the dimension increases, compared to the cost of network training. Future directions include integration of the proposed probabilistic geodesic path into recent one-step methods [11, 12] to further boost the eficiency and efectiveness of FM.

## AI use statement

In this work, we used generative AI tools for polishing presentation of the paper (refining sentences and paragraphs). We have not used generative AI tools for writing the whole paper. In addition, we used generative AI tools for assistance in coding and debuging. We have reviewed all AI-assisted work. For example, the LLM-generated code was verified and tested for correctness by all co-authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## Reproducibility statement

All results are reproducible using python codes that will be published after publication. We include a jupyter notebook of a demonstrative example in the submitted supplementary materials.

## References

[1] M. S. Albergo. Flow-based generative models for markov chain monte carlo in lattice field theory. Physical Review D, 100(3), 2019.

[2] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023.

[3] S. Amari and H. Nagaoka. Methods of Information Geometry, volume 191 of Translations of Mathematical monographs. Oxford University Press, 2000.

[4] Lazar Atanackovic, Xi Zhang, Brandon Amos, Mathieu Blanchette, Leo J Lee, Yoshua Bengio, Alexander Tong, and Kirill Neklyudov. Meta flow matching: Integrating vector fields on the wasserstein manifold. In The Thirteenth International Conference on Learning Representations, 2025.

[5] Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the mongekantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, Jan 2000.

[6] Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In The Twelfth International Conference on Learning Representations, 2024.

[7] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary diferential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

[8] Chaoran Cheng, Jiahan Li, Jian Peng, and Ge Liu. Categorical flow matching on statistical manifolds. In Advances in Neural Information Processing Systems 37, NeurIPS 2024, pages 54787–54819. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024.

[9] Oscar Davis, Samuel Kessler, Mircea Petrache, Ismail Ilkan Ceylan, Michael M. Bronstein, and Joey Bose. Fisher flow matching for generative modeling over discrete data. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[10] K. Fang and Y.T. Zhang. Generalized Multivariate Analysis. Science Press, 1990.

[11] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step difusion via shortcut models. In The Thirteenth International Conference on Learning Representations, 2025.

[12] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

[13] E. G´omez, M.A. Gomez-Viilegas, and J.M. Mar´ın. A multivariate generalization of the power exponential family of distributions. Communications in Statistics - Theory and Methods, 27(3):589–600, jan 1998.

[14] Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, and David Duvenaud. Scalable reversible generative models with free-form continuous dynamics. In International Conference on Learning Representations, 2019.

[15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.

[16] Mark E. Johnson. Multivariate Statistical Simulation, chapter 6 Elliptically Contoured Distributions, pages 106–124. Probability and Statistics. John Wiley & Sons, Ltd, 1987.

[17] Kacper Kapusniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong, Michael Bronstein, Avishek Bose, and Francesco Di Giovanni. Metric flow matching for smooth interpolations on the data manifold. In Advances in Neural Information Processing Systems 37, NeurIPS 2024, pages 135011– 135042. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2024.

[18] Y. Lecun, L. Bottou, Y. Bengio, and P. Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[19] Shuyi Li, Michael O' Connor, and Shiwei Lan. Bayesian learning via q-exponential process. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the 37th Conference on Neural Information Processing Systems, volume 36, pages 72867–72887. Curran Associates, Inc., 2023.

[20] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

[21] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv:2412.06264, 12 2024.

[22] Dimitra Maoutsa, Sebastian Reich, and Manfred Opper. Interacting particle solutions of fokker–planck equations through gradient–log–density estimation. Entropy, 22(8):802, July 2020.

[23] Robert J. McCann. A convexity principle for interacting gases. Advances in Mathematics, 128(1):153– 179, 1997.

[24] Radford M. Neal. Slice sampling. The Annals of Statistics, 31(3), jun 2003.

[25] Frank No´e, Simon Olsson, Jonas K¨ohler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457):eaaw1147, 2019.

[26] Kushagra Pandey, Jaideep Pathak, Yilun Xu, Stephan Mandt, Michael Pritchard, Arash Vahdat, and Morteza Mardani. Heavy-tailed difusion models. In The Thirteenth International Conference on Learning Representations, 2025.

[27] Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data, 1(1), August 2014.

[28] Lars Ruddigkeit, Ruud van Deursen, Lorenz C. Blum, and Jean-Louis Reymond. Enumeration of 166 billion organic small molecules in the chemical universe database gdb-17. Journal of Chemical Information and Modeling, 52(11):2864–2875, November 2012.

[29] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations, 2021.

[30] Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024. Expert Certification.

[31] C´edric Villani. Optimal Transport. Springer Berlin Heidelberg, 2009.

[32] F. Alexander Wolf, Philipp Angerer, and Fabian J. Theis. Scanpy: large-scale single-cell gene expression data analysis. Genome Biology, 19(1):15, Feb 2018.

Table .4: Comparison of flow-matching methods.
<table><tr><td>Method</td><td>Manifold of</td><td>Metric</td><td>Geodesic</td><td>Data type</td></tr><tr><td>Riemannian FM [6]</td><td>general (mainly data)</td><td>standard Riemannian</td><td>general</td><td>non-Euclidean</td></tr><tr><td>Metric FM [17]</td><td>Data</td><td>learned</td><td>general</td><td>general</td></tr><tr><td>Meta FM [4]</td><td>probability densities</td><td>2-Wasserstein</td><td>general</td><td>gmbedded in graph</td></tr><tr><td>Categorical FM [8]</td><td>probability measures</td><td>Fisher-Rao</td><td>circle</td><td>discrete</td></tr><tr><td>Fisher FM [9]</td><td>probability densities</td><td>Fisher-Rao</td><td>circle</td><td>discrete</td></tr><tr><td>PG FM (ours)</td><td>probability densities</td><td>Fisher-Rao</td><td>Poincaré semicircle</td><td>continuous, Euclidean</td></tr></table>

## A Proofs

Theorem 3.1. Any probability path $p _ { t } ( \cdot ; \mu _ { t } , \sigma _ { t } )$ in the location-scale family with pdf p<sub>0</sub> for the standard random variable ${ \bf Z } = ( { \bf X } - { \pmb { \mu } } _ { t } ) / \sigma _ { t }$ has the following vector field $\mathbf { v } _ { t }$ that satisfies the continuity equation (3):

$$
\mathbf { v } _ { t } ( \mathbf { x } ) = \dot { \pmb { \mu } } _ { t } + \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } ( \mathbf { x } - \pmb { \mu } _ { t } ) .\tag{12}
$$

Proof of Theorem 3.1. Because $\mathbf { Z } \sim p _ { 0 }$ , we have

$$
{ \bf X } _ { t } = \pmb { \mu _ { t } } + \sigma _ { t } \pmb { \mathrm { Z } } \sim p _ { t } , \quad p _ { t } ( \mathbf { x } ) = p _ { 0 } ( ( \mathbf { x } - \pmb { \mu _ { t } } ) / \sigma _ { t } ) \sigma _ { t } ^ { - d } .\tag{22}
$$

Along this trajectory, we have

$$
\dot { \mathbf { X } } _ { t } = \dot { \pmb { \mu } } _ { t } + \dot { \sigma } _ { t } \mathbf { Z } = \dot { \pmb { \mu } } _ { t } + \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } ( \mathbf { X } _ { t } - \pmb { \mu } _ { t } ) .
$$

Let $\begin{array} { r } { \mathbf { v } _ { t } ( \mathbf { x } ) = \dot { \pmb { \mu } } _ { t } + \frac { \dot { \sigma } _ { t } } { \sigma _ { \bot } } ( \mathbf { x } - \pmb { \mu } _ { t } ) } \end{array}$ . The flow (1) with $\mathbf { v } _ { t }$ then follows a probability path $p _ { t }$ (22) given by the pushforward (2). This is equivalent to the continuity equation (3), which can also be verified. On the one hand, On the one hand

$$
\begin{array} { r l } & { \partial _ { t } p _ { t } ( \mathbf { x } ) = - d \sigma _ { t } ^ { - d - 1 } \dot { \sigma } _ { t } p _ { 0 } ( \mathbf { z } ) + \sigma _ { t } ^ { - d } \nabla p _ { 0 } ( \mathbf { z } ) \cdot \left( - \dot { \mu } _ { t } \sigma _ { t } ^ { - 1 } - ( \mathbf { x } - \mu _ { t } ) \sigma _ { t } ^ { - 2 } \dot { \sigma } _ { t } \right) } \\ & { \qquad = - d \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } p _ { t } ( \mathbf { x } ) - \sigma _ { t } ^ { - d - 1 } \nabla p _ { 0 } ( \mathbf { z } ) \cdot ( \dot { \mu } _ { t } + \dot { \sigma } _ { t } \mathbf { z } ) , } \end{array}
$$

On the other hand,

$$
\begin{array} { l } { - \nabla \cdot ( { \mathbf v } _ { t } p _ { t } ) = - ( \nabla \cdot { \mathbf v } _ { t } ) p _ { t } ( { \mathbf x } ) - { \mathbf v } _ { t } \cdot \nabla p _ { t } ( { \mathbf x } ) } \\ { \displaystyle = - d \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } p _ { t } ( { \mathbf x } ) - ( \dot { \mu } _ { t } + \dot { \sigma } _ { t } { \mathbf z } ) \cdot ( \sigma _ { t } ^ { - d - 1 } \nabla p _ { 0 } ( { \mathbf z } ) ) } \end{array}
$$

Therefore, the two sides of (3) are equal. Hence, the proof is completed.

Theorem 3.2. The flow equation (1) with vector field $\mathbf { v } _ { t }$ (12) has the unique solution as an afine transformation:

$$
\phi _ { t } ( \mathbf { x } ) = \pmb { \mu } _ { t } + \sigma _ { t } \mathbf { x } .\tag{13}
$$

Proof of Theorem 3.2. Let $\mathbf { u } _ { t } = \phi _ { t } ( \mathbf { x } ) - \pmb { \mu } _ { t }$ . With the vector field $\mathbf { v } _ { t } \ \left( 1 2 \right)$ , we substitute it into (1) to get

$$
\dot { \mathbf { u } } _ { t } = \dot { \phi } _ { t } - \dot { \mu } _ { t } = \mathbf { v } _ { t } ( \phi _ { t } ) - \dot { \mu } _ { t } = \dot { \pmb { \mu } } _ { t } + \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } ( \phi _ { t } - \mu _ { t } ) - \dot { \mu } _ { t } = \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } \mathbf { u } _ { t } ,
$$

which already decouples across coordinates. Integrating it as a vector ODE yields

$$
\mathbf { u } _ { t } = \frac { \sigma _ { t } } { \sigma _ { 0 } } \mathbf { u } _ { 0 } .
$$

Since $\phi _ { 0 } ( \mathbf { x } ) = \mathbf { x } , \mu _ { 0 } = \mathbf { 0 }$ and $\sigma _ { 0 } = 1$ , we get $\mathbf { u } _ { 0 } = \mathbf { x }$ . Therefore ${ \mathbf { u } } _ { t } = \sigma _ { t } { \mathbf { x } }$ and $\phi _ { t } ( \mathbf { x } ) = \pmb { \mu } _ { t } + \sigma _ { t } \mathbf { x } .$ Hence the proof is completed. □

Proposition 3.1. The following OT interpolant path minimizes the flow energy (15):

$$
\mu _ { t } = ( 1 - t ) \mu _ { 0 } + t \mu _ { 1 } = t \mathbf { x } _ { 1 } , \quad \sigma _ { t } = ( 1 - t ) \sigma _ { 0 } + t \sigma _ { 1 } = 1 - t + t \sigma _ { \operatorname* { m i n } } .\tag{16}
$$

The minimal energy attained is $e _ { \mathrm { m i n } } = \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| ^ { 2 } + d | \sigma _ { 1 } - \sigma _ { 0 } | ^ { 2 }$

Proof of Proposition 3.1. We compute the energy (15) of the flow path

$$
\begin{array} { l } { { \displaystyle e ( { \bf v } ) = \sum _ { i = 1 } ^ { d } \int _ { { \mathbb R } ^ { d } } \int _ { 0 } ^ { 1 } p _ { t } ( { \bf x } ) v _ { t } ^ { 2 } ( x _ { i } ) d t d { \bf x } } } \\ { { \displaystyle ~ = \sum _ { i = 1 } ^ { d } \int _ { { \mathbb R } } \int _ { 0 } ^ { 1 } p _ { t } ( x _ { i } ) [ \dot { \mu } _ { t , i } + ( x _ { i } - \mu _ { t , i } ) \dot { \sigma } _ { t } / \sigma _ { t } ] ^ { 2 } d t d x _ { i } } } \\ { { \displaystyle ~ = \int _ { 0 } ^ { 1 } \sum _ { i = 1 } ^ { d } [ ( \dot { \mu } _ { t , i } ) ^ { 2 } + ( \dot { \sigma } _ { t } ) ^ { 2 } ] d t } } \\ { { \displaystyle ~ = \int _ { 0 } ^ { 1 } L ( \mu _ { t } , \sigma _ { t } , \dot { \mu } _ { t } , \dot { \sigma } _ { t } ) d t , ~ L ( \mu _ { t } , \sigma _ { t } , \dot { \mu } _ { t } , \dot { \sigma } _ { t } ) = \| \dot { \mu } _ { t } \| ^ { 2 } + d ( \dot { \sigma } _ { t } ) ^ { 2 } } } \end{array}
$$

By the calculus of variation, the Euler–Lagrange equations

$$
{ \frac { d } { d t } } { \frac { \partial L } { \partial { \dot { \mu } } _ { t } } } - { \frac { \partial L } { \partial \mu _ { t } } } = 0 , \quad { \frac { d } { d t } } { \frac { \partial L } { \partial { \dot { \sigma } } _ { t } } } - { \frac { \partial L } { \partial \sigma _ { t } } } = 0
$$

Solving these equations yields the desired geodesic path afine in t.

Substituting (16) into the above energy, we get the minimal value

$$
\operatorname* { m i n } e ( \mathbf { v } ) = \| { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } \| ^ { 2 } + d | { \sigma } _ { 1 } - { \sigma } _ { 0 } | ^ { 2 } .
$$

Lemma A.1. The exponential power distribution $\mathcal { P } _ { q } ( \pmb { \mu } , \sigma ^ { 2 } \mathbf { I } )$ has density

$$
\begin{array} { l } { { \displaystyle p ( { \bf x } ) = \frac { q \Gamma ( \frac { d } { 2 } ) } { 2 \Gamma ( \frac { d } { q } ) } 2 ^ { - \frac { d } { q } } \pi ^ { - \frac { d } { 2 } } \sigma ^ { - d } \exp \left\{ - \frac { r ^ { q / 2 } } { 2 } \right\} , } } \\ { { \displaystyle r ( { \bf x } ) = \frac { \| { \bf x } - { \bf \nabla } \mu \| ^ { 2 } } { \sigma ^ { 2 } } } . } \end{array}
$$

Its Fisher information matrix and Fisher-Rao metric are

$$
g ( \pmb { \mu } , \sigma ) = \frac { 1 } { \sigma ^ { 2 } } \left[ c _ { \mu } \mathbf { I } \quad \mathbf { 0 } ^ { \mathsf { T } } \right] , \quad a n d \quad d s ^ { 2 } = \frac { c _ { \mu } \| d \pmb { \mu } \| ^ { 2 } + c _ { \sigma } d \sigma ^ { 2 } } { \sigma ^ { 2 } } ,\tag{23}
$$

where $\begin{array} { r } { c _ { \mu } ( d , q ) = \frac { 2 ^ { - \frac { 2 } { q } } } { d } \frac { \Gamma ( \frac { d - 2 } { q } + 2 ) } { \Gamma ( \frac { d } { q } ) } q ^ { 2 } } \end{array}$ , and $c _ { \sigma } = q d$

Proof. By property 2 of Proposition 2.1, $S = r ^ { \frac { q } { 2 } } = ( \| \mathbf { x } - { \pmb \mu } \| / \sigma ) ^ { q } \sim \Gamma ( \alpha = { \textstyle { \frac { d } { q } } } , \beta = { \textstyle { \frac { 1 } { 2 } } } )$ [13, 19]. The log-density and their derivatives are

$$
l ( \pmb { \mu } , \sigma ) = - d \log \sigma - \frac { 1 } { 2 } r ^ { \frac { q } { 2 } } + \mathrm { c o n s t . }
$$

$$
\frac { \partial l } { \partial \pmb { \mu } } ( \pmb { \mu } , \sigma ) = \frac { q } { 2 \sigma } S ^ { 1 - \frac { 1 } { q } } \mathbf { u } , \quad \mathbf { u } = \frac { \mathbf { x } - \pmb { \mu } } { \| \mathbf { x } - \pmb { \mu } \| } \sim \operatorname { u n i f } ( S ^ { d - 1 } )
$$

$$
\frac { \partial l } { \partial \sigma } ( \pmb { \mu } , \sigma ) = \frac { 1 } { \sigma } \left[ - d + \frac { q } { 2 } S \right]
$$

$$
\frac { \partial ^ { 2 } l } { \partial \pmb { \mu } \partial \sigma } = - \frac { q ^ { 2 } } { 2 \sigma ^ { 2 } } S ^ { 1 - \frac { 1 } { q } } \mathbf { u }
$$

$$
\frac { \partial ^ { 2 } l } { \partial \sigma ^ { 2 } } ( { \pmb \mu } , \sigma ) = \frac { 1 } { \sigma ^ { 2 } } \left[ d - \frac { q } { 2 } ( 1 + q ) { \cal S } \right]
$$

By the moments of gamma distribution $\begin{array} { r } { \mathbb { E } [ S ^ { \alpha } ] = 2 ^ { \alpha } \frac { \Gamma ( \frac { d } { q } + \alpha ) } { \Gamma ( \frac { d } { q } ) } } \end{array}$ , we have the following Fisher information matrix I:

$$
{ \mathcal { T } } _ { \mu \mu } = \mathbb { E } \left[ { \frac { \partial l } { \partial \mu } } { \frac { \partial l } { \partial \mu } } ^ { \mathsf { T } } \right] = { \frac { c _ { \mu } } { \sigma ^ { 2 } } } { \mathbf { I } } , \quad c _ { \mu } ( d , q ) = { \frac { 2 ^ { - { \frac { 2 } { q } } } } { d } } { \frac { \Gamma ( { \frac { d - 2 } { q } } + 2 ) } { \Gamma ( { \frac { d } { q } } ) } } q ^ { 2 }
$$

$$
{ \mathcal { T } } _ { \mu \sigma } = - \mathbb { E } \left[ { \frac { \partial ^ { 2 } l } { \partial \mu \partial \sigma } } \right] = { \bf 0 } , \quad { \mathcal { T } } _ { \sigma \sigma } = \mathbb { E } \left[ \left( { \frac { \partial l } { \partial \sigma } } \right) ^ { 2 } \right] = - \mathbb { E } \left[ { \frac { \partial ^ { 2 } l } { \partial \sigma ^ { 2 } } } \right] = { \frac { c _ { \sigma } } { \sigma ^ { 2 } } } , ~ c _ { \sigma } = q d
$$

Therefore, the Fisher metric as written below is a hyperbolic metric

$$
g ( \pmb { \mu } , \sigma ) = \frac { 1 } { \sigma ^ { 2 } } \left[ \begin{array} { l l } { c _ { \mu } \mathbf { I } } & { \mathbf { 0 } ^ { \mathsf { T } } } \\ { \mathbf { 0 } } & { c _ { \sigma } } \end{array} \right] , \quad o r \quad d s ^ { 2 } = \frac { c _ { \mu } \| d \pmb { \mu } \| ^ { 2 } + c _ { \sigma } d \sigma ^ { 2 } } { \sigma ^ { 2 } } .
$$

Theorem 4.1. The following geodesic path minimizes the variational energy (18):

$$
\mu _ { t } = \mu _ { 0 } + x ( t ) \mathbf { e } , \quad x ( t ) = \mu _ { c } + R \cos \theta ( t ) , \quad R = \sqrt { \mu _ { c } ^ { 2 } + \lambda ^ { 2 } \sigma _ { 0 } ^ { 2 } } ;\tag{19a}
$$

$$
\sigma _ { t } = \frac { R } { \lambda } \sin \theta ( t ) , \quad \theta ( t ) = 2 \arctan \left( e ^ { L t } \tan \frac { \theta _ { 0 } } { 2 } \right) , \quad L = \log \frac { \tan ( \theta _ { 1 } / 2 ) } { \tan ( \theta _ { 0 } / 2 ) } ,\tag{19b}
$$

where $\begin{array} { r } { \lambda = \sqrt { c _ { \sigma } / c _ { \mu } } , { \bf e } = ( { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } ) / \| { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } \| _ { 2 } , { \mu } _ { c } = \frac { \| { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } \| ^ { 2 } + \lambda ^ { 2 } ( \sigma _ { 1 } ^ { 2 } - \sigma _ { 0 } ^ { 2 } ) } { 2 \| { \pmb { \mu } } _ { 1 } - { \pmb { \mu } } _ { 0 } \| } } \end{array}$ , and $\theta _ { 0 } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 0 } , - \mu _ { c } )$ $\theta _ { 1 } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 1 } , \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| - \mu _ { c } )$ . The minimal energy attained is $e _ { \mathrm { m i n } } = c _ { \sigma } L ^ { 2 }$ , where $\sqrt { c _ { \sigma } } | L |$ is the Fisher-Rao distance between $\mathcal { P } _ { q } ( \mu _ { 0 } , \sigma _ { 0 } ^ { 2 } \mathbf { I } )$ and $\mathscr { P } _ { q } ( \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } \mathbf { I } )$ , for which $| L |$ can be rewritten in the original parameters as

$$
| L | = \operatorname { a r c o s h } \left( 1 + { \frac { c _ { \mu } \| \mu _ { 1 } - \mu _ { 0 } \| ^ { 2 } + c _ { \sigma } | \sigma _ { 1 } - \sigma _ { 0 } | ^ { 2 } } { 2 c _ { \sigma } \sigma _ { 0 } \sigma _ { 1 } } } \right) .\tag{20}
$$

Remark 4. When $q = 2 , c _ { \mu } \equiv 1$ and $c _ { \sigma } = 2 d$ . The above Fisher-Rao formula yields the well-known geodesic distance between two Gaussians $\mathcal { N } ( \mu _ { 0 } , \sigma _ { 0 } ^ { 2 } \mathbf { I } )$ and $\mathcal { N } ( \mu _ { 1 } , \sigma _ { 1 } ^ { 2 } \mathbf { I } )$ :

$$
d _ { F R } = \sqrt { 2 d } \operatorname { a r c o s h } \left( 1 + \frac { \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| ^ { 2 } + 2 d | \sigma _ { 1 } - \sigma _ { 0 } | ^ { 2 } } { 4 d \sigma _ { 0 } \sigma _ { 1 } } \right) .
$$

Proof of Theorem $4 . 1 .$ The geodesic equation under the hyperbolic metric $\begin{array} { r } { d s ^ { 2 } = \frac { c _ { \mu } \| d \pmb { \mu } \| ^ { 2 } + c _ { \sigma } d \sigma ^ { 2 } } { \sigma ^ { 2 } } } \end{array}$ (23) can be solved in either semi-circle parametrization or hyperbolic parametrization, which are detailed as follows.

Poincar´e semi-circle parametrization The metric (23) can be rescaled to a standard Poincar´e metric which leads to a semi-circle in the Poincar´e upper half-plane. Denote $\lambda ~ = ~ { \sqrt { c _ { \sigma } / c _ { \mu } } }$ and $\mathbf { e } = ( \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } ) / \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| _ { 2 }$ . Use the scalar projection $x ( t ) = \langle { \pmb \mu } ( t ) - { \pmb \mu } _ { 0 } , { \bf e } \rangle$ and rescale $y ( t ) = \lambda \sigma ( t )$ The metric (23) reduces to

$$
d s ^ { 2 } = \frac { c _ { \sigma } } { y ^ { 2 } } ( d x ^ { 2 } + d y ^ { 2 } ) .
$$

By calculus of variation $\begin{array} { r } { ( L ( x , y , \dot { x } , \dot { y } ) = \frac { c _ { \sigma } ( \dot { x } ^ { 2 } + \dot { y } ^ { 2 } ) } { y ^ { 2 } } ) } \end{array}$ , we can get

$$
( x - \mu _ { c } ) ^ { 2 } + y ^ { 2 } = R ^ { 2 } , \quad \mu _ { c } = \frac { \| \mu _ { 1 } - \mu _ { 0 } \| ^ { 2 } + \lambda ^ { 2 } ( \sigma _ { 1 } ^ { 2 } - \sigma _ { 0 } ^ { 2 } ) } { 2 \| \mu _ { 1 } - \mu _ { 0 } \| } , \quad R = \sqrt { \mu _ { c } ^ { 2 } + \lambda ^ { 2 } \sigma _ { 0 } ^ { 2 } } .
$$

where $\mu _ { c }$ and R are determined by the boundary conditions $( x _ { 0 } , y _ { 0 } ) \ = \ ( 0 , \lambda \sigma _ { 0 } )$ and $( x _ { 1 } , y _ { 1 } ) =$ $( \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| , \lambda \sigma _ { 1 } )$ . Then the semi-circle can be parametrized by angle $\theta ( t ) \in ( 0 , \pi ) \colon x ( t ) = \mu _ { c } +$ R cos $\theta ( t ) , y ( t ) = R$ sin $\theta ( t )$ with two end points $\theta ( 0 ) = \theta _ { 0 }$ and $\theta ( 1 ) = \theta _ { 1 }$ corresponding to $( x _ { 0 } , y _ { 0 } ) =$ $( 0 , \lambda \sigma _ { 0 } )$ and $( x _ { 1 } , y _ { 1 } ) = ( \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| , \lambda \sigma _ { 1 } )$ respectively:

$$
\theta _ { 0 } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 0 } , - \mu _ { c } ) , \quad \theta _ { 1 } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 1 } , \| \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } \| - \mu _ { c } ) ,
$$

where atan $\operatorname { \ i } 2 ( y , x ) = \arg ( x + i y ) \in ( - \pi , \pi )$

Note the arc-length along the Poincar´e semi-circle has constant speed L: $\begin{array} { r } { d s = \frac { \sqrt { d x ^ { 2 } + d y ^ { 2 } } } { y } = } \end{array}$ ${ \frac { R | d \theta | } { R \sin \theta } } = L d t$ . Therefore, we have

$$
\theta ( t ) = 2 \arctan \left( e ^ { L t } \tan \frac { \theta _ { 0 } } { 2 } \right) , \quad L = \log \frac { \tan ( \theta _ { 1 } / 2 ) } { \tan ( \theta _ { 0 } / 2 ) } .
$$

Therefore, the final geodesic equation in the semi-circle parameterization of Poincar´e becomes

$$
\begin{array} { l } { { \displaystyle { \pmb \mu } ( t ) = { \pmb \mu } _ { 0 } + x ( t ) { \bf e } , \quad { \displaystyle { \boldsymbol x } ( t ) = { \boldsymbol \mu } _ { c } + R \cos \theta ( t ) } ; } } \\ { { \displaystyle { \boldsymbol \sigma } ( t ) = \frac { R } { \lambda } \sin \theta ( t ) . } } \end{array}
$$

Substituting the geodesic solution back to the energy equation, we have

$$
e ( { \bf v } ) = \int _ { \gamma } d s ^ { 2 } = c _ { \sigma } \int _ { 0 } ^ { 1 } \frac { \dot { x } ^ { 2 } + \dot { y } ^ { 2 } } { y ^ { 2 } } d t = c _ { \sigma } \int _ { 0 } ^ { 1 } \frac { \dot { \theta } ^ { 2 } ( t ) } { \sin ^ { 2 } \theta ( t ) } d t = c _ { \sigma } L ^ { 2 } ,
$$

where $| L |$ is the Fisher-Rao distance between two distributions, which can be rewritten in the original parameters as

$$
| L | = \operatorname { a r c o s h } \left( 1 + { \frac { c _ { \mu } \| \mu _ { 1 } - \mu _ { 0 } \| ^ { 2 } + c _ { \sigma } | \sigma _ { 1 } - \sigma _ { 0 } | ^ { 2 } } { 2 c _ { \sigma } \sigma _ { 0 } \sigma _ { 1 } } } \right) .
$$

Hyperbolic parametrization With Einstein’s notation, $\begin{array} { r } { g _ { \mu _ { i } \mu _ { j } } = \frac { c _ { \mu } } { \sigma ^ { 2 } } \delta _ { i j } , g _ { \sigma \sigma } = \frac { c _ { \sigma } } { \sigma ^ { 2 } } } \end{array}$ , we compute the Christofel symbols $\Gamma _ { \mu \nu } ^ { \lambda } = { \textstyle { \frac { 1 } { 2 } } } g ^ { \lambda \rho } [ \partial _ { \mu } g _ { \nu \rho } + \partial _ { \nu } g _ { \mu \rho } - \partial _ { \rho } g _ { \mu \nu } ]$ with inverse metric $\begin{array} { r } { g ^ { \mu _ { i } \mu _ { j } } = \frac { \sigma ^ { 2 } } { c _ { \mu } } \delta _ { i j } , g ^ { \sigma \sigma } = \frac { \sigma ^ { 2 } } { c _ { \sigma } } } \end{array}$ The only nonzero partial derivatives come from $\begin{array} { r } { \partial _ { \sigma } g = - \frac { 2 } { \sigma } g , } \end{array}$ giving

$$
\Gamma _ { \mu _ { j } \sigma } ^ { \mu _ { i } } = - \frac { 1 } { \sigma } \delta _ { i j } , \quad \Gamma _ { \mu _ { i } \mu _ { j } } ^ { \sigma } = \frac { c _ { \mu } } { c _ { \sigma } \sigma } \delta _ { i j } , \quad \Gamma _ { \sigma \sigma } ^ { \sigma } = - \frac { 1 } { \sigma } .
$$

Therefore, the geodesic equations ${ \ddot { x } } ^ { \lambda } + \Gamma _ { \mu \nu } ^ { \lambda } { \dot { x } } ^ { \mu } { \dot { x } } ^ { \nu } = 0$ become

$$
\ddot { \mu } _ { i } - \frac { 2 } { \sigma } \dot { \mu } _ { i } \dot { \sigma } = 0 ,\tag{24a}
$$

$$
\ddot { \sigma } + \frac { c _ { \mu } } { c _ { \sigma } \sigma } \| \dot { \pmb { \mu } } \| ^ { 2 } - \frac { 1 } { \sigma } \dot { \sigma } ^ { 2 } = 0 .\tag{24b}
$$

Solving (24a) yields $\dot { \pmb { \mu } } = \mathbf { c } \sigma ^ { 2 }$ . Substituting it into (24b) gives $\begin{array} { r } { \ddot { \sigma } + \frac { c _ { \mu } } { c _ { \sigma } } \| \mathbf { c } \| ^ { 2 } \sigma ^ { 3 } - \frac { 1 } { \sigma } \dot { \sigma } ^ { 2 } = 0 } \end{array}$ . By changing the variable $\sigma = e ^ { \tau }$ , (24b) becomes a standard autonomous second-order ODE

$$
\ddot { \tau } + \frac { c _ { \mu } } { c _ { \sigma } } \| \mathbf { c } \| ^ { 2 } e ^ { 2 \tau } = 0 ,
$$

which implies the following conserved energy:

$$
\dot { \tau } ^ { 2 } + \frac { c _ { \mu } } { c _ { \sigma } } \| { \mathbf { c } } \| ^ { 2 } e ^ { 2 \tau } = \frac { E } { c _ { \sigma } } , \quad o r \quad \frac { c _ { \mu } \| { \mathbf { c } } \| ^ { 2 } \sigma ^ { 4 } + c _ { \sigma } \dot { \sigma } ^ { 2 } } { \sigma ^ { 2 } } = E .\tag{25}
$$

Denote $\omega = { \sqrt { E / c _ { \sigma } } } , \kappa = \| \mathbf { c } \| { \sqrt { c _ { \mu } / c _ { \sigma } } }$ . This energy equation (25) can be rewritten as a separable ODE $\dot { \sigma } ^ { 2 } = \omega ^ { 2 } \sigma ^ { 2 } \stackrel { \cdot } { - } \kappa ^ { 2 } \sigma ^ { 4 }$ which has the following solution

$$
\sigma ( t ) = \frac { \omega } { \kappa } \mathrm { s e c h } ( \omega t + \alpha ) ,
$$

where ω, κ and α can be determined by the boundary conditions. Using (24a), or equivalent $\dot { \pmb { \mu } } = \mathbf { c } \sigma ^ { 2 }$ one can solve

$$
\pmb { \mu } ( t ) = \pmb { \mu } _ { 0 } + \mathbf { c } \frac { \omega } { \kappa ^ { 2 } } ( \operatorname { t a n h } ( \omega t + \alpha ) - \operatorname { t a n h } ( \alpha ) ) ,
$$

where $\begin{array} { r } { \mathbf { c } = ( \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } ) \frac { \kappa ^ { 2 } } { \omega ( \operatorname { t a n h } ( \omega + \alpha ) - \operatorname { t a n h } ( \alpha ) ) } } \end{array}$ . We also have the conserved energy $E = c _ { \sigma } \omega ^ { 2 }$ along the final geodesic:

$$
\pmb { \mu } ( t ) = \pmb { \mu } _ { 0 } + ( \pmb { \mu } _ { 1 } - \pmb { \mu } _ { 0 } ) \frac { \operatorname { t a n h } ( \omega t + \alpha ) - \operatorname { t a n h } ( \alpha ) } { \operatorname { t a n h } ( \omega + \alpha ) - \operatorname { t a n h } ( \alpha ) } ;\tag{26a}
$$

$$
\sigma ( t ) = \frac { \omega } { \kappa } \mathrm { s e c h } ( \omega t + \alpha ) ,\tag{26b}
$$

where $\omega , \kappa , \alpha$ can be numerically solved from the following algebraic equations:

$$
\frac { \omega } { \kappa } \operatorname { s e c h } ( \alpha ) = \sigma _ { 0 } , \quad \frac { \omega } { \kappa } \operatorname { s e c h } ( \omega + \alpha ) = \sigma _ { 1 } , \quad \kappa = \| \mu _ { 1 } - \mu _ { 0 } \| \frac { \kappa ^ { 2 } \sqrt { c _ { \mu } / c _ { \sigma } } } { \omega ( \operatorname { t a n h } ( \omega + \alpha ) - \operatorname { t a n h } ( \alpha ) ) } .
$$

Remark 5. The Poincar´e semicircle and hyperbolic parametrizations are equivalent through the $f o l -$ lowing identities:

$$
\sin ( 2 \arctan e ^ { - s } ) = \operatorname { s e c h } ( s ) , \quad \cos ( 2 \arctan e ^ { - s } ) = \operatorname { t a n h } ( s ) , \quad s = \omega t + \alpha , \quad R = \frac { \omega } { \kappa } \lambda .
$$

However, the former has fully explicit constants, and yet the latter contains implicit constants that need to be resolved numerically.

In particular, we get the vector field that generates the probability path by (12):

$$
\begin{array} { l } { { { \bf { v } } _ { t } } ( { \bf { x } } ) = \dot { \mu } _ { t } + \frac { { \dot { \sigma } _ { t } } } { { \sigma _ { t } } } ( { \bf { x } } - { \mu _ { t } } ) = { \bf { c } } \sigma _ { t } ^ { 2 } - \omega \operatorname { t a n h } ( \omega t + \alpha ) [ { \bf { x } } - \mu _ { 0 } - { \bf { c } } \frac { \omega } { \kappa ^ { 2 } } ( \operatorname { t a n h } ( \omega t + \alpha ) - \operatorname { t a n h } ( \alpha ) ) ] } \\ { = \omega \left[ \frac { 1 - \operatorname { t a n h } ( \omega t + \alpha ) \operatorname { t a n h } ( \alpha ) } { \operatorname { t a n h } ( \omega + \alpha ) - \operatorname { t a n h } ( \alpha ) } ( \mu _ { 1 } - \mu _ { 0 } ) - \operatorname { t a n h } ( \omega t + \alpha ) ( { \bf { x } } - \mu _ { 0 } ) \right] . } \end{array}
$$

Letting $\pmb { \mu } _ { 0 } = \mathbf { 0 } , \pmb { \mu } _ { 1 } = \mathbf { x } _ { 1 } , \sigma _ { 0 } = 1 , \sigma _ { 1 } = \sigma _ { \operatorname* { m i n } }$ , we have the conditional vector field

$$
\mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) = \omega \left[ \frac { 1 - \operatorname { t a n h } ( \omega t + \alpha ) \operatorname { t a n h } ( \alpha ) } { \operatorname { t a n h } ( \omega + \alpha ) - \operatorname { t a n h } ( \alpha ) } \mathbf { x } _ { 1 } - \operatorname { t a n h } ( \omega t + \alpha ) \mathbf { x } \right] ,
$$

and the conditional flow

$$
\psi _ { t } ( \mathbf { x } ) = \mathbf { x } _ { 1 } \frac { \operatorname { t a n h } ( \omega t + \alpha ) - \operatorname { t a n h } ( \alpha ) } { \operatorname { t a n h } ( \omega + \alpha ) - \operatorname { t a n h } ( \alpha ) } + \frac { \omega } { \kappa } \operatorname { s e c h } ( \omega t + \alpha ) \mathbf { x } .
$$

Let $p _ { t } ( \cdot | \mathbf { x } _ { 1 } ) \sim \mathcal { P } _ { q } ( \mu _ { t } ( \mathbf { x } _ { 1 } ) , \sigma _ { t } ^ { 2 } ( \mathbf { x } _ { 1 } ) \mathbf { I } )$ with $( \mu _ { t } , \sigma _ { t } )$ as in Theorem 4.1. We have the marginal probability path $\begin{array} { r } { p _ { t } ( \mathbf { x } ) = \int p _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) q ( \mathbf { x } _ { 1 } ) d \mathbf { x } _ { 1 } } \end{array}$ , the marginal vector field and the FM loss:

$$
\mathbf { u } _ { t } ( \mathbf { x } ) = \mathbb { E } [ \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) | \mathbf { x } _ { t } = \mathbf { x } ] = \frac { \int \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) p _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) q ( \mathbf { x } _ { 1 } ) d \mathbf { x } _ { 1 } } { p _ { t } ( \mathbf { x } ) } , \ \mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { t , p _ { t } ( \mathbf { x } ) } \| \mathbf { v } _ { t } ( \mathbf { x } ; \theta ) - \mathbf { u } _ { t } ( \mathbf { x } ) \| _ { 2 } ^ { 2 } .
$$

The following theorem A.1 states that the CFM loss (14) along the geodesic in Theorem 4.1 decomposes into the FM loss and the variance of conditional vector field and is bounded from below.

Theorem A.1. For every parameter θ in the neural network $\mathbf { v } _ { t } ( \cdot ; \theta )$ , we have

$$
\begin{array} { r l } & { \mathcal { L } _ { C F M } ( \theta ) = \mathcal { L } _ { F M } ( \theta ) + C , \quad C : = \mathbb { E } _ { t , p _ { t } ( \mathbf { x } ) } \mathrm { V a r } _ { \mathbf { x } _ { 1 } | \mathbf { x } _ { t } = \mathbf { x } } [ \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) ] , } \\ & { \qquad C = \underset { \theta } { \operatorname* { i n f } } \mathcal { L } _ { C F M } ( \theta ) \leq \sigma _ { \operatorname* { m a x } } ^ { 2 } \kappa _ { q , d } \mathbb { E } _ { q ( \mathbf { x } _ { 1 } ) } L ^ { 2 } ( \mathcal { P } _ { q } ( \mathbf { 0 } , \mathbf { I } ) , \mathcal { P } _ { q } ( \mathbf { x } _ { 1 } , \sigma _ { \operatorname* { m i n } } ^ { 2 } \mathbf { I } ) ) , } \end{array}
$$

where C does not depend on the neural network $\mathbf { v } _ { t } ( \cdot ; \theta ) , \sigma _ { \operatorname* { m a x } } , \kappa _ { q , d }$ are constants depending on the geodesic, and $L ^ { 2 } ( \mathcal { P } _ { q } ( \mathbf { 0 } , \mathbf { I } ) , \mathcal { P } _ { q } ( \mathbf { x } _ { 1 } , \sigma _ { \operatorname* { m i n } } ^ { 2 } \mathbf { I } ) )$ is the scaled square Fisher-Rao distance between $\mathcal { P } _ { q } ( \mathbf { 0 } , \mathbf { I } )$ and $\mathcal { P } _ { q } ( \mathbf { x } _ { 1 } , \sigma _ { \mathrm { m i n } } ^ { 2 } \mathbf { I } )$

Proof. Denote $\mathbf { x } _ { t } : = \psi _ { t } ( \mathbf { x } _ { 0 } )$ as the conditional flow (6). With $q ( \mathbf { x } _ { 1 } ) p _ { t } ( \mathbf { x } _ { t } | \mathbf { x } _ { 1 } ) = p _ { t } ( \mathbf { x } _ { t } ) p ( \mathbf { x } _ { 1 } | \mathbf { x } _ { t } )$ , the CFM loss (14) can be rewritten

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { C F M } } ( \boldsymbol \theta ) = \mathbb { E } _ { t } \mathbb { E } _ { g ( \mathbf { x } _ { t } ) } \mathbb { E } _ { p _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 1 } ) } \| \mathbf { v } _ { t } ( \mathbf { x } _ { t } ; \boldsymbol \theta ) - \mathbf { u } _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 1 } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad = \mathbb { E } _ { t } \mathbb { E } _ { p _ { t } ( \mathbf { x } _ { t } ) } \mathbb { E } _ { p ( \mathbf { x } _ { 1 } \mid \mathbf { x } _ { t } ) } \| \mathbf { v } _ { t } ( \mathbf { x } _ { t } ; \boldsymbol \theta ) - \mathbf { u } _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 1 } ) \| _ { 2 } ^ { 2 } } \\ & { \qquad = \underbrace { \mathbb { E } _ { t } \mathbb { E } _ { p _ { t } ( \mathbf { x } _ { t } ) } \| \mathbf { v } _ { t } ( \mathbf { x } _ { t } ; \boldsymbol \theta ) - \mathbf { u } _ { t } ( \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } } _ { \mathcal { L } _ { \mathrm { F M } } ( \boldsymbol \theta ) } + \underbrace { \mathbb { E } _ { t } \mathbb { E } _ { p _ { t } ( \mathbf { x } _ { t } ) } \mathbb { E } _ { p ( \mathbf { x } _ { 1 } \mid \mathbf { x } _ { t } ) } \| \mathbf { u } _ { t } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 1 } ) - \mathbf { u } _ { t } ( \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } } _ { C } . } \end{array}
$$

![](images/0b4841e1fa77b4673f89c1741bb5436d44a31a06a998febd6cb5be983e067536.jpg)

![](images/5527a9ec66d5b4332ef3fd9aa4c4797455d7537a5f321c7e27ad0dc14e341187.jpg)

![](images/1d7f9ea3ac24d8a7b1d6eb69e09fcfc17e232b912934eca3b3b684a1b9e1727a.jpg)

![](images/6f81245a857438e5b85525be796e34d03e07f550e39d87471c67a24e44246cf2.jpg)  
Figure A.1: Euclidean vs Fisher-Rao distances, (probabilistic) geodesic probability paths, flow paths for $q = 1 . 0$ and $q = 2 . 0$ (from left to right) on the manifold of exponential power distributions.

where $C = \mathbb { E } _ { t , p _ { t } ( \mathbf { x } ) } \mathrm { V a r } _ { \mathbf { x } _ { 1 } | \mathbf { x } _ { t } = \mathbf { x } } [ \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) ]$ based on the definition of marginal vector field ${ \bf u } _ { t } ( { \bf x } )$ . Since ${ \mathcal { L } } _ { \mathrm { F M } } ( \theta ) \geq 0$ , we get $\mathcal { L } _ { \mathrm { C F M } } ( \theta ) \geq C$ , with equality achieved if $\left\| \mathbf { v } _ { t } ( \mathbf { x } _ { t } ; \boldsymbol { \theta } ) - \mathbf { u } _ { t } ( \mathbf { x } _ { t } ) \right\| = 0$ for $p _ { t } \mathrm { - a . e } \left( \mathbf { x } , t \right)$ Therefore, C is the lower bound of the CFM loss (14), regardless of the network $\mathbf { v } _ { t } ( \cdot ; \theta )$

To get the upper bound of C for the geodesic path, we notice that with conditional vector field (7)

$$
\begin{array} { r l } & { C \leq \mathbb { E } _ { t } \mathbb { E } _ { q ( \mathbf { x } _ { 1 } ) } \mathbb { E } _ { p _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) } \| \mathbf { u } _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) \| ^ { 2 } } \\ & { \quad = \mathbb { E } _ { t } \mathbb { E } _ { q ( \mathbf { x } _ { 1 } ) } [ \| \dot { \mu } _ { t } ( \mathbf { x } _ { 1 } ) \| ^ { 2 } + ( \dot { \sigma } _ { t } ( \mathbf { x } _ { 1 } ) / \sigma _ { t } ( \mathbf { x } _ { 1 } ) ) ^ { 2 } \mathbb { E } _ { p _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } ) } \| \mathbf { x } - \mu _ { t } ( \mathbf { x } _ { 1 } ) \| ^ { 2 } ] } \\ & { \quad ^ { p r o p 2 . 1 } \mathbb { E } _ { q ( \mathbf { x } _ { 1 } ) } \int _ { 0 } ^ { 1 } [ \| \dot { \mu } _ { t } ( \mathbf { x } _ { 1 } ) \| ^ { 2 } + s _ { q , d } d \dot { \sigma } _ { t } ^ { 2 } ( \mathbf { x } _ { 1 } ) ] d t } \\ & { \quad \leq \mathbb { E } _ { q ( \mathbf { x } _ { 1 } ) } \int _ { 0 } ^ { 1 } \sigma _ { t } ^ { 2 } \operatorname* { m a x } \left\{ \cfrac { 1 } { c _ { \mu } } , \frac { s _ { q , d } d } { c _ { \sigma } } \right\} \frac { c _ { \mu } \| \dot { \mu } _ { t } \| ^ { 2 } + c _ { \sigma } \dot { \sigma } _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } d t } \\ &  \quad \leq \sigma _ { \operatorname* { m a x } } ^ { 2 } \kappa _ { q , d } \mathbb { E } _ { q ( \mathbf { x } _ { 1 } ) } L ^ { 2 } ( \mathcal { P } _ { q } ( 0 , \mathbf { I } ) , \mathcal { P } _ { q } ( \mathbf { x } _ { 1 } , \sigma _ { \operatorname* { m i n } } ^  \end{array}
$$

where $\begin{array} { r } { \sigma _ { \operatorname* { m a x } } : = \operatorname* { s u p } _ { t \in [ 0 , 1 ] , \mathbf { x } _ { 1 } \in \mathrm { s u p p } ( q ( \mathbf { x } _ { 1 } ) ) } \sigma _ { t } ^ { 2 } ( \mathbf { x } _ { 1 } ) } \end{array}$ , and $\begin{array} { r } { \kappa _ { q , d } = \operatorname* { m a x } \left\{ \frac { 1 } { c _ { \mu } } , \frac { s _ { q , d } d } { c _ { \sigma } } \right\} } \end{array}$

Remark 6. Note that the CFM loss bound C depends only on the probability path $p _ { t } ( \cdot | \mathbf { x } )$ and the data distribution $q ( \cdot )$ , not the neural network $\mathbf { v } ( \cdot , \theta )$

Remark 7. With the OT-interpolant path (9), the CFM loss bound collapses to

$$
C = \mathbb { E } _ { t , p _ { t } ( \mathbf { x } ) } \mathrm { V a r } _ { \mathbf { x } _ { 1 } | \mathbf { x } _ { t } } [ \mathbf { x } _ { 1 } ] \left( 1 + \frac { ( 1 - \sigma _ { \operatorname* { m i n } } ) t } { \sigma _ { t } } \right) ^ { 2 } .
$$

This quantity peaks around $\begin{array} { r } { t \approx \frac { 1 } { 2 } } \end{array}$ , where the posterior $p ( \mathbf { x } _ { 1 } | \mathbf { x } _ { t } )$ is most uncertain. Correspondingly, an “energy hump” may appear, as illustrated in Figure 2.

Remark 8. The proof suggests a more natural choice of loss based on the energy (18), i.e. replacing the $L _ { 2 }$ norm with the Fisher-Rao norm in (14). That would yield a tighter bound $f o r C$

## B More Numerical Results

## B.1 Setup for Numerical Experiments

We follow the framework of [20, CFM]. For all simulations, we use a 5-layer dense neural network with sinusoidal (Fourier) time encoding and 512 hidden dimensions and train it with the ADAM optimizer with learning rate $1 e - 4$ . We set $\sigma _ { \operatorname* { m i n } } = 1 e \mathrm { ~ - ~ } 3$ for $\mathrm { P G / H B }$ and $1 e - 3$ for other baselines. In the sampling/inference stage, we adopt the ‘midpoint’ method to solve the flow ODE (1) with step size 0.05.

For image applications (MNIST), we use a classic UNet for MNIST (64 base channels, 2 residual blocks) with sinusoidal time embedding trained by AdamW at a learning rate $2 e - 4$ . The exponential moving average (ema) is also adopted with decay rate 0.999. In sampling, the same ‘midpoint’ method is used for solving ODE with 100 discretized steps. FID scores are evaluated based on 50k samples.

![](images/38221913e48054acf71570df261183e57c1c885166cf3fcef3f545ed119571be.jpg)  
Figure B.1: Sample paths in the Swiss roll (left) and moons (right) examples: VP, OT, sino, PG-2, and PG-1 (from top to bottom).

We refer to Wasserstein-2 distance $\begin{array} { r } { ( W _ { 2 } ^ { 2 } ( P , Q ) \ = \ \operatorname* { i n f } _ { \pi \in \Pi ( P , Q ) } \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } \| x - y \| _ { 2 } ^ { 2 } d \pi ( x , y ) ) } \end{array}$ , energy distance $( \mathrm { E D } ^ { 2 } ( P , Q ) = 2 \mathbb { E } \| X - Y \| - \mathbb { E } \| X - X ^ { \prime } \| - \mathbb { E } \| Y - Y ^ { \prime } \| \mathrm { ~ f o r ~ } X , X ^ { \prime } \overset { i i d } { \sim } P$ and $Y , Y ^ { \prime } \stackrel { i i d } { \sim } Q$ independent), and maximum mean discrepancy $( \mathrm { M M D } ^ { 2 } ( P , Q ; k ) = \mathbb { E } _ { X , X ^ { \prime } } k ( X , X ^ { \prime } ) + \mathbb { E } _ { Y , Y ^ { \prime } } k ( Y , Y ^ { \prime } ) -$ $2 \mathbb { E } _ { X , Y } k ( X , Y ) )$ with radial basis function (rbf) kernel k to measure the discrepancy between synthetic sample distribution and real data distribution. The training time per epoch, ‘time/epoch’, is reported in Tables 1, B.1, B.4, and B.5. The sampler for the exponential power distribution is based on Proposition 2.1.

## B.2 2d Benchmarks

Table B.1: Swiss roll dataset comparison in $W _ { 2 }$ , ED, and MMD (mean ± std over 10 seeds).
<table><tr><td>DIST</td><td>PATH</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED↓</td><td> $\overline { { \mathrm { M M D } \downarrow } }$ </td><td> $\overline { { { \mathrm { t i m e / e p o c h } } } }$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td>VP</td><td> $\overline { { 1 . 4 9 e - 1 \pm 3 . 5 e - 2 } }$ </td><td> $3 . 6 9 e - 2 \pm 1 . 5 e - 2$ </td><td> $3 . 3 3 e - 4 \pm 2 . 2 e - 4$ </td><td> $5 . 8 4 e - 2 \pm 6 . 1 e - 4$ </td></tr><tr><td>q=2.0</td><td>VP</td><td> $1 . 4 2 e { - } 1 \pm 4 . 5 e { - } 2$ </td><td> $2 . 9 3 e - 2 \pm 2 . 3 e - 2$ </td><td> $3 . 0 6 e { - } 4 \pm 3 . 6 e { - } 4$ </td><td> $6 . 2 4 e { - 2 } \pm 5 . 3 e { - 4 }$ </td></tr><tr><td>q=1.0</td><td>OT</td><td> $1 . 1 4 e - 1 \pm 2 . 3 e - 2$ </td><td> $1 . 5 6 e { - 2 } \pm 1 . 1 e { - 2 }$ </td><td> $7 . 5 0 e { - } 5 \pm 9 . 1 e { - } 5$ </td><td> $5 . 6 6 e - 2 \pm 5 . 4 e - 4$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td>OT</td><td> $1 . 1 2 e - 1 \pm 3 . 0 e - 2$ </td><td> $1 . 6 9 e { - 2 } \pm 1 . 6 e { - 2 }$ </td><td> $9 . 0 3 e { - } 5 \pm 1 . 7 e { - } 4$ </td><td> $5 . 4 0 e { - 2 } \pm 6 . 2 e { - 4 }$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td>sino</td><td> $1 . 3 3 e { - } 1 \pm 2 . 5 e { - } 2$ </td><td> $2 . 5 3 e { - } 2 \pm 1 . 3 e { - } 2$ </td><td> $1 . 8 7 e { - } 4 \pm 1 . 2 e { - } 4$ </td><td> $5 . 7 4 e - 2 \pm 3 . 7 e - 4$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td>sino</td><td> $1 . 1 7 e { - } 1 \pm 1 . 4 e { - } 2$ </td><td> $1 . 2 4 e { - 2 } \pm 8 . 3 e { - 3 }$ </td><td> $6 . 0 5 e { - 5 } \pm 5 . 0 e { - 5 }$ </td><td> $5 . 4 6 e - 2 \pm 4 . 2 e - 4$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td>PG</td><td> $1 . 0 5 e { - } 1 \pm 2 . 4 e { - } 2$ </td><td> $9 . 5 6 e - 3 \pm 7 . 8 e - 3$ </td><td> $5 . 2 7 e { - } 5 \pm 8 . 4 e { - } 5$ </td><td> $1 . 0 5 e - 1 \pm 3 . 2 e - 4$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td>PG</td><td> $1 . 0 8 e { - } 1 \pm 2 . 0 e { - } 2$ </td><td> $8 . 6 6 e { - 3 } \pm 8 . 0 e { - 3 }$ </td><td> $3 . 5 5 e { - } 5 \pm 6 . 3 e { - } 5$ </td><td> $1 . 1 0 e { - } 1 \pm 9 . 9 e { - } 3$ </td></tr></table>

Table B.2: Comparison on Moons and Checkerboard datasets in $W _ { 2 }$ , ED, and MMD $( \mathrm { m e a n } \pm \mathrm { s t d }$ over 10 seeds).
<table><tr><td></td><td></td><td colspan="3">Moons</td><td colspan="3">Checkerboard</td></tr><tr><td>DIST</td><td>PATH</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED ↓</td><td>MMD↓</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED ↓</td><td>MMD ↓</td></tr><tr><td>q=1.0</td><td>VP</td><td> $\overline { { 1 . 3 3 e - 1 \pm 2 . 4 e - 2 } }$ </td><td> $\overline { { 2 . 5 6 e { - 2 } \pm 1 . 5 e { - 2 } } }$ </td><td> $\overline { { 2 . 1 4 e { - } 4 } \pm 1 . 8 e { - } 4 }$ </td><td> $\overline { { 3 . 5 8 e { - } 1 \pm 9 . 6 e { - } 2 } }$ </td><td> $\overline { { 4 . 4 5 e { - 2 } \pm 2 . 9 e { - 2 } } }$ </td><td> $\overline { { 3 . 0 1 e - 4 \pm 3 . 0 e - 4 } }$ </td></tr><tr><td>q=2.0</td><td>VP</td><td> $1 . 0 6 e { - } 1 \pm 1 . 7 e { - } 2$ </td><td> $1 . 0 0 e { - 2 } \pm 1 . 0 e { - 2 }$ </td><td> $5 . 6 9 e { - } 5 \pm 6 . 5 e { - } 5$ </td><td> $3 . 5 3 e { - } 1 \pm 6 . 2 e { - } 2$ </td><td> $4 . 2 9 e { - 2 } \pm 1 . 6 e { - 2 }$ </td><td> $2 . 0 5 e { - } 4 \pm 1 . 2 e { - } 4$ </td></tr><tr><td>q=1.0</td><td>OT</td><td> $1 . 1 7 e { - } 1 \pm 2 . 1 e { - } 2$ </td><td> $1 . 9 4 e { - 2 } \pm 1 . 2 e { - 2 }$ </td><td> $1 . 0 8 e { - } 4 \pm 1 . 2 e { - } 4$ </td><td> $3 . 1 5 e \mathrm { - } 1 \pm 5 . 3 e \mathrm { - } 2$ </td><td> $3 . 2 3 e { - 2 } \pm 1 . 7 e { - 2 }$ </td><td> $1 . 6 3 e { - } 4 \pm 1 . 1 e { - } 4$ </td></tr><tr><td>q=2.0</td><td>OT</td><td> $9 . 8 3 e - 2 \pm 1 . 3 e - 2$ </td><td> $9 . 3 9 e - 3 \pm 8 . 6 e - 3$ </td><td> $5 . 5 6 e { - } 5 \pm 6 . 0 e { - } 5$ </td><td> $2 . 9 6 e - 1 \pm 4 . 1 e - 2$ </td><td> $2 . 2 1 e { - } 2 \pm 1 . 6 e { - } 2$ </td><td> $9 . 3 6 e { - } 5 \pm 7 . 4 e { - } 5$ </td></tr><tr><td>q=1.0</td><td>sino</td><td> $1 . 2 5 e { - } 1 \pm 1 . 8 e { - } 2$ </td><td> $2 . 2 6 e { - 2 } \pm 1 . 1 e { - 2 }$ </td><td> $1 . 5 9 e { - } 4 \pm 1 . 1 e { - } 4$ </td><td> $3 . 5 9 e - 1 \pm 9 . 1 e - 2$ </td><td> $4 . 7 6 e { - 2 } \pm 2 . 4 e { - 2 }$ </td><td> $3 . 3 8 e { - } 4 \pm 2 . 5 e { - } 4$ </td></tr><tr><td>q=2.0</td><td>sino</td><td> $1 . 0 6 e { - } 1 \pm 1 . 5 e { - } 2$ </td><td> $1 . 2 2 e - 2 \pm 1 . 0 e - 2$ </td><td> $6 . 2 5 e { - } 5 \pm 5 . 5 e { - } 5$ </td><td> $3 . 0 1 e { - } 1 \pm 4 . 5 e { - } 2$ </td><td> $2 . 3 2 e { - } 2 \pm 1 . 6 e { - } 2$ </td><td> $1 . 0 6 e { - } 4 \pm 7 . 0 e { - } 5$ </td></tr><tr><td>q=1.0</td><td>PG</td><td> $1 . 0 2 e { - } 1 \pm 1 . 3 e { - } 2$ </td><td> $9 . 2 6 e - 3 \pm 9 . 2 e - 3$ </td><td> $6 . 9 4 e { - } 5 \pm 8 . 4 e { - } 5$ </td><td> $3 . 1 7 e - 1 \pm 3 . 8 e - 2$ </td><td> $2 . 7 4 e { \mathrm { - } } 2 \pm 1 . 6 e { \mathrm { - } } 2$ </td><td> $1 . 2 4 e { - } 4 \pm 8 . 3 e { - } 5$ </td></tr><tr><td>q=2.0</td><td>PG</td><td> $1 . 0 0 e \mathrm { - } 1 \pm 1 . 4 e \mathrm { - } 2$ </td><td> $5 . 6 2 e { - 3 } \pm 8 . 0 e { - 3 }$ </td><td> $3 . 6 8 e { - } 5 \pm 7 . 3 e { - } 5$ </td><td> $3 . 0 5 e \mathrm { - } 1 \pm 3 . 2 e \mathrm { - } 2$ </td><td> $2 . 5 6 e { - 2 } \pm 6 . 8 e { - 3 }$ </td><td> $1 . 0 4 e { - } 4 \pm 3 . 0 e { - } 5$ </td></tr></table>

## B.3 Higher Dimensional Simulations

Next, we investigate the more challenging Neal’s funnel distribution [24]:

$$
f ( \mathbf { x } , \nu ) = \mathcal { N } _ { d } ( \mathbf { x } ; \mathbf { 0 } , e ^ { \nu / 2 } \mathbf { I } ) \mathcal { N } ( \nu ; 0 , 3 ) ,
$$

whose density features a cusp on one end and a fat tail on the other (refer to the right panel of Figure B.2, which also demonstrates PG methods’ faster feature learning), posing dificulties for generative models to sample data from it. We consider d = 10/100 and generate 10,000 samples to train these FM models.

Table B.3: PG q-sweep comparison across Swiss roll, moons, and checkerboard in $W _ { 2 }$ and ED (mean ± std over 10 seeds).
<table><tr><td></td><td colspan="2">Swiss Roll</td><td colspan="2">Moons</td><td colspan="2">Checkerboard</td></tr><tr><td>DIST</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED ↓</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td> $\overline { { \mathrm { E D ~ \downarrow ~ } } }$ </td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED↓</td></tr><tr><td> $\overline { { \mathbf { q } { = } 0 . 5 } }$ </td><td> $3 . 7 0 e - 1 \pm 2 . 1 e - 2$ </td><td> $6 . 7 6 e { - 2 \pm 5 . 8 e { - 3 } }$ </td><td> $\overline { { 3 . 1 8 e - 1 \pm 1 . 7 e - 2 } }$ </td><td> $\overline { { 6 . 1 4 e { - 2 } \pm 4 . 6 e { - 3 } } }$ </td><td> $\overline { { 3 . 9 2 e - 1 \pm 4 . 2 e - 2 } }$ </td><td> $3 . 5 6 e - 2 \pm 1 . 3 e - 2$ </td></tr><tr><td> $\mathrm { q } { = } 0 . 8$ </td><td> $1 . 1 1 e { - } 1 \pm 1 . 9 e { - } 2$ </td><td> $1 . 0 2 e { - 2 } \pm 8 . 8 e { - 3 }$ </td><td> $1 . 0 9 e { - } 1 \pm 1 . 6 e { - } 2$ </td><td> $1 . 4 1 e { - } 2 \pm 9 . 6 e { - } 3$ </td><td> $3 . 1 9 e - 1 \pm 4 . 1 e - 2$ </td><td> $2 . 8 0 e { - } 2 \pm 1 . 3 e { - } 2$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td> $1 . 0 5 e { - } 1 \pm 2 . 4 e { - } 2$ </td><td> $9 . 5 6 e - 3 \pm 7 . 8 e - 3$ </td><td> $1 . 0 2 e { - } 1 \pm 1 . 3 e { - } 2$ </td><td> $9 . 2 6 e { - 3 } \pm 9 . 2 e { - 3 }$ </td><td> $3 . 1 7 e - 1 \pm 3 . 8 e - 2$ </td><td> $2 . 7 4 e { - } 2 \pm 1 . 6 e { - } 2$ </td></tr><tr><td> $\mathrm { q } \mathrm { = } 1 . 5$ </td><td> $1 . 0 6 e { - } 1 \pm 1 . 4 e { - } 2$ </td><td> $9 . 1 3 e - 3 \pm 8 . 6 e - 3$ </td><td> $9 . 9 5 e - 2 \pm 1 . 2 e - 2$ </td><td> $9 . 9 3 e { - 3 } \pm 9 . 3 e { - 3 }$ </td><td> $3 . 1 9 e \mathrm { - } 1 \pm 5 . 4 e \mathrm { - } 2$ </td><td> $3 . 0 7 e { - } 2 \pm 1 . 8 e { - } 2$ </td></tr><tr><td> $\mathrm { q = 2 . 0 }$ </td><td> $1 . 0 8 e { - } 1 \pm 2 . 0 e { - } 2$ </td><td> $8 . 6 6 e - 3 \pm 8 . 0 e - 3$ </td><td> $1 . 0 0 e { - } 1 \pm 1 . 4 e { - } 2$ </td><td> $5 . 6 2 e - 3 \pm 8 . 0 e - 3$  _</td><td> $3 . 0 5 e { - } 1 \pm 3 . 2 e { - } 2$ </td><td> $2 . 5 6 e { - 2 } \pm 6 . 8 e { - 3 }$  _</td></tr><tr><td> $\mathrm { { q = } 2 . 5 }$ </td><td> $1 . 1 5 e { - } 1 \pm 1 . 8 e { - } 2$ </td><td> $1 . 2 9 e - 2 \pm 8 . 0 e - 3$ </td><td> $9 . 8 0 e { - 2 \pm 8 . 5 e { - 3 } }$ </td><td> $7 . 6 8 e { - 3 } \pm 6 . 9 e { - 3 }$ </td><td> $3 . 3 7 e - 1 \pm 4 . 5 e - 2$ </td><td> $3 . 6 6 e - 2 \pm 1 . 6 e - 2$ </td></tr></table>

Table B.4: 8-Gaussians (36d): comparison in $W _ { 2 } .$ , ED, and MMD (mean ± std over 10 seeds).
<table><tr><td>DIST</td><td>Method</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED ↓</td><td>MMD↓</td><td> $\overline { { { \mathrm { t i m e / e p o c h } } } }$ </td></tr><tr><td>q=1.0</td><td>OT</td><td> $5 . 6 6 0 e 0 \pm 1 . 4 3 e \cdot$  -2</td><td> $\overline { { 5 . 9 3 e { - 2 } \pm 1 . 2 5 e { - 2 } } }$ </td><td> $\overline { { 1 . 3 9 e - 4 \pm 5 . 0 9 e } } .$  -5</td><td>5.92e  $\overline { { \cdot 2 \pm 1 . 2 7 e - 4 } }$ </td></tr><tr><td>q=2.0</td><td>OT</td><td> $5 . 4 9 3 e 0 \pm 1 . 4 1 e -$  -2</td><td> $6 . 6 2 e - 2 \pm 7 . 9 3 e - 3$ </td><td> $2 . 4 7 e - 4 \pm 4 . 9 9 e -$  -5</td><td>5.70e-  $- 2 \pm 1 . 4 1 e { - 4 }$ </td></tr><tr><td>q=1.0</td><td>Sino</td><td> $\overline { { 5 . 4 9 9 e 0 \pm 1 . 0 3 e { - 2 } } }$ </td><td> $\overline { { 4 . 0 6 e - 2 \pm 6 . 2 8 e - 3 } }$ </td><td>9.67e−5 ± 2.37e−5</td><td>5.82e  $\overline { { - 2 \pm 1 . 1 1 e { - 4 } } }$ </td></tr><tr><td>q=2.0</td><td>Sino</td><td> $5 . 3 2 9 e 0 \pm 2 . 8 6 e - 3$  </td><td> $1 . 2 9 e - 1 \pm 2 . 6 6 e - 3$ </td><td> $1 . 1 7 e - 3 \pm 4 . 8 5 e - 5$ </td><td> $5 . 6 9 e { - 2 } \pm 8 . 2 6 e { - 5 }$ </td></tr><tr><td> $\mathrm { q } { = } 1 . 0$ </td><td>RFM</td><td> $5 . 4 9 e 0 \pm 8 . 7 9 e - 3$ </td><td> $7 . 2 8 e { - 2 } \pm 6 . 7 3 e { - 3 }$ </td><td> $2 . 9 3 e - 4 \pm 4 . 3 0 e - 5$ </td><td> $7 . 4 7 e - 2 \pm 3 . 1 6 e - 3$ </td></tr><tr><td> $\mathrm { q = 2 . 0 }$ </td><td>RFM</td><td> $5 . 5 0 e 0 \pm 1 . 6 1 e - 2$ </td><td> $6 . 4 1 e { - 2 } \pm 7 . 8 9 e { - 3 }$ </td><td> $2 . 2 4 e { - } 4 \pm 6 . 1 0 e { - } 5$ </td><td> $6 . 8 9 e { - 2 } \pm 3 . 3 1 e { - 3 }$ </td></tr><tr><td>q=1.0</td><td>MFM</td><td> $\overline { { 5 . 5 8 e 0 \pm 1 . 4 5 e - 2 } }$ </td><td> $6 . 9 8 e { - 2 } \pm 1 . 1 1 e { - 2 }$ </td><td> $2 . 1 2 e { - } 4 \pm 6 . 1 8 e { - } 5$ </td><td> $\overline { { 4 . 1 0 e - 1 \pm 1 . 4 9 e - 2 } }$ </td></tr><tr><td>q=2.0</td><td>MFM</td><td> $5 . 5 8 e 0 \pm 1 . 3 2 e - 2$ </td><td> $5 . 6 8 e { - 2 } \pm 8 . 4 0 e { - 3 }$ </td><td> $1 . 4 8 e { - } 4 \pm 3 . 7 3 e { - } 5$ </td><td> $3 . 6 6 e - 1 \pm 1 . 1 8 e - 2$ </td></tr><tr><td>q=1.0</td><td>FFM</td><td> $\overline { { 5 . 7 9 e 0 \pm 2 . 2 9 e - 2 } }$ </td><td> $\overline { { 4 . 5 3 e - 2 \pm 3 . 4 8 e - 2 } }$ </td><td> $\overline { { 2 . 1 4 e - 4 \pm 2 . 8 2 e - 4 } }$ </td><td> $\overline { { 8 . 2 3 e - 3 \pm 1 . 1 3 e - 4 } }$ </td></tr><tr><td>q=2.0</td><td>FFM</td><td> $5 . 8 0 e 0 \pm 3 . 7 8 e - 2$ </td><td> $5 . 5 6 e - 2 \pm 7 . 0 5 e - 2$ </td><td> $4 . 8 9 e { - } 4 \pm 1 . 2 7 e { - } 3$ </td><td> $6 . 7 6 e - 3 \pm 1 . 3 1 e - 4$ </td></tr><tr><td>q=1.0</td><td>PG-iso</td><td> $\overline { { 5 . 3 9 9 e 0 \pm 2 . 7 4 e - 2 } }$ </td><td> $\overline { { 1 . 0 5 e - 1 \pm 1 . 0 8 e - 2 } }$ </td><td> $\overline { { 7 . 4 8 e - 4 \pm 1 . 6 2 e - 4 } }$ </td><td> $\overline { { 8 . 0 0 e { - 2 } \pm 1 . 9 8 e { - 3 } } }$ </td></tr><tr><td>q=2.0</td><td>PG-iso</td><td> $5 . 4 5 5 e 0 \pm 7 . 6 4 e - 3$ </td><td> $7 . 6 7 e - 2 \pm 5 . 6 5 e - 3$ </td><td> $3 . 9 3 e { - } 4 \pm 5 . 8 5 e { - } 5$ </td><td> $6 . 7 5 e { - 2 } \pm 1 . 6 1 e { - 3 }$ </td></tr><tr><td>q=1.0</td><td>PG-aniso</td><td> $\overline { { 5 . 5 9 7 e 0 \pm 5 . 5 1 e - 3 } }$  1</td><td> $4 . 0 4 e { - 2 } \pm 6 . 2 3 e { - 3 }$  </td><td> $6 . 5 7 e - 5 \pm 1 . 7 3 e - 5$ </td><td>1  $\overline { { 6 . 5 9 e - 2 \pm 2 . 5 4 e - 4 } }$ </td></tr><tr><td> $\mathrm { q = 2 . 0 }$ </td><td>PG-aniso</td><td> $5 . 5 6 3 e 0 \pm 1 . 6 4 e - 2$ </td><td> $5 . 0 8 e { - 2 } \pm 4 . 6 9 e { - 3 }$ </td><td> $1 . 5 0 e { - 4 } \pm 2 . 2 2 e { - 5 }$ </td><td> $6 . 4 2 e - 2 \pm 4 . 7 7 e - 4$ </td></tr><tr><td> $\mathrm { q } { = } 1 . 0$ </td><td>HB</td><td> $\overline { { 5 . 6 3 8 e 0 \pm 1 . 6 3 e - 2 } }$ </td><td> $\overline { { 4 . 4 4 e { - 2 } \pm 1 . 1 6 e { - 2 } } }$ </td><td> $7 . 3 3 e - 5 \pm 3 . 5 9 e - 5$ </td><td> $7 . 8 4 e - 2 \pm 9 . 4 4 e - 4$ </td></tr><tr><td> $\mathrm { q = 2 . 0 }$ </td><td>HB</td><td> $5 . 5 7 1 e 0 \pm 1 . 1 6 e - 2$ </td><td> $5 . 2 1 e { - 2 } \pm 5 . 0 0 e { - 3 }$ </td><td> $1 . 3 7 e { - } 4 \pm 2 . 6 2 e { - } 5$ </td><td> $6 . 8 5 e - 2 \pm 3 . 8 4 e - 4$ </td></tr></table>

Table B.5: Funnel example (10d) comparison in $W _ { 2 } .$ , ED, and MMD (mean ± std over 10 seeds).
<table><tr><td>DIST</td><td>PATH</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td> $\overline { { \mathrm { E D ~ \downarrow ~ } } }$ </td><td>MMD ↓</td><td>time/epoch</td></tr><tr><td> $\stackrel { \mathrm { \scriptsize ~ q = 1 . 0 } } { }$ </td><td>VP</td><td> $2 . 5 5 e 0 \pm 1 . 0 e - 1$ </td><td> $\overline { { 9 . 6 2 e - 2 \pm 1 . 5 e - 2 } }$ </td><td> $\overline { { 3 . 0 4 e - 3 \pm 1 . 2 e - 3 } }$ </td><td> $\overline { { 5 . 5 0 e { - 2 } \pm 4 . 3 e { - 4 } } }$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td>VP</td><td> $2 . 4 5 e 0 \pm 4 . 6 e - 2$ </td><td> $9 . 4 8 e { - 2 } \pm 3 . 3 e { - 2 }$ </td><td> $4 . 1 0 e { - 3 } \pm 3 . 0 e { - 3 }$ </td><td> $5 . 2 7 e - 2 \pm 1 . 2 e - 4$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td>OT</td><td> $2 . 4 7 e 0 \pm 9 . 2 e - 2$ </td><td> $3 . 2 5 e - 2 \pm 1 . 6 e - 2$ </td><td> $2 . 3 6 e { - } 4 \pm 1 . 5 e { - } 4$ </td><td> $5 . 5 9 e - 2 \pm 5 . 6 e - 4$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td>OT</td><td> $2 . 5 4 e 0 \pm 1 . 5 e - 1$ </td><td> $2 . 6 9 e { - } 2 \pm 1 . 3 e { - } 2$ </td><td> $1 . 6 3 e { - } 4 \pm 8 . 5 e { - } 5$ </td><td> $5 . 3 7 e - 2 \pm 4 . 0 e - 4$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td>sino</td><td> $2 . 5 1 e 0 \pm 8 . 7 e \mathrm { - 2 }$ </td><td> $3 . 6 4 e { - 2 } \pm 1 . 1 e { - 2 }$ </td><td> $2 . 7 6 e { - } 4 \pm 1 . 7 e { - } 4$ </td><td> $5 . 6 4 e - 2 \pm 2 . 0 e - 4$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td>sino</td><td> $2 . 5 5 e 0 \pm 2 . 5 e - 1$ </td><td> $3 . 0 7 e - 2 \pm 8 . 6 e - 3$  </td><td> $1 . 5 0 e { - } 4 \pm 6 . 3 e { - } 5$  </td><td> $5 . 3 0 e { - 2 } \pm 1 . 3 e { - 3 }$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td>PG-iso</td><td> $2 . 6 3 e 0 \pm 2 . 7 e - 1$ </td><td> $3 . 5 4 e { - 2 } \pm 1 . 1 e { - 2 }$ </td><td> $3 . 3 6 e - 4 \pm 2 . 2 e - 4$ </td><td> $8 . 0 0 e { - 2 } \pm 4 . 8 e { - 4 }$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td>PG-iso</td><td> $2 . 8 7 e 0 \pm 4 . 2 e - 1$ </td><td> $3 . 4 4 e - 2 \pm 1 . 6 e - 2$ </td><td> $4 . 4 1 e - 4 \pm 3 . 0 e - 4$ </td><td> $7 . 7 8 e - 2 \pm 6 . 5 e - 4$ </td></tr><tr><td> $\mathrm { q = } 1 . 0$ </td><td> $\mathrm { P G } \mathrm { - a n i s o }$ </td><td> $2 . 6 1 e 0 \pm 1 . 9 e - 1$  _</td><td> $2 . 5 4 e - 2 \pm 1 . 0 e - 2$  </td><td> $2 . 5 3 e { - } 4 \pm 1 . 4 e { - } 4$ </td><td> $6 . 2 4 e { - 2 } \pm 2 . 6 e { - 4 }$ </td></tr><tr><td> $\mathrm { q { = } 2 . 0 }$ </td><td> $\mathrm { P G } \mathrm { - a n i s o }$ </td><td> $3 . 0 9 e 0 \pm 4 . 9 e - 1$ </td><td> $4 . 5 4 e { - 2 } \pm 1 . 2 e { - 2 }$ </td><td> $5 . 7 1 e { - } 4 \pm 2 . 9 e { - } 4$ </td><td> $6 . 0 4 e { - 2 } \pm 2 . 2 e { - 4 }$ </td></tr></table>

![](images/78b9ca09e2e629b656da068b423dc0a60593008b17d2dfd80049d83b6e243fba.jpg)  
Figure B.2: Sample paths in the checkerboard (2d, left) and Neal’s funnel (10d, right) examples: VP, OT, sino, PG-2, and PG-1 (from top to bottom).

![](images/116c45611264a7eaaeb4c390a1be93e38d7f5023c76aeb3fb98eaa754e6c2ee7.jpg)

![](images/dfaf38c6dd8592e1a12cab3b7aa21d5173d9d1f57e19506d8f6d615994054baa.jpg)  
Figure B.3: Energy distance (ED) of OT, sino, PG-2, and PG-1 in the 8-Gaussians (36d, left) and funnel (100d, right) examples.

Tables 1 and B.5 further support the advantage and dimension robustness of PG methods as they achieve the lowest or comparable discrepancy metrics. Except a few cells in these tables, $q = 1 . 0$ almost unanimously beats $q = 2 . 0$ which again justifies the adoption of exponential power distribution when handling heavy-tail and inhomogeneity in the data. Note in Table 1, MFM reaches the same lowest ED at a much higher time cost (4.36e − 1 vs 6.92e − 2 seconds per epoch for HB).

## B.4 Scientific Datasets

Next, we investigate the following four datasets from molecular science, genetics, and quantum field theory: Alanine dipeptide (30d) is the standard small-molecule benchmark popular for generative modeling [25]. QM9 (87d) contains 133,885 stable small organic molecules with up to nine heavy atoms (C, N, O, F) from GDB-17 [28, 27]. Single-cell (sc)RNA (50d) contains 2,700 PBMCs from a healthy donor, freely available from 10x Genomics [32]. Lattice ϕ<sup>4</sup> (64d) is from the 2D scalar field theory, used as a benchmark for generative models [1].

These datasets of various dimensions contain complex data structures and serve as competitive benchmarks to test FM generative models. In Figure B.4, we visualize the sample paths of the FM methods on alanine dipeptide and QM9 datasets by projecting them to 2d subspaces using the t-SNE algorithm. Again PG methods maintain high speed in feature discovery. The PG methods demonstrate better (Table B.7) or comparable (Table B.6) performance.

## B.5 De novo Molecule Generation

The task is to draw a complete molecule from scratch: no reference structure is supplied, and a single sample must specify the 3D coordinates, the atom type and the formal charge of every atom, together with the bond order of every atom pair, jointly and consistently. Writing n for the number of atoms and $d _ { a } , d _ { c } , d _ { e }$ for the number of atom types, charge states and bond types, one molecule is a joint

![](images/2cf820c3e129223828f66e740535e4c13c638b8d65e57b70fa539f383d329a19.jpg)  
Figure B.4: Sample paths in the alanine dipeptide (30d, left) and QM9 (87d, right): VP, OT, sino, PG-2, and PG-1 (from top to bottom). The last column of each panel represents the real data. High dimensions have been projected to 2d using t-SNE for visualization.

Table B.6: Comparison across Alanine dipeptide (30d) and QM9 (87d) in $W _ { 2 }$ , ED, and MMD (mean ± std over 10 seeds).
<table><tr><td rowspan="2">DIST</td><td colspan="4">Alanine dipeptide (30d)</td><td colspan="3">QM9 (87d)</td></tr><tr><td>PATH</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED ↓</td><td>MMD↓</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED ↓</td><td>MMD↓</td></tr><tr><td>1.0</td><td>OT</td><td>1.49e0 ± 1.4e−1</td><td> $7 . 7 5 e - 2 \pm 3 . 1 e - 2$ </td><td> $\overline { { 4 . 0 2 e { - } 4 } } \pm 2 . 6 e { - } 4$ </td><td> $\overline { 7 . 2 7 e 0 \pm 8 . 8 e - 1 }$ </td><td>1.53e−1 ± 1.2e−2</td><td>8.65e−4 ± 1.3e−4</td></tr><tr><td>2.0</td><td>OT</td><td>1.41e0 ± 1.1e−1</td><td></td><td>5.62e-2±2.3e-2 1.71e-4±1.2e-4</td><td>6.84e0 ±3.2e-1</td><td>1.31e−1 ± 1.6e−2</td><td>6.88e−4 ± 1.8e−4</td></tr><tr><td>1.0</td><td>sino</td><td> $1 . 5 6 e 0 \pm 1 . 0 e - 1$ </td><td> $8 . 8 6 e - 2 \pm 3 . 6 e - 2$ </td><td> $5 . 0 0 e - 4 \pm 3 . 0 e - 4$ </td><td> $7 . 0 5 e 0 \pm 2 . 5 e - 1$ </td><td>1.27e−1 ± 1.6e-2</td><td>6.29e−4 ± 1.7e−4</td></tr><tr><td>2.0</td><td>sino</td><td> $1 . 4 5 e 0 \pm 1 . 3 e - 1$ </td><td> $6 . 5 9 e { - 2 } \pm 3 . 3 e { - 2 }$ </td><td> $2 . 8 8 e { - } 4 \pm 2 . 1 e { - } 4$ </td><td> $7 . 0 4 e 0 \pm 3 . 4 e - 1$ </td><td>1.27e−1±3.0e-2</td><td>6.33e−4 ± 3.4e−4</td></tr><tr><td>1.0</td><td>PG</td><td> $1 . 6 2 e 0 \pm 1 . 8 e - 1$ </td><td> $1 . 0 5 e { - } 1 \pm 3 . 7 e { - } 2$ </td><td> $7 . 4 0 e - 4 \pm 4 . 9 e - 4$ </td><td> $6 . 9 4 e 0 \pm 3 . 3 e - 1$ </td><td> $\overline { { 1 . 6 3 e { - } 1 \pm 1 . 9 e { - } 2 } }$ </td><td>9.05e−4 ± 2.0e−4</td></tr><tr><td>2.0</td><td>PG</td><td> $1 . 5 1 e 0 \pm 9 . 4 e - 2$ </td><td> $8 . 4 8 e { - 2 } \pm 2 . 5 e { - 2 }$ </td><td> $4 . 4 0 e { - } 4 \pm 2 . 1 e { - } 4$ </td><td> $7 . 0 0 e 0 \pm 3 . 3 e - 1$ </td><td> $1 . 5 6 e { - } 1 \pm 1 . 8 e { - } 2$ </td><td>8.54e−4 ± 2.2e−4</td></tr><tr><td>1.0</td><td>HB</td><td> $1 . 7 3 e 0 \pm 1 . 9 e - 1$ </td><td> $2 . 0 0 e \mathrm { - } 1 \pm 3 . 4 e \mathrm { - } 2$ </td><td> $3 . 2 2 e - 3 \pm 9 . 7 e - 4$ </td><td> $7 . 2 7 e 0 \pm 2 . 9 e - 1$ </td><td> $2 . 0 2 e - 1 \pm 2 . 0 e - 2$ </td><td>1.58e−3 ± 3.4e−4</td></tr><tr><td>2.0</td><td>HB</td><td> $1 . 6 6 e 0 \pm 7 . 2 e \mathrm { - 2 }$ </td><td> $1 . 9 2 e - 1 \pm 1 . 6 e - 2$ </td><td> $3 . 0 6 e { - 3 \pm 5 . 8 e { - 4 } }$ </td><td> $7 . 4 8 e 0 \pm 2 . 9 e - 1$ </td><td> $2 . 1 7 e - 1 \pm 1 . 5 e - 2$ </td><td>1.85e−3 ± 3.0e−4</td></tr></table>

event of dimension

$$
D ( n ) = \underbrace { \mathsf { 3 } ( n - 1 ) } _ { \mathrm { c o r d i n a t e s } } + \underbrace { d _ { a } n } _ { \mathrm { a t o m ~ t y p e s } } + \underbrace { d _ { c } n } _ { \mathrm { c h a r g e s } } + \underbrace { d _ { e } { \binom { n } { 2 } } } _ { \mathrm { b o n d s } } = \frac { d _ { e } } { 2 } n ^ { 2 } + \Big ( 3 + d _ { a } + d _ { c } - \frac { d _ { e } } { 2 } \Big ) n - 3 ,\tag{27}
$$

where the coordinate block is $3 ( n - 1 )$ rather than 3n because translations are quotiented out, i.e. the flow acts on the zero-centre-of-mass subspace. For QM9 $( d _ { a } \ : = \ : 5 , \ : d _ { c } \ : = \ : 6 , \ : d _ { e } \ : = \ : 5 )$ this is $\begin{array} { r } { D ( n ) = \frac { 5 } { 2 } n ^ { 2 } + \frac { 2 3 } { 2 } n - 3 . } \end{array}$ , dominated by the bond block, which grows quadratically in n:

Table B.7: Comparison across scRNA (50d) and Lattice $\phi ^ { 4 }$ (64d) in $W _ { 2 }$ , ED, and MMD $( \mathrm { { m e a n } \pm }$ std over 10 seeds).
<table><tr><td rowspan="2">DIST</td><td colspan="4">scRNA (50d)</td><td colspan="3">Lattice φ4 (64d)</td></tr><tr><td>PATH</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED↓</td><td>MMD↓</td><td> $\overline { { W _ { 2 } \downarrow } }$ </td><td>ED ↓</td><td>MMD ↓</td></tr><tr><td>1.0</td><td>OT</td><td> $\overline { { 5 . 3 2 e 0 \pm 4 . 0 e - 2 } }$  1</td><td> $\overline { { 5 . 1 7 e - 1 \pm 2 . 0 e - 2 } }$ </td><td> $\overline { { 1 . 8 8 e { - } 2 \pm 1 . 5 e { - } 3 } }$ </td><td> $\overline { { 7 . 5 4 e 0 \pm 9 . 0 e - 3 } }$ </td><td> $\overline { { 2 . 0 8 e { - } 1 \pm 4 . 0 e { - } 3 } }$ </td><td> $2 . 1 8 e - 3 \pm 8 . 0 e - 5$ </td></tr><tr><td>2.0</td><td>OT</td><td> $5 . 3 3 e 0 \pm 5 . 0 e - 2$ </td><td> $5 . 1 0 e { - } 1 \pm 2 . 0 e { - } 2$ </td><td>1.82e−2 ± 1.3e−3</td><td> $7 . 5 3 e 0 \pm 1 . 1 e - 2$ </td><td>2.06e−1 ± 6.0e−3</td><td>2.14e−3 ± 1.2e−4</td></tr><tr><td>1.0</td><td>sino</td><td> $5 . 4 1 e 0 \pm 5 . 0 e - 2$ </td><td> $5 . 3 6 e { - } 1 \pm 2 . 0 e { - } 2$ </td><td> $2 . 0 4 e { - } 2 \pm 1 . 5 e { - } 3$ </td><td> $7 . 5 0 e 0 \pm 1 . 0 e \mathrm { - 2 }$ </td><td>2.25e−1 ± 4.0e−3</td><td>2.64e−3 ± 1.1e−4</td></tr><tr><td>2.0</td><td>sino</td><td> $5 . 4 4 e 0 \pm 5 . 0 e - 2$ </td><td> $5 . 4 6 e - 1 \pm 2 . 0 e - 2$ </td><td> $2 . 0 9 e { - } 2 \pm 1 . 4 e { - } 3$ </td><td> $7 . 4 7 e 0 \pm 1 . 0 e { \mathrm { - } } 2$ </td><td> $2 . 2 5 e - 1 \pm 4 . 0 e - 3$ </td><td> $2 . 7 4 e - 3 \pm 9 . 0 e - 5$ </td></tr><tr><td>1.0</td><td>PG</td><td> $5 . 7 1 e 0 \pm 4 . 0 e - 2$ </td><td> $2 . 9 3 e - 1 \pm 2 . 2 e - 2$ </td><td> $5 . 0 7 e - 3 \pm 6 . 0 e - 4$ </td><td> $7 . 7 3 e 0 \pm 1 . 4 e - 2$ </td><td>1.55e−1 ± 7.0e−3</td><td> $1 . 0 6 e { - 3 } \pm 8 . 0 e { - 5 }$ </td></tr><tr><td>2.0</td><td>PG</td><td> $5 . 7 5 e 0 \pm 5 . 0 e - 2$ </td><td>2.73e−1 ± 1.3e−2</td><td> $4 . 1 4 e - 3 \pm 4 . 0 e - 4$ </td><td> $7 . 6 9 e 0 \pm 2 . 0 e - 2$ </td><td>1.48e-1±7.0e-3</td><td>9.99e-4±1.1e-4</td></tr><tr><td>1.0</td><td>HB</td><td> $5 . 7 2 e 0 \pm 5 . 1 e - 2$ </td><td>1.83e−1 ± 2.1e−2</td><td> $1 . 4 2 e { - } 3 \pm 3 . 7 e { - } 4$ </td><td> $8 . 3 1 e 0 \pm 1 . 5 e - 2$ </td><td>1.78e−1 ± 6.5e−3</td><td> $1 . 2 1 e { - } 3 \pm 9 . 6 e { - } 5$ </td></tr><tr><td>2.0</td><td>HB</td><td> $6 . 0 2 e 0 \pm 4 . 1 e - 2$ </td><td> $1 . 5 8 e { - } 1 \pm 1 . 6 e { - } 2$ </td><td> $8 . 1 8 e { - } 4 \pm 1 . 6 e { - } 4$ </td><td> $8 . 2 8 e 0 \pm 2 . 4 e \mathrm { - 2 }$ </td><td>1.60e−1 ± 1.2e−2</td><td> $1 . 0 7 e { - } 3 \pm 1 . 4 e { - } 4$ </td></tr></table>

<table><tr><td>Modality</td><td>Dimension</td><td> $n = 3$ </td><td> $n = 9$ </td><td> $n = 1 8$ </td><td> $n = 2 9$ </td></tr><tr><td>Coordinates (zero-COM)</td><td> $3 ( n - 1 )$ </td><td>6</td><td>24</td><td>51</td><td>84</td></tr><tr><td>Atom types</td><td> $d _ { a } n$ </td><td>15</td><td>45</td><td>90</td><td>145</td></tr><tr><td>Charges</td><td> $d _ { c } n$ </td><td>18</td><td>54</td><td>108</td><td>174</td></tr><tr><td>Bonds (upper triangle)</td><td> $d _ { e } \binom { n } { 2 }$ </td><td>15</td><td>180</td><td>765</td><td>2030</td></tr><tr><td>Total</td><td> $D ( n )$ </td><td>54</td><td>303</td><td>1014</td><td>2433</td></tr></table>

QM9 molecules contain at most nine heavy atoms and hence up to $n = 2 9$ atoms with hydrogens, averaging roughly n ≈ 18: each molecule is an event of 54–2,433 dimensions, averaging roughly $1 0 ^ { 3 }$ of which the bond block alone accounts for 83% at $n = 2 9$ . De novo generation on this benchmark is therefore a problem of order $1 0 ^ { 3 }$ dimensions per sample, despite QM9 being a small-molecule dataset.

## C Anisotropic Probabilistic Geodesic Flow

We now extend the geodesic flow of Section 4 to the anisotropic location–scale family in which the scalar scale $\sigma _ { t }$ is replaced by a per-coordinate scale vector ${ \pmb { \sigma } } _ { t } \in \mathbb { R } _ { + } ^ { d }$ , so that the covariance becomes diag(σ<sup>2</sup>). Following the component-wise independence assumed in Theorem 3.1, we take the anisotropic exponential power conditional density to be the product of d univariate marginals,

$$
p _ { t } ( \mathbf { x } ; \mu _ { t } , \sigma _ { t } ) \ = \ \prod _ { i = 1 } ^ { d } p _ { t } ( x _ { i } ; \mu _ { t , i } , \sigma _ { t , i } ^ { 2 } ) , \quad p _ { t } ( x _ { i } ; \mu _ { i } , \sigma _ { i } ^ { 2 } ) \ = \ \frac { q } { 2 ^ { 1 + \frac { 1 } { q } } \Gamma ( 1 / q ) \sigma _ { i } } \exp \Bigl \{ - \frac { 1 } { 2 } \bigl | ( x _ { i } - \mu _ { i } ) / \sigma _ { i } \bigr | ^ { q } \Bigr \} .\tag{28}
$$

The endpoints of the conditional flow are unchanged in form, $\begin{array} { r } { p _ { 0 } ( \cdot \mid \mathbf { x } _ { 1 } ) = \prod _ { i } \mathcal { P } _ { q } ( \cdot ; 0 , 1 ) } \end{array}$ and $p _ { 1 } ( \cdot \mid$ $\begin{array} { r } { { \bf x } _ { 1 } ) = \prod _ { i } \mathcal { P } _ { q } ( \cdot ; x _ { 1 , i } , \sigma _ { \operatorname* { m i n } } ^ { 2 } ) , \mathrm { i . e . } \ \mu _ { 0 } = { \bf 0 } , \ \sigma _ { 0 } = { \bf 1 } , \ \mu _ { 1 } = { \bf x } _ { 1 } , \ \sigma _ { 1 } = \sigma _ { \operatorname* { m i n } } { \bf 1 } } \end{array}$

Anisotropic Fisher metric. Because (28) factorizes across coordinates and the parameters $( \mu _ { i } , \sigma _ { i } )$ of distinct factors are decoupled, the joint Fisher information is block-diagonal:

$$
\mathbf { I } ( \mu , \sigma ) = \bigoplus _ { i = 1 } ^ { d } \mathbf { I } ^ { ( 1 ) } ( \mu _ { i } , \sigma _ { i } ) .
$$

where each $2 \times 2$ block is the univariate exponential power Fisher information obtained by specializing the derivation of Section 4 to $d = 1$

$$
{ \bf I } ^ { ( 1 ) } ( \mu _ { i } , \sigma _ { i } ) ~ = ~ { \frac { 1 } { \sigma _ { i } ^ { 2 } } } \left( { c _ { \mu } } \atop 0  \right. \left. { \begin{array} { l } { { 0 } } \\ { { c _ { \sigma } } } \end{array} } \right) , \quad c _ { \mu } ~ = ~ 2 ^ { - 2 / q } q ^ { 2 } { \frac { \Gamma \big ( 2 - { \frac { 1 } { q } } \big ) } { \Gamma ( 1 / q ) } } , \quad c _ { \sigma } ~ = ~ q .\tag{29}
$$

(The vanishing of the of-diagonal $\mu - \sigma$ block follows from $\mathbb { E } [ u _ { i } ] = \mathbb { E } [ u _ { i } u _ { j } ^ { 2 } ] = 0$ on $S ^ { d - 1 }$ , exactly as in the isotropic case.) Consequently the anisotropic Fisher-Rao metric is the product of d rescaled Poincar´e half-plane metrics,

$$
d s ^ { 2 } \ : = \ : \sum _ { i = 1 } ^ { d } \frac { c _ { \mu } d \mu _ { i } ^ { 2 } + c _ { \sigma } d \sigma _ { i } ^ { 2 } } { \sigma _ { i } ^ { 2 } } \ : = \ : \sum _ { i = 1 } ^ { d } d s _ { i } ^ { 2 } .\tag{30}
$$

Per-coordinate geodesic equations. Because (30) is a Riemannian product, geodesics in the joint manifold $( \mathbb { R } \times \mathbb { R } _ { + } ) ^ { d }$ are Cartesian products of geodesics in each $H ^ { 2 }$ factor; the Christofel symbols are nonzero only within a coordinate. Repeating the computation of Section 4 in every factor yields, for each $i \in \{ 1 , \ldots , d \}$ 2

$$
\begin{array} { r } { \ddot { \mu } _ { i } ~ - ~ \frac { 2 } { \sigma _ { i } } \dot { \mu } _ { i } \dot { \sigma } _ { i } = 0 , } \end{array}\tag{31a}
$$

$$
\begin{array} { r } { \ddot { \sigma } _ { i } ~ + ~ \frac { c _ { \mu } } { c _ { \sigma } \sigma _ { i } } \dot { \mu } _ { i } ^ { 2 } ~ - ~ \frac { 1 } { \sigma _ { i } } \dot { \sigma } _ { i } ^ { 2 } = 0 . } \end{array}\tag{31b}
$$

Each pair (31) is identical to the isotropic system (21a)–(21b) of Section 4, decoupled across i.

Closed-form solution: Poincar´e semi-circle (per coordinate). Set $\lambda : = \sqrt { c _ { \sigma } / c _ { \mu } }$ and, for each coordinate i, write $\Delta _ { i } : = \mu _ { 1 , i } - \mu _ { 0 , i }$ . Case $\Delta _ { i } \neq 0$ . The geodesic in the ith $H ^ { 2 }$ factor is the unique semi-circle in the $( \mu _ { i } , \lambda \sigma _ { i } )$ upper half-plane connecting $\left( \mu _ { 0 , i } , \lambda \sigma _ { 0 , i } \right) \mathrm { t o } \left( \mu _ { 1 , i } , \lambda \sigma _ { 1 , i } \right)$

$$
( \mu _ { i } - \mu _ { 0 , i } - \mu _ { c , i } ) ^ { 2 } + ( \lambda \sigma _ { i } ) ^ { 2 } = R _ { i } ^ { 2 } ,\tag{32}
$$

with center and radius

$$
\mu _ { c , i } = \frac { \Delta _ { i } ^ { 2 } + \lambda ^ { 2 } ( \sigma _ { 1 , i } ^ { 2 } - \sigma _ { 0 , i } ^ { 2 } ) } { 2 \Delta _ { i } } , ~ R _ { i } = \sqrt { \mu _ { c , i } ^ { 2 } + \lambda ^ { 2 } \sigma _ { 0 , i } ^ { 2 } } .\tag{33}
$$

Parametrizing $( \mu _ { i } ( t ) - \mu _ { 0 , i } , \lambda \sigma _ { i } ( t ) ) = ( \mu _ { c , i } + R _ { i } \cos \theta _ { i } ( t ) , R _ { i } \sin \theta _ { i } ( t ) )$ with the constant-arc-length parametrization

$$
\theta _ { i } ( t ) = 2 \arctan \bigl ( e ^ { L _ { i } t } \tan ( \theta _ { 0 , i } / 2 ) \bigr ) , \qquad L _ { i } = \log \frac { \tan ( \theta _ { 1 , i } / 2 ) } { \tan ( \theta _ { 0 , i } / 2 ) } ,\tag{34}
$$

where the boundary angles $\theta _ { 0 , i } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 0 , i } , - \mu _ { c , i } )$ and $\theta _ { 1 , i } = \mathrm { a t a n 2 } ( \lambda \sigma _ { 1 , i } , \Delta _ { i } - \mu _ { c , i } )$ both lie in $( 0 , \pi )$ , gives the explicit geodesic

$$
\begin{array} { l } { \displaystyle \mu _ { i } ( t ) = \mu _ { 0 , i } + \mu _ { c , i } + R _ { i } \cos \theta _ { i } ( t ) , } \\ { \displaystyle \sigma _ { i } ( t ) = \frac { R _ { i } } { \lambda } \sin \theta _ { i } ( t ) . } \end{array}\tag{35}
$$

Case $\Delta _ { i } = 0$ . The geodesic is the vertical $\mu _ { i } ( t ) \equiv \mu _ { 0 , i }$ together with the geometric scale interpolation $\sigma _ { i } ( t ) = \sigma _ { 0 , i } ^ { 1 - t } \sigma _ { 1 , i } ^ { t }$ , equivalently log $\sigma _ { i }$ linear in t.

Equivalent hyperbolic (sech/tanh) parametrization. The sech/tanh form of Section 4 carries over coordinate-wise. With $\omega _ { i } , \kappa _ { i } , \alpha _ { i }$ determined from the boundary conditions,

$$
\boxed { \mu _ { i } ( t ) = \mu _ { 0 , i } + ( \mu _ { 1 , i } - \mu _ { 0 , i } ) \frac { \operatorname { t a n h } ( \omega _ { i } t + \alpha _ { i } ) - \operatorname { t a n h } ( \alpha _ { i } ) } { \operatorname { t a n h } ( \omega _ { i } + \alpha _ { i } ) - \operatorname { t a n h } ( \alpha _ { i } ) } , \quad \sigma _ { i } ( t ) = \frac { \omega _ { i } } { \kappa _ { i } } \operatorname { s e c h } ( \omega _ { i } t + \alpha _ { i } ) , }\tag{36}
$$

where $\omega _ { i } , \kappa _ { i } , \alpha _ { i }$ solve

$$
\frac { \omega _ { i } } { \kappa _ { i } } \mathrm { s e c h } ( \alpha _ { i } ) = \sigma _ { 0 , i } ,
$$

$$
{ \frac { \omega _ { i } } { \kappa _ { i } } } \operatorname { s e c h } ( \omega _ { i } + \alpha _ { i } ) = \sigma _ { 1 , i } ,
$$

$$
\kappa _ { i } = | \Delta _ { i } | \sqrt { \frac { c _ { \mu } } { c _ { \sigma } } } \frac { \kappa _ { i } ^ { 2 } } { \omega _ { i } \left( \operatorname { t a n h } ( \omega _ { i } + \alpha _ { i } ) - \operatorname { t a n h } ( \alpha _ { i } ) \right) } .
$$

The two parametrisations are related by

$$
\sin ( 2 \arctan e ^ { - s _ { i } } ) = \operatorname { s e c h } s _ { i } , \qquad \cos ( 2 \arctan e ^ { - s _ { i } } ) = \operatorname { t a n h } s _ { i } ,
$$

where $s _ { i } = \omega _ { i } t + \alpha _ { i }$ and $R _ { i } = \lambda \omega _ { i } / \kappa _ { i }$

Anisotropic Fisher–Rao distance and energy. The arc-length on the ith factor along the geodesic is

$$
| L _ { i } | = \mathrm { a r c o s h } \Bigg ( 1 + { \frac { c _ { \mu } \Delta _ { i } ^ { 2 } + c _ { \sigma } ( \sigma _ { 1 , i } - \sigma _ { 0 , i } ) ^ { 2 } } { 2 c _ { \sigma } \sigma _ { 0 , i } \sigma _ { 1 , i } } } \Bigg ) .\tag{37}
$$

Because the metric is a Riemannian product, the joint squared Fisher–Rao distance and the geodesic energy are simply additive,

$$
d _ { \mathrm { F R } } ^ { 2 } ( p _ { 0 } , p _ { 1 } ) = c _ { \sigma } \sum _ { i = 1 } ^ { d } L _ { i } ^ { 2 } , \qquad e ( v ) = c _ { \sigma } \sum _ { i = 1 } ^ { d } L _ { i } ^ { 2 } .\tag{38}
$$

For the symmetric anisotropic boundary ${ \pmb \sigma } _ { 0 } = { \bf 1 } , ~ { \pmb \sigma } _ { 1 } = \sigma _ { \mathrm { m i n } } { \bf 1 }$ , (37) reduces to $L _ { i } = \mathrm { a r c o s h } ( 1 +$ $[ c _ { \mu } x _ { 1 , i } ^ { 2 } + c _ { \sigma } ( \sigma _ { \mathrm { m i n } } - 1 ) ^ { 2 } ] / ( 2 c _ { \sigma } \sigma _ { \mathrm { m i n } } ) )$

Anisotropic vector field. By Theorem 3.1, applied component-wise, the vector field associated with the anisotropic geodesic path is

$$
\boxed { { \textbf { v } _ { t } } ( { \textbf { x } } ) = \dot { \pmb { \mu } } _ { t } + \frac { \dot { \pmb { \sigma } } _ { t } } { \pmb { \sigma } _ { t } } \odot ( { \bf x } - { \pmb { \mu } } _ { t } ) , }\tag{39}
$$

where $\odot$ and the division act component-wise. From (35) and $\dot { \theta } _ { i } ( t ) = L _ { i }$ sin $\theta _ { i } ( t )$ one gets, for every i with $\Delta _ { i } \neq 0$

$$
\dot { \mu } _ { t , i } = - R _ { i } L _ { i } \sin ^ { 2 } \theta _ { i } ( t ) , \qquad \frac { \dot { \sigma } _ { t , i } } { \sigma _ { t , i } } = L _ { i } \cos \theta _ { i } ( t ) ,\tag{40}
$$

so that

$$
v _ { t , i } ( x _ { i } ) = L _ { i } \bigl [ - R _ { i } \sin ^ { 2 } \theta _ { i } ( t ) + \cos \theta _ { i } ( t ) ( x _ { i } - \mu _ { t , i } ) \bigr ] .\tag{41}
$$

For coordinates with $\Delta _ { i } = 0 , \dot { \mu } _ { t , i } = 0$ and $\dot { \sigma } _ { t , i } / \sigma _ { t , i } = \log \sigma _ { 1 , i } - \log \sigma _ { 0 , i }$ is constant in t, so $v _ { t , i } ( x _ { i } ) =$ $( \log \sigma _ { 1 , i } - \log \sigma _ { 0 , i } ) ( x _ { i } - \mu _ { 0 , i } )$

Anisotropic conditional vector field and flow. Letting ${ \pmb \mu } _ { 0 } = { \bf 0 } , \ { \pmb \mu } _ { 1 } = { \bf x } _ { 1 } , \ { \pmb \sigma } _ { 0 } = { \bf 1 } , \ { \pmb \sigma } _ { 1 } = \sigma _ { \mathrm { m i n } } { \bf 1 }$ formula (41) specialises (per coordinate, for $x _ { 1 , i } \neq 0 )$ to

$$
u _ { t , i } ( x _ { i } \mid x _ { 1 , i } ) ~ = ~ L _ { i } \big [ - \left( R _ { i } \sin ^ { 2 } \theta _ { i } ( t ) + \mu _ { c , i } \cos \theta _ { i } ( t ) + R _ { i } \cos ^ { 2 } \theta _ { i } ( t ) \right) \mathrm { s g n } ( x _ { 1 , i } ) 1 _ { x _ { 1 , i } \neq 0 } + \cos \theta _ { i } ( t ) x _ { i } \big ] ,\tag{42}
$$

which, after collecting $R _ { i } \sin ^ { 2 } \theta _ { i } + R _ { i } \cos ^ { 2 } \theta _ { i } = R _ { i }$ , gives

$$
u _ { t , i } ( x _ { i } \mid x _ { 1 , i } ) ~ = ~ L _ { i } \left[ \cos \theta _ { i } ( t ) x _ { i } ~ - ~ ( R _ { i } + \mu _ { c , i } \cos \theta _ { i } ( t ) ) \mathrm { s g n } ( x _ { 1 , i } ) \right] .\tag{43}
$$

Equivalently, in stacked vector form using component-wise scalars $L _ { i } , R _ { i } , \mu _ { c , i } , \theta _ { i } ( t )$

$$
\mathbf { u } _ { t } ( \mathbf { x } \mid \mathbf { x } _ { 1 } ) \ = \ \mathbf { L } \odot \left[ \cos \theta ( t ) \odot \mathbf { x } \ - \ \left( \mathbf { R } + \pmb { \mu } _ { c } \odot \cos \theta ( t ) \right) \odot \mathrm { s g n } ( \mathbf { x } _ { 1 } ) \right] .\tag{44}
$$

The associated conditional flow has the explicit form

$$
\psi _ { t } ( \mathbf { x } \mid \mathbf { x } _ { 1 } ) \ = \ \big ( \pmb { \mu _ { c } } + \mathbf { R } \odot \cos \pmb { \theta } ( t ) \big ) \odot \mathrm { s g n } ( \mathbf { x } _ { 1 } ) \ + \ \frac { 1 } { \lambda } \mathbf { R } \odot \sin \pmb { \theta } ( t ) \odot \mathbf { x } ,\tag{45}
$$

which reduces to the isotropic flow of Section 4 when ${ \pmb { \sigma } } _ { t } = \sigma _ { t } { \bf 1 }$ for some scalar $\sigma _ { t }$ and the angles $\theta _ { i } ( t )$ collapse to a common $\theta ( t )$

Remark (training). With (44) the FM loss (14) is unchanged in form,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \theta ) = \mathbb { E } _ { t , q ( \mathbf { x } _ { 1 } ) , p _ { 0 } ( \mathbf { x } _ { 0 } ) } \Big | \Big | v _ { t } \big ( \psi _ { t } ( \mathbf { x } _ { 0 } ) ; \theta \big ) } \\ { - \cfrac { d } { d t } \psi _ { t } ( \mathbf { x } _ { 0 } ) \Big | \Big | _ { 2 } ^ { 2 } , } \end{array}
$$

but the per-coordinate scalars $\{ L _ { i } , R _ { i } , \mu _ { c , i } , \theta _ { i } ( t ) \} _ { i = 1 } ^ { d }$ must be precomputed once per sampled $\mathbf { x } _ { 1 }$ (closed form, no ODE solve required). Hence the anisotropic PG path is no more expensive at training time than its isotropic counterpart.