# SCORE: Spectral Correlation Estimation for Multivariate Gaussians

Christopher Bülte\*<sup>1,2</sup>

Emil Partow<sup>∗1,2</sup>

Astha Gupta<sup>1</sup>

Pascal Esser<sup>1,2</sup>

Gitta Kutyniok<sup>1,2,3,4</sup>

<sup>1</sup> LMU Munich

<sup>2</sup>Munich Center for Machine Learning (MCML)

<sup>3</sup>DLR-German Aerospace Center

<sup>4</sup>University of Tromsø

## Abstract

Neural network-based predictive modeling with high-dimensional structured Gaussian targets requires an efficient and numerically stable, yet expressive approximation of the covariance matrix. We propose SCORE: a scalable framework, combining scoring rule training with an expressive covariance approximation learned in spectral space. For d-dimensional data, the learning task is decomposed into learning the marginal distributions and learning a structured correlation matrix, which enables dense dependencies with linear storage and O(d log d) cost. We utilize the closed form Gaussian kernel score for training, which remains defined even for degenerate covariances and admits bounded gradients during optimization. We characterize kernel scores under invertible transforms and prove exact invariance under unitary transforms. At population level, our two-level objective recovers the true marginals and projects the target correlation onto the representable class; finite-sample PAC bounds show that the errors of the two stages enter additively. We evaluate our model on a variety of tasks with a commonly assumed Gaussian domain: Time-series forecasting, monocular depth estimation, and spatial weather prediction, showing improved performance at lower computational cost.

## 1 INTRODUCTION

Probabilistic predictions are essential when models inform decisions under uncertainty and have become increasingly important in machine learning [1, 2]. This is particularly important for high-dimensional structured outputs such as spatial fields or multivariate time series, whose uncertainties are correlated across coordinates [3, 4, 5]. This setting arises naturally from measurement noise, aggregation effects, or from inherently stochastic systems. Therefore, it is widely used in scientific domains, including atmospheric forecasting [6], oceanographic fields [7], biological measurements [8], and astronomical time series [9].

In practice, one often assumes a Gaussian predictive, which presents two major challenges in high dimensions. First, a dense covariance matrix requires $O ( d ^ { 2 } )$ parameters and generic matrix operations up to $\mathcal { O } ( d ^ { 3 } )$ time. Diagonal models scale linearly but discard any dependencies, while lowrank and structured alternatives restrict the representable correlations. Second, the commonly used negative loglikelihood (NLL) [10, 11] is undefined for singular covariances and ill-conditioned near the boundary of the positivedefinite cone, where its gradients may become arbitrarily large [12].

We address these complexities by combining a robust proper scoring rule with a covariance approximation that separates marginal uncertainty from dependence structure. As the multivariate objective, we employ the Gaussian kernel score, which is directly related to the maximum mean discrepancy [13, 14, 15], and available in closed form for Gaussian predictions. Unlike the NLL, it remains well-defined for degenerate distributions and admits bounded gradients. At the same time, it is invariant under unitary transformations of the domain, allowing us to learn the dependence structure in a spectral basis, e.g., the Fourier basis, leading to a more computationally efficient representation without altering the optimization objective. Further, we employ a staged estimation approach, which first learns the marginals and recovers the optimal marginal distributions at a population level. In the second stage, we employ the spectral transformation to efficiently approximate the remaining correlation structure, for which we provide an explicit error bound.

Our staged procedure is summarized in Figure 1. First, the marginal distributions are estimated and used to normalize the target. In this transformed domain, we fit a circulant correlation matrix, using an efficient representation in the Fourier domain. Finally, rescaling the learned correlation by the marginal scales yields the approximate predictive covariance matrix.

![](images/3be7dde7587580063f04552ccb68710f54ecffe0a08d0272ca7517aa2195580e.jpg)  
Figure 1: Schematic overview of SCORE. Stage 1 fits the mean $\pmb { \mu }$ and the marginal variances $\tau$ (orange) and standardizes the target, leaving only its correlation structure. Stage 2 models this correlation by a normalized spectrum $s _ { k } ^ { \star }$ in Fourier coordinates, whose back-transform $F ^ { * }$ is the correlation core $R ^ { \star }$ (purple). Rescaling by the frozen marginals yields the predictive covariance $\Sigma _ { P } ( { \mathrm { g r e e n } } )$

For the discrete Fourier transform (DFT) and a diagonal covariance in the spectral domain, referred to as core, the model learns a spectrum whose inverse transform is a dense circulant correlation matrix. Combining the core with the learned marginals admits an expressive dependence structure with linear covariance storage and FFTbased ${ \mathcal { O } } ( d \log d )$ operations. More generally, this applies to any unitary transform and admissible core class. We show that the population objective recovers the true marginals and selects the best representable correlation, characterize its approximation error, and derive finite-sample PAC bounds in which the errors from the two stages enter additively.

## Contributions. Our main contributions are:

1. We propose to use the Gaussian kernel score as a closed form training objective and establish corresponding boundedness and Lipschitz properties, contrasting its regularity with the singular behavior of the NLL.

2. We introduce SCORE (Spectral CORrelation Estimation), a covariance parameterization combining learned marginal scales with a structured correlation core in a spectral domain, characterize its expressivity and population optimum, and provide conditions for exact recovery of the true distribution.

3. We evaluate the framework on real-world highdimensional spatiotemporal prediction tasks, examining predictive performance, calibration, numerical stability, and computational efficiency.

## 2 BACKGROUND

Throughout the paper, let $\mathcal { X } \subseteq \mathbb { R } ^ { k }$ and $\mathcal { V } \subseteq \mathbb { R } ^ { d }$ denote the input and output spaces, respectively, and let $\mathcal { P } ( \mathcal { V } )$ be a convex set of probability measures on $\mathcal { V } .$ . We consider a training dataset $\mathcal { D } = \{ ( \pmb { x } _ { n } , \pmb { y } _ { n } ) \} _ { n = 1 } ^ { N } \subset ( \pmb { \chi } \times \mathcal { y } ) ^ { N }$ consisting of i.i.d. samples from an unknown joint distribution $\mathbb { P } = \mathbb { P } _ { X , Y }$ with marginal $\mathbb { P } _ { X }$ and conditional law $\mathbb { P } _ { Y | X } ( \cdot \mid x )$ . Expectations with respect to P are written as $\mathbb { E } _ { X , Y \sim \mathbb { P } } [ \cdot ]$ , while $P , Q \in { \mathcal { P } } ( { \mathcal { V } } )$ denote generic probability distributions. Our predictive model is a neural network $f _ { \phi } : \mathcal { X }  \mathcal { P } ( \mathcal { Y } )$ with parameters $\phi \in \mathcal { W } \subseteq \mathbb { R } ^ { q }$ , which maps an input x to a probability distribution $f _ { \phi } ( \pmb { x } )$ . Unless stated otherwise, we restrict our attention to Gaussian predictive distributions, $f _ { \phi } ( \pmb { x } ) = \mathcal { N } ( \pmb { \mu } _ { \phi } ( \pmb { x } ) , \pmb { \Sigma } _ { \phi } ( \pmb { x } ) )$ , where $\pmb { \mu } _ { \phi } ( \pmb { x } ) \in \mathbb { R } ^ { d }$ and $\pmb { \Sigma } _ { \phi } ( \pmb { x } ) \succ 0$ denote the predicted mean vector and positive definite covariance matrix, respectively. A full overview is given in Appendix A.

## 2.1 SCORING RULE MINIMIZATION

Predictive models that output distributional parameters are typically fitted by minimizing a proper scoring rule [16], most commonly the log score (NLL) [10, 11]. A scoring rule is a function $\overset { \cdot } { S } : \mathcal { P } ( \overset { \cdot } { \mathcal { V } } ) \times \overset { \cdot } { \mathcal { V } }  \overline { { \mathbb { R } } }$ that assigns a numerical score $S ( P , \pmb { y } )$ to a probabilistic prediction $P \in \mathcal { P } ( \mathcal { V } )$ when y is observed. Following Gneiting and Raftery [13], S is proper relative to a class $\mathcal { F } \subseteq \mathcal { P } ( \mathcal { V } )$ if

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { Y } \sim Q } [ S ( Q , \pmb { Y } ) ] \le \mathbb { E } _ { \pmb { Y } \sim Q } [ S ( P , \pmb { Y } ) ] , \forall P , Q \in \mathcal { F } , } \end{array}\tag{1}
$$

and strictly proper if equality holds only when $P = Q$ Defining $S ( P , Q ) : = \operatorname { \mathbb { E } } _ { Y \sim Q } [ S ( P , Y ) ]$ , any scoring rule S gives rise to a corresponding entropy $H ( P ) : = S ( P , P )$ and divergence $D ( P , Q ) : = S ( P , Q ) - H ( Q )$

Utilizing scoring rules, the training objective of a neural network can be expressed as

$$
\begin{array} { r l } & { \mathrel { \phantom { = } } \arg \operatorname* { m i n } _ { \phi } \mathbb { E } _ { X , \pmb { Y } \sim \mathbb { P } } \left[ S ( f _ { \phi } ( \pmb { X } ) , \pmb { Y } ) \right] } \\ & { \mathrel { \phantom { = } } \mathrm { { a r g m i n } } _ { \phi } \mathbb { E } _ { X \sim \mathbb { P } _ { \pmb { X } } } \left[ D ( f _ { \phi } ( \pmb { X } ) , \mathbb { P } _ { \pmb { Y } | \pmb { X } } ) \right] . } \end{array}\tag{2}
$$

The above states that optimizing a scoring rule is equivalent to minimizing the score divergence between the predictive model and the true conditional distribution. As mentioned before, this setting includes the commonly used strictly proper log score $S _ { \mathrm { l o g } } ( P , \pmb { y } ) : = - \log p ( \pmb { y } )$ , or NLL. In this case, the corresponding divergence is the KL divergence, recovering the cross-entropy minimization objective.

## 2.2 PARAMETERIZING A MULTIVARIATE GAUSSIAN

Several works focus on ways to approximate a covariance matrix such that it can be parameterized and optimized using a neural network. We revisit three important and commonly used parameterizations here:

The simplest choice is just a diagonal covariance Σ (Diag), which disregards any correlation structure, with $\pmb { \Sigma } = \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , \dots , \sigma _ { d } ^ { 2 } )$ , and $\bar { \sigma _ { k } ^ { 2 } } > 0$ . Since this corresponds to d independent univariate Gaussians, one per dimension, the approach is equivalent to the mean-variance networks proposed by Nix and Weigend [10].

On the other hand, one can consider the Cholesky decomposition (Chol), which decomposes any $\Sigma \succ 0$ into $\boldsymbol { \Sigma } = \boldsymbol { L } \boldsymbol { L } ^ { \top }$ , where $\pmb { L } \in \mathbb { R } ^ { d \times d }$ is lower triangular with a positive diagonal. While this has been successfully combined with neural networks [17, 18] and offers full expressivity of Σ, it comes at a very high computational cost and is infeasible in high dimensions.

A middle ground is given by the low-rank-plus-diagonal approximation (LorD), given as $\pmb { \Sigma } = \pmb { D } + \pmb { U } \pmb { U } ^ { \top }$ , where $\\\bar { D } ^ { \bullet } = \mathrm { d i a g } ( \sigma _ { 1 } ^ { 2 } , \ldots , \sigma _ { d } ^ { 2 } ) \ \in \ \mathbb { R } _ { > 0 } ^ { d \times d }$ and $U \in \mathbb { R } ^ { d \times r }$ with $1 \leq r \ll d$ . While often used in combination with variational inference [19, 20], it has also been used in conditional prediction settings [4, 21].

Table 1: Cost of covariance parameterizations: number of covariance parameters, one log-score evaluation, one sample, and estimation of one covariance entry $\hat { \Sigma } _ { i j }$
<table><tr><td></td><td>Params</td><td>Score eval.</td><td>Sampling</td><td> $\hat { \Sigma } _ { i j }$ </td></tr><tr><td>Chol</td><td> $d ( d + 1 ) / 2$ </td><td>O(d2)</td><td> $\mathcal { O } ( d ^ { 2 } )$ </td><td>O(d)</td></tr><tr><td>Diag</td><td>d</td><td>O(d)</td><td> $\mathcal { O } ( d )$ </td><td>O(1)</td></tr><tr><td>LorD</td><td> $d + d r$ </td><td> $\mathcal { O } ( d r ^ { 2 } + r ^ { 3 } )$ </td><td>O(dr)</td><td>O(r)</td></tr><tr><td>SCORE (Ours)</td><td>2d</td><td> ${ \mathcal { O } } ( d \log d )$ </td><td>O(d log d)</td><td>O(d)</td></tr></table>

Table 1 summarizes the computational cost of these methods and of our proposed method SCORE, introduced in Section 3. Further approximations exist, which we do not consider here. A noteworthy one is the Kronecker product, common in curvature approximation [22, 23]. It factorizes a structured domain, for example, spatiotemporal, into its coordinates, but each factor again requires one of the above approximations, which is the reason we do not consider it here.

## 2.3 TRAINING WITH NEGATIVE LOG-LIKELIHOOD

Regardless of the parameterization, previous work focuses almost exclusively on optimizing or adapting $S _ { \mathrm { l o g : } }$ , which admits a closed form

$$
S _ { \log } ( P , { \pmb y } ) = \frac d 2 \log ( 2 \pi ) + \frac 1 2 \log \operatorname * { d e t } { \pmb \Sigma } _ { P } + \frac 1 2 M ,\tag{3}
$$

with $\begin{array} { r } { M : = ( { \pmb y } - { \pmb \mu } _ { P } ) ^ { \top } { \pmb \Sigma } _ { P } ^ { - 1 } \left( { \pmb y } - { \pmb \mu } _ { P } \right) } \end{array}$ . However, already in the univariate case, the corresponding optimization problem is ill-conditioned and leads to diverging gradients [12, 24]. In the multivariate case, the log score becomes even more unstable.

First, note that for existence $S _ { \mathrm { l o g } }$ requires rank $( \pmb { \Sigma } ) = d .$ However, in high dimensions, we can expect rank $( \Sigma ) < d ,$ since the true distribution might live on a proper subspace. In that case, while the Gaussian distribution can still be defined meaningfully, (3) is undefined. This can occur along any rank-deficient direction, not only in the marginals. Furthermore, $S _ { \mathrm { l o g } }$ suffers from the additional drawback that it admits unbounded gradients and therefore is not Lipschitz continuous, which leads to unstable training with neural networks.

Lemma 2.1 (Unbounded gradients of $S _ { \mathrm { l o g } } )$ . Consider a predictive Gaussian $\scriptstyle { \mathcal { N } } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ , with Cholesky decomposition $\pmb { \Sigma } = \pmb { L } \pmb { L } ^ { \top }$ , and fixed $\pmb { y } \in \mathbb { R } ^ { d }$ . Then $\nabla _ { \mu } S _ { \mathrm { l o g } }$ and $\nabla _ { L } S _ { \mathrm { l o g } }$ are unbounded over $( \mu , L )$ as $\lambda _ { \operatorname* { m i n } } ( \Sigma ) \to 0 ,$ and $S _ { \mathrm { l o g } } ~ i s$ not (globally) Lipschitz continuous in $( \mu , L )$

The proof is given in Appendix B. In general, even for $\Sigma = \sigma ^ { 2 } I , S _ { \mathrm { l o g } }$ is not Lipschitz, since the univariate log score is not Lipschitz in σ. While one can circumvent these issues, for example, by adding εI with $\varepsilon > 0$ , this still leads to numerical issues for small ε and to an overestimation of the aleatoric uncertainty in the predictive distribution for large ε, as highlighted in Figure 7 and Appendix F.

## 3 METHODOLOGY

## 3.1 ROBUST LOSS FUNCTIONS

To circumvent the caveats of the log score, we propose to instead use kernel scores as loss functions, which offer a more robust alternative. They have been extensively used in forecast evaluation [13] and more recently as neural network loss functions [25, 26]. Let $k \colon  { \mathbb { R } } ^ { d } \times  { \mathbb { R } } ^ { d } \to$ R be a positive definite kernel. The kernel score of a distribution $P \in \mathcal { P } ( \mathcal { V } )$ evaluated at observation $\mathbf { \boldsymbol { y } } \in \mathcal { V }$

$$
\begin{array} { r } { S _ { k } ( P , \pmb { y } ) = \frac { 1 } { 2 } \left( \mathbb { E } _ { z , z ^ { \prime } \sim P } \left[ k ( z , z ^ { \prime } ) \right] + c \right) - \mathbb { E } _ { z \sim P } \left[ k ( \pmb { y } , z ) \right] , } \end{array}
$$

with $c = k ( \pmb { y } , \pmb { y } )$ , is strictly proper when k is characteristic [14]. Kernel scores are directly related to maximum mean discrepancies (MMDs) [15], in fact, the divergence in (2) becomes $\scriptstyle { \frac { 1 } { 2 } } \mathrm { M M D } ^ { 2 }$ . Here, we focus on the Gaussian kernel $\begin{array} { r } { k ( \pmb { y } , \pmb { y } ^ { \prime } ) \overset {  } { = } \exp \bigl ( { - \frac { 1 } { \gamma ^ { 2 } } \| \pmb { y } - \pmb { y } ^ { \prime } \| _ { 2 } ^ { 2 } } \bigr ) } \end{array}$ with bandwidth $\gamma > 0$ . For a Gaussian predictive, it admits a closed form expression, as stated in Appendix B.2, and it is bounded, which is the key property behind the following result.

![](images/775d108cfe5230fb24ef1c3852ee00406a0d031a511d33eddf54381fa1edae6d.jpg)  
Figure 2: Toy example. We draw a signal $\pmb { y } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { \mathrm { p r } } )$ with circulant $\Sigma _ { \mathrm { p r } }$ and impose a cut-off on the eigenvalues, $\mathrm { i . e . , } \lambda _ { i } > 0 \mathrm { i f f } | i | \le B$ . Now, we observe a noisy subsample $\pmb { x } = \pmb { S } \pmb { y } + \pmb { \eta }$ with observational noise $\pmb { \eta } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } I _ { n } )$ where $S \in \mathbb { R } ^ { n \times d }$ selects every $( d / n )$ -th coordinate, and train a neural network to reconstruct y given x. The posterior has a closed form with rank $( \Sigma _ { \mathrm { p o s t } } ) = \operatorname* { m i n } ( 2 B + 1 , d )$ By varying the band limit B, we interpolate between a lowrank and a higher-rank target covariance matrix. The figure shows the estimated mean (colored dashed line) and standard deviation (shaded colored area), as well as the true posterior standard deviation (shaded grey area). In the lowrank case, $S _ { \mathrm { l o g } }$ fails to recover the posterior, while $S _ { k }$ does. More details are given in Appendix F.

Theorem 3.1 (Lipschitz continuity). Under the setting of Lemma 2.1, the gradients of $S _ { k }$ are bounded and the score is globally Lipschitz continuous in $( \mu , L )$

The theorem is proved in Appendix B. It highlights that the kernel score is more stable for gradient-based optimization. The toy example in Figure 2 illustrates this further on a rank-deficient target.

## 3.2 SPECTRAL PARAMETERIZATION

Apart from the loss instability, we also address the problem of approximating the high-dimensional predictive covariance matrix. Here, we propose to model the marginals and correlation structure separately, optimizing the latter in spectral space. This is motivated by the fact that a Gaussian is closed under linear transformations. In particular, let $\boldsymbol { Y } \ \sim \ \mathcal { N } ( \mu , \Sigma )$ and $B \in \mathbb { C } ^ { d \times d }$ . Then, $\tilde { Y } = B Y \sim \mathcal { N } ( B { \boldsymbol { \mu } } , B \Sigma B ^ { * } )$ . Further, the kernel score transforms consistently under such maps and is invariant for unitary B.

Proposition 3.2 (Kernel score invariance). Let $k ( \pmb { y } , \pmb { y } ^ { \prime } ) =$ $\kappa ( \| \pmb { y } - \pmb { y } ^ { \prime } \| _ { 2 } ^ { 2 } )$ be isotropic, with $\| \cdot \| _ { 2 }$ the Hermitian norm on $\mathbb { C } ^ { d }$ , and let $B _ { \# } P$ denote the pushforward of $P = \mathcal { N } ( { \pmb \mu } , { \pmb \Sigma } )$ under $\boldsymbol { B } \in \mathbb { C } ^ { \ddot { d } \times d }$

(i) If B is invertible, then $S _ { k } ( B _ { \# } P , B y ) = S _ { k _ { B } } ( P , y )$ with $\begin{array} { r l r } { k _ { B } ( y , y ^ { \prime } ) } & { { } : = } & { \kappa \big ( \lVert y - \ y ^ { \prime } \rVert _ { B ^ { * } B } \big ) } \end{array}$ , where $\| \pmb { v } \| _ { B ^ { * } B } ^ { 2 } = \pmb { v } ^ { * } B ^ { * } B \pmb { v }$ . Hence $S _ { k _ { B } }$ is strictly proper whenever $S _ { k }$ is.

(ii) If B is unitary, then $k _ { B } = k$ and $S _ { k } ( B _ { \# } P , B \pmb { y } ) =$ $S _ { k } ( P , \pmb { y } )$

We provide a proof in Appendix C.1. Part (ii) is what allows us to fit the correlation structure in the spectral domain without changing the objective.

In this work, we mainly consider the DFT, for which we write $B = F$ in the following, since it enables fast computation. Let $\pmb { F } \in \mathbb { C } ^ { d \times d }$ denote the unitary DFT matrix with entries $F _ { j k } = d ^ { - 1 / 2 } e ^ { - 2 \pi i j k / d }$ , so that $\pmb { F } ^ { - 1 } = \pmb { F } ^ { * }$ . Throughout, a hat, as in $\hat { \Sigma }$ , marks quantities in the transformed coordinates $\hat { z } = B z$ . Consider a diagonal covariance matrix in Fourier space, $\hat { \pmb { \Sigma } } = \mathrm { d i a g } ( s _ { 1 } , \ldots , s _ { d } )$ . Its counterpart in physical space $\Sigma = F ^ { * } \hat { \Sigma } F$ is circulant, hence the covariance matrix of a stationary process on the periodic grid, and $\left\{ s _ { k } \right\}$ is its power spectral density by the Wiener–Khinchin theorem. It is dense in physical space, where relevant operations would cost $\mathcal { O } ( d ^ { 3 } )$ . Yet they reduce to ${ \mathcal { O } } ( d \log d )$ via the FFT.

## 3.3 SPECTRAL CORRELATION ESTIMATION

So far we have shown how unitary transforms can be used to approximate a dense covariance matrix with few parameters. However, this trades off dependence structure against a correct marginal fit, since, for example, a circulant matrix implies equal marginal variances in every dimension. To circumvent this issue, and to further enhance expressivity, we propose SCORE, a parameterization that separates the marginals from the correlation structure. Throughout, we suppress the dependence on x in the notation.

Denote corr $\left( \Sigma \right) : = \mathrm { d i a g } ( \Sigma ) ^ { - 1 / 2 } \Sigma ~ \mathrm { d i a g } ( \Sigma ) ^ { - 1 / 2 }$ for the correlation matrix of Σ. Consider the Gaussian target $Q = \mathcal { N } ( \mu _ { Q } , \Sigma _ { Q } )$ and write $R _ { Q } : = \mathrm { c o r r } ( \Sigma _ { Q } )$ and $D _ { Q } : =$ dia $\begin{array} { r } { \mathstrut } \\ { \mathstrut } \\ { \mathstrut } \mathstrut  \end{array} ( \Sigma _ { Q } ) ^ { 1 / 2 } .$ Then its covariance factors into marginal scales and correlation as

$$
\Sigma _ { Q } = D _ { Q } R _ { Q } D _ { Q } .\tag{4}
$$

The factors are complementary: $D _ { Q }$ holds the marginal scales and $R _ { Q }$ the correlation between coordinates. We mirror this factorization in the model.

Fix a unitary matrix $B \in \mathbb { C } ^ { d \times d }$ associated with the chosen transform and refer to the coordinates $B y , y \in \mathbb { R } ^ { d }$ as the B-domain. A prediction for a given input is specified by a triple $( \mu , \tau , \hat { \Sigma } )$ , with mean vector $\pmb { \mu } ~ \in ~ \mathbb { R } ^ { d }$ marginal variances $\tau \in \mathbb { R } _ { > 0 } ^ { d }$ collected in the diagonal scale matrix $D _ { \tau } : = \mathrm { d i a g } ( \sqrt { \tau } )$ , and a Hermitian positive definite matrix $\hat { \Sigma }$ in the B-domain, which we call the core of the parameterization. The predictive distribution is $\mathcal { N } ( \pmb { \mu } , D _ { \tau } B ^ { * } \hat { \Sigma } B D _ { \tau } )$ with $D _ { \tau }$ intended to carry the marginals and the transformed core $B ^ { * } \hat { \Sigma } B$ the correlation of (4). Restricting the core is what keeps the parameterization cheap, and we write $\mathcal { S } \subseteq \{ \hat { \Sigma } \in \mathbb { C } ^ { \hat { d } \times d } : \hat { \Sigma } = \hat { \Sigma } ^ { * } \succ 0 \}$ for the admissible class, e.g. the diagonal or low-rank-plusdiagonal matrices. Our quantitative results mostly concern the case where $B = F$ is the DFT and S the class of diagonal cores.

Although (4) suggests splitting the learning of $\Sigma _ { Q }$ into marginals and correlation, the above parameterization does not enforce this. The transformed core is in general not a correlation. Its diagonal entries $\Delta _ { i } ( \hat { \Sigma } ) : = \big ( B ^ { * } \hat { \Sigma } B \big ) _ { i i } { : }$ , the implied marginal variances $\pmb { \Delta } ( \hat { \pmb { \Sigma } } ) \in \mathbb { R } _ { > 0 } ^ { d } .$ need not equal one, so that $( \Sigma _ { P } ) _ { i i } = \tau _ { i } \Delta _ { i } ( \hat { \Sigma } )$ instead of $\tau _ { i }$

Taking the correlation. The remedy is to pass the transformed core through the same operator that defines the target correlation in (4), so that the parameterization turns into

$$
\Sigma _ { P } : = D _ { \tau } \mathrm { c o r r } \big ( B ^ { * } \hat { \Sigma } B \big ) D _ { \tau } .\tag{5}
$$

By definition, this only requires computing $\Delta ( \hat { \Sigma } )$ , which for a diagonal core $\hat { \Sigma } = \mathrm { d i a g } ( s )$ with $s \in \mathbb { R } _ { > 0 } ^ { d }$ is the sum $\begin{array} { r } { \Delta _ { i } = \sum _ { k } s _ { k } | B _ { k i } | ^ { 2 } } \end{array}$ . Replacing the transformed core with its correlation pins the marginals exactly.

Lemma 3.3 (Correlation core). Let $\boldsymbol { B } \in \mathbb { C } ^ { d \times d }$ be unitary, $\hat { \Sigma } \succ 0$ Hermitian and $\tau \in \mathbb { R } _ { > 0 } ^ { d } .$ . Then the covariance (5) satisfies $( \Sigma _ { P } ) _ { i i } = \tau _ { i } f o r e { \nu } e r y i = 1 , \dots , d .$

The proof is in Appendix C.1. For the DFT and diagonal cores, the parameterization simplifies.

Fourier transform and diagonal core. Let $B = F$ be the DFT and ${ \hat { \Sigma } } = \operatorname { d i a g } ( s )$ with $s \in \mathbb { R } _ { > 0 } ^ { d }$ and $s _ { k } = s _ { d - k } ,$ which makes $F ^ { * } \deg ( s ) F$ real; cf. Appendix C.2.1. This matrix is circulant, so its diagonal is the constant $\bar { s } =$ ${ \frac { 1 } { d } } \sum _ { k } s _ { k }$ . Normalizing therefore only rescales the spectrum, and the prediction becomes

$$
\Sigma _ { P } = D _ { \tau } F ^ { * } \mathrm { d i a g } ( s / { \bar { s } } ) F D _ { \tau } .
$$

We therefore let the network emit logits $\ell \in \mathbb { R } ^ { d }$ and set $s = d \operatorname { s o f t m a x } ( \ell )$ , so that $\bar { s } = 1$ and the transformed core is a correlation matrix already, i.e.,

$$
\mathrm { c o r r } \big ( { \cal F } ^ { * } \mathrm { d i a g } ( s ) { \cal F } \big ) \ = \ { \cal F } ^ { * } \mathrm { d i a g } ( s ) { \cal F } .
$$

This normalization also removes a redundancy. Since corr is invariant under $\hat { \Sigma } \mapsto c \hat { \Sigma }$ for every $c > 0$ , the core is identified only up to a positive factor, the scale gauge. For the DFT and a diagonal core, $\bar { s } = 1$ fixes it, so that each correlation matrix corresponds to a single spectrum; compare Appendix C.2.1.

By Lemma 3.3, the marginal variances equal τ for every correlation core. SCORE exploits this with a staged objective that fits the marginals first and then the correlation with the marginals frozen. Both stages range over maps ${ \pmb x } \mapsto ( \mu , \tau , \hat { \Sigma } )$ that a network realizes.

SCORE. Let $S _ { 1 }$ be a univariate and $S _ { 2 }$ a multivariate scoring rule. The first stage is

$$
( \mu ^ { \star } , \tau ^ { \star } ) = \underset { \mu , \tau } { \arg \operatorname* { m i n } } \mathbb { E } \sum _ { i = 1 } ^ { d } S _ { 1 } \big ( \mathcal { N } ( \mu _ { i } , \tau _ { i } ) , Y _ { i } \big ) ,\tag{6}
$$

fitting the marginal distributions only. Write $D ^ { \star } : = D _ { \tau ^ { \star } }$ m $\smash { \vdots } = ( D ^ { \star } ) ^ { - 1 } \mu ^ { \star }$ and $Z : = ( D ^ { \star } ) ^ { - 1 } Y$ for the standardized target. The second stage holds the marginals fixed and fits the core over the admissible class $s .$

$$
\begin{array} { r } { \hat { \Sigma } ^ { \star } = \underset { \hat { \Sigma } \in \cal S } { \arg \operatorname* { m i n } } \mathbb { E } { S _ { 2 } } \Big ( { \cal N } \big ( m , \mathrm { c o r r } \big ( { \cal B } ^ { \ast } \hat { \Sigma } { \cal B } \big ) \big ) , Z \Big ) . } \end{array}\tag{7}
$$

Dividing out the marginal scales removes the percoordinate scale learned in the first stage, so with $\mathcal { M } : =$ $\left\{ \operatorname { c o r r } \left( B ^ { * } \hat { \Sigma } B \right) \quad : \quad \hat { \Sigma } \in \mathcal { S } \right\}$ the prediction ranges exactly over $\{ \mathcal { N } ( \pmb { m } , R ) : R \in \pmb { \mathscr { M } } \}$ . Writing $R ^ { \star } : =$ $\operatorname { c o r r } ( B ^ { * } \hat { \Sigma } ^ { \star } B ) \in \mathcal { M }$ for the deployed correlation, the prediction is $\mathcal { N } ( \mu ^ { \star } , D ^ { \star } R ^ { \star } D ^ { \star } )$ , whose marginal variances are $\tau ^ { \star }$ by Lemma 3.3. The overall workflow of SCORE is illustrated in Figure 1. In practice, we take $S _ { 1 }$ to be the continuous ranked probability score (CRPS) [13], a parameter-free kernel score, and $S _ { 2 }$ the Gaussian kernel score instead of the log score, motivated by Lemma 2.1 and Theorem 3.1. After a single FFT, it evaluates in closed form at ${ \mathcal { O } } ( d ) ( { \mathrm { A p - } }$ pendix C.2.2). For its bandwidth, the median heuristic [15], resolves to $\gamma ^ { 2 } = 4 d \left( \mathrm { { L e m m a } } \mathrm { { C . 1 } } \right)$ , which we average across different scalings. See Appendix C.2.1 for implementation details.

Remark 3.1 (Why a staged fit). SCORE is a deliberate design choice,for several reasons. Freezing the marginals keeps them exact even when the correlation class is misspecified (Proposition 3.4(i)). A joint fit would achieve a betterjoint score only by distorting the marginals. For the kernel score, a joint fit is also not available at $\mathcal O ( d )$ . The closed form (Appendix C.2.2) requires a circulant, which $\begin{array} { r } { \Sigma _ { P } \ = \ \mathbf { D } _ { \tau } \mathbf { F } ^ { * } \mathrm { d i a g } ( \mathbf { s } ) \mathbf { F } \mathbf { D } _ { \tau } } \end{array}$ is not unless τ is constant. Nor can (7) be optimizedjointly. The marginals enter only through the residual $\mathbf { D } _ { \tau } ^ { - 1 } ( \mathbf { y } - \pmb { \mu } )$ , so it degenerates as $\tau  \infty .$

A sufficiently rich model attains the minimum of (6) and (7) at every input, as the following result describes.

Proposition 3.4 (Population optimum). At a fixed input x, let the target be $Q = \mathcal { N } ( \mu _ { Q } , \Sigma _ { Q } )$ with $\Sigma _ { Q } \succ 0 ,$ , factored as $\Sigma _ { Q } = D _ { Q } R _ { Q } D _ { Q } .$ . Let $S _ { 1 }$ be strictly proper on ${ \mathcal { P } } ( \mathbb { R } )$

let $S _ { 2 }$ be strictly proper and translation invariant $( e . g . , a$ kernel score with isotropic kernel), and write D for the divergence associated with $S _ { 2 }$ . Then

(i) The first stage fits $\pmb { \mu } ^ { \star } = \pmb { \mu } _ { Q }$ and $\tau _ { i } ^ { \star } = ( \pmb { \Sigma } _ { Q } ) _ { i i }$ for $i = 1 , \ldots , d ,$ hence $D ^ { \star } = D _ { Q }$

(ii) The second stage fits the correlation alone, $R ^ { \star } \in$ arg min<sub>R∈M</sub> $D ( \mathcal { N } ( \mathbf { 0 } , R ) , \mathcal { N } ( \mathbf { 0 } , R _ { Q } ) )$ , and the $d e \mathrm { . }$ ployed prediction is $\mathcal { N } ( \mu _ { Q } , D _ { Q } R ^ { \star } \bar { D _ { Q } } )$

The proof is given in Appendix C.1. Let $\varepsilon _ { \mathrm { a p p } } ( { \pmb x } ) \ : =$ $D ( \tilde { \mathcal { N } } ( \mathbf { 0 } , R ^ { \star } ( x \bar { \bf { \rangle } } ) , \mathcal { N } ( \mathbf { 0 } , R _ { Q } \bar { \bf { ( } } { \bf { x } } ) ) )$ denote the approximation error at input x and $\bar { \varepsilon } _ { \mathrm { a p p } } : = \mathbb { E } _ { X } [ \varepsilon _ { \mathrm { a p p } } ( X ) ]$ its average. We provide explicit bounds in Appendix D.

Finite-sample error. Proposition 3.4 is a population statement. At finite sample size, both stages are trained on empirical counterparts of (6) and (7), the second on residuals standardized by the estimated marginals, and one might worry that this amplifies their error. The following shows that it does not.

Theorem 3.5 (SCORE is PAC, informal). Train SCORE with the CRPS in the first stage and the Gaussian kernel score in the second, on N and M i.i.d. samples from P, within the network class ofAssumption E.2, and let $P _ { N , M }$ be the deployed prediction. Under Assumption E.1,for every $\varepsilon > 0$ and $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ , if $\begin{array} { r } { N \in \tilde { \mathcal { O } } ( \varepsilon ^ { - \frac { 1 } { 4 } } \log \frac { 1 } { \delta } ) } \end{array}$ and $M \ \in$ $\tilde { \mathcal { O } } ( \varepsilon ^ { - 2 } \log \frac { 1 } { \delta } )$ , then with probability at least $1 - \delta$

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { X } } \Big [ \frac { 1 } { 2 } \mathrm { M M D } ^ { 2 } \big ( Q ( \pmb { X } ) , P _ { N , M } ( \pmb { X } ) \big ) \Big ] \leq c \bar { \varepsilon } _ { \mathrm { a p p } } + \varepsilon } \end{array}
$$

for a constant $c > 0 ,$ , where $\tilde { \mathcal { O } }$ hides logarithmic factors.

Formal statement and proof are given in Appendix E.

Beyond Fourier and diagonal cores. The parameterization (5), Lemma 3.3 and Proposition 3.4 hold for any unitary B and admissible class S. In particular, the first-stage marginals stay exact regardless of the chosen core and second-stage objective. For the log score, the closed forms even extend to arbitrary B and low-rank cores, which is not the case for the Gaussian kernel score, as shown in Appendix C.2.2.

## 4 EXPERIMENTS

Table 2: Overview of tasks.
<table><tr><td>Task</td><td>Dataset</td><td>Backbone</td><td>Prior work</td></tr><tr><td>Time series</td><td>[27]</td><td>PatchTST [28]</td><td>[4, 29, 30]</td></tr><tr><td>Depth est.</td><td>NYU [31]</td><td>DepthAnything [32]</td><td>[5]</td></tr><tr><td>Gridded temp.</td><td>ERA5 [33]</td><td>U-Cast [34]</td><td>[35, 36]</td></tr><tr><td>Station temp.</td><td>EUPPBench [37]</td><td>GNN [38]</td><td>[38]</td></tr></table>

Datasets & baselines. We consider four tasks where a Gaussian predictive is a reasonable assumption and has been used in prior work, summarized in Table 2. For each task, we fix a backbone and only replace the output layer with the respective covariance parameterization (Appendix C.2.1), such that the comparison isolates the Gaussian representation and all methods are comparable. In all tasks SCORE uses the Fourier transform, except in the station-based task, which uses a graph neural network and models the covariance via the graph Fourier transform, illustrating how SCORE extends beyond the DFT. Across all tasks, we compare SCORE against the same baselines: a deterministic model (Det), a diagonal Gaussian (Diag), and the low-rankplus-diagonal parameterization (LorD) [4, 29], the most common scalable alternative. We set $r = 1$ , which matches the output-layer parameter count of SCORE (larger r in $\mathsf { A p - }$ pendix H.4). A sample-based generative model (SB) with noise injection in the hidden layers and trained with the Gaussian kernel score (KS) [25, 26] serves as a nonparametric, non-Gaussian baseline. Det is trained with the mean squared error (MSE), SB with KS, and SCORE with the CRPS in the first and the KS in the second stage. Diag and LorD are trained twice, with the NLL as the standard choice and with the KS as alternative, which separates the effect of the loss from that of the covariance parameterization in Table 3. In Appendix H we also provide results for a model with a full Cholesky factor (Chol), as a fully expressive baseline. However, since this is only feasible for a few tasks, it is omitted in the main paper.

Evaluation protocol. All models are compared on a common set of metrics, mostly proper scoring rules. Scores are test-set and five-seed averages. The MSE and the CRPS measure mean and marginal fit, respectively. The NLL, the energy score (ES) and the KS are strictly proper for the joint distribution and the variogram score (VS) is proper, shift-invariant, and targets the dependence structure. Note that since the NLL, CRPS, and KS serve as training objectives, one would expect models using these as a loss to perform better in terms of the corresponding metric. Therefore, the performance should be considered with respect to all selected metrics to obtain an overall picture. Sample-based metrics use an ensemble of size $M \ = \ 1 0 0$ ; NLL and MSE are evaluated in closed form (cf. Appendix G.1). All implementations and reproducible experiments are available in a public repository at https://github.com/cbuelt/score

Results. Table 3 shows the average rank of each method on each metric. SCORE attains the best average rank (2.5) and the best rank on MSE, CRPS, ES and VS. The strongest baseline, LorD trained with the KS (3.1), is outranked by SCORE on these metrics at equal parameter count. The margin is largest on the VS, which targets the dependence structure, and on the CRPS, which SCORE inherits from its first stage. Unsurprisingly, the NLL is won by the model trained on it: LorD-NLL. The same holds for KS, but here SCORE ranks similarly quite (±0.2).

![](images/2819a534a6b4a88c7cc073c54e9c70ddbde7cd4ce9551d5d23704370435d5c48.jpg)

![](images/93b9edc733acfc9a6298e12d36351933de0394fc4e6f44753dee895fdb07d3ee.jpg)  
Figure 3: SCORE predictions. Left: Mean prediction and standard deviation of SCORE on the univariate ETTh2 dataset and a comparison with LorD regarding similarity to the empirical correlation structure. Right: observed temperature across stations and predictions and correlations using SCORE and LorD.

Table 3: Average rank of the predictive methods. Ranks are computed per task and seed, and then averaged over the eleven datasets. Best rank in bold.
<table><tr><td>Core</td><td>Objective</td><td>MSE</td><td>CRPS</td><td>ES</td><td>VS</td><td>KS</td><td>NLL/d</td><td>Avg.</td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td>4.0</td><td>6.5</td><td>6.6</td><td>5.1</td><td>5.7</td><td>一</td><td>5.6</td></tr><tr><td>Diag</td><td>NLL</td><td>4.1</td><td>2.8</td><td>3.6</td><td>4.3</td><td>4.2</td><td>2.9</td><td>3.7</td></tr><tr><td>Diag</td><td>KS</td><td>3.2</td><td>3.5</td><td>3.0</td><td>4.1</td><td>3.2</td><td>4.0</td><td>3.5</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>6.1</td><td>5.6</td><td>5.5</td><td>5.0</td><td>5.8</td><td>2.1</td><td>5.0</td></tr><tr><td>LorD r=1</td><td>KS</td><td>3.4</td><td>3.4</td><td>2.6</td><td>3.4</td><td>2.6</td><td>3.2</td><td>3.1</td></tr><tr><td>SB</td><td>KS</td><td>4.2</td><td>4.2</td><td>4.3</td><td>4.3</td><td>3.7</td><td></td><td>4.1</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>3.0</td><td>2.1</td><td>2.4</td><td>1.8</td><td>2.8</td><td>2.8</td><td>2.5</td></tr></table>

Table 4: Number of wins when training the same predictive model with the Gaussian kernel score instead of the NLL. Bold: $p < 0 . 0 5$ under the clustered sign-flip test, Holm-Bonferroni corrected over the four metrics.
<table><tr><td>n</td><td>MSE</td><td>CRPS</td><td>ES</td><td>VS</td></tr><tr><td>167</td><td>150</td><td>140</td><td>156</td><td>139</td></tr></table>

We next separate the two contributions. Table 4 counts, over all data sets and parameterizations, how often training with the kernel score beats the NLL on the evaluation metrics that do not serve as an objective. The kernel score wins the large majority, significantly under a data-set-clustered sign-flip test. Among the KS-trained models, SCORE attains the best average rank on every metric, most clearly on the VS, which no method trains on. This gap is a contribution of the parameterization (Appendix H.4). This is further highlighted in Figure 4, which analyzes all methods trained with the KS with respect to relative improvement in performance and computational cost against Diag. As expected, SCORE is slower than Diag, since the latter is linear in computational complexity. However, it is the fastest among the baselines that model dependence and admits the best performance, with clear margins in all metrics except the MSE, where performance is similar to LorD.

![](images/e8d97d569791b5ec0dc01f5cff696905429c5979b5c9cf6dde3e3647fc47e3c5.jpg)  
Figure 4: Performance improvement (difference in nats per coordinate for NLL, percentage improvement otherwise) vs. increase in compute, relative to the Diag (median over seeds and data sets).

Further, we analyze the probabilistic calibration of the methods with respect to univariate marginals, as well as multivariate location and scale, as shown in Figure 5. SCORE exhibits the best calibrated predictions overall, except on NYU (green), where the depth maps violate the stationarity assumption and SCORE falls behind (Appendix H). LorD shows good marginal calibration, while SB fails especially on the scale calibration.

Detailed results for each task, as well as regarding the runtime, calibration, and additional ablations, are available in Appendix H. Overall, SCORE is the best-ranked method on every metric, with a large margin on the VS, indicating a superior approximation of the dependence structure. Both our design choices contribute to the performance: the KS improves upon the NLL significantly, while with the same loss, SCORE beats established approaches. SCORE achieves this with near-linear cost, due to the FFT, making it a good choice for a predictive Gaussian model in high dimensions. We provide further ablations of the rank r for the LorD method and the impact of the two-stage training process in Appendix H.4.

![](images/accf5c4220e63794ba0a92bc1a438c21f8f64c5af3fbdab8b72a7de067b8d61d.jpg)  
Figure 5: Different calibration measures for LorD, SB and SCORE, using the kernel score across all datasets. The dashed diagonal line indicates optimal calibration.

## 5 RELATED WORK

Neural networks predicting Gaussian distributions date back to mean-variance networks [10], which output a univariate Gaussian trained with the negative log-likelihood (NLL); for multiple variables, this amounts to the diagonal covariance parameterization. This setup is known to train poorly because the NLL gradient of the mean is scaled by the predicted precision, motivating remedies such as reweighted losses, decoupled mean and variance heads, or natural parameterizations [12, 24, 39, 40]. These difficulties persist and are amplified for more expressive covariance structures. Full covariances were already learned by Williams [17], who train a network to output the Cholesky factor by likelihood maximization. Russell and Reale [18] compare this parameterization to the diagonal one across deep learning benchmarks, and Muschinski et al. [36] use it for distributional regression in weather forecasting. The quadratic cost can be reduced via banded Cholesky factors [3] or by training a Taylor expansion of the covariance [41]. Lowrank-plus-diagonal covariances originate in variational inference [19, 20] but also appear in predictive settings such as uncertainty prediction in images [21] and time series modeling with copulas [4] or Kronecker factorizations [42]. Kronecker-factored matrices themselves stem from curvature estimation [22, 23] and variational inference [43, 44] and have only recently been used as direct network outputs [29].

Gaussian processes (GPs) offer a conceptually different route to input-dependent multivariate Gaussian predictions: mean and covariance are induced by a kernel whose hyperparameters are fit by marginal likelihood optimization [45, 46, 47], and deep kernel learning parameterizes the kernel map by a neural network [48]. Exact GP inference costs $\mathcal { O } ( d ^ { \bar { 2 } } )$ memory and $\mathcal { O } ( d ^ { 3 } )$ time, which is prohibitive for gridded targets and is typically addressed by inducing points or low-rank approximations [49, 50, 51]. We instead follow the distributional regression paradigm [1, 52], where the network parameterizes the conditional predictive law directly and describes aleatoric dependence in the target rather than posterior uncertainty about a latent function.

## 6 CONCLUSION

We introduced SCORE, a method for learning highdimensional structured Gaussian distributions with neural networks. Training with kernel scores instead of the negative log-likelihood circumvents the instability of likelihoodbased optimization. In addition, we proposed to parameterize the covariance matrix in a spectral unitary basis, which, owing to proven properties of the kernel score, yields a dense covariance in physical space at low computational cost. A staged objective that separates learning the marginals from learning the correlation structure keeps training efficient, while recovering the true marginals and an approximation of the true correlations. In our experiments, SCORE outperforms established baselines across a range of metrics, is the best calibrated among the compared methods, and offers the best trade-off between compute and performance.

Limitations and future work. The main limitation of SCORE is the circulant covariance structure it implies, which may be unreasonable in some applications. While we quantify the resulting approximation error, and the method is deliberately designed as a fast yet simple approximation, certain predictive tasks will require moving beyond this structure. This points to several directions for future work. On the modeling side, richer parameterizations in the transformed domain, such as low-rank or tridiagonal instead of diagonal, could increase expressivity while keeping computational cost low, and other unitary transforms, such as wavelets, could be an interesting object of study. Moreover, SCORE extends naturally to Gaussian mixtures and other elliptical distributions, such as the multivariate t-distribution, which would broaden its applicability to non-Gaussian and heavy-tailed data. On the theoretical side, sharpening the PAC bounds and deriving an optimal split between the two stages of the objective would further improve the usability of our method.

## Acknowledgements

C. Bülte and G. Kutyniok acknowledge support by the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Research, Technology and Space and the German Research

Foundation under the grant DFG-SPP-2298.

E. Partow and G. Kutyniok acknowledge support by the German Research Foundation (Deutsche Forschungsgemeinschaft, DFG) – GRK 3081 – Project number 534429653.

G. Kutyniok also acknowledges support by the gAIn project, which is funded by the Bavarian Ministry of Science and the Arts (StMWK Bayern) and the Saxon Ministry for Science, Culture and Tourism (SMWK Sachsen). Furthermore, G. Kutyniok is supported by LMUexcellent, funded by the Federal Ministry of Education and Research (BMBF) and the Free State of Bavaria under the Excellence Strategy of the Federal Government and the Länder as well as by the Hightech Agenda Bavaria.

The authors acknowledge the computational and data resources provided by the Leibniz Supercomputing Centre (www.lrz.de).

## References

[1] Tilmann Gneiting and Matthias Katzfuss. Probabilistic forecasting. Annual Review of Statistics and Its Application, 1(Volume 1, 2014):125–151, 2014. ISSN 2326- 831X. doi:https://doi.org/10.1146/annurev-statistics-062713-085831.

[2] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 5580–5590, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

[3] Garoe Dorta, Sara Vicente, Lourdes Agapito, Neill D. F. Campbell, and Ivor Simpson. Structured uncertainty prediction networks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

[4] David Salinas, Michael Bohlke-Schneider, Laurent Callot, Roberto Medico, and Jan Gasthaus. Highdimensional multivariate forecasting with low-rank gaussian copula processes. In Proceedings ofthe 33rd International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2019. Curran Associates Inc.

[5] Ce Liu, Suryansh Kumar, Shuhang Gu, Radu Timofte, and Luc Van Gool. Single image depth prediction made better: A multivariate gaussian take. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17346–17356, June 2023.

[6] Tilmann Gneiting, Adrian E. Raftery, Anton H. Westveld, and Tom Goldman. Calibrated probabilistic forecasting using ensemble model output statistics and

minimum crps estimation. Monthly Weather Review, 133(5):1098 – 1118, 2005. doi:10.1175/MWR2904.1.

[7] Mikael Kuusela and Michael L. Stein. Locally stationary spatio-temporal interpolation of argo profiling float data. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 474(2220):20180400, 12 2018. ISSN 1364-5021. doi:10.1098/rspa.2018.0400.

[8] Juliane Schäfer and Korbinian Strimmer. An empirical bayes approach to inferring largescale gene association networks. Bioinformatics, 21(6):754–764, 03 2005. ISSN 1367-4803. doi:10.1093/bioinformatics/bti062.

[9] Daniel Foreman-Mackey, Eric Agol, Sivaram Ambikasaran, and Ruth Angus. Fast and scalable gaussian process modeling with applications to astronomical time series. The Astronomical Journal, 154(6):220, November 2017. ISSN 1538-3881. doi:10.3847/1538- 3881/aa9332.

[10] D.A. Nix and A.S. Weigend. Estimating the mean and variance of the target probability distribution. In Proceedings of1994 IEEE International Conference on Neural Networks (ICNN’94), volume 1, pages 55– 60 vol.1, 1994. doi:10.1109/ICNN.1994.374138.

[11] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, November 2017. URL http://arxiv.org/abs/1612. 01474. arXiv:1612.01474 [cs, stat].

[12] Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, and Georg Martius. On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=aPOpXlnV1T.

[13] Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American Statistical Association, 102(477): 359–378, 2007. doi:10.1198/016214506000001437.

[14] Ingo Steinwart and Johanna F. Ziegel. Strictly proper kernel scores and characteristic kernels on compact spaces. Applied and Computational Harmonic Analysis, 51:510–542, 2021. ISSN 1063-5203. doi:https://doi.org/10.1016/j.acha.2019.11.005.

[15] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(25):723–773, 2012.

[16] A. Philip Dawid, Monica Musio, and Laura Ventura. Minimum scoring rule inference. Scandinavian Journal of Statistics, 43(1):123–138, 2016. doi:https://doi.org/10.1111/sjos.12168.

[17] Peter M. Williams. Using neural networks to model conditional multivariate densities. Neural Computation, 8(4):843–854, 05 1996. ISSN 0899-7667. doi:10.1162/neco.1996.8.4.843.

[18] Rebecca L. Russell and Christopher Reale. Multivariate uncertainty in deep learning. IEEE Transactions on Neural Networks and Learning Systems, 33(12):7937– 7943, 2022. doi:10.1109/TNNLS.2021.3086757.

[19] Victor M.-H. Ong, David J. Nott, and Michael S. Smith. Gaussian variational approximation with a factor covariance structure. Journal of Computational and Graphical Statistics, 27(3):465–478, 2018. doi:10.1080/10618600.2017.1390472.

[20] Marcin Tomczak, Siddharth Swaroop, and Richard Turner. Efficient low rank gaussian variational inference for neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 4610–4622. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/file/ 310cc7ca5a76a446f85c1a0d641ba96d-Pape pdf.

[21] Miguel Monteiro, Loïc Le Folgoc, Daniel Coelho de Castro, Nick Pawlowski, Bernardo Marques, Konstantinos Kamnitsas, Mark van der Wilk, and Ben Glocker. Stochastic segmentation networks: modelling spatially correlated aleatoric uncertainty. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

[22] James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate curvature. In Francis Bach and David Blei, editors, Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pages 2408–2417, Lille, France, 07–09 Jul 2015. PMLR.

[23] Hippolyt Ritter, Aleksandar Botev, and David Barber. A scalable laplace approximation for neural networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/ forum?id=Skdvd2xAZ.

[24] Andrew Stirn, Harm Wessels, Megan Schertzer, Laura Pereira, Neville Sanjana, and David Knowles. Faithful

heteroscedastic regression with neural networks. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent, editors, Proceedings ofThe 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings ofMachine Learning Research, pages 5593–5613. PMLR, 25–27 Apr 2023.

[25] Lorenzo Pacchiardi, Rilwan Adewoyin, Peter Dueben, and Ritabrata Dutta. Probabilistic forecasting with generative networks via scoring rule minimization. 2021. URL https://arxiv.org/pdf/2112. 08217.pdf.

[26] Domokos M. Kelen, Ádám Jung, Péter Kersch, and Andras A. Benczur. Distribution-Free Data Uncertainty for Neural Network Regression. October 2024. URL https://openreview.net/ forum?id=pDDODPtpx9.

[27] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https:// openreview.net/forum?id=J4gRj6d5Qm.

[28] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=Jbdc0vTOcol.

[29] Vincent Zhihao Zheng and Lijun Sun. Multivariate probabilistic time series forecasting with correlated errors. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/ forum?id=cAFvxVFaii.

[30] Seongjin Choi, Nicolas Saunier, Vincent Zhihao Zheng, Martin Trépanier, and Lijun Sun. Scalable dynamic mixture model with full covariance for probabilistic traffic forecasting. Transportation Science, 59 (4):708–720, 2025. doi:10.1287/trsc.2024.0547.

[31] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.

[32] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024.

[33] Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, András Horányi, Joaquín Muñoz-Sabater, Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, Adrian Simmons, Cornel Soci, Saleh Abdalla,

Xavier Abellan, Gianpaolo Balsamo, Peter Bechtold, Gionata Biavati, Jean Bidlot, Massimo Bonavita, Giovanna De Chiara, Per Dahlgren, Dick Dee, Michail Diamantakis, Rossana Dragani, Johannes Flemming, Richard Forbes, Manuel Fuentes, Alan Geer, Leo Haimberger, Sean Healy, Robin J. Hogan, Elías Hólm, Marta Janisková, Sarah Keeley, Patrick Laloyaux, Philippe Lopez, Cristina Lupu, Gabor Radnoti, Patricia De Rosnay, Iryna Rozum, Freja Vamborg, Sebastien Villaume, and Jean-Noël Thépaut. The ERA5 global reanalysis. Quarterly Journal ofthe Royal Meteorological Society, 146(730):1999–2049, July 2020. doi:10.1002/qj.3803.

[34] Salva Rühling Cachay, Duncan Watson-Parris, and Rose Yu. U-cast: A surprisingly simple and efficient frontier probabilistic AI weather forecaster. In ICML 2026 AI for Science Workshop, 2026. URL https://openreview.net/ forum?id=u9p1tN7v69.

[35] Stephan Rasp and Sebastian Lerch. Neural networks for postprocessing ensemble weather forecasts. Monthly Weather Review, 146(11):3885–3900, 2018. ISSN 0027-0644. doi:10.1175/MWR-D-18-0187.1.

[36] Thomas Muschinski, Georg J. Mayr, Thorsten Simon, Nikolaus Umlauf, and Achim Zeileis. Cholesky-based multivariate Gaussian regression. Econometrics and Statistics, 29:261–281, January 2024. ISSN 2452- 3062. doi:10.1016/j.ecosta.2022.03.001.

[37] J. Demaeyer, J. Bhend, S. Lerch, C. Primo, B. Van Schaeybroeck, A. Atencia, Z. Ben Bouallègue, J. Chen, M. Dabernig, G. Evans, J. Faganeli Pucer, B. Hooper, N. Horat, D. Jobst, J. Merše, P. Mlakar, A. Möller, O. Mestre, M. Taillardat, and S. Vannitsem. The euppbench postprocessing benchmark dataset v1.0. Earth System Science Data, 15(6):2635–2653, 2023. doi:10.5194/essd-15-2635- 2023. URL https://essd.copernicus.org/ articles/15/2635/2023/.

[38] Moritz Feik, Sebastian Lerch, and Jan Stühmer. Graph neural networks and spatial information learning for post-processing ensemble weather forecasts, 2024. URL https://arxiv.org/abs/2407. 11050.

[39] Alexander Immer, Emanuele Palumbo, Alexander Marx, and Julia E Vogt. Effective bayesian heteroscedastic regression with deep neural networks. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https:// openreview.net/forum?id=A6EquH0enk.

[40] Sumedh Vemuganti and Nickvash Kani. Fisher8: Stabilizing neural heteroscedastic regression via

output-layer fisher geometry. In Forty-Second Annual Conference on Uncertainty in Artificial Intelligence, 2026. URL https://openreview.net/ forum?id=e1M3gVpOGp.

[41] Megh Shukla, Mathieu Salzmann, and Alexandre Alahi. TIC-TAC: A framework for improved covariance estimation in deep heteroscedastic regression. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 45244–45257. PMLR, 21–27 Jul 2024.

[42] Zhihao Zheng, Seongjin Choi, and Lijun Sun. Better batch for deep probabilistic time series forecasting. In International Conference on Artificial Intelligence and Statistics, pages 91–99. PMLR, 2024.

[43] Christos Louizos and Max Welling. Structured and efficient variational deep learning with matrix gaussian posteriors. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings ofThe 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1708–1716, New York, New York, USA, 20–22 Jun 2016. PMLR.

[44] Guodong Zhang, Shengyang Sun, David Duvenaud, and Roger Grosse. Noisy natural gradient as variational inference. In Jennifer Dy and Andreas Krause, editors, Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 5852–5861. PMLR, 10–15 Jul 2018.

[45] Edwin V. Bonilla, Kian Ming A. Chai, and Christopher K. I. Williams. Multi-task Gaussian process prediction. In Advances in Neural Information Processing Systems, volume 20, pages 153–160. Curran Associates, Inc., 2008.

[46] Paul W. Goldberg, Christopher K. I. Williams, and Christopher M. Bishop. Regression with inputdependent noise: A Gaussian process treatment. In Advances in Neural Information Processing Systems, volume 10, pages 493–499. MIT Press, 1998.

[47] Miguel Lázaro-Gredilla and Michalis K. Titsias. Variational heteroscedastic Gaussian process regression. In Proceedings of the 28th International Conference on Machine Learning, pages 841–848. Omnipress, 2011.

[48] Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P. Xing. Deep kernel learning. In Proceedings ofthe 19th International Conference on

Artificial Intelligence and Statistics, volume 51 of Proceedings of Machine Learning Research, pages 370– 378. PMLR, 2016.

[49] Michalis K. Titsias. Variational learning of inducing variables in sparse Gaussian processes. In Proceedings of the 12th International Conference on Artificial Intelligence and Statistics, volume 5 of Proceedings of Machine Learning Research, pages 567–574. PMLR, 2009.

[50] Miguel Lázaro-Gredilla, Joaquin Quiñonero-Candela, Carl Edward Rasmussen, and Aníbal R. Figueiras-Vidal. Sparse spectrum Gaussian process regression. Journal ofMachine Learning Research, 11:1865– 1881, 2010.

[51] Andrew Gordon Wilson and Hannes Nickisch. Kernel interpolation for scalable structured Gaussian processes (KISS-GP). In Proceedings ofthe 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 1775–1784. PMLR, 2015.

[52] Nadja Klein, Thomas Kneib, Stefan Lang, and Alexander Sohn. Bayesian structured additive distributional regression with an application to regional income inequality in Germany. The Annals ofApplied Statistics, 9(2):1024 – 1052, 2015. doi:10.1214/15-AOAS823.

[53] Carlo Kneissl, Christopher Bülte, Philipp Scholl, and Gitta Kutyniok. Improved probabilistic regression using diffusion models, 2025. URL https://arxiv. org/abs/2510.04583.

[54] Peter Whittle. Estimation and information in stationary time series. Arkiv för Matematik, 2(5):423–434, Aug 1953. ISSN 1871-2487. doi:10.1007/BF02590998. URL https://doi. org/10.1007/BF02590998.

[55] Damien Garreau, Wittawat Jitkrittum, and Motonobu Kanagawa. Large sample analysis of the median heuristic, 2018. URL https://arxiv.org/ abs/1707.07269.

[56] Chris A. J. Klaassen. Consistent estimation of the influence function of locally asymptotically linear estimators. The Annals ofStatistics, 15(4):1548–1562, 1987. doi:10.1214/aos/1176350609.

[57] Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1):C1–C68, 2018. doi:10.1111/ectj.12097.

[58] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, Oxford, 2013. ISBN 9780199535255. doi:10.1093/acprof:oso/9780199535255.001.0001.

[59] Svante Janson. Hypercontractivity, page 57–77. Cambridge Tracts in Mathematics. Cambridge University Press, 1997.

[60] C. A. T. Ferro. Fair scores for ensemble forecasts. Quarterly Journal of the Royal Meteorological Society, 140(683):1917–1923, 2014. doi:https://doi.org/10.1002/qj.2270.

[61] Michael Scheuerer and Thomas M. Hamill. Variogrambased proper scoring rules for probabilistic forecasts of multivariate quantities. Monthly Weather Review, 143(4):1321 – 1334, 2015. doi:10.1175/MWR-D-14- 00269.1.

[62] Tilmann Gneiting and Johannes Resin. Regression diagnostics meets forecast evaluation: conditional calibration, reliability diagrams, and coefficient of determination. Electronic Journal ofStatistics, 17(2):3226 – 3286, 2023. doi:10.1214/23-EJS2180.

[63] Sam Allen, Johanna Ziegel, and David Ginsbourger. Assessing the calibration of multivariate probabilistic forecasts. Quarterly Journal of the Royal Meteorological Society, 150(760):1315–1335, 2024. doi:https://doi.org/10.1002/qj.4647.

[64] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https:// openreview.net/forum?id=Bkg6RiCqY7.

[65] Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Battaglia, Tyler Russell, Alvaro Sanchez-Gonzalez, Vivian Yang, Rob Carver, Shreya Agrawal, Matthew Chantry, Zied Ben Bouallegue, Peter Dueben, Carla Bromberg, Jared Sisk, Luke Barrington, Aaron Bell, and Fei Sha. Weatherbench 2: A benchmark for the next generation of data-driven global weather models. Journal of Advances in Modeling Earth Systems, 16(6), 2024. doi:10.1029/2023MS004019.

[66] Christopher Bülte, Nina Horat, Julian Quinting, and Sebastian Lerch. Uncertainty quantification for datadriven weather models. Artificial Intelligencefor the Earth Systems, 2026. doi:10.1175/AIES-D-24-0049.1.

[67] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In Proceedings ofthe 28th International Conference on Neural Information Processing Systems - Volume 2, NIPS’14, pages 2366– 2374, Cambridge, MA, USA, 2014. MIT Press.

[68] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.

[69] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. ArXiv preprint, 2021.

[70] Erich L. Lehmann and Joseph P. Romano. Testing Statistical Hypotheses. Springer, 3 edition, 2005.

[71] Bernard Rosner, Robert J. Glynn, and Mei-Ling T. Lee. The Wilcoxon signed rank test for paired comparisons of clustered data. Biometrics, 62(1):185–192, 2006.

[72] Somnath Datta and Glen A. Satten. A signed-rank test for clustered data. Biometrics, 64(2):501–507, 2008.

[73] Jesse Hemerik and Jelle J. Goeman. Exact testing with random permutations. TEST, 27(4):811–825, 2018.

## APPENDIX

A Notation 14   
B Loss gradients 14   
C Spectral Correlation Estimation 17   
D Expressivity of SCORE 21   
E PAC Learnability 23   
F Toy example 34   
G Experiment details 35   
H Detailed results 41

## A NOTATION

Throughout the paper, let $\mathcal { X } \subseteq \mathbb { R } ^ { k }$ and $\mathcal { V } \subseteq \mathbb { R } ^ { d }$ denote the input and output spaces, respectively, and let $\mathcal { P } ( \mathcal { V } )$ be a convex set of probability measures on $\mathcal { V }$ . We consider a training dataset $\mathcal { D } = \{ ( \pmb { x } _ { n } , \pmb { y } _ { n } ) \} _ { n = 1 } ^ { N } \subset \mathcal { \bar { ( } } \mathcal { X } \times \mathcal { \bar { y } ) } ^ { N }$ consisting of i.i.d. samples from an unknown joint distribution $\mathbb { P } = \mathbb { P } _ { X , Y }$ with marginal $\mathbb { P } _ { X }$ and conditional law $\mathbb { P } _ { Y \mid X } ( { \boldsymbol { \cdot } } \mid x )$ . Expectations with respect to $\mathbb { P }$ are written as $\mathbb { E } _ { X , Y \sim \mathbb { P } } [ \cdot ]$ , while $P , Q \in { \mathcal { P } } ( { \mathcal { V } } )$ denote generic probability distributions. Our predictive model is a neural network $f _ { \phi } : \mathcal { X } \to \mathcal { P } ( \mathcal { Y } )$ with parameters $\phi \in \mathcal { W } \subseteq \mathbb { R } ^ { q }$ , which maps an input x to a probability distribution $f _ { \phi } ( \pmb { x } )$ . Unless stated otherwise, we restrict attention to Gaussian predictive distributions, $f _ { \phi } ( \pmb { x } ) = \mathcal { N } ( \pmb { \mu } _ { \phi } ( \pmb { x } ) , \pmb { \Sigma } _ { \phi } ( \pmb { x } ) )$ where $\pmb { \mu } _ { \phi } ( \pmb { x } ) \in \mathbb { R } ^ { d }$ and $\pmb { \Sigma } _ { \phi } ( \pmb { x } ) \succ 0$ denote the predicted mean vector and positive definite covariance matrix, respectively. We write $\operatorname { d i a g } ( { \pmb v } )$ for the diagonal matrix with diagonal entries v and, for a matrix M, diag(M) for the diagonal matrix with the same diagonal as $M$ . Moreover, I denotes the identity matrix, 1 the vector of ones, and ⊙ the entrywise product. We use bold lower-case letters for vectors and bold upper-case letters for matrices, with the exception of correlation matrices, which we write R throughout to distinguish them from covariances. For a matrix A, the transpose and Hermitian transpose are denoted by $A ^ { \top }$ and $A ^ { * }$ , respectively, $\| A \| _ { F }$ is the Frobenius norm, $\lambda _ { \operatorname* { m i n } }$ and $\lambda _ { \mathrm { m a x } }$ are the extreme eigenvalues, and $A \succ 0 \left( A \succeq 0 \right)$ means that a symmetric or Hermitian A is positive (semi)definite. A hat, as in $\hat { \Sigma }$ , marks quantities in the transformed coordinates $\hat { z } = B z$ of a unitary B.

## B LOSS GRADIENTS

This section derives the gradients of $S _ { \mathrm { l o g } }$ and $S _ { k }$ to prove Lemma 2.1 and Theorem 3.1. Throughout, we consider a multivariate normal distribution $P = \mathcal { N } ( \mu , \Sigma )$ with $\Sigma \succ 0$ , decomposed as $\pmb { \Sigma } = \pmb { L } \pmb { L } ^ { \top }$ , where $\pmb { L }$ is a lower triangular matrix with positive diagonal. We analyze Lipschitz continuity of the scores in $( \mu , L )$ through boundedness of their gradients. Let $\boldsymbol { y } \in \mathbb { R } ^ { d }$ denote the observation and $r : = \mu - y$ the residual. Finally, let $\lambda _ { 1 } \geq \cdots \geq \lambda _ { d }$ denote the ordered eigenvalues of $\pmb { \Sigma }$ and $e _ { k } , k = 1 , \ldots , d .$ , the corresponding eigenvectors.

## B.1 LOG SCORE

Recall that the log score of a multivariate Gaussian is given by

$$
S _ { \mathrm { l o g } } ( P , \pmb { y } ) = \frac { 1 } { 2 } \log \operatorname * { d e t } \pmb { \Sigma } + \frac { 1 } { 2 } ( \pmb { y } - \pmb { \mu } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \pmb { y } - \pmb { \mu } ) + \frac { d } { 2 } \log ( 2 \pi ) .
$$

Let $\pmb { w } : = \pmb { \Sigma } ^ { - 1 } \pmb { r }$ and $z : = { \pmb L } ^ { - 1 } r$ , so that ${ \pmb { L } } ^ { \top } { \pmb w } = { \pmb z }$ . The gradients of $S _ { \mathrm { l o g } }$ take the form

$$
\nabla _ { \mu } S _ { \mathrm { l o g } } = \pmb { w } , \qquad \left[ \nabla _ { \pmb { L } } S _ { \mathrm { l o g } } \right] _ { i i } = \frac { 1 } { L _ { i i } } - w _ { i } z _ { i } , \qquad \left[ \nabla _ { \pmb { L } } S _ { \mathrm { l o g } } \right] _ { i j } = - w _ { i } z _ { j } ,
$$

for all $i = 1 , \ldots , d$ and $j < i .$

## B.1.1 Proof of Lemma 2.1

Proof. Representing Σ via its eigendecomposition leads to

$$
\pmb { \Sigma } = \sum _ { k = 1 } ^ { d } \lambda _ { k } \pmb { e } _ { k } \pmb { e } _ { k } ^ { \top } , \quad \pmb { \Sigma } ^ { - 2 } = \sum _ { k = 1 } ^ { d } \lambda _ { k } ^ { - 2 } \pmb { e } _ { k } \pmb { e } _ { k } ^ { \top } .
$$

Further, we rewrite $\textstyle { \pmb { r } } = \sum _ { k = 1 } ^ { d } { \tilde { r } } _ { k } { \pmb { e } } _ { k }$ with $\tilde { r } _ { k } = e _ { k } ^ { \top } r$ . Since $\pmb { \Sigma } ^ { - 1 }$ is symmetric, we obtain

$$
\| \nabla _ { \mu } S _ { \mathrm { l o g } } \| ^ { 2 } = \| \Sigma ^ { - 1 } \pmb { r } \| ^ { 2 } = \pmb { r } ^ { \top } \Sigma ^ { - 2 } \pmb { r } = \sum _ { k = 1 } ^ { d } \lambda _ { k } ^ { - 2 } \tilde { r } _ { k } ^ { 2 } ,
$$

from which we obtain the lower bound

$$
\sum _ { k = 1 } ^ { d } \lambda _ { k } ^ { - 2 } \tilde { r } _ { k } ^ { 2 } \geq \lambda _ { \operatorname* { m i n } } ^ { - 2 } \tilde { r } _ { d } ^ { 2 } \implies \| \nabla _ { \mu } S _ { \log } \| \geq \lambda _ { \operatorname* { m i n } } ^ { - 1 } | \tilde { r } _ { d } | .
$$

Therefore, along any sequence of covariance matrices $\pmb { \Sigma } ^ { ( n ) }$ with $\lambda _ { \operatorname* { m i n } } \bigl ( \Sigma ^ { ( n ) } \bigr ) \to 0$ whose smallest eigenvector is not asymptotically orthogonal to the residual, i.e. lim in $\mathrm { f } _ { n } \vert e _ { d } ^ { ( n ) \top } r \vert > 0$ , we have $\| \nabla _ { \mu } S _ { \log } \|  \infty$

For the second gradient, we have

$$
\| \nabla _ { L } S _ { \log } \| _ { F } ^ { 2 } = \sum _ { i } \left( \frac { 1 } { L _ { i i } } - w _ { i } z _ { i } \right) ^ { 2 } + \sum _ { i > j } w _ { i } ^ { 2 } z _ { j } ^ { 2 } .
$$

Note that a vanishing diagonal entry $L _ { i i } \ \to \ 0$ alone is not sufficient for divergence, since the two terms $1 / L _ { i i }$ and $w _ { i } z _ { i }$ may cancel. It suffices to exhibit a divergent sequence. Consider $\pmb { L } ^ { ( n ) } = \mathrm { d i a g } ( \sigma _ { n } , 1 , \dots , 1 )$ with $\sigma _ { n }  0$ , so that $\lambda _ { \operatorname* { m i n } } ^ { ( n ) } = \sigma _ { n } ^ { 2 } \to 0$ , which gives

$$
\left[ \nabla _ { L } S _ { \mathrm { l o g } } \right] _ { 1 1 } = \frac { 1 } { \sigma _ { n } } - \frac { r _ { 1 } ^ { 2 } } { \sigma _ { n } ^ { 3 } } ,
$$

which diverges for every fixed $r _ { 1 }$ , including $r _ { 1 } = 0$ . Consequently, sup $\| \nabla S _ { \mathrm { l o g } } \|$ is unbounded in every neighborhood of $\lambda _ { \operatorname* { m i n } } = 0$ , and the scoring rule is not (globally) Lipschitz continuous in $( \mu , L )$ □

## B.2 KERNEL SCORE

The closed form of the Gaussian kernel score [53] with bandwidth $\gamma > 0$ is given by

$$
S _ { k } ( P , \pmb { y } ) = \frac { 1 } { 2 } \underbrace { \mathrm { d e t } ( \pmb { A } ) ^ { - 1 / 2 } } _ { c _ { 1 } } - \underbrace { \mathrm { d e t } ( \pmb { G } ) ^ { - 1 / 2 } } _ { c _ { 2 } } \underbrace { \mathrm { e x p } \left( - \frac { 1 } { \gamma ^ { 2 } } r ^ { \top } \pmb { G } ^ { - 1 } r \right) } _ { E } + \frac { 1 } { 2 } ,
$$

for $r : = \mu - y , A : = I + 4 \Sigma / \gamma ^ { 2 }$ and $G : = I + 2 \Sigma / \gamma ^ { 2 }$ . Further, define $w : = G ^ { - 1 } r$ and note that $\begin{array} { r } { \nabla _ { L } E = \frac { 4 E } { \gamma ^ { 4 } } \pmb { w w } ^ { \top } \pmb { L } } \end{array}$ Then, the gradients of the Gaussian kernel score are given by

$$
\nabla _ { \mu } S _ { k } = \frac { 2 c _ { 2 } E } { \gamma ^ { 2 } } G ^ { - 1 } r , \qquad \nabla _ { L } S _ { k } = - \frac { 2 c _ { 1 } } { \gamma ^ { 2 } } A ^ { - 1 } L + \frac { 2 c _ { 2 } E } { \gamma ^ { 2 } } G ^ { - 1 } L - \frac { 4 c _ { 2 } E } { \gamma ^ { 4 } } w w ^ { \top } L .\tag{8}
$$

Here $\nabla _ { L }$ denotes the unconstrained derivative with respect to a full matrix argument; the gradient with respect to a lower-triangular L is its lower-triangular part, whose Frobenius norm is bounded by that of the full matrix.

## B.2.1 Proof of Theorem 3.1

Proof. Throughout, let $\begin{array} { r } { \pmb { \Sigma } = \sum _ { i } \lambda _ { i } \pmb { e } _ { i } \pmb { e } _ { i } ^ { \top } } \end{array}$ be the eigendecomposition of Σ with the eigenvectors $e _ { i }$ introduced above, and let

$$
A = \sum _ { i } \alpha _ { i } e _ { i } e _ { i } ^ { \top } , \alpha _ { i } = 1 + \frac { 4 } { \gamma ^ { 2 } } \lambda _ { i } \geq 1 , \qquad G = \sum _ { i } \beta _ { i } e _ { i } e _ { i } ^ { \top } , \beta _ { i } = 1 + \frac { 2 } { \gamma ^ { 2 } } \lambda _ { i } \geq 1 ,
$$

which are the eigendecompositions of A and $G ,$ since both are functions of Σ. Since all eigenvalues of A and G are at least one, we have $\operatorname* { d e t } ( A )$ , det $( G ) \geq 1$ and therefore

$$
c _ { 1 } = \operatorname* { d e t } ( A ) ^ { - 1 / 2 } \leq 1 , \qquad c _ { 2 } = \operatorname* { d e t } ( G ) ^ { - 1 / 2 } \leq 1 .
$$

Write $\rho _ { i } : = e _ { i } ^ { \top } r$ and define the normalized coefficients

$$
t _ { i } : = \frac { \rho _ { i } } { \gamma \sqrt { \beta _ { i } } } , \qquad \mathrm { s u c h t h a t } \qquad \rho _ { i } ^ { 2 } = \gamma ^ { 2 } \beta _ { i } t _ { i } ^ { 2 } .
$$

The exponent of E is exactly $- \| t \| ^ { 2 }$ , since

$$
\frac { 1 } { \gamma ^ { 2 } } r ^ { \top } G ^ { - 1 } r = \frac { 1 } { \gamma ^ { 2 } } r ^ { \top } \Big ( \sum _ { i } \beta _ { i } ^ { - 1 } e _ { i } e _ { i } ^ { \top } \Big ) r = \sum _ { i } \frac { \rho _ { i } ^ { 2 } } { \gamma ^ { 2 } \beta _ { i } } = \sum _ { i } t _ { i } ^ { 2 } = \| t \| ^ { 2 } ,
$$

hence $E = \exp ( - \| t \| ^ { 2 } )$

Gradient in $\mu .$ With $\beta _ { i } \geq 1$ , we get

$$
\| \pmb { w } \| ^ { 2 } = \| \pmb { G } ^ { - 1 } \pmb { r } \| ^ { 2 } = \sum _ { i } \frac { \rho _ { i } ^ { 2 } } { \beta _ { i } ^ { 2 } } = \gamma ^ { 2 } \sum _ { i } \frac { t _ { i } ^ { 2 } } { \beta _ { i } } \leq \gamma ^ { 2 } \| \pmb { t } \| ^ { 2 } .
$$

Using the fact that $| x | \exp ( - x ^ { 2 } ) \leq 1 / \sqrt { 2 e }$ for all $x \in \mathbb { R }$ , we obtain

$$
\| w \| E \leq \gamma \| t \| \exp ( - \| t \| ^ { 2 } ) \leq \frac { \gamma } { \sqrt { 2 e } } .
$$

Together with $c _ { 2 } \leq 1$ this yields

$$
\| \nabla _ { \pmb { \mu } } S _ { k } \| = \frac { 2 c _ { 2 } E } { \gamma ^ { 2 } } \| \pmb { w } \| \leq \frac { 2 } { \gamma ^ { 2 } } \cdot \frac { \gamma } { \sqrt { 2 e } } = \frac { 1 } { \gamma } \sqrt { \frac { 2 } { e } } .
$$

Gradient in L. Since L need not be symmetric, we use its singular value decomposition $\begin{array} { r } { \pmb { L } = \sum _ { i } \sigma _ { i } \pmb { e } _ { i } \pmb { v } _ { i } ^ { \top } } \end{array}$ ; the left singular vectors of $\pmb { L }$ coincide with the eigenvectors of $\begin{array} { r } { \Sigma = { \cal L } { \cal L } ^ { \top } \stackrel { - } { = } \sum _ { i } \sigma _ { i } ^ { 2 } e _ { i } e _ { i } ^ { \top } } \end{array}$ , so that $\sigma _ { i } ^ { 2 } = \lambda _ { i }$ is the ith eigenvalue of Σ. Consequently

$$
A ^ { - 1 } L = \sum _ { i } \frac { \sigma _ { i } } { \alpha _ { i } } e _ { i } { v } _ { i } ^ { \top } , \qquad G ^ { - 1 } L = \sum _ { i } \frac { \sigma _ { i } } { \beta _ { i } } e _ { i } { v } _ { i } ^ { \top } ,
$$

and, since $e _ { i }$ and ${ \mathbf { } } v _ { i }$ are orthonormal, $\begin{array} { r } { \big \| \sum _ { i } s _ { i } e _ { i } \pmb { v } _ { i } ^ { \top } \big \| _ { F } ^ { 2 } = \sum _ { i } s _ { i } ^ { 2 } } \end{array}$

For the first term $\begin{array} { r } { \pmb { T } _ { 1 } : = - \frac { 2 c _ { 1 } } { \gamma ^ { 2 } } \pmb { A } ^ { - 1 } \pmb { L } } \end{array}$ , we obtain

$$
\Vert { \cal T } _ { 1 } \Vert _ { F } ^ { 2 } = \frac { 4 c _ { 1 } ^ { 2 } } { \gamma ^ { 4 } } \sum _ { i } \frac { \lambda _ { i } } { \alpha _ { i } ^ { 2 } } = \frac { 4 c _ { 1 } ^ { 2 } } { \gamma ^ { 4 } } \sum _ { i } \frac { \lambda _ { i } } { ( 1 + 4 \lambda _ { i } / \gamma ^ { 2 } ) ^ { 2 } } .
$$

Using $c _ { 1 } \leq 1$ and that differentiating $x \mapsto x / ( 1 + c x ) ^ { 2 }$ for $c > 0$ shows that it is maximized at $x = 1 / c$ with value $1 / ( 4 c )$ which gives $\frac { x } { ( 1 + 4 x / \gamma ^ { 2 } ) ^ { 2 } } \leq \gamma ^ { 2 } / 1 6$ for all $x \geq 0$ , we obtain

$$
\Vert \pmb { T } _ { 1 } \Vert _ { F } ^ { 2 } \le \frac { 4 } { \gamma ^ { 4 } } d \frac { \gamma ^ { 2 } } { 1 6 } = \frac { d } { 4 \gamma ^ { 2 } } \implies \Vert \pmb { T } _ { 1 } \Vert _ { F } \le \frac { \sqrt { d } } { 2 \gamma } .
$$

For the second term, $\begin{array} { r } { \pmb { T } _ { 2 } : = \frac { 2 c _ { 2 } E } { \gamma ^ { 2 } } \pmb { G } ^ { - 1 } \pmb { L } } \end{array}$ , we similarly obtain

$$
\| { \bf T } _ { 2 } \| _ { F } ^ { 2 } = \frac { 4 c _ { 2 } ^ { 2 } E ^ { 2 } } { \gamma ^ { 4 } } \sum _ { i } \frac { \lambda _ { i } } { \beta _ { i } ^ { 2 } } .
$$

Using $c _ { 2 } E \le 1$ and, by the same maximization with $\begin{array} { r } { c = 2 / \gamma ^ { 2 } , \frac { \lambda _ { i } } { ( 1 + 2 \lambda _ { i } / \gamma ^ { 2 } ) ^ { 2 } } \le \gamma ^ { 2 } / 8 } \end{array}$ , we obtain

$$
\left. \mathbf { T } _ { 2 } \right. _ { F } ^ { 2 } \leq \frac { 4 } { \gamma ^ { 4 } } d \frac { \gamma ^ { 2 } } { 8 } = \frac { d } { 2 \gamma ^ { 2 } } \implies \left. \mathbf { T } _ { 2 } \right. _ { F } \leq \frac { \sqrt { d } } { \sqrt { 2 } \gamma } .
$$

Finally, consider the third term $\begin{array} { r } { \pmb { T } _ { 3 } : = - \frac { 4 c _ { 2 } E } { \gamma ^ { 4 } } \pmb { w } \pmb { w } ^ { \top } \pmb { L } } \end{array}$ . Since ww ${ } ^ { \top } { \pmb L } = { \pmb w } ( { \pmb L } ^ { \top } { \pmb w } ) ^ { \top }$ has rank one,

$$
\| T _ { 3 } \| _ { F } = \frac { 4 c _ { 2 } E } { \gamma ^ { 4 } } \| { \pmb w } \| \| { \pmb L } ^ { \top } { \pmb w } \| , \quad \| { \pmb w } \| ^ { 2 } = r ^ { \top } { \pmb G } ^ { - 2 } r = \sum _ { i } \frac { \rho _ { i } ^ { 2 } } { \beta _ { i } ^ { 2 } } , \quad \| { \pmb L } ^ { \top } { \pmb w } \| ^ { 2 } = { \pmb w } ^ { \top } { \pmb L } { \pmb L } ^ { \top } { \pmb w } = { \pmb w } ^ { \top } \Sigma { \pmb w } = \sum _ { i } \frac { \lambda _ { i } \rho _ { i } ^ { 2 } } { \beta _ { i } ^ { 2 } } .
$$

Substituting $\rho _ { i } ^ { 2 } = \gamma ^ { 2 } \beta _ { i } t _ { i } ^ { 2 }$ and using $\beta _ { i } \geq 1$ for the first identity gives

$$
\| { \pmb w } \| ^ { 2 } = \gamma ^ { 2 } \sum _ { i } \frac { t _ { i } ^ { 2 } } { \beta _ { i } } \leq \gamma ^ { 2 } \| { \pmb t } \| ^ { 2 } , \quad \| { \pmb L } ^ { \top } { \pmb w } \| ^ { 2 } = \gamma ^ { 2 } \sum _ { i } \frac { \lambda _ { i } } { \beta _ { i } } t _ { i } ^ { 2 } \leq \frac { \gamma ^ { 4 } } { 2 } \| { \pmb t } \| ^ { 2 } ,
$$

where the second bound uses

$$
{ \frac { \lambda _ { i } } { \beta _ { i } } } = { \frac { \lambda _ { i } } { 1 + 2 \lambda _ { i } / \gamma ^ { 2 } } } \leq { \frac { \gamma ^ { 2 } } { 2 } } \qquad { \mathrm { f o r ~ a l l ~ } } \lambda _ { i } \geq 0 .
$$

Combining yields

$$
\| w \| \| L ^ { \top } w \| E \leq \gamma \| t \| \cdot { \frac { \gamma ^ { 2 } } { \sqrt { 2 } } } \| t \| \exp ( - \| t \| ^ { 2 } ) = { \frac { \gamma ^ { 3 } } { \sqrt { 2 } } } \| t \| ^ { 2 } \exp ( - \| t \| ^ { 2 } ) \leq { \frac { \gamma ^ { 3 } } { \sqrt { 2 } e } } ,
$$

since $\begin{array} { r } { \operatorname* { m a x } _ { s \geq 0 } s \exp ( - s ) = e ^ { - 1 } } \end{array}$ . In total, using $c _ { 2 } \leq 1$ ，

$$
\Vert \mathbf { T } _ { 3 } \Vert _ { F } \leq \frac { 4 } { \gamma ^ { 4 } } \frac { \gamma ^ { 3 } } { \sqrt { 2 } e } = \frac { 2 \sqrt { 2 } } { e \gamma } .
$$

Combining the three terms with the triangle inequality, we obtain

$$
\| \nabla _ { L } S _ { k } \| _ { F } \leq \left( \frac { 1 } { 2 } + \frac { 1 } { \sqrt { 2 } } \right) \frac { \sqrt { d } } { \gamma } + \frac { 2 \sqrt { 2 } } { e \gamma } .
$$

All bounds hold for every $\mu , y \in \mathbb { R } ^ { d }$ and every lower-triangular L, including singular L, since no step requires invertibility.   
As the domain of $( \mu , L )$ is convex, the bounded gradients imply that $( \pmb { \mu } , \pmb { L } ) \mapsto S _ { k } ( P , \pmb { y } )$ is globally Lipschitz continuous.

## C SPECTRAL CORRELATION ESTIMATION

This appendix collects the material behind Section 3.3.

## C.1 PROOFS

We provide the proofs of Proposition 3.2, Lemma 3.3 and Proposition 3.4.

## C.1.1 Proof of Proposition 3.2

Proof. Let $z , z ^ { \prime } \sim P$ be independent. Then BZ, $B Z ^ { \prime } \sim B _ { \# } P$ are independent, and for all $z , z ^ { \prime } \in \mathbb { R } ^ { d }$

$$
k ( B z , B z ^ { \prime } ) = \kappa \big ( | | B ( z - z ^ { \prime } ) | | _ { 2 } \big ) = \kappa \big ( | | z - z ^ { \prime } | | _ { B ^ { * } B } \big ) = k _ { B } ( z , z ^ { \prime } ) ,
$$

since $\lVert B v \rVert _ { 2 } ^ { 2 } = v ^ { * } B ^ { * } B v$ . By definition of the kernel score, for the prediction $B _ { \# } P$ and the observation $_ { B y }$

$$
\begin{array} { r l } & { S _ { k } ( B _ { \# } P , B y ) = \frac { 1 } { 2 } \operatorname { \mathbb { E } } \bigl [ k ( B Z , B Z ^ { \prime } ) \bigr ] + \frac { 1 } { 2 } k ( B y , B y ) - \operatorname { \mathbb { E } } \bigl [ k ( B y , B Z ) \bigr ] } \\ & { \qquad = \frac { 1 } { 2 } \operatorname { \mathbb { E } } \bigl [ k _ { B } ( Z , Z ^ { \prime } ) \bigr ] + \frac { 1 } { 2 } k _ { B } ( y , y ) - \operatorname { \mathbb { E } } \bigl [ k _ { B } ( y , Z ) \bigr ] = S _ { k _ { B } } ( P , y ) , } \end{array}
$$

which is the identity in (i). For the second claim in (i), assume that $S _ { k }$ is strictly proper, i.e., that k is characteristic. Let Q be a further distribution on $\mathbb { R } ^ { d }$ and $W , W ^ { \prime } \sim Q$ independent. By the same computation,

$$
\mathrm { M M D } _ { k _ { B } } ^ { 2 } ( P , Q ) = \mathbb { E } \bigl [ k _ { B } ( Z , Z ^ { \prime } ) \bigr ] + \mathbb { E } \bigl [ k _ { B } ( W , W ^ { \prime } ) \bigr ] - 2 \mathbb { E } \bigl [ k _ { B } ( Z , W ) \bigr ] = \mathrm { M M D } _ { k } ^ { 2 } ( B _ { \# } P , B _ { \# } Q ) .
$$

If $P \neq Q$ , then $B _ { \# } P \neq B _ { \# } Q$ because B is invertible, so the right-hand side is positive as k is characteristic. Hence $k _ { B }$ is characteristic, and since the divergence of $S _ { k _ { B } }$ equals $\scriptstyle { \frac { 1 } { 2 } } \mathrm { M M D } _ { k _ { B } } ^ { 2 }$ , the score $S _ { k _ { B } }$ is strictly proper.

For (ii), B unitary means $B ^ { * } B = I , \mathrm { s o } \parallel \cdot \parallel _ { B ^ { * } B } = \parallel \cdot \parallel _ { 2 }$ and $k _ { B } = k$ . The claim follows from (i).

## C.1.2 Proof of Lemma 3.3

Proof. Write $C = B ^ { * } \hat { \Sigma } B$ and $\pmb { \Delta } \in \mathbb { R } ^ { d }$ with $\Delta _ { i } : = C _ { i i }$ . Since B is invertible and $\hat { \Sigma }$ is Hermitian positive definite, C is Hermitian positive definite as well. Indeed, $C ^ { * } = C$ , and $v ^ { * } C v = ( B v ) ^ { * } \hat { \Sigma } ( B v ) > 0$ for $\mathbf { \nabla } \mathbf { \boldsymbol { v } } \neq \mathbf { \boldsymbol { 0 } }$ . Its diagonal entries $\Delta _ { i }$ are therefore real and strictly positive, so corr $\mathbf { \partial } \cdot ( \pmb { C } ) = \mathrm { d i a g } ( \pmb { \Delta } ) ^ { - 1 / 2 } \pmb { C } \mathrm { d i a g } ( \pmb { \Delta } ) ^ { - 1 / 2 }$ is well defined. By the same congruence argument, now with the invertible diagonal matrix $\mathrm { d i a g } ( \Delta ) ^ { - 1 / 2 }$ , corr(C) is again Hermitian positive definite, and its diagonal entries are $\Delta _ { i } ^ { - 1 } C _ { i i } = 1$ . With $D _ { \tau } = \mathrm { d i a g } ( \sqrt { \tau } )$ for $\tau \in \mathbb { R } _ { > 0 } ^ { d } ,$ it follows that $\left( D _ { \tau } \operatorname { c o r r } ( C ) D _ { \tau } \right) _ { i i } = \tau _ { i }$ for every i. □

## C.1.3 Proof of Proposition 3.4

Proof. For (i), the objective of the first stage is separable and its i-th summand is the expected $S _ { 1 }$ -score of $\mathcal { N } ( \mu _ { i } , \tau _ { i } )$ under the i-th marginal of Q, which is $\mathcal { N } ( \mu _ { Q , i } , ( \Sigma _ { Q } ) _ { i i } )$ and thus lies in the candidate family. Since $S _ { 1 }$ is strictly proper, the summand is minimized by the true marginal and by no other, so $\mu _ { i } ^ { \star } = \mu _ { Q , i }$ and $\tau _ { i } ^ { \star } = ( \pmb { \Sigma } _ { Q } ) _ { i i }$ , hence $D ^ { \star } = D _ { Q }$

For (ii), insert $\pmb { \mu } ^ { \star } = \pmb { \mu } _ { Q }$ and $D ^ { \star } = D _ { Q }$ , and write

$$
m : = { D _ { Q } ^ { - 1 } } \mu _ { Q } , \qquad Z : = ( D ^ { \star } ) ^ { - 1 } Y \sim { \mathcal { N } } ( m , R _ { Q } ) , \qquad W : = Z - m \sim { \mathcal { N } } ( \mathbf { 0 } , R _ { Q } ) ,
$$

the law of Z being given by (4). The candidates of the second stage are $\mathcal { N } ( \boldsymbol { m } , R )$ with $R \in { \mathcal { M } } = \{ \operatorname { c o r r } ( B ^ { * } { \hat { \Sigma } } B ) \colon { \hat { \Sigma } } \in S \}$ Then

$$
\begin{array} { r l r l } { R ^ { \star } \in \underset { R \in \mathcal { M } } { \mathrm { a r g } \mathrm { m i n } } \ \mathbb { E } S _ { 2 } \big ( \mathcal { N } ( m , R ) , Z \big ) } \\ & { \quad = \underset { R \in \mathcal { M } } { \mathrm { a r g } \mathrm { m i n } } \ \mathbb { E } S _ { 2 } \big ( \mathcal { N } ( \mathbf { 0 } , R ) , W \big ) } & & { \quad \mathrm { t a r g e t ~ a n d ~ c a n d i d a t e s ~ s h a r e ~ } m } \\ & { \quad = \underset { R \in \mathcal { M } } { \mathrm { a r g } \mathrm { m i n } } \ \Big [ \mathbb { E } S _ { 2 } \big ( \mathcal { N } ( \mathbf { 0 } , R ) , W \big ) - \mathbb { E } S _ { 2 } \big ( \mathcal { N } ( \mathbf { 0 } , R _ { Q } ) , W \big ) \Big ] } & & { \quad \mathrm { s u b r a c t e d ~ t e r m ~ f r e e ~ o f ~ } R } \\ & { \quad = \underset { R \in \mathcal { M } } { \mathrm { a r g } \mathrm { m i n } } \ D \big ( \mathcal { N } ( \mathbf { 0 } , R ) , \mathcal { N } ( \mathbf { 0 } , R _ { Q } ) \big ) , } \end{array}
$$

where the second step is the translation invariance of $S _ { 2 }$ and the last step is the definition of the divergence, the sub tracted term being the score entropy. Hence R<sup>⋆</sup> minimizes $D ( \mathcal { N } ( \mathbf { 0 } , R ) , \mathcal { N } ( \mathbf { 0 } , R _ { Q } ) )$ over $\mathcal { M } .$ . The deployed prediction is $\mathcal { N } ( \mu _ { Q } , D _ { Q } R ^ { \star } D _ { Q } )$ , whose marginal variances are $( \pmb { \Sigma } _ { Q } ) _ { i i }$ by Lemma 3.3. □

## C.2 IMPLEMENTATION

We first describe how the network outputs are turned into a prediction, then the closed forms for each stage.

## C.2.1 From network outputs to a prediction

All predictions are conditional on the input. A single network maps the input to all parameters at once,

$$
{ \pmb x } \longmapsto ( { \pmb \mu } ( { \pmb x } ) , { \pmb \tau } ( { \pmb x } ) , \hat { \Sigma } ( { \pmb x } ) ) ,
$$

and the predictive law at x is $\mathcal { N } ( \pmb { \mu } ( \pmb { x } ) , \pmb { \Sigma } _ { P } ( \pmb { x } ) )$ with $\Sigma _ { P }$ assembled from these outputs by (5). Note that the staged objective of Section 3.3 places no constraint on how the network produces them. We specify the architecture per benchmark in Appendix G. However, common to all of them is the output head. For a diagonal core it emits three groups of d values, $( h _ { \mu } , h _ { \tau } , h _ { s } )$ , mapped to the parameters by

$$
\mu = h _ { \mu } , \qquad \tau = \mathrm { s o f t p l u s } ( h _ { \tau } ) + \epsilon , \qquad s = d \mathrm { s o f t m a x } ( h _ { s } ) ,\tag{9}
$$

where ${ \mathrm { s o f t p l u s } } ( x ) = \log ( 1 + e ^ { x } )$ acts entrywise with a floor $\epsilon > 0$ for numerical safety and the softmax runs over the coefficient index, softmax $\begin{array} { r } { \dot { \mathbf { \eta } } ( \pmb { h } ) _ { k } = e ^ { h _ { k } } / \sum _ { l } e ^ { h _ { l } } } \end{array}$ . The softplus makes the variances positive and the softmax makes the spectrum positive with $\begin{array} { r } { \sum _ { k } s _ { k } = d , } \end{array}$ hence $\bar { s } = 1$ . For the DFT, the spectrum is subsequently symmetrized, $s _ { k } \gets$ $\frac { 1 } { 2 } ( s _ { k } + s _ { d - k } )$ , which preserves $\textstyle \sum _ { k } s _ { k } = d$ and makes $F ^ { * } \deg ( s ) F$ real. The floor $s  ( s + \epsilon ) / ( 1 + \epsilon )$ keeps the core positive definite. The prediction is then assembled as in (5), which only requires the implied marginal variances $\Delta _ { i } = ( B ^ { * } \hat { \Sigma } B ) _ { i i }$

$$
{ \Delta } = { H ^ { \top } } s , \qquad { H } : = { \left( { \left| { \cal B } _ { k i } \right| ^ { 2 } } \right) } _ { k , i } , \qquad \tilde { D } = \mathrm { { d i a g } } \left( { \sqrt { \tau / \Delta } } \right) , \qquad { \Sigma } _ { P } = \tilde { D } { \cal B } ^ { * } \mathrm { { d i a g } } ( s ) { \cal B } \tilde { D } .\tag{10}
$$

For the DFT $| F _ { k i } | ^ { 2 } = 1 / d , \mathrm { s o } \ \Delta = \bar { s } { \bf 1 } = { \bf 1 } , \tilde { D } = \mathrm { d i a g } ( \sqrt { \tau } )$ , and the renormalization is the identity. When the target is a $d \sb { 1 } \times d \sb { 2 }$ grid and the transform is separable, $B = B _ { 1 } \otimes B _ { 2 }$ acting along its axes, then ${ \pmb { H } } = { \pmb { H } } _ { 1 } \otimes { \pmb { H } } _ { 2 }$ with $H _ { j } = ( | ( B _ { j } ) _ { k i } | ^ { 2 } ) _ { k , i }$ , so $\pmb { \Delta }$ reshaped to the grid equals ${ H _ { 1 } ^ { \top } S H _ { 2 } }$ with S the spectrum reshaped likewise. This is one small matrix product per axis, and B itself is never materialized.

Low-rank correction. The diagonal core is our default case, although the construction of SCORE, and in particular Lemma 3.3, is more general. A straightforward extension is to add a low-rank term. A correction of rank $r \ll$ d buys r directions beyond what B diagonalizes. For the DFT these are r non-stationary directions on a stationary background. The head emits r further groups $\pmb { u } _ { j } = \pmb { h } _ { u , j } \in \mathbb { R } ^ { d } , j = 1 , \dots , r$ , collected in $\pmb { U } = ( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { r } )$ , without any constraint, and the transformed core becomes

$$
B ^ { * } \hat { \Sigma } B = B ^ { * } \mathrm { d i a g } ( s ) B + \sum _ { j = 1 } ^ { r } { \pmb { u } } _ { j } { \pmb { u } } _ { j } ^ { \top } ,\tag{11}
$$

which is real by construction, the factors being real fields in the physical domain. In particular, no conjugate symmetry constraint is needed here. The assembly gains one term per factor, that is

$$
\Delta = H ^ { \top } s + \sum _ { j = 1 } ^ { r } u _ { j } \odot u _ { j } , \qquad \Sigma _ { \cal P } = \tilde { \cal D } B ^ { * } \hat { \Sigma } B \tilde { \cal D } , \qquad \hat { \Sigma } = \mathrm { d i a g } ( s ) + \hat { \cal U } \hat { \cal U } ^ { * } , \quad \hat { \cal U } : = { \cal B } { \cal U } .\tag{12}
$$

Note that the scale gauge now acts jointly, $( s , U ) \mapsto ( c s , { \sqrt { c } } U )$ , and is again fixed by $\bar { s } = 1$

## C.2.2 Scores and closed forms

The first stage scores the d marginals, the second a d-dimensional Gaussian assembled from the core, which calls for different closed forms. Throughout, $D : = \mathrm { d i a g } ( \sqrt { \tau } )$ denotes the marginal scales, and ${ \pmb m } = { \pmb D } ^ { - 1 } { \pmb \mu } ; { \pmb Z } = { \pmb D } ^ { - 1 } { \pmb Y }$ are the standardized mean and target of (7).

First stage. As first stage default score we use the CRPS, which for a univariate Gaussian is available in closed form. With $z _ { i } : = ( y _ { i } - \mu _ { i } ) / \sqrt { \tau _ { i } }$ and $\varphi , \Phi$ the standard normal density and distribution function,

$$
\begin{array} { r } { \mathrm { C R P S } \big ( \mathcal { N } ( \mu _ { i } , \tau _ { i } ) , y _ { i } \big ) = \sqrt { \tau _ { i } } \Big [ z _ { i } \big ( 2 \Phi ( z _ { i } ) - 1 \big ) + 2 \varphi ( z _ { i } ) - \frac { 1 } { \sqrt { \pi } } \Big ] , } \end{array}\tag{13}
$$

summed over the d coordinates at $\mathcal O ( d )$

Second stage. Both the log score and the kernel score have closed forms for a d-dimensional Gaussian predictive. For $\scriptstyle { \mathcal { N } } ( \mu , \Sigma )$ observed at y, and the log score,

$$
2 S _ { \mathrm { l o g } } \left( \mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } ) , \pmb { y } \right) = d \log 2 \pi + \log \operatorname* { d e t } \pmb { \Sigma } + ( \pmb { y } - \pmb { \mu } ) ^ { \top } \pmb { \Sigma } ^ { - 1 } ( \pmb { y } - \pmb { \mu } ) ,\tag{14}
$$

and, for the Gaussian kerne $k ( \pmb { u } , \pmb { v } ) = \mathrm { e x p } \big ( - \| \pmb { u } - \pmb { v } \| _ { 2 } ^ { 2 } / \gamma ^ { 2 } \big )$ with bandwidth $\gamma > 0$ (cf. Appendix B.2),

$$
\begin{array} { r } { S _ { k } \big ( \mathcal { N } ( \mu , \Sigma ) , y \big ) = \frac { 1 } { 2 } \Big ( 1 + \operatorname* { d e t } \big ( I + \frac { 4 } { 7 ^ { 2 } } \Sigma \big ) ^ { - 1 / 2 } \Big ) - \operatorname* { d e t } \big ( I + \frac { 2 } { \gamma ^ { 2 } } \Sigma \big ) ^ { - 1 / 2 } \exp \Big ( - ( y - \mu ) ^ { \top } \big ( \gamma ^ { 2 } I + 2 \Sigma \big ) ^ { - 1 } ( y - \mu ) \Big ) . } \end{array}\tag{15}
$$

Each needs one or two determinants and one quadratic form. We now show that these pass to the core. Recall that the deployed covariance (5) reads

$$
\begin{array} { r } { \Sigma _ { P } = \tilde { D } B ^ { * } \hat { \Sigma } B \tilde { D } , \qquad \tilde { D } = \mathrm { d i a g } \left( \sqrt { \tau / \Delta } \right) , \qquad \hat { \Sigma } = \mathrm { d i a g } ( s ) + \hat { U } \hat { U } ^ { * } , \quad \hat { U } : = B U , } \end{array}
$$

with the implied marginal variances $\Delta _ { i } = ( B ^ { * } \hat { \Sigma } B ) _ { i i }$ of (10), the physical factors $\pmb { U } = ( \pmb { u } _ { 1 } , \dots , \pmb { u } _ { r } )$ of (11), and $r = 0$ for a diagonal core. For the DFT with a diagonal core, $\pmb { \Delta } = \bar { s } \pmb { 1 } = \pmb { 1 }$ by the softmax normalization (9), so $\tilde { D } = D$ , the core is $C = F ^ { * } \operatorname { d i a g } ( s ) F$ , and the second stage scores $\mathcal { N } ( m , C )$ at Z. Both quantities that (14) have closed forms, such that

$$
\log \operatorname* { d e t } C = \sum _ { k } \log s _ { k } , \qquad ( Z - m ) ^ { \top } C ^ { - 1 } ( Z - m ) = \sum _ { k } \frac { w _ { k } } { s _ { k } } , \qquad w _ { k } : = \big | \big ( F ( Z - m ) \big ) _ { k } \big | ^ { 2 } ,\tag{16}
$$

with $w _ { k }$ the periodogram of the standardized residual, so that

$$
2 S _ { \mathrm { l o g } } \big ( \mathcal { N } ( m , C ) , Z \big ) \ = \ d \log 2 \pi + \sum _ { k } \Big ( \log s _ { k } + \frac { w _ { k } } { s _ { k } } \Big ) .\tag{17}
$$

Note that (17) is also known as the Whittle likelihood [54], which approximates the likelihood of a stationary Gaussian time series by replacing its Toeplitz covariance with a circulant one. In our setting it is exact, since the core is diagonal by construction, i.e., the back-transformed covariance is already circulant.

Since the standardization is a diagonal change of variables, scoring the deployed prediction instead of the core only adds the constant $\sum _ { i } \log \tau _ { i }$ fixed by the first stage, $\begin{array} { r } { 2 S _ { \mathrm { l o g } } \big ( \mathcal { N } ( \pmb { \mu } , D C D ) , \pmb { y } \big ) = 2 \hat { S } _ { \mathrm { l o g } } \big ( \mathcal { N } ( \pmb { m } , C ) , \pmb { Z } \big ) + \sum } \end{array}$ log τ<sub>i</sub>. For the log score, the two choices are therefore equivalent, at $\mathcal O ( d )$ beyond computing the DFT.

For the kernel score the situation is different. By Proposition 3.2 (i), the two choices correspond to kernel scores with different metrics, since D is not unitary. The second stage therefore scores the core at the standardized residual. Still, it does reduce, as the shift passes through the unitary F, that is

$$
\gamma ^ { 2 } I + c C = F ^ { \ast } \big ( \gamma ^ { 2 } I + c \operatorname { d i a g } ( s ) \big ) F , \qquad c \in \{ 2 , 4 \} .\tag{18}
$$

Both determinants of (15) become products and the linear solve becomes a division. Combining yields

$$
S _ { k } \big ( \mathcal { N } ( m , C ) , Z \big ) = \frac { 1 } { 2 } \bigg ( 1 + \prod _ { k } \big ( 1 + \frac { 4 s _ { k } } { \gamma ^ { 2 } } \big ) ^ { - 1 / 2 } \bigg ) - \prod _ { k } \big ( 1 + \frac { 2 s _ { k } } { \gamma ^ { 2 } } \big ) ^ { - 1 / 2 } \exp \bigg ( - \sum _ { k } \frac { w _ { k } } { \gamma ^ { 2 } + 2 s _ { k } } \bigg ) ,\tag{19}
$$

again $\mathcal O ( d )$ beyond computing the DFT. Including the transform, one evaluation of either objective costs ${ \mathcal { O } } ( d \log d )$ via the FFT, with $\mathcal O ( d )$ memory for the spectrum and the marginals, cf. SCORE in Table 1.

General transforms and low-rank cores. The log score keeps its closed form for every unitary B and any rank r. Again, the renormalization in (12) is a diagonal change of variables, so it contributes only the additive term $2 \sum _ { i }$ log $\ddot { D } _ { i i }$ , and unitarity of B gives

$$
\log \operatorname * { d e t } \pmb { \Sigma } _ { P } = 2 \sum _ { i } \log \tilde { D } _ { i i } + \log \operatorname * { d e t } \hat { \mathbf { \Sigma } } , \qquad \pmb { \Sigma } _ { P } ^ { - 1 } = \tilde { D } ^ { - 1 } B ^ { * } \hat { \Sigma } ^ { - 1 } B \tilde { D } ^ { - 1 } .\tag{20}
$$

A low-rank core keeps that structure through the matrix determinant lemma and the Woodbury identity,

$$
\operatorname* { d e t } \hat { \Sigma } = \Big ( \prod _ { k } s _ { k } \Big ) \operatorname* { d e t } \bigl ( I _ { r } + \hat { U } ^ { * } \operatorname { d i a g } ( s ) ^ { - 1 } \hat { U } \bigr ) ,\tag{21}
$$

$$
\hat { \pmb { \Sigma } } ^ { - 1 } = \mathrm { d i a g } ( \pmb { s } ) ^ { - 1 } - \mathrm { d i a g } ( \pmb { s } ) ^ { - 1 } \hat { U } \big ( \pmb { I } _ { r } + \hat { U } ^ { * } \mathrm { d i a g } ( \pmb { s } ) ^ { - 1 } \hat { U } \big ) ^ { - 1 } \hat { U } ^ { * } \mathrm { d i a g } ( \pmb { s } ) ^ { - 1 } ,\tag{22}
$$

at the price of r further transforms for ${ \hat { U } } , { \mathcal { O } } ( r ^ { 2 } d )$ to assemble the $r \times r$ matrix, and $\mathcal { O } ( r ^ { 3 } )$ to factorize it. The kernel score does not extend likewise.

Sampling. In the DFT case with diagonal core, i.e. $B = F$ and $r = 0 .$ , a draw is formed by

$$
{ \pmb y } = { \pmb \mu } + { \pmb D } { \pmb F } ^ { * } \mathrm { d i a g } ( \sqrt { s } ) { \pmb F } \xi , \qquad { \pmb \xi } \sim \mathcal { N } ( { \bf 0 } , { \pmb I } _ { d } ) ,
$$

at ${ \mathcal { O } } ( d \log d )$ with the FFT. The symmetrization in Appendix C.2.1 enforces $s _ { d - k } = s _ { k }$ , which makes the square root, and hence y, real. In general, draws come from (12) the same way,

$$
y = \mu + { \tilde { D } } { \big ( } B ^ { * } \operatorname { d i a g } ( { \sqrt { s } } ) B \xi _ { 1 } + U \xi _ { 2 } { \big ) } , \qquad \xi _ { 1 } \sim { \mathcal { N } } ( \mathbf { 0 } , I _ { d } ) , \quad \xi _ { 2 } \sim { \mathcal { N } } ( \mathbf { 0 } , I _ { r } ) { \mathrm { ~ i n d e p e n d e n t , } }
$$

at $\mathcal { O } ( d \log d + d r )$ each for the DFT.

Bandwidth selection. The kernel score leaves the choice of the bandwidth $\gamma > 0$ open. We set it by (the mean version of) the median heuristic [15, 55], which matches $\gamma ^ { 2 } / 2$ to the expected squared distance between two independent draws from the predictive distribution at a fixed input. In the second stage the scored quantity is the standardized residual, whose covariance is a correlation matrix, and the heuristic then resolves without data dependence.

Lemma C.1 (Bandwidth in standardized coordinates). Let R be a correlation matrix on $\mathbb { R } ^ { d }$ and let $\boldsymbol { Z } , \boldsymbol { Z } ^ { \prime }$ be independent drawsfrom $\mathcal { N } ( \boldsymbol { m } , R )$ . Then, with the expectation taken over $( Z , Z ^ { \prime } )$

$$
\begin{array} { r } { \mathbb { E } \| Z - Z ^ { \prime } \| ^ { 2 } = 2 \operatorname { t r } R = 2 d , } \end{array}
$$

so matching $\gamma ^ { 2 } / 2$ to this expectation gives $\gamma ^ { 2 } = 4 d ,$ independently of R, of the target, and of the input.

Proof. $Z - Z ^ { \prime } \sim \mathcal { N } ( \mathbf { 0 } , 2 R )$ , hence $\begin{array} { r } { \mathbb { E } \| Z - Z ^ { \prime } \| ^ { 2 } = \mathrm { t r } ( 2 R ) = 2 d \mathrm { s i n c e } R _ { i i } = 1 } \end{array}$ , as it is a correlation matrix. □

The median heuristic only sets an order of magnitude. In practice, we therefore average the second-stage loss over a ladder $\gamma ^ { 2 } = f d , f \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 5 \}$ , around this value, which keeps the score proper. We always evaluate at $\gamma ^ { 2 } = d$ in Appendix G.

## D EXPRESSIVITY OF SCORE

Proposition 3.4 states that under exact marginals SCORE fits the correlation alone, as a projection of $R _ { Q }$ onto the representable class in the respective divergence. Here, we quantify the misspecification that remains within the Gaussian class. Everything is stated at a fixed input $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ , as elsewhere, since the representable class is the same at every input, and we take $B = { \pmb F }$ throughout. Recall that $Q = \mathcal { N } ( \mu _ { Q } , \Sigma _ { Q } )$ with $\Sigma _ { \boldsymbol { Q } } \succ 0$ denotes the target, factored as $\Sigma _ { Q } = D _ { Q } R _ { Q } { \cal D } _ { Q }$ as in (4).

Besides our default diagonal core, we also consider the low-rank extension of (11). Diagonal and diagonal-plus-low-rank cores of rank r give the correlation classes

$$
\mathcal { M } ^ { ( r ) } : = \big \{ \mathrm { c o r r } \big ( \boldsymbol { F } ^ { * } \hat { \boldsymbol { \Sigma } } \boldsymbol { F } \big ) : \hat { \boldsymbol { \Sigma } } \in \mathcal { S } ^ { ( r ) } \big \} , \qquad \mathcal { S } ^ { ( r ) } : = \big \{ \mathrm { d i a g } ( s ) + \boldsymbol { U } \boldsymbol { U } ^ { * } : s \in \mathbb { R } _ { > 0 } ^ { d } , \boldsymbol { U } \in \mathbb { C } ^ { d \times r } \big \} ,
$$

restricted to their real members. We write $\mathcal { M } : = \mathcal { M } ^ { ( 0 ) }$ for the diagonal case.

We call the quantity of interest the approximation error,

$$
\varepsilon _ { D } ^ { ( r ) } ( Q ) : = \operatorname* { i n f } _ { R \in \mathcal { M } ^ { ( r ) } } D \big ( \mathcal { N } ( \mathbf { 0 } , R ) , \mathcal { N } ( \mathbf { 0 } , R _ { Q } ) \big ) ,\tag{23}
$$

which is the error $\varepsilon _ { \mathrm { a p p } }$ of Section 3.3 at the input where the target is $Q ,$ with the divergence D of the respective score and the rank r of the core made explicit. We drop the superscript $( r )$ whenever $r = 0 . { \mathrm { A s } }$ before, we treat the Gaussian kernel score and the log score, whose divergences are $\scriptstyle { \frac { 1 } { 2 } } \mathrm { M M D } ^ { 2 }$ for the Gaussian kernel $k ( \pmb { u } , \pmb { v } ) = \mathrm { e x p } \big ( - \| \pmb { u } - \pmb { v } \| ^ { 2 } / \gamma ^ { 2 } \big )$ and the Kullback–Leibler divergence KL $\left( \mathcal { N } ( \mathbf { 0 } , R _ { Q } ) \| \bar { \mathcal { N } } ( \mathbf { 0 } , R ) \right)$ , respectively.

Gaussian kernel score. Consider the Gaussian kernel with bandwidth $\gamma > 0$ . It turns out that its approximation error is controlled by how well the class approximates the true correlation $R _ { Q }$ in Frobenius norm,

$$
\Psi ^ { ( r ) } ( Q ) : = \operatorname* { i n f } _ { R \in \mathcal { M } ^ { ( r ) } } \left\| R - R _ { Q } \right\| _ { F } ^ { 2 } ,
$$

and in the diagonal case this infimum is the off-diagonal energy of the target in the Fourier domain.

Lemma D.1 (Expressivity under the kernel score). Let $0 \leq r < d .$

(i) For every $\gamma > 0 ,$ , the approximation error is equivalent, up to constants, to the Frobenius projection error $\Psi ^ { ( r ) } ( Q )$

$$
\big ( 1 + { \textstyle \frac { 8 d } { \gamma ^ { 2 } } } \big ) ^ { - { \frac { d + 4 } { 2 } } } \displaystyle \frac { \Psi ^ { ( r ) } ( Q ) } { \gamma ^ { 4 } } \leq \varepsilon _ { \mathrm { M M D } } ^ { ( r ) } ( Q ) \leq \frac { \Psi ^ { ( r ) } ( Q ) } { \gamma ^ { 4 } } ,\tag{24}
$$

so it vanishes if and only if the projection error does, and $\gamma ^ { 4 } \varepsilon _ { \mathrm { M M D } } ^ { ( r ) } ( Q )  \Psi ^ { ( r ) } ( Q ) a s \gamma  \infty .$

(ii) For $r = 0$ the projection is attained at $R _ { F } ^ { \star } = F ^ { \star } \mathrm { d i a g } ( t ^ { \star } ) F$ with $t _ { k } ^ { \star } = ( F R _ { Q } { \cal F } ^ { \ast } ) _ { k k } ,$ , and

$$
\Psi ( Q ) = \sum _ { k \neq l } { \bigl | } ( F R _ { Q } F ^ { * } ) _ { k l } { \bigr | } ^ { 2 } .
$$

Proof. (i) By Bochner’s theorem, $k ( \pmb { x } , \pmb { y } ) = \mathbb { E } e ^ { i \omega ^ { \top } ( \pmb { x } - \pmb { y } ) }$ with ${ \boldsymbol \omega } \sim \mathcal { N } ( { \bf 0 } , { \frac { 2 } { \gamma ^ { 2 } } } I )$ , so $\mathrm { M M D } _ { k } ^ { 2 } = \mathbb { E } | \phi _ { R } ( \omega ) - \phi _ { R _ { Q } } ( \omega ) | ^ { 2 }$ for the characteristic functions $\begin{array} { r } { \phi _ { R } ( \omega ) = e ^ { - \omega ^ { \top } R \omega / 2 } } \end{array}$ of the two centered Gaussians. Substituting $\omega = \sqrt { 2 } z / \gamma$

$$
\mathrm { M M D } _ { k } ^ { 2 } = \mathbb { E } _ { z \sim \mathcal { N } ( \mathbf { 0 } , I ) } { \left( e ^ { - a } - e ^ { - b } \right) } ^ { 2 } , \qquad a = \frac { z ^ { \top } R z } { \gamma ^ { 2 } } , \quad b = \frac { z ^ { \top } R _ { Q } z } { \gamma ^ { 2 } } .
$$

Both R and $R _ { Q }$ are real correlation matrices, so $N : = R - R _ { Q }$ is symmetric and $\begin{array} { r } { \mathbb { E } ( z ^ { \top } N z ) ^ { 2 } = 2 \| N \| _ { F } ^ { 2 } + ( \mathrm { t r } N ) ^ { 2 } = } \end{array}$ $2 \| \pmb { N } \| _ { F } ^ { 2 }$ . Since $a , b \geq 0 ;$ , the mean value theorem gives $e ^ { - a } - e ^ { - b } = - ( a - b ) e ^ { - \xi }$ with ξ between a and $b ,$ hence

$$
e ^ { - 2 ( a + b ) } ( a - b ) ^ { 2 } \ \leq \ \left( e ^ { - a } - e ^ { - b } \right) ^ { 2 } \ \leq \ ( a - b ) ^ { 2 } = { \frac { \left( z ^ { \top } N z \right) ^ { 2 } } { \gamma ^ { 4 } } } .
$$

Taking expectations, the right inequality yields $\mathrm { M M D } _ { k } ^ { 2 } \leq 2 \Vert N \Vert _ { F } ^ { 2 } / \gamma ^ { 4 }$ . For the left one, the largest eigenvalue of a correlation matrix is at most its trace $d ,$ so $a + b \leq 2 d \| z \| ^ { 2 } / \gamma ^ { 2 }$ and, with $\beta : = 1 + 8 d / \gamma ^ { 2 }$

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \big [ e ^ { - 2 ( a + b ) } ( a - b ) ^ { 2 } \big ] \geq \frac { 1 } { \gamma ^ { 4 } } \int \frac { e ^ { - \| z \| ^ { 2 } / 2 } } { ( 2 \pi ) ^ { d / 2 } } e ^ { - 4 d \| z \| ^ { 2 } / \gamma ^ { 2 } } \big ( z ^ { \top } N z \big ) ^ { 2 } \mathrm { d } z } \\ { = \frac { \beta ^ { - d / 2 } } { \gamma ^ { 4 } } \mathbb { E } _ { z \sim \mathcal { N } ( \mathbf { 0 } , \beta ^ { - 1 } I ) } \big ( z ^ { \top } N z \big ) ^ { 2 } = \frac { 2 \beta ^ { - \frac { d + 4 } { 2 } } \| N \| _ { F } ^ { 2 } } { \gamma ^ { 4 } } , } \end{array}
$$

since $e ^ { - \| z \| ^ { 2 } / 2 } e ^ { - 4 d \| z \| ^ { 2 } / \gamma ^ { 2 } } = e ^ { - \beta \| z \| ^ { 2 } / 2 } \mathrm { i } \mathrm { s } \beta ^ { - d / 2 }$ times the density of $\mathcal { N } ( \mathbf { 0 } , \beta ^ { - 1 } I )$ , and under that law $z = \beta ^ { - 1 / 2 } z ^ { \prime }$ with $z ^ { \prime }$ standard normal, so the second moment of the quadratic form is $\beta ^ { - 2 } \cdot 2 \| \boldsymbol { N } \| _ { F } ^ { 2 }$

Hence MM $\begin{array} { r } { \mathrm { ) } _ { k } ^ { 2 } \ge 2 \bigl ( 1 + \frac { 8 d } { \gamma ^ { 2 } } \bigr ) ^ { - \frac { d + 4 } { 2 } } \| { \cal N } \| _ { F } ^ { 2 } / \gamma ^ { 4 } } \end{array}$ . Both bounds are monotone in $\| \boldsymbol { N } \| _ { F } ^ { 2 }$ , so taking the infimum over $\mathcal { M } ^ { ( r ) }$ and the factor $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ of $\begin{array} { r } { D = \frac { 1 } { 2 } \mathrm { M M D } _ { k } ^ { 2 } } \end{array}$ proves (24). Finally, with $c ( \gamma ) : = \beta ^ { - \frac { d + 4 } { 2 } }$ , multiplying (24) by $\gamma ^ { 4 }$ gives

$$
c ( \gamma ) \Psi ^ { ( r ) } ( Q ) \leq \gamma ^ { 4 } \varepsilon _ { \mathrm { M M D } } ^ { ( r ) } ( Q ) \leq \Psi ^ { ( r ) } ( Q ) ,
$$

and $c ( \gamma ) \to 1$ as $\gamma \to \infty$ , which proves the limit.

(ii) Since $| F _ { k i } | ^ { 2 } = 1 / d ,$ , the deployed core is $F ^ { * } \log ( t ) F$ with $\pmb { t } = \pmb { s } / \bar { s }$ , and by unitary invariance

$$
\| R - R _ { Q } \| _ { F } ^ { 2 } = \| \mathrm { d i a g } ( t ) - F R _ { Q } { \cal F } ^ { * } \| _ { F } ^ { 2 } = \sum _ { k } \left( t _ { k } - ( F R _ { Q } { \cal F } ^ { * } ) _ { k k } \right) ^ { 2 } + \sum _ { k \ne l } \lvert ( F R _ { Q } { \cal F } ^ { * } ) _ { k l } \rvert ^ { 2 } ,
$$

whose first sum vanishes at $t _ { k } = ( F R _ { Q } F ^ { * } ) _ { k k }$ . Note that this choice is admissible. Indeed, $t _ { k } ^ { \star } > 0$ since $R _ { Q } ~ \succ ~ 0$ $\begin{array} { r } { \bar { t } ^ { \star } = \frac { 1 } { d } \operatorname { t r } ( { \cal F } R _ { Q } { \cal F } ^ { * } ) = \frac { 1 } { d } \operatorname { t r } R _ { Q } = 1 , \mathrm { a n d } \dot { t } _ { d - k } ^ { \star } = t _ { k } ^ { \star } } \end{array}$ because $R _ { Q }$ is real. What remains is the off-diagonal energy. □

Part (i) uses nothing about F and holds for every unitary B. Its lower constant decays exponentially in d. At the training bandwidth $\gamma ^ { 2 } = 4 d$ it equals $3 ^ { - ( d + 4 ) / 2 }$ , already of order $1 0 ^ { - 2 4 } { \mathrm { ~ a t ~ } } d = 9 6 { \mathrm { ~ } }$ . Although it can be refined to a dimensionindependent constant at this bandwidth (Lemma $\mathrm { E } . 7 ( \mathrm { i i } ) )$ , the sandwich should be read as a qualitative statement, establishing that approximation error and Frobenius projection vanish together.

Logarithmic score. For the log score, it turns out that the approximation error comes down to the spread of the eigenvalues of the target correlation $R _ { Q }$ after transforming and renormalizing it to corr $\left( F R _ { Q } { \pmb F } ^ { * } \right)$ .

Lemma D.2 (Diagonal core under the log score). Denote by $\lambda _ { 1 } , . . . , \lambda _ { d } > 0$ the eigenvalues of $\operatorname { \dot { } c o r r } ( F R _ { Q } F ^ { * } )$ and by $\begin{array} { r } { \mathrm { G M } : = ( \prod _ { k } \lambda _ { k } ) ^ { 1 / d } } \end{array}$ their geometric mean. Then

$$
\frac { 1 } { d } \varepsilon _ { \mathrm { K L } } ( Q ) = - \frac { 1 } { 2 d } \log \operatorname * { d e t } \mathrm { c o r r } ( F R _ { Q } { \cal F } ^ { * } ) = - \frac { 1 } { 2 } \log \mathrm { G M } .\tag{25}
$$

The infimum is attained at $R ^ { \star } = R _ { F } ^ { \star } = F ^ { * } \mathrm { d i a g } ( t ^ { \star } ) F$ with $t _ { k } ^ { \star } = ( F R _ { Q } F ^ { \ast } ) _ { k k }$ , the Frobenius projection of Lemma $D . I \left( i i \right) ,$ and this minimizer is unique. In particular, the error vanishes if and only if F diagonalizes $R _ { Q }$

Proof. The correlation normalization maps a spectrum s to $\mathbfit { \Delta } t = \mathbfit { \Delta } s / \bar { s }$ , so up to the scale gauge $s \mapsto c s$ the class is $\mathcal { M } = \{ F ^ { * } \mathrm { d i a g } ( t ) F : t > 0 , \bar { t } = 1 \}$ . The divergence of the log score is $\mathrm { K L } \bar { \bf \Phi } ( { \mathcal N } ( { \bf 0 } , R _ { Q } ) | | { \mathcal N } ( { \bf 0 } , R ) ) = \textstyle \frac { 1 } { 2 } \big [ \mathrm { t r } ( R ^ { - 1 } R _ { Q } ) -$ $d + \log$ det $R - \log \operatorname* { d e t } R _ { Q } ]$ , and trace and determinant are invariant under conjugation by $F _ { \mathrm { { ; } } }$ , so for $R = F ^ { * } \operatorname { d i a g } ( t ) F$ this equals

$$
{ \scriptstyle { \frac { 1 } { 2 } } } \sum _ { k } \bigg [ { \frac { ( F R _ { Q } { F ^ { * } } ) _ { k k } } { t _ { k } } } + \log t _ { k } \bigg ] - { \scriptstyle { \frac { 1 } { 2 } } } \big [ d + \log \operatorname* { d e t } R _ { Q } \big ] .
$$

The sum is separable and strictly convex in log $t _ { k } ,$ with minimizer $t _ { k } ^ { \star } = ( F R _ { Q } F ^ { \ast } ) _ { k k }$ . It is positive since $R _ { Q } \succ 0$ , satisfies $\begin{array} { r } { \bar { t } ^ { \star } = \frac { 1 } { d } \operatorname { t r } ( F \bar { R } _ { Q } { F } ^ { * } ) = \frac { 1 } { d } \operatorname { t r } { R _ { Q } } = 1 } \end{array}$ , and obeys $t _ { d - k } ^ { \star } = t _ { k } ^ { \star }$ since $R _ { Q }$ is real. Hence $R ^ { \star }$ is admissible, and by strict convexity it is the unique minimizer.

At $t ^ { \star }$ each term $( { F R _ { Q } } { F ^ { * } } ) _ { k k } / t _ { k } ^ { \star }$ equals one, so the first sum contributes $\begin{array} { r } { d + \sum _ { k } \log ( F R _ { Q } F ^ { * } ) _ { k k } } \end{array}$ and the value is $\begin{array} { r } { \frac 1 2 \big \lceil \sum _ { k } \log ( F R _ { Q } { \cal F } ^ { * } ) _ { k k } - \log } \end{array}$ det $R _ { Q } ] \ = \ - \frac { 1 } { 2 }$ log det corr $\begin{array} { r } { ( F R _ { Q } { F } ^ { * } ) = - \frac { 1 } { 2 } \sum _ { k } \overline { { \log } } \lambda _ { k } = - \frac { \overline { { d } } } { 2 } } \end{array}$ log GM, which is (25). Since corr( $( F R _ { Q } { \pmb F } ^ { * } )$ has unit diagonal, $\begin{array} { r } { \frac { 1 } { d } \sum _ { k } ^ { \sim } \lambda _ { k } = 1 } \end{array}$ , so $\mathrm { G M } \le 1$ with equality if and only if all $\lambda _ { k } = 1$ , that is, if and only if $F R _ { Q } F ^ { * }$ is diagonal. □

Summarizing, in the diagonal case both approximation errors penalize the off-diagonal energy of $F R _ { Q } F ^ { * }$ , the kernel score directly through (24) and the log score through the spectral flatness GM of corr $\left( F R _ { Q } { \pmb F } ^ { * } \right)$ . They agree on when the error vanishes, namely when F diagonalizes $R _ { Q }$ , but differ in how they scale what remains.

## E PAC LEARNABILITY

Proposition 3.4 describes the population limit of SCORE under an exact first stage. At finite sample size the first stage is not exact, and stage 2 is fitted to residuals standardized by the learned marginals. One might worry that the second stage amplifies the first-stage error. This section shows that it does not. Instead, the error enters additively, through terms that are linear in the marginal errors and independent of the second-stage sample and of the core class. We restrict ourselves to the case the paper focuses on, which is the DFT with a diagonal core $( \boldsymbol { B } = \boldsymbol { F } , \boldsymbol { r } = 0 )$ , the CRPS in the first stage, and the Gaussian kernel score in the second. As is usual in PAC analyses, we work under idealized conditions. In particular, we assume exact empirical risk minimization and disjoint samples of sizes N and M for the two stages. The latter is a standard device for two-step estimators whose second step use the output of a learned first step [56, 57].

## E.1 SETTING AND ASSUMPTIONS

Let $N , M \in \mathbb { N }$ be the sample budgets of the first and second stage, respectively, and let $( X _ { n } , Y _ { n } ) , n \leq N + M$ , be independent draws from $\mathbb { P } = \mathbb { P } _ { ( \pmb { X } , \pmb { Y } ) }$ . We write $\mathcal { T } _ { N } : = ( X _ { n } , Y _ { n } ) _ { n = 1 } ^ { N }$ and $\mathcal { T } _ { M } : = ( X _ { n } , Y _ { n } ) _ { n = N + 1 } ^ { N + M }$ for the samples used in the two stages. We assume that the conditional law is Gaussian,

$$
Q ( \pmb { x } ) : = \mathbb { P } _ { \pmb { Y } | \pmb { X } = \pmb { x } } = \mathcal { N } \big ( \mu _ { Q } ( \pmb { x } ) , \pmb { \Sigma } _ { Q } ( \pmb { x } ) \big ) , \qquad \pmb { \Sigma } _ { Q } ( \pmb { x } ) \succeq 0 , \quad \pmb { x } \in \mathcal { X } ,
$$

with marginal variances $\tau _ { Q } ( \pmb { x } ) : = \left( ( \pmb { \Sigma } _ { Q } ( \pmb { x } ) ) _ { i i } \right) _ { i = 1 } ^ { d } . \mathrm { A s }$ in (4), we write $\pmb { \Sigma } _ { Q } ( \pmb { x } ) = \pmb { D } _ { Q } ( \pmb { x } ) \pmb { R } _ { Q } ( \pmb { x } ) \pmb { D } _ { Q } ( \pmb { x } )$ with marginal scales $D _ { Q } ( \pmb { x } ) = \mathrm { d i a g } \left( \sqrt { \tau _ { Q } ( \pmb { x } ) } \right)$ and correlation $R _ { Q } ( { \pmb x } )$ , and omit x from the notation whenever a statement is pointwise in the input. Fitted quantities carry the size of the sample they were fitted on as a subscript, so $( \mu _ { N } , \tau _ { N } )$ come from the

first stage and $R _ { M }$ from the second. For $\gamma > 0$ let $k _ { \gamma }$ be the Gaussian kernel $k _ { \gamma } ( \pmb { u } , \pmb { v } ) : = \exp \big ( - \| \pmb { u } - \pmb { v } \| _ { 2 } ^ { 2 } / \gamma ^ { 2 } \big )$ , and for probability measures $P , P ^ { \prime }$ on $\mathbb { R } ^ { d }$ write

$$
\operatorname { M M D } _ { \gamma } ^ { 2 } ( P , P ^ { \prime } ) : = \mathbb { E } k _ { \gamma } ( U , U ^ { \prime } ) + \mathbb { E } k _ { \gamma } ( V , V ^ { \prime } ) - 2 \mathbb { E } k _ { \gamma } ( U , V ) ,
$$

with $U , U ^ { \prime } \sim P$ and $V , V ^ { \prime } \sim P ^ { \prime }$ all independent, so that the divergence of the kernel score $S _ { k _ { \gamma } }$ is $\begin{array} { r } { D _ { \gamma } = \frac { 1 } { 2 } \mathrm { M M D } _ { \gamma } ^ { 2 } } \end{array}$ . In our analysis we use the bandwidth $\widetilde { \gamma } ^ { 2 } = 4 d ,$ , as motivated by Lemma C.1, to fit the core to the standardized residual in the second stage. We restrict the target to a bounded range.

Assumption E.1 (Target). There are $B > 0$ and $0 < \tau _ { \mathrm { m i n } } \le \tau _ { \mathrm { m a x } } <$ ∞ such that $\pmb { \mu } _ { Q } ( \pmb { x } ) \in [ - B , B ] ^ { d }$ and ${ \pmb { \tau } } _ { Q } ( { \pmb x } ) \in$ $[ \tau _ { \operatorname* { m i n } } , \tau _ { \operatorname* { m a x } } ] ^ { d }$ for every $\mathbf { \boldsymbol { x } } \in \mathcal { X }$

Admissible cores. For the DFT and a diagonal core, the second stage optimizes over spectra $s \in \Theta$ . Indexing the DFT frequencies by $k = 0 , \ldots , d - 1$ , we define the set of admissible spectra by

$$
\begin{array} { r } { \Theta : = \Big \{ s \in \mathbb { R } ^ { d } : s _ { k } \geq 0 , \quad \frac { 1 } { d } \sum _ { k } s _ { k } = 1 , \quad s _ { d - k } = s _ { k } \mathrm { ~ f o r ~ } k = 1 , \dots , d - 1 \Big \} . } \end{array}
$$

Note that $\Theta$ is the intersection of the scaled simplex $\begin{array} { r } { \{ \pmb { s } \geq \mathbf { 0 } , ~ \sum _ { k } s _ { k } = d \} } \end{array}$ with the linear subspace of reflection-symmetric vectors, hence convex and compact. With F the unitary DFT on the d grid points, the core with spectrum s is

$$
R _ { s } : = F ^ { * } \mathrm { d i a g } ( s ) F , \qquad s \in \Theta ,\tag{26}
$$

and the reflection condition makes $R _ { s }$ real. Moreover, the condition $\begin{array} { r } { \frac { 1 } { d } \sum _ { k } s _ { k } = 1 } \end{array}$ together with $| F _ { k i } | ^ { 2 } = 1 / d$ gives it unit diagonal, so $R _ { s }$ is a correlation matrix.

The network. Both stages optimize over maps realized by a single network, $\phi \mapsto \left( \mu _ { \phi } , \tau _ { \phi } , \pmb { s } _ { \phi } \right)$ with $\phi \in \mathcal W$ the parameters. We write

$$
\mathcal { G } _ { 1 } = \big \{ ( \pmb { \mu } _ { \phi } , \pmb { \tau } _ { \phi } ) : \phi \in \mathcal { W } \big \} , \qquad \mathcal { G } _ { 2 } = \big \{ \pmb { s } _ { \phi } : \phi \in \mathcal { W } \big \} ,
$$

for the two classes, with $( \pmb { \mu } _ { \phi } , \pmb { \tau } _ { \phi } ) : \mathcal { X }  \mathbb { R } ^ { d } \times \mathbb { R } _ { > 0 } ^ { d }$ and $\pmb { s } _ { \phi } : \mathcal { X }  \Theta$ . The restriction of the image of ${ \pmb s } _ { \phi }$ to Θ can be implemented by a softmax normalization, and the reflection condition by emitting a reflection-symmetric spectrum, as discussed in Appendix C.2.1. The analysis treats $\mathcal { G } _ { 1 }$ and $\mathcal { G } _ { 2 }$ as separate classes, which corresponds to separate heads for $( \mu , \tau )$ and s. We assume the following regularity conditions on the network.

Assumption E.2 (Network class). Let the parameter space be $\mathcal { W } : = [ - W , W ] ^ { q }$ for some $W > 0$ and $q \in \mathbb { N } ,$ , and let every $\phi \in \mathcal W$ give measurable maps $\mu _ { \phi } , \tau _ { \phi } , s _ { \phi }$ on X. We assume:

(i) the network stays within the range of the target,

$$
\mu _ { \phi } ( \boldsymbol { \mathcal { X } } ) \subset [ - B , B ] ^ { d } , \qquad \tau _ { \phi } ( \boldsymbol { \mathcal { X } } ) \subset [ \tau _ { \operatorname* { m i n } } , \tau _ { \operatorname* { m a x } } ] ^ { d } \quad \mathrm { f o r ~ e v e r y ~ } \phi \in \mathcal { W } ,
$$

with $B , \tau _ { \mathrm { m i n } } , \tau _ { \mathrm { m a x } }$ as in Assumption E.1;

(ii) the class $\mathcal { G } _ { 1 }$ contains the first-stage optimum,

$$
( \mu _ { Q } , \tau _ { Q } ) \in \mathcal G _ { 1 } ;
$$

(iii) the marginals are Lipschitz in the weights, that is, for some $L _ { 1 } > 0$ and all $\phi , \phi ^ { \prime } \in \mathcal { W }$

$$
\operatorname* { m a x } \Big \{ \operatorname* { s u p } _ { x } \big \| \mu _ { \phi } ( x ) - \mu _ { \phi ^ { \prime } } ( x ) \big \| _ { \infty } , \operatorname* { s u p } _ { x } \big \| \tau _ { \phi } ( x ) - \tau _ { \phi ^ { \prime } } ( x ) \big \| _ { \infty } \Big \} \ \leq L _ { 1 } \| | \phi - \phi ^ { \prime } | | _ { \infty } ;
$$

(iv) the spectrum is Lipschitz in the weights, that ${ \mathrm { i s } } ,$ for some $L _ { 2 } > 0$ and all $\phi , \phi ^ { \prime } \in \mathcal { W }$

$$
\operatorname* { s u p } _ { \pmb { x } } \left\| \pmb { s } _ { \phi } ( \pmb { x } ) - \pmb { s } _ { \phi ^ { \prime } } ( \pmb { x } ) \right\| _ { \infty } \leq L _ { 2 } \| \phi - \phi ^ { \prime } \| _ { \infty } ;
$$

(v) the class contains a second-stage optimum, that is, at the bandwidth $\widetilde { \gamma } ^ { 2 } = 4 d$ a measurable selection

$$
\begin{array} { r } { s ^ { \star } ( { \pmb x } ) \ \in \ \underset { s \in \Theta } { \operatorname { a r g m i n } } \frac { 1 } { 2 } \mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \big ( { \mathcal N } ( \mathbf { 0 } , R _ { Q } ( { \pmb x } ) ) , { \mathcal N } ( \mathbf { 0 } , R _ { s } ) \big ) , \qquad { \pmb x } \in { \mathcal X } , } \end{array}
$$

lies in $\mathcal { G } _ { 2 }$

For a class G of maps on $\mathcal { X }$ and $\eta > 0$ , we call a finite set $\mathcal { C } \subset \mathcal { G }$ an η-net if for every $g \in { \mathcal { G } }$ there is $g ^ { \prime } \in { \mathcal { C } }$ with $\begin{array} { r } { \operatorname* { s u p } _ { \pmb { x } } \| g ( \pmb { x } ) - g ^ { \prime } ( \pmb { x } ) \| _ { \infty } \leq \eta , } \end{array}$ , where $\| \cdot \| _ { \infty }$ is the maximum over all output coordinates. The covering number $\mathfrak { N } ( \mathcal { G } , \eta )$ is defined as the smallest cardinality of such a net. We abbreviate $\mathfrak { N } _ { j } ( \eta ) : = \mathfrak { N } ( \mathcal { G } _ { j } , \eta )$ for stages $j = 1 , 2$

Lemma E.1 (Covering numbers of the two classes). Under Assumption E.2,for every $\eta > 0$

$$
\log \mathfrak { N } _ { 1 } ( \eta ) \le q \log \Big ( 1 + \frac { 2 W L _ { 1 } } { \eta } \Big ) , \qquad \log \mathfrak { N } _ { 2 } ( \eta ) \le q \log \Big ( 1 + \frac { 2 W L _ { 2 } } { \eta } \Big ) .
$$

Proof. The cube $\mathcal { W } = [ - W , W ] ^ { q }$ admits an ε-net in $\| \cdot \| _ { \infty }$ of cardinality at most $( 1 + 2 W / \varepsilon ) ^ { q }$ with centres in W. Indeed, the grid $\left\{ - W + \varepsilon , - W + 3 \varepsilon , \dots \right\}$ of spacing 2ε has at most $\lceil W / \varepsilon \rceil \leq 1 + 2 W / \varepsilon$ points per coordinate, and every point of $[ - W , W ]$ lies within ε of one of them. If the last grid point exceeds $W$ , we replace it by W, which preserves this property. Take $\varepsilon = \eta / L _ { j }$ and map the grid through $\phi \mapsto g _ { \phi }$ . By the Lipschitz property of Assumption ${ \mathrm { E } } . 2 ( \mathrm { i i i } ) { - } ( \mathrm { i v } )$ , the images form an η-net of $\mathcal { G } _ { j }$ , whose cardinality gives the stated bound on $\mathfrak { N } _ { j } ( \eta )$ □

Stage 1. For parameter values $( \pmb { \mu } , \pmb { \tau } ) \in [ - B , B ] ^ { d } \times [ \tau _ { \operatorname* { m i n } } , \tau _ { \operatorname* { m a x } } ] ^ { d }$ , the range of Assumptions E.1 and $\mathrm { E } . 2 ( \mathrm { i } )$ , and an observation $\pmb { y } \in \mathbb { R } ^ { d }$ let

$$
\ell _ { 1 } ( \pmb { \mu } , \pmb { \tau } ; \pmb { y } ) : = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \mathrm { C R P S } \big ( \mathcal { N } ( \mu _ { i } , \tau _ { i } ) , y _ { i } \big )
$$

be the stage-1 loss. For $g \in { \mathcal { G } } _ { 1 }$ we evaluate it at $g ( { \pmb x } ) = ( \pmb { \mu } ( \pmb { x } ) , \pmb { \tau } ( \pmb { x } ) )$ and write $\ell _ { 1 } ( g ( { \pmb x } ) ; { \pmb y } )$ . Risk and empirical risk over $\mathcal { T } _ { N }$ are

$$
\mathcal { R } _ { 1 } ( g ) : = \mathbb { E } _ { ( X , Y ) } \ell _ { 1 } \big ( g ( X ) ; Y \big ) , \qquad \mathcal { R } _ { 1 , N } ( g ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \ell _ { 1 } \big ( g ( X _ { n } ) ; Y _ { n } \big ) ,
$$

and, under the idealization of exact empirical risk minimization, stage 1 returns a minimizer

$$
( \pmb { \mu } _ { N } , \pmb { \tau } _ { N } ) \in \arg \operatorname* { m i n } _ { g \in \mathcal { G } _ { 1 } } \mathcal { R } _ { 1 , N } ( g ) ,
$$

which exists because $\mathcal { G } _ { 1 }$ is compact, being the image of the compact $\mathcal { W }$ under the map $\phi \mapsto \left( \mu _ { \phi } , \tau _ { \phi } \right)$ , which is continuous by Assumption E.2 (iii). Over all measurable maps the population minimizer of $\mathcal { R } _ { 1 }$ is $( \mu _ { Q } , \tau _ { Q } )$ , by Proposition 3.4 (i) applied at each input, and by Assumption E.2 (ii) it lies in $\mathcal { G } _ { 1 }$ and is therefore the minimizer over the class as well. At an input x we measure the distance of the fit $( \mu _ { N } , \tau _ { N } )$ in the original coordinates by

$$
\begin{array} { r } { \bar { e } ( \pmb { x } ) ^ { 2 } : = \frac { 1 } { d } \Big ( \big \| \pmb { \mu } _ { N } ( \pmb { x } ) - \pmb { \mu } _ { Q } ( \pmb { x } ) \big \| _ { 2 } ^ { 2 } + \big \| \sqrt { \pmb { \tau } _ { N } ( \pmb { x } ) } - \sqrt { \pmb { \tau } _ { Q } ( \pmb { x } ) } \big \| _ { 2 } ^ { 2 } \Big ) . } \end{array}\tag{27}
$$

In the standardized coordinates, with ${ D _ { N } } ( { \pmb x } ) : = \mathrm { d i a g } \left( \sqrt { \pmb \tau _ { N } ( \pmb x ) } \right)$ and $\lambda _ { N } ( { \pmb x } ) : = \sqrt { \pmb \tau _ { Q } ( { \pmb x } ) / \pmb \tau _ { N } ( { \pmb x } ) }$ , we use the standardized mean error and the relative scale error

$$
\begin{array} { r } { \boldsymbol { e } _ { \mu } ( \pmb { x } ) ^ { 2 } : = \frac 1 d \big \| D _ { N } ( \pmb { x } ) ^ { - 1 } \big ( \mu _ { N } ( \pmb { x } ) - \mu _ { Q } ( \pmb { x } ) \big ) \big \| _ { 2 } ^ { 2 } , \qquad \boldsymbol { e } _ { \tau } ( \pmb { x } ) ^ { 2 } : = \frac 1 d \big \| \lambda _ { N } ( \pmb { x } ) - \mathbf { 1 } \big \| _ { 2 } ^ { 2 } . } \end{array}\tag{28}
$$

Both standardize by the fitted scales $D _ { N }$ , since $\scriptstyle D _ { N }$ is the standardization the second stage actually applies. Their population versions drop the argument and average over $X \sim \mathbb { P } _ { X }$ with the fitted maps held fixed,

$$
\bar { e } _ { N } ^ { 2 } : = \mathbb { E } _ { \boldsymbol { X } } \big [ \bar { e } ( \boldsymbol { X } ) ^ { 2 } \big ] , \qquad e _ { \mu , N } ^ { 2 } : = \mathbb { E } _ { \boldsymbol { X } } \big [ e _ { \mu } ( \boldsymbol { X } ) ^ { 2 } \big ] , \qquad e _ { \tau , N } ^ { 2 } : = \mathbb { E } _ { \boldsymbol { X } } \big [ e _ { \tau } ( \boldsymbol { X } ) ^ { 2 } \big ] .
$$

They are functions of the fitted maps $\mu _ { N } , \tau _ { N }$ , hence random through $\mathcal { T } _ { N }$ . Conditionally on $\mathcal { T } _ { N }$ , as in the second stage, they are fixed numbers.

Stage 2. Stage 2 freezes $( \mu _ { N } , \tau _ { N } )$ and works in standardized coordinates. At input x and observation y the standardized residual is $\tilde { \pmb { y } } : = \pmb { D } _ { N } ( \pmb { x } ) ^ { - 1 } \big ( \pmb { y } - \pmb { \mu } _ { N } ( \pmb { x } ) \big )$ , and $\tilde { \mathbf { Y } } : = { \cal D } _ { N } ( { \mathbf { \cal X } } ) ^ { - 1 } \big ( { \mathbf { \cal Y } } - \pmb { \mu } _ { N } ( { \pmb { \cal X } } ) \big )$ is the standardized target. Its conditional law given $\mathbf { \nabla } X = x$ is

$$
\tilde { Q } ( \boldsymbol { x } ) = \mathcal { N } \big ( \tilde { \mu } ( \boldsymbol { x } ) , ~ \boldsymbol { \Lambda } ( \boldsymbol { x } ) R _ { Q } ( \boldsymbol { x } ) \boldsymbol { \Lambda } ( \boldsymbol { x } ) \big ) , \qquad \tilde { \mu } : = D _ { N } ^ { - 1 } ( \mu _ { Q } - \mu _ { N } ) , \quad \boldsymbol { \Lambda } : = D _ { N } ^ { - 1 } D _ { Q } = \mathrm { d i a g } ( \boldsymbol { \lambda } _ { N } ) ,\tag{29}
$$

so $\begin{array} { r } { e _ { \mu } ^ { 2 } \ = \ \frac { 1 } { d } \| \tilde { \mu } \| _ { 2 } ^ { 2 } } \end{array}$ and $e _ { \tau } ^ { 2 } \ = \ \frac 1 d \| \pmb { \lambda } _ { N } - \mathbf { 1 } \| _ { 2 } ^ { 2 }$ measure its deviation from the law under an exact first stage, $\tilde { Q } ^ { \star } ( { \pmb x } ) : =$ $\mathcal { N } ( \mathbf { 0 } , R _ { Q } ( \pmb { x } ) )$

The second stage fits $\mathcal { N } ( \mathbf { 0 } , R _ { s } )$ to $\tilde { \mathbf { Y } }$ with the Gaussian kernel score at the bandwidth $\widetilde { \gamma } ^ { 2 } = 4 d$ fixed above, so its loss is

$$
\ell _ { 2 } ( \pmb { \mathscr { s } } ; \tilde { \pmb { y } } ) : = S _ { k _ { \tilde { \gamma } } } \left( \mathcal { N } ( \pmb { 0 } , R _ { s } ) , \tilde { \pmb { y } } \right) ,
$$

available in closed form by (19) with $\mathbf { \nabla } m = \mathbf { 0 }$ . Risk and empirical risk of the second stage are

$$
\mathcal { R } _ { 2 } ( \pmb { s } ) : = \mathbb { E } _ { ( \pmb { X } , \pmb { Y } ) } \ell _ { 2 } \big ( \pmb { s } ( \pmb { X } ) ; \tilde { \pmb { Y } } \big ) , \qquad \mathcal { R } _ { 2 , M } ( \pmb { s } ) : = \frac { 1 } { M } \sum _ { n = N + 1 } ^ { N + M } \ell _ { 2 } \big ( \pmb { s } ( \pmb { X } _ { n } ) ; \tilde { \pmb { Y } } _ { n } \big ) ,
$$

both conditional on the first N draws.

Under the same idealization as in stage 1, stage 2 returns a minimizer $\begin{array} { r } { \pmb { \mathscr { s } } _ { M } \in \mathop { \mathrm { a r g m i n } } _ { \pmb { \mathscr { s } } \in \mathcal { G } _ { 2 } } \mathscr { R } _ { 2 , M } ( \pmb { \mathscr { s } } ) } \end{array}$ , which exists because $\mathcal { G } _ { 2 }$ is compact, being the image of the compact W under the map $\phi \mapsto s _ { \phi } .$ , which is continuous by Assumption E.2 (iv). The analysis compares it against the pointwise best core $s ^ { \star }$ of Assumption E.2 (v).

Note that $R _ { Q }$ itself will in general not lie in $\{ R _ { s } : s \in \Theta \}$ . Write

$$
\bar { \varepsilon } _ { \mathrm { a p p } } : = \mathbb { E } _ { \pmb { X } } \left[ \operatorname* { m i n } _ { { \pmb { s } } \in \Theta } \frac { 1 } { 2 } \mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ^ { \star } ( \pmb { X } ) , \mathcal { N } ( \mathbf { 0 } , R _ { s } ) \big ) \right]\tag{30}
$$

for the P -average of the approximation error $\varepsilon _ { \mathrm { a p p } } ( X )$ of Section 3 at bandwidth $\widetilde { \gamma } .$ It coincides with the average of the approximation error of Appendix D, where the infimum over strictly positive spectra and the minimum over their closure Θ agree by continuity. Writing $R _ { M } ( \pmb { x } ) : = R _ { \pmb { s } _ { M } ( \pmb { x } ) }$ , the deployed prediction is the affine image of $\mathcal { N } ( \mathbf { 0 } , R _ { M } ( \pmb { x } ) )$ under ${ \tilde { \pmb y } } \mapsto { \pmb \mu } _ { N } ( { \pmb x } ) + D _ { N } ( { \pmb x } ) { \tilde { \pmb y } }$

$$
P _ { N , M } ( { \pmb x } ) : = \mathcal { N } \big ( \mu _ { N } ( { \pmb x } ) , D _ { N } ( { \pmb x } ) R _ { M } ( { \pmb x } ) D _ { N } ( { \pmb x } ) \big ) .
$$

## E.2 THE FIRST STAGE IS PAC

Concentration bounds the excess risk of the empirical minimizer, and the following lemma turns such a bound into one on the parameter error. Together they make the first stage PAC. Throughout, a centred random variable $Z$ is sub-Gaussian with variance proxy $\sigma ^ { 2 }$ if log E $e ^ { \beta Z } \dot { \le } \beta ^ { 2 } \sigma ^ { 2 } / 2$ for all $\beta \in \mathbb { R }$ . Averages of n independent such variables are sub-Gaussian with proxy $\sigma ^ { 2 } / n .$ , and the Chernoff bound gives $\mathbb { P } ( | Z | > t ) \le 2 e ^ { - t ^ { 2 } / ( 2 \sigma ^ { 2 } ) } [ 5 8$ , Sec. 2.3].

Lemma E.2 (Curvature of the CRPS). Denote the standard normal PDF and CDF by $\varphi$ and Φ. Let $m \in \mathbb { R } , \nu > 0 ,$ , and $Y \sim \mathcal { N } ( m , \nu ^ { 2 } )$ , and for $\mu \in \mathbb { R } , \sigma > 0$ write

$$
\operatorname { d i v } ( \mu , \sigma ) : = \mathbb { E } _ { Y } \left[ \operatorname { C R P S } \bigl ( \mathcal { N } ( \mu , \sigma ^ { 2 } ) , Y \bigr ) - \operatorname { C R P S } \bigl ( \mathcal { N } ( m , \nu ^ { 2 } ) , Y \bigr ) \right] .
$$

$I f \vert \mu - m \vert \le 2 B$ and $\sigma , \nu \in [ \sqrt { \tau _ { \operatorname* { m i n } } } , \sqrt { \tau _ { \operatorname* { m a x } } } ]$ , then

$$
\begin{array} { r l r l } & { \mathrm { d i v } ( \mu , \sigma ) \ \geq \ c _ { \star } \big ( ( \mu - m ) ^ { 2 } + ( \sigma - \nu ) ^ { 2 } \big ) , } & & { c _ { \star } : = \operatorname* { m i n } \Big \{ \frac { 1 } { \sqrt { 2 \tau _ { \mathrm { m a x } } } } \varphi \big ( \frac { 2 B } { \sqrt { 2 \tau _ { \mathrm { m i n } } } } \big ) , \ \frac { 1 } { 2 } \sqrt { \frac { 2 } { \pi } } \frac { \tau _ { \mathrm { m i n } } } { ( 2 \tau _ { \mathrm { m a x } } ) ^ { 3 / 2 } } \Big \} . } \end{array}
$$

Proof. Define

$$
\begin{array} { r } { g : \mathbb { R } \times ( 0 , \infty ) \longrightarrow ( 0 , \infty ) , \qquad g ( u , w ) : = \mathbb { E } | Z | \mathrm { ~ f o r ~ } Z \sim \mathcal { N } ( u , w ^ { 2 } ) , } \end{array}
$$

so that $g ( u , w ) = w \sqrt { 2 / \pi } e ^ { - u ^ { 2 } / ( 2 w ^ { 2 } ) } + u \left( 2 \Phi ( u / w ) - 1 \right)$ , and abbreviate $a : = \mu - m$ and $t : = \sqrt { \sigma ^ { 2 } + \nu ^ { 2 } }$ . Let $X , X ^ { \prime } \sim$ $\mathcal { N } ( \mu , \sigma ^ { 2 } )$ be independent of $Y$ and of each other. The kernel form of the CRPS gives

$$
\begin{array} { r } { \mathbb { E } _ { Y } \operatorname { C R P S } \big ( \mathcal { N } ( \mu , \sigma ^ { 2 } ) , Y \big ) = \mathbb { E } | X - Y | - \frac { 1 } { 2 } \mathbb { E } | X - X ^ { \prime } | = g ( a , t ) - \frac { \sigma } { \sqrt { \pi } } , } \end{array}
$$

since $X - Y \sim \mathcal { N } ( a , t ^ { 2 } )$ and $X - X ^ { \prime } \sim \mathcal { N } ( 0 , 2 \sigma ^ { 2 } ) . \mathrm { A t } \left( \mu , \sigma \right) = \left( m , \nu \right)$ the right-hand side equals $\nu / { \sqrt { \pi } } ,$ , and $g ( 0 , t ) =$ $t \sqrt { 2 / \pi } ,$ so

$$
\begin{array} { r l r } { \mathrm { d i v } ( \mu , \sigma ) = \underbrace { g ( a , t ) - g ( 0 , t ) } _ { \mathrm { m e a n ~ t e r m } } + \underbrace { h ( \sigma ) } _ { \mathrm { s c a l e ~ t e r m } } , } & { } & { h ( \sigma ) : = \sqrt { \frac { 2 } { \pi } } \sqrt { \sigma ^ { 2 } + \nu ^ { 2 } } - \frac { \sigma } { \sqrt { \pi } } - \frac { \nu } { \sqrt { \pi } } . } \end{array}
$$

The mean term. Differentiating g in its first argument, the two exponential contributions cancel and $\partial _ { u } g ( u , t ) = 2 \Phi ( u / t ) - 1$ so $\partial _ { u } g ( 0 , t ) = 0$ and $\begin{array} { r } { \partial _ { u } ^ { 2 } g ( u , \bar { t } ) = \frac { 2 } { t } \varphi ( u / t ) } \end{array}$ . On the stated ranges $t \in [ \sqrt { 2 \tau _ { \mathrm { m i n } } } , \sqrt { 2 \tau _ { \mathrm { m a x } } } ]$ and $| u | \le | a | \le 2 B$ , whence $\begin{array} { r } { \partial _ { u } ^ { 2 } g ( u , t ) \ge \frac { 2 } { \sqrt { 2 \tau _ { \operatorname* { m a x } } } } \varphi \big ( \frac { 2 B } { \sqrt { 2 \tau _ { \operatorname* { m i n } } } } \big ) } \end{array}$ on the segment from 0 to a, and second-order Taylor expansion at $u = 0$ gives

$$
\begin{array} { r } { g ( a , t ) - g ( 0 , t ) \ge \frac { 1 } { \sqrt { 2 \tau _ { \operatorname* { m a x } } } } \varphi \big ( \frac { 2 B } { \sqrt { 2 \tau _ { \operatorname* { m i n } } } } \big ) a ^ { 2 } . } \end{array}
$$

The scale term. One checks $h ( \nu ) = h ^ { \prime } ( \nu ) = 0$ and

$$
\begin{array} { r } { h ^ { \prime \prime } ( \sigma ) = \sqrt { \frac { 2 } { \pi } } \nu ^ { 2 } \big ( \sigma ^ { 2 } + \nu ^ { 2 } \big ) ^ { - 3 / 2 } \geq \sqrt { \frac { 2 } { \pi } } \tau _ { \mathrm { m i n } } ( 2 \tau _ { \mathrm { m a x } } ) ^ { - 3 / 2 } , } \end{array}
$$

so second-order Taylor expansion at $\sigma = \nu$ gives

$$
\begin{array} { r } { h ( \sigma ) \geq \frac { 1 } { 2 } \sqrt { \frac { 2 } { \pi } } \tau _ { \mathrm { m i n } } ( 2 \tau _ { \mathrm { m a x } } ) ^ { - 3 / 2 } ( \sigma - \nu ) ^ { 2 } . } \end{array}
$$

Taking $c _ { \star }$ to be the smaller of the two constants and adding the two bounds proves the claim.

With the curvature in hand, a uniform deviation bound over $\mathcal { G } _ { 1 }$ gives the following.

Lemma E.3 (The first stage is PAC). Let $\left( \pmb { \mu } _ { N } , \pmb { \tau } _ { N } \right) \in \mathrm { a r g } \operatorname* { m i n } _ { \pmb { g } \in \mathcal { G } _ { 1 } } \mathcal { R } _ { 1 , N } ( g )$ be the stage-1 empirical risk minimizer, let $\eta _ { 1 } > 0$ and $\delta _ { 1 } \in ( 0 , 1 )$ , let $c _ { \star }$ be the curvature constant ofLemma E.2 and $\Re _ { 1 } ( \eta _ { 1 } )$ the covering number of $\dot { \mathfrak { g } } _ { 1 }$ , bounded in Lemma $E . l ,$ and put

$$
L _ { 1 } ^ { \prime } : = 1 + \frac { 1 } { 2 \sqrt { \pi \tau _ { \mathrm { m i n } } } } , \qquad \sigma _ { 1 } ^ { 2 } : = \tau _ { \mathrm { m a x } } + \Big ( B + \sqrt { \tau _ { \mathrm { m a x } } / \pi } \Big ) ^ { 2 } , \qquad \rho _ { N } ( \eta _ { 1 } ) : = 2 L _ { 1 } ^ { \prime } \eta _ { 1 } + \sigma _ { 1 } \sqrt { \frac { 2 \log \left( 2 \Re _ { 1 } ( \eta _ { 1 } ) / \delta _ { 1 } \right) } { N } } .
$$

Under Assumptions E.1 and $E . 2 \left( i \right) - ( i i i )$ , with probability at least $1 - \delta _ { 1 }$ over $\mathcal { T } _ { N }$ , thefirst-stage error is bounded both in the original and in the standardized coordinates,

$$
\bar { e } _ { N } ^ { 2 } \ \leq \ \frac { 2 \rho _ { N } ( \eta _ { 1 } ) } { c _ { \star } } \qquad a n d \qquad e _ { \mu , N } ^ { 2 } + e _ { \tau , N } ^ { 2 } \ \leq \ \frac { 2 \rho _ { N } ( \eta _ { 1 } ) } { c _ { \star } \tau _ { \operatorname* { m i n } } } .
$$

Proof. We proceed as follows. A uniform deviation bound over $\mathcal { G } _ { 1 } ~ ( \mathrm { S t e p s } ~ 1 - 2 )$ controls the excess risk of the empirical minimizer, and the curvature of the CRPS (Step 3) converts this excess risk into the parameter error. Step 1:for every fixed $g \in { \mathcal { G } } _ { 1 }$ , the centred loss $\ell _ { 1 } \bigl ( g ( \pmb { X } ) ; \pmb { Y } \bigr ) - \mathcal { R } _ { 1 } ( g )$ is sub-Gaussian in $( X , Y )$ with proxy $\sigma _ { 1 } ^ { 2 } .$ Fix $g = ( \mu , \tau ) \in \mathcal { G } _ { 1 }$ . Recall that given $\mathbf { \nabla } X = x$ the target is Gaussian, $\pmb { Y } \sim Q ( \pmb { x } ) = \mathcal { N } \big ( \pmb { \mu } _ { Q } ( \pmb { x } ) , \pmb { \Sigma } _ { Q } ( \pmb { x } ) \big )$ , and write

$$
m _ { g } ( \pmb { x } ) : = \mathbb { E } _ {pmb { Y } \sim Q ( \pmb { x } ) } \left[ \ell _ { 1 } \left( g ( \pmb { x } ) ; \pmb { Y } \right) \right]
$$

for the conditional risk. We split the fluctuation of $\ell _ { 1 } \left( g ( \pmb { X } ) ; \pmb { Y } \right)$ around $\mathcal { R } _ { 1 } ( g )$ into the fluctuation of Y given X, i.e. around $m _ { g } ( { \pmb X } )$ , and the fluctuation of $m _ { g } ( { \pmb X } )$ across inputs X. Indeed, adding and subtracting $m _ { g } ( { \pmb X } )$ ,

$$
\begin{array} { r } { \ell _ { 1 } \big ( g ( \pmb { X } ) ; \pmb { Y } \big ) - \mathcal { R } _ { 1 } ( g ) = \underbrace { \Big [ \ell _ { 1 } \big ( g ( \pmb { X } ) ; \pmb { Y } \big ) - m _ { g } ( \pmb { X } ) \Big ] } _ { \mathrm { c e n t r e d ~ g i v e n } \pmb { X } } + \underbrace { \Big [ m _ { g } ( \pmb { X } ) - \mathcal { R } _ { 1 } ( g ) \Big ] } _ { \pmb { X } \mathrm { \ - m e a s u r a b l e , c e n t r e d } } . } \end{array}
$$

For the first $\mathsf { p a r t } ,$ each summand of $\ell _ { 1 }$ is 1-Lipschitz in its observation, so ${ \pmb y } \mapsto \ell _ { 1 } ( { \pmb \mu } , \tau ; { \pmb y } )$ is $d ^ { - 1 / 2 } \cdot$ Lipschitz in the Euclidean norm, uniformly in the parameters. Given $\mathbf { \nabla } X = \mathbf { \nabla } x ,$ , write ${ \pmb Y } = { \pmb \mu } _ { Q } ( { \pmb x } ) + \pmb \Sigma _ { Q } ( { \pmb x } ) ^ { 1 / 2 } { \pmb \xi }$ with $\pmb { \xi } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { I } )$ . The composition

$$
\xi \longmapsto \ell _ { 1 } \left( g ( { \pmb x } ) ; \mu _ { Q } ( { \pmb x } ) + \Sigma _ { Q } ( { \pmb x } ) ^ { 1 / 2 } \xi \right)
$$

is then, as a composition of Lipschitz maps, Lipschitz in ξ with constant $d ^ { - 1 / 2 } \parallel \Sigma _ { Q } ( \pmb { x } ) ^ { 1 / 2 } \parallel _ { \mathrm { o p } } \leq d ^ { - 1 / 2 } \big ( \tau _ { \mathrm { m a x } } d \big ) ^ { 1 / 2 } =$ $\sqrt { \tau _ { \operatorname* { m a x } } } ,$ since $\| \boldsymbol { \Sigma } _ { \boldsymbol { Q } } ^ { 1 / 2 } \| _ { \mathrm { o p } } = \| \boldsymbol { \Sigma } _ { \boldsymbol { Q } } \| _ { \mathrm { o p } } ^ { 1 / 2 }$ and $\| \pmb { \Sigma } _ { Q } \| _ { \mathrm { o p } } \le \mathrm { t r } \pmb { \Sigma } _ { Q } \le \tau _ { \mathrm { m a x } } d$ by Assumption E.1. Gaussian concentration for Lipschitz functions [58, Thm. 5.5] applies to this composition, whose mean is $m _ { g } ( { \pmb x } )$ by the definition of the conditional risk, and gives for every x and every $\beta \in \mathbb { R }$

$$
\log \mathbb { E } _ { \pmb { Y } \sim Q ( \pmb { x } ) } \Big [ e ^ { \beta ( \ell _ { 1 } ( g ( \pmb { x } ) ; \pmb { Y } ) - m _ { g } ( \pmb { x } ) ) } \Big ] \ \leq \ \frac { \beta ^ { 2 } \tau _ { \operatorname* { m a x } } } { 2 } .
$$

The conditional fluctuation is therefore sub-Gaussian with variance proxy $\tau _ { \mathrm { m a x } } .$ , uniformly in x, the Lipschitz constant above not depending on x. For the second part, we show that $0 \leq m _ { g } ( \mathbf { x } ) \leq 2 B + 2 \sqrt { \tau _ { \operatorname* { m a x } } / \pi }$ for every x. Since $\ell _ { 1 }$ is a coordinate average and the CRPS sees only the marginals,

$$
\begin{array} { r } { m _ { g } ( { \pmb x } ) = \frac { 1 } { d } \displaystyle \sum _ { i = 1 } ^ { d } \mathbb { E } _ { Y _ { i } } \mathrm { C R P S } \big ( \mathcal { N } ( \mu _ { i } ( { \pmb x } ) , \tau _ { i } ( { \pmb x } ) ) , Y _ { i } \big ) , \qquad Y _ { i } \sim \mathcal { N } \big ( \mu _ { Q , i } ( { \pmb x } ) , \tau _ { Q , i } ( { \pmb x } ) \big ) . } \end{array}
$$

Nonnegativity is that of the CRPS. For the upper bound, recall from the proof of Lemma E.2 tha $\mathbb { E } _ { Y _ { i } } \mathrm { C R P S } ( \mathcal { N } ( \mu _ { i } , \tau _ { i } ) , Y _ { i } ) =$ $g ( a _ { i } , t _ { i } ) - \sqrt { \tau _ { i } / \pi }$ with $a _ { i } : = \mu _ { i } - \mu _ { Q , i } , t _ { i } ^ { 2 } : = \tau _ { i } + \tau _ { Q , }$ and $g ( a , t ) = \mathbb { E } | \mathcal { N } ( a , t ^ { 2 } ) | \leq | a | + t \sqrt { 2 / \pi }$ . The ranges of Assumptions E.1 and E.2 (i) give $| a _ { i } | \leq 2 B$ and $t _ { i } \le \sqrt { 2 \tau _ { \operatorname* { m a x } } }$ , hence

$$
\mathbb { E } _ { Y _ { i } } \mathrm { C R P S } \big ( \boldsymbol { \mathcal { N } } ( \mu _ { i } , \tau _ { i } ) , Y _ { i } \big ) \ \le \ 2 B + \sqrt { 2 \tau _ { \operatorname* { m a x } } } \sqrt { 2 / \pi } \ = \ 2 B + 2 \sqrt { \tau _ { \operatorname* { m a x } } / \pi } ,
$$

and averaging over the d coordinates bounds $m _ { g }$ . By Hoeffding’s lemma [58, Ch. 2] a random variable with range ℓ is sub-Gaussian with variance proxy $( \ell / 2 ) ^ { 2 }$ , so $m _ { g } ( \pmb { X } ) - \mathcal { R } _ { 1 } ( g )$ , whose range has length $2 B + 2 \sqrt { \tau _ { \mathrm { m a x } } / \pi }$ , has proxy $\left( B + \sqrt { \tau _ { \operatorname* { m a x } } / \pi } \right) ^ { 2 }$ . The two proxies add. Indeed, the tower rule applies the conditional bound inside and the two log-moment generating functions sum. That is, for every $\beta \in \mathbb { R }$

$$
\begin{array} { l } { \log \mathbb { E } _ { ( X , Y ) } \Big [ e ^ { \beta ( \ell _ { 1 } ( g ( X ) ; Y ) - \mathcal { R } _ { 1 } ( g ) ) } \Big ] \ \leq \ \frac { \beta ^ { 2 } \tau _ { \operatorname* { m a x } } } { 2 } + \log \mathbb { E } _ { X } \Big [ e ^ { \beta ( m _ { g } ( X ) - \mathcal { R } _ { 1 } ( g ) ) } \Big ] } \\ { \ \leq \ \frac { \beta ^ { 2 } } { 2 } \Big ( \tau _ { \operatorname* { m a x } } + \big ( B + \sqrt { \tau _ { \operatorname* { m a x } } / \pi } \big ) ^ { 2 } \Big ) \ = \ \frac { \beta ^ { 2 } \sigma _ { 1 } ^ { 2 } } { 2 } . } \end{array}
$$

Hence $\ell _ { 1 } \bigl ( g ( \pmb { X } ) ; \pmb { Y } \bigr ) - \mathcal { R } _ { 1 } ( g )$ is sub-Gaussian in $( X , Y )$ with proxy $\sigma _ { 1 } ^ { 2 }$ . Note that this statement holds uniformly in $g \in { \mathcal { G } } _ { 1 }$ Step 2: uniform deviation, and the excess risk of the minimizer. Recall from (13), written here with $\sigma = \sqrt { \tau }$ , that with $z : = ( y - \mu ) / \sigma$ the CRPS of a Gaussian has the closed form

$$
\begin{array} { r } { \mathrm { C R P S } \big ( \mathcal { N } ( \mu , \sigma ^ { 2 } ) , y \big ) = \sigma f ( z ) , \qquad f ( z ) : = z \big ( 2 \Phi ( z ) - 1 \big ) + 2 \varphi ( z ) - \frac { 1 } { \sqrt { \pi } } , } \end{array}
$$

and $\varphi ^ { \prime } ( z ) = - z \varphi ( z )$ gives $f ^ { \prime } ( z ) = 2 \Phi ( z ) - 1$ . Hence

$$
\begin{array} { r l r l } { \partial _ { \mu } \mathrm { C R P S } = - f ^ { \prime } ( z ) = 1 - 2 \Phi ( z ) , } & { { } } & { \partial _ { \sigma } \mathrm { C R P S } = f ( z ) - z f ^ { \prime } ( z ) = 2 \varphi ( z ) - \frac { 1 } { \sqrt { \pi } } , } \end{array}
$$

so $| \partial _ { \mu } \mathrm { C R P S } | \leq 1$ and, since $0 < 2 \varphi ( z ) \leq \sqrt { 2 / \pi } < 2 / \sqrt { \pi } , | \partial _ { \sigma } \mathrm { C R P S } | \leq 1 / \sqrt { \pi } ; \mathfrak { b y } \partial _ { \tau } = \partial _ { \sigma } / ( 2 \sigma )$ with $\sigma \geq \sqrt { \tau _ { \mathrm { m i n } } }$ on the range of Assumption $\mathrm { E . 2 ( i ) } , \mathrm { | \it \partial _ { \tau } C R P S | } \leq ( 2 \sqrt { \pi \tau _ { \mathrm { m i n } } } ) ^ { - 1 }$ . The admissible range is a product of intervals, so the segment between any two admissible parameter values stays inside it, and the mean value theorem gives, coordinate by coordinate and hence for the average $\ell _ { 1 }$

$$
\left| \ell _ { 1 } ( \mu , \tau ; y ) - \ell _ { 1 } ( \mu ^ { \prime } , \tau ^ { \prime } ; y ) \right| \leq \operatorname* { m a x } _ { i } \lvert \mu _ { i } - \mu _ { i } ^ { \prime } \rvert + \frac { \operatorname* { m a x } _ { i } \lvert \tau _ { i } - \tau _ { i } ^ { \prime } \rvert } { 2 \sqrt { \pi \tau _ { \operatorname* { m i n } } } } \leq L _ { 1 } ^ { \prime } \left. ( \mu , \tau ) - ( \mu ^ { \prime } , \tau ^ { \prime } ) \right. _ { \infty } ,
$$

uniformly in y. Evaluated at $g ( { \pmb x } )$ and $g ^ { \prime } ( \pmb { x } )$ this gives $| \ell _ { 1 } ( g ( \pmb { x } ) ; \pmb { y } ) - \ell _ { 1 } ( g ^ { \prime } ( \pmb { x } ) ; \pmb { y } ) | \leq L _ { 1 } ^ { \prime } \operatorname* { s u p } _ { \pmb { x } } \| g ( \pmb { x } ) - g ^ { \prime } ( \pmb { x } ) \| _ { \infty }$ for all $g , g ^ { \prime } \in \mathcal { G } _ { 1 }$ and all $( { \pmb x } , { \pmb y } )$ . Let C be an $\eta _ { 1 }$ -net of $\mathcal { G } _ { 1 }$ of cardinality $\Re _ { 1 } ( \eta _ { 1 } )$ , and for $g \in { \mathcal { G } } _ { 1 }$ pick $h \in { \mathcal { C } }$ with $\operatorname { s u p } _ { \pmb { x } } \| g ( \pmb { x } ) -$ $h ( \pmb { x } ) \| _ { \infty } \leq \eta _ { 1 }$ . Both $\mathcal { R } _ { 1 , N }$ and $\mathcal { R } _ { 1 }$ are averages of $\ell _ { 1 }$ and therefore inherit its Lipschitz bound, so each of the outer terms in

$$
\begin{array} { r l } { \left| \mathcal { R } _ { 1 , N } ( g ) - \mathcal { R } _ { 1 } ( g ) \right| } & { { } \le \ \left| \mathcal { R } _ { 1 , N } ( g ) - \mathcal { R } _ { 1 , N } ( h ) \right| + \left| \mathcal { R } _ { 1 , N } ( h ) - \mathcal { R } _ { 1 } ( h ) \right| + \left| \mathcal { R } _ { 1 } ( h ) - \mathcal { R } _ { 1 } ( g ) \right| } \end{array}
$$

is at most $L _ { 1 } ^ { \prime } \eta _ { 1 }$ . Taking the supremum over $g ,$

$$
\operatorname* { s u p } _ { g \in \mathcal { G } _ { 1 } } \left. \mathcal { R } _ { 1 , N } ( g ) - \mathcal { R } _ { 1 } ( g ) \right. \ \leq \ 2 L _ { 1 } ^ { \prime } \eta _ { 1 } + \operatorname* { m a x } _ { h \in \mathcal { C } } \left. \mathcal { R } _ { 1 , N } ( h ) - \mathcal { R } _ { 1 } ( h ) \right. .
$$

Fix $h \in { \mathcal { C } }$ . By Step $1 , \mathcal { R } _ { 1 , N } ( h ) - \mathcal { R } _ { 1 } ( h )$ is an average of N i.i.d. centred terms with variance proxy $\sigma _ { 1 } ^ { 2 }$ , hence sub-Gaussian with proxy $\sigma _ { 1 } ^ { 2 } / N . \mathrm { A }$ union bound over C then gives, with probability at least $1 - \delta _ { 1 }$

$$
\operatorname* { m a x } _ { h \in \mathcal { C } } \left. \mathcal { R } _ { 1 , N } ( h ) - \mathcal { R } _ { 1 } ( h ) \right. \ \leq \ \sigma _ { 1 } \sqrt { \frac { 2 \log \left( 2 \mathfrak { N } _ { 1 } ( \eta _ { 1 } ) / \delta _ { 1 } \right) } { N } } .
$$

On that event the net term $2 L _ { 1 } ^ { \prime } \eta _ { 1 }$ and this deviation add to exactly $\rho _ { N } ( \eta _ { 1 } )$ , that is, $\mathrm { s u p } _ { \mathcal { G } _ { 1 } } \vert \mathcal { R } _ { 1 , N } - \mathcal { R } _ { 1 } \vert \leq \rho _ { N } ( \eta _ { 1 } )$ . On the same event the excess risk of the empirical minimizer is controlled. Indeed, $( \mu _ { Q } , \tau _ { Q } ) \in \mathcal G _ { 1 }$ by Assumption E.2 (ii), so $\mathcal { R } _ { 1 , N } ( \pmb { \mu } _ { N } , \pmb { \tau } _ { N } ) \leq \mathcal { R } _ { 1 , N } ( \pmb { \mu } _ { Q } , \pmb { \tau } _ { Q } )$ by the definition of the empirical risk minimizer, and therefore

$$
\begin{array} { r l } { \mathcal { R } _ { 1 } ( \mu _ { N } , \tau _ { N } ) - \mathcal { R } _ { 1 } ( \mu _ { Q } , \tau _ { Q } ) = \left[ \mathcal { R } _ { 1 } ( \mu _ { N } , \tau _ { N } ) - \mathcal { R } _ { 1 , N } ( \mu _ { N } , \tau _ { N } ) \right] + \left[ \mathcal { R } _ { 1 , N } ( \mu _ { N } , \tau _ { N } ) - \mathcal { R } _ { 1 , N } ( \mu _ { Q } , \tau _ { Q } ) \right] } & { } \\ { + \left[ \mathcal { R } _ { 1 , N } ( \mu _ { Q } , \tau _ { Q } ) - \mathcal { R } _ { 1 } ( \mu _ { Q } , \tau _ { Q } ) \right] \le \rho _ { N } ( \eta _ { 1 } ) + 0 + \rho _ { N } ( \eta _ { 1 } ) = 2 \rho _ { N } ( \eta _ { 1 } ) . } & { } \end{array}
$$

This is the only place where the ERM property and Assumption E.2 (ii) enter. Step 3: curvature turns excess risk into parameter error. Since $\ell _ { 1 }$ is a coordinate average, conditioning on X decomposes,

$$
\mathcal { R } _ { 1 } ( \pmb { \mu } _ { N } , \pmb { \tau } _ { N } ) - \mathcal { R } _ { 1 } ( \pmb { \mu } _ { Q } , \pmb { \tau } _ { Q } ) = \mathbb { E } _ { \pmb { X } } \left[ \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \mathrm { d i v } \left( \mu _ { N , i } ( \pmb { X } ) , \sqrt { \tau _ { N , i } ( \pmb { X } ) } \right) \right] ,
$$

with div as in Lemma E.2, taken against the i-th marginal of the target, that is at $( m , \nu ) = \left( \mu _ { Q , i } ( X ) , \sqrt { \tau _ { Q , i } ( X ) } \right)$ . Its hypotheses hold by Assumptions E.1 and E.2 (i), so each summand is at least $c _ { \star } \left[ ( \mu _ { N , i } - \mu _ { Q , i } ) ^ { 2 } + ( \sqrt { \tau _ { N , i } } - \sqrt { \tau _ { Q , i } } ) ^ { 2 } \right]$ whose coordinate average is $c _ { \star } \bar { e } ( { \mathbf { \cal X } } ) ^ { 2 }$ by (27). Taking expectations and inserting the excess-risk bound of Step 2 gives $c _ { \star } \bar { e } _ { N } ^ { 2 } \leq 2 \rho _ { N } ( \eta _ { 1 } )$ , the first claim. The second follows by dividing each summand by $\tau _ { N , i } \geq \tau _ { \operatorname* { m i n } }$ . Indeed, it holds

$$
\bar { e } ( { \pmb x } ) ^ { 2 } \geq \tau _ { \operatorname* { m i n } } \cdot \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \frac { \big ( \mu _ { N , i } - \mu _ { Q , i } \big ) ^ { 2 } + \big ( \sqrt { \tau _ { N , i } } - \sqrt { \tau _ { Q , i } } \big ) ^ { 2 } } { \tau _ { N , i } } = \tau _ { \operatorname* { m i n } } \big ( e _ { \mu } ( { \pmb x } ) ^ { 2 } + e _ { \tau } ( { \pmb x } ) ^ { 2 } \big ) ,
$$

the last equality being the definitions (28) together with $\left( \sqrt { \tau _ { N , i } } - \sqrt { \tau _ { Q , i } } \right) ^ { 2 } / \tau _ { N , i } = ( 1 - \lambda _ { N , i } ) ^ { 2 }$ . Taking expectations over X and inserting the first claim completes the proof. □

With Lemma E.3 we have bounds on the mean and scale errors in terms of the sample size $N ,$ which is what we need to trace the first-stage error through the second stage.

## E.3 THE SECOND STAGE IN STANDARDIZED COORDINATES IS PAC

The bound we are after concerns the deployed prediction $P _ { N , M }$ in the original coordinates. The second stage, however, is fitted in the standardized ones, so we address the problem there first and return in the next subsection. Recall from (29) that the standardized target has conditional law $\begin{array} { r } { \bar { \tilde { Q } } ( \pmb { x } ) = \mathcal { N } \big ( \tilde { \pmb { \mu } } ( \pmb { x } ) , \pmb { \Lambda } ( \pmb { x } ) R _ { Q } ( \pmb { x } ) \pmb { \Lambda } ( \pmb { x } ) \big ) } \end{array}$ , which equals its counterpart $\tilde { Q } ^ { \star } ( { \pmb x } ) = \mathcal { N } \big ( \mathbf { 0 } , R _ { Q } ( { \pmb x } ) \big )$ under an exact first stage exactly when $\tilde { \pmb { \mu } } ( \pmb { x } ) = \mathbf { 0 }$ and $\pmb { \Lambda } ( \pmb { x } ) = \pmb { I }$ , and that the second stage uses the Gaussian kernel at the bandwidth ${ \tilde { \gamma } } ^ { 2 } = 4 d .$

Lemma E.4 (The implied first-stage error). Fix the first stage by conditioning on $\mathcal { T } _ { N }$ , so that $( \mu _ { N } , \tau _ { N } )$ and with them $\tilde { Q } , e _ { \mu }$ and $e _ { \tau }$ are determined. Then for every input $\mathbf { \boldsymbol { x } } \in \mathcal { X }$ of the second stage and every spectrum $s \in \Theta ;$ , replacing the standardized target by its ideal counterpart costs at most

$$
\begin{array} { r } { \left| \frac 1 2 \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ( x ) , \mathcal { N } ( \mathbf { 0 } , R _ { s } ) \big ) - \frac 1 2 \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ^ { \star } ( x ) , \mathcal { N } ( \mathbf { 0 } , R _ { s } ) \big ) \right| \leq \sqrt { \frac 2 e } \big ( e _ { \mu } ( x ) + e _ { \tau } ( x ) \big ) . } \end{array}
$$

The bound does not depend on s, so it holds in particular at the fitted $s _ { M } .$ . Now condition on $\mathcal { T } _ { M }$ as well, that is, fix the map $\begin{array} { r } { \pmb { s } _ { M } \in \mathcal { G } _ { 2 } } \end{array}$ . Averaging over (afresh) $X \sim \mathbb { P } _ { X }$ then gives

$$
\begin{array} { r } { \mathbb { E } _ { X } \left| \frac { 1 } { 2 } \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ( X ) , \mathcal { N } ( \mathbf { 0 } , R _ { s _ { M } ( X ) } ) \big ) - \frac { 1 } { 2 } \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ^ { \star } ( X ) , \mathcal { N } ( \mathbf { 0 } , R _ { s _ { M } ( X ) } ) \big ) \right| \leq \Delta _ { N } : = \sqrt { \frac { 2 } { e } } \big ( e _ { \mu , N } + c _ { \tau , N } \big ) . } \end{array}
$$

Proof. For the pointwise bound, fix x and drop it from the notation. Couple the two targets by letting $U _ { 0 } \sim \tilde { Q } ^ { \star } = \mathcal { N } ( \mathbf { 0 } , R _ { Q } )$ and setting $U : = \tilde { \mu } + \Lambda U _ { 0 }$ , which has law $\tilde { Q }$ by (29). Let $( U _ { 0 } ^ { \prime } , U ^ { \prime } )$ be an independent copy of $( U _ { 0 } , U )$ , and $V , V ^ { \prime } \sim$ $\mathcal { N } ( \mathbf { 0 } , R _ { s } )$ independent of both and of each other. Now take the squared MMDs and subtract them. That is, writing $k : = k _ { \tilde { \gamma } }$ throughout,

$$
\begin{array} { r l } & { \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } ( \tilde { Q } , \mathcal { N } ( \mathbf { 0 } , R _ { s } ) ) - \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } ( \tilde { Q } ^ { \star } , \mathcal { N } ( \mathbf { 0 } , R _ { s } ) ) = \Big ( \mathbb { E } k ( U , U ^ { \prime } ) + \mathbb { E } k ( V , V ^ { \prime } ) - 2 \mathbb { E } k ( U , V ) \Big ) } \\ & { \qquad - \Big ( \mathbb { E } k ( U _ { 0 } , U _ { 0 } ^ { \prime } ) + \mathbb { E } k ( V , V ^ { \prime } ) - 2 \mathbb { E } k ( U _ { 0 } , V ) \Big ) } \\ & { \qquad = \mathbb { E } \big [ k ( U , U ^ { \prime } ) - k ( U _ { 0 } , U _ { 0 } ^ { \prime } ) \big ] - 2 \mathbb { E } \big [ k ( U , V ) - k ( U _ { 0 } , V ) \big ] . } \end{array}
$$

The kernel is $L _ { \tilde { \gamma } } { - } \mathbf { I }$ ipschitz in each argument, with $L _ { \tilde { \gamma } } : = \sqrt { 2 } / ( \tilde { \gamma } \sqrt { e } )$ the maximal gradient norm of $\pmb { u } \mapsto e ^ { - \| \pmb { u } \| ^ { 2 } / \tilde { \gamma } ^ { 2 } }$ . In the first bracket both arguments change, one at a time,

$$
\left| k _ { \widetilde { \gamma } } ( U , U ^ { \prime } ) - k _ { \widetilde { \gamma } } ( U _ { 0 } , U _ { 0 } ^ { \prime } ) \right| \le L _ { \widetilde { \gamma } } \bigl ( \| U - U _ { 0 } \| + \| U ^ { \prime } - U _ { 0 } ^ { \prime } \| \bigr ) ,
$$

which contributes $2 L _ { \tilde { \gamma } } \mathbb { E } \| \pmb { U } - \pmb { U } _ { 0 } \|$ , the two pairs being identically distributed. In the second bracket both terms have $V$ as their second argument, so the same inequality gives $L _ { \tilde { \gamma } } \lVert U - U _ { 0 } \rVert$ . The difference is therefore at most $4 L _ { \tilde { \gamma } } \mathbb { E } \| U - U _ { 0 } \|$ in absolute value. The coupling makes that distance small. From $U - U _ { 0 } = \tilde { \mu } + ( \Lambda - I ) U _ { 0 }$ and diag $( R _ { Q } ) = { \bf 1 }$

$$
\mathbb { E } \Vert U - U _ { 0 } \Vert \ \leq \ \Vert \tilde { \mu } \Vert + \Big ( \mathbb { E } \Vert ( \mathbf { A } - I ) U _ { 0 } \Vert ^ { 2 } \Big ) ^ { 1 / 2 } = \Vert \tilde { \mu } \Vert + \Big ( \sum _ { i } ( \lambda _ { N , i } - 1 ) ^ { 2 } \Big ) ^ { 1 / 2 } = \sqrt { d } \left( e _ { \mu } + e _ { \tau } \right) ,
$$

the two summands being exactly the mean and the scale error of (28). Halving, since the claim is about $\scriptstyle { \frac { 1 } { 2 } } \mathrm { M M D } ^ { 2 }$ and using $2 L _ { \tilde { \gamma } } \sqrt { d } \ = \ \sqrt { 2 / e }$ at $\widetilde \gamma ^ { 2 } ~ = ~ 4 d$ gives the pointwise bound. Note that the dimension cancels here because the bandwidth scales with ${ \sqrt { d } } .$ For the averaged form, take $\mathbb { E } _ { X }$ of the pointwise bound. The population errors are quadratic means, $e _ { \mu , N } = ( \mathbb { E } _ { X } e _ { \mu } ( X ) ^ { 2 } ) ^ { 1 / 2 }$ and likewise for $e _ { \tau } ,$ , so Jensen’s inequality gives $\mathbb { E } _ { X } [ e _ { \mu } ( X ) ] \le e _ { \mu , N }$ and $\mathbb { E } _ { X } [ e _ { \tau } ( X ) ] \le e _ { \tau , N }$ □

The first-stage error therefore enters the second stage only through $\Delta _ { N }$ , linearly in $e _ { \mu , N }$ and $e _ { \tau , N }$ and independently of $\mathcal { G } _ { 2 }$ and of M. In particular, the first-stage error is not amplified by the second stage.

Lemma E.5 (Regularity of the stage-2 loss). For all spectra $\pmb { s } , \pmb { s } ^ { \prime } \in \Theta$ and every standardized residual $\tilde { \pmb y } \in \mathbb R ^ { d } ;$

$$
0 \leq \ell _ { 2 } ( s ; \tilde { y } ) \leq 1 , \qquad \big | \ell _ { 2 } ( s ; \tilde { y } ) - \ell _ { 2 } ( s ^ { \prime } ; \tilde { y } ) \big | \leq L _ { 2 } ^ { \prime } \| s - s ^ { \prime } \| _ { \infty } , \qquad L _ { 2 } ^ { \prime } : = \frac { 1 } { 2 } + \frac { 3 } { 4 e } ,
$$

where $\| \cdot \| _ { \infty }$ is the maximum over the d coordinates. Both bounds hold uniformly in $\tilde { \mathbf { \pmb { y } } } .$

Proof. By the closed form (19) of the Gaussian kernel score, evaluated at mean 0 and observation $\tilde { \mathbf { \pmb { y } } } .$

$$
\begin{array} { r } { \ell _ { 2 } ( s ; \tilde { y } ) = \frac { 1 } { 2 } ( 1 + \alpha ) - \beta e ^ { - \varrho } , \qquad \alpha : = \displaystyle \prod _ { k } \Big ( 1 + \frac { 4 s _ { k } } { \tilde { \gamma } ^ { 2 } } \Big ) ^ { - 1 / 2 } , \quad \beta : = \displaystyle \prod _ { k } \Big ( 1 + \frac { 2 s _ { k } } { \tilde { \gamma } ^ { 2 } } \Big ) ^ { - 1 / 2 } , \quad \varrho : = \displaystyle \sum _ { k } \frac { w _ { k } } { \tilde { \gamma } ^ { 2 } + 2 s _ { k } } , } \end{array}
$$

with $w _ { k } : = | ( \boldsymbol { F } \tilde { \boldsymbol { y } } ) _ { k } | ^ { 2 }$ the periodogram of the standardized residual. Since $( 1 + 2 u ) ^ { 2 } \geq 1 + 4 u$ we have $\alpha \geq \beta ^ { 2 }$ , hence $\ell _ { 2 } \geq \frac { 1 } { 2 } ( 1 - \dot { \beta } ) ^ { 2 } \geq 0 .$ and $\textstyle \ell _ { 2 } \leq { \frac { 1 } { \gamma } } ( 1 + \alpha ) \leq 1$ . Write $\partial _ { k } : = \partial / \partial s _ { k }$ for the derivative in the k-th spectral coordinate. Then $| \partial _ { k } \alpha | \ \overset { - } { \le } 2 / \tilde { \gamma } ^ { 2 } , | \partial _ { k } \beta | \le 1 / \tilde { \gamma } ^ { 2 }$ and $| \partial _ { k } \varrho | = 2 w _ { k } / ( \tilde { \gamma } ^ { 2 } + 2 s _ { k } ) ^ { 2 }$ , so by the product rule

$$
| \partial _ { k } \ell _ { 2 } | \ \leq \ \frac 1 2 | \partial _ { k } \alpha | + | \partial _ { k } \beta | e ^ { - \varrho } + \beta e ^ { - \varrho } | \partial _ { k } \varrho | .
$$

On $\Theta$ we have $s _ { k } \leq d , \ s o \ \varrho \geq w _ { k } / ( \tilde { \gamma } ^ { 2 } + 2 d )$ and $\begin{array} { r } { w _ { k } e ^ { - \varrho } \leq \operatorname* { s u p } _ { t \geq 0 } t e ^ { - t / ( \tilde { \gamma } ^ { 2 } + 2 d ) } = \big ( \tilde { \gamma } ^ { 2 } + 2 d ) / e } \end{array}$ . At our bandwidth $\widetilde { \gamma } ^ { 2 } = 4 d ,$ it therefore holds

$$
| \partial _ { k } \ell _ { 2 } | \le \frac { 2 } { \tilde { \gamma } ^ { 2 } } + \frac { 2 ( \tilde { \gamma } ^ { 2 } + 2 d ) } { e \tilde { \gamma } ^ { 4 } } = \frac { 1 } { 2 d } + \frac { 3 } { 4 e d } = \frac { L _ { 2 } ^ { \prime } } { d } .
$$

Since $\Theta$ is convex, the segment between s and $s ^ { \prime }$ stays in $\Theta ,$ , and the mean value theorem along it gives

$$
\big | \ell _ { 2 } ( \pmb { \mathscr { s } } ; \tilde { \pmb { y } } ) - \ell _ { 2 } ( \pmb { \mathscr { s } } ^ { \prime } ; \tilde { \pmb { y } } ) \big | \ \le \ \sum _ { k = 0 } ^ { d - 1 } \operatorname* { s u p } _ { \Theta } \big | \partial _ { k } \ell _ { 2 } \big | | s _ { k } - s _ { k } ^ { \prime } \big | \ \le \ d \cdot \ \frac { L _ { 2 } ^ { \prime } } { d } \ \| \pmb { \mathscr { s } } - \pmb { \mathscr { s } } ^ { \prime } \| _ { \infty } \ = \ L _ { 2 } ^ { \prime } \ \| \pmb { \mathscr { s } } - \pmb { \mathscr { s } } ^ { \prime } \| _ { \infty } .
$$

Proposition E.6 (PAC bound in standardized coordinates). Let $\begin{array} { r } { \pmb { \mathscr { s } } _ { M } \in \mathop { \mathrm { a r g m i n } } _ { \pmb { \mathscr { s } } \in \mathcal { G } _ { 2 } } \mathscr { R } _ { 2 , M } ( \pmb { \mathscr { s } } ) } \end{array}$ be the stage-2 empirical risk minimizer, let $\eta _ { 1 } , \eta _ { 2 } > 0$ and $\delta _ { 1 } , \delta _ { 2 } \in ( 0 , 1 )$ , and put

$$
c _ { 2 } : = 2 \sqrt { \frac { 2 } { e c _ { \star } \tau _ { \mathrm { m i n } } } } , \qquad \rho _ { M } ( \eta _ { 2 } ) : = 4 L _ { 2 } ^ { \prime } \eta _ { 2 } + 2 \sqrt { \frac { \log \left( 2 \mathfrak { M } _ { 2 } ( \eta _ { 2 } ) / \delta _ { 2 } \right) } { 2 M } } ,
$$

with $\rho _ { N } ( \eta _ { 1 } )$ as in Lemma E.3. Both $\rho _ { N }$ and $\rho _ { M }$ suppress their dependence on the confidence levels, which enters through log $\left( 2 \mathfrak { N } _ { j } ( \eta _ { j } ) / \delta _ { j } \right)$ . Under Assumptions E.1 and E.2, with probability at least $1 - \delta _ { 1 } - \delta _ { 2 }$

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { x } } \left[ \frac { 1 } { 2 } \mathrm { M M D } _ { \hat { \gamma } } ^ { 2 } \big ( \tilde { Q } ^ { \star } ( \pmb { X } ) , \mathcal { N } ( \mathbf { 0 } , R _ { s _ { M } ( \pmb { X } ) } ) \big ) \right] \ \leq \ \underbrace { \bar { \varepsilon } _ { \mathrm { a p p } } } _ { a p p r } + \underbrace { 2 c _ { 2 } \sqrt { \rho _ { N } ( \eta _ { 1 } ) } } _ { f r s t s t u g e } + \underbrace { \rho _ { M } ( \eta _ { 2 } ) } _ { s e c o n d s t a g e } . } \end{array}\tag{31}
$$

Proof. Condition on $\mathcal { T } _ { N }$ , that is $\mu _ { N }$ and $D _ { N } = \mathrm { d i a g } \left( \sqrt { \tau _ { N } } \right)$ are fixed and the M pairs $( X _ { n } , \tilde { Y } _ { n } ) , n > N$ , are i.i.d. with

$$
\tilde { Y } _ { n } \mid X _ { n } \sim \tilde { Q } ( X _ { n } ) = \mathcal { N } \big ( \tilde { \mu } ( X _ { n } ) , \Lambda ( X _ { n } ) R _ { Q } ( X _ { n } ) \Lambda ( X _ { n } ) \big )
$$

as in (29), with $\pmb { \Lambda } = \pmb { D } _ { N } ^ { - 1 } \pmb { D } _ { Q }$ the diagonal of scale ratios and $\tilde { \pmb { \mu } } = \pmb { { { D } } } _ { N } ^ { - 1 } ( \pmb { \mu } _ { Q } - \pmb { \mu } _ { N } )$ . For every $\boldsymbol { s } \in \mathcal { G } _ { 2 }$ , the expected score gap equals the divergence of the kernel score, pointwise in X and hence on average,

$$
\begin{array} { r } { \mathcal { R } _ { 2 } ( s ) - \mathbb { E } \Big [ S _ { k _ { \tilde { \gamma } } } \big ( \tilde { Q } ( \boldsymbol { X } ) , \tilde { \boldsymbol { Y } } \big ) \Big ] = \mathbb { E } _ { \boldsymbol { X } } \Big [ \frac { 1 } { 2 } \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ( \boldsymbol { X } ) , \mathcal { N } ( \mathbf { 0 } , R _ { s ( \boldsymbol { X } ) } ) \big ) \Big ] . } \end{array}
$$

Since $\mathcal { R } _ { 2 , M } ( \pmb { s } _ { M } ) \leq \mathcal { R } _ { 2 , M } ( \pmb { s } )$ for every $\boldsymbol { s } \in \mathcal { G } _ { 2 }$ , an ERM decomposition analogous to Step 2 of Lemma E.3 gives

$$
\mathbb { E } _ { X } \left[ \frac { 1 } { 2 } \mathrm { M M D } _ { \stackrel { . } { \gamma } } ^ { 2 } \big ( \tilde { Q } ( X ) , \mathcal { N } ( \mathbf { 0 } , R _ { s , { \boldsymbol u } ( X ) } ) \big ) \right] \leq \operatorname* { i n f } _ { s \in \mathcal { G } _ { 2 } } \mathbb { E } _ { X } \left[ \frac { 1 } { 2 } \mathrm { M M D } _ { \stackrel { . } { \gamma } } ^ { 2 } \big ( \tilde { Q } ( X ) , \mathcal { N } ( \mathbf { 0 } , R _ { s ( X ) } ) \big ) \right] + 2 \operatorname* { s u p } _ { s \in \mathcal { G } _ { 2 } } | \mathcal { R } _ { 2 , { \boldsymbol u } } ( s ) - \mathcal { R } _ { 2 } ( s ) | .
$$

We bound the two terms on the right in turn. By the Lipschitz bound of Lemma E.5, passing to an η<sub>2</sub>-net of $\mathcal { G } _ { 2 }$ of size $\mathfrak { N } _ { 2 } ( \eta _ { 2 } )$ moves $\mathcal { R } _ { 2 , M }$ and $\mathcal { R } _ { 2 }$ by at most $L _ { 2 } ^ { \prime } \eta _ { 2 }$ each. At each net point $\mathcal { R } _ { 2 , M } - \mathcal { R } _ { 2 }$ is an average of M independent terms of range 1, so Hoeffding’s inequality [58, Ch. 2] and a union bound give, with probability at least $1 - \delta _ { 2 }$ ,

$$
\operatorname* { s u p } _ { s \in \mathcal { G } _ { 2 } } \left. \mathcal { R } _ { 2 , M } ( \pmb { s } ) - \mathcal { R } _ { 2 } ( \pmb { s } ) \right. \ \leq \ 2 L _ { 2 } ^ { \prime } \eta _ { 2 } + \sqrt { \frac { \log \left( 2 \Re _ { 2 } ( \eta _ { 2 } ) / \delta _ { 2 } \right) } { 2 M } } \ = \ \frac 1 2 \rho _ { M } ( \eta _ { 2 } ) ,
$$

so the deviation term contributes $\rho _ { M } ( \boldsymbol { \eta } _ { 2 } )$ . For the infimum, Lemma E.4 replaces $\tilde { Q }$ by $\tilde { Q } ^ { \star }$ inside it – the first of two uses of that lemma, one on each side of the ERM comparison. Its bound is uniform in s, so it transfers to the infimum, and Assumption E.2 (v) then identifies what remains with the approximation error,

$$
\begin{array} { r l } { \underset { s \in \mathcal { G } _ { 2 } } { \operatorname* { i n f } } \ \mathbb { E } _ { X } \left[ \frac 1 2 \mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \big ( \widetilde { Q } ( X ) , \mathcal { N } ( \mathbf { 0 } , R _ { s ( X ) } ) \big ) \right] } & { \le \underset { s \in \mathcal { G } _ { 2 } } { \operatorname* { i n f } } \ \mathbb { E } _ { X } \left[ \frac 1 2 \mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \big ( \widetilde { Q } ^ { \star } ( X ) , \mathcal { N } ( \mathbf { 0 } , R _ { s ( X ) } ) \big ) \right] + \Delta _ { N } } \\ & { = \ \mathbb { E } _ { X } \left[ \underset { s \in \Theta } { \operatorname* { m i n } } \frac 1 { 2 } \mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \big ( \widetilde { Q } ^ { \star } ( X ) , \mathcal { N } ( \mathbf { 0 } , R _ { s } ) \big ) \right] + \Delta _ { N } } \\ & { = \ \bar { \varepsilon } _ { \mathrm { a p p } } + \Delta _ { N } . } \end{array}
$$

On the event of Lemma E.3, of probability at least $1 - \delta _ { 1 }$ , the first-stage term is controlled. Indeed, using $a + b \leq$ $\sqrt { 2 } ( a ^ { 2 } + b ^ { 2 } ) ^ { 1 / 2 }$

$$
\Delta _ { N } = \sqrt { \textstyle \frac { 2 } { e } } \left( e _ { \mu , N } + e _ { \tau , N } \right) \le \frac { 2 } { \sqrt { e } } \big ( e _ { \mu , N } ^ { 2 } + e _ { \tau , N } ^ { 2 } \big ) ^ { 1 / 2 } \le \frac { 2 } { \sqrt { e } } \sqrt { \frac { 2 \rho _ { N } ( \eta _ { 1 } ) } { c _ { \star } \tau _ { \mathrm { m i n } } } } = c _ { 2 } \sqrt { \rho _ { N } ( \eta _ { 1 } ) } .
$$

Collecting the three contributions in the ERM decomposition,

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { x } } \left[ \frac { 1 } { 2 } \mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ( \pmb { X } ) , \mathcal { N } ( \mathbf { 0 } , R _ { s _ { M } ( \pmb { X } ) } ) \big ) \right] \ \leq \ \bar { \varepsilon } _ { \mathrm { a p p } } + c _ { 2 } \sqrt { \rho _ { N } ( \eta _ { 1 } ) } + \rho _ { M } ( \eta _ { 2 } ) , } \end{array}
$$

which bounds the core against ${ \tilde { Q } } .$ , the law it is actually fitted to. Statement (31) is against the ideal $\tilde { Q } ^ { \star }$ instead, so Lemma E.4 is applied a second time, now on the left-hand side at $\boldsymbol { s } = \boldsymbol { s } _ { M }$ , at the cost of a further $\Delta _ { N } \leq c _ { 2 } \sqrt { \rho _ { N } ( \eta _ { 1 } ) }$ , which accounts for the factor 2 in (31). A union bound over the events of Lemma E.3 and of the Hoeffding step completes the proof.

## E.4 SCORE IS PAC

The deployed prediction $P _ { N , M }$ and the kernel score it is evaluated with both live in the original coordinates, at a bandwidth $\gamma > 0$ that we now keep arbitrary. We carry (31) over by a two-sided comparison of the squared MMD with the Frobenius norm and a coupling bound for the marginal error. Lemma D.1 (i) already provides such a lower bound, but its constant decays exponentially in $d ,$ which would carry an exponential dependence on the dimension into the main theorem below. At the second-stage bandwidth ${ \tilde { \gamma } } ^ { 2 } = 4 d ,$ part (ii) of the following lemma replaces it by a constant that does not decay exponentially in d.

Lemma E.7 (Frobenius sandwich (refined)). Let $\Sigma _ { 1 } , \Sigma _ { 2 } \succeq 0$ be real symmetric with tr $\begin{array} { r } { \Sigma _ { 1 } = \operatorname { t r } \Sigma _ { 2 } , } \end{array}$ , and let $\pmb { \mu } \in \mathbb { R } ^ { d }$ be the common mean.

(i) Upper bound. For every $\gamma > 0 ,$

$$
\mathrm { M M D } _ { \gamma } ^ { 2 } \big ( \mathcal { N } ( { \pmb \mu } , { \pmb \Sigma } _ { 1 } ) , \mathcal { N } ( { \pmb \mu } , { \pmb \Sigma } _ { 2 } ) \big ) \ \leq \ \frac { 2 \| { \pmb \Sigma } _ { 1 } - { \pmb \Sigma } _ { 2 } \| _ { F } ^ { 2 } } { \gamma ^ { 4 } } .
$$

(ii) Lower bound. Ifin addition $\Sigma _ { 1 } , \Sigma _ { 2 }$ are correlation matrices, then at the second-stage bandwidth $\widetilde { \gamma } ^ { 2 } = 4 d ,$

$$
\mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \big ( \mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } _ { 1 } ) , \mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } _ { 2 } ) \big ) \geq \frac { 2 e ^ { - 1 0 } \| \pmb { \Sigma } _ { 1 } - \pmb { \Sigma } _ { 2 } \| _ { F } ^ { 2 } } { \widetilde { \gamma } ^ { 4 } } .
$$

Proof. The kernel is translation invariant, so we may take ${ \pmb \mu } = { \bf 0 }$ throughout. The upper bound (i) then follows exactly as in the proof of Lemma D.1 (i), where the assumption that the two matrices are correlations is used only through $\operatorname { t r } ( \Sigma _ { 1 } - \Sigma _ { 2 } ) = 0$ . For the lower bound, take the Gaussian kernel at $\gamma = \tilde { \gamma }$ and recall from that same proof the representation

$$
\mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } = \mathbb { E } \Big [ \big ( e ^ { - a ( z ) } - e ^ { - b ( z ) } \big ) ^ { 2 } \Big ] , \qquad a ( z ) : = \frac { z ^ { \top } \Sigma _ { 1 } z } { \widetilde { \gamma } ^ { 2 } } , \qquad b ( z ) : = \frac { z ^ { \top } \Sigma _ { 2 } z } { \widetilde { \gamma } ^ { 2 } } ,
$$

where E is with respect $\mathrm { t o } ~ z \sim \mathcal { N } ( \mathbf { 0 } , I )$ here and below. Since $\Sigma _ { 1 } , \Sigma _ { 2 } \succeq 0$ we have $a , b \geq 0 ,$ , so the mean value theorem gives

$$
\left( e ^ { - a } - e ^ { - b } \right) ^ { 2 } \geq ( a - b ) ^ { 2 } e ^ { - 2 ( a + b ) } .
$$

If $\Sigma _ { 1 } = \Sigma _ { 2 }$ there is nothing to prove. Otherwise, with $N : = \Sigma _ { 1 } - \Sigma _ { 2 }$ , the second-moment identity for Gaussian quadratic forms from the proof of Lemma D.1, $\mathbb { E } ( z ^ { \top } N z ) ^ { 2 } = 2 \| N \| _ { F } ^ { 2 } + ( \mathrm { t r } N ) ^ { 2 }$ , gives

$$
\mathbb { E } ( a - b ) ^ { 2 } \ = \ \frac { 2 \lVert \pmb { \Sigma } _ { 1 } - \pmb { \Sigma } _ { 2 } \rVert _ { F } ^ { 2 } } { \tilde { \gamma } ^ { 4 } } \ > \ 0 ,\tag{32}
$$

since tr $N = 0 .$ , and

$$
\mathrm { d } \nu ( z ) : = \frac { \bigl ( a ( z ) - b ( z ) \bigr ) ^ { 2 } } { \mathbb { E } ( a - b ) ^ { 2 } } \frac { e ^ { - \| z \| ^ { 2 } / 2 } } { ( 2 \pi ) ^ { d / 2 } } \mathrm { d } z
$$

defines a probability measure on $\mathbb { R } ^ { d }$ . Jensen’s inequality for expectations with respect to ν and the convex map $x \mapsto e ^ { - 2 x }$ gives

$$
\begin{array} { r } { \mathbb { E } _ { \nu } \left[ e ^ { - 2 ( a + b ) } \right] \geq \exp \big ( - 2 \mathbb { E } _ { \nu } [ a + b ] \big ) , \quad \mathrm { i . e . } \quad \mathbb { E } \big [ ( a - b ) ^ { 2 } e ^ { - 2 ( a + b ) } \big ] \geq \mathbb { E } \left[ ( a - b ) ^ { 2 } \right] \exp \Big ( - 2 \frac { \mathbb { E } \big [ ( a + b ) ( a - b ) ^ { 2 } \big ] } { \mathbb { E } ( a - b ) ^ { 2 } } \Big ) . } \end{array}\tag{33}
$$

Moreover, writing $\| \cdot \| _ { p } : = \left( \mathbb { E } | \cdot | ^ { p } \right) ^ { 1 / p }$ and noting that all moments are finite, since a, b are quadratic forms in a Gaussian vector, Hölder’s inequality with exponents 3 and $\mathit { \Delta } _ { \overline { { 2 } } } ^ { 3 }$ , applied to $a + b$ and $( a - b ) ^ { 2 }$ , gives

$$
\begin{array} { r } { \mathbb { E } \big [ ( a + b ) ( a - b ) ^ { 2 } \big ] \ \leq \ \| a + b \| _ { 3 } \left\| ( a - b ) ^ { 2 } \right\| _ { 3 / 2 } \ = \ \| a + b \| _ { 3 } \| a - b \| _ { 3 } ^ { 2 } . } \end{array}
$$

The last step holds because $\left\| ( a - b ) ^ { 2 } \right\| _ { 3 / 2 } = \left( \mathbb { E } | a - b | ^ { 3 } \right) ^ { 2 / 3 } = \| a - b \| _ { 3 } ^ { 2 }$ . Since $a - b = z ^ { \top } ( \Sigma _ { 1 } - \Sigma _ { 2 } ) z / \tilde { \gamma } ^ { 2 }$ is a polynomial of degree two in the coordinates of the standard Gaussian vector z, Gaussian hypercontractivity [59, Theorem 5.10 and Remark 5.11] bounds its higher moments by the second: for such polynomials $\| \cdot \| _ { p } \leq ( p - 1 ) \| \cdot \| _ { 2 }$ for every $p \geq 2$ , and at $p = 3$

$$
\| a - b \| _ { 3 } \leq 2 \| a - b \| _ { 2 } .
$$

By the mean value bound and (33), together with Hölder and hypercontractivity, therefore,

$$
\begin{array} { r l } & { \mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \ \geq \ \mathbb { E } \big [ ( a - b ) ^ { 2 } e ^ { - 2 ( a + b ) } \big ] \ \geq \ \mathbb { E } \big [ ( a - b ) ^ { 2 } \big ] \ \exp \Big ( - 2 \ \frac { \mathbb { E } \big [ ( a + b ) ( a - b ) ^ { 2 } \big ] } { \mathbb { E } \big [ ( a - b ) ^ { 2 } \big ] } \Big ) } \\ & { \qquad \geq \ \mathbb { E } \big [ ( a - b ) ^ { 2 } \big ] \ \exp \Big ( - 2 \ \frac { \| a + b \| _ { 3 } \| a - b \| _ { 3 } ^ { 2 } } { \mathbb { E } \big [ ( a - b ) ^ { 2 } \big ] } \Big ) \ \geq \ \mathbb { E } \big [ ( a - b ) ^ { 2 } \big ] \ e ^ { - 8 \| a + b \| _ { 3 } } . } \end{array}\tag{34}
$$

It remains to bound $\| a + b \| _ { 3 }$ . Note that $\Sigma _ { 1 } + \Sigma _ { 2 }$ is real symmetric, so there are an orthogonal U and a diagonal $\boldsymbol { D } = \mathrm { d i a g } ( \theta _ { 1 } , \dots , \theta _ { d } )$ with $\Sigma _ { 1 } + \Sigma _ { 2 } = U ^ { \top } D U$ , and $\theta _ { i } \geq 0$ because $\Sigma _ { 1 } , \Sigma _ { 2 } \succeq 0$ . Write

$$
a + b = \frac { z ^ { \top } ( \Sigma _ { 1 } + \Sigma _ { 2 } ) z } { \tilde { \gamma } ^ { 2 } } = \frac { 1 } { \tilde { \gamma } ^ { 2 } } \sum _ { i = 1 } ^ { d } \theta _ { i } ( U z ) _ { i } ^ { 2 } ,
$$

and $U z \sim \mathcal { N } ( \mathbf { 0 } , I )$ since U is orthogonal, so $a + b$ has the same law as $\widetilde { \gamma } ^ { - 2 } \sum _ { i } \theta _ { i } z _ { i } ^ { 2 }$ . Both $\Sigma _ { 1 }$ and $\Sigma _ { 2 }$ are correlation matrices, so $\begin{array} { r } { \sum _ { i } \theta _ { i } = \operatorname { t r } ( \Sigma _ { 1 } + \Sigma _ { 2 } ) = 2 d } \end{array}$ , and Minkowski’s inequality gives

$$
\| a + b \| _ { 3 } \leq \frac { 1 } { \tilde { \gamma } ^ { 2 } } \sum _ { i = 1 } ^ { d } \theta _ { i } \left\| z _ { i } ^ { 2 } \right\| _ { 3 } = \frac { 2 d } { \tilde { \gamma } ^ { 2 } } 1 5 ^ { 1 / 3 } = \frac { 1 } { 2 } 1 5 ^ { 1 / 3 } ,
$$

since $\| z _ { i } ^ { 2 } \| _ { 3 } = ( \mathbb { E } Z ^ { 6 } ) ^ { 1 / 3 } = 1 5 ^ { 1 / 3 }$ for $Z \sim { \mathcal { N } } ( 0 , 1 )$ and $\widetilde { \gamma } ^ { 2 } = 4 d .$ Since $4 \cdot 1 5 ^ { 1 / 3 } < 1 0$ , (34) and (32) therefore give

$$
\mathrm { M M D } _ { \widetilde { \gamma } } ^ { 2 } \ \geq \ \mathbb { E } \big [ ( a - b ) ^ { 2 } \big ] \ e ^ { - 8 \| a + b \| _ { 3 } } \ \geq \ e ^ { - 1 0 } \mathbb { E } \big [ ( a - b ) ^ { 2 } \big ] \ = \ \frac { 2 e ^ { - 1 0 } } { \widetilde { \gamma } ^ { 4 } } \| \Sigma _ { 1 } - \Sigma _ { 2 } \| _ { F } ^ { 2 } \ = \ \frac { e ^ { - 1 0 } } { 8 d ^ { 2 } } \| \Sigma _ { 1 } - \Sigma _ { 2 } \| _ { F } ^ { 2 } .
$$

Theorem E.8 (PAC bound in the original coordinates). Let $( \mu _ { N } , \tau _ { N } )$ and $s _ { M }$ be the empirical risk minimizers ofthe two stages and $P _ { N , M }$ the deployed prediction they define. Let $\gamma > 0 , \eta _ { 1 } , \eta _ { 2 } > 0 ,$ , and $\delta _ { 1 } , \delta _ { 2 } \in ( 0 , 1 )$ , and put

$$
K : = \frac { 3 2 e ^ { 1 0 } \tau _ { \mathrm { m a x } } ^ { 2 } d ^ { 2 } } { \gamma ^ { 4 } } , \qquad c _ { 1 } : = 2 K c _ { 2 } + \frac { 4 \sqrt { 2 d } } { \gamma \sqrt { e c _ { \star } } } ,
$$

with $\rho _ { N } ( \eta _ { 1 } ) , \sigma _ { 1 }$ and $L _ { 1 } ^ { \prime }$ as in Lemma $E . 3 , c _ { 2 }$ and $\rho _ { M } ( \eta _ { 2 } )$ as in Proposition $E . 6 , c _ { \star }$ the curvature constant ofLemma E.2 and $L _ { 2 } ^ { \prime }$ the Lipschitz constant ofLemma E.5. Under Assumptions E.1 and E.2, with probability at least $1 - \delta _ { 1 } - \delta _ { 2 }$

$$
\begin{array} { r } { \mathbb { E } _ { { \pmb X } } \Big [ \frac { 1 } { 2 } \mathrm { M M D } _ { \gamma } ^ { 2 } \big ( Q ( { \pmb X } ) , P _ { N , M } ( { \pmb X } ) \big ) \Big ] ~ \leq ~ \underbrace { K \bar { \varepsilon } _ { \mathrm { a p p } } } _ { a p p r } + \underbrace { c _ { 1 } \sqrt { \rho _ { N } ( \eta _ { 1 } ) } } _ { f i r s t s t a g e } + \underbrace { K \rho _ { M } ( \eta _ { 2 } ) } _ { s e c o n d s t a g e } . } \end{array}\tag{35}
$$

In particular, let $\varepsilon > 0$ and $\delta \in ( 0 , 1 )$ , and choose $\delta _ { 1 } = \delta _ { 2 } = \delta / 2 , \eta _ { 1 } = \varepsilon ^ { 2 } / ( 1 6 c _ { 1 } ^ { 2 } L _ { 1 } ^ { \prime } )$ and $\eta _ { 2 } = \varepsilon / ( 1 6 K L _ { 2 } ^ { \prime } )$ . Choose

$$
N \geq \frac { 1 2 8 c _ { 1 } ^ { 4 } \sigma _ { 1 } ^ { 2 } } { \varepsilon ^ { 4 } } \left( q \log \left( 1 + \frac { 3 2 W L _ { 1 } c _ { 1 } ^ { 2 } L _ { 1 } ^ { \prime } } { \varepsilon ^ { 2 } } \right) + \log \frac 4 \delta \right) ,
$$

$$
M \ge \frac { 3 2 K ^ { 2 } } { \varepsilon ^ { 2 } } \left( q \log \left( 1 + \frac { 3 2 W L _ { 2 } K L _ { 2 } ^ { \prime } } { \varepsilon } \right) + \log \frac { 4 } { \delta } \right) ,
$$

where $q = \dim \mathcal { W }$ is the number ofnetwork parameters, W the bound on them and $L _ { 1 } , L _ { 2 }$ the parameter-Lipschitz constants of Assumption $E . 2 ( i i i ) – ( i \nu )$ and $L _ { 1 } ^ { \prime } , L _ { 2 } ^ { \prime }$ the loss Lipschitz constants. Then

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { X } } \Big [ \frac { 1 } { 2 } \mathrm { M M D } _ { \gamma } ^ { 2 } \big ( Q ( \pmb { X } ) , P _ { N , M } ( \pmb { X } ) \big ) \Big ] ~ \leq ~ K \bar { \varepsilon } _ { \mathrm { a p p } } + \varepsilon } \end{array}
$$

with probability at least $1 - \delta .$

Proof. Fix x and drop it from the notation. Let $Q ^ { \prime \prime } : = \mathcal { N } ( \pmb { \mu } _ { Q } , \pmb { D } _ { Q } R _ { M } \pmb { D } _ { Q } )$ carry the exact marginals and the fitted correlation. Since MMD<sub>γ</sub> is a metric, $( x + y ) ^ { 2 } \leq 2 x ^ { 2 } + 2 y ^ { 2 }$ gives

$$
\begin{array} { r } { \frac 1 2 \mathrm { M M D } _ { \gamma } ^ { 2 } ( Q , P _ { N , M } ) \ \leq \ \mathrm { M M D } _ { \gamma } ^ { 2 } ( Q , Q ^ { \prime \prime } ) + \mathrm { M M D } _ { \gamma } ^ { 2 } ( Q ^ { \prime \prime } , P _ { N , M } ) . } \end{array}
$$

Correlation error. $Q$ and $Q ^ { \prime \prime }$ share the mean $\pmb { \mu } _ { Q }$ , and their covariances share the diagonal $\tau _ { Q }$ and hence their traces, so Lemma $\mathrm { E . 7 ( i ) }$ and $\| D _ { Q } \| _ { \mathrm { o p } } ^ { 2 } \le \tau _ { \mathrm { m a x } }$ give

$$
\mathrm { M M D } _ { \gamma } ^ { 2 } ( Q , Q ^ { \prime \prime } ) \leq \frac { 2 } { \gamma ^ { 4 } } \| { \cal D } _ { Q } ( R _ { Q } - R _ { M } ) { \cal D } _ { Q } \| _ { F } ^ { 2 } \leq \frac { 2 \tau _ { \operatorname * { m a x } } ^ { 2 } } { \gamma ^ { 4 } } \| R _ { Q } - R _ { M } \| _ { F } ^ { 2 } .
$$

By Lemma E.7 (ii), applied to $R _ { Q }$ and $R _ { M }$ , which both have trace $d ,$

$$
\begin{array} { r } { \| R _ { Q } - R _ { M } \| _ { F } ^ { 2 } \ \leq \ 1 6 e ^ { 1 0 } d ^ { 2 } \cdot \frac 1 2 \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ^ { \star } , \mathcal { N } ( \mathbf { 0 } , R _ { M } ) \big ) , } \end{array}
$$

using $\tilde { \gamma } ^ { 4 } = 1 6 d ^ { 2 }$ . Hence $\mathrm { M M D } _ { \gamma } ^ { 2 } ( Q , Q ^ { \prime \prime } ) \leq K \cdot \frac { 1 } { 2 } \mathrm { M M D } _ { \tilde { \gamma } } ^ { 2 } \big ( \tilde { Q } ^ { \star } , \mathcal { N } ( \mathbf { 0 } , R _ { M } ) \big )$ . Marginal error. $Q ^ { \prime \prime }$ and $P _ { N , M }$ share the correlation $R _ { M }$ and differ only in the first-stage parameters. Couple $U : = \mu _ { Q } + D _ { Q } V \sim Q ^ { \prime \prime }$ and $U ^ { \prime } : = \pmb { \mu } _ { N } + \pmb { D } _ { N } \pmb { V } \sim$

$P _ { N , M }$ through one $\pmb { V } \sim \mathcal { N } ( \mathbf { 0 } , R _ { M } )$ , and let $( U _ { 2 } , U _ { 2 } ^ { \prime } )$ be an independent copy of $( U , U ^ { \prime } )$ . By the definition of $\mathrm { M M D } _ { \gamma }$ and the symmetry of the kernel,

$$
\mathrm { M M D } _ { \gamma } ^ { 2 } ( Q ^ { \prime \prime } , P _ { N , M } ) = \mathbb { E } \big [ k _ { \gamma } ( U , U _ { 2 } ) - k _ { \gamma } ( U , U _ { 2 } ^ { \prime } ) \big ] + \mathbb { E } \big [ k _ { \gamma } ( U ^ { \prime } , U _ { 2 } ^ { \prime } ) - k _ { \gamma } ( U ^ { \prime } , U _ { 2 } ) \big ] \leq 2 L _ { \gamma } \mathbb { E } \| U - U ^ { \prime } \| ,
$$

where $L _ { \gamma } : = \sqrt { 2 } / ( \gamma \sqrt { e } )$ is the Lipschitz constant of $k _ { \gamma }$ in each argument. A $\mathbf { \nabla } \cdot \mathbf { s } \ \mathrm { d i a g } ( R _ { M } ) = \mathbf { 1 }$

$$
\begin{array} { r } { \mathbb { E } \| { \boldsymbol { U } } - { \boldsymbol { U } } ^ { \prime } \| \leq \| \mu _ { Q } - \mu _ { N } \| + \mathbb { E } \| ( D _ { Q } - D _ { N } ) { \boldsymbol { V } } \| \leq \| \mu _ { Q } - \mu _ { N } \| + \| \sqrt { \tau _ { Q } } - \sqrt { \tau _ { N } } \| \leq \sqrt { 2 d } \bar { \epsilon } , } \end{array}
$$

so $\begin{array} { r } { \mathrm { M M D } _ { \gamma } ^ { 2 } ( Q ^ { \prime \prime } , P _ { N , M } ) \leq \frac { 4 \sqrt { d } } { \gamma \sqrt { e } } \bar { e } . } \end{array}$ , with e¯ the first-stage error (27). Conclusion. Taking $\mathbb { E } _ { X }$ and using $\mathbb { E } _ { X } \bar { e } ( X ) \le \bar { e } _ { N }$ by Cauchy–Schwarz, the first bound of Lemma E.3,

$$
\bar { e } _ { N } \le \sqrt { \frac { 2 \rho _ { N } ( \eta _ { 1 } ) } { c _ { \star } } } ,
$$

shows that the marginal error contributes at most $\frac { 4 \sqrt { 2 d } } { \gamma \sqrt { e c _ { \star } } } \sqrt { \rho _ { N } ( \eta _ { 1 } ) }$ . The correlation error contributes K times the left-hand side of (31), which by Proposition E.6 is at most $K \bar { \varepsilon } _ { \mathrm { a p p } } + 2 K c _ { 2 } \sqrt { \rho _ { N } ( \eta _ { 1 } ) } + K \rho _ { M } ( \eta _ { 2 } )$ . Both statements hold on the event of Proposition $\operatorname { E . 6 } ,$ which contains that of Lemma E.3 and has probability at least $1 - \delta _ { 1 } - \delta _ { 2 } ;$ collecting the two $\sqrt { \rho _ { N } ( \eta _ { 1 } ) }$ terms into $c _ { 1 }$ proves (35). For the sample complexity, $\delta _ { 1 } = \delta _ { 2 } = \delta / 2$ turns both logarithms into log $\big ( 4 \mathfrak { N } _ { j } ( \eta _ { j } ) / \delta \big )$ , and Lemma E.1 bounds these by q log $\left( 1 + 2 W L _ { j } / \eta _ { j } \right) + \log ( 4 / \delta )$ ; the stated conditions on N and M use the latter and are therefore at least as strong as what the two results require. The choice of $\eta _ { 2 }$ makes 4 $K L _ { 2 } ^ { \prime } \eta _ { 2 } = \varepsilon / 4$ , and the bound on M makes the Hoeffding part of $K \rho _ { M } ( \eta _ { 2 } )$ at most $\varepsilon / 4 ;$ the choice of η makes $2 L _ { 1 } ^ { \prime } \eta _ { 1 } = t / 2$ with $t : = \epsilon ^ { 2 } / ( 4 c _ { 1 } ^ { 2 } )$ , and the bound on N makes the deviation part of $\rho _ { N } ( \eta _ { 1 } )$ at most $t / 2$ , so that $c _ { 1 } \sqrt { \rho _ { N } ( \eta _ { 1 } ) } \le \varepsilon / 2$ . In total the right-hand side of (35) is at most $K \bar { \varepsilon } _ { \mathrm { a p p } } + \varepsilon$ □

## F TOY EXAMPLE

![](images/ff96e3af37995d484e5f0e0de07e6147a252f139cf085e9d415b0b8b6aa00734.jpg)  
Figure 6: Realizations of the signal y for different band limits $B \in \{ 4 , 8 , 1 6 , 3 2 \}$ (upper) and corresponding eigenvalues of the posterior covariance $\pmb { \Sigma } _ { \mathrm { p o s t } }$ (lower).

The toy example considers a reconstruction problem on a periodic grid. We generate a discrete signal $\boldsymbol { y } \in \mathbb { R } ^ { d }$ , which is distributed according to a Gaussian prior $\pmb { y } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } _ { \mathrm { p r } } )$ , where $\pmb { \Sigma } _ { \mathrm { p r } }$ is chosen as circulant with a red power spectrum $\lambda _ { i } \geq 0$ . Then, we impose a band limit, i.e., we set $\lambda _ { i } = 0$ for all $| i | > B$ for some $B \in \mathbb N$ , which implies rank $( { \pmb { \Sigma } } _ { \mathrm { p r } } ) =$ min $( 2 B + 1 , d )$ . Then, we observe a noisy subsample of the signal y as

$$
\begin{array} { r } { \pmb { x } = \pmb { S } \pmb { y } + \pmb { \eta } \in \mathbb { R } ^ { n } , \qquad \pmb { \eta } \sim \mathcal { N } ( \mathbf { 0 } , \sigma ^ { 2 } \pmb { I } _ { n } ) , } \end{array}
$$

Table 5: Frobenius distance $\| \pmb { \Sigma } _ { \phi } - \pmb { \Sigma } _ { \mathrm { p o s t } } \| _ { F }$ between the learned covariance $\Sigma _ { \phi }$ and the true posterior covariance $\pmb { \Sigma } _ { \mathrm { p o s t } }$ across different band limits B. The results are averaged across five seeds with the best model in bold.
<table><tr><td></td><td> $B = 4$ </td><td> $B = 8$ </td><td> $B = 1 6$ </td><td> $B = 3 2$ </td></tr><tr><td> $S _ { k }$ </td><td> $\mathbf { 0 . 0 0 3 5 7 } \pm 3 . 7 \cdot 1 0 ^ { - 6 }$ </td><td> $\mathbf { 0 . 1 0 8 \pm 0 . 0 1 4 }$ </td><td> ${ \bf 0 . 6 1 8 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 5 8 4 \pm 0 . 0 1 0 }$ </td></tr><tr><td> $S _ { \mathrm { l o g } } ~ ( \varepsilon = 1 0 ^ { - 6 } )$ </td><td> $1 . 1 5 \pm 0 . 2 6$ </td><td> $1 . 4 3 \pm 0 . 1 1$ </td><td> $2 . 0 7 \pm 0 . 2 2$ </td><td> $1 . 9 5 \pm 0 . 1 0$ </td></tr><tr><td> $S _ { \mathrm { l o g } }$ </td><td> $0 . 9 5 3 \pm 0 . 1 3$ </td><td> $1 . 3 3 \pm 0 . 1 2$ </td><td> $2 . 0 8 \pm 0 . 2 2$ </td><td> $1 . 9 9 \pm 0 . 2 6$ </td></tr></table>

where $S \in \mathbb { R } ^ { n \times d }$ selects every m-th coordinate with $n = d / m$ and $\sigma ^ { 2 }$ is the variance of the observation noise. Now the task is to reconstruct y given x using a neural network. Since prior and likelihood are Gaussian, the posterior of y given x is also Gaussian. In particular, since y and η are independent, we have that

$$
\begin{array} { r } { \mathrm { C o v } ( y , y ) = \Sigma _ { \mathrm { p r } } , \quad \mathrm { C o v } ( y , x ) = \mathbb { E } [ y ( S y + \eta ) ^ { \top } ] = \Sigma _ { \mathrm { p r } } S ^ { \top } , \quad \mathrm { C o v } ( x , x ) = S \Sigma _ { \mathrm { p r } } S ^ { \top } + \sigma ^ { 2 } I _ { n } . } \end{array}
$$

Then, via Gaussian conditioning, we obtain ${ \pmb y } \mid { \pmb x } \sim \mathcal { N } ( { \pmb \mu } _ { \mathrm { p o s t } } ( { \pmb x } ) , { \pmb \Sigma } _ { \mathrm { p o s t } } )$ , with

$$
\pmb { \mu } _ { \mathrm { p o s t } } ( \pmb { x } ) = \pmb { \Sigma } _ { \mathrm { p r } } \pmb { S } ^ { \top } \big ( \pmb { S } \pmb { \Sigma } _ { \mathrm { p r } } \pmb { S } ^ { \top } + \sigma ^ { 2 } \pmb { I } _ { n } \big ) ^ { - 1 } \pmb { x } ,
$$

$$
\pmb { \Sigma } _ { \mathrm { p o s t } } = \pmb { \Sigma } _ { \mathrm { p r } } - \pmb { \Sigma } _ { \mathrm { p r } } \pmb { S } ^ { \top } \left( \pmb { S } \pmb { \Sigma } _ { \mathrm { p r } } \pmb { S } ^ { \top } + \sigma ^ { 2 } \pmb { I } _ { n } \right) ^ { - 1 } \pmb { S } \pmb { \Sigma } _ { \mathrm { p r } } .
$$

The posterior covariance $\pmb { \Sigma } _ { \mathrm { p o s t } }$ is independent of x, so the task is homoscedastic, and rank $( \Sigma _ { \mathrm { p o s t } } ) = \operatorname* { m i n } ( 2 B + 1 , d )$ . For this toy example, we use $d = 1 2 8 , m = 8 , \sigma ^ { 2 } = 1 0 ^ { - 6 }$ and varying B to analyze the impact of the rank of the posterior on the estimation. Figure 6 shows a realized signal for varying $B ,$ as well as the corresponding eigenvalues of the posterior covariance $\pmb { \Sigma } _ { \mathrm { p o s t } }$ . It can be seen that for $B = 4$ the reconstruction task is deterministic up to the observation noise, which is not visible in the figure. For $B = 8$ the uncertainty in the posterior increases, and for further increasing B, the posterior mean deviates and the standard deviation becomes visible. Thus, for large $B ,$ the signal cannot be exactly reconstructed based on the limited observations.

For the reconstruction, we train a linear map ${ \pmb x } \mapsto { \pmb \mu } _ { \phi }$ and a learnable Cholesky factor $\pmb { \Sigma } _ { \phi } = \pmb { L } \pmb { L } ^ { \top }$ with softplus-constrained diagonal. We train and evaluate on 4096 and 1024 samples, respectively, use a batch size of 256, and train for 5000 steps with Adam (learning rate $1 0 ^ { - 3 }$ for $L , 3 \cdot 1 0 ^ { - 2 }$ for the mean, cosine annealing). We compare the Gaussian kernel score with $\gamma ^ { 2 } = 2 d \cdot \overline { { \mathrm { d i a g } \Sigma _ { \mathrm { p o s t } } } }$ against the log score, unconstrained and with an added floor $\varepsilon I , \varepsilon = 1 0 ^ { - 6 }$ . Results are averaged over five seeds

Table 5 shows the Frobenius norm between the different estimates and the true posterior covariance, averaged across multiple seeds. Evidently, the kernel score obtains the best performance across all band limits. Figure 7 shows visualizations of the true posterior and the corresponding estimates for varying band limit B. For large $B ,$ where $\pmb { \Sigma } _ { \mathrm { p o s t } }$ admits a larger rank, the estimates of the kernel score and the log score are visually indistinguishable, although a significant gap in the Frobenius norm remains. For small $B ,$ the log score cannot resolve the low-rank structure and leads to very noisy and incorrect estimates. Figure 8 visualizes the gradient norms of the different loss functions with respect to the estimated $\Sigma _ { \phi }$ . Even with an added floor, the log score has a gradient norm several orders of magnitude larger than the kernel score, reaching values up to $1 0 ^ { 1 7 }$ . While this improves for increasing B, it remains much larger and only seems to stabilize later during training. This directly connects to Lemma 2.1 and highlights that the log score leads to poor results when parameterized with a neural network and optimized with stochastic gradient descent.

## G EXPERIMENT DETAILS

Every configuration is trained with 5 independent random seeds, and we report the averaged mean and standard deviation across test sets, so that differences between methods can be judged against the run-to-run variability of the training procedure. We use different backbone architectures and only vary the output head, depending on the predictive method. Thus, all methods share the same backbone (and generally, the number of parameters) and are directly comparable. We now further describe our evaluation protocol, as well as the individual datasets and backbones.

## G.1 EVALUATION METRICS

All methods are compared on a common set of evaluation metrics, mostly based on proper scoring rules, with lower values indicating better performance. Below, we define each metric for a single prediction. Unless stated otherwise, reported values

![](images/ca748615a15c6d5b6b5d48ca98110ca0548c553b58433b012ea424220fb6519c.jpg)

Figure 7: True posterior and different estimates for a selected realization and varying band limit $B \in \{ 4 , 8 , 1 6 , 3 2 \}$ . The ranks of the posterior covariance are identical to those in Figure 6.  
![](images/17516d52fdce4583d6e8f7f8aaa9bfc11d8e43784910edc1c8aba056783d5cfe.jpg)  
Figure 8: Gradient norm of the different loss functions with respect to the estimated covariance $\Sigma _ { \phi } ,$ compared across different band limits B.

are averages over all cases of the respective test set.

Fix a test case $( \pmb { x } , \pmb { y } ) \in \mathcal { X } \times \mathbb { R } ^ { d }$ , where x is the input and y the observed target, and let $P = P ( \cdot \mid x )$ denote the predictive distribution returned by the model, with mean $\pmb { \mu } .$ . To ensure a fair comparison against the sample-based method, metrics are evaluated from a simulated ensemble

$$
\pmb { y } ^ { ( 1 ) } , \ldots , \pmb { y } ^ { ( M ) } \stackrel { \mathrm { i . i . d . } } { \sim } P , \qquad M = 1 0 0 ,\tag{36}
$$

using unbiased estimators, unless a closed form is available for all methods. For the deterministic baseline, $P$ is the point mass at $\mu ;$ for the sample-based method, $\pmb { \mu }$ and Σ denote the ensemble mean and covariance of (36).

Mean squared error (MSE). The MSE evaluates the predictive mean $\mu ,$

$$
\mathrm { M S E } ( P , \pmb { y } ) = \frac { 1 } { d } \| \pmb { \mu } - \pmb { y } \| _ { 2 } ^ { 2 } .\tag{37}
$$

It ignores the predicted uncertainty entirely and serves to confirm that differences in the probabilistic scores are not driven by differences in the accuracy of the location.

Continuous ranked probability score (CRPS). The CRPS [13] is a standard univariate proper scoring rule and is applied here to each marginal separately and averaged over the domain,

$$
\mathrm { C R P S } _ { M } ( P , y ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \left( \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left| y _ { i } ^ { ( m ) } - y _ { i } \right| - \frac { 1 } { 2 M ( M - 1 ) } \sum _ { m = 1 } ^ { M } \sum _ { m ^ { \prime } \neq m } \left| y _ { i } ^ { ( m ) } - y _ { i } ^ { ( m ^ { \prime } ) } \right| \right) .\tag{38}
$$

It depends on $P$ only through its univariate margins and therefore measures the marginal fit. The first term rewards accuracy of the ensemble around the observation, the second term rewards spread and thereby prevents overconfident predictions, so that the score is minimized in expectation only when each marginal predictive distribution is calibrated.

Negative log-likelihood (NLL). For a Gaussian prediction $P = \mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ , the NLL per coordinate evaluated at the observation is

$$
\mathrm { N L L } ( P , y ) = \frac { 1 } { 2 d } \Big ( \log \operatorname * { d e t } \bigl ( 2 \pi \Sigma \bigr ) + ( y - \mu ) ^ { \top } \Sigma ^ { - 1 } ( y - \mu ) \Big ) .\tag{39}
$$

It is the only metric requiring an explicit density and is therefore not available for the deterministic and the sample-based baseline. It is strictly proper, rewards concentration of the predictive density at the observation, and is unbounded as Σ approaches the boundary of the positive definite cone, which is precisely the behaviour analysed in Section 2.3.

Energy score (ES). The energy score [13] is the multivariate generalisation of the CRPS, ES $( P , \pmb { y } ) = \mathbb { E } \| \pmb { Y } - \pmb { y } \| _ { 2 } -$ $\begin{array} { r l r } {  { { \frac { 1 } { 2 } } \mathbb { E } \| { \pmb Y } - { \pmb Y } ^ { \prime } \| _ { 2 } } } \end{array}$ with $Y , Y ^ { \prime } \sim P$ independent, estimated from (36) by

$$
\mathrm { E S } _ { M } ( P , y ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left\| y ^ { ( m ) } - y \right\| _ { 2 } - \frac { 1 } { 2 M ( M - 1 ) } \sum _ { m = 1 } ^ { M } \sum _ { m ^ { \prime } \neq m } \left\| y ^ { ( m ) } - y ^ { ( m ^ { \prime } ) } \right\| _ { 2 } .\tag{40}
$$

The pairwise normalisation $M ( M - 1 )$ makes the estimator unbiased, so that its expectation does not reward larger ensembles [60]. The energy score is strictly proper but is known to have limited sensitivity to misspecification of the dependence structure.

Gaussian kernel score (KS). For the Gaussian kernel $k ( \pmb { u } , \pmb { v } ) = \mathrm { e x p } \big ( - \| \pmb { u } - \pmb { v } \| _ { 2 } ^ { 2 } / \gamma ^ { 2 } \big )$ with bandwidth $\gamma > 0$ , the associated kernel score [13] is $\begin{array} { r } { \mathrm { K S } ( P , \pmb { y } ) = \frac { 1 } { 2 } \mathbb { E } k ( \pmb { Y } , \pmb { Y } ^ { \prime } ) - \mathbb { E } k ( \pmb { Y } , \pmb { y } ) + \frac { 1 } { 2 } \dot { k } ( \pmb { y } , \pmb { y } ) } \end{array}$ with $Y , \dot { Y ^ { \prime } } \sim P$ independent, estimated by

$$
\mathrm { K S } _ { M } ( P , y ) = \frac { 1 } { 2 M ( M - 1 ) } \sum _ { m = 1 } ^ { M } \sum _ { m ^ { \prime } \neq m } k \bigl ( y ^ { ( m ) } , y ^ { ( m ^ { \prime } ) } \bigr ) - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } k \bigl ( y ^ { ( m ) } , y \bigr ) + \frac { 1 } { 2 } ,\tag{41}
$$

using $k ( \pmb { y } , \pmb { y } ) = 1$ . We evaluate at $\gamma ^ { 2 } = d \left( { \mathrm { A p p e n d i x } } { \mathrm { C } } \right)$ . It is the sample counterpart of one half of the squared MMD between $P$ and the point mass at $^ { y , }$ and is strictly proper. In contrast to the NLL, it is bounded, which makes it comparable across all training objectives even when a model has degenerate predicted covariance.

Variogram score (VS). The variogram score of order α [61] compares observed and predicted increments,

$$
\mathrm { V S } ^ { ( \alpha ) } ( P , y ) \ = \ \frac { 1 } { \sum _ { i \neq j } w _ { i j } } \sum _ { i \neq j } w _ { i j } \Big ( \vert y _ { i } - y _ { j } \vert ^ { \alpha } \ - \ \mathbb { E } \big \vert Y _ { i } - Y _ { j } \big \vert ^ { \alpha } \Big ) ^ { 2 } , \qquad Y \sim P .\tag{42}
$$

It is proper but not strictly proper. Being a function of increments only, it is invariant to a common shift of the field and hence blind to a constant bias in the mean, which is what makes it a targeted measure of the predicted dependence structure. We select $\alpha = 1$ , for which $Y _ { i } - Y _ { j } \sim { \mathcal { N } } ( \mu _ { i } - \mu _ { j } , \Sigma _ { i i } + \Sigma _ { j j } - 2 \Sigma _ { i j } )$ under a Gaussian $P = \mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } )$ , so that $\mathbb { E } | Y _ { i } - Y _ { j } |$ is the mean of a folded normal distribution and available in closed form. Following Scheuerer and Hamill [61], the weights $w _ { i j }$ are inversely proportional to the distance between coordinates where a physical distance is available and uniform otherwise.

## G.2 CALIBRATION

While the previous metrics provide average performance scores, this does not reveal how a method fails, whether through a biased mean, a misspecified marginal scale, or a dependence structure that is too weak or too strong. We therefore consider evaluation beyond pure score comparison by analyzing the calibration of the predictive distribution. In particular, we consider

probabilistic calibration: the observation should behave like a draw from its own predictive distribution [6, 62]. This type of calibration can easily be analyzed by checking whether the probability integral transform (PIT) of the prediction is uniformly distributed.

Let $( \pmb { x } _ { t } , \pmb { y } _ { t } ) _ { t = 1 } ^ { T }$ denote the evaluation set with $\boldsymbol { y } _ { t } \in \mathbb { R } ^ { d }$ , and let $F _ { t } = \mathcal { N } ( \mu _ { t } , \Sigma _ { t } )$ be the associated predictive distribution, with marginal variances $\sigma _ { t , i } ^ { 2 } = ( \Sigma _ { t } ) _ { i i }$ . All diagnostics test the null hypothesis $\mathbf { \boldsymbol { y } } _ { t } \sim \boldsymbol { F } _ { t }$ . First, we consider margina calibration, which isolates the pointwise fit. Since the margins of $F _ { t }$ are Gaussian, the PIT is available in closed form,

$$
p _ { t , i } ~ = ~ \Phi \left( { \frac { y _ { t , i } - \mu _ { t , i } } { \sigma _ { t , i } } } \right) , \qquad i = 1 , \dots , d ,\tag{43}
$$

and is uniform on [0, 1] under calibration. We pool $\{ p _ { t , i } \}$ over cases and components and inspect the resulting histogram. This diagnostic depends on $\Sigma _ { t }$ only through its diagonal and verifies that all marginals are calibrated. However, it cannot discriminate between different dependence structures, which requires incorporating the full covariance matrix.

To check multivariate calibration, we follow [63] and use rank histograms, which can be thought of as a discrete version of the PIT histogram. The rank histogram rests on the observation that if the observation and M ensemble members are exchangeable, the rank of the observation within the pooled sample is uniform on $\{ 1 , \ldots , M + 1 \}$ . By analyzing the corresponding histogram, one can find systematic departures from uniformity, which indicate specific forms of miscalibration. Since $F _ { t }$ is available in closed form, we simulate $z _ { t } ^ { ( 1 ) } , \ldots , z _ { t } ^ { ( M ) } \overset { \mathrm { i i d } } { \sim } F _ { t }$ with $M = 1 0 0 0$ , which are exchangeable with y under the null. As there is no canonical order on $\mathbb { R } ^ { d }$ , ranking requires a pre-rankfunction $[ 6 3 ] \rho _ { t } : \mathbb { R } ^ { d }  \mathbb { R } .$ , giving

$$
r _ { t } = 1 + \sum _ { m = 1 } ^ { M } \Im \Big \{ \rho _ { t } ( z _ { t } ^ { ( m ) } ) < \rho _ { t } ( y _ { t } ) \Big \} \sim \mathcal { U } \{ 1 , \dots , M + 1 \} .\tag{44}
$$

Here, uniformity is necessary but not sufficient, since a pre-rank function reduces the field to a scalar summary. We therefore use three simple pre-rank functions, chosen so that the shape of the histogram maps onto an identifiable deficiency, whose histograms confound location, scale and dependence. The three are stated for a generic $\pmb { v } \in \mathbb { R } ^ { d }$ and applied to ${ \mathbf { } } _ { \pmb { y } _ { t } }$ and to each $\boldsymbol { z } _ { t } ^ { ( m ) }$ alike.

Mean calibration. The spatial average

$$
{ \rho } ^ { \mathrm { m e a n } } ( \pmb { v } ) = \frac { 1 } { d } \mathbf { 1 } ^ { \top } \pmb { v }\tag{45}
$$

targets the aggregate level of the field. Here, we can utilize a closed form, since $\rho ^ { \mathrm { m e a n } } ( \pmb { v } ) \sim \mathcal { N } \big ( d ^ { - 1 } \pmb { 1 } ^ { \top } \pmb { \mu } _ { t } , ~ d ^ { - 2 } \pmb { 1 } ^ { \top } \pmb { \Sigma } _ { t } \pmb { 1 } \big )$ for $v \sim F _ { t } ,$ , so (44) may be replaced by the exact PIT. The only quantity required is the scalar $\mathbf { 1 } ^ { \top } \pmb { \Sigma } _ { t } \mathbf { 1 }$ , which is inexpensive for all covariance families considered.

Scale calibration. The empirical variance of a single realisation,

$$
{ \rho } ^ { \mathrm { v a r } } ( \pmb { v } ) = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \big ( v _ { i } - \rho ^ { \mathrm { m e a n } } ( \pmb { v } ) \big ) ^ { 2 } ,\tag{46}
$$

quantifies the amount of spatial contrast present in the field and is invariant to its overall level, making it complementary to (45). For fixed margins, stronger predicted correlation reduces the expected contrast, so this pre-rank penalizes predictions whose realisations are systematically too smooth or too rough.

Dependence calibration. A variogram-type statistic over a set $\mathcal { P } _ { h }$ of index pairs at spatial lag $h ,$

$$
\rho _ { h } ^ { \mathrm { v g } } ( \pmb { v } ) \ = \ \frac { 1 } { \vert \mathcal { P } _ { h } \vert } \sum _ { ( i , j ) \in \mathcal { P } _ { h } } w _ { i j } \left. v _ { i } - v _ { j } \right. ^ { \alpha } , \qquad \alpha \in ( 0 , 2 ] ,\tag{47}
$$

measures the strength of dependence at a prescribed scale. Being a function of increments only, it is exactly invariant to constant shifts and hence insensitive to bias in the mean, which separates it cleanly from (45). We follow [63] and use $\alpha = 1$ and set $w _ { i j } = 1$ , except for EUPPBench, where we weight each pair by the inverse distance $w _ { i j } = \lVert \pmb { s } _ { i } - \pmb { s } _ { j } \rVert ^ { - 1 }$ between the two locations.

To make visualization more concise, we do not plot the rank histograms directly, but their ECDF, i.e., $F _ { u } ( u )$ against the normalized rank u. A perfect diagonal is consistent with calibration in terms of the selected pre-rank function only. A curve crossing the diagonal from above indicates that the observed summary falls too often in the extreme ranks, i.e., underdispersion in that aspect, while a crossing from below indicates overdispersion. A curve staying on one side indicates that the observed summary is systematically smaller (above) or larger (below) than its forecast counterpart. Concretely, a one-sided curve in $\rho ^ { \mathrm { m e a n } }$ indicates a bias in the domain-averaged field and a crossing from above that $\mathbf { 1 } ^ { \top } \pmb { \Sigma } _ { t } \mathbf { 1 }$ is too small, which for calibrated margins means correlations that are too weak. A curve below the diagonal in $\rho ^ { \mathrm { v a r } }$ indicates that observed fields carry more spatial contrast than simulated ones, i.e. that the forecast is too smooth, and a one-sided curve in $\rho _ { h } ^ { \mathrm { v g } }$ that dependence at lag h is over- or underestimated. Since uniformity is necessary but not sufficient, compensating errors may cancel in a summary, and we thus report the three pre-rank diagnostics jointly alongside the marginal one.

## G.3 TASK DESCRIPTIONS

All four tasks follow the same experimental protocol; only the data, the backbone and the hyperparameters in Table 6 differ. Every method shares a task-specific feature extractor (the trunk) and differs only in its probabilistic head and loss: the single-stage baselines attach one head for the mean and marginal variances and, where applicable, a second head for the low-rank factors, and are trained with either the log score or the kernel score; SCORE attaches the same first head and a second head for the spectral core, and is trained in two stages, the marginals with the CRPS and the core with the kernel score, using the bandwidths of Appendix C. Each method receives the same total epoch budget; for SCORE it is split between the two stages, and during the second stage the trunk and the marginal head are frozen so that only the core head is trained. Unless stated otherwise, optimization uses AdamW [64] with a cosine learning-rate schedule and three warmup epochs, all inputs and targets are standardized with statistics computed on the training portion only, and evaluation uses the best validation checkpoint. Sample-based scores are computed from M = 100 predictive samples, and every configuration is run with five seeds(1−5).

Table 6: Training configurations for the four tasks. “Epochs” gives the total budget and, in parentheses, the marginal/core split for SCORE.
<table><tr><td></td><td>Time series</td><td>Surface temp.</td><td>Depth</td><td>Post-processing</td></tr><tr><td>Trunk</td><td>PatchTST [28]</td><td>U-Cast [34]</td><td>DepthAnything v2 [32]</td><td>GNN [38]</td></tr><tr><td>Trunk params</td><td>81.8 k-0.9M</td><td>65.9M</td><td>22.1 M (frozen) + 2.7 M</td><td>10.1M</td></tr><tr><td>Epochs</td><td>100 (75/25)</td><td>30 (20/10)</td><td>75 (50/25)</td><td>40 (30/10)</td></tr><tr><td>Batch size</td><td>128 (32 ILI, Electricity; 24 Traffic)</td><td>16</td><td>8</td><td>8</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 } / 2 . 5 \cdot 1 0 ^ { - 3 } \mathrm { ( I L I ) }$ </td><td> $3 \cdot 1 0 ^ { - 4 } \ : ( 7 \cdot 1 0 ^ { - 5 } \mathrm { c o r e } )$ </td><td>10⁻3</td><td> $2 \cdot 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.05 (0 Weather, Electricity, Traffic)</td><td>0.1</td><td>0.05</td><td>0.05</td></tr></table>

Time series benchmark. We follow the long-term forecasting protocol of PatchTST [28] on the eight data sets summarized in Table 7. Each example consists of a contiguous look-back window of length L followed immediately by the prediction horizon T. For the ETT data sets we study both multivariate forecasting of all variables and univariate forecasting of the target variable OT; ILI, Weather, Electricity and Traffic are evaluated in the multivariate setting only. Electricity and Traffic are the two largest data sets, recording the hourly consumption of 321 clients and the hourly road occupancy of 862 sensors, so that the output dimension $d = C T$ reaches 61,632 and 165,504 at $T = 1 9 2$ . The backbone is PatchTST, an encoder-only Transformer whose channel-independent design processes each channel as a separate univariate series while sharing all weights across channels; accordingly, all predictive heads model each channel independently. We use the per-data-set configurations of Nie et al. [28] (patch length $P = 1 6$ and stride S = 8 for L = 336, P = 24 and $S = 2$ for ILI, both yielding 42 patches; $( d _ { \mathrm { m o d e l } } , H , d _ { \mathrm { f f } } ) = ( 1 6 , 4 , 1 2 8 )$ with dropout 0.3 for ETTh1/2 and ILI, and (128, 16, 256) with dropout 0.2 for ETTm1/2, Weather, Electricity and Traffic). For the ETT data sets and ILI, optimization follows the common protocol. For Weather we instead follow the PatchTST recipe for that data set: the learning rate is held constant for three epochs and then multiplied by 0.9 after every epoch, without weight decay and with early stopping on the validation loss (patience 20); for SCORE this recipe is used for the marginal fit, while the core fit keeps the cosine schedule with warmup. For Electricity and Traffic we likewise follow the PatchTST recipes for these data sets, a one-cycle schedule with 20 warmup epochs over the 100-epoch budget, no weight decay, early stopping on the validation loss (patience 10) and batch sizes of 32 and 24 (Table 6); again, SCORE uses this recipe for the marginal fit and the cosine schedule with warmup for the core fit.

Surface temperature prediction. We use ERA5 reanalysis data [33] as provided by WeatherBench2 [65]. Similar to Bülte et al. [66], Pacchiardi et al. [25], we consider a European domain at $0 . 2 5 ^ { \circ }$ resolution, $3 5 ^ { \circ } - 7 5 ^ { \circ } \mathrm { N }$ and $1 2 . 5 ^ { \circ } \mathrm { W } \mathrm { - } 4 2 . 5 ^ { \circ } \mathrm { E } ,$ i.e. a 160 × 220 grid with d = 35,200 target coordinates, and one observation per day at 12 UTC. The data are split by year into training (1990–2014), validation (2015–2019) and test (2020–2022). The input stacks the last two daily observations of five dynamic variables (the 10 m wind components U10 and V10, temperature at 2 m and 850 hPa, and geopotential at 500 hPa) with the land-sea mask and the surface geopotential, giving twelve channels. The target is the change of the standardized 2 m temperature over a lead time of three days, so the predictive distribution is over a single channel on the full grid. As trunk we use the U-Net of U-Cast [34], a recent model that matches state-of-the-art weather forecasters at a fraction of their size, reduced to 96 base channels because of the smaller number of input channels. The head for the mean and marginal variances has 0.17 M parameters; the low-rank models and SCORE add a second head with 0.5 M parameters. In the second stage of SCORE, the core head is trained at the reduced learning rate given in Table 6.

Table 7: Time-series data sets. Splits are chronological. ETT uses 12/4/4 months, the other data sets use 70/10/20 %.
<table><tr><td>Data set</td><td>Variables</td><td>Resolution</td><td>Observations</td><td>Split</td><td>L</td><td>T</td></tr><tr><td>ETTh1, ETTh2</td><td>7</td><td>hourly</td><td>17,420</td><td>12/4/4 m</td><td>336</td><td>{96, 192}</td></tr><tr><td>ETTm1, ETTm2</td><td>7</td><td>15 min</td><td>69,680</td><td>12/4/4 m</td><td>336</td><td>{96, 192}</td></tr><tr><td>Weather</td><td>21</td><td>10 min</td><td>52,696</td><td>70/10/20</td><td>336</td><td>{96, 192}</td></tr><tr><td>Electricity</td><td>321</td><td>hourly</td><td>26,304</td><td>70/10/20</td><td>336</td><td>{96, 192}</td></tr><tr><td>Traffic</td><td>862</td><td>hourly</td><td>17,544</td><td>70/10/20</td><td>336</td><td>{96, 192}</td></tr><tr><td>ILI</td><td>7</td><td>weekly</td><td>966</td><td>70/10/20</td><td>104</td><td>{24, 36}</td></tr></table>

Monocular depth estimation. We predict a depth map from a single RGB image on NYU Depth v2 [31], following the protocol of Liu et al. [5]. Training uses the 24,231 synchronized image–depth pairs of the standard split, with a fixed 10% held out for validation, and evaluation uses the 654 densely annotated images of the test split of Eigen et al. [67]. All frames are restricted to the standard evaluation crop of the 480 × 640 images, i.e. $d = 4 2 6 \times 5 6 0 = 2 3 8 { , } 5 6 0$ pixels. Depth is stored in millimetres, capped at 10 m and rescaled to [0, 1]. The raw training depth contains projection holes (about 14% of the pixels within the crop), which are filled once by nearest-neighbour interpolation at dataset construction; the test depth maps are the official densely in-painted ones. The trunk is the Depth Anything V2 architecture [32], i.e. a frozen DINOv2 ViT-S/14 encoder [68] and a trainable DPT decoder [69], whose scalar output layer is replaced by two convolutional heads of 0.05 M parameters each. Inputs are padded to a multiple of the patch size and the prediction is cropped back to the original resolution.

Graph-based temperature post-processing. The final task shows how our approach extends to non-equidistant domains and the corresponding unitary transforms. We consider post-processing of two-meter temperature across central Europe: the model takes an ensemble of numerical weather prediction (NWP) forecasts of multiple variables as input and predicts the observed temperature at a set of stations at a lead time of 72 h. We use the EUPPBench data [37], which pairs ECMWF medium-range ensemble forecasts with station observations at 122 European stations from 1997 to 2018 and comprises 730 daily operational forecasts with 51 members as well as 4180 reforecasts (NWP runs for past dates) with 11 members, each with 31 variables. We follow the data configuration and the graph neural network of Feik et al. [38], training on reforecasts from 1997–2013 and evaluating on reforecasts from 2014–2017 and on operational forecasts from 2017–2018, but replace their marginal model by a multivariate Gaussian with the covariance parameterizations under study. We use a reduced set of d = 115 stations, as the remaining stations have more than 50% missing values.

The task requires a different transform than the previous ones. Let $G = ( V , E , w )$ be a graph on the d output coordinates, e.g. a k-nearest-neighbour graph with Gaussian edge weights $w _ { i j } = \exp ( - \mathrm { d i s t } ( i , j ) ^ { 2 } / 2 \varepsilon ^ { 2 } )$ built from the Euclidean distance between stations. With $W = \left( w _ { i j } \right)$ and degree matrix $\Delta _ { w } = \mathrm { d i a g } ( W 1 )$ , the symmetric normalized Laplacian $L = I - \Delta _ { w } ^ { - 1 / 2 } W \Delta _ { w } ^ { - 1 / 2 } = U \Lambda U ^ { * }$ is Hermitian positive semi-definite, so U is unitary and $B : = U ^ { * }$ defines the graph Fourier transform. A diagonal core Σ<sup>ˆ</sup> = diag $\left( g _ { \boldsymbol { \theta } } ( \mathbf { \Lambda } \boldsymbol { \Lambda } ) \right)$ amounts to filtering the graph spectrum; we use the Matérn filter

$$
\begin{array} { r } { g _ { \pmb { \theta } } ( \lambda ) = \Big ( \frac { 2 \nu } { \kappa ^ { 2 } } + \lambda \Big ) ^ { - \nu } , \qquad \pmb { \theta } = ( \nu , \kappa ) \in \mathbb { R } _ { > 0 } ^ { 2 } . } \end{array}
$$

Unlike the DFT, a general graph is not vertex-transitive and $| U _ { i k } | ^ { 2 } \neq 1 / d ,$ , so the implied marginal variances

$$
\Delta _ { i } ( \pmb \theta ) = \sum _ { k } g _ { \pmb \theta } ( \lambda _ { k } ) | U _ { i k } | ^ { 2 }
$$

are non-constant and the normalization must be carried out explicitly. It is available in closed form, however, and merges

with D into the single diagonal $A : = D \Delta ( \theta ) ^ { - 1 / 2 }$ , giving

$$
\Sigma _ { P } = A U g _ { \theta } ( \Lambda ) U ^ { * } A ,
$$

so the staged objective of Proposition 3.4 applies verbatim: $\hat { D }$ carries the marginals exactly and $g _ { \pmb { \theta } }$ is fit to the correlation structure alone. There is no computational gain in this case, since the resulting matrices are dense and their evaluation has the same complexity as a Cholesky decomposition.

## H DETAILED RESULTS

## H.1 NUMERICAL RESULTS

The full results are given in Tables 8–18. The log score is reported per coordinate; lower is better for all scores, and we report the mean and standard deviation over five seeds. The best model within one standard deviation is highlighted in bold. The average rank is computed per seed over the scores available for all methods and then averaged over seeds. When two methods are equally good, the corresponding ranks are averaged. A dash marks a score that is not defined for a method.

Time series benchmark. On the time series benchmark (Tables 8–15), SCORE attains the best average rank on 20 of the 24 tasks, with a mean rank of 2.2 against 3.1 for the strongest baseline, LorD trained with the kernel score. At fixed covariance, replacing the NLL by the kernel score improves LorD on at least 19 of the 24 tasks on each of MSE, CRPS, ES, VS and KS, 105 of the 120 task-score pairs in total. Only the log score, which the NLL-trained model optimizes directly, goes the other way. On the three high-dimensional data sets, Weather, Electricity and Traffic, with d between 2,016 and 165,504, LorD trained with the NLL deteriorates sharply on Weather and collapses on Electricity and Traffic, while trained with the kernel score it is on par with the diagonal model (Section 2.3). With the kernel score as the loss function, SCORE achieves a lower CRPS than the diagonal and low-rank models on 23 of the 24 tasks and a lower variogram score on 22 and 21 of the 24 tasks. The CRPS gain comes from the first stage, which fits the marginals with the CRPS, and the variogram gain from the second stage, which adds the correlation along the horizon. ES and KS change by less than 1% in the median, as they are less sensitive to the dependence structure. These results support the assumption of the circulant core, since the target is a window of $T$ consecutive values whose forecast errors persist and follow the cycles of the series, so their correlation depends on the lag between two lead times rather than on the position in the window. The generative sampler is outperformed by SCORE on CRPS and VS on every task, and SCORE runs at close to the cost of the diagonal model (Table 19). On Weather and Electricity, the log score of every covariance trained with the kernel score, SCORE included, explodes, which the bounded kernel score does not penalize (Theorem 3.1).

Table 8: Test scores on ETTh1.
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td colspan="9">Univariate, T = 96</td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td>0.053 ± 0.001</td><td>0.176 ± 0.001</td><td>2.041 ± 0.010</td><td>0.034 ± 0.000</td><td>0.050 ± 0.001</td><td></td><td>7.3</td></tr><tr><td>Diag</td><td>NLL</td><td>0.053 ± 0.000</td><td>0.128 ± 0.001</td><td>1.528 ± 0.006</td><td>0.031 ± 0.001</td><td>0.049 ± 0.000</td><td>+0.043 ± 0.011</td><td>5.2</td></tr><tr><td>Diag</td><td>KS</td><td>0.052 ± 0.000</td><td>0.127 ± 0.000</td><td>1.502 ± 0.006</td><td>0.026 ± 0.000</td><td>0.048 ± 0.000</td><td>+0.113 ± 0.012</td><td>3.8</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.067 ± 0.001</td><td>0.148 ± 0.002</td><td>1.696 ± 0.016</td><td>0.025 ± 0.000</td><td>0.061 ± 0.001</td><td>-0.363 ± 0.016</td><td>6.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.052 ± 0.000</td><td>0.128 ± 0.000</td><td>1.499 ± 0.005</td><td>0.024 ± 0.000</td><td>0.048 ± 0.000</td><td>−0.262 ± 0.015</td><td>3.2</td></tr><tr><td>Chol</td><td>NLL</td><td>0.071 ± 0.000</td><td>0.153 ± 0.001</td><td>1.742 ± 0.009</td><td>0.024 ± 0.000</td><td>0.064 ± 0.000</td><td>−1.146 ± 0.013</td><td>6.3</td></tr><tr><td>Chol</td><td>KS</td><td>0.053 ± 0.001</td><td>0.129 ± 0.001</td><td>1.500 ± 0.009</td><td>0.024 ± 0.000</td><td>0.049 ± 0.001</td><td></td><td>3.6</td></tr><tr><td>SB</td><td>KS</td><td>0.055 ± 0.002</td><td>0.139 ± 0.006</td><td>1.654 ± 0.053</td><td>0.031 ± 0.000</td><td>0.051 ± 0.002</td><td></td><td>6.2</td></tr><tr><td>Spectral space</td><td>CRPS+KS</td><td>0.052 ± 0.000</td><td>0.127 ± 0.000</td><td>1.478 ± 0.006</td><td>0.023 ± 0.000</td><td>0.048 ± 0.000</td><td>−1.009 ± 0.006</td><td>1.9</td></tr><tr><td colspan="9">SCORE</td></tr><tr><td>Univariate, T = 192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td>0.068 ± 0.001</td><td>0.202 ± 0.001</td><td>3.368 ± 0.016</td><td>0.050 ± 0.001</td><td>0.064 ± 0.001</td><td></td><td>7.7</td></tr><tr><td>Diag</td><td>NLL</td><td>0.068 ± 0.001</td><td>0.147 ± 0.001</td><td>2.492 ± 0.013</td><td>0.035 ± 0.001</td><td>0.063 ± 0.001</td><td>+0.196 ± 0.018</td><td>5.1</td></tr><tr><td>Diag</td><td>KS</td><td>0.065 ± 0.000</td><td>0.144 ± 0.001</td><td>2.436 ± 0.008</td><td>0.032 ± 0.000</td><td>0.060 ± 0.000</td><td>+0.273 ± 0.030</td><td>3.0</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.076 ± 0.000</td><td>0.160 ± 0.000</td><td>2.658 ± 0.008</td><td>0.034 ± 0.000</td><td>0.070 ± 0.000</td><td>−0.115 ± 0.010</td><td>6.5</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.065 ± 0.000</td><td>0.146 ± 0.000</td><td>2.440 ± 0.007</td><td>0.032 ± 0.000</td><td>0.060 ± 0.000</td><td>+0.022 ± 0.037</td><td>3.5</td></tr><tr><td>Chol</td><td>NLL</td><td>0.078 ± 0.000</td><td>0.162 ± 0.000</td><td>2.676 ± 0.006</td><td>0.032 ± 0.000</td><td>0.071 ± 0.000</td><td>−1.100 ± 0.010</td><td>6.2</td></tr><tr><td>Chol</td><td>KS</td><td>0.065 ± 0.001</td><td>0.145 ± 0.001</td><td>2.419 ± 0.013</td><td>0.033 ± 0.000</td><td>0.060 ± 0.000</td><td></td><td>3.0</td></tr><tr><td>SB</td><td>KS</td><td>0.069 ± 0.001</td><td>0.154 ± 0.001</td><td>2.664 ± 0.018</td><td>0.044 ± 0.001</td><td>0.063 ± 0.001</td><td></td><td>6.6</td></tr><tr><td>Spectral space</td><td>CRPS+KS</td><td>0.066 ± 0.001</td><td>0.145 ± 0.001</td><td>2.410 ± 0.013</td><td>0.031 ± 0.000</td><td>0.060 ± 0.001</td><td></td><td></td></tr><tr><td colspan="9">SCORE</td></tr><tr><td colspan="9">Multivariate, T = 96 Physical space</td></tr><tr><td></td><td></td><td>0.377 ± 0.002</td><td>0.400 ± 0.001</td><td>15.577 ± 0.043</td><td>0.567 ± 0.003</td><td>0.305 ± 0.001</td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td></td><td>0.286 ± 0.001</td><td></td><td></td><td></td><td></td><td>6.6</td></tr><tr><td>Diag</td><td>NLL</td><td>0.370 ± 0.003</td><td></td><td>11.069 ± 0.038</td><td>0.507 ± 0.004</td><td>0.257 ± 0.001</td><td>+0.660 ± 0.005</td><td>3.0</td></tr><tr><td>Diag</td><td>KS</td><td>0.369 ± 0.002</td><td>0.298 ± 0.001</td><td>11.052 ± 0.028</td><td>0.529 ± 0.002</td><td>0.257 ± 0.001</td><td>+0.795 ± 0.017</td><td>3.8</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.378 ± 0.006</td><td>0.287 ± 0.002</td><td>11.167 ± 0.082</td><td>0.514 ± 0.007</td><td>0.260 ± 0.002</td><td>+0.617 ± 0.009</td><td>3.5</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.369 ± 0.001</td><td>0.297 ± 0.001</td><td>11.041 ± 0.026</td><td>0.527 ± 0.002</td><td>0.256 ± 0.001</td><td>+0.758 ± 0.023</td><td>3.7</td></tr><tr><td>SB</td><td>KS</td><td>0.388 ± 0.006</td><td>0.332 ± 0.018</td><td>12.875 ± 0.726</td><td>0.567 ± 0.006</td><td>0.289 ± 0.012</td><td></td><td>6.2</td></tr><tr><td colspan="9">Spectral space</td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>0.366 ± 0.001</td><td>0.280 ± 0.001</td><td>10.971 ± 0.019</td><td>0.495 ± 0.002</td><td>0.254 ± 0.001</td><td>+0.153 ± 0.008</td><td>1.2</td></tr><tr><td colspan="9">Multivariate, T = 192 Physical space</td></tr><tr><td>Det</td><td>MSE</td><td>0.415 ± 0.001</td><td>0.422 ± 0.001</td><td>23.251 ± 0.024</td><td>0.622 ± 0.001</td><td>0.332 ± 0.001</td><td></td><td></td></tr><tr><td>Diag</td><td>NLL</td><td>0.415 ± 0.003</td><td>0.311 ± 0.003</td><td>16.662 ± 0.063</td><td>0.566 ± 0.009</td><td>0.281 ± 0.001</td><td>+0.790 ± 0.016</td><td>6.5 3.5</td></tr><tr><td></td><td>KS</td><td>0.409 ± 0.002</td><td>0.319 ± 0.001</td><td>16.544 ± 0.052</td><td>0.590 ± 0.005</td><td>0.278 ± 0.001</td><td>+0.876 ± 0.005</td><td></td></tr><tr><td>Diag LorD r=1</td><td>NLL</td><td>0.427 ± 0.002</td><td>0.314 ± 0.003</td><td>16.847 ± 0.024</td><td>0.577 ± 0.008</td><td>0.284 ± 0.001</td><td>+0.749 ± 0.024</td><td>3.6 4.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.410 ± 0.002</td><td>0.318 ± 0.001</td><td>16.538 ± 0.044</td><td>0.586 ± 0.003</td><td>0.278 ± 0.001</td><td></td><td>3.4</td></tr><tr><td>SB</td><td>KS</td><td>0.455 ± 0.062</td><td>0.344 ± 0.008</td><td>18.568 ± 0.290</td><td>0.794 ± 0.308</td><td>0.306 ± 0.004</td><td>+0.855 ± 0.008</td><td>5.2</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>0.406 ± 0.002</td><td>0.299 ± 0.001</td><td>16.423 ± 0.032</td><td>0.541 ± 0.003</td><td>0.275 ± 0.001</td><td>+0.290 ± 0.027</td><td>1.3</td></tr></table>

Table 9: Test scores on ETTh2.
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td colspan="2">Univariate, T = 96</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Physical space</td><td> $0 . 1 2 5 \pm 0 . 0 0 2$ </td><td> $0 . 2 8 2 \pm 0 . 0 0 2$ </td><td> $3 . 2 9 5 \pm 0 . 0 2 1$ </td><td> $0 . 1 0 3 \pm 0 . 0 0 2$ </td><td> $0 . 1 1 5 \pm 0 . 0 0 1$ </td><td></td><td>7.5</td></tr><tr><td>Det</td><td>MSE NLL</td><td> $\mathbf { 0 . 1 2 0 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 1 9 4 \pm 0 . 0 0 0 }$ </td><td> $2 . 3 5 4 \pm 0 . 0 0 3$ </td><td> $0 . 1 0 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 0 5 \pm 0 . 0 0 0 }$ </td><td> $+ 0 . 3 5 1 \pm 0 . 0 0 4$ </td><td>3.7</td></tr><tr><td>Diag Diag</td><td>KS</td><td></td><td></td><td></td><td></td><td> $\mathbf { 0 . 1 0 6 \pm 0 . 0 0 1 }$ </td><td> $+ 0 . 4 0 0 \pm 0 . 0 0 9$ </td><td>4.3</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 1 2 2 \pm 0 . 0 0 1$ </td><td> $0 . 1 9 7 \pm 0 . 0 0 1$ </td><td> $2 . 3 6 2 \pm 0 . 0 1 0$ </td><td> $0 . 0 9 5 \pm 0 . 0 0 1$ </td><td></td><td></td><td>6.1</td></tr><tr><td>LorD r=1</td><td></td><td> $0 . 1 5 7 \pm 0 . 0 0 2$ </td><td> $0 . 2 2 7 \pm 0 . 0 0 3$ </td><td> $2 . 6 3 1 \pm 0 . 0 2 5$ </td><td> $0 . 0 9 1 \pm 0 . 0 0 0$ </td><td> $0 . 1 2 9 \pm 0 . 0 0 2$ </td><td> $+ 0 . 0 4 0 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td></td><td>KS</td><td> $0 . 1 2 4 \pm 0 . 0 0 2$ </td><td> $0 . 2 0 0 \pm 0 . 0 0 2$ </td><td> $2 . 3 5 2 \pm 0 . 0 1 8$ </td><td> $0 . 0 8 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 0 6 \pm 0 . 0 0 2 }$ </td><td> $+ 0 . 0 9 0 \pm 0 . 0 0 6$ </td><td>3.7</td></tr><tr><td>Chol Chol</td><td>NLL</td><td> $0 . 2 0 4 \pm 0 . 0 1 1$ </td><td> $0 . 2 6 8 \pm 0 . 0 0 9$ </td><td> $3 . 0 7 5 \pm 0 . 1 0 8$ </td><td> $0 . 1 0 6 \pm 0 . 0 0 3$ </td><td> $0 . 1 6 3 \pm 0 . 0 0 9$ </td><td> $\mathbf { - 0 . 6 9 0 \pm 0 . 3 7 3 }$ </td><td>7.2</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 1 2 6 \pm 0 . 0 0 3$ </td><td> $0 . 2 0 0 \pm 0 . 0 0 3$ </td><td> $\mathbf { 2 . 3 4 7 \pm 0 . 0 2 9 }$ </td><td> $\mathbf { 0 . 0 8 7 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 1 0 6 \pm 0 . 0 0 2 }$ </td><td> $\left( + 4 . 1 6 2 \pm 9 . 1 0 8 \right) \times { 1 0 } ^ { 1 3 }$ </td><td>4.0</td></tr><tr><td>Spectral space</td><td>KS</td><td> $0 . 1 2 7 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 0 \pm 0 . 0 0 8$ </td><td> $2 . 5 4 1 \pm 0 . 0 8 4$ </td><td> $0 . 0 9 7 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 1 0 8 \pm 0 . 0 0 3 }$ </td><td></td><td>5.1</td></tr><tr><td colspan="2">SCORE</td><td>0.123 ± 0.001</td><td>0.197 ± 0.001</td><td>2.321 ± 0.010</td><td>0.085 ± 0.001</td><td>0.104 ± 0.001</td><td> $\mathbf { - 0 . 5 2 9 \pm 0 . 0 1 8 }$ </td><td>2.7</td></tr><tr><td colspan="2">Univariate, T = 192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td>0.160 ± 0.002</td><td> $0 . 3 2 3 \pm 0 . 0 0 2$ </td><td> $5 . 3 5 6 \pm 0 . 0 3 1$ </td><td> $0 . 1 3 2 \pm 0 . 0 0 2$ </td><td> $0 . 1 4 5 \pm 0 . 0 0 2$ </td><td></td><td>7.5</td></tr><tr><td>Diag</td><td>NLL</td><td>0.157 ± 0.002</td><td> $\mathbf { 0 . 2 2 4 \pm 0 . 0 0 2 }$ </td><td> $3 . 8 4 0 \pm 0 . 0 2 2$ </td><td> $0 . 1 1 5 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 3 3 \pm 0 . 0 0 1 }$ </td><td> $+ 0 . 4 9 1 \pm 0 . 0 0 8$ </td><td>3.8</td></tr><tr><td>Diag</td><td>KS</td><td>0.159 ± 0.002</td><td> $\mathbf { 0 . 2 2 8 \pm 0 . 0 0 2 }$ </td><td> $3 . 8 5 9 \pm 0 . 0 2 9$ </td><td> $0 . 1 0 8 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 1 3 4 \pm 0 . 0 0 2 }$ </td><td> $+ 0 . 5 5 1 \pm 0 . 0 1 4$ </td><td>4.0</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.184 ± 0.001</td><td> $0 . 2 5 1 \pm 0 . 0 0 1$ </td><td> $4 . 1 4 7 \pm 0 . 0 1 2$ </td><td> $0 . 1 0 5 \pm 0 . 0 0 0$ </td><td> $0 . 1 5 1 \pm 0 . 0 0 1$ </td><td> $+ 0 . 2 0 9 \pm 0 . 0 0 6$ </td><td>5.9</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $0 . 1 6 1 \pm 0 . 0 0 1$ </td><td> $0 . 2 3 0 \pm 0 . 0 0 1$ </td><td> $3 . 8 4 0 \pm 0 . 0 1 7$ </td><td> $0 . 1 0 3 \pm 0 . 0 0 0$ </td><td> $0 . 1 3 4 \pm 0 . 0 0 1$ </td><td>+0.225 ± 0.009</td><td>3.7</td></tr><tr><td>Chol</td><td>NLL</td><td> $0 . 2 4 1 \pm 0 . 0 0 1$ </td><td> $0 . 2 9 7 \pm 0 . 0 0 1$ </td><td> $4 . 8 7 5 \pm 0 . 0 1 3$ </td><td> $0 . 1 2 3 \pm 0 . 0 0 1$ </td><td> $0 . 1 9 2 \pm 0 . 0 0 1$ </td><td>−0.428 ± 0.015</td><td>7.1</td></tr><tr><td>Chol</td><td>KS</td><td>0.161 ± 0.002</td><td> $0 . 2 2 9 \pm 0 . 0 0 2$ </td><td>3.813 ± 0.030</td><td>0.101 ± 0.001</td><td> $\mathbf { 0 . 1 3 3 \pm 0 . 0 0 2 }$ </td><td>(+2.392 ± 5.349) × 1028</td><td>3.7</td></tr><tr><td>SB</td><td>KS</td><td>0.160 ± 0.004</td><td> $0 . 2 4 0 \pm 0 . 0 0 7$ </td><td>4.124 ± 0.098</td><td>0.126 ± 0.004</td><td> $\mathbf { 0 . 1 3 3 \pm 0 . 0 0 3 }$ </td><td></td><td>5.6</td></tr><tr><td>Spectral space SCORE</td><td>CRPS+KS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Multivariate, T = 96</td><td>0.158 ± 0.002</td><td>0.227 ± 0.001</td><td>3.792 ± 0.020</td><td>0.101 ± 0.000</td><td>0.131 ± 0.001</td><td>−0.099 ± 0.387</td><td>3.2</td></tr><tr><td colspan="2">Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td>0.284 ± 0.001</td><td>0.345 ± 0.001</td><td> $1 2 . 6 4 5 \pm 0 . 0 2 3$ </td><td>0.395 ± 0.001</td><td> $0 . 2 2 6 \pm 0 . 0 0 1$ </td><td></td><td>5.5</td></tr><tr><td>Diag</td><td>NLL</td><td>0.416 ± 0.035</td><td>0.322 ± 0.015</td><td>11.411 ± 0.558</td><td>0.514 ± 0.051</td><td> $0 . 2 6 5 \pm 0 . 0 1 7$ </td><td>+0.732 ± 0.017</td><td>6.1</td></tr><tr><td>Diag</td><td>KS</td><td>0.278 ± 0.002</td><td>0.251 ± 0.001</td><td>9.078 ± 0.025</td><td>0.355 ± 0.001</td><td> $\mathbf { 0 . 1 9 4 \pm 0 . 0 0 1 }$ </td><td>+0.712 ± 0.031</td><td>2.4</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.361 ± 0.005</td><td>0.297 ± 0.003</td><td>10.520 ± 0.091</td><td>0.454 ± 0.007</td><td> $0 . 2 3 7 \pm 0 . 0 0 3$ </td><td>+0.578 ± 0.013</td><td>4.8</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.277 ± 0.001</td><td>0.250 ± 0.001</td><td>9.055 ± 0.021</td><td>0.354 ± 0.001</td><td> $\mathbf { 0 . 1 9 3 \pm 0 . 0 0 1 }$ </td><td>+0.642 ± 0.043</td><td>2.3</td></tr><tr><td>SB</td><td>KS</td><td>0.284 ± 0.006</td><td>0.309 ± 0.019</td><td> $1 1 . 4 1 3 \pm 0 . 6 5 8$ </td><td>0.388 ± 0.008</td><td>0.220 ± 0.007</td><td></td><td>4.5</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">SCORE</td><td>0.280 ± 0.002</td><td>0.250 ± 0.000</td><td>9.084 ± 0.024</td><td>0.356 ± 0.001</td><td>0.194 ± 0.001</td><td>+0.057 ± 0.031</td><td>1.9</td></tr><tr><td>Multivariate, T = 192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td>0.349 ± 0.010</td><td>0.386 ± 0.004</td><td>20.085 ± 0.258</td><td>0.483 ± 0.010</td><td> $0 . 2 7 1 \pm 0 . 0 0 5$ </td><td></td><td>5.3</td></tr><tr><td>Diag</td><td>NLL</td><td>0.509 ± 0.016</td><td>0.360 ± 0.006</td><td>17.923 ± 0.289</td><td>0.580 ± 0.018</td><td>0.299 ± 0.005</td><td>+0.840 ± 0.015</td><td>6.1</td></tr><tr><td>Diag</td><td>KS</td><td>0.344 ± 0.002</td><td>0.284 ± 0.001</td><td>14.526 ± 0.049</td><td>0.435 ± 0.001</td><td>0.231 ± 0.001</td><td>+0.843 ± 0.039</td><td>2.6</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.391 ± 0.002</td><td>0.314 ± 0.002</td><td>15.651 ± 0.085</td><td>0.495 ± 0.009</td><td>0.255 ± 0.003</td><td>+0.661 ± 0.010</td><td>4.3</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.343 ± 0.002</td><td>0.283 ± 0.001</td><td>14.523 ± 0.056</td><td>0.435 ± 0.002</td><td>0.231 ± 0.001</td><td>+0.730 ± 0.026</td><td>2.3</td></tr><tr><td></td><td>KS</td><td>0.352 ± 0.008</td><td>0.348 ± 0.040</td><td>18.315 ± 2.077</td><td>0.478 ± 0.020</td><td>0.264 ± 0.016</td><td></td><td>4.9</td></tr><tr><td>SB</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Spectral space SCORE</td><td>CRPS+KS</td><td>0.347 ± 0.002</td><td>0.284 ± 0.000</td><td>14.583 ± 0.034</td><td>0.438 ± 0.002</td><td>0.232 ± 0.001</td><td>+0.107 ± 0.018</td><td>2.0</td></tr></table>

Table 10: Test scores on ETTm1.
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td colspan="9">Univariate, T = 96</td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td>0.026 ± 0.000</td><td>0.122 ± 0.001</td><td>1.409 ± 0.004</td><td>0.015 ± 0.000</td><td>0.025 ± 0.000</td><td></td><td>5.1</td></tr><tr><td>Diag</td><td>NLL</td><td>0.026 ± 0.000</td><td>0.087 ± 0.000</td><td>1.048 ± 0.002</td><td>0.017 ± 0.001</td><td>0.025 ± 0.000</td><td>-0.390 ± 0.007</td><td>3.6</td></tr><tr><td>Diag</td><td>KS</td><td>0.026 ± 0.000</td><td>0.089 ± 0.000</td><td>1.060 ± 0.003</td><td>0.013 ± 0.000</td><td>0.025 ± 0.000</td><td>-0.196 ± 0.039</td><td>3.9</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.029 ± 0.001</td><td>0.093 ± 0.002</td><td>1.073 ± 0.015</td><td>0.012 ± 0.000</td><td>0.027 ± 0.001</td><td>−0.920 ± 0.007</td><td>4.8</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.026 ± 0.000</td><td>0.090 ± 0.000</td><td>1.042 ± 0.005</td><td>0.011 ± 0.000</td><td>0.025 ± 0.000</td><td>-0.663 ± 0.044</td><td>3.0</td></tr><tr><td>SB</td><td>KS</td><td>0.026 ± 0.000</td><td>0.121 ± 0.000</td><td>1.408 ± 0.004</td><td>0.015 ± 0.000</td><td>0.025 ± 0.000</td><td></td><td>5.1</td></tr><tr><td>Spectral space</td><td>CRPS+KS</td><td>0.026 ± 0.000</td><td>0.088 ± 0.000</td><td>1.024 ± 0.002</td><td>0.011 ± 0.000</td><td>0.025 ± 0.000</td><td>-1.587 ± 0.031</td><td>2.1</td></tr><tr><td colspan="9">SCORE</td></tr><tr><td colspan="9">Univariate, T = 192</td></tr><tr><td>Physical space</td><td>MSE</td><td>0.039 ± 0.000</td><td>0.151 ± 0.001</td><td>2.477 ± 0.011</td><td>0.023 ± 0.000</td><td>0.038 ± 0.000</td><td></td><td>5.7</td></tr><tr><td>Det Diag</td><td></td><td></td><td></td><td>1.848 ± 0.003</td><td>0.026 ± 0.000</td><td>0.037 ± 0.000</td><td>−0.122 ± 0.006</td><td>3.8</td></tr><tr><td></td><td>NLL</td><td>0.039 ± 0.000</td><td>0.109 ± 0.000</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diag</td><td>KS</td><td>0.039 ± 0.000</td><td>0.111 ± 0.000</td><td>1.843 ± 0.007</td><td>0.019 ± 0.000</td><td>0.037 ± 0.000</td><td>+0.159 ± 0.009</td><td>3.8</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.042 ± 0.001</td><td>0.115 ± 0.003</td><td>1.878 ± 0.053</td><td>0.017 ± 0.002</td><td>0.040 ± 0.002</td><td>-0.652 ± 0.027</td><td>4.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.039 ± 0.000</td><td>0.111 ± 0.001</td><td>1.819 ± 0.013</td><td>0.017 ± 0.000</td><td>0.037 ± 0.000</td><td>-0.419 ± 0.057</td><td>3.1</td></tr><tr><td>SB Spectral space</td><td>KS</td><td>0.039 ± 0.000</td><td>0.150 ± 0.001</td><td>2.456 ± 0.009</td><td>0.023 ± 0.000</td><td>0.038 ± 0.000</td><td></td><td>5.5</td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>0.039 ± 0.000</td><td>0.109 ± 0.000</td><td>1.793 ± 0.006</td><td>0.016 ± 0.000</td><td>0.036 ± 0.000</td><td>−1.458 ± 0.022</td><td>1.6</td></tr><tr><td colspan="9">Multivariate, T = 96</td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td>0.290 ± 0.001</td><td>0.343 ± 0.001</td><td>13.276 ± 0.018</td><td>0.408 ± 0.003</td><td>0.238 ± 0.001</td><td></td><td>6.4</td></tr><tr><td>Diag</td><td>NLL</td><td>0.281 ± 0.007</td><td>0.243 ± 0.004</td><td>9.490 ± 0.130</td><td>0.364 ± 0.007</td><td>0.206 ± 0.004</td><td>+0.664 ± 0.129</td><td>2.8</td></tr><tr><td>Diag</td><td>KS</td><td>0.289 ± 0.004</td><td>0.254 ± 0.002</td><td>9.577 ± 0.053</td><td>0.390 ± 0.004</td><td>0.208 ± 0.001</td><td>+0.756 ± 0.054</td><td>4.4</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.285 ± 0.009</td><td>0.245 ± 0.004</td><td>9.572 ± 0.131</td><td>0.372 ± 0.005</td><td>0.210 ± 0.004</td><td>+0.515 ± 0.096</td><td>3.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.287 ± 0.002</td><td>0.249 ± 0.001</td><td>9.465 ± 0.020</td><td>0.378 ± 0.002</td><td>0.204 ± 0.001</td><td>+0.805 ± 0.055</td><td>3.2</td></tr><tr><td>SB</td><td>KS</td><td>0.311 ± 0.013</td><td>0.260 ± 0.005</td><td>9.963 ± 0.231</td><td>0.410 ± 0.014</td><td>0.214 ± 0.005</td><td></td><td>5.9</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">SCORE</td></tr><tr><td>Multivariate, T = 192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.0</td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td>0.331 ± 0.004</td><td>0.369 ± 0.001</td><td>20.393 ± 0.106</td><td>0.477 ± 0.003</td><td>0.271 ± 0.002</td><td></td><td>6.2</td></tr><tr><td>Diag</td><td>NLL</td><td>0.335 ± 0.002</td><td>0.272 ± 0.002</td><td>14.923 ± 0.066</td><td>0.434 ± 0.002</td><td>0.241 ± 0.001</td><td>+0.964 ± 0.069</td><td>4.0</td></tr><tr><td>Diag</td><td>KS</td><td>0.330 ± 0.003</td><td>0.277 ± 0.001</td><td>14.676 ± 0.061</td><td>0.456 ± 0.003</td><td>0.235 ± 0.001</td><td>+0.872 ± 0.041</td><td>3.8</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.329 ± 0.009</td><td>0.264 ± 0.003</td><td>14.689 ± 0.199</td><td>0.433 ± 0.019</td><td>0.236 ± 0.005</td><td>+0.673 ± 0.124</td><td>2.8</td></tr><tr><td>LorD r=1</td><td>KS</td><td>0.330 ± 0.005</td><td>0.275 ± 0.001</td><td>14.641 ± 0.084</td><td>0.455 ± 0.005</td><td>0.234 ± 0.002</td><td>+0.874 ± 0.020</td><td>3.5</td></tr><tr><td>SB</td><td>KS</td><td>0.361 ± 0.008</td><td>0.284 ± 0.003</td><td>15.169 ± 0.161</td><td>0.516 ± 0.038</td><td>0.240 ± 0.004</td><td></td><td>5.9</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>0.325 ± 0.003</td><td>0.259 ± 0.001</td><td>14.503 ± 0.062</td><td>0.425 ± 0.006</td><td>0.230 ± 0.001</td><td>−0.182 ± 0.038</td><td>1.7</td></tr></table>

Table 11: Test scores on ETTm2.
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅ Rank↓</td></tr><tr><td colspan="2">Univariate, T = 96</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Physical space</td><td></td><td></td><td></td><td></td><td> $0 . 0 6 1 \pm 0 . 0 0 0$ </td><td></td><td>5.4</td></tr><tr><td>Det</td><td>MSE</td><td> $0 . 0 6 5 \pm 0 . 0 0 0$ </td><td> $0 . 1 8 7 \pm 0 . 0 0 1$ </td><td> $2 . 1 9 4 \pm 0 . 0 0 8$ </td><td> $0 . 0 5 3 \pm 0 . 0 0 1$ </td><td></td><td> $- 0 . 0 8 1 \pm 0 . 0 2 6$ </td><td>3.6</td></tr><tr><td>Diag</td><td>NLL</td><td> $\mathbf { 0 . 0 6 4 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 3 3 \pm 0 . 0 0 2 }$ </td><td> $1 . 6 7 6 \pm 0 . 0 2 2$ </td><td> $0 . 0 7 1 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 0 5 8 \pm 0 . 0 0 1 }$ </td><td></td><td></td></tr><tr><td>Diag</td><td>KS</td><td> $\mathbf { 0 . 0 6 4 \pm 0 . 0 0 0 }$ </td><td> $0 . 1 3 8 \pm 0 . 0 0 0$ </td><td> $1 . 6 6 4 \pm 0 . 0 0 3$ </td><td> $0 . 0 5 9 \pm 0 . 0 0 0$ </td><td> $0 . 0 5 8 \pm 0 . 0 0 0$ </td><td> $+ 0 . 2 4 0 \pm 0 . 0 1 8$ </td><td>3.9</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 0 6 8 \pm 0 . 0 0 1$ </td><td> $0 . 1 4 7 \pm 0 . 0 0 2$ </td><td> $1 . 7 5 7 \pm 0 . 0 1 6$ </td><td> $0 . 0 5 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 6 8 \pm 0 . 0 0 2$ </td><td>−0.627 ± 0.033</td><td>5.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $\mathbf { 0 . 0 6 4 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 3 7 \pm 0 . 0 0 1$ </td><td> $\mathbf { 1 . 6 1 0 \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 0 5 1 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 5 7 \pm 0 . 0 0 0 }$ </td><td> $+ 0 . 1 1 9 \pm 0 . 1 0 5$ </td><td>2.9</td></tr><tr><td>SB</td><td>KS</td><td> $\mathbf { 0 . 0 6 5 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 8 4 \pm 0 . 0 0 1$ </td><td> $2 . 1 5 8 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 0 5 2 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 6 0 \pm 0 . 0 0 1$ </td><td></td><td>4.6</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>0.063 ± 0.001</td><td>0.134 ± 0.001</td><td>1.589 ± 0.015</td><td>0.050 ± 0.001</td><td> $\mathbf { 0 . 0 5 6 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { - 1 . 5 2 6 \pm 0 . 0 6 6 }$ </td><td>1.8</td></tr><tr><td colspan="2">Univariate, T = 192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td> $0 . 2 3 2 \pm 0 . 0 0 2$ </td><td> $3 . 8 6 6 \pm 0 . 0 1 5$ </td><td> $0 . 0 7 6 \pm 0 . 0 0 1$ </td><td> $0 . 0 8 7 \pm 0 . 0 0 1$ </td><td></td><td>5.4</td></tr><tr><td>Det</td><td>MSE</td><td> $0 . 0 9 5 \pm 0 . 0 0 1$ </td><td></td><td></td><td></td><td></td><td> $+ 0 . 2 3 2 \pm 0 . 0 3 5$ </td><td></td></tr><tr><td>Diag</td><td>NLL</td><td> $\mathbf { 0 . 0 9 4 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 6 9 \pm 0 . 0 0 3 }$ </td><td> $2 . 9 5 1 \pm 0 . 0 4 7$ </td><td> $0 . 0 9 9 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 0 8 4 \pm 0 . 0 0 2 }$ </td><td></td><td>4.2</td></tr><tr><td>Diag</td><td>KS</td><td> $\mathbf { 0 . 0 9 3 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 7 1 \pm 0 . 0 0 2$ </td><td> $2 . 8 9 9 \pm 0 . 0 2 0$ </td><td> $0 . 0 8 0 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 8 2 \pm 0 . 0 0 1 }$ </td><td>+0.460 ± 0.036</td><td>4.0</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 0 9 7 \pm 0 . 0 0 3$ </td><td> $0 . 1 7 5 \pm 0 . 0 0 2$ </td><td> $2 . 9 5 7 \pm 0 . 0 3 7$ </td><td> $0 . 0 7 8 \pm 0 . 0 0 2$ </td><td> $0 . 0 8 8 \pm 0 . 0 0 2$ </td><td> $- 0 . 1 2 7 \pm 0 . 0 4 7$ </td><td>4.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $\mathbf { 0 . 0 9 3 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 7 0 \pm 0 . 0 0 2 }$ </td><td> $\begin{array} { r } { \mathbf { 2 . 8 3 9 \pm 0 . 0 2 4 } } \\ { 3 . 8 2 2 \pm 0 . 0 3 3 } \end{array}$ </td><td>0.070 ± 0.002</td><td> $\mathbf { 0 . 0 8 1 \pm 0 . 0 0 1 }$ </td><td> $+ 0 . 2 2 6 \pm 0 . 0 9 9$ </td><td>2.6</td></tr><tr><td>SB</td><td>KS</td><td>0.094 ± 0.001</td><td> $0 . 2 3 0 \pm 0 . 0 0 3$ </td><td></td><td>0.075 ± 0.001</td><td> $0 . 0 8 6 \pm 0 . 0 0 1$ </td><td></td><td>5.3</td></tr><tr><td>Spectral space SCORE</td><td>CRPS+KS</td><td></td><td> $\mathbf { 0 . 1 6 7 \pm 0 . 0 0 1 }$ </td><td>2.812 ± 0.018</td><td>0.067 ± 0.001</td><td></td><td>−1.240 ± 0.097</td><td></td></tr><tr><td colspan="2"></td><td>0.093 ± 0.000</td><td></td><td></td><td></td><td> $\mathbf { 0 . 0 8 1 \pm 0 . 0 0 1 }$ </td><td></td><td>1.9</td></tr><tr><td>Multivariate, T = 96</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td>0.168 ± 0.003</td><td> $0 . 2 5 7 \pm 0 . 0 0 3$ </td><td> $9 . 5 7 1 \pm 0 . 0 9 2$ </td><td> $0 . 2 3 3 \pm 0 . 0 0 3$ </td><td> $0 . 1 4 2 \pm 0 . 0 0 2$ </td><td></td><td>7.0</td></tr><tr><td>Diag</td><td>NLL</td><td> $\mathbf { 0 . 1 6 0 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 9 0 \pm 0 . 0 0 1$ </td><td> $6 . 9 1 5 \pm 0 . 0 2 9$ </td><td> $0 . 2 2 3 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 1 2 6 \pm 0 . 0 0 1 }$ </td><td>+0.576 ± 0.043</td><td>4.0</td></tr><tr><td>Diag</td><td>KS</td><td> $\mathbf { 0 . 1 5 9 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 1 8 4 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 6 . 7 6 7 \pm 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 2 1 2 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 2 4 \pm 0 . 0 0 1 }$ </td><td> $+ 4 3 . 0 0 3 \pm 0 . 0 1 4$ </td><td>2.6</td></tr><tr><td>LorD r=1</td><td>NLL</td><td>0.160 ± 0.001</td><td> $0 . 1 8 9 \pm 0 . 0 0 1$ </td><td> $6 . 8 3 9 \pm 0 . 0 1 3$ </td><td> $0 . 2 1 8 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 1 2 5 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { + 0 . 4 7 2 \pm 0 . 0 4 2 }$ </td><td>3.3</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $\mathbf { 0 . 1 6 0 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 1 8 4 \pm 0 . 0 0 1 }$ </td><td> ${ \bf 6 . 7 8 4 \pm 0 . 0 3 7 }$ </td><td> $\mathbf { 0 . 2 1 2 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 1 2 5 \pm 0 . 0 0 1 }$ </td><td>+42.871 ± 0.123</td><td>2.7</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 1 6 3 \pm 0 . 0 0 1$ </td><td> $0 . 1 9 1 \pm 0 . 0 0 2$ </td><td> $7 . 4 1 7 \pm 0 . 0 2 8$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 1$ </td><td> $0 . 1 2 8 \pm 0 . 0 0 1$ </td><td></td><td>5.3</td></tr><tr><td>Spectral space SCORE</td><td>CRPS+KS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td> $\mathbf { 0 . 1 6 1 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 8 3 \pm 0 . 0 0 1 }$ </td><td>6.800 ± 0.030</td><td>0.212 ± 0.001</td><td> $\mathbf { 0 . 1 2 5 \pm 0 . 0 0 1 }$ </td><td>(+2.349 ± 4.820) × 107</td><td>3.0</td></tr><tr><td>Multivariate, T = 192</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $0 . 2 2 3 \pm 0 . 0 0 4$ </td><td> $0 . 2 9 5 \pm 0 . 0 0 3$ </td><td> $1 5 . 5 1 5 \pm 0 . 1 4 1$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 4$ </td><td> $0 . 1 8 0 \pm 0 . 0 0 3$ </td><td></td><td>6.6</td></tr><tr><td>Diag</td><td>NLL</td><td> $0 . 2 2 0 \pm 0 . 0 0 1$ </td><td> $0 . 2 2 3 \pm 0 . 0 0 1$ </td><td> $1 1 . 4 5 0 \pm 0 . 0 8 0$ </td><td> $0 . 2 9 9 \pm 0 . 0 0 7$ </td><td> $0 . 1 6 2 \pm 0 . 0 0 2$ </td><td>+1.157 ± 0.230</td><td>4.8</td></tr><tr><td>Diag</td><td>KS</td><td> $0 . 2 1 4 \pm 0 . 0 0 2$ </td><td> $0 . 2 1 4 \pm 0 . 0 0 1$ </td><td> $1 1 . 1 2 9 \pm 0 . 0 7 8$ </td><td> $0 . 2 7 6 \pm 0 . 0 0 2$ </td><td> $0 . 1 5 7 \pm 0 . 0 0 2$ </td><td> $+ 6 7 . 3 6 6 \pm 0 . 0 2 8$ </td><td>2.9</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 2 1 6 \pm 0 . 0 0 2$ </td><td> $0 . 2 2 2 \pm 0 . 0 0 2$ </td><td> $1 1 . 3 4 7 \pm 0 . 1 0 9$ </td><td> $0 . 2 9 8 \pm 0 . 0 0 8$ </td><td> $0 . 1 6 1 \pm 0 . 0 0 2$ </td><td> $\mathbf { + 0 . 7 1 7 \pm 0 . 0 3 5 }$ </td><td>4.1</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $\mathbf { 0 . 2 1 0 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 2 1 2 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 1 0 . 9 6 9 \pm 0 . 0 1 3 }$ </td><td> $\mathbf { 0 . 2 7 2 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 5 3 \pm 0 . 0 0 0 }$ </td><td> $+ 6 7 . 2 7 0 \pm 0 . 0 5 1$ </td><td>1.5</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 2 2 1 \pm 0 . 0 0 3$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 1$ </td><td> $1 2 . 0 2 1 \pm 0 . 0 3 7$ </td><td> $0 . 2 8 9 \pm 0 . 0 0 3$ </td><td> $0 . 1 6 1 \pm 0 . 0 0 1$ </td><td></td><td>5.2</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>0.212 ± 0.001</td><td>0.212 ± 0.001</td><td>11.037 ± 0.038</td><td>0.273 ± 0.001</td><td>0.155 ± 0.001</td><td>(+8.006 ± 13.424) × 10⁶</td><td>2.8</td></tr></table>

Table 12: Test scores on National illness (multivariate, 7 series).
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td>T = 24</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $2 . 0 7 8 \pm 0 . 1 2 8$ </td><td> $0 . 9 3 2 \pm 0 . 0 5 1$ </td><td> $1 6 . 5 2 3 \pm 0 . 8 1 7$ </td><td> $2 . 4 3 0 \pm 0 . 2 3 6$ </td><td> $0 . 6 6 8 \pm 0 . 0 3 4$ </td><td></td><td>6.3</td></tr><tr><td>Diag</td><td>NLL</td><td> $\mathbf { 1 . 7 5 2 \pm 0 . 0 3 9 }$ </td><td> $\mathbf { 0 . 6 3 1 \pm 0 . 0 2 5 }$ </td><td> ${ \bf 1 1 . 3 8 4 \pm 0 . 3 3 8 }$ </td><td> $\mathbf { 1 . 7 8 9 \pm 0 . 0 4 0 }$ </td><td> $\mathbf { 0 . 4 4 5 \pm 0 . 0 2 1 }$ </td><td> $+ 3 . 5 2 6 \pm 0 . 6 0 4$ </td><td>1.8</td></tr><tr><td>Diag</td><td>KS</td><td> $2 . 1 7 7 \pm 0 . 1 5 5$ </td><td> $0 . 7 4 8 \pm 0 . 0 5 8$ </td><td> $1 2 . 8 4 6 \pm 0 . 7 6 2$ </td><td> $2 . 1 6 1 \pm 0 . 1 4 5$ </td><td> $\mathbf { 0 . 4 6 1 \pm 0 . 0 2 1 }$ </td><td> $\mathbf { + 2 . 4 0 9 \pm 0 . 2 9 0 }$ </td><td>3.8</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $2 . 1 3 6 \pm 0 . 1 0 3$ </td><td> $0 . 7 3 5 \pm 0 . 0 1 5$ </td><td> $1 2 . 8 2 8 \pm 0 . 1 8 2$ </td><td> $2 . 2 0 0 \pm 0 . 0 6 8$ </td><td> $\mathbf { 0 . 4 6 1 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { + 2 . 5 6 8 \pm 0 . 5 5 0 }$ </td><td>4.0</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $2 . 0 6 6 \pm 0 . 0 4 0$ </td><td> $0 . 7 2 1 \pm 0 . 0 1 7$ </td><td> $1 2 . 5 6 3 \pm 0 . 2 4 5$ </td><td> $2 . 0 5 5 \pm 0 . 0 7 8$ </td><td> $\mathbf { 0 . 4 6 6 \pm 0 . 0 1 3 }$ </td><td> $\mathbf { + 2 . 4 9 8 \pm 0 . 2 4 8 }$ </td><td>3.7</td></tr><tr><td>SB</td><td>KS</td><td> $1 . 9 7 4 \pm 0 . 0 9 2$ </td><td> $0 . 7 1 1 \pm 0 . 0 3 7$ </td><td> $1 2 . 7 8 2 \pm 0 . 4 9 5$ </td><td> $2 . 0 2 4 \pm 0 . 0 8 9$ </td><td> $0 . 5 0 7 \pm 0 . 0 2 3$ </td><td></td><td>4.3</td></tr><tr><td>Spectral space SCORE</td><td></td><td>1.962 ± 0.081</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CRPS+KS</td><td></td><td>0.696 ± 0.031</td><td> $1 2 . 3 6 9 \pm 0 . 3 4 0$ </td><td>1.964 ± 0.093</td><td>0.475 ± 0.022</td><td>+2.948 ± 2.796</td><td>3.8</td></tr><tr><td>T = 36</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $\mathbf { 2 . 0 1 1 \pm 0 . 1 9 3 }$ </td><td>0.948 ± 0.053</td><td> $2 0 . 7 2 8 \pm 1 . 2 9 1$ </td><td> $2 . 6 9 8 \pm 0 . 2 7 6$ </td><td> $0 . 7 2 3 \pm 0 . 0 4 1$ </td><td></td><td>5.8</td></tr><tr><td>Diag</td><td>NLL</td><td> $\mathbf { 1 . 7 0 1 \pm 0 . 1 3 5 }$ </td><td> $\mathbf { 0 . 6 3 3 \pm 0 . 0 3 1 }$ </td><td> $\mathbf { 1 4 . 2 3 2 \pm 0 . 5 6 9 }$ </td><td> $\mathbf { 2 . 0 7 1 \pm 0 . 2 5 1 }$ </td><td> $\mathbf { 0 . 4 7 7 \pm 0 . 0 1 5 }$ </td><td> $+ 3 . 2 1 4 \pm 0 . 9 6 5$ </td><td>2.7</td></tr><tr><td>Diag</td><td>KS</td><td> $2 . 1 3 6 \pm 0 . 1 2 0$ </td><td> $0 . 7 6 6 \pm 0 . 0 3 5$ </td><td> $1 6 . 4 5 9 \pm 0 . 5 9 2$ </td><td> $2 . 4 6 8 \pm 0 . 1 8 3$ </td><td> $0 . 5 1 0 \pm 0 . 0 1 7$ </td><td>+2.392 ± 0.218</td><td>4.7</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $\mathbf { 1 . 8 9 7 \pm 0 . 1 4 3 }$ </td><td> $0 . 7 0 8 \pm 0 . 0 1 5$ </td><td> $1 5 . 2 6 7 \pm 0 . 3 4 5$ </td><td> $2 . 3 9 2 \pm 0 . 0 9 3$ </td><td> $\mathbf { 0 . 4 6 9 \pm 0 . 0 0 6 }$ </td><td>+2.708 ± 0.609</td><td>3.4</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $1 . 9 5 4 \pm 0 . 0 3 5$ </td><td> $0 . 7 1 8 \pm 0 . 0 2 1$ </td><td> $1 5 . 6 0 3 \pm 0 . 3 2 4$ </td><td> $2 . 2 9 6 \pm 0 . 0 6 7$ </td><td> $0 . 5 0 2 \pm 0 . 0 1 9$ </td><td>+2.344 ± 0.331</td><td>4.0</td></tr><tr><td>SB</td><td>KS</td><td> $\mathbf { 1 . 9 8 5 \pm 0 . 3 3 2 }$ </td><td> $\mathbf { 0 . 8 3 9 \pm 0 . 2 3 5 }$ </td><td> ${ \bf 1 8 . 2 7 6 \pm 4 . 3 7 6 }$ </td><td> $\mathbf { 2 . 4 1 0 \pm 0 . 4 4 0 }$ </td><td> $0 . 6 3 6 \pm 0 . 1 5 1$ </td><td></td><td>4.1</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td>1.821 ± 0.062</td><td>0.675 ± 0.033</td><td> $1 5 . 2 8 5 \pm 0 . 4 1 9$ </td><td> $\mathbf { 2 . 1 7 0 \pm 0 . 0 5 7 }$ </td><td>0.515 ± 0.019</td><td>+1.072 ± 0.188</td><td>2.8</td></tr></table>

Table 13: Test scores on Weather (multivariate, 21 series).
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td> $T = 9 6$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $\mathbf { 0 . 1 4 7 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 9 6 \pm 0 . 0 0 1$ </td><td> $1 5 . 9 9 0 \pm 0 . 0 3 6$ </td><td> $0 . 2 2 7 \pm 0 . 0 0 1$ </td><td> $0 . 1 3 0 \pm 0 . 0 0 0$ </td><td></td><td>5.3</td></tr><tr><td>Diag</td><td>NLL</td><td> $0 . 1 8 9 \pm 0 . 0 1 4$ </td><td> $0 . 1 8 2 \pm 0 . 0 0 3$ </td><td> $1 3 . 4 2 8 \pm 0 . 2 7 6$ </td><td> $0 . 5 0 9 \pm 0 . 1 0 6$ </td><td> $0 . 1 4 8 \pm 0 . 0 0 4$ </td><td> $\mathbf { + 3 . 7 3 1 \pm 1 . 9 0 1 }$ </td><td>5.2</td></tr><tr><td>Diag</td><td>KS</td><td> $\mathbf { 0 . 1 4 6 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 5 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 1 1 . 6 8 6 \pm 0 . 0 4 0 }$ </td><td> $0 . 2 2 1 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 1 2 1 \pm 0 . 0 0 1 }$ </td><td> $+ 2 4 3 . 6 1 3 \pm 9 . 6 7 8$ </td><td>3.0</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 1 8 2 \pm 0 . 0 0 8$ </td><td> $0 . 1 8 3 \pm 0 . 0 0 5$ </td><td> $1 3 . 5 2 4 \pm 0 . 3 6 8$ </td><td> $0 . 5 2 7 \pm 0 . 1 0 0$ </td><td> $0 . 1 5 2 \pm 0 . 0 0 7$ </td><td> $\mathbf { + 3 . 6 0 1 \pm 1 . 6 8 7 }$ </td><td>5.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $\mathbf { 0 . 1 4 5 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 5 5 \pm 0 . 0 0 0$ </td><td> $\mathbf { 1 1 . 7 0 1 \pm 0 . 0 2 8 }$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 2 1 \pm 0 . 0 0 1 }$ </td><td> $+ 2 4 2 . 5 7 7 \pm 1 0 . 7 1 7$ </td><td>2.8</td></tr><tr><td>SB</td><td>KS</td><td> $\mathbf { 0 . 1 7 9 \pm 0 . 0 5 0 }$ </td><td> $0 . 1 5 6 \pm 0 . 0 0 5$ </td><td> $\mathbf { 1 2 . 2 3 9 \pm 0 . 8 1 1 }$ </td><td> $\mathbf { 0 . 4 2 0 \pm 0 . 2 5 9 }$ </td><td> $\mathbf { 0 . 1 2 3 \pm 0 . 0 0 2 }$ </td><td></td><td>3.2</td></tr><tr><td>Spectral space SCORE</td><td>CRPS+KS</td><td> $\mathbf { 0 . 1 4 6 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 1 3 8 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 1 1 . 6 8 1 \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 2 0 7 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 2 1 \pm 0 . 0 0 0 }$ </td><td> $+ 3 1 1 . 0 8 6 \pm 2 0 . 2 5 9$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.6</td></tr><tr><td> $T = 1 9 2$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $0 . 1 9 2 \pm 0 . 0 0 1$ </td><td> $0 . 2 4 0 \pm 0 . 0 0 2$ </td><td> $2 6 . 3 6 7 \pm 0 . 0 7 4$ </td><td> $0 . 2 8 5 \pm 0 . 0 0 1$ </td><td> $0 . 1 6 8 \pm 0 . 0 0 1$ </td><td></td><td>5.4</td></tr><tr><td>Diag</td><td>NLL</td><td> $0 . 2 3 8 \pm 0 . 0 3 0$ </td><td> $0 . 2 1 6 \pm 0 . 0 0 4$ </td><td> $2 1 . 4 3 5 \pm 0 . 8 2 8$ </td><td> $0 . 5 3 8 \pm 0 . 2 0 7$ </td><td> $0 . 1 7 8 \pm 0 . 0 0 6$ </td><td> $\mathbf { + 4 . 3 1 8 \pm 0 . 4 3 4 }$ </td><td>5.2</td></tr><tr><td>Diag</td><td>KS</td><td> $0 . 1 9 2 \pm 0 . 0 0 0$ </td><td> $0 . 1 9 3 \pm 0 . 0 0 1$ </td><td> ${ \bf 1 9 . 2 3 9 \pm 0 . 0 3 9 }$ </td><td> $0 . 2 7 4 \pm 0 . 0 0 2$ </td><td> $0 . 1 5 5 \pm 0 . 0 0 0$ </td><td> $+ 2 9 0 . 3 0 2 \pm 1 1 . 7 4 6$ </td><td>3.3</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 2 2 3 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 4 \pm 0 . 0 0 1$ </td><td> $2 1 . 2 3 9 \pm 0 . 2 6 4$ </td><td> $0 . 5 2 4 \pm 0 . 0 7 8$ </td><td> $0 . 1 7 9 \pm 0 . 0 0 3$ </td><td> $\mathbf { + 3 . 3 4 9 \pm 0 . 7 5 5 }$ </td><td>5.2</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $0 . 1 9 1 \pm 0 . 0 0 1$ </td><td> $0 . 1 9 0 \pm 0 . 0 0 1$ </td><td> ${ \bf 1 9 . 1 8 7 \pm 0 . 0 3 6 }$ </td><td> $0 . 2 7 2 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 1 5 4 \pm 0 . 0 0 0 }$ </td><td> $+ 3 0 0 . 5 8 4 \pm 1 5 . 1 0 8$ </td><td>2.8</td></tr><tr><td>SB</td><td>KS</td><td> $\mathbf { 0 . 2 0 4 \pm 0 . 0 1 8 }$ </td><td> $0 . 1 8 8 \pm 0 . 0 0 2$ </td><td> ${ \bf 1 9 . 4 3 9 \pm 0 . 2 4 9 }$ </td><td> $\mathbf { 0 . 3 6 9 \pm 0 . 1 6 9 }$ </td><td> $\mathbf { 0 . 1 5 5 \pm 0 . 0 0 1 }$ </td><td></td><td>3.1</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td> $\mathbf { 0 . 1 8 9 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 1 7 1 \pm 0 . 0 0 0 }$ </td><td> ${ \bf 1 9 . 2 3 7 \pm 0 . 0 3 8 }$ </td><td> $\mathbf { 0 . 2 5 7 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 5 4 \pm 0 . 0 0 0 }$ </td><td> $+ 3 6 6 . 6 8 1 \pm 3 2 . 3 7 1$ </td><td>2.2</td></tr></table>

Table 14: Test scores on Electricity (multivariate, 321 series).
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td>T = 96</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td>MSE</td><td> $0 . 1 3 0 \pm 0 . 0 0 0$ </td><td> $0 . 2 2 4 \pm 0 . 0 0 0$ </td><td> $6 1 . 6 0 9 \pm 0 . 0 6 4$ </td><td> $0 . 1 9 6 \pm 0 . 0 0 0$ </td><td> $0 . 1 1 9 \pm 0 . 0 0 0$ </td><td></td><td>4.7</td></tr><tr><td>Det</td><td>NLL</td><td></td><td></td><td> $4 5 . 9 9 5 \pm 1 . 0 1 5$ </td><td></td><td> $0 . 1 2 0 \pm 0 . 0 0 5$ </td><td> $\mathbf { + 0 . 5 8 7 \pm 0 . 0 3 9 }$ </td><td>4.1</td></tr><tr><td>Diag</td><td></td><td> $0 . 1 3 3 \pm 0 . 0 0 1$ </td><td> $0 . 1 7 4 \pm 0 . 0 0 2$ </td><td></td><td> $0 . 2 4 5 \pm 0 . 0 3 5$ </td><td></td><td></td><td></td></tr><tr><td>Diag</td><td>KS</td><td> $0 . 1 3 0 \pm 0 . 0 0 1$ </td><td> $0 . 1 8 4 \pm 0 . 0 0 4$ </td><td> $\mathbf { 4 4 . 0 0 0 \pm 0 . 1 2 7 }$ </td><td> $0 . 2 0 3 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 1 1 2 \pm 0 . 0 0 1 }$ </td><td> $+ 2 2 8 . 3 8 8 \pm 1 5 6 . 7 8 5$ </td><td>3.3</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 3 6 3 \pm 0 . 0 7 0$ </td><td> $0 . 6 0 9 \pm 0 . 0 8 3$ </td><td> $1 3 8 . 6 2 3 \pm 1 7 . 5 5 1$ </td><td> $2 . 1 3 5 \pm 0 . 3 9 7$ </td><td> $0 . 3 8 6 \pm 0 . 0 1 9$ </td><td> $\mathbf { + 0 . 5 6 8 \pm 0 . 0 2 1 }$ </td><td>6.1</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $0 . 1 3 0 \pm 0 . 0 0 1$ </td><td> $0 . 1 8 0 \pm 0 . 0 0 3$ </td><td> $4 4 . 0 4 9 \pm 0 . 0 9 3$ </td><td> $0 . 2 0 5 \pm 0 . 0 0 9$ </td><td> $0 . 1 1 2 \pm 0 . 0 0 0$ </td><td> $+ 2 4 8 . 5 8 6 \pm 1 6 8 . 7 2 4$ </td><td>3.4</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 1 3 3 \pm 0 . 0 0 1$ </td><td> $0 . 1 7 9 \pm 0 . 0 0 9$ </td><td> ${ \bf 4 6 . 1 2 6 \pm 3 . 8 4 5 }$ </td><td> $0 . 1 9 2 \pm 0 . 0 0 5$ </td><td> $0 . 1 1 5 \pm 0 . 0 0 3$ </td><td></td><td>3.7</td></tr><tr><td>Spectral space SCORE</td><td></td><td></td><td> $\mathbf { 0 . 1 6 2 \pm 0 . 0 0 0 }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CRPS+KS</td><td> $\mathbf { 0 . 1 2 8 \pm 0 . 0 0 0 }$ </td><td></td><td> $\mathbf { 4 3 . 8 2 7 \pm 0 . 0 5 8 }$ </td><td> $\mathbf { 0 . 1 7 8 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 1 1 1 \pm 0 . 0 0 0 }$ </td><td> $( + 2 . 6 0 9 \pm 3 . 8 3 1 ) \times 1 0 ^ { 4 }$ </td><td>1.9</td></tr><tr><td> $T = 1 9 2$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $0 . 1 4 9 \pm 0 . 0 0 1$ </td><td> $0 . 2 4 1 \pm 0 . 0 0 0$ </td><td> $9 3 . 7 3 4 \pm 0 . 1 8 6$ </td><td> $0 . 2 2 1 \pm 0 . 0 0 1$ </td><td> $0 . 1 3 6 \pm 0 . 0 0 1$ </td><td></td><td>5.1</td></tr><tr><td>Diag</td><td>NLL</td><td> $0 . 1 5 0 \pm 0 . 0 0 0$ </td><td> $0 . 1 8 6 \pm 0 . 0 0 1$ </td><td> $6 9 . 0 4 9 \pm 0 . 7 1 5$ </td><td> $0 . 2 6 5 \pm 0 . 0 1 6$ </td><td> $0 . 1 3 3 \pm 0 . 0 0 3$ </td><td> $\mathbf { + 0 . 9 8 9 \pm 0 . 0 7 1 }$ </td><td>4.2</td></tr><tr><td>Diag</td><td>KS</td><td> $0 . 1 4 8 \pm 0 . 0 0 0$ </td><td> $0 . 1 9 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 6 6 . 7 7 4 \pm 0 . 0 9 0 }$ </td><td> $0 . 2 2 4 \pm 0 . 0 0 5$ </td><td> $0 . 1 2 6 \pm 0 . 0 0 0$ </td><td> $+ 1 2 1 . 4 8 1 \pm 2 9 . 3 0 6$ </td><td>3.4</td></tr><tr><td> $\mathrm { L o r D } r { = } 1$ </td><td>NLL</td><td> $0 . 3 6 7 \pm 0 . 0 2 3$ </td><td> $0 . 6 2 1 \pm 0 . 0 1 6$ </td><td> $1 9 7 . 5 1 2 \pm 4 . 8 8 0$ </td><td> $1 . 9 5 2 \pm 0 . 2 3 6$ </td><td> $0 . 3 9 1 \pm 0 . 0 0 3$ </td><td> $\mathbf { + 0 . 9 7 2 \pm 0 . 0 8 6 }$ </td><td>6.1</td></tr><tr><td> $\mathrm { L o r D } r { = } 1$ </td><td>KS</td><td> $0 . 1 4 8 \pm 0 . 0 0 1$ </td><td> $0 . 1 9 1 \pm 0 . 0 0 1$ </td><td> $\mathbf { 6 6 . 8 3 1 \pm 0 . 1 9 3 }$ </td><td> $0 . 2 2 4 \pm 0 . 0 0 4$ </td><td> $0 . 1 2 7 \pm 0 . 0 0 1$ </td><td> $+ 6 8 . 6 0 2 \pm 1 5 . 8 9 7$ </td><td>3.4</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 1 5 0 \pm 0 . 0 0 1$ </td><td> $0 . 1 8 7 \pm 0 . 0 0 1$ </td><td> $6 7 . 2 0 5 \pm 0 . 1 9 3$ </td><td> $0 . 2 1 2 \pm 0 . 0 0 1$ </td><td> $0 . 1 2 7 \pm 0 . 0 0 1$ </td><td></td><td>3.1</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td> $\mathbf { 0 . 1 4 5 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 7 4 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 6 6 . 6 2 6 \pm 0 . 1 1 1 }$ </td><td> $\mathbf { 0 . 2 0 0 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 2 5 \pm 0 . 0 0 0 }$ </td><td> $( + 4 . 3 3 9 \pm 6 . 2 6 8 ) \times 1 0 ^ { 4 }$ </td><td>1.8</td></tr></table>

Table 15: Test scores on Traffic (multivariate, 862 series).
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td> $T = 9 6$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $0 . 4 0 0 \pm 0 . 0 0 1$ </td><td> $0 . 2 7 8 \pm 0 . 0 0 1$ </td><td> $1 7 9 . 0 1 3 \pm 0 . 1 3 6$ </td><td> $0 . 6 6 7 \pm 0 . 0 0 1$ </td><td> $0 . 3 2 2 \pm 0 . 0 0 0$ </td><td></td><td>4.9</td></tr><tr><td>Diag</td><td>NLL</td><td> $0 . 4 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 2 1 9 \pm 0 . 0 0 2$ </td><td> $1 2 8 . 5 2 1 \pm 0 . 3 2 6$ </td><td> $0 . 6 6 7 \pm 0 . 0 0 6$ </td><td> $0 . 2 7 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { + 0 . 3 2 1 \pm 0 . 0 1 2 }$ </td><td>3.6</td></tr><tr><td>Diag</td><td>KS</td><td> $0 . 4 0 1 \pm 0 . 0 0 1$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 1$ </td><td> $\mathbf { 1 2 7 . 6 1 1 \pm 0 . 2 2 1 }$ </td><td> $0 . 6 9 8 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 2 7 0 \pm 0 . 0 0 1 }$ </td><td> $+ 0 . 8 3 8 \pm 0 . 0 0 1$ </td><td>3.7</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 4 2 9 \pm 0 . 0 1 2$ </td><td> $0 . 3 5 9 \pm 0 . 0 7 0$ </td><td> $1 7 9 . 8 7 8 \pm 3 2 . 0 5 6$ </td><td> $2 . 2 9 2 \pm 1 . 2 9 2$ </td><td> $0 . 3 7 7 \pm 0 . 0 3 6$ </td><td> $\mathbf { + 0 . 2 9 0 \pm 0 . 0 2 9 }$ </td><td>6.0</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $0 . 4 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 2 5 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 1 2 7 . 6 7 6 \pm 0 . 2 3 4 }$ </td><td> $0 . 6 9 8 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 2 7 0 \pm 0 . 0 0 1 }$ </td><td> $+ 0 . 7 9 7 \pm 0 . 0 0 3$ </td><td>3.5</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 3 9 9 \pm 0 . 0 0 5$ </td><td> $0 . 2 3 6 \pm 0 . 0 1 2$ </td><td> $\mathbf { 1 2 7 . 2 1 4 \pm 0 . 4 0 4 }$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 4$ </td><td> $\mathbf { 0 . 2 6 9 \pm 0 . 0 0 1 }$ </td><td></td><td>3.3</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td> $\mathbf { 0 . 3 8 1 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 1 8 7 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 1 2 7 . 5 4 0 \pm 0 . 1 8 6 }$ </td><td> $\mathbf { 0 . 6 1 4 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 2 7 1 \pm 0 . 0 0 1 }$ </td><td> $+ 1 . 3 3 2 \pm 0 . 0 8 9$ </td><td>2.2</td></tr><tr><td> $T = 1 9 2$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Physical space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Det</td><td>MSE</td><td> $0 . 4 1 5 \pm 0 . 0 0 1$ </td><td> $0 . 2 8 3 \pm 0 . 0 0 1$ </td><td> $2 5 9 . 2 8 4 \pm 0 . 3 9 6$ </td><td> $0 . 6 9 3 \pm 0 . 0 0 2$ </td><td> $0 . 3 3 4 \pm 0 . 0 0 1$ </td><td></td><td>5.3</td></tr><tr><td>Diag</td><td>NLL</td><td> $0 . 4 1 8 \pm 0 . 0 0 3$ </td><td> $0 . 2 2 1 \pm 0 . 0 0 4$ </td><td> $1 8 5 . 2 8 8 \pm 0 . 6 4 0$ </td><td> $0 . 7 0 5 \pm 0 . 0 4 0$ </td><td> $0 . 2 8 1 \pm 0 . 0 0 1$ </td><td> $\mathbf { + 0 . 4 0 6 \pm 0 . 0 1 7 }$ </td><td>3.2</td></tr><tr><td>Diag</td><td>KS</td><td> $0 . 4 1 5 \pm 0 . 0 0 0$ </td><td> $0 . 2 6 0 \pm 0 . 0 0 1$ </td><td> $\mathbf { 1 8 4 . 4 1 9 \pm 0 . 0 8 3 }$ </td><td> $0 . 7 2 5 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 2 7 9 \pm 0 . 0 0 0 }$ </td><td> $+ 0 . 8 7 4 \pm 0 . 0 0 2$ </td><td>3.5</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 4 9 9 \pm 0 . 0 4 0$ </td><td> $0 . 3 0 1 \pm 0 . 0 2 9$ </td><td> $2 1 2 . 7 7 1 \pm 8 . 5 8 3$ </td><td> $0 . 8 1 2 \pm 0 . 0 4 9$ </td><td> $0 . 3 4 0 \pm 0 . 0 1 3$ </td><td> $\mathbf { + 0 . 3 9 0 \pm 0 . 0 1 6 }$ </td><td>5.8</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $0 . 4 1 5 \pm 0 . 0 0 0$ </td><td> $0 . 2 5 9 \pm 0 . 0 0 0$ </td><td> $\mathbf { 1 8 4 . 3 7 2 \pm 0 . 1 0 5 }$ </td><td> $0 . 7 2 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 2 7 9 \pm 0 . 0 0 0 }$ </td><td> $+ 0 . 8 5 1 \pm 0 . 0 0 7$ </td><td>3.3</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 4 1 7 \pm 0 . 0 0 2$ </td><td> $0 . 2 4 4 \pm 0 . 0 1 1$ </td><td> $\mathbf { 1 8 4 . 2 9 4 \pm 0 . 3 8 4 }$ </td><td> $0 . 7 1 6 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 2 7 9 \pm 0 . 0 0 1 }$ </td><td></td><td>3.3</td></tr><tr><td>Spectral space</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td> $\mathbf { 0 . 3 9 8 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 9 3 \pm 0 . 0 0 0 }$ </td><td> $1 8 5 . 1 9 1 \pm 0 . 3 1 9$ </td><td> $\mathbf { 0 . 6 3 9 \pm 0 . 0 0 1 }$ </td><td> $0 . 2 8 3 \pm 0 . 0 0 1$ </td><td> $+ 1 . 4 2 4 \pm 0 . 0 8 5$ </td><td>2.8</td></tr></table>

Surface temperature prediction. On ERA5 (Table 16), SCORE attains the best or tied-best mean on every metric and the best average rank. On MSE and CRPS it coincides with the NLL-trained diagonal model. By Lemma 3.3, these two scores are those of the first stage of SCORE, a CRPS-trained diagonal model on the same backbone that uses 20 of the 30 epochs, so the first stage matches the NLL-trained marginals at two thirds of the budget. The gain from the correlation core shows in the scores that are sensitive to the dependence structure: relative to the NLL-trained diagonal model, the variogram score improves by 7% and the log score by 1.3 nats per coordinate, whereas the ES and KS differences are within about one standard deviation, in line with their weaker sensitivity to dependence (Appendix G.1). Among the baselines, the generative sampler improves CRPS and ES over the deterministic model but not MSE. LorD trained with the NLL collapses. Its mean remains accurate, but its CRPS deteriorates by a factor of four and its VS by more than a factor of sixty, while its log score is better than that of the diagonal model, which indicates a degenerate covariance rather than a poor mean. Trained with the kernel score, LorD is on par with the KS-trained diagonal model but behind the NLL-trained one on CRPS and VS. The assumptions of SCORE are well justified on this task, since temperature anomalies on a regular grid have a distance-dependent, approximately stationary correlation, which a circulant core captures.

Table 16: Test scores on ERA5 surface temperature.
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td> $\mathbf { 0 . 0 8 0 \pm 0 . 0 0 1 }$ </td><td> $0 . 2 0 6 \pm 0 . 0 0 1$ </td><td> $5 1 . 8 8 1 \pm 0 . 1 9 7$ </td><td> $0 . 0 8 6 \pm 0 . 0 0 1$ </td><td> $0 . 0 7 6 \pm 0 . 0 0 1$ </td><td></td><td>5.6</td></tr><tr><td>Diag</td><td>NLL</td><td> $\mathbf { 0 . 0 7 9 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 1 4 4 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 3 6 . 7 0 1 \pm 0 . 1 5 5 }$ </td><td> $0 . 0 7 2 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 7 2 \pm 0 . 0 0 1 }$ </td><td> $- 0 . 0 5 2 \pm 0 . 0 0 4$ </td><td>2.3</td></tr><tr><td>Diag</td><td>KS</td><td> $0 . 0 8 0 \pm 0 . 0 0 0$ </td><td> $0 . 1 5 6 \pm 0 . 0 0 2$ </td><td> $3 7 . 0 4 7 \pm 0 . 0 8 9$ </td><td> $0 . 0 8 1 \pm 0 . 0 0 2$ </td><td> $0 . 0 7 4 \pm 0 . 0 0 0$ </td><td> $+ 0 . 4 4 2 \pm 0 . 1 4 7$ </td><td>4.1</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $0 . 0 8 2 \pm 0 . 0 0 1$ </td><td> $0 . 6 3 1 \pm 0 . 0 3 7$ </td><td> $1 5 3 . 6 9 6 \pm 1 0 . 2 2 8$ </td><td> $4 . 6 0 5 \pm 0 . 5 7 9$ </td><td> $0 . 3 8 3 \pm 0 . 0 0 8$ </td><td> $- 0 . 1 1 3 \pm 0 . 0 0 2$ </td><td>5.9</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $\mathbf { 0 . 0 8 0 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 5 5 \pm 0 . 0 0 3$ </td><td> $3 6 . 9 8 1 \pm 0 . 1 8 9$ </td><td> $0 . 0 7 9 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 7 3 \pm 0 . 0 0 1 }$ </td><td> $+ 0 . 4 8 7 \pm 0 . 3 1 5$ </td><td>3.8</td></tr><tr><td>SB</td><td>KS</td><td> $0 . 0 8 1 \pm 0 . 0 0 0$ </td><td> $0 . 1 5 4 \pm 0 . 0 0 2$ </td><td> $3 6 . 9 9 3 \pm 0 . 0 7 9$ </td><td> $0 . 0 8 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 7 3 \pm 0 . 0 0 0$ </td><td></td><td>4.1</td></tr><tr><td colspan="9">Spectral space</td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td> $\mathbf { 0 . 0 7 9 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 1 4 4 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 3 6 . 5 4 4 \pm 0 . 0 5 6 }$ </td><td> $\mathbf { 0 . 0 6 7 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 0 7 2 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { - 1 . 3 3 4 \pm 0 . 0 4 6 }$ </td><td>1.6</td></tr></table>

Monocular depth estimation. On NYU Depth v2 (Table 17), SCORE performs poorly across scores, ranking behind the diagonal model on every sample-based score. Part of the gap is already in the marginals. The first stage of SCORE, trained with the CRPS for 50 epochs, does not reach the diagonal model trained with the log score for 75 epochs, with CRPS 0.0204 against 0.0192, and the second stage cannot recover this by construction (Lemma 3.3). The remaining gap on ES and VS means that the learned correlation is worse than no correlation at all, which points to the assumptions of SCORE. Depth maps are not stationary. Their correlation is created at object boundaries, so a single circulant kernel couples pixels across edges that are unrelated. Moreover, the log score of SCORE explodes. A plausible explanation for this is that a core trained with the bounded kernel score has no incentive to keep the predicted covariance well conditioned (Theorem 3.1), while the log score penalizes directions of small predicted variance without bound (Section 2.3).

Table 17: Test scores on NYU Depth v2. For readability MSE, CRPS, VS, KS in units of $1 0 ^ { - 3 }$
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td> $\mathbf { 1 . 8 4 7 \pm 0 . 0 1 8 }$ </td><td> $2 4 . 9 8 3 \pm 0 . 1 7 0$ </td><td> $1 7 . 3 6 5 \pm 0 . 0 7 8$ </td><td> $2 . 2 9 9 \pm 0 . 0 1 8$ </td><td> $\mathbf { 1 . 8 3 9 \pm 0 . 0 1 8 }$ </td><td></td><td>4.2</td></tr><tr><td>Diag</td><td>NLL</td><td> $\mathbf { 1 . 7 9 8 \pm 0 . 0 4 1 }$ </td><td> $\mathbf { 1 9 . 2 4 1 \pm 0 . 3 3 4 }$ </td><td> ${ \bf 1 2 . 7 4 6 \pm 0 . 1 6 7 }$ </td><td> $\mathbf { 2 . 1 5 1 \pm 0 . 0 5 3 }$ </td><td> $\mathbf { 1 . 7 8 7 \pm 0 . 0 4 1 }$ </td><td> $+ 1 . 0 5 0 \pm 0 . 7 2 2$ </td><td>2.2</td></tr><tr><td>Diag</td><td>KS</td><td> $\mathbf { 1 . 8 3 8 \pm 0 . 0 3 1 }$ </td><td> $\mathbf { 1 9 . 2 0 2 \pm 0 . 2 1 4 }$ </td><td> $1 3 . 0 9 8 \pm 0 . 1 1 9$ </td><td> $2 . 2 9 0 \pm 0 . 0 4 5$ </td><td> $\mathbf { 1 . 8 2 9 \pm 0 . 0 3 1 }$ </td><td> $- 1 . 6 8 6 \pm 0 . 0 6 1$ </td><td>2.8</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $6 3 . 0 1 5 \pm 6 . 3 2 9$ </td><td> $2 1 6 . 1 1 9 \pm 2 2 . 7 0 4$ </td><td> $1 1 2 . 1 6 3 \pm 1 1 . 8 5 9$ </td><td> $3 0 . 3 4 4 \pm 1 2 . 9 0 2$ </td><td> $1 3 0 . 0 9 7 \pm 2 3 . 3 4 1$ </td><td> $- 0 . 0 5 8 \pm 1 . 0 8 0$ </td><td>6.6</td></tr><tr><td>LorD r=1</td><td>KS</td><td> $\mathbf { 1 . 8 1 8 \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 1 9 . 0 1 8 \pm 0 . 0 7 5 }$ </td><td> $1 2 . 9 9 8 \pm 0 . 0 6 0$ </td><td> $2 . 2 6 4 \pm 0 . 0 2 2$ </td><td> $\mathbf { 1 . 8 0 9 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { - 1 . 8 4 9 \pm 0 . 0 4 5 }$ </td><td>2.6</td></tr><tr><td>SB</td><td>KS</td><td> ${ \bf 1 . 8 5 3 \pm 0 . 0 3 8 }$ </td><td> $2 5 . 0 5 2 \pm 0 . 2 7 1$ </td><td> $1 7 . 3 3 5 \pm 0 . 1 4 3$ </td><td> $2 . 2 7 5 \pm 0 . 0 2 9$ </td><td> ${ \bf 1 . 8 4 4 \pm 0 . 0 3 8 }$ </td><td></td><td>4.4</td></tr><tr><td colspan="9">Spectral space</td></tr><tr><td>SCORE</td><td> $\mathrm { C R P S } { + } \mathrm { K S }$ </td><td> $1 . 9 0 7 \pm 0 . 0 2 1$ </td><td> $2 0 . 4 4 4 \pm 0 . 1 0 3$ </td><td> $1 3 . 7 3 5 \pm 0 . 0 7 9$ </td><td> $2 . 3 0 6 \pm 0 . 0 3 3$ </td><td> $1 . 8 9 7 \pm 0 . 0 2 2$ </td><td> $( + 3 . 7 0 9 \pm 7 . 9 7 7 ) \times 1 0 ^ { 3 }$ </td><td>4.6</td></tr></table>

Graph-based temperature post-processing. This task is the only one where we consider a different transform, namely the Graph Fourier transform. While SCORE naturally extends, the resulting predictive distribution requires a Cholesky decomposition for loss optimization, thus offering no additional computational benefits. Therefore, the results validate our method against fully expressive alternatives that can recover the correct structure, i.e., Cholesky decomposition and the sample-based model. Table 18 shows that the latter takes the best rank and leads to the best performance overall, while the Cholesky decomposition obtains the best score in terms of the VS and NLL. Yet, our method remains competitive and ranks third overall, with the individual metrics being close to the sample-based method and better or comparable to the Cholesky method. However, here LorD trained with the kernel score outperforms SCORE and ranks second overall. However, as shown in Figure 3, LorD estimates many correlations between stations that are far away, and thus are most likely not actually correlated for the specific prediction. SCORE on the other hand, provides more clustered and local dependencies.

Table 18: Test scores on EUPPBench station post-processing.
<table><tr><td>Method</td><td>Objective</td><td>MSE↓</td><td>CRPS↓</td><td>ES↓</td><td>VS↓</td><td>KS↓</td><td>NLL/d↓</td><td>∅Rank↓</td></tr><tr><td colspan="9">Physical space</td></tr><tr><td>Det</td><td>MSE</td><td> $2 . 1 4 4 \pm 0 . 0 1 9$ </td><td> $1 . 1 2 7 \pm 0 . 0 0 5$ </td><td> $1 5 . 0 7 8 \pm 0 . 0 7 0$ </td><td> $2 . 1 0 1 \pm 0 . 0 0 5$ </td><td> $0 . 8 1 1 \pm 0 . 0 0 3$ </td><td></td><td>8.2</td></tr><tr><td>Diag</td><td>NLL</td><td> $2 . 1 4 7 \pm 0 . 0 1 1$ </td><td> $0 . 8 0 1 \pm 0 . 0 0 2$ </td><td> $1 0 . 8 0 0 \pm 0 . 0 2 6$ </td><td> $1 . 9 4 2 \pm 0 . 0 2 9$ </td><td> $0 . 4 7 9 \pm 0 . 0 0 1$ </td><td> $+ 1 . 7 3 6 \pm 0 . 0 0 4$ </td><td>6.0</td></tr><tr><td>Diag</td><td>KS</td><td> ${ \bf 2 . 1 0 0 \pm 0 . 0 2 0 }$ </td><td> $0 . 7 9 3 \pm 0 . 0 0 3$ </td><td> $1 0 . 6 5 3 \pm 0 . 0 4 5$ </td><td> $1 . 8 6 7 \pm 0 . 0 2 7$ </td><td> $0 . 4 7 6 \pm 0 . 0 0 1$ </td><td> $+ 1 . 7 5 0 \pm 0 . 0 0 5$ </td><td>4.5</td></tr><tr><td>LorD r=1</td><td>NLL</td><td> $2 . 2 6 7 \pm 0 . 0 4 9$ </td><td> $0 . 8 3 0 \pm 0 . 0 1 7$ </td><td> $1 1 . 0 7 3 \pm 0 . 2 0 2$ </td><td> $1 . 7 5 7 \pm 0 . 0 0 9$ </td><td> $0 . 4 7 8 \pm 0 . 0 0 2$ </td><td> $+ 1 . 5 8 2 \pm 0 . 0 0 3$ </td><td>5.8</td></tr><tr><td> $\mathrm { L o r D } r { = } 1$ </td><td>KS</td><td> $\mathbf { 2 . 1 0 6 \pm 0 . 0 2 6 }$ </td><td> $\mathbf { 0 . 7 9 5 \pm 0 . 0 0 6 }$ </td><td> $\mathbf { 1 0 . 6 1 4 \pm 0 . 0 6 1 }$ </td><td> $1 . 7 9 6 \pm 0 . 0 1 6$ </td><td> $\mathbf { 0 . 4 7 3 \pm 0 . 0 0 1 }$ </td><td> $+ 1 . 6 6 9 \pm 0 . 0 0 3$ </td><td>3.4</td></tr><tr><td>Chol</td><td>NLL</td><td> $2 . 3 3 5 \pm 0 . 0 9 9$ </td><td> $0 . 8 3 6 \pm 0 . 0 1 9$ </td><td> $1 1 . 1 2 9 \pm 0 . 2 2 5$ </td><td> $\mathbf { 1 . 7 1 4 \pm 0 . 0 2 0 }$ </td><td> $0 . 4 7 6 \pm 0 . 0 0 2$ </td><td> $\mathbf { + 1 . 3 3 9 \pm 0 . 0 0 9 }$ </td><td>5.1</td></tr><tr><td>Chol</td><td>KS</td><td> $\mathbf { 2 . 1 0 9 \pm 0 . 0 3 0 }$ </td><td> $0 . 7 9 7 \pm 0 . 0 0 5$ </td><td> ${ \bf 1 0 . 6 3 6 \pm 0 . 0 7 7 }$ </td><td> $1 . 7 8 3 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 4 7 4 \pm 0 . 0 0 1 }$ </td><td> $+ 3 3 1 . 5 5 3 \pm 4 1 2 . 9 7 0$ </td><td>4.3</td></tr><tr><td>SB</td><td>KS</td><td> $\mathbf { 2 . 0 9 5 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 7 8 8 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 1 0 . 5 5 3 \pm 0 . 0 1 8 }$ </td><td> $1 . 7 8 8 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 4 7 3 \pm 0 . 0 0 0 }$ </td><td></td><td>2.6</td></tr><tr><td colspan="9">Spectral space</td></tr><tr><td>SCORE</td><td>CRPS+KS</td><td> $2 . 1 3 6 \pm 0 . 0 0 9$ </td><td> $0 . 7 9 9 \pm 0 . 0 0 2$ </td><td> $1 0 . 7 0 8 \pm 0 . 0 2 1$ </td><td> $1 . 7 7 5 \pm 0 . 0 0 6$ </td><td> $0 . 4 7 5 \pm 0 . 0 0 0$ </td><td> $+ 1 . 4 9 5 \pm 0 . 0 0 5$ </td><td>4.0</td></tr></table>

## H.2 RUNTIME ANALYSIS

Here, we further analyze the runtime of the different methods and show that our method outperforms the baselines in terms of the compute-performance tradeoff. Similar to Figure 4, Figure 9 and Figure 10 show the increase in computational cost versus the performance improvement against the diagonal baseline, per dataset. In addition, Table 19 shows the average runtime per task in ms per gradient step. Here, all methods use the kernel score as an objective to ensure a fair runtime comparison. Since this objective outperforms the log score significantly (compare Table 4), it does not favor the performance of our method. The figure shows that our method generally obtains the best performance improvement while having the smallest computational increase against the diagonal baseline. These are particularly strong for VS and NLL, but consistent across all metrics. While for some datasets, namely Depth, EUPPBench, and Weather, the gains are not across all metrics and are even negative for some, our method is still the fastest among the baselines that model dependence. This is also highlighted in Table 19. While our method has a compute advantage due to the second stage with frozen weights, the table also shows that using only the first stage, SCORE is comparable to the deterministic or diagonal method. Further, the ablations in H.4 highlight that the second stage improves performance even under decreasing epoch budgets.

## H.3 CALIBRATION RESULTS

Here, we provide additional results regarding the calibration of our method and the baselines, as described in G.2. To ensure that the results mainly reflect the covariance parameterization, we compare all methods using the kernel score as an objective.

![](images/3f11102bc9101f6389c74ab95b30b5c35f73f25d7b293aa02ced6bd9fc4dabea.jpg)  
Figure 9: Comparison of relative improvement and computational cost against the diagonal Gaussian baseline for the time series datasets. All methods are trained on the Gaussian kernel score. Each point represents the median performance over five seeds for one task, i.e., uni-/multivariate and prediction horizon T.

Since the kernel score beats the NLL in performance significantly, this is to no additional advantage for our method. In general, we only evaluate the calibration for predictive tasks with a single channel, i.e., we omit the multivariate time series task, as one would need to analyze each channel separately.

Figure 11 shows diagnostics of the marginal, mean, and scale calibration for the different methods and datasets. In general, almost all methods admit a good marginal calibration. However, SCORE overall shows the best mean and scale calibration, similar to the Cholesky approximation, which however is only available for the low-dimensional tasks. The sampling-based method collapses to a point prediction on multiple datasets and thus admits only poor calibration in these cases. Similarly, Figure 12 shows the calibration diagnostics for the dependence structure across multiple lags h. Here, none of the methods are well calibrated, especially for very small lags. Still, SCORE admits better-calibrated predictions compared to the other methods, especially for larger lags and for the ETTh and EUPPBench datasets. This generally highlights the difficulty of learning a calibrated approximate dependence structure across complex domains, as no method is able to truly recover it.

![](images/d782dacc5ca51a598fcfdb05d7d514f5c54ed2db7b8ee75b0701224ec758da63.jpg)  
Figure 10: Comparison of relative improvement and computational cost against the diagonal Gaussian baseline for the remaining datasets. All methods are trained on the Gaussian kernel score. Each point represents the median performance over five seeds.

Table 19: Training cost of every predictive family on every benchmark task, in milliseconds per training step. Mean ± standard deviation over 5 seeds on the exact same compute requirements, epoch budget and equal batch size within a task. For our SCORE method, the stages are averaged by the number of epochs used, respectively. The fastest method overall is highlighted in italics, while the fastest method that models a correlation structure is highlighted in bold.
<table><tr><td colspan="10"></td></tr><tr><td>Task</td><td>Det</td><td>Diag</td><td>LorD r=1</td><td>LorD r=16</td><td>Chol</td><td>SB</td><td>SCORE</td><td>(stage 1)</td></tr><tr><td>ETTh1 (multiv.), T=96</td><td>12 ± 0.088</td><td>20 ± 0.989</td><td>36 ± 0.208</td><td>37 ± 0.162</td><td>一</td><td>35 ± 0.206</td><td>21 ± 0.189</td><td>21 ± 0.227</td></tr><tr><td>ETTh1 (multiv.), T=192</td><td>24 ± 0.422</td><td>33 ± 0.217</td><td>41 ± 0.461</td><td>43 ± 0.898</td><td></td><td>40 ± 0.087</td><td>33 ± 0.336</td><td>33 ± 0.013</td></tr><tr><td>ETTh1 (univ.), T=96</td><td>11 ± 0.125</td><td>19 ± 0.372</td><td>34 ± 0.278</td><td>35 ± 0.247</td><td>33 ± 0.089</td><td>22 ± 0.308</td><td>21 ± 0.334</td><td>20 ± 0.362</td></tr><tr><td>ETTh1 (univ.), T=192</td><td>24 ± 0.242</td><td>34 ± 0.972</td><td>42 ± 0.098</td><td>41 ± 1.012</td><td>51 ± 0.474</td><td>35 ± 0.202</td><td> ${ \bf 3 4 } \pm { \bf 1 . 0 9 9 }$ </td><td> $3 3 \pm 1 . 6 4 5$ </td></tr><tr><td>ETTh2 (multiv.), T=96</td><td>24 ± 0.066</td><td>34 ± 0.992</td><td>41 ± 0.512</td><td>42 ± 0.772</td><td></td><td>40 ± 0.444</td><td>33 ± 0.049</td><td>33 ± 0.265</td></tr><tr><td>ETTh2 (multiv.), T=192</td><td>24 ± 0.367</td><td>32 ± 1.256</td><td>41 ± 0.852</td><td>42 ± 0.376</td><td></td><td>40 ± 0.110</td><td>34 ± 0.402</td><td>33 ± 0.542</td></tr><tr><td>ETTh2 (univ.), T=96</td><td>24 ± 0.323</td><td>34 ± 1.389</td><td>42 ± 1.108</td><td>42 ± 0.332</td><td>40 ± 0.610</td><td>35 ± 0.915</td><td>34 ± 1.540</td><td>33 ± 1.712</td></tr><tr><td>ETTh2 (univ.), T=192</td><td>23 ± 0.028</td><td>33 ± 0.049</td><td>42 ± 1.473</td><td>41 ± 0.714</td><td>52 ± 0.485</td><td>36 ± 0.444</td><td>33 ± 1.099</td><td>32 ± 1.004</td></tr><tr><td>ETTm1 (multiv.), T=96</td><td>43 ± 0.240</td><td>52 ± 0.156</td><td>78 ± 0.202</td><td>84 ± 0.351</td><td></td><td>115 ± 0.829</td><td>58 ± 0.079</td><td>62 ± 0.095</td></tr><tr><td>ETTm1 (multiv.), T=192</td><td>44 ± 0.132</td><td>52 ± 0.094</td><td>79 ± 0.751</td><td>88 ± 0.217</td><td></td><td>116 ± 0.438</td><td>59 ± 0.523</td><td>63 ± 0.604</td></tr><tr><td>ETTm1 (univ.), T=96</td><td>24 ± 0.050</td><td>33 ± 0.098</td><td>42 ± 1.182</td><td>42 ± 1.591</td><td></td><td>37 ± 0.348</td><td>33 ± 1.145</td><td>33 ± 1.025</td></tr><tr><td>ETTm1 (univ.), T=192</td><td>24 ± 0.356</td><td>33 ± 0.169</td><td>41 ± 0.070</td><td>43 ± 0.805</td><td></td><td>36 ± 0.346</td><td>33 ± 0.512</td><td>33 ± 0.499</td></tr><tr><td>ETTm2 (multiv.), T=96</td><td>44 ± 0.189</td><td>52 ± 0.309</td><td>78 ± 0.167</td><td>84 ± 0.693</td><td></td><td>115 ± 0.745</td><td>58 ± 0.157</td><td>62 ± 0.157</td></tr><tr><td>ETTm2 (multiv.), T=192</td><td>44 ± 0.205</td><td>52 ± 0.196</td><td>79 ± 0.800</td><td>88 ± 0.250</td><td></td><td>116 ± 0.583</td><td>59 ± 0.621</td><td>63 ± 0.659</td></tr><tr><td>ETTm2 (univ.), T=96</td><td>25 ± 0.385</td><td>33 ± 0.053</td><td>41 ± 0.846</td><td>42 ± 0.265</td><td></td><td>36 ± 0.557</td><td>34 ± 1.091</td><td>33 ± 0.873</td></tr><tr><td>ETTm2 (univ.), T=192</td><td>24 ± 0.388</td><td>34 ± 1.623</td><td>42 ± 0.457</td><td>42 ± 0.552</td><td></td><td>36 ± 0.889</td><td>34 ± 0.395</td><td>34 ± 0.226</td></tr><tr><td>Illness, T=24</td><td>18 ± 0.446</td><td>27 ± 2.737</td><td>48 ± 1.243</td><td>50 ± 1.018</td><td></td><td> $3 2 \pm 0 . 9 6 8$ </td><td>30 ± 1.086</td><td>29 ± 1.352</td></tr><tr><td>Illness, T=36</td><td>11 ± 1.039</td><td>26 ± 4.785</td><td>48 ± 2.059</td><td>47 ± 1.360</td><td></td><td> $3 0 \pm 2 . 1 0 5$ </td><td>29 ± 1.575</td><td>28 ± 2.417</td></tr><tr><td>Weather, T=96</td><td>125 ± 0.775</td><td>132 ± 0.950</td><td>191 ± 1.115</td><td>202 ± 2.475</td><td></td><td> $3 2 3 \pm 1 . 2 9 9$ </td><td>153 ± 0.741</td><td>169 ± 0.972</td></tr><tr><td>Weather, T=192</td><td>124 ± 0.884</td><td>133 ± 1.001</td><td>192 ± 1.523</td><td>220 ± 1.185</td><td></td><td> $3 2 5 \pm 0 . 6 7 7$ </td><td>154 ± 0.736</td><td>171 ± 0.933</td></tr><tr><td>Electricity, T=96</td><td>478 ± 2.673</td><td>486 ± 2.578</td><td>658 ± 3.645</td><td>746 ± 2.115</td><td></td><td> $1 2 0 2 \pm 2 . 3 1 1$ </td><td> $\mathbf { 5 6 8 \pm 1 . 5 2 1 }$ </td><td>642 ± 1.653</td></tr><tr><td>Electricity, T=192</td><td>474 ± 2.831</td><td>485 ± 0.582</td><td>664 ± 1.499</td><td>840 ± 0.764</td><td></td><td> $1 2 0 5 \pm 2 . 2 7 3$ </td><td> $\mathbf { 5 6 5 \pm 0 . 9 1 5 }$ </td><td>636 ± 0.936</td></tr><tr><td>Traffic, T=96</td><td>314 ± 1.657</td><td>322 ± 0.575</td><td>445 ± 0.812</td><td>557 ± 0.526</td><td></td><td> $8 0 7 \pm 1 . 3 3 2$ </td><td> $\mathbf { 3 7 6 \pm 0 . 9 7 1 }$ </td><td>422 ± 1.136</td></tr><tr><td>Traffic, T=192</td><td>315 ± 1.116</td><td>326 ± 1.316</td><td>451 ± 0.557</td><td>694 ± 0.817</td><td></td><td> $8 2 4 \pm 1 . 2 5 0$ </td><td> ${ \bf 3 7 9 \pm 0 . 4 9 7 }$ </td><td>426 ± 0.653</td></tr><tr><td>ERA5 t2m</td><td>637 ± 5.115</td><td>641 ± 3.880</td><td>713 ± 4.156</td><td>755 ± 4.311</td><td></td><td> $7 8 3 \pm 2 . 4 0 7$ </td><td> $5 5 8 \pm \ : 1 . 0 4 1$ </td><td>697 ± 1.657</td></tr><tr><td>NYU depth (BTS)</td><td>187 ± 0.301</td><td>194 ± 0.232</td><td>304 ± 0.154</td><td>622 ± 0.394</td><td></td><td> $6 1 1 \pm 1 . 1 7 4$ </td><td> $\mathbf { 2 6 4 \pm 0 . 3 1 3 }$ </td><td>285 ± 0.411</td></tr><tr><td>EUPPBench</td><td>272 ± 0.911</td><td>287 ± 1.133</td><td>289 ± 0.478</td><td>289 ± 1.659</td><td>288 ± 1.548</td><td>277 ± 0.808</td><td>243 ± 0.789</td><td>274 ± 0.917</td></tr></table>

![](images/813bac0e9bec707a4f4a770ba2351829447fa0fdec5aec2143d1999a91d1d750.jpg)  
Figure 11: Empirical CDFs of the PIT values of the marginal, location and scale pre-rank functions for the different predictive methods. A calibrated prediction lies on the diagonal. Results are aggregated over five seeds.

![](images/2c564e47d671f170f4e71f420eb27377fd0a0eb2839cd63e2a3f3e02c727ffa7.jpg)  
Figure 12: Empirical CDFs of the PIT values of the variogram pre-rank function with respect to different spatial (or temporal) lags h for the different predictive methods. A calibrated prediction lies on the diagonal. Results are aggregated over five seeds.

## H.4 ABLATION STUDIES

Log score vs. kernel score. Here we provide details on the results in Table 4, which shows that training with the kernel score instead of the log score yields better performance in the large majority of comparisons, on four evaluation metrics. For each covariance parameterization a and task t, we compare the seed-averaged scores $\bar { s } _ { m } ^ { \mathrm { K S } } ( a , t )$ and $\bar { s } _ { m } ^ { \mathrm { N L L } } ( a , t )$ of the two arms, which differ only in the objective. We do this for each metric m other than the two training losses, giving 167 comparisons per metric (all $( a , t )$ pairs for which both arms were trained). Since metrics are not on a common scale across tasks, we aggregate signs rather than magnitudes, which is scale-free and conservative. Since the tasks of one data set share data and hyperparameters, we treat the $C = 1 1$ data sets as clusters and assume independence between data sets. With $\mathcal { T } _ { c }$ the comparisons of data set $c ,$ let

$$
S _ { c } = \sum _ { ( a , t ) \in \mathcal { T } _ { c } } \mathrm { s i g n } \bigl ( \bar { s } _ { m } ^ { \mathrm { N L L } } ( a , t ) - \bar { s } _ { m } ^ { \mathrm { K S } } ( a , t ) \bigr ) , \qquad S = \sum _ { c } S _ { c } ,
$$

where ties contribute zero (none occurred). Under the sharp null hypothesis that the objective has no effect, the labels KS and NLL are exchangeable within each data set. Swapping them maps $S _ { c } \mapsto - S _ { c }$ and keeps any within-cluster dependence intact, so $( S _ { c } ) _ { c } \overset { d } { = } ( \varepsilon _ { c } S _ { c } )$ for all $\varepsilon \in \{ - 1 , 1 \} ^ { C }$ . This is a cluster-level sign-flip randomization test [70, Ch. 15], the sign-test analogue of the clustered signed-rank tests of Rosner et al. [71], Datta and Satten [72]; see also Hemerik and Goeman [73]. The exact two-sided p-value i

$$
p = 2 ^ { - C } \left| \left\{ \varepsilon : \left| \sum _ { c } \varepsilon _ { c } S _ { c } \right| \ge | S | \right\} \right| \ge 2 / 2 ^ { C } \approx 0 . 0 0 1 .
$$

This floor is reached on MSE, CRPS and ES, where every data set has a kernel-score majority, and remains significant after a Holm correction over the four metrics. On VS, EUPPBench favors the log score $( S _ { c } = - 5 )$ , which gives $p = 1 0 / 2 0 4 8 \approx$ 0.005.

Loss and low-rank comparison. This ablation aims to disentangle the performance gains of the different loss functions from our proposed covariance approximation. Here we focus on the analysis of the low-rank method and simultaneously assess the impact of the size of the rank r. To achieve this, we train the low-rank method with increasing ranks $r \in$ {1, 2, 4, 8, 16} for both the NLL and the kernel score, again across five different seeds. We then analyze, for each metric, the relative performance to our method, as shown in Figure 13. Using the kernel score as a loss function leads to significantly better performance for the LorD approximation across all metrics. However, the performance does not seem to depend on increasing rank r. Using the NLL as an objective leads to a much higher relative loss, but it decreases with larger r, although not for the other metrics.

Impact of stage 2 training and epoch budget. Here, we provide an additional ablation to verify that our staged training approach brings additional benefits compared to simply training the first stage. At the same time, we vary the total training budget to make sure that the additional performance gains do not depend on the chosen number of epochs. For computational reasons, we only provide this ablation for the univariate time series and the EUPPBench task, which nevertheless present a clear picture, as can be seen in Figure 14. These results highlight that, especially for the multivariate metrics, the second-stage training significantly increases performance beyond the marginal first stage. Furthermore, this is irrespective of the training budget, since even for very few epochs, where our model leads to poor marginal performance, the corresponding multivariate metrics are significantly lower compared to training only the first stage.

![](images/4c90b6e0a0b80354e1a07c7f8b925a92a0be0efcf1014a2a35f48f9546fb7dee.jpg)  
Figure 13: Relative performance of the low-rank method against ours with increasing rank r.

![](images/b65bf66dfa11ef754fec0ecd9154e77526f483b8073b3e9f327ccd372b6daa33.jpg)  
Figure 14: Performance of staged training (3:1 split) vs. first-stage only using the same computational budget.