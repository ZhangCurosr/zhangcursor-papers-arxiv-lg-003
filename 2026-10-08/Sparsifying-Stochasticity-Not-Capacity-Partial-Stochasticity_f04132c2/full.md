# Sparsifying Stochasticity, Not Capacity: Partial Stochasticity via Deep Weight Factorization of Prior Scales

Marius P. Linhard Université Grenoble Alpes, France

## Abstract

Bayesian neural networks need not be fully stochastic to be universal conditional density approximators, but it remains open which parameters should be stochastic. We learn this split by applying deep weight factorization to the prior scales, which are the standard deviations of the parameter priors, while fitting the functional prior to a Gaussian process with a maximum mean discrepancy objective. A parameter whose prior scale falls below a cutof becomes deterministic and is optimized during inference, so the regularizer sparsifies stochasticity rather than capacity. We give a certificate for universal conditional density approximation that is checkable in linear time, together with a minimal repair when it fails. We further show that the common hybrid scheme of sampling some parameters and optimizing the others is stochastic approximation for a type-II maximum a posteriori objective, and that coupled step sizes can leave a tracking error that does not vanish as the step size shrinks. On a bimodal target, the learned split stays close to an unconstrained reference across all budgets and is insensitive to the cutof, while random masks that distribute the same prior scales across layers are worse by up to two orders of magnitude. On UCI benchmarks, our method performs on par with a fully stochastic network while keeping about half of its parameters deterministic.

## 1 INTRODUCTION

As neural networks are increasingly deployed in realworld applications, reliable uncertainty quantification becomes essential. Bayesian neural networks address this by treating their parameters as random variables, which combines the flexibility of neural networks with principled uncertainty estimates.

Maurizio Filippone KAUST, Saudi Arabia

A recent work (Sharma et al., 2023) has shown theoretically and empirically that it is not necessary to treat all parameters in a stochastic way to obtain good uncertainty estimates. It is even enough to make the number of stochastic parameters of the network match the target dimension as long as stochasticity is introduced in the early layers and the downstream network has general function approximation properties (Leshno et al., 1993). This result is promising since it allows for the reduction in the number of stochastic parameters.

However, partial stochasticity complicates model selection by adding another modeling choice: which parameters should we treat in a stochastic way? The theory does not give any indication on how to make this choice. Due to the combinatorially large number of possible ways to assign parameters to the stochastic and deterministic groups, it is computationally impractical to find a solution via brute force. An additional complication is how to choose sensible priors over stochastic parameters, which is crucial for obtaining good performance and sound uncertainty quantification.

Building on deep weight factorization (Kolb et al., 2025), we apply the factorization not to the network weights but to the prior standard deviations, which we refer to as prior scales. Sparsity in the scales means sparsity in stochasticity: a parameter whose prior scale collapses to zero is deterministic. Unlike pruning and spike-and-slab-based constructions, which keep pruned parameters at zero in the posterior, our deterministic parameters remain free to be optimized during inference, so the capacity of the fitted network is retained.

Learning the split changes the inference problem. Partially stochastic networks are typically fitted by a hybrid scheme that samples the stochastic parameters while optimizing the deterministic ones. Despite its widespread use, it has not been established what this scheme approximates: it does not sample the joint posterior, nor is the deterministic block a standard maximum a posteriori (map) estimate, since the stochastic parameters are integrated throughout optimization. With this work, we aim to close this gap.

Our contributions are: (1) a method that learns the deterministic/stochastic split by imposing deep weight factorization on prior scales. This sparsifies stochasticity rather than capacity, and for a factorization depth of $D \geq 3$ the non-zero scales are bounded away from zero at stationary points (Theorem 4.1); (2) a functional prior fitted by the maximum mean discrepancy (mmd), where the degree-two polynomial kernel reduces to moment matching with weights fixed by the kernel, together with a decomposition of the Kullback-Leibler (kl) divergence that explains why a sparse network can match at most the moments (Proposition 4.3, Lemma A.3); (3) a deterministic certificate for the universal conditional density approximation property, that is checkable in linear time, and a minimal repair when it fails (Proposition 4.7, Corollary A.9); (4) a characterization of the hybrid inference scheme as stochastic approximation for a type-II map objective, and a tracking bound showing that coupled sampler and optimizer step sizes leave an error term that does not shrink with the step size, which is attained by a Gaussian example (Theorem 4.9, Theorem A.13, Corollary 4.10, Proposition A.14).

## 2 BACKGROUND

Notation Suppose we have a dataset with n inputtarget pairs $\mathcal { D } = \{ { \bf x _ { i } } , { \bf y _ { i } } \} _ { 1 \leq i \leq n }$ . We denote the parameters of a neural network by Θ; the stochastic parameters are denoted by $\Theta _ { s }$ and the deterministic ones by $\Theta _ { d } .$ . By $\Theta ^ { ( l ) }$ we denote the parameters of layer l. Furthermore, we denote the neural network by $\mathbf { f } ( \cdot ; \Theta )$ the l’th layer of the neural network by $\mathbf { f } _ { l } \big ( \cdot ; \Theta ^ { ( l ) } \big )$ , and the hyperparameters of the prior by $\Psi _ { \theta } , \theta \in \Theta$ and $\Psi : = \{ \Psi _ { \theta } , \theta \in \Theta \}$ . In addition, we denote by L the number of layers and by $d _ { i }$ the dimension of the output of the i’th layer, where $d _ { 0 }$ is the dimension of the input data. For a random variable Z we write $p ( \mathbf { Z } )$ for its distribution, which need not have a density.

Functional prior The prior on the parameters also induces a functional prior through

$$
p ( \mathbf { f } ; \Psi ) = \int p ( \mathbf { f } \vert \Theta ) p ( \Theta ; \Psi ) d \Theta
$$

Here, $p ( \mathbf { f } | \Theta )$ is defined by the network architecture and, therefore, deterministic. However, this now enables us to change our view from parameters to functions.

Inference The posterior is intractable due to the intractability of the marginal likelihood $\begin{array} { r l } { p ( \mathcal { D } ) } & { { } = } \end{array}$ $\begin{array} { r l } { \int p ( \mathcal { D } | \Theta ) p ( \Theta ) d \Theta } \end{array}$ . We sample it with stochastic gradient Hamiltonian Monte Carlo (sghmc) (Chen et al., 2014), which scales to large datasets.

Partially and fully stochastic neural networks We first introduce fully-connected neural networks (multi-layer perceptrons (mlps)), where the l’th layer is defined as

$$
\mathbf { f } _ { l } ( \mathbf { x } ; \boldsymbol { \Theta } ^ { ( l ) } ) = \mathbf { W } _ { l } \varphi ( \mathbf { f } _ { l - 1 } ( \mathbf { x } ; \boldsymbol { \Theta } ^ { ( l - 1 ) } ) ) + \mathbf { b } _ { l }
$$

where $\varphi$ is a possibly non-linear function (activation function), $\mathbf { f } _ { l - 1 }$ is the output of the $( l - 1 )$ ’th layer, ${ \bf W } _ { l } \in  { \bf \Psi }$ $\mathbb { R } ^ { d _ { l } \times d _ { l - 1 } }$ is the weight matrix of the l’th layer, and ${ \bf b } _ { l } \in  { }$ $\mathbb { R } ^ { d _ { l } }$ is the bias of the l’th layer. For the first layer we set $\mathbf { f } _ { 1 } ( \mathbf { x } ; \boldsymbol { \Theta } ^ { ( 1 ) } ) = \mathbf { W } _ { 1 } \mathbf { x } + \mathbf { b } _ { 1 }$ . In contrast to fully stochastic neural networks, where all parameters are stochastic (Neal, 1996), partially stochastic neural networks have both deterministic and stochastic parameters (Sharma et al., 2023).

Deep weight factorization Hof (2017) showed that $\ell _ { 1 }$ regularization is equivalent to $\ell _ { 2 }$ regularization of a two-factor Hadamard parameterization, which avoids the non-diferentiability of the $\ell _ { 1 }$ penalty. Kolb et al. (2025) extended this to depth $D > 2$ through:

$$
\operatorname* { m i n } _ { \boldsymbol { \omega } _ { 1 } , \ldots , \boldsymbol { \omega } _ { D } } \mathcal { L } \left( \mathbf { x } , \mathbf { y } ; \bigodot _ { i = 1 } ^ { D } \boldsymbol { \omega } _ { i } \right) + \frac { \lambda } { D } \sum _ { i = 1 } ^ { D } \lVert \boldsymbol { \omega } _ { i } \rVert _ { 2 } ^ { 2 } ,\tag{1}
$$

where $\odot$ denotes the element-wise Hadamard-product. As Equation (1) inflates the dimensionality of the objective, the optimization is very sensitive to the initialization of the parameters as reported in Kolb et al. (2025).

## 3 RELATED WORKS

Functional priors in Bayesian neural networks The specification of priors is an important, yet dificult task as the prior has a direct impact on the performance of the neural network and should capture any prior knowledge about the data. For a thorough review of priors in Bayesian deep learning we refer the reader to the work of Fortuin (2022). A common trick, which was also mentioned by Fortuin (2022), is to change the perspective from the parameter space to the functional space and impose a prior on the function induced by the neural network (Flam-Shepherd et al., 2017; Wild et al., 2022). Tran et al. (2022) showed that optimizing the functional prior induced by the network to emulate a Gaussian process (gp) before sampling from the posterior can improve performance substantially.

Partially stochastic neural networks Fully stochastic neural networks are well-known and extensively researched (MacKay, 1995; Neal, 1996). For a comprehensive review of Bayesian neural networks (bnns) we refer the reader to (Arbel et al., 2023).

The notion of partially stochastic neural networks, however, is a rather recent development driven by the work of Sharma et al. (2023). In this work, the authors have contributed theoretically and empirically to the development and justification of partial stochasticity by proving that many popular architectures have the property of being universal conditional density approximators (ucdas); this means that under mild topological constraints, these models can approximate any conditional distribution. Since then, interest in partially stochastic neural networks, their application, and their properties has grown (e.g., (Calvo-Ordoñez et al., 2024; Allec and Ziatdinov, 2025; Prabhudesai et al., 2023)).

Sparsity in neural networks The weak (Frankle and Carbin, 2019) and strong lottery ticket hypotheses (Natale et al., 2024) give the justification for eforts to sparsify neural networks. One possible way to impose sparsity on the parameters is to perform automatic relevance determination, which is placing hyperpriors on the priors and shrinking scales continuously (Neal, 1996). So a parameter is rarely exactly deterministic and the split is never explicit. Many other works, e.g., (Hubin and Storvik, 2024; Wang et al., 2021; Deng et al., 2019), have focused on spike-and-slab priors (Mitchell and Beauchamp, 1988). However, all of these approaches prune capacity of the network by setting parameters to zero; our approach is diferent. We make the prior scales and mean learnable parameters and we then optimize the functional prior to prune the prior scales. This makes the parameters deterministic and the split explicit.

Selection of the stochastic and deterministic parameters The choice between stochastic and deterministic parameters for partially stochastic neural networks is a relatively new task for model selection. Closest to our setting, Andrade and Sato (2025) select the stochastic parameters from a map estimate. Subnetwork inference (Daxberger et al., 2021), which we include as SNI-PSNN, belongs to the same family of map-based selection methods. These methods first train a full network and select the split from its map estimate. Our split is learned from the functional prior, which uses only the inputs and requires no trained model. Sparse subspace variational inference (Li et al., 2024) removes parameters from the network and thereby reduces its capacity, whereas our deterministic parameters remain free to be optimized during inference. Rochussen (2024) fixes a structured subset of parameters in order to break weight symmetries, which is complementary to learning the split.

## 4 METHODS

Throughout the paper, we fit the functional prior by minimizing the mmd between the functional prior of the network and the target gp (for an introduction see (Rasmussen and Williams, 2008)) at random measurement locations. Previous work used the (sliced) Wasserstein distance for this purpose (Tran et al., 2022, 2021). The mmd (Gretton et al., 2012) is cheap to estimate from samples and allows the kernel to be chosen for the task. We specify the kernels used in our experiments in Section 5. We propose a novel way to find a split between stochastic and deterministic parameters by employing a deep weight factorization (dwf) parameterization (Kolb et al., 2025) for the prior scales of the parameters. In this section, we detail our method while supporting it with theoretical results. All proofs are in Section A.

## 4.1 Learning the split

Parameterization: For every $\theta \in \Theta ^ { ( l ) }$ we introduce factors $( \omega _ { \theta } ^ { i } ) _ { i = 1 } ^ { D } \in \mathbb { R } ^ { D }$ , their product $\begin{array} { r } { u _ { \theta } = \prod _ { i = 1 } ^ { D } \omega _ { \theta } ^ { ( i ) } } \end{array}$ and $\sigma _ { \theta } : = H ( u _ { \theta } ) , H ( u ) = ( h ( u ) - h ( 0 ) ) ^ { 2 } , \bar { h } ( \bar { u } ) =$ $\log ( 1 + \exp ( u ) )$ . We refer to $\sigma _ { \theta }$ as the prior scale of θ. We chose this reparameterization of the prior scales to ensure that the standard deviations are non-negative with $H ( u ) = 0 \iff u = 0$ . The prior is $\theta \sim \mathcal { N } ( \mu _ { \theta } , \sigma _ { \theta } ^ { 2 } )$ for each parameter $\theta \in \Theta$ , where both µ and σ are trainable parameters. We initialize ${ \pmb \mu } = { \bf 0 }$ . With an odd activation and a zero-mean target process, ${ \pmb \mu } = { \bf 0 }$ is a stationary point of the objective, and the learned means remain at zero in our experiments. We perform the dwf on the prior scales only and initialize them according to (Kolb et al., 2025).

For measurement locations $\mathbf { X } = \left( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N } \right)$ , let $P _ { \mathbf { X } } ^ { \mu , \sigma }$ denote the distribution of $( \mathbf { f } ( \mathbf { x } _ { 1 } ; \Theta ) , \dots , \mathbf { f } ( \mathbf { x } _ { N } ; \Theta ) )$ under the prior with means $\pmb { \mu }$ and scales $\sigma ,$ and $Q \mathbf { x }$ the corresponding marginal of the gp. For a kernel k we define

$$
\begin{array} { r } { \mathcal { L } _ { k } ( \pmb { \mu } , \pmb { \sigma } ) : = \mathbb { E } _ { \mathbf { X } \sim \mathcal { G } ^ { N } } \big [ \mathrm { M M D } _ { k } ^ { 2 } \big ( P _ { \mathbf { X } } ^ { \pmb { \mu } , \pmb { \sigma } } , Q _ { \mathbf { X } } \big ) \big ] , } \end{array}
$$

where the generator $\mathcal { G }$ of the measurement locations is described below. In practice both distributions are replaced by the empirical measures of function samples (Remark A.6). We cast the prior optimization problem as

$$
\operatorname* { m i n } _ { \mu , \omega } \mathcal { L } ( \mu , \omega ) : = \mathcal { L } _ { k } ( \mu , H ( \mathbf { u } ) ) + \sum _ { l = 1 } ^ { L } \frac { \lambda _ { l } } { D } \sum _ { \theta \in \Theta ^ { ( l ) } } \sum _ { i = 1 } ^ { D } \bigl ( \omega _ { \theta } ^ { ( i ) } \bigr ) ^ { 2 } .
$$

During the optimization, a parameter whose prior scale falls below a cutof c is set to be deterministic, so that it receives no gradient from the objective. After optimization, we apply the minimal repair of Corollary A.9 whenever the certificate of Proposition 4.7 fails. On the UCI datasets this changes at most 34 parameters, or 0.3% of the network.

A natural question is to what extent c afects sparsity. We show in Theorem 4.1 that at stationary points there is a gap in absolute values between the values that are set to 0 and the surviving prior scales. If c lies within this gap, changing c has no efect on the result. Kolb et al. (2025) have shown that performing dwf of depth D is equivalent to optimizing with a $\ell _ { \frac { 2 } { D } } .$ -quasi norm. Since $2 / D < 1$ for $D \geq 3 .$ , the penalty is only a quasi-norm, as the triangle inequality fails, and it is no longer convex. Its derivative grows without bound as $| u _ { \theta } |  0 .$ , so small non-zero values are expensive and are pushed to exactly zero, which brings the penalty closer to $\ell _ { 0 }$ and makes it sparser than $\ell _ { 1 }$ (cf. Remark A.2). For the statistical properties of $\ell _ { p }$ penalties with $p < 1$ we refer the reader to Knight and Fu (2000) and Chartrand (2007), and for their realization through deep factorizations to Kolb et al. (2025). Proposition A.1 extends this to layer-specific $\lambda _ { l }$ , which yields the following theorem:

Theorem 4.1 (No small non-zero scales). Let $D \geq 3 .$ let $\theta \in \Theta ^ { ( l ) }$ . Let $\tilde { \mathcal { L } } ( \mathbf { u } ) : = \mathcal { L } _ { k } ( \pmb { \mu } , H ( \mathbf { u } ) )$ as a function of u<sub>θ</sub> with all other variables fixed, and suppose $| \tilde { \mathcal { L } } ^ { \prime } ( u _ { \theta } ) | \leq$ $G _ { l } . ~ I f ~ ( \omega _ { \theta } ^ { ( 1 ) } , \ldots , \omega _ { \theta } ^ { ( D ) } )$ is a stationary point of $\mathcal { L }$ in these factors, then either all factors vanish, in which case $u _ { \theta } = \sigma _ { \theta } = 0$ or

$$
| u _ { \theta } | \geq t _ { l } : = \left( \frac { 2 \lambda _ { l } } { D G _ { l } } \right) ^ { \frac { D } { D - 2 } } \quad a n d \quad \sigma _ { \theta } \geq H ( - t _ { l } ) > 0
$$

The bound holds at exact stationary points. An optimizer step stops at approximate stationarity, and $t _ { l }$ shrinks as the bound $G _ { l }$ on the gradient grows, so the gap is not guaranteed in practice. In our experiments the surviving scales extend down to the cutof, so the split depends on the choice of c in practice. The fitted model, however, is insensitive to c over more than an order of magnitude (Figure 4).

Remark 4.2. For comparable $G _ { l } ,$ choosing $\lambda _ { 1 } < \lambda _ { 2 } <$ $\cdots < \lambda _ { L }$ enforces $t _ { 1 } < t _ { 2 } < \cdots < t _ { L }$ : later layers are sparsified more aggressively.

A consequence of this theorem is that $c <$ min<sub>l</sub> $H ( - t _ { l } )$ removes only scales that are exactly zero at every stationary point.

Borgwardt et al. (2006) showed that an mmd with a polynomial kernel of degree d matches moments up to degree d. For the degree-two kernel used in our experiments, this takes the following form.

Proposition 4.3 (The mmd reduced to moment matching). Let $k ( \mathbf { x } , \mathbf { y } ) ~ = ~ ( 1 + \langle \mathbf { x } , \mathbf { y } \rangle ) ^ { 2 } ~ o n ~ \mathbb { R } ^ { N }$ , and let $\mathbf { M } _ { P } : = \Sigma _ { P } + \mu _ { P } \pmb { \mu } _ { P } ^ { \top }$ denote the second-moment matrix. Then for any two distributions with finite second moments,

$$
\mathrm { M M D } _ { 2 } ^ { 2 } ( P , Q ) \ : = \ : 2 \| \pmb { \mu } _ { P } - \pmb { \mu } _ { Q } \| _ { 2 } ^ { 2 } \ : + \ : \| \mathbf { M } _ { P } - \mathbf { M } _ { Q } \| _ { F } ^ { 2 } .
$$

Here the subscript 2 of $\mathrm { M M D } _ { 2 } ^ { 2 } ( P , Q )$ denotes the use of the polynomial kernel of degree 2.

The weights of the mean and the second-moment term are thus fixed by the kernel and need not be tuned. In our implementation the objective is scaled by $\scriptstyle { \frac { 1 } { 2 N } }$ which only rescales λ.

To estimate the objective we draw measurement locations from a generator $\mathcal { G }$ which mixes the empirical distribution of the training inputs and a uniform distribution over the input range as suggested by Sun et al. (2019) and also done by Tran et al. (2022): $\mathcal { G } = \rho _ { M } \hat { \mathbb { P } } _ { \mathbf { X } _ { \mathrm { t r a i n } } } + ( 1 - \rho _ { M } ) \mathcal { U } ( \mathbf { x } _ { \mathrm { m i n } } , \mathbf { x } _ { \mathrm { m a x } } )$ , i.e. each location is a training input with probability $\rho _ { M }$ and uniform otherwise. We use $\rho _ { M } = 0 . 7$ for the experiments. The choice of $\rho _ { M }$ determines where the functional prior is actually constrained: the uniform component forces the match to hold away from the training data, which is precisely the region in which out-of-distribution uncertainty is evaluated.

## 4.2 Preserving expressiveness

We analyze how masking afects the ucda property of Sharma et al. (2023) (Theorem A.7) through two notions.

Definition 4.4 (Bias coverage). Let $S : = \{ \theta \in \Theta ^ { ( 1 ) }$ $\sigma _ { \theta } > 0 \}$ be the surviving set in the first layer. The bias coverage is $U ( S ) : = \# \{ j \in \{ 1 , \dots , d _ { 1 } \} : b _ { j } ^ { ( 1 ) } \in S \}$ , and $T : = \{ j :$ no first-layer parameter of unit j lies in S} is the set of fully deterministic units.

Remark 4.5. The definition counts biases rather than weights for a reason. A stochastic input weight contributes $\sigma _ { j , i } \varepsilon _ { j , i } x _ { i }$ , which vanishes at $\scriptstyle { \textbf { * } } = { \textbf { 0 } }$ , so the scale $s _ { j }$ in the proof of Proposition 4.7 would not be bounded away from zero and the recovery map would fail there. A stochastic bias enters additively and keeps $s _ { j } \geq \sigma _ { j } ^ { b } > 0$ for every input.

Definition 4.6 (ucda of a masked network). Fix a mask. The masked network is a ucda if for every $( \mathbf { X } , \mathbf { Y } )$ as in Theorem A.7 with a continuous generator and every $\varepsilon > 0$ , there exist values of the deterministic parameters and a Gaussian distribution with independent nondegenerate coordinates on $\Theta _ { s }$ such that $\begin{array} { r } { \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { X } } d _ { \mathrm { B L } } ( p ( \mathbf { f } ( \mathbf { x } ; \Theta ) ) , p ( \mathbf { Y } | \mathbf { X } = \mathbf { x } ) ) \le \varepsilon } \end{array}$ , where $d _ { \mathrm { B L } }$ denotes the bounded-Lipschitz metric.

The definition only states that the mask does not restrict expressiveness, not that the approximation is found in practice. It is hard to check directly, but the following certificate can be checked in linear time.

Proposition 4.7 (Deterministic certificate). Let $\mathcal { X } \subset$ $\mathbb { R } ^ { d _ { 0 } }$ be compact and let f have $L - 1 \ge 2$ hidden layers with a continuous, strictly monotone, non-polynomial activation $\varphi .$ Stochastic parameters may occur in any layer. If

$$
U ( S ) \geq d _ { L } \quad a n d \quad | T | \geq d _ { 0 } ,
$$

then the masked network is a ucda. Both conditions are checkable in $\mathcal { O } ( | \Theta ^ { ( 1 ) } | )$ time.

Remark 4.8. If the deterministic first-layer weights are fixed, for example after training, the second condition becomes rank $\bar { \bf W } _ { T } ^ { ( 1 ) } ) = d _ { 0 }$ , where $\bar { \bf W } _ { T } ^ { ( 1 ) }$ are the firstlayer weights of the units in $T ,$ , and Proposition A.8 gives the minimal repair for this case.

Note that X need not be the raw input space. If a fixed continuous map precedes the network, the proposition applies unchanged with $d _ { 0 }$ the dimension of its output.

If the certificate of Proposition 4.7 fails, a minimal repair restores it by making at most $d _ { L }$ first-layer biases stochastic and, if needed, the $d _ { 0 } - | T |$ cheapest units outside these $d _ { L }$ fully deterministic (Corollary A.9).

## 4.3 Inference

The prior optimization provides three inputs to the inference stage. The split fixes the stochastic dimension $m : = | \Theta _ { s } |$ and hence determines whether the model can represent the target conditional at all, the surviving scales $\hat { \pmb { \sigma } } _ { s }$ define the prior over the stochastic block, and the means $\hat { \pmb { \mu } }$ center the prior of both blocks. Let $( \hat { \pmb { \mu } } , \hat { \pmb { \sigma } } )$ be the output of the prior optimization and set $k : = | \Theta _ { d } |$ |. We write $\ell ( \pmb \theta _ { d } , \pmb \theta _ { s } ) : = \log p ( \mathcal { D }$ $\begin{array} { r } { \theta _ { d } , \theta _ { s } ) , \ Z ( \theta _ { d } ) \ : = \ p ( \mathcal { D } \ | \ \theta _ { d } ) \ = \ \int e ^ { \ell ( \theta _ { d } , \theta _ { s } ) } p ( \theta _ { s } ) \mathrm { d } \theta _ { s } } \end{array}$ and $\begin{array} { r l r } { J _ { \tau } ( \pmb { \theta } _ { d } ) } & { : = } & { \log Z ( \pmb { \theta } _ { d } ) - \frac { 1 } { 2 \tau ^ { 2 } } \lVert \pmb { \theta } _ { d } - \hat { \pmb { \mu } } _ { d } \rVert _ { 2 } ^ { 2 } . } \end{array}$ , where $\tau = \gamma \log ( 2 )$ is governed by a hyperparameter γ. We use this reparameterization so that $\gamma = 1$ corresponds to the value of the softplus function at 0. The hyperparameter γ controls how far the deterministic parameters may move from their prior mean. By Proposition A.12, $\gamma \downarrow$ 0 freezes them at $\hat { \mu } _ { d } ,$ a finite γ gives a type-II map estimate, and $\gamma \uparrow \infty$ gives type-II maximum likelihood.

The deterministic block is optimized with Adam (Kingma and Ba, 2014) under the prior $\mathcal { N } ( \hat { \pmb { \mu } } _ { d } , \tau ^ { 2 } \mathbf { I } _ { k } )$ and the stochastic block is sampled with sghmc (Chen et al., 2014) under the prior $\Pi _ { \theta \in \Theta _ { \mathrm { s } } } \mathcal { N } ( \hat { \mu } _ { \theta } , \hat { \sigma } _ { \theta } ^ { 2 } )$ . Let Π denote the joint posterior of $( \theta _ { d } , \bar { \theta } _ { s } )$ under these two priors.

Theorem 4.9 (The near-degenerate approximation is a type-II map estimate). Let $\mathcal { P } _ { G }$ be the set of densities $q _ { s } ,$ with finite entropy and $\mathbb { E } _ { q _ { s } } [ G + | \log p ( \pmb { \theta } _ { s } ) | + | \ell ( \mathbf { 0 } , \pmb { \theta } _ { s } ) | ] <$ $\infty ,$ where G is the envelope from Assumption $A . 1 0 \ ( i i )$ and let $q _ { \varepsilon } : = \mathcal { N } ( \pmb { \theta } _ { d } , \varepsilon ^ { 2 } \pmb { I } _ { k } )$ . Under Assumption A.10, for all $\pmb { \theta } _ { d } \in \mathbb { R } ^ { k }$ and $q _ { s } \in \mathcal { P } _ { G }$

$$
\begin{array} { r } { \mathrm { K L } ( q _ { \varepsilon } \otimes q _ { s } \| \Pi _ { \tau } ) = - \displaystyle \frac { k } { 2 } \log ( 2 \pi e \varepsilon ^ { 2 } ) + F ( \pmb \theta _ { d } , q _ { s } ) + R _ { \varepsilon } } \\ { F ( \pmb \theta _ { d } , q _ { s } ) = \mathrm { K L } ( q _ { s } \| p ( \cdot | \mathcal D , \pmb \theta _ { d } ) ) - J _ { \tau } ( \pmb \theta _ { d } ) + c , } \end{array}
$$

with $c \in \mathbb { R }$ being a constant, $\begin{array} { r } { | R _ { \varepsilon } | \leq C ( \pmb { \theta } _ { d } ) \mathbb { E } _ { q _ { s } } [ G ] \varepsilon + \frac { k \varepsilon ^ { 2 } } { 2 \tau ^ { 2 } } } \end{array}$ $f o r \varepsilon \leq 1$ and $C ( \pmb { \theta } _ { d } )$ locally bounded. The minimizers of F are $q _ { s } ^ { \star } = p ( \cdot | \mathcal { D } , \pmb { \theta } _ { d } ^ { \star } )$ with $\pmb { \theta } _ { d } ^ { \star } \in \mathrm { a r g }$ max $J _ { \tau }$

By Fisher’s identity, ∇<sub>θ</sub> log Z(θ<sub>d</sub>) = $\mathbb { E } _ { p ( \pmb { \theta } _ { s } | \mathcal { D } , \pmb { \theta } _ { d } ) } [ \nabla _ { \pmb { \theta } _ { d } } \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) ]$ (Cappé et al., 2005), so the hybrid scheme is stochastic approximation EM for $J _ { \tau }$ , with the sampler acting as a Monte Carlo E-step. Because the target of the sampler moves with $\theta _ { d } ,$ the step sizes matter. The next result shows that if sampler and optimizer share a step size, the bound on the tracking error does not vanish as the step size shrinks.

Corollary 4.10 (Coupled step sizes yield a non-vanishing tracking error). Assume in addition to the setting of Theorem A.13 that $\rho ( h ) = 1 - c h + o ( h )$ with $c > 0$ and $\delta _ { t } \leq G \eta$ for all t. Then for all t

$$
e _ { t } \leq \rho ( h ) ^ { t } e _ { 0 } + \frac { L _ { \pi } G \eta } { c h + o ( h ) }
$$

$( a )$ Coupled. $h = \eta$ . The stationarity part of the bound tends to $\textstyle { \frac { L _ { \pi } G } { c } }$ as $\eta  0 . \ ( b )$ Decoupled. $I f h  0$ and $\frac { \eta } { h }  0$ , then sup $_ t [ W _ { 1 } ( \pmb { \mu } _ { t } ^ { \pmb { \theta } _ { s } } , \pi _ { \pmb { \theta } _ { d _ { t } } } ) - \rho ( h ) ^ { t } e _ { 0 } ]  0$

This bound is attained for a Gaussian example (Proposition A.14) and therefore sharp. We therefore let the optimizer step size decay faster than the sampler step size (Section E).

## 5 EXPERIMENTS

Unless stated otherwise, we use a mlp with two hidden layers of 100 units and a tanh activation. For regression we use CONCRETE, WINE, BIKESHARING and BOSTON, and for classification BANKNOTE, HTRU2, ONLINESHOPPERS and CREDIT. All of these datasets come from the UCI suite (Kelly et al., 2023). For each dataset we use ten random splits into 90% training and 10% test data and report the mean and standard deviation over the splits. We evaluate regression by root mean squared error (rmse), negative log-likelihood (nll) and expected calibration error (ece), and classification by accuracy (acc), nll and ece. Details are given in Section E. For the prior optimization we use the mmd with the degree-two polynomial kernel $k ( \mathbf x , \mathbf y ) = ( 1 + \langle \mathbf x , \mathbf y \rangle ) ^ { 2 }$ which matches the mean and the second moment of the function values (Proposition 4.3).

![](images/cb1a689f403a410698adc6811171776c124b2b19ef8d02fd2a6408775eecb003.jpg)  
Figure 1: Share of deterministic parameters per layer for DWF-PSNN before repair, with mean and standard deviation over ten splits

## 5.1 Learned Split

Figure 1 shows the learned split. It is stable across data splits, keeps the first and the output layer largely deterministic and leaves about half of the middle layer stochastic. The coverage condition holds in every run. The condition $| T | \geq d _ { 0 }$ fails on datasets with many input features, and the minimal repair then changes between 1 and 34 parameters (Section C).

## 5.2 Conditions of the certificate

![](images/06b4389cca219a533ca53a021f49a4c3d3ccf10920db77bf609210c3f2956d34.jpg)  
Figure 2: Predictive distribution at the bimodal input, with the true conditional dashed and certified configurations in blue. (a) Unconstrained reference, (b) output layer randomized, (c) no nonlinearity, (d) every weight randomized with fixed scales, (e) one randomized bias.

The certificate constrains a masked network through bias coverage, through the number of fully deterministic units, and through the activation. Figure 2 shows one configuration per condition on a target whose conditional distribution is bimodal, so that any Gaussian conditional is excluded by construction. In the reference (a) the noise enters as an additional input and every weight is deterministic, so the map is unconstrained and the reachable error is limited only by optimization. A single randomized bias in an admitted position (e) comes close to that reference. Randomizing the output layer (b) leaves the bias coverage at zero and fails, with sixty-four randomized parameters as much as with one. Randomizing every weight with fixed prior scales (d) fails from the opposite side, since no fully deterministic units remain. Removing the nonlinearity (c) fails while count and position are correct. The three conditions are therefore not vacuous and are needed together. Table 1 numerically supports the findings by providing the mean Wasserstein-1 distance.

Table 1: Mean Wasserstein-1 distance to the true conditional over nine held out inputs, with the standard error over five data seeds. The sampling floor is the distance between two independent samples from the true conditional at this sample size, and the point predictor is what a model reaches by matching the conditional median.
<table><tr><td>variant</td><td>condition violated</td><td> $W _ { 1 }$ </td></tr><tr><td>sampling floor point predictor</td><td></td><td>.020 .681</td></tr><tr><td>noise input,  $d _ { L }$ </td><td>reference, none</td><td> $. 0 5 3 \pm . 0 0 4$ </td></tr><tr><td>layer  $1 , d _ { L }$ </td><td>none</td><td> $. 0 7 4 \pm . 0 0 2$ </td></tr><tr><td>layer 1, dL, weight</td><td>vanishes at  $x = 0$ </td><td> $. 2 8 4 \pm . 0 0 2$ </td></tr><tr><td>output layer,  $d _ { L }$ </td><td>bias coverage</td><td> $. 5 6 8 \pm . 0 0 2$ </td></tr><tr><td>output layer, all</td><td>bias coverage</td><td> $. 4 9 5 \pm . 0 0 4$ </td></tr><tr><td>layer 1, dL, linear</td><td>activation</td><td> $. 9 3 5 \pm . 0 0 2$ </td></tr><tr><td>full</td><td>deterministic units</td><td> $5 . 4 1 1 \pm . 0 0 8$ </td></tr><tr><td>deterministic</td><td>no randomness</td><td> $. 6 8 6 \pm . 0 0 2$ </td></tr></table>

We have added a more detailed version of Table 1 in Section D (Table 15). There we change the notation from layer 1 to first layer.

![](images/1b04bca75836a7ea5731b912073aa900fa0102347a9ce3c10361ef7eb42e21d8.jpg)  
Figure 3: Learned and random splits on the bimodal target over five data seeds. Random masks permute the learned scales within each layer (circles) or across all layers (triangles) and are colored by whether they satisfy the certificate.

Randomizing up to sixty-four parameters of the output layer never approaches the certified configuration (Figure 6 in Section D), so the count of stochastic parameters does not substitute for their position. Figure 3 varies the budget instead. The learned split stays at $W _ { 1 } \approx 0 . 0 6$ between 6.5% and 99.6% deterministic parameters, and permuting its scales within a layer has no efect, since the units of a layer are exchangeable. Permuting them across layers is worse by up to two orders of magnitude and often worse than the fully deterministic model. The learned split thus matters through how it allocates stochasticity across layers. The certificate is suficient but not necessary and leaves the scales free, so it does not separate these masks with fixed scales cleanly. Figure 4 varies the cutof applied after the optimization. The prior scales do not separate into two groups, so the share of deterministic parameters grows with $c ,$ from 90.2% at $c = 0 . 0 0 3$ for $\lambda = 3 \cdot 1 0 ^ { - 6 }$ to 99.7% at $c = 0 . 1$ for $\lambda = 1 \cdot 1 0 ^ { - 5 }$ . The distance to the true conditional nevertheless stays between 0.061 and 0.066 for all cutofs and both values of $\lambda ,$ so the scales removed by a larger cutof are too small to afect the fitted model.

![](images/ba7c7722e0d723acd0557aef6fbcaec5624c7d4038b16ca93b8d5ddcae3fb329.jpg)

![](images/37a80760640d745c4c866461d099d01a643d67cdc0cbcaf0c76321530545f255.jpg)  
Figure 4: Cutof applied after the optimization on the bimodal target, over five data seeds. Left, share of deterministic parameters. Right, distance to the true conditional after refitting with the thresholded scales.

## 5.3 The role of $\gamma$

We study the regimes of Proposition A.12 on CONCRETE. Freezing the deterministic parameters at $\hat { \pmb { \mu } } _ { d } = { \bf 0 }$ in

Table 2: Results on CONCRETE for regimes (ii) and (iii) $( { \mathrm { m e a n } } \pm { \mathrm { s t d } } )$ , with the drift $\delta : = \lVert \pmb { \theta } _ { d } ^ { ( T ) } - \pmb { \hat { \mu } } _ { d } \rVert$ of the deterministic parameters. The sampling hyperparameters difer from those of the benchmark (Section E).
<table><tr><td>γ</td><td>0.03</td><td>2.5</td><td> $1 0 ^ { 1 3 }$ </td></tr><tr><td>RMSE (↓)</td><td> $. 3 1 6 \pm . 0 3 3$ </td><td> $. 2 9 6 \pm . 0 2 9$ </td><td> ${ \bf . 2 9 1 \pm . 0 2 7 }$ </td></tr><tr><td>NLL (↓)</td><td> $. 2 7 4 \pm . 0 9 5$ </td><td> $. 2 0 5 \pm . 0 7 4$ </td><td> $\mathbf { \delta } \mathbf { \cdot 1 9 3 } \pm \mathbf { \delta } . \mathbf { 0 6 6 }$ </td></tr><tr><td>EE (↓)</td><td> $\mathbf { . 0 8 5 \ : \pm { \ : . 0 5 9 } }$ </td><td> $. 1 1 5 \pm . 0 8 3$ </td><td> $. 1 1 9 \pm . 0 7 4$ </td></tr><tr><td> $\delta$ </td><td> $. 0 4 \pm . 0 1$ </td><td> $2 3 . 9 7 \pm . 4 8$ </td><td> $7 6 . 9 4 \pm 4 . 4 1$ </td></tr></table>

regime (i) gives an rmse of $. 9 9 7 \pm . 0 0 1$ and an nll of $4 . 7 4 \pm . 0 2$ , since all deterministic parameters are then zero. Table 2 shows regimes (ii) and (iii). As γ grows, the deterministic parameters move further from their prior mean, and rmse and nll improve monotonically. For $\gamma = 0 . 0 3$ the drift stays small, in line with Corollary A.16. The ece behaves diferently. It is lowest for $\gamma = 0 . 0 3$ and already close to its value in regime (iii) for $\gamma = 2 . 5$ , although all diferences in ece lie within one standard deviation. Small values of $\gamma$ thus trade a little predictive accuracy for better calibration, while every finite $\gamma$ improves substantially on regime (i).

## 5.4 Prior optimization

![](images/e11f15a36869058a15580fb12eacb2b59dc1a12a223eed62b01dc457a4cc6a30.jpg)  
Figure 5: Top (from left to right): the deterministic fraction against iterations and the smallest surviving and the largest pruned prior scales; the mean $( \hat { f } )$ and the uncentered second-moment (gˆ) terms on a logarithmic scale; the covariogram of the optimized prior against the target gp. Bottom (from left to right): DWF-PSNN prior and gp sample paths ; the gp posterior; the DWF-PSNN posterior

Here, we run the prior optimization on a 1D toy dataset. The data is drawn from a gp and we artificially introduce a gap in the middle. Figure 5 shows that the objective and both moment terms converge. The sample paths of the prior and the gp still difer in smoothness, which reflects the non-Gaussian part that the objective does not target (cf. Lemma A.3). The posterior nevertheless gives sensible uncertainty away from the data.

## 5.5 UCI benchmarks

We compare our approach (DWF-PSNN) to a fully stochastic network, where the functional prior is optimized (ALS-BNN); a partially stochastic network, where the first layer is stochastic, the subsequent layers are deterministic, and the functional prior is optimized (FLS-BNN); a partially stochastic neural network that makes the same overall share of parameters deterministic as DWF-PSNN, chosen uniformly at random across all layers and without repair (RAND-PSNN); the partially stochastic neural network as introduced in Daxberger et al. (2021) (SNI-PSNN), which represents map-based selection of the stochastic parameters. On these benchmarks, whose likelihoods are Gaussian or categorical, RAND-PSNN performs on par with DWF-PSNN, so the allocation of stochasticity matters less here than on the bimodal target of Figure 3. We refer the reader to Section B for the detailed results and Section E for more information on the experimental details. Table 3 shows that DWF-PSNN ranks at least as well as the fully stochastic ALS-BNN in all six combinations, although it keeps about half of its parameters deterministic.

Table 3: Mean ranks across the classification and regression datasets, computed from the mean performance on each dataset with tied ranks averaged. Lower ranks indicate better performance. The Nemenyi critical difference (CD) at $\alpha = 0 . 0 5$ is 3.05 for both tasks (N = 4, k = 5). The only pairwise diference that exceeds it is between RAND-PSNN and SNI-PSNN in classification nll.
<table><tr><td colspan="3">Classification (N = 4)</td></tr><tr><td>Method</td><td>ACC (↑)</td><td>NLL (↓)</td><td>ECE (4)</td></tr><tr><td>DWF-PSNN</td><td>2.38</td><td>2.63</td><td>2.88</td></tr><tr><td>ALS-BNN</td><td>2.63</td><td>2.63</td><td>3.13</td></tr><tr><td>FLS-BNN</td><td>2.88</td><td>2.88</td><td>2.50</td></tr><tr><td>RAND-PSNN</td><td>2.38</td><td>1.88</td><td>2.63</td></tr><tr><td>SNI-PSNN</td><td>4.75</td><td>5.00</td><td>3.88</td></tr><tr><td>Friedman p</td><td>0.092</td><td>0.058</td><td>0.736</td></tr></table>

Regression (N = 4)
<table><tr><td>Method</td><td>RMSE (↓)</td><td>NLL (↓)</td><td>ECE (↓)</td></tr><tr><td>DWF-PSNN</td><td>2.00</td><td>2.25</td><td>2.50</td></tr><tr><td>ALS-BNN</td><td>3.25</td><td>2.25</td><td>2.75</td></tr><tr><td>FLS-BNN</td><td>3.38</td><td>3.50</td><td>3.50</td></tr><tr><td>RAND-PSNN</td><td>2.63</td><td>3.00</td><td>2.25</td></tr><tr><td>SNI-PSNN</td><td>3.75</td><td>4.00</td><td>4.00</td></tr><tr><td>Friedman  $p$ </td><td>0.543</td><td>0.434</td><td>0.493</td></tr></table>

Table 4 shows that the certificate fails on the datasets with many input features, both for the learned masks before repair and for random masks with the same budget. The repair changes on average at most 28.1 parameters per split and never more than 34.

## 6 CONCLUSION

Learning with neural networks in a partially stochastic framework requires determining whether each parameter should be treated as deterministic or stochastic.

Table 4: Certificate on the UCI datasets. Share of splits in which the learned mask satisfies the certificate before repair, mean number of parameters changed by the repair, and share of 2,000 random masks with the learned budget that satisfy the certificate.
<table><tr><td>Dataset</td><td> $d _ { 0 }$ </td><td>learned</td><td>repair</td><td>random</td></tr><tr><td>BANKNOTE</td><td>4</td><td>1.00</td><td>0.0</td><td>1.00</td></tr><tr><td>HTRU2</td><td>8</td><td>1.00</td><td>0.0</td><td>1.00</td></tr><tr><td>CONCRETE</td><td>8</td><td>0.90</td><td>0.1</td><td>0.97</td></tr><tr><td>WINE</td><td>10</td><td>0.00</td><td>5.5</td><td>0.09</td></tr><tr><td>BIKESHARING</td><td>12</td><td>0.00</td><td>7.3</td><td>0.01</td></tr><tr><td>BOSTON</td><td>13</td><td>0.00</td><td>7.8</td><td>0.00</td></tr><tr><td>ONLINESHOPPERS</td><td>17</td><td>0.00</td><td>11.7</td><td>0.00</td></tr><tr><td>CREDIT</td><td>23</td><td>0.00</td><td>28.1</td><td>0.00</td></tr></table>

These choices afect guarantees of universal approximation and performance. In this paper, we proposed a novel approach to learn such a split by applying deep weight factorization to the prior scales, while matching the functional prior to a Gaussian process with an mmd objective. A certificate that is cheap to check guarantees universal conditional density approximation, and a minimal repair restores it when it fails. We characterized the inference scheme as a stochastic approximation for a type-II map objective. Then we showed that coupled step sizes of the sampler and the optimizer can leave a persistent tracking error. Our experiments show that the learned split allocates stochasticity where it matters and that the method is competitive on UCI benchmarks.

## 6.1 Limitations and Future Work

Beyond mlps. By the observation after Proposition 4.7, the certificate covers the head of a convolutional or transformer model with a fixed stack below it. Applying the split there is a natural next step, but Corollary A.9 needs a head wider than the feature dimension, which most architectures do not satisfy. The stack itself, with its shared weights, stays outside our analysis.

Closed-loop dynamics. Theorem A.13 and Corollary 4.10 assume that the optimizer path does not depend on the sampler noise. In our algorithm the Adam updates use the current samples, so the analysis covers the tracking part of the error but not the feedback between the two blocks. Extending the bounds to this closed-loop setting is left for future work.

## 6.2 AI Use Statement

In this work, we used generative AI tools to help formulate mathematical claims, to provide ingredients for the proofs and assist in writing them, to give feedback on the methodology and the experiments, and to help implement the methods. Additionally, we used generative AI tools to edit the text for readability and to write and edit code. We have reviewed all AI-assisted work. All proofs were checked line by line by the authors, the code was tested for correctness, and the text was checked for plagiarism. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Allec, S. I. and Ziatdinov, M. (2025). Active and transfer learning with partially Bayesian neural networks for materials and chemicals. Digital Discovery, 4:1284–1297.

Andrade, D. and Sato, K. (2025). On the efectiveness of partially deterministic Bayesian neural networks. Computational Statistics, 40(5):2491–2518.

Arbel, J., Pitas, K., Vladimirova, M., and Fortuin, V. (2023). A primer on Bayesian neural networks: Review and debates. arXiv preprint arXiv:2309.16314.

Borgwardt, K. M., Gretton, A., Rasch, M. J., Kriegel, H.-P., Schölkopf, B., and Smola, A. J. (2006). Integrating structured biological data by Kernel Maximum Mean Discrepancy. Bioinformatics, 22(14):e49– e57.

Calvo-Ordoñez, S., Meunier, M., Piatti, F., and Shi, Y. (2024). Partially stochastic infinitely deep Bayesian neural networks. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org.

Cappé, O., Moulines, É., and Rydén, T. (2005). Inference in hidden Markov models. In Springer Series in Statistics.

Chartrand, R. (2007). Exact reconstruction of sparse signals via nonconvex minimization. IEEE Signal Processing Letters, 14(10):707–710.

Chen, T., Fox, E., and Guestrin, C. (2014). Stochastic Gradient Hamiltonian Monte Carlo. In Xing, E. P. and Jebara, T., editors, Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 1683–1691, Bejing, China. PMLR.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C56S3T.

Csiszár, I. (1975). I-Divergence Geometry of Probability Distributions and Minimization Problems. The Annals of Probability, 3(1):146–158.

Daxberger, E., Nalisnick, E., Allingham, J. U., Antoran, J., and Hernandez-Lobato, J. M. (2021). Bayesian Deep Learning via Subnetwork Inference. In Meila,

M. and Zhang, T., editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 2510–2521. PMLR.

Deng, W., Zhang, X., Liang, F., and Lin, G. (2019). An Adaptive Empirical Bayesian Method for Sparse Deep Learning. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc.

Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5W894.

Flam-Shepherd, D., Requeima, J., and Duvenaud, D. (2017). Mapping Gaussian process priors to Bayesian neural networks. In NIPS Bayesian deep learning workshop, volume 3.

Fortuin, V. (2022). Priors in Bayesian Deep Learning: A Review. International Statistical Review, 90(3):563– 591.

Frankle, J. and Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. In International Conference on Learning Representations.

Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., and Smola, A. (2012). A Kernel Two-Sample Test. Journal of Machine Learning Research, 13(25):723– 773.

Harrison, D. and Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. Journal of Environmental Economics and Management, 5(1):81–102.

Hof, P. D. (2017). Lasso, fractional norm and structured sparse estimation using a Hadamard product parametrization. Computational Statistics & Data Analysis, 115:186–198.

Hubin, A. and Storvik, G. (2024). Sparse Bayesian Neural Networks: Bridging Model and Parameter Uncertainty through Scalable Variational Inference. Mathematics, 12:788.

Kelly, M., Longjohn, R., and Nottingham, K. (2023). The UCI machine learning repository. https://ar chive.ics.uci.edu.

Kingma, D. and Ba, J. (2014). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.

Knight, K. and Fu, W. (2000). Asymptotics for lassotype estimators. The Annals of Statistics, 28(5):1356– 1378.

Kolb, C., Weber, T., Bischl, B., and Rügamer, D. (2025). Deep Weight Factorization: Sparse Learning Through the Lens of Artificial Symmetries. In The Thirteenth International Conference on Learning Representations.

Kuleshov, V., Fenner, N., and Ermon, S. (2018). Accurate Uncertainties for Deep Learning Using Calibrated Regression. In Dy, J. and Krause, A., editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 2796–2804. PMLR.

Leshno, M., Lin, V. Y., Pinkus, A., and Schocken, S. (1993). Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks, 6(6):861–867.

Li, J., Miao, Z., Qiu, Q., and Zhang, R. (2024). Training Bayesian Neural Networks with Sparse Subspace Variational Inference. In Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., and Sun, Y., editors, International Conference on Learning Representations, volume 2024, pages 51238–51249.

Lohweg, V. (2012). Banknote Authentication. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C55P57.

Lyon, R. (2015). HTRU2. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5DK6R.

MacKay, D. J. (1995). Bayesian neural networks and density networks. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 354(1):73–80. Proceedings of the Third Workshop on Neutron Scattering Data Analysis.

Mitchell, T. J. and Beauchamp, J. J. (1988). Bayesian Variable Selection in Linear Regression. Journal of the American Statistical Association, 83(404):1023– 1032.

Natale, E., Ferré, D., Giambartolomei, G., Giroire, F., and Mallmann-Trenn, F. (2024). On the Sparsity of the Strong Lottery Ticket Hypothesis. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C., editors, Advances in Neural Information Processing Systems, volume 37, pages 40565–40592. Curran Associates, Inc.

Neal, R. M. (1996). Bayesian Learning for Neural Networks. Lecture Notes in Statistics. Springer New York, NY, 1 edition.

Prabhudesai, S., Hauth, J., Guo, D., Rao, A., Banovic, N., and Huan, X. (2023). Lowering the computational barrier: Partially Bayesian neural networks for transparency in medical imaging AI. Frontiers in Computer Science, 5:1071174.

Rasmussen, C. E. and Williams, C. K. I. (2008). Gaussian Processes for Machine Learning. Adaptive computation and machine learning. MIT Press, Cambridge, Mass., 3. print edition.

Rochussen, T. (2024). Structured Partial Stochasticity in Bayesian Neural Networks. In Sixth Symposium on Advances in Approximate Bayesian Inference - Non Archival Track.

Sakar, C. and Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5F88Q.

Sharma, M., Farquhar, S., Nalisnick, E., and Rainforth, T. (2023). Do Bayesian Neural Networks Need To Be Fully Stochastic? In Ruiz, F., Dy, J., and van de Meent, J.-W., editors, Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 7694–7722. PMLR.

Sun, S., Zhang, G., Shi, J., and Grosse, R. (2019). Functional Variational Bayesian Neural Networks. In International Conference on Learning Representations.

Tran, B.-H., Rossi, S., Milios, D., and Filippone, M. (2022). All You Need is a Good Functional Prior for Bayesian Deep Learning. Journal of Machine Learning Research, 23(74):1–56.

Tran, B.-H., Rossi, S., Milios, D., Michiardi, P., Bonilla, E. V., and Filippone, M. (2021). Model Selection for Bayesian Autoencoders. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W., editors, Advances in Neural Information Processing Systems, volume 34, pages 19730–19742. Curran Associates, Inc.

Vladimirova, M., Verbeek, J., Mesejo, P., and Arbel, J. (2019). Understanding Priors in Bayesian Neural Networks at the Unit Level. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 6458–6467. PMLR.

Wang, Y., Deng, W., and Lin, G. (2021). Bayesian sparse learning with preconditioned stochastic gradient MCMC and its applications. Journal of Computational Physics, 432:110134.

Welsh, D. J. A. (1970). On Matroid Theorems of Edmonds and Rado. Journal of the London Mathematical Society, s2-2(2):251–256.

Wild, V. D., Hu, R., and Sejdinovic, D. (2022). Generalized Variational Inference in Function Spaces: Gaussian Measures meet Bayesian Deep Learning. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A., editors, Advances in Neural Information Processing Systems, volume 35, pages 3716–3730. Curran Associates, Inc.

Yaida, S. (2020). Non-Gaussian Processes and Neural Networks at Finite Widths. In Proceedings of The First Mathematical and Scientific Machine Learning Conference, volume 107 of Proceedings of Machine Learning Research, pages 165–192. PMLR.

Yeh, I.-C. (1998). Concrete Compressive Strength. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5PK67.

Yeh, I.-C. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C55S3H.

## A THEORETICAL RESULTS AND PROOFS

## A.1 Learning the split

Proposition A.1 (Layer-weighted quasi-norm). Since $\mathcal { L } _ { k }$ depends on the factors only through $\mathbf { u } ,$

$$
\operatorname* { m i n } _ { \omega } \mathcal { L } ( \pmb { \mu } , \omega ) = \operatorname* { m i n } _ { \mathbf { u } } \mathcal { L } _ { k } ( \pmb { \mu } , H ( \mathbf { u } ) ) + \sum _ { l = 1 } ^ { L } \lambda _ { l } \sum _ { \theta \in \Theta ^ { ( l ) } } | u _ { \theta } | ^ { \frac { 2 } { D } }
$$

Proof. Fix θ and $u _ { \theta } = \omega$ . By the AM-GM inequality $\begin{array} { r } { \frac { 1 } { D } \sum _ { i } ( \omega ^ { ( i ) } ) ^ { 2 } \geq ( \Pi _ { i } ( \omega ^ { ( i ) } ) ^ { 2 } ) ^ { \frac { 1 } { D } } = | \omega | ^ { \frac { 2 } { D } } } \end{array}$ , with equality if and only if $| \boldsymbol { \omega } ^ { ( i ) } | = | \boldsymbol { \omega } | ^ { \frac { 1 } { D } }$ for all i. Hence, the inner minimum of $\textstyle \sum _ { i } ( \omega ^ { ( i ) } ) ^ { 2 }$ under the constraint $\begin{array} { r } { \prod _ { i } \omega ^ { ( i ) } = \omega } \end{array}$ equals $D | \omega | ^ { \frac { 2 } { D } }$ . Multiplying by $\frac { \lambda _ { l } } { D }$ and summing over θ and l gives the claim. □

Remark A.2 (Efective penalty of the prior scales). Since for our reparameterization we have $\begin{array} { r } { \sigma _ { \theta } \approx \frac { u _ { \theta } ^ { 2 } } { 4 } } \end{array}$ near zero, the penalty $\lambda _ { l } | u _ { \theta } | ^ { \frac { 2 } { D } }$ behaves like $\lambda _ { l } ( 4 \sigma _ { \theta } ) ^ { \frac { 1 } { D } }$ . On the prior scale itself the regularizer is therefore of $\ell _ { \frac { 1 } { D } }$ type, which is sparser than the $\ell _ { \frac { 2 } { D } }$ quasi-norm acting on u.

Theorem 4.1 (No small non-zero scales). Let $D \geq 3 .$ , let $\theta \in \Theta ^ { ( l ) }$ . Let $\tilde { \mathcal { L } } ( \mathbf { u } ) : = \mathcal { L } _ { k } ( \pmb { \mu } , H ( \mathbf { u } ) )$ as a function of $u _ { \theta }$ with all other variables fixed, and suppose $| \tilde { \mathcal { L } } ^ { \prime } ( u _ { \theta } ) | \leq G _ { l } . \ I f \left( \omega _ { \theta } ^ { ( 1 ) } , \ldots , \omega _ { \theta } ^ { ( D ) } \right)$ is a stationary point of L in these factors, then either all factors vanish, in which case $u _ { \theta } = \sigma _ { \theta } = 0$ or

$$
| u _ { \theta } | \geq t _ { l } : = \left( \frac { 2 \lambda _ { l } } { D G _ { l } } \right) ^ { \frac { D } { D - 2 } } \quad a n d \quad \sigma _ { \theta } \geq H ( - t _ { l } ) > 0
$$

Proof. Write $v : = u _ { \theta }$ and $g : = | \tilde { \mathcal { L } } ^ { \prime } ( v ) |$ . Stationarity in the i-th factor reads $\begin{array} { r } { \tilde { \mathcal { L } } ^ { \prime } ( v ) \Pi _ { j \neq i } v ^ { ( j ) } + \frac { 2 \lambda _ { l } } { D } v ^ { ( i ) } = 0 } \end{array}$ . Multiplying by $v ^ { ( i ) }$ gives $\begin{array} { r } { \tilde { \mathcal { L } } ^ { \prime } ( v ) v + \frac { 2 \lambda _ { l } } { D } ( v ^ { ( i ) } ) ^ { 2 } = 0 } \end{array}$ , so

$$
( v ^ { ( i ) } ) ^ { 2 } = - \frac { D } { 2 \lambda _ { l } } \tilde { \mathcal { L } } ^ { \prime } ( v ) v = : s ^ { 2 } \quad \mathrm { f o r ~ e v e r y ~ } i , \quad | v | = s ^ { D } .
$$

If $s = 0$ , all factors vanish. Otherwise $s > 0 ,$ , and $\begin{array} { r } { s ^ { 2 } = \frac { D } { 2 \lambda _ { l } } g | v | = \frac { D g } { 2 \lambda _ { l } } s ^ { D } } \end{array}$ forces $g > 0$ and $\begin{array} { r } { s ^ { D - 2 } = \frac { 2 \lambda _ { l } } { D g } } \end{array}$ . Hence, $\begin{array} { r } { | v | = s ^ { D } = ( \frac { 2 \lambda _ { l } } { D q } ) ^ { \frac { D } { D - 2 } } } \end{array}$ . This expression is decreasing in g, so $g \leq G _ { l }$ yields $| v | \geq t _ { l }$ . The bound on $\sigma _ { \theta }$ follows from $\begin{array} { r } { \operatorname* { m i n } _ { | u | \geq t } H ( u ) { \bar { = } } H ( - t ) } \end{array}$ □

The following Lemma is a special case of the Pythagorean identity onto exponential families (Csiszár, 1975):

Lemma A.3 (Pythagorean decomposition). Let $P$ be a distribution on $\mathbb { R } ^ { N }$ with finite second moments, mean µ and positive definite covariance matrix $\pmb { \Sigma } _ { i }$ , and let $\tilde { P } : = \mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ . Then $f o r$ every Gaussian $Q = \mathcal { N } ( \mu _ { Q } , \Sigma _ { Q } )$ with positive definite covariance matrix,

$$
\mathrm { K L } ( P \| Q ) = \underbrace { \mathrm { K L } ( P \| \tilde { P } ) } _ { n o n - G a u s s i a n i t y ~ o f ~ P } + \underbrace { \mathrm { K L } ( \tilde { P } \| Q ) } _ { m o m e n t ~ d i s c r e p a n c y } .
$$

In particular, if P has no Lebesgue density, both $\mathrm { K L } ( P \| { \tilde { P } } )$ and $\mathrm { K L } ( P \| Q )$ are infinite.

Proof. $\tilde { P }$ and $Q$ are equivalent to Lebesgue measure. If $P$ is not absolutely continuous, then ${ \mathrm { K L } } ( P \| Q ) =$ $\mathrm { K L } ( P \| \tilde { P } ) = \infty$ and the identity holds trivially. Otherwise let $p$ be the density of $P , q$ the density of $Q .$ , and $\tilde { p }$ be the density of ${ \tilde { P } } .$ . Since log q and log $\tilde { p }$ are quadratic polynomials and $P , \tilde { P }$ share their first two moments, $\mathbb { E } _ { P } [ \log q ] = \mathbb { E } _ { \tilde { P } } [ \log q ]$ and also $\mathbb { E } _ { P } [ \log \tilde { p } ] = \mathbb { E } _ { \tilde { P } } [ \log \tilde { p } ]$ , and all four quantities are finite. The integral R p log p lies in $( - \infty , \infty ]$ because the entropy of $P$ is bounded by that of ${ \tilde { P } } .$ If it equals $\infty ,$ both sides are infinite. Otherwise

$$
\mathrm { K L } ( P \| \tilde { P } ) + \mathrm { K L } ( \tilde { P } \| Q ) = \int p \log p - \mathbb { E } _ { P } [ \log \tilde { p } ] + \mathbb { E } _ { \tilde { P } } [ \log \tilde { p } ] - \mathbb { E } _ { \tilde { P } } [ \log q ] = \int p \log p - \mathbb { E } _ { P } [ \log q ] = \mathrm { K L } ( P \| Q )
$$

Remark A.4 (Why moments). When the number of stochastic parameters drops below the number of measurement points, the network output has no Lebesgue density. By Lemma A.3 both divergences are then infinite, and only the moment discrepancy can be controlled. This motivates an objective based on the first two moments. By Proposition 4.3, $\mathrm { M M D _ { 2 } ^ { 2 } }$ vanishes exactly when the first two moments agree, so $\mathcal { L } _ { 2 }$ is such an objective. For deeper networks the non-Gaussianity term can be substantial (Vladimirova et al., 2019; Yaida, 2020), and a characteristic kernel could target it.

Remark A.5 (Hierarchical targets). When the kernel hyperparameters of the target process carry hyperpriors, as in our experiments, the target Q is a mixture of Gaussians and not itself Gaussian. Proposition 4.3 holds for arbitrary distributions with finite second moments and is unafected. Lemma A.3 applies with $Q$ replaced by its Gaussian moment projection $\tilde { Q } = \mathcal { N } ( \boldsymbol { \mu } _ { Q } , \boldsymbol { \Sigma } _ { Q } )$ , which is exactly the Gaussian whose moments our objective matches.

Proposition 4.3 (The mmd reduced to moment matching). Let $k ( { \bf x } , { \bf y } ) = ( 1 + \langle { \bf x } , { \bf y } \rangle ) ^ { 2 }$ on $\mathbb { R } ^ { N }$ , and let $\begin{array} { r } { \mathbf { M } _ { P } : = \Sigma _ { P } + \mu _ { P } \pmb { \mu } _ { P } ^ { \top } } \end{array}$ denote the second-moment matrix. Then for any two distributions with finite second moments,

$$
\mathrm { M M D } _ { 2 } ^ { 2 } ( P , Q ) \ : = \ : 2 \| \pmb { \mu } _ { P } - \pmb { \mu } _ { Q } \| _ { 2 } ^ { 2 } \ : + \ : \| \mathbf { M } _ { P } - \mathbf { M } _ { Q } \| _ { F } ^ { 2 } .
$$

Here the subscript 2 of $\mathrm { M M D } _ { 2 } ^ { 2 } ( P , Q )$ denotes the use of the polynomial kernel of degree 2.

Proof. Expanding, $k ( \mathbf { x } , \mathbf { y } ) = 1 + 2 \langle \mathbf { x } , \mathbf { y } \rangle + \langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 }$ and $\langle \mathbf { x } , \mathbf { y } \rangle ^ { 2 } = \langle \mathbf { x } \mathbf { x } ^ { \top } , \mathbf { y } \mathbf { y } ^ { \top } \rangle _ { F }$ . Hence, $k ( \mathbf x , \mathbf y ) = \langle \varphi ( \mathbf x ) , \varphi ( \mathbf y ) \rangle$ with the feature map $\varphi ( \mathbf { x } ) = ( 1 , \sqrt { 2 } \mathbf { x } , \mathbf { x } \mathbf { x } ^ { \top } ) \in \mathbb { R } \times \mathbb { R } ^ { N } \times \mathbb { R } ^ { N \times N }$ , where the last component carries the Frobenius inner product. Since $\begin{array} { r } { \mathbb { E } _ { P } [ \| \varphi ( \dot { \mathbf { X } } ) \| ] = \mathbb { E } _ { P } [ \sqrt { k ( \mathbf { X } , \mathbf { X } ) } ] = \mathbb { E } _ { P } [ 1 + \| \mathbf { X } \| ^ { 2 } ] < \infty } \end{array}$ , the mean embedding $m _ { P } = \mathbb { E } _ { P } [ \varphi ( \mathbf { X } ) ] =$ $( 1 , \sqrt { 2 } \mu _ { P } , \mathbf { M } _ { P } )$ exists, and

$$
\mathrm { M M D } _ { 2 } ^ { 2 } ( P , Q ) = \| \mathbf { m } _ { P } - \mathbf { m } _ { Q } \| ^ { 2 } = 2 \| \pmb { \mu } _ { P } - \pmb { \mu } _ { Q } \| _ { 2 } ^ { 2 } + \| \mathbf { M } _ { P } - \mathbf { M } _ { Q } \| _ { F } ^ { 2 }
$$

Remark A.6 (What is optimized in practice). N is the number of measurement locations. In every step we draw X from $\mathcal { G } ^ { N }$ and compute the empirical moments from function samples. The loss of this step is $\frac { \mathrm { M M D } _ { 2 } ^ { 2 } ( \bar { P } _ { \mathbf { X } } , \hat { Q } _ { \mathbf { X } } ) } { 2 N }$ for the empirical measures $\hat { P } _ { \mathbf { X } }$ and $\hat { Q } _ { \bf X }$ , which is the biased mmd estimator scaled by $\scriptstyle { \frac { 1 } { 2 N } }$ . If the gp moments are computed exactly, replace $\hat { Q } _ { \bf X }$ by $Q \mathbf { x }$ . Since the locations are redrawn in every step, the optimized quantity is $\mathbb { E } _ { { \mathbf { X } } \sim { \mathcal { G } } ^ { N } } \left[ \frac { \mathrm { M M D } _ { 2 } ^ { 2 } ( { \hat { P } } _ { \mathbf { X } } , { \hat { Q } } _ { \mathbf { X } } ) } { 2 N } \right]$ , which is the empirical counterpart of $\frac { \mathcal { L } _ { 2 } } { 2 N }$

## A.2 Preserving expressiveness

For simplicity, we repeat the ucda theorem by (Sharma et al., 2023) again:

Theorem A.7 (Universal Conditional Distribution with Finite Stochasticity). Let X take values in a compact $\mathcal { X } \subset \mathbb { R } ^ { d _ { 0 } }$ and Y in $\mathcal { V } \subset \mathbb { R } ^ { d _ { L } }$ . Let ${ \pmb f } _ { v } ( { \pmb Z } , { \pmb x } )$ be a network with deterministic parameters υ and Gaussian noise $Z \in \mathbb { R } ^ { m }$ independent of X with finite mean and variance. Suppose that for some hidden layer $h _ { \ell } ,$ after restricting $z$ to a ball $\| Z \| \le R , ( 1 )$ Z and x are fully input by this layer, (2) $h _ { \ell }$ ranges over a compact set, (3) there is a deterministic, continuous, surjective g with $\begin{array} { r } { g ( h _ { \ell } ) = ( Z , x ) } \end{array}$ , and (4) the downstream network is a universal approximator in the sense of (Leshno et al., 1993). $I f Y | X$ admits a continuous generator, then $\mathbf { \Delta } f _ { v }$ approximates $\mathbf { \xi } _ { Y \mid X }$ arbitrarily well.

Proposition 4.7 (Deterministic certificate). Let $\mathcal { X } \subset \mathbb { R } ^ { d _ { 0 } }$ be compact and let f have $L - 1 \ge 2$ hidden layers with a continuous, strictly monotone, non-polynomial activation $\varphi .$ . Stochastic parameters may occur in any layer. $I f$

$$
U ( S ) \geq d _ { L } \quad a n d \quad | T | \geq d _ { 0 } ,
$$

then the masked network is a ucda. Both conditions are checkable in $\mathcal { O } ( | \Theta ^ { ( 1 ) } | )$ time.

Proof. Let $J _ { 0 } : = \{ j : b _ { i } ^ { ( 1 ) } \in S \}$ and fix $J \subseteq J _ { 0 }$ with $| J | = d _ { L }$ . Units in $T$ carry no stochastic parameter, so $J \cap T = \emptyset$ , and $d _ { 1 } \geq d _ { 0 } ^ { \bullet } + d _ { L }$ follows. Let $E : = \{ 1 , \dots , d _ { 1 } \} \setminus ( T \cup J )$

Step 1: reduced network Consider configurations in which every weight from a unit in $E$ into layer 2 has mean 0, and every stochastic parameter in layers $\geq 2$ has variance $v \geq 0$ . Denote the output by ${ \cal F } _ { v } ( { \bf x } )$ . For $v = 0$

the output depends on the first layer through the units in $T \cup J .$ For $j \in J$ decompose the pre-activation as $u _ { j } = \bar { W } _ { j , \cdot } x + \bar { b } _ { j } + \xi _ { j }$ , where

$$
\xi _ { j } = \sum _ { i : ( j , i ) \in S } \sigma _ { j , i } \varepsilon _ { j , i } x _ { i } + \sigma _ { j } ^ { b } \varepsilon _ { j } ^ { b } , \quad s _ { j } ( x ) ^ { 2 } : = \sum _ { i : ( j , i ) \in S } \sigma _ { j , i } ^ { 2 } x _ { i } ^ { 2 } + ( \sigma _ { j } ^ { b } ) ^ { 2 } \geq ( \sigma _ { j } ^ { b } ) ^ { 2 } > 0
$$

with independent standard Gaussian ε that are independent of X. Set $\begin{array} { r } { Z _ { j } : = \frac { \xi _ { j } } { s _ { i } ( \mathbf { X } ) } } \end{array}$ for $j \in J .$ . Conditionally on $\mathbf { X } = \mathbf { x }$ the vector $\mathbf { Z } _ { J } \mathrm { ~ i s ~ } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d _ { L } } )$ , and since this distribution does not depend on $\mathbf { x } , \mathbf { Z } _ { J }$ is independent of X. The first layer output on $T \cup J$ is $\mathbf { h } = \varphi ( \mathbf { u } )$ with $\mathbf { u } _ { T } = \bar { \mathbf { W } } _ { T , \cdot } \mathbf { x } + \bar { \mathbf { b } } _ { T }$ and $u _ { j } = \bar { \mathbf { W } } _ { j , . } \mathbf { x } + \bar { b } _ { j } + s _ { j } ( \mathbf { x } ) Z _ { j }$ , a continuous function of $( \mathbf { Z } _ { J } , \mathbf { x } )$ . We verify the conditions of Theorem A.7 for $\mathbf { F } _ { 0 }$ with $\ell = 1$ and $m = d _ { L } . \ ( 1 )$ Holds by construction, (2) h is a continuous image of the compact set $\{ \| \mathbf Z _ { J } \| \le R \} \times \mathcal { X } . \ ( 3 ) \ \mathrm { A }$ continuous strictly monotone $\varphi$ has a continuous inverse on the interval $\varphi ( \mathbb { R } )$ , so $\mathbf { u } = \varphi ^ { - 1 } ( \mathbf { h } )$ is recovered continuously. Since $| T | \geq d _ { 0 }$ and the deterministic parameters are free, we may choose their values such that $\mathrm { r a n k } ( \bar { \mathbf { W } } _ { T , \cdot } ) = d _ { 0 }$ . Now we recover $\mathbf { x } = \bar { \mathbf { W } } _ { T , \cdot } ^ { + } ( \mathbf { u } _ { T } - \bar { \mathbf { b } } _ { T } )$ , then $\begin{array} { r } { Z _ { j } = \frac { ( u _ { j } - \bar { \mathbf { W } } _ { j , \cdot } \mathbf { x } - \bar { b } _ { j } ) } { s _ { j } ( \mathbf { x } ) } } \end{array}$ for $j \in J$ , where $s _ { j }$ is continuous and bounded from below. This map g is continuous, recovers $( \mathbf { Z } _ { J } , \mathbf { x } )$ exactly and is surjective onto the restricted domain. (4) Since $L \geq 3 .$ , the network from layer 1 onward has at least one hidden layer with a continuous non-polynomial activation, and its weights out of $T \cup J$ are free. (Leshno et al., 1993) applies. Hence for every $\varepsilon > 0$ , there are parameter values with $\begin{array} { r } { \operatorname* { s u p } _ { \mathbf { x } } d _ { \mathrm { B L } } ( p \mathbf { F } _ { 0 } ( \mathbf { x } ) , p ( \mathbf { Y } | \mathbf { X } = \mathbf { x } ) ) \le \frac { \varepsilon } { 2 } } \end{array}$

Step 2: Downstream noise Write $\pmb { F } _ { v } ( \pmb { x } ) = \pmb { \Psi } ( \pmb { x } , Z , \sqrt { v } \pmb { \varepsilon } ^ { \prime } )$ , where Z collects all first-layer noise, $\varepsilon ^ { \prime }$ the standard Gaussian noise of the stochastic parameters in layers $\geq 2$ , and Ψ is jointly continuous. Then $\Psi ( \pmb { x } , Z , \mathbf { 0 } ) = \pmb { F } _ { 0 } ( \pmb { x } )$ and

$$
d _ { \mathrm { B L } } ( p \boldsymbol { F } _ { v } ( \boldsymbol { x } ) , p \boldsymbol { F } _ { 0 } ( \boldsymbol { x } ) ) \le \mathbb { E } [ \operatorname* { m i n } \{ 2 , \| \Psi ( \boldsymbol { x } , Z , \sqrt { v } \varepsilon ^ { \prime } ) - \Psi ( \boldsymbol { x } , Z , \mathbf { 0 } ) \| \} ] .
$$

Given $\delta > 0 _ { : }$ choose $R ^ { \prime }$ with $\mathbb { P } ( \| ( \mathbf { Z } , \pmb { \varepsilon } ^ { \prime } ) \| > R ^ { \prime } ) \le \delta$ . On the compact set ${ \mathcal { X } } \times \{ \| ( \mathbf { z } , \mathbf { e } ) \| \leq R ^ { \prime } \}$ , the map Ψ is uniformly continuous in x. Choosing $3 \delta \leq \textstyle { \frac { \varepsilon } { 2 } }$ and $v > 0$ accordingly, gives a nondegenerate Gaussian distribution on $\Theta _ { i }$ with $\begin{array} { r } { \operatorname* { s u p } _ { \mathbf { x } } d _ { \mathrm { B L } } ( p \mathbf { F } _ { v } ( \mathbf { x } ) , p ( \mathbf { Y } | \mathbf { X } = \mathbf { x } ) ) \dot { \mathbf { \xi } } \xi \mathbf { \xi } } \end{array}$

Complexity One pass over the bias mask gives $U ( S )$ , one pass over the weight mask gives T. Together this adds up to $| \Theta ^ { ( 1 ) } |$ □

Proposition A.8 (Make $\hat W _ { T } ,$ <sub>·</sub> full rank). For fixed values of the deterministic first-layer weights, the rank condition can be restored as follows. For $j \in \{ 1 , \dotsc , d _ { 1 } \}$ let $w _ { j } : = | S \cap \Theta _ { j } ^ { ( 1 ) } |$ |, where $\Theta _ { j } ^ { ( 1 ) }$ denotes the first-layer parameters of unit $j ,$ be the number of parameters of unit j that survived the prior optimization. Making unit j $f u l l y$ deterministic means setting $\sigma _ { \theta } = 0$ for all $\theta \in \dot { S } \cap \Theta _ { j } ^ { ( \mathrm { i } ) }$ changing $w _ { j }$ parameters.

Let $J \subseteq \{ 1 , \dotsc , d _ { 1 } \}$ be any set of units excluded from the repair and put $A : = \{ 1 , \dots , d _ { 1 } \} \ \backslash \ J$ . Suppose $\mathrm { r a n k } ( \bar { \pmb { W } } _ { A , \cdot } ^ { ( 1 ) } ) = d _ { 0 }$ . Then (i) the family $\mathcal { T } : = \{ B \subseteq A : \mathrm { r a n k } ( \bar { W } _ { B . } ^ { ( 1 ) } ) = | B | \}$ is the set of independent sets of a linear matroid on A, and every basis has exactly $d _ { 0 }$ elements; (ii) sorting A by $w _ { j }$ in increasing order and greedily adding a unit whenever it increases the rank returns a basis $B ^ { \star }$ of minimum total weight $\textstyle \sum _ { j \in B ^ { \star } } w _ { j } ; ( \operatorname { i i i } )$ making every unit of $B ^ { \star }$ fully deterministic yields a mask whose set $T ^ { \prime }$ of fully deterministic units contains $B ^ { \star }$ , so that $\mathrm { r a n k } ( \bar { \cal W } _ { T ^ { \prime } , \cdot } ^ { ( 1 ) } ) = d _ { 0 } ; \mathrm { ( i v ) }$ no repair that leaves the units of J untouched and restores the rank condition changes fewer parameters if no unit in J is fully deterministic. The cost is $\mathcal { O } ( d _ { 1 } \log d _ { 1 } + d _ { 1 } d _ { 0 } ^ { 2 } )$

Proof. (i) Linear independence of rows of a fixed matrix defines a linear (matric) matroid on the row index set with $\mathrm { r a n k } ( \bar { \pmb { W } } _ { A , \cdot } ^ { ( 1 ) } ) = d _ { 0 }$ , so all bases have $d _ { 0 }$ elements. (ii) The weight $\textstyle B \mapsto \sum _ { j \in B } w _ { j }$ is nonnegative and modular, and for such weights the greedy algorithm returns a minimum-weight basis of a matroid (Welsh, 1970). (iii) Every parameter of a unit in $B ^ { \star }$ has $\sigma _ { \theta } = 0$ after the repair, so $B ^ { \star } \subseteq T ^ { \prime }$ , and the rank of a submatrix does not decrease when rows are added. (iv) Any repaired mask satisfying the certificate has some $T ^ { \prime }$ with rank $\bar { \pmb { W } } _ { T ^ { \prime } , \cdot } ^ { ( 1 ) } ) = d _ { 0 }$ , hence $T ^ { \prime } \cap A$ contains a basis $\hat { B ^ { \prime } }$ of the matroid, and every surviving parameter of every unit of $B ^ { \prime }$ must have been set to zero, which costs $\begin{array} { r } { \sum _ { j \in B ^ { \prime } } w _ { j } \geq \sum _ { j \in B ^ { \star } } w _ { j } } \end{array}$ . The complexity of it is one sort plus $d _ { 1 }$ rank-one updates of QR factorization of size $d _ { 0 }$ □

Corollary A.9 (Minimal Repair). Suppose $d _ { 1 } \geq d _ { 0 } + d _ { L }$ . Let $J _ { 0 } : = \{ j : b _ { j } ^ { ( 1 ) } \in S \}$ , so that $U ( S ) = | J _ { 0 } |$ , and for every first-layer unit j let $w _ { j } : = | S \cap \Theta _ { j } ^ { ( 1 ) } |$ be the number of its stochastic parameters.

Step 1 (coverage). $I f U ( S ) < d _ { L }$ , choose $J \supseteq J _ { 0 }$ with $| J | = d _ { L }$ and make the biases of $J \backslash J _ { 0 }$ stochastic. Otherwise choose $J \subseteq J _ { 0 }$ with $| J | = d _ { L }$ . This changes max $\{ 0 , d _ { L } - U ( S ) \}$ parameters and gives $U ( S ) \geq d _ { L }$ , and no repair achieving $U ( S ) \geq d _ { L }$ changes fewer.

Step 2 (deterministic units). Let T be the set of fully deterministic units after Step 1 and $A : = \{ 1 , \dots , d _ { 1 } \} \setminus J$ . If $| T | < d _ { 0 }$ , let $B ^ { \star }$ consist of the $d _ { 0 } - | T |$ units of $A \backslash T$ with the smallest $w _ { j }$ , and make all first-layer parameters of these units deterministic. This changes $\textstyle \sum _ { j \in B ^ { \star } } w _ { j }$ parameters, and no repair that leaves J untouched and achieves $| T ^ { \prime } | \geq d _ { 0 }$ changes fewer, where $T ^ { \prime } : = \overline { { T \cup B ^ { \star } } }$

The resulting mask satisfies the certificate of Proposition 4.7. Step 2 does not undo Step 1 because $B ^ { \star } \cap J = \varnothing$ and Step 1 does not undo Step 2 because the units of J have stochastic biases and are therefore not in $T ^ { \prime }$

Proof. Step 1 makes one bias stochastic per changed parameter, which gives $U ( S ) \geq d _ { L }$ afterwards, and any repair achieving this must change at least $d _ { L } - U ( S )$ biases. For Step 2, the units of J have stochastic biases, so $T \subseteq A$ and $| A \setminus T | = | A | - | T | \geq d _ { 0 } - | T |$ by $d _ { 1 } \geq d _ { 0 } + d _ { L }$ . Hence $B ^ { \star }$ exists. Any repair that leaves J untouched and achieves $| T ^ { \prime } | \geq d _ { 0 }$ must make at least $d _ { 0 } - | T |$ units of $A \backslash T$ fully deterministic, and doing so for a unit j changes exactly $w _ { j }$ parameters. The sum of w over $d _ { 0 } - \vert T \vert$ units is smallest for the units with the smallest $w _ { j } ,$ which are those in $B ^ { \star }$ . After both steps $U ( S ) \geq d _ { L }$ and $| T ^ { \prime } | \geq d _ { 0 }$ , so Proposition 4.7 applies. □

## A.3 Inference

Assumption A.10 (Regularity). (i) For every $\theta _ { s }$ , the map $\pmb { \theta } _ { d } \mapsto \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } )$ is continuously diferentiable. (ii) There are $r \geq 0$ and $G \in L ^ { \mathrm { i } } ( p ( \pmb { \theta } _ { s } ) )$ , with $\lVert \nabla _ { \pmb { \theta } _ { d } } \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) \rVert \leq G ( \pmb { \theta } _ { s } ) ( 1 + \lVert \pmb { \theta } _ { d } \rVert ) ^ { r }$ for all $\begin{array} { r } { \pmb { \theta } _ { d } , \pmb { \theta } _ { s } . \ ( \mathrm { i i i } ) \ B : = \operatorname* { s u p } _ { \pmb { \theta } _ { d } , \pmb { \theta } _ { s } } e ^ { \dot { \ell } ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) } < } \end{array}$ ∞ and $Z ( \pmb { \theta } _ { d } ) > 0$ for all $\theta _ { d }$

Remark A.11. (iii) holds for categorical likelihoods and for Gaussian likelihoods with noise variance bounded from below. (ii) holds whenever the gradient grows polynomially in $\theta _ { s } ,$ since $p ( \pmb { \theta } _ { s } )$ is Gaussian. Under (ii) and (iii), dominated convergence shows that $Z$ is $C ^ { 1 }$ and that $\begin{array} { r } { \pmb { \theta } _ { d } \mapsto \int \varphi ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) e ^ { \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) } p ( \pmb { \theta } _ { s } ) d \pmb { \theta } _ { s } } \end{array}$ is continuous for every bounded continuous $\varphi .$

Let Π denote the joint posterior of $( \theta _ { d } , \theta _ { s } )$ under the prior $\mathcal { N } ( \hat { \pmb { \mu } } _ { d } , \tau ^ { 2 } { \pmb { I } } _ { k } ) \otimes p ( \pmb { \theta } _ { s } )$ , so that $\Pi _ { \tau } ( \mathrm { d } \pmb { \theta } _ { d } , \mathrm { d } \pmb { \theta } _ { s } )$ ∝ $e ^ { \ell \left( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } \right) } \mathcal { N } ( \pmb { \theta } _ { d } ; \hat { \pmb { \mu } } _ { d } , \tau ^ { 2 } \pmb { I } _ { k } ) p ( \pmb { \theta } _ { s } ) \mathrm { d } \pmb { \theta } _ { d } \mathrm { d } \pmb { \theta } _ { s }$

Proposition A.12 (τ interpolates between freezing and type-II ML). Under Assumption A.10 the joint posterior Π<sub>τ</sub> has a Lebesgue density for every $\tau > 0$ . Moreover, (i) $a s \tau \downarrow 0 , \Pi ,$ converges weakly to $\delta _ { \hat { \pmb { \mu } } _ { d } } \otimes p ( \pmb { \theta } _ { s } | \mathcal { D } , \hat { \pmb { \mu } } _ { d } ) .$ (ii) for $0 < \tau < \infty , J _ { \tau }$ is, up to a constant, the log marginal posterior density of $\theta _ { d } ;$ (iii) as $\tau \uparrow \infty , J _ { \tau }  \log Z$ uniformly on compacts. Keeping the degenerate prior $\delta _ { \hat { \pmb { \mu } } _ { d } }$ while optimizing $\theta _ { d }$ is ill-posed, since the log-prior is $- \infty$ of the atom.

Proof. The joint density $\frac { e ^ { \ell ( \pmb { \theta } _ { s } , \pmb { \theta } _ { d } ) } p ( \pmb { \theta } _ { s } ) \mathcal { N } ( \pmb { \theta } _ { d } ; \hat { \pmb { \mu } } _ { d } , \tau ^ { 2 } \pmb { I } ) } { n ( \mathcal { D } ) }$ is integrable by regularity assumption (iii). For (i) let $\varphi$ be bounded and continuous and set $\begin{array} { r } { \Phi ( \pmb { \theta } _ { d } ) : = \int \varphi ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) e ^ { \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) } p ( \pmb { \theta } _ { s } ) d \pmb { \theta } _ { s } } \end{array}$ <sub>s</sub>. Φ and $Z$ are bounded by $\| \varphi \| _ { \infty } B$ and B and continuous by the Remark A.11. Since $\mathcal { N } ( \hat { \pmb { \mu } } _ { d } , \tau ^ { 2 } { \pmb { I } } )  \delta _ { \hat { \pmb { \mu } } _ { d } }$

$$
\mathbb { E } _ { \Pi _ { \tau } \varphi } = \frac { \int \Phi d \mathcal { N } ( \hat { \mu } _ { d } , \tau ^ { 2 } I ) } { \int Z d \mathcal { N } ( \hat { \mu } _ { d } , \tau ^ { 2 } I ) }  \frac { \Phi ( \hat { \mu } _ { d } ) } { Z ( \hat { \mu } _ { d } ) } = \mathbb { E } _ { p ( \pmb { \theta } _ { s } | \mathcal { D } , \hat { \mu } _ { d } ) } [ \varphi ( \hat { \mu } _ { d } , \pmb { \theta } _ { s } ) ] ,
$$

using $Z ( \hat { \mu } _ { d } ) > 0$ . (ii) follows by writing out log $\boldsymbol { Z } ( \pmb { \theta } _ { d } ) + \log \mathcal { N } ( \pmb { \theta } _ { d } ; \hat { \pmb { \mu } } _ { d } , \tau ^ { 2 } \pmb { I } )$ . For (iii), $\begin{array} { r } { \operatorname* { s u p } _ { \| \pmb { \theta } _ { d } \| < \rho } \frac { \| \pmb { \theta } _ { d } - \hat { \pmb { \mu } } _ { d } \| ^ { 2 } } { 2 \tau ^ { 2 } }  0 } \end{array}$ .

Theorem 4.9 (The near-degenerate approximation is a type-II map estimate). Let $\mathcal { P } _ { G }$ be the set of densities $q _ { s }$ with finite entropy and $\mathbb { E } _ { q _ { s } } [ G + | \log p ( \pmb { \theta } _ { s } ) | + | \ell ( \mathbf { 0 } , \pmb { \theta } _ { s } ) | ] < \infty$ , where G is the envelope from Assumption A.10 (ii), and let $q _ { \varepsilon } : = \mathcal { N } ( \pmb { \theta } _ { d } , \varepsilon ^ { 2 } \pmb { I } _ { k } )$ . Under Assumption A.10, for all $\pmb { \theta } _ { d } \in \mathbb { R } ^ { k }$ and $q _ { s } \in \mathcal { P } _ { G }$

$$
\begin{array} { r } { \mathrm { K L } ( q _ { \varepsilon } \otimes q _ { s } \| \Pi _ { \tau } ) = - \displaystyle \frac { k } { 2 } \log ( 2 \pi e \varepsilon ^ { 2 } ) + F ( \pmb \theta _ { d } , q _ { s } ) + R _ { \varepsilon } } \\ { F ( \pmb \theta _ { d } , q _ { s } ) = \mathrm { K L } ( q _ { s } \| p ( \cdot | \mathcal D , \pmb \theta _ { d } ) ) - J _ { \tau } ( \pmb \theta _ { d } ) + c , } \end{array}
$$

with $c \in \mathbb { R }$ being a constant, $\begin{array} { r } { | R _ { \varepsilon } | \leq C ( \pmb { \theta } _ { d } ) \mathbb { E } _ { q _ { s } } [ G ] \varepsilon + \frac { k \varepsilon ^ { 2 } } { 2 \tau ^ { 2 } } \ f o r \ \varepsilon \leq 1 } \end{array}$ and $C ( \pmb { \theta } _ { d } )$ locally bounded. The minimizers of F are $q _ { s } ^ { \star } = p ( \cdot | \mathcal { D } , \pmb { \theta } _ { d } ^ { \star } )$ with ${ \pmb { \theta } } _ { d } ^ { \star } \in$ arg max $J _ { \tau }$ .

Proof. By definition of the kl divergence,

$$
\mathrm { K L } ( q _ { \varepsilon } \otimes q _ { s } \| \Pi _ { \tau } ) = - H ( q _ { \varepsilon } ) - H ( q _ { s } ) - \mathbb { E } _ { q _ { s } \otimes q _ { \varepsilon } } [ \ell ( \theta _ { d } ^ { \prime } , \theta _ { s } ) + \log p ( \theta _ { d } ^ { \prime } ) + \log p ( \theta _ { s } ) ] + \log p ( \mathcal { D } ) ,
$$

and $\begin{array} { r } { H ( q _ { \varepsilon } ) = \frac { k } { \gamma } \log ( 2 \pi e \varepsilon ^ { 2 } ) } \end{array}$ . Exactly $\begin{array} { r } { \mathbb { E } _ { q _ { \varepsilon } } [ \log p ( \pmb { \theta } _ { d } ^ { \prime } ) ] = \log p ( \pmb { \theta } _ { d } ) - \frac { k \varepsilon ^ { 2 } } { 2 \tau ^ { 2 } } } \end{array}$ . By the mean value theorem and (ii) of Assumption A.10, $| \ell ( \pmb { \theta } _ { d } ^ { \prime } , \pmb { \theta } _ { s } ) - \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) | \leq G ( \pmb { \theta } s ) ( 1 + \| \pmb { \theta } _ { d } \| + \| \pmb { \theta } _ { d } ^ { \prime } - \mathbf { \tilde { \theta } } _ { d } \| ) ^ { r } \| \pmb { \theta } _ { d } ^ { \prime } - \pmb { \theta } _ { d } \|$ , so

$$
\begin{array} { r } { | \mathbb { E } _ { q _ { \varepsilon } \otimes q _ { s } } [ \ell ( \theta _ { d } ^ { \prime } , \pmb { \theta } _ { s } ) ] - \mathbb { E } _ { q _ { s } } [ \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) ] | \le \mathbb { E } _ { q _ { s } } [ G ] \mathbb { E } _ { W \sim \mathcal { N } ( \mathbf { 0 } , I _ { k } ) } [ ( 1 + \| \pmb { \theta } _ { d } \| + \| W \| ) ^ { r } \| W \| ] \varepsilon , } \end{array}
$$

which defines $C ( \pmb { \theta } _ { d } )$ . Using $\begin{array} { r } { p ( \pmb { \theta } _ { s } | \mathcal { D } , \pmb { \theta } _ { d } ) = \frac { e ^ { \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) } p ( \pmb { \theta } _ { s } ) } { Z ( \pmb { \theta } _ { d } ) } } \end{array}$ ，

$$
\mathrm { K L } ( q _ { s } \| p ( \cdot , | \mathcal { D } , \pmb { \theta } _ { d } ) ) = - H ( q _ { s } ) - \mathbb { E } _ { q _ { s } } [ \ell ( \pmb { \theta } _ { d } , \pmb { \theta } _ { s } ) ] - \mathbb { E } _ { q _ { s } } [ \log p ( \pmb { \theta } _ { s } ) ] + \log Z ( \pmb { \theta } _ { d } ) ,
$$

and substituting gives the stated form of F. The kl term is nonnegative and vanishes if and only if $q _ { s } = p ( \cdot | \mathcal { D } , \pmb { \theta } _ { d } )$ which lies in $\mathcal { P } _ { G }$ by (iii). It is the only term that depends on $q _ { s }$ , so minimizing $F$ reduces to maximizing $J _ { \tau }$ .

Theorem A.13 (Tracking bound under constant step sizes). Assume for all $\theta _ { d } , \theta _ { d } ^ { \prime } \in K$ (C1) $W _ { 1 } ( \mu P _ { \theta _ { d } } ^ { h } , \nu P _ { \theta _ { d } } ^ { h } ) \leq$ $\rho W _ { 1 } ( \mu , \nu )$ with $\rho < 1 , \mathrm { ( C 2 ) } W _ { 1 } ( \pi _ { \theta _ { d } } ^ { h } , \pi _ { \theta _ { r } ^ { h } } ^ { h } ) \leq L _ { \pi } \lVert \theta _ { d } - \theta _ { d } ^ { \prime } \rVert , \mathrm { ( C 3 ) } \lVert \theta _ { d , t + 1 } - \theta _ { d , t } \rVert \leq \delta _ { t } , \mathrm { ( C 4 ) } W _ { 1 } ( \pi _ { \theta _ { d } } ^ { h } , \pi _ { \theta _ { d } } ^ { - } ) \leq C _ { 1 } h$ for the s-marginals. Let $\boldsymbol { X } _ { t } = ( \boldsymbol { \theta } _ { s , t } , \boldsymbol { v } _ { t } )$ denote the state of the sampler after t steps, with momentum $v _ { t } .$ . Let $\mu _ { t }$ be the distribution of $X _ { t }$ and $e _ { t } : = W _ { 1 } ( \mu _ { t } , \pi _ { \theta _ { d . t } } ^ { h } )$ . Then

$$
e _ { t + 1 } \leq \rho e _ { t } + L _ { \pi } \delta _ { t } , \quad e _ { t } \leq \rho ^ { t } e _ { 0 } + L _ { \pi } \sum _ { j < t } \rho ^ { t - 1 - j } \delta _ { j } ,
$$

$$
W _ { 1 } ( \pmb { \mu } _ { t } ^ { \pmb { \theta } _ { s } } , \pi _ { \pmb { \theta } _ { d _ { t } } } ) \leq e _ { t } + C _ { 1 } h
$$

Proof. Since $\pi _ { \theta _ { d , t } } ^ { h }$ is invariant for $P _ { \theta _ { d , t } } ^ { h }$ and $\mu _ { t + 1 } = \mu _ { t } P _ { \theta _ { d , t } } ^ { h }$ for an open-loop path,

$$
e _ { t + 1 } \le W _ { 1 } ( \mu _ { t } P _ { \theta _ { d , t } } ^ { h } , \pi _ { \theta _ { d , t } } ^ { h } P _ { \theta _ { d , t } } ^ { h } ) + W _ { 1 } ( \pi _ { \theta _ { d , t } } ^ { h } , \pi _ { \theta _ { t + 1 } } ^ { h } ) \le \rho e _ { t } + L _ { \pi } \delta _ { t }
$$

Iterating gives the second bound. The third follows from the triangle inequality, marginalization, and (C4). Corollary 4.10 (Coupled step sizes yield a non-vanishing tracking error). Assume in addition to the setting of Theorem A.13 that $\rho ( h ) = 1 - c h + o ( h )$ with $c > 0$ and $\delta _ { t } \leq G \eta$ for all t. Then for all t

$$
e _ { t } \leq \rho ( h ) ^ { t } e _ { 0 } + \frac { L _ { \pi } G \eta } { c h + o ( h ) }
$$

(a) Coupled. $h = \eta$ . The stationarity part of the bound tends to $\frac { L _ { \pi } G } { c }$ as $\eta  0$ . (b) Decoupled. $I f h  0$ and $\begin{array} { r } { \frac { \eta } { h }  0 } \end{array}$ , then $\mathrm { s u p } _ { t } [ W _ { 1 } ( \pmb { \mu } _ { t } ^ { \pmb { \theta } _ { s } } , \pi _ { \pmb { \theta } _ { d _ { t } } } ) - \rho ( h ) ^ { t } e _ { 0 } ]  0$

Proof. Insert $\delta _ { j } \leq G \eta$ into Theorem A.13 and use $\begin{array} { r } { \sum _ { j } \rho ^ { j } \le \frac { 1 } { 1 - \rho } } \end{array}$ . For (b) add $C _ { 1 } h$ from (C4).

This result gives an upper bound for the error, the following proposition shows that this bound is also attained: Proposition A.14 (The coupled bound is attained). Let $\pi _ { \theta _ { d } } = \mathcal { N } ( \theta _ { d } , 1 )$ on R, let $P _ { \theta _ { d } } ^ { h }$ be the unadjusted Langevin kernel $x \mapsto x - h ( x - \theta _ { d } ) + \sqrt { 2 h } W$ with $h \in ( 0 , 1 )$ , let $X _ { 0 } \sim \pi _ { \theta _ { d , 0 } } ^ { h }$ and let $\theta _ { d , t } = \theta _ { d , 0 } + G \eta t$ . Then (C1) to (C3) hold with $\rho = 1 - h , c = 1 , L _ { \pi } = 1 , \delta _ { t } = G \eta$ , and

$$
W _ { 1 } ( \mu _ { t } , \pi _ { \theta _ { d , t } } ) \geq W _ { 1 } ( \mu _ { t } , \pi _ { \theta _ { d , t } } ^ { h } ) = \frac { G \eta } { h } ( 1 - ( 1 - h ) ^ { t } ) .
$$

In the coupled case $h = \eta$ this is at least $G ( 1 - e ^ { - 1 } )$ for all $\begin{array} { r } { t \geq \frac { 1 } { \eta } . } \end{array}$ , independently of η.

Proof. Synchronous coupling gives (C1) with $\rho = 1 - h$ . The variance recursion $v ^ { \prime } = ( 1 - h ) ^ { 2 } v + 2 h$ has fixed point $\begin{array} { r } { v _ { h } = \frac { 1 } { 1 - \frac { h } { 2 } } } \end{array}$ , so $\pi _ { \theta _ { d } } ^ { h } = \mathcal N ( \theta _ { d } , v _ { h } )$ and (C2) holds with $L _ { \pi } = 1$ . Since $X _ { 0 } \sim \pi _ { \theta _ { d , 0 } } ^ { h }$ and the variance recursion does not involve $\theta _ { d } , \mathcal { N } ( m _ { t } , v _ { t } )$ with $m _ { t + 1 } = ( 1 - h ) m _ { t } + h \theta _ { d , t }$ . The lag $\ell _ { t } : = \ell _ { d , t } - m _ { t }$ satisfies $\ell _ { 0 } = 0$ and $\ell _ { t + 1 } = ( 1 - h ) \ell _ { t } + G \eta$ , hence $\begin{array} { r } { \ell _ { t } = \frac { G \eta } { h } ( 1 - ( 1 - h ) ^ { t } ) } \end{array}$ . Two Gaussians with equal variance and means difering by $\ell _ { t }$ are at W -distance $\ell _ { t } ,$ and $W _ { 1 } ( \mu _ { t } , \pi _ { \theta _ { d , t } } ) \geq | \mathbb { E } [ X _ { t } ] - \theta _ { d , t } | = \ell _ { t }$ . Finally, $\left( 1 - \eta \right) ^ { \frac { 1 } { \eta } } \leq e ^ { - 1 }$ □

We also manage to find an error decomposition, where each term relates to a specific modeling choice. This allows us to better understand how to make the modeling choices, if we want to reduce the error.

Corollary A.15 (Error decomposition). In the setting of Theorem A.13, let $\phi ~ s a t i s f y ~ \| \phi \| _ { \infty } \leq 1$ and $\mathrm { L i p } ( \phi ) \leq 1$ let $T _ { 0 }$ be the burn-in, $\begin{array} { r } { \hat { A } : = \frac { 1 } { T } \sum _ { t = T _ { 0 } + 1 } ^ { T _ { 0 } + T } \phi ( \mathbf { X } _ { t } ^ { s } ) , \bar { \delta } : = \operatorname* { s u p } _ { t } \delta _ { t } } \end{array}$ , and $M : = \operatorname* { s u p } _ { t } \mathbb { E } [ \| \mathbf { X } _ { t } - \mathbf { X } _ { t } ^ { \prime } \| ] < \infty$ with $\mathbf { X } _ { t } ^ { \prime }$ an independent copy of $\mathbf { X } _ { t }$ . Then

$$
\Big ( \mathbb { E } [ ( \hat { A } - \mathbb { E } _ { \pi _ { \theta _ { d } ^ { \star } } } [ \phi ] ) ^ { 2 } ] \Big ) ^ { \frac { 1 } { 2 } } \le \underbrace { C _ { 1 } h } _ { ( a ) \ D i s c r e t i z a t i o n } + \underbrace { \frac { L _ { \pi } \bar { \delta } } { 1 - \rho } } _ { ( b ) \ T r a c k i n g }
$$

$$
+ \underbrace { \frac { L _ { \pi } } { T } \sum _ { t = T _ { 0 } + 1 } ^ { T _ { 0 } + T } \| \pmb { \theta } _ { d _ { t } } - \pmb { \theta } _ { d } ^ { \star } \| } _ { ( c ) \mathrm { ~ } O p t i m i z e r } + \underbrace { \frac { \rho ^ { T _ { 0 } } e _ { 0 } } { T ( 1 - \rho ) } } _ { ( d ) \mathrm { ~ } B u r n - i n } + \underbrace { \sqrt { \frac { 2 \operatorname* { m a x } \{ 1 , M \} } { T ( 1 - \rho ) } } } _ { ( e ) \mathrm { ~ } M o n t e \mathrm { ~ } C a r l o }
$$

Proof. Bias. Decompose

$$
\mathbb { E } [ \hat { A } ] - \pi _ { \pmb { \theta } ^ { \star } } \phi = \frac { 1 } { T } \sum _ { t } ( \mu _ { t } \phi - \pi _ { \pmb { \theta } _ { d , t } } ^ { h } \phi ) + \frac { 1 } { T } \sum _ { t } ( \pi _ { \pmb { \theta } _ { d , t } } ^ { h } \phi - \pi _ { \pmb { \theta } ^ { \star } } ^ { h } \phi ) + ( \pi _ { \pmb { \theta } ^ { \star } } ^ { h } \phi - \pi _ { \pmb { \theta } ^ { \star } } \phi )
$$

Since $\phi$ is 1-Lipschitz, the first sum is at most $\begin{array} { r } { \frac { 1 } { T } \sum _ { t } e _ { t } \le \frac { \rho ^ { T _ { 0 } } e _ { 0 } } { T ( 1 - \rho ) } + \frac { L _ { \pi } \bar { \delta } } { 1 - \rho } } \end{array}$ by Theorem A.13, the second is at most term (c) by (C2), and the third is at most $C _ { 1 } h$ by (C4).

Variance. For $u < t$ let $g : = \phi \Pi _ { i = u } ^ { t - 1 } { \cal P } _ { \theta _ { i } } ^ { h }$ . By (C1) applied to Dirac measures, $\mathrm { L i p } ( g ) \leq \rho ^ { t - u }$ . Hence,

$$
| \mathrm { C o v } ( \phi ( X _ { u } ) , \phi ( X _ { t } ) ) | = | \mathrm { C o v } ( \phi ( X _ { u } ) , g ( X _ { u } ) ) | \le \mathbb { E } [ | g ( X _ { u } ) - \mathbb { E } [ g ( X _ { u } ) ] | ] \le \rho ^ { t - u } M ,
$$

and trivially $| \mathrm { C o v } | \le 1$ . Therefore, $\begin{array} { r } { \mathrm { V a r } ( \hat { A } ) \leq \frac { \operatorname* { m a x } \{ 1 , M \} } { T ^ { 2 } } \sum _ { u , t } \rho ^ { | t - u | } \leq \frac { 2 \operatorname* { m a x } \{ 1 , M \} } { T ( 1 - \rho ) } } \end{array}$ . Minkowski’s inequality combines bias and standard deviation. □

Corollary A.16 (Drift radius). Every stationary point of J<sub>τ</sub> satisfies $\pmb { \theta } _ { d } ^ { \star } - \hat { \pmb { \mu } } _ { d } = \tau ^ { 2 } \nabla$ log $Z ( \theta _ { d } ^ { \star } )$ . If, in addition ∇ log Z is $L _ { g } – L i p s c h i t z$ and $\tau ^ { 2 } L _ { g } < 1$ , then $J _ { \tau }$ is strongly concave, ${ \pmb { \theta } } _ { d } ^ { \star }$ is unique, and with $g _ { 0 } : = \nabla$ log $Z ( \hat { \pmb { \mu } } _ { d } )$

$$
\frac { \tau ^ { 2 } \lVert \pmb { g } _ { 0 } \rVert } { 1 + \tau ^ { 2 } L _ { g } } \leq \lVert \pmb { \theta } _ { d } ^ { \star } - \hat { \pmb { \mu } } _ { d } \rVert \leq \frac { \tau ^ { 2 } \lVert \pmb { g } _ { 0 } \rVert } { 1 - \tau ^ { 2 } L _ { g } } ,
$$

so the drift equals $\tau ^ { 2 } \lVert \mathbf { \boldsymbol { g } } _ { 0 } \rVert ( 1 + \mathcal { O } ( \tau ^ { 2 } ) )$ . By Fisher’s identity, $\pmb { g } _ { 0 }$ is estimable from the frozen run of regime (i) at no extra cost.

Proof. The identity is the stationary condition ∇ log $Z ( \pmb { \theta } _ { d } ^ { \star } ) - \tau ^ { - 2 } ( \pmb { \theta } _ { d } ^ { \star } - \hat { \pmb { \mu } } _ { d } ) = 0$ . An $L _ { g } .$ -Lipschitz gradient makes $\frac { L _ { g } } { \gamma } \Vert \pmb { \theta } \Vert ^ { 2 } - \log Z$ convex, so $J _ { \tau }$ is strongly concave with modulus $\tau ^ { - 2 } - L _ { g } > 0$ . From the identity, $\lVert { \pmb { \theta } } _ { d } ^ { \star } - \hat { { \pmb { \mu } } _ { d } } \rVert = \tau ^ { 2 } \lVert \nabla$ log $Z ( \pmb { \theta } _ { d } ^ { \star } ) \|$ and $\| \nabla$ log $Z ( \pmb { \theta } _ { d } ^ { \star } ) \| - \| \pmb { g } _ { 0 } \| | \leq L _ { g } \| \pmb { \theta } _ { d } ^ { \star } - \hat { \pmb { \mu } } _ { d } \| ,$ ; rearrange. □

## B DETAILED UCI RESULTS

First we are going to provide some information on the datasets. The licenses of the datasets can be found on their respective websites.

<table><tr><td>Dataset</td><td> $( d _ { 0 } , d _ { L } )$ </td><td>Sample size</td><td>Parameters</td><td>Citation</td></tr><tr><td>CONCRETE</td><td>(8, 1)</td><td>1030</td><td>11101</td><td>(Yeh, 1998)</td></tr><tr><td>BOSTON</td><td>(13, 1)</td><td>506</td><td>11601</td><td>(Harrison and Rubinfeld, 1978)</td></tr><tr><td>BIKESHARING</td><td>(12, 1)</td><td>731</td><td>11501</td><td>(Fanaee-T, 2013)</td></tr><tr><td>WINE</td><td>(10, 1)</td><td>4898</td><td>11301</td><td>(Cortez et al., 2009)</td></tr><tr><td>ONLINESHOPPERS</td><td>(17, 2)</td><td>12330</td><td>12102</td><td>(Sakar and Kastro, 2018)</td></tr><tr><td>CREDIT</td><td>(23, 2)</td><td>30000</td><td>12702</td><td>(Yeh, 2009)</td></tr><tr><td>BANKNOTE</td><td>(4, 2)</td><td>1372</td><td>10802</td><td>(Lohweg, 2012)</td></tr><tr><td>HTRU2</td><td>(8, 2)</td><td>17898</td><td>11202</td><td>(Lyon, 2015)</td></tr></table>

Table 5: Dataset characteristics.

Table 6: Results on BANKNOTE $( \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td></td><td>ACC in % (↑)</td><td>NLL (↓)</td><td>ECE (↓)</td><td>Deterministic parameters in %</td></tr><tr><td>DWF-PSNN</td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 2$ </td><td> $5 5 . 3 2 \pm 2 . 5 6$ </td></tr><tr><td>ALS-BNN</td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $\mathbf { 0 . 0 0 4 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 0 4 \ : \pm { \ : 0 . 0 0 0 } }$ </td><td>0.00</td></tr><tr><td>FLS-BNN</td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $\mathbf { 0 . 0 0 4 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 0 4 \ : \pm { \ : 0 . 0 0 1 } }$ </td><td>95.37</td></tr><tr><td>RAND-PSNN</td><td> $\mathbf { 1 0 0 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $5 5 . 3 2 \pm 2 . 5 6$ </td></tr><tr><td>SNI-PSNN</td><td> $9 9 . 2 0 \pm 0 . 6 3$ </td><td> $0 . 0 3 2 \pm 0 . 0 1 0$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 7$ </td><td>90.00</td></tr></table>

Table 7: Results on HTRU2 (mean ± std).
<table><tr><td></td><td>ACC in % (↑)</td><td>NLL (↓)</td><td>ECE (↓)</td><td>Deterministic parameters in %</td></tr><tr><td>DWF-PSNN</td><td> $\mathbf { 9 7 . 8 3 \ : \pm { \ : 0 . 4 8 } }$ </td><td> $\mathbf { 0 . 0 6 9 \ : \pm { \ : 0 . 0 1 6 } }$ </td><td> $\mathbf { 0 . 0 0 7 \ : \pm { \ : 0 . 0 0 3 } }$ </td><td> $5 7 . 3 1 \pm 2 . 7 9$ </td></tr><tr><td>ALS-BNN</td><td> $9 7 . 5 9 \pm 0 . 5 0$ </td><td> $0 . 0 7 8 \pm 0 . 0 1 6$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 3$ </td><td>0.00</td></tr><tr><td>FLS-BNN</td><td> $9 7 . 5 6 \pm 0 . 5 6$ </td><td> $0 . 0 7 9 \pm 0 . 0 1 5$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 3$ </td><td>91.97</td></tr><tr><td>RAND-PSNN</td><td> $9 7 . 8 2 \pm 0 . 4 9$ </td><td> $0 . 0 7 0 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 0 0 7 \ : \pm { \ : 0 . 0 0 3 } }$ </td><td> $5 7 . 3 1 \pm 2 . 7 9$ </td></tr><tr><td>SNI-PSNN</td><td> $9 7 . 4 1 \pm 0 . 5 8$ </td><td> $0 . 0 8 6 \pm 0 . 0 1 6$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 2$ </td><td>90.00</td></tr></table>

Table 8: Results on ONLINESHOPPERS (mean ± std).
<table><tr><td></td><td> $\operatorname { A C C } \mathrm { i n } \ \% ( \uparrow )$ </td><td> $\mathrm { { N L L } \left( \downarrow \right) }$ </td><td> $\mathrm { E C E } \ ( \downarrow )$ </td><td>Deterministic parameters in %</td></tr><tr><td>DWF-PSNN</td><td> $8 9 . 4 3 \pm 0 . 6 4$ </td><td> $\mathbf { 0 . 2 4 3 \ : \pm { \ : 0 . 0 2 0 } }$ </td><td> $0 . 0 1 6 \pm 0 . 0 0 5$ </td><td> $6 1 . 0 1 \pm 3 . 4 3$ </td></tr><tr><td>ALS-BNN</td><td> $8 9 . 2 5 \pm \ : 0 . 7 4$ </td><td> $0 . 2 6 1 \pm 0 . 0 1 9$ </td><td> $0 . 0 2 4 \pm 0 . 0 0 9$ </td><td>0.00</td></tr><tr><td>FLS-BNN</td><td> $8 9 . 2 5 \pm 0 . 8 5$ </td><td> $0 . 2 5 6 \pm 0 . 0 2 0$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 6$ </td><td>85.13</td></tr><tr><td>RAND-PSNN</td><td> ${ \bf 8 9 . 4 8 \pm 0 . 6 9 }$ </td><td> $\mathbf { 0 . 2 4 3 \ : \pm { \ : 0 . 0 2 0 } }$ </td><td> $\mathbf { 0 . 0 1 5 \ : \pm { \ : 0 . 0 0 5 } }$ </td><td> $6 0 . 9 1 \pm 3 . 4 4$ </td></tr><tr><td>SNI-PSNN</td><td> $8 8 . 7 0 \pm 0 . 9 0$ </td><td> $0 . 2 9 2 \pm 0 . 0 1 7$ </td><td> $0 . 0 2 4 \pm 0 . 0 1 2$ </td><td>90.00</td></tr></table>

Table 9: Results on CREDIT $( \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td></td><td>ACC in % (↑)</td><td>NLL (↓)</td><td>E (↓)</td><td>Deterministic parameters in %</td></tr><tr><td>DWF-PSNN</td><td> $7 8 . 0 2 \pm 0 . 6 1$ </td><td> $0 . 5 0 8 \pm 0 . 0 1 0$ </td><td> $0 . 0 2 2 \pm 0 . 0 1 0$ </td><td>60.90 ± 2.61</td></tr><tr><td>ALS-BNN</td><td> ${ \bf 7 8 . 0 3 \pm 0 . 6 1 }$ </td><td> $0 . 5 0 2 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 6$ </td><td>0.00</td></tr><tr><td>FLS-BNN</td><td> ${ \bf 7 8 . 0 3 \pm 0 . 6 1 }$ </td><td> $0 . 5 0 6 \pm 0 . 0 0 9$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 5$ </td><td>81.11</td></tr><tr><td>RAND-PSNN</td><td> $7 8 . 0 2 \pm 0 . 6 1$ </td><td> $\mathbf { 0 . 5 0 0 \ : \pm { \ : 0 . 0 0 9 } }$ </td><td> $0 . 0 2 3 \pm 0 . 0 0 6$ </td><td>60.68 ± 2.61</td></tr><tr><td>SNI-PSNN</td><td> $7 8 . 0 2 \pm 0 . 6 1$ </td><td> $0 . 5 2 7 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 0 1 4 \ : \pm { \ : 0 . 0 1 3 } }$ </td><td>90.00</td></tr></table>

Table 10: Results on CONCRETE (mean ± std).
<table><tr><td></td><td>RMSE (↓)</td><td>NLL (↓)</td><td>ECE (↓)</td><td>Deterministic parameters in %</td></tr><tr><td>DWF-PSNN</td><td> ${ \bf 0 . 2 9 8 \pm 0 . 0 2 7 }$ </td><td> $\mathbf { 0 . 2 1 4 \ : \pm { \ : 0 . 0 7 3 } }$ </td><td> $0 . 1 0 7 \pm 0 . 0 6 6$ </td><td> $4 8 . 6 9 \pm 1 . 2 7$ </td></tr><tr><td>ALS-BNN</td><td> $0 . 3 0 0 \pm 0 . 0 3 2$ </td><td> $0 . 2 8 3 \pm 0 . 0 6 8$ </td><td> $0 . 1 3 4 \pm 0 . 0 6 2$ </td><td>0.00</td></tr><tr><td>FLS-BNN</td><td> $0 . 3 3 5 \pm 0 . 0 3 0$ </td><td> $0 . 2 8 6 \pm 0 . 0 8 5$ </td><td> $0 . 0 8 5 \pm 0 . 0 6 2$ </td><td>91.89</td></tr><tr><td>RAND-PSNN</td><td> $0 . 3 3 1 \pm 0 . 0 3 4$ </td><td> $0 . 3 2 2 \pm 0 . 1 1 3$ </td><td> $\mathbf { 0 . 0 7 5 \ : \pm { \ : 0 . 0 7 3 } }$ </td><td> $4 8 . 6 9 \pm 1 . 2 7$ </td></tr><tr><td>SNI-PSNN</td><td> $0 . 3 3 2 \pm 0 . 0 3 9$ </td><td> $0 . 3 2 6 \pm 0 . 1 3 3$ </td><td> $0 . 0 9 7 \pm 0 . 0 8 8$ </td><td>90.00</td></tr></table>

Table 11: Results on WINE (mean ± std).
<table><tr><td></td><td>RMSE (↓)</td><td>NLL (↓)</td><td>ECE (↓)</td><td>Deterministic parameters in %</td></tr><tr><td>DWF-PSNN</td><td> $0 . 7 7 1 \pm 0 . 0 2 3$ </td><td> $2 . 2 1 0 \pm 0 . 1 4 5$ </td><td> $0 . 1 2 7 \pm 0 . 0 1 5$ </td><td> $4 7 . 1 2 \pm 2 . 9 3$ </td></tr><tr><td>ALS-BNN</td><td> $\mathbf { 0 . 7 2 3 \ : \pm { \ : 0 . 0 3 1 } }$ </td><td> $\mathbf { 1 . 0 4 9 \ : \pm { \ : 0 . 0 6 5 } }$ </td><td> $\mathbf { 0 . 0 1 7 \ : \pm { \ : 0 . 0 1 5 } }$ </td><td>0.00</td></tr><tr><td>FLS-BNN</td><td> $0 . 8 5 6 \pm 0 . 0 2 4$ </td><td> $3 . 3 9 5 \pm 0 . 2 0 2$ </td><td> $0 . 2 0 7 \pm 0 . 0 1 7$ </td><td>90.27</td></tr><tr><td> $\mathrm { R A N D - P S N N }$ </td><td> $0 . 8 4 1 \pm 0 . 0 3 5$ </td><td> $3 . 3 1 4 \pm 0 . 2 9 4$ </td><td> $0 . 1 4 9 \pm 0 . 0 5 0$ </td><td> $4 7 . 0 7 \pm 2 . 9 3$ </td></tr><tr><td>SNI-PSNN</td><td> $0 . 8 4 4 \pm 0 . 0 3 6$ </td><td> $3 . 3 3 8 \pm 0 . 3 0 4$ </td><td> $0 . 1 3 5 \pm 0 . 0 2 9$ </td><td>90.00</td></tr></table>

Table 12: Results on BIKESHARING $( \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td></td><td>RMSE (↓)</td><td>NLL (↓)</td><td>EE (↓)</td><td>Deterministic parameters in %</td></tr><tr><td> $\mathrm { D W F - P S N N }$ </td><td> $0 . 1 0 5 \pm 0 . 0 3 4$ </td><td> $- 0 . 1 5 9 \pm 0 . 0 3 4$ </td><td> $0 . 5 8 5 \pm 0 . 1 9 6$ </td><td> $4 7 . 8 1 \pm 2 . 9 5$ </td></tr><tr><td>ALS-BNN</td><td> $0 . 1 5 5 \pm 0 . 0 3 7$ </td><td> $0 . 0 4 9 \pm 0 . 0 3 8$ </td><td> $\mathbf { 0 . 4 3 2 \ : \pm { \ : 0 . 0 8 4 } }$ </td><td> $0 . 0 0$ </td></tr><tr><td>FLS-BNN</td><td> $\mathbf { 0 . 1 0 1 \pm 0 . 0 4 0 }$ </td><td> $- 0 . 1 6 4 \pm 0 . 0 4 1$ </td><td> $0 . 6 1 4 \pm 0 . 2 1 3$ </td><td>88.70</td></tr><tr><td> $\mathrm { R A N D - P S N N }$ </td><td> $\mathbf { 0 . 1 0 1 \ : \pm 0 . 0 3 7 }$ </td><td> $\mathbf { - 0 . 1 7 4 \ : \pm { \ : 0 . 0 5 0 } }$ </td><td> $0 . 6 0 0 \pm 0 . 1 9 1$ </td><td> $4 7 . 7 5 \pm 2 . 9 5$ </td></tr><tr><td>SNI-PSNN</td><td> $0 . 1 0 4 \pm 0 . 0 3 3$ </td><td> $- 0 . 1 7 3 \pm 0 . 0 3 3$ </td><td> $0 . 6 7 4 \pm 0 . 1 8 0$ </td><td>90.00</td></tr></table>

Table 13: Results on BOSTON $( \mathrm { m e a n } \pm \mathrm { s t d } )$
<table><tr><td></td><td>RMSE (↓)</td><td>NLL (↓)</td><td>ECE (↓)</td><td>Deterministic parameters in %</td></tr><tr><td>DWF-PSNN</td><td> ${ \bf 0 . 3 8 6 \pm 0 . 1 1 3 }$ </td><td> $0 . 4 8 2 \pm 0 . 3 3 1$ </td><td> $0 . 2 0 9 \pm 0 . 1 4 5$ </td><td> $4 8 . 0 3 \pm 2 . 5 3$ </td></tr><tr><td>ALS-BNN</td><td> $0 . 4 5 1 \pm 0 . 1 0 7$ </td><td> $\mathbf { 0 . 4 7 0 \ : \pm { \ : 0 . 1 7 0 } }$ </td><td> $0 . 2 2 6 \pm 0 . 1 6 6$ </td><td>0.00</td></tr><tr><td>FLS-BNN</td><td> $0 . 3 9 2 \pm 0 . 1 1 6$ </td><td> $0 . 5 1 3 \pm 0 . 3 3 7$ </td><td> $0 . 2 1 3 \pm 0 . 1 7 2$ </td><td>87.93</td></tr><tr><td> $\mathrm { R A N D - P S N N }$ </td><td> $0 . 3 9 8 \pm 0 . 1 0 2$ </td><td> $0 . 6 0 8 \pm 0 . 3 8 5$ </td><td> $\mathbf { 0 . 1 9 6 \ : \pm { \ : 0 . 1 9 3 } }$ </td><td> $4 7 . 9 7 \pm 2 . 5 3$ </td></tr><tr><td>SNI-PSNN</td><td> $0 . 4 0 8 \pm 0 . 1 0 6$ </td><td> $0 . 6 5 2 \pm 0 . 4 2 5$ </td><td> $0 . 3 1 0 \pm 0 . 3 0 5$ </td><td>90.00</td></tr></table>

## C THE LEARNED SPLIT

Figure 1 and Table 14 summarize the split learned by the prior optimization. The split is stable across splits of the data, with standard deviations of at most 0.04 in the share of deterministic weights. The first and the output layer are largely deterministic, while the middle layer keeps about half of its weights stochastic. Since the middle layer holds the large majority of the parameters, it dominates the overall share of deterministic parameters reported in Section B. The classification datasets are sparsified slightly more than the regression datasets.

The middle layer is sparsified least, although its regularization weight exceeds that of the first layer. This does not contradict Remark 4.2, which assumes comparable gradient bounds $G _ { l }$ across layers. The middle layer has a larger fan-in, and the gradient of the objective with respect to its prior scales is therefore unlikely to be bounded by the same constant as in the other layers.

In every dataset and split, between 7 and 27 of the 100 first-layer biases remain stochastic. The coverage condition $U ( S ) \geq d _ { L }$ of Proposition 4.7 therefore holds with a wide margin. Before repair, the condition $| T | \geq d _ { 0 }$ holds on BANKNOTE, HTRU2 and in nine of ten splits of CONCRETE. On the other datasets the repair changes on average between 5.5 and 28.1 parameters per split, with the largest number of changes on CREDIT, which has the most input features (cf. Table 4). Whether the output bias is deterministic varies between datasets, from none of the splits on BANKNOTE to all of them on CREDIT and WINE. The certificate places no condition on the output bias, so this variation does not afect the expressiveness of the masked network. All values refer to the learned masks before repair.

Table 14: Learned split of DWF-PSNN. The first columns give the share of deterministic parameters per layer, with mean and standard deviation over the ten splits. The next column gives the range over splits of the number of stochastic first-layer biases out of 100. The last column gives the number of splits in which the output bias is deterministic.
<table><tr><td rowspan="2">Dataset</td><td colspan="3">Deterministic parameters</td><td rowspan="2">Stochastic</td><td rowspan="2">Deterministic output bias</td></tr><tr><td>Layer 1</td><td>Layer 2</td><td>Output layer first-layer biases</td></tr><tr><td>ONLINESHOPPERS</td><td> $0 . 8 7 \pm 0 . 0 1$ </td><td> $0 . 5 6 \pm 0 . 0 4$ </td><td> $0 . 8 6 \pm 0 . 0 3$ </td><td>9-17</td><td>6/10</td></tr><tr><td>CREDIT</td><td> $0 . 8 6 \pm 0 . 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 3$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td><td>7-20</td><td>10/10</td></tr><tr><td>BANKNOTE</td><td> $0 . 8 7 \pm 0 . 0 2$ </td><td> $0 . 5 3 \pm 0 . 0 3$ </td><td> $0 . 8 4 \pm 0 . 0 2$ </td><td>8-18</td><td>0/10</td></tr><tr><td>HTRU2</td><td> $0 . 8 7 \pm 0 . 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 3$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td><td>10-18</td><td>7/10</td></tr><tr><td>CONCRETE</td><td> $0 . 8 0 \pm 0 . 0 1$ </td><td> $0 . 4 6 \pm 0 . 0 1$ </td><td> $0 . 7 8 \pm 0 . 0 4$ </td><td>11-22</td><td></td></tr><tr><td>BOSTON</td><td> $0 . 8 1 \pm 0 . 0 1$ </td><td> $0 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 7 8 \pm 0 . 0 3$ </td><td>12-24</td><td>4/10 5/10</td></tr><tr><td>BIKESHARING</td><td> $0 . 8 2 \pm 0 . 0 1$ </td><td> $0 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 8 0 \pm 0 . 0 4$ </td><td>13-25</td><td>4/10</td></tr><tr><td>WINE</td><td> $0 . 8 0 \pm 0 . 0 1$ </td><td> $0 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 8 6 \pm 0 . 0 3$ </td><td>18-27</td><td>10/10</td></tr></table>

## D THE CONDITIONS OF THE CERTIFICATE

Table 15: The violated conditions are the ones in Proposition 4.7. Values are the mean Wasserstein-1 distance to the true conditional over nine held out inputs, with the standard error over five data seeds. The sampling floor is the distance between two independent samples from the true conditional at this sample size, so no model can go below it, and the point predictor is what a model reaches by matching the conditional median without representing any spread. The last column repeats the study with learnable prior scales.
<table><tr><td rowspan="2">variant</td><td rowspan="2">condition violated</td><td rowspan="2"> $n _ { \mathrm { s t o c h } }$ </td><td colspan="2"> $W _ { 1 }$ </td></tr><tr><td>fixed</td><td>learned</td></tr><tr><td>sampling floor</td><td></td><td></td><td>0.020</td><td>0.020</td></tr><tr><td>point predictor</td><td></td><td>0</td><td>0.681</td><td>0.681</td></tr><tr><td>noise input,  $d _ { L }$ </td><td>reference, none</td><td>0</td><td> $0 . 0 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 0 5 3 \pm 0 . 0 0 4$ </td></tr><tr><td>first layer,  $d _ { L }$ </td><td>none</td><td>1</td><td> $0 . 0 7 4 \pm 0 . 0 0 2$ </td><td> $0 . 0 7 5 \pm 0 . 0 0 2$ </td></tr><tr><td>first layer,  $d _ { L } ,$  weight</td><td>vanishes at  $x = 0$ </td><td>1</td><td> $0 . 2 8 4 \pm 0 . 0 0 2$ </td><td> $0 . 3 2 0 \pm 0 . 0 1 0$ </td></tr><tr><td>output  $\mathrm { l a y e r } , d _ { L }$ </td><td>bias coverage</td><td>1</td><td> $0 . 5 6 8 \pm 0 . 0 0 2$ </td><td> $0 . 5 8 8 \pm 0 . 0 0 3$ </td></tr><tr><td>output layer, all</td><td>bias coverage</td><td>64</td><td> $0 . 4 9 5 \pm 0 . 0 0 4$ </td><td> $0 . 4 6 5 \pm 0 . 0 0 5$ </td></tr><tr><td>first layer,  $d _ { L }$  , linear</td><td>activation</td><td>1</td><td> $0 . 9 3 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 3 5 \pm 0 . 0 0 2$ </td></tr><tr><td>full</td><td>deterministic units</td><td>4353</td><td> $5 . 4 1 1 \pm 0 . 0 0 8$ </td><td> $0 . 1 1 1 \pm 0 . 0 0 5$ </td></tr><tr><td>deterministic</td><td>no randomness</td><td>0</td><td> $0 . 6 8 6 \pm 0 . 0 0 2$ </td><td> $0 . 6 8 6 \pm 0 . 0 0 2$ </td></tr></table>

![](images/6a76269b14f502c5bbc4b2896a3ccb4b3920134404c278e80f1cf1dcbf3daa6c.jpg)  
stochastic parameters in the output layer  
Figure 6: Distance to the true conditional against the number of randomized parameters in the output layer, with the certified configuration and the sampling floor as horizontal lines. Mean over five data seeds with the standard error.

Setup. The target has a bimodal conditional distribution with an input dependent mixture weight, so that any Gaussian conditional is excluded by construction. Since the certificate is a statement about representation, posterior inference would confound capacity with the quality of the sampler. Every variant is therefore fitted directly as a conditional generator by minimizing the energy score, which is strictly proper, and is scored by the Wasserstein-1 distance to the true conditional. Within each data seed we keep the best of five restarts, selected on the validation energy score and never on the reported distance, because the statement is an existence statement about parameters. All variants share one architecture with two hidden layers of width sixty-four and difer only in which parameters are randomized. In the column fixed, every stochastic parameter has prior scale $\sigma = 1$ . In the column learned, the scales are parameterized by a softplus and fitted jointly with the means, without a sparsity penalty.

The reference acts as a gate. In the reference the randomness enters as an additional input and every weight is deterministic, so the map is unconstrained. It reaches 0.053 against a sampling floor of 0.020, which shows that the fitting procedure is not the binding constraint and that the remaining diferences can be read as consequences of the condition each variant violates.

Why biases and not weights. The variant that randomizes a weight of the first layer rather than a bias satisfies the count and the position, and still reaches only 0.284. A stochastic input weight contributes $\sigma _ { j , i } \varepsilon _ { j , i } x _ { i } .$ which vanishes at $\mathbf { x } = \mathbf { 0 } ,$ so the scale $s _ { j }$ in the proof of Proposition 4.7 is not bounded away from zero and the predictive distribution degenerates there. This is the reason Definition 4.4 counts biases.

Fixed and learnable prior scales. Only the fully stochastic row depends on this choice, and it changes by a factor of fifty. With learnable scales that variant contains the certified one as a special case, since it can drive all but $d _ { L }$ standard deviations to zero, and the optimizer finds such a configuration. The condition $| T | \geq d _ { 0 }$ therefore constrains a fixed mask with fixed scales, which is the setting of our two stage method, where the prior scales are determined in the first stage and held during sampling. With fixed scales the variant is not merely worse but wildly overdispersed, at eight times the distance of a point predictor, which is why its panel in Figure 2 appears flat.

Count does not substitute for position. Figure 6 increases the number of randomized parameters in the output layer from one to sixty-four. The distance is flat between 0.46 and 0.59 throughout and never approaches the certified configuration at 0.074, so a factor of sixty-four in the count does not compensate for a position the certificate rejects.

![](images/2ec6db0926ade124bf6a2ddfefbf3da99f7e353eea37747b0ada7cd41e472078.jpg)  
Figure 7: All variants of Table 15 at three inputs, with fixed prior scales. In the panel titles d denotes the output dimension $d _ { L } .$ rank denotes the condition $| T | \geq d _ { 0 }$ on the fully deterministic units and no noise corresponds to no randomness in Table 15. Grey dashed is the true conditional, blue marks the configurations the certificate admits. The vertical range is fixed by the target density, so mass outside it appears flat.

Figure 8 shows the prior scales behind Figure 4. For $\lambda = 1 0 ^ { - 5 }$ most factors are driven many orders of magnitude below the cutof, which shows that the penalty of Section 4.1 pushes pruned scales towards zero. For $\lambda = 3 \cdot 1 0 ^ { - 6 }$ the scales form a continuum across the cutof.

![](images/b351819b14c7a99ae1455cec0f3f6ef62597fa71357a6885d4b040f4e65abf30.jpg)  
Figure 8: Histogram of $\log _ { 1 0 }$ |u<sub>θ</sub>| on the bimodal target over five data seeds, with the cutof $c = 0 . 0 1$ on the prior scale, which corresponds to $| u | \approx 0 . 2 .$ , as a dashed line. For larger λ the pruned products $u _ { \theta }$ collapse towards zero.

## E EXPERIMENTAL DETAILS

We use a mlp with two hidden layers of 100 units each and a tanh activation function, where not stated diferently. All regression and classification experiments were run on an Nvidia A40 GPU, and each model finished in less than two hours. The runs for the role of $\gamma$ and the ucda experiment have also been performed on an Nvidia A40 GPU and finished within hours. All other experiments were performed on a MacBook Pro 2021 with Apple M1 Pro chip and took between a few minutes and a few hours. Unless stated otherwise, we use the random seed 123 for initialization and sampling, and all methods share the same ten data splits. We build on the code base provided by (Tran et al., 2022). The repository does not state a license. The accompanying paper is published under CC-BY 4.0. We have not performed an extensive hyperparameter search, but in most cases tried out a few configurations and picked the best one. Additionally, we use a two-phase learning-rate schedule. Let $T$ denote the total number of optimization steps and let $T _ { 0 } = \lfloor T / 2 \rfloor$ . For optimizer $j ,$ , the learning rate at step $t \in \{ 0 , \ldots , T - 1 \}$ is given by

$$
\eta _ { t } ^ { ( j ) } = \eta _ { 0 } ^ { ( j ) } \left\{ \begin{array} { l l } { 1 , } & { t < T _ { 0 } , } \\ { ( t - T _ { 0 } + 1 ) ^ { - \alpha _ { j } } , } & { t \geq T _ { 0 } , } \end{array} \right.\tag{2}
$$

where $\alpha _ { \mathrm { S G H M C } } = 0 . 2$ and $\alpha _ { \mathrm { A d a m } } = 0 . 4$

The learned masks satisfy the coverage condition in all runs. The condition $| T | \geq d _ { 0 }$ fails on the datasets with many input features, where fewer than $d _ { 0 }$ first-layer units are fully deterministic. In these runs we apply the minimal repair of Corollary A.9, which changes between 1 and 34 parameters per run (Table 4).

## E.1 Regression experiments

For regression, ece denotes the calibration error of Kuleshov et al. (2018). We summarize the predictive distribution of test point i by a Gaussian with the predictive mean and variance and evaluate its cumulative distribution function $F _ { i }$ at the target $y _ { i }$ . For the confidence levels $p _ { j } = j / 1 0$ with $j = 1 , \dots , 1 0$ , let $\hat { p } _ { j }$ be the share of test points with $F _ { i } ( y _ { i } ) < p _ { j }$ . We report $\textstyle \sum _ { j = 1 } ^ { 1 0 } ( p _ { j } - { \hat { p } } _ { j } ) ^ { 2 }$ . This is a sum over the levels and not an average, so its scale is not comparable to that of the classification ece. Additionally, we standardize the targets. The target gp has a constant zero mean function, a variance of 1, a length-scale of $\ell = \sqrt { 2 \cdot d _ { 0 } }$ , and a noise variance of 0.1. On top of that, we have the following hyperpriors:

$$
\ell _ { i } \sim \mathrm { L o g N o r m a l } \big ( \log ( \ell ) , 1 \big ) , \quad i = 1 , \ldots , d _ { 0 }
$$

$$
\alpha ^ { 2 } \sim \mathrm { L o g N o r m a l } ( 0 . 1 , 1 )
$$

We set D=3 and the cutof as 0.01. The configuration for the prior optimization are:

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of iterations</td><td>10000</td></tr><tr><td>Regularization (λ)</td><td>(9, 12, 15) / D</td></tr><tr><td>Number of function samples</td><td>100</td></tr><tr><td>Learning rate</td><td>0.01</td></tr><tr><td>Kernel</td><td>Polynomial of degree 2</td></tr></table>

Table 16: Hyperparameters used for prior optimization  
For sampling we have used the following hyperparameters:

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of samples per chain</td><td>30</td></tr><tr><td>Discarded samples before sampling</td><td>10</td></tr><tr><td>Thinning interval</td><td>200</td></tr><tr><td>Number of chains</td><td>4</td></tr><tr><td>Momentum decay</td><td>0.02</td></tr><tr><td>Burn in steps</td><td>3000</td></tr><tr><td>SGHMC learning rate (h)</td><td>0.03</td></tr><tr><td>Adam learning rate (η)</td><td>0.003</td></tr></table>

Table 17: Hyperparameters used for all models but SNI-PSNN  
We have set γ = 2.5.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Prior precision</td><td>1.0</td></tr><tr><td>Stochastic fraction</td><td>0.1</td></tr><tr><td>Noise variance</td><td>0.1</td></tr><tr><td>Epochs</td><td>1000</td></tr><tr><td>Learning rate</td><td>0.01</td></tr></table>

Table 18: Hyperparameters used for SNI-PSNN

## E.2 Classification experiment

The gp is set up as in the regression experiment. For the prior optimization we also use the same hyperparameters, but with a batch size of 128. For the sampling we use the following hyperparameters:

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of samples per chain</td><td>30</td></tr><tr><td>Discarded samples before sampling</td><td>10</td></tr><tr><td>Thinning interval</td><td>200</td></tr><tr><td>Number of chains</td><td>4</td></tr><tr><td>Momentum decay</td><td>0.02</td></tr><tr><td>Burn in steps</td><td>3000</td></tr><tr><td>SGHMC learning rate (h)</td><td>0.01</td></tr><tr><td>Adam learning rate (η)</td><td>0.001</td></tr></table>

Table 19: Hyperparameters used for all models but SNI-PSNN  
Again we have set $\gamma = 2 . 5 .$

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Prior precision</td><td>1.0</td></tr><tr><td>Stochastic fraction</td><td>0.1</td></tr><tr><td>Noise variance</td><td>0.1</td></tr><tr><td>Epochs</td><td>1000</td></tr><tr><td>Learning rate</td><td>0.01</td></tr></table>

Table 20: Hyperparameters used for SNI-PSNN

## E.3 Prior optimization

The hyperpriors are the same, but the hyperparameters are as follows:

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Number of iterations</td><td>15000</td></tr><tr><td>Regularization (λ)</td><td>(1.2, 2.4, 4.8) / D</td></tr><tr><td>Number of function samples</td><td>100</td></tr><tr><td>Learning rate</td><td>0.03</td></tr><tr><td>Kernel</td><td>Polynomial of degree 2</td></tr></table>

Table 21: Hyperparameters used for prior optimization

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of samples per chain</td><td>60</td></tr><tr><td>Discarded samples before sampling</td><td>20</td></tr><tr><td>Thinning interval</td><td>100</td></tr><tr><td>Number of chains</td><td>4</td></tr><tr><td>Momentum decay</td><td>0.01</td></tr><tr><td>Burn in steps</td><td>10000</td></tr><tr><td>SGHMC learning rate (h)</td><td>0.03</td></tr><tr><td>Adam learning rate (η)</td><td>0.003</td></tr></table>

Table 22: Sampling hyperparameters  
Additionally, we have set $\gamma = 2 5 ,$

## E.4 Role of γ

We have used the same hyperparameters for the prior optimization as in the regression experiments. For sampling we have used:

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Number of samples per chain</td><td>100</td></tr><tr><td>Discarded samples before sampling</td><td>10</td></tr><tr><td>Number of burn-in steps</td><td>10000</td></tr><tr><td>Thinning interval</td><td>20</td></tr><tr><td>Number of chains</td><td>4</td></tr><tr><td>Momentum decay</td><td>0.02</td></tr><tr><td>SGHMC sampler rate (h)</td><td>0.03</td></tr><tr><td>Adam learning rate (η)</td><td>0.003</td></tr></table>

Table 23: Hyperparameters used for sampling

For (ii) we use $\gamma = 0 . 0 3$ and $\gamma = 2 . 5 ,$ , and for (iii) $\gamma = 1 0 ^ { 1 3 }$

## E.5 Learned and random splits on the bimodal target

We use the architecture, target and energy-score fitting of Section D. In a first stage, the prior scales are parameterized as $\sigma _ { \theta } = H ( u _ { \theta } )$ with $D = 3$ factors and fitted jointly with the means under the penalty of Section 4.1, with $\lambda \in \{ 3 \cdot 1 0 ^ { - 7 } , 7 \cdot 1 0 ^ { - 7 } , 1 . 5 \cdot 1 0 ^ { - 6 } , 3 \cdot 1 0 ^ { - 6 } , 6 \cdot 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 . 5 \cdot 1 0 ^ { - 5 } \}$ and layer weights in the ratio 3 : 4 : 5. Scales below $c = 0 . 0 1$ are set to zero. Unlike in the UCI experiments, we apply this threshold only after the first stage, so that all scales receive gradients during the optimization. In a second stage, the means are refitted with the mask and the scales held fixed. Random masks permute the thresholded scales either within each layer and parameter type or across all parameters. We use five data seeds, three random masks of each kind per seed and 12,000 steps per fit.

## E.6 Sensitivity to the cutof

We use the setting of Section E.5 with $\lambda \in \{ 3 \cdot 1 0 ^ { - 6 } , 1 0 ^ { - 5 } \}$ and five data seeds. After the first stage we threshold the learned scales at $c \in \{ 0 . 0 0 3 , 0 . 0 1 , 0 . 0 3 , 0 . 1 \}$ and refit the means with the mask and the scales held fixed, using 12,000 steps per fit.

## E.7 Certificate of random masks

For each dataset we draw 2,000 random first-layer masks with the learned shares of deterministic weights and biases and check both conditions of Proposition 4.7. At the learned budget, random masks satisfy the certificate on BANKNOTE and HTRU2, in 97% of the draws on CONCRETE, and in at most 9% of the draws on the other datasets.