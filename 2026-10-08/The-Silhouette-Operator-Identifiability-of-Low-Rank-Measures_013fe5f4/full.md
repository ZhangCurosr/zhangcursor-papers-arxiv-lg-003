# The Silhouette Operator: Identifiability of Low-Rank Measures from One-Dimensional Projections

Robert A. Vandermeulen

## Abstract

Structured recovery phenomena, such as restricted isometry properties in compressed sensing, have shown that high dimensional objects can often be reconstructed from remarkably low-dimensional linear measurements. This work develops an analogous recovery framework for low-rank signed measures on $\mathbb { R } ^ { 2 }$ , defined here as measures that can be expressed as finite sums of product measures with one-dimensional factors. The framework is based on linear operators, termed silhouette operators, that map a measure to a fixed finite collection of one-dimensional linear pushforwards. The main results show that a suitably chosen collection of 2k projected marginals sufices to identify every compactly supported rank-≤ k signed measure, that this number is optimal, and that the projection directions cannot be chosen arbitrarily. The framework is also extended to higher-dimensional sums of product measures by establishing suficient conditions under which collections of pairwise marginals identify the full model. Building on this framework, a computationally eficient estimator, termed silhouette mixture estimation (SME), is introduced for constructing a low-rank empirical measure from data by matching its one-dimensional projected marginals to the corresponding empirical marginals in Wasserstein distance. When combined with one-dimensional density estimators, SME yields an eficient nonparametric density estimator that performs strongly relative to a range of parametric, nonparametric, and deep-learning baselines in settings of moderate dimension and sample size.

Keywords: Cram´er–Wold theorem, Identifiability, Low-rank measures, Mixture-of-products models, Multiview models, Nonparametric density estimation, Nonparametric mixture models, Restricted isometry property, Sliced Wasser stein distance, Wasserstein distance

## 1 Introduction

High-dimensional problems have become a central focus of modern statistics and machine learning. A perennial challenge in this setting is the curse of dimensionality: statistical and computational dificulty often grows rapidly with dimension. Numerous techniques have been proposed to mitigate these dificulties. Among the most widely used and theoretically grounded techniques are those that exploit sparsity in vectors and low-rank structure in matrices or tensors. Within these frameworks, the restricted isometry property (RIP) [1, 2] and the closely related notion of restricted strong convexity [3] play central roles. For matrices, an RIP operator $T : \mathbb { R } ^ { m \times n }  \mathbb { R } ^ { \ell }$ (with ℓ typically on the order of $k ( m + n ) )$ is a linear operator that approximately preserves the geometry of the space of rank-≤ k matrices. Injectivity of $T$ on rank-≤ k matrices is equivalent, after appropriately rescaling $T ,$ to matrix-RIP of order 2k; see Appendix A. Although the existence of such an embedding is consistent with the intrinsic dimensionality $\Theta ( k ( m + n ) )$ ) of the model class, it is noteworthy that the embedding can be realized by a linear map.

This work establishes an analogous phenomenon for a structured class of measures on Euclidean space. In this work, a finite signed measure $\mu$ on $\mathbb { R } ^ { 2 }$ is said to have rank at most k if it admits a representation of the form

$$
\boldsymbol { \mu } = \sum _ { i = 1 } ^ { k } \boldsymbol { \mu } _ { i , 1 } \times \boldsymbol { \mu } _ { i , 2 } ,\tag{1}
$$

![](images/11656841976fcbd874dc296722c38a6e2f35563c9b984f48a88af55c526c6820.jpg)  
Fig. 1: Illustration of a silhouette operator for $k = 2 ,$ , corresponding to $a = - 2$ and $b _ { 1 } = b _ { 2 } = 1$ in Theorem 1. A measure $\mu$ on $\mathbb { R } ^ { 2 }$ is mapped to its one-dimensional pushforwards along the resulting four projection directions. When $\mu$ has rank at most 2, these four projected measures uniquely determine $\mu .$

where each $\mu _ { i , j }$ is a finite signed measure on R, and $\mu _ { i , 1 } \times \mu _ { i , 2 }$ denotes their product measure on $\mathbb { R } ^ { 2 }$ . This notion is directly analogous to matrix rank. In particular, a measure has rank exactly k if and only if k is the smallest integer for which it admits a representation of the form (1). Equivalently, it has rank exactly k if and only if it admits such a representation for which $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are each linearly independent (Proposition 13).

The core structural contribution of this work, established in Theorems 1 and 3, is that a broad class of rank-≤ k measures, including all compactly supported and, more generally, all sub-Gaussian measures, can be uniquely identified from linear pushforwards to R along a suitably chosen collection of 2k vectors $v _ { 1 } , \dots , v _ { 2 k } \in \mathbb { R } ^ { 2 }$ . Given such a collection, the map sending a measure to the corresponding tuple of projected marginals will be referred to as a silhouette operator, and the resulting tuple as the silhouette of the measure. For probability measures, the result says that the joint distribution $\mu$ of $[ X , Y ] ^ { \top }$ is uniquely determined by the distributions of 2k suitably chosen linear combinations of X and Y. Writing L for the law of a random variable, these projected distributions are

$$
\mathcal { L } ( \langle v _ { 1 } , [ X , Y ] ^ { \top } \rangle ) , \ldots , \mathcal { L } ( \langle v _ { 2 k } , [ X , Y ] ^ { \top } \rangle ) .
$$

Figure 1 illustrates this construction for $k = 2$

This number of projections is optimal: Theorem 8 shows that no collection of 2k − 1 projection directions identifies all rank $^ - \leq$ k signed measures. Even with 2k projections, the directions cannot be chosen arbitrarily: Proposition 9 shows that certain symmetric configurations fail to provide identifiability. Moreover, the silhouette operator is a weak<sup>∗</sup> homeomorphic embedding on the set of rank-≤ k probability measures sup ported in a fixed compact set, and more generally on any uniformly total-variation-bounded set of rank-≤ k signed measures supported in that compact set.

These results provide a mathematically principled basis for exploiting the statistical and algorithmic advantages of one-dimensional methods in multi-dimensional settings. For example, convergence in Wasserstein distance of the projected measures implies convergence in Wasserstein distance on $\mathbb { R } ^ { 2 }$ . Projection-based approaches of this type underlie the sliced Wasserstein distance and related methods $[ 4 - 7 ] ;$ this work shows that a finite collection of such projections preserves the full weak<sup>∗</sup> topology on a rich structured class of measures.

The silhouette framework is extended to higher dimensions by identifying a measure from a collection of its two-dimensional marginals, each of which can be determined through its silhouette operator. For example, one natural approach to inferring the joint distribution of $( X _ { 1 } , X _ { 2 } , X _ { 3 } )$ is to couple the silhouettes of $( X _ { 1 } , X _ { 2 } )$ and $( X _ { 2 } , X _ { 3 } )$ through their shared coordinate. To formalize this approach, Section 3 establishes measure-theoretic identifiability results for models of the form

$$
\mu = \sum _ { i = 1 } ^ { k } w _ { i } \prod _ { j = 1 } ^ { d } \mu _ { i , j } ,\tag{2}
$$

where each $\mu _ { i , j }$ is a probability measure and the weights $w _ { 1 } , \ldots , w _ { k }$ are nonnegative and sum to one. Probability measures of this form will be referred to as mixture-of-products models (also known in various settings as multiview models, nonparametric mixture models, or naive Bayes mixtures). Here $\textstyle \prod _ { j = 1 } ^ { d } \mu _ { j }$ denotes the usual product measure on the product σ-algebra. It is shown that these models can be identified from collections of two-dimensional marginals under assumptions adapted from standard identifiability conditions for matrix and tensor models (including linear independence and joint irreducibility [8, 9], with the latter serving as a measure-theoretic analogue of the anchor-word assumption [10]).

In nonparametric density estimation on $\mathbb { R } ^ { d }$ , mixture-of-products models (2) are known to exhibit favorable statistical behavior under suitable regularity assumptions, particularly in high-dimensional regimes [11–17]. However, practical estimation of these models remains challenging in the continuous-density setting. Existing estimation methods include approaches based on matrix or tensor factorization. A natural example is lowrank factorization of a joint histogram, where the bin resolution must be selected and finer resolutions lead to larger factorization problems. Other approaches repeatedly estimate component densities during model fitting, which can also entail substantial computational cost [18, 19]. These approaches are discussed further in Section 4.1.

Building on the silhouette-based identifiability results above, this work develops a concrete algorithm, termed silhouette mixture estimation (SME), for estimating low-rank empirical measures of the form (2). SME separates fitting the empirical mixture structure from estimating component densities, so choices such as density-estimation bandwidths need not enter the mixture-fitting objective. Several downstream uses of the resulting representations are discussed, with density estimation being a particularly natural application.

SME is computationally eficient because its objective is built from univariate Wasserstein distances that can be evaluated rapidly and in parallel. Combined with standard univariate density estimators, the approach appears particularly efective in moderate-dimensional settings with limited sample sizes. Empirically, the resulting estimator performs strongly on benchmark datasets, outperforming a range of parametric, nonparametric, and deep-learning baselines.

Taken together, the theoretical and algorithmic developments presented above establish a formal connection between sliced methods and nonparametric mixture models of the form (2). The silhouette operator provides a tractable linear transformation from multivariate measures to collections of one-dimensional measures, enabling computation and estimation to be carried out in the projected space without sacrificing identifiability of the underlying low-rank measure. Although these results bear a close structural resemblance to phenomena in compressed sensing, the proofs are wholly independent of compressed-sensing arguments and use original techniques introduced in this work.

Section 7 establishes the sharpness of the projection count for identifiability and shows that, even with 2k projections, the directions cannot be chosen arbitrarily. It further shows that the tail and boundedness assumptions cannot in general be omitted from the identifiability and homeomorphism results, respectively. Open questions and directions for further investigation are also identified.

## 2 Main Theorems

This section establishes the core injectivity and topological results for the silhouette operator on classes of low-rank measures in $\mathbb { R } ^ { 2 }$ . These results form the foundation for the subsequent theoretical and algorithmic developments. Necessary notation and definitions are introduced first.

## 2.1 Notation and Definitions

For a natural number $n ,$ let $[ n ] \triangleq \{ 1 , \dots , n \}$ . For a measurable space $( \Omega , { \mathcal { F } } )$ , write $\mathcal { M } ( \Omega )$ for the Banach space of finite signed measures on $( \Omega , { \mathcal { F } } )$ , equipped with the total variation norm $\| \boldsymbol { \mu } \| \triangleq | \boldsymbol { \mu } | ( \Omega )$ . When Ω is

a topological space $( \mathrm { e . g . } , \mathbb { R } ^ { d } )$ , measures are understood to be finite Borel measures. Throughout, a measure $\mu$ on $\mathbb { R } ^ { d }$ is said to have a density $f _ { \mu }$ if

$$
\mu ( A ) = \int _ { A } f _ { \mu } ( x ) d x
$$

for every measurable set A. If $\mu$ is a measure on $\Omega$ and $f : \Omega \to \Omega ^ { \prime }$ is a measurable function into another measurable space $( \Omega ^ { \prime } , \mathcal { F } ^ { \prime } )$ , the pushforward of $\mu$ under $f ,$ denoted $f \# \mu ,$ , is defined by

$$
f \# \mu ( A ) \triangleq \mu { \big ( } f ^ { - 1 } ( A ) { \big ) } , \qquad A \in { \mathcal { F } } ^ { \prime } .
$$

For fixed $f ,$ the map $\mu \mapsto f \# \mu$ is linear.

Denote by $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { d } ) \subset \mathcal { M } ( \mathbb { R } ^ { d } )$ the class of measures whose Laplace transform exists and is finite on a neighborhood of the origin. Measures with compact support or sub-Gaussian tails lie in $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { d } )$ , so this class includes many standard nonparametric settings. A superscript +, as in $\mathcal { M } ^ { + } ( \mathbb { R } )$ , denotes the cone of finite nonnegative (unsigned) measures. For a vector $a = [ a _ { 1 } , \ldots , a _ { d } ] ^ { \top } \in \mathbb { R } ^ { d }$ , define $\pi _ { a } ( x ) \triangleq \langle a , x \rangle$ . When convenient, the same map will also be written as $\pi _ { [ a _ { 1 } , \ldots , a _ { d } ] } ( x )$ , suppressing the transpose in the subscript.

## 2.2 Injectivity of the Silhouette Operator

Theorem 1 below gives an explicit family of silhouette operators that are injective on low-rank measures admitting representations whose component marginals belong to $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } )$ . The theorem is stated below in measure-theoretic form. A probabilistic interpretation in terms of the distributions of linear combinations of random variables is given immediately after Corollary 2.

Theorem 1. Let $k \in \mathbb N$ and let $a , b _ { 1 } , b _ { 2 } \in \mathbb { R }$ be nonzero with $| a | \neq 1$ . Let $\mu , \mu ^ { \prime } \in \mathcal { M } ( \mathbb { R } ^ { 2 } )$ admit representations

$$
\boldsymbol { \mu } = \sum _ { i = 1 } ^ { k } \mu _ { i , 1 } \times \mu _ { i , 2 } , \qquad \boldsymbol { \mu } ^ { \prime } = \sum _ { i = 1 } ^ { k } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } ,\tag{3}
$$

where $\mu _ { i , j } , \mu _ { i , j } ^ { \prime } \in \mathcal { M } _ { \exp } ( \mathbb { R } )$ for all $i \in [ k ]$ and $j \in [ 2 ] , \ I f$ 7J

$$
\pi _ { [ a ^ { m } b _ { 1 } , b _ { 2 } ] } \# \mu = \pi _ { [ a ^ { m } b _ { 1 } , b _ { 2 } ] } \# \mu ^ { \prime } \qquad \mathrm { ~ f o r ~ } a l l m \in \{ 1 , \ldots , k \} ,\tag{4}
$$

$$
\pi _ { [ b _ { 1 } , a ^ { m } b _ { 2 } ] } \# \mu = \pi _ { [ b _ { 1 } , a ^ { m } b _ { 2 } ] } \# \mu ^ { \prime } \qquad f o r \ a l l \ m \in \{ 0 , \ldots , k - 1 \} ,\tag{5}
$$

then $\mu = \mu ^ { \prime } .$

Proofs for all results in this subsection can be found in Section 9. The component marginals $\left( \mathrm { e . g . } , \mu _ { i , j } \right)$ may be the zero measure, so the theorem applies to measures of rank at most $k ,$ not only those of rank exactly k. When each component marginal $\mu _ { i , j }$ has a density $f _ { \mu _ { i , j } }$ , the density of $\mu$ is

$$
f _ { \mu } ( x , y ) = \sum _ { i = 1 } ^ { k } f _ { \mu _ { i , 1 } } ( x ) f _ { \mu _ { i , 2 } } ( y ) .
$$

When applying Theorem 1 in the context of estimation, if $\mu$ is a target measure with $\mu _ { i , j } \in \mathcal { M } _ { \exp } ( \mathbb { R } )$ and $\mu ^ { \prime }$ is an estimator, the component marginals $\mu _ { i , j } ^ { \prime }$ must also belong to $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } )$ for equality of their silhouettes to imply $\mu = \mu ^ { \prime }$ . The tail condition in the signed setting cannot be substantially weakened: Lemma 10 shows that the nullspace of any silhouette operator formed from finitely many projection directions contains a nonzero rank-one signed measure whose component marginals have smooth densities that decay faster than any polynomial. The required tail condition can nevertheless be ensured simply by working on a fixed compact interval, such as $[ - 1 , 1 ]$ . In the nonnegative setting, however, no corresponding tail condition on the estimator is required: provided $\mu _ { i , j } \in \mathcal { M } _ { \mathrm { e x p } } ^ { + } ( \mathbb { R } )$ , Corollary 2 allows $\mu _ { i , j } ^ { \prime }$ to be arbitrary elements of $\mathcal { M } ^ { + } ( \mathbb { R } )$

Corollary 2. The conclusion of Theorem 1 remains valid under the assumptions

$$
\mu _ { i , j } \in \mathcal { M } _ { \mathrm { e x p } } ^ { + } ( \mathbb { R } ) , \qquad \mu _ { i , j } ^ { \prime } \in \mathcal { M } ^ { + } ( \mathbb { R } ) ,
$$

for all $i \in [ k ]$ and $j \in [ 2 ]$

For a probabilistic interpretation, suppose that $\mu$ and $\mu ^ { \prime }$ are probability measures satisfying the assumptions of Corollary 2, and le

$$
[ X , Y ] ^ { \top } \sim \mu , \qquad [ X ^ { \prime } , Y ^ { \prime } ] ^ { \top } \sim \mu ^ { \prime } .
$$

Let $v _ { 1 } , \ldots , v _ { 2 k }$ denote the 2k directions in $( 4 )  { - } ( 5 )$ . Then $\mu = \mu ^ { \prime }$ if and only if, for every $i \in [ 2 k ]$ , the inner products

$$
\langle v _ { i } , [ X , Y ] ^ { \top } \rangle \quad { \mathrm { a n d } } \quad \langle v _ { i } , [ X ^ { \prime } , Y ^ { \prime } ] ^ { \top } \rangle
$$

have the same distribution.

The projection count 2k in Theorem 1 and Corollary 2 is optimal. Theorem 8 shows that, for any collection of $2 k - 1$ projection directions, there exist two distinct probability measures of mixture-of-products form, each with at most k components, whose projections agree along every prescribed direction, even when all component marginals have smooth, compactly supported densities. Proposition 9 complements this projection-count lower bound by showing that the geometry of the directions also matters. For any distinct $a _ { 1 } , \ldots , a _ { k } > 0$ , the 2k symmetrically paired directions $[ a _ { r } , 1 ] ^ { \top }$ and $[ - a _ { r } , 1 ] ^ { \top } , r \in [ k ]$ , fail to identify mixture-of-products models with as few as $\lfloor k / 2 \rfloor + 1$ components. Identifiability also depends on the tail behavior of the component marginals when $k \geq 2$ . Indeed, Corollary 11 shows that, for any finite collection of projection directions, there exist two distinct probability measures of mixture-of-products form, each with two components, whose projections agree along every prescribed direction, even though all component marginals have densities that decay faster than any polynomial.

If desired, potential tail issues can be mitigated by applying a coordinate-wise transformation that maps R into a compact set, such as arctan. More generally, in d dimensions, let $f _ { 1 } , \ldots , f _ { d } : \mathbb { R } \to$ R and define $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ by

$$
f ( x ) = ( f _ { 1 } ( x _ { 1 } ) , \ldots , f _ { d } ( x _ { d } ) ) .
$$

Then

$$
\begin{array} { r } { f \# \left( \displaystyle \sum _ { i = 1 } ^ { k } \prod _ { j = 1 } ^ { d } \mu _ { i , j } \right) = \displaystyle \sum _ { i = 1 } ^ { k } f \# \left( \prod _ { j = 1 } ^ { d } \mu _ { i , j } \right) } \\ { = \displaystyle \sum _ { i = 1 } ^ { k } \prod _ { j = 1 } ^ { d } f _ { j } \# \mu _ { i , j } , } \end{array}\tag{6}
$$

so coordinate-wise transformations preserve the sum-of-products structure. In the probabilistic setting, one may therefore use $f$ to map the data into a compact set, estimate the transformed model in (6), and, when appropriate, map back via $f ^ { - 1 }$ . Notably, when f is invertible, the linear projections defining the silhouette operator in Theorem 1 may be replaced by generally nonlinear maps of the form $\pi _ { v } \circ f$

## 2.3 Topological and Geometric Properties of the Silhouette Operator

The preceding results establish injectivity of the silhouette operator on low-rank measures. Beyond injectivity, it is natural to ask whether, and under what additional conditions, convergence of silhouettes guarantees convergence of the underlying measures. This subsection examines the topological and geometric structure induced by the silhouette operator. Proofs of all results in this subsection can be found in Section 10.

For $k \in \mathbb N$ , let $S _ { 2 k }$ denote the set of linear operators $V : \mathcal { M } ( \mathbb { R } ^ { 2 } ) \to \mathcal { M } ( \mathbb { R } ) ^ { 2 k }$ obtained from Theorem 1 using the 2k projection directions in (4)–(5). Thus, each choice of admissible parameters $( a , b _ { 1 } , b _ { 2 } )$ determines an element $V \in S _ { 2 k }$ . For measurable spaces $\Omega _ { 1 } , \Omega _ { 2 }$ , with $\Omega _ { 1 } \times \Omega _ { 2 }$ equipped with the product σ-algebra, define

$$
\mathcal { R } _ { \leq k } ( \Omega _ { 1 } , \Omega _ { 2 } ) \subseteq \mathcal { M } ( \Omega _ { 1 } \times \Omega _ { 2 } )
$$

to be the set of finite signed measures that can be written in the form

$$
\mathcal { R } _ { \leq k } ( \Omega _ { 1 } , \Omega _ { 2 } ) = \left\{ \sum _ { i = 1 } ^ { k } \mu _ { i , 1 } \times \mu _ { i , 2 } : \ \mu _ { i , j } \in \mathcal { M } ( \Omega _ { j } ) \right\} .
$$

This is the class of measures of rank at most k. A measure is said to have rank exactly k if k is the smallest integer for which it belongs to $\mathscr { R } _ { \leq k } ( \Omega _ { 1 } , \Omega _ { 2 } )$ . By Proposition 13, this is equivalent to admitting a representation with k terms in which $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are each linearly independent.

To study continuity properties of the silhouette operator, it is necessary to specify the ambient topologies. For a measurable space $\Omega ,$ the product space $\mathcal { M } ( \Omega ) ^ { \ell }$ is equipped with the standard product topology. When $\mathcal { M } ( \Omega )$ carries its weak<sup>∗</sup> topology, the induced product topology on $\mathcal { M } ( \Omega ) ^ { \ell }$ will also be referred to as the weak<sup>∗</sup> topology. This convention is natural and consistent with the identification of duals of finite direct sums; see [20, Proposition 5.5]. When each component is instead equipped with the total variation norm, the induced product topology will be referred to as the norm topology.

The next result shows that, for compact $K _ { 1 } , K _ { 2 } \subset \mathbb { R }$ , silhouette operators act as weak homeomorphisms on total-variation-bounded subsets of $\mathcal { R } { _ { \le k } } ( K _ { 1 } , K _ { 2 } )$ and, in the language of compressed sensing, can be viewed as “restricted weak<sup>∗</sup> homeomorphisms.” For a normed space $X$ , let B denote its closed unit ball.

Theorem 3. Let $K _ { 1 } , K _ { 2 } \subset \mathbb { R }$ be compact, let $k \in \mathbb N$ , let $V \in S _ { 2 k }$ , and fix $r > 0$ . Then the restriction of V to

$$
\mathscr { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } ) \cap r \mathbb { B } _ { \mathcal { M } ( K _ { 1 } \times K _ { 2 } ) }
$$

is a $w e a k ^ { * } - w e a k ^ { * }$ homeomorphism onto its image.

The boundedness assumption cannot in general be omitted. Theorem 12 shows that, for any finite collection of projection directions, the associated map is not a weak<sup>∗</sup>–weak<sup>∗</sup> homeomorphic embedding of $\mathcal { R } _ { \leq 1 } ( [ 0 , 1 ] , [ 0 , 1 ] )$ , regardless of how many directions are included. This remains true even when the domain is restricted to measures with $C ^ { \infty }$ densities.

Theorem 3 has an important practical consequence for probability measures. Every probability measure has total variation norm one, and weak<sup>∗</sup> convergence of probability measures coincides with convergence in distribution. Consequently, let $\mu$ and $( \hat { \mu } _ { i } ) _ { i = 1 } ^ { \infty }$ be probability measures in $\mathcal { R } { _ { \leq k } } ( K _ { 1 } , K _ { 2 } )$ . If each component of $V ( \hat { \mu } _ { i } )$ converges in distribution to the corresponding component of $V ( \mu )$ , then $\hat { \mu } _ { i }$ converges in distribution to $\mu$

At the population level, injectivity also admits a familiar optimization formulation. If d is any discrepancy on $\mathcal { M } ( \mathbb { R } ) ^ { 2 \bar { k } }$ that vanishes exactly when its arguments agree, then $\mu$ is the unique rank-≤ k solution of

$$
\operatorname* { m i n } _ { \nu \in \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } ) } { \mathsf { d } } \big ( V ( \nu ) , V ( \mu ) \big ) .
$$

A natural estimation strategy is therefore to replace the unknown silhouette $V ( \mu )$ by the silhouette of the empirical measure and minimize the resulting loss over low-rank measures. The key computational advantages of the silhouette framework stem from its reduction of high-dimensional estimation problems to collections of one-dimensional projections, the central computational principle underlying sliced methods. Many one-dimensional statistical procedures, including Wasserstein distance computations, admit eficient ${ \bar { O } } ( n )$ algorithms. Moreover, computations associated with diferent projection directions can be performed independently, making the resulting procedures naturally parallelizable. Section 5 develops a practical estimation framework based on this strategy.

The proof of Theorem 3 relies on a structural result for finite-rank tensor products in Banach spaces, established later as Theorem 29, which gives the weak<sup>∗</sup> closedness of $\mathrm { r a n k } - \le k$ measures needed here. This result extends related work in [21] and may be of independent interest.

Silhouette operators also satisfy a restricted isometry-type property on finite-dimensional subspaces. In particular, after rescaling, they preserve distances up to a uniform distortion factor on the intersection of any finite-dimensional subspace with the class of $\mathrm { r a n k } - \le k$ measures. This is captured in the following proposition. In what follows, $\mathcal { M } ( \mathbb { R } ) ^ { 2 k }$ may be equipped with any product norm induced by the total variation norm.

Proposition 4. Let $K _ { 1 } , K _ { 2 } \subset \mathbb { R }$ be compact, let $k \in \mathbb N$ , let S be a finite-dimensional linear subspace of $\mathcal { M } ( K _ { 1 } \times K _ { 2 } )$ , and let $V \in S _ { 2 k }$ . Then there exist $\gamma > 0$ and $\delta \in [ 0 , 1 )$ such that

$$
\begin{array} { r } { \left( 1 - \delta \right) \| \mu - \mu ^ { \prime } \| \leq \| \gamma V ( \mu - \mu ^ { \prime } ) \| \leq ( 1 + \delta ) \| \mu - \mu ^ { \prime } \| , } \end{array}
$$

for all $\mu , \mu ^ { \prime } \in S \cap \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$

An explicit parallel with matrix RIP can be drawn when the component marginals are restricted to finite-dimensional subspaces. Let $S _ { j } \subset \mathcal { M } ( K _ { j } )$ have dimension $d _ { j }$ , and let S be the span of the product measures $\eta _ { 1 } \times \eta _ { 2 }$ with $\eta _ { j } \in S _ { j }$ . After choosing bases for $S _ { 1 }$ and $S _ { 2 }$ , the space S can be identified $\mathrm { w i t h ~ } \mathbf { \bar { \mathbb { R } } } ^ { d _ { 1 } \times d _ { 2 } }$ with $S \cap \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ corresponding to matrices of rank at most k. In these coordinates, the inequality in Proposition 4 takes the same form as the classical RIP for low-rank matrices.

## 3 Higher-Dimensional Extensions of the Silhouette Framework

The previous section established that, in two dimensions, a low-rank measure is uniquely determined by its silhouette. The present section extends this framework to higher-dimensional mixture-of-products models of the form (2), where $w _ { i } > 0 , \sum _ { i = 1 } ^ { k } w _ { i } = 1$ , and each $\mu _ { i , j }$ is a probability measure. The section identifies conditions under which such measures can be recovered from collections of pairwise marginals of the form

$$
\sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , j _ { 1 } } \times \mu _ { i , j _ { 2 } } .
$$

The statements are formulated for general measurable spaces Ω, though the primary case of interest remains $\Omega = \mathbb { R }$ . Proofs of all results in this section can be found in Section 11.

To express component identifiability as an exact equality without imposing an arbitrary ordering on the components, finite mixtures of product measures are represented as probability measures on the space of d-tuples of component marginals. For a measurable space $( \Omega , { \mathcal { F } } )$ , let ${ \mathcal { P } } ( \Omega )$ denote the space of probability measures on Ω. The space ${ \mathcal { P } } ( \Omega ) ^ { d }$ is equipped with the discrete σ-algebra, so that all singleton sets are measurable. An element $( \mu _ { 1 } , \dots , \mu _ { d } ) \in \mathcal { P } ( \Omega ) ^ { d }$ corresponds to the collection of marginals defining a product component $\textstyle \prod _ { j = 1 } ^ { d } \mu _ { j }$

For finite linear combinations of point masses on ${ \mathcal { P } } ( \Omega ) ^ { d }$ , define the linear map R by

$$
R \left( \sum _ { i = 1 } ^ { k } a _ { i } \delta _ { ( \mu _ { i , 1 } , \dots , \mu _ { i , d } ) } \right) \triangleq \sum _ { i = 1 } ^ { k } a _ { i } \prod _ { j = 1 } ^ { d } \mu _ { i , j } .
$$

For $J \subseteq [ d ]$ , write its elements in increasing order as $j _ { 1 } < \cdots < j _ { \ell }$ and define the linear map $R _ { J }$ by

$$
R _ { J } \left( \sum _ { i = 1 } ^ { k } a _ { i } \delta _ { ( \mu _ { i , 1 } , \dots , \mu _ { i , d } ) } \right) \triangleq \sum _ { i = 1 } ^ { k } a _ { i } \prod _ { r = 1 } ^ { \ell } \mu _ { i , j _ { r } } .
$$

The identifiability results below use linear independence together with two additional notions for collections of probability measures: irreducibility and singularity. Let $\mu _ { 1 } , \ldots , \mu _ { k }$ be probability measures on a common measurable space. A measure $\mu _ { 1 }$ is said to be irreducible with respect to $\mu _ { 2 } , \ldots , \mu _ { k }$ if the following holds: whenever $\textstyle \sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i }$ is a probability measure for some real coeficients $w _ { i } .$ , it must be the case that $w _ { 1 . } \geq 0$ . Equivalently, $\mu _ { 1 }$ is irreducible with respect to the others if, for every $w > 0 ,$ , the signed measure $\textstyle \sum _ { i = 2 } ^ { k } \mu _ { i } - w \mu _ { 1 }$ has a nonzero negative part. If each $\mu _ { i }$ is irreducible with respect to all the others, the collection is called jointly irreducible [8, 9]. Finally, $\mu _ { 1 }$ will be called singular with respect to $\mu _ { 2 } , \ldots , \mu _ { k }$ if there exists a measurable set A such that $\mu _ { 1 } ( A ) = 1$ and $\mu _ { i } ( A ) = 0$ for every $i \in \{ 2 , \ldots , k \}$

Irreducibility is a measure-theoretic analogue of the “support condition” or “anchor-word” assumption in nonnegative matrix factorization [22, 23]. For example, $\mu _ { 1 }$ is irreducible if there exists a measurable set A such that $\mu _ { 1 } ( A ) > 0$ while $\mu _ { i } ( A ) = 0$ for every $i \in \{ 2 , \ldots , k \}$ . On a finite discrete space, where measures can be identified with probability vectors, this condition is also necessary. Joint irreducibility is stronger than linear independence, while singularity is stronger than irreducibility.

The results below give structural conditions under which the component marginals and weights in the mixture-of-products model (2) can be recovered from a collection of pairwise marginals. Such a collection can be encoded by a graph $G = ( [ d ] , E )$ , where $\{ j _ { 1 } , j _ { 2 } \} \in E$ indicates that the corresponding pairwise marginal is included in the collection. Within the silhouette framework, each of these marginals can itself be identified from its silhouette. The weaker problem of recovering the mixture $\mu$ itself from such pairwise marginals, without identifying its component marginals and weights, remains open.

Outside of trivial cases, the marginal graph must at least be connected to guarantee recovery of $\mu ;$ if the graph is disconnected, at least two groups of coordinates remain uncoupled and the full model cannot in general be identified. The complete graph yields the strongest constraints but requires estimating $\Theta ( d ^ { 2 } )$ pairwise marginals, whereas a connected graph may have as few as d − 1 edges. Thus, depending on the choice of marginal graph, the number of pairwise marginals that must be estimated ranges from linear to quadratic in $d ,$ with a corresponding impact on computational cost in practical implementations.

Theorem 5 shows that, when the true component marginals are linearly independent along each coordinate, identifying them along a single coordinate sufices to identify the full model from any connected graph of pairwise marginals. The linear independence assumption only needs to hold for the true model and does not need to be imposed as a constraint during estimation. Related propagation phenomena under linear independence assumptions also appear in finite-alphabet tensor models for recovering high-dimensional probability distributions from low-order marginals [24].

Theorem 5. Let $( \Omega , { \mathcal { F } } )$ be a measurable space, let $k , d \in \mathbb { N }$ and $d \ge ~ 2$ . For $i \in [ k ]$ and $j ~ \in ~ [ d ]$ , let $\mu _ { i , j } , \mu _ { i , j } ^ { \prime } \in \mathscr { P } ( \Omega )$ , and let $w _ { i } > 0 , w _ { i } ^ { \prime } \geq 0$ satisfy $\begin{array} { r } { \sum _ { i = 1 } ^ { k } w _ { i } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } = 1 } \end{array}$

Assume that for every $j \in [ d ]$ , the measures $\mu _ { 1 , j } , \ldots , \mu _ { k , j }$ are linearly independent.

Let $G = ( [ d ] , E )$ be a connected graph. Suppose that

$$
R _ { e } \left( \sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \dots , \mu _ { i , d } ) } \right) \ = \ R _ { e } \left( \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \dots , \mu _ { i , d } ^ { \prime } ) } \right)
$$

for every $e \in E$

Suppose further that for some $d ^ { \prime } \in [ d ]$

$$
\{ \mu _ { 1 , d ^ { \prime } } , \ldots , \mu _ { k , d ^ { \prime } } \} = \{ \mu _ { 1 , d ^ { \prime } } ^ { \prime } , \ldots , \mu _ { k , d ^ { \prime } } ^ { \prime } \} .
$$

Then

$$
\sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , . . . , \mu _ { i , d } ) } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , . . . , \mu _ { i , d } ^ { \prime } ) } .\tag{7}
$$

Each summand $w _ { i } \delta _ { ( \mu _ { i , 1 } , \dots , \mu _ { i , d } ) }$ in (7) corresponds to the weighted product component $\begin{array} { r } { w _ { i } \prod _ { j = 1 } ^ { d } \mu _ { i , j } } \end{array}$ of the associated mixture-of-products model. Thus, (7) states not only that the resulting mixtures $\mu$ and $\mu ^ { \prime }$ are equal (by applying R to both sides), but that their underlying components and weights agree up to a permutation of indices, i.e., the model is identifiable. The assumption that $\mu _ { 1 , j } , \ldots , \mu _ { k , j }$ are linearly independent for each $j \in [ d ]$ is a standard and relatively mild structural condition [25]; for example, collections of one-dimensional Gaussian measures with distinct parameter pairs are linearly independent [26].

Theorem 5 shows that once the components along one coordinate have been recovered, the remaining components can be identified by propagating through the marginal graph. It does not, however, show how to obtain that initial set of components from the observed pairwise marginals; they must already be known or have been recovered by some other means. For example, if the component marginals along that coordinate are Gaussian, this initial set could be estimated using standard univariate mixture-model methods. In a fully nonparametric setting, it is desirable for uniqueness to be intrinsic—namely, that any factorization consistent with an observed pairwise marginal must coincide with the true one, without imposing structural conditions on the candidate factorization. The next result shows that joint irreducibility of the true component families provides this property for a pair of coordinates.

Proposition 6. Let $( \Omega , { \mathcal { F } } )$ be a measurable space and let $k \in \mathbb N$ . For $i \in [ k ]$ and $j \in [ 2 ]$ , let $\mu _ { i , j } , \mu _ { i , j } ^ { \prime } \in \mathscr { P } ( \Omega )$ and let $\begin{array} { r } { w _ { i } > 0 , w _ { i } ^ { \prime } \geq 0 \ s a t i s f y \sum _ { i = 1 } ^ { k } w _ { i } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } = 1 } \end{array}$

Assume that the families $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are jointly irreducible.

If

$$
R \biggl ( \sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \mu _ { i , 2 } ) } \biggr ) \ = \ R \biggl ( \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \mu _ { i , 2 } ^ { \prime } ) } \biggr ) \ ,
$$

then

$$
\sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \mu _ { i , 2 } ) } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \mu _ { i , 2 } ^ { \prime } ) } .
$$

The previous proposition identifies a setting in which the true mixture components are uniquely determined from a single pairwise marginal. In conjunction with Theorem 5, identifiability on one pair of coordinates can then propagate through the connected marginal graph to recover the full model.

A limitation of Proposition 6 is that joint irreducibility must hold across every component for the same pair of coordinates. The next theorem shows that, under a stronger singularity assumption, a single component can instead be identified when it is separated from all the others along a single pair of coordinates, and its identification can then be propagated through the remaining pairwise marginals.

Theorem 7. Let $( \Omega , { \mathcal { F } } )$ be a measurable space and let k $, d \in \mathbb { N }$ with $d \geq 2$ . For $i \in [ k ]$ and $j \in [ d ]$ , let $\mu _ { i , j } , \mu _ { i , j } ^ { \prime } \in \mathscr { P } ( \Omega )$ , and let $w _ { i } > 0 , w _ { i } ^ { \prime } \geq 0$ satisfy $\begin{array} { r } { \sum _ { i = 1 } ^ { k } w _ { i } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } = 1 } \end{array}$

Assume that for every $j \in [ d ]$ , the measures $\mu _ { 1 , j } , \ldots , \mu _ { k , j }$ are linearly independent, and that $f o r \ j \in [ 2 ]$ the measure $\mu _ { 1 , j }$ is singular with respect to $\mu _ { 2 , j } , \ldots , \mu _ { k , j }$

Let $G = ( [ d ] , E )$ be a graph such that $\{ 1 , 2 \} \in E$ and every $j > 2$ is adjacent to at least one of 1 or 2. If

$$
R _ { e } \left( \sum _ { i = 1 } ^ { k } w _ { i } \delta _ { \left( \mu _ { i , 1 } , \dots , \mu _ { i , d } \right) } \right) = R _ { e } \left( \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { \left( \mu _ { i , 1 } ^ { \prime } , \dots , \mu _ { i , d } ^ { \prime } \right) } \right)
$$

for every $e \in E$ , then

$$
\sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \dots , \mu _ { i , d } ^ { \prime } ) } \big ( \{ ( \mu _ { 1 , 1 } , \dots , \mu _ { 1 , d } ) \} \big ) = w _ { 1 } .
$$

The conclusion of Theorem 7 implies that any representation of the form $\begin{array} { r } { \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \prod _ { j = 1 } ^ { d } \mu _ { i , j } ^ { \prime } } \end{array}$ consistent with the observed pairwise marginals must assign total weight w to the component $\textstyle \prod _ { j = 1 } ^ { d } \mu _ { 1 , j }$ . Theorem 7 is most relevant when a rich collection of pairwise marginals, such as the complete graph, is available and, for each component, there exist at least two coordinates along which the singularity condition holds. This condition need not be imposed on the candidate factorization; rather, the theorem shows that any such factorization consistent with the observed pairwise marginals must contain the identified component.

The conditions above provide identifiability guarantees in specific regimes, but are unlikely to be exhaustive. In particular, identifiability may emerge from interactions among several structural properties of the components and the connectivity pattern of the marginal graph that are not captured by any single explicit condition, such as irreducibility or singularity.

## 4 Related Work

This work lies at the intersection of several areas. The two primary connections are to mixture-of-products models and restricted isometry-type conditions arising in compressed sensing. A final subsection discusses several additional connections, including projection-based methods, optimal transport, and inverse problems.

## 4.1 Mixture-of-Products Models

Mixture-of-products models are statistical models of the form (2). They admit a natural interpretation in which the d covariates become conditionally independent after conditioning on an unobserved discrete random variable with distribution $( w _ { 1 } , \ldots , w _ { k } )$ . Such models are also naturally viewed as nonparametric mixture models. When the component measures are supported on finite sets, (2) reduces to a low-rank nonnegative tensor model, connecting this setting to the literatures on tensor factorization, latent variable models, and topic modeling [24, 25, 27, 28]. The discussion first focuses on models whose component marginals $\mu _ { i , j }$ have densities.

Early work on mixture-of-products models in the continuous nonparametric setting includes [11, 12], where consistent estimators for the mixture components were introduced. These works also observed that the product structure can yield improved statistical rates. Since then, a number of estimation methods for this model class have been proposed [13–18, 29–32], many of which are based on matrix or tensor factorization techniques.

Much of the literature treats (2) as a nonparametric mixture model, with a corresponding focus on identifiability. A large body of work studies identifiability of such models under structural conditions on the collections $\mu _ { 1 , j } , \ldots , \mu _ { k , j }$ , typically imposing conditions based on linear independence or on stronger forms of separation [17, 24, 25, 32, 33]. At the same time, the product structure can be exploited to improve statistical rates. In [13], it was shown that there exists an estimator achieving rate $\widetilde { \cal O } ( n ^ { - 1 / 3 } )$ in $L ^ { 1 }$ for densities of this form, assuming the component marginals $\mu _ { i , j }$ admit Lipschitz continuous densities. That work also considers related Tucker-style factorizations for multivariate densities together with associated sample complexity bounds. The $\widetilde { O } ( n ^ { - 1 / 3 } )$ rate matches, up to logarithmic factors, the minimax rate for one dimensional Lipschitz continuous density estimation, whereas general d-dimensional Lipschitz continuous densities have minimax rate $O ( n ^ { - 1 / ( 2 + d ) } )$ ; more generally, β-H¨older continuous densities have minimax rate $O ( n ^ { - \beta / ( 2 \beta + d ) } )$ [34–36]. This indicates that the product structure can mitigate the curse of dimensionality.

More recently, [16] proposed a practical estimator in the two-dimensional setting based on a multiscale decomposition of a histogram estimator combined with nuclear-norm regularization via singular value decomposition. That estimator achieves rate ${ \widetilde O } ( n ^ { - \beta / ( 2 \beta + 1 ) } )$ for β-H¨older densities of the above form with $0 < \beta \le 1$ , which, up to logarithmic factors, matches the minimax rate for one-dimensional density estimation. In [17], mixture-of-products density estimation was studied in the general d-dimensional setting, where the same ${ \widetilde O } ( n ^ { - \beta / ( 2 \beta + 1 ) } )$ rate is achieved for densities of this form. Under linear independence assumptions similar to those used in Section 3, that work also establishes recovery guarantees for the mixture components and analyzes how the resulting sample complexity depends on incoherence properties of the component marginals.

Every a.e.-continuous density admits a countable mixture-of-products representation with smooth Lipschitz component marginals [37]. The resulting non-negative Lipschitz spectrum quantifies density complexity through the decay of the mixture weights and the growth of the marginal Lipschitz constants. Because such a spectrum exists throughout this broad class, the resulting theory yields finite-sample bounds and, under suitable decay and growth conditions, dimension-independent convergence rates without requiring an exact finite mixture-of-products representation. This countable representation also places Theorem 1 in a natural relationship with countable-projection versions of the Cram´er–Wold theorem [38–40]. For each finite $k ,$ Theorem 1 shows that 2k projections sufice to identify a rank-≤ k measure. As k grows, both the number of mixture components and the number of required projections become countable, paralleling the fact that, under suitable regularity conditions, countably many projections sufice to identify a general probability distribution.

A closely related setting arises in grouped-observation models of the form

$$
\sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i } ^ { \times d } ,\tag{8}
$$

which may be viewed as a symmetric specialization of (2), where the component measures $\mu _ { i }$ may themselves be arbitrary probability measures on a general measurable space. Identifiability and estimation for such models have been studied in [27, 41–46] and are closely connected to topic models such as latent Dirichlet allocation (LDA) [47]. The identifiability results for models of the form (8) are directly related to corresponding results for LDA in [48].

## 4.2 Restricted Isometry and Low-Rank Matrix Recovery

Restricted isometry and related properties play a central role in compressed sensing and low-rank matrix recovery, where they provide geometric guarantees for stable reconstruction from linear measurements. For a vector $x \in \mathbb { R } ^ { n }$ , let $\| x \| _ { 0 }$ denote its sparsity, i.e., the number of nonzero entries of $x .$

The original formulation of the restricted isometry property (RIP) [1] concerns sparse vectors. More specifically, a matrix $A \in \mathbb { R } ^ { m \times n }$ is said to satisfy the vector-RIP of order k if there exists $\delta \in [ 0 , 1 )$ such that for all $x \in \mathbb { R } ^ { n }$ with $\| x \| _ { 0 } \leq k$

$$
( 1 - \delta ) \| x \| _ { 2 } ^ { 2 } \leq \| A x \| _ { 2 } ^ { 2 } \leq ( 1 + \delta ) \| x \| _ { 2 } ^ { 2 } .\tag{9}
$$

Subsequent work extended this framework to low-rank matrix recovery. In this setting, a linear operator $T : \mathbb { R } ^ { m \times n }  \mathbb { R } ^ { \ell }$ is said to satisfy the matrix-RIP [2] of order k if there exists $\delta \in [ 0 , 1 )$ such that for all $\boldsymbol { X } \in \mathbb { R } ^ { m \times n }$ with rank at most $k ,$

$$
( 1 - \delta ) \| \boldsymbol { X } \| _ { F } ^ { 2 } \leq \| \boldsymbol { T } ( \boldsymbol { X } ) \| _ { 2 } ^ { 2 } \leq ( 1 + \delta ) \| \boldsymbol { X } \| _ { F } ^ { 2 } .\tag{10}
$$

Vector-RIP can hold with m much smaller than $n ,$ and matrix-RIP with ℓ much smaller than mn. In both cases, the measurement map substantially reduces dimension while approximately preserving norms on the restricted class. The framework has also been extended to higher-order objects through notions such as tensor-RIP [49].

Restricted strong convexity is a closely related notion, which appears in several forms in the literature. Following the operator formulation in [3], a linear operator $T : \mathbb { R } ^ { m \times n }  \mathbb { R } ^ { \ell }$ is said to satisfy restricted strong convexity (RSC) on a given class of matrices if there exists $\kappa > 0$ such that, for every X in that class,

$$
\| T ( X ) \| _ { 2 } ^ { 2 } \geq \kappa \| X \| _ { F } ^ { 2 } .
$$

The normalization in [3] is absorbed into κ. Injectivity of $T$ on rank-≤ k matrices is equivalent to restricted strong convexity on $\mathrm { r a n k } . . \leq 2 k$ matrices and, after rescaling $T ,$ to matrix-RIP of order $2 k ;$ see Appendix A for a proof.

RIP and closely related conditions have found numerous applications and are central to the field of compressed sensing [50, 51], which emerged as a major development in signal processing in the 2000s. A fundamental question in this area is the construction of operators satisfying these properties. A canonical approach to establishing vector-RIP (9) is to use random matrices with independent, appropriately scaled, zero-mean Gaussian entries, which satisfy RIP with high probability for suficiently large $m ,$ depending on $n , k ,$ and δ [1, 52]. To prove this guarantee, one shows, via concentration inequalities and covering numbers, that such a matrix is an approximate isometry on any fixed k-dimensional subspace with failure probability exponentially small in $m .$ . Although there are combinatorially many coordinate subspaces corresponding to k-sparse vectors, a union bound combined with exponential concentration allows one to control all such subspaces simultaneously.

Gaussian entries are not essential for vector-RIP: appropriately normalized random matrices with independent, mean-zero sub-Gaussian entries also satisfy RIP with high probability [52]. Sparse random constructions are also possible, generally at the cost of additional measurements or weaker isometry and recovery guarantees [53]. Analogous probabilistic constructions extend to matrix-RIP (10), with measurement counts near the degrees of freedom of low-rank matrices [2].

Obtaining comparably strong classical RIP guarantees from explicit deterministic constructions remains challenging. A related dificulty is that deciding whether a given matrix satisfies vector-RIP is NP-hard [54]. The silhouette operator considered here is explicit and deterministic, and its restricted-isometrytype property on finite-dimensional subspaces of measures is established through a fundamentally diferent approach, without randomization, concentration inequalities, or covering-number arguments.

## 4.3 Other Related Topics

Silhouette operators can be viewed as a kind of analogue of RIP operators for low-rank measures rather than low-rank matrices. The framework also connects to several other areas, including projection-based methods, optimal transport, and inverse problems.

Graphical models provide another way to exploit lower-dimensional structure in density estimation. In a Markov random field, a graph encodes conditional independence relations among the observed coordinates. For tree- and forest-structured models, the joint density is determined by the univariate marginals and the pairwise marginals along the graph edges, leading to density estimators built from one- and two-dimensional marginal estimates [55, 56]. More generally, Markov assumptions can reduce the efective dimension of nonparametric density estimation [57–60]. In the present work, the marginal graph instead specifies the pairwise marginals used to identify the latent mixture-of-products structure, rather than conditional independence relations among the observed coordinates.

A central computational motivation for the silhouette framework is to reduce the cost of Wasserstein distance computations. By measuring the cost of moving probability mass between locations, Wasserstein distance incorporates the geometry of the underlying space into its notion of distance [61]. In contrast, tota variation, $L ^ { 2 }$ distance, and KL divergence do not account for the distance between locations, so even a small spatial shift can produce a large discrepancy. However, for empirical measures in dimension $d \geq 2 .$ , exact computation of Wasserstein distance involves a discrete optimal transport problem with $O ( n ^ { 2 } )$ transport variables for datasets of size $n ,$ leading to quadratic memory requirements and typically superquadratic running time. In one dimension, by contrast, Wasserstein distance can be computed in O(n log n) time via sorting.

One approach to reducing this computational cost is the sliced Wasserstein distance [4, 5]. It is constructed by projecting probability measures onto one-dimensional subspaces, computing Wasserstein distances between the resulting projected probability measures, and aggregating these distances over a collection of projections. By the Cram´er–Wold theorem, suficiently rich collections of projections can distinguish probability measures; related topological and statistical properties of sliced probability metrics have been studied in [6, 62, 63].

Many works have developed sliced Wasserstein methods for statistical and machine learning tasks [7, 64– 68]. For mixture modelling, a sliced-Wasserstein approach to fitting Gaussian mixture models was proposed in [69]. It has been shown that a d-dimensional Gaussian mixture model with m components can be identified from a finite number of projections, with $( 2 m - 1 ) ( d ^ { 2 } + d - 2 ) / 2 + 1$ projections suficing [70]. Related results establish identifiability of other classes, such as elliptical distributions, from $( d ^ { 2 } + d ) / 2$ projections [71].

These finite-projection results are conceptually related to the present work because they likewise identify an underlying distribution from a projection-based representation. When an elliptical distribution has a density, the density has the form $f ( x ) \propto g { \big ( } ( x - \mu ) ^ { \top } \Sigma ^ { - 1 } ( x - \mu ) { \big ) }$ . When second moments exist, $\mu$ is its mean, and Σ is taken to be its covariance matrix, with the corresponding scale absorbed into $g .$ Thus, although $g$ is arbitrary, the finite-projection identification problem principally concerns recovering the finite-dimensional quantities $\mu$ and Σ. Once these are known, a single projection determines $g .$ The count in [71] reflects this: $( d ^ { 2 } + d ) / 2$ is the dimension of the space of symmetric d × d matrices, while $\mu$ can be recovered from a linearly independent subcollection of d projections.

In contrast, the nonparametric freedom in the present work resides in its rank-one factors, rather than in a single shared one-dimensional profile combined with finite-dimensional location and shape parameters. For the silhouette operator, 2k projections sufice to identify a rank-≤ k measure, with each projection yielding a finite signed measure. The number of projected measures, 2k, matches the number of component marginal measures $\mu _ { i , j } , ~ i ~ \in ~ [ k ] , ~ j ~ \in ~ [ 2 ]$ , appearing in the underlying representation (cf. (1)). Thus, rather than recovering a finite collection of scalar parameters, each projection can be viewed as recovering a measurevalued degree of freedom.

A further connection arises with Radon-type transforms [72]. Classically, Radon transforms associate to a function its integrals over afine subspaces, with corresponding extensions to measures. These transforms are closely related to the projection-based constructions discussed above. In particular, projecting a measure onto one-dimensional subspaces can be viewed as a Radon-type operation, a connection that has been observed in the sliced Wasserstein literature [5]. Radon transforms play a central role in inverse problems, especially in computed tomography, where one seeks to reconstruct a high-dimensional object from a collection of projections [73]. In such settings, it is typically infeasible to collect a complete set of projections, and recovery relies on structural assumptions or regularization. The results here suggest a possible connection to this setting, in that low-rank structure provides a mechanism for recovery from a finite set of projections. Silhouette projections of mixture-of-products models are also closely related to blind deconvolution and bilinear inverse problems [74, 75]. For $c \in \mathbb { R }$ , let $T _ { c } : \mathbb { R }  \mathbb { R }$ denote the scaling map $T _ { c } ( x ) = c x$ . If $a = [ a _ { 1 } , \ldots , a _ { d } ] ^ { \top } \in \mathbb { R } ^ { d }$ and $\mu$ is of the form (2), then

$$
\pi _ { a } \# \mu = \sum _ { i = 1 } ^ { k } w _ { i } T _ { a _ { 1 } } \# \mu _ { i , 1 } * \cdot \cdot \cdot * T _ { a _ { d } } \# \mu _ { i , d } .
$$

Recovering the component marginals from projected measures therefore combines mixture identification with a deconvolution problem. Deconvolution is statistically dificult: optimal rates depend strongly on the smoothness of the error distribution. Even in the classical setting where the convolution kernel is known and Gaussian, optimal rates are logarithmic [76]. This provides one motivation for sparse projection directions, especially pairwise silhouettes: restricting a to two nonzero coordinates limits each component to a two-factor convolution, rather than entangling all d coordinate marginals in every projected measure.

## 5 Algorithm and Applications

The results of Sections 2 and 3 show that, under suitable structural conditions, mixture-of-products models can be identified from a finite collection of one-dimensional linear pushforwards. This suggests a practical estimation strategy: fit a mixture-of-products model by matching its one-dimensional projected distributions to the empirical distributions obtained by projecting the observed data along the same directions.

This section describes a concrete implementation of this approach, termed silhouette mixture estimation (SME). To construct a candidate model, the data are partitioned into blocks intended to correspond to the individual mixture components $\textstyle \prod _ { j = 1 } ^ { d } \mu _ { i , j }$ in (2). Within each block, an empirical marginal is formed for each coordinate, and the product of these marginals yields a naive Bayes-type empirical estimate of the corresponding component. These block-level product measures are weighted by their relative block sizes and summed to form the final estimator.

## 5.1 Form of the Estimator

The construction begins with the empirical measure associated with the observed data. Given samples $X _ { 1 } , \ldots , X _ { n } \stackrel { i i d } { \sim } \mu .$ , the empirical measure is

$$
\hat { \mu } _ { n } \triangleq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { X _ { i } } .
$$

To motivate the form of SME, first suppose that $\mu$ admits a naive Bayes factorization

$$
\mu = \prod _ { j = 1 } ^ { d } \mu _ { j } ,
$$

where each $\mu _ { j }$ is a probability measure on R. In this case, a natural factorizable analogue of $\hat { \mu } _ { n }$ is obtained by taking the product of its one-dimensional empirical marginals. Writing $X _ { i } = [ X _ { i , 1 } , \bar { \mathbf { \Phi } } . \bar { \mathbf { \Phi } } . \mathbf { \Phi } . \bar { X _ { i , d } } ] ^ { \top }$ , the resulting factorizable empirical measure is

$$
\prod _ { j = 1 } ^ { d } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { X _ { i , j } } \right) = \frac { 1 } { n ^ { d } } \sum _ { I \in [ n ] ^ { d } } \delta _ { \left( X _ { I _ { 1 } , 1 } , \ldots , X _ { I _ { d } } , d \right) } .
$$

The expansion on the right shows that this measure assigns equal weight to every coordinate-wise recombination of the observed samples.

For a nonempty multiset $\{ x _ { 1 } , \ldots , x _ { m } \}$ in $\mathbb { R } ^ { d }$ , where $\boldsymbol { x } _ { i } = [ x _ { i , 1 } , \ldots , x _ { i , d } ] ^ { \top }$ , define the naive Bayes map NB by

$$
\mathbf { N B } ( \{ x _ { 1 } , \ldots , x _ { m } \} ) \triangleq \prod _ { j = 1 } ^ { d } \left( { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \delta _ { x _ { i , j } } \right) .
$$

Given a partition A of $[ n ]$ into k blocks, SME constructs a factorizable empirical measure from each block and mixes these measures according to their relative block sizes:

$$
{ \hat { \mu } } _ { \mathrm { S M E } } \triangleq \sum _ { A \in { \mathcal { A } } } { \frac { | A | } { n } } \mathbf { N B } ( \{ X _ { i } : i \in A \} ) .\tag{11}
$$

Given the training data, the estimator is therefore completely specified by A. The subsequent subsections describe how this partition is selected.

An important property of $\hat { \mu } _ { \mathrm { S M E } }$ is that, by construction, it matches every coordinate marginal of the empirical distribution exactly, regardless of the partition. Writing $\nu ^ { ( j ) }$ for the $j \mathrm { - t h }$ coordinate marginal of a measure $\nu ,$ one has, for each $j \in [ d ]$ ,

$$
\begin{array} { r l } & { \hat { \mu } _ { \mathrm { S M E } } ^ { ( j ) } = \displaystyle \sum _ { A \in \mathcal { A } } \frac { | A | } { n } \mathbb { N } \mathbb { B } ( \{ X _ { i } : i \in A \} ) ^ { ( j ) } } \\ & { \quad \quad = \displaystyle \sum _ { A \in \mathcal { A } } \frac { | A | } { n } \left( \displaystyle \prod _ { r = 1 } ^ { d } \left( \frac { 1 } { | A | } \displaystyle \sum _ { \ell \in A } \delta _ { x _ { \ell } , r } \right) \right) ^ { ( j ) } } \\ & { \quad \quad = \displaystyle \sum _ { A \in A } \frac { | A | } { n } \left( \frac { 1 } { | A | } \displaystyle \sum _ { \ell \in A } \delta _ { x _ { \ell } , j } \right) } \\ & { \quad \quad = \displaystyle \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \delta _ { x _ { i , j } } = \hat { \mu } _ { n } ^ { ( j ) } . } \end{array}
$$

Thus, the coordinate marginals of the estimator are independent of the partition and coincide exactly with those of the empirical measure.

## 5.2 Objective Function

Let ν denote a candidate estimator of the form (11). Estimation proceeds by matching one-dimensional projections of pairwise marginals of ν to the corresponding empirical projections derived from $\hat { \mu } _ { n }$ . Motivated by the identifiability results of Section 3, the fitting objective aggregates projection discrepancies across selected pairs of coordinates. Let $G = ( [ d ] , E )$ be a graph specifying which coordinate pairs are used for fitting. Write each edge $e \in E$ as $\{ p , q \}$ , with $p \ < \ q$ . Extending the marginal notation above, let $\nu ^ { ( p , q ) }$ denote the two-dimensional marginal of ν on coordinates p and q. That is, if $Z = [ Z _ { 1 } , \ldots , Z _ { d } ] ^ { \top } \sim \nu ,$ then $[ Z _ { p } , Z _ { q } ] ^ { \top } \sim \nu ^ { ( p , q ) }$

As in Section 2, for $v \in \mathbb { R } ^ { 2 }$ define $\pi _ { v } ( x ) = \langle v , x \rangle$ . Let d denote a metric on probability measures on $\mathbb { R }$ The Wasserstein metric is a natural choice because it captures the underlying geometry of the distributions and is eficiently computable in one dimension, although the framework is not restricted to this choice. In all experiments, d is the 1-Wasserstein metric. For each edge $\{ p , q \} \in E$ , let $S _ { ( p , q ) } \subset \mathbb { R } ^ { 2 }$ denote the finite set of projection directions used for that coordinate pair. The directions in (4)–(5) provide a theoretically justified choice. For $\{ p , q \} \in E$ and $v \in S _ { ( p , q ) }$ , consider the discrepancy

$$
{ \sf d } \left( \pi _ { v } \# \nu ^ { \left( p , q \right) } , \pi _ { v } \# \hat { \mu } _ { n } ^ { \left( p , q \right) } \right) .\tag{12}
$$

Writing $\boldsymbol { v } = [ v _ { 1 } , v _ { 2 } ] ^ { \top }$ , the empirical term is computed from the projected samples

$$
v _ { 1 } X _ { i , p } + v _ { 2 } X _ { i , q } , \qquad i \in [ n ] ,
$$

thereby reducing the comparison to a one-dimensional problem.

SME minimizes an aggregate of terms of the form (12) over the selected graph edges and projection directions. In the experiments of Section 6, using substantially fewer than the theoretically suggested 2k projection directions for each coordinate pair worked well in practice. The experiments on benchmark datasets also used non-dense but well-connected graphs. The terms were aggregated using their mean, although alternative aggregation schemes are possible.

## 5.3 Computation of Projected Marginals

This subsection describes how to compute the pushforward appearing in (12) for a candidate estimator of the form (11). Fix a partition $\mathcal { A }$ of the data indices and a coordinate pair $( p , q )$ . Consider

$$
\pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \left( \sum _ { A \in \mathcal { A } } \frac { | A | } { n } \mathbf { N B } ( \{ X _ { i } : i \in A \} ) \right) ^ { ( p , q ) } .
$$

This is the pushforward obtained by first taking the $( p , q )$ -marginal of the SME estimator and then projecting linearly along $[ v _ { 1 } , v _ { 2 } ] ^ { \top }$ . Pushforwards are linear operators on the space of finite signed measures, and taking the $( p , q )$ -marginal simply removes the remaining coordinates. Hence

$$
\begin{array} { r l } { \pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \left( \displaystyle \sum _ { A \in A } \frac { | A | } { n } \mathbf { N B } ( \{ X _ { i } : i \in A \} ) \right) ^ { ( p , q ) } = \pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \left( \displaystyle \sum _ { A \in A } \frac { | A | } { n } \mathbf { N B } \big ( \{ [ X _ { i , p } , X _ { i , q } ] ^ { \top } : i \in A \} \big ) \right) } & { } \\ & { = \displaystyle \sum _ { A \in A } \frac { | A | } { n } \pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \mathbf { N B } \big ( \{ [ X _ { i , p } , X _ { i , q } ] ^ { \top } : i \in A \} \big ) . } \end{array}
$$

Thus, the projected SME marginal can be computed blockwise. That is, for each $A \in A .$ compute

$$
\pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \mathbf { N B } \big ( \{ [ X _ { i , p } , X _ { i , q } ] ^ { \top } : i \in A \} \big ) ,
$$

and then mix these measures using the weights $| A | / n$

Fix a block $A \in { \mathcal { A } }$ . Then

$$
\begin{array} { r l } & { \pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \mathbf { N B } \big ( \{ [ X _ { i , p } , X _ { i , q } ] ^ { \top } : i \in A \} \big ) } \\ & { = \pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \left( \left( \cfrac { 1 } { | A | } \displaystyle \sum _ { i \in A } \delta _ { X _ { i , p } } \right) \times \left( \cfrac { 1 } { | A | } \displaystyle \sum _ { j \in A } \delta _ { X _ { j , q } } \right) \right) } \\ & { = \left( \displaystyle \frac { 1 } { | A | } \displaystyle \sum _ { i \in A } \delta _ { v _ { 1 } X _ { i , p } } \right) * \left( \displaystyle \frac { 1 } { | A | } \displaystyle \sum _ { j \in A } \delta _ { v _ { 2 } X _ { j , q } } \right) } \\ & { = \displaystyle \frac { 1 } { | A | ^ { 2 } } \displaystyle \sum _ { i \in A } \sum _ { j \in A } \delta _ { v _ { 1 } X _ { i , p } + v _ { 2 } X _ { j , q } } , } \end{array}
$$

where ∗ denotes convolution of measures. Substituting this expression back into the full mixture and summing over the blocks yields

$$
\pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \left( \sum _ { A \in \mathcal { A } } \frac { | A | } { n } \mathbf { N B } ( \{ X _ { i } : i \in A \} ) \right) ^ { ( p , q ) } = \sum _ { A \in \mathcal { A } } \frac { 1 } { | A | n } \sum _ { i \in A } \sum _ { j \in A } \delta _ { v _ { 1 } X _ { i , p } + v _ { 2 } X _ { j , q } } .\tag{13}
$$

For comparison, the projected empirical marginal of $\hat { \mu } _ { n }$ is

$$
\pi _ { [ v _ { 1 } , v _ { 2 } ] } \# \hat { \mu } _ { n } ^ { ( p , q ) } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { v _ { 1 } X _ { i , p } + v _ { 2 } X _ { i , q } } .\tag{14}
$$

For each $\{ p , q \} \in E$ and $v = [ v _ { 1 } , v _ { 2 } ] ^ { \top } \in S _ { ( p , q ) }$ , the measures in (13) and (14) are compared using d. These discrepancies are then aggregated to form the SME fitting objective.

## 5.4 Optimization

With the objective defined, estimation reduces to selecting a partition $\mathcal { A }$ that minimizes the aggregate discrepancy over graph edges and projection directions. When partitioning the data into k blocks, it is convenient to represent a partition by an assignment vector $z \in [ k ] ^ { n }$ , where $z _ { i } = r$ indicates that sample $X _ { i }$ is assigned to block r. The induced partition is specified by

$$
A _ { r } \triangleq \{ i \in [ n ] : z _ { i } = r \} , \qquad r \in [ k ] .
$$

An empty assignment class has zero mixture weight and can simply be omitted from ${ \mathcal { A } } .$ The optimization problem is therefore to minimize $L ( z )$ over assignment vectors $z \in [ k ] ^ { n }$ , where $L ( z )$ is obtained by substituting the corresponding SME estimator into (12) and averaging over the selected graph edges and projection directions. The unconstrained assignment space has cardinality $k ^ { n }$ , rendering exhaustive optimization infeasible except in trivial regimes. Nevertheless, the objective possesses structure that permits eficient evaluation and practical optimization strategies.

First, the loss decomposes across graph edges and projection directions. All terms corresponding to distinct graph edges or projection directions may therefore be evaluated independently and aggregated afterward, making the computation highly parallelizable.

For a fixed edge $\{ p , q \}$ and projection vector v, evaluating the corresponding term in $L ( z )$ requires constructing the discrete measure (13) and comparing it to the empirical projection (14). Each block $A _ { r }$ contributes a weighted collection of point masses of the form

$$
\frac { 1 } { | A _ { r } | n } \sum _ { i \in A _ { r } } \sum _ { j \in A _ { r } } \delta _ { v _ { 1 } X _ { i , p } + v _ { 2 } X _ { j , q } } .
$$

Because these weights vary across blocks, it is natural to represent each projected block as a list of support points and associated weights.

When d is the Wasserstein metric, sorting the support points substantially reduces evaluation time. The empirical projection (14) can be computed once and cached for every combination of graph edge and projection direction. Block-level contributions can also be cached: when a local update changes a single assignment $z _ { i } ,$ only the two afected blocks and their corresponding contributions to (13) must be recomputed.

The convolution within each block incurs $O ( | A _ { r } | ^ { 2 } )$ cost in a naive implementation. This cost can be reduced by discretizing the values $X _ { i , j }$ during the optimization phase, for example by rounding to a fixed grid, thereby replacing the $\left| A _ { r } \right|$ sample values with a smaller set of weighted representatives. The projected support can be reduced further by choosing projection coeficients that encourage diferent coordinate combinations to yield the same projected value. For example, if the coordinate values are rounded to $c \mathbb { Z } .$ , with $c > 0$ and the projection families in (4)–(5) use $a \ = \ { \frac { 1 } { 2 } }$ and $b _ { 1 } = b _ { 2 } = 1$ , all scaled coordinate values lie on $( c / 2 ^ { k } ) \mathbb { Z }$ Coincident projected point masses can be combined into a single point mass at their shared location by summing their weights. This aggregation can substantially reduce the efective support size of each component, particularly when the coordinate values contain many repetitions or are discretized onto a coarse grid. Once the partition has been determined, the original full-precision values may be reinstated for downstream use. In the experiments of Section $6 ,$ this approximation yielded substantial computational speedups without materially degrading performance.

Additional computational savings may be obtained by using fewer projection directions and sparser graphs, thereby reducing the number of Wasserstein distances that must be evaluated. In the experiments of Section 6, substantially fewer than the theoretically suggested 2k directions per coordinate pair and relatively sparse but well-connected graphs both worked well in practice.

## 5.5 Utilizing the SME Estimator

After fitting, SME produces a probability measure of the form

$$
\hat { \mu } _ { \mathrm { S M E } } = \sum _ { A \in \mathcal { A } } \frac { \vert A \vert } { n } \prod _ { j = 1 } ^ { d } \left( \frac { 1 } { \vert A \vert } \sum _ { \ell \in A } \delta _ { X _ { \ell , j } } \right) .\tag{15}
$$

When expanded, this representation contains $\textstyle \sum _ { A \in { \mathcal { A } } } | A | ^ { d }$ point-mass terms and may therefore have combinatorially large support. Furthermore, unless all blocks contain equal numbers of samples, the terms contributed by diferent blocks have diferent weights. Consequently, ˆµ<sub>SME</sub> cannot simply be treated as a plug-in replacement for an ordinary empirical measure, and downstream procedures must account for its mixture-of-products structure. This subsection describes several practical ways to work with estimators of the form (15).

Sampling Despite the combinatorial expansion suggested by (15), sampling from $\hat { \mu } _ { \mathrm { S M E } }$ is eficient due to its mixture-of-products structure. To generate a sample, first select a block $A \in { \mathcal { A } }$ with probability $| A | / n$ Then, independently for each coordinate $j \in [ d ]$ , draw an index $\ell _ { j }$ uniformly from A and set $Z _ { j } = X _ { \ell _ { j } , j }$ . The resulting vector $Z \overset { \cdot } { = } [ Z _ { 1 } , \ldots , Z _ { d } ] ^ { \top }$ is distributed according to ˆµ<sub>SME</sub>. Sampling requires only a single draw from the mixture weights together with d independent draws from the selected block and does not require enumerating the $| A | ^ { d }$ point-mass terms contributed by that block.

Density Estimation The representation (15) naturally gives rise to a density estimator. For each block $A \in { \mathcal { A } }$ and coordinate $j \in [ d ]$ , use the one-dimensional sample $( X _ { \ell , j } ) _ { \ell \in A }$ to construct a density estimate $\hat { p } _ { A , j }$ using any suitable one-dimensional method, such as a fitted Gaussian density, a kernel density estimator, or even a one-dimensional normalizing flow. Combining these estimates yields

$$
\hat { p } ( x _ { 1 } , \ldots , x _ { d } ) = \sum _ { A \in \mathcal { A } } \frac { | A | } { n } \prod _ { j = 1 } ^ { d } \hat { p } _ { A , j } ( x _ { j } ) .\tag{16}
$$

The estimator (16) has the mixture-of-products form (2), with each block A corresponding to a component, $| A | / n$ serving as its mixture weight, and $\hat { p } _ { A , j }$ estimating the density of the corresponding component marginal probability measure $\mu _ { i , j } .$ . The one-dimensional estimators can be fitted independently across blocks and coordinates, allowing straightforward parallelization.

A further useful property of mixture-of-products densities is that they permit eficient conditional density evaluation and sampling. Consider a density

$$
p ( x ) = \sum _ { i = 1 } ^ { k } w _ { i } \prod _ { j = 1 } ^ { d } p _ { i , j } ( x _ { j } ) ,\tag{17}
$$

where $w _ { i } > 0 , \sum _ { i = 1 } ^ { k } w _ { i } = 1$ , and each $p _ { i , j }$ is a probability density function. Let $X \sim p .$ . For $S \subset [ d ]$ , let $X _ { S }$ denote the tuple of coordinates indexed by S, listed in increasing index order, and write $S ^ { c } = [ \dot { d } ] \dot { \ } S$ . For conditioning values $y _ { S }$ with positive marginal density,

$$
\begin{array} { l } { { \displaystyle p ( x _ { S ^ { c } } \mid X _ { S } = y _ { S } ) \propto \sum _ { i = 1 } ^ { k } w _ { i } \left( \prod _ { j \in S } p _ { i , j } ( y _ { j } ) \right) \left( \prod _ { j \in S ^ { c } } p _ { i , j } ( x _ { j } ) \right) } } \\ { { \displaystyle \qquad = \sum _ { i = 1 } ^ { k } \tilde { w } _ { i } ( y _ { S } ) \prod _ { j \in S ^ { c } } p _ { i , j } ( x _ { j } ) } , } \end{array}
$$

where $\begin{array} { r } { \tilde { w } _ { i } ( y _ { S } ) \triangleq w _ { i } \prod _ { j \in S } p _ { i , j } ( y _ { j } ) } \end{array}$ is the unnormalized conditional weight of component i. Therefore, the conditional density retains the mixture-of-products form

$$
p ( x _ { S ^ { c } } \mid X _ { S } = y _ { S } ) = \sum _ { i = 1 } ^ { k } { \frac { { \tilde { w } } _ { i } ( y _ { S } ) } { \sum _ { \ell = 1 } ^ { k } { \tilde { w } } _ { \ell } ( y _ { S } ) } } \prod _ { j \in S ^ { c } } p _ { i , j } ( x _ { j } ) .
$$

Thus, conditioning changes only the mixture weights, leaving each component’s marginals on the remaining coordinates unchanged. Notably, a single fitted density of the form (17) can be used for density evaluation and sampling, both unconditionally and conditionally, without fitting separate models or resorting to approximate inference. Conditioning on rectangular events is similarly tractable, since the required component probabilities can be computed from coordinate-wise CDFs.

Distance-Based Methods Many procedures in statistics, such as k-nearest-neighbor density estimation and anomaly detection, are based on order statistics of distances [77]. For a target distribution $\mu$ and a test point $x ,$ such methods implicitly estimate the smallest radius r such that

$$
\mathbb { P } _ { X \sim \mu } ( \| X - x \| \leq r ) \approx { \frac { k } { n } } ,\tag{18}
$$

that ${ \mathrm { i s } } ,$ the ball centered at $x$ contains approximately a fraction $k / n$ of the total mass.

In standard k-nearest-neighbor methods, such radii are estimated directly from the empirical measure $\hat { \mu } _ { n }$ through distances to the observed samples. For $\hat { \mu } _ { \mathrm { S M E } }$ , directly computing such radii, or even evaluating $\mathbb { P } _ { Z \sim \hat { \mu } _ { \mathrm { S M E } } } \big ( \| Z - x \| \le r \big )$ , may be costly because the estimator can have combinatorially many support points. However, under the $\ell _ { \infty }$ norm, the product structure yields a useful simplification. The event $\| Z - x \| _ { \infty } \leq r$ is equivalent to the coordinate-wise constraints $| Z _ { j } - x _ { j } | \le r$ for all $j \in [ d ]$ . Writing $\mathbb { 1 } ( \cdot )$ for the indicator function, (15) implies

$$
\mathbb { P } _ { Z \sim \hat { \mu } _ { \mathrm { S M E } } } ( \| Z - x \| _ { \infty } \leq r ) = \sum _ { A \in \mathcal { A } } \frac { | A | } { n } \prod _ { j = 1 } ^ { d } \left( \frac { 1 } { | A | } \sum _ { \ell \in A } { \mathbb 1 } \left( | X _ { \ell , j } - x _ { j } | \leq r \right) \right) .
$$

Thus, evaluating $\ell _ { \infty } .$ -ball probabilities reduces to computing, for each block, the one-dimensional empirical interval probabilities, multiplying them across coordinates, and taking the convex combination of the resulting products with weights $| A | / n$ . This avoids explicitly enumerating the $| A | ^ { d }$ coordinate recombinations within each block. The radius r in (18) can then be found eficiently by bisection.

For other norms, such as $\ell _ { 2 } ,$ the ball event does not decouple across coordinates, but the corresponding probabilities can instead be approximated by sampling $Z \sim \hat { \mu } _ { \mathrm { S M E } }$ and forming the empirical CDF $\operatorname { o f } \| Z - x \|$ The Dvoretzky–Kiefer–Wolfowitz inequality provides uniform control of this approximation as a function of r.

## 6 Experiments

This section presents evaluations of SME on synthetic and real-world datasets. The synthetic experiment tests whether it recovers the ground-truth rank-one components of an identifiable mixture-of-products model, while the real-world experiments assess its usefulness for downstream density estimation. The SME fitting objective uses the 1-Wasserstein distance in all experiments.

## 6.1 Mixture Recovery on a Synthetic Dataset

To assess whether SME can recover an identifiable mixture-of-products structure from projected marginals, consider the following synthetic example. For an interval $I \subset \mathbb { R }$ , let Unif(I) denote the uniform probability measure on $I ,$ and set

$$
\begin{array} { r l } & { \nu _ { 1 } = \mathrm { U n i f } \left( \left[ - \frac { 5 } { 6 } , \frac { 5 } { 6 } \right] \right) , } \\ & { \nu _ { 2 } = \frac { 1 } { 2 } \mathrm { U n i f } \left( \left[ - \frac { 3 } { 2 } , - \frac { 1 } { 2 } \right] \right) + \frac { 1 } { 2 } \mathrm { U n i f } \left( \left[ \frac { 1 } { 2 } , \frac { 3 } { 2 } \right] \right) , } \\ & { \nu _ { 3 } = \frac { 1 } { 2 } \mathrm { U n i f } \left( \left[ - \frac { 1 3 } { 6 } , - \frac { 7 } { 6 } \right] \right) + \frac { 1 } { 2 } \mathrm { U n i f } \left( \left[ \frac { 7 } { 6 } , \frac { 1 3 } { 6 } \right] \right) . } \end{array}
$$

Define the three mixture components by

$$
\mu _ { 1 } = \nu _ { 1 } \times \nu _ { 1 } , \qquad \mu _ { 2 } = \nu _ { 2 } \times \nu _ { 2 } , \qquad \mu _ { 3 } = \nu _ { 3 } \times \nu _ { 3 } ,
$$

and let

$$
\mu = \frac { 1 } { 3 } \sum _ { i = 1 } ^ { 3 } \mu _ { i } .\tag{19}
$$

The mixture components $\mu _ { 1 } , \mu _ { 2 } , \mu _ { 3 }$ have overlapping supports, but the component marginals $\nu _ { 1 } , \nu _ { 2 } , \nu _ { 3 }$ satisfy the joint irreducibility assumption, so by Proposition $6 ,$ the representation of $\mu$ in (19) is identifiable. Each component is symmetric about both coordinate axes. The “Ground truth” panel of Figure 2 shows 900 samples, with 300 drawn from each component. Point color and marker indicate the true component membership. The components exhibit nontrivial overlap while retaining regions of support unique to each component.

The remaining panels in Figure 2 show partitions obtained by initializing component labels at random and running the SME optimization until a local minimum was reached. The parameters $\ a \ = \ - 2$ and $b _ { 1 } = b _ { 2 } = 1$ were used in (4) and (5), with $k = 1 , 2 , 3$ in the projection formulas for the two-, four-, and six-projection cases, respectively. The objective was the mean Wasserstein distance over the projections. The negative sign gives projection directions with both positive and negative slopes. In these synthetic experiments, projection directions were not normalized to unit length, so the larger vector norms associated with higher powers of a efectively gave those directions greater weight in the objective. This was intended to place greater emphasis on newly introduced higher-order projections as additional projection directions were added. The one-projection case used only the projection onto $[ 1 , 1 ] ^ { \top }$ . As noted in Section 5.1, the coordinate-axis marginals are matched exactly by construction, so these experiments implicitly incorporate the axis projections in addition to the explicitly listed projection directions. For visualization purposes, component labels are permuted to align with the ground-truth labeling.

![](images/3ee437b45df0ede795db23fe045e671d19d64d642f16bc0d5fcb037d1a25994c.jpg)  
Fig. 2: Ground-truth component assignments and partitions recovered by SME using 1, 2, 4, and 6 non-axis projections. Colors and markers indicate component labels. Every fit also matches the empirical coordinate marginals exactly by construction.

As seen in Figure 2, a single projection recovers the main shapes and locations of the underlying components, although some misassignments remain, particularly between the central component (blue circles) and the adjacent component (orange triangles). The recovered central and adjacent clusters extend into one another’s regions, most visibly near the upper-left and lower-right corners of the central square. This produces directional leakage rather than errors confined to the local overlap region. Notably, the leakage occurs approximately orthogonally to the sole projection direction $[ 1 , 1 ] ^ { \top }$

Adding a second non-axis projection substantially reduces this directional leakage and improves agreement with the ground truth. Beyond two projections, the changes are more modest and do not show a uniform improvement: the four- and six-projection fits still retain some errors near overlap regions and a small number of isolated misassignments. This raises the question of whether the theoretical count of $2 k = 6$ projections is necessary for this particular model or primarily reflects a worst-case requirement.

## 6.2 Density Estimation on Real-World Datasets

This subsection evaluates SME and several competing methods on real-world density estimation tasks. For each split, performance is measured by the mean log-likelihood on the test set (equivalently, negative test cross-entropy). Experiments were conducted on the HEPMASS [78] and kin8nm [79] datasets. For HEP-MASS, the preprocessed dataset from [80], which excludes dimensions containing discrete components and standardizes each coordinate to have mean zero and unit variance, was used. For kin8nm, each coordinate was likewise standardized using the mean and variance of the full dataset before splitting. Each dataset was split into training, validation, and test sets, and all experiments were repeated across 20 random splits. Dataset dimensions and training set sizes are included in Table 1 for reference. The validation split is used somewhat diferently across methods depending on the corresponding training procedure, as described below in Section 6.2.1. Additional experimental details are provided in Appendix B.

## 6.2.1 Density Estimation Methods

The experiments compare the SME-based density estimator described in Section 5.5 with several standard density estimation baselines.

Silhouette Mixture Estimation (SME) SME used the nonparametric density estimation procedure described in Section 5.5. For each dataset, a random 4-regular graph was constructed on the d coordinates, giving 2d edges. For every edge, the same four unit-norm projection directions were used. These were obtained from (4) and (5) with $a = - 2$ and $b _ { 1 } = b _ { 2 } = 1$ , followed by normalization. Four projections are suficient according to the theory when $k = 2$ , although the experiments also considered larger numbers of mixture components.

The number of mixture components was selected from $k \in \{ 2 , 3 , 4 \}$ by maximizing validation likelihood. This range was chosen for computational convenience but did not appear to severely constrain model selection: the values selected by validation averaged 3.1 for HEPMASS and 2.2 for kin8nm. To initialize the component assignments, the training samples were assigned independently and uniformly at random to the k components. The algorithm then processed the samples repeatedly in a fixed order. For each sample, the objective was evaluated under every possible component assignment. After all assignments had been evaluated, the sample was assigned to the component yielding the lowest objective value. The procedure terminated when a complete pass through the training set produced no improvement, that is, when it reached a local minimum with respect to single-sample reassignments.

After the partition was obtained, a one-dimensional Gaussian kernel density estimator was fitted to each component-coordinate pair, and the resulting estimators were combined using the procedure described in Section 5.5. For each component and coordinate, the bandwidth was selected by 15-fold cross-validation over a logarithmic grid of 80 values ranging from $1 0 ^ { - 3 }$ to 1. The separate validation split was not used for bandwidth selection.

Kernel Density Estimator (KDE) A standard Gaussian KDE was used, with the bandwidth selected using the validation dataset over the same logarithmic grid used for SME.

Naive Bayes KDE (NB-KDE) This method fits a one-dimensional Gaussian kernel density estimator to each coordinate and uses the product of these estimators as the final density estimate. It is exactly equivalent to the SME density estimation procedure above with all samples assigned to a single component.

Gaussian Mixture Model (GMM) A Gaussian mixture model was fitted using the EM implementation in scikit-learn [81], with default settings except for the number of components. The candidate models ranged from a single multivariate Gaussian density to a 32-component Gaussian mixture, with the number of components selected using the validation log-likelihood.

Axis-Aligned Gaussian Mixture Model (GMM-AA) The axis-aligned GMM used exactly the same fitting and model-selection procedure as the GMM above, except that each component covariance matrix was constrained to be diagonal.

Masked Autoregressive Flow (MAF) [80] Masked autoregressive flows were implemented using the Zuko Python library [82]. Architecture and training hyperparameters were chosen through preliminary experiments, and validation likelihood was used for early stopping. Complete implementation details are provided in Appendix B.

## 6.2.2 Choice of Baselines

The competing methods were chosen to reflect diferent structural assumptions relative to SME. KDE and NB-KDE represent two extremes. If $k = n$ and each sample is assigned to its own component, then the SME estimator reduces to the empirical distribution and the silhouette objective is zero. Smoothing this empirical distribution with Gaussian kernels yields KDE. At the opposite extreme, when k = 1, the model reduces to a single-component factorized estimator, corresponding to NB-KDE. GMM-AA provides a parametric analogue with axis-aligned covariance matrices, preserving the assumption that coordinates are independent within each mixture component. Together, these three methods compare SME with its two extreme cases and with a parametric model having the same within-component factorization. The full-covariance GMM, which includes a single Gaussian model as a special case, serves as a representative parametric clustering and density estimation baseline. Finally, MAF is included as a representative deep density estimator, although the relatively small training sets considered here are not particularly favorable to such models.

## 6.2.3 Results and Discussion

Table 1 summarizes performance across the 20 matched random splits and reports p-values from two-sided Wilcoxon signed-rank tests comparing SME with each competing method. A small p-value provides evidence of a systematic diference in performance between the methods. For SME, the table also reports the mean and standard deviation across splits of the number of components k selected using the validation data.

<table><tr><td>Dataset  $\overline { { ( d , n ) } }$ </td><td>HEPMASS (21, 250)</td><td>kin8nm (9, 1000)</td></tr><tr><td>SME (proposed)</td><td> $\mathbf { - 2 7 . 3 6 \pm 0 . 1 3 ~ ( 3 . 1 \pm 0 . 8 ) }$ </td><td> $\mathbf { - 1 1 . 4 9 5 \pm 0 . 0 5 1 ~ ( 2 . 2 \pm 0 . 4 ) }$ </td></tr><tr><td>KDE</td><td> $- 3 0 . 7 1 \pm 0 . 0 9 \ ( 2 \times 1 0 ^ { - 6 } )$ </td><td> $- 1 2 . 3 8 6 \pm 0 . 0 1 6 ~ ( 2 \times 1 0 ^ { - 6 } )$ </td></tr><tr><td>NB-KDE</td><td> $- 2 9 . 2 4 \pm 0 . 7 0 \ ( 2 \times 1 0 ^ { - 6 } )$ </td><td> $- 1 1 . 5 6 3 \pm 0 . 0 1 2 ~ ( 4 \times 1 0 ^ { - 5 } )$ </td></tr><tr><td>GMM</td><td> $- 2 8 . 1 9 \pm 0 . 2 5 \ ( 2 \times 1 0 ^ { - 6 } )$ </td><td> $- 1 2 . 1 7 6 \pm 0 . 0 5 9 \ ( 2 \times 1 0 ^ { - 6 } )$ </td></tr><tr><td>GMM-AA</td><td> $- 2 7 . 7 3 \pm 0 . 1 2 \ ( 2 \times 1 0 ^ { - 6 } )$ </td><td> $- 1 2 . 2 9 3 \pm 0 . 0 3 7 \ ( 2 \times 1 0 ^ { - 6 } )$ </td></tr><tr><td>MAF</td><td> $- 2 8 . 5 7 \pm 0 . 5 5 ~ ( 2 \times 1 0 ^ { - 6 } )$ </td><td> $- 1 2 . 2 1 6 \pm 0 . 1 0 7 ~ ( 2 \times 1 0 ^ { - 6 } )$ </td></tr></table>

Tab. 1: Test log-likelihood (mean ± standard deviation across 20 random splits). For SME, parentheses report the validation-selected number of components k (mean ± standard deviation); for competing methods, parentheses report two-sided Wilcoxon signed-rank p-values from paired comparisons with SME. Bold indicates the best mean in each column.

SME achieves the highest mean test log-likelihood for both datasets in the experimental settings considered here. These experiments focus on a sample-limited, moderate-dimensional regime in which flexible density estimation remains desirable but fully high-dimensional models can be dificult to estimate reliably. This difers substantially from the HEPMASS setting commonly used to evaluate normalizing flows: the evaluation in [80] uses 315,123 training observations, compared with 250 here, and demonstrates strong flow performance in that large-sample setting. The MAF results in Table 1 should therefore be understood as describing a diferent regime, rather than as a general comparison with flows under conventional large-sample benchmark conditions.

For HEPMASS, the strongest competing method is GMM-AA, while for kin8nm it is NB-KDE. This is consistent with the structural assumptions of these methods: GMM-AA combines mixture structure with within-component coordinate independence, while NB-KDE uses a single fully factorized nonparametric density. The full-covariance GMM must estimate substantially more parameters to capture dependence within each component, while multivariate KDE is well known to be particularly susceptible to the curse of dimensionality. SME occupies an intermediate position, retaining flexibility through its mixture structure while reducing estimation within each component to one-dimensional problems.

Runtime details and further diagnostics, including the relationship between the silhouette objective and downstream density estimation performance, are provided in Appendix B.

## 7 Sharpness of the Main Theorems

This section examines the sharpness of the identifiability and homeomorphism results in Section 2. The first two results concern the number and geometry of the projection directions required for identifiability. Theorem 8 shows that the projection count 2k in Theorem 1 and Corollary 2 is optimal, while Proposition 9 shows that, even with 2k projections, the directions cannot be chosen arbitrarily. The remaining results show that the tail assumption in Theorem 1 and the boundedness restriction in Theorem 3 cannot in general be omitted. Proofs of all results in this section are given in Section 12.

Theorem 8. Let $k \in \mathbb N$ and let $v _ { 1 } , \dotsc , v _ { 2 k - 1 } \in \mathbb { R } ^ { 2 }$ . There exist probability measures $\mu _ { i , j } , \mu _ { i , j } ^ { \prime } \in \mathcal { P } ( \mathbb { R } )$ for $i \in [ k ]$ and $j \in [ 2 ]$ , each admitting a smooth, compactly supported density, and weights $w _ { i } , w _ { i } ^ { \prime } \geq 0 ~ f o r ~ i \in [ k ]$ satisfying

$$
\sum _ { i = 1 } ^ { k } w _ { i } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } = 1 ,
$$

such that the probability measures

$$
\mu = \sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , 1 } \times \mu _ { i , 2 } , \qquad \mu ^ { \prime } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime }
$$

satisfy

$$
\pi _ { v _ { \ell } } \# \mu = \pi _ { v _ { \ell } } \# \mu ^ { \prime } \qquad f o r \ a l l \ \ell \in [ 2 k - 1 ] ,
$$

while $\mu \neq \mu ^ { \prime }$

Thus, no collection of 2k−1 projection directions can identify all $\mathrm { r a n k } - \le k$ probability measures. Notably, the lower bound holds even for probability measures whose component marginals have smooth, compactly supported densities, so it is not an artifact of signed measures, irregularity, or heavy tails.

Although the 2k projections match the number of component marginals in the mixture representation, this correspondence does not itself explain the lower bound. A vector-RIP map cannot represent all k-sparse vectors using fewer than k scalar measurements. In contrast, an element of $\mathcal { M } ( [ 0 , 1 ] ) ^ { \bar { 2 } k }$ can be encoded as a single element of $\mathcal { M } ( [ 0 , 1 ] )$ by contracting and shifting its entries onto disjoint subintervals through afine pushforwards, then summing them. These operations are linear in the measures, and the disjoint supports make their norms add, giving an isometric encoding. In fact, composing the identifying silhouette of Theorem 1 with suitable afine rescalings and this concatenation gives an injective linear representation

$$
\mathcal { R } _ { \leq k } ( [ 0 , 1 ] , [ 0 , 1 ] ) \longrightarrow \mathcal { M } ( [ 0 , 1 ] ) .
$$

Thus, the lower bound reflects the restriction to projection measurements, rather than a dimensional obstruction arising from the number of component marginals.

The number of directions is not the only consideration. The following proposition shows that 2k directions can still fail to provide identifiability when arranged in certain symmetric configurations.

Proposition 9. Let $k \in \mathbb N$ and let $a _ { 1 } , \ldots , a _ { k } > 0$ be distinct. For $r \in [ k ]$ , define

$$
\boldsymbol { v } _ { r } ^ { + } = [ a _ { r } , 1 ] ^ { \top } , \qquad \boldsymbol { v } _ { r } ^ { - } = [ - a _ { r } , 1 ] ^ { \top } .
$$

Then there exist probability measures $\mu _ { i , j } , \mu _ { i , j } ^ { \prime } \in \mathcal { P } ( \mathbb { R } )$ for $i \in [ \lfloor k / 2 \rfloor + 1 ]$ and $j \in [ 2 ]$ , each admitting a smooth, compactly supported density, and weights $w _ { i } , w _ { i } ^ { \prime } \geq 0 ~ f o r ~ i \in [ \vert k / 2 \vert + 1 ]$ , satisfying

$$
\sum _ { i = 1 } ^ { \lfloor k / 2 \rfloor + 1 } w _ { i } = \sum _ { i = 1 } ^ { \lfloor k / 2 \rfloor + 1 } w _ { i } ^ { \prime } = 1 ,
$$

such that the probability measures

$$
\mu = \sum _ { i = 1 } ^ { \lfloor k / 2 \rfloor + 1 } w _ { i } \mu _ { i , 1 } \times \mu _ { i , 2 } , \qquad \mu ^ { \prime } = \sum _ { i = 1 } ^ { \lfloor k / 2 \rfloor + 1 } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime }
$$

satisfy

$$
\pi _ { v _ { r } ^ { + } } \# \mu = \pi _ { v _ { r } ^ { + } } \# \mu ^ { \prime } , \qquad \pi _ { v _ { r } ^ { - } } \# \mu = \pi _ { v _ { r } ^ { - } } \# \mu ^ { \prime } \qquad f o r \ a l l \ r \in [ k ] ,
$$

while $\mu \neq \mu ^ { \prime }$

Together with Theorem 8, this shows that 2k is the optimal projection count and that the geometry of the directions is a substantive part of the construction in Theorem 1. In particular, Proposition 9 identifies reflection symmetry among the projection directions as a potential obstruction to identifiability.

The preceding results concern the number and geometry of the projection directions. The following lemma shows that the tail assumption in Theorem 1 cannot be substantially weakened.

Lemma 10. Let $\ell \in \mathbb { N }$ and let $v _ { 1 } , \ldots , v _ { \ell } \in \mathbb { R } ^ { 2 }$ . There exist nonzero measures $\mu , \nu \in \mathcal { M } ( \mathbb { R } )$ , each admitting a density belonging to the Schwartz class, such that

$$
\pi _ { \boldsymbol { v } _ { i } } \# ( \mu \times \nu ) = 0 \qquad f o r \ a l l \ i \in [ \ell ] .
$$

Functions in the Schwartz class are smooth and decay, together with all their derivatives, faster than any polynomial [83, p. 237]. Decomposing the factors into positive and negative parts and normalizing preserves this tail decay and yields the following corollary for mixture-of-products models.

Corollary 11. Let $\ell \in \mathbb { N }$ and let $v _ { 1 } , \ldots , v _ { \ell } \in \mathbb { R } ^ { 2 }$ . There exist probability measures $\mu _ { i , j } , \mu _ { i , j } ^ { \prime } \in \mathcal { P } ( \mathbb { R } )$ for $i , j \in [ 2 ]$ , each admitting a density that decays faster than any polynomial, and weights $w _ { 1 } , \tilde { w } _ { 2 } , w _ { 1 } ^ { \prime } , w _ { 2 } ^ { \prime } > 0$ satisfying

$$
w _ { 1 } + w _ { 2 } = w _ { 1 } ^ { \prime } + w _ { 2 } ^ { \prime } = 1 ,
$$

such that the mixture-of-products models

$$
\mu = \sum _ { i = 1 } ^ { 2 } w _ { i } \mu _ { i , 1 } \times \mu _ { i , 2 } , \quad \quad \mu ^ { \prime } = \sum _ { i = 1 } ^ { 2 } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime }
$$

satisfy

$$
\pi _ { v _ { i } } \# \mu = \pi _ { v _ { i } } \# \mu ^ { \prime } \qquad f o r \ a l l \ i \in [ \ell ] ,
$$

while $\mu \neq \mu ^ { \prime }$

Thus, for $k \geq 2$ , finite-projection identifiability can fail even when the component densities decay faster than any polynomial and have moments of all orders, whereas Theorem 1 establishes identifiability under the stronger assumption that their Laplace transforms are finite on a neighborhood of the origin.

The preceding results concern identifiability. The next result shows that the boundedness restriction in Theorem 3 cannot be omitted, regardless of the finite collection of projection directions used.

Theorem 12. For any finite collection $v _ { 1 } , \ldots , v _ { \ell } \in \mathbb { R } ^ { 2 }$ , the map

$$
\mu \longmapsto ( \pi _ { v _ { 1 } } \# \mu , \ldots , \pi _ { v _ { \ell } } \# \mu )
$$

is not a wea $k ^ { * } - w e a k ^ { * }$ homeomorphic embedding of $\mathcal { R } _ { \leq 1 } ( [ 0 , 1 ] , [ 0 , 1 ] )$

The obstruction persists even under strong regularity conditions. The proof constructs a sequence of rank-one signed measures with $C ^ { \infty }$ densities supported on $[ 0 , 1 ] ^ { 2 }$ , whose total variation norms diverge while every prescribed projection converges to zero in total variation. Thus, the failure is not caused by singular or irregular measures: compact support and smoothness do not compensate for the absence of a uniform total variation bound.

## 8 Directions for Future Research

Theorems 1 and 3 establish that compactly supported low-rank probability measures on $\mathbb { R } ^ { 2 }$ are identified by their silhouettes and that this representation preserves the weak<sup>∗</sup> topology. Section 3 extends this framework to higher dimensions through collections of pairwise silhouettes, identifying settings in which the underlying measure can be recovered. Sections 5 and 6 develop these ideas algorithmically and demonstrate practical performance gains over standard nonparametric baselines. These results suggest several directions for further investigation.

One broad direction is to better understand the geometry induced by the silhouette representation beyond the injectivity and topological results established here. In finite-dimensional low-rank problems, injectivity is closely related to restricted-isometry-type bounds. This motivates the search for analogous quantitative guarantees for the silhouette operator. For measures, such guarantees are likely to depend substantially on the class of measures, the projection directions, and the metrics used on the domain and range. In particular, it would be useful to identify settings in which stability estimates hold and to determine whether additional projections improve them even after identifiability has been achieved.

A further direction is to use the mixture-of-products assumption to recast statistical problems in silhouette space, rather than using silhouettes only to fit a density estimator. Under the assumptions of the identifiability results, a prescribed finite collection of projected measures determines the full multivariate law. Thus, for example, a two-sample test could compare just those one-dimensional projections, with equality of the projected laws equivalent to equality of the original distributions at the population level. The broader question is which other statistical tasks can be usefully recast in terms of this finite collection of one-dimensional measures, and what guarantees or eficiencies result.

## 9 Proof of Injectivity of the Silhouette Operator on Low-Rank Measures

This section proves the central result of the paper, Theorem 1, along with its immediate corollary, Corollary 2. The proof requires a substantial amount of technical setup and preparatory results before the main proof can be presented.

## 9.1 Signed Measure Preliminaries and Rank

For a measurable space $( \Omega , { \mathcal { F } } )$ let $\mathcal { M } ( \Omega )$ denote the Banach space of finite signed measures on Ω equipped with the total variation norm

$$
\| \mu \| \triangleq \mu ^ { + } ( \Omega ) + \mu ^ { - } ( \Omega ) ,
$$

where $\mu ^ { + }$ and $\mu ^ { - }$ are the components of the Jordan decomposition $\mu = \mu ^ { + } - \mu ^ { - }$ . The subset $\mathcal { P } ( \Omega ) \subset \mathcal { M } ( \Omega )$ denotes probability measures. Recall that, for a measurable function $f : \Omega \to \mathbb { R }$ , the integral of a function with respect to a signed measure is defined by

$$
\int f d \mu \triangleq \int f d \mu ^ { + } - \int f d \mu ^ { - } ,
$$

as in [83]. Thus the integral on the left-hand side is finite if and only if the two integrals on the right-hand side are finite. Consequently, whenever the integral on the left-hand side is finite, the above decomposition is valid.

The following proposition shows that rank admits a characterization analogous to the familiar matrix case: a decomposition is minimal precisely when the component measures are linearly independent in each coordinate.

Proposition 13. Let $( \Omega _ { 1 } , { \mathcal { F } } _ { 1 } )$ and $( \Omega _ { 2 } , \mathscr { F } _ { 2 } )$ be measurable spaces and let $k \in \mathbb N$ . Suppose

$$
\mu = \sum _ { i = 1 } ^ { k } \mu _ { i , 1 } \times \mu _ { i , 2 } \in \mathcal { M } ( \Omega _ { 1 } \times \Omega _ { 2 } ) ,
$$

where $\mu _ { i , j } \in \mathcal { M } ( \Omega _ { j } )$ for $i \in [ k ]$ and $j \in [ 2 ]$ . Then $\mu$ has rank k if and only if $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ <sub>1</sub> are linearly independent and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are linearly independent.

Before proving Proposition 13, some additional measure-theoretic machinery is required. For a measurable space $( \Omega , { \mathcal { F } } )$ denote by $L ^ { \infty } ( \Omega )$ the Banach space of bounded measurable functions $f : \Omega \to { \mathbb { R } }$ , equipped with the norm $\| f \| = \operatorname* { s u p } _ { x \in \Omega } | f ( x ) |$ |. Given measurable spaces $( \Omega _ { 1 } , { \mathcal { F } } _ { 1 } )$ and $( \Omega _ { 2 } , \mathscr { F } _ { 2 } )$ , and a bounded measurable function $f \in L ^ { \infty } ( \Omega _ { 1 } )$ , define the bounded linear operator

$$
T _ { f } : { \mathcal { M } } ( \Omega _ { 1 } \times \Omega _ { 2 } ) \to { \mathcal { M } } ( \Omega _ { 2 } )
$$

by

$$
( T _ { f } \mu ) ( A ) \ \triangleq \ \int _ { \Omega _ { 1 } \times \Omega _ { 2 } } f ( x ) \mathbb { 1 } _ { A } ( y ) d \mu ( x , y ) , \qquad A \in { \mathcal { F } } _ { 2 } .\tag{20}
$$

In particular, if $\mu \in \mathcal { M } ( \Omega _ { 1 } )$ and $\mu ^ { \prime } \in \mathcal { M } ( \Omega _ { 2 } )$ , then

$$
T _ { f } ( \mu \times \mu ^ { \prime } ) = \mu ( f ) \mu ^ { \prime } .
$$

The following lemma provides a convenient way to apply linear algebraic arguments to finite collections of measures.

Lemma 14. Let $( \Omega , { \mathcal { F } } )$ be a measurable space and let $\mu _ { 1 } , \dots , \mu _ { k } \in \mathcal { M } ( \Omega )$ be linearly independent. Then there exist bounded measurable functions $f _ { 1 } , \dots , f _ { k } \in L ^ { \infty } ( \Omega )$ such that

$$
\mu _ { i } ( f _ { j } ) = \mathbb { 1 } ( i = j ) \qquad f o r \ a l l \ i , j \in [ k ] .
$$

Before proving Lemma 14, recall that for a measurable space (Ω, F), ba(Ω) denotes the Banach space of finitely additive signed measures on $( \Omega , { \mathcal { F } } )$ , equipped with the total variation norm. Such measures are not required to be countably additive. The usual space $\mathcal { M } ( \Omega )$ of finite signed (countably additive) measures is a norm-closed subspace of $b a ( \Omega )$ . The space $b a ( \Omega )$ is used here only to justify the duality arguments appearing in the proof. The argument also relies on Goldstine’s theorem [20, Theorem 3.27], which states that the closed unit ball of a Banach space is weak<sup>∗</sup> dense in the closed unit ball of its bidual.

Proof of Lemma 14. It is well known that

$$
L ^ { \infty } ( \Omega ) ^ { * } = b a ( \Omega ) ,
$$

see [84, Theorem 14.4]. Since $\mu _ { 1 } , \ldots , \mu _ { k }$ are linearly independent elements of $b a ( \Omega )$ , the Hahn–Banach theorem yields linear functionals $\mu _ { 1 } ^ { * } , \ldots , \mu _ { k } ^ { * } \in b a ( \Omega ) ^ { * }$ such that

$$
\mu _ { i } ^ { * } ( \mu _ { j } ) = \mathbb { 1 } ( i = j ) .
$$

By Goldstine’s theorem, for every $\varepsilon > 0$ there exist functions $\tilde { f } _ { 1 } , \ldots , \tilde { f } _ { k } \in L ^ { \infty } ( \Omega )$ such that

$$
\big | \mu _ { i } ( \tilde { f } _ { j } ) - \mu _ { i } ^ { * } ( \mu _ { j } ) \big | \leq \varepsilon \qquad \mathrm { f o r ~ a l l ~ } i , j \in [ k ] .
$$

Choosing ε suficiently small ensures that the matrix $\left[ \mu _ { i } ( \tilde { f } _ { j } ) \right] _ { i , j \in [ k ] }$ is invertible. Standard linear algebra then allows one to form suitable linear combinations of the functions $\tilde { f } _ { j }$ to obtain functions $f _ { 1 } , \ldots , f _ { k }$ satisfying $\mu _ { i } ( f _ { j } ) = \mathbb { 1 } ( i = j )$ □

Proof of Proposition 13. Suppose that one of the families is linearly dependent. Without loss of generality assume that $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ are linearly dependent. Then one of the measures, say $\mu _ { p , 1 }$ , can be written as a linear combination of the others. Substituting this relation into

$$
\mu = \sum _ { i = 1 } ^ { k } \mu _ { i , 1 } \times \mu _ { i , 2 }
$$

and using bilinearity of the product measure eliminates one term, yielding a representation with at most $k - 1$ terms. The same argument applies if $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are linearly dependent.

Conversely, suppose that both families $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are linearly independent. Assume for contradiction that

$$
\mu = \sum _ { i = 1 } ^ { k } \mu _ { i , 1 } \times \mu _ { i , 2 } = \sum _ { j = 1 } ^ { m } \nu _ { j , 1 } \times \nu _ { j , 2 }
$$

for some $m < k$ with $\nu _ { j , \ell } \in \mathcal { M } ( \Omega _ { \ell } )$ . By Lemma 14 applied to the family $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ , there exist bounded measurable functions $f _ { 1 , 1 } , \dots , f _ { k , 1 } \in L ^ { \infty } ( \Omega _ { 1 } )$ such that

$$
\mu _ { i , 1 } ( f _ { j , 1 } ) = \mathbb { 1 } ( i = j ) \qquad { \mathrm { f o r ~ a l l ~ } } i , j \in [ k ] .
$$

Applying the operator $T _ { f _ { p , \ell } }$ (defined in (20)) to both representations of $\mu$ yields, for each $p \in [ k ]$

$$
\mu _ { p , 2 } = T _ { f _ { p , 1 } } \mu = \sum _ { j = 1 } ^ { m } \nu _ { j , 1 } ( f _ { p , 1 } ) \nu _ { j , 2 } .
$$

Thus each $\mu _ { p , 2 }$ lies in the linear span of $\nu _ { 1 , 2 } , \ldots , \nu _ { m , 2 }$ , which has dimension at most $m$ . Since $m < k$ , this contradicts the linear independence of $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ □

## 9.2 Laplace Transform Preliminaries

The proof of Theorem 1 relies heavily on structural properties of Laplace transforms of measures. The required facts are collected and reviewed in this section. For a finite signed measure $\mu \in \mathcal { M } ( \mathbb { R } ^ { d } )$ , the (two-sided) Laplace transform is defined as

$$
\mathcal { L } ( \mu ) ( t ) \triangleq \int _ { \mathbb { R } ^ { d } } \exp ( \langle x , t \rangle ) d \mu ( x ) , \qquad t \in \mathbb { R } ^ { d } ,
$$

whenever the integral is finite.

When $\mu$ is a probability measure, ${ \mathcal { L } } ( \mu )$ coincides with the classical moment generating function from probability theory. Throughout this work, Laplace transform terminology and notation are used exclusively; results stated in the literature in terms of moment generating functions are interpreted accordingly.

Following standard usage in probability theory, the Laplace transform ${ \mathcal { L } } ( \mu )$ is said to exist if it is finite on some open neighborhood of the origin in $\mathbb { R } ^ { d }$ . By definition of signed measure integration, this requires that the Laplace transforms of both $\mu ^ { + }$ and $\mu ^ { - }$ also exist and are finite on that set.

The Laplace transform acts linearly on measures. In particular, if for some $t \in \mathbb { R }$ the quantities $\mathcal { L } ( \mu ) ( t )$ and $\mathcal { L } ( \mu ^ { \prime } ) ( t )$ both exist and are finite, then for all $c , c ^ { \prime } \in \mathbb { R }$ ,

$$
\mathscr { L } ( c \mu + c ^ { \prime } \mu ^ { \prime } ) ( t ) = c \mathscr { L } ( \mu ) ( t ) + c ^ { \prime } \mathscr { L } ( \mu ^ { \prime } ) ( t ) .
$$

The following proposition shows that, on R, equality of Laplace transforms on any set having an accumulation point at 0 implies equality of the underlying measures.

Proposition 15. Let $\mu , \mu ^ { \prime } \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } )$ . Then $\mu = \mu ^ { \prime }$ if and only if there exists a set $A \subset \mathbb { R }$ with an accumulation point at 0 such that

$$
\mathcal { L } ( \mu ) ( t ) = \mathcal { L } ( \mu ^ { \prime } ) ( t ) \quad f o r \ a l l \ t \in A .
$$

The proof of Proposition 15 relies on the analyticity of Laplace transforms. Two supporting results are stated first.

Theorem 16 ([85, Theorem 11.8]). Let $\mu , \mu ^ { \prime }$ be probability measures on $\mathbb { R } ^ { d }$ such that at least one of their Laplace transforms exists in a neighborhood of the origin. Then $\mu = \mu ^ { \prime }$ if and only $i f \mathcal { L } ( \mu )$ and $\mathcal { L } ( \mu ^ { \prime } )$ agree on some neighborhood $o f 0$

This can be extended to the following corollary.

Corollary 17. Let $\mu , \mu ^ { \prime } \in \mathcal { M } ( \mathbb { R } ^ { d } )$ , and suppose that at least one of them lies in $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { d } )$ . Then $\mu = \mu ^ { \prime }$ if and only $i f \mathcal { L } ( \mu )$ and $\mathcal { L } ( \mu ^ { \prime } )$ agree on some open neighborhood $o f 0$

Proof of Corollary 17. Suppose that ${ \mathcal { L } } ( \mu )$ and $\mathcal { L } ( \mu ^ { \prime } )$ agree on some open neighborhood of 0. Without loss of generality, assume that $\mu \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { d } )$ . Let $A \subset  { \mathbb { R } } ^ { d }$ be a nonempty open set containing 0 such that ${ \mathcal { L } } ( \mu )$ is finite on A and

$$
{ \mathcal { L } } ( \mu ) ( t ) = { \mathcal { L } } ( \mu ^ { \prime } ) ( t ) \qquad { \mathrm { f o r ~ a l l ~ } } t \in A .
$$

Then $\mathcal { L } ( \mu ^ { \prime } )$ is also finite on A (it equals ${ \mathcal { L } } ( \mu )$ there). Set $\boldsymbol { \nu } = \boldsymbol { \mu } - \boldsymbol { \mu } ^ { \prime }$ . Then

$$
{ \mathcal { L } } ( \nu ) ( t ) = { \mathcal { L } } ( \mu ) ( t ) - { \mathcal { L } } ( \mu ^ { \prime } ) ( t ) = 0 \qquad { \mathrm { f o r ~ a l l ~ } } t \in A .
$$

Write the Jordan decomposition $\nu = \nu ^ { + } - \nu ^ { - }$ with $\nu ^ { \pm }$ finite nonnegative measures, and let $c _ { \pm } = \nu ^ { \pm } ( \mathbb { R } ^ { d } )$ Consider first the case $c _ { + } = 0 ~ \mathrm { o r } ~ c _ { - } = 0$ . Then ν is either nonnegative or nonpositive. For concreteness assume ν is nonnegative, the nonpositive case is analogous. Since $e ^ { \langle \bar { x } , t \rangle } > 0$ for all $x , t ,$ it then follows that the Laplace transform $\mathcal { L } ( \nu )$ is nonnegative on A. As $\mathcal { L } ( \nu ) \equiv 0$ on A, it follows that

$$
\nu ( \mathbb { R } ^ { d } ) = \mathcal { L } ( \nu ) ( 0 ) = 0 .
$$

A nonnegative measure with total mass zero is identically zero, so $\nu = 0$ . Hence $\nu = 0$ and the result follows.

Now consider the case $c _ { + } > 0$ and $c _ { - } > 0$ . Define the probability measures $P = \nu ^ { + } / c _ { + }$ and $Q = \nu ^ { - } / c _ { - }$ By the definition of integration with respect to a signed measure, the finiteness of $\mathcal { L } ( \nu )$ on A implies that $\mathcal { L } ( \nu ^ { + } )$ and $\mathcal { L } ( \nu ^ { - } )$ are finite on A. Hence ${ \mathcal { L } } ( P )$ and $\mathcal { L } ( Q )$ are finite on A, and by linearity,

$$
0 = \mathcal { L } ( \nu ) = c _ { + } \mathcal { L } ( P ) - c _ { - } \mathcal { L } ( Q ) \quad \mathrm { o n ~ } A .
$$

Thus, for all $t \in A$

$$
c _ { + } { \mathcal { L } } ( P ) ( t ) = c _ { - } { \mathcal { L } } ( Q ) ( t ) .
$$

By the definition of Laplace transform, evaluating at $t = 0$ gives $c _ { + } = c _ { - } > 0$ , hence

$$
{ \mathcal { L } } ( P ) ( t ) = { \mathcal { L } } ( Q ) ( t ) \qquad { \mathrm { ~ f o r ~ a l l ~ } } t \in A .
$$

By Theorem 16, this implies $P = Q$ . Thus $\nu ^ { + } = \nu ^ { - }$ and so $\nu = 0 ,$ , i.e. $\mu = \mu ^ { \prime }$

Conversely, suppose that $\mu = \mu ^ { \prime }$ . Then $\mathcal { L } ( \mu ) = \mathcal { L } ( \mu ^ { \prime } )$ on some open neighborhood of 0.

A second key ingredient in the proof of Proposition 15 is the analyticity of Laplace transforms. The following standard result, stated for probability measures, will be used.

Proposition 18 ([86, Exercise 8, Chapter 5]). For a probability measure µ on R, if $\mathcal { L } ( \mu )$ is finite on some nonempty open interval $I \subseteq \mathbb { R }$ then it has an analytic extension on the strip $\{ z \in \mathbb { C } : \Re ( z ) \in I \}$

A corresponding analyticity statement holds for finite signed measures.

Corollary 19. For a finite signed measure µ on $\mathbb { R } , i f \mathcal { L } ( \mu )$ is finite on some nonempty open interval $I \subseteq \mathbb { R }$ then it has an analytic extension on the strip $\{ z \in \mathbb { C } : \Re ( z ) \in I \}$

Proof of Corollary 19. If $\mu = 0 ;$ then $\mathcal { L } ( \mu ) = 0$ and the claim is trivial.

Let $\mu = \mu ^ { + } - \mu ^ { - }$ be the Jordan decomposition, and write $c _ { \pm } = \mu ^ { \pm } ( \mathbb { R } )$ . If $c _ { + } = 0$ , then $\mu = - \mu ^ { - }$ and

$$
{ \mathcal { L } } ( \mu ) = - { \mathcal { L } } ( \mu ^ { - } ) = - c _ { - } { \mathcal { L } } ( \mu ^ { - } / c _ { - } ) ,
$$

where $\mu ^ { - } / c _ { - }$ is a probability measure. Since ${ \mathcal { L } } ( \mu )$ is finite on I, so is $\mathcal { L } ( \mu ^ { - } / c _ { - } )$ , and Proposition 18 implies that $\mathcal { L } ( \mu ^ { - } / c _ { - } )$ admits an analytic extension to the strip $\{ z \in \mathbb { C } : \Re z \in I \}$ . The same then holds for $\mathcal { L } ( \mu )$ The case $c _ { - } = 0$ is analogous.

Assume now that $c _ { + } > 0$ and $c _ { - } > 0 .$ , and write

$$
\mu = c _ { + } \left( \mu ^ { + } / c _ { + } \right) - c _ { - } \left( \mu ^ { - } / c _ { - } \right) ,
$$

where $\mu ^ { + } / c _ { + }$ and $\mu ^ { - } / c _ { - }$ are probability measures. For $t \in I ,$ , finiteness of $\mathcal { L } ( \mu ) ( t )$ implies that both $\mathcal { L } ( \mu ^ { + } ) ( t )$ and $\mathcal { L } ( \mu ^ { - } ) ( t )$ are finite, since by the definition of integration with respect to a signed measure the positive and negative parts must be finite individually. Hence $\mathcal { L } ( \mu ^ { + } / c _ { + } ) ( t )$ and $\mathcal { L } ( \mu ^ { - } / c _ { - } ) ( t )$ are also finite.

Applying Proposition 18 to the probability measures $\mu ^ { + } / c _ { + }$ and $\mu ^ { - } / c _ { - } ,$ it follows that their Laplace transforms admit analytic extensions to the strip $\{ z \in \mathbb { C } : \Re z \in I \}$ . By linearity of the Laplace transform,

$$
\mathscr { L } ( \mu ) ( z ) = c _ { + } \mathscr { L } ( \mu ^ { + } / c _ { + } ) ( z ) - c _ { - } \mathscr { L } ( \mu ^ { - } / c _ { - } ) ( z )
$$

on this strip, and hence ${ \mathcal { L } } ( \mu )$ itself extends analytically to $\{ z \in \mathbb { C } : \Re z \in I \}$

It is now possible to prove Proposition 15.

Proof of Proposition 15. Suppose that $\mu = \mu ^ { \prime }$ . Then ${ \mathcal { L } } ( \mu ) = { \mathcal { L } } ( \mu ^ { \prime } )$ , and hence they agree on any set having an accumulation point at 0.

Conversely, suppose that there exists a set $A \subset \mathbb { R }$ having an accumulation point at 0 such that

$$
{ \mathcal { L } } ( \mu ) ( t ) = { \mathcal { L } } ( \mu ^ { \prime } ) ( t ) \qquad { \mathrm { f o r ~ a l l ~ } } t \in A .
$$

By Corollary 19, both ${ \mathcal { L } } ( \mu )$ and $\mathcal { L } ( \mu ^ { \prime } )$ admit analytic extensions to a complex strip containing 0. Since they agree on a set having an accumulation point at 0, the Identity Theorem implies that they agree on some open neighborhood of 0. By Corollary 17, it follows that $\mu = \mu ^ { \prime } .$ □

The preceding section establishes the basic properties of the Laplace transform needed to prove Theorem 1. The remaining ingredient is the fact that, for suficiently regular classes of signed measures, the full measure can be recovered when a countably infinite collection of linear pushforward measures agree. This may be viewed as an analogue of the Cram´er–Wold theorem, which states that probability measures on $\mathbb { R } ^ { d }$ are determined by their one-dimensional linear pushforwards.

The required regularity assumption is the Carleman condition. A d-dimensional probability measure $\mu$ is said to satisfy the Carleman condition if its moments

$$
m _ { n } = \int \| x \| _ { 2 } ^ { n } d \mu ( x )
$$

satisfy

$$
\sum _ { n \in \mathbb { N } } m _ { n } ^ { - 1 / n } = \infty .
$$

The Carleman condition guarantees that $\mu$ is uniquely determined by its moments. Moreover, any probability measure whose Laplace transform exists satisfies the Carleman condition [40, bottom of p. 203], so this assumption is automatically satisfied in the present setting.

An early version of the theorem used below was established in [38] for compactly supported distributions, extended to densities satisfying the Carleman condition in [39], and generalized to d-dimensional measures in [40]. Let $S ^ { 1 }$ denote the unit circle in $\mathbb { R } ^ { 2 }$

Proposition 20 ([39]). Let $\mu$ and $\mu ^ { \prime }$ be probability measures on $\mathbb { R } ^ { 2 }$ , with $\mu$ satisfying the Carleman condition (in particular, this holds whenever $\mu$ has an existing Laplace transform). Then $\mu = \mu ^ { \prime } \textit { i f }$ and only $i f ,$ for $X \sim \mu$ and $X ^ { \prime } \sim \mu ^ { \prime }$

$$
\mathcal { L } ( \langle X , c \rangle ) = \mathcal { L } ( \langle X ^ { \prime } , c \rangle ) ,
$$

for countably infinitely many $c \in S ^ { 1 }$

The next corollary extends the previous result from probability measures to finite signed measures.

Corollary 21 (Signed–measure version of [39]). Let $\mu , \mu ^ { \prime } \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ . Then $\mu = \mu ^ { \prime }$ if and only if

$$
\pi _ { c } \# \mu = \pi _ { c } \# \mu ^ { \prime } ,
$$

for countably infinitely many $c \in S ^ { 1 }$

Proof of Corollary 21. Suppose that $\mu = \mu ^ { \prime }$ . Then $\pi _ { c } \# \mu = \pi _ { c } \# \mu ^ { \prime }$ for all $c \in S ^ { 1 }$ , and in particular for every c in any countably infinite subset of $S ^ { 1 }$

Conversely, suppose that there exists a countably infinite set $A \subseteq S ^ { 1 }$ such that

$$
\pi _ { c } \# \mu = \pi _ { c } \# \mu ^ { \prime }
$$

for all $c \in A$ . Write the Jordan decompositions

$$
\mu = \mu ^ { + } - \mu ^ { - } , \qquad \mu ^ { \prime } = \mu ^ { \prime + } - \mu ^ { \prime - } .
$$

Note that the nonnegative measures $\mu ^ { + } , \mu ^ { - } , \mu ^ { \prime + } , \mu ^ { \prime - }$ all belong to $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ . By linearity of the pushforward operator, for every $c \in A$

$$
\pi _ { c } \# \mu ^ { + } - \pi _ { c } \# \mu ^ { - } \ = \ \pi _ { c } \# \mu ^ { \prime + } - \pi _ { c } \# \mu ^ { \prime - } .\tag{21}
$$

Consider first the case where one of $\mu$ or $\mu ^ { \prime }$ is the zero measure. Without loss of generality, assume that $\mu ^ { \prime } = 0$ . Then

$$
\pi _ { c } \# \mu ^ { + } = \pi _ { c } \# \mu ^ { - }
$$

for all $c \in A$

Assume for contradiction that $\mu ^ { + } \neq 0$ . Since $\mu ^ { + }$ is a nonnegative measure and

$$
\mu ^ { + } ( { \mathbb R } ^ { 2 } ) = ( \pi _ { c } \# \mu ^ { + } ) ( { \mathbb R } ) = ( \pi _ { c } \# \mu ^ { - } ) ( { \mathbb R } ) = \mu ^ { - } ( { \mathbb R } ^ { 2 } )
$$

for every $c \in A$ , it follows that $\mu ^ { - } ( \mathbb { R } ^ { 2 } ) = \mu ^ { + } ( \mathbb { R } ^ { 2 } ) > 0$ , and hence $\mu ^ { - } \neq 0$

Define

$$
P = \frac { \mu ^ { + } } { \mu ^ { + } ( \mathbb { R } ^ { 2 } ) } , \qquad Q = \frac { \mu ^ { - } } { \mu ^ { - } ( \mathbb { R } ^ { 2 } ) } .
$$

Then $P$ and $Q$ are probability measures satisfying

$$
\pi _ { c } \# P = \pi _ { c } \# Q
$$

for all $c \in A$ . Since $\mu \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ , the measures $\mu ^ { + }$ and $\mu ^ { - }$ also belong to $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ . Consequently, $P$ and $Q$ satisfy the Carleman condition. Proposition 20 therefore implies that $P = Q .$ , and hence $\mu ^ { + } = \mu ^ { - }$ . Thus $\mu = 0$ , a contradiction. This establishes the result whenever one of $\mu$ or $\mu ^ { \prime }$ is the zero measure.

Returning to (21), assume that both $\mu$ and $\mu ^ { \prime }$ are nonzero. For every $c \in A$

$$
\pi _ { c } \# ( \mu ^ { + } + \mu ^ { \prime - } ) = \pi _ { c } \# ( \mu ^ { - } + \mu ^ { \prime + } ) .
$$

Define the finite nonnegative measures

$$
\alpha = \mu ^ { + } + \mu ^ { \prime - } , \qquad \beta = \mu ^ { - } + \mu ^ { \prime + } ,
$$

which belong to $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ . Then

$$
\pi _ { c } \# \alpha = \pi _ { c } \# \beta
$$

for all $c \in A .$ . It therefore follows that

$$
\alpha ( \mathbb { R } ^ { 2 } ) = ( \pi _ { c } \# \alpha ) ( \mathbb { R } ) = ( \pi _ { c } \# \beta ) ( \mathbb { R } ) = \beta ( \mathbb { R } ^ { 2 } )
$$

for every $c \in A$ . Since both $\mu$ and $\mu ^ { \prime }$ are nonzero, the measures α and $\beta$ are also nonzero. Define

$$
\bar { \alpha } = \frac { \alpha } { \alpha ( \mathbb { R } ^ { 2 } ) } , \qquad \bar { \beta } = \frac { \beta } { \beta ( \mathbb { R } ^ { 2 } ) } .
$$

Then $\bar { \alpha }$ and $\bar { \beta }$ are probability measures satisfying

$$
\pi _ { c } \# \bar { \alpha } = \pi _ { c } \# \bar { \beta }
$$

for all $c \in A$

Since $\bar { \alpha } , \bar { \beta } \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ , both satisfy the Carleman condition. Proposition 20 therefore implies that

$$
{ \bar { \alpha } } = { \bar { \beta } } .
$$

Since $\alpha ( \mathbb { R } ^ { 2 } ) = \beta ( \mathbb { R } ^ { 2 } )$ , it follows that $\alpha = \beta$

Finally,

$$
\mu - \mu ^ { \prime } = ( \mu ^ { + } - \mu ^ { - } ) - ( \mu ^ { \prime + } - \mu ^ { \prime - } ) = ( \mu ^ { + } + \mu ^ { \prime - } ) - ( \mu ^ { - } + \mu ^ { \prime + } ) = \alpha - \beta = 0 ,
$$

and hence $\mu = \mu ^ { \prime }$

The proof of Theorem 1 will additionally require a measure-theoretic analogue of the familiar identity from probability theory that the moment-generating function of a linear combination of independent random variables factorizes. Establishing this identity for finite signed measures requires a substitution formula for pushforward measures. The following standard result is first recalled for finite nonnegative measures.

Lemma 22 ([86, Lemma 1.22]). Consider a finite nonnegative measure space $( \Omega , { \mathcal { F } } , \mu )$ , a measurable space (S, S), and measurable mappings $f : \Omega  S$ and $g : S  \mathbb { R }$ . Then

$$
\int g ( f ( x ) ) d \mu ( x ) \ = \ \int g ( y ) d ( f \# \mu ) ( y ) ,
$$

whenever either side exists (possibly as an extended real value); that is, $i f$ one side exists, then so does the other, and the two are equal.

The following lemma extends Lemma 22 to finite signed measures.

Lemma 23. Consider a finite signed measure space $( \Omega , { \mathcal { F } } , \mu )$ , a measurable space $( S , S )$ , and measurable mappings $f : \Omega  S$ and $g : S  \mathbb { R }$ . Then

$$
\int g ( f ( x ) ) d \mu ( x ) \ = \ \int g ( y ) d ( f \# \mu ) ( y ) ,
$$

whenever the left-hand side is finite.

Proof of Lemma 23. Write the Jordan decomposition $\mu = \mu ^ { + } - \mu ^ { - }$ , where $\mu ^ { \pm }$ are finite nonnegative measures. Since pushforward is linear on finite signed measures, it follows that

$$
f \# \mu = f \# \mu ^ { + } - f \# \mu ^ { - } .
$$

By the definition of integration with respect to a signed measure, $\begin{array} { r } { \int g ( f ( x ) ) d \mu ^ { + } ( x ) } \end{array}$ and $\begin{array} { r } { \int g ( f ( x ) ) d \mu ^ { - } ( x ) } \end{array}$ are both finite. Applying Lemma 22 to the nonnegative measures $\mu ^ { \pm }$

$$
\int g ( f ( x ) ) d \mu ^ { \pm } ( x ) = \int g ( y ) d ( f \# \mu ^ { \pm } ) ( y ) .
$$

Subtracting the two identities yields

$$
\int g ( f ( x ) ) d \mu ( x ) = \int g ( y ) d ( f \# \mu ) ( y ) .
$$

□

The primary application of Lemma 23 in this work is to linear projections. For $\nu \in \mathcal { M } ( \mathbb { R } ^ { d } ) , c \in \mathbb { R } ^ { d }$ , and $\pi _ { c } ( x ) = \langle c , x \rangle$

$$
\int _ { \mathbb { R } } g ( y ) d ( \pi _ { c } \# \nu ) ( y ) = \int _ { \mathbb { R } ^ { d } } g ( \langle c , x \rangle ) d \nu ( x ) ,
$$

whenever the right-hand side is finite. The pushforward operator is also linear on finite signed measures:

$$
f \# ( a \mu + b \nu ) = a f \# \mu + b f \# \nu .
$$

The next lemma combines these observations with the product structure of low-rank measures to obtain a factorization formula for Laplace transforms.

Lemma 24 (Laplace transform of a factorizable measure). Let $\mu , \nu \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } )$ and let $a , b \in \mathbb { R }$ . For any t such that $\mathcal { L } ( \mu ) ( a t )$ and $\mathcal { L } ( \nu ) ( b t )$ are finite,

$$
\mathcal { L } \big ( \pi _ { [ a , b ] } \# ( \mu \times \nu ) \big ) ( t ) = \mathcal { L } ( \mu ) ( a t ) \mathcal { L } ( \nu ) ( b t ) .
$$

Proof of Lemma 24. Fix $t \in \mathbb { R }$ such that $\mathcal { L } ( \mu ) ( a t )$ and $\mathcal { L } ( \nu ) ( b t )$ are finite. Write the Jordan decompositions

$$
\mu = \mu ^ { + } - \mu ^ { - } , ~ \nu = \nu ^ { + } - \nu ^ { - } .
$$

Let $f ( x ) = e ^ { a t x }$ and $g ( y ) = e ^ { b t y }$ . Then

$$
\begin{array} { r l } {  { \mathcal { L } ( \mu ) ( a t ) \mathcal { L } ( \nu ) ( b t ) = ( \int f d \mu ^ { + } - \int f d \mu ^ { - } ) ( \int g d \nu ^ { + } - \int g d \nu ^ { - } ) } } \\ & { = ( \int f d \mu ^ { + } ) ( \int g d \nu ^ { + } ) + ( \int f d \mu ^ { - } ) ( \int g d \nu ^ { - } ) } \\ & { ~ - ( \int f d \mu ^ { + } ) ( \int g d \nu ^ { - } ) - ( \int f d \mu ^ { - } ) ( \int g d \nu ^ { + } ) , } \end{array}\tag{22}
$$

where all terms are finite. Hence, by the product-measure factorization identity [83, Chapter 2, Exercise 51(b)], (22) becomes

$$
\begin{array} { r l } { \mathcal { L } ( \boldsymbol { \mu } ) ( a t ) \mathcal { L } ( \boldsymbol { \nu } ) ( b t ) = \displaystyle \int _ { \mathbb { R } ^ { 2 } } f ( \boldsymbol { x } ) g ( y ) d ( \boldsymbol { \mu } ^ { + } \times \boldsymbol { \nu } ^ { + } ) ( \boldsymbol { x } , y ) + \displaystyle \int _ { \mathbb { R } ^ { 2 } } f ( \boldsymbol { x } ) g ( y ) d ( \boldsymbol { \mu } ^ { - } \times \boldsymbol { \nu } ^ { - } ) ( \boldsymbol { x } , y ) } \\ { \displaystyle } & { - \displaystyle \int _ { \mathbb { R } ^ { 2 } } f ( \boldsymbol { x } ) g ( y ) d ( \boldsymbol { \mu } ^ { + } \times \boldsymbol { \nu } ^ { - } ) ( \boldsymbol { x } , y ) - \displaystyle \int _ { \mathbb { R } ^ { 2 } } f ( \boldsymbol { x } ) g ( y ) d ( \boldsymbol { \mu } ^ { - } \times \boldsymbol { \nu } ^ { + } ) ( \boldsymbol { x } , y ) . } \end{array}
$$

By the definition of integration with respect to a signed measure and the Jordan decomposition

$$
\mu \times \nu = { \left( \mu ^ { + } \times \nu ^ { + } + \mu ^ { - } \times \nu ^ { - } \right) } - { \left( \mu ^ { + } \times \nu ^ { - } + \mu ^ { - } \times \nu ^ { + } \right) } ,
$$

it follows that

$$
\mathscr { L } ( \mu ) ( a t ) \mathscr { L } ( \nu ) ( b t ) = \int _ { \mathbb { R } ^ { 2 } } f ( x ) g ( y ) d ( \mu \times \nu ) ( x , y ) .
$$

Substituting $f ( x ) = e ^ { a t x }$ and $g ( y ) = e ^ { b t y }$ , this becomes

$$
\mathscr { L } ( \mu ) ( a t ) \mathscr { L } ( \nu ) ( b t ) = \int _ { \mathbb { R } ^ { 2 } } e ^ { t ( a x + b y ) } d ( \mu \times \nu ) ( x , y ) .
$$

The equality above shows that the right-hand side is finite. Therefore Lemma 23, applied to the projection $\pi _ { [ a , b ] }$ , yields

$$
\int _ { \mathbb { R } ^ { 2 } } e ^ { t ( a x + b y ) } d ( \mu \times \nu ) ( x , y ) = { \mathcal { L } } { \big ( } \pi _ { [ a , b ] } \# ( \mu \times \nu ) { \big ) } ( t ) .
$$

Hence

$$
\mathcal { L } \big ( \pi _ { [ a , b ] } \# ( \mu \times \nu ) \big ) ( t ) = \mathcal { L } ( \mu ) ( a t ) \mathcal { L } ( \nu ) ( b t ) .
$$

The following corollary makes explicit the structural property that will be central to the proof of Theorem 1.

Corollary 25 (Linearity for finite mixtures). Let

$$
\lambda = \sum _ { i = 1 } ^ { r } w _ { i } ( { \boldsymbol { \mu } } _ { i } \times \nu _ { i } ) , \qquad w _ { i } \in \mathbb { R } ,
$$

where each $\mu _ { i } , \nu _ { i } \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } )$ . Then $\lambda \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ . Moreover, for every $a , b \in \mathbb { R }$ , there exists $\alpha > 0$ such that

$$
\mathcal { L } ( \pi _ { [ a , b ] } \# \lambda ) ( t ) = \sum _ { i = 1 } ^ { r } w _ { i } \mathcal { L } ( \mu _ { i } ) ( a t ) \mathcal { L } ( \nu _ { i } ) ( b t )
$$

for all $t \in ( - \alpha , \alpha )$

## 9.3 A Rank Criterion for Laplace-Transform Matrices and Proofs of Theorem 1 and Corollary 2

The next proposition provides a key structural ingredient in the proof of Theorem 1. It shows that linear independence of a finite collection of measures can be characterized through a full-rank condition on a structured matrix of Laplace-transform evaluations.

Proposition 26. Let $k \in \mathbb N$ and let $\mu _ { 1 } , \ldots , \mu _ { k } \in \mathcal { M } _ { \exp } ( \mathbb { R } )$ . For each $i = 1 , \ldots , k$ , write $\mathcal { L } _ { i } ( t ) = \mathcal { L } ( \mu _ { i } ) ( t )$

Then the measures $\mu _ { 1 } , \ldots , \mu _ { k }$ are linearly independent if and only if for all nonzero $a , b \in \mathbb { R }$ with $| a | \neq 1$ there exists $\alpha > 0$ such that, for all $t \in ( - \alpha , 0 ) \cup ( 0 , \alpha )$ , the matrix

$$
L ( t ) \triangleq \left[ \begin{array} { c c c c } { \mathcal { L } _ { 1 } ( a ^ { 0 } b t ) } & { \mathcal { L } _ { 1 } ( a ^ { 1 } b t ) } & { \cdot \cdot } & { \mathcal { L } _ { 1 } ( a ^ { k - 1 } b t ) } \\ { \mathcal { L } _ { 2 } ( a ^ { 0 } b t ) } & { \mathcal { L } _ { 2 } ( a ^ { 1 } b t ) } & { \cdot \cdot } & { \mathcal { L } _ { 2 } ( a ^ { k - 1 } b t ) } \\ { \vdots } & { \vdots } & { \cdot } & { \vdots } \\ { \mathcal { L } _ { k } ( a ^ { 0 } b t ) } & { \mathcal { L } _ { k } ( a ^ { 1 } b t ) } & { \cdot \cdot } & { \mathcal { L } _ { k } ( a ^ { k - 1 } b t ) } \end{array} \right]\tag{23}
$$

has full rank.

The matrix in (23) has a Vandermonde-type structure in the scaling parameter $^ { a , }$ with entries given by evaluations of Laplace transforms at geometrically spaced points.

For a matrix M and index sets $I , J \subset \mathbb { N }$ , let $M _ { I \times J }$ denote the submatrix of M obtained by restricting to the rows indexed by I and the columns indexed by J.

Proof of Proposition 26. Suppose that $\mu _ { 1 } , \ldots , \mu _ { k }$ are linearly dependent. Since the Laplace transform is linear, the corresponding functions $\mathcal { L } _ { 1 } , \ldots , \mathcal { L } _ { k }$ are also linearly dependent. It follows that, for every $t ,$ the rows of $L ( t )$ satisfy the same linear relation and hence $L ( t )$ does not have full rank anywhere all entries exist. Conversely, suppose that Conversely, suppose that are linearly independent. The proof proceeds by induction on k. are linearly independent. The proof proceeds by induction on k

$\mu _ { 1 } , \ldots , \mu _ { k }$

Base case $( k = 1 )$ . If no such α exists then there exists a sequence $t _ { i } \to 0$ such that $L ( t _ { i } ) = 0$ and $\mu _ { 1 } \equiv 0$ due to Proposition 15, contradicting linear independence. Note that the preceding argument shows more generally that if $\mu \not \equiv 0$ and its Laplace transform exists, then there exists $\beta > 0$ such that $\mathcal { L } ( \mu ) ( t ) \neq 0$ for all $t \in ( - \beta , \beta ) \setminus \{ 0 \}$

Induction step. Assume that the proposition holds for some $k \geq 1$ . The result is established for $k + 1$ First consider the case $| a | < 1$ and $b = 1$ ; the cases $| a | > 1$ and $b \neq 1$ are handled afterward.

Assume for the sake of contradiction that no such α exists. Then there is a sequence $t _ { i } \neq 0$ with $t _ { i } \to 0$ such that $L ( t _ { i } )$ is rank-deficient for all i. Since $t \mapsto$ det $L ( t )$ is analytic near $0 ,$ it follows that det $L \equiv 0$ on some open interval I containing 0, by the Identity Theorem. Shrinking I if necessary, assume that I is symmetric about zero.

By the inductive hypothesis there exists $\alpha ^ { \prime } > 0$ such that the minor ${ \cal L } ( t ) _ { \{ 1 , \dots , k \} \times \{ 1 , \dots , k \} }$ (the upper-left $k \times k$ block of $L ( t ) )$ has full rank for all $t \in ( - \alpha ^ { \prime } , \alpha ^ { \prime } ) \setminus \{ 0 \}$

Fix $t ^ { \ast } \in \big ( ( - \alpha ^ { \prime } , \alpha ^ { \prime } ) \setminus \{ 0 \} \big ) \cap I$ . Then $L ( t ^ { * } )$ is rank-deficient while ${ \cal L } ( t ^ { * } ) _ { \{ 1 , \dots , k \} \times \{ 1 , \dots , k \} }$ has full rank. Hence there exists a unique $\gamma \in \mathbb { R } ^ { k }$ with

$$
L ( t ^ { * } ) _ { \{ k + 1 \} \times \{ 1 , \dots , k \} } = \sum _ { i = 1 } ^ { k } \gamma _ { i } L ( t ^ { * } ) _ { \{ i \} \times \{ 1 , \dots , k \} } .
$$

Because $L ( t ^ { * } )$ is rank-deficient while $L ( t ^ { * } ) _ { \{ 1 , \dots , k \} \times \{ 1 , \dots , k + 1 \} }$ has full rank, the $\left( k { + } 1 \right)$ )-st row lies in the span of the first k rows and hence

$$
L ( t ^ { * } ) _ { \{ k + 1 \} \times \{ 1 , \ldots , k + 1 \} } = \sum _ { i = 1 } ^ { k } \gamma _ { i } L ( t ^ { * } ) _ { \{ i \} \times \{ 1 , \ldots , k + 1 \} } .
$$

Within the present contradiction argument, it will be shown by induction on $j$ that

$$
L ( a ^ { j } t ^ { * } ) _ { \{ k + 1 \} \times \{ 1 , \dots , k + 1 \} } = \sum _ { i = 1 } ^ { k } \gamma _ { i } L ( a ^ { j } t ^ { * } ) _ { \{ i \} \times \{ 1 , \dots , k + 1 \} } ,\tag{24}
$$

for all $j \in \mathbb { N } \cup \{ 0 \}$

It has already been shown that (24) holds for $j = 0$ , with $\gamma$ unique. Assume that (24) holds for some $j \in \mathbb { N } \cup \{ 0 \}$ . It will now be shown that (24) also holds for $j + 1$

From the definition of $L$

$$
L ( a ^ { j } t ^ { * } ) _ { \{ 1 , \dots , k \} \times \{ 2 , \dots , k + 1 \} } = L ( a ^ { j + 1 } t ^ { * } ) _ { \{ 1 , \dots , k \} \times \{ 1 , \dots , k \} } .
$$

Since $| a | < 1$

$$
a ^ { j + 1 } t ^ { * } \in \left( ( - \alpha ^ { \prime } , \alpha ^ { \prime } ) \setminus \{ 0 \} \right) \cap I .
$$

By the induction hypothesis on $k ,$ both $k \times k$ matrix minors appearing in the equality above have full rank. By the inductive hypothesis on $j , \gamma$ satisfies the following equation and, by the established full-rank property, is its unique solution,

$$
L ( a ^ { j } t ^ { * } ) _ { \{ k + 1 \} \times \{ 2 , 3 , \ldots , k + 1 \} } = \sum _ { i = 1 } ^ { k } \gamma _ { i } L ( a ^ { j } t ^ { * } ) _ { \{ i \} \times \{ 2 , 3 , \ldots , k + 1 \} } .
$$

Now observe that, similarly to before,

$$
L ( a ^ { j + 1 } t ^ { * } ) _ { \{ 1 , \dots , k + 1 \} \times \{ 1 , \dots , k \} } = L ( a ^ { j } t ^ { * } ) _ { \{ 1 , \dots , k + 1 \} \times \{ 2 , 3 , \dots , k + 1 \} } ,
$$

hence the same $\gamma$ uniquely satisfies

$$
L ( a ^ { j + 1 } t ^ { * } ) _ { \{ k + 1 \} \times \{ 1 , \dots , k \} } = \sum _ { i = 1 } ^ { k } \gamma _ { i } L ( a ^ { j + 1 } t ^ { * } ) _ { \{ i \} \times \{ 1 , \dots , k \} } .\tag{25}
$$

Because $a ^ { j + 1 } t ^ { * } \in \left( \left( - \alpha ^ { \prime } , \alpha ^ { \prime } \right) \backslash \left\{ 0 \right\} \right) \cap I$ , the matrix $L ( a ^ { j + 1 } t ^ { * } )$ is rank deficient. Moreover, the first k rows are linearly independent, since

$$
{ \cal L } ( a ^ { j + 1 } t ^ { * } ) _ { \{ 1 , \dots , k \} \times \{ 1 , \dots , k \} }
$$

has full rank. Therefore the $\left( k + 1 \right)$ -st row of $L ( a ^ { j + 1 } t ^ { * } )$ lies in the span of the first $k$ rows. By the uniqueness established in (25), the coeficients in this linear relation must equal $\gamma .$ . It follows that (25) remains valid when the $( k + 1 )$ -st column is included:

$$
L ( a ^ { j + 1 } t ^ { * } ) _ { \{ k + 1 \} \times \{ 1 , \dots , k + 1 \} } = \sum _ { i = 1 } ^ { k } \gamma _ { i } L ( a ^ { j + 1 } t ^ { * } ) _ { \{ i \} \times \{ 1 , \dots , k + 1 \} } .
$$

This completes the induction on $j .$ Thus (24) holds for all $j$ with $\gamma$ unique. Looking at the entries of the matrix function $L ,$ from (24) it follows that $\begin{array} { r } { \mathcal { L } _ { k + 1 } ( a ^ { j } t ^ { * } ) = \sum _ { i = 1 } ^ { k } \gamma _ { i } \mathcal { L } _ { i } ( a ^ { j } t ^ { * } ) } \end{array}$ for all $j \in \mathbb N$

Since $0 < | a | < 1$ and ${ { t } ^ { * } } \ne 0$ , it follows that $a ^ { j } t ^ { * } \to 0$ . Proposition 15 therefore implies that

$$
\mu _ { k + 1 } = \sum _ { i = 1 } ^ { k } \gamma _ { i } \mu _ { i } ,
$$

contradicting the linear independence of $\{ \mu _ { 1 } , \ldots , \mu _ { k + 1 } \}$ . This completes the induction step and establishes the result for the case $| a | < 1 , b = 1$

For the remaining cases, let $L _ { a } ( t )$ denote the matrix in (23) with $b = 1$ and parameter $a .$ . Then $L _ { a } ( t )$ is obtained from $L _ { a ^ { - 1 } } ( a ^ { k - 1 } t )$ by reversing the order of the columns:

$$
L _ { a } ( t ) { \mathrm { ~ i s ~ t h e ~ c o l u m n - r e v e r s a l ~ o f ~ } } L _ { a ^ { - 1 } } ( a ^ { k - 1 } t ) .
$$

Now assume $| a | > 1$ with $b = 1$ , so that $| a ^ { - 1 } | < 1$ . Since $| a ^ { - 1 } | < 1$ , the $| a | < 1$ case yields $\alpha _ { 0 } > 0$ such that $L _ { a ^ { - 1 } } ( t )$ has full rank for all $t \in ( - \alpha _ { 0 } , \alpha _ { 0 } ) \setminus \{ 0 \}$ . Define

$$
\alpha \triangleq \alpha _ { 0 } \vert a \vert ^ { - ( k - 1 ) } .
$$

Then for every $t \in ( - \alpha , \alpha ) \setminus \{ 0 \}$ one has $a ^ { k - 1 } t \in ( - \alpha _ { 0 } , \alpha _ { 0 } ) \backslash \{ 0 \}$ , and hence $L _ { a ^ { - 1 } } ( a ^ { k ^ { - 1 } } t )$ has full rank. Since $L _ { a } ( t )$ is obtained from $L _ { a ^ { - 1 } } ( a ^ { k ^ { - 1 } } t )$ by a horizontal column reversal, which preserves rank, it follows that $L _ { a } ( t )$ has full rank for all $t \in ( - \alpha , \alpha ) \setminus \{ 0 \}$ , thus taking care of the $| a | > 1$ case.

Now let $L _ { a , b } ( t )$ denote the matrix in (23) with both parameters a and b free. For all admissible $a , b ,$

$$
L _ { a , b } ( t ) = L _ { a , 1 } ( b t ) .
$$

From the case $b = 1$ , there exists $\alpha _ { 0 } > 0$ such that $L _ { a , 1 } ( s )$ has full rank for all $s \in ( - \alpha _ { 0 } , \alpha _ { 0 } ) \setminus \{ 0 \}$ . Define $\alpha \triangleq \alpha _ { 0 } / | b |$ . Then every $t \in ( - \alpha , \alpha ) \setminus \{ 0 \}$ satisfies $b t \in \left( - \alpha _ { 0 } , \alpha _ { 0 } \right) \setminus \left\{ 0 \right\}$ . Hence $L _ { a , b } ( t ) = L _ { a , 1 } ( b t )$ has full rank. □

The proof of Theorem 1 now follows.

Proof of Theorem 1. The proof proceeds by induction, establishing

$$
\pi _ { [ b _ { 1 } , b _ { 2 } a ^ { m } ] ^ { \top } } \# \mu \ = \ \pi _ { [ b _ { 1 } , b _ { 2 } a ^ { m } ] ^ { \top } } \# \mu ^ { \prime } ,\tag{26}
$$

for all $m \in \mathbb { N }$ . Corollary 21 then implies $\mu = \mu ^ { \prime }$ , since $\mu , \mu ^ { \prime } \in \mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } ^ { 2 } )$ by Corollary 25. Base cases. By the hypotheses, (26) holds for $0 \leq m \leq k - 1$

Induction step. For $t , x , y \in \mathbb { R }$ , define

$$
\mathcal { L } _ { [ x , y ] } ( t ) = \mathcal { L } \big ( \pi _ { [ x , y ] } \# \mu \big ) ( t ) \mathrm { a n d } \mathcal { L } _ { [ x , y ] } ^ { \prime } ( t ) = \mathcal { L } \big ( \pi _ { [ x , y ] } \# \mu ^ { \prime } \big ) ( t ) .
$$

Assume that (26) holds for all $0 \leq m ^ { \prime } \leq m$ with $m \geq k - 1$ . From Corollary 25 and the definition of $\mathcal { M } _ { \mathrm { e x p } } ( \mathbb { R } )$ there exists $\alpha > 0$ such that every entry in the $( k + 1 ) \times ( k + 1 )$ matrix $L ( t )$ is finite for all t with $| t | < \alpha$

$$
L ( t ) = \left[ \begin{array} { c c c c c c } { \mathcal { L } _ { [ b _ { 1 a } ^ { 0 } , b _ { 2 a } ^ { 0 } 0 ] } ( t ) } & { \mathcal { L } _ { [ b _ { 1 a } ^ { 0 } , b _ { 2 a } ^ { 0 } 1 ] } ( t ) } & { \cdots } & { \mathcal { L } _ { [ b _ { 1 a } ^ { 0 } , b _ { 2 a } k - 1 ] } ( t ) } & { \left[ \mathcal { L } _ { [ b _ { 1 a } ^ { 0 } , b _ { 2 a } ^ { 0 } m + 1 ] } ( t ) \right] } \\ { \mathcal { L } _ { [ b _ { 1 a } ^ { 1 } , b _ { 2 1 } ] } ( t ) } & { \mathcal { L } _ { [ b _ { 1 a } ^ { 1 } , b _ { 2 a } ^ { 1 } 1 ] } ( t ) } & { \cdots } & { \mathcal { L } _ { [ b _ { 1 a } ^ { 1 } , b _ { 2 a } ^ { 0 } k - 1 ] } ( t ) } & { \mathcal { L } _ { [ b _ { 1 a } ^ { 1 } , b _ { 2 a } m + 1 ] } ( t ) } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } & { \vdots } \\ { \mathcal { L } _ { [ b _ { 1 a } ^ { 0 } , b _ { 2 a } ^ { 0 } 0 ] } ( t ) } & { \mathcal { L } _ { [ b _ { 1 a } ^ { a } , b _ { 2 a } ^ { 1 } ] } ( t ) } & { \cdots } & { \mathcal { L } _ { [ b _ { 1 a } ^ { a } , b _ { 2 a } ^ { 0 } k - 1 ] } ( t ) } & { \mathcal { L } _ { [ b _ { 1 a } ^ { a } , b _ { 2 a } ^ { 0 } m + 1 ] } ( t ) , } \end{array} \right]
$$

with $L ^ { \prime } ( t )$ defined analogously for $\mu ^ { \prime } .$ . By the theorem hypothesis together with the inductive hypothesis, all entries of $L ( t )$ and $L ^ { \prime } ( t )$ coincide for $| t | < \alpha$ , except possibly the boxed entry. Note that some hypotheses may need to be used in rescaled form: for suitable integers $q , r , s .$

$$
[ b _ { 1 } a ^ { q } , b _ { 2 } a ^ { r } ] ^ { \top } = a ^ { s } [ b _ { 1 } a ^ { q - s } , b _ { 2 } a ^ { r - s } ] ^ { \top } ,
$$

so the vectors $[ b _ { 1 } a ^ { q } , b _ { 2 } a ^ { r } ] ^ { \top }$ and $[ b _ { 1 } a ^ { q - s } , b _ { 2 } a ^ { r - s } ] ^ { \top }$ are collinear and rescaling gives

$$
\begin{array} { r } { \mathcal { L } _ { [ b _ { 1 } a ^ { q } , b _ { 2 } a ^ { r } ] } ( t ) = \mathcal { L } _ { [ b _ { 1 } a ^ { q } , b _ { 2 } a ^ { r } ] } ^ { \prime } ( t ) \iff \mathcal { L } _ { [ b _ { 1 } a ^ { q - s } , b _ { 2 } a ^ { r - s } ] } ( a ^ { s } t ) = \mathcal { L } _ { [ b _ { 1 } a ^ { q - s } , b _ { 2 } a ^ { r - s } ] } ^ { \prime } ( a ^ { s } t ) . } \end{array}
$$

It will now be shown that the boxed entry of $L ( t )$ agrees with the corresponding entry of $L ^ { \prime } ( t )$ for all $| t | < \alpha$ This establishes (26).

The proof first treats the case in which $\mu$ has rank exactly k, meaning that the measures $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ are linearly independent, as are $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ (Proposition 13). The lower-rank case is addressed afterward. The argument proceeds in two steps: first, it is shown that there exists ${ \alpha } ^ { \prime } > 0$ such that $L ( t )$ and $L ^ { \prime } ( t )$ both have rank exactly k for all $t \in ( - \alpha ^ { \prime } , \alpha ^ { \prime } ) \setminus \{ 0 \}$ ; second, it is shown that, for such t, the boxed entry is uniquely determined by the remaining entries together with the requirement that the matrix have rank k. Consequently, the boxed entries of $L ( t )$ and $L ^ { \prime } ( t )$ must coincide.

Letting $\mathcal { L } _ { i , j } = \mathcal { L } ( \mu _ { i , j } )$ , define the vector-valued functions

$$
U _ { i } ( t ) = \left[ \mathcal { L } _ { 1 , i } ( t ) \quad \mathcal { L } _ { 2 , i } ( t ) \quad \cdots \quad \mathcal { L } _ { k , i } ( t ) \right] ^ { \top } ,
$$

so that Corollary 25 yields

$$
\begin{array} { r } { \mathcal { L } _ { [ a _ { 1 } , a _ { 2 } ] } ( t ) = U _ { 1 } ( a _ { 1 } t ) ^ { \top } U _ { 2 } ( a _ { 2 } t ) . } \end{array}
$$

Define $U _ { i } ^ { \prime } ( t )$ analogously for $\mu ^ { \prime }$

Observe that, for $| t | < \alpha$

$$
L ( t ) = \left[ \begin{array} { c c c c c } { U _ { 1 } ( b _ { 1 } a ^ { 0 } t ) ^ { \top } } \\ { U _ { 1 } ( b _ { 1 } a ^ { 1 } t ) ^ { \top } } \\ { \vdots } \\ { U _ { 1 } ( b _ { 1 } a ^ { k } t ) ^ { \top } } \end{array} \right] \left[ U _ { 2 } ( b _ { 2 } a ^ { 0 } t )  & { U _ { 2 } ( b _ { 2 } a ^ { 1 } t ) } & { \cdots } & { U _ { 2 } ( b _ { 2 } a ^ { k - 1 } t ) } &  U _ { 2 } ( b _ { 2 } a ^ { m + 1 } t ) \right] .
$$

Since each $U _ { i } : \mathbb { R } \to \mathbb { R } ^ { k }$ , the matrix $L ( t )$ has rank at most k for all $| t | < \alpha$ . The same conclusion holds for $L ^ { \prime } ( t )$ by the corresponding argument applied to $\mu ^ { \prime }$

It will now be shown that there exists $\alpha ^ { \prime } > 0$ such that, for all $t \in ( - \alpha ^ { \prime } , \alpha ^ { \prime } ) \setminus \{ 0 \}$ , the $k \times k$ submatrix

$$
L _ { \{ 2 , 3 , 4 , \ldots , k + 1 \} \times \{ k \} } ( t ) = \left[ \begin{array} { c c c c c } { U _ { 1 } ( b _ { 1 } a ^ { 1 } t ) ^ { \top } } \\ { \vdots } \\ { U _ { 1 } ( b _ { 1 } a ^ { k } t ) ^ { \top } } \end{array} \right] \left[ U _ { 2 } ( b _ { 2 } a ^ { 0 } t )  & { U _ { 2 } ( b _ { 2 } a ^ { 1 } t ) } & { \cdots } &  U _ { 2 } ( b _ { 2 } a ^ { k - 1 } t ) \right]\tag{27}
$$

has full rank k. This submatrix does not involve the boxed entry and therefore agrees with the corresponding submatrix of $L ^ { \prime } ( t )$

To establish this, first observe that the transpose of the right-hand factor in (27) is given by

$$
\left[ \begin{array} { c } { U _ { 2 } ( b _ { 2 } a ^ { 0 } t ) ^ { \top } } \\ { U _ { 2 } ( b _ { 2 } a ^ { 1 } t ) ^ { \top } } \\ { \vdots } \\ { U _ { 2 } ( b _ { 2 } a ^ { k - 1 } t ) ^ { \top } } \end{array} \right] = \left[ \begin{array} { c c c c } { \mathcal { L } _ { 1 , 2 } ( b _ { 2 } a ^ { 0 } t ) } & { \cdots } & { \mathcal { L } _ { k , 2 } ( b _ { 2 } a ^ { 0 } t ) } \\ { \mathcal { L } _ { 1 , 2 } ( b _ { 2 } a ^ { 1 } t ) } & { \cdots } & { \mathcal { L } _ { k , 2 } ( b _ { 2 } a ^ { 1 } t ) } \\ { \vdots } & { \ddots } & { \vdots } \\ { \mathcal { L } _ { 1 , 2 } ( b _ { 2 } a ^ { k - 1 } t ) } & { \cdots } & { \mathcal { L } _ { k , 2 } ( b _ { 2 } a ^ { k - 1 } t ) } \end{array} \right] .
$$

By Proposition ${ \mathrm { 2 6 } } ,$ this matrix has full rank on some punctured open interval about $0 .$

An identical argument applies to the left-hand factor in (27), upon replacing t by at. Consequently, there exists $\alpha ^ { \prime } > 0$ such that both factors exist and have full rank for all $t \in ( - \alpha ^ { \prime } , \alpha ^ { \prime } ) \setminus \{ 0 \}$ . Hence the product matrix $( 2 7 )$ has full rank k throughout this punctured neighborhood. Hence, for each fixed t there exists a unique solution, $\gamma ( t ) \in \mathbb { R } ^ { k }$ , to

$$
L _ { \{ 2 , 3 , 4 , \ldots , k + 1 \} \times \lbrack k \rbrack } ( t ) \gamma ( t ) = L _ { \{ 2 , 3 , 4 , \ldots , k + 1 \} \times \{ k + 1 \} } ( t ) ,\tag{28}
$$

that is, solving for the rightmost column of $L ( t )$ with the first row removed. The same conclusion holds for $L ^ { \prime } ( t )$ , with the identical $\gamma ( t )$ , since only entries where $L ( t )$ and $L ^ { \prime } ( t )$ coincide are involved. Thus $\gamma ( t )$ is the unique solution to

$$
L _ { \{ 2 , 3 , 4 , \ldots , k + 1 \} \times [ k ] } ^ { \prime } ( t ) \gamma ( t ) = L _ { \{ 2 , 3 , 4 , \ldots , k + 1 \} \times \{ k + 1 \} } ^ { \prime } ( t ) .
$$

Since $L ( t )$ has rank at most $k ,$ its rightmost column must lie in the span of its first k columns. The coeficients expressing this column are uniquely determined by the restricted system (28), and hence are $\gamma ( t )$ . Therefore

$$
L _ { [ k + 1 ] \times [ k ] } ( t ) \gamma ( t ) = L _ { [ k + 1 ] \times \{ k + 1 \} } ( t ) .
$$

The same argument applied to $L ^ { \prime } ( t )$ gives

$$
L _ { [ k + 1 ] \times [ k ] } ^ { \prime } ( t ) \gamma ( t ) = L _ { [ k + 1 ] \times \{ k + 1 \} } ^ { \prime } ( t ) .
$$

Since $L ( t )$ and $L ^ { \prime } ( t )$ agree on all entries appearing in the left-hand sides, the right-hand sides agree as well.   
In particular, the boxed entries coincide.

Thus the theorem holds when $\mu$ has rank exactly $k ,$ where k is the parameter in the theorem statement. It remains to consider the case rank $( \mu ) = \ell < k$ . Write

$$
\boldsymbol { \mu } = \sum _ { i = 1 } ^ { \ell } \boldsymbol { \mu } _ { i , 1 } \times \boldsymbol { \mu } _ { i , 2 } ,
$$

with the component families linearly independent. It is first shown that rank $\mathit { \Omega } ( \mu ^ { \prime } ) < k$ as well. Indeed, if rank $( \mu ^ { \prime } ) = k$ , then the already-proved case of the theorem applies with $\mu ^ { \prime }$ in place of $\mu ,$ yielding $\mu ^ { \prime } = \mu .$ This contradicts ran $\operatorname { \varepsilon } ( \mu ) = \ell < k$ . Hence rank $\ : ( \mu ^ { \prime } ) = \ell ^ { \prime } \ :$ for some $\ell ^ { \prime } < k ,$ and one may write

$$
\mu ^ { \prime } = \sum _ { i = 1 } ^ { \ell ^ { \prime } } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } ,
$$

again with linearly independent component families.

Let $k ^ { * } \triangleq \operatorname* { m a x } \{ \ell , \ell ^ { \prime } \}$ . Observe that the 2k pushforward equalities in (4)–(5) contain, as a subset, the $2 k ^ { * }$ equalities obtained by restricting m to $\{ 1 , \ldots , k ^ { * } \}$ and $\{ 0 , \ldots , k ^ { * } - 1 \}$ . Thus the hypotheses of the theorem are satisfied with k replaced by $k ^ { * }$ . Applying the already-established theorem with parameter $k ^ { * }$ yields $\mu = \mu ^ { \prime }$ □

The proof of Corollary 2 relies on the fact that, for finite nonnegative measures, failure of finiteness of the Laplace transform is preserved under convolution with any nonzero positive measure. This property is formalized in the following lemma.

Lemma 27. Let $\mu \in \mathcal { M } ^ { + } ( \mathbb { R } ) \setminus \mathcal { M } _ { \mathrm { e x p } } ^ { + } ( \mathbb { R } )$ . Then for every nonzero $\nu \in \mathcal { M } ^ { + } ( \mathbb { R } )$

$$
\mu \ast \nu \in \mathcal { M } ^ { + } ( \mathbb { R } ) \setminus \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } ) .
$$

Proof of Lemma 27. Since $\mu , \nu \in \mathcal { M } ^ { + } ( \mathbb { R } )$ are finite positive measures, their convolution $\mu \ast \nu$ is again a finite positive measure so $\mu \ast \nu \in \mathcal { M } ^ { + } ( \mathbb { R } )$ ).

Fix $t \in \mathbb { R }$ and define the truncated functions

$$
f _ { n } ( z ) \triangleq e ^ { t z } \wedge n , \qquad n \in \mathbb { N } .
$$

Then each $f _ { n }$ is bounded, Borel measurable, and nonnegative. By [83, Proposition 8.48(b)],

$$
\int f _ { n } d ( \mu * \nu ) = \int _ { \mathbb { R } } \int _ { \mathbb { R } } f _ { n } ( x + y ) d \mu ( x ) d \nu ( y ) .\tag{29}
$$

Since $f _ { n } \uparrow e ^ { t z }$ pointwise as $n \to \infty$ , the monotone convergence theorem gives

$$
\mathcal { L } ( \mu * \nu ) ( t ) = \int e ^ { t z } d ( \mu * \nu ) ( z ) = \operatorname* { l i m } _ { n \to \infty } \int f _ { n } d ( \mu * \nu ) .\tag{30}
$$

Moreover, $f _ { n } ( x + y ) \uparrow e ^ { t ( x + y ) }$ pointwise on $\mathbb { R } ^ { 2 }$ , so applying monotone convergence to the product measure $\mu \times \nu$ yields

$$
\operatorname* { l i m } _ { n \to \infty } \int _ { \mathbb { R } } \int _ { \mathbb { R } } f _ { n } ( x + y ) d \mu ( x ) d \nu ( y ) = \int _ { \mathbb { R } } \int _ { \mathbb { R } } e ^ { t ( x + y ) } d \mu ( x ) d \nu ( y ) .\tag{31}
$$

Combining (29), (30), and (31) gives

$$
\mathcal L ( \mu * \nu ) ( t ) = \int e ^ { t z } d ( \mu * \nu ) ( z ) = \int _ { \mathbb R } \int _ { \mathbb R } e ^ { t ( x + y ) } d \mu ( x ) d \nu ( y ) = \int _ { \mathbb R } \int _ { \mathbb R } e ^ { t x } e ^ { t y } d \mu ( x ) d \nu ( y ) .
$$

Since $( x , y ) \mapsto e ^ { t x } e ^ { t y }$ is nonnegative, Tonelli’s Theorem yields

$$
\mathcal L ( \mu * \nu ) ( t ) = \int _ { \mathbb R } \int _ { \mathbb R } e ^ { t x } e ^ { t y } d \mu ( x ) d \nu ( y ) = \left( \int _ { \mathbb R } e ^ { t x } d \mu ( x ) \right) \left( \int _ { \mathbb R } e ^ { t y } d \nu ( y ) \right) = \mathcal L ( \mu ) ( t ) \mathcal L ( \nu ) ( t ) .
$$

Since $\mu \not \in \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$ , there exists a sequence $( t _ { i } ) _ { i = 1 } ^ { \infty }$ with $t _ { i } \to 0$ and $\mathcal { L } ( \mu ) ( t _ { i } ) = \infty$ for all i. Since ν is a nonzero nonnegative measure, $\mathcal { L } ( \nu ) ( t ) \in ( 0 , \infty ]$ for every $t \in \mathbb { R }$ . Thus

$$
\mathcal { L } ( \mu * \nu ) ( t _ { i } ) = \mathcal { L } ( \mu ) ( t _ { i } ) \mathcal { L } ( \nu ) ( t _ { i } ) = \infty
$$

for all i. Therefore ${ \mathcal { L } } ( \mu * \nu )$ is not finite on any neighborhood of 0, and $\mu \ast \nu \not \in \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$

Proof of Corollary 2. By hypothesis, $\mu$ admits a representation (3) with $\mu _ { i , j } \in \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$ for all $i \in [ k ]$ and $j ~ \in ~ [ 2 ]$ . To apply Theorem 1, it sufices to show that for every i with $\mu _ { i , 1 } ^ { \prime } \times \dot { \mu _ { i , 2 } ^ { \prime } } \neq 0$ , one has $\mu _ { i , 1 } ^ { \prime } , \mu _ { i , 2 } ^ { \prime } \in \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$

Proceeding by contradiction, suppose there exists $i _ { 0 } \in [ k ]$ such that $\mu _ { i _ { 0 } , 1 } ^ { \prime } \times \mu _ { i _ { 0 } , 2 } ^ { \prime } \neq 0$ and $\mu _ { i _ { 0 } , 1 } ^ { \prime } \notin \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$ Relabeling if necessary, assume $i _ { 0 } = 1$ . (The case $\mu _ { i _ { 0 } , 2 } ^ { \prime } \notin \mathcal { M } _ { \mathrm { e x p } } ^ { + } ( \mathbb { R } )$ is analogous.) Since $\mu _ { 1 , 1 } ^ { \prime } \times \dot { \mu } _ { 1 , 2 } ^ { \prime } \neq 0$ , both factors are nonzero.

Let $\boldsymbol { c } = [ c _ { 1 } , c _ { 2 } ] ^ { \top }$ be any direction appearing in (4)–(5), so $c _ { 1 } \neq 0$ and $c _ { 2 } \neq 0$ . By positivity of the measures and linearity of the Laplace transform,

$$
\begin{array} { r l } { \displaystyle \mathcal { L } ( \pi _ { c } \# \mu ^ { \prime } ) ( t ) = \sum _ { i = 1 } ^ { k } \mathcal { L } \big ( \pi _ { c } \# ( \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } ) \big ) ( t ) } & { } \\ { \geq \mathcal { L } \big ( \pi _ { c } \# ( \mu _ { 1 , 1 } ^ { \prime } \times \mu _ { 1 , 2 } ^ { \prime } ) \big ) ( t ) . } \end{array}
$$

It is therefore enough to show that

$$
\pi _ { c } \# ( \mu _ { 1 , 1 } ^ { \prime } \times \mu _ { 1 , 2 } ^ { \prime } ) \notin \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } ) .
$$

Let $S _ { r } ( x ) = r x$ . The pushforward of a product measure under $\pi _ { c } ( x , y ) = c _ { 1 } x + c _ { 2 } y$ is

$$
\pi _ { c } \# ( \mu _ { 1 , 1 } ^ { \prime } \times \mu _ { 1 , 2 } ^ { \prime } ) = ( S _ { c _ { 1 } } \# \mu _ { 1 , 1 } ^ { \prime } ) * ( S _ { c _ { 2 } } \# \mu _ { 1 , 2 } ^ { \prime } ) .
$$

By Lemma 22,

$$
\mathcal { L } ( S _ { c _ { 1 } } \# \mu _ { 1 , 1 } ^ { \prime } ) ( t ) = \mathcal { L } ( \mu _ { 1 , 1 } ^ { \prime } ) ( c _ { 1 } t ) ,
$$

so $S _ { c _ { 1 } } \# \mu _ { 1 , 1 } ^ { \prime } \ \notin \ \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$ , since $c _ { 1 } \neq 0$ . Also $S _ { c _ { 2 } \# \mu _ { 1 , 2 } ^ { \prime } }$ is a nonzero finite positive measure. Lemma 27 therefore gives

$$
\pi _ { c } \# ( \mu _ { 1 , 1 } ^ { \prime } \times \mu _ { 1 , 2 } ^ { \prime } ) \notin \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } ) .
$$

Hence $\pi _ { c } \# \mu ^ { \prime } \notin \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$ . However, $\pi _ { c } \# \mu = \pi _ { c } \# \mu ^ { \prime }$ by hypothesis, while $\pi _ { c } \# \mu \in \mathcal { M } _ { \exp } ^ { + } ( \mathbb { R } )$ , since each component marginal of µ lies in $\mathcal { M } _ { \mathrm { e x p } } ^ { + } ( \mathbb { R } )$ . This is a contradiction. Hence $\mu _ { i , j } ^ { \prime } \in \mathcal { M } _ { \mathrm { e x p } } ^ { \div } ( \mathbb { R } )$ whenever $\mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } \neq 0$ . Any remaining zero terms may be replaced by $0 \times 0$ , so Theorem 1 applies and yields $\mu = \mu ^ { \prime }$ □

## 10 Proofs of Topological and Geometric Results

This section establishes the topological and geometric properties of the silhouette operator. The arguments are conceptually straightforward but rely on the weak<sup>∗</sup> closedness of classes of low-rank measures. In finitedimensional matrix spaces, weak<sup>∗</sup> closedness is equivalent to ordinary closedness, and rank-bounded classes are well known to be closed. Proving weak<sup>∗</sup> closedness for low-rank measures on $\mathbb { R } ^ { 2 }$ , however, is nontrivial and requires tools from Banach space theory, in particular tensor products and the approximation property. All Banach spaces considered in this section are real. For a Banach space X and its dual $X ^ { * }$ , angle brackets denote the canonical dual pairing, with either order permitted:

$$
\langle x , x ^ { * } \rangle = \langle x ^ { * } , x \rangle = x ^ { * } ( x ) , \qquad x \in X , \quad x ^ { * } \in X ^ { * } .
$$

## 10.1 Topological Properties of Low-Rank Measure Classes

The topological and geometric properties of the silhouette operator rely on compactness arguments. Establishing these properties requires showing that the class of low-rank measures is weak<sup>∗</sup> closed. Throughout this subsection, for compact $K _ { 1 } , K _ { 2 } \subset \mathbb { R } , \ M ( K _ { 1 } \times K _ { 2 } )$ is equipped with the weak<sup>∗</sup> topology arising from the canonical identification

$$
{ \mathcal { M } } ( K _ { 1 } \times K _ { 2 } ) \cong C ( K _ { 1 } \times K _ { 2 } ) ^ { * }
$$

given by the Riesz–Markov–Kakutani representation theorem, where $C ( K _ { 1 } \times K _ { 2 } )$ denotes the Banach space of real-valued continuous functions on $K _ { 1 } \times K _ { 2 }$ equipped with the supremum norm. The main result of this subsection is the following.

Theorem 28. Let $K _ { 1 } , K _ { 2 } \subset \mathbb { R }$ both be compact and let $k \in \mathbb N$ . Then $\mathcal { R } { \ } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ is weak<sup>∗</sup> closed.

Because every weak<sup>∗</sup> open set is weakly open and norm open, weak<sup>∗</sup> closed sets are necessarily weakly closed and norm closed. Consequently, $\mathcal { R } { \ } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ is also weakly and norm closed.

The proofs in this subsection make use of tensor products of Banach spaces. A comprehensive treatment may be found in [87]; only the material required for the present arguments is summarized here. Let X and Y be Banach spaces. The algebraic tensor product, denoted $X \otimes Y$ , consists of finite sums of elementary tensors,

$$
\sum _ { i = 1 } ^ { k } \lambda _ { i } x _ { i } \otimes y _ { i } ,\tag{32}
$$

where $x _ { i } \in X , y _ { i } \in Y$ , and $\lambda _ { i }$ are scalars in the underlying field. The algebraic tensor product $X \otimes Y$ is a vector space, with addition and scalar multiplication defined termwise on representations of the form (32). The tensor product is bilinear, meaning that for $x , x _ { 1 } , x _ { 2 } \in X , y , y _ { 1 } , y _ { 2 } \in Y$ , and $\lambda \in \mathbb { R }$

$$
( x _ { 1 } + x _ { 2 } ) \otimes y = x _ { 1 } \otimes y + x _ { 2 } \otimes y ,
$$

$$
x \otimes ( y _ { 1 } + y _ { 2 } ) = x \otimes y _ { 1 } + x \otimes y _ { 2 } ,
$$

and

$$
\lambda ( x \otimes y ) = ( \lambda x ) \otimes y = x \otimes ( \lambda y ) .
$$

In particular,

$$
0 \otimes y = x \otimes 0 = 0 .
$$

Tensor products of Banach spaces admit many natural norms. Two of the most important are the projective and injective tensor product norms. The projective tensor product of $X$ and Y, denoted $X \otimes _ { \pi } Y$ is the algebraic tensor product equipped with the norm $\| \cdot \| .$ <sub>π</sub> defined by

$$
\left\| \sum _ { i = 1 } ^ { k } x _ { i } \otimes y _ { i } \right\| _ { \boldsymbol \pi } = \operatorname* { i n f } \left\{ \sum _ { i = 1 } ^ { k ^ { \prime } } \| x _ { i } ^ { \prime } \| \| y _ { i } ^ { \prime } \| : k ^ { \prime } \in \mathbb { N } , \sum _ { i = 1 } ^ { k ^ { \prime } } x _ { i } ^ { \prime } \otimes y _ { i } ^ { \prime } = \sum _ { i = 1 } ^ { k } x _ { i } \otimes y _ { i } \right\} .
$$

The projective tensor norm may be viewed as an infinite-dimensional analogue of the matrix nuclear norm. The completion of $X \otimes _ { \pi } Y$ is denoted $X { \widehat { \otimes } } _ { \pi } Y$ . A useful structural property of the completed projective tensor product is that every element $u \in X { \widehat { \otimes } } _ { \pi } Y$ admits a representation

$$
u = \sum _ { i = 1 } ^ { \infty } x _ { i } \otimes y _ { i } , \qquad \sum _ { i = 1 } ^ { \infty } \| x _ { i } \| \| y _ { i } \| < \infty ,
$$

see $[ 8 7 ,$ , Proposition 2.8].

For a normed space X, let $\mathbb { B } _ { X }$ denote its closed unit ball; when subscripts become cumbersome, $\mathbb { B } ( X )$ will be used instead. The injective tensor product, denoted $X \otimes _ { \varepsilon } Y$ , is the algebraic tensor product equipped with the norm $\| \cdot \| _ { \varepsilon }$ defined by

$$
\left\| \sum _ { i = 1 } ^ { k } x _ { i } \otimes y _ { i } \right\| _ { \varepsilon } = \operatorname* { s u p } \left\{ \left| \sum _ { i = 1 } ^ { k } \phi ( x _ { i } ) \psi ( y _ { i } ) \right| : \phi \in \mathbb { B } _ { X ^ { * } } , \psi \in \mathbb { B } _ { Y ^ { * } } \right\} .
$$

For the injective tensor product, no optimization over representations of the tensor is required. The completion of $X \otimes _ { \varepsilon } Y$ is denoted $X { \widehat { \otimes } } _ { \varepsilon } Y$

For Banach spaces X and Y , there is a canonical linear mapping

$$
J : X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * } \ \longrightarrow \ ( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * }
$$

characterized on elementary tensors by

$$
\langle J ( x ^ { * } \otimes y ^ { * } ) , x \otimes y \rangle \triangleq x ^ { * } ( x ) y ^ { * } ( y ) , \qquad x \in X , \quad y \in Y , \quad x ^ { * } \in X ^ { * } , \quad y ^ { * } \in Y ^ { * } ,
$$

and extended by linearity and continuity.

One final functional-analytic concept is required before turning to the proof of Theorem 28. A Banach space X is said to have the approximation property if, for every norm-compact subset $K \subset X$ , the identity operator on K can be uniformly approximated by finite-rank operators; that is, for every $\epsilon > 0$ , there exists a finite-rank bounded linear operator $F : X  X$ such that

$$
\operatorname* { s u p } _ { x \in K } \| F ( x ) - x \| \leq \epsilon .
$$

The approximation property is a deep structural property of Banach spaces. It holds for many classical spaces, while the space of bounded linear operators on an infinite-dimensional Hilbert space provides a well-known counterexample [88].

A tensor $u \in X { \widehat { \otimes } } _ { \pi } Y$ is said to have rank at most k if there exist $x _ { 1 } , \ldots , x _ { k } \in X$ and $y _ { 1 } , \dotsc , y _ { k } \in Y$ such that

$$
u = \sum _ { i = 1 } ^ { k } x _ { i } \otimes y _ { i } .
$$

The following theorem generalizes Proposition 1.2 of [21], which establishes weak closedness of the set of rank-one tensors.

Theorem 29. Let X and Y be Banach spaces, let $k \in \mathbb N .$ , and suppose that at least one of $X ^ { \ast }$ or $Y ^ { * }$ satisfies the approximation property. Then, under the canonical mapping

$$
X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * } \longrightarrow ( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * } ,
$$

the set of tensors of rank at most k in $X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * }$ is weak<sup>∗</sup> closed.

The proof of Theorem 29 draws heavily on the methods developed in [21].

Proof of Theorem 29. Let

$$
\begin{array} { r } { Q _ { \leq k } = \left\{ u \in X ^ { * } \widehat { \otimes } _ { \pi } Y ^ { * } : u \mathrm { ~ h a s ~ r a n k ~ a t ~ m o s t ~ } k \right\} , } \end{array}
$$

and let

$$
J : X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * } \longrightarrow ( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * }
$$

denote the canonical mapping. By [87, Corollary 4.8(b)], the canonical mapping

$$
\Theta : X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * } \longrightarrow { \mathcal { B } } _ { N } ( X \times Y ) , \qquad \Theta ( u ) ( x , y ) = \langle J ( u ) , x \otimes y \rangle ,
$$

is an isometric isomorphism under the approximation property assumption, where $B _ { N } ( X \times Y )$ denotes the space of nuclear bilinear forms on $X \times Y$ , equipped with the nuclear norm [87, Section 2.6, pp. 39–40]. To show that J is injective, let $u \in X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * }$ satisfy $J ( u ) = 0$ . Then, for every $x \in X$ and $y \in Y$

$$
\Theta ( u ) ( x , y ) = \langle J ( u ) , x \otimes y \rangle = 0 .
$$

Thus $\Theta ( u )$ is the zero bilinear form. Since Θ is an isometric isomorphism, it is injective, and therefore $u = 0$ Hence ker $J = \left\{ 0 \right\}$ , so the linear map J is injective.

For $H \in ( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * }$ , define the bounded linear operator

$$
T _ { H } : X \longrightarrow Y ^ { * } , \qquad T _ { H } ( x ) ( y ) \triangleq \langle H , x \otimes y \rangle .
$$

The operator $T _ { H }$ is clearly linear. To see that it is bounded, observe that

$$
| T _ { H } ( x ) ( y ) | \leq \| H \| \| x \otimes y \| _ { \varepsilon } = \| H \| \| x \| \| y \| .
$$

It will first be shown that

$$
H \in J ( Q _ { \leq k } ) \quad \Longleftrightarrow \quad \dim T _ { H } ( X ) \leq k .\tag{33}
$$

If

$$
H = J \left( \sum _ { i = 1 } ^ { r } x _ { i } ^ { * } \otimes y _ { i } ^ { * } \right) , \qquad r \leq k ,
$$

then

$$
T _ { H } ( x ) = \sum _ { i = 1 } ^ { r } x _ { i } ^ { * } ( x ) y _ { i } ^ { * } ,
$$

and hence $T _ { H } ( X ) \subseteq \operatorname { s p a n } \{ y _ { 1 } ^ { * } , \dots , y _ { r } ^ { * } \}$ . Thus dim $T _ { H } ( X ) \leq k$

Conversely, suppose that dim $T _ { H } ( X ) = r \leq k . { \mathrm { ~ I f ~ } } r = 0$ , then $T _ { H } = 0$ , and hence

$$
\langle H , x \otimes y \rangle = T _ { H } ( x ) ( y ) = 0
$$

for every $x \in X$ and $y \in Y$ . By linearity, H therefore vanishes on $X \otimes Y .$ , the span of the elementary tensors. Since $X \otimes Y$ is dense in $X { \widehat { \otimes } } _ { \varepsilon } Y$ and H is continuous, it follows that $H = 0$ . Thus

$$
H = J ( 0 ) \in J ( Q _ { \leq k } ) ,
$$

which proves the implication when $r = 0$

Assume now that $r > 0$ . Let

$$
F \triangleq T _ { H } ( X ) ,
$$

and choose a basis $y _ { 1 } ^ { * } , \ldots , y _ { r } ^ { * }$ for F. Let $\lambda _ { 1 } , \ldots , \lambda _ { r } \in F ^ { * }$ be the corresponding coordinate functionals, characterized by

$$
\lambda _ { i } ( y _ { i } ^ { * } ) = 1 , \qquad \lambda _ { i } ( y _ { j } ^ { * } ) = 0 \quad \mathrm { w h e n } \ i \ne j .
$$

Equivalently, the identity operator on F satisfies

$$
I _ { F } ( z ) = \sum _ { i = 1 } ^ { r } \lambda _ { i } ( z ) y _ { i } ^ { * } , \qquad z \in F .
$$

Since $F$ is finite dimensional, each $\lambda _ { i }$ is bounded. Define

$$
x _ { i } ^ { * } \triangleq \lambda _ { i } \circ T _ { H } \in X ^ { * } , \quad \quad i \in [ r ] .
$$

For every $x \in X$ , the preceding representation of $I _ { F }$ gives

$$
T _ { H } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { r } \lambda _ { i } \bigl ( T _ { H } ( \boldsymbol { x } ) \bigr ) y _ { i } ^ { * } = \sum _ { i = 1 } ^ { r } x _ { i } ^ { * } ( \boldsymbol { x } ) y _ { i } ^ { * } .
$$

Consequently, for every $x \in X$ and $y \in Y$

$$
\begin{array} { r l } & { \left. J \left( \displaystyle \sum _ { i = 1 } ^ { r } x _ { i } ^ { * } \otimes y _ { i } ^ { * } \right) , x \otimes y \right. = \displaystyle \sum _ { i = 1 } ^ { r } x _ { i } ^ { * } ( x ) y _ { i } ^ { * } ( y ) } \\ & { \qquad = T _ { H } ( x ) ( y ) } \\ & { \qquad = \langle H , x \otimes y \rangle . } \end{array}
$$

Since $X \otimes Y$ is dense in $X { \widehat { \otimes } } _ { \varepsilon } Y$ , it follows that

$$
H = J \left( \sum _ { i = 1 } ^ { r } x _ { i } ^ { * } \otimes y _ { i } ^ { * } \right) \in J ( Q _ { \leq k } ) ,
$$

proving (33).

For

$$
S = \left( ( x _ { 1 } , \dots , x _ { k + 1 } ) , ( y _ { 1 } , \dots , y _ { k + 1 } ) \right) \in X ^ { k + 1 } \times Y ^ { k + 1 } ,
$$

define

$$
D _ { S } : ( X \widehat { \otimes } _ { \varepsilon } Y ) ^ { * } \longrightarrow \mathbb { R }
$$

by

$$
D _ { S } ( H ) \triangleq \operatorname* { d e t } [ \langle H , x _ { i } \otimes y _ { j } \rangle ] _ { i \in [ k + 1 ] , j \in [ k + 1 ] } .
$$

Each $D _ { S }$ is weak<sup>∗</sup>-continuous, since each mapping

$$
H \longmapsto \langle H , x _ { i } \otimes y _ { j } \rangle
$$

is weak<sup>∗</sup>-continuous and the determinant is a polynomial in these finitely many values. It will now be shown that $H \in J ( Q _ { < k } )$ if and only if $D _ { S } ( H ) = 0$ for every $S \in X ^ { k + 1 } \times Y ^ { k + 1 }$

Suppose that dim $T _ { H } ( X ) \leq k$ . By (33), there exist $r \leq k , x _ { 1 } ^ { * } , \ldots , x _ { r } ^ { * } \in X ^ { * }$ , and $y _ { 1 } ^ { * } , \ldots , y _ { r } ^ { * } \in Y ^ { * }$ such that

$$
H = J \left( \sum _ { \ell = 1 } ^ { r } x _ { \ell } ^ { * } \otimes y _ { \ell } ^ { * } \right) .
$$

Consequently, for every $S \in X ^ { k + 1 } \times Y ^ { k + 1 }$

$$
\begin{array} { l } { \displaystyle \bigl [ \langle H , x _ { i } \otimes y _ { j } \rangle \bigr ] _ { i , j \in [ k + 1 ] } = \left[ \sum _ { \ell = 1 } ^ { r } x _ { \ell } ^ { * } ( x _ { i } ) y _ { \ell } ^ { * } ( y _ { j } ) \right] _ { i , j \in [ k + 1 ] } } \\ { = \displaystyle \sum _ { \ell = 1 } ^ { r } \left[ \begin{array} { c } { x _ { \ell } ^ { * } ( x _ { 1 } ) } \\ { x _ { \ell } ^ { * } ( x _ { 2 } ) } \\ { \vdots } \\ { x _ { \ell } ^ { * } ( x _ { k + 1 } ) } \end{array} \right] \left[ y _ { \ell } ^ { * } ( y _ { 1 } ) \quad y _ { \ell } ^ { * } ( y _ { 2 } ) \quad \cdots \quad y _ { \ell } ^ { * } ( y _ { k + 1 } ) \right] . } \end{array}
$$

Each summand has rank at most one, so the matrix has rank at most $r \leq k$ . Its determinant is therefore zero, and hence $D _ { S } ( H ) = 0$ for every S.

Conversely, suppose that dim $T _ { H } ( X ) > k$ . There then exist $x _ { 1 } , \ldots , x _ { k + 1 } \in X$ such that

$$
T _ { H } ( x _ { 1 } ) , \dots , T _ { H } ( x _ { k + 1 } )
$$

are linearly independent elements of $Y ^ { * }$ . The linear map

$$
R : Y \longrightarrow \mathbb { R } ^ { k + 1 } , \qquad R ( y ) = { \big ( } T _ { H } ( x _ { 1 } ) ( y ) , \dots , T _ { H } ( x _ { k + 1 } ) ( y ) { \big ) } ,
$$

is surjective. Otherwise, a nonzero linear functional annihilating its range would give a nontrivial linear relation among $T _ { H } ( x _ { 1 } ) , \dots , T _ { H } ( x _ { k + 1 } )$ . Let $e _ { 1 } , \ldots , e _ { k + 1 }$ denote the standard basis of $\mathbb { R } ^ { k + 1 }$ . Since R is surjective, for every $j \in [ k + 1 ]$ there exists $y _ { j } \in Y$ such that

$$
\begin{array} { r } { R ( y _ { j } ) = e _ { j } . } \end{array}
$$

It follows that

$$
[ \langle H , x _ { i } \otimes y _ { j } \rangle ] _ { i , j \in [ k + 1 ] } = [ T _ { H } ( x _ { i } ) ( y _ { j } ) ] _ { i , j \in [ k + 1 ] } = I _ { k + 1 } .
$$

The determinant of this matrix is therefore nonzero.

Combining this characterization with (33) gives

$$
J ( Q _ { \leq k } ) = \bigcap _ { S \in X ^ { k + 1 } \times Y ^ { k + 1 } } D _ { S } ^ { - 1 } ( \{ 0 \} ) .
$$

The right-hand side is an intersection of weak<sup>∗</sup> closed subsets of $( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * }$ . Hence $J ( Q _ { \leq k } )$ is weak<sup>∗</sup> closed.

In the dual-space setting of Theorem 29, weak<sup>∗</sup> closedness also implies weak and norm closedness. The preceding argument also yields the following weak closedness result.

Corollary 30. Let X and Y be Banach spaces, at least one of which has the approximation property, and let $k \in \mathbb N$ . Then the set of tensors of rank at most k in $X { \widehat { \otimes } } _ { \pi } Y$ is weakly closed.

It is now possible to prove Theorem 28.

## Proof of Theorem 28. Set

$$
X \triangleq C ( K _ { 1 } ) , \qquad Y \triangleq C ( K _ { 2 } ) .
$$

Since $K _ { 1 }$ and $K _ { 2 }$ are compact metric spaces, the Riesz–Markov–Kakutani representation theorem gives the canonical linear isometric isomorphisms

$$
\begin{array} { r } { X ^ { * } \cong { \mathcal { M } } ( K _ { 1 } ) , \qquad Y ^ { * } \cong { \mathcal { M } } ( K _ { 2 } ) , \qquad { \mathcal { M } } ( K _ { 1 } \times K _ { 2 } ) \cong C ( K _ { 1 } \times K _ { 2 } ) ^ { * } . } \end{array}
$$

Let

$$
\Gamma : X { \widehat { \otimes } } _ { \varepsilon } Y \longrightarrow C ( K _ { 1 } \times K _ { 2 } )
$$

denote the canonical linear isometric isomorphism determined by

$$
\Gamma ( f \otimes g ) ( x , y ) = f ( x ) g ( y )
$$

[87, p. 50]. Under the preceding identification of $\mathcal { M } ( K _ { 1 } \times K _ { 2 } )$ with $C ( K _ { 1 } \times K _ { 2 } ) ^ { * }$ , define

$$
\Phi \triangleq \Gamma ^ { * } : \mathcal { M } ( K _ { 1 } \times K _ { 2 } ) \longrightarrow ( X \widehat { \otimes } _ { \varepsilon } Y ) ^ { * } .
$$

Since Γ is a linear isometric isomorphism, so is its dual operator Φ [20, Exercise 2.41(ii)]. By the definition of the dual operator,

$$
\langle \Phi ( \mu ) , h \rangle = \langle \mu , \Gamma ( h ) \rangle
$$

for every $\mu \in \mathcal { M } ( K _ { 1 } \times K _ { 2 } )$ and $h \in X { \widehat { \otimes } } _ { \varepsilon } Y$

Let

$$
J : X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * } \longrightarrow ( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * }
$$

denote the canonical linear mapping. For $\nu _ { 1 } \in \mathcal { M } ( K _ { 1 } ) , \nu _ { 2 } \in \mathcal { M } ( K _ { 2 } ) , f \in C ( K _ { 1 } )$ , and $g \in C ( K _ { 2 } )$ ,

$$
\begin{array} { r l } & { \langle \Phi ( \nu _ { 1 } \times \nu _ { 2 } ) , f \otimes g \rangle = \langle \nu _ { 1 } \times \nu _ { 2 } , \Gamma ( f \otimes g ) \rangle } \\ & { \qquad = \displaystyle \int _ { K _ { 1 } \times K _ { 2 } } f ( x ) g ( y ) d ( \nu _ { 1 } \times \nu _ { 2 } ) ( x , y ) } \\ & { \qquad = \displaystyle \left( \int _ { K _ { 1 } } f d \nu _ { 1 } \right) \left( \int _ { K _ { 2 } } g d \nu _ { 2 } \right) } \\ & { \qquad = \langle J ( \nu _ { 1 } \otimes \nu _ { 2 } ) , f \otimes g \rangle . } \end{array}
$$

Thus, for each fixed pair $\nu _ { 1 } , \nu _ { 2 }$ , the functionals $\Phi ( \nu _ { 1 } \times \nu _ { 2 } )$ and $J ( \nu _ { 1 } \otimes \nu _ { 2 } )$ agree on every elementary tensor $f \otimes g$ . By linearity, they therefore agree on $X \otimes Y$ . Since $X \otimes Y$ is dense in $\widehat { X } \widehat { \otimes } _ { \varepsilon } Y$ , their continuity implies that

$$
\Phi ( \nu _ { 1 } \times \nu _ { 2 } ) = J ( \nu _ { 1 } \otimes \nu _ { 2 } ) \qquad { \mathrm { f o r ~ a l l ~ } } \nu _ { 1 } , \nu _ { 2 } .\tag{34}
$$

Let $Q _ { < k }$ denote the set of tensors of rank at most k in $X ^ { * } { \widehat { \otimes } } _ { \pi } Y ^ { * }$ . Equation (34) and the linearity of Φ and J show that

$$
\Phi \big ( \mathcal { R } _ { \le k } ( K _ { 1 } , K _ { 2 } ) \big ) = J ( Q _ { \le k } ) .\tag{35}
$$

Indeed, a representation

$$
\mu = \sum _ { i = 1 } ^ { r } \nu _ { i , 1 } \times \nu _ { i , 2 } , \qquad r \leq k ,
$$

is carried by Φ to

$$
J \left( \sum _ { i = 1 } ^ { r } \nu _ { i , 1 } \otimes \nu _ { i , 2 } \right) ,
$$

and every element of $J ( Q _ { < k } )$ arises in this way.

Since $X ^ { * } = C ( K _ { 1 } ) ^ { * }$ has the approximation property $[ 8 7 , \ \mathrm { p } . \ 8 0 ]$ , Theorem 29 implies that $J ( Q _ { \leq k } )$ is weak<sup>∗</sup> closed in $( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * }$

Let $( \mu _ { \alpha } ) _ { \alpha \in A }$ be a net in $\mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ converging weak<sup>∗</sup> in $\mathcal { M } ( K _ { 1 } \times K _ { 2 } )$ to some $\mu .$ . By (35),

$$
\Phi ( \mu _ { \alpha } ) \in J ( Q _ { \leq k } ) \qquad { \mathrm { f o r ~ e v e r y ~ } } \alpha .
$$

Moreover, for every $h \in X { \widehat { \otimes } } _ { \varepsilon } Y$

$$
\langle \Phi ( \mu _ { \alpha } ) , h \rangle = \langle \mu _ { \alpha } , \Gamma ( h ) \rangle \longrightarrow \langle \mu , \Gamma ( h ) \rangle = \langle \Phi ( \mu ) , h \rangle .
$$

Hence $\Phi ( \mu _ { \alpha } ) \longrightarrow \Phi ( \mu )$ weak<sup>∗</sup> in $( X { \widehat { \otimes } } _ { \varepsilon } Y ) ^ { * }$ . Since $J ( Q _ { \leq k } )$ is weak<sup>∗</sup> closed,

$$
\Phi ( \mu ) \in J ( Q _ { \leq k } ) .
$$

Equation (35) and the injectivity of Φ therefore imply that

$$
\mu \in \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } ) .
$$

Thus $\mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ is weak<sup>∗</sup> closed.

## 10.2 Proofs of Theorem 3 and Proposition 4

The proof of Theorem 3 uses the following two standard compactness results.

Lemma 31 ([89, Theorem 5.8]). Let $( \mathcal { X } , \tau _ { \mathcal { X } } )$ be a compact topological space and $( \boldsymbol { \mathrm { { y } } } , \tau _ { \mathcal { Y } } )$ a Hausdorftopological space. $I f f : \mathcal { X } \to \mathcal { Y }$ is continuous and injective, then $f ^ { - 1 } : f ( \mathcal { X } ) \to \mathcal { X }$ is continuous, and hence $f : \mathcal { X } $ $f ( \mathcal X )$ is a homeomorphism.

Theorem 32 (Banach–Alaoglu [20, Theorem 3.21]). If X is a Banach space, then the closed unit ball of $X ^ { * }$ is compact in the weak<sup>∗</sup> topology.

The proof of Theorem 3 also relies on the continuity of the silhouette operator.

Lemma 33. Let $K _ { 1 } , K _ { 2 } \subset \mathbb { R }$ be compact, let $k \in \mathbb { N } ,$ and let $V \in S _ { 2 k }$ . Consider the restriction

$$
V : { \mathcal { M } } ( K _ { 1 } \times K _ { 2 } ) \to { \mathcal { M } } ( { \mathbb { R } } ) ^ { 2 k } .
$$

Then V is norm-to-norm, weak-to-weak, and $w e a k ^ { * } - t o - w e a k ^ { * }$ continuous.

Proof of Lemma 33. It sufices to prove the statement for a single coordinate of $V .$ , since $V \in S _ { 2 k }$ is a finite tuple of linear pushforwards and $\mathcal { M } ( \mathbb { R } ) ^ { 2 k }$ is equipped with the corresponding product topology. Thus let

$$
\pi \# : \mathcal { M } ( K _ { 1 } \times K _ { 2 } ) \to \mathcal { M } ( \mathbb { R } )
$$

be one of the pushforward maps appearing in $V ,$ , where π : $K _ { 1 } \times K _ { 2 } \to$ R is the restriction of the corresponding linear map on $\mathbb { R } ^ { 2 }$

The pushforward map $\pi \#$ is linear, and for every $\mu \in \mathcal { M } ( K _ { 1 } \times K _ { 2 } )$

$$
\| \pi \# \mu \| _ { \mathrm { T V } } \leq \| \mu \| _ { \mathrm { T V } } .
$$

Therefore $\pi \#$ is a bounded linear operator between Banach spaces, and is hence norm-to-norm and weakto-weak continuous.

Let $( \mu _ { \alpha } ) _ { \alpha \in A }$ be a net in $\mathcal { M } ( K _ { 1 } \times K _ { 2 } )$ that converges weak<sup>∗</sup> to $\mu ,$ and let $f \in C _ { 0 } ( \mathbb { R } )$ , where $C _ { 0 } ( \mathbb { R } )$ denotes the space of continuous functions on R that vanish at infinity. To prove weak<sup>∗</sup>-to-weak<sup>∗</sup> continuity, it is enough to show that

$$
\langle \pi \# \mu _ { \alpha } , f \rangle \longrightarrow \langle \pi \# \mu , f \rangle .
$$

For each $\alpha ,$ Lemma 23 gives

$$
\langle \pi \# \mu _ { \alpha } , f \rangle = \langle \mu _ { \alpha } , f \circ \pi \rangle .
$$

Since $f \circ \pi \in C ( K _ { 1 } \times K _ { 2 } )$ and $\mu _ { \alpha }  \mu$ weak<sup>∗</sup>,

$$
\langle \pi \# \mu _ { \alpha } , f \rangle = \langle \mu _ { \alpha } , f \circ \pi \rangle \longrightarrow \langle \mu , f \circ \pi \rangle = \langle \pi \# \mu , f \rangle ,
$$

where the final equality again follows from Lemma 23. Hence $\pi \# \mu _ { \alpha }$ converges weak<sup>∗</sup> to $\pi \# \mu ,$ so $\pi \#$ is weak<sup>∗</sup>-to-weak<sup>∗</sup> continuous. □

Theorem 3 now follows by combining the compactness results above with the continuity of the silhouette operator and its injectivity on $\mathcal { R } { _ { \leq k } } ( K _ { 1 } , K _ { 2 } )$ .

## Proof of Theorem 3. Let

$$
A = \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } ) \cap r \mathbb { B } _ { \mathcal { M } ( K _ { 1 } \times K _ { 2 } ) } .
$$

By Theorem 28, $\mathcal { R } { \ } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ is weak<sup>∗</sup> closed in $\mathcal { M } ( K _ { 1 } \times K _ { 2 } )$ . Moreover, by Theorem 32, $r \mathbb { B } _ { \mathcal { M } ( K _ { 1 } \times K _ { 2 } ) }$ is weak<sup>∗</sup> compact. Hence A is weak<sup>∗</sup> compact. By Theorem 1, the silhouette operator V is injective on $\mathcal { R } { < } _ { k } ( K _ { 1 } , K _ { 2 } )$ , and hence on A. By Lemma 33, its restriction to A is weak<sup>∗</sup>-to-weak<sup>∗</sup> continuous. Since $\bar { \mathcal { M } } ( \mathbb { R } ) ^ { 2 k }$ is Hausdorf in the weak<sup>∗</sup> topology, its subspace $V ( A )$ is also Hausdorf. Lemma 31 therefore applies to the continuous injection

$$
V | _ { A } : A \to V ( A ) ,
$$

and shows that this restriction is a weak<sup>∗</sup>-weak<sup>∗</sup> homeomorphism.

## Proof of Proposition 4. If

$$
S \cap \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } ) = \{ 0 \} ,
$$

then the conclusion is immediate, for example by taking $\gamma = 1$ and $\delta = 0$ . Assume henceforth that this intersection contains a nonzero measure. It will first be shown that there exists $\kappa > 0$ such that

$$
\| V ( \mu ) \| \ge \kappa \| \mu \| \qquad \mathrm { f o r ~ a l l ~ } \mu \in S \cap { \mathcal { R } } _ { \le 2 k } ( K _ { 1 } , K _ { 2 } ) .
$$

Assume for the sake of contradiction that no such κ exists. Then there exists a sequence $( \mu _ { i } ) _ { i = 1 } ^ { \infty }$ of nonzero elements of $S \cap \mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } )$ such that

$$
\frac { \| V ( \mu _ { i } ) \| } { \| \mu _ { i } \| } \longrightarrow 0 .
$$

By normalization, assume that $\| \mu _ { i } \| = 1$ for all i. Let S denote the unit sphere in $\mathcal { M } ( K _ { 1 } \times K _ { 2 } )$ . Then $( \mu _ { i } ) _ { i = 1 } ^ { \infty }$ lies in

$$
S \cap \mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } ) \cap \mathbb { S } .
$$

Since S is finite dimensional, S ∩ S is norm compact. Since $\mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } )$ is norm closed by Theorem 28, the set $S \cap \mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } )$ ∩ S is norm compact. Hence, after passing to a subsequence, $\mu _ { i } \to \mu$ in norm for some nonzero $\mu \in \mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } )$ . By continuity of $V , V ( \mu ) = 0$

Because $\mu \in \mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } ) , \mu = \nu - \nu ^ { \prime }$ for some distinct $\nu , \nu ^ { \prime } \in \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ . Then $V ( \nu ) = V ( \nu ^ { \prime } )$ 2 contradicting the injectivity of V on $\mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ given by Theorem 1. Therefore there exists $\kappa > 0$ such that

$$
\| V ( \mu ) \| ~ \geq ~ \kappa \| \mu \| ~ \mathrm { ~ f o r ~ a l l ~ } \mu \in S \cap \mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } ) .
$$

For $\mu , \mu ^ { \prime } \in S \cap \mathcal { R } _ { \leq k } ( K _ { 1 } , K _ { 2 } )$ , the diference $\mu - \mu ^ { \prime }$ lies in $S \cap \mathcal { R } _ { \leq 2 k } ( K _ { 1 } , K _ { 2 } )$ . Let $M = \| V | _ { S } \|$ denote the operator norm of the restriction of V to S. Since S is finite dimensional, $M < \infty ,$ and hence

$$
\kappa \| \mu - \mu ^ { \prime } \| \leq \| V ( \mu - \mu ^ { \prime } ) \| \leq M \| \mu - \mu ^ { \prime } \| ,
$$

and $0 < \kappa \leq M .$ . Define

$$
\gamma = \frac { 2 } { M + \kappa } , \qquad \delta = \frac { M - \kappa } { M + \kappa } .
$$

Hence $\delta \in [ 0 , 1 )$ , and

$$
\begin{array} { r l } { ( 1 - \delta ) \| \mu - \mu ^ { \prime } \| = \displaystyle \frac { 2 \kappa } { M + \kappa } \| \mu - \mu ^ { \prime } \| } & { } \\ { \displaystyle \leq \frac { 2 } { M + \kappa } \| V ( \mu - \mu ^ { \prime } ) \| } & { } \\ { \displaystyle = \| \gamma V ( \mu - \mu ^ { \prime } ) \| } & { } \\ { \displaystyle \leq \frac { 2 M } { M + \kappa } \| \mu - \mu ^ { \prime } \| } & { } \\ { \displaystyle = ( 1 + \delta ) \| \mu - \mu ^ { \prime } \| . } & { } \end{array}
$$

## 11 Proofs for Higher-Dimensional Extensions of the Silhouette Framework

Proof of Theorem 5. The proof is by induction on d, the number of coordinates in the product measures. To include $d = 1$ as the base case, the induction will prove the conclusion under the additional explicit hypothesis that all one-dimensional marginals agree:

$$
R _ { \{ \ell \} } \biggl ( \sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \dots , \mu _ { i , d } ) } \biggr ) = R _ { \{ \ell \} } \biggl ( \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \dots , \mu _ { i , d } ^ { \prime } ) } \biggr ) \qquad \mathrm { f o r ~ e v e r y ~ } \ell \in [ d ] .\tag{36}
$$

For $d \geq 2$ , this additional hypothesis follows from the assumptions of the theorem. Indeed, because G is connected, every $\ell \in [ d ]$ belongs to some edge $e \in E$ . Taking the ℓ-th marginal of the assumed equality corresponding to e gives (36). It therefore sufices to prove this auxiliary statement by induction, beginning with $d = 1$

For $d = 1$ , the hypothesis

$$
\{ \mu _ { 1 , 1 } , . . . , \mu _ { k , 1 } \} = \{ \mu _ { 1 , 1 } ^ { \prime } , . . . , \mu _ { k , 1 } ^ { \prime } \}
$$

together with (36) gives

$$
\begin{array} { r l r } {  { R _ { \{ 1 \} } ( \sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } ) } ) = R _ { \{ 1 \} } ( \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } ) } ) } } \\ & { \Rightarrow } & { \displaystyle \sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , 1 } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } . } \end{array}
$$

Relabeling the primed components if necessary, assume that $\mu _ { i , 1 } = \mu _ { i , 1 } ^ { \prime }$ for all i. Since $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ are linearly independent, the preceding equality implies $w _ { i } = w _ { i } ^ { \prime }$ for all i. This proves the base case.

Assume that the result has been proved for all numbers of coordinates up to d. Consider the case of $d + 1$ coordinates. Without loss of generality, assume that $G$ is a spanning tree. It is enough to prove the result under this weaker assumption, since equality of the marginals on all edges of G implies equality on the edges of any spanning tree of $G .$ For notational ease, relabel the coordinates so that $d + 1$ is a leaf distinct from $d ^ { \prime }$ and d is its unique neighbor. This relabeling is possible because a spanning tree has at least two leaves, so even if the original coordinate $d ^ { \prime }$ is a leaf, another leaf can be chosen as $d + 1$ . Since removing a leaf from a tree leaves a connected graph on the remaining vertices, the inductive hypothesis applies to the coordinates [d]. Hence

$$
\sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \dots , \mu _ { i , d } ) } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \dots , \mu _ { i , d } ^ { \prime } ) } .
$$

The d-tuples on the left are pairwise distinct because $\mu _ { 1 , d } , \ldots , \mu _ { k , d }$ are linearly independent, and each has positive weight. Since the right-hand side contains k terms with nonnegative weights, the preceding equality

implies that, after relabeling the primed components,

$$
( \mu _ { i , 1 } , \ldots , \mu _ { i , d } ) = ( \mu _ { i , 1 } ^ { \prime } , \ldots , \mu _ { i , d } ^ { \prime } ) \qquad \mathrm { a n d } \qquad w _ { i } = w _ { i } ^ { \prime }
$$

for every $i \in [ k ]$ . Since the pairwise marginals agree on the edge $\{ d , d + 1 \}$ ,

$$
\sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , d } \times \mu _ { i , d + 1 } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \mu _ { i , d } ^ { \prime } \times \mu _ { i , d + 1 } ^ { \prime } .
$$

Using the relabeling above, this becomes

$$
\sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , d } \times \mu _ { i , d + 1 } = \sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , d } \times \mu _ { i , d + 1 } ^ { \prime } .
$$

By Lemma 14, choose bounded measurable functions $f _ { 1 } , \ldots , f _ { k }$ such that

$$
\mu _ { i , d } ( f _ { j } ) = \mathbb { 1 } ( i = j ) , \qquad i , j \in [ k ] .
$$

For each $j \in [ k ]$ , applying the operator $T _ { f _ { j } }$ , defined in (20), to both sides gives

$$
w _ { j } \mu _ { j , d + 1 } = w _ { j } \mu _ { j , d + 1 } ^ { \prime } .
$$

Since $w _ { j } > 0$ , it follows that $\mu _ { j , d + 1 } = \mu _ { j , d + 1 } ^ { \prime }$ for every $j \in [ k ]$ . Hence, after the same relabeling,

$$
( \mu _ { i , 1 } , \ldots , \mu _ { i , d + 1 } ) = ( \mu _ { i , 1 } ^ { \prime } , \ldots , \mu _ { i , d + 1 } ^ { \prime } ) \qquad \mathrm { a n d } \qquad w _ { i } = w _ { i } ^ { \prime }
$$

for all $i ,$ completing the induction.

Proof of Proposition 6. The hypothesis gives

$$
R \biggl ( \sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \mu _ { i , 2 } ) } \biggr ) = R \biggl ( \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \mu _ { i , 2 } ^ { \prime } ) } \biggr ) ,
$$

that is,

$$
\sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , 1 } \times \mu _ { i , 2 } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } .\tag{37}
$$

Since joint irreducibility implies linear independence, the families $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are linearly independent. Thus the left-hand side of (37) has rank k (Proposition 13). Hence the right-hand side also has rank k, which implies that $w _ { i } ^ { \prime } > 0$ for all i, and that the families $\mu _ { 1 , 1 } ^ { \prime } , \ldots , \mu _ { k , 1 } ^ { \prime }$ and $\mu _ { 1 , 2 } ^ { \prime } , \ldots , \mu _ { k , 2 } ^ { \prime }$ are linearly independent.

By Lemma 14, there exist bounded measurable functions $f _ { j , \ell } , f _ { j , \ell } ^ { \prime } \in L ^ { \infty } ( \Omega ) , j \in [ k ] , \ell \in [ 2 ]$ , such that

$$
\int f _ { j , \ell } d \mu _ { i , \ell } = \mathbb { 1 } ( i = j ) , \qquad \int f _ { j , \ell } ^ { \prime } d \mu _ { i , \ell } ^ { \prime } = \mathbb { 1 } ( i = j ) ,
$$

for all $i , j \in [ k ]$ and $\ell \in [ 2 ]$ [2].

Applying the operator $T _ { f _ { j , 1 } ^ { \prime } }$ , defined in (20), to both sides of (37) yields

$$
\sum _ { i = 1 } ^ { k } w _ { i } \left. f _ { j , 1 } ^ { \prime } , \mu _ { i , 1 } \right. \mu _ { i , 2 } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \left. f _ { j , 1 } ^ { \prime } , \mu _ { i , 1 } ^ { \prime } \right. \mu _ { i , 2 } ^ { \prime } = w _ { j } ^ { \prime } \mu _ { j , 2 } ^ { \prime } .
$$

Hence, for each $j \in [ k ]$

$$
\mu _ { j , 2 } ^ { \prime } = \sum _ { i = 1 } ^ { k } \beta _ { j , i } \mu _ { i , 2 } , \qquad \beta _ { j , i } \triangleq \frac { w _ { i } } { w _ { j } ^ { \prime } } \left. f _ { j , 1 } ^ { \prime } , \mu _ { i , 1 } \right. .
$$

The same argument applied to the second coordinate gives

$$
\mu _ { j , 1 } ^ { \prime } = \sum _ { i = 1 } ^ { k } \alpha _ { j , i } \mu _ { i , 1 } , \qquad \alpha _ { j , i } \triangleq \frac { w _ { i } } { w _ { j } ^ { \prime } } \left. f _ { j , 2 } ^ { \prime } , \mu _ { i , 2 } \right. .
$$

Since $\mu _ { j , 1 } ^ { \prime }$ and $\mu _ { j , 2 } ^ { \prime }$ are probability measures and $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are jointly irreducible, all coeficients $\alpha _ { j , i }$ and $\beta _ { j , i }$ are nonnegative.

Suppose, toward a contradiction, that for some fixed $j$ there exist distinct indices $i _ { 1 } \neq i _ { 2 }$ such that

$$
\alpha _ { j , i _ { 1 } } > 0 \qquad \mathrm { a n d } \qquad \beta _ { j , i _ { 2 } } > 0 .
$$

Consider the bounded linear functional on $\mathcal { M } ( \Omega \times \Omega )$ defined by

$$
\nu \mapsto \left. T _ { f _ { i _ { 1 } , 1 } } ( \nu ) , f _ { i _ { 2 } , 2 } \right. = \int f _ { i _ { 1 } , 1 } ( x ) f _ { i _ { 2 } , 2 } ( y ) d \nu ( x , y ) .
$$

Applying this functional to both sides of (37) yields

$$
\sum _ { i = 1 } ^ { k } w _ { i } \left. f _ { i _ { 1 } , 1 } , \mu _ { i , 1 } \right. \left. f _ { i _ { 2 } , 2 } , \mu _ { i , 2 } \right. = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \left. f _ { i _ { 1 } , 1 } , \mu _ { i , 1 } ^ { \prime } \right. \left. f _ { i _ { 2 } , 2 } , \mu _ { i , 2 } ^ { \prime } \right. .\tag{38}
$$

By construction of the biorthogonal system, the left-hand side of (38) is zero, since $i _ { 1 } \neq i _ { 2 }$ . On the other hand, consider the summand with index $i = j$ on the right-hand side of (38). Using the representations $\begin{array} { r } { \mu _ { j , 1 } ^ { \prime } = \sum _ { i } \alpha _ { j , i } \mu _ { i , 1 } } \end{array}$ and $\begin{array} { r } { \mu _ { j , 2 } ^ { \prime } = \sum _ { i } \beta _ { j , i } \mu _ { i , 2 } } \end{array}$ , this summand is

$$
w _ { j } ^ { \prime } \left. f _ { i _ { 1 } , 1 } , \mu _ { j , 1 } ^ { \prime } \right. \left. f _ { i _ { 2 } , 2 } , \mu _ { j , 2 } ^ { \prime } \right. = w _ { j } ^ { \prime } \alpha _ { j , i _ { 1 } } \beta _ { j , i _ { 2 } } > 0 .
$$

More generally, for every $i \in [ k ]$ , the i-th summand on the right-hand side of (38) is

$$
w _ { i } ^ { \prime } \alpha _ { i , i _ { 1 } } \beta _ { i , i _ { 2 } } \geq 0 .
$$

Thus the right-hand side is strictly positive, while the left-hand side is zero, a contradiction.

Taking total masses in the representations of $\mu _ { j , 1 } ^ { \prime }$ and $\mu _ { j , 2 } ^ { \prime }$ gives

$$
\sum _ { i = 1 } ^ { k } \alpha _ { j , i } = \sum _ { i = 1 } ^ { k } \beta _ { j , i } = 1 .
$$

It follows that, for each $j \in [ k ]$ , there exists an index $\sigma ( j ) \in [ k ]$ such that

$$
\alpha _ { j , \sigma \left( j \right) } = \beta _ { j , \sigma \left( j \right) } = 1 ,
$$

and all other coeficients in the j-th rows of $\left( \alpha _ { j , i } \right)$ and $( \beta _ { j , i } )$ vanish. Hence

$$
\mu _ { j , 1 } ^ { \prime } = \mu _ { \sigma ( j ) , 1 } , \qquad \mu _ { j , 2 } ^ { \prime } = \mu _ { \sigma ( j ) , 2 } .
$$

Since the left-hand side of (37) has rank k, the map σ must be surjective, and hence is a permutation of [k]. Since the rank-one terms $\mu _ { i , 1 } \times \mu _ { i , 2 } , i \in [ k ]$ , are linearly independent, it follows that

$$
w _ { j } ^ { \prime } = w _ { \sigma ( j ) } \qquad \mathrm { f o r ~ a l l ~ } j \in [ k ] .
$$

Since the terms on the two sides of (37) difer only by the reordering induced by $\sigma ,$ it follows that

$$
\sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \mu _ { i , 2 } ) } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \mu _ { i , 2 } ^ { \prime } ) } .
$$

Proof of Theorem 7. By the theorem hypothesis for the edge $\{ 1 , 2 \} \in E$

$$
\begin{array} { l } { { \displaystyle { \cal R } _ { \{ 1 , 2 \} } \left( \displaystyle { \sum _ { i = 1 } ^ { k } w _ { i } \delta _ { ( \mu _ { i , 1 } , \dots , \mu _ { i , d } ) } } \right) = { \cal R } _ { \{ 1 , 2 \} } \left( \displaystyle { \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \dots , \mu _ { i , d } ^ { \prime } ) } } \right) } } \\ { { \displaystyle \Rightarrow \qquad \sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , 1 } \times \mu _ { i , 2 } = \displaystyle { \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } } . } } \end{array}\tag{39}
$$

Since $w _ { i } > 0$ for every $i ,$ and since the families $\mu _ { 1 , 1 } , \ldots , \mu _ { k , 1 }$ and $\mu _ { 1 , 2 } , \ldots , \mu _ { k , 2 }$ are linearly independent, Proposition 13 implies that the left-hand side of (39) has rank k. Hence the right-hand side also has rank k. In particular, $w _ { i } ^ { \prime } > 0$ for every i, since otherwise it would be a sum of at most $k - 1$ rank-one measures. Proposition 13 then implies that the families $\mu _ { 1 , 1 } ^ { \prime } , \ldots , \mu _ { k , 1 } ^ { \prime }$ and $\mu _ { 1 , 2 } ^ { \prime } , \ldots , \mu _ { k , 2 } ^ { \prime }$ are linearly independent.

Applying the same argument to the equality associated with each edge $\{ a , b \} \in E$ shows that the primed families corresponding to both endpoints a and b are linearly independent. Since every vertex of $G$ belongs to at least one edge, it follows that $\mu _ { 1 , j } ^ { \prime } , \ldots , \mu _ { k , j } ^ { \prime }$ are linearly independent for every $j \in [ d ]$

Applying the same argument used in the proof of Proposition 6, for each $j \in [ k ]$ there exist real coeficients $\alpha _ { j , \ l }$ <sub>i</sub> and $\beta _ { j , \ l }$ <sub>i</sub> such that

$$
\mu _ { j , 1 } ^ { \prime } = \sum _ { i = 1 } ^ { k } \alpha _ { j , i } \mu _ { i , 1 } , \qquad \mu _ { j , 2 } ^ { \prime } = \sum _ { i = 1 } ^ { k } \beta _ { j , i } \mu _ { i , 2 } .
$$

These coeficients need not all be nonnegative, since joint irreducibility is not assumed. Nevertheless, because $\mu _ { j , 1 } ^ { \prime }$ and $\mu _ { j , 2 } ^ { \prime }$ are probability measures, the singularity assumptions imply that

$$
\alpha _ { j , 1 } \geq 0 , \qquad \beta _ { j , 1 } \geq 0 , \qquad j \in [ k ] .
$$

A negative coeficient on the singular component would force the resulting measure to take a negative value. By the singularity assumption, there exist measurable sets $A _ { 1 } , A _ { 2 } \subset \Omega$ such that

$$
\mu _ { 1 , 1 } ( A _ { 1 } ) = 1 , \quad \mu _ { i , 1 } ( A _ { 1 } ) = 0 , \qquad \mu _ { 1 , 2 } ( A _ { 2 } ) = 1 , \quad \mu _ { i , 2 } ( A _ { 2 } ) = 0 ,
$$

for all $i = 2 , \ldots , k$ . Thus

$$
\mu _ { 1 , 1 } ( A _ { 1 } ^ { c } ) = 0 , \quad \mu _ { i , 1 } ( A _ { 1 } ^ { c } ) = 1 , \qquad \mu _ { 1 , 2 } ( A _ { 2 } ^ { c } ) = 0 , \quad \mu _ { i , 2 } ( A _ { 2 } ^ { c } ) = 1 ,
$$

for all $i = 2 , \ldots , k$

Fix $\ell \in [ 2 ]$ , and suppose that

$$
\nu \ \triangleq \ \sum _ { i = 1 } ^ { k } \gamma _ { i } \mu _ { i , \ell }
$$

is a probability measure.

Let

$$
\nu ^ { \perp } \triangleq \sum _ { i = 2 } ^ { k } \gamma _ { i } \mu _ { i , \ell } ,
$$

so $\nu ^ { \perp }$ denotes the part of ν that is singular with respect to $\mu _ { 1 , \ell }$ . Then

$$
\nu = \gamma _ { 1 } \mu _ { 1 , \ell } + \nu ^ { \perp } .
$$

Now let $B \subset A _ { \ell } ^ { c }$ be measurable. Then

$$
\nu ( B ) = \gamma _ { 1 } \mu _ { 1 , \ell } ( B ) + \sum _ { i = 2 } ^ { k } \gamma _ { i } \mu _ { i , \ell } ( B ) = \sum _ { i = 2 } ^ { k } \gamma _ { i } \mu _ { i , \ell } ( B ) ,
$$

since $\mu _ { 1 , \ell } ( B ) = 0$ . Since ν is a probability measure, $\nu ( B ) \geq 0$ for all such B. Since $\nu ^ { \perp }$ is carried by $A _ { \ell } ^ { c } ,$ , it follows that $\nu ^ { \perp }$ is a nonnegative measure.

If there exists an index $i _ { 0 } \geq 2$ with $\gamma _ { i _ { 0 } } \neq 0$ , then $\nu ^ { \perp } \neq 0$ by linear independence of $\mu _ { 2 , \ell } , \ldots , \mu _ { k , \ell }$ . Since $\nu ^ { \perp }$ is nonnegative, it follows that $\nu ^ { \bot } ( \Omega ) > 0$ . Therefore

$$
\nu ( A _ { \ell } ^ { c } ) = \sum _ { i = 1 } ^ { k } \gamma _ { i } \mu _ { i , \ell } ( A _ { \ell } ^ { c } ) = \sum _ { i = 2 } ^ { k } \gamma _ { i } > 0 .\tag{40}
$$

Thus, if $\gamma _ { i } \neq 0$ for any $i \geq 2$ , then ν assigns strictly positive mass to $A _ { \ell } ^ { c }$

With this established, it will now be shown that no primed component can mix the singular component in one coordinate with the remaining components in the other. To begin note that, for every $i \in [ k ]$

$$
( \mu _ { i , 1 } \times \mu _ { i , 2 } ) ( A _ { 1 } \times A _ { 2 } ^ { c } ) = 0 .
$$

For $i = 1$ , this follows from $\mu _ { 1 , 2 } ( A _ { 2 } ^ { c } ) = 0 ;$ ; for $i \geq 2$ , it follows from $\mu _ { i , 1 } ( A _ { 1 } ) = 0$ . Therefore, evaluating (39) on $A _ { 1 } \times A _ { 2 } ^ { c }$ gives

$$
\sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } ( \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } ) ( A _ { 1 } \times A _ { 2 } ^ { c } ) = \sum _ { i = 1 } ^ { k } w _ { i } ( \mu _ { i , 1 } \times \mu _ { i , 2 } ) ( A _ { 1 } \times A _ { 2 } ^ { c } ) = 0 .\tag{41}
$$

Fix $j \in [ k ]$ , and suppose toward a contradiction that

$$
\alpha _ { j , 1 } > 0 \quad \mathrm { a n d } \quad \beta _ { j , i _ { 0 } } \neq 0 \mathrm { f o r ~ s o m e } i _ { 0 } \geq 2 .
$$

Then

$$
\begin{array} { r l } { \displaystyle \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } ( \boldsymbol { \mu } _ { i , 1 } ^ { \prime } \times \boldsymbol { \mu } _ { i , 2 } ^ { \prime } ) ( A _ { 1 } \times A _ { 2 } ^ { c } ) \geq w _ { j } ^ { \prime } ( \boldsymbol { \mu } _ { j , 1 } ^ { \prime } \times \boldsymbol { \mu } _ { j , 2 } ^ { \prime } ) ( A _ { 1 } \times A _ { 2 } ^ { c } ) } \\ { = w _ { j } ^ { \prime } \left( \displaystyle \sum _ { i = 1 } ^ { k } \alpha _ { j , i } \mu _ { i , 1 } ( A _ { 1 } ) \right) \left( \displaystyle \sum _ { i = 1 } ^ { k } \beta _ { j , i } \mu _ { i , 2 } ( A _ { 2 } ^ { c } ) \right) } \\ { = w _ { j } ^ { \prime } \alpha _ { j , 1 } \displaystyle \sum _ { i = 2 } ^ { k } \beta _ { j , i } \mu _ { i , 2 } ( A _ { 2 } ^ { c } ) } \\ { = w _ { j } ^ { \prime } \alpha _ { j , 1 } \displaystyle \sum _ { i = 2 } ^ { k } \beta _ { j , i } . } \end{array}
$$

By (40), applied with $\ell = 2$ and $\gamma _ { i } = \beta _ { j , i }$ , the assumption $\beta _ { j , i _ { 0 } } \neq 0$ implies $\textstyle \sum _ { i = 2 } ^ { k } \beta _ { j , i } > 0$ . Since $w _ { j } ^ { \prime } > 0$ and $\alpha _ { j , 1 } > 0$ , the last line above is strictly positive, contradicting (41). Consequently,

$$
\alpha _ { j , 1 } > 0 \quad \Longrightarrow \quad \beta _ { j , i } = 0 \quad \mathrm { f o r ~ e v e r y ~ } i \ge 2 .
$$

An identical argument with the two coordinates interchanged gives

$$
\beta _ { j , 1 } > 0 \quad \Longrightarrow \quad \alpha _ { j , i } = 0 \quad \mathrm { f o r ~ e v e r y ~ } i \geq 2 .
$$

Taking total masses in the representations of $\mu _ { j , { \mathrm { i } } } ^ { \prime }$ and $\mu _ { j , 2 } ^ { \prime }$ gives

$$
\sum _ { i = 1 } ^ { k } \alpha _ { j , i } = \sum _ { i = 1 } ^ { k } \beta _ { j , i } = 1 .
$$

Therefore, if $\alpha _ { j , 1 } > 0$ , the first implication gives $\beta _ { j , 1 } = 1$ , and the second then gives $\alpha _ { j , 1 } = 1$ . If $\alpha _ { j , 1 } = 0$ then $\beta _ { j , 1 } = 0$ , since otherwise the second implication would force every $\alpha _ { j , i }$ to vanish. Thus, for each $j \in [ k ]$ either

$$
\alpha _ { j , 1 } = \beta _ { j , 1 } = 1
$$

and all remaining coeficients in the $j { \mathrm { - t h } }$ rows vanish, or

$$
\alpha _ { j , 1 } = \beta _ { j , 1 } = 0 .
$$

Let

$$
I \triangleq \{ j \in [ k ] : \alpha _ { j , 1 } = \beta _ { j , 1 } = 1 \} .
$$

By the dichotomy above, if $j \notin I ,$ then $\alpha _ { j , 1 } = \beta _ { j , 1 } = 0$ . The set I cannot be empty: otherwise all terms on the right-hand side of (39) would have first and second marginals lying in the spans of $\mu _ { 2 , 1 } , \ldots , \mu _ { k , 1 }$ <sub>1</sub> and $\mu _ { 2 , 2 } , \ldots , \mu _ { k , 2 }$ , respectively, so the right-hand side would have rank at most $k - 1$ , contradicting the rank-k property.

The set I also cannot contain more than one index. If $| I | \geq 2 ,$ , then the corresponding summands all have product component $\mu _ { 1 , 1 } \times \mu _ { 1 , 2 }$ and can be combined into a single rank-one term, leaving a representation with at most $k - 1$ rank-one terms. This again contradicts the rank-k property. Hence $I = \{ r \}$ for some $r \in \left\lceil k \right\rceil$

For this index $r ,$

$$
\mu _ { r , 1 } ^ { \prime } = \mu _ { 1 , 1 } , \qquad \mu _ { r , 2 } ^ { \prime } = \mu _ { 1 , 2 } .
$$

Evaluating (39) on $A _ { 1 } \times A _ { 2 }$ gives $w _ { r } ^ { \prime } = w _ { 1 }$ . Relabeling the primed components, assume $r = 1$ . Thus

$$
\mu _ { 1 , 1 } ^ { \prime } = \mu _ { 1 , 1 } , \qquad \mu _ { 1 , 2 } ^ { \prime } = \mu _ { 1 , 2 } , \qquad w _ { 1 } ^ { \prime } = w _ { 1 } .
$$

To complete the proof, it remains to show that $\mu _ { 1 , j } ^ { \prime } = \mu _ { 1 , j }$ for all $j > 2$ . Fix $j > 2$ . By hypothesis, either $\{ 1 , j \} \in E$ or $\{ 2 , j \} \in E$ . The two cases are identical, so assume $\{ 1 , j \} \in E$

The equality of the $\{ 1 , j \}$ -marginals gives

$$
\sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , 1 } \times \mu _ { i , j } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , j } ^ { \prime } .
$$

Let $B \subset \Omega$ be measurable. Evaluating both sides on $A _ { 1 } \times B$ gives

$$
w _ { 1 } \mu _ { 1 , j } ( B ) = w _ { 1 } ^ { \prime } \mu _ { 1 , j } ^ { \prime } ( B ) ,
$$

since $\mu _ { i , 1 } ( A _ { 1 } ) = 0$ for $i \geq 2 ,$ , and since $I = \{ 1 \}$ , as established above, implies $\mu _ { i , 1 } ^ { \prime } ( A _ { 1 } ) = 0$ for $i \geq 2$ . Since $w _ { 1 } ^ { \prime } = w _ { 1 } > 0$ , it follows that $\mu _ { 1 , j } ( B ) = \mu _ { 1 , j } ^ { \prime } ( B )$ . As B was arbitrary, $\mu _ { 1 , j } = \mu _ { 1 , j } ^ { \prime }$ . The case $\{ 2 , j \} \in E$ is identical, using $A _ { 2 }$ instead of $A _ { 1 }$ . Since $I = \{ 1 \}$ , no other primed component agrees with $( \mu _ { 1 , 1 } , \ldots , \mu _ { 1 , d } )$ in its first two coordinates. Therefore

$$
\sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \delta _ { ( \mu _ { i , 1 } ^ { \prime } , \dots , \mu _ { i , d } ^ { \prime } ) } \big ( \{ ( \mu _ { 1 , 1 } , \dots , \mu _ { 1 , d } ) \} \big ) = w _ { 1 } ^ { \prime } = w _ { 1 } ,
$$

completing the proof.

## 12 Proofs of the Sharpness Results

The proofs of Theorem 8 and Proposition 9 use the following elementary facts about convolution. The first states that pushforward under a linear map commutes with convolution.

Lemma 34. Let $\mu _ { 1 } , \mu _ { 2 } \in \mathcal { M } ( \mathbb { R } ^ { 2 } )$ , and let $T : \mathbb { R } ^ { 2 }  \mathbb { R }$ be linear. Then

$$
T \# ( \mu _ { 1 } * \mu _ { 2 } ) = ( T \# \mu _ { 1 } ) * ( T \# \mu _ { 2 } ) .
$$

Proof of Lemma 34. Let $f : \mathbb { R } $ R be bounded and measurable. Then

$$
\begin{array} { r l } {  { \int _ { \mathbb { R } } f d \big ( T \# ( \mu _ { 1 } \ast \mu _ { 2 } ) \big ) = \int _ { \mathbb { R } ^ { 2 } } f \big ( T ( x ) \big ) d ( \mu _ { 1 } \ast \mu _ { 2 } \big ) ( x ) } } \\ & { = \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } f \big ( T ( x _ { 1 } + x _ { 2 } ) \big ) d \mu _ { 1 } ( x _ { 1 } ) d \mu _ { 2 } ( x _ { 2 } ) } \\ & { = \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } f \big ( T ( x _ { 1 } ) + T ( x _ { 2 } ) \big ) d \mu _ { 1 } ( x _ { 1 } ) d \mu _ { 2 } ( x _ { 2 } ) } \\ & { = \int _ { \mathbb { R } ^ { 2 } } \bigg ( \int _ { \mathbb { R } } f \big ( g _ { 1 } + T ( x _ { 2 } ) \big ) d ( T \frac { \mu _ { 1 } } { \mu } \mu _ { 1 } ) ( y _ { 1 } ) \bigg ) d \mu _ { 2 } ( x _ { 2 } ) } \\ & { = \int _ { \mathbb { R } } \bigg ( \int _ { \mathbb { R } } f ( y _ { 1 } + y _ { 2 } ) d ( T \# \mu _ { 1 } ) ( y _ { 1 } ) \bigg ) d ( T \# \mu _ { 2 } ) ( y _ { 2 } ) } \\ & { = \int _ { \mathbb { R } } f d \big ( ( T \# \mu _ { 1 } ) \ast ( T \# \mu _ { 2 } ) \big ) . } \end{array}
$$

The first, fourth, and fifth equalities use Lemma 23; the second and final equalities use the convolution integration formula [83, Proposition 8.48]; and the third uses the linearity of $T .$ Hence

$$
T \# ( \mu _ { 1 } * \mu _ { 2 } ) = ( T \# \mu _ { 1 } ) * ( T \# \mu _ { 2 } ) .
$$

The second fact ensures that the atomic convolutions constructed later in this section are nonzero.

Lemma 35. Let $d , m \in \mathbb { N }$ , and for each $i \in [ m ]$ , let $x _ { i } \in \mathbb { R } ^ { d } \setminus \{ 0 \}$ and $c _ { i } \in \mathbb { R }$ . Then

$$
\left( \delta _ { 0 } + c _ { 1 } \delta _ { x _ { 1 } } \right) * \cdot \cdot \cdot * \left( \delta _ { 0 } + c _ { m } \delta _ { x _ { m } } \right) \neq 0 .
$$

The proof of Lemma 35 uses the Fourier transform. For a finite signed measure $\mu \in \mathcal { M } ( \mathbb { R } ^ { d } )$ , write

$$
\mathcal { F } [ \mu ] ( \boldsymbol { \xi } ) \triangleq \int _ { \mathbb { R } ^ { d } } e ^ { - i \langle \boldsymbol { \xi } , \boldsymbol { x } \rangle } d \mu ( \boldsymbol { x } ) , \qquad \boldsymbol { \xi } \in \mathbb { R } ^ { d } ,
$$

for its Fourier transform.

Proof of Lemma 35. Set

$$
\eta = \left( \delta _ { 0 } + c _ { 1 } \delta _ { x _ { 1 } } \right) * \cdot \cdot \cdot * \left( \delta _ { 0 } + c _ { m } \delta _ { x _ { m } } \right) .
$$

For each $i \in [ m ]$

$$
\mathcal { F } [ \delta _ { 0 } + c _ { i } \delta _ { x _ { i } } ] ( \xi ) = 1 + c _ { i } e ^ { - i \langle \xi , x _ { i } \rangle } .
$$

Since $x _ { i } \neq 0$ , the zero set of this function is either empty or a countable union of afine hyperplanes in $\mathbb { R } ^ { d }$ and therefore has Lebesgue measure zero. By the convolution theorem,

$$
\mathcal { F } [ \eta ] ( \boldsymbol { \xi } ) = \prod _ { i = 1 } ^ { m } \left( 1 + c _ { i } e ^ { - i \langle \boldsymbol { \xi } , \boldsymbol { x } _ { i } \rangle } \right) ,
$$

which is nonzero almost everywhere. The injectivity of the Fourier transform on finite signed measures therefore implies that $\eta \neq 0$ □

The third fact describes how convolution by a product measure acts on a sum of product measures.

Lemma 36. Let $k \in \mathbb N$ , and let $\mu _ { i , j } , \Phi _ { j } \in \mathcal { M } ( \mathbb { R } )$ for $i \in [ k ]$ and $j \in [ 2 ]$ . Then

$$
\left( \sum _ { i = 1 } ^ { k } \mu _ { i , 1 } \times \mu _ { i , 2 } \right) * ( \Phi _ { 1 } \times \Phi _ { 2 } ) = \sum _ { i = 1 } ^ { k } ( \mu _ { i , 1 } * \Phi _ { 1 } ) \times ( \mu _ { i , 2 } * \Phi _ { 2 } ) .
$$

Proof of Lemma 36. By linearity of convolution, it sufices to show that

$$
( \mu _ { 1 } \times \mu _ { 2 } ) * ( \Phi _ { 1 } \times \Phi _ { 2 } ) = ( \mu _ { 1 } * \Phi _ { 1 } ) \times ( \mu _ { 2 } * \Phi _ { 2 } ) .
$$

Let $f : \mathbb { R } ^ { 2 } $ R be bounded and measurable. The convolution integration formula and Fubini’s theorem give

$$
\begin{array} { r l } {  { \int _ { \mathbb { R } ^ { 2 } } f d \big ( ( \mu _ { 1 } \times \mu _ { 2 } ) \ast ( \Phi _ { 1 } \times \Phi _ { 2 } ) \big ) } } \\ & { = \int _ { \mathbb { R } ^ { 2 } } \int _ { \mathbb { R } ^ { 2 } } f ( x _ { 1 } + y _ { 1 } , x _ { 2 } + y _ { 2 } ) d \big ( \mu _ { 1 } \times \mu _ { 2 } \big ) \big ( x _ { 1 } , x _ { 2 } \big ) d \big ( \Phi _ { 1 } \times \Phi _ { 2 } \big ) \big ( y _ { 1 } , y _ { 2 } \big ) } \\ & { = \int _ { \mathbb { R } ^ { 2 } } f d \big ( ( \mu _ { 1 } \ast \Phi _ { 1 } ) \times ( \mu _ { 2 } \ast \Phi _ { 2 } ) \big ) . } \end{array}
$$

Hence

$$
( \mu _ { 1 } \times \mu _ { 2 } ) * ( \Phi _ { 1 } \times \Phi _ { 2 } ) = ( \mu _ { 1 } * \Phi _ { 1 } ) \times ( \mu _ { 2 } * \Phi _ { 2 } ) .
$$

Summing this identity over $i \in [ k ]$ proves the result.

The proof of Theorem 8 follows from the following more general nullspace construction.

Proposition 37. Let $m \in \mathbb { N }$ and let $v _ { 1 } , \ldots , v _ { m } \in \mathbb { R } ^ { 2 }$ . Then there exists a nonzero compactly supported $\eta \in \mathcal { M } ( \mathbb { R } ^ { 2 } )$ of rank at most $m + 1$ such that

$$
\pi _ { v _ { \ell } } \# \eta = 0 , \qquad \ell \in [ m ] .
$$

Moreover, η admits a decomposition

$$
\eta = \mu - \mu ^ { \prime } ,
$$

where $\mu , \mu ^ { \prime }$ are compactly supported positive measures of equal nonzero mass and admit representations

$$
\mu = \sum _ { i = 1 } ^ { r _ { + } } \mu _ { i , 1 } \times \mu _ { i , 2 } , \qquad \mu ^ { \prime } = \sum _ { i = 1 } ^ { r _ { - } } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } ,
$$

where

$$
r _ { + } \leq \left\lceil \frac { m + 1 } { 2 } \right\rceil , \qquad r _ { - } \leq \left\lfloor \frac { m + 1 } { 2 } \right\rfloor ,
$$

and all of the measures $\mu _ { i , j }$ and $\mu _ { i , j } ^ { \prime }$ are compactly supported and nonnegative.

Proof of Proposition 37. It sufices to prove the result when $v _ { i } \neq 0$ for every $i \in [ m ]$ . If $v \ne 0$ , any η satisfying $\pi _ { v } \# \eta = 0$ must also satisfy $\eta ( \mathbb { R } ^ { 2 } ) = 0$ , equivalently $\pi _ { 0 } \# \eta = 0$ . Replacing a zero $v _ { i }$ by an arbitrary nonzero vector therefore only strengthens the required conditions.

Write $\boldsymbol { v } _ { i } = [ a _ { i } , b _ { i } ] ^ { \intercal }$ . For each $i \in [ m ]$ , define

$$
x _ { i } \triangleq \left\{ { \begin{array} { l l } { [ 1 , - a _ { i } / b _ { i } ] ^ { \top } , } & { b _ { i } \neq 0 , } \\ { { [ 0 , 1 ] } ^ { \top } , } & { b _ { i } = 0 . } \end{array} } \right.
$$

Then $\langle v _ { i } , x _ { i } \rangle = 0$ for every $i \in [ m ]$ . For each $i \in [ m ]$ , define

$$
\Delta _ { i } \triangleq \delta _ { 0 } - \delta _ { x _ { i } } .
$$

It follows that

$$
\pi _ { \boldsymbol { v } _ { i } } \# \Delta _ { i } = \delta _ { 0 } - \delta _ { \langle \boldsymbol { v } _ { i } , \boldsymbol { x } _ { i } \rangle } = 0 .
$$

Let $\eta = \Delta _ { 1 } * \cdot \cdot \cdot * \Delta _ { m } .$

By repeated application of Lemma 34, for every $i \in [ m ]$

$$
\pi _ { v _ { i } } \# \eta = \left( \pi _ { v _ { i } } \# \Delta _ { 1 } \right) * \cdot \cdot \cdot * \left( \pi _ { v _ { i } } \# \Delta _ { m } \right) = 0 ,
$$

because $\pi _ { v _ { i } } \# \Delta _ { i } = 0$ . Moreover, Lemma 35 implies that $\eta \neq 0 .$

Because each $x _ { i }$ has first coordinate either 0 or 1, and because $\delta _ { x } * \delta _ { y } = \delta _ { x + y } .$ , every point mass appearing in the expansion of the convolution has first coordinate in $\{ 0 , \ldots , m \}$ . Hence η is supported on $\{ 0 , \ldots , m \} \times \mathbb { R }$ Consequently, for suitable compactly supported signed measures $\rho _ { 0 } , \ldots , \rho _ { m } \in \mathcal { M } ( \mathbb { R } )$

$$
\eta = \sum _ { j = 0 } ^ { m } \delta _ { j } \times \rho _ { j } ,
$$

and $\eta$ has rank at most $m + 1$ . This establishes the first assertion of the proposition.

To construct the measures $\mu$ and $\mu ^ { \prime }$ in the proposition, consider two cases. First, suppose that the col lection contains at most one coordinate-axis direction. After interchanging the two coordinates if necessary, assume without loss of generality that no $v _ { i }$ has the form $[ q , 0 ] ^ { \top }$

Write $x _ { i } = [ 1 , c _ { i } ] ^ { \top }$ , so that

$$
\Delta _ { i } = \delta _ { 0 } - \delta _ { x _ { i } } .
$$

For $\varepsilon = ( \varepsilon _ { 1 } , \ldots , \varepsilon _ { m } ) \in \{ 0 , 1 \} ^ { m }$ , let

$$
| \varepsilon | \triangleq \sum _ { i = 1 } ^ { m } \varepsilon _ { i } .
$$

By distributivity of convolution over addition, expanding $\Delta _ { 1 } * \cdot \cdot * \Delta _ { m }$ produces one term for each $\varepsilon \in \{ 0 , 1 \} ^ { m }$ In this indexing, $\varepsilon _ { i } = 0$ corresponds to choosing $\delta _ { 0 }$ from $\Delta _ { i } .$ , whereas $\varepsilon _ { i } = 1$ corresponds to choosing $- \delta _ { x _ { i } }$ Accordingly,

$$
\begin{array} { l } { { \eta = \displaystyle \sum _ { \varepsilon \in \{ 0 , 1 \} ^ { m } } ( - 1 ) ^ { | \varepsilon | } \left( \delta _ { \varepsilon _ { 1 } x _ { 1 } } * \cdot \cdot \cdot * \delta _ { \varepsilon _ { m } x _ { m } } \right) } } \\ { { \quad = \displaystyle \sum _ { \varepsilon \in \{ 0 , 1 \} ^ { m } } ( - 1 ) ^ { | \varepsilon | } \delta _ { \sum _ { i = 1 } ^ { m } \varepsilon _ { i } x _ { i } } . } } \end{array}
$$

Since $x _ { i } = [ 1 , c _ { i } ] ^ { \top }$

$$
\sum _ { i = 1 } ^ { m } \varepsilon _ { i } x _ { i } = \left[ | \varepsilon | , \sum _ { i = 1 } ^ { m } \varepsilon _ { i } c _ { i } \right] ^ { \top } .
$$

For $j = 0 , \ldots , m .$ , define

$$
\eta _ { j } \triangleq \sum _ { \varepsilon \in \{ 0 , 1 \} ^ { m } } \delta _ { \sum _ { i = 1 } ^ { m } \varepsilon _ { i } c _ { i } } .
$$

Each $\eta _ { j }$ is a compactly supported nonnegative measure on R, and grouping the preceding expansion according to |ε| gives

$$
\eta = \sum _ { j = 0 } ^ { m } ( - 1 ) ^ { j } \delta _ { j } \times \eta _ { j } .
$$

Define

$$
\mu \triangleq \sum _ { \stackrel { 0 \leq j \leq m } { j \mathrm { ~ e v e n } } } \delta _ { j } \times \eta _ { j } , \qquad \mu ^ { \prime } \triangleq \sum _ { \stackrel { 0 \leq j \leq m } { j \mathrm { ~ o d d } } } \delta _ { j } \times \eta _ { j } .
$$

Then $\mu$ and $\mu ^ { \prime }$ are compactly supported nonnegative measures, and

$$
\eta = \mu - \mu ^ { \prime } .
$$

There are

$$
\left\lceil { \frac { m + 1 } { 2 } } \right\rceil
$$

even integers and

$$
\left\lfloor { \frac { m + 1 } { 2 } } \right\rfloor
$$

odd integers in $\{ 0 , \ldots , m \}$ . Thus, the displayed expressions write $\mu$ and $\mu ^ { \prime }$ as sums of at most the required numbers of product measures with compactly supported nonnegative factors.

Finally, pushforward preserves total mass. Hence, for any $i \in [ m ]$

$$
0 = \left( \pi _ { v _ { i } } \# \eta \right) ( \mathbb { R } ) = \eta ( \mathbb { R } ^ { 2 } ) = \mu ( \mathbb { R } ^ { 2 } ) - \mu ^ { \prime } ( \mathbb { R } ^ { 2 } ) .
$$

Thus $\mu$ and $\mu ^ { \prime }$ have equal mass. Since they are nonnegative and $\eta = \mu - \mu ^ { \prime } \neq 0$ , their common mass is nonzero.

It remains to consider the case in which both coordinate-axis directions occur. Then $\boldsymbol { v } _ { i } = [ q , 0 ] ^ { \top }$ for some $i ,$ where $q \neq 0 ,$ and the corresponding vector is $x _ { i } = [ 0 , 1 ] ^ { \top }$ . Hence at most $m - 1$ of the vectors $x _ { 1 } , \ldots , x _ { m }$ have nonzero first coordinate. The preceding convolution argument therefore implies that $\eta$ is supported on $\{ 0 , \ldots , m - 1 \} \times \mathbb { R }$ . Consequently, there exist compactly supported signed measures $\rho _ { 0 } , \ldots , \rho _ { m - 1 } \in \mathcal { M } ( \mathbb { R } )$ such that

$$
\eta = \sum _ { j = 0 } ^ { m - 1 } \delta _ { j } \times \rho _ { j } .
$$

Define compactly supported signed measures $\zeta _ { 0 } , \ldots , \zeta _ { m }$ recursively by

$$
\zeta _ { 0 } = 0 , \qquad \zeta _ { j + 1 } = \zeta _ { j } + ( - 1 ) ^ { j + 1 } \rho _ { j } , \qquad j = 0 , \ldots , m - 1 .
$$

Let

$$
\lambda = \sum _ { r = 0 } ^ { m } | \zeta _ { r } | , \qquad \gamma _ { r } = \lambda + \zeta _ { r } , \qquad r = 0 , \ldots , m .
$$

Since $\lambda \geq | \zeta _ { r } |$ , each $\gamma _ { r }$ is a compactly supported nonnegative measure. Moreover, for $j = 0 , \ldots , m - 1$

$$
\begin{array} { r l } { \gamma _ { j } - \gamma _ { j + 1 } = ( \lambda + \zeta _ { j } ) - ( \lambda + \zeta _ { j + 1 } ) } & { { } } \\ { = \zeta _ { j } - \zeta _ { j + 1 } } & { { } } \\ { = ( - 1 ) ^ { j } \rho _ { j } , } \end{array}
$$

where the final equality follows from the recursion defining the $\zeta _ { j }$ . Therefore,

$$
\rho _ { j } = ( - 1 ) ^ { j } ( \gamma _ { j } - \gamma _ { j + 1 } ) .
$$

For $r = 0 , \ldots , m$ , define

$$
\alpha _ { r } = \left\{ \begin{array} { l l } { \delta _ { 0 } , } & { r = 0 , } \\ { \delta _ { r - 1 } + \delta _ { r } , } & { 1 \leq r \leq m - 1 , } \\ { \delta _ { m - 1 } , } & { r = m . } \end{array} \right.
$$

Substituting the preceding expression for $\rho _ { j }$ and reindexing the terms involving $\gamma _ { j + 1 }$ gives

$$
\begin{array} { l } { \displaystyle \eta = \sum _ { j = 0 } ^ { m - 1 } ( - 1 ) ^ { j } \delta _ { j } \times ( \gamma _ { j } - \gamma _ { j + 1 } ) } \\ { = \sum _ { j = 0 } ^ { m - 1 } ( - 1 ) ^ { j } \delta _ { j } \times \gamma _ { j } + \sum _ { j = 0 } ^ { m - 1 } ( - 1 ) ^ { j + 1 } \delta _ { j } \times \gamma _ { j + 1 } } \\ { = \displaystyle \sum _ { j = 0 } ^ { m - 1 } ( - 1 ) ^ { j } \delta _ { j } \times \gamma _ { j } + \sum _ { j = 1 } ^ { m } ( - 1 ) ^ { j } \delta _ { j - 1 } \times \gamma _ { j } } \\ { = \displaystyle \sum _ { j = 0 } ^ { m - 1 } ( - 1 ) ^ { j } \delta _ { j } \times \gamma _ { j } + \sum _ { r = 1 } ^ { m } ( - 1 ) ^ { r } \delta _ { r - 1 } \times \gamma _ { r } } \\ { = \delta _ { 0 } \times \gamma _ { 0 } + \displaystyle \sum _ { r = 1 } ^ { m - 1 } ( - 1 ) ^ { r } ( \delta _ { r - 1 } + \delta _ { r } ) \times \gamma _ { r } + ( - 1 ) ^ { m } \delta _ { m - 1 } \times \gamma _ { m } } \\ { = \displaystyle \sum _ { r = 0 } ^ { m } ( - 1 ) ^ { r } \alpha _ { r } \times \gamma _ { r } . } \end{array}
$$

Therefore, set

$$
\mu = \sum _ { \stackrel { 0 \leq r \leq m } { r \ \mathrm { e v e n } } } \alpha _ { r } \times \gamma _ { r } , \qquad \mu ^ { \prime } = \sum _ { \stackrel { 0 \leq r \leq m } { r \ \mathrm { o d d } } } \alpha _ { r } \times \gamma _ { r } .
$$

Then $\eta = \mu - \mu ^ { \prime } $ , with $\alpha _ { r }$ and $\gamma _ { r }$ compactly supported and nonnegative for all $r .$ The numbers of even and odd indices in $\{ 0 , \ldots , m \}$ are respectively $\lceil ( m + 1 ) / 2 \rceil$ and $\lfloor ( m + 1 ) / 2 \rfloor$ , giving the required numbers of product terms.

As in the first case, the identities $\pi _ { v _ { i } } \# \eta = 0$ imply that $\mu$ and $\mu ^ { \prime }$ have equal nonzero mass. This completes the proof. □

Proof of Theorem 8. Apply Proposition 37 with $m = 2 k - 1$ . Normalizing the two positive measures by their common mass, normalizing their factors, absorbing the factor masses into mixture weights, and adding zero-weight components if necessary gives nonnegative weights $w _ { i } , w _ { i } ^ { \prime }$ and compactly supported probability measures $\rho _ { i , j } , \rho _ { i , j } ^ { \prime }$ , for $i \in [ k ]$ and $j \in [ 2 ]$ , such that

$$
\sum _ { i = 1 } ^ { k } w _ { i } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } = 1 ,
$$

and the distinct probability measures

$$
\boldsymbol { \rho } = \sum _ { i = 1 } ^ { k } w _ { i } \rho _ { i , 1 } \times \rho _ { i , 2 } , \qquad \boldsymbol { \rho } ^ { \prime } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \rho _ { i , 1 } ^ { \prime } \times \rho _ { i , 2 } ^ { \prime }
$$

satisfy

$$
\pi _ { \boldsymbol { v _ { \ell } } } \# \rho = \pi _ { \boldsymbol { v _ { \ell } } } \# \rho ^ { \prime } \qquad \mathrm { f o r ~ e v e r y ~ } \ell \in [ 2 k - 1 ] .
$$

Let $\Phi _ { 1 } , \Phi _ { 2 }$ be probability measures with smooth, compactly supported densities, and define

$$
\mu _ { i , j } = \rho _ { i , j } * \Phi _ { j } , \qquad \mu _ { i , j } ^ { \prime } = \rho _ { i , j } ^ { \prime } * \Phi _ { j } , \qquad i \in [ k ] , \ j \in [ 2 ] .
$$

Each $\mu _ { i , j }$ and $\mu _ { i , j } ^ { \prime }$ thus has a smooth, compactly supported probability density. Set

$$
\mu = \sum _ { i = 1 } ^ { k } w _ { i } \mu _ { i , 1 } \times \mu _ { i , 2 } , \quad \quad \mu ^ { \prime } = \sum _ { i = 1 } ^ { k } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } .
$$

Writing $\Phi = \Phi _ { 1 } \times \Phi _ { 2 }$ , Lemma 36 gives

$$
\mu = \rho * \Phi , \qquad \mu ^ { \prime } = \rho ^ { \prime } * \Phi .
$$

Hence, by Lemma 34, for every $\ell \in [ 2 k - 1 ]$

$$
\begin{array} { r l } & { \pi _ { v _ { \ell } } \# \mu = ( \pi _ { v _ { \ell } } \# \rho ) * ( \pi _ { v _ { \ell } } \# \Phi ) } \\ & { \qquad = ( \pi _ { v _ { \ell } } \# \rho ^ { \prime } ) * ( \pi _ { v _ { \ell } } \# \Phi ) } \\ & { \qquad = \pi _ { v _ { \ell } } \# \mu ^ { \prime } . } \end{array}
$$

Finally, $\rho - \rho ^ { \prime }$ is nonzero and compactly supported, so its Fourier transform is nonzero and analytic. Since $\mathcal { F } [ \Phi ] ( 0 ) = 1$ , the convolution theorem implies that $\left( \rho - \rho ^ { \prime } \right) * \Phi \neq 0$ . Therefore, $\mu \neq \mu ^ { \prime } .$ □

The proof of Proposition 9 again uses the normalization and smoothing steps from the proof of Theorem $^ { 8 , }$ so those parts are treated briefly. The main task is to exploit the symmetry of the paired directions to construct a nonzero signed measure that is annihilated by all 2k projections and, after normalization, can be written as the diference of two mixture-of-products models, each with at most $\lfloor k / 2 \rfloor + 1$ components.

Proof of Proposition 9. For $r \in [ k ]$ , let

$$
\begin{array} { r } { x _ { r } ^ { + } = \left[ 1 , - a _ { r } \right] ^ { \top } , \qquad x _ { r } ^ { - } = \left[ 1 , a _ { r } \right] ^ { \top } . } \end{array}
$$

Then

$$
\pi _ { v _ { r } ^ { + } } ( x _ { r } ^ { + } ) = 0 , \qquad \pi _ { v _ { r } ^ { - } } ( x _ { r } ^ { - } ) = 0 .
$$

Define

$$
\Delta _ { r } ^ { + } = \delta _ { 0 } - \delta _ { x _ { r } ^ { + } } , ~ \Delta _ { r } ^ { - } = \delta _ { 0 } - \delta _ { x _ { r } ^ { - } } ,
$$

and set

$$
\Gamma _ { r } = \Delta _ { r } ^ { + } * \Delta _ { r } ^ { - } .
$$

Since pushforward commutes with convolution,

$$
\pi _ { v _ { r } ^ { + } } \# \Gamma _ { r } = 0 , \qquad \pi _ { v _ { r } ^ { - } } \# \Gamma _ { r } = 0 .
$$

Consequently, the measure

$$
\eta = \Gamma _ { 1 } * \cdot \cdot * \Gamma _ { k }
$$

satisfies

$$
\pi _ { v _ { r } ^ { + } } \# \eta = \pi _ { v _ { r } ^ { - } } \# \eta = 0 \qquad \mathrm { f o r ~ e v e r y ~ } r \in [ k ] ,
$$

and is nonzero by Lemma 35.

Let

$$
S _ { a } = \frac { 1 } { 2 } ( \delta _ { - a } + \delta _ { a } ) , \qquad A = \delta _ { 0 } + \delta _ { 2 } .
$$

Expanding $\Gamma _ { r }$ gives

$$
\Gamma _ { r } = \delta _ { 0 } - \delta _ { [ 1 , - a _ { r } ] ^ { \top } } - \delta _ { [ 1 , a _ { r } ] ^ { \top } } + \delta _ { [ 2 , 0 ] ^ { \top } } = A \times \delta _ { 0 } - 2 \delta _ { 1 } \times S _ { a _ { r } } .
$$

For $s = 0 , \ldots , k ,$ define the positive measures

$$
\alpha _ { s } = 2 ^ { s } \delta _ { s } * A ^ { * ( k - s ) }
$$

and

$$
\beta _ { s } = \sum _ { { \cal J } \subseteq [ k ] \atop | { \cal J } | = s } * _ { } { \cal S } _ { a _ { r } } ,
$$

where the large asterisk denotes the convolution of the measures $S _ { a _ { \tau } }$ over $r \in J .$ , with the empty convolution interpreted as $\delta _ { 0 }$ . Expanding the convolution according to the number of factors from which the second term is selected gives

$$
\eta = \sum _ { s = 0 } ^ { k } ( - 1 ) ^ { s } \alpha _ { s } \times \beta _ { s } .
$$

Hence, setting

$$
\eta _ { \mathrm { e v e n } } = \sum _ { \stackrel { 0 \leq s \leq k } { s \mathrm { ~ e v e n } } } \alpha _ { s } \times \beta _ { s } , \qquad \eta _ { \mathrm { o d d } } = \sum _ { \stackrel { 0 \leq s \leq k } { s \mathrm { ~ o d d } } } \alpha _ { s } \times \beta _ { s } ,
$$

gives

$$
\eta = \eta _ { \mathrm { e v e n } } - \eta _ { \mathrm { o d d } } .
$$

The measures $\eta _ { \mathrm { e v e n } }$ and $\eta _ { \mathrm { o d d } }$ are positive and compactly supported, with ranks at most

$$
\left\lfloor { \frac { k } { 2 } } \right\rfloor + 1 \qquad { \mathrm { a n d } } \qquad \left\lceil { \frac { k } { 2 } } \right\rceil ,
$$

respectively. The projection equalities imply that $\eta ( \mathbb { R } ^ { 2 } ) = 0$ , so $\eta _ { \mathrm { e v e n } }$ and $\eta _ { \mathrm { o d d } }$ have the same nonzero mass. Set

$$
q = \left\lfloor { \frac { k } { 2 } } \right\rfloor + 1 .
$$

Normalizing as in the proof of Theorem 8 and padding the smaller representation with a zero-weight component gives weights $w _ { i } , w _ { i } ^ { \prime } \geq 0$ and compactly supported probability measures $\rho _ { i , j } , \rho _ { i , j } ^ { \prime } \in \mathscr { P } ( \mathbb { R } )$ , for $i \in [ q ]$ and $j \in [ 2 ]$ , such that

$$
\sum _ { i = 1 } ^ { q } w _ { i } = \sum _ { i = 1 } ^ { q } w _ { i } ^ { \prime } = 1 ,
$$

and the distinct probability measures

$$
\rho = \sum _ { i = 1 } ^ { q } w _ { i } \rho _ { i , 1 } \times \rho _ { i , 2 } , \qquad \rho ^ { \prime } = \sum _ { i = 1 } ^ { q } w _ { i } ^ { \prime } \rho _ { i , 1 } ^ { \prime } \times \rho _ { i , 2 } ^ { \prime }
$$

have equal projections along every $v _ { r } ^ { \pm }$

Let $\Phi _ { 1 } , \Phi _ { 2 }$ be probability measures with smooth, compactly supported densities, and define

$$
\mu _ { i , j } = \rho _ { i , j } * \Phi _ { j } , \qquad \mu _ { i , j } ^ { \prime } = \rho _ { i , j } ^ { \prime } * \Phi _ { j } , \qquad i \in [ q ] , \ j \in [ 2 ] .
$$

Each of these measures has a smooth, compactly supported density. Set

$$
\mu = \sum _ { i = 1 } ^ { q } w _ { i } \mu _ { i , 1 } \times \mu _ { i , 2 } , \qquad \mu ^ { \prime } = \sum _ { i = 1 } ^ { q } w _ { i } ^ { \prime } \mu _ { i , 1 } ^ { \prime } \times \mu _ { i , 2 } ^ { \prime } .
$$

By Lemma $^ { 3 6 , }$ these are obtained by convolving $\rho$ and $\rho ^ { \prime }$ with $\Phi _ { 1 } \times \Phi _ { 2 }$ . Lemma 34 therefore preserves all the prescribed projection equalities, while the same Fourier-analytic argument used in the proof of Theorem 8 gives $\mu \neq \mu ^ { \prime }$ □

Proof of Lemma 10. Recall that the one-dimensional Schwartz space ${ \mathfrak { S } } ( \mathbb { R } )$ consists of smooth functions $f \in C ^ { \infty } ( \mathbb { R } )$ satisfying

$$
| f | _ { N , \alpha } \triangleq \operatorname* { s u p } _ { x \in \mathbb { R } } ( 1 + | x | ) ^ { N } | f ^ { ( \alpha ) } ( x ) | < \infty \qquad \mathrm { f o r ~ a l l ~ i n t e g e r s ~ } N , \alpha \geq 0 ,
$$

and that $\mathfrak { S } ( \mathbb { R } ) \subset L ^ { 1 } ( \mathbb { R } ) \ [ 8 3 , \mathrm { p . ~ 2 3 7 } ]$ . Let $\mathcal { F }$ denote the Fourier transform.

Write

$$
v _ { i } = [ a _ { i } , b _ { i } ] ^ { \top } , \qquad i \in [ \ell ] ,
$$

and choose $c \geq 0$ such that

$$
\left| { \frac { b _ { i } } { a _ { i } } } \right| \leq c \qquad { \mathrm { w h e n e v e r ~ } } a _ { i } \neq 0 .
$$

Such a c exists because the collection of directions is finite.

The measures will be constructed in the Fourier domain. Let $\phi \in { \mathfrak { S } } ( \mathbb { R } )$ be a nonnegative, even $C ^ { \infty }$ bump function [90, Section 13] such that

$$
\phi ( 0 ) = 1 , \qquad \mathrm { s u p p } ( \phi ) \subset [ - 1 , 1 ] .
$$

Define

$$
{ \widehat { \mu } } ( \xi ) = \xi ^ { 2 } \phi ( \xi ) , \qquad { \widehat { \nu } } ( \xi ) = \phi ( \xi - ( c + 2 ) ) + \phi ( \xi + ( c + 2 ) ) .
$$

These functions are nonzero, real-valued, even, and belong to ${ \mathfrak { S } } ( \mathbb { R } )$ . Moreover,

$$
\operatorname { s u p p } ( { \widehat { \mu } } ) \subset [ - 1 , 1 ] , \qquad \operatorname { s u p p } ( { \widehat { \nu } } ) \subset [ c + 1 , c + 3 ] \cup [ - c - 3 , - c - 1 ] ,
$$

and

$$
\widehat { \mu } ( 0 ) = \widehat { \nu } ( 0 ) = 0 .
$$

Let

$$
f _ { \mu } = \mathcal { F } ^ { - 1 } [ \widehat { \mu } ] , \qquad f _ { \nu } = \mathcal { F } ^ { - 1 } [ \widehat { \nu } ] .
$$

Since the Fourier transform maps ${ \mathfrak { S } } ( \mathbb { R } )$ to itself [83, Corollary $8 . 2 3 ]$ , one has $f _ { \mu } , f _ { \nu } \in \mathfrak { S } ( \mathbb { R } ) \subset L ^ { 1 } ( \mathbb { R } )$ . Because $\widehat { \mu }$ and νb are real-valued and even, their inverse Fourier transforms are real-valued. They are also nonzero by the injectivity of the Fourier transform. Let $\mu , \nu \in \mathcal { M } ( \mathbb { R } )$ be the signed measures corresponding to the densities $f _ { \mu } , f _ { \nu } , \mathrm { s o } ~ \mathcal { F } [ \mu ] = \widehat { \mu }$ and $\mathcal { F } [ \nu ] = \widehat { \nu }$ . From this it follows that

$$
\mu ( \mathbb { R } ) = \widehat { \mu } ( 0 ) = 0 , \qquad \nu ( \mathbb { R } ) = \widehat { \nu } ( 0 ) = 0 .
$$

Fix $i \in [ \ell ] . \mathrm { ~ H ~ } a _ { i } = 0$ , then

$$
\pi _ { v _ { i } } \# ( \mu \times \nu ) = \mu ( \mathbb { R } ) T _ { b _ { i } } \# \nu = 0 ,
$$

where $T _ { b } ( x ) = b x .$

Now suppose that $a _ { i } \neq 0$ , and set

$$
s _ { i } = \frac { b _ { i } } { a _ { i } } .
$$

Since

$$
\pi _ { [ 1 , s _ { i } ] } \# ( \mu \times \nu ) = \mu * ( T _ { s _ { i } } \# \nu ) ,
$$

its Fourier transform satisfies

$$
\begin{array} { r } { \mathcal { F } \left[ \pi _ { \left[ 1 , s _ { i } \right] } \# ( \boldsymbol { \mu } \times \boldsymbol { \nu } ) \right] ( \boldsymbol { \xi } ) = \widehat { \mu } ( \boldsymbol { \xi } ) \widehat { \nu } ( s _ { i } \boldsymbol { \xi } ) . } \end{array}
$$

$\operatorname { I f } \xi \in \operatorname { s u p p } ( { \widehat { \mu } } )$ , then $| \xi | \le 1$ , and hence

$$
| s _ { i } \xi | \leq | s _ { i } | \leq c .
$$

Since $\widehat \nu ( \eta ) = 0$ whenever $| \eta | \leq c ,$ one has $\widehat { \nu } ( s _ { i } \xi ) = 0$ whenever $\xi \in \operatorname { s u p p } ( \widehat { \mu } )$ . Thus,

$$
{ \widehat { \mu } } ( \xi ) { \widehat { \nu } } ( s _ { i } \xi ) = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } \xi \in \mathbb { R } .
$$

The injectivity of the Fourier transform therefore gives

$$
\pi _ { [ 1 , s _ { i } ] } \# ( \mu \times \nu ) = 0 .
$$

Since

$$
\pi _ { v _ { i } } = T _ { a _ { i } } \circ \pi _ { [ 1 , s _ { i } ] } ,
$$

it follows that

$$
\pi _ { v _ { i } } \# ( \mu \times \nu ) = T _ { a _ { i } } \# \left( \pi _ { [ 1 , s _ { i } ] } \# ( \mu \times \nu ) \right) = 0 .
$$

Thus $\pi _ { v _ { i } } \# ( \boldsymbol { \mu } \times \boldsymbol { \nu } ) = 0$ for every $i \in [ \ell ]$

Proof of Corollary 11. By the construction in the proof of Lemma 10, there exist nonzero signed measures $\alpha , \beta \in { \mathcal { M } } ( \mathbb { R } )$ , each admitting a density in ${ \mathfrak { S } } ( \mathbb { R } )$ , such that

$$
\alpha ( \mathbb { R } ) = \beta ( \mathbb { R } ) = 0
$$

and

$$
\pi _ { v _ { i } } \# ( \alpha \times \beta ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ } i \in [ \ell ] .
$$

Write

$$
\alpha = \alpha ^ { + } - \alpha ^ { - } , \qquad \beta = \beta ^ { + } - \beta ^ { - } .
$$

Expanding $\alpha \times \beta$ and using linearity gives

$$
\pi _ { v _ { i } } \# \big ( \alpha ^ { + } \times \beta ^ { + } + \alpha ^ { - } \times \beta ^ { - } \big ) = \pi _ { v _ { i } } \# \big ( \alpha ^ { + } \times \beta ^ { - } + \alpha ^ { - } \times \beta ^ { + } \big )
$$

for every $i \in [ \ell ]$ i ∈ [ℓ].

Since α and $\beta$ are nonzero and have total mass zero,

$$
\alpha ^ { + } ( \mathbb { R } ) = \alpha ^ { - } ( \mathbb { R } ) > 0 , \qquad \beta ^ { + } ( \mathbb { R } ) = \beta ^ { - } ( \mathbb { R } ) > 0 .
$$

Normalizing the four factors therefore turns each side into a ran $\mathrm { k } \mathrm { - } \leq 2$ mixture-of-products probability measure with both weights equal to $1 / 2$ . The resulting models are distinct because their diference is a nonzero multiple of $\alpha \times \beta .$

Finally, if $f _ { \alpha }$ and $f _ { \beta }$ denote the Schwartz-class densities of α and $\beta ,$ then the densities of their Jordan components are bounded pointwise by $| f _ { \alpha } |$ and $| f _ { \beta } |$ . They therefore decay faster than any polynomial, and normalization preserves this property. □

The proof of Theorem 12 uses the following form of the uniform boundedness principle.

Theorem 38 (Banach–Steinhaus theorem [20, Theorem 3.15]). Let X be a Banach space and let $M \subset X ^ { * }$ . If

$$
\operatorname* { s u p } _ { m \in M } | m ( x ) | < \infty \qquad f o r \ e v e r y \ x \in X ,
$$

then M is bounded in norm.

Proof of Theorem 12. Let V denote the map in the theorem. Write $\boldsymbol { v } _ { i } = [ a _ { i } , b _ { i } ] ^ { \top }$ for $i \in [ \ell ]$ , and equip $\mathcal { M } ( \mathbb { R } ) ^ { \ell }$ with the norm

$$
\| ( \nu _ { 1 } , \dots , \nu _ { \ell } ) \| = \operatorname* { m a x } _ { i \in [ \ell ] } \| \nu _ { i } \| .
$$

If V is not injective on $\mathcal { R } _ { \leq 1 } ( [ 0 , 1 ] , [ 0 , 1 ] )$ , then it is not a homeomorphic embedding. Hence, assume that V is injective on this class.

Choose nonnegative functions $p ^ { + } , p ^ { - } \in C _ { c } ^ { \infty } ( ( 0 , 1 ) )$ ) with disjoint supports and

$$
\int _ { \mathbb R } p ^ { + } ( x ) d x = \int _ { \mathbb R } p ^ { - } ( x ) d x = 1 .
$$

Let $\mu ^ { + }$ and $\mu ^ { - }$ be the corresponding probability measures and define

$$
\alpha = \mu ^ { + } - \mu ^ { - } , \qquad f = p ^ { + } - p ^ { - } .
$$

Then

$$
\alpha ( \mathbb { R } ) = 0 , \qquad \| \alpha \| = 2 .
$$

Since $f$ is smooth and compactly supported, it has a finite Lipschitz constant, denoted by $L .$

For each $\delta \in ( 0 , 1 )$ , choose nonnegative functions $r _ { \delta } ^ { + } , r _ { \delta } ^ { - } \in C _ { c } ^ { \infty } ( \mathbb { R } )$ satisfying

$$
\int _ { \mathbb R } r _ { \delta } ^ { + } ( y ) d y = \int _ { \mathbb R } r _ { \delta } ^ { - } ( y ) d y = 1
$$

and

$$
\operatorname { s u p p } ( r _ { \delta } ^ { + } ) \subset ( 0 , \delta / 3 ) , \qquad \operatorname { s u p p } ( r _ { \delta } ^ { - } ) \subset ( 2 \delta / 3 , \delta ) .
$$

Let $\rho _ { \delta } ^ { + }$ and $\rho _ { \delta } ^ { - }$ be the corresponding probability measures and set

$$
\beta _ { \delta } = \rho _ { \delta } ^ { + } - \rho _ { \delta } ^ { - } , \qquad z _ { \delta } = \frac { 1 } { 4 } \alpha \times \beta _ { \delta } .
$$

The measure $z _ { \delta }$ belongs to $\mathcal { R } _ { \leq 1 } ( [ 0 , 1 ] , [ 0 , 1 ] )$ and has the $C ^ { \infty }$ density

$$
( x , y ) \longmapsto { \frac { 1 } { 4 } } f ( x ) \bigl ( r _ { \delta } ^ { + } ( y ) - r _ { \delta } ^ { - } ( y ) \bigr ) .
$$

Moreover, the four product measures obtained by expanding $\alpha \times \beta _ { \delta }$ are mutually singular. Therefore,

$$
\| z _ { \delta } \| = { \frac { 1 } { 4 } } \| \alpha \times \beta _ { \delta } \| = 1 .
$$

It remains to bound the projections of $z _ { \delta } .$ . For $c \in \mathbb { R }$ , let $T _ { c } ( x ) = c x$ . By the convolution formula for projections of product measures,

$$
\pi _ { [ a _ { i } , b _ { i } ] } \# ( \alpha \times \beta _ { \delta } ) = T _ { a _ { i } } \# \alpha \ast T _ { b _ { i } } \# \beta _ { \delta } .
$$

If $a _ { i } = 0 .$ , then

$$
\begin{array} { r } { T _ { a _ { i } } \# \alpha = \alpha ( \mathbb { R } ) \delta _ { 0 } = 0 . } \end{array}
$$

Similarly, if $b _ { i } = 0$ , then

$$
T _ { b _ { i } } \# \beta _ { \delta } = \beta _ { \delta } ( \mathbb { R } ) \delta _ { 0 } = 0 .
$$

Thus

$$
\pi _ { v _ { i } } \# z _ { \delta } = 0
$$

whenever $a _ { i } = 0$ or $b _ { i } = 0$

Now suppose that $a _ { i } \neq 0$ and $b _ { i } \neq 0$ . The measure $T _ { a _ { i } } \# \alpha$ has density

$$
f _ { a _ { i } } ( x ) = { \frac { 1 } { | a _ { i } | } } f ( x / a _ { i } ) ,
$$

whose Lipschitz constant is $L / | a _ { i } | ^ { 2 }$ . Let $h _ { \delta , i }$ denote the density of $\pi _ { v _ { i } } \# ( \alpha \times \beta _ { \delta } )$

To determine the density of the projection, consider first $\mu ^ { + } \times \rho _ { \delta } ^ { + }$ . If $X \sim \mu ^ { + }$ and $Y \sim \rho _ { \delta } ^ { + }$ are independent, then

$$
\pi _ { v _ { i } } \# ( \mu ^ { + } \times \rho _ { \delta } ^ { + } ) = \mathcal { L } ( a _ { i } X + b _ { i } Y ) .
$$

Conditioning on $Y = y$ , the random variable $a _ { i } X + b _ { i } Y$ has density

$$
x \longmapsto { \frac { 1 } { \left| a _ { i } \right| } } p ^ { + } \left( { \frac { x - b _ { i } y } { a _ { i } } } \right) .
$$

Averaging this conditional density over the distribution of Y gives

$$
x \longmapsto \int _ { \mathbb { R } } { \frac { 1 } { \left| a _ { i } \right| } } p ^ { + } \left( { \frac { x - b _ { i } y } { a _ { i } } } \right) r _ { \delta } ^ { + } ( y ) d y .
$$

Applying the same argument to the other three terms in

$$
\alpha \times \beta _ { \delta } = \mu ^ { + } \times \rho _ { \delta } ^ { + } - \mu ^ { + } \times \rho _ { \delta } ^ { - } - \mu ^ { - } \times \rho _ { \delta } ^ { + } + \mu ^ { - } \times \rho _ { \delta } ^ { - } ,
$$

and using linearity, shows that $\pi _ { v _ { i } } \# ( \alpha \times \beta _ { \delta } )$ has density

$$
h _ { \delta , i } ( x ) = \int _ { \mathbb { R } } f _ { a _ { i } } ( x - b _ { i } y ) { \big ( } r _ { \delta } ^ { + } ( y ) - r _ { \delta } ^ { - } ( y ) { \big ) } d y .
$$

Using the fact that $r _ { \delta } ^ { + }$ and $r _ { \delta } ^ { - }$ integrate to one, Fubini’s theorem gives

$$
\begin{array} { l } { \displaystyle \tilde { h } _ { \delta , \delta } ( x ) = \int _ { \mathbb { R } } f _ { a , ( x } - b _ { i } y ) r _ { \delta } ^ { + } ( y ) d y - \int _ { \mathbb { R } } f _ { a , ( x } - b _ { i } z ) r _ { \delta } ^ { - } ( z ) d z } \\ { \displaystyle \quad = \left( \int _ { \mathbb { R } } f _ { a _ { 1 } } ( x - b _ { i } y ) r _ { \delta } ^ { + } ( y ) d y \right) \left( \int _ { \mathbb { R } } r _ { \delta } ^ { - } ( z ) d z \right) } \\ { \displaystyle \quad \quad - \left( \int _ { \mathbb { R } } r _ { \delta } ^ { + } ( y ) d y \right) \left( \int _ { \mathbb { R } } f _ { a , ( x } - b _ { i } z ) r _ { \delta } ^ { - } ( z ) d z \right) } \\ { \displaystyle \quad = \iint _ { \mathbb { R } ^ { 2 } } f _ { a , ( x } - b _ { i } y ) r _ { \delta } ^ { + } ( y ) r _ { \delta } ^ { - } ( z ) d y d z } \\ { \displaystyle \quad \quad - \int _ { \mathbb { R } } f _ { a , \mathbb { R } } f _ { a , ( x } - b _ { i } z ) r _ { \delta } ^ { + } ( y ) r _ { \delta } ^ { - } ( z ) d y d z } \\ { \displaystyle \quad = \iint _ { \mathbb { R } ^ { 2 } } \{ f _ { a , ( x } - b _ { i } y ) - f _ { a _ { i } } ( x - b _ { i } z ) \} r _ { \delta } ^ { + } ( y ) r _ { \delta } ^ { - } ( z ) d y d z } \end{array}
$$

Since $f _ { a _ { i } }$ has Lipschitz constant $L / | a _ { i } | ^ { 2 }$ and $y , z \in [ 0 , \delta ]$ on the supports of the integrand,

$$
\vert h _ { \delta , i } ( x ) \vert \leq { \frac { L \vert b _ { i } \vert } { \vert a _ { i } \vert ^ { 2 } } } \delta .
$$

The support of $h _ { \delta , i }$ is contained in an interval of length at most $\left| a _ { i } \right| + \left| b _ { i } \right|$ , independently of $\delta .$ Consequently, there exists $C _ { i } < \infty$ , independent of $\delta ,$ such that

$$
\| \pi _ { v _ { i } } \# z _ { \delta } \| = \frac 1 4 \| h _ { \delta , i } \| _ { 1 } \leq C _ { i } \delta .
$$

Since the collection of directions is finite, there exists $C < \infty$ such that

$$
\| V ( z _ { \delta } ) \| \le C \delta \qquad \mathrm { f o r ~ e v e r y ~ } \delta \in ( 0 , 1 ) .
$$

For $n \geq 2$ , define

Then $x _ { n } \in \mathcal { R } _ { \leq 1 } ( [ 0 , 1 ] , [ 0 , 1 ] )$ and

$$
x _ { n } = n z _ { 1 / n ^ { 2 } } .
$$

$$
\| x _ { n } \| = n , \qquad \| V ( x _ { n } ) \| \leq { \frac { C } { n } } .
$$

Hence $V ( x _ { n } )$ converges to zero in norm and therefore also weak<sup>∗</sup>.

Let

$$
\begin{array} { r } { \widetilde { V } = \left( V \big | _ { \mathcal { R } _ { \leq 1 } ( [ 0 , 1 ] , [ 0 , 1 ] ) } \right) ^ { - 1 } . } \end{array}
$$

If $\widetilde { V }$ were weak<sup>∗</sup>–weak<sup>∗</sup> continuous at zero, then

$$
x _ { n } = { \widetilde { V } } ( V ( x _ { n } ) ) \ { \xrightarrow { \mathrm { w } ^ { * } } } { \widetilde { V } } ( 0 ) = 0 .
$$

In particular, for every $f \in C ( [ 0 , 1 ] ^ { 2 } )$ , the sequence $\left( \left. x _ { n } , f \right. \right) _ { n = 2 } ^ { \infty }$ would be bounded. Theorem 38 would then imply

$$
\operatorname* { s u p } _ { n \geq 2 } \left\| x _ { n } \right\| < \infty ,
$$

contradicting $\| x _ { n } \| = n$ . Thus $\widetilde { V }$ is not $\mathrm { w e a k ^ { * } { - } w e a k ^ { * } }$ continuous, and $V$ is not a wea $\boldsymbol { \xi } ^ { * } \mathrm { - w e a k ^ { * } }$ homeomorphic embedding.

Finally, every $x _ { n }$ constructed above admits a $C ^ { \infty }$ density. The same argument therefore applies when the domain is restricted to ran $\mathrm { k } \mathrm { - } \leq 1$ signed measures admitting $C ^ { \infty }$ densities. □

## Acknowledgments

The author would like to thank his wife, Julia Vandermeulen, for her support throughout this project.

## A Equivalence of RIP, RSC, and Injectivity

This appendix shows that, for linear maps on matrix spaces, injectivity on rank-≤ k matrices is equivalent to restricted strong convexity on rank-≤ 2k matrices and, up to rescaling, to restricted isometry on $\mathrm { r a n k } . . \leq 2 k$ matrices. The argument works with any choice of norms on $\mathbb { R } ^ { m \times n }$ and $\mathbb { R } ^ { \ell }$ . For definiteness, equip these spaces with the Frobenius and Euclidean norms, respectively, both denoted by $\| \cdot \|$ . Linear operators between these spaces are equipped with the corresponding operator norm. For $q \in \mathbb { N } .$ , let $R _ { \leq q } \subseteq \mathbb { R } ^ { m \times n }$ denote the set of matrices of rank at most $q ,$ and let $T : \mathbb { R } ^ { m \times n }  \mathbb { R } ^ { \ell }$ be linear.

Let $k \in \mathbb N .$ To begin, assume that $T$ is injective on $R _ { \leq k }$ . Any nonzero $x \in R _ { < 2 k }$ can be written as $x = y - z$ with $y , z \in R _ { \leq k }$ and $y \ne z$ . Since $T$ is injective on $R _ { \leq k }$ , it follows that $T \bar { ( y ) } \neq T ( z )$ , and hence $T ( x ) \neq 0$

Since $R _ { < 2 k }$ is closed, the set

$$
S _ { 2 k } \triangleq R _ { \leq 2 k } \cap \{ x : \| x \| = 1 \}
$$

is compact. Consider

$$
\kappa \triangleq \operatorname* { m i n } _ { x \in S _ { 2 k } } \Vert T ( x ) \Vert ^ { 2 } .
$$

The minimum is attained by compactness and continuity, and $\kappa > 0$ because $T ( x ) \neq 0$ for every $x \in S _ { 2 k }$ Therefore, for any nonzero $x \in R _ { \leq 2 k }$ ，

$$
\left\| T { \bigg ( } { \frac { x } { \| x \| } } { \bigg ) } \right\| ^ { 2 } \geq \kappa ,
$$

and hence

$$
\| T ( x ) \| ^ { 2 } \geq \kappa \| x \| ^ { 2 } , \qquad x \in R _ { \leq 2 k } ,\tag{42}
$$

which corresponds to restricted strong convexity of the quadratic loss $x \mapsto \| T ( x ) \| ^ { 2 }$ on $R _ { < 2 k } ;$ see $[ 3 ]$ . In $[ 3 ]$ this condition is stated using the Frobenius norm on $\mathbb { R } ^ { m \times n }$ and the Euclidean norm on $\mathbb { R } ^ { \overline { { \ell } } }$ . Thus injectivity of $T$ on $R _ { \leq k }$ implies restricted strong convexity on $R _ { < 2 k }$

Now assume restricted strong convexity in the form (42) for some $\kappa > 0$ . For $x \in R _ { \leq 2 k }$ one has

$$
\| T ( x ) \| ^ { 2 } \geq \kappa \| x \| ^ { 2 } .
$$

Also, since $T$ is linear on a finite-dimensional normed space, its operator norm is finite, and

$$
\| T ( x ) \| ^ { 2 } \leq \| T \| ^ { 2 } \| x \| ^ { 2 } .
$$

Define the rescaled operator

$$
\widetilde T = \sqrt { \frac { 2 } { \kappa + \| T \| ^ { 2 } } } T , \qquad \delta = \frac { \| T \| ^ { 2 } - \kappa } { \| T \| ^ { 2 } + \kappa } \in [ 0 , 1 ) .
$$

Then, for every $x \in R _ { \leq 2 k }$

$$
( 1 - \delta ) \| x \| ^ { 2 } \leq \| \widetilde T ( x ) \| ^ { 2 } \leq ( 1 + \delta ) \| x \| ^ { 2 } .
$$

Thus the restricted strong convexity lower bound on $R _ { \leq 2 k }$ implies, after rescaling, the restricted isometry property on $\mathrm { r a n k } . . \leq 2 k$ matrices.

Now assume that T satisfies the restricted isometry property on rank $- \le 2 k$ matrices with some $\delta \in [ 0 , 1 )$ ). For $x , y \in R _ { \leq k }$ , the diference $x - y$ has rank at most $2 k ,$ so

$$
( 1 - \delta ) \| x - y \| ^ { 2 } \leq \| T ( x - y ) \| ^ { 2 } \leq ( 1 + \delta ) \| x - y \| ^ { 2 } .
$$

If $x , y \in R _ { \leq k }$ with $x \neq y ,$ then $\| x - y \| > 0$ . Hence $\| T ( x - y ) \| > 0 ,$ so $T ( x ) \neq T ( y )$ . Thus the restricted isometry property on ran $\mathrm { k } { - } \le 2 k$ matrices implies injectivity of $T$ on $R _ { \leq k }$

Combining the preceding implications shows that injectivity of $T$ on $R _ { < k }$ , restricted strong convexity on $R _ { \leq 2 k }$ , and, up to rescaling, the restricted isometry property on ran $\mathrm { k } { - } \le 2 k$ matrices are equivalent.

## B Experimental Details

## B.1 Implementation and Runtime Details

In the experiments of Section 6.2, 250 HEPMASS samples were used for training and 250 for validation, while 1000 kin8nm samples were used for training and 300 for validation. For HEPMASS, a fixed test set of 20,000 samples was used across all 20 runs; for kin8nm, the remaining samples formed the test set in each split. These splits were chosen to balance computational cost and stability of model selection, providing suficient data for validation-based tuning while keeping the overall runtime manageable.

For SME, partitioning was performed on samples whose coordinates had been rounded to the nearest multiple of 0.125 for HEPMASS and 0.25 for kin8nm. This allows the use of weighted empirical measures rather than full empirical measures, significantly reducing the cost of Wasserstein distance computations. The rounding was used only for partitioning; the KDEs were fit to the original, unrounded samples using the resulting component labels. In preliminary testing, this rounding step did not appear to noticeably afect density-estimation performance.

This rounding and the incremental update strategy described in Section 5, together with parallelization across 9 CPU cores (Apple M2), substantially reduced computation time relative to a naive implementation of the SME objective. For HEPMASS, after the initial objective evaluation has cached the block-level contributions to the projected measures, evaluating the objective for a single candidate label change—that is, the loss resulting from moving one sample to a diferent component—takes approximately 6–7 ms. This evaluation includes computing and averaging the Wasserstein distances across all projections (42 graph edges with 4 projections per edge). With k = 4 components, the three alternative labels for one HEPMASS sample are evaluated separately and take approximately 21 ms in total, consistent with the $6 { - } 7$ ms cost per label. For kin8nm, evaluating one alternative label for one sample takes approximately 3–4 ms.

For HEPMASS, each objective evaluation comprises 168 independent computations, each consisting of a projection and its associated Wasserstein-distance evaluation. The experiments reported here used 9 CPU cores, leaving room for finer-grained parallelization and further speedups.

## B.1.1 MAF Implementation Details

For both datasets, MAF was implemented using zuko.flows.MAF with five autoregressive transformations and random feature permutations enabled. The hidden-layer widths were (64, 64) for HEPMASS and (32, 32) for kin8nm. Models were trained by minimizing the mean negative log-likelihood using Adam with learning rate $1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 6 }$ . HEPMASS used full-batch updates, while kin8nm used mini-batches of size 50.

Training continued until the validation negative log-likelihood failed to improve for 10 consecutive epochs for HEPMASS or 30 consecutive epochs for kin8nm. The parameter state attaining the lowest validation negative log-likelihood was restored before evaluating the test log-likelihood. All remaining Zuko and Adam settings were left at their default values.

## B.2 Additional Experimental Results

This subsection presents additional experimental results examining the behavior of the SME objective during optimization and its relationship to downstream density-estimation performance. The goal is to better understand the objective and the optimization process rather than provide an additional benchmark comparison.

The first experiment used HEPMASS with 20,000 training samples and $k = 8 0$ components, with the SME objective optimized for approximately 37,000 steps. Every 100 optimization steps, the corresponding mixture-of-Naive-Bayes KDE estimator was evaluated on a held-out test set of size 10,000. The graph structure, projection parameters, KDE fitting, and rounding procedure were the same as in the HEPMASS experiments of Section $6 . 2 .$ . Bandwidths were selected by 5-fold cross-validation over a logarithmic grid of 40 values ranging from $1 0 ^ { - 3 }$ to 1.

![](images/80d79c1251354422e28b22f7538127cc403cb91f78cfbe305743756102feaf15.jpg)  
(a) Single run with 20,000 training samples and k = 80.

![](images/efc6238ac66839d0ea2b7bc9e837735a27d24a676802dca9d7ab92a058c49dea.jpg)  
Silhouette objective Estimator log likelihood

(b) Mean and standard deviation over 20 runs (500 training samples, k = 5).

Fig. 3: Silhouette objective and estimator log-likelihood during SME optimization on HEPMASS.

The full experiment required approximately 90 hours, roughly one half of which was spent periodically fitting the mixture-of-Naive-Bayes KDE estimators and evaluating them on the test set. Evaluating al 79 alternative labels for a single sample took approximately 4.5 seconds. Figure 3a shows the silhouette objective and test log-likelihood throughout optimization.

As the silhouette objective decreases during optimization, test log-likelihood improves nearly monotonically. This provides empirical evidence that minimizing the silhouette objective tends to produce partitions that are increasingly useful for constructing density estimators of the form (16).

Figure 3b shows the mean and standard deviation of the corresponding trajectories over 20 independent HEPMASS runs with 500 training samples and k = 5 components. Each run continued until no single sample reassignment decreased the objective. Because the runs converged after diferent numbers of steps, each trajectory was extended after convergence by holding its last recorded value constant.

As in the larger experiment, estimator log-likelihood improves as the silhouette objective decreases. The improvement then plateaus, with a slight decline late in optimization even as the silhouette objective continues to decrease. The cause of this behavior is unclear.

## References

[1] E. J. Candes and T. Tao. Decoding by linear programming. IEEE Trans. Inf. Theor., 51(12):4203–4215, December 2005. ISSN 0018-9448. doi: 10.1109/TIT.2005.858979. URL https://doi.org/10.1109/ TIT.2005.858979.

[2] Benjamin Recht, Maryam Fazel, and Pablo A. Parrilo. Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization. SIAM Rev., 52(3):471–501, August 2010. ISSN 0036- 1445. doi: 10.1137/070697835. URL http://dx.doi.org/10.1137/070697835.

[3] Sahand Negahban and Martin J. Wainwright. Estimation of (near) low-rank matrices with noise and high-dimensional scaling. Ann. Statist., 39(2):1069–1097, 04 2011. doi: 10.1214/10-AOS850. URL https://doi.org/10.1214/10-AOS850.

[4] Nicolas Bonnotte. Unidimensional and Evolution Methods for Optimal Transportation. Theses, Universit´e Paris Sud - Paris XI ; Scuola normale superiore (Pise, Italie), December 2013. URL https://theses.hal.science/tel-00946781.

[5] Nicolas Bonneel, Julien Rabin, Gabriel Peyr´e, and Hanspeter Pfister. Sliced and Radon Wasserstein barycenters of measures. Journal of Mathematical Imaging and Vision, 51(1):22–45, Jan 2015. ISSN 1573-7683. doi: 10.1007/s10851-014-0506-3. URL https://doi.org/10.1007/s10851-014-0506-3.

[6] Kimia Nadjahi, Alain Durmus, L´ena¨ıc Chizat, Soheil Kolouri, Shahin Shahrampour, and Umut Simsekli. Statistical and topological properties of sliced probability divergences. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 20802–20812. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper\_files/paper/2020/file/eefc9e10ebdc4a2333b42b2dbb8f27b6-Paper.pdf.

[7] Khai Nguyen, Nhat Ho, Tung Pham, and Hung Bui. Distributional sliced-Wasserstein and applications to generative modeling. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=QYjO70ACDK.

[8] Clayton Scott, Gilles Blanchard, and Gregory Handy. Classification with asymmetric label noise: Consistency and maximal denoising. In Shai Shalev-Shwartz and Ingo Steinwart, editors, Proceedings of the 26th Annual Conference on Learning Theory, volume 30 of Proceedings of Machine Learning Research, pages 489–511, Princeton, NJ, USA, 12–14 Jun 2013. PMLR. URL https://proceedings.mlr.press/ v30/Scott13.html.

[9] Gilles Blanchard and Clayton Scott. Decontamination of mutually contaminated models. In Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics, AISTATS 2014, Reykjavik, Iceland, April 22-25, 2014, pages 1–9, 2014. URL http://jmlr.org/proceedings/papers/ v33/blanchard14.html.

[10] Sanjeev Arora, Rong Ge, Ravindran Kannan, and Ankur Moitra. Computing a nonnegative matrix factorization – provably. In Proceedings of the Forty-fourth Annual ACM Symposium on Theory of Computing, STOC ’12, pages 145–162, New York, NY, USA, 2012. ACM. ISBN 978-1-4503-1245-5. doi: 10.1145/2213977.2213994. URL http://doi.acm.org/10.1145/2213977.2213994.

[11] Peter Hall and Xiao-Hua Zhou. Nonparametric estimation of component distributions in a multivariate mixture. The Annals of Statistics, 31(1):201 – 224, 2003. doi: 10.1214/aos/1046294462. URL https: //doi.org/10.1214/aos/1046294462.

[12] Peter Hall, Amnon Neeman, Reza Pakyari, and Ryan Elmore. Nonparametric inference in multivariate mixtures. Biometrika, 92(3):667–678, 09 2005. ISSN 0006-3444. doi: 10.1093/biomet/92.3.667. URL https://doi.org/10.1093/biomet/92.3.667.

[13] Robert A Vandermeulen and Antoine Ledent. Beyond smoothness: Incorporating low-rank analysis into nonparametric density estimation. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 12180–12193. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper/2021/ file/6591d327f6f731e589b0e869adadf940-Paper.pdf.

[14] Caleb Kwon and Eric Mbakop. Estimation of the number of components of nonparametric multivariate finite mixture models. The Annals of Statistics, 49(4):2178 – 2205, 2021. doi: 10.1214/20-AOS2032. URL https://doi.org/10.1214/20-AOS2032.

[15] Magda Amiridi, Nikos Kargas, and Nicholas D. Sidiropoulos. Low-rank characteristic tensor density estimation part I: Foundations. IEEE Transactions on Signal Processing, 70:2654–2668, 2022. doi: 10.1109/TSP.2022.3175608.

[16] Julien Chhor, Olga Klopp, and Alexandre B. Tsybakov. Generalized multi-view model: Adaptive density estimation under low-rank constraints. Journal of Machine Learning Research, 26(236):1–52, 2025. URL http://jmlr.org/papers/v26/24-1729.html.

[17] Yichen Lyu and Pengkun Yang. Identifiability and estimation in high-dimensional nonparametric latent structure models. In Nika Haghtalab and Ankur Moitra, editors, Proceedings ofThirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 3879–3880. PMLR, 30 Jun–04 Jul 2025. URL https://proceedings.mlr.press/v291/lyu25a.html.

[18] Tatiana Benaglia, Didier Chauveau, and David R. Hunter. An EM-like algorithm for semi- and nonparametric estimation in multivariate mixtures. Journal of Computational and Graphical Statistics, 18 (2):505–526, 2009. doi: 10.1198/jcgs.2009.07175. URL https://doi.org/10.1198/jcgs.2009.07175.

[19] Robert A. Vandermeulen. Pmode: Theoretically grounded and modular mixture modeling, 2025. URL https://arxiv.org/abs/2508.21396.

[20] Mari´an Fabian, Petr Habala, Petr H´ajek, Vicente Montesinos Santaluc´ıa, Jan Pelant, and V´aclav Zizler. Functional Analysis and Infinite-Dimensional Geometry. Springer New York, New York, NY, 2001. ISBN 978-1-4757-3480-5. doi: 10.1007/978-1-4757-3480-5. URL https://doi.org/10.1007/ 978-1-4757-3480-5.

[21] Colin Petitjean. The set of elementary tensors is weakly closed in projective tensor products. Bulletin of the Australian Mathematical Society, 111(2):336–343, 2025. doi: 10.1017/S0004972724000376.

[22] David Donoho and Victoria Stodden. When does non-negative matrix factorization give a correct decomposition into parts? In S. Thrun, L. K. Saul, and B. Sch¨olkopf, editors, Advances in Neural Information Processing Systems 16, pages 1141–1148. MIT Press, 2004.

[23] Sanjeev Arora, Rong Ge, and Ankur Moitra. Learning topic models – going beyond SVD. In Proceedings of the 2012 IEEE 53rd Annual Symposium on Foundations of Computer Science, FOCS ’12, pages 1–10, Washington, DC, USA, 2012. IEEE Computer Society. ISBN 978-0-7695-4874-6. doi: 10.1109/FOCS. 2012.49. URL http://dx.doi.org/10.1109/FOCS.2012.49.

[24] Nikos Kargas, Nicholas D. Sidiropoulos, and Xiao Fu. Tensors, learning, and “Kolmogorov extension” for finite-alphabet random vectors. IEEE Transactions on Signal Processing, 66(18):4854–4868, 2018. doi: 10.1109/TSP.2018.2862383.

[25] Elizabeth S. Allman, Catherine Matias, and John A. Rhodes. Identifiability of parameters in latent structure models with many observed variables. Ann. Statist., 37(6A):3099–3132, 12 2009. doi: 10. 1214/09-AOS689. URL http://dx.doi.org/10.1214/09-AOS689.

[26] Sidney J. Yakowitz and John D. Spragins. On the identifiability of finite mixtures. Ann. Math. Statist., 39(1):209–214, 02 1968. doi: 10.1214/aoms/1177698520. URL http://dx.doi.org/10.1214/aoms/ 1177698520.

[27] Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M. Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. Journal of Machine Learning Research, 15:2773– 2832, 2014. URL http://jmlr.org/papers/v15/anandkumar14b.html.

[28] Runshi Tang, Julien Chhor, Olga Klopp, Alexandre B. Tsybakov, and Anru R. Zhang. Optimal estimation of discrete multiview distributions under heteroskedastic multinomial sampling, 2026. URL https://arxiv.org/abs/2605.24858.

[29] Le Song and Bo Dai. Robust low rank kernel embeddings of multivariate distributions. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013. URL https://proceedings.neurips. cc/paper/2013/file/49b8b4f95f02e055801da3b4f58e28b7-Paper.pdf.

[30] Le Song, Animashree Anandkumar, Bo Dai, and Bo Xie. Nonparametric estimation of multi-view latent variable models. In Eric P. Xing and Tony Jebara, editors, Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 640– 648, Beijing, China, 22–24 Jun 2014. PMLR. URL https://proceedings.mlr.press/v32/songa14. html.

[31] St´ephane Bonhomme, Koen Jochmans, and Jean-Marc Robin. Estimating multivariate latent-structure models. The Annals of Statistics, 44(2):540 – 563, 2016. doi: 10.1214/15-AOS1376. URL https: //doi.org/10.1214/15-AOS1376.

[32] Nikos Kargas and Nicholas D. Sidiropoulos. Learning mixtures of smooth product distributions: Identifiability and algorithm. In Kamalika Chaudhuri and Masashi Sugiyama, editors, Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 388–396. PMLR, 16–18 Apr 2019. URL https: //proceedings.mlr.press/v89/kargas19a.html.

[33] Behrooz Tahmasebi, Seyed Abolfazl Motahari, and Mohammad Ali Maddah-Ali. On the Identifiability of Finite Mixtures of Finite Product Measures. arXiv e-prints, art. arXiv:1807.05444, July 2018. doi: 10.48550/arXiv.1807.05444.

[34] Charles J. Stone. Optimal Rates of Convergence for Nonparametric Estimators. The Annals of Statistics, 8(6):1348 – 1360, 1980. doi: 10.1214/aos/1176345206. URL https://doi.org/10.1214/aos/ 1176345206.

[35] Charles J. Stone. Optimal Global Rates of Convergence for Nonparametric Regression. The Annals of Statistics, 10(4):1040 – 1053, 1982. doi: 10.1214/aos/1176345969. URL https://doi.org/10.1214/ aos/1176345969.

[36] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Publishing Company, Incorporated, 1st edition, 2008.

[37] Robert A. Vandermeulen. Sample complexity using infinite multiview models, 2023. URL https: //arxiv.org/abs/2302.04292.

[38] A. R´enyi. On projections of probability distributions. Acta Mathematica Academiae Scientiarum Hungaricae, 3(3):131–142, September 1952. ISSN 0001-5954, 1588-2632. doi: 10.1007/BF02022515. URL http://link.springer.com/10.1007/BF02022515.

[39] Walter M. Gilbert. Projections of probability distributions. Acta Mathematica Academiae Scientiarum Hungarica, 6(1):195–198, March 1955. ISSN 1588-2632. doi: 10.1007/BF02021275. URL https: //doi.org/10.1007/BF02021275.

[40] Juan Antonio Cuesta-Albertos, Ricardo Fraiman, and Thomas Ransford. A Sharp Form of the Cram´er– Wold Theorem. Journal of Theoretical Probability, 20(2):201–209, June 2007. ISSN 1572-9230. doi: 10.1007/s10959-007-0060-7. URL https://doi.org/10.1007/s10959-007-0060-7.

[41] Henry Teicher. Identifiability of finite mixtures. Ann. Math. Statist., 34(4):1265–1269, 12 1963. doi: 10.1214/aoms/1177703862. URL http://dx.doi.org/10.1214/aoms/1177703862.

[42] Yuval Rabani, Leonard J. Schulman, and Chaitanya Swamy. Learning mixtures of arbitrary distributions over large discrete domains. In Proceedings of the 5th Conference on Innovations in Theoretical Computer Science, ITCS ’14, pages 207–224, New York, NY, USA, 2014. ACM. ISBN 978-1-4503-2698-8. doi: 10.1145/2554797.2554818. URL http://doi.acm.org/10.1145/2554797.2554818.

[43] Robert A. Vandermeulen and Clayton D. Scott. An operator theoretic approach to nonparametric mixture models. Ann. Statist., 47(5):2704–2733, 10 2019. doi: 10.1214/18-AOS1762. URL https: //doi.org/10.1214/18-AOS1762.

[44] Alexander Ritchie, Robert A Vandermeulen, and Clayton Scott. Consistent estimation of identifiable nonparametric mixture models from grouped observations. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 11676–11686. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/ 2020/file/866d90e0921ac7b024b47d672445a086-Paper.pdf.

[45] Yun Wei and XuanLong Nguyen. Convergence of de Finetti’s mixing measure in latent structure models for observed exchangeable sequences. The Annals of Statistics, 50(4):1859 – 1889, 2022. doi: 10.1214/21-AOS2120. URL https://doi.org/10.1214/21-AOS2120.

[46] Robert A. Vandermeulen and Ren´e Saitenmacher. Generalized identifiability bounds for mixture models with grouped samples. IEEE Transactions on Information Theory, 70(4):2746–2758, 2024. doi: 10.1109/ TIT.2024.3367433.

[47] David M. Blei, Andrew Y. Ng, and Michael I. Jordan. Latent Dirichlet allocation. J. Mach. Learn. Res., 3:993–1022, March 2003. ISSN 1532-4435. URL http://dl.acm.org/citation.cfm?id=944919. 944937.

[48] Dat Do, Sunrit Chakraborty, Jonathan Terhorst, and XuanLong Nguyen. Dirichlet moment tensors and the correspondence between admixture and mixture of product models, 2025. URL https://arxiv. org/abs/2509.25441.

[49] Holger Rauhut, Reinhold Schneider, and Zeljka Stojanac. Low rank tensor recovery via iterative hard<sup>ˇ</sup> thresholding. Linear Algebra and its Applications, 523:220–262, 2017. ISSN 0024-3795. doi: https: //doi.org/10.1016/j.laa.2017.02.028. URL https://www.sciencedirect.com/science/article/pii/ S0024379517301295.

[50] D. L. Donoho. Compressed sensing. IEEE Transactions on Information Theory, 52(4):1289–1306, April 2006. ISSN 0018-9448. doi: 10.1109/TIT.2006.871582.

[51] Emmanuel J Candes and Terence Tao. Reflections on compressed sensing. IEEE Information Theory Society Newsletter, 58(4):20–23, 2008.

[52] Richard Baraniuk, Mark Davenport, Ronald DeVore, and Michael Wakin. A simple proof of the restricted isometry property for random matrices. Constructive Approximation, 28(3):253–263, Dec 2008. ISSN 1432-0940. doi: 10.1007/s00365-007-9003-x. URL https://doi.org/10.1007/ s00365-007-9003-x.

[53] Anna Gilbert and Piotr Indyk. Sparse recovery using sparse matrices. Proceedings of the IEEE, 98(6): 937–947, 2010. doi: 10.1109/JPROC.2010.2045092.

[54] Afonso S. Bandeira, Edgar Dobriban, Dustin G. Mixon, and William F. Sawin. Certifying the restricted isometry property is hard. IEEE Transactions on Information Theory, 59(6):3448–3450, 2013. doi: 10.1109/TIT.2013.2248414.

[55] Han Liu, Min Xu, Haijie Gu, Anupam Gupta, John Laferty, and Larry Wasserman. Forest density estimation. J. Mach. Learn. Res., 12:907–951, July 2011. ISSN 1532-4435. URL http://dl.acm.org/ citation.cfm?id=1953048.2021032.

[56] L´aszl´o Gy¨orfi, Aryeh Kontorovich, and Roi Weiss. Tree density estimation. IEEE Transactions on Information Theory, 69(2):1168–1176, 2023. doi: 10.1109/TIT.2022.3214888.

[57] Robert A. Vandermeulen, Wai Ming Tai, and Bryon Aragam. Breaking the curse of dimensionality in structured density estimation. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 65852– 65883. Curran Associates, Inc., 2024. doi: 10.52202/079017-2105.

[58] Robert A. Vandermeulen, Wai Ming Tai, and Bryon Aragam. Dimension-independent rates for structured neural density estimation. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 60857–60879. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/ vandermeulen25a.html.

[59] Jianqing Fan, Yihong Gu, and Ximing Li. Optimal estimation of a factorizable density using difusion models with relu neural networks, 2025. URL https://arxiv.org/abs/2510.03994.

[60] Hyeok Kyu Kwon, Dongha Kim, Ilsang Ohn, and Minwoo Chae. Nonparametric estimation of a factorizable density using difusion models. Journal of Machine Learning Research, 27(22):1–125, 2026. URL http://jmlr.org/papers/v27/25-0121.html.

[61] C´edric Villani. Optimal Transport: Old and New, volume 338 of Grundlehren der mathematischen Wissenschaften. Springer-Verlag, Berlin, 2009. ISBN 978-3-540-71049-3.

[62] Jiaqi Xi and Jonathan Niles-Weed. Distributional convergence of the sliced Wasserstein process. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=m8YYs8nJF3T.

[63] Jonathan Niles-Weed and Quentin Berthet. Minimax estimation of smooth densities in wasserstein distance. The Annals of Statistics, 50(3):1519–1540, 2022. ISSN 00905364, 21688966. URL https: //www.jstor.org/stable/27172252.

[64] Ishan Deshpande, Ziyu Zhang, and Alexander Schwing. Generative modeling using the sliced Wasserstein distance. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3483–3491, 2018. doi: 10.1109/CVPR.2018.00367.

[65] Jie Wang, Rui Gao, and Yao Xie. Two-sample test using projected Wasserstein distance. In 2021 IEEE International Symposium on Information Theory (ISIT), pages 3320–3325. IEEE Press, 2021. doi: 10.1109/ISIT45174.2021.9518186. URL https://doi.org/10.1109/ISIT45174.2021.9518186.

[66] Dimitri Meunier, Massimiliano Pontil, and Carlo Ciliberto. Distribution regression with sliced Wasserstein kernels. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 15501–15523. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/meunier22b.html.

[67] Khai Nguyen, Dang Nguyen, and Nhat Ho. Self-attention amortized distributional projection optimization for sliced Wasserstein point-cloud reconstruction. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 26008–26030. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/nguyen23e.html.

[68] Xin Bing, Florentina Bunea, and Jonathan Niles-Weed. Estimation and inference for the Wasserstein distance between mixing measures in topic models, 2024. URL https://arxiv.org/abs/2206.12768.

[69] Soheil Kolouri, Gustavo K. Rohde, and Heiko Hofmann. Sliced Wasserstein distance for learning Gaussian mixture models. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3427–3436, 2018. doi: 10.1109/CVPR.2018.00361.

[70] Ricardo Fraiman, Leonardo Moreno, and Thomas Ransford. A Cram´er–Wold theorem for mixtures. Electronic Communications in Probability, 30:1 – 10, 2025. doi: 10.1214/25-ECP711. URL https: //doi.org/10.1214/25-ECP711.

[71] Ricardo Fraiman, Leonardo Moreno, and Thomas Ransford. A Cram´er–Wold theorem for elliptical distributions. J. Multivar. Anal., 196(C), July 2023. ISSN 0047-259X. doi: 10.1016/j.jmva.2023.105176. URL https://doi.org/10.1016/j.jmva.2023.105176.

[72] Sigurdur Helgason. Integral Geometry and Radon Transforms. Springer, New York, NY, 2010. ISBN 978-1-4419-6055-9. doi: 10.1007/978-1-4419-6055-9.

[73] Avinash C. Kak and Malcolm Slaney. Principles of Computerized Tomographic Imaging. Society for Industrial and Applied Mathematics, 2001. doi: 10.1137/1.9780898719277. URL https://epubs.siam. org/doi/abs/10.1137/1.9780898719277.

[74] Patrizio Campisi and Karen Egiazarian, editors. Blind Image Deconvolution: Theory and Applications. CRC Press, 2007. doi: 10.1201/9781420007299.

[75] Ali Ahmed, Benjamin Recht, and Justin Romberg. Blind deconvolution using convex programming. IEEE Transactions on Information Theory, 60(3):1711–1732, 2014. doi: 10.1109/TIT.2013.2294644.

[76] Jianqing Fan. On the Optimal Rates of Convergence for Nonparametric Deconvolution Problems. The Annals of Statistics, 19(3):1257 – 1272, 1991. doi: 10.1214/aos/1176348248. URL https://doi.org/ 10.1214/aos/1176348248.

[77] G´erard Biau and Luc Devroye. Lectures on the Nearest Neighbor Method. Springer International Publishing, Cham, 2015. ISBN 978-3-319-25388-6. doi: 10.1007/978-3-319-25388-6.

[78] Daniel Whiteson. HEPMASS. UCI Machine Learning Repository, 2016.

[79] Zoubin Ghahramani. Kin family of datasets. DELVE, University of Toronto, 1996. URL https: //www.cs.toronto.edu/<sub>\~</sub>delve/data/kin/desc.html.

[80] George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/file/ 6c1da886822c67822bcf3679d04369fa-Paper.pdf.

[81] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12: 2825–2830, 2011.

[82] Fran¸cois Rozet et al. Zuko: Normalizing flows in PyTorch, 2022. URL https://pypi.org/project/ zuko.

[83] Gerald B. Folland. Real analysis: modern techniques and their applications. Pure and applied mathematics. Wiley, 1999. ISBN 9780471317166. URL http://books.google.com/books?id=uPkYAQAAIAAJ.

[84] Charalambos D. Aliprantis and Kim C. Border. Infinite Dimensional Analysis: A Hitchhiker’s Guide. Springer, 2006. ISBN 978-3-540-29586-0. doi: 10.1007/3-540-29587-9.

[85] Anirban DasGupta. Fundamentals of Probability: A First Course. Springer New York, New York, NY, 2010. ISBN 978-1-4419-5780-1. doi: 10.1007/978-1-4419-5780-1. URL https://doi.org/10.1007/ 978-1-4419-5780-1.

[86] O. Kallenberg. Foundations of Modern Probability. Probability and Its Applications. Springer New York, 2002. ISBN 9780387953137. URL https://books.google.com/books?id=L6fhXh13OyMC.

[87] Raymond A. Ryan. Introduction to Tensor Products of Banach Spaces. Springer London, London, 2002. ISBN 978-1-4471-3903-4. doi: 10.1007/978-1-4471-3903-4. URL https://doi.org/10.1007/ 978-1-4471-3903-4.

[88] Andrzej Szankowski. B(H) does not have the approximation property. Acta Mathematica, 147:89 – 108, 1981. doi: 10.1007/BF02392870. URL https://doi.org/10.1007/BF02392870.

[89] John L. Kelley. General topology. D. Van Nostrand Company, Inc., Toronto-New York-London, 1955.

[90] L.W. Tu. An Introduction to Manifolds. Universitext. Springer New York, 2010. ISBN 9781441973993. URL https://books.google.de/books?id=br1KngEACAAJ.