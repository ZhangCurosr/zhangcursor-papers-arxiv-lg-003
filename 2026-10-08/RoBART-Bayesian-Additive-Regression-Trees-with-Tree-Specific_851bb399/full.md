# RoBART: Bayesian Additive Regression Trees with Tree-Specific Rotations

Jeongung Heo

Yonsei University

Seonghyun Jeong

## Abstract

Bayesian additive regression trees (BART) can require many splits to approximate boundaries misaligned with the predictor axes. RoBART assigns each tree a rotation shared by all internal nodes, retaining axis-aligned splits in rotated coordinates and constant leaves. We jointly propose a Givens rotation sequence and cutpoints on the resulting grid by Metropolis–Hastings and establish reversibility with respect to the conditional posterior with leaf means integrated out. For additive functions with component-specific rotations and anisotropic Hölder smoothness, we prove posterior contraction in empirical $L _ { 2 }$ distance and for the noise standard deviation. Under the stated prior, design, and grid conditions, with fixed numbers of predictors, trees, and components and no more components than trees, the rate is a sum of componentwise rates determined by smoothness and the number of rotated coordinates used. We also establish a posterior contraction lower bound showing that there exist functions for which RoBART adapts to the intrinsic dimension but axis-aligned BART does not.

## 1 INTRODUCTION

Bayesian additive regression trees (BART) represent a regression mean as a sum of regularized trees (Chipman et al., 2010; Hill et al., 2020). Each internal node in standard BART splits on one predictor.

These splits produce rectangular cells, so approximating an oblique boundary can require many splits.

Figure 1(a) compares posterior means with the true function $f ( x ) = \mathbf { 1 } \{ | x _ { 1 } | > | x _ { 2 } | \}$ . A single $4 5 ^ { \circ }$ rotation aligns both diagonal boundaries with the coordinate axes. Away from the boundaries, a depth-two tree with four leaves and cuts at zero represents the function; a finite cutpoint grid may introduce additional approximation error. Figure 1(b) shows a Gaussian spike, $f ( x ) = 0 . 0 2 + 0 . 6 5 \exp \{ - r ^ { 2 } / ( 2$ $0 . 4 ^ { 2 } ) \} + 0 . 7 0 \exp \{ - r ^ { 2 } / ( 2 { \cdot } 0 . 0 8 ^ { 2 } ) \}$ , where $r ^ { 2 } = x _ { 1 } ^ { 2 } + x _ { 2 } ^ { 2 }$

We propose RoBART. For p predictors, each tree uses a $p \times p$ rotation shared by all internal nodes, axis-aligned splits in the rotated coordinates, and constant leaves. Rotations may difer across trees, and each tree may split on only some rotated coordinates.

For $B \in \mathbb { R } ^ { d \times p } , 1 \leq d \leq p , h ( B x )$ is unchanged by $B \mapsto A B$ and $h ( u ) \mapsto h ( A ^ { - 1 } u )$ for invertible $A \in$ $\mathbb { R } ^ { d \times d }$ . Orthonormal rows permit orthogonal basis changes. With known $d \ll p .$ , a row-orthonormal $d \times p$ matrix uses fewer parameters, but learning d requires changes to the dimension, split coordinates, and cutpoint arrays (Green, 1995; Collins et al., 2024). Our $p \times p$ parameterization keeps inputs in R<sup>p</sup> and split coordinates in $\{ 1 , \ldots , p \}$ without resolving nonidentifiability.

With tree structure, splitting coordinates, and cutpoint indices fixed, small rotations can preserve leaf assignments and admissible index sets, leaving the conditional posterior unchanged after integrating out the leaf means. Since the cutpoint grid depends on the rotated predictors, we jointly propose a Givens rotation sequence and afected cutpoints on the rebuilt grid.

![](images/b897bd0f922a24001352b2a2bdbc655b655f2fd877d494ded6ccfd925384af8f.jpg)  
Figure 1: True functions and posterior means for (a) the X-shaped signal and (b) a Gaussian spike. Both use $n = 8 0 0$ uniform inputs on $[ - 1 , 1 ] ^ { 2 }$ and independen $N ( 0 , 0 . 1 ^ { 2 } )$ errors. All methods use 200 trees. Color scales are shared within rows but difer between rows.

We establish reversibility of this joint update and posterior contraction for additive regression functions with component-specific rotations and anisotropic smoothness. The contraction result assumes fixed predictor dimension, tree count, and component count. Theorem 1 gives the rate under the conditions in Section 5.

Theorem 2 also constructs a fixed Hölder ridge function on a tensor design for which the axis-aligned BART posterior, under the same remaining prior specifications, cannot attain the one-dimensional contraction rate achieved by RoBART.

## 2 RELATED WORK

Standard BART uses axis-aligned splits and constant leaf values (Chipman et al., 2010). Soft BART (SBART) replaces hard leaf indicators with smooth routing weights (Linero and Yang, 2018). ridge-BART models leaf outputs as linear combinations of ridge functions (Yee et al., 2024). Our model retains hard splits and constant leaves and learns the coordinates in which each tree splits.

Oblique BART assigns a hyperplane to each split without requiring the normal vectors to come from one orthogonal matrix (Nguyen et al., 2025). We restrict the normal vectors within a tree to rows of one rotation matrix. Random rotation ensembles also use separate rotations for base learners, but choose them before fitting (Blaser and Fryzlewicz, 2016). Here each rotation is learned jointly with its

tree.

Bayesian Projection Pursuit Regression represents the mean as a sum of univariate ridge functions and learns their number by reversible-jump MCMC (Collins et al., 2024). Each of our trees can instead use several rotated coordinates and represent interactions among them.

Our analysis builds on posterior contraction results for Bayesian trees and forests (Ročková and van der Pas, 2020; Ročková and Saha, 2019) and anisotropic BART (Jeong and Ročková, 2023), including the additive extension in Section 6.4 and Theorem 7 of the latter work. We account for a separate unknown rotation and the resulting cutpoint grid for each component. Theorem 1 gives the rate, while Theorem 2 establishes a fixed-truth rate separation from axis-aligned BART; proofs are in Supplements B and C.

Figure 1 compares the five BART variants on two illustrative functions.

## 3 MODEL

## 3.1 Tree-specific rotations

Fix $p \geq 2$ , a finite number of trees $T \geq 1$ , and the training design $X = ( x _ { 1 } , \ldots , x _ { n } ) ^ { \mathsf { T } }$ , with $x _ { i } \in \mathbb { R } ^ { p }$ We model

$$
Y _ { i } = f ( x _ { i } ) + \varepsilon _ { i } , \quad \varepsilon _ { i } \overset { \mathrm { i i d } } { \sim } N ( 0 , \sigma ^ { 2 } ) .
$$

![](images/b4606a0679721c075c565c0d04cd79feb6a81997f112049ed25d131009f419a3.jpg)  
Figure 2: Two-tree illustrations of BART (left) and RoBART (right). Within each half, the two small panels show individual tree contributions, and the larger panel shows their sum. Numbers are illustrative predictions in the original predictor coordinates. Blue and orange identify the two tree partitions.

For tree t, let $Q _ { t }$ be a rotation matrix in $\mathrm { S O } ( p ) =$ $\{ Q \in \mathbb { R } ^ { p \times p } : Q ^ { \mathsf { T } } Q = I _ { p } ,$ det Q = 1}. Let $\mathcal { T } _ { t }$ denote its rooted ordered binary topology, with internal nodes $\mathcal { T } ( \mathcal { T } _ { t } )$ and terminal nodes $\mathcal { L } ( \mathcal { T } _ { t } )$ . An internal node η splits on coordinate $v _ { \eta } \in \{ 1 , \dotsc , p \}$ at cutpoint index $c _ { \eta } ;$ terminal node ℓ has mean $\mu _ { t \ell } .$ Write $V _ { t } = ( v _ { \eta } ) , \dot { C } _ { t } = ( c _ { \eta } )$ , and $M _ { t } = \left( \mu _ { t \ell } \right)$ for these arrays. The regression mean is

$$
f ( x ) = \sum _ { t = 1 } ^ { T } g _ { t } ( Q _ { t } x ; \mathcal { T } _ { t } , V _ { t } , C _ { t } , M _ { t } ) , \quad Q _ { t } \in \mathrm { S O } ( p ) .
$$

Here $g _ { t }$ is the regression tree specified by $( T _ { t } , V _ { t } , C _ { t } , M _ { t } )$ . Let $\xi _ { t , j , k } ( Q _ { t } )$ be the candidate cutpoint with index k for rotated coordinate $j ,$ defined in (1). At an internal node η, an input is sent to the left child exactly when

$$
( Q _ { t } x ) _ { v _ { \eta } } < \xi _ { t , v _ { \eta } , c _ { \eta } } ( Q _ { t } ) ,
$$

and to the right child otherwise. The output $g _ { t }$ is the mean assigned to the unique terminal node reached by these rules. Thus the split is axis-aligned in the rotated coordinates. Let $e _ { j }$ be the jth standard basis vector. In the original predictor space, the split boundary has normal vector $Q _ { t } ^ { \mathsf { T } } e _ { v _ { \eta } } .$ the transpose of row $v _ { \eta }$ of $Q _ { t }$ . Splits on the same rotated coordinate have parallel boundaries.

$$
\begin{array} { r l } & { \mathrm { W r i t e } \quad z _ { t } ( x ) \qquad = \qquad Q _ { t } x \qquad \mathrm { a n d } \quad f _ { t } ( x ) \qquad = } \\ & { g _ { t } ( z _ { t } ( x ) ; \mathcal { T } _ { t } , V _ { t } , C _ { t } , M _ { t } ) , \mathrm { s o ~ t h a t } \ f ( x ) = \sum _ { t = 1 } ^ { T } f _ { t } ( x ) . } \end{array}
$$

Figure 2 illustrates $T = 2 \colon$ standard BART uses $Q _ { 1 } = Q _ { 2 } = I _ { p } ,$ , whereas RoBART may use diferent rotations for the two trees.

The coordinates used by tree t form the set $\boldsymbol { S } _ { t } =$ $\{ v _ { \eta } : \eta \in \mathcal { T } ( \mathcal { T } _ { t } ) \}$ }. The corresponding rows $Q _ { t , S _ { t } }$ <sub>,·</sub> are orthonormal.

## 3.2 Cutpoint grids and prior

For each rotated coordinate $j ,$ fix a positive integer $b _ { j }$ as the number of candidate cutpoints throughout sampling. Define the training extrema by

$$
\underline { { z } } _ { t j } ( Q _ { t } ) = \operatorname* { m i n } _ { i \leq n } ( Q _ { t } x _ { i } ) _ { j } , \quad \overline { { z } } _ { t j } ( Q _ { t } ) = \operatorname* { m a x } _ { i \leq n } ( Q _ { t } x _ { i } ) _ { j } .
$$

For $k = 0 , \ldots , b _ { j } - 1$ , set

$$
\begin{array} { c } { \omega _ { j k } = \frac { k + 1 } { b _ { j } + 1 } , } \\ { \xi _ { t , j , k } ( Q _ { t } ) = ( 1 - \omega _ { j k } ) \underline { { z } } _ { t j } ( Q _ { t } ) + \omega _ { j k } \overline { { z } } _ { t j } ( Q _ { t } ) . } \end{array}\tag{1}
$$

Write $\Xi _ { t } ( Q _ { t } ) = \{ \xi _ { t , j , k } ( Q _ { t } ) \}$ These cutpoints are equally spaced between the training minimum and maximum. A coordinate with zero range has no available split. The grid is determined by $Q _ { t }$ and the training predictors and has no separate prior.

We generate each tree recursively. Write dep(η) for node depth, with root depth zero, and let $\rho _ { \ell } \in [ 0 , 1 ]$ be the split-attempt probability at depth ℓ. For a node η and rotation $Q .$ , let $C _ { \mathrm { a n c } }$ denote its ancestor cutpoint indices. Ancestor splits on coordinate $j$ determine an interval; $\mathscr { A } _ { \eta } ( j ; Q , C _ { \mathrm { a n c } } )$ contains the grid indices with cutpoints strictly inside it and is empty for zero-range coordinates. We impose no minimum number of observations per leaf. The coordinates with an available cutpoint are

$$
\mathcal G _ { \eta } ( Q , C _ { \mathrm { a n c } } ) = \{ j : \mathcal A _ { \eta } ( j ; Q , C _ { \mathrm { a n c } } ) \neq \emptyset \} .
$$

At depth $\ell ,$ a node remains terminal with probability $1 - \rho _ { \ell }$ . Otherwise, choose a coordinate uniformly from $\{ 1 , \ldots , p \}$ . If it has no admissible cutpoint, the node remains terminal. If it does, choose a cutpoint index uniformly from its admissible set and create two children. The probability that node η remains terminal is therefore $1 - \rho _ { \mathrm { d e p } ( \eta ) } | \mathcal { G } _ { \eta } | / p$ . For a valid state ${ \sf s } = ( Q , C )$ , the tree prior is

$$
\begin{array} { r } { \pi _ { \mathrm { d i s c } } ( \mathcal { T } , V , C \mid Q ) = \prod _ { \eta \in \mathbb { Z } ( \mathcal { T } ) } \frac { \rho _ { \mathrm { d e p } ( \eta ) } } { p a _ { \eta } ( \mathbf { s } ) } } \\ { \times \prod _ { \ell \in \mathcal { L } ( \mathcal { T } ) } \left\{ 1 - \rho _ { \mathrm { d e p } ( \ell ) } \frac { g _ { \ell } ( \mathbf { s } ) } { p } \right\} , } \end{array}\tag{2}
$$

where $\begin{array} { r c l } { a _ { \eta } ( \mathsf { s } ) } & { = } & { | \mathcal { A } _ { \eta } ( v _ { \eta } ; Q , C _ { \mathrm { a n c } } ) | } \end{array}$ and $\begin{array} { r l } { g _ { \eta } ( \mathsf { s } ) } & { { } = } \end{array}$ $| \mathcal { G } _ { \eta } ( Q , C _ { \mathrm { a n c } } ) | ;$ ; invalid states have prior mass zero. Normalization follows from the recursive construction: a path cannot reuse a coordinate’s cutpoint index and therefore has depth at most $\textstyle \sum _ { j } b _ { j }$ . Draw the rotation matrices independently from normalized Haar measure on $\mathrm { S O } ( p )$ and generate the trees independently conditional on these matrices. Conditional on the forest structure, all leaf means are independent $N ( 0 , \tau ^ { 2 } )$ variables with fixed $0 < \tau ^ { 2 } < \infty$ Independently, $\sigma ^ { 2 } \sim \mathrm { I G } ( a _ { \sigma } , b _ { \sigma } )$ , with $a _ { \sigma } , b _ { \sigma } > 0 ,$ , using density proportional to $( \sigma ^ { 2 } ) ^ { - a _ { \sigma } - 1 } \exp \{ - b _ { \sigma } / \sigma ^ { 2 } \}$

The joint update assumes $0 ~ < ~ \rho _ { \ell } ~ < ~ 1$ . For the contraction analysis, we use $\rho _ { \ell } = \nu ^ { \ell + 1 }$ with fixed $0 < \nu < 1$

## 3.3 Conditional posterior

When updating tree t, form the partial residuals

$$
r _ { i } ^ { ( t ) } = Y _ { i } - \sum _ { s \neq t } g _ { s } ( Q _ { s } x _ { i } ) .
$$

Suppress t and condition on $( \mathcal { T } , V , \sigma , \tau )$ , the training design, and the complete states of the other trees. Integrate out the current tree’s leaf means. For a terminal node $B _ { ; }$ , let $I _ { B } ( \mathsf { s } )$ be the indices of observations assigned to $B .$ , and set

$$
b _ { B } = \frac { | I _ { B } ( \mathsf { s } ) | } { \sigma ^ { 2 } } , \qquad m _ { B } = \sum _ { i \in I _ { B } ( \mathsf { s } ) } \frac { r _ { i } ^ { ( t ) } } { \sigma ^ { 2 } } .
$$

After integrating out the leaf means, the log marginal likelihood is $\ell _ { H } ( \mathsf { s } )$ up to a constant independent of

s, where

$$
\begin{array} { l } { { \ell _ { B } ( { \mathsf s } ) = - \frac { 1 } { 2 } \log ( 1 + { \tau } ^ { 2 } b _ { B } ) + \frac { { \tau } ^ { 2 } m _ { B } ^ { 2 } } { 2 ( 1 + { \tau } ^ { 2 } b _ { B } ) } , } } \\ { { \ell _ { H } ( { \mathsf s } ) = \sum _ { B \in \mathcal { L } ( \mathcal { T } ) } \ell _ { B } ( { \mathsf s } ) . } } \end{array}
$$

An empty leaf contributes $\ell _ { B } = 0$ . For fixed $( \mathcal { T } , V )$ the space of cutpoint arrays is

$$
\mathcal { C } _ { \operatorname* { m a x } } ( \mathcal { T } , V ) = \prod _ { \eta \in \mathcal { I } ( \mathcal { T } ) } \{ 0 , \dots , b _ { v _ { \eta } } - 1 \} .
$$

With respect to normalized Haar measure on $\mathrm { S O } ( p )$ and counting measure on $\mathcal { C } _ { \operatorname* { m a x } } ( \mathcal { T } , V )$ , the conditional posterior of $( Q , C )$ is

$$
\pi _ { H } ( \mathsf { s } \mid \mathrm { r e s t } ) \propto \exp \{ \ell _ { H } ( \mathsf { s } ) \} \pi _ { \mathrm { d i s c } } ( \mathcal { T } , V , C \mid Q ) ,\tag{3}
$$

with density zero for invalid arrays. Here rest denotes the conditioning quantities listed above. The prior factor is (2) evaluated at the fixed $( \mathcal { T } , V )$ . Normalize the product jointly over $Q$ and $C ,$ without separately normalizing the tree prior over C for each $Q .$

## 4 POSTERIOR COMPUTATION

## 4.1 Selecting coordinate pairs

Write $[ p ] = \{ 1 , \dotsc , p \}$ and let $\mathcal { P } = \{ ( a , b ) \in [ p ] ^ { 2 }$ : $a \neq b \}$ . Let

$$
\begin{array} { r } { S ( \mathcal { T } , V ) = \{ v _ { \eta } : \eta \in \mathbb { Z } ( \mathcal { T } ) \} , \qquad s = | S ( \mathcal { T } , V ) | , } \end{array}
$$

be the set of rotated-coordinate indices used by the current tree. This is $S _ { t }$ with the tree index omitted, not a subset of the original predictors. The set remains fixed during the joint update, although the corresponding rows of $Q$ may change. Conditional on $( \mathcal { T } , V )$ , draw ordered pairs $P _ { k } = ( a _ { k } , b _ { k } ) \in \mathcal { P }$ independently with replacement from $q _ { P } ( \cdot \mid \tau , V )$ Define

$$
\begin{array} { r } { \mathcal { P } _ { \mathrm { i n } } = \{ ( a , b ) \in \mathcal { P } : a , b \in \mathcal { S } \} , } \\ { \mathcal { P } _ { \mathrm { o u t } } = ( \mathcal { S } \times \mathcal { S } ^ { c } ) \cup ( \mathcal { S } ^ { c } \times \mathcal { S } ) , } \end{array}
$$

where $S ^ { c } = [ p ] \mid S .$ . Fix $\rho _ { \mathrm { o u t } } \in ( 0 , 1 )$ . For $2 \leq s < p _ { ; }$

$$
q _ { P } ( P \mid \mathcal { T } , V ) = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 - \rho _ { \mathrm { o u t } } } { s ( s - 1 ) } , } & { P \in \mathcal { P } _ { \mathrm { i n } } , } \\ { \displaystyle \frac { \rho _ { \mathrm { o u t } } } { 2 s ( p - s ) } , } & { P \in \mathcal { P } _ { \mathrm { o u t } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{4}
$$

If $s \in \{ 0 , p \}$ , use the uniform distribution on all $p ( p - 1 )$ ordered distinct pairs. If $s = 1$ , use the uniform distribution on the $2 ( p - 1 )$ pairs with one coordinate in S and the other outside it. Rotating two used rows changes the basis but preserves their joint span. Rotating a used row with an unused row can change the span of the used rows, without changing V or S. Because the pairs are independent draws from the same distribution, reversing their order leaves their joint probability unchanged.

## 4.2 Rotation proposal

For an ordered pair $P = ( a , b )$ , let $G _ { P } ( \theta )$ equal $I _ { p }$ except on the rows and columns indexed by $( a , b )$ ， in that order, where the block is

$$
\left( \begin{array} { c c } { { \cos \theta } } & { { - \sin \theta } } \\ { { \sin \theta } } & { { \cos \theta } } \end{array} \right) .
$$

Thus $G _ { P } ( \theta ) \in \mathrm { S O } ( p )$ and $G _ { P } ( \theta ) ^ { - 1 } = G _ { P } ( - \theta )$ . Fix $K _ { \mathrm { r o t } } ~ \in ~ \{ 1 , 2 , . . . \} , ~ \gamma ~ > ~ 0$ , and $0 ~ < ~ \theta _ { \mathrm { { m a x } } } ~ < ~ \pi$ Starting from $Q _ { 0 } \ = \ Q ,$ , independently draw, for $k = 1 , \ldots , K _ { \mathrm { r o t } }$

$$
P _ { k } \sim q _ { P } ( \cdot \mid T , V ) , \ \theta _ { k } \stackrel { \mathrm { i i d } } { \sim } \mathrm { T N } _ { [ - \theta _ { \operatorname* { m a x } } , \theta _ { \operatorname* { m a x } } ] } ( 0 , \gamma ^ { 2 } ) ,\tag{5}
$$

where TN denotes the $N ( 0 , \gamma ^ { 2 } )$ distribution truncated to the indicated interval. Set $\begin{array} { r l } { Q _ { k } } & { { } = } \end{array}$ $G _ { P _ { k } } ( \theta _ { k } ) Q _ { k - 1 }$ . The proposed rotation is

$$
Q ^ { \prime } = G _ { P _ { K _ { \mathrm { r o t } } } } ( \theta _ { K _ { \mathrm { r o t } } } ) \cdot \cdot \cdot G _ { P _ { 1 } } ( \theta _ { 1 } ) Q \in \mathrm { S O } ( p ) .\tag{6}
$$

Changing the order of Givens rotations can change their product. To recover Q from $Q ^ { \prime }$ , apply the same ordered pairs in reverse order with negated angles.

The centered truncated-normal law in (5) has an even density q<sub>θ</sub>, so the forward and reverse angle densities agree.

## 4.3 Cutpoint proposal

After proposing a rotation, we use the current split to choose a reference location on the new cutpoint grid. We propose a random perturbation around this location rather than fixing the cutpoint there.

Let $\Xi _ { 0 } = \Xi ( Q )$ and $\Xi _ { 1 } = \Xi ( Q ^ { \prime } )$ denote the current and proposed cutpoint grids. Define

$$
\begin{array} { r l } & { \mathcal { I } _ { \mathrm { t o u c h } } = \bigcup _ { k = 1 } ^ { K _ { \mathrm { r o t } } } \{ a _ { k } , b _ { k } \} , } \\ & { \mathcal { N } _ { \mathrm { t o u c h } } = \{ \eta \in \mathcal { I } ( \mathcal { T } ) : v _ { \eta } \in \mathcal { I } _ { \mathrm { t o u c h } } \} . } \end{array}
$$

The sets $\mathcal { I } _ { \mathrm { t o u c h } }$ and $\mathcal { N } _ { \mathrm { t o u c h } }$ contain the coordinates selected for rotation and the internal nodes that split on them. Only these grid columns need to be recomputed; the other rows of $Q$ and rotated predictor values are unchanged.

For each afected node $\eta ,$ let $\mathcal { T } _ { \eta } ^ { 0 }$ be the indices of observations reaching it under the current tree $( Q , C )$ and let $n _ { \eta , L } ^ { 0 }$ be the number sent to its left child. These quantities are computed once and kept fixed while proposing $C ^ { \prime }$

We visit internal nodes in a fixed order, starting at the root and processing each parent before its children. At each afected node, the cutpoints already assigned to its ancestors determine the admissible index interval $\left[ \underline { { c } } _ { \eta } , \overline { { c } } _ { \eta } \right]$ . Other nodes retain their indices. If this interval is empty, the proposal fails (∂) and $( Q , C )$ remains unchanged.

For $0 < n _ { \eta , L } ^ { 0 } < | \mathcal { Z } _ { \eta } ^ { 0 } |$ , let $z _ { \eta , ( k ) } ^ { \prime }$ be the kth order statistic of $\{ ( Q ^ { \prime } x _ { i } ) _ { v _ { \eta } } : i \in \mathcal { T } _ { \eta } ^ { 0 } \}$ . Set

$$
t _ { \eta } ^ { \star } = \frac { z _ { \eta , ( n _ { \eta , L } ^ { 0 } ) } ^ { \prime } + z _ { \eta , ( n _ { \eta , L } ^ { 0 } + 1 ) } ^ { \prime } } { 2 } .\tag{7}
$$

Let $c _ { \eta } ^ { \star }$ index the admissible cutpoint on $\Xi _ { 1 }$ closest to $t _ { \eta } ^ { \star } ,$ breaking ties by the smallest index. This defines the proposal center; it does not guarantee that the observation partition is preserved. If the node contains no observations, or all its observations $_ \mathrm { g o }$ to the same child under the current tree, the center in (7) is undefined. Keep its current cutpoint index if it is admissible under the proposed ancestors; otherwise, return ∂ and retain the original state $( Q , C )$

Fix $0 < \rho _ { \mathrm { c u t } } < \infty$ . With $W _ { \eta } = \bar { c } _ { \eta } - \underline { { { c } } } _ { \eta } + 1$ and $\kappa _ { \eta } = \operatorname* { m a x } \{ 1 , \rho _ { \mathrm { c u t } } W _ { \eta } \}$ , define

$$
Z _ { \eta } = \sum _ { u = \underline { { c } } _ { \eta } } ^ { \overline { { c } } _ { \eta } } \exp \{ - ( u - c _ { \eta } ^ { \star } ) ^ { 2 } / ( 2 \kappa _ { \eta } ^ { 2 } ) \} .
$$

For an afected node with $0 < n _ { \eta , L } ^ { 0 } < | \mathcal { T } _ { \eta } ^ { 0 } |$ , draw the index from

$$
\begin{array} { r } { q _ { \eta } ( c ) = Z _ { \eta } ^ { - 1 } \exp \bigl \{ - \frac { ( c - c _ { \eta } ^ { \star } ) ^ { 2 } } { 2 \kappa _ { \eta } ^ { 2 } } \bigr \} \mathbf { 1 } \{ \underline { { c } } _ { \eta } \leq c \leq \overline { { c } } _ { \eta } \} . } \end{array}
$$

For a generated array $C ^ { \prime }$ , let $q _ { C } ^ { F }$ be the product of the conditional proposal probabilities at successive nodes. A node whose index is fixed by the rule above contributes a factor of one. Failed proposals leave $( Q , C )$ unchanged without resampling, so $q _ { C } ^ { F }$ is not conditioned on success.

To compute $q _ { C } ^ { R }$ , evaluate the probability of proposing C from $( Q ^ { \prime } , \bar { C } ^ { \prime } )$ with proposed rotation Q. Recompute and fix the observation sets and left-child counts under $( Q ^ { \prime } , C ^ { \prime } )$ . In the same node order, use $Q x _ { i } , \Xi _ { 0 }$ and ancestor indices from C to evaluate the probability of each $c _ { \eta }$ instead of drawing it. Their product is $q _ { C } ^ { \check { R } } .$ . If the rule retains $c _ { \eta } ^ { \prime } ,$ only $c _ { \eta } = c _ { \eta } ^ { \prime }$ has positive probability. Any zero factor gives $q _ { C } ^ { R } = 0$ and rejects the joint proposal.

## 4.4 Acceptance probability and reversibility

Write ${ \sf s } = ( Q , C )$ and ${ \mathsf { \pmb { \mathsf { s } } } } ^ { \prime } = ( Q ^ { \prime } , C ^ { \prime } )$ , and define

$$
\Delta _ { \mathrm { d i s c } } = \log \frac { \pi _ { \mathrm { d i s c } } ( { \mathcal T } , V , C ^ { \prime } \mid Q ^ { \prime } ) } { \pi _ { \mathrm { d i s c } } ( { \mathcal T } , V , C \mid Q ) } .
$$

The pair probabilities and angle densities cancel between the proposed Givens sequence and its reverse. The grid is determined by the rotation, so the remaining proposal ratio is $q _ { C } ^ { R } / q _ { C } ^ { F }$ . For $q _ { C } ^ { F } > 0$ and $q _ { C } ^ { R } > 0$ , the log acceptance ratio is

$$
\begin{array} { r } { \mathrm { ~  ~ \xi ~ } _ { \mathrm { { { J } } } } A = \ell _ { H } ( \mathsf { s } ^ { \prime } ) - \ell _ { H } ( \mathsf { s } ) + \Delta _ { \mathrm { d i s c } } + \log q _ { C } ^ { R } - \log q _ { C } ^ { F } . } \end{array}\tag{8}
$$

The split-attempt and coordinate-selection factors cancel because (T, V) is fixed. Thus $\Delta _ { \mathrm { d i s c } }$ contains the changes in terminal-node probabilities and cutpoint prior probabilities. Accept with probability 1 ∧ exp(log A). A zero proposed target density, a zero reverse probability, or a failed cutpoint proposal leaves (Q, C) unchanged.

Proposition 1 (Reversibility of the joint update). Fix the conditioning quantities in Section 3 and a feasible (T , V ) for which (3) has a positive finite normalizing constant. Assume that the cutpoint grid is evaluated exactly and that its map and tie rules are deterministic and measurable. Under (4) and (5), Algorithm 1, using the same sequential cutpoint rule in both directions, including deterministic holds and failure, is reversible with respect to (3).

Supplement A gives the cutpoint recursion, proves reversibility and invariance of the completed tree update, and analyzes the cost of one joint proposal. Reversibility establishes posterior invariance, not a convergence-rate bound.

## 4.5 Completing the tree update

At the retained rotation and grid, hold $r ^ { ( t ) } , \sigma , \tau$ , and the other trees fixed and apply birth–death kernels preserving the leaf-collapsed conditional posterior. Birth proposals may sample uniformly among coordinates with available cutpoints, but their actual forward and reverse probabilities and the prior (2) must enter the acceptance ratio. Without using old leaf means, complete these updates by drawing all leaf means from their Gaussian full conditional and restoring the tree contribution, even after rotation rejection or failure. Supplement A proves joint conditional invariance of this completed update; a sweep through all trees followed by valid updates of the remaining parameters preserves the full posterior.

Algorithm 1 Joint rotation and cutpoint update   
for tree t   
1: Remove tree t from the fitted sum and form $r ^ { ( t ) }$   
2: Save ${ \mathfrak { s } } = ( Q , C ) , { \Xi } _ { 0 }$ , the rotated design, obser  
vation assignments, $\ell _ { H } ( \mathsf { s } )$ , and log $\pi _ { \mathrm { d i s c } } ( \mathsf { \pmb { s } } )$   
3: Initialize the retained state to all saved quantities   
4: Draw $P _ { \mathrm { 1 : } K _ { \mathrm { r o t } } }$ and $\theta _ { 1 : K _ { \mathrm { r o t } } } ;$ construct $Q ^ { \prime }$ by (6)   
5: Recompute grid $\Xi _ { 1 }$ for coordinates in J<sub>touch</sub>   
6: Propose $C ^ { \prime }$ starting at the root, processing each   
node before its children; accumulate log $q _ { C } ^ { F }$   
7: if the cutpoint proposal does not return ∂ and   
the candidate is valid then   
8: Compute proposed observation assignments,   
$\ell _ { H } ( \mathsf { \pmb { s } } ^ { \prime } )$ , and log $\pi _ { \mathrm { d i s c } } ( \mathsf { \pmb { s } } ^ { \prime } )$   
9: Evaluate log $q _ { C } ^ { R }$ by the same recursion from   
$\mathsf { \mathsf { S } } ^ { \prime }$ to s, including the rules that retain an index   
10: if $q _ { C } ^ { R } > 0$ and the proposed target density is   
positive then   
11: With probability 1 ∧ exp(log A) from (8),   
replace the retained state by the proposed state   
and its associated quantities   
12: end if   
13: end if

## 5 POSTERIOR CONTRACTION

## 5.1 Fixed-design model and additive truth

For asymptotics, make the sample-size dependence explicit:

$$
Y _ { i , n } = f _ { 0 } ( x _ { i , n } ) + \varepsilon _ { i , n } , \quad \varepsilon _ { i , n } \overset { \mathrm { i i d } } { \sim } N ( 0 , \sigma _ { 0 } ^ { 2 } ) ,\tag{9}
$$

where $\sigma _ { 0 } > 0$ and $i = 1 , \ldots , n$ . Define

$$
\begin{array} { r } { \left. g \right. _ { n } ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } g ( x _ { i , n } ) ^ { 2 } , } \end{array}
$$

$$
\boldsymbol { Y } ^ { ( n ) } = ( Y _ { 1 , n } , \ldots , Y _ { n , n } ) ^ { \boldsymbol { \mathsf { T } } } .
$$

Let $\mathbb { E } _ { 0 }$ denote expectation under (9). Let $\Pi _ { n } ( \cdot \mid$ $Y ^ { ( n ) } )$ denote the posterior induced by the prior below, with $b _ { n }$ candidate cutpoints on each rotated coordinate. The sequence $b _ { n } \geq 1$ consists of deterministic integers.

Assumption 1 (Fixed design and additive truth). The ambient dimension $p \geq 2$ , number of trees $T ,$ and number of components R are fixed, with $1 \leq R \leq$ $T .$ The fixed designs are contained in one compact set $\mathcal { X } \subset \mathbb { R } ^ { p }$ satisfying

$$
x _ { i , n } \in { \mathcal { X } } , \qquad \operatorname* { s u p } _ { x \in { \mathcal { X } } } \| x \| _ { 2 } \leq M _ { X } < \infty .
$$

The truth has at least one representation on X of the form

$$
f _ { 0 } ( x ) = \sum _ { r = 1 } ^ { R } h _ { 0 r } \{ \pi _ { S _ { r } } ( Q _ { 0 r } x ) \} , \qquad R \leq T ,
$$

where $Q _ { 0 r } \in \mathrm { S O } ( p )$ and $S _ { r } = \{ s _ { r 1 } < \cdot \cdot \cdot < s _ { r d _ { r } } \} \subseteq$ $\{ 1 , \ldots , p \}$ , with $1 \ \leq \ d _ { r } \ \leq \ p$ . The projection is $\pi _ { S _ { r } } ( u ) = ( u _ { s _ { r 1 } } , \ldots , u _ { s _ { r d _ { r } } } ) ^ { \mathsf { T } }$ . For fixed $\lambda _ { r } > 0$ and $\alpha _ { r k } \in ( 0 , 1 ]$ , each $h _ { 0 r } : [ - M _ { X } , M _ { X } ] ^ { d _ { r } } $ R satisfies $\| h _ { 0 r } \| _ { \infty } \leq C _ { h } < \infty$ and, for all $u , v$ in its domain,

$$
| h _ { 0 r } ( u ) - h _ { 0 r } ( v ) | \leq \lambda _ { r } \sum _ { k = 1 } ^ { d _ { r } } | u _ { k } - v _ { k } | ^ { \alpha _ { r k } } .
$$

The dimensions $d _ { r } ,$ , radii $\lambda _ { r }$ , true rotation matrices and functions, and smoothness vectors are fixed as $n \to \infty$

Here S selects coordinates of $Q _ { 0 r } x .$ , and $\alpha _ { r k }$ describes smoothness in coordinate $s _ { r k }$ . Since $Q _ { 0 r }$ may difer across components, the same index may represent diferent directions. The dimension $d _ { r }$ refers to the chosen representation; minimality and identifiability are not assumed.

Let

$$
\bar { \alpha } _ { r } ^ { - 1 } = d _ { r } ^ { - 1 } \sum _ { k = 1 } ^ { d _ { r } } { \alpha _ { r k } ^ { - 1 } } .
$$

Constants below may depend on the fixed truth parameters, $p , R , T , \ M _ { X } , C _ { h } , r _ { \operatorname* { m i n } } ,$ and fixed prior hyperparameters, but not on n or on a rotation matrix in a stated neighborhood.

## 5.2 Prior and grid conditions

Use the prior in Section 3 with $b _ { j } = b _ { n }$ for all $j$ and $\rho _ { \ell } = \nu ^ { \ell + 1 }$ for fixed $0 < \nu < 1$ . Keep all remaining prior hyperparameters fixed.

For $j \in S _ { r }$ , define the range of rotated coordinate $j$ by

$$
R _ { n , r j } ( Q ) = \operatorname* { m a x } _ { i \leq n } e _ { j } ^ { \mathsf { T } } Q x _ { i , n } - \operatorname* { m i n } _ { i \leq n } e _ { j } ^ { \mathsf { T } } Q x _ { i , n } .
$$

For component $r ,$ define

$$
\widetilde { K } _ { n , r } = \left\{ \frac { n ( \lambda _ { r } d _ { r } ) ^ { 2 } } { \log n } \right\} ^ { d _ { r } / ( 2 \bar { \alpha } _ { r } + d _ { r } ) } ,
$$

and choose a leaf count $K _ { n , r } = 2 ^ { L _ { n , r } }$ with $L _ { n , r } \in \mathbb { N }$ and $K _ { n , r } \asymp \widetilde { K } _ { n , r }$

Assumption 2 (Active-range and grid conditions). There exist $r _ { \operatorname* { m i n } } > 0 , C _ { b } < \infty$ , and $n _ { 0 }$ such that

$$
\begin{array} { r l } { \mathrm { ( D 1 ) } } & { \underset { n \geq n _ { 0 } } { \operatorname* { i n f } } \underset { r \leq R } { \operatorname* { m i n } } \underset { j \in S _ { r } } { \operatorname* { m i n } } R _ { n , r j } ( Q _ { 0 r } ) \geq r _ { \operatorname* { m i n } } , } \\ { \mathrm { ( D 2 ) } } & { b _ { n } + 1 \geq \underset { r \leq R } { \operatorname* { m a x } } K _ { n , r } , \quad \log b _ { n } \leq C _ { b } \log n . } \end{array}\tag{10}
$$

Here D2 is required for all $n \geq n _ { 0 }$ , after enlarging $n _ { 0 }$ if necessary.

Condition D1 requires a positive lower bound on the ranges of the coordinates used by the truth; it does not require full rank or a space-filling design. The choice $b _ { n } = n$ satisfies D2 for all suficiently large n, since $d _ { r } / ( 2 \bar { \alpha } _ { r } + d _ { r } ) < 1$ , and requires no knowledge of the smoothness or coordinate sets.

## 5.3 Contraction rate

Define

$$
\begin{array} { l } { \displaystyle \varepsilon _ { n , r } = ( \lambda _ { r } d _ { r } ) ^ { { d _ { r } } / ( 2 \bar { \alpha } _ { r } + { d _ { r } } ) } \left( \frac { \log { n } } { n } \right) ^ { \bar { \alpha } _ { r } / ( 2 \bar { \alpha } _ { r } + { d _ { r } } ) } , } \\ { \displaystyle \varepsilon _ { n } = \sum _ { r = 1 } ^ { R } \varepsilon _ { n , r } . } \end{array}
$$

Theorem 1 (Posterior contraction). Under the fixed-design model (9), Assumptions 1 and 2, and the preceding prior specification, there exists a sufficiently large constant $M < \infty$ , independent of $n ,$ such that

$$
\mathbb { E } _ { 0 } \Pi _ { n } \big [ \| f - f _ { 0 } \| _ { n } + | \sigma - \sigma _ { 0 } | > M \varepsilon _ { n } ~ | ~ Y ^ { ( n ) } \big ] \to 0 .
$$

For fixed $R ,$ , the slowest component determines the order of $\varepsilon _ { n }$ . Supplement B uses one rotated tree per component to verify the local prior-mass, entropy, and sieve-tail conditions of Jeong (2025). The result concerns $f$ and $\sigma ,$ not the component representations.

## 5.4 Comparison with axis-aligned BART

The preceding upper bound does not determine whether axis-aligned BART can attain the same rate. Let $\Pi _ { n } ^ { \mathrm { B } }$ denote the posterior under the preceding prior specification with $Q _ { t } = I _ { p }$ deterministically for every tree. All remaining specifications, including the hyperparameters, empirical-range grid rule, and budget $b _ { n }$ , are unchanged.

Theorem 2 (Posterior contraction lower bound for BART). Fix integers $p \geq 2$ and $T \geq 1$ , and constants $0 ~ < ~ \alpha ~ < ~ 1$ and $\lambda , C _ { h } , \sigma _ { 0 } \ > \ 0$ Let $\mathcal { X } \ : = \ : [ 0 , 1 ] ^ { p }$ For integers m $\geq 1$ and $n = 2 ^ { m p }$ , let the design enumerate

$$
\mathcal X _ { n } = \left\{ ( i + 1 / 2 ) 2 ^ { - m } : i = 0 , \ldots , 2 ^ { m } - 1 \right\} ^ { p } .
$$

Assume (10) with $R = d _ { 1 } = 1 , \lambda _ { 1 } = \lambda$ , and $\alpha _ { 1 1 } = \alpha$ There exist $Q _ { 0 } \in \mathrm { S O } ( p )$ and $h _ { 0 } : [ - \sqrt { p } , \sqrt { p } ] \to \mathbb { R }$ both independent of $n ,$ such that $\| h _ { 0 } \| _ { \infty } \leq C _ { h }$ and $\left| h _ { 0 } ( u ) - h _ { 0 } ( v ) \right| \le \lambda \left| u - v \right| ^ { \alpha }$ on this interval, and the truth $f _ { 0 } ( x ) = h _ { 0 } \{ \pi _ { \{ 1 \} } ( Q _ { 0 } x ) \}$ on $[ 0 , 1 ] ^ { p }$ has the following properties. Set

$$
\varepsilon _ { n } = \lambda ^ { 1 / ( 2 \alpha + 1 ) } \left( \frac { \log n } { n } \right) ^ { \alpha / ( 2 \alpha + 1 ) } ,
$$

$$
\varepsilon _ { n } ^ { \mathrm { B } } = \left( { \frac { \log n } { n } } \right) ^ { \alpha / ( 2 \alpha + p ) } .
$$

For some fixed $c > 0$ and a suficiently large fixed $M _ { 0 } < \infty$

$$
\begin{array} { r } { \mathbb { E } _ { 0 } \Pi _ { n } ^ { \mathrm { B } } \big [ \| f - f _ { 0 } \| _ { n } \leq c \varepsilon _ { n } ^ { \mathrm { B } } \mid Y ^ { ( n ) } \big ]  0 , } \end{array}
$$

$$
\mathbb { E } _ { 0 } \Pi _ { n } \big [ \| f - f _ { 0 } \| _ { n } + | \sigma - \sigma _ { 0 } | > M _ { 0 } \varepsilon _ { n } \ | \ Y ^ { ( n ) } \big ] \to 0 .
$$

All limits are taken as $m  \infty$ along $n = 2 ^ { m p }$

Since $\varepsilon _ { n } = o ( \varepsilon _ { n } ^ { \mathrm { B } } )$ , the BART posterior assigns vanishing mass to the ball of radius $M \varepsilon _ { n }$ for every fixed $M > 0$ at this truth. Thus the stated axis-aligned prior need not adapt to a one-dimensional rotated representation. The proof combines an approximation lower bound for the entire forest with posterior control of its total leaf count. Supplement C gives the construction and proof; the RoBART bound follows from Theorem 1.

## 6 SIMULATION STUDY

## 6.1 Benchmark design

We compare five methods on four regression functions of d nonorthogonal linear combinations of p observed predictors. We use $( p , d ) \ = \ ( 1 0 , 5 )$ $( 2 0 , 5 ) , ( 2 0 , 1 0 ) , ( 3 0 , 5 )$ , and (30, 10), training sizes $n \in \{ 1 , 5 0 0 , 5 , 0 0 0 \}$ , and $N = 2 { , } 0 0 0$ test inputs.

We generate $X _ { i } \sim N _ { p } ( 0 , I _ { p } )$ and $Y _ { i } = f _ { s } ( X _ { i } ) + \varepsilon _ { i } .$ with independent $N ( 0 , 0 . 5 ^ { 2 } )$ errors. For each raw function $r _ { s }$ below, set $f _ { s } = ( r _ { s } - \mu _ { s } ) / \sqrt { v _ { s } }$ , where $\mu _ { s }$ and $v _ { s }$ are its sample mean and variance on 100,000 independent calibration inputs. These constants remain fixed across sample sizes and replications.

Let $U \ = \ [ u _ { 1 } , \ldots , u _ { d } ] \ \in \ \mathbb { R } ^ { p \times d }$ have orthonormal columns, with $u _ { 1 } ~ \stackrel { \cdot } { = } ~ p ^ { - 1 / 2 } \mathbf { 1 } _ { p }$ The remaining columns are obtained by orthogonalizing Gaussian vectors against $u _ { 1 }$ and each other. All four scenarios use $z = A x$ , where $A = ( 0 . 2 5 I _ { d } + 0 . 7 5 \mathbf { 1 } _ { d } \mathbf { 1 } _ { d } ^ { \top } ) ^ { 1 / 2 } U ^ { \top }$ and the square root is symmetric positive definite. Geometrically, the transformation turns projected spherical Gaussian contours into ellipsoids elongated along $\mathbf { 1 } _ { d } .$ . The coordinates of AX have unit variance and pairwise correlation 0.75. Write $t = \Phi ( z )$ , with the standard normal distribution function applied coordinatewise.

Scenario 1: Sum of sines. The raw function is $\begin{array} { r } { r _ { 1 } ( x ) = d ^ { - 1 / 2 } \sum _ { j = 1 } ^ { d } \sin ( 3 . 3 z _ { j } ) } \end{array}$

Scenario 2: Smooth surface. Define $H _ { d } ( t ) =$ sin $\begin{array} { r l r } {  { [ ( 1 0 / \sqrt { d } ) \{ \sum _ { i = 1 } ^ { d } ( t _ { j } - 1 / 2 ) ^ { 2 } - d / 1 2 \} ] } } \end{array}$ ] (Jeong and Ročková, 2023) and set $r _ { 2 } ( x ) = H _ { d } ( t )$

Scenario 3: Friedman blocks. Let $F ( t _ { 1 } , \dots , t _ { 5 } ) = 1 0 \sin ( \pi t _ { 1 } t _ { 2 } ) + 2 0 ( t _ { 3 } - 1 / 2 ) ^ { 2 } + 1 0 t _ { 4 } +$ $5 t _ { 5 }$ . For $d = 5$ , set $r _ { 3 } ( x ) = F ( t _ { 1 } , \dots , t _ { 5 } )$ ; for $d = 1 0$ set $r _ { 3 } ( x ) = \{ F ( t _ { 1 } , \dots , t _ { 5 } ) + F ( t _ { 6 } , \dots , t _ { 1 0 } ) \} / \sqrt { 2 }$

Scenario 4: Smooth surface with jumps. Let $\begin{array} { r } { S ( t ) = \sum _ { i = 1 } ^ { d } ( t _ { j } - 1 / 2 ) } \end{array}$ and $\begin{array} { r } { W ( t ) = \sum _ { i = 1 } ^ { d } ( - 1 ) ^ { j } ( t _ { j } - } \end{array}$ $1 / 2 )$ Define $J _ { d } ( t ) = \mathbf { 1 } \{ S ( t ) \leq 0 , \ \check { W } ( t ) > 0 \} +$ ${ \mathbf 1 } \{ S ( t ) > 0 , W ( t ) \leq 0 \}$ and set $\begin{array} { r } { r _ { 4 } ( x ) = H _ { d } ( t ) + J _ { d } ( t ) } \end{array}$ The jump boundaries are hyperplanes in t-space but need not be hyperplanes in the observed predictor space.

For each $( p , d )$ , scenarios and thirty replications share a fixed transformation matrix and training/test inputs; only noise and MCMC seeds vary. Methods use identical datasets, and scenarios share noise within each replication. The smaller training sample contains the first 1,500 observations of the larger one, including their noise, and both sizes use the same test inputs.

All methods fit 200 trees to the observed data and average 1,000 unthinned draws after 9,000 burn-in iterations for prediction. RoBART uses $K _ { \mathrm { r o t } } = 5$ and $\gamma = 0 . 0 8$ . See Appendix D.

![](images/a191dc3b191fa5b59cd2e000d3ab884cc87ea2dc3a7ffa9f5b5db5354870bd43.jpg)  
Figure 3: RMSPE (a) and elapsed time in seconds (b) at $n \in \{ 1 , 5 0 0 , 5 , 0 0 0 \}$ . Rows give Scenarios 1–4 and columns give (p, d). Boxes summarize thirty replications; RMSPE is measured against the true regression function.

## 6.2 Prediction accuracy and runtime

Figure 3(a) shows lower median RMSPE for RoBART across the displayed settings in Scenarios 2– 4. The improvement is less uniform in Scenario 1, where errors remain close to one in some $p = 3 0$ settings at $n = 1 , 5 0 0$

Figure 3(b) compares elapsed times under the reported fitting settings. RoBART requires more time than BART and Oblique BART, but less than Soft BART and, in most displayed settings, ridgeBART.

## 7 DISCUSSION

RoBART combines tree-specific rotations with a joint rotation and cutpoint update that accounts for the changing numerical grid. We establish reversibility of the joint conditional update and posterior contraction for additive regression functions with component-specific rotations. For a fixed Hölder ridge function on a tensor design, the axisaligned counterpart need not attain RoBART’s onedimensional contraction rate.

The contraction result assumes fixed p, T, R with $R \leq T$ . Extensions to growing p and convergence analysis of the sampler require further study.

## ACKNOWLEDGMENTS

This research was supported by the Basic Science Research Program through the National Research Foundation of Korea (NRF), funded by the Ministry of Education (RS-2025-25426611).

## References

Blaser, R. and Fryzlewicz, P. (2016). Random rotation ensembles. Journal of Machine Learning Research, 17(4):1–26.

Chipman, H. A., George, E. I., and McCulloch, R. E. (2010). BART: Bayesian additive regression trees. The Annals of Applied Statistics, 4(1):266–298.

Collins, G., Francom, D., and Rumsey, K. (2024). Bayesian projection pursuit regression. Statistics and Computing, 34(1):29.

Ghosal, S., Ghosh, J. K., and van der Vaart, A. W. (2000). Convergence rates of posterior distributions. The Annals of Statistics, 28(2):500–531.

Ghosal, S. and van der Vaart, A. W. (2007). Convergence rates of posterior distributions for non-i.i.d. observations. The Annals of Statistics, 35(1):192– 223.

Ghosal, S. and van der Vaart, A. W. (2017). Fundamentals of Nonparametric Bayesian Inference. Cambridge University Press.

Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. Biometrika, 82(4):711–732.

Hill, J., Linero, A. R., and Murray, J. S. (2020). Bayesian additive regression trees: A review and look forward. Annual Review of Statistics and Its Application, 7:251–278.

Jeong, S. (2025). L -norm posterior contraction in Gaussian models with unknown variance. Statistics & Probability Letters, 226:110495.

Jeong, S. and Ročková, V. (2023). The art of BART: Minimax optimality over nonhomogeneous smoothness in high dimension. Journal of Machine Learning Research, 24(337):1–65.

Linero, A. R. and Yang, Y. (2018). Bayesian regression tree ensembles that adapt to smoothness and sparsity. Journal of the Royal Statistical Society: Series B, 80(5):1087–1110.

Nguyen, P.-H. V., Yee, R., and Deshpande, S. K. (2025). Oblique Bayesian additive regression trees. Transactions on Machine Learning Research.

Ročková, V. and Saha, E. (2019). On theory for BART. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 2839–2848.

Ročková, V. and van der Pas, S. (2020). Posterior concentration for Bayesian regression trees and forests. The Annals of Statistics, 48(4):2108–2131.

Yee, R., Ghosh, S., and Deshpande, S. K. (2024). Scalable piecewise smoothing with BART. arXiv preprint arXiv:2411.07984.

## Appendices

## A REVERSIBILITY OF THE JOINT ROTATION AND CUTPOINT UPDATE

## A.1 Conditional target and proposal path

Fix $p \geq 2 ,$ , the training design, and positive integer budgets $b _ { 1 } , \ldots , b _ { p }$ . For one tree, condition on its ordered binary topology $\tau ,$ split-label array V , partial residual vector $r ^ { ( t ) } , \sigma , \tau > 0 .$ , and the complete states of the other trees. Suppress the tree index. The joint update acts on ${ \mathfrak { s } } = ( Q , C )$ in $Z = \operatorname { S O } ( p ) \times { \mathcal { C } } _ { \operatorname* { m a x } }$ , where $\begin{array} { r } { \mathcal { C } _ { \operatorname* { m a x } } = \prod _ { \eta \in \mathcal { I } ( \mathcal { T } ) } \{ 0 , \dotsc , b _ { v _ { \eta } } - 1 \} } \end{array}$ . Let λ be normalized Haar measure times counting measure on this space. With $\ell _ { H }$ and the full tree prior $\pi _ { \mathrm { d i s c } }$ as defined in the main paper, the collapsed conditional target is

$$
\widetilde \pi _ { H } ( \mathsf { s } ) = e ^ { \ell _ { H } ( \mathsf { s } ) } \pi _ { \mathsf { d i s c } } ( { \mathcal T } , V , C \mid Q ) , \qquad \Pi _ { H } ( d \mathsf { s } ) = Z ^ { - 1 } \widetilde \pi _ { H } ( \mathsf { s } ) \lambda ( d \mathsf { s } ) .
$$

Invalid arrays have density zero. As in the main proposition, assume $\begin{array} { r } { 0 < Z = \int \widetilde { \pi } _ { H } d \lambda < \infty } \end{array}$ . Normalization is joint over Q and C: the tree-prior factor is not separately normalized over C for each Q.

The $\mathrm { g r i d } \equiv ( Q )$ is the main paper’s coordinatewise empirical-range grid, using only the full training design. Its evaluation is exact, with deterministic measurable tie rules as in the main proposition. For a node η and label $j , \mathcal { A } _ { \eta } ( j ; Q , C _ { \mathrm { a n c } } )$ is the set of grid indices strictly inside the interval imposed by its same-label ancestors; it is empty for a zero-range coordinate. There is no minimum leaf occupancy, and split equality goes to the right child. Write $a _ { \eta } ( \mathsf { s } ) = | \mathcal { A } _ { \eta } ( v _ { \eta } ; Q , C _ { \mathrm { a n c } } ) |$ and $g _ { \eta } ( \mathsf { s } ) = \# \{ j : \mathcal { A } _ { \eta } ( j ; Q , C _ { \mathrm { a n c } } ) \neq \emptyset \}$ . The depth probabilities satisfy $0 < \rho _ { \ell } < 1 $ ; no asymptotic design or grid condition is needed for this conditional update. Let $\mathcal { P } = \{ ( a , b ) \in [ p ] ^ { 2 } : a \neq b \}$ and $K = K _ { \mathrm { r o t } } \ge 1$ . Conditional on the fixed $( \mathcal { T } , V )$ , draw $P _ { 1 } , \ldots , P _ { K }$ independently from the pair law $q _ { P }$ in the main paper. Independently draw $\theta _ { 1 } , \ldots , \theta _ { K }$ from its centered $N ( 0 , \gamma ^ { 2 } )$ law truncated to $[ - \theta _ { \mathrm { m a x } } , \theta _ { \mathrm { m a x } } ]$ , where $\gamma > 0$ and $0 < \theta _ { \mathrm { m a x } } < \pi$ . Its density q is even. All tuning parameters and activation schedules are fixed in advance. Write $H ( P , \theta ) = G _ { P _ { K } } ( \theta _ { K } ) \cdot \cdot \cdot G _ { P _ { 1 } } ( \theta _ { 1 } )$ , so $Q ^ { \prime } = H ( P , \theta ) Q$ . The reversed path $( P _ { K } , - \theta _ { K } ) , \dots , ( P _ { 1 } , - \theta _ { 1 } )$ reconstructs Q because $G _ { P } ( \theta ) ^ { - 1 } = G _ { P } ( - \theta )$ The coordinates within each ordered pair are not exchanged. Independence under the same fixed pair law and evenness of q make the forward and reversed path probabilities equal. We keep the realized path as an auxiliary variable; an endpoint proposal density with respect to Haar measure is not assumed.

## A.2 Sequential cutpoint proposal

For the realized path, set $\mathcal { T } _ { \mathrm { t o u c h } } = \bigcup _ { k = 1 } ^ { K } \{ a _ { k } , b _ { k } \}$ and $\mathcal { N } _ { \mathrm { t o u c h } } = \{ \eta \in \mathcal { I } ( \mathcal { T } ) : v _ { \eta } \in \mathcal { I } _ { \mathrm { t o u c h } } \}$ . Only those grid columns change under the coordinatewise grid rule. Fix one parent-before-child order on $\mathcal { T } ( \mathcal { T } )$ and $\rho _ { \mathrm { c u t } } > 0$ Under a departure state ${ \mathsf { \pmb { \mathsf { s } } } } ^ { 0 } = ( Q ^ { 0 } , C ^ { 0 } )$ , compute the observation indices $\mathcal { T } _ { \eta } ^ { 0 }$ reaching each node and its left-child count $n _ { \eta , L } ^ { 0 }$ . Keep both fixed throughout the recursion. Working ancestor indices determine the admissible target interval; they do not change these reference observations.

At a touched node with $0 < n _ { \eta , L } ^ { 0 } < | \mathcal { Z } _ { \eta } ^ { 0 } |$ , let $z _ { \eta , ( k ) } ^ { \prime }$ be the order statistics of $\{ ( Q ^ { \prime } x _ { i } ) _ { v _ { \eta } } : i \in \mathcal { T } _ { \eta } ^ { 0 } \}$ . For the current admissible interval $\mathcal { A } _ { \eta } ^ { F } = \{ \underline { { c } } _ { \eta } , \dots , \overline { { c } } _ { \eta } \}$ , set $t _ { \eta } ^ { \star } = ( z _ { \eta , ( n _ { \eta , L } ^ { 0 } ) } ^ { \prime } + z _ { \eta , ( n _ { \eta , L } ^ { 0 } + 1 ) } ^ { \prime } ) / 2$ and choose $c _ { \eta } ^ { \star } =$ min arg min $\smash { \ v { c } } { \in } \mathcal { A } _ { \eta } ^ { F } \ \big | \xi _ { v _ { \eta } , c } ( Q ^ { \prime } ) - t _ { \eta } ^ { \star } \big |$ . With $W _ { \eta } = | A _ { \eta } ^ { F } |$ and $\kappa _ { \eta } = \operatorname* { m a x } \{ 1 , \rho _ { \mathrm { c u t } } W _ { \eta } \}$ , use

$$
q _ { \eta } ( c ) = \frac { \exp \{ - ( c - c _ { \eta } ^ { \star } ) ^ { 2 } / ( 2 \kappa _ { \eta } ^ { 2 } ) \} } { \sum _ { u \in \mathcal { A } _ { \eta } ^ { F } } \exp \{ - ( u - c _ { \eta } ^ { \star } ) ^ { 2 } / ( 2 \kappa _ { \eta } ^ { 2 } ) \} } \mathbf { 1 } \{ c \in \mathcal { A } _ { \eta } ^ { F } \} .\tag{A.1}
$$

This center need not preserve observation counts or partitions after snapping to the grid or changing ancestors. At an untouched node, or when all reference observations go to one child or the node is empty, retain $c _ { \eta } ^ { 0 }$ if admissible; otherwise return failure ∂. An empty admissible set also gives $\partial .$

Algorithm A.1 Sequential cutpoint proposal or mass evaluation   
Require: Departure state $( Q ^ { 0 } , C ^ { 0 } )$ , target $Q ^ { \prime }$ and $\Xi ( Q ^ { \prime } ) , \mathcal { N } _ { \mathrm { t o u c h } } ;$ generation or evaluation mode, with   
requested D in evaluation mode   
1: Compute and freeze $\mathcal { T } _ { \eta } ^ { 0 } , n _ { \eta , L } ^ { 0 }$ under $( Q ^ { 0 } , C ^ { 0 } )$   
2: Initialize $\bar { C }  C ^ { 0 }$ and log $q \gets 0$   
3: for internal nodes $\eta \in \mathcal { T } ( \mathcal { T } )$ in the fixed parent-before-child order do   
4: $\mathcal { A }  \mathcal { A } _ { \eta } ( v _ { \eta } ; Q ^ { \prime } , \bar { C } _ { \mathrm { a n c } } )$   
5: if $A = \emptyset$ then   
6: return ∂ in generation mode, or $( D , - \infty )$ in evaluation mode   
7: end if   
8: if $\eta \notin \mathcal { N } _ { \mathrm { t o u c h } }$ or $n _ { \eta , L } ^ { 0 } \in \{ 0 , | \mathcal { Z } _ { \eta } ^ { 0 } | \}$ then   
9: if $c _ { \eta } ^ { 0 } \notin { \mathcal { A } }$ then   
10: return $\partial$ in generation mode, or $( D , - \infty )$ in evaluation mode   
11: end if   
12: $h _ { \eta } ( c ) \gets \mathbf { 1 } \{ c = c _ { \eta } ^ { 0 } \}$   
13: else   
14: Compute $t _ { \eta } ^ { \star } , c _ { \eta } ^ { \star } , \kappa _ { \eta }$ and set $h _ { \eta }$ by (A.1)   
15: end if   
16: if generation mode then   
17: Draw $d _ { \eta } \sim h _ { \eta }$   
18: else   
19: $d _ { \eta }  D _ { \eta }$   
20: if $h _ { \eta } ( d _ { \eta } ) = 0$ then   
21: return $( D , - \infty )$   
22: end if   
23: end if   
24: log q ← log q + log $h _ { \eta } ( d _ { \eta } )$ ; commit $\bar { c } _ { \eta } \gets d _ { \eta }$   
25: end for   
26: return $( { \bar { C } } , \log q )$

Algorithm A.1 defines a measurable law on $\mathcal { C } _ { \mathrm { m a x } } \cup \{ \partial \}$ : paths, order statistics, and nearest-index choices are measurable finite operations with deterministic ties. Each local law is a normalized point mass or a finite distribution with a positive normalizer. Sequential composition assigns each completed array its product of local masses and assigns failed prefixes to $\partial .$ Thus completed-array and failure probabilities sum to one. For a completed array $C ^ { \prime }$ , call its product probability $q _ { C } ^ { F }$ . It is not conditioned on success. Failure leaves the entire (Q, C) state unchanged, without resampling or committing a partial array. For a valid departure state, an untouched index remains admissible: its frame row and all same-label ancestor indices are unchanged.

To obtain $q _ { C } ^ { R }$ , run the evaluation mode with departure $( Q ^ { \prime } , C ^ { \prime } )$ , target rotation $Q ,$ and requested array $D = C .$ . Recompute and freeze reference observations and left counts at $( Q ^ { \prime } , C ^ { \prime } ) ;$ ; initialize the working array at $C ^ { \prime }$ and commit each requested old index before visiting descendants. Recompute all reverse centers, scales, and normalizers. A reverse deterministic hold assigns positive mass only when $c _ { \eta } = c _ { \eta } ^ { \prime }$ . Reverse evaluation uses the same node order, not its reversal. The reversed Givens path has the same touched set.

## A.3 Reversibility of the joint update

Proof of the reversibility proposition in the main paper. Use $\mathsf { U } \ = \ \mathcal { P } ^ { K } \times [ - \theta _ { \mathrm { m a x } } , \theta _ { \mathrm { m a x } } ] ^ { K } \times \mathcal { C } _ { \mathrm { m a x } }$ for nonfailure auxiliary outputs. Its reference measure κ is counting measure on pairs and cutpoint arrays times Lebesgue measure on angles. For $\mathsf { \pmb { \mathsf { s } } } = ( Q , C )$ and $u = ( P _ { 1 : K } , \theta _ { 1 : K } , C ^ { \prime } )$ , set $\mathsf { \Omega } \mathsf { s } ^ { \prime } = ( H ( P , \theta ) Q , C ^ { \prime } )$ and $u ^ { \dagger } = ( P _ { K : 1 } , - \theta _ { K : 1 } , C )$ . The map $\mathfrak { J } ( \mathfrak { s } , u ) = ( \mathfrak { s } ^ { \prime } , u ^ { \dag } )$ is a measurable involution on the full product space, including tuples with zero target or proposal density: applying the reversed rotations, array exchange, and signed path reversal twice restores every coordinate.

For each fixed pair–angle path, $Q \mapsto H ( P , \theta ) Q$ preserves Haar measure. Reversing the pair sequence and exchanging $C , C ^ { \prime }$ preserve counting measure, and the signed permutation of angles preserves Lebesgue measure. Applying the Haar change of variables first for each path and then these permutations shows by Tonelli’s theorem that $\begin{array} { r } { \int h \circ \mathfrak { J } d ( \lambda \otimes \kappa ) = \int h d ( \lambda \otimes \kappa ) } \end{array}$ for every nonnegative measurable h. This accounts for the dependence of H on the angles. The grid is a function of $Q ,$ not an extra coordinate of the reference measure.

The non-failure auxiliary density is $\begin{array} { r } { q _ { 5 } ( u ) = \{ \prod _ { k = 1 } ^ { K } q _ { P } ( P _ { k } ~ | ~ \mathcal { T } , V ) q _ { \theta } ( \theta _ { k } ) \} q _ { C } ^ { F } } \end{array}$ . Give inadmissible outputs density zero, and extend this density by zero on zero-target departure states. Its integral is at most one; the missing mass is failure probability, not a normalizing factor. Define $F ( \mathfrak { s } , u ) = \widetilde { \pi } _ { H } ( \mathfrak { s } ) q _ { \mathfrak { s } } ( u )$ . Since $\Im$ is an involution, $( F \circ \mathfrak { J } ) ( \mathsf { s } , u ) = \widetilde { \pi } _ { H } ( \mathsf { s } ^ { \prime } ) q _ { \mathsf { s } ^ { \prime } } ( u ^ { \dagger } )$ . At positive-density endpoints with positive forward and reverse mass, the pair and angle factors cancel, giving

$$
\log \frac { F \circ \Im } { F } = \ell _ { H } ( \mathsf { s } ^ { \prime } ) - \ell _ { H } ( \mathsf { s } ) + \Delta _ { \mathrm { d i s c } } + \log q _ { C } ^ { R } - \log q _ { C } ^ { F } .
$$

Only the cutpoint and terminal-node prior factors remain in $\Delta _ { \mathrm { d i s c } } .$ , because $( \mathcal { T } , V )$ is fixed:

$$
\Delta _ { \mathrm { d i s c } } = \sum _ { \eta \in \mathcal { T } ( \mathcal { T } ) } \log \frac { a _ { \eta } ( \mathsf { s } ) } { a _ { \eta } ( \mathsf { s } ^ { \prime } ) } + \sum _ { \ell \in \mathcal { L } ( \mathcal { T } ) } \log \frac { 1 - \rho _ { \mathrm { d e p } ( \ell ) } g _ { \ell } ( \mathsf { s } ^ { \prime } ) / p } { 1 - \rho _ { \mathrm { d e p } ( \ell ) } g _ { \ell } ( \mathsf { s } ) / p } .
$$

For $F > 0 ,$ , let $\alpha = 1 \wedge ( F \circ \Im ) / F ;$ set $\alpha = 0$ when $F = 0 . \mathrm { ~ A ~ }$ zero proposed target or reverse mass therefore rejects the proposal without evaluating an undefined log ratio. This is the acceptance rule of the main paper, and

$$
F \alpha = \operatorname* { m i n } \{ F , F \circ { \mathfrak { J } } \} = ( F \circ { \mathfrak { J } } ) ( \alpha \circ { \mathfrak { J } } ) .
$$

For measurable $B _ { 0 } , B _ { 1 } \subseteq Z$ , the target-weighted accepted transition mass is

$$
\frac { 1 } { Z } \int \mathbf { 1 } \{ \mathsf { s } \in B _ { 0 } \} \mathbf { 1 } \{ \mathsf { s ^ { \prime } } \in B _ { 1 } \} \operatorname* { m i n } \{ F , F \circ \mathfrak { I } \} \lambda ( d \mathsf { s } ) \kappa ( d u ) .
$$

The measure-preserving involution exchanges the indicators and fixes the minimum, so this expression is symmetric in $B _ { 0 } , B _ { 1 }$ . The remaining probability is $\begin{array} { r } { r ( \mathsf { s } ) = 1 - \int q _ { \mathsf { s } } ( u ) \alpha ( \mathsf { s } , u ) \kappa ( d u ) \in [ 0 , 1 ] } \end{array}$ . Rejections and failures thus contribute the symmetric mass $\begin{array} { r } { \int _ { B _ { 0 } \cap B _ { 1 } } r ( \mathsf { s } ) \Pi _ { H } ( d \mathsf { s } ) } \end{array}$ . Together these prove detailed balance for the full joint-update kernel. □

The proof uses the exact grid and proposal laws. A subsequent numerical projection of $Q ^ { \prime }$ without accounting for it defines a diferent kernel. The result establishes invariance, not irreducibility or a mixing-rate bound.

## A.4 Completing the tree update

For tree t, write $\zeta _ { t } = ( Q _ { t } , C _ { t } , \mathcal { T } _ { t } , V _ { t } )$ and let $M _ { t }$ be its leaf vector. Conditional on $r ^ { ( t ) } , \sigma , \tau$ , and the other trees, write the joint posterior as $\Pi _ { t } ( d \zeta _ { t } , d M _ { t } ) = \overline { { \Pi } } _ { t } ( d \zeta _ { t } ) \Pi _ { t } ( d M _ { t } \mid \zeta _ { t } )$

Corollary A.1 (Invariance of the completed tree update). Let $L _ { t }$ be a finite composition of structural kernels preserving ${ \overline { { \Pi } } } _ { t } .$ with the above conditioning quantities fixed and without using old leaf values. Following $L _ { t }$ with a draw of all leaf means from $\Pi _ { t } ( d M _ { t } ^ { \prime } \mid \zeta _ { t } ^ { \prime } )$ preserves $\Pi _ { t }$

Proof. Since $\overline { { \Pi } } _ { t } L _ { t } = \overline { { \Pi } } _ { t }$ , the target-weighted completed transition satisfies

$$
\begin{array} { r l } {  { \int \Pi _ { t } ( d \zeta _ { t } , d M _ { t } ) L _ { t } ( \zeta _ { t } , d \zeta _ { t } ^ { \prime } ) \Pi _ { t } ( d M _ { t } ^ { \prime } \mid \zeta _ { t } ^ { \prime } ) } } \\ & { = \int \overline { { \Pi } } _ { t } ( d \zeta _ { t } ) L _ { t } ( \zeta _ { t } , d \zeta _ { t } ^ { \prime } ) \Pi _ { t } ( d M _ { t } ^ { \prime } \mid \zeta _ { t } ^ { \prime } ) } \\ & { = \overline { { \Pi } } _ { t } ( d \zeta _ { t } ^ { \prime } ) \Pi _ { t } ( d M _ { t } ^ { \prime } \mid \zeta _ { t } ^ { \prime } ) = \Pi _ { t } ( d \zeta _ { t } ^ { \prime } , d M _ { t } ^ { \prime } ) . } \end{array}
$$

An independent draw from $\overline { { \Pi } } _ { t }$ is not required.

The joint kernel leaves $( \mathcal { T } , V )$ fixed and preserves their conditional target, so it preserves $\overline { { \Pi } } _ { t }$ . Composing it with birth–death kernels for the same collapsed target gives the $L _ { t }$ used above. Their actual proposal probabilities and the full discrete prior must enter their acceptance ratios, including when a birth proposal samples only among available labels. Refresh all leaf means after the final structural state, even after rotation rejection or failure, before restoring the tree contribution or using its leaf values in another block. Completed tree kernels in systematic backfitting order, followed by valid updates of the remaining model parameters, preserve the full posterior. The sweep need not itself be reversible, and no finite-burn-in guarantee follows from this invariance statement.

## A.5 Computational cost

We count arithmetic and comparison operations for one joint proposal, including reverse probability evaluation, with elementary functions and scalar random variates treated as unit-cost operations. The current rotated training design is cached, its proposed version uses a full scratch copy, unchanged grid columns are shared, and only touched frame rows are bufered. Birth–death moves and the subsequent leaf refresh are not included in this bound.

Let $D$ be the maximum leaf depth, N the number of nodes, $J = | \mathcal { I } _ { \mathrm { t o u c h } } | , I = | \mathcal { N } _ { \mathrm { t o u c h } } |$ , and $b = \operatorname* { m a x } _ { j } b _ { j }$ . For departure or proposed routing $a \in \{ 0 , 1 \}$ , let $d _ { i } ^ { a }$ be the depth reached by observation i. Then

$$
\sum _ { \eta \in \mathcal { I } ( \mathcal { T } ) } | \mathcal { T } _ { \eta } ^ { a } | = \sum _ { i = 1 } ^ { n } d _ { i } ^ { a } \leq n D .
$$

Following each observation’s path collects the touched-node values in both directions within $2 n D$ incidences. Only two adjacent order statistics are needed per nondegenerate node. Worst-case linear-time selection therefore gives the costs in Table $_ { \mathrm { A . 1 ; } }$ metadata and point-mass checks add $O ( N )$

Table A.1: Operation bounds for one joint proposal under the stated storage convention.
<table><tr><td>Operation</td><td>Cost</td></tr><tr><td>Copy the rotated training design</td><td> $O ( n p )$ </td></tr><tr><td>Apply the Givens path to design columns and frame rows</td><td> $O \{ K _ { \mathrm { r o t } } ( n + p ) \}$ </td></tr><tr><td>Recompute touched extrema and materialize their grid columns</td><td> $O \{ J ( n + b ) \}$ </td></tr><tr><td>Route observations, collect reference values, select order statistics, and evaluate collapsed likelihoods</td><td> $O \{ n ( 1 + D ) + N \}$ </td></tr><tr><td>Enumerate local cutpoint normalizers and evaluate or draw indices</td><td> ${ \cal O } ( I b )$ </td></tr><tr><td>Evaluate discrete-prior factors by inspecting labels and ancestors</td><td> $O \{ p N ( 1 + D ) \}$ </td></tr></table>

Direct enumeration costs $O ( W _ { \eta } ) \le O ( b )$ per touched node and direction; nearest-index search is included. With cached range flags and index bounds, the prior calculation inspects at most $p$ labels and D ancestors

per node. Touched-row storage costs $O ( J p )$ , absorbed by $J \le 2 K _ { \mathrm { r o t } }$ . Combining the terms gives

$$
O \{ n p + K _ { \mathrm { r o t } } ( n + p ) + J ( n + b ) + n ( 1 + D ) + I b + p N ( 1 + D ) \} .\tag{A.2}
$$

Sorting every reference set instead of linear-time selection replaces the corresponding $n ( 1 + D )$ term by the conservative bound $n ( 1 + D ) \log ( n + 1 )$ . An average-linear selection routine gives only its corresponding average guarantee. Early rejection can shorten an attempt; failed attempts are not retried until success.

With all reference sets materialized, one tree’s resident-plus-working storage is $\begin{array} { r } { O \{ n p + p ^ { 2 } + \sum _ { j } b _ { j } + n ( 1 + \Biggr . } \end{array}$ $D ) + N + K _ { \mathrm { { r o t } } } \}$ under these conventions. For fixed $p , K _ { \mathrm { r o t } } , D , b , N \le 2 ^ { D + 1 } - 1$ makes $\left( \mathrm { A . 2 } \right)$ an $O ( n )$ bound. Here D is the realized depth, not a prior depth cap. This is a per-attempt bound, not a uniform linear-time result for increasing budgets, depths, complete sweeps, or a prescribed Monte Carlo accuracy.

## B POSTERIOR CONTRACTION FOR RoBART

## B.1 Introduction

We prove the posterior contraction theorem in the main paper under the prior and design conditions below. Following the additive construction of Jeong and Ročková (2023), we assign one approximating tree to each component, allowing separate rotations and rotation-dependent cutpoint grids, and take the remaining trees to be nearly zero. We verify local prior mass, sieve entropy, and prior tail bounds, then apply the Gaussian contraction criterion of Jeong (2025). Theorem B.3.1 restates the main paper’s result; numbered references prefixed by B refer to this supplement.

## B.2 Model, prior, and assumptions

## B.2.1 Fixed-design regression model

We observe

$$
Y _ { i , n } = f _ { 0 } ( x _ { i , n } ) + \varepsilon _ { i , n } , \qquad \varepsilon _ { i , n } \overset { \mathrm { i i d } } { \sim } N ( 0 , \sigma _ { 0 } ^ { 2 } ) , \qquad i = 1 , \dots , n ,\tag{B.1}
$$

with fixed $\sigma _ { 0 } > 0$ and fixed design points $x _ { i , n } \in \mathbb { R } ^ { p }$ . The ambient dimension $p \geq 2$ is fixed. For a measurable function $g$ on the design space, define the empirical norm

$$
\left\| g \right\| _ { n } ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | g ( x _ { i , n } ) | ^ { 2 } .
$$

We consider a sequence of fixed designs contained in one bounded set: there exists $M _ { X } < \infty$ such that

$$
\operatorname* { s u p } _ { n \geq 1 } \operatorname* { m a x } _ { 1 \leq i \leq n } \| x _ { i , n } \| _ { 2 } \leq M _ { X } .\tag{B.2}
$$

Take a compact design set $\mathcal { X }$ contained in the Euclidean ball of radius $M _ { X }$ and containing every $x _ { i , n }$ . Write $\boldsymbol { Y } ^ { ( n ) } = ( Y _ { 1 , n } , \ldots , Y _ { n , n } ) ^ { \top }$ . Let $\begin{array} { r } { P _ { f , \sigma } ^ { ( n ) } = \bigotimes _ { i = 1 } ^ { n } N \{ f ( x _ { i , n } ) , \sigma ^ { 2 } \} } \end{array}$ , set $P _ { 0 } ^ { ( n ) } = P _ { f _ { 0 } , \sigma _ { 0 } } ^ { ( n ) }$ , and denote expectation under $P _ { 0 } ^ { ( n ) }$ by $\mathbb { E } _ { 0 }$

Write $\mathbb { P } _ { n }$ for the prior on the forest parameters and the noise scale, and $\Pi _ { n }$ for its induced prior on $( f , \sigma )$ . The corresponding posterior is $\textstyle \prod _ { n } ( { \cdot } \mid { \bar { Y } } ^ { ( n ) } )$ . The prior induced on $( \mu _ { f } , \sigma )$ , where $\mu _ { f } = ( f ( x _ { 1 , n } ) , \ldots , f ( x _ { n , n } ) ) ^ { \top }$ is denoted by $\textstyle { \overline { { \Pi } } } _ { n }$

Define $O ( p ) = \{ Q : Q ^ { \top } Q = I _ { p } \}$ and $\mathrm { S O } ( p ) = \{ Q \in O ( p ) : \operatorname* { d e t } Q = 1 \}$ . Every $Q \in \mathrm { S O } ( p )$ preserves Euclidean norm, so $Q x _ { i , n } \in D _ { X } = [ - M _ { X } , M _ { X } ] ^ { p }$

## B.2.2 Additive rotated truth and smoothness classes

Fix integers $1 \leq R \leq T$ , both independent of $n .$ On $x ,$ , the true regression function is assumed to satisfy

$$
f _ { 0 } ( x ) = \sum _ { r = 1 } ^ { R } h _ { 0 r } ( \pi _ { S _ { r } } ( Q _ { 0 r } x ) ) , \qquad Q _ { 0 r } \in \mathrm { S O } ( p ) , \qquad S _ { r } \subset \{ 1 , \ldots , p \} , \qquad | S _ { r } | = d _ { r } ,\tag{B.3}
$$

For each component let $1 \leq d _ { r } \leq p ,$ write $S _ { r } = \{ s _ { r 1 } < \cdot \cdot \cdot < s _ { r d _ { r } } \}$ , and define $\pi _ { S _ { r } } ( u ) = ( u _ { s _ { r 1 } } , \ldots , u _ { s _ { r d _ { r } } } ) ^ { \top }$ The set $S _ { r }$ indexes coordinates in the rth rotated system, not necessarily in the original system.

For any orthogonal matrix $Q \in \mathrm { S O } ( p )$ , we use the composition notation

$$
h _ { 0 r } ( \pi _ { S _ { r } } ( Q \cdot ) ) : x \mapsto h _ { 0 r } ( \pi _ { S _ { r } } ( Q x ) ) .
$$

Thus $h _ { 0 r } ( \pi _ { S _ { r } } ( Q _ { 0 r } \cdot ) )$ is the rth true component, while $h _ { 0 r } ( \pi _ { S _ { r } } ( Q \cdot ) )$ for $Q \ne Q _ { 0 r }$ means that the same canonical anisotropic template is evaluated after a diferent rotation.

Definition B.2.1 (Componentwise sparse anisotropic Hölder classes). For $r \ = \ 1 , \ldots , R .$ , let $\begin{array} { r l } { \alpha _ { r } } & { { } = } \end{array}$ $( \alpha _ { r 1 } , \ldots , \alpha _ { r d _ { r } } )$ with $\alpha _ { r k } \in ( 0 , 1 ]$ and let $\lambda _ { r } > 0$ . We write

$$
h _ { 0 r } \in \mathcal { H } _ { \lambda _ { r } } ^ { \alpha _ { r } , d _ { r } }
$$

if

$$
| h _ { 0 r } ( u ) - h _ { 0 r } ( v ) | \leq \lambda _ { r } \sum _ { k = 1 } ^ { d _ { r } } | u _ { k } - v _ { k } | ^ { \alpha _ { r k } }
$$

for all $u , v \in [ - M _ { X } , M _ { X } ] ^ { d _ { r } }$ . Each $h _ { 0 r }$ <sub>r</sub> is defined on this box and satisfies $\| h _ { 0 r } \| _ { \infty } \leq C _ { h }$ , with the supremum taken over the entire box. The harmonic-mean and minimal smoothness parameters are

$$
\bar { \alpha } _ { r } ^ { - 1 } = \frac { 1 } { d _ { r } } \sum _ { k = 1 } ^ { d _ { r } } \alpha _ { r k } ^ { - 1 } , \qquad \alpha _ { r , \mathrm { m i n } } = \operatorname* { m i n } _ { 1 \leq k \leq d _ { r } } \alpha _ { r k } .
$$

The exponent $\alpha _ { r k }$ corresponds to ambient row $s _ { r k }$ . All truth parameters are fixed as $n \to \infty$ . Constants may depend on these quantities, $p , R , T$ , and fixed prior hyperparameters, but not on n or a rotation in a stated neighborhood.

## B.2.3 RoBART prior with uniform coordinate selection

The prior is a tree-specific rotation forest. For $t = 1 , \dots , T$

$$
f ( x ) = \sum _ { t = 1 } ^ { T } g _ { t } ( Q _ { t } x ) , \qquad Q _ { t } \in \mathrm { S O } ( p ) .
$$

For tree $t ,$ let $\mathcal { T } _ { t }$ be its rooted ordered binary topology, V its split-label array, $C _ { t }$ its array of zero-based grid indices, and $M _ { t } = ( \mu _ { t \ell } ) _ { \ell \in \mathcal { L } ( \mathcal { T } _ { t } ) }$ its leaf-mean vector. Write $K _ { t } = | \mathcal { L } ( \mathcal { T } _ { t } ) |$ |. As in the main paper, $g _ { t } ( Q _ { t } x )$ abbreviates $g _ { t } ( Q _ { t } x ; T _ { t } , V _ { t } , C _ { t } , M _ { t } )$ . The complete discrete state is $( T _ { t } , V _ { t } , C _ { t } )$ , not the topology alone. The number of trees T is fixed.

We impose the following prior structure.

(P1) The rotations $Q _ { 1 } , \ldots , Q _ { T }$ are independent draws from normalized Haar measure on $\mathrm { S O } ( p )$

(P2) At each split attempt, a coordinate is drawn uniformly from $\{ 1 , \ldots , p \}$ , so every coordinate has selection probability $1 / p$ . These draws use fresh randomness independently across nodes and trees and independently of the rotations and split-attempt draws.

(P3) Conditional on $Q _ { 1 } , \ldots , Q _ { T }$ , the discrete trees $( \mathcal T _ { t } , V _ { t } , C _ { t } ) , t = 1 , \ldots , T .$ , are generated independently. At each node of depth $\ell ,$ a split is attempted with probability $\nu ^ { \ell + 1 }$ , where $\nu \in ( 0 , 1 )$ , using independent Bernoulli draws. If no split is attempted, the node is terminal. Otherwise, a coordinate is drawn according to (P2). If that coordinate has no admissible cutpoint in the current box, the node is terminal; otherwise, its cutpoint is drawn uniformly from the admissible elements of the corresponding rotated split-net, using a fresh draw conditional on the current box.

(P4) Conditional on $Q _ { 1 } , \ldots , Q _ { T } , ( T _ { t } , V _ { t } , C _ { t } ) _ { t = 1 } ^ { T }$ , all leaf means are mutually independent $N ( 0 , \tau ^ { 2 } )$ variables, where $0 < \tau ^ { 2 } < \infty$ is fixed.

(P5) Independently of the forest, $\sigma ^ { 2 } \sim \mathrm { I G } ( a _ { \sigma } , b _ { \sigma } )$ with fixed $a _ { \sigma } , b _ { \sigma } ~ > ~ 0$ and density proportional to $( \sigma ^ { 2 } ) ^ { - a _ { \sigma } - 1 } \exp \{ - b _ { \sigma } / \sigma ^ { 2 } \}$

For a fixed valid state $( T _ { t } , V _ { t } , C _ { t } )$ and rotation $Q _ { t }$ , let $a _ { \eta }$ be the number of admissible indices for the prescribed label at internal node $\eta ,$ and let $g _ { \ell }$ be the number of labels with an admissible index at terminal node ℓ. These counts depend on $Q _ { t }$ and the ancestor indices. Writing dep for node depth, the recursion gives $\begin{array} { r } { \pi _ { \mathrm { d i s c } } ( \mathcal { T } _ { t } , V _ { t } , C _ { t } \mid Q _ { t } ) = \prod _ { \eta \in \mathcal { T } ( \mathcal { T } _ { t } ) } \{ \nu ^ { \mathrm { d e p } ( \eta ) + 1 } / ( p a _ { \eta } ) \} \prod _ { \ell \in \mathcal { L } ( \mathcal { T } _ { t } ) } \{ 1 - \nu ^ { \mathrm { d e p } ( \ell ) + 1 } g _ { \ell } / p \} } \end{array}$ . Invalid states have mass zero. Each terminal factor is at least $1 - \nu$ . The factor $1 / p$ is used along the generating path, before conditioning on any subsequent descendant generation.

## B.2.4 Contraction rate and approximation scale

For $r = 1 , \ldots , R .$ , define

$$
\varepsilon _ { n , r } = ( \lambda _ { r } d _ { r } ) ^ { d _ { r } / ( 2 \bar { \alpha } _ { r } + d _ { r } ) } \left( \frac { \log { n } } { n } \right) ^ { \bar { \alpha } _ { r } / ( 2 \bar { \alpha } _ { r } + d _ { r } ) } , \qquad \varepsilon _ { n } = \sum _ { r = 1 } ^ { R } \varepsilon _ { n , r } .\tag{B.4}
$$

For each component choose an integer $L _ { n , r }$ and a leaf count $K _ { n , r } = 2 ^ { L _ { n , r } }$ such that

$$
K _ { n , r } \asymp \left( \frac { n ( \lambda _ { r } d _ { r } ) ^ { 2 } } { \log n } \right) ^ { d _ { r } / ( 2 \bar { \alpha } _ { r } + d _ { r } ) } .\tag{B.5}
$$

Then

$$
K _ { n , r } \log n \asymp n \varepsilon _ { n , r } ^ { 2 } , \qquad \sum _ { r = 1 } ^ { R } K _ { n , r } \log n \lesssim n \varepsilon _ { n } ^ { 2 } .\tag{B.6}
$$

We also define the shrinking active-row rotation radii

$$
\delta _ { n , r } : = c _ { \delta , r } \left( \frac { \varepsilon _ { n , r } } { \lambda _ { r } d _ { r } } \right) ^ { 1 / \alpha _ { r , \mathrm { m i n } } } ,\tag{B.7}
$$

where $c _ { \delta , r } > 0$ is suficiently small. Since all component parameters are fixed,

$$
\begin{array} { r } { \log ( 1 / \varepsilon _ { n } ) \lesssim \log n , \qquad \log ( 1 / \delta _ { n , r } ) \lesssim \log n . } \end{array}
$$

If

$$
a _ { r } = \frac { \bar { \alpha } _ { r } } { 2 \bar { \alpha } _ { r } + d _ { r } } , \qquad a _ { \star } = \operatorname* { m i n } _ { 1 \leq r \leq R } a _ { r } \in ( 0 , 1 / 2 ) ,
$$

then

$$
\varepsilon _ { n } \asymp \left( \frac { \log n } { n } \right) ^ { a _ { \star } } , \qquad \frac { n \varepsilon _ { n } ^ { 2 } } { \log n } \to \infty .\tag{B.8}
$$

Moreover, for every fixed $\kappa _ { K } \in ( 0 , 1 - 2 a _ { \star } )$

$$
{ \frac { n \varepsilon _ { n } ^ { 2 } } { \log n } } \geq n ^ { \kappa _ { K } }\tag{B.9}
$$

for all suficiently large n. This polynomial growth will be used in the sieve-tail bound.

## B.2.5 Rotation-adaptive split-nets and additive anisotropic k-d trees

Empirical intervals and candidate cutpoints. For each $n _ { \colon }$ , fix a positive integer budget $b _ { n }$ . At every $Q \in \mathrm { S O } ( p )$ , the prior uses the full empirical box of $Q x _ { 1 , n } , \ldots , Q x _ { n , n }$ and the $b _ { n }$ equally spaced interior cutpoints on each coordinate of positive range. A zero-range coordinate has no candidate cutpoints. The prior selects among all $p$ coordinates, and imposes no leaf-occupancy restriction. Only the approximating tree for component r is restricted to $S _ { r }$

For component r, define

$$
\mathcal { Q } _ { n , r } = \left\{ Q \in \mathrm { S O } ( p ) : \| Q _ { S _ { r } , \cdot } - Q _ { 0 r , S _ { r } , \cdot } \| _ { F } \leq \delta _ { n , r } \right\} .
$$

For any $Q \in \mathrm { S O } ( p )$ , write $u _ { i } ( Q ) = Q x _ { i , n }$ and, for $j \in S _ { r }$ , set

$$
m _ { r j } ( Q ) : = \operatorname* { m i n } _ { 1 \leq i \leq n } u _ { i j } ( Q ) , \qquad M _ { r j } ( Q ) : = \operatorname* { m a x } _ { 1 \leq i \leq n } u _ { i j } ( Q ) , \qquad R _ { n , r j } ( Q ) : = M _ { r j } ( Q ) - m _ { r j } ( Q ) .
$$

The active empirical box is

$$
D _ { Q , r } : = \prod _ { j \in S _ { r } } [ m _ { r j } ( Q ) , M _ { r j } ( Q ) ] .
$$

For $R _ { n , r j } ( Q ) > 0$ , define the coordinatewise candidate set directly by

$$
[ Z _ { n , r } ( Q ) ] _ { j } = \left\{ m _ { r j } ( Q ) + \frac { k } { b _ { n } + 1 } R _ { n , r j } ( Q ) : k = 1 , \ldots , b _ { n } \right\} .
$$

If $R _ { n , r j } ( Q ) = 0$ , set $[ Z _ { n , r } ( Q ) ] _ { j } = \emptyset$ . The family $Z _ { n , r } ( Q ) = ( [ Z _ { n , r } ( Q ) ] _ { j } : j \in S _ { r } )$ is the rotation-adaptive uniform split-net. Its entries are the corresponding candidate sets of the full prior grid; they depend only on Q and the design.

Admissible splits and tree partitions. For a box $\begin{array} { r } { D = \prod _ { j \in S } I _ { j } } \end{array}$ indexed by an ordered coordinate set $S ,$ write $[ D ] _ { j } = I _ { j }$ and let len $( [ D ] _ { j } )$ be its length. A split-net on $D$ is a family $Z = ( [ Z ] _ { j } : j \in S )$ of finite candidate sets $[ Z ] _ { j } \subset [ D ] _ { j }$ . For a sub-box $\Omega \subset D$ , the admissible cutpoints along j are

$$
\mathcal { C } _ { j } ( \Omega ; Z ) : = [ Z ] _ { j } \cap \mathrm { i n t } ( [ \Omega ] _ { j } ) .
$$

For $\tau \in \mathcal { C } _ { j } ( \Omega ; Z )$ , split Ω into

$$
\begin{array} { r } { \Omega ^ { - } = \Omega \cap \{ u \in D : u _ { j } < \tau \} , \qquad \Omega ^ { + } = \Omega \cap \{ u \in D : u _ { j } \geq \tau \} . } \end{array}
$$

We use this left/right convention throughout, including for design points on a boundary. A Z-tree partition $\mathcal { T } = \{ \Omega _ { 1 } , . . . , \Omega _ { K } \}$ is obtained from $D$ by finitely many such binary splits. Its step-function class is

$$
\mathcal { F } _ { \mathcal { T } } = \left\{ u \mapsto \sum _ { k = 1 } ^ { K } \beta _ { k } \mathbf { 1 } ( u \in \Omega _ { k } ) : \beta _ { 1 } , \dots , \beta _ { K } \in \mathbb { R } \right\} .
$$

For a bounded $g : D \to { \mathbb { R } }$ , write $\| g \| _ { \infty , D } = \operatorname* { s u p } _ { u \in D } | g ( u ) |$

Midpoint split and anisotropic schedule. If $\mathcal { C } _ { j } ( \Omega ; Z ) = \{ \tau _ { ( 1 ) } < \cdot \cdot \cdot < \tau _ { ( m ) } \}$ with $m \geq 1$ , its midpoint split uses $\tau _ { ( \lceil m / 2 \rceil ) }$ . If $m = 0$ , the split is unavailable.

Let $S = \{ s _ { 1 } < \cdots < s _ { d } \}$ and let $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { d } )$ with $\alpha _ { k } \in ( 0 , 1 ]$ , where $\alpha _ { k }$ corresponds to $s _ { k }$ . For $L \in \mathbb { N } .$ define $A _ { k d , S } ( D ; Z , \alpha , L )$ as follows.

(i) Start from {D} and initialize $l _ { k } = 0$ for $k = 1 , \ldots , d .$

(ii) Choose $k ^ { \star } \in$ arg min<sub>1≤k≤d</sub> α<sub>k</sub>l<sub>k</sub>, breaking ties by the smallest k.

(iii) If a midpoint split along coordinate $s _ { k ^ { \star } }$ is unavailable in at least one current leaf, stop.

(iv) Otherwise, split every current leaf along $s _ { k ^ { \star } }$ ⋆ and set $l _ { k ^ { \star } } \gets l _ { k ^ { \star } } + 1$

(v) Repeat until $\begin{array} { r } { \sum _ { k = 1 } ^ { d } l _ { k } = L } \end{array}$ or a required split is unavailable.

The output is a $Z .$ -tree partition with split counts $( l _ { 1 } , \ldots , l _ { d } )$ and the recorded split labels and global cutpoint ranks. Reaching depth L gives $2 ^ { L }$ leaves. Since $S$ is ordered, the tie rule is also the smallest-coordinate-label rule.

Discrete template and numerical partition. Apply the preceding schedule with $S _ { r } , \alpha _ { r } , L _ { n , r }$ to the integer grid whose root interval has endpoints 0 and $b _ { n } + 1$ on every active coordinate. Record the ordered topology, split labels, and global grid ranks; a global rank $k \in \{ 1 , \ldots , b _ { n } \}$ is stored as the main paper’s index $c = k - 1$ . Denote this complete discrete state by $\mathfrak { t } _ { n , r } .$ . It is determined without choosing a rotation. Whenever its prescribed splits are available under $Q { \mathrm { . } }$ let ${ \mathcal { P } } _ { n , r } ( Q )$ be the numerical partition of $D _ { Q , r }$ obtained by evaluating these indices on $Z _ { n , r } ( Q )$ . Proposition B.2.5 shows that the construction completes all $L _ { n , r }$ rounds and is valid throughout $\mathcal { Q } _ { n , r }$ for all suficiently large n. Leaving coordinates outside $S _ { r }$ unsplit gives the corresponding p-dimensional tree. Values of $h _ { 0 r }$ use the ordered coordinates $s _ { r 1 } , \ldots , s _ { r d _ { r } }$

Assumption B.2.2 (Additive rotation-adaptive uniform split-net regime). The active split-net $Z _ { n , r } ( Q )$ is constructed as above for every $r \leq R$ and every $Q \in \mathcal { Q } _ { n , r }$ . We assume the following.

(D1) Active-coordinate design nondegeneracy. There exist constants $r _ { \operatorname* { m i n } } > 0$ and $n _ { 0 } \in \mathbb { N }$ such that, for all $n \geq n _ { 0 }$ 2

$$
\operatorname* { m i n } _ { 1 \leq r \leq R } \operatorname* { m i n } _ { j \in S _ { r } } R _ { n , r j } ( Q _ { 0 r } ) \geq r _ { \operatorname* { m i n } } .
$$

(D2) Grid resolution. The sequence $b _ { n }$ satisfies

$$
\operatorname* { m a x } _ { 1 \leq r \leq R } 2 ^ { L _ { n , r } } \leq b _ { n } + 1 , \qquad \log b _ { n } \leq C _ { b } \log n
$$

for some constant $C _ { b } > 0$ and all suficiently large n.

Remark B.2.3 (Active-coordinate ranges). Condition (D1) keeps the empirical grid intervals in the truthactive directions nondegenerate under local rotations, as quantified in Proposition B.2.5(i). A suficient condition is that, for all large n, the convex hull of the design contains a Euclidean ball of fixed radius $c > 0 :$ every unit-direction range is then at least 2c. This is stronger than (D1), which restricts only the finitely many truth-active directions. In particular, (D1) does not require a full-rank empirical covariance, a design density, or a positive number of observations in every leaf; it may hold even for designs supported on a lower-dimensional afine subspace.

Remark B.2.4 (Grid resolution). The lower bound in (D2) ensures that every prescribed midpoint split is available; the upper bound controls the cutpoint contribution to the negative log prior mass. Let $\kappa _ { \star } = \operatorname* { m a x } _ { 1 \leq r \leq R } d _ { r } / ( 2 \bar { \alpha } _ { r } + d _ { r } ) < 1$ . By $( \mathrm { B } . 5 ) , b _ { n } = \lceil n ^ { \kappa } \rceil$ with $\kappa > \kappa ,$ <sub>⋆</sub> satisfies (D2), so $b _ { n } = n$ works without knowing the component smoothness. The resolution requirement is conservative: $b _ { n } + 1 \geq \operatorname* { m a x } _ { r , k } 2 ^ { l _ { r k } }$ sufices for the prescribed full-depth split counts, whereas (D2) uses $2 ^ { L _ { n , r } }$ . We retain the stated condition for simplicity. It requires an increasing cutpoint budget and does not cover a fixed finite grid.

Proposition B.2.5 (Properties of the additive rotation-adaptive split-nets). Under Assumption B.2.2, for all suficiently large n, every $r \leq R$ , and every $Q \in \mathcal { Q } _ { n , r }$ , the active split-net $Z _ { n , r } ( Q )$ satisfies the following properties.

(i) Local range stability and positivity. For every $j \in S _ { r }$

$$
\left| m _ { r j } ( Q ) - m _ { r j } ( Q _ { 0 r } ) \right| \leq M _ { X } \left\| Q _ { S _ { r } , \cdot } - Q _ { 0 r , S _ { r } , \cdot } \right\| _ { F } ,
$$

$$
\left| M _ { r j } ( Q ) - M _ { r j } ( Q _ { 0 r } ) \right| \leq M _ { X } \left\| Q _ { S _ { r } , \cdot } - Q _ { 0 r , S _ { r } , \cdot } \right\| _ { F } ,
$$

and

$$
| R _ { n , r j } ( Q ) - R _ { n , r j } ( Q _ { 0 r } ) | \leq 2 M _ { X } \left. Q _ { S _ { r } , \cdot } - Q _ { 0 r , S _ { r } , \cdot } \right. _ { F } .
$$

In particular, $R _ { n , r j } ( Q ) \geq R _ { n , r j } ( Q _ { 0 r } ) / 2 > 0$ for all suficiently large n.

(ii) Cutpoint count and side-length control. Each active coordinate has $b _ { n }$ candidate cutpoints, with log $b _ { n } \leq C _ { b }$ log n. Moreover, $\mathbf { t } _ { n , r }$ is valid under Q and reaches full depth $L _ { n , r } . \ I f \left( l _ { r k } : 1 \leq k \leq d _ { r } \right)$ denotes its split counts, then every leaf $\Omega \in { \mathcal { P } } _ { n , r } ( Q )$ satisfies

$$
\begin{array} { r } { \mathrm { l e n } \big ( [ \Omega ] _ { s _ { r k } } \big ) \leq 2 R _ { n , r , s _ { r k } } ( Q ) 2 ^ { - l _ { r k } } , \qquad k = 1 , \ldots , d _ { r } . } \end{array}
$$

(iii) Uniform sparse anisotropic approximation on $D _ { Q , r }$ . There exists a step function $\widehat { h } _ { r , Q } \in { \mathcal { F } _ { \mathcal { P } _ { n , r } \left( Q \right) } }$ such that

$$
\begin{array} { r } { \left\| h _ { 0 r } - \widehat { h } _ { r , Q } \right\| _ { \infty , D _ { Q , r } } \leq C _ { A } \lambda _ { r } d _ { r } 2 ^ { - L _ { n , r } \bar { \alpha } _ { r } / d _ { r } } \lesssim \varepsilon _ { n , r } . } \end{array}
$$

(iv) Fixed template and coeficient centers. The state $\mathrm { t } _ { n , r }$ is the same for all $Q \in \mathcal { Q } _ { n , r } ,$ its numerical thresholds and partition may vary. Choose $\widehat { h } _ { r , Q }$ by assigning each leaf the value of $h _ { 0 r }$ at the midpoint of its closure, and write $\beta _ { n , r } ( Q )$ for these values in $l e f t - t o - r i g h t$ leaf order. The map $Q \mapsto \beta _ { n , r } ( Q )$ is continuous on $\mathcal { Q } _ { n , r ; }$ , hence Borel measurable, and satisfies $\left\| \beta _ { n , r } ( Q ) \right\| _ { \infty } \leq C _ { h }$ and $\left\| \beta _ { n , r } ( Q ) \right\| _ { 2 } ^ { 2 } \leq C _ { h } ^ { 2 } K _ { n , r } ,$

Proof. Fix r and $Q \in \mathcal { Q } _ { n , r } .$

For (i), fix $j \in S _ { r }$ . Since

$$
m _ { r j } ( Q ) = \operatorname * { m i n } _ { 1 \leq i \leq n } e _ { j } ^ { \top } Q x _ { i , n } ,
$$

one has

$$
| m _ { r j } ( Q ) - m _ { r j } ( Q _ { 0 r } ) | \leq \operatorname* { m a x } _ { 1 \leq i \leq n } | e _ { j } ^ { \top } ( Q - Q _ { 0 r } ) x _ { i , n } | \leq M _ { X } \left. Q _ { S _ { r } , \cdot } - Q _ { 0 r , S _ { r } , \cdot } \right. _ { F } .
$$

The same argument applies to $M _ { r j } ( Q )$ , and the bound for $R _ { n , r j } ( Q )$ follows by the triangle inequality. Since $\delta _ { n , r }  0$ , condition (D1) gives positivity uniformly over $Q \in \mathcal { Q } _ { n , r }$ for suficiently large n.

For (ii), part (i) makes the $b _ { n }$ equally spaced candidates distinct on each active coordinate. Fix $k \in \{ 1 , \ldots , d _ { r } \}$ and put $j = s _ { r k }$ . The interval $[ m _ { r j } ( Q ) , M _ { r j } ( Q ) ]$ is partitioned into $b _ { n } + 1$ elementary cells, each of length

$$
\Delta _ { r j } ( Q ) = \frac { R _ { n , r j } ( Q ) } { b _ { n } + 1 } .
$$

Every subinterval generated by recursive midpoint splits is a union of consecutive elementary cells. An interval containing $N \geq 2$ cells has $N - 1$ admissible cutpoints; its midpoint split has rank $\lceil ( N - 1 ) / 2 \rceil = \lfloor N / 2 \rfloor$ and gives children with $\lfloor N / 2 \rfloor$ and $\lceil N / 2 \rceil$ cells. Let $N _ { j } ( \Omega )$ be the number of elementary cells in the jth coordinate interval of a leaf Ω. Iterating these floor and ceiling bounds, after $l _ { r k }$ midpoint splits along coordinate j,

$$
\left[ \frac { b _ { n } + 1 } { 2 ^ { l _ { r k } } } \right] \leq N _ { j } ( \Omega ) \leq \left\lceil \frac { b _ { n } + 1 } { 2 ^ { l _ { r k } } } \right\rceil .
$$

Before the ath split along coordinate j, every current interval contains at least

$$
\left\lfloor { \frac { b _ { n } + 1 } { 2 ^ { a - 1 } } } \right\rfloor
$$

elementary cells. Since $b _ { n } + 1 \geq 2 ^ { L _ { n , r } }$ , this quantity is at least two for every $a \leq L _ { n , r }$ . Therefore every requested midpoint split is available and the algorithm reaches full depth $L _ { n , r }$ . Finally,

$$
\begin{array} { r } { \mathrm { l e n } ( [ \Omega ] _ { j } ) = N _ { j } ( \Omega ) \Delta _ { r j } ( Q ) \leq 2 R _ { n , r j } ( Q ) 2 ^ { - l _ { r k } } , } \end{array}
$$

because $b _ { n } + 1 \geq 2 ^ { L _ { n , r } } \geq 2 ^ { l _ { r k } }$

For (iii), for each leaf $\Omega \in { \mathcal { P } } _ { n , r } ( Q )$ let $u _ { \Omega }$ be the coordinatewise midpoint of its closure and define

$$
\widehat { h } _ { r , Q } ( u ) = \sum _ { \Omega \in \mathcal { P } _ { n , r } ( Q ) } h _ { 0 r } ( u _ { \Omega } ) \mathbf { 1 } ( u \in \Omega ) .
$$

The split counts satisfy

$$
\operatorname* { m a x } _ { 1 \leq k \leq d _ { r } } \alpha _ { r k } l _ { r k } - \operatorname* { m i n } _ { 1 \leq k \leq d _ { r } } \alpha _ { r k } l _ { r k } \leq 1 .
$$

Indeed, the claim holds at initialization, and each update increments a coordinate minimizing $\alpha _ { r k } l _ { r k }$ by at most one because $\alpha _ { r k } \leq 1$ . If $\begin{array} { r } { m _ { r } = \operatorname* { m i n } _ { 1 \leq k \leq d _ { r } } \alpha _ { r k } l _ { r k } } \end{array}$ , then

$$
L _ { n , r } = \sum _ { k = 1 } ^ { d _ { r } } l _ { r k } \leq ( m _ { r } + 1 ) \sum _ { k = 1 } ^ { d _ { r } } \alpha _ { r k } ^ { - 1 } = ( m _ { r } + 1 ) \frac { d _ { r } } { \bar { \alpha } _ { r } } ,
$$

so $m _ { r } \geq L _ { n , r } \bar { \alpha } _ { r } / d _ { r } - 1$ . Hölder continuity and part (ii) now give

$$
| h _ { 0 r } ( u ) - \widehat h _ { r , Q } ( u ) | \leq C ( M _ { X } ) \lambda _ { r } \sum _ { k = 1 } ^ { d _ { r } } 2 ^ { - \alpha _ { r k } l _ { r k } } \lesssim \lambda _ { r } d _ { r } 2 ^ { - L _ { n , r } \bar { \alpha } _ { r } / d _ { r } } .
$$

Taking the supremum and using (B.5) proves the claim.

For (iv), the selected coordinate at each round depends only on the scaled split counts and their deterministic tie rule. An index interval with integer endpoints $a < b$ has candidates $a + 1 , \dotsc , b - 1 ;$ when it is split, its median rank is $a + \lfloor ( b - a ) / 2 \rfloor$ . This recursion does not depend on $Q ,$ and part (ii) ensures that every required split exists. Thus the ordered topology, labels, and global indices in $\mathrm { t } _ { n , r }$ are fixed. Each numerical leaf endpoint is $m _ { r j } ( Q ) + k R _ { n , r j } ( Q ) / ( b _ { n } + 1 )$ for a fixed integer $k \in \{ 0 , \ldots , b _ { n } + 1 \}$ . Finite empirical minima and maxima are continuous in Q, as are these endpoints and their midpoints. Composing with the continuous $h _ { 0 r }$ proves continuity of the ordered vector $\beta _ { n , r } ( Q )$ . Its norm bounds follow from $\| h _ { 0 r } \| _ { \infty } \leq C _ { h }$ This argument concerns the coeficient centers, not continuity of hard routing. □

Remark B.2.6. The side-length conclusion in Proposition B.2.5(ii) is the regular-split-net property used in earlier BART approximation arguments, including Jeong and Ročková (2023). Here it is not imposed through a separate abstract definition; it follows from the explicit uniform split-net construction.

## B.2.6 Auxiliary inputs

We collect the probabilistic inputs used later. The inputs are a Gaussian Euclidean-ball bound, a tree-size tail, an active-row Haar bound on $\mathrm { S O } ( p )$ , and a binary sign-pattern bound. Exponentially powerful local tests for empirical $L _ { 2 }$ loss and unknown Gaussian variance are supplied by Theorem 3 of Jeong (2025), which we quote in the form needed here.

Lemma B.2.7 (Gaussian Euclidean-ball lower bound). Let $B \sim N ( 0 , \Sigma )$ in $\mathbb { R } ^ { m }$ , where every eigenvalue of Σ lies in $[ \underline { { \lambda } } , \overline { { \lambda } } ]$ with $0 < \underline { { \lambda } } \le \overline { { \lambda } } < \infty$ . Then there exists a constant $C > 0$ , depending only on $( \underline { { \lambda } } , \overline { { \lambda } } )$ , such that for every $b \in \mathbb { R } ^ { m }$ and every $r \in ( 0 , 1 )$ ，

$$
\begin{array} { r } { - \log \operatorname* { P r } ( \left\| B - b \right\| _ { 2 } \leq r ) \leq C \{ m \log ( m / r ) + \left\| b \right\| _ { 2 } ^ { 2 } + 1 \} . } \end{array}
$$

Proof. Write $\Sigma ^ { 1 / 2 }$ for the symmetric square root of Σ and set $Z = \Sigma ^ { - 1 / 2 } B \sim N ( 0 , I _ { m } )$ and $\mu = \Sigma ^ { - 1 / 2 } b$ Take $c _ { 1 } = \underline { { \lambda } } ^ { - 1 / 2 }$ and $c _ { 2 } = \operatorname* { m a x } \{ 1 , \sqrt { \bar { \lambda } } \}$ , so that $\left\| \Sigma ^ { 1 / 2 } u \right\| _ { 2 } \leq c _ { 2 } \left\| u \right\| _ { 2 }$ and $\left\| \Sigma ^ { - 1 / 2 } u \right\| _ { 2 } \leq c _ { 1 } \left\| u \right\| _ { 2 }$ . Hence

$$
\operatorname* { P r } ( \| B - b \| _ { 2 } \leq r ) \geq \operatorname* { P r } ( \| Z - \mu \| _ { 2 } \leq r / c _ { 2 } ) .
$$

Let $\rho = r / c _ { 2 } \in ( 0 , 1 )$ . On $\left\{ z : \left\| z \right\| _ { 2 } \leq \rho \right\}$

$$
\begin{array} { r } { \varphi _ { m } ( z + \mu ) \ge ( 2 \pi ) ^ { - m / 2 } \exp \{ - ( \| \mu \| _ { 2 } + \rho ) ^ { 2 } / 2 \} . } \end{array}
$$

The ball volume is $\pi ^ { m / 2 } \rho ^ { m } / \Gamma ( m / 2 + 1 )$ . Consequently, $- \log \mathrm { P r } ( \| B - b \| _ { 2 } \leq r ) \leq ( m / 2 ) \log 2 + \log \Gamma ( m / 2 +$ $1 ) + m \log ( c _ { 2 } / r ) + ( \| \mu \| _ { 2 } + \rho ) ^ { 2 } / 2$ . Applying Stirling’s bound to the same density–volume calculation yields

$$
\begin{array} { r } { - \log \operatorname* { P r } ( \left\| Z - \mu \right\| _ { 2 } \leq \rho ) \leq C _ { 1 } \{ m \log ( m / \rho ) + \left\| \mu \right\| _ { 2 } ^ { 2 } + 1 \} . } \end{array}
$$

The claim follows from $\left\| \mu \right\| _ { 2 } ^ { 2 } \lesssim \left\| b \right\| _ { 2 } ^ { 2 }$ and $\rho \asymp r$

Proposition B.2.8 (Gaussian contraction criterion with unknown variance, quoted). For a regression function $f ,$ let

$$
\mu _ { f } = ( f ( x _ { 1 , n } ) , \ldots , f ( x _ { n , n } ) ) ^ { \top } \in \mathbb { R } ^ { n } ,
$$

and define

$$
d _ { G } ^ { 2 } ( ( \mu , \sigma ) , ( \mu ^ { \prime } , \sigma ^ { \prime } ) ) = \frac 1 n \left. \mu - \mu ^ { \prime } \right. _ { 2 } ^ { 2 } + | \sigma - \sigma ^ { \prime } | ^ { 2 } .\tag{B.10}
$$

Let $\widetilde { \varepsilon } _ { n } > 0$ satisfy $\widetilde { \varepsilon } _ { n } / \sigma _ { 0 }  0$ and $n \widetilde { \varepsilon } _ { n } ^ { 2 } / \sigma _ { 0 } ^ { 2 } \to \infty$ . Let $\overline { { \Pi } } _ { n }$ be a prior on $( \mu , \sigma ) \in \mathbb { R } ^ { n } \times ( 0 , \infty )$ and let $\overline { { \Theta } } _ { n }$ be a measurable sieve. Suppose that, for finite constants $C , D > 0$ and a suficiently large constant $E > 0$

$$
\overline { { \Pi } } _ { n } \left\{ \left. \mu - \mu _ { f _ { 0 } } \right. _ { 2 } ^ { 2 } \leq n \widetilde { \varepsilon } _ { n } ^ { 2 } , \quad \left. \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } \right. \leq \sigma _ { 0 } \widetilde { \varepsilon } _ { n } \right\} \geq \exp \left\{ - C \frac { n \widetilde { \varepsilon } _ { n } ^ { 2 } } { \sigma _ { 0 } ^ { 2 } } \right\} ,
$$

$$
\log N ( \widetilde { \varepsilon } _ { n } , \overline { { \Theta } } _ { n } , d _ { G } ) \leq D \frac { n \widetilde { \varepsilon } _ { n } ^ { 2 } } { \sigma _ { 0 } ^ { 2 } } , \qquad \overline { { \Pi } } _ { n } ( \overline { { \Theta } } _ { n } ^ { c } ) \leq \exp \left\{ - E \frac { n \widetilde { \varepsilon } _ { n } ^ { 2 } } { \sigma _ { 0 } ^ { 2 } } \right\} .
$$

Then there exists $M > 0$ such that

$$
\mathbb { E } _ { 0 } \overline { { \Pi } } _ { n } ( d _ { G } ( ( \mu , \sigma ) , ( \mu _ { f _ { 0 } } , \sigma _ { 0 } ) ) > M \widetilde { \varepsilon } _ { n } \mid Y ^ { ( n ) } )  0 .
$$

This is Theorem $\mathcal { B }$ of Jeong (2025), specialized to the fixed-design mean vector µ . In its proof, the required $\mu _ { f }$ tail exponent E depends on the local prior-mass constant and local Gaussian bounds, not on the entropy constant D. Once the sieve is fixed, D enters the choice of the final multiplier $M .$

Lemma B.2.9 (Exact Gaussian Kullback–Leibler formulas). For model (B.1),

$$
\mathrm { K L } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) = \frac { n } { 2 } \left\{ \log \frac { \sigma ^ { 2 } } { \sigma _ { 0 } ^ { 2 } } + \frac { \sigma _ { 0 } ^ { 2 } } { \sigma ^ { 2 } } - 1 + \frac { \| f - f _ { 0 } \| _ { n } ^ { 2 } } { \sigma ^ { 2 } } \right\} ,
$$

and the centered second moment of the log-likelihood ratio is

$$
\mathrm { V } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) = \frac { n } { 2 } \left( \frac { \sigma _ { 0 } ^ { 2 } } { \sigma ^ { 2 } } - 1 \right) ^ { 2 } + \frac { n \sigma _ { 0 } ^ { 2 } } { \sigma ^ { 4 } } \left\| f - f _ { 0 } \right\| _ { n } ^ { 2 } .
$$

Consequently, in a fixed neighborhood of $\sigma _ { 0 } > 0$

$$
\mathrm { K L } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) + \mathrm { V } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) \lesssim n \{ \| f - f _ { 0 } \| _ { n } ^ { 2 } + | \sigma - \sigma _ { 0 } | ^ { 2 } \} .
$$

Proof. The formulas follow by expanding the Gaussian log-likelihood ratio observation by observation. They are also the direct calculations used in the proof of Theorem 3 of Jeong (2025). The final inequality follows by Taylor expansion around $\sigma = \sigma _ { 0 }$ □

Lemma B.2.10 (Tree-size tail under the ν-depth prior). Under (P3), there are constants $c _ { K } > 0$ and $k _ { 0 } \geq 2 _ { : }$ , depending only on $\nu ,$ such that for every $n ,$ tree t, rotation $Q _ { t } \in \mathrm { S O } ( p )$ , and integer $k \geq k _ { 0 }$ ,

$$
\mathbb { P } _ { n } ( K _ { t } > k \mid Q _ { t } ) \le e ^ { - c _ { K } k \log k } .
$$

The same bound holds after integrating over $Q _ { t }$

Proof. Fix n and $Q _ { t }$ . Every realized tree is a finite full binary tree: along any path, a coordinate cannot reuse a cutpoint, so the depth is at most $p b _ { n }$ . Fix a full binary tree shape τ with $K$ leaves. Let $I ( \tau )$ be the internal nodes, let $L ( \tau )$ be the leaves, and let $d ( u )$ be the depth of a node u. Pre-generate independent split-attempt coins at all potential nodes. Realizing this shape requires a successful coin at each of its prescribed internal nodes; unavailable labels can only terminate a path earlier. Dropping all other requirements bounds the probability of this unlabeled shape by

$$
\prod _ { u \in I ( \tau ) } \nu ^ { d ( u ) + 1 } = \nu ^ { \sum _ { u \in I ( \tau ) } ( d ( u ) + 1 ) } .
$$

Let $\begin{array} { r } { I _ { \tau } = \sum _ { u \in I ( \tau ) } d ( u ) } \end{array}$ and $\begin{array} { r } { E _ { \tau } = \sum _ { v \in L ( \tau ) } d ( v ) } \end{array}$ . For full binary trees, $E _ { \tau } = I _ { \tau } + 2 ( K - 1 )$ . If the leaf depths are $d _ { 1 } , \ldots , d _ { K }$ , Kraft’s equality gives $\begin{array} { r } { \sum _ { j = 1 } ^ { K } 2 ^ { - d _ { j } } = 1 } \end{array}$ , and Jensen’s inequality implies $E _ { \tau } \geq K \log _ { 2 } K$ Therefore

$$
\sum _ { u \in I ( \tau ) } \left( d ( u ) + 1 \right) = I _ { \tau } + K - 1 = E _ { \tau } - K + 1 \geq K \log _ { 2 } K - K + 1 .
$$

The number of full binary tree shapes with K leaves is the Catalan number $C _ { K - 1 } \leq 4 ^ { K - 1 }$ . Hence

$$
\mathbb { P } _ { n } ( K _ { t } = K \mid Q _ { t } ) \le 4 ^ { K - 1 } \nu ^ { K \log _ { 2 } K - K + 1 } \le e ^ { - c K \log K }
$$

for all suficiently large K, with constants depending only on ν. Summing over integers $K > k$ and decreasing the exponential constant gives the conditional tail bound uniformly in n and $Q _ { t }$ . Integrating over $Q _ { t }$ gives its marginal version. □

Lemma B.2.11 (Active-row Haar small ball). Let $Q \sim$ Haar $( \mathrm { S O } ( p ) )$ , let $S \subset \{ 1 , \ldots , p \}$ with $| S | = d _ { \cdot }$ , and assume $p \geq 2 , 1 \leq d \leq p$ . For every fixed $Q _ { 0 } \in \mathrm { S O } ( p )$ and all suficiently small $r \in ( 0 , 1 )$ ,

$$
- \log \operatorname* { P r } \left( \| Q _ { S , \cdot } - Q _ { 0 , S , \cdot } \| _ { F } \leq r \right) \leq C \left\{ D _ { p , d } \log ( C / r ) + D _ { p , d } \log ( e D _ { p , d } ) \right\}
$$

where

$$
D _ { p , d } = d p - \frac { d ( d + 1 ) } { 2 } .
$$

In particular, when p and d are fixed and $\log ( 1 / r ) \lesssim \log n$ , the negative logarithm is $O ( \log n )$

Proof. Let H and $H _ { + }$ denote normalized Haar measure on $O ( p )$ and $\mathrm { S O } ( p )$ , respectively, and put $U _ { 0 } = Q _ { 0 , S } ,$ Let $G \in \mathbb { R } ^ { d \times p }$ have independent $N ( 0 , 1 )$ entries and define

$$
U ( G ) = ( G G ^ { \top } ) ^ { - 1 / 2 } G \in \mathrm { S t } ( d , p ) , \qquad \mathrm { S t } ( d , p ) = \{ U \in \mathbb { R } ^ { d \times p } : U U ^ { \top } = I _ { d } \} .
$$

The matrix $G$ has full row rank almost surely. For every $R \in O ( p ) , G R { \stackrel { d } { = } } G$ and $U ( G R ) = U ( G ) R$ . The right action of $O ( p )$ on $\operatorname { S t } ( d , p )$ is transitive, so $U ( G )$ has the invariant Stiefel distribution, which is also the law of $Q _ { S } ,$ <sub>·</sub> under H.

Write $G = U _ { 0 } + E$ and $A = G G ^ { \top }$ . Since $\| U _ { 0 } \| _ { \mathrm { o p } } = 1$

$$
A - I _ { d } = U _ { 0 } E ^ { \top } + E U _ { 0 } ^ { \top } + E E ^ { \top } , \qquad \left\| A - I _ { d } \right\| _ { F } \leq 2 \left\| E \right\| _ { F } + \left\| E \right\| _ { F } ^ { 2 } .
$$

For $\| E \| _ { F }$ below a fixed numerical constant, the eigenvalues of A lie in $[ 1 / 2 , 3 / 2 ]$ . The identity

$$
A ^ { - 1 / 2 } - I _ { d } = A ^ { - 1 / 2 } ( I _ { d } - A ) ( I _ { d } + A ^ { 1 / 2 } ) ^ { - 1 }
$$

gives $\left\| A ^ { - 1 / 2 } - I _ { d } \right\| _ { F } \leq C \left\| E \right\| _ { F }$ . Hence

$$
\begin{array} { r l } & { \left\| { U ( G ) - U _ { 0 } } \right\| _ { F } = \left\| { ( A ^ { - 1 / 2 } - I _ { d } ) U _ { 0 } } + A ^ { - 1 / 2 } E \right\| _ { F } } \\ & { \qquad \leq \left\| { A ^ { - 1 / 2 } - I _ { d } } \right\| _ { F } + \sqrt { 2 } \left\| { E } \right\| _ { F } \leq C \left\| { E } \right\| _ { F } . } \end{array}
$$

Thus, for a fixed numerical $c \in ( 0 , 1 )$ and all suficiently small $r ,$

$$
H \{ Q : \| Q _ { S , \cdot } - Q _ { 0 , S , \cdot } \| _ { F } \leq r \} \geq \operatorname* { P r } ( \| G - U _ { 0 } \| _ { F } \leq c r ) .
$$

Vectorize G in $\mathbb { R } ^ { d p }$ . Its covariance is the identity, $\begin{array} { r } { \left\| U _ { 0 } \right\| _ { F } ^ { 2 } = d , } \end{array}$ and $c r < 1$ . Lemma B.2.7 therefore yields

$$
- \log \operatorname* { P r } ( \| G - U _ { 0 } \| _ { F } \leq c r ) \leq C \{ d p \log ( d p / ( c r ) ) + d + 1 \} .
$$

Since

$$
D _ { p , d } \geq \frac { d ( p - 1 ) } { 2 } , \qquad d p \leq \frac { 2 p } { p - 1 } D _ { p , d } \leq 4 D _ { p , d } ,
$$

and $d + 1 \le C D _ { p , d }$ for $p \geq 2$ , this proves the asserted bound under H.

Left multiplication by a reflection preserves H and exchanges its two determinant components, each of mass $1 / 2 ;$ its normalized restriction to $\mathrm { S O } ( p )$ is $H _ { + }$ . If $d < p ,$ , choose $j \not \in S$ and put $D = I _ { p } - 2 e _ { j } e _ { j } ^ { \intercal }$ Multiplication by D exchanges those components and leaves $Q _ { S , }$ unchanged. Consequently,

$$
H _ { + } \{ Q : \lVert Q _ { S , \cdot } - Q _ { 0 , S , \cdot } \rVert _ { F } \leq r \} = H \{ Q : \lVert Q _ { S , \cdot } - Q _ { 0 , S , \cdot } \rVert _ { F } \leq r \} .
$$

If $d = p$ and det $Q = - 1$ , then $Q Q _ { 0 } ^ { \top }$ is orthogonal with determinant −1. Its nonreal eigenvalues occur in conjugate pairs, so it has an eigenvalue −1 and

$$
\begin{array} { r } { \left\| \boldsymbol { Q } - \boldsymbol { Q } _ { 0 } \right\| _ { F } = \left\| \boldsymbol { Q } \boldsymbol { Q } _ { 0 } ^ { \top } - \boldsymbol { I } _ { p } \right\| _ { F } \geq \left\| \boldsymbol { Q } \boldsymbol { Q } _ { 0 } ^ { \top } - \boldsymbol { I } _ { p } \right\| _ { \mathrm { o p } } \geq 2 . } \end{array}
$$

For $r < 1 \AA$ , the full-matrix ball lies in $\mathrm { S O } ( p )$ , and its $H _ { + } { \mathrm { - p r o b a b i l i t y } }$ is twice its H-probability. Thus the bound transfers to $H _ { + }$ in both cases. For fixed $p , d ,$ it is $O \{ \log ( 1 / r ) \}$ , giving the final assertion. □

Lemma B.2.12 (Linear sign-pattern bound). Let $m , k \geq 1$ and let $a _ { 1 } , \ldots , a _ { m }$ be afine functions on $\mathbb { R } ^ { k }$ For any $A \subset \mathbb { R } ^ { k }$ , the number of distinct binary vectors $( \mathbf { 1 } \{ a _ { i } ( z ) \geq 0 \} ) _ { i = 1 } ^ { m } , z \in A _ { }$ , is at most $\Sigma _ { j = 0 } ^ { \operatorname* { m i n } ( m , k ) } \left( { m \atop j } \right)$ In particular, it is at most $( e m / k ) ^ { k }$ when $m \geq k$ , and at most $2 ^ { m }$ when m $< k$ . Consequently, $i f m \le n ^ { C } k$ for a fixed $C > 0$ , then the logarithm of this number is $O ( k \log n ) ~ f o r ~ n \ge 2$

Proof. The claim is immediate if A is empty. Otherwise, choose one point of A for each realized binary vector. There are finitely many such points. Choose $\epsilon > 0$ smaller than the absolute value of every nonzero $a _ { i }$ at these points, with arbitrary $\epsilon > 0$ if all these values are zero. At every chosen point, all $a _ { i } + \epsilon$ are nonzero, and ${ \bf 1 } \{ a _ { i } \geq 0 \} = { \bf 1 } \{ a _ { i } + \epsilon > 0 \}$ . Thus the original binary vectors are realized as strict sign patterns of one fixed perturbed afine arrangement.

Let $N ( m , k )$ be the maximal number of regions determined by at most m afine hyperplanes in $\mathbb { R } ^ { k }$ . Adding one hyperplane creates at most $N ( m - 1 , k - 1 )$ new regions, so $N ( m , k ) \leq N ( m - 1 , k ) + N ( m - 1 , k - 1 )$ With $N ( 0 , k ) = N ( m , 0 ) = 1$ , induction gives $\begin{array} { r } { N ( m , k ) \leq \sum _ { j = 0 } ^ { \operatorname* { m i n } ( m , k ) } \binom { m } { j } } \end{array}$ . Each strict sign pattern is a region, which proves the first bound. For $m \geq k , \sum _ { i = 0 } ^ { k } { \binom { m } { i } } \leq ( m / k ) ^ { k } ( 1 + k / m ) ^ { m } \leq ( e m / k ) ^ { k }$ ; for $m < k$ , the sum equals $2 ^ { m }$ . The logarithmic bound follows by considering these two cases. □

## B.3 Main theorem

The following is the posterior contraction theorem of the main paper, restated with the prior and design conditions above.

Theorem B.3.1 (Fixed-dimensional additive posterior contraction for RoBART). Assume (B.2), (B.3), the componentwise Hölder and boundedness conditions in Definition $B . { \mathcal { Q } } . 1 ,$ the prior conditions (P1)–(P5), and Assumption B.2.2. Let $\varepsilon _ { n }$ be given by (B.4). Then there exists $M > 0$ such that

$$
\mathbb { E } _ { 0 } \Pi _ { n } \left( \big \| f - f _ { 0 } \big \| _ { n } + | \sigma - \sigma _ { 0 } | > M \varepsilon _ { n } \ | \ Y ^ { ( n ) } \right) \to 0 .\tag{B.11}
$$

Consequently, because $\sigma _ { 0 }$ is fixed and strictly positive,

$$
\mathbb { E } _ { 0 } \Pi _ { n } \left( \big \| f - f _ { 0 } \big \| _ { n } + | \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | > M ^ { \prime } \varepsilon _ { n } \ | \ Y ^ { ( n ) } \right) \to 0
$$

for a suficiently large constant $M ^ { \prime } > 0$

Remark B.3.2 (Interpretation of the rate). For fixed R and component parameters, (B.4) gives

$$
\varepsilon _ { n } \asymp \operatorname* { m a x } _ { 1 \leq r \leq R } \left( \frac { \log n } { n } \right) ^ { \bar { \alpha } _ { r } / ( 2 \bar { \alpha } _ { r } + d _ { r } ) } .
$$

The factors involving $\lambda _ { r } d _ { r }$ are constants in $n ,$ while $d _ { r }$ remains in the exponents. Localizing the rotations costs $O ( \log n )$ . The prescribed uniform split labels contribute $\begin{array} { r } { \sum _ { r } ( K _ { n , r } - 1 ) \log p = O ( \sum _ { r } K _ { n , r } ) = o ( n \varepsilon _ { n } ^ { 2 } ) } \end{array}$ to the negative log bound in Proposition B.5.2. These costs are lower order than the tree approximation cost in (B.6). This is a function-level upper bound; it does not assert recovery of the rotations, active sets, or the possibly nonidentified additive decomposition.

## B.4 Approximation and local rotation control

## B.4.1 Active-row rotation perturbation

The first deterministic ingredient controls how much a component template changes when the active rows of its rotation matrix are perturbed.

Lemma B.4.1 (Componentwise sparse active-row rotation perturbation). For every $r \leq R$ and every $Q , Q ^ { \prime } \in \mathrm { S O } ( p )$ ,

$$
\| h _ { 0 r } ( \pi _ { S _ { r } } ( Q \cdot ) ) - h _ { 0 r } ( \pi _ { S _ { r } } ( Q ^ { \prime } \cdot ) ) \| _ { n } \leq C _ { r } \lambda _ { r } d _ { r } \left\| Q _ { S _ { r } , \cdot } - Q _ { S _ { r } , \cdot } ^ { \prime } \right\| _ { F } ^ { \alpha _ { r , \operatorname* { m i n } } } ,
$$

where $C _ { r }$ depends only on $M _ { X }$ and the fixed smoothness vector. In particular, $i f Q \in \mathcal { Q } _ { n , r }$ , then

$$
\| h _ { 0 r } ( \pi _ { S _ { r } } ( Q \cdot ) ) - h _ { 0 r } ( \pi _ { S _ { r } } ( Q _ { 0 r } \cdot ) ) \| _ { n } \lesssim \varepsilon _ { n , r } .
$$

Proof. For each design point $x _ { i , n }$ , Hölder continuity gives

$$
| h _ { 0 r } ( \pi _ { S _ { r } } ( Q x _ { i , n } ) ) - h _ { 0 r } ( \pi _ { S _ { r } } ( Q ^ { \prime } x _ { i , n } ) ) | \leq \lambda _ { r } \sum _ { k = 1 } ^ { d _ { r } } | e _ { s _ { r k } } ^ { \top } ( Q - Q ^ { \prime } ) x _ { i , n } | ^ { \alpha _ { r k } } .
$$

By Cauchy’s inequality and (B.2),

$$
\begin{array} { r } { | e _ { s _ { r k } } ^ { \top } ( Q - Q ^ { \prime } ) x _ { i , n } | \leq \| ( Q - Q ^ { \prime } ) _ { s _ { r k } , \cdot } \| _ { 2 } \| x _ { i , n } \| _ { 2 } \leq M _ { X } \left\| Q _ { S _ { r } , \cdot } - Q _ { S _ { r } , \cdot } ^ { \prime } \right\| _ { F } . } \end{array}
$$

Taking empirical norms bounds the diference in the lemma by $\begin{array} { r } { \lambda _ { r } \sum _ { k = 1 } ^ { d _ { r } } M _ { X } ^ { \alpha _ { r k } } \parallel ( Q - Q ^ { \prime } ) _ { s _ { r k } , \cdot } \parallel _ { 2 } ^ { \alpha _ { r k } } } \end{array}$ . Every selected row is a unit vector, so $\left\| Q _ { S _ { r } , \cdot } - Q _ { S _ { r } , \cdot } ^ { \prime } \right\| _ { F } \leq 2 \sqrt { d _ { r } }$ . For positive distance, each power $\alpha _ { r k }$ is bounded by the power $\alpha _ { r , \mathrm { { m i n } } }$ times $( 2 \sqrt { d _ { r } } ) ^ { \alpha _ { r k } - \alpha _ { r , \mathrm { m i n } } } ;$ at zero distance the claim is immediate. Hence the last bound is at most $ C _ { r } \lambda _ { r } d _ { r } \| Q _ { S _ { r } , \cdot } - \dot { Q } _ { S _ { r } , \cdot } ^ { \prime } \| _ { F } ^ { \alpha _ { r , \mathrm { m i n } } }$ , which proves the first claim. The second follows from (B.7) with ${ { c } _ { \delta , r } }$ suficiently small. □

## B.4.2 Approximation in the additive rotated coordinates

The approximation of component r is built on ${ \mathcal { P } } _ { n , r } ( Q )$ in the active empirical box $D _ { Q , r }$ . Coordinates outside $S _ { r }$ are left unsplit. Applying the same split rules beyond this box extends $\widehat { h } _ { r , Q }$ to $\mathbb { R } ^ { d _ { r } }$ and gives the corresponding p-dimensional tree function after rotation. Only values at the design points enter the approximation bound.

Lemma B.4.2 (Componentwise rotated anisotropic approximation). Under Assumption B.2.2, let $\widehat { h } _ { r , Q }$ be the step function in Proposition B.2.5. For all suficiently large $n ,$ every $r \leq R ,$ and every $Q \in \mathcal { Q } _ { n , r } ,$

$$
\left\| h _ { 0 r } ( \pi _ { S _ { r } } ( Q _ { 0 r } . ) ) - \widehat { h } _ { r , Q } ( \pi _ { S _ { r } } ( Q \cdot ) ) \right\| _ { n } \lesssim \varepsilon _ { n , r } .
$$

Proof. By Proposition $\mathrm { B } . 2 . 5 ( \mathrm { i i i } )$ , the uniform approximation error on $D _ { Q , r }$ <sub>r</sub> is $O ( \varepsilon _ { n , r } )$ . For every design point, $\pi _ { S _ { r } } ( Q x _ { i , n } ) \in D _ { Q , r }$ by definition, and hence

$$
\begin{array} { r } { \left\| h _ { 0 r } ( \pi _ { S _ { r } } ( Q \cdot ) ) - \widehat { h } _ { r , Q } ( \pi _ { S _ { r } } ( Q \cdot ) ) \right\| _ { n } \leq \left\| h _ { 0 r } - \widehat { h } _ { r , Q } \right\| _ { \infty , D _ { Q , r } } \lesssim \varepsilon _ { n , r } . } \end{array}
$$

Lemma B.4.1 controls the diference between Q and $Q _ { 0 r }$ . The triangle inequality proves the claim. □

For every tuple $( Q _ { 1 } , \ldots , Q _ { R } )$ with $Q _ { r } \in \mathcal { Q } _ { n , r }$ , summing Lemma B.4.2 over the fixed number of components gives

$$
\bigg \vert \bigg \vert f _ { 0 } - \sum _ { r = 1 } ^ { R } \widehat { h } _ { r , Q _ { r } } ( \pi _ { S _ { r } } ( Q _ { r } \cdot ) ) \bigg \vert \bigg \vert _ { n } \lesssim \sum _ { r = 1 } ^ { R } \varepsilon _ { n , r } = \varepsilon _ { n } .
$$

## B.5 Kullback–Leibler prior mass

The prior-mass lower bound is built from a good set in which the first R trees carry the R additive components and the remaining $T - R$ trees are nearly null. This construction exploits the redundancy of the forest without requiring every tree-specific rotation to be close to a truth frame.

## B.5.1 An R-signal-tree good set

For $r \leq R ,$ , use the fixed state $\mathbf { t } _ { n , r } ,$ partition ${ \mathcal { P } } _ { n , r } ( Q )$ , and ordered coeficient center $\beta _ { n , r } ( Q )$ from Proposition B.2.5. In particular,

$$
\begin{array} { r } { \| \beta _ { n , r } ( Q ) \| _ { \infty } \leq C _ { h } , \qquad \| \beta _ { n , r } ( Q ) \| _ { 2 } ^ { 2 } \leq C _ { h } ^ { 2 } K _ { n , r } . } \end{array}\tag{B.12}
$$

Use the same left-to-right leaf order for $M _ { r }$ and $\beta _ { n , r } ( Q _ { r } )$ . Define the nested events

$$
A _ { Q } = \bigcap _ { r = 1 } ^ { R } \{ Q _ { r } \in { \mathcal { Q } } _ { n , r } \} , \qquad A _ { T } = A _ { Q } \cap \bigcap _ { r = 1 } ^ { R } \{ ( { \mathcal { T } } _ { r } , V _ { r } , C _ { r } ) = \mathbf { t } _ { n , r } \} ,
$$

$$
A _ { M } = A _ { T } \cap \bigcap _ { r = 1 } ^ { R } \{ \| M _ { r } - \beta _ { n , r } ( Q _ { r } ) \| _ { 2 } \leq c _ { 1 } \varepsilon _ { n , r } \} .
$$

The coeficient comparisons are evaluated only on $A _ { T } \mathrm { . }$ , so their centers are defined and their dimensions agree. For $t > R$ , let $\mu _ { t }$ denote the unique leaf mean when $K _ { t } = 1$ , and set

$$
A _ { \mathrm { n u l l } } = \bigcap _ { t = R + 1 } ^ { T } \{ K _ { t } = 1 , \ | \mu _ { t } | \leq c _ { 2 } \varepsilon _ { n } / T \} , \qquad A _ { \sigma } = \{ | \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | \leq c _ { 3 } \sigma _ { 0 } \varepsilon _ { n } \} .
$$

Finally, let $A _ { n } = A _ { M } \cap A _ { \mathrm { n u l l } } \cap A _ { \sigma }$ . All constants $c _ { 1 } , c _ { 2 } , c _ { 3 }$ are fixed and positive. Proposition $\mathrm { B . 2 . 5 ( i v ) }$ makes these events measurable. Empty intersections equal the whole generating space when $T = R$

Proposition B.5.1 (Approximation on the good set). For suficiently small fixed constants $c _ { 1 } , c _ { 2 } , c _ { 3 } > 0$ and all suficiently large n, every ensemble in $A _ { n }$ induces a pair $( f , \sigma )$ satisfying

$$
\| f - f _ { 0 } \| _ { n } \lesssim \varepsilon _ { n } , \qquad | \sigma - \sigma _ { 0 } | \lesssim \varepsilon _ { n } .
$$

Consequently,

$$
\mathrm { K L } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) + \mathrm { V } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) \lesssim n \varepsilon _ { n } ^ { 2 } .
$$

Proof. Write

$$
f ( x ) = \sum _ { r = 1 } ^ { R } g _ { r } ( Q _ { r } x ) + \sum _ { t = R + 1 } ^ { T } \mu _ { t } ,
$$

where $g _ { r }$ uses the discrete state $\mathrm { t } _ { n , r }$ under $Q _ { r }$ <sub>r</sub> and leaf vector $M _ { r }$ . Then

$$
\begin{array} { r l r } {  { \| f - f _ { 0 } \| _ { n } \le \sum _ { r = 1 } ^ { R } \Big \| g _ { r } ( Q _ { r } \cdot ) - \widehat { h } _ { r , Q _ { r } } ( \pi _ { S _ { r } } ( Q _ { r } \cdot ) ) \Big \| _ { n } } } \\ & { } & { + \sum _ { r = 1 } ^ { R } \Big \| \widehat { h } _ { r , Q _ { r } } ( \pi _ { S _ { r } } ( Q _ { r } \cdot ) ) - h _ { 0 r } ( \pi _ { S _ { r } } ( Q _ { 0 r } \cdot ) ) \Big \| _ { n } + \sum _ { t = R + 1 } ^ { T } | \mu _ { t } | . } \end{array}
$$

For the first sum, let $d _ { M , r } = M _ { r } - \beta _ { n , r } ( Q _ { r } )$ and let $n _ { r , k }$ be the number of design points falling in leaf $k .$ Since the two functions live on the same partition,

$$
\left\| g _ { r } ( Q _ { r } \cdot ) - \widehat { h } _ { r , Q _ { r } } ( \pi _ { S _ { r } } ( Q _ { r } \cdot ) ) \right\| _ { n } ^ { 2 } = \frac { 1 } { n } \sum _ { k = 1 } ^ { K _ { n , r } } n _ { r , k } d _ { M , r , k } ^ { 2 } \leq \| M _ { r } - \beta _ { n , r } ( Q _ { r } ) \| _ { 2 } ^ { 2 } .
$$

The second sum is $O ( \sum _ { r } \varepsilon _ { n , r } )$ by Lemma B.4.2. The null-tree sum is $O ( \varepsilon _ { n } )$ because $T$ is fixed. This proves the function bound. On $A _ { \sigma } , | \sigma - \sigma _ { 0 } | = | \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | / ( \sigma + \sigma _ { 0 } ) \leq c _ { 3 } \varepsilon _ { n }$ . Lemma B.2.9 then gives the final assertion. □

Proposition B.5.2 (Prior mass of the good set). Under Assumption B.2.2, there exists $C > 0$ such that

$$
\mathbb { P } _ { n } ( A _ { n } ) \geq e ^ { - C n \varepsilon _ { n } ^ { 2 } }
$$

for all suficiently large n.

Proof. Fix $r \leq R$ and $Q _ { r } \in \mathcal { Q } _ { n , r }$ . The state $\mathrm { t } _ { n , r }$ has $K _ { n , r } = 2 ^ { L _ { n , r } }$ leaves and $K _ { n , r } - 1$ internal nodes. Every prescribed label and cut is available by Proposition B.2.5(ii). In the discrete mass formula of Section B.2.3, $a _ { \eta } \leq b _ { n }$ and each terminal factor is at least $1 - \nu .$ . The complete-depth topology satisfies $\begin{array} { r l } { ~ } & { { } \sum _ { \ell = 0 } ^ { L _ { n , r } - 1 } ( \ell + 1 ) 2 ^ { \ell } = } \end{array}$ $( \dot { L _ { n , r } } - 1 ) K _ { n , r } + 1$ . Hence $\mathbb { P } _ { n } \{ ( \mathcal { T } _ { r } , V _ { r } , C _ { r } ) = \mathbf { t } _ { n , r } ~ \vert ~ Q _ { r } \} \geq ( \bar { 1 } - \nu ) ^ { K _ { n , r } } \nu ^ { ( L _ { n , r } ^ { - } - 1 ) K _ { n , r } + 1 } ( p b _ { n } ) ^ { - ( K _ { n , r } - 1 ) }$ . Taking negative logarithms gives − log ${  { \mathbb P } } _ { n } \{ ( \mathcal T _ { r } , V _ { r } , C _ { r } ) = \mathfrak t _ { n , r } \ | \ Q _ { r } \} \le C K _ { n , r } ( L _ { n , r } + \log b _ { n } + 1 ) + ( K _ { n , r } - 1 ) \log p \le C K _ { n , r } ( L _ { n , r } + \log b _ { n } + 1 ) .$ $C K _ { n , r } \log n$ , using fixed p and (D2).

Given the actual rotations and discrete states, (P4) gives $M _ { r } \sim N ( 0 , \tau ^ { 2 } I _ { K _ { n . r } } )$ on the prescribed state. By (B.12) and Lemma B.2.7, $\begin{array} { r } { \mathbb { P } _ { n } \{ \| M _ { r } - \beta _ { n , r } ( Q _ { r } ) \| _ { 2 } \le c _ { 1 } \varepsilon _ { n , r } \ | \ Q _ { r } , ( T _ { r } , V _ { r } , C _ { r } ) = \mathfrak { t } _ { n , r } \} \ge \exp ( - C K _ { n , r } \log n ) } \end{array}$ uniformly over $Q _ { r } \in \mathcal { Q } _ { n , r }$ . Here log $K _ { n , r } + \log ( 1 / \varepsilon _ { n , r } ) = O ( \log n )$ , and the fixed conditional Gaussian kernel depends on the structural state only through its leaf count.

Let H be normalized Haar measure on $\mathrm { S O } ( p )$ . To integrate the moving centers, tree independence gives $\begin{array} { r } { \mathbb { P } _ { n } ( A _ { M } ) = \int _ { \Pi _ { r } ^ { R } , \ Q _ { n , r } } \prod _ { r = 1 } ^ { R } \mathbb { P } _ { n } \{ ( \mathcal T _ { r } , V _ { r } , C _ { r } ) = \mathfrak { t } _ { n , r } , \ \lVert M _ { r } - \beta _ { n , r } ( Q _ { r } ) \rVert _ { 2 } \le c _ { 1 } \varepsilon _ { n , r } \ | \ Q _ { r } \} \ \prod _ { r = 1 } ^ { R } H ( d Q _ { r } ) } \end{array}$ where each coeficient comparison is evaluated only on the prescribed state. The two uniform bounds above imply $\begin{array} { r } { \mathbb { P } _ { n } ( A _ { M } ) \ge \exp \{ - C \sum _ { r = 1 } ^ { R } K _ { n , r } \log n \} \mathbb { P } _ { n } ( A _ { Q } ) } \end{array}$ . By (P1) and Lemma $\mathrm { B . 2 . 1 1 , ~ - l o g } \mathbb { P } _ { n } ( A _ { Q } ) \lesssim$ $\begin{array} { r } { \sum _ { r } \log ( 1 / \delta _ { n , r } ) \lesssim } \end{array}$ log n.

For $t > R ,$ , not attempting a root split gives $\mathbb { P } _ { n } ( K _ { t } = 1 \mid Q _ { t } ) \ge 1 - \nu .$ . For $Z \sim N ( 0 , \tau ^ { 2 } )$ and $0 < a \le 1$ , $\operatorname* { P r } ( | Z | \leq a ) \geq 2 a ( 2 \pi \tau ^ { 2 } ) ^ { - { \bar { 1 } } / 2 } \exp \{ - { \bar { 1 } } / ( { \bar { 2 \tau } } ^ { 2 } ) \}$ . With $a = c _ { 2 } \varepsilon _ { n } / T$ , this yields ${ \mathbb P } _ { n } \{ K _ { t } = 1 , ~ | \mu _ { t } | \le c _ { 2 } \varepsilon _ { n } / T$ $Q _ { t } \} \geq c \varepsilon _ { n }$ uniformly in $Q _ { t }$ . Integrating over these unrestricted rotations and using independence across trees gives $\mathbb { P } _ { n } ( A _ { \mathrm { n u l l } } ) \geq ( c \varepsilon _ { n } ) ^ { T - R }$ . The bound is one when $T = R$

Let $p _ { V }$ be the prior density of $V = \sigma ^ { 2 }$ . For suficiently large $n ,$ the variance interval in $A _ { \sigma }$ lies in $[ \sigma _ { 0 } ^ { 2 } / 2 , 3 \sigma _ { 0 } ^ { 2 } / 2 ]$ Positivity and continuity of $p _ { V }$ give $\begin{array} { r } { \mathbb { P } _ { n } ( A _ { \sigma } ) \ge 2 c _ { 3 } \sigma _ { 0 } \varepsilon _ { n } \operatorname* { i n f } _ { v \in [ \sigma _ { 0 } ^ { 2 } / 2 , 3 \sigma _ { 0 } ^ { 2 } / 2 ] } p _ { V } ( v ) \ge c \varepsilon _ { n } } \end{array}$

The signal-tree event, the remaining-tree event, and the noise event concern independent prior blocks under $( \mathrm { P 1 } ) \mathrm { - } ( \mathrm { P 5 } )$ . Multiplying the established bounds, − log $\begin{array} { r } { \mathbb { P } _ { n } ( A _ { n } ) \le C \sum _ { r = 1 } ^ { R } K _ { n , r } \log \bar { n } + C \log n \lesssim n \varepsilon _ { n } ^ { 2 } } \end{array}$ , by (B.6) and (B.8). □

Corollary B.5.3 (Kullback–Leibler and Gaussian local prior mass). There exists a constant $A _ { 0 } \geq 1$ such that

$$
\Pi _ { n } \left\{ \left. f - f _ { 0 } \right. _ { n } \leq A _ { 0 } \varepsilon _ { n } , \quad \left. \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } \right. \leq \sigma _ { 0 } A _ { 0 } \varepsilon _ { n } \right\} \geq e ^ { - C n \varepsilon _ { n } ^ { 2 } } .
$$

The same set is contained in a Kullback–Leibler neighborhood satisfying

$$
\mathrm { K L } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) \vee \vee ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) \le C n \varepsilon _ { n } ^ { 2 } .
$$

Proof. Proposition B.5.1 gives $\| f - f _ { 0 } \| _ { n } \leq C \varepsilon _ { n }$ on $A _ { n }$ , while $A _ { \sigma }$ directly gives $| \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | \le c _ { 3 } \sigma _ { 0 } \varepsilon _ { n }$ . Choose $A _ { 0 } \geq \operatorname* { m a x } \{ 1 , C , c _ { 3 } \}$ . The good set lies in the preimage of the displayed neighborhood, so Proposition B $. 5 . 2$ gives the stated bound for the induced prior $\Pi _ { n } .$ For every pair in that entire neighborhood, $| \sigma - \sigma _ { 0 } | =$ $\bar { | } \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | / ( \sigma + \sigma _ { 0 } ) \le A _ { 0 } \varepsilon _ { n }$ . This places $\sigma$ in a fixed positive neighborhood of $\sigma _ { 0 }$ for all suficiently large $n ,$ and Lemma B.2.9 gives both remaining bounds. □

## B.6 Direct entropy of the additive rotated forest sieve

For fixed topology and split labels, empirical routing is determined by afine comparisons in the tree-specific rotation matrices and thresholds. We count these routing patterns and then cover the leaf coeficients.

## B.6.1 The sieve

Let $\begin{array} { r } { K _ { + } = \sum _ { t = 1 } ^ { T } K _ { t } } \end{array}$ be the total number of leaves across the forest. Define

$$
\Theta _ { n } = \mathcal { F } ^ { \mathrm { r o t } } ( \bar { K } _ { n } , M _ { n } ) \times I _ { n } ,
$$

where

$$
\bar { K } _ { n } = \left\lceil M _ { K } \frac { n \varepsilon _ { n } ^ { 2 } } { \log n } \right\rceil , \qquad M _ { n } = n ^ { A _ { \mu } } , \qquad I _ { n } = [ n ^ { - A _ { \sigma } } , e ^ { A _ { \sigma } n \varepsilon _ { n } ^ { 2 } } ] ,
$$

and

$$
\mathcal { F } ^ { \mathrm { r o t } } ( \bar { K } _ { n } , M _ { n } ) = \left\{ f ( x ) = \sum _ { t = 1 } ^ { T } g _ { t } ( Q _ { t } x ) : Q _ { t } \in \mathrm { S O } ( p ) , \ K _ { + } \leq \bar { K } _ { n } , \ \operatorname* { m a x } _ { t \leq T } \left\| M _ { t } \right\| _ { \infty } \leq M _ { n } \right\} .
$$

The coeficient constraint bounds every leaf mean. Membership means that at least one representation satisfying these bounds and the declared grid and admissibility rules exists. No split-label support restriction is needed because p is fixed.

For use with Proposition B.2.8, define the image sieve

$$
\overline { { \Theta } } _ { n } = \{ ( \mu _ { f } , \sigma ) : ( f , \sigma ) \in \Theta _ { n } \} .
$$

Let $\Re _ { n }$ be the finite set of routing configurations realized by this sieve, including topology and leaf order. For $\mathcal { R } \in \Re _ { n }$ , stack the leaf vectors $M _ { t }$ in tree-by-tree order as $\beta ,$ and let $H _ { \mathcal R }$ be the leaf-incidence matrix. Its $( i , ( t , q ) )$ ) entry is one exactly when observation i reaches leaf q of tree t. Thus $\mu _ { f } = H _ { \mathcal { R } } \beta$ , and $H _ { \mathcal { R } }$ has $K _ { + } ( \mathcal { R } )$ columns. The image sieve is exactly $\begin{array} { r } { \overline { { \Theta } } _ { n } = \bigcup _ { \mathcal { R } \in \Re _ { n } } \{ H _ { \mathcal { R } } [ - M _ { n } , M _ { n } ] ^ { K _ { + } ( \mathcal { R } ) } \} ^ { \bullet } \times I _ { n } \colon } \end{array}$ : every coeficient vector in a cube can be paired with a fixed valid realization of that routing. This finite union of compact linear images, crossed with $I _ { n } .$ , is compact and Borel measurable. No continuity of hard routing in $Q _ { t }$ is used.

## B.6.2 Routing maps generated by shared rotations

Fix a forest topology with $I _ { + }$ internal nodes and $K _ { + }$ leaves, where $I _ { + } = K _ { + } - T \le \bar { K } _ { n }$ . Also fix a split-label sequence $j ( \ell ) \in \{ 1 , \ldots , p \}$ for internal nodes $\ell = 1 , \ldots , I _ { + }$ . For node $\ell ,$ let $t ( \ell )$ denote the tree containing it. Conditional on topology and labels, the routing of the design points is determined by the binary indicators

$$
\begin{array} { r } { { \bf 1 } \{ e _ { j ( \ell ) } ^ { \top } Q _ { t ( \ell ) } x _ { i , n } - \tau _ { \ell } \geq 0 \} , \qquad i = 1 , \ldots , n , \quad \ell = 1 , \ldots , I _ { + } , } \end{array}\tag{B.13}
$$

where $\tau _ { \ell }$ is the numerical threshold at node $\ell .$ Recording comparisons even at nodes that an observation does not visit can only enlarge the collection of routing descriptions.

The quantities $e _ { j ( \ell ) } ^ { \top } Q _ { t ( \ell ) }$ are rows of shared tree-specific rotations, not independent node-specific normals. For an upper bound, we ignore the orthogonality constraints and allow every $Q _ { t }$ to range freely over $\mathbb { R } ^ { p \times p }$ we also allow all thresholds to be arbitrary real numbers. Both enlargements can only increase the number of sign patterns.

Lemma B.6.1 (Direct routing count for rotation forests). For $n \geq 2$ and fixed topology and split-label sequence, the number of empirical routing maps generated by tree-specific rotations and thresholds is at most $\exp \{ C ( p ^ { 2 } + I _ { + } )$ log n}, where $C$ may depend on the fixed T and $p .$

Proof. If $I _ { + } = 0$ , there is one routing map. Suppose $I _ { + } \geq 1$ . The routing decisions are the binary indicators in (B.13), with equality assigned to the right child. Allowing $Q _ { 1 } , \ldots , Q _ { T }$ to range over $\mathbb { R } ^ { p \times p }$ and all thresholds to range over R can only increase the number of routing maps. The $m = n I _ { + }$ comparisons are afine functions of $k = T p ^ { 2 } + I _ { + }$ real variables. If $m \geq k$ , Lemma B.2.12 bounds the number of binary patterns by $( e m / k ) ^ { k } \leq ( { \bar { e } } n ) ^ { k }$ , since $m / k \le n$ . If $m < k$ , the bound $2 ^ { m } \leq 2 ^ { k }$ gives the same logarithmic order. Each binary pattern determines all leaf assignments in the fixed forest topology. Since $T$ is fixed, the stated bound follows. □

Proposition B.6.2 (Entropy of the additive rotation forest sieve). For every fixed $C > 0$

$$
\begin{array} { r } { \log N \left( C \varepsilon _ { n } , \mathcal { F } ^ { \mathrm { r o t } } ( \bar { K } _ { n } , M _ { n } ) , \left. \cdot \right. _ { n } \right) \lesssim \bar { K } _ { n } \log n + p ^ { 2 } \log n . } \end{array}
$$

Consequently,

$$
\begin{array} { r } { \log N \left( C \varepsilon _ { n } , \mathcal { F } ^ { \mathrm { r o t } } ( \bar { K } _ { n } , M _ { n } ) , \left. \cdot \right. _ { n } \right) \lesssim n \varepsilon _ { n } ^ { 2 } . } \end{array}
$$

Proof. We count the components of a cover.

Tree topologies and split labels. The number of rooted binary forest topologies with total leaf count at most $\bar { K } _ { n }$ is at most $\exp \{ C \bar { K } _ { n } \}$ , since $T$ is fixed and Catalan numbers are bounded by exponential functions. $\mathrm { A }$ forest with $I _ { + }$ internal nodes has at most $p ^ { I _ { + } }$ split-label sequences. Since $p$ is fixed, the topology and label cost is $\exp \{ C \bar { K } _ { n } \}$ .

Routing maps from shared rotations. For each fixed topology and label sequence, Lemma B.6.1 gives at most

$$
\exp \{ C ( p ^ { 2 } + { \bar { K } } _ { n } ) \log n \}
$$

empirical routing maps. Free numerical thresholds already include the original cutpoint-index and framedependent-grid choices, so no additional $b _ { n } ^ { I _ { + } }$ factor is needed.

Leaf coeficients. For each realized $\mathcal { R }$ , select one valid rotation and discrete-tree realization. Every row of $H _ { \mathcal { R } }$ contains exactly $T$ ones, including when some columns represent empty leaves. Thus $\| f _ { \beta } - f _ { \beta ^ { \prime } } \| _ { n } \leq$ $\| H _ { \mathcal { R } } ( \beta - \beta ^ { \prime } ) \| _ { \infty } \leq T \| \beta - \beta ^ { \prime } \| _ { \infty }$ . An $\ell _ { \infty }$ cover of $[ - M _ { n } , M _ { n } ] ^ { K _ { + } }$ with radius $C \varepsilon _ { n } / T$ and centers in that cube therefore gives an empirical cover of radius $C \varepsilon _ { n }$ . Pairing its centers with the selected valid realization keeps every covering function in the original sieve. The logarithmic covering size is at most $K _ { + } \log \{ 1 +$ $2 T M _ { n } / ( C \varepsilon _ { n } ) \}$ , which is bounded by

$$
C \bar { K } _ { n } \log ( M _ { n } / \varepsilon _ { n } ) \lesssim \bar { K } _ { n } \log n .
$$

Combining the three bounds gives the first assertion. By definition, ${ \bar { K } } _ { n } \log n \lesssim n \varepsilon _ { n } ^ { 2 }$ , and fixed $p$ implies $p ^ { 2 } \log n = o ( n \varepsilon _ { n } ^ { 2 } )$ by (B.8). □

Corollary B.6.3 (Joint entropy including the noise scale). For the metric $d _ { G }$ in (B.10),

$$
\log N ( C \varepsilon _ { n } , \overline { { \Theta } } _ { n } , d _ { G } ) \lesssim n \varepsilon _ { n } ^ { 2 } .
$$

Proof. Use radius $C \varepsilon _ { n } / \sqrt { 2 }$ for both the function cover from Proposition B.6.2 and the scalar noise-scale cover. The interval $I _ { n } = [ n ^ { - A _ { \sigma } } , e ^ { A _ { \sigma } n \varepsilon _ { n } ^ { 2 } } ]$ contributes

$$
\log N ( C \varepsilon _ { n } / \sqrt { 2 } , I _ { n } , | \cdot | ) \lesssim \log \left( \frac { e ^ { A _ { \sigma } n \varepsilon _ { n } ^ { 2 } } } { \varepsilon _ { n } } \right) \lesssim n \varepsilon _ { n } ^ { 2 } .
$$

The empirical function metric is exactly the mean-vector part of $d _ { G }$ . The two squared covering errors add to at most $C ^ { 2 } \varepsilon _ { n } ^ { 2 } .$ , and the logarithms of the cover sizes add. □

## B.7 Sieve complement prior mass

Because the rotation group is compact and $p$ is fixed, the complement of the sieve involves only tree sizes, leaf magnitudes, and the noise scale. No support-size restriction is needed.

Proposition B.7.1 (Sieve complement prior mass). For every prescribed $C _ { \mathrm { t a i l } } > 0$ , suficiently large constants $M _ { K } , A _ { \mu } , A _ { \sigma } > 0$ satisfy

$$
\Pi _ { n } ( \Theta _ { n } ^ { c } ) \leq e ^ { - C _ { \mathrm { t a i l } } n \varepsilon _ { n } ^ { 2 } } , \qquad \overline { { \Pi } } _ { n } ( \overline { { \Theta } } _ { n } ^ { c } ) \leq e ^ { - C _ { \mathrm { t a i l } } n \varepsilon _ { n } ^ { 2 } }
$$

for all suficiently large $n .$

Proof. If none of the following generating-space events occurs, the induced pair lies in $\Theta _ { n }$ and its mean-vector image lies in $\overline { { \Theta } } _ { n }$ . Both $\Pi _ { n } ( \Theta _ { n } ^ { c } )$ and $\overline { { \Pi } } _ { n } \big ( \overline { { \Theta } } _ { n } ^ { c } \big )$ are therefore at most $\mathbb { P } _ { n } ( E _ { K , n } ) + \mathbb { P } _ { n } ( E _ { \mu , n } ) + \mathbb { P } _ { n } ( E _ { \sigma , n } )$ , where

$$
E _ { K , n } = \{ K _ { + } > \bar { K } _ { n } \} , \qquad E _ { \mu , n } = \{ \operatorname* { m a x } _ { t < T } \left\| M _ { t } \right\| _ { \infty } > M _ { n } , \ K _ { + } \leq \bar { K } _ { n } \} , \qquad E _ { \sigma , n } = \{ \sigma \not \in I _ { n } \} .
$$

Large trees. Since $T$ is fixed, the union bound and Lemma B.2.10 give

$$
\mathbb { P } _ { n } ( E _ { K , n } ) \le C _ { T } \exp \{ - c \bar { K } _ { n } \log \bar { K } _ { n } \} .
$$

By (B.9), log $\bar { K } _ { n } \gtrsim \log n$ , while $\bar { K } _ { n } \asymp M _ { K } n \varepsilon _ { n } ^ { 2 } / \log n$ . Taking $M _ { K }$ suficiently large gives

$$
\begin{array} { r } { \mathbb { P } _ { n } ( E _ { K , n } ) \le e ^ { - ( C _ { \mathrm { t a i l } } + 2 ) n \varepsilon _ { n } ^ { 2 } } . } \end{array}
$$

Large leaf magnitudes. Condition on $Q _ { 1 } , \ldots , Q _ { T } , ( T _ { t } , V _ { t } , C _ { t } ) _ { t = 1 } ^ { T }$ with $K _ { + } \leq \bar { K } _ { n }$ . By (P4), each conditional Gaussian variance equals $\tau ^ { 2 }$ . The union bound gives a conditional probability at most $2 \bar { K } _ { n } \exp \{ - M _ { n } ^ { 2 } / ( 2 \tau ^ { 2 } ) \}$ that any leaf mean exceeds $M _ { n }$ in absolute value. Integrating this uniform bound over ensembles with $K _ { + } \leq \bar { K } _ { n }$ yields $\mathbb { P } _ { n } ( E _ { \mu , n } ) \le 2 \bar { K } _ { n } \exp \{ - { M _ { n } ^ { 2 } } / { ( 2 \tau ^ { 2 } ) } \}$ }. Since $n \varepsilon _ { n } ^ { 2 } = o ( n )$ and log $\bar { K } _ { n } = O ( \log n )$ , any fixed $A _ { \mu } \geq 1 / 2$ makes this bound at most $e ^ { - ( C _ { \mathrm { t a i l } } + 2 ) n \varepsilon _ { n } ^ { 2 } }$ for all suficiently large n.

Variance tail. Write $V = \sigma ^ { 2 } \sim \mathrm { I G } ( a _ { \sigma } , b _ { \sigma } )$ . The upper tail is polynomial in $V ,$ and therefore the exponential upper endpoint yields

$$
\mathbb { P } _ { n } ( \sigma > e ^ { A _ { \sigma } n \varepsilon _ { n } ^ { 2 } } ) \leq C \exp \{ - 2 a _ { \sigma } A _ { \sigma } n \varepsilon _ { n } ^ { 2 } \} .
$$

For the lower tail, $1 / V$ is gamma and

$$
\mathbb { P } _ { n } ( \sigma < n ^ { - A _ { \sigma } } ) = \mathbb { P } _ { n } ( 1 / V > n ^ { 2 A _ { \sigma } } ) \le \exp \{ - c n ^ { 2 A _ { \sigma } } \} .
$$

Taking $A _ { \sigma }$ suficiently large makes both terms at most $e ^ { - ( C _ { \mathrm { t a i l } } + 2 ) n \varepsilon _ { n } ^ { 2 } }$

The union bound and the preceding image-space inequalities prove the proposition. These inequalities need not be equalities, since a function or mean vector may have more than one forest representation. □

## B.8 Proof of the main theorem

Proof of Theorem B.3.1. The likelihood depends on $( f , \sigma )$ only through $( \mu _ { f } , \sigma )$ . Its normalizing integral is positive and finite: the likelihood is bounded by $( 2 \pi ) ^ { - n / 2 } { \boldsymbol { \sigma } } ^ { - n }$ , and the inverse-gamma prior gives $\mathbb { E } _ { \mathbb { P } _ { n } } ( \sigma ^ { - n } ) =$ $b _ { \sigma } ^ { - n / 2 } \Gamma ( a _ { \sigma } + n / 2 ) / \Gamma ( a _ { \sigma } ) < \infty$ . Bayes’ formula therefore gives $\overline { { \Pi } } _ { n } ( B \mid Y ^ { ( n ) } ) = \Pi _ { n } \{ ( f , \sigma ) : ( \mu _ { f } , \sigma ) \in B \mid Y ^ { ( n ) } \}$ for every Borel set $B \subset \mathbb { R } ^ { n } \times ( 0 , \infty )$

Local prior mass. Fix $A _ { 0 } \geq 1$ as in Corollary B.5.3, and set $\widetilde { \varepsilon } _ { n } = A _ { 0 } \varepsilon _ { n }$ . Since $\sigma _ { 0 }$ and $A _ { 0 }$ are fixed, (B.8) gives $\widetilde { \varepsilon } _ { n } / \sigma _ { 0 }  0$ and $n \widetilde { \varepsilon } _ { n } ^ { 2 } / \sigma _ { 0 } ^ { 2 } \to \infty$ . The identity $\left\| \mu _ { f } - \mu _ { f _ { 0 } } \right\| _ { 2 } ^ { 2 } = n \left\| f - f _ { 0 } \right\| _ { n } ^ { 2 }$ and Corollary B.5.3 give

$$
\overline { { \Pi } } _ { n } \left\{ \| \mu - \mu _ { f _ { 0 } } \| _ { 2 } ^ { 2 } \leq n \widetilde { \varepsilon } _ { n } ^ { 2 } , \quad | \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | \leq \sigma _ { 0 } \widetilde { \varepsilon } _ { n } \right\} \geq e ^ { - C n \widetilde { \varepsilon } _ { n } ^ { 2 } / \sigma _ { 0 } ^ { 2 } }
$$

for a fixed C. Both C and $A _ { 0 }$ are fixed before the sieve cutofs are chosen.

Tail bound and sieve choice. Choose a fixed E large enough for the denominator bound in the proof of Theorem 3 of Jeong (2025). As noted after Proposition B.2.8, this choice depends on the local mass and local Gaussian bounds, not on the eventual entropy constant. Fix $C _ { \mathrm { t a i l } } \geq E A _ { 0 } ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ . Proposition B.7.1 then provides fixed $M _ { K } , A _ { \mu } , A _ { \sigma }$ and a sieve sequence satisfying

$$
\overline { { \Pi } } _ { n } ( \overline { { \Theta } } _ { n } ^ { c } ) \leq e ^ { - E n \widetilde { \varepsilon } _ { n } ^ { 2 } / \sigma _ { 0 } ^ { 2 } } .
$$

Entropy and contraction. For this selected sieve, Corollary B.6.3 gives a finite constant D such that

$$
\log N ( \widetilde { \varepsilon } _ { n } , \overline { { \Theta } } _ { n } , d _ { G } ) \leq D n \widetilde { \varepsilon } _ { n } ^ { 2 } / \sigma _ { 0 } ^ { 2 } .
$$

The constant $D$ may depend on the selected cutofs. We choose the final contraction multiplier only after these choices. The three conditions in Proposition B.2.8 now hold simultaneously, giving contraction in $d _ { G }$ for a suficiently large fixed multiplier.

By the posterior identity above and $\begin{array} { r } { \| f - f _ { 0 } \| _ { n } + | \sigma - \sigma _ { 0 } | \le \sqrt { 2 } d _ { G } ( ( \mu _ { f } , \sigma ) , ( \mu _ { f _ { 0 } } , \sigma _ { 0 } ) ) } \end{array}$ , this proves (B.11), after absorbing $A _ { 0 }$ and $\sqrt { 2 }$ into M. For suficiently large $n , | \sigma - \sigma _ { 0 } | \leq M \varepsilon _ { n }$ implies $\sigma + \sigma _ { 0 } \le 2 \sigma _ { 0 } + 1$ . Hence $| \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | \le ( 2 \bar { \sigma } _ { 0 } + 1 ) | \sigma - \sigma _ { 0 } |$ on that event, and the variance-scale conclusion follows by enlarging the fixed multiplier. □

## C POSTERIOR CONTRACTION LOWER BOUND FOR AXIS-ALIGNED BART

We use the fixed-design Gaussian model and notation of Supplement B. Let $\Pi _ { n }$ be the RoBART posterior and let $\Pi _ { n } ^ { \mathrm { B } }$ be the posterior under the same tree, leaf, noise, and grid specifications with $Q _ { t } = I _ { p }$ deterministically for every tree. Both priors use the same fixed $T , \ 0 < \nu < 1 , \ \tau ^ { 2 } , a _ { \sigma } , b _ { \sigma } > 0$ , and deterministic integer budget $b _ { n } \geq 1$ . Thus split attempts have probability $\nu ^ { \ell + 1 }$ at depth $\ell ,$ coordinate choices are uniform over all $p$ coordinates, and cutpoints are uniform among admissible candidates. The grid uses the full training design’s empirical range, rotated for RoBART; nodewise admissibility is its strict-interior restriction. There is no minimum leaf occupancy, and equality is routed to the right. Write $\mathbb { P } _ { n } ^ { \mathrm { B } }$ and $\mathbb { P } _ { n } ^ { \mathrm { B } } ( \cdot \mid Y ^ { ( n ) } )$ for the BART generating prior and posterior. The complete states $( T _ { t } , V _ { t } , C _ { t } )$ , leaf vectors $M _ { t } ,$ and total leaf count $\begin{array} { r } { K _ { + } = \sum _ { t = 1 } ^ { T } K _ { t } } \end{array}$ retain their meanings from Supplement B.

The following restates Theorem 2 with its design and grid conditions made explicit.

Theorem C.1 (Posterior contraction lower bound for axis-aligned BART). Fix integers $p \geq 2$ and $T \geq 1$ $0 < \alpha < 1$ , and $\lambda , C _ { h } , \sigma _ { 0 } > 0$ . Let $\mathcal { X } = [ 0 , 1 ] ^ { p }$ . For $m \geq 1$ and $n = 2 ^ { m p }$ , let the fixed design enumerate

$$
{ \mathcal { X } } _ { n } = \left\{ \left( { \frac { i _ { 1 } + 1 / 2 } { 2 ^ { m } } } , \ldots , { \frac { i _ { p } + 1 / 2 } { 2 ^ { m } } } \right) : i _ { 1 } , \ldots , i _ { p } \in \{ 0 , \ldots , 2 ^ { m } - 1 \} \right\} \subset [ 0 , 1 ] ^ { p } .
$$

Choose $K _ { n , 1 } = 2 ^ { L _ { n , 1 } } \asymp ( n \lambda ^ { 2 } / \log n ) ^ { 1 / ( 2 \alpha + 1 ) }$ with $L _ { n , 1 } \in \mathbb { N } _ { . }$ , and assume $K _ { n , 1 } \leq b _ { n } + 1$ and log $b _ { n } \leq C _ { b }$ log n for a fixed $C _ { b } < \infty$ and all suficiently large n.

There exist $Q _ { 0 } \in \mathrm { S O } ( p )$ and $h _ { 0 } : [ - \sqrt { p } , \sqrt { p } ] \to \mathbb { R }$ , both independent of n, such that $\| h _ { 0 } \| _ { \infty } \leq C _ { h }$ and $| h _ { 0 } ( u ) - h _ { 0 } ( v ) | \leq \lambda | u - v | ^ { \alpha }$ on this interval, and the fixed truth $f _ { 0 } ( x ) = h _ { 0 } \{ \pi _ { \{ 1 \} } ( Q _ { 0 } x ) \}$ on $[ 0 , 1 ] ^ { p }$ has the following properties. Set

$$
\varepsilon _ { n } = \lambda ^ { 1 / ( 2 \alpha + 1 ) } \left( \frac { \log n } { n } \right) ^ { \alpha / ( 2 \alpha + 1 ) } , \qquad \varepsilon _ { n } ^ { \mathrm { B } } = \left( \frac { \log n } { n } \right) ^ { \alpha / ( 2 \alpha + p ) } .
$$

For some fixed $c > 0$

$$
\mathbb { E } _ { 0 } \Pi _ { n } ^ { \mathrm { B } } \big ( \lVert f - f _ { 0 } \rVert _ { n } \leq c \varepsilon _ { n } ^ { \mathrm { B } } \ | \ Y ^ { ( n ) } \big ) \longrightarrow 0 .\tag{C.1}
$$

On the same design and at the same truth, for a suficiently large fixed $M _ { 0 }$

$$
\mathbb { E } _ { 0 } \Pi _ { n } \big ( \lVert f - f _ { 0 } \rVert _ { n } + \lvert \sigma - \sigma _ { 0 } \rvert > M _ { 0 } \varepsilon _ { n } \ \lvert \ Y ^ { ( n ) } \big ) \longrightarrow 0 .\tag{C.2}
$$

Since $\varepsilon _ { n } = o ( \varepsilon _ { n } ^ { \mathrm { B } } )$ , the BART assertion also holds with $c \varepsilon _ { n } ^ { \mathrm { B } }$ replaced by $M \varepsilon _ { n }$ for every fixed $M > 0$ . All limits are taken as $m  \infty$ along $n = 2 ^ { m p }$

The budget assumptions are Supplement B’s (D2) for $R = d _ { 1 } = 1 ; b _ { n } = n$ is a suficient choice, not a requirement.

Proof. Constants may depend on the fixed truth, prior, and grid parameters, but not on n.

Construction of a fixed truth. Choose an integer $J \geq 1$ suficiently large that

$$
\left( { \frac { \pi } { 2 } } \right) ^ { p } \sum _ { v = 1 } ^ { \infty } 2 ^ { - J v ( p - \alpha ) } \leq { \frac { 1 } { 2 } } \left( { \frac { 2 } { \pi } } \right) ^ { p } .
$$

For $\kappa > 0$ to be chosen, define on R

$$
h _ { 0 } ( u ) = \kappa \sum _ { s = 1 } ^ { \infty } 2 ^ { - J s \alpha } \cos \left( 2 \pi 2 ^ { J s } \sqrt { p } u - \frac { p \pi } { 2 } \right) .
$$

The series converges absolutely and uniformly. For $0 < d = | u - v | \leq 1$

$$
\left| h _ { 0 } ( u ) - h _ { 0 } ( v ) \right| \le C \kappa \left\{ d \sum _ { 2 ^ { J _ { s } } \le d ^ { - 1 } } 2 ^ { J s ( 1 - \alpha ) } + \sum _ { 2 ^ { J _ { s } } > d ^ { - 1 } } 2 ^ { - J s \alpha } \right\} \le C \kappa d ^ { \alpha } .
$$

Boundedness handles $d > 1$ . Taking κ suficiently small gives the required Hölder and sup-norm bounds on $[ - \sqrt { p } , \sqrt { p } ]$ . Let $a = p ^ { - 1 / 2 } ( 1 , \ldots , 1 ) ^ { \top }$ and complete $a ^ { \top }$ to the rows of $Q _ { 0 } \in \mathrm { S O } ( p )$ , reversing the sign of another row if necessary. Then the resulting fixed truth is

$$
f _ { 0 } ( x ) = \kappa \sum _ { s = 1 } ^ { \infty } 2 ^ { - J s \alpha } \cos \left( 2 \pi 2 ^ { J s } \sum _ { \ell = 1 } ^ { p } x _ { \ell } - \frac { p \pi } { 2 } \right) .\tag{C.3}
$$

Empirical tensor-Haar coeficients. Define

$$
\psi ( u ) = \mathbf { 1 } _ { [ 0 , 1 / 2 ) } ( u ) - \mathbf { 1 } _ { [ 1 / 2 , 1 ) } ( u ) , \qquad \psi _ { j , k } ( x ) = 2 ^ { j p / 2 } \prod _ { \ell = 1 } ^ { p } \psi ( 2 ^ { j } x _ { \ell } - k _ { \ell } ) ,
$$

for integers $j \geq 0$ and $k \in \mathcal { T } _ { j } = \{ 0 , \ldots , 2 ^ { j } - 1 \} ^ { p }$ , and write $\begin{array} { r } { \langle u , v \rangle _ { n } = n ^ { - 1 } \sum _ { i } u ( x _ { i , n } ) v ( x _ { i , n } ) } \end{array}$ . For $j \le m - 1$ ， these functions are empirically orthonormal: their supports are disjoint, each contains $2 ^ { p ( m - j ) }$ observations, and the squared function value on its support is $2 ^ { j p }$

For even $N \geq 2$ , pairing the two halves of the sum gives

$$
\begin{array} { l } { \displaystyle { D _ { N } ( t ) : = \frac { 1 } { N } \sum _ { r = 0 } ^ { N - 1 } \psi \left( \frac { r + 1 / 2 } { N } \right) \exp \left( 2 \pi \mathrm { i } t \frac { r + 1 / 2 } { N } \right) } } \\ { \displaystyle { \phantom { \frac { 1 } { N } \sum _ { r = 0 } ^ { N - 1 } \psi } = \frac { 1 - e ^ { \pi \mathrm { i } t } } { N } \sum _ { r = 0 } ^ { N / 2 - 1 } \exp \left( 2 \pi \mathrm { i } t \frac { r + 1 / 2 } { N } \right) . } } \end{array}
$$

Consequently,

$$
\begin{array} { c c c } { { D _ { N } ( 1 ) = \displaystyle { \frac { 2 \mathrm { i } } { N \sin ( \pi / N ) } } , \qquad } } & { { | D _ { N } ( 1 ) | \ge \displaystyle { \frac { 2 } { \pi } } , } } & { { } } \\ { { D _ { N } ( 2 ^ { v } ) = 0 \quad ( v \ge 1 ) , \qquad } } & { { | D _ { N } ( t ) | \le \displaystyle { \frac { \pi t } { 2 } } \quad ( 0 < t < 1 ) , } } \end{array}
$$

where v is an integer. The last bound uses $| 1 - e ^ { \pi \mathrm { i } t } | \leq \pi t$ . For $j = J s \le m - 1$ and $N = 2 ^ { m - j }$ , tensorization yields

$$
\left. \exp \left( 2 \pi \mathrm { i } 2 ^ { j } t \sum _ { \ell = 1 } ^ { p } x _ { \ell } \right) , \psi _ { j , k } \right. _ { n } = 2 ^ { - j p / 2 } \exp \left( 2 \pi \mathrm { i } t \sum _ { \ell = 1 } ^ { p } k _ { \ell } \right) D _ { N } ( t ) ^ { p } .
$$

The sth summand of (C.3) has $t = 1$ . Its spatial phase is one, and $e ^ { - p \pi \mathrm { i } / 2 }$ cancels the phase of $D _ { N } ( 1 ) ^ { p }$ . It therefore contributes at least $\kappa ( 2 / \pi ) ^ { p } 2 ^ { - j ( \alpha + p / 2 ) }$ to every coeficient. All higher-frequency summands vanish since $t = 2 ^ { J v }$ . The absolute sum of the lower-frequency contributions is bounded by

$$
\begin{array} { c } { { \kappa 2 ^ { - j p / 2 } \displaystyle \sum _ { v = 1 } ^ { s - 1 } 2 ^ { - ( j - J v ) \alpha } \left( \displaystyle \frac { \pi } { 2 } 2 ^ { - J v } \right) ^ { p } = \kappa \left( \displaystyle \frac { \pi } { 2 } \right) ^ { p } 2 ^ { - j ( \alpha + p / 2 ) } \displaystyle \sum _ { v = 1 } ^ { s - 1 } 2 ^ { - J v ( p - \alpha ) } } } \\ { { \leq \displaystyle \frac { \kappa } { 2 } \left( \displaystyle \frac { 2 } { \pi } \right) ^ { p } 2 ^ { - j ( \alpha + p / 2 ) } . } } \end{array}
$$

Absolute convergence permits termwise evaluation in these finite sums. Thus, for every positive multiple j of J with $j \le m - 1$

$$
\langle f _ { 0 } , \psi _ { j , k } \rangle _ { n } \geq c _ { H } 2 ^ { - j ( \alpha + p / 2 ) } , \qquad c _ { H } = \frac { \kappa } { 2 } \left( \frac { 2 } { \pi } \right) ^ { p } > 0 , \qquad k \in \mathbb { Z } _ { j } .\tag{C.4}
$$

Approximation lower bound for a forest. For an axis-aligned rectangle B, the coeficient $\langle \mathbf { 1 } _ { B } , \psi _ { j , k } \rangle _ { n }$ factors into univariate coeficients. An interval indicator can vary on at most two level-j dyadic intervals; on every other interval its coeficient vanishes because the two halves contain equal numbers of design points. Hence a rectangle contributes to at most $2 ^ { p }$ coeficients. Any forest with at most K total leaves is $\begin{array} { r } { g = \sum _ { \ell = 1 } ^ { K } b _ { \ell } \mathbf { 1 } _ { B _ { \ell } } } \end{array}$ , allowing overlapping rectangles and unrestricted signed weights, so

$$
\# \{ k : \langle g , \psi _ { j , k } \rangle _ { n } \neq 0 \} \leq \sum _ { \ell = 1 } ^ { K } \# \{ k : \langle \mathbf { 1 } _ { B _ { \ell } } , \psi _ { j , k } \rangle _ { n } \neq 0 \} \leq 2 ^ { p } K .
$$

When $2 ^ { j p } \geq 2 ^ { p + 1 } K$ , at least half of the coeficients in (C.4) are zero for $g .$ Bessel’s inequality gives

$$
\| f _ { 0 } - g \| _ { n } ^ { 2 } \geq \sum _ { k \in \mathcal { T } _ { j } : } \sum _ { \langle g , \psi _ { j , k } \rangle _ { n } = 0 } \vert \langle f _ { 0 } , \psi _ { j , k } \rangle _ { n } \vert ^ { 2 } \geq \frac { c _ { H } ^ { 2 } } { 2 } 2 ^ { - 2 j \alpha } .
$$

Choose the smallest positive multiple $j$ of J such that $2 ^ { j p } \geq 2 ^ { p + 1 } K ;$ then $2 ^ { j p } \asymp K$ . The largest multiple of J not exceeding $m - 1$ is at least $m - J$ , so the selected j is admissible whenever $K \leq c _ { \mathcal { X } } n$ for a suficiently small fixed $c _ { \mathcal { X } } > 0$ and large m. Writing ${ \mathcal { F } } ^ { \mathrm { a x } } ( K )$ for all axis-aligned constant-leaf forests with at most K total leaves, we obtain

$$
\operatorname* { i n f } _ { g \in { \mathcal { F } } ^ { \mathrm { a x } } ( K ) } \| g - f _ { 0 } \| _ { n } \geq c _ { A } K ^ { - \alpha / p } , \qquad 1 \leq K \leq c _ { { \mathcal { X } } } n .\tag{C.5}
$$

This class allows arbitrary numerical cutpoints and leaf values, so the bound applies to the declared empirical-range prior.

Local prior mass. Put $q _ { n } = n ( \varepsilon _ { n } ^ { \mathrm { B } } ) ^ { 2 }$ and $l _ { n } = \lfloor \log _ { 2 } ( n / \log n ) / ( 2 \alpha + p ) \rfloor$ . Choose a complete tree with depth $L _ { n } ^ { \mathrm { B } } = p l _ { n }$ and $K _ { n } ^ { \mathrm { B } } = 2 ^ { p l _ { r } }$ leaves, cycling through the $p$ coordinates. Then

$$
K _ { n } ^ { \mathrm { B } } \asymp ( n / \log n ) ^ { p / ( 2 \alpha + p ) } , \qquad 2 ^ { - \alpha l _ { n } } \asymp \varepsilon _ { n } ^ { \mathrm { B } } , \qquad K _ { n } ^ { \mathrm { B } } \log n \asymp q _ { n } \longrightarrow \infty .\tag{C.6}
$$

The budget is suficient for the $l _ { n }$ splits in each coordinate, since

$$
\frac { 2 ^ { l _ { n } } } { b _ { n } + 1 } \leq \frac { 2 ^ { l _ { n } } } { K _ { n , 1 } } \lesssim ( n / \log n ) ^ { 1 / ( 2 \alpha + p ) - 1 / ( 2 \alpha + 1 ) } \longrightarrow 0 .
$$

Use the lower median admissible candidate at every split. The integer-cell recursion $N ^ { \prime } \mapsto \left( \lfloor N ^ { \prime } / 2 \rfloor , \lceil N ^ { \prime } / 2 \rceil \right)$ from Proposition B.2.5(ii) of Supplement B gives, after a coordinate splits, between $\lfloor ( b _ { n } + 1 ) / 2 ^ { a } \rfloor$ and $\lceil ( b _ { n } + 1 ) / 2 ^ { a } \rceil$ cells in each coordinate interval. Since $2 ^ { l _ { n } } \leq b _ { n } + 1$ , each requested split starts with at least two cells. The final coordinate widths are bounded by

$$
( 1 - 2 ^ { - m } ) \frac { \lceil ( b _ { n } + 1 ) / 2 ^ { l _ { n } } \rceil } { b _ { n } + 1 } \leq 2 ^ { 1 - l _ { n } } .
$$

These cells cover the empirical-range box, and empty leaves are allowed. The bound $| f _ { 0 } ( x ) - f _ { 0 } ( y ) | \leq$ $\lambda p ^ { \alpha / 2 } \left\| x - y \right\| _ { \infty } ^ { \alpha }$ therefore gives an approximating tree $\check { g } _ { n } ^ { \mathrm { B } }$ with $\left\| \widehat { g } _ { n } ^ { \mathrm { B } } - f _ { 0 } \right\| _ { n } \leq C \varepsilon _ { n } ^ { \mathrm { B } }$ by taking $f _ { 0 }$ at a point in each leaf closure. Denote its ordered leaf center by $\beta _ { n } ^ { \mathrm { B } }$ ; then $\left\| \beta _ { n } ^ { \mathrm { B } } \right\| _ { \infty } \leq C _ { h }$

Use this as the first tree and take the remaining $T - 1$ trees to be root-only. Let $\mathbf { t } _ { n } ^ { \mathrm { B } }$ be this prescribed tuple of complete discrete states. Each prescribed coordinate and cut contributes at least $( p b _ { n } ) ^ { - 1 }$ , and each terminal factor is at least $1 - \nu .$ Since $\begin{array} { r } { \sum _ { d = 0 } ^ { L _ { n } ^ { \mathrm { B } } - 1 } ( d + 1 ) 2 ^ { d } = ( L _ { n } ^ { \mathrm { B } } - 1 ) K _ { n } ^ { \mathrm { B } } + 1 } \end{array}$

$$
\begin{array} { r l } & { \mathbb { P } _ { n } ^ { \mathrm { B } } \{ ( \mathcal { T } _ { t } , V _ { t } , C _ { t } ) _ { t = 1 } ^ { T } = \mathfrak { t } _ { n } ^ { \mathrm { B } } \} } \\ & { \quad \ge ( 1 - \nu ) ^ { K _ { n } ^ { \mathrm { B } } + T - 1 } ( p b _ { n } ) ^ { - ( K _ { n } ^ { \mathrm { B } } - 1 ) } \nu ^ { ( L _ { n } ^ { \mathrm { B } } - 1 ) K _ { n } ^ { \mathrm { B } } + 1 } \ge e ^ { - C K _ { n } ^ { \mathrm { B } } \log n } , } \end{array}\tag{C.7}
$$

using $L _ { n } ^ { \mathrm { B } } = O ( \log n )$ and log $b _ { n } \leq C _ { b } \log n$

Within the prescribed-state event, define $A _ { n } ^ { \mathrm { B } } \ \mathrm { b y } \ \left\| M _ { 1 } - \beta _ { n } ^ { \mathrm { B } } \right\| _ { \infty } \leq \varepsilon _ { n } ^ { \mathrm { B } } / ( 2 T ) , \ | \mu _ { t } | \leq \varepsilon _ { n } ^ { \mathrm { B } } / ( 2 T )$ for $t = 2 , \ldots , T$ and $| \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | \leq \sigma _ { 0 } \varepsilon _ { n } ^ { \mathrm { B } }$ . Here $M _ { 1 }$ and $\beta _ { n } ^ { \mathrm { B } }$ use the same left-to-right leaf order, and $\mu _ { t }$ is the unique leaf mean of the tth root-only tree. The coeficient comparisons are evaluated only on that state; when $T = 1$ the remaining-tree restrictions are absent. The Gaussian densities have a common positive lower bound around these bounded centers, and the inverse-gamma density is positive near $\sigma _ { 0 } ^ { 2 } .$ . Each scalar restriction therefore has probability at least $c \varepsilon _ { n } ^ { \mathrm { B } }$ for large n. Independence and (C.6)–(C.7) give

$$
\mathbb { P } _ { n } ^ { \mathrm { B } } ( A _ { n } ^ { \mathrm { B } } ) \geq e ^ { - C _ { P } q _ { n } } .\tag{C.8}
$$

On $A _ { n } ^ { \mathrm { B } }$ the total leaf perturbation is at most $\varepsilon _ { n } ^ { \mathrm { B } } / 2$ at every design point. Thus $\| f - f _ { 0 } \| _ { n } \leq C _ { f } \varepsilon _ { n } ^ { \mathrm { B } }$ and $| \sigma - \tilde { \sigma _ { 0 } } | = | \sigma ^ { 2 } - \sigma _ { 0 } ^ { 2 } | / ( \bar { \sigma } + \sigma _ { 0 } ) \le \varepsilon _ { n } ^ { \mathrm { B } } \le$

A marginal likelihood lower bound. For a generating-space state θ inducing $( f , \sigma )$ , let $L _ { n } ( \theta ) =$ $p _ { f , \sigma } ^ { ( n ) } ( Y ^ { ( n ) } ) / p _ { 0 } ^ { ( n ) } ( Y ^ { ( n ) } )$ and $\begin{array} { r } { D _ { n } = \int L _ { n } ( \theta ) d \mathbb { P } _ { n } ^ { \mathrm { B } } ( \theta ) } \end{array}$ , where the $p \mathrm { { s } }$ denote densities. The normalizer is positive and finite because the likelihood is bounded by $( 2 \pi ) ^ { - n / 2 } \sigma ^ { - n }$ and the inverse-gamma prior has finite negative moments. The local error bounds and Lemma B.2.9 of Supplement B imply

$$
\operatorname* { s u p } _ { \theta \in A _ { n } ^ { \mathrm { B } } } \left[ \mathrm { K L } ( P _ { 0 } ^ { ( n ) } , P _ { f , \sigma } ^ { ( n ) } ) + \mathrm { V a r } _ { 0 } \{ \log L _ { n } ( \theta ) \} \right] \leq C _ { G } q _ { n } .
$$

Let $\widetilde { \mathbb { P } } _ { n } ^ { \mathrm { B } }$ be the prior restricted to $A _ { n } ^ { \mathrm { B } }$ and normalized, and put $\begin{array} { r } { Z _ { n } = \int \log L _ { n } ( \theta ) d \widetilde { \mathbb { P } } _ { n } ^ { \mathrm { B } } ( \theta ) } \end{array}$ . Then $\mathbb { E } _ { 0 } Z _ { n } \geq - C _ { G } q _ { n }$ and, by Jensen’s inequality,

$$
\operatorname { V a r } _ { 0 } ( Z _ { n } ) \leq \int \operatorname { V a r } _ { 0 } \{ \log L _ { n } ( \theta ) \} d \widetilde { \mathbb { P } } _ { n } ^ { \mathrm { B } } ( \theta ) \leq C _ { G } q _ { n } .
$$

Moreover, log $D _ { n } \geq \log \mathbb { P } _ { n } ^ { \mathrm { B } } ( A _ { n } ^ { \mathrm { B } } ) + Z _ { n }$ . Chebyshev’s inequality and (C.8), with $C _ { D } = C _ { P } + C _ { G } + 1$ , therefore give

$$
P _ { 0 } ^ { ( n ) } ( { \mathcal { D } } _ { n } ^ { c } ) \leq C _ { G } / q _ { n } \longrightarrow 0 , \qquad { \mathcal { D } } _ { n } = \{ D _ { n } \geq e ^ { - C _ { D } q _ { n } } \} .\tag{C.9}
$$

Posterior complexity and the lower bound. The split-attempt proof of Lemma B.2.10 of Supplement B applies to the axis-aligned recursion uniformly in n and the finite grid budget. A union bound over the fixed number T of trees gives

$$
\mathbb { P } _ { n } ^ { \mathrm { B } } ( K _ { + } > k ) \le C _ { 2 } \exp ( - c _ { 2 } k \log k )\tag{C.10}
$$

for fixed $c _ { 2 } , C _ { 2 } > 0$ and all suficiently large k. Set $\bar { K } _ { n } ^ { \mathrm { B } } = \lceil C _ { K } q _ { n } / \log n \rceil$ . Then log $\bar { K } _ { n } ^ { \mathrm { B } } \asymp \log n , \bar { K } _ { n } ^ { \mathrm { B } } = o ( n )$ and $( \bar { K } _ { n } ^ { \mathrm { B } } ) ^ { - \alpha / p } \asymp \varepsilon _ { n } ^ { \mathrm { B } }$ . Choose $C _ { K }$ after $C _ { D }$ so that (C.10) yields $\mathbf { \bar { \mathbb { P } } } _ { n } ^ { \mathrm { B } } ( K _ { + } > \bar { K } _ { n } ^ { \mathrm { B } } ) \le e ^ { - ( \bar { C } _ { D } + 2 ) \bar { q _ { n } } }$ . Since $\mathbb { E } _ { 0 } L _ { n } ( \theta ) = 1$ , Tonelli’s theorem and (C.9) give

$$
\begin{array} { r l } { \mathbb { E } _ { 0 } \mathbb { P } _ { n } ^ { \mathrm { B } } ( K _ { + } > \bar { K } _ { n } ^ { \mathrm { B } } \mid Y ^ { ( n ) } ) \le P _ { 0 } ^ { ( n ) } ( \mathcal { D } _ { n } ^ { c } ) + e ^ { C _ { D } q _ { n } } \mathbb { E } _ { 0 } \int _ { \{ K _ { + } > \bar { K } _ { n } ^ { \mathrm { B } } \} } L _ { n } ( \theta ) d \mathbb { P } _ { n } ^ { \mathrm { B } } ( \theta ) } & { } \\ { = P _ { 0 } ^ { ( n ) } ( \mathcal { D } _ { n } ^ { c } ) + e ^ { C _ { D } q _ { n } } \mathbb { P } _ { n } ^ { \mathrm { B } } ( K _ { + } > \bar { K } _ { n } ^ { \mathrm { B } } ) } & { } \\ { \le C _ { G } / q _ { n } + e ^ { - 2 q _ { n } } \longrightarrow 0 . } & { } \end{array}\tag{C.11}
$$

By (C.5), every state with $K _ { + } \leq \bar { K } _ { n } ^ { \mathrm { B } }$ has error at least $c _ { A } ( \bar { K } _ { n } ^ { \mathrm { B } } ) ^ { - \alpha / p }$ for large n. Taking $c > 0$ suficiently small gives the generating-space inclusion

$$
\{ \| f - f _ { 0 } \| _ { n } \leq c \varepsilon _ { n } ^ { \mathrm { B } } \} \subseteq \{ K _ { + } > \bar { K } _ { n } ^ { \mathrm { B } } \} .
$$

The induced posterior bound (C.1) follows from (C.11); no uniqueness of the forest representation is needed.   
Since $p \ge 2 , \varepsilon _ { n } = o ( \varepsilon _ { n } ^ { \mathrm { B } } )$ ), which gives the assertion for every fixed $M \varepsilon _ { n }$ as well.

Application of the RoBART theorem. The constructed truth satisfies the bounded-design and Hölder conditions of Supplement B with $M _ { X } = \sqrt { p } , R = d _ { 1 } = 1 , S _ { 1 } = \{ 1 \} , \lambda _ { 1 } = \lambda$ , and $\alpha _ { 1 1 } = \alpha$ . Its active range is $R _ { n , 1 , 1 } ( Q _ { 0 } ) = \sqrt { p } ( 1 - 2 ^ { - m } ) \ge \sqrt { p } / 2$ , so (D1) holds. Condition (D2) is assumed, and the remaining prior specifications coincide. Theorem B.3.1 of Supplement B therefore gives (C.2). □

## D EXPERIMENTAL SETTINGS

Benchmark. The design in Section 6.1 is fixed across thirty replications; methods share inputs and noise within each replication. All methods use 200 trees and 1,000 unthinned draws after 9,000 burn-in iterations. RoBART starts from identity rotations and uses $K _ { \mathrm { r o t } } = 5 , \gamma = 0 . 0 8$ and $\theta _ { \mathrm { m a x } } = 0 . 3 5 \gamma / 0 . 1 5$ , with joint cutpoint retuning and support-expansion probability 0.5. BART keeps the identity. Both use 100 cutpoints per coordinate, leaf shrinkage $k = 2 ,$ , noise-prior parameters sigdf=3 and sigquant=0.9, and polynomia split probability $0 . 9 5 / ( 1 + \ell ) ^ { 2 }$ at depth ℓ. The noise scale is estimated.

Soft BART uses training-only empirical-distribution preprocessing, $k = 2$ and temperature 1. Noise, leafscale, splitting-probability, α and bandwidth updates are enabled; beta, gamma, tau\_mean updates and tree caching are disabled. Oblique BART and ridgeBART use training-based coordinatewise afine scaling to $[ - 1 , 1 ]$ , applied to test inputs without clipping. Oblique BART uses prob\_aa=0.5 with adaptation, alpha\_dp=5, phi\_option=7, and a\_theta=b\_theta=0. ridgeBART uses all predictors for splitting and smoothing, one ReLU basis, p\_change=0.2, sparse=FALSE, rho\_alpha=400, rho\_nu=3, and rho\_lambda equal to the median of $\chi _ { 3 } ^ { 2 }$ divided by 3.

Illustrative fits. Figure 1 compares the same five methods using 200 trees, 4,000 burn-in iterations, 1,000 unthinned draws, and a 151 × 151 prediction grid. The data seed is 1; sampler seeds are 12, 42, 54, 38 and 24 for BART, Soft BART, ridgeBART, Oblique BART and RoBART. RoBART uses $\theta _ { \mathrm { m a x } } = 0 . 3 5$ . Soft BART receives the original inputs and updates only the noise scale, leaf scale and bandwidth. ridgeBART uses uniform cuts and scales inputs by the largest absolute training entry, clipping only prediction inputs to [−1, 1]. Oblique BART uses uniform cuts and prob\_aa=0 without adaptation.