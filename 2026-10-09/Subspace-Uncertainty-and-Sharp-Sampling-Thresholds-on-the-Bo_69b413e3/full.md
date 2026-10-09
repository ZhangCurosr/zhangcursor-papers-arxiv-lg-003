# Subspace Uncertainty and Sharp Sampling Thresholds on the Boolean Cube

Thomas Weinberger

School of Computer and Communication Sciences, EPFL, Switzerland thomas.weinberger@epfl.ch

October 9, 2026

## Abstract

We study Gaussian regression under squared population $L _ { 2 }$ loss in a known m-dimensional subspace of degree-at-most-k functions on the d-dimensional Boolean cube. Random inputs can undersample regions essential for prediction, delaying the parametric rate even when the model is known.

For fixed $q _ { 0 } < 1 / 2 , 1 \le k \le q _ { 0 } d ,$ and suficiently large fixed A, the worst-subspace sample threshold for minimax error $A \sigma ^ { 2 } ( m + t ) / n$ with confidence $1 - e ^ { - t } , t \geq \log 4$ , is

$$
N = ( m + t ) \exp \{ E _ { d , k } + O ( k ^ { 1 / 3 } ) \} , \quad E _ { d , k } = d \Psi ( k / d ) ,
$$

where $\Psi ( q ) = \log 2 - \mathsf { H } ( \frac { 1 } { 2 } - \sqrt { q ( 1 - q ) } )$ and H is binary entropy with natural logarithms. The upper bound holds for every feasible $m ;$ the matching lower bound holds when $m \leq { \binom { d } { \lfloor k ^ { 1 / 3 } \rfloor } } \ \mathrm { o r } \ t \geq m$

We sharpen the Polyanskiy–Samorodnitsky uncertainty principle in two respects. First, for fixed leakage $\rho \in ( 0 , 1 )$ , the smallest set carrying a fraction $1 - \rho$ of a nonzero degree-at-most-k polynomial’s energy has probability ex $\jmath { \{ - E _ { d , k } + O _ { \rho , q _ { 0 } } ( k ^ { 1 / 3 } ) \} }$ . An Airy-kernel construction proves that the remainder cannot be $o ( k ^ { 1 / 3 } )$ in general. Second, we construct a subspace of dimension $\scriptstyle { \binom { d } { \lfloor k ^ { 1 / 3 } \rfloor } }$ such that every function in the subspace has at least a fraction $1 - \rho$ of its energy on the same set, whose probability is at most exp $\{ - E _ { d , k } + C _ { \rho , q _ { 0 } } k ^ { 1 / 3 } \}$ . For suficiently large $k ,$ this set is a Hamming ball.

A striking consequence is an exponential cost of noise: the parametric rate can require $( m + t ) 4 ^ { k } \exp \{ - O ( \bar { k } ^ { 1 / 3 } ) \}$ samples, whereas $O ( ( m + t ) 2 ^ { k } )$ sufice for noiseless identification. As $k \to \infty$ with $k / d \to 0$ , the noisy threshold is $( m + t ) \exp \{ 2 k + o ( k ) \}$ .

Keywords: minimax linear regression, uncertainty principle, sample complexity, boolean cube, low-degree polynomials.

## Contents

1 Introduction 1   
1.1 Preliminaries 2   
1.2 Main results 4   
1.3 Related work 6   
2 Constructing a polynomial concentrated on a Hamming ball 7   
2.1 Designing the polynomial in the Krawchouk basis 7   
2.2 Choosing the coeficients and bounding the error 9   
2.3 From an approximate eigenvector to a rare event 10   
3 From moment bounds to stable sampling 11   
3.1 The Kirshner–Samorodnitsky moment bound 12   
3.2 Cancellation at the second moment 12   
3.3 From moments to clipping . . 12   
3.4 Uniform empirical control after clipping 13   
4 Many directions on one Hamming ball 14   
4.1 Separating radial concentration from variation within slices 14   
4.2 Uniform energy control across harmonic directions 15   
4.3 Proof of Theorem 1.8 . 17   
5 From concentration to minimax regression 17   
5.1 From shared concentration to small empirical eigenvalues 17   
5.2 From empirical eigenvalues to minimax risk 18   
5.3 Proof of Theorem 1.5 . 19   
6 Sharpness of the uniform uncertainty remainder 19   
6.1 Convergence and limits 19   
6.2 A Gaussian polynomial with a fixed amount of tail energy 20   
6.3 Transferring the construction to a Boolean subspace 21   
7 Discussion 23   
References 23   
A The entropy curve and the cubic remainder 26   
A.1 The entropy curve 26   
A.2 Uniform cubic remainder . 26   
B Classical analytic and probabilistic tools 28   
B.1 Bonami–Beckner hypercontractivity 28   
B.2 A bound independent of the ambient dimension 29   
B.3 A bounded empirical-process inequality 29   
B.4 Symmetrization and contraction 29   
B.5 Elementary Gaussian probability bounds . 30   
C The radial-weight identity 31   
D Probability constants for the sampling lower bounds 31   
E Comparison to the Polyanskiy–Samorodnitsky uncertainty boundary 33   
F Further bounds and model examples 33   
F.1 Ambient dimension and model dimension 34   
F.2 Known Walsh coordinates and finite-space caps 35

## 1 Introduction

A familiar promise of linear regression is that prediction error eventually scales like the number of unknown coeficients divided by the number of observations. How long must one wait before this parametric rate becomes valid? For a known model and random inputs, the answer depends on how well the observations reveal the model’s population geometry. A small model can still be dificult if much of its predictive variation occurs on rarely sampled inputs. This issue connects statistical estimation to the stability of random least squares and sampling discretization [CDL13; KKLT22; Mou22; EHE24].

We study this question for low-degree functions on the Boolean cube. The learner is given a model (in the form of a linear subspace) before observing noisy evaluations at uniformly random inputs. A model has three parameters: the ambient dimension counts input coordinates; the degree bounds the order of interactions; the subspace dimension counts unknown coeficients. The model dimension determines the number of coeficients to estimate, but not how well random observations revea them. Stricter bounds on degree and ambient dimension restrict the class of possible models and can rule out models that are dificult to estimate. Knowing the model specifies the basis functions, but leaves their coeficients unrestricted. Some choices may therefore remain dificult to estimate from random observations, which delays the point at which the parametric rate can be guaranteed uniformly over all targets in the model. This raises the question of how these three parameters jointly determine the sample size needed for accurate prediction.

Statistical question. Among known subspaces with a prescribed ambient dimension, degree, and model dimension, how many random examples are needed before the usual parametric prediction rate is guaranteed?

A central distinction is between identification and accurate prediction. Without noise, the sampled evaluations determine the coeficients exactly when the evaluation map from the model subspace to the observed values has a trivial null space, leaving no indistinguishable targets. With noise, merely distinguishing targets in principle is insuficient: their observed responses must difer enough to be distinguished reliably.

The distribution of energy determines this separation. Throughout this work, we focus on diferences between possible targets and denote these functions by h. Using a basis orthonormal under the uniform input distribution, the squared coeficient distance between two targets equals $\mathbb { E } [ h ( X ) ^ { 2 } ]$ ]. On the observed inputs $X _ { 1 } , \ldots , X _ { n } .$ , however, their squared separation is $\scriptstyle \sum _ { i = 1 } ^ { n } h ( X _ { i } ) ^ { 2 }$ . If this quantity is small relative to the noise variance, the data cannot reliably distinguish the targets, even when their coeficients are far apart. Concentration of the energy of h on a rarely sampled set can create precisely this discrepancy. Exact Gaussian minimax results formalize this connection through the spectrum of the empirical covariance matrix [Mou22; EHE24].

The number of poorly observed directions also matters. For noise variance $\sigma ^ { 2 }$ , the parametric rate allows error of order $\sigma ^ { 2 } / n$ per coeficient, hence total error of order $\sigma ^ { 2 } m / n$ in a model of dimension m. An increase confined to one direction may still fit within this total allowance. To exceed it, a single direction would have to be exceptionally dificult, or many directions would have to be dificult simultaneously. Our approach makes many directions dificult at once, so their error contributions accumulate. Concentrating their energy on one shared set ensures that samples containing few points from that set leave many directions poorly observed. Each observation inside the set supplies at most one linear constraint, while observations outside it carry little energy in these directions.

The relevant geometric tool is an uncertainty principle. Low Fourier degree restricts how much of a function’s energy can concentrate on a small set of inputs. Polyanskiy and Samorodnitsky identify a sharp asymptotic boundary for this tradeof [PS19]. Our statistical lower bound calls for a quantitative version that also accounts for how many directions can concentrate on the same set.

Geometric question. How small can the probability of one set of Boolean inputs be while that set still carries most of the energy of every function in a large low-degree subspace?

We provide an answer via a quantitative subspace uncertainty principle, with a remainder of uniformly optimal order. It yields matching upper and lower sample thresholds in a substantial range of model dimensions. For unrestricted ambient dimension, the leading exponential factor is $e ^ { 2 k }$ , compared with $2 ^ { k }$ for noiseless identification. Our refinement also establishes exponential savings at finite ambient dimension.

## 1.1 Preliminaries

Let $\mu _ { d }$ be uniform measure on $\{ - 1 , 1 \} ^ { d }$ . The Walsh characters $\begin{array} { r } { \chi _ { S } ( x ) = \prod _ { i \in S } x _ { i } } \end{array}$ form an orthonormal basis, and

$$
\mathcal { P } _ { d , k } = \operatorname { s p a n } \{ \chi _ { S } : | S | \leq k \} , \qquad D _ { d , k } = \dim \mathcal { P } _ { d , k } = \sum _ { j = 0 } ^ { \operatorname* { m i n } ( k , d ) } { \binom { d } { j } } .
$$

Thus d counts coordinates and k bounds interaction order. The learner receives a fixed m-dimensiona $V \subseteq \mathcal { P } _ { d , k }$ , including an evaluable basis. There are no computational restrictions. Choose a population-orthonormal basis $\Phi _ { V } = ( \phi _ { 1 } , \ldots , \phi _ { m } ) ^ { \top }$

We write $\begin{array} { r } { \mathbb { E } [ f ] = \int f ( x ) d \mu _ { d } ( x ) } \end{array}$ and $\begin{array} { r } { \mathbb { E } _ { n } [ f ] = n ^ { - 1 } \sum _ { i } f ( X _ { i } ) } \end{array}$ . The random experiment is

$$
X _ { i } \overset { \mathrm { i i d } } { \sim } \mu _ { d } , \qquad Y _ { i } = f _ { \boldsymbol { \theta } } ( X _ { i } ) + \xi _ { i } , \qquad f _ { \boldsymbol { \theta } } = { \boldsymbol { \theta } } ^ { \top } \Phi _ { V } , \qquad \xi _ { i } \overset { \mathrm { i i d } } { \sim } N ( 0 , \sigma ^ { 2 } ) ,\tag{1}
$$

with independent inputs and noise, $\sigma > 0$ , and unknown $\theta \in \mathbb { R } ^ { m }$ . The learner knows V and $\mu _ { d }$ and may know $\sigma . \ V$ and θ are fixed before sampling. Given the observations $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ , it outputs an estimate $\widehat { \theta } \in \mathbb { R } ^ { m }$ , defining ${ \widehat { f } } = { \widehat { \theta } } ^ { \top } \Phi _ { V }$ . By orthonormality, the population prediction loss is $\mathbb { E } \Big [ ( \widehat { f } - f _ { \theta } ) ^ { 2 } \Big ] = \| \widehat { \theta } - \theta \| _ { 2 } ^ { 2 }$

The empirical covariance and its population counterpart are

$$
{ \widehat { \boldsymbol { \Sigma } } } _ { V } = { \frac { 1 } { n } } \sum _ { i } \Phi _ { V } ( X _ { i } ) \Phi _ { V } ( X _ { i } ) ^ { \top } , \qquad { \boldsymbol { \Sigma } } _ { V } = \mathbb { E } [ \Phi _ { V } ( X ) \Phi _ { V } ( X ) ^ { \top } ] = I _ { m } .
$$

For a nonnegative, possibly infinite random variable $Z ,$ define $Q _ { 1 - \delta } ( Z ) = \operatorname* { i n f } \{ r \geq 0 : \mathbb { P } ( Z \leq r ) \geq$ $1 - \delta \}$ , with inf $\mathcal { O } = + \infty$ . Throughout, we write $t = \log ( 1 / \delta )$ for the confidence parameter. The supplied-model minimax risk is

$$
\mathcal { R } _ { n , \delta } ^ { * } ( V ) = \operatorname* { i n f } _ { \widehat { \theta } _ { V } } \operatorname* { s u p } _ { \theta \in \mathbb { R } ^ { m } } Q _ { 1 - \delta , \theta } \big ( \| \widehat { \theta } _ { V } - \theta \| _ { 2 } ^ { 2 } \big ) .
$$

Probabilities include inputs, noise, and any estimator randomness.

Our main object of interest is the minimax prediction risk for the worst admissible subspace, with the learner allowed to adapt its estimator to the supplied $V$

Definition 1.1 (Worst-subspace minimax risk). Define

$$
\Re _ { n , \delta } ( d , k , m ) : = \operatorname* { s u p } _ { \stackrel { V \subseteq \mathcal { P } _ { d , k } } { \dim V = m } } \mathcal { R } _ { n , \delta } ^ { * } ( V ) = \operatorname* { s u p } _ { \stackrel { V \subseteq \mathcal { P } _ { d , k } } { \dim V = m } } \operatorname* { i n f } _ { \theta \in \mathbb { R } ^ { m } } Q _ { 1 - \delta , \theta } ( \| \widehat { \theta } _ { V } - \theta \| _ { 2 } ^ { 2 } ) .
$$

Remark 1.2 (Why the subspace is known). Giving V to the learner separates coeficient estimation from discovering the relevant low-degree directions. Without this information, even $m = 1$ permits every degree-at-most-k target, since each lies in its own one-dimensional span. Knowing V fixes the number of unknown coeficients at m, allowing us to isolate how a degree bound k restricts the complexity of the known model and the resulting dificulty of estimation.

We ask how many samples are needed before this risk reaches the parametric rate $\sigma ^ { 2 } ( m +$ $\log ( 1 / \delta ) ) / n$ . The following threshold is the central quantity we determine.

Definition 1.3 (Parametric threshold). For a fixed benchmark constant $A \geq 1 6$ and $t = \log ( 1 / \delta )$ let $N _ { \mathrm { p a r } } ( d , k , m , A , \delta )$ be the least integer $N \geq 1$ such that

$$
\Re _ { n , \delta } ( d , k , m ) \leq { \frac { A \sigma ^ { 2 } ( m + t ) } { n } } \qquad { \it f o r \ e v e r y \ n \geq N } .
$$

We take $A \geq 1 6$ to simplify the proof constants; the value 16 has no special significance. Our results have the same leading exponent for every fixed $A \geq 1 6$

We also track when empirical squared norms uniformly preserve a fixed fraction of population squared norms.

Definition 1.4 (Stable-sampling threshold). For fixed $c \in ( 0 , 1 )$ , let $N _ { \mathrm { s t a b } } ( d , k , m , c , \delta )$ be the least integer $N \geq 1$ such that, for every $n \geq N$ and every fixed eligible V ,

$$
\mathbb { P } \{ \mathbb { E } _ { n } [ h ^ { 2 } ] \ge c \mathbb { E } [ h ^ { 2 } ] \ f o r \ a l l \ h \in V \} \ge 1 - \delta .
$$

Equivalently, $\mathbb { P } \{ \widehat { \Sigma } _ { V } \succeq c I _ { m } \} \geq 1 - \delta$

Our proof of Theorem 1.5 will reveal that, in its stated parameter range, $N _ { \mathrm { s t a b } }$ has the same worst-subspace exponential scale as $N _ { \mathrm { p a r } }$ , with constants depending on $q _ { 0 }$ and c. Suficiently strong stable sampling guarantees the parametric regression rate, but need not be necessary for it.

Throughout, all logarithms are natural. We define the binary entropy and the uncertainty curve by

$$
\begin{array} { r } { { \mathsf { H } } ( u ) = - u \log u - ( 1 - u ) \log ( 1 - u ) , \qquad { \Psi } ( q ) = \log 2 - { \mathsf { H } } \biggl ( \frac { 1 } { 2 } - \sqrt { q ( 1 - q ) } \biggr ) , } \end{array}
$$

with 0 log $0 = 0 , u \in [ 0 , 1 ]$ , and $q \in [ 0 , 1 / 2 ]$ . Set

$$
E _ { d , k } = d \Psi ( k / d ) , \qquad M _ { d , k } = { \binom { d } { \lfloor k ^ { 1 / 3 } \rfloor } } .
$$

Further notation. For real numbers, $a \vee b = \operatorname* { m a x } \{ a , b \} , a \wedge b = \operatorname* { m i n } \{ a , b \}$ , and $( a ) _ { + } = \operatorname* { m a x } \{ a , 0 \}$ For symmetric matrices, $A \succeq B$ means $a ^ { \top } ( A - B ) a \geq 0$ for every a. We use $\begin{array} { r } { \| a \| _ { 2 } ^ { 2 } = \sum _ { j } a _ { j } ^ { 2 } } \end{array}$ for vectors and $\| f \| _ { 2 } ^ { 2 } = \mathbb { E } [ f ^ { 2 } ] , \langle f , g \rangle = \mathbb { E } [ f g ]$ for functions on the cube under discussion. The indicator of E is ${ \bf 1 } _ { E }$ We abbreviate the feature map $\bar { \Phi } _ { V } = ( \phi _ { 1 } , \ldots , \phi _ { m } ) ^ { \top }$ by Φ when V is clear, and write ${ \widehat { f } } ( S ) = \mathbb { E } [ f \chi _ { S } ]$ for Fourier coeficients. The Rayleigh quotient of a symmetric matrix B is $\mathcal { R } _ { B } ( a ) = a ^ { \top } B a / \| a \| _ { 2 } ^ { 2 }$ for $a \neq 0$ . We write $a \lesssim b$ (equivalently $b \gtrsim a )$ for an inequality up to a positive constant, and $a \asymp b$ for both inequalities; allowed parameter dependence is stated locally. Finally, $\begin{array} { r } { \chi _ { r } ^ { 2 } = \sum _ { j = 1 } ^ { r } G _ { j } ^ { 2 } } \end{array}$ for independent $G _ { j } \sim N ( 0 , 1 )$ has the chi-squared distribution with r degrees of freedom, and $Z \succeq _ { \mathrm { s t } } Y$ means $\mathbb { P } ( Z > \dot { u } ) \ge \mathbb { P } ( Y > u )$ for every real u.

## 1.2 Main results

Throughout, $1 \leq k \leq q _ { 0 } d$ for fixed $q _ { 0 } < 1 / 2$ . Constants may depend on $q _ { 0 }$ and fixed target parameters, but not on $d , k , m , n , \delta$

## 1.2.1 Sharp sample thresholds for regression

Our first result determines when the worst-subspace minimax risk reaches the parametric rate, with the dependence on degree and ambient dimension captured by $E _ { d , k }$

Theorem 1.5 (Minimax sample threshold). Fix $q _ { 0 } \in ( 0 , 1 / 2 )$ and $A \geq 1 6$ . For $1 \leq k \leq q _ { 0 } d _ { \mathrm { { ; } } }$ $1 \leq m \leq D _ { d , k }$ and $0 < \delta \le 1 / 4$ , put $t = \log ( 1 / \delta )$ . If $m \leq M _ { d , k } \lor t ,$ , then

$$
\left| \log \frac { N _ { \mathrm { p a r } } ( d , k , m , A , \delta ) } { m + t } - E _ { d , k } \right| \leq C _ { q _ { 0 } , A } k ^ { 1 / 3 } .\tag{2}
$$

The upper bound holds for every feasible m and is attained by ordinary least squares.

Remark 1.6 (Polynomial Range). In particular, for every fixed $B > 0$ , equation (2) holds for every feasible $m \leq d ^ { B }$ , with constants depending additionally on B: for suficiently large k depending on $B , M _ { d , k } \geq d ^ { B }$ , while for bounded k, the estimate follows from $N _ { \mathrm { p a r } } \gtrsim m + t$ and $E _ { d , k } \leq 2 k$

Taking the worst case over all ambient dimensions gives a particularly simple threshold, valid for every model dimension m.

Corollary 1.7 (Unrestricted ambient dimension). For integers $k , m \ge 1$ , fixed $A \ge 1 6$ , and $0 < \delta \le 1 / 4$ , put $t = \log ( 1 / \delta )$ and define

$$
N _ { \mathrm { p a r } } ^ { \infty } ( k , m , A , \delta ) : = \operatorname* { s u p } _ { \stackrel { d \geq 1 } { m \leq \bar { D } _ { d , k } } } N _ { \mathrm { p a r } } ( d , k , m , A , \delta ) \cdot
$$

Then

$$
\left| \log \frac { N _ { \mathrm { p a r } } ^ { \infty } ( k , m , A , \delta ) } { m + t } - 2 k \right| \leq C _ { A } k ^ { 1 / 3 } .
$$

The analogous statement holds for the stable-sampling threshold, with a constant depending on its fixed conditioning level c.

The upper bound is uniform over all ambient dimensions. For the lower bound, choosing d suficiently large makes $m \le M _ { d , k }$ and $E _ { d , k } = 2 k + O ( k ^ { 1 / 3 } )$ . See Appendix F.1 for details.

The finite-dimensional correction. The dependence on d can substantially reduce the threshold. As $k  \infty$ with $k / d \to 0$ , provided that $m \leq M _ { d , k }$ or $t \geq m$ 2

$$
\log \frac { N _ { \mathrm { p a r } } ( d , k , m , A , \delta ) } { m + t } = 2 k - \frac { 2 k ^ { 2 } } { 3 d } + O _ { A } \bigg ( \frac { k ^ { 3 } } { d ^ { 2 } } + k ^ { 1 / 3 } \bigg ) .
$$

When $k \ll d \ll k ^ { 5 / 3 }$ , the correction $2 k ^ { 2 } / ( 3 d )$ dominates both error terms. Consequently, the threshold is smaller than its unrestricted-dimension counterpart by a factor

$$
\frac { N _ { \mathrm { p a r } } ^ { \infty } ( k , m , A , \delta ) } { N _ { \mathrm { p a r } } ( d , k , m , A , \delta ) } = \exp \left\{ \left( \frac { 2 } { 3 } + o ( 1 ) \right) \frac { k ^ { 2 } } { d } \right\} .
$$

This follows from the expansion of Ψ, see Appendix F.1.

## 1.2.2 The exponential cost of noise

A direct statistical consequence of Theorem 1.5 is an exponential separation between noiseless identification and noisy prediction.

Indeed, concavity of Ψ, together with $\Psi ( 0 ) = 0$ and $\Psi ( 1 / 2 ) = \log 2$ , gives $\Psi ( q ) \geq 2 q$ log 2 for $\begin{array} { r } { 0 \leq q \leq \frac { 1 } { 2 } } \end{array}$ and hence $E _ { d , k } \geq k \log { 4 }$ . Thus, for every $1 \leq k \leq q _ { 0 } d$ with fixed $q _ { 0 } < 1 / 2$ , whenever $m \le M _ { d , k }$ or $t \geq m$

$$
N _ { \mathrm { p a r } } ( d , k , m , A , \delta ) \geq ( m + t ) 4 ^ { k } \exp \{ - C _ { q _ { 0 } , A } k ^ { 1 / 3 } \} .
$$

In contrast, $C 2 ^ { k } ( m + t )$ samples sufice for noiseless identification in every admissible subspace. Indeed, a nonzero degree-at-most-k polynomial is nonzero on at least $\mathrm { ~ a ~ } 2 ^ { - k }$ fraction of the cube. To see this, choose a monomial of maximal degree $r \leq$ k with nonzero coeficient. For every assignment of the other coordinates, this coeficient remains unchanged, so the resulting polynomial is nonzero on at least one of the $2 ^ { r }$ assignments to those r coordinates. Whenever the evaluation matrix has rank below $m ,$ its null space therefore contains a function that the next independent input detects with probability at least $2 ^ { - k }$ , increasing the rank. Consequently,

$$
\mathbb { P } ( \operatorname { r a n k } < m ) \leq \mathbb { P } \{ \mathrm { B i n } ( n , 2 ^ { - k } ) < m \} \leq e ^ { - t } \qquad \mathrm { i f ~ } n \geq 8 \cdot 2 ^ { k } ( m + t ) ,
$$

by a standard binomial lower-tail bound. Full rank permits exact recovery of the coeficients.

The separation is even larger as $k  \infty$ with $k / d  0 \colon$ the expansion in Appendix F.1 gives $E _ { d , k } = 2 k + o ( k )$ . In the range m $\le M _ { d , k }$ or $t \geq m$ , where our upper and lower bounds agree up to an $O ( k ^ { 1 / 3 } )$ error in the exponent, the parametric threshold is therefore $( m + t ) \exp \{ 2 k + o ( k ) \}$ compared with the noiseless upper bound $O ( ( m + t ) 2 ^ { k } )$ .

## 1.2.3 A quantitative subspace uncertainty principle

Our geometric result quantifies energy concentration jointly over a subspace.

Theorem 1.8 (Subspace uncertainty principle). Fix $\rho \in ( 0 , 1 )$ . For every set $E _ { i }$ , if some nonzero $h \in \mathcal { P } _ { d , k }$ satisfies $\mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] \leq \rho \mathbb { E } [ h ^ { 2 } ]$ , then

$$
\mu _ { d } ( E ) \geq \exp \{ - E _ { d , k } - C _ { \rho , q _ { 0 } } k ^ { 1 / 3 } \} .
$$

Conversely, there exist $W \subseteq { \mathcal { P } } _ { d , k }$ of dimension $M _ { d , k }$ and a permutation-invariant set E such that

$$
\mu _ { d } ( E ) \leq \exp \{ - E _ { d , k } + C _ { \rho , q _ { 0 } } k ^ { 1 / 3 } \} , \qquad \mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] \leq \rho \mathbb { E } [ h ^ { 2 } ] \quad ( h \in W ) .\tag{3}
$$

For suficiently large $k , E$ is a Hamming ball.

We also refer to the parameter $\rho$ as leakage. Let $p _ { \rho } ( d , k , m )$ be the smallest probability of a set admitting such an m-dimensional concentrating subspace. The theorem implies

$$
\left| \log \frac { 1 } { p _ { \rho } ( d , k , m ) } - E _ { d , k } \right| \le C _ { \rho , q _ { 0 } } k ^ { 1 / 3 } , \qquad 1 \le m \le M _ { d , k } .\tag{4}
$$

It gives both a one-dimensional obstruction to concentration and a subspace construction at the same precision, uniformly even as $k / d \to 0$ . The next result shows that the order of this uniform remainder is optimal and cannot be improved to $o ( k ^ { 1 / 3 } )$ in general.

Proposition 1.9 (Optimal order of the uniform remainder). There are a fixed $\rho _ { * } \in ( 0 , 1 )$ , integers $d _ { k } \geq k ^ { 3 }$ , and $C < \infty$ such that, for all suficiently large k and all $1 \leq m \leq { \binom { d _ { k } } { \lfloor k ^ { 1 / 3 } \rfloor } }$ 2

$$
2 k ^ { 1 / 3 } \leq \log \frac { 1 } { p _ { \rho _ { * } } ( d _ { k } , k , m ) } - E _ { d _ { k } , k } \leq C k ^ { 1 / 3 } .\tag{5}
$$

## 1.3 Related work

Minimax regression and low-degree learning. Mourtada relates expected minimax leastsquares risk to inverse sample covariance [Mou22]. El Hanchi, Maddison, and Erdogdu give the exact quantile identity used here [EHE24]. We do not prove any new Gaussian minimax theorem. Rather, we determine its consequences for the worst known low-degree subspace by controlling the empirical spectrum sharply. Learning bounded low-degree targets without a supplied subspace [EI22; EIS23] is a diferent problem: target normalization and model selection change both the relevant error guarantees and the role of dimension. Noiseless interpolation and Reed–Muller erasure recovery [ASW15; BHSS22] concern rank rather than noise amplification. These distinctions explain why knowing the model removes model selection but leaves a substantial statistical question about random observation.

Random least squares and sampling stability. Cohen, Davenport, and Leviatan contro random least squares through the pointwise envelope $\mathrm { s u p } _ { x } \| \Phi _ { V } ( x ) \| _ { 2 } ^ { 2 }$ [CDL13]. Cohen and Migliorati show that choosing a model-dependent sampling measure and suitable weights permits stability with sample size of order m log m [CM17]. In contrast, our input law is fixed and uniform: the learner can adapt its estimator to V, but cannot redirect observations toward informative regions. We seek the worst geometry compatible with Fourier degree, and obtain a uniform lower bound on empirical energy even when the feature vectors can have very large norms. This is a one-sided sampling-discretization problem [KKLT22], closely related to lower bounds on empirical covariance matrices [Yas14; Oli16] and the small-ball method [Men14; Men21].

Uncertainty and log-Sobolev inequalities. Log-Sobolev inequalities relate concentration of a function’s energy to its Fourier complexity. With the normalization $\begin{array} { r } { \mathcal { D } ( f , f ) = \sum _ { S } | S | \widehat { f } ( S ) ^ { 2 } } \end{array}$ , the classical logarithmic Sobolev inequality on the cube [BLM13, Chapter 5], for $\mathbb { E } [ f ^ { 2 } ] = 1$ , reads

$$
\begin{array} { r } { \operatorname { E n t } _ { \mu _ { d } } ( f ^ { 2 } ) : = \mathbb { E } [ f ^ { 2 } \log f ^ { 2 } ] \leq 2 \mathcal { D } ( f , f ) \leq 2 k \qquad ( f \in \mathcal { P } _ { d , k } ) . } \end{array}
$$

For an exactly supported unit function, Jensen’s inequality gives Ent $( f ^ { 2 } ) \geq \log ( 1 / \mu _ { d } ( \operatorname { s u p p } f ) )$ . Thus entropy inequalities already constrain spatial concentration. For approximate concentration, the retained fraction $1 - \rho$ also enters and a sharp fixed-leakage boundary requires more than this elementary bound. Samorodnitsky’s modified log-Sobolev inequality strengthens the entropy–energy relation and connects it to small-set geometry [Sam08]. Polyanskiy and Samorodnitsky develop nonlinear log-Sobolev inequalities and improved hypercontractivity to obtain their uncertainty curve [PS19]. We use the same uncertainty curve Ψ, but sharpen the energy-concentration bounds and extend them jointly to every function in a large subspace.

Hypercontractivity and sharp moment bounds. The classical Bonami–Beckner inequality [Bon70; Bec75] bounds higher moments of low-degree polynomials in terms of their second moment; see Lemma B.1. Its well-known fourth-moment specialization gives $\mathbb { E } [ h ^ { 4 } ] \leq 9 ^ { k }$ for a unit-norm degree-at-most-k polynomial, yielding a loose upper bound of order $9 ^ { k } ( m + t )$ on the worst-subspace parametric threshold, for fixed target constants. However, using moments approaching two already gives exp $2 k + O ( k ^ { 1 / 3 } )$ from classical hypercontractivity (Proposition B.2), which is sharp for unrestricted ambient dimension. For our dimension-dependent results, classical hypercontractivity, even when tuned for the optimal moment parameter, is not sharp enough. Instead, to match the sharper finite-dimensional exponent $E _ { d , k } = d \Psi ( k / d )$ , we optimize the Kirshner–Samorodnitsky moment bound [KS21].

Harmonic analysis on Boolean slices. Harmonic multilinear polynomials provide the additional directions in our construction. Filmus constructs an explicit orthogonal basis for functions on a

Boolean slice, organized by harmonic degree [Fil16]. Filmus–Mossel establish the dimension formula for each harmonic degree and describe how inner products behave under exchangeable measures [FM19]. We use these latter results directly: harmonic polynomials let us turn one concentrated polynomial into many orthogonal functions whose energy concentrates on the same set.

Krawchouk polynomials and concentration. Krawchouk polynomials connect Fourier analysis on the cube with the geometry of Hamming distance. Levenshtein uses them to derive universal bounds for codes and designs in Hamming spaces [Lev95]. Their extreme zeros also enter the analysis of sum-of-squares bounds for binary polynomial optimization [SL23]. The extreme zeros indicate where the polynomial in our subsequent construction concentrates its energy, providing a geometric connection with these classical results.

Clipping and empirical quadratic processes. Combining truncation with empirical-process bounds is an established method for controlling empirical quadratic forms from below. Mendelson uses bounded Lipschitz approximations in the small-ball method [Men14], and clipped squares together with contraction and concentration in [Men21]. We follow this approach, using Bousquet’s inequality [Bou02] to track the dependence on the clipping level explicitly. Our contribution to this approach is a sharper degree- and dimension-dependent clipping scale, derived from the Kirshner–Samorodnitsky moment bound.

Airy asymptotics and sharpness. Airy functions arise in the asymptotic behavior of classical orthogonal polynomials near the outer edges of their oscillatory region. For Krawchouk polynomials, Dai and Wong obtain Airy asymptotics as part of their global analysis at a fixed positive degreeto-size ratio [DW07]. Our sharpness argument uses the convergence of the Hermite kernel to the Airy kernel [DG07; SX22]. From this, we construct a polynomial with fixed positive energy on a Gaussian tail interval, then transfer it to the cube and use harmonic polynomials to construct many orthogonal directions concentrating on the same set. The resulting tail probability establishes the optimal order $k ^ { 1 / 3 }$ of the uniform uncertainty remainder.

## 2 Constructing a polynomial concentrated on a Hamming ball

We first construct one unit polynomial whose energy is concentrated on a small set of inputs. Any such set could serve the statistical lower bound, but we aim for a Hamming ball. The motivation behind this choice is threefold: (i) its probability follows a binomial tail, naturally giving rise to the small sets we are looking for; (ii) radial polynomials admit a convenient three-term recurrence in the Krawchouk basis; and (iii) its permutation symmetry allows the later extension to many directions within V. In particular, consider $S _ { D } = x _ { 1 } + \cdot \cdot \cdot + x _ { D }$ , for which $E _ { u } = \{ S _ { D } \geq u \}$ is the ball around the all-plus vertex with radius $\lfloor ( D - u ) / 2 \rfloor$ . Our aim is to place most of a polynomial’s energy where $S _ { D }$ is well above its typical scale $\sqrt { D }$ under the uniform measure.

## 2.1 Designing the polynomial in the Krawchouk basis

We seek a polynomial whose energy is concentrated where $S _ { D } ( x ) = x _ { 1 } + \cdot \cdot \cdot + x _ { D }$ is unusually large. The set $E _ { u } = \{ S _ { D } \geq u \}$ is a Hamming ball, and increasing u shrinks it. Since $E _ { u }$ depends only on $S _ { D }$ , we look for a polynomial in $S _ { D }$

Krawchouk polynomials form an orthonormal basis for such functions under the uniform measure, and their three-term recurrence lets us express concentration near a chosen value of $S _ { D }$ as a condition on the polynomial’s coeficients.

Our polynomial must satisfy four constraints:

• Bounded degree: deg $p \leq \ell .$

• Unit norm: $\mathbb { E } [ p ^ { 2 } ] = 1$ , so $p ^ { 2 } d \mu _ { D }$ is a probability measure.

• Radial symmetry: p depends only on $S _ { D }$ , allowing the harmonic lift in Section 4.

• Concentration at a high threshold: for a prescribed leakage $\rho ,$

$$
\mathbb { E } [ p ^ { 2 } \mathbf { 1 } _ { \{ S _ { D } < u \} } ] \le \rho ,
$$

with u large enough that $E _ { u }$ has the required small probability.

The Krawchouk basis makes the first three constraints explicit and gives a useful algebraic description of the fourth. Define

$$
e _ { j } ( x ) = { \binom { D } { j } } ^ { - 1 / 2 } \sum _ { | S | = j } \chi _ { S } ( x ) , \qquad 0 \leq j \leq D .
$$

These are orthonormal radial functions of degree $j$ that form a basis for all functions of $S _ { D }$ . Thus $\textstyle p = \sum _ { j = 0 } ^ { \ell } c _ { j } e _ { j }$ with $\textstyle \sum _ { j = 0 } ^ { \ell } c _ { j } ^ { 2 } = 1$ automatically satisfies the degree, symmetry, and normalization constraints.

There is also a connection to heavy tails: Krawchouk polynomials nearly attain sharp highermoment bounds among polynomials of the same degree and $L _ { 2 }$ norm. Kirshner and Samorodnitsky [KS21] establish this near extremality. However, large moments alone do not give our prescribed concentration on one ball. Indeed, $e _ { j } ( - x ) = ( - 1 ) ^ { j } e _ { j } ( x )$ , so a single $e _ { j }$ has equally heavy upper and lower tails. We will combine several degrees to concentrate energy in the upper tail.

Multiplication measures concentration. For any unit $p ,$

$$
\varepsilon ^ { 2 } : = \| ( S _ { D } - \tau ) p \| _ { 2 } ^ { 2 } = \mathbb { E } [ ( S _ { D } - \tau ) ^ { 2 } p ^ { 2 } ]
$$

describes the mean squared distance of $S _ { D }$ from $\tau$ under $p ^ { 2 } d \mu _ { D }$ . Hence

$$
\mathbb { E } [ p ^ { 2 } \mathbf { 1 } _ { \{ S _ { D } < \tau - a \} } ] \leq \frac { \varepsilon ^ { 2 } } { a ^ { 2 } } \qquad ( a > 0 ) .
$$

For $\varepsilon > 0$ , taking $a = \varepsilon / \sqrt { \rho }$ gives the desired leakage on the ball with cutof $u = \tau - \varepsilon / \sqrt { \rho } .$

If $\varepsilon = 0$ , all energy is within $S _ { D } = \tau$ , so $u = \tau$ works. We therefore want a large center τ and a small multiplication residual $\varepsilon .$ Both quantities matter: a large residual would force us to lower the cutof, making the ball larger.

The following recurrence allows us to express this objective in terms of the coeficients $c _ { j }$

Lemma 2.1 (Radial recurrence). Put $a _ { j } = \sqrt { ( j + 1 ) ( D - j ) }$ , with $a _ { - 1 } = a _ { D } = 0$ and $e _ { - 1 } = e _ { D + 1 } =$ 0. Then

$$
\begin{array} { r } { S _ { D } e _ { j } = a _ { j } e _ { j + 1 } + a _ { j - 1 } e _ { j - 1 } . } \end{array}\tag{6}
$$

Each $e _ { j }$ is a univariate polynomial of degree $j$ in $S _ { D }$ .

Proof. Multiplying a Walsh character by $x _ { i }$ adds i to its index set if absent and removes it if present. Each resulting set of size $j + 1$ occurs $j + 1$ times, and each set of size $j - 1$ occurs $D - j + 1$ times. Normalizing gives (6). Starting with $e _ { 0 } = 1$ , solving successively for $e _ { j + 1 }$ proves the degree assertion. □

Consequently, multiplying p by $S _ { D }$ transforms its coeficient vector according to the linear map

$$
\begin{array} { r } { J _ { D } = \left( \begin{array} { c c c c c } { 0 } & { a _ { 0 } } & { 0 } & { \cdots } & { 0 } \\ { a _ { 0 } } & { 0 } & { a _ { 1 } } & { \ddots } & { \vdots } \\ { 0 } & { a _ { 1 } } & { 0 } & { \ddots } & { 0 } \\ { \vdots } & { \ddots } & { \ddots } & { \ddots } & { a _ { D - 1 } } \\ { 0 } & { \cdots } & { 0 } & { a _ { D - 1 } } & { 0 } \end{array} \right) , \qquad ( J _ { D } c ) _ { j } = a _ { j - 1 } c _ { j - 1 } + a _ { j } c _ { j + 1 } . } \end{array}
$$

Here $c _ { j } = 0$ outside $0 , \ldots , \ell .$ Orthonormality gives the exact identity

$$
\| ( S _ { D } - \tau ) p \| _ { 2 } = \| ( J _ { D } - \tau I ) c \| _ { 2 } .
$$

Choosing the coeficients $c _ { j }$ . We now seek a unit vector supported on degrees at most ℓ for which $\| ( J _ { D } - \tau I ) c \| _ { 2 }$ is small: in other words, an approximate eigenvector of $J _ { D }$ with approximate eigenvalue $\tau .$ Let $L$ denote the number of nonzero coeficients. Three constraints guide our construction:

• Use consecutive degrees. $J _ { D }$ couples only adjacent indices. Consecutive coeficients allow the two neighboring contributions to reinforce each other while gaps interrupt this reinforcement.

• Place them near ℓ. The weights $a _ { j }$ increase through the relevant range below $D / 2$ . Degrees near the largest permitted degree therefore give large weights. Over a short interval near $\ell ,$ both neighboring weights are close to $a _ { * } = \sqrt { \ell ( D - \ell ) }$ . If $c _ { j - 1 } + c _ { j + 1 } \approx 2 c _ { j }$ , then $( J _ { D } c ) _ { j } \approx 2 a _ { * } c _ { j }$ . This suggests $\tau = 2 a _ { * }$ . When $\ell \ll D$ , this center is approximately $2 \sqrt { D \ell }$ , well above the uniform scale $\sqrt { D }$ for large ℓ. A cutof close to it therefore defines a rare event.

• Taper them to zero. The degree restriction requires $c _ { \ell + 1 } = 0$ , but $( ( J _ { D } - \tau I ) c ) _ { \ell + 1 } = a _ { \ell } c _ { \ell }$ . An abrupt jump from $c _ { \ell } = L ^ { - 1 / 2 }$ to zero would therefore contribute $a _ { \ell } / \sqrt { L }$ to the residual:

$$
\| ( J _ { D } - \tau I ) c \| _ { 2 } \geq \frac { a _ { \ell } } { \sqrt { L } } .
$$

We instead make the coeficients rise gradually from zero and fall gradually back to zero, keeping $\Delta ^ { 2 } c _ { j } = c _ { j - 1 } - 2 c _ { j } + c _ { j + 1 }$ small, including at the ends of the selected interval.

We are hence looking to construct a smooth window: a coeficient sequence supported on L consecutive degrees and tapered to zero at both ends, with small second diferences. The next section constructs such a sequence. Increasing $L$ permits smaller second diferences, but includes matrix weights farther from $^ { a _ { * } }$ . Balancing these two errors determines L.

## 2.2 Choosing the coeficients and bounding the error

The next lemma makes this plan quantitative.

Lemma 2.2 (A concentrating polynomial). Fix $q _ { 0 } < 1 / 2$ . For all suficiently large $\ell \leq q _ { 0 } D$ , there is a unit radial polynomial p of exact degree ℓ such that

$$
\| ( S _ { D } - 2 \sqrt { \ell ( D - \ell ) } ) p \| _ { 2 } \leq C _ { q _ { 0 } } \sqrt { D } \ell ^ { - 1 / 6 } .\tag{7}
$$

Proof. We use the $L = \lfloor \ell ^ { 1 / 3 } \rfloor$ degrees $\ell - L + 1 , \ldots , \ell .$ . To make the coeficients small near the two ends of this interval, define

$$
b ( x ) = \left\{ \begin{array} { l l } { x ^ { 2 } ( 1 - x ) ^ { 2 } , } & { 0 \leq x \leq 1 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad c _ { \ell - L + r } = Z _ { L } ^ { - 1 / 2 } b \bigg ( \frac { r } { L + 1 } \bigg ) \quad ( 1 \leq r \leq L ) ,
$$

where $\begin{array} { r } { Z _ { L } = \sum _ { r = 1 } ^ { L } b ( r / ( L + 1 ) ) ^ { 2 } } \end{array}$ normalizes $\textstyle \sum _ { j } c _ { j } ^ { 2 }$ to one. All other coeficients are zero, including at integer indices outside $0 , \ldots , D$ . For large $\bar { \ell } , \bar { 2 } \leq L \leq \ell / 4$

We first bound the second diferences of this sequence. The function b is bounded above and below by positive constants on $[ 1 / 4 , 3 / 4 ]$ . A fixed positive fraction of the L sampling points lie in that interval, so $c L \leq Z _ { L } \leq C L$ for absolute positive constants $c , C$ . In particular, the normalization contributes a factor of order $L ^ { - 1 / 2 }$

Both b and its first derivative vanish at 0 and 1. Extending b by zero therefore leaves its derivative continuous, with $| b ^ { \prime } ( u ) - b ^ { \prime } ( v ) | \leq C | u - v |$ for all $u , v \colon$ inside $[ 0 , 1 ]$ its second derivative is bounded, and outside it the derivative is zero. For the sampling step $\delta _ { L } = 1 / ( L + 1 )$ , it follows that

$$
\left| b ( x + \delta _ { L } ) - 2 b ( x ) + b ( x - \delta _ { L } ) \right| = \left| \int _ { 0 } ^ { \delta _ { L } } \bigl ( b ^ { \prime } ( x + t ) - b ^ { \prime } ( x - \delta _ { L } + t ) \bigr ) d t \right| \le C \delta _ { L } ^ { 2 } .
$$

This estimate also holds when one of the sampled points is outside [0, 1]. Consequently, for $\Delta ^ { 2 } c _ { j } = c _ { j + 1 } - 2 c _ { j } + c _ { j - 1 }$ , every second diference has magnitude at most $C Z _ { L } ^ { - 1 / 2 } \delta _ { L } ^ { 2 } \leq C L ^ { - 5 / 2 }$ . Only the $L + 2$ indices $\ell - L , \ldots , \ell + 1$ can contribute, since at all other indices the three coeficients are zero. Hence

$$
\| \Delta ^ { 2 } c \| _ { \ell _ { 2 } ( \mathbb { Z } ) } = \left( \sum _ { j \in \mathbb { Z } } | \Delta ^ { 2 } c _ { j } | ^ { 2 } \right) ^ { 1 / 2 } \leq \sqrt { L + 2 } C L ^ { - 5 / 2 } \leq C L ^ { - 2 } .
$$

Set $\begin{array} { r } { p = \sum _ { j } c _ { j } e _ { j } } \end{array}$ , so $\| p \| _ { 2 } = 1$ and $c _ { \ell } > 0$

For $a _ { * } = \sqrt { \ell ( D - \ell ) }$ , the coeficient of $( S _ { D } - 2 a _ { * } ) p$ at $j$ is

$$
( ( J _ { D } - 2 a _ { * } I ) c ) _ { j } = a _ { * } \Delta ^ { 2 } c _ { j } + ( a _ { j } - a _ { * } ) c _ { j + 1 } + ( a _ { j - 1 } - a _ { * } ) c _ { j - 1 } .
$$

Whenever a shifted coeficient is nonzero, $| j - \ell | \leq L + 2$ and $j$ is comparable to $\ell .$ Since

$$
| a _ { j } ^ { 2 } - a _ { * } ^ { 2 } | \leq C D ( | j - \ell | + 1 ) , \qquad a _ { j } + a _ { * } \geq c _ { q _ { 0 } } \sqrt { D \ell } ,
$$

we have $| a _ { j } - a _ { * } | \leq C _ { q _ { 0 } } \sqrt { D / \ell } ( L + 3 )$ . Thus

$$
\| ( J _ { D } - 2 a _ { * } I ) c \| _ { 2 } \leq C _ { q _ { 0 } } \left( \frac { \sqrt { D \ell } } { L ^ { 2 } } + \sqrt { \frac { D } { \ell } } ( L + 3 ) \right) \leq C _ { q _ { 0 } } \sqrt { D } \ell ^ { - 1 / 6 } .
$$

The estimate includes the two indices $\ell - L$ and $\ell + 1$ , just outside the chosen interval of degrees. Orthogonality of the $e _ { j }$ proves (7). □

## 2.3 From an approximate eigenvector to a rare event

Lemma 2.2 places the energy of $p$ near $\tau = 2 \sqrt { \ell ( D - \ell ) }$ , but does not determine how much lies above $\tau .$ . The ball $\{ S _ { D } \ge \tau \}$ could therefore miss substantial energy. Instead, put its cutof at $u = \tau - a .$ , with $a > 0$ . Every excluded point then has distance greater than a from τ, so

$$
\mathbb { E } [ p ^ { 2 } \mathbf { 1 } _ { \{ S _ { D } < u \} } ] \leq \frac { \| ( S _ { D } - \tau ) p \| _ { 2 } ^ { 2 } } { a ^ { 2 } } \leq \frac { C _ { q _ { 0 } } ^ { 2 } D \ell ^ { - 1 / 3 } } { a ^ { 2 } } .
$$

Choosing $a = M _ { \rho , q _ { 0 } } \sqrt { D } \ell ^ { - 1 / 6 }$ with a suficiently large constant makes this leakage at most $\rho .$ This displacement is only $a / \tau = O _ { \rho , q _ { 0 } } ( \ell ^ { - 2 / 3 } )$ relative to the center. The next lemma also shows that enlarging the ball in this way costs only $O _ { \rho , q _ { 0 } } ( \ell ^ { 1 / 3 } )$ in its probability exponent.

Lemma 2.3 (One-dimensional concentration witness). Fix $\rho \in ( 0 , 1 )$ and $q _ { 0 } \in ( 0 , 1 / 2 )$ . There are constants $M _ { \rho , q _ { 0 } } , C _ { \rho , q _ { 0 } } > 0$ such that, for all suficiently large $\ell \le q _ { 0 } D$ , there is a unit radial polynomial p of exact degree ℓ for which the cutof $\tau = 2 \sqrt { \ell ( D - \ell ) }$ $u = \tau - M _ { \rho , q _ { 0 } } \sqrt { D } \ell ^ { - 1 / 6 } > 0$ satisfies

$$
\begin{array} { r } { { \mathbb E } [ p ^ { 2 } \mathbf { 1 } _ { \{ S _ { D } < u \} } ] \le \rho , \qquad \mu _ { D } ( S _ { D } \ge u ) \le \exp \{ - D \Psi ( \ell / D ) + C _ { \rho , q _ { 0 } } \ell ^ { 1 / 3 } \} . } \end{array}
$$

Proof. Step 1: retaining the energy. Take p from Lemma 2.2 and set $a = M _ { \rho , q _ { 0 } } \sqrt { D } \ell ^ { - 1 / 6 }$ , so $u = \tau - a$ . On $\{ S _ { D } < u \}$ , the distance from $S _ { D }$ to τ exceeds a. Consequently,

$$
\mathbb { E } [ p ^ { 2 } \mathbf { 1 } _ { \{ S _ { D } < u \} } ] \le \frac { \mathbb { E } [ ( S _ { D } - \tau ) ^ { 2 } p ^ { 2 } ] } { a ^ { 2 } } \le \frac { C _ { q _ { 0 } } ^ { 2 } D \ell ^ { - 1 / 3 } } { M _ { \rho , q _ { 0 } } ^ { 2 } D \ell ^ { - 1 / 3 } } \le \rho
$$

if $M _ { \rho , q _ { 0 } } \geq C _ { q _ { 0 } } / \sqrt { \rho }$ . Also $\tau \geq 2 \sqrt { ( 1 - q _ { 0 } ) D \ell }$ , so $a / \tau = O _ { \rho , q _ { 0 } } ( \ell ^ { - 2 / 3 } )$ . Thus $\ 0 < u < \tau < D$ for large ℓ.

Step 2: bounding the uniform probability. For $0 \leq z < 1$ , write $I ( z ) = \log 2 - { \mathsf { H } } ( ( 1 - z ) / 2 ) , I ^ { \prime } ( z ) =$ arctanh z. The binomial Chernof bound gives $\mu _ { D } ( S _ { D } \geq u ) \leq \exp \{ - D I ( u / D ) \}$ . At the center $\tau$ the rate is exactly

$$
D I ( \tau / D ) = D \Psi ( \ell / D ) .
$$

We therefore need to bound the loss in this rate when the cutof moves from τ down to $u .$ Since I<sup>′</sup> is increasing,

$$
D \Psi ( \ell / D ) - D I ( u / D ) = D \int _ { u / D } ^ { \tau / D } I ^ { \prime } ( z ) d z \leq ( \tau - u ) I ^ { \prime } ( \tau / D ) .\tag{8}
$$

The assumption $\ell / D \le q _ { 0 } < 1 / 2$ gives $\tau / D \leq 2 \sqrt { q _ { 0 } ( 1 - q _ { 0 } ) } < 1$ . On this interval, arctanh $z \le C _ { q _ { 0 } } z$ and therefore

$$
I ^ { \prime } ( \tau / D ) \leq C _ { q _ { 0 } } \sqrt { \ell / D } .
$$

Combining the last two bounds yields $D \Psi ( \ell / D ) - D I ( u / D ) \leq C _ { \rho , q _ { 0 } } \sqrt { D } \ell ^ { - 1 / 6 } \sqrt { \ell / D } = C _ { \rho , q _ { 0 } } \ell ^ { 1 / 3 }$ Substituting into the Chernof bound proves the second assertion. □

## 3 From moment bounds to stable sampling

The preceding construction shows how a low-degree polynomial can “hide” its energy on rarely sampled inputs. For an upper sampling bound, we need the converse: a level B at which clipping $h ^ { 2 }$ to $h ^ { 2 } \wedge B$ preserves a fixed amount of energy for every unit polynomial h. These clipped functions are bounded, so their empirical averages can be controlled uniformly over a known subspace. Since $\mathbb { E } _ { n } [ h ^ { 2 } ] \geq \mathbb { E } _ { n } [ h ^ { 2 } \wedge B ]$ , this gives the required lower bound on the empirical covariance without an upper spectral bound. The construction also bounds the necessary clipping level from below. Optimizing a moment bound will give the matching upper estimate.

For unit-norm $h ,$ define

$$
b _ { \eta } ( h ) = \operatorname* { i n f } \{ B > 0 : \mathbb { E } [ h ^ { 2 } \wedge B ] \geq \eta \} , \qquad T _ { d , k } ( \eta ) = \operatorname* { s u p } _ { h \in \mathcal { P } _ { d , k } } b _ { \eta } ( h ) .
$$

## 3.1 The Kirshner–Samorodnitsky moment bound

For $p \geq 2$ and $0 < q < 1 / 2$ , let $z \in ( 0 , 1 )$ be the unique solution of

$$
q = \frac { z ( ( 1 + z ) ^ { p - 1 } - ( 1 - z ) ^ { p - 1 } ) } { ( 1 + z ) ^ { p } + ( 1 - z ) ^ { p } } .\tag{9}
$$

Define

$$
F ( p , q ) = \log \frac { ( 1 + z ) ^ { p } + ( 1 - z ) ^ { p } } { 2 } - p q \log z - \frac { p } { 2 } { \sf H } ( q ) .\tag{10}
$$

At the endpoints, set $F ( p , 0 ) = 0$ and $F ( p , 1 / 2 ) = ( p / 2 - 1 )$ log 2, the respective limits of the formula.

The moment bound stated below is precisely [KS21, Corollary 1.4]. We verify uniqueness in Section A.2.

Theorem 3.1 (Kirshner–Samorodnitsky [KS21]). For $p \ge 2 , 0 \le k \le d / 2$ , and nonzero $f \in \mathcal { P } _ { d , k }$

$$
\frac { \mathbb { E } [ | f | ^ { p } ] } { ( \mathbb { E } [ f ^ { 2 } ] ) ^ { p / 2 } } \le \exp ( d F ( p , k / d ) ) ,
$$

with F defined in (10).

## 3.2 Cancellation at the second moment

The next lemma identifies $\Psi ( q )$ as the linear term in the moment bound near $p = 2$ , with no quadratic error. The cubic remainder here will be responsible for the $k ^ { 1 / 3 }$ correction in the subsequent clipping bound.

Lemma 3.2 (Cancellation at the second moment). For every $q _ { 0 } < 1 / 2$ , there is $C _ { 0 } < \infty$ such that, for $0 < q \le q _ { 0 }$ and $0 \leq s \leq 1 / 2$

$$
F ( 2 + 2 s , q ) \leq s \Psi ( q ) + C _ { 0 } q s ^ { 3 } .\tag{11}
$$

Consequently, $i f \mathbb { E } [ h ^ { 2 } ] = 1$ and deg $h \leq k \leq q _ { 0 } d _ { \mathrm { { \scriptsize ~ i ~ } } }$ then

$$
\log \mathbb { E } [ | h | ^ { 2 + 2 s } ] \leq s E _ { d , k } + C _ { 0 } k s ^ { 3 } .\tag{12}
$$

The proof uses an elementary variational comparison and Taylor’s theorem. Evaluating the variational formula at a carefully chosen function of s gives an upper bound whose second derivative vanishes at $s = 0$ , leaving only a cubic remainder. See Appendix A.2 for the details.

## 3.3 From moments to clipping

The remaining step is the standard moment-to-tail argument, which exploits that a moment slightly above two bounds the energy removed by truncation. We give the parameter choice explicitly because it determines the $k ^ { 1 / 3 }$ correction.

Proposition 3.3 (One-dimensional clipping). For every fixed $\eta \in ( 0 , 1 )$ ,

$$
| \log T _ { d , k } ( \eta ) - E _ { d , k } | \le C _ { \eta , q _ { 0 } } k ^ { 1 / 3 } .
$$

Upper bound. Step 1: bounding the energy removed by clipping. Let $\mathbb { E } [ h ^ { 2 } ] = 1 , L _ { \eta } = \log ( 1 / ( 1 - \eta ) )$ 2 and $0 < s \le 1 / 2$ . For $z \geq 0$ J,

$$
( z - B ) _ { + } \leq z \mathbf { 1 } _ { \{ z > B \} } \leq z \left( \frac { z } { B } \right) ^ { s } = z ^ { 1 + s } B ^ { - s } .
$$

Using Lemma 3.2,

$$
\begin{array} { r } { \mathbb { E } [ ( h ^ { 2 } - B ) _ { + } ] \le B ^ { - s } \mathbb { E } [ | h | ^ { 2 + 2 s } ] \le \exp \bigl ( - s ( \log B - E _ { d , k } ) + C _ { 0 } k s ^ { 3 } \bigr ) . } \end{array}
$$

Step 2: balancing the two error terms. To make the last exponent at most $- L _ { \eta }$ , it sufices to take log $B - E _ { d , k } = C _ { 0 } k s ^ { 2 } + L _ { \eta } / s$ . Balancing these terms suggests s of order $k ^ { - 1 / 3 }$ . Choose

$$
s = \frac { 1 } { 2 } \wedge \Big ( \frac { L _ { \eta } } { 2 C _ { 0 } k } \Big ) ^ { 1 / 3 } , \qquad \log B = E _ { d , k } + C _ { 0 } k s ^ { 2 } + \frac { L _ { \eta } } { s } .\tag{13}
$$

Then $\begin{array} { r } { \mathbb { E } [ ( h ^ { 2 } - B ) _ { + } ] \le 1 - \eta _ { \cdot } } \end{array}$ so $\mathbb { E } [ h ^ { 2 } \wedge B ] \geq \eta$ . If the second value in the minimum in (13) is selected, the two error terms are each $O _ { q _ { 0 } , \eta } ( k ^ { 1 / 3 } )$ . Otherwise, k is bounded in terms of $q _ { 0 } , \eta$ , and the same conclusion holds after increasing the constant. Taking the supremum proves the claim. □

Lower bound. Fix the retained fraction η. For large k, apply Lemma 2.3 with $D = d , \ell = k$ , and leakage $\rho = \eta / 2$ . Let $E = \{ S _ { d } \geq u \}$ be its small ball. Outside E, the polynomial has at most $\eta / 2$ energy; inside E, clipping at B leaves at most B at each input. Hence

$$
\begin{array} { r } { \mathbb { E } [ p ^ { 2 } \wedge B ] \leq \mathbb { E } [ p ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] + B \mu _ { d } ( E ) \leq \eta / 2 + B \mu _ { d } ( E ) . } \end{array}
$$

Every $B < \eta / ( 2 \mu _ { d } ( E ) )$ therefore retains less than η energy. By the definitions of $b _ { \eta }$ and $T _ { d , k } ( \eta )$

$$
T _ { d , k } ( \eta ) \geq b _ { \eta } ( p ) \geq \frac { \eta } { 2 \mu _ { d } ( E ) } \geq \frac { \eta } { 2 } \exp \{ E _ { d , k } - C _ { \eta , q _ { 0 } } k ^ { 1 / 3 } \} .
$$

For bounded k, the constant polynomial gives $T _ { d , k } ( \eta ) \geq \eta$ . Since $E _ { d , k } \leq 2 k$ , increasing $C _ { \eta , q _ { 0 } }$ ensures log $T _ { d , k } ( \eta ) \geq E _ { d , k } - C _ { \eta , q _ { 0 } } k ^ { 1 / 3 }$ in this case as well. □

## 3.4 Uniform empirical control after clipping

We now convert the clipping estimate into a uniform lower bound on empirical energy. The argument combines three standard techniques: symmetrization, contraction, and Bousquet’s inequality; see Appendix B.

Proposition 3.4 (Clipping gives stable sampling). Let V have dimension m and an orthonormal basis. Suppose $\mathbb { E } [ h ^ { 2 } \wedge B ] \geq \alpha$ for every unit $h \in V$ , where $B \geq 1$ and $0 < c < \alpha < 1$ . Then

$$
n \geq C _ { \alpha , c } B ( m + t ) \quad \Longrightarrow \quad \mathbb { P } ( \widehat { \Sigma } _ { V } \succeq c I _ { m } ) \geq 1 - e ^ { - t } .
$$

Proof. Let $\psi _ { B } ( u ) = u ^ { 2 } \wedge B$ , and let $S = \{ h \in V : \mathbb { E } [ h ^ { 2 } ] = 1 \}$ . This function is 2 B-Lipschitz and vanishes at zero. Define

$$
D _ { B } = \operatorname* { s u p } _ { h \in \mathcal { S } } ( \mathbb { E } [ \psi _ { B } ( h ) ] - \mathbb { E } _ { n } [ \psi _ { B } ( h ) ] ) , \qquad \overline { { D } } _ { B } = \operatorname* { m a x } \{ D _ { B } , 0 \} .
$$

Since $\overline { { D } } _ { B }$ is bounded by the corresponding absolute supremum, Theorem $\mathrm { B . 4 ( i ) }$ (symmetrization), followed by part (ii) (contraction) conditionally on the inputs with $\varphi = \psi _ { B }$ and $L = 2 { \sqrt { B } } , { \mathrm { g i v e s } }$

$$
\mathbb { E } \overline { { D } } _ { B } \leq 8 \sqrt { B } \mathbb { E } \operatorname* { s u p } _ { h \in S } \left| \frac { 1 } { n } \sum _ { i } \varepsilon _ { i } h ( X _ { i } ) \right| = 8 \sqrt { B } \mathbb { E } \left\| \frac { 1 } { n } \sum _ { i } \varepsilon _ { i } \Phi _ { V } ( X _ { i } ) \right\| _ { 2 } \leq 8 \sqrt { \frac { B m } { n } } .
$$

The last step uses Jensen’s inequality and $\begin{array} { r } { \mathfrak { Z } \left\| \frac { 1 } { n } \sum _ { i } \varepsilon _ { i } \Phi _ { V } ( X _ { i } ) \right\| _ { 2 } ^ { 2 } = \frac { 1 } { n } \mathbb { E } \| \Phi _ { V } ( X ) \| _ { 2 } ^ { 2 } = \frac { m } { n } } \end{array}$ . For a unit h, the centered function

$$
g _ { h } = \frac { \mathbb { E } [ \psi _ { B } ( h ) ] - \psi _ { B } ( h ) } { B }
$$

has absolute value at most one and variance at most $1 / B$ since $\mathbb { E } [ \psi _ { B } ( h ) ^ { 2 } ] \leq B \mathbb { E } [ \psi _ { B } ( h ) ] \leq B \mathbb { E } [ h ^ { 2 } ] =$ $B .$

Apply Theorem B.3 (Bousquet’s inequality) to these functions and the zero function. Its supremum is $Z = n \overline { { D } } _ { B } / B$ . After rescaling, with probability at least $1 - e ^ { - t }$ ，

$$
\overline { { D } } _ { B } \leq \mathbb { E } \overline { { D } } _ { B } + \sqrt { \frac { 2 B t } { n } ( 1 + 2 \mathbb { E } \overline { { D } } _ { B } ) } + \frac { B t } { 3 n } .
$$

When $n \geq C B ( m + t )$ with C large, the expectation is at most one. The right side is then bounded by

$$
C ^ { \prime } \left( { \sqrt { \frac { B ( m + t ) } { n } } } + { \frac { B ( m + t ) } { n } } \right) .
$$

Choose the sample-size constant so this is at most $\alpha - c .$ On that event, for every $h \in S$

$$
\begin{array} { r } { \mathbb { E } _ { n } [ h ^ { 2 } ] \ge \mathbb { E } _ { n } [ \psi _ { B } ( h ) ] \ge \mathbb { E } [ \psi _ { B } ( h ) ] - \overline { { D } } _ { B } \ge \alpha - ( \alpha - c ) = c . } \end{array}
$$

Finally, homogeneity gives the claim for all $h \in V$

## 4 Many directions on one Hamming ball

The one-dimensional construction gives one polynomial concentrated on a small Hamming ball. We now seek a whole subspace for which every function concentrates on the same ball.

## 4.1 Separating radial concentration from variation within slices

A radial polynomial $p ( S _ { d } )$ is constant on each slice $\{ x : S _ { d } ( x ) = s \}$ . The Krawchouk basis lets us choose its values across these slices so that its energy concentrates at large s. To obtain more than one direction, we multiply a shared radial factor by polynomials that can distinguish points within a slice:

$$
h _ { f } ( \boldsymbol { x } ) = f ( \boldsymbol { x } ) p _ { r } ( S _ { d } ( \boldsymbol { x } ) ) , \qquad f \in \mathcal { H } _ { d , r } .
$$

Here $\mathcal { H } _ { d , r }$ is the space of homogeneous multilinear degree-r harmonic polynomials, defined by the polynomial identity

$$
\sum _ { i = 1 } ^ { d } \partial _ { i } f = 0 .
$$

For example, $\mathcal { H } _ { d , 0 }$ consists of constants, while $\begin{array} { r } { \mathcal { H } _ { d , 1 } = \{ \sum _ { i } v _ { i } x _ { i } : \sum _ { i } v _ { i } = 0 \} } \end{array}$

We choose deg $\begin{array} { r } { p _ { r } \le k - r } \end{array}$ , so each product has Boolean degree at most k. The available number of harmonic directions is

$$
\dim { \mathcal { H } } _ { d , r } = { \binom { d } { r } } - { \binom { d } { r - 1 } } , \qquad { \binom { d } { - 1 } } = 0 ,
$$

see [FM19, Corollary 3.9(b)].

## 4.2 Uniform energy control across harmonic directions

All unit polynomials in $\mathcal { H } _ { d , r }$ have the same distribution of squared energy across slices. A shared radial factor therefore changes this distribution identically in every direction. Moreover, diferent harmonic degrees remain orthogonal under radial weighting. These facts follow from the identity below, a direct specialization of [FM19, Theorem 3.10]. See Appendix C for the proof.

Lemma 4.1 (Radial-weight identity). For integers $0 \leq r , s \leq \lfloor d / 2 \rfloor , f \in \mathcal { H } _ { d , r } , g \in \mathcal { H } _ { d , s }$ , and any real function H of the coordinate sum,

$$
\mathbb { E } _ { \mu _ { d } } [ f g H ( S _ { d } ) ] = \left\{ \begin{array} { l l } { 0 , } & { r \neq s , } \\ { \langle f , g \rangle _ { d } \mathbb { E } _ { \mu _ { d - 2 r } } [ H ( S _ { d - 2 r } ) ] , } & { r = s . } \end{array} \right.\tag{14}
$$

Here $\langle f , g \rangle _ { d } = \mathbb { E } _ { \mu _ { d } } [ f g ]$ and $S _ { 0 } = 0$

To see why $d - 2 r$ appears, consider the unit harmonic polynomial

$$
Q _ { r } ( x ) = \prod _ { j = 1 } ^ { r } { \frac { x _ { 2 j - 1 } - x _ { 2 j } } { \sqrt { 2 } } } .
$$

Taking $f = g = Q _ { r }$ introduces the weight $Q _ { r } ^ { 2 }$ in the uniform expectation. This weight vanishes unless $x _ { 2 j - 1 } = - x _ { 2 j }$ for every $j \le r$ . Each pair contributes zero to $S _ { d } .$ , leaving the sum of $d - 2 r$ independent signs. The lemma says that the same law of the coordinate sum results from weighting by $f ^ { 2 } / \Vert f \Vert _ { 2 } ^ { 2 }$ for every nonzero $f \in \mathcal { H } _ { d , r }$

In particular, choose $p _ { r }$ with unit norm under $S _ { d - 2 r }$ and energy at most ρ below a cutof u. Apply the identity with $H ( s ) = p _ { r } ( s ) ^ { 2 }$ and with $H ( s ) = p _ { r } ( s ) ^ { 2 } \mathbf { 1 } _ { \{ s < u \} }$ . For every nonzero $h _ { f } = f p _ { r } ( S _ { d } )$ 2 this gives

$$
\| h _ { f } \| _ { 2 } ^ { 2 } = \| f \| _ { 2 } ^ { 2 } , \qquad \frac { \mathbb { E } [ h _ { f } ^ { 2 } { \mathbf { 1 } } _ { \{ S _ { d } < u \} } ] } { \| h _ { f } \| _ { 2 } ^ { 2 } } = \mathbb { E } [ p _ { r } ( S _ { d - 2 r } ) ^ { 2 } { \mathbf { 1 } } _ { \{ S _ { d - 2 r } < u \} } ] \le \rho .
$$

Thus multiplication by the radial factor preserves dimension and transfers the one-dimensional concentration bound to the entire space $W _ { r } = \{ f p _ { r } ( S _ { d } ) : f \in \mathcal { H } _ { d , r } \}$ , with no dimension factor in the leakage. Diferent harmonic degrees also remain orthogonal after radial weighting, including restriction to a Hamming ball. This lets us combine the spaces $W _ { r }$ and their corresponding concentration bounds once we choose a cutof valid for all of them. We call this a harmonic lift: a single one-dimensional witness $p _ { r }$ is turned into a whole space of witnesses by the map $f \mapsto f p _ { r } ( S _ { d } )$ on $\mathcal { H } _ { d , r }$

Combining harmonic degrees $0 \leq r \leq R$ supplies $( _ { R } ^ { d } )$ concentrated directions, but leaves only $k - r$ degrees for the radial factor in sector r. Its efective coordinate sum has $d - 2 r$ terms, so the one-dimensional construction has center $2 { \sqrt { ( k - r ) ( d - k - r ) } }$ . This center decreases with $^ { r } \cdot$ The common cutof must therefore lie below the smallest center, attained at R. We define that center and its binomial tail exponent as

$$
\tau _ { R } = 2 \sqrt { ( k - R ) ( d - k - R ) } , \qquad \mathscr { E } _ { d , k } ( R ) = d I ( \tau _ { R } / d ) , \qquad 0 \leq R \leq k .
$$

Thus $\mathcal { E } _ { d , k } ( R )$ determines the leading exponential decay of the ball’s probability after reserving degree R for the harmonic factors, before adjusting the cutof to ensure energy concentration. At $R = 0$ it equals $E _ { d , k }$ , and increasing R trades a larger subspace for a smaller concentration exponent. Explicitly parametrizing this tradeof will allow for the subsequent optimized choice $R = \lfloor k ^ { 1 / 3 } \rfloor$

Proposition 4.2 (Harmonic lift). Fix $\rho \in \left( 0 , 1 \right)$ and $q _ { 0 } ~ \in ~ ( 0 , 1 / 2 )$ . For integers $d , k , R$ with   
$1 \leq k \leq q _ { 0 } d$ and $0 \leq R \leq k$ , there are $W _ { \leq R } \subseteq { \mathcal { P } } _ { d , k }$ of dimension $( _ { R } ^ { d } )$ and a permutation-invariant   
E such that, with $K = k - R$ ，

$$
\mu _ { d } ( E ) \leq e ^ { - \mathcal { E } _ { d , k } ( R ) + C _ { \rho , q _ { 0 } } K ^ { 1 / 3 } } , \qquad \mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] \leq \rho \mathbb { E } [ h ^ { 2 } ] \quad ( h \in W _ { \leq R } ) .
$$

Consequently,

$$
\mu _ { d } ( E ) \le e ^ { - E _ { d , k } + C _ { \rho , q _ { 0 } } ( R + k ^ { 1 / 3 } ) } .\tag{15}
$$

Proof. Write $M = M _ { \rho , q _ { 0 } }$ from Lemma 2.3. Fix an integer $K _ { 0 } = K _ { 0 } ( \rho , q _ { 0 } ) \ge 1$ above that lemma’s degree threshold and large enough that $M K _ { 0 } ^ { - 2 / 3 } < 2 \sqrt { 1 - 2 q _ { 0 } }$

Case $( i ) \colon K \geq K _ { 0 }$

Step 1: choosing a common ball. For $0 \leq r \leq R$ , put $D _ { r } = d - 2 r$ and $K _ { r } = k - r$ . Since $K _ { r } \geq K _ { 0 }$ and $K _ { r } / D _ { r } \leq k / d \leq q _ { 0 }$ , Lemma 2.3 provides a polynomial $p _ { r }$ of degree $K _ { r } ,$ with unit norm under $S _ { D _ { \eta } }$ and leakage at most $\rho$ below

$$
u _ { r } = 2 \sqrt { ( k - r ) ( d - k - r ) } - M \sqrt { D _ { r } } K _ { r } ^ { - 1 / 6 } .
$$

Set $u = \tau _ { R } - M \sqrt { d } K ^ { - 1 / 6 }$ and $E = \{ S _ { d } \geq u \}$ . The centers decrease with $^ { r , }$ while $\sqrt { D _ { r } } K _ { r } ^ { - 1 / 6 } \leq$ $\sqrt { d } K ^ { - 1 / 6 }$ , so $u \leq u _ { r }$ for all $r .$ . Moreover, $\tau _ { R } \geq 2 \sqrt { ( 1 - 2 q _ { 0 } ) d K }$ ; our choice of $K _ { 0 }$ therefore ensures $0 < u < \tau _ { R } < d .$

Step 2: combining directions without increasing leakage. Define $W _ { r } = \{ f p _ { r } ( S _ { d } ) : f \in \mathcal { H } _ { d , r } \}$ , whose elements have degree at most $r + K _ { r } = k$ . By Lemma 4.1, multiplication by $p _ { r } ( S _ { d } )$ preserves norms, and

$$
\mathbb { E } [ h _ { r } ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] \leq \rho \| h _ { r } \| _ { 2 } ^ { 2 } \qquad ( h _ { r } \in W _ { r } ) .
$$

To extend this bound to $h = \textstyle \sum _ { r } h _ { r }$ , we must also control the cross terms in both $\| h \| _ { 2 } ^ { 2 }$ and $\mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ]$ For $r \neq s$ , apply Lemma 4.1 with the weights $H ( t ) = p _ { r } ( t ) p _ { s } ( t )$ and $H ( t ) = p _ { r } ( t ) p _ { s } ( t ) \mathbf { 1 } _ { \{ t < u \} }$ respectively. This yields

$$
\mathbb { E } [ h _ { r } h _ { s } ] = \mathbb { E } [ h _ { r } h _ { s } \mathbf { 1 } _ { E ^ { c } } ] = 0 .
$$

Hence $W _ { \leq R } = \oplus _ { r = 0 } ^ { R } W _ { \tau }$ has dimension $\begin{array} { r } { \sum _ { r = 0 } ^ { R } [ { \binom { d } { r } } - { \binom { d } { r - 1 } } ] = { \binom { d } { R } } } \end{array}$ , and

$$
\mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] = \sum _ { r } \mathbb { E } [ h _ { r } ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] \leq \rho \sum _ { r } \| h _ { r } \| _ { 2 } ^ { 2 } = \rho \| h \| _ { 2 } ^ { 2 } .
$$

Step 3: bounding the probability of the ball. Use the integral estimate in (8), now with $D = d$ and center $\tau _ { R }$ . Since $\tau _ { R } / d \le 2 \sqrt { q _ { 0 } ( 1 - q _ { 0 } ) } < 1$ and $I ^ { \prime } ( \tau _ { R } / d ) \leq C _ { q _ { 0 } } \sqrt { K / d }$ , it gives

$$
\mathcal { E } _ { d , k } ( R ) - d I ( u / d ) \leq ( \tau _ { R } - u ) I ^ { \prime } ( \tau _ { R } / d ) \leq C _ { \rho , q _ { 0 } } \sqrt { d } K ^ { - 1 / 6 } \sqrt { K / d } = C _ { \rho , q _ { 0 } } K ^ { 1 / 3 } .
$$

The binomial Chernof bound $\mu _ { d } ( E ) \leq e ^ { - d I ( u / d ) }$ now proves the asserted probability bound.

Case $( i i ) \colon 0 \leq K < K _ { 0 }$ . Take E to be the whole cube and $W _ { \le R }$ to be the span of all degree-R Walsh characters. This space has dimension $( _ { R } ^ { d } )$ , lies in $\mathcal { P } _ { d , k }$ , and has zero leakage. To check the probability bound, note that $\tau _ { R } \leq 2 \sqrt { K ( d - K ) }$ and $\Psi ( q ) \leq 2 q$ give $0 \le \xi _ { d , k } ( R ) \le d \Psi ( K / d ) \le 2 K$ For $1 \le K < K _ { 0 }$ , choosing $C _ { \rho , q _ { 0 } } \geq 2 K _ { 0 } ^ { 2 / 3 }$ makes $- \mathcal { E } _ { d , k } ( R ) + C _ { \rho , q _ { 0 } } K ^ { 1 / 3 } \ge 0$ , so the claimed upper bound is at least $\mu _ { d } ( E ) = 1$ . For $K = 0$ , both exponents are zero.

Finally, for $z _ { R } = \tau _ { R } / d$ and by continuity at $R = k$ ，

$$
\left| \frac { d } { d R } \mathcal { E } _ { d , k } ( R ) \right| = 2 ( 1 - 2 R / d ) \frac { \operatorname { a r c t a n h } z _ { R } } { z _ { R } } \leq C _ { q _ { 0 } } .
$$

Integrating gives $\mathcal { E } _ { d , k } ( R ) \geq E _ { d , k } - C _ { q _ { 0 } } R .$ , which proves (15).

## 4.3 Proof of Theorem 1.8

Proof of Theorem 1.8. Step 1: the universal lower bound on $\mu _ { d } ( E )$ . Let $h \neq 0$ satisfy $\mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] \leq$ $\rho \| h \| _ { 2 } ^ { 2 }$ and rescale it to $\| h \| _ { 2 } = 1$ . Set $\eta ~ = ~ ( 1 + \rho ) / 2 ~ > ~ \rho$ . By Proposition 3.3, clipping at $B = \exp \{ E _ { d , k } + C _ { \eta , q _ { 0 } } k ^ { 1 / 3 } \}$ retains at least η energy. On $E ^ { c }$ the clipped energy is at most $\rho ,$ while on E the integrand is at most B. Hence

$$
\eta \le \mathbb { E } [ h ^ { 2 } \wedge B ] \le \mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] + B \mu _ { d } ( E ) \le \rho + B \mu _ { d } ( E ) .
$$

It follows that $\mu _ { d } ( E ) \geq ( 1 - \rho ) / ( 2 B )$ , giving the asserted bound after absorbing the fixed prefactor into $C _ { \rho , q _ { 0 } }$

Step 2: a subspace attaining the bound up to the remainder. Apply Proposition 4.2 with $R = \lfloor k ^ { 1 / 3 } \rfloor$ It supplies a space of dimension $( \mathbf { \Sigma } _ { R } ^ { d } ) = M _ { d , k }$ and one event E with leakage at most $\rho$ for every function in that space. Moreover, (15) gives

$$
\mu _ { d } ( E ) \le \exp \{ - E _ { d , k } + C _ { \rho , q _ { 0 } } ( R + k ^ { 1 / 3 } ) \} \le \exp \{ - E _ { d , k } + C _ { \rho , q _ { 0 } } ^ { \prime } k ^ { 1 / 3 } \} .
$$

Remark 4.3 (Rare event is a Hamming ball). For suficiently large k, the choice $R = \lfloor k ^ { 1 / 3 } \rfloor$ gives $K = k - R \geq K _ { 0 } ( \rho , q _ { 0 } )$ . We therefore land in case (i) of the proof of Proposition $4 . 2 ,$ which constructs $E = \{ S _ { d } \geq u \}$ with $u > 0$ , rather than the whole-cube fallback of case $( i i )$ . Since $S _ { d } ( x ) = d - 2$ dist $_ H ( x , ( 1 , \dots , 1 ) )$ , this event is a Hamming ball, establishing the final assertion of Theorem 1.8.

## 5 From concentration to minimax regression

We now convert the geometric bounds into regression guarantees. These reductions apply to any isotropic feature distribution.

## 5.1 From shared concentration to small empirical eigenvalues

To lower-bound the sample size required for the parametric regression rate, we use a subspace whose functions concentrate on a common set $E .$ . In an orthonormal basis, this gives

$$
\mathbb { E } [ \Phi ( X ) \Phi ( X ) ^ { \top } \mathbf { 1 } _ { \{ X \notin { \cal E } \} } ] \preceq \rho I _ { m } , \qquad \mathbb { E } [ \Phi ( X ) \Phi ( X ) ^ { \top } ] = I _ { m } .\tag{16}
$$

When $n \mu _ { d } ( E )$ is small relative to $m ,$ few observations fall inside E, while those outside contribute little total energy. Together, these properties force many eigenvalues of the empirical covariance matrix to be small.

Lemma 5.1 (Many small empirical eigenvalues). Let $\mathbb { P } ( E ) = p \in ( 0 , 1 / 2 ]$ . (i) If (16) holds, $m \geq 2 ,$ and $n \leq m / ( 1 2 8 p )$ , then with probability at least $1 5 / 1 6$ , at least $\lfloor m / 2 \rfloor$ eigenvalues $o f \widehat { \Sigma }$ are at most $1 2 8 \rho$ . (ii) $I f t \geq \log 4 , n \leq t / ( 5 1 2 p )$ , and a fixed unit vector v satisfies $\mathbb { E } [ ( v ^ { \top } \Phi ) ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] \le \rho _ { }$ , then with probability at least $( 1 5 / 1 6 ) e ^ { - t / 2 5 6 }$ the smallest eigenvalue $o f \widehat { \Sigma }$ is at most $3 2 \rho$

Proof. Split the empirical covariance according to where each observation falls:

$$
\hat { \Sigma } = \hat { \Sigma } _ { E } + \hat { \Sigma } _ { E ^ { c } } , \qquad \hat { \Sigma } _ { A } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Phi ( X _ { i } ) \Phi ( X _ { i } ) ^ { \top } { \bf 1 } _ { \{ X _ { i } \in A \} } , \quad A \in \{ E , E ^ { c } \} .
$$

Both matrices are positive semidefinite.

Step 1: many small eigenvalues (part (i)). Let $\begin{array} { r } { N _ { E } = \sum _ { i } \mathbf { 1 } _ { \{ X _ { i } \in E \} } } \end{array}$ . Then rank $( \widehat { \Sigma } _ { E } ) \ \leq \ N _ { E }$ $\mathbb { E } [ N _ { E } ] = n p \leq m / 1 2 8$ , and $\mathbb { E } [ \mathrm { t r } \widehat { \Sigma } _ { E ^ { c } } ] \leq m \rho$ by (16). Markov’s inequality and a union bound show that, with probability at least $1 5 / 1 6$ , both $N _ { E } \leq m / 4$ and tr ${ \widehat \Sigma } _ { E ^ { c } } \leq 3 2 m \rho$ hold. We condition on this event and set $L = \ker \widehat { \Sigma } _ { E }$ , so that dim $L \ge m - \lfloor m / 4 \rfloor$ . Let $P _ { L }$ be orthogonal projection onto L. The operator

$$
B _ { L } = ( P _ { L } \widehat { \Sigma } _ { E ^ { c } } P _ { L } ) | _ { L } : L  L
$$

describes the compression to $L \colon$ it has the same quadratic form as $\widehat { \Sigma } _ { E ^ { c } }$ on vectors in $L ,$ and tr $B _ { L } \leq \mathrm { t r } \widehat { \Sigma } _ { E ^ { c } } \leq 3 2 m \rho$ . Since its eigenvalues are nonnegative, at most $\lfloor m / 4 \rfloor$ can exceed $1 2 8 \rho$ Let $U \subseteq L$ be the span of the eigenvectors of $B _ { L }$ with eigenvalues at most 128ρ. Then dim $U \geq$ dim $L - \lfloor m / 4 \rfloor \ge \lfloor m / 2 \rfloor$

For $0 \neq w \in U$

$$
\mathcal { R } _ { \widehat { \Sigma } } ( w ) = \frac { w ^ { \top } \widehat { \Sigma } _ { E ^ { c } } w } { \Vert w \Vert _ { 2 } ^ { 2 } } = \frac { w ^ { \top } B _ { L } w } { \Vert w \Vert _ { 2 } ^ { 2 } } \leq 1 2 8 \rho ,
$$

because $\widehat { \Sigma } _ { E } w = 0$ . The min–max principle now gives the claimed number of small eigenvalues of $\widehat { \Sigma }$ Step 2: one small eigenvalue (part $( i i ) )$ . Fix a unit vector v satisfying the assumption in (ii) and condition on $N _ { E } = 0$ . Independence gives

$$
\operatorname* { P r } ( N _ { E } = 0 ) = ( 1 - p ) ^ { n } \ge e ^ { - 2 n p } \ge e ^ { - t / 2 5 6 } , \qquad \mathbb { E } [ v ^ { \top } \widehat \Sigma v \mid N _ { E } = 0 ] = \frac { \mathbb { E } [ ( v ^ { \top } \Phi ) ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] } { 1 - p } \le 2 \rho .
$$

Markov’s inequality gives $v ^ { \top } \widehat { \Sigma } v \leq 3 2 \rho$ with conditional probability at least $1 5 / 1 6$ . Since $\lambda _ { \operatorname* { m i n } } ( \widehat { \Sigma } ) \leq$ $\mathcal { R } _ { \widehat { \Sigma } } ( v ) = v ^ { \top } \widehat { \Sigma } v$ , the unconditional probability is at least $( 1 5 / 1 6 ) e ^ { - t / 2 5 6 }$ □ Σb

## 5.2 From empirical eigenvalues to minimax risk

For the design matrix X with rows $\Phi _ { V } ( X _ { i } ) ^ { \top }$ , a full-rank sample gives

$$
( \widehat { \theta } _ { \mathrm { O L S } } - \theta ) \ : | \ : \mathbf { X } \sim N \left( 0 , \frac { \sigma ^ { 2 } } { n } \widehat { \Sigma } _ { V } ^ { - 1 } \right) .
$$

Thus an empirical eigenvalue λ produces variance $\sigma ^ { 2 } / ( n \lambda )$ in a unit population direction. The following identity shows that the corresponding quantile is minimax over all estimators and is a direct corollary of the results in [EHE24].

Theorem 5.2 (El Hanchi–Maddison–Erdogdu). For a fixed known subspace V in (1), let $G =$ $( G _ { 1 } , \dots , G _ { m } ) \sim N ( 0 , I _ { m } )$ be independent of the sampled inputs, and define

$$
\begin{array} { r } { Z _ { n , V } = \left\{ \begin{array} { l l } { ( \sigma ^ { 2 } / n ) G ^ { \top } \widehat \Sigma _ { V } ^ { - 1 } G , } & { \widehat \Sigma _ { V } \ i n v e r t i b l e , } \\ { + \infty , } & { \widehat \Sigma _ { V } \ s i n g u l a r . } \end{array} \right. } \end{array}
$$

Then $\mathcal { R } _ { n , \delta } ^ { * } ( V ) = Q _ { 1 - \delta } ( Z _ { n , V } )$ , and any measurable least-squares minimizer attains this minimax value. Moreover,

$$
\mathcal { R } _ { n , \delta } ^ { * } ( V ) = + \infty \quad \Longleftrightarrow \quad \operatorname* { P r } ( \widehat { \Sigma } _ { V } \ s i n g u l a r ) \ge \delta .
$$

Conditionally on any nonsingular design with at least r eigenvalues at most $\lambda > 0$ , the theorem’s comparison variable satisfies

$$
Z _ { n , V } \succeq _ { \mathrm { s t } } \frac { \sigma ^ { 2 } } { n \lambda } \chi _ { r } ^ { 2 } .
$$

The comparison also holds on singular designs. If $t < m ,$ the many eigenvalues in Lemma 5.1(i) provide the factor m. If $t \geq m$ , part (ii) and a Gaussian tail provide the factor t. Choosing $\rho$ suficiently small depending only on $A$ , the benchmark is unattainable whenever

$$
n \leq { \frac { m + t } { 1 0 2 4 p } } .\tag{17}
$$

Appendix D applies the Gaussian tail estimates from Lemma B.5 to give the complete probability calculation, including the cases of bounded $m , t .$ . It also proves the stable-sampling lower bound, which uses only Lemma 5.1, with $\rho$ chosen depending on c.

## 5.3 Proof of Theorem 1.5

Proof of Theorem 1.5. Write $N = N _ { \mathrm { p a r } } ( d , k , m , A , \delta )$ . For the upper bound, use Proposition 3.3 with a fixed retained fraction greater than $1 / 2$ , then Proposition 3.4 at covariance level $1 / 2$ and failure probability $\delta / 2$ . Conditional on this event, the error of ordinary least squares is bounded by $( 2 \sigma ^ { 2 } / n ) \chi _ { m } ^ { 2 }$ . Corollary B.6 gives the benchmark with constant $1 2 \leq A$ . This proves $N _ { \mathrm { p a r } } \leq C ( m + t ) e ^ { E _ { d , k } + C k ^ { 1 / 3 } }$ for every feasible m.

For the lower bound choose the leakage required in Proposition D.3. For large k, Theorem 1.8 gives a common set with $0 < p \leq e ^ { - E _ { d , k } + C k ^ { \overline { { 1 } } / 3 } } \leq 1 / 2$ . If $m \leq M _ { d , k }$ , take an m-dimensional subspace of the concentrating space. If $t \geq m$ , extend one concentrated unit vector to any feasible m-dimensional space. The common-set or one-direction version of (17) gives $N \geq ( m + t ) / ( 1 0 2 4 p )$ : failure at every positive integer up to this quantity forces the eventual threshold above its integer part, otherwise, if the quantity is below one, use $N \geq 1$ . For bounded k, the baseline $N \geq c _ { 0 } ( m + t )$ in Lemma D.1, together with $E _ { d , k } \leq 2 k$ , sufices. Taking logarithms absorbs fixed factors into $C k ^ { 1 / 3 }$

For stable sampling the upper proof uses retained fraction $( 1 + c ) / 2$ and Proposition 3.4. The lower bound is proved by Proposition D.2. □

## 6 Sharpness of the uniform uncertainty remainder

We prove Proposition 1.9 by constructing a polynomial with a fixed fraction $\beta > 0$ of its squared L -norm concentrated on the chosen set, a sequence of dimensions $d _ { k } \geq k ^ { 3 }$ , and spaces of dimension $\binom { d _ { k } } { | k ^ { 1 / 3 } | }$ concentrated on events of probability at most $e ^ { - E _ { d _ { k } , k } - 2 k ^ { 1 / 3 } }$ . In other words, here the leakage is $\rho _ { * } \doteq 1 - \beta$ , and only a small fixed fraction of energy needs to be retained. This difers from the construction of Section 2, which allows any prescribed leakage.

The proof has three stages. We first construct a polynomial of degree at most N whose energy under the standard Gaussian measure has a fixed positive fraction concentrated on $\{ x > 2 { \sqrt { N } } \}$ For each suficiently large degree k, we transfer that polynomial to a suficiently large Boolean cube and apply the harmonic lift. Finally, a tail bound shows that the common event has the required probability.

## 6.1 Convergence and limits

Two diferent limits underlie the construction. We describe them below.

From the cube to Gaussian polynomials: fix the degree, let the ambient dimension grow. Write the

normalized radial Krawchouk polynomial as

$$
e _ { D , j } ( x ) = { \binom { D } { j } } ^ { - 1 / 2 } \sum _ { | A | = j } \prod _ { i \in A } x _ { i } = P _ { D , j } ( S _ { D } ( x ) / { \sqrt { D } } ) .
$$

For every fixed $j ,$ the coeficients of the univariate polynomial $P _ { D , j }$ converge as $D \to \infty$ to those of $H _ { j } ( t ) = \mathrm { H e } _ { j } ( t ) / \sqrt { j ! }$ , the degree-j orthonormal Hermite polynomial for the standard Gaussian distribution. In particular, $P _ { D , j } ( t ) \to H _ { j } ( t )$ uniformly on bounded intervals of t. This explains why Gaussian polynomials are useful models for radial polynomials on large cubes. Our transfer below only needs the central limit theorem and moment convergence for a fixed polynomial.

From Hermite polynomials to the Airy kernel as the degree cutof increases. We combine Hermite degrees $0 , \ldots , N - 1$ into the kernel $K _ { N }$ defined below. The external asymptotic theorem we use evaluates this kernel at points

$$
x _ { N } ( s ) = 2 \sqrt { N } + s N ^ { - 1 / 6 } ,
$$

where s remains in a fixed bounded interval as $N  \infty .$ . After multiplying the kernel by $N ^ { - 1 / 6 }$ , its limit is the Airy kernel in (18). Here it describes the limiting behavior of the Hermite kernel near the edge of its concentration region.

## 6.2 A Gaussian polynomial with a fixed amount of tail energy

Step 1: use the Hermite kernel to combine degrees. Let $\gamma ~ = ~ N ( 0 , 1 )$ have density $\varphi ( x ) = ( 2 \pi ) ^ { - 1 / 2 } e ^ { - x ^ { 2 } / 2 }$ , and define

$$
H _ { j } ( x ) = { \frac { \mathrm { H e } _ { j } ( x ) } { \sqrt { j ! } } } , \qquad \mathrm { H e } _ { j } ( x ) = ( - 1 ) ^ { j } e ^ { x ^ { 2 } / 2 } { \frac { d ^ { j } } { d x ^ { j } } } e ^ { - x ^ { 2 } / 2 } .
$$

The $H _ { j }$ are orthonormal in $L _ { 2 } ( \gamma )$ . Set $\psi _ { j } = H _ { j } { \sqrt { \varphi } }$ and define the Hermite kernel

$$
K _ { N } ( x , y ) = \sum _ { j = 0 } ^ { N - 1 } \psi _ { j } ( x ) \psi _ { j } ( y ) .
$$

The Airy function is defined for real s by the oscillatory integral

$$
\operatorname { A i } ( s ) = { \frac { 1 } { \pi } } \operatorname* { l i m } _ { R \to \infty } \int _ { 0 } ^ { R } \cos \left( { \frac { u ^ { 3 } } { 3 } } + s u \right) d u .
$$

The associated Airy kernel is

$$
K _ { \mathrm { A i } } ( s , t ) = \int _ { 0 } ^ { \infty } \mathrm { A i } ( s + u ) \mathrm { A i } ( t + u ) d u .
$$

To describe the Hermite kernel near $2 \sqrt { N }$ , put $x _ { N } ( s ) = 2 \sqrt { N } + s N ^ { - 1 / 6 }$ . Then

$$
N ^ { - 1 / 6 } K _ { N } ( x _ { N } ( s ) , x _ { N } ( t ) ) \longrightarrow K _ { \mathrm { A i } } ( s , t )\tag{18}
$$

as $N  \infty$ , uniformly for $( s , t )$ in compact subsets of $\mathbb { R } ^ { 2 }$ . This convergence follows from [SX22, Theorem 2.3], which in turn restates the result of [DG07].

We center a kernel vector at $y _ { N } = x _ { N } ( 3 )$ , slightly beyond $2 \sqrt { N }$ . This is the standard reproducingkernel construction: the coeficients are the basis values at the point where we want concentration. For $K = N - 1$ , define

$$
q _ { K } ( x ) = \frac { \sum _ { j = 0 } ^ { K } H _ { j } ( x ) H _ { j } ( y _ { N } ) } { ( \sum _ { j = 0 } ^ { K } H _ { j } ( y _ { N } ) ^ { 2 } ) ^ { 1 / 2 } } .
$$

Orthonormality gives deg $q _ { K } \leq K$ and $\textstyle \int q _ { K } ^ { 2 } d \gamma = 1$ . Moreover,

$$
q _ { K } ( x ) ^ { 2 } \varphi ( x ) = \frac { K _ { N } ( x , y _ { N } ) ^ { 2 } } { K _ { N } ( y _ { N } , y _ { N } ) } .
$$

Step 2: retain positive energy on a fixed scaled interval. We estimate the energy on the interval $[ 2 \sqrt { N } + 3 N ^ { - 1 / 6 } , 2 \sqrt { N } + 4 N ^ { - 1 / 6 } ]$ . Under the change of variables $x = 2 \sqrt { N } + s N ^ { - 1 / 6 }$ , this becomes $s \in [ 3 , 4 ]$ , where the uniform convergence in (18) lets us pass to the limit inside the integral. The left endpoint is chosen far enough above $2 \sqrt { N }$ to accommodate the degree reserved for the harmonic lift below. Any fixed right endpoint larger than 3 would work. A positive lower bound on the energy in this interval sufices, since the interval lies within the tail event we will use.

To track the scaling, write $\widetilde { K } _ { N } ( s , t ) = N ^ { - 1 / 6 } K _ { N } ( x _ { N } ( s ) , x _ { N } ( t ) )$ . Since $d x = N ^ { - 1 / 6 } d s$ , the kernel factors cancel and yield

$$
\int _ { x _ { N } ( 3 ) } ^ { x _ { N } ( 4 ) } q _ { N - 1 } ( x ) ^ { 2 } d \gamma ( x ) = \frac { \int _ { 3 } ^ { 4 } \widetilde { K } _ { N } ( s , 3 ) ^ { 2 } d s } { \widetilde { K } _ { N } ( 3 , 3 ) } \xrightarrow [ N  \infty ] { } \alpha : = \frac { \int _ { 3 } ^ { 4 } K _ { \mathrm { A i } } ( s , 3 ) ^ { 2 } d s } { K _ { \mathrm { A i } } ( 3 , 3 ) } > 0 .\tag{19}
$$

Indeed, $\mathrm { A i } ( s ) > 0$ for $s \geq 0$ , so the denominator and integrand in the limit are positive. Also $\alpha \leq 1$ since each $q N { - } 1$ has unit norm. Fix $\beta = \alpha / 8 > 0$ , independently of all subsequent degrees and dimensions. By convergence to $\alpha ,$ there is $N _ { 0 }$ such that for every $N \geq N _ { 0 }$ the left side of (19) is at least $\alpha / 2 = 4 \beta$ . The factor four leaves room for approximation errors in passing to the cube and comparing norms.

## 6.3 Transferring the construction to a Boolean subspace

Proof of Proposition 1.9. Step 1: reserving enough degree for the harmonic directions. For each suficiently large integer $k ,$ set

$$
R = \lfloor k ^ { 1 / 3 } \rfloor , \qquad K = k - R , \qquad N = K + 1 , \qquad a _ { k } = 2 \sqrt { k } + k ^ { - 1 / 6 } .
$$

As $k \to \infty , N = k - \lfloor k ^ { 1 / 3 } \rfloor + 1 \to \infty$ , so eventually $N \geq N _ { 0 }$ . We need the interval used above to lie beyond $a _ { k }$ , which is measured relative to the larger degree k. For such k, $N \leq k$ and

$$
2 ( \sqrt { k } - \sqrt { N } ) = \frac { 2 ( R - 1 ) } { \sqrt { k } + \sqrt { N } } \leq 2 k ^ { - 1 / 6 } , \qquad 3 N ^ { - 1 / 6 } \geq 3 k ^ { - 1 / 6 } .
$$

Hence $x _ { N } ( 3 ) \geq a _ { k }$ , and (19) implies that $\textstyle \int q _ { K } ^ { 2 } \mathbf { 1 } _ { [ a _ { k } , \infty ) } d \gamma \geq 4 \beta .$

Step 2: transfer to the cube with k fixed. Fix one such k, and hence fix $R , K , N , a _ { k }$ and the polynomial $q _ { K }$ . We now let the ambient dimension d tend to infinity. For each $r \in \{ 0 , \ldots , R \}$ , the radial-weight identity (Lemma 4.1) expresses energy in harmonic degree $r$ as an expectation over a cube with $d - 2 r$ coordinates. Accordingly, define the total energy and the energy above $a _ { k }$ by

$$
c _ { r , d } = \mathbb { E } _ { \mu _ { d - 2 r } } [ q _ { K } ( S _ { d - 2 r } / \sqrt { d } ) ^ { 2 } ] , \qquad b _ { r , d } = \mathbb { E } _ { \mu _ { d - 2 r } } \Big [ q _ { K } ( S _ { d - 2 r } / \sqrt { d } ) ^ { 2 } \mathbf { 1 } _ { \{ S _ { d - 2 r } / \sqrt { d } \geq a _ { k } \} } \Big ] \ .
$$

For each fixed $^ { r , }$ the central limit theorem gives $S _ { d - 2 r } / \sqrt { d } \to G$ in distribution, where $G \sim N ( 0 , 1 )$ To obtain convergence of the expectations above, we must also control the growth of $q _ { K } ^ { 2 }$ . Since $q _ { K }$ has fixed degree K and

$$
\operatorname* { s u p } _ { d > 2 r } \mathbb { E } [ | S _ { d - 2 r } / \sqrt { d } | ^ { 2 K + 2 } ] < \infty ,
$$

the contribution to either expectation from $| S _ { d - 2 r } / \sqrt { d } | > M$ tends to zero uniformly in d as $M \to \infty$ Together with $\mathbb { P } ( G = a _ { k } ) = 0$ , this gives

$$
c _ { r , d } \underset { d  \infty } { \longrightarrow } \mathbb { E } [ q _ { K } ( G ) ^ { 2 } ] = 1 , \qquad b _ { r , d } \underset { d  \infty } { \longrightarrow } \mathbb { E } [ q _ { K } ( G ) ^ { 2 } \mathbf { 1 } _ { \{ G \geq a _ { k } \} } ] \geq 4 \beta .
$$

These limits hold with k fixed; no uniform convergence as $k$ grows is needed. Since there are only $R + 1$ values of $r _ { \mathrm { { ; } } }$ , we may choose $d = d _ { k } \geq k ^ { 3 }$ suficiently large that

$$
{ \frac { 1 } { 2 } } \leq c _ { r , d } \leq 2 , \qquad b _ { r , d } \geq 2 \beta \qquad \mathrm { f o r ~ e v e r y ~ } r \in \{ 0 , \ldots , R \} .
$$

Step 3: lift the one-dimensional witness to many directions. On the chosen cube, let $E =$ $\{ S _ { d } / \sqrt { d } \geq a _ { k } \}$ and

$$
W _ { r } = q _ { K } ( S _ { d } / \sqrt { d } ) \mathcal { H } _ { d , r } , \qquad W = \bigoplus _ { r = 0 } ^ { R } W _ { r } .
$$

The degree of each product is at most $K + r \leq k$ . By Lemma $4 . 1$ , multiplication on $\mathcal { H } _ { d , r }$ changes the squared norm by the positive factor $c _ { r , d }$ , so it is injective. For a nonzero $h _ { r } = f q _ { K } ( S _ { d } / \sqrt { d } ) \in W _ { r } ,$ the same identity gives

$$
\frac { \mathbb { E } [ h _ { r } ^ { 2 } \mathbf { 1 } _ { E } ] } { \mathbb { E } [ h _ { r } ^ { 2 } ] } = \frac { b _ { r , d } \| f \| _ { 2 } ^ { 2 } } { c _ { r , d } \| f \| _ { 2 } ^ { 2 } } \geq \frac { 2 \beta } { 2 } = \beta .
$$

Diferent $W _ { r }$ are orthogonal both globally and on $E _ { i }$ , by the same radial-weight identity. Thus

$$
\dim W = \sum _ { r = 0 } ^ { R } \left( { \binom { d } { r } } - { \binom { d } { r - 1 } } \right) = { \binom { d } { R } } , \qquad \mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E } ] \geq \beta \mathbb { E } [ h ^ { 2 } ] \quad ( h \in W ) .\tag{20}
$$

This proves the common-event assertion for every linear combination.

Step 4: bound the common event’s probability. The standard Chernof argument uses $\mathbb { E } [ e ^ { \lambda S _ { d } / \sqrt { d } } ] =$ cosh $( \lambda / \sqrt { d } ) ^ { d } \leq e ^ { \lambda ^ { 2 } / 2 }$ . Applying Markov’s inequality to $e ^ { \lambda S _ { d } / \sqrt { d } }$ with $\lambda = a _ { k }$ gives

$$
\begin{array} { r } { \mu _ { d } ( E ) \le e ^ { - a _ { k } ^ { 2 } / 2 } = \exp \Bigl ( - 2 k - 2 k ^ { 1 / 3 } - \frac 1 2 k ^ { - 1 / 3 } \Bigr ) \le e ^ { - E _ { d , k } - 2 k ^ { 1 / 3 } } , } \end{array}
$$

using $\Psi ( q ) \leq 2 q$ from Lemma A.1. The cross term in $a _ { k } ^ { 2 } / 2 = ( 2 \sqrt { k } + k ^ { - 1 / 6 } ) ^ { 2 } / 2$ is $2 k ^ { 1 / 3 }$ : this is why moving the cutof above $2 \sqrt { k }$ yields the required correction. Take $\rho _ { * } = 1 - \beta$ . Restricting $W$ to any m-dimensional subspace in (20) proves the lower inequality in (5). The upper inequality is (4), applied, for example, with $q _ { 0 } = 1 / 4$ . Since $d _ { k } \geq k ^ { 3 }$ , this range holds eventually. □

For each suficiently large $k ,$ the transfer works once d is large enough; choose one such dimension $d _ { k } \geq k ^ { 3 }$ . The argument does not give an explicit upper bound on $d _ { k }$

This proves that a positive correction of order $k ^ { 1 / 3 }$ is necessary at one fixed leakage level, for all subspace dimensions in the stated range. The smooth-window construction remains essential: it works for every $k \leq q _ { 0 } d$ and allows any prescribed leakage $\rho > 0$ . By contrast, the Airy construction guarantees only that a fixed fraction $\beta > 0$ of the energy lies in E. The remaining fraction $1 - \beta$ need not be small enough to establish the regression lower bound for a prescribed benchmark constant $A \geq 1 6$

## 7 Discussion

Our subspace uncertainty principle determines the worst-subspace parametric threshold, throughout the matching range, to an $O ( k ^ { 1 / 3 } )$ error in the exponent. It explains the exponential cost of noise relative to identification. The uncertainty remainder is optimal in general.

Section F derives the finite-dimensional correction, the worst-case threshold over ambient di mensions, and lower bounds for larger model dimensions, including the leading exponent when $\log m = o ( d )$ and $k / d \to q \in ( 0 , 1 / 2 )$

Limitations and open questions. At fixed confidence, determining the threshold beyond $m \le M _ { d , k }$ remains open, particularly when log m is comparable to d. How does increasing the model dimension change the exponent $E _ { d , k } ?$ A necessary constraint is $\mu _ { d } ( E ) \geq ( 1 - \rho ) m / \dim \mathcal { P } _ { d , k }$ whenever an m-dimensional subspace concentrates on E with leakage ρ.

Our sharpness result concerns one fixed leakage level and suficiently large ambient dimensions. Optimality of the remainder for every prescribed leakage or regression benchmark, and uniform estimates as $k / d \uparrow 1 / 2$ , remain open. Another question is to identify geometric properties of a given subspace V that predict its sample threshold before sampling. Such criteria should distinguish models requiring exponentially many samples in k from models, such as spans of Walsh characters, that attain the parametric rate after $O ( m ( \log m + t ) )$ samples, see Section F.2.

## AI methodologies

During the preparation of this paper, the author used OpenAI’s ChatGPT 5.6 Sol and 6.0 Astra for the tasks described below. Through an extended exchange with ChatGPT 6.0 Astra, the author explored a matching converse establishing uniform sharpness of the uncertainty remainder. ChatGPT eventually suggested the Airy approach presented in Section 6, whose proof is substantially based on the argument it supplied. ChatGPT 5.6 Sol assisted with proofreading and minor copy-editing. The author independently re-derived and verified the proofs in Section 6 and takes full responsibility for the paper’s content.

## References

[ASW15] Emmanuel Abbe, Amir Shpilka, and Avi Wigderson. Reed–Muller codes for random erasures and errors. Proceedings of STOC, pp. 297–306, 2015. Journal version: IEEE Transactions on Information Theory 61(10):5229–5252, 2015. arXiv:1411.4590.

[Bec75] William Beckner. Inequalities in Fourier analysis. Annals of Mathematics 102(1):159– 182, 1975. doi:10.2307/1970980.

[BHSS22] Siddharth Bhandari, Prahladh Harsha, Ramprasad Saptharishi, and Srikanth Srinivasan. Vanishing spaces of random sets and applications to Reed–Muller codes. Computational Complexity Conference, LIPIcs 234, Article 31, 2022. doi:10.4230/LIPIcs.CCC.2022.31.

[BLM13] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 2013. doi:10.1093/acprof:oso/9780199535255.001.0001.

[Bon70] Aline Bonami. Étude des coeficients de Fourier des fonctions de L<sup>p</sup>(G). Annales de l’Institut Fourier 20(2):335–402, 1970. doi:10.5802/aif.357.

[Bou02] Olivier Bousquet. A Bennett concentration inequality and its application to suprema of empirical processes. Comptes Rendus Mathématique 334(6):495–500, 2002. doi:10.1016/S1631-073X(02)02292-6.

[CDL13] Albert Cohen, Mark A. Davenport, and Dany Leviatan. On the stability and accuracy of least squares approximations. Foundations of Computational Mathematics 13:819–834, 2013. arXiv:1111.4422.

[CM17] Albert Cohen and Giovanni Migliorati. Optimal weighted least-squares methods. SMAI Journal of Computational Mathematics 3:181–203, 2017. doi:10.5802/smai-jcm.24.

[DG07] Percy Deift and Dimitri Gioev. Universality at the edge of the spectrum for unitary, orthogonal, and symplectic ensembles of random matrices. Communications on Pure and Applied Mathematics 60(6):867–910, 2007. doi:10.1002/cpa.20164. arXiv:mathph/0507023.

[DW07] Dan Dai and Roderick Wong. Global asymptotics of Krawtchouk polynomials: a Riemann–Hilbert approach. Chinese Annals of Mathematics, Series B 28(1):1–34, 2007. doi:10.1007/s11401-006-0195-3.

[EHE24] Ayoub El Hanchi, Chris J. Maddison, and Murat A. Erdogdu. Minimax linear regression under the quantile risk. Proceedings of COLT, PMLR 247:1516–1572, 2024. PMLR proceedings.

[EI22] Alexandros Eskenazis and Paata Ivanisvili. Learning low-degree functions from a logarithmic number of random queries. Proceedings of STOC, pp. 203–207, 2022. arXiv:2109.10162.

[EIS23] Alexandros Eskenazis, Paata Ivanisvili, and Lauritz Streck. Low-degree learning and the metric entropy of polynomials. Discrete Analysis, Article 17, 2023. arXiv:2203.09659.

[Fil16] Yuval Filmus. An orthogonal basis for functions over a slice of the Boolean hypercube. Electronic Journal of Combinatorics 23(1), Paper P1.23, 2016. doi:10.37236/4567. arXiv:1406.0142v2.

[FM19] Yuval Filmus and Elchanan Mossel. Harmonicity and invariance on slices of the Boolean cube. Probability Theory and Related Fields 175:721–782, 2019. doi:10.1007/s00440-019- 00900-w. arXiv:1507.02713v5.

[KKLT22] Boris Kashin, Egor Kosov, Irina Limonova, and Vladimir Temlyakov. Sampling discretization and related problems. Journal of Complexity 71:101653, 2022. arXiv:2109.07567.

[KS21] Naomi Kirshner and Alex Samorodnitsky. A moment ratio bound for polynomials and some extremal properties of Krawchouk polynomials and Hamming spheres. IEEE Transactions on Information Theory 67(6):3509–3541, 2021. arXiv:1909.11929v1.

[Lev95] Vladimir I. Levenshtein. Krawtchouk polynomials and universal bounds for codes and designs in Hamming spaces. IEEE Transactions on Information Theory 41(5):1303–1321, 1995. doi:10.1109/18.412678.

[Men14] Shahar Mendelson. Learning without concentration. Proceedings of COLT, PMLR 35:25– 39, 2014. PMLR proceedings.

[Men21] Shahar Mendelson. Extending the scope of the small-ball method. Studia Mathematica 256(2):147–167, 2021. doi:10.4064/sm190420-21-11. arXiv:1709.00843.

[Mou22] Jaouad Mourtada. Exact minimax risk for linear least squares, and the lower tail of sample covariance matrices. Annals of Statistics 50(4):2157–2178, 2022. arXiv:1912.10754.

[Oli16] Roberto Imbuzeiro Oliveira. The lower tail of random quadratic forms with applications to ordinary least squares. Probability Theory and Related Fields 166:1175–1194, 2016. arXiv:1312.2903.

[PS19] Yury Polyanskiy and Alex Samorodnitsky. Improved log-Sobolev inequalities, hypercontractivity and uncertainty principle on the hypercube. Journal of Functional Analysis 277(11):108280, 2019. arXiv:1606.07491v3.

[Sam08] Alex Samorodnitsky. A modified logarithmic Sobolev inequality for the Hamming cube and some applications. arXiv preprint, 2008. arXiv:0807.1679.

[SL23] Lucas Slot and Monique Laurent. Sum-of-squares hierarchies for binary polynomial optimization. Mathematical Programming 197:621–660, 2023 (published online in 2022). doi:10.1007/s10107-021-01745-9. arXiv:2011.04027v3.

[SX22] Kevin Schnelli and Yuanyuan Xu. Convergence rate to the Tracy–Widom laws for the largest eigenvalue of Wigner matrices. Communications in Mathematical Physics 393:839–907, 2022. doi:10.1007/s00220-022-04377-y.

[Tropp12] Joel A. Tropp. User-friendly tail bounds for sums of random matrices. Foundations of Computational Mathematics 12(4):389–434, 2012. arXiv:1004.4389v7.

[Yas14] Pavel Yaskov. Lower bounds on the smallest eigenvalue of a sample covariance matrix. Electronic Communications in Probability 19, Article 83, pp. 1–10, 2014. doi:10.1214/ECP.v19-3807. arXiv:1409.6188.

## A The entropy curve and the cubic remainder

We collect the elementary properties of the entropy curve and the proof of Lemma 3.2, whose statement and use in clipping appear in Section 3.

## A.1 The entropy curve

Recall that all logarithms are natural and

$$
\mathsf { H } ( x ) = - x \log x - ( 1 - x ) \log ( 1 - x ) , \qquad \Psi ( q ) = \log 2 - \mathsf { H } \Bigl ( \frac { 1 } { 2 } - \sqrt { q ( 1 - q ) } \Bigr ) , \qquad E _ { d , k } = d \Psi ( k / d ) ,
$$

with $0 \log 0 = 0$ . The following properties follow by elementary diferentiation; we include the details used later.

Lemma A.1 (Elementary calculus). For $q \in [ 0 , 1 / 2 ]$ , the function Ψ is increasing and concave, with

$$
\Psi ( 0 ) = 0 , \qquad \Psi ( 1 / 2 ) = \log 2 , \qquad \Psi ^ { \prime } ( 0 ) = 2 .
$$

For any $q _ { 0 } < 1 / 2$ , uniformly on $0 \leq q \leq q _ { 0 }$ 2

$$
\Psi ( q ) = 2 q - \frac { 2 } { 3 } q ^ { 2 } - \frac { 8 } { 1 5 } q ^ { 3 } + { \cal O } _ { q _ { 0 } } ( q ^ { 4 } ) , \qquad \frac { \Psi ( q _ { 0 } ) } { q _ { 0 } } q \leq \Psi ( q ) \leq 2 q .
$$

Proof. Step 1: monotonicity and concavity. Let $z = 2 { \sqrt { q ( 1 - q ) } }$ and $w = 1 - 2 q$ , so $z ^ { 2 } + w ^ { 2 } = 1$ Then

$$
\begin{array} { l l l } { \displaystyle \Psi ( q ) = \frac { 1 + z } 2 \log ( 1 + z ) + \frac { 1 - z } 2 \log ( 1 - z ) , } \\ { \displaystyle \Psi ^ { \prime } ( q ) = \frac { 2 w } { z } \mathrm { a r c t a n h } z , } \\ { \displaystyle \Psi ^ { \prime \prime } ( q ) = \frac { 4 ( z - \mathrm { a r c t a n h } z ) } { z ^ { 3 } } < 0 . } \end{array}
$$

Here $\Psi ^ { \prime } ( q ) > 0$ and $\Psi ^ { \prime \prime } ( q ) < 0$ for $0 < q < 1 / 2$ . The endpoint values and $\Psi ^ { \prime } ( 0 ) = 2$ follow by continuity. Concavity places Ψ above its chord from 0 to $q _ { 0 }$ and below its tangent at 0, proving the linear bounds.

Step 2: expansion at zero. Integrating the power series for arctanh z gives

$$
\frac { 1 + z } { 2 } \log ( 1 + z ) + \frac { 1 - z } { 2 } \log ( 1 - z ) = \sum _ { j \geq 1 } \frac { z ^ { 2 j } } { 2 j ( 2 j - 1 ) } .
$$

Substitute $z ^ { 2 } = 4 q ( 1 - q )$ and collect the first three powers of $q .$ The omitted term is uniformly $O _ { q _ { 0 } } ( q ^ { 4 } )$ by analyticity at zero and smoothness on a compact interval ending below $1 / 2$ □

## A.2 Uniform cubic remainder

Proof of Lemma 3.2. Step 1: express the known rate as a minimum. This is an algebraic reformulation of the rate in (10). Take z = tanh v. Then

$$
\begin{array} { c } { { \log \displaystyle \frac { ( 1 + z ) ^ { p } + ( 1 - z ) ^ { p } } { 2 } = \log \cosh ( p v ) - p \log \cosh v , } } \\ { { F ( p , q ) = \displaystyle \operatorname* { m i n } _ { v > 0 } \{ \log \cosh ( p v ) - p g _ { q } ( v ) \} , } } \\ { { g _ { q } ( v ) = \log \cosh v + q \log \operatorname { t a n h } v + \displaystyle \frac { 1 } { 2 } { \sf H } ( q ) . } } \end{array}
$$

Diferentiating the above expression in braces gives

$$
\begin{array} { c l l } { { } } & { { } } & { { \displaystyle \frac { \partial } { \partial v } \big \{ \log \cosh ( p v ) - p g _ { q } ( v ) \big \} = \displaystyle \frac { p } { \sinh v \cosh v } \big ( q _ { p } ( v ) - q \big ) , } } \\ { { } } & { { } } & { { q _ { p } ( v ) = \sinh v \cosh v \big ( \operatorname { t a n h } ( p v ) - \operatorname { t a n h } v \big ) = \displaystyle \frac { 1 } { 2 } \left( 1 - \displaystyle \frac { \cosh \big ( ( p - 2 ) v \big ) } { \cosh ( p v ) } \right) . } } \end{array}\tag{21}
$$

For $p \geq 2$ , the ratio on the last line decreases strictly from one to zero. This can be seen by inspecting its logarithmic derivative $( p - 2 ) \operatorname { t a n h } ( ( p - 2 ) v ) - p \operatorname { t a n h } ( p v ) < 0 .$

Thus $q _ { p }$ increases strictly from zero to $1 / 2$ . The derivative in (21) changes sign exactly once, from negative to positive. This proves the variational form and uniqueness. It also shows that evaluation at any value of v gives an upper bound on F.

Step 2: choose a test point v. Fix $q \in ( 0 , q _ { 0 } ]$ and set

$$
a = \operatorname { a r c t a n h } \sqrt { \frac { q } { 1 - q } } , \qquad q = \frac { \sinh ^ { 2 } a } { \cosh ( 2 a ) } .\tag{22}
$$

At $p = 2 .$ , the minimizer is $v = a$ . For $p = 2 + 2 s .$ , we can use $v = a / ( 1 + s )$ since this keeps $p v = 2 a$ fixed, making the first term of the variational expression constant. Define

$$
M _ { q } ( s ) = \log \cosh ( 2 a ) - 2 ( 1 + s ) g _ { q } \biggl ( \frac { a } { 1 + s } \biggr ) .
$$

Then $F ( 2 + 2 s , q ) \leq M _ { q } ( s )$

Step 3: verify cancellation up to the second order. Let a<sub>0</sub> = arctanh $\sqrt { q _ { 0 } / ( 1 - q _ { 0 } ) }$ . Basic algebra using (22) yields

$$
\begin{array} { c } { { 2 g _ { q } ( a ) = \log \cosh ( 2 a ) , } } \\ { { g _ { q } ^ { \prime } ( a ) = \operatorname { t a n h } ( 2 a ) , } } \\ { { g _ { q } ^ { \prime \prime } ( a ) = 0 , } } \\ { { 2 a \operatorname { t a n h } ( 2 a ) - \log \cosh ( 2 a ) = \Psi ( q ) . } } \end{array}\tag{23}
$$

To verify these identities, one can simply take $z = \operatorname { t a n h } a$ . Then $q = z ^ { 2 } / ( 1 + z ^ { 2 } )$ and

$$
\begin{array} { r l } & { { \sf H } ( q ) = \log ( 1 + z ^ { 2 } ) - 2 q \log z , } \\ & { g _ { \boldsymbol { q } } ( a ) = \log \cosh a + \frac { 1 } { 2 } \log ( 1 + z ^ { 2 } ) = \frac { 1 } { 2 } \log \cosh ( 2 a ) , } \\ & { g _ { \boldsymbol { q } } ^ { \prime \prime } ( a ) = \mathrm { s e c h } ^ { 2 } a - q \frac { \cosh ( 2 a ) } { \sinh ^ { 2 } a \cosh ^ { 2 } a } = 0 . } \end{array}
$$

The first derivative follows from $g _ { q } ^ { \prime } ( a ) = \operatorname { t a n h } a + q / ( \sinh a \cosh a ) = \operatorname { t a n h } ( 2 a )$ . Also tanh $( 2 a ) =$ $2 { \sqrt { q ( 1 - q ) } }$ . For $u = \operatorname { t a n h } ( 2 a )$ , log $2 - { \mathsf { H } } ( ( 1 - u ) / 2 ) = 2 a u - \log \cosh ( 2 a )$ , which proves the last line of (23).

Write $w = 1 + s$ and $v = a / w$ . Diferentiation yields

$$
\begin{array} { l } { { { \displaystyle M _ { q } ^ { \prime } ( s ) = 2 \{ v g _ { q } ^ { \prime } ( v ) - g _ { q } ( v ) \} , } } } \\ { { { \displaystyle M _ { q } ^ { \prime \prime } ( s ) = - \frac { 2 v ^ { 2 } } { w } g _ { q } ^ { \prime \prime } ( v ) } , } } \\ { { { \displaystyle M _ { q } ^ { \prime \prime \prime } ( s ) = \frac { 2 } { w ^ { 2 } } \{ 3 v ^ { 2 } g _ { q } ^ { \prime \prime } ( v ) + v ^ { 3 } g _ { q } ^ { \prime \prime \prime } ( v ) \} } . } } \end{array}\tag{24}
$$

It follows from (23) that

$$
M _ { q } ( 0 ) = 0 , \qquad M _ { q } ^ { \prime } ( 0 ) = \Psi ( q ) , \qquad M _ { q } ^ { \prime \prime } ( 0 ) = 0 .
$$

Step 4: bound the remainder uniformly as $q \downarrow 0$ . Taylor’s theorem concludes the proof once $| M _ { q } ^ { \prime \prime \prime } ( s ) | \le C _ { q _ { 0 } } q$

Let ℓ(v) = log tanh v. Its derivatives satisfy

$$
\begin{array} { l } { { \ell ^ { \prime \prime } ( v ) = - \displaystyle \frac { 4 \cosh ( 2 v ) } { \sinh ^ { 2 } ( 2 v ) } , } } \\ { { \ell ^ { \prime \prime \prime } ( v ) = - \displaystyle \frac { 8 } { \sinh ( 2 v ) } + \displaystyle \frac { 1 6 \cosh ^ { 2 } ( 2 v ) } { \sinh ^ { 3 } ( 2 v ) } . } } \end{array}
$$

For $0 < v \le a _ { 0 }$ , the quantities $v ^ { 2 } | \ell ^ { \prime \prime } ( v ) |$ and $v ^ { 3 } | \ell ^ { \prime \prime \prime } ( v ) |$ are bounded by constants depending only on $a _ { 0 }$ . This can be seen directly from sinh $( 2 v ) \geq 2 v$ , since their limits at zero are 1 and 2. Moreover,

$$
v ^ { 2 } \leq a ^ { 2 } \leq q \cosh ( 2 a _ { 0 } ) ,
$$

because $q = \sinh ^ { 2 } a / \cosh ( 2 a )$ and sinh $a \geq a$ . The derivatives of log cosh v are bounded on $[ 0 , a _ { 0 } ]$ In $g _ { q } ( v ) =$ log cosh $v + q \ell ( v ) + \mathsf { H } ( q ) / 2$ , the first term therefore contributes $O _ { q _ { 0 } } ( v ^ { 2 } )$ and $O _ { q _ { 0 } } ( v ^ { 3 } )$ to the weighted second and third derivatives respectively, while the $\ell$ term contributes $O _ { q 0 } ( q )$ . Since $v \leq a _ { 0 }$ and $v ^ { 2 } = O _ { q 0 } ( q )$ , this gives

$$
\vert v ^ { 2 } g _ { q } ^ { \prime \prime } ( v ) \vert \leq C _ { q _ { 0 } } q \ , \quad \vert v ^ { 3 } g _ { q } ^ { \prime \prime \prime } ( v ) \vert \leq C _ { q _ { 0 } } q .
$$

For $0 \leq s \leq 1 / 2$ , (24) yields

$$
| M _ { q } ^ { \prime \prime \prime } ( s ) | \leq C _ { q _ { 0 } } q .
$$

Taylor’s theorem and the bound on the third derivative give

$$
M _ { q } ( s ) \leq s \Psi ( q ) + C _ { 0 } q s ^ { 3 } .
$$

Since $F ( 2 + 2 s , q ) \leq M _ { q } ( s )$ , this proves (11). Finally, we apply Theorem 3.1 and multiply by $d ,$ using $d q = k$ , to obtain (12). □

## B Classical analytic and probabilistic tools

This section collects the hypercontractive, empirical-process, and Gaussian probability bounds used in the sampling arguments.

## B.1 Bonami–Beckner hypercontractivity

We use the classical degree form of the Bonami–Beckner inequality [Bon70; Bec75].

Lemma B.1 (Bonami–Beckner inequality). For $p \geq 2$ and every $h \in \mathcal { P } _ { d , k }$

$$
\begin{array} { r } { ( \mathbb { E } [ | h | ^ { p } ] ) ^ { 1 / p } \leq ( p - 1 ) ^ { k / 2 } ( \mathbb { E } [ h ^ { 2 } ] ) ^ { 1 / 2 } . } \end{array}
$$

In particular, for $\mathbb { E } [ h ^ { 2 } ] = 1$ 2

$$
\mathbb { E } [ h ^ { 4 } ] \leq 9 ^ { k } , \qquad \log \mathbb { E } [ | h | ^ { 2 + 2 s } ] \leq k ( 1 + s ) \log ( 1 + 2 s ) \quad ( s \geq 0 ) .
$$

## B.2 A bound independent of the ambient dimension

For our unrestricted-dimension result we also need the classical Bonami–Beckner hypercontractive estimate. The clipping step is the same as in Proposition 3.3.

Proposition B.2 (Classical hypercontractive comparison). For $k \ge 1 , \eta \in ( 0 , 1 )$ , and $T _ { k } ( \eta ) =$ $\mathrm { s u p } _ { d \geq 1 } T _ { d , k } ( \eta )$ ，

$$
\log T _ { k } ( \eta ) \leq 2 k + \left( \frac { 1 } { 6 } + 2 \log \frac { 1 } { 1 - \eta } \right) k ^ { 1 / 3 } .
$$

Proof. If $\mathbb { E } [ h ^ { 2 } ] = 1$ and deg $h \leq k$ , Lemma B.1 with $p = 2 + 2 s$ gives

$$
\log \mathbb { E } [ | h | ^ { 2 + 2 s } ] \leq k ( 1 + s ) \log ( 1 + 2 s ) \leq 2 k s + \frac { 2 } { 3 } { k s ^ { 3 } } \qquad ( 0 \leq s \leq 1 / 2 ) .
$$

Step 1: a Taylor bound. Indeed, $f ( s ) = ( 1 + s ) \log ( 1 + 2 s )$ satisfies $f ( 0 ) = f ^ { \prime \prime } ( 0 ) = 0 , f ^ { \prime } ( 0 ) = 2$ , and $f ^ { \prime \prime \prime } ( s ) = 4 ( 1 - 2 s ) / ( 1 + 2 s ) ^ { 3 } \leq 4$ on this interval.

Step 2: clipping. We take $L = \log ( 1 / ( 1 - \eta ) ) , s = k ^ { - 1 / 3 } / 2$ , and log $B = 2 k + ( 2 / 3 ) k s ^ { 2 } + L / s$ Then

$$
\begin{array} { r } { \mathbb { E } [ ( h ^ { 2 } - B ) _ { + } ] \le B ^ { - s } \mathbb { E } [ | h | ^ { 2 + 2 s } ] \le e ^ { - L } = 1 - \eta , } \end{array}
$$

so $\mathbb { E } [ h ^ { 2 } \wedge B ] \geq \eta$ . Substituting s proves the bound uniformly over d and h.

## B.3 A bounded empirical-process inequality

We use the following result in its bounded, centered form. For convenience, we include the zero function in the class, ensuring that $Z \geq 0$

Theorem B.3 (Bousquet, Theorem 2.3 of [Bou02]). Let $X _ { 1 } , \ldots , X _ { n }$ be independent and identically distributed, and let $\mathcal { G }$ be a countable class of measurable real functions such that

$$
\mathbb { E } [ g ] = 0 , \qquad | g | \le 1 , \qquad \mathbb { E } [ g ^ { 2 } ] \le v \quad ( g \in \mathcal { G } ) .
$$

Assume $0 \in \mathcal G$ , and put $Z = \operatorname* { s u p } _ { g \in { \mathcal { G } } } \sum _ { i = 1 } ^ { n } g ( X _ { i } )$ . Then, for every $t > 0$

$$
\mathbb { P } \left( Z > \mathbb { E } Z + \sqrt { 2 ( n v + 2 \mathbb { E } Z ) t } + \frac { t } { 3 } \right) \le e ^ { - t } .
$$

By continuity in the coeficient vector, all suprema may be taken over a countable dense subset of the unit sphere, so Bousquet’s countability assumption is satisfied.

## B.4 Symmetrization and contraction

The following are two standard results, see [BLM13, Lemma 11.4 and Theorem 11.6].

Lemma B.4 (Symmetrization and contraction). Let $\varepsilon _ { 1 } , \ldots , \varepsilon _ { n }$ be independent random signs, each uniform on $\{ - 1 , 1 \}$

(i) Symmetrization. Let $X _ { 1 } , \ldots , X _ { n }$ be independent and identically distributed, independently of the signs. For a countable class F of integrable functions,

$$
\mathbb { E } \operatorname* { s u p } _ { f \in \mathcal { F } } | \mathbb { E } _ { n } [ f ] - \mathbb { E } [ f ] | \leq 2 \mathbb { E } \operatorname* { s u p } _ { f \in \mathcal { F } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \varepsilon _ { i } f ( X _ { i } ) \right| .
$$

The expectation on the right includes both the sample and the signs.

(ii) Contraction. For a bounded set ${ \mathcal { A } } \subseteq \mathbb { R } ^ { n }$ and an L-Lipschitz function $\varphi : \mathbb { R }  \mathbb { R }$ satisfying $\varphi ( 0 ) = 0$

$$
\mathbb { E } _ { \varepsilon } \operatorname* { s u p } _ { a \in \mathcal { A } } \left. \sum _ { i = 1 } ^ { n } \varepsilon _ { i } \varphi ( a _ { i } ) \right. \leq 2 L \mathbb { E } _ { \varepsilon } \operatorname* { s u p } _ { a \in \mathcal { A } } \left. \sum _ { i = 1 } ^ { n } \varepsilon _ { i } a _ { i } \right. .
$$

## B.5 Elementary Gaussian probability bounds

Lemma B.5 (Bounds with all ranges explicit). For $r \geq 1$ , the chi-squared variable defined in Section 1.1 satisfies the following bounds. For $u \geq 0$ and $0 < a < 1$ ，

$$
\mathbb { P } ( \chi _ { r } ^ { 2 } \geq 4 ( r + u ) ) \leq e ^ { - u } ,\tag{25}
$$

$$
\mathbb { P } ( \chi _ { r } ^ { 2 } \leq a r ) \leq ( a e ^ { 1 - a } ) ^ { r / 2 } .\tag{26}
$$

For a one-dimensional standard Gaussian $G$ and every $t \geq$ log 4,

$$
\frac { 1 5 } { 1 6 } e ^ { - t / 2 5 6 } \mathbb { P } ( G ^ { 2 } \geq t / 6 4 ) > e ^ { - t } .\tag{27}
$$

Proof. Completing the square in the Gaussian integral gives $\mathbb { E } [ e ^ { v \chi _ { r } ^ { 2 } } ] = ( 1 - 2 v ) ^ { - r / 2 }$ for $v < 1 / 2$ Exponential Markov with $v = 1 / 4$ gives

$$
\begin{array} { r } { \mathbb { P } ( \chi _ { r } ^ { 2 } \geq 4 ( r + u ) ) \leq e ^ { - r - u } 2 ^ { r / 2 } \leq e ^ { - u } . } \end{array}
$$

For the lower tail use $v = - ( 1 - a ) / ( 2 a ) < 0 { : }$

$$
\mathbb { P } ( \chi _ { r } ^ { 2 } \le a r ) \le e ^ { - v a r } ( 1 - 2 v ) ^ { - r / 2 } = ( a e ^ { 1 - a } ) ^ { r / 2 } .
$$

To prove $( 2 7 )$ , first suppose log $4 \leq t \leq 1 6$ . Then $\sqrt { t / 6 4 } \le { 1 / 2 }$ . The Gaussian density is bounded by $( 2 \pi ) ^ { - 1 / 2 }$ , so

$$
\mathbb { P } ( | G | \ge 1 / 2 ) \ge 1 - ( 2 \pi ) ^ { - 1 / 2 } > 3 / 5 .
$$

The left side of (27) is then greater than $( 1 5 / 1 6 ) e ^ { - 1 / 1 6 } ( 3 / 5 ) > 1 / 2$ , while $e ^ { - t } \leq 1 / 4$

For $t \geq 1 6 ,$ , put $b = \sqrt { t } / 8$ . Integrate over $[ b , b + 1 ]$ and $[ - b - 1 , - b ]$

$$
\mathbb { P } ( G ^ { 2 } \geq t / 6 4 ) \geq { \sqrt { 2 / \pi } } e ^ { - ( b + 1 ) ^ { 2 } / 2 } = { \sqrt { 2 / \pi } } e ^ { - 1 / 2 } e ^ { - t / 1 2 8 - { \sqrt { t } } / 8 } > { \frac { 1 } { 4 } } e ^ { - 5 t / 1 2 8 } ,
$$

since $\sqrt { t } / 8 \leq t / 3 2$ . Thus the left side of (27) exceeds $( 1 5 / 6 4 ) e ^ { - 1 1 t / 2 5 6 }$ . This is larger than $e ^ { - t }$ for every $t \geq 1 6$ □

Corollary B.6 (OLS after lower conditioning). $I f \mathbb { P } ( \widehat { \Sigma } _ { V } \succeq \frac { 1 } { 2 } I _ { m } ) \geq 1 - \delta / 2$ , where $\delta \leq 1 / 4$ , then

$$
\mathcal { R } _ { n , { \delta } } ^ { * } ( V ) \leq 1 2 \sigma ^ { 2 } \frac { m + \log ( 1 / \delta ) } { n } .
$$

OLS satisfies the same bound uniformly over θ.

Proof. Conditionally on a design with $\widehat { \Sigma } _ { V } \succeq I _ { m } / 2$

$$
\| \widehat { \theta } _ { \mathrm { O L S } } - \theta \| _ { 2 } ^ { 2 } \ \preceq _ { \mathrm { s t } } \ \frac { 2 \sigma ^ { 2 } } { n } \chi _ { m } ^ { 2 } .
$$

Use (25) with $u = t + \log 2$ , where $t = \log ( 1 / \delta )$ , and add the design failure probability. Since $t \geq \log 4$ , log $2 \leq t / 2$ , and $8 ( m + t + \log 2 ) \leq 1 2 ( m + t )$

The minimax assertion follows because OLS is an admissible estimator (or from Theorem 5.2).

## C The radial-weight identity

We derive Lemma 4.1 directly from [FM19, Theorem 3.10]. We first apply that theorem to the uniform cube to identify its measure-independent constant, and then to the cube weighted by $H ( S _ { d } )$

Proof. Filmus–Mossel’s theorem states that, for homogeneous harmonic multilinear $f , g$ of degrees $r , s$ and any exchangeable probability measure $\alpha , \mathbb { E } _ { \alpha } [ f g ] = 0 { \mathrm { ~ i f ~ } } r \neq s$ . If $r = s .$ , there is a constant $C _ { f , g }$ independent of α such that

$$
\mathbb { E } _ { \alpha } [ f g ] = C _ { f , g } \mathbb { E } _ { \alpha } \left[ \prod _ { j = 1 } ^ { r } ( x _ { 2 j - 1 } - x _ { 2 j } ) ^ { 2 } \right] .
$$

Apply this first with $\alpha ~ = ~ \mu _ { d }$ . Each squared diference has expectation 2, and the pairs are independent, hence

$$
C _ { f , g } = 2 ^ { - r } \langle f , g \rangle _ { d } .
$$

For $H \geq 0$ with $Z = \mathbb { E } _ { \mu _ { d } } [ H ( S _ { d } ) ] > 0$ , now choose

$$
\alpha ( x ) = \frac { H ( S _ { d } ( x ) ) } { Z } \mu _ { d } ( x ) .
$$

This probability measure is exchangeable because both $S _ { d }$ and $\mu _ { d }$ are invariant under coordinate permutations. Substituting it into the theorem and multiplying by Z gives

$$
\mathbb { E } _ { \mu _ { d } } [ f g H ( S _ { d } ) ] = \langle f , g \rangle _ { d } \mathbb { E } _ { \mu _ { d } } [ Q _ { r } ^ { 2 } H ( S _ { d } ) ] , \qquad Q _ { r } = \prod _ { j = 1 } ^ { r } \frac { x _ { 2 j - 1 } - x _ { 2 j } } { \sqrt { 2 } } .
$$

On the other hand, for unequal degrees, the same substitution yields zero.

To evaluate the remaining expectation, let $A _ { r }$ be the event that the signs difer in each of the first r pairs. Then

$$
\operatorname* { P r } _ { \mu _ { d } } ( A _ { r } ) = 2 ^ { - r } , \qquad Q _ { r } ^ { 2 } = 2 ^ { r } \mathbf { 1 } _ { A _ { r } } .
$$

Conditional on $A _ { r }$ , those pairs contribute zero to $S _ { d }$ , while the remaining $d - 2 r$ coordinates are independent uniform signs. Thus

$$
\begin{array} { r } { \mathbb { E } _ { \mu _ { d } } [ Q _ { r } ^ { 2 } H ( S _ { d } ) ] = \mathbb { E } _ { \mu _ { d } } [ H ( S _ { d } ) \mid A _ { r } ] = \mathbb { E } _ { \mu _ { d - 2 r } } [ H ( S _ { d - 2 r } ) ] , } \end{array}
$$

which is the equal-degree case of (14). The empty product for $r = 0$ is one. If $H \geq 0$ and $Z = 0$ the weighted identities are immediate since $H ( S _ { d } ) = 0$ on the cube. Finally, decomposing a real H into its positive and negative parts extends the result to signed weights by linearity. □

## D Probability constants for the sampling lower bounds

We give the complete probability bounds for the stable-sampling and Gaussian-risk obstructions from Lemma 5.1. Throughout, $\mathbb { P } ( E ) = p \in ( 0 , 1 / 2 ]$ . The shared-set case uses (16). For $t \geq m$ , the directional assumption in Lemma 5.1(ii) sufices.

Lemma D.1 (Baseline sample complexity). For $d , k \geq 1 , 1 \leq m \leq D _ { d , k } , 0 < \delta \leq 1 / 4 , 0 < c < 1$ and $A \geq 1 6$ , there is a universal $c _ { 0 } > 0$ such that

$$
N _ { \mathrm { s t a b } } ( d , k , m , c , \delta ) \wedge N _ { \mathrm { p a r } } ( d , k , m , A , \delta ) \geq c _ { 0 } \big ( m + \log ( 1 / \delta ) \big ) .
$$

Proof of Lemma $D . 1$ . If $n < m$ , every feature matrix is singular, so both guarantees fail. This gives the lower bound m.

For the confidence term, include $h ( x ) = ( 1 + x _ { 1 } ) / \sqrt { 2 }$ in the model and extend it to an orthonormal basis of any feasible size m. If $t = \log ( 1 / \delta )$ , then

$$
n \leq \frac { t } { 2 \log 2 } \quad \Longrightarrow \quad \mathbb { P } ( X _ { i 1 } = - 1 \mathrm { ~ f o r ~ e v e r y ~ } i ) = 2 ^ { - n } \geq e ^ { - t / 2 } > \delta .
$$

On this event the feature column for h is zero, so both guarantees fail. Taking an integer sample size below the displayed bound yields a universal constant times t. If that range contains no positive integer, use $N \geq 1$ . Finally, m $\vee t \geq ( m + t ) / 2$ □

Proposition D.2 (Stable-sampling obstruction). Fix $0 < c < 1$ and choose $0 < \rho < c / 2 5 6$ . Under (16), or under the directional assumption in Lemma ${ 5 . 1 ( i i ) }$ when $t \geq m ,$ for every $t \geq \log 4$ and every positive integer $n \leq ( m + t ) / ( 1 0 2 4 p )$ , we have

$$
\mathbb { P } ( \widehat { \Sigma } \not \sqgeq c I _ { m } ) > e ^ { - t } .
$$

Proof. If $t < m$ , then $m \geq 2$ and $n \leq m / ( 5 1 2 p )$ , so Lemma $5 . 1 ( \mathrm { i } )$ applies. Its small eigenvalues are strictly below $^ { c , }$ and its probability $1 5 / 1 6$ exceeds $e ^ { - t } \leq 1 / 4$ . If $t \geq m$ , then $n \leq t / ( 5 1 2 p )$ , and part (ii) applies. Now $3 2 \rho < c .$ , and

$$
{ \frac { 1 5 } { 1 6 } } e ^ { - t / 2 5 6 } > e ^ { - t } \qquad ( t \geq \log 4 ) .
$$

This proves the strict failure bound in both regimes.

Proposition D.3 (Estimator-independent parametric obstruction). Fix $A \geq 1 6$ . There is $\rho _ { A } > 0$ such that, under (16), or under the directional assumption in Lemma ${ 5 . 1 ( i i ) }$ when $t \geq m$ , with $\rho \leq \rho _ { A }$ , for every $t \geq$ log 4 and every positive integer $n \leq ( m + t ) / ( 1 0 2 4 p )$ , the Gaussian model satisfies

$$
\mathcal { R } _ { n , e ^ { - t } } ^ { * } ( V ) > A \sigma ^ { 2 } \frac { m + t } { n } .
$$

Proof. Take $b _ { 0 } = 1 / ( 1 0 2 4 e )$ and choose $\begin{array} { r } { 0 < \rho _ { A } < \operatorname* { m i n } \left\{ \frac { 1 } { 8 } , \frac { b _ { 0 } } { 3 0 7 2 A } , \frac { 1 } { 1 6 3 8 4 A } \right\} } \end{array}$ . Use the random variable $Z _ { n , V }$ in Theorem 5.2. A singular design makes it infinite, so it is enough to focus on nonsingular designs.

Suppose first that $t < m$ . Then $m \geq 2$ . With probability at least $1 5 / 1 6$ , at least $r = \lfloor m / 2 \rfloor \geq m / 3$ eigenvalues of $\widehat { \Sigma }$ are at most $1 2 8 \rho$ . Conditional on any such nonsingular design,

$$
Z _ { n , V } ~ \succeq _ { \mathrm { s t } } ~ \frac { \sigma ^ { 2 } } { 1 2 8 \rho n } \chi _ { r } ^ { 2 } .
$$

By (26), $\mathbb { P } ( \chi _ { r } ^ { 2 } \ < \ b _ { 0 } r ) \ \le \ ( e b _ { 0 } ) ^ { r / 2 } \ \le \ 1 / 3 2$ . Thus, with unconditional probability at least $( 1 5 / 1 6 ) ( 3 1 / 3 2 ) > 1 / 4$

$$
Z _ { n , V } \geq \frac { \sigma ^ { 2 } b _ { 0 } r } { 1 2 8 \rho n } \geq \frac { \sigma ^ { 2 } b _ { 0 } m } { 3 8 4 \rho n } > 8 A \sigma ^ { 2 } \frac { m } { n } > A \sigma ^ { 2 } \frac { m + t } { n } .
$$

$\mathrm { ~ I f ~ } t \ge m$ , Lemma 5.1(ii) gives an event D of probability at least $( 1 5 / 1 6 ) e ^ { - t / 2 5 6 }$ on which $\lambda _ { 1 } ( \widehat { \Sigma } ) \leq 3 2 \rho$ . Conditional on such a design, one coordinate of the independent standard Gaussian vector yields

$$
Z _ { n , V } \succeq _ { \mathrm { s t } } \frac { \sigma ^ { 2 } } { 3 2 \rho n } G ^ { 2 } .
$$

By (27), the event

$$
Z _ { n , V } \geq \frac { \sigma ^ { 2 } t } { 2 0 4 8 \rho n }
$$

has probability strictly greater than $e ^ { - t }$ . Its displayed threshold is greater than $8 A \sigma ^ { 2 } t / n$ , hence greater than the benchmark because $m \leq t$ . This proves that the lower $\left( 1 - e ^ { - t } \right)$ -quantile of $Z _ { n , V }$ strictly exceeds the benchmark. Lastly, apply Theorem 5.2. □

## E Comparison to the Polyanskiy–Samorodnitsky uncertainty boundary

We restate the Polyanskiy–Samorodnitsky uncertainty principle in our notation and explain how it yields the leading exponent of our one-dimensional clipping threshold when $k / d$ tends to a fixed value in $( 0 , 1 / 2 )$

Let $B _ { d , \iota }$ denote a Hamming ball of radius $\lfloor r d \rfloor$ and put $\begin{array} { r } { r _ { * } ( q ) = \frac { 1 } { 2 } - \sqrt { q ( 1 - q ) } } \end{array}$

Let $\Pi { \le } k$ be orthogonal projection onto $\mathcal { P } _ { d , k }$ and $C _ { E } = \Pi _ { \le k } \mathbf { 1 } _ { E } \Pi _ { \le k }$ its concentration operator, where $\mathbf { 1 } _ { E }$ acts by multiplication. Its eigenvalues are denoted $\kappa _ { 1 } ( C _ { E } ) \geq \kappa _ { 2 } ( C _ { E } ) \geq \cdot \cdot \cdot$

Then, Theorem 9 of [PS19] implies the following statements for fixed $q \in ( 0 , 1 / 2 )$ and fixed $r \in ( 0 , 1 / 2 )$ , with $k = \lfloor q d \rfloor$ : for some $c = c ( q , r ) > 0$ and all suficiently large $d ,$

$$
\begin{array} { r l } { r < r _ { * } ( q ) , \quad | E | \le e ^ { d { \mathsf { H } } ( r ) } } & { \Longrightarrow \quad \kappa _ { 1 } ( C _ { E } ) \le e ^ { - c d } , } \\ { r > r _ { * } ( q ) , \quad E = B _ { d , r } } & { \Longrightarrow \quad \kappa _ { 1 } ( C _ { E } ) \ge 1 - e ^ { - c d } . } \end{array}
$$

The cosine of their angle between the spatial and Fourier subspaces is the square root of $\kappa _ { 1 } ( C _ { E } )$ squaring only changes the constants in these bounds. The boundary equation is $( 1 - 2 r ) ^ { 2 } + ( 1 - 2 q ) ^ { 2 } =$ 1, and $\mu _ { d } ( B _ { d , r } ) = \exp \{ - d ( \log 2 - \mathsf { H } ( r ) ) + o ( d ) \}$ . Thus the critical probability exponent is precisely $E _ { d , k }$

For each fixed $\eta \in ( 0 , 1 )$ , these statements imply

$$
\log T _ { d , \lfloor q d \rfloor } ( \eta ) = d \Psi ( q ) + o ( d ) .
$$

For the upper bound, choose $B = \exp \{ d ( \Psi ( q ) + \varepsilon ) \}$ with fixed $\varepsilon > 0$ . For every unit polynomial h of degree at most $\lfloor q d \rfloor$ , the set $E = \{ h ^ { 2 } > B \}$ has probability at most $1 / B$ . The first implication above therefore gives $\mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E } ] = o ( 1 )$ , uniformly in h, and hence $\mathbb { E } [ h ^ { 2 } \wedge B ] \geq 1 - o ( 1 ) \geq \eta$ for suficiently large d.

For the lower bound, the second implication provides a unit polynomial h with $\mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] = o ( 1 )$ on a Hamming ball E whose probability exponent is arbitrarily close to $\Psi ( q )$ . Choosing this ball so that $\mu _ { d } ( E ) \leq \exp \{ - d ( \Psi ( q ) - \varepsilon / 2 ) \}$ and setting $B = \exp \{ d ( \Psi ( q ) - \varepsilon ) \}$ gives

$$
\mathbb { E } [ h ^ { 2 } \wedge B ] \leq \mathbb { E } [ h ^ { 2 } \mathbf { 1 } _ { E ^ { c } } ] + B \mu _ { d } ( E ) = o ( 1 ) < \eta .
$$

Thus log $T _ { d , \lfloor q d \rfloor } ( \eta ) / d  \Psi ( q )$

## F Further bounds and model examples

We derive the additional bounds and justify the model examples discussed in Section 7.

## F.1 Ambient dimension and model dimension

The finite-dimensional correction. By Lemma $\mathrm { A . 1 }$

$$
\Psi ( q ) = 2 q - \frac { 2 } { 3 } q ^ { 2 } - \frac { 8 } { 1 5 } q ^ { 3 } + O ( q ^ { 4 } ) .
$$

Substituting this expansion into Theorem 1.5 gives, throughout the matching range and as $k / d \to 0$

$$
\log \frac { N _ { \mathrm { p a r } } } { m + t } = 2 k - \frac { 2 k ^ { 2 } } { 3 d } + O \left( \frac { k ^ { 3 } } { d ^ { 2 } } + k ^ { 1 / 3 } \right) .
$$

Thus the first correction to the unrestricted-dimension exponent $2 k \mathrm { ~ i s ~ } - 2 k ^ { 2 } / ( 3 d )$ . Relative to its magnitude, the two error terms have orders $k / d$ and $d / k ^ { 5 / 3 }$ , respectively. Both vanish when $k \ll d \ll k ^ { 5 / 3 }$ . Alternatively, if $k / d \to q \in ( 0 , 1 / 2 )$ , then

$$
\frac { 1 } { d } \log \frac { N _ { \mathrm { p a r } } } { m + t } \longrightarrow \Psi ( q )
$$

throughout the matching range.

Worst case over the ambient dimension. Let $N _ { \mathrm { p a r } } ^ { \infty }$ be the least threshold valid on every cube admitting a model of dimension $m ,$ and define $N _ { \mathrm { s t a b } } ^ { \infty }$ analogously. For either threshold $N ^ { \infty }$ , we show that

$$
\log \frac { N ^ { \infty } } { m + t } = 2 k + O ( k ^ { 1 / 3 } ) ,
$$

with constants depending on the fixed target parameters.

For the upper bound, Propositions B.2 and 3.4 and Corollary B.6 give

$$
N ^ { \infty } \leq C ( m + t ) \exp \{ 2 k + C k ^ { 1 / 3 } \} ,
$$

uniformly over $d ,$ including $k > d .$ . For the lower bound, it sufices to exhibit one suitable ambient dimension. Choose

$$
\begin{array} { r } { d = \left\lceil \operatorname* { m a x } \{ m + 1 , 1 0 k , k ^ { 5 / 3 } \} \right\rceil . } \end{array}
$$

Then $m < d \leq M _ { d , k }$ and $k / d \leq 1 / 1 0$ , so the matching lower bound applies. Moreover,

$$
E _ { d , k } \geq 2 k - C k ^ { 2 } / d \geq 2 k - C k ^ { 1 / 3 } .
$$

The lower bound of Theorem 1.5, and its stable-sampling counterpart, therefore prove the claim. The same choice of $d ,$ together with the classical upper bound and Proposition 3.3, also gives

$$
\log \operatorname* { s u p } _ { d } T _ { d , k } ( \eta ) = 2 k + O _ { \eta } ( k ^ { 1 / 3 } ) .
$$

A lower bound for larger model dimensions. By allocating more degree to the harmonic directions, we can extend the lower bound beyond $m \le M _ { d , k }$ , at the cost of a larger error. Specifically, for any integer $0 \leq R \leq k$ with $m \leq { \binom { d } { R } }$ , we have

$$
\log \frac { N _ { \mathrm { p a r } } } { m + t } \geq E _ { d , k } - C ( R + k ^ { 1 / 3 } ) .
$$

The same bound holds for $N _ { \mathrm { s t a b } } .$ , with constants depending on its fixed conditioning level.

To prove this, Proposition 4.2 supplies a common concentration set of probability at most $e ^ { - a _ { R } }$ where

$$
a _ { R } = \mathcal { E } _ { d , k } ( R ) - C ( k - R ) ^ { 1 / 3 } .
$$

If $a _ { R } \geq \log 2$ , this probability is at most $1 / 2 ,$ so Proposition D.3 gives

$$
\log \frac { N _ { \mathrm { p a r } } } { m + t } \geq a _ { R } - O ( 1 ) \geq E _ { d , k } - C ^ { \prime } ( R + k ^ { 1 / 3 } ) ,
$$

where the last inequality uses (15). If $a _ { R } < \log 2$ , the same entropy comparison gives $E _ { d , k } \ \leq$ $C ^ { \prime } ( R + k ^ { 1 / 3 } ) + O ( 1 )$ . In this case, the baseline bound $N _ { \mathrm { p a r } } \geq c _ { 0 } ( m + t )$ from Lemma D.1 proves the claim after increasing the constant. For stable sampling, use Proposition D.2 in place of Proposition D.3.

The leading exponent for subexponential model dimensions. Suppose $k / d \to q \in ( 0 , 1 / 2 )$ and log $m = o ( d )$ . Then, even outside the cube-root matching range,

$$
\log \frac { N _ { \mathrm { p a r } } } { m + t } = E _ { d , k } + o ( d ) ,
$$

and likewise for $N _ { \mathrm { s t a b } }$

Indeed, choose $R = \lceil \log _ { 2 } m \rceil$ , with $R = 0$ when $m = 1$ . We have $R = o ( d )$ , so eventually $R \le k / 2$ and $d \geq 2 R$ . For $R > 0$ ，

$$
{ \binom { d } { R } } \geq ( d / R ) ^ { R } \geq 2 ^ { R } \geq m .
$$

The preceding lower bound therefore applies, with error $O ( R + k ^ { 1 / 3 } ) = o ( d )$ . The upper bound of Theorem 1.5 also has error $o ( d )$ and holds for every feasible m. Together these establish the leading exponent, uniformly in $t \geq \log 4$ , without asserting an $O ( k ^ { 1 / 3 } )$ remainder throughout this larger dimension range.

## F.2 Known Walsh coordinates and finite-space caps

For an isotropic feature vector with $\| \Phi ( x ) \| _ { 2 } ^ { 2 } \leq L .$ , matrix Chernof [Tropp12, Corollary 5.2] gives, for $0 < c < 1$ ，

$$
\mathbb { P } ( \lambda _ { \operatorname* { m i n } } ( \widehat { \Sigma } ) \leq c ) \leq m \exp \left( - \frac { ( 1 - c ) ^ { 2 } n } { 2 L } \right) .\tag{28}
$$

Indeed, its matrix summands are positive semidefinite, have norm at most $L ,$ and their expectations sum to $n I _ { m }$ . The standard Chernof factor is $[ e ^ { - u } / ( 1 - u ) ^ { 1 - u } ] ^ { n / L }$ with $u = 1 - c .$ . The inequality $- u - ( 1 - u ) \log ( 1 - u ) \leq - u ^ { 2 } / 2$ follows by diferentiation.

For a known span of m Walsh characters, $\| \Phi ( x ) \| _ { 2 } ^ { 2 } = m$ exactly. Hence $n \geq C _ { c } m ( \log m + t )$ gives stable sampling. Choosing $c = 1 / 2$ , failure probability $\delta / 2 ,$ and using Corollary B.6 gives parametric prediction. Character values need not be mutually independent; only independent sample rows are used.

Let $\chi _ { \leq k } ( x ) = ( \chi _ { S } ( x ) ) _ { | S | \leq \operatorname* { m i n } ( k , d ) } \in \mathbb { R } ^ { D _ { d , k } }$ be the column vector of Walsh characters. Expanding each basis function of V in this basis gives

$$
\Phi _ { V } ( x ) = U \chi _ { \leq k } ( x ) ,
$$

where $U \in \mathbb { R } ^ { m \times D _ { d , k } }$ contains the expansion coeficients.

Since U has orthonormal rows, $\| \Phi _ { V } ( x ) \| _ { 2 } ^ { 2 } \leq \| \chi _ { \leq k } ( x ) \| _ { 2 } ^ { 2 } = D _ { d , k }$ . Thus (28) gives

$$
N \leq C D _ { d , k } ( \log m + t )
$$

for either the stable-sampling or parametric threshold, with C depending on the fixed target parameters.

Likewise, every unit polynomial $h \in \mathcal { P } _ { d , k }$ satisfies $h ( x ) ^ { 2 } \leq D _ { d , k }$ . The inequality $z \wedge ( \eta D _ { d , k } ) \geq \eta z$ for $0 \leq z \leq D _ { d , k }$ therefore gives

$$
\mathbb { E } [ h ^ { 2 } \wedge ( \eta D _ { d , k } ) ] \geq \eta , \qquad T _ { d , k } ( \eta ) \leq \eta D _ { d , k } .
$$

When $k \geq d ,$ equality follows by taking $h = 2 ^ { d / 2 } { \bf 1 } _ { \{ x = x _ { 0 } \} }$ for any cube vertex $x _ { 0 }$ . These bounds reflect the finite size of the cube: once $k \geq d ,$ , the polynomial space no longer grows, so the threshold cannot continue to scale as $e ^ { 2 k }$