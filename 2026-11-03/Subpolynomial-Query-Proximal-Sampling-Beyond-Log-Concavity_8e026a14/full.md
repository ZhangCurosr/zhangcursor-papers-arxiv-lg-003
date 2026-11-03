# Subpolynomial-Query Proximal Sampling Beyond Log-Concavity

Wei Guo<sup>1</sup> Bo Yuan<sup>1</sup> Ye He<sup>2</sup> Jiaming Liang<sup>3</sup> Molei Tao<sup>1</sup> Yongxin Chen<sup>1∗</sup>

<sup>1</sup>Georgia Institute of Technology <sup>2</sup>University of Colorado Boulder <sup>3</sup>University of Rochester

## Abstract

We study the gradient query complexity of sampling from a density $\pi ( x ) \propto \mathrm { e } ^ { - V ( x ) }$ on $\mathbb { R } ^ { d } .$ where V has a β-Lipschitz gradient but need not be convex. Our approach uses the proximal sampler, which alternates Gaussian perturbation with sampling a position conditional on the noisy observation. We implement its observation chain using the high-order noisy conditional sampler of [Ope26], obtaining improved dimension dependence under functional inequalities. If π satisfies a log-Sobolev inequality (LSI) with constant $\alpha ,$ then for every fixed $\nu > 0$ , there is a finite constant $C _ { \nu }$ depending only on ν such that the algorithm returns a sample within total-variation distance ε using at most $C _ { \nu } \kappa ( d + \kappa + \varepsilon ^ { - 1 } + \| \nabla V ( x _ { 0 } ) \| ^ { 2 } / \beta ) ^ { \nu }$ queries, where $\kappa = \beta / \alpha$ and $x _ { 0 }$ is an arbitrary reference point for initialization. The analysis controls center-dependent local errors along exact proximal histories and removes the final observation noise by conditional decoding. Under a Poincaré inequality (PI), we obtain a corresponding guarantee from a warm start with a known bound on its Rényi divergence. An oracle lower bound shows that a stationary point alone, which sufices under LSI, is insuficient under PI.

## 1 Introduction

Sampling from a density $\pi ( x ) \propto \mathrm { e } ^ { - V ( x ) } \mathrm { o n } \mathbb { R } ^ { d }$ is a fundamental task in statistics, scientific computing, and machine learning. A major line of research studies how the cost of sampling scales with dimension under strong log-concavity and smoothness. Underdamped Langevin dynamics and randomized midpoint discretizations have improved the polynomial dimension dependence of Wasserstein sampling guarantees [CCBJ18, SL19]. Recently, OpenAI [Ope26] established a qualitatively diferent result: for $C ^ { 2 }$ potentials with a known minimizer and $I _ { d } \preceq \nabla ^ { 2 } V \preceq 2 I _ { d }$ , sampling to total-variation accuracy $1 / 1 0$ requires at most $C _ { \nu } d ^ { \nu }$ exact first-order queries for every fixed $\nu > 0$ , with unrestricted computation between queries. Thus, a fixed positive power of the dimension is not intrinsic to query complexity in this well-conditioned setting.

A parallel line of research replaces strong log-concavity with a log-Sobolev inequality (LSI) or a Poincaré inequality (PI), allowing the potential to be nonconvex. Analyses of Langevin and proximal samplers have established polynomial-complexity guarantees under these weaker assumptions, with appropriate initialization conditions [VW19, CEL<sup>+</sup>22, CCSW22, FYC23]; see [Che22] for a comprehensive review. Under LSI and smoothness assumptions, representative improvements include $d ^ { 1 / 2 }$ dimension dependence for rejection-based proximal sampling [FYC23] and, more recently, $d ^ { 1 / 5 }$ dependence for the Gaussian cloud sampler [CCLZ26b]. These guarantees have diferent accuracy and initialization requirements. Together with [Ope26], they raise a natural question: can subpolynomial query complexity in dimension also be achieved for smooth targets satisfying LSI or PI, with the required initialization and its cost made explicit? We investigate this question through the proximal sampling framework.

Introduced in [LST21], the proximal sampler alternates Gaussian perturbation with sampling a position conditional on the noisy observation. This is Gibbs sampling on an augmented space. The conditional draw, called the restricted Gaussian oracle (RGO), adds quadratic regularization to the potential. When the proximal step size is $\tau < 1 / \beta$ , the conditional potential has Hessian bounded below by $( \tau ^ { - 1 } - \beta ) I _ { d }$ and is therefore strongly convex. The proximal sampler admits convergence guarantees under LSI and PI [CCSW22], with eficient RGO implementations based on approximate rejection sampling [FYC23]. Accelerated variants, including Proximal BPS and reflected proximal dynamics, have also been developed under strong log-concavity [CCLZ26a, YGC26]. Our goal is to improve the dimension dependence of proximal sampling for smooth targets that need not be log-concave.

We build on the local sampling algorithm of [Ope26] to obtain query bounds with an arbitrarily small positive dimension exponent. Its output includes the Gaussian noise needed for the next proximal step, allowing us to run the chain on noisy positions and recover an approximate sample from π only at the end. The main dificulty is that the error in each sampling step depends on the gradient of the potential at the current noisy position. We use LSI or PI to control these errors along the chain without assuming global convexity.

Contributions. For every fixed $\nu > 0$ , we obtain the following guarantees with a finite constant $C _ { \nu }$ depending only on ν. The algorithm only requires gradient queries to the target potential, and the query complexity holds uniformly over its internal randomness.

• Under LSI and smoothness assumptions, we obtain a sampler with total-variation error at most ε using

$$
C _ { \nu } \kappa \big ( d + \kappa + \varepsilon ^ { - 1 } + G _ { 0 } \big ) ^ { \nu }
$$

gradient queries for every fixed $\nu > 0$ (Thm. 3). Here $\kappa = \beta / \alpha$ , where $\beta$ is the smoothness constant and α is the LSI constant. The algorithm starts from $\mathcal { N } ( x _ { 0 } , \beta ^ { - 1 } I _ { d } )$ at any supplied point $x _ { 0 } .$ whose quality enters through $G _ { 0 } = \| \nabla V ( x _ { 0 } ) \| ^ { 2 } / \beta$

• Under PI and smoothness assumptions, suppose an initial sample from η<sub>0</sub> can be generated using at most $Q _ { \mathrm { i n i t } }$ gradient queries, with a known bound $D _ { 2 } ( \eta _ { 0 } \| \pi ) \le B , B \ge 1$ . For every fixed $\nu > 0$ we obtain a sampler with total-variation error at most ε using at most

$$
Q _ { \mathrm { i n i t } } + C _ { \nu } \kappa \big [ B + \log ( 8 / \varepsilon ) \big ] \big ( d + \kappa + B + \varepsilon ^ { - 1 } \big ) ^ { \nu }
$$

gradient queries (Thm. 4), where $\kappa = \beta /$ α and α is the PI constant. We also show that a supplied stationary point alone is insuficient for a uniform query bound under PI, even when it is a strict local minimum (Thm. 11).

Technical overview. The local sampler of [Ope26] approximates a conditional position with an independent Gaussian perturbation added. This is exactly the noise used to form the next proximal observation, so one call implements an observation update. We therefore run the chain on noisy positions and, at the end, approximately sample a position conditioned on the last observation. We call this final conditional sampling procedure the decoder; Fig. 1 summarizes the algorithm.

![](images/906f850306e57061991b8e13fc558db512193a69514d13df22e180662d28de71.jpg)  
Figure 1: Structure of Alg. 1. The padded target has dimension $n \geq d .$ . After the observation updates and final conditional sampling, we discard the appended coordinates and undo the rescaling to obtain x.

The error of each local call depends on the potential gradient at the current observation. Our comparison argument averages these errors along the exact proximal chain, where LSI or PI supplies mixing and gradient-moment bounds. This avoids having to control gradient moments along the approximate chain. The decoder error is bounded under the stationary observation law, and Prop. 8 combines the mixing, transition, and decoder errors.

## 2 Background and main results

Assumption on the target distribution. Throughout this paper, we consider the target distribution $\pi ( x ) \propto \mathrm { e } ^ { - V ( x ) } \mathrm { o n } \mathbb { R } ^ { d }$ with $d \geq 2$ , where $V \in C ^ { 2 } (  { \mathbb { R } } ^ { d } )$ is possibly non-convex but has a β-Lipschitz gradient, so that $- \beta I _ { d } \preceq \nabla ^ { 2 } V \preceq \beta I _ { d }$ . The complexity of our algorithms is quantified through the number of gradient queries $x \mapsto \nabla V ( x )$ , and potential values $V ( x )$ are not required for the algorithms.

Notations and definitions. For a random variable X, Law(X) denotes its distribution, and for a measurable map T, $T _ { \# } \mu : = \operatorname { L a w } ( T ( X ) )$ for $X \sim \mu$ . For probability densities µ and η on $\mathbb { R } ^ { d }$ , the total-variation distance is $\operatorname { T V } ( \mu , \eta ) = \operatorname* { s u p } _ { A } | \mu ( A ) - \eta ( A ) |$ |, the Kullback–Leibler (KL) divergence is $\begin{array} { r } { \mathrm { K L } ( \mu \| \eta ) = \int \mu ( x ) \log \frac { \mu ( x ) } { \eta ( x ) } } \end{array}$ dx, the $\chi ^ { 2 }$ divergence is $\begin{array} { r } { \chi ^ { 2 } ( \mu \| \eta ) = \int \frac { \mu ( x ) ^ { 2 } } { \eta ( x ) } \mathrm { d } x - 1 } \end{array}$ , and the Rényi divergence of order two is $D _ { 2 } ( \mu \| \eta ) = \log ( 1 + \chi ^ { 2 } ( \mu \| \eta ) )$ . Pinsker’s inequality gives $2 \mathrm { T V } ^ { 2 } \leq \mathrm { K L }$ , and Jensen’s inequality gives $\mathrm { K L } \leq \log ( 1 + \chi ^ { 2 } )$ . We write C for a positive constant whose value may change from line to line. It never depends on $d , n , \kappa , \varepsilon ,$ or the initialization; when it depends on the fixed parameters $( \gamma , K , \omega )$ of NoisySample $\left( \mathrm { P r o p . ~ 5 } \right)$ , we mention it explicitly.

## 2.1 The proximal sampler

The proximal sampler [LST21, CCSW22] augments $x \sim \pi$ with an auxiliary variable y under the joint density $\Gamma _ { \tau } ( x , \bar { y } ) \propto \mathrm { \tilde { e } } ^ { - V ( x ) - \| x - y \| ^ { 2 } / ( 2 \tau ) }$ , where $\tau > 0$ is the step size. Its x-marginal is $\pi _ { \mathrm { : } }$ , and its conditional distributions are

$$
y \mid x \sim { \mathcal { N } } ( x , \tau I _ { d } ) , \qquad x \mid y \sim q _ { y } ^ { \tau } ( x ) \propto \mathrm { e } ^ { - V ( x ) - \| x - y \| ^ { 2 } / ( 2 \tau ) } .\tag{1}
$$

An exact draw from $q _ { y } ^ { \tau }$ is called a restricted Gaussian oracle (RGO) draw. The proximal sampler is the Gibbs chain that draws $y _ { k } \sim \mathcal N ( x _ { k } , \tau I _ { d } )$ and then $x _ { k + 1 } \sim q _ { y _ { k } } ^ { \tau }$ . It preserves $\Gamma _ { \tau }$ for every $\tau ,$ so the position chain $( x _ { k } )$ has stationary law $\pi ,$ and the observation chain $( y _ { k } )$ has stationary law

$\pi _ { \tau } : = \pi \ast { \mathcal { N } } ( 0 , \tau I _ { d } )$ . Since $\nabla ^ { 2 } V \succeq - \beta I _ { d }$ , every $q _ { y } ^ { \tau }$ is strongly log-concave when $\tau < 1 / \beta _ { \mathrm { { i } } }$ , even if π is not. Smaller step sizes make the RGO easier but slow the mixing of the chain, as we will see in Prop. 2.

## 2.2 Functional inequalities

Definition 1 (Functional inequalities). Let Ent<sub>π</sub> $\begin{array} { r } { . ( g ) : = \mathbb { E } _ { \pi } [ g \log \frac { g } { \mathbb { E } _ { \pi } g } ] } \end{array}$ for $g \geq 0$ . We say that π satisfies an α-logarithmic Sobolev inequality (LSI) or an α-Poincaré inequality (PI) if, respectively, the following inequalities hold for all smooth f:

$$
\mathrm { E n t } _ { \pi } ( f ^ { 2 } ) \leq { \frac { 2 } { \alpha } } \mathbb { E } _ { \pi } \| \nabla f \| ^ { 2 } ,\tag{LSI}
$$

$$
\begin{array} { r } { \operatorname { V a r } _ { \pi } f \leq \displaystyle \frac { 1 } { \alpha } \mathbb { E } _ { \pi } \| \nabla f \| ^ { 2 } . } \end{array}\tag{PI}
$$

Note that α-strongly log-concave targets satisfy α-LSI, and α-LSI implies α-PI [BGL14].

Unlike strong log-concavity, both inequalities are stable under bounded perturbations of V [BGL14], so they allow saddle points and multiple local minima (see the example at the end of Sec. 4). A larger α means the distribution is better concentrated and typically implies faster mixing, and $\kappa : = \beta / \alpha$ plays the role of a condition number as in the strongly log-concave case. Valid constants always satisfy $\alpha \le \beta$ (Lem. 18), so $\kappa \geq 1$

Proposition 2 (Convergence of the ideal proximal sampler). Fix $\tau > 0$ and run (1) with exact conditional draws. Let $\rho _ { k } = \operatorname { L a w } ( x _ { k } )$ and assume the relevant initial divergence below is finite. For every integer $k \geq 0$

$$
\begin{array} { r } { \mathrm { K L } ( \rho _ { k } | | \pi ) \le ( 1 + \alpha \tau ) ^ { - 2 k } \mathrm { K L } ( \rho _ { 0 } | | \pi ) , \qquad i f \pi \ s a t i s f i e s \ \alpha \ – G I , } \end{array}\tag{2}
$$

$$
\chi ^ { 2 } ( \rho _ { k } \| \pi ) \leq ( 1 + \alpha \tau ) ^ { - 2 k } \chi ^ { 2 } ( \rho _ { 0 } \| \pi ) , \mathrm { ~ } i f \pi \mathit { \ s a t i s f i e s \ \alpha } \alpha - P I .\tag{3}
$$

The same respective upper bounds hold for the divergences of the observation law $\sigma _ { k } = \rho _ { k } { * } \mathcal { N } ( 0 , \tau I _ { d } ) =$ Law $( y _ { k } )$ from $\pi _ { \tau }$

Proof. The position bounds follow from [CCSW22, Thms. 3 and 4]. Applying the same Gaussian convolution kernel to $\rho _ { k }$ and $\pi$ contracts both divergences by data-processing inequalities, giving the observation bounds. □

These bounds require neither global convexity nor $\tau < 1 / \beta$ for the ideal chain, and they furthermore imply geometric convergence in TV with factor $( 1 + \alpha \tau ) ^ { - k }$ . With a step size τ of order $1 / \beta ,$ about ${ \widetilde { O } } ( \kappa )$ iterations (up to a logarithmic factor in the initial divergence and the accuracy threshold) therefore sufice for the desired accuracy.

## 2.3 Sampling guarantees

We now state the main theorems of the complexity of the proposed algorithm, Alg. 1.

Theorem 3 (Gaussian-start sampling under LSI). Suppose π satisfies $\mathbf { \alpha } _ { \alpha - L S I }$ . For every fixed $\nu > 0$ there is a finite $C _ { \nu }$ such that the following holds. Let $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d }$ be any point provided as an input to the algorithm, and set $G _ { 0 } = \| \nabla V ( x _ { 0 } ) \| ^ { 2 } / \beta$ . For every $0 < \varepsilon < 1 / 4$ , Alg. 1 returns a sample whose law πb satisfies $\operatorname { T V } ( { \widehat { \pi } } , \pi ) \leq \varepsilon$ using at most

$$
C _ { \nu } \kappa ( d + \kappa + \varepsilon ^ { - 1 } + G _ { 0 } ) ^ { \nu }\tag{4}
$$

gradient queries.

Theorem 4 (Finite-Rényi-start sampling under PI). Suppose π satisfies α-PI. For every fixed $\nu > 0$ , there is a finite $C _ { \nu }$ such that the following holds. Assume an initial sample x<sub>init</sub> ∼ η<sub>0</sub> can be generated using only gradient access to $V _ { i }$ , with deterministic query cost at most $Q _ { \mathrm { i n i t } }$ , and a known $B \geq 1$ satisfies $D _ { 2 } ( \eta _ { 0 } \| \pi ) \le B$ . For every $0 < \varepsilon < 1 / 4 , A l g$ . 1 returns a sample whose law $\widehat { \pi }$ satisfies $\operatorname { T V } ( { \widehat { \pi } } , \pi ) \leq \varepsilon$ using at most

$$
Q _ { \mathrm { i n i t } } + C _ { \nu } \kappa [ B + \log ( 8 / \varepsilon ) ] ( d + \kappa + B + \varepsilon ^ { - 1 } ) ^ { \nu }\tag{5}
$$

gradient queries.

Interpretation. Both theorems run Alg. 1 and difer only in its initialization. Under LSI, $x _ { 0 }$ is an arbitrary input point; the cost of finding it is not counted, and its quality enters only through $G _ { 0 }$ , which vanishes at any stationary point, including a saddle point. Under PI, the initial sampler and the bound B are inputs, and the cost $Q _ { \mathrm { i n i t } }$ of the initial sample is charged to the total. The prefactor κ reflects the roughly κ iterations needed by the exact proximal sampler (Prop. 2), and all other parameters enter through a power $\nu > 0$ that can be taken arbitrarily small, with the algorithm and $C _ { \nu }$ depending on ν. Thus, for fixed $\kappa , \varepsilon ,$ , and $G _ { 0 }$ or $B _ { i }$ , the number of gradient queries grows more slowly than any positive power of the dimension.

## 3 Implementation of the proximal sampler

The proximal sampler of Sec. 2.1 needs an exact draw from the RGO $q _ { y } ^ { \tau }$ at every step. Rather than implementing these draws directly, our implementation rests on two observations.

First, the Gaussian noise already present in the proximal sampler allows us to work with a smoothed conditional distribution. Given an observation $y ,$ the next observation is $y ^ { \prime } = x ^ { \prime } + { \sqrt { \tau } } G$ , where $x ^ { \prime } \sim q _ { y } ^ { \tau }$ and $G \sim \mathcal { N } ( 0 , I _ { d } )$ are independent. Writing $x ^ { \prime } = y + { \sqrt { \tau } } Z$ , we have $y ^ { \prime } = y + { \sqrt { \tau } } ( Z + G )$ 2 where Z follows

$$
\operatorname { L a w } ( Z ) ( v ) \propto \operatorname { e } ^ { - { \frac { \| v \| ^ { 2 } } { 2 } } - V ( y + { \sqrt { \tau } } v ) } .
$$

Thus, updating the observation requires a sample of $Z + G$ . The additional Gaussian noise smooths the distribution of $Z .$ . The construction of [Ope26] combines this smoothing with high-order approximation to implement the noisy conditional draw using few gradient queries. This is central to obtaining an arbitrarily small dimension exponent.

Second, we observe that the same sampling construction can extend beyond strong log-concavity to targets satisfying LSI or PI. These weaker assumptions still provide enough control over the chain to establish convergence and bound the accumulated sampling errors. The two settings require diferent initialization conditions and analysis, which we make explicit below.

In the following, we first state the guarantee of NoisySample from [Ope26] in Sec. 3.1. We then describe how to rescale the target and append independent Gaussian coordinates (Sec. 3.2), and how to sample a position conditioned on the final noisy observation (Sec. 3.4). The complete procedure is given in Alg. 1, and Prop. 8 bounds its total approximation error. Throughout, hats denote approximate kernels and probability laws.

## 3.1 The noisy local sampling algorithm

For a perturbation $F \in C ^ { 2 } ( \mathbb { R } ^ { n } )$ , center $c \in \mathbb { R } ^ { n }$ , and scale $0 < r \leq 1$ , define

$$
\rho _ { c , r } ^ { F } ( v ) \propto \mathrm { e } ^ { - \| v \| ^ { 2 } / 2 - F ( c + r v ) } .\tag{6}
$$

With $n = d , F = F _ { y } , c = 0$ , and $r = 1$ , this is the law of $Z$ above after the transformation of the RGO target q<sup>τ</sup><sub>y</sub> . Given $c , r ,$ , and access to $\nabla F _ { \ i }$ , the algorithm NoisySample approximates $Z + G$ for independent $\check { Z } \sim \rho _ { c , r } ^ { F }$ and $G \sim { \mathcal { N } } ( 0 , I _ { n } )$ without exact sample of $Z .$ . Its construction is explained in App. B.1. The next result establishes the convergence guarantee of NoisySample and the required gradient query complexity.

Proposition 5 (Noisy local sampling guarantee). Fix a localization exponent $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ , an approximation order $K \geq 2$ , and a query-overhead exponent $\omega > 0$ such that

$$
4 0 ( K + 2 ) \omega < 1 / 4 .\tag{7}
$$

There are a constant C and a dimension threshold, both depending only on $( \gamma , K , \omega )$ , and a bound $A _ { n } \leq C ( 1 + \log n ) ^ { C }$ , with the following property. For every n above the threshold, $\delta \ : = \ : n ^ { - \gamma }$ ， $\mathrm { L i p } ( \nabla F ) \leq \delta , c \in \mathbb { R } ^ { n }$ , and $0 < r \leq 1$ , the algorithm NoisySample $( F , c , r )$ uses at most $A _ { n } \delta ^ { - \omega }$ gradient queries to F and satisfies

$$
\mathrm { T V } ( \mathrm { L a w } ( \mathsf { N o i s y } \mathsf { S a m p l e } ( F , c , r ) ) , \rho _ { c , r } ^ { F } \ast \mathcal { N } ( 0 , I _ { n } ) ) \leq A _ { n } \delta ^ { K } ( \sqrt { n } + \| \nabla F ( c ) \| ) .\tag{8}
$$

The constants are uniform in $F , c , r$

This guarantee follows from [Ope26, Thm. 2.1]. We suppress the fixed parameters $( \gamma , K , \omega )$ in the notation NoisySample $( F , c , r )$ .

## 3.2 Normalization and Gaussian padding

We now construct the padded target described at the beginning of this section. Let $a = x _ { 0 }$ under LSI and $a = 0$ under PI. For $w = ( z , \xi ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { n - d }$ , define the forward and output maps

$$
T _ { a } ( x ) = \sqrt { \beta } ( x - a ) , \qquad H _ { a } ( w ) = a + z / \sqrt { \beta } ,\tag{9}
$$

and the padded potential and target

$$
\widetilde V ( w ) = V ( a + z / \sqrt { \beta } ) + \Vert \xi \Vert ^ { 2 } / 2 , \qquad \Pi = ( T _ { a } ) _ { \# } \pi \otimes \mathcal { N } ( 0 , I _ { n - d } ) \propto \mathrm { e } ^ { - \widetilde V } .\tag{10}
$$

Then $H _ { a \# } \Pi = \pi$ , the potential $\widetilde { V }$ is 1-smooth with LSI or PI constant $1 / \kappa ,$ , and each gradient of $\widetilde { V }$ costs one gradient of V (Lem. 12). In these coordinates the step size becomes $h = \beta \tau$

Proximal chain on the padded target. The two conditionals of (1), for Π and step size $h ,$ are the kernels

$$
A _ { h } ( y \mid w ) = ( 2 \pi h ) ^ { - n / 2 } \mathrm { e } ^ { - \| y - w \| ^ { 2 } / ( 2 h ) } , \qquad R _ { h } ( w \mid y ) \propto \mathrm { e } ^ { - { \widetilde { V } } ( w ) - \| w - y \| ^ { 2 } / ( 2 h ) } .
$$

For a law $\mu ,$ , write $\begin{array} { r } { \mu A _ { h } ( y ) = \int \mu ( w ) A _ { h } ( y \mid w ) } \end{array}$ dw, similarly for $R _ { h }$ , and compose kernels from left to right by convention. The observation chain $y  w ^ { \prime }  y ^ { \prime }$ then has kernel $P _ { h } = R _ { h } A _ { h }$ . With $\Pi _ { h } : = \Pi A _ { h } = \Pi * { \mathcal { N } } ( 0 , h I _ { n } )$ , we have

$$
\Pi _ { h } { \cal P } _ { h } = \Pi _ { h } , \qquad \Pi _ { h } { \cal R } _ { h } = \Pi ,
$$

i.e., the observation chain is stationary at $\Pi _ { h }$ , and a position drawn given a stationary observation is an exact sample of Π, the step the decoder in Sec. 3.4 approximates. Let $X _ { j } \sim \lambda _ { j }$ and $Y _ { j } \sim \mu _ { j }$ be the positions and observations of the exact chain, so that $\mu _ { j } = \lambda _ { j } A _ { h } = \mu _ { 0 } P _ { h } ^ { j }$

## 3.3 Implementing one observation step

Let $F _ { y } ( v ) = \widetilde { V } ( y + \sqrt { h } v )$ . By the argument at the beginning of this section, $P _ { h } ( \cdot \mid y )$ is the law of $y + { \sqrt { h } } ( Z + G )$ for independent $Z \sim \rho _ { 0 , 1 } ^ { F _ { y } }$ and $G \sim { \mathcal { N } } ( 0 , I _ { n } )$ . The implemented kernel $\widehat { P } _ { h } ( \cdot \mid y )$ replaces $Z + G$ by the output of NoisySample:

$$
\widehat { Y } ^ { \prime } = y + \sqrt { h } \mathsf { N o i s y } \mathsf { S a m p l e } ( F _ { y } , 0 , 1 ) .
$$

The gradient $g _ { y } ( v ) = \nabla F _ { y } ( v ) = \sqrt { h } \nabla  { \widetilde { V } } ( y + \sqrt { h } v )$ is h-Lipschitz, and in Alg. 1 we have $h = \delta / 2 \leq \delta$ and n above the threshold of Prop. 5. Since TV is unchanged by the common afine map $u \mapsto y + { \sqrt { h } } u .$ (8) with $c = 0 , r = 1$ , and $\nabla F _ { y } ( 0 ) = \sqrt { h } \nabla \tilde { V } ( y )$ gives

$$
\begin{array} { r } { \mathrm { T V } ( \widehat { P } _ { h } ( \cdot  { |  { y } _ { h } ( \cdot  { |  { y } ) } ) } \le A _ { n } \delta ^ { K } ( \sqrt { n } + \sqrt { h } \| \nabla \widetilde { V } ( y ) \| ) . } \end{array}\tag{11}
$$

## 3.4 The final decoder

After N steps the observation chain is close to $\Pi _ { h } .$ , not Π. A position sample requires one draw from $R _ { h } ( \cdot \mid y )$ at $y = Y _ { N }$ , that is, $y + { \sqrt { h } } Z$ with $Z \sim \rho _ { 0 . 1 } ^ { F _ { y } }$ , but NoisySample only returns the noisy variable $Z + G$ . The decoder closes this gap with the following identity.

Lemma 6 (Variance halving [Ope26]). Let $Z \sim \rho _ { c , r } ^ { F }$ and $G \sim { \mathcal { N } } ( 0 , I _ { n } )$ be independent, and set $U = Z + G$ . Then

$$
( Z \mid U = u ) \stackrel { \mathrm { l a w } } { = } u / 2 + Z ^ { \prime } / \sqrt { 2 } , \qquad w h e r e \quad Z ^ { \prime } \sim \rho _ { c + r u / 2 , r / \sqrt { 2 } } ^ { F } .\tag{12}
$$

Equivalently, given one noisy observation u, the variable $c + r Z$ has the law of $c ^ { \prime } + r ^ { \prime } Z ^ { \prime }$ for the same family with center $c ^ { \prime } = c + r u / 2$ and scale $r ^ { \prime } = r / \sqrt { 2 }$ . Repeating this with fresh calls to NoisySample shrinks the scale geometrically, until the remaining conditional is nearly Gaussian and can be sampled directly. For fixed $y ,$ the decoder runs as follows.

1. Set $c _ { 0 } = 0 , r _ { 0 } = 1$ , and $J = \left\lceil 4 \log _ { 2 } n \right\rceil$ . For $j = 0 , \ldots , J - 1$ , draw $U _ { j } = { \mathsf { N o i s y } } { \mathsf { S a m p l e } } ( F _ { y } , c _ { j } , r _ { j } )$ and set $c _ { j + 1 } = c _ { j } + r _ { j } U _ { j } / 2$ and $r _ { j + 1 } = r _ { j } / \sqrt { 2 }$

2. Now $r _ { J } \leq n ^ { - 2 }$ , so $\rho _ { c _ { J } , r _ { J } } ^ { F _ { y } }$ is close to a Gaussian with identity covariance centered at its mode, which is the fixed point of $v \mapsto - r _ { J } g _ { y } ( c _ { J } + r _ { J } v )$ . Approximate the mode by $m = \lceil \log n \rceil$ fixed-point steps,

$$
v ^ { ( 0 ) } = 0 , \qquad v ^ { ( k + 1 ) } = - r _ { J } g _ { y } ( c _ { J } + r _ { J } v ^ { ( k ) } ) .\tag{13}
$$

3. Draw a fresh $G \sim { \mathcal { N } } ( 0 , I _ { n } )$ and return

$$
\mathsf { D e c o d e } ( y ) = y + \sqrt { h } \{ c _ { J } + r _ { J } ( v ^ { ( m ) } + G ) \} .\tag{14}
$$

Write $\widehat { R } _ { h }$ for the kernel $y \mapsto \operatorname { L a w } ( \mathsf { D e c o d e } ( y ) )$ . With exact draws in place of NoisySample and of the final Gaussian approximation, Lem. 6 shows that Decode $( y ) \sim R _ { h } ( \cdot \mid y )$ exactly. For an approximate implementation, we quantify the error and query complexity in the following proposition.

Proposition 7 (Decoder accuracy). For every smooth normalizable $\widetilde { V }$ as in Lem. 13, the decoder satisfies

$$
\mathrm { T V } ( \Pi _ { h } \widehat { R } _ { h } , \Pi ) \leq C J A _ { n } \delta ^ { K } \sqrt { n } + C \delta n ^ { - 7 / 2 } .\tag{15}
$$

Its deterministic query cost is at most $J A _ { n } \delta ^ { - \omega } + m$

Algorithm 1 Subpolynomial-Query Proximal sampling   
Require: Normalized and padded potential $\widetilde { V }$ on $\mathbb { R } ^ { n _ { \mathrm { . } } }$ ; initial position $X _ { 0 } ;$ step size $h ;$ number of iterations   
N; original dimension $d ;$ shift $a \in \mathbb { R } ^ { d }$ and scale $\beta > 0 .$   
Observation updates   
1: Draw $G \sim \mathcal { N } ( 0 , \overset { \cdot } { I } _ { n } )$ and set $Y _ { 0 } \gets X _ { 0 } + \sqrt { h } G .$   
2: for $j = 0 , \ldots , N - 1$ do   
3: Define $F _ { Y _ { j } } ( v ) = \widetilde { V } ( Y _ { j } + \sqrt { h } v ) .$   
4: U<sub>j</sub> ← NoisySample $( F _ { Y _ { j } } , 0 , 1 )$   
5: $Y _ { j + 1 } \gets Y _ { j } + \sqrt { h } U _ { j }$   
6: end for   
Sample a position conditioned on $Y _ { N }$   
7: Define $F ( v ) \overline { { } } = \widetilde { V } ( Y _ { N } + \sqrt { h } v )$   
8: Set $c \gets 0 , r \gets 1$ , and $J \gets \left\lceil 4 \log _ { 2 } n \right\rceil$   
9: for $\ell = 1 , \ldots , J$ do   
10: U ← NoisySample $( F , c , r )$   
11: $c  c + { \textstyle { \frac { r } { 2 } } } U .$   
12: $r  r / \sqrt { 2 } .$   
13: end for   
Approximate the remaining conditional by a Gaussian   
14: Set $v  0$ and $m \gets \lceil \log n \rceil .$   
15: for $k = 1 , \ldots , m$ do   
16: $v  - r \sqrt { h } \nabla \widetilde { V } \Big ( Y _ { N } + \sqrt { h } ( c + r v ) \Big )$ ▷ Approximate the conditional mode   
17: end for   
18: Draw a fresh $G \sim { \mathcal { N } } ( 0 , I _ { n } )$   
19: $\widehat { W } \gets Y _ { N } + \sqrt { h } \big [ c + r ( v + G ) \big ]$   
Return to the original coordinates   
20: $\widehat { \boldsymbol { x } } \gets \boldsymbol { a } + \beta ^ { - 1 / 2 } ( \widehat { W } _ { 1 } , \ldots , \widehat { W } _ { d } ) ^ { \mathrm { T } }$   
21: return ${ \widehat { x } } .$

The proof is in App. B. Its global input is the stationary score identity $\mathbb { E } _ { \Pi } \Vert \nabla \widetilde { V } \Vert ^ { 2 } \leq n$ . It does not require global convexity or proximity to a minimizer.

The outer algorithm Alg. 1 assembles the pieces of this section: padding, N implemented observation steps, and the decoder. The dimension n and the number of steps N are explicit functions of the inputs; they involve a fixed threshold $n _ { * } ( \nu )$ depending only on $\nu \ ( \mathrm { A p p . \ G } )$

Query accounting. All random draws in Alg. 1 are independent, and $n , N ,$ and all loop lengths are determined by the inputs, so the number of gradient queries has a deterministic upper bound. The algorithm uses only gradients of V . Taking $n \geq n _ { * } ( \nu )$ also avoids a low-dimensional fallback in [Ope26] that would require function values. Under PI, the initial sampler is likewise assumed to use only gradients.

## 3.5 Accumulation of approximation errors

Alg. 1 has three sources of error: the exact observation chain is run for only N steps, each step of the observation chain uses NoisySample with error (11), and the decoder is approximate. The following

proposition combines them. We remark that this result does not require functional inequalities;   
those will enter only through the two inputs $\varepsilon _ { \mathrm { m i x } }$ and $M _ { N }$ , which concern the exact chain.

Proposition 8 (From exact histories to implemented samples). Let $\mu _ { j } \ : = \ : \mu _ { 0 } P _ { h } ^ { j }$ be the exact observation laws and $\widehat { \mu } _ { N } = \mu _ { 0 } \widehat { P } _ { h } ^ { N }$ the implemented one. Suppose (11) holds, $\mathrm { T V } ( \mu _ { N } , \Pi _ { h } ) \le \varepsilon _ { \mathrm { m i x } }$ 2 and $\mathrm { T V } ( \Pi _ { h } \widehat { R } _ { h } , \Pi ) \leq \varepsilon _ { \mathrm { d e c } }$ . Define

$$
M _ { N } = \sum _ { j = 0 } ^ { N - 1 } \mathbb { E } _ { \mu _ { j } } [ \sqrt { n } + \sqrt { h } \| \nabla \widetilde { V } ( Y ) \| ] .\tag{16}
$$

Then

$$
\mathrm { T V } ( \widehat { \mu } _ { N } \widehat { R } _ { h } , \Pi ) \leq \varepsilon _ { \mathrm { m i x } } + A _ { n } \delta ^ { K } M _ { N } + \varepsilon _ { \mathrm { d e c } } .\tag{17}
$$

Moreover, the number $Q _ { \mathrm { p o s t } }$ of gradient queries made after initialization satisfies the deterministic bound

$$
Q _ { \mathrm { p o s t } } \leq ( N + J ) A _ { n } \delta ^ { - \omega } + m .\tag{18}
$$

Proof. For $0 \leq k \leq N$ , let $\theta _ { k } = \mu _ { 0 } P _ { h } ^ { k } \widehat { P } _ { h } ^ { N - k }$ be the law after k exact steps followed by $N - k$ implemented steps, so that $\theta _ { 0 } = \widehat { \mu } _ { N }$ and $\theta _ { N } = \mu _ { N }$ . The laws $\theta _ { k }$ and $\theta _ { k + 1 }$ difer only in step $k + 1$ ， which starts from $\mu _ { k }$ , and TV does not increase under the common remaining kernel $\widehat { P } _ { h } ^ { N - k - 1 }$ by the data-processing inequality. By the triangle inequality, summing over k and using (11), we have

$$
\operatorname { T V } ( { \widehat { \mu } } _ { N } , \mu _ { N } ) \leq \sum _ { k = 0 } ^ { N - 1 } \mathbb { E } _ { \mu _ { k } } \operatorname { T V } ( { \widehat { P } } _ { h } ( \cdot \vert Y ) , P _ { h } ( \cdot \vert Y ) ) \leq A _ { n } \delta ^ { K } M _ { N } .\tag{19}
$$

Applying the same decoder kernel $\widehat { R } _ { h }$ to $\widehat { \mu } _ { N } , \mu _ { N }$ , and $\Pi _ { h }$ and using the triangle inequality,

$$
\mathrm { T V } ( \widehat { \mu } _ { N } \widehat { R } _ { h } , \Pi ) \leq \mathrm { T V } ( \widehat { \mu } _ { N } , \mu _ { N } ) + \mathrm { T V } ( \mu _ { N } , \Pi _ { h } ) + \mathrm { T V } ( \Pi _ { h } \widehat { R } _ { h } , \Pi ) ,\tag{20}
$$

which gives (17). There are $N + J$ calls to NoisySample and m fixed-point steps, which proves (18). □

In summary, Prop. 8 reduces the analysis of Alg. 1 to two properties of the exact chain: its mixing error $\varepsilon _ { \mathrm { m i x } }$ and the score budget $M _ { N }$ , which averages the gradient under the exact laws $\mu _ { j }$ rather than the implemented ones. In the next two sections, we bound both quantities under LSI (Sec. 4) and under PI (Sec. 5).

## 4 Sampling under LSI

Under LSI, both $\varepsilon _ { \mathrm { m i x } }$ and $M _ { N }$ follow from a bound on the KL divergence of the Gaussian initialization (Lem. 14). This divergence decays geometrically along the exact chain (Prop. 2), which controls $\varepsilon _ { \mathrm { { m i x } } } ;$ since it never increases, Talagrand’s transport inequality also bounds the gradient moments in $M _ { N }$ (Lem. 15).

Proposition 9 (Exact-chain inputs under LSI). Let $n \geq \operatorname* { m a x } \{ d , \kappa , G _ { 0 } , \varepsilon ^ { - 1 } \} , \delta \leq 1 / 2$ , and $h = \delta / 2$ and let N be as in (32). For the exact chain started from $\lambda _ { 0 } = { \mathcal { N } } ( 0 , I _ { n } )$

$$
\varepsilon _ { \mathrm { m i x } } \le \varepsilon / 4 , \qquad M _ { N } \le C N \kappa \sqrt { n } , \qquad N \le C \kappa \delta ^ { - 1 } \log n .
$$

Substituting these bounds into Prop. 8, the implementation error is at most $C \kappa ^ { 2 } A _ { n } \delta ^ { K - 1 } \sqrt { n }$ log n and the algorithm makes at most $C \kappa A _ { n } ( \log n ) n ^ { \gamma ( 1 + \omega ) }$ queries. Choosing $\gamma$ small makes the query exponent less than $\nu ,$ and choosing $K$ large relative to $1 / \gamma$ then makes the implementation and decoder errors at most $\varepsilon / 4$ each. We detail the explicit choices in App. G. Mapping the output back with $H _ { a }$ gives Thm. 3. Full proof can be found in App. C. Note that the initial divergence can be of order κn, but it enters N only logarithmically.

A non-log-concave example. The potential $V _ { d } ( x ) = \| x \| ^ { 2 } / 2 + 2$ cos $x _ { 1 }$ is 3-smooth, has a negative Hessian eigenvalue at zero, and has LSI constant at least $\mathrm { e ^ { - 4 } }$ by bounded perturbation (App. B.6). Since $\nabla V _ { d } ( 0 ) = 0$ , Thm. 3 applies with $x _ { 0 } = 0$ and $G _ { 0 } = 0$ . As $\kappa \leq 3 \mathrm { { e } ^ { 4 } }$ is a constant, it gives $O _ { \nu } ( ( d + \varepsilon ^ { - 1 } ) ^ { \nu } )$ gradient queries for every fixed $\nu > 0$ . For fixed accuracy, this improves the dimension dependence of the $\widetilde { O } ( \kappa { \sqrt { d } } ) = \widetilde { O } ( \sqrt { d } )$ bound in [FYC23] and the $\widetilde { \cal O } ( \kappa d ^ { 1 / 5 } ) = \widetilde { \cal O } ( d ^ { 1 / 5 } )$ bound in [CCLZ26b], whose condition $G _ { 0 } \leq d / \kappa$ holds here.

## 5 Sampling under PI and initialization

## 5.1 Proof outline

We outline the proof of Thm. 4. As under LSI, it supplies the two exact-chain inputs $\varepsilon _ { \mathrm { m i x } }$ and $M _ { N }$ of Prop. 8, but two ingredients change. First, PI does not bound the divergence of a simple Gaussian start, so the initial law $\eta _ { 0 }$ and its bound $D _ { 2 } ( \eta _ { 0 } \| \pi ) \le B$ are inputs. Under PI the exact chain contracts $\chi ^ { 2 } \ : ( \mathrm { P r o p . ~ 2 } ) \ :$ , and $D _ { 2 } \leq B$ allows $\chi ^ { 2 }$ as large as $\mathrm { e } ^ { B } - 1$ , so the number of steps grows linearly in $B$ rather than logarithmically. Second, PI alone does not imply the quadratic transport inequality used in the LSI argument. It instead gives exponential concentration of Lipschitz functions (Lem. 16), which, by a change of measure, bounds the first gradient moment under any law within KL divergence B of Π (Lem. 17). A first moment is all that $M _ { N }$ requires. The result is as follows: Proposition 10 (Exact-chain inputs under PI). Let $n \geq$ max $\{ d , \kappa ( B + 1 ) ^ { 2 } \} , ~ \delta ~ \leq ~ 1 / 2$ , and $h = \delta / 2$ , let N be as in (38), and set $H = B + \log ( 8 / \varepsilon )$ . For the exact chain started from $\lambda _ { 0 } = ( T _ { 0 } ) _ { \# } \eta _ { 0 } \otimes \mathcal { N } ( 0 , I _ { n - d } )$

$$
\varepsilon _ { \mathrm { m i x } } \le \varepsilon / 4 , \qquad M _ { N } \le C N \sqrt { n } , \qquad N \le C \kappa H \delta ^ { - 1 } .
$$

Substituting these bounds into Prop. 8, the implementation error is at most $C \kappa H A _ { n } \delta ^ { K - 1 } \sqrt { n } .$ , and the algorithm makes at most $Q _ { \mathrm { i n i t } } + C \kappa H A _ { n } n ^ { \gamma ( 1 + \omega ) }$ queries. The parameters are then chosen as under LSI $\left( \operatorname { A p p . ~ G } \right)$ , and mapping the output back with $H _ { 0 }$ gives Thm. 4; App. D contains the full proof.

## 5.2 Initialization under PI

Thm. 4 takes the initial law $\eta _ { 0 }$ and its bound $D _ { 2 } ( \eta _ { 0 } \| \pi ) \le B$ as inputs. What information about π sufices to construct them? We now demonstrate that a global minimizer does, but an arbitrary stationary point does not.

A global minimizer sufices. If $x _ { * }$ is a global minimizer of $V _ { ; }$ , the Gaussian $q _ { 0 } = \mathcal { N } ( x _ { * } , ( 2 \beta ) ^ { - 1 } I _ { d } )$ satisfies $\begin{array} { r } { D _ { 2 } ( q _ { 0 } \| \pi ) \le \frac { d } { 2 } \log ( 2 \mathrm { e } \kappa ) } \end{array}$ (see Prop. 19, which also covers approximate minimizers). Thm. 4 thus applies with $Q _ { \mathrm { i n i t } } = 0$ and $\begin{array} { r } { B = \operatorname* { m a x } \{ 1 , \frac { d } { 2 } \log ( 2 \mathrm { e } \kappa ) \} } \end{array}$ , which gives $O ( d ^ { 1 + \zeta } )$ gradient queries for every fixed $\zeta > 0$ at fixed κ and ε. The extra factor of d compared with LSI comes from $B ,$ which enters the bound linearly.

![](images/b6db5825697e5b975802790beef88f0602e8756046cd9a08c9ee9caf7755631c.jpg)  
Figure 2: Schematic potential behind Thm. 11, not to scale. Every $V _ { R }$ has a strict local minimum at the reference point 0, but most of the mass of $\pi _ { R } \propto \operatorname { e } ^ { - V _ { R } }$ lies near the remote location R. Oracle answers in the shaded region $x \leq R - 1$ are the same for every $R ,$ so an algorithm with a fixed query budget cannot locate the mass when R is large.

A stationary point does not. Under LSI, any point $x _ { 0 }$ works, and its quality is measured by $G _ { 0 }$ . Under PI, no such guarantee is possible, even when $x _ { 0 }$ is a strict local minimum.

Theorem 11 (No uniform bound from an arbitrary stationary point). For fixed $d \geq 2$ , there is no finite query bound depending only on $d , \beta , \alpha , \varepsilon$ and $\| \nabla V ( x _ { 0 } ) \|$ that guarantees TV error $\varepsilon = 1 / 1 0$ for all smooth PI targets from an arbitrary supplied reference point. This holds for randomized adaptive exact value-and-gradient oracle algorithms with unrestricted computation between queries. The reference point can be a strict local minimum with Hessian $I _ { d }$

The proof uses a one-dimensional family of targets $\pi _ { R } \propto \operatorname { e } ^ { - V _ { R } }$ , sketched in Fig. 2. Each $\pi _ { R }$ is a smoothed Laplace distribution centered at a remote point $R ,$ plus a small bounded bump near the origin that creates a strict local minimum there. The bump changes the potential by a bounded amount, so all $\pi _ { R }$ are uniformly smooth and satisfy PI with a common constant. The key property is that, to the left of $R - 1$ , the potential and its gradient are the same for every R. An algorithm with a fixed query budget, started at the origin, therefore receives the same answers for every suficiently large $R ,$ so its queries and output stay in a bounded region that does not depend on $R .$ Taking R large, the output misses the mass of $\pi _ { R }$ . Appending Gaussian coordinates extends the example to any $d \geq 2$ . App. F gives the construction and the argument for randomized adaptive algorithms.

This lower bound concerns the information in the initialization, not the algorithm: it holds even with access to function values, and it is consistent with Thm. 4, which charges the initialization explicitly.

## 6 Conclusions and Discussions

In this paper, we showed that for smooth targets satisfying LSI or PI, the proximal sampler can be implemented with a number of gradient queries that grows more slowly than any power of the dimension at fixed condition number and accuracy. The implementation runs the observation chain with the noisy local sampler of [Ope26], controls the accumulated error by comparison with the exact chain, and removes the observation noise once at the end with a decoder. Under LSI, any supplied point can serve for initialization; under PI, the guarantee requires an initialization with a known bound on its Rényi divergence, and a lower bound shows that a stationary point alone does not sufice.

Several questions remain open. Our dependence on accuracy is a small power ε<sup>−ν</sup>, and it is unknown whether an arbitrarily small dimension exponent can be combined with polylogarithmic dependence on $\varepsilon ^ { - 1 }$ . Under PI, it would be useful to find initializations that need less information than a global minimizer, or that have a smaller divergence bound B, which enters our bound linearly. Finally, intermediate functional inequalities such as the Latała–Oleszkiewicz inequalities [LO00] may fit the same framework, given suitable convergence and gradient estimates.

AI usage. ChatGPT and Codex assisted with mathematical exploration and proof development, manuscript drafting and revision, literature searches, and generation of code and computational checks. The authors are responsible for the mathematical claims, citations, and final presentation.

## References

[AC24] Jason M. Altschuler and Sinho Chewi. Faster high-accuracy log-concave sampling via algorithmic warm starts. J. ACM, 71(3), June 2024.

[BGL14] Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and geometry of Markov difusion operators. Grundlehren der mathematischen Wissenschaften. Springer Cham, 2014.

[CCBJ18] Xiang Cheng, Niladri S. Chatterji, Peter L. Bartlett, and Michael I. Jordan. Underdamped Langevin MCMC: A non-asymptotic analysis. In Sébastien Bubeck, Vianney Perchet, and Philippe Rigollet, editors, Proceedings of the 31st Conference On Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 300–323. PMLR, 06–09 Jul 2018.

[CCLZ26a] Fan Chen, Sinho Chewi, Jianfeng Lu, and Matthew S Zhang. Accelerated high-accuracy sampling from a warm start via the proximal bouncy particle sampler. arXiv preprint arXiv:2609.06905, 2026.

[CCLZ26b] Fan Chen, Sinho Chewi, Jianfeng Lu, and Matthew S Zhang. High-accuracy simulation of Picard HMC, part I: Gaussian cloud correction. arXiv preprint arXiv:2609.38710, 2026.

[CCSW22] Yongxin Chen, Sinho Chewi, Adil Salim, and Andre Wibisono. Improved analysis for a proximal algorithm for sampling. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 2984–3014. PMLR, 02–05 Jul 2022.

[CEL<sup>+</sup>22] Sinho Chewi, Murat A Erdogdu, Mufan Li, Ruoqi Shen, and Shunshi Zhang. Analysis of Langevin Monte Carlo from Poincaré to log-Sobolev. In Po-Ling Loh and Maxim Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 1–2. PMLR, 02–05 Jul 2022.

[Che22] Sinho Chewi. Log-concave sampling. Book draft available at https: // chewisinho. github. io , 2022.

[FYC23] Jiaojiao Fan, Bo Yuan, and Yongxin Chen. Improved dimension dependence of a proximal algorithm for sampling. In Gergely Neu and Lorenzo Rosasco, editors, Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 1473–1521. PMLR, 12–15 Jul 2023.

[GLL22] Sivakanth Gopi, Yin Tat Lee, and Daogao Liu. Private convex optimization via exponential mechanism. In Proceedings of the Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 1948–1989, 2022.

[LC23] Jiaming Liang and Yongxin Chen. A proximal algorithm for sampling. Transactions on Machine Learning Research, 2023.

[LO00] R. Latała and K. Oleszkiewicz. Between sobolev and poincaré, pages 147–168. Springer Berlin Heidelberg, Berlin, Heidelberg, 2000.

[LST21] Yin Tat Lee, Ruoqi Shen, and Kevin Tian. Structured logconcave sampling with a restricted Gaussian oracle. In Mikhail Belkin and Samory Kpotufe, editors, Proceedings of Thirty Fourth Conference on Learning Theory, volume 134 of Proceedings of Machine Learning Research, pages 2993–3050. PMLR, 15–19 Aug 2021.

[Ope26] OpenAI. Subpolynomial query complexity for well-conditioned log-concave sampling. OpenAI Math Release preprint OAI:Subpolynomial-query-complexity-for-wellconditioned-log-concave-sampling-September-26-2026, 2026.

[OV00] F. Otto and C. Villani. Generalization of an inequality by Talagrand and links with the logarithmic Sobolev inequality. Journal of Functional Analysis, 173(2):361–400, 2000.

[SL19] Ruoqi Shen and Yin Tat Lee. The randomized midpoint method for log-concave sampling. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[VW19] Santosh Vempala and Andre Wibisono. Rapid convergence of the unadjusted Langevin algorithm: Isoperimetry sufices. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché- Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[YGC26] Bo Yuan, Wei Guo, and Yongxin Chen. Reflected proximal dynamics for accelerated high-accuracy log-concave sampling. arXiv preprint arXiv:2610.16248, 2026.

## A Related Work

Functional inequalities provide a way to analyze sampling beyond global log-concavity. [VW19] proved KL convergence guarantees for the unadjusted Langevin algorithm under LSI and smoothness, without convexity. [CEL<sup>+</sup>22] developed Rényi-divergence guarantees spanning PI and LSI. The proximal analysis in [CCSW22] separates global convergence from conditional implementation: exact proximal updates contract under the functional inequality, while the Gaussian quadratic makes each conditional strongly log-concave when the step size satisfies $\tau < 1 / \beta$ . Our analysis uses this separation to control implementation errors along exact histories.

Several approaches implement these conditionals. [GLL22] developed rejection sampling for Gaussianregularized Lipschitz potentials in private convex optimization. [LC23] treated weakly smooth, nonsmooth, and composite potentials through rejection-based proximal sampling. The approximate rejection sampler in [FYC23] improves dimension dependence through Gaussian concentration. A diferent approach combines underdamped Langevin warm starts with MALA [AC24], yielding highaccuracy conditional sampling and proximal guarantees beyond log-concavity. Our implementation instead uses the Gaussian-convolved output in [Ope26] to approximate the observation transition directly; it recovers a position sample only through terminal decoding.

Acceleration ofers another direction within the proximal framework. The Proximal Bouncy Particle Sampler [CCLZ26a] combines Gaussian augmentation with local particle dynamics and obtains $\widetilde { \cal O } ( \sqrt { \kappa } d ^ { 1 / \bar { 4 } } )$ expected gradient queries, with polylogarithmic inverse-accuracy dependence, for smooth strongly log-concave targets from an O(1) order-two Rényi start. Its curvature and initialization assumptions difer from the nonconvex LSI and PI settings considered here.

Gaussian cloud correction for Picard HMC [CCLZ26b, Thm. 1.1] also handles smooth nonconvex LSI targets: it uses ${ \cal \tilde { O } } ( \kappa d ^ { 1 / 5 } )$ expected gradient queries with polylogarithmic inverse-accuracy dependence, assuming $G _ { 0 } \leq d / \kappa$ . Our bounds permit any fixed positive dimension exponent, have deterministic query caps, and charge arbitrary $G _ { 0 }$ , but have weaker accuracy dependence. Neither guarantee uniformly dominates.

## B Auxiliary estimates and local sampling identities

## B.1 How the noisy local sampler works

This appendix makes the mechanism behind Prop. 5 self-contained at the algorithmic level. The construction has three ingredients: a Gaussian probability flow that reduces sampling to conditional means, a second stationary flow that produces noisy estimates of those means, and a finite recursive implementation of both flows using gradient queries. The quantitative derivative estimates, recursive error analysis, and query bound are proved in [Ope26, Sections 3–7] and are not repeated here.

Gaussian transport reduces sampling to conditional means. For independent $Z \sim \rho _ { c , \prime } ^ { F }$ and $G \sim { \mathcal { N } } ( 0 , I _ { n } )$ , consider the interpolation $Y _ { s } = s Z + \sqrt { 1 - s ^ { 2 } } G , 0 \leq s < 1$ , and define

$$
m _ { s } ( y ) = \mathbb { E } [ \nabla F ( c + r Z ) \mid Y _ { s } = y ] .
$$

Completing the square shows that this is another conditional mean of the same type, at a smaller scale:

$$
\begin{array} { r } { m _ { s } ( y ) = \mathbb { E } _ { Q \sim \rho _ { c + r s y , r \sqrt { 1 - s ^ { 2 } } } ^ { F } } \nabla F ( c + r s y + r \sqrt { 1 - s ^ { 2 } } Q ) . } \end{array}\tag{21}
$$

The deterministic probability flow

$$
\frac { \mathrm { d } \overline { { Y } } _ { s } } { \mathrm { d } s } = - r m _ { s } ( \overline { { Y } } _ { s } ) , \qquad \overline { { Y } } _ { 0 } \sim { \mathcal N } ( 0 , I _ { n } ) ,\tag{22}
$$

satisfies $\mathrm { L a w } ( \overline { { Y } } _ { s } ) = \mathrm { L a w } ( Y _ { s } )$ . This is equality of marginal laws: the interpolating process $Y _ { s }$ does not itself solve the deterministic flow equation. In particular, stopping at $s = 1 / \sqrt { 2 }$ and rescaling by $\sqrt { 2 }$ gives the law of $Z + G$ . Gaussian conditioning makes the velocity smooth even though $F$ has only two derivatives. The analysis in [Ope26] controls its higher derivatives with dimension dependence compatible with high-order approximation.

Conditional velocity. For the interpolation $Y _ { s } = s Z + { \sqrt { 1 - s ^ { 2 } } } G$ above, the conditional density gives

$$
\mathbb { E } [ Z \mid Y _ { s } = y ] = s y - ( 1 - s ^ { 2 } ) r m _ { s } ( y )
$$

by integration by parts. Since the derivative of the interpolation is $( Z - s Y _ { s } ) / ( 1 - s ^ { 2 } )$ , its conditional expectation given $Y _ { s } = y \ \mathrm { i s } \ - r m _ { s } ( y )$ . Diferentiating the expectation of a compactly supported smooth test function therefore gives the continuity equation with that velocity. The globally Lipschitz flow in (22) has the same marginals; [Ope26, Proposition 3.3] establishes the required regularity under the stated $C ^ { 2 }$ assumption. Completing the conditional square proves (21).

Finite recursion replaces the unknown means by gradient queries. The flow is discretized by polynomial quadrature and a fixed number of Picard iterations. Here a Picard iteration substitutes the preceding trajectory approximation into the velocity and integrates; repeated substitution gives a finite expression involving conditional means at other centers. These means are not available as oracles. The construction therefore introduces a second algorithm targeting

$$
\mathbb { E } _ { \rho _ { c , r } ^ { F } } \nabla F ( c + r Z ) + \sigma G ,
$$

for a prescribed $\sigma > 0$ and independent standard Gaussian $G ,$ , using a stationary flow with an auxiliary Gaussian momentum. Discretizing that flow produces further conditional-mean expressions. The sampling and mean algorithms call each other with prescribed error orders and perturbation scales until simple gradient-based approximations sufice. For example, a terminal sampling call has the form

$$
G ^ { ( 0 ) } - r \nabla F ( c + r G ^ { ( 0 ) } ) + \frac { \sqrt { 3 } \sigma _ { \mathrm { a u x } } } { 2 } G ^ { ( 1 ) } + \frac { \sigma _ { \mathrm { a u x } } } { 2 } G ^ { ( 2 ) } ,
$$

where $\sigma _ { \mathrm { a u x } } > 0$ is the requested Gaussian noise standard deviation, distinct from the proximal step size h and the initial density $\eta _ { 0 }$ . With independent standard Gaussian vectors, this approximates $\rho _ { c , r } ^ { F } * \mathcal { N } ( 0 , \sigma _ { \mathrm { a u x } } ^ { 2 } I _ { n } )$ at a terminal accuracy level. Nonterminal calls use the higher-order flow formulas. The recursion depth is fixed before the quadrature mesh is selected. A suficiently small mesh exponent, followed by a suficiently high polynomial approximation order, makes the total query exponent as small as prescribed. Gaussian derivative estimates control the dimension cost of raising that order. The entire finite tree satisfies the deterministic query cap in Prop. 5, with constants depending on $\gamma , K , \omega$

Absorbing a noisy center. Suppose a seeded mean algorithm $Q ( c ; g )$ , with M vector-valued seed slots, has the deterministic translation identity

$$
Q ( c + \Delta ; g + v \otimes \Delta / r ) = Q ( c ; g ) ,
$$

and preserves every physical gradient-query location in that identity. Here $v \in \mathbb { R } ^ { M }$ is fixed, $L > 0$ ， and $\lVert \boldsymbol { v } \rVert \leq L$ . Put

$$
b = { r } / ( { 2 L } ) , \qquad P = ( I _ { M } - { v v ^ { \operatorname { T } } } / ( { 4 L ^ { 2 } } ) ) ^ { 1 / 2 } .
$$

For independent standard Gaussian seed slots G and a standard Gaussian vector $Z ,$ the call $Q ( c + b Z ; ( P \otimes I _ { n } ) G )$ has exactly the law of $Q ( c ; G )$ . Indeed, the translation identity rewrites its seed array at center c as $P G - v \otimes Z / ( 2 L )$ , with the tensor identity suppressed. This array is centered Gaussian with slot covariance

$$
P P ^ { \mathrm { T } } + v v ^ { \mathrm { T } } / ( 4 L ^ { 2 } ) = I _ { M } .
$$

Its slots are therefore independent standard Gaussians. The translation property is maintained recursively; the covariance identity itself is exact. Approximation errors in supplied centers are handled separately by fixed-seed stability, not by assuming the actual centers already have the ideal law.

Reserved smoothing noise. Let $W _ { 2 } ( \mu , \eta )$ be the Wasserstein-2 distance, the infimum of $( \mathbb { E } \Vert X -$ $X ^ { \prime } \| ^ { 2 } ) ^ { 1 / 2 }$ over couplings of $X \sim \mu$ and $X ^ { \prime } \sim \eta$ . The noisy local sampler reserves independent Gaussian noise until all queries are complete. More precisely, the output has the form $U _ { \mathrm { p r e } } + G / 2$ where G is independent of $U _ { \mathrm { p r e } }$ and all queries, and

$$
\begin{array} { r } { W _ { 2 } ( \mathrm { L a w } ( U _ { \mathrm { p r e } } ) , \rho _ { c , r } ^ { F } * \mathcal { N } ( 0 , \frac { 3 } { 4 } I _ { n } ) ) \le A _ { n } \delta ^ { K } ( \sqrt { n } + \| \nabla F ( c ) \| ) . } \end{array}\tag{23}
$$

For any two laws with finite second moments and any $\sigma > 0$ , a coupling and the Gaussian shift bound give

$$
\mathrm { T V } \big ( \mu \ast \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { n } ) , \eta \ast \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { n } ) \big ) \leq \frac { W _ { 2 } ( \mu , \eta ) } { 2 \sigma } .
$$

Apply this with $\sigma = 1 / 2$ and the pre-reserve laws in (23). Adding the independent reserve raises the Gaussian covariance from $3 I _ { n } / 4$ to $I _ { n } ,$ yielding the TV guarantee.

## B.2 Normalization and Gaussian padding

Lemma 12 (Coordinate and oracle accounting). Let $T _ { a } , H _ { a } , \widetilde { V }$ , Π be as in (9) to (10), with $n \geq d$ Then ∇Ve is 1-Lipschitz, and α-LSI (respectively PI) for π implies $1 / \kappa \ – L S I$ (respectively PI) for Π. For any law λ on $\mathbb { R } ^ { d } .$ , set $\overline { { \lambda } } = ( T _ { a } ) _ { \# } \lambda \otimes \mathcal { N } ( 0 , I _ { n - d } )$ . Its KL and $D _ { 2 }$ divergences from Π equal those of λ from π. Moreover, $H _ { a \# } \Pi = \pi$ and

$$
\mathrm { T V } ( H _ { a \# } \eta , \pi ) \le \mathrm { T V } ( \eta , \Pi )
$$

for every law η on $\mathbb { R } ^ { n }$ . One gradient query to $\widetilde { V }$ costs one gradient query to $V .$

Proof. The normalized Hessian is $\beta ^ { - 1 } \nabla ^ { 2 } V$ , and the padding block is the identity. A change of variables in either functional inequality gives constant $\alpha / \beta = 1 / \kappa$ for $( T _ { a } ) _ { \# } \pi$ Since $\alpha \leq \beta .$ tensorization with the standard Gaussian retains this constant. Under the invertible map $T _ { a } .$ , density ratios are carried along unchanged. Adding the same independent Gaussian to both laws leaves their $\mathrm { K L }$ and $D _ { 2 }$ integrals unchanged as well. Projection and inverse rescaling give $H _ { a \# } \Pi = \pi$ , and data processing gives the TV bound. Finally, at $w = ( z , \xi )$

$$
\nabla \widetilde { V } ( w ) = ( \beta ^ { - 1 / 2 } \nabla V ( a + z / \sqrt { \beta } ) , \xi ) ,
$$

so the gradient of $\widetilde { V }$ follows from a single gradient query at $a + z / \sqrt { \beta }$

## B.3 Stationary score control

Lemma 13 (Stationary score identity). $I f \widetilde { V } \in C ^ { 2 } ( \mathbb { R } ^ { n } )$ has 1-Lipschitz gradient and $\mathrm { e } ^ { - \widetilde { V } }$ is integrable, then

$$
\mathbb { E } _ { \Pi } \lVert \nabla \tilde { V } \rVert ^ { 2 } = \mathbb { E } _ { \Pi } \Delta \tilde { V } \leq n .\tag{24}
$$

Proof. Choose smooth compactly supported cutofs $\chi _ { R }$ tending pointwise to one, with $0 \leq \chi _ { R } \leq 1$ and $\| \nabla \chi _ { R } \| \le C / R$ . Integration by parts gives

$$
\begin{array} { r } { I _ { R } : = \mathbb { E } _ { \Pi } \chi _ { R } ^ { 2 } \| \nabla \widetilde { V } \| ^ { 2 } = \mathbb { E } _ { \Pi } \chi _ { R } ^ { 2 } \Delta \widetilde { V } + 2 \mathbb { E } _ { \Pi } \chi _ { R } \nabla \chi _ { R } \cdot \nabla \widetilde { V } . } \end{array}
$$

Since $| \Delta \tilde { V } | \leq n$ , Young’s inequality gives $I _ { R } \leq n + I _ { R } / 2 + C / R ^ { 2 }$ . Fatou’s lemma proves square integrability of the score. The cross term then tends to zero by Cauchy–Schwarz. Dominated convergence for the bounded Hessian trace and the integrable squared score proves (24). □

## B.4 Local estimates used by the decoder

We also use the following two local consequences from [Ope26, Lemma 2.2, Eq. (2.6), and Lemma 8.4]. For $\mathrm { L i p } ( \nabla F ) \leq \delta$ and $\delta r ^ { 2 } \le 1 / 2$ , if $Z \sim \rho _ { c , r } ^ { F }$ and G is an independent standard Gaussian, then

$$
( \mathbb { E } \| Z + G \| ^ { 2 } ) ^ { 1 / 2 } \leq C ( { \sqrt { n } } + r \| \nabla F ( c ) \| ) .\tag{25}
$$

If $v _ { * }$ is the mode of $\rho _ { c , r } ^ { F }$ , then

$$
\mathrm { T V } ( \rho _ { c , r } ^ { F } , \mathcal { N } ( v _ { * } , I _ { n } ) ) \leq C \delta r ^ { 2 } \sqrt { n } .\tag{26}
$$

These are local strongly log-concave estimates and remain valid when the global target is non-logconcave.

## B.5 Proof of the decoder guarantee

Proof. Let $Y = X + { \sqrt { h } } G$ with X ∼ Π and independent standard Gaussian G. By Lem. 13 and Lipschitz continuity,

$$
\begin{array} { r } { \mathbb { E } \| \nabla \tilde { V } ( Y ) \| ^ { 2 } \leq 2 n + 2 h n \leq 3 n . } \end{array}
$$

Exact-history moments. Define a separate exact process by $c _ { 0 } ^ { \star } = 0 , r _ { j } = 2 ^ { - j / 2 }$ , and

$$
U _ { j } ^ { \star } \mid ( Y , c _ { j } ^ { \star } ) \sim \rho _ { c _ { j } ^ { \star } , r _ { j } } ^ { F _ { Y } } \ast \mathcal { N } ( 0 , I _ { n } ) , \qquad c _ { j + 1 } ^ { \star } = c _ { j } ^ { \star } + r _ { j } U _ { j } ^ { \star } / 2 .
$$

These are exact draws, not calls to NoisySample. Put $b _ { j } = ( \mathbb { E } \| g _ { Y } ( c _ { j } ^ { \star } ) \| ^ { 2 } ) ^ { 1 / 2 }$ , where expectation includes $Y \sim \Pi _ { h }$ and the exact history. Initially $b _ { 0 } \leq \sqrt { 3 h n }$ . Conditional on $Y , { c } _ { j } ^ { \star }$ , (25) applies directly:

$$
( { \mathbb E } [ \| U _ { j } ^ { \star } \| ^ { 2 } \mid Y , c _ { j } ^ { \star } ] ) ^ { 1 / 2 } \leq C ( \sqrt { n } + r _ { j } \| g _ { Y } ( c _ { j } ^ { \star } ) \| ) .
$$

The exact center update, $\mathrm { L i p } ( g _ { Y } ) \leq \delta$ , and the $L ^ { 2 }$ triangle inequality imply

$$
b _ { j + 1 } \leq ( 1 + C \delta r _ { j } ^ { 2 } ) b _ { j } + C \delta r _ { j } \sqrt { n } .
$$

Both $\textstyle \sum _ { j } r _ { j }$ and $\textstyle \sum _ { j } r _ { j } ^ { 2 }$ are bounded by absolute constants, so iteration gives $b _ { j } \leq C \sqrt { n }$ uniformly in $j .$

Implementation error. Let $\mathcal { H } _ { j }$ be the output law, averaged over $Y \sim \Pi _ { h }$ , obtained with the first $j$ observations exact and the remaining $J - j$ observations implemented by NoisySample. Every hybrid uses the same terminal Gaussian rule (13) to (14). Thus $\mathcal { H } _ { 0 } = \Pi _ { h } \widehat { R } _ { h }$ , while $\mathcal { H } _ { J }$ has an entirely exact observation history. Consecutive hybrids share the exact prefix $( Y , c _ { j } ^ { \star } )$ . Conditional on this prefix, (8) bounds the difering observation; all subsequent computation is a common Markov kernel. Hence

$$
\mathrm { T V } ( \mathcal { H } _ { j } , \mathcal { H } _ { j + 1 } ) \leq A _ { n } \delta ^ { K } \mathbb { E } [ \sqrt { n } + \| g _ { Y } ( c _ { j } ^ { \star } ) \| ] \leq C A _ { n } \delta ^ { K } \sqrt { n } .
$$

Summing yields

$$
\begin{array} { r } { \mathrm { T V } ( \mathcal { H } _ { 0 } , \mathcal { H } _ { J } ) \leq C J A _ { n } \delta ^ { K } \sqrt { n } = : E _ { \mathrm { d e s c e n t } } . } \end{array}\tag{27}
$$

No moment bound for the implemented histories is used.

Terminal approximation. Under the exact history, let $v _ { * } ^ { \star }$ be the mode of $\rho _ { c _ { I } ^ { \star } , r _ { J } } ^ { F _ { Y } }$ and let $v ^ { ( m ) , \star }$ denote the fixed-point iterate at that center. The mode satisfies $v _ { * } ^ { \star } = - r _ { J } g _ { Y } \lvert \dot { c } _ { J } ^ { \star } + r _ { J } v _ { * } ^ { \star } \rvert$ . This map is l-contractive, where $l = \delta r _ { J } ^ { 2 } \leq \delta n ^ { - 4 } \leq 1 / 2$ , so

$$
\| v ^ { ( m ) , \star } - v _ { * } ^ { \star } \| \leq l ^ { m } \| v _ { * } ^ { \star } \| \leq \frac { l ^ { m } r _ { J } \| g _ { Y } ( c _ { J } ^ { \star } ) \| } { 1 - l } .
$$

By (26), the Gaussian shift inequality, and $b J \leq C { \sqrt { n } }$ , the averaged error in replacing this last conditional by the terminal Gaussian is at most

$$
C \delta r _ { J } ^ { 2 } \sqrt { n } + C \frac { ( \delta r _ { J } ^ { 2 } ) ^ { m } r _ { J } \sqrt { n } } { 1 - \delta r _ { J } ^ { 2 } } \le C \delta n ^ { - 7 / 2 } = : E _ { \mathrm { b a s e } } .
$$

The last inequality uses $m \geq 1$ and $r _ { J } \leq 1$ . Conditional afine reconstruction contracts TV. With an exact terminal draw, repeated use of (12) reconstructs $R _ { h } ( \cdot \mid Y )$ , whose average under $\Pi _ { h }$ is Π. Thus $\mathrm { T V } ( \mathcal { H } _ { J } , \Pi ) \leq E _ { \mathrm { b a s e } }$ , and the triangle inequality proves (15). The cost is J calls to NoisySample and m gradient iterations. □

## B.6 The bounded-perturbation example

The LSI comparison used in Sec. 4 follows from

$$
\operatorname { E n t } _ { \mu } ( f ^ { 2 } ) = \operatorname* { i n f } _ { a > 0 } \mathbb { E } _ { \mu } [ f ^ { 2 } \log ( f ^ { 2 } / a ) - f ^ { 2 } + a ] .
$$

The integrand is nonnegative. If $m \le \mu / \Gamma \le M$ and Γ is standard Gaussian, its LSI gives $\mathrm { E n t } _ { \mu } ( f ^ { 2 } ) \leq M \mathrm { E n t } _ { \Gamma } ( f ^ { 2 } ) \leq 2 ( M / m ) \mathbb { E } _ { \mu } \| \nabla f \| ^ { 2 }$ . For the perturbation $\mathrm { 2 \cos { x _ { 1 } , \ M / m } \ \leq \ e ^ { 4 } }$ . Its Hessian eigenvalues lie between −1 and 3, proving the example’s other claims. Direct Gaussian rejection uses acceptance $\mathrm { e } ^ { - 2 - 2 \cos x _ { 1 } }$ ; this additional structure explains why the example is not a complexity-separation witness.

## C Proofs for the LSI analysis

Throughout, Π has LSI constant $1 / \kappa$ (Lem. 12), and since $a = x _ { 0 }$ , we have $\lVert \nabla \widetilde { V } ( 0 ) \rVert ^ { 2 } = G _ { 0 }$ Lemma 14 (Gaussian initial entropy). For $\lambda _ { 0 } = { \mathcal { N } } ( 0 , I _ { n } )$

$$
\begin{array} { r } { \mathrm { K L } ( \lambda _ { 0 } \| \Pi ) \le \kappa ( G _ { 0 } + 4 n ) . } \end{array}\tag{28}
$$

Proof. For $Z \sim { \mathcal { N } } ( 0 , I _ { n } )$ , the relative score is $\nabla \log ( \lambda _ { 0 } / \Pi ) ( Z ) = \nabla \tilde { V } ( Z ) - Z$ . Since $\boldsymbol { \nabla } \tilde { V }$ is 1- Lipschitz,

$$
\lVert \nabla \tilde { V } ( Z ) - Z \rVert \leq \lVert \nabla \tilde { V } ( 0 ) \rVert + 2 \lVert Z \rVert ,
$$

so $\mathbb { E } _ { \lambda _ { 0 } } \| \nabla \log ( \lambda _ { 0 } / \Pi ) \| ^ { 2 } \le 2 G _ { 0 } + 8 n$ . Apply (LSI), with constant $1 / \kappa$ , to $f = ( \lambda _ { 0 } / \Pi ) ^ { 1 / 2 }$ . Then $\mathrm { E n t } _ { \Pi } ( f ^ { 2 } ) = \mathrm { K L } ( \lambda _ { 0 } \| \Pi )$ and $4 \mathbb { E } _ { \Pi } \| \nabla f \| ^ { 2 } = \mathbb { E } _ { \lambda _ { 0 } } \| \nabla \log ( \lambda _ { 0 } / \Pi ) \| ^ { 2 }$ , so

$$
\mathrm { K L } ( \lambda _ { 0 } \| \Pi ) \le \frac { 2 \kappa } { 4 } ( 2 G _ { 0 } + 8 n ) = \kappa ( G _ { 0 } + 4 n ) .
$$

The last expectation is finite, so $f$ is an admissible test function; alternatively, apply (LSI) to truncations and pass to the limit. □

The same argument without padding $( n = d )$ shows that the initialization in the original coordinates satisfies $\mathrm { K L } ( \mathcal { N } ( x _ { 0 } , \beta ^ { - 1 } I _ { d } ) \| \pi ) \le \kappa ( G _ { 0 } + 4 d )$

Lemma 15 (Gradient moments from relative entropy). $I f \operatorname { K L } ( \lambda \| \Pi ) \leq b$ , then

$$
\begin{array} { r } { \mathbb { E } _ { \lambda } \| \nabla \tilde { V } ( X ) \| ^ { 2 } \leq 2 n + 4 \kappa b . } \end{array}\tag{29}
$$

$I f X \sim \lambda , G \sim { \mathcal { N } } ( 0 , I _ { n } )$ is independent, and $Y = X + { \sqrt { h } } G$ with $h \leq 1 / 2$ , then

$$
\mathbb { E } \| \nabla \widetilde { V } ( Y ) \| ^ { 2 } \leq 5 n + 8 \kappa b .\tag{30}
$$

Proof. Since Π satisfies LSI with constant $1 / \kappa ,$ it satisfies Talagrand’s transport inequality [OV00]

$$
W _ { 2 } ^ { 2 } ( \lambda , \Pi ) \leq 2 \kappa \mathrm { K L } ( \lambda \| \Pi ) .\tag{31}
$$

Couple $X \sim \lambda$ to $X _ { * } \sim \Pi$ . Since $\boldsymbol { \nabla } \tilde { V }$ is 1-Lipschitz,

$$
\begin{array} { r } { \mathbb { E } \| \nabla \widetilde { V } ( X ) \| ^ { 2 } \leq 2 \mathbb { E } \| \nabla \widetilde { V } ( X _ { * } ) \| ^ { 2 } + 2 \mathbb { E } \| X - X _ { * } \| ^ { 2 } . } \end{array}
$$

Taking the infimum over couplings, the first term is at most 2n by Lem. 13, and the second is at most $2 W _ { 2 } ^ { 2 } ( \lambda , \Pi ) \leq 4 \kappa b$ by (31). For Y, the same argument gives

$$
\begin{array} { r } { \mathbb { E } \| \nabla \widetilde { V } ( Y ) \| ^ { 2 } \leq 2 \mathbb { E } \| \nabla \widetilde { V } ( X ) \| ^ { 2 } + 2 h \mathbb { E } \| G \| ^ { 2 } \leq 4 n + 8 \kappa b + 2 h n \leq 5 n + 8 \kappa b . } \end{array}
$$

Proof of Prop. 9. Set $B _ { 0 } = 5 \kappa n$ . Since $G _ { 0 } \leq n$ , Lem. 14 gives $\mathrm { K L } ( \lambda _ { 0 } \| \Pi ) \le B _ { 0 }$ . By Prop. 2, applied to Π with LSI constant $1 / \kappa$ and step size $h ,$

$$
\begin{array} { r } { \mathrm { K L } ( \lambda _ { j } | | \Pi ) \le ( 1 + h / \kappa ) ^ { - 2 j } B _ { 0 } \qquad \mathrm { f o r ~ a l l ~ } j \ge 0 . } \end{array}
$$

Choose

$$
N = \lceil \frac { \log ( 8 B _ { 0 } / \varepsilon ^ { 2 } ) } { 2 \log ( 1 + h / \kappa ) } \rceil \leq C \kappa \delta ^ { - 1 } \log n ,\tag{32}
$$

where the inequality uses $\log ( 1 + h / \kappa ) \geq h / ( 2 \kappa )$ and $\kappa , \varepsilon ^ { - 1 } \leq n$ . Then $\mathrm { K L } ( \lambda _ { N } \| \Pi ) \le \varepsilon ^ { 2 } / 8$ . Since $\mu _ { N } = \lambda _ { N } A _ { h }$ and $\Pi _ { h } = \Pi A _ { h }$ , data processing and Pinsker’s inequality give $\varepsilon _ { \mathrm { m i x } } = \mathrm { T V } ( \mu _ { N } , \Pi _ { h } ) \leq$ $\mathrm { T V } ( \lambda _ { N } , \Pi ) \le \varepsilon / 4$ . For the score budget, every $\lambda _ { j }$ has divergence at most $B _ { 0 }$ , and $\mu _ { j }$ is the law of $X + { \sqrt { h } } G$ with $X \sim \lambda _ { j }$ . Hence Lem. 15 gives $\mathbb { E } _ { \mu _ { i } } \| \nabla \tilde { V } ( Y ) \| ^ { 2 } \leq 5 n + 4 0 \kappa ^ { 2 } n \leq 4 5 \tilde { \kappa } ^ { 2 } n$ , and by Cauchy–Schwarz each summand of $M _ { N }$ is at most $C \kappa \sqrt { n }$ . Therefore

$$
M _ { N } \leq C N \kappa \sqrt { n } , \qquad A _ { n } \delta ^ { K } M _ { N } \leq C \kappa ^ { 2 } A _ { n } \delta ^ { K - 1 } \sqrt { n } \log n .\tag{33}
$$

Proof of Thm. 3. By (18), $\delta ^ { - 1 } = n ^ { \gamma }$ , and (32), the algorithm makes at most $C \kappa A _ { n } ( \log n ) n ^ { \gamma ( 1 + \omega ) }$ queries after computing $G _ { 0 }$ . With n as in (46) and the parameter choices of App. G, the implementation error (33) and the decoder error (15) are each at most $\varepsilon / 4$ , so Props. 8 and 9 bound the padded output error by ε, and the query count is of the form (4). Finally, for the law $\widehat { \pi }$ returned by Alg. 1, Lem. 12 transfers the error back to the original target:

$$
\mathrm { T V } ( \widehat { \pi } , \pi ) = \mathrm { T V } \big ( H _ { a \# } ( \widehat { \mu } _ { N } \widehat { R } _ { h } ) , H _ { a \# } \Pi \big ) \leq \mathrm { T V } ( \widehat { \mu } _ { N } \widehat { R } _ { h } , \Pi ) \leq \varepsilon .\tag{34}
$$

## D Proofs for the PI analysis

Throughout, Π has PI constant $1 / \kappa \ ( \mathrm { L e m . \ 1 2 } )$

Lemma 16 (A PI exponential-moment bound). If Π satisfies $1 / \kappa \ – P I$ and F is integrable and 1-Lipschitz, then

$$
\log \mathbb { E } _ { \Pi } \mathrm { e } ^ { ( F - \mathbb { E } _ { \Pi } F ) / \sqrt { \kappa } } \leq 1 .\tag{35}
$$

Proof. For bounded F, write $M ( s ) = \mathbb { E } \mathrm { e } ^ { s F }$ . Applying PI to $\mathrm { e } ^ { s F / 2 }$ gives

$$
M ( s ) - M ( s / 2 ) ^ { 2 } \leq { \frac { \kappa s ^ { 2 } } { 4 } } M ( s ) .
$$

Set $s = \kappa ^ { - 1 / 2 }$ and iterate the corresponding logarithmic inequality at $s , s / 2 , s / 4 , \dots$ . After subtracting the centered linear term,

$$
\log M ( s ) - s \mathbb { E } F \leq \sum _ { j = 0 } ^ { \infty } 2 ^ { j } \left[ - \log \left( 1 - { \frac { 1 } { 4 \cdot 4 ^ { j } } } \right) \right] \leq { \frac { 1 } { 2 } } \sum _ { j = 0 } ^ { \infty } 2 ^ { - j } = 1 .
$$

The omitted limiting centered term is zero because F is bounded. The second inequality uses $- \log ( 1 - u ) \leq 2 u$ for $u \leq 1 / 4$ . Apply this result to Lipschitz truncations of $F .$ Their means converge by integrability, and Fatou’s lemma gives the claim. Weak-gradient approximation justifies the test functions when F is only Lipschitz. □

Lemma 17 (PI change of measure for the score). $I f \operatorname { K L } ( \lambda \| \Pi ) \leq B$ , then

$$
\mathbb { E } _ { \lambda } \| \nabla \widetilde { V } ( X ) \| \le \sqrt { n } + \sqrt { \kappa } ( B + 1 ) .\tag{36}
$$

For $X \sim \lambda$ and $Y = X + { \sqrt { h } } G$ with independent $G \sim { \mathcal { N } } ( 0 , I _ { n } )$ ，

$$
\mathbb { E } \| \nabla \widetilde { V } ( Y ) \| \le ( 1 + \sqrt { h } ) \sqrt { n } + \sqrt { \kappa } ( B + 1 ) .\tag{37}
$$

Proof. The function $F = \| \nabla \widetilde { V } \|$ is 1-Lipschitz and $\mathbb { E } _ { \Pi } F \le \sqrt { n }$ by Lem. 13. The entropy variational inequality and Lem. 16, with $s = \kappa ^ { - 1 / 2 }$ , give

$$
\begin{array} { r } { s \mathbb { E } _ { \lambda } F \le \mathrm { K L } ( \lambda \| \Pi ) + \log \mathbb { E } _ { \Pi } \mathrm { e } ^ { s F } \le B + s \mathbb { E } _ { \Pi } F + 1 . } \end{array}
$$

Use nonnegative truncations of F first if its integrability under λ has not yet been established. This proves (36). The Gaussian observation adds at most ${ \sqrt { h } } \mathbb { E } \| G \| \leq { \sqrt { h n } }$ by Lipschitz continuity.

Proof of Prop. 10. By Lem. 12, normalizing the initialization and appending independent Gaussian coordinates preserves its $D _ { 2 }$ divergence, so $\chi ^ { 2 } ( \lambda _ { 0 } \| \Pi ) \leq \mathrm { e } ^ { B } - 1$ and $\begin{array} { r } { \mathrm { K L } ( \lambda _ { 0 } \| \Pi ) \le B } \end{array}$ . By Prop. 2, applied to Π with PI constant $1 / \kappa$ and step size $h ,$ together with data processing under $A _ { h }$

$$
\chi ^ { 2 } ( \mu _ { j } \| \Pi _ { h } ) \le ( \mathrm { e } ^ { B } - 1 ) ( 1 + h / \kappa ) ^ { - 2 j } , \qquad \mathrm { K L } ( \lambda _ { j } \| \Pi ) \le B .
$$

Choose

$$
N = \lceil \frac { B + 2 \log ( 2 / \varepsilon ) } { 2 \log ( 1 + h / \kappa ) } \rceil \leq C \kappa H \delta ^ { - 1 } .\tag{38}
$$

Then $\chi ^ { 2 } ( \mu _ { N } \vert \vert \Pi _ { h } ) \leq \varepsilon ^ { 2 } / 4$ , and $4 \mathrm { T V } ^ { 2 } \leq \chi ^ { 2 }$ gives $\varepsilon _ { \mathrm { m i x } } \leq \varepsilon / 4$ . For the score budget, $\mu _ { j }$ is the law of $X + { \sqrt { h } } G$ with $X \sim \lambda _ { j }$ , and ${ \sqrt { \kappa } } ( B + 1 ) \leq { \sqrt { n } }$ . Hence Lem. 17 gives $\mathbb { E } _ { \mu _ { j } } \| \nabla \tilde { V } ( Y ) \| \overset {  } { \leq } 3 \sqrt { n }$ , and therefore

$$
M _ { N } \leq C N \sqrt { n } , \qquad A _ { n } \delta ^ { K } M _ { N } \leq C \kappa H A _ { n } \delta ^ { K - 1 } \sqrt { n } .
$$

Proof of Thm. 4. The decoder bound is the same as under LSI. With n as in (47) and the parameter choices of $\mathrm { A p p . ~ G }$ , the implementation and decoder errors are each at most $\varepsilon / 4$ , so Props. 8 and 10 bound the padded output error by ε, and (48) gives a query count of the form (5). Applying (34) with $a = 0$ proves the original-space guarantee. □

## E Constants and Gaussian initialization under PI

## E.1 Functional-inequality constants and smoothness

Lemma 18. $I f \nabla V$ is $\beta - L i p s c h i t z$ and $\pi \propto \operatorname { e } ^ { - V }$ satisfies $\alpha { - } P I ,$ then $\alpha \le \beta$

Proof. Fix a unit vector $e \in \mathbb { R } ^ { d }$ , let $X \sim \pi$ , and write $m = \mathbb { E } \langle e , X \rangle$

Since $\nabla \mathrm { e } ^ { - V } = - \nabla V \mathrm { e } ^ { - V }$ , every $C ^ { 1 }$ function $g \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ such that g and $\nabla g$ grow at most linearly satisfies

$$
\begin{array} { r } { \mathbb { E } \langle e , \nabla g ( X ) \rangle = \mathbb { E } [ g ( X ) \langle e , \nabla V ( X ) \rangle ] , } \end{array}\tag{39}
$$

by applying the divergence theorem to the function $g \mathrm { e } ^ { - V } e$

Taking $g \equiv 1$ in (39) gives $\mathbb { E } \langle e , \nabla V ( X ) \rangle = 0$ . Taking $g ( x ) = \langle e , x \rangle$ gives $\mathbb { E } [ \langle e , X \rangle \langle e , \nabla V ( X ) \rangle ] =$ $\| e \| ^ { 2 } = 1 ;$ ; subtracting m $\mathbb { E } \langle e , \nabla V ( X ) \rangle = 0$ yields

$$
\mathbb { E } [ ( \langle e , X \rangle - m ) \langle e , \nabla V ( X ) \rangle ] = 1 .
$$

Taking $\boldsymbol { g } ( \boldsymbol { x } ) = \langle \boldsymbol { e } , \nabla V ( \boldsymbol { x } ) \rangle$ and using $\nabla ^ { 2 } V \preceq \beta I _ { d }$ gives

$$
\begin{array} { r } { \mathbb { E } [ \langle e , \nabla V ( X ) \rangle ^ { 2 } ] = \mathbb { E } \langle e , \nabla ^ { 2 } V ( X ) e \rangle \leq \beta . } \end{array}
$$

By the Cauchy–Schwarz inequality,

$$
\begin{array} { r } { 1 = ( \mathbb { E } [ ( \langle c , X \rangle - m ) \langle c , \nabla V ( X ) \rangle ] ) ^ { 2 } \leq \mathrm { V a r } \langle c , X \rangle \mathbb { E } [ \langle c , \nabla V ( X ) \rangle ^ { 2 } ] \leq \beta \mathrm { V a r } \langle c , X \rangle \implies \mathrm { V a r } \langle c , X \rangle \geq 1 / \beta . } \end{array}
$$

On the other hand, applying (PI) to $f ( x ) = \langle e , x \rangle { \mathrm { ~ g i v e s ~ V a r } } \langle e , X \rangle \leq 1 / \alpha$ . Combining the two bounds completes the proof. □

## E.2 Gaussian initialization near a global minimizer

Proposition 19 (Gaussian divergence near a global minimizer). Suppose π satisfies $\alpha { - } P I$ and a global minimizer x of V exists. For a point x and a known $A _ { 0 } \geq V ( x _ { 0 } ) - V ( x _ { * } )$ , let

$$
q _ { 0 } = { \mathcal N } ( x _ { 0 } , ( 2 \beta ) ^ { - 1 } I _ { d } ) , \qquad G _ { 0 } = \| \nabla V ( x _ { 0 } ) \| ^ { 2 } / \beta .
$$

Then

$$
D _ { 2 } ( q _ { 0 } \| \pi ) \le D _ { \infty } ( q _ { 0 } \| \pi ) \le A _ { 0 } + G _ { 0 } / 2 + \frac { d } { 2 } \log ( 2 \mathrm { e } \kappa ) \le 2 A _ { 0 } + \frac { d } { 2 } \log ( 2 \mathrm { e } \kappa ) .\tag{40}
$$

In particular, at $x _ { 0 } = x _ { * }$ the first two terms vanish.

Proof. For this proof, write $\begin{array} { r } { Z _ { V } = \int _ { \mathbb { R } ^ { d } } \mathrm { e } ^ { - V ( x ) } \mathrm { d } x } \end{array}$ . First, PI implies $ { \mathrm { C o v } } _ { \pi } X \preceq \alpha ^ { - 1 } I _ { d }$ . One can justify finite second moments before applying PI to a coordinate as follows. Truncate a coordinate centered at a median to $[ - R , R ]$ . The truncation is 1-Lipschitz and retains zero as a median, so its variance is at most $\alpha ^ { - 1 }$ and the square of its mean is at most twice that variance. Fatou’s lemma gives a finite second moment about the median. Applying PI to truncations of arbitrary linear functionals and passing to the limit gives the covariance bound. Smoothness then gives $\mathbb { E } _ { \pi } | V | < \infty$

The Gaussian maximum-entropy bound, or nonnegativity of KL relative to a Gaussian with covariance $\alpha ^ { - 1 } I _ { d } .$ , implies

$$
\mathsf { h } ( \pi ) \leq \frac { d } { 2 } \log \frac { 2 \pi \mathrm { e } } { \alpha } .
$$

Since $\mathtt { h } ( \pi ) = \log Z _ { V } + \mathbb { E } _ { \pi } V$ and $\mathbb { E } _ { \pi } V \geq V ( x _ { * } )$

$$
\log Z _ { V } + V ( x _ { * } ) \leq \frac { d } { 2 } \log \frac { 2 \pi \mathrm { e } } { \alpha } .\tag{41}
$$

For $u = x - x _ { 0 }$ , the smoothness upper bound for V gives

$$
\begin{array} { l } { \displaystyle { \log \frac { q _ { 0 } ( { \boldsymbol x } ) } { \pi ( { \boldsymbol x } ) } \leq \log Z _ { V } + V ( { \boldsymbol x } _ { 0 } ) + \frac { d } { 2 } \log \frac { \beta } { \pi } + \nabla V ( { \boldsymbol x } _ { 0 } ) \cdot { \boldsymbol u } - \frac { \beta } { 2 } \| { \boldsymbol u } \| ^ { 2 } } } \\ { \displaystyle \leq A _ { 0 } + \frac { G _ { 0 } } { 2 } + \frac { d } { 2 } \log ( 2 \mathrm { e } \kappa ) } .  \end{array}
$$

Here $D _ { \infty } ( q _ { 0 } \| \pi ) = \log \mathrm { e s s } \operatorname* { s u p } ( q _ { 0 } / \pi )$ . Taking the essential supremum proves the first upper bound. Finally, the descent lemma evaluated at $x _ { 0 } - \nabla V ( x _ { 0 } ) / \beta$ gives $G _ { 0 } / 2 \le V ( x _ { 0 } ) - V ( x _ { * } ) \le A _ { 0 }$ □

## F The remote-mass construction

Define

$$
U ( t ) = \left\{ \begin{array} { l l } { \frac { 3 } { 8 } + \frac { 3 } { 4 } t ^ { 2 } - \frac { 1 } { 8 } t ^ { 4 } , } & { | t | \leq 1 , } \\ { | t | , } & { | t | \geq 1 , } \end{array} \right. \quad \phi ( x ) = ( x + x ^ { 2 } / 2 ) \mathrm { e } ^ { - x ^ { 2 } } ,
$$

and, for $R \geq 2$ , set

$$
V _ { R } ( x ) = U ( x - R ) - R + \phi ( x ) , \qquad \pi _ { R } ( x ) \propto \mathrm { e } ^ { - V _ { R } ( x ) } .\tag{42}
$$

Lemma 20 (Uniform geometry and identical local responses). The potentials $V _ { R }$ are normalizable, $C ^ { 2 }$ , and 20-smooth. They satisfy

$$
V _ { R } ( 0 ) = 0 , \qquad V _ { R } ^ { \prime } ( 0 ) = 0 , \qquad V _ { R } ^ { \prime \prime } ( 0 ) = 1 .
$$

Their Gibbs laws have PI constant at least $\alpha _ { 0 } = ( 4 \mathrm { e } ^ { 1 9 / 8 } ) ^ { - 1 }$ , independently of R. For $A \leq R$

$$
\begin{array} { r } { \pi _ { R } ( ( - \infty , A ] ) \leq \frac { 1 } { 2 } \mathrm { e } ^ { 1 9 / 8 } \mathrm { e } ^ { A - R } . } \end{array}\tag{43}
$$

For every $x \leq R - 1$ , the value-and-gradient oracle returns

$$
V _ { R } ( x ) = - x + \phi ( x ) , \qquad V _ { R } ^ { \prime } ( x ) = - 1 + \phi ^ { \prime } ( x ) ,\tag{44}
$$

independently of R.

Proof. The polynomial branch of U matches |t| in value and in its first two derivatives at $\pm 1$ . It satisfies $0 \leq U - | \cdot | \leq 3 / 8$ and $0 \leq U ^ { \prime \prime } \leq 3 / 2$ . Also $\phi ( 0 ) = 0 , \phi ^ { \prime } ( 0 ) = \phi ^ { \prime \prime } ( 0 ) = 1 , \| \phi \| _ { \infty } < 1$ , and

$$
\phi ^ { \prime \prime } ( x ) = ( 1 - 6 x - 5 x ^ { 2 } + 4 x ^ { 3 } + 2 x ^ { 4 } ) \mathrm { e } ^ { - x ^ { 2 } } .
$$

Since $\mathrm { s u p } _ { x } | x | ^ { j } \mathrm { e } ^ { - x ^ { 2 } } \leq 1$ for $j = 0 , \ldots , 4$ , we have $\| \phi ^ { \prime \prime } \| _ { \infty } \leq 1 8$ . These facts prove the smoothness, normalizability, and origin claims.

The centered Laplace density $\ell ( x ) = \mathrm { e } ^ { - | x | } / 2$ has PI constant $1 / 4$ . Indeed, for $g ( 0 ) = 0$ , integration by parts on a half-line gives

$$
\int _ { 0 } ^ { \infty } g ( x ) ^ { 2 } \mathrm { e } ^ { - x } \mathrm { d } x = 2 \int _ { 0 } ^ { \infty } g ( x ) g ^ { \prime } ( x ) \mathrm { e } ^ { - x } \mathrm { d } x \leq 2 \left( \int _ { 0 } ^ { \infty } g ( x ) ^ { 2 } \mathrm { e } ^ { - x } \mathrm { d } x \right) ^ { 1 / 2 } \left( \int _ { 0 } ^ { \infty } g ^ { \prime } ( x ) ^ { 2 } \mathrm { e } ^ { - x } \mathrm { d } x \right) ^ { 1 / 2 } .
$$

Apply the resulting factor-four inequality to $g = f - f ( 0 )$ on both half-lines, and use Var $f \leq$ $\mathbb { E } _ { \ell } ( f - f ( 0 ) ) ^ { 2 }$ . Bounded smooth functions are justified by cutof approximation.

Relative to the Laplace density translated by R, the potential perturbation is

$$
b _ { R } ( x ) = U ( x - R ) - | x - R | + \phi ( x ) , \qquad \mathrm { o s c } \ : b _ { R } \leq 1 9 / 8 .
$$

If a density ratio lies between m and M, comparison of the variance using its infimum over constants and comparison of the Dirichlet form worsen the PI constant by at most $M / m$ . Here $M / m \leq \mathrm { e } ^ { 1 9 / 8 }$ giving PI constant α<sub>0</sub>. The same density comparison and the translated Laplace left tail $\mathrm { g i v e \ ( 4 3 ) }$ When $x \leq R - 1 , U ( x - R ) = R - x$ , proving (44). □

## F.1 Proof of the lower bound

Proof. First work in dimension one. Suppose a total oracle algorithm has a uniform deterministic query cap q and the asserted accuracy on the family in Lem. 20, with common supplied bounds $\beta = 2 0 , \alpha = \alpha _ { 0 }$ , and reference point zero.

Run its decision rules against the common response function $V _ { \infty } ( x ) = - x + \phi ( x )$ . This is a device for generating transcripts, not a normalizable target. Every finite transcript is consistent with some admissible $V _ { R } ,$ , by choosing R beyond all query locations in that transcript. Thus the rules produce a finite next query or an output, and they cannot make a $( q + 1 )$ st query: such a finite prefix would also occur for an admissible target and violate the uniform cap. Totality on admissible inputs likewise rules out nontermination between queries along a realizable finite prefix.

For a random seed, the at most q query locations and the final output are finitely many finite real random variables. Choose a deterministic A such that, with probability at least 0.99, all are at most $A .$ For $R > A + 1$ , couple the two runs with the same seed. On that event all oracle replies and the final output agree with the common-response run. The algorithm’s output under $V _ { R }$ therefore lies in $( - \infty , A ]$ with probability at least 0.99. Increase R until (43) is at most 0.01. Its TV error is at least 0.98, a contradiction.

For fixed $d \geq 2 .$ , use

$$
\widetilde V _ { R } ( \boldsymbol { x } ) = V _ { R } ( \boldsymbol { x } _ { 1 } ) + \frac { 1 } { 2 } \sum _ { i = 2 } ^ { d } x _ { i } ^ { 2 } .
$$

Tensorization preserves the common PI lower bound, smoothness remains at most 20, and the Hessian at the origin is $I _ { d }$ . Apply the same transcript argument to the first coordinate of each query and output. □

## G Parameter selection and deterministic query accounting

The local-sampler parameters are fixed before the target dimension and accuracy are supplied. A convenient explicit recipe is

$$
\gamma _ { \mathrm { L S I } } = \operatorname* { m i n } \{ 1 / 8 , \nu / 4 \} , \qquad \gamma _ { \mathrm { P I } } = \operatorname* { m i n } \{ 1 / 8 , \nu / 1 2 \} , \qquad K = \lfloor 2 + 4 / \gamma \rfloor + 1 .
$$

In either regime take $\omega$ as in (45) below. A fixed integer threshold $n _ { * } ( \nu )$ dominates the threshold in Prop. 5, ensures $n \geq 2$ and $\delta \leq 1 / 2$ , and absorbs the fixed-order logarithmic factors. Its size is not quantitatively bounded here. We verify below that the choices have strict margins for both error and cost.

## G.1 LSI parameter bounds

Write $E _ { \mathrm { o u t e r } } = A _ { n } \delta ^ { K } M _ { N }$ , and let $E _ { \mathrm { d e s c e n t } }$ and $E _ { \mathrm { b a s e } }$ denote the two decoder contributions bounded in $\mathrm { A p p }$ . B.

Fix $\nu > 0$ . Choose $0 < \gamma < 1 / 4$ with $2 \gamma < \nu$ , choose an integer $K > 2 + 4 / \gamma$ , and set, for example,

$$
\omega = { \frac { 1 } { 1 0 0 0 ( K + 2 ) } } .\tag{45}
$$

The constraint in (7) holds, and $\gamma ( K - 1 ) > 4 , \omega < 1$ , and $\gamma ( 1 + \omega ) < \nu$ . Using $\kappa \leq n$ in (33),

$$
E _ { \mathrm { o u t e r } } \leq C A _ { n } ( \log n ) n ^ { 5 / 2 - \gamma ( K - 1 ) } = o ( n ^ { - 1 } ) .
$$

Also $E _ { \mathrm { d e s c e n t } } \leq C A _ { n } ( \log n ) n ^ { 1 / 2 - \gamma K } = o ( n ^ { - 1 } )$ and $E _ { \mathrm { b a s e } } \leq C n ^ { - 7 / 2 - \gamma } = o ( n ^ { - 1 } )$ . The polylogarithmic factors have constants depending only on the fixed local-sampler parameters. Hence a single threshold $n _ { * } ( \nu )$ makes each of these three errors at most $1 / ( 4 n )$ for all $n \geq n _ { * } ( \nu )$

Now take

$$
n = \operatorname* { m a x } \{ d , n _ { * } ( \nu ) , \lceil \kappa \rceil , \lceil \varepsilon ^ { - 1 } \rceil , \lceil G _ { 0 } \rceil \} .\tag{46}
$$

Then $1 / n \le \varepsilon$ , and Prop. 8 proves the accuracy claim. By (18) and (32),

$$
Q \leq 1 + C \kappa \mathrm { p o l y l o g } ( n ) n ^ { \gamma ( 1 + \omega ) } \leq C _ { \nu } \kappa n ^ { \nu } .
$$

The second inequality absorbs both the initial query and the logarithmic factor by increasing $C _ { \nu }$ using $\kappa n ^ { \nu } \geq 1$ . The dimension choice satisfies $n \leq C _ { \nu } ( d + \kappa + \varepsilon ^ { - 1 } + G _ { 0 } )$ , proving (4) after projection and inverse rescaling. All loop counts depend only on known parameters after the first reference query. The query cap of NoisySample is uniform in all subsequently generated centers, so the resulting query bound is deterministic. □

## G.2 PI parameter bounds

Fix $\nu > 0$ , choose $0 < \gamma < 1 / 4$ with $6 \gamma < \nu ,$ choose $K > 2 + 4 / \gamma$ , and choose ω by (45). Let

$$
\begin{array} { r } { n = \operatorname* { m a x } \{ d , n _ { * } ( \nu ) , \lceil \kappa ( B + 1 ) ^ { 2 } \rceil , \lceil \varepsilon ^ { - 1 } \rceil \} , } \end{array}\tag{47}
$$

where the threshold is enlarged below as needed. Then $\kappa \leq n , B \leq { \sqrt { n } } .$ , and $H \leq C n$ . Thus

$$
E _ { \mathrm { o u t e r } } \leq C A _ { n } n ^ { 5 / 2 - \gamma ( K - 1 ) } = o ( n ^ { - 1 } ) .
$$

The decoder errors are the same as in the LSI proof and are also $o ( n ^ { - 1 } )$ . A fixed threshold makes each at most $1 / ( 4 n ) \leq \varepsilon / 4$ , proving accuracy by Prop. 8.

By (18),

$$
Q - Q _ { \mathrm { i n i t } } \leq C \kappa H \mathrm { p o l y l o g } ( n ) n ^ { \gamma ( 1 + \omega ) } \leq C _ { \nu } \kappa H n ^ { \nu / 3 } ,\tag{48}
$$

since $3 \gamma ( 1 + \omega ) < 6 \gamma < \nu .$ . Put $T = d + \kappa + B + { \varepsilon } ^ { - 1 }$ . Because $\kappa ( B + 1 ) ^ { 2 } \leq 4 T ^ { 3 }$ and the other padding terms are also at most $C _ { \nu } T ^ { 3 }$ , we have $n \leq C _ { \nu } T ^ { 3 }$ . Substitution into (48) proves (5). As in the LSI case, every invocation is uniformly capped and every loop length is predetermined from supplied parameters. □