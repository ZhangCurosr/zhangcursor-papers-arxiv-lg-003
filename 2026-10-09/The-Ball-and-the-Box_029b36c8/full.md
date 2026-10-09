PREPRINT • October 2026

# The Ball and the Box

Two Geometries of Computation in Superposition

Xiaoyu Li University of New South Wales xiaoyu.li2@unsw.edu.au

Lequan Lin University of Sydney lequan.lin@sydney.edu.au

Dai Shi   
University of Cambridge   
ds2213@cam.ac.uk

Jiaojiao Jiang University of New South Wales jiaojiao.jiang@unsw.edu.au

Junbin Gao University of Sydney junbin.gao@sydney.edu.au

Andi Han University of Sydney andi.han@sydney.edu.au

## ABSTRACT

Neural representations can encode more features than they have dimensions, a phenomenon known as superposition. We study the dimension needed to compute Boolean gates from such representations. For a single threshold layer with a Gaussian random dictionary and uniformly random sparse Boolean inputs, we derive sharp dimension thresholds under two error criteria. A vanishing expected error count can require more dimensions than correctness of every output with high probability. Shared reads explain the gap: rare realizations can produce many errors at once. The expected-count threshold has ball geometry, while joint reliability has box geometry when a gate is evaluated on every feature tuple. Optimizing shared readout weights and biases gives explicit thresholds for conjunction, disjunction, and majority. For pairwise conjunction, the analysis also describes the transition near the threshold, in agreement with exact simulations.

Keywords: computation in superposition, sparse feature representations, threshold circuits, Gaussian extreme values, sharp thresholds, reliability, mechanistic interpretability

## CONTENTS

1 Introduction 2 A Guide to the appendix . 18   
2 d l d i i 3 B Read-vector reductions and the supremum 19   
3 The exact read law 5 C Proofs for the ball . 22   
4 Counting errors: the ball 6 D Proofs for the box . 24   
5 Reliability: the supremum and the box 7 E Proofs for threshold gates 27   
6 Threshold gates 9 F The critical window 29   
7 The critical window 11   
G Extensions of the model . 32   
8 Extensions 12   
H Statements and proofs for Section 8 33   
9 Exact simulations 13 Window formulas and the expected-count gap 39   
10 Discussion 14   
Further comparisons and experiments 40   
AI Use Statement 14   
References 14 K Finite checks 43   
L Experimental protocol and additional results 44   
Appendix M Applications, implications and open problems 47

## 1 Introduction

A network computes in superposition when many decisions read a vector that stores more features than it has dimensions (Elhage et al., 2022; Hänni et al., 2024; Adler and Shavit, 2024). Each feature has a direction in $\mathbb { R } ^ { d } .$ , and a threshold unit reads a weighted sum of these directions. If only � of � features are active, a unit probe sees interference of order ${ \sqrt { k / d } } .$ . This controls one decision. What dimension is needed when a whole family of decisions shares the same stored vector? We count the width � of that vector; each gate has its own readout unit.

Pairwise conjunction gives a concrete example. Store � binary features in a Gaussian dictionary and read all $\binom m 2$ products $x _ { i } x _ { j }$ with probes $e _ { i } + e _ { j }$ . There are two choices: require the expected number of wrong outputs to vanish, or require every output to be correct with probability tending to one. The bias can be fixed at the midpoint $3 / 2$ or optimized. These choices give four sharp thresholds (Table 1). When log $k = o ( \log m )$ , the constants are $1 6 , 8 ,$ 4 and 2 in units of � log �. When log $m \sim$ log �, all four thresholds equal 32� log � to first order.

Reliability retains dependence between gate errors. Group gates by their active-input pattern $z \in \{ 0 , 1 \} ^ { r }$ . A gate in this class has weights � and margin $s _ { z } ;$ it fails when its interference crosses the margin. Counting errors uses only the law of each gate. Its answer is unchanged if the $N _ { z }$ interference vectors are independent Gaussian points, whose extreme projections have the support function of a ball of radius $\sqrt { 2 \log N _ { z } }$ . Reliability uses the actual joint law. In the family of all gates, every interference vector is assembled from the same active and inactive feature reads. Their extremes form a box (Figure 1), with half-sides $\lambda _ { 1 } = { \sqrt { 2 \log k } }$ and $\lambda _ { 0 } = \sqrt { 2 \log ( m - k ) }$ according to the input state. In either case, a class costs

$$
d \approx k \frac { h ( a ) ^ { 2 } } { s _ { z } ^ { 2 } } ,\tag{1}
$$

where $h ( a ) = \| a \| \sqrt { 2 \log N _ { z } }$ for the ball and $\begin{array} { r } { h ( { a } ) = \sum _ { l } \lvert a _ { l } \rvert \lambda _ { z _ { l } } } \end{array}$ for the box. The box lies inside the ball to first order. The Cauchy–Schwarz inequality between their support functions explains why the expected count can require more dimensions than reliability. Optimizing the bias produces a separate reduction by assigning more margin to classes with larger extremes.

Results The exact read law gives a common probabilistic reduction (Theorem 3.1). Expectedcount thresholds depend on class sizes; reliability depends on Gaussian suprema (Theorems 4.1 and 5.1). The optimized count is a second-order cone program with a converse over all gate-local readouts (Theorem 4.2). For the universal family, the box program has a converse over common representations, even if chosen after observing the reads (Theorem 5.4). Symmetric threshold gates have closed forms (Theorem 6.1). Pairwise conjunction has explicit Gumbel–Poisson windows, tested by exact simulation (Theorem 7.1).

Previous work Superposition explains polysemantic neurons in models of sparse features (Olah et al., 2020; Elhage et al., 2022). Hänni et al. (2024) and Adler and Shavit (2024) study neural computation with superposed hidden activations. Here we ask which Boolean functions a single threshold layer can read from one stored code. Newgas (2025) studies learned universal AND circuits; Bhagat et al. (2026) distinguish compressed computation from computation in superposition. Garg et al. (2026) distinguish linear storage from threshold accessibility, and

Table 1. Sharp first-order thresholds for pairwise conjunction, with $\lambda _ { 1 } = { \sqrt { 2 \log k } }$ and $\lambda _ { 0 } = \sqrt { 2 \log ( m - k ) }$ Second lines give the limit when log $k \ = \ o ( \log m )$ . All entries are 32� log � to first order when <sup>log</sup> <sup>�</sup> ∼ <sup>log</sup> <sup>�.</sup>
<table><tr><td></td><td>Midpoint bias Optimized bias</td><td></td></tr><tr><td>Count errors, EX → 0 (ball)</td><td> $8 k ( \lambda _ { 1 } ^ { 2 } + \lambda _ { 0 } ^ { 2 } )$ </td><td> $2 k \big ( \sqrt { 2 } \lambda _ { 1 } + \big ( \lambda _ { 1 } ^ { 2 } + \lambda _ { 0 } ^ { 2 } ) ^ { 1 / 2 } \big ) ^ { 2 }$ </td></tr><tr><td rowspan="2">Forbid errors, P(X = 0) → 1 (box)</td><td> $\sim 1 6 k \log m$ </td><td>∼ 4 k log m</td></tr><tr><td> $4 k ( \lambda _ { 1 } + \lambda _ { 0 } ) ^ { 2 }$ </td><td> $k ( 3 \lambda _ { 1 } + \lambda _ { 0 } ) ^ { 2 }$ </td></tr><tr><td></td><td> $\sim 8 k \log m$ </td><td> $\sim 2 k \log m$ </td></tr></table>

Ivanitskiy et al. (2026) analyze feature recovery from superposition. Our comparison with recovery uses the same dictionary, input law and error criterion.

Sharp thresholds also appear in compressed sensing and support recovery (Candès and Tao, 2006; Donoho, 2006; Donoho and Tanner, 2009; Fletcher et al., 2009; Wainwright, 2009). Sparse superposition codes supply the Gaussian conditioning identity used here (Barron and Joseph, 2011; Joseph and Barron, 2014); vector-symbolic architectures also store superposed random vectors (Plate, 1995; Thomas et al., 2021). Diferent error criteria change capacity constants in associative memories (McEliece et al., 1987). The present geometry identifies that diference for threshold gate families. The proofs use Gaussian concentration and comparison (Borell, 1975; Tsirel’son et al., 1976; Slepian, 1962) and Gaussian extreme-value theory (Leadbetter et al., 1983; Resnick, 1987). The gap between the expected and typical error counts is an annealed–quenched distinction (Derrida, 1981).

## 2 Model and criteria

Let $E = [ e _ { 1 } | \cdots | e _ { m } ] \in \mathbb { R } ^ { d \times m }$ have independent columns $e _ { i } \sim N ( 0 , I _ { d } / d )$ . Independently choose � <sup>uniformly</sup> <sup>among</sup> <sup>the</sup> <sup>�-subsets</sup> <sup>of</sup> [<sup>�</sup>]<sup>.</sup> <sup>Write</sup> $x = \mathbf { 1 } _ { S } , n = m - k$ and

$$
u = E x = \sum _ { i \in S } e _ { i } , \qquad R = E ^ { \top } u .\tag{2}
$$

The feature read is $R _ { i } = \langle e _ { i } , u \rangle$ . Throughout, $k  \infty$ and $m / k \to \infty ;$ other restrictions are stated where needed.

Gates and classes A gate is an ordered tuple $I = \left( I _ { 1 } , \ldots , I _ { r } \right)$ of distinct features and a nonconstant threshold function $g : \{ 0 , 1 \} ^ { r }  \{ 0 , 1 \}$ . Its target is $g ( x _ { I } )$ . A family $\mathcal { F }$ has common arity � and function �. Finitely many types can be treated separately. The class of a gate is its input pattern $z = x _ { I , }$ , and $\mathscr { F } _ { z } ( S )$ <sup>contains</sup> <sup>the</sup> <sup>gates</sup> <sup>of</sup> <sup>that</sup> <sup>class.</sup> <sup>With</sup> <sup>falling</sup> <sup>factorials</sup> (<sup>�</sup>)� <sup>and</sup> $\begin{array} { r } { | z | = \sum _ { l } z _ { l } , } \end{array}$

$$
N _ { z } = \mathbb { E } | \mathcal { F } _ { z } ( S ) | = | \mathcal { F } | \frac { ( k ) _ { | z | } ( n ) _ { r - | z | } } { ( m ) _ { r } } .\tag{3}
$$

The universal family $\mathcal { U } _ { r } ( g )$ contains all $( m ) _ { i }$ <sub>�</sub> tuples. Its class sizes are deterministic: $N _ { z } =$ $( k ) _ { | z | } ( n ) _ { r - | z | }$ . For symmetric gates, unordered subsets change these counts only by constant factors and leave first-order thresholds unchanged.

![](images/a563c084eed5153749424184fe799563dc9a1f3ea0ed12c0bb24f07ed862b205.jpg)

![](images/ea325ef482cd8c17d0ba1510c921c6084f8e211bf8f888cf50917037db95fbe6.jpg)  
Figure 1. Two geometries for the one-active class of pairwise conjunction. Left: independent vectors have the extreme projections of a disc of radius ${ \sqrt { 2 \log N } } ,$ , where $N = k ( m - k )$ . Right: shared active and inactive reads form a grid in $[ - \lambda _ { 1 } , \lambda _ { 1 } ] \times [ - \lambda _ { 0 } ^ { \mathrm { ' } } , \lambda _ { 0 } ]$ . A gate fails beyond the line normal to $a = ( 1 , 1 )$ at distance $s _ { z } / ( \sigma _ { d } \lVert a \rVert )$ . The expected count must clear the disc; reliability must clear the rectangle. Here $\lambda _ { 0 } = 3 \lambda _ { 1 } ;$ points are illustrative.

Readouts and margins Gate � outputs $\mathbf { 1 } \{ \langle w _ { \gamma } , u \rangle > \theta _ { \gamma } \}$ . A gate-local rule chooses $( w _ { \gamma } , \theta _ { \gamma } )$ from its own columns $E _ { I }$ and independent randomness, before drawing the input. Rules may difer between gates. They can use their own column lengths and angles, but cannot use the support or other columns. A canonical rule uses $\begin{array} { r } { w _ { \gamma } = \sum _ { l } a _ { l } e _ { I _ { l } } } \end{array}$ and a common bias �. The pair $( a , \theta )$ represents � when

$$
s _ { z } ( a , \theta ) = ( 2 g ( z ) - 1 ) \bigl ( \langle a , z \rangle - \theta \bigr ) > 0 \quad \mathrm { f o r ~ a l l ~ } z .\tag{4}
$$

Write $\textstyle \operatorname { R e p } ( g )$ for these representations and $\varsigma _ { z } = ( - 1 ) ^ { g ( z ) }$ for the error direction. Pairwise conjunction with $a = ( 1 , 1 )$ has clean scores $0 , 1 , 2$ and margins $\theta , \theta - 1 , 2 - \theta$

Error criteria Let � count wrong outputs on a draw of �, � . We study

$$
\mathbb { E } X \to 0 \qquad \mathrm { a n d } \qquad \mathbb { P } ( X = 0 ) \to 1 .\tag{5}
$$

The first implies the second by Markov’s inequality. The converse can fail: one rare draw can produce many errors. A sharp threshold $D ( k , m )$ has a multiplicative $1 \pm \varepsilon$ transition for every fixed $\varepsilon \in ( 0 , 1 )$ . Below it, failure means $\mathbb { E } X  \infty$ for the count and $\mathbb { P } ( X = 0 )  0$ for reliability.

Levels and prices Set

$$
\lambda _ { 1 } = \sqrt { 2 \log k } , \quad \lambda _ { 0 } = \sqrt { 2 \log n } , \quad \beta = \lambda _ { 0 } / \lambda _ { 1 } .\tag{6}
$$

For class �, let $\lambda _ { z } = ( \lambda _ { z _ { 1 } } , \ldots , \lambda _ { z _ { r } } )$ . Eventually $\beta \geq 1$ , and for the universal family 2 log $N _ { z } =$ $\| \lambda _ { z } \| ^ { 2 } ( 1 + o ( 1 ) )$ . Finite-size calculations use $\hat { \lambda } _ { 1 } = \overline { { \Phi } } ^ { - 1 } \big ( 1 / k \big )$ and $\hat { \lambda } _ { 0 } = \overline { { \Phi } } ^ { - 1 } ( 1 / n ) ; \hat { \lambda } _ { p } = \lambda _ { p } ( 1 + o ( 1 ) )$

Definition 2.1 (Ball, box and prices). For a class � let $\rho _ { z } = { \sqrt { 2 \log N _ { z } } }$ and

$$
\mathsf { B a l l } _ { z } = \{ v \in \mathbb { R } ^ { r } : \| v \| \leq \rho _ { z } \} ,
$$

$$
{ \tt B o x } _ { z } = \prod _ { l = 1 } ^ { r } [ - \lambda _ { z _ { l } } , \lambda _ { z _ { l } } ] ,
$$

with support functions $h _ { \mathsf { B a l l } _ { z } } ( a ) = \rho _ { z } \| a \|$ and $\begin{array} { r } { h _ { \mathsf { B o x } _ { z } } ( a ) = \sum _ { l } \lvert a _ { l } \rvert \lambda _ { z _ { l } } } \end{array}$ . For $( a , \theta ) \in \operatorname { R e p } ( g )$ define the ball price and the box price

$$
D _ { \mathsf { b a l l } } ( a , \theta ) = k \operatorname* { m a x } _ { z } \frac { h _ { \mathsf { B a l l } _ { z } } ( a ) ^ { 2 } } { s _ { z } ( a , \theta ) ^ { 2 } } ,\tag{7}
$$

$$
D _ { \mathsf { b o x } } ( a , \theta ) = k \operatorname* { m a x } _ { z } \frac { h _ { \mathsf { B o x } _ { z } } ( a ) ^ { 2 } } { s _ { z } ( a , \theta ) ^ { 2 } } ,
$$

and the optimal values $\begin{array} { r } { V _ { \mathsf { b a l l } } = \operatorname* { i n f } _ { \mathsf { R e p } ( g ) } D _ { \mathsf { b a l l } } / k } \end{array}$ and $\begin{array} { r } { V _ { \mathsf { b o x } } = \operatorname* { i n f } _ { \mathrm { R e p } ( g ) } D _ { \mathsf { b o x } } / k . } \end{array}$

Both prices are invariant under positive rescaling of $( a , \theta )$ . The box corners have norm $\| \lambda _ { z } \| ,$ so the box lies inside the ball to first order for the universal family.

## 3 The exact read law

All canonical gates use the same read vector �. Conditioning on � gives an exact law, the noiseless case of the lemmas for sparse superposition codes in Barron and Joseph (2011, Lemma 1) and Joseph and Barron (2014, Lemma 3). By symmetry fix $S = [ k ]$ , and write $A _ { i } = R _ { i }$ for $i \leq k$ and $B _ { j } = R _ { k + j }$ for $j \leq n$

Theorem 3.1 (Exact conditional law). Let $H \sim \chi _ { d } ^ { 2 } / d , Z _ { 1 } , \ldots , Z _ { k } \sim N ( 0 , 1 )$ and $W _ { 1 } , \ldots , W _ { n } \sim$ $N ( 0 , 1 )$ be mutually independent. Write $\bar { Z } = k ^ { - 1 } \textstyle \sum _ { i } Z _ { i }$ and $\sigma = \sqrt { k H / d }$ . Then, jointly in $i \leq k$ and $j \leq n$

$$
( A _ { i } , B _ { j } ) \stackrel { d } { = } \big ( H + \sigma ( Z _ { i } - \bar { Z } ) , \sigma W _ { j } \big ) .\tag{8}
$$

The identity holds for all $d , k , n \geq 1$ .

Proof. Condition on $\begin{array} { r } { u = \sum _ { i \leq k } e _ { i } } \end{array}$ . Gaussian conditioning gives $e _ { i } = u / k + \eta _ { i }$ for $i \leq k$ , where $( \eta _ { i } ) _ { i \leq k }$ is independent of � and jointly Gaussian with Cov ${ \boldsymbol { \mathbf { \mathit { \Pi } } } } ( { \eta } _ { i } , { \eta } _ { l } ) = d ^ { - 1 } ( { \bf 1 } \{ i = l \} - 1 / k ) { \cal I } _ { d }$ . Hence $A _ { i } = \| u \| ^ { 2 } / k + \langle \eta _ { i } , u \rangle$ . Given $u ,$ the vector $( \langle \eta _ { i } , u \rangle ) _ { i \leq k }$ is centered Gaussian with covariance $( \| u \| ^ { 2 } / d ) ( I _ { k } - \mathbf { 1 1 } ^ { \top } / k )$ , which is the covariance of $( \| u \| / \sqrt { d } ) ( Z _ { i } - \bar { Z } ) _ { i \leq k }$ . The inactive columns are independent of $( u , \eta )$ , so given � the reads $B _ { j }$ are independent $N ( 0 , \| u \| ^ { 2 } / d )$ variables, independent of the active ones. The conditional law depends on � only through $\lVert u \rVert$ , and $H = \bar { \| } u \| ^ { 2 } / k \sim \chi _ { d } ^ { 2 } / d ;$ since $\| u \| / { \sqrt { d } } = \sigma ,$ , this is (8). ■

The active sample is centered at its own mean and shifted by $H ;$ the inactive sample is independent. Both share the scale �. Without conditioning on $H ,$ , the reads are not jointly Gaussian: $\mathbb { E } A _ { i } = 1$ , Var $A _ { i } = ( k + 1 ) / d$ and $\mathrm { C o v } ( A _ { i } , A _ { l } ) = 1 / d$ for $i \neq l$ . We use the conditional

representation throughout.

Put $\xi _ { i } = Z _ { i } - \bar { Z }$ on � and $\xi _ { k + j } = W _ { j }$ of �. A gate of class � has score � $\langle a , z \rangle + \sigma \langle a , \xi _ { I } \rangle$ . Let $M _ { z } =$ m $\mathsf { a x } _ { I \in \mathcal { F } _ { z } ( S ) } \big \langle \varsigma _ { z } a , \xi _ { I } \big \rangle$ , with max $\varnothing = - \infty$ . For a fixed representation, the whole network is correct almost surely exactly when every class satisfies $\sigma M _ { z } < s _ { z } ^ { H }$ , where $s _ { z } ^ { H } = ( 2 g ( z ) - 1 ) ( H \langle a , z \rangle - \theta )$ (proposition B.1). Counting errors instead uses the following marginal law.

Proposition 3.2 (Exact one-gate law). Let $a \neq 0 .$ The score of a canonical gate of class � has the law $o f$

$$
\begin{array} { r } { Y _ { z } = \langle a , z \rangle J + \sqrt { \big ( k \| a \| ^ { 2 } - \langle a , z \rangle ^ { 2 } \big ) J / d } \zeta , } \end{array}\tag{9}
$$

where $J \sim \chi _ { d } ^ { 2 } / d$ and $\zeta \sim N ( 0 , 1 )$ are independent. Consequently $\begin{array} { r } { \mathbb E X = \sum _ { z } N _ { z } p _ { z } } \end{array}$ with $p _ { z } =$ $\mathbb { P } \big ( \varsigma _ { z } ( Y _ { z } - \theta ) \big ) \geq 0 \big )$

The proof is in Appendix B. For pairwise conjunction, $a = ( 1 , 1 )$ gives $Y _ { j } = j J + { \sqrt { ( 2 k - j ^ { 2 } ) J / d } } \zeta ,$ $j \in \{ 0 , 1 , 2 \}$ . Thus each class error probability is a one-dimensional chi-square integral of a normal tail.

## 4 Counting errors: the ball

Linearity of expectation removes dependence between gates. At the exponential scale, a class with margin $y \sigma _ { d } \lVert a \rVert$ contributes $\bar { N _ { z } } \exp ( - y ^ { 2 } / 2 )$ errors. The class size sets the Gaussian tail level.

Theorem $4 . 1$ (Ball law). Let $\mathcal { F }$ be a gate family of fixed arity � with $N _ { z } $ ∞ for every <sup>�</sup>, and let $( a , \theta ) \in \operatorname { R e p } ( g )$ be any sequence of representations for the canonical rule. For each fixed $\varepsilon \in ( 0 , 1 )$

$$
\begin{array} { r l } & { d \geq ( 1 + \varepsilon ) D _ { \mathsf { b a l l } } ( a , \theta ) \implies \mathbb { E } X \to 0 , } \\ & { d \leq ( 1 - \varepsilon ) D _ { \mathsf { b a l l } } ( a , \theta ) \implies \mathbb { E } X \to \infty . } \end{array}
$$

Proof sketch. Concentrate � in (9) in a window proportional to the margin. This keeps the bound uniform when margins shrink. Gaussian upper tails control every class above the threshold; Mills’ lower bound gives divergence in a binding class below it. See Appendix C.

Optimizing the price also gives a converse over all local readouts.

Theorem $\pmb { 4 . 2 }$ (Weighted margin law). Let  be as in Theorem 4.1, and let

$$
V _ { \mathsf { b a l l } } = \operatorname* { m i n } _ { ( a , \theta ) \in \mathrm { R e p } ( g ) } \ \operatorname* { m a x } _ { z } \frac { 2 \log N _ { z } \| a \| ^ { 2 } } { s _ { z } ( a , \theta ) ^ { 2 } } .\tag{10}
$$

Fix $ { \varepsilon } \in \left( 0 , 1 \right)$ . If $\dot { \mathbf { \zeta } } d \geq ( 1 + \varepsilon ) k V _ { \mathsf { b a l l } } ,$ , the canonical rule with a minimizing representation has $\mathbb { E } X \to 0 . \ I f d \leq ( 1 - \varepsilon ) k V _ { \mathsf { b a l l } } ,$ , every gate-local readout rule satisfies

$$
\mathbb { E } X \geq ( \operatorname* { m i n } _ { z } N _ { z } ) ^ { \varepsilon - o ( 1 ) }  \infty .
$$

Proofsketch. Normalize a local probe to unit length. Conditional on its own columns and class $z ,$ its interference is $N ( 0 , ( k - | z | ) / d )$ . On a high-probability Gram-matrix event, its clean weights have norm at most $1 + o ( 1 )$ . The program forces a small margin in some class; a Gaussian lower tail then bounds that gate’s error. Summing these marginal bounds needs no independence between errors. Bounded � uses a separate two-class argument (Appendix C).

For computation, maximize � over $( a , \theta , \kappa )$ with $\| a \| \leq 1$ and

$$
( 2 g ( z ) - 1 ) \big ( \langle a , z \rangle - \theta \big ) \geq \kappa \sqrt { 2 \log N _ { z } } \quad \mathrm { f o r ~ a l l ~ } z .\tag{11}
$$

Then $V _ { \sf b a l l } = \kappa _ { * } ^ { - 2 }$ . This second-order cone program has $r + 2$ variables and $2 ^ { r }$ linear constraints.   
It maximizes margin relative to class size: a less numerous class can tolerate a smaller margin.   
Every class remains a constraint.

## 5 Reliability: the supremum and the box

Reliability depends on the largest interference in each class. Let $G \sim N ( 0 , I _ { m } )$ be independent of $S ,$ and define

$$
\Gamma _ { z } ( S ) = \mathbb { E } { \Big [ } \operatorname* { m a x } _ { I \in { \mathcal { F } } _ { z } ( S ) } \left. \varsigma _ { z } a , G _ { I } \right. { \Big | } \ S { \Big ] } ,\tag{12}
$$

with max $\varnothing = - \infty$

Theorem 5.1 (Supremum law). Let $\mathcal { F }$ be any gate family of fixed arity $r ,$ let $( a , \theta ) \in \operatorname { R e p } ( g )$ be fixed, and assume $d / k \to \infty$ . Define the random threshold

$$
d ^ { * } ( S ) = k \operatorname* { m a x } _ { z } \frac { \Gamma _ { z } ( S ) _ { + } ^ { 2 } } { s _ { z } ( a , \theta ) ^ { 2 } } .
$$

Then for each fixed $ { \varepsilon } \in ( 0 , 1 )$ ,

$$
\begin{array} { r l } & { \mathbb { P } \big ( X = 0 , \ d \leq ( 1 - \varepsilon ) d ^ { * } ( S ) \big ) \to 0 , } \\ & { \mathbb { P } \big ( X \neq 0 , \ d \geq ( 1 + \varepsilon ) d ^ { * } ( S ) \big ) \to 0 . } \end{array}
$$

In particular, when $d ^ { * }$ is deterministic it is the sharp reliability threshold among sequences with $d / k \to \infty ; _ { \prime }$ for the universal family Theorem 5.4 removes this restriction.

Proof sketch. The class maximum is $\| a \| { - } \mathrm { I }$ ipschitz in $G ,$ so Gaussian concentration controls its deviation from $\Gamma _ { z } ( S )$ (Borell, 1975; Tsirel’son et al., 1976). The terms from $H - 1$ and $\bar { Z }$ vanish relative to fixed margins when $d / k \to \infty$ . The two threshold inequalities then give success or failure (Appendix B).

Corollary 5.2 (Counting is the union bound). In the setting of Theorem 5.1, $\Gamma _ { z } ( S ) \quad \leq$ $\| a \| \sqrt { 2 \log | \mathcal { F } _ { z } ( S ) | } .$ , so

$$
d ^ { * } ( S ) \leq k \operatorname* { m a x } _ { z } \frac { 2 \log | \mathcal { F } _ { z } ( S ) | \| a \| ^ { 2 } } { s _ { z } ( a , \theta ) ^ { 2 } } ,
$$

and $f o r$ the universal family the right side is $D _ { \mathsf { b a l l } } ( a , \theta )$ . If the tuples in $\mathscr { F } _ { z } ( S )$ are pairwise disjoint and $\begin{array} { r } { | \mathscr { F } _ { z } ( S ) |  \infty , } \end{array}$ then $\Gamma _ { z } ( S ) = \| a \| \sqrt { 2 \log | \mathcal { F } _ { z } ( S ) | } \left( 1 + o ( 1 ) \right)$ : without shared features, the reliability price of the class equals its ball price.

For nonnegative weights, shared features raise covariances and can lower the expected supremum (Sudakov, 1971; Fernique, 1975). Disjoint gates attain the ball price. A family with private partner sets shows when sharing first reduces the price.

```latex
Proposition 5.3 (Partial sharing). Consider pairwise gates with $a = ( 1 , 1 )$ , and suppose that on the
support � the gates of the one-active class are the pairs $( i , j )$ with $i \in S$ and $j \in C ( i )$ , where the
partner sets $C ( i ) , i \in S ,$ , are disjoint sets of $D = D ( k )$ inactivefeatures. Then, with $\lambda _ { D } = { \sqrt { 2 \log D } } ,$
$\Gamma _ { 1 0 } ( S ) = ( 1 + o ( 1 ) ) \left\{ \begin{array} { l l } { { 2 \sqrt { \log ( k D ) } } } & { { i f D \leq k , } } \\ { { \lambda _ { 1 } + \lambda _ { D } } } & { { i f D \geq k . } } \end{array} \right.$
Equivalently, $\Gamma _ { 1 0 } ( S ) = ( 1 + o ( 1 ) ) h _ { K _ { D } } ( 1 , 1 ) f o r K _ { D } = \{ v \in \mathbb { R } ^ { 2 } : v _ { 1 } ^ { 2 } \leq \lambda _ { 1 } ^ { 2 } , v _ { 1 } ^ { 2 } + v _ { 2 } ^ { 2 } \leq 2 \log ( k D ) \} ,$
the ball of the class cut by the slab of its shared coordinate.
```

The maximum is $\mathrm { m a x } _ { i \in S } ( Z _ { i } + M _ { i } )$ , where $M _ { i }$ is the largest of � independent partner reads. For $D \leq k ,$ both coordinates can split the extreme-value budget evenly. For $D \geq k ,$ the active coordinate saturates at $\lambda _ { 1 }$ and the partner contributes $\lambda _ { D }$ . The expressions agree at $D = k ;$ disjoint partner sets require $k D \leq n$ . The proof is in Appendix D.2.

The universal family Every tuple is available, so the largest class projection is a rearranged sum of extreme feature reads (lemma B.2). Each position contributes its weight magnitude times the level of its active or inactive pool. The finitely many relevant order statistics approximate these levels uniformly over weights. This gives the box law.

```latex
Theorem 5.4 (Box law). Let $\mathcal { F } = \mathcal { U } _ { r } ( g )$ with $k $ and $m / k \to \infty .$
(i) For every sequence $( a , \theta ) \in \mathrm { R e p } ( g )$ and fixed $\varepsilon \in ( 0 , 1 ) \colon$ $\mathbb { P } ( X = 0 )  1$ if $d \geq ( 1 +$
$\varepsilon ) D _ { \mathsf { b o x } } ( a , \theta )$ , and $\mathbb { P } ( X = 0 ) \to 0 i f d \leq ( 1 - \varepsilon ) D _ { \mathsf { b o x } } ( a , \theta ) .$
(ii) $I f d \leq ( 1 - \varepsilon ) k V _ { \mathsf { b o x } } ,$ then with probability tending to one no common representation makes
the canonical network correct: $\mathbb { P } \big ( \exists ( a , \theta ) \in \mathrm { R e p } ( g ) : X _ { a , \theta } = 0 \big ) \to 0 ,$ where $X _ { a , \theta }$ is the error
count of the rule $( a , \theta )$ on the same draw.
(iii) $\Gamma _ { z } = h _ { \mathsf { B o x } _ { z } } ( a ) ( 1 + o ( 1 ) )$ uniformly in $a \neq 0 .$
In particular $k V _ { \mathsf { b o x } }$ is the sharp reliability threshold over common representations, and it is attained
by any sequence of near-minimizers of $D _ { \mathsf { b o x } } .$
```

Proof sketch. On one event of probability tending to one, all upper and lower order statistics needed by any gate are within a factor $1 \pm \delta$ of their levels. The box identity then gives the support function uniformly over all representations. The common-scale and centering errors are smaller, so correctness reduces to the box-margin inequalities $( \mathrm { A p p e n d i x D } )$

The representation in part (ii) may depend on the reads and is shared by all gates. Extending the reliability converse to all gate-local rules is an open problem.

Corollary 5.5 (Counting overprices reliability). For the universal family, every representation and every class,

$$
h _ { { \tt B o x } _ { z } } ( a ) = \langle | a | , \lambda _ { z } \rangle \leq \| a \| \| \lambda _ { z } \| = h _ { { \tt B a l l } _ { z } } ( a ) ( 1 + o ( 1 ) ) ,
$$

with equality if and only $i f \left| a \right|$ is parallel to $\lambda _ { z }$ Hence $D _ { \mathsf { b o x } } \leq D _ { \mathsf { b a l l } } ( 1 + o ( 1 ) )$ and $V _ { \sf b o x } \ \leq$ $V _ { \sf b a l l } ( 1 + o ( 1 ) )$ , and the ball price of a class exceeds its box price by the factor se $\Upsilon ^ { 2 } / ( | a | , \lambda _ { z } ) ( 1 + o ( 1 ) )$

Mixed active and inactive classes create the gap: their side lengths need not align with the weights.

For pairwise conjunction, the one-active class gives a concrete calculation. Its weights are $( 1 , 1 )$ and its size is $k n ,$ so the ball support is $\sqrt { 2 ( \lambda _ { 1 } ^ { 2 } + \lambda _ { 0 } ^ { 2 } ) } \left( 1 + o ( 1 ) \right)$ , whereas the box support is $\lambda _ { 1 } + \lambda _ { 0 }$ . Dividing both by the same margin gives the ratio $2 ( 1 + \beta ^ { 2 } ) / ( 1 + \beta ) ^ { 2 }$ between the two class prices. This ratio equals one at $\beta = 1$ and tends to two as $\beta \to \infty$ . The largest inactive read is reused by every active partner. Counting each pair separately charges for �� opportunities; reliability uses the two shared pools.

Optimizing the box Write

$$
V _ { \mathsf { b o x } } = \lambda _ { 1 } ^ { 2 } v _ { g } ( \beta ) ,
$$

$$
v _ { g } ( \beta ) = \operatorname* { i n f } _ { ( a , \theta ) \in \mathrm { R e p } ( g ) } \operatorname* { m a x } _ { z } \Bigl ( \frac { \sum _ { l } \lvert a _ { l } \rvert \beta ^ { 1 - z _ { l } } } { s _ { z } ( a , \theta ) } \Bigr ) ^ { 2 } .\tag{13}
$$

For fixed $\kappa ,$ the constraints $\begin{array} { r } { s _ { z } \ge \kappa \sum _ { l } \vert a _ { l } \vert \beta ^ { 1 - z _ { l } } } \end{array}$ form a convex polyhedral cone. Introduce $u _ { l } \geq \pm a _ { l } ,$ , require $s _ { z } \geq 1$ , and use linear feasibility with bisection on $\kappa .$ This computes $v _ { g } ( \beta ) ^ { - 1 / 2 }$ (proposition B.3). For symmetric threshold gates, permutation averaging makes equal weights optimal in both programs.

## 6 Threshold gates

For $1 \leq t \leq r ,$ let $\mathrm { T H R } _ { t } ^ { r } ( z ) = \mathbf { 1 } \{ | z | \geq t \}$ . This includes conjunction $\mathrm { A N D } _ { r } ,$ disjunction $\mathrm { O R } _ { r } ,$ majority $\mathrm { M A J } _ { 2 t - 1 }$ and feature recovery $\mathrm { T H R } _ { 1 } ^ { 1 }$ . Equal weights ${ a } = \mathbf { 1 }$ are optimal, leaving only the bias. Set $\rho _ { j } = ( j \lambda _ { 1 } ^ { 2 } + ( r - j ) \lambda _ { 0 } ^ { 2 } ) ^ { 1 / 2 }$ for $0 \leq j \leq r .$

Theorem 6.1 (Threshold gates). Let $\mathcal { F } = \mathcal { U } _ { r } ( \mathrm { T H R } _ { t } ^ { r } )$ with $k  \infty$ and $m / k \to \infty$ . Only the classes with $t - 1$ and � active inputs can bind; both bind at the optimized biases, and the class $t - 1$ binds at the midpoint. The following are sharp thresholds.

(a) Forbidding errors, optimized: $k V _ { \mathrm { b o x } } = k \left( ( 2 t - 1 ) \lambda _ { 1 } + ( 2 r - 2 t + 1 ) \lambda _ { 0 } \right) ^ { 2 }$ . It is attained by ${ \boldsymbol { a } } = \mathbf { 1 }$ with the half-slack bias $\begin{array} { r } { \theta = t - \frac 1 2 + \frac 1 2 \sigma _ { d } ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } ) } \end{array}$ , which leaves equal slack to the two binding classes (the levels $\lambda _ { p }$ in place $o f \hat { \lambda } _ { p }$ give the same threshold), and below it no common representation succeeds.

(b) Forbidding errors, midpoint $\begin{array} { r } { a = \mathbf { 1 } , \theta = t - \frac { 1 } { 2 } \colon 4 k \big ( ( t - 1 ) \lambda _ { 1 } + ( r - t + 1 ) \lambda _ { 0 } \big ) ^ { 2 } . } \end{array}$

(c) Counting errors, midpoint: 4� � $\rho _ { t - 1 } ^ { 2 } \left( 1 + o ( 1 ) \right) = 8 r k \log N _ { t - 1 } ( 1 + o ( 1 ) ) .$

(d) Counting errors, optimized: $k V _ { \mathsf { b a l l } } = r k ( \rho _ { t } + \rho _ { t - 1 } ) ^ { 2 } ( 1 + o ( 1 ) )$ . It is attained by ${ \boldsymbol { a } } = \mathbf { 1 }$ with $\theta = t - 1 + \rho _ { t - 1 } / ( \rho _ { t - 1 } + \rho _ { t } )$ , and below it every gate-local rule has $\mathbb { E } X  \infty$

Proof sketch. With equal weights, class � has score $j ,$ box support $j \lambda _ { 1 } + ( r - j ) \lambda _ { 0 }$ and ball support ${ \sqrt { r } } \rho _ { j }$ . Add the constraints for $j = t - 1$ , � to optimize the bias. Classes farther from the boundary gain enough clean margin to stay slack. The converses follow from Theorems 4.2 and $5 . 4 ;$ details are in Appendix E.

At first order, start with the two clean scores beside the decision boundary. They are $t - 1$ and $t ,$ so their margins are $\theta - ( t - 1 )$ and $t - \theta$ and sum to one. Moving the bias transfers margin from one class to the other. To minimize the largest support-to-margin ratio, assign the margins in proportion to the two supports. For the ball, these are $\sqrt { r } \rho _ { t - 1 }$ and $\sqrt { r } \rho _ { t } ;$ for the box, they are $( t - 1 ) \lambda _ { 1 } + ( r - t + 1 ) \lambda _ { 0 }$ and $t \lambda _ { 1 } + ( r - t ) \lambda _ { 0 }$ . Adding the two inequalities gives the dimension, and balancing them gives the bias.

For pairwise conjunction this gives Table 1. The half-slack bias assigns more margin to negative classes because their inactive inputs have larger extremes.

Corollary 6.2 (One point, four limits). $I f \beta  1$ , all four thresholds of Theorem 6.1 equal $4 r ^ { 2 } k \lambda _ { 1 } ^ { 2 } ( 1 +$ $o ( 1 ) ) = 8 r ^ { 2 } k$ log $k \left( 1 + o ( 1 ) \right)$ $I f \beta \to \infty ,$ , in units of 2� log � they tend to (a) $( 2 r - 2 t + 1 ) ^ { 2 } .$ (b) $4 ( r - t + 1 ) ^ { 2 } ,$ (c) $4 r ( r - t + 1 )$ and $( d ) \ r ( { \sqrt { r - t } } + { \sqrt { r - t + 1 } } ) ^ { 2 }$ For conjunction these are $1 , 4 , 4 r , r ,$ so counting overprices reliability by a factor tending to � at both biases. For disjunction the midpoint prices (b) and (c) coincide for every $\beta .$

Corollary 6.3 (Computation versus recovery). Forfeature recovery the two criteria share the threshold $k ( \lambda _ { 1 } + \lambda _ { 0 } ) ^ { 2 } .$ , attained by $\begin{array} { r } { \theta = \frac { 1 } { 2 } + \frac { 1 } { 2 } \sigma _ { d } ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } ) , } \end{array}$ ; below it the probes $e _ { i }$ fail with every common bias, and every gate-local recovery rule has a diverging expected number of errors. A decoder that recovers every feature by such threshold probes and then applies exact Boolean conjunctions therefore needs this dimension, while direct canonical �-way conjunction needs $k ( ( 2 r - 1 ) \lambda _ { 1 } + \lambda _ { 0 } ) ^ { 2 }$ Their ratio $\left( ( 2 r - 1 + \beta ) / ( 1 + \beta ) \right) ^ { 2 }$ decreases from $r ^ { 2 }$ at $\beta = 1$ to 1 as $\beta \to \infty$

The ratio comes from the readout architecture. A common conjunction bias must separate the � smallest active reads from the $r - 1$ largest active reads plus one inactive read. It therefore pays for $2 r - 1$ active levels; recovery pays for one. $\mathrm { A t ~ } m \mathrm { ~ = ~ } k ^ { 4 }$ , recovery needs 18� log � dimensions and direct pairwise conjunction needs 50� log �. The midpoint pairwise rule needs 72� log � for reliability and 80� log � for the expected count.

Other decoders have diferent requirements. The Lasso recovers support from about $2 k \log ( m - k )$ measurements (Wainwright, 2009), and $\ell _ { 1 }$ minimization recovers a typical nonnegative sparse vector from about 2� log � � noiseless Gaussian measurements (Donoho <sup>and</sup> <sup>Tanner,</sup> <sup>2009).</sup> <sup>In</sup> <sup>the</sup> <sup>skewed</sup> <sup>regime</sup> <sup>both</sup> <sup>match</sup> <sup>2�</sup> <sup>log</sup> <sup>�</sup> <sup>to</sup> <sup>first</sup> <sup>order;</sup> <sup>when</sup> <sup>log</sup> <sup>�</sup> ∼ <sup>log</sup> <sup>�,</sup> the latter is � � log � . The constants here describe one threshold layer reading a fixed random code. Further comparisons, including asymmetric comparator gates, are in Appendix J.

## 7 The critical window

Near the threshold, a few extreme reads decide success. Their fluctuations have scales $1 / \lambda _ { 1 }$ and $1 / \lambda _ { 0 }$ in units of $\sigma _ { d }$ . For pairwise conjunction, define the quantile centers

$$
\begin{array} { r } { d _ { 0 } = 4 k \big ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } \big ) ^ { 2 } , } \\ { d _ { 0 } ^ { \mathrm { o p t } } = k \big ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } \big ) ^ { 2 } , } \end{array}\tag{14}
$$

and $\Psi _ { \beta } ( x ) = \mathbb { P } ( G _ { A } + G _ { B } / \beta \le x )$ for independent standard Gumbel variables $G _ { A } , G _ { B }$

Theorem 7.1 (Window profiles). Let $\mathcal { F }$ be the universalfamily of pairwise conjunctions, $k \to \infty$   
and $m / k \to \infty$ . Uniformly on compact sets of �:   
(i) the midpoint rule at $d = d _ { 0 } + 8 k ( 1 + \beta ) \mathfrak { x }$ � has $\mathbb { P } ( X = 0 ) = \Psi _ { \beta } ( x ) + o ( 1 ) ;$   
(ii) at $d = d _ { 0 } ^ { \mathrm { o p t } } + 2 k ( 3 + \beta ) x$ , the best deterministic common bias has   
<sup>sup P</sup>(<sup>�</sup> <sup>=</sup> <sup>0</sup>) <sup>=</sup> <sup>sup Ψ</sup>�(<sup>�</sup>) <sup>�</sup> (<sup>�</sup> − <sup>�</sup>) + <sup>�</sup>(<sup>1</sup>)<sup>,</sup>   
� <sup>�</sup>∈<sup>R</sup>   
where $F _ { + }$ is the distribution function of log $\dot { \omega } _ { 1 } ( \omega _ { 1 } + \omega _ { 2 } ) ]$ for independent unit exponentials   
$\omega _ { 1 } , \omega _ { 2 }$ , and the half-slack bias of Theorem $6 . 1 ( a )$ has $\mathbb { P } ( X = 0 ) = \Psi _ { \beta } ( x / 2 ) F _ { + } ( x / 2 ) + o ( 1 )$

Proof sketch. The midpoint rule is controlled by the largest active and inactive reads. The optimized bias must also protect the two smallest active reads. Gaussian upper and lower extremes become independent, giving Gumbel variables and the first two points of a unit Poisson process. The bias divides the available slack between these constraints. Common-scale and centering errors vanish in the window units (Appendix F).

The two factors in the optimized profile have separate roles. The one-active negative class is protected when the sum of the largest active read and the largest inactive read stays below the bias; this gives $\Psi _ { \beta }$ . The positive class is protected when the sum of the two smallest active reads stays above it; this gives $F _ { + }$ . Increasing the bias helps the first event and hurts the second. The variable � allocates the window slack between them. The half-slack rule splits it equally; the supremum chooses the allocation that maximizes the product.

The optimized window is narrower by $4 ( 1 + \beta ) / ( 3 + \beta ) \in [ 2 , 4 )$ . The half-slack bias is optimal to first order; a bias tuned to a finite-size success target can improve its limiting profile. The profiles have explicit integral forms:

$$
\Psi _ { \beta } ( x ) = \int _ { 0 } ^ { \infty } \exp ( - \tau - \tau ^ { - \beta } e ^ { - \beta x } ) d \tau ,
$$

and $\begin{array} { r } { F _ { + } ( x ) = e ^ { - \sqrt { c } } + \int _ { 0 } ^ { \sqrt { c } } e ^ { - c / s } d s } \end{array}$ with $c = e ^ { - x }$ . For $\beta = 1 , \Psi _ { 1 } ( x ) = 2 e ^ { - x / 2 } K _ { 1 } ( 2 e ^ { - x / 2 } )$ , where $K _ { 1 }$ is the modified Bessel function; as $\beta \to \infty , \Psi _ { \beta } ( x ) \to$ <sup>exp</sup>(−<sup>�−�</sup>)<sup>.</sup> <sup>At</sup> <sup>the</sup> <sup>centers,</sup> <sup>the</sup> <sup>midpoint</sup> profile is $\Psi _ { \beta } ( 0 )$ and the half-slack profile is $\Psi _ { \beta } ( 0 ) F _ { + } ( 0 )$ , with $F _ { + } ( 0 ) = 0 . 5 1 6$

At the midpoint, the number of inactive features that cause an error has a mixed Poisson limit. Along $\beta \to \beta _ { 0 } \in [ 1 , \infty )$ its intensity is $\Lambda _ { \infty } = e ^ { \beta _ { 0 } \left( G _ { A } - x \right) }$ . Thus $\mathbb { P } ( X = 0 ) \to \mathbb { E } e ^ { - \Lambda _ { \infty } } = \Psi _ { \beta _ { 0 } } ( x ) \in ( 0 , 1 )$ while $\mathfrak { j } \Lambda _ { \infty } = \infty$ and $\mathbb { E } X  \infty$ . Rare large active reads make many gates fail together. The full argument is retained in Appendix I.

![](images/f3f84f733c9423fd8cd6147d7f208bb8d0b7e37dd04e0719409ed3cb5052e8f4.jpg)

![](images/354b28f6fc117c4ed289443866bdba08c286ca7e3ab2e861956f14e5f47463ef.jpg)

![](images/d428a94647e825da833f749a970e96d666fbcb84169aad353cac927bed737a89.jpg)

![](images/327c45e1196468a3ac04a8634a342667fb556fd3dd4acd6a6b036fd4c4f0d412.jpg)  
Figure 2. Bias, gate type and the separation of reliability from expected errors, all at $k = 1 0 2 4 .$ (a) Three pairwise biases and feature recovery on shared draws at $m = k ^ { 4 } .$ (b) Four gates with the half-slack bias at $m = k ^ { 4 }$ . Here $d _ { \mathrm { b o x } } = k ( ( 2 t - 1 ) \hat { \lambda } _ { 1 } + ( 2 r - 2 t + 1 ) \hat { \lambda } _ { 0 } ) ^ { 2 } .$ , with $( r , t ) = ( 2 , 2 )$ in (a) and each gate’s own $( r , t )$ in (b). (c,d) The midpoint pairwise rule at $m = k ^ { 4 } , k ^ { 1 6 } , k ^ { 6 4 } .$ , normalized by $d _ { 0 }$ of (14). Points and bands in (a–c): 4096 exact draws per cell and pointwise 95% Wilson intervals. Dashed curves in (c): $\Psi _ { \hat { \beta } }$ from Theorem 7.1(i). Panel (d): quadrature of proposition 3.2; dotted lines mark the first-order ball centers.

## 8 Extensions

Three variations show how the dimension requirement depends on the model: supports chosen from the dictionary, non-Gaussian entries and multilayer computation. The proofs, circuit statements and encoding constructions are in Appendix H.

## 8.1 Adversarial supports

If the support can depend on the dictionary, the class cloud need not be a product of two samples. An adversary can select $k - 1$ columns that align with one probe. The resulting interference grows linearly in $k ,$ rather than as $\sqrt { k }$

Proposition 8.1 (Uniform correctness). Let $k $ ∞ and <sup>log</sup> $k = o ( \log m )$ , and use canonical pairwise probes with bias $3 / 2 .$ There are absolute constants $0 < c < C$ such that the network is correct for every support of size at most � with probability tending to one i $\begin{array} { r } { f d \geq C k ^ { 2 } \log m } \end{array}$ , and fails on some support of size � with probability tending to one if $d \le c k ^ { 2 }$ log � and $d \to \infty$

Related bounds for uniform feature recovery appear in (Garg et al., 2026; Ivanitskiy et al., 2026); random and worst-case supports also difer for sparse regression (Candès and Plan, 2009). The detailed comparisons are in Appendix G.

## 8.2 Other entry laws

The ball uses one-gate tails. For the midpoint pairwise network, their leading exponent is universal. Let the entries of $\sqrt { d } E$ be independent with a fixed centered, unit-variance, sub-Gaussian law, and let $q _ { k , d }$ be the probability that a midpoint pairwise read with one active input exceeds $3 / 2$

Theorem 8.2 (Universality of the one-gate exponent). $I f k  \infty$ and $d / k  \infty$ , then

$$
\log q _ { k , d } = - \frac { d } { 1 6 k } \biggl [ 1 + { \cal O } \biggl ( \frac { 1 } { k } + \frac { k } { d } \log \frac { d } { k } \biggr ) \biggr ] ,
$$

with constants depending on the entry law. Consequently the midpoint pairwise network has $\mathbb { E } X  0$ once $d \geq ( 1 6 + \varepsilon ) k \log ( k m )$ .

Exponential tilting of coordinate rows proves the result. Appendix H.2 gives the next term and sparse-entry extensions. The reliability law uses the Gaussian joint identity; its universality remains open.

## 8.3 Composition and unrestricted encodings

A later layer must store the outputs it reads: pairwise conjunction has $\binom { k } { 2 }$ active outputs. For a Gaussian stream in which each active output is written once, Theorem H.5 gives the suficient budget $d \geq ( 1 + \varepsilon ) \kappa _ { \operatorname* { m i n } } ^ { - 1 } K \log Q$ . Here $Q$ counts reads, � is the largest stream activity before a read, and $2 \kappa _ { \mathrm { m i n } }$ is the smallest squared clean margin of a unit-norm read. The proof applies a union bound to an ideal execution.

Without the linear dictionary, order � dimensions sufice for all pairwise conjunctions. In the skewed regime they are also necessary for polynomially many threshold units, by sign-pattern counting (Cover, 1965) (proposition H.6). The coeficients of $\textstyle \prod _ { l \in S } ( y - l ) ^ { 2 }$ give a 2�-dimensional nonlinear code, with large coeficients and small normalized margins (Barvinok, <sup>2010,</sup> <sup>Theorem</sup> <sup>15.2)</sup> <sup>(proposition</sup> <sup>H.7).</sup> <sup>Thus</sup> <sup>the</sup> <sup>Θ</sup>(<sup>�</sup> <sup>log</sup> <sup>�</sup>) <sup>scale</sup> <sup>describes</sup> <sup>accessibility</sup> <sup>from</sup> <sup>a</sup> random linear dictionary.

## 9 Exact simulations

The exact law samples a network without constructing �. Draw $H ,$ � active Gaussians and the few inactive order statistics required by lemma B.2. These order statistics can be sampled exactly by sequential uniform tail draws and transformed by $\Phi ^ { - 1 }$ <sup>.</sup> <sup>Each</sup> <sup>draw</sup> <sup>costs</sup> <sup>�</sup>(<sup>�</sup> <sup>log</sup> <sup>�</sup>)<sup>,</sup> independent of � and �.

Each simulation point uses 4096 independent draws and pointwise 95% Wilson intervals. Expected counts use quadrature of proposition 3.2 with exact unordered class sizes: rare draws can dominate the mean. The protocol and complete results are in Appendices J and L.

Bias and gate type At � = 1024, $m = 1 0 2 4 ^ { 4 }$ and $d = 1 . 3 5 k ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ^ { 2 }$ , the half-slack pairwise rule (Figure 2(a)) succeeds on 3984 of 4096 draws. The count-optimal bias succeeds on 3912, and the midpoint on 2. At the quantile center, the half-slack success frequencies for $k = 6 4 , 2 5 6 ,$ , 1024 are 0.143, 0.139, 0.155; the predicted limits are 0.155, 0.154, 0.154. For $\mathrm { A N D } _ { 2 } , \mathrm { A N D } _ { 3 } , \mathrm { O R } _ { 2 }$ and $\mathrm { M A J } _ { 3 }$ at $m = k ^ { 4 }$ , the transitions cluster near their box centers (Figure 2(b)). $\mathrm { A t } k = 1 0 2 4 ,$ success is at most 3.8% at ratio 0.9 and between 82.7% and 97.9% at 1.2. The curves steepen as � grows (Figure J.3).

Separation of the criteria Fix $k = 1 0 2 4$ and use the midpoint pairwise rule at $m = k ^ { 4 } , k ^ { 1 6 } , k ^ { 6 4 }$ (Figure $2 ( \mathrm { c } , \mathrm { d } ) )$ ). At the quantile box center, success frequencies 0.295, 0.324, 0.340 agree with profile values 0.298, 0.326, 0.346. The expected count crosses one at about 1.23, 1.45, 1.67 times $d _ { 0 } ,$ tracking the first-order ball positions 1.11, 1.36, 1.60. $\mathrm { A t } \ m = k ^ { 6 4 }$ and $d = 1 . 5 d _ { 0 }$ , all 4096 draws are correct, yet quadrature gives $\mathbb { E } X \approx 2 . 8 \times 1 0 ^ { 1 9 }$ . Here the network is correct on every sampled draw while its expected error count remains large. The exact sampler reaches these dictionary sizes by sampling extremes rather than constructing the dictionary.

## 10 Discussion

Expected error counts use marginal tails and give a ball. Joint reliability uses shared reads through a Gaussian supremum; for the universal family, its geometry is a box.

The converse for reliability leaves a question about the readout class: can gate-local probes adapted to their own Gram matrices improve on common representations? Structured gate families raise a related geometric problem. The disjoint and partial-sharing examples suggest looking for sharp supremum constants; generic chaining gives the order (Talagrand, 2014), while a variational description may resemble the generalized random energy model (Derrida, 1985). Reliability for non-Gaussian, correlated or learned dictionaries is another question. For general gates, a window law would require the relevant extreme order statistics. For circuits, a box law would also need to track the dependence created by earlier writes; the present composition bound controls reads by a union bound.

## AI Use Statement

The original research ideas and initial manuscript draft were developed by the authors. AI tools were used to assist with checking mathematical details and the correctness of arguments, filling in routine proof steps, coding assistance, and language polishing. The authors take full responsibility for the content and correctness of the final manuscript.

## References

Micah Adler and Nir Shavit. On the complexity of neural computation in superposition. arXiv:2409.15318, 2024. URL https://arxiv.org/abs/2409.15318.

Richard Arratia, Larry Goldstein, and Louis Gordon. Two moments sufice for Poisson approximations: The Chen–Stein method. The Annals of Probability, 17(1):9–25, 1989. doi: 10.1214/aop/1176991491.

A. D. Barbour, Lars Holst, and Svante Janson. Poisson Approximation, volume 2 of Oxford Studies in Probability. Oxford University Press, Oxford, 1992. doi: 10.1093/oso/9780198522355.001.0001.

Andrew R. Barron and Antony Joseph. Analysis of fast sparse superposition codes. In 2011 IEEE International Symposium on Information Theory Proceedings, pages 1772–1776, 2011. doi: 10.1109/ISIT. 2011.6033853.

Alexander Barvinok. MATH 669: Combinatorics of polytopes, 2010. URL https://sites.lsa.umich. edu/barvinok/wp-content/uploads/sites/1434/2025/05/polynotes669.pdf. Winter 2010 lecture notes, University of Michigan, Theorem 15.2.

Jai Bhagat, Sara Molas-Medina, Giorgi Giglemiani, and Stefan Heimersheim. Compressed computation is (probably) not computation in superposition. arXiv:2606.14673, 2026. URL https://arxiv.org/ abs/2606.14673.

Christer Borell. The Brunn–Minkowski inequality in Gauss space. Inventiones Mathematicae, 30(2): 207–216, 1975. doi: 10.1007/BF01425510.

Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration Inequalities: A Nonasymptotic Theory ofIndependence. Oxford University Press, Oxford, 2013. doi: 10.1093/acprof:oso/9780199535255. 001.0001.

R. Creighton Buck. Partition of space. The American Mathematical Monthly, 50(9):541–544, 1943. doi: 10.1080/00029890.1943.11991447.

Emmanuel J. Candès and Yaniv Plan. Near-ideal model selection by ℓ<sub>1</sub> minimization. The Annals of Statistics, 37(5A):2145–2177, 2009. doi: 10.1214/08-AOS653.

Emmanuel J. Candès and Terence Tao. Near-optimal signal recovery from random projections: Universal encoding strategies? IEEE Transactions on Information Theory, 52(12):5406–5425, 2006. doi: 10.1109/ TIT.2006.885507.

Thomas M. Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. IEEE Transactions on Electronic Computers, EC-14(3):326–334, 1965. doi: 10.1109/PGEC.1965.264137.

Bernard Derrida. Random-energy model: An exactly solvable model of disordered systems. Physical Review B, 24(5):2613–2626, 1981. doi: 10.1103/PhysRevB.24.2613.

Bernard Derrida. A generalization of the Random Energy Model which includes correlations between energies. Journal de Physique Lettres, 46(9):401–407, 1985. doi: 10.1051/jphyslet:01985004609040100.

David L. Donoho. Compressed sensing. IEEE Transactions on Information Theory, 52(4):1289–1306, 2006. doi: 10.1109/TIT.2006.871582.

David L. Donoho and Jared Tanner. Counting faces of randomly projected polytopes when the projection radically lowers dimension. Journal of the American Mathematical Society, 22(1):1–53, 2009. doi: 10.1090/S0894-0347-08-00600-0.

Hanna Döring, Sabine Jansen, and Kristina Schubert. The method of cumulants for the normal approximation. Probability Surveys, 19:185–270, 2022. doi: 10.1214/22-PS7. arXiv:2102.01459.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. Transformer Circuits Thread (Anthropic), 2022. URL https://transformer-circuits.pub/2022/ toy\_model/index.html. arXiv:2209.10652.

William S. Evans and Leonard J. Schulman. Signal propagation and noisy circuits. IEEE Transactions on Information Theory, 45(7):2367–2373, 1999. doi: 10.1109/18.796377.

Xavier Fernique. Régularité des trajectoires des fonctions aléatoires gaussiennes. In École d’Été de Probabilités de Saint-Flour IV—1974, volume 480 of Lecture Notes in Mathematics, pages 1–96. Springer, Berlin, 1975. doi: 10.1007/BFb0080190.

Alyson K. Fletcher, Sundeep Rangan, and Vivek K. Goyal. Necessary and suficient conditions for sparsity pattern recovery. IEEE Transactions on Information Theory, 55(12):5758–5772, 2009. doi: 10.1109/TIT.2009.2032726.

Nikhil Garg, Jon Kleinberg, and Kenny Peng. How many features can a language model store under the linear representation hypothesis? In Steve Hanneke and Tor Lattimore, editors, Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 5358–5376. PMLR, 2026. URL https://proceedings.mlr.press/v336/garg26a.html.

Kaarel Hänni, Jake Mendel, Dmitry Vaintrob, and Lawrence Chan. Mathematical models of computation in superposition. ICML 2024 Mechanistic Interpretability Workshop; arXiv:2408.05451, 2024. URL https://arxiv.org/abs/2408.05451.

Johan Håstad. On the size of weights for threshold gates. SIAM Journal on Discrete Mathematics, 7(3): 484–492, 1994. doi: 10.1137/S0895480192235878.

Michael I. Ivanitskiy, John Jasper, Emily J. King, and Dustin G. Mixon. Towards a mathematical theory of superposition. arXiv:2608.27540, 2026. URL https://arxiv.org/abs/2608.27540.

Antony Joseph and Andrew R. Barron. Fast sparse superposition codes have near exponential error probability for � < . IEEE Transactions on Information Theory, 60(2):919–942, 2014. doi: 10.1109/TIT.2013.2289865.

Béatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. The Annals ofStatistics, 28(5):1302–1338, 2000. doi: 10.1214/aos/1015957395.

M. R. Leadbetter, Georg Lindgren, and Holger Rootzén. Extremes and Related Properties of Random Sequences and Processes. Springer Series in Statistics. Springer-Verlag, New York, 1983. doi: 10.1007/ 978-1-4612-5449-2.

Robert J. McEliece, Edward C. Posner, Eugene R. Rodemich, and Santosh S. Venkatesh. The capacity of the Hopfield associative memory. IEEE Transactions on Information Theory, 33(4):461–482, 1987. doi: 10.1109/TIT.1987.1057328.

Saburo Muroga. Threshold Logic and Its Applications. Wiley-Interscience, New York, 1971. ISBN 978-0-471-62530-8.

Adam Newgas. Compressed computation: Dense circuits in a toy model of the universal-AND problem. arXiv:2507.09816, 2025. URL https://arxiv.org/abs/2507.09816.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 5(3), 2020. doi: 10.23915/distill.00024.001.

Valentin V. Petrov. Sums of Independent Random Variables, volume 82 of Ergebnisse der Mathematik und ihrer Grenzgebiete. Springer-Verlag, Berlin, 1975. doi: 10.1007/978-3-642-65809-9.

Tony A. Plate. Holographic reduced representations. IEEE Transactions on Neural Networks, 6(3):623–641, 1995. doi: 10.1109/72.377968.

Sidney I. Resnick. Extreme Values, Regular Variation, and Point Processes, volume 4 of Applied Probability. Springer-Verlag, New York, 1987.

Dai Shi, Xiaoyu Li, Andi Han, and José Miguel Hernández-Lobato. Feature superposition in neural networks: From theory to practice. arXiv:2609.06862, 2026. URL https://arxiv.org/abs/2609. 06862.

David Slepian. The one-sided barrier problem for Gaussian noise. Bell System Technical Journal, 41(2): 463–501, 1962. doi: 10.1002/j.1538-7305.1962.tb02419.x.

Vladimir N. Sudakov. Gaussian random processes and measures of solid angles in Hilbert space. Soviet Mathematics Doklady, 12:412–415, 1971.

Michel Talagrand. Upper and Lower Bounds for Stochastic Processes: Modern Methods and Classical Problems, volume 60 of Ergebnisse der Mathematik und ihrer Grenzgebiete. 3. Folge. Springer, Berlin, 2014. doi: 10.1007/978-3-642-54075-2.

Anthony Thomas, Sanjoy Dasgupta, and Tajana Rosing. A theoretical perspective on hyperdimensional computing. Journal of Artificial Intelligence Research, 72:215–249, 2021. doi: 10.1613/jair.1.12664.

Boris S. Tsirel’son, Ildar A. Ibragimov, and Vladimir N. Sudakov. Norms of Gaussian sample functions. In Proceedings of the Third Japan–USSR Symposium on Probability Theory, volume 550 of Lecture Notes in Mathematics, pages 20–41. Springer, Berlin, 1976. doi: 10.1007/BFb0077482.

Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press, Cambridge, 2018. doi: 10.1017/9781108231596.

John von Neumann. Probabilistic logics and the synthesis of reliable organisms from unreliable components. In Claude E. Shannon and John McCarthy, editors, Automata Studies, number 34 in Annals of Mathematics Studies, pages 43–98. Princeton University Press, 1956. doi: 10.1515/ 9781400882618-003.

Martin J. Wainwright. Sharp thresholds for high-dimensional and noisy sparsity recovery using ℓ<sub>1</sub>- constrained quadratic programming (Lasso). IEEE Transactions on Information Theory, 55(5):2183–2202, 2009. doi: 10.1109/TIT.2009.2016018.

Robert O. Winder. Partitions of �-space by hyperplanes. SIAM Journal on Applied Mathematics, 14(4): 811–818, 1966. doi: 10.1137/0114068.

## Appendix

The Ball and the Box

## A Guide to the appendix

The article keeps the main threshold statements and the proof of the exact read law. The appendix gives the read-vector reductions, the supremum proof, the box identity and its optimization program (Appendix B); the remaining ball, box, gate and window proofs (Appendices C to F); and the full extensions of the model (Appendices G and H). Additional window formulas, comparisons and experimental results are in Appendices I and J. Appendix K records finite computational checks; Appendix L gives the frozen experimental protocol. Appendix M turns the thresholds into design rules for width, capacity, sizing and evaluation, and states open problems. The notation table and dependency map provide a reading guide.

![](images/6d9af10af159ea5fb957af0ab6696a6e312a6f7d58eeaeac1311b1f1d9d61dcf.jpg)  
Figure A.1. Dependency map. An arrow points from a result to one that builds on it. Blue: expected errors (the ball); orange: reliability (the supremum and the box); grey: the exact law and its reduction. Small italic text names the classical tool each proof adds.

Table A.1. Notation.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $E = \left[ e _ { 1 } \right| \cdots \left| e _ { m } \right]$ </td><td>dictionary, independent columns  $e _ { i } \sim N ( 0 , I _ { d } / d )$ </td></tr><tr><td> $S , x = \mathbf { 1 } _ { S } , k , n = m - k$ </td><td>active set (uniform k-subset), input, sparsity, number of inactive</td></tr><tr><td> $u = E x , R = E ^ { \top } u$ </td><td>features stored vector and feature reads;  $A _ { i } = R _ { i } ( i \in S ) , B _ { j }$  (inactive reads)</td></tr><tr><td> $H , Z , W , \sigma$ </td><td> $H \sim \chi _ { d } ^ { 2 } / d ,$  independent Gaussian samples of sizes k and  $n , \sigma =$   $\sqrt { k H / d }$  (Theorem 3.1)</td></tr><tr><td> $\sigma _ { d } = \sqrt { k / d }$ </td><td>deterministic interference scale</td></tr><tr><td> $\xi , \xi _ { I }$ </td><td> $\xi _ { i } = Z _ { i } - \bar { Z }$  on S and  $W _ { j }$  off  $S ;$  interference vector of a tuple I</td></tr><tr><td> $( I , g ) , r , \mathcal { F } , \mathcal { U } _ { r } ( g )$ </td><td>gate (tuple and threshold function), arity, gate family, universal family</td></tr><tr><td> $z , { \mathcal { F } } _ { z } ( S ) , N _ { z }$ </td><td>class (input pattern), gates of class  $z ,$  expected class size (3)</td></tr><tr><td> $( a , \theta ) , \operatorname { R e p } ( g )$ </td><td>weights and common bias of the canonical rule; representations of  $g$ </td></tr><tr><td> $s _ { z } ( a , \theta ) , \varsigma _ { z }$ </td><td>class margin (4); error direction  $1 - 2 g ( z )$ </td></tr><tr><td>X</td><td>number of wrong outputs</td></tr><tr><td> $\lambda _ { 1 } , \lambda _ { 0 } , \lambda _ { z } , \beta$ </td><td>levels  ${ \sqrt { 2 \log k } } , { \sqrt { 2 \log n } } ,$  their vector for class  $z ,$  ratio  $\lambda _ { 0 } / \lambda _ { 1 }$ </td></tr><tr><td> $\rho _ { z } = { \sqrt { 2 \log N _ { z } } }$ </td><td>ball radius of class  $z ; \rho _ { j }$  in Section 6 for classes with j active inputs</td></tr><tr><td> $\mathsf { B a l l } _ { z } , \mathsf { B o x } _ { z } , h _ { K }$ </td><td>ball and box of Definition 2.1; support function of a set K</td></tr><tr><td> $D _ { \mathsf { b a l l } } , D _ { \mathsf { b o x } } , V _ { \mathsf { b a l l } } , V _ { \mathsf { b o x } }$ </td><td>ball and box prices (7) and their optimal values</td></tr><tr><td> $\Gamma _ { z } ( S ) , d ^ { * } ( S )$ </td><td>expected supremum of the class process (12); reliability threshold of</td></tr><tr><td></td><td>Theorem 5.1</td></tr><tr><td> $\mathrm { T H R } _ { t } ^ { r }$ </td><td>threshold gate firing when at least t of r inputs are active</td></tr><tr><td> $\hat { \lambda } _ { 1 } , \hat { \lambda } _ { 0 }$ </td><td> $\overline { { \Phi } } ^ { - 1 } ( 1 / k )$   $\overline { { \Phi } } ^ { - 1 } ( 1 / n )$  quantile levels and</td></tr><tr><td> $\Psi _ { \beta } , F _ { + }$ </td><td>window profiles of Theorem  $7 . 1$ </td></tr><tr><td> $G _ { A } , G _ { B } , \omega _ { 1 } , \omega _ { 2 }$ </td><td>standard Gumbel and unit exponential variables of Theorem 7.1</td></tr><tr><td> $D , C ( i ) , \lambda _ { D }$ </td><td>number of partners, partner set and level  $\sqrt { 2 \log D }$  of proposition 5.3</td></tr><tr><td> $Q , K , \kappa _ { \mathrm { m i n } }$ </td><td>number of reads, stream activity and margin rate of Theorem H.5</td></tr></table>

## B Read-vector reductions and the supremum

The following statements and proofs give the reductions used in Sections 3 and 5.

## B.1 Reliability as a support function

<table><tr><td>Proposition B.1 (Reliability as a support function). For the canonical rule with  $( a , \theta ) \in \operatorname { R e p } ( g )$  and any gate family, let  $C _ { z } = \{ \xi _ { I } : I \in \mathcal { F } _ { z } ( S ) \}$  and</td></tr><tr><td> $M _ { z } = \operatorname* { m a x } _ { I \in \mathcal { F } _ { z } ( S ) } \langle \varsigma _ { z } a , \xi _ { I } \rangle = h _ { C _ { z } } ( \varsigma _ { z } a ) , \qquad s _ { z } ^ { H } = ( 2 g ( z ) - 1 ) \big ( H \langle a , z \rangle - \theta \big ) ,$ </td></tr><tr><td></td></tr><tr><td>with max  $\varnothing = - \infty$  For every realization,  $\begin{array} { r } { \bigcap _ { z } \{ \sigma M _ { z } < s _ { z } ^ { H } \} \subseteq \{ X = 0 \} \subseteq \bigcap _ { z } \{ \sigma M _ { z } \leq s _ { z } ^ { H } \} } \end{array}$  , and for a ixed representation the two outer events differ by an event of probability zero.</td></tr></table>

Proof. A gate of class � is correct when $( 2 g ( z ) - 1 ) ( \langle a , R _ { I } \rangle - \theta ) > 0$ , that is, when $\sigma \langle \varsigma _ { z } a , \xi _ { I } \rangle < s _ { z } ^ { H } .$ and it may also be correct when its score equals $\theta ;$ this gives the two inclusions. For a fixed representation such ties have probability zero: given � the score is Gaussian, and when its conditional variance vanishes the score equals $H \langle a , z \rangle$ , which has a density if $\langle a , z \rangle \neq 0$ and equals $0 \neq \theta$ otherwise, because $s _ { z } > 0$ . Take the maximum over the class and intersect over the finitely many classes. ■

## B.2 Proof of the exact one-gate law

Proof. The coordinates of $( w , u )$ are independent pairs of centered Gaussians with variances $\| a \| ^ { 2 } / d$ and $k / d$ and covariance $\langle a , z \rangle / d$ . Hence $u \ = \ ( \langle a , z \rangle / \| a \| ^ { 2 } ) w + v$ with $v \sim N \left( 0 , \left( k - \right) \right)$ $\langle a , z \rangle ^ { 2 } / \| a \| ^ { 2 } ) I _ { d } / d \big )$ independent of $w ,$ and $\langle w , u \rangle = ( \langle a , z \rangle / \| a \| ^ { 2 } ) \| w \| ^ { 2 } + \langle w , v \rangle$ . Given $w ,$ the last term is centered Gaussian with variance $\begin{array} { r } { \| w \| ^ { 2 } ( k \| a \| ^ { 2 } - \langle a , z \rangle ^ { 2 } ) / ( d \| a \| ^ { 2 } ) . \mathrm { P u t } J = \| w \| ^ { 2 } / \| a \| ^ { 2 } } \end{array}$ . The formula for E� is linearity of expectation over gates, each gate contributing its class error probability with weight $\mathbb { P } ( z _ { \gamma } ( S ) = z )$ ■

## B.3 Proof of the supremum law

Proof. Work on the representation of Theorem 3.1 and let $G \in \mathbb { R } ^ { m }$ collect � on � and � of $S ;$ it is $N ( 0 , I _ { m } )$ and independent of �. For a tuple of class $z , \langle a , \xi _ { I } \rangle = \langle a , G _ { I } \rangle - \bar { Z } \langle a , z \rangle$ , so $M _ { z } = T _ { z } - \varsigma _ { z } \bar { Z } \langle a , z \rangle$ with $\begin{array} { r } { T _ { z } = \mathrm { m } \mathrm { a x } _ { I \in \mathcal { F } _ { z } ( S ) } \langle \varsigma _ { z } a , G _ { I } \rangle } \end{array}$ . The map $G \mapsto T _ { z }$ is $\| a \| { \mathrm { - L i p s c h i t z } } ,$ because the coordinates of $G _ { I }$ are distinct coordinates of �. By Gaussian concentration (Borell, 1975; Tsirel’son et al., 1976),

$$
\begin{array} { r } { \mathbb { P } \big ( | T _ { z } - \Gamma _ { z } ( S ) | \ge y \mid S \big ) \le 2 e ^ { - y ^ { 2 } / ( 2 \| a \| ^ { 2 } ) } \qquad ( y > 0 ) . } \end{array}\tag{B.1}
$$

Moreover $| H - 1 | = O _ { \mathbb { P } } ( d ^ { - 1 / 2 } ) , | \bar { Z } | = O _ { \mathbb { P } } ( k ^ { - 1 / 2 } ) , \sigma = \sigma _ { d } \sqrt { H }$ and $s _ { z } ^ { H } = s _ { z } + ( 2 g ( z ) - 1 ) ( H - 1 ) \langle a , z \rangle$ with $\left| \langle a , z \rangle \right| \leq { \sqrt { r } } \| a \|$ . Because $( a , \theta )$ is fixed and $\sigma _ { d } \to 0 ,$ , every margin satisfies $s _ { z } / ( \sigma _ { d } \lVert a \rVert ) \to \infty$ Choose $L _ { d } $ with $\sigma _ { d } L _ { d } \to 0$ and $\delta _ { d } \to 0$ slowly, so that with probability tending to one $\left| H - 1 \right| \leq \delta _ { d } , \left| \bar { Z } \right| \leq \delta _ { d } ,$ , and, by $( \mathrm { B } . 1 ) , T _ { z } \leq \Gamma _ { z } ( S ) + L _ { d } \| a \|$ for all $z .$ .

On $\{ d \geq ( 1 + \varepsilon ) d ^ { * } ( S ) \}$ every class has $\sigma _ { d } \Gamma _ { z } ( S ) _ { + } \leq s _ { z } / \sqrt { 1 + \varepsilon }$ , and then

$$
\sigma M _ { z } \leq ( 1 + \delta _ { d } ) \Big ( \frac { s _ { z } } { \sqrt { 1 + \varepsilon } } + \sigma _ { d } \| a \| \big ( L _ { d } + \sqrt { r } \delta _ { d } \big ) \Big ) < s _ { z } - \delta _ { d } \sqrt { r } \| a \| \leq s _ { z } ^ { H }
$$

for all large $d ,$ because $\sigma _ { d } \| a \| ( L _ { d } + \sqrt { r } \delta _ { d } ) = o ( s _ { z } )$ and $\delta _ { d } \| a \| = o ( s _ { z } )$ . On $\{ d \leq ( 1 - \varepsilon ) d ^ { * } ( S ) \}$ some class has $\sigma _ { d } \Gamma _ { z } ( S ) \geq s _ { z } / \sqrt { 1 - \varepsilon } , \ s o \ \Gamma _ { z } ( S ) / \| a \| \to \infty$ and (B.1) with $y ~ = ~ { \textstyle \frac { \varepsilon } { 4 } } \Gamma _ { z } ( S )$ gives $T _ { z } \geq ( 1 - { \frac { \varepsilon } { 4 } } ) \Gamma _ { z } ( S )$ with probability tending to one. For this class

$$
\sigma M _ { z } \geq ( 1 - \delta _ { d } ) \bigg ( \frac { ( 1 - \varepsilon / 4 ) s _ { z } } { \sqrt { 1 - \varepsilon } } - \sigma _ { d } \sqrt { r } \| a \| \delta _ { d } \bigg ) > s _ { z } + \delta _ { d } \sqrt { r } \| a \| \geq s _ { z } ^ { H } ,
$$

since $( 1 - \varepsilon / 4 ) / \sqrt { 1 - \varepsilon } > 1$ . There are $2 ^ { r }$ classes, so both statements follow.

## B.4 The box identity

For the universal family the maximizing tuple can be written down. Group the positions of a class-� gate by the pool they draw from $( p = 1$ for the active features, $p = 0$ for the inactive ones) and by the sign of their weight in the direction $b = \varsigma _ { z } a$

Lemma B.2 (Box identity). Let $\mathcal { F } = \mathcal { U } _ { r } ( g )$ and assume that each pool � has at least as many features as the class has positions with $z _ { l } = p$ . For every $\xi \in \mathbb { R } ^ { m } , S ,$ � and $b \in \mathbb { R } ^ { r }$

$$
\operatorname* { m a x } _ { I \in { \mathcal { F } _ { z } } \left( S \right) } \left. b , \xi _ { I } \right. = \sum _ { p \in \left\{ 0 , 1 \right\} } \sum _ { \pm } \sum _ { j = 1 } ^ { c _ { p } ^ { \pm } } \bigl | b \bigr | _ { [ j ] } ^ { \left( p , \pm \right) } \left( \pm \xi \right) _ { ( j ) } ^ { \left( p \right) } ,
$$

where $c _ { p } ^ { + }$ and $c _ { p } ^ { - }$ count the positions � with $z _ { l } = p$ and $b _ { l } \geq 0 .$ , respectively $b _ { l } < 0 , | b | _ { [ 1 ] } ^ { ( p , \pm ) } \geq$ $| b | _ { [ 2 ] } ^ { ( p , \pm ) } \geq \cdots$ are their absolute weights in decreasing order, and $( \pm \xi ) _ { ( 1 ) } ^ { ( p ) } \geq ( \pm \xi ) _ { ( 2 ) } ^ { ( p ) } \geq \cdot \cdot \cdot$ are the values $\pm \xi _ { i } ,$ � in pool $p ,$ in decreasing order.

Proof. For any admissible tuple, split $\langle b , \xi _ { I } \rangle$ into the four groups. Within a group the features are distinct members of one pool, so by the rearrangement inequality the group sum is at most the displayed sum of products of decreasing weights with decreasing values of $\pm \xi .$ . The bound is attained by placing, in each pool, the $c _ { p } ^ { + }$ largest values on the positions with $b _ { l } \geq 0$ and the $c _ { p } ^ { - }$ smallest on those with $b _ { l } < 0 ;$ these features are distinct because the pool has at least $c _ { p } ^ { + } + c _ { p } ^ { - }$ elements, which is the assumption. ■

Each group sum involves at most � extreme order statistics of one Gaussian sample, and each of these is $\lambda _ { p } ( 1 + o _ { \mathbb { P } } ( 1 ) )$ . Hence the maximal projection of the class is $\begin{array} { r } { \sum _ { l } | a _ { l } | \lambda _ { z _ { l } } = h _ { \mathsf { B o x } _ { z } } ( a ) } \end{array}$ to first order: the cloud fills the box. Because the relative error comes from finitely many order statistics that do not depend on $^ { a , }$ the approximation is uniform over all representations at once.

## B.5 Computing the box program

Proposition B.3 (Computing the box). For each $\kappa \geq 0$ the set $o f ( a , \theta )$ with $\begin{array} { r } { s _ { z } ( a , \theta ) \geq \kappa \sum _ { l } \vert a _ { l } \vert \beta ^ { 1 - z _ { l } } } \end{array}$ for all � is a convex polyhedral cone, so $v _ { g } ( \beta ) ^ { - 1 / 2 }$ is the largest �for which this cone contains a point with $s _ { z } \geq 1$ for all $z ;$ after introducing $u _ { l } \geq \pm a _ { l }$ this is a linearfeasibility problem, and � isfound by bisection. $I f g$ is symmetric, both $V _ { \mathsf { b o x } }$ and $V _ { \mathsf { b a l l } }$ are attained at ${ \boldsymbol { a } } = \mathbf { 1 }$ when $g$ is nondecreasing, as for $\mathrm { T H R } _ { t } ^ { r } .$ , and at $a = - 1$ when $g$ is nonincreasing; every symmetric threshold function is one of the two.

Proof. Each constraint says that a linear function minus � times a convex piecewise-linear function is nonnegative, which defines a convex polyhedral set; homogeneity makes it a cone. If $g$ is symmetric, permuting the coordinates of � permutes the classes and preserves the constraint system of both programs, so averaging a feasible point over all permutations gives a feasible point with equal weights. Its margins are averages of margins over permutation orbits, hence positive, so its common weight � is nonzero, with $c > 0 { \mathrm { i f } } \ g$ is nondecreasing and $c < 0$ if $g$ is nonincreasing; rescale to $a = \pm 1$ ■

## B.6 Partial sharing

For nonnegative weights, two gates of the same class that share a feature have positively correlated interference, and every additional shared feature raises the correlation. By the

Sudakov–Fernique inequality (Sudakov, 1971; Fernique, 1975) the expected supremum of a class can only fall below that of independent gates with the same variances, which is the ball. The universal family is the opposite extreme to disjoint gates: every tuple of features is a gate, and the supremum splits into a sum of independent maxima, one for each position. Between the two extremes the supremum depends on how much sharing there is, and a simple family already shows where the supremum leaves the ball.

The partial-sharing statement is given in proposition 5.3.

Proof sketch (details in Appendix D.2). The class supremum is $\mathrm { m a x } _ { i \in S } ( Z _ { i } + M _ { i } )$ with $M _ { i } =$ $\begin{array} { r } { \operatorname* { m a x } _ { j \in C ( i ) } W _ { j } , } \end{array}$ a maximum of � independent copies of a Gaussian plus a maximum of � Gaussians. When $D \leq k _ { * }$ , many active features have moderately large reads, and among them one finds a partner with an equally large read: the two coordinates split the budget 2 log �� evenly, as in the ball. When $D \geq k$ the shared coordinate saturates at $\lambda _ { 1 }$ and the partner adds $\lambda _ { D }$ □

Sharing therefore lowers the price of a class only once every active feature has more partners than there are active features. The two formulas agree at $D = k$ . Disjoint partner sets require $k D \leq n ;$ continued formally to $D = n$ , where partner sets overlap as in the universal family, the second formula becomes the box value $\lambda _ { 1 } + \lambda _ { 0 }$ of Theorem 5.4(iii). Partner sets drawn at random, as in a random regular graph of gates, satisfy the disjointness hypothesis with probability tending to one when $k ^ { 2 } \bar { D ^ { 2 } } = \overset { \bar { } } { \underset { } { o } } ( m )$

## C Proofs for the ball

Throughout, ${ \overline { { \Phi } } } = 1 - \Phi$ is the standard normal tail, and we use the two Mills bounds

$$
\overline { { \Phi } } ( x ) \leq e ^ { - x ^ { 2 } / 2 } \quad ( x \geq 0 ) , \qquad \overline { { \Phi } } ( x ) \geq \frac { x } { 1 + x ^ { 2 } } \frac { e ^ { - x ^ { 2 } / 2 } } { \sqrt { 2 \pi } } \quad ( x > 0 ) ,\tag{C.1}
$$

and the chi-square bounds of Laurent and Massart (2000): for $J \sim \chi _ { d } ^ { 2 } / d$ and $0 < \eta \leq 1$

$$
\mathbb { P } ( | J - 1 | \ge \eta ) \le 2 e ^ { - d \eta ^ { 2 } / 8 } , \qquad \mathrm { V a r } J = 2 / d .\tag{C.2}
$$

For a representation $( a , \theta )$ of $^ { g , }$ the bias lies strictly between the largest rejected and the smallest accepted clean value, and every clean value has modulus at most $\| a \| _ { 1 } \leq { \sqrt { r } } \| a \| ;$ ; hence

$$
0 < s _ { z } ( a , \theta ) \leq 2 \sqrt { r } \left\| a \right\| \qquad \mathrm { f o r ~ e v e r y ~ } z .\tag{C.3}
$$

## C.1 Proof of Theorem 4.1

Fix a class � and abbreviate $s = s _ { z } ( a , \theta ) , c = \langle a , z \rangle$ and $v ^ { 2 } = k \| a \| ^ { 2 } – c ^ { 2 }$ . Since $c ^ { 2 } \leq | z | \| a \| ^ { 2 } \leq r \| a \| ^ { 2 } .$ we have $( k - r ) \| a \| ^ { 2 } \leq v ^ { 2 } \leq k \| a \| ^ { 2 }$ . By proposition 3.2 and $\varsigma _ { z } ( c - \theta ) = - s ,$ , a gate of class � errs exactly when $v \sqrt { J / d } G ^ { \prime } \geq s - \varsigma _ { z } c ( J - 1 )$ with $G ^ { \prime } = \varsigma _ { z } G \sim N ( 0 , 1 )$ independent of $J ,$ so

$$
p _ { z } = \mathbb { E } \overline { { \Phi } } \Big ( \frac { s - \varsigma _ { z } c ( J - 1 ) } { v \sqrt { J / d } } \Big ) .\tag{C.4}
$$

It sufices to prove each claim along a subsequence of an arbitrary subsequence.

Upper bound. Let $d \ge ( 1 + \varepsilon ) D _ { \mathsf { b a l l } } ( a , \theta )$ , so that $d s ^ { 2 } / ( k \Vert a \Vert ^ { 2 } ) \geq ( 1 + \varepsilon ) \rho _ { z } ^ { 2 }$ for every �. Put $\eta = s / ( 2 \sqrt { r } \| a \| k ^ { 1 / 4 } )$ , which is at most $k ^ { - 1 / 4 }$ by (C.3). On $\{ | J - 1 | \leq \eta \}$ we have $| c ( J - 1 ) | \leq$ $\textstyle { \sqrt { r } } \| a \| \eta = { \frac { 1 } { 2 } } s k ^ { - 1 / 4 }$ and $v \sqrt { J / d } \leq \| a \| \sqrt { k ( 1 + \eta ) / d } ,$ , so the argument of $\overline { { \Phi } }$ in (C.4) is at least

$$
x _ { z } = \frac { s ( 1 - \frac { 1 } { 2 } k ^ { - 1 / 4 } ) } { \| a \| \sqrt { k ( 1 + k ^ { - 1 / 4 } ) / d } } , \qquad x _ { z } ^ { 2 } \geq ( 1 + \varepsilon ) ( 1 - 3 k ^ { - 1 / 4 } ) \rho _ { z } ^ { 2 } .
$$

By (C.1) this part contributes at most $N _ { z } ^ { - ( 1 + \varepsilon ) ( 1 - 3 k ^ { - 1 / 4 } ) } \leq N _ { z } ^ { - 1 - \varepsilon / 2 }$ for large �. By (C.2) the complement has probability at most

$$
2 \exp \Bigl ( - \frac { d s ^ { 2 } } { 3 2 r \| a \| ^ { 2 } k ^ { 1 / 2 } } \Bigr ) \le 2 \exp \Bigl ( - \frac { ( 1 + \varepsilon ) k ^ { 1 / 2 } } { 3 2 r } \rho _ { z } ^ { 2 } \Bigr ) = 2 N _ { z } ^ { - ( 1 + \varepsilon ) k ^ { 1 / 2 } / ( 1 6 r ) } .
$$

Hence $N _ { z } p _ { z } \leq N _ { z } ^ { - \varepsilon / 2 } + 2 N _ { z } ^ { 1 - k ^ { 1 / 2 } / ( 1 6 r ) }  0$ for every class, and $\begin{array} { r } { \mathbb { E } X = \sum _ { z } N _ { z } p _ { z } \to 0 } \end{array}$ because there are $2 ^ { r }$ classes.

Lower bound. Let $d \leq ( 1 - \varepsilon ) D _ { \mathsf { b a l l } } ( a , \theta )$ and pass to a subsequence along which one class � attains the maximum in $D _ { \mathsf { b a l l } } ;$ then $x _ { 0 } ^ { 2 } : = d s ^ { 2 } / ( k | | \boldsymbol { \hat { a } } | | ^ { 2 } ) \leq ( 1 - \varepsilon ) \rho _ { z } ^ { 2 }$ . Pass to a further subsequence along which either $x _ { 0 } $ or $x _ { 0 }$ stays bounded.

If $x _ { 0 }  \infty ,$ put $\eta = s / ( \sqrt { r } \| a \| \log k ) \leq 2 / \log k$ . By (C.3), $d \ge k x _ { 0 } ^ { 2 } / ( 4 r ) \to \infty ,$ , and $d \eta ^ { 2 } =$ $k x _ { 0 } ^ { 2 } / ( r \log ^ { 2 } k ) \to \infty ,$ so $\mathbb { P } ( | J - 1 | \le \eta ) \to 1$ . On this event $| c ( J - 1 ) | \leq s / \log k$ and $v \sqrt { J / d } \geq$ $\| a \| \sqrt { ( k - r ) ( 1 - \eta ) / d }$ , so the argument of $\overline { { \Phi } }$ is at most $x _ { 1 } = x _ { 0 } ( 1 + 1 / \log k ) \{ ( 1 - r / k ) ( 1 - \eta ) \} ^ { - 1 / 2 }$ with $x _ { 1 } ^ { 2 } \le ( 1 - \varepsilon ) ( 1 + o ( 1 ) ) \rho _ { z } ^ { 2 }$ . The lower Mills bound gives $p _ { z } \geq ( 1 - o ( 1 ) ) N _ { z } ^ { - ( 1 - \varepsilon ) - o ( 1 ) }$ , and $N _ { z } p _ { z } \ge N _ { z } ^ { \varepsilon - o ( 1 ) } \to \infty$

If $x _ { 0 }$ stays bounded, consider first subsequences with $d \to \infty$ . By Chebyshev’s inequality $\mathbb { P } ( | J - 1 | \leq 2 \sqrt { 2 / d } ) \geq 3 / 4 ;$ on this event $| c ( J - 1 ) | \leq 2 \sqrt { 2 r } \| a \| / \sqrt { d }$ and $v \sqrt { J / d } \geq \| a \| \sqrt { ( k - r ) / ( 2 d ) }$ for $d \ge 3 2$ , so the argument of $\overline { { \Phi } }$ is at most $\sqrt { 2 } x _ { 0 } \sqrt { k / ( k - r ) } + 4 \sqrt { r / ( k - r ) }$ , which is bounded. Hence $p _ { z }$ is bounded below and $N _ { z } p _ { z }  \infty$ . If instead � stays bounded, then $\mathbb { P } ( J \in [ 1 / 2 , 2 ] )$ is bounded below, and on that event the argument of $\overline { { \Phi } }$ is at most $( s + { \sqrt { r } } \| a \| ) / ( \| a \| { \sqrt { ( k - r ) / ( 2 d ) } } ) \to 0$ by (C.3); again $p _ { z }$ is bounded below. In every case $\mathbb { E } X \geq N _ { z } p _ { z }  \infty$ □

## C.2 Proof of Theorem 4.2

The program. Write $L = \mathbf { m a x } _ { z }$ log $N _ { z }$ . A fixed nonconstant threshold function has a representation with positive margins, which gives $V _ { \sf b a l l } \leq C _ { g } L$ . Conversely, by (C.3) every representation with $\| a \| \leq 1$ has margins at most $2 { \sqrt { r } }$ , so $V _ { \sf b a l l } \geq L / ( 2 r ) ,$ ; thus $V _ { \sf b a l l } = \Theta _ { g } ( L )$ . In the cone form (11) the bias may be restricted to $| \theta | \leq { \sqrt { r } }$ and � to a bounded interval, so the feasible set is compact and the maximum $t \sb * > 0$ is attained; at a maximizer all margins are positive. This minimizer of (10) has $D _ { \mathsf { b a l l } } = k V _ { \mathsf { b a l l } }$ , and the first claim is Theorem 4.1.

Converse. Fix a gate � with tuple � and condition on $E _ { I }$ and on the independent randomness of its rule; these determine $( w , \theta _ { \gamma } )$ . Write $q _ { z } = N _ { z } / | \mathcal { F } |$ for the probability that $\gamma$ has class $z ,$ which is the same for every gate of the family. If $w = 0$ the output is constant and the gate errs with probability at least $\mathrm { m i n } _ { z } q _ { z }$ . Otherwise rescale $( w , \theta _ { \gamma } )$ so that $\lVert \boldsymbol { w } \rVert = 1$ , which does not change the output, and put $a = E _ { \scriptscriptstyle I } ^ { \scriptscriptstyle \mathsf { T } } w$ . Given the class $z ,$ the set $S \setminus I$ is a uniform $\left( k - | z | \right)$ -subset of the other features, whose columns are independent of $E _ { I }$ and $w ;$ hence the score is $\langle a , z \rangle + \zeta$ with $\zeta \sim N ( 0 , ( k - | z | ) / d )$ given $E _ { I } ,$ � and the class. Only the marginal law of the class of one gate enters, so no exchangeability of the rules and no independence between gates is needed.

Pass to subsequences along which either $d \to \infty$ or � stays bounded, and consider first $d \to \infty$ . Let $\eta _ { d } = d ^ { - 1 / 4 }$ and let $\mathcal { G } _ { \gamma } = \{ \| E _ { I } ^ { \top } E _ { I } - I _ { r } \| _ { \mathrm { o p } } \leq \eta _ { d } \}$ . Gram concentration for a fixed number � of Gaussian columns (Vershynin, 2018, Theorem 4.6.1) gives $\mathbb { P } ( \boldsymbol { \mathcal { G } } _ { \gamma } ^ { c } ) \leq C _ { r } e ^ { - c _ { r } d \boldsymbol { \eta } _ { d } ^ { 2 } } \to 0$ On $\begin{array} { r } { \mathcal { G } _ { \gamma } , \| a \| ^ { 2 } \leq \| E _ { I } ^ { \top } E _ { I } \| _ { \mathrm { o p } } \leq 1 + \eta _ { d } ; \mathrm { p u t } c = ( 1 + \eta _ { d } ) ^ { 1 / 2 } , \mathrm { s o } \| a / c \| \leq 1 } \end{array}$

If $( a / c , \theta _ { \gamma } / c )$ does not represent $^ { g , }$ some class � has $s _ { z } ( a , \theta _ { \gamma } ) \leq 0 .$ , and since $\zeta$ is symmetric the gate errs in that class with conditional probability at least $1 / 2 ;$ its error probability is at least min<sub>�</sub> $q _ { z } / 2$ . Otherwise, by the definition of $V _ { \mathsf { b a l l } }$ some class � has $s _ { z } ( a / c , \theta _ { \gamma } / c ) ^ { 2 } \leq 2 \log N _ { z } / V _ { \mathsf { b a l l } } ,$ hence $s _ { z } ( a , \theta _ { \gamma } ) ^ { 2 } \leq ( 1 + \eta _ { d } ) 2 \log N _ { z } / V _ { \mathsf { b a l l } }$ . For $d \leq ( 1 - \varepsilon ) k V _ { \mathsf { b a l l } }$ the squared margin-to-noise ratio of this class is

$$
\frac { d s _ { z } ( a , \theta _ { \gamma } ) ^ { 2 } } { k - | z | } \leq ( 1 - \varepsilon ) ( 1 + \eta _ { d } ) \frac { k } { k - r } 2 \log N _ { z } = 2 ( 1 - \varepsilon ) ( 1 + o ( 1 ) ) \log N _ { z } ,
$$

uniformly in the rule and in the class. By (C.1) its conditional error probability is at least $N _ { z } ^ { - ( 1 - \varepsilon ) - o \check { ( } 1 ) }$ (and at least a constant if the ratio stays bounded), so the gate errs with probability at least $q _ { z } N _ { z } ^ { - ( 1 - \varepsilon ) - o ( 1 ) } = N _ { z } ^ { \varepsilon - o ( 1 ) } / | \mathcal { F } |$ . In both cases, on $\mathcal { G } _ { \gamma }$

$$
\mathbb { P } ( \gamma { \mathrm { ~ e r r s ~ } } | E _ { I } , { \mathrm { r u l e } } ) \geq { \frac { ( \operatorname* { m i n } _ { z } N _ { z } ) ^ { \varepsilon - o ( 1 ) } } { | { \mathcal { F } } | } } .
$$

Integrating over $\mathcal { G } _ { \gamma } ,$ whose probability tends to one uniformly in � when $d \to \infty ,$ , and summing <sup>over</sup> <sup>the</sup> |ℱ | <sup>gates</sup> <sup>gives</sup> $\mathbb { E } X \ge ( \operatorname* { m i n } _ { z } N _ { z } ) ^ { \varepsilon - o ( 1 ) }$ . No event simultaneous over all gates is used.

It remains to treat subsequences along which $d \leq D$ stays bounded. Let $\Omega _ { \gamma } = \{ \| E _ { I } \| _ { \mathsf { o p } } ^ { 2 } \leq 2 r \} ;$ since $\mathbb { E } \Vert E _ { I } \Vert _ { F } ^ { 2 } = r .$ , Markov’s inequality gives $\mathbb { P } ( \Omega _ { \gamma } ) \geq 1 / 2$ . Because � is nonconstant, some edge of the cube joins a class � with $g ( z ) = 1$ to a class $z ^ { \prime }$ with $g ( z ^ { \prime } ) = 0 ,$ difering in one position �. With $\| w \| = 1$ and $a = E _ { I } ^ { \top } w ,$ , on $\Omega _ { \gamma }$ we have $\left. a _ { l } \right. \le \sqrt { 2 r }$ . Put $u = \theta _ { \gamma } - \langle a , z \rangle$ . If $u \geq 0 ,$ , class � errs with conditional probability $\mathbb { P } ( \zeta \leq u ) \geq 1 / 2$ . If $u < 0$ , class $z ^ { \prime }$ errs when its noise exceeds $\theta _ { \gamma } - \langle a , z ^ { \prime } \rangle \leq u + | a _ { l } | < \sqrt { 2 r }$ , which has probability at least $\overline { { \Phi } } ( \sqrt { 2 r D / ( k - r ) } )  1 / 2$ . Hence each gate errs with probability at least $( { \textstyle \frac { 1 } { 4 } } - o ( 1 ) ) \operatorname* { m i n } _ { z } q _ { z } ,$ , and $\begin{array} { r } { \mathbb { E } X \geq ( \frac { 1 } { 4 } - o ( 1 ) ) \operatorname* { m i n } _ { z } N _ { z }  \infty . } \end{array}$ □

Remark C.1 (Adapting to the local Gram matrix). A gate-local rule may use the dual frame $w = E _ { I } ( E _ { I } ^ { \top } E _ { I } ) ^ { - 1 } a ,$ , whose clean weights equal � exactly. By Theorem 4.2 this cannot lower the first-order expected-error threshold below $k V _ { \mathsf { b a l l } }$ , and by Theorem 4.1 the canonical rule already attains it: the Gram fluctuations of a canonical probe are absorbed exactly by the chi-square variable � of proposition 3.2.

## D Proofs for the box

## D.1 Extremes of a Gaussian sample

Lemma D.1. Let $Z _ { 1 } , \ldots , Z _ { q }$ be independent standard Gaussians, $q  \infty ,$ and $f i x j \ge 1$ . The � largest values divided by $\sqrt { 2 \log { q } }$ converge to 1 in probability, and so do the � smallest values divided $b y - { \sqrt { 2 \log q } }$ . Their expectations, divided by $\pm { \sqrt { 2 \log q } } ,$ , also converge to 1.

Proof. Fix $\delta \in ( 0 , 1 )$ and put $\ell = \sqrt { 2 \log q } . \mathrm { ~ B y ~ ( C . 1 ) } , q \overline { { \Phi } } ( ( 1 + \delta ) \ell ) \leq q ^ { - 2 \delta - \delta ^ { 2 } } \to 0 ,$ so by a union bound no value exceeds $( 1 + \delta ) \ell$ with probability tending to one. Also $q { \overline { { \Phi } } } ( ( 1 - \delta ) \ell ) \geq q ^ { \delta } / ( C \ell ) \to$ $\infty ,$ so the number of values above $( 1 - \delta ) \ell$ is binomial with a diverging mean and is at least � with probability tending to one. Apply the same argument to �. For the expectations, each of these order statistics is at most max<sub>�</sub> $\left| Z _ { i } \right|$ in modulus, and E max<sub>�</sub> $Z _ { i } ^ { 2 } = ( \mathbb { E } \operatorname* { m a x } _ { i } \lvert Z _ { i } \rvert ) ^ { 2 } + \mathrm { V a r } ( \operatorname* { m a x } _ { i } \lvert Z _ { i } \rvert ) \leq$ $2 \log ( 2 q ) + 1$ by the maximal inequality (Boucheron et al., 2013, Section 2.5) and the Gaussian Poincaré inequality for the 1-Lipschitz function $\mathbf { m a x } _ { i } | Z _ { i } | ;$ so the ratios are bounded in $L ^ { 2 }$ and hence uniformly integrable. ■

## D.2 Proof of proposition 5.3

Let � be the value claimed in the proposition, that is, $c = 2 { \sqrt { \log ( k D ) } }$ if $D \leq k$ and $c = \lambda _ { 1 } + \lambda _ { D }$ if $D \geq k$ . In both cases $2 \log ( k D ) \leq c ^ { 2 } \leq 4 \log ( k D )$ . Since $a = ( 1 , 1 )$ and � has the law of $- G ,$ we may take $\varsigma _ { 1 0 } = 1 $ ; the class supremum is then $V ^ { * } = \operatorname* { m a x } _ { i \in S } V _ { i }$ with $V _ { i } = Z _ { i } + M _ { i }$ and $M _ { i } = { \mathrm { m a x } } _ { j \in C ( i ) } W _ { j } ,$ and $\Gamma _ { 1 0 } ( S ) = \mathbb { E } V ^ { * } .$ ; the 2� variables $Z _ { i } , M _ { i }$ are independent, and each $M _ { i }$ is the maximum of � independent standard Gaussians. For $x \ge 0$ and $u \in [ 0 , x ] \mathrm { p u t }$

$$
F _ { x } ( u ) = \frac { u ^ { 2 } } { 2 } + \Big ( \frac { ( x - u ) ^ { 2 } } { 2 } - \log D \Big ) _ { + } .
$$

The exponent. On $\{ u \leq x - \lambda _ { D } \} , F _ { x } ( u ) = u ^ { 2 } / 2 + ( x - u ) ^ { 2 } / 2 - \log D ,$ , and on $\{ u \ge x - \lambda _ { D } \}$ $F _ { x } ( u ) = u ^ { 2 } / 2$ . Hence mi $\mathfrak { i } _ { u } F _ { x } ( u ) = x ^ { 2 } / 4 - \log D \mathrm { i f } x \geq 2 \lambda _ { D }$ and $( x - \lambda _ { D } ) ^ { 2 } / 2 \operatorname { i f } \lambda _ { D } \leq x \leq 2 \lambda _ { D }$ . If $D \leq k$ then $c \geq 2 \lambda _ { D }$ and $c ^ { 2 } / 4 - \log D = \log k ; { \mathrm { i f } } D \geq k$ then $\lambda _ { D } \leq c \leq 2 \lambda _ { D }$ and $( c - \lambda _ { D } ) ^ { 2 } / 2 = \log k$ So min<sub>�</sub> $F _ { c } ( u ) = \log { 1 }$ � in both cases. For $s \geq 1 _ { \ast }$ , substituting $u = s u ^ { \prime }$ and using $s ^ { 2 } \log D \geq \log D$ gives $F _ { s x } ( s u ^ { \prime } ) \ge s ^ { 2 } F _ { x } ( u ^ { \prime } )$ , hence min $_ u F _ { s c } ( u ) \geq s ^ { 2 } \log k$ . Maximizing $v _ { 1 } + v _ { 2 }$ over the set $K _ { D }$ of proposition 5.3 gives the same two cases, so $h _ { K _ { D } } ( 1 , 1 ) = c$

Upper bound. Fix $\delta \in ( 0 , 1 )$ , let $x = ( 1 + \delta ) c , L = lceil 2 ( 1 + \delta ) / \delta \rceil$ and $u _ { l } = l x / L$ . If $V _ { i } > x ,$ then $M _ { i } > x$ and $Z _ { i } \leq 0 .$ , or $Z _ { i } \in \left( u _ { l } , u _ { l + 1 } \right]$ and $M _ { i } > x - u _ { l + 1 }$ for some $l < L ,$ or $Z _ { i } ~ > ~ x $ . With $\overline { { \Phi } } ( v ) \leq e ^ { - v ^ { 2 } / 2 }$ and $\mathbb { P } ( M _ { i } > v ) \le \operatorname* { m i n } \{ 1 , D e ^ { - v ^ { 2 } / 2 } \}$ for $v \geq 0 ,$ , and $u _ { l + 1 } = u _ { l } + x / L ,$

$$
k \mathbb { P } ( V _ { i } > x ) \le 2 k D e ^ { - x ^ { 2 } / 2 } + k \sum _ { l < L } e ^ { - F _ { x - x / L } ( u _ { l } ) } \le 2 ( k D ) ^ { 1 - ( 1 + \delta ) ^ { 2 } } + L k ^ { 1 - ( 1 + \delta / 2 ) ^ { 2 } } ,
$$

because $x ^ { 2 } \ge 2 ( 1 + \delta ) ^ { 2 } \log ( k D )$ and $x - x / L \geq ( 1 + \delta / 2 ) c$ . Both terms tend to zero, so $\mathbb { P } ( V ^ { \ast } > ( 1 + \delta ) c )  0 .$

Lower bound. If $D \geq k ,$ , let �∗ maximize $Z _ { i }$ . By lemma D.1, $Z _ { i ^ { * } } \geq ( 1 - \delta ) \lambda _ { 1 }$ with probability tending to one, and $M _ { i ^ { * } }$ is independent of $( Z _ { i } )$ and at least $( 1 - \delta ) \lambda _ { D }$ with probability tending to one because $D \geq k  \infty ;$ hence $V ^ { * } \geq ( 1 - \delta ) c$ . If $D \leq k ,$ , let $y = ( 1 - \delta ) \sqrt { \log ( k D ) }$ and let � count the � with $Z _ { i } \geq y$ and $M _ { i } \geq y , \mathsf { a }$ binomial variable with success probability $p = \overline { { \Phi } } ( y ) \mathbb { P } ( M _ { 1 } \geq y )$ Since $\begin{array} { r } { \mathbb { P } ( M _ { 1 } \geq y ) = 1 - ( 1 - \overline { { \Phi } } ( y ) ) ^ { D } \geq \frac { 1 } { \jmath } \operatorname* { m i n } \{ 1 , D \overline { { \Phi } } ( y ) \} , \overline { { \Phi } } ( y ) \geq e ^ { - y ^ { 2 } / 2 } / ( C ( 1 + y ) ) } \end{array}$ and $k D \le k ^ { 2 }$

$$
k p \ge \frac { 1 } { 2 } \operatorname* { m i n } \Bigl \{ \frac { ( k D ) ^ { 1 - ( 1 - \delta ) ^ { 2 } } } { C ^ { 2 } ( 1 + y ) ^ { 2 } } , \frac { k ^ { 1 - ( 1 - \delta ) ^ { 2 } } } { C ( 1 + y ) } \Bigr \} \to \infty ,
$$

so $\mathbb { P } ( N \geq 1 ) \to 1$ , and on $\{ N \ge 1 \} , V ^ { * } \ge 2 y = ( 1 - \delta ) c$

Expectations. $| V ^ { * } | \leq \mathrm { m a x } _ { i } | Z _ { i } | + \mathrm { m a x } _ { j } | W _ { j } | .$ , the second maximum over the �� partner reads, so the bound in the proof of lemma D.1 gives $\begin{array} { r } { \mathbb { E } ( V ^ { * } ) ^ { 2 } \leq 4 \log ( 2 k ) + 4 \log ( 2 k D ) + 4 = O ( c ^ { 2 } ) } \end{array}$ . By Cauchy–Schwarz, $\mathbb { E } ( V ^ { \ast } - ( 1 + \delta ) c ) _ { + }$ and $\mathbb { E } ( ( 1 - \delta ) c - V ^ { * } ) .$ are at most $O ( c )$ times the square roots of $\mathbb { P } ( V ^ { * } > ( 1 + \delta ) c )$ and $\mathbb { P } ( V ^ { * } < ( 1 - \delta ) c )$ , which tend to zero. Hence $( 1 - \delta ) c - o ( c ) \leq$ $\Gamma _ { 1 0 } ( S ) \leq ( 1 + \delta ) c + o ( c )$ for every �. □

## D.3 Proof of Theorem 5.4

Work in the representation of Theorem 3.1 and let $G \in \mathbb { R } ^ { m }$ collect � on � and � of �. By proposition B.1, for every realization and every representation $( a , \theta )$ the strict criterion

$$
\sigma _ { d } \sqrt { H } \left( T _ { z } ( a ) - \varsigma _ { z } \bar { Z } \langle a , z \rangle \right) < s _ { z } ( a , \theta ) + ( 2 g ( z ) - 1 ) ( H - 1 ) \langle a , z \rangle \qquad \mathrm { f o r ~ a l l } ~ z ,\tag{D.1}
$$

where $\begin{array} { r } { T _ { z } ( a ) = \mathfrak { m a x } _ { I \in \mathcal { F } _ { z } ( S ) } \langle \varsigma _ { z } a , G _ { I } \rangle } \end{array}$ , implies $X _ { a , \theta } = 0 ,$ , and $X _ { a , \theta } = 0$ implies (D.1) with < replaced by $\leq .$ . The success statements use the first implication and the failure statements only the second, so no almost-sure qualification over the uncountably many representations is needed. By lemma B.2 applied to $\xi = G$ and $b = \varsigma _ { z } a , T _ { z } ( a )$ is a sum over the positions of $\left| a _ { l } \right|$ times one <sup>of</sup> <sup>the</sup> <sup>�</sup> <sup>largest</sup> <sup>values</sup> <sup>of</sup> <sup>�</sup> <sup>or</sup> <sup>of</sup> −<sup>�</sup> <sup>in</sup> <sup>the</sup> <sup>pool</sup> <sup>of</sup> <sup>that</sup> <sup>position,</sup> <sup>the</sup> <sup>values</sup> <sup>being</sup> <sup>assigned</sup> <sup>by</sup> rearrangement. Both pools have at least � elements for large $k ,$ so lemma B.2 applies to every class.

Fix $\delta \in ( 0 , 1 / 8 )$ and let $\mathcal { E } _ { \delta }$ be the event on which (i) the � largest and the � smallest values of $( Z _ { i } ) _ { i \leq k }$ lie within a factor $1 \pm \delta \ \mathrm { o f } \pm \lambda _ { 1 } .$ , and those of $( W _ { j } ) _ { j \leq n }$ within a factor $1 \pm \delta \ \mathrm { o f } \pm \lambda _ { 0 } ;$ (ii) $| \bar { Z } | \le \delta \lambda _ { 1 } ; ( \mathrm { i i i } ) | H - 1 | \le \delta \sigma _ { d } \lambda _ { 1 } ;$ and $\left( \mathrm { i v } \right) \lvert \sqrt { H } - 1 \rvert \leq \delta$ . By lemma D.1, (i) and (ii) hold with probability tending to one. For (iii), Chebyshev’s inequality gives $\mathbb { P } ( | H - 1 | > \delta \sigma _ { d } \lambda _ { 1 } ) \le$ $( 2 / d ) ^ { \overline { { / } } ( \delta ^ { 2 } k \lambda _ { 1 } ^ { 2 } / d ) } = 2 / ( \delta ^ { 2 } k \lambda _ { 1 } ^ { 2 } )  0$ , for every �. Condition (iv) holds with probability tending to one when $d \to \infty ;$ we return to bounded � at the end.

On $\mathcal { E } _ { \delta . }$ , for every $a \ \ne \ 0$ and every class, (i) gives $T _ { z } ( a ) \in [ 1 - \delta , 1 + \delta ] h _ { \mathsf { B o x } _ { z } } ( a )$ . Since $\lambda _ { 0 } \ \geq \ \lambda _ { 1 }$ eventually, $h _ { \mathsf { B o x } _ { z } } ( a ) \geq \lambda _ { 1 } \| a \| _ { 1 } \geq \lambda _ { 1 } | \langle a , z \rangle |$ , so by (ii) $| \bar { Z } \langle a , z \rangle | \ \leq \ \delta h _ { \mathsf { B o x } _ { z } } ( a )$ and by $\mathrm { ( i i i ) } \ | ( H - 1 ) \langle a , z \rangle | \ \leq \ \delta \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a ) . \ \mathrm { W i t h ~ ( i v ) , }$ the non-strict form of (D.1) implies $s _ { z } ~ \ge ~ ( 1 -$ $\delta ) ( 1 - 2 \delta ) \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a ) - \delta \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a ) \geq ( 1 - 4 \delta ) \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a )$ for all $z ,$ and it is implied by $s _ { z } ~ \geq$ $( 1 + 5 \delta ) \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a )$ for all $z ,$ since then the left side is at most $( 1 + \delta ) ( 1 + 2 \delta ) \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a )$ and the right side at least $( 1 + 4 \delta ) \sigma _ { d } h _ { \sf B o x _ { z } } ( a )$ . These two implications hold simultaneously for all representations, because $\mathcal { E } _ { \delta }$ does not depend on $( a , \theta )$

$( i ) \operatorname { I f } d \geq ( 1 + \varepsilon ) D _ { \mathsf { b o x } } ( a , \theta )$ , every class has $s _ { z } \geq \sqrt { 1 + \varepsilon } \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a )$ , which exceeds $( 1 + 5 \delta ) \sigma _ { d } h _ { \mathsf { B } \mathsf { o x } _ { z } } ( a )$ once � is small; so $X _ { a , \theta } = 0 \mathrm { o n } \mathcal { E } _ { \delta }$ . Here $d \geq D _ { \mathsf { b o x } } \geq 4 k \lambda _ { 1 } ^ { 2 } $ (see below), so (iv) is available. If $d \leq ( 1 - \varepsilon ) D _ { \mathsf { b o x } } ( a , \theta )$ , some class has $s _ { z } \leq \sqrt { 1 - \varepsilon } \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a ) < ( 1 - 4 \delta ) \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a ) , \mathsf { s o } X _ { a , \theta } \neq 0$ on $\mathcal { E } _ { \delta }$

(ii) If $d \leq ( 1 - \varepsilon ) k V _ { \mathsf { b o x } } ,$ every representation has $D _ { \mathsf { b o x } } ( a , \theta ) \geq k V _ { \mathsf { b o x } } \geq d / ( 1 - \varepsilon )$ , so the previous argument applies to all of them at once on $\mathcal { E } _ { \delta }$

(iii) By lemma B.2, $\Gamma _ { z } = \mathbb { E } T _ { z } ( a )$ is a combination of expectations of the � largest values of $\pm G$ in the two pools with coeficients $| a _ { l } | .$ , and lemma D.1 gives $\Gamma _ { z } = h _ { \mathsf { B o x } _ { z } } ( a ) ( 1 + o ( 1 ) )$ uniformly in $^ { a . }$ Small dimensions. Because � is nonconstant, some edge of the cube joins a class � with $g ( z ) = 1$ to a class $z ^ { \prime }$ with $g ( z ^ { \prime } ) = 0 ;$ they difer in one position $l ,$ so $s _ { z } + s _ { z ^ { \prime } } = | a _ { l } | \leq \| a \| .$ <sub>1</sub> and $s _ { z } ^ { H } + s _ { z ^ { \prime } } ^ { H } = H \langle a , z - z ^ { \prime } \rangle \leq H \| a \| _ { 1 }$ . Since $h _ { \mathsf { B o x } _ { z } } ( a ) \geq \lambda _ { 1 } \| a \| _ { 1 }$ for every class, the first identity gives

$D _ { \mathsf { b o x } } ( a , \theta ) \geq 4 k \lambda _ { 1 } ^ { 2 }$ for every representation; in particular $d $ in the first half of (i). Adding the two non-strict criteria for � and $z ^ { \prime }$ shows that success requires $\sigma _ { d } \sqrt { H } \left( M _ { z } + M _ { z ^ { \prime } } \right) \leq H \| a \| _ { 1 }$ with $M _ { z } = T _ { z } ( a ) - \varsigma _ { z } \bar { Z } \langle a , z \rangle$ . On the event $( \mathrm { i } ) { - } ( \mathrm { i i } )$ $M _ { z } + M _ { z ^ { \prime } } \geq 2 ( 1 - 2 \delta ) \lambda _ { 1 } \| a \| _ { 1 }$ , so success requires $H \geq 4 ( 1 - 2 \delta ) ^ { 2 } \lambda _ { 1 } ^ { 2 } k / d$ . Along subsequences with bounded $d$ this has probability tending to zero, simultaneously for all representations, which completes the failure statements in (i) and (ii) without (iv). □

Remark D.2 (What the converse covers). The event $\mathcal { E } _ { \delta }$ does not depend on $( a , \theta )$ , and on it every common representation fails. The converse in Theorem 5.4(ii) therefore covers any common representation, including one chosen after observing $( A _ { i } )$ and $( B _ { j } ) _ { \ l }$ , for instance the sample-adaptive bias recorded as an oracle in Appendix L. It does not cover rules in which diferent gates use diferent weights; for those, Theorem 4.2 bounds only the expected count from below, and no lower bound on their reliability threshold is known.

## E Proofs for threshold gates

## E.1 Proof of Theorem 6.1

By proposition B.3 it sufices, for the optimized thresholds, to consider $a = \mathbf { 1 } ;$ for the midpoint thresholds ${ \boldsymbol { a } } = \mathbf { 1 }$ by definition. With ${ a } = \mathbf { 1 }$ , all classes with � active inputs have clean score $j ,$ box support $h _ { j } = j \lambda _ { 1 } + ( r - j ) \lambda _ { 0 }$ and ball support $\sqrt { r } \rho _ { z }$ with $\rho _ { z } ^ { 2 } = 2 \log \bar { N } _ { z } = \rho _ { i } ^ { 2 } ( 1 + o ( 1 ) )$ , where $\rho _ { j } ^ { 2 } = j \lambda _ { 1 } ^ { 2 } + ( r - j ) \lambda _ { 0 } ^ { 2 }$ . A bias represents THR<sup>�</sup> if and only if $\theta \in \left( t - 1 , t \right)$ , and then the margins are $s _ { j } = \theta - j$ for $j \leq t - 1$ and $s _ { j } = j - \theta$ for $j \geq t$ . Since $\lambda _ { 0 } \geq \lambda _ { 1 } ,$ both $h _ { j }$ and $\rho _ { j }$ decrease in $j .$ Only the classes $t - 1$ and � bind. For $j = t - 1 - i$ with $i \geq 1 , s _ { j } = s _ { t - 1 } + i$ and $h _ { j } = h _ { t - 1 } + i ( \lambda _ { 0 } - \lambda _ { 1 } )$ so $h _ { j } / s _ { j } \le \operatorname* { m a x } \{ h _ { t - 1 } / s _ { t - 1 } , \lambda _ { 0 } - \lambda _ { 1 } \}$ . For $j = t + i , s _ { j } = s _ { t } + i$ and $h _ { j } \leq h _ { t } ,$ , so $h _ { j } / s _ { j } \le h _ { t } / s _ { t }$ . Since $s _ { t - 1 } < 1$ and $h _ { t - 1 } \geq \lambda _ { 0 } > \lambda _ { 0 } - \lambda _ { 1 }$ , the first maximum is $h _ { t - 1 } / s _ { t - 1 } \colon$ the classes below $t - 1$ never exceed the ratio of class $t - 1$ . For the ball the same argument uses $\rho _ { j } ^ { 2 } = \rho _ { t - 1 } ^ { 2 } + i ( \lambda _ { 0 } ^ { 2 } - \lambda _ { 1 } ^ { 2 } ) \leq$ $\rho _ { t - 1 } ^ { 2 } + i \lambda _ { 0 } ^ { 2 } \le ( 1 + i ) \rho _ { t - 1 } ^ { 2 } , \mathrm { ~ s o ~ } \rho _ { j } / s _ { j } \le \sqrt { 1 + i } \rho _ { t - 1 } / ( s _ { t - 1 } + i ) \le \rho _ { t - 1 } / ( \sqrt { 2 } s _ { t - 1 } )$ because $s _ { t - 1 } \leq 1 ;$ the classes above � satisfy $\rho _ { j } / s _ { j } \le \rho _ { t } / ( 2 s _ { t } )$ . These strict factors absorb the $1 + o ( 1 )$ diference between $\rho _ { z }$ and $\rho _ { | z | }$

(a) Equalizing $h _ { t - 1 } / ( \theta - t + 1 ) = h _ { t } / ( t - \theta )$ gives $\theta - t + 1 = h _ { t - 1 } / ( h _ { t - 1 } + h _ { t } )$ and the common value $h _ { t - 1 } + h _ { t } = ( 2 t - 1 ) \lambda _ { 1 } + ( 2 r - 2 t + 1 ) \lambda _ { 0 } ;$ any other bias increases one of the two ratios. Hence $V _ { \mathsf { b o x } } = ( ( 2 t - 1 ) \lambda _ { 1 } + ( 2 r - 2 t + 1 ) \lambda _ { 0 } ) ^ { 2 }$ exactly. For the half-slack bias $\begin{array} { r } { \theta _ { d } = t - \frac 1 2 + \frac 1 2 \sigma _ { d } \big ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } \big ) } \end{array}$ put $\hat { h } _ { j } = j \hat { \lambda } _ { 1 } + ( r - j ) \hat { \lambda } _ { 0 } = h _ { j } ( 1 + o ( 1 ) ) ;$ ; one checks $s _ { t - 1 } - \sigma _ { d } \hat { h } _ { t - 1 } = s _ { t } - \sigma _ { d } \hat { h } _ { t } = \textstyle { \frac { 1 } { 2 } } ( 1 - \sigma _ { d } ( \hat { h } _ { t - 1 } + \hat { h } _ { t } ) )$ If $d \geq ( 1 + \varepsilon ) k V _ { \mathsf { b o x } , }$ , then $\sigma _ { d } ( \hat { h } _ { t - 1 } + \hat { h } _ { t } ) \leq ( 1 + \varepsilon ) ^ { - 1 / 2 } ( 1 + o ( 1 ) )$ , so both binding classes, and by the first step all classes, satisfy $s _ { j } \geq ( 1 + \eta _ { \varepsilon } ) \sigma _ { d } h _ { j }$ for some $\eta _ { \varepsilon } > 0$ and all large $k ;$ that is, $\bar { D _ { \mathfrak { b o x } } } ( \mathbf { 1 } , \theta _ { d } ) \leq \bar { d } / ( 1 + \eta _ { \varepsilon } ) ^ { 2 }$ , and Theorem 5.4(i) applies. The same argument applies with $\lambda _ { p }$ in place of $\hat { \lambda } _ { p }$ . The converse is Theorem 5.4(ii).

(b) At $\textstyle \theta \ = \ t - { \frac { 1 } { 2 } }$ both binding margins equal $\frac { 1 } { 2 }$ and $h _ { t - 1 } \geq h _ { t }$ , so $D _ { \mathsf { b o x } } = k ( 2 h _ { t - 1 } ) ^ { 2 }$ , and Theorem 5.4(i) gives the threshold.

(c) At the midpoint the binding ball ratio is $2 \sqrt { r } \rho _ { t - 1 } , \mathrm { { s o } } \ D _ { \mathsf { b a l l } } = 4 r k \rho _ { t - 1 } ^ { 2 } ( 1 + o ( 1 ) )$ and Theorem 4.1 gives the threshold.

(d) Equalizing $\sqrt { r } \rho _ { t - 1 } / ( \theta - t + 1 ) \ = \ \sqrt { r } \rho _ { t } / ( t - \theta )$ gives $\theta - t + 1 = \rho _ { t - 1 } / ( \rho _ { t - 1 } + \rho _ { t } )$ and $V _ { \mathsf { b a l l } } = r ( \rho _ { t - 1 } + \rho _ { t } ) ^ { 2 } ( 1 + o ( 1 ) )$ . Theorem 4.1 at this bias and Theorem 4.2 give achievability and

the gate-local converse.

For $\beta  1$ every $h _ { j }$ equals $r \lambda _ { 1 } ( 1 + o ( 1 ) )$ and every $\rho _ { j }$ equals $\sqrt { r } \lambda _ { 1 } ( 1 + o ( 1 ) )$ , which gives the first statement of corollary $6 . 2 ;$ dividing $( \mathsf { a } ) \mathsf { - } ( \mathsf { d } )$ by $\lambda _ { 0 } ^ { 2 }$ and letting $\beta $ gives the second. For corollary $6 . 3 , r = t = 1$ gives $h _ { 0 } + h _ { 1 } = \rho _ { 0 } + \rho _ { 1 } = \lambda _ { 0 } + \lambda _ { 1 }$ , and the ratio $( ( 2 r - 1 + \beta ) / ( 1 + \beta ) ) ^ { 2 }$ has derivative of the sign of $2 - 2 r \leq 0$ in $\beta .$

## E.2 Unweighted margins of symmetric gates

For a representation $( a , \theta )$ write $\tau = \mathrm { m i n } _ { z } s _ { z } ( a , \theta )$ and $\kappa = \tau ^ { 2 } / ( 2 \| a \| ^ { 2 } )$ , the unweighted margin rate; its supremum $\kappa ^ { * } ( g )$ over representations is the price of $g$ when all classes are equally numerous. For $\mathrm { T H R } _ { t } ^ { r } ,$ , the representation $\begin{array} { r } { a = \mathbf { 1 } , \theta = t - \frac { \bar { 1 } } { 2 } } \end{array}$ has $\begin{array} { r } { \tau = \frac { 1 } { 2 } } \end{array}$ and $\kappa = 1 / ( 8 r )$ . It is optimal: for each position � choose a pattern of weight $t - 1$ that excludes $j ;$ turning � on changes the output, so $a _ { j } = s _ { z } + s _ { z + e _ { j } } \geq 2 \tau _ { \cdot }$ , and $\| a \| ^ { 2 } \geq 4 r \tau ^ { 2 }$ . Thus conjunction, disjunction and majority have the same unweighted rate $1 / ( 8 r ) ;$ their weighted prices difer because their class sizes difer.

## E.3 Comparators

The comparator at threshold $T \in \{ 1 , \ldots , 2 ^ { r } - 1 \}$ is $\begin{array} { r } { \mathrm { C O M P } _ { r } ^ { T } ( z ) = \mathbf { 1 } \{ \sum _ { l = 1 } ^ { r } 2 ^ { l - 1 } z _ { l } \ \ge \ T \} } \end{array}$ . Its unweighted rate depends on �, because � selects a Boolean function and diferent functions have diferent best representations.

Binary weights realize every threshold. With $a = ( 1 , 2 , \ldots , 2 ^ { r - 1 } ) , \| a \| ^ { 2 } = ( 4 ^ { r } - 1 ) / 3 .$ . The subset sums of distinct powers of two are all integers $0 , \ldots , 2 ^ { r } - 1$ , so the bias $\begin{array} { r } { T - { \frac { 1 } { 2 } } } \end{array}$ has $\begin{array} { r } { \tau = \frac { 1 } { 2 } } \end{array}$ , and

$$
\kappa = \frac { ( 1 / 2 ) ^ { 2 } } { 2 ( 4 ^ { r } - 1 ) / 3 } = \frac { 3 } { 8 ( 4 ^ { r } - 1 ) } = \Theta ( 4 ^ { - r } ) .\tag{E.1}
$$

This is the rate of a representation, not of the gate: the dictator $T = 2 ^ { r - 1 }$ is the function $z _ { r }$ of one input, with $\kappa ^ { * } = 1 / 8 ,$ , and for the hardest thresholds the best representation is not binary.

The hardest thresholds have Fibonacci weights. Let $F _ { 1 } = F _ { 2 } = 1$ and $F _ { i + 1 } = F _ { i } + F _ { i - 1 }$ . The alternating threshold $T _ { r } = ( 2 ^ { r + 1 } + ( - 1 ) ^ { r } ) / 3$ and its Boolean dual attain

$$
\kappa ^ { * } ( \mathrm { C O M P } _ { r } ^ { T _ { r } } ) = \frac { 1 } { 8 F _ { r } F _ { r + 1 } } = \Theta \Big ( \Big ( \frac { 3 + \sqrt { 5 } } { 2 } \Big ) ^ { - r } \Big ) ,\tag{E.2}
$$

The representation $a _ { i } = 2 F _ { i }$ has minimum margin one and $\begin{array} { r } { \| \boldsymbol { a } \| ^ { 2 } = 4 \sum _ { i } F _ { i } ^ { 2 } = 4 F _ { r } F _ { r + 1 } } \end{array}$ . The following induction proves its optimality and that no other comparator threshold has a smaller optimal rate.

Every comparator has a bounded Fibonacci representation. Trailing zero bits of � make the lowest inputs irrelevant, so assume the lowest bit of � is one. Starting from the least significant one bit, the comparator is a nested monotone chain $h _ { 1 } = z _ { 1 } , h _ { i } = z _ { i } \circ _ { i } h _ { i - 1 } $ , where $\circ _ { i }$ is AND for a one bit of $T$ and OR for a zero bit. Let $P _ { i }$ be the minimum accepted weighted sum of the prefix, $Q _ { i }$ the maximum rejected sum, and $A _ { i }$ the sum of the prefix weights; maintain $P _ { i } - Q _ { i } = 1$ and write $D _ { i } = A _ { i } - P _ { i } , E _ { i } = Q _ { i }$ . Start with $a _ { 1 } = 1 , D _ { 1 } = E _ { 1 } = 0$ . An AND step sets $a _ { i } = D _ { i - 1 } + 1$ after which $D _ { i } = D _ { i - 1 }$ and $E _ { i } = A _ { i - 1 } ;$ an OR step sets $a _ { i } = E _ { i - 1 } + 1$ , after which $D _ { i } = A _ { i - 1 }$ and $E _ { i } = E _ { i - 1 }$ . These identities follow from the two values of $z _ { i } { \mathrm { : ~ } } \mathrm { A N D }$ has $P _ { i } = a _ { i } + P _ { i - 1 }$ and

$Q _ { i } = \operatorname* { m a x } \{ A _ { i - 1 } , a _ { i } + Q _ { i - 1 } \}$ , and OR has $P _ { i } = { \mathrm { m i n } } \{ a _ { i } , P _ { i - 1 } \}$ and $Q _ { i } = Q _ { i - 1 }$ . In the coordinates $U = D + 1 , V = E + 1$ each step is $( U , V ) \mapsto ( U , U + V ) \mathrm { o r } ( U + V , V )$ , and the new weight is the unchanged coordinate. By induction the two sorted coordinates after $i - 1$ steps are at most $( F _ { i } , F _ { i + 1 } ) ;$ before the �th weight is assigned there have been $i - 2$ updates, so that weight is at most $F _ { i }$ . The bias $( P _ { r } + Q _ { r } ) / 2$ has margin ${ \frac { 1 } { 2 } } ,$ giving rate at least $1 / ( 8 \textstyle \sum _ { i } F _ { i } ^ { 2 } ) = 1 / ( 8 F _ { r } F _ { r + 1 } )$ for every threshold; discarded irrelevant inputs only decrease the norm.

No representation beats the alternating chain. Take any representation of an alternating chain with minimum margin $\tau > 0$ . Every input is essential and monotone, so fixing the other inputs across an output-changing edge shows that its weight is at least 2�. Fixing the higher inputs to their pass-through values, each prefix inherits a separating threshold with gap at least $2 \tau$ With $P _ { i } , Q _ { i } , A _ { i } , D _ { i } , E _ { i }$ as above, an AND step forces $a _ { i } \geq D _ { i - 1 } + 2 \tau$ and $E _ { i } \ge E _ { i - 1 } + a _ { i }$ while $D _ { i } = D _ { i - 1 } ;$ an OR step forces $a _ { i } \geq E _ { i - 1 } + 2 \tau$ and $D _ { i } \geq D _ { i - 1 } + a _ { i }$ while $E _ { i } = E _ { i - 1 }$ . Let $\psi _ { i }$ be the driver $D _ { i - 1 }$ or $E _ { i - 1 }$ used at step �. Alternation gives $\psi _ { 2 } = 0 , \psi _ { 3 } \geq 2 \tau$ and $\psi _ { i } \geq \psi _ { i - 2 } + a _ { i - 1 }$ for $i \geq 4 .$ , and simultaneous induction yields $\psi _ { i } \geq 2 \tau ( F _ { i } - 1 )$ and $a _ { i } \geq 2 \tau F _ { i }$ . Hence $\| a \| ^ { 2 } \geq 4 \tau ^ { 2 } F _ { r } F _ { r + 1 } .$ which is the reverse bound in (E.2). The binary rate $\Theta ( 4 ^ { - r } )$ is therefore only achievable, and it overstates the dimension cost of the hardest comparators by a factor $\Theta ( ( \bar { 6 } - 2 \sqrt { 5 } ) ^ { r } ) ;$ the base $( 3 + { \sqrt { 5 } } ) / 2$ is the square of the golden ratio.

These constructions belong to the classical theme of weight geometry in threshold logic (Muroga, 1971; Håstad, 1994); here they show that for an asymmetric gate the choice of representation, not only of the bias, enters the price. Exhaustive finite checks are recorded in Appendix K.

## F The critical window

Throughout this section $\mathcal { F }$ is the universal family of pairwise conjunctions, $k \to \infty , n = m - k$ and $n / k  \infty$ . We use the representation of Theorem 3.1, write $\alpha _ { i } = A _ { i } - 1$ for the centered active reads, and use the quantile levels $\hat { \lambda } _ { 1 } = \overline { { \Phi } } ^ { - 1 } \big ( 1 / k \big )$ and $\hat { \lambda } _ { 0 } = \overline { { \Phi } } ^ { - 1 } ( 1 / n )$ with $\beta = \sqrt { \log n / \log k }$ By Mills’ ratio, $\overline { { \Phi } } ^ { - 1 } ( 1 / q ) ^ { 2 } = 2 \log q - \log ( 4 \pi \log q ) + o ( 1 )$ as $q $ ∞<sup>;</sup> <sup>applied</sup> <sup>to</sup> $q = k$ and $q = n$ this gives $\hat { \lambda } _ { 1 } = \lambda _ { 1 } ( 1 + o ( 1 ) )$ and $\hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 } = \beta ( 1 + o ( 1 ) )$ .

## F.1 Normalization

For independent standard Gaussians, Mills’ ratio gives the classical normalization of maxima (Leadbetter et al., 1983):

$$
k \overline { { \Phi } } ( \hat { \lambda } _ { 1 } + y / \hat { \lambda } _ { 1 } )  e ^ { - y } , \qquad n \overline { { \Phi } } ( \hat { \lambda } _ { 0 } + y / \hat { \lambda } _ { 0 } )  e ^ { - y } ,
$$

so $\hat { \lambda } _ { 1 } ( \operatorname* { m a x } _ { i } Z _ { i } - \hat { \lambda } _ { 1 } ) \Rightarrow G _ { A }$ and $\hat { \lambda } _ { 0 } ( \operatorname* { m a x } _ { j } W _ { j } - \hat { \lambda } _ { 0 } ) \Rightarrow G _ { B }$ , independent standard Gumbel variables. On either window of Theorem $7 . 1 , d \times i ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ^ { 2 }$ , and the exact representation contributes only negligible perturbations in units of $\sigma _ { d } / \hat { \lambda } _ { 1 }$

$$
\frac { \hat { \lambda } _ { 1 } ( H - 1 ) } { \sigma _ { d } } = O _ { \mathbb { P } } \Big ( \frac { \hat { \lambda } _ { 1 } } { \sqrt { k } } \Big ) , \qquad \hat { \lambda } _ { 1 } \bar { Z } = O _ { \mathbb { P } } \Big ( \frac { \hat { \lambda } _ { 1 } } { \sqrt { k } } \Big ) , \qquad \hat { \lambda } _ { 1 } ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ( \sqrt { H } - 1 ) = O _ { \mathbb { P } } \Big ( \frac { \hat { \lambda } _ { 1 } } { \sqrt { k } } \Big ) .\tag{F.1}
$$

The centers (14) use the scale $\sigma _ { d } = \sqrt { k / d }$ of the Gaussian part $\sigma Z _ { i }$ of a read; using the variance $( k + 1 ) / d$ of an active read instead changes them by $\bar { O ( \lambda _ { 1 } ^ { 2 } / k ) } = o ( 1 )$ in these units. Since

$\begin{array} { r } { \hat { \lambda } _ { 1 } ^ { 2 } \sim 2 \log k , } \end{array}$ , all these errors vanish without any restriction on log $n / k ,$ even when $\hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 }  \infty$

## F.2 The midpoint profile

The one-active class succeeds when max $\large ( i A _ { i } + \mathbf { m } { \bf { a } } { \bf { x } } _ { j } B _ { j } \leq 3 / 2$ , that is, when

$$
\operatorname* { m a x } _ { i } Z _ { i } + \operatorname* { m a x } _ { j } W _ { j } \leq \frac { 1 / 2 - ( H - 1 ) } { \sigma _ { d } \sqrt { H } } + \bar { Z } .
$$

At $d = d _ { 0 } + w x$ with $w = 8 k ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) / \hat { \lambda } _ { 1 } = 8 k ( 1 + \beta ) ( 1 + o ( 1 ) )$ , the right side equals $\hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } +$ $x / \hat { \lambda } _ { 1 } + o _ { \mathbb { P } } ( 1 / \hat { \lambda } _ { 1 } )$ by (F.1). Multiplying the slack by $\hat { \lambda } _ { 1 } ,$ , the success event of this class becomes

$$
G _ { A } + \frac { \hat { \lambda } _ { 1 } } { \hat { \lambda } _ { 0 } } G _ { B } \leq x + o _ { \mathbb { P } } ( 1 ) .
$$

For $\beta \to \beta _ { 0 } < \infty$ its probability tends to $\mathbb P ( G _ { A } + G _ { B } / \beta _ { 0 } \le x ) = \mathbb E \exp ( - e ^ { \beta _ { 0 } ( G _ { A } - x ) } ) ;$ ; for $\beta $ the second fluctuation vanishes and the limit is the Gumbel distribution function. The convergence is uniform on compact �-intervals, because the limits are continuous and the errors in (F.1) do not depend on �.

The two other classes do not enter. The smallest both-active score is $2 - 2 \sigma _ { d } \hat { \lambda } _ { 1 } + O _ { \mathbb { P } } \big ( \sigma _ { d } / \hat { \lambda } _ { 1 } \big ) ,$ while the midpoint balance is $\begin{array} { r } { \sigma _ { d } ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) = \frac { 1 } { 2 } + O ( \sigma _ { d } / \hat { \lambda } _ { 1 } ) ; } \end{array}$ ; so the extra slack of this class, in units of $\sigma _ { d } / \hat { \lambda } _ { 1 . }$ , is of order $\hat { \lambda } _ { 1 } ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } ) ,$ , which tends to infinity because $\hat { \lambda } _ { 0 } ^ { 2 } - \hat { \lambda } _ { 1 } ^ { 2 } \sim 2 \log ( n / k ) $ 8 when $\hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 }  1$ , and trivially otherwise. The largest zero-active score is $2 \sigma _ { d } \hat { \lambda } _ { 0 } + o _ { \mathbb { P } } ( 1 ) \leq$ $1 + o _ { \mathbb { P } } ( 1 ) < 3 / 2$ . This proves Theorem $7 . 1 ( \mathrm { i } )$

The change of variables $\tau = e ^ { - G _ { A } }$ gives $\begin{array} { r } { \Psi _ { \beta } ( x ) = \int _ { 0 } ^ { \infty } \exp ( - \tau - \tau ^ { - \beta } e ^ { - \beta x } ) d \tau } \end{array}$ . At $\beta = 1$ the identity $\begin{array} { r } { \int _ { 0 } ^ { \infty } e ^ { - \tau - c / \tau } d \tau = 2 \sqrt { c } K _ { 1 } ( 2 \sqrt { c } ) } \end{array}$ gives the Bessel form, dominated convergence gives the Gumbel limit as $\beta  \infty ,$ , and minimizing $\tau + \tau ^ { - \beta } e ^ { - \beta x }$ shows log $\Psi _ { \beta } ( x ) \asymp - e ^ { - x \beta / ( 1 + \beta ) } \mathtt { a s } x \to - \infty$ for fixed $\beta$

## F.3 Conditional firing counts

Given the active columns, the inactive reads are independent $N ( 0 , \sigma ^ { 2 } )$ variables. Let $X _ { \mathrm { a b s } }$ be the number of inactive features that form at least one wrong one-active gate at the midpoint. Then, with $M = { \mathrm { m a x } } _ { i } \alpha _ { i } ,$

$$
X _ { \mathrm { a b s } } \mid E _ { S } \sim \mathrm { B i n } ( n , q _ { S } ) , \qquad q _ { S } = \overline { { { \Phi } } } \big ( ( 1 / 2 - M ) / \sigma \big ) , \qquad \Lambda = n q _ { S } .
$$

An inactive feature can create several gate errors, so $X _ { \mathrm { a b s } } \leq X$ and in general $X _ { \mathrm { a b s } } \neq X$

Proposition F.1 (Poisson approximation). Along window sequences with $x = ( d - d _ { 0 } ) / w \to x _ { 0 }$ and $\beta \to \beta _ { 0 } \in [ 1 , \infty )$

$$
\begin{array} { r } { \mathbb { E } d _ { T V } \big ( \mathcal { L } ( X _ { \mathrm { a b s } } \mid E _ { S } ) , \mathrm { P o i s s o n } ( \Lambda ) \big ) \le \mathbb { E } \operatorname* { m i n } \{ 1 , n q _ { S } ^ { 2 } \} \to 0 , } \end{array}
$$

and $\Lambda \Rightarrow e ^ { \beta _ { 0 } \left( G _ { A } - x _ { 0 } \right) }$

Proof. The bound is the binomial–Poisson estimate of Le Cam type (Arratia et al., 1989; Barbour et al., 1992). By the computation of the previous subsection, $( 1 / 2 - M ) / \sigma = \hat { \lambda } _ { 0 } + \beta _ { 0 } ( x _ { 0 } -$ $G _ { A } ) / \hat { \lambda } _ { 0 } + o _ { \mathbb { P } } ( 1 / \hat { \lambda } _ { 0 } ) , \mathrm { s o } \Lambda = n \overline { { \Phi } } \big ( \hat { \lambda } _ { 0 } + \beta _ { 0 } ( x _ { 0 } - G _ { A } ) / \hat { \lambda } _ { 0 } + o _ { \mathbb { P } } ( 1 / \hat { \lambda } _ { 0 } ) \big ) \implies e ^ { \beta _ { 0 } ( G _ { A } - x _ { 0 } ) }$ . Since Λ is tight and $n  \infty , n q _ { \cal S } ^ { 2 } = \Lambda ^ { 2 } / n  0$ in probability, and bounded convergence gives the limit of the expectation. ■

The limit intensity $e ^ { \beta _ { 0 } \left( G _ { A } - x _ { 0 } \right) }$ has infinite mean for $\beta _ { 0 } \geq 1$ , which is the statement used at the end of Section 7.

## F.4 Two-sided extremes

Standardize the active extremes by $T ( y ) = { \overline { { \Phi } } } ^ { - 1 } ( e ^ { - y } / k )$ : let max<sub>�</sub> $\alpha _ { i } = \sigma _ { d } T ( Y ^ { + } )$ , and let the two smallest be $- \sigma _ { d } T ( Y _ { 1 } )$ and $- \sigma _ { d } T ( Y _ { 2 } )$ with $Y _ { 1 } \geq Y _ { 2 }$

Lemma F.2 (Asymptotic independence of the two tails). Jointly, $\begin{array} { r l } { ( Y ^ { + } , Y _ { 1 } , Y _ { 2 } ) } & { { } \Rightarrow } \end{array}$ $\left( G _ { A } , - \log \omega _ { 1 } , - \log ( \omega _ { 1 } + \omega _ { 2 } ) \right)$ , where $G _ { A }$ is standard Gumbel, $\omega _ { 1 } , \omega _ { 2 }$ are unit exponentials, and the three are independent; $\omega _ { 1 } < \omega _ { 1 } + \omega _ { 2 }$ are the first two points of a unit Poisson process. In particular $Y _ { 1 } + Y _ { 2 } \Rightarrow - \log [ \omega _ { 1 } ( \omega _ { 1 } + \omega _ { 2 } ) ]$ , whose distribution function is

$$
F _ { + } ( x ) = e ^ { - \sqrt { c } } + \int _ { 0 } ^ { \sqrt { c } } e ^ { - c / s } d s , \qquad c = e ^ { - x } ,\tag{F.2}
$$

$$
w i t h r i g h t t a i l 1 - F _ { + } ( x ) = ( { \textstyle { \frac { 1 } { 2 } } } + o ( 1 ) ) x e ^ { - x } a s x  \infty .
$$

Proof. For the independent Gaussians $\left( Z _ { i } \right)$ of Theorem 3.1, partition the upper and lower tails into finitely many disjoint extreme-level intervals. An observation belongs to at most one interval, and the interval probabilities are $( c _ { j } + o ( 1 ) ) / k$ by Mills’ ratio. The joint probability generating function of the counts is $[ 1 + \textstyle \sum _ { i } ( c _ { j } + o ( 1 ) ) ( t _ { j } - 1 ) / k ] ^ { k } \to \exp ( \textstyle \sum _ { i } c _ { j } ( t _ { j } - 1 ) )$ , so the upper and lower point processes converge jointly to independent Poisson processes with intensity $e ^ { - y } d y$ . Their first upper point and first two lower points have the stated laws. The perturbations (F.1) are $o _ { \mathbb { P } } ( 1 )$ on both extreme scales, so the convergence transfers to $\left( \alpha _ { i } \right)$ , and $T ( y ) = \hat { \lambda } _ { 1 } + y / \hat { \lambda } _ { 1 } + o ( 1 / \hat { \lambda } _ { 1 } )$ uniformly on bounded �-intervals. Conditionally on $\omega _ { 1 } = s ,$ the event $s ( s + \omega _ { 2 } ) \geq c$ has probability $e ^ { - ( c / s - s ) }$ for $s < { \sqrt { c } }$ and one otherwise, which integrates to (F.2). Finally $\begin{array} { r } { 1 - F _ { + } ( x ) = 1 - e ^ { - \sqrt { c } } - \sqrt { c } + \int _ { 0 } ^ { \sqrt { c } } ( 1 - e ^ { - c / s } ) d s ; } \end{array}$ the first three terms are $O ( c )$ , and splitting the integral at $s = c$ shows that it is $\textstyle { \frac { 1 } { 2 } } c \log ( 1 / c ) + O ( c )$ . Finite-sample upper and lower extremes can be dependent; only their limits are independent. ■

## F.5 The optimal common bias

At the optimal bias two classes bind, and their extremes come from opposite tails of the same active sample; lemma F.2 decouples them.

Proof of Theorem 7.1(ii). Use the slack coordinates

$$
x _ { - } = \frac { \hat { \lambda } _ { 1 } } { \sigma _ { d } } \big ( \theta - 1 - \sigma _ { d } ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) \big ) , \qquad x _ { + } = \frac { \hat { \lambda } _ { 1 } } { \sigma _ { d } } \big ( 2 - \theta - 2 \sigma _ { d } \hat { \lambda } _ { 1 } \big ) ,
$$

for the one-active and both-active classes. For bounded $( x _ { - } , x _ { + } )$ , lemma F.2 and the previous subsections give the joint success limit $\Psi _ { \beta } ( x _ { - } ) F _ { + } ( x _ { + } ) ;$ : the upper active extreme and the bottom pair are independent in the limit, and the inactive sample is independent of both. Their total slack is $x _ { - } + x _ { + } = \hat { \lambda } _ { 1 } ( \sqrt { d / k } - 3 \hat { \lambda } _ { 1 } - \hat { \lambda } _ { 0 } ) = x + o ( 1 )$ at $d = d _ { 0 } ^ { \mathrm { o p t } } + 2 k ( 3 + \beta ) x$ . The zero-active class is negligible by lemma F.3. Maximizing over the deterministic bias is therefore the optimization over $y = x _ { - }$ . To pass the supremum through the limit, first restrict $x _ { - }$ to a compact interval: if $x _ { - } \to - \infty$ the one-active class fails with probability tending to one, and if $x _ { - } \to$ +∞ <sup>with</sup> bounded total slack the both-active class does. Tightness of the limiting extremes makes these bounds uniform, and on compact intervals the continuous joint distribution functions converge uniformly. ■

Lemma F.3 (The zero-active class stays slack). On the optimal window, for biases with bounded slack coordinates, $\mathbb { P } ( B _ { ( 1 ) } + B _ { ( 2 ) } > \theta )  0 _ { }$ , where $B _ { ( 1 ) } \geq B _ { ( 2 ) }$ are the two largest inactive reads.

Proof. At the center, $1 = \sigma _ { d } ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) + o ( \sigma _ { d } / \hat { \lambda } _ { 1 } )$ and $\theta = 1 + \sigma _ { d } ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) + O ( \sigma _ { d } / \hat { \lambda } _ { 1 } )$ , so $\theta - 2 \sigma _ { d } \hat { \lambda } _ { 0 } = 4 \sigma _ { d } \hat { \lambda } _ { 1 } + O ( \sigma _ { d } / \hat { \lambda } _ { 1 } )$ uniformly on compact windows. The two largest inactive reads sum to $2 \sigma _ { d } \hat { \lambda } _ { 0 } + O _ { \mathbb { P } } ( \sigma _ { d } / \hat { \lambda } _ { 0 } ) + o _ { \mathbb { P } } ( \sigma _ { d } / \hat { \lambda } _ { 1 } )$ . The gap is of order $\hat { \lambda } _ { 1 } ^ { 2 }  \infty$ in units of $\sigma _ { d } / \hat { \lambda } _ { 1 }$ , also when $\hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 }  \infty$ ■

The half-slack bias of Theorem 6.1(a) is $\begin{array} { r } { \theta = \frac { 3 } { 2 } + \frac { 1 } { 2 } \sigma _ { d } ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } ) } \end{array}$ , which gives $x _ { - } = x _ { + } = x / 2$ and hence $\Psi _ { \beta } ( x / 2 ) F _ { + } ( x / 2 )$ ; this need not equal the supremum. With $\lambda _ { p }$ in place of $\hat { \lambda } _ { p }$ the split becomes $\begin{array} { r } { x _ { - } - x _ { + } = \hat { \lambda } _ { 1 } [ ( \lambda _ { 0 } - \lambda _ { 1 } ) - ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } ) ] = ( \log \beta ) / \beta - \frac { 1 } { 2 } ( 1 - 1 / \beta ) } \end{array}$ log 4� log �  � 1 , which tends $\mathrm { t o } - \infty$ when $\beta  \beta _ { 0 } > 1 _ { , }$ ; that bias has the same first-order threshold but not this profile. The optimal window is narrower than the midpoint window by $w / w _ { \mathrm { o p t } } = 4 ( 1 + \beta ) / ( 3 + \beta ) \in [ 2 , 4 )$ for every $\beta \geq 1$ . The profile concerns a deterministic common bias; it does not optimize gatespecific weights, and it does not determine the law of the full count $X ,$ which would require the number of active features attached to each firing inactive feature.

## G Extensions of the model

The box law has three ingredients: the support is random and independent of the dictionary, the dictionary is Gaussian, and one layer of gates reads a stored input. We change each in turn, and then drop the linear dictionary altogether. Proofs and the precise statements of Section 8.3 are in Appendix H.

## G.1 Adversarial supports

If the support may be chosen after the dictionary, the cloud of a class is no longer a product of two samples: an adversary can place $k - 1$ inactive columns that all align with one probe, and the interference grows linearly in � instead of as ${ \sqrt { k } } .$

The statement is in proposition 8.1.

For recovery by one threshold layer, Garg et al. (2026, Theorem 12) prove the analogous $\Omega ( k ^ { 2 } \log ( m / k ) / \log k )$ bound for arbitrary linear encoders when $k < { \sqrt { m } }$ , and for Gaussian dictionaries Ivanitskiy et al. (2026, Theorem 8 and Proposition 6) place the largest sparsity that is recoverable on every support between the orders $( d / \log m ) ^ { 1 / 2 }$ and $d ^ { 1 / 2 }$ . The factor � between typical and uniform correctness is the price of letting the support depend on the dictionary; random and worst-case supports difer in the same way for sparse regression (Candès and Plan, 2009).

## G.2 Other entry laws

The ball needs only one-gate tails, and for the midpoint pairwise network these are universal. Let the entries of $\dot { \sqrt { d } } E$ be independent with a fixed centered, unit-variance, sub-Gaussian law, and let $q _ { k , d }$ be the probability that a midpoint pairwise read with one active input exceeds $3 / 2$

The statement is in Theorem 8.2.

The proof tilts independent coordinate rows exponentially. Appendix H.2 also computes the next term, log $( q _ { k , d } / \overline { { { \Phi } } } ( x _ { K } ) ) = c _ { \nu } d / k ^ { 2 } + \cdot \cdot \cdot$ with $x _ { K } = \sqrt { d / [ 8 ( k - 1 ) ] }$ , in which the entry law enters through its skewness and excess kurtosis, and treats sparse Rademacher entries. The theorem concerns the midpoint pairwise network. The box is diferent: it rests on the exact joint law of Theorem 3.1, which has no counterpart for general entries, and whether the reliability threshold is universal is open (Section 10).

## G.3 Composition and unrestricted encodings

Writing outputs for a later layer costs dimensions of its own: pairwise conjunction has $\binom { k } { 2 }$ active outputs. For feedforward threshold circuits in which every wire has its own Gaussian direction and every active output is written once into a shared stream, Theorem H.5 shows that $d \geq ( 1 + \varepsilon ) \kappa _ { \operatorname* { m i n } } ^ { - 1 } K \log Q$ sufices, where � is the number of reads, � the largest number of active directions in the stream before a read and $2 \kappa _ { \mathrm { m i n } }$ the smallest squared clean margin of a unit-norm read. Its proof is a union bound over an ideal execution, that is, a ball argument. Without the linear dictionary, order � dimensions sufice for all pairwise conjunctions and, when log $k = o ( \log m )$ , are necessary for polynomially many threshold units: counting sign patterns (Cover, 1965) gives $d \ge \left( k \log ( m / k ) + \log ( 1 - \delta ) \right) / ( 1 + \log T )$ for � units correct on a fraction $1 - \delta$ of supports (proposition H.6), and the coeficients of $\textstyle \prod _ { l \in S } ( y - l ) ^ { 2 } .$ , the polynomial behind the neighborliness of cyclic polytopes (Barvinok, 2010, Theorem 15.2), form a 2�-dimensional <sup>code</sup> <sup>with</sup> <sup>large</sup> <sup>coeficients</sup> <sup>and</sup> <sup>small</sup> <sup>normalized</sup> <sup>margins</sup> <sup>(proposition</sup> <sup>H.7).</sup> <sup>The</sup> <sup>Θ</sup>(<sup>�</sup> <sup>log</sup> <sup>�</sup>) scale of this paper therefore belongs to accessible computation from a random linear dictionary rather than to the task itself.

## H Statements and proofs for Section 8

## H.1 Uniform correctness

Proof of proposition 8.1. For the upper bound, chi-square tail bounds and a union bound give max $_ i \| e _ { i } \| ^ { 2 } - 1 | \leq \delta$ with probability tending to one for fixed small $\delta > 0 ,$ since $d \geq C k ^ { 2 } \log m$ Conditionally on one column and its norm bound, the inner product with another column is Gaussian, so a second union bound gives ma $\mathfrak { c } _ { i \neq j } | \langle e _ { i } , e _ { j } \rangle | \leq C _ { 0 } \sqrt { \log m / d }$ . For any support of size at most �, the score of any gate difers from its clean value 0, 1 or 2 by at most $2 \delta + 2 k C _ { 0 } \sqrt { \log m / d }$ Choosing $\delta < 1 / 8$ and then � large makes this less than $1 / 2$ on one dictionary event that controls every support and every gate.

For the lower bound fix the gate 1, 2 and let $w = e _ { 1 } + e _ { 2 }$ . As $d \to \infty , \| w \| ^ { 2 } \to 2$ and $\langle w , e _ { 1 } \rangle \to 1$ in probability. Conditionally on these two columns, the alignments $\langle w , e _ { j } \rangle , j \geq 3 ,$ are independent $N ( 0 , \| w \| ^ { 2 } / d )$ variables. At the level $( 1 - \delta ) \sqrt { 2 \log ( m / k ) }$ in standard units, Mills’ bound makes the expected number of exceedances among the $m - 2$ alignments much larger than $k ,$ and Chernof’s inequality makes fewer than $k - 1$ exceedances unlikely. Let the support consist of feature 1 and the $k - 1$ best aligned features other than feature 2. The gate $( 1 , 2 )$ then has one active input and score at least

$$
1 + o _ { \mathbb { P } } ( 1 ) + ( 1 - o _ { \mathbb { P } } ( 1 ) ) ( k - 1 ) \| w \| \sqrt { 2 \log ( m / k ) / d } .
$$

Since lo $\mathrm { g } ( m / k ) \sim$ log �, a suficiently small � makes this exceed $3 / 2$ with probability tending to one when $d \leq c k ^ { 2 } \log m$ . The support is allowed to depend on the dictionary because the claim is uniform over supports. ■

The upper bound is an existence statement for the canonical network and the lower bound concerns canonical probes; neither is a statement about all decoders.

## H.2 Other entry laws

Let the columns be $e _ { i } = y _ { i } / { \sqrt { d } } ,$ , where all entries of all $y _ { i }$ are independent with a fixed centered, variance-one, sub-Gaussian law $\nu ,$ and write $\gamma _ { 1 } = \mathbb { E } Z ^ { 3 }$ and $\gamma _ { 2 } = \mathbb { E } Z ^ { 4 } - 3$ for the skewness and excess kurtosis of $Z \sim \nu ;$ in this subsection $Z$ and $Z _ { l }$ denote entries, not the active sample of Theorem 3.1. Examples are Gaussian $( 0 , 0 )$ , Rademacher $( 0 , - 2 )$ and uniform on $[ - \sqrt { 3 } , \sqrt { 3 } ]$ $\left( 0 , - 6 / 5 \right)$ . Constants written $O _ { \nu }$ may depend on �. Let $q _ { k , d }$ be the probability that a midpoint pairwise read with one active and one inactive input exceeds $3 / 2 ;$ it includes the fluctuation of the signal, not only the centered interference.

Proposition H.1 (Second-order term). For afixed law � and $d / k \to \infty ,$ put $x _ { K } = \sqrt { d / [ 8 ( k - 1 ) ] } .$ Then

$$
\log \frac { q _ { k , d } } { \overline { { \Phi } } ( x _ { K } ) } = c _ { \nu } \frac { d } { k ^ { 2 } } + O _ { \nu } \Big ( \frac { d } { k ^ { 3 } } + \frac { 1 } { \sqrt { k } } + \frac { k } { d } \Big ) , \qquad c _ { \nu } = \frac { 1 3 2 + 4 1 \gamma _ { 2 } } { 1 0 2 4 } + \frac { \gamma _ { 1 } ^ { 2 } } { 1 9 2 } .
$$

In particular $q _ { k , d } / \overline { { \Phi } } ( x _ { K } ) $ 1 when $k \ll d \ll k ^ { 2 } .$ , and $c _ { \nu } \geq 5 0 / 1 0 2 4$ , with equalityfor Rademacher entries.

The correction dominates the remainder when $d \ \gg \ k ^ { 3 / 2 }$ . Even for Gaussian entries $c _ { \nu } = 1 3 2 / 1 0 2 4 \neq 0 \mathrm { { ; } }$ : the term corrects the Gaussian tail $\overline { { \Phi } } ( x _ { K } )$ of a read, and the entry law enters only through $\gamma _ { 1 } ^ { 2 }$ and $\gamma _ { 2 } ,$ , the skewness appearing squared by invariance under a global sign change of the entries.

Corollary H.2 (Sparse entries). Let $Z = \epsilon \chi / \sqrt { q }$ with $\chi \sim \operatorname { B e r } ( q )$ and an independent Rademacher sign �, where $q = q ( k ) \in ( 0 , 1 ]$ $I f q k $ ∞ and $d / k \to \infty ,$ then log $\begin{array} { r } { q _ { k , d } = - \frac { d } { 1 6 k } [ 1 + O ( ( q k ) ^ { - 1 } + } \end{array}$ $( k / d ) \log ( d / k ) ) ]$ . If also $d / ( q k ^ { 2 } ) \to 0 ,$ , then $q _ { k , d } / \overline { { \Phi } } ( x _ { K } )  1$

At $d = \Theta ( k \log ( k m ) )$ <sup>the</sup> <sup>first</sup> <sup>condition</sup> <sup>corresponds</sup> <sup>to</sup> <sup>��</sup>/<sup>log</sup> <sup>�</sup> → ∞ <sup>expected</sup> <sup>nonzeros</sup> per column, and the second to $d q / \log ^ { 2 } m  \infty ;$ both are suficient conditions, not sparsity

thresholds.

An independent-row proof Let $A , B , Z _ { 1 } , \dotsc , Z _ { k - 1 }$ be independent with law $\begin{array} { r } { \nu , S = \sum _ { l < k } Z _ { l } , } \end{array}$ and

$$
T = ( A + B ) ( A + S ) , \qquad C = A + B , \qquad P = A ^ { 2 } + A B - 1 .
$$

The one-active score is $d ^ { - 1 } \sum _ { h = 1 } ^ { d } T _ { h }$ with independent copies $T _ { h , \ast }$ , and $T - 1 = P + C S$ . Write $K _ { k } ( t ) = \log \mathbb { E } e ^ { t ( T - 1 ) }$

Lemma H.3 (Row cumulants and tilting). For a fixed law $\nu ,$ uniformly on $\vert t \vert \le C _ { 0 } / k$ for any fixed $C _ { 0 } ,$

$$
\begin{array} { c } { { { \displaystyle K _ { k } ( t ) = \frac { 2 k + \gamma _ { 2 } + 1 } { 2 } t ^ { 2 } + \frac { k ( 1 2 + 3 \gamma _ { 2 } + 2 \gamma _ { 1 } ^ { 2 } ) + O _ { \nu } ( 1 ) } { 6 } t ^ { 3 } } } } \\ { { + \frac { k ^ { 2 } ( 2 4 + 6 \gamma _ { 2 } ) + O _ { \nu } ( k ) } { 2 4 } t ^ { 4 } + O _ { \nu } ( k ^ { - 3 } ) . } } \end{array}
$$

The rate at displacement $1 / 2$ is

$$
I _ { k } = \operatorname * { s u p } _ { t } \{ t / 2 - K _ { k } ( t ) \} = \frac { 1 } { 1 6 k } - \Big ( \frac { 6 8 + 4 1 \gamma _ { 2 } } { 1 0 2 4 } + \frac { \gamma _ { 1 } ^ { 2 } } { 1 9 2 } \Big ) \frac { 1 } { k ^ { 2 } } + O _ { \nu } ( k ^ { - 3 } ) .
$$

$I f t _ { k }$ solves $K _ { k } ^ { \prime } ( t _ { k } ) = 1 / 2$ and $\eta _ { k } = t _ { k } \sqrt { d K _ { k } ^ { \prime \prime } ( t _ { k } ) }$ , then

$$
\begin{array} { r } { \log q _ { k , d } = - d I _ { k } - \log \eta _ { k } - \frac 1 2 \log ( 2 \pi ) + O _ { \nu } ( k / d + k ^ { - 1 / 2 } ) , } \end{array}
$$

with $\eta _ { k } = \sqrt { d / ( 8 k ) } [ 1 + { \cal O } _ { \nu } ( k ^ { - 1 } ) ] .$

Proof. The variables �, � and $S / \sqrt { k }$ have uniformly bounded sub-Gaussian norms, so $( T - 1 ) / \sqrt { k }$ has a uniform exponential moment near zero; this also bounds its tilted absolute third moment uniformly for $t ~ = ~ O ( 1 / k )$ . Expanding $P + C S$ with $\mathbb { E } S ^ { 2 } \ : = \ : k - 1 , \ : \mathbb { E } S ^ { 3 } \ : = \ : ( k - 1 ) \gamma _ { 1 }$ and $\mathbb { E } S ^ { 4 } = 3 ( k - 1 ) ^ { 2 } + ( k - 1 ) \gamma _ { 2 }$ gives

$$
\begin{array} { c } { { \kappa _ { 2 } ( T ) = 2 k + \gamma _ { 2 } + 1 , \qquad \kappa _ { 3 } ( T ) = k ( 1 2 + 3 \gamma _ { 2 } + 2 \gamma _ { 1 } ^ { 2 } ) + O _ { \nu } ( 1 ) , } } \\ { { \kappa _ { 4 } ( T ) = k ^ { 2 } ( 2 4 + 6 \gamma _ { 2 } ) + O _ { \nu } ( k ) . } } \end{array}
$$

The fifth cumulant is $O _ { \nu } ( k ^ { 2 } )$ , since odd moments of � through order five are $O _ { \nu } ( k ^ { 2 } )$ and the remaining terms contain at most four copies of �. The exponential-moment bound controls the Taylor remainder from order six by $\hat { O _ { \nu } ( k ^ { 3 } t ^ { 6 } ) }$ on this interval, with the same bounds for the derivatives. Since $K _ { k } ^ { \prime \prime } ( t ) = 2 k + O _ { \nu } ( 1 )$ , the saddle point is $t _ { k } = 1 / ( 4 k ) + { \cal O } _ { \nu } ( k ^ { - 2 } )$ , and substituting $1 / ( 4 k )$ into $t / 2 - K _ { k } ( t )$ costs only $ { \mathcal { O } _ { \nu } } ( k ^ { - 3 } )$ , which gives $I _ { k }$

Under the tilted measure with density exp $\begin{array} { r } { \left( t _ { k } \sum _ { h } ( T _ { h } - 1 ) - d K _ { k } ( t _ { k } ) \right) } \end{array}$ , let � be the centered sum divided by $\sqrt { d K _ { k } ^ { \prime \prime } ( t _ { k } ) }$ . The Berry–Esseen theorem gives Kolmogorov distance $O _ { \nu } ( d ^ { - 1 / 2 } )$ from a standard Gaussian. The function $e ^ { - \eta _ { k } u } \mathbf { 1 } \{ u > 0 \}$ has total variation at most two, so

$$
q _ { k , d } = e ^ { - d I _ { k } } \left[ e ^ { \eta _ { k } ^ { 2 } / 2 } \overline { { { \Phi } } } ( \eta _ { k } ) + { \cal O } _ { \nu } ( d ^ { - 1 / 2 } ) \right] ;
$$

this applies to lattice laws as well, with the strict threshold. Since $e ^ { \eta ^ { 2 } / 2 } \overline { { \Phi } } ( \eta ) = ( \sqrt { 2 \pi } \eta ) ^ { - 1 } [ 1 +$ ${ \cal O } ( \eta ^ { - 2 } ) ]$ , the relative Berry–Esseen error is $O _ { \nu } ( \eta _ { k } / \sqrt { d } ) = O _ { \nu } ( k ^ { - 1 / 2 } )$ . This is the exponentialtilting argument behind the Cramér–Petrov theory (Petrov, 1975; Döring et al., 2022). ■

Proof of Theorem 8.2. By lemma H.3, the leading terms give the exponent. For proposition H.1, use log $\begin{array} { r } { \overline { { \Phi } } ( x _ { K } ) \ = \ - d / [ 1 6 ( k - 1 ) ] \ - \log x _ { K } - \frac 1 7 \log ( 2 \pi ) \ + \ O ( k / d ) , \ \log ( \eta _ { k } / x _ { K } ) \ = \ O _ { \nu } ( k ^ { - 1 } ) } \end{array}$ and $d / [ 1 6 ( k - 1 ) ] = d / ( 1 6 k ) + d / ( 1 6 k ^ { 2 } ) + { \cal O } ( d / k ^ { 3 } ) .$ ; the extra $6 4 / 1 0 2 4$ turns $6 8 + 4 1 \gamma _ { 2 }$ into $1 3 2 + 4 1 \gamma _ { 2 }$ Pearson’s moment inequality $\gamma _ { 2 } \geq \gamma _ { 1 } ^ { 2 } - 2 ,$ , from the positive semidefiniteness of the covariance matrix of $( Z , Z ^ { 2 } )$ , gives $c _ { \nu } \geq 5 0 / 1 0 2 4$ . For the expected count of the midpoint network, the both-active lower tail has the same leading exponent and the zero-active upper tail has exponent $( 9 + o ( 1 ) ) d / ( 1 6 k )$ by the same row argument; union bounds over $O ( k m )$ and $O ( m ^ { 2 } )$ gates give the suficient dimension $( 1 6 + \varepsilon ) k \log ( k m )$ ■

Proof of corollary H.2. For the sparse law the row moment generating function has the exact finite form

$$
\mathbb { E } e ^ { t T } = \mathbb { E } _ { A , B } \left[ e ^ { t ( A + B ) A } \big \{ 1 - q + q \cosh \big ( t ( A + B ) / \sqrt { q } \big ) \big \} ^ { k - 1 } \right] ,
$$

with $A , B \in \{ 0 , \pm q ^ { - 1 / 2 } \}$ of probabilities $1 - q , q / 2 , q / 2$ . For $\vert t \vert \le C _ { 0 } / k$ every exponential or hyperbolic argument is $O ( ( q k ) ^ { - 1 } )$ , and Taylor expansion inside the braces and then over the nine cases gives, uniformly, $K _ { k } ( t ) = k t ^ { 2 } [ 1 + O ( ( q k ) ^ { - 1 } ) ] , K _ { k } ^ { \prime } ( t ) = 2 k t [ 1 + O ( ( q k ) ^ { - 1 } ) ]$ and $K _ { k } ^ { \prime \prime } ( t ) = 2 k [ 1 + O ( ( q k ) ^ { - 1 } ) ]$ . The leading $k t ^ { 2 }$ comes from $\mathbb { E } ( A + \ddot { B } ) ^ { 2 } = 2$ and $\mathbb { E } S ^ { 2 } = k - 1 ;$ the remaining quadratic term is $O ( t ^ { 2 } / q )$ . Hence $t _ { k } = ( 4 k ) ^ { - 1 } [ 1 + O ( ( q k ) ^ { - 1 } ) ]$ and $I _ { k } = ( 1 6 k ) ^ { - 1 } [ 1 +$ $O ( ( q k ) ^ { - 1 } ) ]$ . The standardized tilted third absolute moment is $O ( q ^ { - 1 / 2 } ) \colon \mathbb { E } | A + B | ^ { 3 } = O ( q ^ { - 1 / 2 } )$ $\mathbb { E } | S | ^ { \dot { 3 } } = O ( k ^ { 3 / 2 } + k q ^ { - 1 / 2 } )$ and $\mathbb { E } | P | ^ { 3 } = O ( q ^ { - 2 } )$ , divided by $k ^ { 3 / 2 }$ with $q k \to \infty$ . The Berry–Esseen step therefore has relative error $O ( ( q k ) ^ { - 1 / 2 } )$ , and

$$
\begin{array} { r } { \log q _ { k , d } = - d I _ { k } - \log \sqrt { d / ( 8 k ) } - \frac 1 2 \log ( 2 \pi ) + O \big ( ( q k ) ^ { - 1 / 2 } + k / d \big ) , } \end{array}
$$

which proves the first claim; comparison with Mills’ formula gives $\log ( q _ { k , d } / \overline { { \Phi } } ( x _ { K } ) ) = $ $O ( d / ( q \bar { k } ^ { 2 } ) + ( q k ) ^ { - 1 / 2 } + k / d )$ and the second. ■

These results concern one gate and hence the ball. They do not extend the reliability thresholds of Theorem 5.4 or the window profiles to non-Gaussian dictionaries, because the exact law of Theorem 3.1 is Gaussian. For correlated columns, isotropy and marginal coherence tails alone do not provide the joint moment control used here, and below $q k \asymp 1$ the sparse expansion no longer applies.

## H.3 Composition

Reading a superposed output vector Let $N = { \binom { m } { \gamma } } , K ^ { \prime } = { \binom { k } { \gamma } }$ and $J ( S )$ the set of active pairs. Draw independent output directions $f _ { \gamma } \sim { \cal N } ( 0 , I _ { p } / p )$ for the � gates, independent of the input dictionary, and let $\begin{array} { r } { v ^ { \ast } = \sum _ { \gamma \in J ( S ) } f _ { \gamma } } \end{array}$ be the ideal output stream. The set $J ( S )$ is independent of the output dictionary, although its indicators are dependent across gates.

Proposition H.4 (Output channel). Suppose $K ^ { \prime }  \infty , N / K ^ { \prime }  \infty , \log N = o ( K ^ { \prime } )$ and $p / K ^ { \prime } $ . The matched-filter reads ${ \bf 1 } \{ \langle f _ { \gamma } , v ^ { * } \rangle > 1 / 2 \}$ have a vanishing expected number of errors if $p \ge ( 8 + \varepsilon ) K ^ { \prime }$ log �. Optimizing their common bias gives the expected-error threshold

$$
P _ { \mathrm { o u t } } = 2 K ^ { \prime } \big ( \sqrt { \log ( N - K ^ { \prime } ) } + \sqrt { \log K ^ { \prime } } \big ) ^ { 2 } , \qquad \theta _ { \mathrm { o u t } } = \frac { \sqrt { \log ( N - K ^ { \prime } ) } } { \sqrt { \log ( N - K ^ { \prime } ) } + \sqrt { \log K ^ { \prime } } } ,
$$

and below $( 1 - \varepsilon ) P _ { \mathrm { o u t } }$ every assignment of local output probes has a diverging expected number of read errors.

Proof. Condition on the input. For an inactive wire, $f _ { \gamma }$ is independent of $v ^ { * } ,$ , and the read has mean zero and variance $K ^ { \prime } / p$ . For an active wire, $\left. f _ { \gamma } , v ^ { * } \right. - 1$ has variance $( K ^ { \prime } + 1 ) / p ,$ , the extra term coming from the random self-norm. An inactive coordinate row has the law of �� with $A \sim N ( 0 , 1 )$ and $S \sim N ( 0 , K ^ { \prime } )$ independent, and an active row the law of $A ( A + S )$ with $\mathrm { V a r } S = K ^ { \prime } - 1$ ; their moment generating functions are $( 1 - K ^ { \prime } t ^ { 2 } ) ^ { - 1 / 2 }$ and $( 1 - 2 t - ( K ^ { \prime } - 1 ) t ^ { 2 } ) ^ { - 1 / 2 }$ Expanding the centered logarithms at displacement $\mu$ gives the rate $\mu ^ { 2 } / ( 2 K ^ { \prime } ) [ 1 + O ( 1 / K ^ { \prime } ) ]$ and the tilting argument of lemma H.3 gives the matching lower bound. At the midpoint both displacements are $1 / 2 ,$ , and summing over the � reads gives the first claim. At a free bias the constraints are $p \theta ^ { 2 } / ( 2 K ^ { \prime } ) > \log ( N - K ^ { \prime } )$ and $p ( 1 - \theta ) ^ { 2 } / ( 2 K ^ { \prime } ) > \log K ^ { \prime }$ , and balancing them gives $P _ { \mathrm { o u t } }$ and $\theta _ { \mathrm { o u t } }$ . For the local converse, each wire is active with probability $K ^ { \prime } / N$ Normalize its probe and condition on its own direction; the clean signals are 0 and $\langle w _ { \gamma } , f _ { \gamma } \rangle$ , of modulus at most $1 + o ( 1 )$ on a high-probability norm event, and the other active directions stay independent Gaussians given any input, with projection variance $( K ^ { \prime } - t ) / p$ for local state $t \in \{ 0 , 1 \}$ . The two-class argument of Appendix C applies with expected class counts $N - K ^ { \prime }$ and $K ^ { \prime } ,$ using only one-wire marginals. ■

For pairwise conjunction $K ^ { \prime } = \Theta ( k ^ { 2 } )$ , so the output stage can dominate the input dimension even though it reads one wire at a time. The hypotheses hold at the output threshold, for example, when log $m = o ( k ) .$ ; they must be imposed separately when combining with the input thresholds of the main text.

Comparison with an ideal execution Suppose an input stage produces bits $\widehat { y } _ { \gamma }$ and writes $\widehat { v } =$ $\textstyle \sum _ { \gamma } { \widehat { y } } _ { \gamma } f _ { \gamma }$ . Let $\mathsf { B } _ { \mathrm { i n } }$ be the event that some input-stage gate is wrong and $\mathsf { B } _ { \mathrm { o u t } }$ b bthe event that some bread of the ideal stream $v ^ { * }$ is wrong. Deterministically,

$$
\{ \mathrm { s o m e ~ f i n a l ~ r e a d ~ i s ~ w r o n g } \} \subseteq \mathsf { B } _ { \mathrm { i n } } \cup \mathsf { B } _ { \mathrm { o u t } } ,\tag{H.1}
$$

because outside $\mathsf { B } _ { \mathrm { i n } }$ the actual stream equals $v ^ { * }$ . An input budget with failure probability $\delta _ { \mathrm { i n } }$ and an output budget with $\delta _ { \mathrm { o u t } }$ therefore compose with failure at most $\delta _ { \mathrm { i n } } + \delta _ { \mathrm { o u t } }$ , without conditioning on earlier success and without independence of the two events. In particular the box threshold of Theorem 5.4 composes with proposition H.4.

The fresh-write circuit class Fix a feedforward Boolean threshold circuit and an input, independent of all dictionaries. Each input wire and each gate output has its own independent direction $f \sim { \cal N } ( 0 , I _ { d } / d )$ , and every active output is written once into the shared stream. A gate $\gamma$ of arity $r _ { \gamma } \leq r _ { 0 }$ reads previously written wires with $\begin{array} { r } { w _ { \gamma } = \sum _ { j } a _ { \gamma , j } f _ { w _ { j } } } \end{array}$ and bias $\theta _ { \gamma }$ , where the coeficients represent its clean threshold function, are fixed before the dictionary is drawn, satisfy $\| a _ { \gamma } \| = 1$ , and have margin $\tau _ { \gamma } = \mathrm { m i n } _ { z } ( 2 g _ { \gamma } ( z ) - 1 ) ( \langle a _ { \gamma } , z \rangle - \theta _ { \gamma } ) \geq \tau _ { 0 } > 0$ . Let � be the number of reads, � the initial activity and � the largest number of active directions in the ideal stream before a read; without erasure � is the input activity plus all earlier ideal active writes. Put $\kappa _ { \mathrm { m i n } } = \operatorname* { m i n } _ { \gamma } \tau _ { \gamma } ^ { 2 } / 2$

Theorem H.5 (Write budget). Assume $k \to \infty , Q \to \infty$ and log $Q = o ( k )$ . For fixed $\varepsilon > 0 ,$ , if $d \geq ( 1 + \varepsilon ) \kappa _ { \operatorname* { m i n } } ^ { - 1 }$ � log �, every gate of the circuit is evaluated correctly with probability at least $1 - Q ^ { - \varepsilon + o ( 1 ) }$ . If a fixed schedule exactly erases wires that are no longer needed, � may be replaced by the peak number of live active wires.

Proof of Theorem H.5. Let the ideal stream before layer ℓ contain the true values of all earlier gates. Given the input, its active set is deterministic and independent of the dictionary. For gate $\gamma ,$ condition on its at most $r _ { 0 }$ queried directions; the ideal interference from the other active wires is exactly Gaussian with variance at most $K \| w _ { \gamma } \| ^ { 2 } / d$ . On the local Gram event $\| G _ { \gamma } - I \| \leq \eta$ the probe norm is $1 + O ( \eta )$ and the clean score difers from $\langle a _ { \gamma } , z \rangle$ by ${ \cal { O } } ( \eta )$ , uniformly because the arity is bounded. Since $\tau _ { \gamma } \geq \tau _ { 0 } ,$ the Gaussian tail is at most $\mathrm { e x p } \{ - ( 1 - o ( 1 ) ) \dot { \kappa } _ { \mathrm { m i n } } d / K \} \le Q ^ { - ( 1 + \dot { \varepsilon } ) + o ( 1 ) }$ . Choose $\eta = ( \log Q / k ) ^ { 1 / 4 } \to 0 ;$ Gram concentration bounds the bad event by $C e ^ { - c \eta ^ { 2 } d }$ , and $\eta ^ { 2 } d / \log Q \gtrsim \eta ^ { 2 } k  \infty$ makes even its union over all � reads negligible. Let $\mathsf { B } _ { \gamma }$ be a wrong read of the ideal stream. Outside $\mathsf { U } _ { \gamma } \mathsf { B } _ { \gamma }$ , induction over layers shows that the actual and ideal streams agree before every read, so actual failure is contained in the union, and summing the marginal bounds gives $Q ^ { - \varepsilon + o ( 1 ) }$ . Under exact erasure by a fixed schedule the live active set at each ideal read is still determined by the input and the schedule and has at most � elements, and the induction preserves equality through erasure, because the subtracted actual and ideal contributions agree whenever all earlier reads do. ■

The theorem counts every read. Replacing � by the number of near-boundary gates is valid only if the remaining gates have a separately controlled sum of error probabilities. Non-local reads, error correction, reuse of directions, learned correlations and growing arity need further analysis. The connection to reliable computation from unreliable components (von Neumann, 1956; Evans and Schulman, 1999) is at the level of composing error bounds; here the errors come from shared geometry rather than independent faults.

## H.4 Unrestricted encodings

Let $\chi _ { k }$ be the �-subsets of $[ m ] , 2 \leq k \leq m - 1$ , and for $S \in {  { \mathcal { X } } } _ { k }$ let the target be the indicator vector of its pairs; distinct supports have distinct targets. Let $\varphi : X ^ { \prime } \to \mathbb { R } ^ { d }$ be any map on some $X ^ { \prime } \subseteq X _ { k }$

Proposition H.6 (Counting floor). Let � be any mapfrom the �-subsets of � to $\mathbb { R } ^ { d } .$ , read by � afine threshold units followed by arbitrary Boolean processing. If the outputs of all pairwise conjunctions are correct on afraction $1 - \delta > 0$ of supports, then $d \ge \left( k \log ( m / k ) + \log ( 1 - \delta ) \right) / ( 1 + \log T )$ For polynomial � and log $k = o ( \log m ) t h i s i s \Omega ( k )$

Proposition H.7 (A 2�-dimensional code). Encode a support � by the 2� nonleading coeficients of $\begin{array} { r } { p _ { S } ( y ) = \prod _ { l \in S } ( y - l ) ^ { 2 } } \end{array}$ . Then $x _ { i } x _ { j } = \mathbf { 1 } \{ p _ { S } ( i ) + p _ { S } ( j ) < 1 / 2 \}$ is an afine threshold of the code for every pair, so all pairwise conjunctions on �-sparse inputs have sign-rank at most $2 k + 1$

Proof of proposition H.6. If � afine threshold units followed by arbitrary Boolean processing are correct on $X ^ { \prime } ,$ , their sign vectors are distinct on $X ^ { \prime }$ . On the finite set $\varphi ( X ^ { \prime } )$ , ties can be removed without changing any output by moving each bias slightly upward, less than the smallest positive slack. Distinct sign vectors then occupy distinct open cells of an arrangement of � afine hyperplanes in $\mathbb { R } ^ { d } .$ , whose number $C ( T , d )$ satisfies $C ( T , d ) \leq C ( T - 1 , d ) + C ( T - 1 , d - 1 )$ with $C ( 0 , d ) = C ( T , 0 ) = 1$ , hence $\begin{array} { r } { C ( T , d ) \leq \sum _ { i \leq d } { \binom { T } { i } } } \end{array}$ (Buck, 1943; Cover, 1965; Winder, 1966). For $1 \leq d \leq T ,$ with $\begin{array} { r } { \vartheta = d / T , \vartheta ^ { d } \sum _ { i \leq d } \binom { T } { i } \leq ( 1 + \vartheta ) ^ { \dot { T } } \leq e ^ { d } } \end{array}$ , so the sum is at most $( e T / d ) ^ { d } \leq ( e T ) ^ { d } ;$ for $d > T$ use $2 ^ { T } \leq ( e T ) ^ { d }$ . Hence $| X ^ { \prime } | \leq ( e T ) ^ { d }$ , and $\vert X ^ { \prime } \vert \geq ( 1 - \delta ) { \binom { m } { k } } \geq ( 1 - \delta ) ( m / k ) ^ { k }$ gives the bound. ■

The guarantee concerns whole output vectors and cannot be replaced by a small average fraction of wrong gates: changing one support element changes only $2 ( k - 1 )$ pairs. The same count covers $L - 1$ hidden ReLU layers of widths $T _ { 1 } , \dots , T _ { L - 1 }$ followed by $T _ { L }$ threshold units: after removing ties, each layer is afine on every region cut out by the earlier activation patterns, so $\begin{array} { r } { | X ^ { \prime } | \leq \prod _ { \ell } \bar { \sum _ { i \leq d } } \binom { T _ { \ell } } { i } } \end{array}$ and $d \geq \log \lvert X ^ { \prime } \rvert / \sum _ { \ell } \log ( e T _ { \ell } )$ . Constant depth and polynomial width still need $\Omega ( k )$ dimensions in the skewed regime.

Proof of proposition H.7. Write $\begin{array} { r c l r } { p _ { S } ( y ) } & { = } & { \prod _ { l \in S } ( y - l ) ^ { 2 } ~ = ~ y ^ { 2 k } + \sum _ { i < 2 k } c _ { j } ( S ) y ^ { j } } \end{array}$ and $\varphi ( S ) \ =$ $( c _ { 0 } ( S ) , \dots , c _ { 2 k - 1 } ( S ) )$ . At an integer $i \in [ m ] , p _ { S } ( i ) = 0 { \mathrm { ~ i f ~ } } i \in S$ and $p _ { S } ( i ) \geq 1$ otherwise, so $x _ { i } x _ { j } = \mathbf { 1 } \{ p _ { S } ( i ) + p _ { S } ( j ) < 1 / 2 \}$ . For fixed $( i , j )$ the right side is an afine threshold of the 2� coeficients, the leading terms being absorbed into the bias; lifting the bias adds one coordinate, so the sign matrix has sign-rank at most $2 k + 1$ ■

With proposition $\mathrm { H } . 6 , \Theta ( k )$ dimensions are necessary and suficient for arbitrary encodings with one threshold per pair in the skewed regime. The code is nonlinear in the input bits, its coeficients are large and its normalized margins small. Unrestricted readout complexity alone would make dimension vacuous, since a single real number can encode a finite binary input to arbitrary precision; both the encoding and the readout must be specified when a dimension lower bound is interpreted.

## I Window formulas and the expected-count gap

The profiles are explicit: $\begin{array} { r } { \Psi _ { \beta } ( x ) = \int _ { 0 } ^ { \infty } \exp ( - \tau - \tau ^ { - \beta } e ^ { - \beta x } ) } \end{array}$ ��, with $\Psi _ { 1 } ( x ) = 2 e ^ { - x / 2 } K _ { 1 } ( 2 e ^ { - x / 2 } )$ for the modified Bessel function $K _ { 1 }$ and $\Psi _ { \beta } ( x )  \exp ( - e ^ { - x } )$ as $\beta \to \infty ;$ and $F _ { + } ( x ) = e ^ { - { \sqrt { c } } } +$ $\int _ { 0 } ^ { \sqrt { c } } e ^ { - c / s } d s$ with $c = e ^ { - x }$ . The optimized window is narrower than the midpoint window by the factor $4 ( 1 + \beta ) / ( 3 + \beta ) \in [ 2 , 4 )$ . The half-slack bias is optimal to first order, but its profile $\Psi _ { \beta } ( x / 2 ) F _ { + } ( x / 2 )$ is in general smaller than the supremum; a bias tuned to a target finite-size success level allocates the slack diferently. At the centers the limits are $\Psi _ { \beta } ( 0 )$ for the midpoint and $\Psi _ { \beta } ( 0 ) F _ { + } ( 0 )$ for the half-slack bias, with $F _ { + } ( 0 ) = 0 . 5 1 6 ;$ Section $9$ compares both with simulation.

![](images/ae29f673676ca01186f55a6b55e039af7c30c169ebbce0711998b2285b0887aa.jpg)  
Figure J.1. The four first-order thresholds of pairwise conjunction and the threshold of feature recovery, in units of � log �, as functions of log � log �. All four conjunction thresholds equal 32� log � when <sup>log</sup> <sup>�</sup> ∼ <sup>log</sup> <sup>�;</sup> <sup>they</sup> <sup>separate</sup> <sup>toward</sup> <sup>16,</sup> <sup>8,</sup> <sup>4</sup> <sup>and</sup> <sup>2</sup> <sup>(dotted).</sup> <sup>The</sup> <sup>vertical</sup> <sup>lines</sup> <sup>mark</sup> <sup>the</sup> <sup>regimes</sup> <sup>of</sup> <sup>the</sup> simulations in Section 9.

The Markov gap inside the window Conditionally on the active reads, the inactive reads are independent, so the number of inactive features that create at least one wrong one-active gate is binomial, and it is asymptotically Poisson with a random intensity Λ (Appendix F). At the midpoint, inside the window and along sequences with $\beta \to \beta _ { 0 } \in [ 1 , \infty )$ , Λ converges in law to $\Lambda _ { \infty } \stackrel { \bf { \tilde { \mu } } } { = } e ^ { \beta _ { 0 } ( G _ { A } - x ) }$ . Hence

$$
\mathbb { P } ( X = 0 ) \to \mathbb { E } e ^ { - \Lambda _ { \infty } } = \Psi _ { \beta _ { 0 } } ( x ) \in ( 0 , 1 ) ,
$$

while

$$
\mathbb { E } \Lambda _ { \infty } = e ^ { - \beta _ { 0 } x } \int _ { 0 } ^ { \infty } \tau ^ { - \beta _ { 0 } } e ^ { - \tau } d \tau = \infty ,
$$

and by Fatou’s lemma the expected number of firing inactive features, and with it E�, tends to infinity throughout the window. The limiting intensity is heavy-tailed: an unusually large active read lets an unusually large number of inactive features fire together. A typical draw has no error with probability $\Psi _ { \beta _ { 0 } } ( x )$ , while the mean is carried by rare draws. This is the second-order form of the gap between the ball and the box.

## J Further comparisons and experiments

## J.1 The four pairwise thresholds

## J.2 Comparison with other decoders

The ratio reflects the architecture rather than the task. A common bias for �-way conjunction <sup>must</sup> <sup>separate</sup> <sup>the</sup> <sup>sum</sup> <sup>of</sup> <sup>the</sup> <sup>�</sup> <sup>smallest</sup> <sup>active</sup> <sup>reads</sup> <sup>from</sup> <sup>the</sup> <sup>sum</sup> <sup>of</sup> <sup>the</sup> <sup>�</sup> − <sup>1</sup> <sup>largest</sup> <sup>active</sup> reads and the largest inactive read, so its box support contains $2 r - 1$ active levels, whereas that of recovery contains one. For $m = k ^ { 4 } .$ , where $\beta = 2 ,$ recovery needs 18� log � dimensions and direct pairwise conjunction 50� log �; the midpoint pairwise network needs 72� log � to be reliable and 80� log � for a vanishing expected count. Decoders that are not a single threshold layer can do better. The Lasso recovers the support from about 2� log �  � measurements (Wainwright, 2009), and $\ell _ { 1 }$ minimization recovers a typical nonnegative �-sparse vector from about $2 k \log ( m / k )$ noiseless Gaussian measurements (Donoho and Tanner, 2009). When log $k = o ( \log m )$ both are 2� log � to first order, the optimized reliability threshold of direct pairwise conjunction; when log � log � the second is � � log � , far below 32� log �. The constants of this paper are therefore those of one layer of threshold readouts of a fixed random code.

![](images/a019564b91df86c988bbafacd866adec19ae4873af47296123929eb5a885bf33.jpg)

![](images/2258d4aba5c8c58767b6906626cbed453bdf97777c1af2c8b4061e11e82d57bb.jpg)  
Figure J.2. Pairwise conjunction, 4096 exact draws per point, pointwise 95% Wilson bands. (a) Three deterministic biases and feature recovery evaluated on the same draws; the count-optimal and halfslack curves nearly coincide, and recovery switches on near ratio 0.39, left of the plotted range. (b) The half-slack bias of Theorem 6.1(a) as � grows; dashed curves are the profiles $\Psi _ { \hat { \beta } } ( x / 2 ) F _ { + } ( x / 2 )$ of Theorem 7.1(ii) at the slack $x = \hat { \lambda } _ { 1 } ( \sqrt { d / k } - 3 \hat { \lambda } _ { 1 } - \hat { \lambda } _ { 0 } )$

For asymmetric gates the weights matter as well as the bias: at the alternating threshold of the comparator $\mathbf { 1 } \{ \sum _ { l } 2 ^ { { \bar { l } } - 1 } z _ { l } \geq T \}$ the optimal weights are Fibonacci numbers, and when all classes count equally binary weights overstate the price by a factor exponential in � (Appendix E).

## J.3 Additional experimental figures

The first two panels of Figure 2 show the $k = 1 0 2 4$ comparisons. The figures below add the other activities, window profiles and bias rules.

## J.4 Full campaign results

The bias matters before the limit The first campaign (Figure J.2) evaluates three deterministic biases for pairwise conjunction on shared draws, together with feature recovery, at $k =$ 64, 256, 1024 with $m = k ^ { 4 }$ and at $k = 1 0 2 4$ with $m = k ^ { 2 } ,$ , with 13 dimensions per case and 4096 draws per cell. $\operatorname { A t } k = 1 0 2 4 , m = 1 0 2 4 ^ { 4 }$ and $d = 1 . 3 5 k ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ^ { 2 }$ , the half-slack bias of Theorem 6.1(a) succeeds on 3984 of 4096 draws, the bias optimized for counting on 3912, and the midpoint on 2. At the smallest dimension of the grid feature recovery succeeds on 4056 draws and every conjunction rule on none, as corollary 6.3 predicts. At the quantile center the three cases with $m = k ^ { 4 }$ succeed with frequencies 0.143, 0.139 and 0.155, and Theorem 7.1(ii) predicts $\Psi _ { \hat { \beta } } ( 0 ) F _ { + } ( 0 ) = 0 . 1 5 5 , 0 . 1 5 4$ and 0.154 for the half-slack bias, with $\begin{array} { r } { \hat { \beta } = \hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 } ; } \end{array}$ the dashed curves in Figure J.2(b) are the whole predicted profiles, and Table L.4 adds the case $m = k ^ { 2 }$ and an independent replication.

![](images/8b5b5a931b4cdc8e9a237c328517196e230371971287c7229f968625cefef05d.jpg)  
Figure J.3. Reliability of four threshold gates at $m = k ^ { 4 }$ with the half-slack bias of Theorem 6.1(a) for $k = 6 4 , 2 5 6 , 1 0 2 4 _ { \AA }$ , and with the midpoint bias and the sample-adaptive oracle bias for $k = 1 0 2 4$ . The horizontal axis is the dimension in units of the quantile box center of each gate. Dashed line: that center; dotted line: the midpoint box threshold of Theorem 6.1(b) in the same units.

The box beyond conjunction The second campaign tests Theorem 6.1(a) for AND<sub>2</sub>, AND<sub>3</sub>, OR<sub>2</sub> and $\mathrm { M A J } _ { 3 }$ at $m = k ^ { 4 } \left( { \mathrm { F i g u r e } } J . 3 \right)$ . In units of the quantile box center $k ( ( 2 t - 1 ) \hat { \lambda } _ { 1 } + ( 2 r - 2 t + 1 ) \hat { \lambda } _ { 0 } ) ^ { 2 }$ all four gates show the same picture at $k = 1 0 2 4 \colon$ success is at most 3.8% at 0.9, between 15.6% and 30.7% at 1, and between 82.7% and 97.9% at 1.2, and the curves steepen with �. The oracle, which chooses the bias after seeing the reads, reaches 50% success between 0.98 and 1.03 times the center, 2% to 7% earlier in dimension than the half-slack rule and consistent with Theorem 5.4(ii). The midpoint rule of each gate switches on near its own box threshold, marked by a dotted line; for $\mathrm { A N D } _ { 2 }$ it succeeds on none of the 4096 draws at 1.2.

Separation of the two thresholds The third campaign fixes the midpoint pairwise network at $k = 1 0 2 4$ and increases log � log � from 4 to 16 to 64, so that $\beta = 2 , 4 , 8$ (Figure 2). Reliability switches on at the box: at the center the success frequencies 0.295, 0.324 and 0.340 match the limits $\Psi _ { \hat { \beta } } ( 0 ) = 0 . 2 9 8 , 0 . 3 2 6$ and 0.346 of Theorem 7.1(i). The expected number of wrong outputs crosses one at about 1.23, 1.45 and 1.67 times the center $d _ { 0 }$ of (14), tracking the first-order ball positions $2 ( 1 + \beta ^ { 2 } ) / ( 1 + \beta ) ^ { 2 } = 1 . 1 1$ , 1.36 and 1.60 with ofsets 0.12, 0.09 and 0.06 that shrink as � grows. At $m = k ^ { 6 4 }$ and $d = 1 . 5 d _ { 0 }$ all 4096 draws are correct, while $\mathbb { E } X \approx 2 . 8 \times 1 0 ^ { 1 9 }$ . Values of � this large are a mathematical stress test of the first-order theory, reachable because the exact law never builds the dictionary.

## K Finite checks

The checks below test identities, closed forms and implementations on finite instances. The code-and-data archive contains the scripts and outputs.

The box identity The script check\_box\_identity.py draws 400 random instances with $r \in$ $\{ 1 , 2 , 3 \}$ , pools of sizes between 2� and $2 r + 2 ,$ Gaussian reads and Gaussian weights (every third instance with nonnegative weights). For every class � and both error directions it enumerates all ordered tuples of distinct features and compares the maximum of $\langle b , \xi _ { I } \rangle$ with the rearrangement formula of lemma B.2. All 3720 class–direction pairs agree to within $1 . 8 \times 1 0 ^ { - 1 5 }$

The two programs The script check\_programs.py solves the ball program (11), with the radii $| | \lambda _ { z } | |$ in units $\lambda _ { 1 } = 1 , \lambda _ { 0 } = \beta ,$ as a second-order cone program, and the box program of proposition B.3 by bisection over linear programs, in both cases over all representations $( a , \theta )$ without imposing symmetry. For every THR<sup>�</sup> with $r \leq 4$ and every $\beta \in \{ 1 , 1 . 5 , 2 , 4 , 1 0 \}$ , the optimal values agree with (a) and (d) of Theorem 6.1, and the midpoint prices with (b) and (c), to relative accuracy $8 \times 1 0 ^ { - 7 }$ . The optimal box weights returned by the solver are equal across positions up to solver tolerance. At $\beta = 1$ all four values equal $4 r ^ { 2 }$ in these units.

The reduced sampler The script check\_thr\_direct.py compares the exact reduced sampler with direct simulation: for each draw it generates a Gaussian dictionary, forms $u ,$ and enumerates every �-subset. For the largest negative and the smallest positive score, the two-sample Kolmogorov–Smirnov statistics over 4000 draws per method are as follows.
<table><tr><td>Gate,  $( k , m , d )$ </td><td>largest negative</td><td>p</td><td>smallest positive</td><td> $p$ </td></tr><tr><td> $\mathrm { A N D } _ { 3 } , ( 8 , 3 0 , 1 2 0 )$ </td><td>0.013</td><td>0.87</td><td>0.025</td><td>0.18</td></tr><tr><td> $\mathrm { O R } _ { 2 } , ( 8 , 3 0 , 1 2 0 )$ </td><td>0.020</td><td>0.42</td><td>0.022</td><td>0.31</td></tr><tr><td> $\mathrm { M A J } _ { 3 } , ( 8 , 3 0 , 1 2 0 )$ </td><td>0.020</td><td>0.40</td><td>0.024</td><td>0.22</td></tr><tr><td>recovery, (8, 40, 60)</td><td>0.015</td><td>0.76</td><td>0.014</td><td>0.81</td></tr></table>

For pairwise conjunction the frozen campaign of Appendix L used an independent directmatrix implementation at $( k , m , d ) = ( 8 , 4 0 , 1 0 0 )$ with 12000 draws per method: the Kolmogorov– Smirnov distances for the largest active read, the sum of the two smallest active reads, the largest inactive read and the largest negative score are 0.0121, 0.0112, 0.0075 and 0.0117. A separate enumeration confirms the all-pairs success identity of proposition B.1 and lemma B.2 at five biases.

Algebra The script verify\_algebra\_v2.py checks all 1013 nonconstant comparator thresholds of arities 1 to 9 against their 348502 truth-table entries: the unit-gap construction, the Fibonacci bound on every weight, and attainment by the alternating threshold. It evaluates the polynomial code of proposition H.7 in integer arithmetic on 2427 supports $( 3 ~ \leq ~ m ~ \leq ~ 1 1 , ~ 2 ~ \leq ~ k ~ \leq$ <sup>min</sup>(<sup>5,</sup> <sup>�</sup> − <sup>1</sup>)<sup>),</sup> <sup>checking</sup> <sup>106658</sup> <sup>pair</sup> <sup>outputs,</sup> <sup>and</sup> <sup>verifies</sup> <sup>the</sup> <sup>row</sup> <sup>cumulants</sup> <sup>and</sup> <sup>the</sup> <sup>coeficient</sup> $c _ { \nu }$ of proposition H.1 symbolically.

## L Experimental protocol and additional results

## L.1 Sampling

Each draw samples $H \sim \chi _ { d } ^ { 2 } / d$ and � standard Gaussians, forms the active reads of (8), and sorts them. For the inactive sample only its � smallest and � largest values are needed $( j = 2$ for pairwise conjunction, $j = r$ in general). Writing each inactive value through its distribution function, the smallest of � uniforms is $1 - U ^ { 1 / n }$ ; conditionally on the smallest � values, the other points are uniform above the $j \mathrm { t h }$ , and their largest values follow by the same rule on the shorter interval. The $2 j$ uniforms are mapped through $\Phi ^ { - 1 }$ in the upper-tail parametrization, using expm1 and logarithms to keep precision when � is as large as $\bar { k } ^ { 6 4 }$ . Success is checked by proposition B.1 and lemma B.2 with the strictness of the readout; ties have probability zero. $\mathrm { A }$ draw costs � � log � time and $O ( k )$ memory, independent of � and $d ,$ while direct simulation needs $O ( d m )$ memory.

Expected error counts are computed by one-dimensional quadrature of proposition 3.2, $\begin{array} { r } { \mathbb { E } X = \sum _ { j } C _ { j } p _ { j } } \end{array}$ with the exact class sizes $\begin{array} { r } { \dot { C } _ { j } = \binom { k } { j } \binom { n } { r - j } } \end{array}$ of the unordered family, integrating over � in the logarithmic domain between chi-square quantiles of probability $1 0 ^ { - 1 5 }$ with relative tolerance $1 0 ^ { - 1 0 }$ . For pairwise conjunction this reproduces the frozen values of the first campaign, for example 6.7416 and 0.32739 at ratios 1.2 and 1.35 below.

## L.2 Campaign E1: pairwise conjunction (frozen 22 September 2026)

The grid uses $m = k ^ { 4 }$ with $k = 6 4 , 2 5 6 , 1 0 2 4$ and $m = k ^ { 2 }$ with $k = 1 0 2 4 ,$ the dimension ratios $\{ 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 , 1 , 1 . 1 , 1 . 2 , 1 . 3 5 , 1 . 5 , 1 . 7 5 , 2 , 2 . 5 , 3 \}$ relative to $k ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ^ { 2 }$ , rounded to integers, and 4096 draws per cell: 52 cells and 212992 independent draws. Seed 20260922 generates one SeedSequence per cell. The three biases, fixed before sampling, are the midpoint $3 / 2$ , the bias optimized for counting $1 + \sqrt { \log N _ { - } } / ( \sqrt { \log N _ { - } } + \sqrt { \log N _ { + } } )$ with $N _ { - } = k n$ and $\begin{array} { r } { N _ { + } = { \binom { k } { 2 } } } \end{array}$ , and the half-slack bias $3 / 2 + \textstyle { \frac { 1 } { 2 } } \sigma _ { d } ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } )$ of Theorem $6 . 1 ( \mathsf { a } ) .$ ; feature recovery uses $\begin{array} { r } { \frac { 1 } { 2 } + \frac { 1 } { 2 } \sigma _ { d } ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } ) } \end{array}$ . All rules share the draws of a cell. The sample-adaptive event that the largest negative score lies below the smallest positive score is stored as a feasibility bound and is not counted as a rule. No cell or draw was excluded. The run used Python 3.11.3, NumPy 1.26.4 and SciPy 1.16.3 on one CPU node with one BLAS thread, 81 seconds of CPU time and about 221 MiB of memory; runner and configuration hashes match the recorded run.

Estimating the mean is harder than estimating reliability The mean count is carried by rare draws, so Monte Carlo estimates of it can mislead even when their standard errors are small. At $k = 1 0 2 4 , m = 1 0 2 4 ^ { 4 }$ and ratio 1.35, the Rao–Blackwellized sample mean of � is 0.063 with standard error 0.011, while quadrature of the exact one-gate law gives 0.327 (Table L.1). The main text therefore reports quadrature values of E� and uses simulation only for $\mathbb { P } ( X = 0 )$ .

![](images/4fdeeb0ce1a4828fdce70aceab3d7334362512b434d029556f5ec978bcdf0179.jpg)  
Figure L.1. All four cases of campaign E1. Each point uses 4096 independent draws; bands are pointwise 95% Wilson intervals. The three bias rules and feature recovery share each cell’s draws.

## L.3 Campaigns E2 and E3 (frozen 27 September 2026)

The configuration box-v1.json was written before any run. Campaign E2 covers $\mathrm { A N D } _ { 2 } , \mathrm { A N D } _ { 3 } ,$ $\mathrm { O R } _ { 2 }$ and $\mathrm { M A J } _ { 3 }$ with $k = 6 4$ , 256, 1024 and $m = k ^ { 4 }$ , the same 13 dimension ratios relative to the quantile box center $k ( ( 2 t - 1 ) \hat { \lambda } _ { 1 } + ( 2 r - 2 t + 1 ) \hat { \lambda } _ { 0 } ) ^ { 2 }$ , and 4096 draws per cell; the rules are the half-slack bias $\begin{array} { r } { t - \frac { 1 } { 2 } + \frac { 1 } { 2 } \sigma _ { d } ( \hat { \lambda } _ { 0 } - \hat { \lambda } _ { 1 } ) } \end{array}$ , the midpoint $t - { \frac { 1 } { 2 } }$ , and the oracle, which succeeds on a draw when some common bias separates its largest negative score from its smallest positive score; by Theorem 5.4(ii) even this sample-adaptive rule fails below the box threshold. Campaign E3 fixes the midpoint $\mathrm { A N D } _ { 2 }$ network at $k = 1 0 2 4$ with $m = k ^ { 4 } , k ^ { 1 6 } , k ^ { 6 4 }$ and 11 ratios relative to $4 k ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ^ { 2 }$ . Seed 20260927 generates one SeedSequence per cell. The run used Python 3.13.9, NumPy 1.26.4 and SciPy 1.16.3 on a laptop CPU and took 188 seconds; all per-draw extremes are stored.

## L.4 Window profiles at the center

Table L.4 compares the success frequencies at the quantile centers with the limits of Theorem 7.1, evaluated at $\hat { \hat { \beta } } = \hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 } \colon \Psi _ { \hat { \beta } } ( 0 ) F _ { + } ( \hat { 0 } )$ for the half-slack bias, whose slack splits evenly, and $\Psi _ { \hat { \beta } } ( 0 )$ for the midpoint. The pairwise rows of campaign E2 use the same rule as E1 with independent

Table L.1. The half-slack bias at $k = 1 0 2 4 ,$ $m = 1 0 2 4 ^ { 4 }$ (campaign E1). Mean error is the Rao–Blackwellized sample mean over the inactive reads with its Monte Carlo standard error; the last column is the quadrature of proposition 3.2. Rare draws carry the mean, so the sample mean and its standard error can both be misleading, as at ratio 1.35.
<table><tr><td>Ratio</td><td> $d$ </td><td> $\mathbb { P } ( X = 0 )$  [95% interval]</td><td>MC mean</td><td>MC s.e.</td><td>Quadrature</td></tr><tr><td>0.60</td><td>164032</td><td>0.0000 [0.0000, 0.0009]</td><td> $2 . 4 6 2 \mathrm { e } { + } 0 6$ </td><td> $4 . 1 9 9 \mathrm { e } + 0 5$ </td><td> $2 . 4 8 5 \mathrm { e } { + } 0 6$ </td></tr><tr><td>1.00</td><td>273387</td><td>0.1548 [0.1440, 0.1662]</td><td>312.3</td><td>127.7</td><td>427.3</td></tr><tr><td>1.20</td><td>328064</td><td>0.8252 [0.8133, 0.8365]</td><td>10.46</td><td>8.232</td><td>6.742</td></tr><tr><td>1.35</td><td>369072</td><td>0.9727 [0.9672, 0.9772]</td><td>0.06288</td><td>0.01097</td><td>0.3274</td></tr><tr><td>1.50</td><td>410081</td><td>0.9976 [0.9955, 0.9987]</td><td>0.002693</td><td>0.0008277</td><td>0.0178</td></tr><tr><td>1.75</td><td>478427</td><td>0.9995 [0.9982, 0.9999]</td><td>0.0004904</td><td>0.0003452</td><td>0.0002069</td></tr><tr><td>2.00</td><td>546774</td><td>1.0000 [0.9991, 1.0000]</td><td>1.202e-07</td><td>1.188e-07</td><td>4.052e-06</td></tr></table>

Table L.2. Campaign E2 at $k = 1 0 2 4 , m = k ^ { 4 } ;$ : success frequency of the half-slack rule (out of 4096) and of the oracle at selected dimension ratios to the quantile box center.
<table><tr><td>Ratio</td><td>0.8</td><td>0.9</td><td>1</td><td>1.1</td><td>1.2</td><td>1.35</td><td>1.5</td></tr><tr><td> $\mathrm { A N D } _ { 2 } ,$  half-slack</td><td>0</td><td>21</td><td>668</td><td>2262</td><td>3387</td><td>3980</td><td>4081</td></tr><tr><td> $\mathrm { A N D } _ { 2 } ,$  oracle</td><td>1</td><td>141</td><td>1573</td><td>3305</td><td>3952</td><td>4092</td><td>4096</td></tr><tr><td> $\mathrm { A N D } _ { 3 } ,$  half-slack</td><td>1</td><td>154</td><td>1258</td><td>2919</td><td>3758</td><td>4065</td><td>4093</td></tr><tr><td> $\mathrm { A N D } _ { 3 } ,$  oracle</td><td>11</td><td>571</td><td>2456</td><td>3788</td><td>4077</td><td>4095</td><td>4096</td></tr><tr><td>OR2, half-slack</td><td>0</td><td>0</td><td>640</td><td>3298</td><td>4009</td><td>4092</td><td>4096</td></tr><tr><td> $\mathrm { O R } _ { 2 } ,$  oracle</td><td>0</td><td>0</td><td>1308</td><td>3944</td><td>4094</td><td>4096</td><td>4096</td></tr><tr><td> $\mathrm { M A J } _ { 3 } ,$  half-slack</td><td>0</td><td>0</td><td>738</td><td>3167</td><td>4011</td><td>4095</td><td>4096</td></tr><tr><td> $\mathrm { M A J } _ { 3 } ,$  oracle</td><td>0</td><td>2</td><td>1602</td><td>3865</td><td>4094</td><td>4096</td><td>4096</td></tr></table>

Table L.3. Campaign E3, the midpoint pairwise network at $k = 1 0 2 4 \colon$ success frequency (out of 4096) and exact expected number of wrong outputs at selected dimension ratios to $d _ { 0 } = \hat { 4 } k ( \hat { \lambda } _ { 1 } \dot { { } ^ { - } } \hat { \lambda } _ { 0 } ) ^ { 2 }$
<table><tr><td rowspan="2">Ratio</td><td colspan="2"> $m = k ^ { 4 }$ </td><td colspan="2"> $m = k ^ { 1 6 }$ </td><td colspan="2"> $m = k ^ { 6 4 }$ </td></tr><tr><td>succ.</td><td>EX</td><td>succ.</td><td>EX</td><td>succ.</td><td>EX</td></tr><tr><td>0.9</td><td>33</td><td> $5 . 8 1 \times 1 0 ^ { 3 }$ </td><td>0</td><td> $9 . 0 3 \times 1 0 ^ { 1 8 }$ </td><td>0</td><td> $2 . 0 1 \times 1 0 ^ { 8 9 }$ </td></tr><tr><td>1</td><td>1208</td><td>422</td><td>1327</td><td> $3 . 2 9 \times 1 0 ^ { 1 5 }$ </td><td>1392</td><td> $4 . 5 1 \times 1 0 ^ { 7 7 }$ </td></tr><tr><td>1.1</td><td>3191</td><td>30.9</td><td>3896</td><td> $1 . 2 1 \times 1 0 ^ { 1 2 }$ </td><td>4091</td><td> $1 . 0 2 \times 1 0 ^ { 6 6 }$ </td></tr><tr><td>1.2</td><td>3922</td><td>2.26</td><td>4089</td><td> $4 . 4 3 \times 1 0 ^ { 8 }$ </td><td>4096</td><td> $2 . 3 2 \times 1 0 ^ { 5 4 }$ </td></tr><tr><td>1.35</td><td>4084</td><td>0.0453</td><td>4096</td><td> $3 . 1 5 \times 1 0 ^ { 3 }$ </td><td>4096</td><td> $7 . 9 8 \times 1 0 ^ { 3 6 }$ </td></tr><tr><td>1.5</td><td>4095</td><td> $9 . 1 0 \times 1 0 ^ { - 4 }$ </td><td>4096</td><td> $0 . 0 2 2 5$ </td><td>4096</td><td> $2 . 7 6 \times 1 0 ^ { 1 9 }$ </td></tr><tr><td>1.75</td><td>4096</td><td> $1 . 3 7 \times 1 0 ^ { - 6 }$ </td><td>4096</td><td> $6 . 0 1 \times 1 0 ^ { - 1 1 }$ </td><td>4096</td><td> $2 . 2 2 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>2</td><td>4096</td><td> $2 . 0 7 \times 1 0 ^ { - 9 }$ </td><td>4096</td><td> $1 . 6 2 \times 1 0 ^ { - 1 9 }$ </td><td>4096</td><td> $1 . 7 9 \times 1 0 ^ { - 3 9 }$ </td></tr></table>

seeds. $\operatorname { A t } k = 1 0 2 4$ all six limits lie inside the 95% intervals; at $k = 6 4$ and 256 the frequencies lie between 0.006 and 0.015 below the limits, a finite-size lag that disappears by $k = 1 0 2 4$

Table L.4. Success frequency at the quantile center (dimension ratio 1, 4096 draws) against the limit of Theorem 7.1.
<table><tr><td>Campaign, rule</td><td>m</td><td> $k$ </td><td> $\hat { \beta }$ </td><td>Limit</td><td>Simulation [95% interval]</td></tr><tr><td>E1, half-slack</td><td> $k ^ { 4 }$ </td><td>64</td><td>2.458</td><td>0.155</td><td>0.143 [0.133, 0.154]</td></tr><tr><td>E1, half-slack</td><td> $k ^ { 4 }$ </td><td>256</td><td>2.342</td><td>0.154</td><td>0.139 [0.129, 0.150]</td></tr><tr><td>E1, half-slack</td><td> $k ^ { 4 }$ </td><td>1024</td><td>2.275</td><td>0.154</td><td>0.155 [0.144, 0.166]</td></tr><tr><td>E1, half-slack</td><td> $k ^ { 2 }$ </td><td>1024</td><td>1.538</td><td>0.147</td><td>0.149 [0.139, 0.160]</td></tr><tr><td>E2, half-slack</td><td> $k ^ { 4 }$ </td><td>64</td><td>2.458</td><td>0.155</td><td>0.149 [0.138, 0.160]</td></tr><tr><td>E2, half-slack</td><td> $k ^ { 4 }$ </td><td>256</td><td>2.342</td><td>0.154</td><td>0.141 [0.131, 0.152]</td></tr><tr><td>E2, half-slack</td><td> $k ^ { 4 }$ </td><td>1024</td><td>2.275</td><td>0.154</td><td>0.163 [0.152, 0.175]</td></tr><tr><td>E3, midpoint</td><td> $k ^ { 4 }$ </td><td>1024</td><td>2.275</td><td>0.298</td><td>0.295 [0.281, 0.309]</td></tr><tr><td>E3, midpoint</td><td> $k ^ { 1 6 }$ </td><td>1024</td><td>4.730</td><td>0.326</td><td>0.324 [0.310, 0.338]</td></tr><tr><td>E3, midpoint</td><td> $k ^ { 6 4 }$ </td><td>1024</td><td>9.570</td><td>0.346</td><td>0.340 [0.325, 0.354]</td></tr></table>

## M Applications, implications and open problems

A threshold in this paper answers a design question: how wide must a shared stored vector be before a family of logical reads becomes dependable? The answer depends on what “dependable” means, on the bias, and on what else the vector carries. This section turns the results into five rules that make these dependencies explicit. Appendix M.1 orders the error criteria by the width they need. Appendix M.2 inverts the thresholds into a feature capacity at fixed width. Appendix M.3 converts the window profiles into the width for a prescribed success probability. Appendix M.4 checks every gate of a universal layer on any realized representation from a few extreme reads. Appendix M.5 separates the part of a threshold paid for activity from the part paid for the read family. Appendices M.6 and M.7 draw the consequences for evaluation and for claims about superposition, and Appendix M.8 states open problems. The corollaries and propositions below are proved from the results of the paper. Numbers new to this section are computed by check\_sizing\_rule.py from the window profiles and the frozen campaigns of Appendix L, without new sampling; the others are quoted from Appendices J and L.

## M.1 Which failure, which width

A layer of   gates can be judged by the fraction of wrong outputs, by the expected number of wrong outputs, by the probability that some output is wrong, or by correctness on every input. The four criteria protect diferent objects: a typical gate, an additive loss, a typical input and every input. The widths they require difer by more than constants, and the first criterion carries no logarithm at all.

Corollary M.1 (Average accuracy carries no logarithm). Let $\mathcal { F }$ be any gate family of fixed arity �, let $( a , \theta ) \in \operatorname { R e p } ( g )$ be fixed, and use the canonical rule. Then E� $/ | \mathcal { F } |  0$ if and only if $d / k  \infty$ No further condition on � and no condition on the structure of  is needed.

Proof. A fixed gate has class � with probability $N _ { z } / | \mathcal { F } | ,$ so $\begin{array} { r } { \mathbb { E } X / | \mathcal F | = \sum _ { z } ( N _ { z } / | \mathcal F | ) p _ { z } } \end{array}$ is a convex combination of the $2 ^ { r }$ class error probabilities. By $( \mathsf C . 4 ) , p _ { z } = \mathbb E \overline { { \Phi } } ( T _ { z } )$ with $T _ { z } \ =$ $( s _ { z } - \varsigma _ { z } \langle a , z \rangle ( J - 1 ) ) / ( v _ { z } \sqrt { J / d } )$ and $( k - r ) \| a \| ^ { 2 } \leq v _ { z } ^ { 2 } \leq k \| a \| ^ { 2 }$ . If $d / k \to \infty ,$ , then $J  1$ in probability, and $s _ { z } \sqrt { d } / ( \| a \| \sqrt { k } ) $ because $s _ { z }$ and � are fixed. Hence $T _ { z }  \infty$ in probability, $p _ { z }  0$ for every class by bounded convergence, and so does the convex combination. If instead $d \leq C k$ along a subsequence, then on $\{ 1 / 2 \le J \le 2 \}$ the bound (C.3) gives a numerator at most $3 { \sqrt { r } } \| a \|$ and a denominator at least $\| a \| { \sqrt { ( k - r ) / ( 2 d ) } }$ , so $T _ { z } \leq 6 \sqrt { r C }$ once $k \geq 2 r$ . The probability of this event is positive for every � and tends to one, so it is at least some $c _ { 0 } > 0$ Thus $p _ { z } \geq c _ { 0 } { \overline { { \Phi } } } ( 6 { \sqrt { r C } } )$ for every class, and the convex combination stays bounded away from zero. ■

The average is dominated by the most numerous class. For sparse inputs that class has no active input, and its gates fail only when a Gaussian interference of scale $\sigma _ { d }$ crosses a fixed margin; no extreme value enters. Campaign E1 shows the size of this blind spot. $\operatorname { A t } k = 1 0 2 4 .$ $m = 1 0 2 4 ^ { 4 }$ and the quantile box centre, the half-slack network has a wrong output on 85% of draws and an expected count of 427, while the expected fraction of wrong outputs is $7 . 1 \times 1 0 ^ { - 2 2 }$ At 0.6 times the centre no draw is correct, and the fraction is $4 . 1 \times 1 0 ^ { - 1 8 }$ . The zero-active class contains all but about a fraction $2 k / m$ of the gates, and at the centre it contributes $1 0 ^ { - 5 4 }$ expected errors.

Table M.1. Four error criteria for the universal family of pairwise conjunctions when log $k = o ( \log m )$ , in order of the width they require. The first row is a condition on $d / k _ { , }$ , the middle rows are sharp first-order thresholds, and the last row is sharp up to constants. With optimized biases, the reliability threshold holds against every common representation and the expected-count threshold against every gate-local rule.
<table><tr><td>Criterion</td><td>Protects</td><td>Midpoint bias</td><td>Optimized bias</td><td>Source</td></tr><tr><td> $\mathbb { E } X / | \mathcal { F } |  0$ </td><td>a typical gate</td><td> $d / k \to \infty ( \mathrm { a n y }$ </td><td>fixed bias)</td><td>Corollary M.1</td></tr><tr><td> $\mathbb { P } ( X = 0 )  1$ </td><td>a typical input</td><td>8k log m</td><td>2k log m</td><td>Theorem 6.1(a,b)</td></tr><tr><td> $\mathbb { E } X  0$ </td><td>the total additive loss</td><td>16k log m</td><td>4k log m</td><td>Theorem 6.1(c,d)</td></tr><tr><td>correct on every support</td><td>every input, chosen after E</td><td> $\Theta ( k ^ { 2 } \log m )$ </td><td></td><td>Proposition 8.1</td></tr></table>

Table M.1 is a specification checklist. A sentence such as “the layer computes every pairwise conjunction in superposition at width $d ^ { \prime \prime }$ becomes checkable once it names a row. At width of order � log � the first row holds, the last row fails, and the middle rows are decided by constants that range over a factor of eight. The row follows from how the outputs are consumed. A batch of answers that is discarded after one error needs the second row. A cost that accrues per wrong answer needs the third. Inputs that may be chosen with knowledge of the dictionary need the fourth, and for midpoint probes the widths of the middle rows do not buy it: by proposition 8.1, a midpoint network whose width is of order � log � fails on some support with probability tending to one, since such a width is eventually below $c k ^ { 2 }$ log �.

## M.2 Feature capacity at fixed width

Superposition is usually measured by how many features fit into � dimensions. The thresholds can be read the same way: fix � and $k ,$ and ask for the largest dictionary under which a criterion still holds. When $d / ( k \log k ) \to \infty$ this capacity is exponential in $d / k ,$ , and the criterion and the bias change the constant in the exponent.

Corollary M.2 (Capacity at fixed width). Let $\mathcal { F } = \mathcal { U } _ { r } ( \mathrm { T H R } _ { t } ^ { r } )$ , let <sup>�</sup> → ∞ with $d / ( k \log k ) \to \infty ,$ and fix $ { \varepsilon } \in ( 0 , 1 )$ . For the four parts of Theorem 6.1 put

$$
c _ { \mathrm { a } } = ( 2 r - 2 t + 1 ) ^ { 2 } , \quad c _ { \mathrm { b } } = 4 ( r - t + 1 ) ^ { 2 } , \quad c _ { \mathrm { c } } = 4 r ( r - t + 1 ) , \quad c _ { \mathrm { d } } = r \big ( \sqrt { r - t } + \sqrt { r - t + 1 } \big ) ^ { 2 } .
$$

For each part $\bullet \in \{ \mathrm { a } , \mathrm { b } , \mathrm { c } , \mathrm { d } \} \colon i f m / k \to \infty$ and log $m \leq ( 1 - \varepsilon ) d / ( 2 c _ { \bullet } k )$ , its criterion holds for the representation named there; if log $m \geq ( 1 + \varepsilon ) d / ( 2 c _ { \bullet } k )$ , its criterion fails, in part (a) for every common representation and in part (d) for every gate-local rule.

Proof. Each threshold of Theorem 6.1 has the form $k Q ( \lambda _ { 1 } , \lambda _ { 0 } ) ( 1 + o ( 1 ) )$ , where � is continuous, nondecreasing in each argument and positively homogeneous of degree two, with $Q ( 0 , 1 ) = c _ { \bullet } ;$ for example, $Q = ( ( 2 t - 1 ) \lambda _ { 1 } + ( 2 r - 2 t + 1 ) \lambda _ { 0 } ) ^ { 2 }$ in part (a) and $Q = r ( \rho _ { t } + \rho _ { t - 1 } ) ^ { 2 }$ in part (d). Suppose first that log $m \leq ( 1 - \varepsilon ) d / ( 2 c _ { \bullet } k )$ , and put $\Lambda ^ { 2 } = ( 1 - \varepsilon ) d / ( c _ { \bullet } k )$ . Then $\lambda _ { 0 } \leq \Lambda$ , and $\lambda _ { 1 } / \Lambda  0$ because $d / ( k \log k ) \to \infty$ . Monotonicity and homogeneity give

$$
k Q ( \lambda _ { 1 } , \lambda _ { 0 } ) \leq k \Lambda ^ { 2 } Q ( \lambda _ { 1 } / \Lambda , 1 ) = ( 1 - \varepsilon ) d ( 1 + o ( 1 ) ) ,
$$

so for large � the width exceeds the threshold by a factor at least $1 + \varepsilon / 2 ,$ , and Theorem 6.1 gives the criterion. Suppose next that log $m \geq ( 1 + \varepsilon ) d / ( 2 c _ { \bullet } k )$ . Then log � $\iota / { \log k }  \infty ,$ so $m / k $ 8 and $\lambda _ { 0 } ^ { 2 } = 2 \log m \left( 1 + o ( 1 ) \right)$ , and

$$
k Q ( \lambda _ { 1 } , \lambda _ { 0 } ) \geq k Q ( 0 , \lambda _ { 0 } ) = c _ { \bullet } k \lambda _ { 0 } ^ { 2 } \geq ( 1 + \varepsilon ) d ( 1 - o ( 1 ) ) .
$$

The width is thus below the threshold by a fixed factor, and the failure statements and converses of Theorem 6.1 apply. ■

Table M.2. Capacity constants of corollary M.2. At width � and activity $k ,$ the criterion tolerates dictionaries with log � up to $d / ( 2 c _ { \bullet } k )$ to first order, so a smaller constant means exponentially more features. Columns follow the parts of Theorem 6.1.
<table><tr><td rowspan="2">Gate</td><td colspan="2">Reliability</td><td colspan="2">Expected count</td></tr><tr><td>(a) half-slack</td><td>(b) midpoint</td><td>(c) midpoint</td><td>(d) optimized</td></tr><tr><td> $\mathrm { A N D } _ { 2 }$ </td><td>1</td><td>4</td><td>8</td><td>2</td></tr><tr><td> $\mathrm { A N D } _ { 3 }$ </td><td>1</td><td>4</td><td>12</td><td>3</td></tr><tr><td> $\mathrm { O R } _ { 2 }$ </td><td>9</td><td>16</td><td>16</td><td> $6 + 4 \sqrt { 2 } \approx 1 1 . 6 6$ </td></tr><tr><td> $\mathrm { M A J } _ { 3 }$ </td><td>9</td><td>16</td><td>24</td><td> $9 + 6 { \sqrt { 2 } } \approx 1 7 . 4 9$ </td></tr><tr><td> $\mathrm { R e c o v e r y , T H R _ { 1 } ^ { 1 } }$ </td><td>1</td><td>4</td><td>4</td><td>1</td></tr></table>

Because the capacity is exponential, a ratio of constants becomes a power of the dictionary size. For �-way conjunction $c _ { \mathsf { a } } = 1$ and $c _ { \mathrm { d } } = r \colon$ at equal width, the largest dictionary with a vanishing expected count is the �-th root of the largest reliable one, in the sense that the ratio of their logarithms tends to $1 / r$ . For conjunction the midpoint bias costs a fourth root under both criteria, because $c _ { \mathrm { b } } / c _ { \mathrm { a } } = c _ { \mathrm { c } } / c _ { \mathrm { d } } = 4 ;$ one scalar decides whether the layer tolerates � features or $m ^ { 1 / 4 }$ . Error criteria are known to change capacity constants in associative memories (McEliece et $\mathrm { a l . , 1 9 8 7 ) }$ ; here they change the constant in the exponent of an exponential capacity.

The constant $c _ { \mathrm { a } } = ( 2 r - 2 t + 1 ) ^ { 2 }$ counts the inactive positions in the two binding classes, whose level $\lambda _ { 0 }$ dominates when $m \gg k$ . Conjunction binds on the classes with $r - 1$ and � active inputs and pays for one inactive position; disjunction binds on the classes with none and one active input and pays for $2 r - 1$ . Reliable �-way conjunction and feature recovery share $c _ { \mathsf { a } } = 1$ , so in this regime computing a conjunction directly costs no capacity beyond recovering its inputs. At finite $\beta$ the direct route costs the factor $( ( 2 r - 1 + \beta ) / ( 1 + \beta ) ) ^ { 2 }$ in width (corollary 6.3), which tends to one in the regime of corollary M.2.

## M.3 Width for a target success probability

A first-order threshold locates the centre of a transition, not an operating point: at its quantile centre the half-slack network of pairwise conjunction succeeds with probability about 0.15 (Table L.4). Theorem 7.1 supplies the missing quantile. With $\Psi _ { \infty } ( x ) = \exp ( - e ^ { - x } )$ , the limit of $\Psi _ { \beta }$ as $\beta  \infty .$ , write

$$
P _ { \mathrm { m i d } } ( x ) = \Psi _ { \beta _ { 0 } } ( x ) , \qquad P _ { \mathrm { h a l f } } ( x ) = \Psi _ { \beta _ { 0 } } ( x / 2 ) F _ { + } ( x / 2 ) , \qquad P _ { \mathrm { b e s t } } ( x ) = \operatorname* { s u p } _ { y \in \mathbb { R } } \Psi _ { \beta _ { 0 } } ( y ) F _ { + } ( x - y ) .
$$

Corollary M.3 (Width for a target success probability). Let $\mathcal { F }$ be the universal family of pairwise conjunctions, $k \to \infty , m / k \to \infty$ and $\beta \to \beta _ { 0 } \in [ 1 , \infty ]$ . Each of $P _ { \mathrm { m i d } } , P _ { \mathrm { h a l f } }$ and $P _ { \mathrm { b e s t } }$ is a continuous, strictly increasing distribution function. For $\delta \in ( 0 , 1 )$ let $x _ { \bullet } ( \delta )$ be the $( 1 - \delta )$ -quantile of $P _ { \bullet } ,$ and set

$$
d _ { \mathrm { m i d } } ( x ) = 4 k \bigg ( \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } + \frac { x } { \hat { \lambda } _ { 1 } } \bigg ) ^ { 2 } , \qquad d _ { \mathrm { o p t } } ( x ) = k \bigg ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } + \frac { x } { \hat { \lambda } _ { 1 } } \bigg ) ^ { 2 } .\tag{M.1}
$$

Fix $\eta > 0$ . For all large $k ,$ the midpoint network has $\mathbb { P } ( X = 0 ) > 1 - \delta a t d = d _ { \mathrm { m i d } } ( x _ { \mathrm { m i d } } ( \delta ) + \eta )$ and $\mathbb { P } ( X = 0 ) < 1 - \delta a t d = d _ { \mathrm { m i d } } ( x _ { \mathrm { m i d } } ( \delta ) - \eta )$ . The same holds for the half-slack bias of Theorem $6 . 1 ( a )$ with $d _ { \mathrm { o p t } }$ and $x _ { \mathrm { h a l f } } ,$ , and $f o r \operatorname { s u p } _ { \theta } \mathbb { P } ( X = 0 )$ over deterministic common biases with $d _ { \mathrm { o p t } }$ and $x _ { \mathrm { b e s t } } .$

Proof. $\Psi _ { \beta }$ is the distribution function of $G _ { A } + G _ { B } / \beta ,$ , with $G _ { B } / \infty = 0 ,$ , and $F _ { + }$ is that of $- \log [ \omega _ { 1 } ( \omega _ { 1 } + \omega _ { 2 } ) ]$ . Both variables have continuous positive densities on $\mathbb { R } ,$ so $\Psi _ { \beta } , F _ { + }$ and $P _ { \mathrm { h a l f } }$ are continuous and strictly increasing. For $P _ { \mathrm { b e s t } } ,$ , the product $\Psi _ { \beta _ { 0 } } ( y ) F _ { + } ( x - y )$ tends to zero as $y \to \pm \infty .$ , so the supremum is attained, and it increases strictly in � because $F _ { + }$ does; uniform continuity of $F _ { + }$ gives continuity. For every $y ,$ either $y \le x / 2$ or $x - y < x / 2 ,$ , so $P _ { \mathrm { b e s t } } ( x ) \leq \operatorname* { m a x } \{ \Psi _ { \beta _ { 0 } } ( x / 2 ) , F _ { + } ( x / 2 ) \}  0$ as $x \to - \infty ;$ and $P _ { \mathrm { b e s t } } \geq P _ { \mathrm { h a l f } }  1$ as $x \to \infty$

Next, expanding (M.1) gives $d _ { \mathrm { m i d } } ( x ) = d _ { 0 } + 8 k ( 1 + \hat { \beta } ) x + 4 k x ^ { 2 } / \hat { \lambda } _ { 1 } ^ { 2 }$ and $d _ { \mathrm { o p t } } ( x ) = d _ { 0 } ^ { \mathrm { o p t } } + 2 k ( 3 +$ $\hat { \beta } ) x + k x ^ { 2 } / \hat { \lambda } _ { 1 } ^ { 2 }$ , with $\hat { \beta } = \hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 } = \beta ( 1 + o ( 1 ) ) \ \mathrm { ( A p p e n d i x ~ F ) }$ . The window units 8 $\left( 1 + \beta \right) \geq$ 16� and $2 k ( 3 + \beta ) \geq 8 k$ dominate the quadratic terms, so $d _ { \bullet } ( x )$ is the window point of Theorem 7.1 at $x + o ( 1 )$ , uniformly on compact sets. Theorem 7.1 then gives success probabilities $P _ { \bullet } ( x ) + o ( 1 )$ with $\Psi _ { \beta }$ in place of $\Psi _ { \beta _ { 0 } } . \mathrm { ~ A s } \beta  \beta _ { 0 } , G _ { A } + G _ { B } / \beta  G _ { A } + G _ { B } / \beta _ { 0 }$ almost surely and the limit has a continuous distribution function, so $\Psi _ { \beta } \to \Psi _ { \beta _ { 0 } }$ uniformly; the same follows for $P _ { \mathrm { h a l f } }$ and $P _ { \mathrm { b e s t } } ,$ since a supremum moves by at most the uniform distance. Strict monotonicity gives $P _ { \bullet } ( x _ { \bullet } ( \delta ) + \eta ) > 1 - \delta > P _ { \bullet } ( x _ { \bullet } ( \delta ) - \eta )$ ■

Equation (M.1) is the first-order threshold with a slack $x _ { \bullet } ( \delta ) / \hat { \lambda } _ { 1 }$ added to the extreme-value levels. In relative terms the overhead of the half-slack rule is $( 1 + x _ { \mathrm { h a l f } } ( \delta ) / ( \hat { \lambda } _ { 1 } ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ) ) ^ { 2 } - 1$ of order $x _ { \mathrm { h a l f } } ( \delta ) / \log k$ when $\beta$ stays bounded: it vanishes, but only logarithmically. For 99% success at $m = k ^ { 4 }$ , the half-slack width is 2.20, 1.75, 1.54, 1.35 and 1.22 times its centre at $k = 2 ^ { 6 }$ $2 ^ { 8 } , 2 ^ { 1 0 } , 2 ^ { 1 4 }$ and $2 ^ { 2 0 }$ . Table M.3 adds two design facts at $k = 1 0 2 4$ . A bias that needs only $k ,$ � and � costs at most 0.6% in width against the best deterministic bias. The midpoint bias needs 1.30 to 1.46 times the width of the half-slack bias, depending on the target.

Table M.3. Width for a target success probability at $k = 1 0 2 4$ and $m = k ^ { 4 } .$ , from (M.1) with the profiles evaluated at $\hat { \beta } = \hat { \lambda } _ { 0 } / \hat { \lambda } _ { 1 } = 2 . 2 8$ , in units of the optimized quantile centre $k ( 3 \hat { \lambda } _ { 1 } + \hat { \lambda } _ { 0 } ) ^ { 2 }$ . The midpoint centre is 1.54 in these units. The closed-form half-slack bias stays within 0.6% of the best deterministic common bias.
<table><tr><td>Target  $\mathbb { P } ( X = 0 )$ </td><td>0.5</td><td>0.9</td><td>0.99</td><td>0.999</td></tr><tr><td>Midpoint bias</td><td>1.61</td><td>1.81</td><td>2.08</td><td>2.35</td></tr><tr><td>Half-slack bias</td><td>1.10</td><td>1.29</td><td>1.54</td><td>1.80</td></tr><tr><td>Best deterministic common bias</td><td>1.09</td><td>1.29</td><td>1.54</td><td>1.79</td></tr></table>

The limit profiles are conservative at the simulated sizes. In all 77 frozen cells above the quantile centre for pairwise conjunction (the half-slack bias in campaigns E1 and E2, the midpoint in E3), the limit lies below (44 cells) or inside (33 cells) the 95% Wilson interval of the observed success frequency, and never above the observed frequency. $\mathrm { { A t } } ~ k = 1 0 2 4$ and $m = k ^ { 4 }$ , for instance, the half-slack network at 1.5 times the centre succeeds on 99.76% of draws, with interval 99.55%, 99.87% , whereas (M.1) asks for 1.54 times the centre to reach 99%. The direction has a simple source. The window replaces the upper tail $k \overline { { \Phi } } ( \hat { \lambda } _ { 1 } + y / \hat { \lambda } _ { 1 } )$ of a Gaussian maximum by its Gumbel limit $e ^ { - y } .$ , and Mills’ ratio gives $\bar { k } \bar { \Phi } ( \hat { \lambda } _ { 1 } + y / \hat { \lambda } _ { 1 } ) \approx e ^ { - y - y ^ { 2 } / ( 2 \hat { \lambda } _ { 1 } ^ { 2 } ) }$ , a lighter tail exactly where a high success target probes. Within the tested range the rule therefore errs toward extra width.

## M.4 Auditing a layer on a realized representation

The box identity (lemma B.2) is deterministic. It therefore applies to any stored vector and any read directions, provided that each gate reads a fixed combination of per-feature reads. Verifying a universal layer then needs 2� extreme reads per pool instead of an enumeration of (<sup>�</sup>)� <sup>gates.</sup>

Proposition $M . 4$ (Exact audit of a factored layer). Let $u \in \mathbb { R } ^ { d }$ and $v _ { 1 } , \ldots , v _ { m } \in \mathbb { R } ^ { d }$ be arbitrary, let $S \subseteq [ m ]$ satisfy $r \leq | S | \leq m - r ,$ and put $R _ { i } = \langle v _ { i } , u \rangle$ . Evaluate $\mathcal { U } _ { r } ( g )$ with readouts $\begin{array} { r } { w _ { I } = \sum _ { l } a _ { l } v _ { I _ { l } } } \end{array}$ and a common bias �, where $( a , \theta ) \in \operatorname { R e p } ( g )$ , and let $B _ { z } ( R )$ be the right side of lemma B.2 with $\xi = R$ and $b = \varsigma _ { z } a$

(i) Every gate is correct if and only i ${ } ^ { \ell } B _ { z } ( R ) \leq \varsigma _ { z } \theta$ for each class with $g ( z ) = 0$ and $B _ { z } ( R ) < \varsigma _ { z } \theta$ for each class with $g ( z ) = 1$

(ii) Given $R ,$ the $2 ^ { r }$ statistics $B _ { z } ( R )$ depend only on the � largest and the � smallest reads in � and in $[ m ] \backslash S ,$ and they take $O ( r m + 2 ^ { r } r \log r )$ operations. For $r = 2$ the error count � takes <sup>�</sup>(<sup>�</sup> <sup>log</sup> <sup>�</sup>) operations.

(iii) Suppose $\begin{array} { r } { \Delta = \operatorname* { m i n } _ { z } ( \varsigma _ { z } \theta - B _ { z } ( R ) ) > 0 } \end{array}$ . Every gate stays correct for every read vector $R ^ { \prime }$ with $\| R ^ { \prime } - R \| _ { \infty } < \Delta / \| a \| _ { 1 }$ , and hence, $i f$ some $v _ { i } \neq 0 ,$ , for every stored vector $u ^ { \prime }$ with $\| u ^ { \prime } - u \| < \Delta / ( \| a \| _ { 1 }$ max<sub>�</sub> �<sub>�</sub> . Conversely, if a class $z ^ { \ast }$ attains $\Delta$ with a maximizing tuple $I ^ { * }$ and $w _ { I ^ { * } } \neq 0 ,$ , then $u ^ { \prime } = u + t \varsigma _ { z ^ { * } } w _ { I ^ { * } }$ makes gate �∗ wrong for every $t > \Delta / \| w _ { I ^ { * } } \| ^ { 2 }$

Proof. Gate � outputs ${ \bf 1 } \{ \langle a , R _ { I } \rangle > \theta \}$ , because $\langle w _ { I } , u \rangle = \langle a , R _ { I } \rangle$ . For a class with $g ( z ) = 0 ,$ , where $\varsigma _ { z } = 1$ , it is correct exactly when $\langle \varsigma _ { z } a , R _ { I } \rangle \leq \varsigma _ { z } \theta ;$ for a class with $g ( z ) = 1$ , where $\varsigma _ { z } = - 1$ exactly when $\langle \varsigma _ { z } a , R _ { I } \rangle < \varsigma _ { z } \theta$ . Taking the maximum over the class and applying lemma B.2, whose pool condition is $r \leq | S | \leq m - r ,$ gives (i). For (ii), one pass over each pool with sorted bufers of length � finds its � largest and � smallest reads, and each class then needs the sorted weights of its four groups. For $r = 2 \AA$ , sort each pool; for every read of one pool, binary search counts the partners, other than the read itself, in the same or the other pool that put a gate of the corresponding class on the wrong side of $\theta .$ . For (iii), each $\langle \varsigma _ { z } a , R _ { I } \rangle$ is a linear function of � whose coeficients have $\ell _ { 1 }$ norm $\| a \| _ { 1 }$ , because the features of � are distinct. Hence $B _ { z } ( R ^ { \prime } ) \leq B _ { z } ( R ) + \| a \| _ { 1 } \| R ^ { \prime } - R \| _ { \infty } < \varsigma _ { z } \theta$ for every class, and (i) applies; the bound for $u ^ { \prime }$ follows from $\vert R _ { i } ^ { \prime } - R _ { i } \vert \leq \| v _ { i } \| \| u ^ { \prime } - u \|$ . Finally, $\boldsymbol { u } ^ { \prime } = \boldsymbol { u } + t \varsigma _ { z ^ { * } } \boldsymbol { w } _ { I ^ { * } }$ raises $\langle \varsigma _ { z ^ { * } } a , R _ { I ^ { * } } \rangle$ by exactly $t \| w _ { I ^ { * } } \| ^ { 2 } > \Delta ,$ so it exceeds $\varsigma _ { z ^ { * } } \theta$ and gate $I ^ { * }$ is wrong whatever $g ( z ^ { * } )$ is. ■

Under the model of Section $^ { 2 , }$ with $\boldsymbol { v } _ { i } = \boldsymbol { e } _ { i } ,$ part (i) is proposition B.1 without exceptional <sup>events,</sup> <sup>and</sup> <sup>part</sup> <sup>(ii)</sup> <sup>is</sup> <sup>the</sup> <sup>reason</sup> <sup>the</sup> <sup>exact</sup> <sup>sampler</sup> <sup>costs</sup> <sup>�</sup>(<sup>�</sup> <sup>log</sup> <sup>�</sup>) <sup>per</sup> <sup>draw</sup> <sup>at</sup> <sup>any</sup> <sup>�.</sup> <sup>Part</sup> (iii) gives the whole layer a robustness margin, and above the box threshold this margin is of the order of the clean margins. Let $d \geq ( 1 + \varepsilon ) D _ { \mathsf { b o x } } ( a , \theta )$ for the universal family. Since $R _ { I } = H z + \sigma \xi _ { I }$ , we have $\varsigma _ { z } \theta - B _ { z } ( R ) = s _ { z } ^ { H } - \sigma M _ { z }$ , and on the event $\mathcal { E } _ { \delta }$ in the proof of Theorem 5.4 this is at least $s _ { z } - ( 1 + 5 \delta ) \sigma _ { d } h _ { \mathsf { B o x } _ { z } } ( a ) \ge s _ { z } ( 1 - ( 1 + 5 \delta ) ( 1 + \varepsilon ) ^ { - 1 / 2 } )$ for every class. Moreover $d \geq D _ { \mathsf { b o x } } \geq k \lambda _ { 0 } ^ { 2 } / ( 4 r )$ by (C.3), so log $m = o ( d )$ , and chi-square and Gaussian tails with a union bound give m $\mathbf { \rho } _ { \mathbb { X } _ { i } } \lVert e _ { i } \rVert \to 1$ and ma $\mathrm { \sf { i x } } _ { I } \| \mathrm { \top } w _ { I } \| / \| a \| - 1 | \to 0$ in probability. With probability tending to one, therefore, every perturbation of the stored vector with

$$
\| u ^ { \prime } - u \| \leq \big ( 1 - ( 1 + \varepsilon ) ^ { - 1 / 2 } - o ( 1 ) \big ) \frac { \operatorname* { m i n } _ { z } s _ { z } } { \| a \| _ { 1 } }
$$

leaves all $( m ) _ { r }$ outputs correct, while a perturbation of norm $\Delta / \| a \| \left( 1 + o ( 1 ) \right)$ along the probe of a binding gate breaks one. For the representations of Theorem 6.1, which have ${ \boldsymbol { a } } = \mathbf { 1 }$ , both radii are of order one, whereas $\left\| u \right\| \approx { \sqrt { k } }$

Two uses follow. The first is bias calibration from data. The oracle of campaign E2 succeeds when some common bias separates the largest negative score from the smallest positive score; by (ii), this event and a separating bias are computable from the extremes. Theorem 5.4(ii) shows that no such read-dependent common bias moves the reliability threshold, and in campaign E2 at $k = 1 0 2 4$ the oracle reaches 50% success only 2% to $7 \%$ earlier in width than the closed-form half-slack bias (Table L.2). The second is a direct test of the mechanism in learned models, which we propose as an experiment. Toy models of superposition and of learned universal-AND circuits (Elhage et al., 2022; Hänni et al., 2024; Newgas, 2025) have identifiable features. Fit per-feature read directions $v _ { i }$ and a common representation $( a , \theta )$ to the learned readouts. The fitting residual measures the distance from the factored class, and on the factored part (i) predicts every failing input from $2 r$ extremes per pool. Agreement between predicted and observed failures would show that the learned layer fails through shared extremes, as the box describes.

## M.5 Unread activity in the stream

The vector read by a gate family usually carries more than the features that the family reads: other dictionaries, earlier writes, positional content. The exact law extends to independent Gaussian content, and the extension separates the two roles that � plays in every threshold.

Proposition M.5 (Unread content). Let $u = E x + \nu g ,$ , where $g \sim N ( 0 , I _ { d } / d )$ is independent of   
�, � and $\nu \geq 0$ may depend on �. Put $K = k + \nu ^ { 2 }$ and $c = 1 - \nu / \sqrt { K } \in ( 0 , 1 ]$ With �, �, � as   
in Theorem 3.1 and $\sigma = \sqrt { K H / d } ,$   
(�� , ��) <sup>�</sup>=  � + �(�� − ��¯ ), ���    
jointly in $i \leq k$ and $j \leq n ,$ , and a canonical gate of class � has the score law (9) with � replaced by   
�. Consequently Theorems 4.1, 4.2, 5.1, 5.4, 6.1 and 7.1 hold with $\sigma _ { d } = \sqrt { K / d } ;$ the prefactor �   
of every threshold, centre and window width becomes $K ,$ the condition $d / k $ in Theorem 5.1   
becomes � � , and $\lambda _ { 1 } , \lambda _ { 0 } , \hat { \lambda } _ { 1 } , \hat { \lambda } _ { 0 } , N _ { z }$ , the programs (11) and (13) and the window profiles   
are unchanged.

Proof. Fix $S = [ k ]$ . Each coordinate of � is $N ( 0 , K / d )$ , and $\mathrm { C o v } ( e _ { i } , u ) = I _ { d } / d$ for $i \leq k$ . Gaussian conditioning gives $e _ { i } = u / K + \eta _ { i } ,$ , where $( \eta _ { i } ) _ { i \leq k }$ is independent of � with $\mathrm { C o v } ( \eta _ { i } , \eta _ { l } ) = d ^ { - 1 } ( { \bf 1 } \{ i =$ $l \} { - 1 } / K ) I _ { d }$ . Hence $A _ { i } = \| u \| ^ { 2 } / K + \langle \eta _ { i } , u \rangle$ , and given � the vector $( \langle \eta _ { i } , u \rangle ) _ { i \leq k }$ is centered Gaussian with covariance $( \| u \| ^ { 2 } / d ) ( I _ { k } - \mathbf { 1 1 } ^ { \top } / K )$ . Since $( 1 - c ) ^ { 2 } = \nu ^ { 2 } / K = 1 - k / K ,$ the vector $( Z _ { i } - c \bar { Z } ) _ { i \leq k }$ has covariance $I _ { k } - ( 2 c - c ^ { 2 } ) k ^ { - 1 } \mathbf { 1 1 } ^ { \top } = I _ { k } - \mathbf { 1 1 } ^ { \top } / K$ . Put $H = \| u \| ^ { 2 } / K \sim \chi _ { d } ^ { 2 } / d ,$ so that $\| u \| / \sqrt { d } = \sigma$ The inactive columns are independent of $( u , \eta )$ , so given � the reads $B _ { j }$ are independent $N ( 0 , \sigma ^ { 2 } )$ variables, independent of the active reads. For one gate, the coordinates of $( w , u )$ are independent pairs with variances $\| a \| ^ { 2 } / d$ and $K / d$ and covariance $\langle a , z \rangle / d$ , and the proof in Appendix B applies with � in place of �.

The proofs in Appendices B to F use � in two roles. As the size of the active pool, it fixes $\lambda _ { 1 } ,$ $\hat { \lambda } _ { 1 } , N _ { z }$ and the law of the active order statistics. As a variance, it enters through $\sigma _ { d } ^ { 2 } = k / d ,$ the one-gate variance $k \| a \| ^ { 2 } - \langle a , z \rangle ^ { 2 }$ and the conditional variance $( k - | z | ) / d$ of a local probe in the converse of Theorem 4.2. The representation above leaves the first role unchanged and replaces � by � in the second; the local-probe variance becomes $( K - | z | ) / d$ because the unread content is independent of every column. It also replaces $\bar { Z }$ by $c { \bar { Z } }$ , which only shrinks the centering terms, and every estimate that used $k $ in the second role holds a fortiori with $K \geq k$ Auxiliary rates indexed by $k ,$ such as $k ^ { - 1 / 4 }$ and 1 log � in $\mathrm { A p p e n d i x } { \cal C } ,$ can be kept, since every bound they control only improves when $K \geq k$ . These substitutions give the stated forms. ■

Width is paid for activity; the logarithms are paid for the read family. Independent content of squared norm $\nu ^ { 2 }$ adds $\nu ^ { 2 }$ units to the interference variance but no extremes: the levels, the ratio between ball and box, the binding classes and the window profiles are those of the read family alone. If the stream also carries $\nu ^ { 2 }$ active directions of another independent Gaussian dictionary, whose sum is exactly $N ( 0 , \nu ^ { 2 } I _ { d } / d )$ , every threshold in Table 1 is multiplied by $1 + \nu ^ { 2 } / k$ . When the read family accounts for a quarter of the active directions, the widths quadruple. The half-slack bias must be computed with $\sigma _ { d } = \sqrt { K / d }$ as well; a bias calibrated to the read activity alone sits too close to the midpoint. A fixed number of families on independent dictionaries can share one stream: each sees the others as unread content, and a union bound over families composes their reliability. Proposition M.5 is the sharp one-layer counterpart of the stream activity � in Theorem H.5.

## M.6 Evaluating shared errors

The gap between the ball and the box is a gap between a mean and a typical value. When $\mathbb { P } ( X > 0 ) > 0 ,$

$$
{ \mathbb E } X = { \mathbb P } ( X > 0 ) { \mathbb E } [ X \mid X > 0 ] .\tag{M.2}
$$

For the universal family between the two thresholds, $( 1 + \varepsilon ) D _ { \mathsf { b o x } } \leq d \leq ( 1 - \varepsilon ) D _ { \mathsf { b a l l } } .$ , the failure probability tends to zero and the mean to infinity, so the burden $\mathbb { E } [ X \mid X > 0 ]$ of a failed draw <sup>grows</sup> <sup>faster</sup> <sup>than</sup> <sup>the</sup> <sup>mean.</sup> <sup>For</sup> <sup>�</sup> <sup>independent</sup> <sup>evaluation</sup> <sup>draws,</sup> <sup>P</sup>(<sup>no</sup> <sup>draw</sup> ${ \mathrm { f a i l s } } ) = \mathbb { P } ( X =$ $0 ) ^ { B } \geq 1 - B \mathbb { P } ( X > 0 )$ . Whenever $B \mathbb { P } ( X > 0 )  0 .$ , the sample mean of � and its standard error are both zero with probability tending to one, while $\mathbb { E } X  \infty$ . The frozen campaigns show the efect at both ends. $\mathrm { A t } m = k ^ { \dot { 6 } 4 }$ and $d = 1 . 5 d _ { 0 }$ , all 4096 draws are correct while quadrature gives $\mathbb { E } X \approx 2 . 8 \times 1 0 ^ { 1 9 }$ (Table L.3). At $m = k ^ { 4 }$ and 1.35 times the centre, even the Rao–Blackwellized sample mean is 0.063 with standard error 0.011, against the quadrature value 0.327 (Table L.1).

Four quantities make an evaluation of a superposed layer informative. The first is wholeoutput success with a binomial interval. The second is the expected count from an exact one-gate law (proposition 3.2) rather than from a sample mean. The third is the burden $\mathbb { E } [ X \mid X > 0 ]$ together with the class of each wrong output, which separates frequent small failures from rare large ones. The fourth is the slack $\Delta$ of proposition M.4 on every draw. The slack is continuous, so its empirical distribution is informative even when no draw fails, and in the window its law is explicit. For the midpoint network at $d = d _ { 0 } + 8 k ( 1 + \beta ) x$ with $\beta \to \beta _ { 0 } \in [ 1 , \infty ]$ , the computation in Appendix F gives

$$
\frac { \hat { \lambda } _ { 1 } \Delta } { \sigma _ { d } } \Rightarrow x - G _ { A } - G _ { B } / \beta _ { 0 } ,
$$

because the slack of the one-active class is $\begin{array} { r } { \frac { 3 } { 2 } - \operatorname* { m a x } _ { i } A _ { i } - \operatorname* { m a x } _ { j } B _ { j } , } \end{array}$ , whose expansion in Appendix F gives the limit, while the other classes keep a diverging slack in these units. A Gumbel fit to observed slacks therefore extrapolates the failure probability below the level that the draws resolve.

## M.7 Implications for claims about superposition

Models of computation in superposition (Elhage et al., 2022; Hänni et al., 2024; Adler and Shavit, 2024) are used to interpret what a layer of a network computes. Shi et al. (2026) survey the geometry, learning and computation of superposed representations and examine what evaluations of feature-recovery methods establish, noting that accurate reconstruction of activations does not by itself establish feature identity or causal use. The results here give claims about computation in superposition a checkable form in the same spirit.

Name the criterion Losses and error rates averaged over gates measure the first row of Table M.1, which does not depend on �. A layer can have an expected error fraction below $1 0 ^ { - 2 1 }$ and fail on most inputs (Appendix M.1). Learned solutions of the universal-AND task (Newgas, 2025) and of related compressed-computation tasks (Bhagat et al., 2026) need not use the mechanism of theoretical constructions. A claim that a learned layer computes in superposition should therefore state which row of the table it establishes.

Reliability is not robustness The reliability laws average over supports drawn independently of the dictionary. Two adversaries break them at very diferent costs. A support chosen after the dictionary raises the width to order $k ^ { 2 }$ log � (proposition 8.1). A perturbation of the stored vector along the probe of a binding gate needs a norm of order one, out of $\left\| u \right\| \approx { \sqrt { k } }$ (proposition M.4(iii)). A guarantee certified on random inputs is a statement about typical use.

Lower bounds concern accessibility The Θ � log � scale belongs to threshold reads of a random linear code. Without the linear dictionary, 2� coordinates carry all pairwise conjunctions with one threshold unit each (proposition H.7), and order � coordinates are necessary (proposition H.6). A width far below the number of features is therefore consistent with several mechanisms, and a dimension count alone does not identify computation in superposition.

## M.8 Open problems

Reliability for gate-local rules Theorem 5.4(ii) covers every common representation, even one chosen after the reads are observed, and Theorem 4.2 covers every gate-local rule for the expected count. Can a gate-local rule, for instance the dual frame of Remark C.1, make the network reliable below $k V _ { \mathsf { b o x } } ?$ A converse must control the joint extremes of adapted probes. Marginal error bounds cannot force a failure on a typical draw, since $\mathbb { P } ( X > 0 )$ can vanish while E� diverges.

Sharp suprema for structured families Disjoint tuples give the ball, the universal family gives the box, and proposition 5.3 interpolates for one level of private partners. For a family defined by a sharing hypergraph with several levels, generic chaining determines $\Gamma _ { z }$ up to constants (Talagrand, 2014). A sharp variational formula, possibly of the type of the generalized random energy model (Derrida, 1985), would give the reliability price of any query family, for example relation queries in which each feature has a bounded number of partners at each level.

Universality of the box The one-gate exponent is universal (Theorem 8.2), but the box rests on the Gaussian joint law. A first target is the Rademacher dictionary with midpoint pairwise conjunction: is its reliability threshold $4 k ( \lambda _ { 1 } + \lambda _ { 0 } ) ^ { 2 } ( 1 + o ( 1 ) ) ?$ The required input is joint extreme-value control of the active and inactive reads, which are dependent through �.

Reliability of composed circuits Theorem H.5 is a union bound over an ideal execution, that is, a ball argument. In a two-layer universal circuit the second layer reads $\binom { k } { 2 }$ active outputs, whose errors are correlated through the extremes of the first layer. A box law for the second layer needs the joint law of its reads given the first layer’s realized errors; proposition M.5 indicates, through the variance term, that it pays for its full stream activity.

Designed and learned linear dictionaries The Gaussian dictionary needs order � log � dimensions, and arbitrary encodings need order � (propositions H.6 and H.7) at the price of large coeficients and small normalized margins. For uniform recovery, every linear encoder needs $\Omega ( k ^ { 2 } \log ( m / k ) / \log k )$ dimensions when $k < { \sqrt { m } }$ (Garg et al., 2026). For random supports, is order � log � necessary for every linear dictionary read by threshold units, or can a dictionary adapted to the support law, with bounded coeficients and margins, approach order �?

Non-asymptotic sizing The limit profiles of Theorem 7.1 were conservative in every simulated cell above the centre, and the Gaussian tail factor $e ^ { - y ^ { 2 } / ( 2 \hat { \lambda } _ { 1 } ^ { 2 } ) }$ explains the direction (Appendix M.3). A second-order window law with explicit error terms would turn (M.1) into a certified width at finite �. For general threshold gates, even the first-order window requires the joint law of � upper and � lower order statistics in both pools.