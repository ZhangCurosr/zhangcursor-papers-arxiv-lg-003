# The finite-horizon five-expert prediction problem<sup>∗</sup>

Jef Calder<sup>1</sup> and Nadejda Drenska<sup>2</sup>

<sup>1</sup>School of Mathematics, University of Minnesota <sup>2</sup>Department of Mathematics, Louisiana State University

September 30, 2026

## Abstract

We give an explicit solution to the five expert prediction with expert advice partial diferential equation (PDE) in the finite-time horizon setting. The solution formula establishes that the adversary’s rank strategy (1, 0, 1, 0, 0) is globally optimal, and the COMB strategy (1, 0, 1, 0, 1) is optimal exactly on the set where $x _ { 1 } = x _ { 2 }$ and $x _ { 3 } = x _ { 4 }$ The formula is derived from the solution of the geometric-stopping problem given in our companion paper through the transform principle of Bayraktar, Ekren and Zhang, which links the two problems by a Laplace transform. Inverting the transform term by term expresses the solution through a series of Gaussian and complementary error function kernels. The optimality of (1, 0, 1, 0, 0) is reduced to the signs of 41 one-variable Gaussian series, which are certified with computer assistance by Poisson summation, first-mode domination and interval arithmetic on 1616 rational cells. The proofs of our main theorems, certificates included, are also formalized in the Lean proof assistant.

## Contents

1 Introduction

2 Main results

3 Derivation of the finite-horizon formula 9   
3.1 The transform principle 9   
3.2 The inversion kernels . 10   
3.3 Four experts . 13   
3.4 Region III 14   
3.5 Regions I and II . . 17   
4 Verification 20   
4.1 Convergence and the transform identity 21   
4.2 Regularity at the collision set . 24   
4.3 Transfer of curvature identities and the COMB equality set 29   
4.4 The Hamiltonian inequalities 31   
4.5 Proofs of the main theorems 34   
A The Region III certificate 35   
A.1 Normalization and positive heat-exit interpolation 35   
A.2 The corner gaps as Gaussian series 36   
A.3 Small scales: a dual representation 37   
A.4 The compact range and the large-scale tail . 39   
A.5 Conclusion in Region III 40   
B The Regions I and II certificate 41   
B.1 Reduction to four corner families 41   
B.2 Positivity-preserving evolutions and the pure traces 42   
B.3 The derivative cascades and the scalar certificates 45   
B.4 All sixty-four corner controls . 47   
C Independent numerical checks 48

## 1 Introduction

In finite-horizon prediction with expert advice, a player and an adversary play a known number of rounds M. In each round the player picks, possibly at random, one of n experts to follow, the adversary decides which experts are correct, and at the end of the game the player’s performance is measured by their regret—the excess of the player’s cumulative loss over that of the best expert. The game dates back to Cover [11]; for an extensive literature review on regret bounds we refer to [9], and for the history of the exact minimax problem we refer to the introduction of our companion paper [6]. Drenska and Kohn [12,13] showed that if $V ^ { M } ( x )$ is the value of the M-round game with initial regrets x, then $\dot { V } ^ { M } ( \dot { \sqrt { M } } x ) / \sqrt { M } \to U ( 1 , \dot { x } )$ locally uniformly as $M \to \infty$ , where U is the viscosity solution of the nonlinear parabolic partial diferential equation (PDE)

$$
U _ { \tau } = \frac { 1 } { ^ 2 } \operatorname* { m a x } _ { \mathbf { v } \in \{ 0 , 1 \} ^ { n } } \mathbf { v } ^ { T } \nabla ^ { 2 } U \mathbf { v } , \qquad U ( 0 , x ) = \operatorname* { m a x } _ { 1 \leq i \leq n } x _ { i } ,\tag{1.1}
$$

on $( 0 , \infty ) \times \mathbb { R } ^ { n }$ ; in particular $V ^ { M } ( 0 ) = U ( 1 , 0 ) \sqrt { M } + o ( \sqrt { M } )$ . Here $\tau$ is the time remaining and $x _ { i }$ the player’s regret with respect to expert i. The gradient ∇U gives the player’s asymptotically optimal mixed strategy, and the maximizing controls v give the adversary’s.

Explicit solutions of (1.1) allow for the exact characterization of optimal strategies, but only the cases of $n \leq 4$ have so far been addressed. For $n \leq 3$ the solution is elementary, and Abbasi-Yadkori, Bartlett and Gabillon [1] constructed near-minimax strategies for the three-expert game. For $n = 4$ , Bayraktar, Ekren and Zhang [3] derived the solution of the $n = 4$ finite-horizon game from their solution [4] of the geometric-horizon version of the problem via the inversion of a Laplace transform. In both settings COMB, which ranks the experts by regret and makes either the odd-ranked or the even-ranked ones correct, with probability $\textstyle { \frac { 1 } { 2 } }$ each, is optimal for four experts (as is the strategy that makes the two middle-ranked experts correct) and Gravin, Peres and Sivan [14] conjectured that COMB stays asymptotically optimal for every n. Numerical evidence [8, 10] pointed the other way for $n \geq 5$ , and in [6] the authors gave an explicit solution to the five-expert geometric-horizon problem: the non-COMB rank direction $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ attains the maximum everywhere in the ordered sector $x _ { 1 } \geq \dots \geq x _ { 5 }$ , while COMB is optimal only where $x _ { 1 } = x _ { 2 }$ and $x _ { 3 } = x _ { 4 }$ The same formula for geometric stopping was derived independently by Bayraktar, Ekren and Kolliopoulos [2] in a preprint that appears shortly after [6].

In this paper we solve the five-expert finite time horizon problem. Theorem 2.1 gives an explicit function U that is a classical solution of (1.1) with $n = 5$ for which $\mathbf { v } _ { * }$ attains the maximum at every point. In particular $V ^ { M } ( 0 ) / \sqrt { M }  U ( 1 , 0 ) = 4 5 \pi ^ { 3 / 2 } / ( 2 5 6 \sqrt { 2 } )$ as $M \to \infty$ , which is $2 / { \sqrt { \pi } }$ times the geometric-horizon constant $4 5 \pi ^ { 2 } / ( 5 1 2 { \sqrt { 2 } } )$ of [6]. This is the ratio conjectured by Gravin, Peres and Sivan [14, §1.2], and Bayraktar, Ekren and Zhang [3] observed that it holds whenever one strategy is optimal for both problems. Theorem 2.2 shows that COMB attains the maximum exactly on the set $\{ x _ { 1 } = x _ { 2 } , \ x _ { 3 } = x _ { 4 } \}$ found in the geometric problem, so COMB is not globally optimal for five experts at any horizon.

The construction uses the transform principle of Bayraktar, Ekren and Zhang [3], stated here as Proposition 3.1: if one control v attains the maximum everywhere for the geometrichorizon solution $u ,$ then $\lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ solves the equation obtained by Laplace transforming the finite-horizon problem in $\tau$ with the adversary frozen at that control, and its inverse transform is a natural candidate for U. For four experts they inverted this transform along the branch cut of $\sqrt { \lambda }$ , which produces an oscillatory integral. We invert it term by term instead. We expand the formulas of [6] in exponentials of linear functions of x with polynomial prefactors, and each such term inverts to an explicit kernel built from a Gaussian and the complementary error function. The result is a series with rational coeficients; for four experts the same procedure recovers the solution of [3] as such a series (Proposition 3.9).

Most of the paper is devoted to proving that U solves (1.1). We obtain its regularity where experts tie, including the collision set on which the series degenerate, by continuing the stationary formulas to complex scaling parameters and inverting the transform on a Hankel contour. By the transform principle, the inequalities expressing that $\mathbf { v } _ { * }$ attains the maximum are equivalent to the complete monotonicity in λ of finitely many curvature gaps of the stationary solution, which we reduce to the signs of 41 explicit one-variable Gaussian series. These signs are proved with the assistance of a computer, on the whole positive axis rather than at sample points: by Poisson summation for small arguments, by first-mode domination for large ones, and in between by outward-rounded interval arithmetic over 1616 rationa cells, with scripts and data in the supplement [5]. The proofs of Theorems 2.1 and 2.2, these certificates included, have also been formalized in the Lean proof assistant with the Mathlib library [5], and Table 1 lists the Lean theorem behind each claim; the $\sqrt { M }$ asymptotics are the cited theorem of Drenska and Kohn and are not part of the formalization.

## 2 Main results

We consider the finite-horizon prediction with expert advice PDE (1.1) with five experts, which in the time to maturity $\tau = T - t$ reads

$$
\begin{array} { r } { U _ { \tau } = \frac { 1 } { 2 } \ \underset { { \mathbf v } \in \{ 0 , 1 \} ^ { 5 } } { \operatorname* { m a x } } { \mathbf v } ^ { T } \nabla ^ { 2 } U { \mathbf v } , \qquad U ( 0 , x ) = \varphi ( x ) : = \underset { 1 \leq i \leq 5 } { \operatorname* { m a x } } x _ { i } , } \end{array}\tag{2.1}
$$

on $( 0 , \infty ) \times \mathbb { R } ^ { 5 }$ . The stationary 5-expert problem solved in the companion paper [6] is

$$
u ( x ) - \frac { 1 } { 2 } \operatorname* { m a x } _ { \mathbf { v } \in \{ 0 , 1 \} ^ { 5 } } \mathbf { v } ^ { T } \nabla ^ { 2 } u ( x ) \mathbf { v } = \varphi ( x ) , \qquad x \in \mathbb { R } ^ { 5 } ,\tag{2.2}
$$

whose solution is the value of the geometric-horizon game. Throughout we write $D _ { \mathbf { v } } ^ { 2 } w =$ $\mathbf { v } ^ { T } \nabla ^ { 2 } w \mathbf { v }$ for the second derivative of a function w in the direction v, and we work in the ordered sector $\{ x _ { 1 } \geq x _ { 2 } \geq x _ { 3 } \geq x _ { 4 } \geq x _ { 5 } \}$ , with $k = \sqrt { 2 }$

We use the scaled gaps $y _ { i } = k ( x _ { i } - x _ { i + 1 } ) \geq 0$ and the coordinates of the companion paper,

$$
\begin{array} { l } { z _ { 1 } = \cfrac { x _ { 1 } + x _ { 2 } - x _ { 3 } - x _ { 4 } } { k } , ~ z _ { 2 } = \cfrac { x _ { 1 } - x _ { 2 } + x _ { 3 } - x _ { 4 } } { k } , } \\ { z _ { 3 } = \cfrac { x _ { 1 } - x _ { 2 } - x _ { 3 } + x _ { 4 } } { k } , ~ z _ { 4 } = y _ { 4 } - \operatorname* { m a x } ( z _ { 3 } , 0 ) , } \\ { a _ { 1 } = \cfrac { y _ { 1 } - y _ { 3 } - 2 y _ { 4 } } { 3 } , ~ a _ { 2 } = a _ { 1 } + y _ { 4 } , ~ a _ { 3 } = a _ { 2 } + y _ { 3 } , ~ a _ { 4 } = a _ { 3 } + y _ { 2 } . } \end{array}\tag{2.3}
$$

Note that the ordered sector in x transforms to the set $\{ z _ { 1 } \geq z _ { 2 } \geq | z _ { 3 } | \}$ , and the fifth expert enters only through $z _ { 4 }$ . As in the companion paper, the sector splits into three regions:

<table><tr><td>Region Scaled gaps</td><td></td><td>Coordinates</td></tr><tr><td>I</td><td> $y _ { 1 } \le y _ { 3 }$ </td><td> $z _ { 3 } \le 0 \le z _ { 4 }$ </td></tr><tr><td>II</td><td> $y _ { 3 } \leq y _ { 1 } \leq y _ { 3 } + 2 y _ { 4 }$ </td><td> $z _ { 3 } \ge 0 , z _ { 4 } \ge 0$ </td></tr><tr><td>III</td><td> $y _ { 1 } \geq y _ { 3 } + 2 y _ { 4 }$ </td><td> $z _ { 3 } \ge 0 , z _ { 4 } \le 0$ </td></tr></table>

In Region III one has $0 \leq a _ { 1 } \leq a _ { 2 } \leq a _ { 3 } \leq a _ { 4 }$ and $y _ { 1 } = a _ { 1 } + a _ { 2 } + a _ { 3 }$ . Where $z _ { 3 } \geq 0$ one has $a _ { 1 } = - { \frac { 2 } { 3 } } z _ { 4 }$ , so Region III is $\{ a _ { 1 } \geq 0 \}$ , its interface $z _ { 4 } = 0$ with Region II is $\{ a _ { 1 } = 0 \}$ , and on that interface $( a _ { 2 } , a _ { 3 } , a _ { 4 } ) = ( z _ { 3 } , z _ { 2 } , z _ { 1 } )$

A rank strategy acts on the coordinates in decreasing order, so a function defined on the ordered sector is extended to $\mathbb { R } ^ { 5 }$ by sorting the coordinates. The stationary solution u and the function $U$ constructed below are both $x _ { 1 }$ plus a function of the diferences of the coordinates. Their Hessians therefore annihilate 1, so a control v and its complement $\mathbb { 1 } - \mathbf { v }$ give the same second derivatives, and we identify complementary controls throughout. We write $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ for the rank direction that attains the Hamiltonian maximum of (2.2) throughout the sector [6], and we write C for the set of states at which the four largest coordinates coincide; in the ordered sector $\mathcal { C } = \{ x _ { 1 } = x _ { 2 } = x _ { 3 } = x _ { 4 } \} = \{ z _ { 1 } = 0 \}$ , which contains the diagonal $\{ a _ { 4 } = 0 \}$ . The mode series below degenerate exactly on ${ \mathcal { C } } ,$ and the regularity of $U$ there is proved separately in Section 4.2.

The formula for $U$ is written in terms of kernels $\mathcal { H } _ { j } : \mathbb { R } \times ( 0 , \infty ) \to \mathbb { R } , j \geq 0$ , defined as follows. With

$$
\mathcal { G } ( X , \tau ) = \frac { 1 } { \sqrt { \pi \tau } } e ^ { - X ^ { 2 } / ( 4 \tau ) } , \qquad \mathcal { E } ( X , \tau ) = \mathrm { e r f c } \big ( X / ( 2 \sqrt { \tau } ) \big ) ,\tag{2.4}
$$

the first three kernels are

$$
\mathcal { H } _ { 0 } = 2 \tau \mathcal { G } - X \mathcal { E } , \qquad \mathcal { H } _ { 1 } = \mathcal { E } , \qquad \mathcal { H } _ { 2 } = \mathcal { G } ,\tag{2.5}
$$

and the others are defined by the recursion

$$
\mathcal { H } _ { j + 1 } = \frac { X } { 2 \tau } \mathcal { H } _ { j } - \frac { j - 2 } { 2 \tau } \mathcal { H } _ { j - 1 } , \qquad j \geq 2 .\tag{2.6}
$$

Every $\mathcal { H } _ { j }$ is an elementary combination of a Gaussian and the complementary error function. We also note that the kernels can be expressed as inverse Laplace transforms: for every $\tau > 0$ ，

$$
\mathcal H _ { j } ( X , \tau ) = \mathcal L ^ { - 1 } \Bigl [ \lambda ^ { ( j - 3 ) / 2 } e ^ { - X \sqrt { \lambda } } \Bigr ] ( \tau ) \qquad \mathrm { w h e n ~ } X > 0 , \mathrm { ~ o r ~ } X = 0 \mathrm { ~ a n d ~ } j \le 2 ,\tag{2.7}
$$

where $\mathcal { L }$ denotes the Laplace transform in $\tau$ and λ the Laplace variable. At $X = 0$ the identity fails for $j \geq 3$ . For $j = 3$ its right side $ { \mathcal { L } } ^ { - 1 } [ 1 ]$ is not a function while $\mathcal { H } _ { 3 } ( 0 , \tau ) = 0$ For $j = 4$ the kernel $\mathcal { H } _ { 4 } ( 0 , \tau ) = - 1 / ( 2 \sqrt { \pi } \tau ^ { 3 / 2 } )$ is not integrable at $\tau = 0$ . The formula below uses kernels of order $j \geq 3$ only where their argument X is positive. The transform identity, together with the diferentiation rules and the bounds on the $\mathcal { H } _ { j }$ that we use, is proved in Lemma 3.3.

The formula for $U$ is derived from the stationary solution $u = x _ { 1 } + F / k$ of (2.2). In each region we expand $F$ in a series of terms $P ( x ) e ^ { - X ( x ) }$ , which we call modes, with X linear and $P$ a homogeneous polynomial, and replace each mode by $P ( x ) \mathcal { H } _ { j } ( X ( x ) , \tau )$ , where $j$ is the degree of $P$ (Corollary 3.6). We now give the coeficients and exponents of these expansions.

In Region III the formula is built from the trace $e$ of the companion paper, defined by the quadrature

$$
e ( X ) = 3 \cosh X \int _ { X } ^ { \infty } I ( t ) { \mathrm { ~ s e c h } } ^ { 2 } t { \mathrm { ~ c s c h } } ^ { 5 } t d t , \qquad I ( X ) = \int _ { 0 } ^ { X } { \sinh } ^ { 4 } s { \cosh } ^ { 2 } s d s .\tag{2.8}
$$

The integrand of $e$ is regular at $t = 0$ since $\begin{array} { r } { I ( t ) = \frac { 1 } { 5 } t ^ { 5 } + \mathcal { O } ( t ^ { 7 } ) } \end{array}$ , and decays like ${ \textstyle \frac { 1 } { 3 } } e ^ { - t }$ at infinity; hence e is positive and bounded. We also need the function $\Lambda ( L ) = \log \coth ( L / 2 )$ .<sup>1</sup> Set

(2.9)

$$
A _ { 0 } = \textstyle { \frac { 1 } { 2 } } , A _ { m } = - \frac { 1 2 m ^ { 6 } - 2 5 m ^ { 4 } - m ^ { 2 } - 4 } { 4 ( 4 m ^ { 2 } - 1 ) ^ { 2 } } ( m \geq 1 ) , B _ { m } = \frac { m ( m ^ { 2 } - 1 ) ( m ^ { 2 } - 4 ) } { 2 ( 4 m ^ { 2 } - 1 ) } ,\tag{2.10}
$$

$$
J _ { 0 } = K _ { 0 } = 1 , \qquad J _ { m } = \frac { - 2 } { 4 m ^ { 2 } - 1 } , \qquad K _ { m } = \frac { 4 m } { 4 m ^ { 2 } - 1 } \quad ( m \geq 1 ) ,
$$

and, with $\delta _ { m 0 }$ the Kronecker delta,

$$
\begin{array} { l l l } { { \alpha _ { m } ^ { ( 0 ) } = A _ { m } , } } & { { \beta _ { m } ^ { ( 0 ) } = B _ { m } , } } & { { } } \\ { { \alpha _ { m } ^ { ( 1 ) } = \frac 1 6 \big ( B _ { m } - 2 m A _ { m } - 3 \delta _ { m 0 } \big ) , } } & { { \beta _ { m } ^ { ( 1 ) } = - \frac 1 { 3 } m B _ { m } , } } & { { } } \\ { { \alpha _ { m } ^ { ( 2 ) } = \frac 1 { 2 1 } \big ( ( 2 m ^ { 2 } + 3 ) A _ { m } - 2 m B _ { m } + 9 J _ { m } \big ) , } } & { { \beta _ { m } ^ { ( 2 ) } = \frac 1 { 2 1 } ( 2 m ^ { 2 } + 3 ) B _ { m } , } } & { { } } \\ { { \alpha _ { m } ^ { ( 3 ) } = \frac 1 { 8 4 } \big ( - 2 m ( m ^ { 2 } + 5 ) A _ { m } + ( 3 m ^ { 2 } + 5 ) B _ { m } \big ) } } & { { } } & { { } } \\ { { \phantom { \alpha _ { m } ^ { ( 1 ) } = \frac 1 { 4 } \big ( 6 K _ { m } - 1 3 \delta _ { m 0 } \big ) , } } } & { { \beta _ { m } ^ { ( 3 ) } = - \frac 1 { 4 2 } m ( m ^ { 2 } + 5 ) B _ { m } . } } & { { } } \end{array}\tag{2.11}
$$

Proposition 3.10 shows that $A _ { m }$ and $B _ { m }$ are the coeficients of the exponential expansion

$$
e ( L ) = \sum _ { m \geq 0 } ( A _ { m } + B _ { m } L ) e ^ { - 2 m L }
$$

of the trace, $J _ { m }$ and $K _ { m }$ are those of $\Lambda ( L )$ sinh L and $\Lambda ( L )$ cosh L, and $\alpha _ { m } ^ { ( j ) } , \beta _ { m } ^ { ( j ) }$ are those of the four coeficient functions of the Region III stationary formula from [6]. For $\epsilon \in \{ \pm 1 \} ^ { 3 }$ let $s _ { j } ( \epsilon )$ denote the j-th elementary symmetric polynomial of $\epsilon _ { 1 } , \epsilon _ { 2 } , { \epsilon _ { 3 } } , ^ { 2 }$ and define

$$
c _ { m , \epsilon } ^ { a } = \sum _ { j = 0 } ^ { 3 } \alpha _ { m } ^ { ( j ) } s _ { j } ( \epsilon ) , c _ { m , \epsilon } ^ { b } = \sum _ { j = 0 } ^ { 3 } \beta _ { m } ^ { ( j ) } s _ { j } ( \epsilon ) , X _ { m , \epsilon } = 2 m a _ { 4 } - \epsilon _ { 1 } a _ { 1 } - \epsilon _ { 2 } a _ { 2 } - \epsilon _ { 3 } a _ { 3 } .\tag{2.12}
$$

They are the coeficients and exponents of the mode expansion of the stationary solution in Region III, derived in the proof of Proposition 3.13: for $a _ { 4 } > 0$

$$
{ \cal F } = \frac { 1 } { 8 } \sum _ { m \geq 0 } \sum _ { \epsilon \in \{ \pm 1 \} ^ { 3 } } \bigl ( c _ { m , \epsilon } ^ { a } + c _ { m , \epsilon } ^ { b } a _ { 4 } \bigr ) e ^ { - X _ { m , \epsilon } } .
$$

In Regions I and II the formula has a four-expert background and a correction. Write $\theta ( L ) = \arctan ( e ^ { - L } )$ . The stationary four-expert solution of Bayraktar, Ekren and Zhang [4, Theorem 3.1] is, in the present notation, $\begin{array} { r } { u _ { 4 } = x _ { 1 } + \frac { 1 } { k } F _ { 4 } } \end{array}$ with

(2.13) $F _ { 4 } = \theta ( z _ { 1 } )$ cosh z<sub>1</sub> cosh z<sub>2</sub> cosh $z _ { 3 } + { \textstyle \frac { 1 } { 2 } } \Lambda ( z _ { 1 } )$ sinh z<sub>1</sub> sinh z<sub>2</sub> sinh $z _ { 3 } - { \textstyle \frac { 1 } { 2 } } \sinh ( z _ { 2 } + z _ { 3 } )$ 2

and Lemma 3.8 gives its mode expansion

$$
F _ { 4 } = \sum _ { m \geq 0 } \sum _ { \stackrel { \epsilon \in \{ \pm 1 \} ^ { 3 } } { \epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } = ( - 1 ) ^ { m } } } c _ { m , \epsilon } ^ { ( 4 ) } e ^ { - X _ { m , \epsilon } ^ { ( 4 ) } } , \qquad z _ { 1 } > 0 ,
$$

where

$$
\begin{array} { c } { { c _ { m , \epsilon } ^ { ( 4 ) } = \displaystyle \frac { ( - 1 ) ^ { m } } { 4 ( 2 m + 1 ) } \quad \mathrm { f o r } \epsilon \in \{ \pm 1 \} ^ { 3 } \mathrm { w i t h } \epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } = ( - 1 ) ^ { m } , } } \\ { { X _ { m , \epsilon } ^ { ( 4 ) } = ( 2 m + 1 - \epsilon _ { 1 } ) z _ { 1 } - \epsilon _ { 2 } z _ { 2 } - \epsilon _ { 3 } z _ { 3 } , } } \end{array}\tag{2.14}
$$

with the two exceptions $c _ { 0 , ( 1 , 1 , 1 ) } ^ { ( 4 ) } = 0 \mathrm { ~ a n d ~ } c _ { 0 , ( 1 , - 1 , - 1 ) } ^ { ( 4 ) } = \textstyle { \frac { 1 } { 2 } }$ . The fifth expert enters through the density

$$
p ( X ) = \textstyle { \frac { 1 } { 2 } } \operatorname { t a n h } X - 3 \coth X + \frac { 1 5 I ( X ) } { \sinh ^ { 6 } X } ,\tag{2.15}
$$

which is positive. For integers $j \geq 0$ define the rationals $r _ { j , n }$ and $s _ { j , n } , n \geq 0$ , as the coeficients of the power series, convergent for $| w | < 1$ ，

$$
\sum _ { n \geq 0 } r _ { j , n } w ^ { n } = \frac { w ^ { j + 2 } \left( 1 - 1 4 w - 9 4 w ^ { 2 } - 1 4 w ^ { 3 } + w ^ { 4 } \right) } { 2 ( 1 + w ) ( 1 - w ) ^ { j + 7 } } ,\tag{2.16}
$$

$$
\sum _ { n \geq 0 } s _ { j , n } w ^ { n } = { \frac { 6 0 w ^ { j + 4 } } { ( 1 - w ) ^ { j + 8 } } } .
$$

Finally, for $\epsilon _ { 2 } , \epsilon _ { 3 } \in \{ \pm 1 \}$ and $n \geq 2$ put

$$
c = c ( n , \epsilon _ { 2 } , \epsilon _ { 3 } ) = 2 n - \epsilon _ { 2 } - \epsilon _ { 3 } , \qquad X _ { j , n , \epsilon } = c z _ { 1 } + \epsilon _ { 2 } z _ { 2 } + \epsilon _ { 3 } | z _ { 3 } | + 2 z _ { 4 } ,\tag{2.17}
$$

and

$$
\alpha _ { j , n } ( c ) = \frac { 2 r _ { j , n } } { c ^ { 2 } - 1 } + \frac { 4 c s _ { j , n } } { ( c ^ { 2 } - 1 ) ^ { 2 } } , \qquad \beta _ { j , n } ( c ) = \frac { 2 s _ { j , n } } { c ^ { 2 } - 1 } .\tag{2.18}
$$

Since $c \geq 2$ , the denominators in (2.18) do not vanish. The mode expansion of the correction, derived in Lemma 3.15, has the exponents $X _ { j , n , \epsilon }$ , and its coeficients are built from $\alpha _ { j , n } ( c )$ and $\beta _ { j , n } ( c )$ . We are now ready to state our main result.

Theorem 2.1. Define U on the ordered sector $\{ x _ { 1 } \geq \dots \geq x _ { 5 } \}$ by

$$
U ( \tau , x ) = x _ { 1 } + \frac { 1 } { 8 k } \sum _ { m \geq 0 } \sum _ { \epsilon \in \{ \pm 1 \} ^ { 3 } } \Bigl [ c _ { m , \epsilon } ^ { a } \mathcal { H } _ { 0 } \bigl ( X _ { m , \epsilon } , \tau \bigr ) + c _ { m , \epsilon } ^ { b } a _ { 4 } \mathcal { H } _ { 1 } \bigl ( X _ { m , \epsilon } , \tau \bigr ) \Bigr ]\tag{2.19}
$$

in Region III outside C, by

$$
\begin{array} { r l } & { U ( \tau , x ) = x _ { 1 } + \displaystyle \frac { 1 } { k } \sum _ { m \ge 0 } \displaystyle \sum _ { \epsilon \in \{ \pm 1 \} ^ { 3 } } c _ { m , \epsilon } ^ { ( 4 ) } \mathcal { H } _ { 0 } \big ( X _ { m , \epsilon } ^ { ( 4 ) } , \tau \big ) } \\ & { \quad \quad \quad \quad \quad \quad + \displaystyle \frac { 1 } { 2 k } \displaystyle \sum _ { \epsilon _ { 2 } , \epsilon _ { 3 } \in \{ \pm 1 \} } \epsilon _ { 2 } \epsilon _ { 3 } \displaystyle \sum _ { j \ge 0 } \frac { \big ( - 4 \big ) ^ { j } z _ { 4 } ^ { j } } { j ! } \displaystyle \sum _ { n \ge j + 2 } \Big [ \alpha _ { j , n } ( c ) \mathcal { H } _ { j } \big ( X _ { j , n , \epsilon } , \tau \big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad + \beta _ { j , n } ( c ) z _ { 1 } \mathcal { H } _ { j + 1 } ( X _ { j , n , \epsilon } , \tau ) \Big ] } \end{array}\tag{2.20}
$$

in Regions I and II outside C, and by

$$
U ( \tau , x ) = x _ { 1 } + \frac { \sqrt { \pi \tau } } { 2 \sqrt { 2 } } + \frac { 1 } { k } \int _ { 0 } ^ { \infty } \sinh t p ( t ) \mathcal { H } _ { 0 } \bigl ( 2 z _ { 4 } \coth t , \tau \bigr ) d t\tag{2.21}
$$

on ${ \mathcal { C } } ,$ and extend U to $\mathbb { R } ^ { 5 }$ by sorting the coordinates. Then the series converge absolutely, locally uniformly on the sets where they are used, together with the series of their derivatives of every order, each series being diferentiated within its own region, where its coordinates are linear in x; and the following hold.

(i) U is a classical solution of (2.1) on $( 0 , \infty ) \times \mathbb { R } ^ { 5 } .$ : for every $\tau > 0 , { \cal U } ( \tau , \cdot ) \in { \cal C } ^ { 2 } ( \mathbb { R } ^ { 5 } )$ , the derivatives $U _ { \tau }$ , ∇U and $\nabla ^ { 2 } U$ are jointly continuous in $( \tau , x )$ , and the equation holds at every point. Hence U is a viscosity solution, and $| U ( \tau , x ) - \varphi ( x ) | \leq C { \sqrt { \tau } }$ uniformly in x.

(ii) For every $\tau > 0$ and at every point x of the ordered sector, the rank direction $\mathbf { v } _ { * } =$ $( 1 , 0 , 1 , 0 , 0 )$ attains the Hamiltonian maximum,

$$
D _ { \mathbf { v } _ { * } } ^ { 2 } U ( \tau , x ) = 2 U _ { \tau } ( \tau , x ) \geq D _ { \mathbf { v } } ^ { 2 } U ( \tau , x ) \qquad f o r \ a l l \ \mathbf { v } \in \{ 0 , 1 \} ^ { 5 } .
$$

(iii) U has the parabolic scaling $U ( \ell ^ { 2 } \tau , \ell x ) = \ell U ( \tau , x )$ for $\ell > 0$ , its Laplace transform in τ is $\lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ , where u is the stationary solution of (2.2), and at the origin

$$
U ( \tau , 0 ) = \frac { 4 5 \pi ^ { 3 / 2 } } { 2 5 6 \sqrt { 2 } } \sqrt { \tau } , \qquad \nabla ^ { 2 } U ( \tau , 0 ) = \frac { 7 5 \pi ^ { 3 / 2 } } { 5 1 2 \sqrt { 2 \tau } } \Bigl ( I - { \textstyle \frac { 1 } { 5 } } \mathtt { 1 } \mathtt { 1 } \mathtt { 1 } ^ { T } \Bigr ) .\tag{2.22}
$$

Since C meets Region III only in the diagonal $\{ a _ { 4 } = 0 \} , ( 2 . 1 9 )$ applies at the points of Region III with $a _ { 4 } > 0$ . It agrees with (2.20) on the interface $z _ { 4 } = 0$ outside ${ \mathcal { C } } ,$ where both apply, and (2.21) is the limit of both at C (Theorem 4.3). The constant in (2.22) is $2 / { \sqrt { \pi } }$ times the stationary value $u ( 0 ) = 4 5 \pi ^ { 2 } / ( 5 1 2 \sqrt { 2 } )$ of [6], and the Hessian is $( \pi \tau ) ^ { - 1 / 2 }$ times the stationary Hessian $\begin{array} { r } { \frac { 5 } { 3 } u ( 0 ) ( I - \frac { 1 } { 5 } \mathbf { 1 } \mathbf { 1 } ^ { T } ) } \end{array}$ at the origin. As Bayraktar, Ekren and Zhang observed [3], the transform identity in (iii) forces the factor $2 / \sqrt { \pi } = \mathcal { L } ^ { - 1 } [ \lambda ^ { - 3 / 2 } ] ( 1 )$ whenever a single strategy is optimal for both problems; it is the ratio between the finite-horizon and geometric-horizon values conjectured in general by Gravin, Peres and Sivan [14, §1.2]. Since $U$ is a classical solution of (2.1) with $| U - \varphi | \leq C \sqrt { \tau }$ , it is the viscosity solution to which the rescaled values of the discrete game converge by Drenska and Kohn [13], so that the value $V ^ { M }$ of the M-round game satisfies lim $_ { M  \infty } V ^ { M } ( 0 ) / \sqrt { M } = U ( 1 , 0 ) = 4 5 \pi ^ { 3 / 2 } / ( 2 5 6 \sqrt { 2 } )$

Our second result identifies the set on which the COMB control is optimal. In decreasing rank order, COMB is $\mathbf { v } _ { C } = ( 1 , 0 , 1 , 0 , 1 )$ . Define the curvature gap

$$
\Delta ( \tau , x ) = D _ { \mathbf { v } _ { * } } ^ { 2 } U ( \tau , x ) - D _ { \mathbf { v } _ { C } } ^ { 2 } U ( \tau , x ) .\tag{2.23}
$$

By Theorem 2.1, $\Delta \geq 0$ in the ordered sector, and the following theorem identifies the set where equality holds.

Theorem 2.2. For every $\tau > 0$ and every point x of the ordered sector, COMB attains the Hamiltonian maximum, i.e., $\Delta ( \tau , x ) = 0$ , if and only if

$$
x _ { 1 } = x _ { 2 } a n d x _ { 3 } = x _ { 4 } .\tag{2.24}
$$

At every other point $\Delta > 0$

The equality set (2.24) is the same as for the stationary problem [6, Theorem 2.2]. In addition, the optimal control is not unique. Corollary 4.6 shows that, for every $\tau > 0$ , the controls $( 1 , 0 , 0 , 1 , 1 )$ in Region I, (1, 0, 0, 1, 0) in Region II, and $( 1 , 0 , 0 , 0 , 1 )$ and $( 1 , 0 , 0 , 1 , 0 )$ in Region III have the same curvature as $\mathbf { v } _ { * } ;$ these are exactly the ties of the stationary problem [6, Remark 2.3], transferred by the Laplace transform.

The rest of the paper is organized into two sections and three appendices. Section 3 proves the transform principle, collects the properties of the kernels $\mathcal { H } _ { j }$ , and derives the three expressions for U from the mode expansions of the stationary solution. Section 4 proves that the resulting function satisfies (2.1) and has the properties claimed in Theorem 2.1, and locates the COMB equality set of Theorem 2.2. The Hamiltonian inequality in Theorem 2.1 (ii) is proved with the assistance of a computer, in the sense made precise at the beginning of Section 4. In particular, the reduction to finitely many one-variable Gaussian series is proved in Section 4 and Appendices A and B—their signs are certified by exact rational arithmetic with scripts and data in the supplement [5], and Appendix C reports independent numerical checks.

## 3 Derivation of the finite-horizon formula

This section derives the explicit formula of Theorem 2.1. The derivation has three steps. We first explain the transform principle, Proposition 3.1, which produces from the stationary solution a solution of the Laplace-transformed problem in which the adversary is frozen at $\mathbf { v } _ { * } ;$ its inverse transform is our candidate. We then show that every exponential mode of the stationary solution inverts, by itself, to an exact solution of the linear equation, expressed through the kernels $\mathcal { H } _ { j }$ of (2.5) and (2.6). Finally we expand the stationary solution into such modes, region by region, and read of the finite-horizon formula. The expansions are exact and every mode inverts exactly; inverting a whole series term by term needs the summability of Lemma 4.1(c), and the passage from the series to the properties of U claimed in Theorem 2.1 requires the convergence and regularity results of Section 4, where the formula is proved to solve (2.1).

## 3.1 The transform principle

Suppose the adversary is frozen at the rank direction $\mathbf { v } _ { * }$ . The resulting linear problem is

$$
\begin{array} { r } { U _ { \tau } = \frac { 1 } { 2 } D _ { \mathbf { v } _ { \ast } } ^ { 2 } U , \qquad U ( 0 , \cdot ) = \varphi , } \end{array}\tag{3.1}
$$

in the ordered sector with the permutation face conditions $\partial _ { e _ { i } - e _ { i + 1 } } U = 0$ on $\{ x _ { i } = x _ { i + 1 } \}$ $1 \leq i \leq 4$ . These are exactly the conditions under which the extension of U to $\mathbb { R } ^ { 5 }$ by sorting the coordinates is $C ^ { 1 }$ across the walls $\{ x _ { i } = x _ { i + 1 } \}$ . Since ${ \mathcal L } [ U _ { \tau } ] = \lambda \widehat U - \varphi$ , the Laplace transform $\begin{array} { r } { \widehat { U } ( \lambda , x ) = \int _ { 0 } ^ { \infty } e ^ { - \lambda \tau } U ( \tau , x ) } \end{array}$ dτ of a solution should solve the resolvent equation

$$
\begin{array} { r } { \lambda \widehat U - \frac { 1 } { 2 } D _ { { \mathbf v } _ { * } } ^ { 2 } \widehat U = \varphi . } \end{array}\tag{3.2}
$$

The following observation, which we call the transform principle, produces a solution of (3.2) from the stationary solution alone. For four experts it is due to Bayraktar, Ekren and Zhang [3], who also state it for the nonlinear equations when the same strategy is optimal for both problems.

Proposition 3.1. Let u be the $C ^ { 2 }$ solution of (2.2) with $u - \varphi$ bounded, and suppose $\mathbf { v } _ { * }$ attains its Hamiltonian maximum at every x of the sector. For $\lambda > 0$ put $u ^ { \lambda } ( x ) = \lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ Then $u ^ { \lambda }$ solves (3.2) and satisfies the permutation face conditions.

Proof. By hypothesis u solves the linear equation $\begin{array} { r } { u \mathrm { ~ - ~ } \frac { 1 } { 2 } D _ { \mathbf { v } _ { * } } ^ { 2 } u = \varphi . } \end{array}$ . Since $\nabla ^ { 2 } u ^ { \lambda } ( x ) =$ $\lambda ^ { - 1 / 2 } ( \nabla ^ { 2 } u ) ( \sqrt { \lambda } x )$ and $\varphi$ is positively homogeneous of degree one,

$$
\begin{array} { r } { \lambda u ^ { \lambda } - \frac { 1 } { 2 } D _ { \mathbf { v } _ { * } } ^ { 2 } u ^ { \lambda } = \lambda ^ { - 1 / 2 } \Big [ u - \frac { 1 } { 2 } D _ { \mathbf { v } _ { * } } ^ { 2 } u \Big ] ( \sqrt { \lambda } x ) = \lambda ^ { - 1 / 2 } \varphi ( \sqrt { \lambda } x ) = \varphi ( x ) . } \end{array}
$$

The face conditions hold for u [6], and they are invariant under the dilation $x \mapsto { \sqrt { \lambda } } x$ □

Proposition 3.1 suggests that a solution of (3.1) has the Laplace transform

$$
\widehat { U } ( \lambda , x ) = \lambda ^ { - 3 / 2 } u \Big ( \sqrt \lambda x \Big ) ,\tag{3.3}
$$

that is,

$$
U ( \tau , x ) = \mathcal { L } ^ { - 1 } \Bigl [ \lambda \mapsto \lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x ) \Bigr ] ( \tau ) ,\tag{3.4}
$$

and we use (3.4) as the definition of our candidate U. Bayraktar, Ekren and Zhang [3] computed this inverse transform for four experts, heuristically, by moving the contour onto the branch cut, and then verified the result; we invert it mode by mode instead. It is important to point out that we do not need to prove that (3.4) is the unique solution of (3.2); instead in Section 4 we prove directly that the function defined by (3.4) is a classical solution of (2.1).

The most useful immediate consequence of (3.3) is at the origin.

Corollary 3.2. The candidate (3.4) satisfies $\begin{array} { r } { U ( \tau , 0 ) = \frac { 2 } { \sqrt { \pi } } u ( 0 ) \sqrt { \tau } } \end{array}$ . For five experts $u ( 0 ) =$ $4 5 \pi ^ { 2 } / ( 5 1 2 { \sqrt { 2 } } )$ , which gives the value of $U ( \tau , 0 )$ in (2.22).

Proof. By homogeneity $u ( \sqrt { \lambda } \cdot 0 ) = u ( 0 )$ , so (3.3) at $x = 0$ reads $\widehat { U } ( \lambda , 0 ) = u ( 0 ) \lambda ^ { - 3 / 2 }$ , and $\mathcal { L } ^ { - 1 } [ \lambda ^ { - 3 / 2 } ] = 2 \sqrt { \tau / \pi }$ . The value u(0) is given in [6, Theorem 2.1]. □

Proposition 4.2(i) proves the same identity for the function of Theorem 2.1.

## 3.2 The inversion kernels

We now collect several useful properties of the kernels $\mathcal { H } _ { j }$

Lemma 3.3. The kernels defined by (2.5) and (2.6), with G and E as in (2.4), satisfy

$$
\partial _ { X } \mathcal { H } _ { j } = - \mathcal { H } _ { j + 1 } , \qquad \partial _ { \tau } \mathcal { H } _ { j } = \partial _ { X } ^ { 2 } \mathcal { H } _ { j } = \mathcal { H } _ { j + 2 }\tag{3.5}
$$

for every $j \geq 0$ , every real X and every $\tau > 0$ . They satisfy (2.7) for every $j \geq 0$ when $X > 0$ , and $f o r j \le 2$ also when $X = 0$ . Moreover $\mathcal { H } _ { 0 } \geq 0$ and $\mathcal { H } _ { 1 } \geq 0 , \mathcal { H } _ { 0 } ( 0 , \tau ) = 2 \sqrt { \tau / \pi }$ ， and for $X > 0$ every $\mathcal { H } _ { j } ( X , \tau )  0 \ a s \ \tau \downarrow 0$ , faster than any power of τ .

Proof. Since $\begin{array} { r } { \partial _ { X } \mathcal { G } = - \frac { X } { 2 \tau } \mathcal { G } } \end{array}$ and $\partial _ { X } { \mathcal { E } } = - { \mathcal { G } }$ , diferentiating (2.5) gives $\partial _ { X } \mathcal { H } _ { j } = - \mathcal { H } _ { j + 1 }$ for $j = 0 , 1 , 2$ . If this holds for all indices up to some $j \geq 2$ , then diferentiating (2.6) gives

$$
\partial _ { X } \mathcal { H } _ { j + 1 } = \frac { 1 } { 2 \tau } \mathcal { H } _ { j } - \frac { X } { 2 \tau } \mathcal { H } _ { j + 1 } + \frac { j - 2 } { 2 \tau } \mathcal { H } _ { j } = - \Big ( \frac { X } { 2 \tau } \mathcal { H } _ { j + 1 } - \frac { j - 1 } { 2 \tau } \mathcal { H } _ { j } \Big ) = - \mathcal { H } _ { j + 2 } ,
$$

so $\mathcal { H } _ { j + 1 } = - \partial _ { X } \mathcal { H } _ { j }$ for every $j ,$ every real X and every $\tau > 0$ , and $\partial _ { X } ^ { 2 } \mathcal { H } _ { j } = \mathcal { H } _ { j + 2 }$ . For the time derivative, diferentiating (2.5) gives $\begin{array} { r } { \partial _ { \tau } \mathcal { E } = \frac { X } { 2 \tau } \mathcal { G } = \mathcal { H } _ { 3 } } \end{array}$ and $\partial _ { \tau } \mathcal { H } _ { 0 } = \mathcal { G } = \mathcal { H } _ { 2 }$ , and since $\partial _ { \tau }$ commutes with $\partial _ { X }$ , induction on $j$ with $\mathcal { H } _ { j + 1 } = - \partial _ { X } \mathcal { H } _ { j }$ gives $\partial _ { \tau } \mathcal { H } _ { j } = \mathcal { H } _ { j + 2 }$ for every $j .$ In particular the identity holds at $X = 0$ , where some of the exponents below vanish on walls of the sector.

The transforms of ${ \mathcal { G } } , { \mathcal { E } }$ and $2 \tau { \mathcal { G } } - X { \mathcal { E } }$ are the classical $\lambda ^ { - 1 / 2 } e ^ { - X \sqrt { \lambda } } , \lambda ^ { - 1 } e ^ { - X \sqrt { \lambda } }$ and $\lambda ^ { - 3 / 2 } e ^ { - X \sqrt { \lambda } }$ for $X \geq 0$ , which is (2.7) for $j \le 2$ . For $j \geq 3$ and $X > 0$ we argue by induction on $j$ . On $X \geq \delta > 0$ the bound (3.6) of Lemma 3.4 below, whose proof uses only (2.5) and $\left( 2 . 6 \right)$ , dominates $e ^ { - \lambda \tau } | \mathcal { H } _ { j } ( X , \tau ) |$ by an integrable function of τ. So the transform of $\mathcal { H } _ { j } =$ $- \partial _ { X } \mathcal { H } _ { j - 1 }$ is $- \partial _ { X }$ of the transform of $\mathcal { H } _ { j - 1 }$ , that is $- \partial _ { X } \left( \lambda ^ { ( j - 4 ) / 2 } e ^ { - X \sqrt { \lambda } } \right) = \lambda ^ { ( j - 3 ) / 2 } e ^ { - \dot { X } \sqrt { \lambda } }$ At $X = 0$ the argument stops at $j = 2$ , because the bound is not integrable there once $j \geq 3$ . Nonnegativity of $\mathcal { H } _ { 0 }$ follows from $\textstyle { \mathcal { H } } _ { 0 } = \int _ { X } ^ { \infty } { \mathcal { E } } ( s , \tau ) d s$ , and the small-τ statement from $\mathcal { E } ( X , \tau ) \leq e ^ { - X ^ { 2 } / ( 4 \tau ) }$ and the Gaussian factor of $\mathcal { G }$ □

The expansion in Regions I and II uses kernels $\mathcal { H } _ { j }$ of unbounded order, so we record how fast they can grow.

Lemma 3.4. There is an absolute constant $c _ { 0 }$ such that for all $j \ge 2 , X \ge 0$ and $\tau > 0$

$$
\left| \mathcal { H } _ { j } ( X , \tau ) \right| \le \frac { c _ { 0 } ^ { j } j ^ { j / 2 } } { \tau ^ { ( j - 1 ) / 2 } } e ^ { - X ^ { 2 } / ( 8 \tau ) } .\tag{3.6}
$$

Consequently, for every $Z \geq 0$

$$
\frac { ( 4 Z ) ^ { j } } { j ! } \big | \mathcal { H } _ { j } ( X , \tau ) \big | \leq c _ { 1 } \sqrt { \tau } \left( \frac { c _ { 1 } Z } { \sqrt { j \tau } } \right) ^ { j } e ^ { - X ^ { 2 } / ( 8 \tau ) } ,\tag{3.7}
$$

which tends to 0 faster than any geometric sequence in $j ,$ , locally uniformly in $Z$ and $\tau > 0$

Proof. Write $\mathcal { H } _ { n + 2 } = \mathcal { G } P _ { n }$ . By (2.5) and (2.6), $P _ { 0 } = 1 , P _ { 1 } = X / ( 2 \tau )$ and

$$
P _ { n + 1 } = { \frac { X } { 2 \tau } } P _ { n } - { \frac { n } { 2 \tau } } P _ { n - 1 } , \qquad n \geq 1 .
$$

Put $a = X / ( 2 \tau ) , b _ { n } = \sqrt { n / ( 2 \tau ) }$ and $A _ { n } = a + b _ { n }$ , which increases with $n .$ . We claim that $| P _ { n } | \leq A _ { n } ^ { n }$ . This holds for $n = 0 , 1$ , and if it holds up to $n _ { : }$ , then

$$
A _ { n + 1 } ^ { n + 1 } = a A _ { n + 1 } ^ { n } + b _ { n + 1 } A _ { n + 1 } A _ { n + 1 } ^ { n - 1 } \geq a A _ { n } ^ { n } + b _ { n + 1 } ^ { 2 } A _ { n - 1 } ^ { n - 1 } \geq a | P _ { n } | + { \frac { n } { 2 \tau } } | P _ { n - 1 } | \geq | P _ { n + 1 } | .
$$

With $w = X / \sqrt { 2 \tau }$ we have $A _ { n } ^ { n } = ( 2 \tau ) ^ { - n / 2 } ( w + \sqrt { n } ) ^ { n }$ and $e ^ { - X ^ { 2 } / ( 4 \tau ) } = e ^ { - w ^ { 2 } / 2 }$ , and since $w ^ { n } e ^ { - w ^ { 2 } / 4 } \leq ( 2 n / e ) ^ { n / 2 }$

$$
( w + \sqrt { n } ) ^ { n } e ^ { - w ^ { 2 } / 4 } \leq 2 ^ { n } \operatorname* { m a x } \bigl ( w , \sqrt { n } \bigr ) ^ { n } e ^ { - w ^ { 2 } / 4 } \leq 2 ^ { n } ( 2 n ) ^ { n / 2 } .
$$

Hence

$$
\vert \mathcal { H } _ { n + 2 } \vert \leq \frac { 2 ^ { n } ( 2 n ) ^ { n / 2 } } { \sqrt { \pi \tau } ( 2 \tau ) ^ { n / 2 } } e ^ { - X ^ { 2 } / ( 8 \tau ) } = \frac { 2 ^ { n } n ^ { n / 2 } } { \sqrt { \pi } \tau ^ { ( n + 1 ) / 2 } } e ^ { - X ^ { 2 } / ( 8 \tau ) } ,
$$

which is (3.6) with $j = n + 2$ after absorbing constants. For (3.7) use $j ! \geq ( j / e ) ^ { j }$ , which gives it with $c _ { 1 } = \operatorname* { m a x } ( 1 , 4 e c _ { 0 } )$ , since $\tau ^ { - ( j - 1 ) / 2 } = \sqrt { \tau } \tau ^ { - j / 2 }$ □

The key structural fact is that every mode of the expansions below is by itself an exact solution of the linear problem.

Lemma 3.5. Let $X ( x ) = d \cdot x$ be linear with $d \cdot \mathbf { v } _ { * } = \pm k$ , and let P be a polynomial with $\partial _ { \mathbf { v } _ { * } } P = 0$ . Then for every $j \geq 0$ the function $\Psi ( \tau , x ) = P ( x ) \mathcal { H } _ { j } ( X ( x ) , \tau )$ satisfies

$$
\begin{array} { r } { \partial _ { \tau } \Psi = \frac 1 2 D _ { \mathbf { v } _ { * } } ^ { 2 } \Psi . } \end{array}\tag{3.8}
$$

Moreover, if $P$ is homogeneous of degree $p ,$ then at every x with $X ( x ) > 0$ , and for $p \leq 2$ also at every x with $X ( x ) = 0$ 2

$$
\mathcal { L } \big [ P \mathcal { H } _ { p } ( X , \cdot ) \big ] ( \lambda ) = \lambda ^ { - 3 / 2 } P ( \sqrt { \lambda } x ) e ^ { - X ( \sqrt { \lambda } x ) } .\tag{3.9}
$$

Proof. Because $\partial _ { \mathbf { v } _ { * } } P = 0$ and $\partial _ { \mathbf { v } _ { * } } X = d \cdot \mathbf { v } _ { * }$ 2

$$
D _ { \mathbf { v } _ { * } } ^ { 2 } \left[ P \mathcal { H } _ { j } \right] = P \left( d \cdot \mathbf { v } _ { * } \right) { } ^ { 2 } \partial _ { X } ^ { 2 } \mathcal { H } _ { j } = k ^ { 2 } P \mathcal { H } _ { j + 2 } = 2 P \mathcal { H } _ { j + 2 } ,
$$

which equals 2∂<sub>τ</sub>Ψ by (3.5). For (3.9), $P ( { \sqrt { \lambda } } x ) = \lambda ^ { p / 2 } P ( x )$ and $\lambda ^ { - 3 / 2 } \lambda ^ { p / 2 } = \lambda ^ { ( p - 3 ) / 2 }$ , which under the stated conditions on $X ( x )$ is the transform of $\mathcal { H } _ { p } ( X ( x ) , \cdot )$ by Lemma 3.3. □

Combining the definition (3.4) with Lemma 3.5 gives the recipe we follow for the rest of this section.

Corollary 3.6. Suppose that on a region of the sector the stationary solution can be written as

$$
u ( x ) = x _ { 1 } + \frac { 1 } { k } \sum _ { \iota } C _ { \iota } P _ { \iota } ( x ) e ^ { - X _ { \iota } ( x ) } ,\tag{3.10}
$$

with $X _ { \iota }$ linear, $X _ { \iota } \geq 0$ on the region, $X , > 0$ on the region whenever $p _ { \iota } \geq 3 , d _ { \iota } \cdot \mathbf { v } _ { \ast } = \pm k$ and $P _ { \iota }$ homogeneous of degree $p _ { \iota }$ with $\partial _ { \mathbf { v } _ { * } } P _ { \iota } = 0$ . Let x be a point of the region at which the series in (3.11) below converges for every $\tau > 0$ to a continuous function of $\tau ,$ and at which, for every $\lambda > 0$ 2

$$
\sum _ { \iota } \int _ { 0 } ^ { \infty } e ^ { - \lambda \tau } \bigl | C _ { \iota } P _ { \iota } ( x ) { \mathcal { H } } _ { p _ { \iota } } ( X _ { \iota } ( x ) , \tau ) \bigr | d \tau < \infty .
$$

Then

$$
U ( \tau , x ) = x _ { 1 } + \frac { 1 } { k } \sum _ { \iota } C _ { \iota } P _ { \iota } ( x ) \mathcal { H } _ { p _ { \iota } } ( X _ { \iota } ( x ) , \tau ) ,\tag{3.11}
$$

the inverse transform in (3.4) being the continuous function of τ with the transform (3.3). Every term of (3.11) satisfies (3.8). If moreover $X , > 0$ at a point x of the region for every $\iota ,$ and the series converges uniformly for small τ near x, then $U ( \tau , \cdot ) \to x _ { 1 } = \varphi ~ a s ~ \tau \downarrow 0$ near x.

Proof. The summability lets the Laplace transform of the series in (3.11) be taken term by term. By (3.9), and by (3.10) at $\sqrt { \lambda } x$ , which lies in the region because the regions are cones, the result is $\lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ ; since a continuous function is determined by its Laplace transform, this gives (3.11). The terms satisfy (3.8) by Lemma 3.5, and the last statement holds because $\mathcal { H } _ { j } ( X , \tau )  0$ faster than any power of $\tau$ for each fixed $X > 0$ □

Remark 3.7. Nonnegativity of every exponent $X _ { \iota }$ is what makes the terminal condition come out of the series at all. It is a nontrivial structural property of the expansions below: several families of modes that appear formally with negative exponents turn out to have identically vanishing coeficients (Lemma 3.12). Nonnegativity alone is not quite enough for a pointwise statement, since $\mathcal { H } _ { 1 } ( 0 , \tau ) = 1 , \mathcal { H } _ { 2 } ( 0 , \tau ) = ( \pi \tau ) ^ { - 1 / 2 }$ is unbounded, and for $j \geq 3$ the recursion (2.6) gives $\mathcal { H } _ { j } ( 0 , \tau ) = 0$ for odd j and, for even $j ,$ , nonzero multiples of $\tau ^ { ( 1 - j ) / 2 }$ with alternating signs; of the kernels that are not identically zero at $X = 0$ , only $\mathcal { H } _ { 0 } ( 0 , \tau )$ tends to zero as $\tau \downarrow 0$ . The exponents do vanish somewhere, since on the diagonal every $X _ { \iota }$ is zero, and there the Region III series diverges. We therefore do not prove the terminal condition from the series: it is Theorem $4 . 3 ( \mathrm { d } )$ , which proves the stronger, uniform bound $| U ( \tau , x ) - \varphi ( x ) | \leq C { \sqrt { \tau } }$ from the contour representation of Section 4.2.

## 3.3 Four experts

The four-expert problem was solved by Bayraktar, Ekren and Zhang, in the geometric case in [4] and in the finite-horizon case in [3]. Nevertheless, we apply the recipe from this paper to the four-expert problem for two reasons: the mode expansion of the stationary four-expert solution is the background term in Regions I and II, and inverting it mode by mode gives the finite-horizon solution of [3] as a series. Throughout this subsection $\mathbf { v } _ { * }$ means the four-expert rank direction (1, 0, 1, 0), which is the restriction of the five-expert $\mathbf { v } _ { * }$ to the first four coordinates. In the coordinates (2.3), which involve only $x _ { 1 } , \ldots , x _ { 4 }$ , the ordered four-expert sector is $z _ { 1 } \ge z _ { 2 } \ge | z _ { 3 } |$ and the stationary solution is $\begin{array} { r } { u _ { 4 } = x _ { 1 } + \frac { 1 } { k } F _ { 4 } } \end{array}$ with $F _ { 4 }$ as in (2.13).

Lemma 3.8. For $z _ { 1 } \ge z _ { 2 } \ge | z _ { 3 } | \ge 0$ with $z _ { 1 } > 0$

$$
F _ { 4 } = \sum _ { m \ge 0 } \sum _ { \epsilon \in \{ \pm 1 \} ^ { 3 } \atop \epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } = ( - 1 ) ^ { m } } c _ { m , \epsilon } ^ { ( 4 ) } e ^ { - X _ { m , \epsilon } ^ { ( 4 ) } } ,\tag{3.12}
$$

with $c _ { m , \epsilon } ^ { ( 4 ) }$ and $X _ { m , \epsilon } ^ { ( 4 ) }$ as in (2.14), the series converging absolutely. Every exponent occurring with a nonzero coeficient satisfies $X _ { m , \epsilon } ^ { ( 4 ) } \geq 0 .$ , and $\partial _ { \mathbf { v } _ { * } } X _ { m , \epsilon } ^ { ( 4 ) } = - \epsilon _ { 2 } k$

Proof. Let $z _ { 1 } > 0$ , and expand the two transcendental factors of (2.13) in their exponential series. With n = 2m + 1,

$$
\theta ( z _ { 1 } ) \cosh z _ { 1 } = \frac { _ 1 } { 2 } \sum _ { m \geq 0 } \frac { ( - 1 ) ^ { m } } { n } \sum _ { \epsilon _ { 1 } = \pm 1 } e ^ { - ( n - \epsilon _ { 1 } ) z _ { 1 } } ,
$$

$$
{ \textstyle \frac { 1 } { 2 } } \Lambda ( z _ { 1 } ) \sinh z _ { 1 } = { \textstyle \frac { 1 } { 2 } } \sum _ { m \geq 0 } { \frac { 1 } { n } } \sum _ { \epsilon _ { 1 } = \pm 1 } \epsilon _ { 1 } e ^ { - ( n - \epsilon _ { 1 } ) z _ { 1 } } ,
$$

while cosh $z _ { 2 }$ cosh $\begin{array} { r } { z _ { 3 } = \frac { 1 } { 4 } \sum _ { \epsilon _ { 2 } , \epsilon _ { 3 } } e ^ { \epsilon _ { 2 } z _ { 2 } + \epsilon _ { 3 } z _ { 3 } } } \end{array}$ and sinh $z _ { 2 }$ sinh $\begin{array} { r } { z _ { 3 } = \frac { 1 } { 4 } \sum _ { \epsilon _ { 2 } , \epsilon _ { 3 } } \epsilon _ { 2 } \epsilon _ { 3 } e ^ { \epsilon _ { 2 } z _ { 2 } + \epsilon _ { 3 } z _ { 3 } } } \end{array}$ . Adding the two products gives

$$
\frac { 1 } { 8 } \sum _ { m \ge 0 } \frac { 1 } { 2 m + 1 } \sum _ { \epsilon \in \{ \pm 1 \} ^ { 3 } } \left[ ( - 1 ) ^ { m } + \epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } \right] e ^ { - X _ { m , \epsilon } ^ { ( 4 ) } } ,
$$

an absolutely convergent series in which the bracket is $2 ( - 1 ) ^ { m }$ when $\epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } = ( - 1 ) ^ { m }$ and 0 otherwise. This is (3.12) with the stated coeficients, except for the $m = 0$ entries. The single term with $\epsilon = ( 1 , 1 , 1 )$ has $X _ { 0 , ( 1 , 1 , 1 ) } ^ { ( 4 ) } = - ( z _ { 2 } + z _ { 3 } ) \le 0$ and coeficient $\textstyle { \frac { 1 } { 4 } }$ , that is ${ \frac { 1 } { 4 } } e ^ { z _ { 2 } + z _ { 3 } } ;$ the remaining elementary term of (2.13) is $- { \textstyle { \frac { 1 } { 2 } } } \sinh ( z _ { 2 } + z _ { 3 } ) = - { \textstyle { \frac { 1 } { 4 } } } e ^ { z _ { 2 } + z _ { 3 } } + { \textstyle { \frac { 1 } { 4 } } } e ^ { - ( z _ { 2 } + z _ { 3 } ) }$ . The growing exponentials cancel identically, and the surviving $\frac { 1 } { 4 } e ^ { - ( z _ { 2 } + z _ { 3 } ) }$ adds to the coeficient $\textstyle { \frac { 1 } { 4 } }$ of the mode $\epsilon = ( 1 , - 1 , - 1 )$ , which carries the same exponent $X _ { 0 , ( 1 , - 1 , - 1 ) } ^ { ( 4 ) } = z _ { 2 } + z _ { 3 }$ . This is the stated modification.

For the sign of the exponents, if $m \geq 1$ then $X _ { m , \epsilon } ^ { ( 4 ) } \geq 2 m z _ { 1 } - z _ { 2 } - | z _ { 3 } | \geq 2 m z _ { 1 } - 2 z _ { 1 } \geq 0 ;$ if $m = 0$ and $\epsilon _ { 1 } = - 1$ then $X ^ { ( 4 ) } \geq 2 z _ { 1 } - z _ { 2 } - | z _ { 3 } | \geq 0 ;$ and the two remaining $m = 0$ modes are $\epsilon = ( 1 , 1 , 1 )$ , whose coeficient is now zero, and $\epsilon = ( 1 , - 1 , - 1 )$ , with $X ^ { ( 4 ) } = z _ { 2 } + z _ { 3 } \ge 0$ Finally $\partial _ { \mathbf { v } _ { * } } z _ { 1 } = \partial _ { \mathbf { v } _ { * } } z _ { 3 } = 0$ and $\partial _ { \mathbf { v } _ { * } } z _ { 2 } = k$ , which gives $\partial _ { \mathbf { v } _ { * } } X _ { m , \epsilon } ^ { ( 4 ) } = - \epsilon _ { 2 } k$ □

Proposition 3.9. For $x _ { 1 } \geq x _ { 2 } \geq x _ { 3 } \geq x _ { 4 }$ not all equal and $\tau > 0$ , the solution $U _ { 4 }$ of the finite-horizon four-expert problem found in $I { \boldsymbol { \mathscr { 3 } } } J$ is

$$
U _ { 4 } ( \tau , x ) = x _ { 1 } + \frac { 1 } { k } \sum _ { m \geq 0 } \sum _ { \epsilon \in \{ \pm 1 \} ^ { 3 } \atop \epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } = ( - 1 ) ^ { m } } c _ { m , \epsilon } ^ { ( 4 ) } \mathcal { H } _ { 0 } \bigl ( X _ { m , \epsilon } ^ { ( 4 ) } , \tau \bigr ) ,\tag{3.13}
$$

the series converging absolutely and uniformly on compact sets. At the origin,

$$
U _ { 4 } ( \tau , 0 ) = \frac { 1 } { k } \cdot \frac { \pi } { 4 } \cdot 2 \sqrt { \tau / \pi } = \textstyle { \frac { 1 } { 2 } } \sqrt { \pi \tau / 2 } .\tag{3.14}
$$

Proof. The coordinates are not all equal exactly when $z _ { 1 } > 0$ , since $z _ { 1 } \ge z _ { 2 } \ge | z _ { 3 } |$ . Absolute convergence follows from $X _ { m , \epsilon } ^ { ( 4 ) } \geq 2 ( m - 1 ) z _ { 1 }$ for $m \ge 1 , 0 \le \mathcal { H } _ { 0 } ( X , \tau ) \le 2 \sqrt { \tau / \pi } e ^ { - X ^ { 2 } / ( 4 \tau ) }$ and $\begin{array} { r } { | c _ { m , \epsilon } ^ { ( 4 ) } | \leq \frac { 1 } { 2 ( 2 m + 1 ) } } \end{array}$ , which covers the exceptional coeficient $\begin{array} { r } { c _ { 0 , ( 1 , - 1 , - 1 ) } ^ { ( 4 ) } = \frac { 1 } { 2 } . } \end{array}$ . Since $\mathcal { H } _ { 0 } \geq 0$ has the transform $\lambda ^ { - 3 / 2 } e ^ { - X \sqrt { \lambda } }$ , the same bounds let the Laplace transform of (3.13) be taken term by term, and by Lemmas 3.5 and 3.8 it is $\lambda ^ { - 3 / 2 } u _ { 4 } ( \sqrt { \lambda } x )$ . Bayraktar, Ekren and Zhang prove that $U _ { 4 }$ is a classical solution with the same Laplace transform [3]; both functions are continuous in $\tau _ { \mathrm { { ; } } }$ so they coincide. At $x = 0$ the transform is $\lambda ^ { - 3 / 2 } u _ { 4 } \bar { ( 0 ) }$ with $u _ { 4 } ( 0 ) = F _ { 4 } ( 0 , 0 , 0 ) / k = \pi / ( 4 k )$ , which gives (3.14), the value found in [3]. □

Formula (3.13) is the four-expert solution of Bayraktar, Ekren and Zhang [3] in a diferent form. They invert the same transform on the branch cut, which gives an oscillatory integral $\begin{array} { r } { \int _ { - \infty } ^ { \infty } r ^ { - 2 } e ^ { - \tau r ^ { 2 } } W ( r , x ) } \end{array}$ dr whose integrand contains the 2π-periodic square wave and whose convergence at $r = 0$ rests on a cancellation. Inverting mode by mode gives instead a series of Gaussians and complementary error functions with explicit rational coeficients and nonnegative exponents, whose terms are bounded, so that no singular terms need to cancel.

## 3.4 Region III

In Region III the stationary solution is $\textstyle u = x _ { 1 } + { \frac { 1 } { k } } F$ with

$$
F = \sum _ { j = 0 } ^ { 3 } b _ { j } \sigma _ { j } ( a _ { 1 } , a _ { 2 } , a _ { 3 } ) ,\tag{3.15}
$$

where $\sigma _ { j }$ is the sum of the $\textstyle { \binom { 3 } { j } }$ products of $j$ factors sinh $a _ { i }$ and $3 - j$ factors cosh $a _ { i } .$ , and $b _ { j } = \mathcal { K } ^ { j } e$ evaluated at $a _ { 4 }$ . Here

$$
( { \cal Z } f ) ( X ) = 2 \sinh ^ { 2 } X \int _ { X } ^ { \infty } \frac { f ( t ) } { \sinh ^ { 3 } t } d t , \qquad K = - \coth X + \operatorname { c s c h } X { \cal 2 }\tag{3.16}
$$

are the operators of [6, Section 3], and e is the trace (2.8). Only the value $e ( a _ { 4 } )$ requires the quadrature: the derivatives of e in the $b _ { j }$ are eliminated by the compatibility relation $6 b _ { 1 } = e ^ { \prime } - 3$ and the trace equation (3.22) below. Explicitly, [6, Section 3.6] gives

$$
\begin{array} { c c c } { { b _ { 0 } = e , } } & { { 6 b _ { 1 } = e ^ { \prime } - 3 , } } & { { 4 2 b _ { 2 } = e ^ { \prime \prime } + 6 e + 1 8 \Lambda \sinh L , } } \\ { { } } & { { 3 3 6 b _ { 3 } = e ^ { \prime \prime \prime } + 2 0 e ^ { \prime } + 1 4 4 \Lambda \cosh L - 3 1 2 , } } & { { } } \end{array}\tag{3.17}
$$

all evaluated at $L = a _ { 4 } ;$ these follow from the two integration-by-parts identities $\partial _ { X } \mathcal { Z } =$ $\mathcal { T } ( \partial _ { X } + \mathcal { K } )$ and $\mathcal { K } ^ { 2 } - [ \partial _ { X } , \mathcal { K } ] =$ Id together with $\mathcal { K } 1 = - \Lambda$ sinh X and $\mathcal { K } ^ { 2 } 1 = 2 - \Lambda$ cosh X. We begin with the exponential expansion of the trace.

Proposition 3.10. For $L > 0$

$$
e ( L ) = \sum _ { m \geq 0 } \bigl ( A _ { m } + B _ { m } L \bigr ) e ^ { - 2 m L } ,\tag{3.18}
$$

with $A _ { m }$ and $B _ { m }$ as in (2.9). The series converges for every $L > 0$ and diverges at $L = 0$ where $e ( 0 ) = { 4 5 \pi ^ { 2 } } / { 5 1 2 }$ instead.

Proof. Diferentiating (2.8) once gives the first-order equation

$$
\cosh L e ^ { \prime } ( L ) - \sinh L e ( L ) = - \frac { 3 I ( L ) } { \sinh ^ { 5 } L } ,\tag{3.19}
$$

which is (B.17) of Appendix B. Insert the ansatz (3.18). With $w = e ^ { - 2 L }$ , the m-th summand $( A _ { m } + B _ { m } L ) w ^ { m }$ contributes $\big ( B _ { m } - 2 m ( A _ { m } + B _ { m } L ) \big ) w ^ { m }$ to $e ^ { \prime } ,$ , while $2 e ^ { - L }$ cosh $L = 1 + w$ and $2 e ^ { - L }$ sinh $L = 1 - w$ , and the closed form of I in the proof of Lemma 3.14 gives

$$
\frac { 6 e ^ { - L } I ( L ) } { \sinh ^ { 5 } L } = \frac { N ( w ) + 1 2 L w ^ { 3 } } { ( 1 - w ) ^ { 5 } } , \qquad N ( w ) = \textstyle { \frac { 1 } { 2 } } \bigl ( 1 - 3 w - 3 w ^ { 2 } + 3 w ^ { 4 } + 3 w ^ { 5 } - w ^ { 6 } \bigr ) .
$$

Multiplying (3.19) by $2 e ^ { - L }$ and collecting the coeficient of $w ^ { n }$ gives, with $A _ { - 1 } = B _ { - 1 } = 0$

$$
\begin{array} { r l } & { ( 2 n + 1 ) B _ { n } + ( 2 n - 3 ) B _ { n - 1 } = \frac { 1 } { 2 } ( n + 1 ) n ( n - 1 ) ( n - 2 ) , } \\ & { ( 2 n + 1 ) A _ { n } + ( 2 n - 3 ) A _ { n - 1 } = B _ { n } + B _ { n - 1 } + \nu _ { n } , } \end{array}\tag{3.20}
$$

where $\nu _ { n }$ is the coeficient of $w ^ { n }$ in the Taylor series of $N ( w ) ( 1 - w ) ^ { - 5 }$ at $w = 0$ , that is, $\nu _ { 0 } = { \textstyle { \frac { 1 } { 2 } } } , \nu _ { 1 } = 1$ and $\nu _ { n } = - { \textstyle \frac { 1 } { 2 } } ( 2 n - 1 ) ( n ^ { 2 } - n - 1 )$ for $n \geq 2$ . The coeficient $2 n + 1$ never vanishes, so (3.20) determines every coeficient. It gives $\begin{array} { r } { B _ { 0 } = B _ { 1 } = B _ { 2 } = 0 , B _ { 3 } = \frac { 1 2 } { 7 } } \end{array}$ $A _ { 0 } = A _ { 1 } = { \textstyle { \frac { 1 } { 2 } } } , A _ { 2 } = - { \frac { 2 } { 5 } }$ and, from $\begin{array} { r } { 7 A _ { 3 } = B _ { 3 } + B _ { 2 } + \nu _ { 3 } - 3 A _ { 2 } = \frac { 1 2 } { 7 } - \frac { 2 5 } { 2 } + \frac { 6 } { 5 } . } \end{array}$

$$
A _ { 3 } = - { \frac { 6 7 1 } { 4 9 0 } } .\tag{3.21}
$$

The closed forms (2.9) satisfy both relations of (3.20) for every n: after clearing denominators each is a polynomial identity in $n ,$ and $n = 0 , 1$ are checked directly. Since $A _ { m } = \mathcal { O } ( m ^ { 2 } )$ and $B _ { m } = \mathcal { O } ( m ^ { 3 } )$ , the series (3.18) converges for $| w | < 1$ together with its termwise derivative, so it solves (3.19) for $L > 0$ , and it tends to $\begin{array} { r } { A _ { 0 } = \frac { 1 } { 2 } } \end{array}$ as $L  \infty$ . The trace e solves the same equation and is bounded, so the diference of the two solves $y ^ { \prime } =$ tanh $L y$ and is a bounded multiple of cosh $L ,$ hence zero. This proves (3.18) for $L > 0$ . At $L = 0$ the terms $\begin{array} { r } { A _ { m } \sim - \frac { 3 } { 1 6 } m ^ { 2 } } \end{array}$ do not tend to zero, while $e ( 0 ) = 4 5 \pi ^ { 2 } / 5 1 2 \mid$ [6, Lemma 3.1]. □

Remark 3.11. Diferentiating (3.19) once more, and using $I ^ { \prime } = \sinh ^ { 4 } X \cosh ^ { 2 } X$ and (3.19) itself, gives the second-order trace equation $e ^ { \prime \prime } + 5$ coth X $e ^ { \prime } - 6 e = - 3$ coth X. After multiplication by sinh X it reads

$$
\sinh L e ^ { \prime \prime } + 5 \cosh L e ^ { \prime } - 6 \sinh L e + 3 \cosh L = 0 ,\tag{3.22}
$$

and inserting (3.18) gives the recursion

$$
\begin{array} { r l } & { ( 5 - 4 n ) B _ { n } + 2 ( 2 n + 1 ) ( n - 3 ) ( A _ { n } + B _ { n } L ) } \\ & { \qquad + ( 4 n + 1 ) B _ { n - 1 } - 2 ( 2 n - 3 ) ( n + 2 ) ( A _ { n - 1 } + B _ { n - 1 } L ) + 3 \delta _ { n 0 } + 3 \delta _ { n 1 } = 0 . } \end{array}\tag{3.23}
$$

Its constant part is resonant at $n = 3$ , where the coeficient of $A _ { 3 }$ vanishes, so (3.23) determines every coeficient except $A _ { 3 } ,$ , the amplitude of the homogeneous mode $e ^ { - 6 L } ;$ ; equivalently, it determines e only up to a multiple of a decaying solution of (3.22). This is what a local analysis of (3.22) cannot see. The first-order equation (3.19), which remembers the lower limit of the integral $I ,$ has no resonance and fixes $A _ { 3 }$ . The value (3.21) carries the same information as $e ( 0 ) = { 4 5 \pi ^ { 2 } } / { 5 1 2 }$ , which comes from the closed evaluation of the quadrature in [6, Lemma 3.1] and ultimately from $\textstyle \int _ { 0 } ^ { \infty }$ s csch $s d s = \pi ^ { 2 } / 4$ . As independent checks, we have compared (3.18) with the quadrature (2.8) to 40 digits at $L = 0 . 3 5 , 0 . 6 , 1 , 2$ , and (3.23) with (2.9) as exact rationals for every $m \leq 2 0 0$

Expanding each $b _ { j }$ in the same way, write

$$
b _ { j } ( L ) = \sum _ { m \ge 0 } \bigl ( \alpha _ { m } ^ { ( j ) } + \beta _ { m } ^ { ( j ) } L \bigr ) e ^ { - 2 m L } , \qquad j = 0 , 1 , 2 , 3 .\tag{3.24}
$$

Using (3.17) and $\Lambda ( L )$ sinh $\begin{array} { r } { L = \sum _ { m } J _ { m } e ^ { - 2 m L } , \Lambda ( L ) } \end{array}$ cosh $\begin{array} { r } { L = \sum _ { m } K _ { m } e ^ { - 2 m L } } \end{array}$ with $J _ { m } , K _ { m }$ as in (2.10), we obtain exactly the coeficients (2.11): diferentiating (3.18) term by term replaces $\left( A _ { m } , B _ { m } \right)$ by $( B _ { m } - 2 m A _ { m } , - 2 m B _ { m } )$ , and each further derivative repeats this substitution. Two properties of the resulting coeficients $c _ { m , \epsilon } ^ { a } , c _ { m , \epsilon } ^ { b }$ of (2.12) are what make the finite-horizon formula work.

Lemma 3.12. $c _ { m , \epsilon } ^ { b } = 0$ for $m \le 2$ and all ϵ. Moreover

$$
c _ { 0 , \epsilon } ^ { a } = \textstyle { \frac { 1 } { 2 } } \prod _ { i = 1 } ^ { 3 } ( 1 - \epsilon _ { i } ) , \qquad c _ { 1 , \epsilon } ^ { a } = \textstyle { \frac { 4 } { 3 } } \mathbb { 1 } \{ \epsilon _ { 1 } + \epsilon _ { 2 } + \epsilon _ { 3 } = - 1 \} , \qquad c _ { 2 , \epsilon } ^ { a } = 0 \quad i f \epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } = 1 .\tag{3.25}
$$

Consequently $c _ { 0 , \epsilon } ^ { a } = c _ { 0 , \epsilon } ^ { b } = 0$ unless $\epsilon = ( - 1 , - 1 , - 1 )$ , where $c _ { 0 , \epsilon } ^ { a } = 4$ , and $c _ { 1 , ( 1 , 1 , 1 ) } ^ { a } =$ $c _ { 1 , ( 1 , 1 , 1 ) } ^ { b } = 0$

Proof. Every $\beta _ { m } ^ { ( j ) }$ is a multiple of $B _ { m }$ , and $B _ { 0 } = B _ { 1 } = B _ { 2 } = 0$ . For the rest, (2.11) gives $\begin{array} { r } { \big ( \alpha _ { 0 } ^ { ( j ) } \big ) _ { j } = \big ( \frac { 1 } { 2 } , - \frac { 1 } { 2 } , \frac { 1 } { 2 } , - \frac { 1 } { 2 } \big ) , \mathrm { ~ s o ~ } c _ { 0 , \epsilon } ^ { a } = \frac { 1 } { 2 } \big ( s _ { 0 } - s _ { 1 } + s _ { 2 } - s _ { 3 } \big ) = \frac { 1 } { 2 } \prod ( 1 - \epsilon _ { i } ) ; \big ( \alpha _ { 1 } ^ { ( j ) } \big ) _ { j } = \big ( \frac { 1 } { 2 } , - \frac { 1 } { 6 } , - \frac { 1 } { 6 } , \frac { 1 } { 2 } \big ) } \end{array}$ so $\begin{array} { r } { c _ { 1 , \epsilon } ^ { a } = \frac { 1 } { 2 } ( 1 + s _ { 3 } ) - \frac { 1 } { 6 } ( s _ { 1 } + s _ { 2 } ) } \end{array}$ , which is checked to be $\textstyle { \frac { 4 } { 3 } }$ when $s _ { 1 } = - 1$ and 0 in the three other cases; and $\begin{array} { r } { \left( \alpha _ { 2 } ^ { ( j ) } \right) _ { j } = \left( - \frac { 2 } { 5 } , \frac { 4 } { 1 5 } , - \frac { 4 } { 1 5 } , \frac { 2 } { 5 } \right) , \mathrm { s o } \ c _ { 2 , \epsilon } ^ { a } = - \frac { 2 } { 5 } ( 1 - s _ { 3 } ) + \frac { 4 } { 1 5 } ( s _ { 1 } - s _ { 2 } ) } \end{array}$ , which vanishes when $s _ { 3 } = 1$ □

Proposition 3.13. On Region III with $a _ { 4 } > 0 , f o r \tau > 0$ , the candidate (3.4) is given by (2.19). Every exponent occurring with a nonzero coeficient satisfies $X _ { m , \epsilon } \ge 0$ on Region III, and $\partial _ { \mathbf { v } _ { * } } X _ { m , \epsilon } = - \epsilon _ { 3 } k , \partial _ { \mathbf { v } _ { * } } a _ { 4 } = 0$

Proof. Expanding each cosh $a _ { i }$ and sinh $a _ { i }$ in (3.15) gives $\begin{array} { r } { \sigma _ { j } = \frac { 1 } { 8 } \sum _ { \epsilon } s _ { j } ( \epsilon ) e ^ { \epsilon \cdot a } } \end{array}$ , so by (3.24) and (2.12),

$$
F = \frac { _ { 1 } } { 8 } \sum _ { m \ge 0 } \sum _ { \epsilon } \bigl ( c _ { m , \epsilon } ^ { a } + c _ { m , \epsilon } ^ { b } a _ { 4 } \bigr ) e ^ { - X _ { m , \epsilon } } .
$$

The prefactors are 1 and $a _ { 4 }$ , homogeneous of degrees 0 and 1, so Corollary 3.6 yields (2.19) with $\mathcal { H } _ { 0 }$ and $\mathcal { H } _ { 1 } ;$ its hypotheses of continuity in τ and summability of the transforms are Lemma 4.1(a) and $\mathrm { ( c ) }$ . For the exponents: if $m \geq 2$ then $\epsilon \cdot a \le a _ { 1 } + a _ { 2 } + a _ { 3 } \le 3 a _ { 4 } \le 2 m a _ { 4 }$ If $m = 1$ then the only sign vector with $\epsilon \cdot a > 2 a _ { 4 }$ possible is $\epsilon = ( 1 , 1 , 1 )$ , whose coeficients vanish by Lemma 3.12; for the other seven, ${ \epsilon \cdot a \leq 2 a _ { 4 } }$ because at most two of the $a _ { i }$ enter with a + sign and each is at most $a _ { 4 }$ . If $m = 0$ then only $\epsilon = ( - 1 , - 1 , - 1 )$ has a nonzero coeficient, and then $X _ { 0 , \epsilon } = a _ { 1 } + a _ { 2 } + a _ { 3 } = y _ { 1 } \geq 0$ . Finally the direction $\mathbf { v } _ { * }$ moves y along $k ( 1 , - 1 , 1 , 0 )$ , so $\partial _ { \mathbf { v } _ { * } } a _ { 1 } = \partial _ { \mathbf { v } _ { * } } a _ { 2 } = \partial _ { \mathbf { v } _ { * } } a _ { 4 } = 0$ and $\partial _ { \mathbf { v } _ { * } } a _ { 3 } = k$ □

The lowest modes are worth recording. By Lemma 3.12, at $m = 0$ the only mode is $c ^ { a } = 4 { \mathrm { ~ a t ~ } } \epsilon = ( - , - , - )$ , with $X _ { 0 , \epsilon } = y _ { 1 } ;$ at $m = 1$ one has $\textstyle c ^ { a } = { \frac { 4 } { 3 } }$ at the three ϵ with one + sign; at $m = 2 , c ^ { a } = - \textstyle { \frac { 1 2 } { 5 } } \mathrm { ~ a t ~ } ( - , - , - )$ and $c ^ { a } = - { \frac { 4 } { 1 5 } }$ at the three ϵ with two + signs; and at $m = 3$ the linear prefactor first appears, with $c ^ { b } = { \frac { 9 6 } { 7 } }$ at $\epsilon = ( - , - , - )$

## 3.5 Regions I and II

In Regions I and II the stationary solution is $\textstyle u = x _ { 1 } + { \frac { 1 } { k } } F$ with

$$
F = F _ { 4 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) + \Delta F , \qquad \Delta F = \int _ { z _ { 1 } } ^ { \infty } \sinh t p ( t ) e ^ { - 2 z _ { 4 } \coth t } \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( z _ { 2 } ) \Phi _ { t } ( | z _ { 3 } | ) d t ,\tag{3.26}
$$

where $\Phi _ { t } ( X ) = \sinh ( t - X ) / \sinh t$ for $0 \leq X \leq t$ and 0 otherwise, and $p$ is the density (2.15) [6, Theorem 2.1]. The background $F _ { 4 }$ was treated in Section 3.3; it remains to expand $\Delta F$

A remark on the coordinates before we begin. The formula (3.26) involves $\left| z _ { 3 } \right|$ and $z _ { 4 } = y _ { 4 } - \operatorname* { m a x } ( z _ { 3 } , 0 )$ , which are only piecewise linear in x: they are linear on Region I, where $z _ { 3 } \leq 0$ and $z _ { 4 } = y _ { 4 }$ , and linear on Region II, where $z _ { 3 } \geq 0$ and $z _ { 4 } = y _ { 4 } - z _ { 3 }$ . Everything below is therefore carried out on each of the two open regions separately, so that the hypotheses of Lemma 3.5, linear exponents and homogeneous prefactors, are met. The two resulting expressions agree across the interface $z _ { 3 } = 0$ , because the correction depends on $z _ { 3 }$ only through |z<sub>3</sub>| and because $z _ { 4 } = y _ { 4 }$ there. We first put the density in exponential form.

Lemma 3.14. With $w = e ^ { - 2 t }$ and $\Pi ( w ) = 1 - 1 4 w - 9 4 w ^ { 2 } - 1 4 w ^ { 3 } + w ^ { 4 } , f o r t > 0 ,$

$$
p ( t ) = \frac { w \Pi ( w ) } { 2 ( 1 + w ) ( 1 - w ) ^ { 5 } } + \frac { 6 0 t w ^ { 3 } } { ( 1 - w ) ^ { 6 } } .\tag{3.27}
$$

Proof. Using sinh $\begin{array} { r } { ( 2 \ell t ) = \frac { 1 } { 2 } e ^ { 2 \ell t } ( 1 - w ^ { \ell } ) } \end{array}$ and $\begin{array} { r } { \sinh ^ { 6 } t = \frac { 1 } { 6 4 } e ^ { 6 t } ( 1 - w ) ^ { 6 } \sin I ( X ) = \frac { 1 } { 1 9 2 } \sinh 6 X - } \end{array}$ $\frac { 1 } { 6 4 }$ sinh $4 X - { \frac { 1 } { 6 4 } }$ sinh $2 X + { \overline { { 1 6 } } }$ gives

$$
\begin{array} { r l r } {  { \frac { 1 5 I ( t ) } { \sinh ^ { 6 } t } = \frac { \frac { 5 } { 2 } ( 1 - 3 w - 3 w ^ { 2 } + 3 w ^ { 4 } + 3 w ^ { 5 } - w ^ { 6 } ) + 6 0 t w ^ { 3 } } { ( 1 - w ) ^ { 6 } } } } \\ & { } & { = \frac { 5 ( 1 - 2 w - 5 w ^ { 2 } - 5 w ^ { 3 } - 2 w ^ { 4 } + w ^ { 5 } ) } { 2 ( 1 - w ) ^ { 5 } } + \frac { 6 0 t w ^ { 3 } } { ( 1 - w ) ^ { 6 } } , } \end{array}
$$

the last step by dividing the sextic by ${ 1 - w }$ . Adding $\textstyle { \frac { 1 } { 2 } }$ tanh $\begin{array} { r } { t = \frac { 1 - w } { 2 ( 1 + w ) } } \end{array}$ and −3 coth $t =$ $- \frac { 3 ( 1 + w ) } { 1 - w }$ over the common denominator $2 ( 1 + w ) ( 1 - w ) ^ { 5 }$ produces the numerator

$$
( 1 - w ) ^ { 6 } - 6 ( 1 + w ) ^ { 2 } ( 1 - w ) ^ { 4 } + 5 ( 1 + w ) \big ( 1 - 2 w - 5 w ^ { 2 } - 5 w ^ { 3 } - 2 w ^ { 4 } + w ^ { 5 } \big ) = w \Pi ( w ) ,
$$

which is (3.27).

Lemma 3.15. For $z _ { 1 } > 0 , z _ { 1 } \geq z _ { 2 } \geq | z _ { 3 } | \geq 0$ , and $z _ { 4 } \geq 0$

$$
\Delta F = \frac { 1 } { 2 } \sum _ { \epsilon _ { 2 } , \epsilon _ { 3 } \in \{ \pm 1 \} } \epsilon _ { 2 } \epsilon _ { 3 } \sum _ { j \geq 0 } \frac { ( - 4 z _ { 4 } ) ^ { j } } { j ! } \sum _ { n \geq j + 2 } \left[ \alpha _ { j , n } ( c ) + \beta _ { j , n } ( c ) z _ { 1 } \right] e ^ { - X _ { j , n , \epsilon } } ,\tag{3.28}
$$

with $c , X _ { j , n , \epsilon } , \alpha _ { j , n }$ and $\beta _ { j , n }$ as in $( 2 . 1 7 ) \mathrm { - } ( 2 . 1 8 )$ . All exponents satisfy $X _ { j , n , \epsilon } \geq 2 z _ { 1 } > 0$ , and $\partial _ { \mathbf { v } _ { * } } X _ { j , n , \epsilon } = \epsilon _ { 2 } k$ while $\partial _ { \mathbf { v } _ { * } } z _ { 1 } = \partial _ { \mathbf { v } _ { * } } z _ { 4 } = 0$

Proof. Write $d = | z _ { 3 } |$ and expand the three ingredients of the integrand in $w = e ^ { - 2 t }$ . First,

$$
\sinh t \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( z _ { 2 } ) \Phi _ { t } ( d ) = \frac { \prod _ { i } \sinh ( t - \zeta _ { i } ) } { \sinh ^ { 2 } t } = \frac { 1 } { 2 } \sum _ { \epsilon \in \{ \pm 1 \} ^ { 3 } } \Bigl ( \prod _ { i } \epsilon _ { i } \Bigr ) e ^ { - \epsilon \cdot \zeta } \frac { e ^ { \sigma t } w } { ( 1 - w ) ^ { 2 } } ,
$$

where ${ \boldsymbol { \zeta } } = ( z _ { 1 } , z _ { 2 } , d )$ and $\sigma = \epsilon _ { 1 } + \epsilon _ { 2 } + \epsilon _ { 3 }$ , using $\Pi _ { i }$ sinh $\begin{array} { r } { . ( t - \zeta _ { i } ) = \frac { 1 } { 8 } \sum _ { \epsilon } ( \prod \epsilon _ { i } ) e ^ { \sigma t } e ^ { - \epsilon \cdot \zeta } } \end{array}$ and $\sinh ^ { - 2 } t = 4 w ( 1 - w ) ^ { - 2 }$ . Second, coth $t = 1 + 2 w / ( 1 - w )$ gives

$$
e ^ { - 2 z _ { 4 } \coth t } = e ^ { - 2 z _ { 4 } } \sum _ { j \geq 0 } \frac { ( - 4 z _ { 4 } ) ^ { j } } { j ! } \frac { w ^ { j } } { ( 1 - w ) ^ { j } } ,
$$

which converges absolutely for $t \geq z _ { 1 } > 0$ . Third, Lemma 3.14 gives

$$
\frac { w ^ { 1 + j } p ( t ) } { ( 1 - w ) ^ { 2 + j } } = \frac { w ^ { j + 2 } \Pi ( w ) } { 2 ( 1 + w ) ( 1 - w ) ^ { j + 7 } } + \frac { 6 0 t w ^ { j + 4 } } { ( 1 - w ) ^ { j + 8 } } = \sum _ { n } \bigl ( r _ { j , n } + t s _ { j , n } \bigr ) w ^ { n } .
$$

Since $e ^ { \sigma t } w ^ { n } = e ^ { - \nu t }$ with $\nu = 2 n - \sigma$ , and $\nu \geq 2 ( j + 2 ) - 3 > 0$ on the support of $r _ { j , \ast }$ <sub>·</sub>, the t-integrals are elementary:

$$
\int _ { z _ { 1 } } ^ { \infty } e ^ { - \nu t } d t = \frac { e ^ { - \nu z _ { 1 } } } { \nu } , \qquad \int _ { z _ { 1 } } ^ { \infty } t e ^ { - \nu t } d t = \frac { e ^ { - \nu z _ { 1 } } } { \nu } \Big ( z _ { 1 } + \frac { 1 } { \nu } \Big ) .
$$

Assembling, the ϵ-term carries the exponential $e ^ { - \nu z _ { 1 } - \epsilon \cdot \zeta - 2 z _ { 4 } }$ , that is $e ^ { - X _ { j , n , \epsilon } }$ with $X _ { j , n , \epsilon } =$ $( \nu + \epsilon _ { 1 } ) z _ { 1 } + \epsilon _ { 2 } z _ { 2 } + \epsilon _ { 3 } d + 2 z _ { 4 }$ . Because $\nu + \epsilon _ { 1 } = 2 n - \epsilon _ { 2 } - \epsilon _ { 3 } = c _ { ; }$ the exponent does not depend on $\epsilon _ { 1 }$ , so the two branches $\nu = c \mp 1$ may be summed first. With the signs $\begin{array} { r } { \prod _ { i } \epsilon _ { i } = \epsilon _ { 1 } \epsilon _ { 2 } \epsilon _ { 3 } } \end{array}$

$$
\sum _ { \epsilon _ { 1 } = \pm 1 } \frac { \epsilon _ { 1 } } { c - \epsilon _ { 1 } } = \frac { 2 } { c ^ { 2 } - 1 } , \qquad \sum _ { \epsilon _ { 1 } = \pm 1 } \frac { \epsilon _ { 1 } } { ( c - \epsilon _ { 1 } ) ^ { 2 } } = \frac { 4 c } { ( c ^ { 2 } - 1 ) ^ { 2 } } ,
$$

which give exactly (2.18) and (3.28).

For the sign of the exponents, recall that $c = 2 n - \epsilon _ { 2 } - \epsilon _ { 3 }$ with $n \geq 2$ . If $\epsilon _ { 2 } = \epsilon _ { 3 } = 1$ then $X _ { j , n , \epsilon } \geq c z _ { 1 } \geq 2 z _ { 1 }$ . Otherwise $c \geq 2 n \geq 4 .$ , and $z _ { 2 } \leq z _ { 1 } , | z _ { 3 } | \leq z _ { 1 } , z _ { 4 } \geq 0$ give

$$
X _ { j , n , \epsilon } \geq c z _ { 1 } - z _ { 2 } - | z _ { 3 } | + 2 z _ { 4 } \geq ( c - 2 ) z _ { 1 } \geq 2 z _ { 1 } .
$$

Finally $\partial _ { \mathbf { v } _ { * } } z _ { 1 } = \partial _ { \mathbf { v } _ { * } } z _ { 3 } = \partial _ { \mathbf { v } _ { * } } z _ { 4 } = 0$ and $\partial _ { \mathbf { v } _ { * } } z _ { 2 } = k$

Proposition 3.16. In Regions I and II with $z _ { 1 } > 0 , f o r \tau > 0$ , the candidate (3.4) is given by (2.20).

Proof. Apply Corollary 3.6 to (3.26) using Lemmas 3.8 and 3.15. The prefactor $z _ { 4 } ^ { j }$ is homogeneous of degree $j$ and $z _ { 4 } ^ { j } z _ { 1 }$ of degree $j + 1$ , and both are annihilated by $\partial _ { \mathbf { v } _ { * } }$ , which produces the kernels $\mathcal { H } _ { j }$ and $\mathcal { H } _ { j + 1 }$ respectively; their exponents are positive by Lemma 3.15, as Corollary 3.6 requires for the kernels of order at least three. The continuity in $\tau$ and the summability of the transforms that Corollary 3.6 also asks for are Lemma 4.1(b) and (c).

The correction in (2.20) contains $z _ { 3 }$ only through $\left| z _ { 3 } \right|$ , its background is the four-expert formula, valid on the whole four-expert sector, and at $z _ { 3 } = 0$ one has $z _ { 4 } = y _ { 4 }$ in both regions, so the single expression (2.20) serves Regions I and II: the two region-wise applications of Lemma 3.5 produce the same function, and the interface $z _ { 3 } = 0$ needs no separate matching. The interface between Regions II and III is $z _ { 4 } = 0$ , that is $a _ { 1 } = 0$ , where outside $\mathcal { C }$ the two formulas (2.19) and (2.20) agree together with their first and second derivatives; this is a consequence of Theorem 4.3. At $z _ { 1 } = 0$ the first four experts are tied, and one must use the separate quadrature (2.21) rather than substitute into the series, since the series converges only for $z _ { 1 } > 0$ . We derive that quadrature now.

Proposition 3.17. On the top-four collision set ${ \mathcal { C } } ,$ where $x _ { 1 } = x _ { 2 } = x _ { 3 } = x _ { 4 } \geq x _ { 5 }$ and $z _ { 1 } = z _ { 2 } = z _ { 3 } = 0$ , the candidate (3.4) is given by the single quadrature (2.21).

Proof. On C one has $\Phi _ { t } ( z _ { 1 } ) = \Phi _ { t } ( z _ { 2 } ) = \Phi _ { t } ( | z _ { 3 } | ) = \Phi _ { t } ( 0 ) = 1$ and $F _ { 4 } ( 0 , 0 , 0 ) = \theta ( 0 ) = \pi / 4$ , so (3.26) reduces to $\begin{array} { r } { F = \frac { \pi } { 4 } + \int _ { 0 } ^ { \infty } } \end{array}$ sinh $t p ( t ) e ^ { - 2 z _ { 4 } \coth t } d t$ . The set C is a cone on which $z _ { 4 }$ scales, so $\lambda ^ { - 3 / 2 } F ( { \sqrt { \lambda } } . )$ is $\textstyle { \frac { \pi } { 4 } } \lambda ^ { - 3 / 2 } + \int _ { 0 } ^ { \infty }$ sinh $t p ( t ) \lambda ^ { - 3 / 2 } e ^ { - \sqrt { \lambda } 2 z _ { 4 } \coth t } d t$ , and $\mathcal { L } ^ { - 1 } [ \lambda ^ { - 3 / 2 } ] = 2 \sqrt { \tau / \pi }$ together with (2.7) gives (2.21); the exponent $2 z _ { 4 } \coth t$ is nonnegative, and the integral converges because sinh $\begin{array} { r } { t p ( t ) = \frac { 1 } { 1 4 } t ^ { 2 } + \mathcal { O } ( t ^ { 4 } ) } \end{array}$ at zero and, by (3.27), sinh $\begin{array} { r } { \mathrm { ~  ~ \xi ~ } p ( t ) = \frac { 1 } { 4 } e ^ { - t } + \mathcal { O } ( e ^ { - 3 t } ) } \end{array}$ at infinity. □

Formula (2.21) is a useful independent check. At $x = 0$ it must reproduce Corollary 3.2, which forces $\textstyle \int _ { 0 } ^ { \infty }$ sinh $\begin{array} { r } { t p ( t ) d t = \frac { 4 5 \pi ^ { 2 } } { 5 1 2 } - \frac { \pi } { 4 } ; } \end{array}$ that is exactly $e ( 0 ) - e _ { 4 } ( 0 )$ , in agreement with the Green representation of $e - e _ { 4 }$ in the companion paper. Numerically both sides equal 0.08204753591704610024 to 20 digits, and (2.21) returns $4 5 \pi ^ { 3 / 2 } \sqrt { \tau } / ( 2 5 6 \sqrt { 2 } )$ at $x = 0$ for $\tau = 1$ and $\tau = 4$ to all digits computed.

## 4 Verification

The derivation of Section 3 produces a function U that inverts the Laplace transform of the stationary solution mode by mode, but it does not prove that U solves (2.1). This section addresses this in several steps. We first prove in Section 4.1 that the series converge, that U solves the linear equation (3.1) in each region, that it has the parabolic scaling, and that its Laplace transform is (3.3). Section 4.2 proves that $U ( \tau , \cdot )$ is $C ^ { 2 }$ on all of $\mathbb { R } ^ { 5 }$ including the collision set $\mathcal { C }$ where the mode series degenerate, and that $| U - \varphi | \leq C \sqrt { \tau }$ , by continuing the stationary formulas to a sector of complex scaling parameters and inverting the Laplace transform on a Hankel contour; this is the regularity of a classical solution, and it removes every regularity question from the rest of the argument. Section 4.3 shows that every curvature identity of the stationary solution transfers to U, which identifies the ties among the optimal controls, and locates the COMB equality set by a symmetry. Section 4.4 states the Hamiltonian inequalities $D _ { \mathbf { v } } ^ { 2 } U \leq D _ { \mathbf { v } _ { \ast } } ^ { 2 } U$ in the two regional forms in which they are proved, explains the reduction to finitely many scalar sign certificates, and describes what is delegated to the computer. Section 4.5 assembles the proofs of Theorems 2.1 and 2.2, extending the regional inequalities to every state by continuity.

Part of the verification is delegated to a computer, and the division of labor is as follows. The arguments that reduce the Hamiltonian inequalities to finite computations are given in full in the text and in Appendices A and B: the transform identity that turns each inequality into the complete monotonicity of a curvature gap of the stationary solution, the positive heat-exit interpolation from a cube or a square to its corners, and the positivity-preserving evolutions in the transverse variable. What remains after these reductions is finite and of two kinds. The first is a list of exact identities among rational functions and stored operator trees, among them the 64 Hessian contractions in Region III, the 64 corner decompositions in Regions I and II, and the partial fractions of the mode coeficients; these are verified by exact rational arithmetic. The second is the sign of 41 explicit functions of one variable, each a lattice Gaussian series of the form (4.25) below with rational coeficients; their positivity is proved on the whole positive axis, by Poisson summation for small arguments, by first-mode domination for large ones, and by outward-rounded interval arithmetic on 1616 cells in between, with a rigorous bound for the truncated tail. Numerical quadrature and sign sampling enter nowhere, and rounding is controlled by directed enclosure. These computations are carried out three times, on three trusted bases. The certificate scripts find the cell subdivisions and certify them with the arbitrary-precision interval arithmetic of mpmath, so their trusted base is the integer arithmetic of CPython, the symbolic algebra of SymPy for the identity audits, and the interval library. A separate checker re-derives every identity and every cell from the stored witnesses using only the Python standard library, in exact rational arithmetic with an interval exponential built from the Taylor series with outward rounding; no floating-point arithmetic enters this route, and its trusted base is CPython alone. The Lean formalization described at the end of Section 4.4 proves every enclosure from Mathlib’s real numbers. In this sense the computer-assisted part of the proof is an exact computation with the same logical status as a long hand calculation. The supplement [5] is a self-contained, versioned package that runs ofline in a few minutes, and Section 4.4 says at each point which of its scripts establishes what. Independently of these scripts, the proofs of Theorems 2.1 and 2.2, the 1616 interval certificates included, have been formalized in the Lean proof assistant; the end of Section 4.4 describes the scope of that formalization.

## 4.1 Convergence and the transform identity

We begin with the convergence of the two series. In Region III only the kernels $\mathcal { H } _ { 0 }$ and $\mathcal { H } _ { 1 }$ occur, and the argument is the one already used for four experts; in Regions I and II the order of the kernel is unbounded, and Lemma 3.4 is what makes the double series converge.

Lemma 4.1. (a) The Region III series (2.19) converges absolutely, and uniformly on sets where $a _ { 4 } \geq \delta > 0$ , |x| is bounded and τ lies in a compact subset of $( 0 , \infty )$ . The same is true of the series obtained by diferentiating term by term any number of times in x and in τ.

(b) The Regions I/II series (2.20) converges absolutely, and uniformly on sets where $z _ { 1 } \ge \delta > 0$ , |x| is bounded and τ lies in a compact subset of $( 0 , \infty )$ . The same is true of the series obtained by diferentiating term by term any number of times in x and in $\tau ,$ the terms being diferentiated as functions of the regional coordinates, which are linear in x on each of Regions I and II.

(c) Let $\lambda > 0$ . At every x of Region III with $a _ { 4 } > 0$ , and at every x of Regions I and II with $z _ { 1 } > 0$ , the terms $T _ { \iota }$ of the series (2.19), respectively (2.20), satisfy

$$
\sum _ { \iota } \int _ { 0 } ^ { \infty } e ^ { - \lambda \tau } \bigl | T _ { \iota } ( \tau , x ) \bigr | d \tau < \infty .
$$

In (a) and $( b )$ the majorants of the twice diferentiated series are $\mathcal { O } ( \tau ^ { - 1 / 2 } )$ as $\tau \downarrow 0$ 2 uniformly in x on the stated sets.

Proof. We first prove (a). By (2.11) and (2.9) the coeficients satisfy $| c _ { m , \epsilon } ^ { a } | + | c _ { m , \epsilon } ^ { b } | \le$ $C ( 1 + m ) ^ { 6 }$ , while $X _ { m , \epsilon } \geq ( 2 m - 3 ) a _ { 4 }$ for $m \geq 2$ by the proof of Proposition 3.13. Since $0 \leq \mathcal { H } _ { 0 } ( X , \tau ) \leq 2 \sqrt { \tau / \pi } e ^ { - X ^ { 2 } / ( 4 \tau ) }$ and $0 \leq \mathcal { H } _ { 1 } ( X , \tau ) \leq e ^ { - X ^ { 2 } / ( 4 \tau ) }$ , the m-th group of terms is at most $C ( 1 + m ) ^ { 6 } ( 1 + a _ { 4 } ) e$ −(2m $- 3 ) ^ { 2 } \delta ^ { 2 } / ( 4 \tau )$ , which is summable and dominates uniformly on the stated sets. Diferentiating a term ℓ times in x and once in τ replaces $\mathcal { H } _ { 0 } , \mathcal { H } _ { 1 }$ by kernels of order at most $\ell + 3$ , with polynomial prefactors in $x ,$ and by (3.6) the bound $C ( 1 + m ) ^ { 6 + \ell } ( 1 + | x | ) ^ { \ell } \tau ^ { - ( \ell + 2 ) / 2 } e ^ { - X _ { m , \epsilon } ^ { 2 } / ( 8 \tau ) }$ holds for these; for $m \geq 2$ the Gaussian factor again gives a summable majorant. The finitely many modes with $m \leq 1$ have exponents that are strictly positive except for the single mode $m = 0 , \epsilon = ( - 1 , - 1 , - 1 )$ , whose exponent is $y _ { 1 }$ and whose coeficient $c ^ { b }$ vanishes; its second derivatives involve $\mathcal { H } _ { 2 } = \mathcal { G } \le ( \pi \tau ) ^ { - 1 / 2 }$ at most, which gives the $\mathcal { O } ( \tau ^ { - 1 / 2 } )$ statement.

We now prove (b). The four-expert background is the series of Lemma 3.8 inverted term by term, and it and its diferentiated series are treated as in (a), since $| c _ { m , \epsilon } ^ { ( 4 ) } | \leq 1 / ( 2 ( 2 m + 1 ) )$ the exceptional $c _ { 0 , ( 1 , - 1 , - 1 ) } ^ { ( 4 ) } = \textstyle { \frac { 1 } { 2 } }$ included, and $X _ { m , \epsilon } ^ { ( 4 ) } \geq 2 ( m - 1 ) z _ { 1 }$ for $m \geq 1$ . For the correction, expanding (2.16) by the binomial series gives $s _ { j , n } = 6 0 \binom { n + 3 } { j + 7 }$ and, after writing

(1 $\ b + w ) ^ { - 1 } = ( 1 - w ) ( 1 - w ^ { 2 } ) ^ { - 1 }$

$$
\begin{array} { c } { { r _ { j , n } = \displaystyle \frac 1 2 \sum _ { \ell \geq 0 } \Big [ { \binom { n + 3 - 2 \ell } { j + 5 } } - 1 4 { \binom { n + 2 - 2 \ell } { j + 5 } } - 9 4 { \binom { n + 1 - 2 \ell } { j + 5 } } } } \\ { { - 1 4 { \binom { n - 2 \ell } { j + 5 } } + { \binom { n - 1 - 2 \ell } { j + 5 } } \Big ] , } } \end{array}
$$

with $\binom { N } { K } = 0$ for $N < K$ , so $r _ { j , n }$ is a sum of at most $\textstyle { \frac { 1 } { 2 } } ( n + 1 )$ groups of five binomial coeficients $\left( { \bf \Phi } _ { j + 5 } \right)$ with upper index at most $n + 3$ and weights of total size $1 + 1 4 + 9 4 + 1 4 + 1 = 1 2 4$ hence

$$
| r _ { j , n } | + | s _ { j , n } | \leq C ( n + 1 ) 2 ^ { n } .
$$

Since $c = 2 n - \epsilon _ { 2 } - \epsilon _ { 3 } \geq 2$ , (2.18) gives $| \alpha _ { j , n } ( c ) | + | \beta _ { j , n } ( c ) | \le C ( n + 1 ) 2 ^ { n }$ as well. By (3.7), applied to $\mathcal { H } _ { j }$ with $Z = z _ { 4 }$ and, when $z _ { 4 } > 0$ , to $\mathcal { H } _ { j + 1 }$ through $( 4 z _ { 4 } ) ^ { j } / j ! = { \textstyle \frac { j + 1 } { 4 z _ { 4 } } } ( 4 z _ { 4 } ) ^ { j + 1 } / ( j + 1 ) !$ (when $z _ { 4 } = 0$ only the terms with $j = 0$ remain), and the elementary bounds on $\mathcal { H } _ { 0 } , \mathcal { H } _ { 1 }$ , the term of (2.20) indexed by $( j , n , \epsilon )$ is bounded in absolute value by

$$
C ( 1 + | x | ) \big ( 1 + \sqrt { \tau } \big ) ( n + 1 ) 2 ^ { n } \left( \frac { c _ { 2 } z _ { 4 } } { \sqrt { ( j + 1 ) \tau } } \right) ^ { j } e ^ { - X _ { j , n , \epsilon } ^ { 2 } / ( 8 \tau ) } ,
$$

with an absolute constant $c _ { 2 }$ . The j-sum converges faster than any geometric series, uniformly for $z _ { 4 }$ bounded. For the n-sum, note that $X _ { j , n , \epsilon } \geq ( 2 n - 2 ) z _ { 1 } { \mathrm { : ~ i f ~ } } \epsilon _ { 2 } = \epsilon _ { 3 } = 1$ then $c = 2 n - 2$ and the terms $\epsilon _ { 2 } z _ { 2 } + \epsilon _ { 3 } | z _ { 3 } |$ are nonnegative, while otherwise $c \geq 2 n$ and $\epsilon _ { 2 } z _ { 2 } + \epsilon _ { 3 } | z _ { 3 } | \geq - 2 z _ { 1 }$ Hence $e ^ { - X _ { j , n , \epsilon } ^ { 2 } / ( 8 \tau ) } ~ \leq ~ e ^ { - ( 2 n - 2 ) ^ { 2 } \delta ^ { 2 } / ( 8 \tau ) }$ beats $2 ^ { n }$ and the n-sum converges as well. The diferentiated series obey the same bounds with $j$ replaced by $j + \ell$ and an extra factor $C ( 1 + | x | ) ^ { \ell } ( 1 + \tau ^ { - \ell / 2 } )$ , by (3.5) and (3.6). Finally, every exponent of the correction in (2.20) is at least $2 z _ { 1 }$ by Lemma 3.15, so the Gaussian factors make the majorants of the correction bounded as $\tau \downarrow 0$ , uniformly on the stated sets. In the four-expert background the exponents with $m \geq 2$ satisfy $X _ { m , \epsilon } ^ { ( 4 ) } \geq 2 ( m - 1 ) z _ { 1 } > 0$ , while the finitely many modes with $m \leq 1$ have exponents that vanish on some collision walls with $z _ { 1 } > 0$ , for instance $X _ { 0 , ( 1 , - 1 , - 1 ) } ^ { ( 4 ) } = y _ { 1 }$ on $\{ x _ { 1 } = x _ { 2 } \}$ ; their prefactors are constants, so their second derivatives are bounded by a multiple of $\dot { \mathcal { H } } _ { 2 } \leq ( \pi \tau ) ^ { - 1 / 2 }$ , which gives the $\mathcal { O } ( \tau ^ { - 1 / 2 } )$ statement.

We now prove (c), fixing $\lambda > 0$ and writing $C _ { \lambda }$ for constants that depend only on $\lambda .$ In Region III the kernels are $\mathcal { H } _ { 0 } , \mathcal { H } _ { 1 } \geq 0$ , whose integrals against $e ^ { - \lambda \tau }$ are $\lambda ^ { - 3 / 2 } e ^ { - X \sqrt { \lambda } }$ and $\lambda ^ { - 1 } e ^ { - X { \sqrt { \lambda } } }$ by Lemma 3.3. The sum in (c) is therefore at most $\begin{array} { r l } { \frac { 1 } { 8 k } \sum _ { m , \epsilon } \bigl ( | c _ { m , \epsilon } ^ { a } | \lambda ^ { - 3 / 2 } + } & { { } } \end{array}$ $| c _ { m , \epsilon } ^ { b } | a _ { 4 } \lambda ^ { - 1 } ) e ^ { - X _ { m , \epsilon } \sqrt { \lambda } }$ , which converges because the coeficients are $\mathcal { O } ( ( 1 + m ) ^ { 6 } )$ and $X _ { m , \epsilon } \geq$ $( 2 m - 3 ) a _ { 4 }$ for m $\geq 2$ . The four-expert background of (2.20) is treated in the same way, with $X _ { m , \epsilon } ^ { ( 4 ) } \geq 2 ( m - 1 ) z _ { 1 }$ . For the correction, the coeficient bound $C ( n + 1 ) 2 ^ { n }$ used in (b) is too weak, because for large $\tau$ the Gaussian factor $e ^ { - X _ { j , n , \epsilon } ^ { 2 } / ( 8 \tau ) }$ gives no decay in $n ,$ , so we use two sharper estimates. First, by the expansions of $r _ { j , n }$ and $s _ { j , n }$ in the proof of (b), $s _ { j , n } = 6 0 \binom { n + 3 } { j + 7 }$ and $r _ { j , n }$ is a combination of binomial coeficients $\binom { N } { j + 5 }$ with $N \leq n + 3$ and with weights of total size at most 31n; since $c \geq 2$ , it follows that

$$
| \alpha _ { j , n } ( c ) | + | \beta _ { j , n } ( c ) | \leq 2 \big ( | r _ { j , n } | + | s _ { j , n } | \big ) \leq C \frac { ( n + 4 ) ^ { j + 7 } } { j ! } .
$$

Second, $\mathcal { H } _ { i + 2 } ( \cdot , \tau ) = ( - \partial _ { X } ) ^ { i } \mathcal { G } ( \cdot , \tau )$ by (3.5), and $\mathcal { G } ( \cdot , \tau )$ is entire. On the circle $| \zeta -$ $X | = X / 4$ one has Re $\zeta ^ { 2 } \ge \frac { 9 } { 1 6 } X ^ { 2 }$ , hence $| \mathcal { G } ( \zeta , \tau ) | \le \mathcal { G } ( X / \sqrt { 2 } , \tau )$ , and Cauchy’s estimate gives $| \mathcal { H } _ { i + 2 } ( X , \tau ) | \le i ! ( 4 / X ) ^ { i } \mathcal { G } ( X / \sqrt { 2 } , \tau )$ for $X ~ > ~ 0$ . The transform of ${ \mathcal { G } } ( X / { \sqrt { 2 } } , \cdot )$ is $\lambda ^ { - 1 / 2 } e ^ { - X \sqrt { \lambda / 2 } }$ , and $\mathcal { H } _ { 0 } , \mathcal { H } _ { 1 }$ are covered by Lemma 3.3 as before, so for $X > 0$ and every $i \geq 0$

$$
\int _ { 0 } ^ { \infty } e ^ { - \lambda \tau } \big | \mathcal { H } _ { i } ( X , \tau ) \big | d \tau \leq C _ { \lambda } i ! ( 4 / X ) ^ { i } ( 1 + X ) ^ { 2 } e ^ { - X \sqrt { \lambda / 2 } } .
$$

Apply this with $X = X _ { j , n , \epsilon }$ . By Lemma 3.15 and the lower bound $X _ { j , n , \epsilon } \geq ( 2 n - 2 ) z _ { 1 }$ from the proof of (b), $X \ge \operatorname* { m a x } ( 2 , 2 n - 2 ) z _ { 1 } \ge ( n + 4 ) z _ { 1 } / 3$ , so that $1 6 ( n + 4 ) / X \le 4 8 / z _ { 1 }$ and $4 z _ { 1 } / X \le 2$ . The $( j , n , \epsilon )$ term of the correction therefore contributes at most

$$
\begin{array} { r l } {  { C _ { \lambda } \frac { ( 4 z _ { 4 } ) ^ { j } } { j ! } \frac { ( n + 4 ) ^ { j + 7 } } { j ! } \Big [ j ! \Big ( \frac { 4 } { X } \Big ) ^ { j } + ( j + 1 ) ! \Big ( \frac { 4 } { X } \Big ) ^ { j + 1 } z _ { 1 } \Big ] ( 1 + X ) ^ { 2 } e ^ { - X \sqrt { \lambda / 2 } } } \qquad } & { } \\ & { \leq C _ { \lambda } \frac { ( 9 6 z _ { 4 } / z _ { 1 } ) ^ { j } } { j ! } ( n + 4 ) ^ { 7 } e ^ { - ( n - 1 ) z _ { 1 } \sqrt { \lambda / 2 } } , } \end{array}
$$

where we used $2 j + 3 \leq 3 \cdot 2 ^ { j }$ and $( 1 + X ) ^ { 2 } e ^ { - X \sqrt { \lambda / 2 } } \leq C _ { \lambda } e ^ { - X \sqrt { \lambda / 2 } / 2 }$ . The sum over $j$ is at most $e ^ { 9 6 z _ { 4 } / z _ { 1 } }$ , and the sum over n converges. An estimate of the same form, with explicit constants, is proved in the Lean formalization. □

Proposition 4.2. Let U be the function of Theorem 2.1. Then, for every $\tau > 0$

(i) U is well defined: every exponent that occurs with a nonzero coeficient is nonnegative on its region, and on the diagonal the value is $\begin{array} { r } { x _ { 1 } + \frac { 2 } { \sqrt { \pi } } u ( 0 ) \sqrt { \tau } ; } \end{array}$

(ii) in the interior of each region U satisfies the linear equation $\begin{array} { r } { U _ { \tau } = \frac { 1 } { 2 } D _ { { \mathbf v } _ { \ast } } ^ { 2 } U _ { \cdot } } \end{array}$

(iii) $U ( \ell ^ { 2 } \tau , \ell x ) = \ell U ( \tau , x ) ~ f o r ~ \ell > 0 ;$

(iv) the Laplace transform of U in $\tau$ is $\lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ , where u is the stationary five-expert solution, and the Laplace integral converges absolutely for every $\lambda > 0$ and every x.

Proof. Convergence is Lemma 4.1, nonnegativity of the exponents is Lemma 3.8, Proposition 3.13 and Lemma 3.15, and the collision set $\mathcal { C }$ is covered by Proposition 3.17. The value on the diagonal is proved after (iv).

For (ii), apply Lemma 3.5 term by term: every exponent X has $| \partial _ { \mathbf { v } _ { * } } X | = k$ and every prefactor is annihilated by $\partial _ { \mathbf { v } _ { * } }$ , and term-by-term diferentiation is legitimate by Lemma 4.1.

Statement (iv) is the identity (3.3) for the function of Theorem 2.1. At $x \notin { \mathcal { C } } .$ Lemma 4.1(c) shows that the Laplace integral converges absolutely and may be taken term by term. Each term transforms as in Lemma 3.5, whose hypothesis holds because the kernels of order at least three have positive exponents (Lemma 3.15). The result is the mode expansion of $\lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ , which is valid at $\sqrt { \lambda }$ x because the regions are cones. On C the integrand of (2.21) is nonnegative, so Tonelli’s theorem justifies the computation of the transform in the proof of Proposition 3.17. The transform of the curvatures is treated in Theorem 4.3(b), at every $\boldsymbol { x } \in \mathbb { R } ^ { 5 }$

At $x = 0 , ( \mathrm { i v } )$ reads $\widehat { U } ( \lambda , 0 ) = u ( 0 ) \lambda ^ { - 3 / 2 }$ , which is the transform of $\scriptstyle { \frac { 2 } { \sqrt { \pi } } } u ( 0 ) { \sqrt { \tau } }$ . Both functions are continuous in $\tau ,$ so they agree; this is Corollary 3.2 for the function of

Theorem 2.1. On the diagonal $x = c \mathbb { 1 }$ one has $z _ { 4 } = 0$ , so (2.21) gives $U ( \tau , c \mathbb { 1 } ) = c + U ( \tau , 0 )$ and this completes (i).

Statement (iii) follows directly from $\mathcal { H } _ { j } ( \ell X , \ell ^ { 2 } \tau ) = \ell ^ { 1 - j } \mathcal { H } _ { j } ( X , \tau )$ , which is immediate from (2.7), because every exponent is linear in x and every prefactor that multiplies $\mathcal { H } _ { j }$ is homogeneous of degree j. It also follows from (iv), since $\ell \lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ is the transform of $\tau \mapsto U ( \ell ^ { 2 } \tau , \ell x )$ when (iv) holds. □

## 4.2 Regularity at the collision set

The series of Theorem 2.1 converge, with all their derivatives, only away from ${ \mathcal { C } } ,$ and on $\mathcal { C }$ the function is given by the separate quadrature (2.21). Nothing so far says that the derivatives of U match up across ${ \mathcal { C } } ,$ or even across the interfaces and walls where they do converge. The following theorem settles all of this at once. Its proof does not use the series: it continues the stationary formulas into a sector of complex scaling parameters, where the stationary regularity result of [6] propagates by the identity theorem, and then represents U by an absolutely convergent inverse Laplace integral in which one may diferentiate under the integral sign at every point of $\mathbb { R } ^ { 5 }$ . Throughout, $u _ { 0 } = u ( 0 ) = 4 5 \pi ^ { 2 } / ( 5 1 2 \sqrt { 2 } )$ and $\begin{array} { r } { \bar { x } = \frac { 1 } { 5 } \sum _ { i } x _ { i } } \end{array}$

Theorem 4.3. For every $\tau > 0$ the function $U ( \tau , \cdot )$ is $C ^ { 2 }$ on all $o f \mathbb { R } ^ { 5 }$ , including the collision set C and the diagonal. The spatial derivatives of U through order two are jointly continuous in $( \tau , x )$ on $( 0 , \infty ) \times \mathbb { R } ^ { 5 }$ , and so are all time derivatives ∂<sup>j</sup><sub>τ</sub>U together with their spatial derivatives through order two. Moreover:

(a) $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathbb { R } ^ { 5 } } \| D _ { x } ^ { 2 } U ( \tau , x ) \| \le C \tau ^ { - 1 / 2 } ; } \end{array}$

(b) for every $\mathbf { v } \in \{ 0 , 1 \} ^ { 5 }$ , every $\boldsymbol { x } \in \mathbb { R } ^ { 5 }$ and every $\lambda > 0$

$$
\int _ { 0 } ^ { \infty } e ^ { - \lambda \tau } D _ { \mathbf { v } } ^ { 2 } U ( \tau , x ) d \tau = D _ { \mathbf { v } } ^ { 2 } \big [ \lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x ) \big ] = \lambda ^ { - 1 / 2 } \big ( D _ { \mathbf { v } } ^ { 2 } u \big ) \big ( \sqrt { \lambda } x \big ) ,\tag{4.1}
$$

and the two regional expressions (2.19) and (2.20) agree, together with their first and second derivatives, at the points of the interface $a _ { 1 } = 0$ outside ${ \mathcal { C } } ,$

(c) on the diagonal,

$$
D _ { x } ^ { 2 } U ( \tau , c \mathbb { 1 } ) = \frac { 7 5 \pi ^ { 3 / 2 } } { 5 1 2 \sqrt { 2 \tau } } \Big ( I - \frac { 1 } { 5 } \mathbb { 1 } \mathbb { 1 } ^ { T } \Big ) , s o t h a t \ D _ { \mathbf { v } } ^ { 2 } U ( \tau , c \mathbb { 1 } ) = \frac { 7 5 \pi ^ { 3 / 2 } } { 5 1 2 \sqrt { 2 \tau } } \cdot \frac { m ( 5 - m ) } { 5 }\tag{4.2}
$$

for a binary control v with m ones;

$$
( d ) \ | U ( \tau , x ) - \varphi ( x ) | \leq C \sqrt { \tau } \ f o r \ e v e r y \ x \in \mathbb { R } ^ { 5 } \ a n d \ \tau > 0 .
$$

The theorem does not assert a locally Lipschitz Hessian: it provides two continuous spatial derivatives, and Lipschitz continuity of the spatial Hessian across the collision sets is a diferent question, which we leave open.

The main new estimate is a lower bound for $\operatorname { R e } ( z \coth z )$ on a sector; it is what allows the moving endpoint of the Regions I/II quadrature to be kept, so that no truncation error ever has to be inverted.

Lemma 4.4. There is $c > 0$ such that

$$
\mathrm { R e } \big ( z \coth z \big ) \geq c ( 1 + | z | ) \qquad f o r | \arg z | \leq \pi / 3 .\tag{4.3}
$$

In particular, for $| \eta | \le \pi / 3$ and $t > 0$

$$
\operatorname { R e } \left( e ^ { i \eta } \coth ( e ^ { i \eta } t ) \right) \geq c ( 1 + t ^ { - 1 } ) .\tag{4.4}
$$

Proof. Write $z = a + i b$ with $a > 0$ and $\vert b \vert \le \sqrt { 3 } a$ . A direct calculation gives

$$
\operatorname { R e } ( z \coth z ) = { \frac { a \sinh ( 2 a ) + b \sin ( 2 b ) } { \cosh ( 2 a ) - \cos ( 2 b ) } } ,\tag{4.5}
$$

whose denominator is positive. $\mathrm { I f } \ b \sin ( 2 b ) \geq 0$ the numerator is positive. Otherwise $| b | > \pi / 2 .$ hence $a > \pi / ( 2 \sqrt { 3 } )$ , and

$$
a \sinh ( 2 a ) + b \sin ( 2 b ) \geq a \bigl ( \sinh ( 2 a ) - { \sqrt { 3 } } \bigr ) > 0 ,
$$

because sinh $( \pi / { \sqrt { 3 } } ) > { \sqrt { 3 } }$ . So $\mathrm { R e } ( z \coth z ) > 0$ on the closed sector. $\mathrm { A s } ~ z  0 .$ , z coth $z  1$ uniformly in the sector, and as $| z | \to \infty$ in the sector coth $z  1$ uniformly while $a \geq | z | / 2 ;$ compactness on the intervening annulus gives (4.3). Applying it to $z = e ^ { i \eta } t$ and dividing by t gives (4.4). □

Proof of Theorem 4.3. Let $\Sigma = \{ s \neq 0 : | \arg s | < \pi / 3 \}$ . For real $x ,$ sort the coordinates and retain the resulting real regional coordinates, and let $f _ { s } ( x )$ be the radial analytic continuation of $u ( s x )$ : in the regional stationary formulas, replace every linear coordinate by s times that coordinate, so that in particular $d = | z _ { 3 } |$ is replaced by sd and not by $\left| s z _ { 3 } \right|$ , and scale the integration ray of the Regions $\mathrm { I } / \mathrm { I I }$ quadrature by s as well. For $s > 0$ this is $u ( s x )$ . We shall prove five properties of $f _ { s } .$ for each fixed x the map $s \mapsto f _ { s } ( x )$ is holomorphic on $\Sigma ;$ for each fixed $s \in \Sigma$ the complex-valued function $f _ { s }$ of the real variable x is globally $C ^ { 2 } ;$ ; with a constant independent of s and $x ,$

(4.6)

$$
\begin{array} { r } { \| D _ { x } ^ { 2 } f _ { s } ( x ) \| \leq C | s | ^ { 2 } ; } \end{array}\tag{4.7}
$$

$$
f _ { s } ( 0 ) = u _ { 0 } , \qquad D _ { x } f _ { s } ( 0 ) = \frac { s } { 5 } \mathbb { 1 } ;
$$

and, with $x _ { 1 }$ the largest coordinate of $x .$

$$
| f _ { s } ( x ) - s x _ { 1 } | \leq C .\tag{4.8}
$$

For the estimates it sufices to take $s = e ^ { i \eta }$ with $| \eta | \le \pi / 3$ , since $f _ { \rho e ^ { i \eta } } ( x ) = f _ { e ^ { i \eta } } ( \rho x )$ and $\rho e ^ { i \eta } x _ { 1 } = e ^ { i \eta } ( \rho x ) .$ <sub>1</sub> for $\rho > 0 ;$ all bounds below are uniform in η.

We first treat Regions I and II, whose collision face $z _ { 1 } = 0$ is the delicate part. Put $r = z _ { 1 }$ $q = z _ { 4 }$ and $\zeta = ( r , z _ { 2 } , d )$ , so that $0 \leq d \leq z _ { 2 } \leq r$ and $q \geq 0$ , and write $a = e ^ { i \eta }$ . Changing variables $t \mapsto$ at in the correction of (3.26), the continued correction is

$$
G _ { a } ( \zeta , q ) = a \int _ { r } ^ { \infty } \frac { p ( a t ) } { \sinh ^ { 2 } ( a t ) } \prod _ { j = 1 } ^ { 3 } \sinh \bigl ( a ( t - \zeta _ { j } ) \bigr ) e ^ { - 2 a q \coth ( a t ) } d t .\tag{4.9}
$$

For positive real scaling this is exactly the stationary correction, and the expression with a replaced by $s \in \Sigma$ is holomorphic in $s ,$ because on compact subsets of Σ the integral converges locally uniformly by the bounds that follow; those bounds also cover $r \ = \ 0$ Let $m ( t ) = \operatorname* { m i n } ( t , 1 )$ . The elementary hyperbolic estimates on the sector, together with $p ( z ) = z / 1 4 + \mathcal { O } ( z ^ { 3 } )$ at zero and the rational form (3.27) at infinity, give, for $0 \leq \zeta \leq t$

$$
\begin{array} { r l r } & { } & { | \sinh ( a t ) p ( a t ) | \le C m ( t ) ^ { 2 } ( 1 + t ) e ^ { - t / 2 } , \qquad | \coth ( a t ) | \le C / m ( t ) , } \\ & { } & { \displaystyle | \frac { \sinh ( a ( t - \zeta ) ) } { \sinh ( a t ) } \Big | \le C , \qquad \Big | \partial _ { \zeta } ^ { j } \frac { \sinh ( a ( t - \zeta ) ) } { \sinh ( a t ) } \Big | \le C m ( t ) ^ { - j } \quad ( j = 1 , 2 ) ; } \end{array}\tag{4.10}
$$

they follow from $| \sinh ( a t ) | \geq c m ( t ) e ^ { t \cos \eta } , | \sinh ( a ( t - \zeta ) ) | \leq C m ( t ) e ^ { ( t - \zeta ) \cos \eta } \mathrm { ~ a n d ~ } | \cosh ( a ( t - \zeta ) ) | ,$ $\zeta ) ) | \leq C e ^ { ( t - \zeta ) \cos \eta }$ . By (4.4) the exponential in (4.9) has modulus at most $\exp \{ - 2 c q ( 1 +$ $t ^ { - 1 } ) \} \leq 1$ . Consequently any derivative of total order $j \le 2$ in the spatial variables $( r , z _ { 2 } , d , q )$ taken in the integrand with t held fixed, has modulus at most

$$
C m ( t ) ^ { 2 - j } ( 1 + t ) e ^ { - t / 2 } ,\tag{4.11}
$$

which is integrable on $( 0 , \infty )$ uniformly in all spatial variables in the regional cone. There is one moving-endpoint term to check. Writing $K ( t ; \zeta , q )$ for the integrand of (4.9) including the leading factor $^ { a , }$ one has $K ( r ; \zeta , q ) = 0$ , because the first hyperbolic factor vanishes at $t = r ;$ so the first r-derivative of $G _ { a }$ has no endpoint contribution, and among the second derivatives only

$$
\partial _ { r } ^ { 2 } G _ { a } = \int _ { r } ^ { \infty } \partial _ { r } ^ { 2 } K ( t ; \zeta , q ) d t - \partial _ { r } K ( r ; \zeta , q )\tag{4.12}
$$

acquires an endpoint term, the other first derivatives of K still containing the vanishing factor. Moreover

$$
| \partial _ { r } K ( r ; \zeta , q ) | \leq C m ( r ) ( 1 + r ) e ^ { - r / 2 } ,\tag{4.13}
$$

because for small $r$ the factor $p ( a r ) / \sinh ^ { 2 } ( a r ) { \mathrm { ~ i s ~ } } \mathcal { O } ( r ^ { - 1 } )$ while the two remaining hyperbolic factors are $\mathcal { O } ( r ^ { 2 } )$ , and the exponential has modulus at most one. Dominated convergence and (4.13) now show that $G _ { a } ,$ , its gradient and its Hessian have continuous limits as $r \downarrow 0 ,$ uniformly also as $q \downarrow 0$ , and that all these derivatives are bounded on the whole regional cone. This is the step that includes the full diagonal. The same majorant with $j = 0$ bounds $| G _ { a } |$ itself by a constant, uniformly on the regional cone.

The four-expert background $F _ { 4 }$ of (2.13), continued to $F _ { 4 } ( a z _ { 1 } , a z _ { 2 } , a z _ { 3 } )$ , has the same property. For $r \leq 1$ its only nonanalytic contribution is $\Lambda ( a r ) \sinh ( a r ) \sinh ( a z _ { 2 } ) \sinh ( a z _ { 3 } )$ and since $\Lambda ( z ) = \log ( 2 / z ) + \mathcal { O } ( z ^ { 2 } )$ and $| z _ { 2 } | , | z _ { 3 } | \leq r ,$ its derivatives of order $j \le 2$ are $\mathcal { O } ( r ^ { 3 - j } ( 1 + | \log r | ) )$ , so its Hessian tends to zero; the remaining terms are analytic at the origin. For $r \geq 1$ we use the mode expansion of Lemma 3.8: the finitely many modes with $m \leq 1$ have nonnegative exponents, for $m \geq 2$ one has $X _ { m , \epsilon } ^ { ( 4 ) } \geq 2 ( m - 1 ) r$ , the coeficients after two spatial derivatives grow at most polynomially in $m .$ , and $\vert e ^ { - a X _ { m , \epsilon } ^ { ( 4 ) } } \vert \leq e ^ { - X _ { m , \epsilon } ^ { ( 4 ) } / 2 }$ because cos $\begin{array} { r } { \eta \geq \frac { 1 } { 2 } . } \end{array}$ ; the diferentiated series is therefore bounded uniformly for $r \geq 1$ . The undiferentiated series is bounded there too, its terms being $\mathcal { O } ( m ^ { - 1 } ) e ^ { - X _ { m , \epsilon } ^ { ( 4 ) } / 2 }$ , while on $r \leq 1$ the expression $F _ { 4 } ( a z )$ is continuous on a compact set of $( \eta , z )$ . Since $f _ { s } ( x ) - s x _ { 1 } = k ^ { - 1 } [ F _ { 4 } ( a z ) + G _ { a } ]$ in these regions, this proves (4.8) there.

We next treat Region III. Set $L = a _ { 4 }$ , with $0 \leq a _ { 1 } , a _ { 2 } , a _ { 3 } \leq L$ . The trace is analytic at zero, since its quadrature (2.8) may be written as

$$
e ( z ) = \cosh z \Bigl [ e ( 0 ) - 3 \int _ { 0 } ^ { z } I ( t ) \mathrm { s e c h } ^ { 2 } t \mathrm { c s c h } ^ { 5 } t d t \Bigr ] ,\tag{4.14}
$$

whose integrand is analytic at zero with value ${ \frac { 1 } { 5 } } .$ . From the coeficients (3.17), the only nonanalytic terms of the Region III expression (3.15) are constant multiples of

$$
\Lambda ( a L ) \sinh ( a L ) \sigma _ { 2 } ( a a _ { 1 } , a a _ { 2 } , a a _ { 3 } ) \qquad \mathrm { a n d } \qquad \Lambda ( a L ) \cosh ( a L ) \sigma _ { 3 } ( a a _ { 1 } , a a _ { 2 } , a a _ { 3 } ) ,
$$

whose derivatives of order $j \le 2$ are $\mathcal { O } ( L ^ { 3 - j } ( 1 + | \log L | ) )$ as $L \downarrow 0$ , uniformly over all directions in the cone and all $\eta .$ Thus the function and its first two derivatives have direction-independent limits at the origin within the region. For $L \geq 1$ we use the expansion of Proposition 3.13: the modes with $m \le 2$ have constant prefactors and, whenever their coeficients are nonzero, nonnegative exponents; for $m \geq 3 , X _ { m , \epsilon } \geq ( 2 m - 3 ) L$ , the coeficients after two spatial derivatives grow only polynomially in m with at most one further factor $L ,$ and $| e ^ { - a D _ { m , \epsilon } } | \leq e ^ { - ( 2 m - 3 ) L / 2 }$ bounds the diferentiated tail uniformly. The same bounds without derivatives, with the continuity of the expression on $L \leq 1$ , give (4.8) in Region III, where $f _ { s } ( x ) - s x _ { 1 } \ \mathrm { i s } \ k ^ { - 1 }$ times the continued expression.

It remains to match the continued regional formulas. All one-sided spatial jets through order two, including the collision limits just constructed, are holomorphic functions of $s \in \Sigma$ by the locally uniform integral bounds, the local analytic and logarithmic formulas, and the convergent mode expansions away from zero. At every regional interface and at every permutation wall the corresponding one-sided jets agree for all positive real $s ,$ because the real stationary solution $u$ is globally $C ^ { 2 }$ [6, Theorem 2.1]; the identity theorem extends these equalities to all of $\Sigma .$ , and the same argument applies to the limiting jets at the collision set. The gluing lemma of [6, Lemma 4.3], applied to the real and imaginary parts, now gives $f _ { s } \in C ^ { 2 } ( \mathbb { R } ^ { 5 } )$ for every $s \in \Sigma$ . The regional bounds, the fixed linear coordinate changes and the scaling $f _ { \rho e ^ { i \eta } } ( x ) = f _ { e ^ { i \eta } } ( \rho x )$ prove (4.6); and permutation symmetry with translation equivariance of the stationary solution give $\boldsymbol { \nabla } \boldsymbol { u } ( 0 ) = \frac { 1 } { 5 } \mathbb { 1 }$ , so (4.7) follows by analytic continuation. The bound (4.8) for general s follows from the case $| s | = 1$ by the scaling noted at the start. This completes the proof of the five properties.

We now invert the transform on an absolutely convergent contour. Use the principal square root and fix $\beta = 3 \pi / 5$ , so that $\pi / 2 < \beta < 2 \pi / 3$ ; let $\gamma$ be the Hankel contour consisting of the ray arg $\lambda = - \beta ,$ oriented from infinity to zero, followed by the ray arg $\lambda = \beta$ , oriented from zero to infinity. Define

$$
R ( \lambda , x ) = \lambda ^ { - 3 / 2 } \big [ f _ { \sqrt { \lambda } } ( x ) - u _ { 0 } - \sqrt { \lambda } \bar { x } \big ] ,\tag{4.15}
$$

which is holomorphic for $| \arg \lambda | < 2 \pi / 3$ . Taylor’s theorem in the real variable $x ,$ together with (4.6) and (4.7), gives on this sector

$$
| R ( \lambda , x ) | \leq C | x | ^ { 2 } | \lambda | ^ { - 1 / 2 } , \quad | D _ { x } R ( \lambda , x ) | \leq C | x | | \lambda | ^ { - 1 / 2 } , \quad \| D _ { x } ^ { 2 } R ( \lambda , x ) \| \leq C | \lambda | ^ { - 1 / 2 } ,\tag{4.16}
$$

both near zero and at infinity. Now define

$$
W ( \tau , x ) = \frac { 2 u _ { 0 } } { \sqrt { \pi } } \sqrt { \tau } + \bar { x } + \frac { 1 } { 2 \pi i } \int _ { \gamma } e ^ { \lambda \tau } R ( \lambda , x ) d \lambda .\tag{4.17}
$$

Along $\gamma$ one has $| e ^ { \lambda \tau } | = e ^ { - c _ { \beta } \tau | \lambda | }$ with $c _ { \beta } = - \cos \beta > 0$ , so (4.16) supplies the integrable majorant $C _ { K } e ^ { - c _ { \beta } \tau \rho } \rho ^ { - 1 / 2 } d \rho$ for (4.17) and for its first two spatial derivatives, locally uniformly in x. Diferentiating under the integral sign is therefore justified, at every point of $\mathbb { R } ^ { 5 }$ including the collision points, and $W ( \tau , \cdot )$ is globally $C ^ { 2 }$ , with spatial derivatives through order two jointly continuous in $( \tau , x )$

To identify W with $U ,$ , fix $\mu > 0$ . Absolute convergence and Fubini give

$$
\int _ { 0 } ^ { \infty } e ^ { - \mu \tau } \Bigl [ { \frac { 1 } { 2 \pi i } } \int _ { \gamma } e ^ { \lambda \tau } R ( \lambda , x ) d \lambda \Bigr ] d \tau = { \frac { 1 } { 2 \pi i } } \int _ { \gamma } { \frac { R ( \lambda , x ) } { \mu - \lambda } } d \lambda = R ( \mu , x ) ,\tag{4.18}
$$

the last equality being Cauchy’s formula in the sector bounded by $\gamma ;$ the small circular arc contributes $\mathcal { O } ( \varepsilon ^ { 1 / 2 } )$ and the large one $\mathcal { O } ( M ^ { - 1 / 2 } )$ by (4.16), so both disappear, and the stated orientation together with the denominator $\mu - \lambda$ gives the positive sign. With the elementary transforms of $\sqrt { \tau }$ and of a constant, (4.18) yields

$$
\int _ { 0 } ^ { \infty } e ^ { - \mu \tau } W ( \tau , x ) d \tau = u _ { 0 } \mu ^ { - 3 / 2 } + \bar { x } \mu ^ { - 1 } + R ( \mu , x ) = \mu ^ { - 3 / 2 } u ( \sqrt { \mu } x ) ,\tag{4.19}
$$

which is the transform of U by Proposition $4 . 2 ( \mathrm { i v } )$ . Both Laplace integrals converge absolutely for every $\mu > 0 \colon$ : that of U by Proposition $4 . 2 ( \mathrm { i v } )$ , and that of W because the integral in (4.17) is bounded by $C | x | ^ { 2 } \tau ^ { - 1 / 2 }$ , by the majorant above. Since $U$ and $W$ are also continuous $\tau > 0$ , uniqueness of the Laplace transform gives $W = U$ . In particular the Hessian is represented by

$$
D _ { x } ^ { 2 } U ( \tau , x ) = \frac { 1 } { 2 \pi i } \int _ { \gamma } e ^ { \lambda \tau } \lambda ^ { - 3 / 2 } D _ { x } ^ { 2 } f _ { \sqrt \lambda } ( x ) d \lambda ,\tag{4.20}
$$

whose integrand is bounded by $C e ^ { - c _ { \beta } \tau | \lambda | } | \lambda | ^ { - 1 / 2 }$ uniformly in $x \in \mathbb { R } ^ { 5 }$ . This proves the continuity of the Hessian across the entire collision set and the bound (a). Time diferentiation multiplies the integrand by powers of λ, and the same argument applies to every number of time derivatives on compact subsets of $( 0 , \infty )$ ; the time derivatives of the explicit $\sqrt { \tau }$ term are harmless there.

For (b), (4.20) and Fubini give, exactly as in (4.18),

$$
\int _ { 0 } ^ { \infty } e ^ { - \mu \tau } D _ { x } ^ { 2 } U ( \tau , x ) d \tau = \mu ^ { - 3 / 2 } D _ { x } ^ { 2 } f _ { \sqrt { \mu } } ( x ) = \mu ^ { - 1 / 2 } ( D ^ { 2 } u ) ( \sqrt { \mu } x )
$$

at every $x ,$ which is (4.1); and the two regional expressions are restrictions of the single $C ^ { 2 }$ function $U$ to their domains, both of which contain the points of the interface outside ${ \mathcal { C } } ,$ where each converges with its derivatives by Lemma 4.1, so their jets agree there.

For (c), permutation symmetry and translation equivariance give $\begin{array} { r } { D ^ { 2 } u ( 0 ) = A ( I - \frac { 1 } { 5 } \mathbb { 1 } \mathbb { 1 } ^ { T } ) } \end{array}$ and the stationary equation at zero, in which the maximum is attained by any binary control with two ones, gives $\begin{array} { r } { u _ { 0 } = \frac { 1 } { 2 } A ( 2 - \frac { 4 } { 5 } ) = \frac { 3 } { 5 } A } \end{array}$ , i.e. $\begin{array} { r } { A = \frac { 5 } { 3 } u _ { 0 } ; } \end{array}$ this is the expansion [6, equation (2.18)]. Since $D _ { x } ^ { 2 } f _ { \sqrt { \lambda } } ( 0 ) = \lambda D ^ { 2 } u ( 0 )$ , formula (4.20) and the inverse transform ${ \mathcal L } ^ { - 1 } [ \lambda ^ { - 1 / 2 } ] ( \tau ) ~ = ~ ( \pi \tau ) ^ { - 1 / 2 }$ give $\begin{array} { r } { D _ { x } ^ { 2 } U ( \tau , 0 ) \ = \ \frac { 5 } { 3 } u _ { 0 } ( \pi \tau ) ^ { - 1 / 2 } ( I - \frac { 1 } { 5 } \Im { \Im \Im } T ) } \end{array}$ , which is (4.2) at $c = 0 { \mathrm { ; } }$ ; translation equivariance $U ( \tau , x + c \mathbb { 1 } ) = U ( \tau , x ) + c$ moves it along the diagonal, and $\mathbf { v } ^ { T } ( I - \textstyle { \frac { 1 } { 5 } } \mathbb { 1 } \mathbb { 1 } ^ { T } ) \mathbf { v } = m ( 5 - m ) / 5$ for a binary v with m ones.

For (d), fix x with sorted coordinates, so that $\varphi ( x ) = x _ { 1 }$ , and fix $\tau > 0$ . The integrand $e ^ { \lambda \tau } R ( \lambda , x )$ is holomorphic on the sector $| \arg \lambda | < 2 \pi / 3$ and $R = \mathcal { O } ( | \lambda | ^ { - 1 / 2 } )$ at the origin, so, exactly as in (4.18), the contour $\gamma$ in (4.17) may be replaced by the contour $\gamma _ { \tau }$ that follows the ray arg $\lambda = - \beta$ from infinity to $\tau ^ { - 1 } e ^ { - i \dot { \beta } }$ , the arc $| \lambda | = \tau ^ { - 1 }$ through the positive real axis to $\tau ^ { - 1 } e ^ { i \beta }$ , and the ray arg $\lambda = \beta$ to infinity. On $\gamma _ { \tau }$ the origin is avoided, and we may split

$$
R ( \lambda , x ) = \lambda ^ { - 1 } ( x _ { 1 } - \bar { x } ) + \lambda ^ { - 3 / 2 } \big [ f _ { \sqrt { \lambda } } ( x ) - \sqrt { \lambda } x _ { 1 } - u _ { 0 } \big ] .
$$

Closing $\gamma _ { \tau }$ by a large arc through the negative real axis, on which $e ^ { \lambda \tau }$ decays, and using the residue at the origin, $\begin{array} { r } { \frac { 1 } { 2 \pi i } \int _ { \gamma _ { \tau } } e ^ { \lambda \tau } \lambda ^ { - 1 } d \lambda = 1 } \end{array}$ , so the first term contributes exactly $x _ { 1 } - \bar { x }$ to $W$ and cancels the x¯ of (4.17) against $x _ { 1 }$ . In the second term the bracket is bounded by $C + u _ { 0 }$ by (4.8). On the two rays, where $| \lambda | \geq \tau ^ { - 1 }$ and $| e ^ { \lambda \tau } | = e ^ { - c _ { \beta } \tau | \lambda | }$ , the substitution $| \lambda | = \sigma / \tau$ bounds its contribution by $\pi ^ { - 1 } ( \dot { C } + u _ { 0 } ) \sqrt { \tau } \int _ { 1 } ^ { \infty } e ^ { - c _ { \beta } \dot { \sigma } } \sigma ^ { - 3 / 2 } d \sigma$ , and on the arc, of length at most $2 \pi / \tau$ , where $| e ^ { \lambda \tau } | \le e$ and $| \lambda | ^ { - 3 / 2 } = \tau ^ { 3 / 2 }$ , it is at most $e ( C + u _ { 0 } ) \sqrt { \tau }$ . Hence

$$
| U ( \tau , x ) - x _ { 1 } | \leq \Bigl ( \frac { 2 u _ { 0 } } { \sqrt { \pi } } + C ^ { \prime } \Bigr ) \sqrt { \tau }
$$

uniformly in x. This is (d).

Two consequences are used repeatedly below. First, since U is symmetric under permutations of the coordinates and $C ^ { 1 }$ , it satisfies the permutation face conditions of Section 3.1 on every wall. Second, the linear equation of Proposition $4 . 2 ( \mathrm { i i } )$ is an identity between functions that are continuous on $( 0 , \infty ) \times \mathbb { R } ^ { 5 }$ , so it holds on the closed ordered sector, C included:

$$
\begin{array} { r } { U _ { \tau } = \frac { 1 } { 2 } D _ { \mathbf { v } _ { \ast } } ^ { 2 } U \qquad \mathrm { o n } ~ ( 0 , \infty ) \times \{ x _ { 1 } \geq \cdot \cdot \cdot \geq x _ { 5 } \} , } \end{array}\tag{4.21}
$$

and on the image $\sigma ( S )$ of the sector under a permutation $\sigma$ the same holds with $\mathbf { v } _ { * }$ replaced by $\sigma { \mathbf { v } _ { * } }$

## 4.3 Transfer of curvature identities and the COMB equality set

The transform principle turns every equality between curvatures of the stationary solution into the same equality for the finite-horizon solution, provided the set on which it holds is a cone. This is immediate but useful, because the relevant sets, the three regions, are cones.

Proposition 4.5. Let $\Omega \subseteq \mathbb { R } ^ { 5 }$ be a cone, $\Omega = \{ \ell x : \ell > 0 , \ x \in \Omega \}$ , and let $\mathbf { v } , \mathbf { v } ^ { \prime } \in \{ 0 , 1 \} ^ { 5 }$ satisfy $D _ { \mathbf { v } } ^ { 2 } u = D _ { \mathbf { v } ^ { \prime } } ^ { 2 } u$ at every $x \in \Omega$ . Then $D _ { \mathbf { v } } ^ { 2 } U ( \tau , x ) = D _ { \mathbf { v ^ { \prime } } } ^ { 2 } U ( \tau , x )$ for every $\tau > 0$ and every $x \in \Omega$

Proof. Fix $x \in \Omega$ . Since Ω is a cone, $\sqrt { \lambda } x \in \Omega$ for every $\lambda > 0$ , so the right side of (4.1), which holds at every x by Theorem $4 . 3 ( \mathrm { b } )$ , is the same for v and $\mathbf { v } ^ { \prime }$ . Hence the continuous functions $\tau \mapsto D _ { \mathbf { v } } ^ { 2 } U ( \tau , x )$ and $\tau \mapsto D _ { \mathbf { v ^ { \prime } } } ^ { 2 } U ( \tau , x )$ have the same Laplace transform, and therefore coincide. □

Corollary 4.6. For every $\tau > 0$ and every state of the closed sector, the direction $\mathbf { v } _ { * } =$ $( 1 , 0 , 1 , 0 , 0 )$ shares the value of its curvature $D _ { \mathbf { v } _ { * } } ^ { 2 } U$ with $( 1 , 0 , 0 , 1 , 1 )$ in Region I, with $( 1 , 0 , 0 , 1 , 0 )$ in Region II, and with $( 1 , 0 , 0 , 0 , 1 )$ and $( 1 , 0 , 0 , 1 , 0 )$ in Region III. Conversely, if a control attains the Hamiltonian maximum for U at every state of the sector and every $\tau > 0$ , then it attains the Hamiltonian maximum for u throughout the sector, so $\mathbf { v } _ { * }$ is the only control with this property.

Proof. Each of the three regions is a cone, and on each the listed directions have the same curvature as $\mathbf { v } _ { * }$ for the stationary solution [6, Remark 2.3]; apply Proposition 4.5. For the converse, if $D _ { \mathbf { v } ^ { \prime } } ^ { 2 } U \geq D _ { \mathbf { v } } ^ { 2 } U$ for all v at every $( \tau , x )$ , then integrating against $e ^ { - \lambda \tau }$ and using (4.1) gives $D _ { \mathbf { v } ^ { \prime } } ^ { 2 } u \geq D _ { \mathbf { v } } ^ { 2 } u$ at every ${ \sqrt { \lambda } } x .$ , hence throughout the sector, and $\mathbf { v } _ { * }$ is the only control optimal throughout the sector for the stationary problem [6]. □

The COMB strategy of Gravin, Peres and Sivan [14] is, in decreasing rank order, $\mathbf { v } _ { C } =$ $( 1 , 0 , 1 , 0 , 1 )$ . For the stationary problem it attains the Hamiltonian maximum exactly on $\{ x _ { 1 } = x _ { 2 } \} \cap \{ x _ { 3 } = x _ { 4 } \}$ [6, Theorem 2.2]. For the finite-horizon problem the equality on this set is a consequence of symmetry alone; the strict comparison away from it is part of Theorems 4.10 and 4.11 below.

Proposition 4.7. Let $\Delta$ be the curvature gap (2.23). Then, for every $\tau > 0$ and every state of the ordered sector,

$$
\Delta ( \tau , x ) = 0 \qquad w h e n e v e r \quad x _ { 1 } = x _ { 2 } \quad a n d \ x _ { 3 } = x _ { 4 } .\tag{4.22}
$$

Moreover $\Delta ( \tau , x ) = \tau ^ { - 1 / 2 } \Delta ( 1 , x / \sqrt { \tau } )$ , so $\Delta > 0$ of the set (4.22) for all τ if and only if it holds for $\tau = 1$

Proof. Let $P$ be the matrix of the permutation that exchanges x with x and $x _ { 3 }$ with $x _ { 4 }$ It fixes every point of $S = \{ x _ { 1 } = x _ { 2 } , \ x _ { 3 } = x _ { 4 } \}$ , and $U ( \tau , P x ) = U ( \tau , x )$ for every x, so diferentiating twice at a point of S gives $\nabla ^ { 2 } U = P ^ { T } \nabla ^ { 2 } U P$ there, and hence $D _ { P \mathbf { v } } ^ { 2 } U = D _ { \mathbf { v } } ^ { 2 } U$ on S for every v. Now $P \mathbf { v } _ { * } = ( 0 , 1 , 0 , 1 , 0 )$ is COMB, so $D _ { \mathbf { v } _ { * } } ^ { 2 } U = D _ { \mathbf { v } _ { C } } ^ { 2 } U$ on S, which is (4.22). The scaling identity follows from Proposition $4 . 2 ( \mathrm { i i i } )$ by diferentiating twice: $D _ { \mathbf { v } } ^ { 2 } U ( \ell ^ { 2 } \tau , \ell x ) = \ell ^ { - 1 } D _ { \mathbf { v } } ^ { 2 } U ( \tau , x )$ ; take $\ell = 1 / \sqrt { \tau }$ . Because S is a cone, $x \not \in S$ if and only if $x / \sqrt { \tau } \notin S$ □

Remark 4.8. In numerical evaluations of S the gap is positive but extremely small when the coordinates are spread out relative to $\sqrt { \tau }$ . By the scaling of Proposition 4.7 the whole function is determined by its profile at $\tau = 1$ , and that profile decays roughly like $e ^ { - c | x | ^ { 2 } }$ . For instance, in 60-digit arithmetic, at $x = ( 1 . 2 , 0 . 4 , 0 . 3 , 0 . 2 5 , 0 )$ one finds $\Delta ( 1 , x ) = 2 . 5 4 4 \times 1 0 ^ { - 2 }$ at $x = ( 2 , 1 . 4 , 0 . 9 , 0 . 3 5 , 0 )$ $1 . 3 2 5 8 \times 1 0 ^ { - 5 } ;$ ; and at $\boldsymbol { x } = ( 2 . 9 8 , 2 . 6 3 , 2 . 0 4 , 1 . 4 9 , 0 ) , 2 . 0 6 9 6 \times 1 0 ^ { - 1 1 }$ Correspondingly $\Delta ( \tau , x )$ at fixed x increases from below double precision at small τ to a maximum near $\tau \sim | x | ^ { 2 }$ and then decays faster than $\tau ^ { - 1 / 2 }$ : by the scaling, $\sqrt { \tau } \Delta ( \tau , x ) =$ $\Delta ( 1 , x / \sqrt { \tau } )  \Delta ( 1 , 0 ) = 0$ , since $0 \in S$ and $\nabla ^ { 2 } U$ is continuous. Any numerical test of strict positivity must be read with this in mind: at large $| x | / { \sqrt { \tau } }$ a genuine strict inequality is indistinguishable from equality in floating point. The integrated statement is however sharp and easy to check, because (4.1) gives

$$
\int _ { 0 } ^ { \infty } e ^ { - \tau } \Delta ( \tau , x ) d \tau = D _ { \mathbf { v } _ { \ast } } ^ { 2 } u ( x ) - D _ { \mathbf { v } _ { C } } ^ { 2 } u ( x ) ,
$$

the right side being the stationary gap, which is bounded below on compacta away from S.

Remark 4.9. Inversion also transfers inequalities when the stationary curvature gap has a mode expansion with nonnegative data. Indeed, by (4.1), if on a conic region

$$
D _ { \mathbf { v } _ { \ast } } ^ { 2 } u - D _ { \mathbf { v } } ^ { 2 } u = \sum _ { \iota } C _ { \iota } P _ { \iota } ( x ) e ^ { - X _ { \iota } ( x ) }
$$

with $C _ { \iota } \geq 0 , X _ { \iota } \geq 0$ linear, $P _ { \iota } \geq 0$ homogeneous of degree $p _ { \iota } \in \{ 0 , 1 \}$ , and $X _ { \iota } ( x ) > 0$ whenever $p _ { \iota } = 1$ , then

$$
D _ { \mathbf { v } _ { * } } ^ { 2 } U - D _ { \mathbf { v } } ^ { 2 } U = \sum _ { \iota } C _ { \iota } P _ { \iota } ( \boldsymbol { x } ) \mathcal { H } _ { p _ { \iota } + 2 } \bigl ( X _ { \iota } ( \boldsymbol { x } ) , \tau \bigr ) \geq 0 ,
$$

because $\mathcal { H } _ { 2 } = \mathcal { G } > 0$ and $\begin{array} { r } { \mathcal { H } _ { 3 } = \frac { X } { 2 \tau } \mathcal { G } \geq 0 } \end{array}$ for $X \geq 0$ . Degrees $p \geq 2$ are excluded, since $\mathcal { H } _ { 4 }$ changes sign. In Region III the prefactors are exactly 1 and $^ { a _ { 4 } , }$ of degrees 0 and 1, but their coeficients have mixed signs. The corner reduction and the Gaussian certificates of Appendix A handle these cancellations.

## 4.4 The Hamiltonian inequalities

So far, we have proved that U solves the linear problem (3.1) obtained by freezing the direction $\mathbf { v } _ { * } ,$ , that its Laplace transform in $\tau$ is $\lambda ^ { - 3 / 2 } u ( \sqrt { \lambda } x )$ , that it attains the terminal data, and that it is $C ^ { 2 }$ in space. The additional verification, as in the four-expert work of Bayraktar, Ekren and Zhang [3, Section 4.2], concerns the Hamiltonian inequality

$$
D _ { \mathbf { v } _ { * } } ^ { 2 } U ( \tau , x ) \geq D _ { \mathbf { v } } ^ { 2 } U ( \tau , x ) \qquad { \mathrm { f o r ~ a l l ~ } } \mathbf { v } \in \{ 0 , 1 \} ^ { 5 } ,\tag{4.23}
$$

which upgrades U from a solution of the frozen linear equation to the solution of (2.1). It is proved in the following two regional forms, whose proofs occupy Appendices A and B.

Theorem 4.10. For every $\tau > 0$ and every point of the closed Region III with $a _ { 4 } > 0$

$$
D _ { \mathbf { v } } ^ { 2 } U ( \tau , x ) \leq D _ { \mathbf { v } _ { \ast } } ^ { 2 } U ( \tau , x ) = 2 U _ { \tau } ( \tau , x ) , \qquad \mathbf { v } \in \{ 0 , 1 \} ^ { 5 } .
$$

For COMB the inequality is strict unless $a _ { 1 } = a _ { 2 } = a _ { 3 } = 0$ , that $i s ,$ unless $x _ { 1 } = x _ { 2 }$ and $x _ { 3 } = x _ { 4 } = x _ { 5 }$

Theorem 4.11. For every $\tau > 0$ and every state of Regions I and II with $z _ { 1 } > 0 , D _ { \mathbf { v } } ^ { 2 } U ( \tau , x ) \leq$ $D _ { \mathbf { v } _ { * } } ^ { 2 } U ( \tau , x )$ for all $\mathbf { v } \in \{ 0 , 1 \} ^ { 5 }$ . The COMB comparison is strict when $z _ { 2 } > 0$ and is an equality when $z _ { 2 } = 0$ , that is, when $x _ { 1 } = x _ { 2 }$ and $x _ { 3 } = x _ { 4 }$

The argument has three parts, which we outline here. The first part is to transform; by (4.1), for each binary $\mathbf { v } ,$

$$
F _ { \mathbf { v } } ( \lambda , x ) : = 2 \lambda { \widehat { U } } ( \lambda , x ) - 2 \varphi ( x ) - D _ { \mathbf { v } } ^ { 2 } { \widehat { U } } ( \lambda , x ) = \lambda ^ { - 1 / 2 } { \big ( } D _ { \mathbf { v _ { * } } } ^ { 2 } u - D _ { \mathbf { v } } ^ { 2 } u { \big ) } ( { \sqrt { \lambda } } x ) ,\tag{4.24}
$$

which is the Laplace transform of $2 U _ { \tau } - D _ { \mathbf { v } } ^ { 2 } U$ . We write CM for the class of functions of $\lambda > 0$ that are Laplace transforms of nonnegative measures on $[ 0 , \infty )$ ; by Bernstein’s theorem CM is the class of completely monotone functions, and it is closed under sums, products, pointwise limits, and integration against nonnegative weights. So (4.23) is equivalent to the statement that each transformed gap (4.24) lies in CM. That is what we certify. It is a statement about the stationary solution u, which is globally $C ^ { 2 } \ [ 6 ] ;$ the finite-horizon regularity never enters. The passage back from complete monotonicity to the inequality is the following elementary fact, which is where the appendices end. Its second statement gives the strict comparisons for COMB: a positive transform alone shows only that a gap does not vanish identically in $\tau ,$ and strict positivity at every $\tau > 0$ needs a strictly positive summand whose weight charges every neighborhood of $\tau = 0$

Lemma 4.12. Let h be continuous on $( 0 , \infty )$ with $| h ( \tau ) | \leq C ( 1 + \tau ^ { - 1 / 2 } + \tau ^ { N } )$ for some $N _ { \cdot }$ and suppose that its Laplace transform $\begin{array} { r } { \widehat { h } ( \lambda ) = \int _ { 0 } ^ { \infty } e ^ { - \lambda \tau } h ( \tau ) } \end{array}$ dτ is completely monotone. Then $h \geq 0$ . If moreover $\widehat { h } - \widehat { \kappa } \widehat { g }$ is completely monotone, where $\widehat { \kappa }$ is the Laplace transform of a nonnegative measure κ on $[ 0 , \infty )$ with $\widehat { \kappa } ( \lambda ) \geq e ^ { - A \sqrt { \lambda } }$ for some constant A and all suficiently large $\lambda ,$ and $\widehat g$ is the Laplace transform of a function g that is continuous and strictly positive on $( 0 , \infty )$ , then $h > 0$ on $( 0 , \infty )$

Proof. By Bernstein’s theorem the transform of $h$ is the transform of a nonnegative measure ν on $[ 0 , \infty )$ , and the growth bound makes h dτ a locally finite signed measure with a Laplace transform for every $\lambda > 0$ . Uniqueness of Laplace transforms of such measures gives $h d \tau = \nu$ on $( 0 , \infty )$ , so $h \geq 0$ almost everywhere, hence everywhere by continuity.

For the second statement we first note that κ charges every interval $[ 0 , t )$ with $t > 0$ Indeed, if $\kappa ( [ 0 , t ) ) = 0$ , then $\begin{array} { r } { \widehat { \kappa } ( \lambda ) = \int _ { [ t , \infty ) } e ^ { - \lambda s } \kappa ( d s ) \leq e ^ { - ( \lambda - 1 ) t } \widehat { \kappa } ( 1 ) } \end{array}$ for $\lambda \geq 1$ , which is incompatible with ${ \widehat { \kappa } } ( \lambda ) \geq e ^ { - A { \sqrt { \lambda } } }$ as $\lambda  \infty$ . Next, $\widehat { \kappa } \widehat { g }$ is the transform of the function κ ∗ $\begin{array} { r } { g ( \tau ) = \int _ { [ 0 , \tau ) } g ( \tau - s ) \kappa ( d s ) } \end{array}$ , and $\widehat { h } - \widehat { \kappa } \widehat { g }$ is the transform of a nonnegative measure, so the uniqueness argument above gives $h \geq \kappa * g$ almost everywhere. Now fix $\tau _ { 0 } > 0$ and let $m > 0$ be the minimum of $g$ on $[ \tau _ { 0 } / 4 , 5 \tau _ { 0 } / 4 ]$ . If $| \tau - \tau _ { 0 } | \le \tau _ { 0 } / 4$ and $0 \le s < \tau _ { 0 } / 2$ , then $\tau - s \in [ \tau _ { 0 } / 4 , 5 \tau _ { 0 } / 4 ]$ , hence

$$
\kappa \ast g ( \tau ) \geq m \kappa \big ( [ 0 , \tau _ { 0 } / 2 ) \big ) > 0 .
$$

Thus h is bounded below by a positive constant almost everywhere near $\tau _ { 0 }$ , and $h ( \tau _ { 0 } ) > 0$ by continuity. □

Every membership in CM that Appendices A and B establish is obtained in representation form: the function is exhibited as the Laplace transform of a nonnegative measure, an exittime law, a Gamma density or a certified profile, or as a product, an integral or a pointwise limit of such transforms. Bernstein’s theorem is therefore used only for convenience, in Lemma 4.12 and in the closure of CM under limits; a treatment that carries the measures themselves needs only the uniqueness theorem for the Laplace transform and its continuity theorem, which is the route the Lean formalization described at the end of Section 4.4 takes. Appendices A and B prove that each transformed gap (4.24) lies in CM at every state of the ordered sector with $a _ { 4 } > 0$ , respectively $z _ { 1 } > 0$ , and Lemma 4.12 turns this into Theorems 4.10 and 4.11.

The second part is the reduction to finitely many scalar profiles. Both regions admit an exact interpolation identity that writes the transformed gap at a general state as a combination, with completely monotone weights, of its values at finitely many distinguished corner states, plus, in Region I, one explicitly positive Dirichlet kernel. In Region III (Appendix $\mathrm { A } )$ the interpolation is over a cube in $( a _ { 1 } , a _ { 2 } , a _ { 3 } )$ , and the 64 corner gaps are already functions of the single variable $a _ { 4 }$ , ten of which generate the rest. In Regions I and II (Appendix B) the interpolation is over a square in $\left( z _ { 2 } , | z _ { 3 } | \right)$ and leaves 64 corner gaps in four two-variable families; positive evolutions in the transverse variable $z _ { 4 }$ reduce those to a further 31 one-variable profiles. Each of the resulting 41 profiles is a Gaussian series with rational coeficients.

The third part certifies the scalar signs on the whole axis. Each profile has the form

$$
\Gamma ( \xi ) = { \textstyle { \frac { 1 } { 2 } } } \sum _ { k \equiv p ( 2 ) } \bigl [ A ( k ) + \xi C ( k ) \bigr ] e ^ { - k ^ { 2 } \xi } , \qquad \xi = \frac { L ^ { 2 } } { 4 \tau } ,\tag{4.25}
$$

with A, C even rational functions of k whose poles lie of the summation lattice. Positivity is proved by Poisson summation for $0 < \xi \leq 1 / 1 0 0$ , by first-mode domination for $\xi \ge 1$ , and by outward-rounded interval arithmetic on $[ 1 / 1 0 0 , 1 ]$ , with an explicit bound for the infinite tail. These are proofs over ranges, not tests at sample points. The proof of Theorem 2.1 in Section 4.5 then extends (4.23) from the two regional statements to every state, by the continuity of the Hessian.

The supplement [5] contains fourteen certificate scripts, run in dependency order by run\_certificates.py at the top level of the archive; a clean run takes a few minutes with SymPy 1.14 and mpmath 1.3, and prints

PASS: all 14 steps, 64 physical corner controls, and 1616 compact intervals.

The counts are 807 intervals for the Region III generators, 251 for nine of the ten critical initial profiles, 513 for the twenty further derivative profiles and 45 for the last boundary profile; the tenth critical profile, the endpoint datum of Appendix B.3, is certified by an exact polynomial argument and needs no cells, so 40 of the 41 profiles carry interval certificates. The runner re-derives every exact rational datum from scratch rather than reading it back, checks that the interval partitions have neither gaps nor overlaps, and checks that every stored margin is positive. A second program, witness/check.py, re-checks the same certificates from their stored witnesses using only the $\mathrm { P y }$ thon standard library: it rebuilds the sixty-four Region I/II corner expressions from the geometric construction, re-derives the 106 exact identities of the corner audit, and re-certifies all 1616 cells, recomputing every mode coeficient from its rational formula, proving the coeficient bound behind the truncated tail, and recomputing each Taylor bound and every exponential from its series rather than trusting a stored margin.

Finally, the proofs of Theorems 2.1 and 2.2 have been formalized in the Lean 4 proof assistant against Mathlib, using only Lean’s standard axioms and no hypotheses beyond those in the statements. The development is in the directory witness/lean of the supplement; it builds on the formalization of the companion paper [7], which the supplement includes and from which it takes the properties of the stationary solution. For each of the 1616 cells the supplement generates a Lean theorem stating that the truncated profile, with its actual rational coeficients written out, is positive on the cell whenever the truncated tail is bounded by the constant of Appendix A; the midpoint value, the derivative bounds and the top-derivative majorant are all proved from shared rational enclosures of $e ^ { - x }$ , and the compact-cell criterion itself is proved in Lean at every order the certificates use. On top of the cells, the development proves Theorems 2.1 and 2.2 for the contour representation (4.17) of $U ,$ including the regularity of Section 4.2, the reductions of Appendices A and B, and the smallscale and large-scale ranges of every profile, and it proves that this function is the function U defined in Theorem 2.1. Each clause of the two theorems is then restated for the explicit formula in the module PaperIndex, the file witness/lean/Fhcorner/PaperIndex.lean of the supplement, which Table 1 lists, so that every clause can be compared with its formal statement in one place. The rest of the paper is formalized only as far as these proofs use it. In particular Theorem 4.3(a), Corollary 4.6 and the statements of Theorem 4.3 about time derivatives other than $U _ { \tau }$ are proved here only, and the limit of $V ^ { M } ( 0 ) / \sqrt { M }$ stated after Theorem 2.1 is the cited theorem of Drenska and Kohn [13]. The detailed notes in the supplement record the derivations and the exact coeficient data behind Appendices A and B, and Appendix C reports the independent numerical checks, which are not part of the proof.

<table><tr><td>Clause</td><td>Content</td><td>Lean theorem in PaperIndex</td></tr><tr><td>Thm. 2.1</td><td>contour representation = formula</td><td>contour_eq_formula</td></tr><tr><td></td><td>absolute convergence of (2.19)</td><td>formula_summable_III</td></tr><tr><td></td><td>absolute convergence of (2.20)</td><td>formula_summable_I_II</td></tr><tr><td></td><td>integrability in (2.21)</td><td>formula_integrable_coll</td></tr><tr><td></td><td>derivatives of every order</td><td>formula_derivatives</td></tr><tr><td></td><td>sum differentiated term by term, Region III</td><td>formula_termwise_III</td></tr><tr><td></td><td>the same in Regions I and II</td><td>formula_termwise_I_II</td></tr><tr><td>(i)</td><td> $U ( \tau , \cdot ) \in C ^ { 2 } ( \mathbb { R } ^ { 5 } )$ </td><td>thm_2_1_i_C2</td></tr><tr><td></td><td> $\begin{array} { r } { U _ { \tau } = \frac { 1 } { 2 } \operatorname* { m a x } _ { \mathbf { v } } D _ { \mathbf { v } } ^ { 2 } U } \end{array}$ </td><td>thm_2_1_i_equation</td></tr><tr><td></td><td> $\boldsymbol { U } , \boldsymbol { U } _ { \ u { \tau } } , \boldsymbol { \nabla } \boldsymbol { U } , \boldsymbol { \nabla } ^ { 2 } \boldsymbol { U }$ </td><td>thm_2_1_i_continuous</td></tr><tr><td></td><td>joint continuity of</td><td></td></tr><tr><td></td><td>viscosity solution</td><td>thm_2_1_i_viscosity</td></tr><tr><td></td><td>limit as  $\tau \downarrow 0$ </td><td>thm_2_1_i_initial</td></tr><tr><td>(ii)</td><td> $| U - \varphi | \leq C \sqrt { \tau }$ </td><td>thm_2_1_i_terminal</td></tr><tr><td>(iii)</td><td>Hamiltonian maximum at  $\mathbf { v } _ { * }$ </td><td>thm_2_1_ii</td></tr><tr><td></td><td>parabolic scaling</td><td>thm_2_1_iii_scaling</td></tr><tr><td></td><td>Laplace transform</td><td>thm_2_1_iii_laplace</td></tr><tr><td>Thm. 2.2</td><td>value and Hessian on the diagonal</td><td>thm_2_1_iii_diagonal</td></tr></table>

Table 1: The Lean statements of Theorems 2.1 and 2.2, all about the explicit formula of Theorem 2.1. Each depends only on Lean’s standard axioms.

## 4.5 Proofs of the main theorems

Proof of Theorem 2.1. The convergence statements are Lemma $4 . 1 ( \mathrm { a } )$ and (b). For (i), the spatial regularity is Theorem 4.3 and the bound $| U - \varphi | \leq C \sqrt { \tau }$ is Theorem 4.3(d). At a state of Region III with $a _ { 4 } > 0$ the inequality (4.23) is Theorem 4.10, and at a state of Region I or II with $z _ { 1 } > 0$ it is Theorem 4.11. Every other state x of the ordered sector has $z _ { 1 } = 0$ since in Region III $0 \leq a _ { 1 } \leq a _ { 2 } \leq a _ { 3 } \leq a _ { 4 } = 0$ forces $y = 0$ . The states $x + t ( 4 , 3 , 2 , 1 , 0 )$ with $t > 0$ are strictly ordered and have $z _ { 1 } = 4 t / k > 0$ , so each of them lies in Region III with $a _ { 4 } > 0$ or in Region I or II with $z _ { 1 } > 0$ , and letting $t \downarrow 0$ gives (4.23) at x by the continuity of the Hessian. Thus (4.23) holds on the closed sector. Together with (4.21) this gives

$$
\begin{array} { r } { U _ { \tau } = \frac { 1 } { 2 } D _ { \mathbf { v } _ { \ast } } ^ { 2 } U = \frac { 1 } { 2 } \displaystyle \operatorname* { m a x } _ { \mathbf { v } \in \{ 0 , 1 \} ^ { 5 } } D _ { \mathbf { v } } ^ { 2 } U } \end{array}
$$

on the closed sector. Since U is invariant under permutations of the coordinates, which permute the binary controls, the same equation holds on the image of the sector under every permutation, that is, on $( 0 , \infty ) \times \mathbb { R } ^ { 5 }$ . Its right side is jointly continuous by Theorem 4.3, and hence so is $U _ { \tau }$ . Thus U is a classical solution of (2.1), and a classical solution is a viscosity solution. Statement (ii) is (4.23) together with (4.21). For (iii), the scaling and the transform identity are Proposition 4.2(iii) and (iv), the value at the origin is Proposition 4.2(i) with $u ( 0 ) = 4 5 \pi ^ { 2 } / ( 5 1 2 \sqrt { 2 } )$ [6, Theorem 2.1], and the Hessian at the origin is Theorem 4.3(c).

Proof of Theorem 2.2. By Theorem 2.1(ii), $\Delta \geq 0$ at every state of the ordered sector. Equality on the set (2.24) is Proposition 4.7. Strict positivity of that set is Theorem 4.10 in Region III, where the set (2.24) meets the region exactly in $\{ a _ { 1 } = a _ { 2 } = a _ { 3 } = 0 \}$ , because $y _ { 1 } = a _ { 1 } + a _ { 2 } + a _ { 3 }$ vanishes only when all three do, and Theorem 4.11 in Regions I and II, where $\begin{array} { r } { z _ { 2 } = \frac { 1 } { 2 } ( y _ { 1 } + y _ { 3 } ) } \end{array}$ vanishes exactly on (2.24); the states of Regions I and II with $z _ { 1 } = 0$ and the origin of Region III lie in $\mathcal { C } \subset \{ x _ { 1 } = x _ { 2 } , ~ x _ { 3 } = x _ { 4 } \}$ and are covered by the equality statement. □

## A The Region III certificate

Throughout the appendices $\mu = \sqrt { \lambda }$ denotes the square root of the Laplace variable; the constant $k = \sqrt { 2 }$ of Section 2 does not appear again. We write CM for the class of completely monotone functions of $\lambda > 0$ , as in Section 4.4. By (4.24), proving (4.23) means proving that each transformed gap lies in CM.

## A.1 Normalization and positive heat-exit interpolation

Recall $y _ { i } = k ( x _ { i } - x _ { i + 1 } )$ and the Region III coordinates $0 \leq a _ { 1 } \leq a _ { 2 } \leq a _ { 3 } \leq a _ { 4 } = : L$ of (2.3). For a control $\mathbf { v } \in \{ 0 , 1 \} ^ { 5 }$ put

$$
q _ { \mathbf { v } } = { \frac { 1 } { 3 } } { \left( \begin{array} { l l l l l } { 1 } & { - 1 } & { - 1 } & { - 1 } & { 2 } \\ { 1 } & { - 1 } & { - 1 } & { 2 } & { - 1 } \\ { 1 } & { - 1 } & { 2 } & { - 1 } & { - 1 } \\ { 1 } & { 2 } & { - 1 } & { - 1 } & { - 1 } \end{array} \right) } \ \mathbf { v } , \qquad { \mathrm { s o ~ t h a t } } \qquad D _ { \mathbf { v } } a = k q _ { \mathbf { v } } .\tag{A.1}
$$

Up to the identification of Section 2 there are sixteen controls, and we name each by its representative with first bit 0, as the supplement does. Since $u = x _ { 1 } + F / k$ and $x _ { 1 }$ is linear,

$$
D _ { \mathbf { v } _ { \ast } } ^ { 2 } u - D _ { \mathbf { v } } ^ { 2 } u = k G _ { \mathbf { v } } , \qquad G _ { \mathbf { v } } = F - q _ { \mathbf { v } } ^ { T } D _ { a } ^ { 2 } F q _ { \mathbf { v } } ,\tag{A.2}
$$

and ${ q _ { \mathbf { v } _ { * } } } = ( 0 , 0 , 1 , 0 )$ , so that $G _ { { \bf v } _ { * } } = F - \partial _ { a _ { 3 } } ^ { 2 } F = 0$ as it must be. Because $\partial _ { a _ { i } } ^ { 2 } { \cal F } = { \cal F }$ for $i = { 1 , 2 , 3 }$ and partial derivatives commute,

$$
\partial _ { a _ { i } } ^ { 2 } G _ { \mathbf { v } } = G _ { \mathbf { v } } , \qquad i = 1 , 2 , 3 .\tag{A.3}
$$

For $0 \leq b \leq r$ set

$$
{ \cal W } _ { 0 } ( \lambda ; b , r ) = \frac { \sinh ( ( r - b ) \mu ) } { \sinh ( r \mu ) } , \qquad { \cal W } _ { 1 } ( \lambda ; b , r ) = \frac { \sinh ( b \mu ) } { \sinh ( r \mu ) } .\tag{A.4}
$$

Both weights lie in CM. Indeed, by the product formula sinh $\begin{array} { r } { z = z \prod _ { n \geq 1 } ( 1 + z ^ { 2 } / ( \pi ^ { 2 } n ^ { 2 } ) ) } \end{array}$

$$
W _ { 1 } ( \lambda ; b , r ) = \frac { b } { r } \prod _ { n \geq 1 } \Bigl [ \theta + \frac { 1 - \theta } { 1 + r ^ { 2 } \lambda / ( \pi ^ { 2 } n ^ { 2 } ) } \Bigr ] , \qquad \theta = \frac { b ^ { 2 } } { r ^ { 2 } } \in [ 0 , 1 ] ,
$$

a limit of products of functions of the form $\theta + ( 1 - \theta ) / ( 1 + c \lambda )$ , each in CM, and $W _ { 0 } ( \lambda ; b , r ) = W _ { 1 } ( \lambda ; r - b , r )$ . Probabilistically, $W _ { i } = \mathbb { E } _ { b } [ e ^ { - \lambda \sigma } ; B _ { \sigma } = i r ]$ for a Brownian motion with generator $\partial _ { b b }$ started at b and its first exit time $\sigma$ from $( 0 , r )$ ; at the endpoints the corresponding measure has an atom at time zero.

Lemma A.1. Let $g _ { \lambda }$ solve $\partial _ { b _ { i } b _ { i } } g _ { \lambda } = \lambda g _ { \lambda } ~ f o r ~ i = 1 , 2 , 3$ on the cube $[ 0 , r ] ^ { 3 }$ . Then

$$
g _ { \lambda } ( b ) = \sum _ { \epsilon \in \{ 0 , 1 \} ^ { 3 } } g _ { \lambda } ( r \epsilon ) \prod _ { i = 1 } ^ { 3 } W _ { \epsilon _ { i } } ( \lambda ; b _ { i } , r ) .\tag{A.5}
$$

In particular, $i f$ every corner value $g _ { \lambda } ( r \epsilon )$ lies in CM, so does $g _ { \lambda } ( b )$ at every point of the cube.

Proof. In one variable the two-point boundary value problem $g ^ { \prime \prime } = \lambda g$ on $[ 0 , r ]$ has the unique solution $g ( b ) = g ( 0 ) W _ { 0 } + g ( r ) W _ { 1 }$ , since $W _ { 0 } , W _ { 1 }$ solve the equation, and $( W _ { 0 } , W _ { 1 } ) = ( 1 , 0 )$ at $b = 0$ and $( 0 , 1 )$ at $b = r$ . Applying this in each variable in turn gives (A.5). The last statement holds because CM is closed under products and nonnegative combinations.

By (A.3), the scaled gap $\widehat { \gamma } _ { \mathbf { v } } ( \lambda , x ) = \lambda ^ { - 1 / 2 } G _ { \mathbf { v } } ( \mu a )$ satisfies the hypotheses of Lemma A.1 in the variables $( a _ { 1 } , a _ { 2 } , a _ { 3 } )$ on the cube $[ 0 , L ] ^ { 3 }$ . Hence positivity of the inverse transforms at the eight corners $a \in \{ 0 , L \} ^ { 3 }$ implies positivity throughout the cube, and in particular throughout the physical simplex $0 \leq a _ { 1 } \leq a _ { 2 } \leq a _ { 3 } \leq L$ . The corner with $j$ entries equal to $L$ is

$$
\omega _ { j } = L \cdot ( \underbrace { 0 , \ldots , 0 } _ { 3 - j } , \underbrace { 1 , \ldots , 1 } _ { j } ) , \qquad j = 0 , 1 , 2 , 3 ,\tag{A.6}
$$

the other corners being permutations of these, which give the same gaps after the corresponding permutation of the control.

## A.2 The corner gaps as Gaussian series

Fix a corner $j$ and a control v. The gap $G _ { \mathbf { v } } ( \omega _ { j } , L )$ is a function of the single variable $L ,$ and the object to be signed is its time inverse. Define $\Gamma = \Gamma _ { j , \mathbf { v } }$ by

$$
\mathcal { L } ^ { - 1 } \big [ \lambda ^ { - 1 / 2 } G _ { \mathbf { v } } ( \mu \omega _ { j } , \mu L ) \big ] ( \tau ) = \frac { \Gamma ( \xi ) } { \sqrt { \pi \tau } } , \qquad \xi = \frac { L ^ { 2 } } { 4 \tau } .\tag{A.7}
$$

Proposition A.2. For each of the ten generators $\mathsf { E } _ { 0 } , \mathsf { V } _ { 0 } , \mathsf { C } _ { 1 } , \mathsf { D } _ { 1 } , \mathsf { T } _ { 1 } , \mathsf { A } _ { 2 } , \mathsf { B } _ { 2 } , \mathsf { T } _ { 2 } , \mathsf { P } _ { 3 } , \mathsf { T } _ { 3 }$ listed in Table 2 there are rational functions $a ( k )$ even and $b ( k )$ odd, with all poles at integers of the parity opposite to j and of modulus at most 4, such that

$$
\Gamma ( \xi ) = { \textstyle { \frac { 1 } { 2 } } } \sum _ { k \in \mathbb { Z } , \ k \equiv j ( 2 ) } { \big ( } a ( k ) + 2 k b ( k ) \xi { \big ) } e ^ { - k ^ { 2 } \xi } .\tag{A.8}
$$

In partial fractions in the variable $k ^ { 2 }$ , a has a polynomial part of degree at most two with simple and double poles, and 2kb a polynomial part of degree at most three with simple poles.

Construction. Write the Region III expansion of Proposition 3.13 as $\begin{array} { r } { F = \sum _ { m \geq 0 } \sum _ { \epsilon } ( c _ { m , \epsilon } + } \end{array}$ $L d _ { m , \epsilon } ) e ^ { - 2 m L + \epsilon \cdot a }$ , with $\begin{array} { r } { c _ { m , \epsilon } = \frac { 1 } { 8 } \sum _ { \ell } s _ { \ell } ( \epsilon ) \alpha _ { m } ^ { ( \ell ) } } \end{array}$ and $\begin{array} { r } { d _ { m , \epsilon } = \frac { 1 } { 8 } \sum _ { \ell } s _ { \ell } ( \epsilon ) \beta _ { m } ^ { ( \ell ) } } \end{array}$ in the notation of (2.11). For a control v put $\begin{array} { r } { d = 2 m ( q _ { \mathbf { v } } ) _ { 4 } - \sum _ { i < 3 } \epsilon _ { i } ( q _ { \mathbf { v } } ) _ { i } } \end{array}$ . Contracting the Hessian as in $\left( \mathrm { A . 2 } \right)$ and using that each $\partial _ { a _ { i } }$ acts on $e ^ { \epsilon \cdot a }$ by $\epsilon _ { i }$ and $\partial _ { a _ { 4 } }$ on $e ^ { - 2 m L } ~ \mathrm { { b y } } - 2 m$ , the mode $( m , \epsilon )$ of $G _ { \mathbf { v } }$ has coeficients

$$
c _ { m , \epsilon } ( 1 - d ^ { 2 } ) + 2 d _ { m , \epsilon } ( q _ { \bf v } ) _ { 4 } d , \qquad d _ { m , \epsilon } ( 1 - d ^ { 2 } ) .\tag{A.9}
$$

At the corner $\omega _ { j }$ these modes collect according to $\begin{array} { r } { k = 2 m - \sum _ { i } \epsilon _ { i } ( \omega _ { j } ) _ { i } / L } \end{array}$ , and substituting $\begin{array} { r } { m = \left( k + \sum _ { i } \epsilon _ { i } ( \omega _ { j } ) _ { i } / L \right) / 2 } \end{array}$ into (A.9) gives $a ( k )$ and $b ( k )$ for generic k. The finitely many modes with $m = 0$ or $m < 0$ are treated separately; every one of them either has zero coeficient or reproduces the generic formula, with the mode $k = 0$ carrying half weight. This is the content of the script audit\_corner\_algebra.py, which also re-derives all 64 Hessian contractions independently from the hyperbolic form (3.15) and checks the partial fractions.

Given $a , b ,$ formula (A.8) follows from (A.7): if $\begin{array} { r } { G _ { \mathbf { v } } = \sum _ { k > 0 } [ a ( k ) + L b ( k ) ] e ^ { - k L } } \end{array}$ (plus ${ \scriptstyle { \frac { 1 } { 2 } } } a ( 0 )$ when $j$ is even), then $\begin{array} { r } { \lambda ^ { - 1 / 2 } G _ { \mathbf { v } } ( \mu L ) = \sum _ { k } [ a ( k ) \lambda ^ { - 1 / 2 } + L b ( k ) ] e ^ { - k L \mu } } \end{array}$ , whose inverse transform is $\begin{array} { r } { \sum _ { k } [ a ( k ) \mathcal { H } _ { 2 } + L b ( k ) \mathcal { H } _ { 3 } ] ( k L , \tau ) } \end{array}$ by (2.7); since $\mathcal { H } _ { 2 } = \mathcal { G }$ and $\begin{array} { r } { \mathcal { H } _ { 3 } = \frac { X } { 2 \tau } \mathcal { G } } \end{array}$ and $k L ^ { 2 } / ( 2 \tau ) = 2 k \xi .$ this is $\begin{array} { r } { ( \pi \tau ) ^ { - 1 / 2 } \sum _ { k > 0 } [ a ( k ) + 2 k b ( k ) \xi ] e ^ { - k ^ { 2 } \xi } } \end{array}$ . The sum over $k \in \mathbb { Z }$ in $\left( \mathrm { A } . 8 \right)$ is an even extension: a is even and $2 k b ( k )$ is even, so the symmetric sum is twice the sum over $k > 0$ plus the half-weighted zero mode. It is not an extension of the geometric exponential series. □

For example, for $\mathsf { E } _ { 0 } = e$ itself (the control 00000, where $q _ { \mathbf { v } } = 0 )$ ,

$$
a ( k ) = - \frac { 3 k ^ { 6 } - 2 5 k ^ { 4 } - 4 k ^ { 2 } - 6 4 } { 6 4 ( k ^ { 2 } - 1 ) ^ { 2 } } , \qquad b ( k ) = \frac { k ( k ^ { 2 } - 4 ) ( k ^ { 2 } - 1 6 ) } { 6 4 ( k ^ { 2 } - 1 ) } ,
$$

and $a ( 2 m ) = A _ { m } , b ( 2 m ) = B _ { m }$ for $m \geq 1$ while $a ( 0 ) = 1$ , in agreement with (2.9) and the half weight at $k = 0$

## A.3 Small scales: a dual representation

Set, for $a > 0$

$$
\begin{array} { c l l } { \displaystyle { Q _ { a } ( \xi ) = \frac { \sqrt \pi } { 4 } \int _ { 0 } ^ { \xi } e ^ { - a ^ { 2 } ( \xi - s ) } s ^ { - 1 / 2 } d s = \frac { \sqrt \pi } { 2 a } \mathrm { D a w s o n } ( a \sqrt \xi ) , } } \\ { \displaystyle { \frac { \sqrt \pi } { 2 } \sqrt { \xi } e ^ { - a ^ { 2 } \xi } \le Q _ { a } ( \xi ) \le \frac { \sqrt \pi } { 2 } \sqrt { \xi } . } } \end{array}\tag{A.10}
$$

Poisson summation gives, for $j \in \{ 0 , 1 \}$ ,

$$
F _ { j } ( \xi ) : = \frac { 1 } { 2 } \sum _ { k \equiv j ( 2 ) } e ^ { - k ^ { 2 } \xi } = \frac { \sqrt { \pi } } { 4 \sqrt { \xi } } \Big ( 1 + 2 \sum _ { \ell \geq 1 } ( - 1 ) ^ { j \ell } e ^ { - \pi ^ { 2 } \ell ^ { 2 } / ( 4 \xi ) } \Big ) .\tag{A.11}
$$

Write $F _ { j } = F ^ { ( 0 ) } + F ^ { \mathrm { i m } }$ with $\begin{array} { r } { F ^ { ( 0 ) } = \frac { \sqrt { \pi } } { 4 } \xi ^ { - 1 / 2 } } \end{array}$ the zero image. Substituting the partial-fraction decomposition of a and 2kb into $\left( \mathrm { A } . 8 \right)$ expresses Γ through $F _ { j }$ and its derivatives and through

$$
S _ { a } = \textstyle { \frac { 1 } { 2 } } \sum _ { k \equiv j ( 2 ) } \frac { e ^ { - k ^ { 2 } \xi } } { k ^ { 2 } - a ^ { 2 } } , \qquad T _ { a } = \textstyle { \frac { 1 } { 2 } } \sum _ { k \equiv j ( 2 ) } \frac { e ^ { - k ^ { 2 } \xi } } { ( k ^ { 2 } - a ^ { 2 } ) ^ { 2 } } .\tag{A.12}
$$

These satisfy $S _ { a } ^ { \prime } + a ^ { 2 } S _ { a } = - F _ { j }$ and $T _ { a } ^ { \prime } + a ^ { 2 } T _ { a } = - S _ { a }$ , and the cotangent partial fraction expansion gives the initial values

$$
S _ { a } ( 0 ) = 0 , \qquad T _ { a } ( 0 ) = \frac { \pi ^ { 2 } } { 1 6 a ^ { 2 } } \qquad ( a > 0 ~ \mathrm { o f ~ p a r i t y ~ o p p o s i t e ~ t o } ~ j ) ,\tag{A.13}
$$

together with $S _ { 0 } ( 0 ) = \pi ^ { 2 } / 8$ when $j$ is odd. Replacing $F _ { j }$ by $F ^ { ( 0 ) }$ and keeping (A.13) produces the principal expression M: $S _ { a } ^ { ( 0 ) } = - Q _ { a }$ , and

$$
T _ { a } ^ { ( 0 ) } + \xi S _ { a } ^ { ( 0 ) } = \frac { \pi ^ { 2 } } { 1 6 a ^ { 2 } } e ^ { - a ^ { 2 } \xi } + \frac { Q _ { a } } { 2 a ^ { 2 } } - \frac { \sqrt { \pi } \sqrt { \xi } } { 4 a ^ { 2 } } .\tag{A.14}
$$

At every nonzero pole the coeficient of $\xi S _ { a }$ equals that of $T _ { a }$ , so (A.14) applies; all pure $\sqrt { \xi }$ terms then cancel against the polynomial part and the $a = 0$ term, and all $\xi ^ { - 1 / 2 - h }$ terms with $h \geq 0$ cancel as well. What survives is

$$
M ( \xi ) = \pi ^ { 2 } \sum _ { a } d _ { a } e ^ { - a ^ { 2 } \xi } + \sum _ { a } q _ { a } Q _ { a } ( \xi ) , \qquad d _ { a } = \frac { \beta _ { a } } { 1 6 a ^ { 2 } } , \quad q _ { a } = - \alpha _ { a } + \frac { \beta _ { a } } { 2 a ^ { 2 } } ,\tag{A.15}
$$

where $\alpha _ { a } , \beta _ { a }$ are the simple and double residues of $a ( k )$ at $k ^ { 2 } = a ^ { 2 }$ . For $\mathsf { E } _ { 0 }$ , for instance, $M = 4 5 \pi ^ { 2 } e ^ { - \xi } / 5 1 2$ ; the ten expressions are tabulated in the supplement’s note region-three-verification.tex and recomputed by certify\_corners.py.

Lemma A.3. For $0 < \xi \leq 1 / 1 0 0$ every generator satisfies

$$
\begin{array} { r } { | \Gamma ( \xi ) - M ( \xi ) | \le 1 0 ^ { 6 } \xi ^ { - 1 3 / 2 } e ^ { - 9 / ( 4 \xi ) } < 1 0 ^ { - 4 0 } \sqrt { \xi } . } \end{array}
$$

Moreover $M > \sqrt \xi / 5$ for $\mathsf C _ { 1 } , \mathsf D _ { 1 } , \mathsf A _ { 2 } , \mathsf B _ { 2 } , \mathsf P _ { 3 }$ , and $M > 1 / 5 0$ for $\mathsf { E } _ { 0 } , \mathsf { V } _ { 0 } , \mathsf { T } _ { 1 } , \mathsf { T } _ { 2 } , \mathsf { T } _ { 3 }$

Sketch; the details are in $I 5 J .$ For the remainder, $\partial _ { \xi } ^ { h } [ \xi ^ { - 1 / 2 } e ^ { - b / \xi } ] = \xi ^ { - 1 / 2 - h } e ^ { - b / \xi } P _ { h } ( b / \xi )$ with $P _ { 0 } = 1$ and $P _ { h + 1 } = ( X - h - \textstyle \frac { 1 } { 2 } ) P _ { h } - X P _ { h } ^ { \prime } $ ; the sums $\sigma _ { h }$ of absolute coeficients for $h \leq 3$ are $1 , { \frac { 3 } { 2 } } , { \frac { 1 9 } { 4 } } , { \frac { 1 7 3 } { 8 } }$ . Using $9 < \pi ^ { 2 } < 1 0$ in both $\begin{array} { r } { | P _ { h } ( X ) | \le \sigma _ { h } X ^ { h } \le \sigma _ { h } ( \frac { 5 } { 2 } ) ^ { h } \ell ^ { 2 h } \xi ^ { - h } } \end{array}$ and $e ^ { - \pi ^ { 2 } \ell ^ { 2 } / ( 4 \xi ) } <$ $e ^ { - 9 \ell ^ { 2 } / ( 4 \xi ) }$ , together with $\textstyle \sum _ { \ell > 1 } \ell ^ { 2 h } e ^ { - 9 \ell ^ { 2 } / ( 4 \xi ) } \leq 2 e ^ { - 9 / ( 4 \xi ) }$ , gives $\vert \partial _ { \xi } ^ { h } F ^ { \mathrm { i m } } \vert \le 1 0 0 0 \xi ^ { - 1 3 / 2 } e ^ { - 9 / ( 4 \xi ) }$ for all $h \leq 3$ and $\xi \le 1$ . The polynomial part contributes at most 30 times this, the coeficient norm being below 30 for every generator, and the $S _ { a } , T _ { a }$ remainders are smaller still because their integrating factors are at most one. The second inequality reduces to the endpoint, since $\xi ^ { - 7 } e ^ { - 9 / ( 4 \bar { \xi } ) }$ increases on $( 0 , 1 / 1 0 0 ]$ , and there it reads $1 { \bar { 0 } } ^ { 6 } 1 0 { \bar { 0 } } ^ { 7 } 1 0 ^ { 4 0 } = 1 0 ^ { 6 0 } < 2 ^ { 2 2 { \bar { 5 } } } < e ^ { 2 2 5 }$

For positivity, put $r = \sqrt { \xi } \le 1 / 1 0$ and use $1 - a ^ { 2 } \xi \le e ^ { - a ^ { 2 } \xi } \le 1 , 9 < \pi ^ { 2 } < 1 0$ and $\scriptstyle { \frac { 3 } { 4 } } r ( 1$   
$a ^ { 2 } / 1 0 0 ) \leq Q _ { a } \leq r$ from $\mathrm { ( A . 1 0 ) }$ . When the coeficients $d _ { a }$ sum to zero the leading behavior   
is ∝ r and the resulting rational lower bounds are $\textstyle \frac { 1 4 4 } { 1 7 5 } , \frac { 6 0 6 0 3 } { 1 7 9 2 0 0 } , \frac { 2 0 5 3 2 6 7 } { 2 8 6 7 2 0 0 } , \frac { 1 7 0 6 4 3 } { 7 1 6 8 0 0 } , \frac { 3 8 4 0 0 9 } { 7 1 6 8 0 0 } .$ , each   
exceeding $1 / 5 ;$ otherwise it is constant, with lower bounds <sup>8019</sup> , <sup>2923</sup> , <sup>42643</sup> , <sup>203079</sup> , <sup>15403</sup> <sub>10240 14336 286720 2293760 573440</sub> ,   
each exceeding $1 / 5 0$ . All of these are exact rational computations performed by ${ \mathsf { c e r t i f y } } _ { - }$   
corners.py from the stored partial fractions. □

## A.4 The compact range and the large-scale tail

The remaining two ranges use only rational arithmetic and an outward-rounded interval exponential. We first bound the coeficients. Factoring each denominator and bounding its roots by 4, and using the absolute coeficient norm of the numerator after division by its leading power of k, gives

$$
| a ( k ) | \leq 1 0 0 k ^ { 8 } , \qquad | 2 k b ( k ) | \leq 1 0 0 k ^ { 8 } \qquad ( k \geq 1 0 ) .\tag{A.16}
$$

No estimated constants enter: for $k \geq 1 0$ one has $| k - \rho | \geq k ( 1 - | \rho | / 1 0 )$ for every root $\rho ,$ and $\begin{array} { r } { | n ( k ) | \leq k ^ { \deg n } \sum _ { i } | n _ { i } | 1 0 ^ { i - \deg n } } \end{array}$

Next we bound the tail. Truncating the lattice at $N = 1 2 8$ , and using that $t \mapsto t ^ { 8 } e ^ { - t ^ { 2 } / 1 0 0 }$ decreases for $t \geq 2 0$ 2

$$
\left| \frac { 1 } { 2 } \sum _ { | k | > N } \left[ a ( k ) + 2 k b ( k ) \xi \right] e ^ { - k ^ { 2 } \xi } \right| \leq T _ { N } : = 2 0 0 \int _ { N } ^ { \infty } t ^ { 8 } e ^ { - t ^ { 2 } / 1 0 0 } d t < 4 . 0 2 7 1 \times 1 0 ^ { - 5 3 }\tag{A.17}
$$

uniformly for $1 / 1 0 0 \leq \xi \leq 1$ , the integral being bounded by the exact recursion $I _ { 0 } ( N ) \leq$ $e ^ { - \alpha N ^ { 2 } } / ( \bar { 2 } \alpha N ) , \stackrel { \prime } { I } _ { p } ( N ) = N ^ { p - 1 } e ^ { - \alpha N ^ { 2 } } / ( \stackrel {  } { 2 \alpha } ) + \textstyle \frac { p - 1 } { 2 \alpha } \tilde { I } _ { p - 2 } ( N ) \mathrm { w i t h } \alpha = 1 / 1 0 0$

On the compact range we use a Taylor bound on rational cells. On a cell $[ l , h ]$ with midpoint z and radius $d ,$ write $f _ { N }$ for the truncated sum. Taylor’s theorem gives the rigorous lower bound

$$
\begin{array} { l } { { \mathrm { ( A . 1 8 ) } } } \\ { { \displaystyle \Gamma ( \xi ) \geq f _ { N } ( z ) - d | f _ { N } ^ { \prime } ( z ) | - \frac { 1 } { 2 } d ^ { 2 } B ( l , h ) - T _ { N } , \qquad B ( l , h ) = \sum _ { 0 \leq k \leq N } \left[ k ^ { 4 } ( | a | + | c _ { k } | h ) + 2 k ^ { 2 } | c _ { k } | \right] e ^ { - k ^ { 2 } l } , } } \end{array}
$$

with $c _ { k } = 2 k b ( k )$ and the half weight at $k = 0 .$ ; B bounds $| f _ { N } ^ { \prime \prime } |$ on the whole cell. Starting from $[ 1 / 1 0 0 , 1 ]$ the verifier bisects until the interval evaluation of the right-hand side is strictly positive, and refuses cells of width below $1 0 ^ { - 1 0 }$ . Every exponential in $\mathrm { ( A . 1 8 ) }$ is enclosed by a rational interval obtained from the Taylor series of $e ^ { - x }$ with outward rounding, so the certificate for a cell is a finite list of rational inequalities; these are the cells that the Lean development of the supplement verifies.

On the large range we use first-mode domination. Let $k _ { 0 }$ be the first nonzero mode and α<sub>0</sub> its coeficient, including the half weight at $k = 0$ . In every case $\alpha _ { 0 } > 0$ and $c _ { k _ { 0 } } = 0$ , so for $\xi \ge 1$

$$
e ^ { k _ { 0 } ^ { 2 } \xi } \Gamma ( \xi ) \ge \alpha _ { 0 } - \sum _ { k > k _ { 0 } } \big ( | a ( k ) | + | c _ { k } | \big ) e ^ { - ( k ^ { 2 } - k _ { 0 } ^ { 2 } ) } > 0 ,\tag{A.19}
$$

since both $e ^ { - \delta \xi }$ and $\xi e ^ { - \delta \xi }$ decrease for $\xi \ge 1$ when $\delta = k ^ { 2 } - k _ { 0 } ^ { 2 } \geq 1$ ; the tail beyond 128 is bounded by $2 0 0 e ^ { k _ { 0 } ^ { 2 } } \int _ { 1 2 8 } ^ { \infty } t ^ { 8 } e ^ { - t ^ { 2 } } d t$

<table><tr><td>Generator</td><td>Cells  $k _ { 0 }$ </td><td> $\alpha _ { 0 }$ </td><td>Large-scale margin</td></tr><tr><td> $\mathsf { E } _ { 0 }$ </td><td>16</td><td>0  $1 / 2$ </td><td>0.4908</td></tr><tr><td> $\mathsf { V } _ { 0 }$ </td><td>139</td><td>4  $4 / 3$ </td><td>1.3333</td></tr><tr><td> $\mathsf { C } _ { 1 }$ </td><td>96</td><td>1  $2 / 3$ </td><td>0.6663</td></tr><tr><td> $\mathsf { D } _ { 1 }$ </td><td>135</td><td>3  $4 / 1 5$ </td><td>0.2666</td></tr><tr><td> ${ \sf T } _ { 1 }$ </td><td>101</td><td>5  $1 6 / 2 1$ </td><td>0.7619</td></tr><tr><td> $\mathsf { A } _ { 2 }$ </td><td>26</td><td>2  $4 / 1 5$ </td><td>0.2666</td></tr><tr><td> $\mathsf { B } _ { 2 }$ </td><td>84</td><td>4 16/105</td><td>0.1523</td></tr><tr><td> ${ \mathsf { T } } _ { 2 }$ </td><td>80</td><td>6 32/63</td><td>0.5079</td></tr><tr><td> $\mathsf { P } _ { 3 }$ </td><td>33</td><td>3 16/105</td><td>0.1523</td></tr><tr><td> ${ \mathsf T } _ { 3 }$ </td><td>97</td><td>7 256/693</td><td>0.3694</td></tr></table>

Table 2: The Region III certificate: 807 compact cells in total. Margins are rounded down for display; the certified intervals and margins are archived in corner\_intervals.json.

## A.5 Conclusion in Region III

Proof of Theorem $ 4 . 1 0 .$ The 64 corner gaps decompose, by exact constant-coeficient identities checked in audit\_corner\_algebra.py, into nonnegative combinations of the ten generators of Table 2, of the identically vanishing gaps, and of two quantities with direct positive representations: the gap η of the control 00111 at $j = 0$ , and sinh $L \Lambda ( L )$ , for which (A.20)

$$
\lambda ^ { - 1 / 2 } \eta ( L \mu ) = 3 \int _ { 0 } ^ { L } \left( \frac { \sinh ( s \mu ) } { \sinh ( L \mu ) } \right) ^ { 6 } d s , \qquad \lambda ^ { - 1 / 2 } \sinh ( L \mu ) \Lambda ( L \mu ) = \int _ { L } ^ { \infty } \frac { \sinh ( L \mu ) } { \sinh ( s \mu ) } d s ,
$$

both manifestly integrals of products of Brownian exit-time transforms, hence in CM with strictly positive inverse for $L > 0$ . At the corner $\omega _ { 0 }$ the decomposition can be written down by hand. There every exponential of the sign sum equals one, the only nonzero second derivatives of $F$ in (A.2) are $\partial _ { a _ { i } } ^ { 2 } F = e , \partial _ { a _ { i } } \partial _ { a _ { l } } F = b _ { 2 }$ for $i \neq l \leq 3 , \partial _ { a _ { i } } \partial _ { a _ { 4 } } F = b _ { 1 } ^ { \prime }$ and $\partial _ { a _ { 4 } } ^ { 2 } F = e ^ { \prime \prime }$ , and (3.17) together with $\eta = e - e ^ { \prime \prime }$ gives

$$
G _ { \mathbf { v } } ( \omega _ { 0 } ) = c _ { e } e + c _ { \eta } \eta + c _ { S } \sinh L \Lambda ( L ) ,
$$

where, with $q = q _ { \mathbf { v } } , s _ { 1 } = q _ { 1 } + q _ { 2 } + q _ { 3 }$ and $e _ { 2 } = q _ { 1 } q _ { 2 } + q _ { 1 } q _ { 3 } + q _ { 2 } q _ { 3 }$

$$
\begin{array} { r } { c _ { e } = 1 - q _ { 1 } ^ { 2 } - q _ { 2 } ^ { 2 } - q _ { 3 } ^ { 2 } - q _ { 4 } ^ { 2 } - \frac 1 3 q _ { 4 } s _ { 1 } - \frac 1 3 e _ { 2 } , \qquad c _ { \eta } = q _ { 4 } ^ { 2 } + \frac 1 3 q _ { 4 } s _ { 1 } + \frac 1 { 2 1 } e _ { 2 } , \qquad c _ { S } = - \frac 6 7 e _ { 2 } . } \end{array}
$$

For the controls 00001, 00010 and 00100 the coeficients are $\textstyle { \bigl ( } { \frac { 1 } { 3 } } , { \frac { 2 } { 2 1 } } , { \frac { 2 } { 7 } } { \bigr ) }$ , for 00011, 00101 and 00110 they are $( 0 , \frac { 3 } { 7 } , \frac { 2 } { 7 } )$ , and for 00111 they are $( 0 , 1 , 0 )$ ; all of them are nonnegative. Lemma $\mathrm { A . 3 , ~ ( A . 1 8 ) }$ and (A.19) prove that each generator has a strictly positive inverse transform for every $\xi > 0 ,$ , hence for every $L , \tau > 0$ . Lemma A.1 then propagates positivity from the corners to the whole cube, and CM is preserved. Thus the transformed gap of every control lies in CM at every point of the closed region with $L > 0 ,$ , and Lemma 4.12, applied to $2 U _ { \tau } - D _ { \mathbf { v } } ^ { 2 } U$ , which is continuous in $\tau$ with the growth that Lemma 4.1 provides, gives the inequality.

For strictness, the COMB representative is 01010; its gap vanishes at $\omega _ { 0 }$ and equals the strictly positive generators $\mathsf { D } _ { 1 } , \mathsf { B } _ { 2 } , \mathsf { P } _ { 3 }$ at the other three ordered corners $\omega _ { 1 } , \omega _ { 2 } , \omega _ { 3 }$

Suppose that some $a _ { i } > 0$ , and let $\omega _ { j } , j \geq 1$ , be the ordered corner whose zero entries are exactly those i with $a _ { i } = 0$ . Every summand of (A.5) lies in CM, so the transformed COMB gap minus the summand of $\omega _ { j }$ lies in CM. That summand is ${ \widehat { \kappa } } { \widehat { g } } .$ , where $\widehat { \kappa } \ : = \ :$ $\begin{array} { r } { \prod _ { a _ { i } > 0 } W _ { 1 } ( \lambda ; a _ { i } , L ) } \end{array}$ , the factors $W _ { 0 } ( \lambda ; 0 , L ) = 1$ dropping out, and $\widehat g$ is the transform of the inverse $\Gamma ( \xi ) / \sqrt { \pi \tau }$ of the generator at $\omega _ { j }$ , which is continuous and strictly positive in $\tau > 0$ Since sinh $( a \mu ) / \sinh ( L \mu ) \geq \textstyle { \frac { 1 } { 2 } } e ^ { - ( L - a ) \mu }$ as soon as $e ^ { - 2 a \mu } \leq { \frac { 1 } { 2 } }$ , we have $\widehat { \kappa } ( \lambda ) \geq e ^ { - 4 L \sqrt { \lambda } }$ for all large $\lambda ,$ and the second statement of Lemma 4.12, applied to the COMB gap, gives strict positivity. When $a _ { 1 } = a _ { 2 } = a _ { 3 } = 0$ the corner identity gives equality. □

## B The Regions I and II certificate

In Regions I and II we use the coordinates

$$
L = z _ { 1 } , \qquad b = z _ { 2 } , \qquad c = | z _ { 3 } | , \qquad Z = z _ { 4 } , \qquad 0 \leq c \leq b \leq L , \quad Z \geq 0 ,\tag{B.1}
$$

and the normalized gap $\gamma _ { \mathbf { v } } = ( D _ { \mathbf { v } _ { \ast } } ^ { 2 } U - D _ { \mathbf { v } } ^ { 2 } U ) / k$ , whose transform is $\widehat { \gamma } _ { \mathbf { v } } = \lambda ^ { - 1 / 2 } G _ { \mathbf { v } } ( \mu x )$ as in (4.24). The weights $W _ { 0 } , W _ { 1 }$ of (A.4) are now taken on $( 0 , L )$ , that is with $r = L$

## B.1 Reduction to four corner families

The stationary gaps again satisfy $\partial _ { b b } g = \lambda g$ and $\partial _ { c c } g = \lambda g$ , so two-point interpolation on the square $[ 0 , L ] ^ { 2 }$ is exact. Three of its four corners are physical states; the fourth is not, and the point of the following identity is that its contribution is nonetheless positive once its interpolation weights are included.

Let $P$ interchange the first two entries of a control in Region I (the third and fourth in Region II) and put $h _ { \mathbf { v } } = 1 - ( \mathbf { v } _ { 1 } - \mathbf { v } _ { 2 } ) ^ { 2 } \in \{ 0 , 1 \}$ . The stationary nonphysical-corner identity is, in Region I,

$$
G _ { \mathbf { v } } ( L , 0 , L , Z ) = G _ { P \mathbf { v } } ( L , L , 0 , Z ) + h _ { \mathbf { v } } \sinh L ,\tag{B.2}
$$

and in Region II the same identity without the term $h _ { \mathbf { v } }$ sinh $L .$ . It is a symmetry of the stationary formula (3.26). In the coordinates (B.1), (3.26) is symmetric under $b  c$ except for the term $- { \textstyle \frac { 1 } { 2 } } \sinh ( z _ { 2 } + z _ { 3 } )$ of $F _ { 4 }$ , which equals $- { \frac { 1 } { 2 } } \sinh ( b - c )$ in Region I, where $z _ { 3 } = - c ,$ and is symmetric in Region II, where $z _ { 3 } = c$ . The permutation P exchanges b and c and fixes $L$ and $Z ,$ so the gap of v at $( L , 0 , L , Z )$ is the gap of Pv at $( L , L , 0 , Z )$ , plus, in Region I, the gap of v for the function sinh $( c - b )$ , which is $( 1 - ( \mathbf { v } _ { 1 } - \mathbf { v } _ { 2 } ) ^ { 2 } )$ sinh L because $\partial _ { \mathbf { v } } ( b - c ) = k ( \mathbf { v } _ { 1 } - \mathbf { v } _ { 2 } )$ . The extra term must not be inverted on its own. Including its weights,

$$
\frac { \sinh ( L \mu ) } { \mu } W _ { 0 } ( b ) W _ { 1 } ( c ) = \frac { \sinh ( c \mu ) \sinh ( ( L - b ) \mu ) } { \mu \sinh ( L \mu ) } = : R _ { \lambda } ^ { L } ( b , c ) , \qquad c \leq b ,\tag{B.3}
$$

which is exactly the Dirichlet resolvent kernel of $\lambda - \partial _ { b b }$ on (0, L). It lies in CM: since coth x + coth $y = \sinh ( x + y ) /$ (sinh x sinh y),

$$
R _ { \lambda } ^ { L } ( b , c ) = { \frac { \sinh ( c \mu ) } { \sinh ( b \mu ) } } \cdot { \frac { 1 } { b _ { \lambda } ( b ) + b _ { \lambda } ( L - b ) } } = W _ { 1 } ( \lambda ; c , b ) \cdot 2 \int _ { 0 } ^ { \infty } e ^ { - 2 s b _ { \lambda } ( b ) } e ^ { - 2 s b _ { \lambda } ( L - b ) } d s ,\tag{B.4}
$$

with $b _ { \lambda } ( r ) = \mu \coth ( \mu r )$ as in (B.6) below. The weight $W _ { 1 }$ is in CM, and so is $e ^ { - 2 s b _ { \lambda } ( r ) }$ for every $s , r > 0$ , because $b _ { \lambda } ( r )$ is a Bernstein function (Lemma B.2).

Name the four physical corner families by their scaled gaps:

$$
\begin{array} { l l l } { \mathrm { f a m i l y ~ 0 : ~ } } & { ( y _ { 1 } , y _ { 2 } , y _ { 3 } , y _ { 4 } ) = ( 0 , L , 0 , Z ) , } & { m = 1 , } \\ { \mathrm { f a m i l y ~ 1 : ~ } } & { ( L , 0 , L , Z ) , } & { m = 2 , } \\ { \mathrm { f a m i l y ~ I : ~ } } & { ( 0 , 0 , 2 L , Z ) , } & { m = 3 , } \\ { \mathrm { f a m i l y ~ I I : ~ } } & { ( 2 L , 0 , 0 , Z + L ) , } & { m = 3 , } \end{array}\tag{B.5}
$$

where m is the trace power that will govern the corresponding evolution. Three of the four families meet Region III at $Z = 0 { \mathrm { : } }$ : the states of families 0, 1 and II at $Z = 0$ , with scaled gaps $( 0 , L , 0 , 0 ) , ( L , 0 , L , 0 )$ and $( 2 L , 0 , 0 , L )$ , are the Region III corners $\omega _ { 0 } , \omega _ { 1 } , \omega _ { 2 }$ of $\mathrm { ( A . 6 ) }$ 2 where $a = L ( 0 , 0 , 0 , 1 ) , L ( 0 , 0 , 1 , 1 )$ and $L ( 0 , 1 , 1 , 1 )$ . Since u is $C ^ { 2 } \left[ 6 \right]$ , every gap of these three families at $Z = 0$ is the corresponding Region III corner gap, with no further computation. Family I meets Region III nowhere.

Proposition B.1. In Region I, for every control v,

$$
\begin{array} { r l } & { \widehat { \gamma } _ { \mathbf { v } } ( L , b , c , Z ) = W _ { 0 } ( b ) W _ { 0 } ( c ) \widehat { \gamma } _ { 0 , \mathbf { v } } ( L , Z ) + W _ { 1 } ( b ) W _ { 0 } ( c ) \widehat { \gamma } _ { 1 , \mathbf { v } } ( L , Z ) } \\ & { \qquad + W _ { 0 } ( b ) W _ { 1 } ( c ) \widehat { \gamma } _ { 1 , P \mathbf { v } } ( L , Z ) + W _ { 1 } ( b ) W _ { 1 } ( c ) \widehat { \gamma } _ { \mathrm { I } , \mathbf { v } } ( L , Z ) + h _ { \mathbf { v } } R _ { \lambda } ^ { L } ( b , c ) . } \end{array}
$$

In Region II the same holds with family I replaced by family II, with P the transposition of entries 3 and 4, and with no Green term. Hence nonnegativity of the inverse transforms of the four physical corner families implies (4.23) at every state of Regions I and II with $L > 0$

Proof. Two-point interpolation in b and c as in Lemma A.1, followed by substitution of (B.2) at the one nonphysical corner and of (B.3) for the extra term. Every weight is in CM, and so is the Green term, by (B.4). The two families shared between the regions agree, because U is $C ^ { 2 }$ across the interface by Theorem 4.3. The identities (B.2) and (B.3), the endpoint values and the derivative jump of the Green kernel, and the fact that $h _ { \mathbf { v } } \in \{ 0 , 1 \}$ for all 32 controls, are also verified symbolically in check\_regions12\_reduction.py. □

## B.2 Positivity-preserving evolutions and the pure traces

Everything now happens in the two variables $( L , Z )$ . Write

$$
b _ { \lambda } ( L ) = \mu \coth ( \mu L ) , \qquad \mathcal { R } _ { m } f ( L ) = \mu ^ { 2 } \sinh ^ { m } ( \mu L ) \int _ { L } ^ { \infty } \frac { f ( r ) } { \sinh ^ { m + 2 } ( \mu r ) } d r ,\tag{B.6}
$$

and $K _ { m } = \mathcal { T } _ { m + 1 , m } , \mathcal { A } _ { m } = \mathcal { T } _ { m , m }$

Lemma B.2. Let $\alpha \geq 0$ , let $m \geq 0$ be an integer, and let $\epsilon > 0$ and $Z _ { 0 } ~ > ~ 0$ . Suppose that, for each $\lambda > 0$ , the data $f ( \cdot , 0 )$ are bounded and continuous on $[ \epsilon , \infty )$ , the forcing q is bounded on $[ \epsilon , \infty ) \times [ 0 , Z _ { 0 } ]$ and continuous in each variable, and f is a bounded mild solution of $\partial _ { Z } f = 2 T _ { \alpha , m } f + q$ there: f is bounded on $[ \epsilon , \infty ) \times [ 0 , Z _ { 0 } ]$ , continuous in each variable, and satisfies

$$
f ( L , Z ) = e ^ { - 2 Z b _ { \lambda } ( L ) } f ( L , 0 ) + \int _ { 0 } ^ { Z } e ^ { - 2 ( Z - s ) b _ { \lambda } ( L ) } \big ( 2 \alpha { \mathcal R } _ { m } f ( L , s ) + q ( L , s ) \big ) d s
$$

for $L \geq \epsilon$ and $0 \leq Z \leq Z _ { 0 } . I f f ( L , 0 )$ and $q ( L , s )$ , as functions of λ, lie in CM for all $L \geq \epsilon$ and $s \in [ 0 , Z _ { 0 } ]$ , then so do $f ( L , Z )$ and $f ( L , \dot { Z } ) - e ^ { - 2 Z b _ { \lambda } ( L ) } f ( L , 0 )$ , for all $L \geq \epsilon$ and $Z \in [ 0 , Z _ { 0 } ]$

Proof. The diagonal part contributes the integrating factor $e ^ { - 2 Z b _ { \lambda } ( L ) }$ , which is in CM because $\begin{array} { r } { b _ { \lambda } ( L ) = \frac { 1 } { L } + \frac { 2 } { L } \sum _ { n > 1 } \lambda / ( \lambda + ( \pi n / L ) ^ { 2 } ) } \end{array}$ is a Bernstein function. The kernel of $\alpha \mathcal { R } _ { m }$ factors as

$$
\alpha \left( \frac { \sinh ( \mu L ) } { \sinh ( \mu r ) } \right) ^ { m } \left( \frac { \mu } { \sinh ( \mu r ) } \right) ^ { 2 } , \qquad r \geq L ,
$$

whose first factor is a product of m Brownian exit-time transforms $W _ { 1 } ( \lambda ; L , r )$ , m being an integer, and whose second is in CM by the product formula $r \mu /$ sinh $\begin{array} { r } { ( r \mu ) = \prod _ { n > 1 } ( 1 + } \end{array}$ $r ^ { 2 } \lambda / ( \pi ^ { 2 } n ^ { 2 } ) ) ^ { - 1 }$ . Now fix $\lambda > 0$ . Since sinh $. ( \mu r ) \geq \sinh ( \mu L )$ for $r \geq L .$

$$
\mu ^ { 2 } \sinh ^ { m } ( \mu L ) \int _ { L } ^ { \infty } { \frac { d r } { \sinh ^ { m + 2 } ( \mu r ) } } \leq \mu ^ { 2 } \int _ { L } ^ { \infty } { \frac { d r } { \sinh ^ { 2 } ( \mu r ) } } \leq \mu { \bigl ( } \coth ( \mu \epsilon ) - 1 { \bigr ) } = : K
$$

for $L \geq \epsilon , \mathrm { s o } \left| \mathcal { R } _ { m } \Phi \right| \leq K \mathrm { s u p } \left| \Phi \right|$ , and the Duhamel map

$$
\mathcal { D } \Phi = e ^ { - 2 Z b _ { \lambda } } f ( \cdot , 0 ) + \int _ { 0 } ^ { Z } e ^ { - 2 ( Z - s ) b _ { \lambda } } \bigl ( 2 \alpha \mathcal { R } _ { m } \Phi + q \bigr ) d s
$$

takes bounded functions on $[ \epsilon , \infty ) \times [ 0 , Z _ { 0 } ]$ that are continuous in each variable to functions of the same kind. Its iterates $\Phi _ { 0 } = 0 , \Phi _ { n + 1 } = \mathcal { D } \Phi _ { n }$ satisfy $| f - \Phi _ { n } | \leq M ( 2 \alpha K Z ) ^ { n } / n !$ with $M = \operatorname* { s u p } | f |$ , by induction from the Duhamel identity for f and $0 < \dot { e } ^ { - 2 ( Z - s ) b _ { \lambda } } \le$ 1, so $\Phi _ { n } \ \to \ f$ pointwise. By induction on $n ,$ every $\Phi _ { n } ( L , Z )$ lies in CM, and so does $\begin{array} { r } { \Phi _ { n + 1 } ( L , Z ) - e ^ { - 2 Z b _ { \lambda } ( L ) } f ( L , 0 ) = \int _ { 0 } ^ { Z } e ^ { - 2 ( Z - s ) b _ { \lambda } ( L ) } \big ( 2 \alpha { \mathcal R } _ { m } \Phi _ { n } + q \big ) ( L , s ) d s } \end{array}$ , because CM is closed under products, sums and integrals against nonnegative weights. Since CM is closed under pointwise limits, the lemma follows.

In the applications below, the data, the forcing and the solution are explicit functions that are bounded and continuous on $[ \epsilon , \infty ) \times [ 0 , Z _ { 0 } ]$ for every $\epsilon , Z _ { 0 } > 0$ , and the Duhamel identity is the integrated form of the diferential equation derived for them. The Lean formalization checks these hypotheses for every flow of this appendix. □

At discount one put $\rho ( r ) = \sinh r p ( r )$ with p from (2.15), $c _ { r } = \coth r , \Phi _ { r } ( L ) = \sinh ( r -$ $L ) /$ sinh $r ,$ and

$$
\begin{array} { c } { { h _ { m } ( L , Z ) = \displaystyle \int _ { L } ^ { \infty } \rho ( r ) \Phi _ { r } ( L ) ^ { m } e ^ { - 2 Z c _ { r } } d r , } } \\ { { { \cal D } _ { m } ( L , Z ) = 4 \sinh L \displaystyle \int _ { L } ^ { \infty } \rho ( r ) c _ { r } ( c _ { r } ^ { 2 } - 1 ) \Phi _ { r } ( L ) ^ { m - 1 } e ^ { - 2 Z c _ { r } } d r . } } \end{array}\tag{B.7}
$$

Throughout this subsection a hat denotes $\mu ^ { - 1 }$ times evaluation at $( \mu L , \mu Z )$ . The operators have explicit eigenfunctions. For $r > 0$ and $j \geq 0$ let $\varphi _ { r } ( L ) = \left( \sinh ( \mu ( r - L ) ) / \sinh ( \mu r ) \right) ^ { j }$ for $L \leq r$ and $\varphi _ { r } ( L ) = 0$ for $L > r$ . Since coth x − coth y = sinh $( y - x ) /$ (sinh x sinh y), one has sinh $^ { - j - 2 } ( \mu r ^ { \prime } ) \varphi _ { r } ( r ^ { \prime } ) = ( \coth \mu r ^ { \prime } - \coth \mu r ) ^ { j } / \sinh ^ { 2 } ( \mu r ^ { \prime } )$ , whose antiderivative in $r ^ { \prime }$ is $- ( \coth \mu r ^ { \prime } - \coth \mu r ) ^ { j + 1 } / ( ( j + 1 ) \mu )$ . Hence

$$
( j + 1 ) \mathcal { R } _ { j } \varphi _ { r } ( L ) = \mu \sinh ^ { j } ( \mu L ) \big ( \coth \mu L - \coth \mu r \big ) ^ { j + 1 } = \big ( b _ { \lambda } ( L ) - b _ { \lambda } ( r ) \big ) \varphi _ { r } ( L ) ,
$$

that is, $K _ { j \varphi _ { r } } = - b _ { \lambda } ( r ) \varphi _ { r }$ . After the scaling, $\Phi _ { r } ^ { m - 1 }$ is such an eigenfunction of $\phantom { + } \mathcal { K } _ { m - 1 }$ , with eigenvalue $- b _ { \lambda } ( r )$ , and conjugation by sinh $( \mu L )$ turns $\phantom { + } \mathcal { K } _ { m - 1 }$ into $A _ { m } ,$ so diferentiating under the integral gives the closed scalar equations

$$
\partial _ { Z } \widehat { h } _ { m } = 2 \mathcal { K } _ { m } \widehat { h } _ { m } , \qquad \partial _ { Z } \widehat { D } _ { m } = 2 \mathcal { A } _ { m } \widehat { D } _ { m } .\tag{B.8}
$$

The 00110 gap is $D _ { 1 } + E$ in family 0, where $\begin{array} { r } { E ( L , Z ) = \frac { 1 } { 2 } } \end{array}$ tanh $L - p ( L ) e$ <sup>−2Z</sup> <sup>coth</sup> <sup>L</sup> is the paired-state gap, and $D _ { m }$ in the other three families. For $m = 2 , 3$ the initial values $D _ { m } ( L , 0 )$ are exactly the strictly positive COMB gaps at the Region III corners, so $D _ { m } \in \mathrm { C M }$ by (B.8) and Lemma B.2. For $m = 1$ the initial datum is not a Region III gap; at $Z = 0$

$$
\begin{array} { r } { D _ { 1 } ( L , 0 ) = \frac { 1 } { 7 } \big [ 4 p ( L ) + 2 \sinh L \Lambda ( L ) - 2 \operatorname { t a n h } L \big ] , } \end{array}
$$

and both parts lie in CM after scaling: with the exit weights $\left( \mathrm { { A . 4 } } \right)$ ,

$$
\begin{array} { c } { { \displaystyle \frac { p ( \mu L ) } { \mu } = \frac { 1 } { 2 } \int _ { 0 } ^ { L } W _ { 1 } ( \lambda ; s , L ) ^ { 6 } \mathrm { s e c h } ^ { 2 } ( \mu s ) d s , } } \\ { { \displaystyle \frac { \sinh ( \mu L ) \Lambda ( \mu L ) - \operatorname { t a n h } ( \mu L ) } { \mu } = \int _ { L } ^ { \infty } W _ { 1 } ( \lambda ; L , r ) \mathrm { s e c h } ^ { 2 } ( \mu r ) d r , } } \end{array}
$$

and sech $. ( \mu r ) = 2 W _ { 1 } ( \lambda ; r , 2 r )$ . The first identity holds because both sides vanish at $L = 0$ and satisfy $\begin{array} { r } { \partial _ { L } f = \frac { 1 } { 2 } \operatorname { s e c h } ^ { 2 } ( \mu L ) - 6 \mu \coth ( \mu L ) f . } \end{array}$ , an equation the closed form (2.15) satisfies; the second because $\begin{array} { r } { \dot { \frac { d } { d s } } \big [ \Lambda ( s ) - \operatorname { s e c h } s \big ] = - \operatorname { s e c h } ^ { 2 } s / } \end{array}$ sinh s. Both right-hand sides are integrals of products of exit weights, hence lie in CM, so $D _ { 1 } ( L , 0 ) \in \mathrm { C M }$ , and $D _ { 1 } \in \mathrm { C M }$ by (B.8) and Lemma B.2.

The gaps for the controls 00000 and 00001 are $B _ { m } + h _ { m }$ and $B _ { m } + h _ { m } - \partial _ { Z } ^ { 2 } h _ { m }$ , where $B _ { m }$ is the four-expert background at the corner. Their normalized transforms solve the forced equation

$$
\partial _ { Z } \widehat { \gamma } = 2 \mathcal { K } _ { m } \widehat { \gamma } + F _ { m } ( \mu L ) , \qquad F _ { 1 } = 1 , \quad F _ { 2 } = C - S ^ { 2 } \Lambda , \quad F _ { 3 } = 1 + 3 S ^ { 2 } - 3 S ^ { 2 } C \Lambda ,\tag{B.9}
$$

with S = sinh $L , C = \cosh L ,$ the exact forcing identity being $- 2 \mathcal { K } _ { m } \big | _ { \lambda = 1 } B _ { m } = F _ { m }$ . Each $F _ { m }$ has a nonnegative time inverse: $F _ { 1 } = 1$ is an atom at time zero, and for $m = 2 , 3$

$$
F _ { m } ( \mu L ) = m \int _ { L } ^ { \infty } \left( \frac { \sinh ( \mu L ) } { \sinh ( \mu r ) } \right) ^ { m } \frac { \mu } { \sinh ( \mu r ) } F _ { m - 1 } ( \mu r ) d r ,\tag{B.10}
$$

a positive integral of products of elements of CM. At $Z = 0$ both gap families are certified Region III comparisons, so Lemma B.2 applies to (B.9) and puts both in CM; the same argument, with the background diference $B _ { \mathrm { I } } - B _ { \mathrm { I I } } = S C - S ^ { 3 } \Lambda$ and its representation $\begin{array} { r } { \mu ^ { - 1 } ( B _ { \mathrm { I } } - B _ { \mathrm { I I } } ) ( \mu L ) = 2 \int _ { L } ^ { \infty } ( \sinh ( \mu L ) / \sinh ( \mu r ) ) ^ { 3 } d r } \end{array}$ , covers the Region I corner. The control 00000 is the statement $U _ { \tau } \geq 0$ . The paired-state gap $E$ is treated the same way. All 64 of these identities are checked against independently generated corner Hessians in check\_ positive\_trace\_gaps.py.

The new ingredient is the following identity, which supplies three further positive traces at no extra certification cost.

Proposition B.3. Let $H _ { m } = \partial _ { Z } ^ { 2 } h _ { m }$ and $J _ { m } = H _ { m } - m D _ { m }$ for m = 1, 2, 3. Then

$$
\partial _ { Z } \widehat { J } _ { m } = 2 \mathcal { K } _ { m } \widehat { J } _ { m } + 2 m \mathcal { R } _ { m } \widehat { D } _ { m } ,\tag{B.11}
$$

and the initial data are

$$
\begin{array} { r } { J _ { 1 } ( L , 0 ) = 2 \mathsf { V } _ { 0 } ( L ) , \qquad J _ { 2 } ( L , 0 ) = 2 \mathsf { T } _ { 1 } ( L ) , \qquad J _ { 3 } ( L , 0 ) = 2 \mathsf { T } _ { 2 } ( L ) , } \end{array}\tag{B.12}
$$

twice the unnormalized Region III generators of Table 2. Consequently every $J _ { m }$ , and hence every $H _ { m } = J _ { m } + m D _ { m }$ , lies in CM.

Proof. From (B.8), $\partial _ { Z } \widehat { H } _ { m } = 2 \mathcal { K } _ { m } \widehat { H } _ { m }$ and $\partial _ { Z } \widehat { D } _ { m } = 2 \mathcal { A } _ { m } \widehat { D } _ { m }$ . Subtracting m times the second from the first and writing $\mathcal { K } _ { m } = - b _ { \lambda } + ( m + 1 ) \mathcal { R } _ { m } , \mathcal { A } _ { m } = - b _ { \lambda } + m \mathcal { R } _ { m }$

$$
2 \mathcal { K } _ { m } \widehat { H } _ { m } - 2 m \mathcal { A } _ { m } \widehat { D } _ { m } = 2 \mathcal { K } _ { m } \big ( \widehat { H } _ { m } - m \widehat { D } _ { m } \big ) + 2 m \mathcal { R } _ { m } \widehat { D } _ { m } ,
$$

which is (B.11); this uses only the linearity of $b _ { \lambda }$ and $\mathcal { R } _ { m }$ and is verified as such in audit\_ regions12\_completion.py. The initial values (B.12) are identities among Region III corner gaps: $J _ { m } ( L , 0 )$ is a combination of gaps of family 0, 1 or II at $Z = 0$ , hence, by the remark after (B.5), the same combination of gaps at the corner $\omega _ { m - 1 }$ , where the corner decompositions of Appendix A give (B.12). The same script checks them again from the $Z = 0$ moment expansions, separately in the four-expert background, the continuous spectral kernel and the moving-endpoint atom. Since $\mathcal { R } _ { m } \widehat { D } _ { m }$ is a nonnegative combination of elements of CM, Lemma B.2 applies to (B.11). □

## B.3 The derivative cascades and the scalar certificates

Nine corner gaps are not directly positive combinations of the traces above. Three of them, the critical gaps

$$
C _ { 1 } = G _ { 0 , 0 1 0 0 1 } , \qquad C _ { 2 } = G _ { 1 , 0 1 1 1 0 } , \qquad C _ { 3 } = G _ { \mathrm { I I , 0 1 1 1 0 } } ,\tag{B.13}
$$

vanish identically at $Z = 0 ,$ so it sufices to prove $\partial _ { Z } \gamma \geq 0 ;$ six further gaps have nonnegative initial values, and the same device applies.

Write the continuous spectral kernel of such a gap as $S ^ { m } P ( \alpha , c _ { r } )$ with $S =$ sinh $L _ { ; }$ $\alpha = \coth L$ and deg<sub>α</sub> $P \leq m ,$ , and set $Q = - 2 c _ { r } P$ , so that $\begin{array} { r } { \partial _ { Z } G = S ^ { m } \int _ { L } ^ { \infty } \rho ( r ) Q e ^ { - 2 Z c _ { r } } d r } \end{array}$ up to the endpoint atom. Define

$$
N _ { \ell } ( L , Z ) = \frac { S ^ { m } } { \ell ! } \int _ { L } ^ { \infty } \rho ( r ) \left( \alpha - c _ { r } \right) ^ { \ell } \partial _ { \alpha } ^ { \ell } Q ( \alpha , c _ { r } ) e ^ { - 2 Z c _ { r } } d r , \qquad 0 \le \ell \le m ,\tag{B.14}
$$

equivalently $\begin{array} { r } { N _ { \ell } = \sum _ { i > \ell } \binom { j } { \ell } f _ { j } } \end{array}$ where $f _ { j }$ is the contribution of the jth coeficient in $Q =$ $\begin{array} { r } { \sum _ { j } q _ { j } ( c _ { r } ) ( \alpha - c _ { r } ) ^ { j } } \end{array}$ . Then $N _ { 0 } = \partial _ { Z } G$ and $N _ { m + 1 } = 0$

Proposition B.4. The scaled transforms satisfy the triangular system

$$
\partial _ { Z } \widehat { N } _ { \ell } = 2 \mathcal { T } _ { \ell + 1 , m } \widehat { N } _ { \ell } + 2 ( \ell + 1 ) \mathcal { R } _ { m } \widehat { N } _ { \ell + 1 } , \qquad N _ { m + 1 } = 0 .\tag{B.15}
$$

Here $\widehat { N } _ { \ell }$ carries no $\mu ^ { - 1 }$ prefactor, diferentiation of the scaled gap having cancelled it. For $m = 1$ the moving-endpoint datum $\begin{array} { r } { T ( L , Z ) = \frac { 1 } { 2 } } \end{array}$ coth $L p ( L ) e$ <sup>−2Z</sup> <sup>coth</sup> <sup>L</sup> enters $N _ { 0 }$ with sign + and $N _ { 1 }$ with sign −, so that the second equation acquires the positive source $4 \mathcal { R } _ { 1 } \widehat { T }$ , and $\partial _ { Z } \widehat { T } = - 2 b _ { \lambda } \widehat { T }$

Proof. Writing the integrand of $f _ { j }$ $S ^ { m - j } q _ { j } ( c _ { r } ) \Phi _ { r } ( L ) ^ { j }$ and conjugating the eigenrelation for $\kappa _ { j }$ by $S ^ { m - j }$ gives $\partial _ { Z } \widehat { f _ { j } } = 2 \mathcal { T } _ { j + 1 , m } \widehat { f _ { j } }$ . Summing with weights $\textstyle { \binom { j } { \ell } }$ and using the binomial identity $\begin{array} { r } { ( j + 1 ) \binom { j } { \ell } = ( \ell + 1 ) \big [ \binom { j } { \ell } + \binom { j } { \ell + 1 } \big ] } \end{array}$ , which is Pascal’s rule in the form $\begin{array} { r } { ( \ell + 1 ) { \binom { j + 1 } { \ell + 1 } } = } \end{array}$ $( j + 1 ) \binom { j } { \ell }$ , gives

$$
\partial _ { Z } \widehat { N } _ { \ell } = - 2 b _ { \lambda } \widehat { N } _ { \ell } + 2 ( \ell + 1 ) \mathcal { R } _ { m } \big ( \widehat { N } _ { \ell } + \widehat { N } _ { \ell + 1 } \big ) ,
$$

which is (B.15). The polynomial degrees and the termination $N _ { m + 1 } = 0$ are checked in the data-generation scripts. □

Since every coeficient in (B.15) is nonnegative, Lemma B.2 propagates positivity upward from $\ell = m \mathrm { \ t o \ } \ell = 0$ , provided the $m + 1$ initial profiles $N _ { \ell } ( \cdot , 0 )$ lie in CM. Counting these for the three critical gaps and for the six further gaps

$$
G _ { 0 , 0 0 0 1 1 } , \ G _ { 1 , 0 0 0 1 1 } , \ G _ { 1 , 0 0 1 0 1 } , \ G _ { \mathrm { I } , 0 0 0 1 1 } , \ G _ { \mathrm { I } , 0 0 1 0 1 } , \ G _ { \mathrm { I } , 0 0 1 0 1 } , \ G _ { \mathrm { I I } , 0 0 0 1 1 }
$$

gives 10 and 20 one-variable profiles respectively.

Each of the thirty derivative profiles has an exact expansion

$$
N ( L ) = \sum _ { k > 0 , \ k \equiv p ( 2 ) } \left[ a ( k ) + L b ( k ) \right] e ^ { - k L } , \qquad p \equiv m - 1 { \pmod { 2 } } ,\tag{B.16}
$$

with a odd and $b$ even rational functions whose poles lie at integers of the opposite parity and have order at most two. The derivation uses the trace series (3.18) together with the identity

$$
\operatorname { t a n h } L e ( L ) = e ^ { \prime } ( L ) + { \frac { 3 I ( L ) } { \cosh L \sinh ^ { 5 } L } } ,\tag{B.17}
$$

which follows from (2.8), and it audits every exceptional Laurent mode; the apparent alternating contributions cancel identically. Setting

$$
A ( k ) = k a ( k ) - b ( k ) , \qquad C ( k ) = 2 k ^ { 2 } b ( k ) , \qquad \xi = \frac { L ^ { 2 } } { 4 \tau } ,\tag{B.18}
$$

both A and $C$ are even, and

$$
\mathcal { L } ^ { - 1 } \big [ N ( L \mu ) \big ] ( \tau ) = \frac { L } { 2 \sqrt { \pi } \tau ^ { 3 / 2 } } \cdot \frac { 1 } { 2 } \sum _ { k \equiv p ( 2 ) } \big [ A ( k ) + \xi C ( k ) \big ] e ^ { - k ^ { 2 } \xi } ,\tag{B.19}
$$

which is (4.25). So each profile is again a lattice Gaussian series, and the three ranges are treated as in Appendix A, one derivative further because the polynomial parts now have degree four. For $0 < \xi \leq 1 / 1 0 0$ , Poisson summation and the identities (A.13), (A.14) give a principal expression $\begin{array} { r } { M ( \xi ) = p _ { 0 } \frac { \sqrt { \pi } } { 4 \sqrt { \xi } } + \pi ^ { 2 } \sum _ { a } d _ { a } e ^ { - a ^ { 2 } \xi } + \sum _ { a } q _ { a } Q _ { a } . } \end{array}$ , the zero image (A.11) now surviving because the polynomial parts no longer cancel it, with all $q _ { a } \leq 0$ , and the elementary bounds $\sqrt { \pi } > 3 / 2 , \pi ^ { 2 } < 1 0 , Q _ { a } \leq \sqrt { \xi }$ give $\begin{array} { r } { \sqrt { \xi } M ( \xi ) \geq \frac { 3 } { 8 } p _ { 0 } + \sum _ { a } } \end{array}$ min $\begin{array} { r } { ( d _ { a } , 0 ) + \frac { 1 } { 1 0 0 } \sum _ { a } q _ { a } > \frac { 1 } { 2 0 } } \end{array}$ the discarded images are bounded by $1 0 ^ { 8 } \xi ^ { - 1 7 / 2 } e ^ { - 9 / ( 4 \xi ) } < 1 0 ^ { - 4 0 } / \sqrt { \xi }$ , by the estimate of Lemma A.3 with $h \leq 4$ and $\sigma _ { 4 } = 2 0 2 5 / 1 6$ . For $1 / 1 0 0 \le \xi \le 1$ , the coeficient bound (A.16) applied to A and $C _ { i }$ , the tail bound (A.17), and an order-six interval Taylor bound on rational cells, as in (A.18), account for 251 cells for nine of the ten critical profiles, the tenth being the endpoint datum treated below, and 513 for the twenty further ones. For $\xi \ge 1$ first-mode domination $\left( \mathrm { A . 1 9 } \right)$ applies; for the ten critical profiles this uses in addition an exact Cauchy bound $| a _ { k } | , | b _ { k } | \le C 2 ^ { k }$ , obtained by evaluating each rational prefactor on $\begin{array} { r } { | q | = \frac { 1 } { 2 } } \end{array}$ with $q = e ^ { - L }$ and using $| A _ { e } | \leq 1 0 7 / 5 4 < 2 , | B _ { e } | \leq 2 2 / 8 1 < 1$ , | arctan q| ≤ log $3 / 2 < 1$ and |2 artanh $q | \leq \log 3 < 2$ , which in turn follow from $| A _ { n } | \leq 2 n ^ { 2 }$ and $| B _ { n } | \leq n ^ { 3 } / 6$

Two profiles need a separate treatment. The first is the endpoint datum T. Its zero Poisson image cancels identically and its partial fractions are empty, so Γ is a sum over the nonzero images alone. Each such image is a positive multiple of an explicit rational function of $\xi$ and $B = \pi ^ { 2 } \ell ^ { 2 } / 4$ ; substituting $\begin{array} { r } { B = \frac { 9 } { 4 } + y } \end{array}$ and $\xi = 1 / ( 1 + t )$ and clearing the positive denominator $3 0 7 2 \xi ^ { 7 }$ yields a polynomial in $( y , t )$ with strictly positive coeficients. Since $\pi ^ { 2 } / 4 > 9 / 4$ , this proves every nonzero image positive for $0 < \xi \le 1$ , hence $\Gamma > 0$ there. The second is the last boundary datum. At the Region I corner with $Z = 0$ the only new scalar initial gap is $C _ { * } ( L ) = G _ { \mathrm { I , 0 0 0 1 1 } } ( L , 0 )$ , and the moment recursion gives the exact identity

$$
C _ { * } ( L ) = 3 \mathsf { A } _ { 2 } ( L ) - 3 \mathsf { B } _ { 2 } ( L ) + \sinh L \cosh L - \sinh ^ { 3 } L \Lambda ( L ) .\tag{B.20}
$$

The correction sinh L cosh $L - \sinh ^ { 3 } L \Lambda ( L )$ has a positive inverse, but the negative coeficient of $\mathsf { B } _ { 2 }$ prevents concluding directly. Its exact even-lattice expansion has

$$
a ( k ) = \frac { ( k ^ { 2 } - 4 ) \left( 5 k ^ { 1 0 } - 2 3 0 k ^ { 8 } + 3 1 6 9 k ^ { 6 } - 1 8 9 7 6 k ^ { 4 } + 3 9 2 1 6 k ^ { 2 } - 4 8 3 8 4 \right) } { 1 7 9 2 ( k ^ { 2 } - 1 ) ^ { 2 } ( k ^ { 2 } - 9 ) ^ { 2 } } ,
$$

and the certificate of Appendix A applies verbatim: the small-scale principal lower bound is $\frac { 3 3 4 7 0 9 1 } { 2 8 6 7 2 0 0 } \sqrt { \xi }$ , the compact range takes 45 cells, and for $\xi \ge 1$ the constant zero mode $2 / 3$ dominates a remainder below $7 . 7 1 6 7 \times 1 0 ^ { - 8 }$

## B.4 All sixty-four corner controls

It remains to express every corner gap as a nonnegative combination of certified objects. Besides the identically zero gaps and the gaps already treated, the decompositions are

<table><tr><td>Family</td><td>Control</td><td>Positive decomposition</td></tr><tr><td>0</td><td>00110</td><td> $D _ { 1 } + E$ </td></tr><tr><td>0</td><td>00010</td><td> $G _ { 0 , 0 0 0 1 1 } + \textstyle { \frac { 1 } { 2 } } H _ { 1 } + \textstyle { \frac { 1 } { 2 } } D _ { 1 }$ </td></tr><tr><td>0</td><td>01000</td><td> $C _ { 1 } + { \textstyle { \frac { 1 } { 2 } } } J _ { 1 }$ </td></tr><tr><td>1</td><td>00010</td><td> $G _ { 1 , 0 0 0 1 1 } + D _ { 2 } + { \textstyle \frac { 1 } { 2 } } H _ { 2 }$ </td></tr><tr><td>1</td><td>00100</td><td> $G _ { 1 , 0 0 1 0 1 } + { \textstyle \frac { 1 } { 2 } } H _ { 2 }$ </td></tr><tr><td>1</td><td>01111</td><td> $C _ { 2 } + { \frac { 1 } { 2 } } J _ { 2 }$ </td></tr><tr><td>I</td><td>00010</td><td> $G _ { \mathrm { I , 0 0 0 1 1 } } + \textstyle { \frac { 3 } { 2 } } D _ { 3 } + \textstyle { \frac { 1 } { 2 } } H _ { 3 }$ </td></tr><tr><td>I</td><td>00100</td><td> $G _ { \mathrm { I , 0 0 1 0 1 } } + D _ { 3 } + { \textstyle { \frac { 1 } { 2 } } } J _ { 3 }$ </td></tr><tr><td>II</td><td>00010</td><td> $G _ { \mathrm { I I , 0 0 0 1 1 } } + \textstyle { \frac { 1 } { 2 } } D _ { 3 } \stackrel { \_ } { + } \textstyle { \frac { 1 } { 2 } } H _ { 3 }$ </td></tr><tr><td>II</td><td>01111</td><td> $C _ { 3 } + { \textstyle { \frac { 1 } { 2 } } } J _ { 3 }$ </td></tr></table>

together with $H _ { m } = G _ { f , 0 0 0 0 0 } - G _ { f , 0 0 0 0 1 }$ and $D _ { m } = G _ { f , 0 0 1 1 0 }$ (family 0 excepted, where $D _ { 1 } = G _ { 0 , 0 0 1 1 0 } - G _ { 0 , 0 0 1 1 1 } )$ . The script audit\_regions12\_completion.py verifies 106 exact identities covering all 64 physical corner entries, checking the four-expert background, the continuous spectral kernel and the moving-endpoint atom separately, and confirming that every combination has nonnegative coeficients.

Proof of Theorem 4.11. Proposition B.1 reduces the claim to the four corner families; the decompositions above, Proposition B.3 and the cascades of Proposition B.4 with the certified initial profiles of Appendix B.3 put every corner gap in CM, and Lemma 4.12 turns this into the inequality for $U .$ . For the COMB statement we use its representative 01010. Its gaps in the corner families 0, 1, I and II are $0 , D _ { 2 } , D _ { 3 }$ and $D _ { 3 }$ , its replacement at the nonphysical corner is a vanishing gap, and its Green coeficient is zero, so Proposition B.1 gives, in both regions, the exact transform

$$
\widehat { \gamma } _ { C } = W _ { 1 } ( b ) \big [ W _ { 0 } ( c ) \widehat { D } _ { 2 } ( L , Z ) + W _ { 1 } ( c ) \widehat { D } _ { 3 } ( L , Z ) \big ] .
$$

If $b = z _ { 2 } = 0$ then $W _ { 1 } ( b ) = 0 \quad$ , so the COMB gap has vanishing transform and, being continuous in $\tau ,$ vanishes. Let $b > 0$ . By (B.8) and the second statement of Lemma B.2, $\widehat { D } _ { m } ( L , Z ) - e ^ { - 2 Z b _ { \lambda } ( L ) } \widehat { D } _ { m } ( L , 0 ) \in \mathrm { C M }$ for $m = 2 , 3$ , and every weight lies in CM. Hence, if $c < L ,$

$$
\widehat { \gamma } _ { C } - \widehat { \kappa } \widehat { D } _ { 2 } ( L , 0 ) \in \mathrm { C M } , \qquad \widehat { \kappa } = W _ { 1 } ( b ) W _ { 0 } ( c ) e ^ { - 2 Z b _ { \lambda } ( L ) } ,
$$

and if $c = L$ the same holds with $W _ { 1 } ( c ) = 1$ in place of $W _ { 0 } ( c )$ and ${ \widehat { D } } _ { 3 }$ in place of $\widehat { D } _ { 2 }$ . In either case κb is a product of elements of CM, hence the transform of a nonnegative measure, and it decays no faster than $e ^ { - A { \sqrt { \lambda } } }$ : for $0 < s \le L$ we have sinh $. ( s \mu ) / \sinh ( L \mu ) \geq \frac { 1 } { 2 } e ^ { - ( L - s ) \mu }$ as soon as $e ^ { - 2 s \mu } \leq { \frac { 1 } { 2 } }$ , which applies to $W _ { 1 } ( b )$ with $s = b$ and to $W _ { 0 } ( c )$ with $s = L - c .$ , while coth $y \le 1 + 1 / y$ gives $e ^ { - 2 Z b _ { \lambda } ( L ) } > e ^ { - 2 Z / L } e ^ { - 2 Z \mu }$ . The initial values $\widehat { D } _ { 2 } ( L , 0 )$ and $\widehat { D } _ { 3 } ( L , 0 )$ are the transformed COMB gaps at the Region III corners $\omega _ { 1 }$ and $\omega _ { 2 }$ , that is, the generators $\mathsf { D } _ { 1 }$ and $\mathsf { B } _ { 2 }$ , whose inverses $\Gamma ( \xi ) / \sqrt { \pi \tau }$ are continuous and strictly positive for $\tau > 0$ by Appendix A. The second statement of Lemma 4.12, applied to the COMB gap, which is continuous in τ with the growth that Lemma 4.1 provides, therefore gives strict positivity. Since $z _ { 2 } = ( y _ { 1 } + y _ { 3 } ) / 2$ , the equality set is $y _ { 1 } = y _ { 3 } = 0$ □

## C Independent numerical checks

The checks in this appendix are not part of the proof. They test the formula and its final claims by methods that bypass the reductions of Appendices A and B, and they tie the construction back to the stationary solution. All floating-point computations use either 40-digit arithmetic (mpmath) or double precision with the analytic term-by-term derivatives; the scripts are in the derivation/ directory of the supplement [5].

The expansions of Section 3 agree with what they expand: the Region III series with the closed form (3.15) to 30 digits at six configurations, the Regions $\mathrm { I / I I }$ series with direct quadrature of (3.26) to 55 digits at five configurations, the four-expert expansion of Lemma 3.8 with (2.13) to 19 digits at a point close to the four-coordinate diagonal, where the convergence is slowest, and the trace series as recorded in Remark 3.11. The function U behaves as it should. Central diferences in 40-digit arithmetic give $\begin{array} { r } { | U _ { \tau } - \frac { 1 } { 2 } D _ { \mathbf { v } _ { * } } ^ { 2 } U | \lesssim 1 0 ^ { - 2 1 } } \end{array}$ , the noise floor of the diference quotient, at seven points covering the three regions and $\tau \in \{ 0 . 4 , 1 , 2 . 5 \}$ ; $U ( \tau , x ) - \operatorname* { m a x } _ { i } x _ { i }$ falls below $1 0 ^ { - 1 4 }$ by $\tau = 1 0 ^ { - 2 }$ at points with $x _ { 1 } > x _ { 2 }$ and decays like $\sqrt { \tau }$ where the maximum is attained twice; $U ( \ell ^ { 2 } \tau , \ell x ) = \ell U ( \tau , x )$ holds to machine precision; and at twelve random points of the interface $a _ { 1 } = 0$ the two regional expressions agree in value, in $U _ { \tau }$ and in all sixteen curvatures $D _ { \mathbf { v } } ^ { 2 } U$ to $9 \times 1 0 ^ { - 1 6 }$

The most stringent global check is the transform identity (3.3) and its diferentiated form (4.1), tested by numerical quadrature in τ against an independent evaluation of the stationary solution u from the companion paper. At $\lambda = 1$ and three points spanning Regions I, II and III,

$$
\int _ { 0 } ^ { \infty } e ^ { - \tau } U ( \tau , x ) d \tau = u ( x ) \quad \mathrm { a n d } \quad \int _ { 0 } ^ { \infty } e ^ { - \tau } \bigl ( D _ { \mathbf { v _ { * } } } ^ { 2 } U - D _ { \mathbf { v } _ { C } } ^ { 2 } U \bigr ) ( \tau , x ) d \tau = \bigl ( D _ { \mathbf { v _ { * } } } ^ { 2 } u - D _ { \mathbf { v } _ { C } } ^ { 2 } u \bigr ) ( x )
$$

agree to 12 significant digits, the accuracy of the quadrature.

The next check does not use the derivation of the formula at all. An independent monotone finite-diference solution of (2.1) on the gap lattice, of the kind analyzed in [8], run with the full Bellman maximum over all $\mathbf { v } \in \{ 0 , 1 \} ^ { 5 }$ , was compared with the formula at $\tau = 1$ at seventeen points. The scheme converges at second order in the mesh; after one Richardson step from $\begin{array} { r } { h = \frac { 1 } { 8 } } \end{array}$ and $\textstyle h = { \frac { 1 } { 1 6 } }$ the largest discrepancy is $6 \times 1 0 ^ { - 6 }$ , consistent with the residual discretization error. At the symmetric point, Richardson gives 0.6921242 against the exact value $4 5 \pi ^ { 3 / 2 } / ( 2 5 6 { \sqrt { 2 } } ) = 0 . 6 9 2 1 2 1 5$ of (2.22). The discrete solve also confirms that fixing the policy $\mathbf { v } _ { * }$ and taking the full Bellman maximum give the same value to $1 0 ^ { - 1 3 }$

Finally, we evaluated the five-expert Hamiltonian inequality directly from the series, which bypasses the reduction of Appendices $\mathrm { A }$ and B. By the exact scaling of Theorem 2.1(iii) it sufices to scan at $\tau = 1$ , since $D _ { \mathbf { v } } ^ { 2 } U ( \tau , x ) = \tau ^ { - 1 / 2 } D _ { \mathbf { v } } ^ { 2 } U ( 1 , x / \sqrt { \tau } )$ The script scan.py draws 1500 points of the ordered sector with coordinates uniform in [0, 1.6] (numpy default generator, seed 20260914), of which 30% are given one forced collision $x _ { i } = x _ { i + 1 }$ and 20% are placed on $S = \{ x _ { 1 } = x _ { 2 } , \ x _ { 3 } = x _ { 4 } \}$ , and evaluates $U _ { \tau }$ and all sixteen curvatures $D _ { \mathbf { v } } ^ { 2 } U$ in double precision from the analytic term-by-term derivatives. It discards a point when $\vartheta > 1 0$ , with $\vartheta$ as in (C.1) below, or when dropping the last eight modes moves a curvature by more than $1 0 ^ { - 1 0 }$ relative. This discarded 34 points and left 1466: 611 in Region I, 643 in Region II and 212 in Region III, points on the interface $z _ { 3 } = 0$ being counted in Region II, of which 737 are strictly ordered, 446 have one forced collision and 283 lie on $S .$ Over these points max $| U _ { \tau } - { \textstyle \frac { 1 } { 2 } } D _ { \mathbf { v } _ { \ast } } ^ { 2 } U | = 5 . 0 \times 1 0 ^ { - 1 4 }$ , and $\mathbf { v } _ { * }$ attains the maximum at all 1466 of them. The largest apparent excess, max $\mathbf { v } ( D _ { \mathbf { v } } ^ { 2 } U - D _ { \mathbf { v } _ { \ast } } ^ { 2 } U ) = 5 . 6 \times 1 0 ^ { - 1 1 }$ , occurs at $x = ( 0 . 1 7 9 3 , 0 . 1 3 0 2 , 0 . 1 3 0 2 , 0 . 0 4 4 3 , 0 )$ for $\mathbf { v } = ( 1 , 1 , 0 , 0 , 0 )$ , where $x _ { 2 } = x _ { 3 }$ and the two curvatures are equal exactly, by the symmetry explained below. Among the pairs of a point and a control that no such symmetry ties, the largest excess is $1 . 9 \times 1 0 ^ { - 1 1 }$ , for $( 1 , 1 , 0 , 0 , 0 )$ at $x = ( 0 . 2 8 5 3 , 0 . 0 2 1 4 , 0 . 0 1 0 7 , 0 , 0 )$ , where $a _ { 4 }$ is small and the Region III series needs many modes; recomputing that point in 40-digit arithmetic gives curvatures that agree to 14 digits, while the stationary gap there is $3 . 7 \times 1 0 ^ { - 4 }$

The ties are consistent with Section 4.3. On S, where Proposition 4.7 makes the COMB gap vanish identically, the computed $| \Delta |$ still reaches $5 . 4 \times 1 0 ^ { - 1 0 }$ , so a tie tolerance must sit above that figure. We use 10<sup>−9</sup> relative, and the last column records the count at $1 0 ^ { - 1 2 }$ for comparison.

<table><tr><td>control</td><td>total</td><td>I</td><td>II</td><td>III</td><td>at  $1 0 ^ { - 1 2 }$ </td><td>symmetric</td></tr><tr><td> $( 1 , 0 , 1 , 0 , 0 ) = \mathbf { v } _ { * }$ </td><td>1466</td><td>611</td><td>643</td><td>212</td><td>1466</td><td>1466</td></tr><tr><td>(1, 0, 0, 1, 1)</td><td>894</td><td>611</td><td>283</td><td>0</td><td>890</td><td>394</td></tr><tr><td>(1, 0,0, 1, 0)</td><td>856</td><td>1</td><td>643</td><td>212</td><td>855</td><td>397</td></tr><tr><td>(1, 1,0, 0,0)</td><td>391</td><td>177</td><td>170</td><td>44</td><td>295</td><td>113</td></tr><tr><td> $( 1 , 0 , 1 , 0 , 1 ) = \mathbf { v } _ { C }$ </td><td>284</td><td>1</td><td>283</td><td>0</td><td>279</td><td>283</td></tr><tr><td>(1, 0, 0, 0, 1)</td><td>212</td><td>0</td><td>0</td><td>212</td><td>212</td><td>0</td></tr></table>

The three ties of Corollary 4.6 appear with exactly the predicted region coverage: $( 1 , 0 , 0 , 1 , 1 )$ at all 611 points of Region I, (1, 0, 0, 1, 0) at all $6 4 3 + 2 1 2$ points of Regions II and III, and (1, 0, 0, 0, 1) at the 212 points of Region III and nowhere else. Many ties are forced by a symmetry of the state: if a permutation P of the coordinates fixes $x ,$ then $\nabla ^ { 2 } U ( \tau , x ) =$ $P ^ { T } \nabla ^ { 2 } U ( \tau , x ) P$ by the permutation invariance of $U ,$ so $D _ { P \mathbf { v } _ { * } } ^ { 2 } U = D _ { \mathbf { v } _ { * } } ^ { 2 } U$ at $x ,$ for the finitehorizon and for the stationary solution alike; the last column counts these ties. Of the 284 COMB ties, 283 are the points of $S _ { i }$ which lie on the interface $z _ { 3 } = 0$ and are counted in Region II; there the exchanges of $x _ { 1 }$ with $x _ { 2 }$ and of $x _ { 3 }$ with $x _ { 4 }$ carry $\mathbf { v } _ { * }$ to the complement of $( 1 , 0 , 0 , 1 , 1 )$ , to $( 1 , 0 , 0 , 1 , 0 )$ and to $\mathbf { v } _ { C }$ , which accounts for the 283 ties of $( 1 , 0 , 0 , 1 , 1 )$ in Region II, and the exchange of $x _ { 2 }$ with $x _ { 3 }$ carries $\mathbf { v } _ { * }$ to $( 1 , 1 , 0 , 0 , 0 )$ , which accounts for its 113 ties at points with $x _ { 2 } = x _ { 3 }$ . The remaining 278 ties of $( 1 , 1 , 0 , 0 , 0 )$ are numerical: at each of these points the stationary gap $D _ { \mathbf { v } _ { * } } ^ { 2 } u - D _ { ( 1 , 1 , 0 , 0 , 0 ) } ^ { 2 } u .$ , evaluated with geom.py, lies between $2 . 9 \times 1 0 ^ { - 5 }$ and 0.30, so by (4.1) the finite-horizon gap is not identically zero in $\tau ,$ and the tie is a gap below the tolerance at $\tau = 1$ (Remark 4.8). The one tie of $( 1 , 0 , 0 , 1 , 0 )$ in Region I is at $x = ( 1 . 5 4 9 6 , 1 . 5 4 9 6 , 0 . 0 9 0 4 , 0 . 0 7 8 2 , 0 )$ , where $z _ { 3 } = - 0 . 0 0 9$ places the point just inside Region I, next to the interface on which Corollary 4.6 makes this control tie $\mathbf { v } _ { * } ;$ the gap there is $2 \times 1 0 ^ { - 1 0 }$ . Of S the COMB gap was resolved and positive at all 1183 points, and negative at none; its one tie of S at the tolerance $1 0 ^ { - 9 }$ is resolved positive at $1 0 ^ { - 1 2 }$ , as Theorem 2.2 requires.

The series converge for every $a _ { 4 } > 0$ , respectively $z _ { 1 } > 0$ , but the number of modes needed grows as the smallest relevant length scale shrinks. In Region III one needs $m \lesssim \sqrt { \tau } / a _ { 4 }$ modes, since $\mathcal { H } _ { 0 } ( X _ { m , \epsilon } , \tau )$ decays like $e ^ { - m ^ { 2 } a _ { 4 } ^ { 2 } / \tau }$ , and in Regions I and II one needs $n \lesssim \sqrt { \tau } / z _ { 1 }$ In Regions I and II the expansion of $e ^ { - 2 z _ { 4 } \coth t }$ in powers of $z _ { 4 }$ also has intermediate terms of size $e ^ { \vartheta }$

$$
\vartheta = { \frac { 4 z _ { 4 } e ^ { - 2 z _ { 1 } } } { 1 - e ^ { - 2 z _ { 1 } } } } ,\tag{C.1}
$$

so that double precision loses about $\vartheta /$ ln 10 digits on approach to the face $x _ { 1 } + x _ { 2 } = x _ { 3 } + x _ { 4 }$ where $z _ { 1 } = 0$ . Near that face one should either work in extended precision or evaluate the integral in (3.26) directly under the inverse transform, and on C itself the quadrature (2.21) replaces the series.

## References

[1] Y. Abbasi-Yadkori, P. L. Bartlett, and V. Gabillon. Near minimax optimal players for the finite-time 3-expert prediction problem. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017.

[2] E. Bayraktar, I. Ekren, and N. Kolliopoulos. Prediction with five experts and geometric stopping: a probabilistic construction and analytic verification. Preprint, arXiv:2609.17986, 2026.

[3] E. Bayraktar, I. Ekren, and X. Zhang. Finite-time 4-expert prediction problem. Communications in Partial Diferential Equations, 45(7):714–757, 2020.

[4] E. Bayraktar, I. Ekren, and Y. Zhang. On the asymptotic optimality of the comb strategy for prediction with expert advice. Ann. Appl. Probab., 30(6):2517–2546, 2020.

[5] J. Calder and N. Drenska. Certificates and Lean 4 formalization for the finite-horizon five-expert prediction problem, 2026. DOI:10.5281/zenodo.23012438.

[6] J. Calder and N. Drenska. An explicit solution of the five-expert prediction PDE and the exact optimality set of COMB. Preprint, arXiv:2609.14892, 2026.

[7] J. Calder and N. Drenska. A Lean 4 formalization of the five-expert prediction PDE: certificates, regularity, and the optimality set of COMB, version 0.1.12, 2026. DOI:10.5281/zenodo.22821097.

[8] J. Calder, N. Drenska, and D. Mosaphir. Numerical solution of a PDE arising from prediction with expert advice. European J. Appl. Math., 37(1):96–122, 2026.

[9] N. Cesa-Bianchi and G. Lugosi. Prediction, learning, and games. Cambridge University Press, 2006.

[10] Z. Chase. Experimental evidence for asymptotic non-optimality of comb adversary strategy, 2019. arXiv:1912.01548.

[11] T. M. Cover. Behavior of sequential predictors of binary sequences. In Transactions of the Fourth Prague Conference on Information Theory, Statistical Decision Functions, Random Processes (Prague, 1965), pages 263–272, Prague, 1967. Academia.

[12] N. Drenska. A PDE Approach to a Prediction Problem Involving Randomized Strategies. PhD thesis, New York University, 2017.

[13] N. Drenska and R. V. Kohn. Prediction with expert advice: a PDE perspective. J. Nonlinear Sci., 30:137–173, 2020.

[14] N. Gravin, Y. Peres, and B. Sivan. Towards optimal algorithms for prediction with expert advice. In Proceedings of the Twenty-Seventh Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2016), pages 528–547. SIAM, 2016.