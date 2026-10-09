# RobustLDS: Learning linear dynamical systems under adversarial corruptions

Aravinda Kanchana Ruwanpathirana

Hemant Tyagi

Division of Mathematical Sciences, SPMS, NTU Singapore 637371 {aravinda.r, hemant.tyagi}@ntu.edu.sg

## Abstract

We consider the problem of learning linear dynamical systems under adversarial contamination from a single trajectory of length T. While identification of linear dynamical systems itself is well-studied, the problem of robust system identification under adversarial contamination is relatively less explored. In this work, we study the setting where a fraction of the T observations are contaminated by adversarial outliers. We propose diferent estimators based on relaxations of least-trimmed squares along with an alternating minimization algorithm. Furthermore, we also propose two estimators which exploit the group-sparsity (through penalization/hard-constraints) of the outliers. For the estimator with group-sparse penalty, we derive non-asymptotic error bounds which establish its robustness to outliers. We also show empirically that the proposed estimators work well in practice.

## 1 INTRODUCTION

Regression is a fundamental problem in machine learning where there exists a set of covariates $\mathcal { X } \subset \mathbb { R } ^ { d }$ , set of labels $\mathcal { V } \subset \mathbb { R }$ and a distribution D on $\mathcal { X } \times \mathcal { V }$ , from which a learner gets $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m } \stackrel { i . i . d } { \sim } \mathcal { D }$ and attempts to learn the underlying model parameters. For instance in the case of linear regression where $y _ { i } = x _ { i } ^ { \top } \beta ^ { * } + \varepsilon _ { i }$ ( $\varepsilon _ { i }$ is noise), minimax error rates for estimating $\beta ^ { * }$ are now well known even in the high-dimensional regime $( m \ll d )$ (Ye and Zhang, 2010; Raskutti et al., 2011; Lounici et al., 2011).

However, when a fraction of the data $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m }$ is contaminated by an adversary, estimating $\beta ^ { * }$ becomes a more challenging task. This is known as robust regression under adversarial contamination and has been extensively studied in recent years under diferent models of contamination, e.g., inlier-outlier model (Merad and Ga¨ıfas, 2023a,b), Huber contamination (Balakrishnan et al., 2017) and strong contamination (Liu et al., 2020, 2019). A wide range of methods have been explored, $\mathrm { e . g . } _ { \mathrm { } } . _ { }$ , robust gradient-descent (Merad and Ga¨ıfas, 2023a,b; Liu et al., 2020, 2019; Cherapanamjeri et al., 2020) with gradient estimators using Median-of-Means, Trimmed Mean, robust PCA and SDP formulations; least-trimmed squares estimators (Bhatia et al., 2015, 2017) through hard-thresholding; penalized Huber M-estimators (Pensia et al., 2025; Dalalyan and Thompson, 2019; Sasai and Fujisawa, 2020; Sasai, 2022; Sasai and Fujisawa, 2025; Maurya et al., 2026); other approaches such as least-absolutedeviation (Karmalkar and Price, 2019; Pensia et al., 2025) and Dantzig selectors (Chen et al., 2013).

While the aforementioned works provide strong theoretical results when the clean data is i.i.d, whether they can be extended to the setting where the clean data exhibits dependencies is relatively less understood.

Learning linear dynamical systems (LDS). Let $A ^ { * } \in \mathbb { R } ^ { n \times n }$ be an unknown matrix, and consider the LDS<sup>1</sup> (with $x _ { 0 } = 0 )$

$$
x _ { t + 1 } = A ^ { * } x _ { t } + \eta _ { t + 1 } , \quad t = 0 , \ldots , T - 1 ,\tag{1.1}
$$

where $\eta _ { t } \stackrel { i . i . d . } { \sim } \mathcal { N } ( 0 , I _ { n } )$ refers to the process noise (the “excitation” of the system). Given the trajectory $( x _ { t } ) _ { t = 1 } ^ { T }$ , estimating $A ^ { * }$ is by now a well-studied problem, with many applications arising, e.g., in control systems and reinforcement learning. A key challenge in provably estimating $A ^ { * }$ is the dependencies between the observations $( x _ { t } ) _ { t = 1 } ^ { \bar { T } }$ . It was recently shown (Jedra and Proutiere, 2020; Sarkar and Rakhlin, 2019; Sim chowitz et al., 2018; Shirani Faradonbeh et al., 2018)

that the ordinary least squares (OLS) estimator recovers $A ^ { * }$ with non-asymptotic error bounds (up to log factors) of $O ( 1 / \sqrt { T } )$ holding with high probability $\left( \mathrm { w . h . p } \right)$ when $T$ is suitably large w.r.t n. The results hold under diferent sets of stability assumptions on $A ^ { * }$ , captured through its spectral radius $\rho ( A ^ { * } )$ , such as: strict stability $( \rho ( A ^ { * } ) < 1 )$ (Jedra and Proutiere, 2020), the more challenging marginal stability regime $\rho ( A ^ { * } ) ~ \leq ~ 1$ (Sarkar and Rakhlin, 2019; Simchowitz et al., 2019), and also for the unstable regime $\rho ( A ^ { * } ) >$ 1 (Sarkar and Rakhlin, 2019; Shirani Faradonbeh et al., 2018). Results under diferent structural assumptions on $A ^ { * }$ also exist, including sparsity (Basu and Michai lidis, 2015; Kock and Callot, 2015; Song and Bickel, 2011; Melnyk and Banerjee, 2016), low-rank plus sparsity (Lv et al., 2021; Basu et al., 2019), and general convex constraints (Tyagi and Efimov, 2024). These works show that $A ^ { * }$ can be provably recovered under milder requirements on T w.r.t n as opposed to the unstructured case.

Our setting. In this work, we focus on a setting in which an ϵ-fraction of the states are adversarially contaminated. Formally, the adversary corrupts data residing on a subset $s \subset [ T ]$ of time-indices, referred to as the set of outliers, with $| S | = s = \epsilon T$ (with $0 <$ $\epsilon < 1 )$ . The goal is to recover an estimate $\widehat { A }$ of $A ^ { * } \in$ A using the corrupted data $( \tilde { x } _ { t } ) _ { t = 1 } ^ { T }$ , and ensure it is robust to the corruptions. Here, $\mathcal { A } \subset \mathbb { R } ^ { n \times n }$ is a closed, convex set that captures prior structural information about $A ^ { * }$ . As discussed in Section 1.2, existing work in this setting for (1.1) is scarce, and typically not focused on the adversarial setting.

## 1.1 Our contributions

Our contributions are threefold.

1. In Section 2, we introduce diferent estimators for jointly estimating $A ^ { * }$ and the set of outliers, each based on a particular continuous relaxation of least-trimmed squares (Rousseeuw, 1984; Rousseeuw and Driessen, 2006). We also provide a simple alternating minimization based metaalgorithm (cf., Algorithm 1) which provides an eficient procedure for solving these relaxations.

2. In Section 3, we introduce two estimators which exploit the group-sparse nature of the outliers, and amount to minimizing a constrained, weighted least-squares objective with either (i) a $\ell _ { 1 } / \ell _ { 2 }$ group-sparse penalty, or (ii) group-sparse hard constraints. For the former, we obtain nonasymptotic error rates in the Frobenius norm of

(up to log-factors)

$$
O \left( { \sqrt { \frac { k } { T } } } + { \sqrt { \frac { n s } { T } } } \right)
$$

for joint estimation of k-sparse $A ^ { * }$ (with A chosen to be a $\ell _ { 1 }$ ball) and the contamination-levels. See Theorem 1 for details.

3. We provide a through empirical evaluation of the proposed estimators on synthetic examples, which show that they work well in practice.

Our theoretical results hold for an oblivious adversary (Assumption 1), analogous to the inlier-outlier model considered in robust regression: the choice of S is independent of $( x _ { t } ) _ { t = 1 } ^ { T }$ . Once S is chosen, the corruptions on $t \in S$ are allowed to be arbitrary – they can even depend on the data in the inlier set $S ^ { c }$ . Moreover, $A ^ { * }$ is required to be strictly stable (Assumption 2).

Our error rates are likely near-optimal (up to log factors) if one is also interested in estimating the outliers, but not for estimation rates solely for $A ^ { * } ;$ see discussion after Theorem 1. To our knowledge, these are the first theoretical results for provably robust estimation of LDSs, under adversarial contamination, from a single trajectory.

## 1.2 Related work

Existing works for robust estimation of LDSs can be categorized as follows. See Appendix C.2 for more details.

Heavy-tailed covariates or process noise. There exists a substantial recent line of work that provide robust estimators for learning $A ^ { * }$ under (1.1) (or higherorder versions thereof) from a single trajectory, with $( \eta _ { t } ) _ { t \geq 1 }$ assumed to be heavy-tailed (Qiu et al., 2015; Wong et al., 2020; Halder and Michailidis, 2022; Wang and Tsay, 2023; Wang et al., 2024; Dijk and Cho, 2025; Lu et al., 2025). Recently Kanakeri and Mitra (2025) studied this in the setting of multiple i.i.d trajectories of the form (1.1). These works typically provide nonasymptotic error bounds, with Wang and Tsay (2023) providing minimax optimal rates.

Sparse process noise. There exists a line of work that considers learning under (slightly general forms of) (1.1) from a single trajectory, where $\eta _ { t }$ is either drawn from a sparse adversarial process (Kim and Lavaei, 2026b,a) or involves an additive sparse outlier term (Kim, 2026; Farahmand et al., 2011). The results of Kim and Lavaei $\left( \mathrm { 2 0 2 6 b , a } \right)$ involve non-asymptotic error rates for parameter-recovery while Kim (2026) establish asymptotic normality for their estimator.

Learning from corrupted states. Bhatia et al. (2016) consider a univariate $( n ~ = ~ 1 )$ , higher-order version of (1.1), and assume that the learner observes $\tilde { x } _ { t } ~ = ~ x _ { t } + \theta _ { t }$ where $\theta \ = \ \left\lceil \theta _ { 1 } \quad \theta _ { 2 } \quad \ldots \quad \theta _ { T } \right\rceil$ is sparse. Assuming the outliers are independent of noise<sup>2</sup> they propose a block-sparse hard thresholding algorithm and provide non-asymptotic rates. The work of Kanakeri and Mitra (2025) discussed earlier, further shows (for the strong contamination adversarial model) that if an ϵ-fraction of the trajectories are corrupted, then their estimator still recovers $A ^ { * }$ up to a small error (w.r.t operator norm). Their theoretical result for this adversarial setting, and even their algorithm, does not apply to the single trajectory setting considered in our work. Wong (2017) considers a higher-order version (1.1). Assuming that the learner observes $\tilde { x } _ { t } = x _ { t } + w _ { t }$ where $( w _ { t } ) _ { t \geq 1 }$ are i.i.d centered Gaussians (with known covariance), they establish non-asymptotic recovery guarantees for the penalized lasso. Finally, we note that a line of work has considered the setting where $\tilde { x } _ { t }$ is formed by independently sub-sampling the entries of $x _ { t }$ (with missing entries set to zero), where $( x _ { t } ) _ { t = 1 } ^ { T }$ is generated from (1.1). Non-asymptotic error bounds are known for diferent structral assumptions on $A ^ { * }$ (under strict stability), e.g., sparsity and low-rankness (Rao et al., 2016, 2017b; Dalle and Castro, 2025; Jalali and Willett, 2018; Wong, 2017). Rao et al. (2017a) consider the setting where $\tilde { x } _ { t }$ is formed by taking compressive linear measurements of $x _ { t }$ (at each t), and study the limits of covariance-based estimation.

## 1.3 Notation

Let $x \in \mathbb { R } ^ { n }$ be a vector and $B , A \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ be matrices. For $i , j \in [ n _ { 1 } ] \times [ n _ { 2 } ]$ let $A _ { i , : } \in \mathbb { R } ^ { 1 \times n _ { 2 } }$ be the i-th row, $A _ { : , j } \in \mathbb { R } ^ { n _ { 1 } }$ be the j-th column and $A _ { i j }$ be the $( i , j ) -$ th entry of A. $\left\| x \right\| _ { p }$ denotes the standard $\ell _ { p }$ norm and $\left. A \right. _ { 2 }$ and $\left\| A \right\| _ { F } ^ { }$ denotes the spectral norm and the Frobenius norm of A respectively. For integers $p , q \geq 1$ , recall the (mixed) entry-wise $L _ { p , q }$ norm

$$
\left\| A \right\| _ { p , q } : = \left( \sum _ { j = 1 } ^ { n _ { 2 } } \left\| A _ { : , j } \right\| _ { p } ^ { q } \right) ^ { 1 / q } .
$$

Furthermore, we denote $\left. A \right. _ { 2 , 0 }$ := $\textstyle \sum _ { j \in [ n _ { 2 } ] } { \mathbb { 1 } } \left\{ \left\| A _ { : , j } \right\| _ { 2 } > 0 \right\}$ where 1 {E} denotes the indicator variable for event E. Note that $\lVert \cdot \rVert _ { 2 , 0 }$ is not a norm. The inner product of $B , A$ is defined as $\left. B , A \right. = \operatorname { T r } \left( B ^ { \top } A \right)$ , and $\mathrm { v e c } ( A ) \ \in \ \mathbb { R } ^ { n _ { 1 } n _ { 2 } }$ denotes the vector formed by stacking the columns of A. The all-ones vector in $\mathbb { R } ^ { n }$ is denoted by ${ \bf 1 } _ { n }$ . For matrices in $\mathbb { R } ^ { n \times n }$ , we denote the unit Frobenius sphere by $\mathbb { S } _ { n } ,$ the unit Frobenius ball by $B _ { n } ,$ and the identity matrix by $I _ { n }$ . The Kronecker product is denoted by $\otimes$

As far as possible, calligraphic letters are reserved for sets. The normal distribution with mean µ and covariance Σ is denoted by $\mathcal { N } \left( \boldsymbol { \mu } , \boldsymbol { \Sigma } \right)$ . For $a , b > 0$ , we write $a \lesssim b$ if there exists a constant $C > 0$ such that $a \leq C b$ . If a $\iota \lesssim b$ and $a \gtrsim b .$ , then we write $a \asymp b$ . The values of symbols used for denoting constants (e.g., $c , C$ etc.) may change from line-to-line.

## 2 ROBUST ESTIMATORS BASED ON LEAST-TRIMMED SQUARES

Consider the linear dynamical process given in equation (1.1) and the contamination process where s observations are contaminated. Recall that $( \tilde { x } _ { t } ) _ { t = 1 } ^ { T }$ denotes the contaminated observations. Denoting

$$
z _ { t } ^ { * } = \mathbb { 1 } \left\{ t \notin S \right\} ,
$$

consider the following A-constrained least squares problem for estimating $A ^ { * }$

$$
\operatorname* { m i n } _ { A \in \mathcal { A } } \sum _ { t = 1 } ^ { T - 1 } z _ { t } ^ { * } z _ { t + 1 } ^ { * } \big \| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 } .
$$

The above objective function only considers summands where both $\tilde { x } _ { t + 1 }$ and $\tilde { x } _ { t }$ are uncontaminated, thus filtering out the efects of contaminated entries. Of course, $( z _ { t } ^ { * } ) _ { t = 1 } ^ { T }$ is not known as S is unknown, and what is known is that $\textstyle \sum _ { t = 1 } ^ { T } z _ { t } ^ { * } = T - s$ . Hence, a natural approach is to jointly estimate $( A ^ { \ast } , ( z _ { t } ^ { \ast } ) _ { t = 1 } ^ { T } )$ via the following combinatorial optimization problem.

$$
\operatorname* { m i n } _ { A \in \mathcal { A } } \sum _ { t = 1 } ^ { T - 1 } z _ { t } z _ { t + 1 } \big \| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 }\tag{P1}
$$

$$
{ \mathrm { s . t . ~ } } \sum _ { t = 1 } ^ { T } z _ { t } = T - s { \mathrm { ~ a n d ~ } } z _ { t } \in \{ 0 , 1 \} \ \forall t \in [ T ] .
$$

Note that $( A ^ { \ast } , ( z _ { t } ^ { \ast } ) _ { t = 1 } ^ { T } )$ is feasible for (P1).

Remark 1. Note that Problem (P1) is analogous to the least trimmed squares estimator Rousseeuw $( 1 9 8 4 ) _ { ; }$ ; Rousseeuw and Driessen (2006) which is known to be robust to strong contamination Bhatia et al. (2015, 2017), in i.i.d. regression. However, unlike the standard least trimmed squares, problem (P1) involves products $z _ { t } z _ { t + 1 }$ and is thus formulated as a quadratic program w.r.t. $( z _ { t } ) _ { t = 1 } ^ { T }$

Remark 2. We assume s is known a priori, following Diakonikolas et al. (2019, 2023, 2024); Maurya et al. (2026); Pensia et al. (2025).

We can rewrite problem (P1) as follows. Given a matrix $\ b { A } \in \mathbb { R } ^ { n \times n }$ , let $\dot { Q } ( A ) ~ \in ~ \mathbb { R } ^ { T \times T }$ be defined as follows: for all $1 \ \leq \ t \ \leq \ T - 1 , \ Q ( A ) _ { t , t + 1 } \ =$ $\begin{array} { r } { Q ( A ) _ { t + 1 , t } = \frac { 1 } { 2 } \big \| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 } } \end{array}$ and $Q ( A ) _ { t , t ^ { \prime } } = 0$ for all other $t , t ^ { \prime } \in [ \bar { T } ]$ . Note that

T−1   
X z<sub>t</sub>z<sub>t+1</sub> x˜<sub>t+1</sub> − Ax˜<sub>t</sub> <sup>2</sup><sub>2</sub>   
t=1   
T−1   
= X z<sub>t</sub>z<sub>t+1</sub>(Q(A)<sub>t,t+1</sub> + Q(A)<sub>t+1,t</sub>)   
t=1   
T T   
= X X z<sub>t</sub>Q(A)<sub>t,t˜</sub>z<sub>t˜</sub> = z<sup>⊤</sup>Q(A)z,   
t=1 <sub>t</sub>˜<sub>=1</sub>

where $z ~ = ~ \left[ z _ { 1 } \quad z _ { 2 } \quad \ldots \quad z _ { T } \right] ^ { \top }$ . Therefore, problem (P1) is equivalent to

$$
\operatorname* { m i n } _ { A \in A , \ z \in \{ 0 , 1 \} ^ { T } } z ^ { \top } Q ( A ) z \ \mathrm { s . t } \ \mathbf { 1 } _ { T } ^ { \top } z = T - s .\tag{P2}
$$

Algorithm 1 Meta algorithm for jointly estimating   
$A ^ { * }$ and $( z _ { t } ^ { * } ) _ { t = 1 } ^ { T }$   
Require: Initial guess $A ^ { ( 0 ) }$ , convergence threshold $\epsilon ,$   
maximum iterations $K .$   
Ensure: Final estimates $( \widehat { A } , \widehat { z } )$   
1: $k  0$   
2: repeat   
3: Step 1: Update z   
4: Given $A ^ { ( k ) }$ solve a relaxed formulation (cf.,   
(P3),(P6),(P4) and (P5)) to obtain ˜z.   
5: Round ˜z using (2.1) to obtain $z ^ { ( k + 1 ) } \in \left\{ 0 , 1 \right\} ^ { T }$   
6: Step 2: Update A   
7: Given $z ^ { ( k + \hat { 1 } ) }$ , solve (P2) to find $A ^ { ( k + 1 ) }$   
8: $k \gets k + 1$   
9: until ${ \frac { \left\| A ^ { ( k ) } - A ^ { ( k - 1 ) } \right\| _ { F } } { \left\| A ^ { ( k - 1 ) } \right\| _ { F } } } \leq \epsilon { \mathrm { ~ o r ~ } } k \geq K$   
10: return $\stackrel { \cdot \cdot } { A } = A ^ { ( k ) } , \stackrel { \cdot } { z } = z ^ { ( k ) }$

Overview of relaxations of (P2) Note that problem (P2) is a combinatorial problem w.r.t the binary variables $z _ { t } \mathrm { \dot { s } } .$ . In order to circumvent this, we next introduce several estimators all of which are essentially obtained by considering diferent continuous relaxations of $\{ 0 , \dot { 1 } \} ^ { T }$

1. Sphere relaxation. Using the fact that $\mathbf { 1 } _ { T } ^ { \top } z =$ $\left. { z } \right. _ { 2 } ^ { 2 }$ when $z \in \left\{ 0 , 1 \right\} ^ { T }$ , we can relax (P2) as follows. See Appendix D.1 for details.

$$
\begin{array} { r l } & { \underset { A \in \mathcal { A } , z \in \mathbb { R } ^ { T } } { \operatorname* { m i n } } z ^ { \top } Q ( A ) z } \\ & { \mathrm { s . t . } \left\| z \right\| _ { 2 } ^ { 2 } = T - s , \ \mathbf { 1 } _ { T } ^ { \top } z = T - s . } \end{array}\tag{P3}
$$

Note that when A is fixed, (P3) is a variant of the Trust Region Subproblem. Since $Q ( A )$ contains only $O ( T )$ non-zero values, we can use the Mor´e-Sorensen algorithm (Mor´e and Sorensen, 1983) which takes $O ( n ^ { 2 } T + T K )$ time – here K is the number of iterations needed by the algorithm.

2. Semi-definite program (SDP) relaxation. Since $z ^ { \top } Q ( A ) z \ = \ \left. Q ( A ) , z z ^ { \top } \right.$ we can replace $z z ^ { \top }$ with the p.s.d matrix Y, and use the dual of Shor’s relaxation (Boyd and Vandenberghe, 1997) to derive the following relaxation of (P2). See $\mathrm { A p \mathrm { - } }$ pendix D.2 for details.

$$
\begin{array} { r l r } & { \underset { A \in \mathcal { A } , Y \succeq 0 , z \in [ 0 , 1 ] ^ { T } } { \mathrm { m i n } } \langle Q ( A ) , Y \rangle } & { \mathrm { ( P 4 ) } } \\ & { \mathrm { s . t . ~ } \mathrm { d i a g } ( Y ) = z , \mathbf { 1 } _ { T } ^ { \top } z = T - s \mathrm { ~ a n d ~ } \left[ 1 { \quad } z ^ { \top } \right] \succeq 0 . } \end{array}
$$

Note that for a fixed A, (P4) is a SDP and can be solved, $\mathrm { e . g . }$ . using interior point methods, in time polynomial in $T$ to obtain $Y , z .$ For eficiency, we implement an equivalent Burer-Monteiro method (Burer and Monteiro, 2003) running in $O ( n ^ { 2 } T +$ $T K )$ time. While the iteration count K lacks theoretical bounds, it remains small in practice.

3. Bi-convex relaxation. Since the filtering in (P1) is on each $\left\| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \right\| _ { 2 } ^ { 2 } ,$ instead of using $z _ { t } z _ { t + 1 }$ , we could introduce a $b _ { t }$ such that $b _ { t } = 1$ if and only if $z _ { t } z _ { t + 1 } = 1$ . It can be shown that this can be represented using a set of linear constraints involving $b _ { t } , z _ { t }$ and $z _ { t + 1 }$ . Using this, and relaxing the binary constraints on $( b _ { t } ) _ { t = 1 } ^ { \check { T } - 1 } , ( z _ { t } ) _ { t = 1 } ^ { T }$ we can derive the following bi-convex optimization problem (see Appendix D.3 for details).

$$
\begin{array} { r l r } { \displaystyle \operatorname* { m i n } _ { A \in A } } & { \displaystyle \sum _ { t = 1 } ^ { T - 1 } b _ { t } \big \| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 } } & { \mathrm { ( P 5 ) } } \\ { \displaystyle _ { z \in [ 0 , 1 ] ^ { T - 1 } } \sum _ { t = 1 } ^ { T } } & \\ { \displaystyle _ { b \in [ 0 , 1 ] ^ { T - 1 } } } & \\ { \displaystyle \mathrm { s . t . } \sum _ { t = 1 } ^ { T } z _ { t } = T - s } & \\ { \displaystyle z _ { t } + z _ { t + 1 } - 1 \leq b _ { t } \leq \operatorname* { m i n } \big \{ z _ { t } , z _ { t + 1 } \big \} } & { \forall t \in [ T - 1 ] . } \end{array}
$$

Note that the objective is bi-convex in A, b while the constraints are convex w.r.t A, b and z. In particular, when A is fixed, (P5) is a linear program w.r.t $b , z ,$ and hence can be solved eficiently in practice, e.g. using interior point methods, in time polynomial in $n , T$

4. Convexifying $Q ( A )$ to estimate $z .$ For any fixed A, define ${ \tilde { Q } } ( A ) : = Q ( A ) + \alpha ( A ) I _ { T }$ where $\alpha ( A )$ is chosen so that ${ \tilde { Q } } ( A )$ is p.s.d. It can be easily seen that in the setting where $z$ is binary, this does not change the objective. Now we can replace $Q ( A )$ with ${ \tilde { Q } } ( A )$ in (P2), and relax the constraints on z to $[ 0 , \dot { 1 } ] ^ { \dot { T } }$ , to obtain (see $\mathrm { A p - }$ pendix D.4 for details)

$$
\operatorname* { m i n } _ { z \in [ 0 , 1 ] ^ { T } } z ^ { \top } \tilde { Q } ( A ) z \quad \mathrm { s . t . } \quad \mathbf { 1 } _ { T } ^ { \top } z = T - s .\tag{P6}
$$

Note that (P6) is a convex problem. Moreover, unlike the other relaxations described above, (P6) requires an estimate A of $A ^ { * }$ to be available. Similar to the case of (P5), when A is fixed, (P6) can be solved in time polynomial in $n , T .$

Note that while the formulations (P3)–(P5) are nonconvex, they share a common property: if we fix $A \in { \mathcal { A } }$ then the resulting problem is tractable in the remaining variables, and we can solve for $z \in \{ 0 , 1 \}$ . On the other hand, if we fix $z ~ \in ~ \{ 0 , 1 \}$ , then we can solve the ensuing A-constrained least-squares problem in (P2) to estimate $A ^ { * }$ . This motivates the alternating minimization based meta algorithm described in Algorithm 1 to find estimates $( \widetilde { A } , ( \widehat { z } _ { t } ) _ { t = 1 } ^ { T } )$

Remark 3. The convergence of Algorithm 1 will $d e \mathrm { - }$ pend on the initialization $A ^ { ( 0 ) }$ . Since the relaxations considered are all non-convex, while a choice of $A ^ { ( 0 ) }$ within a small neighborhood of the ground-truth might lead to convergence, its unclear whether this holds for any random initialization. In practice, a potential choice of $A ^ { ( 0 ) }$ is to solve the A-constrained least squares problem on the contaminated data – our experiments follow this strategy.

Remark 4. Suppose we fix $z \in \left\{ 0 , 1 \right\} ^ { T }$ . Then (P3) is essentially just A-constrained weighted least squares which is the same as solving (P2) (with fixed z). For (P5), we can replace $b _ { t }$ with $z _ { t } z _ { t + 1 }$ (which is what the envelope constraints would be doing) and therefore, it reduces to solving (P2) with fixed z. For (P4) if we set $Y = z z ^ { \top }$ , then $\left. Q ( A ) , Y \right. = z ^ { \top } Q ( A ) z$ which for fixed z is the same as (P2). Solving (P2) over A takes $O ( n ^ { 2 } T + n ^ { 3 } )$ time $f o r \mathcal { A } = \mathbb { R } ^ { n \times n }$ , and ${ \dot { O } } ( n ^ { 2 } T + n ^ { 3 } K )$ for $\mathcal { A } = \left\{ A \in \mathbb { R } ^ { n \times n } : \left\| A \right\| _ { 1 , 1 } \leq \left\| A ^ { * } \right\| _ { 1 , 1 } \right\}$ (with K iterations of the FISTA algorithm (Beck and Teboulle, 2009)).

Rounding procedure. For rounding $\tilde { z } \in \mathbb { R } ^ { T }$ in $\mathrm { A l - }$ gorithm 1, we use the following procedure.

$$
\bar { z } _ { t } = \left\{ \begin{array} { l l } { 1 , } & { t \in \widehat { \mathcal { S } } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \widehat { \mathcal { S } } \in \underset { S ^ { \prime } \subseteq [ T ] , ~ | S ^ { \prime } | = T - s } { \operatorname * { a r g m a x } } \sum _ { t \in S ^ { \prime } } | \tilde { z } _ { t } | ,\tag{2.1}
$$

where $\bar { z } \in \left\{ 0 , 1 \right\} ^ { T }$ is the rounded solution. Note that this can be computed in O(T log T) time per iteration.

## 3 ROBUST LEARNING USING GROUP-SPARSE LASSO

Consider the linear dynamical system defined in (1.1) and recall that $s \subset [ T ]$ is the set of outliers. For each $t \in S ,$ , the adversary corrupts $x _ { t }$ to obtain $\tilde { x } _ { t }$ . Let $\theta _ { t } ^ { * } = \tilde { x } _ { t } - x _ { t }$ and note that $\theta _ { t } ^ { * } = 0$ for all $t ~ \in ~ S ^ { c }$ Furthermore,

$$
\begin{array} { r l } & { \tilde { x } _ { t + 1 } = x _ { t + 1 } + \theta _ { t + 1 } ^ { * } } \\ & { \qquad = A ^ { * } x _ { t } + \eta _ { t + 1 } + \theta _ { t + 1 } ^ { * } } \\ & { \qquad = A ^ { * } ( \tilde { x } _ { t } - \theta _ { t } ^ { * } ) + \eta _ { t + 1 } + \theta _ { t + 1 } ^ { * } } \\ & { \qquad = A ^ { * } \tilde { x } _ { t } + \eta _ { t + 1 } + \left( \theta _ { t + 1 } ^ { * } - A ^ { * } \theta _ { t } ^ { * } \right) . } \end{array}
$$

Now denoting $\begin{array} { r l r } { u _ { t } ^ { * } } & { { } : = } & { \left( \theta _ { t + 1 } ^ { * } - A ^ { * } \theta _ { t } ^ { * } \right) } \end{array}$ , we obtain $\tilde { x } _ { t + 1 } ~ = ~ A ^ { * } \tilde { x } _ { t } + \eta _ { t + 1 } + u _ { t } ^ { * }$ . Let us denote $U ^ { * } =$ $\left[ u _ { 1 } ^ { * } \quad u _ { 2 } ^ { * } \quad \ldots \quad u _ { T - 1 } ^ { * } \right] , u ^ { * } : = \mathrm { v e c } \left( U ^ { * } \right)$ and

$$
\mathcal { S } ( u ^ { * } ) = \{ t \in [ T - 1 ] : \left\| u _ { t } ^ { * } \right\| _ { 2 } \neq 0 \} .
$$

Then it is easy to see that $| S ( u ^ { * } ) | \leq 2 | S | = 2 s$ (recall that $| { \cal S } | = s )$

Consider the matrices

$$
\begin{array} { l } { { X = \left[ x _ { 1 } \quad x _ { 2 } \quad \ldots \quad x _ { T - 1 } \right] , \bar { X } = \left[ x _ { 2 } \quad x _ { 3 } \quad \ldots \quad x _ { T } \right] , } } \\ { { \Theta ^ { * } = \left[ \theta _ { 1 } ^ { * } \quad \theta _ { 2 } ^ { * } \quad \ldots \quad \theta _ { T - 1 } ^ { * } \right] , \bar { \Theta } ^ { * } = \left[ \theta _ { 2 } ^ { * } \quad \theta _ { 3 } ^ { * } \quad \ldots \quad \theta _ { T } ^ { * } \right] . } } \end{array}
$$

where $( x _ { t } ) _ { t \in [ T ] }$ are the uncontaminated data. For the contaminated data, we observe that

$$
\begin{array} { l l l l } { { Y : = \left[ \tilde { x } _ { 1 } \quad \tilde { x } _ { 2 } \quad \dots \quad \tilde { x } _ { T - 1 } \right] = X + \Theta ^ { * } \ \mathrm { a n d } , } } \\ { { \bar { Y } : = \left[ \tilde { x } _ { 2 } \quad \tilde { x } _ { 3 } \quad \dots \quad \tilde { x } _ { T } \right] = \bar { X } + \bar { \Theta } ^ { * } . } } \end{array}
$$

Furthermore, we also have $U ^ { * } = \bar { \Theta } ^ { * } - A ^ { * } \Theta ^ { * }$ where as seen above, $\bigl | \bigl | U ^ { * } \bigr | \bigr | _ { 2 , 0 } \leq 2 s$

Note that since we do not have any assumptions on the magnitude of the outliers, they can be potentially unbounded. To handle this, we use a filtering procedure that removes a pair $( \tilde { x } _ { t } , \tilde { x } _ { t + 1 } )$ if either $\left\| \tilde { \boldsymbol { x } } _ { t } \right\| _ { \infty }$ or $\left\| \tilde { x } _ { t + 1 } \right\| _ { \infty }$ is large. Formally, we introduce a diagonal matrix $\mathbf { \widetilde { \Gamma } } D \in \mathbb { R } ^ { ( T - 1 ) \times ( T - 1 ) }$ with $D _ { t }$ denoting the t-th diagonal entry of D. $D _ { t }$ represents the weight on the term $\left\| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \right\| _ { 2 } ^ { 2 } .$ In this work, we set

$$
D _ { t } = \mathbb { 1 } \left\{ \left\| { \tilde { x } } _ { t + 1 } \right\| _ { \infty } \leq r { \mathrm { ~ a n d ~ } } \left\| { \tilde { x } } _ { t } \right\| _ { \infty } \leq r \right\}
$$

for some parameter r (which will be specified in the analysis).

Remark 5. In this work, we consider one of the simplest forms of filtering (with D). In the i.i.d. robust regression setting, works such as Sasai and Fujisawa (2025); Pensia et al. (2025) use more sophisticated approaches, including iterative filtering and weight learning. Extending these to the VAR setting is not straightforward as the dependence between data would break the concentration arguments underlying their analyses.

Given this, we can now introduce our two estimators for jointly estimating $A ^ { * }$ and $U ^ { * }$ , both based on exploiting the block-sparse<sup>3</sup> nature of $U ^ { * }$

1. Least squares with $\ell _ { 1 } / \ell _ { 2 }$ block-sparse penalization. Denoting $U = \left[ u _ { 1 } \quad u _ { 2 } \quad \ldots \quad u _ { T - 1 } \right] \in$ $\mathbb { R } ^ { n \times ( T - 1 ) }$ , the first estimator solves the blocksparse penalized least squares problem

$$
\begin{array} { r } { \Big ( \widehat { A } , \widehat { U } \Big ) \in \Bigg \{ \underset { U \in \mathbb { R } ^ { n \times ( T - 1 ) } } { \mathrm { a r g m i n } } \ \frac { 1 } { 2 } \big \| ( \bar { Y } - A Y ) D - U \big \| _ { F } ^ { 2 } } \\ { + \lambda \big \| U \big \| _ { 2 , 1 } \Bigg \} . \qquad ( \mathrm { P 7 } } \end{array}
$$

Since $\mathcal { A }$ is convex, then (P7) is a convex program which can be solved eficiently in practice.

2. Least squares with block-sparse hard constraints. Since $\left. U ^ { * } \right. _ { 2 , 0 } \leq 2 s$ , we can replace the penalty in (P7) with a hard constraint to obtain,

$$
\left( \widehat { A } , \widehat { U } \right) \in \operatorname * { a r g m i n } _ { \stackrel { A \in \mathcal { A } } { U : \left\| U \right\| _ { 2 , 0 } \leq 2 s } } \frac { 1 } { 2 } \big \| ( \bar { Y } - A Y ) D - U \big \| _ { F } ^ { 2 } .\tag{P8}
$$

As will be seen shortly, our main theoretical result (Theorem 1) is for (P7) for sparse $A ^ { * }$ and with

$$
\mathcal { A } = \left\{ A \in \mathbb { R } ^ { n \times n } : \left\| A \right\| _ { 1 , 1 } \leq \left\| A ^ { * } \right\| _ { 1 , 1 } \right\} .\tag{3.1}
$$

Remark 6. The estimator (P7) is analogous to the penalized lasso that has been well-studied for robust sparse regression with i.i.d inliers $( e . g .$ , Dalalyan and Thompson (2019); Sasai and Fujisawa (2020); Chen et al. (2013)).

Remark 7. In the experiments, where we focus on A as in (3.1), we solve (P7) via a combination of block coordinate descent and constrained weighted least squares via alternating minimization in $O ( ( n ^ { 2 } T \mathrm { ~ + ~ }$ $n ^ { 3 } K ) { \widetilde K } )$ time for $\widetilde { K }$ iterations (where $\widetilde { K }$ is small in practice). (P8) is solved via alternating minimization. Indeed, when A is fixed then U can be solved for exactly by picking the columns of $( { \bar { Y } } - A Y ) D$ with the largest $\ell _ { 2 }$ norms, which can be computed in O(nT log T) time by computing the norms and sorting. When U is fixed, we revert to A-constrained weighted least squares which can be solved eficiently (as described above) in $O ( n ^ { 2 } T + n ^ { 3 } K )$ time (see Remark 4 for specifics on K).

Theorem 1 relies on the following assumptions.

Assumption 1 (Adversary). The adversary first partitions $[ \bar { T } ] = \mathcal { S } \dot { \cup } \mathcal { S } ^ { c }$ where the choice of S is oblivious to the data $( x _ { t } ) _ { t = 1 } ^ { T }$ . The data corresponding to time-indices in $s$ are then corrupted in an arbitrary manner, no assumption is made in this regard.

Assumption 2 (Stability of $A ^ { * } )$ . The quantity $\begin{array} { r } { J ( A ^ { * } ) : = \sum _ { i = 0 } ^ { \infty } \left\| ( A ^ { * } ) ^ { i } \right\| _ { 2 } } \end{array}$ is finite.

The adversarial model in Assumption 1 is similar to the inlier-outlier model that is well-studied in the robust-regression literature (with the clean observations being i.i.d) (see $\mathrm { e . g . }$ , Merad and Ga¨ıfas $\left( 2 0 2 3 \mathrm { b , a } \right) ,$ . The term $J ( A ^ { * } )$ in Assumption (2) was considered in Jedra and Proutiere (2020) for analyzing the OLS estimator. If $\rho ( A ^ { * } ) < 1$ then it can be shown that $J ( A ^ { * } )$ is finite, although it can grown with n. If $\left. A ^ { * } \right. _ { 2 } < 1$ , a direct calculation shows $J ( A ^ { * } ) \leq$ $\left( 1 - \left. A ^ { * } \right. _ { 2 } \right) ^ { - 1 }$ . Furthermore, if there exist constants $C > 0$ and $\beta \in ( 0 , 1 )$ such that $\left\| ( A ^ { * } ) ^ { i } \right\| _ { 2 } \leq C \beta ^ { i }$ holds for all integers $i \geq 1$ , then $\begin{array} { r } { J ( A ^ { * } ) \leq 1 + \frac { C \beta } { 1 - \beta } } \end{array}$ This latter condition is referred to as (C, β)-stability in the literature (cf., (Ziemann and Tu, 2022, Def. 6.2)).

We then obtain the following theorem.

Theorem 1. Under Assumptions 1 and 2, let $A ^ { * }$ be k-sparse and A be as in (3.1). There exist constants $c _ { 1 } , c _ { 2 } , c _ { 3 } , c ^ { \prime } > 1$ such that the following is true. Let $\delta \in ( 0 , 1 )$

$$
L = J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \cdot \log \left( \frac { n ^ { 2 } } { k } \right) \cdot \operatorname* { m a x } \left\{ s , \log \left( \frac { n } { \delta } \right) \right\} ~ a n d ,
$$

$$
\lambda = 2 \left( { \sqrt { n } } + n ^ { 1 / 4 } \log ^ { 1 / 4 } \left( { \frac { T } { \delta } } \right) + 2 ^ { 1 / 2 } \log ^ { 1 / 2 } \left( { \frac { T } { \delta } } \right) \right) .
$$

It holds that if

$$
T \geq c ^ { \prime } \cdot L \log ^ { 2 } \left( \frac { n \cdot L } { \delta } \right) , r = c _ { 3 } J ( A ^ { * } ) \sqrt { \log \left( \frac { T n } { \delta } \right) } ,
$$

then with probability at least $1 - \delta ,$ any solution $( \widehat { A } , \widehat { U } )$ of (P7) satisfies

$$
\begin{array} { r l } & { \left\| \widehat { A } - A ^ { * } \right\| _ { F } ^ { 2 } + \displaystyle \frac { 1 } { T } \big \| ( \widehat { U } - U ^ { * } ) D \big \| _ { F } ^ { 2 } } \\ & { \leq c _ { 1 } \displaystyle \frac { J ^ { 2 } ( A ^ { * } ) k \log \big ( \frac { T n } { \delta } \big ) \log \big ( \frac { n } { \delta } \big ) } { T } } \\ & { + c _ { 2 } \operatorname* { m a x } \left\{ n , \log ^ { 2 } \left( \displaystyle \frac { T } { \delta } \right) \right\} \displaystyle \frac { s } { T } . } \end{array}
$$

The theorem provides an error bound of

$$
O \left( \frac { k \log n \log ( n T / \delta ) } { T } + \frac { s \operatorname* { m a x } \left\{ n , \log ^ { 2 } ( T / \delta ) \right\} } { T } \right)
$$

which holds when T is suitably large. Since $A ^ { * }$ and $U ^ { * }$ are collectively (k + 2ns)-sparse, this rate is likely near optimal (up to log factors) if one is interested in estimating both $A ^ { * }$ and $U ^ { * }$ (this was also noted in (Dalalyan and Thompson, 2019, Section 2) in the i.i.d setting). If only estimating $A ^ { * }$ is of interest then the rate in Theorem 1 is sub-optimal - the $s / T$ factor should be $s ^ { 2 } / T ^ { 2 }$ (analogous to the i.i.d setting, cf., Dalalyan and Thompson (2019); Chen et al. (2016); Gao (2020)). A possible approach for proving this would be to adapt the techniques in Dalalyan and Thompson (2019) (based on incoherence and transfer principles) to our setting of dependent inliers. This is quite non-trivial and left as an important direction for future work.

Remark 8. The stated error bound holds for T suitably large. When $s \leq \log ( n / \delta )$ then $T \gtrsim k ^ { 2 } \log ^ { 4 } ( n / \delta )$ sufices. $I f s \ge \log ( n / \delta )$ then $T \gtrsim k ^ { 2 } s \log n \log ^ { 2 } ( n s )$ sufices. This quadratic dependence on k is an artefact of the proof-technique, in particular, with the argument used for showing the restricted lower isometry for the LHS term $\left\| \Delta Y ^ { \prime } D \right\| _ { F } ^ { 2 }$ in (B.3).

While the proof of Theorem 1 is available in $\mathrm { A p \mathrm { - } }$ pendix B, a sketch is provided below for convenience.

1. In Lemma 2, using the optimality of ${ \widehat { A } } , { \widehat { U } }$ and $A ^ { * } , U ^ { * }$ we derive the inequality (B.3),

$$
\begin{array} { r l } & { \underbrace { \frac { 1 } { 2 } \big \| \Delta Y ^ { \prime } D \big \| _ { F } ^ { 2 } } _ { ( I ) } \leq \underbrace { \Big \langle \Delta _ { A } , \overline { { E } } D Y ^ { \top } \Big \rangle } _ { ( I I ) } } \\ & { \qquad + \frac { \lambda } { 2 } \Bigg ( 3 \big \| ( \Delta _ { U } D ) _ { \widetilde { S } } \big \| _ { 2 , 1 } - \big \| ( \Delta _ { U } D ) _ { \widetilde { S } ^ { c } } \big \| _ { 2 , 1 } \Bigg ) } \end{array}
$$

which holds when $\begin{array} { r } { \lambda \geq 2 \operatorname* { m a x } _ { t \in \{ 2 , . . . , T \} } \left\| \eta _ { t } \right\| _ { 2 } } \end{array}$ . Here $\Delta _ { A } , \Delta _ { U } , \Delta$ denote the error terms, while $\mathbf { \bar { \boldsymbol { Y } } ^ { \prime } }$ depends on Y (see (B.1)). The set $\tilde { s }$ is the set of time-indices $t \in [ T - 1 ]$ for which at least one of $t , t + 1$ lies in $s ;$ see (B.2). Thus $\widetilde { S } ^ { c }$ is contained in the set of inliers $S ^ { c }$

2. The choice of $\lambda , r$ in Theorem 1 is established in Lemma’s 3 and 5 respectively, via standard concentration results. In particular, $r$ is obtained by ensuring that w.h.p, $\left\| \bar { \boldsymbol { x } } _ { t } \right\| _ { \infty } \leq r$ for all $t \in [ T ]$

3. In Lemmas 4, and 6 we establish an upper bound for (II). Lemma 4 involves applying Theorem 3 to control $\begin{array} { r } { \operatorname* { s u p } _ { b \in B ( r , \left| \widetilde { \mathcal { S } } \right| ) } \left| g ^ { \top } b \right| } \end{array}$ for $g { \sim } { \mathcal { N } } \left( 0 , I _ { T - 1 } \right)$ Lemma 6 bounds the tail of an appropriately defined self-normalized martingale using Theorem 4.

4. To bound (I) we use Lemmas 7 and 8. The proofs of both lemmas involve defining appropriate sets of matrices and applying Theorem 2 (Krahmer et al., 2014, Theorem 3.1).

5. Combining the bounds for (I) and (II), we obtain after some simplifications the statement of the theorem.

## 4 EXPERIMENTS

In our experiments, we use the following notation to denote the estimators.

<table><tr><td>Sphere-AM SDP-AM</td><td>Sphere Relaxation. Semi-definite program Relaxation.</td></tr><tr><td>Biconvex-AM</td><td>Bi-convex relaxation.</td></tr><tr><td>ConvexQ(A)-AM</td><td>Convexifying  $Q ( A )$  to estimate z.</td></tr><tr><td>LS-BSP</td><td>LS with block-sparse penalty.</td></tr><tr><td>LS-BSHC OLS</td><td>LS with block-sparse constraints.</td></tr><tr><td></td><td>A-constrained least squares.</td></tr></table>

Here, LS stands for least squares, and with a slight abuse of notation, we denote A-constrained LS by OLS.

System setup. All experiments were run on a machine with a Intel Core U9-285H (16 cores, 16 threads) and 32GB RAM, running Ubuntu 24.04. The code uses standard Python scientific computing libraries including, numpy, scipy, cvxpy, and pandas. All plots were generated with matplotlib and all random seeds were fixed for reproducibility. Each experiment used 50 parallel worker threads.

Implementation details. For fixed A, Sphere-AM uses the Mor´e-Sorensen trust-region method (Mor´e and Sorensen, 1983). SDP-AM is rewritten in an equivalent form with centered variables and solved using Burer-Monteiro method (Burer and Monteiro, 2003). For Biconvex-AM and ConvexQ(A)-AM, the z-update is solved directly with cvxpy. When z is fixed, $\widehat { A }$ is estimated via A-constrained weighted least squares (see Remark 4). For LS-BSP and LS-BSHC, the implementation follows the steps outlined in Remark 7. See Appendix E for more details.

Experimental setup. In all experiments, we consider $A ^ { * } ~ \in ~ \mathbb { R } ^ { n \times n }$ with $\textit { n } = \ 5 0$ , spectral radius $0 . 9$ and sparsity $\begin{array} { r l r } { k } & { { } = } & { 1 5 0 . } \end{array}$ , and we set $\begin{array} { r l } { A } & { { } = } \end{array}$ $\left\{ A \in \mathbb { R } ^ { n \times n } : \left\| A \right\| _ { 1 , 1 } \leq \left\| A ^ { * } \right\| _ { 1 , 1 } \right\}$ . For a fixed $A ^ { * }$ , we run 20 Monte-Carlo simulations; in each simulation, we generate a sequence $( x _ { t } ) _ { t = 0 } ^ { T }$ for a given $T$ by setting $x _ { 0 } = 0$ and using the VAR (1.1). The resulting sequence is then contaminated according to the chosen contamination model. We repeat this process for 20 diferent matrices $A ^ { * }$ and report the mean recovery error together with its variance. For alternating minimization, initialize $A ^ { ( 0 ) }$ as the minimizer for Aconstrained least squares over contaminated data.

![](images/8a55da281197b5c720ee6a4569682cf7ec82bf2e98c3b14da67425cd4a7fb9c5.jpg)  
(a) ϵ = 0

![](images/c6fb0edf434a3a266d6acf48f38fb341a45a979b012177cebed861b64aa4ecbc.jpg)  
(b) ϵ = 0.02

![](images/cee8d0b063174bbe9840013ac767a907140f6e29677c7af41f211b47f5445f7f.jpg)  
(c) ϵ = 0.1

Figure 1: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ for $T \in \{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 \}$ with (a) ϵ = 0, (b) ϵ = 0.02, and (c) ϵ = 0.1, under the random heavy-tailed contamination model.  
![](images/4eb3756670cfb2c84d0baec25eaaf54fa0afd285f65581d154c86d57fa6937dd.jpg)  
(a) T = 100

![](images/b939b23e30b0bf66bd093f5b171b0242b56c8a7e03aa38e2659aaebec3eb102d.jpg)  
(b) T = 400

![](images/07e4ac0c5b600761548459f9d8e195db24d68e59ebbdfc7117ba87ea6eacf178.jpg)  
(c) T = 800  
Figure 2: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } .$ for $\epsilon \in \lbrace 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \rbrace$ with (a) $T = 1 0 0 \mathrm { { ; } }$ , (b) $T = 4 0 0$ , and (c) $T = 8 0 0$ , under the random heavy-tailed contamination model.

Let us consider random heavy-tailed contamination, in which outliers drawn from a Cauchy distribution with location and scale both set to 10, are added to indices selected uniformly at random (with probability depending on s). See Appendix E for results for other contamination models.

Frobenius norm error against T. Figure 1 shows the Frobenius error, $\lVert \widehat { A } - \bar { A } ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ against $T \in$ {50, 100, 200, 400, 800}. When $\epsilon = 0$ , all estimators coincide with OLS and therefore achieve the same error. While Sphere-AM fails to recover $A ^ { * }$ for $\epsilon > 0 . 0 2$ or $T > 1 0 0$ , it performs even better than Biconvex-AM when both ϵ and $T$ are small. In the small-ϵ regime, SDP-AM appears to outperform the other estimators with Biconvex-AM approaching the same error as T increases (see Figure 1b). However, when $\epsilon \ : = \ : 0 . 1$ Biconvex-AM outperforms the other estimators, with LS-BSHC achieving comparable performance. The ad vantage of Biconvex-AM can be attributed to it involving a relatively tighter relaxation of the original problem, which handles filtering both at the level of individual observations and in the interactions between adjacent entries. For all estimators except Sphere-AM and OLS, the error decreases as T increases, although the rate of decrease slows.

Frobenius norm error against ϵ. Figure 2 shows the Frobenius error, $| \hat { A } - | ^ { * } \| _ { F } / \| A ^ { * } \| _ { F }$ against $\epsilon \in$ $\{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \}$ . We observe similar behavior across all T ∈ {100, 400, 800} with OLS and Sphere-AM showing the highest error. As expected, the error increases with ϵ. While SDP-AM achieves a smaller error in the small-ϵ regime, Biconvex-AM performs best as ϵ increases. Across the range of ϵ considered, LS-BSHC achieves a recovery error comparable to that of Biconvex-AM. Note that while Sphere-AM generally performs comparably to OLS, it outperforms LS-BSHC, LS-BSP and Biconvex-AM when $T = 1 0 0$ and $\epsilon \leq 0 . 0 2$ suggesting it may be more suited for small $T , \epsilon$

## AI use statement

We have not used generative AI tools for syntheticdata generation, theoretical or conceptual development, mathematical claims or proofs, hypotheses, research methodology, experiments, dataset cleaning or reformatting, qualitative analysis, or interpretation of results. We did not use generative AI tools for any of the tasks with recommended disclosure, including brainstorming, literature discovery or summarization, drafting or editing text, creating figures, formatting references, or suggesting titles, keywords, or paper structure.

The algorithms and experiments were designed by the authors. Generative AI tools assisted with parts of the implementation, primarily by optimizing and refactoring code. The authors reviewed all AI-assisted code and checked its correctness through testing and by comparing results against expected behavior.

We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## Acknowledgements

This work was supported by a Nanyang Associate Professorship (NAP) grant from NTU Singapore.

## References

Y. Abbasi-yadkori, D. P´al, and C. Szepesv´ari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems, volume 24, 2011.

A. Bakshi and A. Prasad. Robust linear regression: optimal rates in polynomial time. In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, STOC 2021, page 102–115, 2021.

S. Balakrishnan, S. S. Du, J. Li, and A. Singh. Computationally eficient robust sparse estimation in high dimensions. In Proceedings of the 2017 Conference on Learning Theory, volume 65 of Proceedings of Machine Learning Research, pages 169–212, 2017.

S. Basu and G. Michailidis. Regularized estimation in sparse high-dimensional time series models. The Annals of Statistics, 43(4):1535–1567, 2015.

S. Basu, X. Li, and G. Michailidis. Low rank and structured modeling of high-dimensional vector autoregressions. Trans. Sig. Proc., 67(5):1207–1222, 2019.

A. Beck and M. Teboulle. A fast iterative shrinkagethresholding algorithm for linear inverse problems.

SIAM Journal on Imaging Sciences, 2(1):183–202, 2009.

K. Bhatia, P. Jain, and P. Kar. Robust regression via hard thresholding. In Proceedings ofthe 29th International Conference on Neural Information Processing Systems - Volume 1, NIPS’15, page 721–729, 2015.

K. Bhatia, P. Jain, P. Kamalaruban, and P. Kar. Eficient and consistent robust time series analysis. arxiv:1607.00146, 2016.

K. Bhatia, P. Jain, P. Kamalaruban, and P. Kar. Consistent robust regression. In Advances in Neural Information Processing Systems, volume 30, 2017.

N. Boumal, V. Voroninski, and A. S. Bandeira. The non-convex burer–monteiro approach works on smooth semidefinite programs. In Proceedings of the 30th International Conference on Neural Information Processing Systems, page 2765–2773, 2016.

S. Boyd and L. Vandenberghe. Semidefinite Programming Relaxations of Non-Convex Problems in Control and Combinatorial Optimization, pages 279–287. Springer US, 1997.

S. Burer and R. D. Monteiro. A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization. Mathematical Programming, 95(2):329–357, 2003.

M. Chen, C. Gao, and Z. Ren. A general decision theory for Huber’s ϵ-contamination model. Electronic Journal of Statistics, 10(2):3752 – 3774, 2016.

Y. Chen, C. Caramanis, and S. Mannor. Robust sparse regression under adversarial corruption. In Proceedings of the 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pages 774–782, 2013.

Y. Cherapanamjeri, E. Aras, N. Tripuraneni, M. I. Jordan, N. Flammarion, and P. L. Bartlett. Optimal robust linear regression in nearly linear time. arXiv:2007.08137, 2020.

A. Dalalyan and P. Thompson. Outlier-robust estimation of a sparse linear model using \ell 1-penalized huber's m-estimator. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

G. Dalle and Y. D. Castro. Minimax estimation of partially-observed vector autoregressions. Electronic Journal of Statistics, 19(1):2364 – 2410, 2025.

I. Diakonikolas, W. Kong, and A. Stewart. Eficient algorithms and lower bounds for robust linear regression. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms, page 2745–2754, 2019.

I. Diakonikolas, D. Kane, A. Pensia, and T. Pittas. Near-optimal algorithms for gaussians with huber contamination: Mean estimation and linear regression. Advances in Neural Information Processing Systems, 36:43384–43422, 2023.

I. Diakonikolas, D. M. Kane, S. Karmalkar, A. Pensia, and T. Pittas. Robust sparse estimation for gaussians with optimal error under huber contamination. arXiv:2403.10416, 2024.

D. Dijk and H. Cho. Tail-robust estimation of factor-adjusted vector autoregressive models for highdimensional time series. arXiv:2509.22235, 2025.

J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra. Eficient projections onto the l1-ball for learning in high dimensions. In Proceedings of the 25th International Conference on Machine Learning, page 272–279. Association for Computing Machinery, 2008.

S. Farahmand, G. B. Giannakis, and D. Angelosante. Doubly robust smoothing of dynamical processes via outlier sparsity constraints. IEEE Transactions on Signal Processing, 59:4529–4543, 2011.

S. Foucart and H. Rauhut. A Mathematical Introduction to Compressive Sensing. Applied and Numerical Harmonic Analysis. Birkhauser Boston, Secaucus, NJ, 2013.

C. Gao. Robust regression via mutivariate regression depth. Bernoulli, 26(2):pp. 1139–1170, 2020.

S. Halder and G. Michailidis. Robust estimation of sparse, high dimensional time series with polynomial tails. arXiv:2211.07558, 2022.

A. Jalali and R. Willett. Sparse transition matrix estimation for sub-gaussian autoregressive processes with missing data. In 2018 Annual American Control Conference (ACC), pages 1881–1886, 2018.

Y. Jedra and A. Proutiere. Finite-time identification of stable linear systems optimality of the leastsquares estimator. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 996–1001, 2020.

V. Kanakeri and A. Mitra. Outlier-robust linear system identification under heavy-tailed noise. In Proceedings of the 7th Annual Learning for Dynamics &amp; Control Conference, volume 283 of Proceedings of Machine Learning Research, pages 540–551, 2025.

S. Karmalkar and E. Price. Compressed Sensing with Adversarial Sparse Noise via L1 Regression. In 2nd Symposium on Simplicity in Algorithms (SOSA 2019), volume 69 of Open Access Series in Informatics (OASIcs), pages 19:1–19:19, Dagstuhl, Germany, 2019. Schloss Dagstuhl – Leibniz-Zentrum f¨ur Informatik.

H. Kim. Essays on Robust Time Series Estimation and Forecasting under Outliers and Fat Tails. PhD thesis, Columbia University, 2026.

J. Kim and J. Lavaei. Huber-based robust system identification with near-optimal guarantees across independent and adversarial regimes. arXiv:2603.27586, 2026a.

J. Kim and J. Lavaei. On the necessity of two-stage estimation for learning dynamical systems under both noise and node-wise attacks. arXiv:2602.07288, 2026b.

A. B. Kock and L. Callot. Oracle inequalities for high dimensional vector autoregressions. Journal of Econometrics, 186(2):325–344, 2015.

F. Krahmer, S. Mendelson, and H. Rauhut. Suprema of chaos processes and the restricted isometry property. Communications on Pure and Applied Mathematics, 67(11):1877–1904, 2014.

B. Laurent and P. Massart. Adaptive estimation of a quadratic functional by model selection. The Annals of Statistics, 28(5):1302 – 1338, 2000.

L. Liu, T. Li, and C. Caramanis. High dimensional robust m-estimation: Arbitrary corruption and heavy tails. arXiv:1901.08237, 2019.

L. Liu, Y. Shen, T. Li, and C. Caramanis. High dimensional robust sparse regression. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 411– 421, 2020.

K. Lounici, M. Pontil, S. van de Geer, and A. B. Tsybakov. Oracle inequalities and optimal inference under group sparsity. The Annals of Statistics, 39(4): 2164–2204, 2011.

Y. Lu, C. Tao, D. Wang, G. S. Uddin, L. Wu, and X. Zhu. Robust estimation for dynamic spatial autoregression models with nearly optimal rates. Journal of Econometrics, 251:106065, 2025.

X. Lv, W. Cui, and Y. Liu. Linear convergence of gradient methods for estimating structured transition matrices in high-dimensional vector autoregressive models. In Advances in Neural Information Processing Systems, volume 34, pages 16751–16763, 2021.

D. Maurya, A. Barik, and J. Honorio. Robust esti mation of a sparse linear model: Provable guarantees with non-convexity. In The 29th International Conference on Artificial Intelligence and Statistics, 2026.

I. Melnyk and A. Banerjee. Estimating structured vector autoregressive models. In Proceedings of The 33rd International Conference on Machine Learning, pages 830–839, 2016.

I. Merad and S. Ga¨ıfas. Robust supervised learning with coordinate gradient descent. Statistics and Computing, 33(5), 2023a.

I. Merad and S. Ga¨ıfas. Robust methods for highdimensional linear learning. J. Mach. Learn. Res., 24 (1), 2023b.

S. Minsker, M. Ndaoud, and L. Wang. Robust and tuning-free sparse linear regression via square-root slope. SIAM Journal on Mathematics of Data Science, 6(2):428–453, 2024.

J. J. Mor´e and D. C. Sorensen. Computing a trust region step. SIAM Journal on Scientific and Statistical Computing, 4(3):553–572, 1983. doi: 10.1137/ 0904038.

M. Papini, M. Pirotta, and M. Restelli. Smoothing policies and safe policy gradients. Machine Learning, 111(11):4081–4137, 2022.

A. Pensia, V. Jog, and P.-L. Loh. Robust regression with covariate filtering: Heavy tails and adversarial contamination. Journal of the American Statistical Association, 120(550):1002–1013, 2025.

H. Qiu, S. Xu, F. Han, H. Liu, and B. Cafo. Robust estimation of transition matrices in high dimensional heavy-tailed vector autoregressive processes. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 1843–1851, 2015.

M. Rao, A. Kipnis, T. Javidi, Y. C. Eldar, and A. Goldsmith. System identification from partial samples: Non-asymptotic analysis. In 2016 IEEE 55th Conference on Decision and Control (CDC), pages 2938–2944, 2016. doi: 10.1109/CDC.2016. 7798707.

M. Rao, T. Javidi, Y. C. Eldar, and A. Goldsmith. Fundamental estimation limits in autoregressive processes with compressive measurements. In 2017 IEEE International Symposium on Information Theory (ISIT), pages 2895–2899, 2017a.

M. Rao, T. Javidi, Y. C. Eldar, and A. Goldsmith. Estimation in autoregressive processes with

partial observations. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4212–4216, 2017b. doi: 10.1109/ICASSP.2017.7952950.

G. Raskutti, M. J. Wainwright, and B. Yu. Minimax rates of estimation for high-dimensional linear regression over $\ell _ { q }$ -balls. IEEE Transactions on Information Theory, 57(10):6976–6994, 2011.

P. J. Rousseeuw. Least median of squares regression. Journal of the American Statistical Association, 79 (388):871–880, 1984.

P. J. Rousseeuw and K. Driessen. Computing lts regression for large data sets. Data Min. Knowl. Discov., 12(1):29–45, 2006.

T. Sarkar and A. Rakhlin. Near optimal finite time identification of arbitrary linear dynamical systems. In Proceedings of the 36th International Conference on Machine Learning, ICML, volume 97, pages 5610– 5618, 2019.

T. Sasai. Robust and sparse estimation of linear regression coeficients with heavy-tailed noises and covariates. arXiv:2206.07594, 2022.

T. Sasai and H. Fujisawa. Robust estimation with lasso when outputs are adversarially contaminated. arXiv:2004.05990, 2020.

T. Sasai and H. Fujisawa. Outlier robust and sparse estimation of linear regression coeficients. Journal of Machine Learning Research, 26(93):1–79, 2025.

M. K. Shirani Faradonbeh, A. Tewari, and G. Michailidis. Finite time identification in unstable linear sys tems. Automatica, 96:342–353, 2018.

M. Simchowitz, H. Mania, S. Tu, M. I. Jordan, and B. Recht. Learning without mixing: Towards a sharp analysis of linear system identification. In Proceedings of the 31st Conference On Learning Theory, volume 75, pages 439–473, 2018.

M. Simchowitz, R. Boczar, and B. Recht. Learning linear dynamical systems with semi-parametric least squares. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99, pages 2714– 2802, 2019.

S. Song and P. J. Bickel. Large vector auto regressions. arXiv:1106.3915, 2011.

M. Talagrand. The generic chaining. Springer Monographs in Mathematics. Springer, Berlin, Germany, 2005.

M. Talagrand. Upper and lower bounds for stochastic processes, volume 60. Springer, 2014.

H. Tyagi and D. Efimov. Learning linear dynamical systems under convex constraints. arxiv:2303.15121, 2024.

R. Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Number 47 in Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018.

M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.

D. Wang and R. S. Tsay. Rate-optimal robust estimation of high-dimensional vector autoregressive models. The Annals of Statistics, 51(2):846–877, 2023.

Y. Wang, G. Li, Z. Xiao, L. Xu, and W. Zhang. Robust estimation for high-dimensional time series with heavy tails. arXiv:2411.05217, 2024.

K. C. Wong. Lasso Guarantees for Dependent Data. PhD thesis, The University of Michigan, 2017.

K. C. Wong, Z. Li, and A. Tewari. Lasso guarantees for β-mixing heavy-tailed time series. The Annals of Statistics, 48(2):1124 – 1142, 2020.

F. Ye and C.-H. Zhang. Rate minimaxity of the lasso and dantzig selector for the $\ell _ { q }$ loss in $\ell _ { r }$ balls. Journal of Machine Learning Research, 11(114):3519–3540, 2010.

P. Zhang. Robust Estimation and Inference under Huber’s Contamination Model. PhD thesis, University of Pittsburgh, 2023.

I. Ziemann and S. Tu. Learning with little mixing. In Advances in Neural Information Processing Systems, volume 35, pages 4626–4637, 2022.

## Table of Contents (Appendix)

A Technical tools 13   
B Proof of Theorem 1 15   
B.1 Proof of Lemma 2 21   
B.2 Proof of Lemma 3 22   
B.3 Proof of Lemma 4 22   
B.4 Proof of Lemma 5 23   
B.5 Proof of Lemma 6 24   
B.6 Proof of Lemma 7 25   
B.7 Proof of Lemma 8 27   
C Detailed overview of related work 29   
C.1 Robust regression under adversarial contamination 29   
C.2 Robust learning of linear dynamical systems . 31   
C.2.1 Heavy-tailed covariates or process noise 31   
C.3 Sparse process noise 32   
C.4 Learning from corrupted states 33   
D Robust estimators based on least-trimmed squares 33   
D.1 Sphere relaxation . 33   
D.2 Semi-definite program (SDP) relaxation 34   
D.3 Bi-convex relaxation 34   
D.4 Convexifying Q(A) to estimate z 35   
E Additional experiments 36   
E.1 Overview of the implementation 36   
E.2 Additional results on synthetic experiments 37   
E.2.1 Block burst contamination 37   
E.2.2 Sign flip contamination 38   
E.2.3 High-leverage contamination 39

## A TECHNICAL TOOLS

The proof of Theorem 1 utilizes a number of technical tools which we recall here, starting with the definitions of Talagrand’s $\gamma _ { \alpha }$ functionals (Talagrand, 2014).

Definition 1 (Talagrand’s $\gamma _ { \alpha }$ functionals (Talagrand, 2014)). Let $( \mathcal { M } , d )$ be a metric space. We define an adimissible sequence to be a sequence of subsets of M, $( \mathcal { M } _ { r } ) _ { r \ge 0 }$ , where $| \mathcal { M } _ { 0 } | = 1$ and $| \mathcal { M } _ { r } | \leq 2 ^ { 2 ^ { r } }$ for all $r \geq 1$ For any $0 < \alpha < \infty$ , the $\gamma _ { \alpha }$ function of $( \mathcal { M } , d )$ is defined as,

$$
\gamma _ { \alpha } ( \mathcal { M } , d ) : = \operatorname* { i n f } \operatorname* { s u p } _ { s \in S } \sum _ { r = 0 } ^ { \infty } 2 ^ { r / \alpha } d ( s , \mathcal { M } _ { r } )
$$

where the inf is over all admissible sequences of M.

It is useful to note that $\gamma _ { \alpha }$ satisfies the following properties.

1. For any two metrics $d _ { 1 }$ and $d _ { 2 } , d _ { 1 } \leq c \cdot d _ { 2 } \implies \gamma _ { \alpha } ( { \mathcal { M } } , d _ { 1 } ) \leq c \gamma _ { \alpha } ( { \mathcal { M } } , d _ { 2 } ) ( \mathrm { w h e r e } c > 0 ) .$

2. For $\widehat { \mathcal { M } } \subseteq \mathcal { M } , \gamma _ { \alpha } ( \widehat { \mathcal { S } } , d ) \leq c _ { \alpha } \gamma _ { \alpha } ( \mathcal { S } , d )$ where $c _ { \alpha } > 0$ is a constant that only depends on α.

3. If the map $f : ( \mathcal { M } , d _ { 1 } )  ( S , d _ { 2 } )$ is onto, and R-Lipschitz for all $x , y \in { \mathcal { M } }$ , then $\gamma _ { \alpha } ( S , d _ { 2 } ) \leq c _ { \alpha } R \gamma _ { \alpha } ( M , d _ { 1 } )$ where $c _ { \alpha } > 0$ is a constant that only depends on α.

Property 1 follows from the definition while Properties 2 and 3 were stated in (Talagrand, 2005, Theorem 1.3.6) (with $c _ { \alpha } = 1 )$ for an alternative definition of the $\gamma _ { \alpha }$ functional (Talagrand, 2014, Definition 2.2.19) which is equivalent to Definition 1 up to constants depending only on α. See also (Talagrand, 2014, Section 2.3).

For any set $\mathcal { P } \subset \mathbb { R } ^ { n }$ , the Gaussian width of P is defined as (see e.g., Vershynin (2018))

$$
w \left( \mathcal { P } \right) : = \mathbb { E } \left[ \operatorname* { s u p } _ { x \in \mathcal { P } } \left. x , g \right. \right] \mathrm { f o r } g \sim \mathcal { N } \left( 0 , I _ { n } \right) .
$$

Recall Talagrand’s majorizing measure theorem (Talagrand, 2014, Theorem 2.4.1), which tells us that

$$
\gamma _ { 2 } ( \mathcal { P } , \| \cdot \| _ { 2 } ) \asymp w ( \mathcal { P } ) .\tag{A.1}
$$

Let M be a given set of matrices and denote

$$
d _ { 2 } ( \mathcal { M } ) : = \operatorname* { s u p } _ { M \in \mathcal { M } } \left. M \right. _ { 2 } , \quad d _ { F } ( \mathcal { M } ) : = \operatorname* { s u p } _ { M \in \mathcal { M } } \left. M \right. _ { F } .
$$

Theorem 3.1 of Krahmer et al. (2014) provides a concentration bound for the suprema of second order subgaussian choas process involving positive semidefinite matrices.

Theorem 2 (Theorem 3.1 of Krahmer et al. (2014)). Let M be a set of matrices and η be a vector with independent, zero-mean, R-subgaussian entries with unit variance. $L e t ,$

$$
\begin{array} { r l } & { F : = \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) \left[ \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) + d _ { F } ( \mathcal { M } ) \right] + d _ { F } ( \mathcal { M } ) d _ { 2 } ( \mathcal { M } ) , } \\ & { V : = d _ { 2 } ( \mathcal { M } ) \left[ \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) + d _ { F } ( \mathcal { M } ) \right] , \ a n d , \ U : = d _ { 2 } ^ { 2 } ( \mathcal { M } ) . } \end{array}
$$

There exist constants $c _ { 1 } , c _ { 2 } > 0$ depending only on R such that for any $t > 0$

$$
\mathbb { P } \left( \operatorname* { s u p } _ { M \in \mathcal { M } } \left| \left. M \eta \right. _ { 2 } ^ { 2 } - \mathbb { E } \left[ \left. M \eta \right. _ { 2 } ^ { 2 } \right] \right| \geq c _ { 1 } F + t \right) \leq 2 \exp \left( - c _ { 2 } \operatorname* { m i n } \left\{ \frac { t ^ { 2 } } { V } , \frac { t } { U } \right\} \right) .
$$

The following theorem taken from (Wainwright, 2019, Chapter 3) is well-known. It shows that Lipschitz and convex functions of independent and bounded random variables concentrate well.

Theorem 3 (Wainwright (2019)). Consider $X = ( X _ { 1 } , X _ { 2 } , \ldots , X _ { n } )$ independent random variables with $X _ { i } \in [ a , b ]$ almost surely. Let $f : \mathbb { R } ^ { n }  \mathbb { R }$ be convex and R-Lipschitz with respect to $\left\| \cdot \right\| _ { 2 }$ norm. Then $\forall t \geq 0$

$$
\mathbb { P } \left( \lvert f ( X ) - \mathbb { E } f ( X ) \rvert \ge t \right) \le 2 \exp \left( \frac { - t ^ { 2 } } { 2 R ^ { 2 } ( b - a ) ^ { 2 } } \right) .
$$

The following result taken from (Abbasi-yadkori et al., 2011, Theorem 1) bounds the $\ell _ { 2 }$ norm of self-normalized vector-valued martingales.

Theorem 4 ((Abbasi-yadkori et al., 2011, Theorem 1)). Let $( \mathcal { F } _ { t } ) _ { t = 1 } ^ { \infty }$ be a filtration. Let $( \eta _ { t } ) _ { t = 1 } ^ { \infty }$ be a real valued stochastic process such that $\eta _ { t }$ is $\mathcal { F } _ { t }$ measurable, and $\eta _ { t + 1 }$ is $R { - } s u b { - } G$ aussian conditioned on $\mathcal { F } _ { t }$ . Let $( X _ { t } ) _ { t = 1 } ^ { \infty }$ be a stochastic process where $X _ { t } \in \mathbb { R } ^ { d }$ is $\mathcal { F } _ { t }$ measurable. Let $V \in \mathbb { R } ^ { n \times n }$ be a p.d matrix and for $t \geq 1$ , denote

$$
\bar { V } _ { t } = V + \sum _ { s = 1 } ^ { t } X _ { s } X _ { s } ^ { \top } a n d S _ { t } = \sum _ { s = 1 } ^ { t } \eta _ { s + 1 } X _ { s } .
$$

Then for any $\delta > 0$ , it holds with probability $\geq 1 - \delta$ that for all $t \geq 1$

$$
\left\| S _ { t } \right\| _ { \bar { V } _ { t } ^ { - 1 } } ^ { 2 } \leq 2 R ^ { 2 } \log \left( \frac { \operatorname * { d e t } ( \bar { V } _ { t } ) ^ { 1 / 2 } \operatorname * { d e t } ( V ) ^ { - 1 / 2 } } { \delta } \right)
$$

where $\left\| S _ { t } \right\| _ { \bar { V } _ { t } ^ { - 1 } } = \sqrt { S _ { t } ^ { \top } ( \bar { V } _ { t } ^ { - 1 } ) S _ { t } }$

We will also use the following tail bound from Laurent and Massart (2000).

Lemma 1 (Lemma 1.1 of Laurent and Massart (2000)). Let $X _ { 1 } , X _ { 2 } , \ldots , X _ { d } { \overset { \mathrm { i . i . d . } } { \sim } } \mathcal { N } \left( 0 , 1 \right)$ and let $\mu ~ =$ $( \mu _ { 1 } , \mu _ { 2 } , \ldots , \mu _ { d } )$ be a non-negative vector. Define,

$$
\xi = \sum _ { i = 1 } ^ { d } \mu _ { i } ( X _ { i } ^ { 2 } - 1 ) .
$$

Then it holds $\forall v \geq 0$

$$
\operatorname* { P r } \left( \xi \geq 2 \| \mu \| _ { 2 } \sqrt { v } + 2 \| \mu \| _ { \infty } v \right) \leq e ^ { - v } .
$$

## B PROOF OF THEOREM 1

We start by noting that (P7) can be written as

$$
\Big ( \widehat { A } , \widehat { U } \Big ) \in \operatorname * { a r g m i n } _ { \stackrel { A \in A } { U \in \mathbb { R } ^ { n \times ( T - 1 ) } } } \frac { 1 } { 2 } \big \| ( \overline { { Y } } - A Y - U ) D \big \| _ { F } ^ { 2 } + \lambda \big \| U D \big \| _ { 2 , 1 } .
$$

Indeed, for any t such that $D _ { t } = 0$ and $\widehat { \boldsymbol { u } } _ { t } \neq 0$ , setting $\widehat { u } _ { t } = 0$ would reduce the objective value. Therefore,

$$
\left\{ t \in [ T - 1 ] : \widehat { u } _ { t } \neq 0 \right\} \subseteq \left\{ t \in [ T - 1 ] : D _ { t } = 1 \right\} .
$$

Denoting $\overline { { E } } = \left[ \eta _ { 2 } \quad \eta _ { 3 } \quad \cdot \cdot \quad \eta _ { T } \right]$ and $E = \left[ \eta _ { 1 } \quad \eta _ { 2 } \quad \dots \quad \eta _ { T - 1 } \right]$ , it is useful to note that

$$
\overline { { { Y } } } = A ^ { * } Y + \overline { { { E } } } + U ^ { * } .
$$

Moreover, it will be convenient to introduce the notation

$$
{ \Delta } _ { A } = \widehat { A } - { A } ^ { * } , { \Delta } _ { U } = \widehat { U } - { U } ^ { * } , { \Delta } = \left[ { \Delta } _ { A } { \Delta } _ { U } \right] , { \mathrm { a n d } } Y ^ { \prime } = \left[ { Y } _ { I } ^ { \prime } \right] .\tag{B.1}
$$

For any matrix $B \in \mathbb { R } ^ { n \times m }$ and ${ \mathcal { T } } \subseteq [ m ]$ , we will denote $( B ) _ { \mathcal { I } } \in \mathbb { R } ^ { n \times m }$ to be the column-wise restriction of B on to I. The sets

$$
\begin{array} { r } { \widetilde { S } : = \left\{ t \in [ T - 1 ] : t \in S \ \mathrm { ~ o r ~ } \ t + 1 \in S ^ { c } \right\} , \widetilde { S } ^ { c } : = \left\{ t \in [ T - 1 ] : t \in S ^ { c } \ \mathrm { ~ a n d ~ } \ t + 1 \in S ^ { c } \right\} } \end{array}\tag{B.2}
$$

will play a crucial role in the analysis for decomposition of the error terms. Note that

$$
\left| { \widetilde { S } } \right| \leq 2 { \left| { \mathcal { S } } \right| } = 2 s \mathrm { a n d } \left| { \widetilde { S } } ^ { c } \right| = T - 1 - { \left| { \widetilde { S } } \right| } \geq T - 1 - 2 s .
$$

Moreover $\smash { \widetilde { S } ^ { c } \subseteq S ^ { c } , \mathrm { i . e . , } \widetilde { S } ^ { c } }$ is contained in the set of inliers, hence $\tilde { x } _ { t } = x _ { t }$ for each $t \in \widetilde { S } ^ { c }$

The starting point of the analysis is Lemma 2 below, the proof of which follows from a feasibility-optimality argument and is detailed in Appendix B.1.

Lemma 2. $I f \lambda \geq 2 \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} } \left\| \eta _ { t } \right\| _ { 2 } ,$ then

$$
\frac { 1 } { 2 } \big \| \Delta Y ^ { \prime } D \big \| _ { F } ^ { 2 } \leq \Big \langle \Delta _ { A } , \overline { { E } } D Y ^ { \top } \Big \rangle + \frac { \lambda } { 2 } \Bigg ( 3 \big \| \big ( \Delta _ { U } D \big ) _ { \widetilde { \mathcal { S } } } \big \| _ { 2 , 1 } - \big \| \big ( \Delta _ { U } D \big ) _ { \widetilde { \mathcal { S } ^ { c } } } \big \| _ { 2 , 1 } \Bigg ) .\tag{B.3}
$$

In the rest of this section, we will derive upper and lower bounds for the RHS and LHS of (B.3) respectively.

Establishing an upper bound on right-hand side of (B.3). In order to upper bound the RHS of (B.3), we need to control $\left. \Delta _ { A } , \overline { { E } } D Y ^ { \top } \right.$ and λ. We start by showing that $\operatorname* { m a x } _ { t \in \{ 2 , . . . , T \} } \left\| \eta _ { t } \right\| _ { 2 } ^ { 2 }$ is bounded with high probability which we will be used to set λ. For any $\delta \in ( 0 , 1 )$ , consider the event

$$
\mathcal { E } _ { 2 } = \left\{ \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} } \left. \eta _ { t } \right. _ { 2 } \leq \sqrt { n } + 2 ^ { 1 / 2 } n ^ { 1 / 4 } \left( \log \left( \frac { T } { \delta } \right) \right) ^ { 1 / 4 } + 2 ^ { 1 / 2 } \left( \log \left( \frac { T } { \delta } \right) \right) ^ { 1 / 2 } \right\} .
$$

Lemma 3. For any $\delta \in ( 0 , 1 )$ , we have

$$
\mathbb { P } ( \mathcal { E } _ { 2 } ) \geq 1 - \delta .
$$

The proof of Lemma 3 follows from applying Lemma 1 on $\left\| \eta _ { t } \right\| _ { 2 }$ , see Appendix B.2 for details. Setting $\lambda = 2 \overline { { \lambda } }$ (where $\begin{array} { r } { \overline { { \lambda } } = \sqrt { n } + ( 4 n \log ( \frac { T } { \delta } ) ) ^ { 1 / 4 } + \sqrt { 2 \log ( \frac { T } { \delta } ) } ) } \end{array}$ , it follows that on $\mathcal { E } _ { 2 }$ , we have that $\lambda \geq \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} }$ η<sub>t 2</sub> holds. Given the result on $\lambda ,$ the next step is to bound $\left. \Delta _ { A } , \overline { { E } } D Y ^ { \top } \right.$ . Note that,

$$
\begin{array} { r l } & { \left. \Delta _ { A } , \overline { { E } } D Y ^ { \top } \right. = \left. \Delta _ { A } , \overline { { E } } D _ { \tilde { s } } Y _ { \tilde { s } } ^ { \top } \right. + \left. \Delta _ { A } , \overline { { E } } D _ { \tilde { s } ^ { c } } Y _ { \tilde { s } ^ { c } } ^ { \top } \right. } \\ & { \qquad \leq \left\| \Delta _ { A } \right\| _ { 1 , 1 } \left( \left\| \overline { { E } } D _ { \tilde { s } } Y _ { \tilde { s } } ^ { \top } \right\| _ { \infty , \infty } + \left\| \overline { { E } } D _ { \tilde { s } ^ { c } } Y _ { \tilde { s } ^ { c } } ^ { \top } \right\| _ { \infty , \infty } \right) . } \end{array}
$$

Now recall the definition of $D _ { t }$ . From the definition, we know that for all t such that $D _ { t } = 1 , \left\| \tilde { x } _ { t } \right\| _ { \infty } \leq r$ (for some parameter r). Also note that since $D _ { \widetilde { s } } Y _ { \widetilde { s } } ^ { \top }$ is supported on $\widetilde { s }$ (which implies $D _ { \widetilde { s } }$ has at most $| \widetilde { s } |$ non-zero terms on the diagonal), each column can only have at most $| \widetilde s |$ non-zero entries. Therefore, each column of $D _ { \widetilde { s } } Y _ { \widetilde { s } } ^ { \top } \mathrm { ~ i s ~ } { \left| \widetilde { s } \right| }$ -sparse with $\left\| \cdot \right\| _ { \infty } \leq r$ . Denote

$$
\mathcal B \left( r , \left| \widetilde { \mathcal S } \right| \right) = \left\{ b \in \mathbb R ^ { T - 1 } : \left\| b \right\| _ { \infty } \leq r , \left\| b \right\| _ { 0 } \leq \left| \widetilde { \mathcal S } \right| \right\}
$$

and consider the event

$$
\mathcal { E } _ { 1 a } = \left\{ \left. \overline { { E } } D _ { \widetilde { S } } Y _ { \widetilde { S } } ^ { \top } \right. _ { \infty , \infty } \leq c \left( w \left( \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) \right) + r \left| \widetilde { \mathcal { S } } \right| \sqrt { \log \left( \frac { T } { \delta } \right) } \right) \right\} .
$$

for a suitably large constant $c > 1$

Lemma 4. For any $\delta \in ( 0 , 1 )$ , we have

$$
\mathbb { P } ( \mathcal { E } _ { 1 a } ) \ge 1 - \delta .
$$

The proof of Lemma 4 follows from observing that $\| \overline { { E } } D _ { \widetilde { s } } Y _ { \widetilde { s } } ^ { \top } \| _ { \infty , \infty }$ can be bounded by $\begin{array} { r } { \operatorname* { s u p } _ { b \in B ( r , | \widetilde { \mathcal { S } } | ) } | g ^ { \top } b | } \end{array}$ for $g { \sim } { \mathcal { N } } \left( 0 , I _ { T - 1 } \right)$ using the properties of E and $D _ { \widetilde { s } } Y _ { \widetilde { s } } ^ { \top }$ , followed by establishing an upper bound on $\begin{array} { r } { \operatorname* { s u p } _ { b \in B ( r , | \widetilde { s } | ) } | g ^ { \top } b | } \end{array}$ using Theorem 3 (Wainwright, 2019, from Chapter 3). See Appendix B.3 for details.

Denote the event

$$
{ \mathcal { E } } _ { 1 b } = \left\{ D _ { t } = 1 \forall t \in { \widetilde { { \mathcal { S } } } } ^ { c } \right\} .
$$

We can show that for $\delta \in ( 0 , 1 )$ and r suitably large, $\mathcal { E } _ { 1 b }$ holds with probability at least $1 - \delta$

Lemma 5. There exists a constant $c > 1$ such that for any $\delta \in ( 0 , 1 )$ , and $\begin{array} { r } { r = c J ( A ^ { * } ) \sqrt { \log \left( \frac { T n } { \delta } \right) } } \end{array}$ , it holds that

$$
\mathbb { P } \left( \mathcal { E } _ { 1 b } \right) \ge 1 - \delta .
$$

The proof of Lemma 5 follows from using the structure inherent in the uncontaminated data and using standard concentration bounds. The details are available in Appendix B.4.

Next we establish a high-probability upper bound on $\| \overline { { E } } D _ { \widetilde { S } ^ { c } } Y _ { \widetilde { S } ^ { c } } ^ { \top } \| _ { \infty , \infty }$ . To this end, consider the event (for a suitably large constant $c _ { 1 } > 1 )$

$$
\mathcal { E } _ { 1 c } = \left\{ \left. \overline { { E } } D _ { \widetilde { S } ^ { c } } Y _ { \widetilde { S } ^ { c } } ^ { \top } \right. _ { \infty , \infty } \leq c _ { 1 } r \sqrt { T \log \left( \frac { n } { \delta } \right) } \right\} .
$$

Lemma 6. For any $\delta \in ( 0 , 1 )$ , it holds that

$$
\mathbb { P } ( \mathcal { E } _ { 1 c } ) \geq 1 - \delta .
$$

The proof of Lemma 6 involves bounding the tails of an appropriately defined self-normalized martingale using Theorem 4, and is provided in Appendix B.5.

Before moving on to the final term $\left. \Delta _ { A } , \overline { { E } } D Y ^ { \top } \right.$ we first observe the following.

Remark 9. Denote ${ \mathcal { K } } = \operatorname* { s u p p } ( A ^ { * } )$ . Since $\left. A ^ { * } \right. _ { 1 , 1 } \geq \left. \widehat { A } \right. _ { 1 , 1 }$ , we obtain,

$$
\begin{array} { r l } & { \left\| A ^ { * } \right\| _ { 1 , 1 } \geq \left\| \Delta _ { A } + A ^ { * } \right\| _ { 1 , 1 } } \\ & { \qquad = \left\| ( \Delta _ { A } ) _ { K } + ( A ^ { * } ) _ { K } \right\| _ { 1 , 1 } + \left\| ( \Delta _ { A } ) _ { K ^ { c } } \right\| _ { 1 , 1 } } \\ & { \qquad \geq \left\| ( A ^ { * } ) _ { K } \right\| _ { 1 , 1 } + \left\| ( \Delta _ { A } ) _ { K ^ { c } } \right\| _ { 1 , 1 } - \left\| ( \Delta _ { A } ) _ { K } \right\| _ { 1 , 1 } } \end{array}
$$

which implies

$$
\left\| ( \Delta _ { A } ) _ { K ^ { c } } \right\| _ { 1 , 1 } \leq \left\| ( \Delta _ { A } ) _ { K } \right\| _ { 1 , 1 } .
$$

So $\Delta _ { A } \in { \mathcal { C } } _ { \mathcal { K } }$ where,

$$
\begin{array} { r } { \mathcal { C } _ { \mathcal { K } } : = \left\{ H \in \mathbb { R } ^ { n \times n } : \left. ( H ) _ { K ^ { c } } \right. _ { 1 , 1 } \leq \left. ( H ) _ { \mathcal { K } } \right. _ { 1 , 1 } \right\} . } \end{array}
$$

Note that this implies, $\left\| \Delta _ { A } \right\| _ { 1 , 1 } \leq 2 \left\| ( \Delta _ { A } ) \kappa \right\| _ { 1 , 1 } \leq 2 { \sqrt { k } } \left\| \Delta _ { A } \right\| _ { F }$ where $k = | \boldsymbol { \mathcal { K } } |$ . We denote $\widetilde { \mathcal { C } } _ { \mathcal { K } } = \mathcal { C } _ { \mathcal { K } } \cap \mathbb { S } _ { n }$ On $\mathcal { E } _ { 1 a } \cap \mathcal { E } _ { 1 b } \cap \mathcal { E } _ { 1 c }$ , we have,

$$
\begin{array} { r l } & { \left. \Delta _ { A } , \overline { { E } } D Y ^ { \top } \right. \leq \left\| \Delta _ { A } \right\| _ { 1 , 1 } \left( \left\| \overline { { E } } D _ { \tilde { s } } Y _ { \tilde { s } } ^ { \top } \right\| _ { \infty , \infty } + \left\| \overline { { E } } D _ { \tilde { s } ^ { c } } Y _ { \tilde { s } ^ { c } } ^ { \top } \right\| _ { \infty , \infty } \right) } \\ & { \qquad \leq c \| \Delta _ { A } \| _ { 1 , 1 } \left( w \left( \mathcal { B } \left( r , \left| \tilde { s } \right| \right) \right) + r \left( \left| \tilde { s } \right| + \sqrt { T } \right) \sqrt { \log \left( \frac { n } { \delta } \right) } \right) } \\ & { \qquad \leq c \sqrt { k } \left\| \Delta _ { A } \right\| _ { F } \left( w \left( \mathcal { B } \left( r , \left| \tilde { s } \right| \right) \right) + r \left( \left| \tilde { s } \right| + \sqrt { T } \right) \sqrt { \log \left( \frac { n } { \delta } \right) } \right) } \end{array}
$$

where the last inequality comes from the fact $\left\| \Delta _ { A } \right\| _ { 1 , 1 } \leq 2 \sqrt { k } \| \Delta _ { A } \| _ { F }$ (see Remark 9).

Next we can bound w $\left( \boldsymbol { B } \left( \boldsymbol { r } , \left| \boldsymbol { \widetilde { s } } \right| \right) \right)$ . Note that $\mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) \subseteq \left\{ v \in \mathbb { R } ^ { T - 1 } : \left\| v \right\| _ { 1 } \leq r \left| \widetilde { \mathcal { S } } \right| \right\}$ . Therefore,

$$
w \left( \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) \right) \leq w \left( \left\{ v \in \mathbb { R } ^ { T - 1 } : \left\| v \right\| _ { 1 } \leq r \left| \widetilde { \mathcal { S } } \right| \right\} \right) \leq c r \left| \widetilde { \mathcal { S } } \right| \sqrt { \log T }
$$

for a constant $c > 1$ , where we use the fact that w $\left( \left\{ v \in \mathbb { R } ^ { T - 1 } : \left\| v \right\| _ { 1 } \leq 1 \right\} \right) \leq \sqrt { \log T }$ (Vershynin, 2018, refer to Example 7.5.8).

Let $\mathcal { E } _ { 1 } : = \mathcal { E } _ { 1 a } \cap \mathcal { E } _ { 1 b } \cap \mathcal { E } _ { 1 c } .$ . On ${ \mathcal { E } } _ { 1 }$ , we have,

$$
\begin{array} { r } { \left. \Delta _ { A } , \overline { { E } } D Y ^ { \top } \right. \leq c \sqrt { k } \big \| \Delta _ { A } \big \| _ { F } \left( r \left| \widetilde { \mathcal { S } } \right| \sqrt { \log T } + r \left( \left| \widetilde { \mathcal { S } } \right| + \sqrt { T } \right) \sqrt { \log \left( \frac { n } { \delta } \right) } \right) \leq \overline { { V } } \big \| \Delta _ { A } \big \| _ { F } , } \end{array}
$$

where

$$
\overline { { V } } = c r \sqrt { k } \bigg ( \left| \widetilde { \mathcal { S } } \right| \operatorname* { m a x } \bigg \{ \sqrt { \log T } , \sqrt { \log \Big ( \frac { n } { \delta } \Big ) } \bigg \} + \sqrt { T \log \Big ( \frac { n } { \delta } \Big ) } \bigg )
$$

and $\begin{array} { r } { r = c J ( A ^ { * } ) \sqrt { \log \left( \frac { T n } { \delta } \right) } } \end{array}$ . To summarize, we have shown that on $\mathcal { E } _ { 1 } , \left. \Delta _ { A } , \overline { { E } } D Y ^ { \top } \right. \leq \overline { { V } } \| \Delta _ { A } \| _ { F }$ holds.

Setting $\lambda = 2 \overline { { { \lambda } } }$ , on ${ \mathcal { E } } _ { 1 } \cap { \mathcal { E } } _ { 2 }$ , we have from (B.3) that

$$
\frac { 1 } { 2 } \big \| \Delta Y ^ { \prime } D \big \| _ { F } ^ { 2 } \leq \overline { { V } } \big \| \Delta _ { A } \big \| _ { F } + \overline { { \lambda } } \left( 3 \big \| \big ( D \Delta _ { U } \big ) _ { \widetilde { S } } \big \| _ { 2 , 1 } - \big \| \big ( D \Delta _ { U } \big ) _ { \widetilde { S } ^ { c } } \big \| _ { 2 , 1 } \right)
$$

which implies,

$$
\left\| ( D \Delta _ { U } ) _ { \widetilde { \mathcal { S } } ^ { c } } \right\| _ { 2 , 1 } \leq \frac { \overline { { V } } } { \overline { { \lambda } } } \big \| \Delta _ { A } \big \| _ { F } + 3 \big \| ( D \Delta _ { U } ) _ { \widetilde { \mathcal { S } } } \big \| _ { 2 , 1 } .
$$

Establishing a lower bound on the left-hand side of (B.3). Now it remains to lower bound $\begin{array} { r } { \frac { 1 } { 2 } \left. \Delta Y ^ { \prime } D \right. _ { F } ^ { 2 } . } \end{array}$ To this end, note that,

$$
\begin{array} { r l } & { \| \Delta Y ^ { \prime } D \| _ { F } ^ { 2 } = \| [ \Delta _ { { \cal A } } \quad \Delta _ { \cal U } ] \bigg [ { \cal Y } } \\ & { \quad \quad \quad = \| ( \Delta _ { { \cal A } } { \cal Y } + \Delta _ { \cal U } ) D \| _ { F } ^ { 2 } } \\ & { \quad \quad \quad = \| \Delta _ { { \cal A } } { \cal Y } D \| _ { F } ^ { 2 } + \| \Delta _ { { \cal U } } D \| _ { F } ^ { 2 } + 2 \Big \langle \Delta _ { { \cal A } } { \cal Y } D , \Delta _ { { \cal U } } D \Big \rangle . } \end{array}
$$

Therefore, we can lower bound the $\big \| \Delta Y ^ { \prime } D \big \| _ { F } ^ { 2 }$ by lower bounding $\left. \Delta _ { A } Y D \right. _ { F } ^ { 2 }$ and controlling $\left. \Delta _ { A } Y D , \Delta _ { U } D \right.$ We start of with a lower bound on $\left\| \Delta _ { A } Y D \right\| _ { F } ^ { 2 }$ . Consider the event

$$
\mathcal { E } _ { 3 } = \left\{ \left. \Delta _ { A } Y D \right. _ { F } ^ { 2 } \geq \frac { T } { 4 } \big \Vert \Delta _ { A } \big \Vert _ { F } ^ { 2 } \right\} .
$$

Lemma 7. There exists a constant $c _ { 1 } > 1$ such that for any $\delta \in ( 0 , 1 )$ , if

$$
T \geq 4 s + c _ { 1 } J ^ { 4 } ( A ^ { * } ) \log ^ { 2 } \left( { \frac { 1 } { \delta } } \right) k \log \left( { \frac { n ^ { 2 } } { k } } \right)
$$

then $\mathbb { P } ( \mathcal { E } _ { 3 } ) \ge 1 - \delta$

The proof of Lemma 7 is available in Appendix B.6. It follows from an application of Theorem 2 (Theorem 3.1 of Krahmer et al. (2014)) on an appropriate set of matrices and using the results on the Gaussian width of $\widetilde { C } _ { \kappa }$

Next consider $\left. \Delta _ { A } Y D , \Delta _ { U } D \right.$ . For $\delta \in ( 0 , 1 )$ and suitably large constants $c _ { 1 } , c _ { 2 } > 1$ , consider the event

$$
\mathcal { E } _ { 4 } = \left\{ \left| \left. \Delta _ { A } Y D , \Delta _ { U } D \right. \right| \leq c _ { 1 } \frac { \overline { { V } } } { \overline { { \lambda } } } R \big \| \Delta _ { A } \big \| _ { F } ^ { 2 } + c _ { 2 } r R \sqrt { \left| \widetilde { \mathcal { S } } \right| k } \big \| \Delta _ { U } D \big \| _ { F } \big \| \Delta _ { A } \big \| _ { F } \right\}
$$

where $\begin{array} { r } { R = J ( A ^ { * } ) \sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) \log \left( \frac { T } { \delta } \right) } } \end{array}$

Lemma 8. For any $\delta \in ( 0 , 1 )$ , we have

$$
\mathbb { P } \left( \mathcal { E } _ { 4 } \right) \geq 1 - \delta .
$$

We refer the reader to the Appendix B.7 for the complete proof of Lemma 8. The proof involves a carefu decomposition over $\widetilde { s }$ and $\widetilde { S } ^ { c }$ , and bounding the individual terms using tools recalled in Section A.

We can now establish the final lower bound on $\left\| \Delta Y ^ { \prime } D \right\| _ { F } ^ { 2 }$ . On ${ \mathcal { E } } _ { 3 } \cap { \mathcal { E } } _ { 4 }$

$$
\begin{array} { l } { \left\| { \Delta Y ^ { \prime } D } \right\| _ { F } ^ { 2 } = \left\| { \Delta _ { A } Y D } \right\| _ { F } ^ { 2 } + \left\| { \Delta _ { U } D } \right\| _ { F } ^ { 2 } + 2 \Big \langle \Delta _ { A } Y D , \Delta _ { U } D \Big \rangle } \\ { \geq \displaystyle \frac { T } { 4 } \left\| { \Delta _ { A } } \right\| _ { F } ^ { 2 } - c _ { 2 } \frac { \overline { { V } } } { \overline { { \lambda } } } R \left\| { \Delta _ { A } } \right\| _ { F } ^ { 2 } - c _ { 3 } r R \sqrt { k \left| \widetilde { \mathcal { S } } \right| } \left\| { \Delta _ { A } } \right\| _ { F } \left\| { \Delta _ { U } D } \right\| _ { F } + \left\| { \Delta _ { U } D } \right\| _ { F } ^ { 2 } } \\ { \geq \displaystyle \frac { T } { 4 } \left\| { \Delta _ { A } } \right\| _ { F } ^ { 2 } - c _ { 2 } \frac { \overline { { V } } } { \overline { { \lambda } } } R \left\| { \Delta _ { A } } \right\| _ { F } ^ { 2 } - c _ { 3 } ^ { \prime } r ^ { 2 } R ^ { 2 } k \left. \widetilde { \mathcal { S } } \right. \left\| { \Delta _ { A } } \right\| _ { F } ^ { 2 } - \displaystyle \frac { 1 } { 2 } \left\| { \Delta _ { U } D } \right\| _ { F } ^ { 2 } + \left\| { \Delta _ { U } D } \right\| _ { F } ^ { 2 } . } \end{array}
$$

If $\begin{array} { r } { T \geq c \left( \frac { \overline { { V } } } { \overline { { \lambda } } } R + r ^ { 2 } R ^ { 2 } k \left| \widetilde { \mathcal { S } } \right| \right) } \end{array}$ for a suitably large constant $c \geq 1$ , we get that,

$$
\frac { 1 } { 2 } { \left\| { \Delta Y ^ { \prime } D } \right\| } _ { F } ^ { 2 } \geq c _ { 1 } T { \left\| { \Delta _ { A } } \right\| } _ { F } ^ { 2 } + c _ { 2 } { \left\| { \Delta _ { U } D } \right\| } _ { F } ^ { 2 } .
$$

This means that on $\mathcal { E } _ { 1 } \cap \mathcal { E } _ { 2 } \cap \mathcal { E } _ { 3 } \cap \mathcal { E } _ { 4 }$ ，

$$
\begin{array} { r l } { c _ { 1 } T \big \| \Delta _ { \boldsymbol { A } } \big \| _ { F } ^ { 2 } + c _ { 2 } \big \| \Delta _ { \boldsymbol { U } } D \big \| _ { F } ^ { 2 } \leq \overline { { V } } \big \| \Delta _ { \boldsymbol { A } } \big \| _ { F } + \overline { { \lambda } } \left( 3 \big \| \big ( \Delta _ { \boldsymbol { U } } D \big ) _ { \widetilde { \mathcal { S } } } \big \| _ { 2 , 1 } - \big \| \big ( \Delta _ { \boldsymbol { U } } D \big ) _ { \widetilde { \mathcal { S } } ^ { \epsilon } } \big \| _ { 2 , 1 } \right) } & { } \\ { \leq \overline { { V } } \big \| \Delta _ { \boldsymbol { A } } \big \| _ { F } + c _ { 3 } \overline { { \lambda } } \sqrt { \big | \widetilde { \mathcal { S } } \big | \big \| \Delta _ { \boldsymbol { U } } D \big \| _ { F } } } & { } \\ { \leq \frac { \overline { { V } } } { \sqrt { c _ { 1 } T } } \sqrt { c _ { 1 } T } \big \| \Delta _ { \boldsymbol { A } } \big \| _ { F } ^ { 2 } + \frac { c _ { 3 } \overline { { \lambda } } \sqrt { \big | \widetilde { \mathcal { S } } \big | } } { \sqrt { c _ { 2 } } } \sqrt { c _ { 2 } \big \| \Delta _ { \boldsymbol { U } } D \big \| _ { F } ^ { 2 } } } & { } \\ { \leq \bigg ( \frac { \overline { { V } } ^ { 2 } } { c _ { 1 } T } + c _ { 3 } ^ { \prime } \overline { { \lambda } } ^ { 2 } \big | \widetilde { \mathcal { S } } \big | \bigg ) ^ { 1 / 2 } \Big ( c _ { 1 } T \big \| \Delta _ { \boldsymbol { A } } \big \| _ { F } ^ { 2 } + c _ { 2 } \big \| \Delta _ { \boldsymbol { U } } D \big \| _ { F } ^ { 2 } \Big ) ^ { 1 / 2 } . } \end{array}
$$

The above implies that for some constants $c _ { 1 } , c _ { 2 } \geq 1$ ，

$$
\begin{array} { l } { \displaystyle \left\| { \Delta _ { A } } \right\| _ { F } ^ { 2 } + \frac { 1 } { T } \big \| { \Delta _ { U } D } \big \| _ { F } ^ { 2 } \leq { c _ { 1 } } \frac { { r ^ { 2 } k \Big ( \left| \tilde { \mathcal { S } } \right| \operatorname* { m a x } \left\{ \sqrt { \log T } , \sqrt { \log \left( \frac { n } { \delta } \right) } \right\} + \sqrt { T \log \left( \frac { n } { \delta } \right) } \Big ) ^ { 2 } } } { { T ^ { 2 } } } } \\ { \displaystyle \qquad + { c _ { 2 } } \frac { { \Big ( \sqrt { n } + n ^ { 1 / 4 } \log ^ { 1 / 4 } \left( \frac { T } { \delta } \right) + 2 ^ { 1 / 2 } \log ^ { 1 / 2 } \left( \frac { T } { \delta } \right) \Big ) ^ { 2 } \left| \tilde { \mathcal { S } } \right| } } { T } . } \end{array}\tag{B.4}
$$

Lower bound on $T .$ . Note that for the statements to hold, we need,

$$
T \geq c \left( \frac { \overline { { V } } } { \overline { { \lambda } } } R + r ^ { 2 } R ^ { 2 } k \left| \widetilde { S } \right| \right) \mathrm { a n d } T \geq 4 s + c _ { 1 } J ^ { 4 } ( A ^ { * } ) \log ^ { 2 } \left( \frac { 1 } { \delta } \right) k \log n .\tag{B.5}
$$

Recall the definitions of ${ \overline { { V } } } , { \overline { { \lambda } } } ,$ R and $^ { r } \cdot$

$$
\begin{array} { l } { \displaystyle R = J ( A ^ { * } ) \sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) \log \left( \frac { T } { \delta } \right) } } \\ { \displaystyle r = c _ { 1 } J ( A ^ { * } ) \sqrt { \log \left( \frac { n T } { \delta } \right) } } \\ { \displaystyle \overline { { \lambda } } = \sqrt { n } + n ^ { 1 / 4 } \log ^ { 1 / 4 } \left( \frac { T } { \delta } \right) + 2 ^ { 1 / 2 } \log ^ { 1 / 2 } \left( \frac { T } { \delta } \right) } \\ { \displaystyle \overline { { V } } = c _ { 2 } J ( A ^ { * } ) \sqrt { \log \left( \frac { n T } { \delta } \right) } \sqrt { k } \biggl ( \left| \tilde { \mathcal { S } } \right| \operatorname* { m a x } \left\{ \sqrt { \log T } , \sqrt { \log \left( \frac { n } { \delta } \right) } \right\} + \sqrt { T \log \left( \frac { n } { \delta } \right) } \biggr ) . } \end{array}
$$

Since $\begin{array} { r } { \overline { { \lambda } } \geq \sqrt { \log \left( \frac { T } { \delta } \right) } } \end{array}$ , hence $\begin{array} { r } { T \geq c \left( \frac { \overline { { V } } } { \sqrt { \log \left( \frac { T } { \delta } \right) } } R + r ^ { 2 } R ^ { 2 } k \left| \widetilde { \mathcal { S } } \right| \right) } \end{array}$ implies the first condition in (B.5). Also note that since $\left| \widetilde { S } \right| \le 2 s$ , hence

$$
\begin{array} { r l } & { r ^ { 2 } R ^ { 2 } k \left. \widetilde { \mathcal { S } } \right. \leq \widetilde { c } _ { 1 } J ^ { 4 } ( A ^ { \ast } ) s k ^ { 2 } \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) \log \left( \displaystyle \frac { T } { \delta } \right) \log \left( \displaystyle \frac { n T } { \delta } \right) \mathrm { ~ a n d , } } \\ & { \displaystyle \frac { \overline { { V } } } { \sqrt { \log \left( \frac { T } { \delta } \right) } } R \leq \widetilde { c } _ { 2 } J ^ { 2 } ( A ^ { \ast } ) k \sqrt { \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) } \sqrt { \log \left( \displaystyle \frac { n T } { \delta } \right) } \left( s \cdot \operatorname* { m a x } \left\{ \sqrt { \log T } , \sqrt { \log \left( \displaystyle \frac { n } { \delta } \right) } \right\} + \sqrt { T \log \left( \displaystyle \frac { n } { \delta } \right) } \right) . } \end{array}
$$

Now as long as

$$
\begin{array} { r l } & { T \geq c _ { 1 } J ^ { 4 } ( A ^ { * } ) s k ^ { 2 } \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) \log \left( \displaystyle \frac { T } { \delta } \right) \log \left( \displaystyle \frac { n T } { \delta } \right) \mathrm { ~ a n d , } } \\ & { T \geq c _ { 2 } J ^ { 2 } ( A ^ { * } ) k s \sqrt { \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) } \sqrt { \log \left( \displaystyle \frac { n T } { \delta } \right) } \operatorname* { m a x } \left\{ \sqrt { \log T } , \sqrt { \log \left( \displaystyle \frac { n } { \delta } \right) } \right\} , \mathrm { ~ a n d , } } \\ & { T \geq c _ { 3 } J ^ { 2 } ( A ^ { * } ) k \sqrt { \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) } \sqrt { \log \left( \displaystyle \frac { n T } { \delta } \right) } \sqrt { T \log \left( \displaystyle \frac { n } { \delta } \right) } } \end{array}
$$

hold for suitably large constants $c _ { 1 } , c _ { 2 } , c _ { 3 } \ \geq \ 1$ , then $T$ would satisfy (B.5). It is easy to see that if we let, $\begin{array} { r } { T \geq c J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \log \left( \frac { n ^ { 2 } } { k } \right) } \end{array}$ · max $\begin{array} { r } { \left\{ s , \log \left( \frac { n } { \delta } \right) \right\} \log ^ { 2 } \left( \frac { n T } { \delta } \right) } \end{array}$ then (B.5) is ensured. To isolate $T ,$ we can let $M =$ $\textstyle { \sqrt { c } } J ^ { 2 } ( A ^ { * } ) k { \sqrt { \log \left( { \frac { n ^ { 2 } } { k } } \right) } }$ max $\left\{ { \sqrt { s } } , { \sqrt { \log \left( { \frac { n } { \delta } } \right) } } \right\}$ , so that the the aforementioned condition can be rewritten as $\sqrt { T } \geq$ M log $\textstyle \left( { \frac { n T } { \delta } } \right)$ . Using (Papini et al., 2022, Lemma 21), we get that

$$
{ \sqrt { T } } \geq 2 M \log \left( M { \sqrt { \frac { n } { \delta } } } \right)
$$

and taking the square of both sides, we get the suficient condition

$$
T \geq c J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \cdot \log \left( \frac { n ^ { 2 } } { k } \right) \cdot \operatorname* { m a x } \left\{ s , \log \left( \frac { n } { \delta } \right) \right\} \log ^ { 2 } \left( \frac { J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \cdot \log \left( \frac { n ^ { 2 } } { k } \right) \cdot \operatorname* { m a x } \left\{ s , \log \left( \frac { n } { \delta } \right) \right\} n } { \delta } \right) .
$$

Remark 10. Note that,

$$
T \geq c J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \cdot \log \left( { \frac { n ^ { 2 } } { k } } \right) \cdot \operatorname* { m a x } \left\{ s , \log \left( { \frac { n } { \delta } } \right) \right\} \log ^ { 2 } \left( { \frac { J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \cdot \log \left( { \frac { n ^ { 2 } } { k } } \right) \cdot \operatorname* { m a x } \left\{ s , \log \left( { \frac { n } { \delta } } \right) \right\} n } { \delta } } \right)
$$

implies,

$$
{ \frac { s } { T } } \leq { \frac { 1 } { c J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \cdot \log \left( { \frac { n ^ { 2 } } { k } } \right) \cdot \log ^ { 2 } \left( { \frac { n } { \delta } } \right) } }
$$

The final bound. Substituting r in (B.4) we obtain after some simplifications that

$$
\begin{array} { r l } & { \left\| \Delta _ { A } \right\| _ { F } ^ { 2 } + \displaystyle \frac { 1 } { T } \big \| \Delta _ { U } D \big \| _ { F } ^ { 2 } \leq c _ { 1 } J ^ { 2 } ( A ^ { * } ) k \log \left( \frac { T n } { \delta } \right) \left( \operatorname* { m a x } \left\{ \log T , \log \left( \frac { n } { \delta } \right) \right\} \right) \frac { s ^ { 2 } } { T ^ { 2 } } } \\ & { \qquad + c _ { 2 } \frac { J ^ { 2 } ( A ^ { * } ) k \log \left( \frac { T n } { \delta } \right) \log \left( \frac { n } { \delta } \right) } { T } } \\ & { \qquad + c _ { 3 } \operatorname* { m a x } \left\{ n , \log \left( \frac { T } { \delta } \right) \right\} \frac { s } { T } . } \end{array}
$$

Using the result from Remark 10, we can further simplify this to

$$
\begin{array} { l } { \displaystyle \left\| \Delta _ { A } \right\| _ { F } ^ { 2 } + \frac 1 T \left\| \Delta _ { U } D \right\| _ { F } ^ { 2 } \leq c _ { 1 } \frac { J ^ { 2 } ( A ^ { * } ) k \log \left( \frac { T n } { \delta } \right) } { J ^ { 4 } ( A ^ { * } ) k ^ { 2 } \cdot \log \left( \frac { n ^ { 2 } } { k } \right) \cdot \log \left( \frac { n } { \delta } \right) } \frac { s } { T } } \\ { \displaystyle \qquad + c _ { 2 } \frac { J ^ { 2 } ( A ^ { * } ) k \log \left( \frac { T n } { \delta } \right) \log \left( \frac { n } { \delta } \right) } { T } } \\ { \displaystyle \qquad + c _ { 3 } \operatorname* { m a x } \left\{ n , \log \left( \frac { T } { \delta } \right) \right\} \frac { s } { T } } \\ { \displaystyle \qquad \leq c _ { 1 } \frac { J ^ { 2 } ( A ^ { * } ) k \log \left( \frac { T n } { \delta } \right) \log \left( \frac { n } { \delta } \right) } { T } } \\ { \displaystyle \qquad + c _ { 2 } \operatorname* { m a x } \left\{ n , \log ^ { 2 } \left( \frac { T } { \delta } \right) \right\} \frac { s } { T } } \end{array}
$$

which completes the proof.

## B.1 Proof of Lemma 2

Proof. Note that since ${ \widehat { A } } , { \widehat { U } }$ is the optimal solution and $A ^ { * } , U ^ { * }$ is a feasible solution, we get,

$$
\frac { 1 } { 2 } \| \left( \overline { { Y } } - \widehat { A } Y - \widehat { U } \right) D \| _ { F } ^ { 2 } + \lambda \| \widehat { U } D \| _ { 2 , 1 } \leq \frac { 1 } { 2 } \big \| \left( \overline { { Y } } - A ^ { * } Y - U ^ { * } \right) D \big \| _ { F } ^ { 2 } + \lambda \big \| U ^ { * } D \big \| _ { 2 , 1 }
$$

which implies

$$
\frac 1 2 \big \| \big ( \boldsymbol A ^ { * } - \widehat A \big ) \boldsymbol Y \boldsymbol D + \overline { E } \boldsymbol D + \big ( \boldsymbol U ^ { * } - \widehat U \big ) \boldsymbol D \big \| _ { F } ^ { 2 } \leq \frac 1 2 \big \| \overline { E } \boldsymbol D \big \| _ { F } ^ { 2 } + \Big ( \lambda \big \| \boldsymbol U ^ { * } \boldsymbol D \big \| _ { 2 , 1 } - \lambda \big \| \widehat U \boldsymbol D \big \| _ { 2 , 1 } \Big ) .
$$

Denoting $\Delta _ { A } = \widehat { A } - A ^ { * }$ and $\Delta _ { U } = \widehat { U } - U ^ { * }$ note that,

$$
\Delta _ { A } Y + \Delta _ { U } = \underbrace { \left[ \Delta _ { A } \quad \Delta _ { U } \right] } _ { \equiv \Delta } \underbrace { \left[ I \right] } _ { \equiv Y ^ { \prime } } = \Delta Y ^ { \prime } .
$$

Therefore, we $\mathrm { g e t }$

$$
\frac { 1 } { 2 } \big \| ( \Delta Y ^ { \prime } - \overline { { E } } ) D \big \| _ { F } ^ { 2 } \leq \frac { 1 } { 2 } \big \| \widehat { E } D \big \| _ { F } ^ { 2 } + \Big ( \lambda \big \| U ^ { * } D \big \| _ { 2 , 1 } - \lambda \big \| \widehat { U } D \big \| _ { 2 , 1 } \Big )
$$

which is equivalent to,

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \big \| \Delta Y ^ { \prime } D \big \| _ { F } ^ { 2 } \leq \Big \langle \Delta Y ^ { \prime } , \overline { { E } } D \Big \rangle + \Big ( \lambda \big \| U ^ { * } D \big \| _ { 2 , 1 } - \lambda \big \| \widehat { U } D \big \| _ { 2 , 1 } \Big ) } \\ & { \qquad = \Big \langle \Delta _ { A } , \overline { { E } } D Y ^ { \top } \Big \rangle + \Big \langle \Delta _ { U } , \overline { { E } } D \Big \rangle + \lambda \Big ( \big \| U ^ { * } D \big \| _ { 2 , 1 } - \big \| \widehat { U } D \big \| _ { 2 , 1 } \Big ) . } \end{array}
$$

Now note that

$$
\begin{array} { l } { \displaystyle \left. \Delta _ { U } , \overline { { E } } D \right. = \left. \Delta _ { U } D , \overline { { E } } \right. = \displaystyle \sum _ { t = 1 } ^ { T - 1 } \left. D _ { t } ( \widehat { u } _ { t } - { u } _ { t } ^ { * } ) , \eta _ { t + 1 } \right. } \\ { \displaystyle \qquad \leq \displaystyle \sum _ { t = 1 } ^ { T - 1 } \left\| D _ { t } ( \widehat { u } _ { t } - { u } _ { t } ^ { * } ) \right\| _ { 2 } \left\| \eta _ { t + 1 } \right\| _ { 2 } \leq \left( \displaystyle \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} } \left\| \eta _ { t } \right\| _ { 2 } \right) \left\| \Delta _ { U } D \right\| } \\ { \displaystyle \qquad \leq \frac { \lambda } { 2 } \left\| \Delta _ { U } D \right\| } \end{array}
$$

where $\lambda \geq 2 \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} } \left\| \eta _ { t } \right\| _ { 2 }$ . Then we have,

$$
\left. \Delta _ { U } , \overline { { E } } D \right. + \lambda \left( \left\| U ^ { * } D \right\| _ { 2 , 1 } - \left\| \widehat { U } D \right\| _ { 2 , 1 } \right) \leq \lambda \left( \frac 1 2 \big \| \Delta _ { U } D \big \| + \left\| U ^ { * } D \right\| _ { 2 , 1 } - \left\| \widehat { U } D \right\| _ { 2 , 1 } \right) .
$$

By a decomposition over $\widetilde { s }$ and $\widetilde { S } ^ { c }$ , we obtain

$$
\begin{array} { r l } & { \left\| { U ^ { * } D } \right\| _ { 2 , 1 } - \left\| { \widehat { U } D } \right\| _ { 2 , 1 } = \left\| { U ^ { * } D } \right\| _ { 2 , 1 } - \left\| { \Delta _ { U } D + U ^ { * } D } \right\| _ { 2 , 1 } } \\ & { \qquad = \left\| { ( U ^ { * } D ) _ { \widetilde { \mathcal { S } } } } \right\| _ { 2 , 1 } - \left\| { ( \Delta _ { U } D ) _ { \widetilde { \mathcal { S } } } + ( U ^ { * } D ) _ { \widetilde { \mathcal { S } } } } \right\| _ { 2 , 1 } - \left\| { ( \Delta _ { U } D ) _ { \widetilde { \mathcal { S } } ^ { c } } } \right\| _ { 2 , 1 } , } \end{array}
$$

which implies,

$$
\begin{array} { r l } & { \displaystyle \lambda \Big ( \frac { 1 } { 2 } \big \| \Delta _ { U } D \big \| _ { 2 , 1 } + \big \| U ^ { * } D \big \| _ { 2 , 1 } - \big \| \hat { U } D \big \| _ { 2 , 1 } \Big ) } \\ & { \displaystyle = \lambda \Bigg ( \frac { 1 } { 2 } \big \| \Delta _ { U } D \big \| _ { 2 , 1 } + \big \| ( U ^ { * } D ) _ { \tilde { \mathcal { S } } } \big \| _ { 2 , 1 } - \big \| ( \Delta _ { U } D ) _ { \tilde { \mathcal { S } } } + ( U ^ { * } D ) _ { \tilde { \mathcal { S } } } \big \| _ { 2 , 1 } - \big \| ( \Delta _ { U } D ) _ { \tilde { \mathcal { S } } ^ { e } } \big \| _ { 2 , 1 } \Bigg ) } \\ & { \displaystyle \leq \lambda \Bigg ( \frac { 1 } { 2 } \big \| \Delta _ { U } D \big \| _ { 2 , 1 } + \big \| ( \Delta _ { U } D ) _ { \tilde { \mathcal { S } } } \big \| _ { 2 , 1 } - \big \| ( \Delta _ { U } D ) _ { \tilde { \mathcal { S } } ^ { e } } \big \| _ { 2 , 1 } \Bigg ) } \\ & { \displaystyle = \frac { \lambda } { 2 } \Bigg ( 3 \big \| \big ( \Delta _ { U } D \big ) _ { \tilde { \mathcal { S } } } \big \| _ { 2 , 1 } - \big \| \big ( \Delta _ { U } D \big ) _ { \tilde { \mathcal { S } } ^ { e } } \big \| _ { 2 , 1 } \Bigg ) . } \end{array}
$$

Therefore, if $\lambda \geq 2 \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} } \left\| \eta _ { t } \right\| _ { 2 }$ , then,

$$
\frac { 1 } { 2 } \big \| \Delta Y ^ { \prime } D \big \| _ { F } ^ { 2 } \leq \Big \langle \Delta _ { A } , \overline { { E } } D Y ^ { \top } \Big \rangle + \frac { \lambda } { 2 } \Bigg ( 3 \big \| \big ( \Delta _ { U } D \big ) _ { \widetilde { \mathcal { S } } } \big \| _ { 2 , 1 } - \big \| \big ( \Delta _ { U } D \big ) _ { \widetilde { \mathcal { S } ^ { c } } } \big \| _ { 2 , 1 } \Bigg ) .
$$

## B.2 Proof of Lemma 3

Proof. Using the Lemma 1 with $\mu = \left( 1 \ 1 \ . \ . . \ 1 \right) ^ { \intercal }$ , we get

$$
\begin{array} { r } { \mathbb { P } \left( \left\| \eta _ { t } \right\| _ { 2 } ^ { 2 } \leq n + 2 \sqrt { v n } + 2 v \right) \geq 1 - e ^ { - v } . } \end{array}
$$

Letting $\begin{array} { r } { v = \log \left( \frac { T } { \delta } \right) } \end{array}$ , and taking the union over $t = 2 , \ldots , T$ yields

$$
\mathbb { P } \left( \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} } \left. \eta _ { t } \right. _ { 2 } ^ { 2 } \leq n + 2 \sqrt { n \log \left( \frac { T } { \delta } \right) } + 2 \log \left( \frac { T } { \delta } \right) \right) \geq 1 - \delta
$$

and taking the square root leads to

$$
\mathbb { P } \Bigg ( \operatorname* { m a x } _ { t \in \{ 2 , \ldots , T \} } \left. \eta _ { t } \right. _ { 2 } \leq \sqrt { n } + \left( 4 n \log \left( \frac { T } { \delta } \right) \right) ^ { 1 / 4 } + \sqrt { 2 \log \left( \frac { T } { \delta } \right) } \Bigg ) \geq 1 - \delta .
$$

□

## B.3 Proof of Lemma 4

Proof. Note that $\overline { E }$ can be written as follows:

$$
\begin{array} { r } { \overline { { E } } = \left[ \eta _ { 2 } \quad \eta _ { 3 } \quad . . . \quad \eta _ { T } \right] _ { n \times ( T - 1 ) } = \left[ \begin{array} { c } { g _ { 1 } ^ { \top } } \\ { \vdots } \\ { g _ { d } ^ { \top } } \end{array} \right] } \end{array}
$$

where $g _ { i } \overset { \mathrm { i . i . d } } { \sim } \mathcal { N } \left( 0 , I _ { T - 1 } \right)$ . Therefore,

$$
\begin{array} { r } { \operatorname* { s u p } _ { \left\| \boldsymbol { b } \right\| _ { \infty } \leq \boldsymbol { r } , \left\| \boldsymbol { b } \right\| _ { 0 } \leq \left| \widetilde { \mathcal { S } } \right| } \left\| \overline { { E } } \boldsymbol { b } \right\| _ { \infty } = \underset { \boldsymbol { b } \in B \left( \boldsymbol { r } , \left| \widetilde { \mathcal { S } } \right| \right) } { \operatorname* { s u p } } \underset { i \in [ n ] } { \operatorname* { m a x } } \left| g _ { i } ^ { \top } \boldsymbol { b } \right| } \\ { = \underset { i \in [ n ] } { \operatorname* { m a x } } \underset { \boldsymbol { b } \in B \left( \boldsymbol { r } , \left| \widetilde { \mathcal { S } } \right| \right) } { \operatorname* { s u p } } \left| g _ { i } ^ { \top } \boldsymbol { b } \right| . } \end{array}
$$

Now consider $\begin{array} { r } { \operatorname* { s u p } _ { b \in \boldsymbol { B } ( r , | \widetilde { \mathcal { S } } | ) } | g ^ { \top } b | } \end{array}$ for $g { \sim } { \mathcal { N } } \left( 0 , I _ { T - 1 } \right)$ . We can see that bounding m $\begin{array} { r } { \operatorname { 1 a x } _ { i \in [ n ] } \operatorname* { s u p } _ { b \in { \mathcal { B } } \left( r , | \widetilde { \mathcal { S } } | \right) } \left| g _ { i } ^ { \top } b \right| } \end{array}$ can be achieved by finding an upper bound on $\begin{array} { r } { \operatorname* { s u p } _ { b \in B ( r , | \widetilde { \mathcal { S } } | ) } | g ^ { \top } b | } \end{array}$ and then applying union bound over $i \in [ n ]$ . To this end, we show that the supremum function in question is Lipschitz and then use concentration bounds for Lipschitz functions of standard i.i.d Gaussian’s to achieve the desired bound.

Remark 11. Let $\begin{array} { r } { f ( x ) = \operatorname* { s u p } _ { b \in B \left( r , | \widetilde { S } | \right) } | x ^ { \top } b | } \end{array}$ . We can see that $f ( x )$ is Lipschitz over $\mathbb { R }$ with respect to $\left\| \cdot \right\| _ { 2 }$ norm. This can be seen by observing,

$$
\begin{array} { r l } & { | f ( x ) - f ( y ) | \leq ( \underset { b \in \mathcal { B } ( r , | \widetilde { \mathcal { S } } | ) } { \operatorname* { s u p } }  b  _ { 2 } )  x - y  _ { 2 } } \\ & { \qquad \leq r | \widetilde { \mathcal { S } } |  x - y  _ { 2 } } \end{array}
$$

where the last inequality follows from the fact that $\left\| b \right\| _ { 2 } \leq \left\| b \right\| _ { 1 } \leq r \left| \widetilde { S } \right|$ holds for all $b \in B \left( r , \left| \widetilde { S } \right| \right)$

Given Remark 11, we obtain from Theorem 3 (Wainwright, 2019, from Chapter 3) that for any $q > 0$

$$
\mathbb { P } \left( \left| \operatorname* { s u p } _ { b \in \mathcal { B } ( r , | \tilde { \mathcal { S } } | ) } \left| g ^ { \top } b \right| - \mathbb { E } \left( \operatorname* { s u p } _ { b \in \mathcal { B } ( r , | \tilde { \mathcal { S } } | ) } \left| g ^ { \top } b \right| \right) \right| \geq q \right) \leq 2 \exp \left( \frac { - c q ^ { 2 } } { r ^ { 2 } } \right) .
$$

Let $\begin{array} { r } { \mu \left( \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) \right) : = \mathbb { E } \left( \operatorname* { s u p } _ { b \in \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) } \left| g ^ { \top } b \right| \right) } \end{array}$ where we recall that $\mu \left( \boldsymbol { B } \left( \boldsymbol { r } , \left| \boldsymbol { \widetilde { S } } \right| \right) \right)$ is known as the Gaussian complexity of $\boldsymbol { B } \left( \boldsymbol { r } , \left| \boldsymbol { \widetilde { s } } \right| \right)$ (Vershynin, 2018, see Section 7.5.3). This can be bounded in terms of the Gaussian width of $\boldsymbol { B } \left( \boldsymbol { r } , \left| \boldsymbol { \widetilde { s } } \right| \right)$ as follows, see (Vershynin, 2018, Lemma 7.5.11).

$$
\mu \left( \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) \right) \lesssim w \left( \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) \right) + \left\| y \right\| _ { 2 } \mathrm { f o r } y \in \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) .
$$

Since $0 \in B \left( r , \left| \widetilde { S } \right| \right)$ , we have,

$$
\mu \left( \boldsymbol { B } \left( r , \left| \widetilde { \boldsymbol { S } } \right| \right) \right) \lesssim w \left( \boldsymbol { B } \left( r , \left| \widetilde { \boldsymbol { S } } \right| \right) \right) .
$$

Therefore, with probability at least $\begin{array} { r } { 1 - 2 \exp \left( \frac { - c q ^ { 2 } } { r ^ { 2 } \left| \widetilde { S } \right| ^ { 2 } } \right) } \end{array}$

$$
\operatorname* { s u p } _ { \boldsymbol { b } \in \boldsymbol { B } \left( \boldsymbol { r } , \left| \widetilde { \boldsymbol { S } } \right| \right) } \left| \boldsymbol { g } ^ { \top } \boldsymbol { b } \right| \leq c _ { 1 } w \left( \boldsymbol { B } \left( \boldsymbol { r } , \left| \widetilde { \boldsymbol { S } } \right| \right) \right) + q
$$

for some constant $c _ { 1 }$ . Letting $q = c _ { 2 } r \left| \widetilde { S } \right| \sqrt { \log \left( \frac { n } { \delta } \right) }$ for an appropriate constant $c _ { 2 }$ , we get,

$$
\mathbb { P } \Bigg ( \operatorname* { s u p } _ { b \in \mathcal { B } ( r , | \tilde { \mathcal { S } } | ) } | g ^ { \top } b | \leq c _ { 1 } w ( \mathcal { B } ( r , | \tilde { \mathcal { S } } | ) ) + c _ { 2 } r | \tilde { \mathcal { S } } | \sqrt { \log ( \frac { n } { \delta } ) } ) \geq 1 - \frac { \delta } { n }
$$

and taking the union bound over $i \in [ n ]$ ，

$$
\begin{array} { r } { \mathbb { P } \bigg ( \big \| \overline { { E } } D _ { \widetilde { \mathcal { S } } } Y _ { \widetilde { \mathcal { S } } } ^ { \top } \big \| _ { \infty , \infty } \leq c \bigg ( w \left( \mathcal { B } \left( r , \left| \widetilde { \mathcal { S } } \right| \right) \right) + r \left| \widetilde { \mathcal { S } } \right| \sqrt { \log \left( \frac { n } { \delta } \right) } \bigg ) \bigg ) \geq 1 - \delta . } \end{array}
$$

Therefore, $\mathbb { P } \left( \mathcal { E } _ { 1 a } \right) \ge 1 - \delta .$

## B.4 Proof of Lemma 5

Proof. Let $\eta = \operatorname { v e c } \left( \left[ \eta _ { 1 } \quad \eta _ { 2 } \quad \dots \quad \eta _ { T } \right] \right)$ and note that $x _ { t } = \widetilde { A } ( t ) \eta$ where

$$
\widetilde { A } ( t ) : = \big [ ( A ^ { * } ) ^ { t - 1 } \quad ( A ^ { * } ) ^ { t - 2 } \quad \ldots \quad I _ { n } \quad 0 \quad \ldots \quad 0 \big ] \in \mathbb { R } ^ { n \times n T } .
$$

For any $i \in [ n ] , x _ { t i } = a ^ { \top } ( t , i ) \eta$ where $a ^ { \top } ( t , i )$ is the i-th row of $\widetilde { A } ( t )$ and $x _ { t i }$ is the i-th element of $x _ { t }$ . Note that,

$$
\mathbb { E } \left( x _ { t i } ^ { 2 } \right) = \left. a ( t , i ) \right. _ { 2 } ^ { 2 } \leq \left. \widetilde { A } ( t ) \right. _ { 2 } ^ { 2 } .
$$

For any $v = \mathrm { v e c } \left( \left[ v _ { 1 } \quad v _ { 2 } \quad \ldots \quad v _ { T } \right] \right)$ with $\| v \| _ { 2 } = 1$

$$
\bigl \| \widetilde { A } ( t ) v \bigr \| _ { 2 } = \bigl \| \sum _ { t = 1 } ^ { T } ( A ^ { * } ) ^ { t - i } v _ { i } \bigr \| _ { 2 } \leq \sum _ { t = 1 } ^ { T } \bigl \| ( A ^ { * } ) ^ { t - i } \bigr \| _ { 2 } \leq \sum _ { t = 1 } ^ { \infty } \bigl \| ( A ^ { * } ) ^ { t - i } \bigr \| _ { 2 } \leq J ( A ^ { * } ) ,
$$

and therefore, E $\left( x _ { t i } ^ { 2 } \right) \leq J ^ { 2 } ( A ^ { * } )$

Note that since $\eta$ contains i.i.d standard Gaussian entries, $x _ { t i } \sim \mathcal { N } \left( 0 , \left\| a ( t , i ) \right\| _ { 2 } ^ { 2 } \right)$ , and therefore, $\begin{array} { r l } { x _ { t i } } & { { } = } \end{array}$ $\| a ( t , i ) \| _ { 2 } W _ { t i }$ where $W _ { t i } \sim \mathcal { N } ( 0 , 1 )$ . As $W _ { t i } ^ { 2 }$ is sub-exponential, we have

$$
\mathbb { P } \left( W _ { t i } ^ { 2 } \geq q \right) \leq 2 \exp \left( \frac { - q } { c \left\| W _ { t i } ^ { 2 } \right\| _ { \psi _ { 1 } } } \right) = 2 \exp \left( \frac { - q } { c \left\| W _ { t i } \right\| _ { \psi _ { 2 } } ^ { 2 } } \right) = 2 \exp \left( \frac { - q } { c ^ { \prime } } \right) .
$$

By setting $\begin{array} { r } { q = c ^ { \prime } \log \left( \frac { T n } { \delta } \right) } \end{array}$ for all $\delta \in ( 0 , c _ { 1 } )$ for constants $c ^ { \prime }$ and $c _ { 1 } < 1$ , we get P $\begin{array} { r } { \left( W _ { t i } ^ { 2 } \geq c ^ { \prime } \log \left( \frac { T n } { \delta } \right) \right) \leq \frac { \delta } { T n } } \end{array}$ and therefore, $\begin{array} { r } { \mathbb { P } \left( x _ { t i } ^ { 2 } \leq c ^ { \prime } \dot { J } ^ { 2 } ( \dot { A } ^ { * } ) \log \left( \frac { T n } { \delta } \right) \right) \geq 1 - \frac { \delta } { T n } } \end{array}$ . Taking a union bound over $t \in [ T ]$ and $i \in [ n ]$ yields

$$
\mathbb { P } \left( \forall t \in [ T ] : \left. x _ { t } \right. _ { \infty } \leq c J ( A ^ { * } ) \sqrt { \log \left( \frac { T n } { \delta } \right) } \right) \geq 1 - \delta .
$$

Taking $r = c J ( A ^ { * } ) \sqrt { \log \left( \frac { T n } { \delta } \right) }$ , this clearly implies $\mathbb { P } \left( \mathcal { E } _ { 1 b } \right) \ge { 1 - \delta }$

## B.5 Proof of Lemma 6

Proof. Let $b _ { t } = \mathbb { 1 } \left\{ t \in  { \widetilde { S } } ^ { c } \right\}$ and let $M _ { T } = \overline { { E } } Y _ { \widetilde { s } ^ { c } } ^ { \top }$ . We can see that,

$$
M _ { T } = \sum _ { t \in \widetilde { \cal S } ^ { c } } \eta _ { t + 1 } x _ { t } ^ { \top } = \sum _ { t \in [ T - 1 ] } b _ { t } \eta _ { t + 1 } x _ { t } ^ { \top } .
$$

Note that for any $i , j \in [ n ]$ , we have

$$
M _ { T , i j } = e _ { i } ^ { \top } M _ { T } e _ { j } = \sum _ { t \in [ T - 1 ] } b _ { t } \eta _ { t + 1 , i } x _ { t , j } ^ { \top } .
$$

Let $\mathcal { F } _ { t } = \sigma ( \eta _ { 1 } , \dots , \eta _ { t } )$ denote the sigma-algebra at any time $t \geq 1$ . Then $b _ { t } x _ { t }$ is $\mathcal { F } _ { t }$ measurable while $\eta _ { t + 1 } | \mathcal { F } _ { t }$ is sub-Gaussian, for all $t \geq 1$ . We can hence bound $\vert M _ { T , i j } \vert$ using the tail bound for self normalized martingales from Theorem 4.

To this end, let us first denote $\begin{array} { r } { \overline { { v } } = v + \sum _ { t \in [ T - 1 ] } b _ { t } x _ { t j } ^ { 2 } } \end{array}$ (for $v > 0 )$ . Then, there exist constants $c _ { 1 } \geq 1$ and $c _ { 2 } < 1$ such that

$$
\mathbb { P } \left( \overline { { v } } ^ { - 1 } M _ { i j } ^ { 2 } \le c _ { 1 } \log \left( \frac { \overline { { v } } ^ { 1 / 2 } v ^ { - 1 / 2 } } { \delta } \right) \right) \ge 1 - \delta
$$

for $\delta \in ( 0 , c _ { 2 } )$ . Let $\mathcal { E } _ { 1 c } ^ { \prime }$ denote the above event and note that on $\mathcal { E } _ { 1 b }$

$$
\sum _ { t \in [ T - 1 ] } b _ { t } x _ { t j } ^ { 2 } \leq \sum _ { t \in [ T - 1 ] } x _ { t j } ^ { 2 } \leq T r ^ { 2 } .
$$

Choosing $v = T r ^ { 2 }$ , this implies $\overline { { v } } \le v + T r ^ { 2 } \le 2 T r ^ { 2 }$

Hence on $\mathcal { E } _ { 1 b } \cap \mathcal { E } _ { 1 c } ^ { \prime }$ , we have shown that

$$
\frac { 1 } { 2 T r ^ { 2 } } \left| M _ { i j } \right| ^ { 2 } \leq c _ { 1 } \log \left( \frac { 2 } { \delta } \right) \iff \left| M _ { i j } \right| ^ { 2 } \leq 2 c _ { 1 } r ^ { 2 } T \log \left( \frac { 2 } { \delta } \right) .
$$

By suitably rescaling δ and taking a union bound over $i , j \in [ n ]$ , we then obtain

$$
\mathbb { P } \left( \forall i , j \in [ n ] : | M _ { i j } | \leq c _ { 1 } r \sqrt { T \log \left( \frac { n } { \delta } \right) } \right) \geq 1 - \delta .
$$

Since $\mathcal { E } _ { 1 b } \cap \mathcal { E } _ { 1 c } ^ { \prime } \subseteq \mathcal { E } _ { 1 c } ,$ , on $\mathcal { E } _ { 1 b } \cap \mathcal { E } _ { 1 c } ^ { \prime }$ , we obtain the statement of the lemma after suitably rescaling δ.

## B.6 Proof of Lemma 7

Proof. Consider $\left. \Delta _ { A } Y D \right. _ { F } ^ { 2 }$ . Then,

$$
\begin{array} { r l r } { \big \| \Delta _ { A } Y D \big \| _ { F } ^ { 2 } = \big \| \Delta _ { A } Y D _ { \widetilde { \mathcal { S } } } \big \| _ { F } ^ { 2 } + \big \| \Delta _ { A } Y D _ { \widetilde { \mathcal { S } } ^ { c } } \big \| _ { F } ^ { 2 } } & { } & \\ { \geq \big \| \Delta _ { A } Y D _ { \widetilde { \mathcal { S } } ^ { c } } \big \| _ { F } ^ { 2 } } & { } & \\ { = \big \| \Delta _ { A } X D _ { \widetilde { \mathcal { S } } ^ { c } } \big \| _ { F } ^ { 2 } } & { } & { \mathrm { ( s i n c e ~ } Y D _ { \widetilde { \mathcal { S } } ^ { c } } = X D _ { \widetilde { \mathcal { S } } ^ { c } } \big ) } \\ { \geq \big \| \Delta _ { A } \big \| _ { F } ^ { 2 } \underset { H \in \widetilde { \mathcal { C } } _ { \kappa } } { \operatorname* { i n f } } \big \| H X D _ { \widetilde { \mathcal { S } ^ { c } } } \big \| _ { F } ^ { 2 } , } & { } & \end{array}
$$

where the last inequality comes from the fact that $\frac { \Delta _ { A } } { \left\| \Delta _ { A } \right\| _ { F } } \in \mathcal { C } _ { K }$

Note that vec $( H X D _ { \tilde { S } ^ { c } } ) = ( I _ { T - 1 } \otimes H ) \operatorname { v e c } ( X D _ { \tilde { S } ^ { c } } ) = ( I _ { T - 1 } \otimes H ) ( D _ { \tilde { S } ^ { c } } \otimes I _ { n } ) \operatorname { v e c } ( X ) = ( D _ { \tilde { S } ^ { c } } \otimes H ) \operatorname { v e c } ( X ) .$

Let $x = \operatorname { v e c } ( X ) , \eta = \operatorname { v e c } \left( \left[ \eta _ { 1 } \quad \eta _ { 2 } \quad . . . \quad \eta _ { T - 1 } \right] \right)$ and denote $\Gamma \in \mathbb { R } ^ { n ( T - 1 ) \times n ( T - 1 ) } { \mathrm { ~ a s ~ } }$

$$
\Gamma = \left[ \begin{array} { c c c c c } { { I } } & { { 0 } } & { { 0 } } & { { \ldots } } & { { 0 } } \\ { { A ^ { * } } } & { { I } } & { { 0 } } & { { \ldots } } & { { 0 } } \\ { { \left( A ^ { * } \right) ^ { 2 } } } & { { A ^ { * } } } & { { I } } & { { \ldots } } & { { 0 } } \\ { { \vdots } } & { { \vdots } } & { { \vdots } } & { { \ldots } } & { { \vdots } } \\ { { \left( A ^ { * } \right) ^ { T - 2 } } } & { { \left( A ^ { * } \right) ^ { T - 3 } } } & { { \left( A ^ { * } \right) ^ { T - 4 } } } & { { \ldots } } & { { I } } \end{array} \right] .
$$

We can see that $x = \Gamma \eta$ , and hence, $\operatorname { v e c } ( H X D _ { \widetilde { S } ^ { c } } ) = ( D _ { \widetilde { S } ^ { c } } \otimes H ) \Gamma \eta .$

Consider the set $\mathcal { M } = \left\{ ( D _ { \widetilde { \mathcal { S } } ^ { c } } \otimes H ) \Gamma : H \in \widetilde { C } _ { \mathcal { K } } \right\}$ . We then have that

$$
\begin{array} { l } { \underset { H \in \widetilde { C } _ { \kappa } } { \operatorname* { i n f } } \left. H X D _ { \widetilde { \mathcal { S } ^ { c } } } \right. _ { F } ^ { 2 } = \underset { H \in \widetilde { C } _ { \kappa } } { \operatorname* { i n f } } \left. \mathrm { v e c } \left( H X D _ { \widetilde { \mathcal { S } ^ { c } } } \right) \right. _ { 2 } ^ { 2 } } \\ { = \underset { M \in \mathcal { M } } { \operatorname* { i n f } } \left. M \eta \right. _ { 2 } ^ { 2 } . } \end{array}
$$

Applying Theorem 2 (Theorem 3.1 of Krahmer et al. (2014)) to M yields

$$
\mathbb { P } \left( \operatorname* { s u p } _ { M \in \mathcal { M } } \left| \left. M \eta \right. _ { 2 } ^ { 2 } - \mathbb { E } \left( \left. M \eta \right. _ { 2 } ^ { 2 } \right) \right| \geq c _ { 1 } P + q \right) \leq 2 \exp \left( - c _ { 2 } \operatorname* { m i n } \left\{ \frac { q ^ { 2 } } { F ^ { 2 } } , \frac { q } { L } \right\} \right) .\tag{B.6}
$$

where

$$
\begin{array} { r l } & { P = \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) \left( \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) + d _ { F } ( \mathcal { M } ) \right) + d _ { F } ( \mathcal { M } ) d _ { 2 } ( \mathcal { M } ) , } \\ & { F = d _ { 2 } ( \mathcal { M } ) \left( \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) + d _ { F } ( \mathcal { M } ) \right) , } \\ & { \mathrm { a n d ~ } L = d _ { 2 } ^ { 2 } ( \mathcal { M } ) } \end{array}
$$

We start by observing that for any $M \in \mathcal { M } , \left. M \right. _ { F } ^ { 2 } \geq \left| \widetilde { \mathcal { S } } ^ { c } \right| \left. H \right. _ { F } ^ { 2 } = \left| \widetilde { \mathcal { S } } ^ { c } \right|$ . It was shown in Jedra and Proutiere (2020) that $\left\| \Gamma \right\| _ { 2 } \leq J ( A ^ { * } )$ . Therefore,

$$
d _ { 2 } \bigl ( \mathcal { M } \bigr ) = \operatorname* { s u p } _ { H \in \tilde { C } _ { \kappa } } \bigl \| \bigl ( D _ { \tilde { S } ^ { c } } \otimes H \bigr ) \Gamma \bigr \| _ { 2 } \leq \operatorname* { s u p } _ { H \in \tilde { C } _ { \kappa } } \bigl \| D _ { \tilde { S } ^ { c } } \bigr \| _ { 2 } \bigl \| H \bigr \| _ { 2 } \bigl \| \Gamma \bigr \| _ { 2 } \leq J ( A ^ { * } ) \mathrm { ~ a n d , }
$$

$$
d _ { F } ( \mathcal { M } ) = \operatorname* { s u p } _ { H \in \tilde { C } _ { K } } \left\| ( D _ { \tilde { \mathcal { S } } ^ { c } } \otimes H ) \Gamma \right\| _ { F } \leq \left\| \Gamma \right\| _ { 2 } \operatorname* { s u p } _ { H \in \tilde { C } _ { K } } \left\| D _ { \tilde { \mathcal { S } } ^ { c } } \otimes H \right\| _ { F } = J ( A ^ { * } ) \operatorname* { s u p } _ { H \in \tilde { C } _ { K } } \left\| H \right\| _ { F } \left\| D _ { \tilde { \mathcal { S } } ^ { c } } \right\| _ { F } \leq J ( A ^ { * } ) \sqrt { T } .
$$

We next show that the map $( \widetilde { C } \kappa , \| \cdot \| _ { 2 } ) \mapsto ( { M } , \| \cdot \| _ { 2 } )$ is Lipschitz. For any $H _ { 1 } , H _ { 2 } \in \widetilde { C } _ { \mathcal { K } }$

$$
\begin{array} { r } { \big \| ( D _ { \widetilde { S } ^ { \epsilon } } \otimes H _ { 1 } ) \Gamma - ( D _ { \widetilde { S } ^ { \epsilon } } \otimes H _ { 2 } ) \Gamma \big \| _ { 2 } = \big \| ( D _ { \widetilde { S } ^ { \epsilon } } \otimes ( H _ { 1 } - H _ { 2 } ) ) \Gamma \big \| _ { 2 } \leq \big \| D _ { \widetilde { S } ^ { \epsilon } } \big \| _ { 2 } \big \| \Gamma \big \| _ { 2 } \big \| H _ { 1 } - H _ { 2 } \big \| _ { 2 } \leq J ( A ^ { \star } ) \big \| H _ { 1 } - H _ { 2 } \big \| _ { 2 } . } \end{array}
$$

Using Property 3 recalled earlier in Section A, we then obtain,

$$
\begin{array} { r } { \gamma _ { 2 } \left( \mathcal { M } , \lVert \cdot \rVert _ { 2 } \right) \leq c J ( A ^ { * } ) \gamma _ { 2 } \left( \widetilde { C } \kappa , \lVert \cdot \rVert _ { 2 } \right) . } \end{array}
$$

Let us now consider the following respective upper bounds for $P , Q , L$

$$
\overline { { { P } } } = c _ { 1 } J ^ { 2 } ( A ^ { * } ) \gamma _ { 2 } \left( \widetilde { C } _ { K } , \lVert \cdot \rVert _ { 2 } \right) \left( \gamma _ { 2 } \left( \widetilde { C } _ { K } , \lVert \cdot \rVert _ { 2 } \right) + \sqrt { T } \right) + J ^ { 2 } ( A ^ { * } ) \sqrt { T } ,
$$

$$
\overline { { { F } } } = c _ { 1 } J ^ { 2 } ( A ^ { * } ) \left( \gamma _ { 2 } \left( \widetilde { C } _ { K } , \left. \cdot \right. _ { 2 } \right) + \sqrt { T } \right) ,
$$

and, $\overline { { { L } } } = J ^ { 2 } ( A ^ { * } )$

for suficiently large $c _ { 1 } > 1$ . Then from (B.6),

$$
\mathbb { P } \left( \operatorname* { i n f } _ { M \in \mathcal { M } } \left\| M \eta \right\| _ { 2 } ^ { 2 } \geq \left| \widetilde { \mathcal { S } } ^ { c } \right| - c _ { 2 } \overline { { P } } - q \right) \geq 1 - 2 \exp \left( - c _ { 3 } \operatorname* { m i n } \left\{ \frac { q ^ { 2 } } { \overline { { F } } ^ { 2 } } , \frac { q } { \overline { { L } } } \right\} \right) .
$$

Choose $q = \overline { { F } } \sqrt { m }$ for $m \geq 1$ . Since ${ \overline { { F } } } \geq { \overline { { L } } } .$ , we have min $\textstyle \left\{ { \frac { q ^ { 2 } } { \overline { { F } } ^ { 2 } } } , { \frac { q } { L } } \right\} = { \sqrt { m } } .$ hence,

$$
\begin{array} { r } { \mathbb { P } \left( \underset { M \in \mathcal { M } } { \operatorname* { i n f } } \left. M \eta \right. _ { 2 } ^ { 2 } \geq \left| \widetilde { \mathcal { S } } ^ { c } \right| - c _ { 2 } \overline { { P } } - \overline { { F } } \sqrt { m } \right) \geq 1 - 2 \exp \left( - c _ { 3 } \sqrt { m } \right) . } \end{array}
$$

Using Property 1 (since $\left\| \cdot \right\| _ { 2 } \leq \left\| \cdot \right\| _ { F } )$ and recalling the inequality in (A.1), we have

$$
\begin{array} { r } { \gamma _ { 2 } \left( \widetilde { C } _ { K } , \left. \cdot \right. _ { 2 } \right) \lesssim \gamma _ { 2 } \left( \widetilde { C } _ { K } , \left. \cdot \right. _ { F } \right) \asymp w \left( \widetilde { C } _ { K } \right) = w \left( \widetilde { C } _ { K , \mathrm { v e c } } \right) , } \end{array}\tag{B.7}
$$

where $\widetilde { C } _ { K , \mathrm { v e c } }$ is defined as

$$
\widetilde { C } _ { \mathcal { K } , \mathrm { v e c } } = \left\{ \mathrm { v e c } ( H ) \in \mathbb { R } ^ { n ^ { 2 } } : \left\| ( H ) _ { \mathcal { K } ^ { c } } \right\| _ { 1 , 1 } \leq \left\| ( H ) _ { \mathcal { K } } \right\| _ { 1 , 1 } \right\} \cap \left\{ x \in \mathbb { R } ^ { n ^ { 2 } } : \left\| x \right\| _ { 2 } = 1 \right\} .
$$

The last equality in (B.7) follows from the fact that $( \widetilde { C } _ { \mathcal { K } } , \Vert \cdot \Vert _ { F } )  ( \widetilde { C } _ { \mathcal { K } , \mathrm { v e c } } , \Vert \cdot \Vert _ { 2 } )$ is a bijective map. It was shown in (Foucart and Rauhut, 2013, Proposition 9.33) that $w ( \widetilde { C } _ { K , \mathrm { v e c } } ) \lesssim \sqrt { k \log { \left( \frac { n ^ { 2 } } { k } \right) } }$ which implies $\gamma _ { 2 } \left( \widetilde { C } _ { K } , \left. \cdot \right. _ { 2 } \right) \lesssim$ $\sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) }$

Therefore choosing $m = c \log ^ { 2 } \left( { \textstyle { \frac { 1 } { \delta } } } \right)$ , we have that

$$
\begin{array} { r l } & { c _ { 2 } \overline { { P } } + \overline { { F } } \sqrt { m } \leq c ^ { \prime } J ^ { 2 } ( A ^ { * } ) \Bigg ( k \log n + \sqrt { k T \log n } + \sqrt { m T } + \sqrt { m k T \log n } \Bigg ) } \\ & { \qquad \leq \overline { { c } } J ^ { 2 } ( A ^ { * } ) \Bigg ( k \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) + \sqrt { k T \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) } + \log \left( \displaystyle \frac { 1 } { \delta } \right) \left( \sqrt { T } + \sqrt { k T \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) } \right) \Bigg ) } \\ & { \qquad \leq \overline { { c } } J ^ { 2 } ( A ^ { * } ) \log \left( \displaystyle \frac { 1 } { \delta } \right) \left( k \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) + \sqrt { k T \log \left( \displaystyle \frac { n ^ { 2 } } { k } \right) } \right) . } \end{array}
$$

Let $\begin{array} { r } { T \geq c _ { 1 } J ^ { 4 } ( A ^ { * } ) \log ^ { 2 } \left( \frac { 1 } { \delta } \right) k \log \left( \frac { n ^ { 2 } } { k } \right) + 4 s } \end{array}$ for a suitably large constant $c _ { 1 }$ . Then we have that

$$
\frac { \left| \widetilde { S } ^ { c } \right| } { 2 } \geq \frac { T - 2 s - 1 v } { 2 } \geq \overline { { c J } } ^ { 2 } ( A ^ { * } ) \log \left( \frac { 1 } { \delta } \right) \left( k \log \left( \frac { n ^ { 2 } } { k } \right) + \sqrt { k T \log \left( \frac { n ^ { 2 } } { k } \right) } \right) .
$$

Therefore with probability $\geq 1 - \delta .$ , it holds that inf $\smash { \operatorname { \ 4 6 4 } \left\| \boldsymbol { M } \eta \right\| _ { 2 } ^ { 2 } \geq \frac { | \widetilde { \mathcal { S } } ^ { c } | } { 2 } }$ . Note that since $T \geq 4 s$ , this implies $\left| \widetilde { S } ^ { c } \right| \ge T - 2 s \ge T / 2$ which completes the proof. □

## B.7 Proof of Lemma 8

Proof. Note that,

$$
\begin{array} { r l } & { \left| \left. \Delta _ { A } Y D , \Delta _ { U } D \right. \right| \leq \left| \left. \Delta _ { A } Y D , ( \Delta _ { U } D ) _ { \widetilde { S } } \right. \right| + \left| \left. \Delta _ { A } Y D , ( \Delta _ { U } D ) _ { \widetilde { S } ^ { c } } \right. \right| } \\ & { \qquad \leq \left| \left. \Delta _ { A } Y _ { \widetilde { S } } D _ { \widetilde { S } } , ( \Delta _ { U } D ) _ { \widetilde { S } } \right. \right| + \left| \left. \Delta _ { A } Y _ { \widetilde { S } ^ { c } } D _ { \widetilde { S } ^ { c } } , ( \Delta _ { U } D ) _ { \widetilde { S } ^ { c } } \right. \right| } \\ & { \qquad \leq \left\| ( \Delta _ { U } D ) _ { \widetilde { S } } \right\| _ { 2 , 1 } \left\| \Delta _ { A } Y _ { \widetilde { S } } D _ { \widetilde { S } } \right\| _ { 2 , \infty } + \left\| ( \Delta _ { U } D ) _ { \widetilde { S } ^ { c } } \right\| _ { 2 , 1 } \left\| \Delta _ { A } Y _ { \widetilde { S } ^ { c } } D _ { \widetilde { S } ^ { c } } \right\| _ { 2 , \infty } . } \end{array}
$$

Recall that $\begin{array} { r } { \big \| ( \Delta _ { U } D ) _ { \widetilde { \mathcal { S } } ^ { c } } \big \| _ { 2 , 1 } \leq 3 \big \| ( \Delta _ { U } D ) _ { \widetilde { \mathcal { S } } } \big \| _ { 2 , 1 } + \frac { \overline { { V } } } { \lambda } \big \| \Delta _ { A } \big \| _ { F } } \end{array}$ . Therefore,

$$
\left| \left. \Delta _ { A } Y D , \Delta _ { U } D \right. \right| \leq \left\| \left( \Delta _ { U } D \right) _ { \tilde { \mathcal { S } } } \right\| _ { 2 , 1 } \left\| \Delta _ { A } Y _ { \tilde { \mathcal { S } } } D _ { \tilde { \mathcal { S } } } \right\| _ { 2 , \infty } + \left( 3 \left\| \left( \Delta _ { U } D \right) _ { \tilde { \mathcal { S } } } \right\| _ { 2 , 1 } + \frac { V } { \lambda } \| \Delta _ { A } \| _ { F } \right) \left\| \Delta _ { A } Y _ { \tilde { \mathcal { S } } ^ { c } } D _ { \tilde { \mathcal { S } } ^ { c } } \right\| _ { 2 , \infty } .
$$

In order to control $\left| \left. \Delta _ { A } Y D , \Delta _ { U } D \right. \right|$ , we need to control $\left\| \Delta _ { A } Y _ { \widetilde { S } ^ { c } } D _ { \widetilde { S } ^ { c } } \right\| _ { 2 , \infty }$ and $\left\| \Delta _ { A } Y _ { \widetilde { S } } D _ { \widetilde { S } } \right\| _ { 2 , \infty }$

We can first consider $\| \Delta _ { A } Y _ { \widetilde { S } } D _ { \widetilde { S } } \| _ { 2 , \infty }$ . Note that,

$$
\left\| \Delta _ { A } Y _ { \tilde { \mathcal { S } } } D _ { \tilde { \mathcal { S } } } \right\| _ { 2 , \infty } = \operatorname* { m a x } _ { t } \left\| \Delta _ { A } ( Y _ { \tilde { \mathcal { S } } } D _ { \tilde { \mathcal { S } } } ) _ { : , t } \right\| _ { 2 } \leq \left\| \Delta _ { A } \right\| _ { 1 , 1 } \operatorname* { m a x } _ { t } \left\| ( Y _ { \tilde { \mathcal { S } } } D _ { \tilde { \mathcal { S } } } ) _ { : , t } \right\| _ { \infty } \leq r \left\| \Delta _ { A } \right\| _ { 1 , 1 } \leq 2 r \sqrt { k } \left\| \Delta _ { A } \right\| _ { F } .
$$

where $( Y _ { \widetilde { S } } D _ { \widetilde { S } } ) _ { : , t }$ represents the t-th column of $Y _ { \widetilde { s } } D _ { \widetilde { s } }$

Next, let us take $\| \Delta _ { A } Y _ { \widetilde { S } ^ { c } } D _ { \widetilde { S } ^ { c } } \| _ { 2 , \infty }$ . Observe that,

$$
\begin{array} { r l } { \left\| \Delta _ { A } Y _ { \tilde { S } ^ { c } } D _ { \tilde { S } ^ { c } } \right\| _ { 2 , \infty } = } & { \operatorname* { m a x } _ { i \in [ T - 1 ] } \left\| \Delta _ { A } ( Y _ { \tilde { S } ^ { c } } D _ { \tilde { S } ^ { c } } ) _ { \varepsilon , i } \right\| _ { 2 } } \\ & { = \operatorname* { m a x } _ { i \in [ T - 1 ] } \left\| \Delta _ { A } ( X _ { \tilde { S } ^ { c } } D _ { \tilde { S } ^ { c } } ) _ { \varepsilon , i } \right\| _ { 2 } } \\ & { \leq \operatorname* { m a x } _ { i \in [ T ] } \left\| \Delta _ { A } X _ { \varepsilon , i } \right\| _ { 2 } } \\ & { = \operatorname* { m a x } _ { i \in [ T ] } \left\| \Delta _ { A } x _ { \varepsilon , i } \right\| _ { 2 } } \\ & { = \operatorname* { m a x } _ { i \in [ T ] } \left\| \Delta _ { A } x _ { \varepsilon , i } \right\| _ { 2 } } \\ & { = \left\| \Delta _ { A } \right\| _ { F _ { \varepsilon } ^ { \mathrm { H G T } } } \left\| \frac { \Delta _ { A } } { \left\| \Delta _ { A } \right\| _ { F } } x _ { \varepsilon } \right\| _ { 2 } } \\ & { \leq \left\| \Delta _ { A } \right\| _ { F _ { \varepsilon } ^ { \mathrm { H G T } } } \operatorname* { m a x } _ { \varepsilon \in \mathcal { F } } \left\| \mathcal { H } x _ { \varepsilon } \right\| _ { 2 } } \\ & { \leq \left\| \Delta _ { A } \right\| _ { F _ { \varepsilon } ^ { \mathrm { H G T } } } \operatorname* { m a x } _ { i \in [ T ] } \left\| \mathcal { H } \tilde { A } ( t ) \eta \right\| _ { 2 } . } \end{array}
$$

where $( Y _ { \widetilde { S } ^ { c } } D _ { \widetilde { S } ^ { c } } ) _ { : , t }$ and $X _ { : , t }$ represent t-th columns of $Y _ { \widetilde { S } ^ { c } } D _ { \widetilde { S } ^ { c } }$ and X.

Let $t \in [ T ]$ be fixed and let $\mathcal { M } _ { t } = \left\{ H \widetilde { A } ( t ) : H \in \widetilde { C } _ { \mathcal { K } } \right\}$ . Then, we can see that

$$
\operatorname* { s u p } _ { H \in \widetilde { C } _ { \mathcal K } } \left. H \widetilde { A } ( t ) \eta \right. _ { 2 } = \operatorname* { s u p } _ { M \in \mathcal { M } _ { t } } \left. M \eta \right. _ { 2 } .
$$

Note that for any $M \in \mathcal { M } _ { t }$ (where $M = H \widetilde { A } ( t )$ for some $H \in \widetilde { C } _ { \kappa } )$

$$
\begin{array} { r } { \mathbb { E } \left( \left\| M \eta \right\| _ { 2 } ^ { 2 } \right) = \left\| M \right\| _ { F } ^ { 2 } = \left\| H \widetilde { A } ( t ) \right\| _ { F } ^ { 2 } \leq \left\| \widetilde { A } ( t ) \right\| _ { 2 } ^ { 2 } \left\| H \right\| _ { F } ^ { 2 } \leq J ^ { 2 } ( A ^ { * } ) . } \end{array}
$$

Also note that,

$$
d _ { 2 } ( M _ { t } ) = \operatorname* { s u p } _ { M \in \mathcal { M } _ { t } } \left\| M \right\| _ { 2 } = \operatorname* { s u p } _ { H \in \tilde { C } _ { \kappa } } \left\| H \widetilde { A } ( t ) \right\| _ { 2 } \leq \left\| \widetilde { A } ( t ) \right\| _ { 2 } \operatorname* { s u p } _ { H \in \tilde { C } _ { \kappa } } \left\| H \right\| _ { 2 } \leq J ( A ^ { * } )
$$

$$
d _ { F } ( \mathcal { M } _ { t } ) = \operatorname* { s u p } _ { M \in \mathcal { M } _ { t } } \left\| M \right\| _ { F } = \operatorname* { s u p } _ { H \in \widetilde { C } _ { \kappa } } \left\| H \widetilde { A } ( t ) \right\| _ { F } \leq \left\| \widetilde { A } ( t ) \right\| _ { 2 } \operatorname* { s u p } _ { H \in \widetilde { C } _ { \kappa } } \left\| H \right\| _ { F } \leq J ( A ^ { * } ) .
$$

Furthermore, we can also see that the mapping $( \widetilde { C } _ { \mathcal { K } } , \Vert \cdot \Vert _ { 2 } )  ( { M } _ { t } , \Vert \cdot \Vert _ { 2 } )$ is $J ( A ^ { * } )$ -Lipschitz. To see this, we can consider $H _ { 1 } , H _ { 2 } \in \widetilde { C } _ { \mathcal { K } }$ . Then,

$$
\left. H _ { 1 } \widetilde { A } ( t ) - H _ { 2 } \widetilde { A } ( t ) \right. _ { 2 } = \left. \left( H _ { 1 } - H _ { 2 } \right) \widetilde { A } ( t ) \right. _ { 2 } \le \left. \widetilde { A } ( t ) \right. _ { 2 } \left. H _ { 1 } - H _ { 2 } \right. _ { 2 } \le J ( A ^ { * } ) \left. H _ { 1 } - H _ { 2 } \right. _ { 2 } .
$$

Therefore, we can see that $\gamma _ { 2 } \left( \mathcal { M } _ { t } , \left. \cdot \right. _ { 2 } \right) \leq c J ( A ^ { * } ) \gamma _ { 2 } \left( \widetilde C \kappa , \left. \cdot \right. _ { 2 } \right)$ for a constant c.

We can now apply Theorem 2 (Theorem 3.1 of Krahmer et al. (2014)) to $\mathcal { M } .$

$$
\mathbb { P } \left( \operatorname* { s u p } _ { M \in \mathcal { M } } \left| \left. M \eta \right. _ { 2 } ^ { 2 } - \mathbb { E } \left( \left. M \eta \right. _ { 2 } ^ { 2 } \right) \right| \geq c _ { 1 } P + q \right) \leq 2 \exp \left( - c _ { 2 } \operatorname* { m i n } \left\{ \frac { q ^ { 2 } } { F ^ { 2 } } , \frac { q } { L } \right\} \right) .\tag{B.8}
$$

where

$$
\begin{array} { r l } & { P = \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) \left( \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) + d _ { F } ( \mathcal { M } ) \right) + d _ { F } ( \mathcal { M } ) d _ { 2 } ( \mathcal { M } ) , } \\ & { F = d _ { 2 } ( \mathcal { M } ) \left( \gamma _ { 2 } \left( \mathcal { M } , \left. \cdot \right. _ { 2 } \right) + d _ { F } ( \mathcal { M } ) \right) \mathrm { ~ a n d } , L = d _ { 2 } ^ { 2 } ( \mathcal { M } ) . } \end{array}
$$

Let us denote,

$$
\begin{array} { r l } & { \overline { { P } } _ { 2 } = c _ { 1 } J ^ { 2 } ( A ^ { * } ) \gamma _ { 2 } \left( \widetilde { C } _ { K } , \lVert \cdot \rVert _ { 2 } \right) \left( \gamma _ { 2 } \left( \widetilde { C } _ { K } , \lVert \cdot \rVert _ { 2 } \right) + 1 \right) + J ^ { 2 } ( A ^ { * } ) } \\ & { \overline { { F } } _ { 2 } = c _ { 1 } J ^ { 2 } ( A ^ { * } ) \left( \gamma _ { 2 } \left( \widetilde { C } _ { K } , \lVert \cdot \rVert _ { 2 } \right) + 1 \right) \mathrm { ~ a n d } , \overline { { L } } _ { 2 } = J ^ { 2 } ( A ^ { * } ) } \end{array}
$$

for suficiently large $c _ { 1 } > 1$ . Then from (B.8),

$$
\mathbb { P } \left( \operatorname* { s u p } _ { M \in \mathcal { M } _ { t } } \left\| M \eta \right\| _ { 2 } ^ { 2 } \leq J ^ { 2 } ( A ^ { * } ) + c _ { 2 } \overline { { P } } _ { 2 } + q \right) \geq 1 - 2 \exp \left( - c _ { 3 } \operatorname* { m i n } \left\{ \frac { q ^ { 2 } } { \overline { { F } } _ { 2 } ^ { 2 } } , \frac { q } { \overline { { L } } _ { 2 } } \right\} \right) .
$$

Let $q = \overline { { F } } _ { 2 } \sqrt { m }$ for $m \geq 1$ . Since $\overline { { F } } _ { 2 } \geq \overline { { L } } _ { 2 }$ , we have min $\begin{array} { r } { \left\{ \frac { q ^ { 2 } } { \overline { { F } } _ { 2 } ^ { 2 } } , \frac { q } { \overline { { L } } _ { 2 } } \right\} = \sqrt { m } . } \end{array}$ . Therefore,

$$
\begin{array} { r } { \mathbb { P } \left( \underset { M \in \mathcal { M } _ { t } } { \operatorname* { s u p } } \left. M \eta \right. _ { 2 } ^ { 2 } \leq J ( A ^ { * } ) + c _ { 2 } \overline { { P } } _ { 2 } + \overline { { F } } _ { 2 } \sqrt { m } \right) \geq 1 - 2 \exp \left( - c _ { 3 } \sqrt { m } \right) . } \end{array}
$$

Recall from the proof of Lemma 7 that

$$
\gamma _ { 2 } \left( \widetilde { C } _ { K } , \| \cdot \| _ { 2 } \right) \lesssim \gamma _ { 2 } \left( \widetilde { C } _ { K } , \| \cdot \| _ { F } \right) \asymp w \left( \widetilde { C } _ { K } \right) \lesssim \sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) } .
$$

Therefore letting $m = c \log ^ { 2 } \left( { \textstyle { \frac { 1 } { \delta } } } \right)$ ，

$$
\begin{array} { r l } & { c _ { 2 } \overline { { P } } _ { 2 } + \overline { { F } } _ { 2 } \sqrt { m } \leq c ^ { \prime } J ^ { 2 } ( A ^ { * } ) \Bigg ( k \log \left( \frac { n ^ { 2 } } { k } \right) + \sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) } + \sqrt { m } + \sqrt { m } k \log \left( \frac { n ^ { 2 } } { k } \right) + 1 \Bigg ) } \\ & { \qquad \leq \bar { c } J ^ { 2 } ( A ^ { * } ) \Bigg ( k \log \left( \frac { n ^ { 2 } } { k } \right) + \log \left( \frac { 1 } { \delta } \right) \sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) } \Bigg ) } \\ & { \qquad \leq \bar { c } J ^ { 2 } ( A ^ { * } ) k \log \left( \frac { n ^ { 2 } } { k } \right) \log \left( \frac { 1 } { \delta } \right) . } \end{array}
$$

Therefore with probability at least $1 - \delta ,$

$$
\operatorname* { s u p } _ { M \in \mathcal { M } _ { t } } \left\| M \eta \right\| _ { 2 } ^ { 2 } \leq \bar { c } J ^ { 2 } ( A ^ { * } ) k \log \left( \frac { n ^ { 2 } } { k } \right) \log \left( \frac { 1 } { \delta } \right)
$$

Note that this is for a fixed t. Taking union bound over $t \in [ T ]$ with an appropriate rescaling of $\delta ,$ we have that with probability at least $1 - \delta$

$$
\left\| \Delta _ { A } Y _ { \widetilde { \mathcal { S } } ^ { c } } D _ { \widetilde { \mathcal { S } } ^ { c } } \right\| _ { 2 , \infty } \leq c J ( A ^ { * } ) \sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) \log \left( \frac { T } { \delta } \right) } \left\| \Delta _ { A } \right\| _ { F } .
$$

Let $\begin{array} { r } { R = J ( A ^ { * } ) \sqrt { k \log \left( \frac { n ^ { 2 } } { k } \right) \log \left( \frac { T } { \delta } \right) } } \end{array}$ . Combining the bounds on $\left\| \Delta _ { A } Y _ { \widetilde { S } ^ { c } } D _ { \widetilde { S } ^ { c } } \right\| _ { 2 , \infty }$ and $\| \Delta _ { A } Y _ { \widetilde { S } } D _ { \widetilde { S } } \| _ { 2 , \infty }$ , we $\mathrm { g e t }$

$$
\begin{array} { r l } & { |  \Delta _ { A } Y D , \Delta _ { U } D  | \leq \| ( \Delta _ { U } D ) _ { \widetilde { S } } \| _ { 2 , 1 } 2 r \sqrt { k } \| \Delta _ { A } \| _ { F } + ( 3 \| ( \Delta _ { U } D ) _ { \widetilde { S } } \| _ { 2 , 1 } + \displaystyle \frac { \overline { { V } } } { \overline { { \lambda } } } \| \Delta _ { A } \| _ { F } ) c R \| \Delta _ { A } \| _ { F } } \\ & { \qquad \leq c _ { 1 } \displaystyle \frac { \overline { { V } } } { \overline { { \lambda } } } R \| \Delta _ { A } \| _ { F } ^ { 2 } + c _ { 2 } r R \sqrt { k } \| \Delta _ { A } \| _ { F } \| ( \Delta _ { U } D ) _ { \widetilde { S } } \| _ { 2 , 1 } } \\ & { \qquad \leq \widetilde { c } _ { 1 } \displaystyle \frac { \overline { { V } } } { \overline { { \lambda } } } R \| \Delta _ { A } \| _ { F } ^ { 2 } + \widetilde { c } _ { 2 } r R \sqrt { k } \| \widetilde { S } \| \| \Delta _ { A } \| _ { F } \| ( \Delta _ { U } D ) _ { \widetilde { S } } \| _ { F } . } \end{array}
$$

## C DETAILED OVERVIEW OF RELATED WORK

## C.1 Robust regression under adversarial contamination

Consider the supervised learning problem where there exists a distribution D over $\mathcal { X } \times \mathcal { V }$ where the given $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m } \stackrel { i . i . d } { \sim } \mathcal { D }$ , the goal is to learn the relationship between Y and X. Specifically, let $y _ { i } = x _ { i } ^ { \top } \beta ^ { * } + \varepsilon _ { i } ,$ where $\varepsilon _ { i }$ is the noise and $x _ { i } \in \mathbb { R } ^ { d }$ . Given $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m }$ , estimating $\beta ^ { * }$ is a fundamental problem in machine learning and typically can be solved eficiently using methods such as ordinary least squares when the observations follow from $\mathcal { D }$ . However, the problem becomes more challenging when a subset of $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m }$ is contaminated by an adversary. Estimating $\beta ^ { * }$ in when a subset of $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m }$ is contaminated is known as robust regression under (adversarial) contamination. Robust regression under adversarial contamination in the i.i.d. setting has been studied under several formal contamination models.

1. Inlier-Outlier $( { \mathcal { T } } \cup { \mathcal { S } } )$ model: the sample indices are partitioned as $\mathcal { T } \cup \mathcal { S }$ with $| S | \le \epsilon m$ . The adversary selects the outlier set (of indices) S oblivious to the data. Once $s$ is selected, the adversary can potentially observe the entire dataset (including the data in $\mathcal { T } )$ and decide on the corrupted values for the entries in ${ \mathcal { S } } .$

2. Huber’s contamination model: each observation is drawn independently from a mixture distribution, hence with probability $1 - \epsilon$ from ${ \mathcal { D } } ,$ and with probability ϵ from an arbitrary distribution.

3. Strong contamination model: an adversary observes the entire dataset and then selects up to an ϵ- fraction of samples to corrupt arbitrarily. The adversary may adaptively choose corrupted samples after observing the data thus afecting the distributional properties of the overall dataset.

We cover a representative sample of key results for each contamination model, though this list is not exhaustive.

## Inlier-Outlier Model $( { \mathcal { T } } \cup { \mathcal { S } } )$

• Merad and Ga¨ıfas (2023a) study robust supervised learning with coordinate gradient descent under Lipschitz smooth losses and finite moment assumptions on covariates and labels. They compute robust gradient estimates using Median-of-Means (MOM) and Trimmed Mean estimators, and provide bounds on the generalization error.

• Merad and Ga¨ıfas (2023b) consider high-dimensional linear models with an ϵ fraction of corrupted samples, under sparsity assumptions on $\beta ^ { * }$ . Under finite $\ell _ { 4 } { \mathrm { - n o r m } }$ assumptions on covariates and finite $\ell _ { 2 } \cdot$ -norm assumptions on labels, they show that robust gradient estimators enable consistent parameter recovery. For smooth losses, they apply multistage mirror descent; for non-smooth Lipschitz losses satisfying pseudo-linear minorization, they use multistage dual averaging. They further establish that the Trimmed Mean estimator is a robust gradient estimator for sparse regression under ϵ-contamination.

• Dalalyan and Thompson (2019) analyze sparse regression with adversarially corrupted labels and show that under restricted eigenvalue and incoherence conditions on the covariance matrix of the clean data, the ℓ -penalized Huber M-estimator achieves $\tilde { O } ( \sqrt { \frac { k } { m } } + \epsilon )$ when m points are observed and $\beta ^ { * }$ is k-sparse.

## Huber’s Contamination Model.

• Diakonikolas et al. (2019) show that for Huber’s contamination model with ϵ-contamination, no algorithm with statistical query access to the ϵ-contaminated distribution, can approximate $\beta ^ { * }$ to a $\sqrt { \epsilon }$ error without using needing samples of order ${ \cal \tilde { O } } ( d ^ { 2 } )$

• Diakonikolas et al. (2023) explore linear regression when $\mathcal { D } = \mathcal { N } ( 0 , I _ { d } )$ and give an algorithm with sample complexity $\tilde { O } ( d / \epsilon ^ { 2 } )$ with $\ell _ { 2 }$ error of the order $O ( \epsilon )$ in almost linear time by reducing the problem to a robust mean estimation problem.

• Zhang (2023) propose M-estimators for robust regression under Huber’s contamination model. Assuming Lipschitz conditions and locally quadratic nature on the loss function, and specific assumptions on the covariates and the noise, they show that M-estimators can achieve $\tilde { O } ( \sqrt { d / m + \epsilon } )$

• Diakonikolas et al. (2024) extend the ideas from Diakonikolas et al. (2023) to the setting when $\beta ^ { * }$ is k-sparse and $\left\| \beta ^ { * } \right\| _ { 2 }$ is bounded and show that with $\tilde { O } \left( \frac { k ^ { 2 } \log d } { \epsilon ^ { 2 } } \right)$ many samples you can estimate $\beta ^ { * }$ up to $O ( \epsilon )$ error in polynomial time.

## Strong Contamination Model.

• Liu et al. (2019) introduce the Robust Descent Condition (RDC) and shows that gradient estimators satisfying RDC yield statistical guarantees for recovery of $\beta ^ { * }$ when combined with Iterative Hard Thresholding. For linear regression with a k-sparse parameter, with ${ \tilde { O } } ( 1 / { \sqrt { k } } \log d ) .$ -fraction of the data contaminated, they show that a version of trimmed gradient estimator recover $\beta ^ { * }$ upto $\tilde { O } ( \sqrt { k } ( \epsilon \log ( m d ) + \sqrt { \log d / m }$ (in $\ell _ { 2 }$ error) when $m = \Omega ( k \log d )$

• Liu et al. (2020) propose a Robust Sparse Gradient Estimator (RGSE), computed using a convex relaxation of robust PCA followed by hard thresholding and filtering. For sparse regression where the covariance matrix is identity they show how to recover $\beta ^ { * }$ with minimax optimal rate of ${ \tilde { O } } ( \epsilon )$ . When the covariance matrix is sparse but unknown, they show how to achieve ${ \tilde { O } } ( \epsilon )$

• Sasai (2022) propose a weighted ℓ -penalized Huber M-estimator under bounded fourth-moment covariates and i.i.d. noise, where weights are computed via convex relaxation of robust PCA followed by truncation. The extension in Sasai and Fujisawa (2025) consider sub-Gaussian covariates satisfying restricted eigenvalue conditions.

• Pensia et al. (2025) consider arbitrary contamination of at most ϵn samples under mean-zero, identitycovariance, (4, 2)-hypercontractive covariates and independent mean-zero noise, and analyzes Huber regression, Least Trimmed Squares (LTS), and Least Absolute Deviation (LAD) estimators. Under stability conditions, they show that Huber regression achieves $\tilde { O } ( \sqrt { d / m } + \sqrt { \epsilon } )$ . They also establish $\ell _ { 2 }$ error guarantees for LTS and LAD under subset strong convexity and subset strong smoothness conditions.

• Bhatia et al. (2015) establish recovery guarantees for Least Trimmed Squares in high-dimensional settings under adversarial corruption. They provide a thresholding based gradient descent algorithm that converges to an error of $\begin{array} { r } { \le \epsilon \mathrm { i n } \tilde { O } \left( \log \left( \frac { 1 } { \epsilon \sqrt { m } } \right) \right) } \end{array}$ iterations when the corruption is bounded and the normalized covariates satisfy subset strong convexity and subset strong smoothness <sup>4</sup>.

• The work of Karmalkar and Price (2019) analyzes Least Absolute Deviation regression under adversarial contamination. They establish $\ell _ { 2 }$ error guarantees when $\beta ^ { * }$ is k-sparse and the covariates come from a Gaussian distribution. They also establish error guarantees for the setting when $\beta ^ { * }$ is dense as well.

• Chen et al. (2013) consider the setting where both covariates and labels may be corrupted, including distributed corruption across entries of individual samples. They propose algorithms based on thresholding, the Dantzig selector, Lasso regression, and thresholding regression, and show that the three methods modified with a trimmed variant of inner product, recover $\beta ^ { * }$ to small error.

• Sasai and Fujisawa (2020) study robust sparse regression using an $\ell _ { 1 }$ -penalized formulation under adversarial contamination and uses specific properties of the Huber function to establish recovery guarantees similar to that of Dalalyan and Thompson (2019). They show that $\mathrm { w e i g h t e d } { \cdot } \ell _ { 2 }$ error (weighted with the covariance matrix) is of the order of $\tilde { O } ( \sqrt { k / m } + \epsilon )$ when the covariance matrix satisfies restricted eignvalue conditions.

• Cherapanamjeri et al. (2020) develop robust estimators using semidefinite programming to compute gradient estimates under hypercontractive or sub-Gaussian covariates. Under sub-Gaussian assumptions, they achieve sample complexity and error up to logarithmic factors of the mini-max optimal and optimal error and sample complexity optimal up to logarithmic factors, under hypercontractivity assumptions.

• Maurya et al. (2026) propose a bi-convex optimization formulation, proves the existence of an integral KKT point that is locally optimal, and provides an algorithm with provable error guarantees. For k-sparse regression problem, they establish necessary conditions for exact support recovery, and $\tilde { O } ( \sqrt { k } \epsilon ) ~ \ell _ { \mathrm { 2 } } \mathrm { - e r r o r }$

• Bakshi and Prasad (2021) study linear regression under $( c , k )$ -hypercontractive (for some parameters $c , k )$ distributions with an ϵ-fraction of adversarial corruption. They encode the problem as a polynomial system and apply a sum-of-squares relaxation to obtain computationally eficient estimators that achieve informationtheoretically optimal convergence rates establishing tight upper and lower bounds.

• Minsker et al. (2024) consider high-dimensional linear regression in which an ϵ-fraction of the measurements are adversarially corrupted, with the adversary having full knowledge of the data and the underlying distribution. The covariates are assumed to be sub-Gaussian, while the additive noise is allowed to be dense and heavy-tailed. They establish minimax lower bounds that depend on both the contamination level and the tail behavior of the noise. They propose algorithms based on Square-Root Slope estimator that achieves error that is upto logarithmic factors of the minimax optimal rate for the $\epsilon .$

## C.2 Robust learning of linear dynamical systems

While robust regression under adversarial contamination has been extensively studied in the i.i.d. setting, comparatively fewer works address this problem for linear dynamical systems and higher-order VAR models. Existing works primarily focus on the following settings.

## C.2.1 Heavy-tailed covariates or process noise

Kanakeri and Mitra (2025) study the problem of learning a linear dynamical system from N independent trajectories in the presence of heavy-tailed noise. In their work, the data is generated through a linear dynamical system $x _ { t + 1 } = A ^ { * } x _ { t } + \eta _ { t + 1 }$ with $\eta _ { t + 1 }$ coming from a zero-mean i.i.d. heavy-tailed distribution with bounded second and fourth moments. The authors partition trajectories into buckets, estimate the system matrix locally within each bucket, and then aggregate the estimates by taking their median under the Frobenius norm. They show robustness to heavy-tailed noise with non-asymptotic $O ( 1 / \sqrt { N } )$ estimation error rate w.r.t the operator norm. This multi-trajectory setting does not require any stability assumption on $A ^ { * }$

Wong et al. (2020) explores β-mixing with heavy-tailed covariates. They consider a model where the goal is to predict $x _ { t }$ using $( x _ { t - 1 } , x _ { t - 2 } , \ldots , x _ { t - d } )$ , using a linear predictor under the condition that the dynamic process for $x _ { t }$ is stationary. The paper assumes the covariates $x _ { t }$ follow a sub-Weibull distribution allowing for heaviertails. They propose a least-squares estimator with $\ell _ { 1 }$ -penalty and show that when the underlying true predictor matrix is sparse and $( x _ { t } ) _ { t > 1 }$ are centered, the estimator can recover the underlying predictor matrix up to an error dependent on the weight given to the penalty and restricted eigenvalue conditions.

Halder and Michailidis (2022) examines the problem of learning a $\mathrm { V A R ( d ) }$ , i.e. $\begin{array} { r } { x _ { t } = \sum _ { i = 1 } ^ { d } A _ { i } x _ { t - i } + \eta _ { t } } \end{array}$ , where $\eta _ { t }$ can be heavy-tailed. They assume $\eta _ { t }$ has conditionally symmetric density (condition on the history of the observations) and $x _ { t }$ satisfies either β-mixing or Markov chain conditions (as well as $x _ { t }$ is stationary) and show that a variant of Lasso allows for consistency under mixing as well as Markov chain conditions.

Qiu et al. (2015) consider a stationary VAR model $x _ { t + 1 } = A ^ { * } x _ { t } + \eta _ { t + 1 }$ where covariates and process noise follow a joint elliptical distribution, to capture heavy-tailed behavior. They propose a quantile-based robust variant of the Dantzig estimator and show that the recovery rates for $A ^ { * }$ matches the parametric rates under the entry-wise max norm.

Dijk and Cho (2025) study high-dimensional heavy-tailed time series using a factor-adjusted VAR model. They consider a VAR(d) model, $\begin{array} { r } { x _ { t } = \sum _ { i = 1 } ^ { d } A _ { i } x _ { t - i } + \eta _ { t } } \end{array}$ where $\eta _ { t }$ is zero mean with uncorrelated (potentially heavy tailed) entries. The learner observes a $y _ { t } = x _ { t } + \nu _ { t }$ where $\nu _ { t }$ is a factor-adjusted component. The authors apply element-wise truncation and a multi-stage parameter estimation procedure using lasso, under the assumption of finite $( 2 + 2 \epsilon ) \AA ^ { \prime } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { + } \it { 2 \epsilon } \mathrm { { - } \it { 6 \Omega } \mathrm { { \Omega } } } } } } } } } } } } $ moments for some $\epsilon \in ( 0 , 1 )$ .

Lu et al. (2025) analyze dynamic spatial autoregressive models under heavy-tailed distributions, proposing a constrained Yule-Walker estimation framework with statistical recovery guarantees. They consider a spacial autoregressive model where $x _ { t } = A _ { 0 } W x _ { t } + A _ { 1 } x _ { t - 1 } + A _ { 1 } W x _ { t - 1 } + \eta _ { t }$ , with $W$ a weighting matrix capturing the relationships in the spatial domain. The work proposes a robust estimator for covariance estimation under this heavy-tailed setting.

Wang and Tsay (2023) explores system recovery from a VAR(d) model, $\begin{array} { r } { x _ { t } = \sum _ { i = 1 } ^ { d } A _ { i } x _ { t - i } + \eta _ { t } } \end{array}$ where $\eta _ { t }$ is a serially uncorrelated white noise innovation where the white noise can be heavy-tailed. They propose a constrained Yule-Walker estimator that is robust against heavy-tailed noise with bounded 2 + 2ϵ-th moments for any $\epsilon > 0$ . The paper establishes minimax optimal rates by establishing minimax lower bounds matching their recovery guarantees.

Wang et al. (2024) investigate least absolute deviation regression for high-dimensional heavy-tailed time series with finite α-moments for $\alpha \in ( 1 , 2 ]$ . They consider VAR(d) of the form, $\begin{array} { r } { x _ { t } = \sum _ { i = 1 } ^ { d } A _ { i } x _ { t - i } + \eta _ { t } } \end{array}$ , where $( \eta _ { t } ) _ { t \geq 1 }$ is i.i.d. and has zero mean and finite α-moments for $\alpha \in ( 1 , 2 ]$ . The paper proposes a truncated estimator for which they show that the population risk for the estimator converges to the optimal population risk, as $T$ increases.

## C.3 Sparse process noise

The works of Kim and Lavaei (2026b) and Kim and Lavaei (2026a) consider a linearly parameterized (potentially) non-linear system given by $x _ { t + 1 } = A ^ { * } \phi ( x _ { t } ) + \eta _ { t }$ where $\phi : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ is a non-linear mapping and $\eta _ { t }$ is analogous to the process noise in (1.1). They consider scenarios where $\eta _ { t }$ is i.i.d. Gaussian or sub-Gaussian as well as where $\eta _ { t }$ comes from a sparse adversarial process (sub-Gaussian but not necessarily zero mean or independent). Kim and Lavaei (2026b) propose a two-stage algorithm for recovery of $A ^ { * }$ . In the first stage, they solve a Least Absolute Deviation problem independently for each coordinate of the autoregressive vectors to obtain candidate rows of the system matrix A. In the second stage, they apply a filtering step based on the initial estimates and then perform Least Squares on the filtered data, recovering the rows of $A$ up to a small error. The follow-up work by Kim and Lavaei (2026a) extends this analysis and shows that under the same assumptions, Huber estimator recovers the rows of A<sup>∗</sup> up-to a small error. In both works, they show that for symmetric process noise, they can recover the rows of $A ^ { * }$ with a rate of $O ( 1 / \sqrt { T } )$ when the system is linear.

Kim (2026) explores robust learning of a VAR(d) model, i.e., $\begin{array} { r } { x _ { t } \ = \ \sum _ { i = 1 } ^ { d } A _ { i } x _ { t - i } + \eta _ { t } + v _ { t } } \end{array}$ where $\eta _ { t }$ is the process noise and $v _ { t }$ is an innovative outlier. They show that least squares with ℓ<sub>1</sub>-penalty on each $v _ { t } ,$ can be used to recover the matrices $( A _ { i } ) _ { i = 1 } ^ { d }$ (by using an equivalence with Huber estimator and applying iteratively reweighted least squares) under the assumption that $\eta _ { t } + v _ { t }$ is sampled from an i.i.d. distribution with bounded variance (and additional structural assumptions). Their guarantees are in the form of asymptotic normality of $\left[ A _ { 1 } \quad A _ { 2 } \quad \ldots \quad A _ { d } \right]$

Farahmand et al. (2011) explores the related problem of doubly robust smoothing of dynamical systems where the goal is to learn the dynamical system and the corruption, given system dynamics and under distributional assumptions. In this setting, a learner observes a $y _ { t }$ generated through $x _ { t } = A _ { t } x _ { t - 1 } + \eta ( x ) _ { t } + o ( x ) _ { \mathrm { ~ } }$ <sub>t</sub> and $y _ { t } = B _ { t } x _ { t } + \eta ( y ) _ { t } + o ( y ) _ { t }$ where $o ( x ) _ { t } , o ( y ) _ { t }$ are outliers and $\eta ( x ) _ { t } , \eta ( y ) _ { t }$ are noise coming from zero mean distributions with known covariance. The learner is also assumed to know $( A _ { t } , B _ { t } ) _ { t = 1 } ^ { T }$ and the noise covariance. The goal is to learn $( x _ { t } , o ( x ) _ { t } , o ( y ) _ { t } ) _ { t = 1 } ^ { T }$ assuming that the stacked vector of outliers is sparse. They explore a least-squares estimator with ℓ -penalty and introduces a provably convergent algorithm via coordinate descent.

## C.4 Learning from corrupted states

In the work of Kanakeri and Mitra (2025) discussed earlier, the authors further prove (for the strong contamination adversarial model) that if an ϵ-fraction of the N trajectories are corrupted, the estimator still recovers the system matrix up to a small error (w.r.t operator norm). Their theoretical result for this adversarial setting, and even their algorithm does not apply to the single trajectory setting considered in our work.

The work of Wong (2017) considers a $\mathrm { V A R ( d ) }$ model, $\begin{array} { r } { x _ { t } \ = \ \sum _ { i = 1 } ^ { d } A _ { i } x _ { t - i } + \eta _ { t } } \end{array}$ , where $\eta _ { t } \mathrm { ^ { { \sc ~ s } } }$ are independent centered Gaussians. Under the assumption that $\rho ( A _ { i } ) < 1$ for all $i \in [ d ]$ , and assuming that the learner observes $\tilde { x } _ { t } = x _ { t } + w _ { t }$ where $w _ { t }$ comes from an independent zero-mean Gaussian distribution with known covariance, they establish non-asymptotic recovery guarantees for the lasso estimator with $\ell _ { 1 }$ penalty.

Bhatia et al. (2016) consider a 1-dimensional auto-regressive process with degree $d ,$ i.e. $\begin{array} { r } { x _ { t } = \sum _ { i = 1 } ^ { d } A _ { i } x _ { t - i } + \eta _ { t } } \end{array}$ where $( x _ { t - i } ) _ { i = 0 } ^ { d } , \eta _ { t } \in \mathbb { R }$ . They consider an outlier model related to our setting where the learner observes $\tilde { x } _ { t } =$ $x _ { t } + \theta _ { t }$ where $\theta = \left\lceil \theta _ { 1 } \quad \theta _ { 2 } \quad \ldots \quad \theta _ { T } \right\rceil$ is k-sparse. $\mathrm { B y }$ transforming the $\mathrm { A R } ( \mathrm { d } )$ into a form of $\boldsymbol { y } = \boldsymbol { X } ^ { \top } \boldsymbol { w } + \boldsymbol { \eta }$ (where $y = \left\lceil x _ { 1 } \quad x _ { 2 } \quad \ldots \quad x _ { n } \right\rceil$ and $X , w , \eta$ defined appropriately to recover the $\mathrm { A R ( d ) } )$ , under subset strong convexity and subset strong smoothness of the corrupted covariate matrix, and assuming the outliers are independent of noise (which is not stated explicitly but seems to be needed in the proof, e.g., within Appendix B.6) they propose a block-sparse hard thresholding algorithm which is robust against additive outliers and provably consistent.

A line of work has considered the setting where $\tilde { x } _ { t }$ is formed by independently sub-sampling the entries of $x _ { t }$ at each time point t (and setting other entries to 0) for $( x _ { t } ) _ { t = 1 } ^ { T }$ generated from (1.1). When the system is stationary the works of Rao et al. (2016, 2017b) employ covariance estimation to recover $A ^ { * }$ under diferent structural assumptions such as sparsity, and low-rankness. The works of Jalali and Willett (2018); Wong (2017) explore lasso-type estimators with an additional term to correct for the bias (which makes the problem potentially nonconvex), and provably estimate $A ^ { * }$ under sparsity and stability assumptions. When $A ^ { * }$ is sparse and strictly stable, Dalle and Castro (2025) establish minimax optimal recovery guarantees with respect to the entry-wise max norm, for the Dantzig selector. Rao et al. (2017a) consider the setting where $\tilde { x } _ { t }$ is formed by taking compressive linear measurements of $x _ { t }$ (at each t), and study the limits of covariance-based estimation.

## D ROBUST ESTIMATORS BASED ON LEAST-TRIMMED SQUARES

In this section, we provide a detailed overview of the estimators introduced in Section 2.

## D.1 Sphere relaxation

Consider the optimization problem (P2) and the two constraints $z \in \left\{ 0 , 1 \right\} ^ { T }$ and $\begin{array} { r } { \mathbf { 1 } _ { T } ^ { \top } z \ : = \ : T - \ : s . } \end{array}$ . A simple calculation shows that when $z \in \left\{ 0 , 1 \right\} ^ { T } , \mathbf { 1 } _ { T } ^ { \top } z = T - s$ is equivalent to $\left\| z \right\| _ { 2 } ^ { 2 } = z ^ { \top } z = T - s$ . Therefore, we can see that optimization problem (P2) is equivalent to,

$$
\operatorname* { m i n } _ { A \in A , z \in \left\{ 0 , 1 \right\} ^ { T } } z ^ { \top } Q ( A ) z \quad \mathrm { s . t . } \quad \left\| z \right\| _ { 2 } ^ { 2 } = T - s , \quad \mathbf { 1 } _ { T } ^ { \top } z = T - s
$$

where the vector z is a binary vector on the surface of a sphere of radius $\sqrt { T - s }$ centered at the origin. Given this, removing the constraint $z \in \left\{ 0 , 1 \right\} ^ { T + 1 }$ , i.e. letting $z \in \mathbb { R } ^ { T + 1 }$ , results in,

$$
\operatorname* { m i n } _ { A \in A , z \in \mathbb { R } ^ { T } } z ^ { \top } Q ( A ) z \quad { \mathrm { s . t . } } \quad \left\| z \right\| _ { 2 } ^ { 2 } = T - s , \quad \mathbf { 1 } _ { T } ^ { \top } z = T - s .
$$

Given $A ,$ estimating z. Now consider the scenario where we have been given A and we are tasked with finding z. Note that when A is fixed, i.e. when $Q ( A )$ is fixed, the optimization problem (P3) translates to solving a trust-region problem which has polynomial time algorithms (Mor´e and Sorensen, 1983). However, since $z \in { \bar { \mathbb { R } } } ^ { T }$ is not necessarily a binary vector, we project it on to $\{ 0 , 1 \} ^ { T }$ via the following rounding method.

1. Extract the subset of $\widehat { S } \subset \{ 1 , \ldots , T \}$ corresponding to the $T - s$ entries of $\{ z _ { 1 } , \dots , z _ { T } \}$ with the largest magnitudes (ignoring the sign).

2. For $t \in { \widehat { S } }$ set $z _ { t } = 1$ and for $t \not \in { \widehat { S } }$ set $z _ { t } = 0$

Given $z ,$ estimating A. Given a binary vector z (if z is not binary we can use the rounding discussed above), we use (P2) as stated in step 6 of Algorithm 1 to solve a A-constrained weighted least squares problem to derive an estimate A.

## D.2 Semi-definite program (SDP) relaxation

Since $z ^ { \top } Q ( A ) z = \left. Q ( A ) , z z ^ { \top } \right.$ we can replace $z z ^ { \top }$ with the p.s.d matrix Y. Note that when $Y = z z ^ { \top } , Y _ { t t } = z _ { t } ^ { 2 }$ and since $z _ { t } \in \{ 0 , \dot { 1 } \} , Y _ { t t } = z _ { t }$ which gives us dia $\mathbf { g } ( Y ) = z$ . We apply the dual of Shor’s relaxation (Boyd and Vandenberghe, 1997) to derive the following.

$$
\operatorname* { m i n } _ { A \in { \mathcal { A } } , Y \succeq 0 , z \in [ 0 , 1 ] ^ { T } } \langle Q ( A ) , Y \rangle
$$

$$
{ \mathrm { s . t . ~ } } \operatorname { d i a g } ( Y ) = z , \ \mathbf { 1 } _ { T } ^ { \top } z = T - s { \mathrm { ~ a n d ~ } } \left[ { 1 2 \mathrm { ~ } } \begin{array} { l } { z ^ { \top } } \\ { Y } \end{array} \right] \succeq 0 .
$$

Note that for a fixed A, (P4) is a SDP and can be solved eficiently to obtain $Y , z .$

Given A, estimating z. Let $Y , z$ be the solution of (P4) when A is fixed. Since $z \in [ 0 , 1 ] ^ { T }$ is not necessarily a binary vector, we can project it on to $\{ 0 , 1 \} ^ { T }$ as described above.

Given z, estimating A. Given a binary vector $z \ ( \mathrm { i f } \ z$ is not binary we can use the rounding discussed above), we use (P2) as stated in step 6 of Algorithm 1 to solve a A-constrained weighted least squares problem to derive an estimate A.

## D.3 Bi-convex relaxation

Note that while (P1) captures the problem of filtering the contaminants and learning a linear dynamical process over the uncontaminated observations, the formulation involves a product of $z _ { t } z _ { t + 1 }$ (for $t \in \{ 0 , 1 , \ldots , T - 1 \} )$ in the objective function. In this section, we will introduce a set of new variables to remove the multiplicative dependencies of $z _ { t } , z _ { t + 1 }$

Let $\boldsymbol b _ { t } ~ = ~ \boldsymbol z _ { t } \boldsymbol z _ { t + 1 }$ for each $t \in \{ 1 , \ldots , T - 1 \}$ . Then note that the objective of (P1) can be re-written as $\begin{array} { r } { \sum _ { t = 1 } ^ { T - 1 } b _ { t } \big \| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 } } \end{array}$ by introducing an additional constraint $b _ { t } = z _ { t } z _ { t + 1 }$ for $t \in \{ 1 , \ldots , T - 1 \}$ . However, while this removes the multiplicative dependencies between $z _ { t }$ and $z _ { t + 1 }$ in the objective, we are still left with the same dependencies in the constraints. We can remove these dependencies using the following key observation.

Remark 12. When $z _ { t } \in \{ 0 , 1 \} f o r t \in \{ 1 , . . . , T \}$ and $b _ { t } \in \{ 0 , 1 \} f o r t \in \{ 1 , . . . , T - 1 \}$ , the constraint $b _ { t } = z _ { t } z _ { t + 1 }$ is equivalent to the set of constraints $b _ { t } \leq z _ { t } , b _ { t } \leq z _ { t + 1 }$ and $b _ { t } \geq z _ { t } + z _ { t + 1 } - 1$ . To see this we can consider the possible values of $z _ { t }$ and $z _ { t + 1 }$

• Case 1: $( z _ { t } = z _ { t + 1 } = 1 )$ The constraint $b _ { t } \geq z _ { t } + z _ { t + 1 } - 1 = 1 + 1 - 1 = 1 \ i m p l i e s \ b _ { t } = 1 = z _ { t } \cdot z _ { t + 1 }$

• Case 2: $( z _ { t } = 0 \ \mathbf { o r } \ z _ { t + 1 } = 0 )$ The two constraints $b _ { t } \ \leq \ z _ { t }$ and $b _ { t } \ \leq \ z _ { t + 1 }$ imply $b _ { t } \leq \operatorname* { m i n } \left\{ z _ { t } , z _ { t + 1 } \right\} = 0$ which implies $b _ { t } = 0 = z _ { t } z _ { t + 1 }$

Therefore, we can use this to derive the optimization problem,

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { A \in \mathcal { A } } \sum _ { t = 1 } ^ { T - 1 } b _ { t } \big \| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 } } } \\ & { } & { \quad \displaystyle _ { z \in \{ 0 , 1 \} ^ { T - 1 } } \sum _ { t = 1 } ^ { T - 1 } b _ { t } \big \| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 } } \\ & { } & { \quad \mathrm { ~ s . t . ~ } \displaystyle \sum _ { t = 1 } ^ { T } z _ { t } = T - s } \\ & { } & { \quad \displaystyle _ { z _ { t } + z _ { t + 1 } - 1 \leq b _ { t } \leq \operatorname* { m i n } \big \{ z _ { t } , z _ { t + 1 } \big \} ~ \forall t \in [ T - 1 ] . } } \end{array}
$$

Note that the objective is biconvex. Relaxing the binary constraints on $z _ { t }$ and $b _ { t }$ to the interval [0, 1] results in the optimization problem

$$
\operatorname* { m i n } _ { A \in \mathcal { A } \atop z \in [ 0 , 1 ] ^ { T } } \sum _ { t = 1 } ^ { T - 1 } b _ { t } { \left\| \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } \right\| } _ { 2 } ^ { 2 }
$$

$$
{ \mathrm { s . t . } } \sum _ { t = 1 } ^ { T } z _ { t } = T - s
$$

$$
z _ { t } + z _ { t + 1 } - 1 \leq b _ { t } \leq \operatorname* { m i n } \left\{ z _ { t } , z _ { t + 1 } \right\} \forall t \in [ T - 1 ]
$$

which has convex constraints (since A is convex by assumption).

Given A, estimating $( z _ { t } ) _ { t = 1 } ^ { T }$ and $( b _ { t } ) _ { t = 1 } ^ { T - 1 }$ . When A is fixed note that the optimization problem (P5) translates to a convex program with a linear objective. Solving this gives us $( z _ { t } ) _ { t = 1 } ^ { T }$ and $\left( b _ { t } \right) _ { t = 1 } ^ { T - 1 }$ such that $z _ { t } \in [ 0 , 1 ]$ and $b _ { t } \in [ 0 , 1 ]$ . In order to project $z _ { t } , b _ { t }$ on to {0, 1}, we consider the following rounding method.

1. Extract the subset $\widehat { S } \subset \{ 1 , \ldots , T \}$ corresponding to the $T - s$ entries of $\{ z _ { 1 } , \dots , z _ { T } \}$ with the largest magnitude.

2. Set the values at $t \in S$ to 1 and set all other entries $( t \not \in S )$ to 0.

3. For all $t \in [ T - 1 ]$ , set $b _ { t } = z _ { t } z _ { t + 1 }$ where $z _ { t } , z _ { t + 1 }$ come from step 2.

Given z, estimating A. Given a binary vector $z \ ( \mathrm { i f } \ z$ is not binary we can use the rounding discussed above), we use (P2) as stated in step 6 of the Algorithm 1 to solve a A-constrained weighted least squares problem to derive an estimate A.

## D.4 Convexifying $Q ( A )$ to estimate z

Assume A is fixed and the goal is to estimate z. Note that $Q ( A )$ is a 1-banded matrix and is not necessarily positive semi-definite (p.s.d). However, since $\mathbf { 1 } _ { T } ^ { \top } z = T - s$ , when $z \in \mathbb { R } ^ { T }$ , the objective of the optimization problem (P2) can be written as $z ^ { \top } \left( Q ( A ) + \alpha ( { \bar { A } } ) \cdot I _ { T } \right) z$ (where $\alpha ( A )$ is some parameter). To see this, we can simply calculate the addition to the objective $z ^ { \top } \left( \alpha ( A ) \cdot I _ { T } \right) z = \alpha \cdot z ^ { \top } z = \alpha ( A ) \cdot ( T - s )$ which is a fixed amount when $\alpha ( A )$ is fixed. Therefore, this shows that by choosing $\alpha ( A )$ to be suitably large, we can ensure that ${ \tilde { Q } } ( A ) = Q ( A ) + \alpha ( A ) I _ { T }$ is a p.s.d matrix. Let $\alpha ( A ) = \operatorname* { m i n } \left\{ 0 , - \lambda _ { m i n } ( Q ( A ) ) \right\}$ where $\lambda _ { m i n } ( Q ( A ) )$ is the smallest eigenvalue of $Q ( A )$ . This gives us the optimization formulation

$$
\operatorname* { m i n } _ { z \in \{ 0 , 1 \} ^ { T } } z ^ { \top } \tilde { Q } ( A ) z \quad \mathrm { s . t . } \quad \mathbf { 1 } _ { T } ^ { \top } z = T - s .
$$

By relaxing $\{ 0 , 1 \} ^ { T }$ to $[ 0 , 1 ] ^ { T }$ we obtain

$$
\operatorname* { m i n } _ { z \in [ 0 , 1 ] ^ { T } } { z ^ { \top } \tilde { Q } ( A ) z } \quad \mathrm { s . t . } \quad \mathbf { 1 } _ { T } ^ { \top } z = T - s
$$

which is a convex program w.r.t z (for a fixed A).

Given A, estimating z. When A is fixed note that $\tilde { Q } ( A )$ is a fixed p.s.d matrix. Therefore, we can use standard solvers (e.g., interior point methods) to find an estimate z. However, since $z \in [ 0 , 1 ] ^ { T }$ , this does not guarantee a binary vector. In order to project the vector z on to $\{ 0 , 1 \} ^ { T }$ , we consider the following rounding method.

1. Extract the subset $\widehat { S } \subset \{ 1 , \ldots , T \}$ corresponding to the $T - s$ entries of $\{ z _ { 1 } , \dots , z _ { T } \}$ with the largest magnitude.

2. For $t \in S$ set $z _ { t } = 1$ and for $t \not \in S$ set $z _ { t } = 0$

Given $z ,$ estimating A. Given a binary vector z (if z is not binary we can use the rounding discussed in the previous chapter to round it to get a binary vector), we use (P2) as stated in step 6 of the Algorithm 1 to solve an A-constrained weighted least-squares problem to derive an estimate A.

## E ADDITIONAL EXPERIMENTS

## E.1 Overview of the implementation

The following is a detailed overview of the techniques used to implement each estimator in the experiments.   
Note that in some cases use approximation methods due to the computational cost of exact solvers.

Sphere relaxation. Note that the sphere relaxation is a variant of the Trust Region Subproblem. To see this, let $\begin{array} { r } { z = \frac { T - s } { T } \mathbf { 1 } _ { T } + w } \end{array}$ where $\left. w , \mathbf { 1 } _ { T } \right. = 0$ . This gives

$$
\begin{array} { l } { \displaystyle \mathbf { 1 } _ { T } ^ { \top } z = T - s \iff \mathbf { 1 } _ { T } ^ { \top } w = 0 } \\ { \displaystyle \left\| z \right\| _ { 2 } ^ { 2 } = T - s \iff \frac { ( T - s ) ^ { 2 } } { T } + \left\| w \right\| _ { 2 } ^ { 2 } = T - s \iff \left\| w \right\| _ { 2 } ^ { 2 } = \frac { s ( T - s ) } { T } } \\ { z ^ { \top } Q ( A ) z = w ^ { \top } Q ( A ) w + 2 \left( \frac { T - s } { T } \right) \mathbf { 1 } _ { T } ^ { \top } Q ( A ) w + c , } \end{array}
$$

where c is independent of w.

Therefore for fixed A, we obtain a Trust Region Subproblem where the goal is to find a w orthogonal to ${ \bf 1 } _ { T } .$ which can be solved using the Mor´e-Sorensen algorithm (Mor´e and Sorensen, 1983). Our implementation solves a tridiagonal variant of this subproblem using Scipy’s solve banded.

SDP relaxation. Consider our SDP relaxation,

$$
\begin{array} { r l } & { \underset { A \in \mathcal { A } , Y \succeq 0 , z \in [ 0 , 1 ] ^ { T } } { \operatorname* { m i n } } \langle Q ( A ) , Y \rangle } \\ & { \mathrm { s . t . ~ } \mathrm { d i a g } ( Y ) = z , \mathbf { 1 } _ { T } ^ { \top } z = T - s \mathrm { ~ a n d ~ } \left[ 1 z ^ { \top } \right] \succeq 0 . } \end{array}
$$

Let $h = 2 z - \mathbf { 1 } _ { T }$ . Then $\mathbf { 1 } _ { T } ^ { \top } h = T - 2 s$ . Define $X = { \left[ \begin{array} { l l } { 1 } & { h ^ { \top } } \\ { h } & { H } \end{array} \right] }$ so that

$$
\left[ \begin{array} { l l } { 1 } & { z ^ { \top } } \\ { z } & { Y } \end{array} \right] = L X L ^ { \top }
$$

where $L = \left[ \begin{array} { c c } { { 1 } } & { { 0 } } \\ { { \frac { 1 } { 2 } \mathbf { 1 } _ { T } } } & { { \frac { 1 } { 2 } I _ { T } } } \end{array} \right]$ . Since L is invertible, $X \succeq 0$ as well. Moreover, since $\mathrm { d i a g } ( Y ) = z .$ , and since

$$
Y _ { t t } = \frac { 1 } { 4 } ( 1 + 2 h _ { t } + H _ { t t } ) = z _ { t } = \frac { 1 } { 2 } ( 1 + h _ { t } )
$$

we obtain $\mathrm { d i a g } ( H ) = \mathbf { 1 } _ { T }$ . Furthermore, $\left. Q ( A ) , Y \right. = \left. C , X \right.$ +const for a matrix C depending on $Q ( A )$ . Hence, the feasible set is the elliptope intersected with the single linear constraint $\mathbf { 1 } _ { T } ^ { \top } h = T - 2 s$ . The map $( z , Y )  X$ is a bijection between the feasible sets that preserves the objective up to a constant, so the optimal values agree and optimal solutions correspond one-to-one. Given this, we use the Burer-Monteiro (Burer and Monteiro, 2003; Boumal et al., 2016) to solve for z.

Bi-convex relaxation. For the bi-convex relaxation, when A is fixed, we solve for z using cvxpy.

Convexifying $Q ( A )$ to estimate z. Here, we solve the resulting convex problem for z directly using cvxpy.

A-constrained weighted least squares. For Sphere-AM, SDP-AM, Biconvex-AM, and ConvexQ $( \mathtt { A } ) \mathtt { - A M } ,$ when z is fixed, we solve for A using FISTA (Beck and Teboulle, 2009) with a projection on to $\left\| \cdot \right\| _ { 1 , 1 }$ ball (where the projection is computed in $O ( n ^ { 2 } \log { n } )$ by a sorting-based algorithm (Duchi et al., 2008)).

Least squares with $\ell _ { 1 } / \ell _ { 2 }$ block-sparse penalization. Let $\rho _ { t } ( A ) = D _ { t } ( \tilde { x } _ { t + 1 } - A \tilde { x } _ { t } )$ . When A is fixed, the objective separates across columns as $\begin{array} { r } { \sum _ { t } \left[ \frac { 1 } { 2 } \| \rho _ { t } - u _ { t } \| ^ { 2 } + \lambda \| u _ { t } \| _ { 2 } \right] } \end{array}$ , and each term is minimized by block softthresholding, $\begin{array} { r } { u _ { t } = \bigg ( 1 - \frac { \lambda } { \| \rho _ { t } \| _ { 2 } } \bigg ) _ { + } \rho _ { t } } \end{array}$ . When U is fixed, the penalty is a constant, and the problem reduces to weighted least squares,

$$
A = \arg \operatorname* { m i n } _ { A \in \mathcal { A } } \sum _ { t : D _ { t } = 1 } \big \| ( \tilde { x } _ { t + 1 } - u _ { t } ) - A \tilde { x } _ { t } \big \| _ { 2 } ^ { 2 } .
$$

With A set to the $\left\| \cdot \right\| _ { 1 , 1 }$ ball, we again solve for A using FISTA algorithm (Beck and Teboulle, 2009) with a projection on to $\left\| \cdot \right\| _ { 1 , 1 }$ ball (where the projection is computed in $O ( n ^ { 2 } \log { n } )$ by a sorting-based algorithm (Duchi et al., 2008)).

Least squares with block-sparse hard constraints. When A is fixed, solving for U involves selecting the 2s columns of U with the largest $\ell _ { 2 }$ norm. With U fixed this translates to A-constrained weighted-least-squares.

## E.2 Additional results on synthetic experiments

Contamination models. In this set of experiments we consider four diferent contamination models.

1. Random heavy-tailed: outliers drawn from a Cauchy distribution with location and scale both set to 10 are added to indices selected uniformly at random (with probability depending on s).

2. Block burst: equally spaced blocks of five contiguous entries are contaminated with additive errors drawn from a Cauchy distribution with location and scale both set to 10.

3. Sign flip: data points are selected uniformly at random (with probability depending on s), and their signs are flipped (i.e., they are multiplied by −1).

4. High-leverage: the s entries with the largest $\ell _ { 2 }$ norm are contaminated by adding Gaussian noise with standard deviation 8.

## E.2.1 Block burst contamination

![](images/7f1c50ef3df77df78f9c7c06c0e81dd6bb2f77c3130b40c91e9b5e837e1c4b86.jpg)  
(a) ϵ = 0

![](images/98fc8019512c6b473642a384f98e6cbba3d8179acd64195c215b6d4bb071cda6.jpg)  
(b) $\epsilon = 0 . 0 2$

![](images/c60e53040e30018804f7add2c82b41def79bc665c5daede4ecc205e9a7165ecb.jpg)  
(c) ϵ = 0.1  
Figure 3: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ for $T \in \{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 \}$ with (a) ϵ = 0, (b) ϵ = 0.02, and $\mathrm { ( c ) } ~ \epsilon = 0 . 1$ , for block burst contamination model.

Frobenius norm error against T. Figure 3 shows the Frobenius error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ against $T \in$ $\{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 \}$ . When $\epsilon = 0$ , all estimators achieve the same error since they coincide with OLS. While Sphere-AM fails to recover $A ^ { * }$ for $\epsilon > 0 . 0 2 \ \mathrm { o r } \ T > 1 0 0$ , it performs even better than Biconvex-AM when both ϵ and T are small. In small-ϵ regime, SDP-AM initially outperforms the other estimators, with Biconvex-AM achieving the lowest error as T increases (see Figure 3b). However, when $\epsilon = 0 . 1$ , ConvexQ(A)-AM outperforms the other estimators, with Biconvex-AM and SDP-AM achieving comparable performance. For all estimators except Sphere-AM and OLS, the error decreases as $T$ increases, although the rate of decrease slows.

![](images/8b270951612298cdcffbf4b6ccd37f3764f3db6dd6b92b08035aa00b941932d4.jpg)  
(a) T = 100

![](images/e27a30ceb263eeba60a36ac18bbfb1a6eec9c6dc6d6324d90739d72a5e1d0c5d.jpg)  
(b) T = 400

![](images/79b0b93f620a317ddf39dcf0834e718aa57a3ccf895b6ae134e41e37606354b2.jpg)  
(c) T = 800  
Figure 4: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ for $\epsilon = \lbrace 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \rbrace$ with (a) T = 100, (b) $T = 4 0 0$ , and (c) $T = 8 0 0$ , for block burst contamination model.

Frobenius norm error against ϵ. Figure 4 shows the Frobenius error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ against $\epsilon \in$ $\{ 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \}$ . We observe similar behavior across all $T ~ \in ~ \{ 1 0 0 , \ddot { 4 } 0 0 , 8 0 0 \}$ with the OLS and the Sphere-AM showing the highest error. As expected, the error increases with ϵ. While SDP-AM achieves a smaller error in the small-ϵ-regime, Biconvex-AM and ConvexQ(A)-AM perform best as ϵ increases. Note that while Sphere-AM generally performs comparably to OLS, it outperforms all estimators except SDP-AM when $T \leq 1 0 0$ and $\epsilon \leq 0 . 0 2$ , similar to the heavy-tailed contamination example.

## E.2.2 Sign flip contamination

![](images/2f7ecce8081b7ea867d546140e3742ae172a27b7742dffd6325b31ad1806537a.jpg)  
(a) ϵ = 0

![](images/577177d7efd818e5fa8e64b281176fa857674d9ea4a58a1053ab6df16705032c.jpg)  
(b) ϵ = 0.02

![](images/c74fff6afb5ae3c7beecf343c1267e636212248899e6d67872e13a9ed3359900.jpg)  
(c) ϵ = 0.1  
Figure 5: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } .$ , for $T \in \{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 \}$ with (a) ϵ = 0, (b) ϵ = 0.02, and $\mathrm { ( c ) } ~ \epsilon = 0 . 1$ , under the sign flip contamination model.

Frobenius norm error against T. Figure 5 shows the Frobenius error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ against $T \in$ $\{ 5 0 , 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 \}$ When $\epsilon = 0$ , all estimators achieve the same error as they coincide with OLS. An interesting observation is that, under sign flip contamination, all estimators including OLS perform well. Also note that, LS-BSP coincides with OLS, and Sphere-AM outperforms both. There is almost no gap among the remaining estimators, which all exhibit similar rates. The behavior of LS-BSP can be attributed to the fact that sign flip preserves the magnitude of the entries and with our choice of λ, $U = 0$ becomes optimal, and the estimator collapses to OLS. Notably,OLS exhibits the same error-decay with respect to $T$ as the other estimators under sign flip contamination, so this behavior does not conflict with our guarantees for LS-BSP.

Frobenius norm error against ϵ. Figure 6 shows the Frobenius error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ against $\epsilon \in$ {0, 0.01, 0.02, 0.05, 0.1}. We observe similar behavior across all $T \in \{ 1 0 0 , 4 0 0 , 8 0 \ddot { 0 } \}$ , with OLS and LS-BSP showing

SDP-AM Sphere-AM

![](images/c0256fbf5e6e66e7f274ea7af57497d8ac7856bbbe1a6f77a5bc1307ac3b8124.jpg)  
(a) T = 100

![](images/654919d2466c70ef5fc6bcb6196fcb0121579a8bb2e0453dd284c76912c476d2.jpg)  
(b) T = 400

![](images/bf72fa76f0b2dad784bf504afbd0e12a8f7fdad1a0b8845ee40bbcf196d785d1.jpg)  
(c) T = 800  
Figure 6: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ for $\epsilon = \lbrace 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \rbrace$ with (a) T = 100, (b) $T = 4 0 0$ , and (c) $T = 8 0 0$ , under the sign $\mathit { f i p }$ contamination model.

the highest error. As expected, the error increases with ϵ. Note that Biconvex-AM, SDP-AM, and LS-BSHC outperform the other estimators across all settings.

## E.2.3 High-leverage contamination

![](images/0c03e5466baa70c58a01d04cfc780cc19a84f3898e2e73eacbe300ff9c4f4d33.jpg)  
(a) ϵ = 0

![](images/2dce03029b9aed8587e8e71ae3c690310759f1b40aa00c157ad426f7b2bcecc6.jpg)  
(b) ϵ = 0.02

![](images/273c15cf1ff2c13106c8ec1b81f3f75363dcdfa7e8edd35c87c1e69f54fcb958.jpg)  
(c) ϵ = 0.1  
Figure 7: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } .$ , for T ∈ {50, 100, 200, 400, 800} with (a) ϵ = 0, (b) ϵ = 0.02, and (c) ϵ = 0.1, under the high-leverage contamination model.

Frobenius norm error against T. Figure 7 shows Frobenius error, $\| \widehat { A } - A ^ { * } \| _ { F } / \| A ^ { * } \| _ { F }$ against $T \in$ {50, 100, 200, 400, 800}. When $\epsilon = 0$ , all estimators achieve the same error as they coincide with OLS. As ϵ increases, the error of every estimator still decays with T, although OLS and LS-BSP decay considerably more slowly than the others. SDP-AM, Biconvex-AM, and LS-BSHC perform well, with Biconvex-AM achieving a recov ery error close to that of A-constrained least squares on uncontaminated data. Since this contamination model simply adds zero-mean Gaussian noise, we would expect all estimators to have decaying error (albeit diferent rates) which aligns with the observations. The behavior of LS-BSP can be attributed to the choice of $\lambda ,$ which we set match the λ in the theoretical guarantees. This λ is chosen to dominate the Gaussian innovations $\eta _ { t } ,$ , so the additional zero-mean Gaussian corruption could also be absorbed by λ. As a result, LS-BSP behaves comparably to OLS, which also shows decaying error in this setting.

Frobenius norm error against ϵ. Figure 8 shows the Frobenius error, $| \widehat { A } - A ^ { * } | | _ { F } /  A ^ { * }  _ { F }$ against $\epsilon \in$ {0, 0.01, 0.02, 0.05, 0.1}. We observe similar behavior across all $T \in \{ 1 0 0 , 4 0 0 , 8 0 0 \}$ , with OLS and LS-BSP showing the highest error. As expected, the error increases with ϵ. Note that Biconvex-AM,SDP-AM, and LS-BSHC stil achieve strong recovery.

![](images/a1cd994c21654157f7fbbf8bf39a5ad033dc6044f2677817ac3feb982658c1c8.jpg)  
(a) T = 100

![](images/c071a7e31725000efc654b6012a269510de51a10628f50a6d446dd654b698b95.jpg)  
(b) T = 400

![](images/5b2880375ef6c3eeda7570d9cb5cee20fb3faad2ef3acc73174f1cfab56d1396.jpg)  
(c) T = 800  
Figure 8: Frobenius norm error, $\lVert \widehat { A } - A ^ { * } \rVert _ { F } / \lVert A ^ { * } \rVert _ { F } ,$ for $\epsilon = \lbrace 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \rbrace$ with (a) $T = 1 0 0 ,$ (b) $T = 4 0 0$ , and (c) $T = 8 0 0$ , under the high-leverage contamination model.