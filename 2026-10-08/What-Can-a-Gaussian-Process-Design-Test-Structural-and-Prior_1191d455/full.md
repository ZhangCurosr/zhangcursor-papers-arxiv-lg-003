# What Can a Gaussian Process Design Test? Structural and Prior-Based Detection from a Gale Perspective

Ivan De Boi   
Department of Electromechanical Engineering University of Antwerp Campus Groenenborger – Z373   
Groenenborgerlaan 171, 2020 Antwerp, Belgium ivan.deboi@uantwerpen.be

Marnix Van Soom Artificial Intelligence Lab Vrije Universiteit Brussel Pleinlaan 2, 1050 Brussels, Belgium marnix@ai.vub.ac.be

## Abstract

A Gaussian process (GP) model can agree with the data for two reasons: its assumptions are right, or the chosen inputs could never have shown that they are wrong. The distinction can be checked from the design before any responses are observed. Every model implies relations that its noiseless responses must satisfy at the chosen inputs, such as “the middle value lies on the line through its two neighbours”. For GPs built from finitely many features, these relations are exactly the null space of the kernel matrix. Gale duality gives them a geometric interpretation, in which each observation has a vector and the smallest groups of observations that can expose an error are the circuits. For other kernels the relations become soft: response patterns may be improbable under the prior rather than algebraically impossible. A standard test then combines two kinds of evidence. Structural evidence comes from a violated relation and grows without limit as the noise falls. Prior-based evidence only says that a departure is improbable under the prior. With all inputs at the two ends of an interval, for example, a GP can reject a straight line against a large curvature, but only because the implied intercept is improbable, never because curvature was seen. In simulations the predicted power matched the observed rejection rates. Choosing the next input by predicted power raised the power against a localised discrepancy from 0.48 to 0.72, against 0.51 when choosing by predictive variance, and a grid in two dimensions contained exact tests of additivity that a Latin hypercube lacked. The test itself is classical. The contribution is the prospective reading of that test: before observing the responses, the design already determines what kind of contradiction it can produce.

## 1 Introduction

Gaussian processes (GPs) are widely used as surrogate models when data are expensive, most prominently in Bayesian optimisation and active learning [54, 56, 30], with applications ranging from the calibration of measurement systems [21] to clinical assessment [23] and the tuning of accelerator beamlines [53]. Their appeal rests on calibrated uncertainty, but that calibration is conditional on the model. A GP model can pass the checks applied to the collected data for two very diferent reasons: because its assumptions are adequate, or because the data could never have contradicted them. Consider a straight-line model and a design that places all observations at the two ends of an interval. A straight line can always be drawn through the mean responses at two inputs, so the data agree with the model whatever the true curvature is, and no amount of further data at the same inputs can change this. Agreement with the data is then no evidence that the function is straight.

Diagnosing a model therefore involves two diferent questions, which this paper keeps apart. The first is detection: after the data are collected, are some observations incompatible with the model? Classical regression diagnostics and their GP counterparts answer this question [19, 9, 20, 7]. The second is testability: before the data are collected, which violations of the model could the design reveal at all, and with what probability? For linear models and a specified alternative, designs for detecting model inadequacy [12, 2, 40] and for discriminating between rival models [3, 25] answer this question. For GPs, designs that discriminate between specified rival models exist [29, 48]. This work makes explicit, for a single GP model, which departures a given design can detect and with what power. Such an account must handle correlated observations and kernels that need not correspond to a finite set of features. This matters because the designs used with GPs are usually chosen for other purposes. Observations placed where predictive uncertainty is largest need not be the ones that would most strongly challenge the model assumptions.

Figure 1 shows both points on simple examples. In panel (a), a straight-line model is tested against a curvature $c x ^ { 2 }$ with 40 observations. When all inputs lie at the two ends of an interval, the residual test stays at its level for every curvature. A GP test would eventually reject a very large curvature at that design, but only because the intercept it implies is improbable under the prior, not because any relation is violated (Section 5). Moving four of the observations to the centre makes the same curvature structurally detectable. In panel (b), the next observation that would best test a GP against a narrow bump lies at the bump, where the predictive variance is small, and not at the boundary, where it is largest. Both answers are available before any response is measured.

(a) can the design test a straight line?  
![](images/9fc3233896919f9828c3f7ebdddd65625ee80b58d1c04f3a8954e870ee248230.jpg)

(b) where should the next observation go?  
![](images/1f493ccba8bbdb4f141c76f4baa6b45e3179dbfb9a19e49e297a8d46ffa789ef.jpg)  
Figure 1: What a design can test, computed before any data are collected (details in Section 7). (a) Power of two tests of a straight-line model against a curvature $c x ^ { 2 } .$ , for 40 inputs at the two ends of $[ - 1 , 1 ]$ , and for 36 inputs at the ends with 4 at the centre. Lines show predicted power, markers simulated rejection rates. The end-point design contains no consistency relation that involves curvature. (b) Power of a GP test against a narrow bump at $x _ { 0 } = 0 . 1$ (vertical line) after adding one observation at x (solid), and the current predictive variance (dashed, scaled). Ticks mark the current inputs, and the dotted line the current power.

A model can only produce certain response patterns. Every linear relation shared by all of those patterns gives a check on the data. We call such a relation a consistency relation. If a function is assumed to be a straight line, for example, then at three equally spaced inputs the middle value must equal the average of its neighbours. If the model is correct, a consistency relation can fail only by as much as the noise explains. At the two-ended design above, no consistency relation involves curvature, and that is why the design cannot test it.

For GPs built from finitely many features, the consistency relations are exactly the null space of the kernel matrix, the combinations of function values that the prior fixes at zero. We describe these relations geometrically with Gale duality [28, 63, 24]. It gives each observation a vector, its Gale vector, that describes its role in the relations, and it singles out the circuits, the smallest groups of observations that can expose a structural error. For other kernels, such as the radial basis function (RBF) kernel, no exact relations exist. Some patterns are then merely unlikely, and we use the spectrum of the kernel matrix to describe these soft restrictions. Gale duality is the language for the exact case, not the source of the algebra.

The relations span the same residual space as classical diagnostics. Our interest is therefore not in constructing another retrospective residual test, but in using that space prospectively. Before observing y, the model and the chosen inputs already determine which departures can leave a residual and which can be absorbed by the model. This makes testability a property of the design itself.

For a finite-rank GP, the exact relations are the null space of the kernel matrix. Gale duality provides a geometric representation of these relations and identifies their minimal supports through circuits. For a full-rank GP, no exact relations remain, but the kernel spectrum still separates strongly penalised response directions from flexible ones. The corresponding power calculation distinguishes evidence from a violated relation from evidence that is due only to prior improbability.

The residual projector, multiple-case diagnostic identities, noncentral chi-square power calculations and classical discrimination criteria used below are not new. The contribution is the way they are put together to answer a design question: what could this GP design actually contradict, and what kind of evidence would a rejection represent? Because this question can be answered before the responses are measured, it also gives a prospective criterion for choosing observations. The analysis assumes a fixed mean, kernel and noise variance; estimation from the same data and repeated testing are treated as limitations rather than folded into the exact theory.

The same distinction is already latent in the marginal likelihood. If $\begin{array} { r } { K = \sum _ { j } \lambda _ { j } u _ { j } u _ { j } ^ { \intercal } } \end{array}$ , then $y ^ { \top } ( K + \sigma ^ { 2 } I ) ^ { - 1 } y = \textstyle \sum _ { i } ( u _ { i } ^ { \top } y ) ^ { 2 } / ( \lambda _ { j } + \sigma ^ { 2 } )$ . Our use of this decomposition is to attach those directions to exact or soft relations of the design and to compute, before measurement, the power against a specified departure. The experiments then ask whether that prospective calculation agrees with simulation and whether it selects diferent observations from predictive variance.

## 2 Related work

Influence and outlier diagnostics for linear models have a long history, including Cook’s distance [19], the systematic treatments of [9, 20], and robust alternatives [55, 4]. A well-known dificulty is masking: several outliers can conceal one another so that each appears innocuous when assessed individually [8]. Group and subset-based procedures were developed precisely for this situation [37, 35]. Our group tests are GP analogues of these multiple-case diagnostics.

Closed-form leave-one-out (LOO) residuals for kriging and GP regression follow from the inverse covariance matrix [27, 58, 54]. Multiple-fold residuals and their covariances admit analogous block formulae [32]. Diagnostics for GP emulators of computer models, including Mahalanobis-type statistics and pivoted Cholesky errors, were systematised in [7]. The relative behaviour of maximum likelihood and cross-validation for hyperparameter estimation under misspecification is studied in [6]. Approximate LOO for non-Gaussian likelihoods is reviewed in [60]. The group tests of Section 5 are of this classical type. Here they are used prospectively: their power is computed from the design and interpreted through the model’s consistency relations.

Instead of testing a fixed observation model, one can make it more flexible: Student-t likelihoods [41], input-dependent noise [33, 43], or data-point-specific noise variances selected by sequential relevance pursuit [1]. These approaches repair the model rather than diagnose it, and are natural alternatives once a test has identified an incompatibility.

In the calibration of computer models, systematic mismatch between a model and reality is represented by an explicit discrepancy term [42], and ignoring it can lead to biased and overconfident inference [16]. In Bayesian inference, model criticism through predictive checks goes back to [14] and was developed further in [31]. Our tests are frequentist checks of the hypothesis that the data come from a fully specified GP (Section 5), but they share the aim of separating criticism from estimation.

Designs that detect the inadequacy of a regression model go back to the treatment of bias from omitted terms in [12], and were developed into explicit design criteria in [2, 40]. Choosing experiments to discriminate between rival models has a similarly long history, from the entropybased sequential designs of [13] and the T-optimality criterion of [3] to its later analysis and extensions [25, 44, 5]. In machine learning, active model selection chooses observations to discriminate between GP models [29], and GP surrogates make discrimination designs available for non-analytical models [48]. Section 5.5 positions our power results relative to this literature.

Gale duality originates in the study of convex polytopes [28] and is treated in standard texts [34, 63, 46, 24]. Algebraic statistics describes the identifiable polynomial models of a design through its design ideal [50, 51], and circuits have been used to generate randomisation schemes [49]. Classical geometric structure has been combined with GP models before, for example line geometry for surface approximation with GP latent variable models [22]. Here Gale duality serves as the language that makes the relations implied by a GP design, and their efect on the power of its tests, explicit.

## 3 Gale duality and model consistency relations

Consider first a linear regression model with a finite set of features. Its feature matrix spans the response vectors that the model can represent exactly. The orthogonal complement contains the linear relations that every noiseless response must satisfy. Gale duality organises those relations observation by observation and identifies their smallest supports.

## 3.1 Gale duality

We begin with the classical construction. Let $p _ { 1 } , \ldots , p _ { n } \in \mathbb { R } ^ { d }$ be n points whose afine hull is all of $\mathbb { R } ^ { d }$ . An afine dependence of the configuration is a vector $\gamma \in \mathbb { R } ^ { n }$ with

$$
\sum _ { i = 1 } ^ { n } \gamma _ { i } = 0 , \qquad \sum _ { i = 1 } ^ { n } \gamma _ { i } p _ { i } = 0 .\tag{1}
$$

Informally, an afine dependence is a way of weighting the points so that both the weights and the weighted positions cancel. The afine dependences form a linear subspace of $\mathbb { R } ^ { n }$ of dimension $n - d - 1$ . They matter for data because an afine dependence annihilates the values of every afine function. If $y _ { i } = a + b ^ { \top } p _ { i }$ , then $\textstyle \sum _ { i } \gamma _ { i } y _ { i } = a \sum _ { i } \gamma _ { i } + b ^ { \top } \sum _ { i } \gamma _ { i } p _ { i } = 0$ by (1). Example 1 works this out for three points on a line.

Choose a basis of this subspace and write it as the columns of a matrix $G \in \mathbb { R } ^ { n \times ( n - d - 1 ) }$ The n rows $g _ { 1 } ^ { \top } , \ldots , g _ { n } ^ { \top }$ of $G _ { i }$ , with $g _ { i } \in \mathbb { R } ^ { n - d - 1 }$ , form the Gale transform of the configuration: a second configuration of n vectors, one per original point, in a space of complementary dimension [28, 34, 63, 46, 59]. The primal configuration records where the points are. The dual configuration records how they depend on each other. The power of the construction is that properties of the one can be read of from the other. In this paper the property read of is which sets of observations witness which relations (Proposition 1).

The same construction works far beyond points in space. Conditions (1) say that $\gamma$ is orthogonal to every column of the $n \times ( d + 1 )$ matrix with rows $( 1 , p _ { i } ^ { \top } )$ , the vectors that describe the points. Replacing these rows by any feature vectors gives the version we use throughout.

We write X for the input space and $x _ { 1 } , \ldots , x _ { n } \in { \mathcal { X } }$ for the n inputs, which together form the design. A feature map $\phi : \mathcal { X }  \mathbb { R } ^ { m }$ turns each input into m features, and the feature matrix $\Phi \in \mathbb { R } ^ { n \times m }$ has one row $\phi ( x _ { i } ) ^ { \top }$ per observation. Its rank is $r = \mathrm { r a n k } ( \Phi )$ . The relations live in nul $| ( \Phi ^ { \top } ) = \{ v \in \mathbb { R } ^ { n } : \Phi ^ { \top } v = 0 \}$ , a subspace of dimension $n - r .$

Definition 1 (Gale representation of a feature configuration). A matrix $G \in \mathbb { R } ^ { n \times ( n - r ) }$ whose columns form a basis of null $( \Phi ^ { \top } )$ is a Gale representation of the configuration of rows of Φ. Its ith row $g _ { i } ^ { \top }$ , with $g _ { i } \in \mathbb { R } ^ { n - r }$ , is the Gale vector of observation i.

For the afine feature map $\phi ( x ) = ( 1 , x ^ { \top } ) ^ { \top }$ with $\mathcal { X } = \mathbb { R } ^ { d }$ , Definition 1 recovers the classical Gale transform.

Example 1 (Three points on a line). Take the three points 0, 1, 2 of the real line $( n = 3$ $d = 1 )$ with the afine feature map, so that

$$
\Phi = \left( { \begin{array} { l l } { 1 } & { 0 } \\ { 1 } & { 1 } \\ { 1 } & { 2 } \end{array} } \right) , \qquad G = { \frac { 1 } { \sqrt { 6 } } } \left( { \begin{array} { l } { 1 } \\ { - 2 } \\ { 1 } \end{array} } \right) ,
$$

with $r = 2$ and $n - r = 1$ . One checks that $\Phi ^ { \top } G = 0$ . Every afine dependence is a multiple of $( 1 , - 2 , 1 )$ , since $1 - 2 + 1 = 0$ and $1 \cdot 0 - 2 \cdot 1 + 1 \cdot 2 = 0$ , and this is the straight-line relation of the introduction. The values 1, 3, 5 that a straight line takes at the three points satisfy $1 - 2 \cdot 3 + 5 = 0$ , whereas the values 1, 4, 5, which no straight line passes through, give $1 - 2 \cdot 4 + 5 = - 2$ . The Gale vectors are the numbers $1 / \sqrt { 6 } , - 2 / \sqrt { 6 }$ and $1 / { \sqrt { 6 } } ;$ : the middle point plays the opposite role to its two neighbours.

A Gale representation is not unique. Another basis of $\mathrm { \ n u l l } ( \Phi ^ { \top } )$ replaces G by GM for an invertible $M \in \mathbb { R } ^ { ( n - r ) \times ( n - r ) }$ , which changes the Gale vectors but not the relations they encode. Unless stated otherwise we therefore take G with orthonormal columns, $G ^ { \top } G = I _ { n - r } .$ where $I _ { k }$ is the $k \times k$ identity matrix and I without subscript is $n \times n$ . Only rotations then remain free, so the lengths of the Gale vectors and the angles between them are meaningful.

Classically, the information of interest is combinatorial and is read of from the signs of the Gale vectors. In this paper we mainly use their lengths and inner products, because these determine the leverage of an observation (Section 3.3) and the power of the tests (Section 5.2).

## 3.2 Consistency relations of a model

Now attach a response $y _ { i } \in \mathbb { R }$ to each input, and collect them in $y \in \mathbb { R } ^ { n }$ . Consider the linear-in-parameters model

$$
y = \Phi w + \varepsilon , \qquad \varepsilon \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } I ) ,\tag{2}
$$

with parameter vector $w \in \mathbb { R } ^ { m }$ and observation-noise variance $\sigma ^ { 2 } > 0$ . The column space co $( \Phi ) \subseteq \mathbb { R } ^ { n }$ contains every noiseless response vector that the model can represent. Each column $\gamma \in \mathbb { R } ^ { n }$ of a Gale representation satisfies $\gamma ^ { \top } \Phi = 0$ , and hence $\gamma ^ { \top } \Phi w = 0$ for every

w. Every exactly representable response therefore satisfies $\gamma ^ { \top } y = 0$ . We call such a linear constraint a consistency relation of the model at the design. The vector

$$
G ^ { \top } y = \sum _ { i = 1 } ^ { n } y _ { i } g _ { i } \in \mathbb { R } ^ { n - r }\tag{3}
$$

collects the values of a complete, non-redundant set of consistency relations. It vanishes for every response the model can represent exactly, and its departures from zero are attributable either to noise or to a violation of the model. In this picture, observation i contributes its Gale vector $g _ { i }$ , weighted by its response, to the total failure of the relations. The Gale vectors are fixed by the model and the design before any response is observed. They determine how a departure at a given observation can show up in the tests.

## 3.3 Residual projector, leverage and noise

The Gale representation is closely tied to classical regression diagnostics. When Φ has full column rank, the hat matrix $H = \Phi ( \Phi ^ { \top } \Phi ) ^ { - 1 } \Phi ^ { \top } \in \mathbb { R } ^ { n \times n }$ projects orthogonally onto col(Φ), and the residual projector is $P _ { N } = I - H \ [ 2 0 ]$ . To avoid a full-rank assumption, for example when there are more features than distinct inputs, we write more generally

$$
P _ { N } = I - \Phi \Phi ^ { + } = G G ^ { \top } \in \mathbb { R } ^ { n \times n } ,\tag{4}
$$

where $\Phi ^ { + } \in \mathbb { R } ^ { m \times n }$ is the Moore–Penrose pseudo-inverse. The matrix $\Phi \Phi ^ { + }$ is the orthogonal projector onto col(Φ) for any Φ, and it coincides with H in the full-rank case. The second equality holds because both sides are the orthogonal projector onto null $( \Phi ^ { \top } )$ . The vector $\begin{array} { r } { P _ { N } y = G ( G ^ { \top } y ) } \end{array}$ is the ordinary residual vector, so the Gale coordinates $G ^ { \top } y$ are the residuals expressed in an orthonormal basis of the residual space. Under (2),

$$
G ^ { \top } y = G ^ { \top } \varepsilon \sim { \mathcal { N } } ( 0 , \sigma ^ { 2 } I _ { n - r } ) , \qquad { \frac { \| G ^ { \top } y \| ^ { 2 } } { \sigma ^ { 2 } } } = { \frac { \| P _ { N } y \| ^ { 2 } } { \sigma ^ { 2 } } } \sim \chi _ { n - r } ^ { 2 } .\tag{5}
$$

This is the familiar distribution of the residual sum of squares. The distributional statement requires the model and noise assumptions, not merely the existence of a null space: the null space tells us which relations to examine, and the noise model tells us how large their failures may be by chance.

The diagonal of (4) gives

$$
\| g _ { i } \| ^ { 2 } = ( P _ { N } ) _ { i i } = 1 - h _ { i i } ,\tag{6}
$$

where $h _ { i i } = H _ { i i }$ is the leverage of observation i [9], and the of-diagonal entries give $g _ { i } ^ { \top } g _ { j } =$ $( P _ { N } ) _ { i j } . \mathrm { ~ A ~ }$ point with large leverage has a short Gale vector and hence a small footprint in the residual space: the fitted model can absorb most of a perturbation at that point by adjusting its coeficients. Conversely, a long Gale vector means that a perturbation at that point is largely visible in the residuals. Leverage is thus a statement about the observability of a local departure, and this is how we shall use it. Example 2 illustrates these quantities.

Example 2 (A planar model on the unit square). Take the four corners $( 0 , 0 ) , ( 1 , 0 ) , ( 1 , 1 ) , ( 0 , 1 )$ of the unit square, numbered 1 to 4, and the centre $( 1 / 2 , 1 / 2 )$ , numbered 5, with the planar model $\phi ( x ) = ( 1 , x _ { 1 } , x _ { 2 } )$ . Here $n = 5 , m = r = 3$ , and $\Phi \in \mathbb { R } ^ { 5 \times 3 }$ , so there are $n - r = 2$ independent consistency relations. A convenient basis is

$$
y _ { 1 } - y _ { 2 } + y _ { 3 } - y _ { 4 } = 0 , \qquad y _ { 1 } + y _ { 2 } + y _ { 3 } + y _ { 4 } - 4 y _ { 5 } = 0 .\tag{7}
$$

The first states that a plane has zero mixed diference, or ‘twist’, over the square: the change in one coordinate direction is the same on both sides of the square. The second states that the

value of a plane at the centre is the average of its values at the corners. The two coeficient vectors are orthogonal, and normalising them gives $G \in \mathbb { R } ^ { 5 \times 2 }$ with Gale vectors

$$
\begin{array} { r } { g _ { 1 } = g _ { 3 } = \big ( \frac { 1 } { 2 } , \frac { 1 } { \sqrt { 2 0 } } \big ) , \quad g _ { 2 } = g _ { 4 } = \big ( - \frac { 1 } { 2 } , \frac { 1 } { \sqrt { 2 0 } } \big ) , \quad g _ { 5 } = \big ( 0 , - \frac { 4 } { \sqrt { 2 0 } } \big ) , } \end{array}\tag{8}
$$

shown in Figure 2. Hence $( P _ { N } ) _ { i i } = 0 . 3$ for each corner and $( P _ { N } ) _ { 5 5 } = 0 . 8$ , that is, leverages 0.7 and 0.2. A perturbation at the centre violates only the second relation, and 80% of its squared size appears in the residuals. A perturbation at a corner violates both relations, but only 30% of its squared size remains visible, because a tilted plane can absorb most of it. Opposite corners have identical Gale vectors, so no test can tell an ofset at one from an ofset at the other. The Gale vectors sum to zero only because the model contains an intercept.

![](images/ec5366d3792fec09be1229f2ca09e2f7ce1fc76a8621f3fa9a94a652326f85bd.jpg)  
(a) design in $\mathbb { R } ^ { 2 }$

![](images/600f337258254bf0ad1c174b3de63b7f906b0ecb12fc0ebe948a67feeb859f98.jpg)  
(b) Gale vectors in $\mathbb { R } ^ { 2 }$  
Figure 2: Example 2. (a) The design, five inputs in $\mathbb { R } ^ { 2 }$ , with the leverage $h _ { i i }$ of each observation under the planar model. (b) The corresponding Gale vectors $g _ { i } \in \mathbb { R } ^ { n - r } = \mathbb { R } ^ { 2 }$ . Opposite corners share a Gale vector. The centre has the longest Gale vector and the smallest leverage, so departures there are the most visible in the residuals.

Among all consistency relations, those with minimal support play a special role. A non-zero $\gamma \in \mathrm { n u l l } ( \Phi ^ { \top } )$ is a circuit of the configuration if no other non-zero element of null $( \Phi ^ { \top } )$ has a support strictly contained in that of γ [63, 10]. Circuits are the most local tests a design ofers. Each involves as few observations as possible, so its failure involves only those observations, although it does not identify which of them is at fault. The relation (1, −2, 1) of Example 1 is a circuit. Up to scaling, Example 2 has three circuits: the twist relation and the two diagonal relations $y _ { 1 } + y _ { 3 } - 2 y _ { 5 } = 0$ and $y _ { 2 } + y _ { 4 } - 2 y _ { 5 } = 0$ , which state that the value at the centre is the average of each pair of opposite corners. The centre relation of Example 2 is the sum of the two diagonal relations and is therefore not a circuit. Proposition 1 collects standard facts of matroid theory [10], stated in the form we use for experimental design.

Proposition 1 (Circuits are minimal witnesses). For a set $C \subseteq \{ 1 , \ldots , n \}$ of observations, write Φ<sub>C</sub> for the rows of Φ indexed by C.

(i) C supports a non-zero consistency relation exactly when the rows of $\Phi _ { C }$ are linearly dependent, that is, when rank $\left( \Phi _ { C } \right) < \left| C \right|$

(ii) The supports of the circuits are the minimal sets C with this property.

(iii) The circuits span null $( \Phi ^ { \top } )$ . A departure δ therefore violates some consistency relation, $G ^ { \top } \delta \neq 0$ , exactly when it violates some circuit.

Proof. A relation supported on C is a non-zero γ with $\gamma _ { i } = 0$ for $i \not \in C$ and $\gamma _ { C } ^ { \top } \Phi _ { C } = 0 .$ , which exists exactly when the rows of $\Phi _ { C }$ are dependent. Part (ii) is the definition of a circuit. For (iii), let $\gamma \neq 0$ be a relation. Among the relations whose support lies in that of $\gamma _ { ; }$ , one of minimal support is a circuit $\gamma ^ { \prime }$ . For a coordinate i in the support of $\gamma ^ { \prime }$ , the relation $\gamma - ( \gamma _ { i } / \gamma _ { i } ^ { \prime } ) \gamma ^ { \prime }$ has a strictly smaller support than γ, and induction on the size of the support writes $\gamma$ as a combination of circuits. If every circuit annihilated $\delta ,$ so would every relation. □

A circuit support is thus the smallest set of measurements on which the model imposes a non-trivial exact constraint. Every proper subset of it permits arbitrary noiseless responses, whereas the responses on the full circuit must satisfy one consistency relation. By (iii), every structural contradiction that a design can produce is witnessed by such a minimal set. In Example 2 the twist is witnessed by the four corners alone, and the diagonal relations by three observations each. For polynomials of degree $q$ in one dimension, any $q + 2$ distinct inputs form a circuit, which is the minimal design of Example 3. This is the role of Gale duality in this paper: it organises the exact relations through their minimal supports, making the local structure of structural testability explicit.

## 3.4 Structural assumptions from feature maps

Definition 1 applies to any feature map, and the feature map determines which structural assumption the relations encode. For polynomial models the relevant feature map is the Veronese map. In one dimension a polynomial of degree at most $q$ is a linear function of

$$
\phi _ { q } ( x ) = \left( 1 , x , x ^ { 2 } , \ldots , x ^ { q } \right) ^ { \top } \in \mathbb { R } ^ { q + 1 } ,\tag{9}
$$

and in d dimensions of the vector $\phi _ { q } ( x )$ of all $\textstyle { \binom { d + q } { q } }$ monomials of degree at most $q ,$ which is the afine form of the Veronese map of algebraic geometry [47]. The Gale representation of the configuration $\phi _ { q } ( x _ { 1 } ) , . . . , \phi _ { q } ( x _ { n } )$ therefore contains exactly the linear relations that every polynomial of degree at most q satisfies at the design. A departure of higher degree is structurally visible exactly when it violates one of them, that is, when it is not representable by a polynomial of degree q at the design. Example 3 works this out for the question that runs through the paper.

Example 3 (A straight line against curvature). Take the straight-line model, $\phi _ { 1 } ( x ) = ( 1 , x ) ^ { \top }$ and a curvature $\delta _ { i } = x _ { i } ^ { 2 }$ . Suppose that the design has k distinct inputs, each possibly repeated. Then $r = \mathrm { r a n k } ( \Phi ) = \mathrm { m i n } ( k , 2 )$ , and there are $n - r$ consistency relations.

• Two distinct inputs $a \neq b$ . Every relation compares replicates, $y _ { i } - y _ { j } = 0$ for $x _ { i } = x _ { j }$ Any function takes equal values at equal inputs, so no relation involves curvature. Indeed, the vector $( x _ { i } ^ { 2 } )$ equals the straight line $a ^ { 2 } + ( a + b ) ( x - a )$ at both inputs, so it lies in col(Φ) and $G ^ { \top } \delta = 0$ . No number of observations at the two inputs can reveal curvature.

• Three distinct inputs $a < b < c$ . Besides the replicate comparisons there is one relation between the responses at the three inputs, $\left( c - b \right) y ( a ) - \left( c - a \right) y ( b ) + \left( b - a \right) y ( c ) = 0$ which for $a , b , c = 0 , 1 , 2$ is the relation $( 1 , - 2 , 1 )$ of Example 1. On the curvature it takes the value $( c - b ) a ^ { 2 } - ( c - a ) b ^ { 2 } + ( b - a ) c ^ { 2 } = ( b - a ) ( c - b ) ( c - a ) > 0$ , so curvature always violates it, and more strongly the more the inputs are spread out.

The same argument settles degree q against degree $q + 1$ . If the design has at most $q + 1$ distinct inputs, a polynomial of degree q can interpolate $x ^ { q + 1 }$ at all of them, so the departure violates no relation. If it has $q + 2$ distinct inputs, no such polynomial exists, since $\boldsymbol { x } ^ { q + 1 } - \boldsymbol { p } ( \boldsymbol { x } )$ would then be a non-zero polynomial of degree $q + 1$ with $q + 2$ roots. A design can therefore test degree q against degree $q + 1$ exactly when it has at least $q + 2$ distinct inputs. In several dimensions the number of distinct inputs is no longer enough, and their position matters. The twist $x _ { 1 } x _ { 2 }$ , for example, cannot be tested against a plane on any design whose inputs all lie on one line parallel to an axis, since there it equals an afine function. Which departures a design can test is then a question about the Veronese configuration of the inputs, and Section 5.4 answers it with the Gale representation. Which polynomial terms a design can identify, and which are aliased with one another, is the subject of the algebraic theory of designs, in which the design is described by a polynomial ideal [50, 51]. This example is the simplest case of that theory. Our use of it is to carry the relations over to GP kernels and to their power.

Other structural hypotheses imply relations of the same kind.

Example 4 (Periodicity and additivity). A model with known period τ implies $f ( x ) -$ $f ( x + \tau ) = 0$ for the noiseless values whenever both inputs are observed. $\mathrm { A n }$ additive model $f ( x _ { 1 } , x _ { 2 } ) = f _ { 1 } ( x _ { 1 } ) + f _ { 2 } ( x _ { 2 } )$ implies, on any $2 \times 2$ grid of inputs $\{ a , a ^ { \prime } \} \times \{ b , b ^ { \prime } \}$ , the mixed diference

$$
f ( a , b ) - f ( a , b ^ { \prime } ) - f ( a ^ { \prime } , b ) + f ( a ^ { \prime } , b ^ { \prime } ) = 0 ,\tag{10}
$$

and the twist relation of Example 2 is the special case in which both $f _ { 1 }$ and $f _ { 2 }$ are linear.

Changing the feature map therefore changes the assumptions being tested. A polynomial lift is appropriate for a polynomial hypothesis. It does not automatically describe the assumptions of a GP with a radial basis function (RBF) kernel. For this reason, Section 4 constructs the relations from the kernel itself.

## 4 Consistency tests for Gaussian processes

The same construction can be read directly from a GP kernel. For a finite-rank kernel, the null space of the kernel matrix is exactly the space of consistency relations. A strictly positive-definite kernel has no non-trivial null space at distinct inputs; in that case the relevant information is spectral rather than algebraic.

A Gaussian process (GP) is a probability distribution over functions $f : \mathcal { X } $ R such that, for any finite set of inputs, the vector of function values is jointly Gaussian [54]. A zero-mean GP is determined by its covariance function, or kernel, $k ( x , x ^ { \prime } ) = \operatorname { C o v } ( f ( x ) , f ( x ^ { \prime } ) )$ , and we write $f \sim \mathcal { G P } ( 0 , k )$ . The linear model of Section 3 is a special case. If the parameters in (2) receive a prior $\boldsymbol { w } \sim \mathcal { N } ( \boldsymbol { 0 } , \Sigma _ { w } )$ , with a positive definite $\Sigma _ { w } \in \mathbb { R } ^ { m \times m }$ , then $f ( x ) = \phi ( x ) ^ { \top } w$ is a (degenerate) GP with kernel $k ( x , x ^ { \prime } ) = \phi ( x ) ^ { \top } \Sigma _ { w } \phi ( x ^ { \prime } )$ . This kernel has finite rank, because it is built from m features. Kernels such as the RBF kernel $k ( x , x ^ { \prime } ) = s \exp ( - \| x - x ^ { \prime } \| ^ { 2 } / 2 \ell ^ { 2 } )$ with lengthscale $\ell > 0$ and output scale $s > 0$ , correspond to infinitely many features and encode smoothness rather than a finite-dimensional functional form [54]. Conditioning a GP on noisy observations gives a posterior mean and variance at every input in closed form, which is what makes GPs attractive as surrogate models. We now ask: which consistency relations does such a model imply at a design?

Model the responses as $y _ { i } = f ( x _ { i } ) + \varepsilon _ { i }$ , with a latent function $f \sim \mathcal { G P } ( 0 , k )$ with kernel $k : \mathcal { X } \times \mathcal { X }  \mathbb { R }$ , and independent noise $\varepsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ . We work with a fixed zero mean, or with responses centred by a specified mean function. Let $K \in \mathbb { R } ^ { n \times n }$ be the positive-semidefinite kernel matrix, $K _ { i j } = k ( x _ { i } , x _ { j } )$ . Two further matrices appear throughout,

$$
A = K + \sigma ^ { 2 } I , \qquad Q = A ^ { - 1 } ,\tag{11}
$$

both in $\mathbb { R } ^ { n \times n }$ . The matrix A is the covariance of the responses $y ,$ and $Q$ is its inverse, the precision matrix. Both depend on $\sigma ^ { 2 }$ , and we write $A ( \sigma ^ { 2 } )$ or $Q ( \sigma ^ { 2 } )$ only where that matters.

Unless stated otherwise, the kernel hyperparameters and the mean are fixed. Fitting them to the same observations changes the calibration of the tests, a point we return to in Section 5.

## 4.1 Finite-rank kernels: exact relations

For a finite-rank kernel $K = \Phi \Phi ^ { \top }$ with $\Phi \in \mathbb { R } ^ { n \times m }$ R

$$
\mathrm { \ n u l l } ( K ) = \mathrm { n u l l } ( \Phi ^ { \top } ) ,\tag{12}
$$

since for any $v \in \mathbb { R } ^ { n }$

$$
v ^ { \top } K v = \| \Phi ^ { \top } v \| ^ { 2 } .
$$

The same identity holds for $K = \Phi \Sigma _ { w } \Phi ^ { \top }$ whenever $\Sigma _ { w } \in \mathbb { R } ^ { m \times m }$ is positive definite, as in the GP obtained from the Bayesian linear model (2) with prior w $\sim \mathcal { N } ( 0 , \Sigma _ { w } )$ . The exact consistency relations can therefore be recovered directly from $K$ , even when the feature map Φ is unknown or has no convenient form.

The null space of K also has a direct reading in terms of the prior. Write $f \in \mathbb { R } ^ { n }$ for the vector of latent values $f ( x _ { i } )$ , so that $y = f + \varepsilon .$ For any $v \in \mathbb { R } ^ { n } , \operatorname { V a r } ( v ^ { \top } f ) = v ^ { \top } K v .$ , which vanishes exactly when $K v = 0$ . The null space of K is therefore the set of linear combinations of function values that the prior fixes at zero with certainty, before any data are seen. For such $\boldsymbol { v } , \boldsymbol { v } ^ { \top } \boldsymbol { y } = \boldsymbol { v } ^ { \top } \boldsymbol { f } + \boldsymbol { v } ^ { \top } \boldsymbol { \varepsilon } = \boldsymbol { v } ^ { \top } \boldsymbol { \varepsilon }$ , so any systematic signal along these directions contradicts the model. The dimension $n - \mathrm { r a n k } ( K )$ counts the (linearly) independent checks that the data can run against the prior at this design. When it is zero, the data can challenge the model only in the soft sense of Section 4.2. A relation supported on a set $C$ of observations exists exactly when $v _ { C } ^ { \top } K _ { C C } v _ { C } = 0$ for some $v _ { C } \neq 0$ , that is, when the kernel matrix $K _ { C C }$ of those inputs is singular, so the circuits of Proposition 1 can also be found from K alone.

Example 5 (Polynomial kernel). The polynomial kernel $k ( x , x ^ { \prime } ) = ( 1 + x x ^ { \prime } ) ^ { q }$ on R is the kernel of a random polynomial of degree q. Indeed, it factorises as $K = \Phi \Sigma _ { w } \Phi ^ { \top }$ with $\phi = \phi _ { q }$ and a diagonal positive definite $\Sigma _ { w } \ { \in } \ \mathbb { R } ^ { ( \bar { q } + 1 ) \times ( q + 1 ) }$ of binomial coeficients. It therefore has rank at most $q + 1$ on any design. $\operatorname { A t } q + 2$ distinct inputs its null space is spanned by a single relation, the divided diference of order $q + 1$ , which for $q = 1$ is the relation of Example 3. The kernel matrix alone thus recovers the test of polynomial degree, without any feature map.

For $q = 1 , k ( x , x ^ { \prime } ) = 1 + x x ^ { \prime }$ , and at the inputs 0, 1, 2 the kernel matrix is

$$
K = { \left( \begin{array} { l l l } { 1 } & { 1 } & { 1 } \\ { 1 } & { 2 } & { 3 } \\ { 1 } & { 3 } & { 5 } \end{array} \right) } ~ .\tag{13}
$$

One checks directly that $K ( 1 , - 2 , 1 ) ^ { \top } = 0$ . The prior therefore insists that $f ( 0 ) - 2 f ( 1 ) + f ( 2 ) =$ 0: the middle value is the average of the outer ones. With outer observations 1 and 5 the model expects a value of about 3 in the middle. An observed 4 gives $y _ { 0 } - 2 y _ { 1 } + y _ { 2 } = - 2$ . Under the model this combination has variance $( 1 , - 2 , 1 ) A ( 1 , - 2 , 1 ) ^ { \top } = 6 \sigma ^ { 2 }$ , so its size relative to ${ \sqrt { 6 } } \sigma$ decides whether it is plausibly noise or a failure of the straight-line assumption. The null space supplies the relation, while the data and the noise model supply the verdict.

For a strictly positive-definite kernel, such as the RBF or Matérn kernels, the kernel matrix at distinct inputs is nonsingular and the exact null space is trivial. Every response vector is then representable. This does not mean that every response is equally plausible under the prior, and the next subsection makes this precise. Repeated inputs are the exception for every kernel. If $x _ { i } = x _ { j }$ , rows i and $j$ of K coincide, so the vector with entries +1 at $i , - 1$ at j and 0 elsewhere lies in $\mathrm { \ n u l l } ( K )$ . The relation $y _ { i } - y _ { j } = 0$ then tests the noise model alone, which is the classical pure-error check for replicated observations. Inputs that are close but not equal give small eigenvalues in (15) instead of zero ones.

Adding kernels keeps only the relations that both kernels share. For positive-semidefinite $K _ { 1 } , K _ { 2 } \in \mathbb { R } ^ { n \times n } , v ^ { \top } ( K _ { 1 } + K _ { 2 } ) v = 0$ exactly when $v ^ { \top } K _ { 1 } v = v ^ { \top } K _ { 2 } v = 0$ , so null $\left( K _ { 1 } + K _ { 2 } \right) =$ null $( K _ { 1 } ) \cap \mathrm { n u l l } ( K _ { 2 } )$ . This can still leave exact relations even when both kernels are flexible. For an additive kernel $k ( x , x ^ { \prime } ) = k _ { 1 } ( x _ { 1 } , x _ { 1 } ^ { \prime } ) + k _ { 2 } ( x _ { 2 } , x _ { 2 } ^ { \prime } )$ , with each $k _ { j }$ depending on the jth input only, the mixed diference (10) has zero variance under both components. It is therefore an exact relation even when $k _ { 1 }$ and $k _ { 2 }$ are RBF kernels, and an $n _ { \mathrm { 1 } } \times n _ { \mathrm { 2 } }$ grid of inputs carries $( n _ { 1 } - 1 ) ( n _ { 2 } - 1 )$ independent relations of this kind. Experiment E3 uses this.

## 4.2 Full-rank kernels: spectral coordinates

Exact Gale vectors exist only when K is singular. For a full-rank kernel we use instead the spectral coordinates of a residual matrix. They play the same role for lengths and inner products, but they are not an exact Gale transform, and the combinatorial reading of Section 3, through signs and circuits, does not carry over to them.

The posterior mean of $f$ at the training inputs is $K Q y$ . Since $K Q = ( A - \sigma ^ { 2 } I ) Q = I - \sigma ^ { 2 } Q$ the corresponding residual is

$$
e = y - K Q y = S y \in \mathbb { R } ^ { n } , \qquad S = \sigma ^ { 2 } Q \in \mathbb { R } ^ { n \times n } .\tag{14}
$$

We call e the soft residual and $S$ the residual matrix. Let $K = U \Lambda U ^ { \top }$ be an eigendecomposition, with $U \in \mathbb { R } ^ { n \times n }$ orthogonal, columns $u _ { 1 } , \ldots , u _ { n }$ , and $\Lambda = \mathrm { d i a g } ( \lambda _ { 1 } , . . . , \lambda _ { n } ) , \lambda _ { j } \geq 0$ . Then

$$
\boldsymbol { S } = \boldsymbol { U } \mathrm { d i a g } \left( \frac { \sigma ^ { 2 } } { \lambda _ { j } + \sigma ^ { 2 } } \right) \boldsymbol { U } ^ { \top } .\tag{15}
$$

The residual matrix acts by spectral shrinkage: $S y$ projects y onto the eigenvectors of $K$ damps the coordinate along $u _ { j }$ by the factor $\sigma ^ { 2 } / ( \lambda _ { j } + \sigma ^ { 2 } )$ , and maps the result back. The spectrum therefore sorts the directions of response space into three kinds, and we use the following vocabulary throughout.

$\lambda _ { j } = 0$ : a hard structural relation. The prior fixes $u _ { j } ^ { \top } f$ at zero, and all of $u _ { j } ^ { \top } y$ passes to the residual.

$0 < \lambda _ { j } \ll \sigma ^ { 2 } ;$ : a soft prior restriction. The prior lets $u _ { j } ^ { \top } f$ vary, but by much less than the noise, so almost all of $u _ { j } ^ { \top } y$ passes to the residual. This is a statement about plausibility, not an algebraic constraint.

$\lambda _ { j } \gg \sigma ^ { 2 }$ : a prior-flexible direction. The residual keeps only a small fraction of $u _ { j } ^ { \top } y$ because the model readily explains such patterns.

Figure 3 shows that for an RBF kernel the soft prior restrictions oscillate between neighbouring inputs and resemble higher-order finite diferences, but unlike the relations of Section 3 they exclude no response pattern.

Unlike the exact projector $P _ { N }$ , the residual matrix $S$ is generally not idempotent, and $\| S y \| ^ { 2 } / \sigma ^ { 2 }$ does not in general have a chi-square distribution. Since $S$ is symmetric and positive definite, it has the symmetric square root

$$
B = S ^ { 1 / 2 } = U \mathrm { d i a g } \left( \sqrt { \frac { \sigma ^ { 2 } } { \lambda _ { j } + \sigma ^ { 2 } } } \right) U ^ { \top } \in \mathbb { R } ^ { n \times n } , \qquad B B ^ { \top } = S .\tag{16}
$$

Its rows $b _ { i } ^ { \top }$ , with $b _ { i } \in \mathbb { R } ^ { n }$ , are the spectral coordinates of the observations. They play the role of Gale vectors for a general kernel. As for orthonormal Gale vectors, only their lengths $\| b _ { i } \| ^ { 2 } = S _ { i i }$ and inner products $b _ { i } ^ { \top } b _ { j } = S _ { i j }$ matter, since any other factor of S has the form BO with O orthogonal. The length $\dot { \Vert } b _ { i } \Vert ^ { 2 }$ generalises (6): a perturbation of size one at observation i leaves a soft residual of size $S _ { i i }$ at that observation.

Two quadratic quantities must be kept apart. The quadratic term of the marginal likelihood is $y ^ { \top } Q y = y ^ { \top } S y / \sigma ^ { 2 } = \| B ^ { \top } y \| ^ { 2 } / \sigma ^ { 2 }$ , and the group test of Section 5 is built from it. The soft residual energy $\| S y \| ^ { 2 } = y ^ { \top } S ^ { 2 } y$ instead enters the noise estimate of Section 6. The two coincide for an exact projector but not for S.

(a) spectrum of K  
![](images/8ddb436d596939ee8a5e0a8f8306b4550abc0fa5e1855be96cb1dcdc4eec3c55.jpg)

(b) RBF eigenvectors  
![](images/23171cde595789a499b3405880d3292674bfbce803740f79938f9da989257be3.jpg)

(c) a soft relation  
![](images/92a8b6631b1b85894264777bc4dd942bf89852af51896cd2113d017fdc89c5f4.jpg)  
offset from x<sub>i</sub> (inputs)  
Figure 3: Hard structural relations and soft prior restrictions on $n = 4 0$ equally spaced inputs in [−1, 1], with noise variance $\sigma ^ { 2 } = 2 . 5 \times 1 0 ^ { - 3 }$ . (a) Eigenvalues $\lambda _ { j }$ of the kernel matrix for the RBF kernel with lengthscale $\ell = 0 . 3$ and unit output scale, and for the polynomial kernel $( 1 + x x ^ { \prime } ) ^ { 3 }$ . Values below machine precision relative to $\lambda _ { 1 }$ are shown at that level. The polynomial kernel has rank $^ { 4 , }$ so it imposes 36 exact relations. The RBF kernel has full rank, and the 29 eigendirections below $\sigma ^ { 2 }$ are soft prior restrictions. (b) The leading eigenvector $u _ { 1 }$ of the RBF kernel matrix and the first eigenvector $u _ { 1 2 }$ whose eigenvalue lies below $\sigma ^ { 2 }$ . (c) The row of the residual matrix S at the central input, scaled to 1 at that input, next to the straight-line relation $( - \frac { 1 } { 2 } , 1 , - \frac { 1 } { 2 } )$ of Example 3. It compares $y _ { i }$ with a weighted average of its neighbours.

## 4.3 The residual matrix in the small-noise limit

Lemma 1. For a fixed positive-semidefinite K,

$$
\operatorname * { l i m } _ { \sigma ^ { 2 } \downarrow 0 } \sigma ^ { 2 } ( K + \sigma ^ { 2 } I ) ^ { - 1 } = P _ { N } ,\tag{17}
$$

where $P _ { N }$ is the orthogonal projector onto null(K). For any factorisation $K = \Phi \Sigma _ { w } \Phi ^ { \top }$ with positive definite $\Sigma _ { w } { _ { ; } }$ this is the residual projector (4), since null $( K ) = \mathrm { n u l l } ( \Phi ^ { \top } )$ by (12).

Proof. In (15), a zero eigenvalue has weight one for every $\sigma ^ { 2 } > 0$ , while each positive eigenvalue has weight $\sigma ^ { 2 } / ( \lambda _ { j } + \sigma ^ { 2 } ) \to 0$ . Since the matrix is finite dimensional, convergence holds in operator norm. □

Together with (12), this shows that for a finite-rank kernel the residual matrix S converges to the projector $P _ { N } = G G ^ { \top }$ of the underlying feature configuration. This is a fixed-design limit. For full-rank K the limit is zero, even when several eigenvalues are very small. Truncating the spectrum at a positive noise level is therefore an approximation, not an exact Gale transform of a full-rank kernel. Convergence of the matrix also does not make the GP test and the residual test identical, because their degrees of freedom can difer (Section 5.2).

## 4.4 Hard structural relations and soft prior restrictions

A finite-rank GP imposes hard structural relations, exactly as a linear model does. Four names used in this paper refer to this one object seen from diferent sides: the null space of K, the combinations of function values that the prior fixes at zero, the consistency relations of the model at the design, and the residual space null $( \Phi ^ { \top } )$ , whose orthonormal basis G has the Gale vectors as its rows. A full-rank GP imposes only soft prior restrictions. It can fit any data,

but it pays a price in residual energy for patterns that its prior regards as implausible. Table 1 lists the corresponding objects. We write Gale vectors and structural only for the exact case, and spectral coordinates and soft for the general case.
<table><tr><td></td><td>Finite-rank kernel (exact)</td><td>General kernel (spectral)</td></tr><tr><td>Coordinates of observa- Gale vector gi, row of G tion i</td><td></td><td>spectral coordinates  $b _ { i } ,$  row of  $B =$   $\bar { S } ^ { 1 / 2 }$ </td></tr><tr><td>Residual operator</td><td>projector  $P _ { N } = G G ^ { \top }$ </td><td>residual matrix  $S = B B ^ { \top } = \sigma ^ { 2 } Q$ </td></tr><tr><td>Restriction</td><td>hard structural relation  $v ^ { \top } f = 0$  for  $v \in { \mathrm { n u l l } } ( K )$ </td><td>soft prior restriction,  $\mathrm { V a r } ( u _ { j } ^ { \top } f ) =$   $\lambda _ { j } \ll \sigma ^ { 2 }$ </td></tr><tr><td>A departure along it is</td><td>a structural contradiction,  $\eta \propto$   $1 / \sigma _ { 0 } ^ { 2 }$ </td><td>improbable under the prior,  $\eta $   $( u _ { j } ^ { \top } \delta ) ^ { 2 } / \lambda _ { j }$  as  $\sigma _ { 0 }  0$ </td></tr><tr><td>Visible fraction at i</td><td> $\| g _ { i } \| ^ { 2 } = 1 - h _ { i i }$ </td><td> $\| \bar { b _ { i } } \| ^ { 2 } = S _ { i i }$ </td></tr><tr><td>Minimal witnesses</td><td> $\mathrm { c i r c u i t s ~ ( P r o p o s i t i o n ~ 1 ) }$ </td><td>no direct combinatorial analogue</td></tr><tr><td>Small-noise limit</td><td> $S \to P _ { N }$ </td><td> $S \to 0$ </td></tr></table>

Table 1: Dictionary between the exact finite-rank construction and its spectral counterpart for general kernels. Here η is the noncentrality of Section 5 and $\sigma _ { 0 } ^ { 2 }$ the noise variance.

## 5 Group consistency tests and their power

A departure can be detectable for two diferent reasons. It may violate an exact relation of the design, in which case its signal-to-noise ratio can grow without bound as the measurements become more precise. Or it may remain representable by the model and be rejected only because the prior assigns it little probability. The distinction is visible directly in the noncentrality of the test.

Before any test is chosen, the precision matrix $Q$ already describes how sensitive a design is to every departure from the model. Suppose that the responses contain a deterministic discrepancy $\delta \in \mathbb { R } ^ { n }$ in addition to the GP and the noise, $y = f + \delta + \varepsilon$ with $\varepsilon \sim \mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I )$ Section 5.2 shows that the test of all observations then has noncentrality

$$
\eta = \delta ^ { \top } Q \delta = \sum _ { j = 1 } ^ { n } \frac { ( u _ { j } ^ { \top } \delta ) ^ { 2 } } { \lambda _ { j } + \sigma _ { 0 } ^ { 2 } } .\tag{18}
$$

Here η is the noncentrality parameter of the chi-square test: larger values mean that the alternative is more strongly separated from the null and is therefore easier to detect. The matrix Q and the eigenpairs $( \lambda _ { j } , u _ { j } )$ of K are those of Section 4, with $\sigma ^ { 2 } = \sigma _ { 0 } ^ { 2 } .$

The spectrum of K shows where this detectability comes from (Figure 4). Along an eigenvector in null(K), the contribution to η is $( u _ { j } ^ { \top } \delta ) ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ . This part of the departure violates an exact consistency relation, so its detectability grows without bound as the noise decreases. We call this structural detection. Along an eigenvector with $\lambda _ { j } \gg \sigma _ { 0 } ^ { 2 } .$ , the contribution is bounded by $( u _ { j } ^ { \top } \delta ) ^ { 2 } / \lambda _ { j }$ as the noise decreases. Such a departure remains representable by the model and can only be judged improbable under the prior, so we call this prior-based detection.

A soft prior restriction, with $0 < \lambda _ { j } \ll \sigma _ { 0 } ^ { 2 }$ , initially behaves like a structural direction because its contribution is approximately $( u _ { i } ^ { \top } \delta ) ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ while the noise dominates $\lambda _ { j }$ . As $\sigma _ { 0 } ^ { 2 }  0$ however, the contribution approaches $( u _ { j } ^ { \top } \delta ) ^ { 2 } / \lambda _ { j }$ and therefore remains bounded. We reserve the word structural for directions in null(K).

At the end-point design of Figure 1(a), curvature lies entirely in the prior-based part. The GP can therefore reject a suficiently large curvature even though no consistency relation involving curvature is violated (Figure 4(b)). We next express this distinction for tests of groups of observations and for families of departures.

(a) two parts of a departure  
![](images/0c20af3aad2782948da63e43647eb1ae3f56ea36f07465c34bbce4a1af26c2aa.jpg)

(b) straight line against curvature  
![](images/3174fd10862b57490c534da283771c1a0deccc907a4805d96887198a65f69647.jpg)  
Figure 4: Structural and prior-based detection for a finite-rank kernel. (a) A departure δ splits into a part that the model can represent, in col(Φ), and a part that violates consistency relations, in $\mathrm { { n u l l } } ( \Phi ^ { \top } )$ . The first can be detected only through the prior, with a noncentrality that stays bounded as the noise decreases. The second is detected structurally, with a noncentrality proportional to $1 / \sigma _ { 0 } ^ { 2 }$ . (b) Noncentrality (18) of the test of all 40 observations against a unit curvature $\delta ( x ) = x ^ { 2 }$ , under the straight-line kernel $1 + x x ^ { \prime }$ , as a function of the noise standard deviation $\sigma _ { 0 } .$ , for the two designs of Figure 1(a). At the end-point design the curvature is an intercept shift, and its noncentrality stays close to 1. With four inputs at the centre it grows as $1 / \sigma _ { 0 } ^ { 2 }$

## 5.1 The group test

The group test asks a simple question. Can the responses of a group of observations be predicted from all the other responses, within the uncertainty of that prediction? If not, the group is incompatible with the model.

To make this precise, split the observation indices $\{ 1 , \ldots , n \}$ into a set C of candidate observations, which we test, and the set R of the remaining, retained observations. A subscript selects entries: $y _ { C } \in \mathbb { R } ^ { | C | }$ contains the responses $y _ { i }$ with $i \in C ,$ , and $A _ { R C } \in \mathbb { R } ^ { | R | \times | C | }$ is the block of A with rows in R and columns in $C ,$ so that, after reordering,

$$
A = \left( \begin{array} { l l } { { A _ { R R } } } & { { A _ { R C } } } \\ { { A _ { C R } } } & { { A _ { C C } } } \end{array} \right) , \qquad A _ { C R } = A _ { R C } ^ { \top } .\tag{19}
$$

For a fixed noise variance $\sigma ^ { 2 }$ , Gaussian conditioning (Appendix A) gives the prediction of $y _ { C }$ from $y _ { R }$ and its uncertainty,

$$
y _ { C } \mid y _ { R } \sim { \mathcal { N } } ( A _ { C R } A _ { R R } ^ { - 1 } y _ { R } , V _ { C } ) , \qquad V _ { C } = Q _ { C C } ^ { - 1 } .\tag{20}
$$

The prediction error $z _ { C } = y _ { C } - A _ { C R } A _ { R R } ^ { - 1 } y _ { R } \in \mathbb { R } ^ { | C | }$ is the cross-validation residual of the group [27, 58, 32]. The group test statistic measures its size against its covariance,

$$
T _ { C } = z _ { C } ^ { \top } V _ { C } ^ { - 1 } z _ { C } .\tag{21}
$$

In words, $T _ { C }$ measures how surprising the responses in C are, given all other responses, after accounting for the uncertainty of the prediction and for the correlations within the group. Appendix A shows two equivalent forms. With the representer weights $\alpha = Q y$ of the posterior mean [54], $T _ { C } = \alpha _ { C } ^ { \top } Q _ { C C } ^ { - 1 } \alpha _ { C } = e _ { C } ^ { \top } S _ { C C } ^ { - 1 } e _ { C } / \sigma ^ { 2 }$ . The last form uses only the soft residual and the residual matrix, so a test computed in spectral coordinates is exactly the classical multiple-case test, and changing coordinates cannot change which groups are rejected.

Two diferent null hypotheses can be tested, and they must be kept apart. The structural null of a linear model states only that the noiseless responses lie in col(Φ), with fixed but unknown coeficients. Its residual test $\| G ^ { \top } y \| ^ { 2 } / \sigma ^ { 2 } \sim \chi _ { n - r } ^ { 2 }$ uses the consistency relations alone. The GP null defined next also fixes a prior on the coeficients or the function, the mean and the noise. A rejection of the GP null can therefore come from the prior, the mean or the noise specification, and it does not by itself show that a structural assumption, such as the degree of a polynomial, is violated.

We call the model fully specified when its mean function, its kernel including all hyperparameters, and its noise variance are fixed in advance rather than estimated from the data under test. The GP null is the null hypothesis that the responses were generated by this model, that is, $y \sim { \mathcal { N } } ( 0 , A )$ with the specified kernel and noise variance. Under the fully specified GP null,

$$
T _ { C } \mid y _ { R } \sim \chi _ { | C | } ^ { 2 } ,\tag{22}
$$

exactly, for a fixed candidate group. Hyperparameters estimated from the same data generally destroy this exact calibration [6], and Section 6 shows that a fitted noise variance can absorb the very discrepancy under test, so known measurement noise, replicates or a separate reference set are preferable.

## 5.2 Power of the group test

The power of a test is the probability that it rejects the model when the model is wrong. For chi-square tests it is governed by a single number, the noncentrality. Proposition 2 is a classical consequence of Gaussian conditioning and of the distribution of quadratic forms in normal variables, stated in the form needed here. Its proof is in Appendix B.

Proposition 2. Let $y = f + \delta + \varepsilon$ , with $f \sim \mathcal { G P } ( 0 , k ) , \varepsilon \sim \mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I )$ , and a deterministic discrepancy $\delta \in \mathbb { R } ^ { n }$ , and compute $T _ { C }$ in (21) with the same kernel k and the true noise variance $\sigma ^ { 2 } = \sigma _ { 0 } ^ { 2 }$ . Then, conditionally on y<sub>R</sub>, and hence also unconditionally, $T _ { C }$ has a noncentral chi-square distribution with |C| degrees of freedom and noncentrality

$$
\eta _ { C } = \Delta _ { C } ^ { \top } Q _ { C C } \Delta _ { C } = ( Q \delta ) _ { C } ^ { \top } Q _ { C C } ^ { - 1 } ( Q \delta ) _ { C } ,\tag{23}
$$

where $\Delta _ { C } = \delta _ { C } - A _ { C R } A _ { R R } ^ { - 1 } \delta _ { R } \in \mathbb { R } ^ { | C | }$ is the part of the discrepancy on C that the remaining observations do not predict. $I f \delta$ is zero outside C, then $\Delta _ { C } = \delta _ { C }$ and $\eta _ { C } = \delta _ { C } ^ { \top } Q _ { C C } \delta _ { C }$

In words, the test is more likely to reject the larger the part of the discrepancy that the other observations cannot predict, measured against the uncertainty of that prediction. For $C = \{ 1 , \dots , n \} , \Delta _ { C } = \delta$ and (23) reduces to (18). The power at significance level $\beta$ is $\mathrm { P r } ( T _ { C } > \chi _ { | C | , 1 - \beta } ^ { 2 } )$ under the noncentral chi-square distribution, where $\chi _ { k , q } ^ { 2 }$ denotes the q-quantile of the chi-square distribution with k degrees of freedom. It holds for every kernel and every noise level, and it depends only on the kernel, the noise variance, the design and the discrepancy, so it can be computed before any response is observed.

## 5.3 Geometric meaning

Proposition 2 has a geometric form (Appendix B),

$$
\eta _ { C } = \frac { \| \Pi _ { C } B ^ { \top } \delta \| ^ { 2 } } { \sigma _ { 0 } ^ { 2 } } , \qquad \mathrm { a n d } \qquad \eta _ { C } = \frac { \| B _ { C } ^ { \top } \delta _ { C } \| ^ { 2 } } { \sigma _ { 0 } ^ { 2 } } \quad \mathrm { i f ~ } \delta \mathrm { ~ i s ~ z e r o ~ o u t s i d e ~ } C ,\tag{24}
$$

where $B _ { C } \in \mathbb { R } ^ { | C | \times n }$ contains the spectral coordinates $b _ { i } , ~ i ~ \in ~ C$ , of (16) as its rows, and $\Pi _ { C } \in \mathbb { R } ^ { n \times n }$ is the orthogonal projector onto their span. Each observation has a vector $b _ { i } .$ , and the discrepancy is represented by the weighted sum

$$
B ^ { \top } \delta = \sum _ { i } \delta _ { i } b _ { i } \in \mathbb { R } ^ { n } .
$$

The noncentrality, and hence the power, is determined by how much of this discrepancy vector lies in the span of the spectral coordinates of the tested observations, relative to the noise level. Short spectral coordinates make local departures less visible, while nearly aligned coordinates provide similar rather than independent directions of detection.

For finite-rank kernels the picture becomes exact. For a discrepancy supported on C and a fixed finite-rank kernel, Lemma 1 gives $\sigma _ { 0 } ^ { 2 } Q _ { C C }  ( P _ { N } ) _ { C C } = G _ { C } G _ { C } ^ { \top }$ as $\sigma _ { 0 } ^ { 2 } \to 0$ , where $G _ { C } \in \mathbb { R } ^ { | C | \times ( n - r ) }$ contains the Gale vectors of the observations in C as its rows. Hence

$$
\operatorname* { l i m } _ { \sigma _ { 0 } ^ { 2 } \to 0 } \sigma _ { 0 } ^ { 2 } \eta _ { C } = \| G _ { C } ^ { \top } \delta _ { C } \| ^ { 2 } ,\tag{25}
$$

so that $\eta _ { C } \approx \| G _ { C } ^ { \top } \delta _ { C } \| ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ for small noise whenever $G _ { C } ^ { \top } \delta _ { C } \neq 0$ . This is the noncentrality of the residual test $( P _ { N } y ) _ { C } ^ { \top } ( P _ { N } ) _ { C C } ^ { + } ( P _ { N } y ) _ { C } / \sigma _ { 0 } ^ { 2 }$ in the linear model (2). The two tests still difer. The GP test has |C| degrees of freedom and the residual test rank $\left( ( P _ { N } ) _ { C C } \right)$ , so their thresholds and their power difer even when their noncentralities agree. In both cases the power depends on the design only through the lengths of, and the angles between, the vectors of the afected observations. These are Gale vectors for a finite-rank kernel and spectral coordinates otherwise. Three consequences follow. We state them for exact Gale vectors and the residual tests of the linear model.

1. A departure that the model can represent violates no relation. At the end-point design of Figure 1(a), the curvature $c x ^ { 2 }$ takes the same value c at both inputs, so it equals an intercept shift, and no residual test can see it, however large c is. The GP test can still reject, because a large intercept is improbable under the prior on the coeficients. This prior-based power grows with c but not as the noise decreases, and it says nothing about curvature (experiment E1).

2. Ofsets at observations with high leverage are hard to detect. Such observations have short Gale vectors. In Example 2, an ofset at a corner keeps 30% of its squared size in the tests, and an ofset at the centre 80%.

3. Observations with identical Gale vectors cannot be told apart. In Example 2, ofsets c at corners 1 and 3 produce the same failures of the relations as an ofset 2c at corner 1.

## 5.4 Structured departures

A structural hypothesis is usually tested against a family of departures rather than a single one, such as all curvatures or all interactions. Let the departures of interest be $\delta = D \theta$ for a matrix $D \in \mathbb { R } ^ { n \times p }$ , whose columns are p basis directions such as a curvature or an interaction, and an unknown $\theta \in \mathbb { R } ^ { p }$

Proposition 3. Under the assumptions of Proposition 2, the noncentrality of $T _ { C }$ against δ = Dθ is the quadratic form $\eta _ { C } ( \theta ) = \theta ^ { \top } M _ { C } \theta$ with $M _ { C } = D ^ { \top } Q _ { : C } Q _ { C C } ^ { - 1 } Q _ { C : } D \in \mathbb { R } ^ { p \times p }$ , where $Q { : } C$ denotes the columns of Q indexed by C. The least detectable unit departure has noncentrality min $\lvert \lvert \theta \rvert \rvert { = } 1 \eta _ { C } ( \theta ) = \lambda _ { \mathrm { m i n } } ( M _ { C } )$ , and for $\theta \sim \mathcal { N } ( 0 , \tau ^ { 2 } I _ { p } )$ the mean noncentrality is E $\eta _ { C } = \tau ^ { 2 } \mathrm { t r } { \cal M _ { C } }$ For the test of all observations, $C = \{ 1 , \ldots , n \} , M _ { C } = D ^ { \top } Q D$ , and for a fixed finite-rank kernel

$$
\operatorname* { l i m } _ { \sigma _ { 0 } ^ { 2 } \to 0 } \sigma _ { 0 } ^ { 2 } M _ { C } = ( G ^ { \top } D ) ^ { \top } ( G ^ { \top } D ) .\tag{26}
$$

The departures $\theta \in \operatorname { n u l l } ( G ^ { \top } D )$ violate no consistency relation of the design and are invisible to every residual test. The family is structurally testable in every direction exactly when $G ^ { \top } D$ has full column rank $p _ { : }$ , that is, when no non-zero departure in the family is representable by the model at the design.

In words, for a whole family of departures one small matrix says which members a design can detect structurally and which only through the prior. Both quantities depend on how the columns of D are scaled, so they are comparable only under a stated amplitude convention, and we take the columns of $D$ to be linearly independent. The mean noncentrality is not the average power. Average power averages the noncentral chi-square tail probability over the departures, as in experiment E2, and need not rank designs in the same order. Both are nevertheless computable from the design before any response is observed. Structural testability is characterised by the rank of $G ^ { \top } D$ . The GP test still gives every non-zero departure a positive noncentrality, since Q is positive definite, but for $\theta \in \operatorname { n u l l } ( G ^ { \top } D )$ this is the prior-based part of Section 5: it does not grow as the noise decreases, as at the end-point design.

The columns of $G ^ { \top } D$ are the Gale images $\sum _ { i } D _ { i j } g _ { i }$ of the basis directions of the family, so the Gale configuration shows which structured departures a design can detect structurally. It also defines a minimal structurally informative design, the smallest design for which $G ^ { \top } D$ has full column rank. For a polynomial of degree q against a departure of degree $q + 1$ , this is any design with $q + 2$ distinct inputs (Example 3). Failure to reject may consequently reflect an untestable discrepancy, insuficient power, or genuine consistency. Propositions 2 and 3 do not decide between these from the data, but they identify in advance which discrepancies no consistency relation of the design can reveal and which it can detect only weakly. Because adding an observation changes the configuration, (23) also gives the power that the test would have after a candidate new observation, before that observation is made. Experiments E1 to E3 use this to choose where to observe.

## 5.5 Relation to discrimination design and Fisher information

Choosing observations to maximise the power of a lack-of-fit test is a classical idea. Designs that detect the inadequacy of a regression model were constructed in [2, 40], building on the treatment of bias from omitted terms in [12]. The best-known criterion is T-optimality [3]. One of two rival regression models is assumed to be true, with known parameters, and the design maximises the sum of squared deviations between the true model and the best fit of the rival model. When the rival is the linear model (2) and the true model difers from it by $\delta ,$ this sum is min $_ { 1 w } \| \delta - \Phi w \| ^ { 2 } = \| P _ { N } \delta \| ^ { 2 } = \| G ^ { \top } \delta \| ^ { 2 }$ , which is $\sigma _ { 0 } ^ { 2 }$ times the noncentrality of the residual test, the limit in (25). A T-optimal design therefore maximises the length of the Gale image of the departure. In plain terms, T-optimality asks whether the specified departure can still be mimicked by the null model on the chosen design. If the minimum is zero, the design cannot distinguish the two models structurally, and if it is large, the departure necessarily leaves a large residual. At the end-point design of Figure $1 ( \mathrm { a } ) , x ^ { 2 }$ cannot be told apart from an intercept shift, so the T-optimality criterion is zero, and adding centre points makes the curvature leave a non-zero residual. Like experiment E1, it requires the departure to be specified in advance. Properties of such designs are studied in [25], extensions to non-normal models in [44], and the general theory in [5]. Sequential and Bayesian approaches choose experiments to discriminate between mechanistic models [13] or, with GP models and GP surrogates, between kernels and between mechanistic models [29, 48].

Two further classical quantities coincide with these noncentralities. For a fixed discrepancy, the conditional distributions of y<sub>C</sub> given $y _ { R }$ under the alternative and under the null are Gaussian with the same covariance $V _ { C }$ and means that difer by $\Delta _ { C } .$ , so η<sub>C</sub> is twice their Kullback–Leibler divergence. For the family $\delta = D \theta$ and the test of all observations, the responses are distributed as $\mathcal { N } ( D \theta , A )$ , whose Fisher information about θ is $D ^ { \top } Q D = M _ { C }$ The worst-case criterion of Proposition 3 is therefore E-optimality for the departure parameters [5], and the structural part of this information is the part that grows as the noise decreases.

For finite-rank models and a specified alternative, maximising the structural noncentrality is therefore classical discrimination design. It is T-optimality, and for Gaussian models with a common covariance it is also maximisation of the projected Kullback–Leibler divergence [44], $\begin{array} { r } { \operatorname* { m i n } _ { \boldsymbol { w } } \| \boldsymbol { \delta } - \boldsymbol { \Phi } \boldsymbol { w } \| ^ { 2 } / 2 \sigma _ { 0 } ^ { 2 } = \| \boldsymbol { G } ^ { \intercal } \boldsymbol { \delta } \| ^ { 2 } / 2 \sigma _ { 0 } ^ { 2 } } \end{array}$ . The rival model is fitted by its best coeficients in this quantity. The full-GP divergence $\frac { 1 } { 2 } \delta ^ { \top } Q \delta$ is diferent because the GP prior also penalises coeficients within the representable subspace.

This diference is the point of the decomposition used here. It records whether the noncentrality comes from a component outside the model’s representable space or from a component that remains representable but is improbable under the prior. For general kernels the same distinction is read spectrally, while in the finite-rank case the circuits identify the smallest sets of observations on which an exact contradiction can occur. At the end-point design of experiment E1, the T-optimality criterion and the projected divergence are zero for curvature because $x ^ { 2 }$ is representable there as an intercept shift. The GP test and the full-GP divergence can nevertheless be large because that intercept is improbable under the prior.

## 6 Fitted noise as aggregate test failure

The calibration above assumes that the noise variance is known. When it is estimated from the same observations that are being tested, discrepancy energy can instead be absorbed into the fitted noise level. For fixed K, let $\hat { \sigma } ^ { 2 }$ maximise the log marginal likelihood, and write $\hat { Q } = Q ( \hat { \sigma } ^ { 2 } ) , \hat { S } = S ( \hat { \sigma } ^ { 2 } )$ and $\hat { e } = \hat { S } y$ . At any positive interior maximum (Appendix C),

$$
\hat { \sigma } ^ { 2 } = \frac { \lVert \hat { e } \rVert ^ { 2 } } { \operatorname { t r } \hat { S } } = \frac { \lVert \hat { e } \rVert ^ { 2 } } { n - \operatorname { t r } ( K \hat { Q } ) } .\tag{27}
$$

In words, the fitted noise variance is the leftover misfit divided by the efective number of directions that the model cannot fit. The numerator is the total soft residual energy, that is, the aggregate failure of the hard relations and the soft prior restrictions. The denominator is an efective number of residual degrees of freedom, n minus the efective number of parameters $\mathrm { t r } ( K \hat { Q } )$ of a linear smoother [36, 61]. Equation (27) is the GP form of MacKay’s evidence fixed point [45]. For a finite-rank kernel it approaches the classical estimator $\| P _ { N } y \| ^ { 2 } / ( n - r )$ when $\hat { \sigma } ^ { 2 }$ is small compared with the non-zero eigenvalues of K. A discrepancy with energy along hard relations or soft prior restrictions passes almost unattenuated into the numerator and leaves the denominator nearly unchanged, so it can inflate $\hat { \sigma } ^ { 2 }$ (Appendix C). The inflated value then enters the predictive variance at every input, and it also shrinks the statistic of the test of all observations, and typically the group statistics, so fitting the noise to the data under test can switch the checks of. In experiment S1 (Appendix F), three shifted observations among 40 raise the fitted noise variance to almost four times its true value (Table 4).

Tests computed with a fitted noise variance therefore tend to reject less often when the model is wrong, and a non-rejection is weaker evidence than it appears. The consequences for deleting data are examined in Appendix E.

## 7 Experiments

Three experiments test the claims of Section 5 on controlled problems with known discrepancies. E1 asks whether a test can reject a model without testing the assumption in question, and whether the observation that best tests a GP is the one of largest predictive variance. E2 asks whether prospective power still helps when the location of the discrepancy is unknown. E3 asks which two-dimensional designs can test additivity. In all three the null model is fully specified, so Proposition 2 applies exactly, and predicted powers are checked against rejection rates in 4000 simulated data sets. All tests have level 0.05. Full settings are given in Appendix D, and a simulation study of detection and forgetting is in Appendicex F.

## 7.1 Structural contradiction or prior improbability (E1)

Question. Can a test reject a model without testing the assumption in question? Set-up. The null model is the straight-line GP with kernel $1 + x x ^ { \prime }$ of Example 5, whose prior $w \sim \mathcal { N } ( 0 , I )$ on the intercept and slope is implied by the kernel, with noise variance $\sigma _ { 0 } ^ { 2 } = 0 . 0 0 2 5$ . The alternative adds a curvature $c x ^ { 2 }$ . We compare the GP test of all 40 observations, which tests the GP null, with the exact residual test $\| G ^ { \top } y \| ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ , which tests the structural null and does not depend on the prior. The designs are 20 inputs at each end of [−1, 1], 18 at each end with 4 at the centre, and 40 equally spaced inputs.

Results. At the end-point design, the 38 consistency relations compare replicates only, and the residual test has power equal to its level at every curvature, up to $c = 1 0$ (Figure 5). The GP test does reject eventually, with power 0.74 at $c = 5$ and essentially 1 at $c = 1 0$ , which is 200 noise standard deviations. It rejects because a curvature at two inputs is an intercept shift that the prior finds improbable, a rejection that is valid for the GP null but says nothing about curvature. Over the range $c \leq 0 . 2 5$ its power is at most 0.051. With four inputs at the centre, the design contains circuits such as $y _ { i } + y _ { j } - 2 y _ { k } = 0$ , with one observation at each of −1, 1 and 0, and both tests detect the curvature through them, with power 0.69 and 0.68 at $c = 0 . 1 2 5$ . Equally spaced inputs give 0.75 and 0.73. Adding centre points to a two-level design in order to test for curvature is standard practice in response-surface methodology [15], so the structural half of this result is classical. What the GP adds is the other half: the GP test does reject the end-point design for a large curvature, and the decomposition of Section 5 shows that this rejection comes from the prior, not from a violated relation. Conclusion. A rejection by the GP test does not show that the suspected assumption is wrong, and the design that is optimal for estimating a straight line contains no relation that tests whether the line is straight.

![](images/223a51ee7cc9d4f95952d2ee831e8a9f8c1b49b084b9ed91764bfc863db67acf.jpg)

![](images/c43b1d566fa7a2296d4b08a012df8c02359a33d5ed0d3f013cf857af7d719dd3.jpg)  
Figure 5: E1, testing a straight-line assumption against a curvature $c x ^ { 2 }$ . Solid lines show the power of the GP test predicted by Proposition 2, dashed lines the power of the exact residual test, and markers the rejection rates in 4000 simulated data sets. The dotted line marks the level 0.05. (a) Three designs of 40 inputs. (b) The end-point design for large curvatures. The residual test stays at the level for every curvature, because no consistency relation of the end-point design involves curvature. The GP test eventually detects a very large curvature, as an intercept shift that is improbable under the prior.

## 7.2 Choosing for power, for discrimination or for predictive variance (E1, continued)

Question. Is the observation that best tests a GP the one of largest predictive variance, and how does a classical discrimination criterion compare? Set-up. The null model is a GP with an RBF kernel of lengthscale 0.3 and unit output scale, zero mean and noise variance $\sigma _ { 0 } ^ { 2 } = 0 . 0 0 2 5$ on designs of 40 inputs in [−1, 1], of which 36 are drawn uniformly and 4 lie in a cluster at 0.2. The alternative is a narrow bump $\delta ( x ) = 0 . 2 \exp ( - ( x - x _ { 0 } ) ^ { 2 } / 2 w ^ { 2 } )$ of width $w = 0 . 0 4$ at a specified location $x _ { 0 }$ , tested with the group C of observations within 0.1 of $x _ { 0 }$ . Each of 401 candidate inputs is scored in three ways:

• the power rule: the power the test would have after adding the candidate, from (23);

• the KL rule: the full-GP Kullback–Leibler divergence $\frac { 1 } { 2 } \delta ^ { \top } Q \delta$ between alternative and null after adding it;

• the variance rule: its current predictive variance.

We compare the choices over 20 designs and 19 bump locations, first with the true hyperparameters, and then with the lengthscale or the noise standard deviation wrong by a factor of 2 when choosing.

Results. Figure 6(b) first verifies the power formula against common ofsets at four kinds of group. The simulated rejection rates agree with the prediction to within 0.022 in every case, and an ofset of four noise standard deviations is detected with probability 0.65 at the boundary observation and 0.95 at an interior one, because a GP absorbs an ofset at the boundary more easily $( S _ { i i } = 0 . 3 4$ against 0.79). For the bump at $x _ { 0 } = 0 . 1$ (Figure 1(b)), the current design has power 0.11. Adding the observation of largest predictive variance, at the boundary, leaves the power at 0.11. Adding the observation of largest predicted power, at $x = 0 . 1$ , raises it to 0.64, although the predictive variance there is fourteen times smaller. Table 2 shows that this is typical. Averaged over designs and locations, the power rule raises the power from 0.48 to 0.72, whereas the variance rule gives 0.51 and a random candidate 0.49. The KL rule gives 0.72 as well and chooses the same input as the power rule in 77% of the cases. The two are close because both reward noncentrality against the specified bump. They difer because the power rule scores the conditional test of the group near $x _ { 0 }$ , whose membership and degrees of freedom change with the candidate, whereas the KL rule scores all observations of the augmented design. When the hyperparameters assumed for choosing are wrong, the power rule still gives 0.67 to 0.72 under the true model, and its paired advantage over the variance rule stays between 0.16 and 0.22. Here only the choice uses the wrong hyperparameters. A test that used them as well would no longer be exactly calibrated (Section 8). Conclusion. An observation that reduces predictive uncertainty is, in general, not the observation that tests the model. The prospective-power choice agrees with classical discrimination where it should, and it degrades gracefully when the lengthscale or noise level are misspecified by a factor of two.

## 7.3 Discrepancies of unknown location (E2)

Question. Does prospective power still help when the location of the discrepancy is unknown? Set-up. The null model and the 20 starting designs are those of E1. The alternative is a family of bumps of height 0.2 and width 0.04 whose centre lies in [0.3, 0.9], with prior weights from a normal density centred at 0.6, and the test is the group test of all observations in that region. Five observations are added one at a time from 61 candidates in the region, either by maximising the prior-weighted mean of the rejection probabilities over the family (average power) or their minimum (worst case), or by one of five baselines: the prior-weighted mean Kullback–Leibler divergence (KL), largest predictive variance, largest prior-weighted squared bump height (amplitude), equally spaced inputs (space-filling) and random inputs. The additions do not depend on any response, since the power depends only on the design. Averaging the power keeps the criterion tied to the test. A Bayesian design would instead maximise an expected information gain [17].

![](images/45921e6a09ddc0c96f532c159c67c656fee43459d7ce9adecadd23c45ee24680.jpg)

(b) same ofset, diferent locations  
![](images/15de9ec2b9f1a96f786c20642010d6ec1410de1b2b4fb08eb0ace4c6ba9902cb.jpg)  
Figure 6: E1, design-dependent power. (a) One data set simulated from the GP null on the design used in (b) and in Figure 1(b), with the latent function (solid), the latent function plus the bump at $x _ { 0 } = 0 . 1$ (dashed), and the observations without discrepancy (dots). Circles mark the boundary observation, an interior observation and the cluster. The arrow shows the ofset $c = 0 . 2$ , four noise standard deviations, added at the boundary observation. (b) Verification: power of the group test against a common ofset c at four kinds of group in this design. Lines show the power predicted by Proposition 2, markers the rejection rate in 4000 simulated data sets. The dotted line marks the level 0.05. The candidate scores for the bump at $x _ { 0 } = 0 . 1$ are shown in Figure 1(b).

<table><tr><td>Assumed when choosing</td><td>Power rule ↑</td><td>KL rule ↑</td><td>Variance rule ↑</td><td>Power — variance ↑</td></tr><tr><td>True hyperparameters</td><td>0.72</td><td>0.72</td><td>0.51</td><td> $0 . 2 1 7 \pm 0 . 0 0 8$ </td></tr><tr><td>Lengthscale  $\times 0 . 5$ </td><td>0.67</td><td>0.67</td><td>0.51</td><td> $0 . 1 5 6 \pm 0 . 0 0 8$ </td></tr><tr><td>Lengthscale  $\times 2$ </td><td>0.71</td><td>0.70</td><td>0.50</td><td> $0 . 2 0 8 \pm 0 . 0 0 8$ </td></tr><tr><td>Noise s.d.  $\times 0 . 5$ </td><td>0.72</td><td>0.71</td><td>0.50</td><td> $0 . 2 1 7 \pm 0 . 0 0 8$ </td></tr><tr><td>Noise s.d. ×2</td><td>0.72</td><td>0.72</td><td>0.50</td><td> $0 . 2 1 8 \pm 0 . 0 0 8$ </td></tr></table>

Table 2: E1, power of the group test against a bump of specified location after one added observation, averaged over 20 designs and 19 bump locations. The rules choose under the hyperparameters in the first column, and every choice is evaluated under the true model (lengthscale 0.3, noise variance 0.0025). The current design has mean power 0.48. The KL rule maximises the Kullback–Leibler divergence between the alternative and the null for all observations, and chooses the same input as the power rule in $7 7 \%$ of the cases with the true hyperparameters. The last column is the paired diference between the power and variance rules, averaged per design, with one standard error over the 20 designs.

Results. After five observations the average-power rule reaches a mean power of 0.52, the best of the six rules in each of the 20 designs (Figure 7(a)). The strongest baseline is space-filling, at 0.46, and the average-power rule exceeds it by $0 . 0 5 9 \pm 0 . 0 0 5$ , paired over designs (one standard error). The KL rule reaches 0.520, against 0.523, so the two criteria choose nearly the same designs.

Two limits remain. The power against the least detectable bump stays at most 0.16 for every rule (Figure 7(b)), and Proposition 2 shows this before any observation is made. When the true bump difers from the assumed one (Table 3), the average-power rule stays the best, but its gain over space-filling falls from 0.03–0.06 for narrower or higher bumps to 0.01–0.02 for wider or lower ones. Conclusion. Prospective power remains useful within a specified family of alternatives, it agrees with a classical discrimination criterion, and its advantage shrinks as the alternative is misspecified.

![](images/c60885017587fb8ea7cf7fe722c3c37f652ed8e203a5743988794eeeba876c5e.jpg)

(b) worst location  
![](images/6e81adef7177d03f189eee66cb060019af13d6953375ba79ed93a6bfd2dbdabe.jpg)  
Figure 7: E2, choosing five observations against a bump of unknown location in [0.3, 0.9], averaged over 20 designs. (a) Prior-weighted mean power over the family of bump locations. (b) Power against the least detectable bump. The average-power and worst-case rules use Proposition 2. The other rules are baselines described in the text. The KL rule is not shown, since its curve is indistinguishable from that of the average-power rule.

## 7.4 Testing additivity in two dimensions (E3)

Question. Which two-dimensional designs can test additivity? Set-up. The null model is an additive GP on $[ - 1 , 1 ] ^ { 2 }$ , with kernel $k ( x , x ^ { \prime } ) = k _ { 1 } ( x _ { 1 } , x _ { 1 } ^ { \prime } ) + k _ { 2 } ( x _ { 2 } , x _ { 2 } ^ { \prime } )$ , where $k _ { 1 }$ and $k _ { 2 }$ are RBF kernels with lengthscale 0.5 and unit output scale, and noise variance $\sigma _ { 0 } ^ { 2 } = 0 . 0 0 2 5$ Although both components have full rank, their sum imposes exact relations (Section 4), so this is a GP with genuine exact structure that is not a polynomial regression. The alternative is an interaction c x x . We compare designs of 16 inputs: a $4 \times 4$ grid, Latin hypercube samples and uniformly random inputs (50 draws each), and a design chosen greedily from a $2 1 \times 2 1$ candidate grid by the noncentrality of the GP test against $x _ { 1 } x _ { 2 }$

Results. The grid carries the $( 4 - 1 ) ( 4 - 1 ) = 9$ mixed-diference relations (10). They are spanned by the mixed diferences of its $2 \times 2$ subgrids, and each of these is a circuit of four observations: any assignment of values on three of the four inputs can be matched by an additive function, whereas the values on all four must satisfy the mixed-diference relation. The GP test of the grid detects $c = 0 . 1$ with probability 0.81, all of it structurally (Figure 8). A Latin hypercube has no two inputs that share a coordinate, so for every set C of its inputs $K _ { C C }$ is non-singular and no set supports a circuit. The Latin hypercube and random designs detect the same interaction with probability 0.09 and 0.08 on average, through the prior and soft prior restrictions only. The power-optimised design places its inputs on the two lines $x _ { 2 } = \pm 1$ , which forms an $8 \times 2$ grid with 7 relations, and it detects $c = 0 . 1$ with probability 0.999. On the grid and on the power-optimised design, simulated rejection rates agree with the prediction to within 0.002. Conclusion. A design whose inputs share coordinates contains circuits that test additivity, which a Latin hypercube lacks, and a design chosen for power finds such circuits by itself, at the price of being tailored to the interaction it was chosen for.

<table><tr><td>True discrepancy</td><td>Average power ↑</td><td>Gain over space-filling ↑</td><td>Gain over variance ↑</td><td>Gain over random ↑</td></tr><tr><td>as assumed</td><td>0.52</td><td> $0 . 0 5 9 \pm 0 . 0 0 5$ </td><td> $0 . 1 1 0 \pm 0 . 0 0 9$ </td><td> $0 . 1 0 2 \pm 0 . 0 0 7$ </td></tr><tr><td>narrower bump (width 0.02)</td><td>0.44</td><td> $0 . 0 5 7 \pm 0 . 0 0 4$ </td><td> $0 . 1 1 1 \pm 0 . 0 0 6$ </td><td> $0 . 0 8 1 \pm 0 . 0 0 4$ </td></tr><tr><td>wider bump (width 0.08)</td><td>0.24</td><td> $0 . 0 1 7 \pm 0 . 0 0 2$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 3$ </td><td> $0 . 0 2 9 \pm 0 . 0 0 3$ </td></tr><tr><td>lower bump (height 0.1)</td><td>0.14</td><td> $0 . 0 1 1 \pm 0 . 0 0 1$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 1$ </td></tr><tr><td>higher bump (height 0.3)</td><td>0.87</td><td> $0 . 0 6 2 \pm 0 . 0 1 0$ </td><td> $0 . 1 1 2 \pm 0 . 0 1 7$ </td><td> $0 . 1 2 5 \pm 0 . 0 1 7$ </td></tr><tr><td>uniform location prior</td><td>0.46</td><td> $0 . 0 3 4 \pm 0 . 0 0 4$ </td><td> $0 . 0 6 0 \pm 0 . 0 0 7$ </td><td> $0 . 0 5 7 \pm 0 . 0 0 5$ </td></tr></table>

Table 3: E2, sensitivity to a misspecified alternative. The designs were chosen for bumps of height 0.2 and width 0.04 with a normal location prior centred at 0.6, and are evaluated here under the true discrepancy in the first column, without re-selection. The second column is the prior-weighted mean power of the average-power rule after five additions. The other columns are paired diferences in mean power between the average-power rule and each baseline, with one standard error over the 20 starting designs.

![](images/1b22d24d1091c4abfa21f737d61baaf0efcfa85d8978495c86821d2348d55257.jpg)

(b) additivity against c x<sub>1</sub>x<sub>2</sub>  
![](images/f6e83a027f87dc5489bf07db7e941dc150c958925dd6177ad26430d45e13431d.jpg)  
Figure 8: E3, testing an additive GP against an interaction $c x _ { 1 } x _ { 2 }$ with 16 inputs. (a) The grid, one Latin hypercube draw and the power-optimised design. (b) Power of the GP test (solid) and of the exact residual test on the null space of K (dashed), with the number of exact consistency relations of each design. Latin hypercube and random curves are means over 50 draws and have no residual test. Markers show simulated rejection rates of the GP test.

## 8 Discussion

## 8.1 What the experiments show

Experiment E1 exposes a distinction that is easy to miss when power is treated as a single number. At the end-point design, the GP can reject a suficiently large curvature even though the observations contain no structural information about curvature. The curvature is representable there as an intercept shift, and the rejection comes from that shift being improbable under the prior. Moving only four observations to the centre changes the nature of the evidence: curvature then violates an exact consistency relation.

The same issue appears in two dimensions. The 4 × 4 grid in E3 contains mixed-diference circuits that test additivity directly, while a Latin hypercube of the same size contains none. Both designs may still assign low probability to an interaction under the GP prior, but only the grid contains an exact algebraic check of additivity. This is why a scalar discrimination score can be insuficient for interpretation: the full-GP KL criterion can be large even when the projected KL and T-optimality criteria are zero.

The prospective power calculation itself behaved as expected. Proposition 2 tracked the simulated rejection probabilities closely. In E1, choosing the next observation by predicted power increased average power from 0.48 to 0.72, compared with 0.51 for predictive variance, and most of the advantage remained when the lengthscale or noise level used for choosing was wrong by a factor of two. Where the problem reduces to classical discrimination, as in E1 and E2, the resulting choices agree closely with those criteria.

## 8.2 Limits of the present analysis

The exact calibration assumes a fixed mean, kernel and noise variance with Gaussian homoscedastic noise. E1 suggests that the choice of design is not overly fragile to moderate hyperparameter misspecification, but the nominal test level is no longer exact when those hyperparameters are estimated from the observations being tested. Section 6 gives one concrete failure mode: discrepancy energy can inflate the fitted noise variance and weaken the test.

The power calculation also requires a specified discrepancy or family of discrepancies. This is useful when a scientifically meaningful failure mode is known, but it can lead to designs that are deliberately specialised. The optimised design in E3 is a clear example. E2 shows a milder gain when the discrepancy location is unknown.

Two further restrictions are structural. The present results concern a fixed design tested once; repeated testing as observations arrive requires control of false rejections over time. They also concern linear consistency relations. Rank constraints and other algebraic hypotheses lead to determinantal or more general non-linear relations and need a diferent treatment. Finally, all experiments here are synthetic and low-dimensional. A rejection should therefore be read only as incompatibility with the specified model, not as identification of a faulty observation or assumption. Model repair—through a robust likelihood [41, 43, 1], an explicit discrepancy term [42], or a physically informed mean [21, 53]—is a separate decision.

## 8.3 Implications for sequential design

A model can fit the data because it is adequate or because the design never challenged it [52, 14]. Testability is also diferent from usefulness for the optimisation task. In Bayesian optimisation, an evaluation used to challenge the model is an evaluation not used directly to improve the objective [57, 11]. The distinction can be summarised as

$$
\Big | \mathrm { f i t ~ \neq ~ t e s t a b i l i t y ~ \neq ~ d e c i s i o n ~ v a l u e } \Big | .\tag{28}
$$

The results here make the middle term computable for a fixed design. Extending this to a sequence of decisions raises several separate problems. Repeated tests need time-uniform error control, for example through confidence sequences [39]. Non-linear structural hypotheses, such as low-rank interactions, replace null-space relations by algebraic varieties; for a Segre-type rank model, distances to the relevant secant variety and its Euclidean distance degree [47, 26] are natural analogues of the projector used here. An optimisation algorithm must also decide whether the next evaluation is worth spending on testing rather than on improvement. Finally, uncertainty in the discrepancy family and in the GP hyperparameters must be propagated into that decision rather than treated as fixed. These questions are deliberately left open here.

## 9 Conclusion

A GP can reject a departure even when the design never observed the structure that was supposedly violated. The end-point example makes this concrete: curvature can be rejected through an improbable intercept although no consistency relation of that design involves curvature.

For finite-rank kernels, the null space separates these cases exactly. Gale duality then shows which observations participate in the structural relations and which minimal subsets can witness a contradiction. For general kernels, the same question becomes spectral rather than algebraic. In both cases the calculation is available before the responses are measured, so testability can be inspected as a property of the design rather than inferred only after a failed model check.

What remains is the sequential problem. During optimisation the design, the GP and the structural hypothesis all evolve. Deciding when a new evaluation should improve the objective and when it should challenge the model requires error control, uncertainty over the assumed structure and a decision rule that trades testing against optimisation.

## Funding

Ivan De Boi is funded by the Research Foundation – Flanders (FWO) through postdoctoral fellowship 1217125N.

## Data availability

The data supporting the findings of this study are available upon reasonable request from the corresponding author.

## A Block deletion identities

Order the indices as $( R , C )$ and write $A _ { R R } , A _ { R C }$ and $A _ { C C }$ for the blocks of A. The Schur complement of $A _ { R R }$ is $V _ { C } = A _ { C C } - A _ { R C } ^ { \top } A _ { R R } ^ { - 1 } A _ { R C } \in \mathbb { R } ^ { | C | \times | C | }$ , and

$$
\begin{array} { r } { A ^ { - 1 } = \left( \begin{array} { c c } { A _ { R R } ^ { - 1 } + A _ { R R } ^ { - 1 } A _ { R C } V _ { C } ^ { - 1 } A _ { R C } ^ { \top } A _ { R R } ^ { - 1 } } & { - A _ { R R } ^ { - 1 } A _ { R C } V _ { C } ^ { - 1 } } \\ { - V _ { C } ^ { - 1 } A _ { R C } ^ { \top } A _ { R R } ^ { - 1 } } & { V _ { C } ^ { - 1 } } \end{array} \right) . } \end{array}\tag{29}
$$

Hence $Q _ { C C } \ = \ V _ { C } ^ { - 1 }$ , and reading of the C block of $\alpha \ = \ A ^ { - 1 } y$ gives $\alpha _ { C } ~ = ~ V _ { C } ^ { - 1 } ( y _ { C } ~ -$ $A _ { R C } ^ { \top } A _ { R R } ^ { - 1 } y _ { R } ) = Q _ { C C } z _ { C }$ , which establishes (20) and $T _ { C } = \alpha _ { C } ^ { \top } Q _ { C C } ^ { - 1 } \alpha _ { C }$ . Since $e = S y = \sigma ^ { 2 } \alpha$ and $S _ { C C } = \sigma ^ { 2 } Q _ { C C }$ , also $\bar { T _ { C } } = e _ { C } ^ { \top } S _ { C C } ^ { - 1 } e _ { C } / \sigma ^ { 2 }$ . Expanding the quadratic form,

$$
y ^ { \top } A ^ { - 1 } y = y _ { R } ^ { \top } A _ { R R } ^ { - 1 } y _ { R } + z _ { C } ^ { \top } V _ { C } ^ { - 1 } z _ { C } = y _ { R } ^ { \top } A _ { R R } ^ { - 1 } y _ { R } + T _ { C } ,\tag{30}
$$

so $T _ { C }$ is the drop in the quadratic term of the log marginal likelihood when $C$ is deleted, and det A = det $A _ { R R }$ det V<sub>C</sub> gives the corresponding change in the log determinant.

## B Proofs

Proof of Proposition 2. The discrepancy shifts the conditional mean of $y _ { C }$ given $y _ { R }$ by $\delta _ { C }$ and its prediction $A _ { C R } A _ { R R } ^ { - 1 } y _ { R }$ by $A _ { C R } A _ { R R } ^ { - 1 } \delta _ { R }$ , so the mean of $z _ { C }$ in (20) becomes $\Delta _ { C }$ , while its covariance $V _ { C } = Q _ { C C } ^ { - 1 }$ is unchanged. Hence $V _ { C } ^ { - 1 / 2 } z _ { C } \mid y _ { R } \sim \mathcal { N } ( V _ { C } ^ { - 1 / 2 } \Delta _ { C } , I _ { | C | } )$ , and $T _ { C } = \| V _ { C } ^ { - 1 / 2 } z _ { C } \| ^ { 2 }$ is noncentral chi-square with noncentrality $\Delta _ { C } ^ { \top } V _ { C } ^ { - 1 } \Delta _ { C } = \Delta _ { C } ^ { \top } Q _ { C C } \Delta _ { C }$ . The block inverse in Appendix A gives $( Q \delta ) _ { C } = Q _ { C C } \Delta _ { C }$ , which yields the second form. For the geometric form (24), $Q = B B ^ { \top } / \sigma _ { 0 } ^ { 2 }$ gives $( Q \delta ) _ { C } = B _ { C } B ^ { \top } \delta / \sigma _ { 0 } ^ { 2 }$ and $Q _ { C C } = B _ { C } B _ { C } ^ { \top } / \sigma _ { 0 } ^ { 2 }$ , so the second form equals $\delta ^ { \top } B B _ { C } ^ { \top } ( B _ { C } B _ { C } ^ { \top } ) ^ { - 1 } B _ { C } B ^ { \top } \delta / \sigma _ { 0 } ^ { 2 } = \| \Pi _ { C } B ^ { \top } \delta \| ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ . If $\delta _ { R } = 0$ , then $\Delta _ { C } = \delta _ { C } .$ and $\delta _ { C } ^ { \top } Q _ { C C } \delta _ { C } = \| \boldsymbol { B } _ { C } ^ { \top } \delta _ { C } \| ^ { 2 } / \sigma _ { 0 } ^ { 2 }$ □

Proof of Proposition 3. By linearity, $( Q \delta ) _ { C } = Q _ { C : } D \theta$ , and (23) gives $\eta _ { C } ( \theta ) = \theta ^ { \top } M _ { C } \theta$ . The minimum over unit θ is the smallest eigenvalue, and the mean of a quadratic form in $\mathcal { N } ( 0 , \tau ^ { 2 } I _ { p } )$ is $\tau ^ { 2 } \mathrm { t r } M _ { C }$ . For $C = \{ 1 , \dots , n \} , Q _ { : C } Q _ { C C } ^ { - 1 } Q _ { C : } = Q \quad$ , and $\sigma _ { 0 } ^ { 2 } Q  P _ { N } = G G ^ { \top }$ as $\sigma _ { 0 } ^ { 2 } \to 0$ by Lemma 1. Finally, $G ^ { \top } D \theta = 0$ exactly when $D \theta \in \mathrm { c o l } ( \Phi )$ □

## C The noise fixed point

For fixed K, the log marginal likelihood as a function of the noise variance is

$$
{ \cal L } ( \sigma ^ { 2 } ) = - \frac { 1 } { 2 } y ^ { \top } Q ( \sigma ^ { 2 } ) y - \frac { 1 } { 2 } \log \operatorname * { d e t } A ( \sigma ^ { 2 } ) - \frac { n } { 2 } \log ( 2 \pi ) ,\tag{31}
$$

with derivative $\begin{array} { r } { L ^ { \prime } ( \sigma ^ { 2 } ) = \frac { 1 } { 2 } y ^ { \top } Q ^ { 2 } y - \frac { 1 } { 2 } \operatorname { t r } Q } \end{array}$ . Setting $L ^ { \prime } ( \hat { \sigma } ^ { 2 } ) = 0$ , multiplying by $2 \hat { \sigma } ^ { 4 }$ and using $\hat { \sigma } ^ { 2 } \hat { Q } y = \hat { e }$ and $\hat { \sigma } ^ { 2 } \mathrm { t r } \hat { Q } = \mathrm { t r } \hat { S }$ gives (27). The second form follows from $\hat { S } = I - K \hat { Q }$ . Both sides depend on $\hat { \sigma } ^ { 2 }$ , so (27) is a stationarity identity rather than a closed-form estimator. It holds for the maximum of the marginal likelihood. The experiments of Appendix F fit the noise by maximum a posteriori estimation with a weak prior, for which it holds only approximately.

Now let $y = f + \delta + \varepsilon$ , where $f \in \mathbb { R } ^ { n }$ is represented well by the model, $\delta \in \mathbb { R } ^ { n }$ is a deterministic discrepancy and $\varepsilon \sim \mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I )$ . At any fixed value of $\sigma ^ { 2 }$

$$
\begin{array} { r } { \mathbb { E } \| S y \| ^ { 2 } = \| S ( f + \delta ) \| ^ { 2 } + \sigma _ { 0 } ^ { 2 } \operatorname { t r } ( S ^ { 2 } ) . } \end{array}\tag{32}
$$

A discrepancy with energy along hard relations or soft prior restrictions passes almost unattenuated into the residual. At a fixed noise level it increases the expected numerator of (27) while leaving the denominator essentially unchanged, so it can raise $\hat { \sigma } ^ { 2 }$ above $\sigma _ { 0 } ^ { 2 }$ . This is not a general monotone result. The efect depends on the direction of the discrepancy, on its cross term with $f ,$ and on how S and the denominator change with the fitted variance.

## D Experimental details

E1. The null model is a zero-mean GP with kernel $k ( x , x ^ { \prime } ) = \exp ( - ( x - x ^ { \prime } ) ^ { 2 } / 2 \ell ^ { 2 } ) , \ell = 0 . 3$ and noise variance $\sigma _ { 0 } ^ { 2 } = 0 . 0 0 2 5$ . A design consists of $n = 4 0$ inputs in $[ - 1 , 1 ]$ , of which 36 are drawn uniformly and 4 lie in a tight cluster at $0 . 2 \pm 0 . 0 1$ . The check of Figure 6(b) adds a common ofset $c \in [ 0 , 0 . 3 ]$ , up to six noise standard deviations, to four kinds of group in one design: the observation at the right boundary, an interior observation near $x = - 0 . 4 .$ , one member of the cluster, and the whole cluster. The predicted power is compared with the rejection rate in 4000 data sets $y \sim { \mathcal { N } } ( \delta , A )$ for each ofset. For the bump, which is small but not zero outside the tested group, the power is computed from the general form (23). The candidates are 401 equally spaced inputs in $[ - 1 , 1 ]$ , and one observation is added. A candidate joins the tested group when it lies within 0.1 of $x _ { 0 }$ , so the degrees of freedom and the rejection threshold are recomputed for each candidate, and candidates are scored by power rather than by noncentrality. The predictive variance is that of the latent function, and the random choice is the mean power over all candidates. The detailed example uses $x _ { 0 } = 0 . 1$ , and the sweep combines 20 designs with $x _ { 0 } \in \{ - 0 . 9 , - 0 . 8 , \ldots , 0 . 9 \}$ . For the mismatch check, the rules choose with lengthscale 0.15 or 0.6, or with noise variance $6 . 2 5 \times 1 0 ^ { - 4 }$ or 0.01, and each chosen design is evaluated, and tested, with the true hyperparameters. The straight-line part uses the kernel $1 + x x ^ { \prime }$ , curvatures $c \leq 0 . 2 5$ for all three designs and $c \leq 1 0$ for the end-point design. The residual test has $n - r = 3 8$ degrees of freedom at all three designs.

E2. The bump centres lie on a grid of 31 points in [0.3, 0.9], with prior weights proportional to a normal density with mean 0.6 and standard deviation 0.15. The tested group contains all observations in the region, including those that will be added. The random baseline is averaged over 10 draws. The KL rule maximises the prior-weighted mean of $\frac { 1 } { 2 } \delta ^ { \top } Q \delta$ over the family, for all observations of the augmented design. For the final design of the average-power rule in the first case, simulated rejection rates at five bump locations agree with the predicted power to within 0.01. The sensitivity scenarios of Table 3 keep the designs and change the width, height or location prior of the true bump.

E3. The Latin hypercube samples place one input in each of 16 equal strips of each coordinate. The greedy design starts from the empty design and adds, at each step, the candidate that maximises the noncentrality of the test of all observations against $x _ { 1 } x _ { 2 } .$ without repeating a candidate. An eigenvalue of K is treated as zero when it is below $1 0 ^ { - 1 2 }$ times the largest. The smallest eigenvalue of K over all Latin hypercube and random draws is about $2 \times 1 0 ^ { - 1 1 }$ times the largest, far below $\sigma _ { 0 } ^ { 2 } .$ so these designs do have soft prior restrictions, but the interaction has little energy along them.

Computation. Every power evaluation solves a linear system of size $n ,$ and choosing an observation repeats this for every candidate. This is cheap for the designs used here, with at most 45 observations, but not for large designs without low-rank or incremental updates.

## E Background material

This appendix collects background material that the main text shortened.

The F2I objective. Forgetting to Improve (F2I) [62] selects a retained set R of observations that minimises the predictive variance over a finite target set $A \subset { \mathcal { X } }$

$$
J ( R ) = \sum _ { x \in A } \left[ k ( x , x ) - k _ { R x } ^ { \top } ( K _ { R R } + \hat { \sigma } _ { R } ^ { 2 } I _ { | R | } ) ^ { - 1 } k _ { R x } + \hat { \sigma } _ { R } ^ { 2 } \right] ,\tag{33}
$$

where $k _ { R x }$ has entries $k ( x _ { i } , x )$ for $i \in R , K _ { R R }$ is the kernel matrix of the retained inputs, and $\hat { \sigma } _ { R } ^ { 2 }$ is the noise variance refitted on them. By the noise fixed point (27), every observation with a large soft residual raises the fitted noise, including honest observations whose residuals are large by chance. An objective that rewards a small $\hat { \sigma } _ { R } ^ { 2 }$ therefore also rewards removing the upper tail of the honest noise distribution, which leads to overconfidence. A smaller predicted variance is not evidence that the model has become more correct.

Deletion at fixed noise. Deleting observations interacts with this mechanism. For a prediction input $x \in \mathcal { X }$ , let $k _ { C x } \in \mathbb { R } ^ { | C | }$ have entries $k ( x _ { i } , x )$ for $i \in C ,$ , let $d _ { C } ( x ) = k _ { C x } -$ $A _ { C R } A _ { R R } ^ { - 1 } k _ { R x } \in \mathbb { R } ^ { | C | }$ be the conditional covariance between $f ( x )$ and $y _ { C }$ given $y _ { R } .$ , with $V _ { C }$ as in (20), and let $v _ { R } ( x )$ denote the posterior variance of $f ( x )$ given $y _ { R }$ , all at the same noise variance $\sigma ^ { 2 }$

Proposition 4. The latent posterior variances satisfy

$$
v _ { R } ( x ) - v _ { R \cup C } ( x ) = d _ { C } ( x ) ^ { \top } V _ { C } ^ { - 1 } d _ { C } ( x ) \geq 0 .\tag{34}
$$

Proof. Conditioning first on $y _ { R }$ and then on $y _ { C } ,$ Gaussian conditioning on $y _ { C }$ subtracts the displayed quadratic form from the variance. The matrix $V _ { C }$ is positive definite because $\sigma ^ { 2 } > 0$ □

At fixed kernel and noise, deleting data therefore never reduces uncertainty. Any improvement after deletion must come from re-estimating model quantities, here the noise variance, rather than from the deletion itself. Deleting observations that carry discrepancy energy removes more from the numerator of (27) than from its denominator, and so tends to lower $\bar { \sigma } ^ { 2 }$

This explains both the appeal and the risk of deleting data to reduce uncertainty, as in F2I. Every observation with a large soft residual raises the fitted noise, including unperturbed observations whose residuals are large by chance, and the two cannot be told apart one observation at a time. A rejection says only that a group of observations is incompatible with the specified model. It does not show that these observations are wrong, and when they reflect real behaviour of the function, revising the model is preferable to deleting them. Appendix F shows that a greedy deletion rule of our own, which is not the published F2I implementation, becomes overconfident in this way (Tables 4 and 5).

Masking. Closely spaced observations with a common bias predict each other well when each is held out alone. The other members of the group pull each leave-one-out prediction towards the biased value, and every individual residual is small. Holding out the whole group removes this mutual support, and the group residual $z _ { C }$ of (20) reveals the shared ofset. This is the masking problem of classical multiple-outlier diagnostics [8, 37, 35]. In the language of (21), masking is a cancellation within $S _ { C C }$ . The individual statistic of observation i is $\alpha _ { i } ^ { 2 } / Q _ { i i }$ , and under a common ofset $c$ on $C$ the mean of $\alpha _ { i }$ is $c \textstyle \sum _ { j \in C } Q _ { i j }$ . For nearby observations the of-diagonal entries $Q _ { i j }$ , and hence the inner products $b _ { i } ^ { \top } b _ { j } = S _ { i j }$ of their spectral coordinates, are typically negative, so this sum is small and the ofset largely cancels in every individual statistic. The joint statistic $T _ { C }$ uses the whole block $Q _ { C C }$ and does not sufer this cancellation. The remedy is therefore the joint hypothesis, and it is available without any change of coordinates.

Several candidate groups. With several candidate groups, a Bonferroni or Holm correction [38] controls the probability of any false rejection for a fixed family. Adaptive remove-and-refit searches, such as those of S1–S3, require further care [18], which is why their calibration is checked empirically below.

Exact and truncated coordinates. The last form of (21) shows that the group test computed in spectral coordinates is identical to the classical multiple-case test. A diference between the two can arise only from approximating the residual matrix, for instance by a hard truncation of the spectrum of $K$ . The method “Group test, Gale coordinates” below measures what such a truncation costs.

Soft relations as stencils. For smooth stationary kernels and closely spaced inputs, the eigenvectors of K with small eigenvalues typically oscillate across neighbouring inputs, so the soft relations behave like approximate higher-order finite diferences. The first eigenvector below the noise level in Figure 3 alternates in sign between neighbouring inputs. A row of the residual matrix $S$ is a local stencil: it compares $y _ { i }$ with a weighted average of its neighbours, much like the straight-line relation, and it nearly annihilates constant, linear and quadratic trends.

A classical use of Gale duality. Properties of a point configuration can be read of from its Gale transform. For example, a subset of the points is the set of points lying on some face of their convex hull exactly when the origin lies in the relative interior of the convex hull of the Gale vectors of the remaining points. This makes Gale duality particularly efective for configurations with only a few more points than dimensions, where the dual space is small.

Afine and projective Veronese maps. The feature map (9) of all monomials of degree at most $q$ is the afine form of the Veronese map [47]. The classical Veronese map is projective and uses the monomials of degree exactly q in homogeneous coordinates $( x _ { 0 } , x _ { 1 } , \ldots , x _ { d } )$ Setting $x _ { 0 } = 1$ turns these into the monomials of degree at most $q$ in $( x _ { 1 } , \ldots , x _ { d } )$ , so the two descriptions give the same consistency relations.

## F Supplementary experiments: detection and forgetting

This appendix reports a simulation study of detection and forgetting, with three supplementary experiments numbered S1–S3.

Experiments S1–S3 turn from testability to detection and forgetting. They check the claims of Section 6 that a local discrepancy inflates the fitted noise everywhere and that uncertainty-minimising deletion removes this inflation at the price of overconfidence, and they compare this deletion rule with the test-based rules of Section 5. The three experiments share one data model, pipeline and set of methods, and difer only in the kind of discrepancy: a few shifted observations (S1), a region that the kernel genuinely cannot represent (S2), and a batch of shifted observations that mask one another (S3).

Data. Experiments S1, S2 and S3 use $n = 4 0$ training inputs in $[ - 1 , 1 ]$ , Gaussian measurement noise with variance $\sigma _ { 0 } ^ { 2 } = 0 . 0 0 2 5$ , the target region $\mathcal { A } = [ 0 . 3 , 0 . 9 ]$ , and 200 held-out test inputs drawn uniformly in A. Each configuration is repeated for 20 random seeds.

• $S 1 ,$ structured function with a cluster. The latent function is the cubic $f ( x ) = 0 . 5 +$ $0 . 8 x - 1 . 2 x ^ { 2 } + 0 . 9 x ^ { 3 }$ , with inputs drawn uniformly. The three observations nearest to $x = - 0 . 1$ , just outside $\mathcal { A } ,$ , are shifted by $+ 0 . 4 .$ which is eight noise standard deviations.

• $S 2 ,$ genuine model mismatch. The latent function is $f ( x ) = \sin 3 x + 1 . 2 \sin ( 3 5 x )$ for $x < - 0 . 3$ and $f ( x ) = \sin 3 x$ for $x \ge - 0 . 3$ . On $x < - 0 . 3$ it oscillates far faster than an RBF kernel with a single lengthscale can follow. The roughly 13 observations with $x < - 0 . 3$ are correct measurements of $f ,$ but they are counted as misfit observations because the kernel cannot represent them.

• S3, a masked batch. The latent function is the cubic of S1. There are $4 0 - k$ inputs drawn uniformly and a batch of k inputs at $x = 0 . 2 \pm 0 . 0 1$ , just outside $\mathcal { A } .$ whose observations are all shifted by +0.3. This mimics a mis-calibrated sensor during a run of repeated measurements. We vary $k \in \{ 2 , 3 , 4 , 5 \}$ . A variant places the batch inside the target region, at $x = 0 . 5 \pm 0 . 0 1$

Pipeline. The pipeline is identical for every method, so that diferences between methods are attributable to the deletion rule alone. (i) The lengthscale ℓ, output scale s and noise variance $\sigma ^ { 2 }$ of the RBF kernel $k ( x , x ^ { \prime } ) = s \exp ( - ( x - x ^ { \prime } ) ^ { 2 } / 2 \ell ^ { 2 } )$ are fitted by maximum a posteriori (MAP) estimation on all 40 observations, with a log-normal prior log $\ell \sim \mathcal { N } ( 0 , 1 )$ restricted to $\ell \geq 0 . 2$ , a flat prior on log s, and a Gamma prior with shape 2 and rate 0.15 on $\sigma ^ { 2 }$ . The lower bound on ℓ expresses the modelling assumption of a smooth latent function. Without it, the fit in S2 selects $\ell \approx 0 . 0 5$ in most seeds, chases the fast oscillation, and generalises poorly in A for every method. (ii) The method returns the observations to keep. (iii) The noise variance alone is refitted by MAP on the kept observations, with the kernel fixed. (iv) The GP predicts at the test inputs.

Methods. The tables compare six methods, named as in the table rows. The first two are references that bracket the comparison, and the other four are deletion rules.

• Full GP. Keep every observation. This shows the efect of the misfit observations when nothing is done about them.

• Oracle. Delete exactly the planted misfit observations and nothing else. No real method can know which observations these are, so the oracle is a best-case reference rather than a competitor.

• F2I-style. Greedy deletion on the objective (33), the total predictive variance over a grid of 30 inputs in A. At every step each single deletion is evaluated with an exact refit of the noise variance, the best one is applied, and the search stops when no deletion lowers (33). This uses the published objective [62] with exact refits instead of first-order influence approximations, and is not the authors’ implementation.

• Single-observation test. Repeatedly delete the observation with the largest standardised leave-one-out residual $| \alpha _ { i } | / \sqrt { Q _ { i i } }$ , recomputed after every deletion, while it exceeds the Bonferroni threshold $z _ { 1 - 0 . 0 5 / ( 2 n ) } \approx 3 . 2$ , the corresponding quantile of the standard normal distribution.

• Group test, classical. The test (21) applied greedily. Candidate groups are all single observations and all runs of 2 to $s _ { \mathrm { m a x } }$ neighbouring inputs, with $s _ { \operatorname* { m a x } } = 3$ in S1 and S2 and $s _ { \mathrm { m a x } } = 5$ in S3. At every step the candidate with the smallest p-value under (22) is deleted, and the search stops when that p-value exceeds 0.05/M, where M is the number of remaining candidates.

• Group test, Gale coordinates. The same candidates, search and stopping rule, but with the test computed in a Gale basis. The basis G<sup>˜</sup> consists of the eigenvectors of K whose eigenvalues do not exceed the reference noise, a hard-threshold approximation of the soft construction of Section 4. With $\tilde { P } _ { N } = \tilde { G } \tilde { G } ^ { \top }$ and $\tilde { e } = \tilde { P } _ { N } y ;$ a group C is scored by the increase in explained test failure $\tilde { e } _ { C } ^ { \top } ( ( \tilde { P } _ { N } ) _ { C C } ) ^ { + } \tilde { e } _ { C }$ , which is the hard-truncated Gale form of (21). With the exact soft operator in place of ${ \tilde { P } } _ { N }$ , this method would coincide with the classical group test (Section 5). It is included to measure what the truncation costs.

All three tests use the same reference noise variance: the MAP noise of the GP restricted to the training observations inside A. This avoids judging failures against a noise estimate inflated by the very discrepancy under test (Section 5), but it assumes that the model is adequate inside A.

Metrics. On the test inputs in A we report the mean squared error (MSE) of the predictive mean against the latent function, the empirical coverage of the 95% predictive intervals for new noisy observations (ideally 0.95), the negative log predictive density (NLPD) of those observations, and the mean predictive variance (sharpness), which is the F2I objective divided by the number of target inputs. On the training set we report the fraction of misfit observations deleted (higher is better), the fraction of honest observations deleted (lower is better), and the refitted noise variance $\hat { \sigma } ^ { 2 }$ , whose true value is $\sigma _ { 0 } ^ { 2 } = 0 . 0 0 2 5$ . All numbers are produced by the code accompanying this paper.

Noise inflation and overconfidence. Tables 4 and 5 confirm the mechanism of Section 6. In S1, three shifted observations raise the fitted noise variance of the full GP to $9 . 8 \times 1 0 ^ { - 3 }$ almost four times the true value. In S2 the rough region raises it to 0.22, about ninety times the true value. Because the same noise enters every prediction, the intervals in the target region are far too wide, and their coverage is 1.00 instead of 0.95.

F2I-style deletion removes this inflation but does not stop there. It keeps only about 12 of the 40 observations, deletes roughly two thirds of the honest ones, and drives the fitted noise variance to about 10<sup>−5</sup>, two orders of magnitude below the truth. Its predictions are the sharpest of all methods, yet only 9 to 14% of new observations fall inside its nominal 95% intervals, and its NLPD exceeds 100, against values between −1.5 and 0.2 for every other method. Figure 10 shows this trade-of for all three experiments: F2I-style deletion wins on sharpness, the criterion it optimises, and loses on calibration. Figure 9 shows typical fits. The test-based methods delete at most 3.2% of the honest observations, recover a noise variance close to the truth (with one exception, discussed below), and keep the coverage between 0.93 and 0.97.

S2 is the hardest case for every method. The rough region contains stretches where the fast oscillation happens to look smooth, so even the best test finds only about 80% of the misfit observations. Nevertheless the remaining misfit has little influence on the target region, and the accuracy of every test-based method there is within 15% of the oracle.

<table><tr><td>Method</td><td>Kept</td><td> $\hat { \sigma } ^ { 2 } \ : \left( \times 1 0 ^ { - 3 } \right)$ </td><td>MSE  $( \times 1 0 ^ { - 4 } ) \downarrow$ </td><td>Coverage</td><td>NLPD ↓</td><td>Misfit del. ↑</td><td>Honest del. ↓</td></tr><tr><td>Full GP</td><td>40.0</td><td>9.78</td><td> $8 . 8 \pm 1 0 . 2$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td><td>-1.18</td><td>0.00</td><td>0.000</td></tr><tr><td>Oracle</td><td>37.0</td><td>2.60</td><td> $5 . 6 \pm 6 . 5$ </td><td> $0 . 9 4 \pm 0 . 0 3$ </td><td>-1.46</td><td>1.00</td><td>0.000</td></tr><tr><td>F2I-style</td><td>12.2</td><td>0.01</td><td> $1 0 . 2 \pm 6 . 7$ </td><td> $0 . 1 2 \pm 0 . 0 6$ </td><td>154.46</td><td>0.85</td><td>0.681</td></tr><tr><td>Single-observation test</td><td>37.5</td><td>3.11</td><td> $7 . 0 \pm 8 . 6$ </td><td> $\mathbf { 0 . 9 4 \ : \pm { \ : 0 . 0 5 } }$ </td><td>-1.43</td><td>0.65</td><td>0.014</td></tr><tr><td>Group test, classical</td><td>37.0</td><td>2.54</td><td> ${ \bf 5 . 6 \pm 6 . 5 }$ </td><td> ${ \bf 0 . 9 4 } \pm \bf 0 . 0 4$ </td><td>-1.45</td><td>0.95</td><td>0.003</td></tr><tr><td>Group test, Gale coordinates</td><td>37.2</td><td>2.66</td><td> $6 . 6 \pm 8 . 7$ </td><td> $\mathbf { 0 . 9 4 \ : \pm { \ : 0 . 0 5 } }$ </td><td>-1.45</td><td>0.80</td><td>0.011</td></tr></table>

Table 4: S1, a cubic function with a cluster of three shifted observations just outside the target region. Means over 20 seeds, ± one standard deviation across seeds. The true noise variance is $\sigma _ { 0 } ^ { 2 } = 2 . 5 \times 1 0 ^ { - 3 }$ . Rows: $F u l l ~ G P$ keeps all observations. Oracle deletes exactly the planted misfit observations, which no real method can know, and serves as a best-case reference. F2I-style deletes greedily to minimise predictive variance in ${ \mathcal { A } } .$ The remaining three rows are the single-observation test and the group test in classical and in Gale coordinates. Columns: number of observations kept, refitted noise variance, mean squared error and NLPD in A, coverage of the 95% predictive intervals, and the fractions of misfit and of honest observations deleted. Arrows indicate whether lower or higher values are better. Bold marks the best deletion rule in each column, where best means closest to the true noise variance and to the nominal coverage 0.95. The Full GP and Oracle rows are references and are not marked.

Deletion trajectories. Figure 11 follows F2I-style deletion one step at a time, refitting the noise and evaluating the predictions in A after every single deletion, in order to see where it goes wrong. Each column is one experiment, and the horizontal axis counts the observations deleted so far. The top row shows, for each step, the fraction of seeds in which the observation deleted at that step is a planted misfit one, so a bar near 1 means that a misfit observation was removed and a bar near 0 that an honest one was. The middle row shows the refitted noise variance relative to the truth, where 1 is correct, and the bottom row the coverage of the 95% predictive intervals in A, where 0.95 is ideal. Thin lines are individual seeds and thick lines their median. The dashed vertical line marks the number of planted misfit observations, which is where a method that deleted exactly the misfit observations would stop. The green lines follow the classical group test in the same way, one selected group at a time in its own order of selection, and markers show where each method stops in each seed, red for F2I-style deletion and green for the group test, so that its path and stopping point can be compared with those of F2I-style deletion and with the ideal stopping point. The group test stops after a few deletions, mostly of misfit observations, so its noise variance ends close to the truth and its coverage close to 0.95. The early deletions mostly hit misfit observations: about 70% of the first deletions in each experiment are misfit ones. When as many observations have been deleted as there are misfit ones, the median fitted noise variance in S1 and S3 is within 2% of the true value, and the median coverage is 0.94. At that point F2I-style deletion is essentially as good as the oracle. It does not stop there, however, because every further deletion still lowers its objective. Removing an honest observation whose residual is large by chance lowers the refitted noise, and hence the predictive variance in A, just as removing a misfit one does. F2I-style deletion stops only when so few observations remain that losing another raises the predictive variance more than the smaller noise lowers it. The group test stops for a diferent reason: it deletes a group only if that group is statistically surprising under the reference noise, and once the misfit observations are gone no remaining group passes the Bonferroni threshold. One rule stops when deleting no longer pays, the other when nothing left is surprising. It goes on to delete between 17 and 25 honest observations on average, the fitted noise falls two to three orders of magnitude below the truth, and coverage collapses. The group test stops close to the point where F2I-style deletion should have stopped (green markers). The failure of F2I-style deletion is therefore not in what it deletes first, but in the absence of a stopping rule that distinguishes a misfit observation from an honest one with a large residual. A test-based stopping rule supplies such a rule, approximately calibrated, as the next paragraph shows.

S2: Group test, classical, $\hat { \sigma } ^ { 2 } = 3 . 2 \mathrm { e } { \cdot } 0 3$
<table><tr><td>Method</td><td>Kept</td><td> $\hat { \sigma } ^ { 2 } \ : \left( \times 1 0 ^ { - 3 } \right)$ </td><td>MSE  $( \times 1 0 ^ { - 4 } ) \downarrow$ </td><td>Coverage</td><td>NLPD ↓</td><td>Misfit del. ↑</td><td>Honest del. ↓</td></tr><tr><td>Full GP</td><td>40.0</td><td>222.49</td><td> $5 3 . 6 \pm 4 9 . 4$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td>0.21</td><td>0.00</td><td>0.000</td></tr><tr><td>Oracle</td><td>26.7</td><td>2.45</td><td> $5 . 5 \pm 6 . 0$ </td><td> $0 . 9 3 \pm 0 . 0 5$ </td><td>-1.45</td><td>1.00</td><td>0.000</td></tr><tr><td>F2I-style</td><td>11.9</td><td>0.03</td><td> $1 5 . 3 \pm 9 . 3$ </td><td> $0 . 1 4 \pm 0 . 0 8$ </td><td>143.86</td><td>0.80</td><td>0.648</td></tr><tr><td>Single-observation test</td><td>30.7</td><td>3.77</td><td> $6 . 3 \pm 5 . 8$ </td><td> $\mathbf { 0 . 9 5 \ : \pm { \ : 0 . 0 5 } }$ </td><td>-1.41</td><td>0.64</td><td>0.032</td></tr><tr><td>Group test, classical</td><td>28.5</td><td>2.86</td><td> $5 . 7 \pm 6 . 2$ </td><td> $0 . 9 4 \pm 0 . 0 5$ </td><td>-1.43</td><td>0.81</td><td>0.027</td></tr><tr><td>Group test, Gale coordinates</td><td>29.6</td><td>10.44</td><td> ${ \bf 5 . 6 \pm 4 . 2 }$ </td><td> $0 . 9 7 \pm 0 . 0 4$ </td><td>-1.22</td><td>0.75</td><td>0.015</td></tr></table>

Table 5: S2, a function that oscillates faster than the kernel can follow for $x < - 0 . 3 .$ . Same layout as Table 4.

![](images/a108826a934678c1f85964ce1e149023cfb8e6ed6ba5947688ea5ce2278184d7.jpg)

![](images/b6255c15111e6ee3c2bba324222da92fb6c36a594a595076226ccd6dc71b377b.jpg)

![](images/bb2c6f77908f706456394dcb0ebf158827f27b5b35be942c29c64e31e49095e8.jpg)

![](images/4b108f76271b6babe672c1dcce140d2851ea47b585ae081c7377dc742b58b894.jpg)

![](images/ea674c4688cedb3ee5d386f79d2ca4485fe4eac7fdfe2632e0ff19c1b8007bf4.jpg)

S3 (k = 4): Group test, classical, σ̂<sup>2</sup> = 2.1e-03  
![](images/4c6966fc88df2f8a0bc5e01b890314a71b4a1cac9ea776c3e37e27a51a289421.jpg)  
Figure 9: One seed of each experiment. Left: F2I-style deletion. Right: the classical group test. Dots are kept observations and crosses deleted ones. Red marks the planted misfit observations, and the dashed curve is the latent function. Lines and bands show the predictive mean and the 95% predictive interval for new observations. The target region A is shaded. F2I-style deletion removes most honest observations and leaves intervals that are far too narrow. The group test removes the misfit observations and little else.

![](images/b675b892506f58ac8fef8ff8dba4df0164ad0fef52b040d27f6da060adf9f490.jpg)  
mean predictive variance in   
Figure 10: Sharpness against calibration in S1, S2 and S3 (k = 4). Each marker is the mean over 20 seeds for one method and one experiment. The dotted line marks the nominal coverage of 0.95. A small predictive variance is only useful if the coverage remains close to this line.

Calibration of the complete pipeline. Equation (22) is exact only for a fixed group and a specified covariance, whereas S1–S3 estimate the kernel from the same data, estimate the reference noise, and select and remove groups adaptively. Table 6 therefore runs the complete pipeline on 1000 data sets per configuration that contain no misfit observations, generated from the cubic of S1 with uniform inputs or with the cluster of S3. Any deletion is then a false rejection. The pipeline is moderately liberal. At least one observation is deleted in 6.7 to 7.6% of the data sets by the group tests and in 8.8 to 9.4% by the single-observation test, against a nominal 5%, and on average fewer than 1% of the observations are deleted. The data are generated from a fixed cubic rather than drawn from the GP, so this benchmark does not isolate the calibration of the test under its own null hypothesis. The excess over the nominal level may reflect both the plug-in of estimated hyperparameters [6] and the mismatch between the cubic and the GP prior. Table 7 separates the two efects with data drawn from the GP itself, 1000 data sets per design. With the true kernel and noise variance, the single-observation test deletes something in 4.8 to 5.3% of the data sets, close to the nominal 5%, and the group tests in 1.5 to 3.2%, because the Bonferroni correction over overlapping candidate groups is conservative. With hyperparameters estimated from the same data, the rates rise to 6.3 to 8.5%. The excess over the nominal level in Table 6 is therefore mainly caused by estimating the hyperparameters, not by the mismatch between the cubic and the GP prior. It is small compared with the deletions of F2I-style selection, but the stopping rule should be read as approximately rather than exactly calibrated.

<table><tr><td></td><td colspan="2">Uniform inputs</td><td colspan="2">With a cluster</td></tr><tr><td>Method</td><td>Any deletion ↓</td><td>Deleted ↓</td><td>Any deletion ↓</td><td>Deleted ↓</td></tr><tr><td>Single-observation test</td><td>0.088</td><td>0.0036</td><td>0.094</td><td>0.0044</td></tr><tr><td>Group test, classical</td><td>0.073</td><td>0.0059</td><td>0.067</td><td>0.0082</td></tr><tr><td>Group test, Gale coordinates</td><td>0.076</td><td>0.0060</td><td>0.072</td><td>0.0087</td></tr></table>

Table 6: Calibration of the complete deletion pipeline of S1–S3 when the data contain no misfit observations, over 1000 seeds per configuration. “Any deletion” is the fraction of seeds in which at least one observation is deleted, which the Bonferroni stopping rule is meant to keep near 0.05. “Deleted” is the mean fraction of the 40 observations deleted. Bold marks the best value in each column.

<table><tr><td></td><td colspan="2">Uniform inputs</td><td colspan="2">With a cluster</td></tr><tr><td>Method</td><td>Fixed</td><td>Estimated</td><td>Fixed</td><td>Estimated</td></tr><tr><td>Single-observation test</td><td>0.053</td><td>0.084</td><td>0.048</td><td>0.085</td></tr><tr><td>Group test, classical</td><td>0.025</td><td>0.072</td><td>0.015</td><td>0.063</td></tr><tr><td>Group test, Gale coordinates</td><td>0.032</td><td>0.076</td><td>0.023</td><td>0.067</td></tr></table>

Table 7: Calibration under a true GP null: fraction of 1000 data sets drawn from the GP in which at least one observation is deleted, against a nominal 0.05. “Fixed” uses the true kernel and noise variance, “Estimated” the pipeline of S1–S3 with hyperparameters fitted to the same data.

Masking. Table 8 and Figure 12 show the masked batch of S3. Each observation in the batch is predicted well by its shifted neighbours when it is held out alone, so the single-observation test finds only 68% of a batch of four, and its error in the target region is 2.4 times that of the group tests. The gap widens with the batch size. At k = 5 the single-observation test finds half of the batch, while the group tests still find 78%. When the batch lies inside the target region, every test fails to find it, and the error in A is about twenty times that of the oracle. The reason is the reference noise. It is estimated from the observations in A, which now include the batch, and its mean of $2 . 2 \times 1 0 ^ { - 2 }$ is about six times the value in the other S3 runs. Against this inflated reference nothing looks surprising. This is a genuine limitation of a target-region reference, discussed in Section 5.

![](images/a6144a614a415f7b8d4a7ab682f847cb44885f8c11bca08f24b6486d72dfb2f6.jpg)  
Figure 11: Deletion trajectories of F2I-style deletion, from no deletion until its stopping point. Top: fraction of seeds in which the observation deleted at that step is a misfit one. Middle: refitted noise variance relative to the truth (logarithmic scale). Bottom: coverage of the 95% predictive intervals in A. Thin lines are individual seeds and thick lines the median. The dashed vertical line marks the number of misfit observations, and the dotted horizontal lines mark the ideal values. Green lines follow the classical group test one selected group at a time. Markers show where each method stops in each seed, red for F2I-style deletion and green for the group test.

Gale coordinates against the classical group test. Table 9 compares the two group tests directly. In S3 they make identical decisions in 99 of the 100 runs, including the run with the batch inside A. The gain over single-observation tests in S3 is therefore entirely a gain of testing groups, which the classical test delivers without any Gale vocabulary. In S1 and S2 the two tests difer more often, because the Gale version uses a hard spectral threshold that approximates the soft operator S of Section 4. These diferences never favour the Gale version: in S1 it finds fewer misfit observations (80% against 95%), and in S2 it leaves a noise variance nearly four times larger, with equal accuracy. These results quantify what the hard truncation costs. Together with the identity of Section 5, they show that, for data already collected, the improvement comes from testing the right group jointly, not from the coordinates in which the test is expressed. The Gale view does not detect more discrepancies in such data. Its contribution is the prospective one shown in E1 to E3: before data are collected, it says which assumptions a design can test and with what power.

<table><tr><td>Method</td><td>Kept</td><td> $\hat { \sigma } ^ { 2 } \ : \left( \times 1 0 ^ { - 3 } \right)$ </td><td> $\mathrm { M S E } \left( \times 1 0 ^ { - 4 } \right) \downarrow$ </td><td>Coverage</td><td>NLPD ↓</td><td>Misfit del. ↑</td><td>Honest del. ↓</td></tr><tr><td>Full GP</td><td>40.0</td><td>8.33</td><td> $2 9 . 3 \pm 1 2 . 7$ </td><td> $0 . 9 8 \pm 0 . 0 3$ </td><td>-1.11</td><td>0.00</td><td>0.000</td></tr><tr><td>Oracle</td><td>36.0</td><td>2.31</td><td> $3 . 9 \pm 2 . 6$ </td><td> $0 . 9 4 \pm 0 . 0 2$ </td><td>-1.50</td><td>1.00</td><td>0.000</td></tr><tr><td>F2I-style</td><td>11.8</td><td>0.01</td><td> $2 4 . 2 \pm 3 4 . 2$ </td><td> $0 . 0 9 \pm 0 . 0 7$ </td><td>649.65</td><td>0.91</td><td>0.682</td></tr><tr><td>Single-observation test</td><td>36.8</td><td>3.03</td><td> $1 7 . 8 \pm 2 4 . 3$ </td><td> ${ \bf 0 . 9 3 \pm 0 . 0 5 }$ </td><td>-1.32</td><td>0.68</td><td>0.014</td></tr><tr><td>Group test, classical</td><td>36.1</td><td>2.47</td><td> ${ \bf 7 . 3 \pm 1 1 . 2 }$ </td><td> ${ \bf 0 . 9 3 \pm 0 . 0 4 }$ </td><td>-1.45</td><td>0.90</td><td>0.008</td></tr><tr><td>Group test, Gale coordinates</td><td>36.1</td><td>2.47</td><td> ${ \bf 7 . 3 \pm 1 1 . 2 }$ </td><td> ${ \bf 0 . 9 3 \pm 0 . 0 4 }$ </td><td>-1.45</td><td>0.90</td><td>0.008</td></tr></table>

Table 8: S3, a masked batch of k = 4 shifted observations just outside the target region. Same layout as Table 4.

![](images/adf1eaca3e4415d7989e407e331628d3012809793e1013f2cb4ba4c07d0c58a1.jpg)  
Figure 12: S3, a masked batch of k shifted observations. Left: mean squared error in the target region, with error bars of one standard error over 20 seeds (logarithmic scale). Right: fraction of the batch deleted. The two group tests coincide.

<table><tr><td>Experiment</td><td>Identical decisions</td><td>MSE, single (×10−4) ↓</td><td>MSE, classical (×10−4) ↓ MSE, Gale (×10−4) ↓</td></tr><tr><td>S1</td><td>17/20</td><td>7.0</td><td>5.6</td></tr><tr><td>S2</td><td>3/20</td><td>6.3</td><td>5.7</td></tr><tr><td>S3, k = 2</td><td>20/20</td><td>5.7</td><td>5.5</td></tr><tr><td>S3, k = 3</td><td>20/20</td><td>13.0</td><td>6.3</td></tr><tr><td>S3, k = 4</td><td>20/20</td><td>17.8</td><td>7.3</td></tr><tr><td>S3, k = 5</td><td>19/20</td><td>19.8</td><td>7.3 9.7 9.8</td></tr><tr><td>S3, k = 4, batch inside A</td><td>20/20</td><td>99.6</td><td>99.6 99.6</td></tr></table>

Table 9: Group tests in Gale coordinates against classical group tests with the same candidate groups, reference noise and stopping rule. “Identical decisions” counts the seeds in which both delete exactly the same observations. The single-observation test is shown for comparison. Means over 20 seeds. Bold marks a method whose MSE is smaller than that of at least one other method by more than two paired standard errors over the seeds, and not significantly larger than that of any. Arrows indicate that lower values are better.

## References

[1] S. Ament, E. Santorella, D. Eriksson, B. Letham, M. Balandat, and E. Bakshy. Robust Gaussian processes via relevance pursuit. In Advances in Neural Information Processing Systems 37, 2024.

[2] A. C. Atkinson. Planning experiments to detect inadequate regression models. Biometrika, 59(2):275–293, 1972.

[3] A. C. Atkinson and V. V. Fedorov. The design of experiments for discriminating between two rival models. Biometrika, 62(1):57–70, 1975.

[4] A. C. Atkinson and M. Riani. Robust Diagnostic Regression Analysis. Springer, 2000.

[5] A. C. Atkinson, A. N. Donev, and R. D. Tobias. Optimum Experimental Designs, with SAS. Oxford University Press, 2007.

[6] F. Bachoc. Cross validation and maximum likelihood estimations of hyper-parameters of Gaussian processes with model misspecification. Computational Statistics & Data Analysis, 66:55–69, 2013.

[7] L. S. Bastos and A. O’Hagan. Diagnostics for Gaussian process emulators. Technometrics, 51(4):425–438, 2009.

[8] R. J. Beckman and R. D. Cook. Outlier..........s. Technometrics, 25(2):119–149, 1983.

[9] D. A. Belsley, E. Kuh, and R. E. Welsch. Regression Diagnostics: Identifying Influential Data and Sources of Collinearity. Wiley, 1980.

[10] A. Björner, M. Las Vergnas, B. Sturmfels, N. White, and G. M. Ziegler. Oriented Matroids. Second edition. Encyclopedia of Mathematics and its Applications 46, Cambridge University Press, 1999.

[11] I. Bogunovic and A. Krause. Misspecified Gaussian process bandit optimization. In Advances in Neural Information Processing Systems 34, 2021.

[12] G. E. P. Box and N. R. Draper. A basis for the selection of a response surface design. Journal of the American Statistical Association, 54(287):622–654, 1959.

[13] G. E. P. Box and W. J. Hill. Discrimination among mechanistic models. Technometrics, 9(1):57–71, 1967.

[14] G. E. P. Box. Sampling and Bayes’ inference in scientific modelling and robustness (with discussion). Journal of the Royal Statistical Society, Series A, 143(4):383–430, 1980.

[15] G. E. P. Box and N. R. Draper. Empirical Model-Building and Response Surfaces. Wiley, 1987.

[16] J. Brynjarsdóttir and A. O’Hagan. Learning about physical parameters: the importance of model discrepancy. Inverse Problems, 30(11):114007, 2014.

[17] K. Chaloner and I. Verdinelli. Bayesian experimental design: a review. Statistical Science, 10(3):273–304, 1995.

[18] S. Chen and J. Bien. Valid inference corrected for outlier removal. Journal of Computational and Graphical Statistics, 29(2):323–334, 2020.

[19] R. D. Cook. Detection of influential observation in linear regression. Technometrics, 19(1):15–18, 1977.

[20] R. D. Cook and S. Weisberg. Residuals and Influence in Regression. Chapman and Hall, 1982.

[21] I. De Boi, S. Sels, and R. Penne. Semidata-driven calibration of galvanometric setups using Gaussian processes. IEEE Transactions on Instrumentation and Measurement, 71:1–8, 2022.

[22] I. De Boi, C. H. Ek, and R. Penne. Surface approximation by means of Gaussian process latent variable models and line element geometry. Mathematics, 11(2):380, 2023.

[23] I. De Boi, E. Embrechts, Q. Schatteman, R. Penne, S. Truijen, and W. Saeys. Assessment and treatment of visuospatial neglect using active learning with Gaussian processes regression. Artificial Intelligence in Medicine, 149:102770, 2024.

[24] J. A. De Loera, J. Rambau, and F. Santos. Triangulations: Structures for Algorithms and Applications. Springer, 2010.

[25] H. Dette and S. Titof. Optimal discrimination designs. Annals of Statistics, 37(4):2056– 2082, 2009.

[26] J. Draisma, E. Horobeţ, G. Ottaviani, B. Sturmfels, and R. R. Thomas. The Euclidean distance degree of an algebraic variety. Foundations of Computational Mathematics, 16(1):99–149, 2016.

[27] O. Dubrule. Cross validation of kriging in a unique neighborhood. Mathematical Geology, 15(6):687–699, 1983.

[28] D. Gale. Neighboring vertices on a convex polyhedron. In H. W. Kuhn and A. W. Tucker, editors, Linear Inequalities and Related Systems, Annals of Mathematics Studies 38, pp. 255–263. Princeton University Press, 1956.

[29] J. R. Gardner, G. Malkomes, R. Garnett, K. Q. Weinberger, D. Barbour, and J. P. Cunningham. Bayesian active model selection with an application to automated audiometry. In Advances in Neural Information Processing Systems 28, 2015.

[30] R. Garnett. Bayesian Optimization. Cambridge University Press, 2023.

[31] A. Gelman, X.-L. Meng, and H. Stern. Posterior predictive assessment of model fitness via realized discrepancies. Statistica Sinica, 6(4):733–807, 1996.

[32] D. Ginsbourger and C. Schärer. Fast calculation of Gaussian process multiple-fold crossvalidation residuals and their covariances. Journal of Computational and Graphical Statistics, 34(1):1–14, 2025.

[33] P. W. Goldberg, C. K. I. Williams, and C. M. Bishop. Regression with input-dependent noise: a Gaussian process treatment. In Advances in Neural Information Processing Systems 10, pp. 493–499, 1998.

[34] B. Grünbaum. Convex Polytopes. Second edition, prepared by V. Kaibel, V. Klee and G. M. Ziegler. Graduate Texts in Mathematics 221, Springer, 2003.

[35] A. S. Hadi and J. S. Simonof. Procedures for the identification of multiple outliers in linear models. Journal of the American Statistical Association, 88(424):1264–1272, 1993.

[36] T. J. Hastie and R. J. Tibshirani. Generalized Additive Models. Chapman and Hall, 1990.

[37] D. M. Hawkins, D. Bradu, and G. V. Kass. Location of several outliers in multipleregression data using elemental sets. Technometrics, 26(3):197–208, 1984.

[38] S. Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979.

[39] S. R. Howard, A. Ramdas, J. McAulife, and J. Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. Annals of Statistics, 49(2):1055–1080, 2021.

[40] E. R. Jones and T. J. Mitchell. Design criteria for detecting model inadequacy. Biometrika, 65(3):541–551, 1978.

[41] P. Jylänki, J. Vanhatalo, and A. Vehtari. Robust Gaussian process regression with a Student-t likelihood. Journal of Machine Learning Research, 12:3227–3257, 2011.

[42] M. C. Kennedy and A. O’Hagan. Bayesian calibration of computer models. Journal of the Royal Statistical Society, Series B, 63(3):425–464, 2001.

[43] K. Kersting, C. Plagemann, P. Pfaf, and W. Burgard. Most likely heteroscedastic Gaussian process regression. In Proceedings of the 24th International Conference on Machine Learning, pp. 393–400, 2007.

[44] J. López-Fidalgo, C. Tommasi, and P. C. Trandafir. An optimal experimental design criterion for discriminating between non-normal models. Journal of the Royal Statistical Society, Series B, 69(2):231–242, 2007.

[45] D. J. C. MacKay. Bayesian interpolation. Neural Computation, 4(3):415–447, 1992.

[46] J. Matoušek. Lectures on Discrete Geometry. Graduate Texts in Mathematics 212, Springer, 2002.

[47] M. Michałek and B. Sturmfels. Invitation to Nonlinear Algebra. Graduate Studies in Mathematics 211, American Mathematical Society, 2021.

[48] S. Olofsson, M. P. Deisenroth, and R. Misener. Design of experiments for model discrimination hybridising analytical and data-driven approaches. In Proceedings of the 35th International Conference on Machine Learning, pp. 3908–3917, 2018.

[49] E. Pesce, F. Rapallo, E. Riccomagno, and H. P. Wynn. Generation of all randomizations using circuits. Annals of the Institute of Statistical Mathematics, 75:683–704, 2023.

[50] G. Pistone and H. P. Wynn. Generalised confounding with Gröbner bases. Biometrika, 83(3):653–666, 1996.

[51] G. Pistone, E. Riccomagno, and H. P. Wynn. Algebraic Statistics: Computational Commutative Algebra in Statistics. Chapman & Hall/CRC, 2001.

[52] K. R. Popper. The Logic of Scientific Discovery. Hutchinson, 1959.

[53] S. Ramos Garces, I. De Boi, J. P. Ramos, M. Dierckx, L. Popescu, and S. Derammelaere. Eficient tuning of an isotope separation online system through safe Bayesian optimization with simulation-informed Gaussian process for the constraints. Mathematics, 12(23):3696, 2024.

[54] C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. MIT Press, 2006.

[55] P. J. Rousseeuw and A. M. Leroy. Robust Regression and Outlier Detection. Wiley, 1987.

[56] B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas. Taking the human out of the loop: a review of Bayesian optimization. Proceedings of the IEEE, 104(1):148–175, 2016.

[57] N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger. Gaussian process optimization in the bandit setting: no regret and experimental design. In Proceedings of the 27th International Conference on Machine Learning, pp. 1015–1022, 2010.

[58] S. Sundararajan and S. S. Keerthi. Predictive approaches for choosing hyperparameters in Gaussian processes. Neural Computation, 13(5):1103–1118, 2001.

[59] R. R. Thomas. Lectures in Geometric Combinatorics. Student Mathematical Library 33, American Mathematical Society, 2006.

[60] A. Vehtari, T. Mononen, V. Tolvanen, T. Sivula, and O. Winther. Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models. Journal of Machine Learning Research, 17(103):1–38, 2016.

[61] G. Wahba. Spline Models for Observational Data. CBMS-NSF Regional Conference Series in Applied Mathematics 59, SIAM, 1990.

[62] M. Wendl, E. Englesson, A. Krause, and C. H. Ek. Forgetting to Improve: principled data removal in active learning. ICML 2026 Workshop on Decision-Making from Ofline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning, 2026.

[63] G. M. Ziegler. Lectures on Polytopes. Graduate Texts in Mathematics 152, Springer, 1995.